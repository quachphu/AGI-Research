    Resolution is positional per launch: within a launch (one ``correlation_id``) the records
    are sorted by sync id and the k-th record is assigned the k-th event node of that launch's
    exec graph, by node id. A launch resolves only when its record count matches the graph's
    node count; that count check is the safety net against nested bodies, dropped records, and
    launches split across export windows. This never keys on the CUDA event object, so it is
    correct when a producer recycles one ``cudaEvent_t`` across nodes.
    """
    corr_to_exec: dict[int, int] = {}
    for corr, gnid in corr_exec_pairs:
        if gnid:
            corr_to_exec.setdefault(corr, gnid >> 32)
    by_corr: dict[int, list[tuple[int, int]]] = {}
    for i, (corr, sid) in enumerate(event_rows):
        by_corr.setdefault(corr, []).append((sid, i))
    resolved: list[int | None] = [None] * len(event_rows)
    for corr, rows in by_corr.items():
        exec_id = corr_to_exec.get(corr)
        if exec_id is None:
            continue
        ordered = recorder.graph_event_nodes.get(exec_id)
        if not ordered or len(ordered) != len(rows):
            continue
        rows.sort()  # by sync id, which follows node-id order within the launch
        for pos, (_sid, i) in enumerate(rows):
            resolved[i] = ordered[pos]
    return resolved


def _reset_for_test() -> None:
    """Test-only: unregister the instantiate hook and drop the singleton, so the next
    ``_EventNodeRecorder()`` starts unarmed with an empty map."""
    with _EventNodeRecorder._instance_lock:
        inst = _EventNodeRecorder._instance
        if inst is not None and inst._handle is not None:
            inst._handle.remove()
        _EventNodeRecorder._instance = None
<|endoftext|># mypy: allow-untyped-defs
"""Shared window-finalization machinery for CUPTI monitor observers.

An observer that publishes per *window* (a span ended by a boundary -- a training step or
profiling window) stamps each boundary in CUPTI's native record clock and finalizes the
window only once its records are *naturally* delivered -- no device sync on the measured
timeline (CUPTI delivers buffers in fill order, so a delivered record starting at/after a
boundary means every earlier one is in hand). This mixin owns the boundary queue, poll
thread, cover-detection, and teardown; the subclass supplies ``_collect_delivered(sync)``,
``_window_watermark_ns()`` (max delivered record start, native clock), and
``_finalize_window(window_id, boundary_ns)``. Boundaries use ``now_record_ns()`` from
:class:`CuptiMonitorObserver`, the same timebase as record START/END.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable  # noqa: TC003
from typing import TYPE_CHECKING


logger = logging.getLogger(__name__)


class _PollThread(threading.Thread):
    """Daemon thread that calls ``fn`` every ``interval_s`` until stopped."""

    def __init__(self, fn: Callable[[], None], interval_s: float, name: str) -> None:
        super().__init__(name=name, daemon=True)
        self._fn = fn
        self._interval_s = interval_s
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            try:
                self._fn()
            except Exception:
                logger.exception("CUPTI window poll cycle failed")

    def stop(self) -> None:
        self._stop_event.set()


class WindowFinalizerMixin:
    """Boundary queue + background poller + cover-and-finalize loop. See the module
    docstring. Subclasses call :meth:`_init_observation_window` in ``__init__`` and implement
    ``_collect_delivered`` / ``_window_watermark_ns`` / ``_finalize_window``."""

    def _init_observation_window(
        self,
        *,
        poll_interval_ms: int = 50,
        thread_name: str = "cupti-window-poller",
        auto_start_poller: bool = True,
    ) -> None:
        # auto_start_poller=False: never spin up the poll thread; the consumer finalizes
        # synchronously via _stop_observation_window (e.g. a synchronous export).
        self._auto_start_poller = auto_start_poller
        self._win_lock = threading.Lock()
        # (window_id, boundary_ns) per ended window, oldest first; the poller pops one
        # once its boundary is covered by delivered records.
        self._boundaries: deque[tuple[int, int]] = deque()
        self._next_window_id = 0
        self._poll_interval_s = max(1, poll_interval_ms) / 1000.0
        self._poll_thread: _PollThread | None = None
        self._poll_thread_name = thread_name

    def mark_boundary(self) -> int:
        """Stamp the native record clock as a window boundary and queue it for finalization
        once delivered records cover it. Returns the window id. Lazily starts the poller on
        the first call (unless auto_start_poller is False). Does NOT flush -- the caller
        decides whether to nudge a flush so the poller sees the records."""
        boundary = self._boundary_clock_ns()
        with self._win_lock:
            window_id = self._next_window_id
            self._next_window_id += 1
            self._boundaries.append((window_id, boundary))
            if self._poll_thread is None and self._auto_start_poller:
                self._poll_thread = _PollThread(
                    self._poll_once, self._poll_interval_s, self._poll_thread_name
                )
                self._poll_thread.start()
        return window_id

    def _poll_once(self, drain_all: bool = False, *, sync: bool = False) -> None:
        """One poll cycle: collect delivered records, then finalize every queued boundary
        they now cover. ``drain_all`` (teardown) finalizes every remaining boundary
        regardless of watermark. ``sync`` sync-flushes CUPTI in the collect (only safe on
        the monitor-flusher thread); leave False to rely on naturally-delivered records."""
        self._collect_delivered(sync=sync)
        watermark = self._window_watermark_ns()
        ready: list[tuple[int, int]] = []
        with self._win_lock:
            while self._boundaries and (
                drain_all or self._boundaries[0][1] <= watermark
            ):
                ready.append(self._boundaries.popleft())
        for window_id, boundary in ready:
            self._finalize_window(window_id, boundary)

    def _stop_observation_window(self, *, sync: bool = True) -> None:
        """Stop the poller and finalize whatever remains. Idempotent. ``sync`` (default)
        sync-flushes CUPTI in the final drain -- correct on the monitor-flusher thread (e.g.
        teardown on the training thread). ``sync=False`` finalizes only naturally-delivered
        records, for off-thread use where a flush would race the monitor."""
        thread = self._poll_thread
        if thread is not None:
            thread.stop()
            thread.join(timeout=5.0)
            self._poll_thread = None
        self._poll_once(drain_all=True, sync=sync)

    def _boundary_clock_ns(self) -> int:
        """Clock a boundary is stamped in -- must match :meth:`_window_watermark_ns`.
        Default: CUPTI's native record clock (matches raw START/END). Override to a
        converted clock if the subclass stores records converted (convert the boundary the
        same way; convert_time is monotonic, so the comparison stays order-equivalent)."""
        return self.now_record_ns()

    if TYPE_CHECKING:
        # Provided by the co-class CuptiMonitorObserver in the MRO; declared here only for
        # the type checker (a real def would shadow it at runtime).
        def now_record_ns(self) -> int: ...

    def _collect_delivered(self, *, sync: bool) -> None:
        raise NotImplementedError

    def _window_watermark_ns(self) -> int:
        raise NotImplementedError

    def _finalize_window(self, window_id: int, boundary_ns: int) -> None:
        raise NotImplementedError
<|endoftext|># mypy: allow-untyped-defs
"""Chrome-trace observer for the CUPTI activity multiplexer: accumulates the monitor's
decoded records + trace metadata into the window dict monitor_trace splices into a Kineto
trace. Collection, annotation join, and window management live here."""

from __future__ import annotations

import contextlib
import ctypes
import os
import threading
import time
from typing import Any, TYPE_CHECKING

import numpy as np
from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

import torch
from torch.profiler._cupti.cupti_python import OVERHEAD_KIND_NAMES
from torch.profiler._cupti.monitor_trace import merge_trace_window_into_chrome_trace
from torch.profiler._cupti.observers.base import (
    CuptiMonitorObserver,
    default_graph_annotation_resolver,
    ObserverAnnotationSettings,
)
from torch.profiler._cupti.observers.observation_window import WindowFinalizerMixin
from torch.profiler._cupti.records import (
    Api,
    CudaEvent,
    Environment,
    ExternalCorrelation,
    Field,
    GraphHostNode,
    Kernel,
    Memcpy,
    Memcpy2,
    Memset,
    Overhead,
    Sync,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


def _current_thread_resource_tuple() -> tuple[int, int, int]:
    # (pid, opaque 32-bit thread id, system thread id) for trace lane naming.
    opaque_tid = ctypes.c_int32(threading.get_ident() & 0xFFFFFFFF).value
    return (os.getpid(), opaque_tid, threading.get_native_id())


_DEMANGLE_CACHE: dict[str, str] = {}


def _demangle_symbol(name: str) -> str:
    cached = _DEMANGLE_CACHE.get(name)
    if cached is None:
        cached = _DEMANGLE_CACHE[name] = torch._C._demangle(name)
    return cached


# GPU work plus the CPU-side runtime/driver/overhead records and external correlation for
# the annotation join. Omits fields the chrome trace never consumes (which also have no v2
# user-defined-record id).
PROFILER_FIELDS: dict[ActivityKind, set[Field]] = {
    ActivityKind.CONCURRENT_KERNEL: {
        Kernel.START,
        Kernel.END,
        Kernel.DEVICE_ID,
        Kernel.CONTEXT_ID,
        Kernel.STREAM_ID,
        Kernel.CORRELATION_ID,
        Kernel.GRAPH_NODE_ID,
        Kernel.GRAPH_ID,
        Kernel.NAME,
        # Launch config (kineto reports these; the only source for eager kernels).
        Kernel.GRID_X,
        Kernel.GRID_Y,
        Kernel.GRID_Z,
        Kernel.BLOCK_X,
        Kernel.BLOCK_Y,
        Kernel.BLOCK_Z,
        Kernel.REGISTERS_PER_THREAD,
        Kernel.STATIC_SHARED_MEMORY,
        Kernel.DYNAMIC_SHARED_MEMORY,
        Kernel.LAUNCH_PRIORITY,
        Kernel.QUEUED,
        Kernel.CHANNEL_ID,
        Kernel.CHANNEL_TYPE,
    },
    ActivityKind.MEMCPY: {
        Memcpy.START,
        Memcpy.END,
        Memcpy.DEVICE_ID,
        Memcpy.CONTEXT_ID,
        Memcpy.STREAM_ID,
        Memcpy.CORRELATION_ID,
        Memcpy.GRAPH_NODE_ID,
        Memcpy.GRAPH_ID,
        Memcpy.BYTES,
        Memcpy.COPY_KIND,
        Memcpy.SRC_KIND,
        Memcpy.DST_KIND,
        Memcpy.FLAGS,
        # Copies run on their own channel (CUPTI_CHANNEL_TYPE_ASYNC_MEMCPY), distinct from the
        # compute channel kernels use -- select it like the kernel path does.
        Memcpy.CHANNEL_ID,
        Memcpy.CHANNEL_TYPE,
    },
    # Peer-to-peer / cross-device copies (e.g. tensor.to(other_gpu), pipeline sends). CUPTI
    # records these under MEMCPY2, NOT MEMCPY, so without this they never appear as GPU spans
    # even though they drive NVLink. Folded into the same "gpu_memcpy" frame (see
    # _memcpy2_columns) so they render as Memcpy spans on the issuing device's lane.
    ActivityKind.MEMCPY2: {
        Memcpy2.START,
        Memcpy2.END,
        Memcpy2.DEVICE_ID,
        Memcpy2.CONTEXT_ID,
        Memcpy2.STREAM_ID,
        Memcpy2.CORRELATION_ID,
        Memcpy2.GRAPH_NODE_ID,
        Memcpy2.GRAPH_ID,
        Memcpy2.BYTES,
        Memcpy2.COPY_KIND,
        Memcpy2.SRC_KIND,
        Memcpy2.DST_KIND,
        Memcpy2.FLAGS,
        Memcpy2.CHANNEL_ID,
        Memcpy2.CHANNEL_TYPE,
    },
    ActivityKind.MEMSET: {
        Memset.START,
        Memset.END,
        Memset.DEVICE_ID,
        Memset.CONTEXT_ID,
        Memset.STREAM_ID,
        Memset.CORRELATION_ID,
        Memset.GRAPH_NODE_ID,
        Memset.GRAPH_ID,
        Memset.BYTES,
        Memset.VALUE,
        Memset.MEMORY_KIND,
        Memset.FLAGS,
        Memset.CHANNEL_ID,
        Memset.CHANNEL_TYPE,
    },
    # Graph host nodes run a CPU callback as a CUDA-graph node; CUPTI reports them with a
    # graph_node_id + start/end on the stream that waits for them, so they render like the
    # other graphed GPU ops (attributed via the graph annotation resolver).
    ActivityKind.GRAPH_HOST_NODE: {
        GraphHostNode.START,
        GraphHostNode.END,
        GraphHostNode.DEVICE_ID,
        GraphHostNode.CONTEXT_ID,
        GraphHostNode.STREAM_ID,
        GraphHostNode.CORRELATION_ID,
        GraphHostNode.GRAPH_NODE_ID,
        GraphHostNode.GRAPH_ID,
        GraphHostNode.PROCESS_ID,
        GraphHostNode.THREAD_ID,
    },
    ActivityKind.RUNTIME: {
        Api.CBID,
        Api.START,
        Api.END,
        Api.PROCESS_ID,
        Api.THREAD_ID,
        Api.CORRELATION_ID,
    },
    ActivityKind.DRIVER: {
        Api.CBID,
        Api.START,
        Api.END,
        Api.PROCESS_ID,
        Api.THREAD_ID,
        Api.CORRELATION_ID,
    },
    ActivityKind.EXTERNAL_CORRELATION: {
        ExternalCorrelation.EXTERNAL_KIND,
        ExternalCorrelation.EXTERNAL_ID,
        ExternalCorrelation.CORRELATION_ID,
    },
    ActivityKind.OVERHEAD: {
        Overhead.OVERHEAD_KIND,
        Overhead.START,
        Overhead.END,
        Overhead.CORRELATION_ID,
    },
}


# SYNCHRONIZATION carries kineto's cuda_sync spans; selected only under enable_cuda_sync_events.
# The CUDA_EVENT records these spans join against (the wait_on inputs) live in EVENT_FIELDS.
SYNC_FIELDS: dict[ActivityKind, set[Field]] = {
    ActivityKind.SYNCHRONIZATION: {
        Sync.TYPE,
        Sync.START,
        Sync.END,
        Sync.CORRELATION_ID,
        Sync.CONTEXT_ID,
        Sync.STREAM_ID,
        Sync.CUDA_EVENT_ID,
        Sync.CUDA_EVENT_SYNC_ID,
    },
}


# CUDA_EVENT records are the wait_on join inputs for cuda_sync spans and the device-timestamp
# source for graph event-record node spans. CUPTI emits them for graph event-record nodes via
# the graph-replay trace path without SYNCHRONIZATION co-enabled, so they are selected for both
# enable_cuda_sync_events and enable_event_node_ids. (Eager cudaEventRecord records do require
# SYNCHRONIZATION co-enabled, but those only matter under enable_cuda_sync_events, where it is.)
EVENT_FIELDS: dict[ActivityKind, set[Field]] = {
    ActivityKind.CUDA_EVENT: {
        CudaEvent.CORRELATION_ID,
        CudaEvent.CONTEXT_ID,
        CudaEvent.STREAM_ID,
        CudaEvent.EVENT_ID,
        CudaEvent.DEVICE_ID,
        CudaEvent.CUDA_EVENT_SYNC_ID,
        # Device-side GPU timestamp of the event-record node, in the kernel START/END clock
        # domain -- used to place graph event-record nodes as spans (see _cuda_event_columns).
        CudaEvent.DEVICE_TIMESTAMP,
    },
}


# Periodically-sampled GPU environment (power/clock/thermal/cooling), selected only under
# enable_environment_counters. Rendered as counter tracks; DATA is the 20-byte metric union,
# split by ENVIRONMENT_KIND. Opt-in: the sampling adds overhead, so it is off by default.
ENVIRONMENT_FIELDS: dict[ActivityKind, set[Field]] = {
    ActivityKind.ENVIRONMENT: {
        Environment.DEVICE_ID,
        Environment.TIMESTAMP,
        Environment.ENVIRONMENT_KIND,
        Environment.ENVIRONMENT_KIND_DATA,
    },
}


class ProfilerObserver(WindowFinalizerMixin, CuptiMonitorObserver):
    """Accumulates decoded records and exports them as chrome-trace windows. A window opens
    at trace start (:meth:`open_window`), closes at stop (:meth:`close_window`), and its
    merge + write is deferred until the records are naturally delivered -- no device sync on
    the measured timeline (see :class:`WindowFinalizerMixin`)."""

    def __init__(
        self,
        metadata_resolver: Callable[[int], str | None] | None = None,
        enable_cuda_sync: bool = False,
        enable_environment_counters: bool = False,
        defer_export: bool = True,
        enable_pm_sampling: bool = False,
        pm_metrics: Iterable[str] | None = None,
        enable_graph_dependencies: bool = False,
        enable_event_node_ids: bool = False,
        pftrace_compression_level: int = 1,
    ) -> None:
        self._lock = threading.Lock()
        # gzip level for the native .pftrace encoder (0-9; 1 = fast, the default).
        self._pftrace_compression_level = pftrace_compression_level
        # Decoded activity kept COLUMNAR (frames of named numpy columns, not per-record
        # dicts), so window bucketing is a mask and the build is vectorized. Timed frames
        # (with a start_ns column) bucket into windows by start time.
        self._timed_frames: list[tuple[str, dict[str, Any]]] = []
        # Untimestamped join inputs (EXTERNAL_CORRELATION, CUDA_EVENT): each finalized window
        # consumes whatever is buffered (a harmless superset) and clears them.
        self._ext_frames: list[tuple[str, dict[str, Any]]] = []
        # {external_id: blob} from the metadata store, joined via correlation_id ->
        # external_id -> blob at finalize. Single-consumer (the store is process-global).
        self._ext_metadata: dict[int, str] = {}
        # graph_node_id -> blob for graph-captured collectives (no external-correlation link
        # on replay); resolved like graph-node names. NOT reclaimed per graph, reset per run.
        self._metadata_resolver = metadata_resolver
        # Passive CUDA_EVENT -> graph event-record node bridge (event_id -> graph_node_id),
        # shared process-global recorder; None unless enabled. See _attach_event_node_ids.
        self._event_node_recorder: Any = None
        if enable_event_node_ids:
            from torch.profiler._cupti._event_nodes import _EventNodeRecorder

            self._event_node_recorder = _EventNodeRecorder()
            self._event_node_recorder.arm()
        # pid -> {opaque_tid: system_tid}, for naming GPU/CPU lanes.
        self._thread_resource_map: dict[int, dict[int, int]] = {}
        self._open_start: int | None = None  # open window start (None when none open)
        # window_id -> pending state (start, annotations/thread map, paths, built dict);
        # written + dropped once built AND paths are present.
        self._windows: dict[int, dict[str, Any]] = {}
        selection = {k: set(v) for k, v in PROFILER_FIELDS.items()}
        # CUDA_EVENT records place graph event-record node spans and are the wait_on join inputs
        # for cuda_sync spans, so either feature selects them. SYNCHRONIZATION carries the
        # cuda_sync spans themselves and is selected only under enable_cuda_sync.
        if enable_cuda_sync or enable_event_node_ids:
            selection.update({k: set(v) for k, v in EVENT_FIELDS.items()})
        if enable_cuda_sync:
            selection.update({k: set(v) for k, v in SYNC_FIELDS.items()})
        if enable_environment_counters:
            selection.update({k: set(v) for k, v in ENVIRONMENT_FIELDS.items()})
        # Graph naming on via the default registry resolver (the profiler always wants graph
        # captures named -- it's free when there are none and a no-op for eager-only runs).
        # Eager naming stays off (the default): the full profiler already selects
        # EXTERNAL_CORRELATION + RUNTIME and does the correlation -> external_id -> annotation
        # join in the trace build itself, so the base eager machinery would be redundant.
        super().__init__(
            selection,
            annotations=ObserverAnnotationSettings(
                graph_annotation_resolver=default_graph_annotation_resolver,
                # Node->node dependency arrows are opt-in at the monitor level (extra work at
                # graph instantiate + arrow rendering): the observer records the topology into
                # its own map and draws the arrows only when this is set.
                record_graph_dependencies=enable_graph_dependencies,
            ),
        )
        if self.available:
            # defer_export=False (synchronous export) finalizes on the calling thread in
            # join(), so the background poll thread is never needed -- don't spawn one.
            self._init_observation_window(
                poll_interval_ms=20,
                thread_name="cupti-profiler-export",
                auto_start_poller=defer_export,
            )
            # Purge learned event-node state when a graph is destroyed so the maps don't grow
            # across a long run (mirrors the annotation/dependency destroy hooks).
            if self._event_node_recorder is not None:
                from torch.cuda.graphs import register_graph_destroy_hook

                recorder = self._event_node_recorder
                self._destroy_hook_handles.append(
                    register_graph_destroy_hook(recorder.purge_exec_ids)
                )
        # Opt-in PM sampling (true SM-active % + DRAM-throughput %) is a feature of the CUPTI
        # monitor: it registers us as a consumer (with our metrics) of the current device's shared
        # session, delivering decoded frames to on_pm_samples (they render as GPU counter tracks).
        # Off by default; also a no-op when no metrics are configured (pm_metrics).
        self._pm_metrics = list(pm_metrics or [])
        self._pm_enabled = (
            enable_pm_sampling
            and bool(self._pm_metrics)
            and self.available
            and torch.cuda.is_available()
        )
        # PM samples are timestamped, so they bucket into windows exactly like the activity
        # records (via _timed_frames); no separate buffer. Per-device max start_ns delivered:
        # a monotonic guard so each sample is enqueued at most once.
        self._pm_last_ns: dict[int, int] = {}
        if self._pm_enabled and self._monitor is not None:
            self._monitor.request_pm_sampling(self.on_pm_samples, self._pm_metrics)

    def _boundary_clock_ns(self) -> int:
        # Stamp the boundary in the converted clock the events' start_ns use (convert_time
        # is monotonic, so the comparison stays order-equivalent).
        return self.convert_time(self.now_record_ns())

    def _on_activities(self, columns: dict[Any, dict[int, Any]]) -> None:
        # Worker thread: build a named-column frame per kind (convert/demangle/resolve while
        # the active-id chain is live) and accumulate -- but only while a window is open or
        # pending, else drop the columns no window will consume.
        if not columns:
            return
        with self._lock:
            active = self._open_start is not None or bool(self._windows)
        if not active:
            return
        convert = self.convert_time_array
        timed: list[tuple[str, dict[str, Any]]] = []
        ext: list[tuple[str, dict[str, Any]]] = []
        for kind, cols in columns.items():
            spec = _COLUMN_BUILDERS.get(int(kind))
            if spec is None:
                continue
            kind_str, builder, is_timed = spec
            frame = builder(cols, convert, self._annotation_resolver)
            if frame is None or _named_len(frame) == 0:
                continue
            # Pluggable graphed-op lane assignment: attach (logical_lane, lane_name) columns
            # for work kinds so the trace builder can reassign graphed ops onto a dedicated
            # lane. No-op when no resolver is installed.
            if (
                self._lane_resolver is not None
                and "graph_node_id" in frame
                and "stream_id" in frame
            ):
                frame["logical_lane"], frame["lane_name"] = _resolve_lane_columns(
                    self._lane_resolver, frame
                )
            # Host nodes: attach the recorded callback name/address (see _resolve_host_fn_columns).
            if kind_str == "graph_host_node" and self._graph_host_fns:
                frame["host_fn"], frame["host_fn_addr"] = _resolve_host_fn_columns(
                    self._graph_host_fns, frame
                )
            (timed if is_timed else ext).append((kind_str, frame))
        if not timed and not ext:
            return
        for kind_str, frame in ext:
            if kind_str == "external_correlation":
                self._resolve_user_external_ids(frame)
        meta = (
            self._monitor.take_external_metadata() if self._monitor is not None else {}
        )
        with self._lock:
            self._timed_frames.extend(timed)
            self._ext_frames.extend(ext)
            if meta:
                self._ext_metadata.update(meta)

    def _resolve_user_external_ids(self, frame: dict[str, Any]) -> None:
        """Override each EXTERNAL_CORRELATION row's ``user_external_id`` with the innermost
        ENCLOSING named-region id (via the monitor's live active-id chain), so a kernel
        nested below a named region gets that region's name. Defaults to the raw id."""
        if self._monitor is None:
            return
        names = set(self.annotation_names())
        if not names:
            return
        chain = self._monitor.external_id_chain
        resolved = frame["user_external_id"].tolist()
        cache: dict[int, int] = {}
        for i, eid in enumerate(frame["external_id"].tolist()):
            mapped = cache.get(eid)
            if mapped is None:
                mapped = cache[eid] = next(
                    (c for c in reversed(chain(eid)) if c in names), eid
                )
            resolved[i] = mapped
        frame["user_external_id"] = np.asarray(resolved, dtype=np.int64)

    def push_annotation(self, name: str) -> int | None:
        # Record the calling thread (for lane naming) on top of the base push.
        self._record_calling_thread()
        return super().push_annotation(name)

    def _record_calling_thread(self) -> None:
        pid, opaque_tid, sys_tid = _current_thread_resource_tuple()
        with self._lock:
            self._thread_resource_map.setdefault(pid, {})[opaque_tid] = sys_tid

    # --- async window API (the cupti_monitor profiler backend drives these) ----

    def on_pm_samples(self, frame: dict[str, Any]) -> None:
        # Monitor flush-thread hook: enqueue the frame as a timed frame so each finalized window
        # slices its [start, boundary) samples (a frame spanning a boundary is split like any other
        # timed frame -- see _finalize_window). Keep only samples newer than the last per device:
        # decode drains (each delivered once, in increasing start_ns), so this is a cheap monotonic
        # guard against any duplicate or out-of-order delivery.
        ts = frame.get("start_ns")
        if ts is None or not len(ts):
            return
        dev = frame["device_id"]
        with self._lock:
            keep = np.zeros(len(ts), dtype=bool)
            for d in np.unique(dev):
                keep |= (dev == d) & (ts > self._pm_last_ns.get(int(d), -1))
            if not keep.any():
                return
            kept = _slice_frame(frame, keep)
            kts, kdev = kept["start_ns"], kept["device_id"]
            for d in np.unique(kdev):
                self._pm_last_ns[int(d)] = int(kts[kdev == d].max())
            self._timed_frames.append(("pm_sampling", kept))

    def open_window(self) -> None:
        """Start a trace window; records before this are excluded (no prepare-phase leak)."""
        # Capture the starting thread so its RUNTIME/DRIVER records map to the OS tid
        # (matching its cpu_ops) -- else CUPTI's raw threadId lands them on a phantom lane.
        self._record_calling_thread()
        with self._lock:
            self._open_start = self._boundary_clock_ns()

    def close_window(self) -> int | None:
        """End the open window and queue it for deferred export; snapshots its annotations +
        thread map now. Pair with :meth:`set_export` for the paths. Returns the window id."""
        if not self.available:
            return None
        with self._lock:
            start = self._open_start if self._open_start is not None else 0
            self._open_start = None
            annotations = self.annotation_names(reset=True)
            thread_map = {
                pid: dict(mapping) for pid, mapping in self._thread_resource_map.items()
            }
        window_id = self.mark_boundary()
        with self._lock:
            self._windows[window_id] = {
                "start": start,
                "annotations": annotations,
                "thread_map": thread_map,
                "cpu": None,
                "out": None,
                "built": None,
            }
        return window_id

    def set_export(
        self,
        window_id: int,
        cpu_trace_path: str | os.PathLike[str],
        output_path: str | os.PathLike[str],
    ) -> None:
        """Supply the captured Kineto CPU trace + output path for a closed window. The merged
        file is written once the window's records are covered (now, or by the poller)."""
        with self._lock:
            w = self._windows.get(window_id)
            if w is None:
                return
            w["cpu"] = os.fspath(cpu_trace_path)
            # Monitor traces are always gzipped (chrome JSON or .pftrace alike); the writer
            # keys gzip + format off the suffix.
            out = os.fspath(output_path)
            w["out"] = out if out.endswith(".gz") else out + ".gz"
        self._maybe_write(window_id)

    def join(self, *, force: bool = True, timeout_s: float = 30.0) -> None:
        """Finalize + write every pending window, then unregister. Idempotent. ``force``
        (default) sync-flushes the tail, for use on the training thread. ``force=False`` (an
        off-thread finalize) must NOT flush, so it waits up to ``timeout_s`` for the poller to
        cover the windows, force-draining only if it stalls."""
        # Release PM sampling first: its final tail decode must land in _timed_frames BEFORE the
        # windows finalize, so those samples can be sliced into the closing window.
        if self._pm_enabled and self._monitor is not None:
            self._monitor.release_pm_sampling(self.on_pm_samples)
        if getattr(self, "_boundaries", None) is not None:
            sync = force
            if not force:
                deadline = time.monotonic() + timeout_s
                while time.monotonic() < deadline:
                    with self._win_lock:
                        if not self._boundaries:
                            break
                    time.sleep(self._poll_interval_s)
                with self._win_lock:
                    sync = bool(
                        self._boundaries
                    )  # stalled -> fall back to a forced drain
            self._stop_observation_window(sync=sync)
        # Write on the foreground (here and in set_export), never the poll thread, so it
        # stays inside the caller's temp-dir lifetime.
        for window_id in list(self._windows):
            self._maybe_write(window_id)
        super().close()

    # --- WindowFinalizerMixin hooks -------------------------------------------

    def _collect_delivered(self, *, sync: bool) -> None:
        # Events build in _on_activities as buffers arrive; only flush the tail at teardown.
        if sync and self._monitor is not None:
            self._monitor.flush(sync=True)

    def _window_watermark_ns(self) -> int:
        with self._lock:
            return max(
                (
                    int(frame["start_ns"].max())
                    for _, frame in self._timed_frames
                    if len(frame["start_ns"])
                ),
                default=-1,
            )

    def _finalize_window(self, window_id: int, boundary_ns: int) -> None:
        # Poll thread: records are all in hand, so BUILD the trace-window dict (rows in
        # [start, boundary)) and drop them. Writing is foreground-only (set_export / join).
        with self._lock:
            w = self._windows.get(window_id)
            if w is None:
                return
            start = w["start"]
            in_window: list[tuple[str, dict[str, Any]]] = []
            keep: list[tuple[str, dict[str, Any]]] = []
            for kind_str, frame in self._timed_frames:
                s = frame["start_ns"]
                in_mask = (s >= start) & (s < boundary_ns)
                if in_mask.any():
                    in_window.append((kind_str, _slice_frame(frame, in_mask)))
                # Rows at/after the boundary belong to a later window; rows before
                # ``start`` are prepare-phase noise and dropped (as the dict path did).
                keep_mask = s >= boundary_ns
                if keep_mask.any():
                    keep.append((kind_str, _slice_frame(frame, keep_mask)))
            self._timed_frames = keep
            # Untimestamped join frames ride along; consume the buffered ones now.
            ext, self._ext_frames = self._ext_frames, []
            meta, self._ext_metadata = self._ext_metadata, {}
        columns = _concat_frames_by_kind(in_window + ext)
        # Attach the per-collective blob as a "metadata" column on the GPU-op kinds (these
        # columns are this thread's now, no lock). No-op without comms metadata.
        _attach_metadata(columns, meta, self._metadata_resolver)
        _attach_event_node_ids(
            columns, self._event_node_recorder, self._annotation_resolver
        )
        if self._event_node_recorder is not None:
            _add_graph_event_node_spans(
                columns, start, boundary_ns, self._lane_resolver
            )
        graph_deps = _window_graph_deps(self._dependency_resolver, columns)
        with self._lock:
            w = self._windows.get(window_id)
            if w is None:
                return
            w["built"] = {
                "columns": columns,
                "user_annotations": w["annotations"],
                "thread_resource_map": w["thread_map"],
                "start_ns": start,
                "graph_deps": graph_deps,
                # event-record node graph_node_id -> cudaEvent_t handle, to tag EventRecord
                # spans at render (the CUDA_EVENT record has no event field). Snapshot the
                # shared recorder map so a later graph-destroy purge can't race the merge.
                "graph_event_record_events": dict(self._graph_event_record_events),
            }

    def _maybe_write(self, window_id: int) -> None:
        # Write once the window is both built and has its paths; foreground-only. The
        # del-under-lock makes the writer single.
        with self._lock:
            w = self._windows.get(window_id)
            if w is None or w["built"] is None or w["out"] is None:
                return
            built, cpu, out = w["built"], w["cpu"], w["out"]
            del self._windows[window_id]
        try:
            merge_trace_window_into_chrome_trace(
                cpu,
                out,
                built,
                trace_name=out,
                pftrace_compression_level=self._pftrace_compression_level,
            )
        finally:
            with contextlib.suppress(OSError):  # the CPU trace is a throwaway snapshot
                os.remove(cpu)


# --- raw columns -> named-column frames --------------------------------------
# Turn the monitor's per-kind columns ({field_id: array}) into named numpy columns the merge
# consumes directly. Owns column shape, demangling, clock conversion, and graph-annotation
# resolution -- on the worker thread while the active-id chain is live.


def _named_len(frame: dict[str, Any]) -> int:
    for col in frame.values():
        return len(col)
    return 0


def _slice_frame(frame: dict[str, Any], mask: Any) -> dict[str, Any]:
    return {name: col[mask] for name, col in frame.items()}


def _concat_frames_by_kind(
    frames: list[tuple[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Group same-kind frames and concatenate their columns (same-kind frames share a column
    set); a single frame is passed through without a copy."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for kind_str, frame in frames:
        grouped.setdefault(kind_str, []).append(frame)
    out: dict[str, dict[str, Any]] = {}
    for kind_str, group in grouped.items():
        if len(group) == 1:
            out[kind_str] = group[0]
        else:
            out[kind_str] = {
                name: np.concatenate([g[name] for g in group]) for name in group[0]
            }
    return out


def _attach_metadata(
    columns: dict[str, dict[str, Any]],
    external_metadata: dict[int, str],
    metadata_resolver: Any,
) -> None:
    """Attach the per-collective blob as a ``metadata`` object column on the GPU-op kinds by
    two routes: eager ``correlation_id -> external_id -> blob`` and graph ``graph_node_id``
    via ``metadata_resolver``. Mutates ``columns`` in place; no-op when neither applies."""
    if not external_metadata and metadata_resolver is None:
        return
    # Scope to ids we have a blob for: an op may carry an EXTERNAL_CORRELATION record per
    # active kind, so this picks the collective's and avoids a region id shadowing it.
    corr_to_ext: dict[int, int] = {}
    ext = columns.get("external_correlation")
    if external_metadata and ext is not None:
        corr_to_ext = {
            corr: external_id
            for corr, external_id in zip(
                ext["correlation_id"].tolist(), ext["external_id"].tolist()
            )
            if external_id in external_metadata
        }
    for kind_str in ("kernel", "gpu_memcpy", "gpu_memset"):
        c = columns.get(kind_str)
        if not c or not len(c["correlation_id"]):
            continue
        corr_l = c["correlation_id"].tolist()
        gnid_l = c["graph_node_id"].tolist()
        meta = np.empty(len(corr_l), dtype=object)
        meta[:] = None
        for i in range(len(corr_l)):
            if external_metadata:
                external_id = corr_to_ext.get(corr_l[i])
                if external_id is not None:
                    blob = external_metadata.get(external_id)
                    if blob is not None:
                        meta[i] = blob
                        continue  # eager hit; don't also consult the graph resolver
            if metadata_resolver is not None:
                node = gnid_l[i] or 0
                if node:
                    blob = metadata_resolver(node)
                    if blob is not None:
                        meta[i] = blob
        c["metadata"] = meta


def _attach_event_node_ids(
    columns: dict[str, dict[str, Any]], recorder: Any, resolver: Any = None
) -> None:
    """Learn ``event_id -> graph_node_id`` from this window's launches and attach a
    ``graph_node_id`` + ``annotation`` column to the ``cuda_event`` frame. Passive: reads only
    records that already flow (never touches the events). Mutates ``columns`` in place; no-op
    when the bridge is disabled or there are no CUDA_EVENT records.

    Each graph launch's graphed kernel/memset records share the launch ``correlation_id`` and
    carry the exec graph id in the upper 32 bits of ``graph_node_id``; the same-launch
    CUDA_EVENT records execute in ``cuda_event_sync_id`` order. Matching the k-th event record
    to the graph's k-th ordered event node (recorded at instantiate) yields the mapping."""
    if recorder is None:
        return
    ce = columns.get("cuda_event")
    if ce is None or not len(ce["event_id"]):
        return
    from torch.profiler._cupti._event_nodes import resolve_window

    corr_exec_pairs: list[tuple[int, int]] = []
    for kind_str in ("kernel", "gpu_memcpy", "gpu_memset"):
        c = columns.get(kind_str)
        if c is not None and "graph_node_id" in c:
            corr_exec_pairs.extend(
                zip(c["correlation_id"].tolist(), c["graph_node_id"].tolist())
            )
    event_rows = list(
        zip(ce["correlation_id"].tolist(), ce["cuda_event_sync_id"].tolist())
    )
    resolved = resolve_window(recorder, corr_exec_pairs, event_rows)
    gnid = np.array([n or 0 for n in resolved], dtype=np.int64)
    ce["graph_node_id"] = gnid
    # Same resolver the GPU-op column builders use, so an event node's annotation arrives in
    # the shape the export expects (the default resolver reads torch.cuda._graph_annotations;
    # an installed one may merge a node's annotation list into a single dict). Resolving it
    # here rather than reading the store directly is what lets _annotation_to_args spread the
    # fields into args -- a raw list is re-serialized to an opaque "annotation" string instead.
    ce["annotation"] = _resolve_annotation_column(resolver, gnid)


def _add_graph_event_node_spans(
    columns: dict[str, dict[str, Any]],
    start_ns: int,
    boundary_ns: int,
    lane_resolver: Any = None,
) -> None:
    """Derive a timed ``graph_event_node`` span frame from the resolved CUDA_EVENT records in
    this window, so graph event-record nodes render as (point) spans and become dependency-arrow
    endpoints. Keeps only rows resolved to a graph node (graph_node_id != 0) whose device
    timestamp falls in ``[start_ns, boundary_ns)`` -- the cuda_event join frame is an unbucketed
    superset, so the window bound is applied here. No-op when the bridge attached nothing."""
    ce = columns.get("cuda_event")
    if ce is None or "graph_node_id" not in ce or "start_ns" not in ce:
        return
    gnid = ce["graph_node_id"]
    s = ce["start_ns"]
    mask = (gnid != 0) & (s >= start_ns) & (s < boundary_ns)
    if not mask.any():
        return
    g = gnid[mask]
    frame = {
        "start_ns": s[mask],
        "end_ns": ce["end_ns"][mask],
        "device_id": ce["device_id"][mask],
        "context_id": ce["context_id"][mask],
        "stream_id": ce["stream_id"][mask],
        "correlation_id": ce["correlation_id"][mask],
        "graph_node_id": g,
        "graph_id": g >> 32,
        "annotation": ce["annotation"][mask],
    }
    # This frame is synthesized after _on_activities has already run the lane resolver over the
    # _COLUMN_BUILDERS frames, so it has to attach its own lane columns -- without them the
    # export's reassignment is gated off and an event node strands on its capture stream while
    # the collective's kernel moves to the process-group lane.
    if lane_resolver is not None:
        frame["logical_lane"], frame["lane_name"] = _resolve_lane_columns(
            lane_resolver, frame
        )
    columns["graph_event_node"] = frame


def _nearest_present_preds(resolver: Any, node: int, present: set[int]) -> list[int]:
    """Predecessors of ``node`` collapsed to the nearest ones that render as spans.

    Walks the recorded predecessor edges, stepping THROUGH nodes absent from ``present``
    (event_record / wait_event / empty -- graph nodes CUPTI does not render as GPU-op spans, so
    they can never be arrow endpoints) until it reaches present ancestors. This is what lets a
    dependency arrow span a kernel -> event_record -> wait_event -> kernel chain (e.g. the nodes
    NCCL inserts under NCCL_GRAPH_MIXING_SUPPORT) instead of vanishing at the non-rendered node.
    """
    out: list[int] = []
    seen: set[int] = set()
    stack = list(resolver(node) or ())
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        if p in present:
            out.append(p)
        else:
            stack.extend(resolver(p) or ())
    return out


def _window_graph_deps(
    resolver: Any, columns: dict[str, dict[str, Any]]
) -> dict[int, list[int]]:
    """Resolve the graph node->node dependency edges for the graph_node_ids present in this
    window, so the export can draw node->node arrows. Predecessors are collapsed through
    non-rendered nodes (see :func:`_nearest_present_preds`) so an arrow reaches the nearest node
    that actually renders; for kernel-only graphs this is identical to the raw one-hop edges.
    Empty when there is no resolver, no graphed ops, or no recorded dependencies."""
    if resolver is None:
        return {}
    present: set[int] = set()
    for kind_str in (
        "kernel",
        "gpu_memcpy",
        "gpu_memset",
        "graph_event_node",
        "graph_host_node",
    ):
        c = columns.get(kind_str)
        if c is not None and "graph_node_id" in c:
            present.update(int(g) for g in np.unique(c["graph_node_id"]) if g)
    deps: dict[int, list[int]] = {}
    for g in present:
        preds = _nearest_present_preds(resolver, g, present)
        if preds:
            deps[g] = preds
    return deps


def _demangle_column(names: Any) -> Any:
    out = np.empty(len(names), dtype=object)
    for i, raw in enumerate(names.tolist()):
        out[i] = _demangle_symbol(raw)
    return out


def _resolve_annotation_column(resolver, gnid: Any) -> Any:
    """Per-row graph annotation as an object column. None resolver -> all-None column, no
    calls. The resolver is memoized per graph_node_id by the observer (see
    CuptiMonitorObserver._annotation_resolver), so distinct nodes resolve once for its
    lifetime."""
    n = len(gnid)
    out = np.empty(n, dtype=object)
    if resolver is None:
        out[:] = None
        return out
    for i, g in enumerate(gnid.tolist()):
        out[i] = resolver(g)
    return out


def _resolve_host_fn_columns(host_fns: dict[int, Any], frame: dict[str, Any]) -> Any:
    """Per-row (host_fn name, host_fn addr) for host-node rows, from the recorder's
    graph_node_id -> (name, addr) map (recorded at graph instantiate; host nodes carry no
    name in the CUPTI record). name is None and addr 0 for nodes not in the map."""
    gnid = frame["graph_node_id"]
    n = len(gnid)
    names = np.full(n, None, dtype=object)
    addrs = np.zeros(n, dtype=np.int64)
    for i, g in enumerate(gnid.tolist()):
        entry = host_fns.get(g)
        if entry is not None:
            names[i], addrs[i] = entry
    return names, addrs


def _resolve_lane_columns(lane_resolver, frame: dict[str, Any]) -> Any:
    """Per-row (logical_lane, lane_name) for graphed ops. Graphed rows (graph_node_id != 0)
    get the resolver's (lane, name), or keep their CUDA stream when it returns None; eager rows
    keep their CUDA stream and no name (the monitor names those "stream N"). The resolver is
    keyed and memoized on graph_node_id by the observer (see CuptiMonitorObserver._lane_resolver),
    so distinct nodes resolve once for its lifetime."""
    gnid = frame["graph_node_id"]
    n = len(gnid)
    logical = np.array(
        frame["stream_id"], dtype=np.int64
    )  # default: the op's CUDA stream
    names = np.full(n, None, dtype=object)
    for i, g in enumerate(gnid.tolist()):
        if not g:
            continue
        res = lane_resolver(g)
        if res is not None:
            logical[i], names[i] = res
    return logical, names


def _kernel_columns(cols, convert, resolver):
    gnid = cols[Kernel.GRAPH_NODE_ID.id].astype(np.int64)
    corr = cols[Kernel.CORRELATION_ID.id].astype(np.int64)
    return {
        "start_ns": convert(cols[Kernel.START.id]),
        "end_ns": convert(cols[Kernel.END.id]),
        "device_id": cols[Kernel.DEVICE_ID.id].astype(np.int64),
        "context_id": cols[Kernel.CONTEXT_ID.id].astype(np.int64),
        "stream_id": cols[Kernel.STREAM_ID.id].astype(np.int64),
        "correlation_id": corr,
        "graph_node_id": gnid,
        "graph_id": cols[Kernel.GRAPH_ID.id].astype(np.int64),
        "name": _demangle_column(cols[Kernel.NAME.id]),
        "annotation": _resolve_annotation_column(resolver, gnid),
        "grid_x": cols[Kernel.GRID_X.id].astype(np.int64),
        "grid_y": cols[Kernel.GRID_Y.id].astype(np.int64),
        "grid_z": cols[Kernel.GRID_Z.id].astype(np.int64),
        "block_x": cols[Kernel.BLOCK_X.id].astype(np.int64),
        "block_y": cols[Kernel.BLOCK_Y.id].astype(np.int64),
        "block_z": cols[Kernel.BLOCK_Z.id].astype(np.int64),
        "registers_per_thread": cols[Kernel.REGISTERS_PER_THREAD.id].astype(np.int64),
        "static_shared_memory": cols[Kernel.STATIC_SHARED_MEMORY.id].astype(np.int64),
        "dynamic_shared_memory": cols[Kernel.DYNAMIC_SHARED_MEMORY.id].astype(np.int64),
        "priority": cols[Kernel.LAUNCH_PRIORITY.id].astype(np.int64),
        "queued": convert(cols[Kernel.QUEUED.id]),
        "channel": cols[Kernel.CHANNEL_ID.id].astype(np.int64),
        "channel_type": cols[Kernel.CHANNEL_TYPE.id].astype(np.int64),
    }


def _memcpy_columns(cols, convert, resolver):
    gnid = cols[Memcpy.GRAPH_NODE_ID.id].astype(np.int64)
    corr = cols[Memcpy.CORRELATION_ID.id].astype(np.int64)
    return {
        "start_ns": convert(cols[Memcpy.START.id]),
        "end_ns": convert(cols[Memcpy.END.id]),
        "device_id": cols[Memcpy.DEVICE_ID.id].astype(np.int64),
        "context_id": cols[Memcpy.CONTEXT_ID.id].astype(np.int64),
        "stream_id": cols[Memcpy.STREAM_ID.id].astype(np.int64),
        "correlation_id": corr,
        "graph_node_id": gnid,
        "graph_id": cols[Memcpy.GRAPH_ID.id].astype(np.int64),
        "annotation": _resolve_annotation_column(resolver, gnid),
        "bytes": cols[Memcpy.BYTES.id].astype(np.int64),
        "copy_kind": cols[Memcpy.COPY_KIND.id].astype(np.int64),
        "src_kind": cols[Memcpy.SRC_KIND.id].astype(np.int64),
        "dst_kind": cols[Memcpy.DST_KIND.id].astype(np.int64),
        "flags": cols[Memcpy.FLAGS.id].astype(np.int64),
        "channel": cols[Memcpy.CHANNEL_ID.id].astype(np.int64),
        "channel_type": cols[Memcpy.CHANNEL_TYPE.id].astype(np.int64),
    }


def _memcpy2_columns(cols, convert, resolver):
    # Peer-to-peer (MEMCPY2): same output columns as _memcpy_columns so the frames concatenate
    # under one "gpu_memcpy" kind; reads the MEMCPY2 field ids (src/dst device fields shift
    # correlation/graph ids). src/dst device aren't surfaced (the span on the issuing device's
    # lane is what's wanted), but they're available on Memcpy2 if needed later.
    gnid = cols[Memcpy2.GRAPH_NODE_ID.id].astype(np.int64)
    corr = cols[Memcpy2.CORRELATION_ID.id].astype(np.int64)
    return {
        "start_ns": convert(cols[Memcpy2.START.id]),
        "end_ns": convert(cols[Memcpy2.END.id]),
        "device_id": cols[Memcpy2.DEVICE_ID.id].astype(np.int64),
        "context_id": cols[Memcpy2.CONTEXT_ID.id].astype(np.int64),
        "stream_id": cols[Memcpy2.STREAM_ID.id].astype(np.int64),
        "correlation_id": corr,
        "graph_node_id": gnid,
        "graph_id": cols[Memcpy2.GRAPH_ID.id].astype(np.int64),
        "annotation": _resolve_annotation_column(resolver, gnid),
        "bytes": cols[Memcpy2.BYTES.id].astype(np.int64),
        "copy_kind": cols[Memcpy2.COPY_KIND.id].astype(np.int64),
        "src_kind": cols[Memcpy2.SRC_KIND.id].astype(np.int64),
        "dst_kind": cols[Memcpy2.DST_KIND.id].astype(np.int64),
        "flags": cols[Memcpy2.FLAGS.id].astype(np.int64),
        "channel": cols[Memcpy2.CHANNEL_ID.id].astype(np.int64),
        "channel_type": cols[Memcpy2.CHANNEL_TYPE.id].astype(np.int64),
    }


def _memset_columns(cols, convert, resolver):
    gnid = cols[Memset.GRAPH_NODE_ID.id].astype(np.int64)
    corr = cols[Memset.CORRELATION_ID.id].astype(np.int64)
    return {
        "start_ns": convert(cols[Memset.START.id]),
        "end_ns": convert(cols[Memset.END.id]),
        "device_id": cols[Memset.DEVICE_ID.id].astype(np.int64),
        "context_id": cols[Memset.CONTEXT_ID.id].astype(np.int64),
        "stream_id": cols[Memset.STREAM_ID.id].astype(np.int64),
        "correlation_id": corr,
        "graph_node_id": gnid,
        "graph_id": cols[Memset.GRAPH_ID.id].astype(np.int64),
        "annotation": _resolve_annotation_column(resolver, gnid),
        "bytes": cols[Memset.BYTES.id].astype(np.int64),
        "value": cols[Memset.VALUE.id].astype(np.int64),
        "memory_kind": cols[Memset.MEMORY_KIND.id].astype(np.int64),
        "flags": cols[Memset.FLAGS.id].astype(np.int64),
        "channel": cols[Memset.CHANNEL_ID.id].astype(np.int64),
        "channel_type": cols[Memset.CHANNEL_TYPE.id].astype(np.int64),
    }


def _graph_host_node_columns(cols, convert, resolver):
    gnid = cols[GraphHostNode.GRAPH_NODE_ID.id].astype(np.int64)
    return {
        "start_ns": convert(cols[GraphHostNode.START.id]),
        "end_ns": convert(cols[GraphHostNode.END.id]),
        "device_id": cols[GraphHostNode.DEVICE_ID.id].astype(np.int64),
        "context_id": cols[GraphHostNode.CONTEXT_ID.id].astype(np.int64),
        "stream_id": cols[GraphHostNode.STREAM_ID.id].astype(np.int64),
        "correlation_id": cols[GraphHostNode.CORRELATION_ID.id].astype(np.int64),
        "graph_node_id": gnid,
        "graph_id": cols[GraphHostNode.GRAPH_ID.id].astype(np.int64),
        "annotation": _resolve_annotation_column(resolver, gnid),
        "process_id": cols[GraphHostNode.PROCESS_ID.id].astype(np.int64),
        "thread_id": cols[GraphHostNode.THREAD_ID.id].astype(np.int64),
    }


def _api_columns(cols, convert, resolver):
    del resolver
    return {
        "cbid": cols[Api.CBID.id].astype(np.int64),
        "start_ns": convert(cols[Api.START.id]),
        "end_ns": convert(cols[Api.END.id]),
        "process_id": cols[Api.PROCESS_ID.id].astype(np.int64),
        "thread_id": cols[Api.THREAD_ID.id].astype(np.int64),
        "correlation_id": cols[Api.CORRELATION_ID.id].astype(np.int64),
    }


def _external_correlation_columns(cols, convert, resolver):
    del convert, resolver
    external_id = cols[ExternalCorrelation.EXTERNAL_ID.id].astype(np.int64)
    return {
        "external_kind": cols[ExternalCorrelation.EXTERNAL_KIND.id].astype(np.int64),
        "external_id": external_id,
        "correlation_id": cols[ExternalCorrelation.CORRELATION_ID.id].astype(np.int64),
        # Default to the raw innermost id; _resolve_user_external_ids overrides the
        # rows that resolve to an enclosing named region (chain is live at dispatch).
        "user_external_id": external_id.copy(),
    }


def _overhead_columns(cols, convert, resolver):
    del resolver
    kinds = cols[Overhead.OVERHEAD_KIND.id].astype(np.int64)
    names = np.empty(len(kinds), dtype=object)
    for i, k in enumerate(kinds.tolist()):
        names[i] = OVERHEAD_KIND_NAMES.get(k, f"overhead_{k}")
    return {
        "start_ns": convert(cols[Overhead.START.id]),
        "end_ns": convert(cols[Overhead.END.id]),
        "correlation_id": cols[Overhead.CORRELATION_ID.id].astype(np.int64),
        "name": names,
    }


def _sync_columns(cols, convert, resolver):
    del resolver
    return {
        "sync_type": cols[Sync.TYPE.id].astype(np.int64),
        "start_ns": convert(cols[Sync.START.id]),
        "end_ns": convert(cols[Sync.END.id]),
        "context_id": cols[Sync.CONTEXT_ID.id].astype(np.int64),
        "stream_id": cols[Sync.STREAM_ID.id].astype(np.int64),
        "correlation_id": cols[Sync.CORRELATION_ID.id].astype(np.int64),
        "cuda_event_id": cols[Sync.CUDA_EVENT_ID.id].astype(np.int64),
        "cuda_event_sync_id": cols[Sync.CUDA_EVENT_SYNC_ID.id].astype(np.int64),
    }


def _cuda_event_columns(cols, convert, resolver):
    del resolver
    # start_ns is the event-record node's device timestamp (0 when device timestamps are off);
    # a point span (end == start). The frame stays a join input (is_timed=False); the timed
    # graph_event_node span frame is derived from it per window in _add_graph_event_node_spans.
    start_ns = convert(cols[CudaEvent.DEVICE_TIMESTAMP.id])
    return {
        "cuda_event_sync_id": cols[CudaEvent.CUDA_EVENT_SYNC_ID.id].astype(np.int64),
        "correlation_id": cols[CudaEvent.CORRELATION_ID.id].astype(np.int64),
        "device_id": cols[CudaEvent.DEVICE_ID.id].astype(np.int64),
        "context_id": cols[CudaEvent.CONTEXT_ID.id].astype(np.int64),
        "stream_id": cols[CudaEvent.STREAM_ID.id].astype(np.int64),
        "event_id": cols[CudaEvent.EVENT_ID.id].astype(np.int64),
        "start_ns": start_ns,
        "end_ns": start_ns,
    }


def _environment_columns(cols, convert, resolver):
    del resolver
    # DATA is the union's first 8 bytes (u64): the primary metric pair (e.g. POWER ->
    # power | powerLimit<<32, SPEED -> smClock | memoryClock<<32). Split by environment_kind
    # in the consumer (monitor_trace) into the counter tracks.
    return {
        "start_ns": convert(cols[Environment.TIMESTAMP.id]),
        "device_id": cols[Environment.DEVICE_ID.id].astype(np.int64),
        "environment_kind": cols[Environment.ENVIRONMENT_KIND.id].astype(np.int64),
        "data": cols[Environment.ENVIRONMENT_KIND_DATA.id].astype(np.uint64),
    }


# kind -> (chrome-trace tag, column builder, is_timed). Timed kinds bucket into windows;
# untimed kinds (external_correlation, cuda_event) are join inputs that ride along.
_COLUMN_BUILDERS: dict[int, tuple[str, Any, bool]] = {
    int(ActivityKind.CONCURRENT_KERNEL): ("kernel", _kernel_columns, True),
    int(ActivityKind.MEMCPY): ("gpu_memcpy", _memcpy_columns, True),
    int(ActivityKind.MEMCPY2): ("gpu_memcpy", _memcpy2_columns, True),
    int(ActivityKind.MEMSET): ("gpu_memset", _memset_columns, True),
    int(ActivityKind.GRAPH_HOST_NODE): (
        "graph_host_node",
        _graph_host_node_columns,
        True,
    ),
    int(ActivityKind.RUNTIME): ("cuda_runtime", _api_columns, True),
    int(ActivityKind.DRIVER): ("cuda_driver", _api_columns, True),
    int(ActivityKind.EXTERNAL_CORRELATION): (
        "external_correlation",
        _external_correlation_columns,
        False,
    ),
    int(ActivityKind.OVERHEAD): ("overhead", _overhead_columns, True),
    int(ActivityKind.SYNCHRONIZATION): ("cuda_sync", _sync_columns, True),
    int(ActivityKind.CUDA_EVENT): ("cuda_event", _cuda_event_columns, False),
    int(ActivityKind.ENVIRONMENT): ("environment", _environment_columns, True),
}
<|endoftext|># mypy: allow-untyped-defs
"""Always-on per-graph-node activity-duration observer.

The smallest real consumer of the monitor: registers for one or more activity
kinds with just the compact fields needed for timing (START, END,
GRAPH_NODE_ID, STREAM_ID) and buffers the raw per-kernel spans the monitor's
worker thread delivers. :meth:`NodeTimerObserver.drain` returns them as flat numpy
columns ``(graph_node_id, start_ns, end_ns, stream_id)`` -- consumers aggregate or
bucket as they need (e.g. total duration per node, or kernels bucketed into training
steps by start time, or split per stream), staying vectorized.

By default only CONCURRENT_KERNEL is timed -- the common case. Opt into MEMCPY /
MEMCPY2 (peer-to-peer) / MEMSET via ``kinds`` to time those nodes too. Requesting
MEMCPY2 also enables MEMCPY implicitly (CUPTI emits no MEMCPY2 records unless
MEMCPY is on too). Every kind here times the same 4
fields (START, END, GRAPH_NODE_ID, STREAM_ID), so all records are one size and the
monitor decodes them via its vectorized stride + kind-dispatch path; this observer
just buffers the raw columns (the cost is in the monitor's decode, not here).

Durations are keyed by graph_node_id alone, kind-agnostic: each CUDA-graph node
is a single op, so its kind is unambiguous. Eager (non-graph) activities report
``graph_node_id == 0`` and collapse into one node-0 bucket.

**Named regions.** Pass ``annotations=ObserverAnnotationSettings(...)`` (from the
base); :meth:`drain_annotated` then returns ``{name: [(start_ns, end_ns), ...]}``,
resolving each span graph-first then eager-fallback:

  * **Graph** (``graph_annotation_resolver``) -- ``graph_node_id -> name`` via the resolver
    (a custom one or the default registry). Needs no extra record kinds, so it stays on the
    monitor's **vectorized** decode path. On by default; ``None`` disables it.
  * **Eager** (``support_eager_annotations``) -- ``record_function``-style regions bracketed
    with :meth:`push_annotation`/:meth:`annotate` (inherited from the base) become
    external-correlation ids, joined ``correlation_id -> external_id -> name``. This
    folds in the EXTERNAL_CORRELATION + RUNTIME record kinds (CUPTI only emits the
    former when the latter is enabled), whose differing record sizes drop the decode
    onto the slower per-record walk -- off by default for that cost. External ids don't
    survive CUDA-graph capture, so eager covers only eager activity.

With no ``annotations`` at all, nothing is named (everything in ``""``).
"""

from __future__ import annotations

import threading
from typing import Any, TYPE_CHECKING

from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

from torch.profiler._cupti.observers.base import (
    CuptiMonitorObserver,
    ObserverAnnotationSettings,
)
from torch.profiler._cupti.records import (
    CORRELATION_FIELD,
    ExternalCorrelation,
    Kernel,
    Memcpy,
    Memcpy2,
    Memset,
)


if TYPE_CHECKING:
    from collections.abc import Iterable


# kind -> (start_field_id, end_field_id, graph_node_field_id, stream_field_id) for
# duration timing. stream lets consumers attribute spans to their CUDA stream.
_TIMED_FIELDS: dict[int, tuple[int, int, int, int]] = {
    int(ActivityKind.CONCURRENT_KERNEL): (
        int(Kernel.START),
        int(Kernel.END),
        int(Kernel.GRAPH_NODE_ID),
        int(Kernel.STREAM_ID),
    ),
    int(ActivityKind.MEMCPY): (
        int(Memcpy.START),
        int(Memcpy.END),
        int(Memcpy.GRAPH_NODE_ID),
        int(Memcpy.STREAM_ID),
    ),
    # Peer-to-peer / cross-device copies. Field ids differ from MEMCPY (src/dst
    # device fields shift correlation/graph ids), but the timed span is the same.
    int(ActivityKind.MEMCPY2): (
        int(Memcpy2.START),
        int(Memcpy2.END),
        int(Memcpy2.GRAPH_NODE_ID),
        int(Memcpy2.STREAM_ID),
    ),
    int(ActivityKind.MEMSET): (
        int(Memset.START),
        int(Memset.END),
        int(Memset.GRAPH_NODE_ID),
        int(Memset.STREAM_ID),
    ),
}

_EXTERNAL = int(ActivityKind.EXTERNAL_CORRELATION)


class NodeTimerObserver(CuptiMonitorObserver):
    """Buffers raw per-activity ``(graph_node_id, start_ns, end_ns, stream_id)`` spans the
    monitor delivers; :meth:`drain` returns them as flat numpy columns. Construct with
    ``annotations=ObserverAnnotationSettings(...)`` and :meth:`drain_annotated` returns
    ``{name: [(start_ns, end_ns), ...]}``. See the module docstring."""

    def __init__(
        self,
        kinds: Iterable[int] | None = None,
        *,
        annotations: ObserverAnnotationSettings | None = None,
    ) -> None:
        self._kinds: tuple[int, ...] = (
            tuple(kinds) if kinds is not None else (ActivityKind.CONCURRENT_KERNEL,)
        )
        unknown = [k for k in self._kinds if int(k) not in _TIMED_FIELDS]
        if unknown:
            raise ValueError(
                f"NodeTimerObserver: unsupported activity kind(s) {unknown}; "
                f"timing supports {sorted(_TIMED_FIELDS)}"
            )
        # CUPTI only emits MEMCPY2 (peer-to-peer) records when MEMCPY is also enabled;
        # MEMCPY2 alone silently yields nothing. Enable MEMCPY alongside it implicitly.
        kind_ints = {int(k) for k in self._kinds}
        memcpy, memcpy2 = int(ActivityKind.MEMCPY), int(ActivityKind.MEMCPY2)
        if memcpy2 in kind_ints and memcpy not in kind_ints:
            self._kinds += (ActivityKind.MEMCPY,)
        self._lock = threading.Lock()
        # Raw span column chunks as the worker thread delivers them: each is
        # (graph_node_id, start, end, stream, correlation_id|None -- the id only when
        # eager). Kept raw (not pre-aggregated) so consumers build per-kernel timing on top.
        self._chunks: list[tuple[Any, Any, Any, Any, Any]] = []
        # EXTERNAL_CORRELATION chunks (external_id, correlation_id) for the eager name
        # join; only populated when eager naming is on.
        self._ext_chunks: list[tuple[Any, Any]] = []
        # Base timing selection per kind. The base folds in the eager fields
        # (CORRELATION_ID + EXTERNAL_CORRELATION + RUNTIME) and sets up the resolver
        # from `annotations`. Register last so the buffers are ready before delivery.
        fields = {int(k): set(_TIMED_FIELDS[int(k)]) for k in self._kinds}
        super().__init__(fields, annotations=annotations)

    def _on_activities(self, columns: dict[Any, dict[int, Any]]) -> None:
        # Worker thread: just stash the columns (cheap append); grouping/joining
        # happens after drain, off this hot path.
        spans: list[tuple[Any, Any, Any, Any, Any]] = []
        exts: list[tuple[Any, Any]] = []
        for kind, cols in columns.items():
            k = int(kind)
            if k == _EXTERNAL:
                eid = cols.get(int(ExternalCorrelation.EXTERNAL_ID))
                corr = cols.get(int(ExternalCorrelation.CORRELATION_ID))
                if eid is not None and corr is not None:
                    if self._eager:
                        eid = self._resolve_named_ancestor(eid)
                    exts.append((eid, corr))
                continue
            spec = _TIMED_FIELDS.get(k)
            if spec is None:
                continue
            sf, ef, gf, stf = spec
            start, end, gnode, stream = (
                cols.get(sf),
                cols.get(ef),
                cols.get(gf),
                cols.get(stf),
            )
            if start is None or end is None or gnode is None or stream is None:
                continue
            corr = cols.get(CORRELATION_FIELD[k]) if self._eager else None
            spans.append((gnode, start, end, stream, corr))
        if spans or exts:
            with self._lock:
                self._chunks.extend(spans)
                self._ext_chunks.extend(exts)

    def _resolve_named_ancestor(self, eid_col):
        """Map each (innermost) external id to the innermost ENCLOSING id that names a
        region, via the monitor's active-id chain -- so a kernel nested below a named
        region (e.g. a collective inside it) still resolves to that region. Resolved
        at dispatch, while the chain is live (it is dropped shortly after the pop). The
        innermost id is returned as-is when no named region encloses it (incl. the
        common non-nested case) or no monitor is available."""
        names = set(self.annotation_names())
        if not names or self._monitor is None:
            return eid_col
        import numpy as np

        chain = self._monitor.external_id_chain
        return np.asarray(
            [
                next((c for c in reversed(chain(int(e))) if c in names), int(e))
                for e in eid_col.tolist()
            ],
            dtype=eid_col.dtype,
        )

    def _take(self) -> tuple[list, list]:
        with self._lock:
            chunks, self._chunks = self._chunks, []
            ext_chunks, self._ext_chunks = self._ext_chunks, []
        return chunks, ext_chunks

    def drain(self, flush: bool = False) -> tuple[Any, Any, Any, Any]:
        """Return the raw per-activity spans delivered since the last call as four
        parallel numpy columns ``(graph_node_id, start_ns, end_ns, stream_id)`` (dtypes
        ``<u8, <i8, <i8, <u8``), and reset. Empty -> four length-0 arrays.

        Flat and vectorized on purpose: consumers bucket/aggregate with numpy
        (e.g. ``bincount``/``searchsorted``) without a Python loop over the (~50k)
        kernels. A node's duration is ``end - start``; eager activities share
        ``graph_node_id == 0``.

        Cheap by default: reads only what the worker thread has already delivered,
        so it must NOT force a synchronous flush on every call (records still
        buffered are picked up on the next drain). Pass ``flush=True`` to nudge
        CUPTI to hand over completed buffers first -- a plain ``cuptiActivityFlushAll``,
        not a sync fence (the timer is a rolling consumer; stragglers are
        picked up on the next drain)."""
        import numpy as np

        if flush and self._monitor is not None:
            self._monitor.flush()
        chunks, _ = self._take()
        if not chunks:
            return (
                np.empty(0, dtype="<u8"),
                np.empty(0, dtype="<i8"),
                np.empty(0, dtype="<i8"),
                np.empty(0, dtype="<u8"),
            )
        gnode = np.concatenate([c[0] for c in chunks]).astype("<u8", copy=False)
        start = np.concatenate([c[1] for c in chunks]).astype("<i8", copy=False)
        end = np.concatenate([c[2] for c in chunks]).astype("<i8", copy=False)
        stream = np.concatenate([c[3] for c in chunks]).astype("<u8", copy=False)
        return gnode, start, end, stream

    def drain_annotated(self, flush: bool = False) -> dict[str, list[tuple[int, int]]]:
        """Resolve the spans delivered since the last call to region names and
        return ``{name: [(start_ns, end_ns), ...]}`` (and reset). Each span is named
        graph-first, eager-fallback: the graph resolver on ``graph_node_id`` for
        captured nodes, else the ``correlation_id -> external_id -> name`` join for
        eager regions. Spans that resolve to no name (incl. when no naming is enabled)
        go into the ``""`` bucket. ``flush`` behaves as in :meth:`drain`."""
        if flush and self._monitor is not None:
            self._monitor.flush()
        chunks, ext_chunks = self._take()
        names = self.annotation_names(reset=True)  # external_id -> name (this window)
        if not chunks:
            return {}
        import numpy as np

        gnode = np.concatenate([c[0] for c in chunks]).astype("<u8", copy=False)
        start = np.concatenate([c[1] for c in chunks]).astype("<i8", copy=False)
        end = np.concatenate([c[2] for c in chunks]).astype("<i8", copy=False)

        # Resolve a name per span ("" == unnamed), vectorized: name only the *unique*
        # graph nodes / correlation ids (a small set), then gather back to per-span.
        span_names = np.empty(len(gnode), dtype=object)
        span_names[:] = ""

        resolver = self._annotation_resolver
        if resolver is not None:
            uniq_g, inv_g = np.unique(gnode, return_inverse=True)
            g_names = np.array(
                [(resolver(int(g)) or "") if g else "" for g in uniq_g.tolist()],
                dtype=object,
            )
            span_names = g_names[inv_g]

        # Eager fallback for spans the graph resolver didn't name: correlation_id ->
        # external_id -> name. The external ids were resolved at dispatch through the
        # monitor's active-id chain to the innermost ENCLOSING named region (so a
        # collective nested in a region maps to the region, not its own untracked id);
        # here we just keep those that name a region (in `names`) and map their
        # correlation ids.
        if self._eager and names:
            corr = np.concatenate([c[4] for c in chunks]).astype("<u8", copy=False)
            corr_to_ext: dict[int, int] = {}
            for eid_col, corr_col in ext_chunks:
                for eid, c in zip(eid_col.tolist(), corr_col.tolist()):
                    if int(eid) in names:
                        corr_to_ext[int(c)] = int(eid)
            if corr_to_ext:
                uniq_c, inv_c = np.unique(corr, return_inverse=True)
                c_names = np.array(
                    [
                        names.get(corr_to_ext[ci], "") if ci in corr_to_ext else ""
                        for ci in (int(c) for c in uniq_c.tolist())
                    ],
                    dtype=object,
                )
                eager = c_names[inv_c]
                missing = span_names == ""
                span_names[missing] = eager[missing]

        # Bucket (start, end) by name -- group over the unique names (a small set),
        # selecting each name's spans with a numpy mask. Spans with no graph/eager
        # name fall into the "" bucket (nothing is dropped).
        out: dict[str, list[tuple[int, int]]] = {}
        all_names = span_names.astype(str)
        uniq_names, inv = np.unique(all_names, return_inverse=True)
        for i, nm in enumerate(uniq_names.tolist()):
            mask = inv == i
            out[nm] = list(zip(start[mask].tolist(), end[mask].tolist()))
        return out
<|endoftext|># mypy: allow-untyped-defs
"""Base class for CUPTI activity-monitor observers.

An observer registers the activity kinds it wants with the shared CUPTI monitor and, on
the monitor's worker thread, is handed the demuxed columns (``{ActivityKind: {field_id:
column}}``) sliced to its selection -- what it does with them is the subclass's
``_on_activities`` hook. This base handles registration, availability, teardown, the
clock passthroughs, and the user-annotation push/pop for naming regions.
"""

from __future__ import annotations

import contextlib
import functools
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch.utils.hooks import RemovableHandle


# graph_node_id -> annotation name (or None). The graph naming mechanism shared by
# observers: map a CUDA-graph node id to its registered region name (survives graph
# replay, needs no extra record kinds).
GraphAnnotationResolver = Callable[[int], "Any | None"]

# graph_node_id -> (logical_lane, lane_name), or None to leave the op on its CUDA stream:
# which display lane a graphed op renders on and how that lane is named (the op's CUDA stream
# is preserved as ``original_stream`` when the lane differs). The resolver itself is optional
# (see ``ObserverAnnotationSettings.graph_lane_resolver`` -- unset means no reassignment); when
# set it is called once per distinct graph_node_id and returns that node's logical lane (a
# graph node's annotation, hence its lane, is stable once baked), or None to keep the op on its
# CUDA stream. Keyed on graph_node_id alone, so it is wrapped in functools.cache like the
# annotation resolver; it reads the node's name via the graph annotation registry.
LaneResolver = Callable[[int], "tuple[int, str] | None"]

# graph_node_id -> predecessor graph_node_ids (or None when the node has no recorded
# dependencies): the CUDA-graph node->node edges the export draws as dependency arrows, read
# from the observer's own dependency map (recorded at graph instantiate, keyed on graph_node_id
# like the annotation resolver). See ObserverAnnotationSettings.record_graph_dependencies.
GraphDependencyResolver = Callable[[int], "list[int] | None"]


def default_graph_annotation_resolver(graph_node_id: int) -> Any | None:
    """Default resolver: map a CUDA-graph node id to its registered annotation, or None when
    it has none."""
    if graph_node_id == 0:
        return None
    try:
        from torch.cuda._graph_annotations import get_kernel_annotations

        annotations = get_kernel_annotations()
    except Exception:
        return None
    return annotations.get(graph_node_id)


@dataclass(frozen=True)
class ObserverAnnotationSettings:
    """How an observer attributes activity to named regions. Each source enforces its own
    fields when enabled (folded into the selection) and contributes nothing when disabled.

    - ``graph_annotation_resolver`` -- graph-node naming (``graph_node_id -> name``) from the
      recorded CUDA-graph registry (survives replay); collection-free beyond graph_node_id.
      Pluggable; defaults to ``None`` (disabled). Pass ``default_graph_annotation_resolver``
      (or a custom resolver) to enable.
    - ``support_eager_annotations`` -- eager ``record_function`` naming via the built-in
      external-correlation join (``correlation_id -> external_id -> name``). Folds in
      EXTERNAL_CORRELATION + RUNTIME (-> slower per-record decode), so off by default.
    """

    graph_annotation_resolver: GraphAnnotationResolver | None = None
    support_eager_annotations: bool = False
    # Pluggable graphed-op lane assignment (see LaneResolver). None -> ops render on their
    # CUDA stream lane (no reassignment). Independent of graph_annotation_resolver, though a
    # consumer's implementation typically maps the node's annotation to a lane.
    graph_lane_resolver: LaneResolver | None = None
    # Record each graph's node->node dependency topology at instantiate into the observer's own
    # map (the source for dependency arrows). Off by default (extra work at graph instantiate +
    # arrow rendering); the observer registers an instantiate hook + a resolver over its map
    # only when this is set.
    record_graph_dependencies: bool = False


class CuptiMonitorObserver:
    """Base for observers backed by the shared CUPTI monitor.

    Subclasses set up state, then call ``super().__init__(activities)`` with the kinds they
    want (a set, or a field map ``{kind: field ids | "all"}``) -- registration last, so
    state is ready before the worker thread delivers buffers. They implement
    ``_on_activities(columns)`` (``{ActivityKind: {field_id: column}}`` sliced to their
    selection) and typically a ``drain()``.

    Observers can also bracket regions with ``push_annotation``/``pop_annotation`` (or
    ``annotate``): each push registers a global external-correlation id mapped here to a
    name, attributing activity until the pop via ``correlation_id -> external_id -> name``
    (eager only -- external ids don't survive graph capture; under graphs use
    ``graph_node_id``)."""

    # Both graph resolvers are keyed on graph_node_id: a node's annotation and lane are stable
    # once its graph is baked, so each resolves once for this observer's lifetime (reused across
    # every buffer delivery). Both take the int graph_node_id and are wrapped in functools.cache
    # on assignment. The caches grow with distinct graph_node_ids (each recapture mints new ids);
    # the per-resolver graph-destroy hooks (_register_graph_destroy_hooks) invalidate them
    # when a graph is destroyed, bounding growth in long runs.
    @property
    def _annotation_resolver(self) -> GraphAnnotationResolver | None:
        return self._annotation_resolver_cached

    @_annotation_resolver.setter
    def _annotation_resolver(self, fn: GraphAnnotationResolver | None) -> None:
        self._annotation_resolver_cached = (
            functools.cache(fn) if fn is not None else None
        )

    @property
    def _lane_resolver(self) -> LaneResolver | None:
        return self._lane_resolver_cached

    @_lane_resolver.setter
    def _lane_resolver(self, fn: LaneResolver | None) -> None:
        self._lane_resolver_cached = functools.cache(fn) if fn is not None else None

    @property
    def _dependency_resolver(self) -> GraphDependencyResolver | None:
        return self._dependency_resolver_cached

    @_dependency_resolver.setter
    def _dependency_resolver(self, fn: GraphDependencyResolver | None) -> None:
        self._dependency_resolver_cached = (
            functools.cache(fn) if fn is not None else None
        )

    def __init__(
        self,
        activities: Any,
        *,
        annotations: ObserverAnnotationSettings | None = None,
    ) -> None:
        record_deps = annotations is not None and annotations.record_graph_dependencies
        # Graph node->node dependency map (graph_node_id -> predecessor graph_node_ids), read
        # by our dependency resolver. When recording, share the process-global recorder's map
        # by reference: it is armed early (before graphs are captured) and persists across
        # windows, so it holds topology this per-window observer -- created at prepare_trace,
        # long after warm-up capture, and torn down each window -- would otherwise never see.
        if record_deps:
            from torch.profiler._cupti._graph_deps import _GraphDependencyRecorder

            rec = _GraphDependencyRecorder()
            rec.arm()
            self._graph_dependencies: dict[int, list[int]] = rec.deps
            # event-record node graph_node_id -> cudaEvent_t handle, recorded by the same
            # recorder (its CUDA_EVENT record has no event handle). Shared by reference.
            self._graph_event_record_events: dict[int, int] = rec.event_record_events
            # graph_node_id -> (host_fn_name, host_fn_addr), recorded by the same recorder
            # (host nodes carry no name in the CUPTI record). Shared by reference.
            self._graph_host_fns: dict[int, tuple[str | None, int]] = rec.host_fns
        else:
            self._graph_dependencies = {}
            self._graph_event_record_events = {}
            self._graph_host_fns = {}
        # Region naming (see ObserverAnnotationSettings): an enabled source folds its
        # required fields into the selection (graph: just graph_node_id; eager: extra kinds).
        if annotations is None:
            self._annotation_resolver = None
            self._lane_resolver = None
            self._dependency_resolver = None
            self._eager = False
        else:
            self._annotation_resolver = annotations.graph_annotation_resolver
            self._lane_resolver = annotations.graph_lane_resolver
            self._dependency_resolver = (
                self._make_dependency_resolver() if record_deps else None
            )
            self._eager = annotations.support_eager_annotations
        if self._annotation_resolver is not None:
            activities = self._with_graph_fields(activities)
        if self._eager:
            activities = self._with_eager_fields(activities)
        # frozenset of requested kinds (a field map collapses to keys) for the observer's
        # own "is this kind mine?" checks; the full request goes to the monitor singleton.
        self._activities: frozenset[int] = frozenset(activities)
        self._monitor: Any = None
        self._obs = None
        # external_id -> annotation name for the monitor's global pushes. Guarded by
        # _ann_lock (push on the caller's thread; a drain may read/reset from another).
        self._ann_lock = threading.Lock()
        self._ext_names: dict[int, str] = {}
        # Degrade gracefully (available == False) if the monitor can't be reached or
        # registration fails (CUPTI subscribe rejected, libcupti lacks v2)
        try:
            from torch.profiler._cupti.monitor import CuptiMonitor

            self._monitor = CuptiMonitor()
            self._obs = self._monitor.register(activities, self._on_activities)
        except Exception:
            self._obs = None
        # Register a graph-destroy hook per installed graph-node resolver so a
        # destroyed CUDA graph purges that resolver's registry and invalidates
        # its cache. Registering any hook is also the "monitor active" gate
        # torch.cuda.graphs checks before arming its destroy callback. Handles are
        # removed in close() so nothing leaks this observer.
        self._destroy_hook_handles: list[RemovableHandle] = []
        if self.available:
            self._register_graph_destroy_hooks()

    def _make_dependency_resolver(self) -> GraphDependencyResolver:
        """A resolver over the shared graph dependency map: graph_node_id -> predecessor
        graph_node_ids (None when the node has none). Captures the map, not self, so the
        cached resolver never pins the observer."""
        deps = self._graph_dependencies

        def resolve(graph_node_id: int) -> list[int] | None:
            if graph_node_id == 0:
                return None
            return deps.get(graph_node_id)

        return resolve

    @property
    def available(self) -> bool:
        """True when the monitor was available and this observer registered."""
        return self._obs is not None

    def _register_graph_destroy_hooks(self) -> None:
        """Register a graph-destroy hook per installed graph-node resolver. Each hook
        purges its backing store (the annotation module registry, or this observer's own
        dependency map) and clears its resolver cache (cache_clear is global per resolver
        -- it drops every graph's cached lookups, not just the destroyed graph's --
        acceptable on the infrequent destroy path and what bounds cache growth over a
        long run). Hooks capture only the cache wrapper + purge fn (never self, so they
        cannot pin this observer -- the dependency purge closes over the map dict, not the
        observer); the destroy fan-out also swallows any error they raise
        (finalizer-safe, since a destroy may fire from a GC/finalizer thread). The lane
        resolver is externally backed (nothing to purge), so its hook only clears the
        cache."""
        from torch.cuda._graph_annotations import remove_kernel_annotations
        from torch.cuda.graphs import register_graph_destroy_hook

        handles = self._destroy_hook_handles
        deps = self._graph_dependencies
        event_record_events = self._graph_event_record_events
        host_fns = self._graph_host_fns

        def purge_deps(ids: set[int]) -> None:
            for key in [k for k in deps if k >> 32 in ids]:
                del deps[key]
            for key in [k for k in event_record_events if k >> 32 in ids]:
                del event_record_events[key]
            for key in [k for k in host_fns if k >> 32 in ids]:
                del host_fns[key]

        def add(cache: Any, purge: Any) -> None:
            def hook(ids: set[int]) -> None:
                if purge is not None:
                    purge(ids)
                cache.cache_clear()

            handles.append(register_graph_destroy_hook(hook))

        if self._annotation_resolver_cached is not None:
            add(self._annotation_resolver_cached, remove_kernel_annotations)
        if self._dependency_resolver_cached is not None:
            add(self._dependency_resolver_cached, purge_deps)
        if self._lane_resolver_cached is not None:
            add(self._lane_resolver_cached, None)

    def _on_activities(self, columns: dict[Any, dict[int, Any]]) -> None:
        """Worker-thread hook: ``{ActivityKind: {field_id: column}}`` demuxed by the
        monitor and sliced to this observer's selection. Implemented by subclasses."""
        raise NotImplementedError

    @staticmethod
    def _with_eager_fields(activities: Any) -> dict[int, set[int]]:
        """Augment a field map for the eager join: add each kind's CORRELATION_ID plus the
        EXTERNAL_CORRELATION and RUNTIME records (CUPTI only emits the former when RUNTIME
        is enabled; RUNTIME is just the carrier). Expects a ``{kind: fields}`` map."""
        from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

        from torch.profiler._cupti.records import CORRELATION_FIELD, ExternalCorrelation

        aug: dict[int, set[int]] = {}
        for kind, sel in dict(activities).items():
            k = int(kind)
            fields = {int(f) for f in sel}
            if k in CORRELATION_FIELD:
                fields.add(CORRELATION_FIELD[k])
            aug[k] = fields
        aug[int(ActivityKind.EXTERNAL_CORRELATION)] = {
            int(ExternalCorrelation.EXTERNAL_ID),
            int(ExternalCorrelation.CORRELATION_ID),
        }
        aug[int(ActivityKind.RUNTIME)] = {CORRELATION_FIELD[int(ActivityKind.RUNTIME)]}
        return aug

    @staticmethod
    def _with_graph_fields(activities: Any) -> dict[int, set[int]]:
        """Augment a field map so the graph resolver can name nodes: add each GPU-op kind's
        GRAPH_NODE_ID. Collection-free (it's a normal record field, no extra kinds, stays on
        the vectorized path). Expects a ``{kind: fields}`` map."""
        from torch.profiler._cupti.records import GRAPH_NODE_FIELD

        aug: dict[int, set[int]] = {}
        for kind, sel in dict(activities).items():
            k = int(kind)
            fields = {int(f) for f in sel}
            if k in GRAPH_NODE_FIELD:
                fields.add(GRAPH_NODE_FIELD[k])
            aug[k] = fields
        return aug

    def push_annotation(self, name: str) -> int | None:
        """Push a global external-correlation id (mapped here to ``name``) so activity
        until the pop is attributed via ``correlation_id -> external_id -> name``. Eager
        only. No-op returning None when the monitor is unavailable."""
        if not self.available or self._monitor is None:
            return None
        ext_id = self._monitor.push_external_correlation_id()
        if ext_id is not None:
            with self._ann_lock:
                self._ext_names[ext_id] = name
        return ext_id

    def pop_annotation(self) -> int | None:
        """Pop the most recent external-correlation id (balances push_annotation)."""
        if not self.available or self._monitor is None:
            return None
        return self._monitor.pop_external_correlation_id()

    @contextlib.contextmanager
    def annotate(self, name: str) -> Iterator[int | None]:
        """Context-manager form of push_annotation/pop_annotation."""
        ext_id = self.push_annotation(name)
        try:
            yield ext_id
        finally:
            self.pop_annotation()

    def annotation_names(self, *, reset: bool = False) -> dict[int, str]:
        """Snapshot of the ``external_id -> name`` map pushed so far; pass
        ``reset=True`` to also clear it (e.g. when closing a window)."""
        with self._ann_lock:
            snapshot = dict(self._ext_names)
            if reset:
                self._ext_names = {}
        return snapshot

    def now_ns(self) -> int:
        """Current time on the same unix-epoch clock as record timestamps --
        passthrough to the monitor."""
        return self._monitor.now_unix_ns() if self._monitor is not None else 0

    def now_record_ns(self) -> int:
        """Current value of CUPTI's native record clock -- the unconverted timebase of
        decoded record START/END. Use this to stamp a window boundary compared against
        raw record timestamps (see NodeTimerObserver bucketing). 0 if unavailable."""
        return self._monitor.now_record_ns() if self._monitor is not None else 0

    def convert_time(self, value: int) -> int:
        """Convert a CUPTI-clock timestamp to unix-epoch ns -- passthrough to the
        monitor (identity if clock alignment is unavailable)."""
        return self._monitor.convert_time(value) if self._monitor is not None else value

    def convert_time_array(self, values: Any) -> Any:
        """Vectorized :meth:`convert_time` over a whole column -- passthrough to the
        monitor (identity if clock alignment is unavailable)."""
        if self._monitor is None:
            return values
        return self._monitor.convert_time_array(values)

    def close(self) -> None:
        """Unregister from the monitor. Idempotent."""
        for handle in self._destroy_hook_handles:
            handle.remove()
        self._destroy_hook_handles = []
        if self._obs is not None and self._monitor is not None:
            self._monitor.unregister(self._obs)
            self._obs = None
<|endoftext|># mypy: allow-untyped-defs

from torch._C import _compile_graph_to_code_table, _generate_upgraders_graph


def format_bytecode(table):
    # given a nested tuple, convert it to nested list
    def listify(content):
        if not isinstance(content, tuple):
            return content
        return [listify(i) for i in content]

    formatted_table = {}
    for entry in table:
        identifier = entry[0]
        content = entry[1]
        content = listify(content)
        formatted_table[identifier] = content
    return formatted_table


def generate_upgraders_bytecode() -> list:
    yaml_content = []
    upgraders_graph_map = _generate_upgraders_graph()
    for upgrader_name, upgrader_graph in upgraders_graph_map.items():
        bytecode_table = _compile_graph_to_code_table(upgrader_name, upgrader_graph)
        entry = {upgrader_name: format_bytecode(bytecode_table)}
        yaml_content.append(entry)
    return yaml_content


if __name__ == "__main__":
    raise RuntimeError("This file is not meant to be run directly")
<|endoftext|># mypy: allow-untyped-defs
import math
from collections.abc import Callable
from typing import Any, Optional, Union


number = Union[int, float]
# flake8: noqa

###
# There are generated files that depend on this file
# To re-generate, please run from the root of the repo:
# python torchgen/shape_functions/gen_jit_shape_functions.py

# How to test:
# After regenerating files, compile PyTorch.
# Then run: ./build/bin/test_jit --gtest_filter=TestShapeGraphLinting.Basic
# If you have enabled opinfo testing for the op, also run:
# python test/test_ops_jit.py TestJitCPU.test_variant_consistency_jit_[FAILING_OP]_cpu_float32
# to reproduce errors from opinfo tests.

# Example PR: https://github.com/pytorch/pytorch/pull/80860/files
####

import torch


def broadcast(a: list[int], b: list[int]):
    dimsA = len(a)
    dimsB = len(b)
    ndim = max(dimsA, dimsB)
    expandedSizes: list[int] = []

    for i in range(ndim):
        offset = ndim - 1 - i
        dimA = dimsA - 1 - offset
        dimB = dimsB - 1 - offset
        sizeA = a[dimA] if (dimA >= 0) else 1
        sizeB = b[dimB] if (dimB >= 0) else 1

        if sizeA != sizeB and sizeA != 1 and sizeB != 1:
            # TODO: only assertion error is bound in C++ compilation right now
            raise AssertionError(
                f"The size of tensor a {sizeA} must match the size of tensor b ({sizeB}) at non-singleton dimension {i}"
            )

        expandedSizes.append(sizeB if sizeA == 1 else sizeA)

    return expandedSizes


def broadcast_three(a: list[int], b: list[int], c: list[int]):
    return broadcast(broadcast(a, b), c)


def broadcast_one_three(a: list[int], b: Any, c: list[int]):
    return broadcast(a, c)


def adaptive_avg_pool2d(self: list[int], out: list[int]):
    if len(out) != 2:
        raise AssertionError(f"Expected out to have length 2, but got {len(out)}")
    if not (len(self) == 3 or len(self) == 4):
        raise AssertionError(
            f"Expected self to have length 3 or 4, but got {len(self)}"
        )
    for i in range(1, len(self)):
        if self[i] == 0:
            raise AssertionError(f"Expected self[{i}] to be non-zero, but got 0")

    shape: list[int] = []
    for i in range(0, len(self) - 2):
        shape.append(self[i])
    for elem in out:
        shape.append(elem)
    return shape


def _copy(self: list[int]):
    out: list[int] = []
    for elem in self:
        out.append(elem)
    return out


def unary(self: list[int]):
    return _copy(self)


def broadcast_inplace(a: list[int], b: list[int]):
    dimsA = len(a)
    dimsB = len(b)
    if dimsB > dimsA:
        raise AssertionError(
            f"The dims of tensor b ({dimsB}) must be less than or equal to the dims of tensor a ({dimsA}) "
        )
    for dimA in range(dimsA):
        dimB = dimsB - dimsA + dimA
        sizeA = a[dimA]
        sizeB = b[dimB] if (dimB >= 0) else 1
        if sizeA != sizeB and sizeB != 1:
            # TODO: only assertion error is bound in C++ compilation right now
            raise AssertionError(
                "The size of tensor a {} must match the size of tensor b ("
                "{}) at non-singleton dimension {}".format(sizeA, sizeB, dimA)
            )
    return _copy(a)


def expand(self: list[int], sizes: list[int]):
    if len(sizes) < len(self):
        raise AssertionError(
            f"Expected len(sizes) ({len(sizes)}) >= len(self) ({len(self)})"
        )
    ndim = len(sizes)
    tensor_dim = len(self)
    if ndim == 0:
        return _copy(sizes)
    out: list[int] = []
    for i in range(ndim):
        offset = ndim - 1 - i
        dim = tensor_dim - 1 - offset
        size = self[dim] if dim >= 0 else 1
        targetSize = sizes[i]
        if targetSize == -1:
            if dim < 0:
                raise AssertionError(f"Expected dim ({dim}) >= 0 when targetSize is -1")
            targetSize = size
        if size != targetSize:
            if size != 1:
                raise AssertionError(
                    f"Expected size ({size}) == 1 when size != targetSize ({targetSize})"
                )
            size = targetSize
        out.append(size)
    return out


def expand_one_unused(self: list[int], sizes: list[int], inp0: Any):
    return expand(self, sizes)


def infer_size_impl(shape: list[int], numel: int) -> list[int]:
    newsize = 1
    infer_dim: Optional[int] = None
    for dim in range(len(shape)):
        if shape[dim] == -1:
            if infer_dim is not None:
                raise AssertionError("only one dimension can be inferred")
            infer_dim = dim
        elif shape[dim] >= 0:
            newsize *= shape[dim]
        else:
            raise AssertionError("invalid shape dimensions")
    if not (
        numel == newsize
        or (infer_dim is not None and newsize > 0 and numel % newsize == 0)
    ):
        raise AssertionError("invalid shape")
    out = _copy(shape)
    if infer_dim is not None:
        out[infer_dim] = numel // newsize
    return out


def numel(sizes: list[int]):
    numel = 1
    for elem in sizes:
        numel *= elem
    return numel


def view(self: list[int], sizes: list[int]):
    return infer_size_impl(sizes, numel(self))


def view_one_unused(self: list[int], sizes: list[int], *, implicit: bool = False):
    return view(self, sizes)


def sum_mean_dim(
    self: list[int], opt_dims: Optional[list[int]], keep_dim: bool, dt: Any
):
    out: list[int] = []
    if opt_dims is None or len(opt_dims) == 0:
        dims: list[int] = list(range(len(self)))
    else:
        dims = opt_dims

    for idx in range(len(self)):
        is_mean_dim: bool = False
        for reduce_dim in dims:
            if idx == maybe_wrap_dim(reduce_dim, len(self)):
                is_mean_dim = True
        if is_mean_dim:
            if keep_dim:
                out.append(1)
        else:
            out.append(self[idx])
    return out


def max_dim(self: list[int], dim: int, keep_dim: bool):
    out = sum_mean_dim(self, [dim], keep_dim, None)
    return out, out


# note: python already rounds down towards negative infinity on integer division, special arithmetic not needed
def div_rtn(x: int, y: int):
    return x // y


def pooling_output_shape_pad_lr(
    inputSize: int,
    kernelSize: int,
    pad_l: int,
    pad_r: int,
    stride: int,
    dilation: int,
    ceil_mode: bool,
):
    outputSize = (
        div_rtn(
            inputSize
            + pad_l
            + pad_r
            - dilation * (kernelSize - 1)
            - 1
            + (stride - 1 if ceil_mode else 0),
            stride,
        )
        + 1
    )
    if ceil_mode:
        if (outputSize - 1) * stride >= inputSize + pad_l:
            outputSize = outputSize - 1
    return outputSize


def pooling_output_shape(
    inputSize: int,
    kernelSize: int,
    pad_l: int,
    stride: int,
    dilation: int,
    ceil_mode: bool,
):
    if stride == 0:
        raise AssertionError("stride should not be zero")
    return pooling_output_shape_pad_lr(
        inputSize, kernelSize, pad_l, pad_l, stride, dilation, ceil_mode
    )


def pool2d_shape_check(
    input: list[int],
    kH: int,
    kW: int,
    dH: int,
    dW: int,
    padH: int,
    padW: int,
    dilationH: int,
    dilationW: int,
    nInputPlane: int,
    inputHeight: int,
    inputWidth: int,
    outputHeight: int,
    outputWidth: int,
):
    ndim = len(input)

    if not (kW > 0 and kH > 0):
        raise AssertionError(f"Expected kW ({kW}) > 0 and kH ({kH}) > 0")
    if not (dW > 0 and dH > 0):
        raise AssertionError(f"Expected dW ({dW}) > 0 and dH ({dH}) > 0")
    if not (dilationH > 0 and dilationW > 0):
        raise AssertionError(
            f"Expected dilationH ({dilationH}) > 0 and dilationW ({dilationW}) > 0"
        )

    valid_dims = input[1] != 0 and input[2] != 0
    if not (
        ndim == 3
        and input[0] != 0
        and valid_dims
        or (ndim == 4 and valid_dims and input[3] != 0)
    ):
        raise AssertionError(f"Invalid input dimensions: ndim={ndim}, input={input}")

    if not (kW // 2 >= padW and kH // 2 >= padH):
        raise AssertionError(
            f"Expected kW//2 ({kW // 2}) >= padW ({padW}) and "
            f"kH//2 ({kH // 2}) >= padH ({padH})"
        )
    if not (outputWidth >= 1 and outputHeight >= 1):
        raise AssertionError(
            f"Expected outputWidth ({outputWidth}) >= 1 and "
            f"outputHeight ({outputHeight}) >= 1"
        )


def max_pool2d(
    input: list[int],
    kernel_size: list[int],
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    ceil_mode: bool,
):
    if not (len(kernel_size) == 1 or len(kernel_size) == 2):
        raise AssertionError(
            "max_pool2d: kernel_size must either be a single int, or a tuple of two ints"
        )
    kH = kernel_size[0]
    kW = kH if len(kernel_size) == 1 else kernel_size[1]

    if not (len(stride) == 0 or len(stride) == 1 or len(stride) == 2):
        raise AssertionError(
            "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints"
        )
    dH = kH if len(stride) == 0 else stride[0]
    if len(stride) == 0:
        dW = kW
    elif len(stride) == 1:
        dW = dH
    else:
        dW = stride[1]

    if not (len(padding) == 1 or len(padding) == 2):
        raise AssertionError(
            "max_pool2d: padding must either be a single int, or a tuple of two ints"
        )
    padH = padding[0]
    padW = padH if len(padding) == 1 else padding[1]

    if not (len(dilation) == 1 or len(dilation) == 2):
        raise AssertionError(
            "max_pool2d: dilation must be either a single int, or a tuple of two ints"
        )
    dilationH = dilation[0]
    dilationW = dilationH if len(dilation) == 1 else dilation[1]

    if not (len(input) == 3 or len(input) == 4):
        raise AssertionError(f"Expected input length 3 or 4, but got {len(input)}")

    nbatch = input[-4] if len(input) == 4 else 1
    nInputPlane = input[-3]
    inputHeight = input[-2]
    inputWidth = input[-1]

    outputHeight = pooling_output_shape(inputHeight, kH, padH, dH, dilationH, ceil_mode)
    outputWidth = pooling_output_shape(inputWidth, kW, padW, dW, dilationW, ceil_mode)

    pool2d_shape_check(
        input,
        kH,
        kW,
        dH,
        dW,
        padH,
        padW,
        dilationH,
        dilationW,
        nInputPlane,
        inputHeight,
        inputWidth,
        outputHeight,
        outputWidth,
    )

    if len(input) == 3:
        return [nInputPlane, outputHeight, outputWidth]
    else:
        return [nbatch, nInputPlane, outputHeight, outputWidth]


def max_pool2d_with_indices(
    input: list[int],
    kernel_size: list[int],
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    ceil_mode: bool,
):
    out = max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
    return (out, out)


def upsample_nearest2d(
    input: list[int],
    output_size: Optional[list[int]],
    scale_factors: Optional[list[float]],
):
    out: list[int] = []
    out.append(input[0])
    out.append(input[1])

    if scale_factors is None and output_size is None:
        raise AssertionError("Either output_size or scale_factors must be presented")

    if output_size is not None:
        if scale_factors is not None:
            raise AssertionError(
                "Must specify exactly one of output_size and scale_factors"
            )
        if len(output_size) != 2:
            raise AssertionError(
                f"Expected output_size to have length 2, but got {len(output_size)}"
            )
        out.append(output_size[0])
        out.append(output_size[1])

    if scale_factors is not None:
        if output_size is not None:
            raise AssertionError(
                "Must specify exactly one of output_size and scale_factors"
            )
        if len(scale_factors) != 2:
            raise AssertionError(
                f"Expected scale_factors to have length 2, but got {len(scale_factors)}"
            )
        out.append(int(input[2] * scale_factors[0]))
        out.append(int(input[3] * scale_factors[1]))

    return out


def mm(self: list[int], mat2: list[int]):
    if len(self) != 2:
        raise AssertionError(f"self must be a matrix (got {len(self)} dimensions)")
    if len(mat2) != 2:
        raise AssertionError(f"mat2 must be a matrix (got {len(mat2)} dimensions)")

    if self[1] != mat2[0]:
        raise AssertionError(
            f"Matrix dimensions don't match for mm: self[1]={self[1]}, mat2[0]={mat2[0]}"
        )
    return [self[0], mat2[1]]


def dot(self: list[int], tensor: list[int]):
    if not (len(self) == 1 and len(tensor) == 1):
        raise AssertionError(
            f"Expected 1D tensors for dot, got len(self)={len(self)}, "
            f"len(tensor)={len(tensor)}"
        )
    if self[0] != tensor[0]:
        raise AssertionError(
            f"Dot product dimension mismatch: self[0]={self[0]}, tensor[0]={tensor[0]}"
        )
    out: list[int] = []
    return out


def mv(self: list[int], vec: list[int]):
    if not (len(self) == 2 and len(vec) == 1):
        raise AssertionError(
            f"Expected 2D matrix and 1D vector, got len(self)={len(self)}, "
            f"len(vec)={len(vec)}"
        )
    if self[1] != vec[0]:
        raise AssertionError(
            f"Matrix-vector dimension mismatch: self[1]={self[1]}, vec[0]={vec[0]}"
        )
    # TODO: return self
    return [self[0]]


def unsqueeze(li: list[int], dim: int):
    dim = maybe_wrap_dim(dim, len(li) + 1)
    out = _copy(li)
    out.insert(dim, 1)
    return out


def squeeze_nodim(li: list[int]):
    out: list[int] = []
    for i in range(len(li)):
        if li[i] != 1:
            out.append(li[i])
    return out


def squeeze(li: list[int], dim: int):
    out: list[int] = []
    wrapped_dim = maybe_wrap_dim(dim, len(li))
    for i in range(len(li)):
        if i == wrapped_dim:
            if li[i] != 1:
                out.append(li[i])
        else:
            out.append(li[i])
    return out


def squeeze_dims(li: list[int], dims: list[int]):
    if len(dims) == 0:
        return li
    wrapped_dims = _copy(dims)
    for i in range(len(dims)):
        wrapped_dims[i] = maybe_wrap_dim(wrapped_dims[i], len(li))
    result: list[int] = []
    for i in range(len(li)):
        if li[i] == 1:
            if i not in wrapped_dims:
                result.append(li[i])
        else:
            result.append(li[i])
    return result


def index_select(self: list[int], dim: int, index: list[int]):
    dim = maybe_wrap_dim(dim, len(self))
    numel = multiply_integers(index)
    if len(index) > 1:
        raise AssertionError(f"Expected len(index) <= 1, but got {len(index)}")
    if not (dim == 0 or dim < len(self)):
        raise AssertionError(
            f"Expected dim ({dim}) == 0 or dim < len(self) ({len(self)})"
        )
    result_size: list[int] = []
    for i in range(len(self)):
        if dim == i:
            result_size.append(numel)
        else:
            result_size.append(self[i])
    return result_size


def embedding(
    weight: list[int],
    indices: list[int],
    padding_idx: int = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
):
    if len(weight) != 2:
        raise AssertionError(f"Expected weight to be 2D, but got {len(weight)}D")
    if len(indices) == 1:
        return index_select(weight, 0, indices)
    size = _copy(indices)
    size.append(weight[1])
    return size


def max_int():
    return 9223372036854775807


def slice(
    self: list[int], dim: int, start: Optional[int], end: Optional[int], step: int
):
    ndim = len(self)
    if ndim == 0:
        raise AssertionError("Cannot slice a 0-dimensional tensor")
    dim = maybe_wrap_dim(dim, ndim)
    start_val = start if start is not None else 0
    end_val = end if end is not None else max_int()
    if step <= 0:
        raise AssertionError(f"Expected step > 0, but got {step}")
    if start_val == max_int():
        start_val = 0
    if start_val < 0:
        start_val += self[dim]
    if end_val < 0:
        end_val += self[dim]
    if start_val < 0:
        start_val = 0
    elif start_val > self[dim]:
        start_val = self[dim]
    if end_val < start_val:
        end_val = start_val
    elif end_val >= self[dim]:
        end_val = self[dim]
    slice_len = end_val - start_val
    out = _copy(self)
    out[dim] = (slice_len + step - 1) // step
    return out


def check_cat_no_zero_dim(tensors: list[list[int]]):
    for tensor in tensors:
        if len(tensor) <= 0:
            raise AssertionError("Cannot concatenate tensor with 0 dimensions")


def legacy_cat_wrap_dim(dim: int, tensor_sizes: list[list[int]]):
    out_dim: Optional[int] = None
    for size in tensor_sizes:
        if not (len(size) == 1 and size[0] == 0):
            if out_dim is None:
                out_dim = maybe_wrap_dim(dim, len(size))
    if out_dim is None:
        out_dim = dim
    return out_dim


def should_skip(tensor: list[int]):
    return numel(tensor) == 0 and len(tensor) == 1


def check_cat_shape_except_dim(
    first: list[int], second: list[int], dimension: int, index: int
):
    first_dims = len(first)
    second_dims = len(second)
    if first_dims != second_dims:
        raise AssertionError(
            f"Tensors must have same number of dimensions, got {first_dims} and "
            f"{second_dims}"
        )
    for dim in range(0, first_dims):
        if dim != dimension:
            if first[dim] != second[dim]:
                raise AssertionError(
                    f"Sizes of tensors must match except in dimension {dimension}, "
                    f"got {first[dim]} and {second[dim]} at dimension {dim}"
                )


def cat(tensors: list[list[int]], dim: int):
    check_cat_no_zero_dim(tensors)
    dim = legacy_cat_wrap_dim(dim, tensors)
    if len(tensors) <= 0:
        raise AssertionError("Cannot concatenate empty list of tensors")
    not_skipped_tensor: Optional[list[int]] = None
    for tensor in tensors:
        if not should_skip(tensor):
            not_skipped_tensor = tensor
    if not_skipped_tensor is None:
        return [0]

    cat_dim_size = 0

    for i in range(len(tensors)):
        tensor = tensors[i]
        if not should_skip(tensor):
            check_cat_shape_except_dim(not_skipped_tensor, tensor, dim, i)
            cat_dim_size = cat_dim_size + tensor[dim]

    result_size = _copy(not_skipped_tensor)
    result_size[dim] = cat_dim_size
    return result_size


def stack(tensors: list[list[int]], dim: int):
    unsqueezed_tensors: list[list[int]] = []
    for tensor in tensors:
        unsqueezed = unsqueeze(tensor, dim)
        unsqueezed_tensors.append(unsqueezed)
    return cat(unsqueezed_tensors, dim)


def select(self: list[int], dim: int, index: int):
    ndim = len(self)
    if ndim == 0:
        raise AssertionError("Cannot select from a 0-dimensional tensor")
    dim = maybe_wrap_dim(dim, ndim)
    size = self[dim]
    if index < -size or index >= size:
        raise AssertionError(
            f"Index {index} is out of bounds for dimension {dim} with size {size}"
        )
    if index < 0:
        index += size
    out: list[int] = []
    for i in range(ndim):
        if i != dim:
            out.append(self[i])
    return out


def matmul(tensor1: list[int], tensor2: list[int]):
    dim_tensor1 = len(tensor1)
    dim_tensor2 = len(tensor2)
    if dim_tensor1 == 1 and dim_tensor2 == 1:
        return dot(tensor1, tensor2)
    elif dim_tensor1 == 2 and dim_tensor2 == 1:
        return mv(tensor1, tensor2)
    elif dim_tensor1 == 1 and dim_tensor2 == 2:
        return squeeze(mm(unsqueeze(tensor1, 0), tensor2), 0)
    elif dim_tensor1 == 2 and dim_tensor2 == 2:
        return mm(tensor1, tensor2)
    elif dim_tensor1 >= 1 and dim_tensor2 >= 1:
        # We are multiplying b1 x n x m1 by x2 x m2 x p (where b1 can be a list);
        # we track m1 vs m2 separately even though they must match for nicer error messages
        n = tensor1[-2] if dim_tensor1 > 1 else 1
        batch_tensor1: list[int] = []
        # TODO: handling of slice
        for i in range(dim_tensor1 - 2):
            batch_tensor1.append(tensor1[i])
        p = tensor2[-1]
        batch_tensor2: list[int] = []
        # TODO: handling of slice
        for i in range(dim_tensor2 - 2):
            batch_tensor2.append(tensor2[i])

        # expand the batch portion (i.e. cut off matrix dimensions and expand rest)
        expand_batch_portion = broadcast(batch_tensor1, batch_tensor2)

        # todo: copy ?
        output_shape = expand_batch_portion
        if dim_tensor1 > 1:
            output_shape.append(n)

        if dim_tensor2 > 1:
            output_shape.append(p)

        return output_shape
    else:
        raise AssertionError("both arguments to matmul need to be at least 1D")


def t(self: list[int]):
    if len(self) > 2:
        raise AssertionError(
            f"Expected tensor to have <= 2 dimensions, but got {len(self)}"
        )
    self_len = len(self)
    if self_len == 0:
        out: list[int] = []
        return out
    elif self_len == 1:
        return [self[0]]
    else:
        return [self[1], self[0]]


def transpose(self: list[int], dim0: int, dim1: int):
    ndims = len(self)
    dim0 = maybe_wrap_dim(dim0, ndims)
    dim1 = maybe_wrap_dim(dim1, ndims)
    if dim0 == dim1:
        return _copy(self)
    out: list[int] = []
    for i in range(ndims):
        if i == dim0:
            out.append(self[dim1])
        elif i == dim1:
            out.append(self[dim0])
        else:
            out.append(self[i])
    return out


def linear(input: list[int], weight: list[int], bias: Optional[list[int]]):
    out = matmul(input, t(weight))
    if bias is not None:
        if broadcast(bias, out) != out:
            raise AssertionError(
                f"Bias shape {bias} is not broadcastable to output shape {out}"
            )
    return out


def addmm(self: list[int], mat1: list[int], mat2: list[int], beta: Any, alpha: Any):
    return broadcast(self, mm(mat1, mat2))


def check_non_negative(array: list[int]) -> bool:
    # TODO: look into rewriting with early return and getting loop unrolling to fire
    non_negative = False
    for val in array:
        if val < 0:
            non_negative = True
    return non_negative


def check_shape_forward(
    input: list[int],
    weight_sizes: list[int],
    bias: Optional[list[int]],
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    groups: int,
):
    k = len(input)
    weight_dim = len(weight_sizes)

    # TODO: assertions could be expanded with the error messages
    if check_non_negative(padding):
        raise AssertionError(f"Padding must be non-negative, got {padding}")
    if check_non_negative(stride):
        raise AssertionError(f"Stride must be non-negative, got {stride}")

    if weight_dim != k:
        raise AssertionError(f"Expected weight_dim ({weight_dim}) == k ({k})")
    if weight_sizes[0] < groups:
        raise AssertionError(
            f"Expected weight_sizes[0] ({weight_sizes[0]}) >= groups ({groups})"
        )
    if (weight_sizes[0] % groups) != 0:
        raise AssertionError(
            f"Expected weight_sizes[0] ({weight_sizes[0]}) to be divisible by "
            f"groups ({groups})"
        )
    # only handling not transposed
    if input[1] != weight_sizes[1] * groups:
        raise AssertionError(
            f"Expected input[1] ({input[1]}) == weight_sizes[1] * groups "
            f"({weight_sizes[1] * groups})"
        )
    if bias is not None and not (len(bias) == 1 and bias[0] == weight_sizes[0]):
        raise AssertionError(
            f"Expected bias to be None or have shape [1] with value "
            f"weight_sizes[0]={weight_sizes[0]}, got {bias}"
        )

    for i in range(2, k):
        if (input[i] + 2 * padding[i - 2]) < (
            dilation[i - 2] * (weight_sizes[i] - 1) + 1
        ):
            raise AssertionError(
                f"Calculated padded input size ({input[i] + 2 * padding[i - 2]}) "
                f"is smaller than effective kernel size "
                f"({dilation[i - 2] * (weight_sizes[i] - 1) + 1}) at dimension {i}"
            )

    # this is not handling transposed convolution yet


def conv_output_size(
    input_size: list[int],
    weight_size: list[int],
    bias: Optional[list[int]],
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    groups: int,
):
    check_shape_forward(
        input_size, weight_size, bias, stride, padding, dilation, groups
    )

    has_dilation = len(dilation) > 0
    dim = len(input_size)
    output_size: list[int] = []
    input_batch_size_dim = 0
    weight_output_channels_dim = 0
    output_size.append(input_size[input_batch_size_dim])
    output_size.append(weight_size[weight_output_channels_dim])

    for d in range(2, dim):
        dilation_ = dilation[d - 2] if has_dilation else 1
        kernel = dilation_ * (weight_size[d] - 1) + 1
        output_size.append(
            (input_size[d] + (2 * padding[d - 2]) - kernel) // stride[d - 2] + 1
        )
    return output_size


def conv1d(
    input: list[int],
    weight: list[int],
    bias: Optional[list[int]],
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    groups: int,
):
    if len(weight) != 3:
        raise AssertionError(f"Expected 3D weight for conv1d, got {len(weight)}D")
    if len(input) != 3:
        raise AssertionError(f"Expected 3D input for conv1d, got {len(input)}D")
    return conv_output_size(input, weight, bias, stride, padding, dilation, groups)


def conv2d(
    input: list[int],
    weight: list[int],
    bias: Optional[list[int]],
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    groups: int,
):
    if len(weight) != 4:
        raise AssertionError(f"Expected 4D weight for conv2d, got {len(weight)}D")
    if len(input) != 4:
        raise AssertionError(f"Expected 4D input for conv2d, got {len(input)}D")
    return conv_output_size(input, weight, bias, stride, padding, dilation, groups)


def conv_backwards(
    grad_output: list[int],
    input: list[int],
    weight: list[int],
    biases: Optional[list[int]],
):
    # Bias gradient is always generated regardless of if biases is supplied
    return _copy(input), _copy(weight), [grad_output[1]]


def conv_transpose2d_input(
    input: list[int],
    weight: list[int],
    bias: Optional[list[int]] = None,
    stride: Optional[list[int]] = None,
    padding: Optional[list[int]] = None,
    output_padding: Optional[list[int]] = None,
    groups: int = 1,
    dilation: Optional[list[int]] = None,
) -> list[int]:
    if stride is None:
        stride = [1, 1]
    if padding is None:
        padding = [0, 0]
    if output_padding is None:
        output_padding = [0, 0]
    if dilation is None:
        dilation = [1, 1]
    has_dilation = len(dilation) > 0
    dim = len(input)
    output_size: list[int] = []
    input_batch_size_dim = 0
    weight_output_channels_dim = 1
    output_size.append(input[input_batch_size_dim])
    output_size.append(weight[weight_output_channels_dim] * groups)

    for d in range(2, dim):
        dilation_ = dilation[d - 2] if has_dilation else 1
        kernel = dilation_ * (weight[d] - 1)
        output_size.append(
            (input[d] - 1) * stride[d - 2]
            - 2 * padding[d - 2]
            + kernel
            + output_padding[d - 2]
            + 1
        )
    return output_size


def conv_forwards(
    input: list[int],
    weight: list[int],
    bias: Optional[list[int]],
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    transposed: bool,
    output_padding: list[int],
    groups: int,
) -> list[int]:
    has_dilation = len(dilation) > 0
    has_output_padding = len(output_padding) > 0
    dim = len(input)
    output_size: list[int] = []
    input_batch_size_dim = 0
    weight_output_channels_dim = 1 if transposed else 0
    output_size.append(input[input_batch_size_dim])
    if transposed:
        output_size.append(weight[weight_output_channels_dim] * groups)
    else:
        output_size.append(weight[weight_output_channels_dim])

    for d in range(2, dim):
        dilation_ = dilation[d - 2] if has_dilation else 1
        output_padding_ = output_padding[d - 2] if has_output_padding else 0
        if transposed:
            kernel = dilation_ * (weight[d] - 1)
            output_size.append(
                (input[d] - 1) * stride[d - 2]
                - 2 * padding[d - 2]
                + kernel
                + output_padding_
                + 1
            )
        else:
            kernel = dilation_ * (weight[d] - 1) + 1
            output_size.append(
                (input[d] + (2 * padding[d - 2]) - kernel) // stride[d - 2] + 1
            )
    return output_size


def _conv_forwards(
    input: list[int],
    weight: list[int],
    bias: Optional[list[int]],
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    transposed: bool,
    output_padding: list[int],
    groups: int,
    benchmark: bool,
    deterministic: bool,
    cudnn_enabled: bool,
    allow_tf32: bool,
) -> list[int]:
    return conv_forwards(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    )


def batch_norm(
    input: list[int],
    weight: Optional[list[int]],
    bias: Optional[list[int]],
    running_mean: Optional[list[int]],
    running_var: Optional[list[int]],
    training: bool,
    momentum: float,
    eps: float,
    cudnn_enabled: bool,
):
    out: list[int] = []
    for elem in input:
        out.append(elem)
    return out


def conv3d(
    input: list[int],
    weight: list[int],
    bias: Optional[list[int]],
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    groups: int,
):
    if len(weight) != 5:
        raise AssertionError(f"Expected 5D weight for conv3d, got {len(weight)}D")
    if len(input) != 5:
        raise AssertionError(f"Expected 5D input for conv3d, got {len(input)}D")
    return conv_output_size(input, weight, bias, stride, padding, dilation, groups)


def maybe_wrap_dim(dim: int, dim_post_expr: int, wrap_scalar: bool = True):
    if dim_post_expr <= 0:
        if not wrap_scalar:
            raise AssertionError(
                "Expected wrap_scalar to be True when dim_post_expr <= 0"
            )
        dim_post_expr = 1
    min = -dim_post_expr
    max = dim_post_expr - 1
    if dim < min or dim > max:
        raise AssertionError(
            f"Dimension {dim} out of range (expected to be in range [{min}, {max}])"
        )
    if dim < 0:
        dim += dim_post_expr
    return dim


def zero_dim_tensor(input: Any):
    out: list[int] = []
    return out


def multiply_integers(li: list[int]):
    out = 1
    for elem in li:
        out = out * elem
    return out


def arange_end(end: number, inp0: Any, inp1: Any, inp2: Any, inp3: Any):
    if end < 0:
        raise AssertionError(f"Expected end ({end}) >= 0")
    return [int(math.ceil(end))]


def arange_start(
    start: number, end: number, inp0: Any, inp1: Any, inp2: Any, inp3: Any
):
    if end < 0:
        raise AssertionError(f"Expected end ({end}) >= 0")
    if end < start:
        raise AssertionError(f"Expected end ({end}) >= start ({start})")
    return [int(math.ceil(end - start))]


def arange_start_step(
    start: number, end: number, step: number, inp0: Any, inp1: Any, inp2: Any, inp3: Any
):
    if step == 0:
        raise AssertionError("step must not be zero")
    if step < 0:
        if start < end:
            raise AssertionError(
                f"Expected start ({start}) >= end ({end}) when step < 0"
            )
    else:
        if end < start:
            raise AssertionError(
                f"Expected end ({end}) >= start ({start}) when step > 0"
            )
    return [int(math.ceil((end - start) / step))]


def permute(input: list[int], dims: list[int]):
    if len(input) != len(dims):
        raise AssertionError(
            f"Expected len(input) ({len(input)}) == len(dims) ({len(dims)})"
        )
    ndim = len(dims)
    seen_dims: list[int] = []
    newSizes: list[int] = []
    for i in range(ndim):
        dim = maybe_wrap_dim(dims[i], ndim)
        seen_dims.append(dim)
        newSizes.append(input[dim])
    for i in range(1, ndim):
        for j in range(i):
            if seen_dims[i] == seen_dims[j]:
                raise AssertionError(
                    f"Repeated dimension {seen_dims[i]} in permute dimensions"
                )
    return newSizes


def movedim(self: list[int], source: list[int], destination: list[int]) -> list[int]:
    self_dim = len(self)
    if self_dim <= 1:
        return self
    normalized_src: list[int] = []
    normalized_dst: list[int] = []
    for i in range(len(source)):
        normalized_src.append(maybe_wrap_dim(source[i], self_dim))
        normalized_dst.append(maybe_wrap_dim(destination[i], self_dim))
    order = [-1 for i in range(self_dim)]
    src_dims = [i for i in range(self_dim)]
    dst_dims = [i for i in range(self_dim)]

    for i in range(len(source)):
        order[normalized_dst[i]] = normalized_src[i]
        src_dims[normalized_src[i]] = -1
        dst_dims[normalized_dst[i]] = -1

    source_dims: list[int] = []
    destination_dims: list[int] = []
    for ele in src_dims:
        if ele != -1:
            source_dims.append(ele)
    for ele in dst_dims:
        if ele != -1:
            destination_dims.append(ele)

    rest_dim = self_dim - len(source)
    for i in range(rest_dim):
        order[destination_dims[i]] = source_dims[i]
    return permute(self, order)


def flatten(input: list[int], start_dim: int, end_dim: int):
    start_dim = maybe_wrap_dim(start_dim, len(input))
    end_dim = maybe_wrap_dim(end_dim, len(input))
    if start_dim > end_dim:
        raise AssertionError(f"Expected start_dim ({start_dim}) <= end_dim ({end_dim})")
    if len(input) == 0:
        return [1]
    if start_dim == end_dim:
        # TODO: return self
        out: list[int] = []
        for elem in input:
            out.append(elem)
        return out
    slice_numel = 1
    for i in range(start_dim, end_dim + 1):
        slice_numel *= input[i]
    # TODO: use slicing when slice optimization has landed
    # slice_numel = multiply_integers(input[start_dim:end_dim - start_dim + 1])
    shape: list[int] = []
    for i in range(start_dim):
        shape.append(input[i])
    shape.append(slice_numel)
    for i in range(end_dim + 1, len(input)):
        shape.append(input[i])
    return shape


def nonzero_lower_bound(input: list[int]):
    return [0, len(input)]


def nonzero_upper_bound(input: list[int]):
    return [numel(input), len(input)]


def _reduce_along_dim(self: list[int], dim: int, keepdim: bool):
    dim = maybe_wrap_dim(dim, len(self))
    out: list[int] = []
    for i, self_dim in enumerate(self):
        if i == dim:
            if keepdim:
                out.append(1)
        else:
            out.append(self_dim)
    return out


def argmax(
    self: list[int], dim: Optional[int] = None, keepdim: bool = False
) -> list[int]:
    if dim is None:
        return []
    return _reduce_along_dim(self, dim, keepdim)


def bmm(self: list[int], mat2: list[int]) -> list[int]:
    if len(self) != 3:
        raise AssertionError(f"bmm only supports 3D tensors, got {len(self)}D")
    if len(mat2) != 3:
        raise AssertionError(f"bmm only supports 3D tensors, got {len(mat2)}D")
    if self[0] != mat2[0]:
        raise AssertionError(
            f"mismatching batch dimension: self[0]={self[0]}, mat2[0]={mat2[0]}"
        )
    if self[2] != mat2[1]:
        raise AssertionError(
            f"mismatching contracting dimension: self[2]={self[2]}, mat2[1]={mat2[1]}"
        )
    return [self[0], self[1], mat2[2]]


def _shape_as_tensor(self: list[int]) -> list[int]:
    return [len(self)]


def topk(self: list[int], k: int, dim: int = -1) -> tuple[list[int], list[int]]:
    if len(self) == 0:
        result: list[int] = []
    else:
        if k > self[dim]:
            raise AssertionError(
                f"k ({k}) is too big for dimension {dim} of size {self[dim]}"
            )
        result = _copy(self)
        result[dim] = k
    return result, result


def nll_loss_forward(
    self: list[int], target: list[int], weight: Optional[list[int]], reduction: int
) -> tuple[list[int], list[int]]:
    # This is taken shamelessly from the meta function in LossNLL.cpp
    self_dim = len(self)
    target_dim = len(target)
    if not (0 < self_dim <= 2):
        raise AssertionError(f"Expected 0 < self_dim <= 2, but got self_dim={self_dim}")
    if target_dim > 1:
        raise AssertionError(f"Expected target_dim <= 1, but got {target_dim}")
    no_batch_dim = self_dim == 1 and target_dim == 0
    if not (no_batch_dim or (self[0] == target[0])):
        raise AssertionError(
            f"Batch size mismatch: self[0]={self[0]}, target[0]={target[0]}"
        )
    n_classes = self[-1]
    scalar_shape: list[int] = []
    if weight is not None and not (len(weight) == 1 and weight[0] == n_classes):
        raise AssertionError(
            f"Expected weight to be None or have shape [n_classes], "
            f"got {weight} with n_classes={n_classes}"
        )
    if reduction == 0 and self_dim == 2:
        reduction_shape = [self[0]]
    else:
        reduction_shape = scalar_shape
    return reduction_shape, scalar_shape


def native_layer_norm(
    input: list[int], normalized_shape: list[int]
) -> tuple[list[int], list[int], list[int]]:
    reduction_shape: list[int] = []
    num_unreduced_dimensions = len(input) - len(normalized_shape)
    if num_unreduced_dimensions < 0:
        raise AssertionError(
            f"Expected len(input) ({len(input)}) >= len(normalized_shape) "
            f"({len(normalized_shape)})"
        )
    for i in range(num_unreduced_dimensions):
        reduction_shape.append(input[i])
    for i in range(num_unreduced_dimensions, len(input)):
        reduction_shape.append(1)
    return _copy(input), reduction_shape, reduction_shape


def native_batch_norm(
    input: list[int],
    weight: Optional[list[int]],
    bias: Optional[list[int]],
    running_mean: Optional[list[int]],
    running_var: Optional[list[int]],
    training: bool,
) -> tuple[list[int], list[int], list[int]]:
    if training:
        _size = [input[1]]
    else:
        _size = [0]
    return _copy(input), _size, _size


def _batch_norm_with_update(
    input: list[int],
    weight: Optional[list[int]],
    bias: Optional[list[int]],
    running_mean: Optional[list[int]],
    running_var: Optional[list[int]],
) -> tuple[list[int], list[int], list[int], list[int]]:
    _size = [input[1]]
    return _copy(input), _size, _size, [0]


def cross_entropy_loss(
    self: list[int],
    target: list[int],
    weight: Optional[list[int]] = None,
    reduction: int = 1,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> list[int]:
    result_shape = nll_loss_forward(self, target, weight, reduction)[0]
    return result_shape


"""
Currently deferring the enabling of this, as part of the propoasal to suspend
adding ops.
There are currently cases in the test case where this is being called
in the SSA opinfo tests with unexpected values (eg list of two ints, see the first
opinfo test). The behavior of index is significantly dependent on the inputs.

This could be an error with how we are matching up shape functions, or that this
function needs to just implement everything.

def index_Tensor(self: List[int], indices: List[Optional[List[int]]]) -> List[int]:
    assert len(indices) <= len(self), "More indices than dimensions to index"
    broadcasted_shape: List[int] = []
    for index_tensor_shape in indices:
        if index_tensor_shape is not None:
            broadcasted_shape = broadcast(broadcasted_shape, index_tensor_shape)
    return broadcasted_shape
"""

ScriptFn = torch._C.ScriptFunction
shape_compute_graph_mapping: dict[str, ScriptFn] = {}
bounded_compute_graph_mapping: dict[str, tuple[ScriptFn, ScriptFn]] = {}
script_func_map: dict[Callable, ScriptFn] = {}


def process_func(func: Callable):
    if func not in script_func_map:
        scripted_func = torch.jit.script(func)

        torch._C._jit_pass_inline(scripted_func.graph)

        for _ in range(2):
            torch._C._jit_pass_peephole(scripted_func.graph)
            torch._C._jit_pass_constant_propagation(scripted_func.graph)

        script_func_map[func] = scripted_func
    return script_func_map[func]


def add_shape_compute_mapping(operator_schema: str, func: Callable):
    global shape_compute_graph_mapping

    shape_compute_graph_mapping[operator_schema] = process_func(func)


def add_bounded_compute_mapping(
    operator_schema: str, lower_bound_func: Callable, upper_bound_func: Callable
):
    # Adds a shape compute function for both upper and lower bounds
    fns = (process_func(lower_bound_func), process_func(upper_bound_func))
    bounded_compute_graph_mapping[operator_schema] = fns


add_shape_compute_mapping(
    "aten::contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)",
    unary,
)
add_shape_compute_mapping(
    "aten::rsub.Tensor(Tensor self, Scalar other, Scalar alpha=1) -> Tensor", unary
)
add_shape_compute_mapping(
    "aten::dropout(Tensor input, float p, bool train) -> Tensor", unary
)
add_shape_compute_mapping(
    "aten::adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor",
    adaptive_avg_pool2d,
)
add_shape_compute_mapping(
    "prim::NumToTensor.Scalar(Scalar a) -> Tensor", zero_dim_tensor
)
add_shape_compute_mapping("prim::NumToTensor.bool(bool a) -> Tensor", zero_dim_tensor)
add_shape_compute_mapping(
    "aten::zeros(int[] size, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)",
    unary,
)
add_shape_compute_mapping(
    "aten::to.dtype(Tensor(a) self, int dtype, bool non_blocking=False, bool copy=False, int? memory_format=None) -> (Tensor(a))",
    unary,
)
add_shape_compute_mapping(
    "aten::arange(Scalar end, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)",
    arange_end,
)
add_shape_compute_mapping(
    "aten::arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
    arange_start,
)
add_shape_compute_mapping(
    "aten::arange.start_step(Scalar start, Scalar end, Scalar step, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
    arange_start_step,
)
add_shape_compute_mapping("aten::squeeze(Tensor(a) self) -> Tensor(a)", squeeze_nodim)
add_shape_compute_mapping(
    "aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)", squeeze
)
add_shape_compute_mapping(
    "aten::squeeze.dims(Tensor(a) self, int[] dim) -> Tensor(a)", squeeze_dims
)
add_shape_compute_mapping(
    "aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)", unsqueeze
)
add_shape_compute_mapping(
    "aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)",
    slice,
)
add_shape_compute_mapping(
    "aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)", select
)
add_shape_compute_mapping(
    "aten::index_select(Tensor self, int dim, Tensor index) -> Tensor", index_select
)
add_shape_compute_mapping(
    "aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, "
    "float eps=1e-05, bool cudnn_enable=True) -> Tensor",
    unary,
)
add_shape_compute_mapping(
    "aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor", unary
)
add_shape_compute_mapping(
    "aten::_no_grad_embedding_renorm_(Tensor weight, Tensor input, float max_norm, float norm_type) -> Tensor",
    unary,
)
add_shape_compute_mapping(
    "aten::embedding_renorm_(Tensor(a!) self, Tensor indices, float max_norm, float norm_type) -> Tensor(a!)",
    unary,
)
add_shape_compute_mapping(
    "aten::embedding(Tensor weight, Tensor indices, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor",
    embedding,
)
add_shape_compute_mapping("aten::mm(Tensor self, Tensor mat2) -> Tensor", mm)
add_shape_compute_mapping("aten::dot(Tensor self, Tensor tensor) -> Tensor", dot)
add_shape_compute_mapping("aten::mv(Tensor self, Tensor vec) -> Tensor", mv)
add_shape_compute_mapping("aten::matmul(Tensor self, Tensor other) -> Tensor", matmul)
add_shape_compute_mapping(
    "aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor", linear
)
add_shape_compute_mapping(
    "aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor",
    max_pool2d,
)
add_shape_compute_mapping(
    "aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)",
    max_pool2d_with_indices,
)
add_shape_compute_mapping("aten::t(Tensor(a) self) -> Tensor(a)", t)
add_shape_compute_mapping(
    "aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)", transpose
)
add_shape_compute_mapping(
    "aten::conv1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] dilation=1, int groups=1) -> Tensor",
    conv1d,
)
add_shape_compute_mapping(
    "aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor",
    conv2d,
)
add_shape_compute_mapping(
    "aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor",
    batch_norm,
)
add_shape_compute_mapping(
    "aten::conv3d(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1, int groups=1) -> Tensor",
    conv3d,
)
add_shape_compute_mapping(
    "aten::convolution_backward(Tensor grad_output, Tensor input, Tensor weight, int[]? bias_sizes, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)",
    conv_backwards,
)
add_shape_compute_mapping(
    "aten::convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor",
    conv_forwards,
)
add_shape_compute_mapping(
    "aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) -> Tensor",
    _conv_forwards,
)
add_shape_compute_mapping(
    "aten::conv_transpose2d.input(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int groups=1, int[2] dilation=1) -> Tensor",
    conv_transpose2d_input,
)
add_shape_compute_mapping(
    "aten::flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1) -> Tensor(a)",
    flatten,
)
add_shape_compute_mapping("aten::cat(Tensor[] tensors, int dim=0) -> Tensor", cat)
add_shape_compute_mapping("aten::stack(Tensor[] tensors, int dim=0) -> Tensor", stack)
add_shape_compute_mapping(
    "aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)", permute
)
add_shape_compute_mapping(
    "aten::movedim.intlist(Tensor(a) self, int[] source, int[] destination) -> Tensor(a)",
    movedim,
)
add_shape_compute_mapping("aten::view(Tensor(a) self, int[] size) -> Tensor(a)", view)
add_shape_compute_mapping(
    "aten::expand_as(Tensor(a) self, Tensor other) -> Tensor(a)", expand
)
add_shape_compute_mapping(
    "aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> Tensor(a)",
    expand_one_unused,
)
add_shape_compute_mapping(
    "aten::mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor",
    sum_mean_dim,
)
add_shape_compute_mapping(
    "aten::sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor",
    sum_mean_dim,
)
add_shape_compute_mapping(
    "aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)",
    max_dim,
)
add_shape_compute_mapping(
    "aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor", zero_dim_tensor
)
add_shape_compute_mapping(
    "aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor", zero_dim_tensor
)
add_shape_compute_mapping(
    "aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor",
    addmm,
)
add_shape_compute_mapping(
    "aten::upsample_nearest2d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> (Tensor)",
    upsample_nearest2d,
)
add_shape_compute_mapping(
    "aten::quantize_per_tensor(Tensor self, float scale, int zero_point, ScalarType dtype) -> Tensor",
    unary,
)
add_shape_compute_mapping(
    "aten::quantize_per_tensor.tensor_qparams(Tensor self, Tensor scale, Tensor zero_point, ScalarType dtype) -> Tensor",
    unary,
)
add_shape_compute_mapping("aten::dequantize(Tensor self) -> Tensor", unary)
add_shape_compute_mapping(
    "quantized::add(Tensor qa, Tensor qb, float scale, int zero_point) -> Tensor qc",
    broadcast,
)
add_shape_compute_mapping(
    "aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor", argmax
)
add_shape_compute_mapping("aten::bmm(Tensor self, Tensor mat2) -> Tensor", bmm)
add_shape_compute_mapping(
    "aten::_shape_as_tensor(Tensor self) -> Tensor", _shape_as_tensor
)
add_shape_compute_mapping(
    "aten::topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)",
    topk,
)
add_shape_compute_mapping(
    "aten::nll_loss_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index) -> (Tensor output, Tensor total_weight)",
    nll_loss_forward,
)
add_shape_compute_mapping(
    "aten::native_layer_norm(Tensor input, int[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)",
    native_layer_norm,
)
add_shape_compute_mapping(
    "aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)",
    native_batch_norm,
)
add_shape_compute_mapping(
    "aten::_native_batch_norm_legit(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)",
    native_batch_norm,
)
add_shape_compute_mapping(
    "aten::_native_batch_norm_legit.no_stats(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)",
    native_batch_norm,
)
add_shape_compute_mapping(
    "_batch_norm_with_update(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, float momentum, float eps) -> (Tensor, Tensor, Tensor, Tensor)",
    _batch_norm_with_update,
)

add_shape_compute_mapping(
    "aten::cross_entropy_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, SymInt ignore_index=-100, float label_smoothing=0.0) -> Tensor",
    cross_entropy_loss,
)
# add_shape_compute_mapping("aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor", index_Tensor)

# TODO: migrate over all of symbolic_shape_registry_util.cpp
# These are duplicated here so that the functions will be serialized
add_shape_compute_mapping(
    "aten::lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> Tensor",
    broadcast_three,
)
add_shape_compute_mapping(
    "aten::where.ScalarSelf(Tensor condition, Scalar self, Tensor other) -> Tensor",
    broadcast_one_three,
)
add_shape_compute_mapping(
    "aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)",
    broadcast_inplace,
)

# quantized_conv_prepack TODO

# Shape Compute Fn with upper and lower bounds
add_bounded_compute_mapping(
    "aten::nonzero(Tensor self) -> (Tensor)", nonzero_lower_bound, nonzero_upper_bound
)
<|endoftext|># mypy: allow-untyped-defs
import torch
from torch import Tensor


aten = torch.ops.aten
import inspect
import warnings
from collections.abc import Callable
from typing import TypeVar
from typing_extensions import ParamSpec

from torch.types import Number


decomposition_table: dict[str, torch.jit.ScriptFunction] = {}
function_name_set: set[str] = set()

_T = TypeVar("_T")
_P = ParamSpec("_P")


def check_decomposition_has_type_annotations(f) -> None:
    inspect_empty = inspect._empty  # type: ignore[attr-defined]
    sig = inspect.signature(f)
    for param in sig.parameters.values():
        if param.annotation == inspect_empty:
            raise AssertionError(
                f"No signature on param {param.name} for function {f.name}"
            )

    if sig.return_annotation == inspect_empty:
        raise AssertionError(f"No return annotation for function {f.name}")


def signatures_match(decomposition_sig, torch_op_sig):
    decomp_params = decomposition_sig.parameters
    op_params = torch_op_sig.parameters

    if len(decomp_params) != len(op_params):
        return False

    for decomp_param, op_param in zip(decomp_params.values(), op_params.values()):
        # can't check full equality yet because not all fields are correctly deduced
        # in the torch_op_sig - like default value
        # can't check 'kind' bc
        # kwarg-only values with defaults not yet supported in TS
        inspect_empty = inspect._empty  # type: ignore[attr-defined]
        for field in ["name", "annotation"]:
            if field == "name" and decomp_param.name == "self":
                warnings.warn(
                    "PyTorch uses 'input' instead of 'self' on public api", stacklevel=2
                )

            if getattr(decomp_param, field) != getattr(op_param, field):
                return False

        decomp_default = decomp_param.default
        op_default = op_param.default
        # default value not always correctly inferred as being present on torch schema,
        # but if specified on both they should be equal
        if decomp_default != inspect_empty and op_default != inspect_empty:
            if decomp_default != op_default:
                return False

    return decomposition_sig.return_annotation == torch_op_sig.return_annotation


def register_decomposition(
    aten_op: torch._ops.OpOverload,
    registry: dict[str, torch.jit.ScriptFunction] | None = None,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    def decomposition_decorator(f: Callable[_P, _T]) -> Callable[_P, _T]:
        nonlocal registry
        if registry is None:
            registry = decomposition_table

        if not isinstance(aten_op, torch._ops.OpOverload):
            raise AssertionError(
                f"Expected aten_op to be OpOverload, got {type(aten_op)}"
            )

        # Need unique name for jit function serialization
        if f.__name__ in function_name_set:
            raise AssertionError(f"Duplicated function name {f.__name__}")
        function_name_set.add(f.__name__)

        scripted_func = torch.jit.script(f)
        torch._C._jit_pass_inline(scripted_func.graph)

        for _ in range(2):
            torch._C._jit_pass_peephole(scripted_func.graph)
            torch._C._jit_pass_constant_propagation(scripted_func.graph)

        registry[str(aten_op._schema)] = scripted_func
        return f

    return decomposition_decorator


# TODO: replace torch.sigmoid -> aten.sigmoid


@register_decomposition(aten.var.correction)
def var_decomposition(
    input: Tensor,
    dim: list[int] | None = None,
    correction: Number | None = None,
    keepdim: bool = False,
) -> Tensor:
    if dim is None:
        dim_i: list[int] = []
        dim = dim_i

    if isinstance(dim, (tuple, list)) and len(dim) == 0:
        n = input.numel()
    else:
        n = 1
        for dim_i in dim:  # type: ignore[assignment]
            n *= input.shape[dim_i]  # type: ignore[call-overload]

    mean = aten.mean(input, dim, True)
    sub = input - mean
    sq = sub * sub
    sum = aten.sum(sq, dim, keepdim)

    if correction is None:
        denom = float(n - 1)
    else:
        if isinstance(correction, int):
            denom = float(n - correction)
        elif isinstance(correction, float):
            denom = float(n) - correction
        else:
            raise RuntimeError("correction must be int or float")

    # pyrefly: ignore [no-matching-overload]
    return sum / max(0, denom)


@register_decomposition(aten.var.default)
def var(input: Tensor, unbiased: bool = True) -> Tensor:
    return var_decomposition(input, correction=(1 if unbiased else 0))
<|endoftext|># mypy: allow-untyped-defs
"""JIT-related state.

This module stores various pieces of Python-global state relating to the JIT.

This is not intended to be imported directly; please use the exposed
functionalities in `torch.jit`.
"""

import os
import weakref
from typing import Any

import torch


class EnabledProxy:
    """Stores whether the JIT is enabled or not.

    This is just a wrapper for a bool, so that we get reference semantics
    """

    def __init__(self) -> None:
        self.enabled = self.parse_env(
            "PYTORCH_JIT", True, "> Using PyTorch JIT", "> PyTorch JIT DISABLED"
        )

    def parse_env(self, name, default, true_message, false_message):
        value = os.environ.get(name)
        if value is None:
            return default
        if value.lower() in {"1", "true", "yes"}:
            return True
        elif value.lower() in {"0", "false", "no"}:
            return False
        if value == "1v":
            print(true_message)
            return True
        elif value == "0v":
            print(false_message)
            return False
        raise ValueError(f"Unknown setting of {name}. Try using 0 or 1.")

    def __bool__(self) -> bool:
        return self.enabled


_enabled = EnabledProxy()


def disable() -> None:
    _enabled.enabled = False


def enable() -> None:
    _enabled.enabled = True


# The Python CompilationUnit. All functions and modules defined in Python will
# live in here. It's defined in Python because doing in cpp creates static
# destruction order issues.
_python_cu = torch._C.CompilationUnit()


# python class => ScriptClass mapping
_script_classes: dict[type[Any], type[Any]] = {}
_name_to_pyclass: dict[str, type[Any]] = {}


def _add_script_class(python_class, script_class) -> None:
    _script_classes[python_class] = script_class
    _name_to_pyclass[script_class.qualified_name()] = python_class


def _get_script_class(python_class):
    override = getattr(python_class, "_jit_override_qualname", None)
    if override is not None:
        python_class = _get_python_class(override)
    return _script_classes.get(python_class)


def _get_python_class(qualified_name):
    return _name_to_pyclass.get(qualified_name)


def _clear_class_state() -> None:
    _script_classes.clear()
    _name_to_pyclass.clear()


# Caching: we currently cache compilation of free functions and overloaded functions.
# To cache free functions we hold a weak ref to the function object and
# map to the compiled fn's qualified name.
# To cache overloaded functions we hold a weak ref to the function obj and
# map to all of its overloaded compiled fns.
# In the future we could consider caching more types of objects so that
# aliasing is preserved across separate compilations of the same object.

_jit_caching_layer: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_jit_function_overload_caching: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _try_get_jit_cached_overloads(key):
    qual_names = _jit_function_overload_caching.get(key, None)
    if qual_names:
        return [_python_cu.find_function(qual_name) for qual_name in qual_names]
    else:
        return None


def _set_jit_overload_cache(key, compiled_fns) -> None:
    _jit_function_overload_caching[key] = [fn.qualified_name for fn in compiled_fns]


def _try_get_jit_cached_function(key):
    if getattr(key, "__disable_jit_function_caching__", False) is True:
        return None
    qual_name = _jit_caching_layer.get(key, None)
    if qual_name:
        return _python_cu.find_function(qual_name)
    else:
        return None


def _set_jit_function_cache(key, value) -> None:
    # only free functions currently supported
    if not isinstance(value, torch.jit.ScriptFunction):
        raise AssertionError(f"Expected ScriptFunction, got {type(value)}")
    _jit_caching_layer[key] = value.qualified_name
<|endoftext|># mypy: allow-untyped-defs
import torch
from torch._jit_internal import _Await
from torch.jit._builtins import _register_builtin
from torch.utils import set_module


set_module(_Await, "torch.jit")


def _awaitable(func, *args, **kwargs):
    r"""Create Await object that will call specified function with specified args, when it is requested for the result."""
    return torch._C._awaitable(func, *args, **kwargs)


def _awaitable_wait(aw):
    r"""Request await the result of execution, if Await is not completed yet, the func will be called immediately."""
    return torch._C._awaitable_wait(aw)


def _awaitable_nowait(o):
    r"""Create completed Await with specified result."""
    return torch._C._awaitable_nowait(o)


_register_builtin(_awaitable_wait, "prim::awaitable_wait")
_register_builtin(_awaitable_nowait, "prim::awaitable_nowait")
<|endoftext|># mypy: allow-untyped-defs
import cmath
import math
import warnings
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
from torch.nn.modules.utils import (
    _list_with_default,
    _pair,
    _quadruple,
    _single,
    _triple,
)


_builtin_table: dict[int, str] | None = None

_modules_containing_builtins = (
    torch,
    torch._C._nn,
    torch._C._fft,  # type: ignore[attr-defined]
    torch._C._linalg,  # type: ignore[attr-defined]
    torch._C._nested,  # type: ignore[attr-defined]
    torch._C._sparse,  # type: ignore[attr-defined]
    torch._C._special,  # type: ignore[attr-defined]
)

_builtin_ops = [
    # Pairs of (function, op_name)
    (_pair, "aten::_pair"),
    (_quadruple, "aten::_quadruple"),
    (_single, "aten::_single"),
    (_triple, "aten::_triple"),
    (_list_with_default, "aten::list_with_default"),
    (OrderedDict, "aten::dict"),
    (dict, "aten::dict"),
    (cudnn.is_acceptable, "aten::cudnn_is_acceptable"),
    (math.ceil, "aten::ceil"),
    (math.copysign, "aten::copysign"),
    (math.erf, "aten::erf"),
    (math.erfc, "aten::erfc"),
    (math.exp, "aten::exp"),
    (math.expm1, "aten::expm1"),
    (math.fabs, "aten::fabs"),
    (math.floor, "aten::floor"),
    (math.gamma, "aten::gamma"),
    (math.lgamma, "aten::lgamma"),
    (math.log, "aten::log"),
    (math.log10, "aten::log10"),
    (math.log1p, "aten::log1p"),
    (math.pow, "aten::pow"),
    (math.sqrt, "aten::sqrt"),
    (math.isnan, "aten::isnan"),
    (math.asinh, "aten::asinh"),
    (math.atanh, "aten::atanh"),
    (math.cosh, "aten::cosh"),
    (math.sinh, "aten::sinh"),
    (math.tanh, "aten::tanh"),
    (math.acos, "aten::acos"),
    (math.asin, "aten::asin"),
    (math.atan, "aten::atan"),
    (math.atan2, "aten::atan2"),
    (math.cos, "aten::cos"),
    (math.sin, "aten::sin"),
    (math.tan, "aten::tan"),
    (math.asinh, "aten::asinh"),
    (math.atanh, "aten::atanh"),
    (math.acosh, "aten::acosh"),
    (math.fmod, "aten::fmod"),
    (math.modf, "aten::modf"),
    (math.factorial, "aten::factorial"),
    (math.frexp, "aten::frexp"),
    (math.isinf, "aten::isinf"),
    (math.degrees, "aten::degrees"),
    (math.radians, "aten::radians"),
    (cmath.isnan, "aten::isnan"),
    (cmath.isfinite, "aten::isfinite"),
    (cmath.isinf, "aten::isinf"),
    (cmath.phase, "aten::angle"),
    (cmath.rect, "aten::polar"),
    (cmath.log, "aten::log"),
    (cmath.log10, "aten::log10"),
    (cmath.sqrt, "aten::sqrt"),
    (cmath.exp, "aten::exp"),
    (cmath.sin, "aten::sin"),
    (cmath.tan, "aten::tan"),
    (cmath.cos, "aten::cos"),
    (cmath.asin, "aten::asin"),
    (cmath.acos, "aten::acos"),
    (cmath.atan, "aten::atan"),
    (cmath.sinh, "aten::sinh"),
    (cmath.cosh, "aten::cosh"),
    (cmath.tanh, "aten::tanh"),
    (cmath.asinh, "aten::asinh"),
    (cmath.acosh, "aten::acosh"),
    (cmath.atanh, "aten::atanh"),
    (math.ldexp, "aten::ldexp"),
    (torch._assert, "aten::_assert"),
    (torch.autograd.grad, "aten::grad"),
    (torch.autograd.backward, "aten::backward"),
    (torch._C._infer_size, "aten::_infer_size"),
    (torch.broadcast_shapes, "aten::broadcast_shapes"),
    (
        torch.nn.functional._no_grad_embedding_renorm_,  # type: ignore[attr-defined]
        "aten::_no_grad_embedding_renorm_",
    ),
    (torch.nn.functional.assert_int_or_pair, "aten::_assert_int_or_pair"),
    (torch.nn.init._no_grad_fill_, "aten::_no_grad_fill_"),
    (torch.nn.init._no_grad_normal_, "aten::_no_grad_normal_"),
    (torch.nn.init._no_grad_uniform_, "aten::_no_grad_uniform_"),
    (torch.nn.init._no_grad_zero_, "aten::_no_grad_zero_"),
    (torch._C._get_tracing_state, "aten::_get_tracing_state"),
    (torch._C._get_cpu_capability, "aten::_get_cpu_capability"),
    (warnings.warn, "aten::warn"),
    (torch._VF.stft, "aten::stft"),  # type: ignore[attr-defined]
    (torch._VF.istft, "aten::istft"),  # type: ignore[attr-defined]
    (torch._VF.cdist, "aten::cdist"),  # type: ignore[attr-defined]
    (torch._VF.norm, "aten::norm"),  # type: ignore[attr-defined]
    (torch._VF.unique_dim, "aten::unique_dim"),
    (torch._VF.unique_consecutive, "aten::unique_consecutive"),  # type: ignore[attr-defined]
    (torch._VF.nuclear_norm, "aten::nuclear_norm"),
    (torch._VF.frobenius_norm, "aten::frobenius_norm"),
    (torch._VF.tensordot, "aten::tensordot"),  # type: ignore[attr-defined]
]

# ops in torch.functional are bound to torch
# in these cases, we want to resolve the function to their python implementation
# instead looking up a builtin "aten::" schema


def _gen_torch_functional_registered_ops():
    # eventually ops should encompass all of torch/functional.py, (torch.functional.__all__)
    # but we are currently only able to compile some of the functions. additionally,
    # some functions directly map to their aten:: implementations.
    # TODO: add support for more ops
    ops = [
        "stft",
        "istft",
        "lu",
        "cdist",
        "norm",
        "unique",
        "unique_consecutive",
        "tensordot",
    ]
    return {getattr(torch.functional, name) for name in ops}


_functional_registered_ops = _gen_torch_functional_registered_ops()


def _is_special_functional_bound_op(fn):
    return fn in _functional_registered_ops


# lazily built to ensure the correct initialization order
def _get_builtin_table():
    global _builtin_table
    if _builtin_table is not None:
        return _builtin_table
    _builtin_table = {}

    def register_all(mod) -> None:
        for name in dir(mod):
            v = getattr(mod, name)
            if (
                callable(v)
                and not _is_special_functional_bound_op(v)
                and v is not torch.no_grad
                and v is not torch.autocast
            ):
                # Fixup inconsistency in segment_reduce
                if name == "_segment_reduce":
                    name = name[1:]
                _builtin_ops.append((v, "aten::" + name))

    for mod in _modules_containing_builtins:
        register_all(mod)

    _builtin_ops.append((math.gcd, "aten::gcd"))
    _builtin_ops.append((math.isfinite, "aten::isfinite"))
    _builtin_ops.append((math.remainder, "aten::mathremainder"))  # type: ignore[attr-defined]

    import torch.distributed.autograd as dist_autograd

    if dist_autograd.is_available():
        _builtin_ops.append((dist_autograd.get_gradients, "aten::get_gradients"))
        _builtin_ops.append((dist_autograd.backward, "aten::dist_backward"))

    # populate the _builtin_table from _builtin_ops
    for builtin, aten_op in _builtin_ops:
        _builtin_table[id(builtin)] = aten_op

    return _builtin_table


def _register_builtin(fn, op) -> None:
    _get_builtin_table()[id(fn)] = op


def _find_builtin(fn):
    return _get_builtin_table().get(id(fn))
<|endoftext|># mypy: allow-untyped-defs
import collections
import functools
import inspect
import textwrap
import types
import warnings

import torch
import torch._jit_internal as _jit_internal
from torch._sources import fake_range
from torch.jit._builtins import _find_builtin
from torch.jit._check import AttributeTypeIsSupportedChecker
from torch.jit._state import _add_script_class, _get_script_class, _python_cu
from torch.jit.frontend import (
    get_class_properties,
    get_default_args,
    get_jit_class_def,
    get_jit_def,
)
from torch.nn import Module


ScriptMethodStub = collections.namedtuple(
    "ScriptMethodStub", ("resolution_callback", "def_", "original_method")
)
PropertyStub = collections.namedtuple("PropertyStub", ("resolution_callback", "def_"))


# TODO: there should be a more principled way of doing this.
ignored_attributes = [
    "_version",
    "_parameters",
    "_buffers",
    "_non_persistent_buffers_set",
    "_backward_hooks",
    "_backward_pre_hooks",
    "_forward_hooks",
    "_forward_hooks_with_kwargs",
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_forward_hooks_always_called",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
    "_modules",
    "_initializing",
    "dump_patches",
]


def _compile_and_register_class(obj, rcb, qualified_name):
    script_class = _get_script_class(obj)

    if not script_class:
        ast = get_jit_class_def(obj, obj.__name__)
        defaults = torch.jit.frontend.get_default_args_for_class(obj)
        script_class = torch._C._jit_script_class_compile(
            qualified_name, ast, defaults, rcb
        )
        _add_script_class(obj, script_class)

    return script_class


def make_stub(func, name):
    rcb = _jit_internal.createResolutionCallbackFromClosure(func)
    ast = get_jit_def(func, name, self_name="RecursiveScriptModule")
    return ScriptMethodStub(rcb, ast, func)


def make_stub_from_method(nn_module, method_name):
    func = getattr(nn_module, method_name)
    if isinstance(func, ScriptMethodStub):
        return func
    # Make sure the name present in the resulting AST will match the name
    # requested here. The only time they don't match is if you do something
    # like:
    #   def _forward(self):
    #       pass
    #   forward = _forward
    # In this case, the actual function object will have the name `_forward`,
    # even though we requested a stub for `forward`.
    return make_stub(func, method_name)


def make_stubs_from_exported_methods(mod):
    stubs = []
    for name in dir(mod):
        item = getattr(mod, name, None)
        if (
            _jit_internal.get_torchscript_modifier(item)
            is _jit_internal.FunctionModifiers.EXPORT
        ):
            stubs.append(make_stub_from_method(mod, name))

    return stubs


def jit_ignored_properties(module):
    user_annotated_ignored_attributes = getattr(
        module, "__jit_ignored_attributes__", []
    )

    def get_properties_names(module):
        return {k for k, v in vars(module).items() if isinstance(v, property)}

    properties = get_properties_names(type(module))
    user_annoted_ignored_properties = set()

    for ignored_attr in user_annotated_ignored_attributes:
        if ignored_attr in properties:
            user_annoted_ignored_properties.add(ignored_attr)
    return user_annoted_ignored_properties


# base types that can be constants
# in addition, tuples and lists of these base types are also considered constants
# If you edit this list, then you also need to edit the handlers in
# ConstantValue in jit/script/init.cpp
_constant_types = (
    bool,
    float,
    int,
    str,
    type(None),
    torch.device,
    torch.layout,
    torch.dtype,
    torch.qscheme,
)


def _get_valid_constant(attr, v, owner_type):
    if isinstance(v, _constant_types):
        return v
    elif isinstance(v, (tuple, list)):
        return tuple(_get_valid_constant(attr, x, owner_type) for x in v)
    constants = ", ".join(torch.typename(typ) for typ in _constant_types)
    raise TypeError(
        textwrap.dedent(
            f"""
        '{torch.typename(type(v))}' object in attribute '{owner_type}.{attr}' is not a valid constant.
        Valid constants are:
        1. a nn.ModuleList
        2. a value of type {{{constants}}}
        3. a list or tuple of (2)
        """
        )
    )


class SourceContext(torch._C._jit_tree_views.SourceRangeFactory):
    pass


def get_annotations(obj):
    # In Python-3.10+ it is recommended to use inspect.get_annotations
    # See https://docs.python.org/3.10/howto/annotations.html
    # But also, in 3.10 annotations from base class are not inherited
    # by unannotated derived one, so they must be manually extracted
    annotations = inspect.get_annotations(obj)
    if annotations:
        return annotations

    def get_cls_annotations(cls):
        cls_annotations = inspect.get_annotations(cls)
        if cls_annotations:
            return cls_annotations
        for base in cls.__bases__:
            cls_annotations = get_cls_annotations(base)
            if cls_annotations:
                return cls_annotations
        return {}

    cls = obj if isinstance(obj, type) else type(obj)
    return get_cls_annotations(cls)


def infer_concrete_type_builder(nn_module, share_types=True):
    """
    Build a ConcreteModuleTypeBuilder from an nn.Module.

    This ConcreteModuleType doesn't have a JIT type associated with it yet, it
    must be filled in by the caller.
    """
    concrete_type_builder = torch._C.ConcreteModuleTypeBuilder(type(nn_module))
    if isinstance(nn_module, (torch.nn.ModuleDict)):
        concrete_type_builder.set_module_dict()
    if isinstance(nn_module, (torch.nn.ModuleList, torch.nn.Sequential)):
        concrete_type_builder.set_module_list()
    if isinstance(nn_module, (torch.nn.ParameterList)):
        concrete_type_builder.set_parameter_list()
    if isinstance(nn_module, (torch.nn.ParameterDict)):
        concrete_type_builder.set_parameter_dict()

    class_annotations = get_annotations(nn_module)
    if isinstance(nn_module, (torch.ao.quantization.QuantWrapper)):
        class_annotations = {}

    # Get user-annotated ignored attributes.
    user_annotated_ignored_attributes = getattr(
        nn_module, "__jit_ignored_attributes__", []
    )
    concrete_type_builder.add_ignored_attributes(user_annotated_ignored_attributes)
    ignored_properties = jit_ignored_properties(nn_module)

    # try to infer the type from type annotation or from the object itself
    def infer_type(name, item):
        # The forward function from Module is special; never use this annotations; we
        # need to infer type directly using JIT.  I originally wanted to write
        # this test as isinstance(class_annotations[name], Callable) but
        # isinstance on typing things doesn't seem to work: isinstance(list, Callable)
        # is also true!
        inferred = False
        try:
            if (
                name in class_annotations
                and class_annotations[name]
                != torch.nn.Module.__annotations__["forward"]
            ):
                ann_to_type = torch.jit.annotations.ann_to_type(
                    class_annotations[name], fake_range()
                )
                attr_type = torch._C.InferredType(ann_to_type)
            elif isinstance(item, torch.jit.Attribute):
                ann_to_type = torch.jit.annotations.ann_to_type(item.type, fake_range())
                attr_type = torch._C.InferredType(ann_to_type)
            else:
                attr_type = torch._C._jit_try_infer_type(item)
                inferred = True
        except RuntimeError as re:
            raise RuntimeError(f"Error inferring type for {name}: {item}: {re}") from re

        return attr_type, inferred

    added_names = set()

    for name, item in nn_module._parameters.items():
        if name in user_annotated_ignored_attributes:
            continue

        if not (item is None or isinstance(item, torch.Tensor)):
            raise AssertionError(
                f"Expected parameter '{name}' to be None or Tensor, got {type(item)}"
            )
        attr_type, _ = infer_type(name, item)
        # We currently have the invariant in various places in our code
        # that parameters must be Tensors. However, the nn.Module API also
        # allows NoneType parameters. These parameters are not returned as
        # part of `parameters()` and its variants, but are available
        # through direct attribute access.
        concrete_type_builder.add_attribute(name, attr_type.type(), True, False)
        added_names.add(name)

    for name, item in nn_module._buffers.items():
        if name in user_annotated_ignored_attributes:
            continue

        if not (item is None or isinstance(item, torch.Tensor)):
            raise AssertionError(
                f"Expected buffer '{name}' to be None or Tensor, got {type(item)}"
            )
        attr_type, _ = infer_type(name, item)
        concrete_type_builder.add_attribute(name, attr_type.type(), False, True)
        added_names.add(name)

    for name, item in nn_module._modules.items():
        if name in user_annotated_ignored_attributes:
            continue

        attr_type, _ = infer_type(name, item)
        if item is None:
            # Modules can be None. We don't have direct support for optional
            # Modules, so we register it as a NoneType attribute instead.
            concrete_type_builder.add_attribute(name, attr_type.type(), False, False)
            continue
        if attr_type.success():
            if not attr_type.type().is_interface_type():
                raise AssertionError(
                    f"Expected inferred type to be interface type for '{name}'"
                )
            # if the type can be inferred, it should be a module interface type
            sub_concrete_type = torch._C.ConcreteModuleType.from_jit_type(
                attr_type.type()
            )
        else:
            # otherwise we get the concrete module type for item and add it to concrete_type
            sub_concrete_type = get_module_concrete_type(item, share_types)
        concrete_type_builder.add_module(name, sub_concrete_type)

        added_names.add(name)

    # populate constants_set
    constants_set = set(getattr(nn_module, "__constants__", ()))

    # Constants annotated via `Final[T]` rather than being added to `__constants__`
    for name, ann in class_annotations.items():
        if torch._jit_internal.is_final(ann):
            constants_set.add(name)

    for name in constants_set:
        if name in added_names:
            # TODO: We should really error in this case, but its bc-breaking so
            # we need to warn for at least one release
            if name in nn_module._modules:
                hint = "submodule"
            elif name in nn_module._buffers:
                hint = "buffer"
            elif name in nn_module._parameters:
                hint = "parameter"
            else:
                raise AssertionError(
                    "added_names must be submodule, parameter, or buffer"
                )

            warnings.warn(
                f"'{name}' was found in ScriptModule constants, "
                f" but it is a non-constant {hint}. Consider removing it.",
                stacklevel=2,
            )
            continue
        if not hasattr(nn_module, name):
            # TODO: We should really error in this case, but its bc-breaking so
            # we need to warn for at least one release
            warnings.warn(
                f"'{name}' was found in ScriptModule constants, "
                "but was not actually set in __init__. "
                "Consider removing it.",
                stacklevel=2,
            )
            continue
        value = getattr(nn_module, name)
        concrete_type_builder.add_constant(
            name, _get_valid_constant(name, value, type(nn_module).__name__)
        )
        added_names.add(name)

    # populate overloads
    overloads = getattr(nn_module, "__overloads__", {})
    # update with any annotated overloads
    overloads.update(
        get_overload_name_mapping(
            get_overload_annotations(nn_module, ignored_properties)
        )
    )
    for name, overloaded_names in overloads.items():
        concrete_type_builder.add_overload(name, overloaded_names)

    for name, value in nn_module.__dict__.items():
        if name in ignored_attributes or name.startswith("__"):
            # Python objects have lots of random attributes attached to them;
            # PyTorch adds a few more. Prevent these from getting compiled.
            continue

        if name in user_annotated_ignored_attributes:
            continue

        if name in added_names:
            # Don't re-add anything we already added
            continue

        isoverloadpacket = isinstance(value, torch._ops.OpOverloadPacket)
        if isoverloadpacket:
            value = value.op
        # Handle Python function attributes
        if inspect.isfunction(value):
            try:
                scripted_fn = torch.jit.script(value)
                concrete_type_builder.add_function_attribute(
                    name, torch._C._jit_try_infer_type(scripted_fn).type(), value
                )
            except Exception as e:
                # If we fail to script the function, it isn't a hard error.
                # Instead, we will add it to the list of attributes we failed
                # to convert, with the compilation error.
                hint = (
                    "(This function exists as an attribute on the Python module, "
                    "but we failed to compile it to a TorchScript function. "
                    f"\nThe error stack is reproduced here:\n{e})"
                )
                concrete_type_builder.add_failed_attribute(name, hint)

            continue

        # Handle calls to builtin functions (either bespoke builtins from torch.jit._builtins or
        # a call to an aten function like torch.add)
        builtin_symbol_name = _find_builtin(value)
        if builtin_symbol_name:
            concrete_type_builder.add_builtin_function(name, builtin_symbol_name)
            continue

        # Handle Script function attributes
        if isinstance(value, torch.jit.ScriptFunction):
            concrete_type_builder.add_function_attribute(
                name, torch._C._jit_try_infer_type(value).type(), value
            )
            continue

        # If we got here, this is a regular "data" attribute, add it to the concrete type
        attr_type, inferred = infer_type(name, value)
        if attr_type.success():
            concrete_type_builder.add_attribute(name, attr_type.type(), False, False)
        else:
            # TODO: could add more detail here. For example, what the user should do
            # when the pytype is `list` or `NoneType`
            inferred_msg = (
                "Its type was inferred; try adding a type annotation for the attribute."
                if inferred
                else ""
            )
            additional_info = f"{attr_type.reason()}. {inferred_msg}"
            hint = (
                "(This attribute exists on the Python module, "
                f"but we failed to convert Python type: '{torch.typename(type(value))}' "
                f"to a TorchScript type. {additional_info})"
            )
            concrete_type_builder.add_failed_attribute(name, hint)

    # add hooks to concrete type
    for hook in nn_module._forward_hooks.values():
        concrete_type_builder.add_forward_hook(hook)
    for pre_hook in nn_module._forward_pre_hooks.values():
        concrete_type_builder.add_forward_pre_hook(pre_hook)

    return concrete_type_builder


class ConcreteTypeStore:
    type_store: dict[type[Module], list[torch._C.ConcreteModuleType]]
    methods_compiled: set[torch._C.ConcreteModuleType]

    def __init__(self) -> None:
        # Python module type => List[ConcreteModuleType)]
        self.type_store = {}
        # ConcreteTypes that have had their methods already compiled
        self.methods_compiled = set()

    def get_or_create_concrete_type(self, nn_module):
        """Infer a ConcreteType from this `nn.Module` instance. Underlying JIT types are reused if possible."""
        concrete_type_builder = infer_concrete_type_builder(nn_module)

        nn_module_type = type(nn_module)
        if nn_module_type not in self.type_store:
            self.type_store[nn_module_type] = []

        # Search the type store for an already-available JIT type
        known_types = self.type_store[nn_module_type]
        for known_type in known_types:
            if known_type.equals(concrete_type_builder):
                return known_type

        # We didn't find anything; generate a new JIT type from this concrete type
        concrete_type = concrete_type_builder.build()
        self.type_store[nn_module_type].append(concrete_type)
        return concrete_type


concrete_type_store = ConcreteTypeStore()


def create_methods_and_properties_from_stubs(
    concrete_type, method_stubs, property_stubs
) -> None:
    method_defs = [m.def_ for m in method_stubs]
    method_rcbs = [m.resolution_callback for m in method_stubs]
    method_defaults = [get_default_args(m.original_method) for m in method_stubs]

    property_defs = [p.def_ for p in property_stubs]
    property_rcbs = [p.resolution_callback for p in property_stubs]

    concrete_type._create_methods_and_properties(
        property_defs, property_rcbs, method_defs, method_rcbs, method_defaults
    )


def create_hooks_from_stubs(concrete_type, hook_stubs, pre_hook_stubs) -> None:
    hook_defs = [h.def_ for h in hook_stubs]
    hook_rcbs = [h.resolution_callback for h in hook_stubs]

    pre_hook_defs = [h.def_ for h in pre_hook_stubs]
    pre_hook_rcbs = [h.resolution_callback for h in pre_hook_stubs]

    concrete_type._create_hooks(hook_defs, hook_rcbs, pre_hook_defs, pre_hook_rcbs)


def get_module_concrete_type(nn_module, share_types=True):
    """
    Get a concrete type for nn_modules.

    If share_types is True, the concrete type is fetched from concrete_type_store.
    If it is False, a new concrete type is created without first searching concrete_type_store.

    Args:
        nn_module:  The original Python nn.Module that we are creating a ScriptModule for.
        share_types = Whether to share underlying JIT types between modules (if possible).

    Returns:
        A concrete type for nn_module.
    """
    if not isinstance(nn_module, Module):
        raise AssertionError(f"Expected Module, got {type(nn_module)}")
    if isinstance(nn_module, torch.jit.ScriptModule) and hasattr(
        nn_module, "_concrete_type"
    ):
        return nn_module._concrete_type

    if share_types:
        # Look into the store of cached JIT types
        concrete_type = concrete_type_store.get_or_create_concrete_type(nn_module)
    else:
        # Get a concrete type directly, without trying to reuse an existing JIT
        # type from the type store.
        concrete_type_builder = infer_concrete_type_builder(nn_module, share_types)
        concrete_type_builder.set_poisoned()
        concrete_type = concrete_type_builder.build()

    return concrete_type


def create_script_class(obj):
    """
    Create and return a RecursiveScriptClass instance from a Python object.

    Arguments:
        obj: A Python object.
    """
    qualified_class_name = _jit_internal._qualified_name(type(obj))
    rcb = _jit_internal.createResolutionCallbackForClassMethods(type(obj))
    # Script the type of obj if it hasn't already been scripted.
    _compile_and_register_class(type(obj), rcb, qualified_class_name)
    class_ty = _python_cu.get_class(qualified_class_name)
    # Create an empty torch._C.ScriptObject with the scripted type.
    cpp_object = torch._C._create_object_with_type(class_ty)
    # Copy all of the attributes over to the torch._C.ScriptObject.
    for name, value in obj.__dict__.items():
        cpp_object.setattr(name, value)

    # Wrap the torch._C.ScriptObject in a RecursiveScriptClass instance.
    return wrap_cpp_class(cpp_object)


def create_script_module(nn_module, stubs_fn, share_types=True, is_tracing=False):
    """
    Create a new ScriptModule from an nn.Module.

    Args:
        nn_module:  The original Python nn.Module that we are creating a ScriptModule for.
        stubs_fn:  Lambda that takes an nn.Module and generates a list of ScriptMethodStubs to compile.
        share_types:  Whether to share underlying JIT types between modules (if possible).
            NOTE: Only set to False this when we cannot guarantee type sharing will work
                correctly. This only happens today for traced modules, where the same
                module can produce different traced methods depending on the inputs.
        is_tracing: Whether this function is called during tracing or scripting. If tracing,
                we don't need to do AttributeTypeIsSupportedChecker because all the unsupported
                attributes will be baked as constant in the tracing graph. In addition,
                this check significantly slows down the traced modules when the module size is big.
    """
    if isinstance(nn_module, torch.jit.RecursiveScriptModule):
        raise AssertionError("Cannot script a RecursiveScriptModule (already compiled)")
    check_module_initialized(nn_module)
    concrete_type = get_module_concrete_type(nn_module, share_types)
    if not is_tracing:
        AttributeTypeIsSupportedChecker().check(nn_module)
    return create_script_module_impl(nn_module, concrete_type, stubs_fn)


def create_script_module_impl(nn_module, concrete_type, stubs_fn):
    """
    Convert an nn.Module to a RecursiveScriptModule.

    Args:
        nn_module:  The original Python nn.Module that we are creating a ScriptModule for.
        concrete_type:  The fully initialized ConcreteType of the module.
        stubs_fn:  Lambda that takes an nn.Module and generates a list of ScriptMethodStubs to compile.
    """
    cpp_module = torch._C._create_module_with_type(concrete_type.jit_type)
    method_stubs = stubs_fn(nn_module)
    property_stubs = get_property_stubs(nn_module)
    hook_stubs, pre_hook_stubs = get_hook_stubs(nn_module)
    ignored_properties = jit_ignored_properties(nn_module)

    def init_fn(script_module) -> None:
        # Initialize the ScriptModule:
        # 1. Copy the attributes/parameters/buffers from the original `nn_module` to the new ScriptModule.
        for name in concrete_type.get_attributes():
            orig_value = getattr(nn_module, name)
            orig_value = (
                orig_value.value
                if isinstance(orig_value, torch.jit.Attribute)
                else orig_value
            )
            cpp_module.setattr(name, orig_value)

        # 2. Copy the submodules from the original `nn_module` to the new ScriptModule,
        #    recursively scripting them.
        for name, sub_concrete_type in concrete_type.get_modules():
            orig_value = getattr(nn_module, name)
            if not isinstance(orig_value, Module):
                raise AssertionError(f"Expected Module but got {type(orig_value)}")
            module_type = sub_concrete_type.jit_type
            if isinstance(module_type, torch._C.InterfaceType):
                # use the interface inference rule to compile the module
                scripted = interface_script(module_type, orig_value)
            elif isinstance(orig_value, torch.jit.ScriptModule):
                scripted = orig_value
            else:
                # always reuse the provided stubs_fn to infer the methods to compile
                scripted = create_script_module_impl(
                    orig_value, sub_concrete_type, stubs_fn
                )

            cpp_module.setattr(name, scripted)
            script_module._modules[name] = scripted

        # 3. Copy @ignored/@unused methods and attrs from the original `nn_module` to the new ScriptModule.
        #    This ensures we can access these Python methods on the ScriptModule.
        for name in dir(nn_module):
            if name in ignored_properties:
                continue
            item = getattr(nn_module, name, None)
            if inspect.ismethod(item) and _jit_internal.is_ignored_fn(item):
                unbound_function = getattr(nn_module, name).__func__
                bound_method = unbound_function.__get__(script_module)
                setattr(script_module, name, bound_method)
            elif concrete_type.is_ignored_attribute(name):
                setattr(script_module, name, item)

        # For convenience, attach the concrete type to the new ScriptModule
        script_module._concrete_type = concrete_type

    # Actually create the ScriptModule, initializing it with the function we just defined
    script_module = torch.jit.RecursiveScriptModule._construct(cpp_module, init_fn)

    # Compile methods if necessary
    if concrete_type not in concrete_type_store.methods_compiled:
        create_methods_and_properties_from_stubs(
            concrete_type, method_stubs, property_stubs
        )
        # Create hooks after methods to ensure no name collisions between hooks and methods.
        # If done before, hooks can overshadow methods that aren't exported.
        create_hooks_from_stubs(concrete_type, hook_stubs, pre_hook_stubs)
        torch._C._run_emit_module_hook(cpp_module)
        concrete_type_store.methods_compiled.add(concrete_type)

    # Copy the forward hooks and pre-hooks to the new ScriptModule
    # to allow the hooks to be run from eager as ScriptFunctions
    for idx, fn in enumerate(script_module._c._get_forward_pre_hooks()):
        script_module._forward_pre_hooks[idx] = fn
    for idx, fn in enumerate(script_module._c._get_forward_hooks()):
        script_module._forward_hooks[idx] = fn

    # Special handling so methods like __len__ work in script methods on classes derived from containers
    if (
        isinstance(
            nn_module, (torch.nn.ModuleList, torch.nn.Sequential, torch.nn.ModuleDict)
        )
        and "__len__" not in cpp_module._method_names()
    ):
        script_module.define(f"def __len__(self):\n   return {len(nn_module)}\n")
    if (
        isinstance(nn_module, torch.nn.ModuleDict)
        and "__contains__" not in cpp_module._method_names()
    ):
        if len(nn_module.keys()):
            keys = repr(list(nn_module.keys()))
            script_module.define(
                f"def __contains__(self, key: str):\n   return key in {keys}\n"
            )
        else:
            script_module.define("def __contains__(self, key: str):\n   return False\n")

    # Make the compiled methods available to the Python ScriptModule class.
    for method_stub in method_stubs:
        if method_stub.original_method is None:
            # define()'d methods don't have a Python original_method, so we
            # don't need to do any Python re-wrapping stuff
            continue

        name = method_stub.original_method.__name__
        if name != method_stub.def_.name().name:
            # TODO: Why skip this? Because @torch.jit._overload_method will
            # mangle the name of the function.
            continue
        script_method = cpp_module._get_method(name)

        # Wrap the original to propagate docstrings and such.
        # TODO: we don't currently do this functions that are recursively
        # compiled, we should.
        wrapped_script_method = functools.wraps(method_stub.original_method)(
            script_method
        )

        # Add the methods to the script_module directly. This ensures they will
        # be found first when `name` is looked up (as opposed to the stubs or
        # nn.Module.forward)
        script_module.__dict__[name] = wrapped_script_method

    # Make module properties available on the Python ScriptModule class.
    for property_stub in property_stubs:
        property_name = property_stub.def_.name().name
        fget = cpp_module._get_method(property_stub.def_.getter_name().name)
        # Setter is optional, so it may not exist.
        setter_name = property_stub.def_.setter_name()
        fset = cpp_module._get_method(setter_name.name) if setter_name else None
        script_module.__dict__[property_name] = property(property_name, fget, fset)  # type: ignore[arg-type]

    # copy over python methods to script module if they aren't defined on the script module
    # this is currently an internal api used only on module containers
    for name in dir(nn_module):
        if name in ignored_properties:
            continue
        item = getattr(nn_module, name, None)
        if (
            _jit_internal.get_torchscript_modifier(item)
            is _jit_internal.FunctionModifiers.COPY_TO_SCRIPT_WRAPPER
        ):
            add_python_attr_to_scripted_model(script_module, nn_module, name)

    return script_module


# We define shims of certain attributes on the RecursiveScriptModule to support
# magic methods. To check if a script model defines an attribute we need
# to also check that the attribute is not the shim
def script_model_defines_attr(script_model, attr):
    script_attr = getattr(script_model, attr, None)
    if script_attr is None:
        return False
    default_attr = getattr(torch.jit.RecursiveScriptModule, attr, None)
    if default_attr is None:
        return False
    return script_attr != default_attr


def add_python_attr_to_scripted_model(script_model, orig, attr) -> None:
    if hasattr(orig, attr) and script_model_defines_attr(script_model, attr):
        setattr(script_model, attr, getattr(orig, attr))


def get_overload_annotations(mod, jit_ignored_properties):
    # original function => [(mangled overload name, overload function)]
    overloads = {}

    for name in dir(type(mod)):
        if name in jit_ignored_properties:
            continue
        item = getattr(mod, name, None)
        if not callable(item):
            continue

        # builtin functions like repr() in python 2 do not have __module__ defined
        if hasattr(item, "__module__") and item.__module__ is not None:
            method_overloads = _jit_internal._get_overloaded_methods(
                item, mod.__class__
            )
            if method_overloads is None:
                continue

            # pyrefly: ignore [missing-attribute]
            if item.__func__ in method_overloads:
                raise RuntimeError(
                    _jit_internal.get_overload_no_implementation_error_message(
                        "method", item.__func__
                    )
                )

            names = [name + "__" + str(i) for i in range(len(method_overloads))]
            overloads[item] = list(zip(names, method_overloads))

    return overloads


def get_overload_name_mapping(overload_info):
    # Same format as __overloads__
    # original function => [overload names]
    overload_name_mappings: dict[str, list[str]] = {}
    for orig_fn, overloads in overload_info.items():
        original_name = orig_fn.__name__
        if original_name not in overload_name_mappings:
            overload_name_mappings[original_name] = []

        for overload_name, _ in overloads:
            overload_name_mappings[original_name].append(overload_name)
    return overload_name_mappings


def _check_no_signature(func) -> None:
    signature = torch.jit.annotations.get_signature(
        func, None, fake_range(), inspect.ismethod(func)
    )
    if signature is None:
        qual_name = _jit_internal._qualified_name(func)
        raise RuntimeError(
            f"Must explicitly add type annotations to overloaded functions: {qual_name}"
        )


def make_stubs_for_overloads(overload_info):
    overload_stubs = []
    for orig_fn, overloads in overload_info.items():
        orig_ast = get_jit_def(
            orig_fn, orig_fn.__name__, self_name="RecursiveScriptModule"
        )
        for overload_name, overload_fn in overloads:
            _check_no_signature(overload_fn)
            over_ast = get_jit_def(
                overload_fn, overload_fn.__name__, self_name="RecursiveScriptModule"
            )
            new_ast = torch._C._replace_overloaded_method_decl(
                over_ast.decl(), orig_ast, overload_name
            )
            _rcb = _jit_internal.createResolutionCallbackFromClosure(orig_fn)
            overload_stubs.append(ScriptMethodStub(_rcb, new_ast, overload_fn))
    return overload_stubs


def check_module_initialized(mod) -> None:
    if not isinstance(mod, torch.nn.Module):
        raise AssertionError(f"Expected torch.nn.Module, got {type(mod)}")
    if not hasattr(mod, "_parameters"):
        raise RuntimeError(
            f"'{torch.typename(type(mod))}' has not been initialized, did you forget to call 'super()'?"
        )

    # This is to avoid importing torch.distributed.nn
    if not hasattr(mod, "remote_parameters"):
        for name, param in mod._parameters.items():
            if param is not None and torch.nn.parameter.is_lazy(param):
                raise RuntimeError(
                    f"'{torch.typename(type(mod))}' has uninitialized parameters {name}. Did you forget to run a forward pass?"
                )
        for name, buf in mod._buffers.items():
            if buf is not None and torch.nn.parameter.is_lazy(buf):
                raise RuntimeError(
                    f"'{torch.typename(type(mod))}' has uninitialized buffers {name}. Did you forget to run a forward pass?"
                )


def infer_methods_to_compile(nn_module):
    """Implement the default rules for which methods should act as starting points for compilation.

    (TODO add a link when the rules are published).
    """
    check_module_initialized(nn_module)
    ignored_properties = jit_ignored_properties(nn_module)

    methods: list[str] = []
    if hasattr(nn_module, "forward") and not _jit_internal.is_ignored_fn(
        nn_module.forward
    ):
        forward_func = getattr(nn_module.forward, "__func__", None)
        module_forward = getattr(torch.nn.Module, "forward", None)
        if forward_func != module_forward:
            methods = ["forward"]

    exported = []
    for name in dir(nn_module):
        if name in ignored_properties:
            continue
        item = getattr(nn_module, name, None)
        if (
            _jit_internal.get_torchscript_modifier(item)
            is _jit_internal.FunctionModifiers.EXPORT
        ):
            exported.append(name)

    methods = methods + exported

    overload_name_mappings = dict(getattr(nn_module, "__overloads__", {}))
    overload_info = get_overload_annotations(nn_module, ignored_properties)
    overload_name_mappings.update(get_overload_name_mapping(overload_info))
    overload_stubs = make_stubs_for_overloads(overload_info)

    nn_module.__overloads__ = overload_name_mappings

    # we shouldn't directly compile overloaded methods, just its overloads
    def ignore_overloaded(method_name):
        return method_name not in overload_name_mappings

    filtered_methods = filter(ignore_overloaded, methods)

    # Unique the methods. We don't want to use a set to store the methods because it
    # introduces non-determinism to compile order.
    uniquer: set[str] = set()
    uniqued_methods = []
    for name in filtered_methods:
        if name in uniquer:
            continue
        uniqued_methods.append(name)
        uniquer.add(name)

    stubs = [make_stub_from_method(nn_module, method) for method in uniqued_methods]
    return overload_stubs + stubs


def get_hook_stubs(nn_module):
    """Return forward hook and pre_hook ScriptModuleStubs."""
    check_module_initialized(nn_module)
    hook_map: dict = {}

    hook_stubs = []
    for hook in nn_module._forward_hooks.values():
        if hook.__name__ in hook_map:
            if id(hook) != id(hook_map[hook.__name__]):
                raise RuntimeError(
                    f"Hook '{hook.__name__}' on {type(nn_module).__name__} "
                    "has at least two different python definitions."
                    " Please use unique names for all hooks."
                )
        else:
            hook_map[hook.__name__] = hook
        hook_stubs.append(make_stub(hook, hook.__name__))

    pre_hook_stubs = []
    for pre_hook in nn_module._forward_pre_hooks.values():
        if pre_hook.__name__ in hook_map:
            if id(pre_hook) != id(hook_map[pre_hook.__name__]):
                raise RuntimeError(
                    f"Pre-hook '{pre_hook.__name__}' on {type(nn_module).__name__} "
                    "has at least two different python definitions."
                    " Please use unique names for all hooks."
                )
        else:
            hook_map[pre_hook.__name__] = pre_hook
        pre_hook_stubs.append(make_stub(pre_hook, pre_hook.__name__))

    return hook_stubs, pre_hook_stubs


def get_property_stubs(nn_module):
    """Create property stubs for the properties of the module by creating method stubs for the getter and setter."""
    module_ty = type(nn_module)
    properties_asts = get_class_properties(module_ty, self_name="RecursiveScriptModule")
    rcbs = {}

    for name in dir(module_ty):
        item = getattr(module_ty, name, None)
        if isinstance(item, property):
            if not item.fget:
                raise RuntimeError(
                    f"Property {name} of {nn_module.__name__} must have a getter"
                )

            rcbs[name] = _jit_internal.createResolutionCallbackFromClosure(item.fget)

    stubs = [PropertyStub(rcbs[ast.name().name], ast) for ast in properties_asts]
    return stubs


def interface_script(mod_interface, nn_module):
    """
