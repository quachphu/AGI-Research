        to get determinism.  To seed all GPUs, use :func:`manual_seed_all`.
    """
    seed = int(seed)

    def cb():
        idx = current_device()
        default_generator = torch.cuda.default_generators[idx]
        default_generator.manual_seed(seed)

    _lazy_call(cb, seed=True)


def manual_seed_all(seed: int) -> None:
    r"""Set the seed for generating random numbers on all GPUs.

    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.

    Args:
        seed (int): The desired seed.
    """
    seed = int(seed)

    def cb():
        for i in range(device_count()):
            default_generator = torch.cuda.default_generators[i]
            default_generator.manual_seed(seed)

    _lazy_call(cb, seed_all=True)


def seed() -> None:
    r"""Set the seed for generating random numbers to a random number for the current GPU.

    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.

    .. warning::
        If you are working with a multi-GPU model, this function will only initialize
        the seed on one GPU.  To initialize all GPUs, use :func:`seed_all`.
    """

    def cb():
        idx = current_device()
        default_generator = torch.cuda.default_generators[idx]
        default_generator.seed()

    _lazy_call(cb)


def seed_all() -> None:
    r"""Set the seed for generating random numbers to a random number on all GPUs.

    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.
    """

    def cb():
        random_seed = 0
        seeded = False
        for i in range(device_count()):
            default_generator = torch.cuda.default_generators[i]
            if not seeded:
                default_generator.seed()
                random_seed = default_generator.initial_seed()
                seeded = True
            else:
                default_generator.manual_seed(random_seed)

    _lazy_call(cb)


def initial_seed() -> int:
    r"""Return the current random seed of the current GPU.

    .. warning::
        This function eagerly initializes CUDA.
    """
    _lazy_init()
    idx = current_device()
    default_generator = torch.cuda.default_generators[idx]
    return default_generator.initial_seed()
<|endoftext|># mypy: allow-untyped-defs
import contextlib

from . import check_error, cudart


__all__ = ["start", "stop", "profile"]

DEFAULT_FLAGS = [
    "gpustarttimestamp",
    "gpuendtimestamp",
    "gridsize3d",
    "threadblocksize",
    "streamid",
    "enableonstart 0",
    "conckerneltrace",
]


def start():
    r"""Starts cuda profiler data collection.

    .. warning::
        Raises CudaError in case of it is unable to start the profiler.
    """
    check_error(cudart().cudaProfilerStart())


def stop():
    r"""Stops cuda profiler data collection.

    .. warning::
        Raises CudaError in case of it is unable to stop the profiler.
    """
    check_error(cudart().cudaProfilerStop())


@contextlib.contextmanager
def profile():
    """
    Enable profiling.

    Context Manager to enabling profile collection by the active profiling tool from CUDA backend.
    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> import torch
        >>> model = torch.nn.Linear(20, 30).cuda()
        >>> inputs = torch.randn(128, 20).cuda()
        >>> with torch.cuda.profiler.profile() as prof:
        ...     model(inputs)
    """
    try:
        start()
        yield
    finally:
        stop()
<|endoftext|>"""Annotate CUDA graph kernel nodes during capture.

During CUDA graph capture, ``mark_kernels`` records the current capture
frontier and the direct dependents already attached to that frontier.
On scope exit it walks only the newly added dependent edges to find the
nodes created within the scope. Each kernel, memcpy, or memset node found
is annotated by its ``toolsId`` so it can later be matched to profiler
trace events.

``mark_kernels`` now snapshots capture state from whatever stream is
current on scope entry, so that stream must already be participating in
the capture. ``mark_stream`` handles this by starting ``mark_kernels``
before switching to the target stream.

The annotations can be pickled and later merged into a Chrome profiler
trace using ``torch.cuda._annotate_cuda_graph_trace``.

Requires ``cuda.bindings`` package and a CUDA driver that supports
``cudaGraphNodeGetToolsId`` (CUDA >= 13.1 or appropriate cuda-compat).
When unavailable, ``mark_kernels`` silently becomes a no-op.

Annotation recording is enabled per capture via ``torch.cuda.graph``'s
``enable_annotations`` argument, which also resolves and remaps the recorded
annotations to the exec graph automatically on context exit.

Usage during capture::

    from torch.cuda._graph_annotations import mark_kernels, get_kernel_annotations

    with torch.cuda.graph(graph, enable_annotations=True):
        with mark_kernels("phase_A"):
            y = workload_a(x)
        with mark_kernels("phase_B"):
            z = workload_b(y)

    annotations = get_kernel_annotations()

When you need to drive this outside the context manager's automatic path,
``resolve_and_remap(graph)`` is shorthand for ``resolve_pending_annotations()``
followed by ``remap_to_exec_graph(graph)``; call those directly for finer
control (e.g. resolving once before remapping several graphs).
"""

from __future__ import annotations

import importlib.metadata
import threading
import warnings
from collections import defaultdict
from contextlib import contextmanager
from logging import getLogger
from typing import Any, NamedTuple, TYPE_CHECKING, TypeAlias


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

import torch
from torch.cuda._utils import (
    _check_cuda_bindings,
    _check_cuda_bindings_driver,
    _HAS_CUDA_BINDINGS,
)


try:
    from cuda.bindings import (  # pyrefly: ignore[missing-import]
        driver as _cuda_driver,
        runtime as _cuda_runtime,
    )
except ImportError:
    _cuda_driver = None  # type: ignore[assignment]
    _cuda_runtime = None  # type: ignore[assignment]


logger = getLogger(__name__)


_CaptureState: TypeAlias = tuple[Any, list[Any]]
_ExistingDirectDependents: TypeAlias = dict[int, set[int]]


# Tri-state: None = not probed, True = available, False = unavailable.
# Deferred to first use to avoid premature CUDA initialization.
_tools_id_available: bool | None = None

# Whether annotation recording is active. Scoped to a ``torch.cuda.graph``
# capture: set by the context manager on entry (from its ``enable_annotations``
# argument) and cleared on exit. When False, mark_kernels/mark_stream and the
# capture-id stamp are no-ops. Module-level (not thread-local) because capture
# can span threads (e.g. autograd).
_annotations_enabled: bool = False


# Id of the top-level capture graph of the active capture. A conditional node's body is
# captured into a separate cudaGraph_t (see CUDAGraph::begin_capture_to_conditional_node),
# and its node ids live in that graph's id space, which remap_to_exec_graph never rekeys --
# so mark_kernels compares against this to detect a scope inside a body.
_capture_root_graph_id: int | None = None


def _set_annotations_enabled(enabled: bool) -> None:
    """Set whether annotation recording is active. Used by ``torch.cuda.graph``
    to scope annotations to a capture; not a public API."""
    global _annotations_enabled, _capture_root_graph_id
    _annotations_enabled = enabled
    if not enabled:
        _capture_root_graph_id = None


def _graph_id(graph: Any) -> int:
    """The driver's unique id for a ``cudaGraph_t``.

    Identifies a graph across its whole lifetime, unlike the handle itself: that is a
    pointer, so a graph created after another is destroyed can reuse the value and
    compare equal to it."""
    return _check_cuda_bindings(
        _cuda_runtime.cudaGraphGetId(graph)  # pyrefly: ignore[missing-attribute]
    )


def maybe_stamp_capture_root(stream: Any) -> None:
    """Record the top-level capture graph of the capture just begun on ``stream``.

    Called by ``torch.cuda.graph`` right after ``capture_begin``, while ``stream`` is
    still capturing into the top-level graph. Not a public API."""
    global _capture_root_graph_id
    _capture_root_graph_id = None
    if not _annotations_enabled or _is_tools_id_unavailable():
        return
    stream_handle = _cuda_runtime.cudaStream_t(  # pyrefly: ignore[missing-attribute]
        init_value=stream.cuda_stream
    )
    state = _get_capture_state(stream_handle)
    if state is not None:
        _capture_root_graph_id = _graph_id(state[0])


def _probe_tools_id() -> bool:
    """Probe whether cudaGraphNodeGetToolsId is supported by the driver.

    Calls with a null node: cudaErrorInvalidValue means the API exists
    in the driver (good), cudaErrorCallRequiresNewerDriver means it
    does not (bad).
    """
    if not hasattr(_cuda_runtime, "cudaGraphNodeGetToolsId"):
        # API is missing from cuda-bindings - likely version too old
        cuda_bindings_version = importlib.metadata.version("cuda-bindings")

        logger.warning(
            "cudaGraphNodeGetToolsId API not found in cuda-bindings. "
            "Current version: %s, required: >= 13.1.0. "
            "CUDA graph kernel annotations will be disabled. "
            "To enable annotations, upgrade cuda-bindings: "
            "pip install --upgrade cuda-bindings",
            cuda_bindings_version,
        )
        return False
    err, *_ = _cuda_runtime.cudaGraphNodeGetToolsId(
        0
    )  # pyrefly: ignore[missing-attribute]
    if (
        err
        == _cuda_runtime.cudaError_t.cudaErrorCallRequiresNewerDriver  # pyrefly: ignore[missing-attribute]
    ):
        logger.info(
            "cudaGraphNodeGetToolsId requires a newer driver "
            "(missing cuda-compat?); "
            "CUDA graph kernel annotations will be disabled"
        )
        return False
    return True


def _is_tools_id_unavailable() -> bool:
    """Return True if cudaGraphNodeGetToolsId is not usable."""
    global _tools_id_available
    if not _HAS_CUDA_BINDINGS:
        return True
    if _tools_id_available is not None:
        return not _tools_id_available
    _tools_id_available = _probe_tools_id()
    return not _tools_id_available


def is_available() -> bool:
    r"""is_available() -> bool

    Return whether CUDA graph annotation recording is supported.

    Requires a CUDA device, the ``cuda-bindings`` package, and a driver
    that supports ``cudaGraphNodeGetToolsId`` (CUDA >= 13.1 or an
    equivalent cuda-compat package). When this returns ``False``,
    :func:`mark_kernels` is a silent no-op and no annotations are
    recorded.

    The first call may probe the CUDA driver; the result is cached.
    """
    return torch.cuda.is_available() and not _is_tools_id_unavailable()


def _get_capture_state(stream: Any) -> _CaptureState | None:
    """Return ``(graph, frontier)`` for an active capture, else ``None``."""
    status, _id, graph, _deps, _edge_data, _num_deps = _check_cuda_bindings(
        _cuda_runtime.cudaStreamGetCaptureInfo(  # pyrefly: ignore[missing-attribute]
            stream
        )
    )
    if (
        status
        != _cuda_runtime.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive  # pyrefly: ignore[missing-attribute]
    ):
        return None
    return graph, list(_deps[:_num_deps])


def _get_root_nodes(graph: Any) -> list[Any]:
    """Return the current root nodes for the graph."""
    _, num_roots = _check_cuda_bindings(
        _cuda_runtime.cudaGraphGetRootNodes(  # pyrefly: ignore[missing-attribute]
            graph
        )
    )
    if num_roots == 0:
        return []
    roots, num_roots = _check_cuda_bindings(
        _cuda_runtime.cudaGraphGetRootNodes(  # pyrefly: ignore[missing-attribute]
            graph, pNumRootNodes=num_roots
        )
    )
    return list(roots[:num_roots])


def _get_dependent_nodes(node: Any) -> list[Any]:
    """Return the direct dependents of a graph node."""
    _, _, num_dependents = _check_cuda_bindings(
        _cuda_runtime.cudaGraphNodeGetDependentNodes(  # pyrefly: ignore[missing-attribute]
            node
        )
    )
    if num_dependents == 0:
        return []
    dependents, _edge_data, num_dependents = _check_cuda_bindings(
        _cuda_runtime.cudaGraphNodeGetDependentNodes(  # pyrefly: ignore[missing-attribute]
            node, pNumDependentNodes=num_dependents
        )
    )
    return list(dependents[:num_dependents])


def _get_node_type(node: Any) -> Any:
    """Return graph node type without tripping runtime bugs on newer node kinds.

    The runtime ``cudaGraphNodeGetType`` can return ``cudaErrorUnknown`` (999)
    for valid nodes whose driver type is ``CU_GRAPH_NODE_TYPE_BATCH_MEM_OP``.
    Query via the driver API instead.
    """
    return _check_cuda_bindings_driver(
        _cuda_driver.cuGraphNodeGetType(node)  # pyrefly: ignore[missing-attribute]
    )


def _collect_descendants(
    start_nodes: list[Any],
    *,
    existing_direct_dependents: _ExistingDirectDependents | None = None,
    include_start_nodes: bool = False,
) -> dict[int, Any]:
    """Walk dependent edges starting at ``start_nodes``.

    ``existing_direct_dependents`` maps each node in ``start_nodes`` to
    the direct dependent node keys that were already present at scope
    entry. Those edges are skipped so the traversal only follows nodes
    added after scope entry.
    """
    existing_direct_dependents = existing_direct_dependents or {}
    seen = {int(node) for node in start_nodes}
    descendants: dict[int, Any] = {}
    stack = list(start_nodes)

    if include_start_nodes:
        for node in start_nodes:
            descendants[int(node)] = node

    while stack:
        node = stack.pop()
        old_dependents = existing_direct_dependents.get(int(node), set())
        for dependent in _get_dependent_nodes(node):
            dependent_key = int(dependent)
            if dependent_key in old_dependents or dependent_key in seen:
                continue
            seen.add(dependent_key)
            descendants[dependent_key] = dependent
            stack.append(dependent)

    return descendants


# toolsId -> list of annotation objects.
_kernel_annotations: defaultdict[int, list[Any]] = defaultdict(list)

# Node types we annotate (kernels, memcpys, memsets, batch mem ops, event
# record/wait nodes, and host nodes), as driver node-type enums. Event and host
# nodes are stream-ordered but do not occupy the device; annotating them lets the
# profiler place their spans on the intended stream lane. Initialized lazily to
# avoid touching cuda.bindings at import time.
_ANNOTATABLE_TYPES: set[Any] | None = None


def _get_annotatable_types() -> set[Any]:
    global _ANNOTATABLE_TYPES
    if _ANNOTATABLE_TYPES is None:
        node_types = _cuda_driver.CUgraphNodeType  # pyrefly: ignore[missing-attribute]
        _ANNOTATABLE_TYPES = {
            node_types.CU_GRAPH_NODE_TYPE_KERNEL,
            node_types.CU_GRAPH_NODE_TYPE_MEMCPY,
            node_types.CU_GRAPH_NODE_TYPE_MEMSET,
            node_types.CU_GRAPH_NODE_TYPE_BATCH_MEM_OP,
            node_types.CU_GRAPH_NODE_TYPE_EVENT_RECORD,
            node_types.CU_GRAPH_NODE_TYPE_WAIT_EVENT,
            node_types.CU_GRAPH_NODE_TYPE_HOST,
        }
    return _ANNOTATABLE_TYPES


# Node types whose work lives in a separate cudaGraph_t (child graphs, conditional
# bodies). The dependent-edge walk does not descend into such a node, and the nodes
# inside are numbered in the body graph's id space, which remap_to_exec_graph never
# rekeys -- so annotations there would be silently lost. See mark_kernels. Initialized
# lazily, like _ANNOTATABLE_TYPES above: _cuda_driver is None when cuda.bindings is
# absent, so reading the enum at import time would break `import torch`.
_NESTED_GRAPH_TYPES: set[Any] | None = None


def _get_nested_graph_types() -> set[Any]:
    global _NESTED_GRAPH_TYPES
    if _NESTED_GRAPH_TYPES is None:
        node_types = _cuda_driver.CUgraphNodeType  # pyrefly: ignore[missing-attribute]
        _NESTED_GRAPH_TYPES = {
            node_types.CU_GRAPH_NODE_TYPE_GRAPH,
            node_types.CU_GRAPH_NODE_TYPE_CONDITIONAL,
        }
    return _NESTED_GRAPH_TYPES


# Pending scopes: (annotation, toolsIds discovered for the scope).
#
# Notably, this is NOT the current dynamic scope; instead, we fill this in
# after we exit each mark_kernels region, and this holds the FULL extra
# CUDA graph node attribution, steadily growing as we execute.
#
# Scopes are recorded as we exit a `mark_kernels`.  Later, when we resolve
# all of the annotations, we do merges on the annotations where
# first-recorded-wins (it works: you end up preferring inner scopes over
# outer scopes, what you'd expect).  For example:
#
#     with mark_kernels({"name": "outer", "color": "red"}):
#         with mark_kernels({"name": "inner"}):
#             y = x + 1  # toolsId: 101
#         z = y * 2      # toolsId: 102
#
# where the add captures kernel node with toolsId 101 and the mul 102. After
# executing ALL of this code, the _pending_scopes would be (inner exits first,
# so it is recorded first):
#
#     ({"name": "inner"}, [101])
#     ({"name": "outer", "color": "red"}, [101, 102])
#
# toolsId 101 appears in both lists, and so the first-recorded-wins rule then
# means we merge all the dicts to {"name": "inner", "color": "red"}.
_pending_scopes: list[tuple[Any, list[int]]] = []


# TLS slot carrying the current annotation region: the annotations of every
# open backward-annotating ``mark_kernels`` scope collapsed into one dict
# (inner scopes win common keys). It lives in ThreadLocalPythonObjects rather
# than a ContextVar or threading.local because that is the one Python-writable
# TLS that ``at::ThreadLocalState`` snapshots into autograd engine worker and
# device threads, so a backward bracket can re-establish the region for nodes
# created while it executes (higher-order grad, checkpoint recomputation) even
# across engine thread hops. Region dicts are never mutated after creation, so
# holding a reference is a snapshot.
_REGION_TLS_KEY = "cuda_graph_annotation_region"


def _current_region() -> dict[str, Any] | None:
    if torch._C._is_key_in_tls(_REGION_TLS_KEY):
        return torch._C._get_obj_in_tls(_REGION_TLS_KEY)
    return None


def _enter_region(collapsed: dict[str, Any]) -> dict[str, Any] | None:
    """Install ``collapsed`` as the current region; return the previous one."""
    prev = _current_region()
    torch._C._stash_obj_in_tls(_REGION_TLS_KEY, collapsed)
    return prev


def _exit_region(prev: dict[str, Any] | None) -> None:
    if prev is None:
        torch._C._remove_obj_from_tls(_REGION_TLS_KEY)
    else:
        torch._C._stash_obj_in_tls(_REGION_TLS_KEY, prev)


# node.metadata marker recording that a node's backward hook pair is already
# installed, so redundant firings of ``_freeze_region_hook`` (one per live
# push, e.g. nested scopes) attach only once. An identity sentinel so no
# other metadata user can collide with it.
_HOOKED_KEY: Any = object()

# Bumped by clear_kernel_annotations. Node hooks record only while the
# generation they were created under is current, so clearing revokes
# recording from every scope opened before the clear (the hooks stay on
# the graph but become inert) without mutating live autograd graphs.
_annotation_generation: int = 0


class _KernelScope(NamedTuple):
    """Capture-frontier snapshot taken at scope entry, consumed at scope exit."""

    graph: Any
    frontier: list[Any]
    entry_root_keys: set[int] | None
    entry_direct_dependents: _ExistingDirectDependents


def _begin_kernel_scope() -> _KernelScope | None:
    """Snapshot the current stream's capture frontier; ``None`` if not capturing."""
    stream = _cuda_runtime.cudaStream_t(  # pyrefly: ignore[missing-attribute]
        init_value=torch.cuda.current_stream().cuda_stream
    )
    capture_state = _get_capture_state(stream)
    if capture_state is None:
        return None
    graph, frontier = capture_state

    entry_root_keys: set[int] | None = None
    entry_direct_dependents = {
        int(node): {int(dep) for dep in _get_dependent_nodes(node)} for node in frontier
    }
    if not frontier:
        entry_root_keys = {int(node) for node in _get_root_nodes(graph)}
    return _KernelScope(graph, frontier, entry_root_keys, entry_direct_dependents)


def _end_kernel_scope(scope: _KernelScope) -> list[int]:
    """Walk nodes captured since ``scope`` was begun; return their toolsIds."""
    if scope.frontier:
        scope_nodes = _collect_descendants(
            scope.frontier,
            existing_direct_dependents=scope.entry_direct_dependents,
        )
    else:
        new_roots = [
            node
            for node in _get_root_nodes(scope.graph)
            if int(node) not in (scope.entry_root_keys or set())
        ]
        scope_nodes = _collect_descendants(new_roots, include_start_nodes=True)

    annotatable = _get_annotatable_types()
    nested = _get_nested_graph_types()
    nested_seen: set[str] = set()
    tools_ids: list[int] = []
    for node in scope_nodes.values():
        node_type = _get_node_type(node)
        if node_type in nested:
            nested_seen.add(node_type.name)
        if node_type not in annotatable:
            continue
        tools_ids.append(
            _check_cuda_bindings(
                _cuda_runtime.cudaGraphNodeGetToolsId(  # pyrefly: ignore[missing-attribute]
                    node
                )
            )
        )

    if nested_seen:
        # The annotations recorded above are still correct -- the exec graph preserves
        # top-level node ids -- they just do not cover the nested body.
        warnings.warn(
            f"mark_kernels: this scope contains {sorted(nested_seen)} nodes; the work "
            "inside them is in a separate cudaGraph_t that this walk does not descend "
            "into, so it is left unannotated",
            stacklevel=4,
        )
    return tools_ids


class _BracketState(threading.local):
    """Per-thread stack of open backward brackets for one hook pair.

    Thread-local so concurrent executions of a retained node (separate
    graph tasks on different engine threads) cannot interleave each
    other's scope snapshots; within one thread pre/post pair up LIFO.

    If the node throws, its posthook never runs (the engine's
    call_function has no try/finally around fn) and the prehook's entry
    is left on the stack. That is harmless. Say a retained node runs
    three times on one thread and the second run throws:

        run 1: prehook push [e1]; posthook pops e1     -> []
        run 2: prehook push [e2]; node throws, no pop  -> [e2]
        run 3: prehook push [e2, e3]; posthook pops e3 -> [e2]

    Pops always take the top, so the leaked e2 sits below all later
    entries and every subsequent run still pops its own. The only loss
    is e2 itself: that run's kernels go unrecorded, which is moot since
    its backward failed. (The region/creation-hook TLS the prehook set
    is not leaked -- the engine task's ThreadLocalStateGuard restores
    those on unwind.)
    """

    def __init__(self) -> None:
        # Entries are (scope, prev_region, merged_region) or _SKIPPED.
        self.stack: list[Any] = []


# Bracket-stack marker for "prehook did nothing": distinct from an entry with
# scope=None, which means the region and creation hook were installed but no
# capture was active (the posthook must still pop them).
_SKIPPED: Any = object()


def _freeze_region_hook(node: Any, generation: int) -> None:
    """Node creation hook: freeze the current region into ``node``'s bracket.

    Reads the region TLS slot and, if a region is open, installs the node's
    single backward hook pair closing over that snapshot. Every backward-
    annotating ``mark_kernels`` scope pushes one of these (and so does an
    executing hooked node, to cover nodes created during its backward), so
    a node created under nested scopes sees several identical firings; the
    ``node.metadata`` marker makes only the first attach.

    AccumulateGrad is always excluded: a leaf's node is created once and
    cached (possibly during warmup), so which scope would own it is an
    accident of first use, and its work (the ``.grad`` accumulation)
    belongs to the leaf rather than to any forward region. If accumulation
    ever needs annotation it should get a dedicated leaf-accumulation
    region, not first-use assignment.
    """
    if isinstance(node, torch._C._functions.AccumulateGrad):
        return
    if _HOOKED_KEY in node.metadata:
        return
    frozen = _current_region()
    if frozen is None:
        return
    node.metadata[_HOOKED_KEY] = True
    _attach_backward_hooks(node, frozen, generation)


def _attach_backward_hooks(node: Any, frozen: dict[str, Any], generation: int) -> None:
    """Register one pre/post hook pair bracketing ``node``'s backward execution.

    ``frozen`` is the region that was current when the node was created:
    the annotations of every scope then open, already collapsed inner-wins.
    The pair brackets the node's execution the same way ``mark_kernels``
    brackets forward work: snapshot the capture frontier before, collect
    the new nodes after. It runs on the autograd engine thread executing
    the node, which during a whole-graph capture participates in the
    capture; when backward runs outside a capture the prehook sees no
    active capture and the pair is a no-op.

    Executing the node is semantically re-entering its frozen region, so
    the prehook merges ``frozen`` over the live region (a ``mark_kernels``
    enclosing the whole ``backward()`` call, if any) and installs the
    result while the node runs. That gives nodes created during its
    backward (``create_graph=True``, checkpoint recomputation) the forward
    region's ownership -- a creation hook is pushed alongside so they get
    hooked even when this backward runs eagerly and no scope's hook is
    live -- and it puts a ``mark_kernels`` opened inside the execution
    dynamically inner, so it outranks ``frozen``, which outranks the
    enclosing scope. The posthook records the merged region once; scope
    completion order then yields exactly that precedence at resolve.

    The pops are exception-safe without a finally: each engine task runs
    under an ``at::ThreadLocalStateGuard`` that restores both the
    creation-hook TLS and the region slot even when the node throws.

    ``generation`` is the annotation generation the owning scope was opened
    under; the bracket only records (and only propagates to created nodes)
    while it is current, so ``clear_kernel_annotations`` makes stale hooks
    inert.
    """
    state = _BracketState()

    def creation_hook(child: Any) -> None:
        _freeze_region_hook(child, generation)

    def prehook(_grad_outputs: Any) -> None:
        if generation != _annotation_generation or _is_tools_id_unavailable():
            state.stack.append(_SKIPPED)
            return
        # Re-establish ownership even when this backward is not itself
        # captured: kernels of nodes created here may be captured by a
        # later (e.g. second-order) annotated capture.
        merged = {**(_current_region() or {}), **frozen}
        prev = _enter_region(merged)
        torch._C._autograd._push_node_creation_hook(creation_hook)
        scope = _begin_kernel_scope() if _annotations_enabled else None
        state.stack.append((scope, prev, merged))

    def posthook(_grad_inputs: Any, _grad_outputs: Any) -> None:
        if not state.stack:
            return
        entry = state.stack.pop()
        if entry is _SKIPPED:
            return
        scope, prev, merged = entry
        torch._C._autograd._pop_node_creation_hook()
        _exit_region(prev)
        if scope is None:
            return
        tools_ids = _end_kernel_scope(scope)
        if tools_ids:
            tagged = {**merged, "autograd_phase": "backward"}
            _pending_scopes.append((tagged, tools_ids))

    node.register_prehook(prehook)
    node.register_hook(posthook)


@contextmanager  # type: ignore[arg-type]
def mark_kernels(annotation: str | dict[str, Any], *, backward: bool = True):
    r"""mark_kernels(annotation, *, backward=True)

    Context manager that annotates GPU work captured within its scope.

    Must be used inside an active :class:`torch.cuda.graph` capture with
    ``enable_annotations=True``. Every kernel, memcpy, and memset node the
    capture adds within the scope is tagged with :attr:`annotation`. Outside
    a capture, with annotations disabled, or when :func:`is_available` is
    ``False``, the context manager is a no-op.

    When scopes overlap on the same node (e.g. nested scopes), their
    annotation dicts are merged key-by-key with the inner scope winning
    common keys.

    By default backward work is annotated too: autograd nodes created by
    forward operations inside the scope get hooks (via
    :class:`torch.autograd.graph.node_creation_hook`) that bracket their
    backward execution, so when the backward pass is itself captured --
    in the same capture as the forward or in a later one -- its kernels
    are tagged with the same annotation, plus an ``"autograd_phase":
    "backward"`` key marking them as backward work (``"autograd_phase"``
    is therefore reserved: backward annotation overwrites it). When
    backward runs outside a capture the hooks are a no-op. Ownership
    extends to higher-order gradients: nodes created while a hooked node
    executes (``create_graph=True``, checkpoint recomputation) inherit its
    annotations, so a later grad-of-grad capture is attributed too.
    ``AccumulateGrad`` nodes are never annotated: a leaf's node is created
    once and cached, so scope ownership would be an accident of first use.
    Pass ``backward=False`` to annotate only the forward work, e.g. when a
    wrapper implements its own backward attribution. The keyword's
    presence also serves as the feature probe for that native backward
    support: ``"backward" in inspect.signature(mark_kernels).parameters``.

    Implementation: on entry, records the current stream's capture frontier
    and its existing direct dependents; on scope exit, walks only the
    dependent nodes added since entry (falling back to newly created graph
    roots when the scope is the first captured work).

    Args:
        annotation (str or dict): Metadata to attach to each captured node.
            A string ``s`` is recorded as ``{"name": s}``. Dict values must
            be picklable. The key ``"name"`` names the region in trace
            tooling; ``"stream"`` is reserved for stream-lane assignment.
        backward (bool): Whether to also annotate the backward kernels of
            autograd nodes created inside the scope. Default: ``True``.

    .. note::
        The nodes to annotate must be reachable from the capture frontier of
        the stream that is current on scope entry. Work on a different
        already-capturing stream must be synchronized with the current
        stream first.

    .. note::
        Child-graph and conditional nodes have bodies in a separate
        ``cudaGraph_t`` that this walk does not descend into, so their work is
        left unannotated and a warning is issued. Descending is possible
        (``cudaGraphNodeGetParams`` exposes the body graphs), but would not be
        enough on its own: a body's nodes are numbered in that graph's id space
        and are renumbered again when the exec graph inlines them, and nothing
        exposes that renumbering, so :func:`remap_to_exec_graph` could not key
        the annotations to what a profiler reports. For the same reason a scope
        *inside* a conditional body (``torch.cond`` / ``torch.while_loop``)
        records nothing at all.

    .. warning::
        This API is in prototype and may change in future releases.

    Example::

        >>> # xdoctest: +SKIP("requires cuda-bindings and driver >= 13.1")
        >>> g = torch.cuda.CUDAGraph()
        >>> x = torch.randn(8, device="cuda")
        >>> with torch.cuda.graph(g, enable_annotations=True):
        ...     with torch.cuda.graph_annotations.mark_kernels("phase_A"):
        ...         y = x + 1
    """
    if not _annotations_enabled or _is_tools_id_unavailable():
        yield
        return

    if isinstance(annotation, str):
        annotation = {"name": annotation}

    scope = _begin_kernel_scope()
    if scope is None:
        yield
        return

    if (
        _capture_root_graph_id is not None
        and _graph_id(scope.graph) != _capture_root_graph_id
    ):
        # Inside a conditional node's body: torch.cond / torch.while_loop capture into a
        # separate cudaGraph_t. Its node ids are in that graph's id space and are
        # renumbered again in the exec graph, so anything recorded here would be a key
        # that matches nothing in a trace. Record nothing rather than dead keys.
        warnings.warn(
            "mark_kernels: this scope is inside a CUDA graph conditional-node body "
            "(torch.cond / torch.while_loop), which is captured into a separate "
            "cudaGraph_t whose node ids are never remapped to the exec graph; "
            "nothing is annotated for it",
            stacklevel=3,
        )
        yield
        return

    generation = _annotation_generation

    def creation_hook(node: Any) -> None:
        _freeze_region_hook(node, generation)

    # Enter the region even when backward=False: the region must reflect the
    # full dynamic scope so that nodes hooked by a nested backward=True scope
    # (or by an executing hooked node) freeze this annotation too.
    prev = _enter_region({**(_current_region() or {}), **annotation})
    try:
        if backward:
            with torch.autograd.graph.node_creation_hook(creation_hook):
                yield
        else:
            yield
    finally:
        _exit_region(prev)

    tools_ids = _end_kernel_scope(scope)
    if tools_ids:
        _pending_scopes.append((annotation, tools_ids))


def maybe_stamp_capture_graph_id(torch_cuda_graph: torch.cuda.CUDAGraph) -> None:
    """Record the captured graph's id on the graph object for later remap.

    Called from ``capture_end`` in the window after capture ends and before the
    graph is finalized, where the template ``cudaGraph_t`` is live for both
    ``keep_graph`` modes (and reachable via ``raw_cuda_graph``). ``remap_to_exec_graph``
    later matches this graph's annotations to its exec id via this stamp, without
    relying on call ordering. No-op if annotations are disabled or
    ``cudaGraphNodeGetToolsId`` is unavailable. Harmless when no ``mark_kernels``
    regions run: the annotation map stays empty so resolve and remap are no-ops.
    """
    if not _annotations_enabled or _is_tools_id_unavailable():
        return
    # Past the _is_tools_id_unavailable() guard cuda-bindings is present and the
    # driver supports the toolsId API (same version gate as cudaGraphGetId), so
    # any error here is unexpected: error-check and let it raise. cudaGraphGetId
    # accepts the raw cudaGraph_t handle (int) directly.
    torch_cuda_graph._capture_graph_id = _check_cuda_bindings(
        _cuda_runtime.cudaGraphGetId(  # pyrefly: ignore[missing-attribute]
            torch_cuda_graph.raw_cuda_graph()
        )
    )
    # Fresh capture: annotations are keyed by this capture id until remapped.
    torch_cuda_graph._remapped_exec_id = None


def resolve_pending_annotations() -> None:
    """Resolve pending scope toolsIds into kernel annotations."""
    if not _pending_scopes:
        return

    try:
        per_tools_id: defaultdict[int, list[Any]] = defaultdict(list)
        for annotation, tools_ids in _pending_scopes:
            for tools_id in tools_ids:
                per_tools_id[tools_id].append(annotation)

        for tools_id, ann_list in per_tools_id.items():
            if len(ann_list) == 1:
                _kernel_annotations[tools_id].append(ann_list[0])
                continue

            merged: dict[str, Any] = {}
            for annotation in ann_list:
                if isinstance(annotation, dict):
                    for key, value in annotation.items():
                        merged.setdefault(key, value)
                else:
                    merged.setdefault("name", annotation)
            _kernel_annotations[tools_id].append(merged)
    except Exception:
        logger.exception("resolve_pending_annotations failed")
    finally:
        _pending_scopes.clear()


def remap_to_exec_graph(torch_cuda_graph: torch.cuda.CUDAGraph) -> None:
    """Remap annotation keys from capture graph ID to exec graph ID.

    During capture, toolsId encodes the capture graph's ID in the upper
    32 bits. After instantiation, the profiler uses the exec graph's ID.
    This function rewrites the keys so annotations match the trace.

    The graph's capture id is read from the ``_capture_graph_id`` stamped on it
    by ``maybe_stamp_capture_graph_id`` in the capture_end window, so only the annotations
    belonging to this graph are rekeyed. This is order-independent and correct
    when several graphs are captured in sequence: call once per graph. Graphs
    captured with annotations disabled have no capture id and are skipped.

    The exec graph id is only defined once the graph is instantiated. With
    ``keep_graph=True`` instantiation is deferred past the ``torch.cuda.graph()``
    context, so the remap is driven from the graph's ``instantiate()``/
    ``replay()`` instead of context exit. Each ``instantiate()`` (even on an
    unmodified template) produces a fresh exec graph id, so this rekeys from
    wherever the annotations are currently keyed -- the capture id before the
    first remap, the previous exec id after a re-instantiate -- to the current
    exec id. ``_remapped_exec_id`` tracks that current key; when it already
    matches the live exec id (e.g. replay after instantiate) this is a no-op.
    """
    capture_graph_id = torch_cuda_graph._capture_graph_id
    if not _kernel_annotations or capture_graph_id is None:
        return

    exec_graph_id = _check_cuda_bindings(
        _cuda_runtime.cudaGraphExecGetId(  # pyrefly: ignore[missing-attribute]
            torch_cuda_graph.raw_cuda_graph_exec()
        )
    )

    current_key_id = (
        capture_graph_id
        if torch_cuda_graph._remapped_exec_id is None
        else torch_cuda_graph._remapped_exec_id
    )
    if current_key_id == exec_graph_id:
        return

    remapped = _rekey_annotations(_kernel_annotations, current_key_id, exec_graph_id)
    _kernel_annotations.clear()
    _kernel_annotations.update(remapped)
    torch_cuda_graph._remapped_exec_id = exec_graph_id


def _rekey_annotations(
    annotations: dict[int, list[Any]],
    capture_graph_id: int,
    exec_graph_id: int,
) -> dict[int, list[Any]]:
    """Rekey one graph's annotations from its capture id to its exec id.

    A toolsId packs the graph id in the upper 32 bits and the node id in the
    lower 32. Only entries whose upper bits match ``capture_graph_id`` are
    rewritten to ``exec_graph_id``; entries from other graphs (already remapped
    to their own exec ids, or pending their own remap) are kept as-is. When two
    capture-side ids collide on the same rekeyed id their lists are merged.
    """
    remapped: dict[int, list[Any]] = {}
    for tools_id, ann_list in annotations.items():
        if tools_id >> 32 != capture_graph_id:
            remapped[tools_id] = ann_list
            continue
        node_id = tools_id & 0xFFFFFFFF
        new_tools_id = (exec_graph_id << 32) | node_id
        if new_tools_id in remapped:
            remapped[new_tools_id].extend(ann_list)
        else:
            remapped[new_tools_id] = list(ann_list)
    return remapped


def resolve_and_remap(torch_cuda_graph: torch.cuda.CUDAGraph) -> None:
    """Resolve any pending scopes and remap one graph in a single call.

    Shorthand for ``resolve_pending_annotations()`` followed by
    ``remap_to_exec_graph(graph)``; the pair normally run after a capture.
    """
    resolve_pending_annotations()
    remap_to_exec_graph(torch_cuda_graph)


def get_kernel_annotations() -> Mapping[int, list[Any]]:
    r"""get_kernel_annotations() -> Mapping[int, list]

    Return the live registry of recorded kernel annotations.

    Keys are opaque integers matching the ``graph node id`` field that
    CUPTI-based profilers attach to kernel events; values are the lists of
    annotation dicts recorded for that node. The registry accumulates
    across captures and is global to the process.

    The returned mapping is a **live view**: it is updated in place when a
    graph is instantiated (annotation keys are rekeyed to the executable
    graph's ids), so a reference obtained early stays current. Keys are
    valid for joining against a profiler trace once the corresponding
    graphs have been instantiated. Treat the mapping as read-only; snapshot
    it with ``dict(...)`` if isolation is needed.

    .. warning::
        This API is in prototype and may change in future releases.

    Example::

        >>> # xdoctest: +SKIP("requires cuda-bindings and driver >= 13.1")
        >>> annotations = torch.cuda.graph_annotations.get_kernel_annotations()
        >>> with open("annotations.pkl", "wb") as f:
        ...     pickle.dump(dict(annotations), f)
    """
    return _kernel_annotations


def clear_kernel_annotations() -> None:
    r"""clear_kernel_annotations() -> None

    Clear all recorded kernel annotations.

    The annotation registry is process-global and accumulates across
    captures; long-running workloads that capture many graphs should clear
    it once recorded annotations have been consumed (e.g. after saving
    them alongside a profiler trace).

    Clearing forgets everything recorded so far and revokes recording from
    every scope opened before the clear: backward hooks that
    :func:`mark_kernels` attached to existing autograd nodes stay on the
    graph but become inert. Scopes opened after the clear record normally.

    .. warning::
        This API is in prototype and may change in future releases.
    """
    global _annotation_generation
    _kernel_annotations.clear()
    _pending_scopes.clear()
    _annotation_generation += 1


def remove_kernel_annotations(exec_graph_ids: Iterable[int]) -> None:
    """Drop kernel-annotation entries whose exec graph id (tools_id >> 32) is in
    exec_graph_ids, so the map does not grow across the run. Run by the annotation
    resolver's graph-destroy handler."""
    ids = set(exec_graph_ids)
    if not ids:
        return
    for key in [k for k in _kernel_annotations if k >> 32 in ids]:
        del _kernel_annotations[key]


# Counter-based stream ID registry. IDs start at 60 (above the highest
# observed non-graphed CUDA stream ID) so every assigned lane is visually
# distinct in Perfetto and doesn't collide with real streams.
_stream_id_counter: int = 60
_stream_id_map: dict[int, int] = {}


def _get_stream_id(stream: torch.cuda.Stream) -> int:
    """Return a small, stable stream ID for the given CUDA stream."""
    global _stream_id_counter
    key = stream.cuda_stream
    if key not in _stream_id_map:
        _stream_id_map[key] = _stream_id_counter
        _stream_id_counter += 1
    return _stream_id_map[key]


def get_stream_for_pg(pg_key: str) -> int:
    """Return a unique stream ID for the given process group key."""
    global _stream_id_counter
    if pg_key not in _stream_id_map:
        _stream_id_map[pg_key] = _stream_id_counter  # type: ignore[assignment]
        _stream_id_counter += 1
    return _stream_id_map[pg_key]  # type: ignore[return-value]


@contextmanager  # type: ignore[arg-type]
def mark_stream(stream: torch.cuda.Stream, annotation: str | dict[str, Any]):
    """Switch to stream, inject its ID into annotation, and mark kernels.

    If *stream* is already the current stream, no stream switch or stream ID
    injection happens — the kernels stay on whatever stream is active (which
    keeps the trace faithful when e.g. FSDP uses the current stream for
    copy-in instead of a separate one). When switching to a different stream,
    this snapshots the current capturing branch before the target stream
    runs marked work. If the target stream is already capturing, the marked
    work must still be synchronized with the current stream so it is
    reachable from that snapped frontier.
    """
    if not _annotations_enabled:
        with torch.cuda.stream(stream):
            yield
        return
    if stream.cuda_stream == torch.cuda.current_stream().cuda_stream:
        with mark_kernels(annotation):
            yield
    else:
        if isinstance(annotation, str):
            annotation = {"name": annotation}
        if isinstance(annotation, dict):
            annotation["stream"] = _get_stream_id(stream)
        with mark_kernels(annotation):
            with torch.cuda.stream(stream):
                yield
<|endoftext|># mypy: allow-untyped-defs
r"""This package adds support for device memory management implemented in CUDA."""

import collections
import contextlib
import ctypes
import pickle
import sys
import threading
import warnings
from inspect import signature
from typing import Any, Literal, TYPE_CHECKING
from typing_extensions import deprecated

import torch
from torch import _C
from torch._utils import _augment_memory_snapshot_stack_traces, _dummy_type

from . import (
    _get_amdsmi_device_index,
    _get_device_index,
    _get_nvml_device_index,
    _lazy_init,
    is_initialized,
)
from ._memory_viz import memory as _memory, segments as _segments


if TYPE_CHECKING:
    from torch.types import Device


__all__ = [
    "caching_allocator_alloc",
    "caching_allocator_delete",
    "caching_allocator_disabled",
    "caching_allocator_enable",
    "get_per_process_memory_fraction",
    "set_per_process_memory_fraction",
    "empty_cache",
    "memory_stats",
    "memory_stats_as_nested_dict",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
    "reset_max_memory_allocated",
    "reset_max_memory_cached",
    "host_memory_stats",
    "host_memory_stats_as_nested_dict",
    "reset_accumulated_host_memory_stats",
    "reset_peak_host_memory_stats",
    "memory_allocated",
    "max_memory_allocated",
    "memory_reserved",
    "max_memory_reserved",
    "memory_cached",
    "max_memory_cached",
    "memory_snapshot",
    "memory_summary",
    "list_gpu_processes",
    "mem_get_info",
    "get_allocator_backend",
    "CUDAPluggableAllocator",
    "change_current_allocator",
    "MemPool",
    "use_mem_pool",
]


if not hasattr(torch._C, "_cuda_CUDAAllocator"):
    # Define dummy base classes
    torch._C.__dict__["_cuda_CUDAAllocator"] = _dummy_type("_cuda_CUDAAllocator")


if not hasattr(torch._C, "_MemPool"):
    # Define dummy base classes
    torch._C.__dict__["_MemPool"] = _dummy_type("_MemPool")
    torch._C.__dict__["_cuda_beginAllocateToPool"] = _dummy_type(
        "_cuda_beginAllocateToPool"
    )
    torch._C.__dict__["_cuda_beginAllocateCurrentThreadToPool"] = _dummy_type(
        "_cuda_beginAllocateCurrentThreadToPool"
    )
    torch._C.__dict__["_cuda_endAllocateToPool"] = _dummy_type(
        "_cuda_endAllocateToPool"
    )
    torch._C.__dict__["_cuda_releasePool"] = _dummy_type("_cuda_releasePool")

from torch._C import (  # noqa: F401
    _cuda_beginAllocateCurrentThreadToPool,
    _cuda_beginAllocateToPool,
    _cuda_CUDAAllocator,
    _cuda_endAllocateToPool,
    _cuda_releasePool,
    _MemPool,
)


def _host_allocator():
    _lazy_init()
    return torch._C._cuda_cudaHostAllocator()


def caching_allocator_alloc(size, device: "Device" = None, stream=None):
    r"""Perform a memory allocation using the CUDA memory allocator.

    Memory is allocated for a given device and a stream, this
    function is intended to be used for interoperability with other
    frameworks. Allocated memory is released through
    :func:`~torch.cuda.caching_allocator_delete`.

    Args:
        size (int): number of bytes to be allocated.
        device (torch.device or int, optional): selected device. If it is
            ``None`` the default CUDA device is used.
        stream (torch.cuda.Stream or int, optional): selected stream. If is ``None`` then
            the default stream for the selected device is used.

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    if device is None:
        device = torch.cuda.current_device()
    device = _get_device_index(device)
    if stream is None:
        stream = torch.cuda.current_stream(device)
    if isinstance(stream, torch.cuda.streams.Stream):
        stream = stream.cuda_stream
    if not isinstance(stream, int):
        raise TypeError(
            "Invalid type for stream argument, must be "
            "`torch.cuda.Stream` or `int` representing a pointer "
            "to an existing stream"
        )
    with torch.cuda.device(device):
        return torch._C._cuda_cudaCachingAllocator_raw_alloc(size, stream)


def caching_allocator_delete(mem_ptr):
    r"""Delete memory allocated using the CUDA memory allocator.

    Memory allocated with :func:`~torch.cuda.caching_allocator_alloc`.
    is freed here. The associated device and stream are tracked inside
    the allocator.

    Args:
        mem_ptr (int): memory address to be freed by the allocator.

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    torch._C._cuda_cudaCachingAllocator_raw_delete(mem_ptr)


def caching_allocator_enable(value: bool = True) -> None:
    r"""Enable or disable the CUDA memory allocator. On by default."""
    if is_initialized():
        torch._C._cuda_cudaCachingAllocator_enable(value)


@contextlib.contextmanager
def caching_allocator_disabled():
    r"""Context manager that temporarily disables the CUDA caching allocator."""
    # pyrefly: ignore [missing-attribute]
    prev = torch._C._cuda_cudaCachingAllocator_is_enabled()
    caching_allocator_enable(False)
    try:
        yield
    finally:
        caching_allocator_enable(prev)


def set_per_process_memory_fraction(fraction, device: "Device" = None) -> None:
    r"""Set memory fraction for a process.

    The fraction is used to limit an caching allocator to allocated memory on a CUDA device.
    The allowed value equals the total visible memory multiplied fraction.
    If trying to allocate more than the allowed value in a process, will raise an out of
    memory error in allocator.

    Args:
        fraction(float): Range: 0~1. Allowed memory equals total_memory * fraction.
        device (torch.device or int, optional): selected device. If it is
            ``None`` the default CUDA device is used.
    .. note::
        In general, the total available free memory is less than the total capacity.
    """
    _lazy_init()
    if device is None:
        device = torch.cuda.current_device()
    device = _get_device_index(device)
    if not isinstance(fraction, float):
        raise TypeError("Invalid type for fraction argument, must be `float`")
    if fraction < 0 or fraction > 1:
        raise ValueError(f"Invalid fraction value: {fraction}. Allowed range: 0~1")

    torch._C._cuda_setMemoryFraction(fraction, device)


def get_per_process_memory_fraction(device: "Device" = None) -> float:
    r"""Get memory fraction for a process.

    Args:
        device (torch.device or int, optional): selected device. If it is
            ``None`` the default CUDA device is used.
    Returns:
        memory fraction, in range 0~1. Allowed memory equals total_memory * fraction.
    """
    _lazy_init()
    if device is None:
        device = torch.cuda.current_device()
    device = _get_device_index(device)
    return torch._C._cuda_getMemoryFraction(device)


def empty_cache() -> None:
    r"""Release all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other GPU application and visible in
    `nvidia-smi`.

    .. note::
        :func:`~torch.cuda.empty_cache` doesn't increase the amount of GPU
        memory available for PyTorch. However, it may help reduce fragmentation
        of GPU memory in certain cases. See :ref:`cuda-memory-management` for
        more details about GPU memory management.
    """
    if is_initialized():
        torch._C._cuda_emptyCache()


def _recurse_add_to_result(result, prefix, obj, format_key):
    if isinstance(obj, dict):
        if prefix:
            prefix += "."
        for key, value in obj.items():
            _recurse_add_to_result(result, prefix + format_key(key), value, format_key)
    else:
        result.append((prefix, obj))


def memory_stats(device: "Device" = None) -> dict[str, Any]:
    r"""Return a dictionary of CUDA memory allocator statistics for a given device.

    The return value of this function is a dictionary of statistics, each of
    which is a non-negative integer.

    Core statistics:

    - ``"allocated.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of allocation requests received by the memory allocator.
    - ``"allocated_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of allocated memory.
    - ``"segment.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of reserved segments from ``cudaMalloc()``.
    - ``"reserved_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of reserved memory.
    - ``"active.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of active memory blocks.
    - ``"active_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of active memory.
    - ``"inactive_split.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of inactive, non-releasable memory blocks.
    - ``"inactive_split_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of inactive, non-releasable memory.

    For these core statistics, values are broken down as follows.

    Pool type:

    - ``all``: combined statistics across all memory pools.
    - ``large_pool``: statistics for the large allocation pool
      (as of June 2025, for size >= 1MB allocations).
    - ``small_pool``: statistics for the small allocation pool
      (as of June 2025, for size < 1MB allocations).

    Metric type:

    - ``current``: current value of this metric.
    - ``peak``: maximum value of this metric.
    - ``allocated``: historical total increase in this metric.
    - ``freed``: historical total decrease in this metric.

    In addition to the core statistics, we also provide some simple event
    counters:

    - ``"num_alloc_retries"``: number of failed ``cudaMalloc`` calls that
      result in a cache flush and retry.
    - ``"num_ooms"``: number of out-of-memory errors thrown.
    - ``"num_sync_all_streams"``: number of ``synchronize_and_free_events`` calls.
    - ``"num_device_alloc"``: number of CUDA allocation calls. This includes both
      cuMemMap and cudaMalloc.
    - ``"num_device_free"``: number of CUDA free calls. This includes both cuMemUnmap
      and cudaFree.
    - ``"num_oom_rejections"``: number of allocations preemptively rejected by the
        throw_on_cudamalloc_oom + per_process_memory_fraction policy.

    The caching allocator can be configured via ENV to not split blocks larger than a
    defined size (see Memory Management section of the Cuda Semantics documentation).
    This helps avoid memory fragmentation but may have a performance
    penalty. Additional outputs to assist with tuning and evaluating impact:

    - ``"max_split_size"``: blocks above this size will not be split.
    - ``"oversize_allocations.{current,peak,allocated,freed}"``:
      number of over-size allocation requests received by the memory allocator.
    - ``"oversize_segments.{current,peak,allocated,freed}"``:
      number of over-size reserved segments from ``cudaMalloc()``.

    The caching allocator can be configured via ENV to round memory allocations in order
    to reduce fragmentation. Sometimes the overhead from rounding can be higher than
    the fragmentation it helps reduce. The following stat can be used to check if
    rounding adds too much overhead:

    - ``"requested_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      memory requested by client code, compare this with allocated_bytes to check if
      allocation rounding adds too much overhead.
    - ``"reserved_bytes_by_private_pools"``: nested dictionary keyed by
      ``torch.cuda.MemPool.id`` tuples. Each value has the same
      ``{all,large_pool,small_pool}.{current,peak,allocated,freed}`` structure
      as ``reserved_bytes``, but scoped to a single private pool. In
      :func:`~torch.cuda.memory_stats`, tuple keys are flattened by joining
      their stringified elements with ``"_"``, so ``(0, 1)`` becomes ``"0_1"``.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistics for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.

    .. note::
        With :ref:`backend:cudaMallocAsync<cuda-memory-envvars>`, some stats are not
        meaningful, and are always reported as zero.
    """
    result = []

    def _format_key(key):
        if isinstance(key, str):
            return key
        if isinstance(key, tuple):
            return "_".join(str(part) for part in key)
        return str(key)

    stats = memory_stats_as_nested_dict(device=device)
    _recurse_add_to_result(result, "", stats, _format_key)
    result.sort()

    return collections.OrderedDict(result)


def memory_stats_as_nested_dict(device: "Device" = None) -> dict[str, Any]:
    r"""Return the result of :func:`~torch.cuda.memory_stats` as a nested dictionary."""
    if not is_initialized():
        return {}
    device = _get_device_index(device, optional=True)
    return torch._C._cuda_memoryStats(device)


def reset_accumulated_memory_stats(device: "Device" = None) -> None:
    r"""Reset the "accumulated" (historical) stats tracked by the CUDA memory allocator.

    See :func:`~torch.cuda.memory_stats` for details. Accumulated stats correspond to
    the `"allocated"` and `"freed"` keys in each individual stat dict, as well as
    `"num_alloc_retries"` and `"num_ooms"`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    device = _get_device_index(device, optional=True)
    return torch._C._cuda_resetAccumulatedMemoryStats(device)


def reset_peak_memory_stats(device: "Device" = None) -> None:
    r"""Reset the "peak" stats tracked by the CUDA memory allocator.

    See :func:`~torch.cuda.memory_stats` for details. Peak stats correspond to the
    `"peak"` key in each individual stat dict.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    device = _get_device_index(device, optional=True)
    return torch._C._cuda_resetPeakMemoryStats(device)


def host_memory_stats() -> dict[str, Any]:
    r"""Return a dictionary of pinned (host) allocator statistics.

    Core statistics (host pinned allocator):

    - ``"allocations.{current,peak,allocated,freed}"``:
      pinned blocks owned by the allocator (active + cached). Grows when a new
      block is created via CUDA and shrinks when cached blocks are returned.
    - ``"allocated_bytes.{current,peak,allocated,freed}"``:
      bytes of pinned blocks owned by the allocator (active + cached), using
      the rounded block size requested from CUDA.
    - ``"active_requests.{current,peak,allocated,freed}"``:
      blocks currently checked out to callers (increments on handout, decrements
      when the block becomes reusable after stream deps finish).
    - ``"active_bytes.{current,peak,allocated,freed}"``:
      bytes corresponding to active blocks.

    Metric type:

    - ``current``: current value.
    - ``peak``: maximum value.
    - ``allocated``: historical total increase.
    - ``freed``: historical total decrease.

    Event/timing counters:

    - ``"num_host_alloc"`` / ``"num_host_free"``: blocks created to grow the
      pool / cached blocks returned to CUDA (matches allocations allocated/freed).
    - ``"host_alloc_time.{total,max,min,count,avg}"``: time in CUDA alloc calls
      when growing the pool (microseconds).
    - ``"host_free_time.{total,max,min,count,avg}"``: time in CUDA free calls
      when cached blocks are returned (microseconds).

    Block sizes are rounded up to the next power of two before calling CUDA, so
    byte stats reflect the rounded size. Peak values are aggregated per bucket
    and are a best-effort approximation of the true peak.
    """
    result = []

    stats = host_memory_stats_as_nested_dict()
    _recurse_add_to_result(result, "", stats, str)
    result.sort()

    return collections.OrderedDict(result)


def host_memory_stats_as_nested_dict() -> dict[str, Any]:
    r"""Return the result of :func:`~torch.cuda.host_memory_stats` as a nested dictionary."""
    if not is_initialized():
        return {}
    return torch._C._cuda_hostMemoryStats()


def reset_accumulated_host_memory_stats() -> None:
    r"""Reset the "accumulated" (historical) stats tracked by the host memory allocator.

    See :func:`~torch.cuda.host_memory_stats` for details. Accumulated stats correspond to
    the `"allocated"` and `"freed"` keys in each individual stat dict.
    """
    return torch._C._cuda_resetAccumulatedHostMemoryStats()


def reset_peak_host_memory_stats() -> None:
    r"""Reset the "peak" stats tracked by the host memory allocator.

    See :func:`~torch.cuda.host_memory_stats` for details. Peak stats correspond to the
    `"peak"` key in each individual stat dict.
    """
    return torch._C._cuda_resetPeakHostMemoryStats()


def reset_max_memory_allocated(device: "Device" = None) -> None:
    r"""Reset the starting point in tracking maximum GPU memory occupied by tensors for a given device.

    See :func:`~torch.cuda.max_memory_allocated` for details.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. warning::
        This function now calls :func:`~torch.cuda.reset_peak_memory_stats`, which resets
        /all/ peak memory stats.

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    warnings.warn(
        "torch.cuda.reset_max_memory_allocated now calls torch.cuda.reset_peak_memory_stats, "
        "which resets /all/ peak memory stats.",
        FutureWarning,
        stacklevel=2,
    )
    return reset_peak_memory_stats(device=device)


def reset_max_memory_cached(device: "Device" = None) -> None:
    r"""Reset the starting point in tracking maximum GPU memory managed by the caching allocator for a given device.

    See :func:`~torch.cuda.max_memory_cached` for details.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. warning::
        This function now calls :func:`~torch.cuda.reset_peak_memory_stats`, which resets
        /all/ peak memory stats.

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    warnings.warn(
        "torch.cuda.reset_max_memory_cached now calls torch.cuda.reset_peak_memory_stats, "
        "which resets /all/ peak memory stats.",
        FutureWarning,
        stacklevel=2,
    )
    return reset_peak_memory_stats(device=device)


def memory_allocated(device: "Device" = None) -> int:
    r"""Return the current GPU memory occupied by tensors in bytes for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        This is likely less than the amount shown in `nvidia-smi` since some
        unused memory can be held by the caching allocator and some context
        needs to be created on GPU. See :ref:`cuda-memory-management` for more
        details about GPU memory management.
    """
    return memory_stats(device=device).get("allocated_bytes.all.current", 0)


def max_memory_allocated(device: "Device" = None) -> int:
    r"""Return the maximum GPU memory occupied by tensors in bytes for a given device.

    By default, this returns the peak allocated memory since the beginning of
    this program. :func:`~torch.cuda.reset_peak_memory_stats` can be used to
    reset the starting point in tracking this metric. For example, these two
    functions can measure the peak allocated memory usage of each iteration in a
    training loop.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    return memory_stats(device=device).get("allocated_bytes.all.peak", 0)


def memory_reserved(device: "Device" = None) -> int:
    r"""Return the current GPU memory managed by the caching allocator in bytes for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    return memory_stats(device=device).get("reserved_bytes.all.current", 0)


def max_memory_reserved(device: "Device" = None) -> int:
    r"""Return the maximum GPU memory managed by the caching allocator in bytes for a given device.

    By default, this returns the peak cached memory since the beginning of this
    program. :func:`~torch.cuda.reset_peak_memory_stats` can be used to reset
    the starting point in tracking this metric. For example, these two functions
    can measure the peak cached memory amount of each iteration in a training
    loop.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.

    .. note::
        Under ``PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync``, the peak is
        computed by summing the high-water marks of the default mempool and the
        device graph-memory pool (CUDA graph captures reserve backing in the
        latter). Because those two high-water marks need not occur at the same
        instant, the reported peak is a conservative *upper bound* on the true
        simultaneous peak. The current value
        (:func:`~torch.cuda.memory_reserved`) is exact.
    """
    return memory_stats(device=device).get("reserved_bytes.all.peak", 0)


@deprecated(
    "`torch.cuda.memory_cached` has been renamed to `torch.cuda.memory_reserved`",
    category=FutureWarning,
)
def memory_cached(device: "Device" = None) -> int:
    r"""Deprecated; see :func:`~torch.cuda.memory_reserved`."""
    return memory_reserved(device=device)


@deprecated(
    "`torch.cuda.max_memory_cached` has been renamed to `torch.cuda.max_memory_reserved`",
    category=FutureWarning,
)
def max_memory_cached(device: "Device" = None) -> int:
    r"""Deprecated; see :func:`~torch.cuda.max_memory_reserved`."""
    return max_memory_reserved(device=device)


def memory_snapshot(mempool_id=None, include_traces=True):
    r"""Return a snapshot of the CUDA memory allocator state across all devices.

    Interpreting the output of this function requires familiarity with the
    memory allocator internals.

    Args:
        mempool_id: Optional memory pool ID to get snapshot for a specific pool
        include_traces: Whether to include trace entries in the snapshot.
            If True (default), all trace entries are included.
            If False, no trace entries are included (lightweight/fast snapshot).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    if mempool_id is None:
        # pyrefly: ignore [bad-argument-type]
        return torch._C._cuda_memorySnapshot((0, 0, include_traces))["segments"]
    else:
        return torch._C._cuda_memorySnapshot(
            # pyrefly: ignore [bad-argument-type]
            (mempool_id[0], mempool_id[1], include_traces)
        )["segments"]


def memory_summary(device: "Device" = None, abbreviated: bool = False) -> str:
    r"""Return a human-readable printout of the current memory allocator statistics for a given device.

    This can be useful to display periodically during training, or when
    handling out-of-memory exceptions.

    Args:
        device (torch.device or int, optional): selected device. Returns
            printout for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).
        abbreviated (bool, optional): whether to return an abbreviated summary
            (default: False).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    device = _get_device_index(device, optional=True)
    stats = memory_stats(device=device)

    def _format_size(sz, pref_sz):
        prefixes = ["B  ", "KiB", "MiB", "GiB", "TiB", "PiB"]
        prefix = prefixes[0]
        for new_prefix in prefixes[1:]:
            if pref_sz < 768 * 1024:
                break
            prefix = new_prefix
            sz //= 1024
            pref_sz /= 1024
        return f"{sz:6d} {prefix}"

    def _format_count(cnt, pref_cnt):
        prefixes = [" ", "K", "M"]
        prefix = prefixes[0]
        for new_prefix in prefixes[1:]:
            if pref_cnt < 750 * 1000:
                break
            prefix = new_prefix
            cnt //= 1000
            pref_cnt /= 1000
        return f"{cnt:7d} {prefix} "

    metrics_to_display = [
        ("allocated_bytes", "Allocated memory", _format_size),
        ("active_bytes", "Active memory", _format_size),
        ("requested_bytes", "Requested memory", _format_size),
        ("reserved_bytes", "GPU reserved memory", _format_size),
        ("inactive_split_bytes", "Non-releasable memory", _format_size),
        ("allocation", "Allocations", _format_count),
        ("active", "Active allocs", _format_count),
        ("segment", "GPU reserved segments", _format_count),
        ("inactive_split", "Non-releasable allocs", _format_count),
    ]

    lines = []
    lines.append("=" * 75)
    lines.append(" {_:16} PyTorch CUDA memory summary, device ID {device:<17d} ")
    lines.append("-" * 75)
    lines.append(
        "  {_:9} CUDA OOMs: {num_ooms:<12d} | {_:6} cudaMalloc retries: {num_alloc_retries:<8d}  "
    )
    lines.append("=" * 75)
    lines.append(
        "        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  "
    )

    for metric_key, metric_name, formatter in metrics_to_display:
        lines.append("-" * 75)
        submetrics = [("all", metric_name)]
        if not abbreviated:
            submetrics.append(("large_pool", "      from large pool"))
            submetrics.append(("small_pool", "      from small pool"))

        current_prefval, peak_prefval, allocated_prefval, freed_prefval = (
            None,
            None,
            None,
            None,
        )

        for submetric_key, submetric_name in submetrics:
            prefix = metric_key + "." + submetric_key + "."

            current = stats[prefix + "current"]
            peak = stats[prefix + "peak"]
            allocated = stats[prefix + "allocated"]
            freed = stats[prefix + "freed"]

            if current_prefval is None:
                current_prefval = current
                peak_prefval = peak
                allocated_prefval = allocated
                freed_prefval = freed

            lines.append(
                # pyrefly: ignore [bad-argument-type]
                f" {submetric_name:<21} | {formatter(current, current_prefval)} | {formatter(peak, peak_prefval)} | "
                f"{formatter(allocated, allocated_prefval)} | {formatter(freed, freed_prefval)} ",
            )

    metrics_to_display = [
        ("oversize_allocations", "Oversize allocations", _format_count),
        ("oversize_segments", "Oversize GPU segments", _format_count),
    ]

    for metric_key, metric_name, formatter in metrics_to_display:
        lines.append("-" * 75)

        prefix = metric_key + "."

        current = stats[prefix + "current"]
        peak = stats[prefix + "peak"]
        allocated = stats[prefix + "allocated"]
        freed = stats[prefix + "freed"]

        lines.append(
            # pyrefly: ignore [bad-argument-type]
            f" {metric_name:<21} | {formatter(current, current)} | {formatter(peak, peak)} | "
            f"{formatter(allocated, allocated)} | {formatter(freed, freed)} ",
        )

    lines.append("=" * 75)

    fmt_dict = {"_": "", "device": device}
    for k, v in stats.items():
        fmt_dict[k.replace(".", "-")] = v
    return "|" + "|\n|".join(lines).format(**fmt_dict) + "|\n"


def list_gpu_processes(device: "Device" = None) -> str:
    r"""Return a human-readable printout of the running processes and their GPU memory use for a given device.

    This can be useful to display periodically during training, or when
    handling out-of-memory exceptions.

    Args:
        device (torch.device or int, optional): selected device. Returns
            printout for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    if not torch.version.hip:
        try:
            import pynvml  # type: ignore[import]
        except ModuleNotFoundError:
            return "pynvml module not found, please install nvidia-ml-py"
        # pyrefly: ignore [import-error, missing-import, missing-module-attribute]
        from pynvml import NVMLError_DriverNotLoaded

        try:
            pynvml.nvmlInit()
        except NVMLError_DriverNotLoaded:
            return "cuda driver can't be loaded, is cuda enabled?"

        device = _get_nvml_device_index(device)
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    else:
        try:
            import amdsmi  # type: ignore[import]
        except ModuleNotFoundError:
            return "amdsmi module not found, please install amdsmi"
        try:
            amdsmi.amdsmi_init()  # type: ignore[attr-defined]
        except amdsmi.AmdSmiException:  # type: ignore[attr-defined]
            return "amdsmi driver can't be loaded, is ROCm installed?"

        device = _get_amdsmi_device_index(device)

        try:
            handle = amdsmi.amdsmi_get_processor_handles()[device]  # type: ignore[attr-defined]
            procs = amdsmi.amdsmi_get_gpu_process_list(handle)  # type: ignore[attr-defined]
        except amdsmi.AmdSmiException:  # type: ignore[attr-defined]
            return "amdsmi cannot list processes from other users"

    lines = []
    lines.append(f"GPU:{device}")
    if len(procs) == 0:
        lines.append("no processes are running")
    for p in procs:
        if not torch.version.hip:
            mem = p.usedGpuMemory / (1024 * 1024)
            pid = p.pid
        else:
            try:
                proc_info = amdsmi.amdsmi_get_gpu_process_info(handle, p)  # type: ignore[possibly-undefined]
            except AttributeError:
                # https://github.com/ROCm/amdsmi/commit/c551c3caedbd903ba828e7fdffa5b56d475a15e7
                # is a BC-breaking change that removes amdsmi_get_gpu_process_info API from amdsmi
                proc_info = p
            mem = proc_info["memory_usage"]["vram_mem"] / (1024 * 1024)
            pid = proc_info["pid"]
        lines.append(f"process {pid:>10d} uses {mem:>12.3f} MB GPU memory")
    return "\n".join(lines)


def mem_get_info(device: "Device" = None) -> tuple[int, int]:
    r"""Return the global free and total GPU memory for a given device using cudaMemGetInfo.

    Args:
        device (torch.device or int or str, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default) or if the device index is not specified.

    .. note::
        See :ref:`cuda-memory-management` for more
        details about GPU memory management.
    """
    if device is None:
        device = torch.cuda.current_device()
    # optional=True allows `device = torch.device('cuda')` for which device.index is None
    device = _get_device_index(device, optional=True)
    return torch.cuda.cudart().cudaMemGetInfo(device)


def _record_memory_history_legacy(
    enabled: bool,
    record_context=True,
    trace_alloc_max_entries=1,
    trace_alloc_record_context=False,
    device: "Device" = None,
    record_context_cpp=False,
    clear_history=False,
    compile_context=False,
    global_record_annotations=False,
    skip_actions=None,
    record_pinned_host_memory=False,
    record_cuda=True,
):
    _C._cuda_record_memory_history_legacy(  # type: ignore[call-arg]
        enabled,
        record_context,
        # pyrefly: ignore [bad-argument-type]
        trace_alloc_max_entries,
        trace_alloc_record_context,
        record_context_cpp,
        clear_history,
        compile_context,
        global_record_annotations,
        # pyrefly: ignore [bad-argument-count]
        skip_actions if skip_actions is not None else [],
        record_pinned_host_memory,
        record_cuda,
    )


def _record_memory_history(
    enabled: Literal["state", "all"] | None = "all", *args, **kwargs
) -> None:
    """Enable recording of stack traces associated with memory
    allocations, so you can tell what allocated any piece of memory in
    :func:`torch.cuda.memory._snapshot()`.

    In addition to keeping stack traces with each current allocation and free,
    this will also enable recording of a history of all alloc/free events.

    Use :func:`torch.cuda.memory._snapshot()` to retrieve this information,
    and the tools in `_memory_viz.py` to visualize snapshots.

    Buffer behavior
    ---------------

    This will store up to `max_entries` instances of `TraceEntry` when enabled.
    Python trace collection defaults to `sys.maxsize`, meaning long-running
    or indefinitely running jobs should set a reasonable limit to avoid excessive
    memory use. Expect each entry to be several KB.

    Longer running workflows or those with smaller `max_entries` values will only
    store the last accumulated `max_entries` entries, meaning new entries overwrite
    older entries.

    C++ implementation for reference to ring buffer implementation:

    .. code-block:: cpp

        if (record_history) {
          if (alloc_trace->size() < alloc_trace_max_entries_) {
            alloc_trace->emplace_back(te);
          } else {
            (*alloc_trace)[alloc_trace_next++] = te;
            if (alloc_trace_next == alloc_trace_max_entries_) {
              alloc_trace_next = 0;
            }
          }
        }

    Latency impact
    --------------

    The Python trace collection is fast (2us per trace), so you may consider
    enabling this on production jobs if you anticipate ever having to debug
    memory issues.

    C++ trace collection is also fast (~50ns/frame), which for many typical programs
    works out to ~2us per trace, but can vary depending on stack depth.

    Args:
        enabled (Literal[None, "state", "all"], optional):
            `None`, disable recording memory history.
            `"state"`, keep information for currently allocated memory.
            `"all"`, additionally keep a history of all alloc/free calls.
            Defaults to "all".
        context (Literal[None, "state", "alloc", "all"], optional):
            `None`, Do not record any tracebacks.
            `"state"`, Record tracebacks for currently allocated memory.
            `"alloc"`, additionally keep tracebacks for alloc calls.
            `"all"`, additionally keep tracebacks for free calls.
            Defaults to "all".
        stacks (Literal["python", "all"], optional):
            `"python"`, include Python, TorchScript, and inductor frames in tracebacks
            `"all"`, additionally include C++ frames
            Defaults to "all".
        max_entries (int, optional): Keep a maximum of `max_entries`
            alloc/free events in the recorded history recorded.
        clear_history (bool, optional): Clear history when enabling, defaults to False.
        skip_actions (list[str], optional): List of action types to skip when recording
            memory history. This can be used to reduce memory overhead by excluding
            certain types of events from being recorded. Valid action types are:

            - `"alloc"`: Memory allocation events
            - `"free_requested"`: Free requests (memory marked for freeing)
            - `"free_completed"`: Completed free operations (memory actually freed)
            - `"segment_alloc"`: Segment allocation from cudaMalloc
            - `"segment_free"`: Segment freed back to CUDA via cudaFree
            - `"oom"`: Out-of-memory exceptions
            - `"snapshot"`: Memory snapshot generation events

            For example, to skip recording free_requested events:
            `skip_actions=["free_requested"]`

            Defaults to None (record all actions).
        record_pinned_host_memory (bool, optional): If True, also record memory history for
            the CPU pinned memory (host) allocator. Host allocator traces will
            appear in the ``host_segments`` and ``host_traces`` keys of the
            snapshot returned by :func:`_snapshot`. Defaults to False.
        record_cuda (bool, optional): If True, record memory history for the
            CUDA device allocator. Set to False (together with
            ``record_pinned_host_memory=True``) to record only host pinned memory.
            Defaults to True.

    """
    if isinstance(enabled, bool):
        return _record_memory_history_legacy(enabled, *args, **kwargs)
    else:
        return _record_memory_history_impl(enabled, *args, **kwargs)


def _record_memory_history_impl(
    enabled: str | None = "all",
    context: str | None = "all",
    stacks: str = "all",
    max_entries: int = sys.maxsize,
    device: "Device" = None,
    clear_history: bool = False,
    compile_context: bool = False,
    global_record_annotations: bool = False,
    skip_actions: list[str] | None = None,
    record_pinned_host_memory: bool = False,
    record_cuda: bool = True,
):
    _C._cuda_record_memory_history(  # type: ignore[call-arg]
        enabled,
        context,
        stacks,
        max_entries,
        clear_history,
        compile_context,
        global_record_annotations,
        # pyrefly: ignore [bad-argument-count]
        skip_actions if skip_actions is not None else [],
        record_pinned_host_memory,
        record_cuda,
    )


_record_memory_history.__signature__ = signature(_record_memory_history_impl)  # type: ignore[attr-defined]


def _allocation_traceback(data_ptr: int) -> list[dict[str, Any]] | None:
    r"""Return the allocation traceback for a currently-allocated CUDA pointer,
    or ``None`` if the pointer is not found or recording was not enabled.

    ``data_ptr`` must be the base address of a recorded allocation, e.g. the
    result of ``tensor.untyped_storage().data_ptr()``. Passing the result of
    ``tensor.data_ptr()`` for a view with a nonzero ``storage_offset`` may
    return ``None`` even if the underlying allocation is still live.

    Requires :func:`_record_memory_history` to have been called with
    ``context != None`` beforehand.
    """
    return torch._C._cuda_allocationTraceback(data_ptr)


def _snapshot(device: "Device" = None, augment_with_fx_traces=False):
    """Save a snapshot of CUDA memory state at the time it was called.

    The state is represented as a dictionary with the following structure.

    .. code-block:: python

        class Snapshot(TypedDict):
            segments: List[Segment]
            device_traces: List[List[TraceEntry]]


        class Segment(TypedDict):
            # Segments are memory returned from a cudaMalloc call.
            # The size of reserved memory is the sum of all Segments.
            # Segments are cached and reused for future allocations.
            # If the reuse is smaller than the segment, the segment
            # is split into more than one Block.
            # empty_cache() frees Segments that are entirely inactive.
            address: int
            total_size: int  #  cudaMalloc'd size of segment
            stream: int
            segment_type: Literal["small", "large"]  # 'large' (>1MB)
            segment_pool_id: Tuple[
                int, int
            ]  # id of the memory pool owning this segment
            allocated_size: int  # size of memory in use
            active_size: int  # size of memory in use or in active_awaiting_free state
            blocks: List[Block]


        class Block(TypedDict):
            # A piece of memory returned from the allocator, or
            # current cached but inactive.
            size: int
            requested_size: int  # size requested during malloc, may be smaller than
            # size due to rounding
            address: int
            state: Literal[
                "active_allocated",  # used by a tensor
                "active_awaiting_free",  # waiting for another stream to finish using
                # this, then it will become free
                "inactive",
            ]  # free for reuse
            frames: List[Frame]  # stack trace from where the allocation occurred


        class Frame(TypedDict):
            filename: str
            line: int
            name: str
            # Optional FX debug fields (present when augment_with_fx_traces=True
            # and the frame corresponds to FX-generated code)
            fx_node_op: str  # FX node operation type (e.g., 'call_function', 'output')
            fx_node_name: str  # FX node name (e.g., 'linear', 'relu_1')
            fx_original_trace: str  # Original model source code stack trace


        class TraceEntry(TypedDict):
            # When `torch.cuda.memory._record_memory_history()` is enabled,
            # the snapshot will contain TraceEntry objects that record each
            # action the allocator took.
            action: Literal[
                "alloc"  # memory allocated
                "free_requested",  # the allocated received a call to free memory
                "free_completed",  # the memory that was requested to be freed is now
                # able to be used in future allocation calls
                "segment_alloc",  # the caching allocator ask cudaMalloc for more memory
                # and added it as a segment in its cache
                "segment_free",  # the caching allocator called cudaFree to return memory
                # to cuda possibly trying free up memory to
                # allocate more segments or because empty_caches was called
                "oom",  # the allocator threw an OOM exception. 'size' is
                # the requested number of bytes that did not succeed
                "snapshot",  # the allocator generated a memory snapshot
                # useful to correlate a previously taken
                # snapshot with this trace
                "annotate",  # metadata was attached to a live allocation
                # via _annotate_tensor. 'addr' is the allocation's base
                # address and 'user_metadata' holds the annotation
            ]
            addr: int  # not present for OOM
            frames: List[Frame]
            size: int
            stream: int
            device_free: int  # only present for OOM, the amount of
            # memory cuda still reports to be free
            pool_id: Tuple[int, int]  # id of the memory pool for this entry

    Args:
        device: Device to capture snapshot for. If None, captures for current device.
        augment_with_fx_traces: If True, augment stack trace frames with FX debug information
                                that maps generated FX code back to original model source code.
                                This adds fx_node_op, fx_node_name, fx_original_trace, and
                                fx_node_info fields to Frame objects. Default: False.

    Returns:
        The Snapshot dictionary object
    """
    s = _C._cuda_memorySnapshot(None)
    if augment_with_fx_traces:
        s = _augment_memory_snapshot_stack_traces(s)  # type: ignore[assignment, arg-type]
    return s


def _dump_snapshot(filename="dump_snapshot.pickle", augment_with_fx_traces=False):
    """
    Save a pickled version of the `torch.memory._snapshot()` dictionary to a file.

    This file can be opened by the interactive snapshot viewer at pytorch.org/memory_viz

    Snapshot file sizes scale with `max_entries` and stack trace depth per entry,
    with several KB per entry. These can easily be in the GB range for longer running
    workflows with large `max_entries`.

    Args:
        filename (str, optional): Name of the file to create. Defaults to "dump_snapshot.pickle".
        augment_with_fx_traces (bool, optional): If True, augment the snapshot with FX debug information
                                                  before dumping. This maps generated FX code stack traces
                                                  back to original model source code. Defaults to False.
    """
    s = _snapshot(augment_with_fx_traces=augment_with_fx_traces)

    with open(filename, "wb") as f:
        pickle.dump(s, f)


def _memory_metadata_supported() -> bool:
    """
    Return whether the active CUDA allocator backend records memory metadata.

    Only the native caching allocator supports :func:`_set_memory_metadata`.
    With other backends (e.g. ``cudaMallocAsync`` or a pluggable allocator),
    :func:`_set_memory_metadata` is a no-op and :func:`_get_memory_metadata`
    always returns the empty string. Callers doing best-effort labeling can
    check this before setting metadata to avoid the one-time warning emitted on
    unsupported backends.
    """
    # pyrefly: ignore [missing-attribute]
    return torch._C._cuda_memoryMetadataSupported()


def _set_memory_metadata(metadata: str):
    """
    Set custom metadata to be recorded on memory history trace entries.

    While memory history recording is enabled (see :func:`_record_memory_history`),
    every allocator trace event generated by the calling thread carries the
    current metadata string; it appears as the ``user_metadata`` key of trace
    entries in the snapshot returned by :func:`_snapshot` and is displayed by
    the pytorch.org/memory_viz visualizer. If memory history recording is not
    enabled, this setting has no observable effect. The metadata does not
    appear in the snapshot's segments/blocks view, in :func:`memory_stats`, or
    on pinned host-memory allocations.

    The metadata is thread-local: allocations made by other threads (including
    autograd's backward threads) are not affected. It applies to allocations
    on all devices used by the calling thread.

    This is only supported by the native caching allocator; with other
    backends (e.g. ``cudaMallocAsync`` or a pluggable allocator) it is
    silently ignored.

    Args:
        metadata (str): Custom metadata string to record on trace entries.
                       Pass an empty string to clear the metadata.
    """
    # pyrefly: ignore [missing-attribute]
    torch._C._cuda_setMemoryMetadata(metadata)


def _get_memory_metadata() -> str:
    """
    Get the calling thread's current memory history metadata.

    See :func:`_set_memory_metadata`. Note that with allocator backends that
    do not support metadata this always returns the empty string, which is
    indistinguishable from no metadata being set.

    Returns:
        str: The current metadata string, or empty string if no metadata is set.
    """
    # pyrefly: ignore [missing-attribute]
    return torch._C._cuda_getMemoryMetadata()


def _annotate_tensor(tensor: torch.Tensor, metadata: str):
    """
    Attach metadata to the allocation backing ``tensor``, post facto.

    Unlike :func:`_set_memory_metadata`, which stamps metadata onto trace
    events as they are generated, this records a dedicated ``annotate`` trace
    event for an allocation that already exists. This is useful when the
    information only becomes available after the allocation happened, e.g.
    noting that a tensor was packed into the autograd graph. Metadata recorded
    at allocation time is not modified; annotations accumulate as separate
    trace entries keyed to the allocation's address, each carrying the
    annotation string as its ``user_metadata`` and a stack trace of the
    annotation site (subject to the ``context`` setting of
    :func:`_record_memory_history`). The pytorch.org/memory_viz visualizer
    shows annotations alongside the allocation's own metadata.

    The annotation is keyed to the base address of the tensor's untyped
    storage, so it works for views and tensors with a nonzero
    ``storage_offset``.

    If memory history recording is not enabled (see
    :func:`_record_memory_history`), this has no observable effect. It is only
    supported by the native caching allocator; with other backends (e.g.
    ``cudaMallocAsync`` or a pluggable allocator) it is silently ignored.

    Args:
        tensor (torch.Tensor): CUDA tensor whose backing allocation to
                               annotate.
        metadata (str): Annotation string to record.
    """
    if not tensor.is_cuda:
        raise ValueError(f"expected a CUDA tensor, got device {tensor.device}")
    # pyrefly: ignore [missing-attribute]
    torch._C._cuda_annotateMemory(tensor.untyped_storage().data_ptr(), metadata)


def _save_segment_usage(filename="output.svg", snapshot=None):
    if snapshot is None:
        snapshot = _snapshot()
    with open(filename, "w") as f:
        f.write(_segments(snapshot))


def _save_memory_usage(filename="output.svg", snapshot=None):
    if snapshot is None:
        snapshot = _snapshot()
    with open(filename, "w") as f:
        f.write(_memory(snapshot))


@deprecated(
    "torch.cuda._set_allocator_settings is deprecated. Use torch._C._accelerator_setAllocatorSettings instead.",
    category=FutureWarning,
)
def _set_allocator_settings(env: str):
    # pyrefly: ignore [missing-attribute]
    return torch._C._accelerator_setAllocatorSettings(env)


def get_allocator_backend() -> str:
    r"""Return a string describing the active allocator backend as set by
    ``PYTORCH_ALLOC_CONF``. Currently available backends are
    ``native`` (PyTorch's native caching allocator) and `cudaMallocAsync``
    (CUDA's built-in asynchronous allocator).

    .. note::
        See :ref:`cuda-memory-management` for details on choosing the allocator backend.
    """
    return torch._C._cuda_getAllocatorBackend()


class _CUDAAllocator:
    r"""Wrapper over internal CUDA memory allocators."""

    def __init__(self, allocator: torch._C._cuda_CUDAAllocator):
        self._allocator = allocator

    def allocator(self):
        return self._allocator


class CUDAPluggableAllocator(_CUDAAllocator):
    r"""CUDA memory allocator loaded from a so file."""

    def __init__(self, path_to_so_file: str, alloc_fn_name: str, free_fn_name: str):
        r"""Memory allocators are compiled in .so files and loaded dynamically using ctypes.

        To change the active allocator use the :func:`torch.memory.cuda.change_current_allocator` function.

        Args:
            path_to_so_file(str): Path in the filesystem to the `.so` file containing
                the allocator functions
            alloc_fn_name(str): Name of the function to perform the memory allocation
                in the so file. The signature must be:
                void* alloc_fn_name(ssize_t size, int device, cudaStream_t stream);
            free_fn_name(str): Name of the function to perform the memory release
                in the so file. The signature must be:
                void free_fn_name(void* ptr, size_t size, cudaStream_t stream);

        .. warning::
            This is currently supported only in unix OSs

        .. note::
            See :ref:`cuda-memory-management` for details on creating and using a custom allocator
        """
        allocator = ctypes.CDLL(path_to_so_file)
        alloc_fn = ctypes.cast(getattr(allocator, alloc_fn_name), ctypes.c_void_p).value
        free_fn = ctypes.cast(getattr(allocator, free_fn_name), ctypes.c_void_p).value
        if alloc_fn is None:
            raise AssertionError(f"alloc_fn '{alloc_fn_name}' is None")
        if free_fn is None:
            raise AssertionError(f"free_fn '{free_fn_name}' is None")
        self._allocator = torch._C._cuda_customAllocator(alloc_fn, free_fn)


def change_current_allocator(allocator: _CUDAAllocator) -> None:
    r"""Change the currently used memory allocator to be the one provided.

    If the current allocator has already been used/initialized, this function will error.


    Args:
        allocator (torch.cuda.memory._CUDAAllocator): allocator to be set as the active one.
    .. note::
        See :ref:`cuda-memory-management` for details on creating and using a custom allocator
    """
    torch._C._cuda_changeCurrentAllocator(allocator.allocator())


def _get_current_allocator() -> _CUDAAllocator:
    r"""Return the allocator being currently used.

    .. note::
        See :ref:`cuda-memory-management` for details on creating and using a custom allocator
    """
    return _CUDAAllocator(torch._C._cuda_getAllocator())


class MemPool(_MemPool):
    r"""MemPool represents a pool of memory in a caching allocator. Currently,
    it's just the ID of the pool object maintained in the CUDACachingAllocator.

    Args:
        allocator(torch._C._cuda_CUDAAllocator, optional): a
            torch._C._cuda_CUDAAllocator object that can be used to
            define how memory gets allocated in the pool. If :attr:`allocator`
            is ``None`` (default), memory allocation follows the default/
            current configuration of the CUDACachingAllocator.
        use_on_oom(bool): a bool that indicates if this pool can be used
            as a last resort if a memory allocation outside of the pool fails due
            to Out Of Memory. This is False by default.
        no_split(bool): a bool that indicates if this pool should not split a segment.
            This is False by default.
    """

    def __init__(
        self,
        allocator: _cuda_CUDAAllocator | None = None,
        use_on_oom: bool = False,
        no_split: bool = False,
    ):
        # pyrefly: ignore [bad-argument-count]
        super().__init__(allocator, True, use_on_oom, no_split)

    @property
    def id(self) -> tuple[int, int]:
        r"""Returns the ID of this pool as a tuple of two ints."""
        return super().id

    def use_count(self) -> int:  # pylint: disable=useless-parent-delegation
        r"""Returns the reference count of this pool."""
        return super().use_count()

    def snapshot(self, include_traces=True):
        r"""Return a snapshot of the CUDA memory allocator pool state across all
        devices.

        Interpreting the output of this function requires familiarity with the
        memory allocator internals.

        Args:
            include_traces: Whether to include trace entries in the snapshot.
                If True (default), all trace entries are included.
                If False, no trace entries are included (lightweight/fast snapshot).

        .. note::
            See :ref:`cuda-memory-management` for more details about GPU memory
            management.
        """
        snapshot = torch.cuda.memory_snapshot(self.id, include_traces=include_traces)
        return snapshot


@contextlib.contextmanager
def use_mem_pool(pool: MemPool, device: "Device" = None):
    r"""A context manager that routes allocations to a given pool.

    Args:
        pool(torch.cuda.MemPool): a MemPool object to be made active so that
            allocations route to this pool.
        device (torch.device or int, optional): selected device. Uses MemPool on
            the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        This context manager makes only current thread's allocations route to
        the given pool. If a new thread is spawned inside the context manager
        (e.g. by calling backward) the allocations in that thread will not
        route to the given pool.

    .. note::
        When used during :class:`~torch.cuda.CUDAGraph` capture, the graph
        retains the pool until the graph is reset or destroyed.
    """
    device_index = (
        torch.cuda.current_device() if device is None else _get_device_index(device)
    )
    _cuda_beginAllocateCurrentThreadToPool(device_index, pool.id)
    try:
        if torch.cuda.is_current_stream_capturing():
            graph = torch.cuda.CUDAGraph.get_currently_capturing_graph()
            graph._retain_pool(pool)
        yield
    finally:
        _cuda_endAllocateToPool(device_index, pool.id)
        _cuda_releasePool(device_index, pool.id)


# Process-lifetime singleton: the UVM pluggable allocator + the ctypes closures
# behind it, lazily built (lock-guarded) by _make_uvm_allocator and reused.
_UVM_ALLOCATOR = None
_UVM_ALLOCATOR_LOCK = threading.Lock()


def _make_uvm_allocator():
    r"""Build the UVM ``CUDAPluggableAllocator`` and the ctypes closures behind it.

    Returns ``(c_alloc, c_free, allocator)``. UVM alloc/free are stateless (device
    is passed per call), so one allocator serves all calls and devices. Raises
    ``ImportError`` if ``cuda-python`` is unavailable.
    """
    import logging
    import traceback

    try:
        from cuda.bindings import runtime as _rt  # pyrefly: ignore[missing-import]
    except ImportError:
        raise ImportError(
            "torch.cuda._use_uvm() requires the 'cuda-python' package "
            "(cuda.bindings.runtime) for cudaMallocManaged, cudaMemAdvise, "
            "and cudaFree."
        ) from None

    log = logging.getLogger(__name__)

    _ALLOC_FN = ctypes.CFUNCTYPE(
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p
    )
    _FREE_FN = ctypes.CFUNCTYPE(
        None, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p
    )

    def _check(result, msg: str = ""):
        err = result if not isinstance(result, tuple) else result[0]
        if err != _rt.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA error: {err}. {msg}")

    _advise_uses_struct = hasattr(_rt, "cudaMemLocation")

    def _mem_advise(ptr, size, advice, device_id, _runtime=_rt):
        """Call cudaMemAdvise, handling struct-vs-int API difference.

        cuda-bindings 13.x requires a cudaMemLocation struct;
        cuda-bindings 12.x expects a plain int device ordinal.
        We try struct first and latch to int on TypeError.
        """
        nonlocal _advise_uses_struct
        if _advise_uses_struct:
            try:
                loc = _runtime.cudaMemLocation()
                loc.type = _runtime.cudaMemLocationType.cudaMemLocationTypeDevice
                loc.id = device_id
                return _runtime.cudaMemAdvise(ptr, size, advice, loc)
            except TypeError:
                _advise_uses_struct = False
        return _runtime.cudaMemAdvise(ptr, size, advice, device_id)

    _uvm_advise_supported_cache: dict[tuple[int, int], bool] = {}

    def _device_supports_uvm_advise(device_id, _runtime=_rt):
        cache_key = (device_id, id(_runtime))
        if cache_key in _uvm_advise_supported_cache:
            return _uvm_advise_supported_cache[cache_key]
        attr = getattr(
            _runtime.cudaDeviceAttr, "cudaDevAttrConcurrentManagedAccess", None
        )
        if attr is None:
            supported = False
        else:
            result = _runtime.cudaDeviceGetAttribute(attr, device_id)
            err = result if not isinstance(result, tuple) else result[0]
            if err != _runtime.cudaError_t.cudaSuccess:
                supported = False
            else:
                supported = bool(result[1])
        _uvm_advise_supported_cache[cache_key] = supported
        return supported

    def _uvm_alloc(size, device, stream, _runtime=_rt):
        try:
            err, ptr = _runtime.cudaMallocManaged(size, _runtime.cudaMemAttachGlobal)
            _check(err, f"cudaMallocManaged({size})")
            ptr = int(ptr)
            if device >= 0 and _device_supports_uvm_advise(device, _runtime):
                _check(
                    _mem_advise(
                        ptr,
                        size,
                        _runtime.cudaMemoryAdvise.cudaMemAdviseSetPreferredLocation,
                        device,
                    ),
                    "cudaMemAdvise(SetPreferredLocation)",
                )
                _check(
                    _mem_advise(
                        ptr,
                        size,
                        _runtime.cudaMemoryAdvise.cudaMemAdviseSetAccessedBy,
                        device,
                    ),
                    "cudaMemAdvise(SetAccessedBy)",
                )
            return ptr
        except Exception:
            log.error(
                "[_use_uvm] FAILED to allocate %d bytes (%.2f GiB) via UVM."
                " CUDACachingAllocator will raise an OOM error as a result."
                " You can ignore free-memory numbers reported by PyTorch"
                " as they are irrelevant for UVM.\nException:\n%s",
                size,
                size / (1024**3),
                traceback.format_exc(),
            )
            return 0

    def _uvm_free(ptr, size, device, stream, _runtime=_rt):
        """Best-effort free; guards against interpreter shutdown."""
        try:
            if ptr:
                _check(_runtime.cudaFree(ptr))
        except Exception:
            if log is not None and traceback is not None:
                try:
                    log.error(
                        "[_use_uvm] exception in free:\n%s",
                        traceback.format_exc(),
                    )
                except Exception:
                    pass

    c_alloc = _ALLOC_FN(_uvm_alloc)
    c_free = _FREE_FN(_uvm_free)
    alloc_ptr = ctypes.cast(c_alloc, ctypes.c_void_p).value
    free_ptr = ctypes.cast(c_free, ctypes.c_void_p).value
    # pyrefly: ignore[bad-argument-type]
    allocator = torch._C._cuda_customAllocator(alloc_ptr, free_ptr)
    return c_alloc, c_free, allocator


@contextlib.contextmanager
def _use_uvm(device: "Device" = None):
    r"""A context manager that routes CUDA allocations through ``cudaMallocManaged`` (UVM).

    All tensors allocated inside this context use CUDA Unified Virtual Memory,
    which allows oversubscribing GPU device memory by transparently paging to
    system RAM on demand. Numerics are identical to regular device allocations;
    only performance is affected due to page migration overhead.

    Args:
        device (torch.device or int, optional): selected device. Uses the
            current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Example::

        >>> # xdoctest: +SKIP(reason="requires CUDA and cuda-python")
        >>> with torch.cuda._use_uvm():
        ...     x = torch.randn(1024, 1024, device="cuda")
        ...     y = x @ x.T  # computed on GPU, pages in/out as needed

    .. note::
        Only the current thread's allocations are routed to managed memory.
        Allocations in threads spawned inside this context (e.g. by backward)
        will use the default allocator. Use
        ``torch.autograd.set_multithreading_enabled(False)`` to force backward
        onto the calling thread so that backward allocations also use UVM.

    .. note::
        Requires the ``cuda-python`` package (``cuda.bindings``).
    """
    # A tensor's block can stay cached in the PrivatePool after _use_uvm() returns
    # and be freed later by a global empty_cache(); per-call ctypes closures would
    # be GC'd by then, dangling the free callback. Build once and reuse instead.
    global _UVM_ALLOCATOR
    if _UVM_ALLOCATOR is None:
        with _UVM_ALLOCATOR_LOCK:
            if _UVM_ALLOCATOR is None:
                _UVM_ALLOCATOR = _make_uvm_allocator()
    allocator = _UVM_ALLOCATOR[2]

    pool = MemPool(allocator=allocator)
    with use_mem_pool(pool, device=device):
        yield pool
<|endoftext|># mypy: allow-untyped-defs
r"""This package adds support for NVIDIA Tools Extension (NVTX) used in profiling."""

from contextlib import contextmanager


try:
    from torch._C import _nvtx
except ImportError:

    class _NVTXStub:
        @staticmethod
        def _fail(*args, **kwargs):
            raise RuntimeError(
                "NVTX functions not installed. Are you sure you have a CUDA build?"
            )

        rangePushA = _fail
        rangePop = _fail
        markA = _fail

    _nvtx = _NVTXStub()  # type: ignore[assignment]

__all__ = ["range_push", "range_pop", "range_start", "range_end", "mark", "range"]


def range_push(msg):
    """
    Push a range onto a stack of nested range span.  Returns zero-based depth of the range that is started.

    Args:
        msg (str): ASCII message to associate with range
    """
    return _nvtx.rangePushA(msg)


def range_pop():
    """Pop a range off of a stack of nested range spans.  Returns the  zero-based depth of the range that is ended."""
    return _nvtx.rangePop()


def range_start(msg) -> int:
    """
    Mark the start of a range with string message. It returns a unique handle
    for this range to pass to the corresponding call to rangeEnd().

    A key difference between this and range_push/range_pop is that the
    range_start/range_end version supports range across threads (start on one
    thread and end on another thread).

    Returns: A range handle (uint64_t) that can be passed to range_end().

    Args:
        msg (str): ASCII message to associate with the range.
    """
    # pyrefly: ignore [missing-attribute]
    return _nvtx.rangeStartA(msg)


def range_end(range_id) -> None:
    """
    Mark the end of a range for a given range_id.

    Args:
        range_id (int): a unique handle for the start range.
    """
    # pyrefly: ignore [missing-attribute]
    _nvtx.rangeEnd(range_id)


def _device_range_start(msg: str, stream: int = 0) -> object:
    """
    Marks the start of a range with string message.
    It returns an opaque heap-allocated handle for this range
    to pass to the corresponding call to device_range_end().

    A key difference between this and range_start is that the
    range_start marks the range right away, while _device_range_start
    marks the start of the range as soon as all the tasks on the
    CUDA stream are completed.

    Returns: An opaque heap-allocated handle that should be passed to _device_range_end().

    Args:
        msg (str): ASCII message to associate with the range.
        stream (int): CUDA stream id.
    """
    # pyrefly: ignore [missing-attribute]
    return _nvtx.deviceRangeStart(msg, stream)


def _device_range_end(range_handle: object, stream: int = 0) -> None:
    """
    Mark the end of a range for a given range_handle as soon as all the tasks
    on the CUDA stream are completed.

    Args:
        range_handle: a unique handle for the start range.
        stream (int): CUDA stream id.
    """
    # pyrefly: ignore [missing-attribute]
    _nvtx.deviceRangeEnd(range_handle, stream)


def mark(msg):
    """
    Describe an instantaneous event that occurred at some point.

    Args:
        msg (str): ASCII message to associate with the event.
    """
    return _nvtx.markA(msg)


@contextmanager
def range(msg, *args, **kwargs):
    """
    Context manager / decorator that pushes an NVTX range at the beginning
    of its scope, and pops it at the end. If extra arguments are given,
    they are passed as arguments to msg.format().

    Args:
        msg (str): message to associate with the range
    """
    range_push(msg.format(*args, **kwargs))
    try:
        yield
    finally:
        range_pop()
<|endoftext|>import torch
from torch._C import dtype


__all__ = ["GPULimits"]


class GPULimits:
    r"""Utility class that provides the theoretical limits of Nvidia GPU devices. The
    limits don't take into account thermal throttling (assume that the GPU run at its
    peak rated frequency). This is because user hardware configuration may influence
    power behavior.
    """

    def __init__(self, target_device: torch.device):
        # The device properties object is obtained by calling 'cudaGetDeviceProperties' CUDA
        # runtime function. We need the total memory bus width and the memory clock rate to
        # calculate the memory bandwidth.
        self.device_properties = torch.cuda.get_device_properties(target_device)

        # The compute capability is needed to determine the number of FLOPs per cycle per SM
        self.compute_capability = int(
            f"{self.device_properties.major}{self.device_properties.minor}"
        )

    # FLOPs per cycle information derived from Table 2 in:
    # https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c

    # Returns the number of FMA instructions retired per cycle per SM for a given
    # data type, when tensor cores are NOT used
    def get_fma_per_cycle_per_sm_cuda_cores(self, data_type: dtype) -> int:
        hardcoded_device_values = {
            # Ampere Architecture
            "fp16_80": 256,
            "fp32_80": 64,
            "fp64_80": 32,
            # Hopper Architecture
            "fp16_90": 64,
            "fp32_90": 128,
            "fp64_90": 64,
            # Blackwell Architecture
            "fp16_100": 256,
            "fp32_100": 128,
            "fp64_100": 64,
        }
        dict_key = ""
        if data_type is torch.float16:
            dict_key = f"fp16_{self.compute_capability}"
        elif data_type is torch.float32:
            dict_key = f"fp32_{self.compute_capability}"
        elif data_type is torch.float64:
            dict_key = f"fp64_{self.compute_capability}"
        else:
            dict_key = "unknown"

        if dict_key not in hardcoded_device_values:
            raise RuntimeError(
                f"No data for sm_{self.compute_capability} and {data_type}."
            )

        return hardcoded_device_values[dict_key]

    # Returns the number of FMA instructions retired per cycle per SM for a given
    # data type, when tensor cores ARE used
    def get_fma_per_cycle_per_sm_tensor_cores(self, data_type: dtype) -> int:
        hardcoded_device_values = {
            # Ampere Architecture
            "int8_80": 2048,
            "fp16_80": 1024,
            "fp32_80": 512,
            "fp64_80": 64,
            # Hopper Architecture
            "int8_90": 4096,
            "fp8_90": 4096,
            "fp16_90": 2048,
            "fp32_90": 1024,
            "fp64_90": 128,
            # Blackwell Architecture
            "int8_100": 8192,
            "fp8_100": 8192,
            "fp16_100": 4096,
            "fp32_100": 2048,
        }
        dict_key = ""
        if data_type is torch.float16:
            dict_key = f"fp16_{self.compute_capability}"
        elif data_type is torch.bfloat16:
            # FP16 and BF16 are equivalent in terms of FLOPs per cycle per SM
            dict_key = f"fp16_{self.compute_capability}"
        elif data_type is torch.float32:
            dict_key = f"fp32_{self.compute_capability}"
        elif data_type is torch.int8:
            dict_key = f"int8_{self.compute_capability}"
        elif data_type is torch.float64:
            dict_key = f"fp64_{self.compute_capability}"
        else:
            dict_key = "unknown"

        if dict_key not in hardcoded_device_values:
            raise RuntimeError(
                f"No data for sm_{self.compute_capability} and {data_type}."
            )

        return hardcoded_device_values[dict_key]

    def get_tflops_per_second(
        self, data_type: dtype, use_tensor_cores: bool = True
    ) -> float:
        num_sms = self.device_properties.multi_processor_count
        clock_rate = self.device_properties.clock_rate  # KHz

        fma_per_cycle = 0
        if use_tensor_cores:
            fma_per_cycle = self.get_fma_per_cycle_per_sm_tensor_cores(data_type)
        else:
            fma_per_cycle = self.get_fma_per_cycle_per_sm_cuda_cores(data_type)

        # 1 FMA counts as 2 floating point operations
        # Clock rate is in KHz
        tflops_per_second = num_sms * fma_per_cycle * 2 * clock_rate / 1e9
        return tflops_per_second

    def get_memory_bandwidth_Bps(self) -> int:
        # DRAM devices are Double-Data which means they provide an output at both fronts of
        # a clock beat
        bus_bytes_per_cycle = int(2 * self.device_properties.memory_bus_width / 8)
        mem_clock_rate_Hz = self.device_properties.memory_clock_rate * 1000
        bytes_per_second = bus_bytes_per_cycle * mem_clock_rate_Hz
        return bytes_per_second

    def get_shared_memory_bandwidth_Bps(self) -> int:
        # Each warp can LD or ST 32 x 4 bytes per cycle. To calculate the
        # device's throughput we need to multiply with frequency and number of SMs.
        num_sms = self.device_properties.multi_processor_count
        bytes_per_cycle_per_sm = 128
        bytes_per_cycle_per_device = num_sms * bytes_per_cycle_per_sm
        bytes_per_second = (
            bytes_per_cycle_per_device * self.device_properties.clock_rate * 1000
        )
        return bytes_per_second
<|endoftext|># mypy: allow-untyped-defs
r"""
This module introduces CUDA Sanitizer, a tool for detecting synchronization errors between kernels ran on different streams.

It stores information on accesses to tensors to determine if they are synchronized
or not. When enabled in a python program and a possible data race is detected, a
detailed warning will be printed and the program will exit.

It can be enabled either by importing this module and calling
:func:`enable_cuda_sanitizer()` or by exporting the ``TORCH_CUDA_SANITIZER``
environment variable.
"""

import enum
import functools
import inspect
import io
import logging
import re
import sys
import textwrap
import traceback
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, TypeVar

import torch
import torch.cuda._gpu_trace as gpu_trace
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode


aten = torch.ops.aten

DEFAULT_STREAM_ID = 0

TK = TypeVar("TK")
TVa = TypeVar("TVa")
TVb = TypeVar("TVb")

DataPtr = int
StreamId = int
EventId = int
SeqNum = int

logger = logging.getLogger(__name__)

# Note that this is only factories that take Tensor as input as they are
# the ones we care about.
FACTORY_FUNCTION_REGEX = re.compile("(new_.*|.*_like)")


class AccessType(enum.Enum):
    READ = enum.auto()
    WRITE = enum.auto()

    def __str__(self):
        return "reading from" if self is AccessType.READ else "writing to"


@dataclass
class Access:
    r"""Stores information about a single access to a tensor by a kernel.

    Args:
        type: either AccessType.READ or AccessType.Write.
        seq_num: the sequential number of the kernel performing the access.
        stream: the stream id of the stream executing the kernel.
        operator: the schema of the launched kernel, which lists the
            arguments and return type.
        aliases: the arguments in the schema this access corresponds to.
        is_output: Whether the tensor was an output of the kernel.
        stack_trace: the stack summary object captured during access.
    """

    type: AccessType
    seq_num: SeqNum
    stream: StreamId
    operator: str
    aliases: list[str]
    is_output: bool
    stack_trace: traceback.StackSummary


class SynchronizationError(Exception):
    """Base class for errors detected by CUDA Sanitizer."""


class UnsynchronizedAccessError(SynchronizationError):
    """Stores information about two unsynchronized accesses to one data pointer."""

    def __init__(
        self,
        data_ptr: DataPtr,
        allocation_stack_trace: traceback.StackSummary | None,
        current_access: Access,
        previous_access: Access,
    ):
        self.data_ptr = data_ptr
        self.allocation_stack_trace = allocation_stack_trace
        self.current_access = current_access
        self.previous_access = previous_access

    def __str__(self):
        def format_access(access: Access):
            message.write(f"{access.operator}\n{access.type}")
            if access.aliases:
                message.write(" argument(s) " + ", ".join(access.aliases))
                if access.is_output:
                    message.write(", and to")
            if access.is_output:
                message.write(" the output")
            message.write(
                f"\nWith stack trace:\n{''.join(access.stack_trace.format())}\n"
            )

        with io.StringIO() as message:
            message.write(
                textwrap.dedent(
                    f"""\
                    ============================
                    CSAN detected a possible data race on tensor with data pointer {self.data_ptr}
                    Access by stream {self.current_access.stream} during kernel:
                    """
                )
            )
            format_access(self.current_access)

            message.write(
                f"Previous access by stream {self.previous_access.stream} during kernel:\n"
            )
            format_access(self.previous_access)

            if self.allocation_stack_trace:
                message.write(
                    "Tensor was allocated with stack trace:\n"
                    f"{''.join(self.allocation_stack_trace.format())}"
                )
            else:
                message.write("Trace for tensor allocation not found.")
            return message.getvalue()


class CUDASanitizerErrors(Exception):
    """Wrapper class for errors reported by CUDA Sanitizer."""

    def __init__(self, errors: list[SynchronizationError]):
        self.errors = errors

    def __str__(self):
        return f"detected {len(self.errors)} errors"


@dataclass
class TensorInfo:
    r"""Stores information about a single tensor and recent accesses to it.

    Args:
        allocation_stack_trace: the stack summary object captured during tensor
            allocation. Can be ``None`` if the allocation wasn't caught by CSAN.
        reads: list of read accesses to the tensor that were performed since
            the last write.
        write: the last write access to the tensor.
    """

    allocation_stack_trace: traceback.StackSummary | None
    reads: list[Access] = field(default_factory=list)
    write: Access | None = None


class _TensorsAccessed:
    def __init__(self) -> None:
        self.accesses: dict[DataPtr, TensorInfo] = {}

    def ensure_tensor_exists(self, data_ptr: DataPtr) -> None:
        if data_ptr not in self.accesses:
            logger.info(
                "Found tensor with pointer: %s, but no matching tensor "
                "allocation in the trace. Backfilling the trace now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
                data_ptr,
            )
            self.create_tensor(data_ptr, None)

    def ensure_tensor_does_not_exist(self, data_ptr: DataPtr) -> None:
        if data_ptr in self.accesses:
            logger.info(
                "Found duplicate tensor allocation in the trace for tensor with "
                "pointer: %s. Assuming the trace for tensor deallocation "
                "wasn't caught and backfilling it now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
                data_ptr,
            )
            self.delete_tensor(data_ptr)

    def create_tensor(
        self, data_ptr: DataPtr, stack_trace: traceback.StackSummary | None
    ) -> None:
        self.accesses[data_ptr] = TensorInfo(stack_trace)

    def delete_tensor(self, data_ptr: DataPtr) -> None:
        del self.accesses[data_ptr]

    def were_there_reads_since_last_write(self, data_ptr: DataPtr) -> bool:
        return bool(self.accesses[data_ptr].reads)

    def get_allocation_stack_trace(
        self, data_ptr: DataPtr
    ) -> traceback.StackSummary | None:
        return self.accesses[data_ptr].allocation_stack_trace

    def get_write(self, data_ptr: DataPtr) -> Access | None:
        return self.accesses[data_ptr].write

    def get_reads(self, data_ptr: DataPtr) -> list[Access]:
        return self.accesses[data_ptr].reads

    def add_read(self, data_ptr: DataPtr, access: Access) -> None:
        self.accesses[data_ptr].reads.append(access)

    def set_write(self, data_ptr: DataPtr, access: Access) -> None:
        self.accesses[data_ptr].write = access
        self.accesses[data_ptr].reads = []


class StreamSynchronizations:
    def __init__(self) -> None:
        self.current_sync_states: dict[StreamId, dict[StreamId, SeqNum]] = {}
        self.recorded_sync_states: dict[EventId, dict[StreamId, SeqNum]] = {}
        self.host_sync_state: dict[StreamId, SeqNum] = {}
        self.create_stream(DEFAULT_STREAM_ID)

    def _ensure_stream_exists(self, stream: StreamId) -> None:
        if stream not in self.current_sync_states:
            logger.info(
                "Found Stream with id: %s, but no matching stream "
                "creation in the trace. Backfilling the trace now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
                stream,
            )
            self.create_stream(stream)

    def _ensure_event_exists(self, event: EventId) -> None:
        if event not in self.recorded_sync_states:
            logger.info(
                "Found Event with id: %s, but no matching event "
                "creation in the trace. Backfilling the trace now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
                event,
            )
            self.create_event(event)

    def _ensure_event_does_not_exist(self, event: EventId) -> None:
        if event in self.recorded_sync_states:
            logger.info(
                "Found duplicate event creation in the trace for event with "
                "id: %s. Assuming the trace for event deletion wasn't caught "
                "and backfilling it now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
                event,
            )
            self.delete_event(event)

    def create_stream(self, stream: StreamId) -> None:
        if stream in self.current_sync_states:
            logger.info(
                "Found duplicate Stream creation in the trace for Stream with "
                "id: %s. PyTorch Streams are only created once, so this "
                "trace entry is ignored.",
                stream,
            )
        else:
            self.host_sync_state[stream] = 0
            self.current_sync_states[stream] = self.host_sync_state.copy()

    def create_event(self, event: EventId) -> None:
        self._ensure_event_does_not_exist(event)
        self.recorded_sync_states[event] = {}

    def delete_event(self, event: EventId) -> None:
        self._ensure_event_exists(event)
        del self.recorded_sync_states[event]

    def update_seq_num(self, stream: StreamId, seq_num: SeqNum) -> None:
        self._ensure_stream_exists(stream)
        self.current_sync_states[stream][stream] = seq_num

    def record_state(self, event: EventId, stream: StreamId) -> None:
        self._ensure_event_exists(event)
        self._ensure_stream_exists(stream)
        self.recorded_sync_states[event] = self.current_sync_states[stream].copy()

    def _state_wait_for_other(
        self, state: dict[StreamId, SeqNum], other: dict[StreamId, SeqNum]
    ) -> None:
        for stream, seq_num in other.items():
            state[stream] = max(state.get(stream, -1), seq_num)

    def stream_wait_for_event(self, stream: StreamId, event: EventId) -> None:
        self._ensure_stream_exists(stream)
        self._ensure_event_exists(event)
        self._state_wait_for_other(
            self.current_sync_states[stream], self.recorded_sync_states[event]
        )

    def all_streams_wait_for_event(self, event: EventId) -> None:
        self._ensure_event_exists(event)
        for stream in self.current_sync_states:
            self.stream_wait_for_event(stream, event)

        self._state_wait_for_other(
            self.host_sync_state, self.recorded_sync_states[event]
        )

    def all_streams_wait_for_stream(self, stream: StreamId) -> None:
        self._ensure_stream_exists(stream)
        for state in self.current_sync_states.values():
            self._state_wait_for_other(state, self.current_sync_states[stream])

        self._state_wait_for_other(
            self.host_sync_state, self.current_sync_states[stream]
        )

    def sync_all_streams(self) -> None:
        for stream, state in self.current_sync_states.items():
            self.host_sync_state[stream] = state[stream]

        for state in self.current_sync_states.values():
            self._state_wait_for_other(state, self.host_sync_state)

    def is_ordered_after(
        self, current_stream: StreamId, seq_num: SeqNum, other_stream: StreamId
    ) -> bool:
        self._ensure_stream_exists(current_stream)
        self._ensure_stream_exists(other_stream)
        return seq_num <= self.current_sync_states[current_stream].get(other_stream, -1)


class EventHandler:
    """Analyzes CSAN trace for synchronization errors.

    Stores information on each stream's synchronizations with other streams as well
    as tensor accesses to determine whether a given kernel launch might cause a
    data race.
    """

    def __init__(self) -> None:
        self.tensors_accessed = _TensorsAccessed()
        self.syncs = StreamSynchronizations()
        self.seq_num: SeqNum = 0

    def _handle_kernel_launch(
        self,
        stream: StreamId,
        read_only: set[DataPtr],
        read_write: set[DataPtr],
        outputs: set[DataPtr],
        operator: str,
        tensor_aliases: dict[int, list[str]],
    ) -> list[SynchronizationError]:
        def check_conflict(
            data_ptr: DataPtr, current_access: Access, previous_access: Access | None
        ) -> None:
            if previous_access is None:
                return
            if not self.syncs.is_ordered_after(
                current_access.stream, previous_access.seq_num, previous_access.stream
            ):
                error_list.append(
                    UnsynchronizedAccessError(
                        data_ptr,
                        self.tensors_accessed.get_allocation_stack_trace(data_ptr),
                        current_access,
                        previous_access,
                    )
                )

        error_list: list[SynchronizationError] = []
        self.seq_num += 1
        self.syncs.update_seq_num(stream, self.seq_num)
        stack_trace = traceback.StackSummary.extract(
            traceback.walk_stack(inspect.currentframe()), lookup_lines=False
        )
        # The stack trace generated in this way is in the inverse order, so it must be
        # reversed.
        stack_trace.reverse()

        for data_ptr in read_only:
            self.tensors_accessed.ensure_tensor_exists(data_ptr)
            current_access = Access(
                AccessType.READ,
                self.seq_num,
                stream,
                operator,
                tensor_aliases[data_ptr],
                data_ptr in outputs,
                stack_trace,
            )
            check_conflict(
                data_ptr, current_access, self.tensors_accessed.get_write(data_ptr)
            )
            self.tensors_accessed.add_read(data_ptr, current_access)

        for data_ptr in read_write:
            self.tensors_accessed.ensure_tensor_exists(data_ptr)
            current_access = Access(
                AccessType.WRITE,
                self.seq_num,
                stream,
                operator,
                tensor_aliases[data_ptr],
                data_ptr in outputs,
                stack_trace,
            )
            if self.tensors_accessed.were_there_reads_since_last_write(data_ptr):
                for previous_access in self.tensors_accessed.get_reads(data_ptr):
                    check_conflict(data_ptr, current_access, previous_access)
            else:
                check_conflict(
                    data_ptr, current_access, self.tensors_accessed.get_write(data_ptr)
                )
            self.tensors_accessed.set_write(data_ptr, current_access)

        return error_list

    def _handle_event_creation(self, event: EventId) -> None:
        self.syncs.create_event(event)

    def _handle_event_deletion(self, event: EventId) -> None:
        self.syncs.delete_event(event)

    def _handle_event_record(self, event: EventId, stream: StreamId) -> None:
        self.syncs.record_state(event, stream)

    def _handle_event_wait(self, event: EventId, stream: StreamId) -> None:
        self.syncs.stream_wait_for_event(stream, event)

    def _handle_memory_allocation(self, data_ptr: DataPtr) -> None:
        self.tensors_accessed.ensure_tensor_does_not_exist(data_ptr)
        stack_trace = traceback.StackSummary.extract(
            traceback.walk_stack(inspect.currentframe()), lookup_lines=False
        )
        # The stack trace generated in this way is in the inverse order, so it must be
        # reversed.
        stack_trace.reverse()
        self.tensors_accessed.create_tensor(
            data_ptr,
            stack_trace,
        )

    def _handle_memory_deallocation(self, data_ptr: DataPtr) -> None:
        self.tensors_accessed.ensure_tensor_exists(data_ptr)
        self.tensors_accessed.delete_tensor(data_ptr)

    def _handle_stream_creation(self, stream: StreamId) -> None:
        self.syncs.create_stream(stream)

    def _handle_device_synchronization(self) -> None:
        self.syncs.sync_all_streams()

    def _handle_stream_synchronization(self, stream: StreamId) -> None:
        self.syncs.all_streams_wait_for_stream(stream)

    def _handle_event_synchronization(self, event: EventId) -> None:
        self.syncs.all_streams_wait_for_event(event)


def zip_by_key(a: dict[TK, TVa], b: dict[TK, TVb]) -> Iterator[tuple[TK, TVa, TVb]]:
    for arg, value in a.items():
        if arg in b:
            yield arg, value, b[arg]


def zip_arguments(
    schema: torch.FunctionSchema, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Iterator[tuple[torch.Argument, Any]]:
    schema_args = schema.arguments[: len(args)]
    schema_kwargs = {arg.name: arg for arg in schema.arguments[len(args) :]}

    yield from zip(schema_args, args)

    for _, argument, value in zip_by_key(schema_kwargs, kwargs):
        yield (argument, value)


class ArgumentHandler:
    def __init__(self) -> None:
        self.dataptrs_read: set[DataPtr] = set()
        self.dataptrs_written: set[DataPtr] = set()
        self.tensor_aliases: dict[DataPtr, list[str]] = {}
        self.outputs: set[DataPtr] = set()

    def _handle_argument(
        self,
        value: Any,
        is_write: bool,
        metadata_only: bool,
        name: str | None = None,
        is_output: bool = False,
    ) -> None:
        if isinstance(value, torch.Tensor) and value.is_cuda:
            # data_ptr() is preferred, but distinguish Tensors with null data_ptr()
            # otherwise two empty Tensors could incorrectly match as a conflict
            data_ptr = value.data_ptr() if value.data_ptr() else id(value)
            if is_write:
                self.dataptrs_written.add(data_ptr)
            elif not metadata_only:
                self.dataptrs_read.add(data_ptr)

            self.tensor_aliases.setdefault(data_ptr, [])
            if name is not None:
                self.tensor_aliases[data_ptr].append(name)
            if is_output:
                self.outputs.add(data_ptr)

    def parse_inputs(
        self,
        schema: torch.FunctionSchema,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        is_factory: bool,
    ) -> None:
        for argument, value in zip_arguments(schema, args, kwargs):
            is_write = argument.alias_info is not None and argument.alias_info.is_write
            # A change is metadata only if it is a view or a factory function that
            # reads only metadata
            metadata_only = is_factory or (
                argument.alias_info is not None and not argument.alias_info.is_write
            )
            pytree.tree_map_(
                functools.partial(
                    self._handle_argument,
                    is_write=is_write,
                    name=argument.name,
                    metadata_only=metadata_only,
                ),
                value,
            )

    def parse_outputs(
        self, schema: torch.FunctionSchema, outputs: Any, *, is_factory: bool
    ) -> None:
        for res, value in zip(schema.returns, (outputs,)):
            metadata_only = is_factory or (
                res.alias_info is not None and not res.alias_info.is_write
            )
            pytree.tree_map_(
                functools.partial(
                    self._handle_argument,
                    is_write=not metadata_only,
                    is_output=True,
                    metadata_only=metadata_only,
                ),
                value,
            )


class CUDASanitizerDispatchMode(TorchDispatchMode):
    def __init__(self) -> None:
        self.event_handler = EventHandler()
        torch._C._activate_gpu_trace()
        gpu_trace.register_callback_for_event_creation(
            self.event_handler._handle_event_creation
        )
        gpu_trace.register_callback_for_event_deletion(
            self.event_handler._handle_event_deletion
        )
        gpu_trace.register_callback_for_event_record(
            self.event_handler._handle_event_record
        )
        gpu_trace.register_callback_for_event_wait(
            self.event_handler._handle_event_wait
        )
        gpu_trace.register_callback_for_memory_allocation(
            self.event_handler._handle_memory_allocation
        )
        gpu_trace.register_callback_for_memory_deallocation(
            self.event_handler._handle_memory_deallocation
        )
        gpu_trace.register_callback_for_stream_creation(
            self.event_handler._handle_stream_creation
        )
        gpu_trace.register_callback_for_device_synchronization(
            self.event_handler._handle_device_synchronization
        )
        gpu_trace.register_callback_for_stream_synchronization(
            self.event_handler._handle_stream_synchronization
        )
        gpu_trace.register_callback_for_event_synchronization(
            self.event_handler._handle_event_synchronization
        )

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # record_stream is not a kernel dispatch, skip it
        if func is aten.record_stream.default:
            return func(*args, **kwargs)

        is_factory = bool(FACTORY_FUNCTION_REGEX.match(func._schema.name))

        argument_handler = ArgumentHandler()
        argument_handler.parse_inputs(func._schema, args, kwargs, is_factory=is_factory)

        outputs = func(*args, **kwargs)

        argument_handler.parse_outputs(func._schema, outputs, is_factory=is_factory)
        errors = self.event_handler._handle_kernel_launch(
            torch.cuda.current_stream().cuda_stream,
            argument_handler.dataptrs_read - argument_handler.dataptrs_written,
            argument_handler.dataptrs_written,
            argument_handler.outputs,
            func._schema,
            argument_handler.tensor_aliases,
        )
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            raise CUDASanitizerErrors(errors)

        return outputs


class CUDASanitizer:
    """Manages the lifetime of a CUDASanitizer dispatch mode object.

    The CUDASanitizer class wraps the entering/exiting functions of the dispatch mode
    context manager in the enable function/destructor, respectively. This is to
    explicitly set the lifetime of the dispatch mode object to that of the application.
    This approach was deemed more elegant than using the atexit module.
    """

    def __init__(self) -> None:
        self.dispatch = CUDASanitizerDispatchMode()
        self.enabled = False

    def enable(self):
        self.dispatch.__enter__()
        self.enabled = True

    def disable(self):
        self.dispatch.__exit__(None, None, None)
        self.enabled = False

    def __del__(self):
        # Since this object lifetime is linked to the `torch.cuda._sanitizer` python
        # module, it often gets deleted as part of the overall `torch` module cleanup
        # At that time, depending on CPython version, the torch.* module might be in
        # different states of being already cleaned up.
        # Similarly other imports might already have been cleaned up so `sys` might
        # be already gone as well.
        # Skip exiting the mode if it outlived the runtime.
        if (sys is not None) and (not sys.is_finalizing()) and self.enabled:
            self.disable()


def enable_cuda_sanitizer():
    """Enable CUDA Sanitizer.

    The sanitizer will begin to analyze low-level CUDA calls invoked by torch functions
    for synchronization errors. All data races found will be printed to the standard
    error output along with stack traces of suspected causes. For best results, the
    sanitizer should be enabled at the very beginning of the program.
    """
    cuda_sanitizer.enable()


cuda_sanitizer = CUDASanitizer()
<|endoftext|>import os
import sys
from collections.abc import Callable

import torch
from torch.types import Storage


__all__: list[str] = [
    "is_available",
    "gds_register_buffer",
    "gds_deregister_buffer",
    "GdsFile",
]


def _dummy_fn(name: str) -> Callable:
    def fn(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError(f"torch._C.{name} is not supported on this platform")

    return fn


def is_available() -> bool:
    r"""Return ``True`` if GDS (GPUDirect Storage) support is built in.

    This requires PyTorch to be built with cuFile (CUDA) or hipFile (ROCm).
    """
    return torch._C._has_gds


# Without GDS built in, install dummy stubs so the API raises a clear error.
if not is_available():
    if hasattr(torch._C, "_gds_deregister_buffer"):
        raise AssertionError(
            "_gds_deregister_buffer exists but _gds_register_buffer does not"
        )
    if hasattr(torch._C, "_gds_register_handle"):
        raise AssertionError(
            "_gds_register_handle exists but _gds_register_buffer does not"
        )
    if hasattr(torch._C, "_gds_deregister_handle"):
        raise AssertionError(
            "_gds_deregister_handle exists but _gds_register_buffer does not"
        )
    if hasattr(torch._C, "_gds_load_storage"):
        raise AssertionError(
            "_gds_load_storage exists but _gds_register_buffer does not"
        )
    if hasattr(torch._C, "_gds_save_storage"):
        raise AssertionError(
            "_gds_save_storage exists but _gds_register_buffer does not"
        )
    # Define functions
    torch._C.__dict__["_gds_register_buffer"] = _dummy_fn("_gds_register_buffer")
    torch._C.__dict__["_gds_deregister_buffer"] = _dummy_fn("_gds_deregister_buffer")
    torch._C.__dict__["_gds_register_handle"] = _dummy_fn("_gds_register_handle")
    torch._C.__dict__["_gds_deregister_handle"] = _dummy_fn("_gds_deregister_handle")
    torch._C.__dict__["_gds_load_storage"] = _dummy_fn("_gds_load_storage")
    torch._C.__dict__["_gds_save_storage"] = _dummy_fn("_gds_save_storage")


def gds_register_buffer(s: Storage) -> None:
    """Registers a storage on a CUDA device as a cufile buffer.

    Example::

        >>> # xdoctest: +SKIP("gds filesystem requirements")
        >>> src = torch.randn(1024, device="cuda")
        >>> s = src.untyped_storage()
        >>> gds_register_buffer(s)

    Args:
        s (Storage): Buffer to register.
    """
    torch._C._gds_register_buffer(s)


def gds_deregister_buffer(s: Storage) -> None:
    """Deregisters a previously registered storage on a CUDA device as a cufile buffer.

    Example::

        >>> # xdoctest: +SKIP("gds filesystem requirements")
        >>> src = torch.randn(1024, device="cuda")
        >>> s = src.untyped_storage()
        >>> gds_register_buffer(s)
        >>> gds_deregister_buffer(s)

    Args:
        s (Storage): Buffer to register.
    """
    torch._C._gds_deregister_buffer(s)


class GdsFile:
    r"""Wrapper around cuFile.

    cuFile is a file-like interface to the GPUDirect Storage (GDS) API.

    See the `cufile docs <https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufile-io-api>`_
    for more details.

    Args:
        filename (str): Name of the file to open.
        flags (int): Flags to pass to ``os.open`` when opening the file. ``os.O_DIRECT`` will
            be added automatically.

    Example::

        >>> # xdoctest: +SKIP("gds filesystem requirements")
        >>> src1 = torch.randn(1024, device="cuda")
        >>> src2 = torch.randn(2, 1024, device="cuda")
        >>> file = torch.cuda.gds.GdsFile(f, os.O_CREAT | os.O_RDWR)
        >>> file.save_storage(src1.untyped_storage(), offset=0)
        >>> file.save_storage(src2.untyped_storage(), offset=src1.nbytes)
        >>> dest1 = torch.empty(1024, device="cuda")
        >>> dest2 = torch.empty(2, 1024, device="cuda")
        >>> file.load_storage(dest1.untyped_storage(), offset=0)
        >>> file.load_storage(dest2.untyped_storage(), offset=src1.nbytes)
        >>> torch.equal(src1, dest1)
        True
        >>> torch.equal(src2, dest2)
        True

    """

    def __init__(self, filename: str, flags: int):
        if sys.platform == "win32":
            raise RuntimeError("GdsFile is not supported on this platform.")
        self.filename = filename
        self.flags = flags
        self.fd = os.open(filename, flags | os.O_DIRECT)  # type: ignore[attr-defined]
        self.handle: int | None = None
        self.register_handle()

    def __del__(self) -> None:
        if self.handle is not None:
            self.deregister_handle()
        os.close(self.fd)

    def register_handle(self) -> None:
        """Registers file descriptor to cuFile Driver.

        This is a wrapper around ``cuFileHandleRegister``.
        """
        if self.handle is not None:
            raise AssertionError("Cannot register a handle that is already registered.")
        self.handle = torch._C._gds_register_handle(self.fd)

    def deregister_handle(self) -> None:
        """Deregisters file descriptor from cuFile Driver.

        This is a wrapper around ``cuFileHandleDeregister``.
        """
        if self.handle is None:
            raise AssertionError("Cannot deregister a handle that is not registered.")
        torch._C._gds_deregister_handle(self.handle)
        self.handle = None

    def load_storage(self, storage: Storage, offset: int = 0) -> None:
        """Loads data from the file into the storage.

        This is a wrapper around ``cuFileRead``. ``storage.nbytes()`` of data
        will be loaded from the file at ``offset`` into the storage.

        Args:
            storage (Storage): Storage to load data into.
            offset (int, optional): Offset into the file to start loading from. (Default: 0)
        """
        if self.handle is None:
            raise AssertionError("Cannot load data from a file that is not registered.")
        torch._C._gds_load_storage(self.handle, storage, offset)

    def save_storage(self, storage: Storage, offset: int = 0) -> None:
        """Saves data from the storage into the file.

        This is a wrapper around ``cuFileWrite``. All bytes of the storage
        will be written to the file at ``offset``.

        Args:
            storage (Storage): Storage to save data from.
            offset (int, optional): Offset into the file to start saving to. (Default: 0)
        """
        if self.handle is None:
            raise AssertionError("Cannot save data to a file that is not registered.")
        torch._C._gds_save_storage(self.handle, storage, offset)
<|endoftext|># mypy: allow-untyped-defs
import re
from collections.abc import Callable

import torch
from torch import Tensor


__all__: list[str] = []


class _CodeParser:
    def __init__(self, code_string: str):
        optional_ws = r"\s*"
        required_ws = r"\s+"
        template_params = r"(?P<template_params>\<.+\>)"
        return_type = r"(?P<return_type>\w+)"
        function_name = r"(?P<function_name>\w+)"
        function_params = r"(?P<function_params>\(.+\))"
        function_body = r"(?P<function_body>\{.+\})"

        pattern = (
            optional_ws
            + "template"
            + optional_ws
            + template_params
            + optional_ws
            + return_type
            + required_ws
            + function_name
            + optional_ws
            + function_params
            + optional_ws
            + function_body
            + optional_ws
        )

        result = re.match(
            pattern, code_string, re.DOTALL
        )  # DOTALL for matching multiline

        if result is None:
            raise Exception(  # noqa: TRY002
                f"Couldn't parse code, please check correctness:\n {code_string}"
            )

        self.template_params = result["template_params"]
        self.return_type = result["return_type"]
        self.function_name = result["function_name"]
        self.function_params = result["function_params"]
        self.function_body = result["function_body"]


class _JittedFunction:
    def __init__(
        self, code_string: str, return_by_ref: bool, num_outputs: int, **kwargs
    ):
        self.code_string = code_string

        if not (return_by_ref or num_outputs == 1):
            raise AssertionError("Return by value only works for single output.")
        self.return_by_ref = return_by_ref
        self.num_outputs = num_outputs

        parsed_code = _CodeParser(code_string)
        self.kernel_name = parsed_code.function_name

        self.kwargs_dict = kwargs
        self.is_cuda_available = torch.cuda.is_available()

    def __call__(self, *tensors: Tensor, **kwargs):
        # Jiterator follow torch.cuda's lazy initialization behavior
        # Defer checking cuda's availability at the function invocation time
        if not self.is_cuda_available:
            raise AssertionError(
                "Jiterator is only supported on CUDA and ROCm GPUs, none are available."
            )

        if len(tensors) > 8:
            raise AssertionError(
                f"jiterator only supports up to 8 tensor inputs, got {len(tensors)}"
            )

        expanded_kwargs = self.kwargs_dict.copy()
        for key, value in kwargs.items():
            if key in self.kwargs_dict:
                expanded_kwargs[key] = value
            else:
                raise KeyError(f"{key} is not declared in function definition")

        return torch._C._cuda_jiterator_compile_and_launch_kernel(
            self.code_string,
            self.kernel_name,
            self.return_by_ref,
            self.num_outputs,
            tensors,
            expanded_kwargs,
        )


def _create_jit_fn(code_string: str, **kwargs) -> Callable:
    """
    Create a jiterator-generated cuda kernel for an elementwise op.

    The code string has to be a valid CUDA function that describes the computation for a single element. The code
    string has to follow the c++ template pattern, as shown in the example below. This function will be inlined
    into elementwise kernel template, and compiled on the fly. Compiled kernel will be cached in memory, as well as
    local temp dir.

    Jiterator-generated kernels accept noncontiguous tensors, and support broadcasting and type promotion.

    Args:
        code_string (str): CUDA code string to be compiled by jiterator. The entry functor must return by value.
        kwargs (Dict, optional): Keyword arguments for generated function

    Example::

        code_string = "template <typename T> T my_kernel(T x, T y, T alpha) { return -x + alpha * y; }"
        jitted_fn = create_jit_fn(code_string, alpha=1.0)
        a = torch.rand(3, device="cuda")
        b = torch.rand(3, device="cuda")
        # invoke jitted function like a regular python function
        result = jitted_fn(a, b, alpha=3.14)

    code_string also allows multiple function definitions, and the last function will be treated as the entry function.

    Example::

        code_string = (
            "template <typename T> T util_fn(T x, T y) { return ::sin(x) + ::cos(y); }"
        )
        code_string += "template <typename T> T my_kernel(T x, T y, T val) { return ::min(val, util_fn(x, y)); }"
        jitted_fn = create_jit_fn(code_string, val=0.0)
        a = torch.rand(3, device="cuda")
        b = torch.rand(3, device="cuda")
        # invoke jitted function like a regular python function
        result = jitted_fn(a, b)  # using default val=0.0

    Jiterator can be used together with python registration to override an operator's cuda kernel.
    Following example is overriding gelu's cuda kernel with relu.

    Example::

        code_string = "template <typename T> T my_gelu(T a) { return a > 0 ? a : 0; }"
        my_gelu = create_jit_fn(code_string)
        my_lib = torch.library.Library("aten", "IMPL")
        my_lib.impl("aten::gelu", my_gelu, "CUDA")
        # torch.nn.GELU and torch.nn.function.gelu are now overridden
        a = torch.rand(3, device="cuda")
        torch.allclose(torch.nn.functional.gelu(a), torch.nn.functional.relu(a))

    .. warning::
        This API is in beta and may change in future releases.

    .. warning::
        This API only supports up to 8 inputs and 1 output

    .. warning::
        All input tensors must live in CUDA device
    """
    return _JittedFunction(code_string, return_by_ref=False, num_outputs=1, **kwargs)


def _create_multi_output_jit_fn(
    code_string: str, num_outputs: int, **kwargs
) -> Callable:
    """
    Create a jiterator-generated cuda kernel for an elementwise op that supports returning one or more outputs.

    Args:
        code_string (str): CUDA code string to be compiled by jiterator. The entry functor must return value by reference.
        num_outputs(int): number of outputs return by the kernel
        kwargs (Dict, optional): Keyword arguments for generated function

    Example::

        code_string = "template <typename T> void my_kernel(T x, T y, T alpha, T& out) { out = -x + alpha * y; }"
        jitted_fn = create_jit_fn(code_string, alpha=1.0)
        a = torch.rand(3, device="cuda")
        b = torch.rand(3, device="cuda")
        # invoke jitted function like a regular python function
        result = jitted_fn(a, b, alpha=3.14)

    .. warning::
        This API is in beta and may change in future releases.

    .. warning::
        This API only supports up to 8 inputs and 8 outputs
    """
    return _JittedFunction(
        code_string, return_by_ref=True, num_outputs=num_outputs, **kwargs
    )
<|endoftext|>r"""Annotate kernels captured in CUDA graphs for profiler traces.

During CUDA graph capture, :func:`mark_kernels` tags the GPU work nodes
(kernels, memcpys, memsets) captured within its scope with user metadata.
The recorded annotations are keyed by the same ``graph node id`` values
that CUPTI-based profilers attach to kernel events during graph replay, so
the annotations can later be joined against a profiler trace to identify
which kernels belong to which region of the captured workload.

Annotation recording is enabled per capture via the ``enable_annotations``
argument of :class:`torch.cuda.graph`::

    import torch
    from torch.cuda.graph_annotations import mark_kernels, get_kernel_annotations

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, enable_annotations=True):
        with mark_kernels("phase_A"):
            y = workload_a(x)
        with mark_kernels({"name": "phase_B", "dtype": "bf16"}):
            z = workload_b(y)

    annotations = get_kernel_annotations()

The annotation mapping is typically pickled and joined against a profiler
trace offline (matching each kernel event's ``graph node id`` field).

Requires the ``cuda-bindings`` package and a CUDA driver that supports
``cudaGraphNodeGetToolsId`` (CUDA >= 13.1, or an equivalent cuda-compat
package). When unavailable, recording silently degrades to a no-op; use
:func:`is_available` to check support programmatically.

.. warning::
    This API is in prototype and may change in future releases.
"""

from torch.cuda._graph_annotations import (
    clear_kernel_annotations,
    get_kernel_annotations,
    is_available,
    mark_kernels,
)


__all__ = [
    "clear_kernel_annotations",
    "get_kernel_annotations",
    "is_available",
    "mark_kernels",
]

# The implementation lives in the private module (which predates this one and
# has external users); re-export and claim the names so they present as
# torch.cuda.graph_annotations APIs (test_correct_module_names requires it).
for _name in __all__:
    globals()[_name].__module__ = "torch.cuda.graph_annotations"
del _name
<|endoftext|># mypy: allow-untyped-defs
r"""
This package adds support for CUDA tensor types.

It implements the same function as CPU tensors, but they utilize
GPUs for computation.

It is lazily initialized, so you can always import it, and use
:func:`is_available()` to determine if your system supports CUDA.

:ref:`cuda-semantics` has more details about working with CUDA.
"""

import glob
import importlib
import importlib.util
import os
import platform
import threading
import traceback
import warnings
from collections.abc import Callable
from functools import lru_cache
from typing import Any, cast, NewType, Optional, TYPE_CHECKING

import torch
import torch._C
from torch._utils import _dummy_type, _LazySeedTracker, classproperty
from torch.types import Device

from . import _device_limits, gds
from ._utils import _get_device_index
from .graphs import (
    CUDAGraph,
    export_dot,
    export_graph_data,
    graph,
    graph_pool_handle,
    is_current_stream_capturing,
    make_graphed_callables,
)
from .green_contexts import GreenContext
from .streams import Event, ExternalStream, Stream


try:
    from torch._C import _cudart  # type: ignore[attr-defined]
except ImportError:
    _cudart = None

_initialized = False
_tls = threading.local()
_initialization_lock = threading.Lock()
_queued_calls: list[
    tuple[Callable[[], None], list[str]]
] = []  # don't invoke these until initialization occurs
_is_in_bad_fork = getattr(torch._C, "_cuda_isInBadFork", lambda: False)

_HAS_PYNVML = False
_HAS_AMDSMI = False
_PYNVML_ERR = None
_AMDSMI_ERR = None
try:
    from torch import version as _version

    try:
        if not _version.hip:
            import pynvml  # type: ignore[import]
        else:
            import ctypes
            from pathlib import Path

            # In ROCm (at least up through 6.3.2) there're 2 copies of libamd_smi.so:
            # - One at lib/libamd_smi.so
            # - One at share/amd_smi/amdsmi/libamd_smi.so
            #
            # The amdsmi python module hardcodes loading the second one in share-
            # https://github.com/ROCm/amdsmi/blob/1d305dc9708e87080f64f668402887794cd46584/py-interface/amdsmi_wrapper.py#L174
            #
            # See also https://github.com/ROCm/amdsmi/issues/72.
            #
            # This creates an ODR violation if the copy of libamd_smi.so from lib
            # is also loaded (via `ld` linking, `LD_LIBRARY_PATH` or `rpath`).
            #
            # In order to avoid the violation we hook CDLL and try using the
            # already loaded version of amdsmi, or any version in the processes
            # rpath/LD_LIBRARY_PATH first, so that we only load a single copy
            # of the .so.
            class _amdsmi_cdll_hook:
                def __init__(self) -> None:
                    self.original_CDLL = ctypes.CDLL  # type: ignore[misc,assignment]
                    paths = ["libamd_smi.so"]
                    if rocm_home := os.getenv("ROCM_HOME", os.getenv("ROCM_PATH")):
                        paths = [os.path.join(rocm_home, "lib/libamd_smi.so")] + paths
                    # TheRock ROCm wheels install the amdsmi python package in
                    # site-packages/amdsmi and ship the native library under the
                    # _rocm_sdk_core wheel (site-packages/_rocm_sdk_core/lib)
                    # rather than on the loader path or under $ROCM_HOME. The
                    # file there is a versioned soname (e.g. libamd_smi.so.26),
                    # so amdsmi's bare CDLL("libamd_smi.so") can't find it. Locate
                    # the package and append the concrete versioned files as a
                    # last-resort fallback so the hook can redirect to them.
                    if platform.system() == "Linux":
                        paths = paths + self._rocm_sdk_core_amdsmi_paths()
                    self.paths: list[str] = paths

                @staticmethod
                def _rocm_sdk_core_amdsmi_paths() -> list[str]:
                    try:
                        spec = importlib.util.find_spec("_rocm_sdk_core")
                    except (ImportError, ValueError):
                        return []
                    if spec is None or not spec.submodule_search_locations:
                        return []
                    found: list[str] = []
                    for location in spec.submodule_search_locations:
                        found += glob.glob(
                            os.path.join(location, "lib", "libamd_smi.so*")
                        )
                    return sorted(found)

                def hooked_CDLL(
                    self, name: str | Path | None, *args: Any, **kwargs: Any
                ) -> ctypes.CDLL:
                    if name and Path(name).name.startswith("libamd_smi.so"):
                        for path in self.paths:
                            try:
                                return self.original_CDLL(path, *args, **kwargs)
                            except OSError:
                                pass
                    return self.original_CDLL(name, *args, **kwargs)  # type: ignore[arg-type]

                def __enter__(self) -> None:
                    ctypes.CDLL = self.hooked_CDLL  # type: ignore[misc,assignment]

                def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
                    ctypes.CDLL = self.original_CDLL  # type: ignore[misc]

            try:
                with _amdsmi_cdll_hook():
                    import amdsmi  # type: ignore[import]
                _HAS_AMDSMI = True
            except ModuleNotFoundError as err:
                _AMDSMI_ERR = err
                raise
            except (KeyError, OSError) as err:
                # The amdsmi python package is installed but its native library
                # (libamd_smi.so) could not be discovered/loaded -- e.g. TheRock
                # ROCm wheels lay amdsmi out so that its own find_smi_library()
                # misses the versioned libamd_smi.so.* and raises KeyError. Treat
                # this like a missing optional dependency (degrade to
                # _HAS_AMDSMI=False) instead of aborting `import torch`.
                _AMDSMI_ERR = err
                raise ModuleNotFoundError(
                    "amdsmi is installed but libamd_smi.so could not be loaded"
                ) from err

        _HAS_PYNVML = True
    except ModuleNotFoundError:
        pass
    finally:
        del _version
except ImportError as err:
    _PYNVML_ERR = err  # sometimes a lib is installed but the import fails for some other reason, so we log the error for later

_lazy_seed_tracker = _LazySeedTracker()

# Define dummy _CudaDeviceProperties type if PyTorch was compiled without CUDA
if hasattr(torch._C, "_CudaDeviceProperties"):
    _CudaDeviceProperties = torch._C._CudaDeviceProperties
else:
    _CudaDeviceProperties = _dummy_type("_CudaDeviceProperties")  # type: ignore[assignment, misc]

if hasattr(torch._C, "_cuda_exchangeDevice"):
    _exchange_device = torch._C._cuda_exchangeDevice
else:

    def _exchange_device(device: int) -> int:
        if device < 0:
            return -1
        raise RuntimeError("PyTorch was compiled without CUDA support")


if hasattr(torch._C, "_cuda_maybeExchangeDevice"):
    _maybe_exchange_device = torch._C._cuda_maybeExchangeDevice
else:

    def _maybe_exchange_device(device: int) -> int:
        if device < 0:
            return -1
        raise RuntimeError("PyTorch was compiled without CUDA support")


has_half: bool = True
has_magma: bool = torch._C._has_magma

default_generators: tuple[torch._C.Generator] = ()  # type: ignore[assignment]


def _is_compiled() -> bool:
    r"""Return true if compile with CUDA support."""
    return hasattr(torch._C, "_cuda_getDeviceCount")


def _nvml_based_avail() -> bool:
    return os.getenv("PYTORCH_NVML_BASED_CUDA_CHECK") == "1"


def is_available() -> bool:
    r"""
    Return a bool indicating if CUDA is currently available.

    .. note:: This function will NOT poison fork if the environment variable
        ``PYTORCH_NVML_BASED_CUDA_CHECK=1`` is set. For more details, see
        :ref:`multiprocessing-poison-fork-note`.
    """
    if not _is_compiled():
        return False
    if _nvml_based_avail():
        # The user has set an env variable to request this availability check that attempts to avoid fork poisoning by
        # using NVML at the cost of a weaker CUDA availability assessment. Note that if NVML discovery/initialization
        # fails, this assessment falls back to the default CUDA Runtime API assessment (`cudaGetDeviceCount`)
        return device_count() > 0
    else:
        # The default availability inspection never throws and returns 0 if the driver is missing or can't
        # be initialized. This uses the CUDA Runtime API `cudaGetDeviceCount` which in turn initializes the CUDA Driver
        # API via `cuInit`
        return torch._C._cuda_getDeviceCount() > 0


def is_bf16_supported(including_emulation: bool = True):
    r"""Return a bool indicating if the current CUDA/ROCm device supports dtype bfloat16."""
    # Check for ROCm, if true return true, no ROCM_VERSION check required,
    # since it is supported on AMD GPU archs.
    if torch.version.hip:
        return True

    # If CUDA is not available, then it does not support bf16 either
    if not is_available():
        return False

    device = torch.cuda.current_device()

    if torch.cuda.get_device_properties(device).major >= 8:
        return True

    if not including_emulation:
        return False

    # Finally try to create a bfloat16 device.
    return _check_bf16_tensor_supported(device)


@lru_cache(maxsize=16)
def _check_bf16_tensor_supported(device: Device):
    try:
        torch.tensor([1.0], dtype=torch.bfloat16, device=device)
        return True
    except Exception:
        return False


def is_tf32_supported() -> bool:
    r"""Return a bool indicating if the current CUDA/ROCm device supports dtype tf32."""
    if torch.version.hip:
        prop_name = torch.cuda.get_device_properties().gcnArchName
        archs = ("gfx94", "gfx95")
        for arch in archs:
            if arch in prop_name:
                return True
        return False

    # Otherwise, tf32 is supported on CUDA platforms that natively (i.e. no emulation)
    # support bfloat16.
    return is_bf16_supported(including_emulation=False)


def _sleep(cycles):
    torch._C._cuda_sleep(cycles)


def _extract_arch_version(arch_string: str) -> int:
    """Extracts the architecture string from a CUDA version"""
    base = arch_string.split("_", maxsplit=2)[1]
    base = base.removesuffix("a").removesuffix("f")
    return int(base)


class _CompatInterval:
    """
    Defines a range of compute capabilities starting at a given
    version and going up to the end of that major version. This
    also allows excluding specific versions from the range.
    """

    def __init__(self, start, exclude: set[int] | None = None):
        self.major, self.minor = start // 10, start % 10
        self.exclude = set() if exclude is None else exclude

    def __contains__(self, x):
        if x in self.exclude:
            return False
        x_major, x_minor = x // 10, x % 10
        return x_major == self.major and x_minor >= self.minor

    def __str__(self):
        result = f">={self.major}.{self.minor},<{self.major + 1}.0"
        if len(self.exclude) > 0:
            exceptions = ", ".join(f"{x // 10}.{x % 10}" for x in self.exclude)
            result += f" except {{{exceptions}}}"
        return result


class _CompatSet:
    """
    A set of compute capabilities. It exists primarily to support custom
    printing logic and is otherwise equivalent to a plain python set().
    """

    def __init__(self, values: set[int]):
        self.values = values

    def __contains__(self, x):
        return x in self.values

    def __str__(self):
        return "{" + ", ".join(f"{v // 10}.{v % 10}" for v in self.values) + "}"


# (code SM)->(device SM required to execute the code)
#
# Developer Notes:
# - This dict should be kept up to date with keys corresponding
#   to SM versions that PyTorch can be built for. An out of date
#   mapping will lead to false warnings.
# - The keys in dict correspond to known sm versions but the values
#   are merely rules based on sm compatibility guarantees for NVIDIA
#   devices while accounting for incompatibility of iGPU and dGPU.
DEVICE_REQUIREMENT: dict[int, _CompatSet | _CompatInterval] = {
    50: _CompatInterval(start=50, exclude={53}),
    52: _CompatInterval(start=52, exclude={53}),
    53: _CompatSet({53}),
    60: _CompatInterval(start=60, exclude={62}),
    61: _CompatInterval(start=61, exclude={62}),
    62: _CompatSet({62}),
    70: _CompatInterval(start=70, exclude={72}),
    72: _CompatSet({72}),
    75: _CompatInterval(start=75),
    80: _CompatInterval(start=80, exclude={87}),
    86: _CompatInterval(start=86, exclude={87}),
    87: _CompatSet({87}),
    89: _CompatInterval(start=89),
    90: _CompatInterval(start=90),
    100: _CompatInterval(start=100, exclude={101}),
    101: _CompatSet({101, 110}),  # 101 was renamed to 110
    103: _CompatInterval(start=103),
    110: _CompatSet({101, 110}),  # 101 was renamed to 110
    120: _CompatInterval(start=120),
    121: _CompatInterval(start=121),
}

# CUDA 13.2 allows SBSA binaries to run on Jetson devices.
# This dict can be combined with DEVICE_REQUIREMENT once
# the minimum supported CUDA version is 13.2.
DEVICE_REQUIREMENT_POST_JETSON_SBSA_UNIFICATION: dict[
    int, _CompatSet | _CompatInterval
] = DEVICE_REQUIREMENT | {
    70: _CompatInterval(start=70),
    80: _CompatInterval(start=80),
    86: _CompatInterval(start=86),
}

# TORCH_CUDA_ARCH_LIST for PyTorch releases, keyed by host arch.
# Kept in sync with .ci/manywheel/build_cuda.sh by the validator in
# .github/scripts/generate_binary_build_matrix.py.
PYTORCH_RELEASES_CODE_CC: dict[str, dict[str, set[int]]] = {
    "12.6": {
        "x86_64": {50, 60, 70, 75, 80, 86, 90},
        "aarch64": {80, 90},
    },
    "13.0": {
        "x86_64": {75, 80, 86, 90, 100, 120},
        "aarch64": {80, 90, 100, 110, 120},
    },
    "13.2": {
        "x86_64": {75, 80, 86, 90, 100, 120},
        "aarch64": {80, 90, 100, 110, 120},
    },
    "13.4": {
        "x86_64": {75, 80, 86, 90, 100, 120},
        "aarch64": {80, 90, 100, 110, 120},
    },
}


def _device_requirement(code_cc):
    if torch.version.cuda is None:
        return None
    requirement = (
        DEVICE_REQUIREMENT_POST_JETSON_SBSA_UNIFICATION
        if tuple(int(x) for x in torch.version.cuda.split(".")) >= (13, 2)
        else DEVICE_REQUIREMENT
    )
    return requirement.get(code_cc, None)


def _host_arch_key() -> str:
    machine = platform.machine().lower()
    return "aarch64" if machine == "aarch64" else "x86_64"


def _code_compatible_with_device(device_cc: int, code_cc: int):
    compatible_devices = _device_requirement(code_cc)
    if compatible_devices is None:
        warnings.warn(
            f"PyTorch was compiled with an unknown compute capability {code_cc // 10}.{code_cc % 10}. "
            + " Please create an issue on Github if this is a valid compute capability.",
            stacklevel=2,
        )
        return device_cc in _CompatInterval(start=code_cc)
    return device_cc in compatible_devices


def _warn_unsupported_code(device_index: int, device_cc: int, code_ccs: list[int]):
    name = get_device_name(device_index)

    arch = _host_arch_key()
    compatible_releases: list[str] = []
    for cuda, by_arch in PYTORCH_RELEASES_CODE_CC.items():
        build_ccs = by_arch.get(arch, set())
        if any(_code_compatible_with_device(device_cc, cc) for cc in build_ccs):
            compatible_releases.append(cuda)

    lines = [
        f"Found GPU{device_index} {name} which is of compute capability (CC) {device_cc // 10}.{device_cc % 10}.",
        "The following list shows the CCs this version of PyTorch was built for and the hardware CCs it supports:",
    ] + [
        f"- {cc // 10}.{cc % 10} which supports hardware CC {_device_requirement(cc)}"
        for cc in code_ccs
    ]

    if len(compatible_releases) > 0:
        version = torch.__version__
        base_version = version.split("+")[0]
        is_nightly = "dev" in base_version
        index_root = (
            "https://download.pytorch.org/whl/nightly"
            if is_nightly
            else "https://download.pytorch.org/whl"
        )
        lines.append(
            f"Your installed torch=={version} does not include kernels for this GPU. "
            "Reinstall the same version against a CUDA build that does, e.g.:"
        )
        for cuda in compatible_releases:
            cu_tag = "cu" + cuda.replace(".", "")
            lines.append(
                f"  For CUDA {cuda} use pip install torch=={base_version} --index-url {index_root}/{cu_tag}"
            )
    else:
        lines.append(
            f"No published PyTorch CUDA builds for release {torch.__version__} support this GPU. "
            "Visit https://pytorch.org/get-started/locally/ to find a compatible release."
        )

    warnings.warn("\n".join(lines), stacklevel=2)


def _check_capability():
    if torch.version.cuda is None:  # on ROCm we don't want this check
        return

    arch_list = get_arch_list()
    if len(arch_list) == 0:
        return

    code_ccs = [_extract_arch_version(cc) for cc in arch_list]
    for d in range(device_count()):
        major, minor = get_device_capability(d)
        device_cc = 10 * major + minor
        if not any(
            _code_compatible_with_device(device_cc, code_cc) for code_cc in code_ccs
        ):
            _warn_unsupported_code(d, device_cc, code_ccs)


def _check_cubins():
    incompatible_device_warn = """
{} with CUDA capability sm_{} is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities {}.
If you want to use the {} GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/
"""
    if torch.version.cuda is None:  # on ROCm we don't want this check
        return
    arch_list = get_arch_list()
    if len(arch_list) == 0:
        return
    supported_sm = [_extract_arch_version(arch) for arch in arch_list if "sm_" in arch]
    for idx in range(device_count()):
        cap_major, cap_minor = get_device_capability(idx)
        # NVIDIA GPU compute architectures are backward compatible within major version
        supported = any(sm // 10 == cap_major for sm in supported_sm)
        if not supported:
            device_name = get_device_name(idx)
            capability = cap_major * 10 + cap_minor
            warnings.warn(
                incompatible_device_warn.format(
                    device_name, capability, " ".join(arch_list), device_name
                ),
                stacklevel=2,
            )


def is_initialized():
    r"""Return whether PyTorch's CUDA state has been initialized."""
    return _initialized and not _is_in_bad_fork()


def _lazy_call(callable, **kwargs):
    with _initialization_lock:
        if is_initialized():
            callable()
        else:
            # TODO(torch_deploy): this accesses linecache, which attempts to read the
            # file system to get traceback info. Patch linecache or do something
            # else here if this ends up being important.
            global _lazy_seed_tracker
            if kwargs.get("seed_all", False):
                _lazy_seed_tracker.queue_seed_all(callable, traceback.format_stack())
            elif kwargs.get("seed", False):
                _lazy_seed_tracker.queue_seed(callable, traceback.format_stack())
            else:
                # Don't store the actual traceback to avoid memory cycle
                _queued_calls.append((callable, traceback.format_stack()))


_lazy_call(_check_capability)
_lazy_call(_check_cubins)


class DeferredCudaCallError(Exception):
    pass


AcceleratorError = torch._C.AcceleratorError
OutOfMemoryError = torch._C.OutOfMemoryError


def init():
    r"""Initialize PyTorch's CUDA state.

    You may need to call this explicitly if you are interacting with
    PyTorch via its C API, as Python bindings for CUDA functionality
    will not be available until this initialization takes place.
    Ordinary users should not need this, as all of PyTorch's CUDA methods
    automatically initialize CUDA state on-demand.

    Does nothing if the CUDA state is already initialized.
    """
    _lazy_init()


def _lazy_init():
    global _initialized, _queued_calls
    if is_initialized() or hasattr(_tls, "is_initializing"):
        return
    with _initialization_lock:
        # We be double-checked locking, boys!  This is OK because
        # the above test was GIL protected anyway.  The inner test
        # is for when a thread blocked on some other thread which was
        # doing the initialization; when they get the lock, they will
        # find there is nothing left to do.
        if is_initialized():
            return
        # It is important to prevent other threads from entering _lazy_init
        # immediately, while we are still guaranteed to have the GIL, because some
        # of the C calls we make below will release the GIL
        if _is_in_bad_fork():
            raise RuntimeError(
                "Cannot re-initialize CUDA in forked subprocess. To use CUDA with "
                "multiprocessing, you must use the 'spawn' start method"
            )
        if not hasattr(torch._C, "_cuda_getDeviceCount"):
            raise AssertionError("Torch not compiled with CUDA enabled")
        if _cudart is None:
            raise AssertionError(
                "libcudart functions unavailable. It looks like you have a broken build?"
            )
        # This function throws if there's a driver initialization error, no GPUs
        # are found or any other error occurs
        torch._C._cuda_init()
        # Some of the queued calls may reentrantly call _lazy_init();
        # we need to just return without initializing in that case.
        # However, we must not let any *other* threads in!
        _tls.is_initializing = True

        _queued_calls.extend(calls for calls in _lazy_seed_tracker.get_calls() if calls)

        try:
            for queued_call, orig_traceback in _queued_calls:
                try:
                    queued_call()
                except Exception as e:
                    msg = (
                        f"CUDA call failed lazily at initialization with error: {str(e)}\n\n"
                        f"CUDA call was originally invoked at:\n\n{''.join(orig_traceback)}"
                    )
                    raise DeferredCudaCallError(msg) from e
        finally:
            delattr(_tls, "is_initializing")
        _initialized = True


def cudart():
    r"""Retrieves the CUDA runtime API module.


    This function initializes the CUDA runtime environment if it is not already
    initialized and returns the CUDA runtime API module (_cudart). The CUDA
    runtime API module provides access to various CUDA runtime functions.

    Args:
        ``None``

    Returns:
        module: The CUDA runtime API module (_cudart).

    Raises:
        RuntimeError: If CUDA cannot be re-initialized in a forked subprocess.
        AssertionError: If PyTorch is not compiled with CUDA support or if libcudart functions are unavailable.

    Example of CUDA operations with profiling:
        >>> import torch
        >>> from torch.cuda import cudart, check_error
        >>> import os
        >>>
        >>> os.environ["CUDA_PROFILE"] = "1"
        >>>
        >>> def perform_cuda_operations_with_streams():
        >>>     stream = torch.cuda.Stream()
        >>>     with torch.cuda.stream(stream):
        >>>         x = torch.randn(100, 100, device='cuda')
        >>>         y = torch.randn(100, 100, device='cuda')
        >>>         z = torch.mul(x, y)
        >>>     return z
        >>>
        >>> torch.cuda.synchronize()
        >>> print("====== Start nsys profiling ======")
        >>> check_error(cudart().cudaProfilerStart())
        >>> with torch.autograd.profiler.emit_nvtx():
        >>>     result = perform_cuda_operations_with_streams()
        >>>     print("CUDA operations completed.")
        >>> check_error(torch.cuda.cudart().cudaProfilerStop())
        >>> print("====== End nsys profiling ======")

    To run this example and save the profiling information, execute:
        >>> $ nvprof --profile-from-start off --csv --print-summary -o trace_name.prof -f -- python cudart_test.py

    This command profiles the CUDA operations in the provided script and saves
    the profiling information to a file named `trace_name.prof`.
    The `--profile-from-start off` option ensures that profiling starts only
    after the `cudaProfilerStart` call in the script.
    The `--csv` and `--print-summary` options format the profiling output as a
    CSV file and print a summary, respectively.
    The `-o` option specifies the output file name, and the `-f` option forces the
    overwrite of the output file if it already exists.
    """
    _lazy_init()
    return _cudart


class cudaStatus:
    SUCCESS: int = 0
    ERROR_NOT_READY: int = 34


class CudaError(RuntimeError):
    def __init__(self, code: int) -> None:
        # pyrefly: ignore [missing-attribute]
        msg = _cudart.cudaGetErrorString(_cudart.cudaError(code))
        super().__init__(f"{msg} ({code})")


def check_error(res: int) -> None:
    r"""Raise an error if the result of a CUDA runtime API call is not success."""
    # pyrefly: ignore [missing-attribute]
    if res != _cudart.cudaError.success:
        raise CudaError(res)


class _DeviceGuard:
    def __init__(self, index: int):
        self.idx = index
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = torch.cuda._exchange_device(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        self.idx = torch.cuda._maybe_exchange_device(self.prev_idx)
        return False


class device:
    r"""Context-manager that changes the selected device.

    Args:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device: Any):
        self.idx = _get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = torch.cuda._exchange_device(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        self.idx = torch.cuda._maybe_exchange_device(self.prev_idx)
        return False


class device_of(device):
    r"""Context-manager that changes the current device to that of given object.

    You can use both tensors and storages as arguments. If a given object is
    not allocated on a GPU, this is a no-op.

    Args:
        obj (Tensor or Storage): object allocated on the selected device.
    """

    def __init__(self, obj):
        idx = obj.get_device() if obj.is_cuda else -1
        super().__init__(idx)


def set_device(device: Device) -> None:
    r"""Set the current device.

    Usage of this function is discouraged in favor of :any:`device`. In most
    cases it's better to use ``CUDA_VISIBLE_DEVICES`` environmental variable.

    Args:
        device (torch.device or int): selected device. This function is a no-op
            if this argument is negative.
    """
    device = _get_device_index(device)
    if device >= 0:
        torch._C._cuda_setDevice(device)


def get_device_name(device: Device = None) -> str:
    r"""Get the name of a device.

    Args:
        device (torch.device or int or str, optional): device for which to return the
            name. This function is a no-op if this argument is a negative
            integer. It uses the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Returns:
        str: the name of the device
    """
    return get_device_properties(device).name


def get_device_capability(device: Device = None) -> tuple[int, int]:
    r"""Get the cuda capability of a device.

    Args:
        device (torch.device or int or str, optional): device for which to return the
            device capability. This function is a no-op if this argument is
            a negative integer. It uses the current device, given by
            :func:`~torch.cuda.current_device`, if :attr:`device` is ``None``
            (default).

    Returns:
        tuple(int, int): the major and minor cuda capability of the device
    """
    prop = get_device_properties(device)
    return prop.major, prop.minor


# pyrefly: ignore [not-a-type]
def get_device_properties(device: Device = None) -> _CudaDeviceProperties:
    r"""Get the properties of a device.

    Args:
        device (torch.device or int or str, optional): device for which to return the
            properties of the device.  It uses the current device, given by
            :func:`~torch.cuda.current_device`, if :attr:`device` is ``None``
            (default).

    Returns:
        _CudaDeviceProperties: the properties of the device
    """
    _lazy_init()  # will define _get_device_properties
    device = _get_device_index(device, optional=True)
    if device < 0 or device >= device_count():
        raise AssertionError("Invalid device id")
    return _get_device_properties(device)  # type: ignore[name-defined]


def can_device_access_peer(device: Device, peer_device: Device) -> bool:
    r"""Check if peer access between two devices is possible."""
    _lazy_init()
    device = _get_device_index(device, optional=True)
    peer_device = _get_device_index(peer_device)
    if device < 0 or device >= device_count():
        raise AssertionError("Invalid device id")
    if peer_device < 0 or peer_device >= device_count():
        raise AssertionError("Invalid peer device id")
    return torch._C._cuda_canDeviceAccessPeer(device, peer_device)


class StreamContext:
    r"""Context-manager that selects a given stream.

    All CUDA kernels queued within its context will be enqueued on a selected
    stream.

    Args:
        Stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note:: Streams are per-device.
    """

    cur_stream: Optional["torch.cuda.Stream"]

    def __init__(self, stream: Optional["torch.cuda.Stream"]):
        self.stream = stream
        self.idx = _get_device_index(None, True)
        if not torch.jit.is_scripting():
            if self.idx is None:
                # pyrefly: ignore [bad-assignment]
                self.idx = -1

        self.src_prev_stream = (
            None if not torch.jit.is_scripting() else torch.cuda.default_stream(None)
        )
        self.dst_prev_stream = (
            None if not torch.jit.is_scripting() else torch.cuda.default_stream(None)
        )

    def __enter__(self):
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # Return if stream is None or CUDA device not available
        if cur_stream is None or self.idx == -1:
            return
        self.src_prev_stream = torch.cuda.current_stream(None)

        # If the stream is not on the current device, then
        # set the current stream on the device
        if self.src_prev_stream.device != cur_stream.device:
            with device(cur_stream.device):
                self.dst_prev_stream = torch.cuda.current_stream(cur_stream.device)
        torch.cuda.set_stream(cur_stream)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # If stream is None or no CUDA device available, return
        if cur_stream is None or self.idx == -1:
            return

        # Reset the stream on the original device
        # and destination device
        if self.src_prev_stream.device != cur_stream.device:  # type: ignore[union-attr]
            torch.cuda.set_stream(self.dst_prev_stream)  # type: ignore[arg-type]
        torch.cuda.set_stream(self.src_prev_stream)  # type: ignore[arg-type]


def stream(stream: Optional["torch.cuda.Stream"]) -> StreamContext:
    r"""Wrap around the Context-manager StreamContext that selects a given stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note::
        In eager mode stream is of type Stream class while in JIT it is
        an object of the custom class ``torch.classes.cuda.Stream``.
    """
    return StreamContext(stream)


def _set_stream_by_id(stream_id, device_index, device_type):
    r"""set stream specified by the stream id, device index and
        device type

    Args: stream_id (int): stream id in stream pool
          device_index (int): device index in topo
          device_type (int): enum device type
    """
    torch._C._cuda_setStream(
        stream_id=stream_id,
        device_index=device_index,
        device_type=device_type,
    )


def set_stream(stream: Stream):
    r"""Set the current stream. This is a wrapper API to set the stream.
        Usage of this function is discouraged in favor of the ``stream``
        context manager.

    Args:
        stream (Stream): selected stream. This function is a no-op
            if this argument is ``None``.
    """
    if stream is None:
        return
    _set_stream_by_id(
        stream_id=stream.stream_id,
        device_index=stream.device_index,
        device_type=stream.device_type,
    )


def _parse_visible_devices() -> list[int] | list[str]:
    r"""Parse CUDA_VISIBLE_DEVICES environment variable."""
    var = os.getenv("CUDA_VISIBLE_DEVICES")

    if torch.version.hip:
        hip_devices = os.getenv("HIP_VISIBLE_DEVICES")
        rocr_devices = os.getenv("ROCR_VISIBLE_DEVICES")

        # You must take care if both HIP and ROCR env vars are set as they have
        # different meanings. Both env vars accept either a list of ints or a
        # list of UUIDs. The ROCR env var is processed first which then reduces
        # the number of GPUs that HIP can select from.
        if rocr_devices is not None:
            rocr_count = len(rocr_devices.split(","))
            if hip_devices is not None:
                # sanity check if both env vars are set
                if len(hip_devices.split(",")) > rocr_count:
                    raise RuntimeError(
                        "HIP_VISIBLE_DEVICES contains more devices than ROCR_VISIBLE_DEVICES"
                    )
                # HIP_VISIBLE_DEVICES is preferred over ROCR_VISIBLE_DEVICES
                var = hip_devices
            else:
                return list(range(rocr_count))
        elif hip_devices is not None:
            var = hip_devices

    if var is None:
        return list(range(64))

    def _strtoul(s: str) -> int:
        """Return -1 or positive integer sequence string starts with."""
        if not s:
            return -1
        for idx, c in enumerate(s):
            if not (c.isdigit() or (idx == 0 and c in "+-")):
                break
            if idx + 1 == len(s):
                idx += 1
        return int(s[:idx]) if idx > 0 else -1

    def parse_list_with_prefix(lst: str, prefix: str) -> list[str]:
        rcs: list[str] = []
        for elem in lst.split(","):
            # Repeated id results in empty set
            if elem in rcs:
                return cast(list[str], [])
            # Anything other but prefix is ignored
            if not elem.startswith(prefix):
                break
            rcs.append(elem)
        return rcs

    if var.startswith("GPU-"):
        return parse_list_with_prefix(var, "GPU-")
    if var.startswith("MIG-"):
        return parse_list_with_prefix(var, "MIG-")
    # CUDA_VISIBLE_DEVICES uses something like strtoul
    # which makes `1gpu2,2ampere` is equivalent to `1,2`
    rc: list[int] = []
    for elem in var.split(","):
        x = _strtoul(elem.strip())
        # Repeated ordinal results in empty set
        if x in rc:
            return cast(list[int], [])
        # Negative value aborts the sequence
        if x < 0:
            break
        rc.append(x)
    return rc


def _raw_device_count_amdsmi() -> int:
    if not _HAS_AMDSMI:
        return -1
    try:
        amdsmi.amdsmi_init()
    except amdsmi.AmdSmiException as e:
        warnings.warn(
            f"Can't initialize amdsmi - Error code: {e.err_code}", stacklevel=2
        )
        return -1
    socket_handles = amdsmi.amdsmi_get_processor_handles()
    return len(socket_handles)


def _raw_device_count_nvml() -> int:
    r"""Return number of devices as reported by NVML or negative value if NVML discovery/initialization failed."""
    from ctypes import byref, c_int, CDLL

    nvml_h = CDLL("libnvidia-ml.so.1")
    rc = nvml_h.nvmlInit()
    if rc != 0:
        warnings.warn("Can't initialize NVML", stacklevel=2)
        return -1
    dev_count = c_int(-1)
    rc = nvml_h.nvmlDeviceGetCount_v2(byref(dev_count))
    if rc != 0:
        warnings.warn("Can't get nvml device count", stacklevel=2)
        return -1
    del nvml_h
    return dev_count.value


def _raw_device_uuid_amdsmi() -> list[str] | None:
    from ctypes import byref, c_int, c_void_p, CDLL, create_string_buffer

    if not _HAS_AMDSMI:
        return None
    try:
        amdsmi.amdsmi_init()
    except amdsmi.AmdSmiException:
        warnings.warn("Can't initialize amdsmi", stacklevel=2)
        return None
    try:
        socket_handles = amdsmi.amdsmi_get_processor_handles()
        dev_count = len(socket_handles)
    except amdsmi.AmdSmiException:
        warnings.warn("Can't get amdsmi device count", stacklevel=2)
        return None
    uuids: list[str] = []
    for idx in range(dev_count):
        try:
            handler = amdsmi.amdsmi_get_processor_handles()[idx]
        except amdsmi.AmdSmiException:
            warnings.warn("Cannot get amd device handler", stacklevel=2)
            return None
        try:
            uuid = amdsmi.amdsmi_get_gpu_asic_info(handler)["asic_serial"][
                2:
            ]  # Removes 0x prefix from serial
        except amdsmi.AmdSmiException:
            warnings.warn("Cannot get uuid for amd device", stacklevel=2)
            return None
        uuids.append(
            str(uuid).lower()
        )  # Lower-case to match expected HIP_VISIBLE_DEVICES uuid input
    return uuids


def _raw_device_uuid_nvml() -> list[str] | None:
    r"""Return list of device UUID as reported by NVML or None if NVM discovery/initialization failed."""
    from ctypes import byref, c_int, c_void_p, CDLL, create_string_buffer

    nvml_h = CDLL("libnvidia-ml.so.1")
    rc = nvml_h.nvmlInit()
    if rc != 0:
        warnings.warn("Can't initialize NVML", stacklevel=2)
        return None
    dev_count = c_int(-1)
    rc = nvml_h.nvmlDeviceGetCount_v2(byref(dev_count))
    if rc != 0:
        warnings.warn("Can't get nvml device count", stacklevel=2)
        return None
    uuids: list[str] = []
    for idx in range(dev_count.value):
        dev_id = c_void_p()
        rc = nvml_h.nvmlDeviceGetHandleByIndex_v2(idx, byref(dev_id))
        if rc != 0:
