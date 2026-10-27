    # comment on `_native_decomp_overrides` for why).
    if overload is not None:
        _native_decomp_overrides[overload] = compile_router


def _register_all_overrides() -> None:
    """
    Perform all registration calls from previously-built override graphs.
    """
    for key, graph in _graphs.items():
        op_symbol, dispatch_key = key

        _register_overrides_from_graph(
            op_symbol,
            dispatch_key,
            graph,
        )


def reorder_graphs_from_user_function(
    fn: UserOrderingFn,
    *,
    reregister_overrides: bool = False,
) -> None:
    """
    Reorder override graphs using a user-provided ordering function.

    Args:
        fn: User-provided function that takes (op_symbol, dispatch_key, graph)
            and returns a reordered graph
        reregister_overrides: Whether to reregister graphs that have changed

    Note:
        This function uses the common graph transformation pattern and can serve
        as an example for other graph manipulation operations.
    """
    _apply_graph_transformation(
        transformation_fn=fn,
        reregister_overrides=reregister_overrides,
    )


def _apply_graph_filter(
    filter_fn: Callable[[str, str, _OverrideNode], bool],
    *,
    reregister_overrides: bool = False,
) -> None:
    """
    Apply a filter function to remove nodes from graphs.

    This is a convenience function that uses the graph transformation pattern
    to filter out unwanted nodes.

    Args:
        filter_fn: Function that takes (op_symbol, dispatch_key, node) and
            returns True to keep the node, False to remove it
        reregister_overrides: Whether to reregister modified graphs

    Example:
        # Remove all nodes with "deprecated" in the DSL name
        _apply_graph_filter(
            lambda op, dk, node: "deprecated" not in node.dsl_name,
            reregister_overrides=True
        )

    Note:
        If filter_fn raises an exception for a specific graph, the original
        graph will be preserved and processing will continue.
    """

    def filtering_transformation(
        op_symbol: str, dispatch_key: str, graph: list[_OverrideNode]
    ) -> list[_OverrideNode]:
        """Apply filter_fn to graph with error handling."""
        try:
            return [node for node in graph if filter_fn(op_symbol, dispatch_key, node)]
        except (TypeError, ValueError, AttributeError, RuntimeError):
            log.warning(
                "Graph transformation failed for %s/%s. Preserving original graph.",
                op_symbol,
                dispatch_key,
                exc_info=True,
            )
            return graph
        except Exception:
            log.exception(
                "Unexpected error in graph transformation for %s/%s. Preserving original graph.",
                op_symbol,
                dispatch_key,
            )
            return graph

    _apply_graph_transformation(
        transformation_fn=filtering_transformation,
        reregister_overrides=reregister_overrides,
    )


def _apply_selective_reordering(
    condition_fn: Callable[[str, str], bool],
    ordering_fn: UserOrderingFn,
    *,
    reregister_overrides: bool = False,
) -> None:
    """
    Apply reordering only to graphs that match a condition.

    This allows for more targeted reordering operations.

    Args:
        condition_fn: Function that takes (op_symbol, dispatch_key) and
            returns True if the graph should be reordered
        ordering_fn: Ordering function to apply to matching graphs
        reregister_overrides: Whether to reregister modified graphs

    Example:
        # Only reorder CUDA operations
        _apply_selective_reordering(
            condition_fn=lambda op, dk: dk == "CUDA",
            ordering_fn=lambda op, dk, g: sorted(g, key=lambda n: n.dsl_name),
            reregister_overrides=True
        )

    Note:
        If condition_fn or ordering_fn raises an exception for a specific graph,
        the original graph will be preserved and processing will continue.
    """

    def conditional_transformation(
        op_symbol: str, dispatch_key: str, graph: list[_OverrideNode]
    ) -> list[_OverrideNode]:
        """Apply ordering_fn conditionally based on condition_fn result."""
        try:
            should_reorder = condition_fn(op_symbol, dispatch_key)
        except (TypeError, ValueError, AttributeError, RuntimeError):
            log.warning(
                "Graph transformation failed for %s/%s. Preserving original graph.",
                op_symbol,
                dispatch_key,
                exc_info=True,
            )
            return graph
        except Exception:
            log.exception(
                "Unexpected error in graph transformation for %s/%s. Preserving original graph.",
                op_symbol,
                dispatch_key,
            )
            return graph

        if should_reorder:
            try:
                return ordering_fn(op_symbol, dispatch_key, graph)
            except (TypeError, ValueError, AttributeError, RuntimeError):
                log.warning(
                    "Graph transformation failed for %s/%s. Preserving original graph.",
                    op_symbol,
                    dispatch_key,
                    exc_info=True,
                )
                return graph
            except Exception:
                log.exception(
                    "Unexpected error in graph transformation for %s/%s. Preserving original graph.",
                    op_symbol,
                    dispatch_key,
                )
                return graph

        return graph  # Return unchanged if condition doesn't match

    _apply_graph_transformation(
        transformation_fn=conditional_transformation,
        reregister_overrides=reregister_overrides,
    )
<|endoftext|>"""Shared helpers for native ops backed by nvmath-python / cuSOLVER.

Centralizes nvmath availability checks, cuSOLVER version gating, and cuSOLVER
handle management so multiple native ops (e.g. linalg.polar, and upcoming
matmul-family ops) don't each reimplement them.

All functions here are import-light at module load (nvmath is only imported
inside the function bodies, lazily, on first use) so importing this module does
not pull in the nvmath/cuda runtime or initialize CUDA.
"""

import torch

from .common_utils import _unavailable_reason


# (package_name, importable_module) pairs probed without importing.
_NVMATH_DEPS = [
    ("nvmath-python", "nvmath.bindings"),
]


def nvmath_unavailable_reason() -> str | None:
    """Return a human-readable reason nvmath is unavailable, or None if present.

    Does not import nvmath (uses importlib.util.find_spec under the hood).
    """
    return _unavailable_reason(_NVMATH_DEPS)


def cusolver_version() -> int | None:
    """cuSOLVER runtime version as major*1000 + minor*100 + patch, or None.

    Queries cusolverGetVersion through nvmath, against the exact cuSOLVER
    library nvmath dlopens (which is whatever the loader resolves -- the system
    CUDA install or the nvidia-cu* wheels a torch build depends on). The encoded
    value does not include the build number, so it can gate out runtimes that
    lack a symbol entirely but cannot distinguish patch builds.
    """
    try:
        from nvmath.bindings import cusolver  # pyrefly: ignore[missing-import]

        return cusolver.get_version()
    except Exception:
        return None


# One cuSOLVER handle per device, reused for the process lifetime. Handles are
# bound to the CUDA device that is current when cusolverDnCreate runs, so the
# handle must be created -- and later used -- with its device active.
#
# NOTE: this creates a dedicated handle rather than reusing aten's cuSOLVER
# handle. A future improvement is to reuse the handle PyTorch already maintains
# (see at::cuda::getCurrentCUDASolverDnHandle) to avoid the extra context.
_handles: dict[torch.device, int] = {}


def get_cusolver_handle(device: torch.device) -> int:
    """Return a cached cuSOLVER Dn handle bound to ``device``."""
    from nvmath.bindings import cusolverDn as cs  # pyrefly: ignore[missing-import]

    handle = _handles.get(device)
    if handle is None:
        # Create under the target device so the handle is bound to it.
        with torch.cuda.device(device):
            handle = cs.create()
        _handles[device] = handle
    return handle
<|endoftext|>import functools
import logging
import sys
from typing import cast

from torch._vendor.packaging.version import Version

from ..backends import cuda as _cuda
from .common_utils import (
    _available_version,
    _unavailable_reason,
    check_native_jit_disabled,
    check_native_version_skip,
)
from .dsl_registry import dsl_registry, DSLModuleProtocol
from .registry import (
    _OpCondFn,
    _OpImplFn,
    deregister_op_overrides as _deregister_op_overrides_impl,
    register_op_override as _register_op_override_impl,
)


log = logging.getLogger(__name__)


_CUTEDSL_DSL_NAME = "cutedsl"
_CUTEDSL_REQUIRED_VERSIONS: set[Version] = {
    # Current version - Note Version.from_part(release=(4.4.1)) is better
    #                   but > v26 of packaging.
    Version(f"{4}.{4}.{1}"),
    Version(f"{4}.{4}.{2}"),
    Version(f"{4}.{5}.{2}"),
}


@functools.cache
def _check_runtime_available() -> tuple[bool, Version | None]:
    """
    Check if cutedsl (and deps) are available.

    NOTE: Doesn't import at this point
    """
    # Skip all checks if running on CPU-only binary
    if not _cuda.is_built():
        return (False, None)

    # CuTeDSL kernels require genuine NVIDIA GPUs; on ROCm/HIP builds the
    # CUDA compatibility layer reports is_built()=True but cuInit() will fail
    # because no NVIDIA driver is present.
    import torch

    if torch.version.hip is not None:
        return (False, None)

    deps = [
        ("nvidia_cutlass_dsl", "cutlass"),
        ("apache_tvm_ffi", "tvm_ffi"),
    ]
    reason = _unavailable_reason(deps)
    if reason is None:
        available = True
        version = _available_version("nvidia_cutlass_dsl")
    else:
        # info, not warning: missing optional deps is the common case on stock
        # builds and we don't want to spam stderr on `import torch`. Surface
        # it via TORCH_LOGS=+native_dsl when diagnosing why an override is
        # silent.
        log.info(
            "CuTeDSL operators require optional Python packages "
            "`nvidia-cutlass-dsl` and `apache-tvm-ffi`; "
            "%s",
            reason,
        )
        available = False
        version = None
    return available, version


def runtime_available() -> bool:
    available, _ = _check_runtime_available()
    return available


def runtime_version() -> None | Version:
    _, version = _check_runtime_available()
    return version


@functools.cache
def _version_is_ok() -> bool:
    _, version = _check_runtime_available()
    if check_native_version_skip() or (version in _CUTEDSL_REQUIRED_VERSIONS):
        return True

    log.info(
        "cutedsl version %s is not known-good (ok: %s); "
        "set TORCH_NATIVE_SKIP_VERSION_CHECK=1 to override",
        version,
        _CUTEDSL_REQUIRED_VERSIONS,
    )
    return False


def deregister_op_overrides() -> None:
    """
    Deregister all ops through cuteDSL
    """
    _deregister_op_overrides_impl(disable_dsl_names=_CUTEDSL_DSL_NAME)


def register_op_override(
    lib_symbol: str,
    op_symbol: str,
    dispatch_key: str,
    cond: _OpCondFn | None,
    impl: _OpImplFn,
    *,
    allow_multiple_override: bool = False,
    unconditional_override: bool = False,
) -> None:
    """
    See torch/_native/registry.py for the underlying implementation
    and arguments. This is a thin, DSL-checking wrapper over
    _register_op_override_impl
    """
    available, version = _check_runtime_available()
    if (not available) or check_native_jit_disabled():
        return

    if not _version_is_ok():
        return

    _register_op_override_impl(
        _CUTEDSL_DSL_NAME,
        lib_symbol,
        op_symbol,
        dispatch_key,
        cond,
        impl,
        allow_multiple_override=allow_multiple_override,
        unconditional_override=unconditional_override,
    )


# Register this DSL module with the registry
# Note: Import-time registration ensures DSL is available when module is loaded
dsl_registry.register_dsl("cutedsl", cast(DSLModuleProtocol, sys.modules[__name__]))
<|endoftext|># Owner(s): ["module: dsl-native-ops"]

import functools
import logging
from typing import Protocol

from torch._vendor.packaging.version import Version

from .registry import _OpCondFn, _OpImplFn


log = logging.getLogger(__name__)


class DSLModuleProtocol(Protocol):
    """Complete interface for DSL utility modules"""

    def runtime_available(self) -> bool: ...
    def runtime_version(self) -> Version | None: ...

    def deregister_op_overrides(self) -> None: ...

    def register_op_override(
        self,
        lib_symbol: str,
        op_symbol: str,
        dispatch_key: str,
        cond: _OpCondFn | None,
        impl: _OpImplFn,
        *,
        allow_multiple_override: bool = False,
        unconditional_override: bool = False,
    ) -> None: ...


class DSLRegistry:
    """Registry for DSL modules - calls their existing API functions dynamically"""

    def __init__(self):
        self._dsl_modules: dict[str, DSLModuleProtocol] = {}

    def _validate_dsl_name(self, name: str) -> None:
        """Validate DSL name at runtime"""
        if not isinstance(name, str):
            raise TypeError(f"DSL name must be string, got {type(name).__name__}")

        if not name.strip():
            raise ValueError("DSL name cannot be empty or whitespace")

    def register_dsl(self, name: str, dsl_module: DSLModuleProtocol) -> None:
        """Register a DSL module with required interface"""
        # Runtime validation for name and module interface
        self._validate_dsl_name(name)

        # Validate that module implements the protocol
        required_methods = [
            "runtime_available",
            "runtime_version",
            "register_op_override",
            "deregister_op_overrides",
        ]
        missing_methods = [
            method for method in required_methods if not hasattr(dsl_module, method)
        ]
        if missing_methods:
            raise TypeError(
                f"DSL module '{name}' missing required methods: {missing_methods}"
            )

        # Handle duplicate registration case
        if name in self._dsl_modules:
            existing_module = self._dsl_modules[name]
            if existing_module is dsl_module:
                # Same module re-registering - this is OK (import-time registration)
                log.debug(
                    "DSL '%s' re-registered with same module",
                    name,
                )
                return
            else:
                # Different module object but same name - warn and allow (for testing)
                # This can happen when tests import modules directly
                log.warning(
                    "DSL '%s' re-registered with different module object (possibly from test imports)",
                    name,
                )
                # Continue to allow the registration

        # No cast needed - already properly typed
        self._dsl_modules[name] = dsl_module

        # Clear caches to prevent stale results after registration
        self.is_dsl_available.cache_clear()
        self.get_dsl_version.cache_clear()
        self.list_available_dsls.cache_clear()
        self.list_all_dsls.cache_clear()

        log.info("Successfully registered DSL: %s", name)

    @functools.cache  # noqa: B019
    def is_dsl_available(self, dsl_name: str) -> bool:
        """Check if DSL is available by calling its runtime_available()"""
        dsl_module = self._dsl_modules.get(dsl_name)
        if dsl_module is None:
            return False
        try:
            return dsl_module.runtime_available()
        except ImportError:
            log.debug("DSL %s import error", dsl_name, exc_info=True)
            return False
        except Exception:
            log.exception("Error checking availability for DSL %s", dsl_name)
            return False

    @functools.cache  # noqa: B019
    def get_dsl_version(self, dsl_name: str) -> Version | None:
        """Get DSL version by calling its runtime_version()"""
        dsl_module = self._dsl_modules.get(dsl_name)
        if dsl_module is None:
            return None
        try:
            return dsl_module.runtime_version()
        except Exception:
            log.debug("Error getting version for DSL %s", dsl_name, exc_info=True)
            return None

    @functools.cache  # noqa: B019
    def list_available_dsls(self) -> tuple[str, ...]:
        """Get names of currently available DSLs"""
        available = []
        for name in self._dsl_modules:
            if self.is_dsl_available(name):  # Use cached method
                available.append(name)
        return tuple(available)

    @functools.cache  # noqa: B019
    def list_all_dsls(self) -> tuple[str, ...]:
        """Get all registered DSL names (available or not)"""
        return tuple(self._dsl_modules.keys())

    def get_dsl_module(self, name: str) -> DSLModuleProtocol | None:
        """Get a registered DSL module by name.

        Args:
            name: Name of the DSL to retrieve.

        Returns:
            The DSL module if registered, None otherwise.
        """
        return self._dsl_modules.get(name)


# Global registry instance
dsl_registry = DSLRegistry()
<|endoftext|>"""Structured compile/cache instrumentation for ``torch._native`` ops.

Native DSL ops compile device kernels lazily on first call. Those compiles
are the dominant first-call latency, and a silent cache miss is a common
"why is this slow again?" question. This module surfaces both, with no
runtime cost unless ``TORCH_LOGS=+native_dsl_compile`` is set. That artifact
is off by default to avoid log spew: these wrappers sit on the compile
*cache*, so they run on every op call, cache hits included.

Two sinks, both gated on that artifact and fed by a single
:class:`CompileEvent`:

* The ``native_dsl_compile`` artifact logger: a one-line human-readable
  summary per compile -- outcome, wall time, and running hit/miss totals.
* ``trace_structured`` artifacts (tlparse): a JSON record per compile, for
  production jobs where only the structured trace is retrievable.

The DSLs compile through different machinery, so each has an entry point.
All reduce to the same shared core (:func:`_make_wrapper`): snapshot the
cache, time the call, snapshot again, and flag ``compiled`` when the miss
counter advanced. They differ only in how a snapshot is sampled:

* :func:`instrument_cutedsl_compile` -- for CuTeDSL, stacked *above* the
  vendored ``quack`` ``@jit_cache`` decorator. It reads the cache wrapper's
  ``cache_info()`` and times the wrapped ``cute.compile`` call::

      @instrument_cutedsl_compile("aten::topk")
      @jit_cache
      def _compile_topk_radix(N, K, deterministic): ...

  A ``cache_info().misses`` delta means the cache ran a real
  ``cute.compile``, so the measured wall time *is* the compile time;
  otherwise the key was served from the in-memory or on-disk ``.o`` cache.

* :func:`instrument_triton_kernel` -- for Triton ``@triton.jit`` kernels,
  which compile *and* launch in one ``kernel[grid](...)`` call and keep
  their own per-kernel cache (``JITFunction.device_caches``). Stacked above
  ``@triton.jit``, it wraps the kernel in a transparent proxy that watches
  *that one kernel's* variant count; a launch that grows it means a fresh
  compile. Scoping to the single kernel gives clean per-kernel attribution --
  two kernels in one module never collide. Because compile and launch are
  fused, ``wall_ms`` on a miss is compile + host-launch latency (compile
  dominates); on a hit it is just host-launch latency.

* :func:`instrument_helion_kernel` -- for regular and AOT Helion kernels. It
  tracks compiled configurations across the kernel's bound specialization
  cache; a growing count means the call compiled a new configuration.

The DSLs only expose miss-side signal directly (CuTeDSL's vendored cache
reports aggregate counters; Triton's cache only grows), so finer reasons
(disk-hit vs lock-timeout, Triton's on-disk cache) are not distinguished
here -- the boolean ``compiled`` flag plus wall time covers the common case.

Neither entry point touches the underlying DSL/vendored code, and neither
hijacks Triton's process-global ``knobs.runtime`` hooks (those would also
capture Inductor's unrelated Triton compiles).
"""

from __future__ import annotations

import functools
import time
from dataclasses import asdict, dataclass
from typing import Any, TYPE_CHECKING, TypeVar

import torch._logging


if TYPE_CHECKING:
    from collections.abc import Callable


__all__ = [
    "CompileEvent",
    "InstrumentedHelionKernel",
    "InstrumentedTritonKernel",
    "instrument_cutedsl_compile",
    "instrument_helion_kernel",
    "instrument_triton_kernel",
    "instrumented_cutedsl_cache",
    "instrumented_helion_kernel",
    "instrumented_triton_cache",
]

# tlparse artifact name, and the TORCH_LOGS switch gating both sinks. The
# "artifact" envelope (see trace_structured_artifact) is the well-supported
# transport; the name lets tlparse group these events.
_ARTIFACT_NAME = "native_dsl_compile"

log = torch._logging.getArtifactLogger(__name__, _ARTIFACT_NAME)

R = TypeVar("R")


@dataclass(frozen=True)
class CompileEvent:
    """One compile-function invocation, as recorded to logs and tlparse.

    ``compiled`` is the ground truth (did the cache run a real compile);
    ``outcome`` is its human-readable form. ``hits`` / ``misses`` are the
    cache's running totals *after* this call, useful for spotting churn
    (e.g. misses climbing across calls of the same shape => keys not
    stable, or the persistent cache is disabled).
    """

    op: str
    dsl: str
    outcome: str
    compiled: bool
    wall_ms: float
    key: str
    hits: int
    misses: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _listening() -> bool:
    """True if the ``native_dsl_compile`` artifact is enabled.

    Lets the wrapper skip all instrumentation work (cache sampling, timing,
    key formatting, event construction) on the hot path when nothing is
    listening -- important because the CuTeDSL wrapper sits on the compile
    *cache* and so runs on every op call, cache hits included.

    Both sinks share this one gate; tlparse emission cannot be left to
    ``trace_structured``'s internal ``trace_log.handlers`` check, which is
    true in any job collecting a structured trace.

    ``log_state`` is read off the module because ``TORCH_LOGS`` parsing
    rebinds it.
    """
    from torch._logging import _internal

    return _internal.log_state.is_artifact_enabled(_ARTIFACT_NAME)


def _emit(event: CompileEvent) -> None:
    """Fan the event out to the artifact logger and tlparse."""
    log.info(
        "%s [%s] %s in %.1fms (key=%s, cache hits=%d misses=%d)",
        event.op,
        event.dsl,
        event.outcome,
        event.wall_ms,
        event.key,
        event.hits,
        event.misses,
    )

    # Local import keeps `import torch._native` from pulling torch._logging's
    # heavier transitive imports at registration time.
    from torch._logging._internal import trace_structured

    # Same "artifact" envelope as Dynamo's trace_structured_artifact(); we call
    # trace_structured directly only to pass expect_trace_id=False, since that
    # helper would capture an expensive stack on every eager event (no live
    # CompileContext). trace_structured still reads the live trace id, so a
    # native op compiling inside torch.compile is auto-tagged with the ambient
    # frame ids and nests under that compile in tlparse.
    trace_structured(
        "artifact",
        metadata_fn=lambda: {"name": _ARTIFACT_NAME, "encoding": "json"},
        expect_trace_id=False,
        payload_fn=lambda: _json_payload(event),
    )


def _json_payload(event: CompileEvent) -> str:
    import json

    return json.dumps(event.as_dict(), sort_keys=True)


def _format_key(args: tuple, kwargs: dict, key_fn: Callable | None) -> str:
    if key_fn is not None:
        try:
            return key_fn(*args, **kwargs)
        except Exception:
            pass
    parts = [repr(a) for a in args]
    parts += [f"{k}={v!r}" for k, v in sorted(kwargs.items())]
    return "(" + ", ".join(parts) + ")"


def _make_wrapper(
    fn: Callable[..., R],
    op: str,
    dsl: str,
    key_fn: Callable[..., str] | None,
    sample: Callable[[], tuple[int | None, int | None]],
) -> Callable[..., R]:
    """Shared instrumentation core for both DSL entry points.

    ``sample()`` returns a ``(hits, misses)`` snapshot of the relevant cache;
    a ``misses`` increase across the call means a real compile fired. Timing,
    error handling, classification, and emission are identical across DSLs --
    only ``sample`` (and the reported ``dsl``) differ.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> R:
        # Fast path: do nothing extra unless a sink is listening. This runs on
        # every call to the wrapped fn (for CuTeDSL, every op invocation), so
        # the no-listener cost must stay at one predicate check.
        if not _listening():
            return fn(*args, **kwargs)

        _, misses_before = sample()
        start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            outcome_is_error = False
        except Exception:
            outcome_is_error = True
            raise
        finally:
            wall_ms = (time.perf_counter() - start) * 1e3
            hits_after, misses_after = sample()
            compiled = (
                not outcome_is_error
                and misses_before is not None
                and misses_after is not None
                and misses_after > misses_before
            )
            if outcome_is_error:
                outcome = "error"
            else:
                outcome = "compiled" if compiled else "cache_hit"
            _emit(
                CompileEvent(
                    op=op,
                    dsl=dsl,
                    outcome=outcome,
                    compiled=compiled,
                    wall_ms=wall_ms,
                    key=_format_key(args, kwargs, key_fn),
                    hits=hits_after or 0,
                    misses=misses_after or 0,
                )
            )
        return result

    return wrapper


def _cache_info_sampler(fn: Any) -> Callable[[], tuple[int | None, int | None]]:
    """Sampler reading ``(hits, misses)`` from a ``jit_cache`` wrapper.

    Defensive: the wrapped callable may not be ``jit_cache``-decorated (e.g.
    a plain function in tests). Then we report ``(None, None)`` and the core
    still times the call, just without compile/hit classification.
    """

    def sample() -> tuple[int | None, int | None]:
        info = getattr(fn, "cache_info", None)
        if info is None:
            return None, None
        try:
            ci = info()
            return ci.hits, ci.misses
        except Exception:
            return None, None

    return sample


def instrument_cutedsl_compile(
    op: str,
    *,
    key_fn: Callable[..., str] | None = None,
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """Instrument a CuTeDSL (``@jit_cache``-decorated) compile function.

    Args:
        op: Operator symbol being compiled for, e.g. ``"aten::topk"``.
        key_fn: Optional callable with the wrapped function's signature
            returning a short string describing the compile key for logs.
            Defaults to a repr of the args/kwargs.

    Returns a decorator. The decorated function behaves identically to the
    original (same return value, same caching); it only adds a log line and
    a tlparse artifact per call. Errors raised by the wrapped compile are
    timed, reported with ``outcome="error"``, and re-raised unchanged.
    """

    def decorator(fn: Callable[..., R]) -> Callable[..., R]:
        wrapper = _make_wrapper(fn, op, "cutedsl", key_fn, _cache_info_sampler(fn))
        # Forward jit_cache's bespoke attributes (functools.wraps doesn't copy
        # them) so the instrumented function stays a drop-in for callers that
        # introspect the cache.
        for attr in ("cache", "cache_clear", "cache_info"):
            if hasattr(fn, attr):
                setattr(wrapper, attr, getattr(fn, attr))
        return wrapper

    return decorator


def _triton_cache_size(kernel: Any) -> int | None:
    """Compiled-variant count across one JITFunction's per-device caches.

    Triton stores one entry per specialized variant in
    ``JITFunction.device_caches[device]``, a ``(kernel_cache, ...)`` tuple
    whose first element is the dict of compiled kernels. The count grows by
    one each time a new (signature, constexpr, options) variant compiles, so
    a delta across a launch tells us whether *this* kernel compiled. Returns
    None if the object doesn't expose ``device_caches`` (a future Triton that
    renames this, or a non-kernel in tests).
    """
    caches = getattr(kernel, "device_caches", None)
    if caches is None:
        return None
    try:
        return sum(len(per_device[0]) for per_device in caches.values())
    except Exception:
        return None


class InstrumentedTritonKernel:
    """Transparent proxy over a single ``@triton.jit`` kernel.

    Watches only this kernel's ``device_caches``, so a launch that grows the
    variant count is unambiguously *this* kernel compiling -- two kernels in
    one module never collide. All attribute access except the ``kernel[grid]``
    launch path is delegated to the wrapped kernel, so the proxy is a drop-in
    for the original ``JITFunction`` for attribute/method use (``.warmup``,
    ``.cache_key``).

    Composition with ``torch.library.wrap_triton``: that API does a strict
    ``isinstance(JITFunction)`` check and would reject this proxy, but it
    targets a different path (export/tracing, where the compile does not run
    through our eager launch). When a native op needs ``wrap_triton``, pass the
    raw kernel via :attr:`jit_kernel`: ``wrap_triton(my_kernel.jit_kernel)``.
    """

    def __init__(
        self,
        kernel: Any,
        op: str,
        key_fn: Callable[..., str] | None,
    ) -> None:
        self._kernel = kernel
        self._op = op
        self._key_fn = key_fn

    @property
    def jit_kernel(self) -> Any:
        """The wrapped raw ``JITFunction`` (e.g. for ``wrap_triton``)."""
        return self._kernel

    def _sample(self) -> tuple[int | None, int | None]:
        # Triton has no hit counter; this kernel's variant count stands in for
        # `misses`, so a delta across the launch means a fresh compile.
        return None, _triton_cache_size(self._kernel)

    def __getitem__(self, grid: Any) -> Callable[..., Any]:
        # kernel[grid](*args) is the launch path -- wrap the bound launcher in
        # the shared core so the compile that may fire inside it is recorded.
        launcher = self._kernel[grid]
        return _make_wrapper(launcher, self._op, "triton", self._key_fn, self._sample)

    def __getattr__(self, name: str) -> Any:
        # Delegate everything else (warmup, cache_key, wrap_triton hooks, ...)
        # to the real kernel. Only triggered for names not set on the proxy.
        return getattr(self._kernel, name)


def instrument_triton_kernel(
    op: str,
    *,
    key_fn: Callable[..., str] | None = None,
) -> Callable[[Any], InstrumentedTritonKernel]:
    """Instrument a single ``@triton.jit`` kernel, stacked above the jit::

        @instrument_triton_kernel("aten::bmm")
        @triton.jit
        def _bmm_kernel(...): ...

    Unlike CuTeDSL, a Triton kernel compiles lazily *inside* its
    ``kernel[grid](...)`` launch and caches variants on the kernel object. So
    rather than wrap a separate compile fn, wrap the kernel itself: the proxy
    watches that one kernel's variant count and records a compile when the
    launch grows it. Because compile and launch are fused, ``wall_ms`` on a
    miss is compile + host-launch latency (compile dominates); on a hit it is
    just host-launch latency.

    Args:
        op: Operator symbol being compiled for, e.g. ``"aten::bmm"``.
        key_fn: Optional callable with the kernel's launch-arg signature
            returning a short string for logs (the constexprs it sees, e.g.
            ``BLOCK``, are the actual compile key). Defaults to a repr.
    """

    def decorator(kernel: Any) -> InstrumentedTritonKernel:
        return InstrumentedTritonKernel(kernel, op, key_fn)

    return decorator


def _helion_cache_size(kernel: Any) -> int | None:
    """Compiled-config count across a Helion Kernel's bound specializations.

    Helion stores compiled configs in ``Kernel._bound_kernels[*]._compile_cache``;
    the total grows by one each time a new configuration compiles, so a delta
    across a call tells us whether *this* kernel compiled. Returns None if the
    object doesn't expose these private attributes (a future Helion that renames
    them, or a non-kernel in tests).
    """
    bound_kernels = getattr(kernel, "_bound_kernels", None)
    if bound_kernels is None:
        return None
    try:
        return sum(
            len(getattr(bound, "_compile_cache", ()))
            for bound in bound_kernels.values()
        )
    except Exception:
        return None


class InstrumentedHelionKernel:
    """Transparent proxy over a Helion ``Kernel`` compile cache."""

    def __init__(
        self,
        kernel: Any,
        op: str,
        key_fn: Callable[..., str] | None,
    ) -> None:
        self._kernel = kernel
        self._wrapped = _make_wrapper(kernel, op, "helion", key_fn, self._sample)

    @property
    def helion_kernel(self) -> Any:
        return self._kernel

    def _sample(self) -> tuple[int | None, int | None]:
        return None, _helion_cache_size(self._kernel)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._wrapped(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._kernel, name)


def instrument_helion_kernel(
    op: str,
    *,
    key_fn: Callable[..., str] | None = None,
) -> Callable[[Any], InstrumentedHelionKernel]:
    """Instrument a Helion ``Kernel`` without importing Helion eagerly."""

    def decorator(kernel: Any) -> InstrumentedHelionKernel:
        return InstrumentedHelionKernel(kernel, op, key_fn)

    return decorator


# ---------------------------------------------------------------------------
# Combined decorators: the recommended one-decorator surface for native ops.
# They apply the DSL's own cache/jit *and* the instrumentation, so the op
# author writes a single decorator on a bare function and can't get the
# stacking order wrong. The lower-level instrument_* helpers above remain for
# custom composition.
# ---------------------------------------------------------------------------


def instrumented_cutedsl_cache(
    op: str,
    *,
    key_fn: Callable[..., str] | None = None,
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """Cache + instrument a CuTeDSL compile function in one decorator::

        @instrumented_cutedsl_cache("aten::topk")
        def _compile_topk_radix(N, K, deterministic):
            return cute.compile(...)

    Equivalent to ``instrument_cutedsl_compile(op) `` stacked above the
    vendored ``@jit_cache``. ``jit_cache`` is imported lazily so plain
    ``import torch._native`` doesn't pull in the CuTeDSL runtime.
    """
    # Lazy import: quack.cache pulls in cutlass, absent on CPU-only builds.
    from torch._vendor.quack.cache import jit_cache

    def decorator(fn: Callable[..., R]) -> Callable[..., R]:
        return instrument_cutedsl_compile(op, key_fn=key_fn)(jit_cache(fn))

    return decorator


def instrumented_triton_cache(
    op: str,
    *,
    key_fn: Callable[..., str] | None = None,
) -> Callable[[Callable[..., R]], InstrumentedTritonKernel]:
    """JIT + instrument a Triton kernel in one decorator::

        @instrumented_triton_cache("aten::bmm")
        def _bmm_kernel(...):  # bare kernel body, no @triton.jit
            ...

    Equivalent to ``instrument_triton_kernel(op)`` stacked above
    ``@triton.jit``. ``triton`` is imported lazily so plain
    ``import torch._native`` doesn't pull in the Triton runtime.
    """
    # Lazy import: keep Triton out of `import torch._native`.
    import triton

    def decorator(fn: Callable[..., R]) -> InstrumentedTritonKernel:
        return instrument_triton_kernel(op, key_fn=key_fn)(triton.jit(fn))

    return decorator


def instrumented_helion_kernel(
    op: str,
    *,
    aot: bool = False,
    key_fn: Callable[..., str] | None = None,
    **settings: object,
) -> Callable[[Callable[..., R]], InstrumentedHelionKernel]:
    """Create and instrument a regular or AOT Helion kernel."""
    import helion  # pyrefly: ignore[missing-import]
    import helion.experimental  # pyrefly: ignore[missing-import]

    kernel_decorator = helion.experimental.aot_kernel if aot else helion.kernel

    def decorator(fn: Callable[..., R]) -> InstrumentedHelionKernel:
        return instrument_helion_kernel(op, key_fn=key_fn)(
            kernel_decorator(**settings)(fn)
        )

    return decorator
<|endoftext|>import os
import warnings
from functools import cache
from typing import cast

# This handles collecting registration of all native ops
# Also need to import DSL utils to make sure DSL registration is ok
from . import cutedsl_utils, dsl_registry, helion_utils, ops, registry, triton_utils


@cache
def get_user_ordering_fn() -> registry.UserOrderingFn | None:
    """
    Get a user-supplied graph-ordering function if specified.

    Pass in a `package.submodule.fn` string to the env variable
    `TORCH_PYTHON_NATIVE_USER_GRAPH_ORDER_FN` that implements the
    calling API described in `torch/_native/README.md`. This function
    must be part of an importable path.

    Return either the imported function or `None`
    """
    env_var = os.getenv("TORCH_PYTHON_NATIVE_USER_GRAPH_ORDER_FN")

    if not env_var:
        return None

    try:
        import importlib

        # Split into "package.submodule.fn_name
        module_name, fn_name = env_var.rsplit(".", 1)

        module = importlib.import_module(module_name)
        fn = getattr(module, fn_name)

        if not callable(fn):
            raise TypeError(f"{env_var} does not describe a callable function")

        # Cast needed: getattr returns object, but we've verified fn is callable with correct signature
        return cast(registry.UserOrderingFn, fn)
    except Exception as e:
        raise ValueError(
            f"Could not resolve {env_var} into an importable & callable function"
        ) from e


user_order_fn = get_user_ordering_fn()
if user_order_fn:
    registry.reorder_graphs_from_user_function(user_order_fn)


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="Warning only once for all operators,  other operators may also be overridden\\.",
        category=UserWarning,
    )
    registry._register_all_overrides()
<|endoftext|>import functools
import logging
import sys
from typing import cast

from torch._vendor.packaging.version import Version

from ..backends import cuda as _cuda
from .common_utils import (
    _available_version,
    _unavailable_reason,
    check_native_jit_disabled,
    check_native_version_skip,
)
from .dsl_registry import dsl_registry, DSLModuleProtocol
from .registry import (
    _OpCondFn,
    _OpImplFn,
    deregister_op_overrides as _deregister_op_overrides_impl,
    register_op_override as _register_op_override_impl,
)


log = logging.getLogger(__name__)


_TRITON_DSL_NAME = "triton"
_TRITON_REQUIRED_VERSION_MAJOR = 3
_TRITON_MINIMUM_VERSION_MINOR = 6


@functools.cache
def _check_runtime_available() -> tuple[bool, Version | None]:
    """
    Check if triton is available

    NOTE: must not import at this point
    """
    # Skip all checks if running on CPU-only binary
    if not _cuda.is_built():
        return (False, None)

    deps = [
        ("triton", "triton"),
    ]
    reason = _unavailable_reason(deps)
    if reason is None:
        available = True
        version = _available_version("triton")
    else:
        # info, not warning: see cutedsl_utils._check_runtime_available for
        # rationale (missing optional deps is the common case; surface via
        # TORCH_LOGS=+native_dsl when needed).
        log.info("triton native DSL ops require: `triton` %s", reason)
        available = False
        version = None
    return available, version


def runtime_available() -> bool:
    available, _ = _check_runtime_available()
    return available


def runtime_version() -> None | Version:
    _, version = _check_runtime_available()
    return version


@functools.cache
def _version_is_sufficient() -> bool:
    _, version = _check_runtime_available()

    if version is None:
        return False

    # Either exact version, or same major
    major_ok = version.major == _TRITON_REQUIRED_VERSION_MAJOR
    minor_ok = version.minor >= _TRITON_MINIMUM_VERSION_MINOR

    if (major_ok and minor_ok) or check_native_version_skip():
        return True

    log.info(
        "triton version %s is not sufficient (>= (%s.%s.*)); "
        "set TORCH_NATIVE_SKIP_VERSION_CHECK=1 to override",
        version,
        _TRITON_REQUIRED_VERSION_MAJOR,
        _TRITON_MINIMUM_VERSION_MINOR,
    )
    return False


def deregister_op_overrides() -> None:
    """
    Deregister all ops through triton
    """
    _deregister_op_overrides_impl(disable_dsl_names=_TRITON_DSL_NAME)


def register_op_override(
    lib_symbol: str,
    op_symbol: str,
    dispatch_key: str,
    cond: _OpCondFn | None,
    impl: _OpImplFn,
    *,
    allow_multiple_override: bool = False,
    unconditional_override: bool = False,
) -> None:
    """
    See torch/_native/registry.py for the underlying implementation
    and arguments. This is a thin, DSL-checking wrapper over
    _register_op_override_impl
    """
    available, version = _check_runtime_available()
    if (not available) or check_native_jit_disabled():
        return

    if not _version_is_sufficient():
        return

    _register_op_override_impl(
        _TRITON_DSL_NAME,
        lib_symbol,
        op_symbol,
        dispatch_key,
        cond,
        impl,
        allow_multiple_override=allow_multiple_override,
        unconditional_override=unconditional_override,
    )


# Register this DSL module with the registry
# Note: Import-time registration ensures DSL is available when module is loaded
dsl_registry.register_dsl("triton", cast(DSLModuleProtocol, sys.modules[__name__]))
<|endoftext|>import importlib
import importlib.metadata
import os
from functools import cache

from torch._vendor.packaging import version as _packaging_version


@cache
def check_native_jit_disabled() -> bool:
    """
    Single point to check if native DSL ops are disabled globally,
    checked via:
    TORCH_DISABLE_NATIVE_JIT=1
    """
    return int(os.getenv("TORCH_DISABLE_NATIVE_JIT", 0)) == 1


def _unavailable_reason(deps: list[tuple[str, str]]) -> None | str:
    """
    Check availability of required packages - cuteDSL & deps,
    informing user what (if anything) is missing

    NOTE: Doesn't actually import anything.
    """
    for package_name, module_name in deps:
        # find_spec raises ModuleNotFoundError for dotted names (e.g.
        # "nvmath.bindings") when the parent package is absent.
        try:
            found = importlib.util.find_spec(module_name) is not None
        except (ModuleNotFoundError, ValueError):
            found = False
        if not found:
            return (
                f"missing optional dependency `{package_name}` "
                f"(importlib.util.find_spec({package_name}) failed)"
            )
    return None


def _available_version(package: str) -> _packaging_version.Version | None:
    """
    Get the installed version of a package as (major, minor, patch).

    Handles pre-release suffixes like "0.7.0rc1" or "3.1.0.post1" by
    stripping non-numeric tails from each component. Returns None on
    parse failure.
    """
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None

    try:
        v = _packaging_version.parse(version)
    except _packaging_version.InvalidVersion:
        return None

    return v


@cache
def check_native_version_skip() -> bool:
    """
    Single point to check if native DSL version gating should be skipped,
    checked via:
    TORCH_NATIVE_SKIP_VERSION_CHECK=1
    """
    return int(os.getenv("TORCH_NATIVE_SKIP_VERSION_CHECK", 0)) == 1
<|endoftext|>import torch


class ConstTensorWrapper:
    """Read-only, zero-copy view of a tensor for Triton kernel arguments.

    Triton extracts a kernel pointer argument by duck typing: any object with a
    ``data_ptr()`` method and a ``dtype`` is treated as a pointer (see Triton's
    ``jit.py``). ``ConstTensorWrapper`` overrides ``data_ptr()`` to return
    ``const_data_ptr()``, so passing a copy-on-write tensor to an argument the
    kernel only reads does not materialize it. The remaining attributes Triton
    inspects during specialization and launch are forwarded to the wrapped
    tensor.

    Only wrap arguments the kernel treats as read-only (loads, never stores).
    Triton does not enforce this; wrapping a written argument is a bug.

    Accessors run under DisableTorchFunctionSubclass so the wrapper never
    re-enters __torch_function__: const_data_ptr() and friends are dispatched
    methods, and re-entering while an override is mid-redispatch trips the
    "cannot skip two levels of __torch_function__" guard.
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def data_ptr(self) -> int:
        with torch._C.DisableTorchFunctionSubclass():
            # const_data_ptr() exists at runtime but is absent from the stubs.
            return self._tensor.const_data_ptr()  # type: ignore[attr-defined]

    @property
    def dtype(self) -> torch.dtype:
        return self._tensor.dtype

    @property
    def device(self) -> torch.device:
        return self._tensor.device

    @property
    def shape(self) -> torch.Size:
        return self._tensor.shape

    @property
    def ndim(self) -> int:
        return self._tensor.ndim

    def dim(self) -> int:
        with torch._C.DisableTorchFunctionSubclass():
            return self._tensor.dim()

    def size(self, dim: int | None = None) -> torch.Size | int:
        with torch._C.DisableTorchFunctionSubclass():
            return self._tensor.size() if dim is None else self._tensor.size(dim)

    def stride(self, dim: int | None = None) -> tuple[int, ...] | int:
        with torch._C.DisableTorchFunctionSubclass():
            return self._tensor.stride() if dim is None else self._tensor.stride(dim)

    def element_size(self) -> int:
        with torch._C.DisableTorchFunctionSubclass():
            return self._tensor.element_size()
<|endoftext|>"""Cached cublasLt grouped GEMM via nvmath bindings."""

import ctypes

from cuda.bindings.runtime import cudaDataType  # pyrefly: ignore[missing-import]

from nvmath.bindings import cublasLt  # pyrefly: ignore[missing-import]
from nvmath.bindings.cublas import (  # pyrefly: ignore[missing-import]
    ComputeType,
    Operation,
)

import torch


_cublaslt_workspaces: dict[tuple, torch.Tensor] = {}


def _get_cublaslt_workspace(device="cuda"):
    ws_size = torch.backends.cuda.cublaslt_workspace_size()
    key = (torch.device(device), ws_size)
    if key not in _cublaslt_workspaces:
        _cublaslt_workspaces[key] = torch.empty(
            ws_size, dtype=torch.uint8, device=device
        )
    return _cublaslt_workspaces[key], ws_size


def _set_attr(setter, handle, attr, val, ctype):
    buf = (ctype * 1)(val)
    setter(handle, attr, ctypes.addressof(buf), ctypes.sizeof(buf))


class ForeachMMCublasLt:
    """Cached cublasLt grouped GEMM. Supports uniform or mixed shapes per group."""

    def __init__(
        self,
        shapes,
        G,
        a_row_major=True,
        b_row_major=True,
        dtype=torch.bfloat16,
        device="cuda",
    ):
        """
        Args:
            shapes: tuple of (M, N, K) triples, one per group
            G: number of groups (must equal len(shapes))
        """
        if dtype != torch.bfloat16:
            raise ValueError(f"ForeachMMCublasLt only supports bf16, got {dtype}")

        self.G = G
        self._dtype = dtype
        elem_size = dtype.itemsize
        alignment = 16 // elem_size

        # cuBLAS col-major: C^T = mat2^T * self^T
        # cuBLAS-A = mat2, cuBLAS-B = self
        opa = Operation.N if b_row_major else Operation.T
        opb = Operation.N if a_row_major else Operation.T

        # Per-group dimensions
        Ms, Ns, Ks = zip(*shapes)
        ldas = [n if b_row_major else k for n, k in zip(Ns, Ks)]
        ldbs = [k if a_row_major else m for m, k in zip(Ms, Ks)]
        N_paddeds = [(n + alignment - 1) // alignment * alignment for n in Ns]

        for vals in (Ms, Ns, Ks, ldas, ldbs, N_paddeds):
            for v in vals:
                if v > 2**31 - 1:
                    raise ValueError(
                        f"ForeachMMCublasLt: dimension {v} exceeds int32 range"
                    )

        # Per-group output layout: allocate one contiguous buffer with
        # max(M) x max(N_padded) per group, then narrow each slice to actual (M, N)
        self._out_Ns = Ns
        self._out_Ms = Ms
        self._out_N_paddeds = N_paddeds
        self._max_M = max(Ms)
        self._max_N_padded = max(N_paddeds)
        self._out_stride_bytes = self._max_M * self._max_N_padded * elem_size
        self._uniform = len(set(Ms)) == 1 and len(set(Ns)) == 1
        # ldd must match the physical stride in the contiguous output buffer
        N_paddeds = [self._max_N_padded] * G

        # Device-side dim arrays (6 x G)
        # cublas_m=N, cublas_n=M, cublas_k=K
        cublas_ms = list(Ns)
        cublas_ns = list(Ms)
        cublas_ks = list(Ks)
        dims = torch.tensor(
            [cublas_ms, cublas_ns, cublas_ks, ldas, ldbs, N_paddeds],
            dtype=torch.int32,
            device=device,
        )
        d_m, d_n, d_k, d_lda, d_ldb, d_ldd = [dims[i] for i in range(6)]

        # Pinned host + device buffers for pointer arrays [Aptr|Bptr|Dptr]
        self._pinned = torch.empty(3 * G, dtype=torch.int64, pin_memory=True)
        self._pinned_raw = ctypes.cast(
            self._pinned.data_ptr(), ctypes.POINTER(ctypes.c_int64)
        )
        self._dev_ptrs = torch.empty(3 * G, dtype=torch.int64, device=device)

        # alpha=1.0, beta=0.0 as device float32
        scalars = torch.tensor([1.0, 0.0], dtype=torch.float32, device=device)
        self._alpha_ptr = scalars.data_ptr()
        self._beta_ptr = scalars.data_ptr() + 4
        self._scalars = scalars  # prevent GC

        self._workspace, self._ws_bytes = _get_cublaslt_workspace(device)
        self._ws_ptr = self._workspace.data_ptr()
        self._dims = dims  # prevent GC

        # cublasLt matmul descriptor
        # Reuse PyTorch's cuBLAS handle (cublasHandle_t is valid as cublasLtHandle_t)
        handle = torch.cuda.current_blas_handle()
        self._handle = handle
        desc = cublasLt.matmul_desc_create(
            ComputeType.COMPUTE_32F, cudaDataType.CUDA_R_32F
        )

        def _set_desc(attr, val):
            _set_attr(
                cublasLt.matmul_desc_set_attribute, desc, attr, val, ctypes.c_int32
            )

        _set_desc(cublasLt.MatmulDescAttribute.TRANSA, opa)
        _set_desc(cublasLt.MatmulDescAttribute.TRANSB, opb)
        _set_desc(
            cublasLt.MatmulDescAttribute.POINTER_MODE, cublasLt.PointerMode.DEVICE
        )
        self._desc = desc

        # Grouped matrix layouts: A=mat2, B=self, D=output (C reuses D since beta=0)
        # Layout rows/cols are the physical dimensions seen by cuBLAS (before op)
        cdt = cudaDataType.CUDA_R_16BF
        _gl = cublasLt.grouped_matrix_layout_create
        a_rows = d_m if opa == Operation.N else d_k
        a_cols = d_k if opa == Operation.N else d_m
        b_rows = d_k if opb == Operation.N else d_n
        b_cols = d_n if opb == Operation.N else d_k
        self._Ad = _gl(cdt, G, a_rows.data_ptr(), a_cols.data_ptr(), d_lda.data_ptr())
        self._Bd = _gl(cdt, G, b_rows.data_ptr(), b_cols.data_ptr(), d_ldb.data_ptr())
        self._Dd = _gl(cdt, G, d_m.data_ptr(), d_n.data_ptr(), d_ldd.data_ptr())

        # Heuristic search (run once, algo reused for all calls)
        pref = cublasLt.matmul_preference_create()
        Attr = cublasLt.MatmulPreferenceAttribute
        _set_attr(
            cublasLt.matmul_preference_set_attribute,
            pref,
            Attr.MAX_WORKSPACE_BYTES,
            self._ws_bytes,
            ctypes.c_uint64,
        )
        avg_N = sum(Ns) // G
        avg_M = sum(Ms) // G
        avg_K = sum(Ks) // G
        for attr, val in [
            (Attr.GROUPED_DESC_D_AVERAGE_ROWS, avg_N),
            (Attr.GROUPED_DESC_D_AVERAGE_COLS, avg_M),
            (Attr.GROUPED_AVERAGE_REDUCTION_DIM, avg_K),
        ]:
            _set_attr(
                cublasLt.matmul_preference_set_attribute,
                pref,
                attr,
                val,
                ctypes.c_int64,
            )

        # cublasLtMatmulHeuristicResult_t is 96 bytes; algo field is at offset 0
        heur_buf = (ctypes.c_byte * 96)()
        ret = (ctypes.c_int32 * 1)(0)
        cublasLt.matmul_algo_get_heuristic(
            handle,
            desc,
            self._Ad,
            self._Bd,
            self._Dd,
            self._Dd,
            pref,
            1,
            ctypes.addressof(heur_buf),
            ctypes.addressof(ret),
        )
        if ret[0] == 0:
            raise RuntimeError("cublasLt grouped GEMM: no algorithm found")
        cublasLt.matmul_preference_destroy(pref)

        self._algo_ptr = ctypes.addressof(heur_buf)
        self._heur_buf = heur_buf  # prevent GC

        # Cached pointer offsets into _dev_ptrs
        base = self._dev_ptrs.data_ptr()
        ptr_size = torch.int64.itemsize
        self._dev_Aptr = base
        self._dev_Bptr = base + G * ptr_size
        self._dev_Dptr = base + 2 * G * ptr_size

    def __call__(self, self_list, mat2_list):
        G = self.G
        max_M = self._max_M
        max_Np = self._max_N_padded

        # Single contiguous allocation, sliced per group
        out_buf = torch.empty(
            G, max_M, max_Np, dtype=self._dtype, device=self._dev_ptrs.device
        )

        # Fill pinned host buffer with device pointers, async copy to GPU
        h = self._pinned_raw
        out_base = out_buf.data_ptr()
        out_stride = self._out_stride_bytes
        for i in range(G):
            h[i] = mat2_list[i].data_ptr()
            h[G + i] = self_list[i].data_ptr()
            h[2 * G + i] = out_base + i * out_stride
        self._dev_ptrs.copy_(self._pinned, non_blocking=True)

        cublasLt.matmul(
            self._handle,
            self._desc,
            self._alpha_ptr,
            self._dev_Aptr,
            self._Ad,
            self._dev_Bptr,
            self._Bd,
            self._beta_ptr,
            self._dev_Dptr,
            self._Dd,  # C (unused, beta=0)
            self._dev_Dptr,
            self._Dd,  # D (output)
            self._algo_ptr,
            self._ws_ptr,
            self._ws_bytes,
            torch.cuda.current_stream().cuda_stream,
        )

        # unbind is a single C++ call creating all G views at once.
        # Narrow only when output shape differs from the buffer slot size.
        results = out_buf.unbind(0)
        if self._uniform:
            M, N = self._out_Ms[0], self._out_Ns[0]
            if M != self._max_M or N != self._max_N_padded:
                return [r[:M, :N] for r in results]
            return list(results)
        return [r[: self._out_Ms[i], : self._out_Ns[i]] for i, r in enumerate(results)]

    def __del__(self):
        for attr in ("_Dd", "_Bd", "_Ad"):
            d = getattr(self, attr, None)
            if d is not None:
                try:
                    cublasLt.matrix_layout_destroy(d)
                except Exception:
                    pass
        desc = getattr(self, "_desc", None)
        if desc is not None:
            try:
                cublasLt.matmul_desc_destroy(desc)
            except Exception:
                pass
<|endoftext|>"""
Python override for aten::_foreach_mm. Supports bf16 and fp32.

Dispatch prerequisites:
  CUDA SM90+, not HIP
  equal-length lists with >= 2 tensors
  same CUDA device, same bf16/fp32 dtype
  2D dense/non-overlapping tensors
  same row/column-major orientation across groups
  matching K, K,N > 0, K,N 16-byte aligned

Dispatch:
  nvmath cublasLt grouped GEMM:
  - bf16
  - nvmath installed
  |-- YES --> nvmath cublasLt grouped GEMM (uniform and mixed shapes, any G)
  |
  torch._grouped_mm with offs:
  - non-nvmath path
  - max(M,N,K) < 2048
  - uniform K,N
  - G < 1024
  |-- YES --> torch.cat + torch._grouped_mm with offs (uniform or variable-M)
  |
  fallback loop:
  - non-nvmath path
  - max(M,N,K) < 2048
  |-- YES --> fallback loop: torch.mm per input pair
  |
  |-- else -> native _foreach_mm fallback (loop of at::mm)

Notes:
- cublasLt grouped GEMM is bf16-only (fp32 returns CUBLAS_STATUS_NOT_SUPPORTED).
- At dim >= 2048 each mm saturates the GPU; native fallback is equivalent.
"""

import warnings
from functools import cache
from typing import Literal

import torch

from ... import registry
from ...common_utils import _unavailable_reason


_nvmath_warned = False

_NVMATH_DEPS = [
    ("nvmath-python", "nvmath.bindings"),
]

# cublasLtGroupedMatrixLayoutCreate, used by the nvmath grouped GEMM path, was
# introduced in cuBLAS 13.2.0 (CUDA 13.1) and is experimental as of that release.
# cublasLtGetVersion() encodes the version as 10000*major + 100*minor + patch.
# This is the cuBLAS library version, not the CUDA toolkit version.
_MIN_CUBLASLT_GROUPED_GEMM_VERSION = 130200


# Cached: availability is a process-global property of the loaded cuBLASLt.
@cache
def _check_nvmath_cublaslt() -> bool:
    if _unavailable_reason(_NVMATH_DEPS) is not None:
        return False
    try:
        from nvmath.bindings import cublasLt  # pyrefly: ignore[missing-import]

        return cublasLt.get_version() >= _MIN_CUBLASLT_GROUPED_GEMM_VERSION
    except Exception:
        return False


def _k_n_16_byte_aligned(a: torch.Tensor, b: torch.Tensor, elem_size: int) -> bool:
    return (a.size(1) * elem_size) % 16 == 0 and (b.size(1) * elem_size) % 16 == 0


def _foreach_mm_cond(
    self: list[torch.Tensor],
    mat2: list[torch.Tensor],
) -> bool:
    if len(self) < 2 or len(self) != len(mat2):
        return False

    first_a = self[0]
    first_b = mat2[0]

    if not first_a.is_cuda or not first_b.is_cuda:
        return False
    if torch.version.hip:
        return False
    if first_a.dtype != first_b.dtype:
        return False
    if first_a.dtype not in {torch.bfloat16, torch.float32}:
        return False
    props = torch.cuda.get_device_properties(first_a.device)
    if props.major < 9:
        return False

    first_a_rm = first_a.stride(-1) == 1
    first_b_rm = first_b.stride(-1) == 1

    elem_size = first_a.element_size()
    nvmath_ok = first_a.dtype == torch.bfloat16 and _check_nvmath_cublaslt()
    max_dim = 0

    for i in range(len(self)):
        a, b = self[i], mat2[i]
        if a.dim() != 2 or b.dim() != 2:
            return False
        if a.size(1) != b.size(0):
            return False
        # K or N == 0 not supported by cutlass_grouped_mm
        if a.size(1) == 0 or b.size(1) == 0:
            return False
        if not torch.ops.aten.is_non_overlapping_and_dense.default(a):
            return False
        if not torch.ops.aten.is_non_overlapping_and_dense.default(b):
            return False
        if not a.is_cuda or not b.is_cuda:
            return False
        if a.device != first_a.device or b.device != first_a.device:
            return False
        if a.dtype != first_a.dtype or b.dtype != first_a.dtype:
            return False
        K_i, N_i = a.size(1), b.size(1)
        aligned = _k_n_16_byte_aligned(a, b, elem_size)
        if nvmath_ok and not aligned:
            nvmath_ok = False
        if not nvmath_ok and not aligned:
            return False
        max_dim = max(max_dim, a.size(0), K_i, N_i)
        if i > 0:
            if (a.stride(-1) == 1) != first_a_rm or (b.stride(-1) == 1) != first_b_rm:
                return False

    if not nvmath_ok and max_dim >= 2048:
        return False

    return True


def _foreach_mm_impl_stack(
    self: list[torch.Tensor],
    mat2: list[torch.Tensor],
) -> list[torch.Tensor]:
    # Router guarantees uniform K,N and group count < _CUTLASS_GROUPED_MM_GROUP_LIMIT.
    # _grouped_mm with offs handles uniform and variable-M shapes in one kernel.
    M_sizes = [a.size(0) for a in self]
    A_cat = torch.cat(self, dim=0)  # (sum_M, K)
    B_3d = torch.stack(mat2)  # (G, K, N)
    offs = torch.tensor(M_sizes, dtype=torch.int32, device=self[0].device).cumsum(
        0, dtype=torch.int32
    )
    out_cat = torch._grouped_mm(A_cat, B_3d, offs=offs)  # (sum_M, N)
    return list(out_cat.split(M_sizes, dim=0))


_ForeachMMCublasLt = None


def _get_nvmath_cls():
    global _ForeachMMCublasLt
    if _ForeachMMCublasLt is None:
        from torch._native.ops.foreach_mm.nvmath_impl import ForeachMMCublasLt

        _ForeachMMCublasLt = ForeachMMCublasLt
    return _ForeachMMCublasLt


_nvmath_cache: dict[tuple, object] = {}


def _foreach_mm_impl_nvmath(
    self: list[torch.Tensor],
    mat2: list[torch.Tensor],
) -> list[torch.Tensor]:
    G = len(self)
    first_a = self[0]
    a_row_major = first_a.stride(-1) == 1
    b_row_major = mat2[0].stride(-1) == 1
    shapes = tuple((a.size(0), b.size(1), a.size(1)) for a, b in zip(self, mat2))
    key = (shapes, a_row_major, b_row_major, first_a.device)
    if key not in _nvmath_cache:
        _nvmath_cache[key] = _get_nvmath_cls()(
            shapes, G, a_row_major=a_row_major, b_row_major=b_row_major
        )
    return _nvmath_cache[key](self, mat2)  # pyrefly: ignore[not-callable]


# torch._grouped_mm's bf16 kernel requires group_count < 1024; nvmath has no cap.
_CUTLASS_GROUPED_MM_GROUP_LIMIT = 1024

_Route = Literal["nvmath_cublaslt_grouped_mm", "cutlass_grouped_mm", "mm_loop"]


def _can_use_nvmath_cublaslt_grouped_mm(
    self: list[torch.Tensor],
    mat2: list[torch.Tensor],
) -> bool:
    # Metadata check only (bf16 + 16-byte-aligned N,K); does not check nvmath is installed
    # (_check_nvmath_cublaslt). dtype on [0]; _foreach_mm_cond ensures uniform.
    if self[0].dtype != torch.bfloat16 or mat2[0].dtype != torch.bfloat16:
        return False
    elem_size = self[0].element_size()
    return all(_k_n_16_byte_aligned(a, b, elem_size) for a, b in zip(self, mat2))


def _can_use_cutlass_grouped_mm(
    self: list[torch.Tensor],
    mat2: list[torch.Tensor],
) -> bool:
    # torch._grouped_mm needs uniform K,N; its bf16 kernel needs < cap groups
    if len(self) >= _CUTLASS_GROUPED_MM_GROUP_LIMIT:
        return False
    K, N = self[0].size(1), mat2[0].size(1)
    return all(a.size(1) == K for a in self) and all(
        b.size(0) == K and b.size(1) == N for b in mat2
    )


def _foreach_mm_route(self: list[torch.Tensor], mat2: list[torch.Tensor]) -> _Route:
    """Choose a backend route; assumes _foreach_mm_cond already validated inputs."""
    if _check_nvmath_cublaslt() and _can_use_nvmath_cublaslt_grouped_mm(self, mat2):
        return "nvmath_cublaslt_grouped_mm"
    if _can_use_cutlass_grouped_mm(self, mat2):
        return "cutlass_grouped_mm"
    return "mm_loop"


def _warn_nvmath_unavailable_once() -> None:
    global _nvmath_warned
    if _nvmath_warned:
        return
    _nvmath_warned = True
    reason = (
        _unavailable_reason(_NVMATH_DEPS)
        or "cuBLASLt lacks cublasLtGroupedMatrixLayoutCreate"
    )
    warnings.warn(
        f"_foreach_mm: nvmath cublasLt grouped GEMM unavailable ({reason}), "
        f"using slower fallback.",
        stacklevel=3,
    )


def _foreach_mm_impl(
    self: list[torch.Tensor],
    mat2: list[torch.Tensor],
) -> list[torch.Tensor]:
    route = _foreach_mm_route(self, mat2)
    if route == "nvmath_cublaslt_grouped_mm":
        return _foreach_mm_impl_nvmath(self, mat2)
    # warn once when nvmath-eligible inputs fall back only because nvmath is absent.
    if (
        not _nvmath_warned
        and _can_use_nvmath_cublaslt_grouped_mm(self, mat2)
        and not _check_nvmath_cublaslt()
    ):
        _warn_nvmath_unavailable_once()
    if route == "cutlass_grouped_mm":
        return _foreach_mm_impl_stack(self, mat2)
    return [torch.mm(a, b) for a, b in zip(self, mat2)]  # mm_loop


def register_to_dispatch() -> None:
    registry.register_op_override(
        "native",
        "aten",
        "_foreach_mm",
        "CUDA",
        cond=_foreach_mm_cond,
        impl=_foreach_mm_impl,
    )
<|endoftext|>"""CuTeDSL override registrations for ``aten::sum`` (inner-tree reduction).

CuTeDSL port of the Triton-style inner-tree reduction sum kernel that
otherwise lives in ``aten/src/ATen/native/cuda/ReduceSumProdKernel.cu``
(``try_inner_tree_reduction`` + the inner-tree kernels). This override
runs only when ``PYTORCH_SUM_INNER_TREE`` is set to a truthy value, which
is the feature-rollout gate for this operator -- it is *not* gated by the
global ``TORCH_DISABLE_NATIVE_JIT`` kill switch alone (that is handled by
``cutedsl_utils`` at registration time).

We register two ATen dispatcher entries on ``CUDA``:

* ``sum.dim_IntList`` -- functional/method ``x.sum(dim=...)``.
* ``sum.IntList_out`` -- the structured ``.out`` target.

``sum.dim_IntList`` is ``structured_delegate: sum.IntList_out``, so its
delegation to the ``.out`` kernel happens *below* the dispatcher; overriding
``.out`` alone would not intercept ``x.sum(dim=1)``. Both are separate
dispatcher entries and are registered explicitly (the same reason
``scatter_add_`` is registered separately from ``scatter_add.out``).

Eligibility mirrors ``try_inner_tree_reduction`` (ReduceSumProdKernel.cu):
build the reduction ``TensorIterator`` over ``(out, self)`` and accept only
the coalesced geometry the kernels know how to run -- a single contiguous
reduced (fastest) dimension whose non-reduced dims collapse to at most one
outer-strided dimension. Anything else (multi-dim reduction, non-contiguous
reduced dim, dtype-casting sum, integer/complex dtypes, non-collapsing outer
layout) falls through to aten.

Dispatch note: this CuTeDSL override is the only inner-tree path -- the ATen
CUDA backend has no inner-tree kernel of its own. When this predicate accepts,
the CuTeDSL kernel runs; when it rejects (or ``PYTORCH_SUM_INNER_TREE`` is
unset), the call falls through to the unchanged ATen reduction.
"""

from __future__ import annotations

import os

import torch
from torch._subclasses.fake_tensor import is_fake_tensor
from torch._tensor_iterator import reduce_op, TensorIterator

from ... import cutedsl_utils as cu


# Reduction input dtypes the kernel handles. fp32/fp64 are the bitwise-
# validated set (the inner-tree bitwise tests upcast fp16/bf16/fp8 data to
# fp32 before summing, and fp64 sums in fp64). fp16/bf16 native reductions
# (acc in fp32) are functionally supported but not part of the bitwise
# contract.
_SUPPORTED_DTYPES = frozenset(
    {torch.float16, torch.bfloat16, torch.float32, torch.float64}
)

# The kernel indexes rows/elements and sizes the launch grid with Int32. Decline
# reductions whose sizes would overflow that so we fall through to ATen rather
# than silently wrap (mirrors the magnitude guard from the CUDA review).
_INT32_MAX = 2**31 - 1


def _int32_indexing_safe(self: torch.Tensor, d: int) -> bool:
    """Decline giant reductions so the kernel's Int32 row/element indexing and
    launch grid never wrap. The two-kernel grid is ``m * num_batches``;
    ``num_batches`` is bounded above by ``ceil(n / (32 * vec_size))`` (the
    smallest possible per-batch width), so this bound covers every path."""
    n = self.shape[d]
    if n == 0:
        return True
    m = self.numel() // n
    vec_size = max(1, 16 // self.element_size())
    num_batches_ub = -(-n // (32 * vec_size))
    return n <= _INT32_MAX and m <= _INT32_MAX and m * num_batches_ub <= _INT32_MAX


def _inner_tree_enabled() -> bool:
    # The CUDA gate is `use_inner_tree_reduction()`; the CuTeDSL predicate
    # requires the explicit feature flag (an empty/"0" value is off).
    return os.getenv("PYTORCH_SUM_INNER_TREE", "") not in ("", "0")


def _normalize_dim(dim: int, ndim: int) -> int:
    return dim + ndim if dim < 0 else dim


def _single_dim(dim, ndim: int) -> int | None:
    """Return the one normalized reduced dim, or ``None`` if this is not a
    single-dim reduction (matches ``num_reduce_dims == 1``)."""
    if dim is None:
        return None
    if isinstance(dim, int):
        dims = [dim]
    else:
        dims = list(dim)
    if len(dims) != 1:
        return None
    return _normalize_dim(dims[0], ndim)


def _keepdim_out_shape(self: torch.Tensor, d: int) -> list[int]:
    shape = list(self.shape)
    shape[d] = 1
    return shape


def _out_keepdim_view(
    self: torch.Tensor, d: int, keepdim: bool, out: torch.Tensor
) -> torch.Tensor | None:
    """Return a keepdim-shaped ``(.., 1, ..)`` view of ``out`` so the kernel
    writes ``out``'s storage directly. ``unsqueeze`` is always a view (unlike
    ``reshape``, which may copy a non-contiguous ``out``). Returns ``None`` if
    ``out`` is not the expected reduced shape."""
    expected = _keepdim_out_shape(self, d)
    try:
        out_kd = out if keepdim else out.unsqueeze(d)
    except IndexError:
        return None
    return out_kd if list(out_kd.shape) == expected else None


def _eligibility(
    self: torch.Tensor, d: int, out: torch.Tensor
) -> TensorIterator | None:
    """Return the reduction ``TensorIterator`` if ``(self, d, out)`` fits the
    kernel's expected geometry, else ``None``. ``d`` is already normalized
    and ``out`` is a keepdim-shaped output (broadcast-aligned with ``self``).

    Mirrors ``try_inner_tree_reduction``: build the reduction iterator, then
    require a single coalesced reduced (fastest) dimension with element
    stride 1 on the input, and at most one non-reduced (outer) dimension so
    the operands canonicalize to a single ``(M, N)`` view.
    """
    try:
        it = reduce_op(out, self)
    except RuntimeError:
        return None

    if it.numel == 0:
        return None
    if it.ndim == 0 or it.ndim > 2:
        return None

    input_index = it.ntensors - 1  # operands are (out, self); input is last
    es_in = it.element_strides(input_index)
    # Input must be contiguous on the reduced (fastest, dim-0) axis.
    if es_in[0] != 1:
        return None
    # Reduction must be on the fastest dim: either the whole tensor is the
    # reduction (ndim == 1) or the reduced-dim stride is the smallest.
    if it.ndim == 2 and not (es_in[0] < es_in[1]):
        return None
    return it


def _geometry(self: torch.Tensor, out: torch.Tensor, it: TensorIterator):
    """Recover ``(M, N, in_row_stride, out_row_stride)`` (element units) from
    the coalesced reduction iterator. ``M`` output rows, each a contiguous
    ``N``-element reduction; ``in_row_stride`` / ``out_row_stride`` step
    between rows of the canonical ``(M, N)`` / ``(M,)`` views."""
    m = out.numel()
    n = self.numel() // m if m else 0
    if it.ndim == 2:
        in_row_stride = it.element_strides(it.ntensors - 1)[1]
        out_row_stride = it.element_strides(0)[1]
    else:
        in_row_stride = n
        out_row_stride = 1
    return m, n, in_row_stride, out_row_stride


def _base_cond_ok(self: torch.Tensor, dim, dtype) -> int | None:
    """Shared front gate. Returns the normalized reduced dim if the call is a
    candidate, else ``None``."""
    if not _inner_tree_enabled():
        return None
    if not self.is_cuda or is_fake_tensor(self):
        return None
    # dtype-casting sum (an explicit out/result dtype) is out of scope.
    if dtype is not None:
        return None
    if self.dtype not in _SUPPORTED_DTYPES:
        return None
    d = _single_dim(dim, self.ndim)
    if d is None or not (0 <= d < self.ndim):
        return None
    if not _int32_indexing_safe(self, d):
        return None
    return d


def _cond(self, dim, keepdim=False, *, dtype=None) -> bool:
    d = _base_cond_ok(self, dim, dtype)
    if d is None:
        return False
    out = torch.empty(_keepdim_out_shape(self, d), dtype=self.dtype, device=self.device)
    return _eligibility(self, d, out) is not None


def _out_cond(self, dim, keepdim=False, *, dtype=None, out) -> bool:
    d = _base_cond_ok(self, dim, dtype)
    if d is None:
        return False
    if out.dtype != self.dtype or not out.is_cuda or is_fake_tensor(out):
        return False
    # The kernel writes ``out`` directly via a keepdim-aligned view; ``out``
    # must already have the reduced shape. Reject mis-shaped ``out`` and let
    # aten produce the proper error.
    out_kd = _out_keepdim_view(self, d, keepdim, out)
    if out_kd is None:
        return False
    return _eligibility(self, d, out_kd) is not None


def _kernel():
    from .inner_tree_kernel import inner_tree_sum_into

    return inner_tree_sum_into


def _run(self: torch.Tensor, d: int, out_kd: torch.Tensor) -> None:
    """Run the kernel writing the reduction of ``self`` along ``d`` into the
    keepdim-shaped ``out_kd``. Caller has validated eligibility."""
    it = _eligibility(self, d, out_kd)
    if it is None:
        raise RuntimeError("sum cutedsl: cond approved but iter rebuild failed")
    m, n, in_rs, out_rs = _geometry(self, out_kd, it)
    if m == 0 or n == 0:
        return
    in_2d = self.as_strided((m, n), (in_rs, 1))
    out_1d = out_kd.as_strided((m,), (out_rs,))
    _kernel()(out_1d, in_2d)


def _impl(self, dim, keepdim=False, *, dtype=None):
    d = _normalize_dim(dim[0] if not isinstance(dim, int) else dim, self.ndim)
    out_kd = torch.empty(
        _keepdim_out_shape(self, d), dtype=self.dtype, device=self.device
    )
    _run(self, d, out_kd)
    return out_kd if keepdim else out_kd.squeeze(d)


def _out_impl(self, dim, keepdim=False, *, dtype=None, out):
    d = _normalize_dim(dim[0] if not isinstance(dim, int) else dim, self.ndim)
    out_kd = _out_keepdim_view(self, d, keepdim, out)
    assert out_kd is not None  # noqa: S101  # _out_cond guarantees non-None
    _run(self, d, out_kd)
    return out


def register_to_dispatch() -> None:
    cu.register_op_override("aten", "sum.dim_IntList", "CUDA", cond=_cond, impl=_impl)
    cu.register_op_override(
        "aten", "sum.IntList_out", "CUDA", cond=_out_cond, impl=_out_impl
    )
<|endoftext|>from typing import NamedTuple


_WARP_SIZE = 32
# Persistent/loop batch-size boundary (kInnerTreeThreshold).
_K_INNER_TREE_THRESHOLD = 8192
# Tiny-N multirow boundary (kMultiRowMaxLoads).
_K_MULTIROW_MAX_LOADS = 8
# Looped-vs-split boundary on num_batches (kTwoKernelThreshold).
_K_TWO_KERNEL_THRESHOLD = 3
_K_TARGET_WARPS_PER_BLOCK = 8  # kTargetWarpsPerBlock


def _previous_power_of_2(n: int) -> int:
    power = 1
    while power * 2 <= n:
        power <<= 1
    return power


def _next_power_of_2(n: int) -> int:
    power = 1
    while power < n:
        power <<= 1
    return power


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def vec_size(itemsize: int) -> int:
    """16-byte vectorized loads: fp32->4, fp64->2, fp16/bf16->8."""
    return max(1, 16 // max(1, itemsize))


class InnerTreeParams(NamedTuple):
    num_warps: int
    batch_total_elements: int  # Inductor R0_BLOCK (the per-batch tile).
    # Inductor split count: ceil(N / batch_total_elements).
    num_batches: int
    depth: int
    rows_per_block: int
    effective_loads: int


def compute_inner_tree_params(
    inputs_per_output: int, num_outputs: int, vec_size: int
) -> InnerTreeParams:
    """Plan the inner-tree reduction for one row of inputs.

    ``vec_size`` is ``vec_size(itemsize)``. The result is the tiling used by
    the eager kernel. Inductor consumes ``batch_total_elements`` (R0_BLOCK)
    and ``num_batches`` (split count).
    """
    wle = _WARP_SIZE * vec_size
    threshold = _K_INNER_TREE_THRESHOLD
    n = inputs_per_output

    if n > threshold:
        num_batches_est = _ceil_div(n, threshold)
        if num_batches_est <= _K_TWO_KERNEL_THRESHOLD:
            total_ideal_warps = 0
            for b in range(num_batches_est):
                batch_start = b * threshold
                batch_elements = min(threshold, n - batch_start)
                total_ideal_warps += min(16, max(1, batch_elements // wle))
            num_warps = max(1, total_ideal_warps // num_batches_est)
        else:
            num_warps = min(16, max(1, threshold // wle))
    else:
        num_warps = min(16, max(1, n // wle))
    num_warps = _previous_power_of_2(num_warps)

    loads_per_warp = _ceil_div(n, num_warps * wle)
    if loads_per_warp > 1:
        loads_per_warp = _next_power_of_2(loads_per_warp)

    max_loads_per_batch = max(1, threshold // (wle * num_warps))
    max_loads_per_batch = _previous_power_of_2(max_loads_per_batch)
    effective_loads = min(loads_per_warp, max_loads_per_batch)

    batch_total_elements = effective_loads * wle * num_warps
    num_batches = _ceil_div(n, batch_total_elements)

    if num_warps < _K_TARGET_WARPS_PER_BLOCK and num_batches <= _K_TWO_KERNEL_THRESHOLD:
        rows_per_block = min(num_outputs, _K_TARGET_WARPS_PER_BLOCK // num_warps)
    else:
        rows_per_block = 1

    depth = 0
    nn = effective_loads + 1
    while nn > 1:
        depth += 1
        nn >>= 1
    if depth < 1:
        depth = 1

    return InnerTreeParams(
        num_warps,
        batch_total_elements,
        num_batches,
        depth,
        rows_per_block,
        effective_loads,
    )
<|endoftext|>"""CuTeDSL inner-tree sum reduction kernels.

Bitwise port of the inner-tree reduction kernels in
``aten/src/ATen/native/cuda/ReduceSumProdKernel.cu`` (with helpers from
``ReduceInnerTree.cuh`` / ``WarpReduce.cuh``). Each output element is the
sum of ``N`` contiguous input elements; ``N`` is baked in at ``cute.compile``
time so the per-row vector tree, streaming tree, and launch planning all
unroll at trace time and the accumulation order matches the CUDA kernel
exactly.

Three execution shapes, selected on the host exactly as
``try_inner_tree_reduction`` does:

* Multirow (``inner_tree_reduction_multirow_kernel``): one thread per row,
  used when ``N <= vec_size * kMultiRowMaxLoads``.
* Looped (``inner_tree_reduction_looped_kernel``): warps cooperate per row,
  batches accumulated inside the kernel; used when ``num_batches <= 3``.
* Two-kernel (``inner_tree_reduction_kernel`` +
  ``inner_tree_accumulate_multirow_kernel``): a partials kernel followed by a
  linear per-row accumulate; used otherwise.

Bitwise-critical ordering reproduced here:

* ``vec_size = 16 / itemsize`` grouping; the in-vec tree order
  (``inner_tree_reduce``) for vec_size >= 4, else linear. Zero-padded tails
  are bitwise identical between tree and linear (``x + 0 == x``), so a single
  uniform choice covers both the full and tail cases.
* Warp butterfly with ASCENDING shuffle-XOR offsets ``1,2,4,...`` (matches
  ``WarpReduceDirection::ASCENDING``); we hand-roll it because
  ``cute.arch.warp_reduction`` uses descending offsets, which is a different
  (and thus bitwise-different) order.
* Streaming inner-tree carry across loads (``streaming_inner_tree_step``,
  merge count = trailing-zero bits of ``load + 1``).
* Accumulation dtype: fp32 for fp16/bf16/fp32, fp64 for fp64 -- matches
  ``reduce_dispatch`` so rounding is identical. The inner-tree bitwise tests
  only exercise fp32 and fp64 (they upcast narrower data first); fp16/bf16
  native reductions are functionally supported but not bitwise-validated.

Because ``N`` is compile-time, a warp/batch that owns fewer than the unrolled
number of loads contributes zeros for the extra loads; zeros are additive
identities, so this reproduces the ragged CUDA loop bit-for-bit.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]  # noqa: TC002

import cutlass
import cutlass.cute as cute

# fp64 is part of the bitwise contract; import Float64 directly so an
# unsupported runtime surfaces a clear error rather than silently dropping it.
from cutlass import BFloat16, const_expr, Float16, Float32, Float64, Int32

import torch
from torch._native.instrumentation import instrumented_cutedsl_cache

from .inner_tree_plan import (
    _ceil_div,
    _K_MULTIROW_MAX_LOADS,
    _K_TWO_KERNEL_THRESHOLD,
    _next_power_of_2,
    _WARP_SIZE,
    compute_inner_tree_params,
    InnerTreeParams,
    vec_size as _vec_size,
)


_MULTIROW_THREADS = 128
_MULTIROW_MAX_DEPTH = 6  # kMaxDepth in the multirow kernel
_ACCUM_THREADS = 128  # kAccumulateThreads

_TORCH_TO_CUTE = {
    torch.float16: Float16,
    torch.bfloat16: BFloat16,
    torch.float32: Float32,
    torch.float64: Float64,
}


def _acc_for(in_dt):
    return Float64 if in_dt is Float64 else Float32


# Torch accumulator dtype for the two-kernel partials buffer (fp32 for
# fp16/bf16/fp32, fp64 for fp64) -- keeps cross-batch accumulation in the
# accumulator dtype, matching ATen.
_ACC_TORCH_DTYPE = {
    torch.float16: torch.float32,
    torch.bfloat16: torch.float32,
    torch.float32: torch.float32,
    torch.float64: torch.float64,
}


# ---------------------------------------------------------------------------
# Trace-time reduction helpers. These run during ``cute.compile`` tracing and
# emit the exact add DAG; ``vals`` / ``tree`` are Python lists of cute
# ``Numeric`` register values.
# ---------------------------------------------------------------------------


def _linear_reduce(vals):
    acc = vals[0]
    for i in range(1, len(vals)):
        acc = acc + vals[i]
    return acc


def _inner_tree_reduce(vals):
    v = list(vals)
    n = len(v)
    stride = 1
    while stride < n:
        i = 0
        while i + stride < n:
            v[i] = v[i] + v[i + stride]
            i += stride * 2
        stride *= 2
    return v[0]


def _reduce_vec(vals, vec_size: int):
    return _inner_tree_reduce(vals) if vec_size >= 4 else _linear_reduce(vals)


def _streaming_push(tree, val, load: int, max_depth: int) -> None:
    """Port of ``streaming_inner_tree_step``: merge count is the number of
    trailing zero bits of ``load + 1`` (``__ffs(load + 1) - 1``), capped at
    ``max_depth``. ``carry = tree_accs[--top] + carry`` keeps the existing
    accumulator on the left."""
    trailing_zeros = ((load + 1) & -(load + 1)).bit_length() - 1
    carry = val
    for _ in range(min(trailing_zeros, max_depth)):
        carry = tree.pop() + carry
    tree.append(carry)


def _warp_butterfly(val):
    """Ascending-offset shuffle-XOR butterfly add reduce (matches the CUDA
    ``warp_reduce`` ASCENDING direction)."""
    offset = 1
    while offset < _WARP_SIZE:
        val = val + cute.arch.shuffle_sync_bfly(
            val, offset=offset, mask_and_clamp=_WARP_SIZE - 1
        )
        offset <<= 1
    return val


def _load_vec_static(mIn, row, base: int, N: int, vec_size: int, acc, zero):
    """Load + reduce one vector group with compile-time ``base`` (multirow)."""
    vals = []
    for i in range(vec_size):
        off = base + i
        vals.append(acc(mIn[row, off]) if off < N else zero)
    return _reduce_vec(vals, vec_size)


@cute.jit
def _load_vec_dyn(
    mIn,
    row,
    base,
    N: cutlass.Constexpr,
    vec_size: cutlass.Constexpr,
    acc: cutlass.Constexpr,
    zero,
):
    """Load + reduce one vector group with a runtime ``base`` (warp paths).

    ``@cute.jit`` so the per-element bounds check is lowered by the DSL
    instead of evaluated as a Python bool at trace time (the kernel body's
    AST transform does not reach into plain helpers). Each element is guarded
    against the row width so out-of-row reads never happen and tail elements
    contribute zero.
    """
    vals = []
    for i in cutlass.range_constexpr(vec_size):
        off = base + Int32(i)
        v = zero
        if off < Int32(N):
            v = acc(mIn[row, off])
        vals.append(v)
    return _reduce_vec(vals, vec_size)


# ---------------------------------------------------------------------------
# Kernel builders. Each returns a ``@cute.jit`` launcher specialized on the
# compile-time parameters.
# ---------------------------------------------------------------------------


def _make_multirow(in_dt, acc, out_dt, N: int, vec_size: int):
    num_loads = _next_power_of_2(_ceil_div(N, vec_size))

    @cute.kernel
    def _kernel(mIn: cute.Tensor, mOut: cute.Tensor, num_outputs: Int32):
        tx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        row = bx * Int32(_MULTIROW_THREADS) + tx
        if row < num_outputs:
            zero = acc(0.0)
            tree = []
            for load in cutlass.range_constexpr(num_loads):
                val = _load_vec_static(
                    mIn, row, load * vec_size, N, vec_size, acc, zero
                )
                _streaming_push(tree, val, load, _MULTIROW_MAX_DEPTH)
            mOut[row] = out_dt(tree[0])

    @cute.jit
    def _launch(
        mIn: cute.Tensor,
        mOut: cute.Tensor,
        stream: cuda.CUstream,
        num_outputs: Int32,
        grid: Int32,
    ):
        _kernel(mIn, mOut, num_outputs).launch(
            grid=[grid, 1, 1], block=[_MULTIROW_THREADS, 1, 1], stream=stream
        )

    return _launch


def _make_looped(in_dt, acc, out_dt, N: int, vec_size: int, p: InnerTreeParams):
    wpr = p.num_warps  # warps_per_reduction == num_warps in launch_looped
    rows_per_block = p.rows_per_block
    total_warps = wpr * rows_per_block
    wle = _WARP_SIZE * vec_size

    # Per-batch geometry is fully compile-time (num_batches <= 3, fixed N).
    batches = []
    for b in range(p.num_batches):
        batch_offset = b * p.batch_total_elements
        remaining = min(p.batch_total_elements, N - batch_offset)
        lpw = _ceil_div(remaining, wpr * wle)
        if lpw > 1:
            lpw = _next_power_of_2(lpw)
        batches.append((batch_offset, remaining, lpw, lpw * wle))

    @cute.kernel
    def _kernel(mIn: cute.Tensor, mOut: cute.Tensor, num_outputs: Int32):
        tx, ty, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        lane = tx
        global_warp_id = ty
        row_in_block = global_warp_id // Int32(wpr)
        warp_id = global_warp_id % Int32(wpr)
        row = bx * Int32(rows_per_block) + row_in_block
        row_active = row < num_outputs

        warp_writes = None  # bound for type checker; used only when wpr > 1
        if const_expr(wpr > 1):
            smem = cutlass.utils.SmemAllocator()
            warp_writes = smem.allocate_tensor(
                acc, cute.make_layout(total_warps), byte_alignment=8
            )

        zero = acc(0.0)
        final_sum = zero
        for b in cutlass.range_constexpr(p.num_batches):
            batch_offset, remaining_c, lpw, warp_chunk = batches[b]
            rem = Int32(0)
            if row_active:
                rem = Int32(remaining_c)
            warp_off = warp_id * Int32(warp_chunk)
            if warp_off > rem:
                warp_off = rem
            warp_start = Int32(batch_offset) + warp_off
            this_warp_elements = Int32(warp_chunk)
            tail = rem - warp_off
            if this_warp_elements > tail:
                this_warp_elements = tail
            this_batch_loads = (this_warp_elements + Int32(wle - 1)) // Int32(wle)

            tree = []
            for load in cutlass.range_constexpr(lpw):
                base = warp_start + Int32(load * wle) + lane * Int32(vec_size)
                v = zero
                if Int32(load) < this_batch_loads:
                    v = _warp_butterfly(
                        _load_vec_dyn(mIn, row, base, N, vec_size, acc, zero)
                    )
                _streaming_push(tree, v, load, p.depth)
            warp_acc = tree[0]

            if const_expr(wpr > 1):
                assert warp_writes is not None  # noqa: S101
                if lane == Int32(0):
                    warp_writes[row_in_block * Int32(wpr) + warp_id] = warp_acc
                cute.arch.barrier()
                merged = zero
                if lane < Int32(wpr):
                    merged = warp_writes[row_in_block * Int32(wpr) + lane]
                warp_acc = _warp_butterfly(merged)
                if const_expr(b + 1 < p.num_batches):
                    cute.arch.barrier()

            final_sum = final_sum + warp_acc

        if row_active:
            if lane == Int32(0):
                if warp_id == Int32(0):
                    mOut[row] = out_dt(final_sum)

    @cute.jit
    def _launch(
        mIn: cute.Tensor,
        mOut: cute.Tensor,
        stream: cuda.CUstream,
        num_outputs: Int32,
        grid: Int32,
    ):
        _kernel(mIn, mOut, num_outputs).launch(
            grid=[grid, 1, 1], block=[_WARP_SIZE, total_warps, 1], stream=stream
        )

    return _launch


def _make_two_partial(in_dt, acc, out_dt, N: int, vec_size: int, p: InnerTreeParams):
    wpr = p.num_warps
    num_batches = p.num_batches
    wle = _WARP_SIZE * vec_size
    eff = p.effective_loads

    # Only two distinct per-batch widths: full batches and the (shorter) last
    # batch. Precompute both; the kernel selects on ``batch_idx`` at runtime.
    full_remaining = p.batch_total_elements
    last_remaining = N - (num_batches - 1) * p.batch_total_elements
    warp_chunk_full = eff * wle

    def _lpw(remaining: int) -> int:
        lpw = _ceil_div(remaining, wpr * wle)
        return _next_power_of_2(lpw) if lpw > 1 else lpw

    warp_chunk_last = _lpw(last_remaining) * wle

    @cute.kernel
    def _kernel(mIn: cute.Tensor, mPartials: cute.Tensor):
        # Grid is exactly num_outputs * num_batches, so no row bounds check.
        tx, ty, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        lane = tx
        warp_id = ty
        row = bx // Int32(num_batches)
        batch_idx = bx % Int32(num_batches)

        remaining = Int32(full_remaining)
        warp_chunk = Int32(warp_chunk_full)
        if batch_idx == Int32(num_batches - 1):
            remaining = Int32(last_remaining)
            warp_chunk = Int32(warp_chunk_last)
        batch_offset = batch_idx * Int32(p.batch_total_elements)

        warp_writes = None  # bound for type checker; used only when wpr > 1
        if const_expr(wpr > 1):
            smem = cutlass.utils.SmemAllocator()
            warp_writes = smem.allocate_tensor(
                acc, cute.make_layout(wpr), byte_alignment=8
            )

        warp_off = warp_id * warp_chunk
        if warp_off > remaining:
            warp_off = remaining
        warp_start = batch_offset + warp_off
        this_warp_elements = warp_chunk
        tail = remaining - warp_off
        if this_warp_elements > tail:
            this_warp_elements = tail
        this_batch_loads = (this_warp_elements + Int32(wle - 1)) // Int32(wle)

        zero = acc(0.0)
        tree = []
        for load in cutlass.range_constexpr(eff):
            base = warp_start + Int32(load * wle) + lane * Int32(vec_size)
            v = zero
            if Int32(load) < this_batch_loads:
                v = _warp_butterfly(
                    _load_vec_dyn(mIn, row, base, N, vec_size, acc, zero)
                )
            _streaming_push(tree, v, load, p.depth)
        warp_acc = tree[0]

        if const_expr(wpr > 1):
            assert warp_writes is not None  # noqa: S101
            if lane == Int32(0):
                warp_writes[warp_id] = warp_acc
            cute.arch.barrier()
            merged = zero
            if lane < Int32(wpr):
                merged = warp_writes[lane]
            warp_acc = _warp_butterfly(merged)

        if lane == Int32(0):
            if warp_id == Int32(0):
                # Store partials in the accumulator dtype so the cross-batch
                # sum stays in fp32 for fp16/bf16 (matches ATen's
                # accumulate-in-fp32-round-once); downcast happens only in the
                # accumulate kernel's final write.
                mPartials[row * Int32(num_batches) + batch_idx] = acc(warp_acc)

    @cute.jit
    def _launch(
        mIn: cute.Tensor,
        mPartials: cute.Tensor,
        stream: cuda.CUstream,
        grid: Int32,
    ):
        _kernel(mIn, mPartials).launch(
            grid=[grid, 1, 1], block=[_WARP_SIZE, wpr, 1], stream=stream
        )

    return _launch


def _make_two_accum(acc, out_dt, num_batches: int):
    @cute.kernel
    def _kernel(mPartials: cute.Tensor, mOut: cute.Tensor, num_outputs: Int32):
        tx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        row = bx * Int32(_ACCUM_THREADS) + tx
        if row < num_outputs:
            base = row * Int32(num_batches)
            # Partials are in the accumulator dtype; accumulate in that dtype
            # and downcast to the output dtype only on the final store.
            s = mPartials[base]
            # Runtime loop, NOT range_constexpr: num_batches grows with N
            # (thousands for very large reductions), so unrolling would make
            # cute.compile blow up. Linear accumulation order matches the CUDA
            # kernel regardless of unroll.
            for b in cutlass.range(1, num_batches):
                s = s + mPartials[base + b]
            mOut[row] = out_dt(s)

    @cute.jit
    def _launch(
        mPartials: cute.Tensor,
        mOut: cute.Tensor,
        stream: cuda.CUstream,
        num_outputs: Int32,
        grid: Int32,
    ):
        _kernel(mPartials, mOut, num_outputs).launch(
            grid=[grid, 1, 1], block=[_ACCUM_THREADS, 1, 1], stream=stream
        )

    return _launch


# ---------------------------------------------------------------------------
# Compile cache + host dispatch.
# ---------------------------------------------------------------------------


def _fake_2d(dt, N: int):
    return cute.runtime.make_fake_tensor(
        dt, (cute.sym_int(), N), stride=(cute.sym_int64(), 1)
    )


def _fake_1d(dt):
    return cute.runtime.make_fake_tensor(
        dt, (cute.sym_int(),), stride=(cute.sym_int64(),)
    )


def _fake_1d_contig(dt):
    return cute.runtime.make_fake_tensor(dt, (cute.sym_int(),), stride=(1,))


@instrumented_cutedsl_cache(
    "aten::sum",
    key_fn=lambda torch_dtype, N: f"multirow {torch_dtype} N={N}",
)
def _compile_multirow(torch_dtype: torch.dtype, N: int):
    in_dt = _TORCH_TO_CUTE[torch_dtype]
    acc = _acc_for(in_dt)
    launcher = _make_multirow(in_dt, acc, in_dt, N, _vec_size(torch_dtype.itemsize))
    return cute.compile(
        launcher,
        _fake_2d(in_dt, N),
        _fake_1d(in_dt),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        Int32(0),
        Int32(0),
        options="--enable-tvm-ffi",
    )


@instrumented_cutedsl_cache(
    "aten::sum",
    key_fn=lambda torch_dtype, N, num_warps, _bte, num_batches, *_rest: (
        f"looped {torch_dtype} N={N} warps={num_warps} batches={num_batches}"
    ),
)
def _compile_looped(
    torch_dtype: torch.dtype,
    N: int,
    num_warps: int,
    batch_total_elements: int,
    num_batches: int,
    depth: int,
    rows_per_block: int,
    effective_loads: int,
):
    in_dt = _TORCH_TO_CUTE[torch_dtype]
    acc = _acc_for(in_dt)
    p = InnerTreeParams(
        num_warps,
        batch_total_elements,
        num_batches,
        depth,
        rows_per_block,
        effective_loads,
    )
    launcher = _make_looped(in_dt, acc, in_dt, N, _vec_size(torch_dtype.itemsize), p)
    return cute.compile(
        launcher,
        _fake_2d(in_dt, N),
        _fake_1d(in_dt),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        Int32(0),
        Int32(0),
        options="--enable-tvm-ffi",
    )


@instrumented_cutedsl_cache(
    "aten::sum",
    key_fn=lambda torch_dtype, N, num_warps, _bte, num_batches, *_rest: (
        f"two_partial {torch_dtype} N={N} warps={num_warps} batches={num_batches}"
    ),
)
def _compile_two_partial(
    torch_dtype: torch.dtype,
    N: int,
    num_warps: int,
    batch_total_elements: int,
    num_batches: int,
    depth: int,
    effective_loads: int,
):
    in_dt = _TORCH_TO_CUTE[torch_dtype]
    acc = _acc_for(in_dt)
    p = InnerTreeParams(
        num_warps,
        batch_total_elements,
        num_batches,
        depth,
        1,
        effective_loads,
    )
    launcher = _make_two_partial(in_dt, acc, acc, N, _vec_size(torch_dtype.itemsize), p)
    return cute.compile(
        launcher,
        _fake_2d(in_dt, N),
        _fake_1d_contig(acc),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        Int32(0),
        options="--enable-tvm-ffi",
    )


@instrumented_cutedsl_cache(
    "aten::sum",
    key_fn=lambda torch_dtype, num_batches: (
        f"two_accum {torch_dtype} batches={num_batches}"
    ),
)
def _compile_two_accum(torch_dtype: torch.dtype, num_batches: int):
    out_dt = _TORCH_TO_CUTE[torch_dtype]
    acc = _acc_for(out_dt)
    launcher = _make_two_accum(acc, out_dt, num_batches)
    return cute.compile(
        launcher,
        _fake_1d_contig(acc),
        _fake_1d(out_dt),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        Int32(0),
        Int32(0),
        options="--enable-tvm-ffi",
    )


def inner_tree_sum_into(out: torch.Tensor, src: torch.Tensor) -> None:
    """Compute ``out[r] = sum(src[r, :])`` for every row ``r``.

    ``src`` is a 2D ``(M, N)`` view with inner-dim stride 1 (outer row stride
    may differ from N); ``out`` is a 1D ``(M,)`` view (its stride may differ
    from 1). Both carry their strides into the kernel via the cute tensor, so
    strided outer inputs and strided outputs are handled directly.
    """
    m, n = src.shape
    dtype = src.dtype
    vec_size = _vec_size(dtype.itemsize)

    if n <= vec_size * _K_MULTIROW_MAX_LOADS:
        compiled = _compile_multirow(dtype, n)
        grid = _ceil_div(m, _MULTIROW_THREADS)
        compiled(src, out, m, grid)
        return

    p = compute_inner_tree_params(n, m, vec_size)
    if p.num_batches <= _K_TWO_KERNEL_THRESHOLD:
        compiled = _compile_looped(
            dtype,
            n,
            p.num_warps,
            p.batch_total_elements,
            p.num_batches,
            p.depth,
            p.rows_per_block,
            p.effective_loads,
        )
        grid = _ceil_div(m, p.rows_per_block)
        compiled(src, out, m, grid)
        return

    partials = torch.empty(
        m * p.num_batches, dtype=_ACC_TORCH_DTYPE[dtype], device=src.device
    )
    c1 = _compile_two_partial(
        dtype,
        n,
        p.num_warps,
        p.batch_total_elements,
        p.num_batches,
        p.depth,
        p.effective_loads,
    )
    c1(src, partials, m * p.num_batches)
    c2 = _compile_two_accum(dtype, p.num_batches)
    c2(partials, out, m, _ceil_div(m, _ACCUM_THREADS))
<|endoftext|>"""cuSOLVER Xpolar (QDWH) polar decomposition via nvmath bindings."""

import ctypes

import torch

from ... import nvmath_utils


# cuBLAS fill mode passed to Xpolar. The general polar decomposition reads the
# entire input A (it is not assumed symmetric), so FULL is required -- LOWER is
# rejected and UPPER silently reads only a triangle, yielding wrong factors.
_CUBLAS_FILL_MODE_FULL = 2


def _cuda_dtype(dtype: torch.dtype):
    # cuSOLVER Xpolar supports only real single/double precision. Complex inputs
    # are routed to the SVD path by the caller and never reach here.
    from cuda.bindings.runtime import cudaDataType  # pyrefly: ignore[missing-import]

    return {
        torch.float32: cudaDataType.CUDA_R_32F,
        torch.float64: cudaDataType.CUDA_R_64F,
    }[dtype]


def _empty_colmajor(m: int, n: int, A: torch.Tensor) -> torch.Tensor:
    # Column-major (m, n) buffer for cuSOLVER (leading dimension m). The view
    # owns its storage and presents logical shape (m, n) so values read back
    # directly without a transpose.
    return A.new_empty((n, m)).mT


def _polar_2d(
    A: torch.Tensor, *, return_residual: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    from nvmath.bindings import cusolverDn as cs  # pyrefly: ignore[missing-import]

    m, n = A.shape
    dtype = A.dtype
    device = A.device

    # cuSOLVER handles are device-bound; run everything (handle lookup, workspace
    # allocation, and the xpolar call) with A's device current so a non-default
    # device (e.g. cuda:1) doesn't trigger an illegal memory access.
    with torch.cuda.device(device):
        handle = nvmath_utils.get_cusolver_handle(device)
        params = cs.create_params()
        return _polar_2d_impl(A, m, n, dtype, device, handle, params, return_residual)


def _polar_2d_impl(A, m, n, dtype, device, handle, params, return_residual):
    from nvmath.bindings import cusolverDn as cs  # pyrefly: ignore[missing-import]

    try:
        cs.set_stream(handle, torch.cuda.current_stream(device).cuda_stream)

        # cuSOLVER is column-major and overwrites the input (m x n, ld=m) with U.
        # Use column-major buffers so the input is untouched and both factors
        # read back with the correct (non-transposed/non-conjugated) values.
        U_col = _empty_colmajor(m, n, A)
        U_col.copy_(A)
        H_col = _empty_colmajor(n, n, A)

        # Real single/double: input, output, and compute share one data type.
        cdt = _cuda_dtype(dtype)
        dtype_a = dtype_h = compute_type = cdt

        ws_dev, ws_host = cs.xpolar_buffer_size(
            handle,
            params,
            _CUBLAS_FILL_MODE_FULL,
            m,
            n,
            dtype_a,
            U_col.data_ptr(),
            m,
            dtype_h,
            H_col.data_ptr(),
            n,
            compute_type,
        )

        # Allocate exactly the queried sizes. When a size is 0, cuSOLVER does
        # not dereference the corresponding pointer (an empty CUDA tensor's
        # data_ptr is null, which is the expected "no workspace" convention).
        buf_dev = torch.empty(ws_dev, dtype=torch.uint8, device=device)
        buf_host = bytearray(ws_host)
        host_ptr = (ctypes.c_char * ws_host).from_buffer(buf_host)

        # Xpolar reports convergence diagnostics into device scalars: three
        # double* (res_nrm, A_nrmF, rcond) and one int* (info). Pack all four
        # into a single byte buffer (8-byte aligned slots) so a single kernel
        # zero-initializes them.
        #
        # IMPORTANT: these MUST be zero-initialized. cuSOLVER 12.2.0.1's Xpolar
        # reads res_nrm/A_nrmF/rcond as accumulators during the QDWH iteration
        # (not write-only outputs); passing uninitialized memory (e.g. a block
        # the caching allocator just recycled from prior GPU work) makes the
        # iteration diverge and silently return a wrong factorization with
        # info=0. Zeroing them makes the result deterministic and correct.
        diag = torch.zeros(28, dtype=torch.uint8, device=device)
        diag_ptr = diag.data_ptr()
        d_res_nrm = diag_ptr  # double
        d_a_nrm_f = diag_ptr + 8  # double, == ||A||_F
        d_rcond = diag_ptr + 16  # double
        d_info = diag_ptr + 24  # int32

        cs.xpolar(
            handle,
            params,
            _CUBLAS_FILL_MODE_FULL,
            m,
            n,
            dtype_a,
            U_col.data_ptr(),
            m,
            dtype_h,
            H_col.data_ptr(),
            n,
            compute_type,
            buf_dev.data_ptr(),
            ws_dev,
            ctypes.addressof(host_ptr),
            ws_host,
            d_res_nrm,
            d_a_nrm_f,
            d_rcond,
            d_info,
        )

        # H is symmetric/Hermitian up to round-off; symmetrize to make it exact.
        H = 0.5 * (H_col + H_col.mH)

        # Optionally expose the relative residual ||A - U*H||_F / ||A||_F that
        # Xpolar reports, as a device scalar. We never sync on it ourselves;
        # callers who want it can copy it to host (taking the sync hit) for
        # debugging or to validate convergence.
        residual = None
        if return_residual:
            diag_f64 = diag[:24].view(torch.float64)
            a_nrm = diag_f64[1]
            residual = diag_f64[0] / a_nrm.clamp_min(torch.finfo(torch.float64).tiny)
        return U_col, H, residual
    finally:
        cs.destroy_params(params)


def polar_xpolar(
    A: torch.Tensor, *, return_residual: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Polar decomposition A = U @ H of a single 2-D matrix via cuSOLVER Xpolar.

    Fully asynchronous: issues work on the current CUDA stream and never
    synchronizes. When ``return_residual`` is True, also returns the relative
    residual ``||A - U*H||_F / ||A||_F`` as a device scalar for callers that
    want to validate convergence.

    cuSOLVER Xpolar has no batched API, so only 2-D inputs are supported; the
    dispatch override declines batched inputs (they take the batched SVD
    kernel instead of a slower per-matrix loop).
    """
    torch._check(A.dim() == 2, lambda: "polar_xpolar expects a 2-D matrix")
    return _polar_2d(A, return_residual=return_residual)
<|endoftext|>"""
Python override for aten::linalg_polar (polar decomposition A = U @ H).

This override is a CUDA-only accelerator: when nvmath-python is available and
the cuSOLVER runtime exposes Xpolar, it computes the decomposition with the
QR-based Dynamically Weighted Halley (QDWH) algorithm. For every other case --
CPU, ROCm, complex inputs, missing/old nvmath, or unsupported shapes -- the
cond returns False and the router falls through to the structured aten kernel,
which computes the same decomposition via the SVD. That fallthrough is a
correct, fully-functional kernel (it also backs compile/export), so this
override never needs to provide its own CPU/SVD path.

Both the functional (linalg_polar) and out= (linalg_polar.out) overloads are
overridden so the two call forms use the same algorithm. Only single 2-D
matrices are accelerated: cuSOLVER Xpolar has no batched API, so a batched
input would require a Python loop that is slower than the batched SVD kernel;
batched inputs therefore fall through to aten.
"""

import torch

from ... import nvmath_utils, registry


# Dtypes for which the cuSOLVER Xpolar (QDWH) fast path is attempted. Xpolar
# supports only real single/double; complex falls through to the aten SVD
# kernel. This is the single source of truth for the override's dtype gating.
_XPOLAR_DTYPES = (torch.float32, torch.float64)

# Minimum cuSOLVER runtime version exposing cusolverDnXpolar (CUDA 13.2 ships
# cuSOLVER 12.2). The cuSOLVER nvmath loads is whatever the loader resolves
# (system CUDA, or the nvidia-cu* wheels a torch build pulls in), which is not
# guaranteed to be >= 12.2 just because nvmath is installed -- so we gate on the
# reported version and fall through to the SVD kernel when it is too old.
_MIN_CUSOLVER_VERSION = 12200

# cuSOLVER Xpolar availability. Checked lazily on first use; flipped to False
# permanently if a call ever hits a hard failure (e.g. runtime lacks the symbol).
_nvmath_available: "bool | None" = None


def _check_nvmath() -> bool:
    global _nvmath_available
    if _nvmath_available is None:
        # The cuSOLVER Xpolar path is NVIDIA-only; never attempt it on ROCm
        # (torch.version.hip is set for HIP builds, where tensors still report
        # device.type == "cuda"). ROCm falls through to the aten SVD kernel.
        version = (
            None if torch.version.hip is not None else nvmath_utils.cusolver_version()
        )
        _nvmath_available = (
            torch.version.hip is None
            and nvmath_utils.nvmath_unavailable_reason() is None
            and version is not None
            and version >= _MIN_CUSOLVER_VERSION
        )
    return _nvmath_available


def _out_matches(A: torch.Tensor, U: torch.Tensor, H: torch.Tensor) -> bool:
    # The override writes results into U/H with copy_, which neither enforces the
    # output dtype nor resizes like the structured out= machinery does. Only fire
    # when the provided outputs already satisfy that contract (correct dtype,
    # device, and the meta-declared contiguous shapes); otherwise decline so the
    # call falls through to the aten .out kernel, which validates and resizes.
    m, n = A.shape
    return (
        U.dtype == A.dtype
        and H.dtype == A.dtype
        and U.device == A.device
        and H.device == A.device
        and tuple(U.shape) == (m, n)
        and tuple(H.shape) == (n, n)
        and U.is_contiguous()
        and H.is_contiguous()
    )


def _polar_cond_cuda(A: torch.Tensor, *args, **kwargs) -> bool:
    # Only fire when we can actually accelerate: a single non-empty real 2-D
    # matrix with m >= n, and a usable cuSOLVER Xpolar runtime. Batched inputs
    # (dim > 2) are declined -- Xpolar has no batched API, so looping in Python
    # is slower than the batched SVD kernel. Everything else (CPU, complex,
    # batched, unsupported runtime) falls through to the structured aten kernel.
    if A.dtype not in _XPOLAR_DTYPES:
        return False
    if A.dim() != 2 or A.size(0) < A.size(1) or A.size(1) == 0:
        return False
    if not _check_nvmath():
        return False
    # For the out= overload, only accelerate when the outputs already match the
    # structured contract; otherwise fall through to aten for validation/resize.
    U, H = kwargs.get("U"), kwargs.get("H")
    if U is not None or H is not None:
        if U is None or H is None or not _out_matches(A, U, H):
            return False
    return True


def _run_xpolar(A: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor] | None":
    # Returns (U, H) from cuSOLVER Xpolar, or None on a hard failure (after
    # permanently disabling the fast path).
    global _nvmath_available
    from .nvmath_impl import polar_xpolar

    try:
        U, H, _ = polar_xpolar(A)
        return U, H
    except Exception:
        # Hard failure (missing symbol, bad runtime). Disable the fast path so
        # subsequent calls (and the fall-throughs below) go straight to aten.
        _nvmath_available = False
        return None


def _polar_impl_cuda(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    result = _run_xpolar(A)
    if result is None:
        return torch.ops.aten.linalg_polar.default(A)
    # Match the meta function's contiguous output layout.
    U, H = result
    return U.contiguous(), H.contiguous()


def _polar_out_impl_cuda(
    A: torch.Tensor, *, U: torch.Tensor, H: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    result = _run_xpolar(A)
    if result is None:
        return torch.ops.aten.linalg_polar.out(A, U=U, H=H)
    U_qdwh, H_qdwh = result
    U.copy_(U_qdwh)
    H.copy_(H_qdwh)
    return U, H


def register_to_dispatch() -> None:
    # This is a CUDA-only accelerator. Registering a CUDA override captures the
    # existing aten CUDA kernel as the fallback, which only exists in a CUDA
    # build -- on CPU/XPU/ROCm-only builds there is no aten::linalg_polar.out
    # CUDA kernel and registration would raise at import time. Gate on the
    # fork-safe build-time check (NOT is_available(), which would init CUDA).
    if not torch.backends.cuda.is_built():
        return
    registry.register_op_override(
        "native",
        "aten",
        "linalg_polar",
        "CUDA",
        cond=_polar_cond_cuda,
        impl=_polar_impl_cuda,
    )
    registry.register_op_override(
        "native",
        "aten",
        "linalg_polar.out",
        "CUDA",
        cond=_polar_cond_cuda,
        impl=_polar_out_impl_cuda,
    )
<|endoftext|>"""CuTeDSL override registrations for ``aten::topk``.

Two kernels, picked by (K, N) - see ``cutedsl_kernels.py``:

  * Register-resident (small K, small N): K in {16, 32}, N a power of 2
    in a per-K range (see ``_REGISTER_N_RANGE``). Each warp sorts one
    row entirely in registers and writes only K outputs to gmem.
    Bit-exact to aten on values; indices on ties land in
    ``(value desc, idx asc)`` order which doesn't match aten's small-K
    CUDA kernel - same gather invariant though.

  * Fused radix-select (larger K): K in {64, 128, 256, 512, 1024}.
    Four radix byte passes over smem histograms, then a cooperative
    bitonic sort of the K survivors. Two phase-2/phase-3 specialisations:
      - Deterministic (under ``torch.use_deterministic_algorithms``):
        block-wide prefix-sum gather + lex ``(ord, -idx)`` sort.
        Bit-exact match to aten on values and indices.
      - Non-deterministic: smem atomic-counter gather + ord-only sort.
        Faster (~5-10%); indices may differ across runs on threshold ties.

Common eligibility (see ``_cond``):
  - fp32 input, CUDA, not COW
  - ``largest=True``, ``sorted=True``
  - reducing over the last axis, ``self`` contiguous (2D flatten is a view)
  - row count at least one full wave of SMs (perf gate)

Per-kernel additional eligibility:
  - register: K in {16, 32}, N pow2 in supported range above
  - radix: K in {64, 128, 256, 512, 1024}, ``N >= MIN_N_MULT[k] * k``,
    ``N % 4 == 0`` (128-bit vector loads)

Anything else falls through to aten.
"""

import functools
import math

import torch

from ... import cutedsl_utils as cu
from ._common import (
    any_cow,
    flatten_last_dim,
    last_dim_row_major_ok,
    unflatten_last_dim,
)


_RADIX_KS: frozenset[int] = frozenset({64, 128, 256, 512, 1024})
_REGISTER_KS: frozenset[int] = frozenset({16, 32})
_SUPPORTED_KS: frozenset[int] = _RADIX_KS | _REGISTER_KS

# Per-K minimum N for the radix kernel below which aten wins on B200.
# At low N the radix sweep (4 passes + gather) has too many passes over
# too little data to beat aten's launch-bound multi-pass kernel.
# K=512 needs ~8*K, K=1024 needs ~32*K because their smem footprint
# drops occupancy.
_RADIX_MIN_N_MULTIPLIER: dict[int, int] = {
    64: 2,
    128: 2,
    256: 2,
    512: 8,
    1024: 32,
}

# Register kernel per-K (min, max) N range, both bounds powers of 2.
# Below the min, VEC = N/32 is too small (most lanes idle, bitonic
# overhead dominates) and aten wins at large M. Above the max, radix
# (K=32) or aten (K=16 N>2048) takes over.
# Tuned on B200: K=16 wins down to N=64 across all M; K=32 only beats
# both aten and radix at exactly N=256.
_REGISTER_N_RANGE: dict[int, tuple[int, int]] = {
    16: (64, 2048),
    32: (256, 256),
}


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _kernel_for(k: int, n: int) -> str | None:
    """Pick the kernel for (K, N): "register", "radix", or None (aten)."""
    if k in _REGISTER_KS:
        n_min, n_max = _REGISTER_N_RANGE[k]
        if _is_pow2(n) and n_min <= n <= n_max:
            return "register"
    if k in _RADIX_KS:
        if n >= _RADIX_MIN_N_MULTIPLIER[k] * k and (n % 4) == 0:
            return "radix"
    return None


@functools.cache
def _min_rows_for_full_wave(device_idx: int) -> int:
    """Row threshold below which the one-CTA-per-row kernel underutilises
    the GPU. A full wave is SM-count CTAs; below that, aten's multi-CTA
    kernel gets more parallelism out of the same rows and wins."""
    return torch.cuda.get_device_properties(device_idx).multi_processor_count


def _eligible(
    self: torch.Tensor, k: int, dim: int, largest: bool, sorted_: bool
) -> bool:
    if not self.is_cuda or self.dtype != torch.float32:
        return False
    if any_cow(self):
        return False
    if not largest or not sorted_:
        return False
    if not last_dim_row_major_ok(self, dim):
        return False
    N = self.shape[-1] if self.ndim >= 1 else 0
    if _kernel_for(k, N) is None:
        return False
    # Performance gate: reject shapes where aten is faster. One CTA per
    # row (radix) or one warp per row (register) - either way row_count
    # below SM_count leaves the GPU underutilised.
    M = math.prod(self.shape[:-1]) if self.ndim >= 1 else 0
    if M < _min_rows_for_full_wave(self.device.index or 0):
        return False
    return True


def _cond(
    self: torch.Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
    *args,
    **kwargs,
) -> bool:
    return _eligible(self, int(k), int(dim), bool(largest), bool(sorted))


def _out_cond(
    self: torch.Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
    *,
    values: torch.Tensor,
    indices: torch.Tensor,
) -> bool:
    if not _cond(self, k, dim, largest, sorted):
        return False
    if any_cow(values, indices):
        return False
    expected_shape = self.shape[:-1] + (k,)
    if values.dtype != torch.float32 or values.shape != expected_shape:
        return False
    if indices.dtype != torch.int64 or indices.shape != expected_shape:
        return False
    return True


def _run(self: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    from .cutedsl_kernels import topk_radix, topk_register

    self_2d = flatten_last_dim(self)
    N = self_2d.shape[-1]
    kernel = _kernel_for(k, N)

    def _launch() -> tuple[torch.Tensor, torch.Tensor]:
        if kernel == "register":
            return topk_register(self_2d, k)
        # Pick the deterministic kernel under torch.use_deterministic_algorithms
        # (kept off by default for perf; the non-det kernel still produces
        # correct top-K values but indices may differ on ties).
        deterministic = torch.are_deterministic_algorithms_enabled()
        return topk_radix(self_2d, k, deterministic=deterministic)

    # The Python-native dispatch path does not get an automatic CUDA device
    # guard before launching the kernel (unlike the generated C++ ATen path),
    # so the CuTeDSL kernel runs on the current device's stream. Guard the
    # launch when the input is not already on the current device, while leaving
    # the common already-current path direct to avoid the context-manager
    # overhead. See #187983 for the same fix on the bmm override.
    device = self.get_device()
    if device == torch.cuda.current_device():
        values_2d, indices_2d = _launch()
    else:
        with torch.cuda.device(device):
            values_2d, indices_2d = _launch()
    return unflatten_last_dim(values_2d, indices_2d, self, k)


def _impl(
    self: torch.Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
    *args,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _run(self, int(k))


def _out_impl(
    self: torch.Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
    *,
    values: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    v, i = _run(self, int(k))
    values.copy_(v)
    indices.copy_(i)
    return values, indices


def register_to_dispatch() -> None:
    for op_symbol, cond, impl in (
        ("topk", _cond, _impl),
        ("topk.values", _out_cond, _out_impl),
    ):
        cu.register_op_override(
            "aten",
            op_symbol,
            "CUDA",
            cond=cond,
            impl=impl,
            allow_multiple_override=True,
        )
<|endoftext|>"""CuTeDSL fp32 top-K kernels.

Two kernels live here, picked by K and N in ``cutedsl_impl.py``:

``topk_radix`` - fused radix-select for K in {64,128,256,512,1024}.
Layout per CTA (one row):
  * Phase 1 - four radix byte passes (MSB to LSB) build shared-memory
    histograms via smem atomic-add; thread 0 scans descending to locate
    the bin that contains the k-th-largest ordinal.
  * Phase 2 - gather elements at-or-above the radix threshold. Two
    specialisations selected by the ``deterministic`` constexpr:
      - True: per-step block-wide exclusive prefix sums hand each
        thread an exact write slot, eq-tied elements stage in a side
        buffer; first ``remaining_k`` of them are appended in input
        order. Stable across runs.
      - False: smem atomic counters claim slots on the fly. Cheaper
        but indices on threshold ties depend on atomic resolution
        order.
  * Phase 3 - cooperative bitonic sort descending. Block size is
    ``max(K, 256)``; K threads own one element each, the rest sit the
    sort out but still reach every barrier. Comparator is lex
    ``(ord, -idx)`` (deterministic) or ord-only (non-deterministic).
  * Phase 4 - write (values, indices) to global memory.

``topk_register`` - register-resident top-K for K in {16, 32} with
N a power of 2 in [K, 2048]. Each warp owns one row; ``ROWS_PER_CTA``
warps share a CTA. Per-thread VEC = N/32 elements held in registers,
encoded as Int64 ``(ord << 32) | ~idx`` keys so a descending sort gives
``(value desc, idx asc)`` matching aten. Local bitonic sort, then a
warp-cooperative bitonic-topk merge across butterfly partners. No smem,
no global staging - only the final K writes hit gmem. Bit-exact to aten.

Both kernels do one launch; only loads/stores of the input and final
outputs hit global memory.
"""

import math

import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T

import torch
from torch._native.instrumentation import instrumented_cutedsl_cache


_NEG_INF_BITS: int = 0xFF800000
_N_HIST_BINS: int = 256


def _num_threads_for_k(k: int) -> int:
    """Pick block size for a given K.

    Bitonic sort in phase 3 assigns one element per thread, so we need
    at least K threads. Using ``max(K, _N_HIST_BINS)`` keeps the phase-1
    histogram-zero loop one-shot for small K while letting K grow up to
    the CUDA 1024-thread cap.
    """
    return max(k, _N_HIST_BINS)


@dsl_user_op
def _elem_ptr(x: cute.Tensor, coord, *, loc=None, ip=None) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@dsl_user_op
def _bitcast_f32_to_i32(val, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.bitcast(T.i32(), Float32(val).ir_value()))


@dsl_user_op
def _bitcast_i32_to_f32(val, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.bitcast(T.f32(), Int32(val).ir_value()))


@cute.jit
def _f32_to_radix_ord(val: Float32) -> Int32:
    """Map fp32 to a monotone Int32 ordinal whose unsigned MSB->LSB byte
    order matches descending fp32 order. Negatives invert all bits;
    non-negatives keep their bits. With the byte-0 xor of 0x80 (applied
    at the histogram-bin level), this gives a total order suitable for
    MSB-first radix select.

    NaN of either sign maps to 0x7FFFFFFF (the largest ordinal under the
    byte-0 xor), matching aten's TopKTypeConfig<float> behaviour: NaN
    sorts at the very top of largest-first topk regardless of sign.
    """
    bits = _bitcast_f32_to_i32(val)
    ord_ = bits ^ ((bits >> Int32(31)) & Int32(0x7FFFFFFF))
    # NaN detection: bits with all-ones exponent and non-zero mantissa.
    # Equivalent to ``val != val``, but the DSL doesn't expose that
    # comparator; the bit test is exact for IEEE-754 fp32.
    is_nan = (bits & Int32(0x7FFFFFFF)) > Int32(0x7F800000)
    return Int32(0x7FFFFFFF) if is_nan else ord_


def _smem_scalar(smem: cutlass.utils.SmemAllocator, dtype) -> cute.Tensor:
    """Allocate a single-element smem tensor (scalar bookkeeping slot)."""
    return smem.allocate_tensor(dtype, cute.make_layout((1,)), byte_alignment=4)


@cute.jit
def _load_vec_f32(ptr: cute.Pointer, vec_elems: cutlass.Constexpr) -> cute.Tensor:
    """Vectorized gmem -> rmem load of ``vec_elems`` contiguous fp32.

    Cute picks the widest safe instruction (e.g. LDG.128 for a 4-wide
    fp32 tile) from the static shape.
    """
    gsrc = cute.make_tensor(ptr, cute.make_layout(vec_elems))
    rvals = cute.make_rmem_tensor(vec_elems, Float32)
    cute.autovec_copy(gsrc, rvals)
    return rvals


@cute.jit
def _warp_inclusive_prefix(val: Int32, lane: Int32) -> Int32:
    """Inclusive Hillis-Steele prefix sum across a single warp."""
    for i in cutlass.range_constexpr(int(math.log2(cute.arch.WARP_SIZE))):
        offset = 1 << i
        # mask_and_clamp=0 so lanes < offset get 0 (no contribution).
        partial = cute.arch.shuffle_sync_up(val, offset=offset, mask_and_clamp=0)
        if lane >= Int32(offset):
            val = val + partial
    return val


@cute.jit
def _block_excl_prefix_i32(
    val: Int32,
    scratch: cute.Tensor,
    lane: Int32,
    warp: Int32,
    num_warps: cutlass.Constexpr,
) -> tuple[Int32, Int32]:
    """Block-wide exclusive prefix sum over per-thread ``val``.

    Returns ``(excl_pfx, total)``: ``excl_pfx`` is this thread's
    exclusive prefix; ``total`` is the sum across all threads in the
    block. ``scratch`` must have at least ``num_warps + 1`` Int32 slots
    and is clobbered.

    Implementation: warp-level Hillis-Steele scan, then warp 0 scans
    the per-warp totals to produce per-warp offsets, then add.
    """
    WARP_SIZE = const_expr(cute.arch.WARP_SIZE)
    # Step 1: per-warp inclusive scan.
    incl = _warp_inclusive_prefix(val, lane)
    excl_intra = incl - val
    # Step 2: last lane in each warp publishes warp's inclusive total.
    if lane == Int32(WARP_SIZE - 1):
        scratch[warp] = incl
    cute.arch.barrier()
    # Step 3: warp 0 prefix-sums the per-warp totals.
    if warp == Int32(0):
        warp_val = Int32(0)
        if lane < Int32(num_warps):
            warp_val = scratch[lane]
        warp_incl = _warp_inclusive_prefix(warp_val, lane)
        warp_excl = warp_incl - warp_val
        if lane < Int32(num_warps):
            scratch[lane] = warp_excl
        # Stash the block total in a slot beyond the per-warp set so
        # other warps can read it without a second reduction.
        if lane == Int32(num_warps - 1):
            scratch[num_warps] = warp_incl
    cute.arch.barrier()
    excl_pfx = scratch[warp] + excl_intra
    total = scratch[num_warps]
    cute.arch.barrier()  # let everyone consume scratch before next call
    return excl_pfx, total


class _RadixSelectTopK:
    """One-CTA-per-row fused radix-select + bitonic sort.

    ``N``, ``K`` and ``deterministic`` are specialised at compile time:
    the JIT cache keys on ``(N, K, deterministic)`` so a row-length or
    determinism-mode change forces a recompile.

    When ``deterministic`` is True, phase 2 uses block-wide prefix-sum
    scans for input-stable gather and phase 3's bitonic comparator uses
    a lex ``(ord, -idx)`` key. Output matches aten bit-exactly.

    When ``deterministic`` is False, phase 2 uses smem atomic counters
    for the gather and phase 3 sorts by ord-only. Faster (no extra smem
    bookkeeping, no per-step block barrier on the gather), but ties at
    the threshold radix bin are resolved non-deterministically and
    indices may differ across runs.
    """

    VEC: int = 4

    def __init__(self, N: int, K: int, deterministic: bool = True):
        self.N = N
        self.K = K
        self.deterministic = deterministic
        self.NUM_THREADS = _num_threads_for_k(K)
        self.NUM_STAGES = int(math.log2(K))
        TILE = self.NUM_THREADS * self.VEC
        self.VEC_ITERS = N // TILE
        self.VEC_TAIL_START = self.VEC_ITERS * TILE
        self.SCALAR_TAIL_ITERS = (
            N - self.VEC_TAIL_START + self.NUM_THREADS - 1
        ) // self.NUM_THREADS

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mValues: cute.Tensor,
        mIndices: cute.Tensor,
        stream: cuda.CUstream,
    ):
        M = mX.shape[0]
        self.kernel(mX, mValues, mIndices).launch(
            grid=[M, 1, 1],
            block=[self.NUM_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mValues: cute.Tensor, mIndices: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()

        N = const_expr(self.N)
        K = const_expr(self.K)
        NT = const_expr(self.NUM_THREADS)
        VEC = const_expr(self.VEC)
        VEC_ITERS = const_expr(self.VEC_ITERS)
        VEC_TAIL_START = const_expr(self.VEC_TAIL_START)
        SCALAR_TAIL_ITERS = const_expr(self.SCALAR_TAIL_ITERS)
        N_HIST_BINS = const_expr(_N_HIST_BINS)

        smem = cutlass.utils.SmemAllocator()
        s_hist = smem.allocate_tensor(
            Int32, cute.make_layout((256,)), byte_alignment=16
        )
        s_prefix = _smem_scalar(smem, Int32)
        s_mask = _smem_scalar(smem, Int32)
        s_rem_k = _smem_scalar(smem, Int32)
        # Phase-2 bookkeeping. Non-deterministic gather uses two
        # atomic-add counters; deterministic gather carries running
        # totals in registers.
        s_write_ctr = _smem_scalar(smem, Int32)
        s_eq_ctr = _smem_scalar(smem, Int32)
        s_vals = smem.allocate_tensor(
            Float32, cute.make_layout((K,)), byte_alignment=16
        )
        s_ords = smem.allocate_tensor(Int32, cute.make_layout((K,)), byte_alignment=16)
        s_idxs = smem.allocate_tensor(Int32, cute.make_layout((K,)), byte_alignment=16)
        # Side buffer for equal-threshold elements (deterministic gather only).
        # Always allocated to keep the smem layout straightforward;
        # measurements show no occupancy benefit from making it conditional.
        s_eq_vals = smem.allocate_tensor(
            Float32, cute.make_layout((K,)), byte_alignment=16
        )
        s_eq_idxs = smem.allocate_tensor(
            Int32, cute.make_layout((K,)), byte_alignment=16
        )

        if tidx == 0:
            s_prefix[0] = Int32(0)
            s_mask[0] = Int32(0)
            s_rem_k[0] = Int32(K)
        cute.arch.barrier()

        # Phase 1: four radix byte passes, MSB to LSB.
        for byte_pos in cutlass.range_constexpr(4):
            shift = (3 - byte_pos) * 8
            xor_val = 128 if byte_pos == 0 else 0

            # Zero the 256-bin histogram. When NT > 256 (e.g. K=1024) only
            # the first 256 threads participate; the histogram is always
            # exactly 256 bins regardless of K.
            if tidx < N_HIST_BINS:
                s_hist[tidx] = Int32(0)
            cute.arch.barrier()

            prefix = s_prefix[0]
            decided_mask = s_mask[0]

            if byte_pos == 0:
                # First pass: decided_mask==0, every element participates.
                for step in cutlass.range(VEC_ITERS):
                    base = step * NT * VEC + tidx * VEC
                    rvals = _load_vec_f32(_elem_ptr(mX, (row, base)), VEC)
                    for vi in cutlass.range_constexpr(self.VEC):
                        ords = _f32_to_radix_ord(rvals[vi])
                        byte_val = ((ords >> Int32(shift)) & Int32(255)) ^ Int32(
                            xor_val
                        )
                        cute.arch.atomic_add(_elem_ptr(s_hist, byte_val), Int32(1))
                for step in cutlass.range(SCALAR_TAIL_ITERS):
                    idx = VEC_TAIL_START + step * NT + tidx
                    if idx < N:
                        ords = _f32_to_radix_ord(mX[row, idx])
                        byte_val = ((ords >> Int32(shift)) & Int32(255)) ^ Int32(
                            xor_val
                        )
                        cute.arch.atomic_add(_elem_ptr(s_hist, byte_val), Int32(1))
            else:
                for step in cutlass.range(VEC_ITERS):
                    base = step * NT * VEC + tidx * VEC
                    rvals = _load_vec_f32(_elem_ptr(mX, (row, base)), VEC)
                    for vi in cutlass.range_constexpr(self.VEC):
                        ords = _f32_to_radix_ord(rvals[vi])
                        if (ords & decided_mask) == prefix:
                            byte_val = ((ords >> Int32(shift)) & Int32(255)) ^ Int32(
                                xor_val
                            )
                            cute.arch.atomic_add(_elem_ptr(s_hist, byte_val), Int32(1))
                for step in cutlass.range(SCALAR_TAIL_ITERS):
                    idx = VEC_TAIL_START + step * NT + tidx
                    if idx < N:
                        ords = _f32_to_radix_ord(mX[row, idx])
                        if (ords & decided_mask) == prefix:
                            byte_val = ((ords >> Int32(shift)) & Int32(255)) ^ Int32(
                                xor_val
                            )
                            cute.arch.atomic_add(_elem_ptr(s_hist, byte_val), Int32(1))
            cute.arch.barrier()

            # Thread 0 picks the bin that contains the k-th-largest.
            if tidx == 0:
                remaining_k = s_rem_k[0]
                acc = Int32(0)
                found = Int32(0)
                sel_bin = Int32(0)
                elems_above = Int32(0)

                for b in cutlass.range(256):
                    bin_idx = Int32(255) - b
                    count = s_hist[bin_idx]
                    if found == Int32(0):
                        if acc + count >= remaining_k:
                            sel_bin = bin_idx
                            elems_above = acc
                            found = Int32(1)
                    acc = acc + count

                actual_byte = sel_bin ^ Int32(xor_val)
                s_prefix[0] = prefix | (actual_byte << Int32(shift))
                s_mask[0] = decided_mask | (Int32(255) << Int32(shift))
                s_rem_k[0] = remaining_k - elems_above

            cute.arch.barrier()

        # Phase 2: gather elements at-or-above the radix threshold.
        # Two implementations selected at compile time:
        #   * Deterministic: per-step block-wide exclusive prefix sums
        #     give each thread an exact write slot; eq-threshold ties
        #     stage in a side buffer and the first ``remaining_k`` get
        #     appended in input-index order.
        #   * Non-deterministic: smem atomic counters claim slots on
        #     the fly. Faster but tied elements at the threshold bin
        #     are picked in atomic-resolution order.
        threshold = s_prefix[0]
        remaining_k = s_rem_k[0]

        if tidx == 0:
            s_write_ctr[0] = Int32(0)
            s_eq_ctr[0] = Int32(0)
        if tidx < K:
            s_vals[tidx] = _bitcast_i32_to_f32(Int32(_NEG_INF_BITS))
            s_idxs[tidx] = Int32(0)
        cute.arch.barrier()

        if const_expr(self.deterministic):
            WARP_SIZE = const_expr(cute.arch.WARP_SIZE)
            NUM_WARPS = const_expr(NT // WARP_SIZE)
            lane_idx = tidx % Int32(WARP_SIZE)
            warp_idx = tidx // Int32(WARP_SIZE)

            # --- Vec + scalar tail (deterministic) ---
            # Pack (n_above, n_eq) into a single Int32 as (high16, low16) so
            # we can run one block-wide prefix sum per step instead of two.
            # Per-step bounds: above+eq counts <= VEC*NT = 4*1024 = 4096,
            # comfortably below 2^16; the sums share no bits so the low-half
            # never carries into the high half.
            #
            # Carry the running bases in registers (every thread has
            # ``packed_step_total`` from the scan) instead of going through
            # smem each step.
            above_base = Int32(0)
            eq_base = Int32(0)
            for step in cutlass.range(VEC_ITERS):
                base = step * NT * VEC + tidx * VEC
                rvals = _load_vec_f32(_elem_ptr(mX, (row, base)), VEC)

                # Hoist per-element classification into per-lane registers
                # so the count pass and write pass don't redo the radix-ord
                # transform and threshold compare.
                # Class encoding: 2 = above, 1 = eq, 0 = below.
                cls_reg = cute.make_rmem_tensor(VEC, Int32)
                packed_local = Int32(0)
                for vi in cutlass.range_constexpr(self.VEC):
                    o = _f32_to_radix_ord(rvals[vi])
                    if o > threshold:
                        cls_reg[vi] = Int32(2)
                        packed_local = packed_local + Int32(1 << 16)
                    elif o == threshold:
                        cls_reg[vi] = Int32(1)
                        packed_local = packed_local + Int32(1)
                    else:
                        cls_reg[vi] = Int32(0)

                # Block-wide exclusive prefix sum. ``s_hist`` is reused as
                # scratch for the warp-total scan (Phase 1 is done with it).
                packed_pfx, packed_step_total = _block_excl_prefix_i32(
                    packed_local, s_hist, lane_idx, warp_idx, NUM_WARPS
                )
                above_pfx = packed_pfx >> Int32(16)
                eq_pfx = packed_pfx & Int32(0xFFFF)

                # Per-thread write offsets within this step.
                my_above = above_base + above_pfx
                my_eq = eq_base + eq_pfx
                for vi in cutlass.range_constexpr(self.VEC):
                    val_f32 = rvals[vi]
                    cls = cls_reg[vi]
                    idx = base + Int32(vi)
                    if cls == Int32(2):
                        if my_above < Int32(K):
                            s_vals[my_above] = val_f32
                            s_idxs[my_above] = Int32(idx)
                        my_above = my_above + Int32(1)
                    elif cls == Int32(1):
                        if my_eq < remaining_k:
                            s_eq_vals[my_eq] = val_f32
                            s_eq_idxs[my_eq] = Int32(idx)
                        my_eq = my_eq + Int32(1)

                # Advance running bases. Cap eq at remaining_k - we never
                # need more. All threads update in registers; no smem.
                above_base = above_base + (packed_step_total >> Int32(16))
                eq_base = min(
                    remaining_k, eq_base + (packed_step_total & Int32(0xFFFF))
                )

            # --- Scalar tail ---
            for step in cutlass.range(SCALAR_TAIL_ITERS):
                idx = VEC_TAIL_START + step * NT + tidx
                valid = idx < N
                packed_local = Int32(0)
                if valid:
                    ords = _f32_to_radix_ord(mX[row, idx])
                    if ords > threshold:
                        packed_local = Int32(1 << 16)
                    elif ords == threshold:
                        packed_local = Int32(1)

                packed_pfx, packed_step_total = _block_excl_prefix_i32(
                    packed_local, s_hist, lane_idx, warp_idx, NUM_WARPS
                )
                above_pfx = packed_pfx >> Int32(16)
                eq_pfx = packed_pfx & Int32(0xFFFF)

                my_above = above_base + above_pfx
                my_eq = eq_base + eq_pfx
                if valid:
                    val_f32 = mX[row, idx]
                    ords = _f32_to_radix_ord(val_f32)
                    if ords > threshold:
                        if my_above < Int32(K):
                            s_vals[my_above] = val_f32
                            s_idxs[my_above] = Int32(idx)
                    elif ords == threshold:
                        if my_eq < remaining_k:
                            s_eq_vals[my_eq] = val_f32
                            s_eq_idxs[my_eq] = Int32(idx)

                above_base = above_base + (packed_step_total >> Int32(16))
                eq_base = min(
                    remaining_k, eq_base + (packed_step_total & Int32(0xFFFF))
                )

            # The eq-merge below has each thread read s_eq_vals[tidx - n_above],
            # i.e. a slot written by some other thread during the gather above.
            # The last barrier inside _block_excl_prefix_i32 fires *before* the
            # final per-step writes to s_eq_vals, so cross-thread visibility of
            # those writes isn't guaranteed without a fresh barrier here.
            cute.arch.barrier()

            # Append eq-bucket survivors after the above-bucket ones to
            # fill ``s_vals`` / ``s_idxs`` to exactly K entries.
            n_above = above_base
            if tidx < K:
                offset_into_eq = Int32(tidx) - n_above
                if offset_into_eq >= Int32(0) and offset_into_eq < remaining_k:
                    s_vals[tidx] = s_eq_vals[offset_into_eq]
                    s_idxs[tidx] = s_eq_idxs[offset_into_eq]
            cute.arch.barrier()
        else:
            # Non-deterministic atomic-counter gather.
            for step in cutlass.range(VEC_ITERS):
                base = step * NT * VEC + tidx * VEC
                rvals = _load_vec_f32(_elem_ptr(mX, (row, base)), VEC)
                for vi in cutlass.range_constexpr(self.VEC):
                    val_f32 = rvals[vi]
                    idx = base + Int32(vi)
                    ords = _f32_to_radix_ord(val_f32)
                    if ords > threshold:
                        pos = cute.arch.atomic_add(_elem_ptr(s_write_ctr, 0), Int32(1))
                        if pos < Int32(K):
                            s_vals[pos] = val_f32
                            s_idxs[pos] = Int32(idx)
                    elif ords == threshold:
                        eq_pos = cute.arch.atomic_add(_elem_ptr(s_eq_ctr, 0), Int32(1))
                        if eq_pos < remaining_k:
                            pos = cute.arch.atomic_add(
                                _elem_ptr(s_write_ctr, 0), Int32(1)
                            )
                            if pos < Int32(K):
                                s_vals[pos] = val_f32
                                s_idxs[pos] = Int32(idx)
            for step in cutlass.range(SCALAR_TAIL_ITERS):
                idx = VEC_TAIL_START + step * NT + tidx
                if idx < N:
                    val_f32 = mX[row, idx]
                    ords = _f32_to_radix_ord(val_f32)
                    if ords > threshold:
                        pos = cute.arch.atomic_add(_elem_ptr(s_write_ctr, 0), Int32(1))
                        if pos < Int32(K):
                            s_vals[pos] = val_f32
                            s_idxs[pos] = Int32(idx)
                    elif ords == threshold:
                        eq_pos = cute.arch.atomic_add(_elem_ptr(s_eq_ctr, 0), Int32(1))
                        if eq_pos < remaining_k:
                            pos = cute.arch.atomic_add(
                                _elem_ptr(s_write_ctr, 0), Int32(1)
                            )
                            if pos < Int32(K):
                                s_vals[pos] = val_f32
                                s_idxs[pos] = Int32(idx)
            cute.arch.barrier()

        # Phase 3: cooperative bitonic sort (descending) on a composite
        # ``(ord, ~idx)`` Int64 key so equal-value entries break ties by
        # input index, matching aten's stable ordering.
        # K lanes own one element each; lanes >= K clamp their smem
        # index to a dummy slot so they can participate in barriers
        # without OOB reads/writes. The writes from those lanes are
        # discarded via the `tidx < K` guard on the store.
        sort_tidx = tidx if tidx < Int32(K) else Int32(0)
        if tidx < K:
            s_ords[tidx] = _f32_to_radix_ord(s_vals[sort_tidx])
        cute.arch.barrier()

        for stage in cutlass.range_constexpr(self.NUM_STAGES):
            for sub_rev in cutlass.range_constexpr(stage + 1):
                sub = stage - sub_rev
                step_size = 1 << sub

                # Snapshot both participants before the barrier; after the
                # barrier the other thread may have overwritten its slot,
                # so we must read our own ord now. Inactive lanes clamp
                # their partner index to 0 (a safe in-bounds read) - their
                # resulting comparisons are discarded by the store guard.
                partner = (tidx ^ Int32(step_size)) if tidx < Int32(K) else Int32(0)
                my_o = s_ords[sort_tidx]
                my_i = s_idxs[sort_tidx]
                p_o = s_ords[partner]
                p_v = s_vals[partner]
                p_i = s_idxs[partner]
                cute.arch.barrier()

                # Comparator. Deterministic mode uses a lex
                # ``(ord, -idx)`` tie-break so equal-ord entries land in
                # ascending-idx order (matching aten). Non-deterministic
                # mode skips the tiebreak (cheaper compare-exchange).
                if const_expr(self.deterministic):
                    self_lt_partner = (my_o < p_o) or (my_o == p_o and my_i > p_i)
                    self_gt_partner = (my_o > p_o) or (my_o == p_o and my_i < p_i)
                else:
                    self_lt_partner = my_o < p_o
                    self_gt_partner = my_o > p_o
                block_dir = (tidx >> Int32(stage + 1)) & Int32(1)
                if tidx < K:
                    if tidx < partner:
                        if block_dir == Int32(0):
                            if self_lt_partner:
                                s_ords[tidx] = p_o
                                s_vals[tidx] = p_v
                                s_idxs[tidx] = p_i
                        else:
                            if self_gt_partner:
                                s_ords[tidx] = p_o
                                s_vals[tidx] = p_v
                                s_idxs[tidx] = p_i
                    else:
                        if block_dir == Int32(0):
                            if self_gt_partner:
                                s_ords[tidx] = p_o
                                s_vals[tidx] = p_v
                                s_idxs[tidx] = p_i
                        else:
                            if self_lt_partner:
                                s_ords[tidx] = p_o
                                s_vals[tidx] = p_v
                                s_idxs[tidx] = p_i
                cute.arch.barrier()

        # Phase 4: results to gmem.
        if tidx < K:
            mValues[row, tidx] = s_vals[tidx]
            mIndices[row, tidx] = s_idxs[tidx]


def _make_fake_tensor(dtype, shape, divisibility=1):
    stride = tuple(
        cute.sym_int64(divisibility=divisibility) if i != len(shape) - 1 else 1
        for i in range(len(shape))
    )
    return cute.runtime.make_fake_tensor(
        dtype,
        shape,
        stride=stride,
        assumed_align=divisibility * dtype.width // 8,
    )


@instrumented_cutedsl_cache(
    "aten::topk",
    key_fn=lambda N, K, deterministic: f"radix N={N} K={K} det={deterministic}",
)
def _compile_topk_radix(N: int, K: int, deterministic: bool):
    batch_sym = cute.sym_int()
    div_n = math.gcd(4, N)
    div_k = math.gcd(4, K)
    x_fake = _make_fake_tensor(Float32, (batch_sym, N), div_n)
    v_fake = _make_fake_tensor(Float32, (batch_sym, K), div_k)
    i_fake = _make_fake_tensor(Int32, (batch_sym, K), div_k)
    return cute.compile(
        _RadixSelectTopK(N, K, deterministic=deterministic),
        x_fake,
        v_fake,
        i_fake,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def topk_radix(
    x: torch.Tensor, k: int, *, deterministic: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused radix-select top-K for fp32 2D contiguous ``x``.

    Returns ``(values[M, K]: fp32, indices[M, K]: int64)`` with K sorted
    descending per row. Caller must ensure ``k`` is in the supported set
    (see ``cutedsl_impl.py``) and ``N % 4 == 0``.

    ``deterministic`` controls the gather + sort tie-breaking strategy:
    True (default) uses prefix-sum gather + lex comparator and matches
    aten bit-exactly; False uses smem atomic gather + ord-only sort and
    is faster but indices may differ on ties.
    """
    M, N = x.shape
    out_v = torch.empty(M, k, dtype=torch.float32, device=x.device)
    # Kernel writes int32 indices; widen before returning to match aten.
    out_i_i32 = torch.empty(M, k, dtype=torch.int32, device=x.device)
    _compile_topk_radix(N, k, deterministic)(x, out_v, out_i_i32)
    return out_v, out_i_i32.to(torch.int64)


# ---------------------------------------------------------------------------
# Register-resident top-K (small K, small N).
# ---------------------------------------------------------------------------


@cute.jit
def _make_key(val: Float32, idx: Int32) -> Int64:
    """Compose the descending-sort key: ``(ord_u32 << 32) | (~idx_u32)``.

    Sorting this Int64 descending gives ``(value desc, idx asc)`` -
    exactly aten's tie-break order. Real fp32 inputs (no NaN) keep keys
    away from INT64_MIN, leaving that bit pattern free as a sentinel.
    """
    ord32 = _f32_to_radix_ord(val)
    ord64 = Int64(ord32) & Int64(0xFFFFFFFF)
    inv_idx64 = Int64(~idx) & Int64(0xFFFFFFFF)
    return (ord64 << Int64(32)) | inv_idx64


@cute.jit
def _decode_key(key: Int64) -> tuple[Float32, Int32]:
    ord32 = Int32(key >> Int64(32))
    inv_idx = Int32(key & Int64(0xFFFFFFFF))
    val_bits = ord32 ^ ((ord32 >> Int32(31)) & Int32(0x7FFFFFFF))
    return Float32(llvm.bitcast(T.f32(), val_bits.ir_value())), ~inv_idx


@cute.jit
def _cas_desc(arr: cute.Tensor, i: cutlass.Constexpr, j: cutlass.Constexpr) -> None:
    a = arr[i]
    b = arr[j]
    if a < b:
        arr[i] = b
        arr[j] = a


@cute.jit
def _bitonic_sort_desc(arr: cute.Tensor, n: cutlass.Constexpr) -> None:
    if const_expr(n > 1):
        num_stages = int(math.log2(n))
        for s in cutlass.range_constexpr(num_stages):
            for sub_rev in cutlass.range_constexpr(s + 1):
                step = 1 << (s - sub_rev)
                for i in cutlass.range_constexpr(n):
                    j = i ^ step
                    if j > i:
                        block_dir = (i >> (s + 1)) & 1
                        if block_dir == 0:
                            _cas_desc(arr, i, j)
                        else:
                            a = arr[i]
                            b = arr[j]
                            if a > b:
                                arr[i] = b
                                arr[j] = a


@cute.jit
def _bitonic_merge_desc(arr: cute.Tensor, n: cutlass.Constexpr) -> None:
    """Merge a bitonic sequence of length n into descending sorted."""
    if const_expr(n > 1):
        num_levels = int(math.log2(n))
        for level in cutlass.range_constexpr(num_levels):
            length = n >> level
            step = length // 2
            for i in cutlass.range_constexpr(n // length):
                start_i = i * length
                for j in cutlass.range_constexpr(step):
                    _cas_desc(arr, start_i + j, start_i + j + step)


@cute.jit
def _topk_merge_desc(a: cute.Tensor, b: cute.Tensor, K: cutlass.Constexpr) -> None:
    """Top-K of two K-sorted-descending sequences, in place into ``a``.

    ``a[i] <- max(a[i], b[K-1-i])`` yields a bitonic sequence in ``a``;
    one bitonic merge of length K finishes it.
    """
    for i in cutlass.range_constexpr(K):
        x = a[i]
        y = b[K - 1 - i]
        if y > x:
            a[i] = y
    _bitonic_merge_desc(a, K)


class _RegisterTopK:
    """Register-resident top-K. Each warp owns one row; ``ROWS_PER_CTA``
    warps per CTA. K in {16, 32}, N a power of 2 in [K, 2048].
    """

    def __init__(self, N: int, K: int, rows_per_cta: int = 4):
        self.N = N
        self.K = K
        self.VEC = N // 32
        self.ROWS_PER_CTA = rows_per_cta
        self.NUM_THREADS = 32 * rows_per_cta

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mValues: cute.Tensor,
        mIndices: cute.Tensor,
        stream: cuda.CUstream,
    ):
        M = mX.shape[0]
        ROWS = const_expr(self.ROWS_PER_CTA)
        num_blocks = (M + ROWS - 1) // ROWS
        self.kernel(mX, mValues, mIndices).launch(
            grid=[num_blocks, 1, 1],
            block=[self.NUM_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mValues: cute.Tensor, mIndices: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        K = const_expr(self.K)
        VEC = const_expr(self.VEC)
        ROWS = const_expr(self.ROWS_PER_CTA)
        WARP_SIZE = const_expr(cute.arch.WARP_SIZE)

        warp_idx = tidx // Int32(WARP_SIZE)
        lane_idx = tidx % Int32(WARP_SIZE)
        row = Int32(bidx) * Int32(ROWS) + warp_idx

        M = mX.shape[0]
        # CuTeDSL can't early-return on a dynamic predicate. Clamp the row
        # for loads to keep them in-bounds, and gate the writes only.
        row_safe = row if row < M else Int32(0)
        in_bounds = row < M

        # Per-thread Int64 keys covering this row's VEC-sized stripe.
        keys = cute.make_rmem_tensor(VEC, Int64)
        for i in cutlass.range_constexpr(VEC):
            col = lane_idx * Int32(VEC) + Int32(i)
            keys[i] = _make_key(mX[row_safe, col], col)

        _bitonic_sort_desc(keys, VEC)

        # Build per-thread top-K. If VEC < K, pad with INT64_MIN so the
        # sentinel sorts strictly below any real key; if VEC >= K take the
        # leading K (already sorted).
        topk = cute.make_rmem_tensor(K, Int64)
        if const_expr(VEC >= K):
            for i in cutlass.range_constexpr(K):
                topk[i] = keys[i]
        else:
            SENTINEL = const_expr(-(1 << 63))
            for i in cutlass.range_constexpr(VEC):
                topk[i] = keys[i]
            for i in cutlass.range_constexpr(K - VEC):
                topk[VEC + i] = Int64(SENTINEL)
            _bitonic_sort_desc(topk, K)

        # Warp-cooperative top-K merge via butterfly shuffles.
        log2_warp = const_expr(int(math.log2(WARP_SIZE)))
        for s in cutlass.range_constexpr(log2_warp):
            other = cute.make_rmem_tensor(K, Int64)
            for i in cutlass.range_constexpr(K):
                other[i] = cute.arch.shuffle_sync_bfly(topk[i], offset=Int32(1 << s))
            _topk_merge_desc(topk, other, K)

        # Lane 0 of each warp now holds the row's top-K descending.
        if lane_idx == Int32(0) and in_bounds:
            for i in cutlass.range_constexpr(K):
                v, idx = _decode_key(topk[i])
                mValues[row, i] = v
                mIndices[row, i] = idx


@instrumented_cutedsl_cache("aten::topk", key_fn=lambda N, K: f"register N={N} K={K}")
def _compile_topk_register(N: int, K: int):
    batch_sym = cute.sym_int()
    div_n = math.gcd(4, N)
    div_k = math.gcd(4, K)
    x_fake = _make_fake_tensor(Float32, (batch_sym, N), div_n)
    v_fake = _make_fake_tensor(Float32, (batch_sym, K), div_k)
    i_fake = _make_fake_tensor(Int32, (batch_sym, K), div_k)
    return cute.compile(
        _RegisterTopK(N, K),
        x_fake,
        v_fake,
        i_fake,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def topk_register(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Register-resident top-K for fp32 2D contiguous ``x``.

    Returns ``(values[M, K]: fp32, indices[M, K]: int64)``, descending,
    bit-exact to aten on both values and indices. Caller must ensure
    ``k`` in {16, 32} and ``N`` is a power of 2 in [k, 2048].
    """
    M, N = x.shape
    out_v = torch.empty(M, k, dtype=torch.float32, device=x.device)
    out_i_i32 = torch.empty(M, k, dtype=torch.int32, device=x.device)
    _compile_topk_register(N, k)(x, out_v, out_i_i32)
    return out_v, out_i_i32.to(torch.int64)
<|endoftext|>"""Shared eligibility checks and reshape helpers for topk overrides.

Both the CuTeDSL and Triton paths operate on a 2D ``(M, N)`` fp32 row-major
view of ``self`` where ``dim`` points at the last axis. This module hosts the
checks the conds share (contiguity, dim normalisation, COW) and the reshape
helpers the impls share.
"""

import math

import torch


def any_cow(*tensors: torch.Tensor) -> bool:
    return any(
        torch._C._is_cow_tensor(t)  # pyrefly: ignore[missing-attribute]
        for t in tensors
    )


def normalise_dim(self: torch.Tensor, dim: int) -> int:
    """Translate a possibly-negative dim into its non-negative form.

    Returns ``-1`` if the dim is out of range (cond callers should reject
    in that case).
    """
    if self.ndim == 0:
        return -1
    if dim < -self.ndim or dim >= self.ndim:
        return -1
    return dim % self.ndim


def last_dim_row_major_ok(self: torch.Tensor, dim: int) -> bool:
    """True iff reducing over ``dim`` is equivalent to reducing over the last
    axis of a 2D ``(prod(leading dims), N)`` row-major view.

    Requires ``dim`` to be the last axis and ``self`` to be contiguous in the
    standard row-major sense; together they ensure the flatten-to-2D view we
    pass to the kernel is a view (not a copy) and has stride-1 on N.
    """
    norm = normalise_dim(self, dim)
    if norm != self.ndim - 1:
        return False
    return self.is_contiguous()


def flatten_last_dim(self: torch.Tensor) -> torch.Tensor:
    """View nD self as 2D ``(M, N)`` with ``N = self.shape[-1]``.

    Caller must have verified ``last_dim_row_major_ok`` so this is a view.
    """
    if self.ndim == 2:
        return self
    if self.ndim == 1:
        return self.view(1, self.shape[0])
    N = self.shape[-1]
    M = math.prod(self.shape[:-1])
    return self.view(M, N)


def unflatten_last_dim(
    flat_values: torch.Tensor, flat_indices: torch.Tensor, self: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reshape the 2D ``(M, K)`` kernel outputs back into ``self``'s shape."""
    if self.ndim == 2:
        return flat_values, flat_indices
    if self.ndim == 1:
        return flat_values.view(k), flat_indices.view(k)
    out_shape = self.shape[:-1] + (k,)
    return flat_values.view(out_shape), flat_indices.view(out_shape)
<|endoftext|>import triton
import triton.language as tl

import torch
from torch._native.instrumentation import instrumented_triton_cache

from ...triton import ConstTensorWrapper


def _bmm_log_key(a, b, out, B, M, N, *strides, BLOCK_M, BLOCK_N) -> str:
    # Receives the kernel's launch args; BLOCK_M/BLOCK_N are the constexprs
    # that (with shapes/dtype) form the Triton compile key.
    return f"bmm_outer B={B} M={M} N={N} {a.dtype} BLOCK_M={BLOCK_M} BLOCK_N={BLOCK_N}"


@instrumented_triton_cache("aten::bmm", key_fn=_bmm_log_key)
def _bmm_outer_product_kernel(
    A_ptr,
    B_ptr,
    OUT_ptr,
    B_dim,
    M,
    N,
    stride_ab,
    stride_am,
    stride_bb,
    stride_bn,
    stride_ob,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    tiles_per_batch = grid_m * grid_n

    pid_b = pid // tiles_per_batch
    pid_mn = pid % tiles_per_batch
    pid_m = pid_mn // grid_n
    pid_n = pid_mn % grid_n

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = rm < M
    mask_n = rn < N

    a = tl.load(A_ptr + pid_b * stride_ab + rm * stride_am, mask=mask_m, other=0.0)
    b = tl.load(B_ptr + pid_b * stride_bb + rn * stride_bn, mask=mask_n, other=0.0)

    out = a[:, None] * b[None, :]

    mask = mask_m[:, None] & mask_n[None, :]  # pyrefly: ignore[bad-index]
    tl.store(
        OUT_ptr + pid_b * stride_ob + rm[:, None] * stride_om + rn[None, :] * stride_on,
        out,
        mask=mask,
    )


def _pick_block_sizes(m: int, n: int) -> tuple[int, int]:
    """I swept over some shapes and in the future we should figure out @autotune story"""
    if m <= 32:
        block_m = triton.next_power_of_2(m)
    elif m <= 96:
        block_m = 32
    elif m <= 192:
        block_m = 64
    else:
        block_m = 128
    return block_m, min(triton.next_power_of_2(n), 128)


def bmm_outer_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    B, M, _ = a.shape
    N = b.shape[2]

    out = torch.empty(B, M, N, dtype=a.dtype, device=a.device)

    BLOCK_M, BLOCK_N = _pick_block_sizes(M, N)

    # a and b are read-only inputs; wrap them so a copy-on-write tensor is read
    # through const_data_ptr() and not materialized. out is written directly.
    _bmm_outer_product_kernel[(B * triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)](
        ConstTensorWrapper(a),
        ConstTensorWrapper(b),
        out,
        B,
        M,
        N,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return out
<|endoftext|>import torch

from ... import triton_utils as tu


def _is_outer_product(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.ndim == 3
        and b.ndim == 3
        and a.shape[2] == 1
        and b.shape[1] == 1
        and a.numel() > 0
        and b.numel() > 0
        and not a.is_complex()
    )


def _bmm_outer_product_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    *args,
    **kwargs,
) -> torch.Tensor:
    from .triton_kernels import bmm_outer_product

    with torch.accelerator.device_index(a.get_device()):
        return bmm_outer_product(a, b)


def _is_acc_tensor(t: torch.Tensor) -> bool:
    acc = torch.accelerator.current_accelerator()
    return acc is not None and acc.type == t.device.type


def _bmm_outer_product_cond(
    a: torch.Tensor,
    b: torch.Tensor,
    *args,
    **kwargs,
) -> bool:
    # a and b are read-only here: the kernel wraps them in ConstTensorWrapper and
    # reads through const_data_ptr(), so copy-on-write inputs are not
    # materialized and need not be excluded.
    if _is_acc_tensor(a) and a.device == b.device and _is_outer_product(a, b):
        return True
    return False


def _register_for_dispatch_key(dispatch_key: str) -> None:
    tu.register_op_override(
        "aten",
        "bmm",
        dispatch_key,
        cond=_bmm_outer_product_cond,
        impl=_bmm_outer_product_impl,
        allow_multiple_override=True,
    )


def register_to_dispatch() -> None:
    _register_for_dispatch_key("CUDA")
    if torch.xpu._is_compiled():
        _register_for_dispatch_key("XPU")
<|endoftext|>"""Adaptor for quack's 2-D RMSNorm kernel interface to match ATen op signatures.

These functions handle tensor reshaping, memory allocation, and call quack's
compiled kernels directly (bypassing the ``@torch.library.custom_op`` wrapper
to avoid dispatcher overhead).
"""

from __future__ import annotations

import functools
import math

import torch
from torch._native.instrumentation import instrument_cutedsl_compile


# quack's rmsnorm compile fns are vendored, so we can't decorate them in
# place. Wrap them once at the call site instead -- they're @jit_cache so
# the instrumentation reads their cache_info(). Memoized so the wrapper (and
# its forwarded cache_info) is stable across calls. N is positional arg 6 in
# the fwd compile signature and arg 0 in the bwd one.
@functools.cache
def _instrumented_rmsnorm_fwd():
    from torch._vendor.quack.rmsnorm import _compile_rmsnorm_fwd

    return instrument_cutedsl_compile(
        "aten::_fused_rms_norm", key_fn=lambda *a, **k: f"fwd N={a[6]}"
    )(_compile_rmsnorm_fwd)


@functools.cache
def _instrumented_rmsnorm_bwd():
    from torch._vendor.quack.rmsnorm import _compile_rmsnorm_bwd

    return instrument_cutedsl_compile(
        "aten::_fused_rms_norm_backward",
        key_fn=lambda *a, **k: f"bwd N={a[0]}",
    )(_compile_rmsnorm_bwd)


# quack imports `cutlass`, which is only installed on CUDA-enabled x86 Linux
# builds. Keep the import lazy so `torch._native.ops.norm.norms` stays
# importable on CPU-only builds (test_public_bindings /
# test_circular_dependencies walk every module under `torch.`).
def _torch2cute(t: torch.Tensor | None):  # type: ignore[no-untyped-def]
    from torch._vendor.quack import cute_dsl_utils as _cute_dsl_utils

    if t is None:
        return None
    return _cute_dsl_utils.torch2cute_dtype_map[t.dtype]


def _required_align_bytes(t: torch.Tensor, N: int) -> int:
    # quack compiles the kernel assuming the input base pointer is aligned to
    # the vectorized cp.async width: gcd(N, 128 // dtype_bits) elements. A
    # contiguous tensor with a non-zero storage offset (e.g. buf[1:].view(M, N))
    # has fine strides but a misaligned base, which trips CuTe's runtime
    # alignment check ("Misaligned Tensor data on argument #0").
    itemsize = t.element_size()
    return math.gcd(N, 128 // (itemsize * 8)) * itemsize


def _const_data_ptr(t: torch.Tensor) -> int:
    with torch._C.DisableTorchFunctionSubclass():
        return t.const_data_ptr()  # type: ignore[attr-defined]


def _read_only(t: torch.Tensor | None):  # type: ignore[no-untyped-def]
    if t is None:
        return None
    from cutlass.cute.runtime import from_dlpack

    from torch.utils.dlpack import ReadOnlyTensorWrapper

    with torch._C.DisableTorchFunctionSubclass():
        t = ReadOnlyTensorWrapper(t)
    return from_dlpack(t, enable_tvm_ffi=True)


def _reshape_2d(t: torch.Tensor, M: int, N: int) -> torch.Tensor:
    align = _required_align_bytes(t, N)
    if t.ndim == 2 and t.shape[0] == M and t.shape[1] == N and t.is_contiguous():
        # .contiguous() is a no-op on an already-contiguous tensor and would
        # preserve a misaligned base; clone to force a fresh (aligned) buffer.
        return t if _const_data_ptr(t) % align == 0 else t.clone()
    out = t.reshape(M, N).contiguous()
    # reshape can return a view that keeps the original (misaligned) storage
    # offset, and .contiguous() then leaves it untouched.
    return out if _const_data_ptr(out) % align == 0 else out.clone()


def _aligned_weight(w: torch.Tensor, N: int) -> torch.Tensor:
    # Same trap as _reshape_2d: reshape(N).contiguous() is a no-op for a
    # contiguous-but-offset weight, leaving a misaligned base. Weight is only
    # N elements, so always clone rather than perf-gating like the input.
    w = w.reshape(N).contiguous()
    if _const_data_ptr(w) % _required_align_bytes(w, N) != 0:
        w = w.clone()
    return w


def _flatten_rstd(t: torch.Tensor, M: int) -> torch.Tensor:
    if t.ndim == 1 and t.shape[0] == M:
        return t
    if t.is_contiguous() and t.numel() == M:
        return t.detach().view(M)
    return t.reshape(M).contiguous()


def quack_rmsnorm_fwd(
    input: torch.Tensor,
    weight: torch.Tensor | None,
    normalized_shape: list[int],
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_shape = input.shape
    N = math.prod(normalized_shape)
    M = input.numel() // N
    # quack's kernel requires a contiguous 2-D input with trailing stride 1;
    # a plain `.reshape(M, N)` can return a non-contiguous view.
    x = _reshape_2d(input, M, N)

    out = torch.empty_like(x)
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)

    if weight is not None:
        weight = _aligned_weight(weight, N)

    dtype = _torch2cute(x)
    out_dtype = _torch2cute(out)
    weight_dtype = _torch2cute(weight)

    kernel = _instrumented_rmsnorm_fwd()(
        dtype,
        out_dtype,
        None,
        weight_dtype,
        None,
        None,
        N,
        True,
        False,
        False,
        per_head=False,
    )
    # compile order: (x, weight, bias, res, out, res_out, rstd, mean, eps)
    kernel(_read_only(x), _read_only(weight), None, None, out, None, rstd, None, eps)

    out = out.reshape(input_shape)
    stat_shape = list(input_shape[: -len(normalized_shape)]) + [1] * len(
        normalized_shape
    )
    rstd = rstd.view(stat_shape)
    return out, rstd


def quack_rmsnorm_bwd(
    grad_out: torch.Tensor,
    input: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor | None,
    normalized_shape: list[int],
    dw_mask: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    N = math.prod(normalized_shape)
    M = input.numel() // N
    x = _reshape_2d(input, M, N)
    dout = _reshape_2d(grad_out, M, N)
    rstd_flat = _flatten_rstd(rstd, M)

    dx = torch.empty_like(x)
    from torch._vendor.quack.rmsnorm_config import get_sm_count

    sm_count = get_sm_count(N, x.device)

    # quack's kernel requires a contiguous 1-D weight with matching dtype.
    if weight is not None:
        weight = _aligned_weight(weight, N)

    # quack treats `weight_dtype` (whether weight is present in the kernel)
    # and `has_dw_partial` (whether to accumulate dw) as independent flags.
    # dx's formula depends on weight, so we always hand weight to the kernel
    # when the user supplied one; dw_mask only gates the dw_partial buffer.
    dw_partial: torch.Tensor | None = None
    if weight is not None and dw_mask:
        dw_partial = torch.empty(sm_count, N, device=x.device, dtype=torch.float32)

    dtype = _torch2cute(x)
    dout_dtype = _torch2cute(dout)
    dx_dtype = _torch2cute(dx)
    weight_dtype = _torch2cute(weight)

    kernel = _instrumented_rmsnorm_bwd()(
        N,
        dtype,
        dout_dtype,
        dx_dtype,
        weight_dtype,
        False,
        None,
        None,
        dw_partial is not None,
        per_head=False,
    )
    # compile order: (x, weight, dout, dres_out, rstd, dx, dw_partial, dres, db_partial, sm_count)
    kernel(
        _read_only(x),
        _read_only(weight),
        _read_only(dout),
        None,
        _read_only(rstd_flat),
        dx,
        dw_partial,
        None,
        None,
        sm_count,
    )

    dx = dx.reshape(input.shape)
    # `dw_partial is not None` implies `weight is not None` (invariant from
    # allocation above), but pyrefly can't see the dependency.
    if dw_partial is not None:
        weight_dtype_for_sum = weight.dtype  # pyrefly: ignore[missing-attribute]
        dw = dw_partial.sum(dim=0).reshape(normalized_shape).to(weight_dtype_for_sum)
    else:
        dw = None
    return dx, dw
<|endoftext|>"""Quack-backed RMSNorm overrides for aten fused RMSNorm operators.

Uses the vendored quack subset at ``torch._vendor.quack``.
"""
# mypy: allow-untyped-defs

from __future__ import annotations

import math

import torch

from ... import cutedsl_utils as cu
from .norms import _const_data_ptr, _required_align_bytes


def _is_supported(input: torch.Tensor) -> bool:
    if input.device.type != "cuda":
        return False
    if torch.version.hip is not None:
        return False
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    major, _ = torch.cuda.get_device_capability(input.device)
    return major in (9, 10)


# quack splits each row across a CTA cluster (at most 16 on SM90/SM100, the
# only archs _is_supported accepts) and stages the per-CTA row tile in shared
# memory. Rows whose tile exceeds the smem budget even at the max cluster size
# cannot launch, and far beyond that (e.g. N=2^28) the CuTe DSL compiler hangs
# or crashes before any smem check could fire (gh-186800). Bound N here so
# such rows fall back to aten. The reserve covers the reduction buffer,
# mbarriers, and smem alignment (mirrors quack's _BWD_SMEM_RESERVED_BYTES).
_SMEM_RESERVED_BYTES = 4 * 1024
_MAX_CLUSTER_N = 16
# The kernel rounds the per-CTA tile up to vecsize * threads_per_row elements
# (reduction_base._get_tiled_copy). Both are powers of 2 with vecsize <= 8 and
# threads_per_row <= 256, so rounding up to 2048 never under-estimates.
_TILE_ROUND_ELEMS = 2048
# RmsNormBwdConfig.smem_stages default; both analytical heuristics use it.
_BWD_SMEM_STAGES = 2


def _smem_budget_bytes(device: torch.device) -> int:
    props = torch.cuda.get_device_properties(device)
    smem = getattr(
        props, "shared_memory_per_block_optin", props.shared_memory_per_block
    )
    return smem - _SMEM_RESERVED_BYTES


def _row_tile_elems(n: int) -> int:
    per_cta = -(-n // _MAX_CLUSTER_N)
    return -(-per_cta // _TILE_ROUND_ELEMS) * _TILE_ROUND_ELEMS


def _fwd_fits_smem(input: torch.Tensor, n: int) -> bool:
    # Fwd smem holds one row tile of x (rmsnorm.py sX).
    return _row_tile_elems(n) * input.element_size() <= _smem_budget_bytes(input.device)


def _bwd_fits_smem(input: torch.Tensor, grad_out: torch.Tensor, n: int) -> bool:
    # Quack's RMSNormBackward.__init__ raises for N > 128K with fp32 x; fall
    # back instead of surfacing its ValueError.
    if input.element_size() >= 4 and n > 128 * 1024:
        return False
    # Bwd smem holds smem_stages buffers of both x and dout (rmsnorm.py
    # sX/sdO).
    tile_bytes = (
        _row_tile_elems(n)
        * _BWD_SMEM_STAGES
        * (input.element_size() + grad_out.element_size())
    )
    return tile_bytes <= _smem_budget_bytes(input.device)


# A contiguous input with a misaligned base pointer must be cloned before the
# kernel can run (see norms._reshape_2d). The clone's extra read+write only
# pays off once quack's bandwidth advantage over aten can absorb it: measured
# on B200 (fwd, bf16/fp32), clone+quack wins 1.8-2.9x at 2^24 elements and
# loses below 2^23. Misaligned inputs smaller than this fall back to aten.
# The threshold is a perf heuristic, not a correctness bound; it has not been
# measured on H100 (SM90), where the crossover may differ. Make it per-arch
# if H100 measurements show a meaningfully different break-even.
_MISALIGNED_MIN_NUMEL = 1 << 24


def _misaligned_clone_unprofitable(t: torch.Tensor, n: int) -> bool:
    # Only contiguous tensors hit the clone path in norms._reshape_2d;
    # non-contiguous ones pay the reshape+contiguous materialization either
    # way, which always lands on an aligned fresh buffer.
    if not t.is_contiguous():
        return False
    if _const_data_ptr(t) % _required_align_bytes(t, n) == 0:
        return False
    return t.numel() < _MISALIGNED_MIN_NUMEL


def _n_yields_valid_cp_size(n: int, dtype: torch.dtype) -> bool:
    # quack picks vecsize = gcd(N, 128 // dtype_bits) and lowers each thread's
    # gmem->smem copy to cp.async, whose PTX cp_size only accepts 32, 64, or
    # 128 bits. Narrow dtypes with an unfriendly N (e.g. odd N for bf16/fp16)
    # produce a 16-bit vector copy that fails CuTe IR verification at compile
    # time; fall through to aten in that case.
    dtype_bits = torch.finfo(dtype).bits
    vecsize = math.gcd(n, 128 // dtype_bits)
    return vecsize * dtype_bits in (32, 64, 128)


def _shape_is_valid(
    input: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor | None,
) -> bool:
    # Mirror aten's _check_layer_norm_inputs: on any mismatch the aten path
    # raises, so we fall through and let it produce the proper error rather
    # than silently running the override on ill-shaped inputs.
    n = len(normalized_shape)
    if n < 1 or input.ndim < n:
        return False
    if list(input.shape[-n:]) != list(normalized_shape):
        return False
    if weight is not None and list(weight.shape) != list(normalized_shape):
        return False
    return True


def _fused_rms_norm_cond(
    input: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor | None,
    eps: float | None,
) -> bool:
    if not _is_supported(input):
        return False
    if not _shape_is_valid(input, normalized_shape, weight):
        return False
    # Weight must match input dtype and device for quack's kernel; mismatches
    # are legal under aten (it casts), so fall through.
    if weight is not None and (
        weight.dtype != input.dtype or weight.device != input.device
    ):
        return False
    # Empty inputs crash quack with cudaErrorInvalidConfiguration -- the bad
    # launch config poisons the CUDA context for subsequent calls. Quack's own
    # rmsnorm_bwd guards against this with `if x.numel() > 0` (rmsnorm.py:1111)
    # but the fwd path doesn't.
    if input.numel() == 0:
        return False
    if not _n_yields_valid_cp_size(math.prod(normalized_shape), input.dtype):
        return False
    if not _fwd_fits_smem(input, math.prod(normalized_shape)):
        return False
    if _misaligned_clone_unprofitable(input, math.prod(normalized_shape)):
        return False
    # Non-contiguous weight would require a reshape+copy that we haven't
    # measured; fall through to aten until we do.
    if weight is not None and not weight.is_contiguous():
        return False
    return True


def _fused_rms_norm_impl(
    input: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor | None,
    eps: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if eps is None:
        # Match aten/src/ATen/native/cuda/layer_norm_kernel.cu:1841-1847:
        # aten picks eps from the *accumulator* dtype, which is float32 for
        # fp16/bf16/fp32 inputs (the only dtypes our cond accepts).
        eps = torch.finfo(torch.float32).eps

    from .norms import quack_rmsnorm_fwd

    return quack_rmsnorm_fwd(input, weight, normalized_shape, eps)


def _fused_rms_norm_backward_cond(
    grad_out: torch.Tensor,
    input: torch.Tensor,
    normalized_shape: list[int],
    rstd: torch.Tensor,
    weight: torch.Tensor | None,
    output_mask: list[bool],
) -> bool:
    if not _is_supported(input):
        return False
    if not _shape_is_valid(input, normalized_shape, weight):
        return False
    if weight is not None and (
        weight.dtype != input.dtype or weight.device != input.device
    ):
        return False
    if input.numel() == 0:
        return False
    if not _n_yields_valid_cp_size(math.prod(normalized_shape), input.dtype):
        return False
    if not _bwd_fits_smem(input, grad_out, math.prod(normalized_shape)):
        return False
    n = math.prod(normalized_shape)
    if _misaligned_clone_unprofitable(input, n) or _misaligned_clone_unprofitable(
        grad_out, n
    ):
        return False
    if weight is not None and not weight.is_contiguous():
        return False
    return True


def _fused_rms_norm_backward_impl(
    grad_out: torch.Tensor,
    input: torch.Tensor,
    normalized_shape: list[int],
    rstd: torch.Tensor,
    weight: torch.Tensor | None,
    output_mask: list[bool],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    from .norms import quack_rmsnorm_bwd

    grad_input, grad_weight = quack_rmsnorm_bwd(
        grad_out,
        input,
        rstd,
        weight,
        normalized_shape,
        dw_mask=output_mask[1],
    )

    if not output_mask[0]:
        grad_input = None
    return grad_input, grad_weight


def register_rmsnorm_overrides() -> None:
    # Don't gate on torch.cuda.is_available() here: it calls cuInit and
    # poisons fork (vLLM EngineCore relies on fork). cu.register_op_override
    # already short-circuits on _cuda.is_built() via _check_runtime_available.
    cu.register_op_override(
        "aten",
        "_fused_rms_norm",
        "CUDA",
        cond=_fused_rms_norm_cond,
        impl=_fused_rms_norm_impl,
    )
    cu.register_op_override(
        "aten",
        "_fused_rms_norm_backward",
        "CUDA",
        cond=_fused_rms_norm_backward_cond,
        impl=_fused_rms_norm_backward_impl,
    )
<|endoftext|>"""CuTeDSL override registrations for ``aten::scatter_add``.

Two kernels for the fast-path layout (self/src 2D-shaped after
coalescing, with self contiguous on the slice axis and src either
contiguous or row-strided), tried in order:

1. **TMA** (``tma_kernel.py``, sm_90+): uses ``cp.reduce.async.bulk`` to
   offload the whole reduction to the TMA unit, 3-7x faster than aten on
   high-contention embedding-bag-grad-style workloads.
2. **vec-scatter** (``vec_scatter_kernel.py``, any SM): warp-per-row
   scheduling with ``atomicAdd`` (scalar fp32, paired x2 halves). Covers
   pre-sm_90, and sm_90+ shapes where the TMA alignment constraint isn't
   met.

Eligibility mirrors aten's ``fast_scatter_add_kernel_eligible``
(IndexKernelUtils.h): we restride ``self`` so its scatter-axis stride
is 0 and shape matches ``index``, then build a ``TensorIterator`` on
``(self_restrided, src_restrided, index)``. TensorIterator's coalescing
and stride-magnitude reordering produces a 2D iterator iff the layout
is kernel-friendly; we then pattern-match the resulting iter strides
to confirm slice/index-axis assignment. This delegates the layout
analysis to ``TensorIterator`` rather than reimplementing it.

Anything else (non-2D-coalescing layout, dtype mismatch, deterministic
mode) falls through to aten. Aten is competitive or better on those
shapes, so we intentionally don't override them.

In-place (``scatter_add_``) is registered explicitly; PyTorch's
structured-delegate from ``scatter_add_`` to ``scatter_add.out`` happens
below the dispatcher, so overriding the ``.out`` variant alone doesn't
intercept the in-place method.
"""

import functools

import torch
from torch._tensor_iterator import TensorIterator

from ... import cutedsl_utils as cu
from ...registry import _OpCondFn, _OpImplFn


_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

# 16B alignment for `src.data_ptr()` and `src.stride(0) * elem`: required
# for the LDG.128 / tile-mode TMA load of `src` and the row-start 16B
# boundary that LDG.128 reads from. Applies to both paths.
_SRC_ALIGN_BYTES = 16
# Path-specific destination alignment. TMA: cp.reduce.async.bulk gmem
# operand needs 16B base + stride. vec-scatter: scalar fp32 atomicAdd /
# f16x2 / bf16x2 paired atomics need natural alignment of the 4B atomic
# operand on both base and per-row stride.
_TMA_DST_ALIGN_BYTES = 16
_VEC_DST_ALIGN_BYTES = 4


@functools.cache
def _is_rocm() -> bool:
    # These kernels are built on the CUTLASS Python DSL (CuteDSL), which is
    # NVIDIA-only and performs a hard CUDA-driver dependency check at compile
    # time. On a ROCm/HIP build ``tensor.is_cuda`` is still True, so without
    # this guard the cond approves and ``cute.compile`` fails with
    # CudaDriverDependencyError. Decline so scatter_add falls back to aten.
    return torch.version.hip is not None


@functools.cache
def _has_sm90_plus() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 9


def _deterministic() -> bool:
    # Our kernels use non-deterministic atomicAdd. Under
    # torch.use_deterministic_algorithms(True), aten's scatter_add falls
    # into _scatter_via_index_put instead; we must decline to preserve
    # that behavior.
    return torch.are_deterministic_algorithms_enabled()


def _any_cow(*tensors: torch.Tensor) -> bool:
    return any(
        torch._C._is_cow_tensor(t)  # pyrefly: ignore[missing-attribute]
        for t in tensors
    )


def _base_cond_ok(*tensors: torch.Tensor) -> bool:
    """Pre-checks shared by every path: env sanity, all CUDA, non-COW."""
    if _is_rocm():
        return False
    if _deterministic():
        return False
    if not all(t.is_cuda for t in tensors):
        return False
    if _any_cow(*tensors):
        return False
    return True


# ---------------------------------------------------------------------------
# TI-driven layout analysis.
#
# Mirrors aten's pattern in ScatterGatherKernel.cu / IndexKernelUtils.h:
# restride ``self`` so its scatter-axis stride is 0 and its shape matches
# ``index``, then build a TensorIterator on the three. TensorIterator does
# the broadcast / coalesce / dim reorder; we then pattern-match the post-
# build iter strides to recognize the kernel's expected layout.
# ---------------------------------------------------------------------------


def _normalize_dim(dim: int, ndim: int) -> int:
    return dim + ndim if dim < 0 else dim


def _scatter_add_eligibility(
    self: torch.Tensor, d: int, index: torch.Tensor, src: torch.Tensor
) -> TensorIterator | None:
    """Return the analysis TensorIterator if (self, d, index, src) fits the
    kernel's expected layout, else ``None``. ``d`` is the already-normalized
    scatter axis.

    Mirrors aten's ``fast_scatter_add_kernel_eligible`` (IndexKernelUtils.h):
    restride ``self`` so its scatter-axis stride is 0 (shape = ``index.shape``),
    let TensorIterator coalesce + reorder, then check that the result is
    a 2D iter where dim 0 is the contiguous slice axis and dim 1 is the
    scatter (index) axis.
    """
    if self.dtype not in _SUPPORTED_DTYPES or self.dtype != src.dtype:
        return None
    if index.dtype not in (torch.int32, torch.int64):
        return None
    if self.ndim != src.ndim or self.ndim != index.ndim or self.ndim == 0:
        return None
    if not 0 <= d < self.ndim:
        return None

    self_strides = list(self.stride())
    self_strides[d] = 0  # aten's restride_dim trick
    try:
        self_r = self.as_strided(index.shape, self_strides)
        src_r = src.as_strided(index.shape, src.stride())
        it = TensorIterator(
            outputs=[self_r],
            const_inputs=[src_r, index],
            check_mem_overlap=False,
            check_all_same_dtype=False,
            resize_outputs=False,
        )
    except RuntimeError:
        return None

    if it.ndim != 2:
        return None
    elem = self.element_size()
    idx_elem = index.element_size()
    s_self, s_src, s_idx = it.strides(0), it.strides(1), it.strides(2)
    if (
        s_idx[0] == 0
        and s_idx[1] == idx_elem
        and s_self[0] == elem
        and s_src[0] == elem
        and s_self[1] == 0
    ):
        return it
    return None


def _alignment_contract_ok(
    dst: torch.Tensor, src: torch.Tensor, dim: int, *, dst_ptr_align: int
) -> bool:
    """Alignment of effective ptrs and outer row strides.

    The src side is always read via 16B-wide vector ops (LDG.128 /
    tile-mode TMA load), so its data_ptr and per-row stride must be
    16B-aligned. The dst side's requirement is path-specific:

      - TMA: ``cp.reduce.async.bulk`` writes the gmem operand via a TMA
        descriptor that requires 16B alignment of both base and stride.
      - vec-scatter: writes via scalar fp32 atomicAdd / f16x2 / bf16x2
        paired atomics, which need natural alignment of the 4B atomic
        operand on both base address and per-row stride.

    The "row stride" is the stride along ``dim`` -- the kernel uses it to
    step between rows of the (M, N) views ``_prepare_kernel_inputs`` builds.
    When this returns False the cond rejects and the call falls through
    to aten, which has its own predicate and safe fallback to
    ``vectorized_scatter_add_kernel_launch`` / ``indexFunc{Small,Large}Index``.
    """
    if src.data_ptr() % _SRC_ALIGN_BYTES:
        return False
    if dst.data_ptr() % dst_ptr_align:
        return False
    elem = dst.element_size()
    if (src.stride(dim) * elem) % _SRC_ALIGN_BYTES:
        return False
    if (dst.stride(dim) * elem) % dst_ptr_align:
        return False
    return True


# ---------------------------------------------------------------------------
# Per-kernel eligibility. ``it.shape[0]`` is the slice extent (N elements)
# after coalescing; that's what the alignment / divisibility checks key on.
# ---------------------------------------------------------------------------


def _is_tma_supported(
    self: torch.Tensor, dim: int, index: torch.Tensor, src: torch.Tensor
) -> bool:
    if not _has_sm90_plus():
        return False
    d = _normalize_dim(dim, self.ndim)
    it = _scatter_add_eligibility(self, d, index, src)
    if it is None:
        return False
    if not _alignment_contract_ok(self, src, d, dst_ptr_align=_TMA_DST_ALIGN_BYTES):
        return False
    # TMA chunk_bytes = min(row_bytes, 512); row_bytes must evenly divide
    # by chunk_bytes and chunk_bytes must be 16-byte aligned.
    from .tma_kernel import row_shape_supported

    return row_shape_supported(self.dtype, it.shape[0])


def _is_vec_scatter_supported(
    self: torch.Tensor, dim: int, index: torch.Tensor, src: torch.Tensor
) -> bool:
    # ``red.global.add.noftz.bf16x2`` requires sm_90+. fp16x2 (sm_70+) and
    # scalar fp32 atomicAdd (sm_60+) work everywhere we care about.
    if self.dtype is torch.bfloat16 and not _has_sm90_plus():
        return False
    d = _normalize_dim(dim, self.ndim)
    it = _scatter_add_eligibility(self, d, index, src)
    if it is None:
        return False
    if not _alignment_contract_ok(self, src, d, dst_ptr_align=_VEC_DST_ALIGN_BYTES):
        return False
    # Each lane owns vec_elems consecutive elements; loop is either
    # fully in-bounds or fully skipped per lane. Only requirement is
    # N % vec_elems == 0 (fp32: %4, halves: %8).
    from .vec_scatter_kernel import vec_elems_for

    return it.shape[0] % vec_elems_for(self.dtype) == 0


# ---------------------------------------------------------------------------
# Kernel-input prep. After eligibility the kernel host functions take 2D
# (M, N) tensors and a 1D index. We synthesize those views from the user's
# original operands using the TI-approved geometry.
# ---------------------------------------------------------------------------


def _prepare_kernel_inputs(
    self: torch.Tensor,
    d: int,
    index: torch.Tensor,
    src: torch.Tensor,
    it: TensorIterator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (self_2d, index_1d, src_2d) for the kernel host functions.

    Caller has verified eligibility via ``_scatter_add_eligibility``.
    ``d`` is the already-normalized scatter axis. ``it.shape[0]`` is the
    coalesced slice extent (product of every non-scatter dim of ``self``).
    The 2D views preserve the user's original stride along ``d`` (used by
    the kernel for the row stride).
    """
    N, M_src = it.shape[0], it.shape[1]
    M_self = self.shape[d]
    self_2d = self.as_strided((M_self, N), (self.stride(d), 1))
    src_2d = src.as_strided((M_src, N), (src.stride(d), 1))
    # Eligibility verified the iter has post-coalesce strides
    # ``s_idx = (0, idx_elem)`` on ``index``; that means ``index`` carries
    # M_src consecutive elements at offset 0 with element stride 1, padded
    # by stride-0 broadcast axes. A single ``as_strided`` collapses that
    # to the 1D view the kernels expect.
    index_1d = index.as_strided((M_src,), (1,))
    if index_1d.dtype != torch.int64:
        index_1d = index_1d.to(torch.int64)
    return self_2d, index_1d, src_2d


# ---------------------------------------------------------------------------
# Cond / impl factories. ``_make_cond`` wraps a support check in the shared
# dispatch boilerplate; ``_make_impls`` produces the (functional, .out,
# in-place) triple for a given kernel.
# ---------------------------------------------------------------------------


def _make_cond(
    support_check,
    *,
    out_dst_ptr_align: int | None = None,
) -> _OpCondFn:
    """Build a cond function that runs ``support_check`` behind the shared
    boilerplate (cutedsl availability, non-deterministic mode, CUDA
    tensors, non-COW, dtype/shape match on ``out`` when present).

    ``out_dst_ptr_align`` is None for the functional / inplace variants.
    For the ``.out`` variant it is the path-specific destination
    alignment (TMA: 16, vec-scatter: 4) used to alignment-check ``out``,
    which the ``.out`` impl runs the kernel on instead of ``self``.

    ``support_check`` is ``_is_tma_supported`` or ``_is_vec_scatter_supported``
    -- both take ``(self, dim, index, src)`` and rebuild the analysis iter
    internally.
    """
    if out_dst_ptr_align is not None:

        def _cond(self, dim, index, src, *, out):
            if not _base_cond_ok(self, index, src, out):
                return False
            if out.dtype != self.dtype or out.shape != self.shape:
                return False
            if out.ndim == 0:
                return False
            # ``out`` is the destination of a functional .out call; it
            # plays the role of self in the iter analysis.
            if not support_check(out, dim, index, src):
                return False
            # The .out impl runs the kernel on ``out``, not ``self`` (see
            # _make_impls.out_impl). support_check above validates ``out``'s
            # layout via TI strides, but we still need to alignment-check it
            # explicitly here in case ``out`` is a distinct buffer with
            # different alignment from ``self``.
            d = _normalize_dim(dim, out.ndim)
            return _alignment_contract_ok(out, src, d, dst_ptr_align=out_dst_ptr_align)

        return _cond

    def _cond(self, dim, index, src, *args, **kwargs):
        if not _base_cond_ok(self, index, src):
            return False
        return support_check(self, dim, index, src)

    return _cond


def _copy_if_distinct(out: torch.Tensor, self: torch.Tensor) -> None:
    """Seed ``out`` with ``self``'s values when they're separate buffers.
    Cond guarantees shape match, so no resize is needed."""
    if out.data_ptr() != self.data_ptr():
        out.copy_(self)


def _make_impls(kernel_getter):
    """Build (functional, out, in-place) impl callables.

    ``kernel_getter`` lazily imports the ``*_scatter_add_into`` kernel
    host function (we defer to keep the DSL runtime out of registration).
    Cond has already verified eligibility; we rebuild the iter inside the
    impl to recover the post-coalesce geometry that ``_prepare_kernel_inputs``
    needs. The build is pure C++ and cheap.
    """

    def _run(dst: torch.Tensor, dim: int, index, src) -> torch.Tensor:
        d = _normalize_dim(dim, dst.ndim)
        it = _scatter_add_eligibility(dst, d, index, src)
        if it is None:
            raise RuntimeError(
                "scatter_add cutedsl: cond approved but iter rebuild failed"
            )
        # Mirrors aten's `if (iter.numel() == 0) return;` in
        # ScatterGatherKernel.cu -- the eligibility check is purely
        # layout-shaped, so empty inputs satisfy it but have no work.
        if it.numel == 0:
            return dst
        dst_2d, index_1d, src_2d = _prepare_kernel_inputs(dst, d, index, src, it)
        kernel_getter()(dst_2d, index_1d, src_2d)
        return dst

    def impl(self, dim, index, src, *args, **kwargs):
        return _run(self.clone(), dim, index, src)

    def out_impl(self, dim, index, src, *, out):
        _copy_if_distinct(out, self)
        return _run(out, dim, index, src)

    def inplace_impl(self, dim, index, src, *args, **kwargs):
        return _run(self, dim, index, src)

    return impl, out_impl, inplace_impl


def _tma_kernel():
    from .tma_kernel import tma_scatter_add_into

    return tma_scatter_add_into


def _vs_kernel():
    from .vec_scatter_kernel import vec_scatter_add_into

    return vec_scatter_add_into


_tma_impl, _tma_out_impl, _tma_inplace_impl = _make_impls(_tma_kernel)
_vs_impl, _vs_out_impl, _vs_inplace_impl = _make_impls(_vs_kernel)

_tma_cond = _make_cond(_is_tma_supported)
_tma_out_cond = _make_cond(_is_tma_supported, out_dst_ptr_align=_TMA_DST_ALIGN_BYTES)
_vs_cond = _make_cond(_is_vec_scatter_supported)
_vs_out_cond = _make_cond(
    _is_vec_scatter_supported, out_dst_ptr_align=_VEC_DST_ALIGN_BYTES
)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


# Dispatch priority is first-registered-wins. TMA's cond is strictly
# narrower than vec-scatter's (TMA only fires on sm_90+ with stricter D
# alignment), so TMA goes first.
_PATHS = (
    # (cond, impl, out_cond, out_impl, inplace_impl)
    (_tma_cond, _tma_impl, _tma_out_cond, _tma_out_impl, _tma_inplace_impl),
    (_vs_cond, _vs_impl, _vs_out_cond, _vs_out_impl, _vs_inplace_impl),
)


def _register_path(
    dispatch_key: str,
    cond: _OpCondFn,
    impl: _OpImplFn,
    out_cond: _OpCondFn,
    out_impl: _OpImplFn,
    inplace_impl: _OpImplFn,
) -> None:
    # scatter_add_ has its own dispatcher entry separate from scatter_add
    # and scatter_add.out; it's structured-delegated to .out below the
    # dispatcher layer, so overriding .out alone doesn't catch it.
    for op_symbol, cond_fn, impl_fn in (
        ("scatter_add", cond, impl),
        ("scatter_add.out", out_cond, out_impl),
        ("scatter_add_", cond, inplace_impl),
    ):
        cu.register_op_override(
            "aten",
            op_symbol,
            dispatch_key,
            cond=cond_fn,
            impl=impl_fn,
            allow_multiple_override=True,
        )


def register_to_dispatch() -> None:
    for path in _PATHS:
        _register_path("CUDA", *path)
<|endoftext|>"""TMA-based scatter_add for the ``index.unsqueeze(-1).expand(-1, N)`` pattern.

Port of the CUDA C++ kernel from https://github.com/pytorch/pytorch/pull/182675
to CuTeDSL. Each CTA (one warp) handles one source row (within an assigned
chunk-index range): a tile-mode TMA bulk load (``cute.copy`` with
``CopyBulkTensorTileG2SOp`` via ``make_tiled_tma_atom``) stages
``src[i, d_start:d_end]`` into smem, then
``cp.reduce.async.bulk.global.shared::cta.bulk_group.add`` deposits the
reduction into ``out[index[i], d_start:d_end]``.

Using the tile-mode TMA with a descriptor over the full ``(M_src, N)``
source tensor lets TMA clamp out-of-range column reads to zero, so the
final partial chunk (when ``N`` isn't a multiple of ``chunk_elems``) is
handled natively -- we just reduce only the valid byte count on the
store side. That widens coverage to any ``row_bytes`` that's 16-byte
aligned (the ``cp.reduce.async.bulk`` gmem operand requirement).

Synchronization between the load and the reduce uses
``cutlass.pipeline.PipelineTmaAsync`` with ``num_stages=2``: producer
``acquire``/``commit`` guards the TMA issue, consumer ``wait``/``release``
guards the reduce. Producer and consumer cooperative groups are both
``Agent.Thread, size=1`` -- the single driver thread inside the warp
does both roles. ``PipelineState.advance()`` handles the stage index +
phase bit.

The pipeline is software-pipelined across the flat sequence of
``(entry, chunk)`` pairs: each loop iteration issues the TMA load for
the current pair and consumes the previous one, keeping one TMA load
