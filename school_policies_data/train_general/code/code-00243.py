        pool (optional): Token (returned by :func:`~torch.cuda.graph_pool_handle` or
            :meth:`other_Graph_instance.pool()<torch.cuda.CUDAGraph.pool>`) or
            :class:`~torch.cuda.MemPool` that hints this graph may share memory
            with the indicated pool.  See :ref:`Graph memory management<graph-memory-management>`.
        enable_annotations (bool, optional): If ``True``, the forward and backward
            captures record kernel annotations from
            :func:`torch.cuda.graph_annotations.mark_kernels` scopes inside the
            callables (backward kernels are tagged via the scopes' autograd node
            hooks). See :mod:`torch.cuda.graph_annotations`. Default: ``False``.

    .. note::
        The ``requires_grad`` state of each Tensor in ``sample_args`` must match the state
        that's expected for the corresponding real input in the training loop.

    .. warning::
        This API is in beta and may change in future releases.

    .. warning::
        ``sample_args`` for each callable must contain only Tensors. Other types are not allowed.

    .. warning::
        Returned callables do not support higher order differentiation (e.g., double backward).

    .. warning::
        In any :class:`~torch.nn.Module` passed to :func:`~make_graphed_callables`, only parameters
        may be trainable. Buffers must have ``requires_grad=False``.

    .. warning::
        After you pass a :class:`torch.nn.Module` through :func:`~make_graphed_callables`,
        you may not add or remove any of that Module's parameters or buffers.

    .. warning::
        :class:`torch.nn.Module`\s passed to :func:`~torch.cuda.make_graphed_callables` must not have module hooks
        registered on them at the time they are passed. However, registering hooks on modules *after* passing them
        through :func:`~torch.cuda.make_graphed_callables` is allowed.

    .. warning::
        When running a graphed callable, you must pass its arguments in the same order and format
        they appeared in that callable's ``sample_args``.

    .. warning::
        The automatic mixed precision is supported in :func:`~torch.cuda.make_graphed_callables` only with disabled
        caching. The context manager `torch.cuda.amp.autocast()` must have `cache_enabled=False`.
    """
    if torch.is_autocast_enabled() and torch.is_autocast_cache_enabled():
        raise RuntimeError(
            "make_graphed_callables does not support the autocast caching. Please set `cache_enabled=False`."
        )

    just_one_callable = False

    _sample_args: tuple[tuple[Tensor, ...], ...]
    if not isinstance(callables, tuple):
        just_one_callable = True
        callables = (callables,)
        _sample_args = (typing.cast(tuple[Tensor, ...], sample_args),)
    else:
        _sample_args = typing.cast(tuple[tuple[Tensor, ...], ...], sample_args)

    flatten_sample_args = []

    for c, args in zip(callables, _sample_args):
        if isinstance(c, torch.nn.Module):
            if not (
                len(c._backward_hooks) == 0
                and len(c._forward_hooks) == 0
                and len(c._forward_pre_hooks) == 0
            ):
                raise AssertionError(
                    "Modules must not have hooks registered at the time they are passed. However, registering hooks "
                    + "on modules after passing them through make_graphed_callables is allowed."
                )
            if not all(b.requires_grad is False for b in c.buffers()):
                raise AssertionError(
                    "In any :class:`~torch.nn.Module` passed to "
                    + ":func:`~make_graphed_callables`, only parameters may be trainable. All buffers must have "
                    + "``requires_grad=False``."
                )
        flatten_arg = torch.utils._pytree.arg_tree_leaves(*args)
        flatten_sample_args.append(tuple(flatten_arg))
        if not all(isinstance(arg, torch.Tensor) for arg in flatten_arg):
            raise AssertionError(
                "In the beta API, sample_args "
                + "for each callable must contain only Tensors. Other types are not allowed."
            )

    # If a callable is an nn.Module, its graph's full input surface is the args the user explicitly
    # passes to forward (ie, its sample_args) AND the module's parameter attributes.
    per_callable_len_user_args = [len(args) for args in flatten_sample_args]
    per_callable_module_params = [
        tuple(c.parameters()) if isinstance(c, torch.nn.Module) else ()
        for c in callables
    ]
    per_callable_static_input_surfaces = [
        flatten_sample_args[i] + per_callable_module_params[i]
        for i in range(len(callables))
    ]

    fwd_graphs = [torch.cuda.CUDAGraph() for _ in range(len(callables))]
    bwd_graphs = [torch.cuda.CUDAGraph() for _ in range(len(callables))]

    mempool = graph_pool_handle() if pool is None else _get_pool_id(pool)

    # Warmup
    # Hopefully prevents cudnn benchmarking and other lazy-initialization cuda work
    # from ending up in any captures.
    torch.cuda.synchronize()
    # hipBLASLt handles are per-(device, stream) on ROCm and lazily created.
    # Need to use the same stream for warmup and capture to avoid capture errors.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for func, args, static_input_surface in zip(
            callables, _sample_args, per_callable_static_input_surfaces
        ):
            grad_inputs, outputs, outputs_grad = None, None, None
            for _ in range(num_warmup_iters):
                outputs = torch.utils._pytree.tree_leaves(func(*args))
                outputs_grad = tuple(o for o in outputs if o.requires_grad)
                if len(outputs_grad) > 0:
                    grad_inputs = torch.autograd.grad(
                        outputs=outputs_grad,
                        inputs=tuple(
                            i for i in static_input_surface if i.requires_grad
                        ),
                        grad_outputs=tuple(
                            torch.empty_like(o) for o in outputs if o.requires_grad
                        ),
                        only_inputs=True,
                        allow_unused=allow_unused_input,
                    )
            for v in [outputs, outputs_grad, grad_inputs]:
                del v

    torch.cuda.synchronize()

    # All captures here share a mempool. To avoid replays corrupting each other's memory,
    # the safest approach is to capture all passes in the same order they'll run:
    # fwd 1, fwd 2, ... fwd N, then bwd N, bwd N-1, ... bwd 1.

    # Capture forward graphs
    per_callable_static_outputs = []
    per_callable_output_unflatten_spec = []
    for func, args, fwd_graph in zip(callables, _sample_args, fwd_graphs):
        with torch.cuda.graph(
            fwd_graph,
            stream=stream,
            pool=mempool,
            capture_error_mode=capture_error_mode,
            enable_annotations=enable_annotations,
        ):
            func_outputs = func(*args)

        flatten_outputs, spec = torch.utils._pytree.tree_flatten(func_outputs)
        per_callable_static_outputs.append(tuple(flatten_outputs))
        per_callable_output_unflatten_spec.append(spec)

    # Capture backward graphs in reverse order
    per_callable_static_grad_outputs = []
    per_callable_static_grad_inputs = []
    for static_input_surface, static_outputs, bwd_graph in zip(
        reversed(per_callable_static_input_surfaces),
        reversed(per_callable_static_outputs),
        reversed(bwd_graphs),
    ):
        # For now, assumes all static_outputs require grad
        # assert all(o.requires_grad for o in static_outputs), "Outputs of graphed callables must require grad."
        static_grad_outputs = tuple(
            torch.empty_like(o) if o.requires_grad else None for o in static_outputs
        )

        outputs_grad = tuple(o for o in static_outputs if o.requires_grad)
        grad_inputs = None
        if len(outputs_grad) > 0:
            with torch.cuda.graph(
                bwd_graph,
                stream=stream,
                pool=mempool,
                capture_error_mode=capture_error_mode,
                enable_annotations=enable_annotations,
            ):
                grad_inputs = torch.autograd.grad(
                    outputs=outputs_grad,
                    inputs=tuple(i for i in static_input_surface if i.requires_grad),
                    grad_outputs=tuple(o for o in static_grad_outputs if o is not None),
                    only_inputs=True,
                    allow_unused=allow_unused_input,
                )

        # Constructs a tuple suitable for returning from Graphed.backward:
        # Pads out the actually-needed grads with Nones in gradient slots for inputs that don't require grad.
        # I couldn't think of a slick one-liner for this pattern.
        static_grad_inputs = []
        grad_idx = 0
        for arg in static_input_surface:
            if arg.requires_grad and grad_inputs is not None:
                static_grad_inputs.append(grad_inputs[grad_idx])
                grad_idx += 1
            else:
                static_grad_inputs.append(None)  # type: ignore[arg-type]
        static_grad_inputs = tuple(static_grad_inputs)  # type: ignore[assignment]

        per_callable_static_grad_outputs.append(static_grad_outputs)
        per_callable_static_grad_inputs.append(static_grad_inputs)

    # Reverses the most recent two lists
    per_callable_static_grad_outputs.reverse()
    per_callable_static_grad_inputs.reverse()
    # Now for every per_callable list, per_callable_*[i] holds the stuff for the ith callable.

    def make_graphed_autograd_function(
        fwd_graph: CUDAGraph,
        bwd_graph: CUDAGraph,
        module_params: tuple[torch.nn.Parameter, ...],
        len_user_args: int,
        output_unflatten_spec: torch.utils._pytree.TreeSpec,
        static_input_surface: tuple[Tensor, ...],
        static_outputs: tuple[Tensor, ...],
        static_grad_outputs: tuple[Tensor | None, ...],
        static_grad_inputs: tuple[Tensor, ...],
    ) -> Callable[..., object]:
        class Graphed(torch.autograd.Function):
            @staticmethod
            # pyrefly: ignore [bad-override]
            def forward(ctx: object, *inputs: Tensor) -> tuple[Tensor, ...]:
                # At this stage, only the user args may (potentially) be new tensors.
                for i in range(len_user_args):
                    if static_input_surface[i].data_ptr() != inputs[i].data_ptr():
                        static_input_surface[i].copy_(inputs[i])
                fwd_graph.replay()
                if not isinstance(static_outputs, tuple):
                    raise AssertionError(
                        f"static_outputs must be tuple, got {type(static_outputs)}"
                    )
                return tuple(o.detach() for o in static_outputs)

            @staticmethod
            @torch.autograd.function.once_differentiable
            # pyrefly: ignore [bad-override]
            def backward(ctx: object, *grads: Tensor) -> tuple[Tensor, ...]:
                if len(grads) != len(static_grad_outputs):
                    raise AssertionError(
                        f"len(grads)={len(grads)} != len(static_grad_outputs)={len(static_grad_outputs)}"
                    )
                for g, grad in zip(static_grad_outputs, grads):
                    if g is not None:
                        # don't copy if autograd gods have been kind and the
                        # incoming grad is already in the right place
                        if g.data_ptr() != grad.data_ptr():
                            g.copy_(grad)
                bwd_graph.replay()

                # Input args that didn't require grad expect a None gradient.
                if not isinstance(static_grad_inputs, tuple):
                    raise AssertionError(
                        f"static_grad_inputs must be tuple, got {type(static_grad_inputs)}"
                    )
                return tuple(
                    # pyrefly: ignore [bad-argument-type]
                    b.detach() if b is not None else b
                    for b in static_grad_inputs
                )

        def functionalized(*user_args: object) -> object:
            # Runs the autograd function with inputs == all inputs to the graph that might require grad
            # (explicit user args + module parameters)
            # Assumes module params didn't change since capture.
            flatten_user_args = torch.utils._pytree.arg_tree_leaves(*user_args)
            out = Graphed.apply(*(tuple(flatten_user_args) + module_params))
            return torch.utils._pytree.tree_unflatten(out, output_unflatten_spec)

        return functionalized

    # Put together the final graphed callables
    ret: list[_ModuleOrCallable] = []
    for i, func in enumerate(callables):
        graphed = make_graphed_autograd_function(
            fwd_graphs[i],
            bwd_graphs[i],
            per_callable_module_params[i],
            per_callable_len_user_args[i],
            per_callable_output_unflatten_spec[i],
            per_callable_static_input_surfaces[i],
            per_callable_static_outputs[i],
            per_callable_static_grad_outputs[i],
            per_callable_static_grad_inputs[i],
        )

        if isinstance(func, torch.nn.Module):

            def make_graphed_forward(
                func: torch.nn.Module,
                graph_training_state: bool,
                graphed: Callable[_P, _R],
                orig_fwd: Callable[_P, _R],
            ) -> Callable[_P, _R]:
                def new_fwd(*user_args: _P.args, **user_kwargs: _P.kwargs) -> _R:
                    # If the module's training-or-eval state matches what we graphed,
                    # run the graph, otherwise run the original forward method
                    if func.training == graph_training_state:
                        return graphed(*user_args, **user_kwargs)
                    else:
                        return orig_fwd(*user_args, **user_kwargs)

                return new_fwd

            func.forward = make_graphed_forward(
                func, func.training, graphed, func.forward
            )
            ret.append(func)
        else:
            ret.append(graphed)

    if just_one_callable:
        return ret[0]

    return tuple(ret)
<|endoftext|># mypy: allow-untyped-defs
import functools
import sys
from typing import Any
from typing_extensions import deprecated

import torch


__all__ = ["autocast", "custom_fwd", "custom_bwd"]


@deprecated(
    "`torch.cuda.amp.autocast(args...)` is deprecated. "
    "Please use `torch.amp.autocast('cuda', args...)` instead.",
    category=FutureWarning,
)
class autocast(torch.amp.autocast_mode.autocast):
    r"""See :class:`torch.autocast`.

    ``torch.cuda.amp.autocast(args...)`` is deprecated. Please use ``torch.amp.autocast("cuda", args...)`` instead.
    """

    # TODO: remove this conditional once we stop supporting Python < 3.13
    # Prior to Python 3.13, inspect.signature could not retrieve the correct
    # signature information for classes decorated with @deprecated (unless
    # the __new__ static method was explicitly defined);
    #
    # However, this issue has been fixed in Python 3.13 and later versions.
    if sys.version_info < (3, 13):

        def __new__(
            cls,
            enabled: bool = True,
            dtype: torch.dtype = torch.float16,
            cache_enabled: bool = True,
        ):
            return super().__new__(cls)

        def __init_subclass__(cls):
            pass

    def __init__(
        self,
        enabled: bool = True,
        dtype: torch.dtype = torch.float16,
        cache_enabled: bool = True,
    ):
        if torch._jit_internal.is_scripting():
            self._enabled = enabled
            self.device = "cuda"
            self.fast_dtype = dtype
            return
        super().__init__(
            "cuda", enabled=enabled, dtype=dtype, cache_enabled=cache_enabled
        )

    def __enter__(self):
        if torch._jit_internal.is_scripting():
            return self
        return super().__enter__()

    # TODO: discuss a unified TorchScript-friendly API for autocast
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):  # type: ignore[override]
        if torch._jit_internal.is_scripting():
            return
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, func):
        if torch._jit_internal.is_scripting():
            return func
        return super().__call__(func)


# Preserved only for BC reasons
@deprecated(
    "`torch.cuda.amp.autocast_mode._cast(value, dtype)` is deprecated. "
    "Please use `torch.amp.autocast_mode._cast(value, 'cuda', dtype)` instead.",
    category=FutureWarning,
)
def _cast(value, dtype):
    return torch.amp.autocast_mode._cast(value, "cuda", dtype)


@deprecated(
    "`torch.cuda.amp.custom_fwd(args...)` is deprecated. "
    "Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.",
    category=FutureWarning,
)
def custom_fwd(fwd=None, *, cast_inputs=None):
    """
    ``torch.cuda.amp.custom_fwd(args...)`` is deprecated. Please use
    ``torch.amp.custom_fwd(args..., device_type='cuda')`` instead.
    """
    return functools.partial(torch.amp.custom_fwd, device_type="cuda")(
        fwd=fwd, cast_inputs=cast_inputs
    )


@deprecated(
    "`torch.cuda.amp.custom_bwd(args...)` is deprecated. "
    "Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.",
    category=FutureWarning,
)
def custom_bwd(bwd):
    """
    ``torch.cuda.amp.custom_bwd(args...)`` is deprecated. Please use
    ``torch.amp.custom_bwd(args..., device_type='cuda')`` instead.
    """
    return functools.partial(torch.amp.custom_bwd, device_type="cuda")(bwd)
<|endoftext|># mypy: allow-untyped-defs
from importlib.util import find_spec

import torch


__all__ = ["amp_definitely_not_available"]


def amp_definitely_not_available():
    return not (torch.cuda.is_available() or find_spec("torch_xla"))
<|endoftext|>from typing_extensions import deprecated

import torch

# We need to keep this unused import for BC reasons
from torch.amp.grad_scaler import OptState  # noqa: F401


__all__ = ["GradScaler"]


class GradScaler(torch.amp.GradScaler):
    r"""
    See :class:`torch.amp.GradScaler`.
    ``torch.cuda.amp.GradScaler(args...)`` is deprecated. Please use ``torch.amp.GradScaler("cuda", args...)`` instead.
    """

    @deprecated(
        "`torch.cuda.amp.GradScaler(args...)` is deprecated. "
        "Please use `torch.amp.GradScaler('cuda', args...)` instead.",
        category=FutureWarning,
    )
    def __init__(
        self,
        init_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ) -> None:
        super().__init__(
            "cuda",
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            enabled=enabled,
        )
<|endoftext|># pyrefly: ignore [deprecated]
from .autocast_mode import autocast, custom_bwd, custom_fwd
from .common import amp_definitely_not_available
from .grad_scaler import GradScaler


__all__ = [
    "amp_definitely_not_available",
    "autocast",
    "custom_bwd",
    "custom_fwd",
    "GradScaler",
]
<|endoftext|>import contextlib
from collections.abc import Iterator
from typing import Literal

import torch


__all__ = [
    "start",
    "stop",
    "profile",
    "metal_capture",
    "is_metal_capture_enabled",
    "is_capturing_metal",
]


ProfilerMode = Literal["interval", "event", "interval,event"]


def start(mode: ProfilerMode = "interval", wait_until_completed: bool = False) -> None:
    r"""Start OS Signpost tracing from MPS backend.

    The generated OS Signposts could be recorded and viewed in
    XCode Instruments Logging tool.

    Args:
        mode(str): OS Signpost tracing mode could be "interval", "event",
            or both "interval,event".
            The interval mode traces the duration of execution of the operations,
            whereas event mode marks the completion of executions.
            See document `Recording Performance Data`_ for more info.
        wait_until_completed(bool): Waits until the MPS Stream complete
            executing each encoded GPU operation. This helps generating single
            dispatches on the trace's timeline.
            Note that enabling this option would affect the performance negatively.

    .. _Recording Performance Data:
       https://developer.apple.com/documentation/os/logging/recording_performance_data
    """
    mode_normalized = mode.lower().replace(" ", "")
    torch._C._mps_profilerStartTrace(  # type: ignore[attr-defined]
        mode_normalized, wait_until_completed
    )


def stop() -> None:
    r"""Stops generating OS Signpost tracing from MPS backend."""
    torch._C._mps_profilerStopTrace()  # type: ignore[attr-defined]


@contextlib.contextmanager
def profile(
    mode: ProfilerMode = "interval", wait_until_completed: bool = False
) -> Iterator[None]:
    r"""Context Manager to enabling generating OS Signpost tracing from MPS backend.

    Args:
        mode(str): OS Signpost tracing mode could be "interval", "event",
            or both "interval,event".
            The interval mode traces the duration of execution of the operations,
            whereas event mode marks the completion of executions.
            See document `Recording Performance Data`_ for more info.
        wait_until_completed(bool): Waits until the MPS Stream complete
            executing each encoded GPU operation. This helps generating single
            dispatches on the trace's timeline.
            Note that enabling this option would affect the performance negatively.

    .. _Recording Performance Data:
       https://developer.apple.com/documentation/os/logging/recording_performance_data
    """
    try:
        start(mode, wait_until_completed)
        yield
    finally:
        stop()


def is_metal_capture_enabled() -> bool:
    """Checks if `metal_capture` context manager is usable
    To enable metal capture, set MTL_CAPTURE_ENABLED envvar
    """
    return torch._C._mps_isCaptureEnabled()  # type: ignore[attr-defined, no-any-return]


def is_capturing_metal() -> bool:
    """Checks if metal capture is in progress"""
    return torch._C._mps_isCapturing()  # type: ignore[attr-defined, no-any-return]


@contextlib.contextmanager
def metal_capture(fname: str) -> Iterator[None]:
    """Context manager that enables capturing of Metal calls into gputrace"""
    try:
        torch._C._mps_startCapture(fname)  # type: ignore[attr-defined]
        yield
    finally:
        # _mps_stopCapture waits for work enqueued on MPS streams before
        # stopping the capture, even if the body raised
        torch._C._mps_stopCapture()  # type: ignore[attr-defined]
<|endoftext|># mypy: allow-untyped-defs
r"""
This package enables an interface for accessing MPS (Metal Performance Shaders) backend in Python.
Metal is Apple's API for programming metal GPU (graphics processor unit). Using MPS means that increased
performance can be achieved, by running work on the metal GPU(s).
See https://developer.apple.com/documentation/metalperformanceshaders for more details.
"""

import torch
from torch import Tensor


_is_in_bad_fork = getattr(torch._C, "_mps_is_in_bad_fork", lambda: False)
_default_mps_generator: torch._C.Generator = None  # type: ignore[assignment]


# local helper function (not public or exported)
def _get_default_mps_generator() -> torch._C.Generator:
    global _default_mps_generator
    if _default_mps_generator is None:
        _default_mps_generator = torch._C._mps_get_default_generator()
    return _default_mps_generator


def device_count() -> int:
    r"""Returns the number of available MPS devices."""
    return int(torch._C._has_mps and torch._C._mps_is_available())


def synchronize() -> None:
    r"""Waits for all kernels in all streams on a MPS device to complete."""
    return torch._C._mps_deviceSynchronize()


def get_rng_state(device: int | str | torch.device = "mps") -> Tensor:
    r"""Returns the random number generator state as a ByteTensor.

    Args:
        device (torch.device or int, optional): The device to return the RNG state of.
            Default: ``'mps'`` (i.e., ``torch.device('mps')``, the current MPS device).
    """
    return _get_default_mps_generator().get_state()


def set_rng_state(new_state: Tensor, device: int | str | torch.device = "mps") -> None:
    r"""Sets the random number generator state.

    Args:
        new_state (torch.ByteTensor): The desired state
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'mps'`` (i.e., ``torch.device('mps')``, the current MPS device).
    """
    new_state_copy = new_state.clone(memory_format=torch.contiguous_format)
    _get_default_mps_generator().set_state(new_state_copy)


def manual_seed(seed: int) -> None:
    r"""Sets the seed for generating random numbers.

    Args:
        seed (int): The desired seed.
    """
    # the torch.mps.manual_seed() can be called from the global
    # torch.manual_seed() in torch/random.py. So we need to make
    # sure mps is available (otherwise we just return without
    # erroring out)
    if not torch._C._has_mps:
        return
    seed = int(seed)
    _get_default_mps_generator().manual_seed(seed)


def seed() -> None:
    r"""Sets the seed for generating random numbers to a random number."""
    _get_default_mps_generator().seed()


def empty_cache() -> None:
    r"""Releases all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other GPU applications.
    """
    torch._C._mps_emptyCache()


def set_per_process_memory_fraction(fraction) -> None:
    r"""Set memory fraction for limiting process's memory allocation on MPS device.
    The allowed value equals the fraction multiplied by recommended maximum device memory
    (obtained from Metal API device.recommendedMaxWorkingSetSize).
    If trying to allocate more than the allowed value in a process, it will raise an out of
    memory error in allocator.

    Args:
        fraction(float): Range: 0~2. Allowed memory equals total_memory * fraction.

    .. note::
       Passing 0 to fraction means unlimited allocations
       (may cause system failure if out of memory).
       Passing fraction greater than 1.0 allows limits beyond the value
       returned from device.recommendedMaxWorkingSetSize.
    """

    if not isinstance(fraction, float):
        raise TypeError("Invalid type for fraction argument, must be `float`")
    if fraction < 0 or fraction > 2:
        raise ValueError(f"Invalid fraction value: {fraction}. Allowed range: 0~2")

    torch._C._mps_setMemoryFraction(fraction)


def current_allocated_memory() -> int:
    r"""Returns the current GPU memory occupied by tensors in bytes.

    .. note::
       The returned size does not include cached allocations in
       memory pools of MPSAllocator.
    """
    return torch._C._mps_currentAllocatedMemory()


def driver_allocated_memory() -> int:
    r"""Returns total GPU memory allocated by Metal driver for the process in bytes.

    .. note::
       The returned size includes cached allocations in MPSAllocator pools
       as well as allocations from MPS/MPSGraph frameworks.
    """
    return torch._C._mps_driverAllocatedMemory()


def recommended_max_memory() -> int:
    r"""Returns recommended max Working set size for GPU memory in bytes.

    .. note::
       Recommended max working set size for Metal.
       returned from device.recommendedMaxWorkingSetSize.
    """
    return torch._C._mps_recommendedMaxMemory()


def compile_shader(source: str):
    r"""Compiles compute shader from source and allows one to invoke kernels
    defined there from the comfort of Python runtime
    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_MPS)
        >>> lib = torch.mps.compile_shader(
        ... "kernel void full(device float* out, constant float& val, uint idx [[thread_position_in_grid]]) { out[idx] = val; }"
        ...  )
        >>> x = torch.zeros(16, device="mps")
        >>> lib.full(x, 3.14)
    """
    from pathlib import Path

    from torch.utils._cpp_embed_headers import _embed_headers

    if not hasattr(torch._C, "_mps_compileShader"):
        raise RuntimeError("MPS is not available")
    source = _embed_headers(
        [l + "\n" for l in source.split("\n")],
        [Path(__file__).parent.parent / "include"],
        set(),
    )
    return torch._C._mps_compileShader(source)


def load_metallib(source):
    r"""Loads a precompiled Metal library (.metallib) and returns a shader
    library object that allows invoking kernels defined in it.

    Args:
        source: Either raw metallib bytes (``bytes``/``bytearray``) or a
            filesystem path (``str``/``os.PathLike``) to a ``.metallib`` file.

    This is useful for loading Metal libraries compiled ahead of time or
    generated by external tools (e.g. Triton, MetalASM).

    Example::

        >>> # xdoctest: +SKIP("requires external .metallib file")
        >>> lib = torch.mps.load_metallib("kernels.metallib")
        >>> x = torch.ones(16, device="mps")
        >>> lib.square(x)
    """
    import os

    if isinstance(source, (bytes, bytearray)):
        if not hasattr(torch._C, "_mps_loadMetalllib"):
            raise RuntimeError("MPS is not available")
        return torch._C._mps_loadMetalllib(bytes(source))
    elif isinstance(source, (str, os.PathLike)):
        if not hasattr(torch._C, "_mps_loadMetallibFromPath"):
            raise RuntimeError("MPS is not available")
        return torch._C._mps_loadMetallibFromPath(str(source))
    else:
        raise TypeError(f"expected bytes or path, got {type(source).__name__}")


def is_available() -> bool:
    return device_count() > 0


def _host_alias_storage(storage: "torch.UntypedStorage") -> "torch.UntypedStorage":
    r"""Returns a CPU :class:`torch.UntypedStorage` that aliases the
    host-visible contents of the MTLBuffer backing ``storage``.

    The returned storage shares memory with ``storage``: writes through the
    CPU alias land directly in the MPS-allocated MTLBuffer, avoiding a
    CPU->MPS staging copy. This is intended for advanced interop with bulk
    loaders (e.g. safetensors) that already know how to write into CPU
    memory.

    The alias storage retains a reference to the source MPS storage, so the
    host pointer remains valid for the alias's lifetime even if the original
    tensor is freed.

    Raises an exception if ``storage`` is not backed by a shared-storage
    ``id<MTLBuffer>`` allocated by the MPS allocator.

    .. warning::
        Use with caution. This bypasses the cache-coherence guarantees that
        the higher-level PyTorch APIs (:meth:`torch.Tensor.cpu`,
        :meth:`torch.Tensor.to`, ``copy_``) provide for you, and makes the
        caller responsible for ordering CPU and GPU accesses to the same
        memory. You should **always** call :func:`torch.mps.synchronize`
        both **before** issuing host reads/writes through the alias (to
        drain any in-flight GPU work that may still be touching the
        buffer) and **after** (before launching GPU work that depends on
        the host writes). Failure to do so can produce stale reads,
        torn writes, or data corruption.
    """
    return torch._C._mps_host_alias_storage(storage)


from . import profiler
from .event import Event


__all__ = [
    "compile_shader",
    "load_metallib",
    "device_count",
    "get_rng_state",
    "manual_seed",
    "seed",
    "set_rng_state",
    "synchronize",
    "empty_cache",
    "set_per_process_memory_fraction",
    "current_allocated_memory",
    "driver_allocated_memory",
    "Event",
    "profiler",
    "recommended_max_memory",
    "is_available",
]
<|endoftext|>import torch


class Event:
    r"""Wrapper around an MPS event.

    MPS events are synchronization markers that can be used to monitor the
    device's progress, to accurately measure timing, and to synchronize MPS streams.

    Args:
        enable_timing (bool, optional): indicates if the event should measure time
            (default: ``False``)
    """

    def __init__(self, enable_timing: bool = False) -> None:
        self.__eventId = torch._C._mps_acquireEvent(enable_timing)

    def __del__(self) -> None:
        # checks if torch._C is already destroyed
        if hasattr(torch._C, "_mps_releaseEvent") and self.__eventId > 0:
            torch._C._mps_releaseEvent(self.__eventId)

    def record(self) -> None:
        r"""Records the event in the default stream."""
        torch._C._mps_recordEvent(self.__eventId)

    def wait(self) -> None:
        r"""Makes all future work submitted to the default stream wait for this event."""
        torch._C._mps_waitForEvent(self.__eventId)

    def query(self) -> bool:
        r"""Returns True if all work currently captured by event has completed."""
        return torch._C._mps_queryEvent(self.__eventId)

    def synchronize(self) -> None:
        r"""Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.
        """
        torch._C._mps_synchronizeEvent(self.__eventId)

    def elapsed_time(self, end_event: "Event") -> float:
        r"""Returns the time elapsed in milliseconds after the event was
        recorded and before the end_event was recorded.
        """
        return torch._C._mps_elapsedTimeOfEvents(self.__eventId, end_event.__eventId)
<|endoftext|>"""ONNX exporter exceptions."""

from __future__ import annotations


__all__ = [
    "OnnxExporterWarning",
    "SymbolicValueError",
    "UnsupportedOperatorError",
]

import textwrap
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from torch import _C


class OnnxExporterWarning(UserWarning):
    """Warnings in the ONNX exporter."""


class OnnxExporterError(RuntimeError):
    """Errors raised by the ONNX exporter. This is the base class for all exporter errors."""


class UnsupportedOperatorError(OnnxExporterError):
    """Raised when an operator is unsupported by the exporter."""

    # NOTE: This is legacy and is only used by the torchscript exporter
    # Clean up when the torchscript exporter is removed
    def __init__(self, name: str, version: int, supported_version: int | None) -> None:
        if supported_version is not None:
            msg = (
                f"Exporting the operator '{name}' to ONNX opset version {version} "
                "is not supported. Support for this operator was added in version "
                f"{supported_version}, try exporting with this version"
            )
        elif name.startswith(("aten::", "prim::", "quantized::")):
            msg = (
                f"Exporting the operator '{name}' to ONNX opset version {version} "
                "is not supported"
            )
        else:
            msg = (
                f"ONNX export failed on an operator with unrecognized namespace {name}. "
                "If you are trying to export a custom operator, make sure you registered it with "
                "the right domain and version."
            )

        super().__init__(msg)


class SymbolicValueError(OnnxExporterError):
    """Errors around TorchScript values and nodes."""

    # NOTE: This is legacy and is only used by the torchscript exporter
    # Clean up when the torchscript exporter is removed
    def __init__(self, msg: str, value: _C.Value) -> None:
        message = (
            f"{msg}  [Caused by the value '{value}' (type '{value.type()}') in the "
            f"TorchScript graph. The containing node has kind '{value.node().kind()}'.] "
        )

        code_location = value.node().sourceRange()
        if code_location:
            message += f"\n    (node defined in {code_location})"

        try:
            # Add its input and output to the message.
            message += "\n\n"
            message += textwrap.indent(
                (
                    "Inputs:\n"
                    + (
                        "\n".join(
                            f"    #{i}: {input_}  (type '{input_.type()}')"
                            for i, input_ in enumerate(value.node().inputs())
                        )
                        or "    Empty"
                    )
                    + "\n"
                    + "Outputs:\n"
                    + (
                        "\n".join(
                            f"    #{i}: {output}  (type '{output.type()}')"
                            for i, output in enumerate(value.node().outputs())
                        )
                        or "    Empty"
                    )
                ),
                "    ",
            )
        except AttributeError:
            message += (
                " Failed to obtain its input and output for debugging. "
                "Please refer to the TorchScript graph for debugging information."
            )

        super().__init__(message)
<|endoftext|>"""Backward compatibility module for torch.onnx.symbolic_opset10."""

from __future__ import annotations


__all__: list[str] = []

from torch.onnx._internal.torchscript_exporter.symbolic_opset10 import *  # noqa: F403
from torch.onnx._internal.torchscript_exporter.symbolic_opset10 import (  # noqa: F401
    _slice,
)
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations


__all__ = [
    # Modules
    "errors",
    "ops",
    # Public functions
    "export",
    "is_in_onnx_export",
    # Base error
    "OnnxExporterError",
    "ONNXProgram",
    "ExportableModule",
    "InputObserver",
]

from typing import Any, TYPE_CHECKING

import torch
from torch._C import _onnx as _C_onnx
from torch._C._onnx import (  # Deprecated members that are excluded from __all__
    OperatorExportTypes as OperatorExportTypes,
    TensorProtoDataType as TensorProtoDataType,
    TrainingMode as TrainingMode,
)

from . import errors, ops
from ._internal.exporter._exportable_module import ExportableModule
from ._internal.exporter._input_observer import InputObserver
from ._internal.exporter._onnx_program import ONNXProgram
from ._internal.torchscript_exporter import (  # Deprecated members that are excluded from __all__
    symbolic_helper,
    symbolic_opset10,
    symbolic_opset9,
    utils,
)
from ._internal.torchscript_exporter._type_utils import (
    JitScalarType,  # Deprecated members that are excluded from __all__
)
from ._internal.torchscript_exporter.utils import (  # Deprecated members that are excluded from __all__
    register_custom_op_symbolic,
    select_model_mode_for_export,  # pyrefly: ignore  # deprecated
    unregister_custom_op_symbolic,
)
from .errors import OnnxExporterError


if TYPE_CHECKING:
    import os
    from collections.abc import Callable, Collection, Mapping, Sequence

# Set namespace for exposed private names
ONNXProgram.__module__ = "torch.onnx"
ExportableModule.__module__ = "torch.onnx"
OnnxExporterError.__module__ = "torch.onnx"
InputObserver.__module__ = "torch.onnx"

# TODO(justinchuby): Remove these two properties
producer_name = "pytorch"
producer_version = _C_onnx.PRODUCER_VERSION


def export(
    model: torch.nn.Module
    | torch.export.ExportedProgram
    | torch.jit.ScriptModule
    | torch.jit.ScriptFunction,
    args: tuple[Any, ...] = (),
    f: str | os.PathLike | None = None,
    *,
    kwargs: dict[str, Any] | None = None,
    verbose: bool | None = None,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    opset_version: int | None = None,
    dynamo: bool = True,
    # Dynamo only options
    external_data: bool = True,
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None = None,
    custom_translation_table: dict[Callable, Callable] | None = None,
    report: bool = False,
    optimize: bool = True,
    verify: bool = False,
    profile: bool = False,
    dump_exported_program: bool = False,
    artifacts_dir: str | os.PathLike = ".",
    # BC options
    export_params: bool = True,
    keep_initializers_as_inputs: bool = False,
    dynamic_axes: Mapping[str, Mapping[int, str]]
    | Mapping[str, Sequence[int]]
    | None = None,
    # Deprecated options
    training: _C_onnx.TrainingMode = _C_onnx.TrainingMode.EVAL,
    operator_export_type: _C_onnx.OperatorExportTypes = _C_onnx.OperatorExportTypes.ONNX,
    do_constant_folding: bool = True,
    custom_opsets: Mapping[str, int] | None = None,
    export_modules_as_functions: bool | Collection[type[torch.nn.Module]] = False,
    autograd_inlining: bool = True,
) -> ONNXProgram | None:
    r"""Exports a model into ONNX format.

    Setting ``dynamo=True`` enables the new ONNX export logic
    which is based on :class:`torch.export.ExportedProgram` and a more modern
    set of translation logic. This is the recommended and default way to export models
    to ONNX.

    When ``dynamo=True``:

    The exporter tries the following strategies to get an ExportedProgram for conversion to ONNX.

    #. If the model is already an ExportedProgram, it will be used as-is.
    #. Use :func:`torch.export.export` and set ``strict=False``.
    #. Use :func:`torch.export.export` and set ``strict=True``.

    Args:
        model: The model to be exported.
        args: Example positional inputs. Any non-Tensor arguments will be hard-coded into the
            exported model; any Tensor arguments will become inputs of the exported model,
            in the order they occur in the tuple.
        f: Path to the output ONNX model file. E.g. "model.onnx". This argument is kept for
            backward compatibility. It is recommended to leave unspecified (None)
            and use the returned :class:`torch.onnx.ONNXProgram` to serialize the model
            to a file instead.
        kwargs: Optional example keyword inputs.
        verbose: Whether to enable verbose logging.
        input_names: names to assign to the input nodes of the graph, in order.
        output_names: names to assign to the output nodes of the graph, in order.
            These are labels only and do not affect the order of outputs. If the model
            returns a dictionary, outputs are flattened in the dictionary's iteration
            order regardless of the names specified here.
        opset_version: The version of the
            `default (ai.onnx) opset <https://github.com/onnx/onnx/blob/master/docs/Operators.md>`_
            to target. You should set ``opset_version`` according to the supported opset versions
            of the runtime backend or compiler you want to run the exported model with.
            Leave as default (``None``) to use the recommended version, or refer to
            the ONNX operators documentation for more information.
        dynamo: Whether to export the model with ``torch.export`` ExportedProgram instead of TorchScript.
        external_data: Whether to save the model weights as an external data file.
            This is required for models with large weights that exceed the ONNX file size limit (2GB).
            When False, the weights are saved in the ONNX file with the model architecture.
        dynamic_shapes: A dictionary or a tuple of dynamic shapes for the model inputs. Refer to
            :func:`torch.export.export` for more details. This is only used (and preferred) when dynamo is True.
            Note that dynamic_shapes is designed to be used when the model is exported with dynamo=True, while
            dynamic_axes is used when dynamo=False.
        custom_translation_table: A dictionary of custom decompositions for operators in the model.
            The dictionary should have the callable target in the fx Node as the key (e.g. ``torch.ops.aten.stft.default``),
            and the value should be a function that builds that graph using ONNX Script. This option
            is only valid when dynamo is True.
        report: Whether to generate a markdown report for the export process. This option
            is only valid when dynamo is True.
        optimize: Whether to optimize the exported model. This option
            is only valid when dynamo is True. Default is True.
        verify: Whether to verify the exported model using ONNX Runtime. This option
            is only valid when dynamo is True.
        profile: Whether to profile the export process. This option
            is only valid when dynamo is True.
        dump_exported_program: Whether to dump the :class:`torch.export.ExportedProgram` to a file.
            This is useful for debugging the exporter. This option is only valid when dynamo is True.
        artifacts_dir: The directory to save the debugging artifacts like the report and the serialized
            exported program. This option is only valid when dynamo is True.
        export_params: **When ``f`` is specified**: If false, parameters (weights) will not be exported.

            You can also leave it unspecified and use the returned :class:`torch.onnx.ONNXProgram`
            to control how initializers are treated when serializing the model.
        keep_initializers_as_inputs: **When ``f`` is specified**: If True, all the
            initializers (typically corresponding to model weights) in the
            exported graph will also be added as inputs to the graph. If False,
            then initializers are not added as inputs to the graph, and only
            the user inputs are added as inputs.

            Set this to True if you intend to supply model weights at runtime.
            Set it to False if the weights are static to allow for better optimizations
            (e.g. constant folding) by backends/runtimes.

            You can also leave it unspecified and use the returned :class:`torch.onnx.ONNXProgram`
            to control how initializers are treated when serializing the model.
        dynamic_axes:
            Deprecated: Prefer specifying ``dynamic_shapes`` when ``dynamo=True``.

            By default the exported model will have the shapes of all input and output tensors
            set to exactly match those given in ``args``. To specify axes of tensors as
            dynamic (i.e. known only at run-time), set ``dynamic_axes`` to a dict with schema:

            * KEY (str): an input or output name. Each name must also be provided in ``input_names`` or
                ``output_names``.
            * VALUE (dict or list): If a dict, keys are axis indices and values are axis names. If a
                list, each element is an axis index.

            For example::

                class SumModule(torch.nn.Module):
                    def forward(self, x):
                        return torch.sum(x, dim=1)


                torch.onnx.export(
                    SumModule(),
                    (torch.ones(2, 2),),
                    "onnx.pb",
                    input_names=["x"],
                    output_names=["sum"],
                )

            Produces::

                input {
                  name: "x"
                  ...
                      shape {
                        dim {
                          dim_value: 2  # axis 0
                        }
                        dim {
                          dim_value: 2  # axis 1
                ...
                output {
                  name: "sum"
                  ...
                      shape {
                        dim {
                          dim_value: 2  # axis 0
                ...

            While::

                torch.onnx.export(
                    SumModule(),
                    (torch.ones(2, 2),),
                    "onnx.pb",
                    input_names=["x"],
                    output_names=["sum"],
                    dynamic_axes={
                        # dict value: manually named axes
                        "x": {0: "my_custom_axis_name"},
                        # list value: automatic names
                        "sum": [0],
                    },
                )

            Produces::

                input {
                  name: "x"
                  ...
                      shape {
                        dim {
                          dim_param: "my_custom_axis_name"  # axis 0
                        }
                        dim {
                          dim_value: 2  # axis 1
                ...
                output {
                  name: "sum"
                  ...
                      shape {
                        dim {
                          dim_param: "sum_dynamic_axes_1"  # axis 0
                ...

        training: Deprecated option. Instead, set the training mode of the model before exporting.
        operator_export_type: Deprecated option. Only ONNX is supported.
        do_constant_folding: Deprecated option.
        custom_opsets: Deprecated option.
        export_modules_as_functions: Deprecated option.
        autograd_inlining: Deprecated option.

    Returns:
        :class:`torch.onnx.ONNXProgram` if dynamo is True, otherwise None.

    .. versionchanged:: 2.6
        ``training`` is now deprecated. Instead, set the training mode of the model before exporting.
        ``operator_export_type`` is now deprecated. Only ONNX is supported.
        ``do_constant_folding`` is now deprecated. It is always enabled.
        ``export_modules_as_functions`` is now deprecated.
        ``autograd_inlining`` is now deprecated.
    .. versionchanged:: 2.7
        ``optimize`` is now True by default.
    .. versionchanged:: 2.9
        ``dynamo`` is now True by default.
    .. versionchanged:: 2.11
        ``fallback`` option has been removed.
    """
    if dynamo is True or isinstance(
        model, (torch.export.ExportedProgram, ExportableModule)
    ):
        from torch.onnx._internal.exporter import _compat

        if isinstance(args, torch.Tensor):
            args = (args,)

        return _compat.export_compat(
            model,
            args,
            f,
            kwargs=kwargs,
            export_params=export_params,
            verbose=verbose,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            custom_translation_table=custom_translation_table,
            dynamic_axes=dynamic_axes,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
            external_data=external_data,
            dynamic_shapes=dynamic_shapes,
            report=report,
            optimize=optimize,
            verify=verify,
            profile=profile,
            dump_exported_program=dump_exported_program,
            artifacts_dir=artifacts_dir,
        )
    else:
        import warnings

        from ._internal.torchscript_exporter.utils import export

        warnings.warn(
            "You are using the legacy TorchScript-based ONNX export. Starting in PyTorch 2.9, "
            "the new torch.export-based ONNX exporter has become the default. "
            "Learn more about the new export logic: https://docs.pytorch.org/docs/stable/onnx_export.html. "
            "For exporting control flow: "
            "https://pytorch.org/tutorials/beginner/onnx/export_control_flow_model_to_onnx_tutorial.html",
            category=DeprecationWarning,
            stacklevel=2,
        )

        if dynamic_shapes:
            raise ValueError(
                "The exporter only supports dynamic shapes "
                "through parameter dynamic_axes when dynamo=False."
            )

        export(
            model,
            args,
            f,  # type: ignore[arg-type]
            kwargs=kwargs,
            export_params=export_params,
            verbose=verbose is True,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
            training=training,
            operator_export_type=operator_export_type,
            do_constant_folding=do_constant_folding,
            custom_opsets=custom_opsets,
            export_modules_as_functions=export_modules_as_functions,
            autograd_inlining=autograd_inlining,
        )
        return None


def is_in_onnx_export() -> bool:
    """Returns whether it is in the middle of ONNX export."""
    from torch.onnx._internal.exporter import _flags
    from torch.onnx._internal.torchscript_exporter._globals import GLOBALS

    return GLOBALS.in_onnx_export or _flags._is_onnx_exporting
<|endoftext|>"""Backward compatibility module for torch.onnx.symbolic_opset9."""

from __future__ import annotations


__all__: list[str] = []

from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import *  # noqa: F403
from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import (  # noqa: F401
    _prepare_onnx_paddings,
    _reshape_from_tensor,
    _slice,
    _var_mean,
)
<|endoftext|>"""Internal feature flags for torch.onnx.

NOTE: These flags are experimental only. Any flag here can be removed at any
time without notice.
"""

import logging
import os


logger = logging.getLogger(__name__)


def _load_boolean_flag(
    name: str,
    *,
    this_will: str,
    deprecated: bool = False,
    default: bool = False,
) -> bool:
    """Load a boolean flag from environment variable.

    Args:
        name: The name of the environment variable.
        this_will: A string that describes what this flag will do.
        deprecated: Whether this flag is deprecated.
        default: The default value if envvar not defined.
    """
    undefined = os.getenv(name) is None
    state = os.getenv(name) == "1"
    if state:
        if deprecated:
            logger.error(
                "Experimental flag %s is deprecated. Please remove it from your environment.",
                name,
            )
        else:
            logger.warning(
                "Experimental flag %s is enabled. This will %s.", name, this_will
            )
    if undefined:
        state = default
    return state


ENABLE_DRAFT_EXPORT: bool = _load_boolean_flag(
    "TORCH_ONNX_ENABLE_DRAFT_EXPORT",
    this_will="enable torch.export.draft_export as a strategy for capturing models",
    default=False,
)
PREFER_DEFERRED_RUNTIME_ASSERTS_OVER_GUARDS: bool = _load_boolean_flag(
    "TORCH_ONNX_PREFER_DEFERRED_RUNTIME_ASSERTS_OVER_GUARDS",
    this_will="set prefer_deferred_runtime_asserts_over_guards when calling torch.export",
    default=True,
)
<|endoftext|>"""Implementations of ONNX operators as native Torch ops.

NOTE: Fake implementations:
    Refer to https://docs.pytorch.org/docs/stable/library.html#torch.library.register_fake
    for more details on how to create fake kernels.
"""

import math
from collections.abc import Callable
from typing import TypeVar
from typing_extensions import ParamSpec

import torch
from torch.onnx.ops import _dtype_mappings


# Use ParamSpec for better type preservation instead of bound Callable TypeVar
_P = ParamSpec("_P")
_R = TypeVar("_R")

# ONNX to ATen decomp table
ONNX_ATEN_DECOMP_TABLE: dict[torch._ops.OpOverload, Callable] = {}
_ATTENTION_23_ALLOWED_INTERMEDIATE_PRECISIONS = frozenset(
    {
        1,  # FLOAT
        10,  # FLOAT16
        11,  # DOUBLE
        16,  # BFLOAT16
    }
)


def _onnx_op(
    op_type: str, opset_version: int, fake_impl: Callable[_P, _R]
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Decorator to register an ONNX operator with a custom implementation."""

    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        overload = f"opset{opset_version}"
        torch_op = torch.library.custom_op(
            f"onnx::{op_type}.{overload}", mutates_args=()
        )(func)
        ONNX_ATEN_DECOMP_TABLE[getattr(getattr(torch.ops.onnx, op_type), overload)] = (
            func  # type: ignore[assignment]
        )
        torch_op.register_fake(fake_impl)
        return torch_op  # type: ignore[return-value]

    return decorator


def _rotary_embedding_23_fake_impl(
    x: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    *,
    interleaved: bool = False,
    num_heads: int = 0,
    rotary_embedding_dim: int = 0,
) -> torch.Tensor:
    """Fake implementation for RotaryEmbedding-23 for torch.compile purposes."""
    if x.dim() == 4:
        # Real impl permutes (0,2,1,3), computes via cat (allocates fresh
        # contiguous), then permutes back, yielding strides equivalent to
        # empty_permuted with (0,2,1,3) regardless of input strides.
        return torch.empty_permuted(
            x.shape, (0, 2, 1, 3), dtype=x.dtype, device=x.device
        )
    # 3D path: real impl ends in torch.reshape(contiguous_output, input_shape),
    # which always yields contiguous strides. Force contiguous here so the
    # fake stays aligned even when the caller passes a non-contiguous input.
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


@_onnx_op("RotaryEmbedding", 23, _rotary_embedding_23_fake_impl)
def rotary_embedding_23(
    x: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    *,
    interleaved: bool = False,
    num_heads: int = 0,
    rotary_embedding_dim: int = 0,
) -> torch.Tensor:
    """RotaryEmbedding-23 https://onnx.ai/onnx/operators/onnx__RotaryEmbedding.html#rotaryembedding-23"""
    # x has shape (batch_size, num_heads, sequence_length, head_size)
    # or (batch_size, sequence_length, hidden_size)
    input_shape = x.shape
    input_rank = len(input_shape)
    batch_size = input_shape[0]
    sequence_length = input_shape[-2]

    # Validate position_ids and caches match x
    if position_ids is not None:
        torch._check(
            position_ids.dim() == 2,
            lambda: f"position_ids must be 2D when provided. Received shape {position_ids.shape}",
        )
        torch._check(
            position_ids.shape[0] == batch_size,
            lambda: f"position_ids first dim (batch) must match x.shape[0] ({batch_size}). Received {position_ids.shape[0]}",
        )
        torch._check(
            position_ids.shape[1] == sequence_length,
            lambda: f"position_ids second dim (sequence) must match x.shape[-2] ({sequence_length}). Received {position_ids.shape[1]}",
        )
        torch._check(
            cos_cache.dim() == 2 and sin_cache.dim() == 2,
            lambda: "cos_cache/sin_cache must be 2D when position_ids is provided. "
            f"Received cos_cache shape {cos_cache.shape}, sin_cache shape {sin_cache.shape}",
        )
    else:
        torch._check(
            cos_cache.dim() == 3 and sin_cache.dim() == 3,
            lambda: "cos_cache/sin_cache must be 3D when position_ids is not provided. "
            f"Received cos_cache shape {cos_cache.shape}, sin_cache shape {sin_cache.shape}",
        )

    # First ensure x has shape [batch_size, num_heads, seq_len, head_size]
    # So that the rotation logic can be shared with reshaped 3D inputs
    if input_rank == 4:
        # Reshape from (batch_size, num_heads, seq_len, head_size)
        # to [batch_size, seq_len, num_heads, head_size]
        x = torch.permute(x, (0, 2, 1, 3))
    elif input_rank == 3:
        torch._check(
            num_heads != 0,
            lambda: f"num_heads must be provided for 3D inputs. Received input tensor with shape {input_shape}",
        )
        hidden_size = input_shape[2]
        head_size = hidden_size // num_heads
        new_shape = [batch_size, sequence_length, num_heads, head_size]
        x = torch.reshape(x, new_shape)

    torch._check(len(x.shape) == 4, lambda: "x should be a 4D tensor by now")
    head_size = x.shape[3]

    # Fully or partially perform rotation on x based on rotary_embedding_dim attribute
    if rotary_embedding_dim == 0:
        # If rotary_embedding_dim not provided, perform full rotation by using head_size
        rotary_embedding_dim = head_size
    x_rotate = x[:, :, :, :rotary_embedding_dim]
    x_not_rotate = x[:, :, :, rotary_embedding_dim:]
    rotary_embedding_dim_half = rotary_embedding_dim // 2

    # Retrieve sin and cos caches using position ids
    if position_ids is not None:
        cos = cos_cache[
            position_ids
        ]  # Shape: [batch_size, sequence_length, head_size/2]
        sin = sin_cache[
            position_ids
        ]  # Shape: [batch_size, sequence_length, head_size/2]
    else:
        cos = cos_cache  # Shape: [batch_size, sequence_length, rotary_embedding_dim/2]
        sin = sin_cache  # Shape: [batch_size, sequence_length, rotary_embedding_dim/2]

    torch._check(
        cos.shape[0] == batch_size and cos.shape[1] == sequence_length,
        lambda: f"cos has shape {cos.shape} but expected (batch={batch_size}, seq={sequence_length}, ...)",
    )
    torch._check(
        sin.shape[0] == batch_size and sin.shape[1] == sequence_length,
        lambda: f"sin has shape {sin.shape} but expected (batch={batch_size}, seq={sequence_length}, ...)",
    )
    torch._check(
        cos.shape[-1] == rotary_embedding_dim_half,
        lambda: f"Last dimension of cos cache ({cos.shape[-1]}) should match rotary_embedding_dim/2 ({rotary_embedding_dim_half}).",
    )
    torch._check(
        sin.shape[-1] == rotary_embedding_dim_half,
        lambda: f"Last dimension of sin cache ({sin.shape[-1]}) should match rotary_embedding_dim/2 ({rotary_embedding_dim_half}).",
    )
    cos = torch.unsqueeze(
        cos, 2
    )  # Shape: [batch_size, sequence_length, 1, rotary_embedding_dim/2]
    sin = torch.unsqueeze(
        sin, 2
    )  # Shape: [batch_size, sequence_length, 1, rotary_embedding_dim/2]

    # Either divide the x in halves or interleave (based on interleaved attribute)
    if interleaved:
        x1 = x_rotate[:, :, :, 0::2]
        x2 = x_rotate[:, :, :, 1::2]
    else:
        x1, x2 = torch.chunk(x_rotate, 2, dim=-1)

    # Calculate real and imaginary values
    real = cos * x1 - sin * x2
    imag = sin * x1 + cos * x2

    # Inserted rotated embeddings back to the original x
    if interleaved:
        # x_rotate[:, :, :, 0::2] = real
        # x_rotate[:, :, :, 1::2] = imag
        real = torch.unsqueeze(real, -1)
        imag = torch.unsqueeze(imag, -1)
        x_rotate_concat = torch.cat((real, imag), dim=-1)
        x_rotate = torch.reshape(x_rotate_concat, x_rotate.shape)
    else:
        x_rotate = torch.cat((real, imag), dim=-1)
    output = torch.cat((x_rotate, x_not_rotate), dim=-1)
    if input_rank == 3:
        return torch.reshape(output, input_shape)

    # Return the dimensions to the original order
    return torch.permute(output, (0, 2, 1, 3))


def _get_scale_factor(scale: float | None, head_size: int) -> float:
    """Get the scale factor for attention computation."""
    return scale if scale is not None else (1.0 / math.sqrt(head_size))


def _reshape_3d_to_4d(
    tensor: torch.Tensor, batch_size: int, num_heads: int
) -> torch.Tensor:
    """Reshape 3D tensor to 4D for multi-head attention."""
    sequence_length, hidden_size = tensor.shape[1], tensor.shape[2]
    head_size = hidden_size // num_heads
    return (
        tensor.view(batch_size, sequence_length, num_heads, head_size)
        .transpose(1, 2)
        .contiguous()
    )


def _get_qk_output_for_aten_spda(
    Q: torch.Tensor,
    K: torch.Tensor,
    current_q_num_heads: int,
    current_kv_num_heads: int,
    scale: float | None,
    qk_matmul_output_mode: int,
) -> torch.Tensor:
    """Get QK output tensor based on the specified mode."""
    if qk_matmul_output_mode == 0:
        return _compute_qk_output_for_mode_0(
            Q, K, current_q_num_heads, current_kv_num_heads, scale
        )
    else:
        # For other modes, return a zero tensor with correct shape
        return torch.zeros_like(torch.matmul(Q, K.transpose(-2, -1)))


def _validate_gqa_configuration(
    current_q_num_heads: int, current_kv_num_heads: int
) -> None:
    """Validate Group Query Attention configuration."""
    torch._check(
        current_q_num_heads % current_kv_num_heads == 0,
        lambda: f"q_num_heads ({current_q_num_heads}) must be divisible by kv_num_heads ({current_kv_num_heads}) for GQA",
    )


def _compute_qk_output_for_mode_0(
    Q: torch.Tensor,
    K: torch.Tensor,
    current_q_num_heads: int,
    current_kv_num_heads: int,
    scale: float | None,
) -> torch.Tensor:
    """Helper function to compute QK output for qk_matmul_output_mode == 0."""
    # Handle GQA manually for QK output
    K_for_qk = K
    if current_q_num_heads != current_kv_num_heads:
        repeat_factor = current_q_num_heads // current_kv_num_heads
        K_for_qk = K.repeat_interleave(repeat_factor, dim=1)

    scale_factor = _get_scale_factor(scale, Q.shape[3])
    # Scale both Q and K by sqrt(scale_factor) for numerical stability
    sqrt_scale = math.sqrt(scale_factor)
    Q_scaled = Q * sqrt_scale
    K_scaled = K_for_qk * sqrt_scale
    return torch.matmul(Q_scaled, K_scaled.transpose(-2, -1))


def _attention_23_fake_impl(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    past_key: torch.Tensor | None = None,
    past_value: torch.Tensor | None = None,
    *,
    is_causal: bool = False,
    kv_num_heads: int = 0,
    q_num_heads: int = 0,
    qk_matmul_output_mode: int = 0,
    scale: float | None = None,
    softcap: float = 0.0,
    softmax_precision: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fake implementation for Attention-23 for torch.compile purposes."""
    batch_size = Q.shape[0]

    # Handle 3D vs 4D input shapes
    if len(Q.shape) == 3:
        # 3D input: (batch_size, sequence_length, hidden_size)
        q_sequence_length = Q.shape[1]
        output_shape = Q.shape  # Same shape as Q for 3D output

        # For present_key and present_value, we need 4D shapes
        if past_key is not None:
            present_key_shape = (
                batch_size,
                kv_num_heads,
                past_key.shape[2] + K.shape[1],  # Combined sequence length
                K.shape[2] // kv_num_heads,  # head_size
            )
        else:
            present_key_shape = (
                batch_size,
                kv_num_heads,
                K.shape[1],  # sequence_length
                K.shape[2] // kv_num_heads,  # head_size
            )
        present_value_shape = present_key_shape  # Same shape as present_key

        # QK output shape for 3D input (reshaped to 4D internally)
        qk_output_shape = (
            batch_size,
            q_num_heads,
            q_sequence_length,
            present_key_shape[2],  # kv_sequence_length
        )
    else:
        # 4D input: (batch_size, num_heads, sequence_length, head_size)
        q_sequence_length = Q.shape[2]
        # Same shape as Q for 4D output
        output_shape = Q.shape  # type: ignore[assignment]

        # Handle past key/value concatenation
        if past_key is not None:
            present_key_shape = (
                K.shape[0],  # batch_size
                K.shape[1],  # num_heads
                past_key.shape[2] + K.shape[2],  # Combined sequence length
                K.shape[3],  # head_size
            )
        else:
            present_key_shape = K.shape  # type: ignore[assignment]
        present_value_shape = present_key_shape  # Same shape as present_key

        # QK output shape
        qk_output_shape = (
            Q.shape[0],  # batch_size
            Q.shape[1],  # q_num_heads
            Q.shape[2],  # q_sequence_length
            present_key_shape[2],  # kv_sequence_length
        )

    # Create fake tensors with correct shapes and dtypes
    output = torch.empty(output_shape, dtype=Q.dtype, device=Q.device)
    present_key = torch.empty(present_key_shape, dtype=K.dtype, device=K.device)
    present_value = torch.empty(present_value_shape, dtype=V.dtype, device=V.device)
    qk_output = torch.empty(qk_output_shape, dtype=Q.dtype, device=Q.device)

    return output, present_key, present_value, qk_output


@_onnx_op("Attention", 23, _attention_23_fake_impl)
def attention_23(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    past_key: torch.Tensor | None = None,
    past_value: torch.Tensor | None = None,
    *,
    is_causal: bool = False,
    kv_num_heads: int = 0,
    q_num_heads: int = 0,
    qk_matmul_output_mode: int = 0,
    scale: float | None = None,
    softcap: float = 0.0,
    softmax_precision: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attention-23 https://onnx.ai/onnx/operators/onnx__Attention.html#attention-23"""

    num_head_dim, sequence_dim, head_dim = 1, 2, 3

    # Store original input shape to determine output shape
    input_shape_len = len(Q.shape)
    batch_size = Q.shape[0]

    # Reshape 3D inputs to 4D format
    if len(Q.shape) == 3:
        torch._check(
            q_num_heads != 0 and kv_num_heads != 0,
            lambda: "q_num_heads and kv_num_heads must be provided for 3D inputs",
        )
        q_sequence_length = Q.shape[1]
        Q = _reshape_3d_to_4d(Q, batch_size, q_num_heads)
        K = _reshape_3d_to_4d(K, batch_size, kv_num_heads)
        V = _reshape_3d_to_4d(V, batch_size, kv_num_heads)

    torch._check(
        len(Q.shape) == 4 and len(K.shape) == 4 and len(V.shape) == 4,
        lambda: "Q, K, and V should be 4D tensors by now",
    )

    # Calculate scale factor if not provided
    q_head_size = Q.shape[head_dim]
    scale = _get_scale_factor(scale, q_head_size)

    # Handle past key/value caches
    present_key = (
        torch.cat([past_key, K], dim=sequence_dim)
        if past_key is not None
        else K.clone()
    )
    present_value = (
        torch.cat([past_value, V], dim=sequence_dim)
        if past_value is not None
        else V.clone()
    )

    # Update K and V to include past states
    K, V = present_key, present_value

    # Get current dimensions
    current_q_num_heads = Q.shape[num_head_dim]
    current_kv_num_heads = K.shape[num_head_dim]
    q_sequence_length = Q.shape[sequence_dim]
    kv_sequence_length = K.shape[sequence_dim]

    # Check if we can use the optimized scaled_dot_product_attention (most optimized)
    can_use_sdpa = (
        softcap == 0.0  # No softcap
        and qk_matmul_output_mode == 0  # Default QK output mode
        and softmax_precision is None  # No custom softmax precision
        and (attn_mask is None or attn_mask.dtype == torch.bool)
    )

    _validate_gqa_configuration(current_q_num_heads, current_kv_num_heads)

    if can_use_sdpa:
        # Use PyTorch's optimized scaled_dot_product_attention
        output = torch.nn.functional.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=bool(
                current_q_num_heads != current_kv_num_heads
            ),  # Ensure enable_gqa is not SymBool
        )

        qk_output = _get_qk_output_for_aten_spda(
            Q,
            K,
            current_q_num_heads,
            current_kv_num_heads,
            scale,
            qk_matmul_output_mode,
        )
    else:
        # Fallback to manual implementation for complex cases

        # Handle Group Query Attention (GQA) and Multi-Query Attention (MQA)
        if current_q_num_heads != current_kv_num_heads:
            repeat_factor = current_q_num_heads // current_kv_num_heads
            K = K.repeat_interleave(repeat_factor, dim=num_head_dim)
            V = V.repeat_interleave(repeat_factor, dim=num_head_dim)

        # Create attention bias
        attn_bias = torch.zeros(
            q_sequence_length, kv_sequence_length, dtype=Q.dtype, device=Q.device
        )

        # Apply causal masking
        if is_causal:
            torch._check(
                attn_mask is None, lambda: "Cannot use both is_causal and attn_mask"
            )
            causal_mask = torch.tril(
                torch.ones(
                    q_sequence_length,
                    kv_sequence_length,
                    dtype=torch.bool,
                    device=Q.device,
                )
            )
            attn_bias = attn_bias.masked_fill(~causal_mask, float("-inf"))

        # Apply attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                # Boolean mask: True means participate in attention
                attn_bias = attn_bias.masked_fill(~attn_mask, float("-inf"))
            else:
                # Float mask: added to attention scores
                attn_bias = attn_bias + attn_mask

        # Apply scaling factor
        scale_factor = _get_scale_factor(scale, Q.shape[3])

        # Scale both Q and K by sqrt(scale_factor) for numerical stability
        sqrt_scale = math.sqrt(scale_factor)
        Q_scaled = Q * sqrt_scale
        K_scaled = K * sqrt_scale

        # Compute Q @ K^T
        qk_matmul_output = torch.matmul(Q_scaled, K_scaled.transpose(-2, -1))

        # Initialize QK output based on mode
        qk_output = qk_matmul_output  # Default case for mode 0

        # Add attention bias
        qk_with_bias = qk_matmul_output + attn_bias

        if qk_matmul_output_mode == 1:
            qk_output = qk_with_bias

        # Apply softcap if provided
        if softcap > 0.0:
            qk_with_bias = softcap * torch.tanh(qk_with_bias / softcap)

        if qk_matmul_output_mode == 2:
            qk_output = qk_with_bias

        # Apply softmax with optional precision casting
        if softmax_precision is not None:
            # Map ONNX data type to torch dtype
            if softmax_precision in _ATTENTION_23_ALLOWED_INTERMEDIATE_PRECISIONS:
                original_dtype = qk_with_bias.dtype
                qk_with_bias = qk_with_bias.to(
                    _dtype_mappings.ONNX_DTYPE_TO_TORCH_DTYPE[softmax_precision]
                )
                qk_softmax = torch.softmax(qk_with_bias, dim=-1)
                qk_softmax = qk_softmax.to(original_dtype)
            else:
                qk_softmax = torch.softmax(qk_with_bias, dim=-1)
        else:
            qk_softmax = torch.softmax(qk_with_bias, dim=-1)

        if qk_matmul_output_mode == 3:
            qk_output = qk_softmax

        # Compute attention output
        output = torch.matmul(qk_softmax, V)

    # Reshape output back to 3D if input was 3D
    if input_shape_len == 3:
        # output: (batch_size, q_num_heads, q_sequence_length, v_head_size) -> (batch_size, q_sequence_length, hidden_size)
        output = output.transpose(1, 2).reshape(batch_size, q_sequence_length, -1)

    return output, present_key, present_value, qk_output
<|endoftext|>"""Implementation of symbolic FX ops to represent arbitrary ONNX ops.

This module provides a way to create symbolic FX operators that can represent
arbitrary ONNX operators.

The operators are called "symbolic" because they don't do any actual computation
but instead serve as placeholders in the computation graph.

Each implementation contains two parts: A "real" implementation that produces all
zeros based on the input shape and dtype, and a "fake" implementation that does more
or less the same thing but is required by the `torch.library.custom_op` interface.
"""

import dataclasses
from collections.abc import Sequence

import torch
from torch.onnx.ops import _dtype_mappings


_INT_TYPE = "i"
_FLOAT_TYPE = "f"
_STRING_TYPE = "s"
_INT_SEQ_TYPE = "is"
_FLOAT_SEQ_TYPE = "fs"
_STRING_SEQ_TYPE = "ss"


@dataclasses.dataclass
class EncodedAttrs:
    """Class to encode attributes from dictionary into lists of FX compatible attributes.

    Since FX does not support dictionaries, we need to encode the attributes into
    lists. This class provides a way to encode and decode the attributes.

    Attributes:
        attr_keys: List of attribute keys.
        attr_types: List of attribute types. Values can be "i" (int), "f" (float),
            "s" (string), "is" (int sequence), "fs" (float sequence), or "ss" (string sequence).
        attr_pos: List of tuples representing the start and end positions of each
            attribute in the corresponding list.
        attr_ints: List of integer attributes.
        attr_floats: List of float attributes.
        attr_strs: List of string attributes.
    """

    attr_keys: list[str]
    attr_types: list[str]
    attr_pos: list[tuple[int, int]]
    attr_ints: list[int]
    attr_floats: list[float]
    attr_strs: list[str]

    @classmethod
    def from_dict(
        cls,
        attrs: dict[
            str,
            int
            | float
            | str
            | bool
            | Sequence[int]
            | Sequence[float]
            | Sequence[str]
            | Sequence[bool],
        ],
    ) -> "EncodedAttrs":
        encoded = cls(
            attr_keys=[],
            attr_types=[],
            attr_pos=[],
            attr_ints=[],
            attr_floats=[],
            attr_strs=[],
        )
        for k, v in attrs.items():
            encoded.attr_keys.append(k)
            if isinstance(v, int):
                start_pos = len(encoded.attr_ints)
                encoded.attr_ints.append(v)
                encoded.attr_pos.append((start_pos, start_pos + 1))
                encoded.attr_types.append(_INT_TYPE)
            elif isinstance(v, float):
                start_pos = len(encoded.attr_floats)
                encoded.attr_floats.append(v)
                encoded.attr_pos.append((start_pos, start_pos + 1))
                encoded.attr_types.append(_FLOAT_TYPE)
            elif isinstance(v, str):
                start_pos = len(encoded.attr_strs)
                encoded.attr_strs.append(v)
                encoded.attr_pos.append((start_pos, start_pos + 1))
                encoded.attr_types.append(_STRING_TYPE)
            elif isinstance(v, Sequence):
                if len(v) == 0:
                    raise ValueError(f"Empty sequence for attribute {k}")
                if any(isinstance(elem, float) for elem in v):
                    start_pos = len(encoded.attr_floats)
                    encoded.attr_floats.extend([float(elem) for elem in v])
                    encoded.attr_pos.append((start_pos, start_pos + len(v)))
                    encoded.attr_types.append(_FLOAT_SEQ_TYPE)
                elif isinstance(v[0], int):
                    start_pos = len(encoded.attr_ints)
                    encoded.attr_ints.extend([int(elem) for elem in v])
                    encoded.attr_pos.append((start_pos, start_pos + len(v)))
                    encoded.attr_types.append(_INT_SEQ_TYPE)
                elif isinstance(v[0], str):
                    start_pos = len(encoded.attr_strs)
                    encoded.attr_strs.extend([str(elem) for elem in v])
                    encoded.attr_pos.append((start_pos, start_pos + len(v)))
                    encoded.attr_types.append(_STRING_SEQ_TYPE)
                else:
                    raise ValueError(f"Unsupported sequence type for attribute {k}")
            else:
                raise ValueError(f"Unsupported attribute type for {k}: {type(v)}")
        if len(encoded.attr_keys) != len(encoded.attr_types):
            raise AssertionError(
                f"Mismatch between number of attribute keys and types: {len(encoded.attr_keys)} != {len(encoded.attr_types)}"
            )
        if len(encoded.attr_keys) != len(encoded.attr_pos):
            raise AssertionError(
                f"Mismatch between number of attribute keys and positions: {len(encoded.attr_keys)} != {len(encoded.attr_pos)}"
            )
        return encoded

    def to_dict(
        self,
    ) -> dict[
        str,
        int | float | str | list[int] | list[float] | list[str],
    ]:
        """Convert the encoded attributes back to a dictionary for creating an ONNX node."""
        attrs: dict[
            str,
            int | float | str | list[int] | list[float] | list[str],
        ] = {}
        for i, key in enumerate(self.attr_keys):
            attr_type = self.attr_types[i]
            if attr_type == _INT_TYPE:
                attrs[key] = self.attr_ints[self.attr_pos[i][0]]
            elif attr_type == _FLOAT_TYPE:
                attrs[key] = self.attr_floats[self.attr_pos[i][0]]
            elif attr_type == _STRING_TYPE:
                attrs[key] = self.attr_strs[self.attr_pos[i][0]]
            elif attr_type == _FLOAT_SEQ_TYPE:
                attrs[key] = self.attr_floats[self.attr_pos[i][0] : self.attr_pos[i][1]]
            elif attr_type == _INT_SEQ_TYPE:
                attrs[key] = self.attr_ints[self.attr_pos[i][0] : self.attr_pos[i][1]]
            elif attr_type == _STRING_SEQ_TYPE:
                attrs[key] = self.attr_strs[self.attr_pos[i][0] : self.attr_pos[i][1]]
            else:
                raise ValueError(f"Unsupported attribute type: {attr_type}")
        return attrs


@torch.library.custom_op(
    "onnx_symbolic::_symbolic",
    mutates_args=(),
    schema=(
        "(Tensor?[] inputs, str op_type, int onnx_dtype, *,"
        " SymInt[] shape, str[] attr_keys, str[] attr_types, int[][] attr_pos,"
        " int[] attr_ints, float[] attr_floats, str[] attr_strs, str[] metadata_props_keys,"
        " str[] metadata_props_values, str domain='', int? version=None"
        ") -> Tensor"
    ),
)
def _symbolic(
    inputs: Sequence[torch.Tensor | None],
    op_type: str,
    onnx_dtype: int,
    *,
    shape: Sequence[int | torch.SymInt],
    attr_keys: Sequence[str],
    attr_types: Sequence[str],
    attr_pos: Sequence[tuple[int, int]],
    attr_ints: Sequence[int],
    attr_floats: Sequence[float],
    attr_strs: Sequence[str],
    metadata_props_keys: Sequence[str] = (),
    metadata_props_values: Sequence[str] = (),
    domain: str = "",
    version: int | None = None,
) -> torch.Tensor:
    torch._check(
        onnx_dtype in _dtype_mappings.ONNX_DTYPE_TO_TORCH_DTYPE,
        lambda: f"{onnx_dtype} is invalid as an ONNX data type. Valid values are {list(_dtype_mappings.ONNX_DTYPE_TO_TORCH_DTYPE.keys())}",
    )
    return torch.zeros(
        shape, dtype=_dtype_mappings.ONNX_DTYPE_TO_TORCH_DTYPE[onnx_dtype]
    )


@_symbolic.register_fake
def _(
    inputs: Sequence[torch.Tensor],
    op_type: str,
    onnx_dtype: int,
    *,
    shape: Sequence[int | torch.SymInt],
    attr_keys: Sequence[str],
    attr_types: Sequence[str],
    attr_pos: Sequence[tuple[int, int]],
    attr_ints: Sequence[int],
    attr_floats: Sequence[float],
    attr_strs: Sequence[str],
    metadata_props_keys: Sequence[str] = (),
    metadata_props_values: Sequence[str] = (),
    domain: str = "",
    version: int | None = None,
) -> torch.Tensor:
    torch._check(
        onnx_dtype in _dtype_mappings.ONNX_DTYPE_TO_TORCH_DTYPE,
        lambda: f"{onnx_dtype} is invalid as an ONNX data type. Valid values are {list(_dtype_mappings.ONNX_DTYPE_TO_TORCH_DTYPE.keys())}",
    )
    # NOTE(justinchuby): Use zeros instead of torch.empty because I haven't figured
    # out how it can handle empty shapes
    return torch.zeros(
        shape, dtype=_dtype_mappings.ONNX_DTYPE_TO_TORCH_DTYPE[onnx_dtype]
    )


@torch.library.custom_op(
    "onnx_symbolic::_symbolic_multi_out",
    mutates_args=(),
    schema=(
        "(Tensor?[] inputs, str op_type, int[] onnx_dtypes, *,"
        " SymInt[][] shapes, str[] attr_keys, str[] attr_types, int[][] attr_pos,"
        " int[] attr_ints, float[] attr_floats, str[] attr_strs, str[] metadata_props_keys,"
        " str[] metadata_props_values, str domain='', int? version=None"
        ") -> Tensor[]"
    ),
)
def _symbolic_multi_out(
    inputs: Sequence[torch.Tensor | None],
    op_type: str,
    onnx_dtypes: Sequence[int],
    *,
    shapes: Sequence[Sequence[int | torch.SymInt]],
    attr_keys: Sequence[str],
    attr_types: Sequence[str],
    attr_pos: Sequence[tuple[int, int]],
    attr_ints: Sequence[int],
    attr_floats: Sequence[float],
    attr_strs: Sequence[str],
    metadata_props_keys: Sequence[str] = (),
    metadata_props_values: Sequence[str] = (),
    domain: str = "",
    version: int | None = None,
) -> list[torch.Tensor]:
    outputs = []
    torch._check(
        len(shapes) == len(onnx_dtypes),
        lambda: f"Number of shapes ({len(shapes)}) must match number of ONNX dtypes ({len(onnx_dtypes)})",
    )
    for shape, onnx_dtype in zip(shapes, onnx_dtypes):
        torch._check(
            onnx_dtype in _dtype_mappings.ONNX_DTYPE_TO_TORCH_DTYPE,
            lambda: f"{onnx_dtype} is invalid as an ONNX data type. Valid values are {list(_dtype_mappings.ONNX_DTYPE_TO_TORCH_DTYPE.keys())}",
        )
        outputs.append(
            torch.zeros(
                shape, dtype=_dtype_mappings.ONNX_DTYPE_TO_TORCH_DTYPE[onnx_dtype]
            )
        )
    return outputs


@_symbolic_multi_out.register_fake
def _(
    inputs: Sequence[torch.Tensor],
    op_type: str,
    onnx_dtypes: Sequence[int],
    *,
    shapes: Sequence[Sequence[int | torch.SymInt]],
    attr_keys: Sequence[str],
    attr_types: Sequence[str],
    attr_pos: Sequence[tuple[int, int]],
    attr_ints: Sequence[int],
    attr_floats: Sequence[float],
    attr_strs: Sequence[str],
    metadata_props_keys: Sequence[str] = (),
    metadata_props_values: Sequence[str] = (),
    domain: str = "",
    version: int | None = None,
) -> list[torch.Tensor]:
    outputs = []
    torch._check(
        len(shapes) == len(onnx_dtypes),
        lambda: f"Number of shapes ({len(shapes)}) must match number of ONNX dtypes ({len(onnx_dtypes)})",
    )
    for shape, onnx_dtype in zip(shapes, onnx_dtypes):
        torch._check(
            onnx_dtype in _dtype_mappings.ONNX_DTYPE_TO_TORCH_DTYPE,
            lambda: f"{onnx_dtype} is invalid as an ONNX data type. Valid values are {list(_dtype_mappings.ONNX_DTYPE_TO_TORCH_DTYPE.keys())}",
        )
        # NOTE(justinchuby): Use zeros instead of torch.empty because I haven't figured
        # out how it can handle empty shapes
        outputs.append(
            torch.zeros(
                shape, dtype=_dtype_mappings.ONNX_DTYPE_TO_TORCH_DTYPE[onnx_dtype]
            )
        )
    return outputs
<|endoftext|>"""ONNX operators as native torch.fx operators.

This module provides a set of functions to create ONNX operators in the FX graph
which are exportable to ONNX.
"""

from __future__ import annotations


__all__ = [
    "aten_decompositions",
    "symbolic",
    "symbolic_multi_out",
    "rotary_embedding",
    "attention",
]


from typing import TYPE_CHECKING

import torch
from torch.onnx.ops import _impl, _symbolic_impl


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


# https://github.com/onnx/onnx/blob/f542e1f06699ea7e1db5f62af53355b64338c723/onnx/onnx.proto#L597
_TORCH_DTYPE_TO_ONNX_DTYPE = {
    torch.float32: 1,  # FLOAT
    torch.uint8: 2,  # UINT8
    torch.int8: 3,  # INT8
    torch.uint16: 4,  # UINT16
    torch.int16: 5,  # INT16
    torch.int32: 6,  # INT32
    torch.int64: 7,  # INT64
    str: 8,  # STRING
    torch.bool: 9,  # BOOL
    torch.float16: 10,  # FLOAT16
    torch.double: 11,  # DOUBLE
    torch.uint32: 12,  # UINT32
    torch.uint64: 13,  # UINT64
    torch.complex64: 14,  # COMPLEX64
    torch.complex128: 15,  # COMPLEX128
    torch.bfloat16: 16,  # BFLOAT16
    torch.float8_e4m3fn: 17,  # FLOAT8E4M3FN
    torch.float8_e4m3fnuz: 18,  # FLOAT8E4M3FNUZ
    torch.float8_e5m2: 19,  # FLOAT8E5M2
    torch.float8_e5m2fnuz: 20,  # FLOAT8E5M2FNUZ
    # 21 = UINT4
    # 22 = INT4
    torch.float4_e2m1fn_x2: 23,  # FLOAT4E2M1
}


def aten_decompositions() -> dict[torch._ops.OpOverload, Callable]:
    """Return the ONNX to ATen decomp table."""
    return _impl.ONNX_ATEN_DECOMP_TABLE


def _parse_domain_op_type(domain_op: str) -> tuple[str, str]:
    split = domain_op.split("::", 1)
    if len(split) == 1:
        domain = ""
        op_type = split[0]
    else:
        domain = split[0]
        op_type = split[1]
    return domain, op_type


def symbolic(
    domain_op: str,
    /,
    inputs: Sequence[torch.Tensor | None],
    attrs: dict[
        str,
        int
        | float
        | str
        | bool
        | Sequence[int]
        | Sequence[float]
        | Sequence[str]
        | Sequence[bool],
    ]
    | None = None,
    *,
    dtype: torch.dtype | int,
    shape: Sequence[int | torch.SymInt],
    version: int | None = None,
    metadata_props: dict[str, str] | None = None,
) -> torch.Tensor:
    """Create a symbolic FX operator to represent an arbitrary ONNX operator.

    This function is used to create a symbolic operator with a single output.
    To create an operator with multiple outputs, use :func:`symbolic_multi_out`.

    You may use ``if torch.onnx.is_in_onnx_export()`` to conditionally enable the
    symbolic logic only during ``torch.onnx.export()``.

    Example::

        class CustomOp(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Normal torch operators can interleave with the symbolic ops during ONNX export
                x = x + 1

                # Create a symbolic ONNX operator with the name "CustomOp" in the "custom_domain" domain.
                # The output tensor will have the specified dtype and shape
                val = torch.onnx.ops.symbolic(
                    "custom_domain::CustomOp",
                    (x,),
                    dict(attr_key="attr_value"),
                    dtype=x.dtype,
                    shape=x.shape,
                    version=1,
                )

                # The result of the symbolic op can be used in normal torch operations during ONNX export
                return torch.nn.functional.relu(val)


        # You may then export this model to ONNX using torch.onnx.export(..., dynamo=True).

    Args:
        domain_op: The domain and operator name, separated by "::". For example,
            "custom_domain::CustomOp".
        inputs: The input tensors to the operator.
        attrs: The attributes of the operator. The keys are attribute names and
            the values are attribute values. Valid attribute types are int, float,
            str, bool, and lists of int, float, str, and bool. Tensor attributes
            are unsupported.
        dtype: The data type of the output tensor.This can be either a torch.dtype
            or an integer representing the ONNX data type.
        shape: The shape of the output tensor. This can be a list of integers or
            SymInt values.
        version: The version of the opset used for the operator.
        metadata_props: Metadata properties for the ONNX node.
            This is a dictionary of str-str pairs.

    Returns:
        The output tensor of the operator.
    """
    if not isinstance(dtype, int):
        torch._check(
            dtype in _TORCH_DTYPE_TO_ONNX_DTYPE, lambda: f"Unsupported dtype: {dtype}"
        )
        dtype = _TORCH_DTYPE_TO_ONNX_DTYPE[dtype]
    domain, op_type = _parse_domain_op_type(domain_op)
    if attrs is None:
        attrs = {}
    encoded_attrs = _symbolic_impl.EncodedAttrs.from_dict(attrs)
    # TODO: Parse domain
    return _symbolic_impl._symbolic(
        inputs,
        op_type,
        dtype,
        shape=shape,
        attr_keys=encoded_attrs.attr_keys,
        attr_types=encoded_attrs.attr_types,
        attr_pos=encoded_attrs.attr_pos,
        attr_ints=encoded_attrs.attr_ints,
        attr_floats=encoded_attrs.attr_floats,
        attr_strs=encoded_attrs.attr_strs,
        metadata_props_keys=metadata_props.keys() if metadata_props else [],
        metadata_props_values=metadata_props.values() if metadata_props else [],
        domain=domain,
        version=version,
    )


def symbolic_multi_out(
    domain_op: str,
    /,
    inputs: Sequence[torch.Tensor | None],
    attrs: dict[
        str,
        int
        | float
        | str
        | bool
        | Sequence[int]
        | Sequence[float]
        | Sequence[str]
        | Sequence[bool],
    ]
    | None = None,
    *,
    dtypes: Sequence[torch.dtype | int],
    shapes: Sequence[Sequence[int | torch.SymInt]],
    version: int | None = None,
    metadata_props: dict[str, str] | None = None,
) -> Sequence[torch.Tensor]:
    """Create a symbolic FX operator to represent an arbitrary ONNX operator with multiple outputs.

    You may use ``if torch.onnx.is_in_onnx_export()`` to conditionally enable the
    symbolic logic only during ``torch.onnx.export()``.

    Example::

        class CustomOp(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Normal torch operators can interleave with the symbolic ops during ONNX export
                x = x + 1

                # Create a symbolic ONNX operator with the name "CustomOp" in the "custom_domain" domain.
                # The output tensors will have the specified dtypes and shapes
                (out1, out2) = torch.onnx.ops.symbolic_multi_out(
                    "custom_domain::CustomOp",
                    (x,),
                    dict(attr_key="attr_value"),
                    dtypes=(x.dtype, torch.float32),
                    shapes=(x.shape, [1, 2, 3]),
                    version=1,
                )

                # The result of the symbolic op can be used in normal torch operations during ONNX export
                return torch.nn.functional.relu(out1 + out2)


        # You may then export this model to ONNX using torch.onnx.export(..., dynamo=True).

    Args:
        domain_op: The domain and operator name, separated by "::". For example,
            "custom_domain::CustomOp".
        inputs: The input tensors to the operator.
        attrs: The attributes of the operator. The keys are attribute names and
            the values are attribute values. Valid attribute types are int, float,
            str, bool, and lists of int, float, str, and bool. Tensor attributes
            are unsupported.
        dtypes: The data types of the output tensors. This can be a list of
            torch.dtype or integers representing the ONNX data types. The length
            of this list must be the number of outputs.
        shapes: The shapes of the output tensors. This can be a list of lists of
            integers or SymInt values. The length of this list must be the number of outputs.
        version: The version of the opset used for the operator.
        metadata_props: Metadata properties for the ONNX node.
            This is a dictionary of str-str pairs.

    Returns:
        A list of output tensors of the operator.
    """
    torch._check(
        len(shapes) == len(dtypes),
        lambda: f"Number of shapes ({len(shapes)}) must match number of dtypes ({len(dtypes)})",
    )
    onnx_dtypes = []
    for dtype in dtypes:
        if not isinstance(dtype, int):
            torch._check(
                dtype in _TORCH_DTYPE_TO_ONNX_DTYPE,
                lambda: f"Unsupported dtype: {dtype}",
            )
            onnx_dtypes.append(_TORCH_DTYPE_TO_ONNX_DTYPE[dtype])
        else:
            onnx_dtypes.append(dtype)
    domain, op_type = _parse_domain_op_type(domain_op)
    if attrs is None:
        attrs = {}
    encoded_attrs = _symbolic_impl.EncodedAttrs.from_dict(attrs)
    # Use the size of dtypes to determine the number of outputs
    return _symbolic_impl._symbolic_multi_out(
        inputs,
        op_type,
        onnx_dtypes,
        shapes=shapes,
        attr_keys=encoded_attrs.attr_keys,
        attr_types=encoded_attrs.attr_types,
        attr_pos=encoded_attrs.attr_pos,
        attr_ints=encoded_attrs.attr_ints,
        attr_floats=encoded_attrs.attr_floats,
        attr_strs=encoded_attrs.attr_strs,
        metadata_props_keys=metadata_props.keys() if metadata_props else [],
        metadata_props_values=metadata_props.values() if metadata_props else [],
        domain=domain,
        version=version,
    )


def rotary_embedding(
    X: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    *,
    interleaved: bool = False,
    num_heads: int = 0,
    rotary_embedding_dim: int = 0,
) -> torch.Tensor:
    """RotaryEmbedding op in ONNX.

    https://onnx.ai/onnx/operators/onnx__RotaryEmbedding.html

    RotaryEmbedding is the implementation of rotary positional embeddings (RoPE) based on the paper https://arxiv.org/pdf/2104.09864.
    The key advantage of RoPE is that it allows the model to understand both the absolute position of a token and the relative distances
    between tokens. This is achieved through a rotational mechanism where the extent of rotation is computed based on the token's absolute position (position_ids).

    The rotational mechanism is defined by sine and cosine functions that are used to represent the rotation angles.
    For each token in the sequence, its positional embedding is computed by rotating its embedding vector. This is done by splitting the
    embedding vector either into two halves or interleaving every alternate token and applying the rotation matrix to each half of the embedding vector.
    The rotation matrix is parameterized by the token's position in the sequence. The rotated halves of the embedding vector are concatenated
    to form the final positional embedding for each token. The rotated positional embeddings are used in the self-attention mechanism.
    The rotation ensures that the model captures both absolute and relative positional information.

    Args:
        X: The input tensor representing the token embeddings. 4D tensor with
            shape `(batch_size, num_heads, sequence_length, head_size)` or 3D tensor
            with shape `(batch_size, sequence_length, hidden_size)`. For cases with
            a 4D input tensor, `head_size` has to be even. For cases with a 3D input
            tensor, `num_heads` attribute must be provided and `hidden_size` must
            be an even multiple of `num_heads` where `hidden_size = num_heads * head_size`
        cos_cache: The cosine values for the rotation. 2D tensor with shape `(max_position_id_plus_1, head_size / 2)`
            for full rotation or `(max_position_id_plus_1, rotary_embedding_dim / 2)`
            for partial rotation when `position_ids` are provided. 3D tensor with shape
            `(batch_size, sequence_length, head_size / 2)` for full rotation or
            `(batch_size, sequence_length, rotary_embedding_dim / 2)` for partial
            rotation when `position_ids` are not provided. `max_position_id_plus_1`
            is a parameter to the model.
        sin_cache: The sine values for the rotation. 2D tensor with shape `(max_position_id_plus_1, head_size / 2)`
            for full rotation or `(max_position_id_plus_1, rotary_embedding_dim / 2)`
            for partial rotation when `position_ids` are provided. 3D tensor with shape
            `(batch_size, sequence_length, head_size / 2)` for full rotation or
            `(batch_size, sequence_length, rotary_embedding_dim / 2)` for partial rotation
            when `position_ids` are not provided. `max_position_id_plus_1` is a parameter
            to the model.
        position_ids: The position indices for the tokens. 2D tensor with shape
            `(batch_size, sequence_length)`.
        interleaved: Rotate using interleaved pattern. Default value is 0 (False).
        num_heads: Number of attention heads. Must be provided when input is a 3D tensor.
        rotary_embedding_dim: Rotary embedding dimension used to apply partial rotary embeddings.

    Returns:
        Tensor with same shape as input.
    """
    return _impl.rotary_embedding_23(
        X,
        cos_cache,
        sin_cache,
        position_ids=position_ids,
        interleaved=interleaved,
        num_heads=num_heads,
        rotary_embedding_dim=rotary_embedding_dim,
    )


def attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    past_key: torch.Tensor | None = None,
    past_value: torch.Tensor | None = None,
    *,
    is_causal: bool = False,
    kv_num_heads: int = 0,
    q_num_heads: int = 0,
    qk_matmul_output_mode: int = 0,
    scale: float | None = None,
    softcap: float = 0.0,
    softmax_precision: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attention op in ONNX.

    https://onnx.ai/onnx/operators/onnx__Attention.html

    Computes scaled dot product attention on query, key and value tensors, using an optional attention mask if passed.

    This operator covers self and cross variants of the attention operation based on sequence lengths of K, Q and V.

    For self attention, ``kv_sequence_length`` equals to ``q_sequence_length``.

    For cross attention, query and key might have different lengths.

    This operator also covers the 3 following variants based on the number of heads:

    1. Multi-headed Attention (MHA): Described in the paper https://arxiv.org/pdf/1706.03762, `q_num_heads = kv_num_heads`.
    2. Group-query Attention (GQA): Described in the paper https://arxiv.org/pdf/2305.13245, `q_num_heads > kv_num_heads`, `q_num_heads % kv_num_heads == 0`.
    3. Multi-query Attention (MQA): Described in the paper https://arxiv.org/pdf/1911.02150, `q_num_heads > kv_num_heads`, `kv_num_heads=1`.

    Attention bias to be added is calculated based on ``attn_mask`` input and ``is_causal` `attribute``, only one of which can be provided.

    1. If ``is_causal`` is set to `1`, the attention masking is a lower triangular matrix when the mask is a square matrix. The attention masking has the form of the upper left causal bias due to the alignment.
    2. `attn_mask`: A boolean mask where a value of `True` indicates that the element should take part in attention or a float mask of the same type as query, key, value that is added to the attention score.

    Both past and present state key/values are optional. They shall be used together, and not allowed to use only one of them.
    The following pattern is applied to the Q, K and V inputs after appropriate reshaping of K and V inputs based on sequence lengths and num heads provided::

        The following pattern is applied by this operator:
                Q          K          V
                |          |          |
        Q*sqrt(scale) K*sqrt(scale) |
                |          |          |
                |       Transpose     |
                |          |          |
                ---MatMul---          |
                    |               |
        at_mask---Add              |
                    |               |
            softcap (if provided)     |
                    |               |
                Softmax            |
                    |               |
                    -----MatMul------
                            |
                            Y

    Args:
        Q: Query tensor. 4D tensor with shape `(batch_size, q_num_heads, q_sequence_length, head_size)` or 3D tensor
            with shape `(batch_size, q_sequence_length, q_hidden_size)`. For cases with a 3D input tensor,
            `q_hidden_size = q_num_heads * head_size`
        K: Key tensor. 4D tensor with shape `(batch_size, kv_num_heads, kv_sequence_length, head_size)` or 3D tensor
            with shape `(batch_size, kv_sequence_length, k_hidden_size)`. For cases with a 3D input tensor,
            `k_hidden_size = kv_num_heads * head_size`
        V: Value tensor. 4D tensor with shape `(batch_size, kv_num_heads, kv_sequence_length, v_head_size)` or 3D tensor
            with shape `(batch_size, kv_sequence_length, v_hidden_size)`. For cases with a 3D input tensor,
            `v_hidden_size = kv_num_heads * v_head_size`
        attn_mask: Attention mask. Shape must be broadcastable to 4D tensor with shape
            `(batch_size, q_num_heads, q_sequence_length, total_sequence_length)` where
            `total_sequence_length = past_sequence_length + kv_sequence_length`. Two types of masks are supported.
            A boolean mask where a value of True indicates that the element should take part in attention.
            Also supports a float mask of the same type as query, key, value that is added to the attention score.
        past_key: Past state cache for key with shape `(batch_size, kv_num_heads, past_sequence_length, head_size)`
        past_value: Past state cache for value with shape `(batch_size, kv_num_heads, past_sequence_length, v_head_size)`
        is_causal: If set to True, the attention masking is a lower triangular matrix when the mask is a square matrix.
            The attention masking has the form of the upper left causal bias due to the alignment.
        kv_num_heads: Number of heads of key and value. Must be used with 3D inputs of Q, K and V.
        q_num_heads: Number of heads of query. Must be used with 3D inputs of Q, K and V.
        qk_matmul_output_mode: If set to 0, qk_matmul_output is the output of qk matmul. If set to 1,
            qk_matmul_output includes the addition of the attention mask to the output of qk matmul.
            If set to 2, qk_matmul_output is the output after the softcap operation. If set to 3,
            qk_matmul_output is the output after the softmax operation. Default value is 0.
        scale: Scaling factor applied to Q*K^T. Default value is 1/sqrt(head_size). To prevent numerical overflow,
            scale Q, K by sqrt(scale) before matmul.
        softcap: Softcap value for attention weights. Default value is 0.
        softmax_precision: The floating-point precision used in softmax computation. If softmax precision is not provided,
            the same precision as the input of softmax (Q and K) is used.

    Returns:
        A tuple containing:
        - The output tensor. 4D tensor with shape `(batch_size, q_num_heads, q_sequence_length, v_head_size)` or 3D tensor
          with shape `(batch_size, q_sequence_length, hidden_size)`. For cases with a 3D input tensor,
          `hidden_size = q_num_heads * v_head_size`
        - Updated key cache with shape `(batch_size, kv_num_heads, total_sequence_length, head_size)` where
          `total_sequence_length = past_sequence_length + kv_sequence_length`.
        - Updated value cache with shape `(batch_size, kv_num_heads, total_sequence_length, v_head_size)` where
          `total_sequence_length = past_sequence_length + kv_sequence_length`.
        - The output of QK matmul. 4D tensor with shape `(batch_size, q_num_heads, q_sequence_length, total_sequence_length)`
          where `total_sequence_length = past_sequence_length + kv_sequence_length`.
    """
    return _impl.attention_23(
        Q,
        K,
        V,
        attn_mask=attn_mask,
        past_key=past_key,
        past_value=past_value,
        is_causal=is_causal,
        kv_num_heads=kv_num_heads,
        q_num_heads=q_num_heads,
        qk_matmul_output_mode=qk_matmul_output_mode,
        scale=scale,
        softcap=softcap,
        softmax_precision=softmax_precision,
    )
<|endoftext|>"""Utility to lazily import modules."""

from __future__ import annotations

import importlib
from typing import Any, TYPE_CHECKING


class _LazyModule:
    """Lazily import a module."""

    def __init__(self, module_name: str) -> None:
        self._name = module_name
        self._module: Any = None

    def __repr__(self) -> str:
        return f"<lazy module '{self._name}'>"

    def __getattr__(self, attr: str) -> object:
        if self._module is None:
            self._module = importlib.import_module(".", self._name)
        return getattr(self._module, attr)


# Import the following modules during type checking to enable code intelligence features,
# such as auto-completion in tools like pylance, even when these modules are not explicitly
# imported in user code.
# NOTE: Add additional used imports here.
if TYPE_CHECKING:
    import onnx
    import onnx_ir  # type: ignore[import-untyped, import-not-found]
    import onnxscript
    import onnxscript._framework_apis.torch_2_11 as onnxscript_apis

else:
    onnx = _LazyModule("onnx")
    onnx_ir = _LazyModule("onnx_ir")
    onnxscript = _LazyModule("onnxscript")
    onnxscript_apis = _LazyModule("onnxscript._framework_apis.torch_2_11")
<|endoftext|># mypy: allow-untyped-defs
"""Utilities for converting and operating on ONNX and torch types."""

from __future__ import annotations

from typing import Any
from typing_extensions import TypeIs

import torch


def is_torch_symbolic_type(
    value: Any,
) -> TypeIs[torch.SymBool | torch.SymInt | torch.SymFloat]:
    return isinstance(value, (torch.SymBool, torch.SymInt, torch.SymFloat))


def from_scalar_type_to_torch_dtype(scalar_type: type) -> torch.dtype | None:
    return _SCALAR_TYPE_TO_TORCH_DTYPE.get(scalar_type)


_PYTHON_TYPE_TO_TORCH_DTYPE = {
    bool: torch.bool,
    int: torch.int64,
    float: torch.float32,
    complex: torch.complex64,
}

_SYM_TYPE_TO_TORCH_DTYPE = {
    torch.SymInt: torch.int64,
    torch.SymFloat: torch.float32,
    torch.SymBool: torch.bool,
}

_SCALAR_TYPE_TO_TORCH_DTYPE: dict[type, torch.dtype] = {
    **_PYTHON_TYPE_TO_TORCH_DTYPE,
    **_SYM_TYPE_TO_TORCH_DTYPE,  # type: ignore[dict-item]
}
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import abc
import contextlib
import dataclasses
import difflib
import io
import sys
from typing import Any, TYPE_CHECKING

import torch
import torch.fx
from torch._subclasses.fake_tensor import unset_fake_temporarily


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch._subclasses import fake_tensor


@dataclasses.dataclass
class PackageInfo:
    package_name: str
    version: str | None
    commit_hash: str | None

    def to_onnx_domain_string(self) -> str:
        return ".".join(
            filter(None, ("pkg", self.package_name, self.version, self.commit_hash))
        )

    @classmethod
    def from_python_class(cls, python_class_name: type | str) -> PackageInfo:
        if isinstance(python_class_name, type):
            python_class_name = python_class_name.__module__
        package_name = python_class_name.split(".")[0]
        package = __import__(package_name)
        version = getattr(package, "__version__", None)
        # TODO: Figure out how to retrieve commit hash.
        commit_hash = None
        return cls(package_name, version, commit_hash)


@dataclasses.dataclass
class GraphModuleOnnxMeta:
    package_info: PackageInfo


@contextlib.contextmanager
def _patch_difflib_sequence_matcher_init():
    """Context patching `difflib.SequenceMatcher` for fx readable graph.

    Under this context, the `autojunk` argument of `difflib.SequenceMatcher` will always
    be considered as `False`. This is to prevent `difflib.SequenceMatcher` recognizing
    stacktrace messages in fx readable graph as junk, as these messages tend to be long (>200)
    and repeat multiple times, which falls under the junk filter criteria.

    `difflib.SequenceMatcher` is used underneath by all sorts of diffing functions
    in `difflib`, including `difflib.unified_diff`, `difflib.ndiff`, `difflib.context_diff`.
    Unfortunately, there is no way to pass `autojunk` argument to these functions, and
    they all default to `True`. This context patching will affect all of them.

    `Reference: Automatic junk heuristic <https://docs.python.org/3/library/difflib.html>`_
    """
    original_init = difflib.SequenceMatcher.__init__

    def patched_init(self, isjunk=None, a="", b="", autojunk=True) -> None:
        original_init(self, isjunk, a, b, autojunk=False)

    difflib.SequenceMatcher.__init__ = patched_init  # type: ignore[assignment]
    try:
        yield
    finally:
        difflib.SequenceMatcher.__init__ = original_init  # type: ignore[assignment]


def _unified_diff(a: str, b: str) -> str:
    """Return a string containing the unified diff of two strings.

    This function calls a patched version of `difflib.unified_diff` with `autojunk` set
    to `False` for `difflib.SequenceMatcher` class. More details can be found in
    `_patch_difflib_sequence_matcher_init` function.

    Args:
        a: The first string.
        b: The second string.

    Returns:
        The unified diff of the two strings. If there is no diff, return "<no diff>".

    Example::

        >>> a = '''class GraphModule(torch.nn.Module):
        ...     def forward(self, input_ids : torch.Tensor, attention_mask : torch.Tensor):
        ...         # File: /modeling.py:770, code: input_ids = input_ids.view(-1, input_shape[-1])
        ...         view = input_ids.view(-1, 3);  input_ids = None
        ... '''
        >>> b = '''class <lambda>(torch.nn.Module):
        ...     def forward(self, input_ids: i64[1, 3], attention_mask: i64[1, 3]):
        ...         # File: /modeling.py:770, code: input_ids = input_ids.view(-1, input_shape[-1])
        ...         view: i64[1, 3] = torch.ops.aten.view.default(input_ids, [-1, 3]);  input_ids = None
        ... '''
        >>> print(_unified_diff(a, b))
        ---
        +++
        @@ -1,4 +1,4 @@
        -class GraphModule(torch.nn.Module):
        -    def forward(self, input_ids : torch.Tensor, attention_mask : torch.Tensor):
        +class <lambda>(torch.nn.Module):
        +    def forward(self, input_ids: i64[1, 3], attention_mask: i64[1, 3]):
                # File: /modeling.py:770, code: input_ids = input_ids.view(-1, input_shape[-1])
        -        view = input_ids.view(-1, 3);  input_ids = None
        +        view: i64[1, 3] = torch.ops.aten.view.default(input_ids, [-1, 3]);  input_ids = None
    """

    a_list = a.splitlines(keepends=True)
    b_list = b.splitlines(keepends=True)

    with _patch_difflib_sequence_matcher_init():
        # Set `n` to `sys.maxsize` to show entire graph when there is a diff.
        diff = "".join(difflib.unified_diff(a_list, b_list, n=sys.maxsize))

    if not diff:
        return "<no diff>"
    return diff


def _transform_diagnose_call_message_formatter(
    run: Callable,
    self: Transform,
    *args: Any,
    **kwargs: Any,
) -> str:
    return f"Running {self.__class__.__name__} pass. "


def maybe_fx_graph_tabular(graph: torch.fx.Graph) -> str | None:
    """Return the Graph nodes in tabular format. Equivalent to stdout of `graph.print_tabular()`.
    If `tabulate` is not installed, return `None`.

    Args:
        graph: The Graph to print.

    Returns:
        The Graph printed in a tabular format. None if `tabulate` is not installed.
    """
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        try:
            graph.print_tabular()
        except ImportError:
            return None
    return f.getvalue()


class Transform(abc.ABC):
    """Base class for FX graph transformations to be used by FX-ONNX exporter.

    Similar to `FX Interpreter <https://pytorch.org/docs/stable/fx.html#torch.fx.Interpreter>`_,
    specializations of this class execute the FX graph Node-by-Node.
    Methods in the `Transform` class can be overridden to customize the behavior of the model.
    This pattern can be useful for many things, including writing code transformations as well as analysis passes.

    The following methods can be overridden::

        _run()
            +-- run_node()
                +-- placeholder()
                +-- get_attr()
                +-- call_function()
                +-- call_method()
                +-- call_module()
                +-- output()

    One important aspect to note is that if the transformation modifies the model input and/or output signature,
    (e.g. additional inputs/outputs are added to the model), :class:`InputAdaptStep` and/or :class:`OutputAdaptStep`
    are needed to reconcile :attr:`ONNXProgram.model_proto`.
    That is, the model signature and the model representation must match.

    TODO(bowbao): Add more overridable methods in call hierarchy
    TODO(bowbao): Create an example once more overridable methods are added.
    """

    module: torch.fx.GraphModule
    """The module to be transformed."""

    fake_mode: fake_tensor.FakeTensorMode | None
    """The existing fake mode detected from `self.module`."""

    def __init__(
        self,
        module: torch.fx.GraphModule,
    ) -> None:
        """Initialize the transform.

        Args:
            module: The module to be transformed.
        """
        self.module = module
        self.fake_mode = self._detect_fake_mode()

    def _detect_fake_mode(self) -> fake_tensor.FakeTensorMode | None:
        """Detect fake mode from the graph.

        Scan through all nodes in graph and their meta['val'] to detect fake mode.
        """
        fake_tensors = [node.meta.get("val") for node in self.module.graph.nodes]
        with unset_fake_temporarily():
            return torch._dynamo.utils.detect_fake_mode(fake_tensors)

    def _maybe_fakefy_args(
        self, fake_mode: fake_tensor.FakeTensorMode | None, *args: Any
    ) -> tuple[Any, ...]:
        if fake_mode is None:
            return args
        # NB: This should hit the cache if tensors were fakefied before.
        # E.g., when the fx graph is produced by Dynamo.
        return tuple(
            fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t for t in args
        )

    @abc.abstractmethod
    def _run(self, *args, **kwargs) -> torch.fx.GraphModule: ...

    def run(self, *args, **kwargs) -> torch.fx.GraphModule:
        """Run the transform on `self.module`.

        Note that this method may or may not mutate `self.module`, and the returned
        `GraphModule` could be either `self.module` or a new `GraphModule`.

        Args:
            *args: Positional arguments for `self.module` to run.
            **kwargs: Keyword arguments for `self.module` to run.
        """
        return self._run(*args, **kwargs)
<|endoftext|># mypy: allow-untyped-defs
# Owner(s): ["module: onnx"]
from __future__ import annotations

import abc
import dataclasses
import inspect
import logging
from typing import Any, TYPE_CHECKING

import torch
import torch._dispatch.python
import torch._ops
import torch.fx
import torch.fx.traceback as fx_traceback
from torch import _prims_common, _refs
from torch._prims_common import (
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    wrappers as _prims_common_wrappers,
)
from torch._refs import linalg as _linalg_refs, nn as _nn_refs, special as _special_refs
from torch._refs.nn import functional as _functional_refs
from torch.fx.experimental import proxy_tensor
from torch.onnx._internal.fx import _pass, type_utils as fx_type_utils
from torch.utils import _pytree


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from types import ModuleType

    from torch._subclasses import fake_tensor


logger = logging.getLogger(__name__)


def _try_getclosurevars(func):
    try:
        return inspect.getclosurevars(func)
    except TypeError:
        return None


@dataclasses.dataclass
class TypePromotionSnapshot:
    """Type promotion snapshot for a fx node and its inputs.

    Contains the promoted dtype for args and kwargs that needs promoting.
    Contains the expected node output dtype.
    """

    args_dtypes: Mapping[int, torch.dtype]
    """Mapping from arg position to dtype to promote to."""

    kwargs_dtypes: Mapping[str, torch.dtype]
    """Mapping from kwarg name to dtype to promote to."""

    out_dtype: torch.dtype
    """Expected output dtype of the node."""


class TypePromotionRule(abc.ABC):
    """Base class for type promotion rule per 'torch.ops.{namespace}.{op_name}'."""

    def __init__(self, namespace: str, op_name: str) -> None:
        self.namespace = namespace
        self.op_name = op_name

    # Make this abstract as well because subclass needs to override __eq__().
    # A class that overrides __eq__() and does not define __hash__() will have its __hash__() implicitly set to None.
    # Ref: https://docs.python.org/3/reference/datamodel.html#object.__hash__
    @abc.abstractmethod
    def __hash__(self) -> int: ...

    @abc.abstractmethod
    def __repr__(self) -> str: ...

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool: ...

    def is_valid(self) -> bool:
        """Check if the rule is valid."""
        # This always returns a module. If the module does not exist it will be created.
        module = getattr(torch.ops, self.namespace)
        py_op = getattr(module, self.op_name, None)
        if py_op is None:
            logger.warning(
                "Cannot find op: %s in module: %s", self.op_name, self.namespace
            )
            return False
        if not isinstance(py_op, torch._ops.OpOverloadPacket):
            logger.warning(
                "Op: torch.ops.%s.%s is not an OpOverloadPacket, got: %s",
                self.namespace,
                self.op_name,
                type(py_op),
            )
            return False

        return True

    @abc.abstractmethod
    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot:
        """Preview type promotion results for provided set of args and kwargs.

        Returns a TypePromotionSnapshot object that contains the promoted dtypes for
        the arguments and the expected output dtype.
        """
        ...


class ElementwiseTypePromotionRule(TypePromotionRule):
    """Defines how to perform elementwise type promotion for 'torch.ops.{namespace}.{op_name}'."""

    _USE_OPMATH: bool = False
    """Whether to use opmath to compute the promoted input dtype.
    If used, upcasts will be inserted everywhere for lower precision models.
    Set to False and have torchlib handle upcasts in op implementation internally.
    """

    def __init__(
        self,
        namespace: str,
        op_name: str,
        promote_args_positions: Sequence[int],
        promote_kwargs_names: Sequence[str],
        promotion_kind: _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND,
    ) -> None:
        """Constructs a TypePromotionRule for elementwise operators.

        Args:
            namespace: Namespace of the op. E.g. 'aten' in 'torch.ops.aten.add'.
            op_name: Name of the op. E.g. 'add' in 'torch.ops.aten.add'.
            promote_args_positions: Positions of args to promote.
            promote_kwargs_names: Names of kwargs to promote.
            promotion_kind: Type promotion kind. Refer to [_prims_common.elementwise_dtypes](https://github.com/pytorch/pytorch/blob/main/torch/_prims_common/__init__.py) for detail.  # noqa: B950
        """
        super().__init__(namespace, op_name)
        self.promote_args_positions = promote_args_positions
        self.promote_kwargs_names = promote_kwargs_names
        self.promotion_kind = promotion_kind

    def __repr__(self) -> str:
        return (
            f"ElementwiseTypePromotionRule('{self.namespace}', '{self.op_name}', "
            f"{self.promote_args_positions}, {self.promote_kwargs_names}, {self.promotion_kind})"
        )

    # pyrefly: ignore [bad-override]
    def __eq__(self, other: object, /) -> bool:
        if not isinstance(other, ElementwiseTypePromotionRule):
            return False
        return (
            self.namespace == other.namespace
            and self.op_name == other.op_name
            and self.promote_args_positions == other.promote_args_positions
            and self.promote_kwargs_names == other.promote_kwargs_names
            and self.promotion_kind == other.promotion_kind
        )

    def __hash__(self) -> int:
        return f"{type(self)}:{self.namespace}.{self.op_name}".__hash__()

    def _consolidate_input_dtype(
        self, computed_dtype: torch.dtype, result_dtype: torch.dtype
    ) -> torch.dtype:
        """
        Although opmath is the right thing to do to retain on-par precision, it inserts
        upcasts everywhere in the graph. This is particularly hard for backend to optimize
        since there is no way to differentiate between inserted upcasts and model code
        casts. Hence we consolidate the input dtype to the result dtype to avoid this.
        """
        if not self._USE_OPMATH and self.promotion_kind in (
            _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
            _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        ):
            return result_dtype
        return computed_dtype

    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot:
        candidate_args = {
            i: args[i]
            for i in self.promote_args_positions
            if i < len(args) and args[i] is not None
        }
        candidate_kwargs = {
            name: kwargs[name]
            for name in self.promote_kwargs_names
            if name in kwargs and kwargs[name] is not None
        }

        computed_dtype, result_dtype = _prims_common.elementwise_dtypes(
            *_pytree.arg_tree_leaves(*candidate_args.values(), **candidate_kwargs),
            type_promotion_kind=self.promotion_kind,
        )

        consolidated_input_dtype = self._consolidate_input_dtype(
            computed_dtype, result_dtype
        )

        return TypePromotionSnapshot(
            dict.fromkeys(candidate_args.keys(), consolidated_input_dtype),
            dict.fromkeys(candidate_kwargs.keys(), consolidated_input_dtype),
            result_dtype,
        )


class DivElementwiseTypePromotionRule(ElementwiseTypePromotionRule):
    """Reference type promotion rule from torch._refs.div.

    Rule depends on the value of the `rounding_mode` argument.
    """

    def __init__(self) -> None:
        super().__init__(
            "aten",
            "div",
            promote_args_positions=(0, 1),
            promote_kwargs_names=(),
            promotion_kind=_prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
        )

    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot:
        rounding_mode = kwargs.get("rounding_mode")
        if rounding_mode is None:
            # true_divide
            self.promotion_kind = (
                _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
            )
            return super().preview_type_promotion(args, kwargs)
        if rounding_mode == "trunc":
            # trunc_divide
            self.promotion_kind = _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
            return super().preview_type_promotion(args, kwargs)
        if rounding_mode == "floor":
            # floor_divide
            self.promotion_kind = _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
            return super().preview_type_promotion(args, kwargs)
        raise ValueError(f"Unknown rounding_mode: {rounding_mode}")


class ReductionTypePromotionRule(TypePromotionRule):
    def __init__(
        self,
        namespace: str,
        op_name: str,
        promotion_kind: _prims_common.REDUCTION_OUTPUT_TYPE_KIND,
    ) -> None:
        """Constructs a TypePromotionRule for reduction operators.

        Args:
            namespace: Namespace of the op. E.g. 'aten' in 'torch.ops.aten.sum'.
            op_name: Name of the op. E.g. 'sum' in 'torch.ops.aten.sum'.
            promotion_kind: Type promotion kind. Refer to [_prims_common.reduction_dtypes]((https://github.com/pytorch/pytorch/blob/main/torch/_prims_common/__init__.py)) for detail.  # noqa: B950
        """
        super().__init__(namespace, op_name)
        self.promotion_kind = promotion_kind

    def __repr__(self) -> str:
        return f"ReductionTypePromotionRule('{self.namespace}', '{self.op_name}', {self.promotion_kind})"

    # pyrefly: ignore [bad-override]
    def __eq__(self, other: object, /) -> bool:
        if not isinstance(other, ElementwiseTypePromotionRule):
            return False
        return (
            self.namespace == other.namespace
            and self.op_name == other.op_name
            and self.promotion_kind == other.promotion_kind
        )

    def __hash__(self) -> int:
        return f"{type(self)}:{self.namespace}.{self.op_name}".__hash__()

    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot:
        if len(args) < 1:
            raise AssertionError(
                f"Reduction op torch.ops.{self.namespace}.{self.op_name} expects at least one argument"
            )
        arg = args[0]
        if not isinstance(arg, torch.Tensor):
            raise AssertionError(f"{type(arg)=} is not torch.Tensor")
        dtype: torch.dtype | None = kwargs.get("dtype")

        computation_dtype, result_dtype = _prims_common.reduction_dtypes(
            arg, self.promotion_kind, dtype
        )
        if result_dtype is None:
            # Inspecting code, this can only happen when `promotion_kind` is `KEEP_PROMOTED_TYPE`.
            # Hence set same as computation_dtype.
            result_dtype = computation_dtype

        return TypePromotionSnapshot(
            {0: computation_dtype},
            {},
            result_dtype,
        )


class AllOrAnyReductionTypePromotionRule(ReductionTypePromotionRule):
    """Reference type promotion rule from torch.ops.aten.all or torch.ops.aten.any.

    This is a special case where computation dtype is always torch.bool.
    The result dtype is always uint8 if `dtype` kwarg is uint8, otherwise torch.bool.
    """

    def __init__(self, op_name: str) -> None:
        super().__init__(
            "aten",
            op_name,
            _prims_common.REDUCTION_OUTPUT_TYPE_KIND.ALWAYS_BOOL,
        )

    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot:
        if len(args) < 1:
            raise AssertionError(
                f"Reduction op torch.ops.{self.namespace}.{self.op_name} expects at least one argument"
            )
        arg = args[0]
        if not isinstance(arg, torch.Tensor):
            raise AssertionError(f"{type(arg)=} is not torch.Tensor")
        computation_dtype = torch.bool
        # Preserves uint8 -- probably a legacy mask thing
        result_dtype = torch.uint8 if arg.dtype == torch.uint8 else torch.bool
        return TypePromotionSnapshot(
            {0: computation_dtype},
            {},
            result_dtype,
        )


class SumLikeReductionTypePromotionRule(ReductionTypePromotionRule):
    """Reference type promotion rule from torch.ops.aten.sum.

    This is a special case where computation dtype is always torch.int64 for integral arg,
    unless overridden by `dtype` kwarg.
    """

    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot:
        if len(args) < 1:
            raise AssertionError(
                f"Reduction op torch.ops.{self.namespace}.{self.op_name} expects at least one argument"
            )
        arg = args[0]
        if not isinstance(arg, torch.Tensor):
            raise AssertionError(f"{type(arg)=} is not torch.Tensor")
        dtype: torch.dtype | None = kwargs.get("dtype")
        # The below logic is copied from `torch/_refs/__init__.py` reduction ops impl.
        if dtype is None:
            if _prims_common.is_boolean_dtype(
                arg.dtype
            ) or _prims_common.is_integer_dtype(arg.dtype):
                dtype = torch.int64
            else:
                dtype = arg.dtype
        return super().preview_type_promotion(args, {"dtype": dtype})


# NOTE: [Update type promotion rule]
# BELOW TABLE IS GENERATED FROM `TypePromotionRuleSetGenerator.generate_from_torch_refs`.
# DO NOT EDIT MANUALLY !!!
# For missing rules or discrepancies, please
# 1. Run `pytest test/onnx/test_fx_type_promotion.py` to validate if the generated rule set is current.
#    If it is not, update with new generated set.
# 2. If discrepancies still exist, consider debugging torch._refs or report a bug.
# 3. If rules are still missing, add them to `_EXTRA_TYPE_PROMOTION_RULE_SET` or report a bug.
# Check `TypePromotionRule` class for how each rule is defined and used.
_GENERATED_ATEN_TYPE_PROMOTION_RULE_SET = {
    ElementwiseTypePromotionRule(
        "aten", "abs", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "abs_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "acos", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "acos_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "acosh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "acosh_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "add", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "add_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "addcdiv", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "addcdiv_", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "addcmul", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "addcmul_", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "addr", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "asin", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "asin_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "asinh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "asinh_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "atan", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "atan2", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "atan2_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "atan_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "atanh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "atanh_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "bitwise_and", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "bitwise_and_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten",
        "bitwise_left_shift",
        [0, 1],
        [],
        ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    ),
    ElementwiseTypePromotionRule(
        "aten",
        "bitwise_left_shift_",
        [0, 1],
        [],
        ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    ),
    ElementwiseTypePromotionRule(
        "aten", "bitwise_not", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "bitwise_not_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "bitwise_or", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "bitwise_or_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten",
        "bitwise_right_shift",
        [0, 1],
        [],
        ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    ),
    ElementwiseTypePromotionRule(
        "aten",
        "bitwise_right_shift_",
        [0, 1],
        [],
        ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    ),
    ElementwiseTypePromotionRule(
        "aten", "bitwise_xor", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "bitwise_xor_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "cat", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH
    ),
    ElementwiseTypePromotionRule(
        "aten", "cauchy", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "cauchy_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "ceil", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "ceil_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "celu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "celu_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "clamp", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "clamp_", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "copysign", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "copysign_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "cos", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "cos_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "cosh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "cosh_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "deg2rad", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "deg2rad_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "digamma", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "digamma_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "dot", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "elu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "elu_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "eq", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "eq_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "erf", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "erf_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "erfc", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "erfc_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "erfinv", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "erfinv_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "exp", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "exp2", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "exp2_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "exp_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "expm1", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "expm1_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "exponential", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "exponential_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "fill", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH
    ),
    ElementwiseTypePromotionRule(
        "aten", "floor", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "floor_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "floor_divide", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "floor_divide_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "fmax", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "fmin", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "fmod", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "fmod_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "frac", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "frac_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "gcd", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "gcd_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "ge", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "ge_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "gelu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "geometric", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "geometric_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "glu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "gt", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "gt_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "hardtanh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "heaviside", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "heaviside_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "huber_loss", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "hypot", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "hypot_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "i0", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "i0_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "igamma", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "igamma_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "igammac", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "igammac_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "isfinite", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "isinf", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "isnan", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "isneginf", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "isposinf", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "isreal", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "l1_loss", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "lcm", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "lcm_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "le", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "le_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "leaky_relu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "lerp", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "lerp_", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "lgamma", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "lgamma_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log10", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log10_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log1p", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log1p_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log2", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log2_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log_normal", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log_normal_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "logaddexp", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "logaddexp2", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "logical_and", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "logical_and_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "logical_not", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "logical_not_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "logical_or", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "logical_or_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "logical_xor", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "logical_xor_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "logit", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "logsumexp", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "lt", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "lt_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "maximum", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "minimum", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "mish", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "mish_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "mse_loss", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "mul", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "mul_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "ne", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "ne_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "neg", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "neg_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "nextafter", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH
    ),
    ElementwiseTypePromotionRule(
        "aten", "nextafter_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH
    ),
    ElementwiseTypePromotionRule(
        "aten", "nll_loss", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "normal", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "pdist", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten",
        "poisson_nll_loss",
        [0, 1],
        [],
        ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    ),
    ElementwiseTypePromotionRule(
        "aten", "prelu", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "rad2deg", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "rad2deg_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "reciprocal", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "reciprocal_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "relu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "remainder", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "remainder_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "round", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "rsqrt", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "rsqrt_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "selu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "selu_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sgn", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sgn_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sigmoid", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sigmoid_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sign", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sign_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "signbit", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "sin", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sin_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sinc", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sinc_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sinh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sinh_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten",
        "smooth_l1_loss",
        [0, 1],
        [],
        ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
    ),
    ElementwiseTypePromotionRule(
        "aten", "softplus", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sqrt", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sqrt_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "square", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG
    ),
    ElementwiseTypePromotionRule(
        "aten", "square_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG
    ),
    ElementwiseTypePromotionRule(
        "aten", "sub", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sub_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "tan", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "tan_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "tanh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "tanh_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "threshold", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "threshold_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "true_divide", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "true_divide_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "trunc", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "trunc_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "vdot", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "where", [1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH
    ),
    ElementwiseTypePromotionRule(
        "aten", "xlogy", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "xlogy_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
}

# Manually curated extra type promotion rules. Please see NOTE [Update type promotion rule]
# before adding new rules.
_EXTRA_TYPE_PROMOTION_RULE_SET = {
    # torch._refs skips type promotion decoration for `clamp_min` and `clamp_max` since
    # the call is routed to the decorated `aten.clamp` op.
    ElementwiseTypePromotionRule(
        "aten",
        "clamp_max",
        promote_args_positions=(0, 1),
        promote_kwargs_names=(),
        promotion_kind=_prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    ),
    ElementwiseTypePromotionRule(
        "aten",
        "clamp_min",
        promote_args_positions=(0, 1),
        promote_kwargs_names=(),
        promotion_kind=_prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    ),
    # torch.ops.aten.div.Tensor_mode applies different type promotion rules
    # depending on the value of the `mode` argument.
    DivElementwiseTypePromotionRule(),
    # Manually curating reduction ops since the logic is written inside the op reference
    # implementation.
    AllOrAnyReductionTypePromotionRule("all"),
    AllOrAnyReductionTypePromotionRule("any"),
    ReductionTypePromotionRule(
        "aten",
        "amax",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.SAME,
    ),
    ReductionTypePromotionRule(
        "aten",
        "amin",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.SAME,
    ),
    # torch.ops.aten.mean is a special case that does not need type promotion.
    ReductionTypePromotionRule(
        "aten",
        "std",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT,
    ),
    ReductionTypePromotionRule(
        "aten",
        "std_mean",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT,
    ),
    ReductionTypePromotionRule(
        "aten",
        "var",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT,
    ),
    SumLikeReductionTypePromotionRule(
        "aten",
        "cumprod",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.SAME,
    ),
    SumLikeReductionTypePromotionRule(
        "aten",
        "cumsum",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.SAME,
    ),
    SumLikeReductionTypePromotionRule(
        "aten",
        "prod",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.SAME,
    ),
    SumLikeReductionTypePromotionRule(
        "aten",
        "sum",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.SAME,
    ),
}


class ElementwiseTypePromotionRuleSetGenerator:
    """Hackily distilling info from reference ops decorated with elementwise type promotion rule.

    The goal is to retrieve the decorator

    ```python
        @elementwise_type_promotion_wrapper(
            type_promoting_args=("a", "b"),
            type_promotion_kind=type_promotion_kind,
        )
    ```

    from the reference ops. It provides info as for which arguments are promoted
    and what kind of promotion is applied.
    """

    @classmethod
    def generate_from_torch_refs(cls) -> set[ElementwiseTypePromotionRule]:
        """Parse type promotion rules from reference ops under torch._C._refs."""
        rule_set = set()
        rule_set.update(cls._parse_torch_refs(_refs))
        rule_set.update(cls._parse_torch_refs(_nn_refs))
        rule_set.update(cls._parse_torch_refs(_linalg_refs))
        rule_set.update(cls._parse_torch_refs(_special_refs))
        rule_set.update(cls._parse_torch_refs(_functional_refs))
        return rule_set

    @classmethod
    def _parse_torch_refs(
        cls, ref_module: ModuleType
    ) -> set[ElementwiseTypePromotionRule]:
        logger.info("Processing module: %s", ref_module.__name__)
        rule_set = set()
        for name in ref_module.__all__:
            decorated_op = getattr(ref_module, name)
            rule = cls._parse_type_promotion_rule_from_refs_op(decorated_op)
            if rule is not None and rule.is_valid():
                rule_set.add(rule)

        return rule_set

    @classmethod
    def _parse_type_promotion_rule_from_refs_op(
        cls,
        decorated_op: Callable,
    ) -> ElementwiseTypePromotionRule | None:
        """Retrieve and parse type promotion decorator from op under torch._refs."""
        fn = decorated_op
        type_promo_wrapper = None
        while fn_closure_vars := _try_getclosurevars(fn):
            if "fn" not in fn_closure_vars.nonlocals:
                break
            if "self" in fn_closure_vars.nonlocals and isinstance(
                fn_closure_vars.nonlocals["self"],
                _prims_common_wrappers.elementwise_type_promotion_wrapper,
            ):
                type_promo_wrapper = fn_closure_vars.nonlocals["self"]
                break
            fn = fn_closure_vars.nonlocals["fn"]

        if type_promo_wrapper is not None:
            signature = inspect.signature(decorated_op)

            pos = 0
            promote_args_positions = []
            promote_kwargs_names = []

            if type_promo_wrapper.type_promoting_arg_names is not None:
                for name, param in signature.parameters.items():
                    if name in type_promo_wrapper.type_promoting_arg_names:
                        if param.kind in (
                            param.POSITIONAL_OR_KEYWORD,
                            param.POSITIONAL_ONLY,
                        ):
                            promote_args_positions.append(pos)
                        elif param.kind == param.KEYWORD_ONLY:
                            promote_kwargs_names.append(name)
                    pos += 1

            return ElementwiseTypePromotionRule(
                "aten",
                decorated_op.__name__,
                promote_args_positions=promote_args_positions,
                promote_kwargs_names=promote_kwargs_names,
                promotion_kind=type_promo_wrapper.type_promotion_kind,
            )

        logger.warning(
            "Cannot find type promotion rule for: %s.%s",
            decorated_op.__module__,
            decorated_op.__name__,
        )
        return None


class TypePromotionTable:
    """Type promotion table for torch.ops."""

    def __init__(self) -> None:
        self._rule_table = {}
        for rule in _GENERATED_ATEN_TYPE_PROMOTION_RULE_SET:
            self.add_rule(rule)
        for rule in _EXTRA_TYPE_PROMOTION_RULE_SET:
            self.add_rule(rule)

    def add_rule(self, rule: TypePromotionRule) -> None:
        """Add a type promotion rule for a python op in a torch.ops module.

        Args:
            rule: Type promotion rule.
            module: Module containing the op. E.g. torch.ops.aten.

        Raises:
            ValueError: If the rule is invalid.
        """
        if not rule.is_valid():
            raise ValueError(f"Invalid type promotion rule: {rule}")
        self._rule_table[f"{rule.namespace}.{rule.op_name}"] = rule

    def get_rule(self, py_op: torch._ops.OpOverloadPacket) -> TypePromotionRule | None:
        """Get type promotion rule for a python op under 'torch.ops.<namespace>'."""
        return self._rule_table.get(str(py_op), None)


def get_type_promotion_rule(
    node: torch.fx.Node,
    type_promotion_table: TypePromotionTable,
) -> TypePromotionRule | None:
    """Get type promotion rule for a node.

    Args:
        node: Node to get type promotion rule for.
        type_promotion_table: Type promotion table.

    Returns:
        Type promotion rule for the node. None if no rule is found or if the node is not
        representing a torch operator.
    """
    op = node.target
    if not isinstance(op, torch._ops.OpOverload):
        return None
    if (rule := type_promotion_table.get_rule(op.overloadpacket)) is None:
        return None

    return rule


def find_compatible_op_overload(
    op: torch._ops.OpOverloadPacket, args: tuple, kwargs: dict
) -> torch._ops.OpOverload:
    """Find compatible OpOverload for an OpOverloadPacket using provided args and kwargs.

    Each "call_function" fx.Node in the fx.GraphModule has a target that represents a torch._ops.OpOverload.
    The OpOverload contains an OpOverloadPacket that holds all the available overloads for the operation.

    During the type promotion pass, there are cases where the types of the args and kwargs may change,
    such as promoting Python numbers to tensors. Consequently, the original OpOverload might not be
    compatible with the updated args and kwargs. This function is used to identify the compatible
    OpOverload for the given args and kwargs.

    Args:
        op: OpOverloadPacket to find compatible OpOverload for.
        args: The positional arguments to consider for compatibility.
        kwargs: The keyword arguments to consider for compatibility.

    Returns:
        torch._ops.OpOverload: The compatible OpOverload found for the given args and kwargs.

    Raises:
        RuntimeError: If no compatible op overload is found.

    Examples:
        >>> import torch
        >>> packet = torch.ops.aten.pow
        >>> args = (torch.tensor([1.0, 2.0]), 2)
        >>> find_compatible_op_overload(packet, args, {})._overloadname
        'Tensor_Scalar'
        >>> args = (torch.tensor([1.0, 2.0]), torch.tensor(2.0))
        >>> find_compatible_op_overload(packet, args, {})._overloadname
        'Tensor_Tensor'
    """
    exceptions = {}
    for overload_name in op.overloads():
        op_overload = getattr(op, overload_name)
        if not isinstance(op_overload, torch._ops.OpOverload):
            raise AssertionError(f"Expected OpOverload, got {type(op_overload)}")
        try:
            torch._C._check_schema_allow_fake_script_object(
                op_overload._schema, *args, **kwargs
            )
        except RuntimeError as e:
            exceptions[overload_name] = e
        else:
            return op_overload

    error_messages = "\n".join(
        f"Overload name {name}:\n{exception}" for name, exception in exceptions.items()
    )
    raise RuntimeError(
        f"Cannot find a compatible OpOverload for {op} with the provided "
        f"arguments. Tried overloads:\n{error_messages}"
    )


class _TypePromotionInterpreter(torch.fx.Interpreter):
    """Interpreter that inserts type promotion for each node."""

    def __init__(
        self,
        module: torch.fx.GraphModule,
        type_promotion_table: TypePromotionTable,
    ) -> None:
        super().__init__(module)
        self.type_promotion_table = type_promotion_table

    def _run_node_and_set_meta(self, node) -> Any:
        """Run node and set meta according to `fx_traceback.get_current_meta()`.

        This should be used on new nodes or nodes that have been modified.
        By default `Interpreter.run_node` does not update `node.meta`.
        Set `node.meta` to the current meta, except for `node.meta["val"]`, which is
        recomputed.
        """
        out = super().run_node(node)
        # Update interpreter env state with new output value.
        self.env[node] = out
        node.meta.update(
            (k, v)
            for k, v in fx_traceback.get_current_meta().items()
            if k not in node.meta
        )
        node.meta["val"] = proxy_tensor.extract_val(out)
        return out

    def _create_node(
        self,
        graph: torch.fx.Graph,
        op_type: str,
        target: torch.fx.node.Target,
        args: tuple,
        kwargs: dict,
    ) -> torch.fx.Node:
        """Create a node and set its metadata."""
        if op_type not in (
            "call_function",
            "call_method",
            "get_attr",
            "call_module",
            "placeholder",
            "output",
        ):
            raise AssertionError(f"Unexpected op_type: {op_type}")
        node = getattr(graph, op_type)(target, args, kwargs)
        self._run_node_and_set_meta(node)
        return node

    def _rerun_node_after_type_promotion(
        self,
        node: torch.fx.Node,
        expected_out_dtype: torch.dtype,
    ) -> None:
        """Rerun a node after type promotion and update node.meta["val"] with the output value."""
        node_val = node.meta.get("val", None)
        if node_val is None:
            raise AssertionError(f"Node {node} node.meta['val'] is not set.")
        args, kwargs = self.fetch_args_kwargs_from_env(node)
        target = node.target
        if not isinstance(target, torch._ops.OpOverload):
            raise AssertionError(f"Expected OpOverload, got {type(target)}")
        node.target = find_compatible_op_overload(target.overloadpacket, args, kwargs)

        new_node_val = self._run_node_and_set_meta(node)
        if not isinstance(new_node_val, type(node_val)):
            raise AssertionError(
                f"run_node output type should not change between runs. "
                f"Got {type(new_node_val)}, expect {type(node_val)}."
            )

        if isinstance(node_val, torch.Tensor):
            prev_node_dtype = node_val.dtype

            if prev_node_dtype != expected_out_dtype:
                raise AssertionError(
                    f"node.meta['val'].dtype({prev_node_dtype}) does not agree with "
                    f"type promotion rule({expected_out_dtype})."
                )

            if new_node_val.dtype != expected_out_dtype:
                # With explicit type promotion, the expected result dtype may not be
                # the same as the computation dtype. This is referred to as "op math".
                # We need to explicitly cast the output back to the expected dtype.
                # See more about "op math" topic at `_prims_common.elementwise_dtypes`.
                graph = node.graph
                with graph.inserting_after(node):
                    output_cast_node = self._create_node(
                        graph,
                        "call_function",
                        torch.ops.prims.convert_element_type.default,
                        (node,),
                        {"dtype": expected_out_dtype},
                    )
                    node.replace_all_uses_with(output_cast_node)
                    output_cast_node.args = (node,)
                    logger.info(
                        "Node '%s' output dtype becomes %s due to op math. "
                        "Cast back to %s.",
                        node,
                        new_node_val.dtype,
                        expected_out_dtype,
                    )

        elif fx_type_utils.is_torch_symbolic_type(node_val):
            raise NotImplementedError(
                "Type promotion does not support node output of sym types."
            )
        elif isinstance(node_val, (list, tuple)):
            raise NotImplementedError(
                "Type promotion does not support node output of list or tuple."
            )
        else:
            raise RuntimeError(f"Unexpected node output type: {type(node_val)}.")

    def _maybe_promote_arg(
        self,
        node: torch.fx.Node,
        fx_arg: torch.fx.node.Argument,
        dtype: torch.dtype | None,
    ) -> torch.fx.node.Argument:
        """Promote fx_arg to dtype if necessary."""
        if dtype is None:
            logger.info(
                "Argument %s is not promoted. Not mentioned by type promotion rule.",
                fx_arg,
            )
            return fx_arg

        if isinstance(fx_arg, torch.fx.Node):
            arg_val = self.env[fx_arg]
            if isinstance(arg_val, torch.Tensor):
                if (old_dtype := arg_val.dtype) != dtype:
                    # Promote tensor to dtype.
                    graph = node.graph
                    with graph.inserting_before(node):
                        logger.info(
                            "Argument %s(%s) is promoted to %s.",
                            fx_arg,
                            old_dtype,
                            dtype,
                        )
                        return self._create_node(
                            graph,
                            "call_function",
                            torch.ops.prims.convert_element_type.default,
                            (fx_arg,),
                            {"dtype": dtype},
                        )
                logger.info("Argument %s is not promoted. Already %s.", fx_arg, dtype)
                return fx_arg
            elif fx_type_utils.is_torch_symbolic_type(arg_val):
                arg_type = type(arg_val)
                equivalent_dtype = fx_type_utils.from_scalar_type_to_torch_dtype(
                    arg_type
                )
                if equivalent_dtype is None:
                    raise AssertionError(f"Unexpected arg_type: {arg_type}")
                if equivalent_dtype != dtype:
                    # Promote Sym number to tensor of dtype.
                    graph = node.graph
                    with graph.inserting_before(node):
                        logger.info(
                            "Argument %s(Scalar of equivalent dtype: %s) "
                            "is promoted to %s.",
                            fx_arg,
                            equivalent_dtype,
                            dtype,
                        )
                        return self._create_node(
                            graph,
                            "call_function",
                            torch.ops.aten.scalar_tensor.default,
                            (fx_arg,),
                            {"dtype": dtype},
                        )
                logger.info("Argument %s is not promoted. Already %s.", fx_arg, dtype)
                return fx_arg
        elif (
            equivalent_dtype := fx_type_utils.from_scalar_type_to_torch_dtype(
                type(fx_arg)
            )
        ) is not None:
            if equivalent_dtype != dtype:
                # Promote number to tensor of dtype.
                # The op should have overload that supports tensor for this arg, otherwise
                # the type promotion rule should not suggest promoting this arg.
                graph = node.graph
                with graph.inserting_before(node):
                    logger.info(
                        "Argument %s(Scalar of equivalent dtype: %s) "
                        "is promoted to %s.",
                        fx_arg,
                        equivalent_dtype,
                        dtype,
                    )
                    return self._create_node(
                        graph,
                        "call_function",
                        torch.ops.aten.scalar_tensor.default,
                        (fx_arg,),
                        {"dtype": dtype},
                    )
            logger.info("Argument %s is not promoted. Already %s.", fx_arg, dtype)
            return fx_arg
        elif isinstance(fx_arg, (tuple, list)):
            logger.info("Argument %s is a tuple/list. Promoting each element.", fx_arg)
            return type(fx_arg)(
                self._maybe_promote_arg(node, fx_arg_elem, dtype)
                for fx_arg_elem in fx_arg
            )

        raise NotImplementedError(f"Unknown fx arg type: {type(fx_arg)}")

    def _maybe_promote_node(
        self,
        node: torch.fx.Node,
        rule: TypePromotionRule,
    ) -> torch.fx.Node:
        """Promote node inputs and outputs according to type promotion rule."""
        args, kwargs = self.fetch_args_kwargs_from_env(node)
        type_promotion_info = rule.preview_type_promotion(args, kwargs)
        new_args = []
        new_kwargs = {}
        for i, arg in enumerate(node.args):
            new_args.append(
                self._maybe_promote_arg(
                    node, arg, type_promotion_info.args_dtypes.get(i, None)
                )
            )

        for name, arg in node.kwargs.items():
            new_kwargs[name] = self._maybe_promote_arg(
                node, arg, type_promotion_info.kwargs_dtypes.get(name, None)
            )
        new_args = tuple(new_args)

        if node.args != new_args or node.kwargs != new_kwargs:
            node.args = new_args
            node.kwargs = new_kwargs
            self._rerun_node_after_type_promotion(node, type_promotion_info.out_dtype)

        return node

    def run_node(self, n: torch.fx.Node) -> Any:
        """This method is an override which inserts type promotion nodes as needed.

        For each `call_function` node, an initial check is conducted to determine if a type
        promotion rule is applicable. If a relevant rule exists, type casting nodes are
        introduced for the corresponding arguments. The OpOverload of the node is updated
        to one that accommodates the promoted types. Should the output type be different,
        type casting node is inserted for this output.

        The call `super().run_node(node)` is guaranteed to be invoked for each node.
        In the case of new or modified nodes, the result of `super().run_node(node)` is
        used to update its `node.meta["val"]` value.
        """
        with self._set_current_node(n):
            if rule := get_type_promotion_rule(n, self.type_promotion_table):
                self._maybe_promote_node(n, rule)

        return super().run_node(n)


class InsertTypePromotion(_pass.Transform):
    """Explicitly insert type promotion ops to the graph.

    Underneath, the main pass is driven by `_TypePromotionInterpreter`, which is a subclass
    of `torch.fx.Interpreter` to interpret the fx.Graph and perform the insertion of type
    promotion operations.

    By re-running the new and modified nodes using the interpreter, we can update the
    metadata, specifically the fake tensor stored under node.meta["val"], and ensure it
    reflects the latest changes.
    """

    def __init__(
        self,
        module: torch.fx.GraphModule,
        type_promotion_table: TypePromotionTable | None = None,
    ) -> None:
        super().__init__(module)
        self.interpreter = _TypePromotionInterpreter(
            module, type_promotion_table or TypePromotionTable()
        )

    def _fetch_fake_args(
        self,
    ) -> Sequence[
        fake_tensor.FakeTensor
        | float
        | int
        | bool
        | torch.SymInt
        | torch.SymFloat
        | torch.SymBool
        | None
    ]:
        """Fetch fake args from fx graph.

        For each argument, try to fetch fake tensor from the matching placeholder node.
        """
        fake_args = []
        for node in self.module.graph.nodes:
            if node.op == "placeholder":
                try:
                    # Meta value can be torch.Tensor, int, float, bool,
                    # torch.SymInt, torch.SymFloat, torch.SymBool.
                    meta_value = _val = node.meta.get("val", None)
                except RuntimeError as e:
                    if not node.users:
                        # If the placeholder is not used, we can safely ignore it and put
                        # None as placeholder.
                        meta_value = None
                    else:
                        raise RuntimeError(
                            "Cannot fetch symbolic fake args from fx graph. "
                            "InsertTypePromotion pass needs to run with pre-existing fake args, "
                            "Otherwise the pass will produce inaccurate dynamic shape. "
                        ) from e

                fake_args.append(meta_value)
        return fake_args

    def _run(self, *args, **kwargs) -> torch.fx.GraphModule:
        if args:
            raise AssertionError(
                "`InsertTypePromotion` deduces symbolic fake arguments from the graph. "
                "It does not accept concrete arguments as input because this pass requires "
                "re-running the graph. When executed with newly faked concrete arguments, "
                "the pass loses the symbolic dynamic shape information."
            )
        if kwargs:
            raise AssertionError("`kwargs` is not supported")

        fake_args = self._fetch_fake_args()
        fake_mode = self.fake_mode
        if fake_mode is None:
            raise AssertionError("Cannot detect fake_mode.")

        # Use the python dispatcher to run through some python kernels which
        # can better handle symints. Without this, some SymInts can become static
        # when there are dynamic shapes.
        dispatcher_mode = torch._dispatch.python.enable_python_dispatcher()
        with fake_mode, dispatcher_mode, fx_traceback.preserve_node_meta():
            self.interpreter.run(*fake_args)

        return self.module
<|endoftext|>"""Experimental classes and functions used by ONNX export."""

import dataclasses
from collections.abc import Mapping, Sequence

import torch
import torch._C._onnx as _C_onnx


@dataclasses.dataclass
class ExportOptions:
    """Arguments used by :func:`torch.onnx.export`."""

    # TODO(justinchuby): Deprecate and remove this class.

    export_params: bool = True
    verbose: bool = False
    training: _C_onnx.TrainingMode = _C_onnx.TrainingMode.EVAL
    input_names: Sequence[str] | None = None
    output_names: Sequence[str] | None = None
    operator_export_type: _C_onnx.OperatorExportTypes = _C_onnx.OperatorExportTypes.ONNX
    opset_version: int | None = None
    do_constant_folding: bool = True
    dynamic_axes: Mapping[str, Mapping[int, str] | Sequence[int]] | None = None
    keep_initializers_as_inputs: bool | None = None
    custom_opsets: Mapping[str, int] | None = None
    export_modules_as_functions: bool | set[type[torch.nn.Module]] = False
<|endoftext|># mypy: allow-untyped-defs
"""This file exports ONNX ops for opset 20.

Note [ONNX Operators that are added/updated in opset 20]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-20-of-the-default-onnx-operator-set
New operators:
    AffineGrid
    ConstantOfShape
    DFT
    Gelu
    GridSample
    ImageDecoder
    IsInf
    IsNaN
    ReduceMax
    ReduceMin
    RegexFullMatch
    StringConcat
    StringSplit
"""

import functools

import torch.nn.functional as F
from torch import _C
from torch.onnx._internal.torchscript_exporter import (
    jit_utils,
    registration,
    symbolic_helper,
)


# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

__all__ = ["_grid_sampler", "_affine_grid_generator", "gelu"]


def convert_grid_sample_mode(mode_s):
    return (
        "linear" if mode_s == "bilinear" else "cubic" if mode_s == "bicubic" else mode_s
    )


_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=20)


@_onnx_symbolic("aten::grid_sampler")
@symbolic_helper.parse_args("v", "v", "i", "i", "b")
def _grid_sampler(
    g: jit_utils.GraphContext,
    input: _C.Value,
    grid: _C.Value,
    mode_enum: int,
    padding_mode_enum: int,
    align_corners: bool,
):
    mode_s = {v: k for k, v in F.GRID_SAMPLE_INTERPOLATION_MODES.items()}[mode_enum]  # type: ignore[call-arg, index]
    # mode string changes at https://onnx.ai/onnx/operators/text_diff_GridSample_16_20.html
    mode_s = convert_grid_sample_mode(mode_s)
    padding_mode_s = {v: k for k, v in F.GRID_SAMPLE_PADDING_MODES.items()}[  # type: ignore[call-arg, index]
        padding_mode_enum  # type: ignore[index]
    ]
    return g.op(
        "GridSample",
        input,
        grid,
        align_corners_i=int(align_corners),
        mode_s=mode_s,
        padding_mode_s=padding_mode_s,
    )


@_onnx_symbolic("aten::affine_grid_generator")
@symbolic_helper.parse_args("v", "v", "b")
def _affine_grid_generator(
    g: jit_utils.GraphContext,
    theta: _C.Value,
    size: _C.Value,
    align_corners: bool,
):
    return g.op(
        "AffineGrid",
        theta,
        size,
        align_corners_i=int(align_corners),
    )


@_onnx_symbolic("aten::gelu")
@symbolic_helper.parse_args("v", "s")
def gelu(g: jit_utils.GraphContext, self: _C.Value, approximate: str = "none"):
    return g.op("Gelu", self, approximate_s=approximate)
<|endoftext|># mypy: allow-untyped-defs
"""Utilities for converting and operating on ONNX, JIT and torch types."""

from __future__ import annotations

import enum
import typing
from typing import Literal

import torch
from torch._C import _onnx as _C_onnx
from torch.onnx import errors


if typing.TYPE_CHECKING:
    # Hack to help mypy to recognize torch._C.Value
    from torch import _C  # noqa: F401

ScalarName = Literal[
    "Byte",
    "Char",
    "Double",
    "Float",
    "Half",
    "Int",
    "Long",
    "Short",
    "Bool",
    "ComplexHalf",
    "ComplexFloat",
    "ComplexDouble",
    "QInt8",
    "QUInt8",
    "QInt32",
    "BFloat16",
    "Float8E5M2",
    "Float8E4M3FN",
    "Float8E5M2FNUZ",
    "Float8E4M3FNUZ",
    "Undefined",
]

TorchName = Literal[
    "bool",
    "uint8_t",
    "int8_t",
    "double",
    "float",
    "half",
    "int",
    "int64_t",
    "int16_t",
    "complex32",
    "complex64",
    "complex128",
    "qint8",
    "quint8",
    "qint32",
    "bfloat16",
    "float8_e5m2",
    "float8_e4m3fn",
    "float8_e5m2fnuz",
    "float8_e4m3fnuz",
]


class JitScalarType(enum.IntEnum):
    """Scalar types defined in torch.

    Use ``JitScalarType`` to convert from torch and JIT scalar types to ONNX scalar types.

    Examples:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
        >>> # xdoctest: +IGNORE_WANT("win32 has different output")
        >>> JitScalarType.from_value(torch.ones(1, 2)).onnx_type()
        TensorProtoDataType.FLOAT

        >>> JitScalarType.from_value(torch_c_value_with_type_float).onnx_type()
        TensorProtoDataType.FLOAT

        >>> JitScalarType.from_dtype(torch.get_default_dtype).onnx_type()
        TensorProtoDataType.FLOAT

    """

    # Order defined in https://github.com/pytorch/pytorch/blob/344defc9733a45fee8d0c4d3f5530f631e823196/c10/core/ScalarType.h
    UINT8 = 0
    INT8 = enum.auto()  # 1
    INT16 = enum.auto()  # 2
    INT = enum.auto()  # 3
    INT64 = enum.auto()  # 4
    HALF = enum.auto()  # 5
    FLOAT = enum.auto()  # 6
    DOUBLE = enum.auto()  # 7
    COMPLEX32 = enum.auto()  # 8
    COMPLEX64 = enum.auto()  # 9
    COMPLEX128 = enum.auto()  # 10
    BOOL = enum.auto()  # 11
    QINT8 = enum.auto()  # 12
    QUINT8 = enum.auto()  # 13
    QINT32 = enum.auto()  # 14
    BFLOAT16 = enum.auto()  # 15
    FLOAT8E5M2 = enum.auto()  # 16
    FLOAT8E4M3FN = enum.auto()  # 17
    FLOAT8E5M2FNUZ = enum.auto()  # 18
    FLOAT8E4M3FNUZ = enum.auto()  # 19
    UNDEFINED = enum.auto()  # 20

    @classmethod
    def _from_name(cls, name: ScalarName | TorchName | str | None) -> JitScalarType:
        """Convert a JIT scalar type or torch type name to ScalarType.

        Note: DO NOT USE this API when `name` comes from a `torch._C.Value.type()` calls.
            A "RuntimeError: INTERNAL ASSERT FAILED at "../aten/src/ATen/core/jit_type_base.h" can
            be raised in several scenarios where shape info is not present.
            Instead use `from_value` API which is safer.

        Args:
            name: JIT scalar type name (Byte) or torch type name (uint8_t).

        Returns:
            JitScalarType

        Raises:
           OnnxExporterError: if name is not a valid scalar type name or if it is None.
        """
        if name is None:
            raise errors.OnnxExporterError("Scalar type name cannot be None")
        if valid_scalar_name(name):
            return _SCALAR_NAME_TO_TYPE[name]  # type: ignore[index]
        if valid_torch_name(name):
            return _TORCH_NAME_TO_SCALAR_TYPE[name]  # type: ignore[index]

        raise errors.OnnxExporterError(f"Unknown torch or scalar type: '{name}'")

    @classmethod
    def from_dtype(cls, dtype: torch.dtype | None) -> JitScalarType:
        """Convert a torch dtype to JitScalarType.

        Note: DO NOT USE this API when `dtype` comes from a `torch._C.Value.type()` calls.
            A "RuntimeError: INTERNAL ASSERT FAILED at "../aten/src/ATen/core/jit_type_base.h" can
            be raised in several scenarios where shape info is not present.
            Instead use `from_value` API which is safer.

        Args:
            dtype: A torch.dtype to create a JitScalarType from

        Returns:
            JitScalarType

        Raises:
            OnnxExporterError: if dtype is not a valid torch.dtype or if it is None.
        """
        if dtype not in _DTYPE_TO_SCALAR_TYPE:
            raise errors.OnnxExporterError(f"Unknown dtype: {dtype}")
        # pyrefly: ignore [bad-index]
        return _DTYPE_TO_SCALAR_TYPE[dtype]

    @classmethod
    def from_onnx_type(
        cls, onnx_type: int | _C_onnx.TensorProtoDataType | None
    ) -> JitScalarType:
        """Convert a ONNX data type to JitScalarType.

        Args:
            onnx_type: A torch._C._onnx.TensorProtoDataType to create a JitScalarType from

        Returns:
            JitScalarType

        Raises:
            OnnxExporterError: if dtype is not a valid torch.dtype or if it is None.
        """
        if onnx_type not in _ONNX_TO_SCALAR_TYPE:
            raise errors.OnnxExporterError(f"Unknown onnx_type: {onnx_type}")
        # pyrefly: ignore [redundant-cast]
        return _ONNX_TO_SCALAR_TYPE[typing.cast(_C_onnx.TensorProtoDataType, onnx_type)]

    @classmethod
    def from_value(
        cls, value: torch._C.Value | torch.Tensor | None, default=None
    ) -> JitScalarType:
        """Create a JitScalarType from a value's scalar type.

        Args:
            value: An object to fetch scalar type from.
            default: The JitScalarType to return if a valid scalar cannot be fetched from value

        Returns:
            JitScalarType.

        Raises:
            OnnxExporterError: if value does not have a valid scalar type and default is None.
            SymbolicValueError: when value.type()'s info are empty and default is None
        """

        if not isinstance(value, (torch._C.Value, torch.Tensor)) or (
            isinstance(value, torch._C.Value) and value.node().mustBeNone()
        ):
            # default value of type JitScalarType is returned when value is not valid
            if default is None:
                raise errors.OnnxExporterError(
                    "value must be either torch._C.Value or torch.Tensor objects."
                )
            elif not isinstance(default, JitScalarType):
                raise errors.OnnxExporterError(
                    "default value must be a JitScalarType object."
                )
            return default

        # Each value type has their own way of storing scalar type
        if isinstance(value, torch.Tensor):
            return cls.from_dtype(value.dtype)
        if isinstance(value.type(), torch.ListType):
            try:
                return cls.from_dtype(value.type().getElementType().dtype())
            except RuntimeError:
                return cls._from_name(str(value.type().getElementType()))
        if isinstance(value.type(), torch._C.OptionalType):
            if value.type().getElementType().dtype() is None:
                if isinstance(default, JitScalarType):
                    return default
                raise errors.OnnxExporterError(
                    "default value must be a JitScalarType object."
                )
            return cls.from_dtype(value.type().getElementType().dtype())

        scalar_type = None
        if value.node().kind() != "prim::Constant" or not isinstance(
            value.type(), torch._C.NoneType
        ):
            # value must be a non-list torch._C.Value scalar
            scalar_type = value.type().scalarType()

        if scalar_type is not None:
            return cls._from_name(scalar_type)

        # When everything fails... try to default
        if default is not None:
            return default
        raise errors.SymbolicValueError(
            f"Cannot determine scalar type for this '{type(value.type())}' instance and "
            "a default value was not provided.",
            value,
        )

    def scalar_name(self) -> ScalarName:
        """Convert a JitScalarType to a JIT scalar type name."""
        return _SCALAR_TYPE_TO_NAME[self]

    def torch_name(self) -> TorchName:
        """Convert a JitScalarType to a torch type name."""
        return _SCALAR_TYPE_TO_TORCH_NAME[self]

    def dtype(self) -> torch.dtype:
        """Convert a JitScalarType to a torch dtype."""
        return _SCALAR_TYPE_TO_DTYPE[self]

    def onnx_type(self) -> _C_onnx.TensorProtoDataType:
        """Convert a JitScalarType to an ONNX data type."""
        if self not in _SCALAR_TYPE_TO_ONNX:
            raise errors.OnnxExporterError(
                f"Scalar type {self} cannot be converted to ONNX"
            )
        return _SCALAR_TYPE_TO_ONNX[self]

    def onnx_compatible(self) -> bool:
        """Return whether this JitScalarType is compatible with ONNX."""
        return (
            self in _SCALAR_TYPE_TO_ONNX
            and self != JitScalarType.UNDEFINED
            and self != JitScalarType.COMPLEX32
        )


def valid_scalar_name(scalar_name: ScalarName | str) -> bool:
    """Return whether the given scalar name is a valid JIT scalar type name."""
    return scalar_name in _SCALAR_NAME_TO_TYPE


def valid_torch_name(torch_name: TorchName | str) -> bool:
    """Return whether the given torch name is a valid torch type name."""
    return torch_name in _TORCH_NAME_TO_SCALAR_TYPE


# https://github.com/pytorch/pytorch/blob/344defc9733a45fee8d0c4d3f5530f631e823196/c10/core/ScalarType.h
_SCALAR_TYPE_TO_NAME: dict[JitScalarType, ScalarName] = {
    JitScalarType.BOOL: "Bool",
    JitScalarType.UINT8: "Byte",
    JitScalarType.INT8: "Char",
    JitScalarType.INT16: "Short",
    JitScalarType.INT: "Int",
    JitScalarType.INT64: "Long",
    JitScalarType.HALF: "Half",
    JitScalarType.FLOAT: "Float",
    JitScalarType.DOUBLE: "Double",
    JitScalarType.COMPLEX32: "ComplexHalf",
    JitScalarType.COMPLEX64: "ComplexFloat",
    JitScalarType.COMPLEX128: "ComplexDouble",
    JitScalarType.QINT8: "QInt8",
    JitScalarType.QUINT8: "QUInt8",
    JitScalarType.QINT32: "QInt32",
    JitScalarType.BFLOAT16: "BFloat16",
    JitScalarType.FLOAT8E5M2: "Float8E5M2",
    JitScalarType.FLOAT8E4M3FN: "Float8E4M3FN",
    JitScalarType.FLOAT8E5M2FNUZ: "Float8E5M2FNUZ",
    JitScalarType.FLOAT8E4M3FNUZ: "Float8E4M3FNUZ",
    JitScalarType.UNDEFINED: "Undefined",
}

_SCALAR_NAME_TO_TYPE: dict[ScalarName, JitScalarType] = {
    v: k for k, v in _SCALAR_TYPE_TO_NAME.items()
}

_SCALAR_TYPE_TO_TORCH_NAME: dict[JitScalarType, TorchName] = {
    JitScalarType.BOOL: "bool",
    JitScalarType.UINT8: "uint8_t",
    JitScalarType.INT8: "int8_t",
    JitScalarType.INT16: "int16_t",
    JitScalarType.INT: "int",
    JitScalarType.INT64: "int64_t",
    JitScalarType.HALF: "half",
    JitScalarType.FLOAT: "float",
    JitScalarType.DOUBLE: "double",
    JitScalarType.COMPLEX32: "complex32",
    JitScalarType.COMPLEX64: "complex64",
    JitScalarType.COMPLEX128: "complex128",
    JitScalarType.QINT8: "qint8",
    JitScalarType.QUINT8: "quint8",
    JitScalarType.QINT32: "qint32",
    JitScalarType.BFLOAT16: "bfloat16",
    JitScalarType.FLOAT8E5M2: "float8_e5m2",
    JitScalarType.FLOAT8E4M3FN: "float8_e4m3fn",
    JitScalarType.FLOAT8E5M2FNUZ: "float8_e5m2fnuz",
    JitScalarType.FLOAT8E4M3FNUZ: "float8_e4m3fnuz",
}

_TORCH_NAME_TO_SCALAR_TYPE: dict[TorchName, JitScalarType] = {
    v: k for k, v in _SCALAR_TYPE_TO_TORCH_NAME.items()
}

_SCALAR_TYPE_TO_ONNX = {
    JitScalarType.BOOL: _C_onnx.TensorProtoDataType.BOOL,
    JitScalarType.UINT8: _C_onnx.TensorProtoDataType.UINT8,
    JitScalarType.INT8: _C_onnx.TensorProtoDataType.INT8,
    JitScalarType.INT16: _C_onnx.TensorProtoDataType.INT16,
    JitScalarType.INT: _C_onnx.TensorProtoDataType.INT32,
    JitScalarType.INT64: _C_onnx.TensorProtoDataType.INT64,
    JitScalarType.HALF: _C_onnx.TensorProtoDataType.FLOAT16,
    JitScalarType.FLOAT: _C_onnx.TensorProtoDataType.FLOAT,
    JitScalarType.DOUBLE: _C_onnx.TensorProtoDataType.DOUBLE,
    JitScalarType.COMPLEX64: _C_onnx.TensorProtoDataType.COMPLEX64,
    JitScalarType.COMPLEX128: _C_onnx.TensorProtoDataType.COMPLEX128,
    JitScalarType.BFLOAT16: _C_onnx.TensorProtoDataType.BFLOAT16,
    JitScalarType.UNDEFINED: _C_onnx.TensorProtoDataType.UNDEFINED,
    JitScalarType.COMPLEX32: _C_onnx.TensorProtoDataType.UNDEFINED,
    JitScalarType.QINT8: _C_onnx.TensorProtoDataType.INT8,
    JitScalarType.QUINT8: _C_onnx.TensorProtoDataType.UINT8,
    JitScalarType.QINT32: _C_onnx.TensorProtoDataType.INT32,
    JitScalarType.FLOAT8E5M2: _C_onnx.TensorProtoDataType.FLOAT8E5M2,
    JitScalarType.FLOAT8E4M3FN: _C_onnx.TensorProtoDataType.FLOAT8E4M3FN,
    JitScalarType.FLOAT8E5M2FNUZ: _C_onnx.TensorProtoDataType.FLOAT8E5M2FNUZ,
    JitScalarType.FLOAT8E4M3FNUZ: _C_onnx.TensorProtoDataType.FLOAT8E4M3FNUZ,
}

