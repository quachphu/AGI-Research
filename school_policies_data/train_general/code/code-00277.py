        return artifact_cls

    @classmethod
    def _get_artifact_type(cls, artifact_type_key: str) -> type[CacheArtifact]:
        if artifact_type_key not in cls._artifact_types:
            raise AssertionError(
                f"Artifact of type={artifact_type_key} not registered in mega-cache artifact factory"
            )
        return cls._artifact_types[artifact_type_key]

    @classmethod
    def create(cls, artifact_type_key: str, key: str, content: bytes) -> CacheArtifact:
        artifact_cls = cls._get_artifact_type(artifact_type_key)
        # pyrefly: ignore [bad-instantiation]
        return artifact_cls(key, content)

    @classmethod
    def encode_create(
        cls, artifact_type_key: str, key: str, content: Any
    ) -> CacheArtifact:
        artifact_cls = cls._get_artifact_type(artifact_type_key)
        # pyrefly: ignore [bad-instantiation]
        return artifact_cls(key, artifact_cls.encode(content))


@dataclasses.dataclass
class CacheInfo:
    """
    Return value of serialization and deserialization for the purpose of
    instrumentation
    """

    artifacts: defaultdict[str, list[str]] = dataclasses.field(
        default_factory=lambda: defaultdict(list)
    )

    # Methods set by CacheArtifactFactory.register based on CacheArtifact.type()
    @property
    def inductor_artifacts(self) -> list[str]:  # type: ignore[empty-body]
        ...

    @property
    def autotune_artifacts(self) -> list[str]:  # type: ignore[empty-body]
        ...

    @property
    def aot_autograd_artifacts(self) -> list[str]:  # type: ignore[empty-body]
        ...

    @property
    def pgo_artifacts(self) -> list[str]:  # type: ignore[empty-body]
        ...

    @property
    def precompile_artifacts(self) -> list[str]:  # type: ignore[empty-body]
        ...

    def add(self, artifact: CacheArtifact) -> None:
        self.artifacts[artifact.type()].append(artifact.key)

    def clear(self) -> None:
        self.artifacts.clear()

    def empty(self) -> bool:
        return not self.artifacts


def _serialize_single_cache(
    writer: BytesWriter, cls: "tuple[str, list[CacheArtifact]]"
) -> None:
    writer.write_str(cls[0])
    writer.write_uint64(len(cls[1]))
    for artifact in cls[1]:
        CacheArtifact.serialize(writer, artifact)


def _deserialize_single_cache(
    reader: BytesReader,
) -> "tuple[str, list[CacheArtifact]]":
    artifacts = []
    artifact_type_key = reader.read_str()
    num_artifacts = reader.read_uint64()
    for _ in range(num_artifacts):
        artifacts.append(CacheArtifact.deserialize(artifact_type_key, reader))

    return artifact_type_key, artifacts


CacheArtifactsResult = dict[str, list[CacheArtifact]]


@dataclasses.dataclass(frozen=True)
class CacheArtifactRecorder:
    """
    Shared entry point for cache implementations to record artifacts.
    """

    artifact_type: str
    key: str

    def record(self, content: Any) -> None:
        CacheArtifactManager.record_artifact(self.artifact_type, self.key, content)

    def record_if_present(self, content: Any | None) -> None:
        if content is not None:
            self.record(content)


class CacheArtifactManager:
    """
    Lightweight manager class for collecting and processing cache artifacts for
    hot loading

    Intended Lifecycle:
    - Execute code via torch.compile, this will call
        CacheArtifactRecorder.record on each cache artifact
    - Call CacheArtifactManager.serialize to convert all the cache artifacts
        to portable format
    - Call CacheArtifactManager.deserialize to hot load the cache artifacts on
        a potentially different process

    NOTE: There's no FB/FC guarantees, results of cache artifacts will not be
          used unless code version matches.
    """

    # Protected by the compile_lock
    _new_cache_artifacts: CacheArtifactsResult = defaultdict(list)
    # Keep a separate seen artifacts list to avoid unnecessary duplicates
    # This list will not be cleared between serialize() calls
    _seen_artifacts: OrderedSet[CacheArtifact] = OrderedSet()
    # When serialize() is called, artifacts are transferred from _cache_artifacts to
    # internal data structure of the _serializer
    # This allows us to only pay the cost of serialization if serialize() is called
    _serializer: AppendingByteSerializer[tuple[str, list[CacheArtifact]]] = (
        AppendingByteSerializer(serialize_fn=_serialize_single_cache)
    )
    _cache_info: CacheInfo = CacheInfo()

    @classmethod
    def clear(cls) -> None:
        cls._new_cache_artifacts.clear()
        cls._seen_artifacts.clear()
        cls._serializer.clear()
        cls._cache_info.clear()

    @classmethod
    @contextmanager
    def with_fresh_cache(cls) -> Generator[None, None, None]:
        original_new_cache_artifacts = cls._new_cache_artifacts
        original_seen_artifacts = cls._seen_artifacts
        original_serializer = cls._serializer
        original_cache_info = cls._cache_info

        cls._new_cache_artifacts = defaultdict(list)
        cls._seen_artifacts = OrderedSet()
        cls._serializer = AppendingByteSerializer(serialize_fn=_serialize_single_cache)
        cls._cache_info = cls._cache_info.__class__()
        try:
            yield
        finally:
            cls._new_cache_artifacts = original_new_cache_artifacts
            cls._seen_artifacts = original_seen_artifacts
            cls._serializer = original_serializer
            cls._cache_info = original_cache_info

    @classmethod
    def record_artifact(
        cls,
        artifact_type: str,
        key: str,
        content: Any,
    ) -> None:
        """
        Called from each caching operation to record the artifact in this
        "mega" list
        """
        artifact = CacheArtifactFactory.encode_create(artifact_type, key, content)
        if artifact in cls._seen_artifacts:
            return
        log.debug("Recording %s", artifact)
        cls._new_cache_artifacts[artifact_type].append(artifact)
        cls._seen_artifacts.add(artifact)

    @classmethod
    def need_serialize(cls) -> bool:
        """
        Have we seen new artifacts since last serialize call?
        """
        return len(cls._new_cache_artifacts) != 0

    @classmethod
    def serialize(cls) -> tuple[bytes, CacheInfo] | None:
        """
        Converts the "mega" list into portable format
        """
        for artifact in chain(*cls._new_cache_artifacts.values()):
            log.debug("saving: %s", artifact)
            cls._cache_info.add(artifact)

        if cls._cache_info.empty():
            # If there are no artifacts, don't just return bytes with
            # version.
            return None

        try:
            # We deep copy cls._cache_info since later compilations
            # can keep adding to cache_info
            info = copy.deepcopy(cls._cache_info)
            cls._serializer.extend(cls._new_cache_artifacts.items())
            artifact_bytes = cls._serializer.to_bytes()
            cls._new_cache_artifacts.clear()
            return artifact_bytes, info
        except Exception:
            log.warning("Failed to pickle cache artifacts", exc_info=True)
        return None

    @staticmethod
    def deserialize(serialized_artifacts: bytes) -> CacheArtifactsResult | None:
        """
        Converts the portable format back into CacheArtifacts
        """
        try:
            CacheArtifactManager._ensure_cache_artifacts_registered()
            artifacts = dict(
                AppendingByteSerializer.to_list(
                    serialized_artifacts,
                    deserialize_fn=_deserialize_single_cache,
                )
            )
        except Exception:
            log.warning("Failed to un-pickle cache artifacts", exc_info=True)
            return None

        return artifacts

    @staticmethod
    def populate_caches(artifacts: CacheArtifactsResult) -> CacheInfo:
        info = CacheInfo()
        for artifact in chain(*artifacts.values()):
            log.debug("writing: %s", artifact)
            info.add(artifact)
            artifact.populate_cache()

        return info

    @classmethod
    def _ensure_cache_artifacts_registered(cls) -> None:
        """When deserializing caches in fresh process, we need to ensure that all
        cache artifacts are registered in the cache registry. This is done by
        simply importing all the cache artifacts already wrapped with register call.
        """
        from torch._dynamo.package import PrecompileCacheArtifact  # noqa: F401
        from torch._dynamo.pgo import PGOCacheArtifact  # noqa: F401
        from torch._functorch._aot_autograd.autograd_cache import (  # noqa: F401
            AOTAutogradCacheArtifact,
        )
        from torch._inductor.codecache import InductorCacheArtifact  # noqa: F401
        from torch._inductor.runtime.autotune_cache import (  # noqa: F401
            AutotuneCacheArtifact,
        )
<|endoftext|># mypy: allow-untyped-defs
import contextlib
import io
from collections.abc import Callable
from typing import Any, TYPE_CHECKING, TypeVar
from typing_extensions import ParamSpec

import torch
from torch._higher_order_ops.invoke_subgraph import NestedCompileRegionOptions

# ``torch.compiler.precompile``: make_fx AOT capture -> self-contained Python source
# plus an acceleration cache. Re-exported from the private impl module, whose
# ``_PrecompileApi.__module__`` is forced to "torch.compiler" so this is the single
# public location. Distinct from ``torch._dynamo.config.caching_precompile`` (a
# ``torch.compile`` guard-serialization caching mode), despite the shared word.
# ``PrecompileError`` is also re-exported here as ``torch.compiler.PrecompileError`` so the
# conventional ``except torch.compiler.PrecompileError`` works; its ``__module__`` is already
# forced to "torch.compiler" in the impl module, matching this public location.
from torch._precompile import (
    precompile as precompile,
    PrecompileError as PrecompileError,
)

from . import config
from ._cache import CacheInfo


if TYPE_CHECKING:
    from torch._dynamo.eval_frame import StanceStr


__all__ = [
    "compile",
    "config",
    "assume_constant_result",
    "reset",
    "nonstrict_trace",
    "allow_in_graph",
    "substitute_in_graph",
    "list_backends",
    "disable",
    "set_default_backend",
    "get_default_backend",
    "set_stance",
    "set_enable_guard_collectives",
    "cudagraph_mark_step_begin",
    "cudagraph_mark_warmup_incomplete",
    "load_compiled_function",
    "precompile",
    "PrecompileError",
    "wrap_numpy",
    "is_compiling",
    "is_dynamo_compiling",
    "is_exporting",
    "save_cache_artifacts",
    "load_cache_artifacts",
    "keep_portable_guards_unsafe",
    "skip_guard_on_inbuilt_nn_modules_unsafe",
    "skip_guard_on_all_nn_modules_unsafe",
    "keep_tensor_guards_unsafe",
    "skip_guard_on_globals_unsafe",
    "skip_all_guards_unsafe",
    "nested_compile_region",
]


_P = ParamSpec("_P")
_R = TypeVar("_R")
FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)


def compile(*args, **kwargs):
    """
    See :func:`torch.compile` for details on the arguments for this function.
    """
    # pyrefly: ignore [not-iterable]
    return torch.compile(*args, **kwargs)


def reset() -> None:
    """
    Reset the in-process compiler state.

    This function clears Dynamo's in-memory compilation caches and related
    process-local state used by :func:`torch.compile`. It does not delete
    filesystem caches, such as Inductor's disk cache.
    """
    import torch._dynamo

    torch._dynamo.reset()


def allow_in_graph(fn):
    """
    Tells the compiler frontend (Dynamo) to skip symbolic introspection of the function
    and instead directly write it to the graph when encountered.

    If you are using :func:`torch.compile` (with backend="inductor" (the default)), or
    :func:`torch.export.export`, and trying to black-box a Python function throughout
    all tracing, do not use this API.
    Instead, please create a custom operator (see `PyTorch Custom Operators Landing Page
    <https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html>`_)

    .. warning::

        If you're a typical torch.compile user (e.g. you're applying torch.compile to
        a model to make it run faster), you probably don't want to use this function.
        :func:`allow_in_graph` is a footgun because it skips the compiler frontend
        (Dynamo) that is responsible for doing safety checks (graph breaks, handling
        closures, etc). Incorrect usage will lead to difficult-to-debug silent
        incorrectness issues.

    Given a Python function with no allow_in_graph decorator, regular execution
    of torch.compile traces through the function. :func:`allow_in_graph` changes
    it so that the frontend does not trace inside the function, but the compiler
    backend still traces through it. Compare this to custom operators, which
    treats a function as a black box throughout the torch.compile stack. The following
    table compares these mechanisms.

    +------------------------+-----------------------+--------------------------------+
    | Mechanism              | Frontend (Dynamo)     | Backend (AOTAutograd+Inductor) |
    +========================+=======================+================================+
    | no decorator           | trace inside          | trace inside                   |
    +------------------------+-----------------------+--------------------------------+
    | allow_in_graph         | opaque callable       | trace inside                   |
    +------------------------+-----------------------+--------------------------------+
    | custom op              | opaque callable       | opaque callable                |
    +------------------------+-----------------------+--------------------------------+

    One common use case for :func:`allow_in_graph()` is as an escape hatch for the compiler
    frontend: if you know the function works w.r.t. to the downstream components of the
    compilation stack (AOTAutograd and Inductor) but there is a Dynamo bug that prevents it from
    symbolically introspecting the function properly (or if your code is in C/C++ and
    therefore cannot be introspected with Dynamo), then one can decorate said function
    with :func:`allow_in_graph` to bypass Dynamo.

    We require that ``fn`` adhere to the following restrictions. Failure to adhere
    results in undefined behavior:

    - The inputs to ``fn`` must be Proxy-able types in the FX graph. Valid types include:
      Tensor/int/bool/float/None/List[Tensor?]/List[int?]/List[float?]
      Tuple[Tensor?, ...]/Tuple[int?, ...]/Tuple[float?, ...]/torch.dtype/torch.device
    - The outputs to ``fn`` must be Proxy-able types in the FX graph (see previous bullet)
    - all Tensors used inside of ``fn`` must be passed directly as inputs to ``fn``
      (as opposed to being captured variables).

    See also :func:`nonstrict_trace()`, which has slightly fewer restrictions on the inputs.

    Args:
        fn: A callable representing the function to be included in the graph.
            If ``fn`` is a list or tuple of callables it recursively applies
            :func:`allow_in_graph()` to each function and returns a new list or
            tuple containing the modified functions.

    Example::

        torch.compiler.allow_in_graph(my_custom_function)


        @torch.compile(...)
        def fn(x):
            x = torch.add(x, 1)
            x = my_custom_function(x)
            x = torch.add(x, 1)
            return x


        fn(...)

    Will capture a single graph containing ``my_custom_function()``.

    """
    import torch._dynamo

    return torch._dynamo.allow_in_graph(fn)


def nonstrict_trace(traceable_fn: Callable[_P, _R]) -> Callable[_P, _R]:
    """
    Decorator to mark a function as nonstrict-traceable for dynamo.

    A nonstrict-traced function appears as an opaque call in the dynamo graph.
    Dynamo does not trace into the function body (hence the "nonstrict"), but
    aot_autograd will trace into it.

    This is similar to ``allow_in_graph`` but with enhanced support for:
    - User-defined classes as inputs (must be registered with pytree)
    - ``nn.Module`` as input arguments (parameters and buffers are tracked for autograd)
    - Global/captured tensors treated as constants (assumed not updated during execution)

    Note:
        - With ``backend="eager"``, the original Python function runs directly.
          With ``backend="aot_eager"``, the graph traced by aot_autograd runs.
          With ``backend="inductor"``, the traced graph is compiled with inductor.

        - Training is supported: you can call ``.backward()`` on outputs and gradients
          will flow through the nonstrict-traced function.

    Dangerous patterns (may cause silent incorrectness):
        - Side effects between nonstric_trace'd fn and compiled region: The function should
          not depend on variables mutated by other code inside the compiled function, and code
          after the call should not depend on mutations made by it.

        - Implicit inputs (closures/globals): Tensors captured from enclosing scopes
          are treated as constants. Gradients will NOT flow back to them. Pass tensors
          as explicit arguments if gradients are needed.

    Restrictions:
        - Both inputs and outputs must use pytree-compatible types. User-defined classes
          must be registered via :func:`torch.utils._pytree.register_pytree_node`,
          :func:`torch.utils._pytree.register_dataclass`, or
          :func:`torch.utils._pytree.register_constant`. Tensors, Python primitives (int, float, bool, str),
          symbolic types (SymInt, SymFloat, SymBool), and built-in containers (list,
          tuple, dict) are already handled by default.
        - Primitive values and container structure are specialized per call site:
          each call site expects the same primitives and structure on every execution.

    Example::

        >>> import torch
        >>> @torch.compiler.nonstrict_trace
        ... def traced_forward(model, x):
        ...     # It's OK to have dynamo graph break within nonstrict_trace region
        ...     torch._dynamo.graph_break()
        ...     return model(x) + x
        ...
        >>> class MyModule(torch.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.inner = torch.nn.Linear(10, 10)
        ...
        ...     def forward(self, x):
        ...         return traced_forward(self.inner, x)
        ...
        >>> # Compile and run
        >>> model = MyModule()
        >>> opt_model = torch.compile(model, backend="aot_eager", fullgraph=True)
        >>> out = opt_model(torch.randn(10, 10))
        >>> out.sum().backward()  # Gradients flow through traced_forward

    """
    import torch._dynamo

    return torch._dynamo.nonstrict_trace(traceable_fn)


def substitute_in_graph(
    original_fn: Callable[_P, _R],
    *,
    can_constant_fold_through: bool = False,
    skip_signature_check: bool = False,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """
    Register a polyfill handler for a function, usually a C function from the C extension, to be
    used in place of the original function when inlining the original function in the graph.

    .. note::

        The polyfill handler is only used when inlining the original function. It is not used when
        the original function is called directly. In the eager mode, the decorated function calls
        the performant C function rather than the polyfill handler.

    The polyfill handler is a function that will be called in place of the original function when
    inlining the original function. The polyfill handler should have the same signature and the same
    behavior as the original function.

    Args:
        original_fn (callable): The original function, usually a C function, to register a polyfill
            handler for.
        can_constant_fold_through (bool, optional): Whether the polyfill handler can be constant
            folded through. That is, if the polyfill handler is a pure function and its arguments
            are constant, the result of the polyfill handler can be constant folded during the
            compilation. Defaults to ``False``.
        skip_signature_check (bool, optional): Whether to skip the signature check between the
            original function and the polyfill handler. Defaults to ``False``.

    Returns:
        A decorator that registers the polyfill handler for the original function.

    Example::

        >>> import operator
        >>> operator.indexOf([1, 2, 3, 4, 5], 3)
        2
        >>> torch.compile(operator.indexOf, fullgraph=True)([1, 2, 3, 4, 5], 3)
        ... # xdoctest: +SKIP("Long tracebacks")
        Traceback (most recent call last):
        ...
        torch._dynamo.exc.Unsupported: ...

        >>> @torch.compiler.substitute_in_graph(operator.indexOf)
        ... def indexOf(a, b, /):
        ...     for i, item in enumerate(a):
        ...         if item is b or item == b:
        ...             return i
        ...     raise ValueError("sequence.index(x): x not in sequence")
        >>>
        >>> torch.compile(operator.indexOf, fullgraph=True)([1, 2, 3, 4, 5], 3)
        2
    """
    import torch._dynamo

    return torch._dynamo.substitute_in_graph(
        original_fn,
        can_constant_fold_through=can_constant_fold_through,
        skip_signature_check=skip_signature_check,
    )


def list_backends(exclude_tags=("debug", "experimental")) -> list[str]:
    """
    Return valid strings that can be passed to `torch.compile(..., backend="name")`.

    Args:
        exclude_tags(optional): A tuple of strings representing tags to exclude.
    """
    import torch._dynamo

    return torch._dynamo.list_backends(exclude_tags)


def assume_constant_result(fn):
    """
    This function is used to mark a function `fn` as having a constant result.
    This allows the compiler to optimize away your function.
    Returns The same function `fn`

    Args:
        fn: The function to be marked as having a constant result.

    .. warning::
        `assume_constant_result` can, if invalid, cause safety and soundness issues, :func:`torch.compile`
        will not attempt to validate whether the constant assumption is true or not

    """
    import torch._dynamo

    return torch._dynamo.assume_constant_result(fn)


def disable(fn=None, recursive=True, *, reason=None):
    """
    This function provides a decorator to disable compilation on a function.
    It also provides the option of recursively disabling called functions.

    Args:
        fn (optional): The function to disable
        recursive (optional): A boolean value indicating whether the disabling should be recursive.
        reason (optional): A string value indicating the reason for disabling the function.
    """
    import torch._dynamo

    return torch._dynamo.disable(fn, recursive, reason=reason)


def set_default_backend(backend: str | Callable[..., Any] | None) -> None:
    """Set the default backend for ``torch.compile`` when no ``backend`` argument is specified.

    Passing ``None`` resets the default back to ``"inductor"``.

    Args:
        backend: A backend name (string), a callable backend, or ``None``.

    Example::

        >>> torch.compiler.set_default_backend("eager")
        >>> torch.compiler.get_default_backend()
        'eager'
        >>> torch.compiler.set_default_backend(None)  # reset
        >>> torch.compiler.get_default_backend()
        'inductor'
    """
    from torch._dynamo.backends.registry import set_default_backend

    set_default_backend(backend)


def get_default_backend() -> str | Callable[..., Any]:
    """Return the current default backend for ``torch.compile``.

    Returns:
        The current default backend (string or callable). Initially ``"inductor"``.
    """
    from torch._dynamo.backends.registry import get_default_backend

    return get_default_backend()


def set_stance(
    stance: "StanceStr" = "default",
    *,
    skip_guard_eval_unsafe: bool = False,
    force_backend: str | Callable[..., Any] | None = None,
):
    """
    Set the current stance of the compiler.
    Can be used as a function, context manager, or decorator.
    Do not use this function inside a `torch.compile` region - an error will be raised otherwise.

    .. code-block:: python

        @torch.compile
        def foo(x): ...


        @torch.compiler.set_stance("force_eager")
        def bar():
            # will not be compiled
            foo(...)


        bar()

        with torch.compiler.set_stance("force_eager"):
            # will also not be compiled
            foo(...)

        torch.compiler.set_stance("force_eager")
        # will also not be compiled
        foo(...)
        torch.compiler.set_stance("default")

        # will be compiled
        foo(...)

    Args:
        stance: The stance to set the compiler to. Valid values are:

            - "default": The default stance, used for normal compilation.
            - "force_eager": Ignore all `torch.compile` directives.
            - "eager_on_recompile": Run code eagerly when a recompile is necessary.
              If there is cached compiled code valid for the input, it will still be used.
            - "fail_on_recompile": Raise an error when recompiling a function.
            - "eager_then_compile": Run the first invocation in eager mode, then compile on
              subsequent calls. This is beneficial for dynamic shapes as it allows inferring
              dynamism from the first two invocations instead of wasting a static compile on
              the first invocation.
            - "aot_eager_then_compile": Run the first invocation with AOT eager to get memory
              benefits from activation checkpointing, then compile on subsequent calls. Like
              eager_then_compile, this improves handling of dynamic shapes by avoiding an
              initial static compile.


        skip_guard_eval_unsafe: A flag to run only differentiating guards.
            CAUTION - This flag is unsafe and should only be used if your setup
            meets the following conditions.

            torch.compile uses a guard system to support recompilations and
            choose which compiled artifact to run at runtime.  These guards,
            though efficient, add some overhead, which may impact performance in
            scenarios where you need to optimize for minimal guard processing
            time.  This API enables you to disable guard evaluation, assuming
            that you have warmed up the compiled model with a sufficient variety
            of inputs. This assumption means that, after the warmup phase, no
            further recompilations will be necessary.  If this assumption fails,
            there is a risk of silently producing incorrect results (hence the
            term "unsafe" in the API name).

        force_backend: If `stance` is "default", this argument can be used to force `torch.compile`
            to use a specific backend. Otherwise, an error is raised.
    """
    import torch._dynamo

    return torch._dynamo.set_stance(
        stance,
        skip_guard_eval_unsafe=skip_guard_eval_unsafe,
        force_backend=force_backend,
    )


# forbid in graph
set_stance._dynamo_forbidden = True  # type: ignore[attr-defined]


def set_enable_guard_collectives(enabled: bool):
    """
    Enables use of collectives *during* guard evaluation to synchronize behavior
    across ranks.  This is expensive: we have to issue a collective every time
    we enter a compiled code region, even if no rank actually would need to
    compile.  This can help prevent NCCL hangs by ensuring that we never have a
    situation where one rank starts recompiling while other ranks don't compile;
    it is especially useful in conjunction with enable_compiler_collectives
    where such a situation would immediately cause a hang (as it is necessary
    for all ranks to compile at the same time to run compiler collectives).  Like
    compiler collectives, you can only run this on SPMD programs; you will hang
    otherwise.  Note that a guard collective is only issued if there is any
    compiled code to guard on; if this is the first time we encounter a frame or
    the frame is skipped, we don't issue collectives.

    Returns the previous setting of enabled.
    """
    from torch._C._dynamo.eval_frame import set_guard_complete_hook
    from torch._dynamo.eval_frame import guard_collectives_hook

    if enabled:
        return set_guard_complete_hook(guard_collectives_hook) is not None  # type: ignore[arg-type]
    else:
        return set_guard_complete_hook(None) is not None


set_enable_guard_collectives._dynamo_forbidden = True  # type: ignore[attr-defined]


def cudagraph_mark_step_begin():
    """
    Indicates that a new iteration of inference or training is about to begin.

    CUDA Graphs will free tensors of a prior iteration. A new iteration is started on each invocation of
    torch.compile, so long as there is not a pending backward that has not been called.

    If that heuristic is wrong, such as in the following example, manually mark it with this api.

    .. code-block:: python

        @torch.compile(mode="reduce-overhead")
        def rand_foo():
            return torch.rand([4], device="cuda")


        for _ in range(5):
            torch.compiler.cudagraph_mark_step_begin()
            rand_foo() + rand_foo()

    For more details, see `torch.compiler_cudagraph_trees <https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html>`__  # noqa: B950
    """
    from torch._inductor import cudagraph_trees

    cudagraph_trees.mark_step_begin()


def cudagraph_mark_warmup_incomplete():
    """Request another warmup for the active CUDA Graph Trees function.

    Call this synchronously from an autotuner or other code running during CUDA
    Graph Trees warmup when the current function needs another warmup iteration.
    The function will run eagerly again on its next invocation instead of being
    recorded. This is a no-op outside CUDA Graph Trees warmup, including during
    recording and replay or when CUDA Graph Trees are disabled.
    """
    from torch._inductor import cudagraph_trees

    cudagraph_trees.mark_warmup_incomplete()


def wrap_numpy(fn):
    r"""Decorator that turns a function from ``np.ndarray``\ s to ``np.ndarray``\ s into a function
    from ``torch.Tensor``\ s to ``torch.Tensor``\ s.

    It is designed to be used with :func:`torch.compile` with ``fullgraph=True``. It allows you to
    compile a NumPy function as if it were a PyTorch function. This allows you to run NumPy code
    on CUDA or compute its gradients.

    .. note::

        This decorator does not work without :func:`torch.compile`.

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> # Compile a NumPy function as a Tensor -> Tensor function
        >>> @torch.compile(fullgraph=True)
        >>> @torch.compiler.wrap_numpy
        >>> def fn(a: np.ndarray):
        >>>     return np.sum(a * a)
        >>> # Execute the NumPy function using Tensors on CUDA and compute the gradients
        >>> x = torch.arange(6, dtype=torch.float32, device="cuda", requires_grad=True)
        >>> out = fn(x)
        >>> out.backward()
        >>> print(x.grad)
        tensor([ 0.,  2.,  4.,  6.,  8., 10.], device='cuda:0')
    """
    from torch._dynamo.external_utils import wrap_numpy as wrap

    return wrap(fn)


_is_compiling_flag: bool = False
_is_exporting_flag: bool = False
_is_non_strict_tracing_flag: bool = False


def is_compiling() -> bool:
    """
    Indicates whether a graph is executed/traced as part of torch.compile() or torch.export().

    Note that there are 2 other related flags that should deprecated eventually:
      * torch._dynamo.external_utils.is_compiling()
      * torch._utils.is_compiling()

    Example::

        >>> def forward(self, x):
        >>>     if not torch.compiler.is_compiling():
        >>>        pass # ...logic that is not needed in a compiled/traced graph...
        >>>
        >>>     # ...rest of the function...
    """
    if torch.jit.is_scripting():
        return False
    else:
        return _is_compiling_flag


def _is_non_strict_tracing() -> bool:
    """
    Indicates whether we are inside a non-strict make_fx-based tracing session.
    """
    return _is_non_strict_tracing_flag


@contextlib.contextmanager
def _non_strict_tracing_context():
    """Context manager that sets the non-strict tracing flag."""
    global _is_non_strict_tracing_flag
    old = _is_non_strict_tracing_flag
    try:
        _is_non_strict_tracing_flag = True
        yield
    finally:
        _is_non_strict_tracing_flag = old


@contextlib.contextmanager
def _compile_session_context():
    """Context manager that sets _is_compiling_flag for the duration of a
    torch.compile session.
    """
    global _is_compiling_flag
    old = _is_compiling_flag
    try:
        _is_compiling_flag = True
        yield
    finally:
        _is_compiling_flag = old


@contextlib.contextmanager
def _patch_autograd_grad():
    """Patch autograd.grad for non-strict make_fx tracing.

    This patch installs autograd hooks so traced backward nodes preserve
    stack trace, seq_nr, and autograd_backward metadata before delegating to
    the real torch.autograd.grad.
    """
    import functools

    import torch.autograd
    from torch._dynamo.utils import warn_once
    from torch._functorch._aot_autograd.logging_utils import (
        setup_stacktrace_preservation_hooks_from_tensors,
    )

    warn_once(
        "torch.compiler._patch_autograd_grad() is deprecated; "
        "use torch.compiler._patch_engine_backward() instead."
    )
    # TODO: Remove this helper once Titan no longer depends on it.

    _orig_grad = torch.autograd.grad

    @functools.wraps(_orig_grad)
    def _patched_grad(outputs, inputs, *args, **kwargs):
        if not _is_non_strict_tracing():
            raise AssertionError(
                "_patch_autograd_grad() must be used under "
                "_non_strict_tracing_context()"
            )

        setup_stacktrace_preservation_hooks_from_tensors(outputs)
        return _orig_grad(outputs, inputs, *args, **kwargs)

    torch.autograd.grad = _patched_grad
    try:
        yield
    finally:
        torch.autograd.grad = _orig_grad


@contextlib.contextmanager
def _patch_engine_backward():
    """Patch _engine_run_backward for non-strict make_fx tracing.

    This patch installs autograd hooks so traced backward nodes preserve
    stack trace, seq_nr, and autograd_backward metadata before delegating to
    the real autograd engine entrypoint used by backward().
    """
    import functools

    import torch.autograd
    import torch.autograd.graph
    from torch._functorch._aot_autograd.logging_utils import (
        setup_stacktrace_preservation_hooks_from_tensors,
    )

    _orig_engine_run_backward = torch.autograd.graph._engine_run_backward

    @functools.wraps(_orig_engine_run_backward)
    def _patched_engine_backward(outputs, *args, **kwargs):
        if not _is_non_strict_tracing():
            raise AssertionError(
                "_patch_engine_backward() must be used under "
                "_non_strict_tracing_context()"
            )

        setup_stacktrace_preservation_hooks_from_tensors(outputs)
        return _orig_engine_run_backward(outputs, *args, **kwargs)

    torch.autograd.graph._engine_run_backward = _patched_engine_backward
    torch.autograd._engine_run_backward = _patched_engine_backward
    try:
        yield
    finally:
        torch.autograd.graph._engine_run_backward = _orig_engine_run_backward
        torch.autograd._engine_run_backward = _orig_engine_run_backward


def is_dynamo_compiling() -> bool:
    """
    Indicates whether a graph is traced via TorchDynamo.

    It's stricter than is_compiling() flag, as it would only be set to True when
    TorchDynamo is used.

    Example::

        >>> def forward(self, x):
        >>>     if not torch.compiler.is_dynamo_compiling():
        >>>        pass # ...logic that is not needed in a TorchDynamo-traced graph...
        >>>
        >>>     # ...rest of the function...
    """
    return False


def is_exporting() -> bool:
    """
    Indicates whether we're under exporting.

    It's stricter than is_compiling() flag, as it would only be set to True when
    torch.export is used.

    Example::

        >>> def forward(self, x):
        >>>     if not torch.compiler.is_exporting():
        >>>        pass # ...logic that is not needed in export...
        >>>
        >>>     # ...rest of the function...
    """
    return _is_exporting_flag


def save_cache_artifacts() -> tuple[bytes, CacheInfo] | None:
    """
    Serializes all the cache artifacts that were created during the compilation

    Example:

    - Execute torch.compile
    - Call torch.compiler.save_cache_artifacts()
    """
    from ._cache import CacheArtifactManager

    if torch._dynamo.config.caching_precompile:
        from torch._dynamo.precompile_context import PrecompileContext

        PrecompileContext.save_to_dynamo_cache()

    return CacheArtifactManager.serialize()


def load_cache_artifacts(serialized_artifacts: bytes) -> CacheInfo | None:
    """
    Hot loads cache artifacts that were previously serialized via
    save_cache_artifacts

    Example:

    # From a previous invocation
    artifacts = torch.compiler.save_cache_artifacts()

    torch.compiler.load_cache_artifacts(artifacts[0])
    """
    from ._cache import CacheArtifactManager, CacheInfo

    artifacts = CacheArtifactManager.deserialize(serialized_artifacts)
    if artifacts is not None:
        return CacheArtifactManager.populate_caches(artifacts)
    return None


def keep_portable_guards_unsafe(guard_entries):
    """
    A common function to only keep guards that can be used in both Python and non-Python environments.
    This includes:
    - Tensor metadata and dynamic shape information.
    - Global contexts state (e.g. autocast, no_grad, etc.)

    This is unsafe to use by default.
    To use this API, use guard_filter_fn argument while calling torch.compile

    >> opt_mod = torch.compile(
    >>     mod,
    >>     options={"guard_filter_fn": torch.compiler.keep_global_context_and_tensor_guards_unsafe},
    >> )
    """
    return [
        (
            g.guard_type in ("GLOBAL_STATE", "SHAPE_ENV")
            or (g.guard_type == "TENSOR_MATCH" and not g.is_global)
        )
        for g in guard_entries
    ]


def skip_guard_on_inbuilt_nn_modules_unsafe(guard_entries):
    """
    A common function to skip guards on the inbuilt nn modules like
    torch.nn.Linear. This is unsafe to use by default. But for majority of
    torch.compile users, the model code does not modify the inbuilt nn module
    attributes. They can benefit from reduction in guard latency overhead using
    this API.

    To use this API, use guard_filter_fn argument while calling torch.compile

    >> opt_mod = torch.compile(
    >>     mod,
    >>     options={"guard_filter_fn": torch.compiler.skip_guard_on_all_nn_modules_unsafe},
    >> )
    """
    return [
        not entry.orig_guard.source.is_unspecialized_builtin_nn_module()
        for entry in guard_entries
    ]


def skip_guard_on_all_nn_modules_unsafe(guard_entries):
    """
    A common function to skip guards on all nn modules, both user defined as
    well inbuilt nn modules (like torch.nn.Linear). This is unsafe to use by
    default. But for majority of torch.compile users, the model code does not
    modify the nn module attributes. They can benefit from reduction in guard
    latency overhead using this API.

    To use this API, use guard_filter_fn argument while calling torch.compile

    >> opt_mod = torch.compile(
    >>     mod,
    >>     options={"guard_filter_fn": torch.compiler.skip_guard_on_all_nn_modules_unsafe},
    >> )
    """

    return [
        not entry.orig_guard.source.is_unspecialized_nn_module()
        for entry in guard_entries
    ]


def keep_tensor_guards_unsafe(guard_entries, keep_parameters=False):
    """
    A common function to keep tensor guards on all tensors. This is unsafe to
    use by default. But if you don't expect any changes in the model code, you
    can just keep the tensor guards.


    >> opt_mod = torch.compile(
    >>     mod,
    >>     options={"guard_filter_fn": torch.compiler.keep_tensor_guards},
    >> )
    """

    keep_flags = []
    for entry in guard_entries:
        if entry.guard_type == "TENSOR_MATCH":
            if not isinstance(entry.value, torch.nn.Parameter):
                keep_flags.append(True)
            elif keep_parameters:
                keep_flags.append(True)
            else:
                keep_flags.append(False)
        else:
            keep_flags.append(False)
    return keep_flags


def skip_guard_on_globals_unsafe(guard_entries):
    """
    A common function to skip guards on all globals. This is unsafe to use by
    default. But if you don't expect any changes in the globals, you can just
    keep the tensor guards.

    >> opt_mod = torch.compile(
    >>     mod,
    >>     options={"guard_filter_fn": torch.compiler.skip_guard_on_globals},
    >> )
    """

    return [not entry.is_global for entry in guard_entries]


def skip_all_guards_unsafe(guard_entries):
    """
    A function for skipping all guards on a compiled function.

    WARNING: This function will drop all the safety guarantees from Dynamo
             compiled function. Use this with caution.

    To use this API, use guard_filter_fn argument while calling torch.compile

    >> opt_mod = torch.compile(
    >>     mod,
    >>     options={"guard_filter_fn": torch.compiler.skip_all_guards_unsafe},
    >> )
    """
    return [False for entry in guard_entries]


def nested_compile_region(
    fn=None,
    *,
    options: NestedCompileRegionOptions | None = None,
    max_reuse_entries: int = 8,
    reuse_hash_fn=None,
):
    """
    Tells **``torch.compile``** that the marked set of operations forms a nested
    compile region (which is often repeated in the full model) whose code can be
    compiled once and safely reused.  ``nested_compile_region`` can also be used
    as a decorator.

    During **``torch.compile``** tracing, the compiler applies *hierarchical
    compilation* with ``nested_compile_region``: it emits optimized code for the
    marked region the first time it is encountered and re-emits (or "stamps
    out") the previously compiled code on every subsequent invocation.  This can
    substantially reduce overall compile time for deeply-stacked,
    structurally-identical components such as the transformer layers of a
    large-language-model (LLM).

    Outside a ``torch.compile`` context—i.e., in standard eager execution—the
    call is a no-op, so existing workflows remain unaffected.

    Note that ``nested_compile_region`` **does not** promise that a region will
    be compiled exactly once.  If the compiler detects that new input conditions
    (shape, dtype, device, stride, globals etc.) make the cached version invalid
    to reuse, it will transparently re-compile the region.  Using it is
    therefore *safe*: correctness is always preserved, and you pay the extra
    compilation cost only when required.

    Args:
        fn: The function to wrap
        options: Optional backend to use for compiling the subgraph.
            Warning: this is an experimental feature under development and
            not ready for use yet.
        max_reuse_entries: Maximum number of reuse cache entries per function
            before raising an error. If this limit is hit, guards keep failing
            across invocations and hierarchical compilation is not effective.
        reuse_hash_fn: Optional callable that takes the same ``*args, **kwargs``
            as the wrapped function and returns an integer hash key. When
            provided, Dynamo traces this function to obtain a constant integer
            and uses it as the cache key for subgraph reuse, bypassing the
            automatic fingerprint/guard machinery. Two calls that produce the
            same hash key reuse the same cached subgraph. The hash function
            must be fully traceable (no graph breaks) and must return a
            constant integer.
    """

    if options is not None:
        from torch._dynamo import config as dynamo_config

        if not dynamo_config.enable_invoke_subgraph_regional_compile:
            raise RuntimeError(
                "nested_compile_region config is an experimental feature for testing only."
            )

    from torch._higher_order_ops.invoke_subgraph import (
        mark_compile_region as _mark_compile_region,
    )

    return _mark_compile_region(
        fn,
        options=options,
        max_reuse_entries=max_reuse_entries,
        reuse_hash_fn=reuse_hash_fn,
    )


def load_compiled_function(
    file: io.IOBase,
    *,
    f_globals: dict[str, object] | None = None,
    external_data: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    """
    Load an aot-compiled function from a file.

    .. warning::

        This API is currently experimental and subject to change.

    Args:
        file: A file-like object containing the serialized compiled function.
        f_globals: Optional global scope enclosing the compiled function.
        external_data: Optional data to be loaded into the runtime environment
                       of the compiled function. This should contain the same
                       data as AOTCompileResult.external_data returned from save_compiled_function() call.

    Returns:
        A torch-compiled function with compilation preloaded from disk.
    """
    from torch._dynamo.aot_compile import AOTCompiledFunction

    data = file.read()
    return AOTCompiledFunction.deserialize(data, f_globals, external_data)
<|endoftext|>from typing import TYPE_CHECKING

from torch._C._monitor import *  # noqa: F403
from torch._C._monitor import _WaitCounter, _WaitCounterTracker


if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

STAT_EVENT = "torch.monitor.Stat"


class TensorboardEventHandler:
    """
    TensorboardEventHandler is an event handler that will write known events to
    the provided SummaryWriter.

    This currently only supports ``torch.monitor.Stat`` events which are logged
    as scalars.

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_MONITOR)
        >>> # xdoctest: +REQUIRES(module:tensorboard)
        >>> from torch.utils.tensorboard import SummaryWriter
        >>> from torch.monitor import TensorboardEventHandler, register_event_handler
        >>> writer = SummaryWriter("log_dir")
        >>> register_event_handler(TensorboardEventHandler(writer))
    """

    def __init__(self, writer: "SummaryWriter") -> None:
        """
        Constructs the ``TensorboardEventHandler``.
        """
        self._writer = writer

    def __call__(self, event: Event) -> None:
        if event.name == STAT_EVENT:
            for k, v in event.data.items():
                self._writer.add_scalar(k, v, walltime=event.timestamp.timestamp())
<|endoftext|>from collections.abc import Callable

from torch._utils import CallbackRegistry


EventCreationCallbacks: "CallbackRegistry[int]" = CallbackRegistry("XPU event creation")
EventDeletionCallbacks: "CallbackRegistry[int]" = CallbackRegistry("XPU event deletion")
EventRecordCallbacks: "CallbackRegistry[int, int]" = CallbackRegistry(
    "XPU event record"
)
EventWaitCallbacks: "CallbackRegistry[int, int]" = CallbackRegistry("XPU event wait")
MemoryAllocationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "XPU memory allocation"
)
MemoryDeallocationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "XPU memory deallocation"
)
StreamCreationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "XPU stream creation"
)
DeviceSynchronizationCallbacks: "CallbackRegistry[[]]" = CallbackRegistry(
    "XPU device synchronization"
)
StreamSynchronizationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "XPU stream synchronization"
)
EventSynchronizationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "XPU event synchronization"
)


def register_callback_for_event_creation(cb: Callable[[int], None]) -> None:
    EventCreationCallbacks.add_callback(cb)


def register_callback_for_event_deletion(cb: Callable[[int], None]) -> None:
    EventDeletionCallbacks.add_callback(cb)


def register_callback_for_event_record(cb: Callable[[int, int], None]) -> None:
    EventRecordCallbacks.add_callback(cb)


def register_callback_for_event_wait(cb: Callable[[int, int], None]) -> None:
    EventWaitCallbacks.add_callback(cb)


def register_callback_for_memory_allocation(cb: Callable[[int], None]) -> None:
    MemoryAllocationCallbacks.add_callback(cb)


def register_callback_for_memory_deallocation(cb: Callable[[int], None]) -> None:
    MemoryDeallocationCallbacks.add_callback(cb)


def register_callback_for_stream_creation(cb: Callable[[int], None]) -> None:
    StreamCreationCallbacks.add_callback(cb)


def register_callback_for_device_synchronization(cb: Callable[[], None]) -> None:
    DeviceSynchronizationCallbacks.add_callback(cb)


def register_callback_for_stream_synchronization(cb: Callable[[int], None]) -> None:
    StreamSynchronizationCallbacks.add_callback(cb)


def register_callback_for_event_synchronization(cb: Callable[[int], None]) -> None:
    EventSynchronizationCallbacks.add_callback(cb)
<|endoftext|># mypy: allow-untyped-defs
from collections.abc import Iterable

import torch
from torch import Tensor

from . import _lazy_call, _lazy_init, current_device, device_count, is_initialized


def get_rng_state(device: int | str | torch.device = "xpu") -> Tensor:
    r"""Return the random number generator state of the specified GPU as a ByteTensor.

    Args:
        device (torch.device or int, optional): The device to return the RNG state of.
            Default: ``'xpu'`` (i.e., ``torch.device('xpu')``, the current XPU device).

    .. warning::
        This function eagerly initializes XPU.
    """
    _lazy_init()
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("xpu", device)
    idx = device.index
    if idx is None:
        idx = current_device()
    default_generator = torch.xpu.default_generators[idx]
    return default_generator.get_state()


def get_rng_state_all() -> list[Tensor]:
    r"""Return a list of ByteTensor representing the random number states of all devices."""
    results = [get_rng_state(i) for i in range(device_count())]
    return results


def set_rng_state(new_state: Tensor, device: int | str | torch.device = "xpu") -> None:
    r"""Set the random number generator state of the specified GPU.

    Args:
        new_state (torch.ByteTensor): The desired state
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'xpu'`` (i.e., ``torch.device('xpu')``, the current XPU device).
    """
    if not is_initialized():
        with torch._C._DisableFuncTorch():
            new_state = new_state.clone(memory_format=torch.contiguous_format)

    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("xpu", device)

    def cb() -> None:
        idx = device.index
        if idx is None:
            idx = current_device()
        default_generator = torch.xpu.default_generators[idx]
        default_generator.set_state(new_state)

    _lazy_call(cb)


def set_rng_state_all(new_states: Iterable[Tensor]) -> None:
    r"""Set the random number generator state of all devices.

    Args:
        new_states (Iterable of torch.ByteTensor): The desired state for each device.
    """
    for i, state in enumerate(new_states):
        set_rng_state(state, i)


def manual_seed(seed: int) -> None:
    r"""Set the seed for generating random numbers for the current GPU.

    It's safe to call this function if XPU is not available; in that case, it is silently ignored.

    Args:
        seed (int): The desired seed.

    .. warning::
        If you are working with a multi-GPU model, this function is insufficient
        to get determinism.  To seed all GPUs, use :func:`manual_seed_all`.
    """
    seed = int(seed)

    def cb() -> None:
        idx = current_device()
        default_generator = torch.xpu.default_generators[idx]
        default_generator.manual_seed(seed)

    _lazy_call(cb, seed=True)


def manual_seed_all(seed: int) -> None:
    r"""Set the seed for generating random numbers on all GPUs.

    It's safe to call this function if XPU is not available; in that case, it is silently ignored.

    Args:
        seed (int): The desired seed.
    """
    seed = int(seed)

    def cb() -> None:
        for i in range(device_count()):
            default_generator = torch.xpu.default_generators[i]
            default_generator.manual_seed(seed)

    _lazy_call(cb, seed_all=True)


def seed() -> None:
    r"""Set the seed for generating random numbers to a random number for the current GPU.

    It's safe to call this function if XPU is not available; in that case, it is silently ignored.

    .. warning::
        If you are working with a multi-GPU model, this function will only initialize
        the seed on one GPU.  To initialize all GPUs, use :func:`seed_all`.
    """

    def cb() -> None:
        idx = current_device()
        default_generator = torch.xpu.default_generators[idx]
        default_generator.seed()

    _lazy_call(cb)


def seed_all() -> None:
    r"""Set the seed for generating random numbers to a random number on all GPUs.

    It's safe to call this function if XPU is not available; in that case, it is silently ignored.
    """

    def cb() -> None:
        random_seed = 0
        seeded = False
        for i in range(device_count()):
            default_generator = torch.xpu.default_generators[i]
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
        This function eagerly initializes XPU.
    """
    _lazy_init()
    idx = current_device()
    default_generator = torch.xpu.default_generators[idx]
    return default_generator.initial_seed()


__all__ = [
    "get_rng_state",
    "get_rng_state_all",
    "set_rng_state",
    "set_rng_state_all",
    "manual_seed",
    "manual_seed_all",
    "seed",
    "seed_all",
    "initial_seed",
]
<|endoftext|>import collections
import contextlib
import ctypes
import pickle
import sys
from typing import Any, Literal

import torch
from torch._utils import _augment_memory_snapshot_stack_traces, _dummy_type
from torch.types import Device

from . import _get_device_index, _is_compiled, _lazy_init, is_initialized


if not _is_compiled():
    # Define dummy base classes
    torch._C.__dict__["_xpu_XPUAllocator"] = _dummy_type("_xpu_XPUAllocator")
    torch._C.__dict__["_XPUMemPool"] = _dummy_type("_XPUMemPool")
    torch._C.__dict__["_xpu_beginAllocateCurrentThreadToPool"] = _dummy_type(
        "_xpu_beginAllocateCurrentThreadToPool"
    )
    torch._C.__dict__["_xpu_endAllocateToPool"] = _dummy_type("_xpu_endAllocateToPool")
    torch._C.__dict__["_xpu_releasePool"] = _dummy_type("_xpu_releasePool")


def empty_cache() -> None:
    r"""Release all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other XPU application.

    .. note::
        :func:`~torch.xpu.empty_cache` doesn't increase the amount of XPU
        memory available for PyTorch. However, it may help reduce fragmentation
        of XPU memory in certain cases.
    """
    if is_initialized():
        torch._C._xpu_emptyCache()


def reset_peak_memory_stats(device: Device = None) -> None:
    r"""Reset the "peak" stats tracked by the XPU memory allocator.

    See :func:`~torch.xpu.memory_stats` for details. Peak stats correspond to the
    `"peak"` key in each individual stat dict.

    Args:
        device (torch.device or int or str, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    device = _get_device_index(device, optional=True)
    return torch._C._xpu_resetPeakMemoryStats(device)


def reset_accumulated_memory_stats(device: Device = None) -> None:
    r"""Reset the "accumulated" (historical) stats tracked by the XPU memory allocator.

    See :func:`~torch.xpu.memory_stats` for details. Accumulated stats correspond to
    the `"allocated"` and `"freed"` keys in each individual stat dict.

    Args:
        device (torch.device or int or str, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    device = _get_device_index(device, optional=True)
    return torch._C._xpu_resetAccumulatedMemoryStats(device)


def memory_stats_as_nested_dict(device: Device = None) -> dict[str, Any]:
    r"""Return the result of :func:`~torch.xpu.memory_stats` as a nested dictionary."""
    if not is_initialized():
        return {}
    device = _get_device_index(device, optional=True)
    return torch._C._xpu_memoryStats(device)


def _recurse_add_to_result(
    result: list[tuple[str, Any]], prefix: str, obj: Any
) -> None:
    if isinstance(obj, dict):
        if prefix:
            prefix += "."
        for key, value in obj.items():
            _recurse_add_to_result(result, prefix + key, value)
    else:
        result.append((prefix, obj))


def memory_stats(device: Device = None) -> dict[str, Any]:
    r"""Return a dictionary of XPU memory allocator statistics for a given device.

    The return value of this function is a dictionary of statistics, each of
    which is a non-negative integer.

    Core statistics:

    - ``"allocated_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of allocated memory.
    - ``"reserved_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of reserved memory.
    - ``"active_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of active memory.
    - ``"requested_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      memory requested by client code, compare this with allocated_bytes to check if
      allocation rounding adds too much overhead.

    For these core statistics, values are broken down as follows.

    Pool type:

    - ``all``: combined statistics across all memory pools.
    - ``large_pool``: statistics for the large allocation pool (for size >= 1MB allocations).
    - ``small_pool``: statistics for the small allocation pool (for size < 1MB allocations).

    Metric type:

    - ``current``: current value of this metric.
    - ``peak``: maximum value of this metric.
    - ``allocated``: historical total increase in this metric.
    - ``freed``: historical total decrease in this metric.

    Args:
        device (torch.device or int or str, optional): selected device. Returns
            statistics for the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    result = []

    stats = memory_stats_as_nested_dict(device=device)
    _recurse_add_to_result(result, "", stats)
    result.sort()

    return collections.OrderedDict(result)


def memory_allocated(device: Device = None) -> int:
    r"""Return the current GPU memory occupied by tensors in bytes for a given device.

    Args:
        device (torch.device or int or str, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        This is likely less than the amount shown in `xpu-smi` since some
        unused memory can be held by the caching allocator and some context
        needs to be created on GPU.
    """
    return memory_stats(device=device).get("allocated_bytes.all.current", 0)


def max_memory_allocated(device: Device = None) -> int:
    r"""Return the maximum GPU memory occupied by tensors in bytes for a given device.

    By default, this returns the peak allocated memory since the beginning of
    this program. :func:`~torch.xpu.reset_peak_memory_stats` can be used to
    reset the starting point in tracking this metric. For example, these two
    functions can measure the peak allocated memory usage of each iteration in a
    training loop.

    Args:
        device (torch.device or int or str, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    return memory_stats(device=device).get("allocated_bytes.all.peak", 0)


def memory_reserved(device: Device = None) -> int:
    r"""Return the current GPU memory managed by the caching allocator in bytes for a given device.

    Args:
        device (torch.device or int or str, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    return memory_stats(device=device).get("reserved_bytes.all.current", 0)


def max_memory_reserved(device: Device = None) -> int:
    r"""Return the maximum GPU memory managed by the caching allocator in bytes for a given device.

    By default, this returns the peak cached memory since the beginning of this
    program. :func:`~torch.xpu.reset_peak_memory_stats` can be used to reset
    the starting point in tracking this metric. For example, these two functions
    can measure the peak cached memory amount of each iteration in a training
    loop.

    Args:
        device (torch.device or int or str, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    return memory_stats(device=device).get("reserved_bytes.all.peak", 0)


def mem_get_info(device: Device = None) -> tuple[int, int]:
    r"""Return the global free and total GPU memory for a given device.

    Args:
        device (torch.device or int or str, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).

    Returns:
        tuple[int, int]: a tuple of two integers (free_memory, total_memory) in bytes.
            The first value is the free memory on the device (available across all processes and applications),
            The second value is the device's total hardware memory capacity.
    """
    _lazy_init()
    device = _get_device_index(device, optional=True)
    return torch._C._xpu_getMemoryInfo(device)


def get_per_process_memory_fraction(device: Device = None) -> float:
    r"""
    Retrieve the memory fraction currently set for a process on a given XPU device.
    This fraction represents the portion of the total device memory that
    the caching allocator is allowed to use. The allowed memory is calculated as:

    .. math:: \text{allowed\_memory} = \text{total\_memory} \times \text{fraction}

    Args:
        device (torch.device or int or str, optional): selected device. It uses the current device,
            given by :func:`~torch.xpu.current_device`, if :attr:`device` is ``None`` (default).

    Returns:
        float: The memory fraction in the range 0.0 to 1.0.
    """
    _lazy_init()
    device = _get_device_index(device, optional=True)
    return torch._C._xpu_getMemoryFraction(device)


def set_per_process_memory_fraction(fraction: float, device: Device = None) -> None:
    r"""
    Set the memory fraction for a single process on XPU device.
    This function limits the amount of memory that the caching allocator can allocate
    on the specified XPU device. The allowed memory is computed as:

    .. math:: \text{allowed\_memory} = \text{total\_memory} \times \text{fraction}

    If the process attempts to allocate more than this allowed memory,
    an out-of-memory error will be raised by the allocator.

    Arguments:
        fraction (float): Range: 0~1. Allowed memory equals total_memory * fraction.
        device (torch.device or int or str, optional): selected device. It uses the current device,
            given by :func:`~torch.xpu.current_device`, if :attr:`device` is ``None`` (default).

    .. note:: In general, the total available free memory is less than the total capacity.
    """
    _lazy_init()
    device = _get_device_index(device, optional=True)
    if not isinstance(fraction, float):
        raise TypeError("Invalid type for fraction argument, must be `float`")
    torch._C._xpu_setMemoryFraction(fraction, device)


def memory_snapshot(
    mempool_id: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    r"""
    Return a snapshot of the XPU memory allocator state across all devices.
    Provides detailed information for each memory segment managed by the allocator
    including its size, owning pool, associated stream, call stack traces, and other relevant attributes.

    Arguments:
        mempool_id (tuple[int, int] or None, optional): The memory pool id. If None, the default memory pool is used.

    Returns:
        list[dict[str, Any]]: List of memory segments and their attributes.
    """
    if not is_initialized():
        return []
    return torch._C._xpu_memorySnapshot(mempool_id)["segments"]


def _snapshot(device: Device = None, augment_with_fx_traces: bool = False):
    """
    Capture a snapshot of the XPU memory state at the time this function is called.

    The returned snapshot is a dictionary with the following structure.

    .. code-block:: python

        class Snapshot(TypedDict):
            segments: List[Segment]
            device_traces: List[List[TraceEntry]]


        class Segment(TypedDict):
            # A Segment represents a contiguous memory region returned by the SYCL runtime.
            #
            # All reserved memory is composed of these segments. Segments are
            # cached and reused by the allocator. When allocations are smaller
            # than the segment, the segment may be split into multiple Blocks.
            #
            # Calling :func:`~torch.xpu.memory.empty_cache` releases segments that are entirely inactive.
            address: int
            total_size: int  #  total size of segment
            stream: int
            segment_type: Literal["small", "large"]  # 'large' (>1MB)
            allocated_size: int  # size of memory in use
            active_size: int  # size of memory in use or in active_awaiting_free state
            blocks: List[Block]


        class Block(TypedDict):
            # A sub-region of a Segment, either currently allocated or cached for reuse.
            size: int
            requested_size: int  # Original requested size (may be smaller than `size`)
            address: int
            state: Literal[
                "active_allocated",  # used by a tensor
                "active_awaiting_free",  # waiting for another stream synchronization, then become free
                "inactive",  # free for reuse
            ]
            frames: List[Frame]  # stack trace from where the allocation occurred


        class Frame(TypedDict):
            filename: str
            line: int
            name: str
            # Optional fields when `augment_with_fx_traces=True` and the frame
            # corresponds to FX-generated code.
            fx_node_op: str  # FX node operation type (e.g., 'call_function', 'output')
            fx_node_name: str  # FX node name (e.g., 'linear', 'relu_1')
            fx_original_trace: str  # Original model source code stack trace


        class TraceEntry(TypedDict):
            # Trace entries are recorded only when :func:`~torch.xpu.memory._record_memory_history` is enabled.
            action: Literal[
                "alloc"  # memory allocated
                "free_requested",  # received a call to free memory
                "free_completed",  # memory reclaimed and reusable
                "segment_alloc",  # ask SYCL runtime for more memory
                "segment_free",  # called SYCL runtime to return memory to XPU
                "segment_map",  # ask SYCL runtime to map memory
                "segment_unmap",  # called SYCL runtime to unmap memory
                "snapshot",  # snapshot taken
                "oom",  # threw an OOM exception
            ]
            addr: int  # not present for OOM
            frames: List[Frame]
            size: int
            stream: int
            device_free: int  # only present for OOM, the amount of free memory reported by the device

    Arguments:
        device (torch.device or int or str, optional): selected device. It uses the current device,
            given by :func:`~torch.xpu.current_device`, if :attr:`device` is ``None`` (default).
        augment_with_fx_traces (bool, optional): If True, augment stack trace frames with FX debug information
            that maps generated FX code back to original model source code. This adds the FX-related
            fields (fx_node_op, fx_node_name, fx_original_trace) to Frame objects. Default is ``False``.

    Returns:
        The Snapshot dictionary object
    """
    s = torch._C._xpu_memorySnapshot(None)
    if augment_with_fx_traces:
        s = _augment_memory_snapshot_stack_traces(s)  # type: ignore[assignment, arg-type]
    return s


def _dump_snapshot(
    filename: str = "dump_snapshot.pickle", augment_with_fx_traces: bool = False
) -> None:
    """
    Save a pickled version of the `torch.memory._snapshot()` dictionary to a file.

    This file can be opened by the interactive snapshot viewer at pytorch.org/memory_viz

    Snapshot file sizes scale with `max_entries` and stack trace depth per entry,
    with several KB per entry. These can easily be in the GB range for longer running
    workflows with large `max_entries`.

    Arguments:
        filename (str, optional): Name of the file to create. Defaults to "dump_snapshot.pickle".
        augment_with_fx_traces (bool, optional): If True, augment the snapshot with FX debug information
            before dumping. This maps generated FX code stack traces back to original model
            source code. Defaults to ``False``.
    """
    s = _snapshot(augment_with_fx_traces=augment_with_fx_traces)

    with open(filename, "wb") as f:
        pickle.dump(s, f)


def _record_memory_history(
    enabled: Literal["state", "all"] | None = "all",
    context: Literal["state", "alloc", "all"] | None = "all",
    stacks: Literal["python", "all"] = "all",
    max_entries: int = sys.maxsize,
    clear_history: bool = False,
    skip_actions: list[str] | None = None,
) -> None:
    """
    Enable recording of stack traces associated with memory allocations, so you can
    tell what allocated any piece of memory in :func:`~torch.xpu.memory._snapshot()`.

    In addition to keeping stack traces with each current allocation and free,
    this will also enable recording of a history of all alloc/free events.

    Use :func:`~torch.xpu.memory._snapshot()` to retrieve this information,
    and the tools in `_memory_viz.py` to visualize snapshots.

    Buffer behavior
    ---------------

    This will store up to `max_entries` instances of `TraceEntry` when enabled.
    Python trace collection defaults to `sys.maxsize`, meaning long-running
    or indefinitely running jobs should set a reasonable limit to avoid excessive
    memory use. Expect each entry to be several KB.

    Longer running workflows or those with smaller `max_entries` values will only
    store the last accumulated `max_entries` entries, meaning new entries overwrite
    older entries, reference to ring buffer behavior.

    Latency impact
    --------------

    The Python trace collection is fast (2us per trace), so you may consider
    enabling this on production jobs if you anticipate ever having to debug
    memory issues.

    C++ trace collection is also fast (~50ns/frame), which for many typical programs
    works out to ~2us per trace, but can vary depending on stack depth.

    Arguments:
        enabled (Literal["state", "all"], optional):
            `None`, disable recording memory history.
            `"state"`, keep information for currently allocated memory.
            `"all"`, additionally keep a history of all alloc/free calls.
            Defaults to "all".
        context (Literal["state", "alloc", "all"], optional):
            `None`, Do not record any tracebacks.
            `"state"`, Record tracebacks for currently allocated memory.
            `"alloc"`, additionally keep tracebacks for alloc calls.
            `"all"`, additionally keep tracebacks for free calls.
            Defaults to "all".
        stacks (Literal["python", "all"], optional):
            `"python"`, include Python, TorchScript, and inductor frames in tracebacks.
            `"all"`, additionally include C++ frames.
            Defaults to "all".
        max_entries (int, optional): Keep a maximum of `max_entries`
            alloc/free events in the recorded history recorded.
        clear_history (bool, optional): Clear history when enabling, defaults to ``False``.
        skip_actions (list[str], optional): List of action types to skip when recording
            memory history. This can be used to reduce memory overhead by excluding
            certain types of events from being recorded. Valid action types are:

            - `"alloc"`: Memory allocation events
            - `"free_requested"`: Free requests (memory marked for freeing)
            - `"free_completed"`: Completed free operations (memory actually freed)
            - `"segment_alloc"`: Segment allocation from SYCL runtime
            - `"segment_free"`: Segment freed back to XPU via SYCL runtime
            - `"segment_map"`: Segment map events
            - `"segment_unmap"`: Segment unmap events
            - `"snapshot"`: Memory snapshot generation events
            - `"oom"`: Out-of-memory exceptions

            For example, to skip recording free_requested events:
            `skip_actions=["free_requested"]`

            Defaults to ``None`` (record all actions).
    """
    torch._C._xpu_recordMemoryHistory(
        enabled,
        context,
        stacks,
        max_entries,
        clear_history,
        skip_actions if skip_actions is not None else [],
    )


class _XPUAllocator:
    r"""Wrapper over internal XPU memory allocators."""

    def __init__(self, allocator: torch._C._xpu_XPUAllocator):
        self._allocator = allocator

    def allocator(self):
        return self._allocator


class XPUPluggableAllocator(_XPUAllocator):
    r"""
    XPU memory allocator loaded dynamically from a shared library.

    This lets users provide custom allocation and free functions implemented
    in a separate shared library. The allocator is registered and could become
    available for use via :func:`~torch.xpu.memory.change_current_allocator`.

    Arguments:
        path_to_lib_file (str):
            Filesystem path to the shared library file containing the allocation
            and free functions.
        alloc_fn_name (str):
            Name of the allocation function exported from the shared library.
            The function must have the signature:

                ``void* alloc_fn(size_t size, int device, sycl::queue* queue);``

        free_fn_name (str):
            Name of the free function exported from the shared library.
            The function must have the signature:

                ``void free_fn(void* ptr, size_t size, int device, sycl::queue* queue);``
    """

    def __init__(self, path_to_lib_file: str, alloc_fn_name: str, free_fn_name: str):
        allocator_lib = ctypes.CDLL(path_to_lib_file)

        alloc_fn_ptr = getattr(allocator_lib, alloc_fn_name)
        free_fn_ptr = getattr(allocator_lib, free_fn_name)

        alloc_fn_addr = ctypes.cast(alloc_fn_ptr, ctypes.c_void_p).value
        free_fn_addr = ctypes.cast(free_fn_ptr, ctypes.c_void_p).value

        if alloc_fn_addr is None or free_fn_addr is None:
            raise RuntimeError(
                "Failed to load allocator symbols from the shared library."
            )

        self._allocator = torch._C._xpu_customAllocator(alloc_fn_addr, free_fn_addr)


def change_current_allocator(allocator: _XPUAllocator) -> None:
    r"""Change the currently used memory allocator to be the one provided.

    .. note::
        If the current allocator has already been used/initialized, this function will error.

    Arguments:
        allocator (torch.xpu.memory._XPUAllocator): allocator to be set as the active one.
    """
    torch._C._xpu_changeCurrentAllocator(allocator.allocator())


def _get_current_allocator() -> _XPUAllocator:
    r"""Return the allocator being currently used.

    Returns:
        _XPUAllocator: the allocator being currently used.
    """
    return _XPUAllocator(torch._C._xpu_getAllocator())


class MemPool(torch._C._XPUMemPool):
    r"""MemPool represents a pool of memory in a caching allocator. Currently,
    it's just the ID of the pool object maintained in the XPUCachingAllocator.

    Args:
        allocator(torch._C._xpu_XPUAllocator, optional): a
            torch._C._xpu_XPUAllocator object that can be used to
            define how memory gets allocated in the pool. If :attr:`allocator`
            is ``None`` (default), memory allocation follows the default/
            current configuration of the XPUCachingAllocator.
        use_on_oom(bool): a bool that indicates if this pool can be used
            as a last resort if a memory allocation outside of the pool fails due
            to Out Of Memory. This is ``False`` by default.
    """

    def __init__(
        self,
        allocator: torch._C._xpu_XPUAllocator | None = None,
        use_on_oom: bool = False,
    ):
        super().__init__(allocator, True, use_on_oom)

    @property
    def id(self) -> tuple[int, int]:
        r"""Returns the ID of this pool as a tuple of two ints."""
        return super().id

    @property
    def allocator(self) -> torch._C._xpu_XPUAllocator | None:
        r"""Returns the allocator this MemPool routes allocations to."""
        return super().allocator

    def use_count(self) -> int:
        r"""Returns the reference count of this pool."""
        return super().use_count()

    def snapshot(self):
        r"""Return a snapshot of the XPU memory allocator pool state across all
        devices.

        Interpreting the output of this function requires familiarity with the
        memory allocator internals.
        """
        snapshot = torch.xpu.memory_snapshot(self.id)
        return snapshot


@contextlib.contextmanager
def use_mem_pool(pool: MemPool, device: "Device" = None):
    r"""A context manager that routes allocations to a given pool.

    Args:
        pool(torch.xpu.MemPool): a :class:`MemPool` object to be made active so that
            allocations route to this pool.
        device (torch.device or int, optional): selected device. Uses :class:`MemPool on
            the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        This context manager makes only current thread's allocations route to
        the given pool. If a new thread is spawned inside the context manager
        (e.g. by calling backward) the allocations in that thread will not
        route to the given pool.
    """
    device_index = (
        torch.xpu.current_device() if device is None else _get_device_index(device)
    )
    torch._C._xpu_beginAllocateCurrentThreadToPool(device_index, pool.id)
    try:
        yield
    finally:
        torch._C._xpu_endAllocateToPool(device_index, pool.id)
        torch._C._xpu_releasePool(device_index, pool.id)


def list_gpu_processes(device: "Device" = None) -> str:
    r"""Return a printout of running processes and their GPU memory usage on a given device.

    Args:
        device (torch.device, str or int, optional): selected device. Uses the
            current device, given by :func:`~torch.xpu.current_device`,
            if ``None`` (default).

    Returns:
        str: A human-readable summary of each running process and the given GPU memory usage in MB.

    .. note:: Process status is reported at the physical device level and reflects all processes
        associated with the device.
    """
    from ctypes import byref, c_uint32

    from . import (
        _cached_zes_device_infos,
        _get_pyzes_version,
        _import_pyzes,
        _zes_check,
        _zes_ensure_device_infos,
    )

    pyzes = _import_pyzes()
    version = _get_pyzes_version()
    if version < (0, 1, 2):
        raise RuntimeError(
            f"list_gpu_processes requires pyzes version >= 0.1.2, but found {'.'.join(map(str, version))}"
        )
    device = _get_device_index(device, optional=True)
    _zes_ensure_device_infos(device)

    info = _cached_zes_device_infos[device]
    device_handle = info.device_handle
    proc_count = c_uint32(0)
    _zes_check(
        pyzes.zesDeviceProcessesGetState(device_handle, byref(proc_count), None),
        "Can't get Level Zero Sysman host processes count.",
    )
    process_states = (pyzes.zes_process_state_t * proc_count.value)()
    _zes_check(
        pyzes.zesDeviceProcessesGetState(
            device_handle, byref(proc_count), process_states
        ),
        "Can't get Level Zero Sysman host processes state.",
    )

    lines = [f"GPU:{device}"]
    if proc_count.value == 0:
        lines.append("  no processes are running")
    for state in process_states:
        mem_mb = state.memSize / (1024 * 1024)
        lines.append(
            f"  process {state.processId:>10d} uses {mem_mb:>12.3f} MB GPU memory"
        )
    return "\n".join(lines)


__all__ = [
    "MemPool",
    "XPUPluggableAllocator",
    "change_current_allocator",
    "empty_cache",
    "get_per_process_memory_fraction",
    "list_gpu_processes",
    "max_memory_allocated",
    "max_memory_reserved",
    "mem_get_info",
    "memory_allocated",
    "memory_reserved",
    "memory_snapshot",
    "memory_stats",
    "memory_stats_as_nested_dict",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
    "set_per_process_memory_fraction",
    "use_mem_pool",
]
<|endoftext|># mypy: allow-untyped-defs
r"""
This package introduces support for the XPU backend, specifically tailored for
Intel GPU optimization.

This package is lazily initialized, so you can always import it, and use
:func:`is_available()` to determine if your system supports XPU.
"""

from __future__ import annotations

import dataclasses
import os
import threading
import traceback
import warnings
from ctypes import byref, c_double, c_uint32, c_void_p, cast, pointer
from functools import lru_cache
from typing import Any, NewType, TYPE_CHECKING

import torch
import torch._C
from torch._utils import _dummy_type, _LazySeedTracker


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.types import Device

from ._utils import _get_device_index
from .graphs import (
    graph,
    graph_pool_handle,
    is_current_stream_capturing,
    make_graphed_callables,
    XPUGraph,
)
from .streams import Event, Stream


_initialized = False
_tls = threading.local()
_initialization_lock = threading.Lock()
_queued_calls: list[
    tuple[Callable[[], None], list[str]]
] = []  # don't invoke these until initialization occurs
_is_in_bad_fork = getattr(torch._C, "_xpu_isInBadFork", lambda: False)
_lazy_seed_tracker = _LazySeedTracker()
default_generators: tuple[torch._C.Generator] = ()  # type: ignore[assignment]
_cached_device_count: int | None = None


@dataclasses.dataclass
class _ZesDeviceInfo:
    device_handle: c_void_p
    subdevice_id: int | None = None
    is_integrated: bool = False
    temperature_handle: c_void_p | None = None
    frequency_handle: c_void_p | None = None
    power_handle: c_void_p | None = None
    engine_handle: c_void_p | None = None
    memory_handle: c_void_p | None = None


_cached_zes_device_infos: list[_ZesDeviceInfo] = []
# Interval between two HW counter reads; must be >=100ms for fresh data.
_zes_sample_interval_ms = 150


def _is_compiled() -> bool:
    r"""Return true if compile with XPU support."""
    return torch._C._has_xpu


if _is_compiled():
    _XpuDeviceProperties = torch._C._XpuDeviceProperties
    _exchange_device = torch._C._xpu_exchangeDevice
    _maybe_exchange_device = torch._C._xpu_maybeExchangeDevice
else:
    # Define dummy if PyTorch was compiled without XPU
    _XpuDeviceProperties = _dummy_type("_XpuDeviceProperties")  # type: ignore[assignment, misc]

    def _exchange_device(device: int) -> int:
        raise NotImplementedError("PyTorch was compiled without XPU support")

    def _maybe_exchange_device(device: int) -> int:
        raise NotImplementedError("PyTorch was compiled without XPU support")


def _parse_visible_devices(strict=False) -> list[int]:
    r"""Parse ``ZE_AFFINITY_MASK`` and return visible device ordinals.

    Returns a list of non-negative device ordinals specified by the mask.
    When the mask is unset, returns ``[0, 1, ..., 127]`` (the maximum range
    for ``int8_t`` device indices).  Returns an empty list for unsupported
    COMPOSITE-style masks (e.g. ``"0.0,0.1"``).

    Args:
        strict (bool): If ``True``, raises ``ValueError`` on unsupported mask
            formats (e.g. COMPOSITE-style ``"0.0,0.1"``).  If ``False``
            (default), returns an empty list instead.
    """
    var = os.getenv("ZE_AFFINITY_MASK")
    if var is None:
        # DeviceIndex is stored as int8_t, so valid indices are 0–127
        # (up to 128 devices). Return the full range when no mask is set.
        return list(range(128))

    visible_devices: list[int] = []
    for elem in var.split(","):
        try:
            x = int(elem.strip())
        except ValueError:
            # A non-integer token (e.g. "0.0" in COMPOSITE-mode format)
            # means the mask is unsupported here; signal that by returning
            # an empty list.
            if strict:
                raise ValueError(
                    f"Unsupported ZE_AFFINITY_MASK format: '{var}'. Expected a comma-separated list of integers, e.g. '0,1,2'."
                ) from None
            return []
        if x >= 0 and x not in visible_devices:
            visible_devices.append(x)
    return visible_devices


def _enum_zes_device_infos(visible_mask: list[int]) -> int:
    r"""Enumerate visible XPU devices via Level Zero Sysman and cache their info.

    Enumerates devices from the first Level Zero Sysman driver and counts those
    whose logical index appears in *visible_mask*.  Only devices listed in
    the visible mask participate in counting.
    The populated ``_cached_zes_device_infos`` list is indexed by PyTorch
    device ordinal.

    Discrete GPUs (dGPUs) take priority: if any visible dGPU is found, only
    dGPUs are counted; integrated GPUs (iGPUs) are counted only when no
    visible dGPU exists.

    For tiled dGPUs (``numSubdevices > 0``), the counting depends on
    ``ZE_FLAT_DEVICE_HIERARCHY``:

    - **FLAT / COMBINED** (default): each sub-device is exposed as a
      separate top-level device and counted individually.
    - **COMPOSITE**: sub-devices are hidden; the whole physical device
      counts as one.

    Returns the visible device count, or a negative value on failure.
    """
    try:
        import pyzes  # type: ignore[import]
    except Exception:
        return -1

    global _cached_zes_device_infos

    def _zes_check_warn(rc: int, msg: str) -> bool:
        """Return True if the call failed (rc != ZE_RESULT_SUCCESS) after issuing a warning."""
        if rc != pyzes.ZE_RESULT_SUCCESS:
            warnings.warn(msg, stacklevel=3)
        return rc != pyzes.ZE_RESULT_SUCCESS

    if _zes_check_warn(pyzes.zesInit(0), "Can't initialize Level Zero Sysman"):
        return -1

    driver_count = c_uint32(0)
    if _zes_check_warn(
        pyzes.zesDriverGet(byref(driver_count), None),
        "Can't get Level Zero Sysman driver count",
    ):
        return -1
    if driver_count.value == 0:
        return 0

    drivers = (pyzes.zes_driver_handle_t * driver_count.value)()
    if _zes_check_warn(
        pyzes.zesDriverGet(byref(driver_count), drivers),
        "Can't get Level Zero Sysman driver handles",
    ):
        return -1

    device_count = c_uint32(0)
    if _zes_check_warn(
        pyzes.zesDeviceGet(drivers[0], byref(device_count), None),
        "Can't get Level Zero Sysman device count",
    ):
        return -1

    devices = (pyzes.zes_device_handle_t * device_count.value)()
    if _zes_check_warn(
        pyzes.zesDeviceGet(drivers[0], byref(device_count), devices),
        "Can't get Level Zero Sysman device handles",
    ):
        return -1

    # --- Count visible dGPUs and iGPUs ---
    ZES_DEVICE_PROPERTY_FLAG_INTEGRATED = 1 << 0
    expose_subdevices = os.getenv("ZE_FLAT_DEVICE_HIERARCHY") != "COMPOSITE"

    _cached_zes_device_infos.clear()
    visible = set(visible_mask)
    logical_index = 0
    num_igpu = 0
    num_dgpu = 0

    for device in devices:
        props = pyzes.zes_device_properties_t()
        props.stype = pyzes.ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES
        ext_props = pyzes.zes_device_ext_properties_t()
        ext_props.stype = pyzes.ZES_STRUCTURE_TYPE_DEVICE_EXT_PROPERTIES
        props.pNext = cast(pointer(ext_props), c_void_p)
        if _zes_check_warn(
            pyzes.zesDeviceGetProperties(device, byref(props)),
            "Can't get Level Zero Sysman device properties",
        ):
            return -1

        is_integrated = bool(ext_props.flags & ZES_DEVICE_PROPERTY_FLAG_INTEGRATED)

        # Tiled dGPUs in FLAT/COMBINED mode expose each sub-device as a
        # separate logical device; everything else counts as one slot.
        tiled = not is_integrated and props.numSubdevices > 0 and expose_subdevices
        num_slots = props.numSubdevices if tiled else 1

        for slot in range(num_slots):
            if logical_index in visible:
                _cached_zes_device_infos.append(
                    _ZesDeviceInfo(
                        device_handle=device,
                        subdevice_id=slot if tiled else None,
                        is_integrated=is_integrated,
                    )
                )
                if is_integrated:
                    num_igpu += 1
                else:
                    num_dgpu += 1
            logical_index += 1

    # dGPUs take priority; strip iGPUs when at least one dGPU is visible.
    if num_dgpu and num_igpu:
        _cached_zes_device_infos = [
            info for info in _cached_zes_device_infos if not info.is_integrated
        ]
    return num_dgpu or num_igpu


def _raw_device_count_zes(visible_mask: list[int]) -> int:
    r"""Return the visible XPU device count via Level Zero Sysman, or negative on failure."""
    return _enum_zes_device_infos(visible_mask)


def _device_count_zes() -> int:
    r"""Return the number of visible XPU devices, or -1 on failure."""
    visible_devices = _parse_visible_devices()
    if not visible_devices:
        return -1
    return _raw_device_count_zes(visible_devices)


def device_count() -> int:
    r"""
    Return the number of XPU device available.

    .. note:: This API will NOT poison fork if Level Zero Sysman discovery succeeds.
        See :ref:`multiprocessing-poison-fork-note` for more details.
    """
    if not _is_compiled():
        return 0
    global _cached_device_count
    if _cached_device_count is not None:
        return _cached_device_count
    if _initialized or hasattr(_tls, "is_initializing"):
        count = torch._C._xpu_getDeviceCount()
    else:
        zes_count = _device_count_zes()
        count = torch._C._xpu_getDeviceCount() if zes_count < 0 else zes_count
    # Do not cache the device count prior to XPU initialization, because
    # the number of devices can change due to changes to ZE_AFFINITY_MASK
    # setting prior to XPU initialization.
    if _initialized:
        _cached_device_count = count
    return count


def is_available() -> bool:
    r"""Return a bool indicating if XPU is currently available."""
    # This function never throws.
    return device_count() > 0


def is_bf16_supported(including_emulation: bool = True) -> bool:
    r"""Return a bool indicating if the current XPU device supports dtype bfloat16."""
    if not is_available():
        return False
    return (
        including_emulation
        or torch.xpu.get_device_properties().has_bfloat16_conversions
    )


def is_tf32_supported() -> bool:
    r"""Return a bool indicating if the current XPU device supports dtype tf32."""
    if not is_available():
        return False
    # On Intel Xe architecture and newer, TF32 operations can be accelerated
    # through DPAS (Dot Product Accumulate Systolic) instructions. Therefore,
    # TF32 support can be determined by checking whether the device supports
    # subgroup matrix multiply-accumulate operations.
    return torch.xpu.get_device_properties().has_subgroup_matrix_multiply_accumulate


def is_initialized():
    r"""Return whether PyTorch's XPU state has been initialized."""
    return _initialized and not _is_in_bad_fork()


def _lazy_call(callable, **kwargs) -> None:
    with _initialization_lock:
        if is_initialized():
            callable()
        else:
            global _lazy_seed_tracker
            if kwargs.get("seed_all", False):
                _lazy_seed_tracker.queue_seed_all(callable, traceback.format_stack())
            elif kwargs.get("seed", False):
                _lazy_seed_tracker.queue_seed(callable, traceback.format_stack())
            else:
                # Don't store the actual traceback to avoid memory cycle
                _queued_calls.append((callable, traceback.format_stack()))


def init() -> None:
    r"""Initialize PyTorch's XPU state.
    This is a Python API about lazy initialization that avoids initializing
    XPU until the first time it is accessed. Does nothing if the XPU state is
    already initialized.
    """
    _lazy_init()


def _lazy_init() -> None:
    global _initialized, _queued_calls
    if is_initialized() or hasattr(_tls, "is_initializing"):
        return
    with _initialization_lock:
        # This test was protected via GIL. Double-check whether XPU has
        # already been initialized.
        if is_initialized():
            return
        # Stop promptly upon encountering a bad fork error.
        if _is_in_bad_fork():
            raise RuntimeError(
                "Cannot re-initialize XPU in forked subprocess. To use XPU with "
                "multiprocessing, you must use the 'spawn' start method"
            )
        if not _is_compiled():
            raise AssertionError("Torch not compiled with XPU enabled")
        # This function inits XPU backend and detects bad fork processing.
        torch._C._xpu_init()
        # Some of the queued calls may reentrantly call _lazy_init(); We need to
        # just return without initializing in that case.
        _tls.is_initializing = True

        _queued_calls.extend(calls for calls in _lazy_seed_tracker.get_calls() if calls)

        try:
            for queued_call, orig_traceback in _queued_calls:
                try:
                    queued_call()
                except Exception as e:
                    msg = (
                        f"XPU call failed lazily at initialization with error: {str(e)}\n\n"
                        f"XPU call was originally invoked at:\n\n{''.join(orig_traceback)}"
                    )
                    raise Exception(msg) from e  # noqa: TRY002
        finally:
            delattr(_tls, "is_initializing")
        _initialized = True


class _DeviceGuard:
    def __init__(self, index: int) -> None:
        self.idx = index
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = torch.xpu._exchange_device(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        self.idx = torch.xpu._maybe_exchange_device(self.prev_idx)
        return False


class device:
    r"""Context-manager that changes the selected device.

    Args:
        device (torch.device or int or str): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device: Any) -> None:
        self.idx = _get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = torch.xpu._exchange_device(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        self.idx = torch.xpu._maybe_exchange_device(self.prev_idx)
        return False


class device_of(device):
    r"""Context-manager that changes the current device to that of given object.

    You can use both tensors and storages as arguments. If a given object is
    not allocated on a XPU, this is a no-op.

    Args:
        obj (Tensor or Storage): object allocated on the selected device.
    """

    def __init__(self, obj) -> None:
        idx = obj.get_device() if obj.is_xpu else -1
        super().__init__(idx)


def set_device(device: Device) -> None:
    r"""Set the current device.

    Args:
        device (torch.device or int or str): selected device. This function is a
            no-op if this argument is negative.
    """
    _lazy_init()
    device = _get_device_index(device)
    if device >= 0:
        torch._C._xpu_setDevice(device)


def get_device_name(device: Device = None) -> str:
    r"""Get the name of a device.

    Args:
        device (torch.device or int or str, optional): device for which to
            return the name. This function is a no-op if this argument is a
            negative integer. It uses the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).

    Returns:
        str: the name of the device
    """
    return get_device_properties(device).name


@lru_cache(None)
def get_device_capability(device: Device = None) -> dict[str, Any]:
    r"""Get the xpu capability of a device.

    Args:
        device (torch.device or int or str, optional): device for which to
            return the device capability. This function is a no-op if this
            argument is a negative integer. It uses the current device, given by
            :func:`~torch.xpu.current_device`, if :attr:`device` is ``None``
            (default).

    Returns:
        dict[str, Any]: the xpu capability dictionary of the device
    """
    props = get_device_properties(device)
    # Only keep attributes that are safe for dictionary serialization.
    serializable_types = (int, float, bool, str, type(None), list, tuple, dict)
    return {
        key: value
        for key in dir(props)
        if not key.startswith("__")
        and isinstance((value := getattr(props, key)), serializable_types)
    }


def get_device_properties(
    device: Device = None,
) -> _XpuDeviceProperties:
    r"""Get the properties of a device. Returns _XpuDeviceProperties containing the following device properties:

    - ``name`` (str): device name.
    - ``platform_name`` (str): SYCL platform name.
    - ``vendor`` (str): device vendor.
    - ``device_id`` (int): device identifier (product ID).
    - ``driver_version`` (str): driver version.
    - ``version`` (str): runtime version.
    - ``max_compute_units`` (int): number of parallel compute units.
    - ``gpu_eu_count`` (int): number of EUs (Execution Unit).
    - ``max_work_group_size``: (int): maximum number of work-items permitted in a work-group.
    - ``max_num_sub_groups`` (int): maximum number of sub-groups supported in a work-group.
    - ``memory_clock_rate`` (int) maximum clock rate of device's global memory in MHz.
    - ``memory_bus_width`` (int) maximum bus width between device and memory in bits.
    - ``sub_group_sizes``: (list[int]): a list of supported sub-group sizes.
    - ``local_mem_size`` (int): device local memory capacity that can be allocated per work-group in bytes.
    - ``last_level_cache_size`` (int): size in bytes of the device's last-level memory cache, shared across all Xe Cores (analogous to CUDA ``L2_cache_size``).
    - ``has_fp16`` (bool): whether float16 dtype is supported.
    - ``has_fp64`` (bool): whether float64 dtype is supported.
    - ``has_atomic64`` (bool): whether 64-bit atomic operations are supported.
    - ``has_bfloat16_conversions`` (bool): whether bfloat16 conversions are supported.
    - ``has_subgroup_matrix_multiply_accumulate`` (bool): whether DPAS (Dot Product Accumulate Systolic) is supported.
    - ``has_subgroup_matrix_multiply_accumulate_tensor_float32`` (bool): whether DPAS with tf32 inputs is supported.
    - ``has_subgroup_2d_block_io`` (bool): whether 2D block I/O for efficient matrix multiplication is supported.
    - ``is_integrated_gpu`` (bool): whether the device is an integrated GPU (iGPU).
    - ``total_memory`` (int): device global memory in bytes.
    - ``gpu_subslice_count`` (int): number of subslice.
    - ``architecture`` (int): device architecture identifier (experimental).
    - ``type`` (str): device type, e.g. 'cpu', 'gpu', accelerator', 'host', 'unknown'.
    - ``uuid`` (Any): device UUID (Universal Unique ID), 16 bytes.

    Args:
        device (torch.device or int or str): device for which to return the
            properties of the device.

    Returns:
        _XpuDeviceProperties: the properties of the device
    """
    _lazy_init()
    device = _get_device_index(device, optional=True)
    return _get_device_properties(device)  # type: ignore[name-defined]  # noqa: F821


def current_device() -> int:
    r"""Return the index of a currently selected device."""
    _lazy_init()
    return torch._C._xpu_getDevice()


def _get_device(device: int | str | torch.device) -> torch.device:
    r"""Return the torch.device type object from the passed in device.

    Args:
        device (torch.device or int or str): selected device.
    """
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("xpu", device)
    return device


def can_device_access_peer(device: Device, peer: Device) -> bool:
    r"""Query whether a device can access a peer device's memory.

    Args:
        device (torch.device or int or str): selected device.
        peer (torch.device or int or str): peer device to query access to.

    Returns:
        bool: ``True`` if ``device`` can access ``peer``, ``False`` otherwise.
    """
    _lazy_init()
    device = _get_device_index(device, optional=True)
    peer = _get_device_index(peer, optional=True)
    return torch._C._xpu_canDeviceAccessPeer(device, peer)


class StreamContext:
    r"""Context-manager that selects a given stream.

    All XPU kernels queued within its context will be enqueued on a selected
    stream.

    Args:
        Stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note:: Streams are per-device.
    """

    cur_stream: torch.xpu.Stream | None

    def __init__(self, stream: torch.xpu.Stream | None) -> None:
        self.stream = stream
        self.idx = _get_device_index(None, True)
        if self.idx is None:
            self.idx = -1  # pyrefly: ignore [bad-assignment]

    def __enter__(self):
        cur_stream = self.stream
        if cur_stream is None or self.idx == -1:
            return
        self.src_prev_stream = torch.xpu.current_stream(None)

        # If the stream is not on the current device, then set the current stream on the device
        if self.src_prev_stream.device != cur_stream.device:
            with device(cur_stream.device):
                self.dst_prev_stream = torch.xpu.current_stream(cur_stream.device)
        torch.xpu.set_stream(cur_stream)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        cur_stream = self.stream
        if cur_stream is None or self.idx == -1:
            return

        # Reset the stream on the original device and destination device
        if self.src_prev_stream.device != cur_stream.device:
            torch.xpu.set_stream(self.dst_prev_stream)
        torch.xpu.set_stream(self.src_prev_stream)


def stream(stream: torch.xpu.Stream | None) -> StreamContext:
    r"""Wrap around the Context-manager StreamContext that selects a given stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's ``None``.
    """
    return StreamContext(stream)


def _set_stream_by_id(stream_id, device_index, device_type) -> None:
    r"""set stream specified by the stream id, device index and device type

    Args: stream_id (int): not visible to the user, used to assigned to the specific stream.
          device_index (int): selected device index.
          device_type (int): selected device type.
    """
    torch._C._xpu_setStream(
        stream_id=stream_id,
        device_index=device_index,
        device_type=device_type,
    )


def set_stream(stream: Stream) -> None:
    r"""Set the current stream. This is a wrapper API to set the stream.
        Usage of this function is discouraged in favor of the ``stream``
        context manager.

    Args:
        stream (Stream): selected stream. This function is a no-op
            if this argument is ``None``.
    """
    if stream is None:
        return
    _lazy_init()
    _set_stream_by_id(
        stream_id=stream.stream_id,
        device_index=stream.device_index,
        device_type=stream.device_type,
    )


def current_stream(device: Device = None) -> Stream:
    r"""Return the currently selected :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch.xpu.current_device`, if :attr:`device` is ``None``
            (default).
    """
    _lazy_init()
    streamdata = torch._C._xpu_getCurrentStream(
        _get_device_index(device, optional=True)
    )
    return Stream(
        stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2]
    )


def get_stream_from_external(data_ptr: int, device: Device = None) -> Stream:
    r"""Return a :class:`Stream` from an external SYCL queue.

    This function is used to wrap SYCL queue created in other libraries in order
    to facilitate data exchange and multi-library interactions.

    .. note:: This function doesn't manage the queue life-cycle, it is the user
       responsibility to keep the referenced queue alive while this returned stream is
       being used. The different SYCL queue pointers will result in distinct
       :class:`Stream` objects, even if the SYCL queues they dereference are equivalent.

    Args:
        data_ptr(int): Integer representation of the `sycl::queue*` value passed externally.
        device(torch.device or int, optional): the device where the queue was originally created.
            It is the user responsibility to ensure the device is specified correctly.
    """
    _lazy_init()
    streamdata = torch._C._xpu_getStreamFromExternal(
        data_ptr, _get_device_index(device, optional=True)
    )
    return Stream(
        stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2]
    )


def synchronize(device: Device = None) -> None:
    r"""Wait for all kernels in all streams on a XPU device to complete.

    Args:
        device (torch.device or int, optional): device for which to synchronize.
            It uses the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    _lazy_init()
    device = _get_device_index(device, optional=True)
    return torch._C._xpu_synchronize(device)


def _sleep(cycles: int) -> None:
    r"""Spin the current XPU device for the given number of device clock cycles.

    The kernel is enqueued on the current stream and returns immediately on the
    host; the device-side spin runs asynchronously.

    Args:
        cycles (int): number of device clock cycles to spin.
    """
    torch._C._xpu_sleep(cycles)


def get_arch_list() -> list[str]:
    r"""Return list XPU architectures this library was compiled for."""
    if not _is_compiled():
        return []
    arch_flags = torch._C._xpu_getArchFlags()
    if arch_flags is None:
        return []
    return arch_flags.split()


def get_gencode_flags() -> str:
    r"""Return XPU AOT(ahead-of-time) build flags this library was compiled with."""
    arch_list = get_arch_list()
    if len(arch_list) == 0:
        return ""
    return f"-device {','.join(arch for arch in arch_list)}"


def _get_generator(device: torch.device) -> torch._C.Generator:
    r"""Return the XPU Generator object for the given device.

    Args:
        device (torch.device): selected device.
    """
    idx = device.index
    if idx is None:
        idx = current_device()
    return torch.xpu.default_generators[idx]


def _set_rng_state_offset(
    offset: int, device: int | str | torch.device = "xpu"
) -> None:
    r"""Set the random number generator state offset of the specified GPU.

    Args:
        offset (int): The desired offset
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'xpu'`` (i.e., ``torch.device('xpu')``, the current XPU device).
    """
    final_device = _get_device(device)

    def cb() -> None:
        default_generator = _get_generator(final_device)
        default_generator.set_offset(offset)

    _lazy_call(cb)


def _get_rng_state_offset(device: int | str | torch.device = "xpu") -> int:
    r"""Return the random number generator state offset of the specified GPU.

    Args:
        device (torch.device or int, optional): The device to return the RNG state offset of.
            Default: ``'xpu'`` (i.e., ``torch.device('xpu')``, the current XPU device).

    .. warning::
        This function eagerly initializes XPU.
    """
    _lazy_init()
    final_device = _get_device(device)
    default_generator = _get_generator(final_device)
    return default_generator.get_offset()


def _import_pyzes():
    """Return the imported pyzes module; raise ImportError if missing, RuntimeError if the GPU driver is unavailable."""
    try:
        import pyzes  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "pyzes is required; install it with 'pip install pyzes'."
        ) from None
    except Exception as err:
        raise RuntimeError(
            "Failed to import pyzes. Ensure the GPU driver is installed with Level Zero Sysman support."
        ) from err
    return pyzes


def _get_pyzes_version() -> tuple[int, ...]:
    """
    Return the version of the pyzes package as a tuple of integers (major, minor, patch).
    Always ensure that the pyzes package is installed before calling this function.
    """
    from importlib.metadata import version

    return tuple(map(int, version("pyzes").split(".")))


def _zes_check(rc: int, msg: str) -> None:
    """Raise RuntimeError if the Level Zero Sysman call failed (rc != ZE_RESULT_SUCCESS)."""
    import pyzes  # type: ignore[import]

    if rc != pyzes.ZE_RESULT_SUCCESS:
        raise RuntimeError(f"{msg} (rc={rc})")


def _zes_ensure_device_infos(device: int):
    """Ensure the ZES device info cache is populated and validate the device index."""
    if not _cached_zes_device_infos:
        if _enum_zes_device_infos(_parse_visible_devices(strict=True)) < 0:
            raise RuntimeError("Failed to enumerate devices via Level Zero Sysman.")

    total_devices = len(_cached_zes_device_infos)
    if device >= total_devices:
        raise RuntimeError(
            f"The device {device} is out of range for Level Zero Sysman. It must be in the range [0, {total_devices})."
        )


def _get_zes_temperature_handle(device: Device = None) -> c_void_p:
    r"""Return the Level Zero Sysman GPU temperature sensor handle for the specified device.

    The result is cached in ``_ZesDeviceInfo.temperature_handle`` so that
    repeated calls skip sensor enumeration.  ``_cached_zes_device_infos``
    is lazily populated on the first call.

    Args:
        device (torch.device, str or int, optional): target device. Uses the
            current device, given by :func:`~torch.xpu.current_device`,
            if ``None`` (default).
    """
    pyzes = _import_pyzes()

    device = _get_device_index(device, optional=True)
    _zes_ensure_device_infos(device)

    info = _cached_zes_device_infos[device]
    if info.temperature_handle is not None:
        return info.temperature_handle

    device_handle = info.device_handle
    subdevice_id = info.subdevice_id

    # Note [telemetry handle selection]:
    # For tiled dGPUs, pick the handle whose subdeviceId matches.
    # For non-tiled devices, pick the root-level (non-subdevice) handle.
    temp_count = c_uint32(0)
    _zes_check(
        pyzes.zesDeviceEnumTemperatureSensors(device_handle, byref(temp_count), None),
        "Can't get Level Zero Sysman temperature sensor count.",
    )
    if temp_count.value == 0:
        raise RuntimeError("No Level Zero Sysman temperature sensors found.")
    temp_handles = (pyzes.zes_temp_handle_t * temp_count.value)()
    _zes_check(
        pyzes.zesDeviceEnumTemperatureSensors(
            device_handle, byref(temp_count), temp_handles
        ),
        "Can't get Level Zero Sysman temperature sensor handles.",
    )

    temperature_handle = None
    for temp_handle in temp_handles:
        temp_props = pyzes.zes_temp_properties_t()
        temp_props.stype = pyzes.ZES_STRUCTURE_TYPE_TEMP_PROPERTIES
        _zes_check(
            pyzes.zesTemperatureGetProperties(temp_handle, byref(temp_props)),
            "Can't get Level Zero Sysman temperature properties.",
        )
        if temp_props.type != pyzes.ZES_TEMP_SENSORS_GPU:
            continue
        if subdevice_id is not None:
            if temp_props.onSubdevice and temp_props.subdeviceId == subdevice_id:
                temperature_handle = temp_handle
                break
        else:
            if not temp_props.onSubdevice:
                temperature_handle = temp_handle
                break

    if temperature_handle is None:
        raise RuntimeError("No Level Zero Sysman GPU temperature handle found.")
    info.temperature_handle = temperature_handle
    return temperature_handle


def temperature(device: Device = None) -> float:
    r"""Return the GPU temperature in degrees Celsius.

    Args:
        device (torch.device, str or int, optional): selected device. Uses the
            current device, given by :func:`~torch.xpu.current_device`,
            if ``None`` (default).

    .. note:: This API may require elevated privileges (e.g. ``sudo``) to access GPU temperature information.
    """
    temperature_handle = _get_zes_temperature_handle(device)

    import pyzes  # type: ignore[import]

    temp = c_double(0.0)
    rc = pyzes.zesTemperatureGetState(temperature_handle, byref(temp))
    if rc == pyzes.ZE_RESULT_ERROR_NOT_AVAILABLE:
        raise RuntimeError(
            "GPU temperature querying is not available. Try running with elevated privileges (e.g. sudo)."
        )
    if rc != pyzes.ZE_RESULT_SUCCESS:
        raise RuntimeError(f"Can't get Level Zero Sysman GPU temperature (rc={rc}).")
    return temp.value


def _get_zes_frequency_handle(device: Device = None) -> c_void_p:
    r"""Return the Level Zero Sysman GPU frequency domain handle for the specified device.

    The result is cached in ``_ZesDeviceInfo.frequency_handle`` so that
    repeated calls skip domain enumeration.  ``_cached_zes_device_infos``
    is lazily populated on the first call.

    Args:
        device (torch.device, str or int, optional): target device. Uses the
            current device, given by :func:`~torch.xpu.current_device`,
            if ``None`` (default).
    """
    pyzes = _import_pyzes()

    device = _get_device_index(device, optional=True)
    _zes_ensure_device_infos(device)

    info = _cached_zes_device_infos[device]
    if info.frequency_handle is not None:
        return info.frequency_handle

    device_handle = info.device_handle
    subdevice_id = info.subdevice_id

    # Enumerate all frequency domains under this device handle.
    freq_count = c_uint32(0)
    _zes_check(
        pyzes.zesDeviceEnumFrequencyDomains(device_handle, byref(freq_count), None),
        "Can't get Level Zero Sysman frequency domains count.",
    )
    if freq_count.value == 0:
        raise RuntimeError("No Level Zero Sysman frequency domains found.")
    freq_handles = (pyzes.zes_freq_handle_t * freq_count.value)()
    _zes_check(
        pyzes.zesDeviceEnumFrequencyDomains(
            device_handle, byref(freq_count), freq_handles
        ),
        "Can't get Level Zero Sysman frequency domain handles.",
    )

    if _get_pyzes_version() < (0, 1, 2):
        frequency_handle = freq_handles[0]
        info.frequency_handle = frequency_handle
        return frequency_handle

    frequency_handle = None
    for freq_handle in freq_handles:
        freq_props = pyzes.zes_freq_properties_t()
        freq_props.stype = pyzes.ZES_STRUCTURE_TYPE_FREQ_PROPERTIES
        _zes_check(
            pyzes.zesFrequencyGetProperties(freq_handle, byref(freq_props)),
            "Can't get Level Zero Sysman frequency properties.",
        )
        if freq_props.type != pyzes.ZES_FREQ_DOMAIN_GPU:
            continue
        if subdevice_id is not None:
            if freq_props.onSubdevice and freq_props.subdeviceId == subdevice_id:
                frequency_handle = freq_handle
                break
        else:
            if not freq_props.onSubdevice:
                frequency_handle = freq_handle
                break

    if frequency_handle is None:
        raise RuntimeError("No Level Zero Sysman GPU frequency handle found.")
    info.frequency_handle = frequency_handle
    return frequency_handle


def clock_rate(device: Device = None) -> float:
    r"""Return the GPU clock rate in MHz.

    Args:
        device (torch.device, str or int, optional): selected device. Uses the
            current device, given by :func:`~torch.xpu.current_device`,
            if ``None`` (default).
    """
    frequency_handle = _get_zes_frequency_handle(device)

    import pyzes  # type: ignore[import]

    freq_state = pyzes.zes_freq_state_t()
    rc = pyzes.zesFrequencyGetState(frequency_handle, byref(freq_state))
    if rc != pyzes.ZE_RESULT_SUCCESS:
        raise RuntimeError(f"Can't get Level Zero Sysman GPU clock rate (rc={rc}).")
    return freq_state.actual


def _get_zes_power_handle(device: Device = None) -> c_void_p:
    r"""Return the Level Zero Sysman GPU power domain handle for the specified device.

    The result is cached in ``_ZesDeviceInfo.power_handle`` so that
    repeated calls skip domain enumeration.  ``_cached_zes_device_infos``
    is lazily populated on the first call.

    Args:
        device (torch.device, str or int, optional): target device. Uses the
            current device, given by :func:`~torch.xpu.current_device`,
            if ``None`` (default).
    """
    pyzes = _import_pyzes()

    device = _get_device_index(device, optional=True)
    _zes_ensure_device_infos(device)

    info = _cached_zes_device_infos[device]
    if info.power_handle is not None:
        return info.power_handle

    device_handle = info.device_handle
    subdevice_id = info.subdevice_id

    # Enumerate all power domains under this device handle.
    power_count = c_uint32(0)
    _zes_check(
        pyzes.zesDeviceEnumPowerDomains(device_handle, byref(power_count), None),
        "Can't get Level Zero Sysman power domains count.",
    )
    if power_count.value == 0:
        raise RuntimeError("No Level Zero Sysman power domains found.")
    power_handles = (pyzes.zes_pwr_handle_t * power_count.value)()
    _zes_check(
        pyzes.zesDeviceEnumPowerDomains(
            device_handle, byref(power_count), power_handles
        ),
        "Can't get Level Zero Sysman power domain handles.",
    )

    if _get_pyzes_version() < (0, 1, 2):
        power_handle = power_handles[0]
        info.power_handle = power_handle
        return power_handle

    power_handle = None
    for pwr_handle in power_handles:
        pwr_props = pyzes.zes_power_properties_t()
        pwr_props.stype = pyzes.ZES_STRUCTURE_TYPE_POWER_PROPERTIES
        ext_pwr_props = pyzes.zes_power_ext_properties_t()
        ext_pwr_props.stype = pyzes.ZES_STRUCTURE_TYPE_POWER_EXT_PROPERTIES
        pwr_props.pNext = cast(pointer(ext_pwr_props), c_void_p)
        _zes_check(
            pyzes.zesPowerGetProperties(pwr_handle, byref(pwr_props)),
            "Can't get Level Zero Sysman power properties.",
        )
        if ext_pwr_props.domain not in (
            pyzes.ZES_POWER_DOMAIN_CARD,
            pyzes.ZES_POWER_DOMAIN_PACKAGE,
        ):
            continue
        if subdevice_id is not None:
            if pwr_props.onSubdevice and pwr_props.subdeviceId == subdevice_id:
                power_handle = pwr_handle
                if ext_pwr_props.domain == pyzes.ZES_POWER_DOMAIN_CARD:
                    break
        else:
            if not pwr_props.onSubdevice:
                power_handle = pwr_handle
                if ext_pwr_props.domain == pyzes.ZES_POWER_DOMAIN_CARD:
                    break

    if power_handle is None:
        raise RuntimeError("No Level Zero Sysman GPU power handle found.")
    info.power_handle = power_handle
    return power_handle


def power_draw(device: Device = None) -> float:
    r"""Return the GPU card power draw in watts.

    The value is computed by dividing the energy delta by the time delta between
    two energy-counter reads separated by a 100ms sampling interval.

    Args:
        device (torch.device, str or int, optional): selected device. Uses the
            current device, given by :func:`~torch.xpu.current_device`,
            if ``None`` (default).

    .. note:: This function blocks for approximately 100ms per call due to the
        sampling interval required to compute an accurate power reading.

    .. note:: This API may require elevated privileges (e.g. ``sudo``) to access GPU power information.

    .. note:: On Intel Xe2 and newer GPUs, card-level power is reported directly. On older GPUs,
        package-level power is used as a fallback and may not reflect the full card power draw.
    """
    power_handle = _get_zes_power_handle(device)

    import pyzes  # type: ignore[import]

    counter_start = pyzes.zes_power_energy_counter_t()
    rc = pyzes.zesPowerGetEnergyCounter(power_handle, byref(counter_start))
    if rc == pyzes.ZE_RESULT_ERROR_NOT_AVAILABLE:
        raise RuntimeError(
            "GPU power draw querying is not available. Try running with elevated privileges (e.g. sudo)."
        )
    if rc != pyzes.ZE_RESULT_SUCCESS:
        raise RuntimeError(f"Can't get Level Zero Sysman GPU power draw (rc={rc}).")

    import time

    time.sleep(_zes_sample_interval_ms / 1000.0)

    counter_end = pyzes.zes_power_energy_counter_t()
    _zes_check(
        pyzes.zesPowerGetEnergyCounter(power_handle, byref(counter_end)),
        "Can't get Level Zero Sysman GPU power energy counter.",
    )
    # energy is in microjoules, timestamp is in microseconds (per L0 Sysman spec).
    # microjoules / microseconds = watts, so the micro factors cancel.
    dt = counter_end.timestamp - counter_start.timestamp
    return (counter_end.energy - counter_start.energy) / dt


def _get_zes_engine_handle(device: Device = None) -> c_void_p:
    r"""Return the Level Zero Sysman GPU engine group handle for the specified device.

    The result is cached in ``_ZesDeviceInfo.engine_handle`` so that
    repeated calls skip group enumeration.  ``_cached_zes_device_infos``
    is lazily populated on the first call.

    Args:
        device (torch.device, str or int, optional): target device. Uses the
            current device, given by :func:`~torch.xpu.current_device`,
            if ``None`` (default).
    """
    pyzes = _import_pyzes()

    device = _get_device_index(device, optional=True)
    _zes_ensure_device_infos(device)

    info = _cached_zes_device_infos[device]
    if info.engine_handle is not None:
        return info.engine_handle

    device_handle = info.device_handle
    subdevice_id = info.subdevice_id

    # See Note [telemetry handle selection]
    engine_count = c_uint32(0)
    _zes_check(
        pyzes.zesDeviceEnumEngineGroups(device_handle, byref(engine_count), None),
        "Can't get Level Zero Sysman engine group count.",
    )
    # TODO: zesDeviceEnumEngineGroups does not return ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS on privilege errors;
    # instead it succeeds with count=0. Treat that as an error with a helpful hint about elevated privileges.
    if engine_count.value == 0:
        raise RuntimeError(
            "No Level Zero Sysman engine groups found. The GPU may not support engine monitoring, or try running with elevated privileges (e.g. sudo)."
        )
    engine_handles = (pyzes.zes_engine_handle_t * engine_count.value)()
    _zes_check(
        pyzes.zesDeviceEnumEngineGroups(
            device_handle, byref(engine_count), engine_handles
        ),
        "Can't get Level Zero Sysman engine group handles.",
    )

    engine_handle = None
    for eng_handle in engine_handles:
        eng_props = pyzes.zes_engine_properties_t()
        eng_props.stype = pyzes.ZES_STRUCTURE_TYPE_ENGINE_PROPERTIES
        _zes_check(
            pyzes.zesEngineGetProperties(eng_handle, byref(eng_props)),
            "Can't get Level Zero Sysman engine properties.",
        )
        if eng_props.type != pyzes.ZES_ENGINE_GROUP_ALL:
            continue
        if subdevice_id is not None:
            if eng_props.onSubdevice and eng_props.subdeviceId == subdevice_id:
                engine_handle = eng_handle
                break
        else:
            if not eng_props.onSubdevice:
                engine_handle = eng_handle
                break

    if engine_handle is None:
        raise RuntimeError("No Level Zero Sysman GPU engine handle found.")
    info.engine_handle = engine_handle
    return engine_handle


def utilization(device: Device = None) -> float:
    r"""Return the GPU engine utilization as a percentage.

    The value is computed by dividing the active-time delta by the time delta
    between two engine-activity reads separated by a 100ms sampling interval.

    Args:
        device (torch.device, str or int, optional): selected device. Uses the
            current device, given by :func:`~torch.xpu.current_device`,
            if ``None`` (default).

    .. note:: This function blocks for approximately 100ms per call due to the
        sampling interval required to compute an accurate utilization reading.

    .. note:: This API may require elevated privileges (e.g. ``sudo``) to access GPU utilization information.
    """
    engine_handle = _get_zes_engine_handle(device)

    import pyzes  # type: ignore[import]

    stats_start = pyzes.zes_engine_stats_t()
    rc = pyzes.zesEngineGetActivity(engine_handle, byref(stats_start))
    if rc == pyzes.ZE_RESULT_ERROR_NOT_AVAILABLE:
        raise RuntimeError(
            "GPU utilization querying is not available. Try running with elevated privileges (e.g. sudo)."
        )
    if rc != pyzes.ZE_RESULT_SUCCESS:
        raise RuntimeError(
            f"Can't get Level Zero Sysman GPU engine activity (rc={rc})."
        )

    import time

    time.sleep(_zes_sample_interval_ms / 1000.0)

    stats_end = pyzes.zes_engine_stats_t()
    _zes_check(
        pyzes.zesEngineGetActivity(engine_handle, byref(stats_end)),
        "Can't get Level Zero Sysman GPU engine activity.",
    )
    # activeTime and timestamp are monotonic counters in microseconds.
    dt = stats_end.timestamp - stats_start.timestamp
    return (stats_end.activeTime - stats_start.activeTime) / dt * 100


def _zes_get_memory_handle(device: Device = None) -> c_void_p:
    r"""Return the Level Zero Sysman GPU memory module handle for the specified device.

    The result is cached in ``_ZesDeviceInfo.memory_handle`` so that
    repeated calls skip module enumeration.  ``_cached_zes_device_infos``
    is lazily populated on the first call.

    Args:
        device (torch.device, str or int, optional): target device. Uses the
            current device, given by :func:`~torch.xpu.current_device`,
            if ``None`` (default).
    """
    pyzes = _import_pyzes()

    device = _get_device_index(device, optional=True)
    _zes_ensure_device_infos(device)

    info = _cached_zes_device_infos[device]
    if info.memory_handle is not None:
        return info.memory_handle

    device_handle = info.device_handle
    subdevice_id = info.subdevice_id

    # See Note [telemetry handle selection]
    mem_count = c_uint32(0)
    _zes_check(
        pyzes.zesDeviceEnumMemoryModules(device_handle, byref(mem_count), None),
        "Can't get Level Zero Sysman memory module count.",
    )
    if mem_count.value == 0:
        raise RuntimeError("No Level Zero Sysman memory modules found.")
    memory_handles = (pyzes.zes_mem_handle_t * mem_count.value)()
    _zes_check(
        pyzes.zesDeviceEnumMemoryModules(
            device_handle, byref(mem_count), memory_handles
        ),
        "Can't get Level Zero Sysman memory module handles.",
    )

    memory_handle = None
    for mem_handle in memory_handles:
        mem_props = pyzes.zes_mem_properties_t()
        mem_props.stype = pyzes.ZES_STRUCTURE_TYPE_MEM_PROPERTIES
        _zes_check(
            pyzes.zesMemoryGetProperties(mem_handle, byref(mem_props)),
            "Can't get Level Zero Sysman memory properties.",
        )
        if mem_props.location != pyzes.ZES_MEM_LOC_DEVICE:
            continue
        if subdevice_id is not None:
            if mem_props.onSubdevice and mem_props.subdeviceId == subdevice_id:
                memory_handle = mem_handle
                break
        else:
            if not mem_props.onSubdevice:
                memory_handle = mem_handle
                break

    if memory_handle is None:
        raise RuntimeError("No Level Zero Sysman GPU memory handle found.")
    info.memory_handle = memory_handle
    return memory_handle


def memory_usage(device: Device = None) -> float:
    r"""Return the GPU memory bandwidth usage as a percentage.

    The value is computed by dividing the byte-transfer delta by the time delta
    between two bandwidth-counter reads separated by a 100ms sampling interval,
    then normalizing by the peak bandwidth reported by the hardware.

    Args:
        device (torch.device, str or int, optional): selected device. Uses the
            current device, given by :func:`~torch.xpu.current_device`,
            if ``None`` (default).

    .. note:: This function blocks for approximately 100ms per call due to the
        sampling interval required to compute an accurate bandwidth reading.

    .. note:: This API may require elevated privileges (e.g. ``sudo``) to access GPU memory bandwidth usage information.
    """
    memory_handle = _zes_get_memory_handle(device)

    import pyzes  # type: ignore[import]

    bandwidth_start = pyzes.zes_mem_bandwidth_t()
    rc = pyzes.zesMemoryGetBandwidth(memory_handle, byref(bandwidth_start))
    if rc == pyzes.ZE_RESULT_ERROR_NOT_AVAILABLE:
        raise RuntimeError(
            "GPU memory bandwidth usage querying is not available. Try running with elevated privileges (e.g. sudo)."
        )
    if rc != pyzes.ZE_RESULT_SUCCESS:
        raise RuntimeError(
            f"Can't get Level Zero Sysman GPU memory bandwidth (rc={rc})."
        )

    import time

    time.sleep(_zes_sample_interval_ms / 1000.0)

    bandwidth_end = pyzes.zes_mem_bandwidth_t()
    _zes_check(
        pyzes.zesMemoryGetBandwidth(memory_handle, byref(bandwidth_end)),
        "Can't get Level Zero Sysman GPU memory bandwidth.",
    )
    dt = bandwidth_end.timestamp - bandwidth_start.timestamp
    # readCounter and writeCounter are cumulative byte counts (per L0 Sysman spec).
    read_delta = bandwidth_end.readCounter - bandwidth_start.readCounter
    write_delta = bandwidth_end.writeCounter - bandwidth_start.writeCounter
    # maxBandwidth is in bytes/sec; dt is in microseconds; the 1e6 factor converts dt to seconds.
    return 1e6 * (read_delta + write_delta) / (bandwidth_end.maxBandwidth * dt) * 100


def device_memory_used(device: Device = None) -> int:
    r"""Return the current GPU used global (device) memory in bytes.

    Args:
        device (torch.device, str or int, optional): selected device. Uses the
            current device, given by :func:`~torch.xpu.current_device`,
            if ``None`` (default).

    .. note:: This API may require elevated privileges (e.g. ``sudo``) to access GPU memory usage information.
    """
    memory_handle = _zes_get_memory_handle(device)

    import pyzes  # type: ignore[import]

    mem_state = pyzes.zes_mem_state_t()
    rc = pyzes.zesMemoryGetState(memory_handle, byref(mem_state))
    if rc == pyzes.ZE_RESULT_ERROR_NOT_AVAILABLE:
        raise RuntimeError(
            "GPU memory usage querying is not available. Try running with elevated privileges (e.g. sudo)."
        )
    if rc != pyzes.ZE_RESULT_SUCCESS:
        raise RuntimeError(f"Can't get Level Zero Sysman GPU memory state (rc={rc}).")
    mem_props = pyzes.zes_mem_properties_t()
    mem_props.stype = pyzes.ZES_STRUCTURE_TYPE_MEM_PROPERTIES
    _zes_check(
        pyzes.zesMemoryGetProperties(memory_handle, byref(mem_props)),
        "Can't get Level Zero Sysman memory properties.",
    )
    # TODO: Some drivers report physicalSize as 0 on client GPUs; fall back to
    # the allocatable size as an approximation in that case.
    total = mem_props.physicalSize if mem_props.physicalSize != 0 else mem_state.size
    return total - mem_state.free


# import here to avoid circular import
from .memory import (
    change_current_allocator,
    empty_cache,
    get_per_process_memory_fraction,
    list_gpu_processes,
    max_memory_allocated,
    max_memory_reserved,
    mem_get_info,
    memory_allocated,
    memory_reserved,
    memory_snapshot,
    memory_stats,
    memory_stats_as_nested_dict,
    MemPool,
    reset_accumulated_memory_stats,
    reset_peak_memory_stats,
    set_per_process_memory_fraction,
    use_mem_pool,
    XPUPluggableAllocator,
)
from .random import (
    get_rng_state,
    get_rng_state_all,
    initial_seed,
    manual_seed,
    manual_seed_all,
    seed,
    seed_all,
    set_rng_state,
    set_rng_state_all,
)


_POOL_HANDLE = NewType("_POOL_HANDLE", tuple[int, int])
__all__ = [
    "Event",
    "Stream",
    "StreamContext",
    "XPUPluggableAllocator",
    "XPUGraph",
    "can_device_access_peer",
    "change_current_allocator",
    "clock_rate",
    "current_device",
    "current_stream",
    "default_generators",
    "device",
    "device_of",
    "device_count",
    "device_memory_used",
    "empty_cache",
    "get_arch_list",
    "get_device_capability",
    "get_device_name",
    "get_device_properties",
    "get_gencode_flags",
    "get_per_process_memory_fraction",
    "get_rng_state",
    "get_rng_state_all",
    "get_stream_from_external",
    "graph",
    "graph_pool_handle",
    "init",
    "initial_seed",
    "is_available",
    "is_bf16_supported",
    "is_current_stream_capturing",
    "is_initialized",
    "is_tf32_supported",
    "list_gpu_processes",
    "make_graphed_callables",
    "manual_seed",
    "manual_seed_all",
    "max_memory_allocated",
    "max_memory_reserved",
    "mem_get_info",
    "memory_allocated",
    "memory_reserved",
    "memory_snapshot",
    "memory_stats",
    "memory_stats_as_nested_dict",
    "memory_usage",
    "MemPool",
    "power_draw",
    "use_mem_pool",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
    "seed",
    "seed_all",
    "set_device",
    "set_per_process_memory_fraction",
    "set_rng_state",
    "set_rng_state_all",
    "set_stream",
    "stream",
    "streams",
    "synchronize",
    "temperature",
    "utilization",
]
<|endoftext|># mypy: allow-untyped-defs
# pylint: disable=useless-parent-delegation
from __future__ import annotations

import ctypes

import torch
from torch._utils import _dummy_type


if not hasattr(torch._C, "_XpuStreamBase"):
    # Define dummy base classes
    torch._C.__dict__["_XpuStreamBase"] = _dummy_type("_XpuStreamBase")
    torch._C.__dict__["_XpuEventBase"] = _dummy_type("_XpuEventBase")


class Stream(torch._C._XpuStreamBase):
    r"""Wrapper around a XPU stream.

    A XPU stream is a linear sequence of execution that belongs to a specific
    device, independent from other streams. It supports with statement as a
    context manager to ensure the operators within the with block are running
    on the corresponding stream.

    Args:
        device(torch.device or int, optional): a device on which to allocate
            the stream. If :attr:`device` is ``None`` (default) or a negative
            integer, this will use the current device.
        priority(int, optional): priority of the stream, which can be positive, 0, or negative.
            A lower number indicates a higher priority. By default, the priority is set to 0.
            If the value falls outside of the allowed priority range, it will automatically be
            mapped to the nearest valid priority (lowest for large positive numbers or
            highest for large negative numbers).
    """

    def __new__(cls, device=None, priority=0, **kwargs):
        # setting device manager is expensive, so we avoid it unless necessary
        if device is None or ("stream_id" in kwargs and "device_index" in kwargs):
            return super().__new__(cls, priority=priority, **kwargs)
        else:
            with torch.xpu.device(device):
                return super().__new__(cls, priority=priority, **kwargs)

    def wait_event(self, event: Event | torch.Event) -> None:
        r"""Make all future work submitted to the stream wait for an event.

        Args:
            event (Event, torch.Event): an event to wait for.
        """
        event.wait(self)

    def wait_stream(self, stream: Stream | torch.Stream) -> None:
        r"""Synchronize with another stream.

        All future work submitted to this stream will wait until all kernels
        submitted to a given stream at the time of call complete.

        Args:
            stream (Stream, torch.Stream): a stream to synchronize.
        """
        self.wait_event(stream.record_event())

    def record_event(self, event: Event | torch.Event | None = None):
        r"""Record an event.

        Args:
            event (Event, torch.Event, optional): event to record. If not given, a new one
                will be allocated.

        Returns:
            Recorded event.
        """
        if event is None:
            event = Event()
        event.record(self)
        return event

    def query(self) -> bool:
        r"""Check if all the work submitted has been completed.

        Returns:
            A boolean indicating if all kernels in this stream are completed.
        """
        return super().query()

    def synchronize(self) -> None:
        r"""Wait for all the kernels in this stream to complete."""
        super().synchronize()

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.sycl_queue)

    def __eq__(self, o):
        if isinstance(o, Stream):
            return super().__eq__(o)
        return False

    def __hash__(self):
        return hash((self.sycl_queue, self.device))

    def __repr__(self) -> str:
        return f"torch.xpu.Stream(device={self.device} sycl_queue={self.sycl_queue:#x})"


class Event(torch._C._XpuEventBase):
    r"""Wrapper around a XPU event.

    XPU events are synchronization markers that can be used to monitor the
    device's progress, and to synchronize XPU streams.

    The underlying XPU events are lazily initialized when the event is first
    recorded. After creation, only streams on the same device may record the
    event. However, streams on any device can wait on the event.

    Args:
        enable_timing (bool, optional): indicates if the event should measure time
            (default: ``False``)
    """

    def __new__(cls, enable_timing=False):
        return super().__new__(cls, enable_timing=enable_timing)

    def record(self, stream: Stream | torch.Stream | None = None) -> None:
        r"""Record the event in a given stream.

        Args:
            stream (Stream, torch.Stream, optional): Uses ``torch.xpu.current_stream()`` if no stream is specified.
                The stream's device must match the event's device.
        """
        if stream is None:
            stream = torch.xpu.current_stream()
        super().record(stream)

    def wait(self, stream: Stream | torch.Stream | None = None) -> None:
        r"""Make all future work submitted to the given stream wait for this event.

        Args:
            stream (Stream, torch.Stream, optional): Uses ``torch.xpu.current_stream()`` if no stream is specified.
        """
        if stream is None:
            stream = torch.xpu.current_stream()
        super().wait(stream)

    def query(self) -> bool:
        r"""Check if all work currently captured by event has completed.

        Returns:
            A boolean indicating if all work currently captured by event has
            completed.
        """
        return super().query()

    def elapsed_time(self, end_event: Event):
        r"""Return the time elapsed.

        Time reported in milliseconds after the event was recorded and
        before the end_event was recorded.

        Args:
            end_event (Event): the end event.
        """
        return super().elapsed_time(end_event)

    def synchronize(self) -> None:
        r"""Wait for the event to complete.

        Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.
        """
        super().synchronize()

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.sycl_event)

    def __repr__(self) -> str:
        if self.sycl_event:
            return f"torch.xpu.Event(sycl_event={self.sycl_event:#x})"
        else:
            return "torch.xpu.Event(uninitialized)"
<|endoftext|>from typing import Any

import torch

# The _get_device_index has been moved to torch.utils._get_device_index
from torch._utils import _get_device_index as _torch_get_device_index


def _get_device_index(
    device: Any, optional: bool = False, allow_cpu: bool = False
) -> int:
    r"""Get the device index from :attr:`device`, which can be a torch.device
    object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a XPU device. Note that for a XPU device without a specified index,
    i.e., ``torch.device('xpu')``, this will return the current default XPU
    device if :attr:`optional` is ``True``. If :attr:`allow_cpu` is ``True``,
    CPU devices will be accepted and ``-1`` will be returned in this case.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default XPU
    device if :attr:`optional` is ``True``.
    """
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        if allow_cpu:
            if device.type not in ["xpu", "cpu"]:
                raise ValueError(f"Expected a xpu or cpu device, but got: {device}")
        elif device.type != "xpu":
            raise ValueError(f"Expected a xpu device, but got: {device}")
    if not torch.jit.is_scripting():
        if isinstance(device, torch.xpu.device):
            return device.idx
    return _torch_get_device_index(device, optional, allow_cpu)
<|endoftext|>from __future__ import annotations

import typing
from collections.abc import Callable
from typing import overload, TYPE_CHECKING, TypeAlias
from typing_extensions import ParamSpec, Self, TypeVar

import torch
from torch import Tensor


if TYPE_CHECKING:
    from torch.xpu import _POOL_HANDLE

from .._utils import _dummy_type


__all__ = [
    "is_current_stream_capturing",
    "graph_pool_handle",
    "XPUGraph",
    "graph",
    "make_graphed_callables",
]

_R = TypeVar("_R")
_P = ParamSpec("_P")

if not hasattr(torch._C, "_XpuStreamBase"):
    # Define dummy base classes
    torch._C.__dict__["_XPUGraph"] = _dummy_type("_XPUGraph")
    torch._C.__dict__["_xpu_graph_pool_handle"] = _dummy_type("_xpu_graph_pool_handle")
    torch._C.__dict__["_xpu_isCurrentStreamCapturing"] = _dummy_type(
        "_xpu_isCurrentStreamCapturing"
    )

from torch._C import _xpu_graph_pool_handle, _xpu_isCurrentStreamCapturing, _XPUGraph


def is_current_stream_capturing() -> bool:
    r"""Return True if XPU graph capture is underway on the current XPU stream, False otherwise.

    If a XPU context does not exist on the current device, returns False without initializing the context.
    """
    return _xpu_isCurrentStreamCapturing()


def graph_pool_handle() -> _POOL_HANDLE:
    r"""Return an opaque token representing the id of a graph memory pool."""
    return torch.xpu._POOL_HANDLE(_xpu_graph_pool_handle())


class XPUGraph(_XPUGraph):
    r"""Wrapper around a XPU graph.

    Arguments:
        keep_graph (bool, optional): If ``keep_graph=False``, the
            executable command graph will be instantiated on GPU at the end of
            ``capture_end`` and the underlying modifiable command graph will be
            destroyed. Note that the executable command graph will not be
            instantiated at the end of ``capture_end`` in this
            case. Instead, it will be instantiated via an explicit called
            to ``instantiate`` or automatically on the first call to
            ``replay`` if ``instantiate`` was not already called. Calling
            ``instantiate`` manually before ``replay`` is recommended to
            prevent increased latency on the first call to ``replay``.

    """

    def __new__(cls, keep_graph: bool = False) -> Self:
        return super().__new__(cls, keep_graph)

    def capture_begin(self, pool: _POOL_HANDLE | None = None) -> None:
        r"""Begin capturing XPU work on the current xpu stream.

        Typically, you shouldn't call ``capture_begin`` yourself.
        Use :class:`~torch.xpu.graph`, which call ``capture_begin`` internally.

        Arguments:
            pool (optional): Token (returned by :func:`~torch.xpu.graph_pool_handle` or
                :meth:`other_Graph_instance.pool()<torch.xpu.XPUGraph.pool>`) that hints this graph may share memory
                with the indicated pool.
        """
        super().capture_begin(pool=pool)

    def capture_end(self) -> None:
        r"""End XPU graph capture on the current stream.

        After ``capture_end``, ``replay`` may be called on this instance.

        Typically, you shouldn't call ``capture_end`` yourself.
        Use :class:`~torch.xpu.graph`, which call ``capture_end`` internally.
        """
        super().capture_end()

    def instantiate(self) -> None:
        r"""Instantiate the XPU graph. Will be called by
        ``capture_end`` if ``keep_graph=False``, or by ``replay`` if
        ``keep_graph=True`` and ``instantiate`` has not already been
        explicitly called. Does not destroy the xpu modify command graph returned
        by ``raw_xpu_graph``.
        """
        super().instantiate()

    def replay(self) -> None:
        r"""Replay the XPU work captured by this graph."""
        super().replay()

    def reset(self) -> None:
        r"""Delete the graph currently held by this instance."""
        super().reset()

    def pool(self) -> _POOL_HANDLE:
        r"""Return an opaque token representing the id of this graph's memory pool.

        This id can optionally be passed to another graph's ``capture_begin``,
        which hints the other graph may share the same memory pool.
        """
        return super().pool()

    def enable_debug_mode(self) -> None:
        r"""Enable debugging mode for XPUGraph.debug_dump."""
        return super().enable_debug_mode()

    def debug_dump(self, debug_path: str) -> None:
        r"""
        Arguments:
            debug_path (required): Path to dump the graph to.

        Calls a debugging function to dump the graph if the debugging is
        enabled via XPUGraph.enable_debug_mode()
        """
        return super().debug_dump(debug_path)

    def raw_xpu_graph(self) -> int:
        r"""Returns the underlying xpuGraph_t. ``keep_graph`` must be True.

        XPU doesn't provide APIs to manipulate this object.
        """
        return super().raw_xpu_graph()

    def raw_xpu_graph_exec(self) -> int:
        r"""Returns the underlying xpuGraphExec_t. ``instantiate`` must have been called if ``keep_graph`` is True, or ``capture_end`` must have been called if ``keep_graph`` is False. If you call ``instantiate()`` after ``raw_xpu_graph_exec()``, the previously returned xpuGraphExec_t will be destroyed. It is your responsibility not to use this object after destruction.

        XPU doesn't provide APIs to manipulate this object.
        """
        return super().raw_xpu_graph_exec()


class graph:
    r"""Context-manager that captures XPU work into a :class:`torch.xpu.XPUGraph` object for later replay.

    Arguments:
        xpu_graph (torch.xpu.XPUGraph): Graph object used for capture.
        pool (optional): Opaque token (returned by a call to :func:`~torch.xpu.graph_pool_handle()` or
            :meth:`other_Graph_instance.pool()<torch.xpu.XPUGraph.pool>`) hinting this graph's capture
            may share memory from the specified pool.
        stream (torch.xpu.Stream, optional): If supplied, will be set as the current stream in the context.
            If not supplied, ``graph`` sets its own internal side stream as the current stream in the context.

    .. note::
        For effective memory sharing, if you pass a ``pool`` used by a previous capture and the previous capture
        used an explicit ``stream`` argument, you should pass the same ``stream`` argument to this capture.

    """

    default_capture_stream: torch.xpu.Stream | None = None

    def __init__(
        self,
        xpu_graph: XPUGraph,
        pool: _POOL_HANDLE | None = None,
        stream: torch.xpu.Stream | None = None,
    ):
        # Lazy-init of default_capture_stream helps avoid circular-import errors.
        # Not thread safe, but graphs already have the general (explicitly documented)
        # restriction that only one capture may be underway at a time in the process.
        if self.__class__.default_capture_stream is None:
            self.__class__.default_capture_stream = torch.xpu.Stream()

        self.pool: tuple[()] | tuple[_POOL_HANDLE] = () if pool is None else (pool,)
        self.capture_stream = (
            stream if stream is not None else self.__class__.default_capture_stream
        )
        if self.capture_stream is None:
            raise AssertionError("capture_stream must not be None")
        self.stream_ctx = self.capture_stream
        self.xpu_graph = xpu_graph

    def __enter__(self) -> None:
        # Free as much memory as we can for the graph
        torch.xpu.synchronize()

        torch.xpu.empty_cache()
        self.stream_ctx.__enter__()

        self.xpu_graph.capture_begin(*self.pool)

    def __exit__(self, *args: object) -> None:
        self.xpu_graph.capture_end()
        self.stream_ctx.__exit__(*args)


_ModuleOrCallable: TypeAlias = torch.nn.Module | Callable[..., object]


@overload
def make_graphed_callables(
    callables: _ModuleOrCallable,
    sample_args: tuple[Tensor, ...],
    num_warmup_iters: int = 3,
    allow_unused_input: bool = False,
    pool: _POOL_HANDLE | None = None,
) -> _ModuleOrCallable: ...


@overload
def make_graphed_callables(
    callables: tuple[_ModuleOrCallable, ...],
    sample_args: tuple[tuple[Tensor, ...], ...],
    num_warmup_iters: int = 3,
    allow_unused_input: bool = False,
    pool: _POOL_HANDLE | None = None,
) -> tuple[_ModuleOrCallable, ...]: ...


def make_graphed_callables(
    callables: _ModuleOrCallable | tuple[_ModuleOrCallable, ...],
    sample_args: tuple[Tensor, ...] | tuple[tuple[Tensor, ...], ...],
    num_warmup_iters: int = 3,
    allow_unused_input: bool = False,
    pool: _POOL_HANDLE | None = None,
) -> _ModuleOrCallable | tuple[_ModuleOrCallable, ...]:
    r"""Accept callables (functions or :class:`nn.Module<torch.nn.Module>`\ s) and returns graphed versions.

    Each graphed callable's forward pass runs its source callable's
    forward XPU work as a XPU graph inside a single autograd node.

    The graphed callable's forward pass also appends
    a backward node to the autograd graph. During backward, this node runs the
    callable's backward work as a XPU graph.

    Therefore, each graphed callable should be a drop-in replacement for its source callable
    in an autograd-enabled training loop.

    See :ref:`Partial-network capture<partial-network-capture>` for detailed use and constraints.

    If you pass a tuple of several callables, their captures will use the same memory pool.

    Arguments:
        callables (torch.nn.Module or Python function, or tuple of these): Callable or callables to graph.
            If you pass a tuple of callables, their order in the tuple must be the same order they'll run
            in the live workload.
        sample_args (tuple of Tensors, or tuple of tuples of Tensors): Samples args for each callable.
            If a single callable was passed, ``sample_args`` must be a single tuple of argument Tensors.
            If a tuple of callables was passed, ``sample_args`` must be tuple of tuples of argument Tensors.
        num_warmup_iters (int): The number of warmup iterations. Currently, ``DataDistributedParallel`` needs
            11 iterations for warm up. Default: ``3``.
        allow_unused_input (bool): If False, specifying inputs that were not used when computing outputs
            (and therefore their grad is always zero) is an error. Defaults to False.
        pool (optional): Token (returned by :func:`~torch.xpu.graph_pool_handle` or
            :meth:`other_Graph_instance.pool()<torch.xpu.XPUGraph.pool>`) that hints this graph may share memory
            with the indicated pool.
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
        :class:`torch.nn.Module`\s passed to :func:`~torch.xpu.make_graphed_callables` must not have module hooks
        registered on them at the time they are passed. However, registering hooks on modules *after* passing them
        through :func:`~torch.xpu.make_graphed_callables` is allowed.

    .. warning::
        When running a graphed callable, you must pass its arguments in the same order and format
        they appeared in that callable's ``sample_args``.

    .. warning::
        The automatic mixed precision is supported in :func:`~torch.xpu.make_graphed_callables` only with disabled
        caching. The context manager `torch.amp.autocast()` must have `cache_enabled=False`.
    """
    if torch.is_autocast_enabled("xpu") and torch.is_autocast_cache_enabled():
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
                raise RuntimeError(
                    "Modules must not have hooks registered at the time they are passed. However, registering hooks "
                    + "on modules after passing them through make_graphed_callables is allowed."
                )
            if not all(b.requires_grad is False for b in c.buffers()):
                raise RuntimeError(
                    "In any :class:`~torch.nn.Module` passed to "
                    + ":func:`~make_graphed_callables`, only parameters may be trainable. All buffers must have "
                    + "``requires_grad=False``."
                )
        flatten_arg = torch.utils._pytree.arg_tree_leaves(*args)
        flatten_sample_args.append(tuple(flatten_arg))
        if not all(isinstance(arg, torch.Tensor) for arg in flatten_arg):
            raise TypeError(
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

    fwd_graphs = [torch.xpu.XPUGraph() for _ in range(len(callables))]
    bwd_graphs = [torch.xpu.XPUGraph() for _ in range(len(callables))]

    mempool = graph_pool_handle() if pool is None else pool

    # Warmup
    torch.xpu.synchronize()
    with torch.xpu.stream(torch.xpu.Stream()):
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

    torch.xpu.synchronize()

    # Capture forward graphs
    per_callable_static_outputs = []
    per_callable_output_unflatten_spec = []
    for func, args, fwd_graph in zip(callables, _sample_args, fwd_graphs):
        # each graph uses the same mempool
        with torch.xpu.graph(fwd_graph, pool=mempool):
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
        static_grad_outputs = tuple(
            torch.empty_like(o) if o.requires_grad else None for o in static_outputs
        )

        outputs_grad = tuple(o for o in static_outputs if o.requires_grad)
        grad_inputs = None
        if len(outputs_grad) > 0:
            with torch.xpu.graph(bwd_graph, pool=mempool):
                grad_inputs = torch.autograd.grad(
                    outputs=outputs_grad,
                    inputs=tuple(i for i in static_input_surface if i.requires_grad),
                    grad_outputs=tuple(o for o in static_grad_outputs if o is not None),
                    only_inputs=True,
                    allow_unused=allow_unused_input,
                )

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

    def make_graphed_autograd_function(
        fwd_graph: XPUGraph,
        bwd_graph: XPUGraph,
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
                    raise RuntimeError("static_outputs must be a tuple")
                return tuple(o.detach() for o in static_outputs)

            @staticmethod
            @torch.autograd.function.once_differentiable
            # pyrefly: ignore [bad-override]
            def backward(ctx: object, *grads: Tensor) -> tuple[Tensor, ...]:
                if len(grads) != len(static_grad_outputs):
                    raise RuntimeError(
                        f"Expected {len(static_grad_outputs)} gradients but got {len(grads)}"
                    )
                for g, grad in zip(static_grad_outputs, grads):
                    if g is not None:
                        if g.data_ptr() != grad.data_ptr():
                            g.copy_(grad)
                bwd_graph.replay()

                if not isinstance(static_grad_inputs, tuple):
                    raise RuntimeError("static_grad_inputs must be a tuple")
                return tuple(
                    b.detach() if b is not None else b for b in static_grad_inputs
                )

        def functionalized(*user_args: object) -> object:
            # Runs the new autograd function which replays the XPU graphs
            flatten_user_args = torch.utils._pytree.arg_tree_leaves(*user_args)
            out = Graphed.apply(*(tuple(flatten_user_args) + module_params))
            return torch.utils._pytree.tree_unflatten(out, output_unflatten_spec)

        return functionalized

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
<|endoftext|>import torch
from torch import Tensor

from ._utils import _device_t, _get_device_index


def initial_seed(device: _device_t = None, /) -> int:
    r"""Return the initial seed of the default :class:`torch.Generator` for the current :ref:`accelerator<accelerators>`
    on the specified device.

    Args:
        device (:class:`torch.device`, str, int, optional): The device to return the initial seed of.
            If not given, uses :func:`torch.accelerator.current_device_index` by default.

    Returns:
        int: the initial seed of the default generator for the specified device.

    .. warning::
        This function eagerly initializes the accelerator runtime.
    """
    device_index = _get_device_index(device, optional=True)
    default_generator = torch._C._accelerator_getDefaultGenerator(device_index)
    return default_generator.initial_seed()


def get_rng_state(device: _device_t = None, /) -> Tensor:
    r"""Return the RNG state of the default :class:`torch.Generator` for the current :ref:`accelerator<accelerators>`
    as a `torch.Tensor` of dtype `torch.uint8` for the specified accelerator device.

    Args:
        device (:class:`torch.device`, str, int, optional): The device to return the RNG state of.
            If not given, uses :func:`torch.accelerator.current_device_index` by default.

    Returns:
        torch.Tensor: the RNG state of the default generator for the specified device.

    .. warning::
        This function eagerly initializes the accelerator runtime.
    """
    device_index = _get_device_index(device, optional=True)
    default_generator = torch._C._accelerator_getDefaultGenerator(device_index)
    return default_generator.get_state()


def get_rng_state_all() -> list[Tensor]:
    r"""Return a list of `torch.Tensor` of dtype `torch.uint8` representing the RNG states of all devices for
    the current :ref:`accelerator<accelerators>`.

    Returns:
        list[torch.Tensor]: the RNG states of the default generators for all devices.

    .. warning::
        This function eagerly initializes the accelerator runtime.
    """
    return [get_rng_state(i) for i in range(torch.accelerator.device_count())]


__all__ = ["initial_seed", "get_rng_state", "get_rng_state_all"]
<|endoftext|>from collections import OrderedDict
from typing import Any

import torch

from ._utils import _device_t, _get_device_index


__all__ = [
    "empty_cache",
    "empty_host_cache",
    "get_memory_info",
    "max_memory_allocated",
    "max_memory_reserved",
    "memory_allocated",
    "memory_reserved",
    "memory_stats",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
]


def empty_cache() -> None:
    r"""Release all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other application.

    .. note:: This function is a no-op if the memory allocator for the current
        :ref:`accelerator <accelerators>` has not been initialized.
    """
    if not torch._C._accelerator_isAllocatorInitialized():
        return
    torch._C._accelerator_emptyCache()


def empty_host_cache() -> None:
    r"""Release all unoccupied cached host (pinned) memory currently held by the host caching
    allocator so that it can be used by other applications.

    .. note:: This function is a no-op if the memory allocator for the current
        :ref:`accelerator <accelerators>` has not been initialized.
    """
    torch._C._accelerator_emptyHostCache()


def _flatten_stats(result: list[tuple[str, Any]], prefix: str, value: Any) -> None:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            nested_prefix = f"{prefix}.{key}" if prefix else key
            _flatten_stats(result, nested_prefix, nested_value)
    else:
        result.append((prefix, value))


def memory_stats(device_index: _device_t = None, /) -> OrderedDict[str, Any]:
    r"""Return a dictionary of accelerator device memory allocator statistics for a given device index.

    The return value of this function is a dictionary of statistics, each of
    which is a non-negative integer.

    Core statistics:

    - ``"allocated.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of allocation requests received by the memory allocator.
    - ``"allocated_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of allocated memory.
    - ``"segment.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of reserved segments from device memory allocation.
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

    - ``"num_alloc_retries"``: number of failed device memory allocation calls that
      result in a cache flush and retry.
    - ``"num_ooms"``: number of out-of-memory errors thrown.
    - ``"num_sync_all_streams"``: number of ``synchronize_and_free_events`` calls.
    - ``"num_device_alloc"``: number of device memory allocation calls.
    - ``"num_device_free"``: number of device memory free calls.

    Args:
        device_index (:class:`torch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`torch.accelerator.current_device_index` by default.
            If a :class:`torch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    Returns:
        OrderedDict[str, Any]: an ordered dictionary mapping statistic names to their values.
    """
    if not torch._C._accelerator_isAllocatorInitialized():
        return OrderedDict()
    device_index = _get_device_index(device_index, optional=True)
    stats = torch._C._accelerator_getDeviceStats(device_index)
    flat_stats = []

    _flatten_stats(flat_stats, "", stats)
    flat_stats.sort()
    # pyrefly: ignore [no-matching-overload]
    return OrderedDict(flat_stats)


def memory_allocated(device_index: _device_t = None, /) -> int:
    r"""Return the current :ref:`accelerator<accelerators>` device memory occupied by tensors
    in bytes for a given device index.

    Args:
        device_index (:class:`torch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`torch.accelerator.current_device_index` by default.
            If a :class:`torch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    Returns:
        int: the current memory occupied by live tensors (in bytes) within the current process.
    """
    return memory_stats(device_index).get("allocated_bytes.all.current", 0)


def max_memory_allocated(device_index: _device_t = None, /) -> int:
    r"""Return the current :ref:`accelerator<accelerators>` maximum device memory occupied by tensors
    in bytes for a given device index.

    By default, this returns the peak allocated memory since the beginning of
    this program. :func:`~torch.accelerator.reset_peak_memory_stats` can be used to
    reset the starting point in tracking this metric.

    Args:
        device_index (:class:`torch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`torch.accelerator.current_device_index` by default.
            If a :class:`torch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    Returns:
        int: the peak memory occupied by live tensors (in bytes) within the current process.
    """
    return memory_stats(device_index).get("allocated_bytes.all.peak", 0)


def memory_reserved(device_index: _device_t = None, /) -> int:
    r"""Return the current :ref:`accelerator<accelerators>` device memory managed by the caching allocator
    in bytes for a given device index.

    Args:
        device_index (:class:`torch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`torch.accelerator.current_device_index` by default.
            If a :class:`torch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    Returns:
        int: the current memory reserved by PyTorch (in bytes) within the current process.
    """
    return memory_stats(device_index).get("reserved_bytes.all.current", 0)


def max_memory_reserved(device_index: _device_t = None, /) -> int:
    r"""Return the current :ref:`accelerator<accelerators>` maximum device memory managed by the caching allocator
    in bytes for a given device index.

    By default, this returns the peak cached memory since the beginning of this
    program. :func:`~torch.accelerator.reset_peak_memory_stats` can be used to reset
    the starting point in tracking this metric.

    Args:
        device_index (:class:`torch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`torch.accelerator.current_device_index` by default.
            If a :class:`torch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    Returns:
        int: the peak memory reserved by PyTorch (in bytes) within the current process.
    """
    return memory_stats(device_index).get("reserved_bytes.all.peak", 0)


def reset_accumulated_memory_stats(device_index: _device_t = None, /) -> None:
    r"""Reset the "accumulated" (historical) stats tracked by the current :ref:`accelerator<accelerators>`
    memory allocator for a given device index.

    Args:
        device_index (:class:`torch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`torch.accelerator.current_device_index` by default.
            If a :class:`torch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    .. note:: This function is a no-op if the memory allocator for the current
        :ref:`accelerator <accelerators>` has not been initialized.
    """
    if not torch._C._accelerator_isAllocatorInitialized():
        return
    device_index = _get_device_index(device_index, optional=True)
    return torch._C._accelerator_resetAccumulatedStats(device_index)


def reset_peak_memory_stats(device_index: _device_t = None, /) -> None:
    r"""Reset the "peak" stats tracked by the current :ref:`accelerator<accelerators>`
    memory allocator for a given device index.

    Args:
        device_index (:class:`torch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`torch.accelerator.current_device_index` by default.
            If a :class:`torch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    .. note:: This function is a no-op if the memory allocator for the current
        :ref:`accelerator <accelerators>` has not been initialized.
    """
    if not torch._C._accelerator_isAllocatorInitialized():
        return
    device_index = _get_device_index(device_index, optional=True)
    return torch._C._accelerator_resetPeakStats(device_index)


def get_memory_info(device_index: _device_t = None, /) -> tuple[int, int]:
    r"""Return the current device memory information for a given device index.

    Args:
        device_index (:class:`torch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`torch.accelerator.current_device_index` by default.
            If a :class:`torch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    Returns:
        tuple[int, int]: a tuple of two integers (free_memory, total_memory) in bytes.
            The first value is the free memory on the device (available across all processes and applications),
            The second value is the device's total hardware memory capacity.
    """
    device_index = _get_device_index(device_index, optional=True)
    # pyrefly: ignore [missing-attribute]
    return torch._C._accelerator_getMemoryInfo(device_index)


def _snapshot(device=None, augment_with_fx_traces: bool = False):
    r"""Return a snapshot of the current :ref:`accelerator<accelerators>` memory allocator state.

    Requires :func:`_record_memory_history` on the appropriate device module
    (e.g., :func:`torch.cuda.memory._record_memory_history`) to have been called.

    Args:
        device: the device to snapshot. If not given, uses the current device.
        augment_with_fx_traces (bool, optional): if True, augment stack traces
            with FX graph information. Default: ``False``.

    Returns:
        dict: a dictionary containing memory allocator state information.
    """
    acc = torch.accelerator.current_accelerator()
    if acc is not None and acc.type == "xpu":
        return torch.xpu.memory._snapshot(
            device, augment_with_fx_traces=augment_with_fx_traces
        )
    return torch.cuda.memory._snapshot(
        device, augment_with_fx_traces=augment_with_fx_traces
    )
<|endoftext|>r"""
This package introduces support for the current :ref:`accelerator<accelerators>` in python.
"""

from functools import cache
from typing import Any
from typing_extensions import deprecated

import torch

from . import random
from ._utils import _device_t, _get_device_index
from .graphs import Graph
from .memory import (
    empty_cache,
    empty_host_cache,
    get_memory_info,
    max_memory_allocated,
    max_memory_reserved,
    memory_allocated,
    memory_reserved,
    memory_stats,
    reset_accumulated_memory_stats,
    reset_peak_memory_stats,
)


__all__ = [
    "Graph",
    "current_accelerator",
    "current_device_idx",  # deprecated
    "current_device_index",
    "get_device_capability",
    "current_stream",
    "device_count",
    "device_index",
    "empty_cache",
    "empty_host_cache",
    "get_memory_info",
    "is_available",
    "max_memory_allocated",
    "max_memory_reserved",
    "memory_allocated",
    "memory_reserved",
    "memory_stats",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
    "set_device_idx",  # deprecated
    "set_device_index",
    "set_stream",
    "synchronize",
]


def device_count() -> int:
    r"""Return the number of current :ref:`accelerator<accelerators>` available.

    Returns:
        int: the number of the current :ref:`accelerator<accelerators>` available.
            If there is no available accelerators, return 0.

    .. note:: This API delegates to the device-specific version of `device_count`.
        On CUDA, this API will NOT poison fork if NVML discovery succeeds.
        Otherwise, it will. For more details, see :ref:`multiprocessing-poison-fork-note`.
    """
    acc = current_accelerator()
    if acc is None:
        return 0

    mod = torch.get_device_module(acc)
    return mod.device_count()


def is_available() -> bool:
    r"""Check if the current accelerator is available at runtime: it was built, all the
    required drivers are available and at least one device is visible.
    See :ref:`accelerator<accelerators>` for details.

    Returns:
        bool: A boolean indicating if there is an available :ref:`accelerator<accelerators>`.

    .. note:: This API delegates to the device-specific version of `is_available`.
        On CUDA, when the environment variable ``PYTORCH_NVML_BASED_CUDA_CHECK=1`` is set,
        this function will NOT poison fork. Otherwise, it will. For more details, see
        :ref:`multiprocessing-poison-fork-note`.

    Example::

        >>> assert torch.accelerator.is_available() "No available accelerators detected."
    """
    # Why not just check "device_count() > 0" like other is_available call?
    # Because device like CUDA have a python implementation of is_available that is
    # non-poisoning and some features like Dataloader rely on it.
    # So we are careful to delegate to the Python version of the accelerator here
    acc = current_accelerator()
    if acc is None:
        return False

    mod = torch.get_device_module(acc)
    return mod.is_available()


def current_accelerator(check_available: bool = False) -> torch.device | None:
    r"""Return the device of the accelerator available at compilation time.
    If no accelerator were available at compilation time, returns None.
    See :ref:`accelerator<accelerators>` for details.

    Args:
        check_available (bool, optional): if True, will also do a runtime check to see
            if the device :func:`torch.accelerator.is_available` on top of the compile-time
            check.
            Default: ``False``

    Returns:
        torch.device: return the current accelerator as :class:`torch.device`.

    .. note:: The index of the returned :class:`torch.device` will be ``None``, please use
        :func:`torch.accelerator.current_device_index` to know the current index being used.
        This API does NOT poison fork. For more details, see :ref:`multiprocessing-poison-fork-note`.

    Example::

        >>> # xdoctest:
        >>> # If an accelerator is available, sent the model to it
        >>> model = torch.nn.Linear(2, 2)
        >>> if (current_device := current_accelerator(check_available=True)) is not None:
        >>>     model.to(current_device)
    """
    if (acc := torch._C._accelerator_getAccelerator()) is not None:
        if (not check_available) or (check_available and is_available()):
            return acc
    return None


def current_device_index() -> int:
    r"""Return the index of a currently selected device for the current :ref:`accelerator<accelerators>`.

    Returns:
        int: the index of a currently selected device.
    """
    return torch._C._accelerator_getDeviceIndex()


current_device_idx = deprecated(
    "Use `current_device_index` instead.",
    category=FutureWarning,
)(current_device_index)

current_device_idx.__doc__ = r"""
    (Deprecated) Return the index of a currently selected device for the current :ref:`accelerator<accelerators>`.

    Returns:
        int: the index of a currently selected device.

    .. warning::

        :func:`torch.accelerator.current_device_idx` is deprecated in favor of :func:`torch.accelerator.current_device_index`
        and will be removed in a future PyTorch release.
    """


@cache
def get_device_capability(device: _device_t = None, /) -> dict[str, Any]:
    r"""Return the capability of the currently selected device.

    Args:
        device (:class:`torch.device`, str, int, optional): The device to query capabilities for
            :ref:`accelerator<accelerators>` device type. If not given,
            use :func:`torch.accelerator.current_device_index` by default.

    Returns:
        dict[str, Any]: A dictionary containing device capability information. The dictionary includes:
            - ``supported_dtypes`` (set(torch.dtype)): Set of PyTorch data types for which
              tensors can be allocated on the accelerator and type conversion across
              supported dtypes are supported. Any operator support outside of that
              is not guaranteed

    Examples:
        >>> # xdoctest: +SKIP("requires cuda")
        >>> # Query capabilities for current device
        >>> capabilities = torch.accelerator.get_device_capability("cuda:0")
        >>> print("Supported dtypes:", capabilities["supported_dtypes"])
    """
    device_index = _get_device_index(device, optional=True)
    # pyrefly: ignore [missing-attribute]
    return torch._C._accelerator_getDeviceCapability(device_index)


def set_device_index(device: _device_t, /) -> None:
    r"""Set the current device index to a given device.

    Args:
        device (:class:`torch.device`, str, int): a given device that must match the current
            :ref:`accelerator<accelerators>` device type.

    .. note:: This function is a no-op if this device index is negative.
    """
    device_index = _get_device_index(device, optional=False)
    torch._C._accelerator_setDeviceIndex(device_index)


set_device_idx = deprecated(
    "Use `set_device_index` instead.",
    category=FutureWarning,
)(set_device_index)

set_device_idx.__doc__ = r"""
    (Deprecated) Set the current device index to a given device.

    Args:
        device (:class:`torch.device`, str, int): a given device that must match the current
            :ref:`accelerator<accelerators>` device type.

    .. warning::

        :func:`torch.accelerator.set_device_idx` is deprecated in favor of :func:`torch.accelerator.set_device_index`
        and will be removed in a future PyTorch release.
    """


def current_stream(device: _device_t = None, /) -> torch.Stream:
    r"""Return the currently selected stream for a given device.

    Args:
        device (:class:`torch.device`, str, int, optional): a given device that must match the current
            :ref:`accelerator<accelerators>` device type. If not given,
            use :func:`torch.accelerator.current_device_index` by default.

    Returns:
        torch.Stream: the currently selected stream for a given device.
    """
    device_index = _get_device_index(device, optional=True)
    return torch._C._accelerator_getStream(device_index)


def set_stream(stream: torch.Stream) -> None:
    r"""Set the current stream to a given stream.

    Args:
        stream (torch.Stream): a given stream that must match the current :ref:`accelerator<accelerators>` device type.

    .. note:: This function will set the current device index to the device index of the given stream.
    """
    torch._C._accelerator_setStream(stream)


def synchronize(device: _device_t = None, /) -> None:
    r"""Wait for all kernels in all streams on the given device to complete.

    Args:
        device (:class:`torch.device`, str, int, optional): device for which to synchronize. It must match
            the current :ref:`accelerator<accelerators>` device type. If not given,
            use :func:`torch.accelerator.current_device_index` by default.

    .. note:: This function is a no-op if the current :ref:`accelerator<accelerators>` is not initialized.

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> assert torch.accelerator.is_available() "No available accelerators detected."
        >>> start_event = torch.Event(enable_timing=True)
        >>> end_event = torch.Event(enable_timing=True)
        >>> start_event.record()
        >>> tensor = torch.randn(100, device=torch.accelerator.current_accelerator())
        >>> sum = torch.sum(tensor)
        >>> end_event.record()
        >>> torch.accelerator.synchronize()
        >>> elapsed_time_ms = start_event.elapsed_time(end_event)
    """
    device_index = _get_device_index(device, optional=True)
    torch._C._accelerator_synchronizeDevice(device_index)


class device_index:
    r"""Context manager to set the current device index for the current :ref:`accelerator<accelerators>`.
    Temporarily changes the current device index to the specified value for the duration
    of the context, and automatically restores the previous device index when exiting
    the context.

    Args:
        device (Optional[int]): a given device index to temporarily set. If None,
            no device index switching occurs.

    Examples:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> # Set device 0 as the current device temporarily
        >>> with torch.accelerator.device_index(0):
        ...     # Code here runs with device 0 as the current device
        ...     pass
        >>> # Original device is now restored
        >>> # No-op when None is passed
        >>> with torch.accelerator.device_index(None):
        ...     # No device switching occurs
        ...     pass
    """

    def __init__(self, device: int | None, /) -> None:
        self.idx = device
        self.prev_idx = -1

    def __enter__(self) -> None:
        if self.idx is not None:
            self.prev_idx = torch._C._accelerator_exchangeDevice(self.idx)

    def __exit__(self, *exc_info: object) -> None:
        if self.idx is not None:
            torch._C._accelerator_maybeExchangeDevice(self.prev_idx)
<|endoftext|>from collections.abc import Callable

import torch
from torch.types import Device as _device_t


def _get_device_index(device: _device_t, optional: bool = False) -> int:
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        device = torch.device(device)
    device_index: int | None = None
    if isinstance(device, torch.device):
        acc = torch.accelerator.current_accelerator()
        if acc is None:
            raise RuntimeError("Accelerator expected")
        if acc.type != device.type:
            raise ValueError(
                f"{device.type} doesn't match the current accelerator {acc}."
            )
        device_index = device.index
    if device_index is None:
        if not optional:
            raise ValueError(
                f"Expected a torch.device with a specified index or an integer, but got:{device}"
            )
        return torch.accelerator.current_device_index()
    return device_index


def _lazy_call(callable: Callable[[], None], **kwargs) -> None:
    r"""Defer a callable until the :ref:`accelerator<accelerators>` runtime is initialized.

    If the runtime is already initialized or the backend does not support lazy
    initialization, the callable runs immediately. If no accelerator is
    available, the callable is silently dropped.

    Args:
        callable (Callable[[], None]): The function to be called.
        **kwargs: Additional keyword arguments forwarded to the backend's ``_lazy_call``
            (e.g., ``seed=True``, ``seed_all=True``).
    """
    acc = torch.accelerator.current_accelerator()
    if acc is None:
        return
    device_module = torch.get_device_module(acc)
    if hasattr(device_module, "_lazy_call"):
        device_module._lazy_call(callable, **kwargs)
    else:
        # Backend does not support lazy initialization; run immediately.
        callable()
<|endoftext|>import gc
from typing import Literal
from typing_extensions import Self

import torch
from torch._C import _acceleratorGraph


class Graph(_acceleratorGraph):
    r"""
    Wrapper around an :ref:`accelerator<accelerators>` graph that supports capture and replay.

    A graph captures a sequence of operations and their dependencies, allowing them to be
    replayed efficiently with reduced overhead. This class can be used as a context manager
    to automatically capture operations on the current stream.

    Arguments:
        keep_graph (bool, optional): If ``False``, the underlying graph is destroyed and the
            executable graph is instantiated on the GPU at the end of ``capture_end``.
            If ``True``, the underlying graph is preserved after ``capture_end``. In this case,
            the executable graph is not instantiated automatically; it must be explicitly created
            by calling ``instantiate``, or it will be instantiated on the first call to ``replay``.
            Defaults to ``False``.
        pool (tuple[int, int], optional): Memory pool identifier for this graph. Multiple graphs
            can share the same pool by passing the same identifier, which can reduce memory overhead.
            Defaults to ``None``.
        capture_error_mode (Literal["default", "global", "thread_local", "relaxed"], optional):
            Specifies the behavior of graph capture. The exact semantics are backend-specific.
            ``"default"``: backend-defined default capture behavior.
            ``"global"``: potentially unsafe API calls are prohibited. Errors may occur if capture
            in the current thread affects other threads.
            ``"thread_local"``: potentially unsafe API calls are prohibited. Errors occur only if
            capture in the current thread affects itself.
            ``"relaxed"``: the current thread is allowed to make potentially unsafe API calls, except
            for calls that inherently conflict with stream capture.
            Default: ``"default"``.

    Example::

        >>> # xdoctest: +SKIP
        >>> x = torch.zeros([2000], device=0)

        >>> stream = torch.Stream()
        >>> graph = torch.accelerator.Graph()
        >>> with stream, graph:
        ...     x += 1

        >>> graph.replay()
    """

    def __new__(
        cls,
        keep_graph: bool = False,
        *,
        pool: tuple[int, int] | None = None,
        capture_error_mode: Literal[
            "default", "global", "thread_local", "relaxed"
        ] = "default",
    ) -> Self:
        return super().__new__(cls, keep_graph)

    def __init__(
        self,
        keep_graph: bool = False,
        *,
        pool: tuple[int, int] | None = None,
        capture_error_mode: Literal[
            "default", "global", "thread_local", "relaxed"
        ] = "default",
    ) -> None:
        super().__init__(keep_graph)
        self.graph_pool = pool
        self.capture_error_mode = capture_error_mode

    # pyrefly: ignore [bad-override]
    def capture_begin(self) -> None:
        r"""
        Begin graph capture on the current stream.

        All operations on the current stream after this call will be recorded into the graph until
        ``capture_end`` is called, using the memory pool and capture error mode provided at construction time.
        """
        super().capture_begin(
            pool=self.graph_pool, capture_error_mode=self.capture_error_mode
        )

    def capture_end(self) -> None:
        r"""
        End graph capture on the current stream of the current device.

        After this call, the graph can be replayed via ``replay``.
        """
        super().capture_end()

    def instantiate(self) -> None:
        r"""
        Instantiate the underlying graph. Will be called by ``capture_end``
        if ``keep_graph=False``, or by ``replay`` if ``keep_graph=True`` and
        ``instantiate`` has not already been explicitly called.
        """
        super().instantiate()

    def replay(self) -> None:
        r"""Replay the work captured by this graph."""
        super().replay()

    def reset(self) -> None:
        r"""
        Delete the graph currently held by this instance.

        After this call, the graph can be recaptured. Set :attr:`graph_pool` or
        :attr:`capture_error_mode` beforehand to use different settings on the next capture.
        """
        super().reset()

    def pool(self) -> tuple[int, int]:
        r"""
