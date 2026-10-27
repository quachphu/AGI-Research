        torch.compiler._is_compiling_flag = old_compiling_flag
        torch.compiler._is_exporting_flag = old_exporting_flag


def _fakify_params_buffers(
    fake_mode: FakeTensorMode,
    mod: torch.nn.Module,
) -> dict[str, torch.Tensor | torch.nn.Parameter]:
    params_buffers = {
        **dict(mod.named_parameters(remove_duplicate=False)),
        **dict(mod.named_buffers(remove_duplicate=False)),
    }

    faked_params_buffers = {}
    memo: dict[int, FakeTensor] = {}
    for key, value in params_buffers.items():
        if id(value) in memo:
            fake_tensor = memo[id(value)]
        else:
            fake_tensor = fake_mode.from_tensor(value, static_shapes=True)
            memo[id(value)] = fake_tensor
        faked_params_buffers[key] = fake_tensor
    return faked_params_buffers  # type: ignore[return-value]


def register_module_as_pytree_input_node(cls: type[torch.nn.Module]) -> None:
    """
    Registers a module as a valid input type for :func:`torch.export.export`.

    Args:
        mod: the module instance
        serialized_type_name: The serialized name for the module. This is
        required if you want to serialize the pytree TreeSpec containing this
        module.

    Example::

        import torch


        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return self.linear(x)


        torch._export.utils.register_module_as_pytree_node(InputDataClass)


        class Mod(torch.nn.Module):
            def forward(self, x, m):
                return m(x) + x


        ep = torch.export.export(Mod(), (torch.randn(3), Module()))
        print(ep)

    """
    if not issubclass(cls, torch.nn.Module):
        raise AssertionError(f"expected nn.Module subclass, got {cls}")

    import weakref

    class PrototypeModule(weakref.ref[torch.nn.Module]):
        def __init__(self, m: torch.nn.Module, *args: Any, **kwargs: Any) -> None:
            super().__init__(m, *args, **kwargs)  # type: ignore[call-arg]
            if not isinstance(m, torch.nn.Module):
                raise AssertionError(f"expected nn.Module, got {type(m).__name__}")
            if hasattr(self, "_proto_cls"):
                raise AssertionError("_proto_cls should not be set")
            self._proto_cls = cls

        def __eq__(self, other: object) -> bool:
            return self._proto_cls == other._proto_cls  # type: ignore[attr-defined]

        def __deepcopy__(self, memo: dict[Any, Any]) -> "PrototypeModule":
            # pyrefly: ignore[bad-argument-type]  # TODO referent is alive during deepcopy
            return PrototypeModule(self())

    def default_flatten_fn(obj: Any) -> tuple[list[Any], Context]:
        named_parameters = dict(obj.named_parameters())
        named_buffers = dict(obj.named_buffers())
        params_buffers = {**named_parameters, **named_buffers}
        return list(params_buffers.values()), [
            list(params_buffers.keys()),
            PrototypeModule(obj),
        ]

    def default_unflatten_fn(values: Iterable[Any], context: Context) -> Any:
        flat_names, ref = context
        if ref is None or ref() is None:
            raise RuntimeError("Module has been garbage collected")
        obj = ref()
        if flatten_fn is None:
            raise AssertionError("flatten_fn should not be None")
        flattened, _ = flatten_fn(obj)

        # NOTE: This helper function will replicate an nn.Module in the exactly same
        #       structure to be used together with _reparameterize_module. This will
        #       create a clone of the module with the new parameters and buffers without
        #       affecting the original module.
        def copy_module(mod: torch.nn.Module) -> torch.nn.Module:
            ret = copy.copy(mod)
            ret.__dict__ = {copy.copy(k): copy.copy(v) for k, v in mod.__dict__.items()}
            for name, child in ret.named_children():
                setattr(ret, name, copy_module(child))
            return ret

        if any(v is not o for v, o in zip(values, flattened)):
            with torch.nn.utils.stateless._reparametrize_module(
                obj, dict(zip(flat_names, values)), tie_weights=True, strict=True
            ):
                ret = copy_module(obj)
        else:
            ret = obj
        return ret

    def default_flatten_fn_with_keys(obj: Any) -> tuple[list[Any], Context]:
        flattened, [flat_names, *args] = flatten_fn(obj)  # type: ignore[misc]
        return [(MappingKey(k), v) for k, v in zip(flat_names, flattened)], [
            flat_names,
            *args,
        ]

    flatten_fn = default_flatten_fn
    unflatten_fn = default_unflatten_fn

    serialized_type_name = cls.__module__ + "." + cls.__qualname__

    def to_dumpable_context(context: Context) -> str:
        keys, *_ = context
        return json.dumps([keys, *([None] * len(_))])

    def from_dumpable_context(dumpable: str) -> Any:
        s = json.loads(dumpable)
        s[1] = PrototypeModule(torch.nn.Module())
        return s

    _register_pytree_node(
        cls,
        flatten_fn,
        unflatten_fn,
        serialized_type_name=serialized_type_name,
        flatten_with_keys_fn=default_flatten_fn_with_keys,
        to_dumpable_context=to_dumpable_context,
        from_dumpable_context=from_dumpable_context,
    )

    def default_flatten_fn_spec(obj: Any, spec: "TreeSpec") -> list[Any]:
        flats, context = flatten_fn(obj)
        if context != spec.context:
            raise AssertionError(f"context mismatch: {context} != {spec.context}")
        return flats

    register_pytree_flatten_spec(
        cls,
        default_flatten_fn_spec,
    )


def deregister_module_as_pytree_input_node(cls: type[torch.nn.Module]) -> None:
    _deregister_pytree_node(cls)
    _deregister_pytree_flatten_spec(cls)


def _sync_state(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    if not isinstance(src, torch.nn.Module):
        raise AssertionError(f"Expected {src} to be a nn.Module")
    if not isinstance(dst, torch.nn.Module):
        raise AssertionError(f"Expected {dst} to be a nn.Module")
    # Share state (params, buffers) between modules.
    # This ensures that state mutations are visible across them.
    # Since tensor constants are not mutable, copying (without sharing) is OK.
    # Also, primitive constants are specialized, so copying (without sharing) is OK.
    dst._parameters = src._parameters
    dst._buffers = src._buffers


def sync_state(*wrapped_method_modules: torch.nn.Module) -> None:
    """
    Sync state between exported modules corresponding to wrapped methods.
    This might be necessary after serializing/deserializing due to copying.
    """
    if wrapped_method_modules:
        m, *other_ms = wrapped_method_modules
        for other_m in other_ms:
            _sync_state(m, other_m)


class _WrappedMethod(torch.nn.Module):
    def __init__(self, method: Callable[..., object]) -> None:
        super().__init__()
        # share state of method's self module
        _sync_state(method.__self__, self)  # type: ignore[attr-defined]
        # redirect forward to method
        self.forward = method


def wrap_method(method: Callable[..., object]) -> _WrappedMethod:
    """
    Wrap a method as a module so that it can be exported.
    The wrapped module's forward points to the method, and
    the method's original module state is shared.
    """
    if not ismethod(method):
        raise AssertionError(f"Expected {method} to be a method")
    return _WrappedMethod(method)
<|endoftext|>"""
Configuration module for torch.export.export.

This module contains various configuration flags and settings that control torch.export's
behavior, including:
- Runtime behavior flags
- Debugging and development options
"""

import sys
from typing import Any, TYPE_CHECKING

from torch._environment import is_fbcode
from torch.utils._config_module import install_config_module


# this flag controls whether we use new functional tracer. It
# should be True in the long term.
use_new_tracer_experimental = True

# this flag is used to control whether we want to instrument
# fake tensor creation to track potential leaks. It is off
# by default, but user can turn it on to debug leaks.
detect_non_strict_fake_tensor_leaks = False

# error on potentially pre-dispatch/non-strict tracing limitation
# this type of error usually happens when we encounter an op
# that we don't know how to proxy, resulting in untracked fake tensors
error_on_lifted_constant_tensors = True

# enable auto_functionalized_v2 in export
# We turn this off in fbcode due to downstream users not
# being ready to handle auto_functionalized_v2.
enable_auto_functionalized_v2_for_export = not is_fbcode()

use_legacy_dynamo_graph_capture = True


if TYPE_CHECKING:
    from torch.utils._config_typing import *  # noqa: F403

    def _make_closure_patcher(**changes: Any) -> Any: ...


install_config_module(sys.modules[__name__])
<|endoftext|>import builtins
import contextlib
import functools
import inspect
import logging
import math
import sys
from collections import defaultdict
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from typing import Any, cast, TYPE_CHECKING, TypeGuard, TypeVar

import torch
import torch.utils._pytree as pytree
from torch._custom_class_base import CustomClassBase
from torch._dynamo.source import (
    AttrSource,
    GetItemSource,
    LocalSource,
    TensorProperty,
    TensorPropertySource,
)
from torch._dynamo.variables.builder import TrackedFake
from torch._export.passes.lift_constants_pass import ConstantAttrMap
from torch._export.utils import _fakify_params_buffers
from torch._guards import Source
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import is_custom_class_obj
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.export import Constraint
from torch.export.dynamic_shapes import (
    _check_dynamic_shapes,
    _combine_args,
    _DimHint,
    _DimHintType,
    _IntWrapper,
    _process_dynamic_shapes,
    _RelaxedConstraint,
    _tree_map_with_path,
)
from torch.export.graph_signature import CustomObjArgument
from torch.fx.experimental import _config as config
from torch.fx.experimental.symbolic_shapes import (
    _find_user_code_frame,
    _suggest_fixes_for_data_dependent_error_non_strict,
    ConstraintViolationError,
    DimDynamic,
    EqualityConstraint,
    GuardOnDataDependentSymNode,
    RelaxedUnspecConstraint,
    ShapeEnv,
    StatelessSymbolicContext,
    SymIntSymbolicContext,
    ValueRanges,
)
from torch.utils._pytree import (
    GetAttrKey,
    KeyPath,
    MappingKey,
    SequenceKey,
    tree_map_with_path,
)
from torch.utils._sympy.numbers import int_oo


if TYPE_CHECKING:
    from sympy import Expr, Symbol

    from torch.fx.experimental.dynamic_spec import ParamsSpec, ShapesSpec
    from torch.fx.experimental.symbolic_shapes import SymbolicContext


log = logging.getLogger(__name__)


_ModuleT = TypeVar("_ModuleT", bound=torch.nn.Module)


_DATA_PTR_ACCESS_ERROR = "Cannot access data pointer of Tensor"
_RAW_TRITON_KERNEL_NON_STRICT_EXPORT_ERROR = (
    "Raw Triton kernel calls are not supported by non-strict torch.export. "
    "Wrap the kernel in a torch.library.triton_op and call it through "
    "torch.library.wrap_triton (or torch._library.capture_triton) so export "
    "records a stable custom operator instead of tracing into Triton's runtime."
)


def _is_raw_triton_kernel_data_ptr_error(
    func: Callable[..., object], exc: RuntimeError
) -> bool:
    if getattr(func, "__name__", None) != "data_ptr":
        return False
    if _DATA_PTR_ACCESS_ERROR not in str(exc):
        return False

    for frame in inspect.stack(context=0):
        filename = frame.filename.replace("\\", "/")
        if "/triton/runtime/" in filename:
            return True
    return False


class _KeyPath:
    """
    Wraps `KeyPath` to aid `isinstance` checks.
    """

    def __init__(self, kp: KeyPath) -> None:
        self.kp = kp


class _KeyPathTrie:
    """
    Builds a trie of `KeyPath` prefixes mapping to `Source` leaves.
    """

    def __init__(self) -> None:
        self.root: dict[Any, Any] = {}

    def add(self, kp: KeyPath, src: Source) -> None:
        if len(kp) == 0:
            raise AssertionError("KeyPath must not be empty")
        *path, leaf = kp
        node = self.root
        for k in path:
            if k not in node:
                node[k] = {}
            node = node[k]
        node[leaf] = src

    def get(self, kp: KeyPath) -> tuple[Source, KeyPath]:
        node = self.root
        # pyrefly: ignore [bad-assignment]
        while not isinstance(node, Source):
            if len(kp) == 0:
                raise AssertionError("KeyPath exhausted before reaching Source")
            k, *kp = kp  # type: ignore[assignment]
            node = node[k]
        # pyrefly: ignore [bad-return]
        return node, kp


def make_sourced_prefixes(
    nn_module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> _KeyPathTrie:
    kp_args, kp_kwargs = tree_map_with_path(
        lambda kp, _: _KeyPath(kp),
        (tuple(None for _ in args), {k: None for k in kwargs}),  # noqa: C420
    )
    kp_combined_args = _combine_args(nn_module, kp_args, kp_kwargs)

    sourced_prefixes = _KeyPathTrie()
    for name, struct in kp_combined_args.items():
        src = LocalSource(name)

        if isinstance(struct, _KeyPath):
            sourced_prefixes.add(struct.kp, src)
        elif isinstance(struct, tuple):
            for i, prefix in enumerate(struct):
                if not isinstance(prefix, _KeyPath):
                    raise AssertionError(f"expected _KeyPath, got {type(prefix)}")
                sourced_prefixes.add(prefix.kp, GetItemSource(src, i))
        elif isinstance(struct, dict):
            for k, prefix in struct.items():
                if not isinstance(prefix, _KeyPath):
                    raise AssertionError(f"expected _KeyPath, got {type(prefix)}")
                sourced_prefixes.add(prefix.kp, GetItemSource(src, k))

    return sourced_prefixes


def key_path_to_source(
    kp: KeyPath, sourced_prefixes: _KeyPathTrie | None = None
) -> Source:
    """
    Given a key path, return the source for the key path.
    """
    if sourced_prefixes is None:
        source: Source = LocalSource("args")
    else:
        source, kp = sourced_prefixes.get(kp)

    for k in kp:
        if isinstance(k, SequenceKey):
            source = GetItemSource(source, k.idx)
        elif isinstance(k, MappingKey):
            source = GetItemSource(source, k.key)
        elif isinstance(k, GetAttrKey):
            source = AttrSource(source, k.name)
        else:
            raise ValueError(f"Unknown KeyEntry {k}")

    return source


def _is_constant_argument(t: object) -> bool:
    return t is None or isinstance(t, (float, bool, str))


def fakify(
    mode: FakeTensorMode,
    kp: KeyPath,
    t: Any,
    t_constraints: dict[int, dict[int, Constraint]],
    sources: dict[tuple[int, int], list[Source]],
    sourced_prefixes: _KeyPathTrie | None = None,
) -> Any:
    source = key_path_to_source(kp, sourced_prefixes=sourced_prefixes)
    if (
        _is_constant_argument(t)
        or isinstance(t, (torch.ScriptObject, torch.nn.Module))
        or is_custom_class_obj(t)
    ):
        return t

    if isinstance(t, _IntWrapper):
        if t.dynamism is not None and t.dynamism.type in (  # type: ignore[union-attr]
            _DimHintType.DYNAMIC,
            _DimHintType.AUTO,
        ):
            symint = mode.shape_env.create_unspecified_symint_and_symbol(  # type: ignore[union-attr]
                t.val, source, DimDynamic.DYNAMIC
            )
            context = (
                SymIntSymbolicContext(
                    constraint=RelaxedUnspecConstraint(warn_only=False)
                )
                if t.dynamism.type == _DimHintType.DYNAMIC  # type: ignore[union-attr]
                else None
            )
            mode.shape_env.tracked_fakes.append(  # type: ignore[union-attr]
                TrackedFake(symint, source, context)
            )
            return symint
        else:
            return t.val

    if not isinstance(t, torch.Tensor):
        raise ValueError(
            f"Unsupported input type {type(t)}. "
            "Export only supports pytree containers of basic types (Tensor, int, float, ...) as input. "
            "To register a custom dataclass, use torch.export.register_dataclass. "
            "To register a custom container type, use torch.utils._pytree.register_pytree_node. "
            "To register a constant input, use torch.utils._pytree.register_constant"
        )

    # Create symbolic context (handles subclass recursion internally)
    symbolic_context = _create_symbolic_context_for_tensor(
        t, source, t_constraints, sources, mode
    )

    fake = mode.from_tensor(t, source=source, symbolic_context=symbolic_context)
    mode.shape_env.tracked_fakes.append(TrackedFake(fake, source, symbolic_context))  # type: ignore[union-attr]
    return fake


def _create_symbolic_context_for_tensor(
    t: torch.Tensor,
    source: Source,
    t_constraints: dict[int, dict[int, Constraint]],
    sources: dict[tuple[int, int], list[Source]],
    mode: FakeTensorMode,
) -> "SymbolicContext":
    """Helper function to create symbolic context for a tensor."""
    from torch._dynamo.source import AttrSource
    from torch.fx.experimental.symbolic_shapes import (
        DimDynamic,
        RelaxedUnspecConstraint,
        SubclassSymbolicContext,
    )
    from torch.utils._python_dispatch import is_traceable_wrapper_subclass

    # Common dynamic dimension logic for both regular tensors and subclasses
    n_dims = len(t.shape)
    dynamic_sizes = []
    constraint_sizes = [None] * n_dims

    for i in range(n_dims):
        if i in getattr(t, "_dynamo_weak_dynamic_indices", {}):
            dynamic_sizes.append(DimDynamic.DYNAMIC)
        elif i in getattr(t, "_dynamo_dynamic_indices", {}):
            # bit annoying, but we need to replicate process in _dynamo/variables/builder.py
            # where a RelaxedUnspecConstraint is created for Dim.DYNAMIC, so constraint violations
            # are raised when specializing.
            dynamic_sizes.append(DimDynamic.DYNAMIC)
            constraint_sizes[i] = RelaxedUnspecConstraint(warn_only=False)  # type: ignore[call-overload]
        else:
            dynamic_sizes.append(DimDynamic.STATIC)

    # Handle nested subclasses
    if is_traceable_wrapper_subclass(t):
        # Get inner contexts recursively
        inner_contexts: dict[str, SymbolicContext] = {}
        # pyrefly: ignore[missing-attribute]  # TODO traceable wrapper subclasses define __tensor_flatten__
        attrs, _ = type(t).__tensor_flatten__(t)

        # Propagate outer tensor constraints to inner tensors if not already present
        for attr in attrs:
            match getattr(t, attr):
                case torch.Tensor() as inner_value:
                    inner_source = AttrSource(source, attr)
                    inner_contexts[attr] = _create_symbolic_context_for_tensor(
                        inner_value, inner_source, t_constraints, sources, mode
                    )
                case CustomClassBase():
                    pass
                case unexpected:
                    raise AssertionError(
                        f"expected Tensor or CustomClassBase, got {type(unexpected)}"
                    )

        symbolic_context = SubclassSymbolicContext(
            dynamic_sizes=dynamic_sizes,
            constraint_sizes=constraint_sizes,  # type: ignore[arg-type]
            view_base_context=None,
            tensor_source=source,
            shape_env_to_source_to_symbol_cache={},
            inner_contexts=inner_contexts,
        )
    else:
        symbolic_context: StatelessSymbolicContext = (  # type: ignore[no-redef]
            StatelessSymbolicContext(
                dynamic_sizes=dynamic_sizes,
                constraint_sizes=constraint_sizes,  # type: ignore[arg-type]
            )
        )

    # Apply constraints (common logic)
    t_id = id(t)
    if mode.shape_env is None:
        raise AssertionError("mode.shape_env must not be None")
    if t_id in t_constraints:
        for i, constraint in t_constraints[t_id].items():
            src = TensorPropertySource(base=source, prop=TensorProperty.SIZE, idx=i)
            sources[(t_id, i)].append(src)
            if isinstance(constraint, _RelaxedConstraint):
                continue
            symbolic_context.constraint_sizes[i] = constraint.constraint_range
            mode.shape_env.source_name_to_debug_name[src.name] = constraint.name  # type: ignore[assignment]

    return symbolic_context


def _is_unbacked_symint(symbol: object) -> bool:
    if not isinstance(symbol, torch.SymInt):
        return False

    if symbol.node.shape_env is None:
        raise AssertionError("shape_env should not be None")
    return symbol.node.shape_env.is_unbacked_symint(symbol.node.expr)


def _tensor_min_max(
    *args: Any,
    real_callable: Callable[..., object],
    tensor_callable: Callable[..., object],
    **kwargs: Any,
) -> Any:
    """
    This logic is replicated from dynamo/variables/builtin.py
    """
    if len(args) == 2 and not kwargs:
        arg1, arg2 = args

        # Case 1: Both are tensors
        if isinstance(arg1, torch.Tensor) and isinstance(arg2, torch.Tensor):
            return tensor_callable(arg1, arg2)

        # Case 2: One tensor, one scalar
        elif isinstance(arg1, torch.Tensor) or isinstance(arg2, torch.Tensor):
            if not isinstance(arg1, torch.Tensor):
                arg1, arg2 = arg2, arg1

            if isinstance(arg2, (int, float)):
                kwarg = {"min" if tensor_callable is torch.maximum else "max": arg2}
                return torch.clamp(arg1, **kwarg)  # type: ignore[call-overload]
            else:
                return real_callable(arg1, arg2)

        # Case 3: SymInts
        elif isinstance(arg1, torch.SymInt) or isinstance(arg2, torch.SymInt):
            return (
                torch.sym_max(arg1, arg2)
                if tensor_callable is torch.maximum
                else torch.sym_min(arg1, arg2)
            )

        # Fallback
        else:
            return real_callable(arg1, arg2)

    # Single iterable argument handling
    if len(args) == 1 and not kwargs:
        iterable = args[0]

        if isinstance(iterable, torch.Tensor):
            return tensor_callable(iterable)
        try:
            iterator = iter(iterable)
        except TypeError:
            pass
        else:
            items = list(iterator)
            if not items:
                raise ValueError(f"{real_callable.__name__}() arg is an empty sequence")

            return functools.reduce(
                lambda a, b: _tensor_min_max(
                    a, b, real_callable=real_callable, tensor_callable=tensor_callable
                ),
                items,
            )

    # Fallback to original callable
    return real_callable(*args, **kwargs)


@contextmanager
def _override_builtin_ops() -> Generator[None, None, None]:
    original_max = builtins.max
    original_min = builtins.min
    original_pow = math.pow

    # pyrefly: ignore [bad-assignment]
    builtins.max = functools.partial(
        _tensor_min_max, real_callable=original_max, tensor_callable=torch.maximum
    )

    # pyrefly: ignore [bad-assignment]
    builtins.min = functools.partial(
        _tensor_min_max, real_callable=original_min, tensor_callable=torch.minimum
    )

    math.pow = lambda x, y: x**y  # type: ignore[operator]

    try:
        yield
    finally:
        builtins.max = original_max
        builtins.min = original_min
        math.pow = original_pow


def _make_fake_inputs_with_spec(
    fake_mode: FakeTensorMode,
    nn_module: Any,
    args: Any,
    kwargs: Any,
    sourced_prefixes: Any,
    dynamic_shapes: Any,
    original_signature: inspect.Signature,
) -> tuple[FakeTensorMode, Any, Any, EqualityConstraint, inspect.Signature, Any]:
    from torch.fx.experimental._spec_binding import _bind_spec_to_args
    from torch.fx.experimental.dynamic_spec import (
        _coerce_to_shapes_spec,
        IntVar,
        ShapesSpec,
    )
    from torch.fx.experimental.symbolic_shapes import (
        _finalize_spec_wiring,
        _symbolic_context_from_shapes_spec,
        _wire_spec_assumptions,
        _wire_spec_slot,
        _wire_tensor_spec_dims,
    )

    user_spec = cast(ShapesSpec, _coerce_to_shapes_spec(dynamic_shapes))
    shape_env = fake_mode.shape_env
    if shape_env is None:
        raise AssertionError("fake_mode.shape_env must not be None")

    def _fakify_one_leaf(x: Any, source: Source, leaf_spec: Any) -> Any:
        """Fakify a single flat input leaf against its (already-bound) spec."""
        if isinstance(x, torch.Tensor):
            if leaf_spec is None:
                return fake_mode.from_tensor(x, static_shapes=True, source=source)
            ctx = _symbolic_context_from_shapes_spec(x, source, leaf_spec, None, {})
            fake_x = fake_mode.from_tensor(
                x, static_shapes=False, source=source, symbolic_context=ctx
            )
            _wire_tensor_spec_dims(leaf_spec, fake_x)
            return fake_x
        # NB: don't match on bools.
        if type(x) is int:
            if isinstance(leaf_spec, int):
                # Static int spec: verify the runtime value matches.
                if x != leaf_spec:
                    raise ValueError(
                        f"shapes_spec declared {source.name} as static with "
                        f"value {leaf_spec}, but got {x} at trace time"
                    )
                return x
            if isinstance(leaf_spec, (IntVar, torch.SymInt)):
                sym_node = shape_env.create_unbacked_symint(source=source)
                _wire_spec_slot(leaf_spec, sym_node)
                return sym_node
        return x

    # Wire assumptions BEFORE processing inputs so derived / assumption
    # checks can drain as inputs bind.
    if user_spec._assumptions:
        _wire_spec_assumptions(shape_env, user_spec)

    leaf_specs, flat_args, in_spec = _bind_spec_to_args(
        nn_module, args, kwargs, user_spec
    )
    leaf_sources = [
        key_path_to_source(kp, sourced_prefixes=sourced_prefixes)
        for kp, _ in pytree.tree_flatten_with_path((args, kwargs))[0]
    ]

    with shape_env.ignore_fresh_unbacked_symbols():
        fake_leaves = [
            _fakify_one_leaf(x, leaf_sources[i], leaf_specs[i])
            for i, x in enumerate(flat_args)
        ]

    # Verify every spec assumption / derived check that survived input
    # processing has been emitted.
    _finalize_spec_wiring(shape_env)

    fake_args, fake_kwargs = pytree.tree_unflatten(fake_leaves, in_spec)
    equalities_inputs = EqualityConstraint(
        source_pairs=[],
        derived_equalities=[],
        phantom_symbols=[],
        relaxed_sources=set(),
        warn_only=False,
    )
    return (
        fake_mode,
        fake_args,
        fake_kwargs,
        equalities_inputs,
        original_signature,
        dynamic_shapes,
    )


def make_fake_inputs(
    nn_module: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    dynamic_shapes: "dict[str, Any] | tuple[Any, ...] | list[Any] | ShapesSpec | ParamsSpec | None",
    prefer_deferred_runtime_asserts_over_guards: bool = False,
) -> tuple[
    FakeTensorMode,
    tuple[Any, ...],
    dict[str, Any],
    EqualityConstraint,
    inspect.Signature,
    dict[str, Any] | tuple[Any, ...] | list[Any] | None,
]:
    """
    Given an nn module, example inputs, and constraints, return a new fake mode,
    fake inputs created in that mode whose dynamic shape dimensions are constrained
    by the given ranges, and sources for pairs of dynamic shape dimensions that are
    constrained to be equal.
    """
    # TODO(avik): refactor Dynamo to avoid duplication of the following code
    # between non-strict and strict.
    # Specifically, here (non-strict) we do the following pre-tracing steps:
    #   - Fakify inputs.
    #   - Process input shape equalities.
    # In strict, these steps are spread across multiple files:
    #   - output_graph.py fakifies inputs.
    #   - [post-tracing] guards.py processes input shape equalities.
    import torch._functorch.config as _config
    from torch.fx.experimental.dynamic_spec import ParamsSpec, ShapesSpec

    # ShapesSpec/ParamsSpec drive dynamism via unbacked symbols (handled by the
    # spec fakify branch below); skip the legacy constraint machinery for it.
    is_shapes_spec = isinstance(dynamic_shapes, (ShapesSpec, ParamsSpec))

    constraints: list[Constraint] = []
    t_constraints: dict[int, dict[int, Constraint]] = defaultdict(dict)
    if not is_shapes_spec:
        # Map ints to a wrapper structure to help us mark it as dynamic, if it is
        # dynamic. We will unwrap ints in fakify later.
        args, kwargs = pytree.tree_map_only(
            int, lambda a: _IntWrapper(a), (args, kwargs)
        )

        combined_args = _combine_args(nn_module, args, kwargs)
        _check_dynamic_shapes(combined_args, dynamic_shapes)
        constraints = _process_dynamic_shapes(combined_args, dynamic_shapes)
        for constraint in constraints:
            t_constraints[constraint.t_id][constraint.dim] = constraint

    context = torch._guards.TracingContext.try_get()
    if context is not None:
        # This occurs when we are exporting within dynamo. There already exists
        # a toplevel TracingContext with a fake mode, so we do not want to
        # create another fake mode.
        fake_mode = context.fake_mode
        if fake_mode is None:
            raise AssertionError("context.fake_mode must not be None")
    else:
        if isinstance(nn_module.forward, functools.partial):
            # functools handles nesting by itself, no need to recurse
            code = nn_module.forward.func.__code__
        elif (
            sys.version_info >= (3, 14)
            and (fwd := getattr(nn_module.forward, "__func__", None))
            and isinstance(fwd, functools.partial)
        ):
            # functools.partial is now a method descriptor:
            # https://docs.python.org/3/whatsnew/3.14.html#changes-in-the-python-api
            code = fwd.func.__code__
        else:
            code = nn_module.forward.__code__
        co_fields = {
            "co_name": code.co_name,
            "co_filename": code.co_filename,
            "co_firstlineno": code.co_firstlineno,
        }
        with _config.patch(fake_tensor_allow_unsafe_data_ptr_access=False):
            fake_mode = FakeTensorMode(
                shape_env=ShapeEnv(
                    tracked_fakes=[],
                    co_fields=co_fields,
                    prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
                    trace_asserts=True,
                ),
                allow_non_fake_inputs=True,
                export=True,
            )
    if fake_mode.shape_env is None or fake_mode.shape_env.tracked_fakes is None:
        raise ValueError(
            "Detected fake_mode does not have a shape_env with tracked fakes. "
            "If you constructed the module under a FakeTensorMode, "
            "please initialize it like: FakeTensorMode(shape_env=ShapeEnv(tracked_fakes=[]))"
        )

    with fake_mode:
        original_signature = inspect.signature(nn_module.forward)
        sources: dict[tuple[int, int], list[Source]] = defaultdict(list)
        sourced_prefixes = make_sourced_prefixes(nn_module, args, kwargs)
        if is_shapes_spec:
            return _make_fake_inputs_with_spec(
                fake_mode,
                nn_module,
                args,
                kwargs,
                sourced_prefixes,
                dynamic_shapes,
                original_signature,
            )
        fake_args, fake_kwargs = tree_map_with_path(
            lambda kp, val: fakify(
                fake_mode,
                kp,
                val,
                t_constraints,
                sources,
                sourced_prefixes=sourced_prefixes,
            ),
            (args, kwargs),
        )

        names: dict[str, tuple[int, int]] = {}
        source_pairs: list[tuple[Source, Source]] = []
        derived_equalities: list[
            tuple[Source, Source | Symbol, Callable[..., object]]
        ] = []
        phantom_symbols: dict[str, Symbol] = {}
        relaxed_sources: set[Source] = set()
        for constraint in constraints:
            torch.export.dynamic_shapes._process_equalities(
                constraint,
                lambda t_id, dim: sources[(t_id, dim)],
                fake_mode.shape_env,
                names,
                source_pairs,
                derived_equalities,
                phantom_symbols,
                relaxed_sources,
            )

        equalities_inputs = EqualityConstraint(
            source_pairs=source_pairs,
            derived_equalities=derived_equalities,
            phantom_symbols=list(phantom_symbols.values()),
            relaxed_sources=relaxed_sources,
            warn_only=False,
        )
        return (
            fake_mode,
            fake_args,
            fake_kwargs,
            equalities_inputs,
            original_signature,
            dynamic_shapes,
        )


def _flatten_dynamic_shapes(
    combined_args: dict[str, Any],
    dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any],
) -> list[Any]:
    flat_shapes: list[Any] = []

    def _tree_map_helper(path: KeyPath, t: Any, shape: Any) -> None:
        nonlocal flat_shapes
        flat_shapes.append(shape)

    _tree_map_with_path(_tree_map_helper, combined_args, dynamic_shapes)
    return flat_shapes


def _clean_dynamic_markers(tensor: torch.Tensor) -> None:
    for attr in (
        "_dynamo_weak_dynamic_indices",
        "_dynamo_dynamic_indices",
        "_dynamo_dynamic_range",
        "_dynamo_static_indices",
        "_dynamo_unbacked_indices",
        "_dynamo_propagated_dynamic_indices",
        "_has_dynamo_dim_marking",
    ):
        if hasattr(tensor, attr):
            delattr(tensor, attr)


def produce_guards_and_solve_constraints(
    fake_mode: FakeTensorMode,
    gm: torch.fx.GraphModule,
    dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None,
    equalities_inputs: EqualityConstraint,
    original_signature: inspect.Signature,
) -> None:
    """
    Given a fake mode, sources pairs corresponding to equal dynamic shape dimensions,
    and a graph module, produce guards on the fake mode's shape env (raising constraint
    violations if any), solve (to suggest simplifications or fixes).
    Dynamo already performs this, so this is for non-strict mode.

    Additional inputs:
        equalities_inputs: the equality constraints to use for guards
        original_signature: the signature of the forward method
    """
    shape_env = fake_mode.shape_env
    if shape_env is None:
        raise AssertionError("fake_mode.shape_env must not be None")
    if shape_env.tracked_fakes is None:
        raise AssertionError("shape_env.tracked_fakes must not be None")

    placeholders = [tf.fake for tf in shape_env.tracked_fakes]
    sources = [tf.source for tf in shape_env.tracked_fakes]
    input_contexts = [tf.symbolic_context for tf in shape_env.tracked_fakes]
    constraint_violation_error = None
    try:
        shape_env.produce_guards(
            placeholders,
            sources,
            input_contexts=input_contexts,
            equalities_inputs=equalities_inputs,
            ignore_static=False,
        )
    except ConstraintViolationError as e:
        constraint_violation_error = e

    shape_env.frozen = True
    dim_constraints = shape_env.dim_constraints
    if dim_constraints is None:
        # Expected when shape_env.produce_guards throws an early constraint violation error.
        # There is nothing to solve for in this case.
        # TODO(avik): Maybe record the constraint violation error instead and replay later?
        if not constraint_violation_error:
            raise AssertionError(
                "expected constraint_violation_error when dim_constraints is None"
            )
        raise constraint_violation_error
    dim_constraints.solve()
    forced_specializations = dim_constraints.forced_specializations()

    msg = dim_constraints.prettify_results(
        original_signature,
        dynamic_shapes,  # type: ignore[arg-type]
        constraint_violation_error,
        forced_specializations,  # type: ignore[arg-type]
    )

    if constraint_violation_error:
        if constraint_violation_error.args:
            constraint_violation_error.args = (
                constraint_violation_error.args[0] + msg,
            )
        else:
            constraint_violation_error.args = (msg,)
    elif forced_specializations:
        constraint_violation_error = ConstraintViolationError(msg)
    if constraint_violation_error:
        raise constraint_violation_error


def is_int(x: object) -> TypeGuard[int | torch.SymInt]:
    return isinstance(x, int) or (isinstance(x, torch.SymInt) and x.node.expr.is_number)


def _constrain_user_specified_dimhint_range(
    symint: torch.SymInt,
    hint: int,
    dim: _DimHint,
    range_constraints: dict["Expr", "ValueRanges[Expr]"],
    shape_env: ShapeEnv,
    keypath: KeyPath,
    i: int | None = None,
) -> str | None:
    trace_vr = (
        range_constraints[symint.node.expr]
        if not is_int(symint)
        else ValueRanges(int(symint), int(symint))
    )

    # warn on 0/1 specialization for Dim.AUTO; not an actual error
    if dim.type == _DimHintType.AUTO and trace_vr.is_singleton() and hint in (0, 1):
        pathstr = f"inputs{pytree.keystr(keypath)}"
        if i is not None:
            pathstr += f".shape[{i}]"
        msg = (
            f"dimension {pathstr} 0/1 specialized; Dim.AUTO was specified along "
            + f"with a sample input with hint = {hint}."
        )
        log.warning(msg)

    try:
        user_vr = ValueRanges(
            lower=0 if dim.min is None else dim.min,
            upper=int_oo if dim.max is None else dim.max,
        )
        if is_int(symint):
            out_vr = trace_vr & user_vr
        else:
            range_constraints[symint.node.expr] &= user_vr
            shape_env.var_to_range[symint.node._expr] &= user_vr
            out_vr = range_constraints[symint.node.expr]

        # check for Dim.DYNAMIC specializations; special case error message on 0/1
        if dim.type == _DimHintType.DYNAMIC and out_vr.is_singleton():
            path = f"inputs{pytree.keystr(keypath)}"
            if i is not None:
                path += f".shape[{i}]"
            if (
                trace_vr.is_singleton()
                and hint in (0, 1)
                and not torch.fx.experimental._config.backed_size_oblivious
            ):
                msg = (
                    f"- Received user-specified dim hint Dim.DYNAMIC(min={dim.min}, max={dim.max}), "
                    f"but export 0/1 specialized due to hint of {hint} for dimension {path}."
                )
            else:
                msg = (
                    f"- Received user-specified dim hint Dim.DYNAMIC(min={dim.min}, max={dim.max}), "
                    f"but tracing inferred a static shape of {out_vr.lower} for dimension {path}."
                )
            return msg

    except torch.utils._sympy.value_ranges.ValueRangeError:
        path = f"inputs{pytree.keystr(keypath)}"
        if i is not None:
            path += f".shape[{i}]"
        msg = (
            f"- Received user-specified min/max range of [{dim.min}, {dim.max}], "
            f"conflicting with the inferred min/max range of [{trace_vr.lower}, {trace_vr.upper}], "
            f"for {path}."
        )
        return msg

    return None


def make_constraints(
    fake_mode: FakeTensorMode,
    gm: torch.fx.GraphModule,
    combined_args: dict[str, Any],
    dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None,
    num_lifted_inputs: int,
) -> dict["Expr", "ValueRanges[Expr]"]:
    """
    Given a fake mode's shape env and user-specified dynamic shapes,
    return the resulting range constraints and equality constraints.

    Additional args:
        num_lifted_inputs: the number of non-user-input placeholder nodes in the graph
        (used only to enumerate the user-input nodes)
    """

    shape_env = fake_mode.shape_env
    if shape_env is None:
        raise AssertionError("fake_mode.shape_env must not be None")
    inline_constraints = gm.meta.get("inline_constraints", [])
    range_constraints = defaultdict(lambda: ValueRanges(0, int_oo)) | inline_constraints
    if not dynamic_shapes:
        return dict(range_constraints)

    # clean up dynamic markers from tensors
    flat_paths, flat_args = zip(*pytree.tree_flatten_with_path(combined_args)[0])
    for arg in flat_args:
        if isinstance(arg, torch.Tensor):
            _clean_dynamic_markers(arg)

    # get individual dynamic shapes spec for each input
    if not isinstance(dynamic_shapes, dict):
        if not isinstance(dynamic_shapes, (tuple, list)):
            raise AssertionError(
                f"expected dict, tuple, or list for dynamic_shapes, got {type(dynamic_shapes)}"
            )
        combined_args = type(dynamic_shapes)(combined_args.values())  # type: ignore[assignment, misc]
    flat_dynamic_shapes = _flatten_dynamic_shapes(combined_args, dynamic_shapes)

    # check number of shapes vs. number of inputs
    num_placeholders = [node.op == "placeholder" for node in gm.graph.nodes].count(True)
    if len(flat_dynamic_shapes) != num_placeholders - num_lifted_inputs:
        raise AssertionError(
            f"expected {num_placeholders - num_lifted_inputs} shapes, got {len(flat_dynamic_shapes)}"
        )

    free_symbols = set()
    range_violations: list[str] = []
    for input_index, node in enumerate(gm.graph.nodes):
        meta_val = node.meta.get("val")

        if (
            input_index < num_lifted_inputs
            or node.op != "placeholder"
            or meta_val is None
        ):
            continue

        elif _is_constant_argument(meta_val) or isinstance(meta_val, CustomObjArgument):
            continue

        shape_spec = flat_dynamic_shapes[input_index - num_lifted_inputs]
        keypath = flat_paths[input_index - num_lifted_inputs]
        flat_arg = flat_args[input_index - num_lifted_inputs]

        if isinstance(meta_val, int) or (
            isinstance(meta_val, torch.SymInt) and meta_val.node.expr.is_number
        ):
            pass

        elif isinstance(meta_val, torch.SymInt):
            if shape_spec is not None and isinstance(shape_spec, _DimHint):
                hint = flat_arg
                range_constraints[meta_val.node.expr] &= shape_env.bound_sympy(
                    meta_val.node._expr
                )
                violation = _constrain_user_specified_dimhint_range(
                    meta_val,
                    hint,
                    shape_spec,
                    range_constraints,
                    shape_env,
                    keypath,
                    None,
                )
                if violation:
                    range_violations.append(violation)
            else:
                raise RuntimeError("nyi")
            free_symbols.update(meta_val.node.expr.free_symbols)

        elif isinstance(meta_val, torch.Tensor):
            for i, d in enumerate(node.meta["val"].shape):
                dim = None
                if isinstance(shape_spec, (list, tuple)):
                    dim = shape_spec[i]
                elif isinstance(shape_spec, dict):
                    dim = shape_spec.get(i)
                if not is_int(d):
                    # Compute the range constraint for the symbolic expression corresponding
                    # to this shape dimension and store it.
                    if dim is None or isinstance(dim, _DimHint):
                        range_constraints[d.node.expr] &= shape_env.bound_sympy(
                            d.node.expr
                        )
                    else:
                        range_constraints[d.node.expr] &= ValueRanges(
                            lower=dim.min, upper=dim.max
                        )

                    free_symbols.update(d.node.expr.free_symbols)

                # check user-specified min/max range for DimHints;
                # we might want to do this even if model tracing inferred a static dimension.
                if isinstance(dim, _DimHint):
                    hint = flat_arg.shape[i]
                    violation = _constrain_user_specified_dimhint_range(
                        d, hint, dim, range_constraints, shape_env, keypath, i
                    )
                    if violation:
                        range_violations.append(violation)
        else:
            raise RuntimeError(f"Unfamiliar meta val: {meta_val}")

    if range_violations:
        prefix = "Found the following conflicts between user-specified ranges and inferred ranges from model tracing:\n"
        raise ValueError(prefix + "\n".join(range_violations))

    for symbol in free_symbols:
        if symbol not in range_constraints:
            # Placeholders can have symbolic shapes that are derived expressions.
            # The above code will record direct range constraints for them
            # so that we can do runtime assertions. In addition, for serde checks
            # we want to record range constraints for their root symbols.
            range_constraints[symbol] = shape_env.var_to_range[symbol]

    return dict(range_constraints)


def _gather_constant_attrs(m: torch.nn.Module) -> ConstantAttrMap:
    """Search the module hierarchy, gathering up all tensor and ScriptObject constants.

    Returns a dictionary mapping hash(value) to the name of the constant. We
    have to abuse `hash` here unfortunately, see: [ScriptObject hash].
    """
    constants = ConstantAttrMap()
    buffers_parameters = set(m.buffers())
    buffers_parameters.update(m.parameters())

    def inner(
        m: torch.nn.Module, prefix_atoms: list[str], constants: ConstantAttrMap
    ) -> None:
        for k, v in m.__dict__.items():
            if isinstance(
                v,
                (
                    torch.Tensor,
                    torch.ScriptObject,
                    FakeScriptObject,
                ),
            ):
                if v in buffers_parameters:
                    # filter out buffers and parameters, leaving only constants
                    continue

                fqn = ".".join(prefix_atoms + [k])
                constants.add(v, fqn)
        for k, v in m.named_children():
            inner(v, prefix_atoms + [k], constants)

    inner(m, [], constants)
    return constants


def _get_graph_inputs_of_type_nn_module(
    args: tuple[tuple[Any], dict[Any, Any]] | None,
) -> set[type[torch.nn.Module]]:
    if args is None:
        return set()
    module_types = set()
    for arg in pytree.tree_leaves(args):
        if isinstance(arg, torch.nn.Module):
            module_types.add(type(arg))
    return module_types


def _enter_enable_graph_inputs_of_type_nn_module(
    module_types: set[type[torch.nn.Module]],
) -> None:
    for t in module_types:
        torch._export.utils.register_module_as_pytree_input_node(t)


def _exit_enable_graph_inputs_of_type_nn_module(
    module_types: set[type[torch.nn.Module]],
) -> None:
    for t in module_types:
        torch._export.utils.deregister_module_as_pytree_input_node(t)


@contextlib.contextmanager
def _enable_graph_inputs_of_type_nn_module(
    args: tuple[tuple[Any], dict[Any, Any]] | None,
) -> Generator[None, None, None]:
    if args is None:
        yield
        return

    module_types = _get_graph_inputs_of_type_nn_module(args)
    _enter_enable_graph_inputs_of_type_nn_module(module_types)
    try:
        yield
    finally:
        _exit_enable_graph_inputs_of_type_nn_module(module_types)


@contextlib.contextmanager
def _fakify_module_inputs(
    args: tuple[Any],
    kwargs: dict[Any, Any],
    fake_mode: torch._subclasses.fake_tensor.FakeTensorMode,
) -> Generator[None, None, None]:
    # This context manager is used to fakify module inputs.
    # Inputs:
    #   args, kwargs: the args and kwargs containing module inputs that haven't been fakified.
    #   fake_mode: the fake mode to be used for fakifying script objects. It's the same mode that fakify input tensors.

    ctxs = [_enable_graph_inputs_of_type_nn_module((args, kwargs))]
    for arg in pytree.tree_leaves((args, kwargs)):
        if isinstance(arg, torch.nn.Module):
            fake_params_buffers = _fakify_params_buffers(fake_mode, arg)
            ctxs.append(
                torch.nn.utils.stateless._reparametrize_module(
                    arg,
                    fake_params_buffers,
                    tie_weights=True,
                    strict=True,
                    stack_weights=True,
                )
            )
    with contextlib.ExitStack() as stack:
        for ctx in ctxs:
            stack.enter_context(ctx)
        yield


@contextlib.contextmanager
def _fakify_script_objects(
    mod: _ModuleT,
    args: Sequence[Any],
    kwargs: dict[Any, Any],
    fake_mode: torch._subclasses.fake_tensor.FakeTensorMode | None,
) -> Generator[tuple[_ModuleT, Any, Any, ConstantAttrMap, dict[Any, Any]], None, None]:
    # This context manager is used to fakify script objects into FakeScriptObject.
    # Inputs:
    #   mod: the module to be exported, it (and its recursive submodules)'s script object attrs haven't been fakified.
    #   args, kwargs: the args and kwargs inputs for mod, script object inputs haven't been fakified.
    #   fake_mode: the fake mode to be used for fakifying script objects. It's the same mode that fakify input tensors.
    #
    # Returns:
    #   mod: the patched module, its (and its recursive submodules) script object attrs have been fakified.
    #   fake_args, fake_kwargs: new fakified args and kwargs.
    #        Script object inputs have been fakified. Don't touch the tensors.
    #   fake_constant_attrs: a new map from FakeScriptObject to the fqn of the original script object.
    #   fake_to_real: a mapping between FakeScriptObject and the original script object in order to un-do the patching.

    constant_attrs: ConstantAttrMap = _gather_constant_attrs(mod)
    if any(isinstance(obj, FakeScriptObject) for obj in constant_attrs.values()):
        raise AssertionError("Mod shouldn't contain any FakeScriptObject.")
    if pytree.tree_any(lambda obj: isinstance(obj, FakeScriptObject), (args, kwargs)):
        raise AssertionError("args and kwargs shouldn't contain any FakeScriptObject.")

    patched_attr: dict[str, Any] = {}
    fake_constant_attrs = ConstantAttrMap()
    fake_to_real: dict[Any, Any] = {}

    def _maybe_fakify_obj(obj: object) -> FakeScriptObject | torch.ScriptObject:
        fake_obj = torch._library.fake_class_registry.maybe_to_fake_obj(fake_mode, obj)
        fake_to_real[fake_obj] = obj
        return fake_obj

    def _leaf_mod_and_attr(
        mod: torch.nn.Module, attr_fqn: str
    ) -> tuple[torch.nn.Module, str]:
        *prefix_attr, last_attr = attr_fqn.split(".")
        cur_mod = mod
        for attr in prefix_attr:
            cur_mod = getattr(cur_mod, attr)
        return cur_mod, last_attr

    try:
        for obj, fqns in constant_attrs.items():
            if torch._library.fake_class_registry._is_script_object(
                obj
            ) or is_custom_class_obj(obj):
                fake_script_obj = _maybe_fakify_obj(obj)
                for fqn in fqns:
                    cur_mod, attr = _leaf_mod_and_attr(mod, fqn)
                    if obj is not getattr(cur_mod, attr):
                        raise AssertionError(
                            f"obj mismatch at {fqn}: expected {obj}, got {getattr(cur_mod, attr)}"
                        )
                    setattr(cur_mod, attr, fake_script_obj)
                    fake_constant_attrs.add(fake_script_obj, fqn)
                    patched_attr[fqn] = obj
            else:
                for fqn in fqns:
                    fake_constant_attrs.add(obj, fqn)

        fake_args, fake_kwargs = pytree.tree_map_only(
            torch.ScriptObject, _maybe_fakify_obj, (args, kwargs)
        )
        yield (mod, fake_args, fake_kwargs, fake_constant_attrs, fake_to_real)
    finally:
        for fqn, orig_obj in patched_attr.items():
            cur_mod, attr = _leaf_mod_and_attr(mod, fqn)
            setattr(cur_mod, attr, orig_obj)


class _NonStrictTorchFunctionHandler(torch.overrides.TorchFunctionMode):
    """
    1. Handles data-dependent errors raised by torch function calls in non-strict.

    Any data-dependent error is due to some condition on unbacked symints
    that cannot be resolved. A mechanical way of fixing the error is to use
    a torch._check() call to assert either that condition or its negation.
    The handler suggests these options as code and points to the location
    of the torch function call that raised the error as part of the error
    message shown to the user, who can then simply select and copy-paste
    a suggested fix at that location.

    NOTE: Not all data-dependent errors are raised by torch function calls.
    In particular, conditions on unbacked symints can appear outside such
    calls, and as such are not handled here.

    2. Overrides torch functions that are known to cause problems in non-strict.

    Certain Python features, such as indexing/slicing, cannot be intercepted
    in non-strict. Likewise, certain legacy ops, such as distributed collectives,
    may need to be mapped to other ops. When there is special handling in Dynamo
    for such things, tracing can fail in non-strict (while succeeding in strict).
    Fortunately, redirecting to other torch functions can often fix such issues.

    3. Handles line-of-code logging for each torch function call in non-strict.

    Usage: TORCHEXPORT_EXTENDED_DEBUG_CURRENT_LOC=1 TORCH_LOGS="+export" ...
    """

    def _override(
        self, func: Callable[..., object], args: Sequence[Any], kwargs: dict[str, Any]
    ) -> tuple[Callable[..., object], Sequence[Any], dict[str, Any]]:
        if torch.distributed.is_available():
            from torch.distributed._functional_collectives import (
                _remap_traceable_collective,
            )

            # Redirect legacy collectives to functional collectives, following
            # Dynamo. See torch/distributed/_functional_collectives.py for details.
            remapped = _remap_traceable_collective(func, args, kwargs)
            if remapped is not None:
                return remapped
        if func is torch.tensor:
            # Redirect to Python implementation of torch.tensor for data with symints.
            # NOTE(avik): We don't unconditionally redirect to this implementation
            # because it has some known incompletenesses, e.g., it doesn't support
            # empty data. See https://github.com/pytorch/pytorch/issues/143216
            if any(
                isinstance(a, (torch.SymInt, torch.SymFloat, torch.SymBool))
                for a in pytree.tree_flatten(args[0])[0]
            ):
                return torch._refs.tensor, args, kwargs
        if func.__name__ == "__getitem__" and isinstance(args[0], torch.Tensor):

            def is_scalar_tensor_index(item: object) -> TypeGuard[torch.Tensor]:
                if not isinstance(item, torch.Tensor) or item.ndim != 0:
                    return False
                # vmap BatchedTensor scalar indices must remain tensor indices;
                # calling item() on them is unsupported.
                if torch._C._functorch.is_batchedtensor(item):
                    return False

                from torch._prims_common import is_integer_dtype

                return is_integer_dtype(item.dtype)

            def maybe_tensor_index_item(item: object) -> Any:
                if is_scalar_tensor_index(item):
                    return item.item()
                return item

            def rewrite(
                dim: int, item: Any
            ) -> tuple[int, tuple[Callable[..., object], list[Any]]] | None:
                # Redirect to torch.select for indexing.
                if item is None:
                    return dim + 1, (torch.unsqueeze, [dim])
                if isinstance(item, (int, torch.SymInt)):
                    return dim, (torch.select, [dim, item])
                if is_scalar_tensor_index(item):
                    return dim, (
                        lambda t, dim, item: torch.select(
                            t, dim, maybe_tensor_index_item(item)
                        ),
                        [dim, item],
                    )
                # Redirect to torch.ops.aten.slice for slicing.
                if isinstance(item, slice):
                    step = 1 if item.step is None else item.step
                    if (
                        item.start is None
                        and item.stop is None
                        and isinstance(step, int)
                        and step == 1
                    ):
                        # no-op
                        return dim + 1, (lambda t: t, [])
                    return dim + 1, (
                        lambda t, dim, start, stop, step: torch.ops.aten.slice(
                            t,
                            dim,
                            maybe_tensor_index_item(start),
                            maybe_tensor_index_item(stop),
                            maybe_tensor_index_item(step),
                        ),
                        [dim, item.start, item.stop, step],
                    )
                # Otherwise do nothing.

            items = list(args[1]) if isinstance(args[1], tuple) else [args[1]]

            has_symint = False
            index_ellipsis = None
            t = args[0]
            n_none_slices = t.ndim + 1
            for i, item in enumerate(items):
                if (
                    isinstance(item, torch.SymInt)
                    or (
                        isinstance(item, slice)
                        and any(
                            isinstance(s, torch.SymInt) or is_scalar_tensor_index(s)
                            for s in (item.start, item.stop, item.step)
                        )
                    )
                    or is_scalar_tensor_index(item)
                ):
                    has_symint = True
                if item is Ellipsis:
                    index_ellipsis = i
                if item is not None:
                    n_none_slices -= 1

            # only rewrite when there are symints
            if has_symint:
                if index_ellipsis is not None:
                    none_slices = [slice(None)] * n_none_slices
                    items[index_ellipsis : index_ellipsis + 1] = none_slices

                dim = 0
                # Sequence rewrites.
                sequence: list[tuple[Callable[..., object], list[Any]]] = []
                for item in items:
                    if (r := rewrite(dim, item)) is None:
                        return func, args, kwargs
                    dim, call_spec = r
                    sequence.append(call_spec)

                def run() -> Any:
                    # Run sequence.
                    # pyrefly: ignore [bad-index, index-error]
                    t = args[0]
                    for _method, _args in sequence:
                        t = _method(t, *_args)
                    return t

                return run, [], {}

        return func, args, kwargs

    def __torch_function__(
        self,
        func: Callable[..., object],
        types: Sequence[type],
        args: Sequence[Any] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        if torch.compiler.is_dynamo_compiling():
            return func(*args, **kwargs)

        if log.isEnabledFor(logging.DEBUG) and config.extended_debug_current_loc:
            frame = _find_user_code_frame()
            if frame is not None:
                log.debug(
                    "%s called at %s:%s in %s",
                    func.__qualname__,
                    frame.f_code.co_filename,
                    frame.f_lineno,
                    frame.f_code.co_name,
                )

        func, args, kwargs = self._override(func, args, kwargs)
        try:
            return func(*args, **kwargs)
        except GuardOnDataDependentSymNode as e:
            _suggest_fixes_for_data_dependent_error_non_strict(e)
            raise
        except RuntimeError as e:
            if _is_raw_triton_kernel_data_ptr_error(func, e):
                raise RuntimeError(_RAW_TRITON_KERNEL_NON_STRICT_EXPORT_ERROR) from e
            raise
<|endoftext|># mypy: allow-untyped-defs
import copy
import dataclasses
import functools
import io
import json
import logging
import os
import re
import sys
import types
import warnings
import weakref
import zipfile
from collections import OrderedDict
from contextlib import contextmanager
from functools import lru_cache

from typing import Any, Optional, TYPE_CHECKING, Union
from collections.abc import Callable
from unittest.mock import patch

import torch
import torch.fx
import torch.utils._pytree as pytree

from torch._dispatch.python import enable_python_dispatcher
from torch._guards import compile_context
from torch._utils_internal import log_export_usage
from torch.export._tree_utils import reorder_kwargs
from torch.export.graph_signature import (
    ArgumentSpec,
    ConstantArgument,
    ExportGraphSignature,
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    SymIntArgument,
    SymBoolArgument,
    SymFloatArgument,
    TensorArgument,
)
from torch.fx import traceback as fx_traceback
from torch.fx._compatibility import compatibility
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo

from .wrappers import _wrap_submodules
from .utils import _materialize_cpp_cia_ops
from . import config

if TYPE_CHECKING:
    from torch._C._aoti import AOTIModelContainerRunner

log = logging.getLogger(__name__)

@dataclasses.dataclass
class ExportDynamoConfig:
    """
    Manage Export-specific configurations of Dynamo.
    """
    allow_rnn: bool = True


# We only want to print this once to avoid flooding logs in workflows where aot_compile_warning
# is called multiple times.
@lru_cache
def aot_compile_warning():

    log.warning("+============================+")
    log.warning("|     !!!   WARNING   !!!    |")
    log.warning("+============================+")
    log.warning(
        "torch._export.aot_compile()/torch._export.aot_load() is being deprecated, please switch to "
        "directly calling torch._inductor.aoti_compile_and_package(torch.export.export())/"
        "torch._inductor.aoti_load_package() instead.")


def aot_compile(
    f: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    *,
    dynamic_shapes: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
    remove_runtime_assertions: bool = False,
    disable_constraint_solver: bool = False,
    same_signature: bool = True,
) -> list[Any] | str:
    """
    Note: this function is not stable yet

    Traces either an nn.Module's forward function or just a callable with PyTorch
    operations inside, generates executable cpp code from the program, and returns
    the path to the generated shared library

    Args:
        f: the `nn.Module` or callable to trace.

        args: example positional inputs.

        kwargs: optional example keyword inputs.

        dynamic_shapes: Should either be:
            1) a dict from argument names of ``f`` to their dynamic shape specifications,
            2) a tuple that specifies dynamic shape specifications for each input in original order.
            If you are specifying dynamism on keyword args, you will need to pass them in the order that
            is defined in the original function signature.

            The dynamic shape of a tensor argument can be specified as either
            (1) a dict from dynamic dimension indices to :func:`Dim` types, where it is
            not required to include static dimension indices in this dict, but when they are,
            they should be mapped to None; or (2) a tuple / list of :func:`Dim` types or None,
            where the :func:`Dim` types correspond to dynamic dimensions, and static dimensions
            are denoted by None. Arguments that are dicts or tuples / lists of tensors are
            recursively specified by using mappings or sequences of contained specifications.

        options: A dictionary of options to control inductor

        disable_constraint_solver: Whether the dim constraint solver must be disabled.

    Returns:
        Path to the generated shared library
    """
    from torch.export._trace import _export_to_torch_ir
    from torch._inductor.decomposition import select_decomp_table
    from torch._inductor import config as inductor_config

    aot_compile_warning()

    if inductor_config.is_predispatch:
        gm = torch.export._trace._export(f, args, kwargs, dynamic_shapes, pre_dispatch=True).module()
    else:
        # We want to export to Torch IR here to utilize the pre_grad passes in
        # inductor, which run on Torch IR.
        with torch._export.config.patch(use_new_tracer_experimental=True):
            gm = _export_to_torch_ir(
                f,
                args,
                kwargs,
                dynamic_shapes,
                disable_constraint_solver=disable_constraint_solver,
                same_signature=same_signature,
                # Disabling this flag, because instead we can rely on the mapping
                # dynamo_flat_name_to_original_fqn which is coming from Dynamo.
                restore_fqn=False,
            )

    with torch.no_grad():
        so_path = torch._inductor.aot_compile(gm, args, kwargs, options=options)  # type: ignore[arg-type]

    if not isinstance(so_path, (str, list)):
        raise AssertionError(f"expected str or list, got {type(so_path)}")
    return so_path

def aot_load(so_path: str, device: str) -> Callable:
    """
    Loads a shared library generated by aot_compile and returns a callable

    Args:
        so_path: Path to the shared library

    Returns:
        A callable
    """
    aot_compile_warning()

    if device == "cpu":
        runner: AOTIModelContainerRunner = torch._C._aoti.AOTIModelContainerRunnerCpu(so_path, 1)
    elif device == "cuda" or device.startswith("cuda:"):
        runner = torch._C._aoti.AOTIModelContainerRunnerCuda(so_path, 1, device)
    elif device == "xpu" or device.startswith("xpu:"):
        runner = torch._C._aoti.AOTIModelContainerRunnerXpu(so_path, 1, device)
    elif device == "mps" or device.startswith("mps:"):
        runner = torch._C._aoti.AOTIModelContainerRunnerMps(so_path, 1)
    else:
        raise RuntimeError("Unsupported device " + device)

    def optimized(*args, **kwargs):
        call_spec = runner.get_call_spec()
        in_spec = pytree.treespec_loads(call_spec[0])
        out_spec = pytree.treespec_loads(call_spec[1])
        flat_inputs = pytree.tree_flatten((args, reorder_kwargs(kwargs, in_spec)))[0]
        flat_inputs = [x for x in flat_inputs if isinstance(x, torch.Tensor)]
        flat_outputs = runner.run(flat_inputs)
        return pytree.tree_unflatten(flat_outputs, out_spec)

    return optimized
<|endoftext|># mypy: allow-untyped-defs
import operator
import traceback
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import torch
from torch import fx
from torch._dispatch.python import enable_python_dispatcher
from torch._export.pass_infra.node_metadata import NodeMetadata
from torch._export.pass_infra.proxy_value import ProxyValue
from torch._higher_order_ops.map import _unstack_pytree
from torch._subclasses import FakeTensor, UnsupportedFakeTensorException
from torch._subclasses.fake_tensor import (
    FakeTensorMode,
    is_fake_tensor,
    maybe_get_fake_constant,
    maybe_get_fake_mode,
)
from torch.fx import traceback as fx_traceback
from torch.fx.experimental.proxy_tensor import PythonKeyTracer
from torch.fx.experimental.symbolic_shapes import (
    compute_unbacked_bindings,
    PropagateUnbackedSymInts,
)
from torch.fx.graph import CodeGen
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata
from torch.utils import _pytree as pytree


__all__ = ["_ExportPassBaseDeprecatedDoNotUse"]


Argument = Any
Value = Any
Fn = Callable[..., Any]
PassType = Callable[[torch.fx.GraphModule], PassResult | None]


_TORCH_SYM_OPS: set[Callable] = {
    torch.sym_int,
    torch.sym_float,
    torch.sym_ite,
    torch.sym_max,
    torch.sym_min,
    torch.sym_not,
    torch.sym_sqrt,
}


class ExportPassBaseError(RuntimeError):
    pass


class _ExportPassBaseDeprecatedDoNotUse(PassBase):
    """
    Interpreter-based pass class to help users maintain the IR spec while writing
    transformations.
    """

    @staticmethod
    def _create_dummy_node_metadata():
        return NodeMetadata({"stack_trace": "".join(traceback.format_stack(limit=1))})

    class ExportTracer(PythonKeyTracer):
        def __init__(
            self, callback: "_ExportPassBaseDeprecatedDoNotUse", codegen: CodeGen
        ) -> None:
            super().__init__()
            self.callback = callback
            self.root = torch.nn.Module()
            self.graph = torch.fx.Graph()
            self.graph.set_codegen(codegen)
            self.tensor_attrs: dict[str, torch.Tensor] = {}  # type: ignore[assignment]
            self.fake_tensor_mode: FakeTensorMode | None = None
            self.submodules: dict[torch.nn.Module, str] = {}

        def trace(self) -> None:  # type: ignore[override]
            raise ExportPassBaseError("ExportTracer doesn't support trace().")

        def create_arg(self, a: Argument) -> torch.fx.Node:
            if isinstance(a, torch.nn.Module):
                if a not in self.submodules:
                    name_submodule = f"submodule_{len(self.submodules)}"
                    self.root.add_module(name_submodule, a)
                    self.submodules[a] = name_submodule
            elif is_fake_tensor(a):
                if maybe_get_fake_constant(a) is None:
                    raise ExportPassBaseError(f"Cannot add {a} to graph.")
                a = maybe_get_fake_constant(a)
            node = super().create_arg(a)
            if (
                isinstance(a, torch.Tensor)
                and isinstance(node, torch.fx.Node)
                and node.op == "get_attr"
            ):
                self.set_metadata(node, a)
                self.callback.on_attr(ProxyValue(a, node))
            return node

        def set_metadata(
            self,
            node: torch.fx.Node,
            value: Argument,
        ) -> None:
            # propagate the fake tensor or sym nodes
            def make_val(
                x: Argument,
            ) -> (
                FakeTensor
                | torch.SymInt
                | torch.SymFloat
                | torch.SymBool
                | int
                | float
                | bool
                | str
                | None
            ):
                if isinstance(x, FakeTensor):  # noqa: ISINSTANCE_FAKE_TENSOR
                    return x
                elif isinstance(x, torch.Tensor):
                    if x.is_quantized:
                        # TODO (tmanlaibaatar) properly support Quantized FakeTensor
                        x = torch.dequantize(x)

                    try:
                        if self.fake_tensor_mode is None:
                            raise AssertionError("fake_tensor_mode must not be None")
                        # TODO we should allocate static shapes
                        # for param/buffer values
                        if isinstance(x, torch.nn.Parameter):
                            fake_tensor = self.fake_tensor_mode.from_tensor(
                                x, static_shapes=True
                            )
                        else:
                            fake_tensor = self.fake_tensor_mode.from_tensor(x)
                    except UnsupportedFakeTensorException:
                        # TODO: This is just a workaround to get over the
                        # x.as_subclass error
                        print(
                            "Fakeifying a Tensor subclass is not supported \
                            right now. Instead a TensorMetadata is used."
                        )
                        fake_tensor = None
                    return fake_tensor
                elif isinstance(
                    x,
                    (
                        torch.SymInt,
                        torch.SymFloat,
                        torch.SymBool,
                        int,
                        float,
                        bool,
                        str,
                    ),
                ):
                    return x
                else:
                    return None

            node.meta["val"] = pytree.tree_map(make_val, value)

            # Set the tensor_metadata for values that do not have a corresponding FakeTensor
            def make_tensor_meta(x: Argument) -> TensorMetadata | None:
                if not is_fake_tensor(x) and isinstance(x, torch.Tensor):
                    if x.is_quantized:
                        # TODO (tmanlaibaatar) properly support Quantized FakeTensor
                        x = torch.dequantize(x)

                    try:
                        if self.fake_tensor_mode is None:
                            raise AssertionError("fake_tensor_mode must not be None")
                        _ = self.fake_tensor_mode.from_tensor(x)
                        tensor_meta = None
                    except UnsupportedFakeTensorException:
                        # TODO: This is just a workaround to get over the
                        # x.as_subclass error
                        tensor_meta = _extract_tensor_metadata(x)
                    return tensor_meta
                else:
                    return None

            node.meta["tensor_meta"] = pytree.tree_map(make_tensor_meta, value)

    class ExportInterpreter(fx.Interpreter):
        def __init__(
            self, callback: "_ExportPassBaseDeprecatedDoNotUse", gm: fx.GraphModule
        ) -> None:
            super().__init__(gm)
            self.callback = callback
            self.node: torch.fx.Node = next(iter(gm.graph.nodes))

        # pyrefly: ignore [bad-override]
        def placeholder(
            self,
            target: str,  # type: ignore[override]
            args: tuple[Argument, ...],
            kwargs: dict[str, Argument],
        ) -> ProxyValue:
            arg = super().placeholder(target, args, kwargs)
            return self.callback.placeholder(target, arg, NodeMetadata(self.node.meta))

        def output(
            self,
            target: torch.fx.node.Target,
            args: tuple[Argument, ...],
            kwargs: dict[str, Argument],
        ) -> ProxyValue:
            return self.callback.output(args[0], NodeMetadata(self.node.meta)).data  # type: ignore[return-value]

        def call_function(
            self,
            target: torch.fx.node.Target,
            args: tuple[Argument, ...],
            kwargs: dict[str, Argument],
        ) -> ProxyValue:
            meta = NodeMetadata(self.node.meta)

            if target is operator.getitem:
                value, key = args
                return self.callback.call_getitem(value, key, meta)
            elif getattr(target, "__module__", None) in {
                "_operator",
                "builtins",
                "math",
            }:
                if not callable(target):
                    raise AssertionError(f"expected callable target, got {target}")
                return self.callback.call_sym(target, args, meta)
            elif target in _TORCH_SYM_OPS:
                if not callable(target):
                    raise AssertionError(f"expected callable target, got {target}")
                return self.callback.call_sym(target, args, meta)
            elif isinstance(
                target, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)
            ):
                return self.callback.call_operator(
                    target,
                    args,
                    kwargs,
                    meta,
                )
            elif target is torch.ops.higher_order.cond:
                pred, true_fn, false_fn, inputs = args
                return self.callback.call_cond(pred, true_fn, false_fn, inputs, meta)
            elif target is torch.ops.higher_order.map_impl:
                f, mapped_args, operands = args  # type: ignore[assignment]
                return self.callback.call_map(f, mapped_args, operands, meta)
            # For other unregistered HigherOrderOps, just interpret them blindly
            elif isinstance(target, torch._ops.HigherOrderOperator):
                return self.callback._fx(
                    "call_function",
                    target,
                    args,
                    kwargs,
                    meta,
                )
            else:
                raise ExportPassBaseError(f"Unsupported target type: {target}")

        def get_attr(  # type: ignore[override]
            self,
            target: str,
            args: tuple[Argument, ...],
            kwargs: dict[str, Argument],
        ) -> Argument:
            return super().get_attr(target, args, kwargs)

        def call_module(
            self,
            target: torch.fx.node.Target,
            args: tuple[Argument, ...],
            kwargs: dict[str, Argument],
        ) -> None:
            raise ExportPassBaseError("call_module is not supported.")

        def call_method(  # type: ignore[override]
            self,
            target: str,
            args: tuple[Argument, ...],
            kwargs: dict[str, Argument],
        ) -> None:
            raise ExportPassBaseError("call_method is not supported.")

        def run_node(self, n: torch.fx.Node) -> Argument:
            self.node = n
            self.callback.node_debug_str = n.format_node()
            return super().run_node(n)

    def __init__(self) -> None:
        self.interpreter = PropagateUnbackedSymInts(
            torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        )
        self.tracer = self.ExportTracer(self, CodeGen())
        self.fake_tensor_mode: FakeTensorMode | None = None
        self._initialized = True
        self.node_debug_str: str | None = None

    def _fx(
        self,
        kind: str,
        target: torch.fx.node.Target,
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        args_data, kwargs_data = pytree.tree_map_only(
            ProxyValue, lambda x: x.data, (args, kwargs)
        )
        res_data = getattr(self.interpreter, kind)(target, args_data, kwargs_data)
        args_proxy, kwargs_proxy = pytree.tree_map_only(
            ProxyValue, lambda x: x.proxy, (args, kwargs)
        )

        name = None
        if isinstance(target, torch._ops.OpOverload):
            name = self.tracer.graph._target_to_str(target.overloadpacket.__name__)

        res_proxy = self.tracer.create_proxy(
            kind, target, args_proxy, kwargs_proxy, name=name
        )
        res_proxy.node.meta.update(meta.data)
        if self.fake_tensor_mode and (shape_env := self.fake_tensor_mode.shape_env):
            if symbol_to_path := compute_unbacked_bindings(shape_env, res_data):
                res_proxy.node.meta["unbacked_bindings"] = symbol_to_path
        self.tracer.set_metadata(res_proxy.node, res_data)
        return ProxyValue(res_data, res_proxy)

    def inputs(self, graph_module: torch.fx.GraphModule) -> list[Argument]:
        # TODO(angelayi): Update this with what we decide to do for metadata in
        # the exported graph module
        if (args := graph_module.meta.get("args", None)) is not None:
            return list(args)

        def extract_input(node: torch.fx.Node) -> FakeTensor | None:
            if "val" in node.meta:
                fake = node.meta["val"]
                if hasattr(fake, "constant") and fake.constant is not None:
                    return fake.constant
                return fake
            elif tensor_meta := node.meta.get("tensor_meta"):
                if self.fake_tensor_mode is None:
                    raise AssertionError("fake_tensor_mode must not be None")
                return FakeTensor(
                    self.fake_tensor_mode,
                    torch.empty(
                        tensor_meta.shape,
                        dtype=tensor_meta.dtype,
                        device="meta",
                        requires_grad=tensor_meta.requires_grad,
                        memory_format=tensor_meta.memory_format,
                    ),
                    torch.device("cpu"),
                )
            elif len(node.users) == 0:
                return None
            raise ExportPassBaseError(
                f"Cannot construct an input for graph module: {graph_module}.",
            )

        return [
            extract_input(node)
            for node in graph_module.graph.nodes
            if node.op == "placeholder"
        ]

    def on_attr(self, attr: ProxyValue) -> None:
        pass

    def placeholder(self, name: str, arg: Argument, meta: NodeMetadata) -> ProxyValue:
        arg_proxy = self.tracer.create_proxy("placeholder", name, (), {})
        arg_proxy.node.meta = meta.data
        self.tracer.set_metadata(arg_proxy.node, arg)
        return ProxyValue(arg, arg_proxy)

    def call_operator(
        self,
        op,
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        return self._fx("call_function", op, args, kwargs, meta)

    def call_sym(
        self,
        target: Fn,
        args: tuple[Argument, ...],
        meta: NodeMetadata,
    ) -> ProxyValue:
        return self._fx("call_function", target, args, {}, meta)

    def call_cond(
        self,
        pred: ProxyValue,
        true_fn: torch.fx.GraphModule,
        false_fn: torch.fx.GraphModule,
        inputs: list[Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        true_branch = self.call_submodule(true_fn, tuple(inputs))
        false_branch = self.call_submodule(false_fn, tuple(inputs))
        if true_branch is None:
            raise AssertionError("true_branch must not be None")
        if false_branch is None:
            raise AssertionError("false_branch must not be None")
        return self._fx(
            "call_function",
            torch.ops.higher_order.cond,
            (pred, true_branch.graph_module, false_branch.graph_module, list(inputs)),
            {},
            meta,
        )

    def call_map(
        self,
        f: torch.fx.GraphModule,
        mapped_args: list[ProxyValue],
        operands: list[ProxyValue],
        meta: NodeMetadata,
    ) -> ProxyValue:
        xs = _unstack_pytree([arg.data for arg in mapped_args])[0]
        f_branch = self.call_submodule(f, tuple(xs + [arg.data for arg in operands]))
        if f_branch is None:
            raise AssertionError("f_branch must not be None")
        return self._fx(
            "call_function",
            torch.ops.higher_order.map_impl,
            (f_branch.graph_module, mapped_args, operands),
            {},
            meta,
        )

    def call_getitem(
        self, value: ProxyValue, key: int, meta: NodeMetadata
    ) -> ProxyValue:
        return self._fx("call_function", operator.getitem, (value, key), {}, meta)

    def output(self, results: list[Argument], meta: NodeMetadata) -> ProxyValue:
        return self._fx("output", "output", (results,), {}, meta)

    def call_submodule(
        self, graph_module: fx.GraphModule, inputs: tuple[Argument, ...]
    ) -> PassResult:
        prev_tracer, self.tracer = (
            self.tracer,
            self.ExportTracer(self, graph_module.graph._codegen),
        )
        self.tracer.fake_tensor_mode = prev_tracer.fake_tensor_mode
        interpreter = self.ExportInterpreter(self, graph_module)
        # pyrefly: ignore [bad-assignment]
        prev_interpreter, self.interpreter = (
            self.interpreter,
            torch.fx.Interpreter(  # type: ignore[assignment]
                torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
            ),
        )
        inputs_data = pytree.tree_map_only(ProxyValue, lambda x: x.data, inputs)
        with fx_traceback.preserve_node_meta():
            interpreter.run(*inputs_data)

        new_graph_module = torch.fx.GraphModule(self.tracer.root, self.tracer.graph)

        self.tracer = prev_tracer
        self.interpreter = prev_interpreter
        return PassResult(
            new_graph_module,
            True,
        )

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        if not getattr(self, "_initialized", False):
            raise ExportPassBaseError(
                "ExportPass is not initialized with __init__().",
            )

        inputs = self.inputs(graph_module)

        fake_tensor_mode = None
        for i in inputs:
            if is_fake_tensor(i):
                if (
                    fake_tensor_mode is not None
                    and fake_tensor_mode is not maybe_get_fake_mode(i)
                ):
                    raise AssertionError("Multiple fake tensor mode detected.")
                fake_tensor_mode = maybe_get_fake_mode(i)
        if fake_tensor_mode is None:
            self.tracer.fake_tensor_mode = FakeTensorMode(allow_non_fake_inputs=True)
            fake_tensor_mode = nullcontext()  # type: ignore[assignment]
            dispatcher_mode = nullcontext()  # type: ignore[assignment]
        else:
            fake_tensor_mode.allow_non_fake_inputs = True
            self.tracer.fake_tensor_mode = fake_tensor_mode
            dispatcher_mode = enable_python_dispatcher()  # type: ignore[assignment]
        self.fake_tensor_mode = self.tracer.fake_tensor_mode

        with fake_tensor_mode, dispatcher_mode:  # type: ignore[assignment, union-attr]
            result = self.call_submodule(graph_module, tuple(inputs))

        return result
<|endoftext|># mypy: allow-untyped-defs
import logging
import warnings
from collections.abc import Iterable
from typing import Any

import torch
import torch.export
import torch.export._trace
from torch._utils_internal import log_export_usage


log = logging.getLogger(__name__)

__all__ = ["report_exportability"]


def _generate_inputs_for_submodules(
    model: torch.nn.Module,
    target_submodules: Iterable[str],
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
) -> dict[str, tuple[Any, Any]]:
    """
    Generate inputs for targeting submodules in the given model. Note that if two submodules refer to the same obj, this
    function doesn't work.

    Args:
        model: root model.
        inputs: inputs to the root model.
        target_submodules: submodules that we want to generate inputs for.

    Returns:
        A dict that maps from submodule name to its inputs.
    """
    kwargs = kwargs or {}

    handles = []
    results = {}
    submodule_to_names = {mod: name for name, mod in model.named_modules()}

    def pre_forward(module, module_args, module_kwargs):
        results[submodule_to_names[module]] = (module_args, module_kwargs)

    try:
        for name, mod in model.named_modules():
            if name in target_submodules:
                handles.append(
                    mod.register_forward_pre_hook(pre_forward, with_kwargs=True)
                )
        model(*args, **kwargs)
    except Exception as e:
        warnings.warn(
            f"Failed to generate submodule inputs because of the following error:\n{e}",
            stacklevel=2,
        )
    finally:
        for h in handles:
            h.remove()
    return results


def report_exportability(
    mod: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    *,
    strict: bool = True,
    pre_dispatch: bool = False,
) -> dict[str, Exception | None]:
    """
    Report exportability issues for a module in one-shot.

    Args:
        mod: root module.
        args: args to the root module.
        kwargs: kwargs to the root module.
    Returns:
        A dict that maps from submodule name to the exception that was raised when trying to export it.
        `None` means the module is exportable without issue.
    Sample output:
        {
            '': UnsupportedOperatorException(func=<OpOverload(op='testlib.op_missing_meta', overload='default')>),
            'submod_1': UnsupportedOperatorException(func=<OpOverload(op='testlib.op_missing_meta', overload='default')>),
            'submod_2': None
        }
    """

    log_export_usage(event="export.report_exportability")

    kwargs = kwargs or {}

    all_submod_names = [name for name, _ in mod.named_modules() if name != ""]
    submod_inputs = _generate_inputs_for_submodules(mod, all_submod_names, args, kwargs)

    tried_module_types = set()
    report: dict[str, Exception | None] = {}

    def try_export(module, module_name, args, kwargs):
        nonlocal submod_inputs, report, strict, pre_dispatch, tried_module_types

        if type(module) in tried_module_types:
            return
        tried_module_types.add(type(module))

        if args is not None or kwargs is not None:
            try:
                torch.export._trace._export(
                    module,
                    args,
                    kwargs,
                    strict=strict,
                    pre_dispatch=pre_dispatch,
                )
                report[module_name] = None
                log.info("Successfully exported `%s`", module_name)
                return
            except Exception as e:
                short_msg = repr(e).split("\n")[0]
                log.warning(
                    "Failed exporting `%s` with exception: %s", module_name, short_msg
                )
                report[module_name] = e

        for name, submod in module.named_children():
            sub_module_name = name if module_name == "" else f"{module_name}.{name}"

            submod_args, submod_kwargs = submod_inputs.get(
                sub_module_name, (None, None)
            )

            try_export(submod, sub_module_name, submod_args, submod_kwargs)

        return

    try_export(mod, "", args, kwargs)

    unique_issues = set()
    for exception in report.values():
        if exception is not None:
            key = repr(exception).split("\\n")[0]
            unique_issues.add(key)

    log.warning("Found %d export issues:", len(unique_issues))
    for issue in unique_issues:
        log.warning(issue)

    return report
<|endoftext|># mypy: allow-untyped-defs
import inspect
import math
import operator
from collections.abc import Iterable
from typing import Any, final, TYPE_CHECKING

import torch
from torch._library.opaque_object import is_custom_class
from torch._ops import HigherOrderOperator, OpOverload
from torch._subclasses.fake_tensor import FakeTensor
from torch.export.graph_signature import (
    CustomObjArgument,
    InputKind,
    SymBoolArgument,
    SymFloatArgument,
    SymIntArgument,
    TensorArgument,
    TokenArgument,
)
from torch.fx import GraphModule


if TYPE_CHECKING:
    from torch.export.exported_program import ExportedProgram


class SpecViolationError(Exception):
    pass


def is_functional(op: OpOverload) -> bool:
    return not op._schema.is_mutable


def _check_has_fake_tensor(node: torch.fx.Node) -> None:
    # TODO(angelayi): remove this in favor of _check_val
    return _check_val(node)


def _check_val(node: torch.fx.Node) -> None:
    from torch.fx.experimental.symbolic_shapes import SymBool, SymFloat, SymInt

    def _check_correct_val(val):
        if val is None:
            return True
        elif isinstance(val, (int, bool, str, float)):
            return True
        elif isinstance(
            val, (torch.memory_format, torch.dtype, torch.device, torch.layout)
        ):
            return True
        elif isinstance(
            val, (FakeTensor, torch.Tensor)
        ):  # TODO(zhxchen17) Remove Tensor.
            return True
        elif isinstance(val, (SymInt, SymFloat, SymBool)):
            return True
        elif isinstance(val, CustomObjArgument):
            return True
        elif isinstance(val, Iterable):
            return all(_check_correct_val(x) for x in val)
        elif is_custom_class(type(val)):
            return True
        return False

    def _no_returns(op):
        if not isinstance(op, OpOverload):
            return False
        return len(op._schema.returns) == 0

    if "val" not in node.meta:
        if node.op == "call_function" and _no_returns(node.target):
            return
        raise SpecViolationError(f"Node.meta {node.name} is missing val field.")

    val = node.meta["val"]
    if not _check_correct_val(val):
        raise SpecViolationError(f"Node.meta {node.name} has invalid val field {val}")


def _check_torch_fn(node: torch.fx.Node) -> None:
    torch_fn = node.meta.get("torch_fn")
    if torch_fn is None:
        raise SpecViolationError(
            f"Unable to find torch_fn metadata for node {node.name}"
        )
    if (
        not isinstance(torch_fn, tuple)
        and isinstance(torch_fn[0], str)
        and isinstance(torch_fn[1], str)
    ):
        raise SpecViolationError(
            f"Node.meta {node.name} has invalid torch_fn field {torch_fn}"
        )


class _VerifierMeta(type):
    _registry: dict[str, type["Verifier"]] = {}

    def __new__(metacls, name, bases, attrs):
        if bases:
            if "check" in attrs or "_check_graph_module" in attrs:
                raise SyntaxError("Overriding method check is not allowed.")
            if "dialect" not in attrs or attrs["dialect"] == "ATEN":
                raise AssertionError(
                    f"subclass must define dialect != 'ATEN', got {attrs.get('dialect')}"
                )
        else:
            if "check" not in attrs:
                raise AssertionError("base class must define 'check' method")
            if "_check_graph_module" not in attrs:
                raise AssertionError(
                    "base class must define '_check_graph_module' method"
                )
            if attrs["dialect"] != "ATEN":
                raise AssertionError(
                    f"base class dialect must be 'ATEN', got {attrs['dialect']}"
                )

        if not isinstance(attrs["dialect"], str):
            raise AssertionError(f"dialect must be str, got {type(attrs['dialect'])}")
        ret = type.__new__(metacls, name, bases, attrs)
        metacls._registry[attrs["dialect"]] = ret  # type: ignore[assignment]
        return ret


def getattr_recursive(obj: Any, target: str) -> Any:
    target_atoms = target.split(".")
    attr_itr = obj
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(
                f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}"
            )
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


class Verifier(metaclass=_VerifierMeta):
    dialect = "ATEN"

    def allowed_builtin_ops(self) -> list:
        return [
            operator.getitem,
            operator.add,
            operator.mul,
            operator.sub,
            operator.truediv,
            operator.ge,
            operator.le,
            operator.gt,
            operator.lt,
            operator.eq,
            operator.ne,
            operator.floordiv,
            operator.mod,
            operator.and_,
            operator.or_,
            operator.not_,
            operator.pow,
            operator.neg,
            operator.abs,
            operator.lshift,
            operator.rshift,
            math.ceil,
            math.floor,
            math.trunc,
            round,
        ]

    def allowed_op_types(self) -> tuple[type[Any], ...]:
        return (OpOverload, HigherOrderOperator)

    def allowed_getattr_types(self) -> tuple[type[Any], ...]:
        return (torch.fx.GraphModule, torch.utils._pytree.TreeSpec)

    def allowed_getattr_types_for_subgm(self) -> tuple[type[Any], ...]:
        # subgm in HOP's argument could have getattr(weight) nodes, thus stateful
        return (
            torch.fx.GraphModule,
            torch.nn.parameter.Parameter,
            torch.Tensor,  # for buffer and constant tensor
            torch.utils._pytree.TreeSpec,
        )

    def check_valid_op(self, op):
        pass

    def check_additional(self, gm: GraphModule) -> None:
        """
        Additional checks that are specific to some dialects.
        """

    @final
    def check(self, ep: "ExportedProgram") -> None:
        self._check_graph_module(ep.graph_module)
        _verify_exported_program_module_call_graph(ep)
        _verify_exported_program_signature(ep)

    @final
    def _check_graph_module(self, gm: torch.fx.GraphModule) -> None:
        def _allowed_getattr_types(is_toplevel_gm) -> tuple[type[Any], ...]:
            if is_toplevel_gm:
                ret = self.allowed_getattr_types()
            else:
                ret = self.allowed_getattr_types_for_subgm()
            if any(t is object for t in ret):
                raise AssertionError("allowed_getattr_types must not contain 'object'")
            return ret

        def _check_valid_op(op) -> None:
            def _allowed_builtin_ops() -> list:
                ret = self.allowed_builtin_ops()
                if not all(inspect.isbuiltin(op) for op in ret):
                    raise AssertionError("allowed_builtin_ops must all be builtins")
                return ret

            def _allowed_op_types() -> tuple[type[Any], ...]:
                ret = self.allowed_op_types()
                if any(t is object for t in ret):
                    raise AssertionError("allowed_op_types must not contain 'object'")
                return ret

            # TODO Remove this allowlist.
            _allowed_torch_functions = (
                torch.autograd.grad_mode.set_grad_enabled,
                torch.sym_int,
                torch.sym_float,
                torch.sym_ite,
                torch.sym_max,
                torch.sym_min,
                torch.sym_not,
                torch.sym_sqrt,
                torch.sym_sum,
                torch.export.custom_ops._call_custom_autograd_function_in_pre_dispatch,
                # TODO (tmanlaibaatar)
                # Predispatch export is able to contain autograd ops.
                # These will be modeled as HOO later
                torch._C._set_grad_enabled,
                torch.amp.autocast_mode._enter_autocast,
                torch.amp.autocast_mode._exit_autocast,
                torch.fx.experimental.symbolic_shapes.cast_symbool_to_symint_guardless,
                torch._functorch.predispatch._add_batch_dim,
                torch._functorch.predispatch._remove_batch_dim,
                torch._functorch.predispatch._vmap_increment_nesting,
                torch._functorch.predispatch._vmap_decrement_nesting,
                torch._functorch.predispatch.lazy_load_decompositions,
                torch._functorch.predispatch._make_dual,
                torch._functorch.predispatch._unpack_dual,
                torch._functorch.predispatch._jvp_increment_nesting,
                torch._functorch.predispatch._jvp_decrement_nesting,
                torch._functorch.predispatch._unwrap_for_grad,
                torch._functorch.predispatch._enter_dual_level,
                torch._functorch.predispatch._exit_dual_level,
            )

            if not isinstance(op, _allowed_op_types()):
                if (
                    op not in _allowed_builtin_ops()
                    and op not in _allowed_torch_functions
                ):
                    raise SpecViolationError(
                        f"Operator '{op}' is not an allowed operator type: {_allowed_op_types()}\n"
                        f"Valid builtin ops: {_allowed_builtin_ops()}"
                        f"Valid torch functions: {_allowed_torch_functions}"
                    )

            if isinstance(op, OpOverload):
                # All ops functional
                # TODO (tmanlaibaatar) more proper way is needed here
                if self.dialect != "TRAINING" and not is_functional(op):
                    raise SpecViolationError(f"operator '{op}' is not functional")
            self.check_valid_op(op)

        for mod in gm.modules():
            is_toplevel_gm = mod is gm

            if not isinstance(mod, torch.fx.GraphModule):
                continue

            mod.graph.lint()
            for node in mod.graph.nodes:
                # TODO(T140410192): should have fake tensor for all dialects
                if node.op in {"call_module", "call_method"}:
                    raise SpecViolationError(
                        f"call_module is not valid: got a class '{node.target}' ",
                    )

                elif node.op == "call_function":
                    _check_val(node)

                    _check_valid_op(node.target)

                elif node.op == "get_attr":
                    if not isinstance(node.target, str):
                        raise SpecViolationError(
                            f"Expected get_attr target to be string, but got {type(node.target)}"
                        )

                    attr = getattr_recursive(mod, node.target)
                    if isinstance(attr, torch.nn.Module):

                        def _is_type(name, ty):
                            return isinstance(getattr(attr, name, None), ty)

                        if type(attr).__name__ == "LoweredBackendModule":
                            if (
                                _is_type("backend_id", str)
                                and hasattr(attr, "original_module")
                                and hasattr(attr, "module_name")
                                and getattr(attr, "backend_id", None) == "aoti"
                            ):
                                continue
                            if (
                                _is_type("backend_id", str)
                                and _is_type("processed_bytes", bytes)
                                and _is_type("compile_specs", list)
                                and hasattr(attr, "original_module")
                            ):
                                continue
                            else:
                                backend_id = getattr(attr, "backend_id", None)
                                processed_bytes = getattr(attr, "processed_bytes", None)
                                compile_specs = getattr(attr, "compile_specs", None)
                                raise SpecViolationError(
                                    f"Invalid get_attr type {type(attr)}. \n"
                                    f"LoweredBackendModule fields: "
                                    f"backend_id(str) : {type(backend_id)}, "
                                    f"processed_bytes(bytes) : {type(processed_bytes)}, "
                                    f"compile_specs(list) : {type(compile_specs)}"
                                )
                        elif type(attr).__name__ == "AOTInductorEPModule":
                            continue

                        elif type(attr).__name__ == "AOTInductorRunnerWrapper":
                            continue

                    if not isinstance(attr, _allowed_getattr_types(is_toplevel_gm)):
                        raise SpecViolationError(
                            f"Invalid get_attr type {type(attr)} on target {node.target}. \n"
                            f"Valid get_attr types: {_allowed_getattr_types(is_toplevel_gm)}"
                        )

                elif node.op == "placeholder":
                    _check_val(node)
                # TODO(zhxchen17)
                # elif node.op == "output":
                #     _check_flattened_outputs()

        self.check_additional(gm)


class TrainingIRVerifier(Verifier):
    dialect = "TRAINING"


def _verify_exported_program_module_call_graph(exported_program) -> None:
    module_call_graph = exported_program.module_call_graph
    nodes = {node.name for node in exported_program.graph.nodes}
    for entry in module_call_graph:
        if entry.signature is not None:
            for arg in entry.signature.inputs:
                if arg.name and arg.name not in nodes:
                    raise SpecViolationError(
                        f"Input {arg.name} does not exist in the graph."
                    )
            for arg in entry.signature.outputs:
                if arg.name and arg.name not in nodes:
                    raise SpecViolationError(
                        f"Output {arg.name} does not exist in the graph."
                    )


def _verify_exported_program_signature(exported_program) -> None:
    # Check ExportedProgram signature matches
    gs = exported_program.graph_signature

    # Lazily computed on first use; avoids calling graph_module.state_dict()
    # unless at least one buffer is missing from the top-level state_dict.
    _gm_state_dict: dict | None = None

    # Check every node in the signature exists in the graph
    input_node_names = [
        node.name for node in exported_program.graph.nodes if node.op == "placeholder"
    ]

    if len(input_node_names) != len(gs.input_specs):
        input_spec_names = [
            spec.arg.name for spec in gs.input_specs if hasattr(spec.arg, "name")
        ]
        missing_in_specs = set(input_node_names) - set(input_spec_names)
        missing_in_graph = set(input_spec_names) - set(input_node_names)
        raise SpecViolationError(
            f"Number of graph inputs ({len(input_node_names)}) "
            f"does not match number of inputs in the graph signature ({len(gs.input_specs)})\n"
            f"Placeholders missing input_specs: {missing_in_specs}\n"
            f"Input_specs missing placeholders: {missing_in_graph}"
        )

    for input_spec, node in zip(gs.input_specs, input_node_names):
        if isinstance(
            input_spec.arg,
            (TensorArgument, SymIntArgument, SymFloatArgument, SymBoolArgument),
        ):
            if input_spec.arg.name != node:
                raise SpecViolationError(
                    f"Input spec name {input_spec.arg.name} does not match node name {node}"
                )

        if input_spec.kind == InputKind.USER_INPUT:
            continue

        elif input_spec.kind == InputKind.PARAMETER:
            if not isinstance(input_spec.arg, TensorArgument):
                raise SpecViolationError(
                    f"Parameter {input_spec.name} is not a tensor argument. Found {input_spec.arg} instead."
                )
            if input_spec.target is None:
                raise SpecViolationError(
                    f"InputSpec for {input_spec.name} has no target."
                )

            param = input_spec.target
            if param not in exported_program.state_dict:
                raise SpecViolationError(f"Parameter {param} is not in the state dict.")

            if not isinstance(exported_program.state_dict[param], torch.nn.Parameter):
                raise SpecViolationError(
                    f"State dict entry for parameter {param} is not an instance of torch.nn.Parameter."
                )

        elif input_spec.kind == InputKind.BUFFER:
            if not isinstance(input_spec.arg, TensorArgument):
                raise SpecViolationError(
                    f"Buffer {input_spec.name} is not a tensor argument. Found {input_spec.arg} instead."
                )
            if input_spec.target is None:
                raise SpecViolationError(
                    f"InputSpec for {input_spec.name} has no target."
                )

            buffer = input_spec.target
            if input_spec.persistent is None:
                raise SpecViolationError(
                    f"Buffer {buffer} is missing a persistence flag"
                )

            if (
                input_spec.persistent is True
                and buffer not in exported_program.state_dict
            ):
                # Allow buffers that live in constants or in a subgraph
                # submodule (e.g. lifted tensor constants from
                # invoke_subgraph tracing stored under repeated_subgraph0).
                # Use a lazy copy of the graph module state to avoid the
                # cost of state_dict() when no fallback is needed.
                if buffer not in exported_program.constants:
                    if _gm_state_dict is None:
                        _gm_state_dict = exported_program.graph_module.state_dict()
                    if buffer not in _gm_state_dict:
                        raise SpecViolationError(
                            f"Buffer {buffer} is not in the state dict."
                        )

            if input_spec.persistent is False and buffer in exported_program.state_dict:
                raise SpecViolationError(
                    f"Non-persistent buffer {buffer} is in the state dict, it should not be."
                )
        elif input_spec.kind == InputKind.CONSTANT_TENSOR:
            if not isinstance(input_spec.arg, TensorArgument):
                raise SpecViolationError(
                    f"Constant tensor {input_spec.name} is not a tensor argument. Found {input_spec.arg} instead."
                )
            if input_spec.target is None:
                raise SpecViolationError(
                    f"InputSpec for {input_spec.name} has no target."
                )

            tensor_const = input_spec.target
            if tensor_const not in exported_program.constants:
                raise SpecViolationError(
                    f"Constant tensor {tensor_const} is not in the constants dictionary."
                )
        elif input_spec.kind == InputKind.CUSTOM_OBJ:
            if not isinstance(input_spec.arg, CustomObjArgument):
                raise SpecViolationError(
                    f"Custom object {input_spec.name} is not a custom object argument. Found {input_spec.arg} instead."
                )
            if input_spec.target is None:
                raise SpecViolationError(
                    f"InputSpec for {input_spec.name} has no target."
                )

            custom_obj = input_spec.target
            if custom_obj not in exported_program.constants:
                raise SpecViolationError(
                    f"Custom object {custom_obj} is not in the constants dictionary."
                )
        elif input_spec.kind == InputKind.TOKEN:
            if not isinstance(input_spec.arg, TokenArgument):
                raise SpecViolationError(
                    f"Constant tensor {input_spec.name} is not a tensor argument. Found {input_spec.arg} instead."
                )
        else:
            raise SpecViolationError(f"Unknown InputKind {input_spec.kind}.")

    # Check outputs
    output_node = list(exported_program.graph.nodes)[-1]
    if output_node.op != "output":
        raise AssertionError(f"last node must be output, got {output_node.op}")
    output_nodes = [
        arg.name if isinstance(arg, torch.fx.Node) else arg
        for arg in output_node.args[0]
    ]

    if len(output_nodes) != len(gs.output_specs):
        output_spec_names = [
            spec.arg.name if hasattr(spec.arg, "name") else str(spec.arg)
            for spec in gs.output_specs
        ]
        missing_out_specs = set(output_nodes) - set(output_spec_names)
        missing_out_graph = set(output_spec_names) - set(output_nodes)
        raise SpecViolationError(
            f"Number of output nodes {len(output_nodes)} is different "
            f"Than the number of outputs specified by the graph signature: {len(gs.output_specs)}\n"
            f"Nodes missing output_specs: {missing_out_specs}\n"
            f"Output_specs missing nodes: {missing_out_graph}"
        )

    num_tokens = len(gs.output_tokens)
    buffers_to_mutate = gs.buffers_to_mutate
    parameters_to_mutate = gs.parameters_to_mutate
    user_inputs_to_mutate = gs.user_inputs_to_mutate
    end = (
        len(buffers_to_mutate)
        + len(parameters_to_mutate)
        + len(user_inputs_to_mutate)
        + num_tokens
    )
    mutate_nodes: list[str] = output_nodes[num_tokens:end]
    user_output_nodes = output_nodes[end : end + len(gs.user_outputs)]

    for mutation_node in mutate_nodes:
        if mutation_node in buffers_to_mutate:
            if buffers_to_mutate[mutation_node] not in gs.buffers:
                raise SpecViolationError(
                    f"Buffer output {mutation_node} does not point to a buffer that exists. \n"
                    f"Dict of buffers that are mutated, in order: {buffers_to_mutate} \n"
                    f"Buffer nodes available: {gs.buffers} \n"
                )
        elif mutation_node in parameters_to_mutate:
            if parameters_to_mutate[mutation_node] not in gs.parameters:
                raise SpecViolationError(
                    f"Parameter output {mutation_node} does not point to a parameter that exists. \n"
                    f"Dict of parameters that are mutated, in order: {parameters_to_mutate} \n"
                    f"Parameter nodes available: {gs.parameters} \n"
                )
        elif mutation_node in user_inputs_to_mutate:
            if user_inputs_to_mutate[mutation_node] not in gs.user_inputs:
                raise SpecViolationError(
                    f"User input output {mutation_node} does not point to a user input that exists. \n"
                    f"Dict of user inputs that are mutated, in order: {user_inputs_to_mutate} \n"
                    f"User input nodes available: {gs.user_inputs} \n"
                )
        else:
            raise SpecViolationError(
                f"Mutation node {mutation_node} is neither a buffer nor a user input. "
                f"Buffers to mutate: {buffers_to_mutate}, User inputs to mutate: {user_inputs_to_mutate}"
            )

    for user_output_node, user_output_name in zip(user_output_nodes, gs.user_outputs):
        if user_output_node != user_output_name:
            raise SpecViolationError(
                f"User output {user_output_node} is not in the correct "
                "order or is not found in the "
                f"exported program's user_output list: {gs.user_outputs}. "
            )


def load_verifier(dialect: str) -> type[Verifier]:
    if dialect == "ATEN" or dialect == "":
        return _VerifierMeta._registry.get(dialect, Verifier)
    return _VerifierMeta._registry[dialect]
<|endoftext|># mypy: allow-untyped-defs
import builtins
import logging
import operator
import typing
import warnings
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import Any

import torch
import torch.export._trace
from torch import _C
from torch._export.passes.replace_quantized_ops_with_standard_ops_pass import (
    replace_quantized_ops_with_standard_ops,
)
from torch.export.dynamic_shapes import _tree_map_with_path, Dim
from torch.export.exported_program import ExportedProgram
from torch.export.graph_signature import (
    ConstantArgument,
    CustomObjArgument,
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    TensorArgument,
)
from torch.fx import subgraph_rewriter


log = logging.getLogger(__name__)


def _get_param_count_list(method_graph, args_params):
    param_count_list = []
    for input_, arg_params_ in zip(method_graph.inputs(), args_params):
        if "PackedParams" in str(input_.type()):
            in_vars, _ = torch.jit._flatten(arg_params_)
            param_count_list.append(len(in_vars))
        else:
            param_count_list.append(arg_params_ is not None)

    return param_count_list


def _trace_and_get_graph_from_model(model, args):
    # A basic sanity check: make sure the state_dict keys are the same
    # before and after running the model.  Fail fast!
    orig_state_dict_keys = torch.jit._unique_state_dict(model).keys()

    # Disable Autocast cache because it replaces kernel's weight and bias
    # by (undesired) constants.
    # No perf impact for when there are reused weights since https://github.com/pytorch/pytorch/pull/85665
    prev_autocast_cache_enabled = torch.is_autocast_cache_enabled()
    torch.set_autocast_cache_enabled(False)
    trace_graph, torch_out, _inputs_states = torch.jit._get_trace_graph(
        model,
        args,
        strict=False,
        _force_outplace=False,
        _return_inputs_states=True,
    )
    torch.set_autocast_cache_enabled(prev_autocast_cache_enabled)

    if orig_state_dict_keys != torch.jit._unique_state_dict(model).keys():
        raise RuntimeError(
            "state_dict changed after running the tracer; "
            "something weird is happening in your model!"
        )

    return trace_graph, torch_out


def _create_jit_graph(
    model: torch.nn.Module | torch.jit.ScriptFunction, args: Sequence[Any]
) -> tuple[torch.Graph, list["_C.IValue"], Any, torch.ScriptModule | None]:
    if isinstance(model, (torch.jit.ScriptFunction, torch.jit.ScriptModule)):
        flattened_args = tuple(torch.jit._flatten(tuple(args))[0])
        torch_out = None

        if isinstance(model, torch.jit.ScriptModule):
            try:
                graph = model.forward.graph  # type: ignore[attr-defined]
            except AttributeError as e:
                raise RuntimeError("'forward' method must be a script method") from e
            _C._jit_pass_onnx_function_substitution(graph)
            freezed_module = _C._freeze_module(
                typing.cast(_C.ScriptModule, model._c), preserveParameters=True
            )
            module, params = _C._jit_onnx_list_model_parameters(freezed_module)
            method_graph = module._get_method("forward").graph
            args_params = tuple(args) + tuple(params)
            param_count_list = _get_param_count_list(method_graph, args_params)
            in_vars, _ = torch.jit._flatten(args_params)
            graph = _C._propagate_and_assign_input_shapes(
                method_graph, tuple(in_vars), param_count_list, False, False
            )
            return graph, params, torch_out, module

        # torch.jit.ScriptFunction
        params = []
        graph = model.graph
        _C._jit_pass_onnx_function_substitution(graph)
        param_count_list = _get_param_count_list(graph, args)
        graph = _C._propagate_and_assign_input_shapes(
            graph, flattened_args, param_count_list, False, False
        )
        return graph, params, torch_out, None

    graph, torch_out = _trace_and_get_graph_from_model(model, args)
    _C._jit_pass_onnx_lint(graph)
    state_dict = torch.jit._unique_state_dict(model)
    params = list(state_dict.values())
    graph_inputs = list(graph.inputs())
    user_input_num = len(graph_inputs) - len(state_dict)
    param_names = list(state_dict.keys())
    for i, inp in enumerate(graph_inputs):
        if i >= user_input_num:
            inp.setDebugName(param_names[i - user_input_num])
    _C._jit_pass_onnx_function_substitution(graph)
    return graph, params, torch_out, None


def list_add(a, b):
    return a + b


def list_append(container, element):
    return container + [element]


def execute_subgraph_from_prim_loop(
    subgraph, iter_idx, len_loop_local_arguments, *args, **kwargs
):
    """
    subgraph: GraphModule from sub-block.
    iter_idx: The index of interaction.
    len_loop_local_arguments: The number of loop local arguments in args.
    """

    # Loop local variables. TS graph create those as inputs because their values
    # are updated inside the loop.
    loop_local_args = args[:len_loop_local_arguments]
    # Global variables that are not passed in as inputs to the loop sub-blocks
    # but are directly used. Most of time, their values are not updated, but
    # the only exception is when there are some operations that perform inplace
    # updates.
    global_args = args[len_loop_local_arguments:]
    return subgraph(*global_args, iter_idx, *loop_local_args, **kwargs)


def inplace_optimize_sym_size_div(gm: torch.fx.GraphModule):
    def pattern(im, dim, scale):
        sym_size_int = torch.ops.aten.sym_size.int(im, dim)
        scalar_tensor = torch.ops.aten.scalar_tensor(sym_size_int)
        div_scalar_mode = torch.ops.aten.div.Scalar_mode(
            scalar_tensor, scale, rounding_mode="trunc"
        )
        int_tensor = torch.ops.aten.Int.Tensor(div_scalar_mode)
        return int_tensor

    def replacement(im, dim, scale):
        sym_size_int = torch.ops.aten.sym_size.int(im, dim)
        return sym_size_int // scale

    subgraph_rewriter.replace_pattern(gm, pattern, replacement)


def is_valid_for_codegen(name):
    if len(name) == 0:
        raise RuntimeError("Empty argument name for codegen")
    if name[0].isdigit():
        return False
    return True


def normalize_name(name: str, prefix: str = "rename") -> str:
    name = name.replace(".", "_")
    if is_valid_for_codegen(name):
        return name
    return f"{prefix}_{name}"


def ir_name_to_func_name(name: str) -> str:
    """prim::If -> convert_prim_If"""
    name_list = name.split("::")
    return "convert_" + "_".join(name_list)


def get_node_as_placeholder_or_get_attr(fx_graph, name, is_top_level_graph):
    if is_top_level_graph:
        return fx_graph.get_attr(name)
    return fx_graph.placeholder(name)


_TORCH_DTYPE_TO_ENUM = {
    torch.uint8: 0,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 3,
    torch.int64: 4,
    torch.float16: 5,
    torch.float32: 6,
    torch.float64: 7,
    torch.complex32: 8,
    torch.complex64: 9,
    torch.complex128: 10,
    torch.bool: 11,
    torch.qint8: 12,
    torch.quint8: 13,
    torch.bfloat16: 15,
}

_TORCH_ENUM_TO_DTYPE = {value: key for key, value in _TORCH_DTYPE_TO_ENUM.items()}


def get_dtype_as_int(tensor):
    """
    prim::dtype has the signature "Tensor a) -> int", where it gets the dtype of
    the tensor and returns the integer corresponding to this dtype based on the
    enum in ScalarType.h
    """
    dtype = tensor.dtype
    if dtype not in _TORCH_DTYPE_TO_ENUM:
        raise RuntimeError(f"Unsupported dtype {dtype}")
    return _TORCH_DTYPE_TO_ENUM[dtype]


# Those operators will be automatically populated to an instance method
# of TS2FXGraphConverter with name convert_<namespace>_<opname>().
# Please check __init__ for method population implementations.
kind_to_standard_operators: dict[str, Callable[..., Any]] = {
    "prim::max": builtins.max,
    "prim::min": builtins.min,
    "prim::TupleIndex": operator.getitem,
    "aten::__is__": operator.is_,
    "aten::__isnot__": operator.is_not,
    "aten::__not__": operator.not_,
    "aten::__contains__": operator.contains,
    "prim::dtype": get_dtype_as_int,
    "aten::len": len,
    # Mapping from specialized op to its symbolic counterpart.
    # They currently do not have any other overrides.
    "aten::numel": torch.ops.aten.sym_numel,
    "aten::size": torch.ops.aten.sym_size,
    "aten::storage_offset": torch.ops.aten.sym_storage_offset,
    "aten::stride": torch.ops.aten.sym_stride,
}


def get_ir_value_parent_name_and_attr_name(node):
    irv_parent_name, irv_name = node.input().debugName(), node.output().debugName()
    attr_name = node.s("name")
    return irv_name, irv_parent_name, attr_name


def construct_fqn(ir, ref_map, name_map):
    name_list = []
    while ir in ref_map:
        name_list.append(name_map[ir])
        ir = ref_map[ir]
    return ".".join(reversed(name_list))


def get_block_to_lifted_attrs(
    graph: torch._C.Graph,
) -> tuple[dict[torch._C.Block, set[str]], dict[str, str]]:
    """
    Perform two passes to get a mapping of blocks to a set of FQNs of its lifted attributes.
    When a graph has control flow, the graph will be divided into multiple blocks. We want to convert
    each block to a graph which will be passed into torch.cond. A restriction for torch.cond is that model
    parameters/buffers are expected to be lifted as inputs to the subgraphs. Before converting the model,
    we will run this pass which will:
        1. Figure out which params/buffers are used within blocks through tracing the GetAttr calls.
        2. Process the graph bottom up to find the lifted attributes of each block by taking the union
        of the attributes used in the current block, and the lifted attributes of all its child blocks.

    Returns:
        A mapping of blocks to a set of FQNs of its lifted attributes, and a
        mapping of node names to the FQNs of its lifted attributes.
    """

    # A map from a block to its expected to be lifted arguments.
    blocks_to_lifted_attrs: dict[torch._C.Block, set[str]] = {}

    # Reference map stores the input (i.e., src) and output (i.e., dest) IR of a
    # GetAttr node. By traversing this reference map, we can figure out the
    # full IR aliasing pass and figure out the FQN of an attribute.
    # E.g., %2 = GetAttr(linear)[%1] --> node_to_parent_map["%2"] = "%1"
    node_to_parent_map: dict[str, str] = {}

    # Used for reconstructing the FQN of an attribute based on the reference map.
    # In nutshell, for each GetAttr call, GetAttr(input IR, attribute name) -> output IR
    # This name map stores which attribute name is called for a src IR --> dest IR action.
    # E.g., %2 = GetAttr(linear)[%1] --> node_to_attr_name["%2"] = "linear"
    node_to_attr_name: dict[str, str] = {}

    def _dfs_get_attr_dependency(entry):
        """
        First DFS path to construct reference map and name map.
        """
        for node in entry.nodes():
            if node.kind() == "prim::GetAttr":
                (
                    irv_name,
                    irv_parent_name,
                    attr_name,
                ) = get_ir_value_parent_name_and_attr_name(node)
                node_to_parent_map[irv_name] = irv_parent_name
                node_to_attr_name[irv_name] = attr_name
            for block in node.blocks():
                _dfs_get_attr_dependency(block)

    def _map_blocks_to_lifted_attrs(entry):
        """
        Walk the graph in a bottom-up fashion to build the expected to be
        lifted arguments for each block.
        """
        arguments: set[str] = set()
        for node in entry.nodes():
            for block in node.blocks():
                # Recursively build.
                arguments = arguments.union(_map_blocks_to_lifted_attrs(block))
            if node.kind() == "prim::GetAttr":
                irv_name = node.output().debugName()
                # Skip for intermediate GetAttr, which will anyway not result a FQN.
                # E.g., node_to_parent_name: {"%3": "%2", "%2": "%1"}
                #       node_to_attr_name: {"%3": "weight", "%2": "linear", "%1": "self"}
                #       There is only one FQN %3-->%2-->%1: self.linear.weight
                #       %2-->%1 is not a FQN: self.linear
                if irv_name not in set(node_to_parent_map.values()):
                    arguments.add(
                        construct_fqn(irv_name, node_to_parent_map, node_to_attr_name)
                    )
        if not isinstance(entry, torch._C.Graph):  # Skip the top level.
            blocks_to_lifted_attrs[entry] = arguments
        return arguments

    _dfs_get_attr_dependency(graph)
    _map_blocks_to_lifted_attrs(graph)

    return blocks_to_lifted_attrs, node_to_attr_name


def get_attribute_fqn_from_ts_node(
    name_to_attribute_fqn: dict[str, str], node: torch._C.Node
) -> str:
    def get_attr(name: str):
        if name in name_to_attribute_fqn:
            return name_to_attribute_fqn[name]
        else:
            raise ValueError(f"Attribute {name} not found")

    if node.kind() == "prim::SetAttr":
        input_name = next(node.inputs()).debugName()
    elif node.kind() == "prim::GetAttr":
        input_name = node.input().debugName()
    else:
        raise RuntimeError(
            f"Unexpected node kind when getting attribute fqn. node: {node} "
        )

    attr_name = node.s("name")
    root_attr_name = get_attr(input_name)
    attr_fqn = f"{root_attr_name}.{attr_name}" if root_attr_name else attr_name

    return attr_fqn


def get_op_overload(node: torch._C.Node):
    schema_str = node.schema()
    if schema_str == "(no schema)":
        raise AssertionError(f"got empty schema for {node}")
    schema: torch._C.FunctionSchema = torch._C.parse_schema(schema_str)
    ns, op_name = str(schema.name).split("::")
    override = schema.overload_name

    try:
        op_overload_mod = getattr(torch.ops, ns)
        op_overload_packet = getattr(op_overload_mod, op_name)
        if override:
            op_overload = getattr(op_overload_packet, override)
        else:
            op_overload = op_overload_packet.default
    except Exception as e:
        raise RuntimeError(
            f"Unable to find operator {node.kind()} with schema {node.schema()}"
        ) from e

    return op_overload


class TS2FXGraphConverter:
    def __init__(
        self,
        ts_graph: torch._C.Graph | torch._C.Block,
        name_to_param: dict[str, torch.Tensor],
        name_to_buffer: dict[str, torch.Tensor],
        blocks_to_lifted_attrs: dict[torch._C.Block, set[str]],
        name_to_non_tensor_attribute: dict[str, Any],
        name_to_constant: dict[str, Any],
        name_to_attribute_fqn: dict[str, str],
    ):
        self.ts_graph = ts_graph
        # Mapping of parameter FQN to actual parameter value
        self.name_to_param = name_to_param
        # Mapping of buffer FQN to actual buffer value
        self.name_to_buffer = name_to_buffer

        self.fx_graph: torch.fx.Graph = torch.fx.Graph()
        self.input_specs: list[InputSpec] = []
        self.output_specs: list[OutputSpec] = []

        # Mapping of TS node name to converted FX node
        self.name_to_node: dict[
            str, torch.fx.Node | list[torch.fx.Node] | dict[Any, torch.fx.Node]
        ] = {}
        # Mapping of TS node name to constant value (int, str, TorchBind obj,
        # tensor constants ...)
        self.name_to_constant: dict[str, Any] = name_to_constant

        # Mapping from torchscript node output name to attribute fully qualified name
        self.name_to_attribute_fqn: dict[str, str] = name_to_attribute_fqn

        # Mapping from fully qualified name to real values or a fx graph node
        # During convert, this represents the current value of a non-tensor attribute
        # One use case is:
        #   def forward(self, x):
        #        c1 = self.count
        #        self.count += 1
        #        c2 = self.count
        #        return x + c1 + c2
        self.name_to_non_tensor_attribute_node: dict[str, Any] = {}

        # Mapping from fully qualified name to initial real values inputs
        # We separate it from self.name_to_non_tensor_attribute_node since
        # we need initial real value input when we construct fx.GraphModule
        self.name_to_non_tensor_attribute: dict[str, Any] = name_to_non_tensor_attribute

        self.subgraphs: dict[str, torch.fx.GraphModule] = {}

        # Mapping of block to list of attributes that need to be lifted for each
        # block
        self.blocks_to_lifted_attrs = blocks_to_lifted_attrs

        # Populate methods for the standard operators.
        for k in kind_to_standard_operators:
            handler_func_name = ir_name_to_func_name(k)
            # Create an indirect function call:
            # convert_<namespace>_<opname> --> lambda node: _convert_standard_operator(node)
            setattr(
                self,
                handler_func_name,
                lambda node: self._convert_standard_operators(node),
            )

        # This stores a list of return results that do not appear in the original TS
        # graph's outputs. The reason we maintain this is because some operations in the sub-block
        # might have inplace updates to the variable defined in the parent fx graph. After
        # the execution of that sub-block, the variable defined in the parent fx graph also
        # needs to be updated.
        self.name_update_from_subblock_to_parent: set[str] = set()

    def _is_get_attr_node(self, fqn):
        return (
            fqn in self.name_to_buffer
            or fqn in self.name_to_param
            or (
                fqn in self.name_to_constant
                and isinstance(self.name_to_constant[fqn], torch.ScriptObject)
            )
        )

    def _convert_block_to_subgraph(self, node: torch._C.Node, arguments: list[str]):
        subgraph_nodes, subgraph_converters = [], []
        for block in node.blocks():
            subgraph_converter = TS2FXGraphConverter(
                block,
                self.name_to_param,
                self.name_to_buffer,
                self.blocks_to_lifted_attrs,
                {},
                self.name_to_constant,
                self.name_to_attribute_fqn,
            )

            for block_arg in arguments:
                normalized_block_arg_name = normalize_name(block_arg)
                placeholder_node = subgraph_converter.fx_graph.placeholder(
                    normalized_block_arg_name
                )
                subgraph_converter.name_to_node[block_arg] = placeholder_node

            subgraph = subgraph_converter.convert()
            subgraph_name = self.add_subgraph(subgraph)
            subgraph_nodes.append(self.fx_graph.get_attr(subgraph_name))
            subgraph_converters.append(subgraph_converter)
        return subgraph_nodes, subgraph_converters

    def _identify_inputs_as_arguments(self, entry):
        """
        Identify inputs from the innermost sub-block. This is needed
        for nested sub-blocks when the input is hidden in the nested sub-block.
        E.g., example IR of input is hidden in the nested sub-block.
        Graph[x.1]
        %1 = ...
            Block[]
                Block[x.1]
                    %2 = x.1 ...
        """
        arguments: set[str] = set()
        for block in entry.blocks():
            for block_node in block.nodes():
                for block_node_in in block_node.inputs():
                    if (
                        block_node_in.debugName() in self.name_to_node
                        and block_node_in.debugName() not in self.name_to_attribute_fqn
                    ):
                        arguments.add(block_node_in.debugName())
                arguments = arguments.union(
                    self._identify_inputs_as_arguments(block_node)
                )
        return arguments

    def is_top_level_graph(self):
        return isinstance(self.ts_graph, torch._C.Graph)

    def add_subgraph(self, subgraph) -> str:
        name = f"subgraph_{len(self.subgraphs)}"
        self.subgraphs[name] = subgraph
        return name

    def get_args_kwargs(self, node: torch._C.Node, schema):
        args = []
        kwargs = {}
        for input, schema_arg in zip(node.inputs(), schema.arguments):
            if schema_arg.kwarg_only:
                kwargs[schema_arg.name] = self.get_fx_value_by_ir_value(input)
            else:
                args.append(self.get_fx_value_by_ir_value(input))

        return tuple(args), kwargs

    def get_fx_value_by_ir_value(self, value: torch._C.Value):
        value_name = value.debugName()

        if value_name in self.name_to_node:
            input_node = self.name_to_node[value_name]
            return input_node
        elif value_name in self.name_to_constant:
            if isinstance(self.name_to_constant[value_name], torch.ScriptObject):
                return self.fx_graph.get_attr(value_name)
            return self.name_to_constant[value_name]
        elif value_name in self.name_to_attribute_fqn:
            return self.get_fx_value_by_fqn(self.name_to_attribute_fqn[value_name])
        else:
            raise ValueError(f"Input {value_name} not found")

    def get_fx_value_by_fqn(self, name):
        if name in self.name_to_node:
            fx_node = self.name_to_node[name]
        elif name in self.name_to_constant:
            fx_node = self.name_to_constant[name]
        elif name in self.name_to_non_tensor_attribute_node:
            fx_node = self.name_to_non_tensor_attribute_node[name]
        elif name in self.name_to_non_tensor_attribute:
            fx_node = self.name_to_non_tensor_attribute[name]
        else:
            raise ValueError(f"Attribute {name} not found")
        return fx_node

    def convert(self) -> torch.fx.GraphModule:
        self.convert_graph_inputs()

        for node in self.ts_graph.nodes():
            self.convert_node(node)

        self.convert_graph_outputs()

        # Pass parameter and buffer to the root for lookup.
        gm = torch.fx.GraphModule(
            {
                **self.subgraphs,
                **self.name_to_param,
                **self.name_to_buffer,
                **self.name_to_non_tensor_attribute,
                **self.name_to_constant,
            },
            self.fx_graph,
        )

        inplace_optimize_sym_size_div(gm)

        gm.graph.lint()

        return gm

    def convert_graph_inputs(self):
        for graph_input in self.ts_graph.inputs():
            name = graph_input.debugName()

            if name in self.name_to_param:
                normalized_name = normalize_name(name)
                self.input_specs.append(
                    InputSpec(
                        InputKind.PARAMETER,
                        arg=TensorArgument(name=normalized_name),
                        target=name,
                    )
                )
                fx_node = get_node_as_placeholder_or_get_attr(
                    self.fx_graph, name, self.is_top_level_graph()
                )
            elif name in self.name_to_buffer:
                normalized_name = normalize_name(name)
                self.input_specs.append(
                    InputSpec(
                        InputKind.BUFFER,
                        arg=TensorArgument(name=normalized_name),
                        target=name,
                        persistent=True,
                    )
                )
                fx_node = get_node_as_placeholder_or_get_attr(
                    self.fx_graph, name, self.is_top_level_graph()
                )
            elif name in self.name_to_constant:
                if not isinstance(self.name_to_constant[name], torch.ScriptObject):
                    raise AssertionError(
                        f"Input conversion only handles ScriptObject, got {type(self.name_to_constant[name])}"
                    )
                normalized_name = normalize_name(name)
                self.input_specs.append(
                    InputSpec(
                        InputKind.CUSTOM_OBJ,
                        arg=CustomObjArgument(
                            name=normalized_name, class_fqn=normalized_name
                        ),
                        target=name,
                        persistent=False,
                    )
                )
                fx_node = get_node_as_placeholder_or_get_attr(
                    self.fx_graph, name, self.is_top_level_graph()
                )
            elif isinstance(graph_input.type(), torch.ClassType):
                # Directly skip inputs that are ScriptObject but not used in the graph.
                continue
            else:
                normalized_name = normalize_name(name, prefix="input")
                self.input_specs.append(
                    InputSpec(
                        InputKind.USER_INPUT,
                        arg=TensorArgument(name=normalized_name),
                        target=name,
                    )
                )
                fx_node = self.fx_graph.placeholder(normalized_name)

            self.name_to_node[name] = fx_node

    def convert_aten_Float(self, node: torch._C.Node):
        def to_float_tensor(t):
            return t.to(dtype=torch.float).item()

        inp_list = [self.get_fx_value_by_ir_value(inp) for inp in node.inputs()]
        fx_node = self.fx_graph.call_function(
            to_float_tensor,
            tuple(inp_list),
        )
        self.name_to_node[node.output().debugName()] = fx_node

    def convert_aten_tensor(self, node: torch._C.Node):
        """aten::tensor creates a constant tensor ad-hoc --> GetAttr"""
        args, kwargs = self.get_args_kwargs(node, torch.ops.aten.tensor.default._schema)

        for k in kwargs:
            if k == "requires_grad":
                kwargs[k] = bool(kwargs[k])  # 0 -> False, 1 -> True

        to_tensor = (
            torch.tensor
            if all(isinstance(a, int) for a in args)
            else torch._refs.tensor
        )

        def target(*args, **kwargs):
            if "dtype" in kwargs and kwargs["dtype"] is not None:
                kwargs["dtype"] = _TORCH_ENUM_TO_DTYPE[kwargs["dtype"]]
            return to_tensor(*args, **kwargs)

        # def to_dynamic_tensor(*args, **kwargs):
        #     if "dtype" in kwargs and kwargs["dtype"] is not None:
        #         kwargs["dtype"] = _TORCH_ENUM_TO_DTYPE[kwargs["dtype"]]
        #     return torch._refs.tensor(*args, **kwargs)

        output_name = node.output().debugName()
        fx_node = self.fx_graph.call_function(target, args, kwargs)
        self.name_to_node[output_name] = fx_node

    def convert_aten_append(self, node: torch._C.Node):
        # special handle python list append: "aten::append.t(t[](a!) self, t(c -> *) el) -> t[](a!)"

        # inplace append to the list!! This is kinda crazy, as we are inplace mutating the list
        # This makes the converter "non-functional", and the result depends on the order of the nodes being converted
        # In a sense, the converter now becomes a stateful interpreter
        warnings.warn(
            "Converting aten::append.t, which is an inplace mutation of the list. "
            "This makes the converter non-functional: the result depends on the order of the append nodes being converted!",
            stacklevel=2,
        )

        args = tuple(self.get_fx_value_by_ir_value(inp) for inp in node.inputs())
        fx_node = self.fx_graph.call_function(list_append, args)
        self.name_to_node[node.output().debugName()] = fx_node

        # inplace mutate arg[0], which is the python list
        self.name_to_node[node.inputsAt(0).debugName()] = fx_node

        # Variables that need to be updated to parent module.
        if not self.is_top_level_graph() and args[0].op == "placeholder":
            self.name_update_from_subblock_to_parent.add(node.inputsAt(0).debugName())

    def convert_prim_Constant(self, node: torch._C.Node):
        name = node.output().debugName()

        value: Any = None
        if node.hasAttribute("value"):
            constant_kind = node.kindOf("value")
            if constant_kind == "i":
                value = node.i("value")
            elif constant_kind == "f":
                value = node.f("value")
            elif constant_kind == "s":
                value = node.s("value")
            elif constant_kind == "t":
                alias_name = (
                    f"lifted_tensor_{name}"  # Follow naming convention from EP tracing.
                )
                fx_node = self.fx_graph.get_attr(alias_name)
                self.name_to_node[name] = fx_node
                name, value = alias_name, node.t("value")
            elif constant_kind == "ival":
                value = node.ival("value")
            else:
                raise ValueError(f"Unsupported constant type: {node.kindOf('value')}")
        else:
            value = None

        self.name_to_constant[name] = value

    def convert_prim_CallMethod(self, node: torch._C.Node):
        inp_list = [self.get_fx_value_by_ir_value(inp) for inp in node.inputs()]
        fx_node = self.fx_graph.call_method(
            node.s("name"),
            tuple(inp_list),
        )
        self.name_to_node[node.output().debugName()] = fx_node

    def convert_prim_device(self, node: torch._C.Node):
        input_type = node.input().type()
        if input_type.isSubtypeOf(torch._C.TensorType.get()):
            device = input_type.device()  # type: ignore[attr-defined]
            output_name = node.output().debugName()
            self.name_to_constant[output_name] = device
        else:
            raise ValueError(f"Unsupported JitType ({input_type}) when get device")

    def convert_prim_GetAttr(self, node: torch._C.Node):
        # Build fully qualified name
        attr_fqn = get_attribute_fqn_from_ts_node(self.name_to_attribute_fqn, node)
        output_name = node.output().debugName()
        self.name_to_attribute_fqn[output_name] = attr_fqn

        if self.is_top_level_graph():
            if self._is_get_attr_node(attr_fqn):
                # We insert a get_attr node due to two reasons.
                # First, ts graph does not lift tensor constants as input nodes. So
                # tensor constants may be ignored by in convert_graph_inputs().
                # Second, attr_fqn may have been written to via SetAttr. Two
                # GetAttr may give different values.
                self.name_to_node[output_name] = self.fx_graph.get_attr(attr_fqn)
            else:
                if attr_fqn not in self.name_to_non_tensor_attribute_node:
                    self.name_to_non_tensor_attribute_node[attr_fqn] = (
                        self.name_to_non_tensor_attribute[attr_fqn]
                    )
                self.name_to_node[output_name] = self.name_to_non_tensor_attribute_node[
                    attr_fqn
                ]
        else:
            # Special support for if blocks which do not allow SetAttr TorchScript
            # node and get_attr FX Graph Node.
            if self._is_get_attr_node(attr_fqn):
                self.name_to_node[output_name] = self.name_to_node[attr_fqn]

    def convert_prim_SetAttr(self, node: torch._C.Node):
        attr_fqn = get_attribute_fqn_from_ts_node(self.name_to_attribute_fqn, node)
        attr_value = tuple(node.inputs())[1]
        ts_graph_tensor_input = self.get_fx_value_by_ir_value(attr_value)
        if self._is_get_attr_node(attr_fqn):
            fx_attr_node = self.fx_graph.get_attr(attr_fqn)
            self.fx_graph.call_function(
                torch.Tensor.copy_, (fx_attr_node, ts_graph_tensor_input)
            )
        else:
            self.name_to_non_tensor_attribute_node[attr_fqn] = ts_graph_tensor_input

    def convert_call_function_op(self, node: torch._C.Node):
        target = get_op_overload(node)

        args, kwargs = self.get_args_kwargs(node, target._schema)

        fx_node = self.fx_graph.call_function(target, args, kwargs)

        # TODO: convert sourceRange() into stack_trace
        # fx_node.meta["stack_trace"] = node.sourceRange()

        if node.outputsSize() == 1:
            output_name = node.output().debugName()
            self.name_to_node[output_name] = fx_node
        else:
            for i, outp in enumerate(node.outputs()):
                output_name = outp.debugName()
                next_fx_node = self.fx_graph.call_function(
                    operator.getitem, (fx_node, i)
                )
                self.name_to_node[output_name] = next_fx_node

    def convert_prim_TupleConstruct(self, node: torch._C.Node):
        self._convert_prim_iterator(node)

    def convert_prim_ListConstruct(self, node: torch._C.Node):
        self._convert_prim_iterator(node)

    def _convert_prim_iterator(self, node: torch._C.Node):
        output_list = [self.get_fx_value_by_ir_value(inp) for inp in node.inputs()]

        output_name = node.output().debugName()
        self.name_to_node[output_name] = output_list

    def convert_prim_DictConstruct(self, node: torch._C.Node):
        output_dict = {}
        k, v = None, None
        for i, inp in enumerate(node.inputs()):
            # We assume key value are stored in pair in the DictConstruct.
            # The first element is the key and the following is the value.
            if i % 2 == 0:
                k = self.get_fx_value_by_ir_value(inp)
            else:
                v = self.get_fx_value_by_ir_value(inp)
                if k is None or v is None:
                    raise AssertionError("DictConstruct has an empty key value pair.")
                output_dict[k] = v
                k, v = None, None

        if k is not None or v is not None:
            raise AssertionError(
                "DictConstruct has an odd number of elements (violating our assumption)."
            )

        output_name = node.output().debugName()
        self.name_to_node[output_name] = output_dict

    def convert_prim_ListUnpack(self, node: torch._C.Node):
        self._convert_prim_unpack_iterator(node)

    def convert_prim_TupleUnpack(self, node: torch._C.Node):
        self._convert_prim_unpack_iterator(node)

    def _convert_prim_unpack_iterator(self, node: torch._C.Node):
        # Single input and multiple outputs for unpacking.
        for i, outp in enumerate(node.outputs()):
            outp_name = outp.debugName()
            inp = self.get_fx_value_by_ir_value(node.input())
            fx_node = self.fx_graph.call_function(operator.getitem, (inp, i))
            self.name_to_node[outp_name] = fx_node

    def convert_aten_Int(self, node: torch._C.Node):
        # converts aten::Int as aten._to_copy + aten::_local_scalar_dense
        target = torch.ops.aten._to_copy.default
        args = tuple(self.get_fx_value_by_ir_value(input) for input in node.inputs())
        to_copy_node = self.fx_graph.call_function(target, args, {"dtype": torch.int32})

        fx_node = self.fx_graph.call_function(
            torch.ops.aten._local_scalar_dense.default, (to_copy_node,)
        )

        # TODO: convert sourceRange() into stack_trace
        # fx_node.meta["stack_trace"] = node.sourceRange()

        output_name = node.output().debugName()
        self.name_to_node[output_name] = fx_node

    def convert_prim_NumToTensor(self, node: torch._C.Node):
        # Converts prim::NumToTensor as aten.scalar_tensor.
        # prim::NumToTensor IRs are currently triggered by:
        # .size() https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/frontend/tracer.cpp#L950
        # .numel() https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/frontend/tracer.cpp#L971
        # For both of those APIs, torch.jit.trace implicitly sets the output tensor type
        # to be LongTensor.
        target = torch.ops.aten.scalar_tensor
        args = tuple(self.get_fx_value_by_ir_value(input) for input in node.inputs())

        fx_node = self.fx_graph.call_function(target, args, {"dtype": torch.long})
        output_name = node.output().debugName()
        self.name_to_node[output_name] = fx_node

    def convert_prim_CreateObject(self, node: torch._C.Node):
        output_name = node.output().debugName()
        self.name_to_attribute_fqn[output_name] = ""

    def convert_aten__convolution(self, node: torch._C.Node):
        # converts aten::_convolution as aten.convolution, since aten::_convolution
        # doesn't have a meta function
        target = torch.ops.aten.convolution.default
        args, kwargs = self.get_args_kwargs(node, target._schema)

        fx_node = self.fx_graph.call_function(target, args, kwargs)

        output_name = node.output().debugName()
        self.name_to_node[output_name] = fx_node

    def convert_aten_div(self, node: torch._C.Node):
        target = get_op_overload(node)
        schema = target._schema

        args, kwargs = self.get_args_kwargs(node, schema)

        # converts aten::div.Tensor_mode(x, tensor_constant)
        # as aten.div.Scalar_mode(x, tensor_constant.item())
        if schema.overload_name == "Tensor_mode":
            arg1_name = args[1].name
            if arg1_name in self.name_to_constant and isinstance(
                self.name_to_constant[arg1_name], torch.Tensor
            ):
                tensor_constant = self.name_to_constant[arg1_name]
                if tensor_constant.numel() == 1:
                    updated_args = list(args)
                    updated_args[1] = self.name_to_constant[arg1_name].item()

                    fx_node = self.fx_graph.call_function(
                        torch.ops.aten.div.Scalar_mode,
                        tuple(updated_args),
                        kwargs,
                    )

                    # TODO: convert sourceRange() into stack_trace
                    # fx_node.meta["stack_trace"] = node.sourceRange()

                    output_name = node.output().debugName()
                    self.name_to_node[output_name] = fx_node
                    return

        self.convert_call_function_op(node)

    def convert_aten___getitem__(self, node: torch._C.Node):
        input_container, index = tuple(
            self.get_fx_value_by_ir_value(input) for input in node.inputs()
        )
        fx_node = self.fx_graph.call_function(
            operator.getitem, (input_container, index)
        )
        output_name = node.output().debugName()
        self.name_to_node[output_name] = fx_node

    def convert_aten_to(self, node: torch._C.Node):
        target = get_op_overload(node)
        args, _kwargs = self.get_args_kwargs(node, target._schema)

        # special handle aten.to.dtype and aten.to.prim_dtype followed by inplace_mutation_op
        # coz aten.to + inplace_mutation_op pattern would trigger
        # "cannot mutate tensors with frozen storage" functionalization error.
        # To work around the issue, we override the copy to be True, so that the output
        # is for sure not an alias of input
        if target is torch.ops.aten.to.dtype or target is torch.ops.aten.to.prim_dtype:
            user_nodes = [use.user for use in node.output().uses()]
            user_targets = [
                get_op_overload(user_node)
                for user_node in user_nodes
                if user_node.schema() != "(no schema)"
            ]
            has_mutable_target = any(
                target._schema.is_mutable for target in user_targets
            )

            if has_mutable_target:
                if len(args) < 4:
                    raise AssertionError(f"expected at least 4 args, got {len(args)}")
                new_args = list(args)
                new_args[3] = True  # copy, override to True
                fx_node = self.fx_graph.call_function(
                    torch.ops.aten.to.dtype, tuple(new_args)
                )
                # temp hack to work around the issue https://github.com/pytorch/pytorch/issues/131679
                # When this issue is fixed, the clone node would be no longer needed
                clone_node = self.fx_graph.call_function(
                    torch.ops.aten.clone.default, (fx_node,)
                )
                output_name = node.output().debugName()
                self.name_to_node[output_name] = clone_node
                return

        self.convert_call_function_op(node)

    def convert_aten_add(self, node: torch._C.Node):
        if node.schema() == "(no schema)":
            if isinstance(node.inputsAt(0).type(), torch.ListType) and isinstance(
                node.inputsAt(1).type(), torch.ListType
            ):
                target = torch.ops.aten.add.t
            else:
                raise RuntimeError(f"unable to determined the target for {node}")
        else:
            target = get_op_overload(node)

        if target is torch.ops.aten.add.t:
            # special handle python list/tuple add: "aten::add.t(t[] a, t[] b) -> t[]" for
            # RuntimeError: aten::add() Expected a value of type 'List[t]' for argument 'a' but instead found type 'immutable_list'.
            args, _kwargs = self.get_args_kwargs(node, target._schema)
            output_name = node.output().debugName()
            self.name_to_node[output_name] = self.fx_graph.call_function(list_add, args)
        else:
            self.convert_call_function_op(node)

    def _check_prim_loop_support(self, node):
        inputs = list(node.inputs())

        # TODO: (1/N) stage.
        if inputs[0].debugName() not in self.name_to_constant:
            raise RuntimeError(
                "prim::Loop currently cannot run with dynamic value of number of iterations."
            )

        # Make sure the condition is not updated in the subblock.
        subblock = next(node.blocks())
        condition_output_name = next(subblock.outputs()).debugName()
        for node in subblock.nodes():
            if (
                node.outputsSize() == 1
                and node.output().debugName() == condition_output_name
            ):
                raise RuntimeError(
                    "prim::Loop currently cannot run with dynamic value of condition."
                )
            if node.outputsSize() >= 2:
                for outp in node.outputs():
                    if outp.debugName() == condition_output_name:
                        raise RuntimeError(
                            "prim::Loop currently cannot run with dynamic value of condition."
                        )

    def convert_prim_Loop(self, node: torch._C.Node):
        inputs = list(node.inputs())
        self._check_prim_loop_support(node)

        num_iterations = self.get_fx_value_by_ir_value(inputs[0])

        # Find inputs.
        loop_local_arguments = [inp.debugName() for inp in inputs[2:]]

        global_arguments = self._identify_inputs_as_arguments(node)

        # Lift parameters as inputs.
        for block in node.blocks():
            global_arguments = global_arguments.union(
                self.blocks_to_lifted_attrs[block]
            )

        global_arguments = list(global_arguments)

        subgraph_nodes, subgraph_converters = self._convert_block_to_subgraph(
            node, global_arguments
        )

        if len(subgraph_nodes) != 1:
            raise AssertionError(f"expected 1 subgraph node, got {len(subgraph_nodes)}")
        subgraph_converter = subgraph_converters[0]
        if not self.is_top_level_graph():
            self.name_update_from_subblock_to_parent = (
                self.name_update_from_subblock_to_parent.union(
                    subgraph_converter.name_update_from_subblock_to_parent
                )
            )

        fx_block_args = [
            self.get_fx_value_by_fqn(name)
            for name in loop_local_arguments + global_arguments
        ]
        for iter_idx in range(num_iterations):
            loop_node = self.fx_graph.call_function(
                execute_subgraph_from_prim_loop,
                # Check execute_node function for the expected arguments order.
                (
                    subgraph_nodes[0],
                    iter_idx,
                    len(loop_local_arguments),
                    *fx_block_args,
                ),
                {},
            )

            # Update the value of loop local variables.
            if node.outputsSize() >= 1:
                for i, outp in enumerate(node.outputs()):
                    output_name = outp.debugName()
                    self.name_to_node[output_name] = self.fx_graph.call_function(
                        operator.getitem,
                        (
                            loop_node,
                            i + 1,
                        ),  # + 1 because the 0th element is the condition.
                    )
                    fx_block_args[i] = self.name_to_node[output_name]

            # Update the value of global variables, whose values are modified inplace.

            for i, name in enumerate(
                subgraph_converter.name_update_from_subblock_to_parent
            ):
                self.name_to_node[name] = self.fx_graph.call_function(
                    operator.getitem,
                    (
                        loop_node,
                        i + node.outputsSize() + 1,
                    ),  # + 1 because the 0th element is the condition.
                )
                global_argument_index = global_arguments.index(name)
                fx_block_args[i + node.outputsSize() + global_argument_index] = (
                    self.name_to_node[name]
                )

    def _check_set_attr_in_if_block(self, if_node: torch._C.Node):
        for block in if_node.blocks():
            for node in block.nodes():
                if node.kind() == "prim::SetAttr":
                    raise RuntimeError(
                        "During converting prim::If to torch.cond, found prim::SetAttr op"
                        " which is not supported yet. Please file an issue if you come "
                        "across this error."
                    )

    def convert_prim_If(self, node: torch._C.Node):
        self._check_set_attr_in_if_block(node)

        inputs = list(node.inputs())
        if len(inputs) != 1:
            raise AssertionError(f"expected 1 input for prim::If, got {len(inputs)}")
        predicate = self.get_fx_value_by_ir_value(inputs[0])

        # Find inputs.
        arguments = self._identify_inputs_as_arguments(node)

        # Lift parameters as inputs.
        for block in node.blocks():
            arguments = arguments.union(self.blocks_to_lifted_attrs[block])

        arguments = list(arguments)
        subgraph_nodes, _ = self._convert_block_to_subgraph(node, arguments)

        if len(subgraph_nodes) != 2:
            raise AssertionError(
                f"expected 2 subgraph nodes, got {len(subgraph_nodes)}"
            )

        fx_block_args = [self.get_fx_value_by_fqn(name) for name in arguments]

        args = (
            predicate,
            subgraph_nodes[0],
            subgraph_nodes[1],
            tuple(fx_block_args),
        )

        cond_node = self.fx_graph.call_function(torch.cond, args, {})

        # prim::If can also have zero output.
        if node.outputsSize() == 1:
            output_name = node.output().debugName()
            self.name_to_node[output_name] = cond_node
        elif node.outputsSize() > 1:
            for i, output in enumerate(node.outputs()):
                output_name = output.debugName()
                getitem = self.fx_graph.call_function(operator.getitem, (cond_node, i))
                self.name_to_node[output_name] = getitem

    def convert_aten_Bool(self, node: torch._C.Node):
        self._convert_as_noop(node)

    def convert_prim_Enter(self, node: torch._C.Node):
        # export generally treats prim::Enter as noop
        # The only context manager export supports is aten::enable_grad.
        # Unfortunately, TorchScript does not support aten::enable_grad yet.
        # TODO: support aten::enable_grad in both TorchScript and Converter.
        return

    def convert_prim_Exit(self, node: torch._C.Node):
        # export treats prim::Exit as noop
        return

    def _convert_as_noop(self, node: torch._C.Node):
        # Converts the node as a no-op by mapping its output node as arg[0]

        target = get_op_overload(node)
        schema = target._schema

        args, _kwargs = self.get_args_kwargs(node, schema)

        output_name = node.output().debugName()
        self.name_to_node[output_name] = args[0]

    def convert_profiler__record_function_exit(self, node: torch._C.Node):
        # _record_function_exit has side effect so we keep it in fx.graph
        # currently, _record_function_enter_new and _record_function_exit are
        # discarded during `retrace_as_exported_program`.
        target = torch.ops.profiler._record_function_exit
        args = tuple(self.get_fx_value_by_ir_value(input) for input in node.inputs())
        self.fx_graph.call_function(target, args)

    def convert_prim_tolist(self, node: torch._C.Node):
        # prim::tolist cannot be supported by `_convert_standard_operators`
        # since it requires call_method instead of call_function.
        target = "tolist"
        args = (self.get_fx_value_by_ir_value(next(node.inputs())),)
        fx_node = self.fx_graph.call_method(target, args)
        output_name = node.output().debugName()
        self.name_to_node[output_name] = fx_node

    def convert_prim_Uninitialized(self, node: torch._C.Node):
        # `prim::Uninitialized` is inserted by the compiler when it can prove
        # the value will never be used. It can be introduced by exceptions,
        # breaks, continues, and returns.
        # So we add a dummy constant to the graph.
        output_name = node.output().debugName()
        self.name_to_constant[output_name] = torch.Tensor()

    def _convert_standard_operators(self, node: torch._C.Node):
        target = kind_to_standard_operators[node.kind()]
        args = tuple(self.get_fx_value_by_ir_value(input) for input in node.inputs())
        fx_node = self.fx_graph.call_function(target, args)
        output_name = node.output().debugName()
        self.name_to_node[output_name] = fx_node

    def convert_node(self, node: torch._C.Node):
        node_kind = node.kind()

        # Get handler based on namespace and operator name.
        # Provide a default node handler as well in case we don't find
        # matching converter for that.
        handler_func_name = ir_name_to_func_name(node_kind)
        handler_func = getattr(self, handler_func_name, self.convert_call_function_op)

        # str calls print function implemented in CPP. To avoid repeating
        # the entire logic here, we simply keep first line from node string (getting rid
        # of sub-blocks IR prints).
        node_str = "".join(str(node).split("\n")[:1])
        log.debug("[%s] converts [%s]", handler_func.__name__, node_str)
        try:
            handler_func(node)
        except Exception as e:
            raise RuntimeError(f"TS2EPConverter failed for node {node_kind}") from e

    def convert_graph_outputs(self):
        args = []
        outp_name_list = [outp.debugName() for outp in self.ts_graph.outputs()] + list(
            self.name_update_from_subblock_to_parent
        )
        for output_name in outp_name_list:
            if output_name in self.name_to_node:
                fx_node = self.name_to_node[output_name]
                # TODO: Revisit this later after HigherOrderOp design changes.
                # Currently, we cannot directly return input as output.
                if (
                    not self.is_top_level_graph()
                    and isinstance(fx_node, torch.fx.Node)
                    and fx_node.op == "placeholder"
                ):
                    fx_node = self.fx_graph.call_function(torch.clone, (fx_node,))
                args.append(fx_node)
                self.output_specs.append(
                    OutputSpec(
                        OutputKind.USER_OUTPUT,
                        arg=TensorArgument(name=output_name),
                        target=output_name,
                    )
                )
            elif output_name in self.name_to_constant:
                args.append(self.name_to_constant[output_name])
                self.output_specs.append(
                    OutputSpec(
                        OutputKind.USER_OUTPUT,
                        arg=ConstantArgument(
                            name=output_name, value=self.name_to_constant[output_name]
                        ),
                        target=output_name,
                    )
                )
            else:
                raise ValueError(f"Output {output_name} not found")

        if len(args) == 0:
            # Sub-block of prim::If can have zero output.
            self.fx_graph.output([])
        elif len(args) == 1:
            self.fx_graph.output(
                args[0]
            )  # Get rid of an extra list wrapped around final output.
        elif len(args) > 1:
            self.fx_graph.output(
                args
            )  # For prim::Loop and prim::If with multiple outputs.
        else:
            # Sub-block of prim::Loop can have multiple outputs.
            self.fx_graph.output(args)


class ExplainTS2FXGraphConverter(TS2FXGraphConverter):
    """
    Run TS2FXGraphConverter in an explain mode. It collects all failed operators conversions
    and provide that information to users. In order to collect all failed conversions, it
    also mocks some internal attributes (e.g., name_to_node).
    """

    class _DictMock(dict):
        def __init__(self, dict_data, mock_value):
            super().__init__(dict_data)
            self.mock_value = mock_value

        def __getitem__(self, key):
            # If the original dictionary has the key, return its value.
            # Otherwise, return the mock value.
            if not super().__contains__(key):
                return self.mock_value
            return super().__getitem__(key)

        def __contains__(self, key):
            return True

    def __init__(
        self,
        ts_graph: torch._C.Graph | torch._C.Block,
        name_to_param: dict[str, torch.Tensor],
        name_to_buffer: dict[str, torch.Tensor],
        blocks_to_lifted_attrs: dict[torch._C.Block, set[str]],
        name_to_non_tensor_attribute: dict[str, Any],
        name_to_constant: dict[str, Any],
        name_to_attribute_fqn: dict[str, str],
    ):
        super().__init__(
            ts_graph,
            name_to_param,
            name_to_buffer,
            blocks_to_lifted_attrs,
            name_to_non_tensor_attribute,
            name_to_constant,
            name_to_attribute_fqn,
        )

        # Data to keep track of unsupported nodes.
        self.unsupported_node_list: list[torch._C.Node] = []

        # Add mock to needed attributes.
        self.name_to_node = ExplainTS2FXGraphConverter._DictMock(
            self.name_to_node,
            # Dummy node.
            torch.fx.Node(
                None,  # type: ignore[arg-type]
                "mock",
                "call_function",
                lambda: None,
                (),
                {},
            ),
        )

    def explain(self):
        self.convert_graph_inputs()
        for node in self.ts_graph.nodes():
            self.convert_node(node)
        self.convert_graph_outputs()

    def convert_node(self, node):
        try:
            super().convert_node(node)
        except Exception:
            self.unsupported_node_list.append(node)


@contextmanager
def disable_logging(log):
    disabled = log.disabled
    log.disabled = True
    try:
        yield
    finally:
        log.disabled = disabled


class TS2EPConverter:
    # TorchScript model to ExportedProgram converter
    def __init__(
        self,
        ts_model: torch.jit.ScriptModule | torch.jit.ScriptFunction,
        sample_args: tuple[Any, ...],
        sample_kwargs: dict[str, Any] | None = None,
    ):
        self.ts_model = ts_model
        self.ts_graph, self.params, _, _ = _create_jit_graph(ts_model, sample_args)

        self.sample_args = sample_args
        self.sample_kwargs = sample_kwargs

        self.name_to_param: dict[str, torch.Tensor] = {}
        self.name_to_buffer: dict[str, torch.Tensor] = {}
        param_list = (
            list(self.ts_model.parameters())
            if not isinstance(self.ts_model, torch._C.ScriptFunction)
            else []
        )
        if not isinstance(self.ts_model, torch._C.ScriptFunction):
            for k, tensor in self.ts_model.state_dict().items():  # type: ignore[union-attr]
                # Check if tensor belongs to any parameter.
                if any(
                    (tensor == param).all()
                    for param in param_list
                    if tensor.shape == param.shape
                ):
                    self.name_to_param[k] = tensor
                else:
                    self.name_to_buffer[k] = tensor

        self.name_to_non_tensor_attributes: dict[str, Any] = {}
        self.name_to_constant: dict[str, Any] = {}

        self.lift_get_attr()

    def convert(self) -> ExportedProgram:
        log.info(
            """
TS2EPConverter logging starts from here.

INFO: (TORCH_LOGS="export" <cmd>)
    * Log TorchScript IR.

DEBUG: (TORCH_LOGS="+export" <cmd>), additionally
    * Log conversion IR by IR in a format of [<conversion handler name>] converts [<IR>].
        """
        )
        log.info("TorchScript graph\n\n%s\n", self.ts_graph)

        blocks_to_lifted_attrs, name_to_attribute_fqn = get_block_to_lifted_attrs(
            self.ts_graph
        )

        graph_converter = TS2FXGraphConverter(
            self.ts_graph,
            self.name_to_param,
            self.name_to_buffer,
            blocks_to_lifted_attrs,
            self.name_to_non_tensor_attributes,
            self.name_to_constant,
            name_to_attribute_fqn,
        )
        gm = graph_converter.convert()

        # Post-processing step to deal with quantized operators.
        replace_quantized_ops_with_standard_ops(gm)
        log.info("GraphModule: %s", gm.print_readable(print_output=False))

        ep = self.retrace_as_exported_program(
            gm,
            graph_converter.name_to_constant,
        )
        log.info("%s", ep)

        # Post-processing step to ensure ExportedProgram has the same state_dict as
        # the original TorchScript model. Throw warnings for additionally populated
        # state_dict entries.
        if not isinstance(self.ts_model, torch._C.ScriptFunction):
            for k, tensor in self.ts_model.state_dict().items():  # type: ignore[union-attr]
                if k not in ep.state_dict:
                    warnings.warn(
                        f"Manually populate {k} into state_dict ExportedProgram, but it is never used by the ExportedProgram.",
                        stacklevel=2,
                    )
                    ep.state_dict[k] = tensor

        return ep

    @disable_logging(log)
    def explain(self, print_output=True):
        blocks_to_lifted_attrs, name_to_attribute_fqn = get_block_to_lifted_attrs(
            self.ts_graph
        )

        graph_converter = ExplainTS2FXGraphConverter(
            self.ts_graph,
            self.name_to_param,
            self.name_to_buffer,
            blocks_to_lifted_attrs,
            self.name_to_non_tensor_attributes,
            self.name_to_constant,
            name_to_attribute_fqn,
        )
        graph_converter.explain()
        if len(graph_converter.unsupported_node_list) > 0:
            explain_str = "Unsupported nodes are found in the following list:"
            for i, n in enumerate(graph_converter.unsupported_node_list):
                node_str = "".join(str(n).split("\n")[:1])
                explain_str += f"\n\n    {i}. {n.kind()} [{node_str}]"
        else:
            explain_str = "Success!"
        if print_output:
            print(explain_str)
        return explain_str

    def retrace_as_exported_program(
        self,
        gm: torch.fx.GraphModule,
        name_to_constant: dict[str, Any],
    ):
        dynamic_shapes = _tree_map_with_path(
            lambda path, x: (
                [Dim.AUTO] * x.dim() if isinstance(x, torch.Tensor) else None
            ),
            self.sample_args,
        )

        # TODO: adjust input orders to match GraphSignature convention
        ep = torch.export._trace._export(
            gm,
            self.sample_args,
            dynamic_shapes=dynamic_shapes,
            strict=False,
            pre_dispatch=True,
        )

        # Post-processing to make sure the ExportedProgram states are correct.
        # Because during conversion, we set tensor constants as GetAttr,
        # retracing cannot recognize them as tensor constants but instead
        # treat them as buffers. We need to set them again here.
        ep._constants.update(
            {
                k: v
                for k, v in name_to_constant.items()
                if isinstance(v, (torch.Tensor, torch.ScriptObject))
            }
        )
        for k in name_to_constant:
            ep.state_dict.pop(k, None)

        for spec in ep.graph_signature.input_specs:
            # Mark as constant tensors for erroneously traced buffers.
            if spec.kind == InputKind.BUFFER and spec.target in name_to_constant:
                if not isinstance(name_to_constant[spec.target], torch.Tensor):
                    raise AssertionError(
                        f"{type(name_to_constant[spec.target])} has been erroneously marked as buffer"
                    )
                spec.kind = InputKind.CONSTANT_TENSOR
                spec.persistent = None
        ep.verifier().check(ep)

        return ep

    def lift_get_attr(self):
        # This function lifts multiple data types.

        #     1. Tensor constants attributes (e.g., self.data = torch.tensor([2,3]))
        #     to buffers. Currently, when there are tensor constants, export
        #     would error and ask users to register tensor constants as buffers.
        #     Since it is hard to manually do so for TorchScript models
        #     (e.g., source code is missing), this function automatically
        #     lifts tensor constants to be buffers.

        #     2. ScriptObbject to constant. It will then be converted to getattr in
        #     in the fx graph.
        #
        # This function should happen in TS2EPConverter instead of
        # TS2FXGraphConverter since it gets attributes from self.ts_model
        # which is not accessible in TS2FXGraphConverter. It is similar to where
        # we collect self.name_to_param and self.name_to_buffer.
        name_to_attribute_fqn: dict[str, str] = {}

        def get_attr(fqn: str):
            name = fqn.split(".")
            v = self.ts_model
            for n in name:
                v = getattr(v, n)
            return v

        def get_fqn(node: torch._C.Node):
            attr_name = node.s("name")
            input_name = node.input().debugName()
            root_attr_name = name_to_attribute_fqn[input_name]
            attr_fqn = f"{root_attr_name}.{attr_name}" if root_attr_name else attr_name
            return attr_fqn

        def _dfs_get_attr(block):
            for node in block.nodes():
                if node.kind() == "prim::CreateObject":
                    output_name = node.output().debugName()
                    name_to_attribute_fqn[output_name] = ""

                if node.kind() == "prim::GetAttr":
                    attr_fqn = get_fqn(node)
                    value = get_attr(attr_fqn)
                    output_name = node.output().debugName()
                    name_to_attribute_fqn[output_name] = attr_fqn
                    if isinstance(value, torch.Tensor):
                        if attr_fqn not in self.name_to_buffer:
                            # Lift tensor constants to be a buffer
                            self.name_to_buffer[attr_fqn] = value
                    elif isinstance(value, torch.ScriptObject):
                        if attr_fqn not in self.name_to_constant:
                            self.name_to_constant[attr_fqn] = value
                    else:
                        self.name_to_non_tensor_attributes[attr_fqn] = value

                for subblock in node.blocks():
                    _dfs_get_attr(subblock)

        _dfs_get_attr(self.ts_graph)
<|endoftext|>from enum import Enum


class ExportErrorType(Enum):
    # User providing invalid inputs to either tracer, or other public facing APIs
    INVALID_INPUT_TYPE = 1

    # User returning values from their models that we don't support.
    INVALID_OUTPUT_TYPE = 2

    # Generated IR does not conform to Export IR Specification.
    VIOLATION_OF_SPEC = 3

    # User's code contains types and functionalities we don't support.
    NOT_SUPPORTED = 4

    # User's code didn't provide necessary details for us to successfully trace and export.
    # For example, we use a lot of decorators and ask users to annotate their model.
    MISSING_PROPERTY = 5

    # User is using an API without proper initialization step.
    UNINITIALIZED = 6


def internal_assert(pred: bool, assert_msg: str) -> None:
    """
    This is exir's custom assert method. It internally just throws InternalError.
    Note that the sole purpose is to throw our own error while maintaining similar syntax
    as python assert.
    """

    if not pred:
        raise InternalError(assert_msg)


class InternalError(Exception):
    """
    Raised when an internal invariance is violated in EXIR stack.
    Should hint users to report a bug to dev and expose the original
    error message.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ExportError(Exception):
    """
    This type of exception is raised for errors that are directly caused by the user
    code. In general, user errors happen during model authoring, tracing, using our public
    facing APIs, and writing graph passes.
    """

    def __init__(self, error_code: ExportErrorType, message: str) -> None:
        prefix = f"[{error_code}]: "
        super().__init__(prefix + message)
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import contextlib
import copy
import operator
from typing import TYPE_CHECKING

import torch

from ..utils import node_replace_, nodes_map


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch._ops import HigherOrderOperator
    from torch.export.graph_signature import ExportGraphSignature


def _replace_with_hop_helper(
    node: torch.fx.Node,
    enter_block_node: torch.fx.Node,
    wrap_hoo: HigherOrderOperator,
) -> None:
    graph: torch.fx.Graph = node.graph
    if graph.owning_module is None:
        raise AssertionError("graph.owning_module must not be None")
    gm: torch.fx.GraphModule = graph.owning_module
    if not isinstance(node.target, str):
        raise AssertionError(f"expected str target, got {type(node.target)}")
    sub_gm = getattr(gm, node.target)

    def set_hoo_node_meta(call_func_node):
        call_func_node.meta["nn_module_stack"] = copy.copy(
            enter_block_node.meta.get("nn_module_stack", {})
        )
        call_func_node.meta["torch_fn"] = (
            f"{wrap_hoo.__name__}",
            # pyrefly: ignore [missing-attribute]
            f"{wrap_hoo.__class__.__name__}.{wrap_hoo.__name__}",
        )
        if isinstance(output_args, (tuple, list)):
            call_func_node.meta["val"] = tuple(arg.meta["val"] for arg in output_args)
        elif isinstance(output_args, torch.fx.Node):
            call_func_node.meta["val"] = (output_args.meta["val"],)

    with graph.inserting_before(node):
        get_attr_node = graph.get_attr(node.target)
        get_attr_node.meta["nn_module_stack"] = copy.copy(
            enter_block_node.meta.get("nn_module_stack", {})
        )
        output_node = next(iter(reversed(sub_gm.graph.nodes)), None)
        # Split_module pass intentionally doesn't add output node
        # if the graph doesn't return anything.
        # TODO (tmanlaibaatar) Figure out if this is right behaviour
        # for split_module
        if isinstance(output_node, torch.fx.Node) and output_node.op != "output":
            output_node = None
        if output_node is not None:
            if len(output_node.args) != 1:
                raise AssertionError(
                    f"expected 1 output arg, got {len(output_node.args)}"
                )
            output_args = output_node.args[0]
            enter_block_node_args = enter_block_node.args
            if isinstance(output_args, (tuple, list)):
                call_func_node = graph.call_function(
                    wrap_hoo,
                    (*enter_block_node_args, get_attr_node, *node.args),
                    {},
                )
                # Create the metadata
                set_hoo_node_meta(call_func_node)
                node_replace_(node, call_func_node)

                # Rename the name of getitem nodes to the actual name of its contents
                # for passing verifier and better readability, also propagate metadata
                for get_item_node in call_func_node.users:
                    idx: int = get_item_node.args[1]  # type: ignore[assignment]
                    output_node = output_args[idx]
                    get_item_node._rename(output_node.name)
                    get_item_node.meta = output_node.meta

            elif isinstance(output_args, torch.fx.Node):
                call_func_node = graph.create_node(
                    "call_function",
                    wrap_hoo,
                    (*enter_block_node_args, get_attr_node, *node.args),
                    {},
                    output_args.name,
                )
                # Modify the subgraph to output a singleton list.
                output_node.args = ((output_args,),)
                # Add in an extra `getitem(wrap_hoo, 0)` node to the toplevel graph.
                get_item_node = graph.create_node(
                    "call_function",
                    operator.getitem,
                    (call_func_node, 0),
                    {},
                )
                # Create the metadata
                get_item_node.meta = output_args.meta
                set_hoo_node_meta(call_func_node)
                node_replace_(node, get_item_node)
            else:
                raise NotImplementedError(
                    f"replace_with_hop_pass doesn't support output type {type(output_args)}"
                )
        else:
            # TODO (shangdiy): remove this line, since the export graph can be non-functional
            node.graph.erase_node(node)


def _sequential_split_and_maybe_inline_subgraphs_helper(
    new_gm: torch.fx.GraphModule,
    graph_signature: ExportGraphSignature | None,
    maybe_inline_or_replace_with_hop: Callable[[torch.fx.Node], None],
) -> tuple[torch.fx.GraphModule, ExportGraphSignature | None]:
    """
    Helper function for replacing graph nodes with higher order nodes.
    For each subgraph in `new_gm`, decides whether to construct a HOO subgraph, or inline the calls
    back into the parent graph module, depending on `maybe_inline_or_replace_with_hop`.
    """
    # new_gm is a new graph module that could have different output args names.
    # We need to fix the graph signature.
    replace_ctx = contextlib.nullcontext()
    new_signature = None
    if graph_signature is not None:
        # Cannot deep copy a real ScriptObject, which is referenced
        # in the FakeScriptObject. Copy should be good enough to guard
        # against accidental mutation to original graph_signature.
        new_signature = copy.copy(graph_signature)
        new_gm_out_node = next(reversed(new_gm.graph.find_nodes(op="output")))
        if new_gm_out_node.op != "output" or len(new_gm_out_node.args[0]) != len(
            new_signature.output_specs
        ):
            raise AssertionError(
                f"output node mismatch: {new_gm_out_node.op}, "
                f"{len(new_gm_out_node.args[0])} vs {len(new_signature.output_specs)}"
            )
        for arg_node, out_spec in zip(
            new_gm_out_node.args[0], new_signature.output_specs
        ):
            if arg_node is None:
                if out_spec.arg.value is not None:  # type: ignore[union-attr]
                    raise AssertionError(
                        f"expected None out_spec.arg.value, got {out_spec.arg.value}"  # type: ignore[union-attr]
                    )
            elif (
                isinstance(arg_node, torch.fx.Node)
                and out_spec.arg.name != arg_node.name
            ):
                out_spec.arg.name = arg_node.name

        replace_ctx = new_gm._set_replace_hook(new_signature.get_replace_hook())  # type: ignore[assignment]

    with replace_ctx:
        nodes_map(
            list(new_gm.graph.nodes),
            lambda node: (
                maybe_inline_or_replace_with_hop(node)
                if node.op == "call_module"
                else node
            ),
        )
    new_gm.recompile()
    new_gm.graph.lint()
    return new_gm, new_signature


def _replace_with_hop_pass_helper(
    gm: torch.fx.GraphModule,
    graph_signature: ExportGraphSignature | None,
    sequential_split_and_maybe_inline_subgraphs: Callable[
        [torch.fx.GraphModule, ExportGraphSignature | None],
        tuple[torch.fx.GraphModule, ExportGraphSignature | None],
    ],
) -> tuple[torch.fx.GraphModule, ExportGraphSignature | None]:
    """
    Split gm into sub-graph-modules using `sequential_split_and_maybe_inline_subgraphs`, and
    then recursively call itself on each of the submodules.
    """
    new_gm, new_signature = sequential_split_and_maybe_inline_subgraphs(
        gm, graph_signature
    )
    # recursively call
    for node in new_gm.graph.nodes:
        if node.op == "get_attr":
            subgm = getattr(new_gm, node.target)
            if not isinstance(subgm, torch.fx.GraphModule):
                continue
            new_subgm, _ = _replace_with_hop_pass_helper(
                subgm,
                None,
                sequential_split_and_maybe_inline_subgraphs,
            )
            setattr(new_gm, node.target, new_subgm)

    new_gm.recompile()
    new_gm.graph.lint()
    return new_gm, new_signature
<|endoftext|>import torch
from torch.fx.passes.infra.pass_base import PassBase, PassResult


class _RemoveRuntimeAssertionsPass(PassBase):
    """
    Remove runtime assertions inserted by the
    _AddRuntimeAssertionsForInlineConstraintsPass.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False
        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                if node.target in [
                    torch.ops.aten._assert_async.msg,
                    torch.ops.aten._assert_scalar.default,
                    torch.ops.aten.sym_constrain_range_for_size.default,
                    torch.ops.aten.sym_constrain_range.default,
                    torch.ops.aten._assert_tensor_metadata.default,
                ]:
                    assert_async_node = node
                    if len(assert_async_node.users) > 0:
                        continue
                    module.graph.erase_node(assert_async_node)
                    # the upstream scalar_tensor <- {le, ge} <- sym_size
                    # linear chain of nodes of nodes is removed by the
                    # downstream dead code elimination
                    modified = True

        # We don't necessarily want to run DCE here because it could affect
        # nodes that are in the module_call_graph attribute of the exported
        # program. We will leave it to the pass caller to call DCE.
        return PassResult(graph_module, modified)
<|endoftext|># mypy: allow-untyped-defs

import torch
from torch._export.error import InternalError
from torch._export.pass_base import _ExportPassBaseDeprecatedDoNotUse
from torch._ops import HigherOrderOperator, OpOverload


__all__ = ["ReplaceViewOpsWithViewCopyOpsPass"]


_NON_FUNCTIONAL_OPS_TO_FUNCTIONAL_OPS: dict[OpOverload, OpOverload] = {
    torch.ops.aten._unsafe_view.default: torch.ops.aten.view_copy.default,
}


def is_view_op(schema: torch._C.FunctionSchema) -> bool:
    if len(schema.arguments) == 0:
        return False
    alias_info = schema.arguments[0].alias_info
    return (alias_info is not None) and (not alias_info.is_write)


def get_view_copy_of_view_op(schema: torch._C.FunctionSchema) -> OpOverload | None:
    if is_view_op(schema) and schema.name.startswith("aten::"):
        view_op_name = schema.name.split("::")[1]
        view_op_overload = (
            schema.overload_name if schema.overload_name != "" else "default"
        )
        view_copy_op_name = view_op_name + "_copy"
        if not hasattr(torch.ops.aten, view_copy_op_name):
            raise InternalError(f"{schema.name} is missing a view_copy variant")

        view_copy_op_overload_packet = getattr(torch.ops.aten, view_copy_op_name)

        if not hasattr(view_copy_op_overload_packet, view_op_overload):
            raise InternalError(f"{schema.name} is missing a view_copy variant")

        return getattr(view_copy_op_overload_packet, view_op_overload)

    return None


class ReplaceViewOpsWithViewCopyOpsPass(_ExportPassBaseDeprecatedDoNotUse):
    """
    Our backend expects pure functional operators. For efficiency
    purposes, we keep view ops around while functionalizing the exported
    program. This pass replaces view ops with view copy ops for backends that
    need AOT memory planning.
    """

    def call_operator(self, op, args, kwargs, meta):
        if op in _NON_FUNCTIONAL_OPS_TO_FUNCTIONAL_OPS:
            return super().call_operator(
                (_NON_FUNCTIONAL_OPS_TO_FUNCTIONAL_OPS[op]), args, kwargs, meta
            )

        if isinstance(op, HigherOrderOperator):
            return super().call_operator(op, args, kwargs, meta)

        if view_copy_op := get_view_copy_of_view_op(op._schema):
            return super().call_operator(view_copy_op, args, kwargs, meta)

        return super().call_operator(op, args, kwargs, meta)
<|endoftext|>import copy

import torch
from torch._export.pass_base import (
    _ExportPassBaseDeprecatedDoNotUse,
    Argument,
    PassResult,
)
from torch._export.pass_infra.node_metadata import NodeMetadata
from torch._export.pass_infra.proxy_value import ProxyValue
from torch._ops import OpOverload


aten = torch.ops.aten

_NON_FUNCTIONAL_TO_FUNCTIONAL_SIDE_EFFECTFUL_FUNCS: dict[OpOverload, OpOverload] = {
    aten.sym_constrain_range.default: aten._functional_sym_constrain_range.default,
    aten._assert_async.msg: aten._functional_assert_async.msg,
}


class _FunctionalizeSideEffectfulOpsPass(_ExportPassBaseDeprecatedDoNotUse):
    """
    Functionalize ops with side effect in graph module by replacing the op with
    functional version of it. A new dependency token (`dep_token`) will be
    created and propagated through functional ops to output.
    For example:
    ```
    def f(x):
        sym_constrain_range(x.shape[0], min=1, max=3)
        return x.add(3)
    ```
    Will be transformed to:
    ```
    def f(x):
        dep_token0 = _make_dep_token()
        dep_token1 = _functional_sym_constrain_range(
            x.shape[0], min=1, max=3, dep_token=dep_token0
        )

        return x.add(3), dep_token1
    ```
    """

    def __init__(self) -> None:
        super().__init__()
        self._dep_token: ProxyValue | None = None
        self._next_dep_token_index: int | None = None

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        # Early return if no non-functional assertions.
        if not any(
            n.target in _NON_FUNCTIONAL_TO_FUNCTIONAL_SIDE_EFFECTFUL_FUNCS
            for n in graph_module.graph.nodes
        ):
            return PassResult(graph_module=graph_module, modified=False)

        gm = copy.deepcopy(graph_module)
        self._dep_token = None
        self._next_dep_token_index = None
        return super().call(gm)

    def call_operator(
        self,
        op: OpOverload,
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op not in _NON_FUNCTIONAL_TO_FUNCTIONAL_SIDE_EFFECTFUL_FUNCS:
            return super().call_operator(op, args, kwargs, meta)

        if self._dep_token is None:
            self._dep_token = super().call_operator(
                aten._make_dep_token,
                args=(),
                kwargs={},
                meta=self._create_dummy_node_metadata(),
            )
            self._dep_token.node.name = "dep_token0"
            self._next_dep_token_index = 1

        self._dep_token = super().call_operator(
            _NON_FUNCTIONAL_TO_FUNCTIONAL_SIDE_EFFECTFUL_FUNCS[op],
            args=args,
            kwargs={**kwargs, "dep_token": self._dep_token},
            meta=meta,
        )
        if self._next_dep_token_index is None:
            raise AssertionError("_next_dep_token_index must not be None")
        self._dep_token.node.name = f"dep_token{self._next_dep_token_index}"
        self._next_dep_token_index += 1

        return self._dep_token

    def output(self, results: list[Argument], meta: NodeMetadata) -> ProxyValue:
        if self._dep_token is None:
            raise AssertionError("_dep_token must not be None")

        return super().output(results=(*results, self._dep_token), meta=meta)  # type: ignore[arg-type]
<|endoftext|># mypy: allow-untyped-defs
import logging
import operator

import torch
import torch.export._trace
from torch._ops import OpOverload
from torch.ao.quantization.fx._decomposed import (
    dequantize_per_channel,
    dequantize_per_tensor,
    quantize_per_tensor,
)
from torch.ao.quantization.utils import calculate_qmin_qmax
from torch.fx.graph_module import _assign_attr


log = logging.getLogger(__name__)

# Those values will need to be carried over multiple operators.
_INPUT_Q_DTYPE: torch.dtype | torch.fx.Node | None = None
_SCALE: float | torch.fx.Node | None = None
_ZERO_POINT: float | torch.fx.Node | None = None


def int_to_valid_dtype(val: int) -> torch.dtype:
    from torch._export.converter import _TORCH_ENUM_TO_DTYPE  # No circular import.

    if isinstance(val, torch.dtype):
        return val
    dtype = _TORCH_ENUM_TO_DTYPE[val]
    if dtype == torch.quint8:
        return torch.uint8
    elif dtype == torch.qint8:
        return torch.int8
    return dtype


def fx_enum_to_dtype(gm: torch.fx.GraphModule, val: int) -> torch.fx.Node:
