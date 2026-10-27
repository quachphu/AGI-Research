                raise AssertionError(
                    "cannot set both static_shapes and symbolic_context"
                )
            shape_env = None
        return self.fake_tensor_converter.from_real_tensor(
            self,
            tensor,
            shape_env=shape_env,
            source=source,
            symbolic_context=symbolic_context,
            trace=trace,
        )


_StoragePointer = object


def _validate_symbolic_output_for_caching(
    state: _CacheKeyState, output: FakeTensor
) -> None:
    """
    Validate symbolic content in output and raise _BypassDispatchCache if
    caching should be bypassed.

    Args:
        state: Cache key state containing known symbols
        output: Output to validate
        proxy_mode_active: Whether PROXY dispatch mode is currently active

    Raises: _BypassDispatchCache: If output contains symbolic content that
        prevents caching

    Details:

    If our output contains any symbols that didn't appear in the input then we
    need to bypass. Usually this will be unbacked symbols which can't be
    properly reconstructed but there could be "weird" cases where backed symbols
    spontaneously appear (from non-input state)?

    If we're proxy (symbol) tracing and the output contains ANY symbols then we
    need to bypass. The problem is that ProxyTorchDispatchMode relies on SymNode
    object identity and being able to see the construction of SymNodes.

    We could improve the proxy tracing case in a few ways:

    1. If the output SymNodes are directly copied from inputs then this is
       actually fine - they're already tracked. This would probably be the
       biggest bang/buck.

    2. If the output (tensors) are all direct copies of the inputs then this is
       also fine - since they're inputs they must be tracked. We already compute
       this we just don't plumb it around enough.

    3. If the output SymNodes are already tracked by the proxy then this is also
       actually fine - they're properly tracked. This probably wouldn't be
       common since for most outputs we use torch.empty_strided() and recompute
       strides.

    4. We could use the proxy to track "how" the SymNodes were computed and when
       using the cache we could "replay" them properly to teach the proxy how to
       build them.
    """
    from torch.fx.experimental.symbolic_shapes import _iterate_exprs, _iterate_nodes

    is_tracing = torch.fx.experimental.proxy_tensor.get_proxy_mode() is not None
    if is_tracing:
        # Check for SymNode types in PROXY mode - this should bypass caching
        # regardless of whether symbols are known or not
        for _ in _iterate_nodes(output):
            raise _BypassDispatchCache("Proxy mode with SymNode output")
    else:
        # Check for unrepresented symbols in tensor expressions
        for s in _iterate_exprs(output):
            for symbol in s.free_symbols:
                if symbol not in state.known_symbols:
                    raise _BypassDispatchCache("unrepresented symbol in output")


# NB: returns fake tensors
def run_fallback_kernel(
    fake_mode: FakeTensorMode,
    func: OpOverload,
    flat_args: Sequence[object],
    args_spec: PyTree,
    orig_not_implemented_exception: RuntimeError,
) -> FakeTensor:
    # these should all be supported, just to be safe
    # avoid fallback for operators which inplace modify metadata
    # because the input fake tensors would be umodified
    if torch.Tag.inplace_view in func.tags:
        raise orig_not_implemented_exception

    # pyrefly: ignore [implicit-any]
    inp_impls = {}

    # Don't use in_kernel_invocation_manager(fake_mode) as we want to do
    # REAL compute (not with meta device)
    with no_dispatch():

        def to_real_tensor(e: T) -> T | Tensor:
            if fake_mode.is_our_fake(e):
                out = torch.zeros_like(e, device=e.fake_device)
                if e.is_sparse:
                    out._coalesced_(e.is_coalesced())
                inp_impls[id(out)] = e
                return out
            return e

        flat_args = [to_real_tensor(a) for a in flat_args]
        args, kwargs = pytree.tree_unflatten(flat_args, args_spec)

        r = func(*args, **kwargs)

    storages: set[_StoragePointer] = set()

    for e in flat_args:
        if isinstance(e, Tensor):
            if not is_sparse_any(e):
                storages.add(e._typed_storage()._cdata)

    # TODO: also check metadata change on inputs
    # proper aliasing/metadata relationship between outputs and inputs will
    # not be set up, bc of conversion to device, unless we can reuse an
    # input impl

    def map_out(e: T) -> T | FakeTensor:
        if id(e) not in inp_impls and (
            isinstance(e, Tensor)
            and not is_sparse_any(e)
            and e._typed_storage()._cdata in storages
        ):
            raise orig_not_implemented_exception

        if isinstance(e, Tensor):
            if id(e) in inp_impls:
                return inp_impls[id(e)]
            else:
                return fake_mode.fake_tensor_converter.from_real_tensor(fake_mode, e)
        else:
            return e

    return pytree.tree_map(map_out, r)


def _set_cache_key_for_shape_env(
    cache: dict[_DispatchCacheKey, _DispatchCacheEntry],
    key: _DispatchCacheKey,
    entry: _DispatchCacheEntry,
) -> None:
    key.strip_shape_env()
    cache[key] = entry


def _set_cache_key(
    cache: dict[_DispatchCacheKey, _DispatchCacheEntry],
    key: _DispatchCacheKey,
    entry: _DispatchCacheEntry,
) -> None:
    cache[key] = entry


# Just for use to allow copying a module to fake tensors,
# does not apply elsewhere
class FakeCopyMode(TorchFunctionMode):
    def __init__(self, fake_mode: FakeTensorMode) -> None:
        self.fake_mode = fake_mode

    def __torch_function__(
        self,
        func: OpOverload,
        types: Sequence[type],
        args: Sequence[object] = (),
        kwargs: Mapping[str, object] | None = None,
    ) -> FakeTensor:
        kwargs = kwargs if kwargs else {}

        # clone will get called in Parameter deepcopy
        if func is torch._C.TensorBase.clone:
            if not isinstance(args[0], Tensor):
                raise AssertionError(f"Expected Tensor, got {type(args[0])}")
            return func(
                self.fake_mode.from_tensor(args[0], static_shapes=True), **kwargs
            )
        elif func is Tensor.__deepcopy__:
            if len(args) != 2 or len(kwargs) != 0:
                raise AssertionError(
                    f"Expected 2 args and 0 kwargs for __deepcopy__, got {len(args)} args and {len(kwargs)} kwargs"
                )
            tensor = cast(Tensor, args[0])
            memo = cast(dict[int, FakeTensor], args[1])

            if id(tensor) in memo:
                return memo[id(tensor)]

            out = self.fake_mode.from_tensor(tensor, static_shapes=True)
            memo[id(tensor)] = out
            return out
        else:
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)


def _device_handler(args: Sequence[object]) -> torch.device:
    # NB: Don't use is_our_fake, just serve the fake information
    # as is.  Notice we don't use 'self'; we use args[0].fake_mode
    # because they may not be the same.  It would also be possible
    # to return NotImplemented here, in which case the FakeTensor
    # handler on args[0] would handle it, but we're being nice and
    # short-circuiting quickly.
    if len(args) != 1 or not isinstance(args[0], FakeTensor):  # noqa: ISINSTANCE_FAKE_TENSOR
        raise AssertionError(
            "Expected exactly one FakeTensor argument for _device_handler"
        )
    if args[0].fake_mode.in_kernel_invocation:
        return torch.device("meta")
    else:
        return args[0].fake_device


# [subclass inputs]
# Suppose we enable fake tensor mode.  This means that fake tensor
# mode will run first.  But what if we do an operation that
# involves a tensor subclass that will desugar into normal tensor
# operations?  Without returning NotImplemented, fake tensor mode will run first,
# decide that a conversion was made (since there was a non fake
# tensor argument), and report an error that converting non
# fake tensor is not supported.  What we actually wanted to happen
# was to give the subclass a chance to figure out what it wants to
# before erroring out. Returning NotImplemented here allows this.
def _check_for_subclass(flat_args: Sequence[object]) -> bool:
    return any(_check_for_subclass_arg(x) for x in flat_args)


def _check_for_subclass_arg(x: object) -> bool:
    if isinstance(x, FakeTensor):  # noqa: ISINSTANCE_FAKE_TENSOR
        return False
    if not isinstance(x, Tensor):
        return False

    x_type = type(x)
    if x_type is Tensor:
        return False

    # Use the concrete type instead of isinstance because wrapper subclasses
    # marked as Parameters (e.g. Parameter(TwoTensor(...))) still need subclass
    # dispatch. Only metadata-only Parameter subclasses that inherit Tensor's
    # disabled __torch_dispatch__ can follow the regular Parameter path.
    is_metadata_only_parameter_subclass = (
        issubclass(x_type, torch.nn.Parameter)
        and x_type.__torch_dispatch__ is Tensor.__torch_dispatch__
    )
    if is_metadata_only_parameter_subclass:
        return False

    return True


_DISPATCH_META_HANDLERS = {
    torch.ops.prim.device.default: _device_handler,
    torch.ops.aten.size.default: lambda args: tuple(
        int(s) for s in cast(Tensor, args[0]).size()
    ),
    torch.ops.aten.stride.default: lambda args: tuple(
        int(s) for s in cast(Tensor, args[0]).stride()
    ),
    torch.ops.aten.storage_offset.default: lambda args: int(
        cast(Tensor, args[0]).storage_offset()
    ),
}

_DISPATCH_HANDLE_DIRECTLY = ordered_set(
    torch.ops.aten.is_coalesced.default,
    torch.ops.aten.dense_dim.default,
    torch.ops.aten.sparse_dim.default,
    # _RecordFunction doesn't support __eq__ so make sure not to attempt to
    # cache it.
    torch.ops.profiler._record_function_exit._RecordFunction,
)

from torch._subclasses.fake_impls import (  # noqa: F401
    _device_not_kwarg_ops,
    _is_tensor_constructor,
    _like_tensor_constructors,
    contains_tensor_types,
    get_fast_op_impls,
    has_meta,
    maybe_to_dense_mkldnn,
    op_implementations_checks,
    stride_incorrect_op,
)


def evict_fake_tensor_cache_key(key: _DispatchCacheKey) -> None:
    if key in FakeTensorMode.cache:
        FakeTensorMode.cache.pop(key)


@atexit.register
def dump_cache_stats() -> None:
    log.info("FakeTensor cache stats:")
    log.info("  cache_hits: %s", FakeTensorMode.cache_hits)
    log.info("  cache_misses: %s", FakeTensorMode.cache_misses)
    bypasses = FakeTensorMode.cache_bypasses
    if bypasses:
        log.info("  cache_bypasses:")
        width = max(len(k) for k in bypasses)
        for k, v in sorted(bypasses.items(), key=lambda i: -i[1]):
            log.info("    %-*s %s", width + 1, f"{k}:", v)


def _infer_fake_from_real_tensor(
    mode: FakeTensorMode, op: torch._ops.OpOverload, real_out: torch.Tensor
) -> torch.Tensor:
    def unsupported(reason: str) -> None:
        raise RuntimeError(
            f"propagate_real_tensors: we cannot infer a Fake kernel "
            f"(meta kernel) for operator {op._name} because {reason}. "
            f"Please use torch.library.register_fake to add a Fake kernel."
        )

    if real_out.storage_offset() != 0:
        unsupported(
            f"a return has a non-zero storage offset {real_out.storage_offset()}"
        )

    # Since PT2 is rank specialized, there's no such thing as a symbolic
    # output rank. So we can assume the fake tensor has the same number of
    # dimensions as the real tensor output.
    #
    # We shouldn't assume the Fake sizes/strides are exactly what we see on
    # the real tensor output (perhaps we should give users a lever to toggle
    # this). This is because there's a good amount of operators that return
    # outputs with data-dependent output shape.
    # So we infer the output sizes to all be unbacked symints
    fake_shape = [
        torch._library.fake_impl.allocate_size(mode.shape_env)
        for _ in range(real_out.dim())
    ]

    # We infer what the strides are. We had a couple of options for this:
    # - assume the strides are computable from the sizes
    # - use new fresh unbacked symints in the strides
    #   This doesn't work that well (PT2 doesn't support unbacked symint strides well)
    # - use the real strides
    #   This can only be used if we assume the strides are static.
    # We went with the first option.
    fake_strides = [-1] * real_out.dim()
    strides = [(s, idx) for idx, s in enumerate(real_out.stride())]
    strides.sort(key=lambda x: (x[0], -x[1]))
    expected = 1
    fake_stride = expected
    for s, idx in strides:
        if s != expected:
            unsupported(
                f"a return was not dense in memory (sizes {real_out.shape} strides {real_out.stride()})"
            )
        fake_strides[idx] = fake_stride
        expected = expected * real_out.shape[idx]
        fake_stride = fake_stride * fake_shape[idx]

    with mode:
        return torch.empty_strided(
            fake_shape,
            fake_strides,
            device=real_out.device,
            dtype=real_out.dtype,
            layout=real_out.layout,
        )


def inferred_fake_kernel_from_real_out(
    mode: FakeTensorMode, op: torch._ops.OpOverload, real_out: Any
) -> Any:
    if mode.shape_env is None:
        raise AssertionError("mode.shape_env must not be None")

    # Only support operators that have all Tensor outputs
    # This is a general limitation on custom ops that we impose for PT2
    # to avoid baking non-symbolic float/int outputs into the graph.
    real_flat_out, spec = pytree.tree_flatten(real_out)
    if not all(isinstance(t, torch.Tensor) for t in real_flat_out):
        raise RuntimeError(
            f"propagate_real_tensors: we don't support operators that return "
            f"non-Tensors. Got {op._schema}"
        )

    fake_flat_out = [_infer_fake_from_real_tensor(mode, op, t) for t in real_flat_out]
    return pytree.tree_unflatten(fake_flat_out, spec)
<|endoftext|>from __future__ import annotations

import contextlib
import functools
import warnings
import weakref
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import Any, cast, TYPE_CHECKING
from typing_extensions import Self


if TYPE_CHECKING:
    import builtins
    from collections.abc import Callable, Generator, Sequence
    from types import TracebackType

    from torch._functorch.pyfunctorch import FunctionalizeInterpreter
    from torch._ops import OpOverload

import torch
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree
from torch._C import _functionalization_reapply_views_tls as _reapply_views
from torch._ops import _get_dispatch_mode_pre_dispatch, TorchBindOpOverload
from torch._subclasses.meta_utils import is_sparse_any
from torch.utils._python_dispatch import (
    _detect_infra_mode,
    _disable_infra_mode,
    autograd_would_have_decomposed,
    return_and_correct_aliasing,
    TorchDispatchMode,
)


not_implemented_log = torch._logging.getArtifactLogger(__name__, "not_implemented")


def _has_unrecognized_tensor_types(types: Sequence[type]) -> bool:
    unrecognized_types = [
        t
        for t in types
        if t not in (torch.Tensor, torch._subclasses.FakeTensor, FunctionalTensor)
    ]
    if unrecognized_types:
        not_implemented_log.debug(
            "FunctionalTensor unrecognized subclass(es): %s", unrecognized_types
        )
    return bool(unrecognized_types)


@functools.lru_cache(maxsize=512)
def _can_decompose_fast(
    func: OpOverload, export: bool, pre_dispatch: bool
) -> bool | None:
    """Fast path for _can_decompose that depends only on (func, export, pre_dispatch).

    Returns True/False for a definitive answer, or None to fall through
    to the slow path (autograd_would_have_decomposed).
    """
    if export and func is torch.ops.aten.dropout.default:
        return False

    from torch._decomp import _should_decompose_because_unsafe_op

    if _should_decompose_because_unsafe_op(func):
        return True

    alias_info_present = any(arg.alias_info for arg in func._schema.arguments)
    if alias_info_present or func._schema.is_mutable:
        return True

    if export:
        if pre_dispatch:
            if func.namespace not in ("aten", "prim") and func._can_decompose():
                warnings.warn(
                    f"At pre-dispatch tracing, we assume that any custom op marked with "
                    f"CompositeImplicitAutograd and have functional schema are safe to not decompose. "
                    f"Found {func} to be one such op.",
                    stacklevel=3,
                )
            return False
        return True

    return None


def _assert_functionalize_not_active(msg: str) -> None:
    is_included = torch._C._dispatch_tls_is_dispatch_key_included(
        torch._C.DispatchKey.Functionalize
    )
    is_excluded = torch._C._dispatch_tls_is_dispatch_key_excluded(
        torch._C.DispatchKey.Functionalize
    )
    if not is_excluded and is_included:
        raise AssertionError(msg)


# NOTE Some special handling for tensor conversion during export is needed.
# Normally, when tracing through the model with tensor.to(), the maybe-aliasing
# relationship between input and output tensors will be baked into the graph.
# For example, if we got a tensor with device cpu and call tensor.to("cpu"),
# it will become a no-op in the graph. For a whole graph capture, this is not
# sound so we need to do something different. Instead, in export we will try to
# preserve the tensor conversion by forcing a non-semantic-breaking aten::_to_copy
# operator to be traced in the graph, and subsequently banning mutations on all
# such converted tensors.
# In addition to patching .to() method call in functionalization, we will have to
# patch other similar methods like float() and cpu(), because they intentionally
# don't fall back to .to() methods, but have the same behavior as .to() according to
# pytorch document. https://pytorch.org/docs/stable/generated/torch.Tensor.float.html
# thus we simply force them to go through .to() call.
def _conversion_method_template(**extra_kwargs: Any) -> Callable[..., Any]:
    def _(self: FunctionalTensor, *args: Any, **kwargs: Any) -> Any:
        return self.to(*args, **{**kwargs, **extra_kwargs})

    return _


class FunctionalTensor(torch.Tensor):
    """
    Functional tensors represent tensors that will remove mutations
    from a program. If you perform a mutable operation on a functional tensor,
    it will re-dispatch to the functional variant of that operation.

    Historically, functionalization is implemented in C++ in the dispatcher.
    This class is a lightweight python shim around the C++ functionalization logic.

    FunctionalTensor is required to be used with a corresponding
    FunctionalTensorMode active, because it relies
    on using the mode for dispatch (which can properly handle factory functions).
    """

    elem: torch.Tensor
    # Indicates to our torch_dispatch dispatching infra that
    # this is an "infra" mode with lower dispatching precedence.
    _mode_key = torch._C._TorchDispatchModeKey.FUNCTIONAL

    # Note: The reason we add these extra keys to our FunctionalTensor subclass
    # is to mirror the behavior of C++ functionalization (we can choose to change this
    # later, as long as it doesn't break anything).
    # FunctionalTensorWrapper copies **all** dispatch keys from the inner tensor
    # to the wrapper, excluding functorch and python dispatch keys.
    # Here I'm trying to reuse the keyset the functorch wrapper subclasses copy,
    # except that they don't include ZeroTensor so I'm manually adding it in.
    _extra_dispatch_keys = torch._C._additional_keys_to_prop_for_wrapper_tensors.add(
        torch._C.DispatchKey.ZeroTensor
    )

    # These are all aten ops that correspond to metadata queries.
    # We want FunctionalTensor to be able to handle them directly.
    metadata_fns = frozenset(
        {
            torch.ops.aten.is_contiguous.default,
            torch.ops.aten.is_contiguous.memory_format,
            torch.ops.aten.is_strides_like_format.default,
            torch.ops.aten.is_non_overlapping_and_dense.default,
            torch.ops.aten.size.default,
            torch.ops.aten.sym_size.default,
            torch.ops.aten.stride.default,
            torch.ops.aten.sym_stride.default,
            torch.ops.aten.storage_offset.default,
            torch.ops.aten.sym_storage_offset.default,
            torch.ops.aten.numel.default,
            torch.ops.aten.sym_numel.default,
            torch.ops.aten.dim.default,
            torch.ops.prim.device.default,
        }
    )

    # Used by auto_functionalize to determine base of tensors during inference mode.
    _inference_mode_base: FunctionalTensor | None = None

    def __new__(cls, elem: torch.Tensor, mode: FunctionalTensorMode) -> Self:
        if not torch._is_functional_tensor(elem):
            raise AssertionError("elem must be a functional tensor")

        # In general, we'd like our functional tensor subclass to only be in charge of functionalization,
        # and defer to the inner subclass for all other functionality.
        # Example: If our inner tensor is a ZeroTensor, we would want to defer running the ZeroTensor fallback
        # until after we redispatch to our inner ZeroTensor.
        # However, there are a few keys that we need to mirror between the inner and outer tensors.
        #   Conjugate
        #   Negative
        # Why? These keys are used to test metadata queries, like `.is_conj()` and `.is_neg()`.
        # We **need** calls to is_conj() to return the same thing on the outer and inner tensors,
        # Because user code / framework code that branches like so needs to do the same thing
        # when it sees the outer FunctionalTensor:
        #     if (x.is_conj()) {
        #         return at::view_as_real(x.resolve_conj());
        #     } else {
        #         return at::view_as_real(x);
        #     }
        extra_dispatch_keys = (
            FunctionalTensor._extra_dispatch_keys & torch._C._dispatch_keys(elem)
        )

        out = torch.Tensor._make_wrapper_subclass(
            # TODO: right now, _make_wrapper_subclass's dynamic shape interaction is not great.
            # Calling the overload that has kwargs causes us to go down the first overload path,
            # which will **always** specialize sizes.
            # We should probably eventually fix this so that the first overload can just handle dynamic shapes.
            cls,
            elem.shape,  # sizes
            elem.stride() if not is_sparse_any(elem) else None,  # strides
            (
                elem.storage_offset() if not is_sparse_any(elem) else None
            ),  # storage_offset
            None,  # memory_format
            elem.dtype,  # dtype
            elem.layout,  # layout
            elem.device,  # device
            False,  # pin_memory
            elem.requires_grad,  # requires_grad
            None,  # dispatch_sizes_strides_policy
            False,  # dispatch_device
            False,  # dispatch_layout
            extra_dispatch_keys,  # _extra_dispatch_keys
        )
        torch._C._set_throw_on_mutable_data_ptr(out)
        out.elem = elem

        if (
            torch._export.config.enable_auto_functionalized_v2_for_export
            and torch.is_inference_mode_enabled()
            and torch._inductor.config.enable_auto_functionalized_v2
        ):
            storage = out.elem.untyped_storage()
            if out.is_base_tensor():
                out._inference_mode_base = None
                # This assumes that the FunctionalTensor.elem does not change its storage after this point.
                # Otherwise this would be invalid.
                mode._storage_to_base[storage] = out
            else:
                out._inference_mode_base = mode._storage_to_base.get(storage)
                if out._inference_mode_base is None:
                    mode._storage_to_base[storage] = out
        return out

    def __torch_dispatch__(  # type: ignore[override]
        self,
        func: OpOverload,
        types: Sequence[type],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if _has_unrecognized_tensor_types(types):
            return NotImplemented

        if kwargs is None:
            kwargs = {}
        # FunctionalTensor needs to plumb all metadata requests to the inner tensor.
        # In theory we don't have to do this - but if we want to service metadata requests here,
        # we need to carefully make sure all metadata is accurate (including metadata mutations)
        if func in FunctionalTensor.metadata_fns:
            # All metadata accesses should be plumbed to the inner tensor, that way we don't have to worry
            # about the problem of keeping metadata in sync between the wrapper and inner tensor.
            # This also alleviates us from having to manually handle metadata mutations on the wrapper.
            if len(kwargs) != 0:
                raise AssertionError("kwargs must be empty for metadata functions")
            if func in [
                torch.ops.aten.is_strides_like_format.default,
                torch.ops.aten.is_contiguous.memory_format,
            ]:
                if len(args) != 2 or not isinstance(args[0], FunctionalTensor):
                    raise AssertionError("Expected 2 args with FunctionalTensor first")
                return func(torch._from_functional_tensor(args[0].elem), args[1])
            if len(args) != 1 or not isinstance(args[0], FunctionalTensor):
                raise AssertionError("Expected 1 arg with FunctionalTensor")

            return func(torch._from_functional_tensor(args[0].elem))
        if func is torch.ops.aten.detach.default:
            # detach can reach this dispatch when no FunctionalTensorMode is active
            # (e.g. snapshot_fake under ProxyTorchDispatchMode). Returning self is safe:
            # autograd has already run above this layer, so detach has no effect here,
            # and we must avoid falling through to the "on its own" RuntimeError below.
            if len(args) == 1 and isinstance(args[0], FunctionalTensor):
                return args[0]
        # Originally I tried to implement my subclass without giving it a torch_dispatch, but I gave up:
        # - _make_wrapper_subclass requires a __torch_dispatch__
        # - If we want to use _make_subclass(), we have a problem: the subclass will share a TensorImpl with the inner tensor,
        #   which is of type FunctionalTensorWrapper! We explicitly do not want our wrapper to be a FunctionalTensorWrapper.
        # - If we use the default tensor.__new__(), we have another problem: it returns inner_tensor.alias(),
        #   which causes every subclass created above autograd to have autograd view metadata
        #   (in addition to also being a FunctionalTensorWrapper).
        raise RuntimeError(
            "Attempting to use FunctionalTensor on its own. Instead, please use it with a corresponding FunctionalTensorMode()"
        )

    def __repr__(self, *, tensor_contents: object | None = None) -> str:
        return f"FunctionalTensor({repr(self.elem)})"

    @staticmethod
    def to_functional(x: torch.Tensor) -> FunctionalTensor:
        # We will do the wrapping for the user.

        if torch._is_functional_tensor(x):
            raise AssertionError("x must not already be a functional tensor")
        # The only autograd metadata we care about on the FunctionalTensor is:
        # - requires_grad (so autograd runs)
        # - is_leaf (so that mutations on graph inputs that are not leaves are allowed by the autograd engine)
        #   this is handled by FunctionalTensor.to_functional
        x_functional = torch._to_functional_tensor(x)
        # Technically the FunctionalTensormode here is unnecessary,
        # but it avoids spurious NotImplemented logs during `ProxyTorchDispatchMode` tracing.
        # _mirror_autograd_meta_to queries tensor sizes,
        # and otherwise the sym_size() call will go to the proxy mode before hitting
        # FunctionalTensor.__torch_dispatch__

        functional_mode = _detect_infra_mode(torch._C._TorchDispatchModeKey.FUNCTIONAL)
        if functional_mode is None:
            raise AssertionError("functional_mode must not be None")

        with functional_mode:
            torch._mirror_autograd_meta_to(x, x_functional)  # type: ignore[attr-defined]
            out = FunctionalTensor(x_functional, functional_mode)
            torch._mirror_autograd_meta_to(x_functional, out)  # type: ignore[attr-defined]
        return out

    def from_functional(self) -> torch.Tensor:
        torch._sync(self)
        return torch._from_functional_tensor(self.elem)

    def is_base_tensor(self) -> bool:
        return torch._is_functional_tensor_base(self.elem)

    def replace_(self, output: torch.Tensor) -> None:
        torch._functionalize_replace(self.elem, output)

    def commit_update(self) -> None:
        torch._functionalize_commit_update(self.elem)

    def sync(self) -> None:
        torch._functionalize_sync(self.elem)

    def mark_mutation_hidden_from_autograd(self) -> None:
        torch._functionalize_mark_mutation_hidden_from_autograd(self.elem)

    def tolist(self) -> Any:
        if self.elem.dim() == 0:
            return self.elem.item()
        elif self.elem.dim() == 1:
            return [elem.item() for elem in self.elem]
        else:
            return [elem.tolist() for elem in self.elem]

    def to(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        if _detect_infra_mode(torch._C._TorchDispatchModeKey.FUNCTIONAL).export:
            torch.ops.aten._assert_tensor_metadata(
                self,
                dtype=self.dtype,
                device=self.device,
                layout=self.layout,
            )
        return super().to(*args, **kwargs)

    # pyrefly: ignore[bad-override]
    def cuda(
        self, device: torch.device | int | str | None = None, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        device = device or torch.cuda.current_device()
        if len(args) > 0:
            return self.to(device, *args, **kwargs)
        else:
            return self.to(device=device, **kwargs)

    char = _conversion_method_template(dtype=torch.int8)
    cpu = _conversion_method_template(device=torch.device("cpu"))
    bfloat16 = _conversion_method_template(dtype=torch.bfloat16)
    byte = _conversion_method_template(dtype=torch.uint8)
    double = _conversion_method_template(dtype=torch.float64)
    float = _conversion_method_template(dtype=torch.float32)
    bool = _conversion_method_template(dtype=torch.bool)
    half = _conversion_method_template(dtype=torch.float16)
    int = _conversion_method_template(dtype=torch.int32)
    long = _conversion_method_template(dtype=torch.int64)

    # TODO(sparse-team): fixes #133174 but can we do without the relay?
    def to_dense(
        self,
        dtype: torch.dtype | None = None,
        *,
        masked_grad: builtins.bool | None = None,
    ) -> torch.Tensor:
        if self.layout == torch.strided:
            return self.to(dtype=dtype) if dtype is not None else self

        inner = torch._from_functional_tensor(self.elem)
        if not torch._subclasses.fake_tensor.is_fake_tensor(inner):
            out = self.elem.to_dense(dtype=dtype, masked_grad=masked_grad)
            if isinstance(out, torch.Tensor) and torch._is_functional_tensor(out):
                functional_mode = _detect_infra_mode(
                    torch._C._TorchDispatchModeKey.FUNCTIONAL
                )
                if functional_mode is None:
                    raise AssertionError("functional_mode must not be None")
                with functional_mode:
                    return FunctionalTensor(out, functional_mode)
            return out

        return torch.ops.aten.to_dense.default(
            self, dtype=dtype, masked_grad=masked_grad
        )

    @property
    # pyrefly: ignore[bad-override]
    def is_mkldnn(self) -> builtins.bool:
        return torch._from_functional_tensor(self.elem).is_mkldnn

    @property
    # pyrefly: ignore[bad-override]
    def layout(self) -> torch.layout:
        return torch._from_functional_tensor(self.elem).layout

    def __bool__(self) -> builtins.bool:
        return bool(self.item())


class FunctionalTensorMode(TorchDispatchMode):
    def __init__(
        self,
        pre_dispatch: bool = False,
        export: bool = False,
        _allow_token_discovery: bool = False,
        _keep_input_mutations: bool = True,
    ) -> None:
        super().__init__()
        self.export = export
        self.is_on_stack = False
        self.enter_stack = []
        # Indicates to our torch_dispatch dispatching infra that
        # this is an "infra" mode with lower dispatching precedence.
        self._mode_key = torch._C._TorchDispatchModeKey.FUNCTIONAL
        self.pre_dispatch = pre_dispatch
        # This will be turned off later for pre-dispatch functionalization
        self._dispatch_key = torch._C.DispatchKey.PreDispatch if pre_dispatch else None  # type: ignore[attr-defined]
        # Map of effect type (ex. _EffectType.ORDERED) to a token. The tokens help keep
        # track of the ordering between side effectful operations.
        self._tokens: dict[Any, torch.Tensor] = {}

        # Filled after forward tracing.
        self._tokens_forward_output: dict[Any, torch.Tensor] = {}

        # Functionalization runs twice in AOTAutograd, once in
        # `run_functionalized_fw_and_collect_metadata` to collect metadata to
        # see which tensors need to be functionalized and discover how many
        # tokens we need, and another time in `make_fx` which does the actual
        # tracing to replace ops with their functional variants and handling
        # side-effectful ops. In the second stage there should be no token
        # discovery. This flag distinguishes between the two stages.
        self._allow_token_discovery = _allow_token_discovery
        # Controls whether invoke_subgraph keeps input mutations in its graph.
        self._keep_input_mutations = _keep_input_mutations

        self._storage_to_base: weakref.WeakKeyDictionary[
            torch.storage.UntypedStorage, FunctionalTensor | None
        ] = weakref.WeakKeyDictionary()

    # No-op if FunctionalTensorMode is already in use
    def __enter__(self) -> Self:
        def _get_prev_mode() -> FunctionalTensorMode | None:
            if self._dispatch_key == torch._C.DispatchKey.PreDispatch:
                return _get_dispatch_mode_pre_dispatch(
                    torch._C._TorchDispatchModeKey.FUNCTIONAL
                )
            return cast(
                "FunctionalTensorMode | None",
                torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FUNCTIONAL),
            )

        if _get_prev_mode() is None:
            self.enter_stack.append(True)
            return super().__enter__()
        else:
            self.enter_stack.append(False)
            return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        is_on_stack = self.enter_stack.pop()
        if is_on_stack:
            super().__exit__(exc_type, exc_val, exc_tb)

    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: Sequence[type],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        if _has_unrecognized_tensor_types(types):
            return NotImplemented

        if (
            func not in FunctionalTensor.metadata_fns
            and self._can_decompose(func, args, kwargs)
            # Not all funcs from __torch_dispatch__ are actual dispatcher ops,
            # e.g. prim.device
            and torch._C._dispatch_has_kernel(func.name())
        ):
            with self:
                r = func.decompose(*args, **kwargs)
                if r is not NotImplemented:
                    return r

        def wrap(x: object) -> object:
            # Only wrap our outputs in subclasses if the inner functionalization call
            # also wrapped outputs into FunctionalTensorWrappers.
            # When can this happen? e.g. `torch.div(2, 2)`
            if isinstance(x, FunctionalTensor):
                raise AssertionError("x must not be a FunctionalTensor in wrap()")
            if isinstance(x, torch.Tensor) and torch._is_functional_tensor(x):
                return FunctionalTensor(x, self)
            return x

        def unwrap(x: FunctionalTensor) -> torch.Tensor:
            return x.elem

        from torch._higher_order_ops.auto_functionalize import (
            can_auto_functionalize,
            do_auto_functionalize,
            do_auto_functionalize_v2,
        )

        if can_auto_functionalize(
            func
        ) and not torch._C._dispatch_has_kernel_for_dispatch_key(
            func.name(), torch._C.DispatchKey.Functionalize
        ):
            import torch._export.config as export_config
            import torch._inductor.config as inductor_config

            if torch.compiler.is_exporting():
                # NB: out= ops are not yet handled here; they only go through v2 below.
                if export_config.enable_auto_functionalized_v2_for_export:
                    return do_auto_functionalize_v2(self, func, args, kwargs)

                return do_auto_functionalize(self, func, args, kwargs)

            if inductor_config.enable_auto_functionalized_v2 or (
                isinstance(func, torch._ops.OpOverload)
                and (
                    torch._library.utils.is_out(func)
                    or torch._library.utils.is_inplace(func)
                )
            ):
                return do_auto_functionalize_v2(self, func, args, kwargs)
            return do_auto_functionalize(self, func, args, kwargs)

        from torch._higher_order_ops.effects import handle_effects, has_effects

        if has_effects(func):
            if torch._C._dispatch_has_kernel_for_dispatch_key(
                func.name(), torch._C.DispatchKey.Functionalize
            ):
                raise AssertionError(
                    f"func {func.name()} with effects should not have a kernel for Functionalize dispatch key"
                )
            return handle_effects(
                self._allow_token_discovery, self._tokens, func, args, kwargs
            )

        args_unwrapped, kwargs_unwrapped = pytree.tree_map_only(
            FunctionalTensor, unwrap, (args, kwargs)
        )

        # Expectation: functionalization should not **already** be enabled above our mode.
        # Why would that be bad? when we return a FunctionalTensor here, we don't want functionalization
        # to run above this mode and further wrap that output in **another** C++ FunctionalTensorWrapper.
        _assert_functionalize_not_active(
            "Functionalization should not already be enabled above this mode"
        )
        include_to_set = (
            torch._C._dispatch_tls_local_include_set()
            | torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
        )
        exclude_to_set = (
            torch._C._dispatch_tls_local_exclude_set().remove(
                torch._C.DispatchKey.Functionalize
            )
            - FunctionalTensor._extra_dispatch_keys
        )

        if isinstance(func, TorchBindOpOverload):
            # When the function is a TorchBindOpOverload, meaning some of the
            # inputs are FakeScriptObjects, we need to skip c++ dispatcher and
            # dispatch in python because C++ dispatcher will check the schema
            # and cannot recognize FakeScriptObject.
            ctx = PythonFunctionalizeAPI()
            fully_unwrapped_args = ctx.unwrap_tensors(args)
            fully_unwrapped_kwargs = ctx.unwrap_tensors(
                kwargs  # pyrefly: ignore[bad-argument-type]
            )
            outs_unwrapped = func(
                *fully_unwrapped_args,
                **fully_unwrapped_kwargs,
            )
            outs_wrapped = ctx.wrap_tensors(outs_unwrapped)
        else:
            # All we want to do here is reuse the existing C++ functionalization logic.
            # This requires swizzling our TLS dispatch keys so that the Functionalize key is active.
            with torch._C._ForceDispatchKeyGuard(include_to_set, exclude_to_set):
                try:
                    # By default for python functionalization (for AOTAutograd), we reapply views.
                    old_apply_views = torch._functionalize_enable_reapply_views(True)  # type: ignore[attr-defined]

                    # Sometimes these functions cannot be directly dispatched to functionalize key
                    # because args are sometimes not functional tensors for some reason?
                    if func in FunctionalTensor.metadata_fns:
                        # With dynamic shapes, metadata ops (e.g. aten.sym_size)
                        # on the unwrapped inner tensor re-enter Python dispatch
                        # because symbolic sizes require Python-side evaluation.
                        # The inner tensors don't have proxy slots (only the outer
                        # FunctionalTensor does), so ProxyTorchDispatchMode would
                        # store them as spurious graph constants. The metadata op
                        # was already recorded on the FunctionalTensor by the proxy
                        # mode above us, so we disable proxy tracing here.
                        with torch.fx.experimental.proxy_tensor.disable_proxy_modes_tracing():
                            outs_unwrapped = func(*args_unwrapped, **kwargs_unwrapped)
                        outs_wrapped = pytree.tree_map_only(
                            torch.Tensor, wrap, outs_unwrapped
                        )
                    else:
                        self._sync_view_replay_annotations(args, kwargs)

                        # When we dispatch to the C++ functionalization kernel, we might need to jump back to the
                        # PreDispatch mode stack afterwards, to handle any other PreDispatch modes underneath
                        # FunctionalTensorMode. If we call func() directly, we would need to exclude PreDispatch
                        # from the TLS in order to avoid infinite looping, but this would prevent us from coming
                        # back to PreDispatch later
                        outs_unwrapped = func._op_dk(
                            torch._C.DispatchKey.Functionalize,
                            *args_unwrapped,
                            **kwargs_unwrapped,
                        )

                        if self.export:
                            if func is torch.ops.aten.dropout.default:
                                torch._freeze_functional_tensor(outs_unwrapped)  # type: ignore[attr-defined]
                        outs_wrapped = pytree.tree_map_only(
                            torch.Tensor, wrap, outs_unwrapped
                        )
                finally:
                    torch._disable_functionalization()
                    torch._functionalize_enable_reapply_views(old_apply_views)  # type: ignore[attr-defined]

        _assert_functionalize_not_active(
            "Functionalization should not already be enabled above this mode after dispatch"
        )

        if (
            (
                func is torch.ops.aten.alias.default
                or func is torch.ops.aten.detach.default
            )
            and len(args) == 1
            and isinstance(args[0], FunctionalTensor)
        ):
            input_unwrapped = torch._from_functional_tensor(args[0].elem)
            if (
                isinstance(  # noqa: ISINSTANCE_FAKE_TENSOR
                    input_unwrapped, torch._subclasses.FakeTensor
                )
                and input_unwrapped.dispatch_keys is not None
            ):
                input_dispatch_keys = input_unwrapped.dispatch_keys

                def preserve_dispatch_keys(out: object) -> None:
                    if isinstance(out, FunctionalTensor):
                        unwrapped = torch._from_functional_tensor(out.elem)
                        if isinstance(  # noqa: ISINSTANCE_FAKE_TENSOR
                            unwrapped, torch._subclasses.FakeTensor
                        ):
                            unwrapped.dispatch_keys = input_dispatch_keys

                pytree.tree_map_(preserve_dispatch_keys, outs_wrapped)

        if (
            # If no outputs are our functional subclass, then don't try to fix up aliasing
            not any(
                isinstance(x, FunctionalTensor)
                for x in pytree.tree_leaves(outs_wrapped)
            )
            # Since lift_fresh lifts its argument into a functional tensor, we can skip the
            # aliasing correction step. Otherwise, we would be setting the storage of a
            # lifted tensor to that of an unlifted tensor.
            # Ref: https://github.com/pytorch/pytorch/issues/111506
            or func is torch.ops.aten.lift_fresh.default
        ):
            return outs_wrapped
        # for metadata mutations, need to manually mutate the metadata of the FunctionalTensor wrapper
        if (
            torch.Tag.inplace_view in func.tags
            and func is not torch.ops.aten.set_.source_Tensor
        ):
            with torch.utils._mode_utils.no_dispatch():
                func(*args, **kwargs)
        # Wrapper tensor subclasses do not have correct aliasing info! Use this util to manually correct the output aliasing.
        # inplace ops like `aten.add_()` are expected to return inputs **directly**, instead of creating fresh tensor objects.
        # Use this util to figure out the right thing to return.
        # If none of our inputs were wrapped, then we have no FunctionalTensor outputs that we need to fix up storages for.
        return return_and_correct_aliasing(func, args, kwargs, outs_wrapped)

    def _sync_view_replay_annotations(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        """Sync FunctionalTensor args so view replay uses correct fx node metadata.

        When functionalization encounters a mutation, it handles aliases by lazily
        regenerating them at the first time they are next used. This is a problem when
        plumbing user annotations during tracing: we want view ops from view replay to
        have the same annotation the user specified on the original views. But view
        replay happens the next time the alias is used (e.g.
        second_op(alias_with_pending_mutation)), so the regenerated views would get the
        metadata for second_op instead.

        To fix this, we remember the node metadata from the original views and globally
        set it when we lazily perform view replay. The globally set metadata will be
        used to populate the fx node created for the replayed operation.
        """
        m = torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY)
        if m is not None:
            for a in pytree.tree_leaves([args, kwargs]):
                if not isinstance(a, FunctionalTensor):
                    continue
                unwrapped = torch._from_functional_tensor(a.elem)
                try:
                    # pyrefly: ignore[missing-attribute]
                    tracker_entry = m.tracer.tensor_tracker[unwrapped]
                except KeyError:
                    # A tensor constant lifted from a nested HOP subgraph
                    # (e.g. invoke_subgraph / nested_compile_region) is wrapped
                    # as a FunctionalTensor during dispatch but is owned by the
                    # inner subgraph's tracer, not the outer one. Constants do
                    # not require grad; for any grad-requiring tensor that is
                    # missing from the tracker, re-raise to preserve visibility
                    # of genuine bugs.
                    if unwrapped.requires_grad:
                        raise RuntimeError(
                            f"cannot find {unwrapped} in tensor_tracker"
                        ) from None
                    continue
                curr_node = tracker_entry.proxy.node
                with fx_traceback.set_current_replay_node(curr_node):
                    torch._sync(a)

    def _can_decompose(
        self,
        func: OpOverload,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> bool:
        result = _can_decompose_fast(func, self.export, self.pre_dispatch)
        if result is not None:
            return result

        # in normal torch.compile IR, we only decompose an op if autograd
        # would have decomposed it (NB: autograd may have been skipped if
        # we are in inference mode)
        # TODO: the flatten here can potentially be deduped with the
        # unwrapping pytree_map later
        flat_args_kwargs, _ = pytree.tree_flatten((args, kwargs))
        return autograd_would_have_decomposed(func, flat_args_kwargs)

    @classmethod
    def is_infra_mode(cls) -> bool:
        return True


@contextlib.contextmanager
def disable_functional_mode() -> Generator[None, None, None]:
    return _disable_infra_mode(torch._C._TorchDispatchModeKey.FUNCTIONAL)


# This is similar to torch.func.functionalize, but:
# - It uses FunctionalTensorMode, and FunctionalTensor (a python subclass).
#   One important advantage to using this mode is that it will let us
#   run functionalization underneath __torch_dispatch__,
#   which we need in AOTAutograd.
# - Doing so means that it does not automatically compose with other
#   functorch transforms, since these transforms always run above __torch_dispatch__.
#   That's why this util lives here, and not in functorch.
# - Input mutations are only propagated back when propagate_input_mutations is set.
def dispatch_functionalize(
    func: Callable[..., Any],
    mode: FunctionalTensorMode = FunctionalTensorMode(),
    *,
    propagate_input_mutations: bool = False,
) -> Callable[..., Any]:
    # TODO: pull these from aot autograd
    def to_fun(t: object) -> object:
        if isinstance(t, torch.Tensor):
            return FunctionalTensor.to_functional(t)
        return t

    def from_fun(t: object) -> object:
        if not isinstance(t, FunctionalTensor):
            # quick sanity check
            if isinstance(t, torch.Tensor):
                if torch._is_functional_tensor(t):
                    raise AssertionError(
                        "Non-FunctionalTensor torch.Tensor should not be a functional tensor"
                    )
            return t
        torch._sync(t)
        return torch._from_functional_tensor(t.elem)

    def inner(*args: Any, **kwargs: Any) -> Any:
        disable_above = torch._C._ExcludeDispatchKeyGuard(
            torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
        )
        flat_inputs: list[Any] = []
        flat_func_inputs: list[Any] = []
        with disable_above:
            with mode:
                func_args = pytree.tree_map_only(torch.Tensor, to_fun, args)
                func_kwargs = pytree.tree_map_only(torch.Tensor, to_fun, kwargs)
                if propagate_input_mutations:
                    # A boxed func clears its input list, so flatten before the call.
                    flat_inputs = pytree.arg_tree_leaves(*args, **kwargs)
                    flat_func_inputs = pytree.arg_tree_leaves(*func_args, **func_kwargs)
                func_outputs = func(*func_args, **func_kwargs)
                outputs = pytree.tree_map_only(FunctionalTensor, from_fun, func_outputs)

            if propagate_input_mutations:
                from torch._C._functorch import _propagate_functional_input_mutation

                # Runs outside of mode so the copy_ mutates the caller's tensors.
                for arg, func_arg in zip(flat_inputs, flat_func_inputs):
                    if isinstance(func_arg, FunctionalTensor):
                        _propagate_functional_input_mutation(arg, func_arg.elem)

            return outputs

    return inner


class BaseFunctionalizeAPI(ABC):
    @abstractmethod
    def wrap_tensors(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        pass

    @abstractmethod
    def unwrap_tensors(self, args: torch.Tensor | tuple[torch.Tensor, ...]) -> Any:
        pass

    @abstractmethod
    def functionalize(self, inner_f: Callable[..., Any]) -> Callable[..., Any]:
        pass

    @abstractmethod
    def redispatch_to_next(self) -> AbstractContextManager[None]:
        pass

    def replace(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> None:
        torch._functionalize_replace(input_tensor, output_tensor)

    def commit_update(self, tensor: torch.Tensor) -> None:
        torch._functionalize_commit_update(tensor)

    def sync(self, tensor: torch.Tensor) -> None:
        torch._functionalize_sync(tensor)

    def mark_mutation_hidden_from_autograd(self, tensor: torch.Tensor) -> None:
        torch._functionalize_mark_mutation_hidden_from_autograd(tensor)


class PythonFunctionalizeAPI(BaseFunctionalizeAPI):
    def __init__(
        self, mode: FunctionalTensorMode | None = None, pre_dispatch: bool = False
    ) -> None:
        super().__init__()
        self.mode = mode if mode else FunctionalTensorMode()
        self.pre_dispatch = pre_dispatch

    def wrap_tensors(self, args: tuple[Any]) -> tuple[Any]:
        with self.mode:
            return torch.utils._pytree.tree_map_only(
                torch.Tensor, FunctionalTensor.to_functional, args
            )

    def unwrap_tensors(
        self, args: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]
    ) -> Any:
        return torch.utils._pytree.tree_map_only(
            FunctionalTensor, FunctionalTensor.from_functional, args
        )

    # pyrefly: ignore [implicit-any]
    def functionalize(self, inner_f: Callable) -> Callable:
        return dispatch_functionalize(inner_f, self.mode)

    def redispatch_to_next(self) -> AbstractContextManager[None]:
        # [NOTE] We don't do anything here because at the time
        # we exercise this path, we would have already popped the
        # FunctionalTensorMode from mode stack. Since FunctionalTensorMode
        # is now stateful, it is better to explicitly pass in correct mode
        # directly instead of globally setting it.
        return contextlib.nullcontext()

    @staticmethod
    def _check_cast_functional(tensor: torch.Tensor, name: str) -> FunctionalTensor:
        if not isinstance(tensor, FunctionalTensor):
            raise AssertionError(
                f"{name} must be a FunctionalTensor, got {type(tensor)}"
            )
        return tensor

    def replace(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> None:
        ft = self._check_cast_functional(input_tensor, "input_tensor")
        if isinstance(output_tensor, FunctionalTensor):
            raise AssertionError("output_tensor must not be a FunctionalTensor")
        ft.replace_(output_tensor)

    def commit_update(self, tensor: torch.Tensor) -> None:
        self._check_cast_functional(tensor, "tensor").commit_update()

    def sync(self, tensor: torch.Tensor) -> None:
        self._check_cast_functional(tensor, "tensor").sync()

    def mark_mutation_hidden_from_autograd(self, tensor: torch.Tensor) -> None:
        self._check_cast_functional(
            tensor, "tensor"
        ).mark_mutation_hidden_from_autograd()


class CppFunctionalizeAPI(BaseFunctionalizeAPI):
    def wrap_tensors(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        from torch._functorch.eager_transforms import _wrap_all_tensors_to_functional

        return _wrap_all_tensors_to_functional(args, level=0)

    def unwrap_tensors(
        self, args: torch.Tensor | tuple[torch.Tensor, ...]
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        from torch._functorch.eager_transforms import (
            _unwrap_all_tensors_from_functional,
        )

        return _unwrap_all_tensors_from_functional(args, reapply_views=_reapply_views())

    # pyrefly: ignore [implicit-any]
    def functionalize(self, inner_f: Callable) -> Callable:
        return torch.func.functionalize(inner_f)

    def redispatch_to_next(self) -> AbstractContextManager[None]:
        return torch._C._ExcludeDispatchKeyGuard(
            torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
        )


class FunctorchFunctionalizeAPI(BaseFunctionalizeAPI):
    def __init__(self, interpreter: FunctionalizeInterpreter) -> None:
        self.interpreter = interpreter

    def wrap_tensors(self, args: tuple[Any]) -> tuple[Any]:
        from torch._functorch.eager_transforms import _wrap_all_tensors_to_functional

        return _wrap_all_tensors_to_functional(args, level=self.interpreter.level())

    def unwrap_tensors(
        self, args: torch.Tensor | tuple[torch.Tensor, ...]
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        from torch._functorch.eager_transforms import (
            _unwrap_all_tensors_from_functional,
        )

        return _unwrap_all_tensors_from_functional(
            args, reapply_views=self.interpreter.functionalize_add_back_views()
        )

    # pyrefly: ignore [implicit-any]
    def functionalize(self, inner_f: Callable) -> Callable:
        return torch.func.functionalize(
            inner_f,
            remove=(
                "mutations_and_views"
                if self.interpreter.functionalize_add_back_views()
                else "mutations"
            ),
        )

    def redispatch_to_next(self) -> AbstractContextManager[None]:
        return self.interpreter.lower()


def mb_unwrap_functional_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, FunctionalTensor):
        return torch._from_functional_tensor(tensor.elem)
    return tensor
<|endoftext|>import torch
from torch._subclasses.fake_tensor import (
    DynamicOutputShapeException,
    FakeTensor,
    FakeTensorMode,
    UnsupportedFakeTensorException,
)
from torch._subclasses.fake_utils import CrossRefFakeMode


__all__ = [
    "FakeTensor",
    "FakeTensorMode",
    "UnsupportedFakeTensorException",
    "DynamicOutputShapeException",
    "CrossRefFakeMode",
]
<|endoftext|>from __future__ import annotations

import contextlib
import dataclasses
import functools
import threading
import typing
import weakref
from abc import abstractmethod
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import (
    Any,
    ClassVar,
    Generic,
    NewType,
    Protocol,
    TYPE_CHECKING,
    TypeGuard,
    TypeVar,
)
from typing_extensions import override, TypedDict, Unpack

import torch
from torch._C._autograd import CreationMeta
from torch._C._functorch import (
    _add_batch_dim,
    _unwrap_functional_tensor,
    _wrap_functional_tensor,
    get_unwrapped,
    is_batchedtensor,
    is_functorch_wrapped_tensor,
    is_gradtrackingtensor,
    is_legacy_batchedtensor,
    maybe_get_bdim,
    maybe_get_level,
    peek_interpreter_stack,
)
from torch._custom_class_base import CustomClassBase
from torch._dispatch.python import enable_python_dispatcher
from torch._logging import trace_structured
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._typing_utils import not_none
from torch.utils.weak import WeakIdKeyDictionary


if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from torch._C._functorch import CInterpreter
    from torch._guards import Source
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    # Import here to avoid cycle
    # Import the following modules during type checking to enable code intelligence features,
    # Do not import unconditionally, as they import sympy and importing sympy is very slow
    from torch.fx.experimental.symbolic_shapes import ShapeEnv, SymbolicContext
    from torch.types import IntLikeType


def _unwrap_python_functional_tensor(t: torch.Tensor) -> torch.Tensor:
    # Avoid a top-level import cycle: functional_tensor imports this module.
    from torch._subclasses.functional_tensor import FunctionalTensor

    if isinstance(t, FunctionalTensor):
        return t.from_functional()
    return t


def _clone_real_storage_from_tensor(t: torch.Tensor) -> torch.UntypedStorage:
    from torch._subclasses.fake_tensor import is_fake_tensor, maybe_get_real_tensor

    t = _unwrap_python_functional_tensor(t)
    if is_fake_tensor(t):
        real_tensor = maybe_get_real_tensor(t)
        if real_tensor is None:
            raise AssertionError(
                "t.real_tensor must not be None when copy_data is True"
            )
        t = real_tensor
    return t.untyped_storage().clone()


DimList = list
_TensorLikeT = TypeVar("_TensorLikeT", "MetaTensorDesc[Any]", torch.Tensor)
_T = TypeVar("_T")
_TensorT = TypeVar("_TensorT", bound=torch.Tensor)
_TensorT_cov = TypeVar("_TensorT_cov", bound=torch.Tensor, covariant=True)


def safe_is_leaf(t: MetaTensorDesc[Any] | torch.Tensor) -> bool:
    try:
        return t.is_leaf
    except RuntimeError:
        # inference mode can trigger this
        return False


def safe_grad(t: _TensorLikeT) -> _TensorLikeT | None:
    with torch._logging.hide_warnings(torch._logging._internal.safe_grad_filter):
        # pyrefly: ignore [bad-return]
        return t.grad


def _expect_safe_grad(t: _TensorLikeT) -> _TensorLikeT:
    grad = safe_grad(t)
    if grad is None:
        raise AssertionError("Expected tensor to have a gradient but grad is None")
    return grad


def assert_eq(a: _T, b: _T) -> None:
    if a != b:
        raise AssertionError(f"{a} != {b}")


tls = threading.local()
# Turns off inference mode for fake tensor propagation. This is turned to True
# only for `torch.compile`. Also look at
# _dynamo.config.fake_tensor_disable_inference_mode
tls.disable_inference_mode = False


@contextmanager
def disable_inference_mode_for_fake_prop() -> Generator[None, None, None]:
    prior = getattr(tls, "disable_inference_mode", False)
    tls.disable_inference_mode = True
    try:
        yield
    finally:
        tls.disable_inference_mode = prior


def assert_metadata_eq(
    assert_eq: Callable[[object, object], None],
    m1: MetaTensorDesc[Any] | torch.Tensor,
    m2: torch.Tensor,
    *,
    skip_symbolic: bool = False,
    skip_leaf: bool = False,
) -> None:
    m1 = (
        MetaTensorDescriber().describe_tensor(m1)
        if isinstance(m1, torch.Tensor)
        else m1
    )

    def go(m1: MetaTensorDesc[Any], m2: torch.Tensor) -> None:
        assert_eq(m1.dtype, m2.dtype)
        if not skip_symbolic:
            assert_eq(m1.shape, m2.shape)
        assert_eq(m1.requires_grad, m2.requires_grad)
        if not skip_leaf:
            assert_eq(m1.is_leaf, m2.is_leaf)
        # MetaTensorDesc doesn't store grad_fn; inferred from leaf
        # assert_eq(m1.grad_fn is None, m2.grad_fn is None)
        assert_eq(m1.is_sparse, m2.is_sparse)
        if not getattr(tls, "disable_inference_mode", False):
            assert_eq(m1.is_inference, m2.is_inference())
        else:
            assert_eq(m1.is_inference, False)
        assert_eq(m1.is_conj, m2.is_conj())
        assert_eq(m1.is_neg, m2.is_neg())
        assert_eq(m1.grad is not None, safe_grad(m2) is not None)
        if m1.grad is not None:
            go(m1.grad, _expect_safe_grad(m2))
        # TODO: move "assert_eq(m1.layout, m2.layout)" out of sparse
        #       branches (but not ready for prime time yet)...
        if m1.is_sparse:
            assert_eq(m1.layout, m2.layout)
            assert_eq(m1.dense_dim, m2.dense_dim())
            assert_eq(m1.sparse_dim, m2.sparse_dim())
            assert_eq(m1.is_coalesced, m2.is_coalesced())
        elif is_sparse_compressed(m1):
            assert_eq(m1.layout, m2.layout)
            assert_eq(m1.dense_dim, m2.dense_dim())
            assert_eq(m1.sparse_dim, m2.sparse_dim())
        else:
            if not skip_symbolic:
                assert_eq(m1.stride, m2.stride())
                assert_eq(m1.storage_offset, m2.storage_offset())
            assert_eq(m1.is_view, m2._is_view())
            if m1.is_view:
                if m1.base is None:
                    raise AssertionError("m1.base must not be None for a view tensor")
                if m2._base is None:
                    raise AssertionError("m2._base must not be None for a view tensor")
                go(m1.base, m2._base)
        # TODO: test if is resizable (no direct query for this atm)
        # TODO: audit AutogradMeta to see if it matches
        # TODO: test forward AD

    return go(m1, m2)


# TypeGuard (not TypeIs): False does not imply !torch.Tensor
def is_sparse_coo(t: object) -> TypeGuard[torch.Tensor]:
    return isinstance(t, torch.Tensor) and t.layout is torch.sparse_coo


def is_sparse_compressed_layout(layout: torch.layout) -> bool:
    return layout in {
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    }


# TypeGuard (not TypeIs): False does not imply !torch.Tensor
def is_sparse_compressed(t: object) -> TypeGuard[torch.Tensor]:
    return isinstance(t, torch.Tensor) and is_sparse_compressed_layout(t.layout)


# TypeGuard (not TypeIs): False does not imply !torch.Tensor
def is_sparse_any(t: object) -> TypeGuard[torch.Tensor]:
    return is_sparse_coo(t) or is_sparse_compressed(t)


def _checked_cast(ty: type[_T], obj: object) -> _T:
    if not isinstance(obj, ty):
        raise AssertionError(f"expected {ty} but got {type(obj)}")
    return obj


def _get_real_storage(base: torch.UntypedStorage) -> torch.UntypedStorage:
    return base.real_storage  # type: ignore[attr-defined]


def _set_real_storage(
    base: torch.UntypedStorage, real_storage: torch.UntypedStorage
) -> None:
    base.real_storage = real_storage  # type: ignore[attr-defined]


# Don't use id() directly, because those can get reallocated over time.
MetaStorageId = NewType("MetaStorageId", int)
MetaTensorId = NewType("MetaTensorId", int)


_DescriberId = NewType("_DescriberId", int)
DESCRIBER_NEXT_ID = _DescriberId(0)


class MetaTensorDescriber:
    """
    Given a Tensor/Storage, generate a MetaTensorDesc/MetaStorageDesc
    for it, which is enough information to reconstruct a meta tensor/fake tensor
    corresponding to a Tensor as faithfully as possible.

    This is a stateful conversion object because we keep track of the IDs
    of the tensors/storages passed to us, so we can consistently give
    the same ID when we see the same tensor/storage.
    """

    def __init__(self, *, copy_data: bool = False) -> None:
        global DESCRIBER_NEXT_ID
        self.id = DESCRIBER_NEXT_ID
        DESCRIBER_NEXT_ID = _DescriberId(DESCRIBER_NEXT_ID + 1)
        self.next_tensor_id: MetaTensorId = MetaTensorId(0)
        self.next_storage_id: MetaStorageId = MetaStorageId(0)
        # Tensor -> int
        self.lookup_tensor = WeakIdKeyDictionary()
        # Storage -> int
        self.lookup_storage = WeakIdKeyDictionary()
        self.copy_data = copy_data
        self.traced_tensors: set[int] = set()
        self.traced_storages: set[int] = set()

    def get_tensor_id(self, t: torch.Tensor) -> MetaTensorId:
        if t not in self.lookup_tensor:
            self.lookup_tensor[t] = self.next_tensor_id
            self.next_tensor_id = MetaTensorId(self.next_tensor_id + 1)
        return self.lookup_tensor[t]

    def get_storage_id(self, s: torch.UntypedStorage) -> MetaStorageId:
        if s not in self.lookup_storage:
            self.lookup_storage[s] = self.next_storage_id
            self.next_storage_id = MetaStorageId(self.next_storage_id + 1)
        return self.lookup_storage[s]

    def describe_storage(
        self,
        s: torch.UntypedStorage,
        *,
        data: torch.Tensor | None = None,
        trace: bool = False,
    ) -> MetaStorageDesc:
        r = MetaStorageDesc(
            id=self.get_storage_id(s),
            size=s.size(),
            # NB: We don't do the copy yet; copy happens when we start
            # creating the new storages
            data=data if data is not None else s if self.copy_data else None,
        )
        if trace and r.id not in self.traced_storages:
            trace_structured(
                "describe_storage",
                metadata_fn=lambda: r.as_json(self.id),
            )
            self.traced_storages.add(r.id)
        return r

    def describe_tensor(
        self, t: torch.Tensor, *, recurse: bool = True, trace: bool = False
    ) -> MetaTensorDesc[Any]:
        data = _unwrap_python_functional_tensor(t) if self.copy_data else None
        is_leaf = safe_is_leaf(t)
        is_view = t._is_view()
        is_sparse = t.is_sparse
        layout = t.layout
        is_nested = t.is_nested
        is_traceable_wrapper_subclass_v = is_traceable_wrapper_subclass(t)
        is_functorch_wrapped = is_functorch_wrapped_tensor(t)
        is_mkldnn = t.is_mkldnn
        is_batchedtensor_v = is_batchedtensor(t)
        is_legacy_batchedtensor_v = is_legacy_batchedtensor(t)
        is_gradtrackingtensor_v = is_gradtrackingtensor(t)
        is_functional = torch._is_functional_tensor(t)

        storage = None
        # NB: For compatibility, I default this to zero, as sometimes people
        # still have stuffed zero into storage offset even though the tensor
        # doesn't meaningfully have an offset
        storage_offset = 0
        if not (
            is_sparse
            or is_sparse_compressed_layout(layout)
            or (is_nested and not is_traceable_wrapper_subclass_v)
            or is_mkldnn
            # TODO: TBH, functorch wrapped tensors probably should have
            # storage associated with them
            or is_functorch_wrapped
            or is_legacy_batchedtensor_v
        ):
            # NB: We actually don't use storage to do views, but might as well
            # put it in for accuracy
            storage = self.describe_storage(t.untyped_storage(), data=data, trace=trace)
            storage_offset = t.storage_offset()  # type: ignore[assignment]

        stride = None
        if not (
            is_sparse
            or is_sparse_compressed_layout(layout)
            or (is_nested and not is_traceable_wrapper_subclass_v)
        ):
            # stride/storage_offset are called from is_functorch_wrapped,
            # view_from_base, empty_create_subclass,
            # sym_sizes_strides_storage_offset (empty_create)
            stride = t.stride()

        # NB: this technically should refer to functorch unwrapped tensor, but
        # I am (perhaps abusively) using it to store both the functorch and
        # non-functorch functional tensor
        unwrapped = None
        autograd_meta_from = None
        current_level = None
        if is_batchedtensor_v or is_gradtrackingtensor_v:
            unwrapped = self.describe_tensor(get_unwrapped(t), trace=trace)
        # xla and lazy tensors present as functional tensors, but we want them
        # to be handled specially
        elif is_functional and t.device.type not in ("xla", "lazy"):
            if t._is_view():
                raise RuntimeError(
                    "Cannot safely fakify a view because this process drops the view information right now."
                )
            if not is_functorch_wrapped:
                torch._sync(t)
                unwrapped = self.describe_tensor(
                    torch._from_functional_tensor(t), trace=trace
                )
                autograd_meta_from = t
            else:
                reapply_views = torch._C._functionalization_reapply_views_tls()
                # NB: has side effects!
                unwrapped = self.describe_tensor(
                    _unwrap_functional_tensor(t, reapply_views), trace=trace
                )
                # TODO: It's pretty suspicious that functional tensors don't have
                # valid level and thus we just grab whatever the current level
                # is
                current_level = torch._C._functorch.current_level()

        maybe_functorch_stack = None
        if is_functorch_wrapped:
            with (
                torch._functorch.pyfunctorch.temporarily_clear_interpreter_stack()
            ) as maybe_functorch_stack:
                pass

        attrs = None
        opaque_attrs = None
        ctx = None
        type_v = None
        if is_traceable_wrapper_subclass_v:
            if not hasattr(t, "__tensor_flatten__"):
                raise AssertionError(
                    "Traceable wrapper subclass must have __tensor_flatten__ method"
                )
            raw_attrs, ctx = t.__tensor_flatten__()
            attrs = {}
            opaque_attrs = {}
            for attr in raw_attrs:
                inner = getattr(t, attr)
                match inner:
                    case torch.Tensor():
                        attrs[attr] = self.describe_tensor(inner, trace=trace)
                    case CustomClassBase():
                        from torch._library.fake_class_registry import (
                            maybe_unwrap_fake_script_object,
                        )

                        opaque_attrs[attr] = maybe_unwrap_fake_script_object(inner)
                    case _:
                        raise AssertionError(
                            f"expected Tensor or CustomClassBase, got {type(inner)}"
                        )
            type_v = type(t)

        from torch.nested._internal.nested_tensor import _tensor_symint_registry

        view_func = ViewFunc.from_tensor(t)

        # TODO: Is it important to enable torch.inference_mode before querying
        # these values?
        is_inference_mode_disabled = getattr(tls, "disable_inference_mode", False)
        r: MetaTensorDesc[Any] = MetaTensorDesc(
            id=self.get_tensor_id(t),
            storage=storage,
            is_inference=False if is_inference_mode_disabled else t.is_inference(),
            is_leaf=is_leaf,
            requires_grad=t.requires_grad,
            # NB: ndim should be OK too but there is a disaster at
            # python test/dynamo/test_subclasses.py -k test_user_overridden_property_unsupported
            # Actually, this means that we have a little bit of a problem
            # here, which is that there is some sensitivity to how exactly an
            # access is done if you have a __torch_function__ subclass.  Maybe
            # should disable torch function before doing accesses?
            ndim=t.dim(),
            dtype=t.dtype,
            is_sparse=is_sparse,
            is_mkldnn=is_mkldnn,
            is_functorch_wrapped=is_functorch_wrapped,
            is_batchedtensor=is_batchedtensor_v,
            is_legacy_batchedtensor=is_legacy_batchedtensor_v,
            is_gradtrackingtensor=is_gradtrackingtensor_v,
            is_view=is_view,
            is_conj=t.is_conj(),
            is_neg=t.is_neg(),
            is_parameter=isinstance(t, torch.nn.Parameter),
            is_traceable_wrapper_subclass=is_traceable_wrapper_subclass_v,
            is_nested=is_nested,
            nested_int=(
                _tensor_symint_registry[t].node.nested_int()
                if t in _tensor_symint_registry
                else None
            ),
            is_functional=is_functional,
            layout=layout,
            device=t.device,
            size=t.size(),
            stride=stride,
            # pyrefly: ignore [bad-argument-type]
            storage_offset=storage_offset,
            dynamo_dynamic_indices=list(getattr(t, "_dynamo_dynamic_indices", ())),
            dynamo_hint_overrides=getattr(t, "_dynamo_hint_overrides", {}),
            sparse_dim=(
                t.sparse_dim() if t.is_sparse or is_sparse_compressed(t) else None
            ),
            dense_dim=t.dense_dim() if t.is_sparse or is_sparse_compressed(t) else None,
            is_coalesced=t.is_coalesced() if t.is_sparse else None,
            # TODO: I actually think recursing here is correct, but we have at
            # least an infinite cycle from base -> values -> base
            # https://github.com/pytorch/pytorch/issues/122089
            crow_indices=(
                self.describe_tensor(t.crow_indices(), recurse=False, trace=trace)
                if recurse and t.layout in {torch.sparse_csr, torch.sparse_bsr}
                else None
            ),
            col_indices=(
                self.describe_tensor(t.col_indices(), recurse=False, trace=trace)
                if recurse and t.layout in {torch.sparse_csr, torch.sparse_bsr}
                else None
            ),
            ccol_indices=(
                self.describe_tensor(t.ccol_indices(), recurse=False, trace=trace)
                if recurse and t.layout in {torch.sparse_csc, torch.sparse_bsc}
                else None
            ),
            row_indices=(
                self.describe_tensor(t.row_indices(), recurse=False, trace=trace)
                if recurse and t.layout in {torch.sparse_csc, torch.sparse_bsc}
                else None
            ),
            values=(
                self.describe_tensor(t.values(), recurse=False, trace=trace)
                if recurse and is_sparse_compressed(t)
                else None
            ),
            grad=(
                self.describe_tensor(grad, trace=trace)
                if (grad := safe_grad(t)) is not None
                else None
            ),
            creation_meta=(
                torch._C._autograd._get_creation_meta(t) if t._is_view() else None
            ),
            unwrapped=unwrapped,
            level=(
                maybe_get_level(t)
                if is_batchedtensor_v or is_gradtrackingtensor_v
                else None
            ),
            bdim=maybe_get_bdim(t) if is_batchedtensor_v else None,
            base=(
                self.describe_tensor(t._base, trace=trace)
                if recurse and t._is_view() and t._base is not None
                else None
            ),
            fake_mode=torch._subclasses.fake_tensor.maybe_get_fake_mode(t),
            view_func=view_func,
            # pyrefly: ignore [bad-argument-type]
            attrs=attrs,
            opaque_attrs=opaque_attrs if opaque_attrs else None,
            ctx=ctx,
            type=type_v,
            # NB: even if functorch is enabled, don't actually save the
            # interpreter stack here unless we are actually functorch wrapped;
            # it's irrelevant for non-functorch stuff
            functorch_stack=maybe_functorch_stack,
            autograd_meta_from=autograd_meta_from,
            current_level=current_level,
            data=data,
        )
        if trace and r.id not in self.traced_tensors:
            trace_structured(
                "describe_tensor",
                metadata_fn=lambda: r.as_json(self.id),
            )
            self.traced_tensors.add(r.id)
        return r


@dataclass(frozen=True)
class MetaStorageDesc:
    id: MetaStorageId
    size: int
    # NB: this is only populated with copy_data True, it is not directly
    # serializable in JSON, you want to do something special here anyway
    data: torch.UntypedStorage | torch.Tensor | None

    def as_json(self, describer_id: _DescriberId) -> dict[str, object]:
        metadata: dict[str, object] = {
            "id": self.id,
            "describer_id": describer_id,
            "size": self.size if isinstance(self.size, int) else repr(self.size),
        }
        if isinstance(self.size, torch.SymInt):
            node = self.size.node
            free_symbols = node.expr.free_symbols
            if free_symbols:
                shape_env = not_none(node.shape_env)
                if all(
                    symbol in shape_env.var_to_hint_override for symbol in free_symbols
                ):
                    metadata["size_hint"] = shape_env.optimization_hint(node.expr)
        return metadata


@dataclass(frozen=True)
class ViewFunc(Generic[_TensorT]):
    @abstractmethod
    def apply(
        self,
        t: _TensorT,
        new_base: _TensorT,
        symint_visitor_fn: Callable[[int], int] | None = None,
        tensor_visitor_fn: Callable[[torch.Tensor], _TensorT] | None = None,
    ) -> _TensorT: ...

    @staticmethod
    def from_tensor(t: torch.Tensor) -> ViewFunc[Any]:
        from torch._subclasses.fake_tensor import is_fake_tensor

        if is_fake_tensor(t):
            return _FakeTensorViewFunc()
        else:
            return _CustomViewFunc(t._view_func_unsafe)


@dataclass(frozen=True)
class _FakeTensorViewFunc(ViewFunc["FakeTensor"]):
    @override
    def apply(
        self,
        t: torch.Tensor,
        new_base: torch.Tensor,
        symint_visitor_fn: Callable[[int], int] | None = None,
        tensor_visitor_fn: Callable[[torch.Tensor], FakeTensor] | None = None,
    ) -> FakeTensor:
        return torch._subclasses.fake_tensor.FakeTensor._view_func_unsafe(
            # pyrefly: ignore [bad-argument-type]
            t,
            new_base,
            symint_visitor_fn,
            tensor_visitor_fn,
        )


@dataclass(frozen=True)
class _CustomViewFunc(ViewFunc[_TensorT], Generic[_TensorT]):
    func: Callable[
        [
            torch.Tensor,
            Callable[[int], int] | None,
            Callable[[torch.Tensor], _TensorT] | None,
        ],
        _TensorT,
    ]

    @override
    def apply(
        self,
        t: torch.Tensor,
        new_base: torch.Tensor,
        symint_visitor_fn: Callable[[int], int] | None = None,
        tensor_visitor_fn: Callable[[torch.Tensor], _TensorT] | None = None,
    ) -> _TensorT:
        # ignore `t`
        return self.func(new_base, symint_visitor_fn, tensor_visitor_fn)


# A callback where the device is either optional or required.
# All of these satisfy this protocol:
#   def mk(arg: Callable[[], torch.Tensor], device: Union[torch.device, str])
#   def mk(arg: Callable[[], torch.Tensor], device: Union[torch.device, str] = "meta")
#   def mk(arg: Callable[[], torch.Tensor], device: Optional[Union[torch.device, str]] = None)
class _MetaTensorCallback(Protocol, Generic[_TensorT_cov]):
    def __call__(
        self, arg: Callable[[], torch.Tensor], /, *, device: torch.device | str
    ) -> _TensorT_cov: ...


class _MetaTensorCallbackKwargs(TypedDict, total=False):
    device: torch.device | str


# A callback where the device may not be provided (is optional).
# All of these satisfy this protocol:
#   def mk(arg: Callable[[], torch.Tensor], device: Union[torch.device, str] = "meta")
#   def mk(arg: Callable[[], torch.Tensor], device: Optional[Union[torch.device, str]] = None)
class _MetaTensorCallbackOptDevice(Protocol, Generic[_TensorT_cov]):
    def __call__(
        self,
        arg: Callable[[], torch.Tensor],
        /,
        **kwargs: Unpack[_MetaTensorCallbackKwargs],
    ) -> _TensorT_cov: ...


@dataclass(frozen=True)
class MetaTensorDesc(Generic[_TensorT]):
    id: MetaTensorId
    ndim: int
    dtype: torch.dtype
    device: torch.device

    # NB: Sometimes, size, stride and storage_offset contain SymInt, in which
    # case this is NOT serializable.  That only happens when you're
    # re-fakeifying a fake tensor with an existing ShapeEnv... maybe we
    # can get rid of this use case entirely.  Notably, even if we are
    # fakeifying a real tensor into a fake tensor with symbolic shapes, the
    # size here is NOT dynamic
    # NB: These also contain SymInt because wrap_meta_outputs_with_default_device_logic
    # goes through this codepath.  But it really should not LOL.
    # NB: size could potentially be None as you can override it and make it
    # throw an error, but we don't currently have any subclasses that do this
    # except C++ nested tensor but we're going to have nested int to make this
    # defined on NJT
    size: tuple[int, ...]
    dynamo_dynamic_indices: list[int]
    dynamo_hint_overrides: dict[int, int]

    layout: torch.layout = torch.strided
    is_inference: bool = False
    is_leaf: bool = False
    requires_grad: bool = False
    is_sparse: bool = False
    is_mkldnn: bool = False
    is_functorch_wrapped: bool = False
    is_batchedtensor: bool = False
    is_legacy_batchedtensor: bool = False
    is_gradtrackingtensor: bool = False
    is_view: bool = False
    is_nested: bool = False
    # We eagerly symbolicize the associated nested int for e.g. offsets / lengths
    # metadata if that offsets is already associated with a nested int.
    # See test_construct_from_jagged_with_input_offsets_mixed_case.
    nested_int: int | None = None
    is_traceable_wrapper_subclass: bool = False
    is_functional: bool = False
    is_conj: bool = False
    is_neg: bool = False
    is_parameter: bool = False
    stride: tuple[int, ...] | None = None
    storage_offset: int = 0
    # NB: We have a choice whether or not to store the id or a direct pointer
    # to the data structure.  For ease of use, we store the data structure,
    # but this means that when we serialize, we have to swizzle these pointers
    # back into ids (so we have accurate aliasing relationships)
    storage: MetaStorageDesc | None = None
    sparse_dim: int | None = None  # is_sparse, is_sparse_compressed
    dense_dim: int | None = None  # is_sparse, is_sparse_compressed
    is_coalesced: bool | None = None  # is_sparse
    crow_indices: MetaTensorDesc[Any] | None = None  # is_sparse_compressed
    col_indices: MetaTensorDesc[Any] | None = None  # is_sparse_compressed
    ccol_indices: MetaTensorDesc[Any] | None = None  # is_sparse_compressed
    row_indices: MetaTensorDesc[Any] | None = None  # is_sparse_compressed
    values: MetaTensorDesc[Any] | None = None  # is_sparse_compressed
    unwrapped: MetaTensorDesc[Any] | None = None  # is_functorch_wrapped
    bdim: int | None = None  # is_functorch_wrapped
    base: MetaTensorDesc[Any] | None = None  # is_view
    attrs: dict[str, MetaTensorDesc[Any]] | None = None  # is_traceable_wrapper_subclass
    # A Tensor subclass containing opaque references is almost certainly NOT
    # serializable.
    opaque_attrs: dict[str, CustomClassBase] | None = (
        None  # non-tensor attrs from __tensor_flatten__
    )
    creation_meta: CreationMeta | None = None
    grad: MetaTensorDesc[Any] | None = None

    # Everything below is NOT serializable, need some more work

    _UNSERIALIZABLE: ClassVar[set[str]] = {
        "ctx",
        "type",
        "fake_mode",
        # view_func isn't serializable when it's a _CustomViewFunc
        "view_func",
        "level",
        "current_level",
        "functorch_stack",
        "autograd_meta_from",
        "data",
        "nested_int",
    }

    ctx: object | None = None  # is_traceable_wrapper_subclass
    type: type | None = None  # is_traceable_wrapper_subclass
    fake_mode: FakeTensorMode | None = None
    view_func: ViewFunc[Any] | None = None
    # level looks serializable, but actually it is meaningless without
    # the functorch_stack below
    level: int | None = None  # is_functorch_wrapped
    current_level: int | None = None
    functorch_stack: list[CInterpreter] | None = None
    autograd_meta_from: torch.Tensor | None = None

    # This is only populated on copy_data, and typically is not used at all,
    # except for some of our meta-ification paths that don't properly use
    # storage (pro-tip: you should use storage)
    data: torch.Tensor | None = None

    # Faithfully serializing functorch tensors will not be too difficult.
    # We only need to consider grad/vmap interpreters, and their internal
    # state is only bools (mostly what the grad enabled/disabled state
    # should be in the lower layer).  Beyond that, tensors just need to
    # precisely indicate which particular interpreter they correspond
    # to (we then replace level with a pointer to the interpreter stack.)
    # However, this use of functorch is very "non-lexical" so it's not
    # entirely clear how to make it all lexical again, so we haven't done
    # it for now.

    # NB: This will reference numeric IDs, and it is assumed that you've
    # already serialized everything this recursively references
    def as_json(self, describer_id: _DescriberId) -> dict[str, object]:
        def json(k: str, v: object) -> object:
            # Some best-effort debugging serialization for unserializable
            # fields (feel free to add other special cases as appropriate)
            if k in ["data", "autograd_meta_from"]:
                return None  # never repr these
            if k in MetaTensorDesc._UNSERIALIZABLE:
                return repr(v)
            if isinstance(v, (torch.device, torch.dtype, torch.layout)):
                return repr(v)
            if isinstance(v, torch.SymInt):
                return repr(v)
            if isinstance(v, (tuple, list)):
                return [json(k, v1) for v1 in v]
            if isinstance(v, (MetaStorageDesc, MetaTensorDesc)):
                return v.id
            if isinstance(v, CreationMeta):
                return str(v)
            if k == "attrs" and isinstance(v, dict):
                return {k1: v1.id for k1, v1 in v.items()}
            return v

        r = {
            field.name: json(field.name, getattr(self, field.name))
            for field in dataclasses.fields(self)
            if not (
                getattr(self, field.name) is field.default
                or (
                    field.name == "dynamo_dynamic_indices"
                    and not getattr(self, field.name)
                )
            )
        }
        r.update({"describer_id": describer_id})
        return r

    @property
    def shape(self) -> tuple[int, ...]:
        return self.size


# A more faithful reproduction would do a copy on the entire
# storage, but this needs to be done carefully because the
# underlying storage could have larger extent than is implied
# by size/stride.  The real fix is to properly call
# meta_storage recursively here.
#
# These "safe" functions are intended to be used under no_dispatch() mode.
# The no_dispatch() here is intended to prevent ambient fake tensor mode from
# fakeifying the operation.  But if we are given an honest to goodness
# FakeTensor as src, we MUST NOT run the copy/clone operation.  A better way
# to do this would be to not use no_dispatch and instead just disable fake
# tensor mode only (allowing for subclass dispatch to occur)
def _safe_copy(dst: torch.Tensor, src: torch.Tensor | None) -> None:
    if type(src) is not torch.Tensor:
        return
    dst.copy_(src)


def _safe_clone(src: torch.Tensor) -> torch.Tensor | None:
    if type(src) is not torch.Tensor:
        return None
    return src.clone()


def _grad_context_compatible(
    symbolic_context: torch.fx.experimental.symbolic_shapes.SymbolicContext,
    grad_desc: MetaTensorDesc[torch.Tensor],
) -> bool:
    """Check if a symbolic_context is compatible with a grad tensor.

    Returns False when the view base structure in symbolic_context doesn't
    match the grad, which means we need a fresh symbolic context.  This
    happens in FSDP2 where param._local_tensor is a view of an N-D padded
    base while grad._local_tensor is a view of a 1-D flat gradient buffer.

    We check at both the outer level and the inner (subclass attr) level.
    """
    from torch.fx.experimental.symbolic_shapes import (
        StatelessSymbolicContext,
        SubclassSymbolicContext,
    )

    def _view_base_compatible(
        ctx: StatelessSymbolicContext[Any, Any],
        grad_t: MetaTensorDesc[torch.Tensor],
    ) -> bool:
        vbc = ctx.view_base_context
        if grad_t.is_view and vbc is None:
            return False
        if not grad_t.is_view and vbc is not None:
            return False
        if (
            grad_t.is_view
            and vbc is not None
            and isinstance(vbc, StatelessSymbolicContext)
            and grad_t.base is not None
            and len(vbc.dynamic_sizes) != grad_t.base.ndim
        ):
            return False
        return True

    if not isinstance(symbolic_context, StatelessSymbolicContext):
        return True

    # Check outer level
    if not _view_base_compatible(symbolic_context, grad_desc):
        return False

    if grad_desc.is_traceable_wrapper_subclass and not isinstance(
        symbolic_context, SubclassSymbolicContext
    ):
        return False

    # Check inner (subclass) level
    if isinstance(symbolic_context, SubclassSymbolicContext):
        if grad_desc.attrs is None:
            return False
        for attr, inner_ctx in symbolic_context.inner_contexts.items():
            if attr not in grad_desc.attrs:
                return False
            if isinstance(
                inner_ctx, StatelessSymbolicContext
            ) and not _view_base_compatible(inner_ctx, grad_desc.attrs[attr]):
                return False

    return True


# This is a class for converting multiple tensors into meta tensors which
# share the same view/storage structure.  The operation model is you allocate
# one of these, and then call it repeatedly on all the tensors you want to
# convert.  It's important to use the same object for tensors you want to
# share storage because this is how we correlate shared storages to the same
# meta storages. This class will hold weak references to cached tensors
# and tensor storages.
class MetaConverter(Generic[_TensorT]):
    def __init__(self, *, copy_data: bool = False) -> None:
        # Maps MetaStorageId to UntypedStorage
        self.storage_memo: weakref.WeakValueDictionary[
            MetaStorageId, torch.UntypedStorage
        ] = weakref.WeakValueDictionary()
        # Maps MetaTensorId to torch.Tensor (typically a meta tensor or
        # FakeTensor)
        self.tensor_memo: weakref.WeakValueDictionary[MetaTensorId, _TensorT] = (
            weakref.WeakValueDictionary()
        )
        self.hit = 0
        self.miss = 0
        self.del_hook = None
        self.arg_cnt = 0
        # Ensures real_storage/real_tensor are populated on the resulting
        # metaified storage/tensor.  The naming of this attribute is load
        # bearing: FakeTensor relies on real tensor being set to exactly this
        # value
        self.copy_data = copy_data
        self.describer = MetaTensorDescriber(copy_data=copy_data)

    def successful(self) -> bool:
        return self.hit > 0 and self.miss == 0

    def get_tensor_memo(self, t: MetaTensorDesc[Any]) -> torch.Tensor | None:
        return self.tensor_memo.get(t.id, None)

    def _checked_get_tensor_memo(self, t: MetaTensorDesc[Any]) -> _TensorT:
        r = self.tensor_memo.get(t.id, None)
        if r is None:
            raise AssertionError(f"Tensor memo for id {t.id} is None")
        return r

    def set_tensor_memo(self, t: MetaTensorDesc[Any], v: _TensorT) -> None:
        self.tensor_memo[t.id] = v

    def get_storage_memo(self, s: MetaStorageDesc) -> torch.UntypedStorage | None:
        return self.storage_memo.get(s.id, None)

    def set_storage_memo(self, s: MetaStorageDesc, v: torch.UntypedStorage) -> None:
        self.storage_memo[s.id] = v

    def meta_storage(
        self,
        s: MetaStorageDesc,
        callback: Callable[[Callable[[], torch.Tensor]], _TensorT],
    ) -> torch.UntypedStorage:
        # If we are fakeifying a tensor that has a secretly-zero-sized storage,
        # Need to make sure to resize the meta storage too.
        if (memo := self.get_storage_memo(s)) is None:
            r_s = callback(
                lambda: torch.empty(s.size, dtype=torch.uint8, device="meta"),
            ).untyped_storage()
            if self.copy_data:
                # NB: no_dispatch is needed because internally storage copy is
                # implemented as Tensor operations
                with torch.no_grad(), no_dispatch():
                    if s.data is None:
                        raise AssertionError(
                            "s.data must not be None when copy_data is True"
                        )
                    if isinstance(s.data, torch.Tensor):
                        real_storage = _clone_real_storage_from_tensor(s.data)
                    else:
                        real_storage = s.data.clone()
                    _set_real_storage(r_s, real_storage)
            self.set_storage_memo(s, r_s)
            return r_s
        else:
            return memo

    @classmethod
    def _checked_cast_tensor_t(cls, t: torch.Tensor) -> _TensorT:
        # TODO: how to check _TensorT?
        return typing.cast(_TensorT, t)

    @classmethod
    def _identity_callable(
        cls,
        t: Callable[[], torch.Tensor],
        device: torch.device | str | None = None,
    ) -> _TensorT:
        return cls._checked_cast_tensor_t(t())

    @classmethod
    def _backward_error(cls, t: _TensorT) -> _TensorT:
        errfn = torch._C._functions.DelayedError(
            "Internal error: Tried to backward() through example input",
            1,
        )
        err = errfn(t)
        return typing.cast(_TensorT, err)

    def _empty_create_subclass(
        self,
        t: MetaTensorDesc[Any],
        outer_size: tuple[int, ...] | None,
        outer_stride: tuple[int, ...] | None,
        shape_env: ShapeEnv | None,
        symbolic_context: torch.fx.experimental.symbolic_shapes.SymbolicContext | None,
        callback: _MetaTensorCallbackOptDevice[_TensorT],
        source: torch._guards.Source,
    ) -> _TensorT:
        from torch._dynamo.source import AttrSource
        from torch.fx.experimental.symbolic_shapes import SubclassSymbolicContext

        # We are hitting plain meta_desc tensor so actually
        # create a tensor here.
        if t.attrs is None:
            return self.meta_tensor(
                t,
                shape_env,
                callback,
                source,
                symbolic_context,
            )

        inner_tensors: dict[str, torch.Tensor | CustomClassBase] = {}
        for attr, meta_tensor_desc in t.attrs.items():
            current_context = None
            if symbolic_context is not None:
                if not isinstance(symbolic_context, SubclassSymbolicContext):
                    raise AssertionError(
                        f"Expected SubclassSymbolicContext, got {type(symbolic_context)}"
                    )
                if attr not in symbolic_context.inner_contexts:
                    raise AssertionError(
                        f"tensor attr {attr!r} missing from inner_contexts"
                    )
                if (
                    current_context_ := symbolic_context.inner_contexts[attr]
                ) is not None:
                    current_context = _checked_cast(
                        torch.fx.experimental.symbolic_shapes.SymbolicContext,
                        current_context_,
                    )

            current_source = AttrSource(source, attr)
            inner_callback = functools.partial(callback, device=meta_tensor_desc.device)
            new_empty_tensor = self._empty_create_subclass(
                meta_tensor_desc,
                meta_tensor_desc.size,
                meta_tensor_desc.stride,
                shape_env,
                current_context,
                inner_callback,
                current_source,
            )
            inner_tensors[attr] = new_empty_tensor

        # Pass through opaque (non-tensor) attrs
        if t.opaque_attrs:
            inner_tensors.update(t.opaque_attrs)

        if t.type is None:
            raise AssertionError("t.type must not be None for subclass")
        return t.type.__tensor_unflatten__(  # type: ignore[attr-defined]
            inner_tensors, t.ctx, outer_size, outer_stride
        )

    # This function assumes that it's possible to do the conversion
    # NB: name here is used in a conventional way by Dynamo; it corresponds
    # precisely to the Source.name of the tensor we're fakeifying and
    # corresponds to a valid Python expression.  When we construct sub-names
    # as part of this process, we will maintain this invariant!  (Even though
    # other users of this may not need it this property to be upheld.)
    def meta_tensor(
        self,
        t: MetaTensorDesc[Any],
        shape_env: ShapeEnv | None,
        callback_: _MetaTensorCallback[_TensorT],
        source: Source | None,
        symbolic_context: SymbolicContext | None,
    ) -> _TensorT:
        from torch._subclasses.fake_tensor import is_fake_tensor, maybe_get_real_tensor

        callback: _MetaTensorCallbackOptDevice[_TensorT] = functools.partial(
            callback_, device=t.device
        )
        if source is None:
            from torch._dynamo.source import ConstantSource

            # TODO: make a dedicated UnknownSource for this?
            source = ConstantSource(
                f"__meta_utils_unknown_tensor{len(self.tensor_memo)}"
            )

        msg = (
            " This indicates you set no_dispatch() before calling into this"
            " function.  This is an error: we may be creating fake tensors and"
            " will perform operations on them which need fake tensor mode to"
            " be active.  You will segfault if you are in a no_dispatch() block."
        )
        if torch._C._dispatch_tls_local_exclude_set().has(torch._C.DispatchKey.Python):
            raise AssertionError(msg)
        self.arg_cnt += 1

        # When we make as_strided calls, we end up generating a guard
        # that the new as_strided tensor is in bounds for the old storage
        # for the base (since as_strided calls can "bust" out of their
        # bounding box.)  This guard is unnecessary: if a user is able
        # to provide us a tensor with the view base setup this way, we
        # don't need to produce a guard, because the fact that they
        # were able to produce the view base means its in bounds.
        #
        # Now, ordinarily, this guard would be harmless.  However, the
        # generated guard refers to variables bound on the base variable.
        # At the moment, Dynamo doesn't actually guard on x._base, because
        # according to Voz this results in a lot of spurious invalidations,
        # and also if the user doesn't directly make use of _base, its
        # pointless anyway (because programs should be parametric over
        # whether or not the input tensor is a view or not--unless you're
        # mutating the input, but that's a whole 'nother ballgame).  So
        # for expediency, we suppress these guards so we don't have to
        # deal with this (yet, anyway.)
        #
        # NB: An old version of this code suppressed guards for ALL operations
        # happening during meta conversion, not just as_strided calls.
        # This is too aggressive: we do duck sizing and 0/1 simplification
        # as we allocate variables, and we do need to register guards for
        # these cases.
        maybe_suppress: Callable[[], Any] = contextlib.nullcontext
        if shape_env is not None:
            maybe_suppress = shape_env.suppress_guards

        def sym_sizes_strides_storage_offset(
            t: MetaTensorDesc[Any],
            src: torch._guards.Source,
            symbolic_context: torch.fx.experimental.symbolic_shapes.SymbolicContext
            | None = symbolic_context,
        ) -> tuple[tuple[IntLikeType, ...], tuple[IntLikeType, ...], IntLikeType]:
            # local import to prevent circular import
            from torch.fx.experimental.symbolic_shapes import is_symbolic

            if t.stride is None:
                raise AssertionError("t.stride must not be None")

            if shape_env is not None:
                fake_mode = t.fake_mode
                has_symbolic = (
                    any(is_symbolic(sz) for sz in t.size)
                    or any(is_symbolic(sd) for sd in t.stride)
                    or is_symbolic(t.storage_offset)
                )
                if fake_mode is not None and fake_mode.shape_env is shape_env:
                    # Don't reallocate the sizes; the shape envs are the same,
                    # so reuse the old sizes/strides/etc
                    return (t.size, t.stride, t.storage_offset)
                elif (
                    fake_mode is not None
                    and not has_symbolic
                    and symbolic_context is None
                ):
                    return (t.size, t.stride, t.storage_offset)
                elif fake_mode is None:
                    # Real tensor (not a FakeTensor).  Use
                    # _create_symbolic_sizes_strides_storage_offset directly
                    # since there are no foreign symbols to transfer.
                    return shape_env._create_symbolic_sizes_strides_storage_offset(
                        t.size,
                        t.stride,
                        t.storage_offset,
                        [False] * t.ndim,
                        src,
                        symbolic_context=symbolic_context,
                        hint_overrides=t.dynamo_hint_overrides,
                    )
                else:
                    # FakeTensor from a different ShapeEnv.  Transfer symbols.
                    # _transfer_foreign_expr picks the hint up via the foreign env.
                    return shape_env.transfer_symbols_from_foreign_shape_env(
                        t.size,
                        t.stride,
                        t.storage_offset,
                        src,
                        symbolic_context=symbolic_context,
                    )
            else:
                return (t.size, t.stride, t.storage_offset)

        def sym_sparse_sizes(
            t: MetaTensorDesc[Any],
            src: torch._guards.Source,
            symbolic_context: torch.fx.experimental.symbolic_shapes.SymbolicContext
            | None = symbolic_context,
        ) -> tuple[IntLikeType, ...]:
            if shape_env is None:
                return t.size

            from torch._prims_common import make_contiguous_strides_for
            from torch.fx.experimental.symbolic_shapes import is_symbolic

            fake_mode = t.fake_mode
            has_symbolic = any(is_symbolic(sz) for sz in t.size)
            if fake_mode is not None and fake_mode.shape_env is shape_env:
                return t.size
            if symbolic_context is None and not (
                fake_mode is not None and has_symbolic
            ):
                return t.size

            sparse_tensor_with_synthetic_strides = dataclasses.replace(
                t,
                stride=make_contiguous_strides_for(t.size),
                storage_offset=0,
            )
            sizes, _, _ = sym_sizes_strides_storage_offset(
                sparse_tensor_with_synthetic_strides,
                src,
                symbolic_context,
            )
            return sizes

        def empty_create(
            inner_t: MetaTensorDesc[Any],
            inner_src: torch._guards.Source,
            symbolic_context: torch.fx.experimental.symbolic_shapes.SymbolicContext
            | None = symbolic_context,
        ) -> torch.Tensor:
            (
                inner_sizes,
                inner_strides,
                _inner_storage_offset,
            ) = sym_sizes_strides_storage_offset(inner_t, inner_src, symbolic_context)
            return torch.empty_strided(
                inner_sizes,
                inner_strides,
                dtype=inner_t.dtype,
                device="meta",
            )

        # Creates a subclass instance with empty inner tensors according to the specified
        # symbolic context.
        def empty_create_subclass(
            t: MetaTensorDesc[Any],
            outer_size: tuple[IntLikeType, ...],
            outer_stride: tuple[IntLikeType, ...],
            symbolic_context: torch.fx.experimental.symbolic_shapes.SymbolicContext
            | None = symbolic_context,
            source: torch._guards.Source | None = source,
        ) -> _TensorT:
            from torch.fx.experimental.symbolic_shapes import SubclassSymbolicContext

            if t.attrs is None:
                raise AssertionError("t.attrs must not be None for subclass")
            if t.type is None:
                raise AssertionError("t.type must not be None for subclass")
            # NB: t.ctx could be None if the subclass in question has no
            # meaningful context

            # Note: transform_subclass will use __tensor_unflatten__ to generate
            # a fresh subclass wrapper with outer sizes / strides according to the
            # outer symbolic context (passed in to this function). Inner size / stride
            # / storage offset symbols are allocated according to the appropriate inner
            # symbolic contexts, after which the checks in transform_subclass() will
            # relate them to the outer metadata as possible.
            #
            # Morally, the code here is same as transform_subclass, but we've
            # written it from scratch to read EmptyCreateSubclass
            outer_size = outer_size if outer_size is not None else t.size
            # pyrefly: ignore [bad-assignment]
            outer_stride = outer_stride if outer_stride is not None else t.stride

            if symbolic_context is not None and not isinstance(
                symbolic_context, SubclassSymbolicContext
            ):
                raise AssertionError(
                    f"Expected SubclassSymbolicContext or None, got {type(symbolic_context)}"
                )

            if source is None:
                raise AssertionError("source must not be None")
            sub = self._empty_create_subclass(
                t,
                # pyrefly: ignore[bad-argument-type]
                outer_size,
                # pyrefly: ignore[bad-argument-type]
                outer_stride,
                shape_env,
                symbolic_context,
                callback,
                source,
            )

            # NB: Purposefully guard here to simplify the inner / outer symbols.
            # Using sym_eq() for symbolic comparison can result in an expression that's too
            # difficult to guard on, so we use == here.
            if sub.shape != outer_size:
                raise AssertionError(
                    f"Expected return value from {t.type}__tensor_unflatten__() to have "
                    f"shape equal to {outer_size}, but got: {sub.shape}"
                )
            if sub.stride() != outer_stride:
                raise AssertionError(
                    f"Expected return value from {t.type}__tensor_unflatten__() to have "
                    f"stride equal to {outer_stride}, but got: {sub.stride()}"
                )

            return sub

        # Returns an all-dynamic symbolic context used for metafying the given tensor with
        # fully dynamic dims. This is useful when fake-ifying intermediate tensors in
        # closed-over ViewFunc state, as we don't have symbolic contexts for them, but we
        # don't want to over-specialize during view replay.
        def all_dynamic_symbolic_context(
            t: MetaTensorDesc[Any],
            source: torch._guards.Source,
            shape_env: torch.fx.experimental.symbolic_shapes.ShapeEnv | None,
            callback: _MetaTensorCallback[_TensorT],
        ) -> torch.fx.experimental.symbolic_shapes.SymbolicContext:
            from torch._dynamo.source import AttrSource
            from torch.fx.experimental.symbolic_shapes import (
                DimDynamic,
                StatelessSymbolicContext,
                SubclassSymbolicContext,
            )

            view_base_context: (
                torch.fx.experimental.symbolic_shapes.SymbolicContext | None
            ) = None
            if t.is_view:
                if t.base is None:
                    raise AssertionError("t.base must not be None for view tensor")
                view_base_context = all_dynamic_symbolic_context(
                    t.base, AttrSource(source, "_base"), shape_env, callback
                )

            t_symbolic_context: torch.fx.experimental.symbolic_shapes.SymbolicContext
            t_dynamic_sizes = [DimDynamic.DYNAMIC] * t.ndim
            if t.is_traceable_wrapper_subclass:
                if t.attrs is None:
                    raise AssertionError("t.attrs must not be None for subclass")
                inner_contexts: dict[
                    str, torch.fx.experimental.symbolic_shapes.SymbolicContext
                ] = {}
                for attr, inner in t.attrs.items():
                    if not isinstance(attr, str):
                        raise AssertionError(
                            f"Expected attr to be str, got {type(attr)}"
                        )
                    inner_contexts[attr] = all_dynamic_symbolic_context(
                        inner, AttrSource(source, attr), shape_env, callback
                    )
                t_symbolic_context = SubclassSymbolicContext(
                    dynamic_sizes=t_dynamic_sizes,
                    constraint_sizes=[None] * t.ndim,
                    inner_contexts=inner_contexts,  # type: ignore[arg-type]
                    tensor_source=source,
                    view_base_context=view_base_context,
                )
            else:
                t_symbolic_context = StatelessSymbolicContext(
                    dynamic_sizes=t_dynamic_sizes,
                    constraint_sizes=[None] * t.ndim,
                    view_base_context=view_base_context,
                )

            return t_symbolic_context

        # Returns a fake-ified version of an input view tensor t, given an already fake-ified
        # base. At a high level, we want two things:
        #   1. fake_t should have the same view relationship to the given fake base as the
        #      input t has to its _base.
        #   2. fake_t should have symbolic sizes / strides / storage offset according to the
        #      appropriate symbolic context (i.e. from the automatic dynamic algorithm).
        #
        # We currently take different strategies across view types:
        #   * For dense -> dense views, accomplish both (1) and (2) simultaneously via an
        #     as_strided() call on the fake-ified base, passing symbolic metadata.
        #   * For views involving subclasses, perform view replay using view funcs to
        #     achieve (1). It's necessary for (2) to swap out any closed-over state in
        #     the view funcs with symbolicized SymInts and fake-ified tensors. Doing this
        #     avoids specialization (and thus over-eager simplification of symbols) that
        #     could occur during view replay on the fake-ified base.
        #
        # Examples:
        #   * t.unsqueeze(-1) with dense t is a dense -> dense view. It can be modeled
        #     with an as_strided() call on the fake base passing symbolic metadata.
        #   * sub.select(dim=0, index=3) is a subclass -> subclass view. The index arg
        #     is made symbolic to avoid invalid specialization and view replay is then
        #     done to reconstruct the view.
        #   * _nested_from_jagged(values, offsets) is a dense -> subclass view
        #     that returns a subclass instance from a dense values tensor. The offsets
        #     tensor is closed over in the view func, as it can be considered view metadata.
        #     First, the offsets tensor is fake-ified according to the inner symbolic
        #     context and with the correct relationship to the outer size / stride metadata.
        #     Then view replay is done, swapping in the fake offsets so the view replay output
        #     is fully fake with no invalid specialization.
        def view_from_base(
            base: _TensorT,
            t: MetaTensorDesc[Any],
            shape_env: torch.fx.experimental.symbolic_shapes.ShapeEnv
            | None = shape_env,
        ) -> _TensorT:
            with enable_python_dispatcher():
                # fake-ify t's metadata according to the outer symbolic context
                (sizes, strides, storage_offset) = sym_sizes_strides_storage_offset(
                    t, source
                )
                if (
                    not t.is_traceable_wrapper_subclass
                    and not is_traceable_wrapper_subclass(base)
                ):
                    # Dense -> Dense view case uses as_strided() to construct view relationship.
                    # TODO: Change this logic to use view replay for consistency?
                    # It's likely there is no view func available.
                    with maybe_suppress():
                        return self._checked_cast_tensor_t(
                            base.as_strided(sizes, strides, storage_offset)
                        )

                from torch._dynamo.source import EphemeralSource
                from torch.fx.experimental.symbolic_shapes import (
                    StatelessSymbolicContext,
                    sym_eq,
                )

                def symint_visitor_fn(s: int) -> IntLikeType:
                    nonlocal symbolic_context
                    from torch.fx.experimental.symbolic_shapes import DimDynamic

                    all_static_sizes = (
                        symbolic_context is not None
                        and isinstance(symbolic_context, StatelessSymbolicContext)
                        and all(
                            x is DimDynamic.STATIC
                            for x in symbolic_context.dynamic_sizes
                        )
                    )
                    # Can't just rely on shape env being None - dynamo always initializes it
                    if all_static_sizes or shape_env is None:
                        return s

                    # NB: The symbol here is expected to be simplified out because we a priori
                    # allocate inner and outer symbols according to the appropriate symbolic
                    # contexts and prefer those over this symbol during symbol simplification
                    # (via usage of EphemeralSource below). This -shouldn't- happen, but if
                    # this symbol somehow leaks out beyond the view tensor's shape metadata, our
                    # assumption of it being simplified out will fail and it may be guarded on,
                    # which will hard error.
                    sym_source = EphemeralSource("symint_visitor_fn")

                    symbol = shape_env.create_symbol(s, sym_source, positive=None)
                    return shape_env.create_symintnode(
                        symbol, hint=s, source=sym_source
                    )

                real_to_fake_mapping = {}
                if t.is_traceable_wrapper_subclass:
                    if t.attrs is None:
                        raise AssertionError("t.attrs must not be None for subclass")
                    # NB: t.ctx could be None if the subclass in question has no
                    # meaningful context
                    if t.type is None:
                        raise AssertionError("t.type must not be None for subclass")

                    # Fake-ify t naively here; this is only done so we can get fake-ified inner
                    # tensors with the correct relationships to the outer sizes / strides for use
                    # in view replay. It's done beforehand here because it's not easy to do when
                    # visiting tensors one-by-one during view replay.
                    #
                    # Example:
                    #   Consider a Dense -> NJT view. NJT has (values, offsets) components and we
                    #   want a view of values with the offsets closed over. As the offsets component
                    #   is needed to describe the output view, it's important that it's fakeified
                    #   correctly.
                    fake_t: _TensorT = empty_create_subclass(
                        t, outer_size=sizes, outer_stride=strides
                    )
                    attrs, _ = fake_t.__tensor_flatten__()  # type: ignore[attr-defined]
                    for attr in attrs:
                        if attr in t.attrs:
                            real_to_fake_mapping[t.attrs[attr].id] = getattr(
                                fake_t, attr
                            )

                def tensor_visitor_fn(
                    visited_t: torch.Tensor,
                    # These arguments are never passed, we just use them to close
                    # over these relevant values
                    shape_env: torch.fx.experimental.symbolic_shapes.ShapeEnv
                    | None = shape_env,
                    callback: _MetaTensorCallbackOptDevice[_TensorT] = callback,
                ) -> torch.Tensor:
                    # It's possible to close over an undefined tensor (e.g. NJT's lengths).
                    if visited_t is None:
                        # pyrefly: ignore [bad-return]
                        return None

                    # NB: visited_t being a Tensor here is very naughty!  Should
                    # have already been described

                    # Fake inner tensors of view subclasses will come from the mapping built above.
                    visited_id = self.describer.get_tensor_id(visited_t)
                    fake_visited_t = real_to_fake_mapping.get(visited_id)
                    if fake_visited_t is not None:
                        return fake_visited_t

                    visited_desc = self.describer.describe_tensor(visited_t)

                    # For other closed-over tensor state, fake-ify it as all dynamic with an
                    # ephemeral source. This avoids invalid specialization during view replay.
                    # If we find that in practice the usage of ephemeral sources isn't enough
                    # to guarantee that we don't have guards on these symbols, we may need to
                    # explicitly suppress guards (as is done for _base in the dense -> dense
                    # view case).
                    temp_source = EphemeralSource("tensor_visitor_fn")
                    return self.meta_tensor(
                        visited_desc,
                        shape_env,
                        callback,
                        temp_source,
                        all_dynamic_symbolic_context(
                            visited_desc, temp_source, shape_env, callback
                        ),
                    )

                # Replay the view, swapping out any non-symbolic SymInts or real tensors
                # for symbolic SymInts or fake tensors.
                if t.view_func is None:
                    raise AssertionError("t.view_func must not be None for view replay")
                # NB: we do NOT suppress guards here, we need to remove ephemeral
                # sources
                fake_t = t.view_func.apply(
                    t,
                    base,
                    # pyrefly: ignore[bad-argument-type]
                    symint_visitor_fn,
                    tensor_visitor_fn,
                )

                # Ensure the output has symbolic shapes according to the outer symbolic context.
                # These checks should simplify out any symbols created for closed-over view func
                # SymInts.
                torch._check(sym_eq(fake_t.size(), sizes))
                torch._check(sym_eq(fake_t.stride(), strides))
                torch._check(sym_eq(fake_t.storage_offset(), storage_offset))
                return fake_t

        if self.get_tensor_memo(t) is None:
            GRAD_TENSOR_SENTINEL_VALUE = -2

            with torch.inference_mode(t.is_inference):
                if t.is_sparse:
                    is_leaf = t.is_leaf

                    sparse_size = sym_sparse_sizes(t, source, symbolic_context)
                    if t.sparse_dim is None:
                        raise AssertionError("t.sparse_dim must not be None")
                    if t.dense_dim is None:
                        raise AssertionError("t.dense_dim must not be None")

                    # The lambda function below is similar to
                    # `t.to(device='meta')` except the latter
                    # preserves nnz value
                    r = callback(
                        lambda: torch.ops.aten._sparse_coo_tensor_with_dims_and_tensors(
                            t.sparse_dim,
                            t.dense_dim,
                            sparse_size,
                            torch.empty(
                                (t.sparse_dim, 0),
                                dtype=torch.int64,
                                device="meta",
                            ),
                            torch.empty(
                                (0, *sparse_size[t.sparse_dim :]),
                                dtype=t.dtype,
                                device="meta",
                            ),
                            dtype=t.dtype,
                            layout=torch.sparse_coo,
                            device="meta",
                            is_coalesced=t.is_coalesced,
                        )
                    )
                    if self.copy_data:
                        # Pray that sparse clone doesn't lose information
                        if t.data is None:
                            raise AssertionError(
                                "t.data must not be None when copy_data is True"
                            )
                        with torch.no_grad(), no_dispatch():
                            if not is_fake_tensor(r):
                                raise AssertionError("Expected r to be a FakeTensor")
                            # pyrefly: ignore[missing-attribute]
                            r.real_tensor = _safe_clone(t.data)
                    if not safe_is_leaf(r):
                        raise AssertionError(
                            "the callback you passed in doesn't detach"
                        )
                    # Note [is_coalesced is dispatched]
                    # Strangely enough, is_coalesced() is a dispatched operator,
                    # which means that it will get caught by fake tensor mode.
                    # Ordinarily this would error, but there's some logic in
                    # fake tensor ensure this doesn't happen.
                    r._coalesced_(bool(t.is_coalesced))
                    if t.requires_grad:
                        r.requires_grad = True
                    if t.requires_grad and not is_leaf:
                        # This should probably use DelayedError,
                        # but clone is fine for now for sparse tensors.
                        # (DelayedError does not work for sparse because it causes
                        # the Fake sparse tensor to "lose" its fakeness)
                        r = self._checked_cast_tensor_t(r.clone())
                        with torch.enable_grad():
                            r._coalesced_(bool(t.is_coalesced))
                elif is_sparse_compressed_layout(t.layout):
                    is_leaf = t.is_leaf

                    if t.layout in {torch.sparse_bsr, torch.sparse_bsc}:
                        if t.sparse_dim is None:
                            raise AssertionError(
                                "t.sparse_dim must not be None for sparse block layout"
                            )
                        if t.dense_dim is None:
                            raise AssertionError(
                                "t.dense_dim must not be None for sparse block layout"
                            )
                        if t.values is None:
                            raise AssertionError(
                                "t.values must not be None for sparse block layout"
                            )
                        batch_dim = t.ndim - t.sparse_dim - t.dense_dim
                        blocksize = t.values.shape[batch_dim + 1 : batch_dim + 3]
                    else:
                        blocksize = ()
                    if t.layout in {torch.sparse_csr, torch.sparse_bsr}:
                        if t.crow_indices is None:
                            raise AssertionError(
                                "t.crow_indices must not be None for sparse csr/bsr layout"
                            )
                        index_dtype = t.crow_indices.dtype
                    else:
                        if t.ccol_indices is None:
                            raise AssertionError(
                                "t.ccol_indices must not be None for sparse csc/bsc layout"
                            )
                        index_dtype = t.ccol_indices.dtype

                    r = callback(
                        lambda: torch.ops.aten._sparse_compressed_tensor_with_dims(
                            0,
                            t.dense_dim,
                            t.shape,
                            blocksize,
                            index_dtype,
                            layout=t.layout,
                            dtype=t.dtype,
                            device="meta",
                        )
                    )
                    if self.copy_data:
                        # Pray sparse clone doesn't lose information
                        if t.data is None:
                            raise AssertionError(
                                "t.data must not be None when copy_data is True"
                            )
                        with torch.no_grad(), no_dispatch():
                            if not is_fake_tensor(r):
                                raise AssertionError("Expected r to be a FakeTensor")
                            # pyrefly: ignore[missing-attribute]
                            r.real_tensor = _safe_clone(t.data)
                    if not safe_is_leaf(r):
                        raise AssertionError(
                            "the callback you passed in doesn't detach"
                        )
                    if t.requires_grad:
                        r.requires_grad = True
                    if t.requires_grad and not is_leaf:
                        r = self._backward_error(r)
                elif t.is_nested and not t.is_traceable_wrapper_subclass:
                    # TODO: Handle this better in Dynamo?
                    # There are checks there now, but this can still be triggered by a dense
                    # tensor graph input that is a view of a strided NT.
                    from torch._dynamo.exc import unimplemented

                    # NOTE this graph break will NOT be present in Dynamo's graph break registry
                    unimplemented(
                        gb_type="attempted to apply meta conversion to strided nested tensor",
                        context=str(t),
                        explanation="This is not supported.",
                        hints=[],
                    )
                elif t.is_mkldnn:
                    is_leaf = t.is_leaf
                    (
                        sizes,
                        strides,
                        _storage_offset,
                    ) = sym_sizes_strides_storage_offset(t, source)
                    # TODO: This doesn't seem right, where's the MKLDNN'ness
                    # lol
                    r = callback(
                        lambda: torch.empty_strided(
                            sizes, strides, dtype=t.dtype, device="meta"
                        )
                    )
                    if self.copy_data:
                        with torch.no_grad(), no_dispatch():
                            if t.size is None:
                                raise AssertionError(
                                    "t.size must not be None when copy_data is True"
                                )
                            if t.stride is None:
                                raise AssertionError(
                                    "t.stride must not be None when copy_data is True"
                                )
                            if not is_fake_tensor(r):
                                raise AssertionError("Expected r to be a FakeTensor")
                            real_tensor = torch.empty_strided(
                                t.size, t.stride, dtype=t.dtype, device=t.device
                            )
                            # pyrefly: ignore[missing-attribute]
                            r.real_tensor = real_tensor
                            if t.data is None:
                                raise AssertionError(
                                    "t.data must not be None when copy_data is True"
                                )
                            _safe_copy(real_tensor, t.data)
                    if not safe_is_leaf(r):
                        raise AssertionError(
                            "the callback you passed in doesn't detach"
                        )
                    if t.requires_grad:
                        r.requires_grad = True
                    if t.requires_grad and not is_leaf:
                        r = self._backward_error(r)
                elif t.is_functorch_wrapped:
                    if t.is_view:
                        from torch._dynamo.exc import unimplemented

                        unimplemented(
                            gb_type="attempted to apply meta conversion to view functorch tensor",
                            context=str(t),
                            explanation="This is not supported.",
                            hints=[],
                        )

                    # Wraps a functorch tensor class (BatchedTensor, GradTrackingTensor)
                    # in a FakeTensor
                    def _to_fake_tensor(t: MetaTensorDesc[Any]) -> _TensorT:
                        # TODO: why aren't the recursive calls going to
                        # meta_tensor
                        r: _TensorT
                        if t.is_batchedtensor:
                            if t.unwrapped is None:
                                raise AssertionError(
                                    "t.unwrapped must not be None for batchedtensor"
                                )
                            if t.level is None:
                                raise AssertionError(
                                    "t.level must not be None for batchedtensor"
                                )
                            if t.bdim is None:
                                raise AssertionError(
                                    "t.bdim must not be None for batchedtensor"
                                )
                            ft = _to_fake_tensor(t.unwrapped)
                            lvl = t.level
                            bdim = t.bdim
                            # You cannot create functorch tensors without
                            # having the ambient funtorch interpreter stack
                            # available, as the level refers to things in the
                            # stack
                            with torch._functorch.pyfunctorch.temporarily_restore_interpreter_stack(
                                t.functorch_stack
                            ):
                                r = self._checked_cast_tensor_t(
                                    _add_batch_dim(ft, bdim, lvl)
                                )
                        elif t.is_gradtrackingtensor:
                            if t.unwrapped is None:
                                raise AssertionError(
                                    "t.unwrapped must not be None for gradtrackingtensor"
                                )
                            if t.level is None:
                                raise AssertionError(
                                    "t.level must not be None for gradtrackingtensor"
                                )
                            disable_functorch = torch._C._DisableFuncTorch
                            with disable_functorch():
                                ft = _to_fake_tensor(t.unwrapped)
                            lvl = t.level
                            if lvl == GRAD_TENSOR_SENTINEL_VALUE:
                                r = ft
                            else:
                                with torch._functorch.pyfunctorch.temporarily_restore_interpreter_stack(
                                    t.functorch_stack
                                ):
                                    r = self._checked_cast_tensor_t(
                                        torch._C._functorch._wrap_for_grad(ft, lvl),
                                    )

                            is_leaf = t.is_leaf
                            if t.requires_grad and safe_is_leaf(r):
                                r.requires_grad = True
                            elif t.requires_grad and not is_leaf:
                                r = self._backward_error(r)
                        elif t.is_functional:
                            if t.unwrapped is None:
                                raise AssertionError(
                                    "t.unwrapped must not be None for functional tensor"
                                )
                            if t.current_level is None:
                                raise AssertionError(
                                    "t.current_level must not be None for functional tensor"
                                )
                            ft = self.meta_tensor(
                                t.unwrapped,
                                shape_env,
                                callback,
                                # NB: reuse these exactly, we treat the
                                # functional tensor as "invisible".
                                # TODO: Actually this all probably doesn't
                                # work, take a closer look.
                                source,
                                symbolic_context,
                            )
                            r = self._checked_cast_tensor_t(
                                _wrap_functional_tensor(ft, t.current_level),
                            )
                            # TODO: is_leaf/requires_grad?
                        else:
                            if t.stride is None:
                                raise AssertionError("t.stride must not be None")

                            sizes = t.size
                            strides = t.stride
                            r = callback(
                                lambda: torch.empty_strided(
                                    sizes,
                                    strides,
                                    dtype=t.dtype,
                                    device="meta",
                                ),
                                # device="meta",
                            )
                            if self.copy_data:
                                with torch.no_grad(), no_dispatch():
                                    r.real_tensor = torch.empty_strided(  # type: ignore[attr-defined]
                                        t.size,
                                        t.stride,
                                        dtype=t.dtype,
                                        device=t.device,
                                    )
                                    if t.data is None:
                                        raise AssertionError(
                                            "t.data must not be None when copy_data is True"
                                        )
                                    _safe_copy(r.real_tensor, t.data)  # type: ignore[attr-defined]
                        # pyrefly: ignore [bad-return]
                        return r

                    r = _to_fake_tensor(t)

                elif t.is_functional and t.device.type not in ["xla", "lazy"]:
                    if t.unwrapped is None:
                        raise AssertionError(
                            "t.unwrapped must not be None for functional tensor"
                        )
                    if t.is_functorch_wrapped:  # handled above
                        raise AssertionError(
                            "Expected non-functorch wrapped functional tensor"
                        )
                    unwrapped = self.meta_tensor(
                        t.unwrapped,
                        shape_env,
                        callback,
                        source,
                        symbolic_context,
                    )
                    r = self._checked_cast_tensor_t(
                        torch._to_functional_tensor(unwrapped)
                    )
                    torch._mirror_autograd_meta_to(t.autograd_meta_from, r)  # type: ignore[attr-defined]

                elif t.is_view:
                    # Construct views in two steps: recursively meta-fy their
                    # base, and then create view(s) off that.  NB: doing it
                    # directly from storage is WRONG because this won't cause
                    # version counters to get shared.

                    if t.base is None:
                        raise AssertionError("t.base must not be None for view tensor")

                    base_symbolic_context = None
                    if shape_env and symbolic_context is not None:
                        from torch.fx.experimental.symbolic_shapes import (
                            StatelessSymbolicContext,
                        )

                        if not isinstance(symbolic_context, StatelessSymbolicContext):
                            raise AssertionError(
                                f"Expected StatelessSymbolicContext, got {type(symbolic_context)}"
                            )
                        # NB: This should generally be set when the input is a view,
                        # but the exception right now is for fake-ifying grads, which is
                        # a work in progress.
                        if symbolic_context.view_base_context is not None:
                            base_symbolic_context = symbolic_context.view_base_context

                    # We don't track the example values of base tensors.
                    # Allocation of unbacked symbols can happen when mark_unbacked
                    # is called on a base tensor that have unbacked_indices.
                    with (
                        shape_env.ignore_fresh_unbacked_symbols()
                        if shape_env
                        else contextlib.nullcontext()
                    ):
                        base = self.meta_tensor(
                            t.base,
                            shape_env,
                            callback,
                            torch._dynamo.source.AttrSource(source, "_base"),
                            base_symbolic_context,
                        )

                    def is_c_of_r(
                        complex_dtype: torch.dtype, real_dtype: torch.dtype
                    ) -> bool:
                        return (
                            utils.is_complex_dtype(complex_dtype)
                            and utils.corresponding_real_dtype(complex_dtype)
                            == real_dtype
                        )

                    # In some situations, MetaConverter may be called in a
                    # context where autograd is disabled.  For the _is_view
                    # assert to pass, we have to setup the autograd view
                    # metadata anyway.  Do this by reenabling the
                    # ADInplaceOrView key.  This is kind of a hack.
                    old_exclude = torch._C._dispatch_tls_is_dispatch_key_excluded(
                        torch._C.DispatchKey.ADInplaceOrView
                    )
                    torch._C._dispatch_tls_set_dispatch_key_excluded(
                        torch._C.DispatchKey.ADInplaceOrView, False
                    )
                    try:
                        if base.dtype == t.dtype:
                            pass
                        elif is_c_of_r(base.dtype, t.dtype):
                            base = self._checked_cast_tensor_t(torch.view_as_real(base))
                        elif is_c_of_r(t.dtype, base.dtype):
                            base = self._checked_cast_tensor_t(
                                torch.view_as_complex(base)
                            )
                        else:
                            # This is not guaranteed to succeed.  If it fails, it
                            # means there is another dtype-converting view function
                            # that hasn't been handled here
                            base = self._checked_cast_tensor_t(base.view(t.dtype))

                        # This is very tricky.  Naively, you might expect this
                        # to hold:
                        #
                        #   if t.requires_grad and not safe_is_leaf(t)
                        #       assert t._base.requires_grad
                        #
                        # But it's not true!  As you can see in the following
                        # program:
                        #
                        #   x = torch.zeros(4)
                        #   y = x.view(1, 4)
                        #   y.requires_grad = True
                        #   z = y.view(1, 1, 4)
                        #   assert z._base is x
                        #
                        # So we may have to do *two* views out of the base to
                        # recreate this situation.
                        if t.is_leaf:
                            # Leaf views that track view metadata are created by
                            # creating a view inside a no_grad block
                            with torch.no_grad():
                                r = view_from_base(base, t)
                            # As it's a leaf, we can directly assign requires_grad
                            r.requires_grad = t.requires_grad
                        else:
                            if t.base.requires_grad == t.requires_grad:
                                # Easy case, just run the view op
                                with torch.enable_grad():
                                    r = view_from_base(base, t)

                                # NB: We don't actually faithfully replicate
                                # autograd connectivity, but that doesn't matter
                                # today. See following for more info:
                                # https://gist.github.com/soulitzer/e03f015b314c3f5fcf80888c69390913
                            else:
                                # Obscure case.  Create a leaf view and give it the
                                # correct requires_grad, then do the final view.
                                # NB: Can't have a non-leaf without requiring grad!
                                if not t.requires_grad:
                                    raise AssertionError(
                                        "t.requires_grad must be True for non-leaf view"
                                    )
                                with torch.no_grad(), enable_python_dispatcher():
                                    mid = self._checked_cast_tensor_t(
                                        base.view(base.shape)
                                    )
                                mid.requires_grad = t.requires_grad
                                with torch.enable_grad():
                                    r = view_from_base(mid, t)
                        # The CreationMeta influences whether or not inplace
                        # mutation is an error or not.  So we need to make
                        # sure we properly propagate this as well.
                        if t.creation_meta is None:
                            raise AssertionError(
                                "t.creation_meta must not be None for view tensor"
                            )
                        torch._C._autograd._set_creation_meta(r, t.creation_meta)
                    finally:
                        torch._C._dispatch_tls_set_dispatch_key_excluded(
                            torch._C.DispatchKey.ADInplaceOrView, old_exclude
                        )

                    r.fake_device = t.device  # type: ignore[attr-defined]

                else:
                    is_leaf = t.is_leaf

                    # Graph-Break for wrapped tensors
                    if (
                        not (t.is_batchedtensor or t.is_gradtrackingtensor)
                        and t.is_functorch_wrapped
                    ) or t.is_legacy_batchedtensor:
                        # pyrefly: ignore [bad-return]
                        return NotImplemented

                    (
                        sizes,
                        strides,
                        storage_offset,
                    ) = sym_sizes_strides_storage_offset(t, source, symbolic_context)

                    # If we have a subclass that desugars into dense tensors,
                    # perform our callback on each inner tensor.
                    if t.is_traceable_wrapper_subclass:
                        r = empty_create_subclass(
                            t, outer_size=sizes, outer_stride=strides
                        )
                    else:
                        r = callback(
                            lambda: torch.empty_strided(
                                sizes,
                                strides,
                                dtype=t.dtype,
                                device="meta",
                            )
                        )
                        if self.copy_data:
                            with torch.no_grad(), no_dispatch():
                                if t.size is None:
                                    raise AssertionError(
                                        "t.size must not be None when copy_data is True"
                                    )
                                if t.stride is None:
                                    raise AssertionError(
                                        "t.stride must not be None when copy_data is True"
                                    )
                                if not is_fake_tensor(r):
                                    raise AssertionError(
                                        "Expected r to be a FakeTensor"
                                    )
                                real_tensor = torch.empty_strided(
                                    t.size, t.stride, dtype=t.dtype, device=t.device
                                )
                                # pyrefly: ignore[missing-attribute]
                                r.real_tensor = real_tensor
                                _safe_copy(real_tensor, t.data)

                    if not safe_is_leaf(r):
                        raise AssertionError(
                            "the callback you passed in doesn't detach"
                        )
                    if t.requires_grad:
                        r.requires_grad = t.requires_grad
                        if not is_leaf:
                            # Fake up some autograd history.
                            # Note: we *used* to call .clone() here to mock up some autograd history.
                            # This is bad for subclasses.
                            # Consider the case where you have a wrapper subclass that is contiguous,
                            # but its inner tensor is noncontiguous().
                            # .clone() (or other ops) will have the side effect of changing
                            # the metadata of the inner tensor.
                            # So instead, we now have a dedicated fn to set autograd history,
                            # without inadvertently changing other metadata.
                            # pyrefly: ignore [bad-argument-type]
                            r = self._backward_error(r)

                    # Storage aliasing for traceable wrapper subclasses is tracked
                    # through their inner tensors. Memoizing the wrapper placeholder
                    # storage can later force a cross-device set_ on the wrapper.
                    if not t.is_traceable_wrapper_subclass:
                        s = t.storage
                        if s is None:
                            raise AssertionError("t.storage must not be None")
                        from torch.fx.experimental.symbolic_shapes import (
                            guard_or_false,
                            sym_eq,
                        )

                        storage_needs_resize = False
                        source_storage_size = s.size
                        source_storage_is_zero = (
                            isinstance(source_storage_size, int)
                            and source_storage_size == 0
                        )
                        if not r.is_nested:
                            r_storage = r.untyped_storage()
                            r_storage_size = r_storage.size()
                            if isinstance(r_storage_size, int) and isinstance(
                                source_storage_size, int
                            ):
                                storage_needs_resize = (
                                    source_storage_size != 0
                                    and r_storage_size < source_storage_size
                                )
                        if s.id not in self.storage_memo and (
                            r.is_nested
                            or (
                                guard_or_false(sym_eq(r.stride(), strides))
                                and guard_or_false(r.storage_offset() == storage_offset)
                            )
                        ):
                            if not r.is_nested:
                                # The freshly allocated tensor storage can stand in
                                # for the source storage only if it covers the whole
                                # storage. Parameters made from views are not
                                # autograd views, but can still share a larger
                                # storage with later parameters. Leave symbolic
                                # sizes alone; this preserves the fast path for
                                # dynamic inputs without adding storage-size guards.
                                # Zero-sized source storages are handled by the
                                # resize_(0) below.
                                if storage_needs_resize:
                                    r.untyped_storage().resize_(s.size)
                            # You're normal and happy, install the fresh storage into the memo
                            self.set_storage_memo(s, r.untyped_storage())
                            if self.copy_data:
                                if not is_fake_tensor(r):
                                    raise AssertionError(
                                        "Expected r to be a FakeTensor"
                                    )
                                r_real_tensor = maybe_get_real_tensor(r)
                                if r_real_tensor is None:
                                    raise AssertionError(
                                        "r.real_tensor must not be None when copy_data is True"
                                    )
                                if source_storage_is_zero:
                                    _set_real_storage(
                                        r.untyped_storage(),
                                        r_real_tensor.untyped_storage(),
                                    )
                                else:
                                    if t.size is None:
                                        raise AssertionError(
                                            "t.size must not be None when copy_data is True"
                                        )
                                    if t.stride is None:
                                        raise AssertionError(
                                            "t.stride must not be None when copy_data is True"
                                        )
                                    if t.data is None:
                                        raise AssertionError(
                                            "t.data must not be None when copy_data is True"
                                        )
                                    with torch.no_grad(), no_dispatch():
                                        real_storage = _clone_real_storage_from_tensor(
                                            t.data
                                        )
                                        _set_real_storage(
                                            r.untyped_storage(), real_storage
                                        )
                                        r_real_tensor.set_(
                                            real_storage,
                                            t.storage_offset,
                                            t.size,
                                            t.stride,
                                        )
                        else:
                            # You're in crazy town; somehow you gave us a tensor
                            # that wasn't a view, but had nonzero storage offset,
                            # nontrivial strides (such that clone() couldn't
                            # preserve them), or already aliases with another
                            # tensor's storage.  The most typical way to end
                            # up here is with set_.  So use set_ to bludgeon this
                            # in.
                            r_s = self.meta_storage(s, callback=callback)
                            # NB: In principle, this should always work, but there
                            # is some subtle difference in the autograd metadata
                            # that means we will backprop the set_ call, even if
                            # r is declared as an input to grad.
                            # See https://github.com/pytorch/pytorch/issues/87956
                            # for the reproducer.
                            # NB: The in_kernel_invocation_manager here is necessary
                            # for fake tensor.  If we run the set_ call with fake
                            # tensor on, r will improperly report that it is NOT a
                            # meta tensor but a cpu tensor, and then the set_ call
                            # will fail due to device mismatch.  no_dispatch() is
                            # not enough, because the fake tensor will still claim
                            # to be a CPU tensor and you'll end up in the CPU
                            # kernel.  Arguably this is a hack; a cleaner way to
                            # solve this is to have a FakeStorage concept which
                            # would report it's CPU device--no problem now!  But
                            # this is difficult to do because we don't have storage
                            # subclasses.  Relevant test is
                            # DynamicShapesFunctionTests::test_add_dynamic_shapes in
                            # test/dynamo/test_dynamic_shapes.py
                            maybe_fake_mgr: AbstractContextManager[None] = (
                                contextlib.nullcontext()
                            )
                            from torch._subclasses.fake_tensor import (
                                in_kernel_invocation_manager,
                                maybe_get_fake_mode,
                            )

                            mb_fake_mode = maybe_get_fake_mode(r)
                            if mb_fake_mode is not None:
                                maybe_fake_mgr = in_kernel_invocation_manager(
                                    mb_fake_mode
                                )
                            with torch.no_grad(), maybe_suppress():
                                with maybe_fake_mgr:
                                    r.set_(r_s, storage_offset, sizes, strides)
                                if self.copy_data:
                                    with torch.no_grad(), no_dispatch():
                                        if not is_fake_tensor(r):
                                            raise AssertionError(
                                                "Expected r to be a FakeTensor"
                                            )
                                        r_real_tensor = maybe_get_real_tensor(r)
                                        if r_real_tensor is None:
                                            raise AssertionError(
                                                "r.real_tensor must not be None when copy_data is True"
                                            )
                                        if t.stride is None:
                                            raise AssertionError(
                                                "t.stride must not be None when copy_data is True"
                                            )
                                        r_real_tensor.set_(
                                            _get_real_storage(r_s),
                                            t.storage_offset,
                                            t.size,
                                            t.stride,
                                        )

                if t.grad is not None:
                    from torch._dynamo.source import AttrSource

                    grad_source = AttrSource(source, "grad")
                    grad_symbolic_context = symbolic_context

                    # The param's symbolic_context may be incompatible with the
                    # grad when they have different view base dimensionalities.
                    # This happens in FSDP2 where param._local_tensor is a view
                    # of an N-D padded base but grad._local_tensor is a view of
                    # a 1-D flat gradient buffer.  Build a fresh context only in
                    # that case to preserve FX graph cache consistency.
                    if shape_env is not None and symbolic_context is not None:
                        if not _grad_context_compatible(symbolic_context, t.grad):
                            grad_symbolic_context = all_dynamic_symbolic_context(
                                t.grad, grad_source, shape_env, callback
                            )

                    # pyrefly: ignore [unbound-name]
                    r.grad = self.meta_tensor(
                        t.grad,
                        shape_env,
                        callback,
                        grad_source,
                        grad_symbolic_context,
                    )
                # pyrefly: ignore [unbound-name]
                torch._C._set_conj(r, t.is_conj)
                # pyrefly: ignore [unbound-name]
                torch._C._set_neg(r, t.is_neg)
            # This can be skipped if necessary for performance reasons
            skip_leaf = (
                t.is_gradtrackingtensor and t.level == GRAD_TENSOR_SENTINEL_VALUE
            )
            # pyrefly: ignore [unbound-name]
            assert_metadata_eq(assert_eq, t, r, skip_symbolic=True, skip_leaf=skip_leaf)
            # Thanks to storage resizing, it's possible to end up with a tensor
            # that advertises a real size, but has a storage that actually has zero bytes.
            # Need to reflect this in the generated FakeTensor.
            from torch.fx.experimental.symbolic_shapes import guard_or_false

            if t.storage is not None and guard_or_false(t.storage.size == 0):
                # pyrefly: ignore [unbound-name]
                r.untyped_storage().resize_(0)

            if t.is_parameter:
                # pyrefly: ignore [unbound-name]
                r._is_param = True

            # See Note: [Creating symbolic nested int]
            if t.nested_int is not None:
                # pyrefly: ignore [unbound-name]
                if not is_fake_tensor(r):
                    raise AssertionError("Expected r to be a FakeTensor for nested int")
                # pyrefly: ignore [unbound-name, missing-attribute]
                r.nested_int_memo = r.fake_mode.create_symbolic_nested_int(
                    nt_tensor_id=t.nested_int
                )

            # pyrefly: ignore [bad-argument-type, unbound-name]
            self.set_tensor_memo(t, r)

        return self._checked_get_tensor_memo(t)

    def __call__(
        self,
        t: torch.Tensor,
        shape_env: ShapeEnv | None = None,
        *,
        callback: _MetaTensorCallback[_TensorT] | None = None,
        source: Source | None = None,
        symbolic_context: SymbolicContext | None = None,
        # Controls whether or not we should dump the tensor metadata to structured logs
        # when source is not None.  Because we refakify after Dynamo is done,
        # we don't want to dump info again from AOTAutograd, it is redundant.
        trace: bool = True,
    ) -> _TensorT:
        callback_: _MetaTensorCallback[_TensorT]
        if callback is None:
            callback_ = self._identity_callable
        else:
            callback_ = callback
        # TODO: zero tensors?  We appear to have eliminated them by
        # excluding complex for now

        # Filter out cases we don't support
        # TODO: This can probably be simplified quite a bit
        if isinstance(t, torch.Tensor):
            if (
                # Lazy tensors are not supported.  Note that XLA is
                # implemented on top of lazy tensor, not excluded here; we
                # have some special handling for it; this is for XLA Dynamo
                # integration
                t.device.type == "lazy"
                or
                # Quantization is not supported
                t.is_quantized
                or
                # Views out of sparse tensors not currently supported (plain
                # sparse is supported though)
                (t._is_view() and t._base is not None and t._base.is_sparse)
            ):
                self.miss += 1
                # pyrefly: ignore [bad-return]
                return NotImplemented
            else:
                self.hit += 1
        elif torch.overrides.is_tensor_like(t):
            self.miss += 1
            # pyrefly: ignore [bad-return]
            return NotImplemented
        else:
            # non-Tensor types don't count as hit or miss
            return t

        if source is None:
            trace = False

        # Describe the tensor.  NB: do NOT disable ambient modes, we may need
        # to query them when figuring out what to put in here
        t_desc = self.describer.describe_tensor(t, trace=trace)

        if trace:
            if source is None:
                raise AssertionError("source must not be None when trace is True")
            trace_structured(
                "describe_source",
                metadata_fn=lambda: {
                    "describer_id": self.describer.id,
                    "id": t_desc.id,
                    "source": source.name,
                },
            )

        # Do the meta-fication.  Here, we disable all the ambient modes, to
        # better simulate what would be like to re-fakeify from a fresh
        # process
        with contextlib.ExitStack() as exit_stack:
            exit_stack.enter_context(torch._dispatch.python.suspend_functionalization())
            st = peek_interpreter_stack()
            if st is not None:
                exit_stack.enter_context(
                    torch._functorch.pyfunctorch.temporarily_clear_interpreter_stack()
                )

            r = self.meta_tensor(
                t_desc,
                shape_env,
                callback_,
                source,
                symbolic_context,
            )

        if type(t) is torch.nn.Parameter:
            # NB: Cannot directly use Parameter constructor
            # because that would force a detach, not desirable
            r._is_param = True

        # forward the 'is_buffer' metadata if present (for nn.Buffer checks)
        if getattr(t, "_is_buffer", False):
            # pyrefly: ignore [missing-attribute]
            r._is_buffer = True
            if hasattr(t, "persistent"):
                # pyrefly: ignore [missing-attribute]
                r.persistent = t.persistent

        # TODO: return the description for later
        return r


import torch._prims_common as utils
<|endoftext|>from __future__ import annotations

import functools
import itertools
import math
import operator
import sys
from functools import reduce
from typing import Any, cast as typing_cast, TYPE_CHECKING, TypeVar
from typing_extensions import ParamSpec

import torch
import torch._custom_op
import torch._logging
import torch._prims_common as utils
from torch._C._functorch import (
    _unwrap_for_grad,
    _wrap_for_grad,
    is_functorch_wrapped_tensor,
    TransformType,
)
from torch._dispatch.python import no_python_dispatcher
from torch._functorch.pyfunctorch import retrieve_current_functorch_interpreter
from torch._functorch.utils import enable_single_level_autograd_function
from torch._ops import OpOverload
from torch._prims_common import (
    canonicalize_dim,
    elementwise_dtypes,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    is_boolean_dtype,
    is_contiguous,
    is_contiguous_for_memory_format_or_false,
    is_contiguous_or_false,
    is_float_dtype,
    is_integer_dtype,
    make_contiguous_strides_for,
    ShapeType,
)
from torch._subclasses.fake_tensor import (
    DataDependentOutputException,
    DynamicOutputShapeException,
    FakeTensor,
    in_kernel_invocation_manager,
    is_fake_tensor,
    run_fallback_kernel,
    UnsupportedOperatorException,
)
from torch.fx.operator_schemas import _normalize_function_or_error
from torch.utils._stats import count_label


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.types import IntLikeType


FakeTensorLike = FakeTensor | torch.Tensor
_P = ParamSpec("_P")
_R = TypeVar("_R")
_T = TypeVar("_T")

pytree = torch.utils._pytree

__all__ = [
    "op_implementations_checks",
    "get_fast_op_impls",
    "stride_incorrect_op",
    "has_meta",
]

# pyrefly: ignore [implicit-any]
op_implementations_dict = {}
# pyrefly: ignore [implicit-any]
op_implementations_checks = []

aten = torch._ops.ops.aten
_MKLDNN_DISPATCH_KEYS = torch._C.DispatchKeySet(
    torch._C._dispatch_key_parse("MkldnnCPU")
)
_MKLDNN_DTYPES = {
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.uint8,
    torch.int8,
}
_MKLDNN_TO_DENSE_DTYPES = _MKLDNN_DTYPES | {torch.float8_e4m3fn}
_MKLDNN_DTYPE_CHANGE_RESTRICTED_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.float8_e4m3fn,
}
_TO_DENSE_SPARSE_LAYOUTS = frozenset(
    {
        torch.sparse_coo,
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    }
)
_PYTHON_TLS_SNAPSHOT_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.PythonTLSSnapshot
)


def ordered_set(*items: _T) -> dict[_T, bool]:
    return dict.fromkeys(items, True)


# This function indicates if the backend device
# supports non-contiguous tensors
def is_noncontiguous_supported(device: torch.device) -> bool:
    return device.type != "hpu"


def _same_device_or_unspecified_index(a: torch.device, b: torch.device) -> bool:
    if a.type != b.type:
        return False
    return a.index is None or b.index is None or a.index == b.index


_like_tensor_constructors = ordered_set(
    aten.empty_like.default,
    aten.empty_like.out,
    aten.full_like.default,
    aten.full_like.out,
    aten.ones_like.default,
    aten.ones_like.out,
    aten.rand_like.default,
    aten.rand_like.generator,
    aten.rand_like.out,
    aten.rand_like.generator_out,
    aten.randn_like.default,
    aten.randn_like.generator,
    aten.randn_like.out,
    aten.randn_like.generator_out,
    aten.randint_like.default,
    aten.randint_like.generator,
    aten.randint_like.Tensor,
    aten.randint_like.Tensor_generator,
    aten.randint_like.Tensor_out,
    aten.randint_like.Tensor_generator_out,
    aten.randint_like.out,
    aten.randint_like.generator_out,
    aten.randint_like.low_dtype,
    aten.randint_like.low_generator_dtype,
    aten.randint_like.low_dtype_out,
    aten.randint_like.low_generator_dtype_out,
    aten.zeros_like.default,
    aten.zeros_like.out,
    aten.new_empty.default,
    aten.new_empty.out,
    aten.new_empty_strided.default,
    aten.new_empty_strided.out,
    aten.new_full.default,
    aten.new_full.out,
    aten.new_zeros.default,
    aten.new_zeros.out,
    aten.new_ones.default,
    aten.new_ones.out,
)


_device_not_kwarg_ops = ordered_set(
    aten._resize_output_.default,
    aten._nested_tensor_from_tensor_list.default,
    aten._nested_tensor_from_tensor_list.out,
    aten.pin_memory.default,
    aten.to.device,
    aten.to.prim_Device,
    aten.is_pinned.default,
    aten._pin_memory.default,
    aten._pin_memory.out,
    aten._resize_output.default,
    aten._resize_output.out,
)

# this op is never actually used
_non_kwarg_device_constructors = (aten._list_to_tensor,)


def contains_tensor_types(type_: Any) -> bool:
    tensor_type = torch._C.TensorType.get()
    return type_.isSubtypeOf(tensor_type) or any(
        contains_tensor_types(e) for e in type_.containedTypes()
    )


@functools.cache
def _is_tensor_constructor(func: OpOverload) -> bool:
    if not isinstance(func, OpOverload):
        raise AssertionError(f"func must be an OpOverload, got {type(func)}")
    schema = func._schema
    if any(contains_tensor_types(arg.type) for arg in schema.arguments):
        return False
    # TODO: no real reason to restrict multiple outputs
    return (
        len(schema.returns) == 1 and schema.returns[0].type is torch._C.TensorType.get()
    )


@functools.cache
def _has_device_arg(func: OpOverload) -> bool:
    device_type = torch._C.DeviceObjType.get()
    optional_device_type = torch._C.OptionalType(device_type)
    return any(
        arg.name == "device"
        and (
            arg.type.isSubtypeOf(device_type)
            or arg.type.isSubtypeOf(optional_device_type)
        )
        for arg in func._schema.arguments
    )


def register_op_impl(
    run_impl_check: Callable[[OpOverload], bool]
    | OpOverload
    | list[OpOverload]
    | tuple[OpOverload, ...],
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    def impl_decorator(op_impl: Callable[_P, _R]) -> Callable[_P, _R]:
        if isinstance(run_impl_check, OpOverload):
            if run_impl_check in op_implementations_dict:
                raise AssertionError(f"duplicate registration: {run_impl_check}")
            op_implementations_dict[run_impl_check] = op_impl
            schema = run_impl_check._schema
            torch._C._fake_dispatch_register_op_impl(schema.name, schema.overload_name)
        elif isinstance(run_impl_check, (list, tuple)):
            for op in run_impl_check:
                register_op_impl(op)(op_impl)
        else:
            if not callable(run_impl_check):
                raise AssertionError(
                    f"run_impl_check must be callable, got {type(run_impl_check)}"
                )
            op_implementations_checks.append((run_impl_check, op_impl))

        return op_impl

    return impl_decorator


def _is_op_registered_to_fake_rule(op: OpOverload) -> bool:
    return op in op_implementations_dict


def _deregister_op_impl(op: OpOverload) -> None:
    if op in op_implementations_dict:
        op_implementations_dict.pop(op, None)
        torch._C._fake_dispatch_deregister_op_impl(
            op._schema.name, op._schema.overload_name
        )
    for check, impl in op_implementations_checks:
        if check is op:
            op_implementations_checks.remove((check, impl))
            break


@register_op_impl(op_implementations_dict.__contains__)
def dispatch_to_op_implementations_dict(
    fake_mode: FakeTensorMode, func: OpOverload, *args: Any, **kwargs: Any
) -> Any:
    return op_implementations_dict[func](fake_mode, func, *args, **kwargs)


@register_op_impl(_is_tensor_constructor)
@register_op_impl([*_like_tensor_constructors])
def constructors(
    fake_mode: FakeTensorMode, func: OpOverload, *args: Any, **kwargs: Any
) -> FakeTensor:
    if func in _non_kwarg_device_constructors:
        raise AssertionError(
            f"func must not be in _non_kwarg_device_constructors, got {func}"
        )
    _, new_kwargs = _normalize_function_or_error(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    if func in _like_tensor_constructors:
        default_device = new_kwargs["input"].device
        # TODO: file issue
        args = (new_kwargs.pop("input"),)
    else:
        # cpu is default device if none is specified
        default_device = torch.device("cpu")
        args = ()
    has_device_arg = _has_device_arg(func)
    out_device = new_kwargs.pop("device", None) if has_device_arg else None
    out_device = out_device if out_device is not None else default_device
    if has_device_arg:
        new_kwargs["device"] = torch.device("meta")
    # _like constructors have fake tensor inputs (maybe this causes the non-like
    # to fail? hmmm)
    with in_kernel_invocation_manager(fake_mode):
        r = func(*args, **new_kwargs)
    if r.device.type == "meta":
        return fake_mode.fake_tensor_converter.from_meta_and_device(
            fake_mode, r, out_device
        )
    return fake_mode.fake_tensor_converter.from_real_tensor(fake_mode, r)


@register_op_impl(aten.is_pinned.default)
def non_kwarg_is_pinned(
    fake_mode: FakeTensorMode, func: OpOverload, *args: Any, **kwargs: Any
) -> bool:
    _, new_kwargs = _normalize_function_or_error(
        func, args, kwargs, normalize_to_only_use_kwargs=True
    )
    inp = new_kwargs.pop("input")
    # we'll ignore device argument because it is deprecated and not
    # actually used by is_pinned.
    with in_kernel_invocation_manager(fake_mode):
        r = func(inp)
    return r


# Legacy profiler ops return Tensors but don't follow tensor constructor patterns
# They take string arguments and should not have device/dtype parameters added
@register_op_impl(torch.ops.profiler._record_function_enter.default)
def _record_function_enter(
    fake_mode: FakeTensorMode, func: OpOverload, name: str, args: object | None = None
) -> FakeTensor:
    # Call the real implementation to get a real handle tensor
    with in_kernel_invocation_manager(fake_mode):
        real_handle = func(name, args)
    # Create a meta tensor with the same properties as the real handle
    meta_handle = torch.empty_like(real_handle, device="meta")
    # Wrap it as a FakeTensor
    return FakeTensor(fake_mode, meta_handle, torch.device("cpu"))


@register_op_impl(torch.ops.profiler._record_function_exit.default)
def _record_function_exit(
    fake_mode: FakeTensorMode, func: OpOverload, handle: Any
) -> None:
    # Exit doesn't return anything and doesn't need to do anything for fake tensors
    # Just return None (the actual return type is void)
    pass


@register_op_impl(torch.ops.profiler._record_function_enter_new.default)
def _record_function_enter_new(
    fake_mode: FakeTensorMode, func: OpOverload, name: str, args: object | None = None
) -> Any:
    # Call the real implementation - returns a custom class, not a tensor
    # Just pass through without wrapping
    with in_kernel_invocation_manager(fake_mode):
        return func(name, args)


@register_op_impl(aten.to.prim_Device)
@register_op_impl(aten.to.device)
def non_kwarg_to(
    fake_mode: FakeTensorMode, func: OpOverload, *args: Any, **kwargs: Any
) -> FakeTensor:
    _, new_kwargs = _normalize_function_or_error(
        func, args, kwargs, normalize_to_only_use_kwargs=True
    )
    input_device = new_kwargs["device"]
    out_device = input_device if input_device else new_kwargs["input"].device
    new_kwargs["device"] = torch.device("meta")
    inp = new_kwargs.pop("input")
    with in_kernel_invocation_manager(fake_mode):
        r = func(inp, **new_kwargs)
    # TODO: I think this does the wrong thing if r is inp
    return fake_mode.fake_tensor_converter.from_meta_and_device(
        fake_mode, r, out_device
    )


def stride_incorrect_op(op: OpOverload) -> bool:
    return False


# These operators have meta implementations with incorrect strides
@register_op_impl(stride_incorrect_op)
def workaround_stride_incorrect_op(
    fake_mode: FakeTensorMode, func: OpOverload, *args: Any, **kwargs: Any
) -> FakeTensor:
    # This is a workaround for meta implementations with incorrect strides

    def is_symbolic(x: object) -> bool:
        if is_fake_tensor(x):
            return x._has_symbolic_sizes_strides
        if isinstance(x, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            return True
        return False

    # For static shapes, we can fall back to eager for the real strides
    if fake_mode.allow_fallback_kernels:
        require_dynamic = any(
            is_symbolic(x) for x in itertools.chain(args, kwargs.values())
        )
        if not require_dynamic:
            flat_args, args_spec = pytree.tree_flatten((args, kwargs))
            return run_fallback_kernel(
                fake_mode,
                func,
                flat_args,
                args_spec,
                # TODO: refactor to lambda so we don't instantiate extra errors before
                # calling
                RuntimeError("Cannot run fallback kernel for stride_incorrect_op"),
            )

    raise UnsupportedOperatorException(func)


# Don't default to default device handling,
# since the device of `the_template` is ignored
@register_op_impl(aten.resize_as_.default)
def resize_as_(
    fake_mode: FakeTensorMode, func: OpOverload, *args: Any, **kwargs: Any
) -> FakeTensor:
    with in_kernel_invocation_manager(fake_mode):
        return func(*args, **kwargs)


@register_op_impl(aten._sparse_coo_tensor_with_dims_and_tensors.default)
def _sparse_coo_tensor_with_dims_and_tensors(
    fake_mode: FakeTensorMode, func: OpOverload, *args: Any, **kwargs: Any
) -> FakeTensor:
    return constructors(fake_mode, func, *args, **kwargs)


def _spdiags_static_offsets(offsets: FakeTensorLike) -> list[int] | None:
    constant = getattr(offsets, "constant", None)
    if constant is None:
        constant = getattr(offsets, "real_tensor", None)
    if is_fake_tensor(constant):
        return None
    if constant is None or constant.device.type != "cpu":
        return None
    return [int(offset) for offset in constant.reshape(-1).tolist()]


def _spdiags_nnz(offsets: list[int], rows: int, cols: int, diag_cols: int) -> int:
    nnz = 0
    pos_offset_limit = min(cols, diag_cols)
    for offset in offsets:
        if offset <= 0:
            nnz_per_diag = min(offset + rows, diag_cols)
        else:
            nnz_per_diag = pos_offset_limit - offset
        torch._check(
            nnz_per_diag >= 0,
            lambda: f"Trying to create tensor with negative dimension {nnz_per_diag}",
        )
        nnz += nnz_per_diag
    return nnz


def _spdiags_max_nnz_per_diag(rows: int, cols: int, diag_cols: int) -> int:
    non_positive_offset_max = min(rows, diag_cols)
    positive_offset_max = max(min(cols, diag_cols) - 1, 0)
    return max(non_positive_offset_max, positive_offset_max, 0)


def _spdiags_dynamic_size(
    fake_mode: FakeTensorMode,
    func: OpOverload,
    *,
    max_size: int | None,
) -> torch.SymInt:
    if (
        fake_mode.shape_env is None
        or not fake_mode.shape_env.allow_dynamic_output_shape_ops
    ):
        raise DynamicOutputShapeException(func)

    nnz = fake_mode.shape_env.create_unbacked_symint()

    from torch.fx.experimental.symbolic_shapes import _constrain_range_for_size

    _constrain_range_for_size(nnz, min=0, max=max_size)
    return nnz


def _spdiags_is_static_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@register_op_impl(aten._spdiags.default)
def _spdiags(
    fake_mode: FakeTensorMode,
    func: OpOverload,
    diagonals: FakeTensorLike,
    offsets: FakeTensorLike,
    shape: Sequence[int],
    layout: torch.layout | None = None,
) -> FakeTensor:
    diagonals_dim = diagonals.dim()
    torch._check(diagonals_dim in (1, 2), lambda: "Diagonals must be vector or matrix")
    torch._check(len(shape) == 2, lambda: "Output shape must be 2d")

    offsets_dim = offsets.dim()
    torch._check(offsets_dim in (0, 1), lambda: "Offsets must be scalar or vector")

    n_diags = 1 if diagonals_dim == 1 else diagonals.size(0)
    n_offsets = 1 if offsets_dim == 0 else offsets.size(0)
    torch._check(
        n_diags == n_offsets,
        lambda: f"Number of diagonals ({n_diags}) does not match the number of offsets ({n_offsets})",
    )

    if layout is not None:
        torch._check(
            layout in (torch.sparse_coo, torch.sparse_csc, torch.sparse_csr),
            lambda: f"Only output layouts (Sparse, SparseCsc, SparseCsr) are supported, got {layout}",
        )
    torch._check(
        offsets.dtype == torch.long,
        lambda: f"Offset Tensor must have dtype Long but got {offsets.dtype}",
    )
    torch._check(
        diagonals.device.type == "cpu" and offsets.device.type == "cpu",
        lambda: "_spdiags is only implemented for CPU tensors",
    )

    rows, cols = shape
    diag_cols = diagonals.size(0) if diagonals_dim == 1 else diagonals.size(1)
    static_sizes = all(
        _spdiags_is_static_int(value) for value in (rows, cols, diag_cols, n_offsets)
    )

    static_offsets = _spdiags_static_offsets(offsets)
    if static_offsets is not None:
        torch._check(
            len(static_offsets) == len(set(static_offsets)),
            lambda: "Offset tensor contains duplicate values",
        )

    out_layout = layout if layout is not None else torch.sparse_coo
    if static_sizes and (rows == 0 or cols == 0):
        nnz = 0
    elif static_offsets is not None and static_sizes:
        nnz = _spdiags_nnz(static_offsets, rows, cols, diag_cols)  # type: ignore[arg-type]
    elif static_sizes and n_offsets == 0:
        nnz = 0
    else:
        max_nnz = (
            n_offsets * _spdiags_max_nnz_per_diag(rows, cols, diag_cols)
            if static_sizes
            else None
        )
        nnz = _spdiags_dynamic_size(fake_mode, func, max_size=max_nnz)  # type: ignore[arg-type]

    with in_kernel_invocation_manager(fake_mode):
        values = torch.empty((nnz,), dtype=diagonals.dtype, device="meta")
        with torch.sparse.check_sparse_tensor_invariants(False):
            if out_layout == torch.sparse_coo:
                indices = torch.empty((2, nnz), dtype=torch.long, device="meta")
                is_coalesced = nnz <= 1 if isinstance(nnz, int) else False
                out = torch.ops.aten._sparse_coo_tensor_unsafe.default(
                    indices,
                    values,
                    shape,
                    device=torch.device("meta"),
                    is_coalesced=is_coalesced,
                )
            elif out_layout == torch.sparse_csr:
                crow_indices = torch.empty((rows + 1,), dtype=torch.long, device="meta")
                col_indices = torch.empty((nnz,), dtype=torch.long, device="meta")
                out = torch.ops.aten._sparse_compressed_tensor_unsafe.default(
                    crow_indices,
                    col_indices,
                    values,
                    shape,
                    layout=out_layout,
                    device=torch.device("meta"),
                )
            else:
                ccol_indices = torch.empty((cols + 1,), dtype=torch.long, device="meta")
                row_indices = torch.empty((nnz,), dtype=torch.long, device="meta")
                out = torch.ops.aten._sparse_compressed_tensor_unsafe.default(
                    ccol_indices,
                    row_indices,
                    values,
                    shape,
                    layout=out_layout,
                    device=torch.device("meta"),
                )

    return fake_mode.fake_tensor_converter.from_meta_and_device(
        fake_mode, out, diagonals.device
    )


@register_op_impl(aten._to_dense.default)
def _to_dense(
    fake_mode: FakeTensorMode,
    func: OpOverload,
    self: FakeTensor,
    dtype: torch.dtype | None = None,
    masked_grad: bool | None = None,
) -> FakeTensor:
    maybe_mkldnn_out = maybe_to_dense_mkldnn(fake_mode, self, dtype, masked_grad)
    if maybe_mkldnn_out is not NotImplemented:
        return typing_cast(FakeTensor, maybe_mkldnn_out)

    if self.layout is torch.sparse_coo:
        if dtype is not None:
            raise RuntimeError("dtype argument is not supported by sparse_to_dense")
        with in_kernel_invocation_manager(fake_mode):
            out = torch.empty(
                tuple(self.shape),
                dtype=self.dtype,
                device="meta",
            )
        return FakeTensor(fake_mode, out, self.fake_device)

    with in_kernel_invocation_manager(fake_mode):
        out = func(self, dtype=dtype, masked_grad=masked_grad)
    return fake_mode.fake_tensor_converter.from_meta_and_device(
        fake_mode, out, self.fake_device
    )


# index.Tensor data-dependent in only some conditions
@register_op_impl(
    lambda func: torch.Tag.dynamic_output_shape in func.tags
    and func
    not in [aten.index.Tensor, aten.nonzero.default, aten.repeat_interleave.Tensor]
)
def dyn_shape(
    fake_mode: FakeTensorMode, func: OpOverload, *args: Any, **kwargs: Any
) -> None:
    raise DynamicOutputShapeException(func)


def _unique(
    fake_mode: FakeTensorMode,
    func: OpOverload,
    arg: FakeTensor,
    dim: int | None,
    sorted: bool = True,
    return_inverse: bool = False,
    return_counts: bool = False,
    *,
    unique_consecutive: bool = False,
) -> tuple[FakeTensor, FakeTensor, FakeTensor]:
    if (
        fake_mode.shape_env is None
        or not fake_mode.shape_env.allow_dynamic_output_shape_ops
    ):
        # Without symints/symfloats, cannot handle this
        raise DynamicOutputShapeException(func)

    nnz = arg.unique_consecutive_memo if unique_consecutive else arg.unique_memo

    # Do not use a memo for unique_dim
    if dim is not None or nnz is None:
        # Avoid importing sympy at a module level
        from torch.fx.experimental.symbolic_shapes import (
            _constrain_range_for_size,
            has_free_symbols,
        )

        if not has_free_symbols(arg.numel()) and arg.numel() == 0:
            # If numel is zero, then the output size must be zero.
            # In this case, we must not allocate an unbacked SymInt,
            # because if we do, it will immediately get refined to
            # zero, but this will be inconsistent with size oblivious
            # tests (which will continue to claim that the unbacked
            # symint cannot equal zero).  We could also unconditionally
            # allocate an unbacked SymInt and not refine its range,
            # but this seems more precise.
            nnz = 0
        else:
            nnz = fake_mode.shape_env.create_unbacked_symint()

            maxval = sys.maxsize - 1

            numel = arg.numel() if dim is None else arg.size(dim)
            if not has_free_symbols(numel):
                maxval = int(numel)

            _constrain_range_for_size(nnz, max=maxval)

        if dim is None:
            if unique_consecutive:
                arg.unique_consecutive_memo = nnz  # pyrefly: ignore[bad-assignment]
            else:
                arg.unique_memo = nnz  # pyrefly: ignore[bad-assignment]

    if dim is None:
        # pyrefly: ignore[no-matching-overload]
        ret = [arg.new_empty((nnz,))]
    else:
        # pyrefly: ignore[no-matching-overload]
        ret = [arg.new_empty(*arg.shape[:dim], nnz, *arg.shape[dim + 1 :])]

    return_if_dim_and_cpu = dim is not None and arg.fake_device == torch.device("cpu")
    if return_inverse or return_if_dim_and_cpu:
        inverse = arg.new_empty(
            arg.shape if dim is None else (arg.shape[dim],), dtype=torch.int64
        )
    else:
        inverse = arg.new_empty(0, dtype=torch.int64)
    ret.append(inverse)

    if return_counts or return_if_dim_and_cpu:
        counts = arg.new_empty(
            ret[0].shape if dim is None else (ret[0].shape[dim],), dtype=torch.int64
        )
    else:
        counts = arg.new_empty(0, dtype=torch.int64)
    ret.append(counts)

    return tuple(ret)


@register_op_impl(aten._unique2.default)
def unique2(
    fake_mode: FakeTensorMode,
    func: OpOverload,
    arg: FakeTensor,
    sorted: bool = True,
    return_inverse: bool = False,
    return_counts: bool = False,
) -> tuple[FakeTensor, FakeTensor, FakeTensor]:
    return _unique(fake_mode, func, arg, None, sorted, return_inverse, return_counts)


@register_op_impl(aten._unique.default)
def unique(
    fake_mode: FakeTensorMode,
    func: OpOverload,
    arg: FakeTensor,
    sorted: bool = True,
    return_inverse: bool = False,
) -> tuple[FakeTensor, FakeTensor]:
    uniques, inverse, _counts = _unique(
        fake_mode, func, arg, None, sorted, return_inverse, False
    )
    return uniques, inverse


@register_op_impl(aten.select.int)
def meta_select(
    fake_mode: FakeTensorMode,
    func: OpOverload,
    self: FakeTensor,
    dim: int,
    index: IntLikeType,
) -> FakeTensor:
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    if self.is_sparse:
        return NotImplemented

    ndim = self.dim()
    torch._check_index(
        ndim != 0,
        lambda: "select() cannot be applied to a 0-dim tensor.",
    )

    dim = dim if dim >= 0 else dim + ndim
    size = self.size(dim)

    if guard_or_false(index >= size) or guard_or_false(index < -size):
        torch._check_index(
            False,
            lambda: f"select(): index {index} out of range for tensor of size "
            f"{list(self.size())} at dimension {dim}",
        )

    new_size = list(self.size())
    new_stride = list(self.stride())

    new_storage_offset = None
    if guard_or_false(index >= 0):
        new_storage_offset = self.storage_offset() + index * new_stride[dim]
    elif guard_or_false(index < 0):
        new_storage_offset = self.storage_offset() + (index + size) * new_stride[dim]

    if new_storage_offset is None:
        if fake_mode.shape_env is None or (
            not fake_mode.shape_env.allow_scalar_outputs
            and not fake_mode.allow_scalar_outputs
        ):
            raise DataDependentOutputException(func)

        # index is data-dependent, we do not know which index we are accessing it could be index or index+size!
        # we assign a new data-dependent symbol for the storage offset.
        new_storage_offset = fake_mode.shape_env.create_unbacked_symint()

    del new_size[dim]
    del new_stride[dim]
    if new_storage_offset is None:
        raise AssertionError("new_storage_offset must not be None")
    # pyrefly: ignore[bad-return]
    return self.as_strided(new_size, new_stride, new_storage_offset)


@register_op_impl(aten.unique_dim.default)
def unique_dim(
    fake_mode: FakeTensorMode,
    func: OpOverload,
    arg: FakeTensor,
    dim: int,
    sorted: bool = True,
    return_inverse: bool = False,
    return_counts: bool = False,
) -> tuple[FakeTensor, FakeTensor, FakeTensor]:
    return _unique(
        fake_mode,
        func,
        arg,
        # normalize dim to be non-negative
        dim if dim >= 0 else dim % max(arg.ndim, 1),
        sorted,
        return_inverse,
        return_counts,
    )


@register_op_impl(aten.unique_consecutive.default)
def unique_consecutive(
    fake_mode: FakeTensorMode,
    func: OpOverload,
    arg: FakeTensor,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: int | None = None,
) -> tuple[FakeTensor, FakeTensor, FakeTensor]:
    return _unique(
        fake_mode,
        func,
        arg,
        dim,
        False,
        return_inverse,
        return_counts,
        unique_consecutive=True,
    )


# This function is python match of computeStride_impl in TensorUtils.cpp
def _compute_stride(
    old_shape: Sequence[IntLikeType],
    old_stride: Sequence[IntLikeType],
    new_shape: Sequence[IntLikeType],
    size_oblivious: bool = False,
) -> list[IntLikeType] | None:
    from torch.fx.experimental.symbolic_shapes import (
        guard_or_false,
        guard_or_true,
        sym_eq,
    )

    def maybe_guard_or_false(x: Any) -> Any:
        if size_oblivious:
            return guard_or_false(x)

        return x

    def maybe_guard_or_true(x: Any) -> Any:
        if size_oblivious:
            return guard_or_true(x)

        return x

    if len(old_shape) == 0:
        return [1] * len(new_shape)

    numel = reduce(operator.mul, old_shape, 1)
    zero_numel = maybe_guard_or_false(numel == 0)
    if zero_numel and maybe_guard_or_false(sym_eq(old_shape, new_shape)):
        return list(old_stride)

    new_stride: list[IntLikeType] = [0] * len(new_shape)

    if zero_numel:
        for view_d in range(len(new_shape) - 1, -1, -1):
            if view_d == len(new_shape) - 1:
                new_stride[view_d] = 1
            else:
                new_stride[view_d] = (
                    max(new_shape[view_d + 1], 1) * new_stride[view_d + 1]
                )
        return new_stride

    view_d = len(new_shape) - 1
