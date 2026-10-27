                a,
                start_idx,
                count_symints=count_symints,
                with_memory_format=with_memory_format,
            )
            infos.append(subclass_meta)
            cnt = subclass_meta.arg_count
        else:
            infos.append(
                PlainTensorMeta(
                    idx,
                    memory_format=maybe_suggest_memory_format(a, with_memory_format),
                )
            )
            cnt = 1
        idx += cnt
    return infos


def enumerate_filter_symints(lst: Iterable[IntLikeType]) -> list[tuple[int, SymInt]]:
    # Capture all SymInts from the iterable.
    def symint_check(s: IntLikeType) -> TypeGuard[SymInt]:
        return isinstance(s, SymInt) and not s.node.is_nested_int()

    return [(i, s) for i, s in enumerate(lst) if symint_check(s)]


def compute_symint_placeholders(lst: Iterable[None | int | SymInt]) -> list[bool]:
    # Non-nested symints are replaced with None in `make_runtime_safe()`
    return [s is None for s in lst]


# Intended to make it easier to define function that is
# either (AOTInput -> AOTInput) or (AOTOutput -> AOTOutput)
# but not the other combos
AOTDescriptor = TypeVar("AOTDescriptor", AOTInput, AOTOutput)


# This function takes in a pytree of arguments and unwraps any tensor
# subclasses.
#
# NOTE: The reason for "append_symints":
#
# * At compile time: we append extra symint args when unwrapping primals
# (but not tangents, because they should always share symints with primals).
# We also append extra symints when unwrapping the subclass outputs of the
# traced function, so we can return them as extra outputs
#
# * At runtime: we similarly append subclass sizes when we unwrap subclass
# primals (but not tangents) on entry to the forward. See the runtime version of
# this function below.
def unwrap_tensor_subclasses(
    wrapped_args: list[FxValue],
    wrapped_args_descs: Sequence[AOTDescriptor],
    *,
    append_symints: bool,
) -> tuple[list[FxValue], list[AOTDescriptor]]:
    def _maybe_fakeify_opaque(v: Any) -> Any:
        # Registered opaque types need to be wrapped as FakeScriptObject for
        # compile-time FX tracing (proxy slot tracking, hashability, etc.).
        if isinstance(v, CustomClassBase):
            from torch._guards import detect_fake_mode
            from torch._library.fake_class_registry import maybe_to_fake_obj
            from torch._library.opaque_object import is_custom_class

            fake_mode = detect_fake_mode()
            if fake_mode is not None and is_custom_class(type(v)):
                return maybe_to_fake_obj(fake_mode, v)
        return v

    def flatten_subclass(
        t: FxValue,
        desc: AOTDescriptor,
        *,
        out: tuple[list[FxValue], list[AOTDescriptor]],
    ) -> None:
        # unwrap a subclass into plain tensors and their size/stride if "append_symint"
        # is True
        if not is_traceable_wrapper_subclass(t):
            out[0].append(_maybe_fakeify_opaque(t))
            out[1].append(desc)
            if (
                append_symints
                and isinstance(t, Tensor)
                and isinstance(
                    desc, (SubclassGetAttrAOTInput, SubclassGetAttrAOTOutput)
                )
            ):
                sizes = enumerate_filter_symints(t.size())
                strides = enumerate_filter_symints(t.stride())
                out[0].extend(s for _, s in sizes)
                out[0].extend(s for _, s in strides)
                if isinstance(desc, AOTInput):
                    out[1].extend(SubclassSizeAOTInput(desc, i) for i, _ in sizes)
                    out[1].extend(SubclassStrideAOTInput(desc, i) for i, _ in strides)
                else:
                    out[1].extend(SubclassSizeAOTOutput(desc, i) for i, _ in sizes)
                    out[1].extend(SubclassStrideAOTOutput(desc, i) for i, _ in strides)
            return

        attrs, _ = t.__tensor_flatten__()

        SubclassGetAttr: Callable[[AOTInput | AOTOutput, str], AOTDescriptor]
        SubclassSize: Callable[[AOTInput | AOTOutput, int], AOTDescriptor]
        SubclassStride: Callable[[AOTInput | AOTOutput, int], AOTDescriptor]
        if isinstance(desc, AOTInput):
            SubclassGetAttr = SubclassGetAttrAOTInput  # type: ignore[bad-assignment]
            SubclassSize = SubclassSizeAOTInput  # type: ignore[bad-assignment]
            SubclassStride = SubclassStrideAOTInput  # type: ignore[bad-assignment]
        else:
            SubclassGetAttr = SubclassGetAttrAOTOutput  # type: ignore[bad-assignment]
            SubclassSize = SubclassSizeAOTOutput  # type: ignore[bad-assignment]
            SubclassStride = SubclassStrideAOTOutput  # type: ignore[bad-assignment]

        for attr in attrs:
            inner_value = getattr(t, attr)
            n_desc: Any = SubclassGetAttr(desc, attr)
            flatten_subclass(inner_value, n_desc, out=out)

        if append_symints:
            sizes = enumerate_filter_symints(t.size())
            strides = enumerate_filter_symints(t.stride())
            out[0].extend(s for _, s in sizes)
            out[0].extend(s for _, s in strides)
            out[1].extend(SubclassSize(desc, i) for i, _ in sizes)
            out[1].extend(SubclassStride(desc, i) for i, _ in strides)

    xs_inner: list[FxValue] = []
    descs_inner: list[AOTDescriptor] = []

    for x, desc in zip(wrapped_args, wrapped_args_descs):
        flatten_subclass(x, desc, out=(xs_inner, descs_inner))

    return xs_inner, descs_inner


# subclass_metas is needed at runtime to compute which indices are symints in
# the outer_size/outer_stride
def runtime_unwrap_tensor_subclasses(
    wrapped_args: list[Tensor | int],
    *,
    append_symints: bool,
    subclass_metas: list[PlainTensorMeta | SubclassCreationMeta] | None = None,
) -> list[int | Tensor | SymInt | CustomClassBase]:
    def flatten_subclass(
        x: Tensor | TraceableWrapperSubclass,
        subclass_meta: PlainTensorMeta | SubclassCreationMeta | OpaqueMeta | None,
        *,
        out: list[CustomClassBase | SymInt | Tensor | int],
    ) -> list[CustomClassBase | SymInt | Tensor | int]:
        def append_plain_tensor_symints(tensor: Tensor, meta: PlainTensorMeta) -> None:
            size = tensor.size()
            if len(size) != len(meta.size_symbol_placeholders):
                raise AssertionError(
                    f"size length mismatch: {len(size)} != {len(meta.size_symbol_placeholders)}"
                )
            out.extend(
                [
                    r
                    for (r, is_symint) in zip(size, meta.size_symbol_placeholders)
                    if is_symint
                ]
            )

            stride = tensor.stride()
            if len(stride) != len(meta.stride_symbol_placeholders):
                raise AssertionError(
                    f"stride length mismatch: {len(stride)} != {len(meta.stride_symbol_placeholders)}"
                )
            out.extend(
                [
                    r
                    for (r, is_symint) in zip(stride, meta.stride_symbol_placeholders)
                    if is_symint
                ]
            )

        if not is_traceable_wrapper_subclass(x):
            out.append(x)
            if append_symints and isinstance(subclass_meta, PlainTensorMeta):
                append_plain_tensor_symints(x, subclass_meta)
            return out

        if not isinstance(x, Tensor):
            raise AssertionError(f"expected Tensor, got {type(x)}")
        if not isinstance(subclass_meta, SubclassCreationMeta):
            raise AssertionError("subclass_meta should be a SubclassCreationMeta")

        attrs, _ = x.__tensor_flatten__()

        for attr in attrs:
            inner_value = getattr(x, attr)
            match inner_value:
                case CustomClassBase():
                    out.append(inner_value)
                case Tensor():
                    inner_meta = subclass_meta.attrs.get(attr)
                    flatten_subclass(inner_value, inner_meta, out=out)
                case _:
                    raise AssertionError(
                        f"expected Tensor or CustomClassBase, got {type(inner_value)}"
                    )

        if append_symints:
            # outer_size
            size = x.size()
            symint_placeholders = compute_symint_placeholders(subclass_meta.outer_size)
            if len(size) != len(symint_placeholders):
                raise AssertionError(
                    f"size length mismatch: {len(size)} != {len(symint_placeholders)}"
                )
            out.extend(
                [r for (r, is_symint) in zip(size, symint_placeholders) if is_symint]
            )

            # outer_stride
            stride = x.stride()
            symint_placeholders = compute_symint_placeholders(
                subclass_meta.outer_stride
            )
            if len(stride) != len(symint_placeholders):
                raise AssertionError(
                    f"stride length mismatch: {len(stride)} != {len(symint_placeholders)}"
                )
            out.extend(
                [r for (r, is_symint) in zip(stride, symint_placeholders) if is_symint]
            )
        return out

    xs_inner: list[int | Tensor | SymInt | CustomClassBase] = []

    if append_symints:
        if subclass_metas is None:
            raise AssertionError(
                "subclass_metas must not be None when append_symints is True"
            )

    for idx, x in enumerate(wrapped_args):
        if not is_traceable_wrapper_subclass(x):
            xs_inner.append(x)
            continue

        if subclass_metas is None:
            get_plain_tensors(x, out=xs_inner)
        else:
            subclass_meta = subclass_metas[idx]
            if not isinstance(subclass_meta, SubclassCreationMeta):
                raise AssertionError(
                    f"expected SubclassCreationMeta, got {type(subclass_meta)}"
                )
            flatten_subclass(x, subclass_meta, out=xs_inner)

    return xs_inner


def unwrap_tensor_subclasses_with_indices_to_original(
    wrapped_args: list[Any],
) -> tuple[list[Any], list[int]]:
    ret_unwrapped = []
    ret_indices_to_original = []
    for i, a in enumerate(wrapped_args):
        a_unwrapped, _ = unwrap_tensor_subclasses(
            [a], [DummyAOTInput(9999)], append_symints=False
        )
        ret_unwrapped.extend(a_unwrapped)
        n = len(a_unwrapped)
        ret_indices_to_original.extend([i] * n)

    return ret_unwrapped, ret_indices_to_original


def remap_unwrapped_subclass_arg_indices(
    wrapped_args: list[Any], static_input_indices: list[int]
) -> list[int]:
    def static_arg_slots(arg: Any, *, include_plain_tensor_symints: bool) -> list[bool]:
        # Forwarded SymInts occupy flattened arg positions, but static input indices
        # describe tensor/opaque slots whose value should not be copied.
        if not is_traceable_wrapper_subclass(arg):
            if include_plain_tensor_symints and isinstance(arg, Tensor):
                num_symints = len(enumerate_filter_symints(arg.size())) + len(
                    enumerate_filter_symints(arg.stride())
                )
                return [True] + [False] * num_symints
            return [True]

        slots = []
        attrs, _ = arg.__tensor_flatten__()
        for attr in attrs:
            inner_value = getattr(arg, attr)
            match inner_value:
                case CustomClassBase():
                    slots.append(True)
                case Tensor():
                    slots.extend(
                        static_arg_slots(inner_value, include_plain_tensor_symints=True)
                    )
                case _:
                    raise AssertionError(
                        f"expected Tensor or CustomClassBase, got {type(inner_value)}"
                    )

        slots.extend(
            [False]
            * (
                len(enumerate_filter_symints(arg.size()))
                + len(enumerate_filter_symints(arg.stride()))
            )
        )
        return slots

    static_input_indices_set = set(static_input_indices)
    new_ind = 0
    remapped_static_indices = []
    for i, arg in enumerate(wrapped_args):
        slots = (
            static_arg_slots(arg, include_plain_tensor_symints=False)
            if is_traceable_wrapper_subclass(arg)
            else [True]
        )

        for is_static_arg_slot in slots:
            if i in static_input_indices_set and is_static_arg_slot:
                remapped_static_indices.append(new_ind)

            new_ind += 1

    return remapped_static_indices


# Turns a flattened list of tensor arguments into (maybe) subclass tensors.
# This function is used both at trace time and runtime, so we have an is_runtime flag telling us which context we're in.
def wrap_tensor_subclasses(
    unwrapped_args: Sequence[Any],
    *,
    subclass_metas: list[PlainTensorMeta | SubclassCreationMeta],
    num_fw_outs_saved_for_bw: int | None = None,
    included_subclass_symints: bool = False,
    is_runtime: bool = False,
    make_subclass_override: Callable[..., Any] | None = None,
) -> tuple[Any, ...]:
    # pyrefly: ignore [implicit-any]
    wrapped_args = []
    num_args_tallied = 0
    for subclass_meta in subclass_metas:
        if isinstance(subclass_meta, PlainTensorMeta):
            wrapped_args.append(unwrapped_args[subclass_meta.unwrapped_idx])
            num_args_tallied += subclass_meta.arg_count
        else:
            if not isinstance(subclass_meta, SubclassCreationMeta):
                raise AssertionError(
                    f"expected SubclassCreationMeta, got {type(subclass_meta)}"
                )
            if subclass_meta.included_subclass_symints != included_subclass_symints:
                raise AssertionError(
                    f"included_subclass_symints mismatch: {subclass_meta.included_subclass_symints} != {included_subclass_symints}"
                )

            if make_subclass_override:
                wrapped_args.append(
                    make_subclass_override(subclass_meta, is_runtime, unwrapped_args)
                )
            else:
                wrapped_args.append(
                    subclass_meta.creation_fn(
                        unwrapped_args,
                        is_runtime=is_runtime,
                    )
                )
            num_args_tallied += subclass_meta.arg_count

    # Note: [Partitioner handling for Subclasses, Part 2]
    # At the beginning of AOTAutograd, we collect metadata on the inputs and outputs of the user fw,
    # to figure out which inputs/outputs are subclasses, and how to reconstruct the subclasses after flattening them.
    #
    # When this function is called at runtime in the forward,
    # we have been passed a list of (flattened) dense-tensor fw-outs, and need to reconstruct any subclass fw outs.
    #
    # One reasonable question that you should ask: when should the dense_tensor -> subclass_tensor wrapping happen?
    # Answer: we do it **inside of our compiled autograd.Function**.
    # This seems like morally the right place: autograd happens above subclass desugaring,
    # so autograd should see actual tensor subclasses at runtime, and not flattened dense tensors.
    #
    # This causes a tricky interaction though: when we run the min-cut partitioner to divvy up the joint graph
    # into a forward and backward graph, we end up with some activations that show up as extra outputs
    # in the compiled forward graph, that are **not** user outputs.
    # These activations are not visible to the user, and so there's no need for us to wrap them back into subclasses.
    #
    # On top of that, when we first computed subclass metadata (in `run_functionalized_fw_and_collect_metadata`),
    # we computed subclass metadata on every forward output, but this did **not** include activations
    # created by the partitioner.
    # as a result, `unwrapped_args` here will correspond to (*unwrapped_user_fw_outs, *activations),
    # but `subclass_metas` will only correspond to subclass metadata on `user_fw_outs`.
    # We then need to make sure that we return (*wrapped_user_fw_outs, *activations).
    if num_fw_outs_saved_for_bw is not None:
        if len(unwrapped_args) != num_args_tallied + num_fw_outs_saved_for_bw:
            raise AssertionError(
                f"Expected the number actual unwrapped-subclass outputs {len(unwrapped_args)} to equal "
                f"the number of args calculated from subclasses ({num_args_tallied}) plus the number of "
                f"additional activations saved for the backward pass ({num_fw_outs_saved_for_bw})"
            )
        activations = unwrapped_args[num_args_tallied:]
        if isinstance(wrapped_args, tuple) and isinstance(activations, tuple):
            return wrapped_args + activations
        return tuple(list(wrapped_args) + list(activations))
    else:
        if len(unwrapped_args) != num_args_tallied:
            raise AssertionError(
                f"Expected {len(unwrapped_args)} == {num_args_tallied}"
            )
        return tuple(wrapped_args)


# Given a bunch of "dense" tensor arguments, this function (potentially) wraps them into tensor subclasses.
# This function carefully handles the inference vs. joint cases:
# - when is_joint_structure is True, args is (primals, tangents)
# - when is_joint_structure is False, args is [*primals]
def wrap_tensor_subclasses_maybe_joint(
    unwrapped_args: Sequence[Any],
    *,
    is_joint_structure: bool,
    meta: ViewAndMutationMeta,
) -> tuple[Any, ...]:
    # Since this function is reused for both inference and joint graphs,
    if is_joint_structure:
        if not (isinstance(unwrapped_args, tuple) and len(unwrapped_args) == 2):
            unwrapped_len = (
                len(unwrapped_args)
                if isinstance(unwrapped_args, (tuple, list))
                else "N/A"
            )
            raise AssertionError(
                f"expected tuple of length 2 for joint structure, "
                f"got {type(unwrapped_args)} with length {unwrapped_len}"
            )
        if not (
            isinstance(unwrapped_args[0], (tuple, list))
            and isinstance(unwrapped_args[1], (tuple, list))
        ):
            raise AssertionError(
                f"expected primals and tangents to be tuple or list, got {type(unwrapped_args[0])} and {type(unwrapped_args[1])}"
            )
        primals, tangents = unwrapped_args[0], unwrapped_args[1]
        wrapped_primals = wrap_tensor_subclasses(
            primals,
            subclass_metas=meta.subclass_inp_meta,
            included_subclass_symints=True,
        )
        wrapped_tangents = wrap_tensor_subclasses(
            tangents,
            subclass_metas=meta.subclass_tangent_meta,
            included_subclass_symints=False,
        )
        return (wrapped_primals, wrapped_tangents)
    else:
        wrapped_args = wrap_tensor_subclasses(
            unwrapped_args,
            subclass_metas=meta.subclass_inp_meta,
            included_subclass_symints=True,
        )
        return wrapped_args


def compute_inner_mutated_inp_indices_from_subclass_meta(
    fw_metadata: ViewAndMutationMeta,
    inner_metadata: ViewAndMutationMeta,
) -> list[int]:
    # Note: [Recomputing subclass mutation handling]
    #
    # Generally, if a subclass requires grad, its components will not require grad.
    # But for the purposes of tracking returned tensors, we should treat those component
    # tensors as if they require grad.
    #
    # For example, if the subclass tensor requires grad and will be mutated in a way that
    # requires us to handle the mutation outside of the graph, we need to return it
    # from the forward graph. The inner_meta data won't consider the component tensors
    # as if they need to be returned, because they don't require grad; but really, we
    # should handle those tensors the same way we handle the subclass tensor itself; i.e.
    # if we'd include the subclass tensor as part of the outputs, then we should also
    # include the component tensors.
    #
    # To do this, we patch num_mutated_inp_runtime_indices below by expanding the inputs
    # from the outer subclass tensors and propagating

    updated_input_info = []
    inner_idx = 0
    if not fw_metadata.subclass_inp_meta:
        # Sometimes we don't have subclass info, e.g. synthetic_base codepaths
        return inner_metadata.mutated_inp_runtime_indices
    if len(fw_metadata.subclass_inp_meta) != len(fw_metadata.input_info):
        raise AssertionError(
            f"subclass_inp_meta length ({len(fw_metadata.subclass_inp_meta)}) != input_info length ({len(fw_metadata.input_info)})"
        )
    for outer_idx, inp_meta in enumerate(fw_metadata.subclass_inp_meta):
        if isinstance(inp_meta, PlainTensorMeta):
            if outer_idx >= len(fw_metadata.input_info):
                raise AssertionError(
                    f"outer_idx ({outer_idx}) >= len(fw_metadata.input_info) ({len(fw_metadata.input_info)})"
                )
            if inner_metadata is not None:
                if inner_idx >= len(inner_metadata.input_info):
                    raise AssertionError(
                        f"inner_idx ({inner_idx}) >= len(inner_metadata.input_info) ({len(inner_metadata.input_info)})"
                    )
                if (
                    inner_metadata.input_info[inner_idx]
                    != fw_metadata.input_info[outer_idx]
                ):
                    raise AssertionError(
                        f"input_info mismatch at inner_idx={inner_idx}, outer_idx={outer_idx}: "
                        f"{inner_metadata.input_info[inner_idx]} != {fw_metadata.input_info[outer_idx]}"
                    )
            updated_input_info.append(fw_metadata.input_info[outer_idx])
            inner_idx += 1
        else:
            if inp_meta.original_subclass is None:
                raise AssertionError(
                    "inp_meta.original_subclass must not be None for SubclassCreationMeta"
                )
            for _ in range(inp_meta.arg_count):
                updated_input_info.append(fw_metadata.input_info[outer_idx])
                inner_idx += 1
    if inner_metadata is not None:
        if len(inner_metadata.input_info) != len(updated_input_info):
            raise AssertionError(
                f"inner_metadata.input_info length ({len(inner_metadata.input_info)}) "
                f"!= updated_input_info length ({len(updated_input_info)})"
            )

    return [
        i
        for i, inp in enumerate(updated_input_info)
        if inp.mutation_type == MutationType.MUTATED_OUT_GRAPH
    ]
<|endoftext|>"""
This module is responsible for transforming functions to be traced into a form
that is easier for the downstream infra (e.g. Autograd, FX, AOTAutograd analysis)
to handle.

It does so by:
1. functionalization (including RNG functionalzation)
2. creating a joint graph when required
3. transforming mutations into extra outputs
4. dispatching subclasses
"""

import typing
import warnings
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager, ExitStack, nullcontext
from dataclasses import dataclass
from typing import Any, TypeVar
from unittest.mock import patch

import torch
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree
from torch import Tensor
from torch._custom_class_base import CustomClassBase
from torch._decomp.decompositions_for_rng import PhiloxStateTracker
from torch._guards import detect_fake_mode
from torch._prims_common import CUDARngStateHelper
from torch.fx.experimental.proxy_tensor import (
    _proxy_tensor_disable_update_tensor_tracker,
    get_proxy_mode,
    maybe_disable_thunkify,
    maybe_enable_thunkify,
)
from torch.fx.experimental.symbolic_shapes import (
    guard_or_true,
    PropagateUnbackedSymInts,
    sym_eq,
)
from torch.nn.utils import stateless
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._pytree import TreeSpec

from .. import config
from .collect_metadata_analysis import run_functionalized_fw_and_collect_metadata
from .descriptors import (
    AOTInput,
    AOTOutput,
    BackwardTokenAOTOutput,
    ForwardTokenAOTInput,
    ForwardTokenAOTOutput,
    GradAOTOutput,
    InputMutationAOTOutput,
    IntermediateBaseAOTOutput,
    PhiloxBackwardBaseOffsetAOTInput,
    PhiloxBackwardSeedAOTInput,
    PhiloxForwardBaseOffsetAOTInput,
    PhiloxForwardSeedAOTInput,
    PhiloxUpdatedBackwardOffsetAOTOutput,
    PhiloxUpdatedForwardOffsetAOTOutput,
)
from .functional_utils import (
    _check_if_mutation_can_be_in_graph,
    are_all_mutations_hidden_from_autograd,
    are_all_mutations_under_no_grad_or_inference_mode,
    from_fun,
    has_data_mutation,
    has_metadata_mutation,
    is_fun,
    sync_functional_tensor,
    to_fun,
    was_inductor_storage_resized,
)
from .logging_utils import setup_stacktrace_preservation_hooks
from .schemas import (
    AOTConfig,
    FxValue,
    InputAliasInfo,
    JointTraceFn,
    MutationType,
    OutputType,
    PreppedForAutogradTraceFn,
    SubclassMeta,
    SubclassTracingInfo,
    TraceFn,
    ViewAndMutationMeta,
)
from .subclass_utils import (
    create_subclass_meta,
    remap_unwrapped_subclass_arg_indices,
    requires_subclass_dispatch,
    unwrap_tensor_subclasses,
    wrap_tensor_subclasses_maybe_joint,
)
from .utils import (
    _is_tangent,
    call_and_expect_output_descs,
    maybe_to_fresh_input,
    simple_wraps,
    without_output_descs,
)


# This function returns a new function that returns mutated inputs as outputs.
# if keep_data_input_mutations is set, then we assume that data-only mutations
# will be left in the graph, and we only return metadata-mutated inputs as outputs.
def fn_input_mutations_to_outputs(
    fn: Callable[..., Any],
    args_descs: list[AOTInput],
    meta: ViewAndMutationMeta,
    keep_data_input_mutations: bool,
) -> Any:
    @simple_wraps(fn)
    def inner_fn(*args: FxValue) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        outs, outs_descs = call_and_expect_output_descs(fn, args)
        if len(meta.output_info) != len(outs):
            raise AssertionError(
                f"output_info length ({len(meta.output_info)}) != outs length ({len(outs)})"
            )
        # The compiled fw will return mutated input tensors, *including* metadata-only mutation.
        # However, if keep_data_input_mutations is set, the compiled fw only needs to return metadata-mutated inputs.
        # (because data-only input mutations are handled directly in the compiled graph)
        mutated_input_pairs = [
            (x, InputMutationAOTOutput(src))
            for (i, (x, src)) in enumerate(zip(args, args_descs))
            if i in meta.mutated_inp_runtime_indices
        ]
        if mutated_input_pairs:
            mutated_inputs_to_return, mutated_inputs_to_return_descs = zip(
                *mutated_input_pairs
            )
        else:
            mutated_inputs_to_return, mutated_inputs_to_return_descs = (), ()
        return (
            (*mutated_inputs_to_return, *outs),
            (*mutated_inputs_to_return_descs, *outs_descs),
        )

    return inner_fn


@contextmanager
def disable_autocast() -> Generator[None, None, None]:
    with ExitStack() as stack:
        autocast_enabled_devices = torch._C._autocast_supported_devices()
        for device_type in autocast_enabled_devices:
            if hasattr(torch, device_type):
                stack.enter_context(torch.amp.autocast(device_type, enabled=False))
        yield


# This function takes in a fn with external aliasing and mutation,
# and returns a new fn with no external aliasing and mutation,
# as needed for autograd.
# The main transformations are:
# - Return mutated inputs as extra outputs
# - Clone mutated inputs that require gradients,
#   because autograd will require us to pass the pre-mutated inputs into autograd.grad
# - Return intermediate bases of outputs as additional outputs,
#   needed to appease autograd.Function
# The new function returns:
# (1) The updated outputs
# (2) A boolean mask of len(new_fn_outputs),
#     that can be used to tell autograd.grad which outputs should get tangents
#     if we trace the backward.
def fn_prepped_for_autograd(
    fn: TraceFn,
    args_descs: list[AOTInput],
    meta: ViewAndMutationMeta,
    aot_config: AOTConfig,
) -> PreppedForAutogradTraceFn:
    @simple_wraps(fn)
    def inner_fn(
        *args: FxValue,
    ) -> tuple[tuple[list[FxValue], list[bool]], list[AOTOutput]]:
        args_maybe_cloned = [
            maybe_to_fresh_input(i, t, meta) for i, t in enumerate(args)
        ]

        outs, outs_descs = call_and_expect_output_descs(fn, args_maybe_cloned)  # type: ignore[arg-type]
        if not isinstance(outs, (tuple, list)):
            raise AssertionError(f"expected outs to be tuple or list, got {type(outs)}")
        outs = list(outs)
        if len(meta.output_info) != len(outs):
            raise AssertionError(
                f"output_info length ({len(meta.output_info)}) != outs length ({len(outs)})"
            )

        mutated_input_pairs = [
            (x, InputMutationAOTOutput(src))
            for (i, (x, src)) in enumerate(zip(args_maybe_cloned, args_descs))
            if i in meta.mutated_inp_runtime_indices
        ]
        if mutated_input_pairs:
            mutated_inputs_to_return, mutated_inputs_to_return_descs = zip(
                *mutated_input_pairs
            )
        else:
            mutated_inputs_to_return, mutated_inputs_to_return_descs = (), ()

        intermediate_bases = []
        intermediate_bases_descs = []
        for o, info, o_desc in zip(outs, meta.output_info, outs_descs):
            if info.output_type == OutputType.alias_of_intermediate_save_as_output:
                if not isinstance(o, torch.Tensor):
                    raise AssertionError(
                        f"Expected tensor for intermediate base, got {type(o)}"
                    )
                intermediate_bases.append(o._base)
                intermediate_bases_descs.append(IntermediateBaseAOTOutput(o_desc))

        if meta.num_intermediate_bases != len(intermediate_bases):
            raise AssertionError(
                f"num_intermediate_bases ({meta.num_intermediate_bases}) != len(intermediate_bases) ({len(intermediate_bases)})"
            )

        # the compiled forward should return (mutated_inputs, user_outs, intermediate_bases)
        fw_outs_to_return = *mutated_inputs_to_return, *outs, *intermediate_bases
        fw_outs_to_return_descs = (
            *mutated_inputs_to_return_descs,
            *outs_descs,
            *intermediate_bases_descs,
        )

        # Also return a boolean mask specifying which outputs to this function will be used as tangents
        mutated_inputs_grad_mask = [
            meta.input_info[meta.mutated_inp_runtime_indices[i]].mutates_data
            and meta.input_info[meta.mutated_inp_runtime_indices[i]].requires_grad
            for (i, x) in enumerate(mutated_inputs_to_return)
        ]

        # Pass any (non-aliased) outputs in as tangents, since they'll be returned as outputs in the fw
        # For outputs that are aliases of intermediates, we will have returned the output's _base as an output in the graph instead,
        # which we *should* send to grad()
        output_grad_mask = [
            meta.output_info[i].output_type
            in [
                OutputType.non_alias,
                OutputType.unsafe_view_alias,
                OutputType.custom_function_view,
            ]
            # Also, only tensor outputs should participate in the backward
            # (in particular, Symint outputs in the forward graph shouldn't get tangents)
            and issubclass(meta.output_info[i].raw_type, Tensor)
            and meta.output_info[i].requires_grad_for_backward
            for (i, x) in enumerate(outs)
        ]

        intermediate_base_grad_mask = [True for _ in range(len(intermediate_bases))]

        out_grad_mask = (
            mutated_inputs_grad_mask + output_grad_mask + intermediate_base_grad_mask
        )
        if len(out_grad_mask) != len(fw_outs_to_return):
            raise AssertionError(
                f"out_grad_mask length ({len(out_grad_mask)}) != fw_outs_to_return length ({len(fw_outs_to_return)})"
            )

        # Take care to grab and sync the updated inputs from primals_after_cloning (the inputs we actually mutate!)
        # and not primals (the preserved inputs, pre-mutation, that we pass to grad())
        # This is annoying: our joint function needs to be aware of functionalization
        # (syncing mutated inputs before calling autograd.grad())
        # In theory, we could make the autograd engine do this automatically, although that probably isn't any cleaner.
        if not aot_config.disable_functionalization:
            for arg in args_maybe_cloned:
                if not isinstance(arg, Tensor):
                    continue
                sync_functional_tensor(arg)

        # pyrefly: ignore[bad-return]
        return (fw_outs_to_return, out_grad_mask), (
            fw_outs_to_return_descs,
            out_grad_mask,
        )

    return inner_fn


@dataclass
class JointFnHandle:
    post_forward: Callable[..., Any] | None = None


# Given a fn, computes the joint.
# NOTE: fn is expects the following behavior:
# (1) fn() needs to return a tuple of (outs, mask),
#     where `mask` tells us which outputs are meant to have tangents.
#     we don't know this info automatically, because we don't actually want to blindly
#     compute tangents for every output that requires grad.
#     Specifically, outputs that alias inputs won't participate in the backward and get tangents.
# (2) fn() cannot mutate any inputs that require gradient.
#     otherwise, when we compute autograd.grad(), we will not take those input mutations into account
#     (the way this is handled is that we ensure any inputs that normally get mutated are cloned first)
def create_joint(
    fn: Callable[..., Any],
    primals_descs: list[AOTInput] | None = None,
    *,
    aot_config: AOTConfig,
) -> Callable[..., Any]:
    joint_fn_handle = JointFnHandle()

    # post_forward
    # NB: this type is inaccurate when primals_descs is None
    @simple_wraps(fn)
    def inner_fn(
        primals: list[FxValue], tangents: list[FxValue]
    ) -> tuple[
        tuple[list[FxValue], list[Tensor | None]],
        tuple[list[AOTOutput], list[AOTOutput | None]],
    ]:
        outs_descs = None
        if primals_descs is None:
            outs, tangent_mask = fn(*primals)
            if pytree.tree_any(lambda x: isinstance(x, AOTOutput), tangent_mask):
                raise AssertionError(
                    "tangent_mask should not contain AOTOutput instances"
                )
        else:
            (outs, tangent_mask), (outs_descs, _) = call_and_expect_output_descs(
                fn,
                primals,  # type: ignore[arg-type]
            )
        mode = get_proxy_mode()
        if mode is None:
            raise AssertionError("Expected non-None proxy mode")
        for node in mode.tracer.graph.nodes:
            if _is_tangent(node):
                node.meta["partitioner_tag"] = "is_backward"
            else:
                node.meta["partitioner_tag"] = "is_forward"

        # TODO: I think this hook can also be eliminated now
        if joint_fn_handle and joint_fn_handle.post_forward:
            joint_fn_handle.post_forward(primals)

        if len(tangent_mask) != len(outs):
            raise AssertionError(
                f"tangent_mask length ({len(tangent_mask)}) != outs length ({len(outs)})"
            )
        outs_to_grad = [
            o for needs_tangent, o in zip(tangent_mask, outs) if needs_tangent
        ]
        if len(outs_to_grad) != len(tangents):
            raise AssertionError(
                f"outs_to_grad length ({len(outs_to_grad)}) != tangents length ({len(tangents)})"
            )

        # Get the inputs that need gradients
        grad_primals: list[torch.Tensor] = []
        inputs_needs_grads = []
        # Note that we're not using primals here,
        # being carefully not to pass any mutated inputs into autograd.grad()
        for p in primals:
            if isinstance(p, Tensor) and p.requires_grad:
                inputs_needs_grads.append(True)
                if not isinstance(p, torch.Tensor):  # Help mypy understand the type
                    raise AssertionError(f"expected Tensor, got {type(p)}")
                grad_primals.append(p)
            else:
                inputs_needs_grads.append(False)

        # Get the outputs that need gradients
        needed_outs: list[Tensor] = []
        needed_tangents: list[Tensor] = []
        for out, tangent in zip(outs_to_grad, tangents):
            if isinstance(out, Tensor) and out.requires_grad:
                # A bit sketchy, but fixes e.g. test_aot_autograd_exhaustive_matmul_cpu_float32
                # The issue is that we are sensitive to decomps that don't accurately maintain
                # their output's _base.shape compared to eager mode, and this helps mitigate a bit.
                # The guard_or_true also sketchy; if unbacked
                # symints are involved, we're just going to assume that the
                # decomps setup the base shape correctly

                # Return out if the result of out.shape==tangent.shape is unknown or known to be true.
                # otherwise if its a known false return out.view(tangent.shape).
                # tangent should also be a tensor since it corresponds to a tensor output
                if not isinstance(tangent, torch.Tensor):
                    raise AssertionError(
                        f"Expected tensor tangent, got {type(tangent)}"
                    )
                needed_outs.append(
                    out
                    if guard_or_true(sym_eq(out.shape, tangent.shape))
                    else out.view(tangent.shape)
                )
                needed_tangents.append(tangent)

        setup_stacktrace_preservation_hooks(
            [out.grad_fn for out in needed_outs if out.grad_fn is not None]
        )

        if config.functionalize_rng_ops:
            PhiloxStateTracker.mark_beginning_of_backward()
        backward_out: tuple[Tensor, ...] = ()
        # Call the backwards pass
        if grad_primals:
            functional_tensor_mode = torch.utils._python_dispatch._detect_infra_mode(
                torch._C._TorchDispatchModeKey.FUNCTIONAL
            )
            if functional_tensor_mode is not None:
                # Side-Effect Tokens:
                # We want to have independent chains of tokens for forward and backward.
                # functional_tensor_mode._tokens is used by both.
                # We memoize the result tokens of forward in functional_tensor_mode._tokens_forward_output,
                # to return them as joint graph outputs.
                # We clean functional_tensor_mode._tokens before backward, to prevent reuse of forward tokens in backward.
                # Joint graph tracing allows tokens discovery,
                # So all the tokens in backward will be created and added as a graph inputs during tracing.
                functional_tensor_mode._tokens_forward_output = (
                    functional_tensor_mode._tokens
                )
                functional_tensor_mode._tokens = {}  # pyrefly: ignore[implicit-any]

            with (
                set_partitioner_tag_is_backward(),
                fx_traceback.preserve_node_meta(),
                ExitStack() as stack,
            ):
                backward_pass_autocast = torch._functorch.config.backward_pass_autocast
                if backward_pass_autocast == "same_as_forward":
                    # Use the ambient autocast mode(s)
                    pass
                elif backward_pass_autocast == "off":
                    stack.enter_context(disable_autocast())
                else:
                    # Disable autocast, then enable anything in `backward_pass_autocast`.
                    stack.enter_context(disable_autocast())
                    if not isinstance(backward_pass_autocast, list):
                        raise AssertionError(
                            f"expected backward_pass_autocast to be a list, got {type(backward_pass_autocast)}"
                        )
                    for kwargs in backward_pass_autocast:
                        if not isinstance(kwargs, dict):
                            raise AssertionError(
                                f"expected kwargs to be a dict, got {type(kwargs)}"
                            )
                        stack.enter_context(torch.amp.autocast(**kwargs))

                # for full graph export, we always export a joint graph where we assume no tangents are needed.
                if aot_config.no_tangents:
                    if not (
                        len(needed_tangents) == 1 and needed_tangents[0].numel() == 1
                    ):
                        raise AssertionError(
                            f"expected single scalar tangent for no_tangents mode, got {len(needed_tangents)} tangents"
                        )
                    backward_out = torch.autograd.grad(
                        needed_outs,
                        grad_primals,
                        allow_unused=True,
                    )
                else:
                    backward_out = torch.autograd.grad(
                        needed_outs,
                        grad_primals,
                        grad_outputs=needed_tangents,
                        allow_unused=True,
                    )
        backward_out_iter = iter(backward_out)
        final_outs = (
            outs,
            [next(backward_out_iter) if i else None for i in inputs_needs_grads],
        )
        if primals_descs is None:
            return final_outs  # type: ignore[return-value]
        if outs_descs is None:
            raise AssertionError("outs_descs must not be None")
        # pyrefly: ignore[bad-return]
        return final_outs, (
            outs_descs,
            [
                # TODO: ideally we do know this is DifferentiableAOTInput
                # but this is quite an involved refactor
                GradAOTOutput(desc) if i else None  # type: ignore[arg-type]
                for i, desc in zip(inputs_needs_grads, primals_descs)
            ],
        )

    @simple_wraps(inner_fn)
    def inner_fn_with_anomaly(
        primals: list[FxValue], tangents: list[FxValue]
    ) -> tuple[
        tuple[list[FxValue], list[Tensor | None]],
        tuple[list[AOTOutput], list[AOTOutput | None]],
    ]:
        with fx_traceback.preserve_node_meta(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Anomaly Detection has been enabled.")
            with torch.autograd.detect_anomaly(check_nan=False):
                return inner_fn(primals, tangents)

    def joint_helper(
        primals: list[FxValue], tangents: list[FxValue]
    ) -> tuple[
        tuple[list[FxValue], list[Tensor | None]],
        tuple[list[AOTOutput], list[AOTOutput | None]],
    ]:
        return inner_fn_with_anomaly(primals, tangents)

    joint_helper.handle = joint_fn_handle  # type: ignore[attr-defined]

    # pyrefly: ignore[bad-return]
    return joint_helper


def create_functionalized_rng_ops_wrapper(
    func: Callable[..., Any],
    args: Any,
    args_descs: list[AOTInput],
    trace_joint: bool = True,
) -> Any:
    # Functionalization of rng ops changes the calling convention of the joint graph.
    # It goes from (primals, tangents) to (seed, offset, primals, tangents)
    # At runtime, we pass on the current seed and offset. This is hidden from
    # the user.
    fake_mode_det = detect_fake_mode()
    fake_mode: AbstractContextManager[Any] = nullcontext()
    if fake_mode_det is not None:
        fake_mode = fake_mode_det

    def override_get_rng_state(
        device: int | str | torch.device = "cuda",
    ) -> Tensor:
        out = PhiloxStateTracker.get_state_as_tensor()
        return out

    def override_set_rng_state(
        x: Tensor, device: int | str | torch.device = "cuda"
    ) -> None:
        PhiloxStateTracker.set_state_from_tensor(x)

    def append_rng_offsets(outs: Any, outs_descs: Any) -> Any:
        if trace_joint:
            # outs signature before: Tuple(fwd_outputs), Tuple(bwd_outputs)
            # outs signature after: Tuple(fwd_outputs, new_fwd_rng_offset), Tuple(bwd_offset, new_bwd_rng_offset)
            return (
                (
                    (*outs[0], PhiloxStateTracker.get_updated_fwd_offset()),
                    (*outs[1], PhiloxStateTracker.get_updated_bwd_offset()),
                ),
                (
                    (*outs_descs[0], PhiloxUpdatedForwardOffsetAOTOutput()),
                    (*outs_descs[1], PhiloxUpdatedBackwardOffsetAOTOutput()),
                ),
            )
        else:
            # outs signature before: Tuple(fwd_outputs)
            # outs signature after: Tuple(fwd_outputs, new_fwd_rng_offset)
            return (
                (*outs, PhiloxStateTracker.get_updated_fwd_offset()),
                (*outs_descs, PhiloxUpdatedForwardOffsetAOTOutput()),
            )

    def traced_joint(
        primals: list[FxValue],
        tangents: list[FxValue],
        fwd_seed: Tensor,
        fwd_base_offset: Tensor,
        bwd_seed: Tensor,
        bwd_base_offset: Tensor,
    ) -> tuple[
        tuple[tuple[FxValue, ...], tuple[FxValue, ...]],
        tuple[tuple[AOTOutput, ...], tuple[AOTOutput, ...]],
    ]:
        with (
            patch("torch.cuda.get_rng_state", override_get_rng_state),
            patch("torch.cuda.set_rng_state", override_set_rng_state),
        ):
            return append_rng_offsets(*func(primals, tangents))

    def traced_forward(*primals_fwd_seed_fwd_base_offset: Any) -> Any:
        # The signature is (*primals, seed, offset)
        with (
            patch("torch.cuda.get_rng_state", override_get_rng_state),
            patch("torch.cuda.set_rng_state", override_set_rng_state),
        ):
            return append_rng_offsets(*func(*primals_fwd_seed_fwd_base_offset[:-2]))

    if trace_joint:
        # Get the current seed and offset to setup tracing.
        fwd_seed, fwd_base_offset = CUDARngStateHelper.get_torch_state_as_tuple(
            fake_mode
        )
        bwd_seed, bwd_base_offset = CUDARngStateHelper.get_torch_state_as_tuple(
            fake_mode
        )
        PhiloxStateTracker.record_state(fwd_seed, fwd_base_offset, "forward")
        PhiloxStateTracker.record_state(bwd_seed, bwd_base_offset, "backward")
        return (
            traced_joint,
            (
                *args,
                fwd_seed,
                fwd_base_offset,
                bwd_seed,
                bwd_base_offset,
            ),
            (
                *args_descs,
                PhiloxForwardSeedAOTInput(),
                PhiloxForwardBaseOffsetAOTInput(),
                PhiloxBackwardSeedAOTInput(),
                PhiloxBackwardBaseOffsetAOTInput(),
            ),
        )
    else:
        # Get the current seed and offset to setup tracing.
        fwd_seed, fwd_base_offset = CUDARngStateHelper.get_torch_state_as_tuple(
            fake_mode
        )
        PhiloxStateTracker.record_state(fwd_seed, fwd_base_offset, "forward")
        return (
            traced_forward,
            (*args, fwd_seed, fwd_base_offset),
            (
                *args_descs,
                PhiloxForwardSeedAOTInput(),
                PhiloxForwardBaseOffsetAOTInput(),
            ),
        )


@contextmanager
def set_partitioner_tag(tag: str) -> Generator[None, None, None]:
    meta_key = "partitioner_tag"
    if not fx_traceback.has_preserved_node_meta():
        raise AssertionError("expected preserved node meta")

    original_val = fx_traceback.current_meta.get(meta_key, None)
    fx_traceback.current_meta[meta_key] = tag
    try:
        yield
    finally:
        fx_traceback.current_meta[meta_key] = original_val


def set_partitioner_tag_is_backward() -> AbstractContextManager[None]:
    return set_partitioner_tag("is_backward")


def set_partitioner_tag_must_be_in_backward() -> AbstractContextManager[None]:
    return set_partitioner_tag("must_be_in_backward")


def set_partitioner_tag_must_be_in_forward() -> AbstractContextManager[None]:
    return set_partitioner_tag("must_be_in_forward")


@dataclass
class MutationCounters:
    mc_data: int
    mc_storage: int
    mc_inductor_storage_resized: int


T = TypeVar("T")


def sc_visit(
    t: torch.Tensor,
    fn: Callable[[Tensor], T],
    reduce_fn: Callable[[T, T], T],
    accum_init: T,
) -> T:
    if not is_traceable_wrapper_subclass(t):
        return fn(t)

    accum = accum_init

    def visit(e: Any) -> None:
        if not is_traceable_wrapper_subclass(e):
            nonlocal accum
            accum = reduce_fn(accum, fn(e))
            return

        for a in e.__tensor_flatten__()[0]:
            match getattr(e, a):
                case torch.Tensor() as inner:
                    visit(inner)
                case CustomClassBase():
                    pass
                case unexpected:
                    raise AssertionError(
                        f"expected Tensor or CustomClassBase, got {type(unexpected)}"
                    )

    visit(t)
    return accum


def _get_mutation_counter(t: torch.Tensor) -> int:
    return sc_visit(
        t,
        lambda t: torch._functionalize_mutation_counter(t.elem),  # type: ignore[attr-defined]
        lambda l, r: max(l, r),
        -1,
    )


def _get_storage_changed_counter(t: torch.Tensor) -> int:
    return sc_visit(
        t,
        lambda t: torch._functionalize_storage_changed_counter(t.elem),  # type: ignore[attr-defined]
        lambda l, r: max(l, r),
        -1,
    )


def _get_inductor_storage_resized_counter(t: torch.Tensor) -> int:
    return sc_visit(
        t,
        lambda t: torch._functionalize_inductor_storage_resized_counter(t.elem),  # type: ignore[attr-defined]
        lambda l, r: max(l, r),
        -1,
    )


def _get_mutation_counters(t: torch.Tensor) -> MutationCounters:
    return MutationCounters(
        _get_mutation_counter(t),
        _get_storage_changed_counter(t),
        _get_inductor_storage_resized_counter(t),
    )


def apply_in_graph_mutations(
    input_info: InputAliasInfo,
    inpt_old: Tensor,
    inpt_new: Tensor,
    f_inpt: Tensor,
    input_idx: int,
    mcs: MutationCounters | None = None,
    applied_mcs: MutationCounters | None = None,
) -> None:
    if input_info.mutation_type != MutationType.MUTATED_IN_GRAPH:
        raise AssertionError(
            f"expected mutation_type MUTATED_IN_GRAPH, got {input_info.mutation_type}"
        )
    # See Note [set_() Input Mutations in AOTAutograd]
    # all mutations on the input must be under no_grad, so it is safe to put in the graph
    # Here, we're saying that if an input experienced a set call, inp.set_(other),
    # then we can effectively not have to worry about whether its data was mutated.
    # There are 3 cases:
    # (1) We mutate inp *after* the set_() call. other is a graph intermediate.
    #     In this case, we're not really mutating the input storage of "inp";
    #     we're mutating the storage of an intermediate value (other),
    #     and slamming that storage into the input tensor. So no data mutation is necessary.
    # (2) We mutate inp *after* the set_() call. other is a graph *input*.
    #     In this case, the data mutation will be properly handled in the runtime
    #     epilogue during the processing of "other"
    # (3) We mutate inp *before* the set_() call.
    #     This case is *not* currently handled.
    if input_info.mutates_storage_metadata:
        if mcs is None or mcs.mc_storage > applied_mcs.mc_storage:  # type: ignore[union-attr]
            with torch.no_grad():
                if input_info.mutation_is_shallow_copy_data:
                    torch.ops.aten.shallow_copy_data_(inpt_old, inpt_new)
                else:
                    # pyrefly: ignore [bad-argument-type, no-matching-overload]
                    inpt_old.set_(inpt_new)

    # Note [Ordering of resize_() and set_()]
    # Importantly: the common usage in FSDP is that we have a dummy parameter
    # that sees a set_() and **Then** a resize_().
    # We must put those mutations into the graph in the same order,
    # Since running them in the opposite order will have different behavior.
    # We fully ban resize_() followed by set_() for now, although in principal
    # we could support this
    if input_info.mutation_inductor_storage_resize:
        if (
            mcs is None
            or mcs.mc_inductor_storage_resized > applied_mcs.mc_inductor_storage_resized  # type: ignore[union-attr]
        ):
            # resizing is not supported on subclasses (we error earlier if this happens)
            from torch._subclasses.functional_tensor import FunctionalTensor

            if not isinstance(f_inpt, FunctionalTensor):
                raise AssertionError(f"expected FunctionalTensor, got {type(f_inpt)}")
            old_storage_size = torch._functionalize_get_storage_size(  # type: ignore[attr-defined]
                f_inpt.elem, before=True
            )
            new_storage_size = torch._functionalize_get_storage_size(  # type: ignore[attr-defined]
                f_inpt.elem, before=False
            )
            if old_storage_size != new_storage_size:
                if not (old_storage_size == 0 or new_storage_size == 0):
                    raise AssertionError(f"""\
        Encosize during tracing on input {input_idx}. Old nbytes={old_storage_size}, new nbytes={new_storage_size}
        We oresizing on graph inputs as long as the input either starts or ends with a storage size of 0
        (thee for FSDP)""")
                torch.ops.inductor.resize_storage_bytes_(inpt_old, new_storage_size)
            if new_storage_size == 0:
                # Even if we marked the input as having a data mutation (thus needing a copy_()),
                # We should **ignore** it if our input has no storage
                # (this can happen if, e.g. we temporarily resize our input, copy data into it,
                #  and resize it back down to zero)
                return

    # Optimization: if the copy_() is a no-op then don't include it in the graph.
    # In theory inductor could optimize this away, however in fsdp, we end up with
    # param.copy_(param), where param is a zero-storage-size tensor,
    # and running this op in eager mode (using the aot_eager backend) will result in a segfault.
    # So we may as well optimize it away here.
    if inpt_old is inpt_new:
        # (This check needs to be done after putting resize_() in the graph,
        # since a resize_(0) doesn't actually change the FunctionalTensor's inner tensor)
        return
    # We found an input that had a (data-only) mutation.
    # Since keep_input_mutations is set, we need to faithfully apply a copy_()
    # so the compiler will see the input mutation in the graph.

    if not input_info.mutates_data:
        return

    if mcs is not None and mcs.mc_data <= applied_mcs.mc_data:  # type: ignore[union-attr]
        return

    if input_info.mutations_hidden_from_autograd:
        # Hidden from autograd = run under no_grad, **and** don't bump VC
        # (although if the tensor was created in inference mode, it has no VC)
        if inpt_old.is_inference():
            maybe_preserve_vc = nullcontext()
        else:
            maybe_preserve_vc = torch.autograd._unsafe_preserve_version_counter(
                inpt_old  # type: ignore[assignment]
            )
        with torch.no_grad(), maybe_preserve_vc:
            inpt_old.copy_(inpt_new)
    elif input_info.mutations_under_no_grad_or_inference_mode:
        # Under no_grad = run under no_grad (we still bump the VC though)
        # (inference_mode will also bump the VC, as long as the tensor in question
        # was created outside of inference_mode)

        with torch.no_grad():
            inpt_old.copy_(inpt_new)
    else:
        inpt_old.copy_(inpt_new)


# This creates the final function that we want to trace using make_fx(),
# in both aot_dispatch_autograd and aot_dispatch_base.
# Preconditions:
# - fn corresponds to the user's fw function
# - fn arguments have been flattened, duplicate arguments have been handled
# - In the returned function, the "primals" arguments *includes* synthetic bases.
# This function does the work of functionalizing the input function,
# and performing copy_() calls at the end of the function if `keep_input_mutations` is set.
# The function returned has signature that is either:
# (1) "traced_fn(primals: List[Any])" if trace_joint is False
# (2) "traced_fn(primals: List[Any], tangents: List[Any])" if trace_joint is True
# Returns a new (functionalized) function, and updated arguments to call it with.
def create_functionalized_fn(
    fn: Callable[..., Any],
    args: Any,
    args_descs: Any,
    *,
    meta: ViewAndMutationMeta,
    aot_config: AOTConfig,
    trace_joint: bool,
    joint_fn_handle: JointFnHandle | None = None,
) -> Any:
    primals_after_forward = None
    f_args_after_forward = None
    f_args_mutation_counters_after_forward: list[MutationCounters] | None = None
    inputs_mutated_in_graph = [
        info.mutation_type == MutationType.MUTATED_IN_GRAPH for info in meta.input_info
    ]
    has_input_mutated_in_graph = any(inputs_mutated_in_graph)

    @simple_wraps(fn)
    def _functionalized_f_helper(
        *args: list[FxValue],
    ) -> tuple[tuple[list[FxValue], list[Tensor]], list[AOTOutput | None]]:
        with maybe_enable_thunkify():
            # See Note [Disabling Functionalize TLS Above Python Functionalization]
            disable_above = torch._C._ExcludeDispatchKeyGuard(
                torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
            )

            with disable_above:
                # The functionalization code here can potentially trigger traces
                # into the graph, but we'd prefer to NOT do this, because if we
                # trace them now, we will end up with FX nodes that don't have
                # module stack annotations, which makes unflattener unhappy.
                # Wrap inputs into functional wrappers
                f_args = pytree.tree_map(to_fun, args)

                if trace_joint and has_input_mutated_in_graph and joint_fn_handle:
                    # TODO(ivankobzarev): Support fw and bw mutations for subclasses
                    def _post_forward(primals: Any) -> None:
                        nonlocal primals_after_forward
                        primals_after_forward = pytree.tree_map(from_fun, primals)
                        nonlocal f_args_after_forward
                        f_args_after_forward = f_args[0]
                        nonlocal f_args_mutation_counters_after_forward
                        f_args_mutation_counters_after_forward = [
                            MutationCounters(-1, -1, -1)
                            if not inputs_mutated_in_graph[i]
                            else _get_mutation_counters(f_arg)
                            for i, f_arg in enumerate(f_args_after_forward)
                        ]

                    joint_fn_handle.post_forward = _post_forward

                # Run the joint
                f_outs, f_outs_descs = call_and_expect_output_descs(fn, f_args)

            if trace_joint:
                # We support a limited amount of mutation of graph inputs during the backward pass.
                # (This is used e.g. by Float8, which needs to update buffers during the backward pass)
                # Here, we perform extra checks for primals that were mutated in the **backward**
                # We're doing the checks here instead of doing them with the rest of the input mutation handling because:
                # - We need to detect inputs that were mutated in the backward **separately** from mutations that happened
                #   during the forward, because the handling is different: some input mutations from the forward
                #   can be only handled in a fw-only runtime epilogue, and in theory if we wanted to handle those same
                #   types of mutations in the backward we would need a bw-only runtime epilogue.
                # - We could in theory have our analysis pass differentiate mutations in the fw from mutations in
                #   the bw by running our analysis first on the fw-only graph, and then on the joint graph. This would
                #   require an extra round of tracing though, so it's more efficient to do in-line here.
                if not (
                    isinstance(args, tuple)
                    and len(args) == 2
                    and isinstance(args[0], (list, tuple))
                ):
                    raise AssertionError(
                        f"expected args to be tuple of (list/tuple, ...), got {type(args)}"
                    )
                # Only look at mutations that happened to forward inputs (e.g. fw buffers that were saved for bw)
                primals_before = args[0]
                primals_after = pytree.tree_map(from_fun, f_args[0])
                for idx, (f_inpt, before, after, inpt_info) in enumerate(
                    zip(f_args[0], primals_before, primals_after, meta.input_info)
                ):
                    # Store information about mutations in joint(for backward analysis)
                    joint_mutates_data = has_data_mutation(f_inpt)

                    joint_mutates_metadata = has_metadata_mutation(
                        f_inpt, before, check_only_storage_mutation=False
                    )

                    # Ban metadata mutations on fw inputs during the bw
                    if not inpt_info.mutates_metadata:
                        if joint_mutates_metadata:
                            raise AssertionError(
                                "Found a graph input that had its metadata mutated in the backward. This is not supported"
                            )

                    # Ban storage resizing on fw inputs during the bw
                    if not inpt_info.mutation_inductor_storage_resize:
                        if was_inductor_storage_resized(f_inpt):
                            raise AssertionError(
                                "Found a graph input that had storage resizing in the backward. This is not supported"
                            )

                    # Allow data mutations on fw inputs during the bw, but only if they do not require grad
                    # So we can guarantee that we can keep the mutations in the graph
                    if (
                        joint_mutates_data
                        and not inpt_info.mutates_data
                        and not inpt_info.mutates_storage_metadata
                    ):
                        # Not banning here mutations on inpt_info.requires_grad -
                        # we'll check at runtime and fail only when backward is under torch.is_grad_enabled (create_graph)
                        # Add node meta for copy_ for partitioner that this node should be in backward graph.
                        with (
                            torch.fx.traceback.preserve_node_meta(),
                            set_partitioner_tag_must_be_in_backward(),
                        ):
                            # before and after should be tensors if we're calling copy_ on them
                            if not (
                                isinstance(before, torch.Tensor)
                                and isinstance(after, torch.Tensor)
                            ):
                                raise AssertionError(
                                    f"expected both before and after to be Tensors, got {type(before)} and {type(after)}"
                                )
                            # no_grad prevents the FakeTensor's requires_grad from
                            # triggering check_inplace during tracing.  The
                            # requires_grad case is checked at runtime instead
                            with torch.no_grad():
                                before.copy_(after)
                        meta.indices_of_inputs_that_requires_grad_with_mutations_in_bw.append(
                            idx
                        )
                # Now that we covered mutations to *forward* inputs during the backward,
                # we also need to cover mutations to *backward-only* inputs during the backward (e.g. mutation to a grad_out).
                # Today, we will just error in all cases of this happening unless someone needs us to support it.
                tangents_before = args[1]
                tangents_after = pytree.tree_map(from_fun, f_args[1])
                for f_inpt, before, after in zip(
                    f_args[1], tangents_before, tangents_after
                ):
                    if has_metadata_mutation(
                        f_inpt, before, check_only_storage_mutation=False
                    ):
                        raise AssertionError(
                            "Found an input to the backward that had metadata mutated "
                            "during the backward pass. This is not supported"
                        )
                    if has_data_mutation(f_inpt):
                        can_be_in_graph = _check_if_mutation_can_be_in_graph(
                            keep_input_mutations=True,
                            mutates_data=True,
                            mutates_metadata=False,
                            mutations_hidden_from_autograd=are_all_mutations_hidden_from_autograd(
                                f_inpt
                            ),
                            mutations_under_no_grad_or_inference_mode=are_all_mutations_under_no_grad_or_inference_mode(
                                f_inpt
                            ),
                            mutates_storage_metadata=False,
                            mutation_inductor_storage_resize=was_inductor_storage_resized(
                                f_inpt
                            ),
                            requires_grad=f_inpt.requires_grad,
                        )
                        if not can_be_in_graph:
                            raise AssertionError(
                                "a backward input that had data mutated in an autograd-aware way. This is not supported"
                            )
                        # Perform the input mutation
                        with torch.fx.traceback.preserve_node_meta():
                            # before and after should be tensors if we're calling copy_ on them
                            if not (
                                isinstance(before, torch.Tensor)
                                and isinstance(after, torch.Tensor)
                            ):
                                raise AssertionError(
                                    f"expected both before and after to be Tensors, got {type(before)} and {type(after)}"
                                )
                            before.copy_(after)

            if aot_config.keep_inference_input_mutations:
                # Note: This is a bit annoying. There's a layering issue here, where:
                # (1) functionalization needs to operate on **synthetic base** inputs, before unpacking them into the "real" inputs.
                # (2) For keep_input_mutations, we support tracing a call to copy_() directly on mutated inputs.
                #     However, we **only** want to support this for inputs that have data-only (and no metadata) mutations,
                #     because inductor (and backends in generally) would prefer not to see these (e.g. as_strided_(), resize_()).
                #     This makes it pretty difficult for this logic to operate on synthetic bases.
                # (3) In addition, there are cases where it's significantly cheaper to perform the copy on the individual
                #     (unpacked) input aliases, instead of the synthetic base.
                # Example case where (3) could be important:
                #
                #     def f(x, y):
                #         x.mul_(2)
                #         y.mul_(3)
                #         return x, y
                #    a = torch.ones(1'000'000)
                #    x, y = out(a[0:9], a[1:10])
                #
                # It would be much better to add copy_() calls into the graph for the two tiny slices, instead of materializing
                # a giant "updated synthetic base" and copying into a's entire storage.
                #
                # For now, we are pessimistically not performing the optimization from (3);
                # we will materialize an "updated" synthetic base, and copy it back to the synthetic input base.
                # This allows us to factor aot autograd much more nicely, since only one area of the code needs to worry
                # about synthetic bases.

                # Apply in graph forward mutations only in joint case.
                # Note: Mutations of primals in forward AND backward.
                # If we have mutations of the same input in forward and in backward,
                # we can not fuse them into one copy_ node. As in this case partitioner will put it
                # either in forward or in backward. This will lead to incorrect state
                # after forward and before backward.
                # We have to emit two copy_ nodes, marking with additional meta each node,
                # if it must be in forward or backward.
                # We memorize mutation counter of the inputs after forward.
                # Based on this after joint graph we check if backward also mutated input or not.
                # We emit copy_ only in the end of joint tracing, to provide invariant for joint
                # graph passes, that our graph is functional, except only some number of copy_ nodes
                # in the end.
                mcs_applied: list[MutationCounters] = [MutationCounters(0, 0, 0)] * len(
                    meta.input_info
                )
                if f_args_mutation_counters_after_forward is not None:
                    primals_before = args[0]
                    for idx, (f_inpt, before, after, inpt_info) in enumerate(
                        # pyrefly: ignore [no-matching-overload]
                        zip(
                            f_args_after_forward,  # type: ignore[arg-type]
                            primals_before,  # type: ignore[arg-type]
                            primals_after_forward,  # type: ignore[arg-type]
                            meta.input_info,
                        )
                    ):
                        if inpt_info.mutation_type != MutationType.MUTATED_IN_GRAPH:
                            continue

                        mcs_after_forward = f_args_mutation_counters_after_forward[idx]
                        with (
                            torch.fx.traceback.preserve_node_meta(),
                            set_partitioner_tag_must_be_in_forward(),
                            _proxy_tensor_disable_update_tensor_tracker(),
                        ):
                            apply_in_graph_mutations(
                                inpt_info,
                                # pyrefly: ignore [bad-argument-type]
                                before,
                                after,
                                f_inpt,
                                idx,
                                mcs_after_forward,
                                mcs_applied[idx],
                            )
                            mcs_applied[idx] = mcs_after_forward

                for idx, (inpt_old, f_inpt) in enumerate(
                    zip(args, f_args) if not trace_joint else zip(args[0], f_args[0])  # type: ignore[arg-type]
                ):
                    if not isinstance(f_inpt, torch.Tensor):
                        continue
                    if not is_fun(f_inpt):
                        raise AssertionError(
                            f"expected functional tensor, got {type(f_inpt)}"
                        )
                    inpt_new = from_fun(f_inpt)
                    if (
                        meta.input_info[idx].mutation_type
                        != MutationType.MUTATED_IN_GRAPH
                    ):
                        continue
                    mcs: MutationCounters | None = None
                    if f_args_mutation_counters_after_forward is not None:
                        # This could happen for subclasses tracing
                        # Subclasses support for mutations in fw and bw is TBD.
                        mcs = _get_mutation_counters(f_inpt)
                        if mcs == mcs_applied[idx]:
                            # No mutation in backward; mutation was already applied.
                            continue

                    with (
                        torch.fx.traceback.preserve_node_meta(),
                        set_partitioner_tag_must_be_in_backward(),
                    ):
                        apply_in_graph_mutations(
                            meta.input_info[idx],
                            # pyrefly: ignore[bad-argument-type]
                            inpt_old,
                            # pyrefly: ignore[bad-argument-type]
                            inpt_new,
                            f_inpt,
                            idx,
                            mcs,
                            mcs_applied[idx],
                        )

                # When an output tensor is a functionalized mutated input, and we
                # were able to move the mutation in to the graph then we can return
                # the mutated input directly. This prevents duplicating the
                # tensors contents.
                flat_outs, outs_spec = pytree.tree_flatten(f_outs)
                flat_outs = [from_fun(o) for o in flat_outs]
                num_outs = len(meta.output_info)

                for i in range(num_outs):
                    info = meta.output_info[i]
                    if info.output_type != OutputType.is_input:
                        continue

                    if info.base_idx is None:
                        raise AssertionError("info.base_idx must not be None")
                    if (
                        meta.input_info[info.base_idx].mutation_type
                        == MutationType.MUTATED_IN_GRAPH
                    ):
                        fw_args = args[0] if trace_joint else args
                        flat_outs[i] = fw_args[info.base_idx]
                return pytree.tree_unflatten(flat_outs, outs_spec), f_outs_descs

            return pytree.tree_map(from_fun, f_outs), f_outs_descs

    # Kinda annoying, but needed to make sure that the fx graph we trace out has "primals"
    # and "tangents" as its input names (which are special-cased by the partitioner)
    # TODO (tmanlaibaatar) revisit this if we ever need to turn on non-strict joint graph export
    def joint_helper(primals: list[FxValue], tangents: list[FxValue]) -> Any:
        return _functionalized_f_helper(primals, tangents)

    helper = joint_helper if trace_joint else _functionalized_f_helper
    if config.functionalize_rng_ops:
        # Setup the wrapper for functionalization of rng ops
        helper, args, args_descs = create_functionalized_rng_ops_wrapper(
            helper, args, args_descs, trace_joint
        )

    return helper, args, args_descs


def handle_effect_tokens_fn(
    fn: Callable[..., Any],
    args: Any,
    args_descs: list[AOTInput],
    *,
    meta: ViewAndMutationMeta,
    trace_joint: bool,
) -> Any:
    num_tokens = len(meta.tokens)

    @simple_wraps(fn)
    def inner_fn(*args: Any) -> Any:
        # See Note [Disabling Functionalize TLS Above Python Functionalization]
        disable_above = torch._C._ExcludeDispatchKeyGuard(
            torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
        )

        with disable_above:
            # See Note [Side-Effectful Tokens in AOTAutograd]
            if trace_joint:
                if not (isinstance(args, tuple) and isinstance(args[0], (list, tuple))):
                    raise AssertionError(
                        f"expected args to be tuple with first element as list/tuple, got {type(args)}"
                    )
                tokens = args[0][:num_tokens]
                if not all(token.numel() == 0 for token in tokens):
                    raise AssertionError("all tokens must have numel() == 0")
                args = (args[0][num_tokens:], *args[1:])
            else:
                tokens = args[:num_tokens]
                if not all(token.numel() == 0 for token in tokens):
                    raise AssertionError("all tokens must have numel() == 0")
                args = args[num_tokens:]

            # Populate the current FunctionalTensorMode with the tokens per
            # operator. See Note [FunctionalTensorMode is Stateful]
            functional_tensor_mode = torch.utils._python_dispatch._detect_infra_mode(
                torch._C._TorchDispatchModeKey.FUNCTIONAL
            )
            if functional_tensor_mode is None:
                raise AssertionError("functional_tensor_mode must not be None")
            f_tokens = pytree.tree_map(to_fun, tokens)
            for i, k in enumerate(meta.tokens.keys()):
                functional_tensor_mode._tokens[k] = f_tokens[i]

            # Run the joint
            outs, outs_descs = call_and_expect_output_descs(fn, args)

        # Return both the tokens and the outputs
        # See Note [Side-Effectful Tokens in AOTAutograd]
        if trace_joint:
            if len(outs) != 2:
                raise AssertionError(
                    f"expected len(outs) == 2 for joint trace, got {len(outs)}"
                )
            if len(functional_tensor_mode._tokens_forward_output) != num_tokens:
                raise AssertionError(
                    f"expected {num_tokens} forward output tokens, got {len(functional_tensor_mode._tokens_forward_output)}"
                )
            fwd_out_tokens = functional_tensor_mode._tokens_forward_output.values()

            bwd_out_tokens = functional_tensor_mode._tokens.values()

            f_fwd_out_tokens = [from_fun(t) for t in fwd_out_tokens]
            f_bwd_out_tokens = [from_fun(t) for t in bwd_out_tokens]
            f_fwd_out_tokens_descs = [
                ForwardTokenAOTOutput(i) for i in range(len(fwd_out_tokens))
            ]
            f_bwd_out_tokens_descs = [
                BackwardTokenAOTOutput(i) for i in range(len(bwd_out_tokens))
            ]

            meta.num_backward_tokens = len(bwd_out_tokens)
            return (
                ((*f_fwd_out_tokens, *outs[0]), (*outs[1], *f_bwd_out_tokens)),
                (
                    (*f_fwd_out_tokens_descs, *outs_descs[0]),
                    (*outs_descs[1], *f_bwd_out_tokens_descs),
                ),
            )

        out_tokens = [from_fun(t) for t in functional_tensor_mode._tokens.values()]
        # TODO: can probably do a little more resolution here
        out_tokens_descs = [
            ForwardTokenAOTOutput(i)
            for i in range(len(functional_tensor_mode._tokens.values()))
        ]
        return ((*out_tokens, *outs), (*out_tokens_descs, *outs_descs))

    # Additionally pass in tokens as inputs
    # See Note [Side-Effectful Tokens in AOTAutograd]
    additional_fwd_token_inputs = [torch.tensor([])] * num_tokens
    additional_fwd_token_inputs_descs = [
        ForwardTokenAOTInput(i) for i in range(num_tokens)
    ]

    if trace_joint:
        args = ([*additional_fwd_token_inputs, *args[0]], *args[1:])
        args_descs = (  # type: ignore[assignment]
            [*additional_fwd_token_inputs_descs, *args_descs[0]],  # type: ignore[misc]
            *args_descs[1:],
        )
    else:
        args = [*additional_fwd_token_inputs, *args]
        args_descs = [*additional_fwd_token_inputs_descs, *args_descs]

        if num_tokens > 0:
            meta.static_input_indices = [
                idx + num_tokens for idx in meta.static_input_indices
            ]
    return inner_fn, args, args_descs


# Given a function operating on Subclass -> Subclass, returns a function that operates on Tensor -> Tensor
# Also returns:
# - the new set of arguments to pass into this function (now that tensor subclasses have been eliminated)
# - the updated ViewAndMutationMeta for this dense -> dense function.
# The other important arguments are:
# - flat_fn_maybe_joint: when is_joint_structure=True, this is the joint fw-bw function.
#                        when is_joint_structure=False, this is just the forward function.
# - fw_only: this is *always* the forward-only function.
#   Why do we need this? We need to collect updated ViewAndMutationMeta on our new dense -> dense functions.
#   In particular, we need this to tell the partitioner how many dense forward outputs there are.
def aot_dispatch_subclass(
    flat_fn_maybe_joint: JointTraceFn | TraceFn,
    args: list[FxValue] | tuple[list[FxValue], list[FxValue]],
    args_descs: list[AOTInput] | tuple[list[AOTInput], list[AOTInput]],
    *,
    is_joint_structure: bool,
    meta: ViewAndMutationMeta,
    fw_only: Callable[..., Any],
) -> SubclassTracingInfo:
    # Skip logic if we don't need to trace through any subclasses
    req_subclass_dispatch = requires_subclass_dispatch(args, meta)  # type: ignore[arg-type]
    if not req_subclass_dispatch:
        return SubclassTracingInfo(
            plain_tensor_trace_fn=flat_fn_maybe_joint,
            plain_tensor_args=args,
            plain_tensor_args_descs=args_descs,
            maybe_subclass_meta=None,
        )

    # TODO: add subclass guards (later PR).

    # What's going on here? We need to compute subclass metadata about the outputs of the joint (grad_inputs).
    # Annoying: we don't know the grad input metas until we're in the middle of tracing the joint,
    # so we set it later, while we're tracing the joint (see inner_fn() below).
    # Another option would be to run our run_functionalized_fw_and_collect_metadata() function
    # directly on the joint, but this would hurt compile time (adding yet another pass through the joint).
    subclass_meta = SubclassMeta()

    # NB: doesn't take descs, this is going from the NEW flat_args to the
    # subclasses, we don't need to do bookkeeping here
    def inner_fn(fn: Callable[..., Any], args: Any, *, use_trace_joint: bool) -> Any:
        # Step 1: wrap tensor inputs into subclasses if necessary
        all_args = wrap_tensor_subclasses_maybe_joint(
            args, is_joint_structure=use_trace_joint, meta=meta
        )

        # Step 2: call the inner function, with our (maybe subclass) inputs
        wrapped_outs, wrapped_outs_descs = call_and_expect_output_descs(fn, all_args)  # type: ignore[arg-type]

        if use_trace_joint:
            # See Note: [Computing Subclass Metadata about grad_inputs]
            # We also stash subclass info on our grad_inputs, if we're tracing the joint.
            nonlocal subclass_meta
            if not (isinstance(wrapped_outs, tuple) and len(wrapped_outs) == 2):
                raise AssertionError(
                    f"expected wrapped_outs to be tuple of length 2, got {type(wrapped_outs)}, {wrapped_outs_descs}"
                )
            # Don't need fw outs since we already have subclass metadata on them
            grad_inputs = wrapped_outs[1]
            subclass_meta.grad_input_metas = create_subclass_meta(grad_inputs)

            # Add extra symints as outputs to the forward/backward graphs
            # ignore nested ints here
            forward_outs, forward_outs_descs = unwrap_tensor_subclasses(
                wrapped_outs[0],
                wrapped_outs_descs[0],
                append_symints=True,
            )
            # ignore nested ints here
            backward_outs, backward_outs_descs = unwrap_tensor_subclasses(
                wrapped_outs[1],
                wrapped_outs_descs[1],
                append_symints=True,
            )
            return (
                (forward_outs, backward_outs),
                (forward_outs_descs, backward_outs_descs),
            )

        # Step 3: Unwrap any subclass outputs back into dense tensors
        return unwrap_tensor_subclasses(
            wrapped_outs, wrapped_outs_descs, append_symints=True
        )

    def joint_fn(
        primals: list[FxValue], tangents: list[FxValue]
    ) -> tuple[
        tuple[list[FxValue], list[FxValue]], tuple[list[AOTOutput], list[AOTOutput]]
    ]:
        with maybe_enable_thunkify():
            return inner_fn(
                flat_fn_maybe_joint, (primals, tangents), use_trace_joint=True
            )

    def fw_fn(*primals: FxValue) -> tuple[list[FxValue], list[AOTOutput]]:
        with maybe_enable_thunkify():
            return inner_fn(flat_fn_maybe_joint, primals, use_trace_joint=False)

    def metadata_fn(*primals: FxValue) -> tuple[list[FxValue], list[AOTOutput]]:
        @simple_wraps(fw_only)
        def inner_fw_only(*args: Any) -> Any:
            return call_and_expect_output_descs(fw_only, args)

        return inner_fn(inner_fw_only, primals, use_trace_joint=False)

    if is_joint_structure:
        primals_wrapped: list[FxValue] = typing.cast(list[FxValue], args[0])
        primals_wrapped_descs: list[AOTInput] = typing.cast(
            list[AOTInput], args_descs[0]
        )
        tangents_wrapped: list[FxValue] = typing.cast(list[FxValue], args[1])
        tangents_wrapped_descs: list[AOTInput] = typing.cast(
            list[AOTInput], args_descs[1]
        )

        # Add extra symints (size/strides) as input to the forward graph
        primals_unwrapped_pair = unwrap_tensor_subclasses(
            primals_wrapped,
            primals_wrapped_descs,
            append_symints=True,
        )
        # We pass append_symints=False here because the partitioner will
        # capture and add any extra argument.
        tangents_unwrapped_pair = unwrap_tensor_subclasses(
            tangents_wrapped,
            tangents_wrapped_descs,
            append_symints=False,
        )

        args_unwrapped = (primals_unwrapped_pair[0], tangents_unwrapped_pair[0])
        args_descs_unwrapped = (primals_unwrapped_pair[1], tangents_unwrapped_pair[1])
        remapped_static_indices = remap_unwrapped_subclass_arg_indices(
            primals_wrapped,
            meta.static_input_indices,  # type: ignore[arg-type]
        )

        primals_unwrapped = args_unwrapped[0]  # type: ignore[assignment]
        primals_unwrapped_descs = args_descs_unwrapped[0]  # type: ignore[assignment]
        fn_to_trace = joint_fn  # type: ignore[assignment]
    else:
        primals_wrapped: list[FxValue] = typing.cast(list[FxValue], args)
        primals_wrapped_descs: list[AOTInput] = typing.cast(list[AOTInput], args_descs)

        args_unwrapped, args_descs_unwrapped = unwrap_tensor_subclasses(  # type: ignore[assignment]
            primals_wrapped,
            primals_wrapped_descs,
            append_symints=True,
        )
        remapped_static_indices = remap_unwrapped_subclass_arg_indices(
            primals_wrapped,
            meta.static_input_indices,  # type: ignore[arg-type]
        )

        primals_unwrapped = args_unwrapped  # type: ignore[assignment]
        primals_unwrapped_descs = args_descs_unwrapped  # type: ignore[assignment]
        fn_to_trace = fw_fn  # type: ignore[assignment]

    # Note: [Partitioner handling for Subclasses, Part 1]
    # The way the partitioner works is that:
    # (1) we pass is a single graph containing the joint fw/bw,
    #     where the # of graph outputs corresponds to # fw_outputs + # grad_inputs
    # (2) The partitioner accepts an arguments, num_fwd_outputs,
    #     and assumes that the first "num_fwd_outputs" graph outputs correspond
    #     to outputs of the forward graph.
    # How do tensor subclasses enter the picture?
    # the num_fwd_outputs in the final graph is actually non-trivial to compute,
    # because it can be influenced by input mutations and intermediate bases.
    # So we compute it by inspecting the current ViewAndMutationMeta object.
    # However, the original ViewAndMutationMeta that we computed was created
    # on the subclass -> subclass graph,
    # which can have a different number of outputs than the dense -> dense graph.
    # That's why we created a fresh metadata object on the dense -> dense function here,
    # and plumb it back up to the partitioner.
    # See Note: [Partitioner handling for Subclasses, Part 2] for more info.
    with torch._dynamo.eval_frame._use_eager_on_nested_compile():
        meta_updated = run_functionalized_fw_and_collect_metadata(
            without_output_descs(metadata_fn),
            # pyrefly: ignore [bad-argument-type]
            flat_args_descs=primals_unwrapped_descs,
            static_input_indices=remapped_static_indices,
            keep_input_mutations=meta.keep_input_mutations,
            # pyrefly: ignore [not-iterable]
        )(*primals_unwrapped)

    subclass_meta.fw_metadata = meta_updated

    return SubclassTracingInfo(
        plain_tensor_trace_fn=fn_to_trace,
        plain_tensor_args=args_unwrapped,
        plain_tensor_args_descs=args_descs_unwrapped,
        maybe_subclass_meta=subclass_meta,
    )


def create_functional_call(
    mod: Any,
    params_spec: Any,
    params_len: int,
    store_orig_mod: bool = False,
    strict_out_tuple: bool = True,
) -> Callable[..., Any]:
    # Redundant with dynamo, but worth having in case this gets invoked elsewhere.

    @simple_wraps(mod)
    def functional_call(*args: Any, **kwargs: Any) -> Any:
        flat_params = args[:params_len]
        if isinstance(params_spec, TreeSpec):
            params = pytree.tree_unflatten(flat_params, params_spec)
        else:
            if not isinstance(params_spec, list):
                raise AssertionError(
                    f"expected params_spec to be a list, got {type(params_spec)}"
                )
            params = dict(zip(params_spec, flat_params))
        with (
            stateless._reparametrize_module(mod, params),
            maybe_disable_thunkify(),
        ):
            if isinstance(mod, torch.fx.GraphModule):
                if kwargs:
                    # Handle **kwargs. FX only natively supports positional
                    # arguments (through placeholders).
                    arg_list = list(args[params_len:])
                    arg_list.extend(list(kwargs.values()))
                    args = tuple(arg_list)
                else:
                    args = args[params_len:]

                with fx_traceback.preserve_node_meta(), warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", "Anomaly Detection has been enabled."
                    )
                    with torch.autograd.detect_anomaly(check_nan=False):
                        fake_mode = detect_fake_mode()
                        if fake_mode is None:
                            raise AssertionError("fake_mode must not be None")
                        fake_mode.epoch += 1
                        out = PropagateUnbackedSymInts(mod).run(*args)
            else:
                out = mod(*args[params_len:], **kwargs)

        if strict_out_tuple and not isinstance(out, (tuple, list)):
            raise RuntimeError(
                "Graph output must be a (). This is so that we can avoid "
                "pytree processing of the outputs. Please change the module to "
                "have tuple outputs or use aot_module instead."
            )
        return out

    # Note [Preserving the nn module stack metadata during export non-strict mode]
    # This path is currently only used by the non-strict export flow,
    # where we cannot rely on dynamo to preserve nn stack metadata in our captured graph.
    # Instead, we stash the original user nn module here, and rely on `make_fx` to grab
    # this stashed module and use it to track nn module stack metadata
    if store_orig_mod and not hasattr(functional_call, "_orig_mod"):
        functional_call._orig_mod = mod  # type: ignore[attr-defined]

    return functional_call
<|endoftext|>"""
This module contains utility functions for working with joint FX graphs with descriptors
that are produced by AOTAutograd.  They will NOT work on generic FX graphs.  See also
:func:`torch._functorch.aot_autograd.aot_export_joint_with_descriptors`.  We also
recommend reading :mod:torch._functorch._aot_autograd.descriptors`.
"""

from typing import NoReturn

import torch.fx as fx

from .descriptors import (
    AOTInput,
    AOTOutput,
    BufferAOTInput,
    DifferentiableAOTInput,
    DifferentiableAOTOutput,
    GradAOTOutput,
    ParamAOTInput,
    PlainAOTInput,
    PlainAOTOutput,
    SubclassGetAttrAOTInput,
    SubclassGetAttrAOTOutput,
    TangentAOTInput,
)


def _raise_autograd_subclass_not_implemented(
    n: fx.Node, desc: AOTInput | AOTOutput
) -> NoReturn:
    raise RuntimeError(
        "Subclasses are currently not supported by this function, but a desugared subclass input "
        f"was found at {n} ({desc}).  The problem is "
        "that there may not necessarily be a 1-1 correspondence between primals/tangents/outputs/grads "
        "when subclasses are involved: for example, the primal might be a plain tensor "
        "but the tangent a tensor subclass that desugared into multiple plain tensors. "
        "It is not clear what exactly you would like this function to do in this case "
        "(Collect all nodes for the subclass together?  Match up the inner nodes if "
        "subclasses match exactly?)  If you have a concrete use case, please file an "
        "issue so we can understand it and design an API that works for your case."
    )


def get_all_input_and_grad_nodes(
    g: fx.Graph,
) -> dict[DifferentiableAOTInput, tuple[fx.Node, fx.Node | None]]:
    """
    Given a joint graph with descriptors (meta['desc'] on placeholders and
    output), returns the node for every input and its corresponding grad
    output node if it exists.  These tuples are in a dict that is indexed by
    the AOTInput descriptor that describes the input.

    NB: *all* forward tensor inputs are returned, including non-differentiable
    inputs (which simply have a None grad), so it is safe to use this function
    to perform operations on all inputs.  (Non-tensor inputs like symbolic
    integers, tokens or RNG state are NOT traversed by this function.)

    Args:
        g: The FX joint graph with descriptors

    Returns:
        A dictionary mapping each DifferentiableAOTInput descriptor to a tuple
        containing:
        - The input node itself
        - The grad (output) node if it exists, None otherwise

    Raises:
        RuntimeError: If the joint graph has subclass tensor inputs/outputs; this
        is not supported by API as there is not necessarily a 1-1 correspondence
        between inputs and grads when subclasses are involved.
    """
    input_index: dict[DifferentiableAOTInput, tuple[fx.Node, fx.Node | None]] = {}
    for n in g.nodes:
        if n.op == "placeholder":
            desc = n.meta["desc"]
            # Skip inputs that cannot possibly be differentiable
            if not isinstance(desc, DifferentiableAOTInput):
                continue
            if isinstance(desc, SubclassGetAttrAOTInput):
                _raise_autograd_subclass_not_implemented(n, desc)

            input_index[desc] = (n, None)
        elif n.op == "output":
            if "desc" not in n.meta:
                raise AssertionError(f"'desc' not in n.meta for {n}: {n.meta}")
            desc = n.meta["desc"]
            for sub_n, sub_desc in zip(n.args[0], desc):
                if isinstance(sub_desc, SubclassGetAttrAOTOutput):
                    _raise_autograd_subclass_not_implemented(sub_n, sub_desc)
                if isinstance(sub_desc, GradAOTOutput):
                    inp, grad = input_index[sub_desc.grad_of]
                    if grad is not None:
                        raise AssertionError(
                            f"grad already set for {sub_n}, {sub_desc}, {input_index}"
                        )
                    input_index[sub_desc.grad_of] = (inp, sub_n)
    return input_index


def get_all_output_and_tangent_nodes(
    g: fx.Graph,
) -> dict[DifferentiableAOTOutput, tuple[fx.Node, fx.Node | None]]:
    """Get all output nodes and their corresponding tangent nodes from a joint graph.

    Similar to get_all_input_and_grad_nodes, but returns output nodes paired with
    their tangent nodes (if they exist). This function traverses the graph to find
    all differentiable outputs and matches them with their corresponding tangent
    inputs used in forward-mode autodiff.

    NB: *all* forward tensor outputs are returned, including non-differentiable outputs,
    so you can use this function to perform operations on all outputs.

    Args:
        g: The FX joint graph with descriptors

    Returns:
        A dictionary mapping each DifferentiableAOTOutput descriptor to a tuple
        containing:
        - The output node itself
        - The tangent (input) node if it exists, None otherwise

    Raises:
        RuntimeError: If the joint graph has subclass tensor inputs/outputs; this
        is not supported by API as there is not necessarily a 1-1 correspondence
        between outputs and tangents when subclasses are involved.
    """
    output_index: dict[DifferentiableAOTOutput, tuple[fx.Node, fx.Node | None]] = {}
    for n in g.nodes:
        if n.op == "output":
            desc = n.meta["desc"]
            for sub_n, sub_d in zip(n.args[0], desc):
                # Skip outputs that cannot possibly be differentiable
                if not isinstance(sub_d, DifferentiableAOTOutput):
                    continue
                if isinstance(sub_d, SubclassGetAttrAOTOutput):
                    _raise_autograd_subclass_not_implemented(sub_n, sub_d)

                output_index[sub_d] = (sub_n, None)
    for n in g.nodes:
        if n.op == "placeholder":
            desc = n.meta["desc"]
            if isinstance(desc, SubclassGetAttrAOTInput):
                _raise_autograd_subclass_not_implemented(n, desc)
            if isinstance(desc, TangentAOTInput):
                out, tangent = output_index[desc.output]
                if tangent is not None:
                    raise AssertionError(
                        f"tangent already set for {n}, {desc}, {output_index}"
                    )
                output_index[desc.output] = (out, n)
    return output_index


def get_param_and_grad_nodes(
    graph: fx.Graph,
) -> dict[ParamAOTInput, tuple[fx.Node, fx.Node | None]]:
    """Get parameter nodes and their corresponding gradient nodes from a joint graph.

    Args:
        graph: The FX joint graph with descriptors

    Returns:
        A dictionary mapping each ParamAOTInput descriptor to a tuple containing:
        - The parameter input node
        - The gradient (output) node if it exists, None otherwise
    """
    return {
        desc: (n, g)
        for desc, (n, g) in get_all_input_and_grad_nodes(graph).items()
        if isinstance(desc, ParamAOTInput)
    }


def get_plain_input_and_grad_nodes(
    graph: fx.Graph,
) -> dict[PlainAOTInput, tuple[fx.Node, fx.Node | None]]:
    """Get plain input nodes and their corresponding gradient nodes from a joint graph.

    Args:
        graph: The FX joint graph with descriptors

    Returns:
        A dictionary mapping each PlainAOTInput descriptor to a tuple containing:
        - The plain input node
        - The gradient (output) node if it exists, None otherwise
    """
    return {
        desc: (n, g)
        for desc, (n, g) in get_all_input_and_grad_nodes(graph).items()
        if isinstance(desc, PlainAOTInput)
    }


def get_plain_output_and_tangent_nodes(
    graph: fx.Graph,
) -> dict[PlainAOTOutput, tuple[fx.Node, fx.Node | None]]:
    """Get plain output nodes and their corresponding tangent nodes from a joint graph.

    Args:
        graph: The FX joint graph with descriptors

    Returns:
        A dictionary mapping each PlainAOTOutput descriptor to a tuple containing:
        - The plain output node
        - The tangent (input) node if it exists, None otherwise
    """
    return {
        desc: (n, g)
        for desc, (n, g) in get_all_output_and_tangent_nodes(graph).items()
        if isinstance(desc, PlainAOTOutput)
    }


def _raise_fqn_subclass_not_implemented(
    n: fx.Node, desc: AOTInput | AOTOutput
) -> NoReturn:
    raise RuntimeError(
        "Subclasses are currently not supported by this function, but a desugared subclass input "
        f"was found at {n} ({desc}).  The problem is "
        "that there may not necessarily be a 1-1 correspondence between a FQN and a plain tensor "
        "when subclasses are involved: for example, a parameter that is a subclass "
        "would desugar into multiple plain tensors, which we can't uniquely assign the "
        "FQN to.  It's not clear what you want the API to do in this case: do you want to "
        "instead return a struct of nodes showing how to assemble the subclass?  But you "
        "don't (directly) have the metadata for the subclass?  If you have a concrete use "
        "case, please file an issue so we can understand it and design an API that works for your case."
    )


def get_named_param_nodes(graph: fx.Graph) -> dict[str, fx.Node]:
    """Get parameter nodes mapped by their fully qualified names.

    This function traverses the graph to find all parameter input nodes and
    returns them in a dictionary where keys are the parameter names (FQNs)
    and values are the corresponding FX nodes.

    Args:
        graph: The FX joint graph with descriptors

    Returns:
        A dictionary mapping parameter names (str) to their corresponding FX nodes.

    Raises:
        RuntimeError: If subclass tensors are encountered (not yet supported), as
        with subclasses a FQN does not necessarily map to a single plain tensor.
    """
    r = {}
    for n in graph.nodes:
        if n.op == "placeholder":
            desc = n.meta["desc"]
            if isinstance(desc, SubclassGetAttrAOTInput):
                _raise_fqn_subclass_not_implemented(n, desc)
            elif isinstance(desc, ParamAOTInput):
                r[desc.target] = n
    return r


def get_named_buffer_nodes(graph: fx.Graph) -> dict[str, fx.Node]:
    """Get buffer nodes mapped by their fully qualified names.

    This function traverses the graph to find all buffer input nodes and
    returns them in a dictionary where keys are the buffer names (FQNs)
    and values are the corresponding FX nodes.

    Args:
        graph: The FX joint graph with descriptors

    Returns:
        A dictionary mapping buffer names (str) to their corresponding FX nodes.

    Raises:
        RuntimeError: If subclass tensors are encountered (not yet supported), as
        with subclasses a FQN does not necessarily map to a single plain tensor.
    """
    r = {}
    for n in graph.nodes:
        if n.op == "placeholder":
            desc = n.meta["desc"]
            if isinstance(desc, SubclassGetAttrAOTInput):
                _raise_fqn_subclass_not_implemented(n, desc)
            elif isinstance(desc, BufferAOTInput):
                r[desc.target] = n
    return r


def get_param_nodes(graph: fx.Graph) -> list[fx.Node]:
    """Get all parameter nodes from a graph as a list.

    You can rely on this providing the correct order of parameters you need
    to feed into the joint graph (at the very beginning of the argument list,
    before buffers).

    Args:
        graph: The FX joint graph with descriptors

    Returns:
        A list of FX nodes representing all parameters in the graph.

    Raises:
        RuntimeError: If subclass tensors are encountered (not yet supported), as
        it is not clear if you wanted each individual constituent piece of the
        subclasses, or have them grouped up in some way.
    """
    return list(get_named_param_nodes(graph).values())


def get_buffer_nodes(graph: fx.Graph) -> list[fx.Node]:
    """Get all buffer nodes from a graph as a list.

    You can rely on this providing the correct order of buffers you need
    to feed into the joint graph (after parameters).

    Args:
        graph: The FX joint graph with descriptors

    Returns:
        A list of FX nodes representing all buffers in the graph.

    Raises:
        RuntimeError: If subclass tensors are encountered (not yet supported), as
        it is not clear if you wanted each individual constituent piece of the
        subclasses, or have them grouped up in some way.
    """
    return list(get_named_buffer_nodes(graph).values())
<|endoftext|>import operator
from collections.abc import Callable
from typing import Any, TYPE_CHECKING, TypeAlias

import torch.fx
import torch.fx.traceback
import torch.utils._pytree as pytree
from torch._dynamo.graph_utils import _get_flat_args
from torch._dynamo.variables.streams import get_current_stream, new_event
from torch.fx.node import map_arg
from torch.utils._ordered_set import OrderedSet
from torch.utils._runtime_estimation import (
    _FLOAT_TYPES,
    _IGNORE_OPS,
    get_compute_time,
    get_transfer_time,
)


if TYPE_CHECKING:
    from .schemas import ViewAndMutationMeta

from .indexed_dict import IndexedDict


aten = torch.ops.aten

Node: TypeAlias = torch.fx.Node
Graph: TypeAlias = torch.fx.Graph
Partition: TypeAlias = bool

FORWARD: Partition = False
BACKWARD: Partition = True
_EMPTY_LOCATIONS: frozenset[tuple[Partition, int]] = frozenset()

_SYNC_OPS = (
    torch.ops.streams.record_event.default,
    torch.ops.streams.wait_event.default,
    torch.ops.streams.wait_stream.default,
    torch.ops.streams.synchronize_event.default,
    torch.ops.streams.synchronize_device.default,
    torch.ops.streams.synchronize_stream.default,
)


def get_roofline_estimate(node: Node) -> float:
    if node.op != "call_function":
        raise AssertionError(f"non-func node in roofline estimate: {node.op}")

    def map_value(x: Any) -> Any:
        return x.meta.get("value", x) if isinstance(x, Node) else x

    func = node.target
    if func in _IGNORE_OPS:
        return 0.0

    mapped_args = torch.fx.map_arg(node.args, map_value)
    mapped_kwargs = torch.fx.map_arg(node.kwargs, map_value)
    flat_args_kwargs = [map_value(x) for x in _get_flat_args(node, {})]
    flat_outs, _ = pytree.tree_flatten(node.meta.get("value", node))
    out = node.meta.get("value", node)
    out_dtypes = {
        t.dtype
        for t in flat_outs
        if isinstance(t, torch.Tensor) and t.dtype in _FLOAT_TYPES
    }

    return (
        max(
            get_transfer_time(flat_args_kwargs, flat_outs),
            get_compute_time(func, mapped_args, mapped_kwargs, out, out_dtypes),
        )
        / 1e6
    )


def is_gradient_acc(node: Node) -> bool:
    return node.meta.get("is_gradient_acc", False)


def is_bwd_node(node: Node) -> Partition:
    tag = node.meta.get("partitioner_tag")
    return tag == "is_backward" or tag == "must_be_in_backward"


def get_device(node: Node) -> torch.device:
    return node.meta["val"].device


def get_stream(node: Node) -> int | None:
    maybe_annotation = node.meta.get("custom", None)
    if maybe_annotation is not None:
        return node.meta["custom"].get("stream", None)
    else:
        return None


def get_stream_or_current_stream(node: Node) -> int:
    ind = get_stream(node)
    if ind is None:
        ind = get_current_stream(get_device(node))
    return ind


def set_stream(node: Node, ind: int) -> None:
    if "custom" in node.meta:
        node.meta["custom"].update({"stream": ind})
    else:
        node.meta["custom"] = {"stream": ind}


def insert_record_event_after_node(graph: Graph, node: Node, event_ind: int) -> Node:
    with graph.inserting_after(node):
        node = graph.call_function(
            torch.ops.streams.record_event.default,
            (
                event_ind,
                get_stream_or_current_stream(node),
            ),
        )
        node.meta["partitioner_tag"] = "must_be_in_backward"

    return node


def insert_wait_event_before_node(graph: Graph, node: Node, event_ind: int) -> Node:
    with graph.inserting_before(node):
        node = graph.call_function(
            torch.ops.streams.wait_event.default,
            (
                event_ind,
                get_stream_or_current_stream(node),
            ),
        )
        node.meta["partitioner_tag"] = "must_be_in_backward"

    return node


def populate_stream_timeline(
    stream_to_timeline: dict[int | None, IndexedDict[Node, float]],
    graph: Graph,
    stream_index: int | None,
) -> IndexedDict[Node, float]:
    if stream_index not in stream_to_timeline:
        stream_to_timeline[stream_index] = IndexedDict()
        total_time = 0.0
        for node in graph.nodes:
            # mlazos: not sure if we should include forward here too but don't think it matters
            if (
                node.op == "call_function"
                and is_bwd_node(node)
                and get_stream(node) == stream_index
            ):
                total_time += get_roofline_estimate(node)
                stream_to_timeline[stream_index][node] = (
                    total_time  # NB: total time includes the node's runtime
                )

    return stream_to_timeline[stream_index]


# NB: we start all estimates at 0, estimating the total runtime of each stream with timestamps at each node
# we then try and use these timestamps to estimate when to deallocate tensors used in side streams
# See https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html#torch.Tensor.record_stream
# for details on the problem being addressed. Rather than using the automatic memory management approach of record_stream
# we attempt to find the point which to deallocate based on the estimated timestamps.
def handle_synced_deallocation(
    graph: Graph,
    stream_to_exec_trace: dict[int | None, IndexedDict[Node, float]],
    node: Node,
    last_usage: Node,
) -> None:
    if not is_bwd_node(node):
        raise AssertionError(
            "synced allocations should only be handled on backward nodes"
        )
    if not is_bwd_node(last_usage):
        raise AssertionError(
            "synced allocations should only be handled on backward nodes"
        )
    allocating_stream = get_stream(node)
    side_stream = get_stream(last_usage)
    if allocating_stream == side_stream:
        raise AssertionError(
            "allocating and side stream should be different for synced deallocations"
        )
    if not torch.accelerator.is_available():
        # fallback to record_stream in this case
        with graph.inserting_after(node):
            rs = graph.call_function(
                torch.ops.streams.record_stream.default,
                (
                    node,
                    get_stream_or_current_stream(last_usage),
                ),
                {},
            )
        rs.meta["partitioner_tag"] = "must_be_in_backward"

    allocating_stream_trace = populate_stream_timeline(
        stream_to_exec_trace, graph, allocating_stream
    )
    side_stream_trace = populate_stream_timeline(
        stream_to_exec_trace, graph, side_stream
    )

    alloc_ptr = node
    target_side_stream_time = side_stream_trace[last_usage]
    # linear search from first usage of tensor to a point in time after the side stream has finished
    while alloc_ptr is not None:
        alloc_time = allocating_stream_trace[alloc_ptr]

        if alloc_time >= target_side_stream_time:
            break
        elif alloc_time < target_side_stream_time:
            next_ptr = allocating_stream_trace.next_key(alloc_ptr)
            if next_ptr is not None:
                alloc_ptr = next_ptr
            else:
                break

    wait_event = new_event()
    record_node = insert_record_event_after_node(graph, last_usage, wait_event)
    with graph.inserting_after(max(alloc_ptr, record_node)):
        sd = graph.call_function(
            torch.ops.streams.sync_dealloc.default,
            (wait_event, get_stream_or_current_stream(alloc_ptr), node),
            {},
        )
        sd.meta["partitioner_tag"] = "must_be_in_backward"


def insert_sync(
    graph: Graph,
    consumer: Node,
    producer: Node,
    node_to_wait_event_ind: dict[Node, int],
) -> None:
    if producer not in node_to_wait_event_ind:
        node_to_wait_event_ind[producer] = new_event()

        insert_record_event_after_node(
            graph, producer, node_to_wait_event_ind[producer]
        )
        insert_wait_event_before_node(graph, consumer, node_to_wait_event_ind[producer])


def assign_backward_streams(gm: torch.fx.GraphModule) -> None:
    """Assigns backward streams to gradient accumulation nodes"""

    # NB: iterate in reverse order to more closely match eager
    # the user node stream will be populated first
    for node in reversed(list(gm.graph.nodes)):
        if is_gradient_acc(node):
            # Accumulation stream selection. Follow the rules from top to bottom to determine the accumulation stream:
            # 1. Match first stream assignment of the first user with a stream
            # 2. Match first stream assignment encountered in the args from left to right
            # This differs from eager in some cases:
            # Specifically the eager code uses the autograd node to determine the stream,
            # crucially this does not necessarily correspond to the FX graph node. For example,
            # in the backward for an add node with a constant we will passthrough and during backward tracing,
            # no op will be added to the FX graph, so our stream assignment will differ in this case.
            gradients = _get_flat_args(node, {})
            users = list(node.users.keys())

            # All gradients will be on same device, they will be coerced if they were not with a .to() node
            for neighbor in users + gradients:
                ind = get_stream(neighbor)
                if ind is not None:
                    set_stream(node, ind)
                    break


def insert_backward_syncs(gm: torch.fx.GraphModule) -> None:
    """Inserts stream syncs for backward nodes if consumer and producer are on different streams"""
    node_to_wait_event_ind: dict[Node, int] = {}
    for node in gm.graph.nodes:
        if node.op == "call_function" and is_bwd_node(node):
            flat_args = _get_flat_args(node, {})
            cur_node_stream = get_stream(node)

            for arg in flat_args:
                if arg.op == "call_function" and is_bwd_node(arg):
                    arg_stream = get_stream(arg)
                    if arg_stream != cur_node_stream and get_device(arg).type != "cpu":
                        insert_sync(gm.graph, node, arg, node_to_wait_event_ind)


def sync_deallocations(gm: torch.fx.GraphModule) -> None:
    """Handles https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html#torch.Tensor.record_stream"""
    # Note: this is only needed if the last usage of a tensor is on a stream other than
    # the stream the tensor was allocated on

    # an estimated timestamp from the beginning of graph execution (assuming 0 CPU overhead)
    # I think this is fine because you should have large tensors if you're using streams
    # although perhaps I could add a constant 10us per op ahead of the first stream op?
    # a trace of all the nodes running in a given stream
    stream_to_exec_trace: dict[int | None, IndexedDict[Node, float]] = {}
    for node in gm.graph.nodes:
        if node.op == "call_function" and is_bwd_node(node):
            allocating_stream = get_stream(node)
            users = list(node.users.keys())
            if not users:
                continue
            last_user = max(user for user in users)
            if last_user.op == "output":
                continue
            side_stream = get_stream(last_user)
            if allocating_stream != side_stream:
                handle_synced_deallocation(
                    gm.graph, stream_to_exec_trace, node, last_user
                )


def assign_epilogue_copy_streams(gm: torch.fx.GraphModule) -> None:
    for epi_copy in gm.graph.find_nodes(op="call_function", target=aten.copy_.default):
        arg_stream = get_stream(epi_copy.args[1])
        copy_stream = get_stream(epi_copy)
        if arg_stream != copy_stream:
            set_stream(epi_copy, get_stream_or_current_stream(epi_copy.args[1]))


def populate_fw_metadata_with_stream_indices(
    gm: torch.fx.GraphModule, fw_metadata: "ViewAndMutationMeta"
) -> None:
    """
    Populates fw_metadata.mutated_inp_stream_indices with stream indices from the compiled graph.

    The forward graph outputs are structured as:
    (*mutated_inputs, *user_outputs, *intermediate_bases, *saved_tensors, *saved_symints)

    We extract the stream index for each mutated input from the graph's output node.
    """

    num_mutated_inps = fw_metadata.num_mutated_inp_runtime_indices
    if num_mutated_inps == 0:
        fw_metadata.mutated_inp_stream_indices = []
        return

    # Find the output node in the graph
    output_node = None
    for node in gm.graph.find_nodes(op="output"):
        output_node = node
        break

    if output_node is None:
        raise AssertionError(
            "No output node found in the graph when extracting stream indices"
        )

    # The output node's args[0] is a tuple/list of all outputs
    output_args = output_node.args[0]

    # Extract stream indices for the first num_mutated_inps outputs
    stream_indices = []
    for i in range(num_mutated_inps):
        if i < len(output_args):
            output_arg = output_args[i]
            # Get the stream index from the node metadata
            stream_idx = (
                get_stream(output_arg)
                if isinstance(output_arg, torch.fx.Node)
                else None
            )
            stream_indices.append(stream_idx)
        else:
            stream_indices.append(None)

    fw_metadata.mutated_inp_stream_indices = stream_indices


def _expand_dict_returning_deps(
    deps: list[Node],
    visited: set[Node],
    graph: torch.fx.Graph,
    sync_node: Node,
) -> list[Node]:
    """Expand triton_kernel_wrapper_functional dict deps into per-key getitems.

    ``triton_kernel_wrapper_functional`` returns a flat ``dict[str, Tensor]``.
    If such a node is threaded through ``control_deps`` as a pass-through
    value, ``decompose_triton_kernel_wrapper_functional`` later replaces it
    with a raw Python dict, corrupting every user that expected an FX Node.

    To avoid this, we insert ``operator.getitem`` nodes *before* the sync for
    each after-sync getitem user, so that only tensor-valued nodes are passed
    through ``control_deps``.  The dict node may still appear in
    ``additional_deps`` (ordering-only) which is safe since that tuple is
    never decomposed.
    """
    expanded: list[Node] = []
    for dep in deps:
        if not (
            dep.op == "call_function"
            and dep.target is torch.ops.higher_order.triton_kernel_wrapper_functional
        ):
            expanded.append(dep)
            continue

        after_sync_getitems = [
            user
            for user in dep.users
            if user not in visited
            and user.op == "call_function"
            and user.target is operator.getitem
        ]
        if not after_sync_getitems:
            expanded.append(dep)
            continue

        with graph.inserting_before(sync_node):
            for old_gi in after_sync_getitems:
                key = old_gi.args[1]
                new_gi = graph.call_function(operator.getitem, args=(dep, key))
                new_gi.meta.update(old_gi.meta)
                if isinstance(new_gi.meta.get("val"), dict):
                    raise AssertionError(
                        f"nested dict in triton kernel output for key {key}"
                    )
                visited.add(new_gi)
                expanded.append(new_gi)
                old_gi.replace_all_uses_with(new_gi)
                graph.erase_node(old_gi)

    return expanded


def _wrap_sync_node(
    gm: torch.fx.GraphModule,
    sync_node: Node,
    deps_before_sync: list[Node],
    visited: set[Node],
    partition_scoped_deps: set[Node] | None = None,
) -> tuple[Node, list[Node]]:
    """
    Core logic: wrap a single sync node in control_deps.

    Returns (control_deps_node, passthrough_getitems) where passthrough_getitems
    are the getitem nodes that thread dependencies through the control_deps node.
    ``visited`` is the set of nodes at or before the sync node in graph order,
    used to distinguish pre-sync vs post-sync users.
    """
    from torch._inductor.fx_passes.control_dependencies import (
        _create_subgraph_for_node,
        control_deps,
        get_subgraph_name,
    )

    graph = gm.graph
    if partition_scoped_deps is None:
        partition_scoped_deps = set()

    sync_is_bwd = is_bwd_node(sync_node)

    def _user_rewrite_pred(dep: Node) -> Callable[[Node], bool]:
        if dep in partition_scoped_deps:

            def pred(user: Node) -> bool:
                if user in visited:
                    return False
                if user.op == "output":
                    return sync_is_bwd
                return is_bwd_node(user) == sync_is_bwd

            return pred

        dep_is_bwd = is_bwd_node(dep)

        def pred(user: Node) -> bool:
            return user not in visited and (user.op != "output" or dep_is_bwd)

        return pred

    # Use dep.users to find deps with uses after the sync — avoids a forward walk.
    deps_with_uses_after_sync = []
    for dep in deps_before_sync:
        pred = _user_rewrite_pred(dep)
        if any(pred(user) for user in dep.users):
            deps_with_uses_after_sync.append(dep)

    # Expand dict-returning nodes (triton_kernel_wrapper_functional) into
    # individual getitem outputs.  Passing a dict-valued node through
    # control_deps breaks post-grad passes: reinplace + decomposition replace
    # the dict node with a Python dict, corrupting any user that expected an
    # FX Node.  Unwrapping here ensures only tensor-valued nodes flow through.
    deps_with_uses_after_sync = _expand_dict_returning_deps(
        deps_with_uses_after_sync, visited, graph, sync_node
    )

    # Create subgraph that executes sync and passes through only used dependencies
    subgraph_module = _create_subgraph_for_node(
        graph, sync_node, deps_with_uses_after_sync
    )
    subgraph_attr_name = get_subgraph_name(gm, sync_node.name)
    setattr(gm, subgraph_attr_name, subgraph_module)

    # Create control_deps call
    # Note: sync nodes (record_event/wait_event) only take int args, no Node args.
    with graph.inserting_before(sync_node):
        get_subgraph = graph.get_attr(subgraph_attr_name)
        control_deps_node = graph.call_function(
            control_deps,
            args=(
                tuple(deps_before_sync),  # additional_deps (all deps for ordering)
                get_subgraph,  # subgraph
                *deps_with_uses_after_sync,  # only pass through deps that are used
            ),
            kwargs={},
        )

    # Propagate partitioner_tag so the partitioner can distinguish
    # forward vs backward sync control_deps during graph extraction.
    if "partitioner_tag" in sync_node.meta:
        control_deps_node.meta["partitioner_tag"] = sync_node.meta["partitioner_tag"]

    # Mark newly created nodes as visited so subsequent syncs don't
    # misclassify them as "after the sync" during replacement.
    visited.add(get_subgraph)
    visited.add(control_deps_node)

    # The output is (sync_result, *deps_with_uses_after_sync)
    # Reuse the val the subgraph output already carries: the min-cut
    # partitioner's _size_of() raises on any node without `val`, and its escape
    # hatch covers only get_attr and zero-return OpOverloads, not a schema-less
    # HOP.  The subgraph output is unwrapped when it has no pass-through deps.
    sg_val = subgraph_module.graph.output_node().meta.get("val")
    control_deps_node.meta["val"] = sg_val if isinstance(sg_val, tuple) else (sg_val,)

    # Create getitem nodes only for dependencies that have uses after sync
    replacements: dict[Node, Node] = {}
    with graph.inserting_after(control_deps_node):
        for i, dep in enumerate(deps_with_uses_after_sync):
            getitem_node = graph.call_function(
                operator.getitem,
                args=(control_deps_node, i + 1),  # +1 because index 0 is sync result
            )
            getitem_node.meta.update(dep.meta)
            # The getitem is a passthrough projection of an existing value, not
            # a node that *produces* an unbacked symbol - the original ``dep``
            # (still inside the subgraph) is the binding site. Leaving the
            # binding on the getitem trips Inductor's run_node assertion that
            # ``new_unbacked_defs >= renamed_unbacked_bindings``.
            getitem_node.meta.pop("unbacked_bindings", None)
            replacements[dep] = getitem_node
            visited.add(getitem_node)

    # Replace uses of dependencies that come after sync_node.
    # Use map_arg to handle nested structures (e.g. output node's list args).
    for dep, getitem_node in replacements.items():
        pred = _user_rewrite_pred(dep)
        for user in list(dep.users.keys()):
            if user is control_deps_node:
                continue
            if not pred(user):
                continue

            def _replace(n: Node) -> Node:
                return getitem_node if n is dep else n

            if (
                user.op == "call_function"
                and user.target is torch.ops.aten.copy_.default
                and user.args[0] is dep
                and dep in partition_scoped_deps
            ):
                # AOT functionalization leaves keep_input_mutations epilogue
                # copy_ as the only writer of a partition-scoped placeholder.
                # Preserve the destination identity used by mutation bookkeeping.
                user.args = (dep, *map_arg(user.args[1:], _replace))
            else:
                user.args = map_arg(user.args, _replace)
            user.kwargs = map_arg(user.kwargs, _replace)

    # Remove original sync node
    sync_node.replace_all_uses_with(control_deps_node)
    graph.erase_node(sync_node)
    return control_deps_node, list(replacements.values())


def _collect_sync_forward_deps(
    graph: torch.fx.Graph,
) -> tuple[dict[Node, OrderedSet[Node]], dict[Node, OrderedSet[Node]]]:
    """Collect values that must cross sync boundaries.

    ``live_inputs[partition][stream]`` tracks values whose passthrough can anchor
    later work in that partition on that stream. Demand labels propagate through
    producers while each producer also seeds its own execution stream. This is a
    single reverse liveness traversal; the forward pass still handles sync/event
    state introduced while rewriting the graph.
    """
    stream_sync_deps: dict[Node, OrderedSet[Node]] = {}
    full_barrier_deps: dict[Node, OrderedSet[Node]] = {}
    live_inputs: dict[Partition, dict[int, OrderedSet[Node]]] = {
        FORWARD: {},
        BACKWARD: {},
    }
    locations: dict[Node, OrderedSet[tuple[Partition, int]]] = {}
    full_barriers = (
        torch.ops.streams.synchronize_device.default,
        torch.ops.streams.synchronize_event.default,
        torch.ops.streams.synchronize_stream.default,
    )

    def _add_live(node: Node, partition: Partition, stream: int) -> None:
        stream_inputs = live_inputs[partition].get(stream)
        if stream_inputs is None:
            stream_inputs = live_inputs[partition][stream] = OrderedSet()
        stream_inputs.add(node)
        node_locations = locations.get(node)
        if node_locations is None:
            node_locations = locations[node] = OrderedSet()
        node_locations.add((partition, stream))

    for node in reversed(graph.nodes):
        if node.op == "call_function" and node.target in full_barriers:
            deps = OrderedSet(
                dep
                for stream_inputs in live_inputs[is_bwd_node(node)].values()
                for dep in stream_inputs
            )
            if deps:
                full_barrier_deps[node] = deps
            continue
        if (
            node.op == "call_function"
            and node.target is torch.ops.streams.wait_stream.default
        ):
            waiting_stream: int = node.args[0]  # type: ignore[assignment]
            deps = live_inputs[is_bwd_node(node)].get(waiting_stream)
            if deps:
                stream_sync_deps[node] = deps.copy()
            continue
        if (
            node.op == "call_function"
            and node.target is torch.ops.streams.wait_event.default
        ):
            waiting_stream: int = node.args[1]  # type: ignore[assignment]
            deps = live_inputs[is_bwd_node(node)].get(waiting_stream, ())
            graph_inputs = OrderedSet(dep for dep in deps if dep.op == "placeholder")
            if graph_inputs:
                stream_sync_deps[node] = graph_inputs
            continue
        if node.op != "call_function" or node.target in _SYNC_OPS:
            continue

        inherited_locations = locations.pop(node, _EMPTY_LOCATIONS)
        for partition, stream in inherited_locations:
            live_inputs[partition][stream].discard(node)
        execution_stream = get_stream(node)
        if execution_stream is None:
            execution_stream = 0
        target_locations = OrderedSet(
            (*inherited_locations, (is_bwd_node(node), execution_stream))
        )

        def _collect(n: Node) -> Node:
            if "val" in n.meta:
                for partition, stream in target_locations:
                    _add_live(n, partition, stream)
            return n

        map_arg(node.args, _collect)
        map_arg(node.kwargs, _collect)
    return stream_sync_deps, full_barrier_deps


def _collect_full_barrier_deps(
    stream_to_nodes: dict[int | None, list[Node]],
    stream_sync_deps: dict[int | None, list[Node]],
    bwd: bool = False,
) -> list[Node]:
    """Collect all still-live cross-stream deps for a full CPU barrier
    (synchronize_stream/device/event).

    These ops block the CPU, so a consumer is only ordered after them by an
    explicit control_deps edge (nothing on the consumer's stream waits for a CPU
    barrier). Includes live compute nodes (stream_to_nodes) plus the ctrl/
    passthrough nodes accumulated by prior sync ops that already cleared
    stream_to_nodes (stream_sync_deps), deduped by identity preserving order.

    ``bwd`` marks a backward barrier: its accumulated sync deps are filtered to
    backward nodes only, so it never depends on a forward control_deps node.
    That edge would cross the fwd/bwd boundary onto a node that is neither
    saveable nor recomputable, making the min-cut partitioner infeasible.
    Forward compute deps (stream_to_nodes) are kept -- backward legitimately
    depends on forward activations.
    """
    all_deps = [n for nodes in stream_to_nodes.values() for n in nodes]
    seen = set(all_deps)
    for deps in stream_sync_deps.values():
        for dep in deps:
            if dep in seen or (bwd and not is_bwd_node(dep)):
                continue
            all_deps.append(dep)
            seen.add(dep)
    return all_deps


def wrap_all_sync_nodes_with_control_deps(gm: torch.fx.GraphModule) -> None:
    """
    Single-pass wrap of all sync nodes in control_deps.

    Iterates through the graph once, accumulating per-stream node lists.
    When a sync node is encountered, it is wrapped using the accumulated deps
    for that stream, then the deps are reset to the control_deps node
    (maintaining the ordering chain for subsequent syncs on the same stream).
    """
    graph = gm.graph
    if len(graph.nodes) == 0:
        raise RuntimeError("Expected a non-empty graph")

    stream_sync_forward_deps, full_barrier_forward_deps = _collect_sync_forward_deps(
        graph
    )

    stream_to_nodes: dict[int | None, list[Node]] = {}
    # Maps event_index to its latest ordering anchor: either the control_deps that
    # wrapped record_event or a raw record_event with no dependencies.
    event_to_anchor: dict[int, Node] = {}
    # Maps event_index -> getitem nodes threaded through record_event's control_deps,
    # so synchronize_event can thread them through to subsequent ops.
    event_to_passthrough: dict[int, list[Node]] = {}
    # Maps event_index -> stream that the event was recorded on,
    # so synchronize_event can infer its stream.
    event_to_stream: dict[int, int | None] = {}
    # Tracks the most recent control_deps node and its passthrough
    # getitems per stream. synchronize_stream/device consults this to
    # depend on prior sync ops (ctrl_node for ordering) and thread
    # cross-stream data through (passthrough for data edges), even
    # after stream_to_nodes has been cleared by those sync ops.
    stream_sync_deps: dict[int | None, list[Node]] = {}
    visited: set[Node] = set()
    found_sync = False

    # Walk the node linked-list manually so we can mutate the graph
    # (wrapping sync nodes inserts/erases nodes) without losing our place.
    node = next(iter(graph.nodes))
    while node.op != "root":
        next_node = node.next
        visited.add(node)

        if node.op == "call_function":
            if node.target in _SYNC_OPS:
                # synchronize_device and synchronize_stream block the CPU,
                # so all subsequent kernel launches are host-ordered after
                # them. Treat both as full barriers across all streams.
                if node.target in (
                    torch.ops.streams.synchronize_device.default,
                    torch.ops.streams.synchronize_stream.default,
                ):
                    all_stream_deps = _collect_full_barrier_deps(
                        stream_to_nodes, stream_sync_deps, bwd=is_bwd_node(node)
                    )
                    existing_deps = set(all_stream_deps)
                    for dep in full_barrier_forward_deps.get(node, ()):
                        if dep not in existing_deps:
                            all_stream_deps.append(dep)
                            existing_deps.add(dep)
                    if all_stream_deps:
                        found_sync = True
                        ctrl_node_sync, passthrough_sync = _wrap_sync_node(
                            gm, node, all_stream_deps, visited
                        )
                    else:
                        ctrl_node_sync = None
                        passthrough_sync: list[Node] = []
                    stream_to_nodes.clear()
                    stream_sync_deps.clear()
                    sync_anchor = ctrl_node_sync if ctrl_node_sync is not None else node
                    stream_sync_deps[None] = [sync_anchor, *passthrough_sync]
                    while (
                        getattr(next_node, "_erased", False) and next_node.op != "root"
                    ):
                        next_node = next_node.next
                    node = next_node
                    continue

                # wait_stream(waiting, waited_on) creates a dependency on
                # the waited_on stream: all prior work there must complete
                # before the waiting stream proceeds.
                if node.target is torch.ops.streams.wait_stream.default:
                    waiting_stream: int = node.args[0]  # type: ignore[assignment]
                    waited_on_stream: int = node.args[1]  # type: ignore[assignment]
                    deps_before_sync = list(stream_to_nodes.get(waited_on_stream, ()))
                    if waiting_stream != waited_on_stream:
                        deps_before_sync.extend(stream_to_nodes.get(waiting_stream, ()))
                    if None in stream_to_nodes and waited_on_stream is not None:
                        deps_before_sync.extend(stream_to_nodes[None])
                    participating_streams = dict.fromkeys(
                        (waited_on_stream, waiting_stream, None)
                    )
                    prior_syncs = list(
                        dict.fromkeys(
                            dep
                            for stream in participating_streams
                            for dep in stream_sync_deps.get(stream, ())
                        )
                    )
                    if is_bwd_node(node):
                        prior_syncs = [n for n in prior_syncs if is_bwd_node(n)]
                    deps_before_sync.extend(prior_syncs)
                    # Also include inputs of subsequent waiting-stream nodes
                    # so they get threaded through control_deps, creating a
                    # forward ordering constraint (waiting-stream work must
                    # come after wait_stream).
                    existing_deps = set(deps_before_sync)
                    for dep in stream_sync_forward_deps.get(node, ()):
                        if dep not in existing_deps:
                            deps_before_sync.append(dep)
                            existing_deps.add(dep)
                    if deps_before_sync:
                        found_sync = True
                        ctrl_node_ws, passthrough_ws = _wrap_sync_node(
                            gm, node, deps_before_sync, visited
                        )
                    else:
                        ctrl_node_ws = None
                        passthrough_ws: list[Node] = []
                    wait_anchor = ctrl_node_ws if ctrl_node_ws is not None else node
                    # Key under both streams: the wait executes on waiting_stream
                    # and subsumes the waited-on work used to anchor it.
                    stream_sync_deps[waiting_stream] = [
                        wait_anchor,
                        *passthrough_ws,
                    ]
                    stream_sync_deps[waited_on_stream] = [
                        wait_anchor,
                        *passthrough_ws,
                    ]
                    stream_to_nodes[waited_on_stream] = []
                    if None in stream_to_nodes:
                        stream_to_nodes[None] = []
                    while (
                        getattr(next_node, "_erased", False) and next_node.op != "root"
                    ):
                        next_node = next_node.next
                    node = next_node
                    continue

                event_index: int = node.args[0]  # type: ignore[assignment]

                # synchronize_event blocks the CPU thread, so it acts
                # as a barrier across all streams. Collect deps from every
                # stream and reset them all afterward. If the event was
                # recorded externally, thread the graph inputs through so
                # that any post-sync uses depend on the synchronize.
                partition_scoped_deps: list[Node] = []
                if node.target is torch.ops.streams.synchronize_event.default:
                    sync_stream: int | None = event_to_stream.get(event_index)
                    # Like synchronize_stream/device, this is a CPU-side barrier,
                    # so fold in still-live cross-stream data accumulated by prior
                    # sync ops (which cleared stream_to_nodes).
                    all_stream_deps = _collect_full_barrier_deps(
                        stream_to_nodes, stream_sync_deps, bwd=is_bwd_node(node)
                    )
                    deps_before_sync = [
                        *all_stream_deps,
                        *full_barrier_forward_deps.get(node, ()),
                    ]
                else:
                    sync_stream = node.args[1]  # type: ignore[assignment]
                    deps_before_sync = list(stream_to_nodes.get(sync_stream, ()))
                    partition_scoped_deps = (
                        list(stream_sync_forward_deps.get(node, ()))
                        if node.target is torch.ops.streams.wait_event.default
                        and event_index not in event_to_anchor
                        else []
                    )
                    deps_before_sync.extend(partition_scoped_deps)
                    # Nodes without explicit stream annotation (custom.stream=None)
                    # run on the current/default stream. Include them when the sync
                    # op references a stream, since the unannotated nodes are
                    # implicitly on that stream.
                    if None in stream_to_nodes and sync_stream is not None:
                        deps_before_sync.extend(stream_to_nodes[None])
                    # Order this sync op after the prior sync on its own stream
                    # (and after any full barrier, keyed under None). Without this,
                    # a record/wait whose stream_to_nodes was reset by an earlier
                    # sync would have no deps, stay unwrapped, and float as a bare
                    # side-effectful node -- capturing nothing, the same failure as
                    # an unanchored synchronize_stream.
                    prior_syncs = [
                        *stream_sync_deps.get(sync_stream, ()),
                        *stream_sync_deps.get(None, ()),
                    ]
                    # A backward sync must anchor only on backward sync nodes:
                    # depending on a forward control_deps node crosses the fwd/bwd
                    # boundary onto a node that is neither saveable nor
                    # recomputable, which makes the min-cut partitioner infeasible.
                    if is_bwd_node(node):
                        prior_syncs = [n for n in prior_syncs if is_bwd_node(n)]
                    deps_before_sync.extend(prior_syncs)

                # For wait_event and synchronize_event, depend on the matching
                # record_event's control_deps node (ordering) and thread its
                # passthrough getitems through (data edge), so consumers of
                # recorded values are rewired through the wait/synchronize
                # and the scheduler sees the dependency.
                if node.target in (
                    torch.ops.streams.wait_event.default,
                    torch.ops.streams.synchronize_event.default,
                ):
                    if event_index in event_to_anchor:
                        deps_before_sync = [
                            event_to_anchor[event_index],
                            *deps_before_sync,
                        ]
                    if event_index in event_to_passthrough:
                        deps_before_sync = [
                            *deps_before_sync,
                            *event_to_passthrough[event_index],
                        ]
                elif (
                    node.target is torch.ops.streams.record_event.default
                    and event_index in event_to_anchor
                ):
                    # Re-recording replaces an event's state, but the host-side
                    # record operations must retain program order even when the
                    # new record is on an otherwise-empty stream.
                    deps_before_sync.insert(0, event_to_anchor[event_index])

                # Deduplicate (preserving order): the folded stream_sync_deps and
                # the event's own anchor/passthrough can overlap.
                deps_before_sync = list(OrderedSet(deps_before_sync))

                if deps_before_sync:
                    found_sync = True
                    ctrl_node, passthrough = _wrap_sync_node(
                        gm,
                        node,
                        deps_before_sync,
                        visited,
                        set(partition_scoped_deps)
                        if node.target is torch.ops.streams.wait_event.default
                        else None,
                    )
                else:
                    ctrl_node = None
                    passthrough: list[torch.fx.Node] = []

                if node.target is torch.ops.streams.record_event.default:
                    event_to_stream[event_index] = sync_stream
                    event_to_anchor[event_index] = (
                        ctrl_node if ctrl_node is not None else node
                    )
                    event_to_passthrough[event_index] = passthrough

                sync_anchor = ctrl_node if ctrl_node is not None else node
                stream_sync_deps[sync_stream] = [sync_anchor, *passthrough]

                # Reset: ops between this sync and the next will accumulate
                # fresh. Ordering with prior ops is already enforced because
                # their uses were rewired through getitems from control_deps.
                # synchronize_event is a full barrier, so its own passthrough
                # subsumes all prior per-stream data: clear then re-seed. Key the
                # reseed under None (like synchronize_stream/device) so a later
                # record/wait on any stream chains after it.
                if node.target is torch.ops.streams.synchronize_event.default:
                    stream_to_nodes.clear()
                    stream_sync_deps.clear()
                    stream_sync_deps[None] = [sync_anchor, *passthrough]
                else:
                    stream_to_nodes[sync_stream] = []
                    if None in stream_to_nodes:
                        stream_to_nodes[None] = []
            elif "val" in node.meta:
                stream = get_stream(node)
                stream_to_nodes.setdefault(stream, []).append(node)

        # _wrap_sync_node may erase nodes after the sync (e.g.
        # _expand_dict_returning_deps erases getitems).  Skip erased nodes
        # so we don't re-process a dead node into stream_to_nodes.
        while getattr(next_node, "_erased", False) and next_node.op != "root":
            next_node = next_node.next
        node = next_node

    if found_sync:
        gm.recompile()
<|endoftext|>"""Lower a GraphModule to a self-contained Python module via AOTAutograd + Inductor.

This is the AOT half of the backend contract behind ``torch.compiler.precompile``:

    python_code, cache = compile_to_python(gm, example_inputs)

``inductor.compile_to_python`` produces only the inner Inductor ``call`` (kernels
for the post-AOTAutograd dense graph). This module wraps that with the prelude /
epilogue (input-mutation reflection, output-alias regen, subclass wrap/unwrap,
...) by COMPOSING AOTAutograd's own codegen'd runtime-wrapper source -- captured
during compile -- rather than reimplementing it. Every wrapper is spliced verbatim as a
real top-level ``def``, with its closed-over globals hoisted to module-scope assignments,
so the module reads as ordinary code. These nest in the SAME order the runtime builds:
the INNER chain wrappers (subclass / functionalized-RNG) wrap the inner ``call`` and are
composed inside the orchestration, while the OUTER ``CompilerWrapper``s (dedup / synthetic
base) wrap the orchestration itself (via a single-arg adapter over it). Cross-wrapper references, the inner ``call`` chain,
public helpers, and baked metadata objects (reconstructed as source -- see
``_emit_value``) are wired by name; a guard rejects the rare case where a wrapper def
name or hoisted global would collide with another top-level name (a sibling wrapper or
an inner-module binding) rather than silently rebinding one.
The result is a standalone module exposing ``call(flat_inputs) -> outputs`` that
runs on its own (JIT-compiling kernels); ``cache`` is an opaque acceleration (or
None).

Baked metadata is emitted as plain Python source (no pickle / base64 blobs), so the
generated module is fully auditable and exec'ing it never invokes ``pickle.loads``.
A leaf that cannot be expressed as source raises NotImplementedError rather than
falling back to an opaque embedding.

Contract note: the standalone ``call`` deliberately substitutes ``nullcontext`` / a
no-op for the runtime's first-invocation context and profiler prologue, dropping the
cold-start custom-op aliasing analysis and the profiler prologue -- both diagnostics
with no effect on numerics (see the generated-call emission site). One caveat: that
dropped first-invocation custom-op aliasing analysis can itself RAISE under
``config.error_on_custom_op_aliasing`` (default on in CI), so a graph whose custom op
violates the aliasing contract runs SILENTLY in the standalone artifact where the
eager / compiled path would error -- an intentional trade-off, not a numerics bug.
"""

from __future__ import annotations

import ast
import re
import threading
from typing import Any, TYPE_CHECKING

from .codegen import capture_generated_sources, GeneratedSource
from .source_emit import _REBUILD_HELPER, emit_value


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from torch.fx import GraphModule


# Serializes compile_to_python: the wrapper-source capture is thread-local, but the
# AOTAutograd capture pass and the inner inductor compile swap process-global cache state
# (see the THREADING note on compile_to_python), so concurrent compiles must not overlap.
# An RLock (not a
# plain Lock) because this entry point is re-entrant on a single thread: a custom
# backend or inductor pass invoked while the lock is held may call back into this
# lowering to compile a subgraph on the SAME thread, and a plain Lock would
# self-deadlock on that re-entry. On-thread re-entry is safe here: the capture sink and
# the cache-state swap (CacheArtifactManager.with_fresh_cache) are each self-contained
# per call, so a nested compile neither corrupts the outer capture nor the outer cache
# scope.
_COMPILE_LOCK = threading.RLock()


# ======================================================================
# WHAT IS GOING ON HERE: composing AOTAutograd's runtime wrappers as source
# ======================================================================
#
# AOTAutograd does not hand back a single flat function. The dense graph Inductor
# compiles (the inner ``call``) is only the arithmetic core; around it AOTAutograd
# wraps a prelude/epilogue that does what the core cannot express: reflecting input
# mutations back onto the caller's tensors, regenerating outputs that alias an input
# or each other, wrapping/unwrapping tensor subclasses, de-duplicating aliased
# inputs, and threading functionalized RNG state. At runtime AOTAutograd emits each
# wrapper as Python *source*, exec's it (the chokepoint is _compile_and_exec_source
# in codegen.py), and the resulting function runs while closing over a
# globals dict supplied in-process -- i.e. a wrapper is (source text) +
# ({local_name: live_object}).
#
# The objects a wrapper closes over come in a few kinds:
#   - public runtime helpers the codegen'd source references (e.g. increment_version,
#     gen_alias_from_base, _unwrap_tensoralias, mark_dynamo_propagated_dynamic_indices,
#     the CUDARngStateHelper staticmethods) -- ordinary importable objects;
#   - the inner Inductor ``call`` that the chain ultimately invokes;
#   - sibling captured wrappers -- the next link of the runtime chain (an inner subclass /
#     functionalized-RNG wrapper, or an outer dedup / synthetic-base wrapper, whose body
#     calls the link it wraps -- the orchestration's outer closure for the innermost outer
#     wrapper), plus the orchestration's output-alias and mutation epilogue helpers, which
#     it closes over directly by reference;
#   - per-graph metadata baked at compile time (e.g. a ViewMetaSequence for alias
#     regen, tensor-subclass metadata) -- live objects with no import path.
#
# Concretely, the module emitted for ``x.view(-1)`` (an output aliasing its input) has
# one of each (heavily trimmed; the (1)-(4) tags match the kinds listed above):
#
#     from ...standalone_runtime import gen_alias_from_base    # (1) public helper
#     <inner Inductor kernels + ``def call(args): ...``>
#     _inner_call = call                                       # (2) the inner call
#     _vms_0 = ViewMetaSequence._from_parts(...)               # (4) metadata as source
#     def _alias_fn(orig_inputs, fw_outs):                     # (3) sibling wrapper
#         return [gen_alias_from_base(orig_inputs[0], fw_outs[0], False, _vms_0, ...)]
#     _replay_aliases_ = _alias_fn                             # (3) orchestration's ref
#     def _runtime_wrapper(_compiled_fn_, _first_ctx_, _on_before_call_, args):
#         all_outs = _compiled_fn_(args)                       # (2) inner call invoked
#         return _replay_aliases_(orig_inputs, all_outs)       # (3) sibling invoked
#     def call(flat_inputs):
#         return _runtime_wrapper(
#             _inner_call, contextlib.nullcontext, lambda: None, list(flat_inputs))
#
# We do NOT reimplement any of this. We CAPTURE AOTAutograd's exact codegen'd wrapper
# source together with the (pre-exec) globals dict each wrapper closed over: a
# thread-local sink in codegen.py records one GeneratedSource per wrapper.
# To trigger the capture we run AOTAutograd ourselves (under no_grad, the inference path)
# with a capture-only inner compiler: it grabs the dense inner graph and returns a
# placeholder callable, so AOTAutograd still codegen's the runtime-wrapper chain AROUND
# that placeholder -- which is what the sink records. Inductor does not run in that pass;
# it runs once afterward on the captured dense graph (via inductor.compile_to_python), and
# the composer swaps the placeholder for the inner inductor ``call`` by object identity.
#
# THE COMPOSITION PROBLEM. To turn a captured wrapper into a real top-level ``def`` in
# the standalone module we splice its source verbatim. But that source refers to each
# global by the LOCAL ALIAS AOTAutograd happened to choose (e.g. ``compiled_fn`` for
# the inner call), not by any importable name. So for each ``{name: obj}`` in the
# captured globals dict we emit a top-level binding ``name = <source for obj>`` (see
# _emit_inline) -- except when the resolved expression already IS that module-scope
# name (an import, or ``torch``), which needs no binding. The hard part is the right-
# hand side: given only a live object, produce source that reproduces it. That requires
# RECOGNIZING what the object is, the job of _resolve_global and the id-keyed
# structures it consults, in order:
#   - inner_call_id         -> the inner ``call`` becomes ``_inner_call``
#   - fn_id_to_name         -> a sibling wrapper's fn becomes that wrapper's own name
#   - _known_helper_table() -> an importable helper becomes (import, expr)
#   - anything else: reconstruct field-by-field as source (_emit_value), or raise.
# Recognition is by id() (object identity), not value-equality: for "is this the EXACT
# object the wrapper closed over," == is the wrong tool (functions don't compare by
# value, and an equal-but-different object would mis-resolve). Value-equality IS used,
# but later and for a different job -- _emit_value round-trip-checks reconstructed
# metadata (rebuilt == obj) before trusting it. Every hoisted name is _reserve'd: a
# collision with a sibling wrapper's name or an inner-module binding fails loudly
# rather than silently rebinding.
#
# WHY THIS IS SAFE ACROSS PROCESSES AND MACHINES. Every id() above is consulted ONLY
# here, during composition, in the process that just ran the compile -- where all the
# candidate objects (helpers, the inner call, sibling wrappers) are simultaneously
# alive, held by the GeneratedSource records and the captured globals dicts, so no
# address can be freed and reused mid-pass. Nothing the composer emits carries an id()
# value, a live object, or any thread-local capture state: it emits only import lines,
# ``name = expr`` bindings, and verbatim def source (the inner Inductor source is
# likewise spliced as text). Grep the GENERATED module and there is no ``id(`` to
# find. By the time the user holds the Python, all the process/thread-local state the
# composer leaned on is gone; what ships is imports + name bindings + verbatim code.
# Load it on another machine with the same torch version and the imports resolve by
# name and the bindings reconstruct -- identity was a compile-time recognition device,
# never a serialized artifact (live tensors / pickle / base64 blobs are rejected
# outright rather than embedded).
#
# So the one genuine cross-machine contract is not id() but that the *names* the
# artifact imports still resolve on load. Helpers are emitted as either ``import
# torch`` (the torch module and stable public paths, e.g.
# torch.autograd.graph.increment_version) or an import from the single small surface
# standalone_runtime.py (for the AOTAutograd-area internals -- plus CUDARngStateHelper,
# re-exported there for import-ordering -- whose locations are not themselves a stable
# contract). That file's IDENTITY CONTRACT -- re-exports must preserve object id --
# exists purely so the COMPOSER's id-lookup keeps matching; it is a compile-time
# requirement, and the runtime artifact has no id dependency of its own.
# ======================================================================


# Global objects the codegen'd wrappers close over that are reproducible as an
# import in the standalone module (rather than reconstructed field-by-field). Maps
# object id -> (import_statement, expression). Built lazily to avoid import cycles.
def _known_helper_table() -> dict[int, tuple[str, str]]:
    # Generated artifacts import runtime helpers from the single stable surface
    # ``standalone_runtime`` (not scattered AOTAutograd internals).
    import torch

    from . import standalone_runtime as rt

    _RT = "from torch._functorch._aot_autograd.standalone_runtime import"
    table: dict[int, tuple[str, str]] = {
        id(torch): ("import torch", "torch"),
        id(rt.normalize_as_list): (f"{_RT} normalize_as_list", "normalize_as_list"),
        id(rt.mark_dynamo_propagated_dynamic_indices): (
            f"{_RT} mark_dynamo_propagated_dynamic_indices",
            "mark_dynamo_propagated_dynamic_indices",
        ),
        id(torch.autograd.graph.increment_version): (
            "import torch",
            "torch.autograd.graph.increment_version",
        ),
        id(rt.gen_alias_from_base): (
            f"{_RT} gen_alias_from_base",
            "gen_alias_from_base",
        ),
        id(rt._unwrap_tensoralias): (
            f"{_RT} _unwrap_tensoralias",
            "_unwrap_tensoralias",
        ),
        id(rt.CUDARngStateHelper.get_torch_state_as_tuple): (
            f"{_RT} CUDARngStateHelper",
            "CUDARngStateHelper.get_torch_state_as_tuple",
        ),
        id(rt.CUDARngStateHelper.set_new_offset): (
            f"{_RT} CUDARngStateHelper",
            "CUDARngStateHelper.set_new_offset",
        ),
    }
    return table


_MODULE_HEADER = """\
# Generated by torch._functorch.aot_autograd.compile_to_python -- do not edit.
#
# Self-contained, executable module exposing ``call(flat_inputs) -> outputs`` for
# the post-AOTAutograd graph. The Inductor kernels JIT-compile from the inlined
# source on first call (no cache needed). The prelude/epilogue is AOTAutograd's own
# codegen'd runtime wrappers, not reimplemented: each (the orchestration and any chain
# wrappers) is spliced as a real top-level ``def`` with its closed-over globals (inner
# ``call``, sibling wrappers, public helpers, baked metadata reconstructed as source)
# hoisted to module-scope assignments -- so results match eager. The companion opaque
# cache is only an acceleration; this module never reads it.
"""


def _resolve_global(
    obj: object,
    helper_table: dict[int, tuple[str, str]],
    inner_call_id: int | None,
    fn_id_to_name: dict[int, str],
    imports: set[str],
    orch_closure_id: int | None = None,
    orch_entry_name: str | None = None,
) -> str:
    """Return a Python expression (valid in the generated module) that reproduces
    ``obj``, recording any needed import. Raises NotImplementedError if ``obj`` is
    neither the inner call, a sibling wrapper, a known helper, nor source-
    reconstructible (see ``_emit_value``)."""
    if inner_call_id is not None and id(obj) == inner_call_id:
        return "_inner_call"
    # An OUTER wrapper (dedup / synthetic base) closes over the orchestration's outer
    # closure as its inner. That closure is not a captured wrapper and has no import
    # path, so wire it to the single-arg orchestration entry adapter the composer emits.
    if orch_closure_id is not None and id(obj) == orch_closure_id:
        if orch_entry_name is None:
            raise AssertionError("expected orch_entry_name to be not None")
        return orch_entry_name
    if id(obj) in fn_id_to_name:
        return fn_id_to_name[id(obj)]
    if id(obj) in helper_table:
        import_stmt, expr = helper_table[id(obj)]
        if import_stmt:
            imports.add(import_stmt)
        return expr
    # Not a wired reference (inner call / sibling wrapper / helper): emit ``obj`` as
    # plain reconstruction source. Raises if it is not source-expressible.
    return emit_value(obj, imports)


def _module_level_names(tree: ast.Module) -> set[str]:
    """Names bound at module scope by a parsed module. Used to seed ``_reserve`` so an
    inlined wrapper's def name or hoisted global (chain wrapper or orchestration) cannot
    silently shadow a top-level name the inner Inductor module already binds."""
    names: set[str] = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            # Walk each target so tuple/list/starred unpacking (``a, b = ...`` /
            # ``first, *rest = ...``) is covered, not just bare-name targets.
            for t in n.targets:
                names.update(x.id for x in ast.walk(t) if isinstance(x, ast.Name))
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            names.add(n.target.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.Delete):
            # Inductor's inner module does ``async_compile = AsyncCompile()`` then
            # ``del async_compile`` at module scope; a del'd name does not survive, so it
            # must not be reserved (else a hoisted wrapper global of the same name would
            # trip a spurious collision). Body order is assign-then-del, so removing here
            # leaves only names that actually persist.
            for t in n.targets:
                names.difference_update(
                    x.id for x in ast.walk(t) if isinstance(x, ast.Name)
                )
    return names


def _compose_standalone_module(
    inner_python: str, captured: list[GeneratedSource], inner_call_obj: Any
) -> str:
    """Compose the inner Inductor ``call`` with AOTAutograd's captured runtime
    wrappers into one standalone module exposing ``call(flat_inputs) -> outputs``.

    Every wrapper (chain wrappers and the orchestration) is spliced as a real top-level
    ``def`` with its closed-over globals hoisted to module-scope assignments (resolved
    here). They are chained by name in the SAME nesting the runtime builds: the
    orchestration is invoked with the INNER InductorWrapper chain head (subclass / RNG)
    as its inner, and any OUTER ``CompilerWrapper`` (dedup / synthetic base, applied
    AROUND the orchestration in ``graph_compile._aot_stage2c_make_inference_function``)
    wraps a single-arg adapter over the orchestration.

    ``inner_call_obj`` is the placeholder the capture pass returned for the inner call;
    its identity is authoritative (see the note at the inner-call site below).
    """
    # The capture sink is duration-scoped over the inner inductor compile, with no
    # originating-graph id at install time, so a re-entrant on-thread AOTAutograd /
    # inductor lowering during that window that codegen's wrappers into THIS sink would
    # append ITS wrappers here too. (A nested compile_to_python installs its OWN sink via
    # capture_generated_sources, so its wrappers go there, not here -- it is not this
    # case.) Each captured wrapper is tagged at append time with its TracingContext
    # identity (origin_id), which separates such a foreign lowering when it ran under a
    # DISTINCT TracingContext. A same-context re-entrant lowering reuses the ambient
    # TracingContext via try_get() and so shares this origin_id; that case is instead
    # caught by the orchestration count/wiring guards below (a second orchestration trips
    # the len() != 1 check). Filter to the target graph's origin before composing. The
    # target is the origin of the LAST captured orchestration wrapper: a foreign lowering
    # appends its orchestration before the outer one finishes, so the final orchestration
    # is always the outer/target one.
    orchestrations = [
        g for g in captured if g.artifact_name == "runtime_wrapper_orchestration"
    ]
    if orchestrations:
        # target_origin is None only without an ambient TracingContext, which the real
        # precompile path never hits (capture always runs under one); in that defensive
        # case the filter keeps every None-origin wrapper and the count/wiring guards
        # below remain the backstop.
        target_origin = orchestrations[-1].origin_id
        captured = [g for g in captured if g.origin_id == target_origin]

    # Backward wrappers are out of scope for forward lowering; reject them up front.
    # Every other wrapper that can appear in a composable (cacheable) forward graph is
    # codegen'd as source and captured here. The one non-codegen'd wrapper,
    # FakifiedOutWrapper, only activates under fakify_first_call, which makes the graph
    # non-cacheable -- so such a graph is rejected before it ever reaches composition.
    unsupported = [g.artifact_name for g in captured if "backward" in g.artifact_name]
    if unsupported:
        raise NotImplementedError(
            "aot_autograd.compile_to_python cannot yet compose these runtime "
            f"wrappers into standalone source: {sorted(set(unsupported))}."
        )

    orchestration = [
        g for g in captured if g.artifact_name == "runtime_wrapper_orchestration"
    ]
    if len(orchestration) != 1:
        raise NotImplementedError(
            "aot_autograd.compile_to_python expected exactly one forward "
            f"orchestration wrapper, captured {len(orchestration)}."
        )
    orch = orchestration[0]
    non_orch = [g for g in captured if g is not orch]

    # The generated ``call`` invokes the orchestration POSITIONALLY by its own name (see
    # the bottom of this function): _runtime_wrapper(chain_head, contextlib.nullcontext,
    # lambda: None, flat_inputs). That mapping is hardcoded to the codegen'd signature in
    # runtime_wrappers.py (``def _runtime_wrapper(_compiled_fn_, _first_ctx_,
    # _on_before_call_, args)``). Verify the captured signature still matches so a future
    # rename/reorder fails loudly here instead of silently passing wrong arguments.
    expected_orch_params = ["_compiled_fn_", "_first_ctx_", "_on_before_call_", "args"]
    orch_def = next(
        (
            n
            for n in ast.walk(ast.parse(orch.source))
            if isinstance(n, ast.FunctionDef) and n.name == orch.fn_name
        ),
        None,
    )
    args_node = orch_def.args if orch_def is not None else None
    if args_node is None:
        orch_params = None
    else:
        # Compare the FULL signature, not just positional params: the standalone call is
        # purely positional, so a keyword-only / *args / **kwargs param (e.g. an added
        # kw-only-with-default) would be silently dropped. Surface any such param so it
        # trips this guard rather than passing.
        orch_params = [a.arg for a in (*args_node.posonlyargs, *args_node.args)]
        orch_params += [a.arg for a in args_node.kwonlyargs]
        if args_node.vararg is not None:
            orch_params.append("*" + args_node.vararg.arg)
        if args_node.kwarg is not None:
            orch_params.append("**" + args_node.kwarg.arg)
    if orch_params != expected_orch_params:
        raise NotImplementedError(
            "aot_autograd.compile_to_python: the orchestration wrapper signature "
            f"changed (expected {expected_orch_params}, got {orch_params}); the "
            "standalone module invokes it positionally and must be updated to match."
        )

    helper_table = _known_helper_table()
    # Every wrapper is inlined (below) as a real def at module scope under its OWN codegen'd
    # name, so references resolve to that name. Note these names are NOT distinct in general
    # -- the subclass, dedup, and debug-assert chain wrappers all codegen ``inner_fn``; what
    # holds today is that at most one chain wrapper appears per composable forward graph (see
    # the test note on multi-link chains), so the names don't actually clash. ``_reserve``
    # fails loudly if that ever stops holding (the old ``_wrapper_{i}`` scheme could carry
    # multiple same-named wrappers in private exec namespaces; inlining deliberately cannot).
    fn_id_to_name = {id(g.fn): g.fn_name for g in non_orch}

    # A chain wrapper references the inner it wraps via one of these globals
    # (subclass/dedup use ``compiled_fn``; the functionalized-RNG wrapper uses
    # ``_compiled_fn_``). The orchestration takes its inner as a call-time arg, not a
    # global, so it is never a chain wrapper. MAINTAINERS: if AOTAutograd adds a
    # forward chain wrapper that names its inner via a new global, add that name here,
    # otherwise inner-call/chain-head detection silently bypasses it.
    _INNER_NAMES = ("compiled_fn", "_compiled_fn_")

    def _inner_ref(g: GeneratedSource) -> Any:
        for nm in _INNER_NAMES:
            if nm in g.globals_dict:
                return g.globals_dict[nm]
        return None

    # The inner Inductor call is AUTHORITATIVE: it is the placeholder object the capture
    # pass returned (threaded in as ``inner_call_obj``), NOT inferred from capture order.
    # This is what lets the composer tell the inner call apart from the orchestration's
    # own outer closure -- both surface as some wrapper's inner-ref yet neither is a
    # captured wrapper fn -- which is precisely how INNER wrappers (subclass / RNG,
    # wrapping the inner call, composed INSIDE the orchestration) are distinguished from
    # OUTER wrappers (dedup / synthetic base, ``CompilerWrapper``s applied AROUND the
    # orchestration in graph_compile._aot_stage2c_make_inference_function).
    inner_call_id: int = id(inner_call_obj)

    # Name of the single-arg adapter emitted over the orchestration; the innermost outer
    # wrapper closes over the orchestration's outer closure and is wired to this name.
    _ORCH_ENTRY = "_orchestration_entry"

    # The orchestration's outer closure is the object outer wrappers wrap. The composer
    # never captures it directly (the captured orchestration ``fn`` is the inner
    # ``_codegen_runtime_wrapper``), so recognize it structurally: an inner-ref that is
    # neither the inner call nor any captured wrapper's fn can only be that closure.
    orch_closure_ids = {
        id(_inner_ref(g))
        for g in non_orch
        if _inner_ref(g) is not None
        and id(_inner_ref(g)) != inner_call_id
        and id(_inner_ref(g)) not in fn_id_to_name
    }
    if len(orch_closure_ids) > 1:
        raise NotImplementedError(
            "aot_autograd.compile_to_python: captured multiple wrappers whose inner "
            "reference is neither the inner call nor a captured wrapper; cannot tell "
            "which wraps the orchestration."
        )
    orch_closure_id: int | None = next(iter(orch_closure_ids), None)

    # Walk the OUTER chain outward from the orchestration closure: the innermost outer
    # wrapper wraps the closure, the next wraps that wrapper's fn, and so on. Everything
    # else is inner-side -- the subclass / RNG chain wrappers plus the alias / mutation
    # epilogue helpers the orchestration closes over (which reference no inner at all).
    outer_wrappers: list[GeneratedSource] = []
    if orch_closure_id is not None:
        target_id: int | None = orch_closure_id
        while target_id is not None:
            nxt = next(
                (
                    g
                    for g in non_orch
                    if g not in outer_wrappers
                    and _inner_ref(g) is not None
                    and id(_inner_ref(g)) == target_id
                ),
                None,
            )
            if nxt is None:
                break
            outer_wrappers.append(nxt)
            target_id = id(nxt.fn)
    outer_ids = {id(g) for g in outer_wrappers}
    inner_side = [g for g in non_orch if id(g) not in outer_ids]

    # Chain head passed to the orchestration: the outermost INNER InductorWrapper (last
    # inner-side wrapper that wraps via an inner reference), else the inner call. Computed
    # up front (a pure capture-order check) so the order-inversion guard below fires
    # before the later name-uniqueness guard -- a mis-ordered chain is the more specific
    # diagnosis.
    chain_head = "_inner_call"
    chain_head_g: GeneratedSource | None = None
    for g in inner_side:
        if _inner_ref(g) is not None:
            chain_head = fn_id_to_name[id(g.fn)]
            chain_head_g = g

    # "Last with an inner-ref == outermost" holds only when INNER capture order is
    # innermost-to-outermost (it is today: subclass before functionalized-RNG). Back that
    # assumption with a guard: the true outermost inner wrapper is the one NO other inner
    # wrapper wraps, i.e. whose fn is not referenced as another inner wrapper's inner. If
    # the chosen head is itself wrapped, capture order inverted and the chain would be
    # mis-ordered -- reject rather than silently emit a wrong chain (the wiring guard
    # below would not catch this, since every wrapper is still referenced somewhere).
    referenced_inner_ids = {
        id(_inner_ref(g)) for g in inner_side if _inner_ref(g) is not None
    }
    if chain_head_g is not None and id(chain_head_g.fn) in referenced_inner_ids:
        raise NotImplementedError(
            "aot_autograd.compile_to_python: the selected chain head is itself wrapped "
            "by another captured wrapper, so capture order is not innermost-to-outermost "
            "as assumed; refusing to emit a mis-ordered runtime-wrapper chain."
        )

    imports: set[str] = set()

    # Parse the inner module once: to verify it binds a module-level ``call`` (the
    # inner-call contract, checked below) and to collect its top-level names so no inlined
    # wrapper's def name or hoisted global can silently shadow one.
    inner_tree = ast.parse(inner_python)
    inner_module_names = _module_level_names(inner_tree)

    # Every runtime wrapper is inlined: its def is spliced at module scope and its
    # closed-over globals hoisted to top-level assignments (no exec / private namespace).
    # So every emitted top-level name -- each wrapper's def name and each hoisted global --
    # must be unique and must not shadow a name the inner Inductor module binds. This holds
    # in practice because at most one chain wrapper appears per composable forward graph, so
    # its def name (``inner_fn`` is shared across subclass/dedup/debug-assert wrappers) and
    # the inner-ref global ``compiled_fn`` each occur once, and metadata globals are
    # per-wrapper suffixed. ``_reserve`` guards it: a collision fails loudly (rename/namespace
    # needed) rather than silently rebinding a name.
    emitted_names = set(inner_module_names) | {
        "call",
        "_inner_call",
        "_rebuild",
        "contextlib",
        _ORCH_ENTRY,
    }

    def _reserve(name: str) -> None:
        if name in emitted_names:
            raise NotImplementedError(
                "aot_autograd.compile_to_python: generated top-level name "
                f"{name!r} collides with another top-level name in the composed module; "
                "inlining the runtime wrappers would shadow a binding."
            )
        emitted_names.add(name)

    # Reserve every wrapper def name up front (before hoists) so a hoisted global cannot
    # shadow a def and two wrappers cannot share a name.
    for _g in (*non_orch, orch):
        _reserve(_g.fn_name)

    def _resolve_globals(globals_dict: dict[str, object]) -> list[tuple[str, str]]:
        # Resolve each global a wrapper closes over to a standalone source expression.
        # ``globals_dict`` is the pre-exec snapshot from codegen.py, so the
        # interpreter ``__builtins__`` is absent; the skip is kept defensively in case a
        # future caller hands us a post-exec live dict.
        out: list[tuple[str, str]] = []
        for gname, gobj in globals_dict.items():
            if gname == "__builtins__":
                continue
            expr = _resolve_global(
                gobj,
                helper_table,
                inner_call_id,
                fn_id_to_name,
                imports,
                orch_closure_id,
                _ORCH_ENTRY,
            )
            out.append((gname, expr))
        return out

    def _emit_inline(source: str, globals_dict: dict[str, object]) -> str:
        # Splice the wrapper's def verbatim at module scope, hoisting each closed-over global
        # to a top-level assignment (skipping a name already module-available -- an imported
        # helper or ``torch`` -- detected as gname == its resolved expr). No exec / private
        # namespace: the def reads as ordinary code and is referenced by its own name. Each
        # hoisted name is ``_reserve``'d so a collision fails loudly rather than rebinding.
        hoists: list[str] = []
        for gname, expr in _resolve_globals(globals_dict):
            if gname == expr:
                continue  # already at module scope (an import / ``torch``)
            _reserve(gname)
            hoists.append(f"{gname} = {expr}")
        return "\n".join(hoists + [source, ""])

    # Inner-side wrappers first (subclass / RNG chain wrappers innermost-to-outermost,
    # plus the epilogue helpers), then the orchestration, then the outer wrappers -- all
    # spliced as real defs. An inner wrapper's hoisted inner-ref (``compiled_fn``)
    # references ``_inner_call`` / a sibling emitted earlier; an outer wrapper's inner-ref
    # references the orchestration entry adapter / a sibling emitted earlier -- so order
    # is satisfied.
    inner_blocks = [_emit_inline(g.source, g.globals_dict) for g in inner_side]
    orch_block = _emit_inline(orch.source, orch.globals_dict)
    outer_blocks = [_emit_inline(g.source, g.globals_dict) for g in outer_wrappers]

    # Imports (helper table + whatever emit_value added) are emitted BEFORE the inner
    # module and the wrapper blocks, so a later top-level binding of the same name -- an
    # inner-module binding or a hoisted wrapper global -- would shadow the import. And
    # _emit_inline skips hoisting a global whose resolved expr already equals its own name
    # (an imported helper referenced as ``gname == expr``), so such a wrapper would then
    # silently bind to the shadowing object instead of the helper. ``from X import Y``
    # names bind a specific (non-module) object and are the ones at risk; plain ``import
    # mod`` names resolve to the same singleton module no matter who imports them, so a
    # duplicate binding is benign and is left unchecked (else ``import torch`` -- which the
    # inner module also emits -- would trip a spurious collision). ``_reserve`` fails loudly
    # if an inner-module name or a hoisted global collides with one of these import names.
    for stmt in sorted(imports):
        node = ast.parse(stmt).body[0]
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                _reserve(alias.asname or alias.name)

    # The final ``call`` invokes the outermost outer wrapper (else the orchestration
    # directly). Build that line now so the wiring guard's corpus includes it -- the
    # outermost outer wrapper is referenced ONLY there.
    if outer_wrappers:
        outermost_name = fn_id_to_name[id(outer_wrappers[-1].fn)]
        final_invoke = f"    return {outermost_name}(list(flat_inputs))"
    else:
        final_invoke = (
            f"    return {orch.fn_name}(\n"
            f"        {chain_head}, contextlib.nullcontext, lambda: None, "
            "list(flat_inputs)\n    )"
        )

    # The single-arg adapter over the orchestration (emitted only when outer wrappers
    # wrap it). The outer wrappers call their inner as ``fn(args)``; this adapts the
    # orchestration's positional (_compiled_fn_, _first_ctx_, _on_before_call_, args)
    # signature to that. When there are no outer wrappers the orchestration is invoked
    # directly in ``call`` (see final_invoke).
    orch_invoke_comment = [
        "    # The 2nd/3rd positional args INTENTIONALLY substitute contextlib.nullcontext",
        "    # for the runtime's first-invocation context (_FirstInvocationContext) and a",
        "    # no-op for the profiler-prologue exit. This drops two cold-start diagnostics:",
        "    # the first-call custom-op aliasing analysis (_AnalyzeCustomOpInputOutputMode,",
        "    # active when check_custom_op_aliasing is set, which can even RAISE under",
        "    # error_on_custom_op_aliasing) and the profiler prologue. Neither affects",
        "    # numerics, so this is not a bug -- the standalone artifact deliberately omits",
        "    # them. (See the positional-mapping note in _compose_standalone_module.)",
    ]
    entry_block: list[str] = []
    if outer_wrappers:
        entry_block = [
            f"def {_ORCH_ENTRY}(args):",
            "    # Single-arg adapter so the CompilerWrappers applied AROUND the",
            "    # orchestration (dedup / synthetic base) can invoke it as ``fn(args)``.",
            *orch_invoke_comment,
            f"    return {orch.fn_name}(",
            f"        {chain_head}, contextlib.nullcontext, lambda: None, args",
            "    )",
            "",
        ]

    # _INNER_NAMES detection is a hardcoded allowlist (see above). If AOTAutograd adds
    # a forward wrapper that names its inner via an unrecognized global, that wrapper
    # is captured but may never be wired into the module -- silently composing a
    # structurally-wrong result. Enforce that every captured non-orch wrapper is
    # actually referenced somewhere: as the inner chain head, in the final ``call``
    # (the outermost outer wrapper), or by name in another block (another wrapper's
    # globals, the orchestration's epilogue helpers -- e.g. ``_alias_fn`` /
    # ``_apply_mutations`` -- or the entry adapter). A wrapper whose name appears in no
    # other block went unwired, so reject rather than emit a wrong module.
    block_of = {id(g): b for g, b in zip(inner_side, inner_blocks)}
    block_of.update({id(g): b for g, b in zip(outer_wrappers, outer_blocks)})
    other_text = "\n".join(
        inner_blocks + [orch_block] + entry_block + outer_blocks + [final_invoke]
    )
    for g in non_orch:
        name = fn_id_to_name[id(g.fn)]
        own = block_of[id(g)]
        elsewhere = other_text.replace(own, "", 1)
        # Whole-token match: ``name`` is a wrapper def name (e.g. ``inner_fn``); a raw
        # substring test would treat ``inner_fn`` as wired whenever a longer token like
        # ``inner_fn2`` is referenced, silently defeating this guard.
        wired = re.search(r"\b" + re.escape(name) + r"\b", elsewhere) is not None
        if name != chain_head and not wired:
            raise NotImplementedError(
                "aot_autograd.compile_to_python could not wire captured runtime "
                f"wrapper {g.fn_name!r} into the module (an inner-call global may be "
                "unrecognized; see _INNER_NAMES)."
            )

    # The module splices ``_inner_call = call`` below, relying on inner_python binding a
    # module-level ``call`` entry point. Inductor emits this in one of two forms: the
    # flat path defines ``def call(args):`` (FunctionDef) while the graph_partition Runner
    # path binds ``call = runner.call`` (Assign with a Name target). Verify one is present
    # so a future inductor codegen drift fails loudly here -- like the orchestration /
    # chain / wiring guards above -- instead of surfacing as a bare NameError at exec of
    # the generated module. (``inner_tree`` was parsed once up front.)
    binds_call = any(
        (isinstance(n, ast.FunctionDef) and n.name == "call")
        or (
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "call" for t in n.targets)
        )
        for n in inner_tree.body
    )
    if not binds_call:
        raise NotImplementedError(
            "compile_to_python: inner Inductor module does not bind a module-level "
            "'call' entry point (the inner-call contract); the standalone module "
            "splices ``_inner_call = call`` and must be updated to match."
        )

    # Only emit the _rebuild helper if a baked value actually reconstructs through it.
    needs_rebuild = any(
        "_rebuild(" in b for b in (*inner_blocks, orch_block, *outer_blocks)
    )

    # ``call`` invokes the orchestration directly when nothing wraps it; when outer
    # wrappers do, its body just calls the outermost of them (which drives the entry
    # adapter, then the orchestration) -- so the orchestration-substitution comment lives
    # on whichever site actually invokes the orchestration.
    if outer_wrappers:
        call_body = [
            "    # Invokes the outer CompilerWrapper chain (dedup / synthetic base), which",
            f"    # wraps {_ORCH_ENTRY} (the orchestration adapter).",
            final_invoke,
        ]
    else:
        call_body = [
            "    # AOTAutograd orchestration: disables grad, invokes the inner chain,",
            "    # bumps mutated-input versions, applies the output epilogue.",
            "    #",
            *orch_invoke_comment,
            final_invoke,
        ]

    parts = [
        _MODULE_HEADER,
        "import contextlib",
        *sorted(imports),
        "",
        "",
        *(_REBUILD_HELPER if needs_rebuild else []),
        "# " + "=" * 70,
        "# Inner Inductor output code (kernels + ``call``)",
        "# " + "=" * 70,
        inner_python,
        "_inner_call = call",
        "",
        "# " + "=" * 70,
        "# AOTAutograd runtime wrappers (codegen'd): each inlined as a real def with its",
        "# closed-over globals hoisted to module scope -- inner chain wrappers first, then",
        "# the orchestration, then any outer wrappers (dedup / synthetic base) that wrap it",
        "# " + "=" * 70,
        *inner_blocks,
        orch_block,
        *entry_block,
        *outer_blocks,
        "",
        "def call(flat_inputs):  # noqa: F811",
        *call_body,
        "",
    ]
    return "\n".join(parts)


def _find_effectful_op(gm: GraphModule, get_effect: Any) -> Any:
    """Return the first effectful op target reachable from ``gm``, or None.

    The target may be an ``OpOverload`` (e.g. ``aten::_print``) or a
    ``HigherOrderOperator`` -- ``call_torchbind`` / ``hop_print`` /
    ``invoke_leaf_function`` are registered ``_EffectType.ORDERED`` HOPs, not
    OpOverloads -- so both are checked (``get_effect`` returns None for a
    non-effectful HOP like ``cond``).

    Walks the graph and descends into any child GraphModule a node references -- a
    HOP (cond/while_loop/scan) holds its body as a get_attr'd submodule or passes it
    directly as a node arg -- so an effect nested inside a HOP subgraph is caught, not
    just effects at the top level."""
    import torch
    from torch.fx import GraphModule as _GraphModule

    seen: set[int] = set()

    def _scan(g: _GraphModule) -> Any:
        if id(g) in seen:
            return None
        seen.add(id(g))
        for node in g.graph.nodes:
            if (
                node.op == "call_function"
                and isinstance(
                    node.target,
                    (torch._ops.OpOverload, torch._ops.HigherOrderOperator),
                )
                and get_effect(node.target) is not None
            ):
                return node.target
            for sub in _iter_subgraphs(g, node):
                found = _scan(sub)
                if found is not None:
                    return found
        return None

    def _walk_values(value: Any) -> Iterator[_GraphModule]:
        # A GraphModule can appear as a direct node arg OR nested one level (or more)
        # inside a list/tuple/dict arg -- some HOPs pass their branch/body callables
        # inside a container -- so descend into containers before the isinstance check,
        # otherwise the recursive effect scan would never enter that nested subgraph.
        if isinstance(value, _GraphModule):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from _walk_values(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from _walk_values(item)

    def _iter_subgraphs(g: _GraphModule, node: Any) -> Iterator[_GraphModule]:
        # A child graph reaches a node either as an attribute fetched by get_attr or
        # as a (possibly container-nested) argument (the form HOPs use for their
        # branch/body callables). make_fx emits FLAT get_attr targets (e.g.
        # true_graph_0) since _scan recurses per-GraphModule, so a plain getattr
        # suffices (no dotted walk needed here).
        if node.op == "get_attr":
            attr = getattr(g, node.target, None)
            if isinstance(attr, _GraphModule):
                yield attr
        for arg in (*node.args, *node.kwargs.values()):
            yield from _walk_values(arg)

    return _scan(gm)


def _graph_has_dynamic_shapes(gm: GraphModule) -> bool:
    """True if any placeholder is itself a SymInt or carries symbolic (SymInt) size,
    stride, or storage-offset metadata -- i.e. the graph was traced with dynamic dims.
    Drives the shapes mode for the capture pass: dynamic graphs stay dynamic, static
    graphs specialize so the composer can bake their (static) view metadata. Strides and
    storage offset are checked too, not just sizes: a graph dynamic solely via symbolic
    strides has static sizes, and treating it as static would silently specialize the
    artifact to the example strides. (Unbacked symints appearing only in intermediates,
    not on any placeholder, are still missed here, but such a graph fails loudly
    downstream when emit_value rejects the still-symbolic metadata.)"""
    import torch

    def _is_symbolic(v: Any) -> bool:
        return isinstance(v, torch.SymInt)

    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        val = node.meta.get("val")
        if _is_symbolic(val):
            return True
        if isinstance(val, torch.Tensor) and (
            any(_is_symbolic(s) for s in val.shape)
            or any(_is_symbolic(s) for s in val.stride())
            or _is_symbolic(val.storage_offset())
        ):
            return True
    return False


def compile_to_python(
    gm: GraphModule,
    example_inputs: Sequence[Any],
    *,
    options: dict[str, Any] | None = None,
) -> tuple[str, bytes | None]:
    """Compile ``gm`` to ``(python_code, cache)``; see the module docstring.

    THREADING: serialized by a process-global lock (``_COMPILE_LOCK``). The wrapper-source
    capture is thread-local, but the AOTAutograd pass and the inner inductor compile both
    swap process-global cache state (``CacheArtifactManager.with_fresh_cache()``); a
    concurrent compile on another thread would corrupt the captured wrappers or cache
    artifacts, so concurrent calls (including via ``torch.compiler.precompile``) are
    serialized rather than run in parallel.
    """
    import copy

    import torch
    from torch._higher_order_ops.effects import _get_effect
    from torch._inductor import compile_to_python as _inductor_compile_to_python
    from torch._inductor.compile_fx import compile_fx
    from torch._inductor.standalone_compile import (
        _resolve_ignore_shape_env,
        _standalone_context,
    )
    from torch.fx.graph_module import _share_torchbind_and_process_group_on_deepcopy

    from .utils import make_boxed_func

    # Validate up front: this layer dereferences ``gm.graph`` (the effectful-op scan
    # below) before reaching inductor's own type-check, so a non-GraphModule would
    # otherwise surface as an opaque AttributeError instead of this clear contract error.
    if not isinstance(gm, torch.fx.GraphModule):
        raise TypeError(
            "aot_autograd.compile_to_python expects a post-AOTAutograd "
            f"torch.fx.GraphModule, got {type(gm)}."
        )

    # Effectful ops thread effect tokens through a calling convention the standalone
    # composition does not reproduce (and their with_effects HOP is non-cacheable);
    # reject them up front with a concrete reason. Not supported yet. (Detected here
    # too, not only in torch.compiler.precompile's capture-time guard, so direct callers of
    # this lowering get the same clear failure rather than a silently-wrong artifact.)
    # Scan recursively: a HOP (cond/while_loop/scan) carries its body as a child
    # GraphModule referenced by a get_attr node (or passed directly as a node arg), so
    # effects nested in such a subgraph would be missed by a top-level-only scan.
    effectful = _find_effectful_op(gm, _get_effect)
    if effectful is not None:
        raise NotImplementedError(
            "aot_autograd.compile_to_python cannot lower this graph to standalone "
            f"source: it contains an effectful op ({effectful}), which is not "
            "supported yet."
        )

    with _COMPILE_LOCK:
        # Run AOTAutograd ONCE to do two things: (1) produce the dense, decomposed,
        # functionalized inner graph, and (2) codegen its runtime-wrapper chain, which the
        # thread-local ``capture_generated_sources`` sink records. A capture-only inner
        # compiler grabs the dense graph and returns a placeholder boxed callable, so
        # AOTAutograd still builds and codegen's the wrappers AROUND it -- that codegen is
        # what we capture. Inductor is NOT run in this pass; it runs exactly once below, on
        # the captured dense graph, via the ``_inductor_compile_to_python`` call (which
        # drives inductor's ``compile_fx_inner`` directly, not a re-entry into AOTAutograd).
        # The composer swaps the placeholder (the wrappers' inner reference) for the
        # inner inductor ``call`` by object identity, so the placeholder is only a
        # compile-time token and never runs.
        captured: list[GeneratedSource] = []
        dense: dict[str, Any] = {}

        def _capture_inner_compile(dense_gm, dense_inputs, **kwargs):
            if "gm" in dense:
                raise NotImplementedError(
                    "aot_autograd.compile_to_python does not support a graph whose "
                    "AOTAutograd lowering emits more than one inner forward graph."
                )
            dense["gm"] = dense_gm
            # Retain the placeholder's IDENTITY: it is the authoritative inner call the
            # runtime-wrapper chain closes over. The composer needs it to tell the inner
            # call apart from the orchestration's own outer closure (both surface as a
            # wrapper's inner-ref yet neither is a captured wrapper fn), which is what
            # separates INNER wrappers from the OUTER dedup / synthetic-base wrappers.
            placeholder = make_boxed_func(dense_gm.forward)
            dense["placeholder"] = placeholder
            return placeholder

        # Drive inductor's own ``compile_fx`` (i.e. its exact AOTAutograd invocation --
        # decomposition table + aot config) but swap in the capture inner compiler so the
        # dense graph is intercepted before codegen and no inductor compile happens. Using
        # compile_fx (rather than calling aot_autograd directly) guarantees the dense graph
        # matches what the step-2 inductor compile below expects. Pick the shapes mode from
        # the graph (there is no dynamic_shapes knob): a symbolically-traced graph uses
        # ``"from_graph"`` to stay dynamic, a static one ``"from_example_inputs"`` to
        # specialize -- matching what the composer can bake (symbolic view metadata is
        # rejected downstream). no_grad pins the inference path (one forward module).
        # Deepcopy first so compile_fx cannot mutate the caller's gm (torchbind
        # ProcessGroups smuggled through as shared references). Note: the raw-collective /
        # torchbind rewrites are inductor-lowering prereqs and belong to the step-2 inductor
        # compile, which applies them to the dense graph -- not duplicated here.
        shapes_mode = (
            "from_graph" if _graph_has_dynamic_shapes(gm) else "from_example_inputs"
        )
        with (
            torch.no_grad(),
            _standalone_context(gm, shapes_mode, aot=False),
            capture_generated_sources(captured),
        ):
            with _share_torchbind_and_process_group_on_deepcopy():
                gm_owned = copy.deepcopy(gm)
            compile_fx(
                gm_owned,
