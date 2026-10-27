        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "after removing aliasing:\n%s", fw_gm.print_readable(print_output=False)
            )

    def call_forward(self):
        fw_outputs_and_intermediates: tuple[Any] = scan_op(
            self.hop_partitioned_graph.fw_gm, self.init, self.xs, self.additional_inputs
        )  # type: ignore[return-type]
        fw_outs = fw_outputs_and_intermediates[
            : self.hop_partitioned_graph.n_fw_outputs
        ]
        saved_intermediates = fw_outputs_and_intermediates[
            self.hop_partitioned_graph.n_fw_outputs :
        ]
        if len(self.saved_intermediates) != 0:
            raise AssertionError(
                "saved_intermediates should be empty before call_forward"
            )
        self.saved_intermediates.extend(saved_intermediates)
        return tuple(fw_outs)

    def call_backward(self, *grad_fw_outputs):
        """
        Recall that fw_outputs = (*carry, *ys), bw_gm takes in (*fw_intermediates, *grad_carry, *grad_ys)
        and returns (*grad_init, *grad_xs, *grad_additional_inputs)
        The backward is a reversed scan that can be constructed as follows:

          grad_additional_inputs = torch.zeros_like(additional_inputs)
          bw_init = (grad_carry, grad_additional_inputs)
          bw_xs = (fw_intermediates, grad_ys)
          grad_init, grad_additional_inputs, grad_xs = scan(
            combine_fn,
            bw_init,
            bw_xs,
            reverse = True
          )
          , where combine_fn is defined as follows:

           def combine_fn(bw_init, bw_xs):
             grad_carry, grad_additional_inputs = bw_init
             fw_intermediates, grad_y = bw_xs
             nxt_grad_carry, grad_x, nxt_grad_additional_inputs = bw_gm(*fw_intermediates, *grad_carry, *grad_y)
             return (nxt_grad_carry, grad_additional_inputs + nxt_grad_additional_inputs), grad_x

          Note that grad_additional_inputs is accumulated with add, grad_carry is carried over to next iteration and
          grad_x is the ys output, which will be stacked together after the loop and will have the same shape as xs.
        """
        fw_policy = self.forward_intermediates_handling_policies
        saved_intermediates = self.saved_intermediates
        saved_fw_xs = self.saved_fw_xs
        saved_fw_additional_inputs = self.saved_fw_additional_inputs

        n_carry = len(self.init)

        grad_carry, grad_ys = grad_fw_outputs[:n_carry], grad_fw_outputs[n_carry:]
        additional_inputs_tensor_masks = [
            bool(isinstance(t, torch.Tensor)) for t in self.additional_inputs
        ]
        grad_additional_inputs = [
            torch.zeros_like(t)
            for t in filter_with_masks(
                self.additional_inputs, additional_inputs_tensor_masks
            )
        ]

        bw_init = [grad_carry, grad_additional_inputs]
        bw_xs = [
            grad_ys,
            saved_fw_xs,
            saved_intermediates,
        ]
        bw_additional_inputs = saved_fw_additional_inputs

        _, flat_spec = pytree.tree_flatten((bw_init, bw_xs, bw_additional_inputs))

        grad_spec = None

        def bw_single_step_wrapper(*args):
            bw_init, bw_xs, bw_additional_inputs = pytree.tree_unflatten(
                args, flat_spec
            )
            grad_carry, grad_additional_inputs = bw_init
            grad_y, saved_fw_xs, saved_intermediates = bw_xs
            saved_fw_additional_inputs = bw_additional_inputs

            fw_intermediates = []
            xs_it = iter(saved_fw_xs)
            carry_it = iter(saved_intermediates)
            addi_it = iter(saved_fw_additional_inputs)
            for policy in fw_policy:
                if policy in (
                    ScanForwardIntermediatesHandlingPolicy.CLONE,
                    ScanForwardIntermediatesHandlingPolicy.KEEP,
                ):
                    fw_intermediates.append(next(carry_it))
                elif policy == ScanForwardIntermediatesHandlingPolicy.REMOVE_XS:
                    fw_intermediates.append(next(xs_it))
                elif (
                    policy
                    == ScanForwardIntermediatesHandlingPolicy.REMOVE_ADDITIONAL_INPUTS
                ):
                    fw_intermediates.append(next(addi_it))
                else:
                    raise RuntimeError(f"Unknown policy: {policy}")

            grad_fw_outputs = (*grad_carry, *grad_y)

            flat_out = self.hop_partitioned_graph.bw_gm(
                *fw_intermediates,
                *grad_fw_outputs,
            )

            next_grad_carry, grad_xs, grad_addi = split_into_chunks(
                flat_out,  # type: ignore[arg-type]
                [len(self.init), len(self.xs), len(self.additional_inputs)],
            )

            nonlocal grad_spec
            flat_grads, grad_spec = pytree.tree_flatten(
                (
                    next_grad_carry,
                    [
                        prev + cur
                        for prev, cur in zip(
                            grad_additional_inputs,
                            filter_with_masks(
                                grad_addi, additional_inputs_tensor_masks
                            ),
                        )
                    ],
                    grad_xs,
                )
            )
            return flat_grads

        single_step_bw_xs = pytree.tree_map(first_slice_copy, bw_xs)
        bw_single_step_gm = materialize_as_graph(
            bw_single_step_wrapper,
            tuple(
                pytree.tree_flatten((bw_init, single_step_bw_xs, bw_additional_inputs))[
                    0
                ]
            ),
        )

        flat_grads = scan_op(
            bw_single_step_gm,
            pytree.tree_flatten(bw_init)[0],
            # TODO: torch.flip copies the tensor, we should optimize it away
            [torch.flip(x, (0,)) for x in pytree.tree_flatten(bw_xs)[0]],
            pytree.tree_flatten(bw_additional_inputs)[0],
        )
        if grad_spec is None:
            raise AssertionError("grad_spec must not be None after scan_op")
        grad_init, grad_additional_inputs, grad_xs = pytree.tree_unflatten(
            flat_grads, grad_spec
        )
        return (
            *grad_init,
            *[torch.flip(elem, (0,)) for elem in grad_xs],
            *fill_none_with_masks(
                grad_additional_inputs, additional_inputs_tensor_masks
            ),
        )


@scan_op.py_autograd_impl
def scan_autograd(combine_fn, init, xs, additional_inputs, mutated_arg_indices=""):
    with disable_proxy_modes_tracing():
        # If init was passed in with requires_grad=False, AOT joint creation drops it from
        # grad_primals and zero-fills, severing the carry chain and silently
        # zeroing gradients that should reach closed-over additional_inputs from
        # earlier steps. Thus, we temporarily flip requires_grad on the init tensors
        # during tracing and restore it afterwards; this only affects the traced
        # graphs, not the operands passed to apply. init elements are leaves.
        flipped = [
            t
            for t in init
            if (t.dtype.is_floating_point or t.dtype.is_complex) and not t.requires_grad
        ]
        try:
            for t in flipped:
                t.requires_grad_(True)
            hop_partitioned_graph: HopPartitionedGraph = (
                HopGraphMinCutPartitioner.create_partitioned_graph(
                    combine_fn,
                    (*init, *[first_slice_copy(x) for x in xs], *additional_inputs),
                    always_recompute_complex_exprs=True,
                )
            )
        finally:
            for t in flipped:
                t.requires_grad_(False)

    return ScanAutogradOp.apply(
        hop_partitioned_graph,
        len(init),
        len(xs),
        len(additional_inputs),
        *init,
        *xs,
        *additional_inputs,
    )


@scan_op.py_impl(ProxyTorchDispatchMode)
def scan_proxy_mode(
    mode, combine_fn, init, xs, additional_inputs, mutated_arg_indices=""
):
    return trace_scan(
        mode,
        scan_op,
        combine_fn,
        init,
        xs,
        additional_inputs,
        mutated_arg_indices=mutated_arg_indices,
    )


@scan_op.py_impl(FakeTensorMode)
def scan_fake_tensor_mode(
    mode, combine_fn, init, xs, additional_inputs, mutated_arg_indices=""
):
    with mode:
        scan_length = xs[0].shape[0]
        carry, outputs = _extract_carry_and_out(
            combine_fn(
                *init,
                *[first_slice_copy(inp) for inp in xs],
                *additional_inputs,
            ),
            len(init),
        )
        for t in outputs:
            if not isinstance(t, torch.Tensor) and t is not None:
                raise AssertionError(
                    f"Expected leaf to be a Tensor or None, got {type(t)}"
                )
        out = (
            *carry,
            *(
                stack_y(t, scan_length) if isinstance(t, torch.Tensor) else t
                for t in outputs
            ),
        )
        return out


@scan_op.py_functionalize_impl
def scan_functionalize(
    ctx, combine_fn, init, xs, additional_inputs, mutated_arg_indices=""
):
    from torch._higher_order_ops.utils import (
        _check_alias_and_mutation,
        _maybe_run_with_interpreter,
    )

    if hasattr(ctx, "mode"):
        hop_instance = HopInstance.create(
            scan_op,
            combine_fn,
            init,
            xs,
            additional_inputs,
            mutated_arg_indices=mutated_arg_indices,
        )
        if can_auto_functionalize(hop_instance):
            return do_auto_functionalize_v2(
                ctx.mode,
                hop_instance,
                tuple(
                    pytree.tree_flatten((combine_fn, init, xs, additional_inputs))[0]
                ),
                {},
            )

    unwrapped_xs = ctx.unwrap_tensors(xs)
    unwrapped_init = ctx.unwrap_tensors(init)
    unwrapped_additional_inputs = ctx.unwrap_tensors(additional_inputs)

    with ctx.redispatch_to_next():
        functional_combine_fn = ctx.functionalize(
            _maybe_run_with_interpreter(combine_fn)
        )
        sample_unwrapped_xs_sliced = (
            [first_slice_copy(inp) for inp in unwrapped_xs]
            if len(unwrapped_xs) > 0
            else [None]
        )
        sample_inputs = list(
            itertools.chain(
                unwrapped_init,
                sample_unwrapped_xs_sliced,
                unwrapped_additional_inputs,
            )
        )
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        _check_alias_and_mutation(combine_fn, sample_inputs, "scan", pre_dispatch)
        ret = scan_op(
            functional_combine_fn,
            unwrapped_init,
            unwrapped_xs,
            unwrapped_additional_inputs,
            mutated_arg_indices=mutated_arg_indices,
        )
    return ctx.wrap_tensors(ret)


@scan_op.py_impl(torch._C._functorch.TransformType.Vmap)
def scan_batch_rule(
    interpreter, combine_fn, init, xs, additional_inputs, mutated_arg_indices=""
):
    unbatched_args, in_dims = unwrap_batched(
        (init, xs, additional_inputs), interpreter.level()
    )
    # move to last dim to not interfere with scan's batching
    unbatched_init, unbatched_xs, unbatched_additional_inputs = pytree.tree_map(
        lambda x, bdim: x.movedim(bdim, -1) if bdim is not None else x,
        unbatched_args,
        in_dims,
    )
    after_move_dims = tuple(
        pytree.tree_flatten(
            pytree.tree_map(lambda x: -1 if x is not None else None, in_dims)
        )[0]
    )

    with interpreter.lower():
        out_dims = None

        def wrapper(*args):
            nonlocal out_dims
            outputs, per_slice_out_dims = restore_vmap(
                combine_fn,
                after_move_dims,
                interpreter.batch_size(),
                interpreter.randomness(),
            )(*args)
            # Note: outputs are not batched, we just move the batch dim to the end
            # this is to avoid it interfering with scan's batching
            outputs = tuple(
                pytree.tree_map(
                    lambda out, out_bdim: out.movedim(out_bdim, -1)
                    if out_bdim is not None
                    else out,
                    outputs,
                    per_slice_out_dims,
                )
            )
            out_dims = tuple(
                pytree.tree_map(
                    lambda out_bdim: -1 if out_bdim is not None else None,
                    per_slice_out_dims,
                )
            )
            return outputs

        op_kwargs = {}
        if mutated_arg_indices:
            op_kwargs["mutated_arg_indices"] = mutated_arg_indices
        unwrapped_out = scan_op(
            wrapper,
            unbatched_init,
            unbatched_xs,
            unbatched_additional_inputs,
            **op_kwargs,
        )

    if out_dims is None:
        raise AssertionError("out_dims must not be None after scan_op")
    batched_out = wrap_batched(unwrapped_out, out_dims, interpreter.level())
    return batched_out


# dense implementation for scan. Used for testing only.
def _fake_scan(combine_fn, init, xs=None, dim=0, reverse=False, length=None):
    carry_leaves, carry_spec = pytree.tree_flatten(init)
    inp_leaves, inp_spec = pytree.tree_flatten(xs)
    xs_has_tensors = any(isinstance(l, torch.Tensor) for l in inp_leaves)

    if length is not None and not xs_has_tensors:
        if length == 0:
            return init, _build_empty_output_for_length_zero(combine_fn, init)

        _user_combine_fn = combine_fn

        def combine_fn(carry, _ignored):
            return _user_combine_fn(carry, None)

        # Dummy length-N iteration counter; the wrapped combine_fn passes x=None.
        inp_leaves = [torch.zeros(length, dtype=torch.int64)]
        inp_spec = pytree.tree_structure(None)
        xs_has_tensors = True

    if not xs_has_tensors:
        return init, []
    result_flat = []
    carry = carry_leaves
    op = reversed if reverse else lambda x: x

    _, dummy_out = combine_fn(
        pytree.tree_unflatten(carry, carry_spec),
        pytree.tree_unflatten(
            [first_slice_copy(elem, dim) for elem in inp_leaves],
            inp_spec,
        ),
    )
    dummy_out_leaves, dummy_out_spec = pytree.tree_flatten(dummy_out)

    for ind in op(range(inp_leaves[0].size(dim))):
        xs = [elem.select(dim, ind) for elem in inp_leaves]

        carry, y = combine_fn(
            pytree.tree_unflatten(carry, carry_spec),
            pytree.tree_unflatten(xs, inp_spec),
        )
        carry, _ = pytree.tree_flatten(carry)
        y, _ = pytree.tree_flatten(y)
        result_flat.append(y)

    results: list[torch.Tensor | None] = []
    for leaf_idx, leaf in enumerate(dummy_out_leaves):
        if isinstance(leaf, torch.Tensor):
            if len(result_flat) == 0:
                stacked = torch.empty(
                    [0] + list(leaf.shape), dtype=leaf.dtype, device=leaf.device
                )
            else:
                stacked = torch.stack([e[leaf_idx] for e in op(result_flat)])
            results.append(
                torch.movedim(stacked, 0, dim) if dim < stacked.ndim else stacked
            )
        else:
            if leaf is not None:
                raise AssertionError(
                    f"Expected leaf to be a Tensor or None, got {type(leaf)}"
                )
            results.append(leaf)
    return (
        pytree.tree_unflatten(carry, carry_spec),
        pytree.tree_unflatten(results, dummy_out_spec),
    )
<|endoftext|>from collections.abc import Callable
from typing import Any

import torch
from torch._ops import HigherOrderOperator


class RegisterHookOp(HigherOrderOperator):
    """HOP that registers a backward hook on a tensor.

    The hook body is a traced subgraph that takes (grad, *freevars)
    and returns the modified gradient. During make_fx tracing, the
    hook registers on the FakeTensor and fires on the accumulated
    gradient during backward tracing, capturing the hook's effect
    in the joint graph.
    """

    def __init__(self) -> None:
        super().__init__("register_hook", cacheable=True)

    def __call__(
        self,
        tensor: torch.Tensor,
        hook_body: Callable[..., Any],
        *freevars: Any,
    ) -> torch.Tensor:
        def hook(grad: torch.Tensor) -> torch.Tensor | None:
            return hook_body(grad, *freevars)

        tensor.register_hook(hook)
        return tensor


register_hook_op = RegisterHookOp()
<|endoftext|>import typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, overload, TypeAlias, TypeVar
from typing_extensions import ParamSpec, TypeIs, TypeVarTuple, Unpack

import torch
import torch.fx.node
import torch.utils._pytree as pytree
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import is_custom_class
from torch._ops import HigherOrderOperator


_R = TypeVar("_R")
_P = ParamSpec("_P")
_Ts = TypeVarTuple("_Ts")


def is_graphable(val: object) -> TypeIs[torch.fx.node.BaseArgumentTypes]:
    """Definition: a graphable type is a type that is an acceptable input/output type to a FX node."""
    return isinstance(
        val, (*torch.fx.node.base_types, FakeScriptObject)
    ) or is_custom_class(type(val))


def is_graphable_type(typ: type[object]) -> bool:
    """Return whether the given type is graphable."""
    return (
        issubclass(typ, torch.fx.node.base_types)
        or is_custom_class(typ)
        or issubclass(typ, FakeScriptObject)
    )


def to_graphable(stuff: pytree.PyTree) -> tuple[list[object], pytree.TreeSpec]:
    """Flattens stuff into a flat list of graphable types."""
    # We can consider preserving things like List[int] to improve
    # perf and readability (right now that is all flattened out)
    flat_args, spec = pytree.tree_flatten(stuff)
    for arg in flat_args:
        if not is_graphable(arg):
            raise RuntimeError(
                f"Expected all pytree.tree_leaves of (args, kwargs) to be graphable types, but found "
                f"non-fx-graphable type {type(arg)}. If this type is meant to be constant, mark it as "
                f"via pytree.register_constant; otherwise, register it as a pytree."
            )
    return flat_args, spec


def from_graphable(
    flat_args: tuple[Unpack[_Ts]], spec: pytree.TreeSpec
) -> pytree.PyTree:
    """The inverse of to_graphable."""
    stuff = pytree.tree_unflatten(flat_args, spec)
    return stuff


def func_to_graphable(
    func: Callable[..., object],
) -> tuple[list[object], pytree.TreeSpec]:
    """
    Pack and flatten a function type into graphable types.
    This is useful for legalizing the function argument of `flat_apply`.
    """
    return pytree.tree_flatten(_ConstantFunction(func))


@dataclass(frozen=True, slots=True)
class _ConstantFunction(Generic[_P, _R]):
    func: Callable[_P, _R]

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        return self.func(*args, **kwargs)


pytree.register_constant(_ConstantFunction)


_OpTypes = (
    torch._ops.OpOverload | torch._ops.OpOverloadPacket | torch._ops.HigherOrderOperator
)
_op_types = typing.get_args(_OpTypes)


_Base: TypeAlias = torch.fx.node.BaseArgumentTypes
# pyrefly bug: pyrefly is complaining: Expected a type form, got instance of `Literal['_FXOutput']
# pyrefly: ignore[not-a-type]
_FXOutput = _Base | Sequence["_FXOutput"]


class FlatApply(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("flat_apply")

    def __call__(
        self,
        func: _OpTypes | pytree.TreeSpec,
        in_spec: pytree.TreeSpec,
        *flat_args: Unpack[_Ts],
        # If True then the output is checked to be valid. If False then it is up
        # to the caller to ensure the output is appropriate.
        checked_output: bool = True,
        **_unused: object,
    ) -> object:
        """
        Functions that take in non-graphable types cannot directly be put into FX graph.

        Given func(*args, **kwargs), if all of the non-graphable types are pytrees,
        then we're able to store a call to flat_apply(func, in_spec, *flat_args) in the FX graph.

        The semantics of flat_apply(func, in_spec, *flat_args) are roughly equivalent to:

        >>> def flat_apply_impl(func, in_spec, *flat_args):
        >>>     args, kwargs = pytree.tree_unflatten(flat_args, in_spec)
        >>>     output = func(*args, **kwargs)
        >>>     return output

        flat_apply supports the following two cases:
        - an input type is a container type (e.g. of tensors) registered as a pytree.
        We'll tree_flatten the input type and store the spec.
        - an input type is a constant type (i.e. torch.compile will specialize on it)
        registered with pytree.register_constant. The constant type goes directly
        into the spec.
        """
        if not (
            isinstance(func, _op_types)
            or is_custom_class(type(func))
            or pytree._is_constant_holder(func)
        ):
            raise AssertionError(
                f"func must be an op type, constant holder, or opaque callable, got {type(func)}"
            )
        if len(_unused) != 0:
            raise AssertionError(f"unexpected keyword arguments: {_unused}")
        # pyrefly: ignore[bad-argument-type]  # pyrefly bug?
        return impl(func, in_spec, flat_args, checked_output)


@overload
def is_valid_output(x: tuple[object, ...]) -> TypeIs[tuple[_FXOutput, ...]]: ...


@overload
def is_valid_output(x: Sequence[object]) -> TypeIs[Sequence[_FXOutput]]: ...


def is_valid_output(x: object) -> bool:
    if isinstance(x, (tuple, list)):
        return all(map(is_valid_output, x))
    return is_graphable(x)


def impl(
    func: _OpTypes | pytree.TreeSpec,
    in_spec: pytree.TreeSpec,
    flat_args: tuple[Unpack[_Ts]],
    checked_output: bool,
) -> _FXOutput:
    if isinstance(func, pytree.TreeSpec):
        # assume _ConstantFunction
        func = pytree._retrieve_constant(func)
        if not isinstance(func, _ConstantFunction):
            raise AssertionError(
                f"expected retrieved constant to be _ConstantFunction, got {type(func)}"
            )

    from torch._higher_order_ops.invoke_leaf_function import unflatten_args_with_modules

    with unflatten_args_with_modules(flat_args, in_spec) as (args, kwargs):
        out = func(*args, **kwargs)

    if checked_output:
        # For "normal" usage all outputs must either be graphable or
        # lists/tuples of graphables.
        if not is_valid_output(out):
            raise AssertionError(
                f"output must be graphable or nested list/tuple of graphables, got {type(out)}"
            )
    return out


flat_apply = FlatApply()
<|endoftext|># mypy: allow-untyped-defs
import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import (
    autograd_not_implemented,
    reenter_make_fx,
    register_fake,
    unique_graph_id,
)
from torch._ops import HigherOrderOperator
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree


# used for wrapping a function/op with context hints
class HintsWrapper(HigherOrderOperator):
    def __init__(self):
        super().__init__("hints_wrapper")

    def __call__(self, body_fn, args, kwargs, hints):
        r"""
        Call implementation of hints_wrapper

        Args:
            body_fn (Callable): A callable function that is within the scope
             that is being traced.

            args (Tuple of torch.Tensor/int/float/bool): A tuple of inputs to
             body_fn.

            kwargs (dict): Keyword argument to the body_fn.

            hints (dict): A dict of context hints which could be passed to
             backend compiler.
        """
        if not isinstance(args, tuple):
            args = tuple(args)

        if not all(isinstance(t, (torch.Tensor, int, float, bool)) for t in args):
            raise RuntimeError(
                f"args must be a tuple of tensors, ints, floats, or bools, got {args}"
            )

        if not isinstance(kwargs, dict):
            raise RuntimeError(f"kwargs must be a dict, got {type(kwargs)}")

        if len(kwargs) > 0:
            raise RuntimeError(
                f"kwargs except for hints are not supported, got {kwargs}"
            )

        if not isinstance(hints, dict):
            raise RuntimeError(f"hints must be a dict, got {type(hints)}")

        for k, v in hints.items():
            if not isinstance(k, str):
                raise RuntimeError(f"hints key must be a str, got {k}.")

            if not isinstance(v, (int, float, bool, str)):
                raise RuntimeError(
                    "hints must be a dict containing int, float, bool or str "
                    f"value, got value {v} for key {k}."
                )

        # pyrefly: ignore [missing-attribute]
        return super().__call__(body_fn, args, kwargs, hints)


hints_wrapper = HintsWrapper()


@hints_wrapper.py_impl(DispatchKey.CompositeExplicitAutograd)
def hints_wrapper_dense(body_fn, args, kwargs, hints):
    return body_fn(*args, **kwargs)


hints_wrapper.py_autograd_impl(
    autograd_not_implemented(hints_wrapper, deferred_error=True)
)


@register_fake(hints_wrapper, skip_cache=True)
def hints_wrapper_fake_tensor_mode(body_func, args, kwargs, hints):
    flat_args = pytree.tree_leaves(args)
    return body_func(*flat_args, **kwargs)


@hints_wrapper.py_functionalize_impl
def hints_wrapper_functionalize(ctx, body_fn, args, kwargs, hints):
    from torch._higher_order_ops.utils import _check_alias_and_mutation

    unwrapped_args = ctx.unwrap_tensors(args)
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)
    unwrapped_hints = ctx.unwrap_tensors(hints)
    with ctx.redispatch_to_next():
        functional_body_fn = ctx.functionalize(body_fn)
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        _check_alias_and_mutation(
            body_fn, unwrapped_args, "hints_wrapper", pre_dispatch
        )

        outputs = hints_wrapper(
            functional_body_fn,
            unwrapped_args,
            unwrapped_kwargs,
            unwrapped_hints,
        )
        return ctx.wrap_tensors(outputs)


def trace_hints_wrapper(proxy_mode, hints_wrapper, body_fn, args, kwargs, hints):
    flat_args = tuple(pytree.tree_leaves(args))
    body_graph = reenter_make_fx(body_fn)(*flat_args, **kwargs)

    _, body_graph_name = unique_graph_id(proxy_mode, prefix="hints_wrapper_body_graph")
    proxy_mode.tracer.root.register_module(body_graph_name, body_graph)

    new_args: tuple = (body_graph, flat_args, {})
    # merge hints into kwargs
    new_kwargs = {}
    new_kwargs["hints"] = hints

    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, new_args)
    proxy_kwargs = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, new_kwargs)

    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", hints_wrapper, proxy_args, proxy_kwargs, name="hints_wrapper"
    )

    out = body_fn(*flat_args, **kwargs)
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@hints_wrapper.py_impl(ProxyTorchDispatchMode)
def inner(proxy_mode, body_fn, args, kwargs, hints):
    if proxy_mode.enable_tracing:
        return trace_hints_wrapper(
            proxy_mode, hints_wrapper, body_fn, args, kwargs, hints
        )
    else:
        return hints_wrapper(body_fn, args, kwargs, hints)
<|endoftext|># mypy: allow-untyped-defs
import inspect
import itertools
import logging
import weakref
from collections.abc import Callable
from typing import Any
from typing_extensions import ParamSpec, TypeVar

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import (
    redirect_to_mode,
    reenter_make_fx,
    register_fake,
)
from torch._logging import warning_once
from torch._ops import HigherOrderOperator
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.types import _dtype
from torch.utils._debug_mode import DebugMode
from torch.utils.checkpoint import _CachedTorchDispatchMode, _CachingTorchDispatchMode


_P = ParamSpec("_P")
_R = TypeVar("_R")


log = logging.getLogger(__name__)


uid = itertools.count(1)


# Used for testing the HigherOrderOperator mechanism
class Wrap(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("wrap")

    def __call__(
        self, func: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs
    ) -> _R:
        # Dynamo already traces the body of HigherOrderOp beforehand when it
        # so no need to trace into it.
        import torch._dynamo  # noqa: F401
        from torch._dynamo import disable

        @disable
        def wrapper():
            result = func(*args, **kwargs)
            return result

        return wrapper()


wrap = Wrap()


class InductorCompiledCode(HigherOrderOperator):
    """
    Defines a HOP for wrapping inductor compiled functions as a callable.
    When used with torch.compile via "wrap_inductor_compiled_regions",
    this HOP will automatically be wrapped and redirect various torch dispatch modes.
    """

    def __init__(self) -> None:
        super().__init__("inductor_compiled_code")
        self._boxed_arg_indices = (1,)

    def __call__(self, func, inputs, *, name: str | None = None):
        # pyrefly: ignore [missing-attribute]
        return super().__call__(func, inputs, name=name)


inductor_compiled_code = InductorCompiledCode()
inductor_compiled_code.fallthrough(DispatchKey.AutogradCPU)
inductor_compiled_code.fallthrough(DispatchKey.AutogradCUDA)


_inductor_compiled_callable_id = itertools.count()


class InductorCompiledCallable:
    """
    A wrapper class that holds both the Inductor-compiled callable and the
    original FX graph for fake tensor propagation.
    Each instance gets a globally unique idx at creation (via atomic itertools.count).
    """

    def __init__(
        self,
        compiled_callable,
        original_gm=None,
        compile_region_name: str | None = None,
    ):
        self.idx = next(_inductor_compiled_callable_id)
        self.compiled_callable = compiled_callable
        self.original_gm = original_gm
        self.compile_region_name = compile_region_name
        # AOT autograd needs this to know inputs are passed as a list
        self._boxed_call = True

    def __call__(self, inputs):
        return self.compiled_callable(inputs)


class InductorCodeSideTable:
    """
    Side table for storing InductorCompiledCallable objects.

    We cannot put InductorCompiledCallable objects directly into the FX graph
    as graph nodes do not support arbitrary objects. We use this side table
    and pass callable.idx instead.

    Uses WeakValueDictionary so entries are automatically removed when
    the InductorCompiledCallable is no longer referenced elsewhere
    (e.g. after dynamo.reset() drops the compiled code cache).
    """

    def __init__(self):
        self.id_to_callable: weakref.WeakValueDictionary[
            int, InductorCompiledCallable
        ] = weakref.WeakValueDictionary()

    def add_callable(self, callable_obj: InductorCompiledCallable) -> int:
        """Register a callable and return its idx."""
        self.id_to_callable[callable_obj.idx] = callable_obj
        return callable_obj.idx

    def get_callable(self, idx: int) -> InductorCompiledCallable:
        """Get the callable at the given index."""
        if idx not in self.id_to_callable:
            raise AssertionError(f"Invalid inductor code index: {idx}")
        return self.id_to_callable[idx]

    def __getstate__(self):
        # Convert WeakValueDictionary to regular dict for pickling
        return {"id_to_callable": dict(self.id_to_callable)}

    def __setstate__(self, state):
        self.id_to_callable = weakref.WeakValueDictionary(state["id_to_callable"])

    def reset_table(self) -> None:
        """Reset the table."""
        self.id_to_callable = weakref.WeakValueDictionary()


inductor_code_side_table = InductorCodeSideTable()


def _resolve_inductor_callable(
    func: int | InductorCompiledCallable,
) -> InductorCompiledCallable:
    """
    Resolve func to an InductorCompiledCallable.

    func is either an InductorCompiledCallable directly (from post_compile)
    or an int index into the side table (from a traced FX graph node).
    """
    if isinstance(func, int):
        return inductor_code_side_table.get_callable(func)
    if not isinstance(func, InductorCompiledCallable):
        raise AssertionError(f"Unexpected func type: {type(func)}")
    return func


@inductor_compiled_code.py_impl(DispatchKey.CompositeExplicitAutograd)
def inductor_compiled_code_impl(func, inputs, *, name=None):
    resolved = _resolve_inductor_callable(func)
    return resolved.compiled_callable(inputs)


redirect_to_mode(inductor_compiled_code, DebugMode)
redirect_to_mode(inductor_compiled_code, _CachingTorchDispatchMode)
redirect_to_mode(inductor_compiled_code, _CachedTorchDispatchMode)


@register_fake(inductor_compiled_code)
def inductor_compiled_code_fake(func, inputs, *, name=None):
    resolved = _resolve_inductor_callable(func)
    if resolved.original_gm is None:
        raise RuntimeError(
            "inductor_compiled_code original_gm is None — the compiled graph may "
            "have been serialized without it. Recompile to restore."
        )
    # Run the original FX graph under FakeTensorMode to re-derive output
    # shapes, dtypes, and aliasing from the input fake tensors.
    return tuple(resolved.original_gm(*inputs))


@inductor_compiled_code.py_functionalize_impl
def inductor_compiled_code_functionalize(ctx, func, inputs, *, name=None):
    # Unwrap the functional tensors to get the underlying tensors
    unwrapped_inputs = ctx.unwrap_tensors(inputs)

    # Redispatch to the next handler in the dispatch chain
    with ctx.redispatch_to_next():
        kwargs = {"name": name} if name is not None else {}
        result = inductor_compiled_code(func, unwrapped_inputs, **kwargs)
        return ctx.wrap_tensors(result)


@inductor_compiled_code.py_impl(ProxyTorchDispatchMode)
def inductor_compiled_code_proxy(mode, func, inputs, *, name=None):
    resolved = _resolve_inductor_callable(func)

    # Run the fake impl to get example outputs for tracing
    kwargs = {"name": name} if name is not None else {}
    example_out = inductor_compiled_code(func, inputs, **kwargs)

    # Register in side table so the FX node stores a serializable int
    callable_idx = inductor_code_side_table.add_callable(resolved)

    proxy_inputs = pytree.tree_map(mode.tracer.unwrap_proxy, inputs)

    out_proxy = mode.tracer.create_proxy(
        "call_function",
        inductor_compiled_code,
        (callable_idx, proxy_inputs),
        kwargs,
    )

    return track_tensor_tree(example_out, out_proxy, constant=None, tracer=mode.tracer)


class WrapWithSetGradEnabled(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("wrap_with_set_grad_enabled")

    def __call__(
        self,
        enable_grad: bool,
        wrapped_func: Callable[_P, _R],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _R:
        # Dynamo already traces the body of HigherOrderOp beforehand when it
        # so no need to trace into it.
        import torch._dynamo
        from torch._dynamo import disable

        @disable
        def wrapper():
            prev = torch.is_grad_enabled()
            torch.set_grad_enabled(enable_grad)
            res = wrapped_func(*args, **kwargs)
            torch.set_grad_enabled(prev)
            return res

        return wrapper()


wrap_with_set_grad_enabled = WrapWithSetGradEnabled()


class WrapWithAutocast(HigherOrderOperator):
    def __init__(self):
        super().__init__("wrap_with_autocast")

    def __call__(
        self,
        device_type: str,
        dtype: _dtype | None,
        enabled: bool,
        cache_enabled: bool | None,
        wrapped_func: Callable[_P, _R],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _R:
        # Dynamo already traces the body of HigherOrderOp beforehand when it
        # so no need to trace into it.
        import torch._dynamo
        from torch._dynamo import disable

        @disable
        def wrapper():
            with torch.autocast(device_type, dtype, enabled, cache_enabled):
                return wrapped_func(*args, **kwargs)

        return wrapper()


wrap_with_autocast = WrapWithAutocast()


# This HOP allows you to bypass dynamo tracing of the wrapper function while
# still tracing the inner function.
# Takes two callables: The first, `wrapper_fn`, accepts `inner_fn` and returns a
# callable with the same signature. The second is the `inner_fn` itself. Any
# extra *args and **kwargs are forwarded to `wrapper_fn(inner_fn)` when it is
# executed.
class DynamoBypassingWrapper(HigherOrderOperator):
    def __init__(self):
        super().__init__("dynamo_bypassing_wrapper")

    def __call__(
        self,
        wrapper_fn_or_key,
        inner_fn,
        *args,
        **kwargs,
    ):
        # Dynamo already traces the body of HigherOrderOp beforehand when it
        # so no need to trace into it.
        import torch._dynamo
        from torch._dynamo import disable

        is_compiling = isinstance(wrapper_fn_or_key, str)
        if is_compiling:
            if not isinstance(inner_fn, torch.fx.GraphModule):
                raise AssertionError(
                    f"expected inner_fn to be torch.fx.GraphModule, got {type(inner_fn)}"
                )
            wrapper_fn = inner_fn.meta[wrapper_fn_or_key]
        else:
            wrapper_fn = wrapper_fn_or_key

        @disable
        def wrapper():
            return wrapper_fn(inner_fn)(*args, **kwargs)

        return wrapper()


dynamo_bypassing_wrapper = DynamoBypassingWrapper()


class WrapActivationCheckpoint(HigherOrderOperator):
    """
    This operator is used to wrap torch.utils.checkpoint. This avoids
    TorchDynamo to look into saved tensor hooks and directly passes the control
    to AOT Autograd, which is ok with tracing saved tensor hooks. As a result of
    AOT tracing torch.utils.checkpoint code, we have a backward graph with
    recomputed forward nodes.

    However, we might deprecate this operator soon. The difficulty arises in the
    functionalization of rng ops. Today, there are two different
    functionalization of rng ops - one at AOT autograd and other at Inductor.
    And they are difficult to map to each other. The rng states also complicate
    pattern matching in Inductor. Due to the ease of implementation, we are
    currently inclined towards functionalization at Inductor level, which means
    that duplication/recomputation is done as a compiler pass in the
    partitioners. See TagActivationCheckpoint for more information.
    """

    def __init__(self) -> None:
        super().__init__("wrap_activation_checkpoint", cacheable=False)

    def __call__(self, function: GraphModule, *args: Any, **kwargs: Any) -> Any:
        # use_reentrant is set to False because this op is going to be traced.
        # And we ensure that AOT Autograd traces through the non reentrant
        # version of checkpointing.
        import torch.fx.traceback as fx_traceback
        from torch.fx import Interpreter

        kwargs["use_reentrant"] = False
        kwargs["preserve_rng_state"] = False
        # Using interpreter allows preservation of metadata through torch.compile stack.
        with fx_traceback.preserve_node_meta():
            from torch.utils.checkpoint import checkpoint

            return checkpoint(Interpreter(function).run, *args, **kwargs)


wrap_activation_checkpoint = WrapActivationCheckpoint()


class TagActivationCheckpoint(HigherOrderOperator):
    """
    This operator is supposed to be used only with torch.compile stack. This
    accepts a Fx graph module which needs to be checkpointed. This operator adds
    "recomputable" tag to the nodes of the Fx graph that should be recomputed.

    The goal is to:
    1. Avoid using Dynamo to trace through saved tensor hooks.
    2. For selective checkpointing case, let AOTAutograd trace through
       saved tensor hooks but has special logic with TorchDispatchMode to override
       the usual saved_tensor_hooks fn logic in order to tag the nodes.
    3. Rely on the partitioners to actually duplicate the nodes.
    This sits well in the torch.compile stack, because by the time graph
    reaches partitioner, inductor has already run its functionalization of rng
    ops (by setting fixed seed for each random op, see `replace_random_passes`).
    Therefore, the duplication of nodes, by design, respects the rng states in
    the forward and recomputed forward in backward.
    """

    def __init__(self) -> None:
        super().__init__("tag_activation_checkpoint", cacheable=True)

    @staticmethod
    def divide_kwargs(kwargs):
        """
        checkpoint fn can have mixed kwargs between checkpointed fn and
        checkpoint fn itself. For example
        >> def gn(x, y, z=None):
        >>     a = torch.matmul(x, y)
        >>     if z is not None:
        >>         return torch.matmul(a, z)
        >>     return a
        >> def fn(x, y, z):
        >>     return torch.cos(checkpoint(gn, x, y, use_reentrant=False, z=z))
        In the above case, z belongs to checkpointed function gn, but
        use_reentrant belongs to the checkpoint function. This function splits
        the kwargs into checkpoint_kwargs and gmod_kwargs (or
        checkpointed_fn_kwargs).
        We do sorting to ensure same graph from run to run for better
        debuggability. It is not required for correctness.
        """
        from torch.utils.checkpoint import checkpoint

        ckpt_signature = inspect.signature(checkpoint)
        checkpoint_keys = set()
        for name in ckpt_signature.parameters:
            if name in ("function", "args", "kwargs"):
                continue
            checkpoint_keys.add(name)

        # `preserve_rng_state` is not a regular kwarg
        checkpoint_keys.add("preserve_rng_state")

        checkpoint_kwargs = {
            name: kwargs[name] for name in kwargs if name in checkpoint_keys
        }
        gmod_kwargs = {
            name: kwargs[name] for name in kwargs if name not in checkpoint_keys
        }
        return checkpoint_kwargs, gmod_kwargs

    def __call__(self, gmod, *args, **kwargs):
        dispatch_key_set = torch._ops._compute_keyset(
            args, kwargs, self.non_fallthrough_keys
        )
        dispatch_key = dispatch_key_set.highestPriorityTypeId()
        if dispatch_key == torch._C.DispatchKey.PreDispatch:
            # pyrefly: ignore [missing-attribute]
            return super().__call__(gmod, *args, **kwargs)

        return tag_activation_checkpoint_impl(gmod, *args, **kwargs)


tag_activation_checkpoint = TagActivationCheckpoint()


def _always_prefer_recompute(ctx, op, *args, **kwargs):
    from torch.utils.checkpoint import CheckpointPolicy

    return CheckpointPolicy.PREFER_RECOMPUTE


def tag_activation_checkpoint_impl(gmod, *args, **kwargs):
    import functools

    import torch.fx.traceback as fx_traceback
    from torch.fx import Interpreter
    from torch.utils.checkpoint import create_selective_checkpoint_contexts

    unique_graph_id = next(uid)
    if "_checkpoint_context_fn" in gmod.meta:
        context_fn = gmod.meta["_checkpoint_context_fn"]
        warning_once(
            log,
            """
Detected that context_fn is passed to torch.utils.checkpoint under torch.compile.
Please make sure the checkpointed region does not contain in-place ops (e.g. torch.relu_).
""",
        )
    else:
        # Vanilla AC: use the SAC path with a policy that always recomputes.
        context_fn = functools.partial(
            create_selective_checkpoint_contexts, _always_prefer_recompute
        )

    def context_fn_with_graph_id():
        fwd_ctx, recomp_ctx = context_fn()
        # Plumb ac_graph_id so _CachingTorchDispatchMode tags all nodes
        # (including ops from desugared HOPs like custom autograd.Function).
        fwd_ctx.ac_graph_id = unique_graph_id
        return fwd_ctx, recomp_ctx

    # use_reentrant is set to False because this op is going to be traced.
    # And we ensure that AOT Autograd traces through the non reentrant
    # version of checkpointing.
    kwargs["use_reentrant"] = False
    # preserve_rng_state is set to False because we want to prevent AOTAutograd from tracing through
    # `torch.random.fork_rng` op (which is not supported yet under CUDA).
    # This doesn't mean that we don't preserve RNG state. Instead, we will always preserve RNG state
    # regardless of this flag (by doing RNG functionalization via `replace_random_passes` in Inductor
    # instead of in AOTAutograd).
    kwargs["preserve_rng_state"] = False
    kwargs["context_fn"] = context_fn_with_graph_id
    # Disable early stop to prevent _StopRecomputationError from interrupting
    # recomputation between _vmap_increment_nesting and _vmap_decrement_nesting,
    # which would leak a functorch dynamic layer.
    kwargs["early_stop"] = False
    # Using interpreter allows preservation of metadata through torch.compile stack.
    # We use a wrapper instead of passing Interpreter(gmod).run directly because
    # checkpoint's recompute_fn captures the function in a closure. A bound method
    # reference would keep the Interpreter alive, whose env dict retains the output
    # tensors and prevents the autograd graph from being freed.

    def run_with_interpreter(*args):
        return Interpreter(gmod).run(*args)

    with fx_traceback.preserve_node_meta():
        from torch.utils.checkpoint import checkpoint

        return checkpoint(run_with_interpreter, *args, **kwargs)


@tag_activation_checkpoint.py_impl(ProxyTorchDispatchMode)
def proxy_mode_key(
    proxy_mode: ProxyTorchDispatchMode,
    gmod: GraphModule,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor]:
    import torch.fx.traceback as fx_traceback
    from torch.fx import Interpreter

    if not proxy_mode.pre_dispatch:
        raise AssertionError(
            "post-dispatch mode should have inlined in the Autograd key"
        )
    example_out = tag_activation_checkpoint(gmod, *args, **kwargs)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)  # type: ignore[union-attr]
    proxy_kwargs = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, kwargs)  # type: ignore[union-attr]
    qualname = proxy_mode.tracer.get_fresh_qualname("wrap_body")  # type: ignore[union-attr]

    # TODO (tmanlaibaatar) don't we need flat_apply here??
    # Dynamo already traced the gmod body without kwargs
    flat_args, _ = pytree.tree_flatten(args)
    with fx_traceback.preserve_node_meta():
        gmod_aten = reenter_make_fx(Interpreter(gmod).run)(*flat_args)
        gmod_aten.meta["_checkpoint_context_fn"] = gmod.meta["_checkpoint_context_fn"]
    proxy_mode.tracer.root.register_module(qualname, gmod_aten)  # type: ignore[union-attr]
    proxy_gmod = proxy_mode.tracer.unwrap_proxy(gmod_aten)  # type: ignore[union-attr, call-overload]
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function",
        tag_activation_checkpoint,
        (proxy_gmod, *proxy_args),
        proxy_kwargs,
    )
    return track_tensor_tree(
        example_out, out_proxy, constant=None, tracer=proxy_mode.tracer
    )
<|endoftext|>import operator
from collections import Counter
from typing import Any, TYPE_CHECKING

import torch
from torch.fx.graph_module import GraphModule


if TYPE_CHECKING:
    from torch.fx.node import Node


def count_subgraph_uses(gm: GraphModule) -> Counter[str]:
    """Count invoke_subgraph references by identifier string across all
    descendant GraphModules."""
    counts: Counter[str] = Counter()
    for mod in gm.modules():
        if not isinstance(mod, GraphModule):
            continue
        for node in mod.graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.invoke_subgraph
        ):
            counts[str(node.args[1])] += 1
    return counts


def inline_single_use_invoke_subgraph(gm: GraphModule) -> GraphModule:
    """Inline invoke_subgraph HOPs whose subgraph module has exactly one call
    site across the entire module tree.

    When a subgraph has only a single caller, invoke_subgraph adds overhead
    without any deduplication benefit, so we inline it unconditionally.
    Subgraph modules that are shared across sibling modules (e.g. two AC body
    modules referencing the same nested_compile_region) are counted globally
    by module identity and preserved when referenced more than once.
    """
    global_counts = count_subgraph_uses(gm)
    inline_single_use_recursive(gm, global_counts)
    return gm


def inline_single_use_recursive(gm: GraphModule, global_counts: Counter[str]) -> None:
    # Recursively apply to nested subgraph modules first.
    for name, mod in gm.named_modules():
        if name and isinstance(mod, GraphModule):
            inline_single_use_recursive(mod, global_counts)

    invoke_nodes = list(
        gm.graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.invoke_subgraph
        )
    )
    if not invoke_nodes:
        return

    single_use_nodes = [
        node
        for node in invoke_nodes
        if global_counts[str(node.args[1])] == 1
        and not getattr(gm, str(node.args[0].target)).meta.get(
            "nested_region_config", None
        )
    ]
    if not single_use_nodes:
        return

    inline_invoke_subgraph_nodes(gm, single_use_nodes)


def hoist_submodules(gm: GraphModule, subgraph: GraphModule) -> dict[str, str]:
    """Move submodules of *subgraph* onto *gm*, returning a name-remap dict.

    For each direct child module of *subgraph* (e.g. ``fwd_body_0``), register
    it on *gm* under a unique name and return a mapping from the old name to the
    new name so that ``get_attr`` targets copied from *subgraph*'s graph can be
    rewritten.
    """
    name_map: dict[str, str] = {}
    existing_names = {name for name, _ in gm.named_children()}
    for child_name, child_mod in list(subgraph.named_children()):
        new_name = child_name
        counter = 0
        while new_name in existing_names:
            new_name = f"{child_name}_{counter}"
            counter += 1
        gm.add_module(new_name, child_mod)
        existing_names.add(new_name)
        name_map[child_name] = new_name
    return name_map


def inline_invoke_subgraph_nodes(gm: GraphModule, invoke_nodes: list["Node"]) -> None:
    """Shared helper that inlines a list of invoke_subgraph nodes."""
    for node in invoke_nodes:
        get_attr_node: torch.fx.Node = node.args[0]  # pyrefly: ignore[bad-assignment]
        operands = node.args[2:]

        subgraph: GraphModule = getattr(gm, str(get_attr_node.target))

        # Hoist sub-submodules (e.g. fwd_body_0, bwd_body_0) to gm so that
        # get_attr nodes copied from the subgraph resolve correctly.
        name_map = hoist_submodules(gm, subgraph)

        env: dict[Node, Any] = dict(
            zip(subgraph.graph.find_nodes(op="placeholder"), operands)
        )

        with gm.graph.inserting_before(node):
            for sub_node in subgraph.graph.nodes:
                if sub_node.op in ("placeholder", "output"):
                    continue
                new_node = gm.graph.node_copy(sub_node, lambda n: env[n])
                if sub_node.op == "get_attr" and sub_node.target in name_map:
                    new_node.target = name_map[sub_node.target]
                env[sub_node] = new_node

        output_values = subgraph.graph.output_node().args[0]

        for user in list(node.users):
            if user.op == "call_function" and user.target is operator.getitem:
                idx = user.args[1]
                user.replace_all_uses_with(env[output_values[idx]])  # pyrefly: ignore
                gm.graph.erase_node(user)

        gm.graph.erase_node(node)

        if not get_attr_node.users:
            gm.graph.erase_node(get_attr_node)


def inline_invoke_subgraph(gm: GraphModule) -> GraphModule:
    """Inline all invoke_subgraph HOPs, producing a flat FX graph.

    This is useful when downstream compilers (like vllm-compile) don't support
    HOPs or prefer a flat graph, but we still want the Dynamo tracing-time
    benefits of auto-caching (trace once, stamp out cached calls).
    """
    # Recursively inline nested invoke_subgraph calls inside subgraph
    # modules first, so their graphs are flat before we inline them here.
    for name, mod in gm.named_modules():
        if name and isinstance(mod, GraphModule):
            inline_invoke_subgraph(mod)

    invoke_nodes = list(
        gm.graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.invoke_subgraph
        )
    )
    if not invoke_nodes:
        return gm

    inline_invoke_subgraph_nodes(gm, invoke_nodes)
    return gm
<|endoftext|>"""Stateless PRNG APIs.

These are experimental and subject to change without notice.
Access via ``torch.func._random``.
"""

from collections.abc import Sequence

import torch


def key(
    seed: int, *, device: torch.device | None = None, impl: str = "philox4x32-10"
) -> torch.Tensor:
    r"""Create a PRNG key from a seed.

    A key is a tensor that encodes the state needed to deterministically
    produce random values. Keys are consumed by generation functions to produce
    reproducible random tensors without any global state. The internal
    representation of the key depends on the chosen PRNG algorithm.

    Args:
        seed (int): The seed value for the PRNG.
        device (:class:`torch.device`, optional): The desired device for the
            returned key. Default: ``cpu``.
        impl (str): PRNG algorithm. Currently only ``"philox4x32-10"`` is
            supported.

    Returns:
        A tensor representing the PRNG key.

    .. note::

        For the ``"philox4x32-10"`` algorithm, the key is a uint64 tensor of
        shape ``(2,)`` encoding a ``(seed, offset)`` pair. The offset determines
        the starting position in the Philox output stream.

    Example::

        >>> key = torch.func._random.key(42, device="cuda")  # doctest: +SKIP
    """
    if impl != "philox4x32-10":
        raise NotImplementedError(f"key() does not support PRNG impl '{impl}'")

    # (seed, offset)
    return torch.tensor([seed, 0], dtype=torch.uint64, device=device)


def split(key: torch.Tensor, num: int = 2) -> torch.Tensor:
    r"""Split a PRNG key into ``num`` new independent keys.

    Each returned key produces a different, deterministic random sequence.
    This is the primary mechanism for deriving multiple independent keys from
    a single parent key without mutating any state.

    Supports batched keys: if ``key`` has shape ``(*batch, K)``, each key in the
    batch is split independently and the result has shape ``(num, *batch, K)``.

    Args:
        key (Tensor): A PRNG key returned by :func:`key`, :func:`split`, or
            :func:`fold_in`.
        num (int): Number of keys to produce. Default: ``2``.

    Returns:
        A tensor of shape ``(num, *key.shape)`` containing the derived keys.

    Example::

        >>> key = torch.func._random.key(42, device="cuda")  # doctest: +SKIP
        >>> k1, k2 = torch.func._random.split(key)  # doctest: +SKIP
    """
    return torch.ops.aten._philox_key_split(key, num)


def fold_in(key: torch.Tensor, data: int | torch.Tensor) -> torch.Tensor:
    r"""Deterministically derive a new key by folding in an integer value.

    ``data`` may be a Python ``int`` or a single-item ``uint64`` tensor on the
    same device as ``key``. Note that passing ``data`` as a tensor prevents it
    from being baked into a captured CUDA graph, so the graph can be replayed
    with a different value without recapture.

    Equivalent to ``split(key, data + 1)[data]``, but more efficient when
    only a single derived key is needed. Useful for associating a key with
    a loop iteration, layer index, or other integer identifier.

    Supports batched keys: if ``key`` has shape ``(*batch, K)``, ``data`` is
    folded into each key independently.

    Args:
        key (Tensor): A PRNG key returned by :func:`key`, :func:`split`, or
            :func:`fold_in`.
        data (int or Tensor): The value to fold into the key, interpreted as
            uint64. An ``int`` must be within the inclusive range
            ``[-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff]``. Negative inputs
            are remapped to positive values with the formula
            ``0x1_0000_0000_0000_0000 + data``. A tensor must have dtype
            ``uint64``, contain a single value, and reside on the same device
            as ``key``.

    Returns:
        A new key tensor with the same shape as ``key``.

    Example::

        >>> key = torch.func._random.key(42, device="cuda")  # doctest: +SKIP
        >>> k0 = torch.func._random.fold_in(key, 0)  # doctest: +SKIP
        >>> k1 = torch.func._random.fold_in(key, 1)  # doctest: +SKIP
        >>> # Equivalent to split:
        >>> keys = torch.func._random.split(key, 2)  # doctest: +SKIP
        >>> assert torch.equal(k0, keys[0])  # doctest: +SKIP
        >>> assert torch.equal(k1, keys[1])  # doctest: +SKIP
    """
    if isinstance(data, torch.Tensor):
        return torch.ops.aten._philox_key_fold_in.Tensor(key, data)
    data = int(data)
    if not -(1 << 63) <= data <= (1 << 64) - 1:
        raise ValueError(
            f"fold_in: int data must be in [-2**63, 2**64 - 1], got {data}"
        )
    # Reinterpret as signed int64 due to ATen op schema; kernel will cast back
    if data >= (1 << 63):
        data -= 1 << 64
    return torch.ops.aten._philox_key_fold_in(key, data)


def normal_(
    key: torch.Tensor,
    result: torch.Tensor,
    *,
    mean: float = 0.0,
    std: float = 1.0,
) -> torch.Tensor:
    r"""Fill ``result`` in-place with normal random values from a PRNG key.

    The values are drawn from a normal distribution with the specified ``mean``
    and ``std``. The output is fully determined by the key, so calling with the
    same key always produces the same result.

    Supports batched keys: if ``key`` has shape ``(*batch, K)``, the leading
    dimensions of ``result`` must be broadcastable with ``*batch`` and each key
    independently generates its slice of the output.

    Args:
        key (Tensor): A PRNG key returned by :func:`key`, :func:`split`, or
            :func:`fold_in`.
        result (Tensor): The output tensor to fill in-place.
        mean (float): Mean of the normal distribution. Default: ``0.0``.
        std (float): Standard deviation of the normal distribution. Default: ``1.0``.

    Returns:
        ``result``, filled with normal random values.

    Example::

        >>> key = torch.func._random.key(42, device="cuda")  # doctest: +SKIP
        >>> result = torch.empty(1000, device="cuda")  # doctest: +SKIP
        >>> torch.func._random.normal_(key, result)  # doctest: +SKIP
    """
    return torch.ops.aten._philox_normal_(result, key, mean, std)


def normal(
    key: torch.Tensor,
    *shape: tuple[int, ...],
    mean: float = 0.0,
    std: float = 1.0,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    r"""Generate normally distributed random values from a PRNG key.

    Produces a tensor of the given shape filled with values drawn from a normal
    distribution with the specified ``mean`` and ``std``. The output is fully
    determined by the key, so calling with the same key always returns the same
    result. The output is placed on the same device as ``key``.

    Supports batched keys: if ``key`` has shape ``(*batch, K)``, the leading
    dimensions of ``shape`` must be broadcastable with ``*batch`` and each key
    independently generates its slice of the output.

    Args:
        key (Tensor): A PRNG key returned by :func:`key`, :func:`split`, or
            :func:`fold_in`.
        *shape (int): The desired output shape.
        mean (float): Mean of the normal distribution. Default: ``0.0``.
        std (float): Standard deviation of the normal distribution. Default: ``1.0``.
        dtype (:class:`torch.dtype`, optional): The desired dtype. Default: ``torch.float32``.

    Returns:
        A tensor of the given shape filled with normal random values.

    Example::

        >>> key = torch.func._random.key(42, device="cuda")  # doctest: +SKIP
        >>> torch.func._random.normal(key, (1000,))  # doctest: +SKIP
    """
    if len(shape) == 1 and isinstance(shape[0], Sequence):
        # pyrefly: ignore [bad-argument-type]
        shape = tuple(shape[0])
    if dtype is None:
        dtype = torch.float32
    # pyrefly: ignore [no-matching-overload]
    result = torch.empty(shape, dtype=dtype, device=key.device)
    return normal_(key, result, mean=mean, std=std)


def uniform_(
    key: torch.Tensor,
    result: torch.Tensor,
    *,
    low: float = 0.0,
    high: float = 1.0,
) -> torch.Tensor:
    r"""Fill ``result`` in-place with uniform random values from a PRNG key.

    The values are drawn uniformly from the interval ``[low, high)``. The output
    is fully determined by the key, so calling with the same key always produces
    the same result.

    Supports batched keys: if ``key`` has shape ``(*batch, K)``, the leading
    dimensions of ``result`` must be broadcastable with ``*batch`` and each key
    independently generates its slice of the output.

    Args:
        key (Tensor): A PRNG key returned by :func:`key`, :func:`split`, or
            :func:`fold_in`.
        result (Tensor): The output tensor to fill in-place.
        low (float): Lower bound (inclusive) of the uniform distribution. Default: ``0.0``.
        high (float): Upper bound (exclusive) of the uniform distribution. Default: ``1.0``.

    Returns:
        ``result``, filled with uniform random values.

    Example::

        >>> key = torch.func._random.key(42, device="cuda")  # doctest: +SKIP
        >>> result = torch.empty(1000, device="cuda")  # doctest: +SKIP
        >>> torch.func._random.uniform_(key, result)  # doctest: +SKIP
    """
    return torch.ops.aten._philox_uniform_(result, key, low, high)


def uniform(
    key: torch.Tensor,
    *shape: tuple[int, ...],
    low: float = 0.0,
    high: float = 1.0,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    r"""Generate uniformly distributed random values from a PRNG key.

    Produces a tensor of the given shape filled with values drawn uniformly
    from the interval ``[low, high)``. The output is fully determined by the
    key, so calling with the same key always returns the same result. The output
    is placed on the same device as ``key``.

    Supports batched keys: if ``key`` has shape ``(*batch, K)``, the leading
    dimensions of ``shape`` must be broadcastable with ``*batch`` and each key
    independently generates its slice of the output.

    Args:
        key (Tensor): A PRNG key returned by :func:`key`, :func:`split`, or
            :func:`fold_in`.
        *shape (int): The desired output shape.
        low (float): Lower bound (inclusive) of the uniform distribution. Default: ``0.0``.
        high (float): Upper bound (exclusive) of the uniform distribution. Default: ``1.0``.
        dtype (:class:`torch.dtype`, optional): The desired dtype. Default: ``torch.float32``.

    Returns:
        A tensor of the given shape filled with uniform random values.

    Example::

        >>> key = torch.func._random.key(42, device="cuda")  # doctest: +SKIP
        >>> torch.func._random.uniform(key, (1000,))  # doctest: +SKIP
    """
    if len(shape) == 1 and isinstance(shape[0], Sequence):
        # pyrefly: ignore [bad-argument-type]
        shape = tuple(shape[0])
    if dtype is None:
        dtype = torch.float32
    # pyrefly: ignore [no-matching-overload]
    result = torch.empty(shape, dtype=dtype, device=key.device)
    return uniform_(key, result, low=low, high=high)
<|endoftext|>from torch._functorch.apis import grad, grad_and_value, vmap
from torch._functorch.batch_norm_replacement import replace_all_batch_norm_modules_
from torch._functorch.eager_transforms import (
    debug_unwrap,
    functionalize,
    hessian,
    jacfwd,
    jacrev,
    jvp,
    linearize,
    vjp,
)
from torch._functorch.einops import rearrange
from torch._functorch.functional_call import functional_call, stack_module_state


__all__ = [
    "grad",
    "grad_and_value",
    "vmap",
    "replace_all_batch_norm_modules_",
    "functionalize",
    "hessian",
    "jacfwd",
    "jacrev",
    "jvp",
    "linearize",
    "rearrange",
    "vjp",
    "functional_call",
    "stack_module_state",
    "debug_unwrap",
]
<|endoftext|># mypy: allow-untyped-defs
import multiprocessing
import os
import threading
from multiprocessing import reduction
from multiprocessing.util import register_after_fork

import torch


try:
    # Early load resource_sharer to prevent a partially initialized instance
    # from being inherited in a forked child process. The reduce_storage method
    # requires this module indirectly through DupFd(). The built-in mp.Queue
    # class pickles arguments in a background thread which may overlap with the
    # fork.
    import multiprocessing.resource_sharer
except ImportError:
    pass


class StorageWeakRef:
    r"""A weak reference to a Storage.

    The cdata member is a Python number containing the integer representation of
    the Storage pointer.
    """

    __slots__ = ["cdata", "_free_weak_ref"]

    def __init__(self, storage):
        self.cdata = storage._weak_ref()
        # Save a direct reference to _free_weak_ref because the `torch` module
        # might be cleared during Python shutdown before this module is cleared.
        self._free_weak_ref = torch.Storage._free_weak_ref  # type: ignore[attr-defined]

    @classmethod
    def from_weakref(cls, cdata):
        instance = cls.__new__(cls)
        instance.cdata = cdata
        instance._free_weak_ref = torch.Storage._free_weak_ref  # type: ignore[attr-defined]
        return instance

    def expired(self):
        return torch.Storage._expired(self.cdata)  # type: ignore[attr-defined]

    def __del__(self):
        self._free_weak_ref(self.cdata)

    def __hash__(self):
        return self.cdata

    def __eq__(self, other):
        if id(self) == id(other):
            return True
        return self.cdata == other.cdata


class SharedCache(dict):
    """Dictionary from multiprocessing handles to StorageWeakRef."""

    def __init__(self) -> None:
        # free_dead_references() is called if the len exceeds the current
        # limit. The limit scales with the number of remaining live objects.
        self.limit = 128
        # `fork` inherits lock state, so in case we fork when the lock is held,
        # we register a function to reset the lock to a new object to avoid
        # possible deadlocks, following python multiprocessing library design.
        self._after_fork()
        register_after_fork(self, SharedCache._after_fork)

    def _after_fork(self):
        self.lock = threading.Lock()

    def get(self, key):  # type: ignore[override]
        with self.lock:
            return dict.get(self, key)

    def __setitem__(self, key, storage_ref):
        with self.lock:
            dict.__setitem__(self, key, storage_ref)
            if len(self) > self.limit:
                self.free_dead_references()

    def free_dead_references(self):
        live = 0
        for key, storage_ref in list(self.items()):
            if storage_ref.expired():
                del self[key]
            else:
                live += 1
        self.limit = max(128, live * 2)


# mapping from handles to StorageWeakRef objects
shared_cache = SharedCache()


def rebuild_event(device, handle):
    return torch.cuda.Event.from_ipc_handle(device, handle)


def reduce_event(event):
    handle = event.ipc_handle()
    return (rebuild_event, (event.device, handle))


def rebuild_tensor(cls, storage, metadata):
    storage_offset, size, stride, requires_grad = metadata
    t = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
    if cls == torch.nn.parameter.Parameter:
        # we have to pass requires_grad into constructor, rather than set it as an
        # attribute later, because it's an important check for Integer Tensors to
        # have requires_grad=False (or else they raise an error)
        t = torch.nn.parameter.Parameter(t, requires_grad=requires_grad)
    else:
        t.requires_grad = requires_grad
    return t


def rebuild_meta_tensor(
    tensor_cls,
    tensor_size,
    tensor_stride,
    tensor_offset,
    dtype,
    storage_size_bytes,
    requires_grad,
):
    untyped_storage = torch.UntypedStorage(storage_size_bytes, device="meta")

    typed_storage = torch.TypedStorage(
        wrap_storage=untyped_storage, dtype=dtype, _internal=True
    )

    t = torch._utils._rebuild_tensor(
        typed_storage,
        tensor_offset,
        tensor_size,
        tensor_stride,
    )

    if tensor_cls == torch.nn.parameter.Parameter:
        # It is crucial for integer tensors to receive
        # the requires_grad=False as an argument in the constructor
        t = torch.nn.parameter.Parameter(t, requires_grad=requires_grad)
    else:
        t.requires_grad = requires_grad

    return t


def rebuild_cuda_tensor(
    tensor_cls,
    tensor_size,
    tensor_stride,
    tensor_offset,
    storage_cls,
    dtype,
    storage_device,
    storage_handle,
    storage_size_bytes,
    storage_offset_bytes,
    requires_grad,
    ref_counter_handle,
    ref_counter_offset,
    event_handle,
    event_sync_required,
):
    # If storage_handle is None, storage points to nullptr.
    if storage_handle is None or storage_size_bytes == 0:
        storage = storage_cls(0, dtype=dtype, device=storage_device, _internal=True)
    else:
        storage = storage_from_cache(
            storage_cls, (storage_handle, storage_offset_bytes)
        )
        if storage is None:
            torch.cuda._lazy_init()
            storage = storage_cls._new_shared_cuda(
                storage_device,
                storage_handle,
                storage_size_bytes,
                storage_offset_bytes,
                ref_counter_handle,
                ref_counter_offset,
                event_handle,
                event_sync_required,
            )
            shared_cache[(storage_handle, storage_offset_bytes)] = StorageWeakRef(
                storage
            )
        else:
            # We already ref counting this Storage, but producer needs new ref-counters to be released.
            storage_cls._release_ipc_counter(
                ref_counter_handle, ref_counter_offset, device=storage_device
            )

    _storage = (
        storage
        if isinstance(storage, torch.UntypedStorage)
        else storage._untyped_storage
    )

    t = torch._utils._rebuild_tensor(
        torch.storage.TypedStorage(wrap_storage=_storage, dtype=dtype, _internal=True),
        tensor_offset,
        tensor_size,
        tensor_stride,
    )

    if tensor_cls == torch.nn.parameter.Parameter:
        # It is crucial for integer tensors to receive
        # the requires_grad=False as an argument in the constructor
        t = torch.nn.parameter.Parameter(t, requires_grad=requires_grad)
    else:
        t.requires_grad = requires_grad

    return t


def reduce_tensor(tensor):
    if tensor.requires_grad and not tensor.is_leaf:
        raise RuntimeError(
            "Cowardly refusing to serialize non-leaf tensor which requires_grad, "
            "since autograd does not support crossing process boundaries.  "
            "If you just want to transfer the data, call detach() on the tensor "
            "before serializing (e.g., putting it on the queue)."
        )

    torch.utils.hooks.warn_if_has_hooks(tensor)

    # Note [CUDA IPC and the caching allocator]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # When you send a CUDA tensor over IPC, you might expect that you will
    # get out the same storage from the other end.  However, the CUDA caching
    # allocator makes it difficult to preserve this invariant.  Consider
    # the following situation: a tensor of size 0x100 points to offset 0x20 of
    # a storage at 0xA100 of size 0x100.  (For simplicity, all of these
    # sizes are given in bytes).  HOWEVER, with the caching allocator, this storage
    # might be part of a larger cudaMalloc allocation 0xA000 of size 0x4000.
    #
    # When we want to send this CUDA tensor over IPC, we must send the
    # *entire* cudaMalloc allocation, i.e., the 0xA000 region, not just
    # the storage 0xA100 (because that is what CUDA supports).  So, on the
    # other end, there simply isn't any way to say, "Wait, you gave me
    # a bigger region (0xA000) than the one I wanted (0xA100)".
    #
    # OK, so if you sent the cudaMalloc allocation, can you just wrap that up as
    # one storage itself? No, because this cudaMalloc allocation might contain
    # storages of mixed types: float, bytes, double... If you make the entire
    # allocation a single storage of a type A, we'll hit an error when constructing
    # a tensor of type B on the storage.
    #
    # cudaIpcMemHandle is an identifier to access the sender cudaMalloc allocation on the
    # receiver side. However, cudaIpcMemHandles from each device in a given process may
    # only be opened by one context per device per other process.
    # If we open and close a memory handle multiples times in a process, CUDA is allowed
    # to give it a different address; similarly, once we close the memory, we're not
    # allowed to access it(and the storage/tensor built on top of it), even if it is
    # still live in the original process. As we cannot make a cudaMalloc allocation
    # to a single storage in one go, this requires us to cache the device pointer for
    # each cudaIpcMemHandle on C++ side to reconstruct types of storages, while keep
    # the old ones alives.
    # See [https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html]
    #
    # This is fine, because all we need to do is to save our position in the allocation,
    # and reconstruct storage and tensor from it.
    # 0xA000 ->  -------CUDA Allocation------
    #           |                            |
    #           |                            |
    #           |                            |
    #           |                            |
    # 0xA100 ->  --------storage1 begin------
    #           |                            |
    # 0xA120 ->  --------tensor1 begin ------
    #           |                            |
    #           |                            |
    #           |                            |
    #           |                            |
    #           |                            |
    # 0xA160 ->  --------tensor1 end---------
    #           |                            |
    #           |                            |
    #           |                            |
    # 0xA200 ->  --------storage1 end--------
    #           |                            |
    # 0xE000 ->  --------CUDA allocation-----
    #
    # To send tensor1, the following info are required from sender to receiver for
    # storage reconstruction.
    #   1. cudaIpcMemHandle of 0xA000(which can be mapped to a basePtr in receiver process).
    #      basePtr may not be exactly 0xA000 since it's a different process.
    #   2. offset(0xA100) of storage1 in the CUDA allocation.
    #   3. size of storage1(0x100).
    #
    # On receiver side:
    #   1. Get the devPtr of the MemHandle to access the memory, reconstruct a storage
    #      of the same type using (basePtr, offset, size).
    #   2. we can reconstruct the tensor on top of the reconstructed storage
    #   Tensor(size=0x040, offset=0x020, storage=Storage(data=basePtr+0xA100, size=0x0100))
    #
    # This strategy has a few implications:
    #
    # 1. When we serialize a CUDA tensor for IPC, we cannot do it all in one
    #    go (non-compositionally), and this requires to have a global map
    #    memHandle -> devPtr for each process.
    #
    # 2. We MUST NOT let the new IPC tensor be resizable.  Originally, a resize
    #    of the storage beyond 0x100 would merely have caused us to do a
    #    reallocation.  You don't really want to do this, but if you did,
    #    all that would happen is that you would lose IPC sharing.  But if
    #    you do this in the new world, we will happily let you write out of
    #    bounds of your "allocation", clobbering unrelated data in the cached
    #    allocator block.  BAD!
    #
    # By the way, in old versions of PyTorch, we supported this situation
    # natively using a "storage view", which permitted multiple storages to be
    # views on each other.  But this was the *only* use of storage views, so we
    # eliminated it so that we could just use tensor views to implement the same
    # thing.
    #

    # TODO: Handle distinguishing between subclass and non-subclass versions of NT better
    # https://github.com/pytorch/pytorch/issues/110543
    from torch.nested._internal.nested_tensor import NestedTensor

    if tensor.is_nested and not isinstance(tensor, NestedTensor):
        return reduce_nested_tensor(tensor)

    if tensor.layout in {
        torch.sparse_coo,
        torch.sparse_csr,
        torch.sparse_bsr,
        torch.sparse_csc,
        torch.sparse_bsc,
    }:
        return reduce_sparse_tensor(tensor)

    storage = tensor._typed_storage()

    if storage._untyped_storage.device.type == "cuda":
        (
            device,
            handle,
            storage_size_bytes,
            storage_offset_bytes,
            ref_counter_handle,
            ref_counter_offset,
            event_handle,
            event_sync_required,
        ) = storage._share_cuda_()
        tensor_offset = tensor.storage_offset()
        shared_cache[handle] = StorageWeakRef(storage)
        # _backward_hooks purposely omitted here, see
        # Note [Don't serialize hooks]
        return (
            rebuild_cuda_tensor,
            (
                type(tensor),
                tensor.size(),
                tensor.stride(),
                tensor_offset,  # tensor offset in its storage
                type(storage),
                tensor.dtype,
                device,
                handle,  # identifier which CUDA allocation is the storage in.
                storage_size_bytes,  # size(in bytes) of the storage
                storage_offset_bytes,  # offset(in bytes) of the storage in the CUDA allocation
                tensor.requires_grad,
                ref_counter_handle,
                ref_counter_offset,
                event_handle,
                event_sync_required,
            ),
        )
    elif storage._untyped_storage.device.type == "meta":
        return (
            rebuild_meta_tensor,
            (
                type(tensor),
                tensor.size(),
                tensor.stride(),
                tensor.storage_offset(),
                tensor.dtype,
                tensor.untyped_storage().size(),
                tensor.requires_grad,
            ),
        )

    # _backward_hooks purposely omitted here, see Note [Don't serialize hooks]
    metadata = (
        tensor.storage_offset(),
        tensor.size(),
        tensor.stride(),
        tensor.requires_grad,
    )
    return (rebuild_tensor, (type(tensor), storage, metadata))


def rebuild_nested_tensor(
    rebuild_buffer_func,
    rebuild_buffer_args,
    rebuild_sizes_func,
    rebuild_sizes_args,
    rebuild_strides_func,
    rebuild_strides_args,
    rebuild_offsets_func,
    rebuild_offsets_args,
):
    buffer = rebuild_buffer_func(*rebuild_buffer_args)
    sizes = rebuild_sizes_func(*rebuild_sizes_args)
    strides = rebuild_strides_func(*rebuild_strides_args)
    offsets = rebuild_offsets_func(*rebuild_offsets_args)
    return torch._nested_view_from_buffer_copy(buffer, sizes, strides, offsets)


def reduce_nested_tensor(nt):
    rebuild_buffer_func, rebuild_buffer_args = reduce_tensor(nt.values())
    rebuild_sizes_func, rebuild_sizes_args = reduce_tensor(nt._nested_tensor_size())
    rebuild_strides_func, rebuild_strides_args = reduce_tensor(
        nt._nested_tensor_strides()
    )
    rebuild_offsets_func, rebuild_offsets_args = reduce_tensor(
        nt._nested_tensor_storage_offsets()
    )

    return (
        rebuild_nested_tensor,
        (
            rebuild_buffer_func,
            rebuild_buffer_args,
            rebuild_sizes_func,
            rebuild_sizes_args,
            rebuild_strides_func,
            rebuild_strides_args,
            rebuild_offsets_func,
            rebuild_offsets_args,
        ),
    )


def rebuild_sparse_coo_tensor(
    rebuild_indices_func,
    rebuild_indices_args,
    rebuild_values_func,
    rebuild_values_args,
    shape,
    is_coalesced,
):
    indices = rebuild_indices_func(*rebuild_indices_args)
    values = rebuild_values_func(*rebuild_values_args)
    return torch.sparse_coo_tensor(indices, values, shape, is_coalesced=is_coalesced)


def rebuild_sparse_compressed_tensor(
    rebuild_compressed_indices_func,
    rebuild_compressed_indices_args,
    rebuild_plain_indices_func,
    rebuild_plain_indices_args,
    rebuild_values_func,
    rebuild_values_args,
    shape,
    layout,
):
    compressed_indices = rebuild_compressed_indices_func(
        *rebuild_compressed_indices_args
    )
    plain_indices = rebuild_plain_indices_func(*rebuild_plain_indices_args)
    values = rebuild_values_func(*rebuild_values_args)
    return torch.sparse_compressed_tensor(
        compressed_indices, plain_indices, values, shape, layout=layout
    )


def reduce_sparse_tensor(sparse):
    if sparse.layout is torch.sparse_coo:
        rebuild_indices_func, rebuild_indices_args = reduce_tensor(sparse._indices())
        rebuild_values_func, rebuild_values_args = reduce_tensor(sparse._values())
        return (
            rebuild_sparse_coo_tensor,
            (
                rebuild_indices_func,
                rebuild_indices_args,
                rebuild_values_func,
                rebuild_values_args,
                sparse.shape,
                sparse.is_coalesced(),
            ),
        )
    else:
        if sparse.layout in {torch.sparse_csr, torch.sparse_bsr}:
            compressed_indices = sparse.crow_indices()
            plain_indices = sparse.col_indices()
        elif sparse.layout in {torch.sparse_csc, torch.sparse_bsc}:
            compressed_indices = sparse.ccol_indices()
            plain_indices = sparse.row_indices()
        else:
            raise NotImplementedError(sparse.layout)
        (
            rebuild_compressed_indices_func,
            rebuild_compressed_indices_args,
        ) = reduce_tensor(compressed_indices)
        rebuild_plain_indices_func, rebuild_plain_indices_args = reduce_tensor(
            plain_indices
        )
        rebuild_values_func, rebuild_values_args = reduce_tensor(sparse.values())
        return (
            rebuild_sparse_compressed_tensor,
            (
                rebuild_compressed_indices_func,
                rebuild_compressed_indices_args,
                rebuild_plain_indices_func,
                rebuild_plain_indices_args,
                rebuild_values_func,
                rebuild_values_args,
                sparse.shape,
                sparse.layout,
            ),
        )


def fd_id(fd):
    # Returns a tuple which uniquely identifies a file descriptor. In Mac OS,
    # this doesn't work with shared memory handles, which is why we don't
    # support the "file_descriptor" sharing method on that platform.
    stat = os.fstat(fd)
    return (stat.st_ino, stat.st_dev)


def storage_from_cache(cls, key):
    storage_ref = shared_cache.get(key)
    if storage_ref is None:
        return None
    return torch.UntypedStorage._new_with_weak_ptr(storage_ref.cdata)


def rebuild_storage_fd(cls, df, size):
    fd = df.detach()
    try:
        storage = storage_from_cache(cls, fd_id(fd))
        if storage is not None:
            return storage
        storage = cls._new_shared_fd_cpu(fd, size)
        shared_cache[fd_id(fd)] = StorageWeakRef(storage)
        return storage
    finally:
        os.close(fd)


def rebuild_storage_filename(cls, manager, handle, size, dtype=None):
    storage: torch.TypedStorage | torch.UntypedStorage = storage_from_cache(cls, handle)
    if storage is not None:
        return storage._shared_decref()
    if dtype is None:
        storage = torch.UntypedStorage._new_shared_filename_cpu(manager, handle, size)
    else:
        byte_size = size * torch._utils._element_size(dtype)
        untyped_storage: torch.UntypedStorage = (
            torch.UntypedStorage._new_shared_filename_cpu(manager, handle, byte_size)
        )
        storage = torch.TypedStorage(
            wrap_storage=untyped_storage, dtype=dtype, _internal=True
        )
    shared_cache[handle] = StorageWeakRef(storage)
    return storage._shared_decref()


def rebuild_storage_empty(cls):
    return cls()


def rebuild_typed_storage(storage, dtype):
    return torch.storage.TypedStorage(wrap_storage=storage, dtype=dtype, _internal=True)


# Use for torch.storage.TypedStorage
def reduce_typed_storage(storage):
    return (rebuild_typed_storage, (storage._untyped_storage, storage.dtype))


def rebuild_typed_storage_child(storage, storage_type):
    return storage_type(wrap_storage=storage, _internal=True)


# Use for child classes of torch.storage.TypedStorage, like torch.FloatStorage
def reduce_typed_storage_child(storage):
    return (rebuild_typed_storage_child, (storage._untyped_storage, type(storage)))


def reduce_storage(storage):
    from . import get_sharing_strategy

    if storage.is_cuda:
        raise RuntimeError(
            "Cannot pickle CUDA storage; try pickling a CUDA tensor instead"
        )
    elif storage.device.type == "meta":
        raise RuntimeError(
            "Cannot pickle meta storage; try pickling a meta tensor instead"
        )
    elif get_sharing_strategy() == "file_system":
        metadata = storage._share_filename_cpu_()
        cache_key = metadata[1]
        rebuild = rebuild_storage_filename
        if isinstance(storage, torch.TypedStorage):
            metadata += (storage.dtype,)
        storage._shared_incref()
    elif storage.size() == 0:
        # This is special cased because Empty tensors
        # (with size 0) cannot be mmapped.
        return (rebuild_storage_empty, (type(storage),))
    else:
        fd, size = storage._share_fd_cpu_()
        df = multiprocessing.reduction.DupFd(fd)
        cache_key = fd_id(fd)
        metadata = (df, size)
        rebuild = rebuild_storage_fd  # type: ignore[assignment]

    shared_cache[cache_key] = StorageWeakRef(storage)
    return (rebuild, (type(storage),) + metadata)


def init_reductions():
    reduction.register(torch.cuda.Event, reduce_event)

    for t in torch._storage_classes:
        if t.__name__ == "UntypedStorage":
            reduction.register(t, reduce_storage)
        else:
            reduction.register(t, reduce_typed_storage_child)

    reduction.register(torch.storage.TypedStorage, reduce_typed_storage)

    for t in torch._tensor_classes:
        reduction.register(t, reduce_tensor)

    # TODO: Maybe this should be in tensor_classes? :)
    reduction.register(torch.Tensor, reduce_tensor)

    from torch.nn.parameter import Parameter

    reduction.register(Parameter, reduce_tensor)
<|endoftext|># mypy: allow-untyped-defs
import sys


__all__ = ["register_after_fork"]

if sys.platform == "win32":
    import multiprocessing.util as _util

    def _register(func):
        def wrapper(arg):
            func()

        _util.register_after_fork(_register, wrapper)

else:
    import os

    def _register(func):
        os.register_at_fork(after_in_child=func)


def register_after_fork(func):
    """Register a callable to be executed in the child process after a fork.

    Note:
        In python < 3.7 this will only work with processes created using the
        ``multiprocessing`` module. In python >= 3.7 it also works with
        ``os.fork()``.

    Args:
        func (function): Function taking no arguments to be called in the child after fork

    """
    _register(func)
<|endoftext|># mypy: allow-untyped-defs
"""torch.multiprocessing is a wrapper around the native :mod:`multiprocessing` module.

It registers custom reducers, that use shared memory to provide shared
views on the same data in different processes. Once the tensor/storage is moved
to shared_memory (see :func:`~torch.Tensor.share_memory_`), it will be possible
to send it to other processes without making any copies.

The API is 100% compatible with the original module - it's enough to change
``import multiprocessing`` to ``import torch.multiprocessing`` to have all the
tensors sent through the queues or shared via other mechanisms, moved to shared
memory.

Because of the similarity of APIs we do not document most of this package
contents, and we recommend referring to very good docs of the original module.
"""

import multiprocessing
import sys

import torch

from .reductions import init_reductions


__all__ = ["set_sharing_strategy", "get_sharing_strategy", "get_all_sharing_strategies"]


from multiprocessing import *  # noqa: F403


__all__ += multiprocessing.__all__


# This call adds a Linux specific prctl(2) wrapper function to this module.
# See https://github.com/pytorch/pytorch/pull/14391 for more information.
torch._C._multiprocessing_init()


"""Add helper function to spawn N processes and wait for completion of any of
them."""
from .spawn import (
    ENV_VAR_PARALLEL_START,
    ProcessContext,
    ProcessExitedException,
    ProcessRaisedException,
    spawn,
    SpawnContext,
    start_processes,
)


if sys.platform == "darwin" or sys.platform == "win32":
    _sharing_strategy = "file_system"
    _all_sharing_strategies = {"file_system"}
else:
    _sharing_strategy = "file_descriptor"
    _all_sharing_strategies = {"file_descriptor", "file_system"}


def set_sharing_strategy(new_strategy):
    """Set the strategy for sharing CPU tensors.

    Args:
        new_strategy (str): Name of the selected strategy. Should be one of
            the values returned by :func:`get_all_sharing_strategies()`.
    """
    global _sharing_strategy
    if new_strategy not in _all_sharing_strategies:
        raise AssertionError(
            f"invalid sharing strategy {new_strategy!r}, "
            f"expected one of {_all_sharing_strategies}"
        )
    _sharing_strategy = new_strategy


def get_sharing_strategy():
    """Return the current strategy for sharing CPU tensors."""
    return _sharing_strategy


def get_all_sharing_strategies():
    """Return a set of sharing strategies supported on a current system."""
    return _all_sharing_strategies


def _set_thread_name(name: str) -> None:
    """Set the name of the current thread.

    Args:
        name (str): Name of the current thread.
    """
    torch._C._set_thread_name(name)


def _get_thread_name() -> str:
    """Get the name of the current thread.

    Returns:
        str: Name of the current thread.
    """
    return torch._C._get_thread_name()


init_reductions()

# Leak ResourceTracker at exit for Python-3.12 on MacOS
# See https://github.com/pytorch/pytorch/issues/153050 and
# https://github.com/python/cpython/issues/88887 for more details
from multiprocessing.resource_tracker import ResourceTracker as _RT


if (
    sys.platform == "darwin"
    and sys.version_info >= (3, 12, 2)
    and hasattr(_RT, "__del__")
):
    import atexit

    def _leak_RT_at_exit():
        def _noop(x):
            pass

        _RT.__del__ = _noop  # type: ignore[attr-defined]

    atexit.register(_leak_RT_at_exit)
<|endoftext|># mypy: allow-untyped-defs
import io
import multiprocessing.queues
import pickle
from multiprocessing.reduction import ForkingPickler


class ConnectionWrapper:
    """Proxy class for _multiprocessing.Connection which uses ForkingPickler for object serialization."""

    def __init__(self, conn):
        self.conn = conn

    def send(self, obj):
        buf = io.BytesIO()
        ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(obj)
        self.send_bytes(buf.getvalue())

    def recv(self):
        buf = self.recv_bytes()
        return pickle.loads(buf)

    def __getattr__(self, name):
        if "conn" in self.__dict__:
            return getattr(self.conn, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute 'conn'")


class Queue(multiprocessing.queues.Queue):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reader: ConnectionWrapper = ConnectionWrapper(self._reader)
        self._writer: ConnectionWrapper = ConnectionWrapper(self._writer)
        self._send = self._writer.send
        self._recv = self._reader.recv


class SimpleQueue(multiprocessing.queues.SimpleQueue):
    def _make_methods(self):
        if not isinstance(self._reader, ConnectionWrapper):
            self._reader: ConnectionWrapper = ConnectionWrapper(self._reader)
            self._writer: ConnectionWrapper = ConnectionWrapper(self._writer)
        super()._make_methods()  # type: ignore[misc]
<|endoftext|># mypy: allow-untyped-defs
import logging
import multiprocessing
import multiprocessing.connection
import os
import pickle
import signal
import sys
import tempfile
import time
import warnings
from concurrent.futures import as_completed, ThreadPoolExecutor

from . import _prctl_pr_set_pdeathsig  # type: ignore[attr-defined]


ENV_VAR_PARALLEL_START = "TORCH_MP_PARALLEL_START"

log = logging.getLogger(__name__)

__all__ = [
    "ProcessContext",
    "ProcessException",
    "ProcessExitedException",
    "ProcessRaisedException",
    "spawn",
    "SpawnContext",
    "start_processes",
]


class ProcessException(Exception):
    __slots__ = ["error_index", "error_pid"]

    def __init__(self, msg: str, error_index: int, error_pid: int):
        super().__init__(msg)
        self.msg = msg
        self.error_index = error_index
        self.error_pid = error_pid

    def __reduce__(self):
        return type(self), (self.msg, self.error_index, self.error_pid)


class ProcessRaisedException(ProcessException):
    """Exception raised when a process failed due to an exception raised by the code."""


class ProcessExitedException(ProcessException):
    """Exception raised when a process failed due to signal or exited with a specific code."""

    __slots__ = ["exit_code"]

    def __init__(
        self,
        msg: str,
        error_index: int,
        error_pid: int,
        exit_code: int,
        signal_name: str | None = None,
    ):
        super().__init__(msg, error_index, error_pid)
        self.exit_code = exit_code
        self.signal_name = signal_name

    def __reduce__(self):
        return (
            type(self),
            (
                self.msg,
                self.error_index,
                self.error_pid,
                self.exit_code,
                self.signal_name,
            ),
        )


def _wrap(fn, i, args, error_file):
    # prctl(2) is a Linux specific system call.
    # On other systems the following function call has no effect.
    # This is set to ensure that non-daemonic child processes can
    # terminate if their parent terminates before they do.
    _prctl_pr_set_pdeathsig(signal.SIGINT)

    try:
        fn(i, *args)
    except KeyboardInterrupt:
        pass  # SIGINT; Killed by parent, do nothing
    except Exception:
        # Propagate exception to parent process, keeping original traceback
        import traceback

        with open(error_file, "wb") as fh:
            pickle.dump(traceback.format_exc(), fh)
        sys.exit(1)


class ProcessContext:
    def __init__(self, processes, error_files):
        self.error_files = error_files
        self.processes = processes
        self.sentinels = {
            process.sentinel: index for index, process in enumerate(processes)
        }

    def pids(self):
        return [int(process.pid) for process in self.processes]

    def _join_procs_with_timeout(self, timeout: float):
        """Attempt to join all processes with a shared timeout."""
        end = time.monotonic() + timeout
        for process in self.processes:
            # pyrefly: ignore [no-matching-overload]
            time_to_wait = max(0, end - time.monotonic())
            process.join(time_to_wait)

    def join(self, timeout: float | None = None, grace_period: float | None = None):
        r"""Join one or more processes within spawn context.

        Attempt to join one or more processes in this spawn context.
        If one of them exited with a non-zero exit status, this function
        kills the remaining processes (optionally with a grace period)
        and raises an exception with the cause of the first process exiting.

        Returns ``True`` if all processes have been joined successfully,
        ``False`` if there are more processes that need to be joined.

        Args:
            timeout (float): Wait this long (in seconds) before giving up on waiting.
            grace_period (float): When any processes fail, wait this long (in seconds)
                for others to shutdown gracefully before terminating them. If they
                still don't exit, wait another grace period before killing them.
        """
        # Ensure this function can be called even when we're done.
        if len(self.sentinels) == 0:
            return True

        # Wait for any process to fail or all of them to succeed.
        ready = multiprocessing.connection.wait(
            self.sentinels.keys(),
            timeout=timeout,
        )

        error_index = None
        for sentinel in ready:
            index = self.sentinels.pop(sentinel)
            process = self.processes[index]
            process.join()
            if process.exitcode != 0:
                error_index = index
                break

        # Return if there was no error.
        if error_index is None:
            # Return whether or not all processes have been joined.
            return len(self.sentinels) == 0
        # An error occurred. Clean-up all processes before returning.
        # First, allow a grace period for processes to shutdown themselves.
        if grace_period is not None:
            self._join_procs_with_timeout(grace_period)
        # Then, terminate processes that are still alive. Try SIGTERM first.
        for process in self.processes:
            if process.is_alive():
                log.warning("Terminating process %s via signal SIGTERM", process.pid)
                process.terminate()

        # Try SIGKILL if the process isn't going down after another grace_period.
        # The reason is related to python signal handling is limited
        # to main thread and if that is in c/c++ land and stuck it won't
        # to handle it. We have seen processes getting stuck not handling
        # SIGTERM for the above reason.
        self._join_procs_with_timeout(30 if grace_period is None else grace_period)
        for process in self.processes:
            if process.is_alive():
                log.warning(
                    "Unable to shutdown process %s via SIGTERM , forcefully exiting via SIGKILL",
                    process.pid,
                )
                process.kill()
            process.join()

        # The file will only be created if the process crashed.
        failed_process = self.processes[error_index]
        if not os.access(self.error_files[error_index], os.R_OK):
            exitcode = self.processes[error_index].exitcode
            if exitcode < 0:
                try:
                    name = signal.Signals(-exitcode).name
                except ValueError:
                    name = f"<Unknown signal {-exitcode}>"
                raise ProcessExitedException(
                    f"process {error_index:d} terminated with signal {name}",
                    error_index=error_index,
                    error_pid=failed_process.pid,
                    exit_code=exitcode,
                    signal_name=name,
                )
            else:
                raise ProcessExitedException(
                    f"process {error_index:d} terminated with exit code {exitcode:d}",
                    error_index=error_index,
                    error_pid=failed_process.pid,
                    exit_code=exitcode,
                )

        with open(self.error_files[error_index], "rb") as fh:
            original_trace = pickle.load(fh)
        msg = f"\n\n-- Process {error_index:d} terminated with the following error:\n"
        msg += original_trace
        raise ProcessRaisedException(msg, error_index, failed_process.pid)


class SpawnContext(ProcessContext):
    def __init__(self, processes, error_files):
        warnings.warn(
            "SpawnContext is renamed to ProcessContext since 1.4 release.", stacklevel=2
        )
        super().__init__(processes, error_files)


# Note: [start_processes]
# mp.start_processes handles both start_method='spawn' and 'fork'. It's supposed to be a
# more generalized API than mp.spawn. Currently we only document mp.spawn as it's the
# CUDA compatible start_method. However, in environments like IPython notebooks, 'fork'
# works better than 'spawn'. Every helper function we created for mp.spawn is indeed
# general enough, and backends like XLA can reuse them in Colab notebooks as well.
# Currently we only add this API first, we can consider adding it to documentation as
# needed in the future.
def start_processes(
    fn,
    args=(),
    nprocs=1,
    join=True,
    daemon=False,
    start_method="spawn",
):
    # To speed up performance in certain cases (see https://github.com/pytorch/pytorch/issues/133010),
    # this func will start processes in parallel if start_method is 'forkserver'.
    # Please opt in to this perf optimization by setting env var (TORCH_MP_PARALLEL_START) to 1.
    # todo: investigate why spawn does not work with threadpool and raises SIGINT
    if (
        start_method == "forkserver"
        and os.environ.get(ENV_VAR_PARALLEL_START, "0") == "1"
    ):
        log.info("Starting processes in parallel.")
        start_parallel = True
    else:
        # Set env var TORCH_MP_PARALLEL_START to 0 to disable parallel start
        start_parallel = False

    mp = multiprocessing.get_context(start_method)
    error_files = [None] * nprocs
    processes = [None] * nprocs

    def start_process(i):
        # Each process is assigned a file to write tracebacks to.  We
        # use the file being non-empty to indicate an exception
        # occurred (vs an expected shutdown).  Note: this previously
        # used a multiprocessing.Queue but that can be prone to
        # deadlocks, so we went with a simpler solution for a one-shot
        # message between processes.
        tf = tempfile.NamedTemporaryFile(  # noqa: SIM115
            prefix="pytorch-errorfile-", suffix=".pickle", delete=False
        )
        tf.close()
        os.unlink(tf.name)

        process = mp.Process(  # pyrefly: ignore  # missing-attribute
            target=_wrap,
            args=(fn, i, args, tf.name),
            daemon=daemon,
        )

        process.start()
        return i, process, tf.name

    if not start_parallel:
        for i in range(nprocs):
            idx, process, tf_name = start_process(i)
            error_files[idx] = tf_name
            processes[idx] = process
    else:
        with ThreadPoolExecutor(max_workers=nprocs) as executor:
            futures = [executor.submit(start_process, i) for i in range(nprocs)]
            for fut in as_completed(futures):
                idx, process, tf_name = fut.result()
                # idx and process rank needs to be the same.
                error_files[idx] = tf_name
                processes[idx] = process
    context = ProcessContext(processes, error_files)
    if not join:
        return context

    # Loop on join until it returns True or raises an exception.
    while not context.join():
        pass


def spawn(fn, args=(), nprocs=1, join=True, daemon=False, start_method="spawn"):
    r"""Spawns ``nprocs`` processes that run ``fn`` with ``args``.

    If one of the processes exits with a non-zero exit status, the
    remaining processes are killed and an exception is raised with the
    cause of termination. In the case an exception was caught in the
    child process, it is forwarded and its traceback is included in
    the exception raised in the parent process.

    Args:
        fn (function): Function is called as the entrypoint of the
            spawned process. This function must be defined at the top
            level of a module so it can be pickled and spawned. This
            is a requirement imposed by multiprocessing.

            The function is called as ``fn(i, *args)``, where ``i`` is
            the process index and ``args`` is the passed through tuple
            of arguments.

        args (tuple): Arguments passed to ``fn``.
        nprocs (int): Number of processes to spawn.
        join (bool): Perform a blocking join on all processes.
        daemon (bool): The spawned processes' daemon flag. If set to True,
                       daemonic processes will be created.
        start_method (str): (deprecated) this method will always use ``spawn``
                               as the start method. To use a different start method
                               use ``start_processes()``.

    Returns:
        None if ``join`` is ``True``,
        :class:`~ProcessContext` if ``join`` is ``False``

    """
    if start_method != "spawn":
        msg = (
            f"This method only supports start_method=spawn (got: {start_method}).\n"
            "To use a different start_method use:\n\t\t"
            " torch.multiprocessing.start_processes(...)"
        )
        warnings.warn(msg, FutureWarning, stacklevel=2)
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
<|endoftext|>import multiprocessing.pool
import multiprocessing.util as util

from .queue import SimpleQueue


def clean_worker(*args, **kwargs):
    import gc

    multiprocessing.pool.worker(*args, **kwargs)
    # Regular multiprocessing workers don't fully clean up after themselves,
    # so we have to explicitly trigger garbage collection to make sure that all
    # destructors are called...
    gc.collect()


class Pool(multiprocessing.pool.Pool):
    """Pool implementation which uses our version of SimpleQueue.

    This lets us pass tensors in shared memory across processes instead of
    serializing the underlying data.
    """

    def _setup_queues(self):
        self._inqueue = SimpleQueue()
        self._outqueue = SimpleQueue()
        self._quick_put = self._inqueue._writer.send
        self._quick_get = self._outqueue._reader.recv

    def _repopulate_pool(self):
        """Increase the number of pool processes to the specified number.

        Bring the number of pool processes up to the specified number, for use after
        reaping workers which have exited.
        """
        for _ in range(self._processes - len(self._pool)):
            # changed worker -> clean_worker
            args = (
                self._inqueue,
                self._outqueue,
                self._initializer,
                self._initargs,
                self._maxtasksperchild,
            )
            if hasattr(self, "_wrap_exception"):
                args += (self._wrap_exception,)
            w = self.Process(target=clean_worker, args=args)
            self._pool.append(w)
            w.name = w.name.replace("Process", "PoolWorker")
            w.daemon = True
            w.start()
            util.debug("added worker")
<|endoftext|>from __future__ import annotations

import dataclasses
import operator
import struct
from typing import Any, TYPE_CHECKING

import sympy

import torch
import torch.fx as fx
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten


if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    from torch._ops import OpOverloadPacket
    from torch.utils._pytree import TreeSpec

aten = torch.ops.aten


@dataclasses.dataclass(frozen=True)
class _ScalarKey:
    tag: str
    bits: bytes


def _normalize_cse_arg(val: Any) -> Any:
    # Python float hash/eq is not value-identity: nan != nan (hash(nan) is
    # id-based) while -0.0 == 0.0 and hashes equal. Key by bit pattern instead.
    if type(val) is float:
        return _ScalarKey("float", struct.pack(">d", val))
    if type(val) is complex:
        return _ScalarKey("complex", struct.pack(">dd", val.real, val.imag))
    return val


def get_aten_target(node: fx.Node) -> OpOverloadPacket | Callable[..., Any] | str:
    if hasattr(node.target, "overloadpacket"):
        return node.target.overloadpacket
    return node.target


rand_ops = [
    aten.dropout,
    aten._fused_dropout,
    aten._standard_gamma,
    aten.bernoulli,
    aten.multinomial,
    aten.native_dropout,
    aten.normal,
    aten.poisson,
    aten.binomial,
    aten.rrelu,
    aten.rand_like,
    aten.rand,
    aten.randint,
    aten.randn,
    aten.randperm,
]


# return a new copy of torch.fx.graph.Graph with CSE applied to the input graph
def fx_graph_cse(
    fx_g: torch.fx.graph.Graph,
    extra_node_key: Callable[[fx.Node], Hashable] | None = None,
) -> fx.Graph:
    new_graph = fx.Graph()
    env: dict[
        fx.Node, fx.Node
    ] = {}  # map from node in the old graph to node in the new graph
    hash_env: dict[
        tuple[Any, Hashable | None, int], fx.Node
    ] = {}  # map from hash to a node in the new graph
    token_map: dict[
        tuple[Any, Hashable | None, int], dict[str, Any]
    ] = {}  # map from hash to token

    from torch._inductor.pattern_matcher import (
        compute_mutation_region_ids,
        same_mutation_regions,
    )

    compute_mutation_region_ids(fx_g)  # type: ignore[arg-type]

    # Make a set of separate storages returned from the output, which will be preserved
    # when pruning.  This prevents us from deduplicating returned tensors which have
    # experienced identical operations, but are separate data structures in eager mode.
    output_node: fx.Node = list(fx_g.nodes)[-1]
    if output_node.op != "output":
        raise AssertionError(
            f"expected output_node.op to be 'output', got '{output_node.op}'"
        )

    def checkable_node(node: fx.Node) -> bool:
        """We can evaluate only nodes that represent tensors with defined storage."""
        if "val" not in node.meta or not isinstance(node.meta["val"], torch.Tensor):
            return False

        try:
            node.meta["val"].untyped_storage()
        except NotImplementedError:
            return False

        return True

    def custom_context_key(node: fx.Node) -> tuple[int, int | None, int | None]:
        custom = node.meta.get("custom", {})
        return (
            custom.get("stream", 0),
            custom.get("mempool"),
            custom.get("mempool_device"),
        )

    output_storages = {
        StorageWeakRef(n.meta["val"].untyped_storage())
        for n in output_node.all_input_nodes
        if checkable_node(n)
    }
    nodes_that_alias_outputs = {
        n
        for n in fx_g.nodes
        if checkable_node(n)
        and StorageWeakRef(n.meta["val"].untyped_storage()) in output_storages
    }

    # A node used as the mutable base of a functional-wrapper op is a distinct
    # mutable buffer. Two identical factory ops feeding distinct bases (e.g. the
    # forward and backward ``aten.full`` from a pair of ``ones_like`` calls) must
    # not be CSE-merged, or the later mutation aliases a buffer that still has a
    # live downstream view and reinplacing fails (mirrors the aten.empty
    # exclusion below). See pytorch/pytorch#170160.
    from torch._higher_order_ops.auto_functionalize import get_mutable_args
    from torch._higher_order_ops.triton_kernel_wrap import (
        triton_kernel_wrapper_functional,
    )

    def _add_base(node: Any, bases: set[fx.Node]) -> None:
        for b in node if isinstance(node, (list, tuple)) else (node,):
            if isinstance(b, fx.Node):
                bases.add(b)

    nodes_used_as_mutation_base: set[fx.Node] = set()
    for n in fx_g.find_nodes(
        op="call_function", target=torch.ops.higher_order.auto_functionalized_v2
    ):
        # v2 lists its mutable buffers explicitly in ``_all_bases``.
        for base in n.kwargs.get("_all_bases", ()):
            _add_base(base, nodes_used_as_mutation_base)
    for n in fx_g.find_nodes(
        op="call_function", target=torch.ops.higher_order.auto_functionalized
    ):
        # v1 has no ``_all_bases``; the mutated tensors are the args named by the
        # wrapped op's schema (``_mutable_op`` is the first positional).
        mutable_op = n.args[0]
        mutable_args_names, _ = get_mutable_args(mutable_op)
        for name in mutable_args_names:
            _add_base(n.kwargs.get(name), nodes_used_as_mutation_base)
    for n in fx_g.find_nodes(
        op="call_function", target=triton_kernel_wrapper_functional
    ):
        inner_kwargs = n.kwargs.get("kwargs", {})
        for name in n.kwargs.get("tensors_to_clone", ()):
            _add_base(inner_kwargs.get(name), nodes_used_as_mutation_base)

    for n in fx_g.nodes:
        # The placeholder, output, and get_attr nodes are copied to the new graph without change
        # do not CSE away random operations
        if (
            n.op == "placeholder"
            or n.op == "output"
            or n.op == "get_attr"
            or n.is_impure()
            or get_aten_target(n) in rand_ops
            # aten.empty is non-deterministic, so don't CSE it.
            # Also, aten.empty is almost always fusible into its consumer,
            # so it's not worth CSEing.
            or get_aten_target(n) is aten.empty
            or n in nodes_that_alias_outputs
            # Keep distinct functional-wrapper mutation bases independent.
            or n in nodes_used_as_mutation_base
            # This CSE pass currently doesn't handle re-propagation of unbacked
            # meta where it'll sometimes eliminate a _local_scalar_dense but not
            # replace the meta of downstream users. eg. one bug we've seen is:
            #
            # _local_scalar_dense_11: "Sym(u14)" = torch.ops.aten._local_scalar_dense.default(select_10);
            # sym_sum_2: "Sym(u19 + u20 + u21)" = torch.sym_sum((_local_scalar_dense_11, _local_scalar_dense_12, _local_scalar_dense_13))
            #
            # Notice how _local_scalar_dense_11 is u14 but sym_sum_2's meta is incorrectly the old
            # pre-cse value of u19.
            or (
                "val" in n.meta
                and isinstance(n.meta["val"], sympy.Symbol)
                and free_unbacked_symbols(n.meta["val"])
            )
        ):
            new_node = new_graph.node_copy(n, lambda x: env[x])
            env[n] = new_node
        else:  # n.op == 'call_function', should never see n.op == 'call_module' or 'call_method'
            # substitute args and kwargs members to their mapping in env if exists
            # specs can be used to reconstruct nested list/dictionaries
            def substitute(
                arg_list: list[Any] | tuple[Any, ...],
            ) -> tuple[tuple[Any, ...], TreeSpec]:
                arg_list, spec = tree_flatten(arg_list)
                for i in range(len(arg_list)):
                    v = arg_list[i]
                    if isinstance(v, torch.fx.node.Node) and v in env:
                        arg_list[i] = env[v]
                    if isinstance(v, (torch.SymBool, torch.SymInt, torch.SymFloat)):
                        arg_list[i] = v.node
                    arg_list[i] = _normalize_cse_arg(arg_list[i])
                return tuple(arg_list), spec

            args, args_spec = substitute(n.args)
            kwargs, kwargs_spec = substitute(n.kwargs)

            # each token corresponds to a unique node
            # nodes with the same token can be substituted
            extra_key = extra_node_key(n) if extra_node_key is not None else None
            token = {
                "target": n.target,
                "extra_key": extra_key,
                "args": args,
                "args_spec": args_spec,
                "kwargs": kwargs,
                "kwargs_spec": kwargs_spec,
                "custom_context": custom_context_key(n),
            }

            # hash substituted args to a number, do not hash specs because specs are not hashable
            # We need to add type into hash to avoid situations like:
            # hash((primals_2, 1.0)) == hash((primals_2, 1))
            hash_arg = hash(
                (tuple((a, type(a)) for a in args), tuple((a, type(a)) for a in kwargs))
            )
            hash_val = (n.target, extra_key, hash_arg)

            # check if a node has a substitute and can be eliminated
            hash_val_in_hash_env = hash_val in hash_env
            overwrite_due_to_mutation = False
            if hash_val_in_hash_env and token_map[hash_val] == token:
                duplicate_n_prev = hash_env[hash_val]
                if same_mutation_regions(n, duplicate_n_prev):
                    env[n] = duplicate_n_prev
                    continue
                else:
                    # any futures duplicates should replace with n, not duplicate_n_prev
                    overwrite_due_to_mutation = True

            new_node = new_graph.node_copy(n, lambda x: env[x])
            env[n] = new_node
            if overwrite_due_to_mutation or not hash_val_in_hash_env:
                hash_env[hash_val] = new_node
                token_map[hash_val] = token

    return new_graph


def raise_getitems(gm: fx.GraphModule) -> fx.GraphModule:
    # Pre-create a list of nodes to iterate over, as modifying the node order
    # during the loop can lead to infinite loops if not handled properly.
    getitem_nodes = list(
        gm.graph.find_nodes(op="call_function", target=operator.getitem)
    )

    # loop through getitem nodes in the graph and raise them to the parent node
    # in reverse order to preserve their original relative order
    for node in reversed(getitem_nodes):
        if len(node.all_input_nodes) != 1:
            raise AssertionError(
                f"expected node {node.name} to have 1 input node, got {len(node.all_input_nodes)}"
            )
        parent = node.all_input_nodes[0]
        parent.append(node)

    gm.recompile()
    return gm


def strip_overloads(gm: fx.GraphModule) -> None:
    """
    Modifies the target of graph nodes in :attr:`gm` to strip overloads.

    Args:
        gm(fx.GraphModule): The input Fx graph module to be modified
    """
    for node in gm.graph.nodes:
        if isinstance(node.target, torch._ops.OpOverload):
            node.target = node.target.overloadpacket
    gm.recompile()


def get_placeholders(graph: fx.Graph) -> list[Any]:
    return graph.find_nodes(op="placeholder")


def get_outputs(graph: fx.Graph) -> list[fx.Node]:
    for node in graph.find_nodes(op="output"):
        return pytree.tree_leaves(node.args[0])
    raise AssertionError("No output node found")
<|endoftext|>from __future__ import annotations

import contextlib
import json
import operator
import os
import time
from contextlib import AbstractContextManager
from typing import Any, TYPE_CHECKING
from typing_extensions import TypeVar

import torch
from torch.profiler import profile, ProfilerActivity


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


_R = TypeVar("_R")


def synchronize() -> None:
    pass


def dump_chrome_trace(
    f: Callable[[tuple[Any, ...]], _R],
    input_: tuple[Any, ...],
    trace_filename: str,
    optimize_ctx: AbstractContextManager[Any],
    activities: Sequence[ProfilerActivity],
    num_runs: int = 1,
    devices: list[str] | None = None,
    kwargs_for_f: dict[str, Any] | None = None,
    kwargs_for_profiler: dict[str, Any] | None = None,
) -> float:
    """
    Output the chrome trace of running f(input_, **kwargs_for_f) with [optimize_ctx]
    [num_runs] times to [trace_filename].

    [activities] are the activities that the profiler will record, e.g. ProfilerActivity.CUDA.
    Return total runtime without the profiler

    Outputs to trace_filename
    """

    if devices is None:
        devices = ["cuda"]

    global synchronize
    if devices != ["cpu"] and torch.accelerator.is_available():
        synchronize = torch.accelerator.synchronize

    if kwargs_for_f is None:
        kwargs_for_f = {}
    if kwargs_for_profiler is None:
        kwargs_for_profiler = {}

    with optimize_ctx:
        torch.manual_seed(1337)
        for _ in range(5):  # warmup runs
            f(input_, **kwargs_for_f)
            synchronize()
        torch.manual_seed(1337)
        t0 = time.perf_counter()
        for _ in range(num_runs):
            f(input_, **kwargs_for_f)
            synchronize()
        t1 = time.perf_counter()
    timing = t1 - t0

    with profile(activities=activities, **kwargs_for_profiler) as prof:
        with optimize_ctx:
            synchronize()
            torch.manual_seed(1337)
            for _ in range(num_runs):
                f(input_, **kwargs_for_f)
                synchronize()
    prof.export_chrome_trace(trace_filename)

    return timing


def get_chrome_trace_events(filename: str) -> list[dict[str, Any]]:
    with open(filename) as f:
        data = json.load(f)
    events = data["traceEvents"]
    return events


def is_gpu_compute_event(event: dict[str, Any]) -> bool:
    global gpu_pids
    return (
        "pid" in event
        and event["pid"] in gpu_pids
        and "ph" in event
        and event["ph"] == "X"
    )


def get_sorted_gpu_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_gpu_events: list[dict[str, Any]] = []
    for event in events:
        if not is_gpu_compute_event(event):
            continue
        sorted_gpu_events.append(event)
    return sorted(sorted_gpu_events, key=operator.itemgetter("ts"))


def get_duration(sorted_gpu_events: list[dict[str, Any]]) -> int:
    if len(sorted_gpu_events) == 0:
        return 0
    event = sorted_gpu_events[0]
    current_end_time = event["ts"] + event["dur"]
    total_duration = event["dur"]
    for event in sorted_gpu_events[1:]:
        start_time = max(event["ts"], current_end_time)
        end_time = event["ts"] + event["dur"]
        total_duration = total_duration + max(end_time - start_time, 0)
        current_end_time = max(current_end_time, end_time)
    return total_duration


def get_sorted_gpu_mm_conv_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def is_mm_conv_event(event: dict[str, Any]) -> bool:
        return "name" in event and (
            "gemm" in event["name"]
            or "conv" in event["name"]
            or "cutlass" in event["name"]
            or "wgrad" in event["name"]
        )

    gpu_events = get_sorted_gpu_events(events)
    sorted_events: list[dict[str, Any]] = []
    for event in gpu_events:
        if not is_mm_conv_event(event):
            continue
        sorted_events.append(event)
    return sorted_events


gpu_pids: list[Any] = []


def compute_utilization(filename: str, total_length: float) -> tuple[float, float]:
    """
    Process the chrome traces outputs by the pytorch profiler to compute GPU Utilization
    and percent of times spent on matmul and convolution

    Args:
        filename(str): Name of chrome traces file produced by pytorch profiler

        total_length(float): total length of the process without profiler in second

    Return:
        tuple: (GPU Utilization, percent of time spent on matmul and convolution)
    """
    events = get_chrome_trace_events(filename)

    # get pids of GPU events
    global gpu_pids
    gpu_pids = []
    for event in events:
        if "name" not in event:
            continue
        if event["name"] == "process_labels" and "GPU" in event["args"]["labels"]:
            gpu_pids.append(event["pid"])

    total_length = total_length * 1e6
    sorted_gpu_events = get_sorted_gpu_events(events)
    utilization = get_duration(sorted_gpu_events) / total_length

    sorted_gpu_mm_conv_events = get_sorted_gpu_mm_conv_events(events)
    mm_conv_utilization = get_duration(sorted_gpu_mm_conv_events) / total_length

    return utilization, mm_conv_utilization


def benchmark_utilization(
    f: Callable[[tuple[Any, ...]], _R],
    input_: tuple[Any, ...],
    trace_folder: str,
    optimize_ctx: AbstractContextManager[Any] | None = None,
    trace_file_name: str = "tmp_chrome_trace",
    num_runs: int = 1,
) -> tuple[float, float]:
    """
    Benchmark the GPU Utilization and percent of time spent on matmul and convolution operations of
    running f(input_, **kwargs_for_f) with [optimize_ctx] [num_runs] times.
    It will produce a chrome trace file in trace_folder/trace_file_name.json

    Example:

    ```
    def f(a):
        return a.sum()


    a = torch.rand(2**20, device="cuda")
    utilization, mm_conv_utilization = benchmark_utilization(
        f, a, "tmp", trace_file_name="tmp_chrome_trace"
    )
    ```

    Args:
        f: function to benchmark

        input_: input to :attr:`f`

        trace_folder: name of the folder to store the chrome trace

        optimize_ctx: the context in which f will run

        trace_file_name: name of the dumped chrome trace file, default to "tmp_chrome_trace"

        num_runs: number of times to run f, excluding the warm-up runs, default to 1.

    Return:
        tuple: (GPU Utilization, percent of time spent on matmul and convolution)

    """
    isExist = os.path.exists(trace_folder)
    if not isExist:
        os.makedirs(trace_folder)
        print("create folder " + trace_folder)

    if optimize_ctx is None:
        optimize_ctx = contextlib.nullcontext()

    chrome_trace_file_name = os.path.join(trace_folder, trace_file_name + ".json")
    total_length = dump_chrome_trace(
        f,
        input_,
        chrome_trace_file_name,
        optimize_ctx,
        [ProfilerActivity.CUDA],
        num_runs=num_runs,
        devices=["cuda"],
    )
    utilization, mm_conv_utilization = compute_utilization(
        chrome_trace_file_name, total_length
    )

    return utilization, mm_conv_utilization
<|endoftext|>import torch.nn as nn
from torch._functorch.utils import exposed_in


def batch_norm_without_running_stats(module: nn.Module) -> None:
    if (
        isinstance(module, nn.modules.batchnorm._BatchNorm)
        and module.track_running_stats
    ):
        module.running_mean = None
        module.running_var = None
        module.num_batches_tracked = None
        module.track_running_stats = False


@exposed_in("torch.func")
def replace_all_batch_norm_modules_(root: nn.Module) -> nn.Module:
    """
    In place updates :attr:`root` by setting the ``running_mean`` and ``running_var`` to be None and
    setting track_running_stats to be False for any nn.BatchNorm module in :attr:`root`
    """
    # base case
    batch_norm_without_running_stats(root)

    for obj in root.modules():
        batch_norm_without_running_stats(obj)
    return root
<|endoftext|>from __future__ import annotations

import torch
from torch._functorch.utils import exposed_in


__all__ = ["rearrange"]


@exposed_in("torch.func")
def rearrange(
    tensor: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    pattern: str,
    **axes_lengths: int,
) -> torch.Tensor:
    r"""A native implementation of `einops.rearrange`, a reader-friendly smart element reordering for multidimensional
    tensors. This operation includes functionality of transpose (axes permutation), reshape (view), squeeze, unsqueeze,
    stack, concatenate and other operations.

    See: https://einops.rocks/api/rearrange/

    Args:
        tensor (Tensor or sequence of Tensor): the tensor(s) to rearrange
        pattern (str): the rearrangement pattern
        axes_lengths (int): any additional length specifications for dimensions

    Returns:
        Tensor: the rearranged tensor

    Examples:
        >>> from torch.func import rearrange
        >>> # suppose we have a set of 32 images in "h w c" format (height-width-channel)
        >>> images = torch.randn((32, 30, 40, 3))

        >>> # stack along first (batch) axis, output is a single array
        >>> rearrange(images, "b h w c -> b h w c").shape
        torch.Size([32, 30, 40, 3])

        >>> # concatenate images along height (vertical axis), 960 = 32 * 30
        >>> rearrange(images, "b h w c -> (b h) w c").shape
        torch.Size([960, 40, 3])

        >>> # concatenated images along horizontal axis, 1280 = 32 * 40
        >>> rearrange(images, "b h w c -> h (b w) c").shape
        torch.Size([30, 1280, 3])

        >>> # reordered axes to "b c h w" format for deep learning
        >>> rearrange(images, "b h w c -> b c h w").shape
        torch.Size([32, 3, 30, 40])

        >>> # flattened each image into a vector, 3600 = 30 * 40 * 3
        >>> rearrange(images, "b h w c -> b (c h w)").shape
        torch.Size([32, 3600])

        >>> # split each image into 4 smaller (top-left, top-right, bottom-left, bottom-right), 128 = 32 * 2 * 2
        >>> rearrange(images, "b (h1 h) (w1 w) c -> (b h1 w1) h w c", h1=2, w1=2).shape
        torch.Size([128, 15, 20, 3])

        >>> # space-to-depth operation
        >>> rearrange(images, "b (h h1) (w w1) c -> b h w (c h1 w1)", h1=2, w1=2).shape
        torch.Size([32, 15, 20, 12])
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.stack(tensor)

    # Lazy import to avoid circular import:
    # torch._functorch.einops -> functorch.einops (package __init__) -> torch._functorch.einops
    from functorch.einops.rearrange import _create_rearrange_callable

    rearrange_callable = _create_rearrange_callable(
        tensor.ndim, pattern, **axes_lengths
    )

    return rearrange_callable(tensor)
<|endoftext|>from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch._functorch.utils import exposed_in


@exposed_in("torch.func")
def functional_call(
    module: torch.nn.Module,
    parameter_and_buffer_dicts: dict[str, Tensor] | Sequence[dict[str, Tensor]],
    args: Any = None,
    kwargs: dict[str, Any] | None = None,
    *,
    tie_weights: bool = True,
    strict: bool = False,
) -> Any:
    r"""Performs a functional call on the module by replacing the module parameters
    and buffers with the provided ones.

    .. note:: If the module has active parametrizations, passing a value in the
        :attr:`parameter_and_buffer_dicts` argument with the name set to the regular parameter
        name will completely disable the parametrization.
        If you want to apply the parametrization function to the value passed
        please set the key as ``{submodule_name}.parametrizations.{parameter_name}.original``.

    .. note:: If the module performs in-place operations on parameters/buffers, these will be reflected
        in the ``parameter_and_buffer_dicts`` input.


         Example::

            >>> a = {'foo': torch.zeros(())}
            >>> # xdoctest: +SKIP
            >>> mod = Foo()  # does self.foo = self.foo + 1
            >>> print(mod.foo)  # tensor(0.)
            >>> functional_call(mod, a, torch.ones(()))
            >>> print(mod.foo)  # tensor(0.)
            >>> print(a['foo'])  # tensor(1.)

    .. note:: If the module has tied weights, whether or not functional_call respects the tying is determined by the
        tie_weights flag.

        Example::

            >>> a = {'foo': torch.zeros(())}
            >>> # xdoctest: +SKIP
            >>> mod = Foo()  # has both self.foo and self.foo_tied which are tied. Returns x + self.foo + self.foo_tied
            >>> print(mod.foo)  # tensor(1.)
            >>> mod(torch.zeros(()))  # tensor(2.)
            >>> functional_call(mod, a, torch.zeros(()))  # tensor(0.) since it will change self.foo_tied too
            >>> functional_call(mod, a, torch.zeros(()), tie_weights=False)  # tensor(1.)--self.foo_tied is not updated
            >>> new_a = {'foo': torch.zeros(()), 'foo_tied': torch.zeros(())}
            >>> functional_call(mod, new_a, torch.zeros()) # tensor(0.)

    An example of passing multiple dictionaries

    .. code-block:: python

            a = (
                {"weight": torch.ones(1, 1)},
                {"buffer": torch.zeros(1)},
            )  # two separate dictionaries
            mod = nn.Bar(1, 1)  # return self.weight @ x + self.buffer
            print(mod.weight)  # tensor(...)
            print(mod.buffer)  # tensor(...)
            x = torch.randn((1, 1))
            print(x)
            functional_call(mod, a, x)  # same as x
            print(mod.weight)  # same as before functional_call


    And here is an example of applying the grad transform over the parameters
    of a model.

    .. code-block:: python

        import torch
        import torch.nn as nn
        from torch.func import functional_call, grad

        x = torch.randn(4, 3)
        t = torch.randn(4, 3)
        model = nn.Linear(3, 3)


        def compute_loss(params, x, t):
            y = functional_call(model, params, x)
            return nn.functional.mse_loss(y, t)


        grad_weights = grad(compute_loss)(dict(model.named_parameters()), x, t)

    .. note:: If the user does not need grad tracking outside of grad transforms, they can detach all of the
        parameters for better performance and memory usage

        Example::

            >>> detached_params = {k: v.detach() for k, v in model.named_parameters()}
            >>> grad_weights = grad(compute_loss)(detached_params, x, t)
            >>> grad_weights.grad_fn  # None--it's not tracking gradients outside of grad

        This means that the user cannot call ``grad_weight.backward()``. However, if they don't need autograd tracking
        outside of the transforms, this will result in less memory usage and faster speeds.

    Args:
        module (torch.nn.Module): the module to call
        parameters_and_buffer_dicts (Dict[str, Tensor] or tuple of Dict[str, Tensor]): the parameters that will be used in
            the module call. If given a tuple of dictionaries, they must have distinct keys so that all dictionaries can
            be used together
        args (Any or tuple): arguments to be passed to the module call. If not a tuple, considered a single argument.
        kwargs (dict): keyword arguments to be passed to the module call
        tie_weights (bool, optional): If True, then parameters and buffers tied in the original model will be treated as
            tied in the reparameterized version. Therefore, if True and different values are passed for the tied
            parameters and buffers, it will error. If False, it will not respect the originally tied parameters and
            buffers unless the values passed for both weights are the same. Default: True.
        strict (bool, optional): If True, then the parameters and buffers passed in must match the parameters and
            buffers in the original module. Therefore, if True and there are any missing or unexpected keys, it will
            error. Default: False.

    Returns:
        Any: the result of calling ``module``.
    """
    if isinstance(parameter_and_buffer_dicts, dict):
        parameters_and_buffers = parameter_and_buffer_dicts
    elif isinstance(parameter_and_buffer_dicts, Sequence):
        if not all(isinstance(d, dict) for d in parameter_and_buffer_dicts):
            raise ValueError(
                "Expected all elements of parameter_and_buffer_dicts to be dictionaries"
            )
        all_keys = [k for d in parameter_and_buffer_dicts for k in d]
        all_keys_counter: dict[str, int] = {}
        for k in all_keys:
            v = all_keys_counter.get(k, 0)
            all_keys_counter[k] = v + 1
        repeated_keys = [key for key, n in all_keys_counter.items() if n > 1]
        if len(repeated_keys) > 0:
            raise ValueError(
                f"{repeated_keys} appeared in multiple dictionaries; behavior of functional call is ambiguous"
            )
        parameters_and_buffers = {
            k: v for d in parameter_and_buffer_dicts for k, v in d.items()
        }
    else:
        raise ValueError(
            f"Expected parameter_and_buffer_dicts to be a dict, or a list/tuple of dicts, "
            f"but got {type(parameter_and_buffer_dicts)}"
        )

    return nn.utils.stateless._functional_call(
        module,
        parameters_and_buffers,
        args,
        kwargs,
        tie_weights=tie_weights,
        strict=strict,
    )


@exposed_in("torch.func")
def stack_module_state(
    models: Sequence[nn.Module] | nn.ModuleList,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """stack_module_state(models) -> params, buffers

    Prepares a list of torch.nn.Modules for ensembling with :func:`vmap`.

    Given a list of ``M`` ``nn.Modules`` of the same class, returns two dictionaries
    that stack all of their parameters and buffers together, indexed by name.
    The stacked parameters are optimizable (i.e. they are new leaf nodes in the
    autograd history that are unrelated to the original parameters and can be
    passed directly to an optimizer).

    Here's an example of how to ensemble over a very simple model:

    .. code-block:: python

        num_models = 5
        batch_size = 64
        in_features, out_features = 3, 3
        models = [torch.nn.Linear(in_features, out_features) for i in range(num_models)]
        data = torch.randn(batch_size, 3)


        def wrapper(params, buffers, data):
            return torch.func.functional_call(models[0], (params, buffers), data)


        params, buffers = stack_module_state(models)
        output = vmap(wrapper, (0, 0, None))(params, buffers, data)

        assert output.shape == (num_models, batch_size, out_features)

    When there's submodules, this follows state dict naming conventions

    .. code-block:: python

        import torch.nn as nn


        class Foo(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                hidden = 4
                self.l1 = nn.Linear(in_features, hidden)
                self.l2 = nn.Linear(hidden, out_features)

            def forward(self, x):
                return self.l2(self.l1(x))


        num_models = 5
        in_features, out_features = 3, 3
        models = [Foo(in_features, out_features) for i in range(num_models)]
        params, buffers = stack_module_state(models)
        print(list(params.keys()))  # "l1.weight", "l1.bias", "l2.weight", "l2.bias"

    .. warning::
        All of the modules being stacked together must be the same (except for
        the values of their parameters/buffers). For example, they should be in the
        same mode (training vs eval).
    """
    if len(models) == 0:
        raise RuntimeError("stack_module_state: Expected at least one model, got 0.")
    if not (all(m.training for m in models) or all(not m.training for m in models)):
        raise RuntimeError(
            "stack_module_state: Expected all models to have the same training/eval mode."
        )
    model0_typ = type(models[0])
    if not all(type(m) is model0_typ for m in models):
        raise RuntimeError(
            "stack_module_state: Expected all models to be of the same class."
        )
    all_params = [dict(model.named_parameters()) for model in models]
    params = {
        k: construct_stacked_leaf(tuple(params[k] for params in all_params), k)
        for k in all_params[0]
    }
    all_buffers = [dict(model.named_buffers()) for model in models]
    buffers = {
        k: construct_stacked_leaf(tuple(buffers[k] for buffers in all_buffers), k)
        for k in all_buffers[0]
    }

    return params, buffers


def construct_stacked_leaf(
    tensors: tuple[Tensor, ...] | list[Tensor], name: str
) -> Tensor:
    all_requires_grad = all(t.requires_grad for t in tensors)
    none_requires_grad = all(not t.requires_grad for t in tensors)
    if not all_requires_grad and not none_requires_grad:
        raise RuntimeError(
            f"Expected {name} from each model to have the same .requires_grad"
        )
    result = torch.stack(tensors)
    if all_requires_grad:
        result = result.detach().requires_grad_()
    return result
<|endoftext|>from __future__ import annotations

import copy
import logging
import os
import pickle
import random
from contextlib import contextmanager
from functools import partial
from typing import Any, TYPE_CHECKING
from typing_extensions import ParamSpec, TypeVar

import sympy

import torch
import torch.fx as fx
import torch.nn as nn
import torch.utils._pytree as pytree
from torch import SymInt
from torch._decomp import get_decompositions
from torch.fx.experimental.symbolic_shapes import bind_symbols

from .aot_autograd import aot_function, aot_module, make_boxed_compiler
from .compile_utils import strip_overloads
from .partitioners import (
    default_partition,
    draw_graph,
    min_cut_rematerialization_partition,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from torch.fx.node import Node
    from torch.types import IntLikeType


_P = ParamSpec("_P")
_R = TypeVar("_R")

log = logging.getLogger(__name__)


# These canonicalizations are needed here (and not decompositions), as the ops
# we're trying to canonicalize to CompositeImplicitAutograd.
def _canonicalize(fx_g: fx.GraphModule) -> fx.GraphModule:
    for node in fx_g.graph.find_nodes(
        op="call_function", target=torch.ops.aten._to_copy
    ):
        node.target = torch.ops.aten.to
    fx_g.recompile()
    return fx_g


@contextmanager
def _disable_jit_autocast() -> Generator[None, None, None]:
    # pyrefly: ignore [missing-attribute]
    old_jit_autocast_flag = torch._C._jit_set_autocast_mode(False)
    try:
        yield
    finally:
        # pyrefly: ignore [missing-attribute]
        torch._C._jit_set_autocast_mode(old_jit_autocast_flag)


@make_boxed_compiler
def ts_compile(fx_g: fx.GraphModule, inps: Sequence[Any]) -> torch.jit.ScriptModule:
    """
    Compiles the :attr:`fx_g` with Torchscript compiler.

    .. warning::
        This API is experimental and likely to change.

    Args:
        fx_g(fx.GraphModule): The input Fx graph module to be compiled.

    Returns:
        Torch scripted model.
    """

    from torch.fx._lazy_graph_module import _unwrap_lazy_graph_module

    fx_g = _unwrap_lazy_graph_module(fx_g)

    with _disable_jit_autocast():
        strip_overloads(fx_g)

        for node in fx_g.graph.find_nodes(
            op="call_function", target=torch.ops.aten._to_copy
        ):
            if len(node.args) == 1 and len(node.kwargs) == 1 and "dtype" in node.kwargs:
                node.target = torch.ops.aten.to

        for node in fx_g.graph.nodes:
            new_kwargs = {}
            for k, v in node.kwargs.items():
                if isinstance(v, torch.device):
                    v = v.type
                new_kwargs[k] = v
            node.kwargs = new_kwargs

        fx_g.graph.lint()

        fx_g.recompile()

        f = torch.jit.script(fx_g)

        # pyrefly: ignore [missing-attribute]
        torch._C._jit_pass_remove_mutation(f.graph)

        f = torch.jit.freeze(f.eval())
        f = torch.jit.optimize_for_inference(f)
        if not any(torch._subclasses.fake_tensor.is_fake_tensor(t) for t in inps):
            f(*inps)
    return f


def _draw_graph_compile(
    fx_g: fx.GraphModule, _: Any, name: str, clear_meta: bool = True
) -> fx.GraphModule:
    print(fx_g.code)
    draw_graph(fx_g, name, clear_meta=clear_meta)
    return fx_g


def draw_graph_compile(
    name: str,
) -> Callable[[fx.GraphModule, list[Any]], fx.GraphModule]:
    return make_boxed_compiler(partial(_draw_graph_compile, name=name))


@make_boxed_compiler
def nop(fx_g: fx.GraphModule, _: Any) -> fx.GraphModule:
    """
    Returns the :attr:`fx_g` Fx graph module as it is. This is a no-op compiler
    and can be used to check accuracy.

    .. warning::
        This API is experimental and likely to change.

    """
    return fx_g


class DebugInterpreter(fx.Interpreter):
    def run(
        self,
        *args: Any,
        initial_env: dict[Node, Any] | None = None,
        enable_io_processing: bool = True,
    ) -> Any:
        self.symbol_mapping = bind_symbols(
            # pyrefly: ignore[bad-argument-type]
            self.module,
            *args,
        )
        return super().run(
            *args, initial_env=initial_env, enable_io_processing=enable_io_processing
        )

    def run_node(self, n: Node) -> Any:
        def subst_symint(ni: IntLikeType) -> int:
            if not isinstance(ni, SymInt):
                return ni
            r = sympy.expand(ni.node.expr.xreplace(self.symbol_mapping))
            if not r.is_number:
                raise AssertionError(f"expected r to be a number, got {r}")
            return int(r)

        def subst_symint_tuple(nis: tuple[IntLikeType, ...]) -> tuple[int, ...]:
            return tuple(subst_symint(ni) for ni in nis)

        def check_significant_strides(a: torch.Tensor, b: torch.Tensor) -> bool:
            if subst_symint(a.numel()) > 0:
                for idx in range(a.ndim):
                    if (
                        subst_symint(a.stride(idx)) != b.stride(idx)
                        and subst_symint(a.size(idx)) > 1
                    ):
                        return False
            return True

        def check(nv: torch.Tensor, rv: torch.Tensor, desc: Callable[[], str]) -> None:
            if not callable(desc):
                raise AssertionError(f"expected desc to be callable, got {type(desc)}")
            if nv.dtype != rv.dtype:
                raise AssertionError(f"{desc()}: {nv.dtype} != {rv.dtype}")
            if subst_symint_tuple(nv.size()) != rv.size():
                raise AssertionError(
                    f"{desc()}: {nv.size()} aka {subst_symint_tuple(nv.size())} != {rv.size()}"
                )
            same_strides = check_significant_strides(nv, rv)
            if not same_strides:
                raise AssertionError(
                    f"{desc()}: {nv.stride()} aka {subst_symint_tuple(nv.stride())} != {rv.stride()}"
                )

        r = super().run_node(n)
        if "val" in n.meta:
            n_vals, _n_spec = pytree.tree_flatten(n.meta["val"])
            r_vals, _r_spec = pytree.tree_flatten(r)
            # TODO: There is some sort of problem where we record that an
            # operator returned a tuple/list, and then later it turns out the
            # real version of the operator returned a list/tuple. Need to
            # figure out what's actually going on here, the error itself is
            # harmless enough as we only getitem out the outputs.
            # assert n_spec == r_spec, f"{n_spec} != {r_spec}"
            if len(n_vals) != len(r_vals):
                raise AssertionError(f"{len(n_vals)} != {len(r_vals)}")
            for i, nv, rv in zip(range(len(n_vals)), n_vals, r_vals):
                if not isinstance(rv, torch.Tensor):
                    continue
                check(nv, rv, lambda: f"output {i} where {self.symbol_mapping}")
        return r


@make_boxed_compiler
def debug_nop(
    fx_g: fx.GraphModule, _: Any
) -> Callable[[DebugInterpreter, Any, dict[Node, Any] | None, bool], Any]:
    """
    Returns a (slow) interpreter over the FX graph module that also checks
    various debugging properties (e.g., that tracing strides matched real
    strides.)
    """
    return DebugInterpreter(fx_g).run


@make_boxed_compiler
def simple_ts_compile(fx_g: fx.GraphModule, _: Any) -> torch.jit.ScriptModule:
    strip_overloads(fx_g)
    f = torch.jit.script(fx_g)
    f = torch.jit.freeze(f.eval())
    return f


def nnc_jit(f: Callable[..., Any]) -> Callable[..., Any]:
    return aot_function(f, simple_ts_compile)


aten = torch.ops.aten
default_decompositions = {
    aten.detach,
    aten.gelu_backward,
    aten.leaky_relu_backward,
    aten.sigmoid_backward,
    aten.threshold_backward,
    aten.hardtanh_backward,
    aten.hardsigmoid_backward,
    aten.hardswish_backward,
    aten.tanh_backward,
    aten.silu_backward,
    aten.elu_backward,
    aten.cudnn_batch_norm,
    aten.cudnn_batch_norm_backward,
    aten.masked_fill.Scalar,
    aten.masked_fill.Tensor,
    aten.elu,
    aten.leaky_relu,
    aten.hardtanh,
    aten.hardswish,
    aten.hardsigmoid,
    aten.conj_physical,
    aten.is_same_size,
}

# pyrefly: ignore[bad-argument-type]
default_decompositions = get_decompositions(default_decompositions)


@make_boxed_compiler
def print_compile(fx_g: fx.GraphModule, _: Any) -> fx.GraphModule:
    print(fx_g.code)
    return fx_g


def memory_efficient_fusion(
    fn: Callable[_P, _R] | nn.Module,
    **kwargs: Any,
) -> Callable[_P, _R] | nn.Module:
    """
    Wrapper function over :func:`aot_function` and :func:`aot_module` to perform
    memory efficient fusion. It uses the
    :func:`min_cut_rematerialization_partition` partitioner to perform efficient
    recomputation. It uses NVFuser to compile the generated forward and backward
    graphs.

    .. warning::
        This API is experimental and likely to change.

    Args:
        fn (Union[Callable, nn.Module]): A Python function or a ``nn.Module``
            that takes one or more arguments. Must return one or more Tensors.
        **kwargs: Any other overrides you want to make to the settings

    Returns:
        Returns a ``Callable``  or ``nn.Module`` that retains the eager behavior
        of the original :attr:`fn`, but whose forward and backward graphs have
        gone through recomputation optimizations, and the graphs have been
        compiled with nvfuser.

    """
    config = {
        "fw_compiler": ts_compile,
        "bw_compiler": ts_compile,
        "partition_fn": min_cut_rematerialization_partition,
        "decompositions": default_decompositions,
    }
    config.update(kwargs)
    if isinstance(fn, torch.nn.Module):
        return aot_module(fn, **config)  # pyrefly: ignore[bad-argument-type]
    else:
        return aot_function(fn, **config)  # pyrefly: ignore[bad-argument-type]


def debug_compile(
    fx_g: fx.GraphModule, inps: Sequence[torch.Tensor]
) -> torch.jit.ScriptModule:
    fx_g.to_folder("foo")
    print(
        f"""
##############################################################
# To minimize FX graph, copy and paste the below and run it  #
##############################################################

import torch
import torch.fx as fx
from functorch.compile import minifier, check_nvfuser_subprocess, check_nvfuser_correctness_subprocess

inps = {[(i.shape, i.dtype) for i in inps]}
inps = [torch.ones(shape, dtype=dtype, device='cuda') for (shape, dtype) in inps]
from foo import FxModule
mod = FxModule().cuda()

with torch.jit.fuser("fuser2"):
  # check_nvfuser_subprocess can be replaced with check_nvfuser_correctness_subprocess
  minifier(fx.symbolic_trace(mod), inps, check_nvfuser_subprocess)
"""
    )
    # pyrefly: ignore[missing-import, missing-module-attribute]
    from foo import FxModule

    FxModule().cuda()(*inps)

    return ts_compile(fx_g, inps)


graph_index: int = 0


def get_inputs(input_data_path: str) -> list[torch.Tensor]:
    """
    Return a random input for the given inputs meta generated from _save_fx_default.
    """
    inputs: list[torch.Tensor] = []
    with open(input_data_path, "rb") as f:
        inputs_meta = pickle.load(f)
        inputs = []
        for meta in inputs_meta:
            if len(meta) == 1:
                type = meta
                input_ = type(random.random())
            else:
                type, shape, _stride, dtype, device = meta
                if dtype in {
                    torch.int,
                    torch.int32,
                    torch.int64,
                    torch.bool,
                    torch.int,
                    torch.uint8,
                    int,
                    float,
                }:
                    input_ = torch.randint(0, 1, shape, dtype=dtype, device=device)
                else:
                    input_ = torch.rand(shape, dtype=dtype, device=device)
            inputs.append(input_)
    return inputs


def _save_fx_default(
    current_name: str,
    folder_name: str,
    dump_example_input: bool,
    gm: torch.fx.GraphModule,
    example_inputs: list[torch.Tensor],
) -> nn.Module:
    """
    The forward, backward, and joint computation graph will be stored in
    {folder_name}/{current_name}/{current_name}_forward_{graph_index},
    {folder_name}/{current_name}/{current_name}_backward_{graph_index}, and
    {folder_name}/{current_name}/{current_name}_joint_{graph_index} respectively.
    The input shape of the graphs will be stored in the .input files.
    These files can be loaded with pickle,
    and is a list of format (type, shape, stride, dtype, device).
    In the case of type = int or float, it is just (type,).
    For joint graph input, it is a nested list [[],[]]
    where the two inner lists have the same format.
    If dump_example_input is True, example_inputs will be stored in .pt file.
    Since each function might produce multiple graphs,
    the graph_index is used to distinguish different graphs
    """
    from functorch.compile import aot_module_simplified

    def get_input_meta(args: Any) -> list[Any]:
        input_meta = []
        if len(args) > 0 and isinstance(args[0], tuple):  # joint input
            input_meta += get_input_meta(args[0])
            input_meta += get_input_meta(args[1])
            return input_meta
        for arg in args:
            if type(arg) is int or type(arg) is float:
                input_meta.append((type(arg),))
            else:
                input_meta.append(
                    (type(arg), arg.shape, arg.stride(), arg.dtype, arg.device)
                )
        return input_meta

    def graph_saver_helper(
        gm_to_save: fx.GraphModule, args: Any, type_name: str
    ) -> None:
        global graph_index
        if len(gm_to_save.graph.nodes) == 0:
            log.log(
                logging.WARNING,
                "No nodes in graph {%s}_{%s}_{%s}.",
                current_name,
                type_name,
                graph_index,
            )
            return

        gm = copy.deepcopy(gm_to_save)
        gm.graph.set_codegen(torch.fx.graph.CodeGen())  # remove codegen
        gm.recompile()

        input_meta = get_input_meta(args)

        os.makedirs(f"{folder_name}/{current_name}", exist_ok=True)
        gm.to_folder(
            f"{folder_name}/{current_name}/{current_name}_{type_name}_{graph_index}"
        )
        with open(
            f"{folder_name}/{current_name}/{current_name}_{type_name}_{graph_index}/{current_name}_{type_name}_{graph_index}.input",
            "wb",
        ) as f:
            pickle.dump(input_meta, f)
        if dump_example_input:
            torch.save(
                args,
                f"{folder_name}/{current_name}/{current_name}_{type_name}_{graph_index}/{current_name}_{type_name}_{graph_index}.pt",
            )

    def graph_saver_forward(
        gm: fx.GraphModule, example_inputs: list[torch.Tensor]
    ) -> fx.GraphModule:
        graph_saver_helper(gm, example_inputs, "forward")
        return gm

    def graph_saver_backward(
        gm: fx.GraphModule, example_inputs: list[torch.Tensor]
    ) -> fx.GraphModule:
        graph_saver_helper(gm, example_inputs, "backward")
        global graph_index
        graph_index += 1
        return gm

    def graph_saver_joint(
        gm: fx.GraphModule, joint_args: list[torch.Tensor]
    ) -> tuple[fx.GraphModule, fx.GraphModule]:
        graph_saver_helper(gm, joint_args, "joint")
        return default_partition(gm, joint_args)  # pyrefly: ignore[missing-argument]

    # pyrefly: ignore[bad-return]
    return aot_module_simplified(
        gm,
        example_inputs,
        fw_compiler=graph_saver_forward,  # pyrefly: ignore[bad-argument-type]
        bw_compiler=graph_saver_backward,  # pyrefly: ignore[bad-argument-type]
        partition_fn=graph_saver_joint,
        decompositions=default_decompositions,  # pyrefly: ignore[bad-argument-type]
    )


# WARNING: This isn't tested anywhere!!
def graph_dumper_aot(
    current_name: str, folder_name: str, dump_example_input: bool = False
) -> Callable[[bool, nn.Module], Any]:
    """
    Dump the forward, backward, and joint computation graph.
    Example Usage:
    save_fx_func = graph_dumper_aot(current_name, folder_name, dump_example_input = False)
    optimize_ctx = torchdynamo.optimize(
        save_fx_func
    )
    with torch.enable_grad():
        with optimize_ctx:
            result = forward_and_backward_pass(model, example_inputs)
    """
    global graph_index
    graph_index = 0
    return partial(_save_fx_default, current_name, folder_name, dump_example_input)
<|endoftext|># Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings

# TODO: remove this file when the migration of the pytree utility is done
from torch.utils._pytree import tree_map_, treespec_pprint


__all__ = ["tree_map_", "treespec_pprint"]


with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "`torch._functorch.pytree_hacks` is deprecated and will be removed in a future release. "
        "Please `use torch.utils._pytree` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
<|endoftext|>from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, TYPE_CHECKING

import torch
import torch.utils._pytree as pytree
from torch._C._functorch import (
    CFunctionalizeInterpreterPtr,
    CGradInterpreterPtr,
    CInterpreter,
    CJvpInterpreterPtr,
    CVmapInterpreterPtr,
    pop_dynamic_layer_stack,
    push_dynamic_layer_stack,
    RandomnessType,
    TransformType,
)
from torch.autograd.forward_ad import _set_fwd_grad_enabled


if TYPE_CHECKING:
    from collections.abc import Generator


"""
This file contains the functorch integration with PyDispatcher.

PyDispatcher does not understand functorch's DynamicLayerStack dispatching
logic because it is entirely implemented in C++ in the fallbacks for two
dispatch keys, FuncTorchDynamicLayer{Front, Back}Mode (PyDispatcher is unable
to directly reuse C++ boxed fallbacks).

Instead of trying to hammer PyDispatcher into understanding those fallbacks,
we re-implement the logic of peeking the top of the stack for an interpreter,
selecting the interpreter to dispatch on, etc, in Python. This leads to a
simpler design.

The main difference between C++ functorch and PyDispatcher's functorch logic
is that:
- C++ functorch needs to manually tweak dispatch keys to ping-pong between
  DynamicLayerFrontMode and DynamicLayerBackMode.
- PyDispatcher's functorch logic pops an Interpreter from the top of the stack
  and asks it to execute the rule associated with the Interpreter.

In C++ we do the ping-pong because e.g. vmap rules are associated with the
batched DispatchKey, but in PyDispatcher we are able to avoid this by asking
the user to register a batching rule directly to a transform that an
interpreter then invokes.
"""


# FuncTorchInterpreter is the Python version of Interpreter (recall that
# the DynamicLayerStack is a stack of interpreters).
# It is a wrapper around the actual C++ Interpreter object.
#
# Keep the methods in sync with aten/src/ATen/functorch/Interpreter.h
class FuncTorchInterpreter(ABC):
    def __init__(self, cptr: Any) -> None:
        self._cptr = cptr

    # Process an operation. eg for vmap, this is invoking a batching rule.
    # Conceptually this is analogous to Interpreter::process in C++
    @abstractmethod
    def process(self, op: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        pass

    # lower an operation from this Interpreter to the next Interpreter on the stack.
    # Concretely, this involves temporarily popping the current Interpreter.
    # Conceptually this is analogous to Interpreter::sendToNextInterpreter in C++
    def lower(self) -> contextlib.AbstractContextManager[Any]:
        return temporarily_pop_interpreter_stack()

    def level(self) -> int:
        return self._cptr.level()

    def key(self) -> TransformType:
        return self._cptr.key()

    def get_state(self) -> tuple[Any, ...]:
        raise NotImplementedError

    def check_state(self, state: tuple[Any, ...]) -> bool:
        return state == self.get_state()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_cptr", None)
        return state


@contextlib.contextmanager
def temporarily_pop_interpreter_stack() -> Generator[None, None, None]:
    try:
        saved = pop_dynamic_layer_stack()
        yield
    finally:
        push_dynamic_layer_stack(saved)


@contextlib.contextmanager
def temporarily_clear_interpreter_stack() -> Generator[list[Any], None, None]:
    stack: list[Any] = []
    try:
        while torch._C._functorch.peek_interpreter_stack() is not None:
            stack.append(pop_dynamic_layer_stack())
        yield list(stack)
    finally:
        while stack:
            push_dynamic_layer_stack(stack.pop())


@contextlib.contextmanager
def temporarily_restore_interpreter_stack(
    stack: list[Any] | None,
) -> Generator[None, None, None]:
    pushed: list[Any] = []
    if stack is None:
        return
    try:
        for s in reversed(stack):
            push_dynamic_layer_stack(s)
            pushed.append(s)
        yield
    finally:
        for _ in reversed(pushed):
            # TODO: would be nice to assert that the layers are the same, but
            # Python object identity is not preserved
            pop_dynamic_layer_stack()


class VmapInterpreter(FuncTorchInterpreter):
    def __init__(self, cdata: CInterpreter) -> None:
        if cdata.key() != TransformType.Vmap:
            raise AssertionError(f"expected TransformType.Vmap, got {cdata.key()}")
        # NOTE: [Interpreter cdata vs cptr]
        # cdata is a generic CInterpreter. We wrap it in a CVmapInterpreterPtr
        # so that we can access methods specific to the vmap interpreter
        self._cdata = cdata

    @cached_property
    # pyrefly: ignore [bad-override]
    def _cptr(self) -> CVmapInterpreterPtr:
        return CVmapInterpreterPtr(self._cdata)

    def process(self, op: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        kernel = op.functorch_table[TransformType.Vmap]
        return kernel(self, *args, **kwargs)

    def batch_size(self) -> int:
        return self._cptr.batchSize()

    def randomness(self) -> str:
        typ = self._cptr.randomness()
        if typ == RandomnessType.Error:
            return "error"
        elif typ == RandomnessType.Same:
            return "same"
        elif typ == RandomnessType.Different:
            return "different"
        raise RuntimeError(f"Unknown RandomnessType: {typ}")

    def get_state(self) -> tuple[Any, ...]:
        return (self.key().name, self.level(), self.randomness())


@contextlib.contextmanager
def nested(
    *contexts: contextlib.AbstractContextManager[Any],
) -> Generator[tuple[contextlib.AbstractContextManager[Any], ...], None, None]:
    with contextlib.ExitStack() as stack:
        for ctx in contexts:
            stack.enter_context(ctx)
        yield contexts


class GradInterpreter(FuncTorchInterpreter):
    def __init__(self, cdata: CInterpreter) -> None:
        if cdata.key() != TransformType.Grad:
            raise AssertionError(f"expected TransformType.Grad, got {cdata.key()}")
        # See NOTE: [Interpreter cdata vs cptr]
        self._cdata = cdata

    @cached_property
    # pyrefly: ignore [bad-override]
    def _cptr(self) -> CGradInterpreterPtr:
        return CGradInterpreterPtr(self._cdata)

    def lift(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        args, kwargs = pytree.tree_map_only(
            torch.Tensor, self._cptr.lift, [args, kwargs]
        )
        return args, kwargs

    def process(self, op: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        kernel = op.functorch_table[TransformType.Grad]
        args, kwargs = self.lift(args, kwargs)
        return kernel(self, *args, **kwargs)

    # GradInterpreter has custom lower because of the no_grad interaction
    # See NOTE [grad and vjp interaction with no_grad]
    # This logic is mirrored from C++ GradInterpreterPtr::sendToNextInterpreter
    def lower(self) -> contextlib.AbstractContextManager[Any]:
        prev_grad_mode = self.prev_grad_mode()
        if not prev_grad_mode:
            return nested(torch.no_grad(), super().lower())
        return super().lower()

    def prev_grad_mode(self) -> bool:
        return self._cptr.prevGradMode()

    def get_state(self) -> tuple[Any, ...]:
        return (self.key().name, self.level(), self.prev_grad_mode())


class JvpInterpreter(FuncTorchInterpreter):
    def __init__(self, cdata: CInterpreter) -> None:
        if cdata.key() != TransformType.Jvp:
            raise AssertionError(f"expected TransformType.Jvp, got {cdata.key()}")
        # See NOTE: [Interpreter cdata vs cptr]
        self._cdata = cdata

    @cached_property
    # pyrefly: ignore [bad-override]
    def _cptr(self) -> CJvpInterpreterPtr:
        return CJvpInterpreterPtr(self._cdata)

    def lift(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        args, kwargs = pytree.tree_map_only(
            torch.Tensor, self._cptr.lift, [args, kwargs]
        )
        return args, kwargs

    def process(self, op: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        kernel = op.functorch_table[TransformType.Jvp]
        args, kwargs = self.lift(args, kwargs)
        return kernel(self, *args, **kwargs)

    # Jvp has custom lower because of the no_fwd_grad interaction
    # See NOTE [grad and vjp interaction with no_grad] for related info.
    # This logic is mirrored from C++ JvpInterpreterPtr::sendToNextInterpreter
    def lower(self) -> contextlib.AbstractContextManager[Any]:
        prev_fwd_grad_mode = self.prev_fwd_grad_mode()
        if not prev_fwd_grad_mode:
            return nested(_set_fwd_grad_enabled(False), super().lower())
        return super().lower()

    def prev_fwd_grad_mode(self) -> bool:
        return self._cptr.prevFwdGradMode()

    def get_state(self) -> tuple[Any, ...]:
        return (self.key().name, self.level(), self.prev_fwd_grad_mode())


class FunctionalizeInterpreter(FuncTorchInterpreter):
    def __init__(self, cdata: CInterpreter) -> None:
        if cdata.key() != TransformType.Functionalize:
            raise AssertionError(
                f"expected TransformType.Functionalize, got {cdata.key()}"
            )
        self._cdata = cdata

    @cached_property
    # pyrefly: ignore [bad-override]
    def _cptr(self) -> CFunctionalizeInterpreterPtr:
        return CFunctionalizeInterpreterPtr(self._cdata)

    def process(self, op: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        kernel = op.functorch_table[TransformType.Functionalize]
        return kernel(self, *args, **kwargs)

    def functionalize_add_back_views(self) -> bool:
        return self._cptr.functionalizeAddBackViews()

    def get_state(self) -> tuple[Any, ...]:
        return (self.key().name, self.level())


def coerce_cinterpreter(cinterpreter: CInterpreter) -> FuncTorchInterpreter:
    key = cinterpreter.key()
    if key == TransformType.Grad:
        return GradInterpreter(cinterpreter)
    if key == TransformType.Vmap:
        return VmapInterpreter(cinterpreter)
    if key == TransformType.Jvp:
        return JvpInterpreter(cinterpreter)
    if key == TransformType.Functionalize:
        return FunctionalizeInterpreter(cinterpreter)
    raise RuntimeError(f"NYI: PyDispatcher has not implemented support for {key}")


def retrieve_current_functorch_interpreter() -> FuncTorchInterpreter:
    interpreter = torch._C._functorch.peek_interpreter_stack()
    if interpreter is None:
        raise AssertionError("interpreter must not be None")
    return coerce_cinterpreter(interpreter)


def retrieve_all_functorch_interpreters() -> list[FuncTorchInterpreter]:
    cis = torch._C._functorch.get_interpreter_stack()
    if cis is None:
        return []
    return [coerce_cinterpreter(ci) for ci in cis]


def compare_functorch_state(states: list[tuple[Any, ...]]) -> bool:
    # There are four possible cases covered here:
    # 1. Current stack empty AND stack when generated not empty -> Invalidate
    # 2. Current stack not empty AND stack when generated empty -> Invalidate
    # 3. Current stack and generated stack empty -> Valid FX graph
    # 4. Current stack and generated stack not empty -> Valid if both states match
    peek = torch._C._functorch.peek_interpreter_stack()
    if (peek is None and len(states) != 0) or (peek is not None and len(states) == 0):
        return False

    cis = retrieve_all_functorch_interpreters()
    return len(cis) == len(states) and all(
        ci.check_state(state) for ci, state in zip(cis, states)
    )


def dispatch_functorch(op: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    interpreter = retrieve_current_functorch_interpreter()
    # In traditional PyTorch operators, DispatchKey::FuncTorchTensorWrapper's
    # unwrap_dead_tensors fallback handles unwrapping dead tensor wrappers.
    # PyDispatcher sidesteps the PyTorch dispatcher when dealing with functorch
    # transforms, so we manually unwrap the dead tensors here.
    # This logic won't need to exist when we have mode-only functorch.
    args, kwargs = pytree.tree_map_only(
        torch.Tensor, torch._C._functorch.unwrap_if_dead, (args, kwargs)
    )
    return interpreter.process(op, args, kwargs)
<|endoftext|># NOTE: We allow Dynamo to see this file (via torch/_dynamo/trace_rules.py) so that it can
#       trace through functorch transforms.
#       Currently, we can't allow Dynamo to see `eager_transforms.py`/`vmap.py` as that break a lot of thing
#       and there isn't a mechanism to selectively expose only some functions (eg. grad) from a file
#       to Dynamo.

from __future__ import annotations

import functools
from typing import Any, TYPE_CHECKING
from typing_extensions import ParamSpec, TypeVar

from torch._functorch.utils import argnums_t, exposed_in
from torch._functorch.vmap import (
    _check_out_dims_is_int_or_int_pytree,
    _check_randomness_arg,
    _chunked_vmap,
    _process_batched_inputs,
    Callable,
    in_dims_t,
    out_dims_t,
    vmap_impl,
)


if TYPE_CHECKING:
    from collections.abc import Iterable

    import torch

_P = ParamSpec("_P")
_R = TypeVar("_R")

# functools.wraps copies __dict__, but these attrs describe the original
# compiled callable, not the transformed function wrapper.
_DYNAMO_WRAPPER_ATTRS = (
    "_torchdynamo_inline",
    "_torchdynamo_orig_callable",
    "_torchdynamo_wrapper_id",
    "_isolate_recompiles_id",
    "get_compiler_config",
    "aot_compile",
    "_is_torch_compile",
)


def _wraps_without_dynamo_attrs(
    func: Callable[..., Any],
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    def inner(wrapper: Callable[_P, _R]) -> Callable[_P, _R]:
        wrapper = functools.wraps(func)(wrapper)
        for attr in _DYNAMO_WRAPPER_ATTRS:
            wrapper.__dict__.pop(attr, None)
        return wrapper

    return inner


# vmap(func)(inputs) wraps all Tensor inputs to be batched in BatchedTensors,
# sends those into func, and then unwraps the output BatchedTensors. Operations
# on BatchedTensors perform the batched operations that the user is asking for.
#
# vmap's randomness behavior differs from JAX's, which would require a PRNG key
# to be passed everywhere.


@exposed_in("torch.func")
def vmap(
    func: Callable[_P, _R],
    in_dims: in_dims_t = 0,
    out_dims: out_dims_t = 0,
    randomness: str = "error",
    *,
    chunk_size: int | None = None,
) -> Callable[_P, _R]:
    """
    vmap is the vectorizing map; ``vmap(func)`` returns a new function that
    maps ``func`` over some dimension of the inputs. Semantically, vmap
    pushes the map into PyTorch operations called by ``func``, effectively
    vectorizing those operations.

    vmap is useful for handling batch dimensions: one can write a function
    ``func`` that runs on examples and then lift it to a function that can
    take batches of examples with ``vmap(func)``. vmap can also be used to
    compute batched gradients when composed with autograd.

    .. note::
        :func:`torch.vmap` is aliased to :func:`torch.func.vmap` for
        convenience. Use whichever one you'd like.

    Args:
        func (function): A Python function that takes one or more arguments.
            Must return one or more Tensors.
        in_dims (int or nested structure): Specifies which dimension of the
            inputs should be mapped over. ``in_dims`` should have a
            structure like the inputs. If the ``in_dim`` for a particular
            input is None, then that indicates there is no map dimension.
            Default: 0.
        out_dims (int or Tuple[int]): Specifies where the mapped dimension
            should appear in the outputs. If ``out_dims`` is a Tuple, then
            it should have one element per output. Default: 0.
        randomness (str): Specifies whether the randomness in this
            vmap should be the same or different across batches. If 'different',
            the randomness for each batch will be different. If 'same', the
            randomness will be the same across batches. If 'error', any calls to
            random functions will error. Default: 'error'. WARNING: this flag
            only applies to random PyTorch operations and does not apply to
            Python's random module or numpy randomness.
        chunk_size (None or int): If None (default), apply a single vmap over inputs.
            If not None, then compute the vmap :attr:`chunk_size` samples at a time.
            Note that :attr:`chunk_size=1` is equivalent to computing the vmap with a for-loop.
            If you run into memory issues computing the vmap, please try a non-None chunk_size.

    Returns:
        Returns a new "batched" function. It takes the same inputs as
        ``func``, except each input has an extra dimension at the index
        specified by ``in_dims``. It takes returns the same outputs as
        ``func``, except each output has an extra dimension at the index
        specified by ``out_dims``.

    .. warning:
        :func:`vmap` works best with functional-style code. Please do not
        perform any side-effects in ``func``, with the exception of
        in-place PyTorch operations. Examples of side-effects include mutating
        Python data structures and assigning values to variables not captured
        in ``func``.

    One example of using :func:`vmap` is to compute batched dot products. PyTorch
    doesn't provide a batched ``torch.dot`` API; instead of unsuccessfully
    rummaging through docs, use :func:`vmap` to construct a new function.

        >>> torch.dot  # [D], [D] -> []
        >>> batched_dot = torch.func.vmap(torch.dot)  # [N, D], [N, D] -> [N]
        >>> x, y = torch.randn(2, 5), torch.randn(2, 5)
        >>> batched_dot(x, y)

    :func:`vmap` can be helpful in hiding batch dimensions, leading to a simpler
    model authoring experience.

        >>> batch_size, feature_size = 3, 5
        >>> weights = torch.randn(feature_size, requires_grad=True)
        >>>
        >>> def model(feature_vec):
        >>> # Very simple linear model with activation
        >>>     return feature_vec.dot(weights).relu()
        >>>
        >>> examples = torch.randn(batch_size, feature_size)
        >>> result = torch.vmap(model)(examples)

    :func:`vmap` can also help vectorize computations that were previously difficult
    or impossible to batch. One example is higher-order gradient computation.
    The PyTorch autograd engine computes vjps (vector-Jacobian products).
    Computing a full Jacobian matrix for some function f: R^N -> R^N usually
    requires N calls to ``autograd.grad``, one per Jacobian row. Using :func:`vmap`,
    we can vectorize the whole computation, computing the Jacobian in a single
    call to ``autograd.grad``.

        >>> # Setup
        >>> N = 5
        >>> f = lambda x: x**2
        >>> x = torch.randn(N, requires_grad=True)
        >>> y = f(x)
        >>> I_N = torch.eye(N)
        >>>
        >>> # Sequential approach
        >>> jacobian_rows = [torch.autograd.grad(y, x, v, retain_graph=True)[0]
        >>>                  for v in I_N.unbind()]
        >>> jacobian = torch.stack(jacobian_rows)
        >>>
        >>> # vectorized gradient computation
        >>> def get_vjp(v):
        >>>     return torch.autograd.grad(y, x, v)
        >>> jacobian = torch.vmap(get_vjp)(I_N)

    :func:`vmap` can also be nested, producing an output with multiple batched dimensions

        >>> torch.dot  # [D], [D] -> []
        >>> batched_dot = torch.vmap(
        ...     torch.vmap(torch.dot)
        ... )  # [N1, N0, D], [N1, N0, D] -> [N1, N0]
        >>> x, y = torch.randn(2, 3, 5), torch.randn(2, 3, 5)
        >>> batched_dot(x, y)  # tensor of size [2, 3]

    If the inputs are not batched along the first dimension, ``in_dims`` specifies
    the dimension that each inputs are batched along as

        >>> torch.dot  # [N], [N] -> []
        >>> batched_dot = torch.vmap(torch.dot, in_dims=1)  # [N, D], [N, D] -> [D]
        >>> x, y = torch.randn(2, 5), torch.randn(2, 5)
        >>> batched_dot(
        ...     x, y
        ... )  # output is [5] instead of [2] if batched along the 0th dimension

    If there are multiple inputs each of which is batched along different dimensions,
    ``in_dims`` must be a tuple with the batch dimension for each input as

        >>> torch.dot  # [D], [D] -> []
        >>> batched_dot = torch.vmap(torch.dot, in_dims=(0, None))  # [N, D], [D] -> [N]
        >>> x, y = torch.randn(2, 5), torch.randn(5)
        >>> batched_dot(
        ...     x, y
        ... )  # second arg doesn't have a batch dim because in_dim[1] was None

    If the input is a Python struct, ``in_dims`` must be a tuple containing a struct
    matching the shape of the input:

        >>> f = lambda dict: torch.dot(dict["x"], dict["y"])
        >>> x, y = torch.randn(2, 5), torch.randn(5)
        >>> input = {"x": x, "y": y}
        >>> batched_dot = torch.vmap(f, in_dims=({"x": 0, "y": None},))
        >>> batched_dot(input)

    By default, the output is batched along the first dimension. However, it can be batched
    along any dimension by using ``out_dims``

        >>> f = lambda x: x**2
        >>> x = torch.randn(2, 5)
        >>> batched_pow = torch.vmap(f, out_dims=1)
        >>> batched_pow(x)  # [5, 2]

    For any function that uses kwargs, the returned function will not batch the kwargs but will
    accept kwargs

        >>> x = torch.randn([2, 5])
        >>> def fn(x, scale=4.):
        >>>   return x * scale
        >>>
        >>> batched_pow = torch.vmap(fn)
        >>> assert torch.allclose(batched_pow(x), x * 4)
        >>> batched_pow(x, scale=x)  # scale is not batched, output has shape [2, 2, 5]

    .. note::
        vmap does not provide general autobatching or handle variable-length
        sequences out of the box.
    """
    from torch.compiler import is_compiling

    _check_randomness_arg(randomness)
    if not (chunk_size is None or chunk_size > 0):
        raise ValueError(
            f"vmap: chunk_size should be None or greater than 0. (got {chunk_size})"
        )

    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        # pyrefly: ignore[bad-argument-type]
        return vmap_impl(
            # pyrefly: ignore[bad-argument-type]
            func,
            in_dims,
            out_dims,
            randomness,
            chunk_size,
            *args,
            **kwargs,
        )

    if not is_compiling():
        wrapped = _wraps_without_dynamo_attrs(func)(wrapped)

    return wrapped


def chunk_vmap(
    func: Callable[_P, _R],
    in_dims: in_dims_t = 0,
    out_dims: out_dims_t = 0,
    randomness: str = "error",
    chunks: int = 2,
) -> Callable[_P, _R]:
    """
    chunk_vmap is the vectorizing map (vmap) using chunks of input data. It is a mix of vmap (which vectorizes
    everything) and map (which executes things sequentially). ``chunk_vmap`` vectorizes the input with number of
    chunks at a time. For more details about vectorizing map, see :func:`vmap`.

    .. note::
        Please use :func:`vmap` with ``chunk_size`` argument instead of this API.

    Args:
        func (function): A Python function that takes one or more arguments.
            Must return one or more Tensors.
        in_dims (int or nested structure): Specifies which dimension of the
            inputs should be mapped over. ``in_dims`` should have a
            structure like the inputs. If the ``in_dim`` for a particular
            input is None, then that indicates there is no map dimension.
            Default: 0.
        out_dims (int or Tuple[int]): Specifies where the mapped dimension
            should appear in the outputs. If ``out_dims`` is a Tuple, then
            it should have one element per output. Default: 0.
        randomness (str): Specifies whether the randomness in this
            vmap should be the same or different across batches. If 'different',
            the randomness for each batch will be different. If 'same', the
            randomness will be the same across batches. If 'error', any calls to
            random functions will error. Default: 'error'. WARNING: this flag
            only applies to random PyTorch operations and does not apply to
            Python's random module or numpy randomness.
        chunks (int): Number of chunks to use to split the input data. Default is 2.
            If equals to 1 then :func:`vmap` is called.

    Returns:
        Returns a new "batched" function. It takes the same inputs as
        ``func``, except each input has an extra dimension at the index
        specified by ``in_dims``. It takes returns the same outputs as
        ``func``, except each output has an extra dimension at the index
        specified by ``out_dims``.
    """
    _check_randomness_arg(randomness)

    if chunks == 1:
        return vmap(func, in_dims=in_dims, out_dims=out_dims, randomness=randomness)

    def _get_chunk_flat_args(
        flat_args_: Iterable[Any],
        flat_in_dims_: Iterable[int | None],
        chunks_: int,
    ) -> Iterable[Any]:
        flat_args_chunks = tuple(
            t.chunk(chunks_, dim=in_dim)
            if in_dim is not None
            else [
                t,
            ]
            * chunks_
            for t, in_dim in zip(flat_args_, flat_in_dims_)
        )
        # transpose chunk dim and flatten structure
        # chunks_flat_args is a list of flatten args
        chunks_flat_args = zip(*flat_args_chunks)
        return chunks_flat_args

    @_wraps_without_dynamo_attrs(func)
    def wrapped_with_chunks(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        _check_out_dims_is_int_or_int_pytree(out_dims, func)
        _, flat_in_dims, flat_args, args_spec = _process_batched_inputs(
            in_dims, args, func
        )
        # Chunk flat arguments
        chunks_flat_args = _get_chunk_flat_args(flat_args, flat_in_dims, chunks)

        # Apply vmap on chunks
        return _chunked_vmap(
            # pyrefly: ignore[bad-argument-type]
            func,
            flat_in_dims,
            chunks_flat_args,
            args_spec,
            out_dims,
            randomness,
            **kwargs,
        )

    return wrapped_with_chunks


# TODO: Improve the return type of this function
@exposed_in("torch.func")
def grad(
    func: Callable[_P, Any], argnums: argnums_t = 0, has_aux: bool = False
) -> Callable[_P, Any]:
    """``grad`` operator helps computing gradients of ``func`` with respect to the
    input(s) specified by ``argnums``. This operator can be nested to
    compute higher-order gradients.

    Args:
        func (Callable): A Python function that takes one or more arguments.
            Must return a single-element Tensor. If specified ``has_aux`` equals ``True``,
            function can return a tuple of single-element Tensor and other auxiliary objects:
            ``(output, aux)``.
        argnums (int or Tuple[int]): Specifies arguments to compute gradients with respect to.
            ``argnums`` can be single integer or tuple of integers. Default: 0.
        has_aux (bool): Flag indicating that ``func`` returns a tensor and other
            auxiliary objects: ``(output, aux)``. Default: False.

    Returns:
        Function to compute gradients with respect to its inputs. By default, the output of
        the function is the gradient tensor(s) with respect to the first argument.
        If specified ``has_aux`` equals ``True``, tuple of gradients and output auxiliary objects
        is returned. If ``argnums`` is a tuple of integers, a tuple of output gradients with
        respect to each ``argnums`` value is returned.

    Example of using ``grad``:

        >>> # xdoctest: +SKIP
        >>> from torch.func import grad
        >>> x = torch.randn([])
        >>> cos_x = grad(lambda x: torch.sin(x))(x)
        >>> assert torch.allclose(cos_x, x.cos())
        >>>
        >>> # Second-order gradients
        >>> neg_sin_x = grad(grad(lambda x: torch.sin(x)))(x)
        >>> assert torch.allclose(neg_sin_x, -x.sin())

    When composed with ``vmap``, ``grad`` can be used to compute per-sample-gradients:

        >>> # xdoctest: +SKIP
        >>> from torch.func import grad, vmap
        >>> batch_size, feature_size = 3, 5
        >>>
        >>> def model(weights, feature_vec):
        >>> # Very simple linear model with activation
        >>>     assert feature_vec.dim() == 1
        >>>     return feature_vec.dot(weights).relu()
        >>>
        >>> def compute_loss(weights, example, target):
        >>>     y = model(weights, example)
        >>>     return ((y - target) ** 2).mean()  # MSELoss
        >>>
        >>> weights = torch.randn(feature_size, requires_grad=True)
        >>> examples = torch.randn(batch_size, feature_size)
        >>> targets = torch.randn(batch_size)
        >>> inputs = (weights, examples, targets)
        >>> grad_weight_per_example = vmap(grad(compute_loss), in_dims=(None, 0, 0))(
        ...     *inputs
        ... )

    Example of using ``grad`` with ``has_aux`` and ``argnums``:

        >>> # xdoctest: +SKIP
        >>> from torch.func import grad
        >>> def my_loss_func(y, y_pred):
        >>>    loss_per_sample = (0.5 * y_pred - y) ** 2
        >>>    loss = loss_per_sample.mean()
        >>>    return loss, (y_pred, loss_per_sample)
        >>>
        >>> fn = grad(my_loss_func, argnums=(0, 1), has_aux=True)
        >>> y_true = torch.rand(4)
        >>> y_preds = torch.rand(4, requires_grad=True)
        >>> out = fn(y_true, y_preds)
        >>> # > output is ((grads w.r.t y_true, grads w.r.t y_preds), (y_pred, loss_per_sample))

    .. note::
        Using PyTorch ``torch.no_grad`` together with ``grad``.

        Case 1: Using ``torch.no_grad`` inside a function:

            >>> # xdoctest: +SKIP
            >>> def f(x):
            >>>     with torch.no_grad():
            >>>         c = x ** 2
            >>>     return x - c

        In this case, ``grad(f)(x)`` will respect the inner ``torch.no_grad``.

        Case 2: Using ``grad`` inside ``torch.no_grad`` context manager:

            >>> # xdoctest: +SKIP
            >>> with torch.no_grad():
            >>>     grad(f)(x)

        In this case, ``grad`` will respect the inner ``torch.no_grad``, but not the
        outer one. This is because ``grad`` is a "function transform": its result
        should not depend on the result of a context manager outside of ``f``.

    """
    # To avoid cyclical dependency.
    import torch._functorch.eager_transforms as eager_transforms
    from torch.compiler import is_compiling

    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> tuple[Any, torch.Tensor]:
        return eager_transforms.grad_impl(func, argnums, has_aux, args, kwargs)

    if not is_compiling():
        wrapper = _wraps_without_dynamo_attrs(func)(wrapper)

    return wrapper


# TODO: Improve the return type of this function
@exposed_in("torch.func")
def grad_and_value(
    func: Callable[_P, Any], argnums: argnums_t = 0, has_aux: bool = False
) -> Callable[_P, tuple[Any, Any]]:
    """
    Returns a function to compute a tuple of the gradient and primal, or
    forward, computation.

    Args:
        func (Callable): A Python function that takes one or more arguments.
            Must return a single-element Tensor. If specified ``has_aux``
            equals ``True``, function can return a tuple of single-element
            Tensor and other auxiliary objects: ``(output, aux)``.
        argnums (int or Tuple[int]): Specifies arguments to compute gradients
            with respect to. ``argnums`` can be single integer or tuple of
            integers. Default: 0.
        has_aux (bool): Flag indicating that ``func`` returns a tensor and
            other auxiliary objects: ``(output, aux)``. Default: False.

    Returns:
        Function to compute a tuple of gradients with respect to its inputs
        and the forward computation. By default, the output of the function is
        a tuple of the gradient tensor(s) with respect to the first argument
        and the primal computation. If specified ``has_aux`` equals
        ``True``, tuple of gradients and tuple of the forward computation with
        output auxiliary objects is returned. If ``argnums`` is a tuple of
        integers, a tuple of a tuple of the output gradients with respect to
        each ``argnums`` value and the forward computation is returned.

    See :func:`grad` for examples
    """
    from torch._functorch import eager_transforms
    from torch.compiler import is_compiling

    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> tuple[Any, torch.Tensor]:
        return eager_transforms.grad_and_value_impl(
            func, argnums, has_aux, args, kwargs
        )

    if not is_compiling():
        wrapper = _wraps_without_dynamo_attrs(func)(wrapper)

    return wrapper
<|endoftext|>import contextlib
from collections.abc import Generator

from torch._C._functorch import (
    get_single_level_autograd_function_allowed,
    set_single_level_autograd_function_allowed,
    unwrap_dead_wrappers,
)
from torch.utils._exposed_in import exposed_in


__all__ = [
    "exposed_in",
    "argnums_t",
    "enable_single_level_autograd_function",
    "unwrap_dead_wrappers",
]


@contextlib.contextmanager
def enable_single_level_autograd_function() -> Generator[None, None, None]:
    try:
        prev_state = get_single_level_autograd_function_allowed()
        set_single_level_autograd_function_allowed(True)
        yield
    finally:
        set_single_level_autograd_function_allowed(prev_state)


argnums_t = int | tuple[int, ...]
<|endoftext|>from __future__ import annotations

from typing import Any, NamedTuple, TYPE_CHECKING
from typing_extensions import ParamSpec, TypeVar

import torch
import torch.utils._pytree as pytree
from torch._C._functorch import (
    _unwrap_for_grad,
    _wrap_for_grad,
    current_level,
    TransformType,
)
from torch._functorch.apis import vmap
from torch._functorch.utils import enable_single_level_autograd_function
from torch._functorch.vmap import (
    _add_batch_dim,
    _broadcast_to_and_flatten,
    restore_vmap,
    unwrap_batched,
    wrap_batched,
)
from torch._ops import HigherOrderOperator
from torch.autograd.forward_ad import _set_fwd_grad_enabled


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from torch._functorch.pyfunctorch import FuncTorchInterpreter, VmapInterpreter

_P = ParamSpec("_P")
_R = TypeVar("_R")


# autograd.Function technically runs before the regular PyTorch dispatcher.
# This is how features like autocast and torch_dispatch (e.g. PythonTLSSnapshot)
# work with it. One day we might decide to change this, but until then,
# we need to give the illusion that autograd.Function runs before those things.
#
# We do this by using creating a custom HigherOrderOperator that only functorch
# dispatches specially.
class CustomFunctionHigherOrderOperator(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("custom_function_call")

    def __call__(
        self,
        autograd_function: type[torch.autograd.Function],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> Any:
        # When custom_function_call is done dispatching through functorch,
        # it should just invoke the autograd.Function. This is consistent
        # with the autograd.Function behavior of being invoked before the
        # PyTorch dispatcher.
        #
        # This will lead us into trouble later down the line, but this is
        # pre-existing. There is an invariant that a function traced by
        # make_fx should have the same behavior when provided the same
        # Tensor. However, make_fx sees autograd.Function as a composite
        # (because autograd.Function happens before the Python dispatch key)
        # and only traces the forward pass.
        if torch._C._are_functorch_transforms_active():
            # pyrefly: ignore [missing-attribute]
            return super().__call__(autograd_function, *args, **kwargs)
        return autograd_function.apply(*args, **kwargs)


# "custom_function_call"
# This is the mechanism for an autograd.Function that works with functorch transforms.
# It wraps an autograd.Function; interactions with functorch transforms are defined
# via PyDispatcher and HigherOrderOperator rather than through the traditional PyTorch
# dispatcher.
custom_function_call = CustomFunctionHigherOrderOperator()


# The grad rule for custom_function_call is to construct a new _SingleLevelFunction
# (autograd.Function that only works with a single layer (level) of functorch) that:
# - unwraps the inputs
# - redispatches to custom_function_call
# - wraps the outputs
# and whose backward pass calls the original autograd.Function's backward.
#
# Why do we need to redispatch to custom_function_call?
# -----------------------------------------------------
# This is consistent with how ATen operators work with functorch's grad transform:
# they always redispatch to the original operator.
# Consider torch.sin, and let's say we do grad0(grad1(torch.sin))(x)
#
# grad1 will:
# - set up the autograd graph
# - unwrap the inputs
# - redispatch to at::sin (*)
# - rewrap the outputs on the return
#
# On the redispatch in (*), grad0 will:
# - set up the autograd graph
# - unwrap the inputs
# - redispatch to at::sin
# - rewrap the outputs on the return
#
# To "set up the autograd graph", we generate a _SingleLevelFunction
# and apply it.
@custom_function_call.py_impl(TransformType.Grad)
@custom_function_call.py_impl(TransformType.Jvp)
def custom_function_call_grad(
    interpreter: FuncTorchInterpreter,
    autograd_function: type[torch.autograd.Function],
    *operands: Any,
) -> Any:
    Generated = generate_single_level_function(interpreter, autograd_function)
    with enable_single_level_autograd_function():
        # pyrefly: ignore [missing-attribute]
        flat_out = Generated.apply(*operands)
    return flat_out


def generate_single_level_function(
    interpreter: FuncTorchInterpreter,
    autograd_function: type[torch.autograd.Function],
) -> type[torch.autograd.function._SingleLevelFunction]:
    level = interpreter.level()

    def forward(*operands: Any) -> Any:
        unwrapped_operands = pytree.tree_map_only(
            torch.Tensor, lambda x: _unwrap_for_grad(x, level), operands
        )
        # Both enable_grad() and _set_fwd_grad_enabled() are necessary no matter
        # the transform. _SingleLevelFunction will turn off both fwd and bwd
        # gradient computation and we need to turn it back on here.
        with torch.enable_grad(), _set_fwd_grad_enabled(True), interpreter.lower():
            unwrapped_output = custom_function_call(
                autograd_function, *unwrapped_operands
            )

        # See NOTE [mark_dirty object identity check]
        def wrap_fn(output: torch.Tensor) -> torch.Tensor:
            return _wrap_for_grad(output, level)

        return wrap_outputs_maintaining_identity(
            unwrapped_output, unwrapped_operands, operands, wrap_fn
        )

    def setup_context(ctx: Any, inputs: Any, output: Any) -> Any:
        return autograd_function.setup_context(ctx, inputs, output)

    # backward is only used if the transform is TransformType.Grad
    def backward(ctx: Any, *grads: Any) -> Any:
        result = autograd_function.backward(ctx, *grads)
        return result

    # jvp is only used if the transform is TransformType.Jvp
    def jvp(ctx: Any, *tangents: Any) -> Any:
        result = autograd_function.jvp(ctx, *tangents)
        return result

    # This is the sequence of magic words to dynamically generate a Subclass with
    # a given name. A Tensor's .grad_fn field has a class name that is the original
    # autograd.Function's name + Backward, so we do this to generate some
    # meaningful name.
