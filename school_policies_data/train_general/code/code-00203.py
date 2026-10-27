                grids[i] = (grids[i][0], grids[i][1], 1)
            elif len(grids[i]) > 3:
                self.raise_unsupported("Grid can have at most rank 3")

        if len(grids) == 0:
            raise AssertionError("grids cannot be empty")
        if isinstance(variable.kernel, JITFunction):
            constexprs = [p.num for p in variable.kernel.params if p.is_constexpr]
            arg_names = [p.name for p in variable.kernel.params]
        else:
            # If we are looking at an @triton.autotune decorator, the nested function should be a JITFunction
            # This is because we don't support @triton.heuristics or nested @triton.autotune decorators yet
            if not isinstance(variable.kernel, Autotuner):
                raise AssertionError(
                    f"Expected variable.kernel to be an Autotuner, got {type(variable.kernel)}"
                )
            constexprs = [p.num for p in variable.kernel.fn.params if p.is_constexpr]
            arg_names = [p.name for p in variable.kernel.fn.params]

        for idx, arg_name in enumerate(arg_names):
            if idx in constexprs:
                if arg_name in combined_args_raw:
                    # [Note: Specialize tl.constexpr args in user-defined triton kernels]
                    # This arg is marked as tl.constexpr. That means that triton will recompile every time
                    # this value changes.
                    # https://github.com/pytorch/pytorch/issues/136504
                    # One option is to correctly pass the symints in so that the symbolic expressions are defined
                    # when the triton code is being executed.
                    # But since triton will have to recompile either way, we instead just specialize on the value.
                    #
                    # Depending on the type of `variable` we might expect different types for the symbolic args:
                    # either SymNodeVariables (for TritonKernelVariables) or SymInts (TracingTritonKernelWrapper)
                    combined_args_raw[arg_name] = variable.specialize_symbolic(
                        combined_args_raw[arg_name]
                    )
        return self.call_HOP(
            variable, grids, combined_args_raw, launch_kwargs, jit_arg_names, tx
        )


###############################################################################
# Helpers for wrap_triton API that makes a user-defined triton kernel traceable into
# a graph via make_fx or non-strict export (coming soon)


class TracingTritonHOPifier(TritonHOPifier):
    def raise_unsupported(self, msg: str) -> Never:
        raise RuntimeError(msg)

    def is_callable(self, maybe_callable: Any) -> bool:
        return callable(maybe_callable)

    def get_value(self, val: Any) -> Any:
        return val

    def call_grid(
        self,
        grid: "TritonGridCallableType",
        meta: "TritonMetaParamsType",
        tx: None,
    ) -> tuple[int | sympy.Expr | SymInt, ...]:
        if tx is not None:
            raise AssertionError("tx must be None for TracingTritonHOPifier")
        if not isinstance(meta, dict):
            raise AssertionError(f"meta must be a dict, got {type(meta)}")
        if not callable(grid):
            raise AssertionError(f"grid must be callable, got {type(grid)}")
        return grid(meta)

    def wrap_user_defined_obj(
        self,
        user_obj: Any,
        tx: Optional["InstructionTranslatorBase"],
        variable: Union["TritonKernelVariable", "TraceableTritonKernelWrapper"] | None,
        name: str,
    ) -> Any:
        if tx is not None:
            raise AssertionError("tx must be None for TracingTritonHOPifier")
        return user_obj

    def call_user_defined_fn(
        self,
        user_fn: Callable[..., Any],
        args: list,
        kwargs: dict,
        tx: Optional["InstructionTranslatorBase"],
        variable: Union["TritonKernelVariable", "TraceableTritonKernelWrapper"] | None,
    ) -> Any:
        if not isinstance(args, list):
            raise AssertionError(f"args must be a list, got {type(args)}")
        if not isinstance(kwargs, dict):
            raise AssertionError(f"kwargs must be a dict, got {type(kwargs)}")
        if not callable(user_fn):
            raise AssertionError(f"user_fn must be callable, got {type(user_fn)}")
        return user_fn(*args, **kwargs)

    def maybe_unpack_configs(
        self, configs: list["TritonConfig"], tx: Optional["InstructionTranslatorBase"]
    ) -> list["TritonConfig"]:
        if not isinstance(configs, list):
            raise AssertionError(f"configs must be a list, got {type(configs)}")
        return configs

    def maybe_unpack_heuristic_result(self, result: Any) -> Any:
        return result

    def check_grid(
        self,
        grid: "TritonGridType",
    ) -> tuple[int | sympy.Expr | SymInt, ...]:
        if not isinstance(grid, collections.abc.Sequence):
            raise RuntimeError(
                "wrap_triton can only handle grids that resolve to Sequence[int]."
            )
        # normalize to tuple
        return tuple(grid)

    def store_non_graphable_args(
        self,
        combined_args: dict[str, Any],
    ) -> tuple[dict, int]:
        """
        Some args cannot be stored in the FX graph.
        Put them in the side table.
        """

        def is_graphable(val: Any) -> bool:
            return isinstance(val, (fx.node.base_types, fx.Node))

        non_graphable_args = {
            k: v for k, v in combined_args.items() if not is_graphable(v)
        }
        graphable_args = {k: v for k, v in combined_args.items() if is_graphable(v)}

        constant_args_idx = kernel_side_table.add_constant_args(non_graphable_args)

        return graphable_args, constant_args_idx

    def is_dynamic_backend_option(self, value: Any) -> bool:
        # Backend options are compile-time values. In proxy tracing, a value can
        # be a Proxy/Node directly or be nested inside a tuple/list option such
        # as backend_option=(sym_size,). Reject those unless the name is later
        # identified as a real kernel parameter.
        leaves, _ = pytree.tree_flatten(value)
        return any(
            isinstance(
                leaf,
                (
                    fx.Node,
                    fx.Proxy,
                    Tensor,
                    SymInt,
                    SymBool,
                    SymFloat,
                ),
            )
            for leaf in leaves
        )

    def call_HOP(
        self,
        variable: "TraceableTritonKernelWrapper",
        grids: list["TritonGridTupleType"],
        combined_args: dict[str, Any],
        launch_kwargs: tuple[str, ...],
        kernel_arg_names: set[str],
        tx: None,
    ) -> None:
        if tx is not None:
            raise AssertionError("tx must be None for TracingTritonHOPifier")
        if not isinstance(variable, TraceableTritonKernelWrapper):
            raise AssertionError(
                f"Expected TraceableTritonKernelWrapper, got {type(variable)}"
            )

        graphable_args, constant_args_idx = self.store_non_graphable_args(combined_args)
        # launch_kwargs records the names passed as kwargs at the Triton launch
        # site. A non-kernel launch kwarg can only be a compiler option, so it
        # must be concrete before entering the graph. Kernel launch kwargs may
        # also be compiler options, but that target-specific check happens in
        # Inductor after backend.parse_options().
        non_const_options = [
            k
            for k in launch_kwargs
            if k not in kernel_arg_names
            and self.is_dynamic_backend_option(combined_args[k])
        ]
        if non_const_options:
            self.raise_unsupported(
                "Triton backend options must be concrete values: "
                f"{sorted(non_const_options)!r}."
            )

        if not isinstance(variable.kernel_idx, int):
            raise AssertionError(
                f"kernel_idx must be an int, got {type(variable.kernel_idx)}"
            )
        return triton_kernel_wrapper_mutation(
            kernel_idx=variable.kernel_idx,
            constant_args_idx=constant_args_idx,
            grid=grids,  # type: ignore[arg-type]
            # TMA descriptor capturing not yet
            # supported in non-dynamo tracing
            tma_descriptor_metadata={},
            kwargs=graphable_args,
            launch_kwargs=launch_kwargs,
        )


tracing_triton_hopifier_singleton = TracingTritonHOPifier()


class TraceableTritonKernelWrapper:
    kernel: "TritonKernelType"
    kernel_idx: int | None
    grid: Optional["TritonGridType"]
    kernel_source: "Source | None"

    def __init__(
        self,
        kernel: "TritonKernelType",
        kernel_idx: int | None,
        grid: Optional["TritonGridType"],
        kernel_source: "Source | None" = None,
    ) -> None:
        self.kernel = None
        self.grid = None
        self.kernel_source = kernel_source
        tracing_triton_hopifier_singleton.init_variable(self, kernel, kernel_idx, grid)
        if self.kernel is None:
            raise AssertionError("kernel was not initialized properly")

    def __getitem__(self, *args: Sequence[Any]) -> "TraceableTritonKernelWrapper":
        return tracing_triton_hopifier_singleton.call_getitem(self, args)  # type: ignore[return-value]

    def run(self, *args: Sequence[Any], **kwargs: dict[str, Any]) -> Any:
        from torch._library.triton import is_wrap_triton_enabled

        if is_wrap_triton_enabled():
            return tracing_triton_hopifier_singleton.call_run(self, args, kwargs, None)
        else:
            if self.kernel is None:
                raise AssertionError("kernel cannot be None")
            # pyrefly: ignore [missing-attribute]
            return self.kernel.run(*args, **kwargs)

    def __call__(self, *args: Sequence[Any], **kwargs: dict[str, Any]) -> Any:
        from torch._library.triton import is_wrap_triton_enabled

        if is_wrap_triton_enabled():
            return tracing_triton_hopifier_singleton.call_triton_kernel(
                self, args, kwargs, None
            )
        else:
            if self.kernel is None:
                raise AssertionError("kernel cannot be None")
            # pyrefly: ignore [bad-index, index-error]
            return self.kernel[self.grid](*args, **kwargs)

    def specialize_symbolic(self, arg: Sequence[Any]) -> Any:
        import torch

        # See [Note: Specialize tl.constexpr args in user-defined triton kernels]
        if isinstance(arg, (torch.SymInt, torch.SymBool, torch.SymFloat)):
            return guard_scalar(arg)
        return arg
<|endoftext|># mypy: allow-untyped-defs

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from typing import Any, cast

import torch
import torch.utils._pytree as pytree
from torch._higher_order_ops.utils import register_fake
from torch._ops import HigherOrderOperator
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    get_proxy_slot,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.utils._pytree import tree_flatten


class ExecutorchCallDelegate(HigherOrderOperator):
    def __init__(self):
        super().__init__("executorch_call_delegate")

    def __call__(self, lowered_module, *args):
        # pyrefly: ignore [missing-attribute]
        return super().__call__(lowered_module, *args)


executorch_call_delegate = ExecutorchCallDelegate()
executorch_call_delegate.fallthrough(torch._C.DispatchKey.PythonDispatcher)
executorch_call_delegate.fallthrough(torch._C.DispatchKey.PythonTLSSnapshot)
executorch_call_delegate.fallthrough(torch._C.DispatchKey.ADInplaceOrView)
executorch_call_delegate.fallthrough(torch._C.DispatchKey.AutocastCPU)

LOWERED_BACKEND_MODULE_TYPE = "LoweredBackendModule"


# pyre-ignore
def trace_call_delegate(proxy_mode, func_overload, lowered_module, *args):
    # pyre-ignore
    def _unwrap_proxy(e):
        if not isinstance(e, (torch.Tensor, torch.SymInt, torch.SymFloat)):
            return e
        return get_proxy_slot(
            cast(torch.Tensor, e),
            proxy_mode.tracer,
            e,
            lambda e: e.proxy,  # type: ignore[attr-defined]
        )

    if not is_lowered_module(lowered_module):
        raise ValueError(
            "executorch_call_delegate()'s first argument must be a LoweredBackendModule"
        )

    with disable_proxy_modes_tracing():
        out = call_delegate_cpu(lowered_module, *args)

    get_lowered_module_name(proxy_mode.tracer.root, lowered_module)

    node_args = (lowered_module, *args)
    proxy_args = pytree.tree_map(_unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="executorch_call_delegate"
    )
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@executorch_call_delegate.py_impl(torch._C.DispatchKey.CompositeExplicitAutograd)
# pyre-ignore
def call_delegate_cpu(lowered_module, *args):
    # FX creates this immutable_dict/list concept. Get rid of this.
    map_types: dict[type, type] = {
        torch.fx.immutable_collections.immutable_dict: dict,
        torch.fx.immutable_collections.immutable_list: list,
    }
    new_args = pytree.tree_map_only(
        tuple(map_types.keys()),
        lambda a: map_types[type(a)](a),
        args,
        lambda a: isinstance(a, tuple(map_types.keys())),
    )
    return lowered_module.original_module.module()(*new_args)


@executorch_call_delegate.py_autograd_impl
# pyre-ignore
def call_delegate_autograd(lowered_module, *args):
    # TODO: support autograd
    flat_operands, _ = tree_flatten([lowered_module, *args])
    requires_grad = any(
        f.requires_grad for f in flat_operands if isinstance(f, torch.Tensor)
    )

    with torch._C._ExcludeDispatchKeyGuard(
        torch._C.DispatchKeySet(torch._C.DispatchKey.AutogradCPU)
    ):
        res = executorch_call_delegate(lowered_module, *args)

        if requires_grad:
            # Create aliases of the output that has requires_grad=True. We need
            # at least one of the inputs to err_fn to require grad so that the
            # output will have a grad_fn.

            # pyre-ignore
            def fake_requires_grad(var):
                if var is not None:
                    var = var.detach()
                    if torch.is_floating_point(var) or torch.is_complex(var):
                        var.requires_grad = True
                return var

            return pytree.tree_map_only(torch.Tensor, fake_requires_grad, res)

        return res


@executorch_call_delegate.py_impl(ProxyTorchDispatchMode)
# pyre-ignore
def call_delegate_proxy_torch_dispatch_mode(mode, lowered_module, *args):
    res = trace_call_delegate(mode, executorch_call_delegate, lowered_module, *args)
    return res


@register_fake(executorch_call_delegate, skip_cache=True)
# pyre-ignore
def call_delegate_fake_tensor_mode(lowered_module, *args):
    return call_delegate_cpu(lowered_module, *args)


@executorch_call_delegate.py_functionalize_impl
# pyre-ignore
def call_delegate_functionalize(ctx, lowered_module, *args):
    unwrapped_args = tuple(ctx.unwrap_tensors(arg) for arg in args)
    with ctx.redispatch_to_next():
        res = executorch_call_delegate(lowered_module, *unwrapped_args)
        return ctx.wrap_tensors(res)


# pyre-ignore: Missing parameter annotation [2]: Parameter `obj` must have a type other than `Any`.Pyre
def is_lowered_module(obj: Any) -> bool:
    """
    This function is added to avoid using isinstance(obj,
    LoweredBackendModule) as it will import LoweredBackendModule, which may
    cause a circular import.
    """
    return type(obj).__name__ == LOWERED_BACKEND_MODULE_TYPE


def get_lowered_module_name(
    root: torch.nn.Module,
    # pyre-ignore: Undefined or invalid type [11]: Annotation `LoweredBackendModule` is not defined as a type.
    lowered_module: LOWERED_BACKEND_MODULE_TYPE,  # type: ignore[valid-type]
) -> str:
    """
    Adds the given lowered_module into the given root module and returns the
    name of the module added.
    """
    # Find a qualifying name for the lowered submodule
    qualname = None
    i = 0
    while True:
        qualname = f"lowered_module_{i}"
        if not hasattr(root, qualname):
            break
        i += 1
    if qualname is None:
        raise AssertionError(
            "qualname must not be None after finding unused module slot"
        )

    root.add_module(qualname, lowered_module)
    return qualname
<|endoftext|># mypy: allow-untyped-defs
import functools
from collections.abc import Callable, Sequence
from typing import Any
from typing_extensions import TypeVarTuple

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._dispatch.python import suspend_functionalization
from torch._higher_order_ops.auto_functionalize import (
    can_auto_functionalize,
    do_auto_functionalize_v2,
)
from torch._higher_order_ops.utils import (
    _from_fun,
    _maybe_run_with_interpreter,
    _stack_pytree,
    _unstack_pytree,
    create_bw_fn,
    fill_none_with_masks,
    filter_with_masks,
    first_slice_copy,
    get_graph_output_example_values,
    HopInstance,
    materialize_as_graph,
    reenter_make_fx,
    register_fake,
    save_values_for_backward,
    saved_values,
    split_into_chunks,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.functional_tensor import disable_functional_mode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)


class MapImpl(HigherOrderOperator):
    def __init__(self):
        super().__init__("map_impl")

    def __call__(
        self,
        f: Callable[..., Any],
        xs: Sequence[torch.Tensor],
        pos_args: Sequence[torch.fx.node.BaseArgumentTypes],
        *,
        mutated_arg_indices: str = "",
    ) -> Any:
        # pyrefly: ignore [missing-attribute]
        return super().__call__(
            f, xs, pos_args, mutated_arg_indices=mutated_arg_indices
        )

    # pyrefly: ignore [bad-override]
    def gen_schema(self, f, xs, pos_args, mutated_arg_indices: str = ""):
        from torch._higher_order_ops.schema import HopSchemaGenerator

        all_inputs = tuple(
            [
                torch.empty_strided(
                    x.shape[1:],
                    x.stride()[1:],
                    dtype=x.dtype,
                    device=x.device,
                    requires_grad=x.requires_grad,
                )
                for x in xs
            ]
            + list(pos_args)
        )
        body_gm = materialize_as_graph(f, all_inputs)

        # Mutation semantics for map:
        # - xs is mutable: each iteration sees a storage-disjoint slice
        #   (xs[t] and xs[t+1] share no storage), so in-place writes are
        #   race-free regardless of evaluation order, preserving map's
        #   "iterations are independent" contract.
        # - pos_args is NOT mutable: every iteration sees the same
        #   tensor, so mutating it makes iterations depend on each
        #   other, breaking the independence contract and introducing a
        #   data race under any parallel lowering. Users who need a
        #   shared mutable buffer should use scan / while_loop, where
        #   sequential iteration is part of the contract.
        outputs = get_graph_output_example_values(body_gm)
        mutated_set = {int(i) for i in mutated_arg_indices.split(",") if i}

        schema_gen = HopSchemaGenerator(self)
        schema_gen.add_arg("f", body_gm)

        for idx, x in enumerate(xs):
            schema_gen.add_arg(f"xs{idx}", x, is_mutated=idx in mutated_set)

        for idx, arg in enumerate(pos_args):
            schema_gen.add_arg(f"additional_input{idx}", arg)

        for out in outputs:
            schema_gen.add_output(out)

        schema_gen.add_schema_tree_spec(f, xs, pos_args)
        return schema_gen.gen_schema()


map_impl = MapImpl()


def map(
    f: Callable[[pytree.PyTree, tuple[pytree.PyTree, ...]], pytree.PyTree],
    xs: pytree.PyTree | torch.Tensor,
    *args: TypeVarTuple,
):
    r"""
    Performs a map of f with xs. Intuitively, you can think of the semantic being::

        out = []
        for idx in len(xs.size(0)):
            xs_sliced = xs.select(0, idx)
            out.append(f(xs_sliced, *args))
        torch.stack(out)

    .. warning::

        ``torch._higher_order_ops.map`` is a prototype feature in PyTorch. It currently
        does not support autograd and you may run into miscompiles.
        Read more about feature classification at:
        https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype


    Args:
        f (Callable): a callable that takes an input x, that could either be a single Tensor
            or a nested dict, list of tensors and some additional inputs
        xs: the inputs that're to be mapped over. We'll iterate over the first dim of each x
            and perform f on each slice.

        *args: additional arguments provided to each step of f. They could also be omitted and
            map is able to automatically figure out the read dependency.

    Return:
        the stacked output for each step of f

    Example::

        def f(xs):
            return xs[0] + xs[1] + const1 + const2


        xs = [torch.randn(2, 3), torch.randn(2, 3)]
        const1 = torch.randn(2, 3)
        const2 = torch.randn(2, 3)
        # returns a tensor of shape [2, 2, 3]
        torch._higher_order_ops.map(f, xs)

    """
    flat_xs, xs_spec = pytree.tree_flatten(xs)
    flat_args, args_spec = pytree.tree_flatten(args)
    if not all(isinstance(t, torch.Tensor) for t in flat_xs):
        raise RuntimeError(f"Mapped xs can only consist of tensors. Got xs {flat_xs}.")

    shapes = [xs.shape for xs in flat_xs]
    leading_dim_size = shapes[0][0]
    if leading_dim_size == 0:
        raise RuntimeError("Leading dimensions of mapped xs cannot be 0.")

    if any(cur_shape[0] != leading_dim_size for cur_shape in shapes):
        raise RuntimeError(
            f"Leading dimensions of mapped xs must be consistent. Got shapes {shapes}."
        )

    def run_flattened_map(f, flat_xs, flat_args):
        def wrapped_fn(*flat_args, f, xs_tree_spec, args_tree_spec, num_xs):
            xs = pytree.tree_unflatten(flat_args[:num_xs], xs_tree_spec)
            args = pytree.tree_unflatten(flat_args[num_xs:], args_tree_spec)
            return f(xs, *args)

        inner_f = functools.partial(
            wrapped_fn,
            f=f,
            xs_tree_spec=xs_spec,
            args_tree_spec=args_spec,
            num_xs=len(flat_xs),
        )
        return map_impl(inner_f, flat_xs, flat_args)

    from torch._higher_order_ops.utils import _maybe_compile_and_run_fn

    return _maybe_compile_and_run_fn(run_flattened_map, f, flat_xs, flat_args)


class MapAutogradOp(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, f, num_mapped_args, *flat_args):
        ctx._f = f
        ctx._num_mapped_args = num_mapped_args
        ctx._num_pos_args = len(flat_args) - num_mapped_args

        # We snapshot the dispatch keys in forward for materializing
        # the bw_graph in backward.
        ctx._fw_include_key_set = torch._C._dispatch_tls_local_include_set()
        ctx._fw_exclude_key_set = torch._C._dispatch_tls_local_exclude_set()
        save_values_for_backward(ctx, flat_args)
        with torch._C._AutoDispatchBelowAutograd():
            return (
                *map_impl(f, flat_args[:num_mapped_args], flat_args[num_mapped_args:]),
            )

    @staticmethod
    def backward(ctx, *flat_grads):
        fw_args = saved_values(ctx)
        num_mapped_args = ctx._num_mapped_args
        num_pos_args = ctx._num_pos_args
        num_grads = len(flat_grads)

        fw_mapped_args, pos_args = split_into_chunks(
            fw_args,
            [
                num_mapped_args,
                num_pos_args,
            ],
        )

        bw_f = create_bw_fn(ctx._f, fw_args)

        grads_tensor_masks = []

        # Create a wrapper around thefor the bw_f
        def bw_f_wrapper(*args):
            nonlocal grads_tensor_masks

            # Dissect args and re-order them for the ``ctx._bw_f``
            # args provided to the wrapper are composed of [*fw_mapped_args, *flat_grads, *pos_args]
            # The content of ``bw_f_tangents`` are the upstream gradients, i.e. flat_grads
            # The content of ``bw_f_primals`` are the fw_args, i.e., [*fw_mapped_args, *pos_args]
            # The bw_f requires *bw_f_primals, *bw_f_tangents
            fw_m_args, bw_f_tangents, pos_args = split_into_chunks(
                args, [num_mapped_args, num_grads, num_pos_args]
            )
            bw_f_primals = *fw_m_args, *pos_args
            gradients = bw_f(*bw_f_primals, *bw_f_tangents)
            grads_tensor_masks = [
                True if isinstance(out, torch.Tensor) else out for out in gradients
            ]
            return filter_with_masks(gradients, grads_tensor_masks)

        def construct_args_single_step_bw():
            unwrapped_mapped_xs = pytree.tree_map(_from_fun, fw_mapped_args)
            # Use first_slice_copy instead of _unstack_pytree to avoid
            # iterating over batch dim, which would guard on symbolic sizes.
            example_xs = [
                first_slice_copy(x).requires_grad_(x.requires_grad)
                for x in unwrapped_mapped_xs
            ]
            unwrapped_grads = pytree.tree_map(_from_fun, flat_grads)
            example_grads = [
                first_slice_copy(x).requires_grad_(x.requires_grad)
                for x in unwrapped_grads
            ]
            example_pos_args = [
                _from_fun(arg) if isinstance(arg, torch.Tensor) else arg
                for arg in pos_args
            ]
            return *example_xs, *example_grads, *example_pos_args

        with suspend_functionalization(), disable_functional_mode():
            with disable_proxy_modes_tracing():
                args_single_step_bw = construct_args_single_step_bw()

            # TODO: we need to materialize the bw graphs because dynamo is unable to
            # trace through the joint function when torch.compile torch.autograd.grad.
            fn_bw_gm = materialize_as_graph(
                bw_f_wrapper,
                args_single_step_bw,
                ctx._fw_include_key_set,
                ctx._fw_exclude_key_set,
                force_enable_grad=True,
            )

        grads = map_impl(fn_bw_gm, fw_mapped_args + flat_grads, pos_args)

        return None, None, *fill_none_with_masks(grads, grads_tensor_masks)


def _broadcast_to_batch(output, batch_size):
    """Expand each tensor in output pytree to include batch dimension.

    Note: Although outputs are flattened in the compiled path (via torch.compile),
    users may call map_impl directly with pytree outputs, so we support pytrees
    for backward compatibility.
    """

    def expand_with_batch(t):
        if isinstance(t, torch.Tensor):
            # Use contiguous_format to match torch.stack behavior
            return (
                t.unsqueeze(0)
                .expand(batch_size, *t.shape)
                .clone(memory_format=torch.contiguous_format)
            )
        return t

    return pytree.tree_map(expand_with_batch, output)


def trace_map(
    proxy_mode,
    func_overload: HigherOrderOperator,
    f: Callable[..., Any],
    xs: Sequence[torch.Tensor],
    pos_args: Sequence[torch.fx.node.BaseArgumentTypes],
    mutated_arg_indices: str = "",
) -> Any:
    with disable_proxy_modes_tracing():
        # Use first_slice_copy instead of _unstack_pytree to avoid
        # iterating over batch dim, which would guard on symbolic sizes.
        example_input = pytree.tree_map(first_slice_copy, xs)

        body_graph = reenter_make_fx(f)(*example_input, *pos_args)

    next_name = proxy_mode.tracer.get_fresh_qualname("body_graph_")
    proxy_mode.tracer.root.register_module(next_name, body_graph)

    fake_outs = map_impl(
        body_graph, xs, pos_args, mutated_arg_indices=mutated_arg_indices
    )

    node_args = (body_graph, list(xs), list(pos_args))
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function",
        func_overload,
        proxy_args,
        {"mutated_arg_indices": mutated_arg_indices} if mutated_arg_indices else {},
        name="map_impl",
    )
    return track_tensor_tree(
        fake_outs, out_proxy, constant=None, tracer=proxy_mode.tracer
    )


@map_impl.py_impl(DispatchKey.CompositeExplicitAutograd)
def map_dense(f, xs, pos_args, mutated_arg_indices=""):
    pytrees = [f(*inp, *pos_args) for inp in _unstack_pytree(xs)]
    return _stack_pytree(pytrees)


@map_impl.py_autograd_impl
def map_autograd(f, xs, pos_args, mutated_arg_indices=""):
    num_mapped_args = len(xs)
    flat_out = MapAutogradOp.apply(f, num_mapped_args, *xs, *pos_args)
    return flat_out


@map_impl.py_impl(ProxyTorchDispatchMode)
def map_proxy_torch_dispatch_mode(mode, f, xs, pos_args, mutated_arg_indices=""):
    return trace_map(
        mode, map_impl, f, xs, pos_args, mutated_arg_indices=mutated_arg_indices
    )


@register_fake(map_impl, skip_cache=True)
def map_fake_tensor_mode(f, xs, pos_args, mutated_arg_indices=""):
    # Use first_slice_copy instead of _unstack_pytree to avoid
    # iterating over batch dim, which would guard on symbolic sizes.
    first_row = pytree.tree_map(first_slice_copy, xs)
    example_output = f(*first_row, *pos_args)

    flat_xs, _ = pytree.tree_flatten(xs)
    batch_size = flat_xs[0].shape[0]

    return _broadcast_to_batch(example_output, batch_size)


@map_impl.py_functionalize_impl
def map_functionalize(ctx, f, xs, pos_args, mutated_arg_indices=""):
    from torch._higher_order_ops.utils import _check_alias_and_mutation

    if hasattr(ctx, "mode"):
        hop_instance = HopInstance.create(
            map_impl,
            f,
            xs,
            pos_args,
            mutated_arg_indices=mutated_arg_indices,
        )
        if can_auto_functionalize(hop_instance):
            return do_auto_functionalize_v2(
                ctx.mode,
                hop_instance,
                tuple(pytree.tree_flatten((f, xs, pos_args))[0]),
                {},
            )

    unwrapped_xs = ctx.unwrap_tensors(xs)
    unwrapped_args = ctx.unwrap_tensors(pos_args)
    wrapped_fn = ctx.functionalize(_maybe_run_with_interpreter(f))

    with ctx.redispatch_to_next():
        # Use first_slice_copy instead of _unstack_pytree to avoid
        # iterating over batch dim, which would guard on symbolic sizes.
        example_inputs = (
            *pytree.tree_map(first_slice_copy, unwrapped_xs),
            *unwrapped_args,
        )
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        _check_alias_and_mutation(f, example_inputs, "map", pre_dispatch)
        map_return = map_impl(
            wrapped_fn,
            unwrapped_xs,
            unwrapped_args,
            mutated_arg_indices=mutated_arg_indices,
        )
        return ctx.wrap_tensors(map_return)


def _fake_map(f, x, *args):
    from functorch.experimental.control_flow import _stack_pytree, _unstack_pytree

    x_pytrees = _unstack_pytree(x)
    zs = []
    for xp in x_pytrees:
        zs.append(f(xp, *args))
    return _stack_pytree(zs)
<|endoftext|># mypy: allow-untyped-defs
import functools
import itertools
from collections.abc import Callable
from typing import Any

import torch
import torch._prims_common as utils
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import (
    _maybe_compile_and_run_fn,
    _maybe_run_with_interpreter,
    check_input_alias_and_mutation_return_outputs,
    check_meta_consistency,
    create_bw_fn,
    first_slice_copy,
    first_slice_copy_with_grad,
    materialize_as_graph,
    reenter_make_fx,
    register_fake,
    save_values_for_backward,
    saved_values,
    split_into_chunks,
    unique_graph_id,
    validate_subgraph_args_types,
)
from torch._ops import HigherOrderOperator
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)


aten = torch._ops.ops.aten


def wrap_combine_fn_flat(*args, combine_fn, spec, num_leaves):
    if len(args) != 2 * num_leaves:
        raise AssertionError(
            f"Combine_fn received wrong number of arguments, expected {2 * num_leaves}, but got {len(args)}"
        )
    lhs = pytree.tree_unflatten(args[:num_leaves], spec)
    rhs = pytree.tree_unflatten(args[num_leaves:], spec)
    return combine_fn(lhs, rhs)


def _interleave(a, b, dim=0):
    # https://stackoverflow.com/questions/60869537/how-can-i-interleave-5-pytorch-tensors
    if b_trunc := (a.shape[dim] == b.shape[dim] + 1):
        pad = (
            [0] * ((b.ndim - dim - 1) * 2 + 1)
            + [1]
            + [0] * (b.ndim * 2 - ((b.ndim - dim - 1) * 2 + 2))
        )
        b = torch.nn.functional.pad(b, pad)

    stacked = torch.stack([a, b], dim=dim + 1)
    interleaved = torch.flatten(stacked, start_dim=dim, end_dim=dim + 1)
    # pyrefly: ignore [unbound-name]
    if b_trunc:
        # TODO: find torch alternative for slice_along dim for torch.jit.script to work
        interleaved = aten.slice(interleaved, dim, 0, b.shape[dim] + a.shape[dim] - 1)
    return interleaved


def safe_map(f, *args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        if len(arg) != n:
            raise ValueError("length mismatch: {list(map(len, args))}")

    def nf(a):
        return f(*a)

    return list(map(nf, zip(*args)))


class AssociativeScanOp(HigherOrderOperator):
    def __init__(self):
        super().__init__("associative_scan")

    def __call__(self, combine_fn, xs, additional_inputs):
        # There is currently an issue that the ScanOp is sometimes called with
        # the additional_inputs being a list. See https://github.com/pytorch/pytorch/issues/145785
        # Once this issue is resolved, the assertion should only allow tuples
        # and the tuple cast should be removed
        if not isinstance(additional_inputs, (tuple, list)):
            raise AssertionError(
                f"additional_inputs must be a tuple or list, got {type(additional_inputs)}"
            )
        additional_inputs = (
            tuple(additional_inputs)
            if isinstance(additional_inputs, list)
            else additional_inputs
        )
        validate_subgraph_args_types(additional_inputs)
        # pyrefly: ignore [missing-attribute]
        return super().__call__(combine_fn, xs, additional_inputs)

    # pyrefly: ignore [bad-override]
    def gen_schema(self, combine_fn, xs, additional_inputs):
        from torch._higher_order_ops.schema import HopSchemaGenerator

        # For associative scan, we need two copies of xs for the combine function
        # The combine function takes two elements and returns one element
        xs_slice1 = [first_slice_copy(x) for x in xs]
        xs_slice2 = [first_slice_copy(x) for x in xs]
        all_inputs = tuple(xs_slice1 + xs_slice2 + list(additional_inputs))

        combine_gm: torch.fx.GraphModule = materialize_as_graph(combine_fn, all_inputs)
        (
            _,
            _,
            _,
            mutated_inputs,
            outputs,
        ) = check_input_alias_and_mutation_return_outputs(combine_gm)
        if len(mutated_inputs) > 0:
            raise RuntimeError(
                "For associative_scan, combine_fn cannot have in-place mutations but found "
                f"{mutated_inputs}-th inputs are mutated."
            )

        schema_gen = HopSchemaGenerator(self)
        schema_gen.add_arg("combine_fn", combine_gm)

        for idx, x in enumerate(xs):
            schema_gen.add_arg(f"xs{idx}", x)

        for idx, arg in enumerate(additional_inputs):
            schema_gen.add_arg(
                f"additional_input{idx}",
                arg,
            )

        for out in outputs:
            schema_gen.add_output(out)

        schema_gen.add_schema_tree_spec(combine_fn, xs, additional_inputs)
        return schema_gen.gen_schema()


associative_scan_op = AssociativeScanOp()


def associative_scan(
    combine_fn: Callable[[pytree.PyTree, pytree.PyTree], pytree.PyTree],
    xs: pytree.PyTree,
    dim: int,
    reverse: bool = False,
    combine_mode: str = "pointwise",
) -> torch.Tensor:
    r"""
    Performs an inclusive scan with an associative combine function.

    .. warning::

        ``torch.associative_scan`` is a prototype feature in PyTorch. It currently
        does not support autograd and you may run into miscompiles.
        Read more about feature classification at:
        https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    With ``combine_mode="pointwise"``, efficient execution requires runtime code
    generation via ``torch.compile``, and codegen is only supported on backends
    with scan support (currently CUDA and XPU). On other devices the operator
    still runs eagerly via the generic fallback.

    Args:
        combine_fn (Callable): A binary callable with type ``(Tensor, Tensor) -> Tensor``,
            or if input is a pytree ``(pytree, pytree) -> pytree``.
            This function must be pure, i.e., no lifted arguments are supported at the moment,
            satisfy the associative property and have no side-effects.
        xs (torch.Tensor): The input tensor, or nested pytree of tensors.
        dim (int): the dimension to scan over
        reverse (bool): A boolean stating if the scan should be reversed with respect to ``dim``, default ``False``.
        combine_mode (str): A string indicating whether the ``combine_fn`` is ``pointwise`` or ``generic``, default ``pointwise``.
            If ``combine_mode=pointwise``, ``combine_fn`` must be pure and may only contain pointwise
            operations; under ``torch.compile`` ``xs`` must be on a backend with scan codegen support
            (CUDA or XPU), otherwise the generic fallback is used.
            In all other cases ``combine_mode=generic`` should be used.
            Note: ``combine_mode=pointwise`` is more efficient than ``combine_mode=generic``.

    Returns:
        A pytree of the same structure and shape as ``xs``. If the scan dimension has size 0,
        the output mirrors the (empty) input unchanged. The gradient with respect to ``xs``
        is also empty (size 0 along ``dim``), since there are no elements to differentiate through.

    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            return x + y


        cumsum = associative_scan(add, x, dim)

    """
    # TODO: Support lifted arguments in inductor for associative_scan
    # TODO: Support autograd for cases with lifted arguments for combine_mode=pointwise

    # The reason we flatten xs before calling into dynamo is that
    # we want to create a consistent input ordering for combine_fn
    # and we also want to the input ordering matches the output ordering.
    leaves_xs_orig, spec_xs = pytree.tree_flatten(xs)

    def _validate_input(cfn, lxs, d, r, cm):
        # Basic arguments check
        if not callable(cfn):
            raise ValueError(f"Combine_fn must be a callable, but got {cfn}")
        if not isinstance(d, int):
            raise ValueError("Dim must be an int, but got " + str(type(d)))
        if not isinstance(r, bool):
            raise RuntimeError("Reverse must be a bool, but got " + str(type(r)))
        if cm not in ["pointwise", "generic"]:
            raise ValueError(
                f"Combine_mode must either 'pointwise' or 'generic', but got {cm}"
            )

        # Checks for xs
        if len(lxs) == 0:
            raise ValueError("Expected at least 1 xs leaf")
        if any(not isinstance(x, torch.Tensor) for x in lxs):
            raise ValueError("xs leaves must be a Tensor")
        if any(x.is_sparse for x in lxs):
            raise ValueError(
                "xs leaves must dense Tensors, consider using `to_dense()`"
            )
        if any(x.ndim <= d for x in lxs):
            raise ValueError("All xs leaves must have at least 'dim + 1' dimensions")

    ndim = leaves_xs_orig[0].ndim
    dim = utils.canonicalize_dim(ndim, dim)

    _validate_input(combine_fn, leaves_xs_orig, dim, reverse, combine_mode)

    # Move scan dim to 0 and always perform scan on dim 0
    leaves_xs = [torch.movedim(elem, dim, 0) for elem in leaves_xs_orig]

    if reverse:
        leaves_xs = [torch.flip(elem, [0]) for elem in leaves_xs]

    if combine_mode == "generic":
        # The generic_associative_scan implementation calls the combine_fn with a `batch` along the scan dimension
        # For example, consider:
        # def add(x: torch.Tensor, y: torch.Tensor):
        #     return x + y
        # leaves = torch.tensor([[0.0, 1.0, 2.0, 3.0]
        #                        [0.0, 1.0, 2.0, 3.0]])
        # which has shape 2 x 4;
        # dim = 1;
        # In the first iteration of `_scan` the combine_fn gets invoked with
        # combine_fn([torch.tensor([[0.0, 2.0],
        #                           [0.0, 2.0]])],
        #            [torch.tensor([[1.0, 3.0],
        #                           [1.0, 3.0]])])
        # The arguments are of shape 2 x 2, but can be evaluated in parallel along the scan dimension.
        combine_fn = functools.partial(
            wrap_combine_fn_flat,
            combine_fn=torch.vmap(
                combine_fn,
                in_dims=(
                    pytree.tree_unflatten([0] * len(leaves_xs), spec_xs),
                    pytree.tree_unflatten([0] * len(leaves_xs), spec_xs),
                ),
                out_dims=0,
            ),
            spec=spec_xs,
            num_leaves=len(leaves_xs),
        )
        out = generic_associative_scan(combine_fn, leaves_xs, additional_inputs=())
        out = pytree.tree_unflatten(out, spec_xs)
    else:
        combine_fn = functools.partial(
            wrap_combine_fn_flat,
            combine_fn=combine_fn,
            spec=spec_xs,
            num_leaves=len(leaves_xs),
        )

        def run_flattened_associative_scan(combine_fn, leaves_xs):
            return associative_scan_op(combine_fn, leaves_xs, additional_inputs=())

        out = _maybe_compile_and_run_fn(
            run_flattened_associative_scan,
            combine_fn,
            leaves_xs,
        )

    if reverse:
        out = pytree.tree_map(lambda elem: elem.flip([0]), out)

    out = pytree.tree_map(lambda elem: torch.movedim(elem, 0, dim), out)

    return out


def generic_associative_scan(operator, leaves, dim=0, additional_inputs=()):
    r"""
    This function performs the associative_scan operation.
    The algorithm works by recursively collecting neighbours of ``leaves`` and subsequently
    applying the ``operator`` on all pairs in parallel along ``dim``.
    The results of the recursive calls are later combined.

    Args:
        operator (Callable): A binary callable with type ``(Tensor, Tensor) -> Tensor``,
            or if input is a pytree ``(pytree, pytree) -> pytree``.
            This function must be pure, pointwise, and satisfy the associative property.
        leaves (torch.Tensor): A list of torch.Tensors converted from the pytree of
            ``xs`` provided to ``associative_scan``.
            All inputs are expected to have the same shape.
        dim (int): the dimension to scan over
        additional_inputs (Tuple of tensors): A tuple of lifted parameters from the global scope.
            This parameter will be populated internally.

    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            return x + y

        leaves = torch.tensor([0.0, 1.0, 2.0, 3.0])

        First iteration of _scan ->
            # odd_elems -> apply operator on all neighbours
            # odd_elems = operator([torch.tensor([0.0, 2.0])],
            #                      [torch.tensor([1.0, 3.0])])
            odd_elems = torch.tensor([1.0, 5.0])
            Second iteration of _scan ->
                # odd_elems = operator([torch.tensor([1.0])],
                #                      [torch.tensor([5.0])])
                odd_elems = torch.tensor([6.0])
                # even_elems -> apply operator on all odd_elems and
                # every second element of ``elems``, starting from the second element.
                # even_elems is expanded with the first element of ``elems``
                even_elems = [1.0]
                # Merges odd_elems and even_elems
                res = torch.tensor([1.0, 6.0])
            # even_elems -> apply operator on all odd_elems and
            # every second element of ``elems``, starting from the second element.
            # even_elems is expanded with the first element of ``elems``
            even_elems = [0.0, 3.0]
            # Merges odd_elems and even_elems
            res = torch.tensor([0.0, 1.0, 3.0, 6.0])

    """

    def call_operator(*args):
        return pytree.tree_leaves(operator(*args))

    def _scan(elems):
        """Perform the actual recursive scan on ``elems``."""
        num_elems = elems[0].shape[dim]

        if num_elems < 2:
            return elems

        reduced_elems = call_operator(
            *[aten.slice(elem, dim, 0, -1, 2) for elem in elems],
            *[aten.slice(elem, dim, 1, None, 2) for elem in elems],
            *additional_inputs,
        )

        # Recursively compute scan for partially reduced tensors.
        odd_elems = _scan(reduced_elems)

        if num_elems % 2 == 0:
            even_elems = call_operator(
                *[aten.slice(e, dim, 0, -1) for e in odd_elems],
                *[aten.slice(e, dim, 2, None, 2) for e in elems],
                *additional_inputs,
            )
        else:
            even_elems = call_operator(
                *odd_elems,
                *[aten.slice(e, dim, 2, None, 2) for e in elems],
                *additional_inputs,
            )

        # The first element of a scan is the same as the first element
        # of the original `elems`.
        even_elems = [
            torch.cat([aten.slice(elem, dim, 0, 1), result], dim=dim)
            if result.shape.numel() > 0 and elem.shape[dim] > 0
            else result
            if result.shape.numel() > 0
            else aten.slice(
                elem, dim, 0, 1
            )  # Jax allows/ignores concat with 0-dim, Pytorch does not
            for (elem, result) in zip(elems, even_elems)
        ]

        return list(
            safe_map(functools.partial(_interleave, dim=dim), even_elems, odd_elems)
        )

    scans = _scan(leaves)

    return scans


def trace_associative_scan(
    proxy_mode,
    func_overload,
    combine_fn: Callable,
    xs: list[torch.Tensor],
    additional_inputs: tuple[torch.Tensor],
):
    from torch._dynamo.utils import clone_input

    with disable_proxy_modes_tracing():
        sample_xs = [first_slice_copy(x) for x in itertools.chain(xs, xs)]
        sample_additional_inputs = [
            clone_input(x) if isinstance(x, torch.Tensor) else x
            for x in additional_inputs
        ]
        combine_graph = reenter_make_fx(combine_fn)(
            *sample_xs, *sample_additional_inputs
        )

    outputs = None
    for node in combine_graph.graph.nodes:
        if node.op == "output":
            if outputs is not None:
                raise AssertionError("found multiple output nodes in combine_graph")
            if len(node.args) != 1:
                raise AssertionError(
                    f"expected output node to have 1 arg, got {len(node.args)}"
                )
            outputs = node.args[0]

    if outputs is None:
        raise AssertionError("no output node found in combine_graph")
    outputs = pytree.tree_leaves(outputs)
    if len(outputs) != len(xs):
        raise AssertionError(
            f"expected combine_fn to return {len(xs)} results but got {len(outputs)}"
        )

    xs_fake_tensors: list[torch.Tensor | torch.SymInt | int] = [
        first_slice_copy(x) for x in xs
    ]
    output_fake_tensors: list[torch.Tensor | torch.SymInt | int] = [
        c.meta["val"] for c in outputs
    ]
    check_meta_consistency(
        xs_fake_tensors, output_fake_tensors, "init", "carry", include_contiguity=False
    )

    _, combine_graph_name = unique_graph_id(
        proxy_mode, prefix="associative_scan_combine_graph"
    )

    proxy_mode.tracer.root.register_module(combine_graph_name, combine_graph)

    args = (combine_graph, xs, additional_inputs)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="associative_scan"
    )

    with disable_proxy_modes_tracing():
        out = tuple(aten.clone(x) for x in xs)

    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@associative_scan_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def associative_scan_op_dense(combine_fn, xs, additional_inputs):
    return generic_associative_scan(combine_fn, xs, additional_inputs=additional_inputs)


class AssociativeScanAutogradOp(torch.autograd.Function):
    r""" associative_scan
        Example::
            xs = torch.arange(1, 5) = [1, 2, 3, 4]

            def combine_fn(a: torch.Tensor, b: torch.Tensor):
                return a * b

            ys = associative_scan(comine_fn, xs),
            which can be unpacked as:
            ys0 = xs0                                         = 1
            ys1 = combine_fn(ys0, xs1) = combine_fn(1, 2)     = 2
            ...
            ysT = combine_fn(ys(T-1), xsT) = combine_fn(6, 4) = 24
            ys = [1, 2, 6, 24]

            This creates a recursive data dependency structure where each output yst
            depends on all prior inputs xs0 through xst. The dependency can be visualized as:

    Level 0 (Input):    xs0    xs1    xs2    xs3    xs4
                        \    /       |      |      |
                         \  /        |      |      |
    Level 1:              ys1 ───────┘      |      |
                           \               /       |
                            \             /        |
    Level 2:                 ys2 ────────┘         |
                              \                   /
                               \                 /
    Level 3:                    ys3 ────────────┘
                                 \
                                  \
    Level 4:                        ys4


    We could get the following backward gradient graph:


    Level 0 (output):   g_xs0   g_xs1   g_xs2   g_xs3   g_xs4
                         \      /       |       |       |
                          \    /        |       |       |
    Level 1:    gl_ys1  ─> g_ys1  ──────┘       |       |
                            \                  /        |
                             \                /         |
    Level 2:    gl_ys2     ─> g_ys2  ────────┘          |
                               \                       /
                                \                    /
    Level 3:    gl_ys3        ─> g_ys3  ────────────┘
                                  \
                                   \
    Level 4:    gl_ys4           ─> g_ys4,

    where gl_y1 is the gradient of the loss with respect to ys1 and the input of backward.

    To calculate the gradients of the inputs, the chain rule suggests:

    g_xs0 = g_ys1
    g_xs1 = g_ys1 * bw(ys0, xs1) = g_ys1 * bwxs01
    g_xs2 = g_ys2 * bw(ys1, xs2) = g_ys2 * bwxs12
    g_xs3 = g_ys3 * bw(ys2, xs3) = g_ys3 * bwxs23
    g_xs4 = g_ys4 * bw(ys3, xs4) = g_ys4 * bwxs34

    Notice the bw(...) is just the single step bw (instantaneous gradients), whose formula can be computed from combine_fn.
    For example bw(ys3, xs4) (also abbreviated with bwxs34) computes the gradients ∂/∂xs4 combine_fn(ys3, xs4).
    Similarly, bw(ys4, ys3) (also abbreviated with bwys43) computes the gradients ∂/∂ys3 combine_fn(ys3, xs4).

    Let's break down how to calculate g_ys by recursively substituting the unknowns:

    g_ys1 = gl_ys1 + g_ys2 * bw(ys2, ys1)
          = gl_ys1 + (gl_ys2  + g_ys3 * bw(ys3, ys2)) * bw(ys2, ys1)
          = gl_ys1 + gl_ys2 * bw(ys2, ys1) + g_ys3 * bw(ys3, ys2) * bw(y2, y1)
          = gl_ys1 + gl_ys2 * bw(ys2, ys1) + gl_ys3 * bw(ys3, ys2) * bw(y2, y1) \
                   + g_ys4 * bw(ys4, ys3) * bw(ys3, ys2) * bw(ys2, ys1)
          = gl_ys1 + gl_ys2 * bw(ys2, ys1) + gl_ys3 * bw(ys3, ys2) * bw(y2, y1) \
                   + gl_ys4 * bw(ys4, ys3) * bw(ys3, ys2) * bw(ys2, ys1)

    Let's do the same for all the g_ys:
    g_ys2 = gl_ys2 + gl_ys3 * bw(ys3, ys2) + gl_y4 * bw(ys4, ys3) * bw(ys3, ys2)
    g_ys3 = gl_ys3 + gl_ys4 * bw(ys4, ys3)
    g_ys4 = gl_ys4

    Notice that the above can be re-written as columnwise multiplication of y_mat and gl_ys:

    g_ys1   1, bwys21, bwys321, bwys4321       gl_ys1
    g_ys2 = 0,    1  , bwys321, bwys4321   .   gl_ys2
    g_ys3   0,    0  ,     1  , bwys4321       gl_ys3
    g_ys4   0,    0  ,     0  ,        1       gl_ys4,

    where bwys21 is an abbreviation for bw(ys2, ys1),
    bwys321 is an abbreviation for bw(ys3, ys2) * bw(ys2, ys1) so on and so forth.

    We could effectively compute the upper triangular matrix y_mat with:
    cumprod([1, bwys21, bwys32, bwys43]) then masking out the values as needed.
    Thus, only [1, bwys21, bwys32, bwys43] are required to compute the y_mat.


        References: https://justintchiu.com/blog/pscan_diff/

        NOTE: [associative_scan autograd implementation]

        The forward of associative_scan can be computed with the following steps:

        1.) Compute the forward output of the associative_scan
            ys = associative_scan(combine_fn, xs, additional_inputs)

        The backward of associative_scan can be computed with the following steps:

        2.) Prepare the backward graph
            We prepare the backward graph to be used in the backward function.
            We utilize ``create_bw_fn`` to generate the joint function:
            combine_fn_bw = create_bw_fn(combine_fn, operands)
            where operands = [ys{t-1}, xst, additional_inputs]

        3.) Materialize the ``combine_fn_bw``
            This is required because torch.compile and torch.autograd.grad
            cannot trace through the joint backward function dynamically.

        4.) Compute the single step bw (instantaneous gradients) at every step t
            bwys{t-1}, bwxst = combine_fn_bw(ys{t-1}, xst, 1.)
            Here we pass 1 as the upstream gradient to obtain the local partial derivatives.

            This gives:
                bwys = [bw(ys1, ys0), bw(ys2, ys1), ..., bw(ysT, ys{T-1})]
                bwxs = [bw(ys1, xs0), bw(ys2, xs1), ..., bw(ys{T-1}, xsT)]

        5.) Compute the gradient transition matrix y_mat

            As shown in the example above, each input xst affects all later outputs ysi for i ≥ t.
            According to the chain rule, each such path contributes a product of local gradients g_ysk.

            For example:
                ∂ysT/∂xst = ∂ysT/∂ys{T-1} * ∂ys{T-1}/∂ys{T-2} * ... * ∂ys{t+1}/∂yst * ∂yst/∂xst
                        = bw(ysT, ys{T-1}) * bw(ys{T-1}, ys{T-2}) * ... * bw(ys{t+1}, yst) * bw(ys{t-1}, xst)

            This motivates the use of a cumulative product over bwys to compute all such paths efficiently.

            We now construct the matrix of gradient transition paths:

            5.1 Repeat g_y values to form the base matrix
                y_mat = [[1, bwys21, bwys32, bwys43],
                         [1, bwys21, bwys32, bwys43],
                         [1, bwys21, bwys32, bwys43],
                         [1, bwys21, bwys32, bwys43]]

            5.2 Mask the lower triangle (inclusive) with 1s
                y_mat = [[1, bwys21, bwys32, bwys43],
                         [1, 1     , bwys32, bwys43],
                         [1, 1     , 1     , bwys43],
                         [1, 1     , 1     , 1    ]]

            5.3 Apply cumulative product row-wise
                y_mat = cumprod(y_mat, dim=1)
                Resulting in:
                y_mat = [[1, bwys21, bwys32 * bwys21, bwys43 * bwys32 * bwys21],
                         [1, 1      , bwys32         , bwys43 * bwys32         ],
                         [1, 1      , 1              , bwys43                  ],
                         [1, 1      , 1              , 1                       ]]

            5.4 Zero out the lower triangle (exclusive)
                Final y_mat:
                y_mat = [[1, bwys21, bwys32 * bwys21, bwys43 * bwys32 * bwys21],
                         [0, 1      , bwys32         , bwys43 * bwys32         ],
                         [0, 0      , 1              , bwys43                  ],
                         [0, 0      , 0              , 1                       ]]

        6.) Scale the y_mat with the upstream gradients gl_ys
            scaled_y_mat = y_mat * gl_ys
            Each entry now holds the full contribution of ∂L/∂ysj to ∂L/∂xsi via the path through ysj.

        7.) Reduce the scaled_y_mat with a row-wise sum
            summed_y_mat = scaled_y_mat.sum(dim=1)
            This accumulates all downstream contributions for each xst.

        8.) Scale with the instantaneous input gradients bwxs
            g_xs = summed_y_mat * bwxs

            This gives the final input gradients:
                g_xs = [∂L/∂xs0, ∂L/∂xs1, ..., ∂L/∂xsT]

        NOTE: [scan partial grad handling]
            If any element of xs or of the outputs does not require gradients
            (i.e., requires_grad=False), then the corresponding gradients will be returned
            as tensors of zeros with the same shape as the element.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        combine_fn,
        num_xs,
        num_additional_inputs,
        *operands,
    ):
        ctx._num_xs = num_xs
        ctx._num_additional_inputs = num_additional_inputs
        ctx._combine_fn = combine_fn
        xs, additional_inputs = split_into_chunks(
            operands, [num_xs, num_additional_inputs]
        )

        scan_length = xs[0].shape[0]
        ctx._scan_length = scan_length

        # We snapshot the dispatch keys in forward for materializing
        # the bw_graph in backward.
        ctx._fw_include_key_set = torch._C._dispatch_tls_local_include_set()
        ctx._fw_exclude_key_set = torch._C._dispatch_tls_local_exclude_set()

        with torch._C._AutoDispatchBelowAutograd():
            # 1.) Compute the forward output of the associative_scan
            ys = associative_scan_op(combine_fn, xs, additional_inputs)
            save_values_for_backward(ctx, list(operands) + list(ys))

        return (*ys,)

    @staticmethod
    def backward(ctx, *gl_ys):
        r"""
        This function computes the gradients of the scan operation.
        For a detailed description see the document above.

        Args:
            flat_grads (torch.Tensor): The tensor of upstream gradients, or a nested pytree of tensors.
                                       E.g.: Gradient of the loss with respect to the forward output ys
        """

        # The backward of associative_scan is always performed on the first dimension
        dim = 0
        scan_length = ctx._scan_length
        num_xs = ctx._num_xs
        num_additional_inputs = ctx._num_additional_inputs

        # Extract the inputs to the forward path and outputs from the forward path
        flat_args = saved_values(ctx)
        xs, additional_inputs, outs = split_into_chunks(
            flat_args, [num_xs, num_additional_inputs, num_xs]
        )

        if scan_length == 0:
            return (
                *[None] * 3,
                *[torch.zeros_like(x) for x in xs],
                *[None] * num_additional_inputs,
            )

        ndim = outs[0].ndim

        # First_slice_copy does not keep the original requires_grad flag,
        # but we need it here in order to compute the correcte gradients
        xs_slices = first_slice_copy_with_grad(itertools.chain(xs, xs))

        # Construct the operands from the forward, fw_operands
        # and the operands for a single event t of the forward, fw_operands_slice
        fw_operands = (*xs, *additional_inputs)
        fw_operands_slice = (*xs_slices, *additional_inputs)

        # 2.) Prepare the backward graph
        combine_fn_bw = create_bw_fn(ctx._combine_fn, fw_operands_slice)

        # 3.) Materialize the ``combine_fn_bw``
        # TODO: we need to materialize the bw graphs because dynamo is unable to
        # trace through the joint function when torch.compile torch.autograd.grad.
        combine_fn_bw_gm = materialize_as_graph(
            combine_fn_bw,
            (
                *fw_operands_slice,
                *[first_slice_copy(o) for o in outs],
            ),
            ctx._fw_include_key_set,
            ctx._fw_exclude_key_set,
            force_enable_grad=True,
        )

        # vmap joint graph over scan dimension to compute the individual
        # gradients for each time slice ``t`` in parallel.
        # This computation can be parallelized, as these are just the instantaneous gradients and not the full chain-rule
        # pyrefly: ignore [bad-argument-type]
        mapped_combine_fn_bw_gm = torch.vmap(combine_fn_bw_gm, 0, 0)

        # 4.) Compute the single step bw (instantaneous gradients) at every step ``t``
        # Use a ones_like tensor in order not to scale the bwyst and bwxst,
        # with the upstream gradients yet.
        # Note: All bwyst and bwxst are computed in parallel, thus the tensors bwys and bwxs are the result.
        dummy_upstream_grad = (torch.ones_like(x) for x in gl_ys)
        grads = mapped_combine_fn_bw_gm(
            *(o.roll(1, dim) for o in outs), *fw_operands, *dummy_upstream_grad
        )
        bwys, bwxs = split_into_chunks(grads, [num_xs, num_xs])

        def compute_y_mat(bwys: torch.Tensor) -> torch.Tensor:
            # Prepare a ones and a zeros helper mask in order to easily compute the y_mat
            def compute_helper_tril_mask(diagonal):
                def expand_masks(mask):
                    for _ in range(ndim - 1):
                        mask = mask.unsqueeze(-1)
                    return mask

                tril_mask = torch.tril(
                    torch.ones(
                        scan_length, scan_length, device=bwys.device, dtype=torch.bool
                    ),
                    diagonal=diagonal,
                )
                tril_mask = expand_masks(tril_mask)
                tril_mask = tril_mask.expand(-1, -1, *bwys.shape[1:])
                return tril_mask

            # The ones mask is used to fill the main diagonal and all elements below it with 1s
            ones_mask = compute_helper_tril_mask(0)

            # The zero mask is used to set all elements below the main diagonal to 0
            zeros_mask = compute_helper_tril_mask(-1)

            # 5.1) Repeat the elements of bwys to form the square matrix
            y_mat = bwys.unsqueeze(dim).repeat_interleave(scan_length, dim)

            # 5.2) Fill the lower triangular part, including the diagonal,
            # of the h_mat with 1s. I.e., use the ones_mask to fill with 1s.
            y_mat.masked_fill_(ones_mask, 1.0)

            # 5.3) Compute the cumulative products across dim + 1
            y_mat = y_mat.cumprod(dim=dim + 1)

            # 5.4) Replace the elements we filled with 1s before with 0s
            y_mat.masked_fill_(zeros_mask, 0.0)

            return y_mat

        def compute_grad(bwxs, bwys, gl_ys):
            # Set the first gradient component of bwxs to 1.0, per definition.
            torch.select(bwxs, dim, 0).fill_(1.0)

            # 5.) Compute the gradient transition matrix
            y_mat = compute_y_mat(bwys)

            # 6.) scale the y_mat with the upstream gradients gl_ys
            scaled_y_mat = y_mat * gl_ys

            # 7.) Reduce the y_mat with sum along the columns to get the total contributions for xs_t
            summed_y_mat = scaled_y_mat.sum(dim + 1)

            # 8.) Scale with the bwxs to obtain the final gradients g_xs
            g_xs = summed_y_mat * bwxs

            return g_xs

        # Stack all leaves of the gradients along the first dimension.
        # This is useful as later the gradients of those leaves can be computed in parallel.
        bwxs_stacked_leaves = torch.stack(bwxs)
        bwys_stacked_leaves = torch.stack(bwys)
        gl_ys_stacked_leaves = torch.stack(gl_ys)

        # The compute_grad function is parallelized across all individual leaves of xs
        # as these gradients can be computed independently from each other
        # TODO: torch.vmap may create composability issues
        compute_grad_mapped = torch.vmap(compute_grad, 0, 0)

        g_xs = compute_grad_mapped(
            bwxs_stacked_leaves, bwys_stacked_leaves, gl_ys_stacked_leaves
        )

        # TODO: Currently the gradients for the additional_inputs are not computed properly
        return *[None] * 3, *g_xs, *[None] * num_additional_inputs


@associative_scan_op.py_autograd_impl
def associative_scan_autograd(combine_fn, xs, additional_inputs):
    num_xs = len(xs)
    num_additional_inputs = len(additional_inputs)

    if num_additional_inputs > 0:
        raise RuntimeError(
            "Associative_scan does currently not support gradients for lifted parameters!"
        )

    flat_out = AssociativeScanAutogradOp.apply(
        combine_fn,
        num_xs,
        num_additional_inputs,
        *(tuple(xs) + tuple(additional_inputs)),
    )
    return (*flat_out,)


@associative_scan_op.py_impl(ProxyTorchDispatchMode)
def associative_scan_proxy_mode(mode, combine_fn, xs, additional_inputs):
    return trace_associative_scan(
        mode, associative_scan_op, combine_fn, xs, additional_inputs
    )


@register_fake(associative_scan_op, skip_cache=True)
def assoiciative_scan_fake_tensor_mode(combine_fn, xs, additional_inputs):
    return tuple(x.clone() for x in xs)


@associative_scan_op.py_functionalize_impl
def associative_scan_functionalize(ctx, combine_fn, xs, additional_inputs):
    from torch._higher_order_ops.utils import _check_alias_and_mutation

    unwrapped_xs = ctx.unwrap_tensors(xs)
    unwrapped_additional_inputs = ctx.unwrap_tensors(additional_inputs)
    with ctx.redispatch_to_next():
        functional_combine_fn = ctx.functionalize(
            _maybe_run_with_interpreter(combine_fn)
        )
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        sample_unwrapped_xs_sliced = [
            first_slice_copy(inp) for inp in itertools.chain(unwrapped_xs, unwrapped_xs)
        ]
        sample_inputs = list(
            itertools.chain(
                sample_unwrapped_xs_sliced,
                unwrapped_additional_inputs,
            )
        )
        _check_alias_and_mutation(
            combine_fn, sample_inputs, "associative_scan", pre_dispatch
        )
        ret = associative_scan_op(
            functional_combine_fn,
            unwrapped_xs,
            unwrapped_additional_inputs,
        )
    return ctx.wrap_tensors(ret)


def _fake_associative_scan(combine_fn, xs, dim, reverse=False):
    inp_leaves, spec = pytree.tree_flatten(xs)
    result_flat: list[Any] = []
    num_leaves = len(inp_leaves)
    op = reversed if reverse else lambda x: x

    for ind in op(range(inp_leaves[0].size(dim))):
        r = [
            inp_leaves[leave_ind][(slice(None),) * dim + (ind,)]
            for leave_ind in range(num_leaves)
        ]
        if (ind > 0 and not reverse) or (
            ind < (inp_leaves[0].size(dim) - 1) and reverse
        ):
            r = combine_fn(
                pytree.tree_unflatten(result_flat[-1], spec),
                pytree.tree_unflatten(r, spec),
            )
        r_flat, _ = pytree.tree_flatten(r)
        result_flat.append(r_flat)

    if len(result_flat) == 0:
        results = list(inp_leaves)
    else:
        results = [
            torch.stack([e[leave_ind] for e in op(result_flat)], dim)
            for leave_ind in range(num_leaves)
        ]
    return pytree.tree_unflatten(results, spec)
<|endoftext|>from typing import Any, TYPE_CHECKING

import torch
from torch._C import DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented, register_fake
from torch._ops import HigherOrderOperator


if TYPE_CHECKING:
    from torch._subclasses.functional_tensor import BaseFunctionalizeAPI

from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.utils import _pytree as pytree


class RunConstGraph(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("run_const_graph")

    def __call__(self, graph: torch.fx.GraphModule, args: tuple[object, ...]) -> object:
        # pyrefly: ignore [missing-attribute]
        return super().__call__(graph, args)


run_const_graph = RunConstGraph()


@run_const_graph.py_impl(ProxyTorchDispatchMode)
def run_const_graph_dispatch_mode(
    mode: ProxyTorchDispatchMode, graph: torch.fx.GraphModule, args: tuple[object, ...]
) -> object:
    const_gm, weights = graph, args
    p_args = pytree.tree_map(mode.tracer.unwrap_proxy, (graph, args))  # type: ignore[union-attr]
    if not isinstance(const_gm, torch.fx.GraphModule):
        raise AssertionError(
            f"expected const_gm to be torch.fx.GraphModule, got {type(const_gm)}"
        )
    if hasattr(mode.tracer.root, "_const_graph"):  # type: ignore[union-attr]
        raise AssertionError("mode.tracer.root already has _const_graph attribute")
    mode.tracer.root.register_module("_const_graph", const_gm)  # type: ignore[union-attr]

    proxy = mode.tracer.create_proxy("call_function", run_const_graph, p_args, {})

    out = const_gm(*weights)
    return track_tensor_tree(out, proxy, constant=None, tracer=mode.tracer)


@run_const_graph.py_functionalize_impl
def run_const_graph_functional(
    ctx: "BaseFunctionalizeAPI", graph: torch.fx.GraphModule, args: tuple[Any, ...]
) -> Any:
    unwrapped_args = ctx.unwrap_tensors(args)

    with ctx.redispatch_to_next():
        out = run_const_graph(graph, unwrapped_args)
        return ctx.wrap_tensors(out)  # type: ignore[arg-type]


run_const_graph.py_autograd_impl(
    autograd_not_implemented(run_const_graph, deferred_error=True)
)


@register_fake(run_const_graph, skip_cache=True)
def run_const_graph_fake_tensor_mode(
    graph: torch.fx.GraphModule, args: tuple[object, ...]
) -> object:
    if not isinstance(graph, torch.fx.GraphModule):
        raise AssertionError(
            f"expected graph to be torch.fx.GraphModule, got {type(graph)}"
        )
    return graph(*args)


@run_const_graph.py_impl(DispatchKey.CPU)
def run_const_graph_cpu(
    graph: torch.fx.GraphModule, args: tuple[object, ...]
) -> object:
    if not isinstance(graph, torch.fx.GraphModule):
        raise AssertionError(
            f"expected graph to be torch.fx.GraphModule, got {type(graph)}"
        )
    return graph(*args)
<|endoftext|># mypy: allow-untyped-defs
import contextlib
import functools
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, ExitStack, nullcontext
from dataclasses import dataclass
from typing import Any, overload, TypeVar
from typing_extensions import ParamSpec

import torch
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree
from torch._dispatch.python import suspend_functionalization
from torch._guards import detect_fake_mode
from torch._higher_order_ops.schema import HopSchema
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import is_custom_class
from torch._ops import HigherOrderOperator, OperatorBase, OpOverload
from torch._subclasses.fake_tensor import (
    FakeTensor,
    is_fake_tensor,
    maybe_get_fake_mode,
)
from torch._subclasses.functional_tensor import (
    disable_functional_mode,
    FunctionalTensor,
)
from torch.fx.experimental.proxy_tensor import (
    _temp_remove_metadata_torch_function_mode,
    disable_proxy_modes_tracing,
    make_fx,
)
from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


@dataclass
class UnsupportedAliasMutationException(RuntimeError):
    reason: str


def autograd_not_implemented_inner(
    operator: OperatorBase, delayed_error: bool, *args: Any, **kwargs: Any
) -> Any:
    """If autograd is enabled and any of the arguments require grad this will either
    raise an error or return a DelayedError depending on the value of delayed.

    Args:
        operator: The Operator to call with the *args and **kwargs with
        op_name: The name of the Operator
        delayed_error: If True, return a DelayedError instead of raising an error
        args: The flattened operands to the Operator
        kwargs: The keyword arguments to the Operator

    Raises:
        RuntimeError: If autograd is enabled and any of the arguments to the Operator
    """
    with torch._C._AutoDispatchBelowAutograd():
        result = operator(*args, **kwargs)
        flat_operands = pytree.arg_tree_leaves(*args)
        if torch.is_grad_enabled() and any(
            f.requires_grad for f in flat_operands if isinstance(f, torch.Tensor)
        ):
            if delayed_error:
                err_fn = torch._C._functions.DelayedError(
                    f"Autograd not implemented for {str(operator)}",
                    1,
                )

                def fake_requires_grad(tensor):
                    if torch.is_floating_point(tensor) or torch.is_complex(tensor):
                        tensor = tensor.detach()
                        tensor.requires_grad = True
                    return tensor

                return pytree.tree_map_only(
                    torch.Tensor, lambda x: err_fn(fake_requires_grad(x)), result
                )
            else:
                raise RuntimeError(f"Autograd not implemented for {str(operator)}")
        return result


def autograd_not_implemented(op: OperatorBase, deferred_error: bool) -> Callable:
    def inner(*args, **kwargs):
        return autograd_not_implemented_inner(op, deferred_error, *args, **kwargs)

    return inner


def _maybe_run_with_interpreter(fn):
    maybe_interpreted_fn = fn
    if isinstance(fn, torch.fx.GraphModule) and fx_traceback.has_preserved_node_meta():
        # Running graph with interpreter is needed for propagating the stack_trace
        def graph_with_interpreter(*args):
            with fx_traceback.preserve_node_meta():
                return torch.fx.Interpreter(fn).run(*args)

        maybe_interpreted_fn = graph_with_interpreter
    return maybe_interpreted_fn


def _hop_compile_and_call(fn, args, kwargs=None):
    """Compile and call fn with fullgraph=True for HOP eager execution.

    Pre-activates the fullgraph counter so that compile_wrapper treats this as
    a nested compile.  This avoids erroring when a non-infra dispatch mode
    causes the frame to be skipped — the function still executes eagerly within
    compile_wrapper and returns normally.
    """
    from torch._dynamo.eval_frame import set_fullgraph_compiled_frame_count

    with setup_compilation_env() as backend:
        old_count = set_fullgraph_compiled_frame_count(0)
        try:
            return torch.compile(fn, backend=backend, fullgraph=True)(
                *args, **(kwargs or {})
            )
        finally:
            set_fullgraph_compiled_frame_count(old_count)


def _maybe_compile_and_run_fn(fn, *args):
    if not torch.compiler.is_dynamo_compiling():
        return _hop_compile_and_call(fn, args)
    else:
        return fn(*args)


def reenter_make_fx(fn, subgraph_decomp_table=None):
    from torch.fx.experimental.proxy_tensor import _CURRENT_MAKE_FX_TRACER

    @functools.wraps(fn)
    def wrapped(*args):
        if _CURRENT_MAKE_FX_TRACER is None:
            raise AssertionError(
                "Cannot reenter make_fx when we're not under a make_fx tracing session"
            )
        if subgraph_decomp_table is None:
            gm = _CURRENT_MAKE_FX_TRACER.trace_subgraph(
                _maybe_run_with_interpreter(fn), *args
            )
        else:
            gm = _CURRENT_MAKE_FX_TRACER.trace_subgraph_custom_decomp(
                _maybe_run_with_interpreter(fn), subgraph_decomp_table, *args
            )

        return gm

    return wrapped


def _maybe_reenter_make_fx(fn, subgraph_decomp_table=None):
    from torch.fx.experimental.proxy_tensor import _CURRENT_MAKE_FX_TRACER

    if _CURRENT_MAKE_FX_TRACER is not None:
        return reenter_make_fx(fn, subgraph_decomp_table=subgraph_decomp_table)
    else:

        def _maybe_make_fx_with_fake_mode(fn):
            @functools.wraps(fn)
            def wrapped(*args):
                from torch._guards import detect_fake_mode

                fake_mode = detect_fake_mode(args)
                if fake_mode is None:
                    # we creaeta a fake_mode here to make sure we could
                    # trace the graph with data-dependent calls e.g. .item()
                    return make_fx(
                        fn,
                        tracing_mode="fake",
                        decomposition_table=subgraph_decomp_table,
                    )(*args)
                # Tracing with real if all inputs have been fakfied
                return make_fx(fn, decomposition_table=subgraph_decomp_table)(*args)

            return wrapped

        return _maybe_make_fx_with_fake_mode(fn)


def check_meta_consistency(
    lhs_list: list[torch.Tensor | torch.SymInt | int],
    rhs_list: list[torch.Tensor | torch.SymInt | int],
    lhs_name: str,
    rhs_name: str,
    include_contiguity: bool = True,
) -> None:
    def diff_meta_pairs(
        lhs_list: list[torch.Tensor | torch.SymInt | int],
        rhs_list: list[torch.Tensor | torch.SymInt | int],
    ) -> list[str]:
        def diff_meta(
            lhs: torch.Tensor | torch.SymInt | int,
            rhs: torch.Tensor | torch.SymInt | int,
        ) -> str:
            if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
                return ", ".join(
                    diff_tensor_meta(
                        _extract_tensor_metadata(
                            lhs, include_contiguity=include_contiguity
                        ),
                        _extract_tensor_metadata(
                            rhs, include_contiguity=include_contiguity
                        ),
                        check_grad=False,
                    )
                )
            else:

                def _both_int_types(lhs, rhs):
                    return isinstance(lhs, (int, torch.SymInt)) and isinstance(
                        rhs, (int, torch.SymInt)
                    )

                def _both_tensor(lhs, rhs):
                    return isinstance(lhs, torch.Tensor) and isinstance(
                        rhs, torch.Tensor
                    )

                if not _both_int_types(lhs, rhs) and not _both_tensor(lhs, rhs):
                    return f"type: {lhs} vs {rhs}"

            return ""

        # Manually check the device of lhs and rhs as this field is currently not part of TensorMetadata
        def diff_device(
            lhs: torch.Tensor | torch.SymInt | int,
            rhs: torch.Tensor | torch.SymInt | int,
        ) -> str:
            if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
                if (
                    rhs.device.type == lhs.device.type
                    and rhs.device.index == lhs.device.index
                ):
                    return ""
                else:
                    return "device"
            return ""

        if len(lhs_list) != len(rhs_list):
            raise torch._dynamo.exc.UncapturedHigherOrderOpError(
                f"Expected {lhs_name} and {rhs_name} to have same number of outputs but got lhs:{lhs_list} and rhs:{rhs_list}"
            )
        all_diffs = []
        for i, (lhs, rhs) in enumerate(zip(lhs_list, rhs_list)):
            if diff := diff_meta(lhs, rhs):
                all_diffs.append(
                    f"pair[{i}] differ in {diff}, where lhs is {lhs} and rhs is {rhs}"
                )
            if diff := diff_device(lhs, rhs):
                all_diffs.append(
                    f"pair[{i}] differ in {diff}, where lhs is {lhs} and rhs is {rhs}"
                )
        return all_diffs

    if all_diffs := diff_meta_pairs(lhs_list, rhs_list):
        diff_str = "\n".join(all_diffs)
        raise torch._dynamo.exc.UncapturedHigherOrderOpError(
            f"Expected {lhs_name} and {rhs_name} to have same metadata but found:\n{diff_str}"
        )


# Thread-local flag to indicate we're inside HOP internal compilation
import threading


_hop_compile_tls = threading.local()


def _in_hop_compile() -> bool:
    return getattr(_hop_compile_tls, "in_hop_compile", False)


@contextmanager
def setup_compilation_env():
    """
    Context manager that sets up proper environment and backend when invoking torch.compile
    inside torch.export region or inside HOP.
    """
    from torch._dynamo.backends.debugging import (
        make_eager_backend_with_torch_function_modes,
    )
    from torch._dynamo.backends.registry import lookup_backend
    from torch.fx.experimental.proxy_tensor import (
        _temp_remove_pre_dispatch_torch_function_mode,
    )

    old_in_hop_compile = getattr(_hop_compile_tls, "in_hop_compile", False)
    _hop_compile_tls.in_hop_compile = True
    try:
        with (
            _set_compilation_env(),
            torch._dynamo.utils.disable_cache_limit(),
            _temp_remove_pre_dispatch_torch_function_mode() as pre_dispatch_mode,
            _temp_remove_metadata_torch_function_mode() as metadata_mode,
        ):
            modes = [
                mode for mode in (pre_dispatch_mode, metadata_mode) if mode is not None
            ]
            if modes:
                yield make_eager_backend_with_torch_function_modes(modes)
            else:
                yield lookup_backend("eager")
    finally:
        _hop_compile_tls.in_hop_compile = old_in_hop_compile


@contextmanager
def _set_compilation_env():
    _old_is_tracing = torch.fx._symbolic_trace._get_is_fx_tracing()
    _old_allow_empty_graphs = torch._dynamo.config.allow_empty_graphs
    _old_capture_scalar_outputs = torch._dynamo.config.capture_scalar_outputs
    # The issue is tracked in https://github.com/pytorch/pytorch/issues/144360: when dynamo finds
    # the top-level frame produces no graph, the default behavior is to fallback to eager.
    # Then when it encounters an inner function, it will try to trace that function again, which is unnecessary.
    # For while_loop, during inspecting the inner call, we trace into the python dispatcher
    # logic, which is not tracable as of today. So the proper fix can be either 1. allow dispatch
    # logic to be dynamo tracable or 2. fixing https://github.com/pytorch/pytorch/issues/144360.
    # but it exposes some bugs in existing tests so we have to have a temporary flag to control
    # the behavior, which allows dynamo to store an empty graph for a frame without falling back to eager
    try:
        # We need to turn off the is_fx_tracing_flag. Remove this flag check from dynamo
        # once we are confident fx tracing works with dynamo.
        torch.fx._symbolic_trace._set_is_fx_tracing(False)
        # pyrefly: ignore [bad-assignment]
        torch._dynamo.config.allow_empty_graphs = True
        torch._dynamo.config.capture_scalar_outputs = True
        yield
    finally:
        torch.fx._symbolic_trace._set_is_fx_tracing(_old_is_tracing)
        torch._dynamo.config.allow_empty_graphs = _old_allow_empty_graphs
        torch._dynamo.config.capture_scalar_outputs = _old_capture_scalar_outputs


# The invariant here is that we always trace the branch with fake tensor
def _maybe_fake_tracing(fn, inputs: list[Any], pre_dispatch):
    fake_mode_det = None
    for inp in pytree.tree_leaves(inputs):
        if is_fake_tensor(inp):
            fake_mode_det = maybe_get_fake_mode(inp)
            break

    fake_mode: AbstractContextManager = nullcontext()
    tracing_mode = "fake"
    if fake_mode_det is not None:
        fake_mode = fake_mode_det
        tracing_mode = "real"

    # Note: we need to turn off proxy tensor mode to avoid tracing infra
    # code that happens in make_fx e.g. we now call as_strided when wrapping tensor
    # as fake tensor.
    with fake_mode, disable_proxy_modes_tracing():
        gm = make_fx(
            fn,
            tracing_mode=tracing_mode,
            pre_dispatch=pre_dispatch,
            _error_on_data_dependent_ops=False,
        )(*inputs)
        if not isinstance(fake_mode, nullcontext) and fake_mode.shape_env is not None:  # type: ignore[attr-defined]
            insert_deferred_runtime_asserts(
                gm,
                fake_mode.shape_env,  # type: ignore[attr-defined]
                "hoo_maybe_fake_tracing",
                export=True,  # type: ignore[attr-defined]
            )
        return gm


def potential_input_alias_or_mutation(gm, inputs, pre_dispatch=False):
    try:
        gm = _maybe_fake_tracing(gm, inputs, pre_dispatch)
    except UnsupportedAliasMutationException:
        # this can happen when nested cond_op is
        # functionalized
        return True
    except Exception as e:
        raise e

    example_inputs = [
        ph.meta.get("val", None) for ph in gm.graph.find_nodes(op="placeholder")
    ]
    (
        inp_inp_alias_map,
        inp_out_alias_map,
        out_out_alias_map,
        inp_mutation,
    ) = check_input_alias_and_mutation(gm, example_inputs)
    return (inp_inp_alias_map, inp_out_alias_map, out_out_alias_map), inp_mutation


def analyze_potential_input_alias_or_mutation(name, aliases, input_mutations):
    if any(len(a) > 0 for a in aliases):
        # TODO: Investigate here further which node is exactly aliasing
        raise RuntimeError(
            f"{name} where aliases appear. "
            + f"In particular, these inputs \
            {set(el for el_map in aliases if len(el_map.keys()) > 0 for el in el_map)} "  # noqa: C401
            + "get aliased. Please ensure that this doesn't happen."
        )
    if len(input_mutations):
        # TODO: Investigate here further which node is exactly mutating the inputs
        raise RuntimeError(
            f"{name} where the inputs are mutated. "
            + f"In particular, these nodes are mutating the inputs \
            {set(el for el in input_mutations)}."  # noqa: C401
            + "Please ensure that this doesn't happen."
        )


def _has_potential_branch_input_mutation(gm, inputs, pre_dispatch=False):
    (
        (_, _, _),
        inp_mutation,
    ) = potential_input_alias_or_mutation(gm, inputs, pre_dispatch)

    return len(inp_mutation) > 0


def has_potential_input_alias_or_mutation(gm, inputs, pre_dispatch=False):
    (
        (
            inp_inp_alias_map,
            inp_out_alias_map,
            out_out_alias_map,
        ),
        inp_mutation,
    ) = potential_input_alias_or_mutation(gm, inputs, pre_dispatch)
    return (
        any(
            (
                len(inp_inp_alias_map) > 0,
                len(inp_out_alias_map) > 0,
                len(out_out_alias_map) > 0,
            )
        ),
        len(inp_mutation) > 0,
    )


def _collect_fake_inputs(inputs):
    from torch._subclasses.fake_tensor import FakeTensor

    # Get the example values of the inputs.
    inputs_fake: list[FakeTensor | torch.Tensor | int] = []
    for inp in inputs:
        if isinstance(inp, (torch.fx.proxy.Proxy, torch.fx.node.Node)):
            inp = inp.node if isinstance(inp, torch.fx.proxy.Proxy) else inp
            if hasattr(inp, "meta"):
                val = inp.meta["example_value"]
                if isinstance(val, torch.Tensor):
                    if torch._C._functorch.is_batchedtensor(
                        val
                    ) or torch._C._functorch.is_functionaltensor(val):
                        # This case is for batched or functional tensors
                        # Unwrap the tensors
                        while torch._C._functorch.is_batchedtensor(
                            val
                        ) or torch._C._functorch.is_functionaltensor(val):
                            val = torch._C._functorch.get_unwrapped(val)
                        if not is_fake_tensor(val):
                            raise AssertionError(
                                f"Expected FakeTensor after unwrapping, got {type(val)}"
                            )
                        inputs_fake.append(val)
                    elif is_traceable_wrapper_subclass(val):
                        for attr_name in val.__tensor_flatten__()[0]:
                            unwrapped_input = getattr(val, attr_name)
                            if not isinstance(unwrapped_input, torch.Tensor):
                                continue
                            if not is_fake_tensor(unwrapped_input):
                                raise AssertionError(
                                    f"Expected FakeTensor after unwrapping, got {type(unwrapped_input)}"
                                )
                            inputs_fake.append(unwrapped_input)
                    else:
                        # This is the standard case of a TensorVariable
                        if not is_fake_tensor(val):
                            raise AssertionError(
                                f"Expected FakeTensor, got {type(val)}"
                            )
                        inputs_fake.append(val)
                else:
                    # This case is for SymInts and other non-Tensor elements
                    if isinstance(val, torch.Tensor):
                        raise AssertionError(f"Expected non-Tensor, got {type(val)}")
                    inputs_fake.append(val)
        else:
            # This case is for ints
            if not isinstance(inp, int):
                raise AssertionError(f"Expected int, got {type(inp)}")
            inputs_fake.append(inp)

    return inputs_fake


def _check_alias_and_mutation(graph_module, inputs_fake, name, pre_dispatch):
    aliases, inp_mutation = has_potential_input_alias_or_mutation(
        graph_module, inputs_fake, pre_dispatch=pre_dispatch
    )
    if aliases:
        raise RuntimeError(f"{name} might be aliasing the input or the output!")
    if inp_mutation:
        raise RuntimeError(f"{name} might be modifying the input!")


def unique_graph_id(proxy_mode, prefix):
    """Returns a unique name and id for a graph to be added to a proxy_mode tracer"""
    # There are probably better ways - I know that create_arg has some self incrementing name
    # magic to it, but since we explicitly have to get the name for register_module,
    # I was not sure how to do that. This kinda simulates it.
    return unique_graph_name_with_root(proxy_mode.tracer.root, prefix)


def unique_graph_name_with_root(
    root: torch.fx.GraphModule, prefix: str
) -> tuple[int, str]:
    next_name = None
    i = 0
    # pyrefly: ignore [bad-assignment]
    while not next_name:
        candidate = f"{prefix}_{i}"
        if hasattr(root, candidate):
            i += 1
        else:
            next_name = candidate
    return i, next_name


def _from_fun(t):
    from torch._functorch.aot_autograd import from_fun

    if isinstance(t, torch.Tensor):
        if t.dtype != torch.bool:
            return torch.empty_strided(
                t.size(),
                t.stride(),
                dtype=t.dtype,
                requires_grad=t.requires_grad,
                device=t.device,
            )
        else:
            # clone of a functional tensor produces a functional tensor
            # but we want to avoid it so we clone a non-functional version
            maybe_unfunc_t = t
            if isinstance(t, FunctionalTensor):
                torch._sync(t)
                maybe_unfunc_t = from_fun(t)
            elif torch._is_functional_tensor(t):
                # need to handle both types of functionalization here:
                # these are the tensors that came from the user,
                # which could be either FunctionalTensorWrapper or FunctionalTensor
                torch._sync(t)
                maybe_unfunc_t = torch._from_functional_tensor(t)
            # pyrefly: ignore[missing-attribute]
            return maybe_unfunc_t.clone()
    return t


def clone_outputs_aliasing_inputs(args):
    input_storage = {
        StorageWeakRef(arg._typed_storage())
        for arg in args
        if isinstance(arg, torch.Tensor)
    }

    def maybe_clone(t):
        if (
            isinstance(t, torch.Tensor)
            and StorageWeakRef(t._typed_storage()) in input_storage
        ):
            return t.clone()
        return t

    return maybe_clone


def query_requires_grad(t: torch.Tensor) -> bool:
    """requires_grad of ``t``, looking through a functional wrapper.

    A FunctionalTensor does not carry the autograd metadata of the tensor it
    wraps, so asking it directly reports False for something that does require
    grad. This matters when the joint is traced with functionalization active.
    """
    if torch._is_functional_tensor(t):
        t = torch._from_functional_tensor(t)
    return t.requires_grad


def prepare_fw_with_masks(fn):
    def fw_with_masks(*args):
        fw_out = fn(*args)
        return fw_out, [
            bool(isinstance(ret, torch.Tensor) and query_requires_grad(ret))
            for ret in fw_out
        ]

    return fw_with_masks


def prepare_fw_with_masks_all_requires_grad(fn):
    def fw_with_masks(*args):
        fw_out = fn(*args)
        # Note [force all outputs to be require grad]
        # Instead of using the original fn, we set the output of original
        # fn to all require grad. This is consistent with the behavior
        # of autograd.Function, where if any one of the inputs requires grad
        # all output will be require grad. This also makes the downstream
        # require_gradness reasoning much easier.
        if pytree.tree_any_only(torch.Tensor, lambda t: t.requires_grad, args):
            fw_out = pytree.tree_map_only(
                torch.Tensor,
                lambda x: x.requires_grad_(True) if x.dtype.is_floating_point else x,
                fw_out,
            )

        return fw_out, pytree.tree_map_only(torch.Tensor, query_requires_grad, fw_out)

    return fw_with_masks


# This function replaces None gradients with all-zero gradients.
# `None` gradients are problematic for CUDA graphs. Those gradients are
# replaced with an all-zero tensor for better optimization
def unmask_none_gradients(grads, operands):
    allowed_types = (torch.Tensor, int, torch.SymInt)
    if not all(isinstance(o, allowed_types) for o in operands):
        raise AssertionError(
            f"operands can only be of {allowed_types} but got {[type(o) for o in operands]}"
        )

    unmasked_grads = []
    for g, o in zip(grads, operands):
        if g is not None:
            unmasked_grads.append(g)
        else:
            # In case the operand is an int or a torch.SymInt, return None
            # This can happen for lifted_arguments. E.g., the shapes of a dynamic tensor are lifted and passed
            # as additional arguments
            unmasked_grads.append(
                torch.zeros_like(o) if isinstance(o, torch.Tensor) else None
            )

    return unmasked_grads


def _maybe_fake_prop_ignore_unbacked(fn, args):
    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():
            unfunc_args = [_from_fun(arg) for arg in args]
        with ExitStack() as ctx_stack:
            ctx_stack.enter_context(
                torch.utils._python_dispatch._disable_current_modes()
            )
            if (fake_mode := detect_fake_mode(unfunc_args)) is not None:
                ctx_stack.enter_context(fake_mode)
                if fake_mode.shape_env is not None:
                    ctx_stack.enter_context(
                        fake_mode.shape_env.ignore_fresh_unbacked_symbols()
                    )
            return fn(*unfunc_args)


def redirect_to_mode(hop: OperatorBase, mode):
    """Utility for redispatching HOP to underlying mode

    Args:
        hop: The HOP to redispatch
        mode: The mode to redispatch to

    Returns:
        A decorated function that implements the HOP for the given mode
    """

    @hop.py_impl(mode)
    def impl(mode, *args, **kwargs):
        return mode.__torch_dispatch__(hop, [], args, kwargs)

    return impl


# TODO: The parameter use_output_and_grad_bw is required because some operations
# that utilize this function, such as the while_loop, may require (grad, fwd_outputs)
def create_fw_bw_graph(fn, use_output_and_grad_bw, fw_inputs, fw_outputs):
    from torch._functorch.aot_autograd import AOTConfig, create_joint

    # Note:[HOP create fw_bw graph] We create "clean" environments for make_fx by suspending all dispatch keys
    # between Autograd and Python key. Currently, we only suspend functionalization but more can be
    # added when required. Will encounter two problems if we don't suspend functionalization:
    #
    # 1. make_fx fails to capture operations on input: the inputs are wrapped as _to_functional_tensor_wrapper,
    # but they will be unwrapped before entering ProxyTorchDispatchMode as part of the dispatching.
    # However, it's the outside wrapper that tracer creates proxies for. This casuses tracer fail to
    # fetch the proxy for the inputs and fail to capture any operations on them.
    #
    # 2. make_fx fails to capture output: the outputs after ProxyTorchDispatchMode are further
    # wrapped as FunctionalTensorWrapper in Functionalize key after return. However, the tracer
    # only associates the inner tensor with proxy in ProxyTorchDispatchMode. Therefore,
    # when creating the output node, it fails to associate the wrapped tensor with its proxy.
    # Instead, it will create _tensor_constant as output.

    dummy_aot_config = AOTConfig(
        fw_compiler=None,  # type: ignore[arg-type]
        bw_compiler=None,  # type: ignore[arg-type]
        partition_fn=None,  # type: ignore[arg-type]
        decompositions={},
        num_params_buffers=0,
        aot_id=0,
        keep_inference_input_mutations=False,
    )

    example_grad = [_from_fun(out) for out in fw_outputs]
    num_grads = len(example_grad)
    fw_graph = _maybe_reenter_make_fx(fn)(*fw_inputs)

    def joint_fn(*joint_operands_grads):
        if use_output_and_grad_bw:
            grads = joint_operands_grads[0]
            inputs = joint_operands_grads[1][-1:]
        else:
            grads = joint_operands_grads[:num_grads]
            inputs = joint_operands_grads[num_grads:]

        joint = create_joint(prepare_fw_with_masks(fn), aot_config=dummy_aot_config)
        _, grads = joint(
            list(inputs),
            [grad for grad in grads if grad is not None and grad.requires_grad],
        )

        # Unmask None gradients to all-zero gradients
        unmasked_grads = unmask_none_gradients(grads, inputs)

        # In order to keep map functional for backward graph,
        # we clone outputs that are aliasing inputs
        maybe_clone = clone_outputs_aliasing_inputs(joint_operands_grads)

        return pytree.tree_map(maybe_clone, unmasked_grads)

    if use_output_and_grad_bw:
        example_xs_out = list(fw_inputs) + list(fw_outputs)
        joint_graph = _maybe_reenter_make_fx(joint_fn)(
            (list(example_grad), list(example_xs_out))
        )
    else:
        example_xs_out = list(fw_inputs)
        joint_graph = _maybe_reenter_make_fx(joint_fn)(
            *(list(example_grad) + list(example_xs_out))
        )

    return fw_graph, joint_graph


def _unstack_pytree(xs):
    flat_xs, inspec = pytree.tree_flatten(xs)
    if not all(isinstance(xs, torch.Tensor) for xs in flat_xs):
        raise RuntimeError(f"Leaves of xs must be Tensor {flat_xs}")

    if not all(xs.shape[0] == flat_xs[0].shape[0] for xs in flat_xs):
        raise RuntimeError(
            f"Leaves of xs must have same leading dimension size {[xs.shape for xs in flat_xs]}"
        )

    a = zip(*flat_xs)

    pytrees = [pytree.tree_unflatten(tuple, inspec) for tuple in a]
    return pytrees


def _stack_pytree(pytrees):
    flat_out = []
    out_spec = None
    for pt in pytrees:
        flat_pt, out_spec = pytree.tree_flatten(pt)
        flat_out.append(flat_pt)
    if out_spec is None:
        raise AssertionError("out_spec cannot be None")
    b = zip(*flat_out)
    stacked_out = []
    for leaves in b:
        if all(isinstance(leaf, torch.Tensor) for leaf in leaves):
            stacked_out.append(torch.stack(leaves))
        elif all(leaf is None for leaf in leaves):
            # Backward graph can return None output when forward inputs doesn't require grad.
            # When we eagerly execute backward graph, we need to call _stack_pytree on its output,
            # therefore we need to deal with None output.
            stacked_out.append(None)  # type: ignore[arg-type]
        else:
            raise RuntimeError(f"Cannot stack {leaves}.")
    return pytree.tree_unflatten(stacked_out, out_spec)


# We cannot call save_for_backward for symints. This helper function
# can be used to save symints as direct attributes of ctx in autograd.Function.
#
# For example, if args = (x, y, s0, z, s1),
# save_values_for_backward will partition the args into two lists, and a bookkeeping list pos:
#   partitioned_args[0] = (x, y, z)
#   partitioned_args[1] = (s0, s1)
#   pos = (0, 0, 1, 0, 1)
# pos list keeps track of which partition the args
# is partitioned into in order to recover it in saved_values.
#
# In saved_values, we can recover the original args by:
# iterating over the pos list and pop one item from the front of partitioned_args[pos[i]].
# We use t_idx and s_idx to keep track of the next index of the item we are going to pop for the two lists.
def save_values_for_backward(ctx, args):
    if not all(
        isinstance(arg, (torch.Tensor, torch.SymInt, int, type(None), FakeScriptObject))
        or is_custom_class(type(arg))
        for arg in args
    ):
        raise AssertionError(f"Invalid arg types in {args}")
    partitioned_args: list[Any] = [[], []]
    pos = []
    for arg in args:
        idx = 0 if isinstance(arg, torch.Tensor) else 1
        partitioned_args[idx].append(arg)
        pos.append(idx)

    if hasattr(ctx, "non_tensor_args"):
        raise AssertionError("ctx already has non_tensor_args attribute.")
    if hasattr(ctx, "pos"):
        raise AssertionError("ctx already has pos attribute.")
    ctx.save_for_backward(*partitioned_args[0])
    ctx.non_tensor_args = partitioned_args[1]
    ctx.pos = pos


def saved_values(ctx):
    args = []
    t_idx = 0
    s_idx = 0
    saved_tensors = ctx.saved_tensors
    for p in ctx.pos:
        if p == 0:
            args.append(saved_tensors[t_idx])
            t_idx += 1
        else:
            args.append(ctx.non_tensor_args[s_idx])
            s_idx += 1
    if t_idx + s_idx != len(ctx.pos):
        raise AssertionError(
            f"t_idx ({t_idx}) + s_idx ({s_idx}) != len(ctx.pos) ({len(ctx.pos)})"
        )
    return tuple(args)


def split_into_chunks(iterable: Sequence[Any], chunk_sizes: list[int]) -> list[Any]:
    if sum(chunk_sizes) != len(iterable):
        raise AssertionError(
            f"the sum of all chunks ({sum(chunk_sizes)}) needs to match the length of the iterable ({len(iterable)})."
        )
    elements = []
    idx = 0
    for size in chunk_sizes:
        elements.append(iterable[idx : idx + size])
        idx += size
    return elements


def _clone_aliasing_output(inputs: Sequence[Any], outputs: Sequence[Any]):
    # For tensors whose grad is None, create zero tensors as gradients
    # This invariant is useful for cudagraph.

    # Elimitate input-output, output-output aliasing
    seen_input_storages = {
        StorageWeakRef(t._typed_storage())
        for t in inputs
        if isinstance(t, torch.Tensor)
    }
    seen_output_storages = set()
    final_outputs = []
    for out in outputs:
        if isinstance(out, torch.Tensor):
            out_storage = StorageWeakRef(out._typed_storage())
            if (
                out_storage in seen_input_storages
                or out_storage in seen_output_storages
            ):
                out = out.clone()
            seen_output_storages.add(StorageWeakRef(out._typed_storage()))
        final_outputs.append(out)
    return final_outputs


def create_bw_fn(
    fn: Callable, args: tuple[Any, ...], return_fw_outputs: bool = False
) -> Callable:
    """
    For a fn that accepts flat inputs and returns flat outputs:
        fw_out = fn(*args),
    this function returns:
        grad_args = bw_fn(*args_and_grad_output)
    with the following invariants:
      1. args + fw_out has an 1-1 correspondence to args_and_grad_output
      2. grad_args has an 1-1 corresponsence to args
      3. for tensor arg whose requires_grad is False, its corresponding grad in
         grad_args will be a zero tensor with the same shape.
    """

    from torch._functorch.aot_autograd import AOTConfig, create_joint

    # pyrefly: ignore [missing-module-attribute]
    from torch._higher_order_ops.utils import prepare_fw_with_masks_all_requires_grad

    dummy_aot_config = AOTConfig(
        fw_compiler=None,  # type: ignore[arg-type]
        bw_compiler=None,  # type: ignore[arg-type]
        partition_fn=None,  # type: ignore[arg-type]
        decompositions={},
        num_params_buffers=0,
        aot_id=0,
        keep_inference_input_mutations=False,
    )
    n_primals = len(args)

    bw_fn = create_joint(
        prepare_fw_with_masks_all_requires_grad(fn), aot_config=dummy_aot_config
    )

    def flat_fn(*args_and_grad_outs):
        primals = args_and_grad_outs[:n_primals]
        tangents = args_and_grad_outs[n_primals:]
        fw_outs, grad_args = bw_fn(primals, tangents)
        if len(args) != len(grad_args):
            raise AssertionError(
                f"Expected {len(args)} grad_args, got {len(grad_args)}"
            )

        # For tensors whose grad is None, create zero tensors as gradients
        # This invariant is useful for cudagraph.
        grad_args = [
            torch.zeros_like(arg)
            if isinstance(arg, torch.Tensor) and grad is None
            else grad
            for grad, arg in zip(grad_args, primals)
        ]

        final_grads = _clone_aliasing_output(args_and_grad_outs, grad_args)
        if return_fw_outputs:
            return *fw_outs, *final_grads
        return final_grads

    return flat_fn


def get_dummy_aot_autograd_config():
    from torch._functorch.aot_autograd import AOTConfig

    return AOTConfig(
        fw_compiler=None,  # type: ignore[arg-type]
        bw_compiler=None,  # type: ignore[arg-type]
        partition_fn=None,  # type: ignore[arg-type]
        decompositions={},
        num_params_buffers=0,
        aot_id=0,
        keep_inference_input_mutations=False,
    )


# Slices off the first element of a given dimension. Used to build a
# representative single slice for subgraph tracing of control-flow ops.
# When `dim` has size 0 (e.g. a zero-length scan) there is no real slice to
# take, so return an uninitialized tensor with `dim` removed.
def first_slice_copy(t: torch.Tensor, dim: int = 0) -> torch.Tensor:
    if t.shape[dim] == 0:
        shape = list(t.shape)
        del shape[dim]
        return t.new_zeros(shape)
    return torch.select_copy(t, dim, 0)


# Returns a mask whether a list element is a tensor or not
def get_tensor_mask(tensor_list: Iterable[Any]) -> list[bool]:
    return [bool(isinstance(v, torch.Tensor)) for v in tensor_list]


def mask_list(
    mask: list[bool], inp: list[Any], other: list[Any] | None = None
) -> list[Any]:
    # Masks elements on an `inp` list.
    # If other is None, then the elements of the `inp` list where the mask is False are removed
    # If other is not None, then the elements of the `inp` list where the mask is False are
    # replaced with the elements of the `other` list
    if len(mask) != len(inp):
        raise AssertionError(
            f"The length of the mask ({len(mask)}) needs to be identical to the length of the input ({len(inp)})"
        )
    if other is not None:
        if len(inp) != len(other):
            raise AssertionError(
                f"If an input and an other list is provided, they need to have the same length ({len(inp)} != {len(other)})"
            )
        return [i if m else o for m, i, o in zip(mask, inp, other)]
    else:
        return [i for m, i in zip(mask, inp) if m]


def first_slice_copy_with_grad(li: Iterable[Any]) -> list[Any]:
    # First_slice_copy does not keep the original requires_grad flag,
    # but we need it for materialize_as_graph
    # in order to compute the correct gradients
    # The reason why first_slice_copy doesn't keep requires_grad flag is
    # because it's called in torch.autograd.Function.backward/forward.
    slc = [first_slice_copy(x).requires_grad_(x.requires_grad) for x in li]
    return slc


# Reports the difference between meta of two tensors in a string
def diff_tensor_meta(
    meta1: TensorMetadata, meta2: TensorMetadata, check_grad=True
) -> list[str]:
    from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode

    pair_diffs = []
    for meta_name in TensorMetadata._fields:
        if not check_grad and meta_name == "requires_grad":
            continue
        val1 = getattr(meta1, meta_name)
        val2 = getattr(meta2, meta_name)
        try:
            if val1 != val2:
                pair_diffs.append(f"'{meta_name}: {val1} vs {val2}'")
        except GuardOnDataDependentSymNode:
            pair_diffs.append(f"'{meta_name}: {val1} vs {val2}'")
            continue
    return pair_diffs


# Note [lifted arg types in hop]
# For dynamoed hops, we automatically lift the free symbols in tensors as arguments.
# This has implications for the types of lifted args for different dispatch keys:
#   1. functionalization, FakeTensorMode, ProxyTorchDispatchMode, Autograd need to support torch.Symint
#      lifted args because it's on the path of torch.compile(dynamic=True).
#   2. functionalization, FakeTensorMode, ProxyTorchDispatchMode, Autograd, CompositeExplicitAutograd need
#      to support int arguments. In the eager run case, we re-trace the subgraph in AutogradKey, so inner
#      hops may receive int inputs from the shape of outer tensor inputs.
#      However, CompositeExplicitAutograd won't receive SymInt inputs because it only accepts real tensor inputs.
def validate_subgraph_args_types(lifted_args: tuple[Any, ...] | list[Any]):
    allowed_types = (torch.Tensor, int, torch.SymInt)
    if not all(
        isinstance(arg, (torch.Tensor, int, torch.SymInt)) for arg in lifted_args
    ):
        raise AssertionError(
            f"{lifted_args} can only be of {allowed_types} but got {tuple(type(arg) for arg in lifted_args)}"
        )


# TODO: Return a more detailed information as to which node
# causes a mutation or an alias. This may requires a per operator tensor version checking
def check_input_alias_and_mutation(
    gm: torch.fx.GraphModule,
    fake_args: list[FakeTensor],
) -> tuple[dict[int, int], dict[int, int], dict[int, int], list[int]]:
    (
        inp_inp_alias_map,
        inp_out_alias_map,
        out_out_alias_map,
        mutated_inputs,
    ) = check_input_alias_and_mutation_return_outputs(gm)[:-1]
    # pyrefly: ignore [bad-return]
    return inp_inp_alias_map, inp_out_alias_map, out_out_alias_map, mutated_inputs


def _tensor_storage(t) -> StorageWeakRef:
    return StorageWeakRef(t._typed_storage())


def _get_example_value(n):
    if not isinstance(n, torch.fx.Node):
        return n
    if "val" in n.meta:
        return n.meta["val"]
    if "example_value" in n.meta:
        return n.meta["example_value"]
    return None


def get_graph_output_example_values(gm: torch.fx.GraphModule) -> list[Any]:
    output_node = gm.graph.find_nodes(op="output")[0]
    return [_get_example_value(n) for n in pytree.tree_flatten(output_node.args[0])[0]]


def check_input_alias_and_mutation_return_outputs(
    gm: torch.fx.GraphModule,
) -> tuple[
    dict[int, int],
    dict[int, int],
    dict[int, int],
    list[int],
    tuple[Any, ...] | list[Any],
]:
    fake_args = [
        _get_example_value(n)
        for n in gm.graph.find_nodes(op="placeholder")
        if isinstance(n, torch.fx.Node) and "val" in n.meta
    ]
    outputs = get_graph_output_example_values(gm)

    # We need to analyze the original fake_args to detect
    # inp-inp alias.
    inp_storage_map = {
        _tensor_storage(inp): i
        for i, inp in enumerate(fake_args)
        if isinstance(inp, torch.Tensor)
    }
    out_storage_map = {
        _tensor_storage(out): i
        for i, out in enumerate(outputs)
        if isinstance(out, torch.Tensor)
    }
    inp_inp_alias_map = {
        i: inp_storage_map[_tensor_storage(inp)]
        for i, inp in enumerate(fake_args)
        if isinstance(inp, torch.Tensor) and inp_storage_map[_tensor_storage(inp)] != i
    }
    out_out_alias_map = {
        i: out_storage_map[_tensor_storage(out)]
        for i, out in enumerate(outputs)
        if isinstance(out, torch.Tensor) and out_storage_map[_tensor_storage(out)] != i
    }
    inp_out_alias_map = {
        i: out_storage_map[_tensor_storage(inp)]
        for i, inp in enumerate(fake_args)
        if isinstance(inp, torch.Tensor) and _tensor_storage(inp) in out_storage_map
    }
    mutated_inputs = []
    for node in gm.graph.nodes:
        if node.op == "call_function" and isinstance(
            node.target, torch._ops.OpOverload
        ):
            for arg_node, arg_schema in zip(node.args, node.target._schema.arguments):
                if arg_schema.is_write:
                    arg_val = _get_example_value(arg_node)
                    if not isinstance(arg_val, torch.Tensor):
                        raise AssertionError(
                            f"Expected arg_val to be a Tensor, got {type(arg_val)}"
                        )
                    if _tensor_storage(arg_val) in inp_storage_map:
                        mutated_inputs.append(inp_storage_map[_tensor_storage(arg_val)])

    return (
        inp_inp_alias_map,
        inp_out_alias_map,
        out_out_alias_map,
        mutated_inputs,
        outputs,
    )


registered_hop_fake_fns: dict[torch._ops.OpOverload, Callable] = {}

# Set skip_cache=True for HOPs that should not be cached in FakeTensorMode
# (previously these ops were registered with .py_impl(FakeTensorMode)) and
# were never cached so we are just preserving original behaviour
hops_that_skip_faketensor_cache: set[torch._ops.OpOverload] = set()


F = TypeVar("F", bound=Callable)
_P = ParamSpec("_P")


@overload
def register_fake(
    hop, fn: None = None, *, skip_cache: bool = ...
) -> Callable[[F], F]: ...


@overload
def register_fake(hop, fn: F, *, skip_cache: bool = ...) -> F: ...


def register_fake(hop, fn=None, *, skip_cache=False):
    """
    Register a fake function for a HOP. This is conceptually equivalent of the
    register_fake utility for the custom ops. The registered function is called
    inside the fake_tensor _dispatch_impl.

    Set skip_cache=True to keep a HOP's fake output out of the FakeTensorMode
    cache -- either to preserve the historical uncached behavior of
    py_impl(FakeTensorMode), or because its outputs alias in ways the cache
    cannot reconstruct (e.g. auto_functionalized).
    """
    if hop in registered_hop_fake_fns:
        raise AssertionError(f"hop {hop} already registered in registered_hop_fake_fns")

    def register(func: F) -> F:
        from torch._subclasses.fake_tensor import FakeTensorMode

        redirect_to_mode(hop, FakeTensorMode)

        registered_hop_fake_fns[hop] = func
        if skip_cache:
            hops_that_skip_faketensor_cache.add(hop)
        return func

    if fn is None:
        return register
    return register(fn)


class SubgraphCallableWrapper:
    """
    Wraps a callable while preserving the original subgraph identity for
    fake tensor caching. Without this, passing a plain closure to
    invoke_subgraph would bypass the dispatch cache entirely.
    """

    def __init__(self, fn, subgraph):
        self.fn = fn
        self.subgraph = subgraph
        self._boxed_call = getattr(subgraph, "_boxed_call", False)

    def __hash__(self):
        return id(self.subgraph)

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class FunctionalizeCtxWrapper(SubgraphCallableWrapper):
    """Functionalizes a subgraph and optionally returns selected input updates.

    Each update is a new tensor; the original input is not mutated.
    """

    # Prevents PYTORCH_TEST_WITH_DYNAMO=1 test failures
    @torch._disable_dynamo
    def __init__(self, ctx, subgraph, mutated_input_indices=()):
        super().__init__(subgraph, subgraph)
        self.ctx = ctx
        self.mutated_input_indices = mutated_input_indices

    def __repr__(self):
        return f"FunctionalizeCtxWrapper on subgraph {self.subgraph})"

    def __call__(self, *args, **kwargs):
        def append_mutated_inputs(fn):
            if not self.mutated_input_indices:
                return fn

            def wrapped(*args, **kwargs):
                inputs = args[0] if self._boxed_call else args
                # Snapshot before a boxed GraphModule clears its input list.
                mutated_inputs = tuple(inputs[i] for i in self.mutated_input_indices)
                # Dynamo normalizes invoke_subgraph outputs to tuples.
                return (*fn(*args, **kwargs), *mutated_inputs)

            return wrapped

        if isinstance(self.subgraph, torch.fx.GraphModule):
            if self._boxed_call:
                # Not all callers respect _boxed_call (e.g. reenter_make_fx).
                functionalized = self.ctx.functionalize(
                    append_mutated_inputs(self.subgraph)
                )
                if len(args) == 1 and isinstance(args[0], list):
                    return functionalized(args[0])
                return functionalized(list(args))
            else:
                # Running graph with interpreter is needed for propagating the stack_trace
                with fx_traceback.preserve_node_meta():
                    return self.ctx.functionalize(
                        append_mutated_inputs(torch.fx.Interpreter(self.subgraph).run)
                    )(*args, **kwargs)
        functionalized = self.ctx.functionalize(append_mutated_inputs(self.subgraph))
        if self._boxed_call:
            if len(args) == 1 and isinstance(args[0], list):
                return functionalized(args[0])
            return functionalized(list(args))
        return functionalized(*args, **kwargs)


# A wrapper over HigherOrderOperator that also carries its schema
class HopInstance:
    def __init__(self, op: HigherOrderOperator, schema: HopSchema):
        if not isinstance(op, HigherOrderOperator):
            raise AssertionError(f"Expected HigherOrderOperator, got {type(op)}")
        self._op = op
        # Using "_" to be consistent with how we access _schema of OpOverload
        self._schema = schema

    def __call__(self, *args, **kwargs):
        return self._op(*args, **kwargs)

    @staticmethod
    def create(hop: HigherOrderOperator, *args, **kwargs):
        return HopInstance(hop, hop.gen_schema(*args, **kwargs))


# This call_op can be used to call a HopInstance with
# flat args and kwargs. We need to make use of the hop's schema's tree_spec
# to unflatten the args and kwargs before calling the hop.
def call_op(op: OpOverload | HopInstance, args, kwargs):
    if isinstance(op, OpOverload):
        return op(*args, **kwargs)

    if not isinstance(op, HopInstance):
        raise AssertionError(f"Expected HopInstance, got {type(op)}")
    schema = op._schema
    bound_args = list(args)
    bound_kwargs = {}
    for arg in schema.arguments[len(bound_args) :]:
        if arg.name not in kwargs:
            raise AssertionError(f"arg {arg.name} not in kwargs: {kwargs}")
        val = kwargs[arg.name]
        if not arg.kwarg_only:
            bound_args.append(val)
        else:
            bound_kwargs[arg.name] = val

    if schema.tree_spec is not None:
        if len(bound_args) != len(schema.arguments) or len(bound_kwargs) != 0:
            raise AssertionError(
                f"Expected {len(schema.arguments)} bound_args and 0 bound_kwargs, "
                f"got {len(bound_args)} and {len(bound_kwargs)}"
            )
        args, kwargs = pytree.tree_unflatten(bound_args, schema.tree_spec)
        return op(*args, **kwargs)
    else:
        if len(bound_args) + len(bound_kwargs) != len(schema.arguments):
            raise AssertionError(
                f"Expected {len(schema.arguments)} total args, "
                f"got {len(bound_args)} + {len(bound_kwargs)}"
            )
        return op(*bound_args, **bound_kwargs)


def materialize_as_graph(
    fn: Callable,
    args: tuple[Any, ...],
    include_key_set: torch._C.DispatchKeySet | None = None,
    exclude_key_set: torch._C.DispatchKeySet | None = None,
    force_enable_grad=False,
    subgraph_decomp_table: Mapping[OpOverload, Callable] | None = None,
) -> torch.fx.GraphModule:
    if include_key_set is None:
        include_key_set = torch._C._dispatch_tls_local_include_set()
    if exclude_key_set is None:
        exclude_key_set = torch._C._dispatch_tls_local_exclude_set()

    @torch._dynamo.disable(recursive=True, reason=None)
    def _materialize_as_graph_inner():
        from torch._guards import active_fake_mode
        from torch.fx.experimental.proxy_tensor import _CURRENT_MAKE_FX_TRACER

        with suspend_functionalization(), disable_functional_mode():
            fake_mode = None
            if _CURRENT_MAKE_FX_TRACER is not None:
                fake_mode = _CURRENT_MAKE_FX_TRACER.fake_tensor_mode
            if fake_mode is None:
                fake_mode = active_fake_mode()

            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    torch.utils._python_dispatch._disable_current_modes()
                )
                if fake_mode is not None:
                    stack.enter_context(fake_mode)

                with disable_proxy_modes_tracing():
                    unfunc_t = [_from_fun(arg) for arg in args]

                stack.enter_context(
                    torch._C._ForceDispatchKeyGuard(include_key_set, exclude_key_set),
                )
                if force_enable_grad:
                    stack.enter_context(torch.enable_grad())
                return _maybe_reenter_make_fx(
                    fn, subgraph_decomp_table=subgraph_decomp_table
                )(*unfunc_t)

    gm = _materialize_as_graph_inner()
    if gm is None:
        raise AssertionError("materialize_as_graph returned None")
    return gm


def materialize_callable_in_args(op: HopInstance, args, kwargs):
    schema = op._schema
    hop = op._op
    flat_args, flat_spec = pytree.tree_flatten((args, kwargs))

    def wrapped_fn(*flat_args):
        return call_op(op, args, kwargs)

    # We need to trace the higher order op in order to materilaize the callable inputs that
    # are a callable (e.g. after functionalization key)
    gm = reenter_make_fx(wrapped_fn)(*flat_args)
    hop_node = gm.graph.find_nodes(op="call_function", target=hop)[0]
    arg_proxies = pytree.tree_leaves((hop_node.args, hop_node.kwargs))
    if not isinstance(schema, torch._C.FunctionSchema) or len(arg_proxies) != len(
        schema.arguments
    ):
        raise AssertionError(
            f"Expected FunctionSchema with {len(arg_proxies)} arguments"
        )

    # call_op preserves ordering of proxies via schema
    materialized_args = []
    for i, proxy in enumerate(arg_proxies):
        if (
            isinstance(proxy, torch.fx.Node)
            and proxy.op == "get_attr"
            and isinstance(getattr(gm, proxy.target), torch.fx.GraphModule)  # type: ignore[arg-type]
        ):
            if not callable(flat_args[i]):
                raise AssertionError(
                    f"Expected flat_args[{i}] to be callable for {schema}"
                )
            materialized_args.append(getattr(gm, proxy.target))  # type: ignore[arg-type]
        else:
            materialized_args.append(flat_args[i])

    return pytree.tree_unflatten(materialized_args, flat_spec)


def has_user_subclass(args, allowed_subclasses):
    """Check if any tensor arguments are user subclasses.

    This is used to determine if tensor subclasses should get a chance to run
    their own implementation first before falling back to the default implementation.

    Args:
        args: Arguments to check (will be flattened with pytree)
        allowed_subclasses: Tuple of allowed subclass types

    Returns:
        True if user tensor subclasses are found, False otherwise
    """
    flat_args, _ = pytree.tree_flatten(args)

    val = any(
        isinstance(a, torch.Tensor)
        and type(a) is not torch.Tensor
        and not isinstance(a, allowed_subclasses)
        for a in flat_args
    )
    return val


def _has_gen_schema(op: HigherOrderOperator):
    # There is an InvokeQuant argument we cannot gen_schema.
    if op is torch.ops.higher_order.invoke_quant_packed:
        return False
    method = "gen_schema"
    return hasattr(type(op), method) and getattr(type(op), method) is not getattr(
        HigherOrderOperator, method
    )


def filter_with_masks(data: list[torch.Tensor | None], masks: list[bool]):
    if len(data) != len(masks):
        raise AssertionError(
            f"data length ({len(data)}) != masks length ({len(masks)})"
        )
    return [item for item, keep in zip(data, masks) if keep]


def fill_none_with_masks(data: list[torch.Tensor | None], masks: list[bool]):
    data_iter = iter(data)
    return [next(data_iter) if kept else None for kept in masks]


def create_fn_remove_none(
    fn: Callable[_P, Any],
) -> tuple[Callable[_P, list[torch.Tensor]], list[bool]]:
    """Wrap ``fn`` so its non-Tensor output leaves are dropped, and expose the
    mask of which leaves survived.

    Returns ``(wrapped, mask)``:
      - ``wrapped(*args)`` calls ``fn(*args)``, flattens the pytree result and
        returns only its Tensor leaves as a list.
      - ``mask`` is a list populated whenever ``wrapped`` runs -- one bool per
        leaf, ``True`` where the leaf is a Tensor. Callers should read it
        AFTER invoking ``wrapped`` (typically indirectly via
        ``materialize_as_graph``), then pass it to ``fill_none_with_masks``
        to reconstruct the full output with ``None`` at the dropped slots.

    Used in HOP autograd impls in two places:
      - Around a forward branch that may return non-Tensor leaves (None,
        int/SymInt), so ``create_bw_fn``'s joint sees a tensor-only signature.
      - Around the resulting backward joint whose grad list contains ``None``
        at non-differentiable input slots, so the materialized ``GraphModule``
        returns a uniform Tensor list.
    """
    mask: list[bool] = []

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        leaves = pytree.tree_leaves(fn(*args, **kwargs))
        mask.clear()
        mask.extend(isinstance(o, torch.Tensor) for o in leaves)
        return filter_with_masks(leaves, mask)

    return wrapped, mask
<|endoftext|># mypy: allow-untyped-defs
from typing import Any, cast, Union

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.invoke_leaf_function import invoke_leaf_function
from torch._higher_order_ops.print import print as hop_print
from torch._higher_order_ops.schema import HopSchema
from torch._higher_order_ops.torchbind import call_torchbind
from torch._higher_order_ops.utils import register_fake
from torch._library.custom_ops import CustomOpDef
from torch._library.effects import EffectType
from torch._library.utils import RegistrationHandle
from torch._ops import HigherOrderOperator
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)


_op_identifier = Union[
    str,
    "torch._ops.OpOverload",
    "torch._library.custom_ops.CustomOpDef",
    "torch._ops.HigherOrderOperator",
]
OpType = Union["torch._ops.HigherOrderOperator", "torch._ops.OpOverload"]

_EffectType = EffectType


def _get_op_qualname(op: _op_identifier) -> str:
    """Convert an op identifier to a qualified string key."""
    if isinstance(op, torch._ops.OpOverload):
        return op._name
    elif isinstance(op, torch._ops.HigherOrderOperator):
        return f"{op.namespace}::{op.name()}"
    elif isinstance(op, CustomOpDef):
        return op._qualname
    elif isinstance(op, str):
        return op

    raise ValueError(f"Invalid operator input {op}")


def _register_effectful_op(
    op: _op_identifier, effect: EffectType | None
) -> RegistrationHandle:
    qualname = _get_op_qualname(op)
    entry = torch._library.simple_registry.singleton.find(qualname)
    handle = entry.effect.register(effect)
    return handle


def _get_effect(op: _op_identifier) -> _EffectType | None:
    qualname = _get_op_qualname(op)
    entry = torch._library.simple_registry.singleton.find(qualname)
    return entry.effect.effect


_register_effectful_op("aten::_print", _EffectType.ORDERED)
# _linalg_check_errors has no tensor output, but it raises on invalid linalg
# results and must be preserved by functionalization/DCE.
_register_effectful_op("aten::_linalg_check_errors", _EffectType.ORDERED)
_register_effectful_op("profiler::_record_function_exit._RecordFunction", None)
_register_effectful_op(call_torchbind, _EffectType.ORDERED)
_register_effectful_op(hop_print, _EffectType.ORDERED)
_register_effectful_op(invoke_leaf_function, _EffectType.ORDERED)


class WithEffects(HigherOrderOperator):
    """
    with_effects(token, op, args, kwargs) -> (new_token, op_results)

    This HOP helps ensure ordering between side effectful ops like prints or ops
    using torchbind objects. This is needed to ensure a traced graph from
    AOTAutograd is functional so that future optimization passes do not reorder
    these operators. This is done through threading "effect tokens" through the
    graph to enforce data dependence between side effectful ops.

    The tokens are basically dummy values (torch.tensor([])). We create a token
    per "effect type", which are enumerated in the _EffectType enum.
    """

    def __init__(self) -> None:
        super().__init__("with_effects")

    def __call__(
        self,
        token,
        op: OpType,
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any],
    ) -> tuple[Any, ...]:
        if not isinstance(op, (torch._ops.HigherOrderOperator, torch._ops.OpOverload)):
            raise AssertionError(
                f"op must be HigherOrderOperator or OpOverload, got {type(op)}"
            )
        if has_aliasing(op):
            raise AssertionError("Ops with aliasing is not supported")
        if not isinstance(kwargs, dict):
            raise AssertionError(f"kwargs must be a dict, got {type(kwargs)}")
        # pyrefly: ignore [missing-attribute]
        return super().__call__(token, op, *args, **kwargs)


with_effects = WithEffects()


def has_aliasing(op: OpType):
    # NOT FOR PUBLIC USE
    if isinstance(op, torch._ops.HigherOrderOperator):
        return False

    for arg in op._schema.arguments:
        if arg.alias_info is not None:
            return True
    for arg in op._schema.returns:
        if arg.alias_info is not None:
            return True
    return False


def has_effects(op) -> bool:
    return (
        isinstance(op, (torch._ops.HigherOrderOperator, torch._ops.OpOverload))
        and not has_aliasing(op)
        and _get_effect(op) is not None
    )


def new_token_tensor() -> torch.Tensor:
    return torch.tensor([])


@with_effects.py_impl(DispatchKey.CompositeExplicitAutograd)
def with_effects_dense(
    token: torch.Tensor,
    op: torch._ops.OpOverload,
    *args: tuple[Any, ...],
    **kwargs: dict[str, Any],
) -> tuple[torch.Tensor, ...]:
    out = op(*args, **kwargs)
    new_token = new_token_tensor()
    # [NOTE: with_effects return type]
    # Note that we should only do *out for tuple type, but not list type.
    # This is to match the schema of the op.
    # For tuple output, the length of schema output is the same as the length of out.
    # For list output, the length of schema output is 1 (e.g. Tensor[]) regardless of the
    # length of the list.
    if isinstance(out, tuple):
        return (new_token, *out)
    return (new_token, out)


@register_fake(with_effects, skip_cache=True)
def with_effects_fake(
    token: torch.Tensor,
    op: torch._ops.OpOverload,
    *args: tuple[Any, ...],
    **kwargs: dict[str, Any],
) -> tuple[torch.Tensor, ...]:
    result = with_effects_dense(token, op, *args, **kwargs)
    return result


@with_effects.py_impl(ProxyTorchDispatchMode)
def with_effects_proxy(
    mode,
    token: torch.Tensor,
    op: torch._ops.OpOverload,
    *args: tuple[Any, ...],
    **kwargs: dict[str, Any],
) -> tuple[torch.Tensor, ...]:
    with disable_proxy_modes_tracing():
        out = with_effects(token, op, *args, **kwargs)

    proxy_token = mode.tracer.unwrap_proxy(token)
    proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, args)
    proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)

    from torch.fx.node import has_side_effect

    # To avoid the being DCEed by graph.eliminate_dead_code if they.
    # don't have output or their outputs are not used.
    has_side_effect(op)

    out_proxy = mode.tracer.create_proxy(
        "call_function",
        with_effects,
        (proxy_token, op, *proxy_args),
        proxy_kwargs,
    )
    result = track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)
    return result


with_effects.fallthrough(DispatchKey.AutogradCPU)
with_effects.fallthrough(DispatchKey.AutogradCUDA)


@with_effects.py_functionalize_impl
def with_effects_functional(
    ctx,
    token: torch.Tensor,
    op: torch._ops.OpOverload,
    *args: tuple[Any, ...],
    **kwargs: dict[str, Any],
) -> tuple[torch.Tensor, ...]:
    # with_effects is already functional, so just re-emit it.
    unwrapped_token, unwrapped_args, unwrapped_kwargs = ctx.unwrap_tensors(
        [token, args, kwargs]
    )
    with ctx.redispatch_to_next():
        result = with_effects(unwrapped_token, op, *unwrapped_args, **unwrapped_kwargs)
    return ctx.wrap_tensors(result)


_EFFECTFUL_HOPS_WITH_SCHEMA = {hop_print, invoke_leaf_function}


def _get_schema(op, args, kwargs: dict | None = None) -> torch.FunctionSchema:
    if isinstance(op, torch._ops.OpOverload):
        return op._schema
    elif op == call_torchbind:
        return getattr(args[0], args[1]).schema
    elif op in _EFFECTFUL_HOPS_WITH_SCHEMA:
        extra_kwargs = kwargs or {}
        return op.gen_schema(*args, **extra_kwargs)
    else:
        raise RuntimeError(f"Unable to get schema for op {op}")


def handle_effects(
    allow_token_discovery: bool,
    tokens: dict[_EffectType, torch.Tensor],
    op: OpType,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """
    Args:
        allow_token_discovery: Whether or not we are discovering tokens. If this
        is true, we will create a token for every side effect type seen that
        does not have a token assigned yet.  If this is false, the tokens
        should've all been created ahead of time, so we will error if there is
        no token mapping to every effect type.

        tokens: Map of effect type to tokens. This is to chain operators of the
        same effects together so that they do not get reordered in later
        optimization passes.
    """

    # Get a token. We can't do `tokens.get(op, torch.tensor([]))` because
    # this will create an empty tensor during proxy mode tracing if the token
    # doesn't exist. But the tokens should always exist during proxy mode tracing.
    key = _get_effect(op)
    if key is None:
        raise AssertionError(f"effect key must not be None for op {op}")
    if key not in tokens:
        if not allow_token_discovery:
            raise AssertionError(
                f"Could not find a token for effect {key} which came from the function {op}"
            )
        proxy_tensor_mode = cast(
            "ProxyTorchDispatchMode | None",
            torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY),
        )
        if proxy_tensor_mode is not None:
            # If we discovered a new token during tracing, we are in backward.
            # Then we patch the graph, adding additional tangents_token as input to the joint graph.
            tracer = proxy_tensor_mode.tracer

            from torch.fx.experimental.proxy_tensor import (
                disable_proxy_modes_tracing,
                track_tensor_tree,
            )

            with disable_proxy_modes_tracing():
                token_tensor = new_token_tensor()

            token_proxy = proxy_tensor_mode.tracer.create_proxy(
                "placeholder", "tangents_token", (), {}, name="tangents_token"
            )
            track_tensor_tree(token_tensor, token_proxy, constant=None, tracer=tracer)

            tokens[key] = token_tensor
        else:
            tokens[key] = new_token_tensor()

    token = tokens[key]

    from torch._subclasses.functional_tensor import PythonFunctionalizeAPI

    ctx = PythonFunctionalizeAPI()

    unwrapped_token = ctx.unwrap_tensors([token])[0]
    unwrapped_args = ctx.unwrap_tensors(args)
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)  # type: ignore[arg-type]
    with ctx.redispatch_to_next():
        (new_token, *unwrapped_outs) = with_effects(
            unwrapped_token, op, *unwrapped_args, **unwrapped_kwargs
        )

    schema = _get_schema(op, unwrapped_args, unwrapped_kwargs)

    if isinstance(schema, HopSchema):
        if len(schema.returns) == 0:
            unwrapped_outs = ()
        else:
            if len(unwrapped_outs) != len(schema.returns):
                raise AssertionError(
                    f"expected {len(schema.returns)} outputs but got {len(unwrapped_outs)}"
                )
            unwrapped_outs = tuple(unwrapped_outs)
    elif len(schema.returns) == 0:
        if unwrapped_outs[0] is not None:
            raise AssertionError(f"expected no outputs but got {unwrapped_outs[0]}")
        unwrapped_outs = None  # type: ignore[assignment]
    elif len(schema.returns) == 1:
        if len(unwrapped_outs) != 1:
            raise AssertionError(f"expected 1 output but got {len(unwrapped_outs)}")
        unwrapped_outs = unwrapped_outs[0]
    else:
        if len(unwrapped_outs) != len(schema.returns):
            raise AssertionError(
                f"expected {len(schema.returns)} outputs but got {len(unwrapped_outs)}"
            )

    # Add the newly created token into the tokens map for a following call to
    # use this token.
    wrapped_token = ctx.wrap_tensors(new_token)
    if not isinstance(wrapped_token, torch.Tensor):
        raise AssertionError(
            f"expected wrapped_token to be torch.Tensor, got {type(wrapped_token)}"
        )
    tokens[key] = wrapped_token

    # pyrefly: ignore [bad-argument-type]
    return ctx.wrap_tensors(unwrapped_outs)
<|endoftext|># mypy: allow-untyped-defs
import dataclasses
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, cast

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch._higher_order_ops.utils import (
    _check_alias_and_mutation,
    autograd_not_implemented,
    reenter_make_fx,
    register_fake,
    unique_graph_id,
)
from torch._ops import HigherOrderOperator
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND
from torch._prims_common.wrappers import elementwise_type_promotion_wrapper
from torch._subclasses.fake_tensor import is_fake
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree


@dataclasses.dataclass(frozen=True)
class FlexGemmOpSpec:
    """Canonical operand positions for a supported FlexGEMM op."""

    name: str
    mat1_index: int
    mat2_index: int
    input_ndim: int
    bias_index: int | None = None


FLEX_GEMM_OP_SPECS = {
    torch.ops.aten.mm.default: FlexGemmOpSpec("mm", 0, 1, input_ndim=2),
    torch.ops.aten.addmm.default: FlexGemmOpSpec(
        "addmm", 1, 2, input_ndim=2, bias_index=0
    ),
    torch.ops.aten.bmm.default: FlexGemmOpSpec("bmm", 0, 1, input_ndim=3),
    torch.ops.aten.baddbmm.default: FlexGemmOpSpec(
        "baddbmm", 1, 2, input_ndim=3, bias_index=0
    ),
}
FLEX_GEMM_OP_ALIASES = {
    torch.mm: torch.ops.aten.mm.default,
    torch.addmm: torch.ops.aten.addmm.default,
    torch.bmm: torch.ops.aten.bmm.default,
    torch.baddbmm: torch.ops.aten.baddbmm.default,
}
_SUPPORTED_BACKENDS = {"NVGEMM", "QUACK", "TRITON"}


_SUPPORTED_FLEX_GEMM_OP_NAMES = "/".join(
    spec.name for spec in FLEX_GEMM_OP_SPECS.values()
)
_PRESERVE_FLEX_GEMM_GEMM_OP = "preserve_flex_gemm_gemm_op"

# Note [Preserving FlexGEMM body GEMMs]
# FlexGEMM lowering materializes the captured epilogue by finding the body GEMM
# node whose target matches the HOP-carried gemm_op. Generic graph passes can
# rewrite batch-size-1 bmm into mm(...).unsqueeze(0), which removes the matching
# node. This body pass tags FlexGEMM GEMM nodes so bmm_to_mm in joint_graph.py
# skips them.


def mark_flex_gemm_body_gemm_node(
    body_graph: torch.fx.GraphModule, gemm_op: torch._ops.OpOverload
) -> None:
    """Mark body GEMMs so Inductor's batch-1 bmm rewrite keeps them matchable."""
    for node in body_graph.graph.find_nodes(op="call_function", target=gemm_op):
        node.meta[_PRESERVE_FLEX_GEMM_GEMM_OP] = True


FLEX_GEMM_BODY_GRAPH_PASSES: tuple[
    Callable[[torch.fx.GraphModule, torch._ops.OpOverload], None], ...
] = (mark_flex_gemm_body_gemm_node,)


@elementwise_type_promotion_wrapper(
    type_promoting_args=("x",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def flex_gemm_fast_math_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Use the tanh sigmoid identity selected by QUACK fast math."""
    return torch.tanh(x * 0.5) * 0.5 + 0.5


@elementwise_type_promotion_wrapper(
    type_promoting_args=("x",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def flex_gemm_fast_math_silu(x: torch.Tensor) -> torch.Tensor:
    """Use the tanh SiLU identity selected by QUACK fast math."""
    half = x * 0.5
    return half * torch.tanh(half) + half


def flex_gemm_fast_math_gelu(
    x: torch.Tensor,
    approximate: str = "none",
    *,
    fallback: Callable[..., Any],
) -> torch.Tensor:
    """Select the standard tanh GELU approximation for exact GELU calls."""
    if approximate == "none":
        approximate = "tanh"
    return fallback(x, approximate)


FLEX_GEMM_FAST_MATH_DECOMPOSITIONS: dict[torch._ops.OpOverload, Callable[..., Any]] = {
    torch.ops.aten.sigmoid.default: flex_gemm_fast_math_sigmoid,
    torch.ops.aten.silu.default: flex_gemm_fast_math_silu,
}


def flex_gemm_body_decomposition_table(
    kernel_options: dict[str, Any],
    decomposition_table: Mapping[torch._ops.OpOverload, Callable[..., Any]],
) -> dict[torch._ops.OpOverload, Callable[..., Any]] | None:
    """Override composite body decompositions enabled by QUACK fast math."""
    if (
        kernel_options.get("backend") != "QUACK"
        or kernel_options.get("fast_math") is not True
    ):
        return None
    merged_decompositions = dict(decomposition_table)
    merged_decompositions.update(FLEX_GEMM_FAST_MATH_DECOMPOSITIONS)
    gelu = torch.ops.aten.gelu.default
    if gelu in merged_decompositions:
        merged_decompositions[gelu] = partial(
            flex_gemm_fast_math_gelu, fallback=merged_decompositions[gelu]
        )
    return merged_decompositions


def nvfp4_pack(input: torch.Tensor) -> torch.Tensor:
    """Pack adjacent Float32 values into E2M1x2 storage for FlexGEMM tests.

    This private helper lets tests spell the intended epilogue with public inline
    assembly while a blessed user-facing quantization API is still being decided.
    The last dimension is one low/high pair: the PTX x2 conversion consumes both
    values and emits one byte, whose same-width dtype view supplies the logical
    ``torch.float4_e2m1fn_x2`` type without changing storage.
    """
    if input.dtype is not torch.float32 or input.ndim == 0 or input.shape[-1] != 2:
        raise ValueError(
            "NVFP4 pack input must be Float32 with innermost dimension 2, "
            f"got dtype={input.dtype}, shape={tuple(input.shape)}"
        )
    if torch.compiler.is_compiling() or is_fake(input):
        low = input[..., 0]
        high = input[..., 1]
        packed = inline_asm_elementwise(
            high,
            low,
            asm_str=(
                "{ .reg .b8 packed; "
                "cvt.rn.satfinite.e2m1x2.f32 packed, $1, $2; "
                "cvt.u32.u8 $0, packed; }"
            ),
            constraints="=r,r,r",
            dtype=torch.uint8,
        )
        return packed.contiguous().view(torch.float4_e2m1fn_x2)

    # Eager path is mostly for testing.
    input_bits = input.view(torch.int32)
    sign = input_bits & 0x80000000
    magnitude = (input_bits ^ sign).view(torch.float32)
    saturated = magnitude >= 6.0
    denormal = (~saturated) & (magnitude < 1.0)
    normal = ~(saturated | denormal)
    denormal_code = ((magnitude + 4194304.0).view(torch.int32) - 1249902592).to(
        torch.uint8
    )
    normal_bits = magnitude.view(torch.int32)
    mantissa_odd = (normal_bits >> 22) & 1
    normal_code = ((normal_bits - 1054867457 + mantissa_odd) >> 22).to(torch.uint8)
    codes = torch.full_like(magnitude, 7, dtype=torch.uint8)
    codes = torch.where(denormal, denormal_code, codes)
    codes = torch.where(normal, normal_code, codes)
    codes = codes | (((sign >> 28).to(torch.uint8)) & 8)
    packed = (codes[..., 1] << 4) | codes[..., 0]
    return packed.contiguous().view(torch.float4_e2m1fn_x2)


def apply_flex_gemm_body_graph_passes(
    body_graph: torch.fx.GraphModule, gemm_op: torch._ops.OpOverload
) -> None:
    """Apply FlexGEMM body annotations before generic Inductor graph passes."""
    for graph_pass in FLEX_GEMM_BODY_GRAPH_PASSES:
        graph_pass(body_graph, gemm_op)


class FlexGemm(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("flex_gemm")

    def __call__(
        self,
        gemm_op: torch._ops.OpOverload,
        body_fn: Callable[..., Any],
        gemm_args: tuple[Any, ...],
        gemm_kwargs: dict[str, Any],
        kernel_options: dict[str, Any],
    ) -> Any:
        if gemm_op not in FLEX_GEMM_OP_SPECS:
            raise RuntimeError(f"unsupported GEMM op for FlexGEMM: {gemm_op}")
        if not isinstance(gemm_args, (tuple, list)):
            raise RuntimeError(f"gemm_args must be a tuple/list, got {type(gemm_args)}")
        gemm_args = tuple(gemm_args)
        if not all(
            isinstance(
                arg,
                (
                    torch.Tensor,
                    torch.SymInt,
                    torch.SymFloat,
                    torch.SymBool,
                    int,
                    float,
                    bool,
                ),
            )
            for arg in gemm_args
        ):
            raise RuntimeError(
                "gemm_args must be a tuple of tensors, SymInts, SymFloats, "
                f"SymBools, ints, floats, or bools, got {gemm_args}"
            )
        if not isinstance(gemm_kwargs, dict):
            raise RuntimeError(f"gemm_kwargs must be a dict, got {type(gemm_kwargs)}")
        if any(
            isinstance(value, torch.Tensor) for value in pytree.tree_leaves(gemm_kwargs)
        ):
            raise RuntimeError(
                "gemm_kwargs must not contain tensor values; pass tensor inputs through gemm_args"
            )
        if not isinstance(kernel_options, dict):
            raise RuntimeError(
                f"kernel_options must be a dict, got {type(kernel_options)}"
            )

        kernel_options = dict(kernel_options)
        kernel_options.setdefault("backend", "TRITON")
        backend = kernel_options["backend"]
        if backend not in _SUPPORTED_BACKENDS:
            raise RuntimeError(
                f"unsupported FlexGEMM backend: {backend}; "
                f"expected one of {sorted(_SUPPORTED_BACKENDS)}"
            )
        return super().__call__(
            gemm_op, body_fn, gemm_args, gemm_kwargs, kernel_options
        )


flex_gemm_hop = FlexGemm()


def flex_gemm(
    gemm_op: Callable[..., Any],
    gemm_args: tuple[Any, ...],
    epilogue_fn: Callable[[Any], Any],
    *,
    gemm_kwargs: dict[str, Any] | None = None,
    kernel_options: dict[str, Any] | None = None,
) -> Any:
    if gemm_kwargs is None:
        gemm_kwargs = {}
    if kernel_options is None:
        kernel_options = {}
    gemm_op = cast(torch._ops.OpOverload, FLEX_GEMM_OP_ALIASES.get(gemm_op, gemm_op))

    def body_fn(*args: Any) -> Any:
        # Keep the traced body positional-only; the HOP carries gemm_kwargs for lowering.
        return epilogue_fn(gemm_op(*args, **gemm_kwargs))

    return flex_gemm_hop(gemm_op, body_fn, gemm_args, gemm_kwargs, kernel_options)


@flex_gemm_hop.py_impl(DispatchKey.CompositeExplicitAutograd)
def flex_gemm_dense(gemm_op, body_fn, args, kwargs, kernel_options):
    return body_fn(*args)


flex_gemm_hop.py_autograd_impl(
    autograd_not_implemented(flex_gemm_hop, deferred_error=True)
)


@register_fake(flex_gemm_hop)
def flex_gemm_fake_tensor_mode(gemm_op, body_fn, args, kwargs, kernel_options):
    return body_fn(*args)


@flex_gemm_hop.py_functionalize_impl
def flex_gemm_functionalize(ctx, gemm_op, body_fn, args, kwargs, kernel_options):
    unwrapped_args = ctx.unwrap_tensors(args)
    with ctx.redispatch_to_next():
        _check_alias_and_mutation(
            body_fn,
            unwrapped_args,
            "flex_gemm",
            hasattr(ctx, "mode") and ctx.mode.pre_dispatch,
        )
        return ctx.wrap_tensors(
            flex_gemm_hop(
                gemm_op,
                ctx.functionalize(body_fn),
                unwrapped_args,
                ctx.unwrap_tensors(kwargs),
                ctx.unwrap_tensors(kernel_options),
            )
        )


@flex_gemm_hop.py_impl(ProxyTorchDispatchMode)
def flex_gemm_proxy_torch_dispatch_mode(
    proxy_mode, gemm_op, body_fn, args, kwargs, kernel_options
):
    if proxy_mode.enable_tracing:
        flat_args = tuple(args)

        def tracing_body_fn(*flat_body_args):
            return body_fn(*flat_body_args)

        body_graph = reenter_make_fx(
            tracing_body_fn,
            subgraph_decomp_table=flex_gemm_body_decomposition_table(
                kernel_options, proxy_mode.decomposition_table
            ),
        )(*flat_args)
        apply_flex_gemm_body_graph_passes(body_graph, gemm_op)
        _, body_graph_name = unique_graph_id(proxy_mode, prefix="flex_gemm_body_graph")
        proxy_mode.tracer.root.register_module(body_graph_name, body_graph)
        proxy_args = pytree.tree_map(
            proxy_mode.tracer.unwrap_proxy,
            (gemm_op, body_graph, flat_args, kwargs, kernel_options),
        )
        out_proxy = proxy_mode.tracer.create_proxy(
            "call_function",
            flex_gemm_hop,
            proxy_args,
            {},
            name="flex_gemm",
        )
        return track_tensor_tree(
            body_fn(*flat_args),
            out_proxy,
            constant=None,
            tracer=proxy_mode.tracer,
        )
    return flex_gemm_hop(gemm_op, body_fn, args, kwargs, kernel_options)
<|endoftext|>import logging
from collections.abc import Callable
from typing import Any

import torch
from torch._higher_order_ops.utils import create_bw_fn, materialize_as_graph


logger: logging.Logger = logging.getLogger(__name__)


def _find_hop_subgraph_outputs(gm: torch.fx.GraphModule) -> tuple[torch.fx.Node]:
    output_node_args = gm.graph.find_nodes(op="output")[0].args
    if not isinstance(output_node_args, tuple):
        raise AssertionError(
            f"expected output_node_args to be tuple, got {type(output_node_args)}"
        )
    return output_node_args[0]


def is_complex_expr(expr: Any) -> bool:
    return not expr.is_symbol and not expr.is_constant()


class HopPartitionedGraph:
    def __init__(
        self,
        fw_gm: torch.fx.GraphModule,
        bw_gm: torch.fx.GraphModule,
        n_fw_outputs: int,
        n_intermediates: int,
        no_complex_exprs_at_boundary: bool,
    ):
        self.fw_gm = fw_gm
        self.bw_gm = bw_gm
        self.n_fw_outputs = n_fw_outputs
        self.n_intermediates = n_intermediates
        self.no_complex_exprs_at_boundary = no_complex_exprs_at_boundary
        self._reorder_fw_output()
        self._check_partition_boundary()

    def _check_partition_boundary(self) -> None:
        """check partitioned graph is in valid state."""
        invalid_reasons = []
        fw_outputs = _find_hop_subgraph_outputs(self.fw_gm)
        for i, out in enumerate(fw_outputs):
            if "val" not in out.meta:
                invalid_reasons.append(f"fw_gm output[{i}] doesn't have a 'val' meta.")
            elif not isinstance(out.meta["val"], (torch.SymInt, torch.Tensor)):
                invalid_reasons.append(
                    f"fw_gm output[{i}] is of type {type(out.meta['val'])} but only SymInt or Tensor are allowed."
                )

            elif (
                isinstance(out.meta["val"], torch.SymInt)
                and is_complex_expr(out.meta["val"].node.expr)
                and self.no_complex_exprs_at_boundary
            ):
                invalid_reasons.append(
                    f"fw_gm output[{i}] must be of type SymInt with basic symbols or "
                    f"Tensor but got {type(out.meta['val'])} {out.meta['val']}"
                )

        if len(fw_outputs) != self.n_fw_outputs + self.n_intermediates:
            invalid_reasons.append(
                f"len(fw_outputs) ({len(fw_outputs)}) != n_fw_outputs ({self.n_fw_outputs}) + n_intermediates ({self.n_intermediates})"
            )

        bw_phs = list(self.bw_gm.graph.find_nodes(op="placeholder"))

        if len(fw_outputs) != len(bw_phs):
            invalid_reasons.append(
                f"Expect number of fw_gm's output to be the same as bw_gm's input but "
                f"fw_gm has {len(fw_outputs)} outputs, bw_gm takes {len(bw_phs)} inputs."
            )

        original_forward_outputs = fw_outputs[: self.n_fw_outputs]
        fw_intermediates = fw_outputs[self.n_fw_outputs :]

        bw_intermediates = bw_phs[: -self.n_fw_outputs]
        bw_grads = bw_phs[-self.n_fw_outputs :]

        def _match_size_or_expr(
            val1: torch.SymInt | torch.Tensor,
            val2: torch.SymInt | torch.Tensor,
        ) -> bool:
            if type(val1) is not type(val2):
                return False

            if isinstance(val1, torch.SymInt) and isinstance(val2, torch.SymInt):
                return val1.node.expr == val2.node.expr
            elif isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
                return val1.size() == val2.size()

            return False

        for fw, bw in zip(fw_intermediates, bw_intermediates):
            if fw.name != bw.name or not _match_size_or_expr(
                fw.meta["val"], bw.meta["val"]
            ):
                invalid_reasons.append("fw intermediates don't match bw intermediates")

        for fw_out, bw_grad in zip(original_forward_outputs, bw_grads):
            if not _match_size_or_expr(fw_out.meta["val"], bw_grad.meta["val"]):
                invalid_reasons.append("fw outputs don't match bw gradients")

        if len(invalid_reasons) > 0:
            newline = "\n"
            raise RuntimeError(
                f"Invalid HopPartitionedGraph. Reasons:\n{newline.join(invalid_reasons)}"
            )

    def _reorder_fw_output(self) -> None:
        """
        Before the pass, fw_gm returns (*fw_outputs, *intermediates1)
        and bw_gm takes (*intermediates2, *grad_fw_outputs) as input.
        intermediates1 and intermediates2 share the same node names but
        they might be in different order. E.g. this could happen if there
        are inputs that contain symints.

        To simplify downstream processing, this graph pass normalizes the output of fw_gm
        to be consistent with the backward inputs:

        fw_gm:
          - input: fw_args
          - output: (*fw_outputs, *intermediates)

        bw_gm:
          - input: (*intermediates, *grad_fw_outputs)
          - output: grad_fw_args

        Example:

        def fw_gm(x, y, z):
           a, b, c = f(x), g(y), k(z)
           return a, b, c, f_tmp, g_tmp, k_tmp

        , where a, b, c are fw_outputs, f_tmp, g_tmp, k_tmp are intermediates

        The corresponding bw_gm has the following signature:

        def bw_gm(f_tmp, g_tmp, k_tmp, grad_a, grad_b, grac):
          return grad_x, grad_y, grad_z
        """
        fw_gm_output_nodes = _find_hop_subgraph_outputs(self.fw_gm)
        fw_outputs_nodes = fw_gm_output_nodes[: self.n_fw_outputs]
        fw_intermediates_nodes = fw_gm_output_nodes[self.n_fw_outputs :]
        if len(fw_intermediates_nodes) > 0:
            fw_intermediates_name_to_node = {n.name: n for n in fw_intermediates_nodes}

            # First n_intermediates placeholders
            bw_names: list[str] = [
                ph.name
                for ph in list(self.bw_gm.graph.find_nodes(op="placeholder"))[
                    : self.n_intermediates
                ]
            ]
            new_fw_outputs = list(fw_outputs_nodes) + [
                fw_intermediates_name_to_node[name] for name in bw_names
            ]

            output_node = self.fw_gm.graph.find_nodes(op="output")[0]
            output_node.args = (tuple(new_fw_outputs),)

            self.fw_gm.graph.lint()
            self.fw_gm.recompile()


class HopJointGraph:
    def __init__(
        self,
        joint_gm: torch.fx.GraphModule,
        n_primals: int,
        n_fw_outputs: int,
        *,
        functionalized: bool,
    ):
        self.joint_gm = joint_gm
        self.n_primals = n_primals
        self.n_fw_outputs = n_fw_outputs
        self.functionalized = functionalized

        self._rename_phs()
        self._remove_redundant_sym_size_ops()

    def _rename_phs(self) -> None:
        """
        Rename the placeholders for joint_gm so that the partitioner
        could recognize which inputs are primals and which are tangents.
        """
        self.n_tangents = 0
        for i, ph in enumerate(self.joint_gm.graph.find_nodes(op="placeholder")):
            if i < self.n_primals:
                ph.target = f"primals_{i}"
                ph.name = f"primals_{i}"
            else:
                self.n_tangents += 1
                ph.target = f"tangents_{i - self.n_primals}"
                ph.name = f"tangents_{i - self.n_primals}"

        self.joint_gm.graph.lint()
        self.joint_gm.compile()

    def _remove_redundant_sym_size_ops(self) -> None:
        """
        Deletes torch.ops.sym_size.int operators whose output is a
        corresponding placeholder that holds the same symbol, and replace all usage
        of the sym_size node to be directly using the placeholders.

        This is to make sure all basic symbols come from inputs.
        """
        placeholder_exprs = {}
        for node in self.joint_gm.graph.nodes:
            if (
                isinstance(node, torch.fx.Node)
                and node.op == "placeholder"
                and hasattr(node, "meta")
                and "val" in node.meta
            ):
                val = node.meta["val"]
                if isinstance(val, torch.SymInt):
                    placeholder_exprs[val.node.expr] = node

        nodes_to_remove = []
        for node in self.joint_gm.graph.find_nodes(
            op="call_function", target=torch.ops.aten.sym_size.int
        ):
            if not (hasattr(node, "meta") and "val" in node.meta):
                raise AssertionError(
                    f"node {node} must have 'meta' attribute with 'val' key"
                )
            val = node.meta["val"]
            expr = val.node.expr
            if expr in placeholder_exprs:
                placeholder_node = placeholder_exprs[expr]
                node.replace_all_uses_with(placeholder_node)
                nodes_to_remove.append(node)

        for node in nodes_to_remove:
            self.joint_gm.graph.erase_node(node)

        self.joint_gm.graph.lint()
        self.joint_gm.recompile()

    def _mark_complex_exprs_as_must_recompute(self) -> None:
        """
        For control flow operators such as scan, we don't want to
        have symint in the partitioning boundaries because otherwise we would need to support stacking
        the symints up, which causes more entropy in the stack.

        By marking the recompute polify for complex nodes as MUST_RECOMPUTE, the partitioning boundary
        no longer contains complex expressions.

        Note that this pass doesn't exclude basic symbols from partitioning boundary
        and it's up to the downstream to decide whether to return the basic symbol
        or have a separate graph pass to remove them.
        """

        from torch._functorch.partitioners import CheckpointPolicy

        for n in (
            node for node in self.joint_gm.graph.nodes if node.op == "call_function"
        ):
            if "val" not in n.meta:
                continue
            val = n.meta["val"]
            if isinstance(val, torch.SymInt) and is_complex_expr(val.node.expr):
                if n.meta.get("recompute", None) is not None:
                    raise AssertionError(
                        f"node {n} with complex SymInt expression should not have recompute policy set"
                    )

                n.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE

        self.joint_gm.graph.lint()
        self.joint_gm.recompile()

    def partition(
        self, partition_fn: Callable, always_recompute_complex_exprs: bool
    ) -> HopPartitionedGraph:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "before min_cut_partition:\n%s",
                self.joint_gm.print_readable(print_output=False),
            )

        if always_recompute_complex_exprs:
            self._mark_complex_exprs_as_must_recompute()

        fw_gm, bw_gm = partition_fn(
            self.joint_gm, None, num_fwd_outputs=self.n_fw_outputs
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("after partition_fn:")
            logger.debug("fw_gm:\n%s", fw_gm.print_readable(print_output=False))
            logger.debug("bw_gm:\n%s", bw_gm.print_readable(print_output=False))

        n_intermediates = len(_find_hop_subgraph_outputs(fw_gm)) - self.n_fw_outputs

        return HopPartitionedGraph(
            fw_gm,
            bw_gm,
            self.n_fw_outputs,
            n_intermediates,
            always_recompute_complex_exprs,
        )


def create_hop_joint_graph(
    fw_fn: Callable,
    fw_args: tuple[torch.Tensor | torch.SymInt, ...],
    functionalize: bool,
) -> HopJointGraph:
    fw_gm = materialize_as_graph(fw_fn, fw_args, force_enable_grad=True)
    fw_gm_output_nodes = _find_hop_subgraph_outputs(fw_gm)

    if not all(
        isinstance(n, torch.fx.Node) and "val" in n.meta for n in fw_gm_output_nodes
    ):
        raise AssertionError(
            "all fw_gm output nodes must be torch.fx.Node with 'val' in meta"
        )
    fw_gm_output_vals = tuple(n.meta["val"] for n in fw_gm_output_nodes)  # type: ignore[arg-type]

    if not all(isinstance(val, torch.Tensor) for val in fw_gm_output_vals):
        raise AssertionError(
            f"all fw_gm output values must be torch.Tensor, got {[type(v) for v in fw_gm_output_vals]}"
        )
    example_grads = tuple(torch.zeros_like(val) for val in fw_gm_output_vals)

    joint_fn = create_bw_fn(fw_fn, fw_args, return_fw_outputs=True)
    joint_gm = materialize_as_graph(
        joint_fn, fw_args + example_grads, force_enable_grad=True
    )
    if functionalize:
        # Need to first trace out the joint_fn with autograd info on
        # then functionalize the graph otherwise the grad information is lost
        joint_gm = materialize_as_graph(
            # pyrefly: ignore [bad-argument-type]
            torch.func.functionalize(joint_gm, remove="mutations_and_views"),
            fw_args + example_grads,
        )

    return HopJointGraph(
        joint_gm,
        len(fw_args),
        len(fw_gm_output_nodes),
        functionalized=functionalize,
    )


class HopGraphMinCutPartitioner:
    @staticmethod
    def create_partitioned_graph(
        fw_fn: Callable,
        fw_args: tuple[torch.Tensor | torch.SymInt, ...],
        *,
        always_recompute_complex_exprs: bool = False,
    ) -> HopPartitionedGraph:
        """
        Inputs:
            - fw_fn: the forward function that we'll use to create a joint graph and partition
            - fw_args: the flat_args to fw_fn
            - always_recompute_complex_exprs: when set to True, the bw_gm will do a re-compute
              for inputs that are complex expressions such that the partitioning boundary
              only consists of basic symbols and tensors.

        Returns a HopPartitionedGraph
        """
        from torch._functorch.partitioners import min_cut_rematerialization_partition

        joint_graph: HopJointGraph = create_hop_joint_graph(
            fw_fn, fw_args, functionalize=True
        )
        return joint_graph.partition(
            min_cut_rematerialization_partition, always_recompute_complex_exprs
        )
<|endoftext|>from torch._higher_order_ops._invoke_quant import (
    invoke_quant,
    invoke_quant_packed,
    InvokeQuant,
)
from torch._higher_order_ops.aoti_call_delegate import aoti_call_delegate
from torch._higher_order_ops.associative_scan import associative_scan
from torch._higher_order_ops.auto_functionalize import (
    auto_functionalized,
    auto_functionalized_v2,
)
from torch._higher_order_ops.base_hop import BaseHOP
from torch._higher_order_ops.cond import cond
from torch._higher_order_ops.effects import with_effects
from torch._higher_order_ops.executorch_call_delegate import executorch_call_delegate
from torch._higher_order_ops.flat_apply import flat_apply
from torch._higher_order_ops.flex_attention import (
    flex_attention,
    flex_attention_backward,
)
from torch._higher_order_ops.flex_gemm import flex_gemm
from torch._higher_order_ops.foreach_map import _foreach_map, foreach_map
from torch._higher_order_ops.hints_wrap import hints_wrapper
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch._higher_order_ops.invoke_leaf_function import invoke_leaf_function
from torch._higher_order_ops.invoke_subgraph import invoke_subgraph
from torch._higher_order_ops.local_map import local_map_hop
from torch._higher_order_ops.map import map
from torch._higher_order_ops.out_dtype import out_dtype
from torch._higher_order_ops.print import print
from torch._higher_order_ops.run_const_graph import run_const_graph
from torch._higher_order_ops.scan import scan
from torch._higher_order_ops.strict_mode import strict_mode
from torch._higher_order_ops.switch import switch
from torch._higher_order_ops.torchbind import call_torchbind
from torch._higher_order_ops.while_loop import (
    while_loop,
    while_loop_stack_output_op as while_loop_stack_output,
)
from torch._higher_order_ops.wrap import (
    dynamo_bypassing_wrapper,
    inductor_compiled_code,
    tag_activation_checkpoint,
    wrap_activation_checkpoint,
    wrap_with_autocast,
    wrap_with_set_grad_enabled,
)


__all__ = [
    "cond",
    "switch",
    "while_loop",
    "invoke_subgraph",
    "scan",
    "map",
    "flex_attention",
    "flex_attention_backward",
    "hints_wrapper",
    "BaseHOP",
    "flat_apply",
    "foreach_map",
    "_foreach_map",
    "flex_gemm",
    "with_effects",
    "tag_activation_checkpoint",
    "auto_functionalized",
    "auto_functionalized_v2",
    "associative_scan",
    "out_dtype",
    "executorch_call_delegate",
    "call_torchbind",
    "run_const_graph",
    "InvokeQuant",
    "invoke_leaf_function",
    "invoke_quant",
    "invoke_quant_packed",
    "wrap_with_set_grad_enabled",
    "wrap_with_autocast",
    "wrap_activation_checkpoint",
    "dynamo_bypassing_wrapper",
    "strict_mode",
    "aoti_call_delegate",
    "map",
    "while_loop_stack_output",
    "local_map_hop",
    "print",
    "inductor_compiled_code",
    "inline_asm_elementwise",
    "register_hook",
]
<|endoftext|>import contextlib
import math
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch._C import DispatchKey
from torch._higher_order_ops.utils import (
    _has_potential_branch_input_mutation,
    _maybe_reenter_make_fx,
    autograd_not_implemented,
    has_user_subclass,
    redirect_to_mode,
    reenter_make_fx,
    register_fake,
    save_values_for_backward,
    saved_values,
    UnsupportedAliasMutationException,
    validate_subgraph_args_types,
)
from torch._ops import HigherOrderOperator
from torch._subclasses import FakeTensor
from torch.amp.autocast_mode import _cast as _autocast_cast
from torch.fx.experimental.proxy_tensor import (
    make_fx,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.fx.graph_module import GraphModule
from torch.fx.traceback import annotate, current_meta
from torch.utils.checkpoint import _CachedTorchDispatchMode, _CachingTorchDispatchMode


# Duplicate of _inductor/kernel/flex_attention.py to avoid circular import
def _construct_strides(
    sizes: Sequence[int],
    fill_order: Sequence[int],
) -> Sequence[int]:
    """From a list of sizes and a fill order, construct the strides of the permuted tensor."""
    # Initialize strides
    if len(sizes) != len(fill_order):
        raise AssertionError(
            f"Length of sizes must match the length of the fill order, got {len(sizes)} vs {len(fill_order)}"
        )
    strides = [0] * len(sizes)

    # Start with stride 1 for the innermost dimension
    current_stride = 1

    # Iterate through the fill order populating strides
    for dim in fill_order:
        strides[dim] = current_stride
        current_stride *= sizes[dim]

    return strides


def _permute_strides(out: torch.Tensor, query_strides: tuple[int, ...]) -> torch.Tensor:
    """
    Create a new tensor with the same data and shape as the input,
    but with strides permuted based on the input tensor's stride order.

    Args:
        out (torch.Tensor): The output tensor of attention.
        query_strides (List[int]): The stride order of the input query tensor

    Returns:
        torch.Tensor: A new tensor with same shape and data as the input,
        but with strides permuted based on the query tensor's stride order.
    """
    from torch._inductor.ir import get_fill_order

    fill_order = get_fill_order(query_strides)
    if out.storage_offset() != 0:
        raise AssertionError(
            f"Only support storage_offset == 0, got {out.storage_offset()}"
        )
    out_strides = _construct_strides(out.shape, fill_order)

    # Attention kernels require stride[-1]=1 for efficient memory access.
    # Ensure this by moving last dim to front of fill_order if needed.
    if out_strides[-1] != 1:
        last_dim = len(out.shape) - 1
        fill_order = list(fill_order)
        fill_order.remove(last_dim)
        fill_order = [last_dim] + fill_order
        out_strides = _construct_strides(out.shape, fill_order)

    new_out = out.new_empty_strided(out.shape, out_strides)
    new_out.copy_(out)
    return new_out


class FlexAttentionHOP(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("flex_attention", cacheable=True)

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        score_mod: Callable,
        block_mask: tuple,
        scale: float,
        kernel_options: dict[str, Any],
        score_mod_other_buffers: tuple = (),
        mask_mod_other_buffers: tuple = (),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        validate_subgraph_args_types(score_mod_other_buffers + mask_mod_other_buffers)
        # pyrefly: ignore [missing-attribute]
        return super().__call__(
            query,
            key,
            value,
            score_mod,
            block_mask,
            scale,
            kernel_options,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        )


flex_attention = FlexAttentionHOP()


class FlexAttentionBackwardHOP(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("flex_attention_backward", cacheable=True)

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        out: torch.Tensor,
        logsumexp: torch.Tensor,
        grad_out: torch.Tensor | None,
        grad_logsumexp: torch.Tensor | None,
        fw_graph: Callable | GraphModule,
        joint_graph: GraphModule,
        block_mask: tuple,
        scale: float,
        kernel_options: dict[str, Any],
        score_mod_other_buffers: tuple = (),
        mask_mod_other_buffers: tuple = (),
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...]
    ]:
        validate_subgraph_args_types(score_mod_other_buffers + mask_mod_other_buffers)

        # pyrefly: ignore [missing-attribute]
        return super().__call__(
            query,
            key,
            value,
            out,
            logsumexp,
            grad_out,
            grad_logsumexp,
            fw_graph,
            joint_graph,
            block_mask,
            scale,
            kernel_options,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        )


flex_attention_backward = FlexAttentionBackwardHOP()


def _math_attention_inner(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[torch.Tensor, torch.Tensor]:
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex

    working_precision = torch.float64 if query.dtype == torch.float64 else torch.float32

    scores = query.to(working_precision) @ key.to(working_precision).transpose(-2, -1)

    b = torch.arange(0, scores.size(0), device=scores.device)
    h = torch.arange(0, scores.size(1), device=scores.device)
    m = torch.arange(0, scores.size(2), device=scores.device)
    n = torch.arange(0, scores.size(3), device=scores.device)

    captured_buffers_in_dim = (None,) * len(score_mod_other_buffers)
    from torch.nn.attention.flex_attention import _vmap_for_bhqkv

    # first input is score
    score_mod = _vmap_for_bhqkv(score_mod, prefix=(0,), suffix=captured_buffers_in_dim)

    mask_mod = block_mask[-1]
    mask_mod_in_dim_buffers = (None,) * len(mask_mod_other_buffers)
    mask_mod = _vmap_for_bhqkv(mask_mod, prefix=(), suffix=mask_mod_in_dim_buffers)

    with TransformGetItemToIndex():
        scores = (scores * scale).to(working_precision)
        post_mod_scores = torch.where(
            mask_mod(b, h, m, n, *mask_mod_other_buffers),
            score_mod(scores, b, h, m, n, *score_mod_other_buffers),
            torch.tensor(-float("inf"), dtype=working_precision, device=scores.device),
        )

    return scores, post_mod_scores


def math_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Eager implementation

    This implementation uses vmap to vectorize the score_mod function over the batch, head, m, and n dimensions.
    We then apply the vectorized score_mod function to the scores matrix. Each wrap of vmap applies one of the
    batch, head, m, or n dimensions. We need to apply vmap 4 times to vectorized over all 4 dimensions.

    Args:
        query: The query tensor
        key: The key tensor
        value: The value tensor
        score_mod: The score_mod function
        other_buffers: Other buffers that are passed to the score_mod function

    Notes:
        Query and Keys are dtype cast up to float64 (if query.dtype is float64) and float32 otherwise.
        Scores and Values are dtype cast to input query.dtype at the end.
    """
    # broadcast query & key along head dim for GQA
    G = query.size(1) // key.size(1)
    value = torch.repeat_interleave(value, G, dim=1)
    key = torch.repeat_interleave(key, G, dim=1)

    Bq, Bkv = query.size(0), key.size(0)
    if not ((Bq == Bkv) or (Bq > 1 and Bkv == 1)):
        raise RuntimeError(f"Bq and Bkv must broadcast. Got Bq={Bq} and Bkv={Bkv}")

    key = key.expand((Bq, *key.size()[1:]))
    value = value.expand((Bq, *value.size()[1:]))

    _, post_mod_scores = _math_attention_inner(
        query,
        key,
        value,
        score_mod,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    )

    # Set fully masked rows' sumexp to 0.0
    logsumexp = post_mod_scores.logsumexp(dim=-1)
    masked_rows = torch.all(post_mod_scores == -float("inf"), dim=-1)
    logsumexp = torch.where(masked_rows, -float("inf"), logsumexp)

    # working precision will be used so no need to cast to fp32
    max_scores = torch.max(post_mod_scores, dim=-1)[0]

    post_mod_scores = torch._safe_softmax(post_mod_scores, dim=-1)

    # NB: kernel computes in ln2 space, we always convert back at the top level op, so
    # for math impl we divide by log(2) because we will multiply by log(2)

    return (
        post_mod_scores.to(query.dtype) @ value.to(query.dtype),
        logsumexp / math.log(2),
        max_scores / math.log(2),
    )


def _flex_attention_autocast_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple,
    mask_mod_other_buffers: tuple,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward-only autocast shim: cast Q/K/V to the active autocast dtype, then
    redispatch with Autocast keys excluded so we hit the normal implementation.
    """
    device_type = query.device.type
    autocast_dtype = torch.get_autocast_dtype(device_type)

    query = _autocast_cast(query, device_type, autocast_dtype)
    key = _autocast_cast(key, device_type, autocast_dtype)
    value = _autocast_cast(value, device_type, autocast_dtype)

    autocast_keyset = torch._C.DispatchKeySet(
        DispatchKey.AutocastCPU
    ) | torch._C.DispatchKeySet(DispatchKey.AutocastCUDA)
    with torch._C._ExcludeDispatchKeyGuard(autocast_keyset):
        return flex_attention(
            query,
            key,
            value,
            score_mod,
            block_mask,
            scale,
            kernel_options,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        )


@flex_attention.py_impl(DispatchKey.AutocastCUDA)
def flex_attention_autocast_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _flex_attention_autocast_impl(
        query,
        key,
        value,
        score_mod,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    )


@flex_attention.py_impl(DispatchKey.AutocastCPU)
def flex_attention_autocast_cpu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _flex_attention_autocast_impl(
        query,
        key,
        value,
        score_mod,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    )


@flex_attention.py_impl(DispatchKey.CompositeExplicitAutograd)
def sdpa_dense(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out, lse, max_scores = math_attention(
        query,
        key,
        value,
        score_mod,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    )
    out = _permute_strides(out, query.stride())
    return out, lse, max_scores


def trace_flex_attention(
    proxy_mode: ProxyTorchDispatchMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Traces the flex_attention operator with the given score_mod function and other_buffers.

    Trace SDPA will call make_fx with "fake" example vals and then trace the score_mod function
    This will produce a GraphModule that will be stored on the root tracer as "sdpa_score". We
    access this graph module in inductor to inline the score_mod function to the triton template.
    """
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex

    example_out = flex_attention(
        query,
        key,
        value,
        score_mod,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    )
    example_vals = [query.new_zeros((), requires_grad=query.requires_grad)] + [
        query.new_zeros((), dtype=torch.int) for _ in range(4)
    ]
    mask_example_vals = [query.new_zeros((), dtype=torch.int) for _ in range(4)]
    mask_mod = block_mask[-1]
    with TransformGetItemToIndex():
        score_graph = reenter_make_fx(score_mod)(
            *example_vals, *score_mod_other_buffers
        )
        mask_graph = reenter_make_fx(mask_mod)(
            *mask_example_vals, *mask_mod_other_buffers
        )
    if not isinstance(proxy_mode.tracer, torch.fx.Tracer):
        raise AssertionError(
            f"expected proxy_mode.tracer to be torch.fx.Tracer, got {type(proxy_mode.tracer)}"
        )
    block_mask = block_mask[:-1] + (mask_graph,)
    qualname = proxy_mode.tracer.get_fresh_qualname("sdpa_score")
    proxy_mode.tracer.root.register_module(qualname, score_graph)
    mask_qualname = proxy_mode.tracer.get_fresh_qualname("sdpa_mask")
    proxy_mode.tracer.root.register_module(mask_qualname, mask_graph)
    node_args = (
        query,
        key,
        value,
        score_graph,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    )
    # pyrefly: ignore [missing-attribute]
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    with torch.fx.experimental.proxy_tensor.set_original_aten_op(flex_attention):
        out_proxy = proxy_mode.tracer.create_proxy(
            "call_function", flex_attention, proxy_args, {}
        )
    return track_tensor_tree(
        example_out,
        out_proxy,
        constant=None,
        tracer=proxy_mode.tracer,
    )


@flex_attention.py_impl(ProxyTorchDispatchMode)
def flex_attention_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mode is None:
        raise AssertionError("Mode should always be enabled for python fallback key")
    return trace_flex_attention(
        mode,
        query,
        key,
        value,
        score_mod,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    )


@flex_attention.py_functionalize_impl
def flex_attention_functionalize(
    ctx: torch._subclasses.functional_tensor.BaseFunctionalizeAPI,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Defines the functionalization rules for the flex_attention operator.

    Right now we are unwrapping each tensor and then redispatching to the next, however we want to
    guard against any mutations in the score_mod function, to the other_buffers since those
    are free variables.
    """
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex

    query_unwrapped = ctx.unwrap_tensors(query)
    key_unwrapped = ctx.unwrap_tensors(key)
    value_unwrapped = ctx.unwrap_tensors(value)
    block_mask_unwrapped = ctx.unwrap_tensors(block_mask)
    score_mod_other_buffers_unwrapped = ctx.unwrap_tensors(score_mod_other_buffers)
    mask_mod_other_buffers_unwrapped = ctx.unwrap_tensors(mask_mod_other_buffers)

    # Appease the mypy overlords
    if not isinstance(query_unwrapped, torch.Tensor):
        raise AssertionError(
            f"expected query_unwrapped to be torch.Tensor, got {type(query_unwrapped)}"
        )
    if not isinstance(key_unwrapped, torch.Tensor):
        raise AssertionError(
            f"expected key_unwrapped to be torch.Tensor, got {type(key_unwrapped)}"
        )
    if not isinstance(value_unwrapped, torch.Tensor):
        raise AssertionError(
            f"expected value_unwrapped to be torch.Tensor, got {type(value_unwrapped)}"
        )
    if not isinstance(block_mask_unwrapped, tuple):
        raise AssertionError(
            f"expected block_mask_unwrapped to be tuple, got {type(block_mask_unwrapped)}"
        )
    if not isinstance(score_mod_other_buffers_unwrapped, tuple):
        raise AssertionError(
            f"expected score_mod_other_buffers_unwrapped to be tuple, got {type(score_mod_other_buffers_unwrapped)}"
        )
    if not isinstance(mask_mod_other_buffers_unwrapped, tuple):
        raise AssertionError(
            f"expected mask_mod_other_buffers_unwrapped to be tuple, got {type(mask_mod_other_buffers_unwrapped)}"
        )

    example_vals = (
        [query_unwrapped.new_zeros(())]
        + [query_unwrapped.new_zeros((), dtype=torch.int) for _ in range(4)]
        + list(score_mod_other_buffers_unwrapped)
    )
    with ctx.redispatch_to_next():
        functional_score_mod = ctx.functionalize(score_mod)
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        with TransformGetItemToIndex():
            # TODO: So far only the input mutations are checked
            # In the other HOPs, also aliases are checked which is
            # omitted here
            mutates = _has_potential_branch_input_mutation(
                score_mod, example_vals, pre_dispatch
            )
        # We only care about mutations of existing buffers since we can't replay these.
        # However, we can just error if anything is detected
        if mutates:
            raise UnsupportedAliasMutationException("Mutations detected in score_mod")

        out = flex_attention(
            query_unwrapped,
            key_unwrapped,
            value_unwrapped,
            functional_score_mod,
            block_mask_unwrapped,
            scale,
            kernel_options,
            score_mod_other_buffers_unwrapped,
            mask_mod_other_buffers_unwrapped,
        )
    return ctx.wrap_tensors(out)  # type: ignore[return-value, arg-type]


@register_fake(flex_attention)
def flex_attention_fake_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if has_user_subclass(
        (
            query,
            key,
            value,
            score_mod,
            block_mask,
            scale,
            kernel_options,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        ),
        allowed_subclasses=(FakeTensor,),
    ):
        return NotImplemented

    v_head_dim = value.size(-1)
    batch_size, num_heads, seq_len_q, _q_head_dim = query.shape
    logsumexp = query.new_empty(batch_size, num_heads, seq_len_q, dtype=torch.float32)
    max_scores = query.new_empty(batch_size, num_heads, seq_len_q, dtype=torch.float32)
    out_shape = (batch_size, num_heads, seq_len_q, v_head_dim)
    out = query.new_empty(out_shape)
    out = _permute_strides(out, query.stride())
    return out, logsumexp, max_scores


# Registers dispatches for SAC
redirect_to_mode(flex_attention, _CachingTorchDispatchMode)
redirect_to_mode(flex_attention, _CachedTorchDispatchMode)


# ---------------------------- Autograd Implementation ----------------------------
def create_fw_bw_graph(
    score_mod: Callable,
    index_values: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    other_buffers: tuple[Tensor, ...],
) -> tuple[Callable, Callable]:
    # See Note:[HOP create fw_bw graph]

    # All of these imports need to be here in order to avoid circular dependencies
    from torch._dispatch.python import suspend_functionalization
    from torch._dynamo._trace_wrapped_higher_order_op import mod_index
    from torch._functorch.aot_autograd import AOTConfig, create_joint
    from torch._subclasses.fake_tensor import FakeTensorMode, is_fake_tensor
    from torch._subclasses.functional_tensor import disable_functional_mode
    from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing

    dummy_aot_config = AOTConfig(
        fw_compiler=None,  # type: ignore[arg-type]
        bw_compiler=None,  # type: ignore[arg-type]
        partition_fn=None,  # type: ignore[arg-type]
        decompositions={},
        num_params_buffers=0,
        aot_id=0,
        keep_inference_input_mutations=False,
    )

    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():

            def _from_fun(
                t: Tensor | torch.SymInt | int,
            ) -> Tensor | torch.SymInt | int:
                if isinstance(t, torch.Tensor):
                    return torch.empty_strided(
                        t.size(),
                        t.stride(),
                        device=t.device,
                        dtype=t.dtype,
                        requires_grad=t.requires_grad,
                    )
                return t

            # If someone runs this hop under the default compiler backend ("eager")
            # Then this path will be run with the actual user inputs. We convert them
            # to fake tensors in order to not perform any actual compute.
            from torch._guards import detect_fake_mode

            fake_mode = detect_fake_mode(index_values)
            if fake_mode is None:
                fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

            with fake_mode:
                unwrapped_score_mod_indexes = pytree.tree_map(_from_fun, index_values)
                unwrapped_other_buffers = pytree.tree_map(_from_fun, other_buffers)

            if not all(
                is_fake_tensor(t) or isinstance(t, (int, torch.SymInt))
                for t in unwrapped_score_mod_indexes + unwrapped_other_buffers
            ):
                raise AssertionError(
                    f"all unwrapped values must be FakeTensor, int, or SymInt, got "
                    f"{[type(t) for t in unwrapped_score_mod_indexes + unwrapped_other_buffers]}"
                )

            example_flat_out = pytree.tree_map(
                _from_fun,
                score_mod(*unwrapped_score_mod_indexes, *unwrapped_other_buffers),
            )
            if not isinstance(example_flat_out, torch.Tensor):
                raise RuntimeError(
                    "Expected output of score_mod to be a tensor."
                    f"Got type {type(example_flat_out)}."
                )
            example_grad = _from_fun(example_flat_out)

        def joint_f(
            score: Tensor,
            b: Tensor,
            h: Tensor,
            m: Tensor,
            n: Tensor,
            example_grad: Tensor,
            *other_buffers: tuple[Tensor, ...],
        ) -> list[Tensor | None]:
            def fw_with_masks(
                *args: tuple[Tensor, ...],
            ) -> tuple[tuple[Tensor], tuple[bool]]:
                captured_args = [
                    mod_index(buffer, [])
                    if (
                        isinstance(buffer, torch.Tensor)
                        and buffer.requires_grad
                        and buffer.ndim == 0
                    )
                    else buffer
                    for buffer in args[5:]
                ]
                fw_out = score_mod(*args[:5], *captured_args)
                out_requires_grad = fw_out.requires_grad
                return ((fw_out,), (out_requires_grad,))

            joint = create_joint(fw_with_masks, aot_config=dummy_aot_config)
            args = [score, b, h, m, n] + list(other_buffers)
            optional_grad = [example_grad] if example_grad.requires_grad else []
            _, grads = joint(args, optional_grad)
            if grads[0] is None:
                raise RuntimeError(
                    "flex_attention backward requires the output of score_mod to "
                    "depend on score. Got a score_mod whose output does not "
                    "require gradients with respect to score."
                )

            return grads

        joint_graph = make_fx(joint_f)(
            *unwrapped_score_mod_indexes, example_grad, *unwrapped_other_buffers
        )
        # pyrefly: ignore [bad-return]
        return score_mod, joint_graph


class FlexAttentionAutogradOp(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx: Any,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        fw_graph: Callable,
        joint_graph: Callable,
        block_mask: tuple[Any, ...],
        scale: float,
        kernel_options: dict[str, Any],
        mask_mod_other_buffers: tuple[Any, ...],
        *score_mod_other_buffers: tuple[Any, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.set_materialize_grads(False)
        # Capture sparsity_hint from tracing metadata so backward can re-annotate
        ctx._sparsity_hint = current_meta.get("custom", {}).get("sparsity_hint", 0.0)
        any_buffer_requires_grad = any(
            buffer.requires_grad
            for buffer in mask_mod_other_buffers
            if isinstance(buffer, torch.Tensor)
        )
        if any_buffer_requires_grad:
            raise AssertionError(
                "Captured buffers from mask mod that require grad are not supported."
            )
        ctx._fw_graph = fw_graph
        ctx._joint_graph = joint_graph
        ctx._mask_graph = block_mask[-1]
        ctx.scale = scale
        ctx.kernel_options = kernel_options
        ctx._score_mod_other_buffers_len = len(score_mod_other_buffers)
        with torch._C._AutoDispatchBelowAutograd():
            out, logsumexp, max_scores = flex_attention(
                query,
                key,
                value,
                fw_graph,
                block_mask,
                scale,
                kernel_options,
                score_mod_other_buffers,
                mask_mod_other_buffers,
            )
        # no grads for you sir
        ctx.mark_non_differentiable(max_scores)
        save_values_for_backward(
            ctx,
            (
                query,
                key,
