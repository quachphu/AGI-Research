        and interpolation_mode == 0
        and a.is_contiguous(memory_format=torch.contiguous_format)
    )

    output = decomp_grid_sampler_2d(
        a,
        grid=grid,
        interpolation_mode=interpolation_mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
        _expand_grid=_expand_grid,
    )
    return output

    # _foreach_addcmul and _foreach_addcdiv decompositions removed —
    # the inductor lowering in lowering.py handles these with FMA.


@register_decomposition(aten._foreach_lerp.Scalar)
def _foreach_lerp_scalar(
    start_tensors: list[torch.Tensor],
    end_tensors: list[torch.Tensor],
    weight: torch.types.Number,
) -> list[torch.Tensor]:
    def opmath_dtype(t: torch.Tensor) -> torch.dtype:
        if t.dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        return t.dtype

    def cast_weight(dtype: torch.dtype) -> float | complex:
        if isinstance(weight, complex) and not dtype.is_complex:
            if weight.imag != 0:
                return torch.tensor(weight, dtype=dtype).item()
            return torch.tensor(weight.real, dtype=dtype).item()
        return torch.tensor(weight, dtype=dtype).item()

    def use_high_formula(dtype: torch.dtype) -> bool:
        casted_weight = cast_weight(dtype)
        if isinstance(casted_weight, complex):
            return (
                casted_weight.real * casted_weight.real
                + casted_weight.imag * casted_weight.imag
                >= 0.25
            )
        return casted_weight >= 0.5 or casted_weight <= -0.5

    def scalar_tensor(
        dtype: torch.dtype, device: torch.device, high_weight: bool
    ) -> torch.Tensor:
        if isinstance(weight, complex) and not dtype.is_complex:
            if weight.imag != 0:
                weight_tensor = torch.tensor(weight, dtype=dtype, device=device)
                if high_weight:
                    return -(1.0 - weight_tensor)
                return weight_tensor
            weight_tensor = torch.tensor(weight.real, dtype=dtype, device=device)
            if high_weight:
                return -(1.0 - weight_tensor)
            return weight_tensor
        weight_tensor = torch.tensor(weight, dtype=dtype, device=device)
        if high_weight:
            return -(1.0 - weight_tensor)
        return weight_tensor

    dtype = start_tensors[0].dtype
    device = start_tensors[0].device
    compute_dtype = opmath_dtype(start_tensors[0])
    use_foreach = dtype not in (torch.float16, torch.bfloat16) and all(
        t.dtype == dtype and t.device == device for t in start_tensors
    )
    if use_foreach:
        high_weight = use_high_formula(compute_dtype)
        bases = end_tensors if high_weight else start_tensors
        diff = aten._foreach_sub.List(end_tensors, start_tensors)
        scalar = scalar_tensor(dtype, device, high_weight)
        return aten._foreach_addcmul.Scalar(bases, [scalar] * len(diff), diff)

    return [
        torch.addcmul(
            (end if high_weight else t).to(compute_dtype),
            scalar_tensor(compute_dtype, t.device, high_weight),
            end.to(compute_dtype) - t.to(compute_dtype),
        ).to(t.dtype)
        for t, end, compute_dtype, high_weight in (
            (
                t,
                end,
                opmath_dtype(t),
                use_high_formula(opmath_dtype(t)),
            )
            for t, end in zip(start_tensors, end_tensors)
        )
    ]


@register_decomposition(aten._foreach_lerp.ScalarList)
def _foreach_lerp_scalarlist(
    start_tensors: list[torch.Tensor],
    end_tensors: list[torch.Tensor],
    scalars: list[torch.types.Number],
) -> list[torch.Tensor]:
    return aten._foreach_add.List(
        start_tensors,
        aten._foreach_mul.ScalarList(
            aten._foreach_sub.List(end_tensors, start_tensors), scalars
        ),
    )


@aten.miopen_batch_norm.default.py_impl(torch._C.DispatchKey.Autograd)
@register_decomposition(aten.miopen_batch_norm)
def miopen_batch_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    running_mean: torch.Tensor | None,
    running_var: torch.Tensor | None,
    training: bool,
    exponential_average_factor: float,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a, b, c = aten.native_batch_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        exponential_average_factor,
        epsilon,
    )

    if training:
        return (a, b, c)
    return (
        a,
        weight.new_zeros((0,)),
        weight.new_zeros((0,)),
    )


@functools.cache
def fast_random_decomps() -> dict[Any, Callable[..., Any]]:
    return {**decompositions, **extra_random_decomps}


# TODO(aakhundov): replace this (and the above) Any by more
# specific type and fix all the cascading mypy errors
def select_decomp_table() -> dict[Any, Callable[..., Any]]:
    """decomps can change based on config"""
    if config.fallback_random:
        return decompositions
    if config.fallback_embedding_bag_byte_unpack:
        # remove q_embedding_bag_byte_unpack_decomp from decompositions
        decompositions.pop(torch.ops.quantized.embedding_bag_byte_unpack.default, None)
        return decompositions
    result = fast_random_decomps()
    return result


@register_decomposition(aten.masked_scatter)
def masked_scatter(
    self: torch.Tensor,
    mask: torch.Tensor,
    source: torch.Tensor,
) -> torch.Tensor:
    from .codegen.common import BackendFeature, has_backend_feature

    if has_backend_feature(self.device, BackendFeature.MASKED_SCATTER_WITH_INDEX):
        # This two-step algorithm is the same as eager CUDA, for eager CPU we
        # use a 1-shot serial iteration.
        self, mask = aten.broadcast_tensors([self, mask])
        source_idx = mask.reshape(-1).cumsum(0) - 1
        self_flat, mask_flat, source_flat = (x.flatten() for x in (self, mask, source))
        result = aten._unsafe_masked_index(source_flat, mask_flat, [source_idx], 0)
        return torch.where(mask_flat, result, self_flat).view(self.shape)
    return NotImplemented


@register_decomposition(quantized_decomposed.choose_qparams.tensor)
def choose_qparams_tensor(
    input: torch.Tensor,
    quant_min: int,
    quant_max: int,
    eps: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    min_val, max_val = torch.aminmax(input)
    scale = (max_val - min_val) / float(quant_max - quant_min)
    scale = torch.max(scale, torch.Tensor([eps]))
    zero_point = quant_min - torch.round(min_val / scale).to(torch.int)
    zero_point = torch.clamp(zero_point, quant_min, quant_max)
    return scale.to(torch.float64), zero_point.to(torch.int64)


@register_decomposition(aten.put)
def put(
    self: torch.Tensor,
    index: torch.Tensor,
    source: torch.Tensor,
    accumulate: bool = False,
) -> torch.Tensor:
    flattened = self.flatten()
    flattened = torch.index_put(
        flattened, [index], source.reshape(index.shape), accumulate
    )
    return flattened.reshape(self.shape)


@register_decomposition(aten.put_)
def put_(
    self: torch.Tensor,
    index: torch.Tensor,
    source: torch.Tensor,
    accumulate: bool = False,
) -> torch.Tensor:
    out = aten.put(self, index, source, accumulate=accumulate)
    return self.copy_(out)


@register_decomposition(aten._softmax_backward_data.default)
@pw_cast_for_opmath
def _softmax_backward_data(
    grad_output: torch.Tensor,
    output: torch.Tensor,
    dim: int,
    input_dtype: torch.dtype,
) -> torch.Tensor:
    new_grad_output = grad_output * output
    sum_new_grad = torch.sum(new_grad_output, dim=dim, keepdim=True)
    # grad_input = new_grad_output - output * sum_new_grad
    grad_input = inductor_prims.fma(-output, sum_new_grad, new_grad_output)

    # CPU kernel doesn't respect input_dtype, but following check doesn't work for meta tensor
    # if grad_output.device == torch.device("cpu"):
    #     return grad_input.contiguous()

    if grad_output.dtype != input_dtype:
        grad_input = grad_input.to(input_dtype)
    return grad_input.contiguous()


@register_decomposition(aten.index_reduce)
def index_reduce(
    self: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src: torch.Tensor,
    reduction_type: str,
    *,
    include_self: bool = True,
) -> torch.Tensor:
    torch._check(
        self.device == index.device and self.device == src.device,
        lambda: (
            f"index_reduce(): self, index and source expected to be in the same device, "
            f"but got (self) {self.device}, (index) {index.device}, "
            f"and (source) {src.device}"
        ),
    )
    if reduction_type == "mean" and not needs_fallback_due_to_atomic_add_limitations(
        self.dtype
    ):
        true_division = self.dtype.is_floating_point or self.dtype.is_complex
        ones = torch.ones_like(src)
        if include_self:
            out = self
            counts = torch.ones_like(self).index_add(dim, index, ones)
        else:
            out = self.index_fill(dim, index, 0)
            counts = torch.zeros_like(self).index_add(dim, index, ones)
            counts = counts.masked_fill(counts < 1, 1)
        out = out.index_add(dim, index, src)
        return out / counts if true_division else out // counts

    if use_scatter_fallback(
        aten.scatter_reduce_.two,
        reduction_type,
        self.dtype,
        src.dtype,
        src.device.type,
        True,
    ):
        return NotImplemented

    # pyrefly: ignore [missing-attribute]
    repeats = self.shape[dim + 1 :].numel() * self.shape[:dim].numel()
    index_shape = (index.numel(), *self.shape[dim + 1 :], *self.shape[:dim])
    perm = (*range(self.ndim - dim, self.ndim), 0, *range(1, self.ndim - dim))
    scatter_index = (
        index.to(torch.int64)
        .repeat_interleave(repeats)
        .reshape(index_shape)
        .permute(perm)
    )
    return self.scatter_reduce(
        dim,
        scatter_index,
        src,
        reduction_type,
        include_self=include_self,
    )


def _max_pool_with_indices(
    x: torch.Tensor,
    kernel_size: list[int],
    stride: int | list[int] | None,
    padding: int | list[int],
    dilation: int | list[int],
    ceil_mode: bool,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dilation == 1:
        dilation = [1] * dim

    if padding == 0:
        padding = [0] * dim

    if not stride:
        stride = kernel_size

    # pyrefly: ignore [bad-assignment]
    kernel_size = pad_listlike(kernel_size, dim)
    # pyrefly: ignore [bad-assignment]
    dilation = pad_listlike(dilation, dim)
    # pyrefly: ignore [bad-assignment]
    padding = pad_listlike(padding, dim)
    # pyrefly: ignore [bad-assignment]
    stride = pad_listlike(stride, dim)

    window_size = functools.reduce(operator.mul, kernel_size)
    # We fallback when using non-default dilation or when the window size is too large
    if (
        torch._inductor.lowering.should_fallback_max_pool_with_indices(
            kernel_size, n_dim=dim
        )
        or window_size > torch.iinfo(torch.int8).max
    ):
        return NotImplemented

    vals, offsets = prims._low_memory_max_pool_with_offsets(
        x,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    )
    indices = prims._low_memory_max_pool_offsets_to_indices(
        offsets,
        kernel_size,
        x.shape[-dim:],
        stride,
        padding,
        dilation,
    )
    return vals, indices


@register_decomposition(aten.max_pool2d_with_indices)
def max_pool2d_with_indices(
    x: torch.Tensor,
    kernel_size: list[int],
    stride: int | list[int] | None = None,
    padding: int | list[int] = 0,
    dilation: int | list[int] = 1,
    ceil_mode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _max_pool_with_indices(
        x, kernel_size, stride, padding, dilation, ceil_mode, dim=2
    )


@register_decomposition(aten.max_pool3d_with_indices)
def max_pool3d_with_indices(
    x: torch.Tensor,
    kernel_size: list[int],
    stride: int | list[int] | None = None,
    padding: int | list[int] = 0,
    dilation: int | list[int] = 1,
    ceil_mode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _max_pool_with_indices(
        x, kernel_size, stride, padding, dilation, ceil_mode, dim=3
    )


@register_decomposition(aten.searchsorted.Scalar)
def searchsorted_scalar(
    sorted_sequence: torch.Tensor,
    self: torch.types.Number,
    *,
    out_int32: bool = False,
    right: bool = False,
    side: str | None = None,
    sorter: torch.Tensor | None = None,
) -> torch.Tensor:
    return aten.searchsorted(
        sorted_sequence,
        torch.tensor([self], device=sorted_sequence.device),
        out_int32=out_int32,
        right=right,
        side=side,
        sorter=sorter,
    )[0]


@register_decomposition(aten.bucketize.Scalar)
def bucketize_scalar(
    self: torch.types.Number,
    boundaries: torch.Tensor,
    *,
    out_int32: bool = False,
    right: bool = False,
) -> torch.Tensor:
    return aten.bucketize(
        torch.tensor([self], device=boundaries.device),
        boundaries,
        out_int32=out_int32,
        right=right,
    ).squeeze(0)


@decomp.register_decomposition(aten.rrelu_with_noise_functional, extra_random_decomps)
def rrelu_with_noise_functional(
    self: torch.Tensor,
    noise: torch.Tensor,
    lower: float = 0.125,
    upper: float = 0.3333333333333333,
    training: bool = False,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if training:
        not_positive = self <= 0
        r = aten.uniform(self, lower, upper, generator=generator)
        output = torch.where(not_positive, self * r, self)
        noise_out = torch.where(not_positive, r, 1)
        return output, noise_out
    else:
        negative_slope = (lower + upper) / 2
        return aten.leaky_relu(self, negative_slope), torch.Tensor()


@register_decomposition(aten.repeat_interleave.Tensor)
def repeat_interleave_Tensor(
    repeat: torch.Tensor,
    output_size: int | None = None,
) -> torch.Tensor:
    if config.triton.autotune_at_compile_time:
        # We can't compile-time auto-tune this because
        # it expects specific data in `repeat`
        return NotImplemented
    if output_size is None or type(output_size) is not int:
        return NotImplemented
    if repeat.device.type == "mps":
        return NotImplemented
    if repeat.dtype not in [torch.int32, torch.int64]:
        raise AssertionError(
            f"expected repeat dtype int32 or int64, got {repeat.dtype}"
        )
    if repeat.ndim != 1:
        raise AssertionError(f"expected repeat.ndim == 1, got {repeat.ndim}")
    cumsum = repeat.cumsum(0)
    pos = torch.arange(output_size, device=repeat.device)
    indices = torch.searchsorted(
        cumsum, pos, out_int32=(repeat.dtype == torch.int32), right=True
    )
    return torch.clamp(indices, max=repeat.size(0) - 1)


# intentionally not registered
def conv1d_to_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: tuple[int] = (1,),
    padding: tuple[int] = (0,),
    dilation: tuple[int] = (1,),
    groups: int = 1,
) -> torch.Tensor:
    # Shapes:
    # input:  (N, C_in, L_in)
    # weight: (C_out, C_in // groups, K)
    # bias:   (C_out,)
    if not (input.dim() == 3 and weight.dim() == 3):
        raise AssertionError("Expect (N,C_in,L) and (C_out,C_in//groups,K)")

    # pyrefly: ignore [bad-assignment]
    stride = stride[0]
    # pyrefly: ignore [bad-assignment]
    padding = padding[0]
    # pyrefly: ignore [bad-assignment]
    dilation = dilation[0]

    # Unsqueeze to make input 2D: (N,C,L) -> (N,C,L,1)
    input_2d = input.unsqueeze(-1)
    # Unsqueeze kernel: (C_out,C_in/groups,K) -> (C_out,C_in/groups,K,1)
    weight_2d = weight.unsqueeze(-1)

    # Call conv2d with adjusted args
    out_2d = aten.conv2d.default(
        input_2d,
        weight_2d,
        bias,
        stride=(stride, 1),
        padding=(padding, 0),
        dilation=(dilation, 1),
        groups=groups,
    )

    # Squeeze dummy dimension back out: (N,C_out,L_out,1) -> (N,C_out,L_out)
    return out_2d.squeeze(-1)
<|endoftext|>import json

from torch._export.serde.schema import ExternKernelNode, ExternKernelNodes, Node
from torch._export.serde.serialize import _dataclass_to_dict, EnumEncoder
from torch._inductor.ir import ExternKernelNode as inductor_ExternKernelNode


def serialize_extern_kernel_node(
    extern_kernel_node: inductor_ExternKernelNode,
) -> ExternKernelNode:
    if not isinstance(extern_kernel_node.node, Node):
        raise AssertionError(
            f"expected node to be a Node, got {type(extern_kernel_node.node)}"
        )
    return ExternKernelNode(
        name=extern_kernel_node.name,
        node=extern_kernel_node.node,
    )


def extern_node_json_serializer(
    extern_kernel_nodes: list[inductor_ExternKernelNode],
) -> str:
    serialized_nodes = ExternKernelNodes(
        nodes=[serialize_extern_kernel_node(node) for node in extern_kernel_nodes]
    )
    return json.dumps(_dataclass_to_dict(serialized_nodes), cls=EnumEncoder)
<|endoftext|>import collections
import typing_extensions
from collections.abc import Callable
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch._inductor.freezing_utils import maybe_set_is_frozen_param
from torch.utils._ordered_set import OrderedSet


aten = torch.ops.aten

# We would like to split modules into two subgraphs for runtime weight updates to work correctly.
# The use case and more information could be found at:
# https://docs.google.com/document/d/1inZC-8KarJ6gKB7G9egmYLx1V_dKX_apxon0w4zPC0Q/edit?usp=sharing
META_TAG = "MODULE_TYPE"
MODULE_TAG = "_MAIN_MODULE"
CONST_MODULE_TAG = "_CONST_MODULE"

_dont_constant_fold: list[torch.fx.node.Target] = []


def add_dont_constant_fold(op: torch.fx.node.Target) -> None:
    global _dont_constant_fold
    _dont_constant_fold.append(op)


def clear_dont_constant_fold() -> None:
    global _dont_constant_fold
    _dont_constant_fold.clear()


def replace_node_with_constant(
    gm: torch.fx.GraphModule,
    node: torch.fx.Node,
    constant: torch.Tensor | None = None,
    name: str | None = None,
) -> None:
    g = gm.graph

    if name:
        qualname = name
    else:
        if not hasattr(gm, "_frozen_param_count"):
            gm._frozen_param_count = 0  # type: ignore[assignment]
        i = gm._frozen_param_count
        # pyrefly: ignore [bad-assignment]
        while True:
            qualname = f"_frozen_param{i}"
            if not hasattr(gm, qualname):
                break
            i += 1  # type: ignore[assignment, operator]

        gm._frozen_param_count = i + 1  # type: ignore[assignment, operator]

    with g.inserting_before(node):
        if constant is not None:
            new_input_node = g.create_node("get_attr", qualname, (), {})
        else:
            # this is the case for lifted constants
            new_input_node = g.create_node("placeholder", qualname, (), {})
        node.replace_all_uses_with(new_input_node)
        new_input_node.meta.update(node.meta)
        g.erase_node(node)
        new_input_node.name = node.name

    if constant is not None:
        # needed to suppress `does not reference an nn.Module, nn.Parameter, or buffer` warning
        gm.register_buffer(qualname, constant)
        setattr(gm, qualname, constant)
        # mark any constants created during freezing
        maybe_set_is_frozen_param(constant)


def is_const_source(
    node: torch.fx.Node, lifted_constant_names: list[str] | None
) -> bool:
    return node.op == "get_attr" or node.name in (lifted_constant_names or ())


class ConstantFolder(torch.fx.Interpreter):
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        skip_constructors: bool = False,
        lifted_constant_names: list[str] | None = None,
        skip_folding_node_fn: Callable[[torch.fx.Node], bool] | None = None,
    ) -> None:
        super().__init__(gm)
        self.node_replacements: dict[torch.fx.Node, Any] = {}
        self.replaced_uses: dict[torch.fx.Node, int] = collections.Counter()
        self.unknown_value = object()
        self.skip_constructors: bool = skip_constructors

        # overwrite this to deallocate env values if their only remaining use
        # is the output
        self.user_to_last_uses = self.node_to_last_non_output_use()
        self.lifted_constant_names = lifted_constant_names
        self.deferred_value = object()
        self.skip_folding_node_fn = skip_folding_node_fn

    def _support_dynamic_shape(self) -> bool:
        # ConstantFolder not support dynamic shape now
        return False

    def _deduce_value(self, node: torch.fx.Node) -> Any:
        if self.lifted_constant_names is None:
            return super().run_node(node)
        # if lifted_constant_names is passed in, no concrete value is available
        # so we just check if all inputs have values
        if self.skip_folding_node_fn is not None and self.skip_folding_node_fn(node):
            return self.unknown_value
        flattened_node_inps = pytree.arg_tree_leaves(*node.args, **node.kwargs)
        for inp in flattened_node_inps:
            if (
                isinstance(inp, torch.fx.Node)
                and inp.name not in (self.lifted_constant_names or ())
                and self.env[inp] is not self.deferred_value
            ):
                return self.unknown_value
        return self.deferred_value

    def is_impure(self, node: torch.fx.node.Node) -> bool:
        def is_woq_int8_pattern(node: torch.fx.node.Node) -> bool:
            return (
                node.target is torch.ops.prims.convert_element_type.default  # type: ignore[return-value]
                and isinstance(node.args[0], torch.fx.Node)
                and "val" in node.args[0].meta
                and node.args[0].meta["val"].dtype == torch.int8  # type: ignore[union-attr]
                and node.args[1] == torch.bfloat16
            )

        if (
            is_woq_int8_pattern(node)
            or (
                node.target is torch.ops.aten.permute.default
                and len(node.users) == 1
                and is_woq_int8_pattern(next(iter(node.users)))
            )
        ) and is_const_source(
            node.args[0],  # type: ignore[arg-type]
            self.lifted_constant_names,
        ):
            # Case 1: int8_weight -> dq -> bf16_weight
            # Case 2: int8_weight -> permute -> dq -> bf16_weight
            return True

        quant_registered = (
            getattr(torch.ops.quantized_decomposed, "dequantize_per_channel", None)
            is not None
        )
        if quant_registered and node.target in [
            torch.ops.quantized_decomposed.dequantize_per_channel.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
            torch.ops.quantized_decomposed.convert_element_type.no_fuse,
        ]:
            # For the pattern fp32_weight -> q -> dq
            # We only folding fp32_weight -> q
            # int8_weight and leave dq in graph to be fused
            return True

        if node.target in _dont_constant_fold:
            return True
        return False

    def node_to_last_non_output_use(self) -> dict[torch.fx.Node, list[torch.fx.Node]]:
        last_non_output_use = collections.defaultdict(list)
        seen_uses = OrderedSet[torch.fx.Node]()
        output_node = next(iter(reversed(self.module.graph.nodes)))  # type: ignore[arg-type, union-attr]

        for node in reversed(self.module.graph.nodes):  # type: ignore[arg-type, union-attr]
            if node.target == "output":
                continue

            def add_use(inp: torch.fx.Node) -> None:
                if inp in seen_uses:
                    return

                seen_uses.add(inp)
                last_non_output_use[node].append(inp)

            # In-place is fine since we don't mutate
            pytree.tree_map_only_(torch.fx.Node, add_use, (node.args, node.kwargs))

            # if this node is only used in output, we want to gc it right away
            if len(node.users) == 1 and output_node in node.users:
                last_non_output_use[node].append(node)

        return last_non_output_use

    @typing_extensions.override
    def run_node(self, node: torch.fx.Node) -> Any:
        if node.target == "output":
            # because we remove nodes from env on last non output use,
            # re-define them now or we'll get error in interpreter
            def set_env(arg: torch.fx.Node) -> None:
                self.env[arg] = self.unknown_value

            # In-place is fine since we don't mutate
            pytree.tree_map_only_(torch.fx.Node, set_env, node.args)
            return super().run_node(node)

        args, kwargs = self.fetch_args_kwargs_from_env(node)
        flattened_inputs = pytree.arg_tree_leaves(*args, **kwargs)

        # We need to do this weird thing because in cases where flattened_inputs
        # contains a ScriptObject, equality checking results in a type error if
        # the types are different.
        if any(
            type(self.unknown_value) is type(input_) and self.unknown_value == input_
            for input_ in flattened_inputs
        ):
            return self.unknown_value

        # TODO - fix errors with this
        if (
            node.op == "call_function"
            and node.target is aten._efficientzerotensor.default
        ):
            return self.unknown_value

        # TODO - constant folding triton kernel returns the inputs -- fix this
        if (
            node.op == "call_function"
            and node.name == "triton_kernel_wrapper_functional_proxy"
        ):
            return self.unknown_value

        # skip constructors, since inductor generates optimal code for them already
        # and turning into tensor would result in an additional global memory read
        # TODO - more complicated strategy
        if (
            self.skip_constructors
            and not is_const_source(node, self.lifted_constant_names)
            and not any(isinstance(e, torch.Tensor) for e in flattened_inputs)
        ):
            return self.unknown_value

        # All mutations should either be removed or on inputs which we did not make constant
        if (
            isinstance(node.target, torch._ops.OpOverload)
            and torch.Tag.nondeterministic_seeded in node.target.tags
        ):
            return self.unknown_value

        if node.op == "call_function" and isinstance(
            node.target, torch._ops.HigherOrderOperator
        ):
            return self.unknown_value

        out = self._deduce_value(node)

        if isinstance(
            out,
            (
                torch._C.ScriptObject,
                torch._library.fake_class_registry.FakeScriptObject,
            ),
        ):
            return out

        if out is self.unknown_value:
            return self.unknown_value

        if not is_const_source(node, self.lifted_constant_names) and (
            isinstance(out, torch.Tensor) or out is self.deferred_value
        ):
            if (
                out is not self.deferred_value
                and out.device.type == "meta"  # pyrefly: ignore[missing-attribute]
            ):
                return out

            if not self.insertable_tensor_check(
                out  # pyrefly: ignore[bad-argument-type]
            ):
                return out

            if self.is_impure(node):
                return self.unknown_value

            self.add_node_replacement(node, out)  # pyrefly: ignore[bad-argument-type]

            flattened_node_inps = pytree.arg_tree_leaves(*node.args, **node.kwargs)

            for n in flattened_node_inps:
                if not isinstance(n, torch.fx.Node):
                    continue

                self.replaced_uses[n] += 1

            for to_delete in self.user_to_last_uses.get(node, []):
                if self.replaced_uses[to_delete] == len(to_delete.users):
                    self.node_replacements.pop(to_delete, None)

        return out

    def insertable_tensor_check(self, tensor: torch.Tensor) -> bool:
        return True

    def add_node_replacement(self, node: torch.fx.Node, tensor: torch.Tensor) -> None:
        self.node_replacements[node] = tensor

    def run(self) -> Any:  # type: ignore[override]
        env: dict[torch.fx.Node, Any] = {}
        self.insert_placerholder_values(env)
        return super().run(initial_env=env)

    def insert_placerholder_values(self, env: dict[torch.fx.Node, Any]) -> None:
        for n in self.module.graph.find_nodes(op="placeholder"):  # type: ignore[operator, union-attr]
            env[n] = self.unknown_value  # type: ignore[assignment]
        if self.lifted_constant_names is None:
            return
        for n in self.module.graph.nodes:  # type: ignore[union-attr]
            if n.name in (self.lifted_constant_names or ()):
                env[n] = self.deferred_value


def constant_fold(
    gm: torch.fx.GraphModule,
    constraint_fn: Callable[[torch.fx.Node], bool] | None = None,
) -> None:
    with torch.utils._python_dispatch._disable_current_modes():
        cf = ConstantFolder(gm, skip_constructors=True)
        cf.run()

        for node, constant in cf.node_replacements.items():
            if constraint_fn is not None and not constraint_fn(node):
                continue
            replace_node_with_constant(gm, node, constant)

        erased_params = []
        for node in gm.graph.find_nodes(op="get_attr"):
            if len(node.users) == 0:
                if hasattr(gm, node.target):
                    delattr(gm, node.target)
                erased_params.append(node)

        for node in erased_params:
            gm.graph.erase_node(node)

        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()


def constant_graph_tag(
    gm: torch.fx.GraphModule,
    skip_constructors: bool = True,
    lifted_constant_names: list[str] | None = None,
    skip_folding_node_fn: Callable[[torch.fx.Node], bool] | None = None,
) -> None:
    with torch.utils._python_dispatch._disable_current_modes():
        cf = ConstantFolder(
            gm,
            skip_constructors=skip_constructors,
            lifted_constant_names=lifted_constant_names,
            skip_folding_node_fn=skip_folding_node_fn,
        )
        cf.run()

        for node in gm.graph.nodes:
            if skip_folding_node_fn is not None and skip_folding_node_fn(node):
                node.meta[META_TAG] = MODULE_TAG
                continue
            if (
                is_const_source(node, lifted_constant_names)
                or node in cf.node_replacements
                or node in cf.replaced_uses
            ):
                node.meta[META_TAG] = CONST_MODULE_TAG
            else:
                node.meta[META_TAG] = MODULE_TAG


def run_and_get_constant_graph(
    gm: torch.fx.GraphModule,
    skip_constructors: bool = True,
    lifted_constant_names: list[str] | None = None,
    skip_folding_node_fn: Callable[[torch.fx.Node], bool] | None = None,
) -> torch.fx.GraphModule:
    """
    Construct a GraphModule which corresponds to the part which could be
    constant folded in provided gm.
    """

    constant_graph_tag(
        gm, skip_constructors, lifted_constant_names, skip_folding_node_fn
    )

    def untag(node: torch.fx.Node) -> bool:
        used_to_fold = False
        for u in node.users:
            if u.meta[META_TAG] == CONST_MODULE_TAG:
                used_to_fold = True
                break
        if not used_to_fold:
            node.meta[META_TAG] = MODULE_TAG
        return used_to_fold

    # We rewrite the tags, if it's a constant being directly consumed, without
    # any folding opportunity, we keep it in main gm.
    for node in gm.graph.nodes:
        if node.op == "get_attr" or (node.name in (lifted_constant_names or ())):
            untag(node)

    new_graph = torch.fx.Graph()

    node_remapping: dict[torch.fx.Node, torch.fx.Node] = {}
    output_nodes = []
    for node in gm.graph.nodes:
        if node.meta[META_TAG] == MODULE_TAG:
            continue

        new_node = new_graph.node_copy(node, lambda x: node_remapping[x])
        node_remapping[node] = new_node

        for user in node.users:
            if user.meta[META_TAG] == MODULE_TAG:
                output_nodes.append(new_node)
                break

    new_graph.output(tuple(output_nodes))
    new_graph.lint()
    new_gm = torch.fx.GraphModule(gm, new_graph)

    return new_gm
<|endoftext|># mypy: allow-untyped-defs
import functools
from collections.abc import Callable, Sequence
from typing import Any, Optional, Protocol, TYPE_CHECKING, TypeVar

import sympy

import torch
from torch._prims_common import (
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    is_integer_dtype,
    type_to_dtype,
)
from torch.utils._ordered_set import OrderedSet

from .ops_handler import OP_NAMES, OpsHandler
from .utils import upcast_compute_type
from .virtualized import OpsValue, V


T = TypeVar("T")
_MISSING_SHAPE = object()
_UNSIGNED_INT_DTYPES: frozenset[torch.dtype] = frozenset(
    (
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    )
)


class DTypeVar(Protocol):
    @property
    def dtype(self) -> torch.dtype: ...


DTypeArg = DTypeVar | torch.types.Number | str | OpsValue


# Inputs need to be cacheable (e.g., not a CSEVar) in order for the cache to be effective
# So first decompose CSEVars -> tuple before calling this


@functools.cache
def get_promoted_dtype(
    *args: Sequence[tuple[torch.dtype, bool]],
    type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND | None = None,
):
    def construct_input(inp):
        if inp[1]:
            return torch.empty([], dtype=inp[0])
        else:
            return torch.empty([1], dtype=inp[0])

    inps = [construct_input(arg) for arg in args]
    _, dtype = torch._prims_common.elementwise_dtypes(
        *inps,
        type_promotion_kind=(
            type_promotion_kind
            if type_promotion_kind
            else ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
        ),
    )
    return dtype


def promote_types(
    args: Sequence[DTypeArg],
    type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND | None = None,
):
    dtype_prop_candidates = []

    for arg in args:
        if isinstance(arg, str):
            raise AssertionError(f"expected non-str arg, got {type(arg)}")
        if isinstance(arg, OpsValue):
            arg = arg.value
            if not (
                isinstance(arg, torch._prims_common.Number) or hasattr(arg, "dtype")
            ):
                raise AssertionError("expected a Number or an object with a dtype")

        if isinstance(arg, torch._prims_common.Number):
            dtype_prop_candidates.append((type_to_dtype(type(arg)), True))
            continue

        # pyrefly: ignore [missing-attribute]
        dtype_prop_candidates.append((arg.dtype, getattr(arg, "is_scalar", False)))

    dtype = get_promoted_dtype(
        *dtype_prop_candidates,
        type_promotion_kind=type_promotion_kind,
    )

    return dtype


def _unwrap_dtype_arg(arg: DTypeArg) -> DTypeVar | torch.types.Number | str:
    if isinstance(arg, OpsValue):
        return arg.value
    return arg


def _is_scalar_dtype_arg(arg: DTypeArg) -> bool:
    arg = _unwrap_dtype_arg(arg)
    if isinstance(arg, torch._prims_common.Number):
        return True

    is_scalar = getattr(arg, "is_scalar", False)
    if callable(is_scalar):
        if is_scalar():
            return True
    elif is_scalar:
        return True

    shape = getattr(arg, "shape", _MISSING_SHAPE)
    if shape is _MISSING_SHAPE:
        return False

    if shape is None:
        return True
    if not isinstance(shape, Sequence):
        return False

    return len(shape) == 0


def _has_known_nonnegative_scalar_int_value(arg: DTypeArg) -> bool:
    arg = _unwrap_dtype_arg(arg)

    if isinstance(arg, bool):
        return True
    if isinstance(arg, int):
        return arg >= 0

    dtype = getattr(arg, "dtype", None)
    if dtype is None or not is_integer_dtype(dtype):
        return False
    if dtype in _UNSIGNED_INT_DTYPES:
        return True

    lower = getattr(getattr(arg, "bounds", None), "lower", None)
    if lower is None:
        return False
    if isinstance(lower, sympy.Expr):
        return lower.is_nonnegative is True
    return lower >= 0


class DtypePropagationOpsHandler:
    """
    Propagate dtype from args to output
    """

    # Singleton DtypePropagationOpsHandler, because we meta program over a number of op rules.
    # Those are only defined after other inductor state has run.

    _instance: Optional["DtypePropagationOpsHandler"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        for op, rule in torch._inductor.utils.op_dtype_propagation_rules.items():
            fn = (
                functools.partial(self.return_dtype, dtype=rule.override_return_dtype)
                if rule.override_return_dtype
                else functools.partial(
                    self.op_dtype_rule, type_promotion_kind=rule.type_promotion_kind
                )
            )
            setattr(self, op, fn)

        # Set pointwise operation rules
        for op in torch._inductor.codegen.common.pointwise_overrides_data.values():
            if not hasattr(self, op.name):
                setattr(
                    self,
                    op.name,
                    functools.partial(
                        self.op_dtype_rule, type_promotion_kind=op.type_promotion_kind
                    ),
                )

        # Set boolean operation rules
        for op in torch._inductor.utils.boolean_ops():
            if not hasattr(self, op):
                setattr(
                    self, op, functools.partial(self.return_dtype, dtype=torch.bool)
                )

        unimplemented_ops = OP_NAMES - OrderedSet(dir(self))
        torch._check(
            len(unimplemented_ops) == 0,
            lambda: f"Unimplemented dtype rule for ops: {unimplemented_ops}",
        )

    # metaprogrammed in __init__

    @staticmethod
    def op_dtype_rule(
        *args: DTypeArg, type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND
    ) -> torch.dtype:
        return promote_types(args, type_promotion_kind=type_promotion_kind)

    @staticmethod
    def return_dtype(*args: DTypeArg, dtype: torch.dtype) -> torch.dtype:
        return dtype

    # op rules

    @staticmethod
    def constant(value: torch.types.Number, dtype: torch.dtype) -> torch.dtype:
        return upcast_compute_type(dtype)

    @staticmethod
    def load_seed(name: str, offset: int) -> torch.dtype:
        return upcast_compute_type(V.graph.get_dtype(name))

    @staticmethod
    def randint64(seed: int, offset: int, low: int, high: int) -> torch.dtype:
        return torch.int64

    @staticmethod
    def masked(
        mask: DTypeArg, body: Callable[[], DTypeArg], other: DTypeArg
    ) -> torch.dtype:
        from .loop_body import LoopBodyBlock

        if not isinstance(body, LoopBodyBlock):
            raise AssertionError("body must be a LoopBodyBlock")
        # TODO - we avoid calling this in codegen, needs work for non codegen use cases
        loads = body.graph.find_nodes(op="call_method", target="load")
        if len(loads) <= 1:
            return promote_types([other])

        return upcast_compute_type(V.graph.get_dtype(loads[-1].args[1]))

    @staticmethod
    def where(a: DTypeArg, b: DTypeArg, c: DTypeArg) -> torch.dtype:
        return promote_types([b, c])

    @staticmethod
    def index_expr(expr: sympy.Expr, dtype: torch.dtype) -> torch.dtype:
        # `index_expr` is for indexing-style uses: the kernel's chosen
        # indexing dtype wins over the requested int dtype. For uses that
        # must honor `dtype` (e.g. user-typed `arange(int64)` whose result
        # participates in tensor computation), `value_expr` is the right op
        # - see `convert_index_expr_to_value_expr` which rewrites accordingly.
        if dtype not in (torch.int32, torch.int64) or not hasattr(
            V.kernel, "index_dtype"
        ):
            return upcast_compute_type(dtype)

        return V.kernel.get_index_dtype_as_torch_dtype()

    @staticmethod
    def value_expr(expr: sympy.Expr, dtype: torch.dtype) -> torch.dtype:
        # Unlike `index_expr`, `value_expr` always honors the requested
        # dtype because the result feeds tensor-value computation. The
        # sympy expression is emitted literally, so there is no
        # bf16/fp16 -> fp32 compute-type promotion here.
        return dtype

    @staticmethod
    def to_dtype(
        x: DTypeArg,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types=True,
    ) -> torch.dtype:
        return upcast_compute_type(dtype) if use_compute_types else dtype

    @staticmethod
    def to_dtype_bitcast(
        x: DTypeArg, dtype: torch.dtype, src_dtype: torch.dtype
    ) -> torch.dtype:
        return upcast_compute_type(dtype)

    @staticmethod
    def gelu(x: DTypeArg) -> torch.dtype:
        return promote_types([x])

    @staticmethod
    def mul(a: DTypeArg, b: DTypeArg) -> torch.dtype:
        return promote_types([a, b])

    @staticmethod
    def truediv(a: DTypeArg, b: DTypeArg) -> torch.dtype:
        return promote_types([a, b])

    @staticmethod
    def div_rn(a: DTypeArg, b: DTypeArg) -> torch.dtype:
        return promote_types([a, b])

    @staticmethod
    def pow(a: DTypeArg, b: DTypeArg) -> torch.dtype:
        dtype = promote_types([a, b])
        if (
            is_integer_dtype(dtype)
            and _is_scalar_dtype_arg(a)
            and _is_scalar_dtype_arg(b)
            and not _has_known_nonnegative_scalar_int_value(b)
        ):
            # Scalar integer pow follows Python semantics: negative exponents
            # produce a floating result, even though tensor integer pow stays
            # integral or errors out in separate lowering paths.
            return torch.float64
        return dtype

    @staticmethod
    def mod(a: DTypeArg, b: DTypeArg) -> torch.dtype:
        return promote_types([a, b])

    @staticmethod
    def indirect_indexing(
        x: DTypeArg, size: int, check: bool = True, wrap_neg: bool = True
    ) -> torch.dtype:
        return torch.int64

    @staticmethod
    def randn(seed: int, offset: int) -> torch.dtype:
        return torch.float

    @staticmethod
    def rand(seed: int, offset: int) -> torch.dtype:
        return torch.float

    @staticmethod
    def rand_eager(seed, offset, threads_per_round, tid, vec) -> torch.dtype:
        return torch.float

    @staticmethod
    def store_reduction(name: str, index, value: DTypeArg) -> None:
        return None

    @staticmethod
    def reduction(
        dtype: torch.dtype, src_dtype: torch.dtype, reduction_type: str, value: DTypeArg
    ) -> torch.dtype:
        return dtype

    @staticmethod
    def store(name: str, index, value: DTypeArg, mode: str | None = None) -> None:
        return None

    @staticmethod
    def partial_accumulate(
        name: str,
        reduction_type: str,
        value: DTypeArg,
        extra_meta: dict[str, Any],
    ) -> None:
        return None

    @staticmethod
    def load(name: str, index) -> torch.dtype:
        return upcast_compute_type(V.graph.get_dtype(name))

    @staticmethod
    def floor(x: DTypeArg) -> torch.dtype:
        return promote_types(
            [x], type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
        )

    @staticmethod
    def ceil_to_int(x: DTypeArg, dtype: torch.dtype) -> torch.dtype:
        return dtype

    @staticmethod
    def int_truediv(x: DTypeArg, y: DTypeArg) -> torch.dtype:
        return promote_types(
            [x, y], type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
        )

    @staticmethod
    def scan(
        dtypes: tuple[torch.dtype, ...],
        combine_fn: Callable[[tuple[T, ...], tuple[T, ...]], tuple[T, ...]],
        values: tuple[T, ...],
    ) -> tuple[torch.dtype, ...]:
        return dtypes

    @staticmethod
    def fmod(x: DTypeArg, y: DTypeArg) -> torch.dtype:
        return promote_types([x, y])

    @staticmethod
    def round_to_int(x: DTypeArg, dtype: torch.dtype) -> torch.dtype:
        return dtype

    @staticmethod
    def identity(x: DTypeArg) -> torch.dtype:
        return promote_types([x])

    @staticmethod
    def frexp(x: DTypeArg) -> tuple[torch.dtype, torch.dtype]:
        # TODO - need to handle multiple outputs
        return (promote_types([x]), torch.int32)

    @staticmethod
    def sort(
        dtypes: tuple[torch.dtype, ...],
        values: tuple[T, ...],
        stable: bool,
        descending: bool,
    ) -> tuple[torch.dtype, ...]:
        return dtypes

    @staticmethod
    def trunc(x: DTypeArg) -> torch.dtype:
        return promote_types([x])

    @staticmethod
    def bucketize(
        values: DTypeArg,
        boundaries: tuple[str, sympy.Expr, sympy.Expr, sympy.Expr],
        boundary_indices: DTypeArg,
        indexing_dtype: torch.dtype,
        right: bool,
        sorter: tuple[str, sympy.Expr] | None = None,
        sorter_indices: T | None = None,
    ) -> torch.dtype:
        return indexing_dtype

    @staticmethod
    def rshift(x: DTypeArg, y: DTypeArg) -> torch.dtype:
        return promote_types([x])

    @staticmethod
    def round(x: DTypeArg) -> torch.dtype:
        return promote_types(
            [x], type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
        )

    @staticmethod
    def trunc_to_int(x: DTypeArg, dtype: torch.dtype) -> torch.dtype:
        return dtype

    @staticmethod
    def floor_to_int(x: DTypeArg, dtype: torch.dtype) -> torch.dtype:
        return dtype

    @staticmethod
    def truncdiv(x: DTypeArg, y: DTypeArg) -> torch.dtype:
        return promote_types([x, y])

    @staticmethod
    def floordiv(x: DTypeArg, y: DTypeArg) -> torch.dtype:
        return promote_types([x, y])

    @staticmethod
    def halide_clamp(value, size, check):
        # TODO - way of registering dtype for op in backend
        return torch.int32

    @staticmethod
    def dot(x: DTypeArg, y: DTypeArg) -> torch.dtype:
        # triton tl.dot out_dtype is tl.float32 by default.
        return torch.float32

    @staticmethod
    def inline_asm_elementwise(
        *inputs,
        asm,
        constraints=None,
        dtype=torch.float32,
        is_pure=True,
        pack=1,
        input_dtypes=None,
    ):
        return dtype

    @staticmethod
    def lshift(x: DTypeArg, y: DTypeArg) -> torch.dtype:
        return promote_types([x])

    @staticmethod
    def check_bounds(
        expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ) -> None:
        return None

    def output(self, *args: DTypeArg) -> None:
        raise AssertionError(
            f"{type(self).__name__}: ops.output should not appear here"
        )

    def placeholder(self, index: int) -> torch.dtype:
        raise AssertionError(
            f"{type(self).__name__}: ops.placeholder should not appear here"
        )

    @staticmethod
    def device_assert_async(cond, msg: str) -> None:
        return None


if TYPE_CHECKING:
    # pyrefly: ignore [inconsistent-inheritance]
    class _typecheck_DtypePropagation(DtypePropagationOpsHandler, OpsHandler[Any]):
        pass  # mypy will error if we got any of the signatures wrong
<|endoftext|># mypy: allow-untyped-defs
from collections.abc import Sequence
from typing import Any, Optional

import sympy

import torch
from torch._prims_common import make_channels_last_strides_for, StrideType
from torch.utils._ordered_set import OrderedSet

from .ir import (
    ExternKernelAlloc,
    FixedLayout,
    FlexibleLayout,
    get_device_type,
    ir_node_to_tensor,
    IRNode,
    is_contiguous_storage_and_layout,
    Layout,
    may_convert_to_optional,
    MultiOutput,
    MultiOutputLayout,
    MutationOutput,
    NoneLayout,
    TensorBox,
)
from .utils import convert_shape_to_inductor, pad_listlike, SUPPORTED_MKLDNN_DEVICES
from .virtualized import V


def _prepare_convolution_fusion_create(
    cls,
    x: "TensorBox",
    weight: "TensorBox",
    bias: "TensorBox",
    padding: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
    groups: int,
    transposed: bool = False,
    output_padding: Sequence[int] | None = None,
    quantize_args: list["TensorBox"] | None = None,
    other: Optional["TensorBox"] = None,
):
    """
    This function is a helper function to prepare inputs, layout and constant args
    for convolution post-op fusion's create function, including deciding the output
    layout (channels first or channels last), realizing inputs and make them etc. The
    function only supports the CPU/XPU device since conv post-op fusion kernel is only
    supported on CPU/XPU right now.
    """

    # Port from aten/src/ATen/native/ConvUtils.h: _conv_input_size
    def _conv_input_size(
        output_size, weight_size, padding, output_padding, stride, dilation, groups
    ):
        if len(output_size) != len(weight_size):
            raise AssertionError("Expect input dim == weight dim")
        dim = len(output_size)
        if dim <= 2:
            raise AssertionError("Expect input dim > 2")

        BATCH_DIM = 0
        WEIGHT_INPUT_CHANNELS_DIM = 1
        input_size = []
        input_size.append(output_size[BATCH_DIM])
        input_size.append(weight_size[WEIGHT_INPUT_CHANNELS_DIM] * groups)
        for d in range(2, dim):
            kernel = (weight_size[d] - 1) * dilation[d - 2] + 1
            input_size_d = (
                (output_size[d] - 1) * stride[d - 2]
                - (padding[d - 2] * 2)
                + kernel
                + output_padding[d - 2]
            )
            input_size.append(input_size_d)
        return list(map(int, input_size))

    # Port from aten/src/ATen/native/ConvUtils.h: _conv_output_size
    def _conv_output_size(input_size, weight_size, padding, stride, dilation=None):
        has_dilation = dilation is not None
        dim = len(input_size)
        output_size = []
        output_size.append(input_size[0])
        output_size.append(weight_size[0])
        for d in range(2, dim):
            # pyrefly: ignore [unsupported-operation]
            dilation_ = dilation[d - 2] if has_dilation else 1
            kernel = dilation_ * (weight_size[d] - 1) + 1
            output_size_d = (input_size[d] + (2 * padding[d - 2]) - kernel) // stride[
                d - 2
            ] + 1
            output_size.append(output_size_d)
        return output_size

    # The size of prepacked_weight is the prepacked weight size of deconv:
    #   Groups > 1:  [g*o, i/g, ...]
    #   Groups == 1: [o, i, ...]
    # Returns original weight size in [i, o, ...]
    def _original_deconv_weight_size(
        prepacked_weight,
        groups,
    ):
        prepacked_weight_size = prepacked_weight.size()
        dim = len(prepacked_weight_size)
        if dim <= 2:
            raise AssertionError("Expect weight dim > 2")
        if groups > 1:
            weight_size = []
            weight_size.append(prepacked_weight_size[1] * groups)
            weight_size.append(prepacked_weight_size[0] / groups)
            weight_size.extend(prepacked_weight_size[d] for d in range(2, dim))
        else:
            weight_size = prepacked_weight.transpose(0, 1).size()
        return weight_size

    x.realize()
    weight.realize()
    if bias is not None:
        bias.realize()
    with V.graph.fake_mode:
        # TODO <Leslie> cleaned up the fake_tensor trace as Linear implementation
        x_fake = ir_node_to_tensor(x)
        weight_fake = ir_node_to_tensor(weight)
        dims = len(x_fake.size()) - 2
        if not (0 < len(padding) <= dims):
            raise AssertionError(f"expected 0 < len(padding) <= {dims}")
        if not (0 < len(dilation) <= dims):
            raise AssertionError(f"expected 0 < len(dilation) <= {dims}")
        if not (0 < len(stride) <= dims):
            raise AssertionError(f"expected 0 < len(stride) <= {dims}")
        padding = pad_listlike(padding, dims)
        dilation = pad_listlike(dilation, dims)
        stride = pad_listlike(stride, dims)
        if output_padding is None:
            output_padding = pad_listlike([0], dims)
        else:
            if not (0 < len(output_padding) <= dims):
                raise AssertionError(f"expected 0 < len(output_padding) <= {dims}")
            output_padding = pad_listlike(output_padding, dims)
        if not isinstance(groups, (int, sympy.core.numbers.Integer)):
            raise AssertionError(f"expected groups to be int, got {type(groups)}")
        if transposed:
            # When transposed, the size of the prepacked oneDNN weight is different
            # from the PyTorch weight. We're not able to run aten conv with such
            # size. We infer the output size from the input params here:
            weight_size = _original_deconv_weight_size(weight_fake, groups)
            input_size = x_fake.size()
            output_size = _conv_input_size(
                input_size,
                weight_size,
                padding,
                output_padding,
                stride,
                dilation,
                groups,
            )
        else:
            x_shape = list(x_fake.shape)
            weight_shape = list(weight_fake.shape)
            if len(x_shape) != len(weight_shape):
                if not (len(x_shape) == 3 and len(weight_shape) == 4):
                    raise AssertionError(
                        f"expected len(x_shape) == 3 and len(weight_shape) == 4, "
                        f"got {len(x_shape)} and {len(weight_shape)}"
                    )
                weight_shape.pop(2)
            output_size = _conv_output_size(
                x_shape,
                weight_shape,
                padding,
                stride,
                dilation,
            )

        req_stride_order = [0] + list(reversed(range(1, len(stride) + 1)))
        req_stride_order = [len(req_stride_order)] + req_stride_order

    x = cls.require_stride_order(x, req_stride_order)

    # We won't do weight prepack for Conv if dynamic_shapes or if is xpu.
    # In static shape cases, since weight is prepacked, we'll always force output to be channels last in the Conv kernel.
    # In dynamic shape cases, for input with channels = 1, like tensor of size (s0, 1, 28, 28) and stride (784, 784, 28, 1),
    # x = cls.require_stride_order(x, req_stride_order) where req_stride_order is in the channels last order
    # won't change the stride of this tensor since stride for dimensions of size 1 is ignored. While in Conv kernel,
    # this tensor is considered as channels first and the output will be in contiguous format.
    # To align the behavior of the Conv kernel, we set the output_stride in such case to be contiguous instead of channels last.
    dynamic_shapes = not all(isinstance(i, int) for i in (output_size))
    if (
        dynamic_shapes or get_device_type(x) == "xpu"
    ) and is_contiguous_storage_and_layout(x):
        output_stride: StrideType = FlexibleLayout.contiguous_strides(output_size)
    # Currently we don't support channel last for the situation that stride of input's batch dim is 0,
    # eg. input_size = (1, 1280, 64, 64), but input_stride=(0, 1, 81920, 1280).
    # So we use NCHW here instead.
    # Different with cpu, cpu conv always use channels_last for convolution when weight is prepacked,
    # but xpu does not do the prepack, so the problem exposed here is only for xpu.
    # TODO support channels_last for such zero stride input.
    elif get_device_type(x) == "xpu" and x.get_stride()[0] == 0:
        output_stride = FlexibleLayout.contiguous_strides(output_size)
    else:
        output_stride = make_channels_last_strides_for(output_size)

    if get_device_type(x) != get_device_type(weight):
        raise AssertionError(
            f"expected x and weight on same device, got "
            f"{get_device_type(x)} and {get_device_type(weight)}"
        )
    if get_device_type(x) not in SUPPORTED_MKLDNN_DEVICES:
        raise AssertionError(f"unsupported mkldnn device {get_device_type(x)}")
    inputs = [x]

    if quantize_args is not None:
        x_scale, x_zero_point, w_scale, w_zero_point = quantize_args
        x_scale.realize()
        x_zero_point.realize()
        w_scale.realize()
        w_zero_point.realize()
        inputs = inputs + [x_scale, x_zero_point] + [weight] + [w_scale, w_zero_point]
    else:
        inputs += [weight]

    if other is not None:
        other = cls.require_stride_order(other, req_stride_order)
        if not isinstance(other, TensorBox):
            raise AssertionError(f"expected other to be TensorBox, got {type(other)}")
        inputs += [other]

    kernel_layout = FixedLayout(
        x.get_device_or_error(),
        x.get_dtype(),
        convert_shape_to_inductor(output_size),
        convert_shape_to_inductor(output_stride),
    )
    constant_args = [padding, stride, dilation, groups]
    if transposed:
        constant_args.insert(1, output_padding)

    if bias is not None:
        inputs.append(bias)
    else:
        constant_args.insert(0, bias)
    return inputs, constant_args, kernel_layout, req_stride_order, other


def _prepare_linear_fusion_create(
    cls,
    x: "TensorBox",
    weight: "TensorBox",
    bias: "TensorBox",
    quantize_args: list["TensorBox"] | None = None,
    other: Optional["TensorBox"] = None,
    binary_sum: bool = False,
):
    """
    This function is a helper function to prepare inputs, layout and constant args
    for linear post-op fusion's create function. The function only supports the CPU device
    since linear post-op fusion kernel is only supported on CPU right now.
    """
    x.realize()
    weight.realize()
    if bias is not None:
        bias.realize()

    *m, _ = x.get_size()
    # The weight has been transposed during the qlinear weight prepack process.
    # https://github.com/pytorch/pytorch/blob/4979f9c0d72490970e2019bb1d2284f83d93f76b/
    # aten/src/ATen/native/quantized/cpu/qlinear_prepack.cpp#L291
    _, oc = weight.get_size()
    output_size = list(m) + [oc]
    req_stride_order = list(reversed(range(len(x.get_size()))))

    x = cls.require_stride_order(x, req_stride_order)
    if get_device_type(x) != get_device_type(weight):
        raise AssertionError(
            f"expected x and weight on same device, got "
            f"{get_device_type(x)} and {get_device_type(weight)}"
        )
    if get_device_type(x) not in SUPPORTED_MKLDNN_DEVICES:
        raise AssertionError(f"unsupported mkldnn device {get_device_type(x)}")
    inputs = [x]

    if quantize_args is not None:
        x_scale, x_zero_point, w_scale, w_zero_point = quantize_args
        x_scale.realize()
        x_zero_point.realize()
        w_scale.realize()
        w_zero_point.realize()
        inputs = inputs + [x_scale, x_zero_point] + [weight] + [w_scale, w_zero_point]
    else:
        inputs += [weight]

    if other is not None:
        if binary_sum:
            other = cls.require_stride_order(other, req_stride_order)
        inputs = inputs + [other]

    output_stride = FlexibleLayout.contiguous_strides(output_size)
    kernel_layout = FixedLayout(
        x.get_device(),
        x.get_dtype(),
        output_size,
        output_stride,
    )
    constant_args: list[Any] = []

    if bias is not None:
        inputs.append(bias)
    else:
        constant_args.insert(0, bias)
    return inputs, constant_args, kernel_layout, req_stride_order, other


def _create_output_node(packed):
    output_ir = MultiOutput(
        packed.get_layout(),
        packed,
        [],
    )
    packed.layout = MultiOutputLayout(device=packed.get_device())
    packed.outputs = [output_ir]
    return output_ir


class ConvolutionUnary(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
    ) -> None:
        self.device_type = get_device_type(inputs[0])
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            op_overload=torch.ops.mkldnn._convolution_pointwise.default,
            cpp_kernel_name=f"aoti_torch_{self.device_type}_mkldnn__convolution_pointwise",
        )

    def codegen(self, wrapper):
        wrapper.include_extra_header(
            f"torch/csrc/inductor/aoti_torch/c/shim_{self.device_type}.h"
        )
        super().codegen(wrapper)

    @classmethod
    def create(
        cls,
        x: "TensorBox",
        weight: "TensorBox",
        bias: "TensorBox",
        padding_: list[int],
        stride_: list[int],
        dilation_: list[int],
        groups: int,
        attr,
        scalars: list[Any] | None,
        algorithm,
    ):
        (
            inputs,
            constant_args,
            kernel_layout,
            _,
            _,
        ) = _prepare_convolution_fusion_create(
            cls, x, weight, bias, padding_, stride_, dilation_, groups
        )
        constant_args = constant_args + [
            attr,
            may_convert_to_optional(scalars),
            algorithm,
        ]
        packed = ConvolutionUnary(
            layout=kernel_layout,
            inputs=inputs,
            constant_args=constant_args,
        )
        return _create_output_node(packed)


class ConvolutionBinary(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
        cpp_constant_args=(),
    ) -> None:
        self.device_type = get_device_type(inputs[0])
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            op_overload=torch.ops.mkldnn._convolution_pointwise.binary,
            cpp_kernel_name=f"aoti_torch_{self.device_type}_mkldnn__convolution_pointwise_binary",
        )
        self.cpp_constant_args = cpp_constant_args

    def codegen(self, wrapper):
        wrapper.include_extra_header(
            f"torch/csrc/inductor/aoti_torch/c/shim_{self.device_type}.h"
        )
        super().codegen(wrapper)

    @classmethod
    def create(
        cls,
        x: "TensorBox",
        other: "TensorBox",
        weight: "TensorBox",
        bias: "TensorBox",
        padding_: list[int],
        stride_: list[int],
        dilation_: list[int],
        groups: int,
        binary_attr: str,
        binary_alpha: float | None,
        unary_attr: str | None,
        unary_scalars: list[Any] | None,
        unary_algorithm: str | None,
    ):
        (
            inputs,
            constant_args,
            kernel_layout,
            req_stride_order,
            _,
        ) = _prepare_convolution_fusion_create(
            cls, x, weight, bias, padding_, stride_, dilation_, groups
        )
        # pyrefly: ignore [bad-assignment]
        other = cls.require_stride_order(other, req_stride_order)
        inputs.insert(1, other)
        constant_args = constant_args + [
            binary_attr,
            binary_alpha,
            unary_attr,
            may_convert_to_optional(unary_scalars),
            unary_algorithm,
        ]
        packed = ConvolutionBinary(
            layout=kernel_layout,
            inputs=inputs,
            constant_args=constant_args,
        )
        return _create_output_node(packed)


class ConvolutionBinaryInplace(ExternKernelAlloc):
    def __init__(
        self,
        kernel_layout,
        inputs,
        constant_args=(),
    ) -> None:
        # Due to constraint of op.call, other (Tensor&) should be at input[0]
        self.device_type = get_device_type(inputs[0])
        reordered_inputs = [inputs[1], inputs[0]] + inputs[2:]

        super().__init__(
            kernel_layout,
            reordered_inputs,
            constant_args,
            None,
            op_overload=torch.ops.mkldnn._convolution_pointwise_.binary,
            cpp_kernel_name=f"aoti_torch_{self.device_type}_mkldnn__convolution_pointwise_binary_",
        )

        self.mutation_outputs = [
            MutationOutput(NoneLayout(device=inputs[0].get_device()), inputs[0], self),
            MutationOutput(NoneLayout(device=inputs[1].get_device()), inputs[1], self),
        ]

    def codegen(self, wrapper):
        wrapper.include_extra_header(
            f"torch/csrc/inductor/aoti_torch/c/shim_{self.device_type}.h"
        )
        super().codegen(wrapper)

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    @classmethod
    def create(
        cls,
        x: "TensorBox",
        other: "TensorBox",
        weight: "TensorBox",
        bias: "TensorBox",
        padding_: list[int],
        stride_: list[int],
        dilation_: list[int],
        groups: int,
        binary_attr: str,
        binary_alpha: float | None,
        unary_attr: str | None,
        unary_scalars: list[Any] | None,
        unary_algorithm: str | None,
    ):
        (
            inputs,
            constant_args,
            _,
            req_stride_order,
            _,
        ) = _prepare_convolution_fusion_create(
            cls, x, weight, bias, padding_, stride_, dilation_, groups
        )
        # pyrefly: ignore [bad-assignment]
        other = cls.require_stride_order(other, req_stride_order)
        inputs.insert(1, other)
        constant_args = constant_args + [
            binary_attr,
            binary_alpha,
            unary_attr,
            may_convert_to_optional(unary_scalars),
            unary_algorithm,
        ]
        packed = ConvolutionBinaryInplace(
            kernel_layout=NoneLayout(device=inputs[1].get_device()),  # type: ignore[arg-type]
            inputs=inputs,
            constant_args=constant_args,
        )
        # This op mutates in place which means that the result is not the
        # target but rather the input that is being mutated
        # init reorders the inputs, so inputs[1] becomes packed.inputs[0]
        return packed.inputs[0]


class ConvolutionTransposeUnary(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
    ) -> None:
        self.device_type = get_device_type(inputs[0])
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            op_overload=torch.ops.mkldnn._convolution_transpose_pointwise.default,
            cpp_kernel_name=f"aoti_torch_{self.device_type}_mkldnn__convolution_transpose_pointwise",
        )

    def codegen(self, wrapper):
        wrapper.include_extra_header(
            f"torch/csrc/inductor/aoti_torch/c/shim_{self.device_type}.h"
        )
        super().codegen(wrapper)

    @classmethod
    def create(
        cls,
        x: "TensorBox",
        weight: "TensorBox",
        bias: "TensorBox",
        padding_: list[int],
        output_padding_: list[int],
        stride_: list[int],
        dilation_: list[int],
        groups_: int,
        attr,
        scalars: list[Any] | None,
        algorithm,
    ):
        transposed = True
        (
            inputs,
            constant_args,
            kernel_layout,
            _,
            _,
        ) = _prepare_convolution_fusion_create(
            cls,
            x,
            weight,
            bias,
            padding_,
            stride_,
            dilation_,
            groups_,
            transposed,
            output_padding_,
        )
        constant_args = constant_args + [
            attr,
            may_convert_to_optional(scalars),
            algorithm,
        ]
        packed = ConvolutionTransposeUnary(
            layout=kernel_layout,
            inputs=inputs,
            constant_args=constant_args,
        )
        return _create_output_node(packed)


class QConvPointWisePT2E(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
    ) -> None:
        """
        if bias is not None
            - inputs = [x, w, b, weight_scale, weight_zp]
            - const_args is: [stride, padding, dilation, groups, x_scale, x_zp, o_scale, o_zp,
              fp32_output, unary_attr, unary_scalars, unary_algorithm]
        else
            - inputs = [x, w, weight_scale, weight_zp]
            - const_args is: [bias, stride, padding, dilation, groups, x_scale, x_zp, o_scale, o_zp,
              fp32_output, unary_attr, unary_scalars, unary_algorithm]
        """
        self.device_type = get_device_type(inputs[0])
        self.has_bias = len(inputs) == 5
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            op_overload=torch.ops.onednn.qconv_pointwise.tensor,
            cpp_kernel_name=f"aoti_torch_{self.device_type}__qconv_pointwise_tensor",
        )

    def codegen(self, wrapper):
        wrapper.include_extra_header(
            f"torch/csrc/inductor/aoti_torch/c/shim_{self.device_type}.h"
        )
        super().codegen(wrapper)
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    @classmethod
    def create(
        cls,
        qx: "TensorBox",
        x_scale: "TensorBox",
        x_zero_point: "TensorBox",
        qw: "TensorBox",  # qw
        w_scale: "TensorBox",
        w_zero_point,
        bias: "TensorBox",
        stride: list[int],
        padding: list[int],
        dilation: list[int],
        groups: int,
        output_scale: float,
        output_zero_point: int,
        output_dtype,
        attr,
        scalars,
        algorithm,
    ):
        transposed = False
        output_padding = None
        (
            inputs,
            constant_args,
            kernel_layout,
            _,
            _,
        ) = _prepare_convolution_fusion_create(
            cls,
            qx,
            qw,
            bias,
            padding,
            stride,
            dilation,
            groups,
            transposed,
            output_padding,
            [x_scale, x_zero_point, w_scale, w_zero_point],  # type: ignore[list-item]
        )
        # swap padding and stride to align with functional conv arg order
        if bias is None:
            constant_args[1], constant_args[2] = constant_args[2], constant_args[1]
        else:
            constant_args[0], constant_args[1] = constant_args[1], constant_args[0]

        constant_args = constant_args + [
            output_scale,
            output_zero_point,
            output_dtype,
            attr,
            may_convert_to_optional(scalars),
            algorithm,
        ]

        if output_dtype is None:
            raise AssertionError("expected output_dtype to be not None")
        if output_dtype in [torch.float32, torch.bfloat16]:
            # in _prepare_convolution_fusion_create, we use x.dtype (uint8) to create kernel_layout
            # if we set output_dtype is not None, the output buf should be output_dtype instead of uint8.
            kernel_layout.dtype = output_dtype

        return QConvPointWisePT2E(
            layout=kernel_layout,
            inputs=inputs,
            constant_args=constant_args,
        )


class QConvPointWiseBinaryPT2E(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
    ) -> None:
        """
        Needs input/weight/output qparams
        if bias is not None
            - inputs = [x, x_scale, x_zp, w,  w_scale, w_zp, accum, b]
            - const_args = [stride, padding, dilation, groups, o_scale, o_zp,
            output_dtype, accum_scale, accum_zp, binary_attr, alpha, unary_attr, unary_scalars, unary_algorithm]
        else
            - inputs = [x, x_scale, x_zp, w,  w_scale, w_zp, accum]
            - const_args [b, stride, padding, dilation, groups, o_scale, o_zp,
             output_dtype, accum_scale, accum_zp, binary_attr, alpha, unary_attr, unary_scalars, unary_algorithm]
        """
        self.device_type = get_device_type(inputs[0])
        self.has_bias = len(inputs) == 8
        self.idx_for_inplace_sum = 6
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            op_overload=torch.ops.onednn.qconv2d_pointwise.binary_tensor,
            cpp_kernel_name=(
                f"aoti_torch_{self.device_type}__qconv2d_pointwise_binary_tensor"
            ),
        )

    def codegen(self, wrapper):
        wrapper.include_extra_header(
            f"torch/csrc/inductor/aoti_torch/c/shim_{self.device_type}.h"
        )
        super().codegen(wrapper)
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    def get_mutation_names(self) -> Sequence[str]:
        return [self.input_name(self.idx_for_inplace_sum)]

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    @classmethod
    def create(
        cls,
        qx: "TensorBox",
        x_scale: "TensorBox",
        x_zero_point: "TensorBox",
        qw: "TensorBox",  # packed_weight
        w_scale,
        w_zero_point,
        qaccum: "TensorBox",
        bias: "TensorBox",
        stride: list[int],
        padding: list[int],
        dilation: list[int],
        groups: int,
        output_scale: "TensorBox",
        output_zero_point: "TensorBox",
        output_dtype,
        accum_scale,
        accum_zero_point,
        binary_attr,
        alpha,
        unary_attr,
        unary_scalars,
        unary_algorithm,
    ):
        transposed = False
        output_padding = None
        (
            inputs,
            constant_args,
            _kernel_layout,
            req_stride_order,
            qaccum,
        ) = _prepare_convolution_fusion_create(
            cls,
            qx,
            qw,
            bias,
            padding,
            stride,
            dilation,
            groups,
            transposed,
            output_padding,
            [x_scale, x_zero_point, w_scale, w_zero_point],
            qaccum,
        )

        # swap padding and stride to align with functional conv arg order
        if bias is None:
            constant_args[1], constant_args[2] = constant_args[2], constant_args[1]
        else:
            constant_args[0], constant_args[1] = constant_args[1], constant_args[0]

        constant_args = constant_args + [
            output_scale,
            output_zero_point,
            output_dtype,
            accum_scale,
            accum_zero_point,
            binary_attr,
            alpha,
            unary_attr,
            may_convert_to_optional(unary_scalars),
            unary_algorithm,
        ]

        if binary_attr != "sum":
            raise AssertionError(
                "For now, only post op sum is supported in QConvPointWiseBinaryPT2E."
            )

        V.graph.mark_buffer_mutated(qaccum.get_name())
        packed = QConvPointWiseBinaryPT2E(
            layout=NoneLayout(device=qaccum.get_device()),
            inputs=inputs,
            constant_args=constant_args,
        )

        # Return accum since it has been inplace changed.
        return packed.inputs[packed.idx_for_inplace_sum]


class MKLPackedLinear(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
    ) -> None:
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            op_overload=torch.ops.mkl._mkl_linear.default,
        )

    def codegen(self, wrapper):
        wrapper.include_extra_header("torch/csrc/inductor/aoti_torch/c/shim_cpu.h")
        super().codegen(wrapper)

    @classmethod
    def create(cls, x, packed_w, orig_w, B, batch_size):
        x = cls.require_stride1(cls.realize_input(x))
        orig_w = cls.require_stride1(cls.realize_input(orig_w))
        *m, _ = x.get_size()
        oc, _ = orig_w.get_size()
        output_size = list(m) + [oc]
        output_stride = FlexibleLayout.contiguous_strides(output_size)
        inputs = [x, packed_w, orig_w]
        constant_args = [batch_size]
        if B is not None:
            inputs += [B]
        else:
            constant_args.insert(0, None)

        device = x.get_device()
        if device is None:
            raise AssertionError("expected x to have a device")
        return MKLPackedLinear(
            layout=FixedLayout(device, x.get_dtype(), output_size, output_stride),
            inputs=inputs,
            constant_args=constant_args,
        )


class LinearUnary(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
    ) -> None:
        self.device_type = get_device_type(inputs[0])
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            op_overload=torch.ops.mkldnn._linear_pointwise.default,
            cpp_kernel_name=f"aoti_torch_{self.device_type}__linear_pointwise",
        )

    def codegen(self, wrapper):
        wrapper.include_extra_header(
            f"torch/csrc/inductor/aoti_torch/c/shim_{self.device_type}.h"
        )
        super().codegen(wrapper)

    @classmethod
    def create(cls, x, w, B, attr, scalars, algorithm):
        x = cls.require_contiguous(cls.realize_input(x))
        w = cls.require_contiguous(cls.realize_input(w))

        *m, _ic = x.get_size()
        oc, _ic = w.get_size()
        output_size = list(m) + [oc]
        inputs = [x, w]
        constant_args = [attr, scalars if scalars else [-1], algorithm]
        if B is not None:
            B = cls.require_contiguous(cls.realize_input(B))
            inputs.append(B)
        else:
            constant_args.insert(0, None)

        device = x.get_device()
        if device is None:
            raise AssertionError("expected x to have a device")

        packed = LinearUnary(
            layout=FixedLayout(
                device=device,
                dtype=x.get_dtype(),
                size=output_size,
            ),
            inputs=inputs,
            constant_args=constant_args,
        )
        return _create_output_node(packed)

    def apply_constraint(self):
        pass


class LinearBinary(ExternKernelAlloc):
    kernel = "torch.ops.mkldnn._linear_pointwise.binary"

    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
    ) -> None:
        self.device_type = get_device_type(inputs[0])
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            op_overload=torch.ops.mkldnn._linear_pointwise.binary,
            cpp_kernel_name=f"aoti_torch_{self.device_type}__linear_pointwise_binary",
        )

    def codegen(self, wrapper):
        wrapper.include_extra_header(
            f"torch/csrc/inductor/aoti_torch/c/shim_{self.device_type}.h"
        )
        super().codegen(wrapper)

    @classmethod
    def create(cls, x, y, w, B, attr):
        x = cls.require_contiguous(cls.realize_input(x))
        y = cls.require_contiguous(cls.realize_input(y))
        w = cls.require_contiguous(cls.realize_input(w))

        *m, _ic = x.get_size()
        oc, _ic = w.get_size()
        output_size = list(m) + [oc]
        inputs = [x, y, w]
        constant_args = [attr]
        if B is not None:
            B = cls.require_contiguous(cls.realize_input(B))
            inputs.append(B)
        else:
            constant_args.insert(0, B)

        device = x.get_device()
        if device is None:
            raise AssertionError("expected x to have a device")
        packed = LinearBinary(
            layout=FixedLayout(
                device=device,
                dtype=x.get_dtype(),
                size=output_size,
            ),
            inputs=inputs,
            constant_args=constant_args,
        )
        return _create_output_node(packed)

    def apply_constraint(self):
        pass


class QLinearPointwisePT2E(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
        has_bias=True,
    ) -> None:
        """
        if bias is not None
            - inputs = [x, w, b, weight_scale, weight_zp]
            - const_args is: [x_scale, x_zp, o_scale, o_zp,
              fp32_output, unary_attr, unary_scalars, unary_algorithm]
        else
            - inputs = [x, w, weight_scale, weight_zp]
            - const_args is: [bias, x_scale, x_zp, o_scale, o_zp,
              fp32_output, unary_attr, unary_scalars, unary_algorithm]
        """
        self.device_type = get_device_type(inputs[0])
        self.has_bias = has_bias
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            op_overload=(torch.ops.onednn.qlinear_pointwise.tensor),
            cpp_kernel_name=(
                f"aoti_torch_{self.device_type}__qlinear_pointwise_tensor"
            ),
        )

    def codegen(self, wrapper):
        wrapper.include_extra_header(
            f"torch/csrc/inductor/aoti_torch/c/shim_{self.device_type}.h"
        )
        super().codegen(wrapper)

        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    @classmethod
    def create(
        cls,
        qx: "TensorBox",
        x_scale: "TensorBox",
        x_zero_point: "TensorBox",
        qw: "TensorBox",  # packed_weight
        w_scale: "TensorBox",
        w_zero_point: "TensorBox",
        bias: "TensorBox",
        output_scale: float,
        output_zero_point: int,
        output_dtype,
        post_op_name,
        post_op_args,
        post_op_algorithm,
    ):
        (inputs, constant_args, kernel_layout, _, _) = _prepare_linear_fusion_create(
            cls,
            qx,
            qw,
            bias,
            [x_scale, x_zero_point, w_scale, w_zero_point],
        )

        constant_args = constant_args + [
            output_scale,
            output_zero_point,
            output_dtype,
            post_op_name,
            may_convert_to_optional(post_op_args),
            post_op_algorithm,
        ]

        if output_dtype is None:
            raise AssertionError("expected output_dtype to be not None")
        if output_dtype in [torch.float32, torch.bfloat16]:
            # in _prepare_linear_fusion_create, we use x.dtype (uint8) to create kernel_layout
            # if we set fp32_output, the output buf should be dtype float32 instead of uint8.
            kernel_layout.dtype = output_dtype

        return QLinearPointwisePT2E(
            layout=kernel_layout,
            inputs=inputs,
            constant_args=constant_args,
            has_bias=(bias is not None),
        )


class QLinearPointwiseBinaryPT2E(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
        has_bias=True,
    ) -> None:
        """
        if bias is not None
            - inputs = [x, w, x_scale, x_zp, weight_scale, weight_zp, x2, bias]
            - const_args is: [o_scale, o_zp,
              fp32_output, binary_attr, alpha, unary_attr, unary_scalars, unary_algorithm]
        else
            - inputs = [x, w, x_scale, x_zp, weight_scale, weight_zp, x2]
            - const_args is: [bias, o_scale, o_zp,
              fp32_output, binary_attr, alpha, unary_attr, unary_scalars, unary_algorithm]
        """
        self.device_type = get_device_type(inputs[0])
        self.has_bias = has_bias
        self.idx_for_inplace_sum = 6
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            op_overload=(torch.ops.onednn.qlinear_pointwise.binary_tensor),
            cpp_kernel_name=f"aoti_torch_{self.device_type}__qlinear_pointwise_binary_tensor",
        )

    def codegen(self, wrapper):
        wrapper.include_extra_header(
            f"torch/csrc/inductor/aoti_torch/c/shim_{self.device_type}.h"
        )
        super().codegen(wrapper)
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    def get_mutation_names(self) -> Sequence[str]:
        binary_post_op = self.constant_args[-5]
        if binary_post_op == "sum":
            input = self.inputs[self.idx_for_inplace_sum]
            if not isinstance(input, IRNode):
                raise AssertionError(f"expected input to be IRNode, got {type(input)}")
            return [input.get_name()]
        else:
            return []

    @classmethod
    def create(
        cls,
        qx: "TensorBox",
        x_scale: "TensorBox",
        x_zero_point: "TensorBox",
        qw: "TensorBox",  # packed_weight
        w_scale: "TensorBox",
        w_zero_point: "TensorBox",
        other: "TensorBox",
        bias: "TensorBox",
        output_scale: float,
        output_zero_point: int,
        output_dtype,
        other_scale,
        other_zp,
        binary_post_op,
        binary_alpha,
        unary_post_op,
        unary_post_op_args,
        unary_post_op_algorithm,
    ):
        (
            inputs,
            constant_args,
            kernel_layout,
            req_stride_order,
            other,
        ) = _prepare_linear_fusion_create(
            cls,
            qx,
            qw,
            bias,
            [x_scale, x_zero_point, w_scale, w_zero_point],
            other,
            binary_post_op == "sum",
        )

        constant_args = constant_args + [
            output_scale,
            output_zero_point,
            output_dtype,
            other_scale,
            other_zp,
            binary_post_op,
            binary_alpha,
            unary_post_op,
            may_convert_to_optional(unary_post_op_args),
            unary_post_op_algorithm,
        ]

        if binary_post_op == "sum":
            V.graph.mark_buffer_mutated(other.get_name())
            packed = QLinearPointwiseBinaryPT2E(
                layout=NoneLayout(device=other.get_device()),
                inputs=inputs,
                constant_args=constant_args,
                has_bias=(bias is not None),
            )
            # Return other since it has been inplace changed.
            return packed.inputs[packed.idx_for_inplace_sum]

        if output_dtype is None:
            raise AssertionError("expected output_dtype to be not None")
        if output_dtype in [torch.float32, torch.bfloat16]:
            # in _prepare_linear_fusion_create, we use x.dtype (uint8) to create kernel_layout
            # if we set fp32_output, the output buf should be dtype float32 instead of uint8.
            kernel_layout.dtype = output_dtype

        return QLinearPointwiseBinaryPT2E(
            layout=kernel_layout,
            inputs=inputs,
            constant_args=constant_args,
            has_bias=(bias is not None),
        )


class MkldnnRnnLayer(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
    ) -> None:
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            op_overload=torch.ops.aten.mkldnn_rnn_layer.default,
        )

    @classmethod
    def create(
        cls,
        x: "TensorBox",
        w0: "TensorBox",
        w1: "TensorBox",
        w2: "TensorBox",
        w3: "TensorBox",
        hx: "TensorBox",
        cx: "TensorBox",
        reverse: bool,
        batch_sizes: list[int],
        mode: int,
        hidden_size: int,
        num_layers: int,
        has_biases: bool,
        bidirectional: bool,
        batch_first: bool,
        train: bool,
    ):
        # pyrefly: ignore [bad-assignment]
        x = cls.require_stride1(cls.realize_input(x))
        # If batch_first, x has been permuted in lstm before entering the mkldnn_rnn_layer.
        # Make sure x is contiguous in batch_first case.
        x.freeze_layout()
        # pyrefly: ignore [bad-assignment]
        w0 = cls.require_stride1(cls.realize_input(w0))
        # pyrefly: ignore [bad-assignment]
        w1 = cls.require_stride1(cls.realize_input(w1))
        # pyrefly: ignore [bad-assignment]
        w2 = cls.require_stride1(cls.realize_input(w2))
        # pyrefly: ignore [bad-assignment]
        w3 = cls.require_stride1(cls.realize_input(w3))
        # pyrefly: ignore [bad-assignment]
        hx = cls.require_stride1(cls.realize_input(hx))
        hx.freeze_layout()
        # pyrefly: ignore [bad-assignment]
        cx = cls.require_stride1(cls.realize_input(cx))
        cx.freeze_layout()

        input_size = x.get_size()
        if len(input_size) != 3:
            raise AssertionError("Expect lstm input to be 3D")
        # batch_first is handled in the lstm OP. When entering
        # rnn_layer here, we'll always have batch_first = False
        seq_length, mini_batch, input_size = input_size
        output_shape = [seq_length, mini_batch, hidden_size]

        hy_shape = hx.get_size()
        cy_shape = cx.get_size()

        inputs = [x, w0, w1, w2, w3, hx, cx]
        constant_args = [
            reverse,
            batch_sizes,
            mode,
            hidden_size,
            num_layers,
            has_biases,
            bidirectional,
            batch_first,
            train,
        ]

        device = x.get_device()
        if device is None:
            raise AssertionError("expected x to have a device")
        packed = MkldnnRnnLayer(
            MultiOutputLayout(device=device),
            inputs=inputs,
            constant_args=constant_args,
        )

        def get_strides_of_lstm_output(output_shape, batch_first):
            if len(output_shape) != 3:
                raise AssertionError("Expect output_shape to be 3D")
            return FlexibleLayout.contiguous_strides(output_shape)

        # C shim call requires all the outputs to be passed in, and thus the last
        # dummy return value is added.
        output_sizes = [output_shape, hy_shape, cy_shape, [1]]
        output_strides = [
            get_strides_of_lstm_output(output_shape, batch_first),
            FlexibleLayout.contiguous_strides(hy_shape),
            FlexibleLayout.contiguous_strides(cy_shape),
            [1],
        ]
        output_ir = [
            MultiOutput(
                FixedLayout(
                    x.get_device(),  # type: ignore[arg-type]
                    x.get_dtype(),
                    output_size,
                    output_stride,
                ),
                packed,
                [(tuple, i)],
            )
            for i, (output_size, output_stride) in enumerate(
                zip(output_sizes, output_strides)
            )
        ]
        packed.outputs = output_ir

        return output_ir

    def codegen(self, wrapper):
        wrapper.include_extra_header("torch/csrc/inductor/aoti_torch/c/shim_cpu.h")
        return super().codegen(wrapper)


# Add this IR so that we can include shim_cpu.h for cpp_wrapper
class WeightInt4PackMatmul(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
    ) -> None:
        """
        inputs = [x, w, qGroupSize, qScalesAndZeros]
        constant_args = ()
        """
        if len(inputs) != 4:
            raise AssertionError(f"expected 4 inputs, got {len(inputs)}")
        if len(constant_args) != 0:
            raise AssertionError(f"expected 0 constant_args, got {len(constant_args)}")
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            op_overload=(torch.ops.quantized.int4mm_packed_weight_cpu.default),
            cpp_kernel_name=("aoti_torch_cpu__weight_int4pack_mm_cpu_tensor"),
        )

    def codegen(self, wrapper):
        wrapper.include_extra_header("torch/csrc/inductor/aoti_torch/c/shim_cpu.h")
        super().codegen(wrapper)

        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    @classmethod
    def create(
        cls,
        x: "TensorBox",
        w: "TensorBox",
        qGroupSize: "TensorBox",
        qScalesAndZeros: "TensorBox",
    ):
        inputs = [x, w, qGroupSize, qScalesAndZeros]
        *m, _ = x.get_size()
        n, _ = w.get_size()
        output_size = list(m) + [n]
        output_stride = FlexibleLayout.contiguous_strides(output_size)
        kernel_layout = FixedLayout(
            x.get_device(),  # type: ignore[arg-type]
            x.get_dtype(),
            output_size,
            output_stride,
        )
        return WeightInt4PackMatmul(
            layout=kernel_layout,
            inputs=inputs,
        )
<|endoftext|>import atexit
import collections
import dataclasses
import functools
import os
import shutil
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field

from torch._inductor.runtime.cache_dir_utils import cache_dir


# Set the subdirectory name
SUBDIR_NAME = "bisect"


@dataclass
class Subsystem:
    name: str


@dataclass
class BisectSubsystem(Subsystem):
    pass


@dataclass
class BinarySubsystem(Subsystem):
    pass


@dataclass
class ConfigChange(BinarySubsystem):
    name: str = field(init=False)
    config_name: str
    config_field: str
    config_value: object

    def __post_init__(self) -> None:
        self.name = f"{self.config_name}_{self.config_field}"


# Dictionary of backend -> subsystems
BACKENDS: dict[str, list[Subsystem]] = {
    # run dynamo without aot_autograd
    "eager": [],
    # run dynamo with aot_autograd, but no partitioner or decomps
    "aot_eager": [],
    # run dynamo with aot autograd, decompositions and partitioner
    "aot_eager_decomp_partition": [
        ConfigChange("aot_eager_decomp_partition", "cse", False),
        BisectSubsystem(
            "decomposition"
        ),  # number of decompositions we apply in tracing
    ],  # TODO - add cse ?
    # applies CrossRefFakeMode on invocation
    "aot_eager_decomp_partition_crossref": [],
    "inductor": [
        BisectSubsystem("pre_grad_passes"),  # passes applied on pre-grad IR
        BisectSubsystem("joint_graph_passes"),  # passes applied on joint graph
        BisectSubsystem(
            "post_grad_passes"
        ),  # passes applied individually on forward, and backward in inductor
        ConfigChange("inductor", "fallback_random", True),
        ConfigChange("inductor", "emulate_precision_casts", True),
        ConfigChange("inductor", "layout_optimization", False),
        ConfigChange("inductor", "comprehensive_padding", False),
        BisectSubsystem("cudagraphs"),  # cudagraph wrapping of compiled graphs
        BisectSubsystem("lowerings"),  # lowering aten operators to inductor
    ],  # TODO - add more - fusions ?
}

subsystem_call_counter: dict[str, int] = collections.Counter()


@functools.cache
def get_env_val(env_str: str) -> str | None:
    return os.environ.get(env_str, None)


@dataclasses.dataclass
class BisectionResult:
    """
    backend: torch.compile backend responsible for failure
    subsystem: optional, registered component identified for failure
    bisect_number: optional, number of times the subsystem needed to be applied to trigger failure
    debug_info: associated info of the triggering bisect application of subsystem
    """

    backend: str
    subsystem: str | None = None
    bisect_number: int | None = None
    debug_info: str | None = None


class CompilerBisector:
    """
    This class iteratively runs torch.compile backends (eager, aot_eager, inductor) to find the
    first backend that can repro an issue.

    Once it discovers the offending backend it will iteratively disable subsystems within the backend.
    For subsystems which are applied repeatedly, such as the number of post grad passes or number
    of lowering of nodes to inductor ir, it will bisect to find the offending application.

    The idiomatic way to run it is with `do_bisect`. You can also use it by setting the env flags
    `TORCH_BISECT_BACKEND`, `TORCH_BISECT_SUBSYSTEM` and `TORCH_BISECT_MAX`.

    It also supports a CLI interface:

        python -m torch._inductor.compiler_bisector start
        python -m torch._inductor.compiler_bisector good
        python -m torch._inductor.compiler_bisector bad
        python -m torch._inductor.compiler_bisector end

    Or use `run` to automatically bisect by running a command repeatedly:

        python -m torch._inductor.compiler_bisector run <command>

    The command's exit code determines the result: 0 = good, non-zero = bad.
    The TORCH_COMPILE_BACKEND env var is set to the backend being tested.
    """

    bisection_enabled: bool = False

    in_process_cache: str | None = None

    @classmethod
    def clear_call_counter_debug_info(
        cls, backend_name: str, subsystem_name: str
    ) -> None:
        file_path = os.path.join(
            cls.get_dir(),
            backend_name,
            f"{subsystem_name}_call_counter_debug_info.txt",
        )
        cls.write_lines_to_file(file_path, [])

    @classmethod
    def reset_counters(cls) -> None:
        subsystem_call_counter.clear()
        for backend, subsystem_list in BACKENDS.items():
            for subsystem in subsystem_list:
                if isinstance(subsystem, BisectSubsystem):
                    cls.clear_call_counter_debug_info(backend, subsystem.name)

    @classmethod
    def get_dir(cls) -> str:
        return f"{cache_dir() if not cls.in_process_cache else cls.in_process_cache}/{SUBDIR_NAME}"

    @classmethod
    def write_lines_to_file(
        cls, file_path: str, lines: list[str], *, append: bool = False
    ) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "a" if append else "w") as file:
            file.writelines(lines)

    @classmethod
    def read_lines_from_file(cls, file_path: str) -> list[str]:
        if os.path.exists(file_path):
            with open(file_path) as file:
                return file.readlines()
        return []

    @classmethod
    def update_run_state(
        cls, backend_name: str, subsystem: Subsystem, run_state: str
    ) -> None:
        file_path = os.path.join(
            cls.get_dir(), backend_name, f"{subsystem.name}_run_state.txt"
        )
        if isinstance(subsystem, ConfigChange):
            if run_state != "test_disable":
                raise AssertionError(
                    f"expected run_state == 'test_disable', got {run_state}"
                )
            cls.set_config_values(
                backend_name,
                subsystem.name,
                {subsystem.config_field: subsystem.config_value},
            )

        cls.write_lines_to_file(file_path, [run_state])

    @classmethod
    def set_config_values(
        cls, backend: str, subsystem: str, config_data: dict[str, object]
    ) -> None:
        file_path = os.path.join(cls.get_dir(), backend, f"{subsystem}_config.txt")
        lines = [f"{k}={v}\n" for k, v in config_data.items()]
        cls.write_lines_to_file(file_path, lines)

    @classmethod
    def update_bisect_status(cls, backend_name: str, subsystem_name: str) -> None:
        if not isinstance(subsystem_name, str):
            raise AssertionError(
                f"expected subsystem_name to be str, got {type(subsystem_name)}"
            )
        file_path = os.path.join(cls.get_dir(), "bisect_status.txt")
        lines = [f"backend={backend_name}\n", f"subsystem={subsystem_name}\n"]
        cls.write_lines_to_file(file_path, lines)

    @classmethod
    def update_bisect_range(
        cls, backend_name: str, subsystem_name: str, low: int, high: int
    ) -> None:
        if not isinstance(subsystem_name, str):
            raise AssertionError(
                f"expected subsystem_name to be str, got {type(subsystem_name)}"
            )
        file_path = os.path.join(
            cls.get_dir(), backend_name, f"{subsystem_name}_bisect_range.txt"
        )
        lines = [f"low={low}\n", f"high={high}\n"]
        cls.write_lines_to_file(file_path, lines)

    @classmethod
    def add_call_counter_debug_info(
        cls, backend_name: str, subsystem_name: str, call_counter: int, debug_info: str
    ) -> None:
        file_path = os.path.join(
            cls.get_dir(), backend_name, f"{subsystem_name}_call_counter_debug_info.txt"
        )
        # This is only called for the last ~2 items in a bisect, so writing duplicate
        # lines does not have performance implications later.
        lines = [f"{call_counter}={debug_info}\n"]
        cls.write_lines_to_file(file_path, lines, append=True)

    @classmethod
    def get_call_counter_debug_info(
        cls, backend_name: str, subsystem_name: str, call_counter: int
    ) -> str | None:
        file_path = os.path.join(
            cls.get_dir(), backend_name, f"{subsystem_name}_call_counter_debug_info.txt"
        )
        call_counter_str = str(call_counter)
        for line in reversed(cls.read_lines_from_file(file_path)):
            counter, _, debug_info = line.strip().partition("=")
            if counter == call_counter_str:
                return debug_info
        return None

    @classmethod
    def get_backend(cls) -> str | None:
        """
        Returns the active backend, if any
        """
        if val := get_env_val("TORCH_BISECT_BACKEND"):
            return val

        file_path = os.path.join(cls.get_dir(), "bisect_status.txt")
        lines = cls.read_lines_from_file(file_path)
        for line in lines:
            if line.startswith("backend="):
                return line.strip().split("=")[1]
        return None

    @classmethod
    def get_subsystem(cls) -> str | None:
        """
        Returns the active subsystem, if any
        """

        if val := get_env_val("TORCH_BISECT_SUBSYSTEM"):
            return val

        file_path = os.path.join(cls.get_dir(), "bisect_status.txt")
        lines = cls.read_lines_from_file(file_path)
        for line in lines:
            if line.startswith("subsystem="):
                out = line.strip().split("=")[1]
                return out or None
        return None

    @classmethod
    def get_subsystem_object(cls, backend_name: str, subsystem_name: str) -> Subsystem:
        return next(obj for obj in BACKENDS[backend_name] if obj.name == subsystem_name)

    @classmethod
    def get_run_state(cls, backend_name: str, subsystem_name: str) -> str | None:
        """
        Returns the current stage of bisecting, if Any
        """

        file_path = os.path.join(
            cls.get_dir(), backend_name, f"{subsystem_name}_run_state.txt"
        )
        lines = cls.read_lines_from_file(file_path)
        if lines:
            out = lines[0].strip()
            if out not in ("test_disable", "find_max_bounds", "bisect"):
                raise AssertionError(
                    f"unexpected run_state {out!r}, expected one of "
                    "'test_disable', 'find_max_bounds', 'bisect'"
                )
            return out
        return None

    @classmethod
    def get_bisect_range(
        cls, backend_name: str, subsystem_name: str
    ) -> tuple[int, int]:
        file_path = os.path.join(
            cls.get_dir(), backend_name, f"{subsystem_name}_bisect_range.txt"
        )
        lines = cls.read_lines_from_file(file_path)
        low = None
        high = None
        # pyrefly: ignore [bad-assignment]
        for line in reversed(lines):
            if line.startswith("low="):
                low = int(line.strip().split("=")[1])
            elif line.startswith("high="):
                high = int(line.strip().split("=")[1])

            if low is not None and high is not None:
                break

        if low is None or high is None:
            raise RuntimeError(
                f"Trying to get bisect range when it is not set: subsystem {subsystem_name}"
            )

        return low, high

    @classmethod
    def update_config_change(cls, backend: str, subsystem: ConfigChange) -> None:
        file_path = os.path.join(cls.get_dir(), backend, f"{subsystem.name}_config.txt")
        lines = [
            f"config_name={subsystem.config_name}\n",
            f"config_field={subsystem.config_field}\n",
            f"config_value={subsystem.config_value}\n",
        ]
        cls.write_lines_to_file(file_path, lines)

    @classmethod
    def get_config_change(cls, config_name: str) -> dict[str, object] | None:
        backend = cls.get_backend()
        subsystem = cls.get_subsystem()

        if not backend or not subsystem:
            return None

        file_path = os.path.join(cls.get_dir(), backend, f"{subsystem}_config.txt")

        if not os.path.exists(file_path):
            return None

        lines = cls.read_lines_from_file(file_path)
        config_data = {}
        for line in lines:
            key, value = line.strip().split("=", 1)
            config_data[key] = eval(value)

        return config_data

    @classmethod
    def delete_bisect_status(cls) -> None:
        # in process_cache we have created if it exists, just the subdirectory of non created dir
        dir_name = cls.in_process_cache or cls.get_dir()
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print("Bisection status deleted.")
        else:
            print("No bisection status found.")

    @classmethod
    def get_system_counter(cls, name: str, increment: bool = True) -> int:
        global subsystem_call_counter
        curr = subsystem_call_counter[name]
        if increment:
            subsystem_call_counter[name] += 1
        return curr

    @classmethod
    def disable_subsystem(
        cls,
        backend: str,
        subsystem: str,
        debug_info: Callable[[], str] | None = None,
    ) -> bool:
        if not cls.bisection_enabled:
            return False

        if cls.get_backend() != backend:
            return False

        if cls.get_subsystem() != subsystem:
            return False

        if val := get_env_val("TORCH_BISECT_MAX"):
            counter = cls.get_system_counter(subsystem, increment=True)
            return counter > int(val)

        run_state = cls.get_run_state(backend, subsystem)
        if run_state == "test_disable":
            # First run, disable completely
            return True
        elif run_state == "find_max_bounds":
            # Second run, update bisection range and return True to enable the subsystem
            cls.update_bisect_range(
                backend,
                subsystem,
                0,
                cls.get_system_counter(subsystem, increment=True),
            )
            return False
        else:
            if run_state != "bisect":
                raise AssertionError(f"expected run_state == 'bisect', got {run_state}")
            # If the environment variable is not set, use the bisection range midpoint
            low, high = cls.get_bisect_range(backend, subsystem)
            # if high - low <= 2:
            midpoint = (low + high) // 2
            call_counter = cls.get_system_counter(subsystem)

            if (
                call_counter >= low
                and call_counter <= high
                and (high - low) <= 2
                and debug_info is not None
            ):
                cls.add_call_counter_debug_info(
                    backend, subsystem, call_counter, debug_info()
                )

            return call_counter > midpoint

    @classmethod
    def advance_subsystem(
        cls, curr_backend: str, curr_subsystem: Subsystem
    ) -> Subsystem | None:
        """
        Tries to move to the next subsystem within the current system.
        """
        print(f"Disabling {curr_subsystem.name} did not fix the issue.")

        current_subsystems = BACKENDS[curr_backend]
        current_subsystem_index = next(
            i
            for i, subsystem in enumerate(current_subsystems)
            if subsystem.name == curr_subsystem.name
        )

        if current_subsystem_index < len(current_subsystems) - 1:
            next_subsystem = current_subsystems[current_subsystem_index + 1]
            cls.update_bisect_status(curr_backend, next_subsystem.name)
            cls.update_run_state(curr_backend, next_subsystem, "test_disable")
            print(
                f"Moving to the next subsystem: {curr_backend} - {next_subsystem.name}"
            )
            return next_subsystem
        else:
            print(
                f"All subsystems in {curr_backend} have been checked. The issue is not in this system."
            )
            return None

    @classmethod
    def advance_backend(cls, curr_backend: str) -> str | None:
        """
        Tries Move to the next backend.
        """
        current_system_index = list(BACKENDS.keys()).index(curr_backend)

        if current_system_index < len(BACKENDS) - 1:
            curr_backend = list(BACKENDS.keys())[current_system_index + 1]
            cls.update_bisect_status(curr_backend, "")
            print(f"Moving to the next system: {curr_backend}")
            return curr_backend
        else:
            return None

    @classmethod
    def process_subsystem(
        cls,
        curr_backend: str,
        curr_subsystem: Subsystem,
        fn: Callable[[], bool],
        cli_interface: bool = True,
    ) -> bool:
        """
        Process the current subsystem. Returns True if the issue is found, False otherwise.
        """
        if not isinstance(curr_subsystem, Subsystem):
            raise AssertionError(
                f"expected curr_subsystem to be Subsystem, got {type(curr_subsystem)}"
            )
        while True:
            run_state = cls.get_run_state(curr_backend, curr_subsystem.name)
            cls.reset_counters()
            if run_state == "test_disable":
                if not fn():
                    next_subsystem = cls.advance_subsystem(curr_backend, curr_subsystem)
                    if not next_subsystem:
                        return False
                    curr_subsystem = next_subsystem
                else:
                    if isinstance(curr_subsystem, ConfigChange):
                        print(
                            f"Setting config {curr_subsystem.config_name} field {curr_subsystem.config_field} "
                            f"to {curr_subsystem.config_value} fixed the issue"
                        )
                    else:
                        print(f"Disabling {curr_subsystem.name} fixed the issue.")
                    if isinstance(curr_subsystem, BinarySubsystem):
                        return True
                    print("Starting bisect by getting upper bound.")
                    cls.update_run_state(
                        curr_backend, curr_subsystem, "find_max_bounds"
                    )
            elif run_state == "find_max_bounds":
                if fn():
                    raise RuntimeError(
                        f"Function succeeded with 'find_max_bounds' status for {curr_backend} - {curr_subsystem.name}."
                    )
                else:
                    _, high = cls.get_bisect_range(curr_backend, curr_subsystem.name)
                    print(f"Upper bound of {high} found for {curr_backend}.")
                    cls.update_run_state(curr_backend, curr_subsystem, "bisect")
            elif run_state == "bisect":
                low, high = cls.get_bisect_range(curr_backend, curr_subsystem.name)
                midpoint = (low + high) // 2
                print(
                    f"Bisecting {curr_backend} - {curr_subsystem.name} (Range: [{low}, {high}], Midpoint: {midpoint})"
                )
                if fn():
                    cls.update_bisect_range(
                        curr_backend, curr_subsystem.name, midpoint + 1, high
                    )
                else:
                    cls.update_bisect_range(
                        curr_backend, curr_subsystem.name, low, midpoint
                    )
                low, high = cls.get_bisect_range(curr_backend, curr_subsystem.name)
                if low == high:
                    print(
                        f"Binary search completed for {curr_backend} - {curr_subsystem.name}. The bisect number is {low}. "
                        f"Debug info: {cls.get_call_counter_debug_info(curr_backend, curr_subsystem.name, low) or 'not found'}"
                    )
                    return True
            else:
                raise RuntimeError(f"Unexpected run_state {run_state}")

            if cli_interface:
                sys.exit(0)

    @classmethod
    def initialize_system(cls) -> None:
        curr_backend = next(iter(BACKENDS.keys()))
        curr_subsystem = ""
        cls.update_bisect_status(curr_backend, curr_subsystem)
        print(f"Starting bisection process with system: {curr_backend}")

    @classmethod
    def do_bisect(
        cls, fn: Callable[[], bool], cli_interface: bool = False
    ) -> BisectionResult | None:
        """
        Run fn repeatedly attempting to bisect torch.compile. fn should return True on success and False on failure.
        """

        # TODO graph bisecting is not well composed with lowering
        # bisector so far. Use a config to opt-in
        import torch._inductor.config as inductor_config

        if inductor_config.test_configs.bisect_pre_grad_graph:
            BACKENDS["inductor"].insert(0, BisectSubsystem("pre_grad_graph"))

        if not cli_interface:
            bisection_enabled_orig = cls.bisection_enabled
            cls.delete_bisect_status()
            cls.bisection_enabled = True
            # Only create temp dir if not already set (CLI run mode pre-sets it)
            if not cls.in_process_cache:
                cls.in_process_cache = tempfile.mkdtemp()

            def cleanup() -> None:
                cls.bisection_enabled = bisection_enabled_orig
                cls.delete_bisect_status()
                cls.in_process_cache = None

                if BACKENDS["inductor"][0].name == "pre_grad_graph":
                    del BACKENDS["inductor"][0]

            cleanup_handler = atexit.register(cleanup)

            class DisableBisect:
                def __del__(self) -> None:
                    cleanup()
                    atexit.unregister(cleanup_handler)

            _cleanup = DisableBisect()

        curr_backend = cls.get_backend()
        curr_subsystem_name = cls.get_subsystem()

        if not curr_backend:
            cls.initialize_system()
            curr_backend = cls.get_backend()
            if curr_backend is None:
                raise AssertionError(
                    "expected curr_backend to be set after initialize_system"
                )
            curr_subsystem_name = cls.get_subsystem()

        curr_subsystem = (
            cls.get_subsystem_object(curr_backend, curr_subsystem_name)
            if curr_subsystem_name is not None
            else None
        )
        while True:
            if curr_backend is None:
                raise AssertionError("expected curr_backend to be set")
            cls.reset_counters()
            if curr_subsystem:
                result = cls.process_subsystem(
                    curr_backend, curr_subsystem, fn, cli_interface=cli_interface
                )
                if result:
                    curr_subsystem = cls.get_subsystem_object(
                        curr_backend,
                        cls.get_subsystem(),  # type: ignore[arg-type]
                    )

                    if isinstance(curr_subsystem, BinarySubsystem):
                        return BisectionResult(
                            curr_backend,
                            curr_subsystem.name,
                            0,
                            curr_subsystem.name,
                        )

                    low, _ = cls.get_bisect_range(curr_backend, curr_subsystem.name)
                    return BisectionResult(
                        curr_backend,
                        curr_subsystem.name,
                        low,
                        cls.get_call_counter_debug_info(
                            curr_backend, curr_subsystem.name, low
                        ),
                    )

                next_subsystem = cls.advance_subsystem(curr_backend, curr_subsystem)
                if not next_subsystem:
                    print(
                        f"The issue is in the {curr_backend} system, but could not identify subsystem."
                    )
                    if curr_backend is None:
                        raise AssertionError("expected curr_backend to be set")
                    return BisectionResult(curr_backend)

                curr_subsystem = next_subsystem
            else:
                if fn():
                    next_backend = cls.advance_backend(curr_backend)
                    if not next_backend:
                        print("All systems have been checked.")
                        return None

                    curr_backend = next_backend
                else:
                    current_subsystems = BACKENDS[curr_backend]
                    if current_subsystems:
                        curr_subsystem = current_subsystems[0]
                        cls.update_bisect_status(curr_backend, curr_subsystem.name)
                        cls.update_run_state(
                            curr_backend, curr_subsystem, "test_disable"
                        )
                        print(
                            f"The issue is in the {curr_backend} system. Moving to the first subsystem: {curr_subsystem}"
                        )
                    else:
                        print(f"The issue is in the {curr_backend} system.")
                        return BisectionResult(curr_backend)

            if cli_interface:
                sys.exit(0)


HELP_TEXT = """\
Usage: python -m torch._inductor.compiler_bisector <command>

Commands:
  start       Start a new bisection session (manual mode)
  good        Mark the current state as good (no issue)
  bad         Mark the current state as bad (issue present)
  end         End the bisection session and clean up
  run <cmd>   Automatically bisect by running a command repeatedly
              Exit code 0 = good, non-zero = bad

Example - using 'run' for automatic bisection:

  1. Create a test script (e.g., test_bug.py):

      import os, sys, torch

      def main():
          torch._dynamo.reset()
          # Use the backend the bisector is testing
          backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")

          @torch.compile(backend=backend)
          def fn(x):
              return x.sin()  # or your failing operation

          x = torch.randn(10)
          compiled = fn(x)
          expected = x.sin()  # eager reference

          if torch.allclose(compiled, expected, rtol=1e-4, atol=1e-4):
              return 0  # PASS
          else:
              print(f"FAIL: got {compiled}, expected {expected}")
              return 1  # FAIL

      if __name__ == "__main__":
          sys.exit(main())

  2. Run the bisector:

      python -m torch._inductor.compiler_bisector run python test_bug.py

  3. The bisector will:
      - Test backends (eager, aot_eager, inductor) to find which fails
      - Test subsystems within that backend (passes, lowerings, etc.)
      - Binary search to find the exact operation causing the issue

  Output:
      Disabling lowerings fixed the issue.
      Binary search completed for inductor - lowerings. The bisect number is 42.
      Bisection complete: BisectionResult(backend='inductor', subsystem='lowerings', ...)

Environment variables set for your test script:
  TORCH_COMPILE_BACKEND  - The backend being tested (use this in torch.compile)
"""


def command_line_usage() -> None:
    """Entry point for the compiler bisector command-line interface."""
    if len(sys.argv) < 2:
        print(HELP_TEXT)
        sys.exit(1)

    bisection_manager = CompilerBisector()
    command = sys.argv[1]

    if command == "end":
        bisection_manager.delete_bisect_status()
        sys.exit(0)

    if command == "start":
        bisection_manager.delete_bisect_status()
        bisection_manager.initialize_system()
        sys.exit(0)

    if command == "run":
        if len(sys.argv) < 3:
            print(
                "Usage: python -m torch._inductor.compiler_bisector run <command> [args...]"
            )
            sys.exit(1)

        import subprocess

        run_cmd = sys.argv[2:]

        def test_function() -> bool:
            # Pass bisection state to subprocess via environment variables
            env = os.environ.copy()
            backend = bisection_manager.get_backend()
            subsystem = bisection_manager.get_subsystem()

            if backend:
                # For test script to select the right backend
                env["TORCH_COMPILE_BACKEND"] = backend
                # For bisector in subprocess to know which backend we're testing
                env["TORCH_BISECT_BACKEND"] = backend

            if subsystem:
                if backend is None:  # subsystem requires a backend
                    raise AssertionError("subsystem requires a backend")
                env["TORCH_BISECT_SUBSYSTEM"] = subsystem
                # Get run_state to determine TORCH_BISECT_MAX
                run_state = bisection_manager.get_run_state(backend, subsystem)
                if run_state == "test_disable":
                    # -1 means always disable (counter > -1 is always True)
                    env["TORCH_BISECT_MAX"] = "-1"
                # For find_max_bounds and bisect, let the subprocess use file-based
                # mechanisms. The subprocess reads run_state and bisect_range from
                # files, and writes the actual count during find_max_bounds.

            result = subprocess.run(run_cmd, env=env)
            return result.returncode == 0

        bisection_manager.delete_bisect_status()
        bisection_manager.bisection_enabled = True
        # Use shared cache_dir instead of temp dir so subprocesses can access files
        CompilerBisector.in_process_cache = cache_dir()
        result = bisection_manager.do_bisect(test_function, cli_interface=False)
        if result:
            print(f"\nBisection complete: {result}")
        else:
            print("\nBisection complete: no issue found")
        sys.exit(0)

    if command not in ["good", "bad"]:
        print(f"Invalid command: {command}")
        print("Must be 'good', 'bad', 'start', 'end', or 'run'.")
        sys.exit(1)

    def test_function() -> bool:
        return command == "good"

    if not bisection_manager.get_backend():
        raise ValueError("Must call start prior to good or bad")

    bisection_manager.do_bisect(test_function, cli_interface=True)


def get_is_bisection_enabled() -> bool:
    return (
        CompilerBisector.get_subsystem() is not None
        or CompilerBisector.get_backend() is not None
    )


CompilerBisector.bisection_enabled = get_is_bisection_enabled()

if __name__ == "__main__":
    command_line_usage()
<|endoftext|>import functools
import logging
import math
import operator
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import sympy

import torch
import torch.utils._pytree as pytree
from torch.fx.experimental.symbolic_shapes import optimization_hint
from torch.fx.operator_schemas import normalize_function

from . import ir
from .utils import get_dtype_size, snode_args_kwargs, sympy_product
from .virtualized import V


log = logging.getLogger(__name__)


class NCCL_COLL(IntEnum):
    ALL_REDUCE = 0
    ALL_GATHER = 1
    REDUCE_SCATTER = 2
    ALL_TO_ALL = 3
    UNSUPPORTED = 4
    P2P = 5


class NVIDIA_GPU_TYPE(IntEnum):
    VOLTA = 0
    AMPERE = 1
    HOPPER = 2
    BLACKWELL = 3


class InterconnectType(IntEnum):
    NVLINK_V100 = 0
    NVLINK_A100 = 1
    NVLINK_H100 = 2
    NVLINK_B200 = 3
    IB_HDR = 4  # 200 Gbps InfiniBand
    IB_NDR = 5  # 400 Gbps InfiniBand / RoCE
    PCIE = 6  # PCIe (no NVLink)


@dataclass(frozen=True)
class InterconnectProfile:
    """Empirically calibrated interconnect parameters for saturation estimation."""

    bus_bw_GBps: float  # effective bus bandwidth at large messages (GB/s)
    latency_per_hop_us: float  # per-ring-step latency (us)
    base_latency_us: float  # fixed startup latency (us)
    min_saturation_bytes: int = 0  # floor for small group sizes


_MB = 1024 * 1024

# Empirically calibrated per-interconnect profiles for saturation estimation.
# NVLink bus_bw from https://github.com/NVIDIA/nccl/blob/master/src/graph/topo.h
# IB hop_lat/base_lat from MAST sweep profiles on H100 GrandTeton RoCE
# (see agent_space/nccl_model_validation.md for calibration data).
# IB min_saturation_bytes floors the estimate for small group sizes where the
# ring-step formula underestimates.
INTERCONNECT_PROFILES: dict[InterconnectType, InterconnectProfile] = {
    InterconnectType.NVLINK_H100: InterconnectProfile(36.0, 40.0, 10.0),
    InterconnectType.NVLINK_A100: InterconnectProfile(22.0, 45.0, 10.0),
    InterconnectType.NVLINK_B200: InterconnectProfile(60.0, 30.0, 10.0),
    InterconnectType.NVLINK_V100: InterconnectProfile(14.0, 50.0, 10.0),
    InterconnectType.IB_NDR: InterconnectProfile(
        22.0, 12.0, 30.0, min_saturation_bytes=100 * _MB
    ),
    InterconnectType.IB_HDR: InterconnectProfile(
        11.0, 14.0, 30.0, min_saturation_bytes=75 * _MB
    ),
    # PCIe: ~10 GB/s effective NCCL bus BW (GPU->CPU->GPU ring steps)
    InterconnectType.PCIE: InterconnectProfile(10.0, 60.0, 15.0),
}

_GPU_TO_INTRA: dict[NVIDIA_GPU_TYPE, InterconnectType] = {
    NVIDIA_GPU_TYPE.VOLTA: InterconnectType.NVLINK_V100,
    NVIDIA_GPU_TYPE.AMPERE: InterconnectType.NVLINK_A100,
    NVIDIA_GPU_TYPE.HOPPER: InterconnectType.NVLINK_H100,
    NVIDIA_GPU_TYPE.BLACKWELL: InterconnectType.NVLINK_B200,
}
_GPU_TO_INTER: dict[NVIDIA_GPU_TYPE, InterconnectType] = {
    NVIDIA_GPU_TYPE.VOLTA: InterconnectType.IB_HDR,
    NVIDIA_GPU_TYPE.AMPERE: InterconnectType.IB_HDR,
    NVIDIA_GPU_TYPE.HOPPER: InterconnectType.IB_NDR,
    NVIDIA_GPU_TYPE.BLACKWELL: InterconnectType.IB_NDR,
}


@functools.lru_cache
def _has_nvlink() -> bool:
    """Detect NVLink via nvidia-smi topology, falling back to peer access check."""
    import subprocess

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return True  # Single GPU: interconnect irrelevant
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Check GPU-GPU section (before Legend) for "NV" links
            topo = result.stdout.split("Legend")[0]
            return "NV" in topo
    except (subprocess.TimeoutExpired, OSError):
        pass
    # Fallback: peer access generally implies NVLink on datacenter GPUs
    try:
        return torch.cuda.can_device_access_peer(0, 1)
    except (AssertionError, RuntimeError):
        return True


@functools.lru_cache
def get_gpu_type() -> NVIDIA_GPU_TYPE:
    # Prefer compute capability (works for all NVIDIA GPUs, including H200, L40, etc.)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        major = torch.cuda.get_device_properties(0).major
        if major >= 10:
            return NVIDIA_GPU_TYPE.BLACKWELL
        if major >= 9:
            return NVIDIA_GPU_TYPE.HOPPER
        if major >= 8:
            return NVIDIA_GPU_TYPE.AMPERE
        if major >= 7:
            return NVIDIA_GPU_TYPE.VOLTA
    # Fallback: nvidia-smi string matching
    gpu_info = torch.utils.collect_env.get_gpu_info(torch.utils.collect_env.run) or ""
    if "V100" in gpu_info:
        return NVIDIA_GPU_TYPE.VOLTA
    elif "A100" in gpu_info:
        return NVIDIA_GPU_TYPE.AMPERE
    elif "H100" in gpu_info:
        return NVIDIA_GPU_TYPE.HOPPER
    elif any(gpu in gpu_info for gpu in ("B100", "B200", "B300")):
        return NVIDIA_GPU_TYPE.BLACKWELL
    else:
        return NVIDIA_GPU_TYPE.AMPERE


def detect_interconnect(group_size: int) -> InterconnectType:
    """Auto-detect interconnect type from GPU generation and group topology."""
    gpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 8
    gpu_gen = get_gpu_type()
    if math.ceil(group_size / gpus_per_node) == 1:
        if not _has_nvlink():
            return InterconnectType.PCIE
        return _GPU_TO_INTRA.get(gpu_gen, InterconnectType.NVLINK_A100)
    return _GPU_TO_INTER.get(gpu_gen, InterconnectType.IB_HDR)


def get_collective_type_from_kernel_name(kernel_name: str) -> NCCL_COLL:
    if kernel_name is None:
        raise AssertionError("kernel_name must not be None")
    if "all_reduce" in kernel_name:
        return NCCL_COLL.ALL_REDUCE
    elif "all_gather" in kernel_name:
        return NCCL_COLL.ALL_GATHER
    elif "reduce_scatter" in kernel_name:
        return NCCL_COLL.REDUCE_SCATTER
    elif any(comm in kernel_name for comm in ("all_to_all", "alltoall")):
        return NCCL_COLL.ALL_TO_ALL
    elif any(comm in kernel_name for comm in ("isend", "irecv", "batch_p2p")):
        return NCCL_COLL.P2P
    else:
        return NCCL_COLL.UNSUPPORTED


def get_collective_type(node: ir.IRNode) -> NCCL_COLL:
    if not isinstance(node, ir._CollectiveKernel):
        raise ValueError(f"node is not a collective kernel: {node}")

    name = node.python_kernel_name
    if name is None:
        raise AssertionError("node.python_kernel_name must not be None")
    return get_collective_type_from_kernel_name(name)


def get_ir_node_size_numel(size: torch.Size, fallback: int = 4096 * 4096) -> int:
    numel = sympy_product(size)
    if isinstance(numel, sympy.Integer):
        return int(numel)
    return V.graph.sizevars.optimization_hint(numel, fallback=fallback)


def get_fx_node_size_numel(size: torch.Size, fallback: int = 4096 * 4096) -> int:
    numel = functools.reduce(operator.mul, size, 1)
    result = optimization_hint(numel, fallback=fallback)
    return result


def get_collective_input_size_bytes(node: ir.IRNode) -> int:
    sz_bytes = 0
    for inp in node.inputs:  # type: ignore[attr-defined]
        numel = get_ir_node_size_numel(inp.layout.size)
        sz_bytes += numel * get_dtype_size(inp.layout.dtype)
    return sz_bytes


def get_collective_group_size(node: ir.IRNode) -> int:
    if isinstance(node, ir._CollectiveKernel) and not isinstance(node, ir._WaitKernel):
        from torch.distributed.distributed_c10d import _get_group_size_by_name

        return _get_group_size_by_name(node.constant_args[-1])
    else:
        raise TypeError(f"Unsupported collective type: {node}")


####################################################################################################################
# The following code and constants are adapted from https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc #
####################################################################################################################


class NCCL_HW(IntEnum):
    NVLINK = 0
    PCI = 1
    NET = 2


class NCCL_ALGO(IntEnum):
    TREE = 0
    RING = 1


class NCCL_PROTO(IntEnum):
    # Matches NCCL devcomm.h protocol enum
    LL = 0  # Low-latency
    LL128 = 1  # Low-latency 128-byte
    SIMPLE = 2


_NUM_ALGOS = 2  # TREE, RING
_NUM_PROTOS = 3  # LL, LL128, SIMPLE

# Base latencies in us — [algo][proto]
# https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc (ncclTunerConstantsDefaults)
baseLat = [
    # Tree
    [6.8, 14.0, 8.4],
    # Ring
    [6.6, 14.0, 8.4],
]

# Hardware latencies in us — [hw][algo][proto]
# https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc (ncclTunerConstantsDefaults)
hwLat = [
    # NVLINK
    [
        [0.6, 1.25, 4.0],  # Tree
        [0.6, 1.9, 3.4],  # Ring
    ],
    # PCI
    [
        [1.0, 1.9, 4.0],  # Tree
        [1.0, 2.5, 5.7],  # Ring
    ],
    # NET
    [
        [5.0, 8.5, 14.0],  # Tree
        [2.7, 4.0, 14.0],  # Ring
    ],
]


# LL max bus BW — [compCapIndex][scaleIndex(N1/N2/N4+)]
# https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc (ncclTunerConstantsDefaults)
llMaxBws = [
    [39.0, 39.0, 20.4],  # Volta
    [87.7, 22.5, 19.0],  # Ampere
    [141.0, 45.0, 35.0],  # Hopper
    [282.0, 90.0, 70.0],  # Blackwell
]

# Per-channel max BW caps — [compCapIndex][scaleIndex]
# https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc (ncclTunerConstantsDefaults)
perChMaxRingLL128Bws = [
    [20.0, 20.0, 20.0],  # Volta
    [20.0, 20.0, 20.0],  # Ampere
    [36.7, 36.7, 36.7],  # Hopper
    [40.0, 40.0, 40.0],  # Blackwell
]

# https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc (ncclTunerConstantsDefaults)
perChMaxTreeLL128Bws = [
    [20.0, 20.0, 20.0],  # Volta
    [20.0, 20.0, 20.0],  # Ampere
    [36.7, 36.7, 29.0],  # Hopper
    [55.6, 31.67, 20.0],  # Blackwell
]

# https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc (ncclTunerConstantsDefaults)
perChMaxTreeBws = [
    [26.5, 18.5, 10.0],  # Volta
    [24.0, 23.6, 17.8],  # Ampere
    [38.7, 41.4, 36.0],  # Hopper
    [70.0, 42.8, 24.0],  # Blackwell
]

# Tree correction factor for medium message sizes — [proto][logSize]
# logSize = log2(nBytes >> 6), indices 0..23 map to 64B..256MB
# https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc (ncclTunerConstantsDefaults)
treeCorrectionFactor = [
    # LL
    [
        1.0,
        1.0,
        1.0,
        1.0,
        0.9,
        0.8,
        0.7,
        0.7,
        0.7,
        0.7,
        0.6,
        0.5,
        0.4,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ],
    # LL128
    [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.9,
        0.8,
        0.8,
        0.8,
        0.7,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.8,
        0.9,
        0.9,
        0.9,
        0.9,
        1.0,
        1.0,
        1.0,
    ],
    # SIMPLE
    [
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.8,
        0.7,
        0.6,
        0.6,
        0.5,
        0.5,
        0.5,
        0.5,
        0.6,
        0.7,
        0.8,
        0.7,
        0.7,
        0.8,
        0.9,
        0.9,
        0.9,
    ],
]

# Heuristic nChannels per GPU generation.
# NCCL selects dynamically via topology search (up to MAXCHANNELS/2=32 for ring).
# These approximate standard DGX/HGX topologies; see nccl/src/graph/search.cc.
_GPU_NCHANNELS = {
    NVIDIA_GPU_TYPE.VOLTA: 2,
    NVIDIA_GPU_TYPE.AMPERE: 8,
    NVIDIA_GPU_TYPE.HOPPER: 16,
    NVIDIA_GPU_TYPE.BLACKWELL: 32,
}

# NVLink bus bandwidth per GPU (GB/s).
# https://github.com/NVIDIA/nccl/blob/master/src/graph/topo.h (link speed constants)
_GPU_NVLINK_BW: dict[NVIDIA_GPU_TYPE, float] = {
    NVIDIA_GPU_TYPE.VOLTA: 120.0,
    NVIDIA_GPU_TYPE.AMPERE: 240.0,
    NVIDIA_GPU_TYPE.HOPPER: 370.0,
    NVIDIA_GPU_TYPE.BLACKWELL: 720.0,
}

# PCIe effective intra-node bandwidth (GB/s) for NCCL ring.
# Conservative: GPU->CPU->GPU ring steps limit effective BW well below link rate.
_PCIE_INTRA_NODE_BW = 24.0

# Per-GPU inter-node bandwidth (GB/s), assuming 1:1 GPU:NIC ratio.
# HDR: 200Gbps=25GB/s, NDR: 400Gbps=50GB/s per NIC.
_GPU_INTER_NODE_BW: dict[NVIDIA_GPU_TYPE, float] = {
    NVIDIA_GPU_TYPE.VOLTA: 12.0,
    NVIDIA_GPU_TYPE.AMPERE: 25.0,
    NVIDIA_GPU_TYPE.HOPPER: 50.0,
    NVIDIA_GPU_TYPE.BLACKWELL: 50.0,
}


def get_intra_node_bw() -> float:
    """Return intra-node bandwidth in GB/s. Config overrides auto-detection."""
    override = torch._inductor.config.intra_node_bw
    if override is not None:
        return float(override)
    if not _has_nvlink():
        return _PCIE_INTRA_NODE_BW
    return _GPU_NVLINK_BW.get(get_gpu_type(), 240.0)


def get_inter_node_bw() -> float:
    """Return inter-node (IB/RoCE) bandwidth in GB/s. Config overrides auto-detection."""
    override = torch._inductor.config.inter_node_bw
    if override is not None:
        return float(override)
    return _GPU_INTER_NODE_BW.get(get_gpu_type(), 25.0)


def _log2i(n: int) -> int:
    """Integer log2, matching NCCL's log2i (floor of log2)."""
    if n <= 0:
        return 0
    r = 0
    while n > 1:
        n >>= 1
        r += 1
    return r


def estimate_nccl_collective_runtime_nccl_estimator(snode) -> float | None:  # type: ignore[no-untyped-def]
    kernel = snode.node
    if kernel is None:
        raise AssertionError("snode.node must not be None")
    py_kernel_name = getattr(kernel, "python_kernel_name", "")
    pg_name = kernel.constant_args[-1]  # type: ignore[attr-defined]
    from torch.distributed.distributed_c10d import _resolve_process_group

    pg = _resolve_process_group(pg_name)
    rank: int = torch.distributed.get_rank(pg)
    # TODO(ivankobzarev): Figure out how we can use time estimations,
    # without cuda allocations.
    device = torch.device(f"cuda:{rank}")

    fn = eval(py_kernel_name)
    args, kwargs = snode_args_kwargs(snode)

    # TODO(ivankobzarev): fix out variants snode_args_kwargs
    if "all_gather_into_tensor_out" in py_kernel_name:
        args = args[1:] + args[0]

    with torch.distributed._time_estimator(group=pg, device=device) as time_estimator:
        w = fn(*args, **kwargs)
        torch.ops._c10d_functional.wait_tensor.default(w)

    est_time_us = time_estimator.estimated_time
    if est_time_us is None:
        raise AssertionError("NCCL time estimator did not produce a result")
    # -1000 constant is NCCL return in case of error during estimations.
    # Observed it for all_to_all estimations.
    if est_time_us < 0:
        return None
    est_time_ms = est_time_us / 1e3
    return est_time_ms


def _nccl_algo_time(
    nBytes: int,
    group_size: int,
    coll: NCCL_COLL,
    algo: int,
    proto: int,
) -> float:
    """Compute NCCL estimated time in us for a given (algo, proto) pair.

    Mirrors ncclTopoTuneModel bandwidth/latency computation and
    ncclTopoGetAlgoTime from NCCL tuning.cc. Returns -1 if the
    (algo, proto) combination is disabled for this configuration.
    """
    gpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 8
    nNodes = math.ceil(group_size / gpus_per_node)
    nRanks = group_size
    compCapIndex = get_gpu_type()
    index2 = nNodes - 1 if nNodes <= 2 else 2

    # Total bus bandwidth (not per-channel). Auto-detected from GPU
    # generation, or overridden by config. nChannels is only used for
    # per-channel BW cap table lookups.
    bwIntra = get_intra_node_bw()
    bwInter = get_inter_node_bw()
    nChannels = _GPU_NCHANNELS.get(compCapIndex, 8)

    # Detect interconnect for IB-specific corrections
    interconnect = detect_interconnect(group_size)
    is_ib = interconnect in (InterconnectType.IB_NDR, InterconnectType.IB_HDR)

    # LL128: disable for inter-node IB (BW tables calibrated for NVLink only)
    # and for large-scale multi-node (>2 nodes)
    if proto == NCCL_PROTO.LL128:
        if nNodes > 2:
            return -1.0
        if nNodes > 1 and is_ib:
            return -1.0

    # --- Bandwidth computation (mirrors ncclTopoTuneModel) ---
    # For IB interconnect, always use inter-node BW (bwIntra only for NVLink).
    # Original NCCL tuning.cc uses bwIntra for nNodes<=2, designed for NVSwitch;
    # doesn't apply to IB.
    if nNodes == 1:
        bw = bwIntra
    elif is_ib:
        bw = bwInter
    else:
        bw = bwIntra if nNodes <= 2 else bwInter

    # IB corrections calibrated against H100 GrandTeton RoCE profiles
    # (see agent_space/nccl_model_validation.md for calibration data):
    if is_ib and nNodes > 1:
        # NIC efficiency: fewer nodes → fewer NICs utilized → lower effective BW.
        # 2-node (single inter-hop): ~0.80×; 8+ nodes: ~1.0×
        nic_efficiency = 0.77 + 0.23 * min(1.0, (nNodes - 1) / 7)
        bw *= nic_efficiency
        # Multi-rail: larger messages utilize more NICs (8 NICs per H100 node).
        # Power-law fit (exponent 0.17) to non-coalesced profile BW vs size.
        if nNodes > 2:
            multi_rail = (max(nBytes, 50_000_000) / 50_000_000) ** 0.17
            bw *= multi_rail

    busBw = bw

    index1 = compCapIndex if nNodes == 1 else 0
    llMaxBw = llMaxBws[index1][index2]

    if algo == NCCL_ALGO.RING and proto == NCCL_PROTO.LL:
        busBw = min(llMaxBw, busBw * 0.5)
    elif algo == NCCL_ALGO.RING and proto == NCCL_PROTO.LL128:
        busBw = min(
            busBw * (120.0 / 128.0),
            nChannels * perChMaxRingLL128Bws[compCapIndex][index2],
        )
    elif algo == NCCL_ALGO.TREE and proto == NCCL_PROTO.LL:
        busBw = min(busBw * (1.0 / 3.8), llMaxBw)
    elif algo == NCCL_ALGO.TREE and proto == NCCL_PROTO.LL128:
        factor = 7.0 / 9.0 if nNodes == 1 else 120.0 / 128.0
        busBw = min(
            busBw * factor,
            nChannels * perChMaxTreeLL128Bws[compCapIndex][index2],
        )
    elif algo == NCCL_ALGO.TREE and proto == NCCL_PROTO.SIMPLE:
        if coll == NCCL_COLL.ALL_REDUCE:
            busBw = min(busBw * 0.92, nChannels * perChMaxTreeBws[compCapIndex][index2])
    # Ring+SIMPLE: no extra cap beyond busBw

    if busBw <= 0:
        return -1.0

    # nsteps for the collective
    if coll == NCCL_COLL.ALL_REDUCE:
        nsteps = 2 * (nRanks - 1)
    elif coll in (NCCL_COLL.REDUCE_SCATTER, NCCL_COLL.ALL_GATHER):
        nsteps = nRanks - 1
    elif coll == NCCL_COLL.ALL_TO_ALL:
        nsteps = 2 * (nRanks - 1)
    elif coll == NCCL_COLL.P2P:
        nsteps = 1
    else:
        return -1.0

    # Bus BW to algorithm BW conversion
    # Tree allreduce uses 0.5 ratio; Ring uses nRanks/nsteps
    if algo == NCCL_ALGO.TREE and coll == NCCL_COLL.ALL_REDUCE:
        bandwidth = busBw * 0.5
    elif algo == NCCL_ALGO.RING:
        bandwidth = busBw * (1.0 * nRanks / nsteps)
    else:
        bandwidth = busBw * 0.5

    if bandwidth <= 0:
        return -1.0

    # --- Latency computation (mirrors ncclTopoTuneModel) ---
    intraHw = NCCL_HW.PCI if interconnect == InterconnectType.PCIE else NCCL_HW.NVLINK
    lat = baseLat[algo][proto]
    intraLat = hwLat[intraHw][algo][proto]
    interLat = hwLat[NCCL_HW.NET][algo][proto]

    netOverhead = 1.0 if nNodes > 1 else 0.0
    if proto == NCCL_PROTO.SIMPLE and nNodes > 1:
        netOverhead *= 3.0
    intraLat = max(intraLat, netOverhead)

    if algo == NCCL_ALGO.RING:
        nInterSteps = (
            0
            if nNodes == 1
            else (2 * (nNodes - 1) if coll == NCCL_COLL.ALL_REDUCE else nNodes - 1)
        )
        lat += (nsteps - nInterSteps) * intraLat + nInterSteps * interLat
    elif algo == NCCL_ALGO.TREE:
        if coll == NCCL_COLL.ALL_REDUCE:
            lat += 2 * (
                (nRanks // max(nNodes, 1) - 1) * intraLat + _log2i(nNodes) * interLat
            )

    # --- ncclTopoGetAlgoTime: time = lat + nBytes / (1000 * bandwidth) ---
    # Tree correction factor for medium messages
    if algo == NCCL_ALGO.TREE and coll == NCCL_COLL.ALL_REDUCE:
        logSize = _log2i(nBytes >> 6) if nBytes >= 64 else 0
        if 0 <= logSize < 24:
            bandwidth *= treeCorrectionFactor[proto][logSize]

    time_us = lat + nBytes / (1000.0 * bandwidth)
    return time_us


def _nccl_best_algo_time(
    nBytes: int,
    group_size: int,
    coll: NCCL_COLL,
) -> tuple[float, int, int]:
    """Find the best (algo, proto) for a given collective and return (time_us, algo, proto).

    Iterates all supported (algo, proto) combinations and picks the one with
    minimum estimated time, mirroring NCCL's algorithm selection.
    """
    best_time = float("inf")
    best_algo = NCCL_ALGO.RING
    best_proto = NCCL_PROTO.LL

    for algo in (NCCL_ALGO.TREE, NCCL_ALGO.RING):
        # Tree only supports allreduce in our model
        if algo == NCCL_ALGO.TREE and coll != NCCL_COLL.ALL_REDUCE:
            continue
        for proto in (NCCL_PROTO.LL, NCCL_PROTO.LL128, NCCL_PROTO.SIMPLE):
            t = _nccl_algo_time(nBytes, group_size, coll, algo, proto)
            if 0 <= t < best_time:
                best_time = t
                best_algo = algo
                best_proto = proto

    return best_time, best_algo, best_proto


def estimate_nccl_collective_runtime_impl(
    tensor_storage_size_bytes: int, group_size: int, coll: NCCL_COLL
) -> float:
    """Returns estimated NCCL collective runtime in milliseconds (ms).

    Uses a multi-algorithm, multi-protocol model aligned with NCCL's
    ncclTopoTuneModel and ncclTopoGetAlgoTime (tuning.cc). Evaluates
    Ring and Tree algorithms across LL, LL128, and SIMPLE protocols,
    selecting the (algo, proto) pair that minimizes estimated time.
    """
    if group_size <= 1:
        return 0

    if coll == NCCL_COLL.UNSUPPORTED:
        return 0

    time_us, _, _ = _nccl_best_algo_time(tensor_storage_size_bytes, group_size, coll)
    if time_us < 0 or time_us == float("inf"):
        return 0
    return time_us / 1000.0


def compute_min_saturation_bytes(
    group_size: int,
    coll: NCCL_COLL,
    target_efficiency: float = 0.9,
) -> int:
    """Compute min message size for target BW efficiency using empirical profiles.

    Uses per-interconnect profiles calibrated to real NCCL behavior (algorithm,
    protocol, channel selection) rather than the analytical model.
    """
    _SUPPORTED = (NCCL_COLL.ALL_GATHER, NCCL_COLL.REDUCE_SCATTER, NCCL_COLL.ALL_REDUCE)
    if coll not in _SUPPORTED:
        raise AssertionError(
            f"Unsupported collective {coll}, expected one of {_SUPPORTED}"
        )

    if group_size <= 1:
        return 0

    profile = INTERCONNECT_PROFILES[detect_interconnect(group_size)]

    if coll in (NCCL_COLL.ALL_GATHER, NCCL_COLL.REDUCE_SCATTER):
        nsteps = group_size - 1
    elif coll == NCCL_COLL.ALL_REDUCE:
        nsteps = 2 * (group_size - 1)
    else:
        return 0

    total_latency_us = profile.base_latency_us + nsteps * profile.latency_per_hop_us
    latency_s = total_latency_us * 1e-6
    bw_bytes_per_s = profile.bus_bw_GBps * 1e9
    eff_ratio = target_efficiency / (1.0 - target_efficiency)
    min_bytes = int(eff_ratio * latency_s * bw_bytes_per_s)
    return max(min_bytes, profile.min_saturation_bytes)


################################################################################################################
# The above code and constants are adapted from https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc #
################################################################################################################


def estimate_nccl_collective_runtime(node: ir.IRNode) -> float:
    """Returns estimated NCCL collective runtime in milliseconds (ms).

    Uses the multi-algorithm, multi-protocol analytical model aligned with
    NCCL's tuning.cc to select the best (algo, proto) pair for the given
    collective size and topology.
    """
    tensor_storage_size_bytes = get_collective_input_size_bytes(node)
    group_size = get_collective_group_size(node)
    coll = get_collective_type(node)
    return estimate_nccl_collective_runtime_impl(
        tensor_storage_size_bytes, group_size, coll
    )


def estimate_fx_collective_size(fx_node: torch.fx.Node) -> int:
    """Estimate the size of a collective operation in bytes, including inputs and outputs."""
    input_bytes = None

    args, kwargs = fx_node.args, fx_node.kwargs
    kwargs = dict(kwargs)

    # don't double count pre-allocated buffer passed in
    kwargs.pop("out", None)

    def tensor_bytes(t: torch.Tensor) -> int:
        return get_fx_node_size_numel(t.size()) * get_dtype_size(t.dtype)

    def add_inp_bytes(inp: torch.fx.Node):
        inp_val = inp.meta.get("val", None)
        if not isinstance(inp_val, torch.Tensor):
            return

        nonlocal input_bytes
        if input_bytes is None:
            input_bytes = 0
        input_bytes += tensor_bytes(inp_val)

    pytree.tree_map_only(
        torch.fx.Node,
        add_inp_bytes,
        (args, kwargs),
    )

    output_val = fx_node.meta.get("val", None)

    if input_bytes is None or output_val is None:
        return 0

    # Coalesced collectives return a list of tensors
    if isinstance(output_val, (list, tuple)):
        output_bytes = sum(
            tensor_bytes(t) for t in output_val if isinstance(t, torch.Tensor)
        )
    elif isinstance(output_val, torch.Tensor):
        output_bytes = tensor_bytes(output_val)
    else:
        return 0

    return input_bytes + output_bytes


def estimate_fx_collective_memory_footprint(fx_node: torch.fx.Node) -> int:
    """Estimate the memory footprint of a collective operation in bytes.

    This returns the total bytes that need to be live concurrently in memory.
    For all_reduce, we divide by 2 since it can be done in-place.
    """
    from torch._inductor.fx_passes.bucketing import (
        is_all_reduce_tensor as is_all_reduce,
    )

    size = estimate_fx_collective_size(fx_node)
    return size if not is_all_reduce(fx_node) else size // 2


def estimate_nccl_collective_runtime_from_fx_node(
    fx_node: torch.fx.Node,
    override_size: int | None = None,
    use_nccl_estimator: bool = True,
) -> float:
    """Returns estimated NCCL collective runtime in milliseconds (ms).

    Tries the NCCL simulator first (if available and enabled), falls back
    to the multi-algo/proto analytical model from tuning.cc.
    """
    from torch.distributed.distributed_c10d import _get_group_size_by_name

    if fx_node.target is torch.ops._c10d_functional.all_to_all_single.default:
        # TODO(ivankobzarev): Temporarily disabled - NCCL estimator returns internal error.
        # for all_to_all during inductor compilation. Falls back to heuristic estimation.
        use_nccl_estimator = False

    if override_size is None:
        tensor_storage_size_bytes = estimate_fx_collective_size(fx_node)
    else:
        tensor_storage_size_bytes = override_size

    if isinstance(fx_node.target, str):
        raise AssertionError(f"fx_node.target must not be a str, got {fx_node.target}")
    opt_args_kwargs = normalize_function(
        fx_node.target,
        args=fx_node.args,
        kwargs=fx_node.kwargs,
        normalize_to_only_use_kwargs=True,
    )
    if opt_args_kwargs is None:
        raise AssertionError("normalize_function returned None")
    args, kwargs = opt_args_kwargs

    from torch._inductor.fx_passes.bucketing import _resolve_group_name

    group_name = _resolve_group_name(kwargs["group_name"])
    group_size = _get_group_size_by_name(group_name)
    if not isinstance(fx_node.target, torch._ops.OpOverload):
        raise AssertionError(
            f"expected fx_node.target to be an OpOverload, got {type(fx_node.target)}"
        )
    coll = get_collective_type_from_kernel_name(fx_node.target.name())

    def _nccl_estimate() -> float | None:
        # TODO: Refactor with estimate_nccl_collective_runtime_nccl_estimator
        from torch.distributed.distributed_c10d import _resolve_process_group, Backend

        pg = _resolve_process_group(group_name)
        if torch.distributed.distributed_c10d.get_backend(pg) == Backend.FAKE:
            # nccl estimator requires real process group
            return None

        device = torch.device("cuda")
        try:
            backend = pg._get_backend(device)
        except RuntimeError:
            return None
        if not backend.supports_time_estimate:
            return None

        flat_args, flat_args_pytree_spec = pytree.tree_flatten((args, kwargs))

        def _tensor(size, dtype, device) -> torch.Tensor:  # type: ignore[no-untyped-def]
            return torch.empty(
                size if override_size is None else [override_size],
                dtype=dtype,
                device=device,
            )

        def to_real_tensor(e: Any) -> Any:
            if isinstance(e, torch.fx.Node):
                return to_real_tensor(e.meta["val"])
            if isinstance(e, torch.Tensor):
                return _tensor([get_fx_node_size_numel(e.size())], e.dtype, e.device)
            return e

        flat_args = [to_real_tensor(a) for a in flat_args]
        real_args, real_kwargs = pytree.tree_unflatten(flat_args, flat_args_pytree_spec)

        fn = fx_node.target
        if not isinstance(fn, torch._ops.OpOverload):
            raise AssertionError(
                f"expected fx_node.target to be an OpOverload, got {type(fn)}"
            )
        with torch.distributed._time_estimator(
            group=pg, device=device
        ) as time_estimator:
            w = fn(*real_args, **real_kwargs)
            # Coalesced collectives return a list of tensors
            if isinstance(w, (list, tuple)):
                for t in w:
                    torch.ops._c10d_functional.wait_tensor.default(t)
            else:
                torch.ops._c10d_functional.wait_tensor.default(w)
        est_time_us = time_estimator.estimated_time
        if est_time_us is None:
            raise AssertionError("NCCL time estimator did not produce a result")
        # -1000 constant is NCCL return in case of error during estimations.
        # Observed it for all_to_all estimations.
        if est_time_us < 0:
            return None
        est_time_ms = est_time_us / 1e3
        return est_time_ms

    if use_nccl_estimator:
        est_time_ms = _nccl_estimate()
        if est_time_ms is not None:
            return est_time_ms

    return estimate_nccl_collective_runtime_impl(
        tensor_storage_size_bytes, group_size, coll
    )
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import atexit
import functools
import json
import logging
import multiprocessing
import os
import re
import sys
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
)
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from time import time, time_ns
from typing import Any, TYPE_CHECKING

import torch
from torch._dynamo.device_interface import get_registered_device_interfaces
from torch._dynamo.utils import (
    counters,
    dynamo_timed,
    get_metrics_context,
    set_feature_use,
)
from torch._inductor import config
from torch._inductor.codecache import (
    _load_triton_kernel_from_source,
    code_hash,
    CodeCacheFuture,
    CppCodeCache,
    CppPythonBindingsCodeCache,
    CUDACodeCache,
    HalideCodeCache,
    LambdaFuture,
    ROCmCodeCache,
    StaticAutotunerFuture,
    torch_key,
    XPUCodeCache,
)
from torch._inductor.compile_worker.subproc_pool import (
    AnyPool,
    SubprocException,
    SubprocPool,
)
from torch._inductor.compile_worker.tracked_process_pool import (
    TrackedProcessPoolExecutor,
)
from torch._inductor.compile_worker.utils import _async_compile_initializer
from torch._inductor.runtime.compile_tasks import (
    _set_triton_libdevice_path,
    _set_triton_ptxas_path,
    _worker_compile_pycodecache_kernel,
    _worker_compile_triton,
)
from torch._inductor.utils import clear_on_fresh_cache
from torch._inductor.virtualized import V
from torch._utils_internal import log_triton_builds
from torch.hub import _Faketqdm, tqdm
from torch.utils._ordered_set import OrderedSet
from torch.utils._triton import has_triton_package


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch._inductor.runtime.hints import HalideMeta
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

# timing metrics for time spent in the compilation
_cumulative_compile_time = 0.0
_t0: float | None = None

kernel_code_log = torch._logging.getArtifactLogger(__name__, "kernel_code")

log = logging.getLogger(__name__)

_triton_kernel_metrics: dict[str, dict[str, Any]] | None = None

size_hints_regex = re.compile(
    r"size_hints=(\{.*?\})",
)


def _pycodecache_kernel_compile_env() -> dict[str, str | None]:
    env_vars = [
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CUTLASS_DIR",
    ]
    return {v: os.environ.get(v) for v in env_vars}


def pre_fork_setup():
    """
    Setup that must be done prior to forking with a process pool.
    """
    # ensure properties have been calculated before processes
    # are forked
    caching_device_properties()

    # Computing the triton key can be slow. If we call it before fork,
    # it will be cached for the forked subprocesses.
    from torch._inductor.runtime.triton_compat import HAS_TRITON, triton_key

    if HAS_TRITON:
        triton_key()


def caching_device_properties():
    for _, device_interface in get_registered_device_interfaces():
        if device_interface.is_available():
            device_interface.Worker.get_device_properties()


def _compile_start() -> None:
    global _t0, _triton_kernel_metrics
    if _t0 is None:
        _t0 = time()
    if _triton_kernel_metrics is None:
        _triton_kernel_metrics = {}


def _compile_end() -> None:
    global _cumulative_compile_time, _t0, _triton_kernel_metrics
    if _t0 is not None:
        t1 = time()
        _cumulative_compile_time += t1 - _t0
        _t0 = None
        # print("CUMULATIVE COMPILE TIME", _cumulative_compile_time)
    if _triton_kernel_metrics:
        # Log triton kernel info
        sorted_info = dict(sorted(_triton_kernel_metrics.items()))
        torch._logging.trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "triton_kernel_info",
                "encoding": "json",
            },
            payload_fn=lambda: json.dumps(sorted_info),
        )
        _triton_kernel_metrics = None


def _add_triton_kernel_info(kernel_name: str, info: dict[str, Any]):
    global _triton_kernel_metrics
    # Must be called between _compile_start and _compile_end
    if _triton_kernel_metrics is not None:
        _triton_kernel_metrics[kernel_name] = info


def _emit_triton_kernel_compile_metric(
    kernel: CachingAutotuner,
    kernel_name: str,
    elapsed_us: int,
) -> None:
    """Emit per-kernel ``compile_time_us`` to both the dynamo
    ``_triton_kernel_metrics`` map and the ``MetricsContext`` top-N.

    Note: ``kernel.autotune_cache_info`` is only mutated in place when
    non-empty; when ``None`` or ``{}`` the metric still reaches
    ``_triton_kernel_metrics`` via a throwaway dict, but the kernel
    attribute stays untouched.
    """
    info = kernel.autotune_cache_info or {}
    info["compile_time_us"] = elapsed_us
    _add_triton_kernel_info(kernel_name, info)
    get_metrics_context().add_top_n(
        "triton_kernel_compile_times_us", kernel_name, elapsed_us
    )


_IS_WINDOWS = sys.platform == "win32"

log = logging.getLogger(__name__)

# Used to keep track of all process pools invoked so far.
_pool_set = OrderedSet[AnyPool]()


def shutdown_compile_workers() -> None:
    """Shut down all outstanding compile-worker pools."""
    for pool in _pool_set:
        pool.shutdown()
    AsyncCompile._ready_future = None
    after_fork()


def after_fork():
    """Reset pools to initial state without shutting them down"""
    _pool_set.clear()
    AsyncCompile._ready_future = None
    AsyncCompile.process_pool.cache_clear()


try:
    os.register_at_fork(after_in_child=after_fork)
except AttributeError:
    pass  # register_at_fork does not exist on windows


def get_compile_threads() -> int:
    """
    Temporary for internal rollout. Assign config.compile_threads lazily and return it.
    TODO: remove after rollout.
    """
    if config.compile_threads is None:
        config.compile_threads = config.decide_compile_threads()
    return config.compile_threads


def _process_pool_allowed() -> bool:
    # Multiprocessing daemons are not allowed to create child processes. This
    # only applies to direct multiprocessing modes: SubprocPool starts its
    # sidecar with subprocess.Popen, so the sidecar does not inherit the
    # multiprocessing daemon flag and can own its own ProcessPoolExecutor.
    return (
        config.worker_start_method == "subprocess"
        or not multiprocessing.current_process().daemon
    )


@clear_on_fresh_cache
class CompiledTritonKernels:
    """
    In memory cache for storing compiled triton kernels.

    Each triton kernel is keyed by the hash of its source code. Each value stored
    in the cache is a return value of AsyncCompile.triton().

    Currently, the cache stores Future objects, but it should be generalizable for any kernels.
    """

    _cache: dict[str, CodeCacheFuture] = {}

    @staticmethod
    def key(kernel_src: str):
        """
        Generates a cache key given a triton kernel's full source code.
        This source includes the inductor meta, compilation metadata, the kernel itself, etc.
        `kernel_src` should be the exact string passed to async_compile.triton()'s first argument.
        """
        # Hashes the kernel source with torch_key into a single hash key
        return code_hash(kernel_src, extra=torch_key())

    @staticmethod
    def save(kernel_src: str, future: CodeCacheFuture):
        """
        Saves a compiled triton kernel to the cache.
        TODO: We store a LambdaFuture as that's the callable returned by async_compile.triton,
        but the real type we want to return here is actually an abstract triton kernel.

        TODO: Source code here is not just the kernel's source code, but also includes the inductor preamble, etc.
        so it could be less strict.
        """
        key = CompiledTritonKernels.key(kernel_src)
        CompiledTritonKernels._cache[key] = future

    @staticmethod
    def get(kernel_src: str) -> CodeCacheFuture | None:
        key = CompiledTritonKernels.key(kernel_src)
        return CompiledTritonKernels._cache.get(key, None)

    @staticmethod
    def cache_clear():
        CompiledTritonKernels._cache = {}

    @staticmethod
    def remove_future(kernel_src: str) -> None:
        key = CompiledTritonKernels.key(kernel_src)

        # Delete the LambdaFuture if there is one
        if key in CompiledTritonKernels._cache:
            del CompiledTritonKernels._cache[key]


class AsyncCompile:
    """
    Utilities to compile in thread pools or subprocess pools (in the case of Triton).
    """

    _ready_future: Future[Any] | None = None
    _metal_sources: list[tuple[str, str, list[str]]] | None = None

    def __init__(self) -> None:
        pass

    @staticmethod
    @functools.lru_cache(1)
    def pool() -> ThreadPoolExecutor:
        if get_compile_threads() <= 1:
            raise AssertionError(
                f"expected get_compile_threads() > 1, got {get_compile_threads()}"
            )
        return ThreadPoolExecutor(get_compile_threads())

    @staticmethod
    def _get_ready():
        """No-op function to help mark when the subprocess pool is ready."""
        return "ready"

    @staticmethod
    @functools.lru_cache(1)
    def process_pool() -> AnyPool:
        if get_compile_threads() <= 1:
            raise AssertionError(
                f"expected get_compile_threads() > 1, got {get_compile_threads()}"
            )
        if not _process_pool_allowed():
            raise RuntimeError(
                "Inductor async compile process pools are disabled in daemonic "
                "multiprocessing processes. Set "
                "torch._inductor.config.worker_start_method = 'subprocess' "
                "(or TORCHINDUCTOR_WORKER_START=subprocess) to use the "
                "SubprocPool path, which is not affected by the daemon restriction."
            )
        AsyncCompile._ready_future = None
        log.info(
            "Creating '%s' pool with %d workers",
            config.worker_start_method,
            get_compile_threads(),
        )

        pool: AnyPool
        if config.worker_start_method == "subprocess":
            # Wrapper around ProcessPoolExecutor forks in a new process we control
            pool = SubprocPool(
                get_compile_threads(), quiesce=config.quiesce_async_compile_pool
            )
        else:
            if config.worker_start_method == "spawn":
                # Avoid creating pools in the spawned subprocs themselves:
                os.environ["TORCH_WARM_POOL"] = "0"
            pre_fork_setup()
            ctx = multiprocessing.get_context(config.worker_start_method)
            pool = TrackedProcessPoolExecutor(
                get_compile_threads(),
                mp_context=ctx,
                initializer=partial(_async_compile_initializer, os.getpid()),
            )

        # When this pool is created in a multiprocessing subprocess, the normal
        # atexit handler may not run, and we need to register our own handler.
        # exitpriority has to be high, because another one of the finalizers will
        # kill the worker thread that sends the shutdown message to the workers.
        multiprocessing.util.Finalize(None, pool.shutdown, exitpriority=sys.maxsize)

        _pool_set.add(pool)
        return pool

    @classmethod
    def warm_pool(cls) -> None:
        if get_compile_threads() <= 1 or not _process_pool_allowed():
            return
        _compile_start()
        # Pool is created on first access. Note for a SubprocPool, the sidecar process starts,
        # but its ProcessPoolExecutor does not initialize until a wakeup() call or the first
        # job is submitted.
        cls.process_pool()
        _compile_end()

    @classmethod
    def wait_pool_ready(cls, timeout=120) -> None:
        cls.use_process_pool()
        if cls._ready_future is not None:
            cls._ready_future.result(timeout=timeout)

    @classmethod
    def submit(cls, task: Callable[..., Any]) -> Any:
        if get_compile_threads() <= 1:
            return task()
        return cls.pool().submit(task)

    @classmethod
    def use_process_pool(cls):
        if get_compile_threads() <= 1 or not _process_pool_allowed():
            return False

        # Proton instrumentation backend requires compilation to happen in the main
        # process so it can instrument the Triton IR during JIT compilation.
        # Force synchronous compilation when proton profiling is enabled.
        if config.triton.proton_profiling:
            return False

        # Create a dummy job to check if the pool is ready. Submit it here instead of at
        # pool creation so we don't launch the full pool of worker subprocesses until
        # we're sure they're needed.
        if not cls._ready_future:
            cls._ready_future = cls.process_pool().submit(cls._get_ready)
        return cls._ready_future.done()

    @classmethod
    def wait_process_pool_ready(cls, timeout: float = 120) -> bool:
        """Block (up to ``timeout`` s) until the process pool is ready, returning
        whether it's usable.

        Like use_process_pool() but blocking. Use when a backend's serial
        fallback is far costlier than the warmup wait -- e.g. NVGEMM subprocess
        precompile, where skipping the pool forces ~15x-slower lazy compilation
        at benchmark time. (use_process_pool()'s non-blocking readiness check can
        race pool warmup when little other compilation precedes the decision.)
        On timeout, degrade gracefully (return False -> serial) rather than hang
        on a stuck worker.
        """
        if get_compile_threads() <= 1 or not _process_pool_allowed():
            return False
        if config.triton.proton_profiling:
            return False
        if not cls._ready_future:
            cls._ready_future = cls.process_pool().submit(cls._get_ready)
        try:
            cls._ready_future.result(timeout=timeout)
        except FuturesTimeoutError:
            log.warning(
                "Process pool not ready after %ss; falling back to serial", timeout
            )
            return False
        except (BrokenProcessPool, RuntimeError) as e:
            # A warmup worker died or the pool was closed. The readiness probe
            # failing must degrade to serial (the documented contract), not
            # propagate and abort the caller's algorithm selection.
            log.warning("Process pool unusable (%s); falling back to serial", e)
            return False
        return True

    @classmethod
    def wakeup(cls) -> None:
        """
        If using a SubprocPool, signal the sidecar process to start up its
        ProcessPoolExecutor.
        """
        if not cls.use_process_pool():
            return
        pool = cls.process_pool()
        if isinstance(pool, SubprocPool):
            pool.wakeup()

    def triton(self, kernel_name: str, source_code: str, device_str: str = "cuda"):
        """
        Async_compile.triton is more complicated than the other backends because
        we're trying to optimize compile time as much as possible for this hot callsite.

        First of all, the function is cached by CompiledTritonKernels; if there's a kernel
        already compiled, we grab it directly from the cache and return.

        Otherwise, if we have multiple compile threads, we kick off triton compilations on each
        worker process by giving it a kernel and source code to compile. The worker initializes
        a CachingAutotuner, runs triton compilation, and pickles the kernel back to us.
        We use TritonCompileResult to represent the objects being pickled back to us by each
        worker.

        Some maybe not obvious things that are pickled back to us:
        - Most of the time, we can avoid sending back CachingAutotuner.fn and other metadata
          and do not have to pay the cost of loading the triton kernel on the parent. But certain
          cases, like coordesc tuning and dynamic_scale_rblock, require us to reload the function
          in the parent lazily when we require it.
        - The AutotuneCache, if enabled, is constructed on each worker per triton config
          and pickled by to us via `CachingAutotuner.save_cache_hook`.
        """
        load_kernel = functools.partial(
            _load_triton_kernel_from_source, kernel_name, source_code
        )

        def reload_kernel_in_parent():
            # Benchmark how often this happens
            with dynamo_timed("reload_kernel_in_parent"):
                return load_kernel()

        counters["inductor"]["async_compile_cache_miss"] += 1

        kernel_code_log.info("Triton Kernel:\n%s", source_code)
        _compile_start()

        if os.environ.get("TRITON_INTERPRET", "0") == "1":
            return getattr(
                torch._inductor.codecache.PyCodeCache.load(source_code), kernel_name
            )

        is_parallel = self.use_process_pool()
        set_feature_use("parallel_compile_post_warmup", is_parallel)

        compile_id = torch._guards.CompileContext.current_compile_id()
        is_backward = getattr(V.graph, "is_backward", False)

        if (future := CompiledTritonKernels.get(source_code)) is not None:
            counters["inductor"]["async_compile_cache_hit"] += 1
            # Set reload_kernel_from_src properly based on source_code
            if isinstance(future, StaticAutotunerFuture):
                # Remove the future now that we've cache hit
                CompiledTritonKernels.remove_future(source_code)
                future.reload_kernel_from_src = reload_kernel_in_parent
            if is_parallel:
                return future
            else:
                return future.result()

        # Cache miss
        if is_parallel:
            # Ensure libdevice path is set in os.environ before passing to workers
            _set_triton_libdevice_path()
            # We want to support changing these env vars after (and while) the
            # process pool is running, so pass them to the subprocess to reset.
            env_vars = [
                "TORCHINDUCTOR_CACHE_DIR",
                "TRITON_CACHE_DIR",
                "TRITON_LIBDEVICE_PATH",
            ]
            extra_env = {v: os.environ.get(v) for v in env_vars}
            extra_config = {
                "use_static_triton_launcher": torch._inductor.config.use_static_triton_launcher
            }

            if len(torch._inductor.config.autotune_lookup_table) > 0:
                m = size_hints_regex.search(source_code)
                if m:
                    size_hints_str = m.group(1)
                else:
                    size_hints_str = str(None)

                triton_src = source_code.split("@triton.jit\n")[1]
                from torch._inductor.runtime.triton_heuristics import (
                    generate_lookup_hash_from_source_code,
                )

                fn_hash = generate_lookup_hash_from_source_code(
                    size_hints_str, triton_src
                )

                if fn_hash in torch._inductor.config.autotune_lookup_table:
                    extra_config["autotune_lookup_table"] = {  # type: ignore[assignment]
                        fn_hash: torch._inductor.config.autotune_lookup_table[fn_hash]
                    }

            task = self.process_pool().submit(
                _worker_compile_triton,
                load_kernel,
                extra_env,
                extra_config,
            )

            def get_result() -> CachingAutotuner:
                try:
                    kernel, elapsed_us = task.result()
                except SubprocException as e:
                    raise e.with_name(kernel_name) from e

                # Now that we've compiled, we should clear the future
                # so it can't be used again
                kernel.set_compile_info(compile_id, is_backward)
                CompiledTritonKernels.remove_future(source_code)

                kernel.restore_after_unpickle(old_values=None)

                kernel.precompile(
                    warm_cache_only=False,
                    reload_kernel=reload_kernel_in_parent,
                    static_triton_bundle_key=CompiledTritonKernels.key(source_code),
                )
                _emit_triton_kernel_compile_metric(kernel, kernel_name, elapsed_us)
                return kernel

            future = LambdaFuture(get_result, future=task)
            CompiledTritonKernels.save(source_code, future)
            return future
        else:
            with dynamo_timed(
                "async_compile.precompile",
                log_pt2_compile_event=True,
                dynamo_compile_column_us="triton_compile_time_us",
                log_waitcounter=True,
                waitcounter_name_override="compile_triton",
            ):
                fail = None
                try:
                    start_ns = time_ns()
                    _set_triton_ptxas_path()
                    _set_triton_libdevice_path()
                    kernel = load_kernel()
                    kernel.set_compile_info(compile_id, is_backward)
                    kernel.precompile(
                        warm_cache_only=False,
                        static_triton_bundle_key=CompiledTritonKernels.key(source_code),
                    )
                    elapsed_us = (time_ns() - start_ns) // 1000
                    _emit_triton_kernel_compile_metric(kernel, kernel_name, elapsed_us)
                    return kernel
                except Exception as e:
                    fail = str(e)
                    raise
                finally:
                    log_triton_builds(fail=fail)

    def multi_kernel(self, *args, **kwargs) -> Any:
        from torch._inductor.codegen.multi_kernel import MultiKernelCall

        # no need to call this in parallel since the sub-kernels are already parallel tasks
        return MultiKernelCall(*args, **kwargs)

    def size_hint_multi_kernel(self, *args, **kwargs) -> Any:
        from torch._inductor.codegen.multi_kernel import SizeHintMultiKernelCall

        return SizeHintMultiKernelCall(*args, **kwargs)

    def cpp(self, source_code: str):
        kernel_code_log.info("CPP Kernel:\n%s", source_code)
        if get_compile_threads() <= 1:
            return CppCodeCache.load(source_code).kernel
        else:
            get_result = CppCodeCache.load_async(source_code, submit_fn=self.submit)
            return LambdaFuture(lambda: get_result().kernel)

    def cpp_pybinding(self, argtypes: list[str], source_code: str):
        kernel_code_log.info("CPP+Bindings Kernel:\n%s", source_code)
        if get_compile_threads() <= 1:
            return CppPythonBindingsCodeCache.load_pybinding(argtypes, source_code)
        else:
            get_result = CppPythonBindingsCodeCache.load_pybinding_async(
                argtypes, source_code, submit_fn=self.submit
            )
            return LambdaFuture(get_result)

    def cutlass(self, cache_cls, source_code, dst_file_ext, aot_compile=False):
        def task():
            if aot_compile:
                # We rely on JITInductor to compile the CUDA code,
                # so that we can load it into AOTInductor.
                output_path, *_ = cache_cls.compile(source_code, "o")
                cache_cls.aot_kernels_o.append(output_path)
            return cache_cls.load(source_code, dst_file_ext)[0]

        return self.submit(task)

    def cuda(self, source_code, dst_file_ext, aot_compile=False):
        kernel_code_log.info("CUDA Kernel:\n%s", source_code)
        return self.cutlass(CUDACodeCache, source_code, dst_file_ext, aot_compile)

    def xpu(self, source_code, dst_file_ext, aot_compile=False):
        kernel_code_log.info("XPU Kernel:\n%s", source_code)
        return self.cutlass(XPUCodeCache, source_code, dst_file_ext, aot_compile)

    def rocm(
        self,
        source_code,
        dst_file_ext,
        aot_compile=False,
    ):
        kernel_code_log.info("ROCm Kernel:\n%s", source_code)

        def task():
            if aot_compile:
                output_path, *_ = ROCmCodeCache.compile(source_code, dst_file_ext="o")
                ROCmCodeCache.aot_kernels_o.append(output_path)
            if config.rocm.generate_test_runner:
                _ = ROCmCodeCache.compile(source_code, dst_file_ext="exe")
            return ROCmCodeCache.load(source_code, dst_file_ext)[0]

        return self.submit(task)

    def halide(self, meta: HalideMeta, source_code: str):
        kernel_code_log.info("Halide Kernel:\n%r\n%s", meta, source_code)
        if get_compile_threads() <= 1:
            return HalideCodeCache.generate_halide(meta, source_code)
        else:
            get_result = HalideCodeCache.generate_halide_async(
                meta, source_code, submit_fn=self.submit
            )
            return LambdaFuture(get_result)

    def _load_kernel_wrapper(self, kernel_name, main_suffix, wrapper_cls, key, path):
        """Reload a kernel module from PyCodeCache and wrap the entry point."""
        mod = torch._inductor.codecache.PyCodeCache.load_by_key_path(key, path)
        main_func_name = f"{kernel_name}_{main_suffix}"
        return wrapper_cls(getattr(mod, main_func_name), kernel_path=path)

    def cutedsl(self, kernel_name: str, source_code: str, precompile_metadata=None):
        """
        Compile CuteDSL (CUTLASS Python DSL) kernels.

        Args:
            kernel_name: Name of the kernel to be defined
            source_code: Source code of the CuteDSL kernel, as a string
            precompile_metadata: Optional dict with shapes/dtypes for triggering
                real CuTe DSL compilation in the subprocess worker.

        Note:
            CuteDSL currently requires source files to do its compilation, there we
            use the PyCodeCache to write the source code to a file and load it.
        """
        from torch._inductor.codegen.cutedsl.cutedsl_kernel import (
            CuteDSLKernelWrapper,
            MAIN_SUFFIX,
        )

        kernel_code_log.info("CuteDSL Kernel:\n%s", source_code)
        _compile_start()

        is_parallel = self.use_process_pool()

        if is_parallel:
            extra_env = _pycodecache_kernel_compile_env()

            subprocess_task = self.process_pool().submit(
                _worker_compile_pycodecache_kernel,
                kernel_name,
                source_code,
                MAIN_SUFFIX,
                extra_env,
                precompile_metadata,
            )

            def get_result() -> CuteDSLKernelWrapper:
                try:
                    key, path, elapsed_us = subprocess_task.result()
                except SubprocException as e:
                    raise e.with_name(kernel_name) from e
                log.debug(
                    "CuteDSL kernel %s compiled in subprocess in %dus",
                    kernel_name,
                    elapsed_us,
                )
                return self._load_kernel_wrapper(
                    kernel_name, MAIN_SUFFIX, CuteDSLKernelWrapper, key, path
                )

            return LambdaFuture(get_result, future=subprocess_task)
        else:
            key, path = torch._inductor.codecache.PyCodeCache.write(source_code)
            mod = torch._inductor.codecache.PyCodeCache.load_by_key_path(key, path)

            main_func_name = f"{kernel_name}_{MAIN_SUFFIX}"
            if not hasattr(mod, main_func_name):
                available = [name for name in dir(mod) if callable(getattr(mod, name))]
                raise RuntimeError(
                    f"Could not find CuteDSL main kernel function '{main_func_name}'. Available callables: {available}"
                )

            return CuteDSLKernelWrapper(getattr(mod, main_func_name), kernel_path=path)

    def pallas(self, kernel_name: str, source_code: str):
        """
        Compile Pallas (JAX experimental) kernels.

        Args:
            kernel_name: Name of the kernel to be defined
            source_code: Source code of the Pallas kernel, as a string

        Note:
            Pallas kernels are Python code that uses JAX and Pallas APIs.
            We use the PyCodeCache to write the source code to a file and load it.
        """
        from torch._inductor.codegen.pallas import MAIN_SUFFIX, PallasKernelWrapper

        kernel_code_log.info("Pallas Kernel:\n%s", source_code)

        def task():
            key, path = torch._inductor.codecache.PyCodeCache.write(source_code)
            mod = torch._inductor.codecache.PyCodeCache.load_by_key_path(key, path)

            # Find our special entry point named function
            main_func_name = f"{kernel_name}_{MAIN_SUFFIX}"
            if not hasattr(mod, main_func_name):
                available = [name for name in dir(mod) if callable(getattr(mod, name))]
                raise RuntimeError(
                    f"Could not find Pallas main kernel function '{main_func_name}'. Available callables: {available}"
                )

            return PallasKernelWrapper(getattr(mod, main_func_name), kernel_path=path)

        if get_compile_threads() <= 1:
            return task()
        else:
            future = self.submit(task)
            return LambdaFuture(lambda: future.result())

    def nv_universal_gemm(
        self, kernel_name: str, source_code: str, precompile_metadata=None
    ):
        """
        Compile NVIDIA Universal GEMM kernels.

        Args:
            kernel_name: Name of the kernel to be defined
            source_code: Source code of the kernel, as a string
            precompile_metadata: Optional dict with shapes/dtypes for triggering
                real CuTe DSL compilation in the subprocess worker.

        Note:
            NVIDIA Universal GEMM kernels are Python code that calls the cutlass.operators library.
            We use the PyCodeCache to write the source code to a file and load it.
        """
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_kernel import (
            NVUniversalGemmKernelWrapper,
        )
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_scheduling import (
            MAIN_SUFFIX,
        )

        kernel_code_log.info("NVIDIA Universal GEMM Kernel:\n%s", source_code)
        _compile_start()

        is_parallel = self.use_process_pool()

        if is_parallel:
            extra_env = _pycodecache_kernel_compile_env()

            subprocess_task = self.process_pool().submit(
                _worker_compile_pycodecache_kernel,
                kernel_name,
                source_code,
                MAIN_SUFFIX,
                extra_env,
                precompile_metadata,
            )

            def get_result() -> NVUniversalGemmKernelWrapper:
                try:
                    key, path, elapsed_us = subprocess_task.result()
                except SubprocException as e:
                    raise e.with_name(kernel_name) from e
                log.debug(
                    "NV Universal GEMM kernel %s compiled in subprocess in %dus",
                    kernel_name,
                    elapsed_us,
                )
                return self._load_kernel_wrapper(
                    kernel_name, MAIN_SUFFIX, NVUniversalGemmKernelWrapper, key, path
                )

            return LambdaFuture(get_result, future=subprocess_task)
        else:
            key, path = torch._inductor.codecache.PyCodeCache.write(source_code)
            mod = torch._inductor.codecache.PyCodeCache.load_by_key_path(key, path)

            main_func_name = f"{kernel_name}_{MAIN_SUFFIX}"
            if not hasattr(mod, main_func_name):
                available = [name for name in dir(mod) if callable(getattr(mod, name))]
                raise RuntimeError(
                    f"Could not find NVIDIA Universal GEMM main kernel function "
                    f"'{main_func_name}'. Available callables: {available}"
                )

            return NVUniversalGemmKernelWrapper(
                getattr(mod, main_func_name), kernel_path=path
            )

    def nvgemm_precompile(
        self,
        kernel_name,
        variant_name,
        accumulator_type,
        input_tensor_meta,
        output_tensor_meta,
        cuda_ctx,
        scale_type_a=None,
        scale_type_b=None,
        swizzle_type_a=None,
        swizzle_type_b=None,
        has_bias_epilogue=False,
        swap_ab=False,
        metadata=None,
    ):
        """Submit NVGEMM kernel precompilation to the subprocess pool.

        Compiles the CuTeDSL kernel artifact in a subprocess worker so the
        thread-unsafe kernel.compile() call is process-isolated. The compiled
        artifact is saved to the disk cache; the main process loads it on the
        next cache lookup during benchmarking.
        """
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_kernel import (
            _worker_nvgemm_autotuning_precompile,
        )

        extra_env = _pycodecache_kernel_compile_env()

        return self.process_pool().submit(
            _worker_nvgemm_autotuning_precompile,
            kernel_name,
            variant_name,
            accumulator_type,
            input_tensor_meta,
            output_tensor_meta,
            extra_env,
            cuda_ctx,
            scale_type_a,
            scale_type_b,
            swizzle_type_a,
            swizzle_type_b,
            has_bias_epilogue,
            swap_ab,
            metadata,
        )

    def metal(self, kernel_name: str, source: str, headers: list[str]) -> None:
        """Register a Metal kernel body; wait() compiles all registered kernels into one library."""
        if self._metal_sources is None:
            self._metal_sources = []
        self._metal_sources.append((kernel_name, source, headers))

    def wait(self, scope: dict[str, Any]) -> None:
        if get_compile_threads() > 1:
            with dynamo_timed(
                "async_compile.wait",
                log_pt2_compile_event=True,
                dynamo_compile_column_us="triton_compile_time_us",
                log_waitcounter=True,
                waitcounter_name_override="compile_triton",
            ):
                self._wait_futures(scope)

        if self._metal_sources:
            from torch._inductor.runtime.runtime_utils import compile_mps_shaders

            scope.update(compile_mps_shaders(self._metal_sources))
            self._metal_sources.clear()

        _compile_end()

    def _wait_futures(self, scope: dict[str, Any]) -> None:
        kernels = {
            key: value
            for key, value in scope.items()
            if isinstance(value, (Future, CodeCacheFuture))
        }
        pbar = tqdm(
            total=len(kernels),
            desc="Inductor Compilation",
            disable=config.disable_progress,
            delay=0,
        )
        # compile_worker_wait_timeout=0 (default) means "wait forever"; map
        # it to None so both Future.result() and CodeCacheFuture.result()
        # receive the same "no timeout" sentinel.
        wait_timeout = config.compile_worker_wait_timeout or None
        for key, result in kernels.items():
            if config.verbose_progress and not isinstance(pbar, _Faketqdm):
                pbar.set_postfix_str(key)
            try:
                kernel = result.result(timeout=wait_timeout)
                scope[key] = kernel
            except FuturesTimeoutError as e:
                # concurrent.futures.TimeoutError became an alias of the
                # builtin TimeoutError in Python 3.11; on 3.10 it is a
                # distinct class, so catch it explicitly.
                raise RuntimeError(
                    f"Inductor compile-worker future for {key!r} did not "
                    f"complete within {wait_timeout}s. Override with "
                    "TORCHINDUCTOR_COMPILE_WORKER_WAIT_TIMEOUT=<seconds>."
                ) from e
            except BrokenProcessPool as e:
                raise RuntimeError(
                    "A compilation subprocess exited unexpectedly. This "
                    "is likely due to a crash. To facilitate debugging, "
                    "you can re-run with TORCHINDUCTOR_COMPILE_THREADS=1 "
                    "to cause compilation to occur in the main process."
                ) from e
            pbar.update(1)


def maybe_warm_pool() -> None:
    if (
        os.environ.get("TORCH_TNT_IN_USE", "0") == "1"
        or os.environ.get("TORCH_WARM_POOL", "1") != "1"
        # The subprocess pool is only used for the Triton backend
        or not has_triton_package()
        # Skip for fbcode. We have internal reports of usages inside multiprocessing
        # pools that lead a multiplicative number of compile subprocesses.
        or config.is_fbcode()
    ):
        return

    AsyncCompile.warm_pool()
    # TODO: This starts the SubprocPool's internal process pool as early as possible at
    # the expense of creating a bunch of worker processes that might not be needed. We
    # could start them lazily if we're willing to lose a small amount of compile time.
    AsyncCompile.wakeup()


# On exit give the workers a chance to clean themselves up. Without this the
# resource_tracker can complain about leaked semaphores coming from the
# ProcessPoolExecutor:
#   UserWarning: resource_tracker: There appear to be 5 leaked semaphore objects
#   to clean up at shutdown
atexit.register(shutdown_compile_workers)
<|endoftext|>import dataclasses
import itertools
from typing import Any, TYPE_CHECKING

import sympy

import torch
from torch._inductor import config
from torch._inductor.dtype_propagation import DtypePropagationOpsHandler
from torch._inductor.index_propagation import SymPyOps, TypedExpr

from .ops_handler import DefaultHandler
from .virtualized import StoreMode, V


if TYPE_CHECKING:
    from torch._inductor.scheduler import SchedulerNode


def construct_symbol(count: int, dtype: torch.dtype) -> sympy.Symbol:
    return sympy.Symbol(f"unknown_{count}")


class PreservesZeros(SymPyOps, DefaultHandler):
    """
    For prologue kernels where the loads are masked, does the final store of this kernel preserve
    the zeros.
    """

    def __init__(self) -> None:
        self.count = itertools.count(0)
        self.store_preserves_zeros: bool | None = None
        self.dtype_prop = DtypePropagationOpsHandler()

    def load(self, name: str, index: sympy.Expr) -> TypedExpr:
        # In prologue fusion, all loads get broadcasted
        dtype = self.dtype_prop.load(name, index)
        return TypedExpr(
            sympy.Float(0) if dtype.is_floating_point else sympy.Integer(0), dtype
        )

    def store(
        self, name: str, index: sympy.Expr, value: TypedExpr, mode: "StoreMode" = None
    ) -> None:
        if not isinstance(self, PreservesZeros):
            raise AssertionError(f"expected PreservesZeros, got {type(self)}")
        # should only have a single store in prologue
        if self.store_preserves_zeros is not None:
            raise AssertionError("prologue should only have a single store")
        self.store_preserves_zeros = value.is_constant() and value.expr == 0

    def indirect_indexing(self, *args: Any, **kwargs: Any) -> sympy.Expr:
        return construct_symbol(next(self.count), torch.int32)

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        from torch._inductor.codegen.common import OpDecompositions

        if hasattr(OpDecompositions, name):
            return getattr(OpDecompositions, name)(*args, **kwargs).value

        dtype = getattr(self.dtype_prop, name)(*args, **kwargs)
        return TypedExpr(construct_symbol(next(self.count), dtype), dtype)


def prologue_preserves_zero_mask(prologue: "SchedulerNode") -> bool:
    """
    Does this prologue preserve zero masks
    """
    preserves_zeros = PreservesZeros()
    with V.set_ops_handler(preserves_zeros):
        prologue._body(*prologue.get_ranges())

    store_preserves_zeros = preserves_zeros.store_preserves_zeros
    if not isinstance(store_preserves_zeros, bool):
        raise AssertionError(
            f"expected bool store_preserves_zeros, got {type(store_preserves_zeros)}"
        )

    return store_preserves_zeros


@dataclasses.dataclass
class DTypeContainer:
    dtype: torch.dtype
    is_scalar: bool = False


class RecordLowPrecisionOps(DefaultHandler):
    def __init__(self, disallow_fp32_ops: bool = False) -> None:
        self.disallow_fp32_ops = disallow_fp32_ops
        self.low_precision_numeric_op = False
        self.dtype_prop = DtypePropagationOpsHandler()
        self.non_numeric_ops = (
            "to_dtype",
            "constant",
            "where",
        )

    def load(self, name: str, index: sympy.Expr) -> DTypeContainer:
        return DTypeContainer(self.dtype_prop.load(name, index))

    @staticmethod
    def store(
        name: str, index: sympy.Expr, value: TypedExpr, mode: "StoreMode" = None
    ) -> None:
        pass

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ) -> None:
        pass

    @staticmethod
    # pyrefly: ignore [bad-override]
    def indirect_indexing(*args: Any, **kwargs: Any) -> sympy.Expr:
        return sympy.S.Zero

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        out_dtype = getattr(self.dtype_prop, name)(*args, **kwargs)
        out = DTypeContainer(out_dtype, is_scalar=(name == "constant"))
        if name == "constant":
            return DTypeContainer(torch.float, is_scalar=True)

        uses_low_prec = any(
            isinstance(dtype_cont, DTypeContainer)
            and dtype_cont.dtype is not None
            and low_prec_float(dtype_cont.dtype)
            for dtype_cont in itertools.chain((out,), args, kwargs.values())
        )

        if uses_low_prec and name not in self.non_numeric_ops:
            self.low_precision_numeric_op = True

        if (
            self.disallow_fp32_ops
            and out.dtype in (torch.float32, torch.float64)
            and not out.is_scalar
        ):
            self.low_precision_numeric_op = True

        return out


def low_prec_float(dtype: torch.dtype) -> bool:
    return dtype.is_floating_point and dtype.itemsize < 4


def can_codegen_without_upcasts(
    prologue: "SchedulerNode",
    disallow_fp32_ops: bool = False,
) -> bool:
    """
    Can this prologue be run without `upcast_to_fp32` while preserving numerics.

    This is only true if the node only contains dtype conversions, indexing, and other non-arithmetic operators.

    If disallow_fp32_ops is True, then we also disallow ops that are explicitly computed in fp32 or fp64.
    """
    if prologue.get_operation_names() <= V.graph.low_precision_codegen_ops:
        return True

    low_prec_analysis = RecordLowPrecisionOps(disallow_fp32_ops)

    # Need to turn off upcasting to do analysis of whether we can turn it off
    with (
        config.patch("triton.codegen_upcast_to_fp32", False),
        V.set_ops_handler(low_prec_analysis),
    ):
        prologue._body(*prologue.get_ranges())

    return not low_prec_analysis.low_precision_numeric_op
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import collections
import functools
import itertools
import re
from enum import auto, Enum
from typing import Any, NamedTuple, TYPE_CHECKING, TypeVar

import sympy

import torch.fx
from torch._dynamo.utils import identity
from torch.fx.proxy import Scope, TracerBase
from torch.utils._sympy.functions import Mod
from torch.utils._sympy.symbol import SymT

from . import config, dependencies
from .codegen.common import index_prevent_reordering
from .ops_handler import DefaultHandler, OpsHandler, WrapperHandler
from .utils import (
    cache_on_self,
    decompose_index,
    flatten_index,
    reduction_num_outputs,
    sympy_index_symbol_with_prefix,
    sympy_subs,
)
from .virtualized import ops, V


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


T = TypeVar("T")


class InterpreterShim(torch.fx.Interpreter):
    @staticmethod
    @functools.cache
    def _dummy_gm():
        return torch.fx.symbolic_trace(identity)

    def __init__(self, graph, submodules):
        # call super() with a placeholder to avoid constructing a
        # GraphModule which is very expensive (it does codegen).
        super().__init__(self._dummy_gm(), garbage_collect_values=False)
        self.module = self  # type: ignore[assignment]
        self.graph = graph
        self.submodules = submodules
        self.extra_traceback = False
