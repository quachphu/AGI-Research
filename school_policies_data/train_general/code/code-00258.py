        p = Partial(reduce_op)
        strategies.append([p, p, r, r])
    return strategies


@register_single_dim_strategy(
    [aten.grid_sampler_2d.default, aten.grid_sampler_3d.default],
    schema_info=RuntimeSchemaInfo(1),
)
def grid_sampler_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    # grid_sampler_{2,3}d(input[N,C,...], grid[N,...,{2,3}]) -> output[N,C,...]
    # grid has no channel dim, so only batch sharding applies to both inputs.
    # Linear in input: P(sum/avg) on input with replicated grid is valid.
    return [
        [_ShardingPlaceholder(0)] * 3,
        [Partial("sum"), Partial("sum"), Replicate()],
        [Partial("avg"), Partial("avg"), Replicate()],
    ]


@register_single_dim_strategy(
    [aten.grid_sampler_2d_backward.default, aten.grid_sampler_3d_backward.default],
    schema_info=RuntimeSchemaInfo(1),
)
def grid_sampler_backward_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder | None]]:
    # grid_sampler_{2,3}d_backward: 2 outputs (grad_input, grad_grid) + 3 inputs = 5 placements, batch-only
    # The native implementation only honors output_mask[0]; grad_grid is always computed.
    output_mask = args_schema[6]
    return [
        [
            _ShardingPlaceholder(0) if output_mask[0] else None,
            _ShardingPlaceholder(0),
            _ShardingPlaceholder(0),  # grad_output
            _ShardingPlaceholder(0),  # input
            _ShardingPlaceholder(0),  # grid
        ]
    ]


def _adjust_group_norm_scalars(
    input_specs: list[DTensorSpec], schema: OpSchema
) -> OpSchema:
    """Adjust N, C, HxW scalar args in native_group_norm to local values.

    native_group_norm(input, weight?, bias?, N, C, HxW, group, eps)
    The scalar args are derived from the global input shape by the Python frontend.
    When the input is sharded, we recompute them from the local input shape.
    """
    input_spec = input_specs[0]
    if input_spec.tensor_meta is None:
        raise AssertionError("input_spec must have tensor_meta")
    local_shape, _ = compute_local_shape_and_global_offset(
        input_spec.tensor_meta.shape,
        input_spec.mesh,
        input_spec.placements,
        skip_offset=True,
    )
    # N = local_shape[0], C = local_shape[1], HxW = product of remaining dims
    n_local = local_shape[0]
    c_local = local_shape[1]
    hxw_local = 1
    for d in local_shape[2:]:
        hxw_local *= d
    args = list(schema.args_schema)
    # Find scalar arg positions: tensor slots (DTensorSpec or None for optionals)
    # precede N, C, HxW. Count both to find the offset.
    num_tensor_args = sum(isinstance(a, (DTensorSpec, type(None))) for a in args)
    args[num_tensor_args] = n_local
    args[num_tensor_args + 1] = c_local
    args[num_tensor_args + 2] = hxw_local
    return OpSchema(schema.op, tuple(args), schema.kwargs_schema)


# ---------------------------------------------------------------------------
# Normalization ops
#
# Batch norm reduces over batch (dim 0) + spatial dims (2+), keeping only
# channel (dim 1).  Channel-dim sharding is valid for both forward and
# backward: each shard processes independent channels.
#
# Group norm reduces over (C/groups, spatial) within each group per sample.
# Batch dim (0) is safe to shard — each sample is independent.
# ---------------------------------------------------------------------------

BATCH_NORM_3OUT_OPS = [
    aten.native_batch_norm.default,
    aten._native_batch_norm_legit.default,
    aten._native_batch_norm_legit.no_stats,
    aten._native_batch_norm_legit_no_training.default,
]

BATCH_NORM_4OUT_OPS = [
    aten._batch_norm_with_update.default,
]


@register_single_dim_strategy(
    BATCH_NORM_3OUT_OPS + BATCH_NORM_4OUT_OPS,
    schema_info=RuntimeSchemaInfo(1),
)
def batch_norm_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    # Batch norm normalizes per-channel (reduces over batch + spatial dims),
    # so channel-dim sharding is valid: each shard processes independent channels.
    # Unlike group_norm, batch_norm infers shapes from tensors (no scalar N/C/HxW).
    num_outputs = 4 if op in BATCH_NORM_4OUT_OPS else 3
    num_tensor_inputs = sum(isinstance(a, TensorMeta) for a in args_schema)
    # output [N,C,*] shards on dim 1; save_mean, save_invstd [C] shard on dim 0
    rule: list[Placement | _ShardingPlaceholder] = [_ShardingPlaceholder(1)]
    rule.extend([_ShardingPlaceholder(0)] * 2)  # save_mean, save_invstd
    if num_outputs == 4:
        rule.append(Replicate())  # reserve: opaque cuDNN workspace
    # input [N,C,*] shards on dim 1; weight, bias, running_mean, running_var [C] on dim 0
    rule.append(_ShardingPlaceholder(1))  # input
    rule.extend([_ShardingPlaceholder(0)] * (num_tensor_inputs - 1))
    return [rule]


@register_single_dim_strategy(
    [aten.native_batch_norm_backward.default],
    schema_info=RuntimeSchemaInfo(1),
)
def batch_norm_backward_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder | None]]:
    # Backward reduces over batch + spatial dims, orthogonal to the channel dim,
    # so channel sharding commutes with the op (same principle as forward).
    # Tensors are [N,C,*] -> Shard(1) or [C] -> Shard(0).
    output_mask = args_schema[9]
    num_tensor_inputs = sum(isinstance(a, TensorMeta) for a in args_schema)
    rule: list[Placement | _ShardingPlaceholder | None] = [
        _ShardingPlaceholder(1) if output_mask[0] else None,  # grad_input [N,C,*]
        _ShardingPlaceholder(0) if output_mask[1] else None,  # grad_weight [C]
        _ShardingPlaceholder(0) if output_mask[2] else None,  # grad_bias [C]
        _ShardingPlaceholder(1),  # grad_out [N,C,*]
        _ShardingPlaceholder(1),  # input [N,C,*]
    ]
    rule.extend([_ShardingPlaceholder(0)] * (num_tensor_inputs - 2))
    return [rule]


@register_single_dim_strategy(
    [aten.native_group_norm.default],
    schema_info=RuntimeSchemaInfo(1),
)
def group_norm_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    # native_group_norm(input, weight?, bias?, N, C, HxW, group, eps) -> (out, mean, rstd)
    # Batch dim (0) is independent. The scalar N/C/HxW args are adjusted to local
    # values by _adjust_group_norm_scalars in the sharding propagation layer.
    num_tensor_inputs = sum(isinstance(a, TensorMeta) for a in args_schema)
    # 3 outputs + input all shard on batch dim
    placements: list[Placement | _ShardingPlaceholder] = [_ShardingPlaceholder(0)] * 4
    # weight and bias (if present) must be Replicate
    placements.extend([Replicate()] * (num_tensor_inputs - 1))
    return [placements]


# Register scalar shape adjuster for group_norm so the sharding propagator
# rewrites the N/C/HxW args to local values when the input is sharded.
from torch.distributed.tensor._api import DTensor


DTensor._op_dispatcher.sharding_propagator.op_to_scalar_shape_adjuster[
    aten.native_group_norm.default
] = _adjust_group_norm_scalars
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from torch._ops import OpOverload
from torch.distributed.tensor._dtensor_spec import TensorMeta
from torch.distributed.tensor._op_schema import ArgsType, KwargsType
from torch.distributed.tensor._ops.single_dim_strategy import (
    _ShardingPlaceholder,
    register_single_dim_strategy,
)
from torch.distributed.tensor.placement_types import Placement


aten = torch.ops.aten


def _random_inplace_single_dim_strategy(
    op: OpOverload,
    args_schema: ArgsType,
    kwargs_schema: KwargsType,
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Single-dim strategy for in-place random ops (single tensor input, output follows input).

    No Partial inputs: random sampling on partial tensors is undefined.
    """
    self_meta = args_schema[0]
    if not isinstance(self_meta, TensorMeta):
        raise AssertionError
    num_outputs = sum(1 for r in op._schema.returns if "Tensor" in str(r.type))
    placements: list[list[Placement | _ShardingPlaceholder]] = []
    for i in range(len(self_meta.shape)):
        rule: list[Placement | _ShardingPlaceholder] = [_ShardingPlaceholder(i)] * (
            num_outputs + 1
        )
        placements.append(rule)
    return placements


# In-place random sampling ops: output follows input sharding exactly.
_inplace_random_ops = [
    aten.normal_.default,
    aten.uniform_.default,
    aten.native_dropout.default,
    aten.bernoulli_.float,
    aten.bernoulli.default,
    aten.log_normal_.default,
    aten.exponential_.default,
    aten.geometric_.default,
]

for _op in _inplace_random_ops:
    register_single_dim_strategy(_op, allow_uneven_sharding=True)(
        _random_inplace_single_dim_strategy
    )


@register_single_dim_strategy(aten.multinomial.default)
def multinomial_single_dim_strategy(
    op: OpOverload,
    args_schema: ArgsType,
    kwargs_schema: KwargsType,
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Single-dim strategy for multinomial.

    multinomial(self, num_samples, ...) -> Tensor
    Input: [*, n_categories], Output: [*, num_samples] (dtype=long)

    Only batch dims (all except the last) can be sharded — the last dim
    (categories) is consumed by the sampling and maps to a different
    semantic dim (num_samples) in the output.
    """
    self_meta = args_schema[0]
    if not isinstance(self_meta, TensorMeta):
        raise AssertionError
    placements: list[list[Placement | _ShardingPlaceholder]] = []
    for i in range(len(self_meta.shape) - 1):
        rule: list[Placement | _ShardingPlaceholder] = [
            _ShardingPlaceholder(i),
            _ShardingPlaceholder(i),
        ]
        placements.append(rule)
    return placements
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from torch._ops import OpOverload
from torch.distributed.tensor._dtensor_spec import TensorMeta
from torch.distributed.tensor._op_schema import ArgsType, KwargsType, RuntimeSchemaInfo
from torch.distributed.tensor._ops.single_dim_strategy import (
    register_single_dim_strategy,
)
from torch.distributed.tensor.placement_types import (
    _MaskPartial,
    Partial,
    Placement,
    Replicate,
    Shard,
)


aten = torch.ops.aten


@register_single_dim_strategy(aten.embedding.default)
def embedding_strategy(
    op: OpOverload,
    args_schema: ArgsType,
    kwargs_schema: KwargsType,
) -> list[list[Placement]]:
    """Single-dim strategy for embedding: rowwise, colwise, and batch-dim sharding.

    Placement order: [output, weight, indices]
    """
    weight_meta = args_schema[0]
    indices_meta = args_schema[1]
    if not isinstance(weight_meta, TensorMeta) or not isinstance(
        indices_meta, TensorMeta
    ):
        raise AssertionError

    # _MaskPartial hashes offset_shape, but torch.Size with SymInt (from
    # dynamo tracing) is unhashable.  Concretize to int, which adds a
    # standard dynamo guard (recompiles if the shape changes at runtime).
    weight_shape = torch.Size(int(s) for s in weight_meta.shape)
    indices_shape = indices_meta.shape
    output_emb_dim = len(indices_shape)

    strategies: list[list[Placement]] = []

    # colwise: output shard on last dim, weight shard on dim 1, indices replicate
    strategies.append([Shard(output_emb_dim), Shard(1), Replicate()])

    # rowwise: output is MaskPartial, weight shard on dim 0, indices MaskPartial
    # NOTE: same object for output & indices so the mask buffer is shared
    embedding_partial = _MaskPartial(offset_shape=weight_shape, offset_dim=0)
    strategies.append([embedding_partial, Shard(0), embedding_partial])

    # batch dim sharding: weight replicated, indices shard on any dim, output follows
    for i in range(len(indices_shape)):
        strategies.append([Shard(i), Replicate(), Shard(i)])

    return strategies


@register_single_dim_strategy(
    aten.embedding_dense_backward.default,
    schema_info=RuntimeSchemaInfo(static_argnum=2),
)
def embedding_dense_backward_strategy(
    op: OpOverload,
    args_schema: ArgsType,
    kwargs_schema: KwargsType,
) -> list[list[Placement]]:
    """Single-dim strategy for embedding backward.

    Placement order: [output(weight_grad), grad_out, indices]
    """
    grad_out_meta = args_schema[0]
    indices_meta = args_schema[1]
    if not isinstance(grad_out_meta, TensorMeta) or not isinstance(
        indices_meta, TensorMeta
    ):
        raise AssertionError

    grad_out_ndim = len(grad_out_meta.shape)
    indices_shape = indices_meta.shape

    strategies: list[list[Placement]] = []

    # colwise backward: weight grad shard on dim 1, grad_out shard on last dim, indices replicate
    strategies.append([Shard(1), Shard(grad_out_ndim - 1), Replicate()])

    # batch dim sharding: weight grad partial, grad_out/indices shard on same dim
    for i in range(len(indices_shape)):
        strategies.append([Partial(), Shard(i), Shard(i)])

    # grad_out partial, indices replicate, weight grad partial
    strategies.append([Partial(), Partial(), Replicate()])

    return strategies
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
from dataclasses import dataclass

import torch


@dataclass
class MaskBuffer:
    data: torch.Tensor | None = None
    # refcount allows shared usage of the MaskBuffer, as long as all users have the same data
    refcount: int = 0

    def materialize_mask(self, mask):
        if self.refcount == 0:
            self.data = mask
        else:
            if self.data is None:
                raise AssertionError
            if not torch.equal(self.data, mask):
                raise RuntimeError(
                    "MaskBuffer has been materialized with conflicting data"
                )
        self.refcount += 1

    def release_mask(self):
        if self.refcount == 0 or self.data is None:
            raise RuntimeError("MaskBuffer has not been materialized")
        self.refcount -= 1
        if self.refcount == 0:
            self.data = None

    def apply_mask(self, tensor):
        if self.refcount == 0 or self.data is None:
            raise RuntimeError("MaskBuffer has not been materialized")

        # NOTE: MaskPartial is being used by the embedding op and the gather op.
        # For gather, the mask has the same dimension as the output tensor, whereas
        # the output of the embedding op has an additional dimension compare to the input,
        # hence the output masking logic below having two different cases.
        if tensor.ndim == self.data.ndim:
            tensor[self.data] = 0.0
        else:
            tensor[self.data, :] = 0.0
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates
import string
from typing import cast

import torch
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import OpSchema, OutputSharding
from torch.distributed.tensor._ops.utils import prod
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset


def _replace_char_in_str(string: str, new_char: str, idx: int) -> str:
    return string[:idx] + new_char + string[idx + 1 :]


def _gen_reshard_suggestions(
    op_schema: OpSchema,
    input_dims: list[str],
    input_specs: tuple[DTensorSpec, ...],
    dim_to_sharding: dict[str, int],
    pending_sum: list[int],
) -> OutputSharding:
    suggested_arg_specs: list[DTensorSpec] = []
    for input_dim, input_spec in zip(input_dims, input_specs):
        dim_map = [dim_to_sharding[dim] for dim in input_dim]
        suggested_arg_specs.append(
            DTensorSpec.from_dim_map(
                mesh=input_spec.mesh,
                dim_map=dim_map,
                sums=pending_sum,
                tensor_meta=input_spec.tensor_meta,
            )
        )
    suggested_schema = OpSchema(op_schema.op, tuple(suggested_arg_specs), {})
    suggested_schema._inplace_rewrap_schema_suggestion(op_schema)
    return OutputSharding(
        None,
        redistribute_schema=suggested_schema,
    )


def einop_rule(
    equation: str,
    op_schema: OpSchema,
    *,
    linearity: bool = False,
    enforce_sharding: dict[str, int] | None = None,
) -> OutputSharding:
    """
    Propagate the sharding of inputs to output for ops whose data moves according to einsum notation.

    This is mostly borrowed from @zdevito's sharding simulator. Examples:
        mk,kn->mn - einsum
        ij,ij->ij - addition
        ij,j->ij - broadcasted addition
        ij->i - reduction
    Other ops could use this propagation algorithm when applied, note
    that einsum propagation only deal with list of specs (DTensor specs)
    as it only works on list of tensors!

    linearity in einop_rule means that the calling op `f` follows this rule:
        f(a + b) = f(a) + f(b)

    In this case we can propagate the partial sum, note that linearity in einop
    only applies to partial sum, not other operations like min/max (which are
    associative but not linear).
    """
    # parse einop equation and extract arg specs
    inputs, outputs = equation.split("->")
    input_dims, output_dims = inputs.split(","), outputs.split(",")
    input_specs = op_schema.args_spec
    # NOTE: only support single output unless needed in future
    output_dim = output_dims[0]

    dim_to_sharding: dict[str, int] = {}
    dim_to_size: dict[str, int] = {}
    # record pending sum, key is mesh dimension, value is pending sum
    # counter across input specs
    pending_sums_counter: dict[int, int] = {}
    seen_shardings: dict[int, str] = {}
    needs_reshard = False

    def merge_sharding(dim: str, a: int, b: int) -> int:
        # merge the sharding of inputs if it's able to merge, i.e. we can merge
        # replicate and shard to shard, but this will trigger an reshard operation
        if a != b:
            if a == -1 or b == -1:
                # reshard the replicate to match the sharded one
                nonlocal needs_reshard
                needs_reshard = True
                return a if a != -1 else b
            else:
                # TODO: further merge the sharding properly (i.e. reshard one input to replicate)
                raise RuntimeError(
                    f"{equation}: dim {dim} sharded two different ways: {a} and {b}"
                )
        else:
            return a

    for input_dim, input_spec in zip(input_dims, input_specs):
        # deal with partial sums
        input_sums = input_spec.sums
        for sum_dim in input_sums:
            if sum_dim not in pending_sums_counter:
                seen_shardings[sum_dim] = "+"
            # update pending sum counter for pending sum mesh
            # dimension with the occurrence from each input
            pending_sums_counter[sum_dim] = pending_sums_counter.get(sum_dim, 0) + 1

        for idx, (dim, mesh_dim) in enumerate(zip(input_dim, input_spec.dim_map)):
            if enforce_sharding and dim in enforce_sharding:
                if enforce_sharding[dim] != mesh_dim:
                    needs_reshard = True
                dim_to_sharding[dim] = enforce_sharding[dim]
                dim_to_size[dim] = input_spec.shape[idx]
            elif dim not in dim_to_sharding:
                dim_to_sharding[dim] = mesh_dim
                dim_to_size[dim] = input_spec.shape[idx]
            else:
                dim_to_sharding[dim] = merge_sharding(
                    dim, dim_to_sharding[dim], mesh_dim
                )
                if dim_to_size[dim] != input_spec.shape[idx]:
                    raise AssertionError

            # after merging sharding, we check if there're multiple
            # sharding on the same mesh dim.
            merged_sharding_for_dim = dim_to_sharding[dim]
            if merged_sharding_for_dim != -1:
                if (
                    merged_sharding_for_dim in seen_shardings
                    and dim != seen_shardings[merged_sharding_for_dim]
                ):
                    needs_reshard = True
                    seen_shardings[merged_sharding_for_dim] += dim
                else:
                    seen_shardings[merged_sharding_for_dim] = dim

    if pending_sums_counter and not linearity:
        # return reshard suggestion with no pending sum, because we already properly
        # merge the sharding, this reshard suggestion is legit to use
        return _gen_reshard_suggestions(
            op_schema, input_dims, input_specs, dim_to_sharding, []
        )
    else:
        # It's an op that supports linearity, but not all input arguments are partial
        # we fail the sharding propagation with suggestion to make all inputs be
        # partial on the corresponding mesh dim (all inputs should be partial for
        # the mesh dims in order to execute locally and delay the sum reduction)
        for value in pending_sums_counter.values():
            if value != len(input_specs):
                needs_reshard = True

    for mesh_dim, dims in seen_shardings.items():
        if len(dims) > 1:
            # we found different input dims are being sharded on the same mesh dim
            # in order to perform local op computation, we need to reshard inputs
            # base on some simple heuristics, now we simply pick the one with least comm
            # volume. (i.e. the input with least size)
            # TODO: consider a more advanced heuristic to pick the best sharding
            costs = []
            for d in dims:
                cost = 0
                for input_dim, input_spec in zip(input_dims, input_specs):
                    if (
                        d in input_dim
                        and input_spec.dim_map[input_dim.index(d)] == mesh_dim
                    ):
                        if input_spec.tensor_meta is None:
                            raise AssertionError
                        global_shape = input_spec.tensor_meta.shape
                        local_shape, _ = compute_local_shape_and_global_offset(
                            global_shape,
                            input_spec.mesh,
                            input_spec.placements,
                            skip_offset=True,
                        )
                        cost += prod(local_shape) * input_spec.mesh.size(mesh_dim)

                costs.append(cost)
            d_to_keep_sharding = dims[costs.index(max(costs))]
            for d in dims:
                # update dim_to_sharding to keep the sharding of the dim with
                # highest comm and make the rest of the dims to replicate
                if d != d_to_keep_sharding:
                    dim_to_sharding[d] = -1

    pending_sums = list(pending_sums_counter.keys())
    if needs_reshard:
        return _gen_reshard_suggestions(
            op_schema, input_dims, input_specs, dim_to_sharding, pending_sums
        )

    # generate output pending sum if a dim is sharded, and it appears in input
    # but not output
    for dim, shard_on_mesh in dim_to_sharding.items():
        if dim not in output_dims[0] and shard_on_mesh != -1:
            pending_sums.append(shard_on_mesh)

    # if no need to reshard, we directly generate the output sharding
    output_dim_map = []
    output_shape = []
    for dim in output_dim:
        if dim == "1":
            # find output dim that is a singleton dimension, mark sharding and shape
            output_dim_map.append(-1)
            output_shape.append(1)
        else:
            output_dim_map.append(dim_to_sharding[dim])
            output_shape.append(dim_to_size[dim])

    # XXX: since we still need to have intermediate shape calculation, we need
    # to pass in the shape here. We should remove this once sharding decomp works
    # for ops like addmm
    if input_specs[0].tensor_meta is None:
        raise AssertionError
    tensor_meta = TensorMeta(
        torch.Size(output_shape),
        input_specs[0].tensor_meta.stride,
        input_specs[0].tensor_meta.dtype,
    )
    return OutputSharding(
        DTensorSpec.from_dim_map(
            input_specs[0].mesh,
            output_dim_map,
            pending_sums,
            tensor_meta=tensor_meta,
        )
    )


def pointwise_rule(op_schema: OpSchema, linearity: bool = False) -> OutputSharding:
    """
    Propagate the sharding for pointwise operations.

    Examples:
        ij,ij->ij - addition/mul
        ij,j->ij - broadcasted addition
    """
    alphabet = string.ascii_lowercase
    # find the max_dim first in case we need to broadcasting
    input_specs = op_schema.args_spec
    max_dim = max(input.ndim for input in input_specs)
    dimchars = []
    singleton_counter: list[int] = [0] * max_dim
    for input in input_specs:
        start_dim = max_dim - input.ndim
        p = alphabet[start_dim:max_dim]
        # handle the "broadcasting to a common shape case"
        # see https://pytorch.org/docs/stable/notes/broadcasting.html
        # If any of the dimensions is singleton dimension (i.e. 1).
        # we mark the dim char as a special "1" to distinguish with
        # the non-singleton dimension, so that sharding propagation
        # should just ignore the singleton dimension.
        if len(input_specs) > 1:
            for i in range(max_dim):
                if i < start_dim:
                    # treat the leading miss dim chars as singleton
                    singleton_counter[i] += 1
                elif input.shape[i - start_dim] == 1:
                    # mark singleton dim char as a special "1" in einop rule
                    singleton_counter[i] += 1
                    p = _replace_char_in_str(p, "1", (i - start_dim))

        dimchars.append(p)
    out_dimchars = alphabet[:max_dim]
    # check if we replace the all inputs dim char with singleton dimension,
    # if we replace all inputs, we also need to replace the output dimension.
    for output_dim_idx in range(len(out_dimchars)):
        if singleton_counter[output_dim_idx] == len(input_specs):
            out_dimchars = _replace_char_in_str(out_dimchars, "1", output_dim_idx)

    fmt = f"{','.join(p for p in dimchars)}->{out_dimchars}"

    enforce_sharding: dict[str, int] = {}
    if op_schema.is_inplace_op():
        follow_spec = op_schema.args_spec[0]
        enforce_sharding.update(zip(out_dimchars, follow_spec.dim_map))
    elif op_schema.is_out_variant_op():
        follow_spec = cast(DTensorSpec, op_schema.kwargs_schema["out"])
        enforce_sharding.update(zip(out_dimchars, follow_spec.dim_map))

    return einop_rule(
        fmt,
        op_schema,
        linearity=linearity,
        enforce_sharding=enforce_sharding,
    )
<|endoftext|>#  Copyright (c) Meta Platforms, Inc. and affiliates
import functools
import heapq
import logging
import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import count
from typing import Any, cast, TypeAlias, TypeVar
from typing_extensions import TypeIs

import torch
from torch._ops import OpOverload
from torch.distributed.tensor._collective_utils import (
    _compute_placement_transition_cost,
    MeshTopoInfo,
)
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import (
    ArgsType,
    KwargsType,
    OpSchema,
    OpSpec,
    OpStrategy,
    RuntimeSchemaInfo,
    StrategyType,
    TupleStrategy,
)
from torch.distributed.tensor.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Placement,
    Replicate,
    Shard,
)
from torch.utils._pytree import tree_map_only


logger = logging.getLogger(__name__)


def _is_sharding(p: Placement) -> TypeIs[Shard | _StridedShard]:
    return isinstance(p, (Shard, _StridedShard))


class _ShardingPlaceholder:
    """
    A placeholder for a sharding placement that has a specified tensor dim, but the other
    metadata (e.g. split factor if it's a StridedShard) will be filled in later.
    """

    dim: int

    def __init__(self, dim: int):
        self.dim = dim

    def __repr__(self) -> str:
        return f"_ShardingPlaceholder(dim={self.dim})"


_StrategyTypeT = TypeVar("_StrategyTypeT", bound=StrategyType)
_PlacementT = TypeVar("_PlacementT", bound=Placement)
_ShardingPlaceholderT = TypeVar("_ShardingPlaceholderT", bound=_ShardingPlaceholder)
_SingleDimStrategyFunc: TypeAlias = Callable[
    [OpOverload, ArgsType, KwargsType], list[list[_PlacementT | _ShardingPlaceholderT]]
]
_ExpandedSingleDimStrategyFunc: TypeAlias = Callable[
    [OpOverload, ArgsType, KwargsType], _StrategyTypeT
]
_FullMeshStrategyFilter: TypeAlias = Callable[
    [
        DeviceMesh,
        OpSchema,
        list[DTensorSpec],
        DTensorSpec | tuple[DTensorSpec | None, ...],
    ],
    bool,
]


@dataclass
class _SingleDimStrategyInfo:
    func: _SingleDimStrategyFunc
    allow_unbacked_sharding: bool | None = field(default=None)
    allow_uneven_sharding: bool = field(default=False)
    full_mesh_strategy_filter: _FullMeshStrategyFilter | None = field(default=None)
    # Positions (in args_schema) of args that may live on a different mesh
    # than the op's compute mesh.  These args must be Replicate.
    # See Note [Multi-mesh args] in expand_to_full_mesh_op_strategy.
    different_mesh_args: list[int] | None = field(default=None)

    # Delegate to func so this can be used interchangeably with a raw
    # _SingleDimStrategyFunc (e.g. in tests that call strategy functions directly).
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def _insert_single_dim_replication_strategy(
    single_dim_strategies_with_placeholders: list[
        list[Placement | _ShardingPlaceholder | None]
    ],
    num_outputs: int,
    num_input_tensors: int,
    output_tensor_meta: TensorMeta | Sequence[TensorMeta | None] | None = None,
) -> list[list[Placement | _ShardingPlaceholder | None]]:
    """
    Inserts the [Replicate(), Replicate(), ...] strategy after asserting that such strategy does not yet exist.
    For ops with masked-off outputs (e.g. backward ops with output_mask), output positions
    where output_tensor_meta is None are set to None in the all-Replicate rule.
    Output positions that are explicitly None in every strategy are also preserved
    as None in the inserted rule.
    """
    for strategy in single_dim_strategies_with_placeholders:
        if all(isinstance(p, Replicate) or p is None for p in strategy):
            return single_dim_strategies_with_placeholders
    total_len = num_outputs + num_input_tensors
    replicate_rule: list[Placement | _ShardingPlaceholder | None] = [
        Replicate()
    ] * total_len
    # Set None for masked-off output positions based on output_tensor_meta
    if isinstance(output_tensor_meta, Sequence) and not isinstance(
        output_tensor_meta, TensorMeta
    ):
        for i, meta in enumerate(output_tensor_meta):
            if meta is None and i < num_outputs:
                replicate_rule[i] = None
    for i in range(num_outputs):
        if single_dim_strategies_with_placeholders and all(
            strategy[i] is None for strategy in single_dim_strategies_with_placeholders
        ):
            replicate_rule[i] = None
    single_dim_strategies_with_placeholders.insert(0, replicate_rule)
    return single_dim_strategies_with_placeholders


def _fill_single_dim_strategy_placeholders(
    unique_input_placements: set[Placement],
    single_dim_strategies_with_placeholders: list[
        list[Placement | _ShardingPlaceholder | None]
    ],
) -> list[list[Placement | None]]:
    """
    Replace any _ShardingPlaceholder with the specific Sharding types used by the inputs in op_schema.
    Supports implicit replication.

    Example:
    single_dim_strategies_with_placeholders = [[Partial(), _ShardingPlaceholder(1), _ShardingPlaceholder(0)]]
    input0: Shard(0)
    input1: StridedShard(1, split_factor=2)
    returns: [
       [Partial(), Shard(1), Shard(0)],
       [Partial(), StridedShard(1, split_factor=2), StridedShard(0, split_factor=2)],
       [Replicate(), Replicate(), Replicate()]
    ]
    """
    shard_builders: dict[str, Callable[[int], Placement]] = {}
    for placement in unique_input_placements:
        if isinstance(placement, _StridedShard):
            key = f"StridedShard(sf={placement.split_factor})"
            if key not in shard_builders:
                shard_builders[key] = functools.partial(
                    _StridedShard, split_factor=placement.split_factor
                )
        elif isinstance(placement, Shard):
            key = "Shard()"
            if key not in shard_builders:
                shard_builders[key] = lambda tensor_dim: Shard(tensor_dim)

    # if any of the placements is a placeholder, we need to expand the strategy
    # to all possible combinations of placements
    expanded_strategies_over_one_mesh_dim: list[list[Placement | None]] = []
    for s in single_dim_strategies_with_placeholders:
        if any(isinstance(p, _ShardingPlaceholder) for p in s):
            for shard_builder in shard_builders.values():
                expanded_strategy: list[Placement | None] = []
                for maybe_placeholder in s:
                    if isinstance(maybe_placeholder, _ShardingPlaceholder):
                        # we combine the tensor dim to shard from the placeholder
                        # with other metadata (e.g. split_factor) from the sharding class
                        expanded_strategy.append(shard_builder(maybe_placeholder.dim))
                    elif maybe_placeholder is None:
                        expanded_strategy.append(None)
                    else:
                        if not isinstance(maybe_placeholder, Placement):
                            raise AssertionError
                        expanded_strategy.append(maybe_placeholder)
                expanded_strategies_over_one_mesh_dim.append(expanded_strategy)
        else:
            if not all(isinstance(p, Placement) or p is None for p in s):
                raise AssertionError
            expanded_strategies_over_one_mesh_dim.append(
                cast(list[Placement | None], (s))
            )

    return expanded_strategies_over_one_mesh_dim


def _get_unique_placements(op_schema: OpSchema) -> set[Placement]:
    unique_placements = set()

    def _update_placements(obj: Any):
        if isinstance(obj, DTensorSpec):
            unique_placements.update(obj.placements)
        elif isinstance(obj, OpStrategy):
            if len(obj.strategies) != 1:
                raise AssertionError
            unique_placements.update(obj.strategies[0].output_spec.placements)
        elif isinstance(obj, TupleStrategy):
            for child in obj.children:
                _update_placements(child)
        elif isinstance(obj, (list, tuple)):
            for child in obj:
                _update_placements(child)

    for obj in op_schema.args_schema:
        _update_placements(obj)
    # Also include placements from kwargs (e.g., "out" tensor)
    for obj in op_schema.kwargs_schema.values():
        _update_placements(obj)

    return unique_placements


def _get_num_tensor_inputs(op_schema: OpSchema) -> int:
    num_inputs = 0

    def _count(obj: Any) -> int:
        if isinstance(obj, OpStrategy):
            return 1
        elif isinstance(obj, TupleStrategy):
            return sum(1 for child in obj.children if child is not None)
        elif isinstance(obj, (list, tuple)):
            return sum(_count(child) for child in obj)
        return 0

    for obj in op_schema.args_schema:
        num_inputs += _count(obj)
    # Also count tensor kwargs (e.g., "out" for out-variant ops)
    for obj in op_schema.kwargs_schema.values():
        num_inputs += _count(obj)
    return num_inputs


def _build_output_specs(
    mesh: DeviceMesh,
    per_mesh_dim_placements: list[tuple[Placement | None, ...]],
    num_outputs: int,
    output_metas: tuple[TensorMeta | None, ...],
) -> DTensorSpec | tuple[DTensorSpec | None, ...]:
    """Build output spec(s) by transposing per-mesh-dim placements to per-output.

    per_mesh_dim_placements is indexed [mesh_dim][output_idx]. output_metas must
    have exactly num_outputs elements. Outputs where output_metas[i] is None
    (masked-off outputs) produce None specs.
    """
    if num_outputs <= 0:
        raise AssertionError(f"Expected num_outputs > 0, got {num_outputs}")
    if len(output_metas) != num_outputs:
        raise AssertionError(
            f"Expected {num_outputs} output_metas, got {len(output_metas)}"
        )

    def _spec_for_output(out_idx: int) -> DTensorSpec | None:
        maybe_placements = tuple(out[out_idx] for out in per_mesh_dim_placements)
        # All mesh dims agreeing on None means the whole output has no
        # DTensorSpec, e.g. local scalar SDPA philox outputs.
        if all(placement is None for placement in maybe_placements):
            return None
        if any(placement is None for placement in maybe_placements):
            raise AssertionError(
                f"Output {out_idx} mixes None with placements across mesh dims: "
                f"{maybe_placements}"
            )
        if output_metas[out_idx] is None:
            return None
        placements = tuple(cast(Placement, placement) for placement in maybe_placements)
        return DTensorSpec(mesh, placements, tensor_meta=output_metas[out_idx])

    if num_outputs > 1:
        return tuple(_spec_for_output(i) for i in range(num_outputs))
    else:
        spec = _spec_for_output(0)
        if spec is None:
            raise AssertionError("Single-output op cannot have None output meta")
        return spec


class _PreparedSingleDimStrategy:
    """A single-dim strategy materialized for a specific op.

    Expands a strategy function's placeholder-based rules into concrete
    placement rules by filling in the actual shard/partial placements from
    the op_schema. The result is a lookup table (strategy_lookup) that maps
    input placements to output placements for one mesh dimension.

    Provides try_propagate() for matching a multi-dim placement tuple against
    the per-dim rules, and exposes allowed_sharding_per_input /
    allowed_partial_per_input for graph search neighbor generation.
    """

    strategy_lookup: dict[tuple[Placement | None, ...], tuple[Placement | None, ...]]
    expanded_strategies: list[list[Placement | None]]
    num_outputs: int
    num_inputs: int
    output_metas: tuple[TensorMeta | None, ...]
    allowed_sharding_per_input: dict[int, set[Shard | _StridedShard]]
    allowed_partial_per_input: dict[int, set[Placement]]
    allow_unbacked_sharding: bool | None

    # many, but not all ops are able to support unevenly sharded tensors
    # there are existing BC expectations even if we wanted to ban for
    # simplicity, see why justification for why pointwise_ops always work
    # with uneven sharding at
    # https://github.com/pytorch/pytorch/pull/174874#issuecomment-3995152777
    allow_uneven_sharding: bool

    def __init__(
        self,
        strategy_fn: _SingleDimStrategyInfo
        | Callable[
            [OpOverload, ArgsType, KwargsType],
            list[list[Placement | _ShardingPlaceholder]],
        ],
        op_schema: OpSchema,
        output_tensor_meta: TensorMeta | Sequence[TensorMeta | None] | None,
        num_inputs: int | None = None,
    ) -> None:
        # Note: circular import
        from torch.distributed.tensor.placement_types import Partial

        if isinstance(strategy_fn, _SingleDimStrategyInfo):
            self.allow_unbacked_sharding = strategy_fn.allow_unbacked_sharding
            self.allow_uneven_sharding = strategy_fn.allow_uneven_sharding
            self.full_mesh_strategy_filter = strategy_fn.full_mesh_strategy_filter
            different_mesh_args = strategy_fn.different_mesh_args
            func = strategy_fn.func
        else:
            self.allow_unbacked_sharding = None
            self.allow_uneven_sharding = False
            self.full_mesh_strategy_filter = None
            different_mesh_args = None
            func = strategy_fn

        # Determine element_mesh from the first OpStrategy arg.  For foreach
        # per-element schemas the element's inputs may live on a smaller
        # sub-mesh than the global compute_mesh.
        self.element_mesh: DeviceMesh | None = None
        for arg in op_schema.args_schema:
            if isinstance(arg, OpStrategy):
                self.element_mesh = arg.strategies[0].output_spec.mesh
                break

        # Validate that all inputs are on the same mesh (except
        # different_mesh_args which are explicitly allowed to differ).
        if self.element_mesh is not None:
            allowed = set(different_mesh_args or [])
            for i, arg in enumerate(op_schema.args_schema):
                if isinstance(arg, OpStrategy) and i not in allowed:
                    arg_mesh = arg.strategies[0].output_spec.mesh
                    if arg_mesh != self.element_mesh:
                        raise ValueError(
                            f"Cannot run {op_schema.op} on inputs with different "
                            f"meshes: got {self.element_mesh} and {arg_mesh}"
                        )

        # Remap different_mesh_args from args_schema positions to
        # OpStrategy-only positions.  Non-OpStrategy args (e.g. empty lists)
        # are filtered out by expand_to_full_mesh_op_strategy, shifting later
        # indices.
        self.remapped_different_mesh_args: list[int] | None = None
        if different_mesh_args is not None:
            schema_to_strategy: dict[int, int] = {}
            strategy_pos = 0
            for schema_pos, arg in enumerate(op_schema.args_schema):
                if isinstance(arg, OpStrategy):
                    schema_to_strategy[schema_pos] = strategy_pos
                    strategy_pos += 1
            self.remapped_different_mesh_args = [
                schema_to_strategy[i]
                for i in different_mesh_args
                if i in schema_to_strategy
            ]

        if num_inputs is None:
            num_inputs = _get_num_tensor_inputs(op_schema)
        self.num_inputs = num_inputs

        # Strategy functions may return None in output positions for masked-off
        # outputs (e.g. backward ops with output_mask). Widen the type here.
        strategies_with_placeholders = cast(
            list[list[Placement | _ShardingPlaceholder | None]],
            func(op_schema.op, op_schema.args_meta, op_schema.kwargs_meta),
        )

        # Validate strategy length against the op outputs. Tuple outputs are
        # aligned with output_tensor_meta so mixed Tensor/non-Tensor returns
        # keep their None slots for wrapping the real result tuple.
        if len(strategies_with_placeholders) > 0:
            if isinstance(output_tensor_meta, Sequence) and not isinstance(
                output_tensor_meta, TensorMeta
            ):
                schema_num_outputs = len(output_tensor_meta)
            elif any(
                isinstance(r.type, torch.ListType) for r in op_schema.op._schema.returns
            ):
                schema_num_outputs = len(strategies_with_placeholders[0]) - num_inputs
            else:
                schema_num_outputs = sum(
                    1 for r in op_schema.op._schema.returns if "Tensor" in str(r.type)
                )
            expected_len = schema_num_outputs + num_inputs
            actual_len = len(strategies_with_placeholders[0])
            if actual_len != expected_len:
                raise AssertionError(
                    f"Strategy length {actual_len} != expected {expected_len} "
                    f"(schema_outputs={schema_num_outputs} + inputs={num_inputs}) "
                    f"for {op_schema.op}. Strategies must include placements "
                    f"for all outputs, args, and tensor kwargs."
                )

        # Compute num_outputs from strategy structure or output_tensor_meta
        if len(strategies_with_placeholders) > 0:
            num_outputs = len(strategies_with_placeholders[0]) - num_inputs
        elif output_tensor_meta is None:
            num_outputs = 0
        elif isinstance(output_tensor_meta, TensorMeta):
            num_outputs = 1
        else:
            num_outputs = len(output_tensor_meta)
        self.num_outputs = num_outputs

        strategies_with_placeholders = _insert_single_dim_replication_strategy(
            strategies_with_placeholders,
            num_outputs,
            num_inputs,
            output_tensor_meta,
        )

        unique_input_placements = _get_unique_placements(op_schema)
        self.expanded_strategies = _fill_single_dim_strategy_placeholders(
            unique_input_placements, strategies_with_placeholders
        )

        # Build strategy lookup: map input placements -> output placements
        self.strategy_lookup = {}
        for strategy in self.expanded_strategies:
            input_key = tuple(strategy[num_outputs:])
            if input_key not in self.strategy_lookup:
                self.strategy_lookup[input_key] = tuple(strategy[:num_outputs])

        # Precompute allowed placements per input from the expanded rules
        self.allowed_sharding_per_input: dict[int, set[Shard | _StridedShard]] = (
            defaultdict(set)
        )
        self.allowed_partial_per_input: dict[int, set[Placement]] = defaultdict(set)
        for strategy in self.expanded_strategies:
            for input_idx in range(num_inputs):
                p = strategy[num_outputs + input_idx]
                if p is None:
                    continue
                if _is_sharding(p):
                    self.allowed_sharding_per_input[input_idx].add(p)
                elif isinstance(p, Partial):
                    self.allowed_partial_per_input[input_idx].add(p)

        # Resolve output tensor_meta per output index
        if output_tensor_meta is None:
            self.output_metas = (None,) * max(num_outputs, 0)
        elif isinstance(output_tensor_meta, TensorMeta):
            self.output_metas = (output_tensor_meta,)
        else:
            self.output_metas = tuple(output_tensor_meta)

    def try_propagate(
        self,
        mesh: DeviceMesh,
        input_placements: tuple[tuple[Placement, ...], ...],
        input_specs: list[DTensorSpec],
    ) -> OpStrategy | None:
        """Try to match input placements against single-dim strategy rules on every mesh dim.

        Checks whether the given input placements independently match a rule in
        strategy_lookup on each mesh dimension, and that all inputs are shardable
        with those placements. If so, returns an OpStrategy with the matched output
        placements and zero redistribute costs.
        """
        from torch.distributed.tensor._ops.utils import is_tensor_shardable

        selected_output_placements: list[tuple[Placement | None, ...]] = []
        for mesh_dim in range(mesh.ndim):
            input_placements_for_dim = tuple(
                placements[mesh_dim] for placements in input_placements
            )
            output_for_dim = self.strategy_lookup.get(input_placements_for_dim)
            if output_for_dim is not None:
                selected_output_placements.append(output_for_dim)
            else:
                return None

        arg_specs = [
            DTensorSpec(mesh, placements, tensor_meta=input_spec.tensor_meta)
            for placements, input_spec in zip(input_placements, input_specs)
        ]
        if not all(
            is_tensor_shardable(
                spec.tensor_meta.shape,
                spec,
                allow_unbacked_sharding=self.allow_unbacked_sharding,
            )
            or (self.allow_uneven_sharding and input_spec.placements == spec.placements)
            for spec, input_spec in zip(arg_specs, input_specs)
            if spec.tensor_meta is not None
        ):
            return None

        output_spec = (
            _build_output_specs(
                mesh,
                selected_output_placements,
                self.num_outputs,
                self.output_metas,
            )
            if self.num_outputs > 0
            else None
        )
        return OpStrategy(
            [
                OpSpec(
                    output_specs=output_spec,
                    input_specs=arg_specs,
                    redistribute_cost=[[0.0] for _ in input_specs],
                )
            ]
        )


def _has_default_value(arg: Any) -> bool:
    """Return whether a schema argument can be omitted at runtime.

    Foreach schemas sometimes omit trailing scalar args that exist on the scalar
    op only because they have defaults.
    """
    has_default_value = getattr(arg, "has_default_value", None)
    if callable(has_default_value):
        return bool(has_default_value())
    return getattr(arg, "default_value", None) is not None


def _normalize_schema_type(arg: Any, *, list_to_element: bool = False) -> str:
    """Normalize schema type strings for direct and foreach element matching.

    In foreach matching, a `List[Tensor]` argument represents one per-element
    `Tensor` argument to the scalar strategy function.
    """
    type_str = str(arg.type)
    if list_to_element and type_str.startswith("List[") and type_str.endswith("]"):
        return type_str.removeprefix("List[").removesuffix("]")
    return type_str


def _schema_args_match(
    base_args: Sequence[Any],
    candidate_args: Sequence[Any],
    *,
    candidate_lists_are_elements: bool = False,
    allow_base_trailing_defaults: bool = False,
    allow_kwonly_scalar_defaults: bool = False,
) -> bool:
    """Return whether candidate args can safely reuse the base op strategy.

    This is intentionally schema-based so auto-registration does not rely only
    on overload names.
    """
    if len(candidate_args) > len(base_args):
        return False

    for base_arg, candidate_arg in zip(base_args, candidate_args):
        if _normalize_schema_type(base_arg) != _normalize_schema_type(
            candidate_arg, list_to_element=candidate_lists_are_elements
        ):
            return False
        if base_arg.kwarg_only != candidate_arg.kwarg_only:
            base_arg_type = _normalize_schema_type(base_arg)
            if not (
                allow_kwonly_scalar_defaults
                and base_arg.kwarg_only
                and not candidate_arg.kwarg_only
                and base_arg_type == "number"
                and _has_default_value(base_arg)
            ):
                return False

    if len(candidate_args) == len(base_args):
        return True

    if not allow_base_trailing_defaults:
        return False

    return all(_has_default_value(arg) for arg in base_args[len(candidate_args) :])


def _op_namespace_and_base_name(op: OpOverload) -> tuple[str, str]:
    """Split an overload into namespace and overload-agnostic base op name."""
    namespace, op_name = op.name().split("::", 1)
    return namespace, op_name.split(".", 1)[0]


def _get_overload_packet(namespace: str, op_name: str) -> Any | None:
    """Look up torch.ops.<namespace>.<op_name> without raising on misses."""
    namespace_packet = getattr(torch.ops, namespace, None)
    if namespace_packet is None:
        return None
    try:
        return getattr(namespace_packet, op_name)
    except AttributeError:
        return None


def _get_packet_overload(packet: Any, overload_name: str) -> OpOverload | None:
    """Look up one overload from an overload packet without raising on misses."""
    if overload_name not in packet.overloads():
        return None
    return cast(OpOverload, getattr(packet, overload_name))


def _resolve_foreach_elementwise_overload(foreach_op: OpOverload) -> OpOverload | None:
    """Map an elementwise foreach overload to the scalar op for one element.

    Single-dim strategy functions reason about one Tensor at a time; foreach
    expansion calls the scalar strategy once for each list element.
    """
    namespace, foreach_base_name = _op_namespace_and_base_name(foreach_op)
    if namespace != "aten":
        return None
    if foreach_base_name.startswith("_foreach_"):
        base_op_name = foreach_base_name.removeprefix("_foreach_")
    elif foreach_base_name.startswith("_amp_foreach_"):
        base_op_name = foreach_base_name.removeprefix("_amp_foreach_")
    else:
        return None

    base_op_name = base_op_name.removesuffix("_")
    base_packet = _get_overload_packet(namespace, base_op_name)
    if base_packet is None:
        return None

    foreach_args = foreach_op._schema.arguments
    matches: list[tuple[int, OpOverload]] = []
    for overload_name in base_packet.overloads():
        if "out" in overload_name:
            continue
        candidate = _get_packet_overload(base_packet, overload_name)
        if candidate is None or candidate._schema.is_mutable:
            continue
        if _schema_args_match(
            candidate._schema.arguments,
            foreach_args,
            candidate_lists_are_elements=True,
            allow_base_trailing_defaults=True,
            allow_kwonly_scalar_defaults=True,
        ):
            # Prefer exact schema matches over matches that rely on defaulted
            # trailing args, e.g. add.Tensor over add.Scalar when possible.
            score = len(candidate._schema.arguments) - len(foreach_args)
            matches.append((score, candidate))

    if not matches:
        return None
    return min(matches, key=lambda item: item[0])[1]


def _expand_single_dim_strategy_to_mesh(
    mesh: DeviceMesh,
    op_schema: OpSchema,
    strategy_info: _SingleDimStrategyInfo,
    output_tensor_meta: TensorMeta | Sequence[TensorMeta | None] | None,
) -> _ExpandedSingleDimStrategyFunc:
    """
    Expands the single_mesh_dim impl across all mesh dims, and expands ShardingPlaceholder into all
    sharding types used by inputs.

    This supports functional correctness but will generate all possible combinations, which is prohibitively expensive
    for larger numbers of mesh dimensions.

    The expanded_strategy function accesses both the args_schema/kwargs_schema, which contains TensorMeta in place of
    tensor arguments, but also the op_schema which contains OpStrategy in place of Tensor args.

    Args:
        output_tensor_meta: tensor metadata for the output(s), precomputed during sharding prop
    """
    # Note: circular import, failed to untangle with #168221, reverted
    from torch.distributed.tensor._ops.utils import expand_to_full_mesh_op_strategy

    def _create_expanded_strategy_impl(
        op_schema: OpSchema,
        output_tensor_meta: TensorMeta | Sequence[TensorMeta | None] | None,
    ) -> Callable[[OpOverload, ArgsType, KwargsType], StrategyType]:
        def expanded_strategy(
            op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
        ) -> StrategyType:
            prepared_strategy = _PreparedSingleDimStrategy(
                strategy_info, op_schema, output_tensor_meta
            )

            # Detect inplace ops by checking if the base op name ends with '_'
            op_name = op.name()
            base_name = op_name.split("::")[1].split(".")[0]
            is_inplace = base_name.endswith("_")

            element_mesh = prepared_strategy.element_mesh or mesh
            full_mesh_strategy_filter = None
            if prepared_strategy.full_mesh_strategy_filter is not None:

                def full_mesh_strategy_filter(
                    input_specs: list[DTensorSpec],
                    output_specs: DTensorSpec | tuple[DTensorSpec | None, ...],
                ) -> bool:
                    if prepared_strategy.full_mesh_strategy_filter is None:
                        raise AssertionError
                    return prepared_strategy.full_mesh_strategy_filter(
                        element_mesh, op_schema, input_specs, output_specs
                    )

            return expand_to_full_mesh_op_strategy(
                element_mesh,
                op_schema,
                prepared_strategy.expanded_strategies,
                output_tensor_meta=output_tensor_meta,
                inplace_op=is_inplace,
                input_index=prepared_strategy.num_outputs,
                allow_unbacked_sharding=prepared_strategy.allow_unbacked_sharding,
                allow_uneven_sharding=prepared_strategy.allow_uneven_sharding,
                full_mesh_strategy_filter=full_mesh_strategy_filter,
                different_mesh_args=prepared_strategy.remapped_different_mesh_args,
            )

        return expanded_strategy

    # Create a cached version of the impl
    _cached_create_expanded_strategy = functools.lru_cache(
        _create_expanded_strategy_impl
    )

    def _create_expanded_strategy(
        op_schema: OpSchema,
        output_tensor_meta: TensorMeta | Sequence[TensorMeta | None] | None,
    ) -> Callable[[OpOverload, ArgsType, KwargsType], StrategyType]:
        # Try to use cache, but fall back to uncached version if hashing fails
        # (e.g., when TensorMeta contains SymInts from dynamic shapes)
        try:
            return _cached_create_expanded_strategy(op_schema, output_tensor_meta)
        except TypeError:
            # Unhashable types (SymInts), skip caching
            return _create_expanded_strategy_impl(op_schema, output_tensor_meta)

    def _translate_list_op_schema(
        op_schema: OpSchema,
        output_tensor_meta: Sequence[TensorMeta] | None,
        index: int,
    ) -> tuple[OpSchema, TensorMeta | None]:
        """Translate foreach/fused op to per-element version of schema."""
        # select per-element inputs, outputs
        target_args, target_kwargs = tree_map_only(
            TupleStrategy,
            lambda x: x.children[index],
            (op_schema.args_schema, op_schema.kwargs_schema),
            is_leaf=lambda x: isinstance(x, TupleStrategy),
        )
        # For inplace ops, output_tensor_meta is None
        target_output_meta = (
            output_tensor_meta[index] if output_tensor_meta is not None else None
        )

        target_op = _resolve_foreach_elementwise_overload(op_schema.op)
        if target_op is None:
            # Fused ops or unknown foreach-like ops: keep original op, no translation.
            target_op = op_schema.op

        op_schema = OpSchema(
            target_op,  # type: ignore[arg-type]
            args_schema=tuple(target_args),
            kwargs_schema=op_schema.kwargs_schema,
        )
        return op_schema, target_output_meta

    def expanded_foreach_strategy(
        op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
    ) -> StrategyType:
        tensorlist_len: int | None = None
        for i, obj in enumerate(op_schema.args_schema):
            if isinstance(obj, TupleStrategy):
                if tensorlist_len is None:
                    tensorlist_len = len(obj.children)
                elif len(obj.children) != tensorlist_len:
                    raise AssertionError(
                        f"Expected {tensorlist_len} children in index {i}, but found {len(obj.children)}."
                    )

        if tensorlist_len is None:
            raise AssertionError("Must have at least one tuple input to a foreach op")

        child_strategies: list[StrategyType] = []
        for tensorlist_i in range(tensorlist_len):
            per_index_schema, per_index_output_meta = _translate_list_op_schema(
                op_schema,
                output_tensor_meta,  # type: ignore[arg-type]
                tensorlist_i,
            )
            per_index_strategy = _create_expanded_strategy(
                per_index_schema, per_index_output_meta
            )
            child_strategies.append(
                per_index_strategy(
                    op, per_index_schema.args_meta, per_index_schema.kwargs_meta
                )
            )

        return TupleStrategy(children=child_strategies)

    # TODO maybe this could be helped by adding a new 'tag' to the OpOverload?
    # Only use the foreach path if the op has TupleStrategy inputs (i.e., actual
    # list-of-tensor args). The name prefix alone is insufficient because ops like
    # _fused_rms_norm share the "_fused_" prefix but are not foreach/fused-optimizer ops.
    op_name = op_schema.op.name()
    has_tuple_strategy = any(
        isinstance(arg, TupleStrategy) for arg in op_schema.args_schema
    )
    if has_tuple_strategy and op_name.startswith(
        ("aten::_foreach_", "aten::_amp_foreach_", "aten::_fused_")
    ):
        return expanded_foreach_strategy

    return _create_expanded_strategy(op_schema, output_tensor_meta)


def register_single_dim_strategy(
    op: torch._ops.OpOverload | list[torch._ops.OpOverload],
    schema_info: RuntimeSchemaInfo | None = None,
    allow_unbacked_sharding: bool | None = None,
    allow_uneven_sharding: bool = False,
    full_mesh_strategy_filter: _FullMeshStrategyFilter | None = None,
    different_mesh_args: list[int] | None = None,
) -> Callable[[_SingleDimStrategyFunc], _SingleDimStrategyFunc]:
    """
    Registers a single_dim_strategy function for the given op.

    A single_dim_strategy enumerates all the non-trivial sharding specifications for the operator over a single mesh
    dim. Since it generates the full set of valid shardings regardless of the given input placements, tensor inputs
    are represented via TensorMetas instead of OpStrategies like in op_strategy functions. TensorMeta inputs, along
    with all other op inputs (int, float, bool, etc) inform which sharding placements are valid. Single-dim strategies
    are fed into infra that expands them over multiple mesh dims and fills sharding placeholders with concrete sharding
    types.

    Single-dim-strategies should not list the trivial "all Replicate" rule, which is assumed for all ops.

    Sharding placeholders should be used to represent generic sharding rules in single_dim_strategies.  For example,
    most operators that support sharded inputs and outputs do not care how the shards are laid out globally, as long as
    they are consistent.  It is just as valid to run a matmul on (Shard(0), Replicate -> Shard(0)) as on
    (StridedShard(0), Replicate -> StridedShard(0)).  For this reason, we use a 'ShardingPlaceholder' to represent all
    generic sharding types.  Placeholders will be filled by concrete sharding types seen in runtime inputs, not all
    types known to DTensor.

    """
    # Note: circular import, failed to untangle with #168221, reverted
    from torch.distributed.tensor._api import DTensor
    from torch.distributed.tensor._ops.utils import _get_registration_wrapper

    # For every ATen op that accepts any args in this list,
    # the arg itself can impact the strides (and potentially the sharding strategy)
    # of the output tensor.
    # thus, we will detect ATen schemas with any of these args and ensure
    # that they get specialized here.
    arg_names_that_require_specializing_cache_strategy = [
        "memory_format",
    ]
    registration_wrapper = _get_registration_wrapper(
        DTensor._op_dispatcher.sharding_propagator.register_single_dim_op_strategy,
        op,
        schema_info,
        arg_names_that_require_specializing_cache_strategy,
    )

    # Wrap impl in _SingleDimStrategyInfo here rather than adding a generic
    # transform hook to _get_registration_wrapper, so that single-dim-strategy
    # concerns stay in this module and the shared registration util stays simple.
    def wrapper(impl):
        info = _SingleDimStrategyInfo(
            func=impl,
            allow_unbacked_sharding=allow_unbacked_sharding,
            allow_uneven_sharding=allow_uneven_sharding,
            full_mesh_strategy_filter=full_mesh_strategy_filter,
            different_mesh_args=different_mesh_args,
        )
        registration_wrapper(info)
        return impl

    return wrapper


@dataclass(order=True)
class _PQEntry:
    """Priority queue entry for the Dijkstra search in _dijkstra_expand_single_dim_strategy_to_mesh.

    Ordered by (cost, counter) for heap comparison. The counter breaks ties
    in FIFO order so that entries with equal cost are explored in insertion
    order rather than by arbitrary tuple comparison on placements.
    """

    cost: float
    counter: int
    # Per-input placement tuples representing the current search state.
    placements: tuple[tuple[Placement, ...], ...] = field(compare=False)
    # History of (input_idx, mesh_dim, old_placement, new_placement) transitions
    # from the initial state to this state, used for debugging.
    transitions: list[tuple[int, int, Placement, Placement]] = field(compare=False)
    # Accumulated redistribute cost per input (sum of incremental step costs).
    per_input_costs: tuple[float, ...] = field(compare=False)
    # Current communication bytes (in GB) per input, updated as placements change.
    per_input_comm_bytes_gb: tuple[float, ...] = field(compare=False)


def _get_neighbor_placements(
    allowed_sharding: set[Shard | _StridedShard],
    allowed_partial: set[Placement],
    current: Placement,
    input_placements: tuple[Placement, ...],
    mesh_dim: int,
) -> list[Placement]:
    """Return valid one-shot placement transitions for one input on one mesh dim.

    DTensor placements are applied left-to-right, so a tensor dim sharded on
    multiple mesh dims has a specific nesting order. A one-shot collective on
    mesh_dim M can only produce the correct data layout if no mesh dim to the
    RIGHT of M already shards the same tensor dim. For example, going from
    (R, S(0)) to (S(0), S(0)) via local chunk on mesh dim 0 produces a
    strided-shard layout, not the correct left-to-right (S(0), S(0)) layout.

    Transition rules:
    - Replicate -> Shard(d): free local chunk, valid if d not sharded to the right
    - Replicate -> Partial: local view, always valid
    - Shard(d) -> Replicate: allgather, valid if d not sharded to the right
    - Shard(d1) -> Shard(d2): all-to-all, valid if neither d1 nor d2 sharded right
    - Partial -> Replicate: allreduce, always valid
    - Partial -> Shard(d): reduce-scatter, valid if d not sharded to the right

    Shard -> Partial("sum") is intentionally absent. Although explicit
    redistribution supports it, the conversion materializes a global-shape
    local tensor. Because the strategy cost model accounts only for communication,
    we exclude this transition to avoid selecting it without accounting for its
    memory and copy costs.
    """
    # Note: circular import
    from torch.distributed.tensor.placement_types import Partial

    # Tensor dims sharded by mesh dims to the right of this one.
    right_shard_dims: set[int] = set()
    for i in range(mesh_dim + 1, len(input_placements)):
        p = input_placements[i]
        if _is_sharding(p):
            right_shard_dims.add(p.dim)

    neighbors: list[Placement] = []

    if isinstance(current, Replicate):
        neighbors.extend(s for s in allowed_sharding if s.dim not in right_shard_dims)
        neighbors.extend(allowed_partial)

    elif _is_sharding(current):
        cur_dim_ok = current.dim not in right_shard_dims
        if cur_dim_ok:
            neighbors.append(Replicate())
        for s in allowed_sharding:
            if s != current and cur_dim_ok and s.dim not in right_shard_dims:
                neighbors.append(s)

    elif isinstance(current, Partial):
        neighbors.append(Replicate())
        neighbors.extend(s for s in allowed_sharding if s.dim not in right_shard_dims)

    return neighbors


def _dijkstra_expand_single_dim_strategy_to_mesh(
    mesh: DeviceMesh,
    op_schema: OpSchema,
    single_dim_strategy: _SingleDimStrategyInfo
    | Callable[
        [OpOverload, ArgsType, KwargsType], list[list[Placement | _ShardingPlaceholder]]
    ],
    output_tensor_meta: TensorMeta | Sequence[TensorMeta | None] | None = None,
    _collect_all_matches: set[tuple[tuple[Placement, ...], ...]] | None = None,
) -> OpStrategy | None:
    """
    Find the lowest cost sharding for the given op_schema.

    Uses a Dijkstra-like priority-queue search over input placement states. Each
    state is a tuple of per-input placement tuples, and neighbors are generated
    by changing one placement on one mesh dim for one input. The search
    terminates when a state matches a single-dim strategy on every mesh dim.

    This avoids the O(S^N) exhaustive expansion of _expand_single_dim_strategy_to_mesh
    (S = single-dim strategies, N = mesh dims).  Benchmarks with mm on fake
    process groups show:

        1D(4):     S^N=8,   avg 0.2ms
        2D(2,2):   S^N=64,  avg 2.3ms
        3D(2,2,2): S^N=512, avg 41ms, worst 392ms

    The step count is small (avg 0.6-2.0 pops) but per-step cost is dominated
    by cost computation.  Each transition computes an incremental cost via
    _compute_placement_transition_cost for the single changed placement, matching
    the per-step costs used by graph-based transform info planning.

    Returns None if any input has _StridedShard placement, signaling the caller
    to fall back to full expansion.

    Args:
        _collect_all_matches: Testing-only. When non-None, exhaustively explores the
            full transition graph, adding every shardable match to the set. Still
            returns the optimal (first) match.
    """
    # Extract input DTensorSpecs from OpStrategy-wrapped args.
    # Fall back for TupleStrategy (e.g. index tensors in index_put) since the PQ
    # search doesn't model variable-length tuple inputs.
    input_specs: list[DTensorSpec] = []
    for arg in op_schema.args_schema:
        if isinstance(arg, OpStrategy):
            if len(arg.strategies) != 1:
                raise AssertionError
            input_specs.append(arg.strategies[0].output_spec)
        elif isinstance(arg, TupleStrategy):
            return None

    # Fall back if any kwargs are tensor inputs — the PQ search only tracks
    # positional tensor args and would miss redistribute costs for kwargs.
    for kwarg in op_schema.kwargs_schema.values():
        if isinstance(kwarg, (OpStrategy, TupleStrategy)):
            return None

    if len(input_specs) == 0:
        raise AssertionError("broken input")
    num_inputs = len(input_specs)

    # Fall back to full expansion if any input has _StridedShard or symbolic shapes
    # (symbolic shapes produce SymFloat costs that can't be compared in the PQ)
    for spec in input_specs:
        if any(isinstance(p, _StridedShard) for p in spec.placements):
            return None
        if spec.tensor_meta is not None and any(
            isinstance(s, torch.SymInt) for s in spec.tensor_meta.shape
        ):
            return None

    prepared_strategy = _PreparedSingleDimStrategy(
        single_dim_strategy, op_schema, output_tensor_meta, num_inputs=num_inputs
    )

    initial_placements = tuple(spec.placements for spec in input_specs)
    first_result: OpStrategy | None = None

    # Fast path: if initial placements already match a strategy, skip search
    fast_result = prepared_strategy.try_propagate(mesh, initial_placements, input_specs)
    if fast_result is not None:
        fast_result._pq_transitions = []  # type: ignore[attr-defined]
        if _collect_all_matches is not None:
            _collect_all_matches.add(initial_placements)
            first_result = fast_result
        else:
            return fast_result

    # Pre-compute mesh topology and per-input comm bytes for cost computation.
    # comm_bytes_gb reflects the local shard size given current placements;
    # it's tracked per PQ entry and updated as placements change.
    mesh_topo = MeshTopoInfo.build_from_mesh(mesh)
    initial_comm_bytes_gb: list[float] = []
    for spec in input_specs:
        if spec.tensor_meta is None:
            raise AssertionError
        total_bytes = spec.tensor_meta.dtype.itemsize * math.prod(
            spec.tensor_meta.shape
        )
        # TODO: is_shard() misses _StridedShard, use spec.num_shards instead.
        # Not fixing yet: the overestimate biases Dijkstra toward redistributing
        # away from _StridedShard, which is the safer default until _StridedShard
        # is fully validated.
        num_shards = 1
        for i, p in enumerate(spec.placements):
            if p.is_shard():
                num_shards *= mesh_topo.mesh_dim_devices[i]
        initial_comm_bytes_gb.append(total_bytes / num_shards / (1024**3))

    pq: list[_PQEntry] = []
    visited: set[tuple[tuple[Placement, ...], ...]] = set()
    next_counter = count()

    initial_per_input_costs = (0.0,) * num_inputs
    initial_per_input_comm_bytes = tuple(initial_comm_bytes_gb)
    heapq.heappush(
        pq,
        _PQEntry(
            0.0,
            next(next_counter),
            initial_placements,
            [],
            initial_per_input_costs,
            initial_per_input_comm_bytes,
        ),
    )

    def _push_neighbor(
        input_idx: int,
        mesh_dim: int,
        new_placement: Placement,
        source: _PQEntry,
    ) -> None:
        new_input_placements = [list(ps) for ps in source.placements]
        old_placement = new_input_placements[input_idx][mesh_dim]
        new_input_placements[input_idx][mesh_dim] = new_placement
        candidate_placements = tuple(tuple(ps) for ps in new_input_placements)
        if candidate_placements in visited:
            return
        # Check that the NET transition (original -> proposed) is feasible.
        # Individual hops may each be valid (e.g. S->R then R->P) while the
        # net redistribution (S->P) is unsupported by the runtime planner.
        original_p = initial_placements[input_idx][mesh_dim]
        net_cost, _ = _compute_placement_transition_cost(
            original_p,
            new_placement,
            mesh_topo,
            mesh_dim,
            initial_comm_bytes_gb[input_idx],
        )
        if net_cost == float("inf"):
            return
        step_cost, new_comm_bytes = _compute_placement_transition_cost(
            old_placement,
            new_placement,
            mesh_topo,
            mesh_dim,
            source.per_input_comm_bytes_gb[input_idx],
        )
        if step_cost == float("inf"):
            return
        changed_cost = source.per_input_costs[input_idx] + step_cost
        new_per_input_costs = (
            source.per_input_costs[:input_idx]
            + (changed_cost,)
            + source.per_input_costs[input_idx + 1 :]
        )
        new_per_input_comm_bytes = (
            source.per_input_comm_bytes_gb[:input_idx]
            + (new_comm_bytes,)
            + source.per_input_comm_bytes_gb[input_idx + 1 :]
        )
        new_cost = sum(new_per_input_costs)
        new_transitions = source.transitions + [
            (input_idx, mesh_dim, old_placement, new_placement)
        ]
        heapq.heappush(
            pq,
            _PQEntry(
                new_cost,
                next(next_counter),
                candidate_placements,
                new_transitions,
                new_per_input_costs,
                new_per_input_comm_bytes,
            ),
        )

    while pq:
        candidate = heapq.heappop(pq)

        if candidate.placements in visited:
            continue
        visited.add(candidate.placements)

        match_result = prepared_strategy.try_propagate(
            mesh, candidate.placements, input_specs
        )
        if match_result is not None:
            # Use pre-computed per-input costs from the PQ search instead of
            # recomputing via generate_redistribute_costs -> _gen_transform_infos.
            match_spec = match_result.strategies[0]
            if match_spec.input_specs is None:
                raise AssertionError
            op_spec = OpSpec(
                output_specs=match_spec.output_specs,
                input_specs=list(match_spec.input_specs),
                redistribute_cost=[[cost] for cost in candidate.per_input_costs],
            )

            exhaustive = len(prepared_strategy.expanded_strategies) ** mesh.ndim
            logger.debug(
                "returning cost=%f %s, visited=%d, exhaustive=%d, transitions=%s",
                candidate.cost,
                op_spec,
                len(visited),
                exhaustive,
                candidate.transitions,
            )
            result = OpStrategy([op_spec])
            result._pq_transitions = candidate.transitions  # type: ignore[attr-defined]
            if _collect_all_matches is not None:
                _collect_all_matches.add(candidate.placements)
                if first_result is None:
                    first_result = result
            else:
                return result

        # Generate neighbor states
        for mesh_dim in range(mesh.ndim):
            for input_idx in range(len(candidate.placements)):
                current_p = candidate.placements[input_idx][mesh_dim]
                for neighbor_p in _get_neighbor_placements(
                    prepared_strategy.allowed_sharding_per_input[input_idx],
                    prepared_strategy.allowed_partial_per_input[input_idx],
                    current_p,
                    candidate.placements[input_idx],
                    mesh_dim,
                ):
                    _push_neighbor(input_idx, mesh_dim, neighbor_p, candidate)

    if _collect_all_matches is not None and first_result is not None:
        return first_result

    logger.warning(
        "Dijkstra search exhausted without finding a valid strategy for "
        "%s on %s (explored %d combinations); falling back to full expansion",
        op_schema,
        mesh,
        len(visited),
    )
    return None
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor

from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OutputSharding,
    RuntimeSchemaInfo,
)
from torch.distributed.tensor._ops.single_dim_strategy import (
    _ShardingPlaceholder,
    register_single_dim_strategy,
)
from torch.distributed.tensor._ops.utils import register_prop_rule
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.fx.experimental.symbolic_shapes import guard_or_false


aten = torch.ops.aten


@register_prop_rule(aten.convolution.default)
def convolution_rules(op_schema: OpSchema) -> OutputSharding:
    (
        input_spec,
        weight_spec,
        bias_spec,
        stride,
        padding,
        dilation,
        _transposed,
        _output_padding,
        _groups,
    ) = op_schema.args_schema

    if not isinstance(input_spec, DTensorSpec):
        raise AssertionError
    if not isinstance(weight_spec, DTensorSpec):
        raise AssertionError
    # bias_spec can be None (optional parameter in aten.convolution schema)
    if bias_spec is not None:
        if not isinstance(bias_spec, DTensorSpec):
            raise AssertionError
    if input_spec.tensor_meta is None:
        raise AssertionError
    if weight_spec.tensor_meta is None:
        raise AssertionError
    in_shape = input_spec.tensor_meta.shape
    weight_shape = weight_spec.tensor_meta.shape
    if not isinstance(stride, list):
        raise AssertionError(f"stride must be list, got {type(stride)}")
    if not isinstance(padding, list):
        raise AssertionError(f"padding must be list, got {type(padding)}")
    if not isinstance(dilation, list):
        raise AssertionError(f"dilation must be list, got {type(dilation)}")
    # weight_shape might not be torch.Size in all cases (e.g., SymIntArrayRef during tracing)
    # so we don't assert its type, just use it
    out_conv_shape = [
        (d + 2 * padding[i] - dilation[i] * (weight_shape[i + 1] - 1) - 1) // stride[i]
        + 1
        for (i, d) in enumerate(in_shape[2:])
    ]
    output_shape = [in_shape[0], weight_shape[0]] + out_conv_shape
    output_stride = [1]
    for i in range(1, len(output_shape)):
        output_stride.insert(0, output_stride[0] * output_shape[-i])
    output_dim_map = input_spec.dim_map
    pending_sums = input_spec.sums

    tensor_meta = TensorMeta(
        torch.Size(output_shape),
        tuple(output_stride),
        input_spec.tensor_meta.dtype,
    )
    return OutputSharding(
        DTensorSpec.from_dim_map(
            input_spec.mesh,
            output_dim_map,
            pending_sums,
            tensor_meta=tensor_meta,
        )
    )


@register_prop_rule(aten.convolution_backward.default)
def convolution_backward_rules(op_schema: OpSchema) -> OutputSharding:
    input_spec = op_schema.args_schema[0]
    (
        grad_output_spec,
        input_spec,
        weight_spec,
        bias_shape_opt,
        _stride,
        _padding,
        _dilation,
        _transposed,
        _output_padding,
        _groups,
        _output_mask,
    ) = op_schema.args_schema

    if not isinstance(grad_output_spec, DTensorSpec):
        raise AssertionError
    if not isinstance(input_spec, DTensorSpec):
        raise AssertionError
    if not isinstance(weight_spec, DTensorSpec):
        raise AssertionError
    # bias_shape_opt can be None (optional parameter in aten.convolution_backward schema)
    if bias_shape_opt is not None:
        if not isinstance(bias_shape_opt, list):
            raise AssertionError
    if input_spec.tensor_meta is None:
        raise AssertionError
    weight_tensor_meta = weight_spec.tensor_meta

    # Only create bias_tensor_meta if bias_shape_opt is not None
    if bias_shape_opt is not None:
        bias_tensor_meta = TensorMeta(
            torch.Size(bias_shape_opt),
            (1,),
            input_spec.tensor_meta.dtype,
        )
    else:
        bias_tensor_meta = None

    grad_input_spec = input_spec
    grad_weight_spec = DTensorSpec.from_dim_map(
        input_spec.mesh,
        [-1, -1, -1, -1],
        [0],
        tensor_meta=weight_tensor_meta,
    )

    # Only create grad_bias_spec if we have bias_tensor_meta
    if bias_tensor_meta is not None:
        grad_bias_spec = DTensorSpec.from_dim_map(
            input_spec.mesh,
            [-1],
            [0],
            tensor_meta=bias_tensor_meta,
        )
    else:
        grad_bias_spec = None

    # TODO: actually the output_mask is not respected here, we should
    # set the corresponding spec to `None` if the output_mask is not `False`
    # for a certain output Tensor. This also applies to the conv handler
    # in torch/distributed/tensor/_tp_conv.py
    return OutputSharding([grad_input_spec, grad_weight_spec, grad_bias_spec])


# Single-dim strategies for autoparallel optimizer support.
# These coexist with the prop_rules above — strategies take precedence
# in the propagation path, while the prop_rules + custom handlers in
# _tp_conv.py continue to handle runtime dispatch.


def _supports_last_dim_sharding(
    input_meta: TensorMeta,
    weight_meta: TensorMeta,
    stride: list[Any],
    padding: list[Any],
    dilation: list[Any],
    transposed: bool,
) -> bool:
    ndim = len(input_meta.shape)
    return (
        ndim == len(weight_meta.shape)
        and ndim in (3, 4, 5)
        and not transposed
        and guard_or_false(padding[-1] == 0)
        and guard_or_false(dilation[-1] == 1)
        and guard_or_false(stride[-1] == weight_meta.shape[-1])
    )


def _convolution_full_mesh_strategy_filter(
    mesh: DeviceMesh,
    op_schema: OpSchema,
    input_specs: list[DTensorSpec],
    _output_specs: DTensorSpec | tuple[DTensorSpec | None, ...],
) -> bool:
    if op_schema.op == aten.convolution.default:
        input_idx, stride_idx = 0, 3
    elif op_schema.op == aten.convolution_backward.default:
        input_idx, stride_idx = 1, 4
    else:
        raise AssertionError(f"Unexpected op {op_schema.op}")

    input_spec = input_specs[input_idx]
    last_dim = len(input_spec.shape) - 1
    last_dim_mesh_dims = [
        mesh_dim
        for mesh_dim, placement in enumerate(input_spec.placements)
        if placement.is_shard(last_dim)
    ]
    if not last_dim_mesh_dims:
        return True
    # _tp_conv converts placements to dim_map, which cannot represent one tensor
    # dimension sharded over multiple mesh dimensions.
    if len(last_dim_mesh_dims) != 1:
        return False

    last_dim_mesh_dim = last_dim_mesh_dims[0]
    last_dim_placement = input_spec.placements[last_dim_mesh_dim]
    if not isinstance(last_dim_placement, Shard) or isinstance(
        last_dim_placement, _StridedShard
    ):
        return False

    stride = op_schema.args_schema[stride_idx]
    if not isinstance(stride, list):
        raise AssertionError(f"Expected stride list, got {type(stride)}")
    return guard_or_false(
        input_spec.shape[last_dim] % (stride[-1] * mesh.size(last_dim_mesh_dim)) == 0
    )


@register_single_dim_strategy(
    [aten.convolution.default],
    schema_info=RuntimeSchemaInfo(2),
    full_mesh_strategy_filter=_convolution_full_mesh_strategy_filter,
)
def convolution_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    weight_meta = args_schema[1]
    if not isinstance(weight_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(weight_meta)}")
    bias_meta = args_schema[2]
    stride, padding, dilation = args_schema[3:6]
    transposed = args_schema[6]
    # [output, input, weight, (bias)]
    rule: list[Placement | _ShardingPlaceholder] = [
        _ShardingPlaceholder(0),  # output
        _ShardingPlaceholder(0),  # input
        Replicate(),  # weight
    ]
    if bias_meta is not None:
        rule.append(Replicate())  # bias
    strategies = [rule]

    if _supports_last_dim_sharding(
        input_meta, weight_meta, stride, padding, dilation, transposed
    ):
        last_dim = len(input_meta.shape) - 1
        last_dim_rule: list[Placement | _ShardingPlaceholder] = [
            Shard(last_dim),  # output
            Shard(last_dim),  # input
            Replicate(),  # weight
        ]
        if bias_meta is not None:
            last_dim_rule.append(Replicate())  # bias
        strategies.append(last_dim_rule)

    return strategies


@register_single_dim_strategy(
    [aten.convolution_backward.default],
    schema_info=RuntimeSchemaInfo(3),
    full_mesh_strategy_filter=_convolution_full_mesh_strategy_filter,
)
def convolution_backward_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder | None]]:
    input_meta = args_schema[1]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    weight_meta = args_schema[2]
    if not isinstance(weight_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(weight_meta)}")
    bias_sizes = args_schema[3]
    stride, padding, dilation = args_schema[4:7]
    transposed = args_schema[7]
    has_bias = bias_sizes is not None
    # outputs: [grad_input, grad_weight, grad_bias]
    # inputs: [grad_output, input, weight]
    rule: list[Placement | _ShardingPlaceholder | None] = [
        _ShardingPlaceholder(0),  # grad_input
        Partial("sum"),  # grad_weight
        Partial("sum") if has_bias else None,  # grad_bias
        _ShardingPlaceholder(0),  # grad_output
        _ShardingPlaceholder(0),  # input
        Replicate(),  # weight
    ]
    strategies = [rule]

    if _supports_last_dim_sharding(
        input_meta, weight_meta, stride, padding, dilation, transposed
    ):
        last_dim = len(input_meta.shape) - 1
        strategies.append(
            [
                Shard(last_dim),  # grad_input
                Partial("sum"),  # grad_weight
                Partial("sum") if has_bias else None,  # grad_bias
                Shard(last_dim),  # grad_output
                Shard(last_dim),  # input
                Replicate(),  # weight
            ]
        )

    return strategies
<|endoftext|>import itertools
from dataclasses import dataclass

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import OpSpec, OpStrategy
from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)


@dataclass
class EinsumDims:
    contracting_dims: list[str]
    batch_dims: list[str]
    lhs_out_only_dims: list[str]
    rhs_out_only_dims: list[str]

    @classmethod
    def parse_equation(cls, equation: str) -> tuple[list[str], str]:
        # parse einop equation and extract arg specs
        """
        Parse the einsum equation str to input dim chars and output dim char
        """
        inputs, outputs = equation.split("->")
        input_dims, output_dims = inputs.split(","), outputs.split(",")

        # NOTE: only support at most two inputs, and single output
        # extend to support more inputs if needed in future
        if len(input_dims) > 2:
            raise AssertionError("Only support at most two inputs")
        if len(output_dims) != 1:
            raise AssertionError("Only support single output")
        output_dim = output_dims[0]
        return input_dims, output_dim

    @classmethod
    def parse_dims(cls, input_dims: list[str], output_dim: str) -> "EinsumDims":
        """
        Parse the dims and extract the contracting, batch, and free dimensions
        for the left and right hand sides.
        """
        dim_char_set: set[str] = set()
        for input_dim in input_dims:
            dim_char_set.update(input_dim)

        # get a deterministic order of all dim chars
        all_dim_chars = sorted(dim_char_set)

        # parse input and output dimensions
        lhs_out_only_dims, rhs_out_only_dims = [], []
        batch_dims, contracting_dims = [], []

        for dim_char in all_dim_chars:
            if dim_char not in output_dim:
                contracting_dims.append(dim_char)
            else:
                is_batch_dim = True
                for input_dim in input_dims:
                    is_batch_dim = is_batch_dim and dim_char in input_dim

                if is_batch_dim:
                    batch_dims.append(dim_char)
                else:
                    if len(input_dims) != 2:
                        raise AssertionError(
                            "free dimension only supported for two inputs!"
                        )
                    lhs, rhs = input_dims
                    if dim_char in lhs:
                        lhs_out_only_dims.append(dim_char)
                    elif dim_char in rhs:
                        rhs_out_only_dims.append(dim_char)
                    else:
                        raise RuntimeError("Invalid dimension character")

        return cls(
            contracting_dims=contracting_dims,
            batch_dims=batch_dims,
            lhs_out_only_dims=lhs_out_only_dims,
            rhs_out_only_dims=rhs_out_only_dims,
        )


def gen_einsum_strategies(
    equation: str,
    mesh: DeviceMesh,
    *,
    linearity: bool = False,
) -> OpStrategy:
    """
    Generate a strategy list for the ops that follow einsum style notation.

    In principle, each mesh dim is independent of other device mesh dim when we
    generate strategies. So we generate strategy over each device mesh dim and
    do product combination on all mesh dims. We basically follow the below rule
    for each device mesh dim:

    1. Shard on contracting dim: When both inputs shard on contracting dim over
       the same device dim. The result will be Partial over that device dim.

    2. Shard on noncontracting dim:
        2.1: Shard on batch dim: output, both inputs all should shard on batch
        dim.
        2.2: Shard on lhs only dim or rhs only dim: both output and lhs or rhs
        input should shard on this free dim.

    3. Linearity (Partial): If enabled, set Partial on output and inputs over
       the same device mesh dim.
    """
    # parse einop equation and extract dims
    input_dims, output_dim = EinsumDims.parse_equation(equation)
    edims = EinsumDims.parse_dims(input_dims, output_dim)
    all_mesh_dim_strategies = []

    # generate strategies for each mesh dim and do cartesian product for final strategy. E.g., for a 2D mesh, we can have [P(),R,R]
    strategies_over_one_mesh_dim = []

    # placement list stores placements of [output, input1, input2, ...]
    # first we always have replicate all for inputs and output
    placement_list: list[Placement] = [Replicate()] * (len(input_dims) + 1)
    strategies_over_one_mesh_dim.append(placement_list)

    # split batch dim
    for batch_dim in edims.batch_dims:
        output_batch_dim = output_dim.index(batch_dim)
        placement_list = [Shard(output_batch_dim)]
        for input_dim in input_dims:
            input_batch_dim = input_dim.index(batch_dim)
            placement_list.append(Shard(input_batch_dim))

        strategies_over_one_mesh_dim.append(placement_list)

    # split contracting dim
    # NOTE: This is the only strategy that produces a Partial output, and it
    # hardcodes Partial("sum"). No strategy accepts Partial as an input
    # placement, so Partial inputs are always redistributed to Shard or
    # Replicate. This means Partial("avg") inputs cannot be preserved through
    # the op. Use gen_single_dim_einsum_strategies with per-input linearity
    # for proper Partial support.
    for contracting_dim in edims.contracting_dims:
        # Contracting dim can shard on same device axis for both inputs. This
        # results in the output being Partial on that device axis. For example:
        # bmk_{x},k_{x}n -> bmn{Ux} (becomes partial over device axis x)
        placement_list = [Partial()]
        for input_dim in input_dims:
            input_contracting_dim = input_dim.index(contracting_dim)
            placement_list.append(Shard(input_contracting_dim))

        strategies_over_one_mesh_dim.append(placement_list)

    # split lhs free dim
    for lhs_dim in edims.lhs_out_only_dims:
        lhs_free_dim_output = output_dim.index(lhs_dim)
        lhs_free_dim_input = input_dims[0].index(lhs_dim)
        # this means split the lhs input and output
        # i.e. S(0), R -> S(0)
        lhs_placement_list: list[Placement] = [
            Shard(lhs_free_dim_output),
            Shard(lhs_free_dim_input),
            Replicate(),
        ]
        strategies_over_one_mesh_dim.append(lhs_placement_list)

    # split rhs free dim
    for rhs_dim in edims.rhs_out_only_dims:
        rhs_free_dim_output = output_dim.index(rhs_dim)
        rhs_free_dim_input = input_dims[1].index(rhs_dim)
        rhs_placement_list: list[Placement] = [
            Shard(rhs_free_dim_output),
            Replicate(),
            Shard(rhs_free_dim_input),
        ]
        strategies_over_one_mesh_dim.append(rhs_placement_list)

    # linearity strategy
    if linearity:
        linearity_placement_list: list[Placement] = [Partial()]
        for _ in input_dims:
            linearity_placement_list.append(Partial())
        strategies_over_one_mesh_dim.append(linearity_placement_list)

    # generate strategies for entire mesh
    all_mesh_dim_strategies = [strategies_over_one_mesh_dim] * mesh.ndim
    strategy_combs = itertools.product(*all_mesh_dim_strategies)
    all_strategies = []
    for strategy_comb in strategy_combs:
        spec_list = [DTensorSpec(mesh, tuple(specs)) for specs in zip(*strategy_comb)]
        strat = OpSpec(output_specs=spec_list[0], input_specs=spec_list[1:])
        all_strategies.append(strat)

    return OpStrategy(all_strategies)
<|endoftext|># mypy: allow-untyped-defs
import copy
import operator
from collections.abc import Sequence
from typing import Any, cast

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.distributed.tensor import DeviceMesh, distribute_tensor, DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpSpec,
    OutputSharding,
    OutputSpecType,
)
from torch.distributed.tensor._redistribute import redistribute_local_tensor
from torch.distributed.tensor.parallel.style import ColwiseParallel, ParallelStyle
from torch.distributed.tensor.placement_types import Placement, Replicate, Shard
from torch.export import ExportedProgram
from torch.export.exported_program import ExportGraphSignature
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.node import Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.utils import _pytree as pytree


__all__ = ["tensor_parallel_transformation"]

aten = torch.ops.aten


def tensor_parallel_transformation(
    exported_program: ExportedProgram,
    rank: int,
    world_size: int,
    device_type: str,
    parallel_strategies: dict[str, ParallelStyle],
) -> ExportedProgram:
    """
    The entry point function to perform graph transformations on an exported program
    to transform a single-device graph into a tensor parallel graph.

    .. warning::
        This API is experimental and subject to change.
    """

    gm = exported_program.graph_module
    sig = copy.deepcopy(exported_program.graph_signature)
    state_dict = copy.copy(exported_program.state_dict)

    with gm._set_replace_hook(sig.get_replace_hook()):
        res = _TensorParallelTransformPass(
            rank,
            world_size,
            device_type,
            state_dict,
            exported_program.graph_signature,
            parallel_strategies,
        )(gm)
        if res is None:
            raise AssertionError
        gm = res.graph_module

    return exported_program._update(gm, sig, state_dict=state_dict)


class _TensorParallelTransformPass(PassBase):
    """
    This pass is responsible for transforming a single-device graph into a tensor parallel
    graph. It will mark the OpSpec of each node in the graph, partition the graph into
    distributed graph, then shard the parameters/buffers accordingly.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        device_type: str,
        state_dict: dict[str, torch.Tensor],
        graph_signature: ExportGraphSignature,
        parallel_strategies: dict[str, ParallelStyle],
    ) -> None:
        super().__init__()
        self.rank = rank
        self.mesh = DeviceMesh(device_type, torch.arange(world_size))
        self.state_dict: dict[str, torch.Tensor] = state_dict
        self.graph_signature = graph_signature
        self.parallel_strategies = parallel_strategies

    def call(self, graph_module) -> PassResult:
        gm = copy.deepcopy(graph_module)

        parameter_placements = _generate_parameter_and_buffer_placements(
            list(self.state_dict.keys()), self.parallel_strategies
        )
        placement_strategies = _mark_sharding(
            gm, self.graph_signature, self.mesh, parameter_placements
        )
        _partitioner(gm)
        _shard_state_dict(
            self.state_dict, placement_strategies, self.graph_signature, self.mesh
        )
        return PassResult(gm, True)


def _generate_parameter_and_buffer_placements(
    params_and_buffers: list[str],
    parallel_strategies: dict[str, ParallelStyle],
) -> dict[str, Placement]:
    """
    Build parameter placements based on the give parallel style of linear layers.
    """
    parameter_placements: dict[str, Placement] = {}
    for linear_fqn, parallel_style in parallel_strategies.items():
        weight_fqn = f"{linear_fqn}.weight"
        bias_fqn = f"{linear_fqn}.bias"
        if weight_fqn not in params_and_buffers:
            raise AssertionError
        parameter_placements[weight_fqn] = (
            Shard(0) if parallel_style == ColwiseParallel else Shard(1)
        )
        if bias_fqn in params_and_buffers:
            parameter_placements[bias_fqn] = (
                Shard(0) if parallel_style == ColwiseParallel else Replicate()
            )
    return parameter_placements


def _mark_tensor_parallel_shardings(
    gm: GraphModule,
    graph_signature: ExportGraphSignature,
    mesh: DeviceMesh,
    parameter_placements: dict[str, Placement],
) -> dict[Node, OpSpec]:
    """
    Mark the placement strategies of the parameter and buffer placeholder nodes.
    """
    placement_strategies: dict[Node, OpSpec] = {}
    num_params_and_buffers = len(graph_signature.inputs_to_parameters) + len(
        graph_signature.inputs_to_buffers
    )
    placeholder_idx: int = 0
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if placeholder_idx < num_params_and_buffers:
                fqn: str = _get_input_node_fqn(node.name, graph_signature)
                placement: Placement = (
                    parameter_placements[fqn]
                    if fqn in parameter_placements
                    else Replicate()
                )
                placement_strategies[node] = _create_placement_strategy(
                    node,
                    mesh,
                    placements=(placement,),
                )
                placeholder_idx += 1
            else:
                placement_strategies[node] = _create_placement_strategy(
                    node,
                    mesh,
                    placements=(Replicate(),),
                )
    return placement_strategies


def _get_input_node_fqn(input_name: str, graph_signature: ExportGraphSignature) -> str:
    """
    Return the FQN of an input node.
    """
    if input_name in graph_signature.inputs_to_parameters:
        return graph_signature.inputs_to_parameters[input_name]
    elif input_name in graph_signature.inputs_to_buffers:
        return graph_signature.inputs_to_buffers[input_name]
    else:
        raise ValueError(
            f"{input_name} not found in inputs_to_parameters or inputs_to_buffers"
        )


def _mark_sharding(
    gm: GraphModule,
    graph_signature: ExportGraphSignature,
    mesh: DeviceMesh,
    parameter_placements: dict[str, Placement],
) -> dict[Node, OpSpec]:
    """
    Mark the sharding strategy for each node in the graph module.
    """
    placement_strategies: dict[Node, OpSpec] = _mark_tensor_parallel_shardings(
        gm,
        graph_signature,
        mesh,
        parameter_placements,
    )

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if node not in placement_strategies:
                placement_strategies[node] = _create_placement_strategy(
                    node, mesh, placements=(Replicate(),)
                )
            node.meta["sharding"] = placement_strategies[node]
        elif node.op == "call_function":
            if node.target is operator.getitem:
                input_nodes = node.all_input_nodes
                if len(input_nodes) != 1:
                    raise AssertionError(
                        f"non-compute op only support one input now, found node: {node} "
                        f"with length of inputs: {len(node.args)}"
                    )
                arg_strategy = placement_strategies[input_nodes[0]]
                placement_strategies[node] = _create_placement_strategy(
                    node,
                    mesh,
                    placements=arg_strategy.output_spec.placements,
                    input_specs=_get_input_node_specs(node, placement_strategies),
                )
                node.meta["sharding"] = placement_strategies[node]
            else:
                op_schema = _get_op_schema(node, placement_strategies)

                # get DTensor specs for inputs and outputs
                sharding_propagator = DTensor._op_dispatcher.sharding_propagator
                if (
                    op_schema.op not in sharding_propagator.op_strategy_funcs
                    and op_schema.op not in sharding_propagator.op_to_rules
                    and op_schema.op
                    not in sharding_propagator.op_single_dim_strategy_funcs
                ):
                    # Mark all as replicated
                    output_sharding = _generate_default_output_sharding(
                        node,
                        mesh,
                        op_schema,
                    )
                else:
                    output_sharding = DTensor._op_dispatcher.sharding_propagator.propagate_op_sharding(  # type: ignore[assignment]
                        op_schema,
                    )
                placement_strategies[node] = OpSpec(
                    # pyrefly: ignore [bad-argument-type]
                    output_specs=_get_output_spec_from_output_sharding(output_sharding),
                    # pyrefly: ignore [missing-attribute]
                    input_specs=output_sharding.redistribute_schema.args_spec
                    # pyrefly: ignore [missing-attribute]
                    if output_sharding.redistribute_schema is not None
                    else _get_input_node_specs(node, placement_strategies),
                )
                node.meta["sharding"] = placement_strategies[node]
        elif node.op == "output":
            node.meta["sharding"] = None
        else:
            raise RuntimeError(f"op code {node.op} not supported")
    return placement_strategies


def _get_output_spec_from_output_sharding(
    output_sharding: OutputSharding,
) -> DTensorSpec:
    """
    Util function to extract output spec from output sharding.
    """
    if isinstance(output_sharding.output_spec, DTensorSpec):
        return output_sharding.output_spec
    else:
        # For ops that return multiple outputs, the outputs should have the same output spec
        if not isinstance(output_sharding.output_spec, Sequence):
            raise AssertionError
        if output_sharding.output_spec[0] is None:
            raise AssertionError
        output_sharding.output_spec[0].tensor_meta = None
        return output_sharding.output_spec[0]


def _create_placement_strategy(
    node: Node,
    mesh: DeviceMesh,
    placements: tuple[Placement, ...],
    input_specs: Sequence[DTensorSpec] | None = None,
) -> OpSpec:
    """
    Util function to construct an OpSpec for a given node.
    """
    placement = OpSpec(
        input_specs=input_specs,
        output_specs=DTensorSpec(
            mesh=mesh,
            placements=placements,
        ),
    )
    _populate_tensor_meta(node, placement.output_specs)
    return placement


def _populate_tensor_meta(node: Node, output_spec: OutputSpecType) -> None:
    """
    Util function to populate tensor meta of output_spec based on node metadata.
    """
    if isinstance(node.meta["val"], Sequence):
        if not isinstance(output_spec, Sequence):
            raise AssertionError
        for spec, fake_tensor in zip(output_spec, node.meta["val"]):
            if spec is None:
                raise AssertionError
            spec.tensor_meta = TensorMeta(
                shape=fake_tensor.shape,
                stride=fake_tensor.stride(),
                dtype=fake_tensor.dtype,
            )
    else:
        if not isinstance(output_spec, DTensorSpec):
            raise AssertionError
        output_spec.tensor_meta = TensorMeta(
            shape=node.meta["val"].shape,
            stride=node.meta["val"].stride(),
            dtype=node.meta["val"].dtype,
        )


def _generate_default_output_sharding(
    node: Node,
    mesh: DeviceMesh,
    op_schema: OpSchema,
) -> OutputSharding:
    """
    Util function to create a default output sharding that suggests Replicate placement for both args and outputs.
    """

    def update_arg_spec(arg_spec: DTensorSpec) -> DTensorSpec:
        return DTensorSpec(
            mesh=arg_spec.mesh,
            placements=(Replicate(),),
            tensor_meta=arg_spec.tensor_meta,
        )

    new_op_schema = OpSchema(
        op=op_schema.op,
        args_schema=pytree.tree_map_only(
            DTensorSpec, update_arg_spec, op_schema.args_schema
        ),
        kwargs_schema=op_schema.kwargs_schema,
    )

    def create_output_spec(tensor: FakeTensor) -> DTensorSpec:
        return DTensorSpec(
            mesh=mesh,
            placements=(Replicate(),),
            tensor_meta=TensorMeta(
                shape=tensor.shape,
                stride=tensor.stride(),
                dtype=tensor.dtype,
            ),
        )

    return OutputSharding(
        output_spec=pytree.tree_map_only(
            FakeTensor, create_output_spec, node.meta["val"]
        ),
        redistribute_schema=new_op_schema,
        needs_redistribute=True,
    )


def _partitioner(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Graph partitioner that partitions the single device graph
    to distributed graph
    """
    for node in gm.graph.nodes:
        node_sharding = node.meta["sharding"]
        if node.op == "placeholder":
            out_spec = node_sharding.output_spec
            local_val = _partition_val(node.meta["val"], out_spec)
            # update node value
            node.meta["val"] = local_val
        elif node.op == "call_function":
            out_spec = node_sharding.output_spec
            # check if there's misaligned sharding, insert reshard if there is
            expected_input_specs = node_sharding.input_specs
            for idx, input_arg in enumerate(node.all_input_nodes):
                input_arg_sharding = input_arg.meta["sharding"]
                input_arg_spec = input_arg_sharding.output_spec
                desired_spec = (
                    out_spec
                    if expected_input_specs is None
                    else expected_input_specs[idx]
                )
                if input_arg_spec != desired_spec:
                    _insert_reshard_gm(
                        gm, node, input_arg, input_arg_spec, desired_spec
                    )
            # convert output val to its local component
            output_val = node.meta["val"]
            node.meta["val"] = _partition_val(output_val, out_spec)
        elif node.op == "output":
            for input_arg in node.all_input_nodes:
                # input args of output should be Replicate, otherwise redistribution is needed.
                input_args_to_check: Sequence[Node] = (
                    input_arg if isinstance(input_arg, Sequence) else [input_arg]
                )
                for arg in input_args_to_check:
                    arg_sharding = arg.meta["sharding"]
                    arg_spec = arg_sharding.output_spec
                    desired_spec = copy.copy(arg_spec)
                    desired_spec.placements = (Replicate(),)
                    if arg_spec != desired_spec:
                        _insert_reshard_gm(gm, node, arg, arg_spec, desired_spec)
        else:
            raise RuntimeError(f"op code {node} not supported")

    _clean_up_graph_metadata(gm)
    gm.graph.lint()
    gm.recompile()
    return gm


def _partition_val(val: Any, spec: DTensorSpec) -> Any:
    """
    util function to convert a full tensor val to its local component
    """
    if isinstance(val, torch.Tensor):
        local_shard = val
        if val.ndim == 0:
            # If it's already a scalar tensor, it is already local, we don't
            # need to do anything
            return local_shard

        for idx, placement in enumerate(spec.placements):
            # NOTE: is_shard() does not match _StridedShard; see _is_shard_like().
            if placement.is_shard():
                placement = cast(Shard, placement)
                num_chunks = spec.mesh.size(mesh_dim=idx)
                my_coord = spec.mesh.get_coordinate()
                if my_coord is None:
                    raise AssertionError("current rank not in mesh!")
                my_coord_on_mesh_dim = my_coord[idx]
                local_shard = placement._select_split_tensor(
                    local_shard,
                    num_chunks,
                    my_coord_on_mesh_dim,
                    with_padding=False,
                    contiguous=True,
                    clone=False,
                )
        return local_shard
    elif isinstance(val, (list, tuple)):
        return val.__class__(_partition_val(v, spec) for v in val)
    else:
        raise RuntimeError(f"val type {type(val)} not supported")


def _insert_reshard_gm(
    gm: torch.fx.GraphModule,
    node: Node,
    input_arg: Node,
    input_arg_spec: DTensorSpec,
    desired_spec: DTensorSpec,
) -> None:
    """
    Transform the graph for tensor redistribution.
    """
    input_arg_spec.tensor_meta = input_arg.meta["tensor_meta"]
    desired_spec.tensor_meta = input_arg.meta["tensor_meta"]
    input_arg_tensor = input_arg.meta["val"]

    # insert reshard operation
    def reshard_fn(local_tensor: torch.Tensor) -> torch.Tensor:
        return redistribute_local_tensor(
            local_tensor,
            input_arg_spec,
            desired_spec,
        )

    reshard_gm = make_fx(reshard_fn)(input_arg_tensor)
    reshard_gm_nodes = list(reshard_gm.graph.nodes)
    input_node = reshard_gm_nodes[0]
    with gm.graph.inserting_before(node):
        # copy nn_module_stack metadata for output, all-reduce nodes
        for reshard_node in reshard_gm.graph.nodes:
            if reshard_node.op not in ["placeholder", "output"]:
                reshard_node.meta["nn_module_stack"] = (
                    copy.copy(input_arg.meta["nn_module_stack"])
                    if input_arg.op != "placeholder"
                    else copy.copy(node.meta["nn_module_stack"])
                )
        output_node = gm.graph.graph_copy(
            reshard_gm.graph,
            val_map={
                input_node: input_arg,
            },
        )
    node.replace_input_with(input_arg, output_node)  # type: ignore[arg-type]


def _clean_up_graph_metadata(gm: torch.fx.GraphModule) -> None:
    """
    Clean up the graph by removing sharding and partitioning related metadata
    """
    for node in gm.graph.nodes:
        if "sharding" in node.meta:
            del node.meta["sharding"]
        if "val" in node.meta and isinstance(node.meta["val"], torch.Tensor):
            local_tensor_meta = _extract_tensor_metadata(node.meta["val"])
            node.meta["tensor_meta"] = local_tensor_meta


def _get_input_node_specs(
    node: Node, placement_strategies: dict[Node, OpSpec]
) -> tuple[DTensorSpec, ...]:
    """
    Get the input specs of a node.
    """
    input_specs_list: list[DTensorSpec] = []
    for input_arg in node.all_input_nodes:
        if input_arg in placement_strategies:
            output_spec = placement_strategies[input_arg].output_specs
            if not isinstance(output_spec, DTensorSpec):
                raise AssertionError
            input_specs_list.append(output_spec)
        else:
            raise ValueError(f"{input_arg} does not have output_spec populated.")
    return tuple(input_specs_list)


def _get_op_schema(node: Node, placement_strategies: dict[Node, OpSpec]) -> OpSchema:
    """
    Util function to construct the operator schema of a node.
    """
    args_schema_list = pytree.tree_map_only(
        Node, lambda arg: placement_strategies[arg].output_specs, node.args
    )
    op_schema = OpSchema(
        op=cast(torch._ops.OpOverload, node.target),
        args_schema=tuple(args_schema_list),
        kwargs_schema=cast(dict[str, object], node.kwargs),
    )
    return op_schema


def _shard_state_dict(
    state_dict: dict[str, torch.Tensor],
    placement_strategies: dict[Node, OpSpec],
    graph_signature: ExportGraphSignature,
    mesh: DeviceMesh,
) -> None:
    """
    Inplace partition the weights based on the OpSpec
    """
    for node, op_spec in placement_strategies.items():
        if node.op != "placeholder":
            continue
        if node.name in graph_signature.inputs_to_parameters:
            fqn = graph_signature.inputs_to_parameters[node.name]
        elif node.name in graph_signature.inputs_to_buffers:
            fqn = graph_signature.inputs_to_buffers[node.name]
        else:
            continue
        if fqn not in state_dict:
            raise AssertionError(f"{fqn} not found in state dict: {state_dict.keys()}")

        original_param = state_dict[fqn]
        dtensor_param = distribute_tensor(
            original_param,
            mesh,
            op_spec.output_spec.placements,
        )
        local_param = dtensor_param.to_local()
        state_dict[fqn] = (
            torch.nn.Parameter(local_param)
            if isinstance(original_param, torch.nn.Parameter)
            else local_param
        )
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
from collections.abc import Callable, Sequence
from functools import partial

import torch
from torch._ops import OpOverload
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpStrategy,
    PlacementList,
    RuntimeSchemaInfo,
    StrategyType,
    TupleStrategy,
)
from torch.distributed.tensor._ops.utils import expand_to_full_mesh_op_strategy


__all__ = ["register_sharding"]


def register_sharding(op: OpOverload | list[OpOverload]):
    """
    :meth:`register_sharding` is an experimental API that allows users to register sharding
    strategies for an operator when the tensor inputs and outputs are DTensor.
    It can be useful when: (1) there doesn't exist a default sharding strategy for ``op``,
    e.g. when ``op`` is a custom operator that is not supported by :class:`DTensor`; (2)
    when users would like to overwrite default sharding strategies of existing operators.

    Args:
        op (Union[OpOverload, List[OpOverload]]):
            An op or a list of ops to register the customized sharding function.

    Returns:
        A function decorator which can be used to wrap a function that defines the sharding
        strategy for the operator specified in ``op``. The defined sharding strategy will be
        registered to DTensor and will override the default sharding strategy if DTensor has
        already implemented the operator. The customized sharding function takes the same inputs
        as the original op (except that if an arg is a :class:`torch.Tensor`, it will be
        replaced by a tensor-like object that DTensor uses internally). The function should
        return a sequence of 2-tuples, each specifying acceptable output placements and its
        corresponding input placements.

    Example:
        >>> # xdoctest: +SKIP("distributed")
        >>> @register_sharding(aten._softmax.default)
        >>> def custom_softmax_sharding(x, dim, half_to_float):
        >>>     softmax_dim = dim if dim >= 0 else dim + x.ndim
        >>>     acceptable_shardings = []
        >>>
        >>>     all_replicate = ([Replicate()], [Replicate(), None, None])
        >>>     acceptable_shardings.append(all_replicate)
        >>>
        >>>     for sharding_dim in range(x.ndim):
        >>>         if sharding_dim != softmax_dim:
        >>>             all_sharded = (
        >>>                 [Shard(sharding_dim)],
        >>>                 [Shard(sharding_dim), None, None],
        >>>             )
        >>>             acceptable_shardings.append(all_sharded)
        >>>
        >>>     return acceptable_shardings

    .. note:: This API is currently experimental and subject to change
    """

    def custom_strategy(
        custom_sharding_fn: Callable[
            ..., Sequence[tuple[PlacementList, PlacementList]]
        ],
        op_schema: OpSchema,
    ) -> StrategyType:
        def strategy_to_spec(strategy: object) -> object:
            if isinstance(strategy, OpStrategy):
                # take the output spec from the first strategy
                return strategy.strategies[0].output_spec
            elif isinstance(strategy, TupleStrategy):
                return tuple(strategy_to_spec(s) for s in strategy.children)
            else:
                return strategy

        mesh = op_schema.get_mesh_from_args()

        args_schema = tuple(strategy_to_spec(i) for i in op_schema.args_schema)
        kwargs_schema = {
            k: strategy_to_spec(v) for k, v in op_schema.kwargs_schema.items()
        }

        acceptable_shardings = custom_sharding_fn(*args_schema, **kwargs_schema)

        single_mesh_dim_strategies: list[PlacementList] = []
        for output_specs, input_specs in acceptable_shardings:
            single_mesh_dim_strategies.append(output_specs + input_specs)

        # TODO: handle out variant ops
        return expand_to_full_mesh_op_strategy(
            mesh,
            op_schema,
            single_mesh_dim_strategies,
            input_index=len(op_schema.op._schema.returns),
            inplace_op=op_schema.is_inplace_op(),
        )

    def wrapper(custom_sharding_fn):
        def derive_schema_info(op):
            # NOTE: without user directly providing RuntimeSchemaInfo, for now
            #       we create it in a conservative fashion as follows:
            #       1. let static_argnum be the first int argument
            #       2. let static_kwargkey include all the int type kwargs
            #       3. always set needs_pytree=True
            static_argnum = 100
            static_kwargkey: list[str] = []
            for i, arg in enumerate(op._schema.arguments):
                if isinstance(arg.type, torch.IntType) or (
                    isinstance(arg.type, torch.OptionalType)
                    and isinstance(arg.type.getElementType(), torch.IntType)
                ):
                    static_argnum = min(i, static_argnum)
                    if arg.kwarg_only:
                        static_kwargkey.append(arg.name)
            return RuntimeSchemaInfo(
                static_argnum, static_kwargkey or None, needs_pytree=True
            )

        overloads = op if isinstance(op, list) else [op]
        for overload in overloads:
            DTensor._op_dispatcher.sharding_propagator.register_op_strategy(
                overload,
                partial(custom_strategy, custom_sharding_fn),
                derive_schema_info(overload),
            )

        return custom_sharding_fn

    return wrapper
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates
from collections.abc import Iterator
from contextlib import contextmanager

from torch.distributed.tensor._api import DTensor
from torch.distributed.tensor.experimental._attention import context_parallel
from torch.distributed.tensor.experimental._func_map import local_map
from torch.distributed.tensor.experimental._register_sharding import register_sharding


__all__ = ["context_parallel", "implicit_replication", "local_map", "register_sharding"]


@contextmanager
def implicit_replication() -> Iterator[None]:
    """
    This context manager allows :class:`DTensor` to implicitly treat all non-DTensors (``torch.Tensor``)
    in the program be replicate :class:`DTensor` s during the operator computation.

    .. warning:: This might possible lead to incorrect results if ``torch.Tensor`` s are not replicated
        in practice, please use it at your discretion.
    """
    try:
        DTensor._op_dispatcher._allow_implicit_replication = True
        yield
    finally:
        DTensor._op_dispatcher._allow_implicit_replication = False


# Set namespace for exposed private names
context_parallel.__module__ = "torch.distributed.tensor.experimental"
implicit_replication.__module__ = "torch.distributed.tensor.experimental"
local_map.__module__ = "torch.distributed.tensor.experimental"
register_sharding.__module__ = "torch.distributed.tensor.experimental"
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import functools
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.distributed as dist


if dist._is_spmd_types_available():
    import spmd_types as spmd

from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)


try:
    from torch.utils import _cxx_pytree as pytree
except ImportError:
    from torch.utils import _pytree as pytree  # type: ignore[no-redef]


__all__ = ["local_map"]

PlacementType = Sequence[Placement] | None
InputPlacements = tuple[PlacementType, ...] | None
OutputPlacements = PlacementType | tuple[PlacementType, ...]


def _placements_to_spmd_type(
    placements: PlacementType,
    grad_placements: PlacementType,
    device_mesh: DeviceMesh,
) -> dict:
    """Convert DTensor placements to a dict of MeshAxis -> spmd local type.

    Mapping: Shard -> V, Replicate -> R (or I if grad is Replicate), Partial -> P.
    Validates that grad_placements are compatible with forward placements.
    """
    result = {}
    for dim_idx, placement in enumerate(placements):  # pyrefly: ignore
        axis = spmd.MeshAxis.of(device_mesh.get_group(dim_idx))
        grad_p = grad_placements[dim_idx] if grad_placements is not None else None

        if type(placement) is Shard:
            fwd_type = spmd.V
        elif type(placement) is Replicate:
            fwd_type = spmd.I if type(grad_p) is Replicate else spmd.R
        elif type(placement) is Partial:
            fwd_type = spmd.P
        else:
            raise ValueError(
                f"local_map(spmd_types=True) does not support placement type "
                f"{type(placement).__name__}: {placement}"
            )

        if grad_p is not None:
            actual_grad_p = _dtensor_placement_if_compatible(
                fwd_type.backward_type(),  # pyrefly: ignore [missing-attribute]
                grad_p,
            )
            if actual_grad_p is None or type(actual_grad_p) is not type(grad_p):
                raise ValueError(
                    "local_map(spmd_types=True) cannot represent this "
                    "forward/backward placement pair:\n"
                    f"in_placements={placement}, in_grad_placements={grad_p}.\n\n"
                    "The local_map boundary assigns one SPMD type to each "
                    "local tensor, which also fixes the expected backward "
                    f"placement. For in_placements={placement}, valid grad "
                    f"placements are {_valid_grad_placements(placement)}.\n\n"
                    "If the forward and backward layouts intentionally "
                    "diverge in a way not represented by one SPMD type, "
                    "expand the local_map region to include that divergence, "
                    "so the local_map region boundary respects tied FWD-BWD "
                    "typing."
                )

        result[axis] = fwd_type
    return result


def _annotate_spmd_types(
    flat_local_args: list,
    in_placements: InputPlacements,
    in_grad_placements: InputPlacements,
    device_mesh: DeviceMesh,
) -> None:
    """Annotate unwrapped local tensors with spmd_types inferred from placements.

    Assumes local_map has already validated in_placements and unwrapped DTensors.
    """
    for idx, local_arg in enumerate(flat_local_args):
        if not isinstance(local_arg, torch.Tensor):
            continue
        if in_placements is None or in_placements[idx] is None:
            continue
        grad_placements = (
            in_grad_placements[idx] if in_grad_placements is not None else None
        )
        spmd_type = _placements_to_spmd_type(
            in_placements[idx], grad_placements, device_mesh
        )
        spmd.assert_type(local_arg, spmd_type)


def _valid_grad_placements(placement: Placement) -> str:
    if type(placement) is Shard:
        return "Shard (spmd.V)"
    elif type(placement) is Replicate:
        return "Partial (spmd.R) or Replicate (spmd.I)"
    elif type(placement) is Partial:
        return "Replicate (spmd.P)"
    raise ValueError(
        f"local_map(spmd_types=True) does not support placement type {type(placement).__name__}"
    )


def _dtensor_placement_if_compatible(
    local_type, placement: Placement
) -> Placement | None:
    if local_type == spmd.V:
        if type(placement) in (Shard, Partial):
            return placement
        return None
    return spmd.spmd_type_to_dtensor_placement(local_type)


def _out_spmd_types_to_grad_placements(
    flat_out: Sequence[Any],
    out_placements_tuple: tuple[PlacementType, ...],
    device_mesh: DeviceMesh,
) -> tuple[PlacementType, ...]:
    """Validate output SPMD types and return backward grad placements."""
    grad_out_placements: list[PlacementType] = []
    for out, spec in zip(flat_out, out_placements_tuple, strict=True):
        if spec is not None and not isinstance(out, torch.Tensor):
            raise ValueError(
                f"out_placements specifies {spec} but the corresponding "
                f"output is {type(out).__name__}, not a Tensor"
            )
        if spec is None:
            if isinstance(out, torch.Tensor):
                raise ValueError(
                    "out_placements has None for a Tensor output. Provide "
                    "placements for Tensor outputs and use None only for "
                    "non-Tensor outputs."
                )
            grad_out_placements.append(None)
            continue

        # validate out type
        actual_type = spmd.get_local_type(out)
        if not actual_type:
            raise ValueError(
                "Output tensor has no spmd_types annotation but out_placements "
                "expects one. Ensure the function's output is derived from "
                "annotated inputs or is explicitly annotated."
            )
        for dim_idx, placement in enumerate(spec):  # pyrefly: ignore
            axis = spmd.MeshAxis.of(device_mesh.get_group(dim_idx))
            actual = actual_type.get(axis)
            if actual is None:
                if device_mesh.mesh_dim_names is None:
                    mesh_dims: Sequence[Any] = tuple(range(device_mesh.ndim))
                else:
                    mesh_dims = device_mesh.mesh_dim_names
                annotated_dims = [
                    mesh_dims[i]
                    for i in range(device_mesh.ndim)
                    if spmd.MeshAxis.of(device_mesh.get_group(i)) in actual_type
                ]
                raise ValueError(
                    "Output tensor has no spmd_types annotation on "
                    f"DeviceMesh dimension {mesh_dims[dim_idx]} "
                    f"but out_placements expects {placement}. "
                    "Actual annotations on this DeviceMesh are: "
                    f"{annotated_dims}"
                )
            actual_placement = _dtensor_placement_if_compatible(actual, placement)
            if actual_placement is None or type(actual_placement) is not type(
                placement
            ):
                raise ValueError(
                    f"Output tensor placement mismatch on {axis}: "
                    f"out_placements={placement} but spmd_types inferred "
                    f"spmd.{actual.name}"
                )

        # converts out types -> out bwd types -> to DTensor placements, e.g.
        # P (or V) -> R -> Replicate
        # R -> P -> Partial
        # I -> I -> Replicate
        # V -> V -> Shard (shard dim consulted from out_placements)
        grad_spec = []
        for dim_idx, placement in enumerate(spec):
            axis = spmd.MeshAxis.of(device_mesh.get_group(dim_idx))
            if actual_type[axis] == spmd.V and type(placement) is Partial:
                grad_placement = Replicate()
            else:
                grad_placement = _dtensor_placement_if_compatible(
                    actual_type[axis].backward_type(), placement
                )
            if grad_placement is None:
                raise AssertionError("expected grad_placement to be not None")
            grad_spec.append(grad_placement)
        grad_out_placements.append(tuple(grad_spec))

    return tuple(grad_out_placements)


def local_map(
    func: Callable | None = None,
    out_placements: OutputPlacements = None,
    in_placements: InputPlacements = None,
    in_grad_placements: InputPlacements = None,
    device_mesh: DeviceMesh | None = None,
    *,
    redistribute_inputs: bool = False,
    spmd_types: bool = False,
):
    """
    :meth:`local_map` is an experimental API that allows users to pass :class:`DTensor` s
    to a function that is written to be applied on ``torch.Tensor`` s. It is done by extracting
    the local components of :class:`DTensor`, call the function, and wrap the outputs to
    :class:`DTensor` according to the ``out_placements``.

    Args:
        func (Callable): the function to be applied on each local shard of
            :class:`DTensor` s.
        out_placements (Union[`PlacementType`, Tuple[`PlacementType`, ...]]):
            the desired placements of the :class:`DTensor` s in ``func``'s flattened output.
            If the flattened ``output`` is a single value, the ``out_placements`` should be
            of type `PlacementType`. Otherwise if the flattened ``output`` has multiple
            values, the ``out_placements`` should be a tuple of `PlacementType` values 1:1
            mapping to the flattened ``output``.
            Besides, for :class:`Tensor` output, we use `PlacementType` as its
            placements (a `Tuple[Placement]` value). For non-Tensor output, the `PlacementType`
            should be `None`.
            Note that the only exception is when no :class:`DTensor` argument is passed
            in. In this case, even if `out_placements` is not `None`, the result function
            should ignore the desired placements because the function is not running with
            :class:`DTensor` s.
        in_placements (Tuple[`PlacementType`, ...], optional):
            the required placements of the :class:`DTensor` s in the flattened inputs of ``func``.
            If ``in_placements`` is specified, :meth:`local_map` would examine whether the
            placements of each :class:`DTensor` argument is the same as the required
            placements or not. If the placements are not the same and
            ``redistribute_inputs`` is ``False``, an exception will be raised. Otherwise if
            ``redistribute_inputs`` is ``True``, the argument will be first redistributed to
            the required sharding placements before passing its local tensor to ``func``.
            The only exception is when required placements are not ``None`` and the
            argument is a :class:`torch.Tensor`. In this case, the placements examination
            will be skipped and the argument will be directly passed to ``func``.
            If ``in_placements`` is ``None``, no placements examination will be performed.
            Default: None
        in_grad_placements (Tuple[`PlacementType`, ...], optional):
            the placements hint of the :class:`DTensor` s gradient corresponds
            to the flattened input DTensor. This argument is the hint that user
            can give to :meth:`to_local` in case the gradient layout of the
            local tensor input does not match its :class:`DTensor` input layout.
            If not specified, we will assume the gradient layout of the local
            tensor input remains the same as the original :class:`DTensor` input
            and use that for gradient computation. Default: None.
        device_mesh (:class:`DeviceMesh`, optional):
            the device mesh that the output :class:`DTensor` s are placed on. If not
            specified, this will be inferred from the first input :class:`DTensor`'s device
            mesh. Default: None.

    Keyword Args:
        redistribute_inputs (bool, optional):
            the bool value indicating whether to reshard the input :class:`DTensor` s when
            their placements are different from the required input placements. If this
            value is ``False`` and some :class:`DTensor` input has a different placement,
            an exception will be raised. Default: False.
        spmd_types (bool, optional):
            if ``True``, enable runtime SPMD type checking for ``func``. Local
            tensors are annotated with SPMD types (V, R, I, P) inferred from
            ``in_placements`` and ``in_grad_placements``, and ``func`` is executed
            under strict type checking. This helps surface distributed correctness
            bugs such as whether a replicated local tensor's gradient is
            Partial or Replicate.
            See the ``spmd_types`` local SPMD types documentation for the
            meaning of V, R, I, and P and their backward types:
            https://github.com/meta-pytorch/spmd_types/blob/main/docs/local_spmd_types.md
            For Replicate input placements, if a gradient placement is not provided,
            we default to Replicate (R), instead of Invariant (I). Requires the
            ``spmd_types`` package to be installed. Default: False.

    Returns:
        A ``Callable`` that applies ``func`` to each local shard of the input :class:`DTensor`
        and returns a :class:`DTensor` constructed from the return value of ``func``.

    Raises:
        AssertionError: For any non-DTensor output, we require its corresponding
            output placement in ``out_placements`` be None. An AssertionError will be raised
            if this is not the case.

        ValueError: If ``redistribute_inputs=False`` but the input :class:`DTensor` needs
            a redistribution according to ``in_placements``.

    Example:
        >>> # xdoctest: +SKIP("distributed")
        >>> def mm_allreduce_forward(device_mesh, W, X):
        >>>     partial_sum_tensor = torch.mm(W, X)
        >>>     reduced_tensor = funcol.all_reduce(partial_sum_tensor, "sum", device_mesh)
        >>>     return reduced_tensor
        >>>
        >>> W = torch.randn(12, 8, requires_grad=False)
        >>> X = torch.randn(8, 16, requires_grad=False)
        >>> Y = torch.mm(W, X)
        >>> row_wise = [Shard(0)]  # row-wise sharding placements on 1-d mesh
        >>> col_wise = [Shard(1)]  # col-wise sharding placements on 1-d mesh
        >>>
        >>> # local_mm_allreduce_forward is the function wrapped with DTensor/Tensor conversion
        >>> local_mm_allreduce_forward = local_map(
        >>>     mm_allreduce_forward,
        >>>     out_placements=[Replicate()],
        >>>     in_placements=[col_wise, row_wise],
        >>>     device_mesh=device_mesh,
        >>> )
        >>>
        >>> W_dt = distribute_tensor(
        ...     W, device_mesh, (col_wise)
        ... )  # col-wisely sharded W tensor
        >>> X_dt = distribute_tensor(
        ...     X, device_mesh, (row_wise)
        ... )  # row-wisely sharded X tensor
        >>> Y_dt = local_mm_allreduce_forward(
        ...     device_mesh, W_dt, X_dt
        ... )  # apply local_mm_allreduce_forward to DTensors

    .. note:: This API is currently experimental and subject to change
    """

    if func is None:
        # decorator mode
        def decorated(func):
            return local_map(
                func=func,
                out_placements=out_placements,
                in_placements=in_placements,
                in_grad_placements=in_grad_placements,
                device_mesh=device_mesh,
                redistribute_inputs=redistribute_inputs,
                spmd_types=spmd_types,
            )

        return decorated

    return functools.partial(
        _local_map_wrapped,
        func,
        out_placements,
        in_placements,
        in_grad_placements,
        device_mesh,
        redistribute_inputs,
        spmd_types,
    )


def _local_map_wrapped(
    func: Callable,
    out_placements: OutputPlacements,
    in_placements: InputPlacements,
    in_grad_placements: InputPlacements,
    device_mesh: DeviceMesh | None,
    redistribute_inputs: bool,
    enable_spmd_types: bool,
    *args,
    **kwargs,
):
    # process input args
    flat_args, args_spec = pytree.tree_flatten(args)
    if in_placements is not None:
        if len(in_placements) != len(flat_args):
            raise AssertionError(
                f"in_placements length {len(in_placements)} does not match the number "
                f"of input args {len(flat_args)}!"
            )

    # we assume every DTensor object is placed on the same device mesh
    flat_local_args = []
    seen_dtensor_arg = False
    for idx, arg in enumerate(flat_args):
        if isinstance(arg, DTensor):
            # TODO: the current code doesn't consider the uneven sharding case
            # Need to think about what the consequence is when the input DTensor
            # is uneven sharded.
            if device_mesh is None:  # infer device mesh from the DTensor arg
                device_mesh = arg.device_mesh

            # this function is applied to at least one DTensor argument
            seen_dtensor_arg = True

            if in_placements is not None:
                spec = in_placements[idx]
                if spec is None:
                    raise AssertionError(
                        f"DTensor input {arg} expects placements but received {spec}!"
                    )

                if not isinstance(spec, tuple):
                    spec = tuple(spec)

                if arg.placements != spec:
                    if redistribute_inputs:
                        # redistribute to input placements
                        arg = arg.redistribute(placements=spec)
                    else:
                        raise ValueError(
                            f"arg {arg} in local_map has a mismatched placements: "
                            f"arg placements is {arg.placements} but the input "
                            f"placements is {spec}! "
                            "If redistribute_inputs is wanted, set "
                            "redistribute_inputs=True to local_map."
                        )

            if in_grad_placements is not None:
                spec = in_grad_placements[idx]
                if spec is None:
                    raise AssertionError(
                        f"DTensor input {arg} expects in grad placements but received {spec}!"
                    )
                if not isinstance(spec, tuple):
                    spec = tuple(spec)
                local_arg = arg.to_local(grad_placements=spec)
            else:
                local_arg = arg.to_local()

            if isinstance(local_arg, AsyncCollectiveTensor):
                local_arg = local_arg.wait()

            flat_local_args.append(local_arg)
        else:
            # Non-Tensor input must have None in `in_placements`
            if in_placements is not None and not isinstance(arg, torch.Tensor):
                spec = in_placements[idx]
                if spec is not None:
                    raise AssertionError(
                        f"Non-Tensor input {arg} expects None placements "
                        f"but received {spec}!"
                    )

            flat_local_args.append(arg)

    # pyrefly: ignore [bad-argument-type]
    local_args = pytree.tree_unflatten(flat_local_args, args_spec)

    if enable_spmd_types and seen_dtensor_arg:
        if not dist._is_spmd_types_available():
            raise RuntimeError(
                "spmd_types=True requires the spmd_types package to be installed"
            )
        if device_mesh is None:
            raise AssertionError("expected device_mesh to be not None")
        _annotate_spmd_types(
            flat_local_args, in_placements, in_grad_placements, device_mesh
        )
        from spmd_types._checker import typecheck

        with (
            spmd.set_current_mesh(device_mesh),
            typecheck(strict_mode="strict"),
        ):
            out = func(*local_args, **kwargs)
    else:
        out = func(*local_args, **kwargs)

    if seen_dtensor_arg:
        # process output to be DTensor if we've seen DTensor inputs
        flat_out, out_spec = pytree.tree_flatten(out)

        flat_dist_out = []
        out_placements_tuple = (
            out_placements if isinstance(out_placements, tuple) else (out_placements,)
        )
        if len(flat_out) != len(out_placements_tuple):
            raise AssertionError(
                "local_map requires one PlacementType be provided for each output value,"
                f" received {len(out_placements_tuple)} out_placements but"
                f" {len(flat_out)} is expected!"
            )

        grad_out_placements = None
        if enable_spmd_types:
            if device_mesh is None:
                raise AssertionError("expected device_mesh to be not None")
            grad_out_placements = _out_spmd_types_to_grad_placements(
                flat_out,
                out_placements_tuple,  # pyrefly: ignore [bad-argument-type]
                device_mesh,
            )

        for idx, (out, spec) in enumerate(zip(flat_out, out_placements_tuple)):
            if isinstance(out, torch.Tensor):
                if isinstance(out, DTensor):
                    raise AssertionError(
                        f"torch.Tensor output expected but received {type(out)}: {out}"
                    )
                grad_placements = (
                    grad_out_placements[idx]
                    if grad_out_placements is not None
                    else None
                )

                flat_dist_out.append(
                    # pyrefly: ignore [bad-argument-type]
                    DTensor.from_local(
                        out,
                        device_mesh,
                        spec,  # pyrefly: ignore[bad-argument-type]
                        run_check=False,
                        grad_placements=grad_placements,
                    )
                )
            else:
                if spec is not None:
                    raise AssertionError(
                        f"Non-tensor output {out} expects None placements but received {spec}!"
                    )

                flat_dist_out.append(out)

        # pyrefly: ignore [bad-argument-type]
        return pytree.tree_unflatten(flat_dist_out, out_spec)
    else:
        return out
<|endoftext|>import contextlib
import itertools
import logging
import types
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass
from enum import auto, Enum
from functools import partial
from typing import Any, cast, Protocol, TypeAlias

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as ft_c
import torch.distributed.distributed_c10d as c10d
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_tensor, DTensor, Shard
from torch.distributed.tensor.parallel import ParallelStyle
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    BlockMask,
    create_block_mask,
)
from torch.utils._pytree import tree_flatten, tree_unflatten

from ._cp_custom_ops import flex_cp_allgather
from ._load_balancer import _create_default_load_balancer, _LoadBalancer


__all__ = [
    "_CausalBehavior",
    "_context_parallel_shard",
    "_ContextParallel",
    "_cp_options",
    "_disable_context_parallel_dispatcher",
    "_enable_context_parallel_dispatcher",
    "_is_causal_behavior",
    "_RotateMethod",
    "context_parallel",
    "context_parallel_unshard",
    "set_rotate_method",
]


class _CausalBehavior(Enum):
    SKIP = None
    NOT_IS_CAUSAL = False
    IS_CAUSAL = True


class _RotateMethod(Enum):
    ALL_TO_ALL = auto()
    ALL_GATHER = auto()


aten = torch.ops.aten
logger = logging.getLogger(__name__)


class _DispatchMode(Enum):
    MONKEY_PATCH = auto()
    MODULE_WRAPPER = auto()


_dispatch_mode: _DispatchMode = _DispatchMode.MONKEY_PATCH


@dataclass
class _ContextParallelOptions:
    # Whether to upcast parameters and gradients to float32 to avoid accumulation
    # errors. It is likely this is always True, but we currently keep this variable
    # for experimental purposes.
    convert_to_f32: bool = True
    enable_load_balance: bool = True
    rotate_method: _RotateMethod = _RotateMethod.ALL_GATHER


_cp_options = _ContextParallelOptions()


def _is_causal_behavior(
    rank: int, world_size: int, i: int, is_causal: bool
) -> _CausalBehavior:
    """
    Calculate is_causal behavior for each KV block. The attention can either be
    calculated in full, not at all or with the causal mask applied.
    """
    if not is_causal:
        return _CausalBehavior.NOT_IS_CAUSAL

    if i == 0:
        return _CausalBehavior.IS_CAUSAL

    source_rank = (rank - i) % world_size
    if source_rank < rank or _cp_options.enable_load_balance:
        return _CausalBehavior.NOT_IS_CAUSAL
    else:
        return _CausalBehavior.SKIP


def _maybe_wait(tensor: torch.Tensor) -> torch.Tensor:
    """
    When tracing the code, the result tensor is not an AsyncCollectiveTensor,
    so we cannot call ``wait()``.
    """
    if isinstance(tensor, ft_c.AsyncCollectiveTensor):
        return tensor.wait()
    return tensor


def _partial_update(
    original: torch.Tensor,
    new: torch.Tensor,
    dim: int,
    n_chunks: int,
    idx: int,
    add: bool,
) -> torch.Tensor:
    """
    This API partially updates a chunk of ``original`` tensor. The ``original``
    tensor will be first chunked along ``dim`` dimension, then the ``idx`` chunk
    will be updated with ``new``. If ``add`` is True, the chunk will be added
    with ``new``, otherwise the chunk will be replaced by ``new``.

    The result is a tensor that is the same size as ``original``.
    """
    chunks = list(original.chunk(n_chunks, dim=dim))
    if chunks[idx].shape != new.shape:
        raise AssertionError((original.shape, new.shape, idx))
    if add:
        chunks[idx] += new
    else:
        chunks[idx] = new
    return torch.cat(chunks, dim=dim)


class _SDPAMerger:
    """A class to help merge the local SDPA result."""

    def __init__(self, convert_to_f32: bool, seq_dim: int):
        self._seq_dim = seq_dim
        self._out: torch.Tensor | None = None
        self._lse: torch.Tensor | None = None
        self._should_lse_squeeze = False
        self._convert_to_f32 = convert_to_f32
        self._out_dtype = torch.float32
        self._lse_dtype = torch.float32

    def _merge_one(
        self, block_out: torch.Tensor, block_lse: torch.Tensor, partial: bool
    ) -> None:
        # The cuDNN backend preserves the last dimension for LSE.
        # Apply unsqueeze only if the input does not already have
        # the required dimensionality.
        if len(block_lse.shape) < len(block_out.shape):
            block_lse = block_lse.unsqueeze(dim=-1)
            self._should_lse_squeeze = True
        if len(block_lse.shape) != len(block_out.shape):
            raise AssertionError

        if self._lse is None:
            self._lse = block_lse
            self._out = block_out
        else:
            ROUND_ROBIN_CYCLE = 2
            if self._lse is None:
                raise AssertionError
            if self._out is None:
                raise AssertionError
            lse = (
                self._lse.chunk(ROUND_ROBIN_CYCLE, dim=self._seq_dim)[1]
                if partial
                else self._lse
            )
            out = (
                self._out.chunk(ROUND_ROBIN_CYCLE, dim=self._seq_dim)[1]
                if partial
                else self._out
            )

            # The algorithm from
            # github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
            # gives a relatively stable result.
            out = out - F.sigmoid(block_lse - lse) * (out - block_out)
            lse = lse - F.logsigmoid(lse - block_lse)
            if partial:
                self._lse = _partial_update(
                    self._lse,
                    lse,
                    dim=self._seq_dim,
                    n_chunks=ROUND_ROBIN_CYCLE,
                    idx=1,
                    add=False,
                )
                self._out = _partial_update(
                    self._out,
                    out,
                    dim=self._seq_dim,
                    n_chunks=ROUND_ROBIN_CYCLE,
                    idx=1,
                    add=False,
                )
            else:
                self._lse = lse
                self._out = out

    def step(self, out: torch.Tensor, lse: torch.Tensor, partial: bool) -> None:
        self._out_dtype = out.dtype
        self._lse_dtype = lse.dtype

        if self._convert_to_f32:
            out = out.to(torch.float32)
            lse = lse.to(torch.float32)

        self._merge_one(out, lse, partial)

    def results(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._out is None:
            raise AssertionError
        if self._lse is None:
            raise AssertionError
        out = self._out.to(self._out_dtype)
        if self._should_lse_squeeze:
            lse = self._lse.squeeze(-1).to(self._lse_dtype)
        else:
            lse = self._lse.to(self._lse_dtype)
        return out, lse


class _AttentionOp(Protocol):
    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, ...]: ...


class _RingRotater(ABC):
    @abstractmethod
    def __init__(self, pg: dist.ProcessGroup, seq_dim: int) -> None: ...

    @abstractmethod
    def exchange_buffers(self, curr_buffer: torch.Tensor) -> None: ...

    @abstractmethod
    def next_buffer(self) -> torch.Tensor: ...


class _AllToAllRotater(_RingRotater):
    """Use all_to_all to send the kv to the next rank."""

    def __init__(self, pg: dist.ProcessGroup, seq_dim: int) -> None:
        self._pg = pg
        self._seq_dim = seq_dim
        self._buffer: torch.Tensor | None = None

    def exchange_buffers(self, curr_buffer: torch.Tensor) -> None:
        curr_buffer = curr_buffer.contiguous()
        size = dist.get_world_size(self._pg)
        dsts = list(range(1, size)) + [0]
        self._buffer = ft_c.permute_tensor(curr_buffer, dsts, self._pg)

    def next_buffer(self) -> torch.Tensor:
        if self._buffer is None:
            raise AssertionError
        return _maybe_wait(self._buffer)


class _AllGatherRotater(_RingRotater):
    """
    Allgather the kv and return only the required kv.
    Only one communication will be done.
    """

    def __init__(self, pg: dist.ProcessGroup, seq_dim: int) -> None:
        self._pg = pg
        self._seq_dim = seq_dim
        self._aggregated_buffer: torch.Tensor | None = None
        self._idx = 0

    def exchange_buffers(self, curr_buffer: torch.Tensor) -> None:
        # We only need to perform allgather once.
        self._idx += 1
        if self._aggregated_buffer is None:
            self._aggregated_buffer = ft_c.all_gather_single(
                curr_buffer.contiguous(), gather_dim=0, group=self._pg
            )

    def next_buffer(self) -> torch.Tensor:
        rank = dist.get_rank(self._pg)
        idx = rank - self._idx

        if self._aggregated_buffer is None:
            raise AssertionError
        self._aggregated_buffer = _maybe_wait(self._aggregated_buffer)
        return self._aggregated_buffer.chunk(dist.get_world_size(self._pg))[idx]


def _create_rotater(
    pg: dist.ProcessGroup, seq_dim: int, method: _RotateMethod | None = None
) -> _RingRotater:
    if method is None:
        method = _cp_options.rotate_method

    if method == _RotateMethod.ALL_TO_ALL:
        return _AllToAllRotater(pg, seq_dim)
    elif method == _RotateMethod.ALL_GATHER:
        return _AllGatherRotater(pg, seq_dim)
    else:
        raise NotImplementedError(f"Unknown method {method}")


def _templated_ring_attention(
    group: dist.ProcessGroup,
    seq_dim: int,
    op: _AttentionOp,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    **kwargs: object,
) -> tuple[torch.Tensor, ...]:
    """
    A generalized ring attention implementation that can support multiple attention ops.

    Note [Context parallelism load balance algorithm for causal masking]
    =====================
    This explanation uses an example to illustrate the CP algorithm with causal
    masking.

    Consider a scenario where the sequence length of q, k, and v is 4 (e.g.,
    q = (q0, q1, q2, q3)), and there are two ranks. For simplicity, we will discuss
    only q and k, as v follows the same pattern as k.

    The diagram below represents a complete QK^T operation without parallelism.
    The `****` entries indicate that the result is not required due to causal
    masking (e.g., q0k1 is marked as `****`).

    +----+------------------------+
    |    |  k0    k1   k2     k3  |
    +----+------------------------+
    | q0 | q0k0, ****, ****, **** |
    | q1 | q1k0, q1k1, ****, **** |
    | q2 | q2k0, q2k1, q2k2, **** |
    | q3 | q3k0, q3k1, q3k2, q3k3 |
    +----+------------------------+

    ### No Load Balance:

    In this scenario, each rank owns a local chunk of q, k, and v, with each chunk
    containing two elements. Rank0 is responsible for managing (q0, q1) and (k0, k1),
    while rank1 manages (q2, q3) and (k2, k3).

    First Iteration: Both rank0 and rank1 perform SDPA with their local qkv pairs.
    Causal masking is enabled as some results are not required (e.g., q0k1).

    Second Iteration: Local queries remain the same, but local kv pairs are exchanged.
    Rank0 now has (q0, q1) and (k2, k3); rank1 has (q2, q3) and (k0, k1). Rank0 performs
    no computation, while rank1 computes locally without causal masking since all results
    (q2k0, q2k1, q3k0, q3k1) are needed.

    ### Round-robin Load Balance:

    In this setup, each rank owns two local chunks of q, k, and v, with each chunk
    containing one element. Rank0 manages (q0, q3) and (k0, k3); Rank1 manages (q1, q2)
    and (k1, k2). Although the local chunks are not consecutive, they are concatenated to
    enable SDPA to be performed in a single call for each step. Consequently, the chunk()
    function may be required to prepare the correct q, k, and v configurations.

    First Iteration: Both ranks perform SDPA with their local qkv pairs, similar to the
    no-load-balance case. This iteration corresponds to the `if` of the
    (`if, `elif`, `else`) in the implementation.

    Second Iteration: Rank0 now has (q0, q3) and (k1, k2); rank1 has (q1, q2) and
    (k0, k3). For rank0, no computation is needed for q0. However, computations for
    q3k1 and q3k2 are required, so only q3 is used for SDPA. This corresponds to the
    `else` of the (`if`, `elif`, `else`) in the implementation.
    For rank1, k3 is not needed for q1 and q2, so only k0 is used for SDPA. This
    corresponds to the `elif` of (`if`, `elif`, `else`) in the implementation.

    Parameters
    ----------
    op:
        The attention op to use
    *args:
        additional args are passed to the op
    **kwargs:
        additional kwargs are passed to the op

    Returns
    -------
    out:
        The merged attention output
    softmax_lse:
        The logsumexp of the merged attention output
    """
    if is_causal and (query.size(2) != key.size(2)):
        raise NotImplementedError(
            "is_causal requires the same query and context sequence lengths"
        )
    if not is_causal and _cp_options.enable_load_balance:
        raise RuntimeError("Load balancing requires `is_causal=True`.")

    if not isinstance(group, dist.ProcessGroup):
        raise AssertionError("process group must be single dimension")
    rank = dist.get_rank(group)
    size = dist.get_world_size(group)

    next_kv = None

    # Without making key and value contiguous(), the loss curve is bad.
    # TODO(fegin): figure out why this is a requirement since SDPA does not have
    # this requirement.
    key = key.contiguous()
    value = value.contiguous()

    sdpa_merger = _SDPAMerger(_cp_options.convert_to_f32, seq_dim=seq_dim)

    saved_rest: list[Any] | None = None
    out: torch.Tensor
    logsumexp: torch.Tensor

    rotater = _create_rotater(group, 2)

    for i in range(size):
        if i > 0:
            # Wait for the kv from the (cp_rank - 1) rank.
            next_kv = rotater.next_buffer()
            key = next_kv[: key.numel()].reshape(key.shape)
            value = next_kv[key.numel() :].reshape(value.shape)

        if i < (size - 1):
            # Send the k, v to the next rank
            next_kv = torch.cat([key.flatten(), value.flatten()])
            next_kv = rotater.exchange_buffers(next_kv)

        is_causal_behavior = _is_causal_behavior(
            rank=rank, world_size=size, i=i, is_causal=is_causal
        )

        # For a detailed understanding of the load balancing algorithm, see
        # Note [Context parallelism load balance algorithm for causal masking]
        if is_causal_behavior == _CausalBehavior.SKIP:
            # If i > rank and load balancing is not turned on.
            continue

        if i == 0 or (not _cp_options.enable_load_balance or not is_causal):
            # When local balance is enabled, we still need to do SDPA with
            # the both local chunks of q, k, v for the first iteration.
            q, k, v, partial = (query, key, value, False)
        elif i <= rank:
            # Round-robin load balancing case, and i <= rank.
            # We need to do SDPA with only the first local chunk of k, v.
            # Note that q, k, v each contains two local chunks.
            ROUND_ROBIN_CYCLE = 2
            q, k, v, partial = (
                query,
                key.chunk(ROUND_ROBIN_CYCLE, dim=2)[0],
                value.chunk(ROUND_ROBIN_CYCLE, dim=2)[0],
                False,
            )
        else:
            # Round-robin load balancing case, and i > rank.
            # We need to do SDPA with only the second half of q, and update
            # only the second part of logsumexp. So partial is True.
            # Note that q, k, v each contains two chunks.
            q, k, v, partial = query.chunk(2, dim=2)[1], key, value, True

        # See https://github.com/pytorch/pytorch/blob/release/2.4/aten/src/ATen/native/native_functions.yaml#L14695
        # for the SDPA kernel definitions.
        out, logsumexp, *rest = op(
            q,
            k,
            v,
            is_causal=is_causal_behavior.value,
            **kwargs,
        )
        if saved_rest is None:
            saved_rest = rest
        sdpa_merger.step(out, logsumexp, partial)

    if saved_rest is None:
        raise AssertionError("No SDPA op was executed in ring attention forward")
    return *sdpa_merger.results(), *saved_rest


def _templated_ring_attention_backward(
    group: dist.ProcessGroup,
    seq_dim: int,
    op: _AttentionOp,
    grad_out: torch.Tensor,
    grad_out_name: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    is_causal: bool,
    **kwargs: Any,
) -> tuple[torch.Tensor, ...]:
    """This API implements the backward pass of the ring attention."""
    if not is_causal and _cp_options.enable_load_balance:
        raise RuntimeError("Load balancing requires `is_causal=True`.")
    rank = dist.get_rank(group)
    size = dist.get_world_size(group)
    next_kv = None
    next_grad_kv = None
    rest: list[Any]
    grad_query_, grad_key_, grad_value_ = None, None, None

    accum_dtype = torch.float32 if _cp_options.convert_to_f32 else query.dtype
    grad_query = torch.zeros_like(query, dtype=accum_dtype)
    grad_key = torch.zeros_like(key, dtype=accum_dtype)
    grad_value = torch.zeros_like(value, dtype=accum_dtype)

    key = key.contiguous()
    value = value.contiguous()
    kv_rotater = _create_rotater(group, 2)
    dkv_rotater = _create_rotater(group, 2, method=_RotateMethod.ALL_TO_ALL)
    for i in range(size):
        if i > 0:
            # Wait for the kv from the (cp_rank - 1) rank.
            buffer = kv_rotater.next_buffer()
            pointer = 0
            key = buffer[pointer : pointer + key.numel()].reshape(key.shape)
            pointer += key.numel()
            value = buffer[pointer : pointer + value.numel()].reshape(value.shape)
            pointer += value.numel()

        if i != size - 1:
            # Send the kv to the next rank.
            next_kv = torch.cat([key.flatten(), value.flatten()])
            kv_rotater.exchange_buffers(next_kv)

        is_causal_behavior = _is_causal_behavior(
            rank=rank, world_size=size, i=i, is_causal=is_causal
        )

        if is_causal_behavior != _CausalBehavior.SKIP:
            if i == 0 or (not _cp_options.enable_load_balance or not is_causal):
                # We need to do SDPA with the full local q, k, v.
                q, k, v, out_, dout, lse = (query, key, value, out, grad_out, logsumexp)
            elif i <= rank:
                # Round-robin load balancing case, and i <= rank.
                # We need to do SDPA with only the first half of k, v.
                # Note that q, k, v each contains two chunks.
                q, k, v, out_, dout, lse = (
                    query,
                    key.chunk(2, dim=seq_dim)[0],
                    value.chunk(2, dim=seq_dim)[0],
                    out,
                    grad_out,
                    logsumexp,
                )
            else:
                # Round-robin load balancing case, and i > rank.
                # We need to do SDPA with only the second half of q.
                # Note that q, k, v each contains two chunks.
                q, k, v, out_, dout, lse = (
                    query.chunk(2, dim=seq_dim)[1],
                    key,
                    value,
                    out.chunk(2, dim=seq_dim)[1],
                    grad_out.chunk(2, dim=seq_dim)[1],
                    # Need to make logsumexp contiguous, otherwise there will
                    # be numerical error.
                    logsumexp.chunk(2, dim=seq_dim)[1].contiguous(),
                )

            kwargs[grad_out_name] = dout
            iter_kwargs = kwargs
            if _cp_options.enable_load_balance and i > 0:
                iter_kwargs = dict(kwargs)
                if "max_q" in iter_kwargs:
                    iter_kwargs["max_q"] = q.shape[seq_dim]
                if "max_k" in iter_kwargs:
                    iter_kwargs["max_k"] = k.shape[seq_dim]
            # See https://github.com/pytorch/pytorch/blob/release/2.4/aten/src/ATen/native/native_functions.yaml#L14695
            # for the SDPA kernel definitions.
            grad_query_, grad_key_, grad_value_, *rest = op(
                query=q,
                key=k,
                value=v,
                out=out_,
                logsumexp=lse,
                is_causal=is_causal_behavior.value,
                **iter_kwargs,
            )
        else:
            grad_query_ = torch.zeros_like(query, dtype=accum_dtype)
            grad_key_ = torch.zeros_like(key, dtype=accum_dtype)
            grad_value_ = torch.zeros_like(value, dtype=accum_dtype)

        ROUND_ROBIN_CYCLE = 2
        if i == 0:
            grad_key += grad_key_
            grad_value += grad_value_
        else:
            pointer = 0
            # Wait for the kv gradient from (cp_rank - 1) rank.
            next_grad_kv = dkv_rotater.next_buffer()
            grad_key = next_grad_kv[pointer : pointer + grad_key.numel()].reshape(
                grad_key.shape
            )
            pointer += grad_key.numel()
            grad_value = next_grad_kv[pointer : pointer + grad_value.numel()].reshape(
                grad_value.shape
            )

            if i <= rank and _cp_options.enable_load_balance:
                grad_key = _partial_update(
                    grad_key,
                    grad_key_,
                    dim=seq_dim,
                    n_chunks=ROUND_ROBIN_CYCLE,
                    idx=0,
                    add=True,
                )
                grad_value = _partial_update(
                    grad_value,
                    grad_value_,
                    dim=seq_dim,
                    n_chunks=ROUND_ROBIN_CYCLE,
                    idx=0,
                    add=True,
                )
            else:
                grad_key += grad_key_
                grad_value += grad_value_

        next_grad_kv = torch.cat([grad_key.flatten(), grad_value.flatten()])
        # Send the grad key and grad value to the next rank.
        dkv_rotater.exchange_buffers(next_grad_kv)

        if i <= rank or not _cp_options.enable_load_balance:
            grad_query += grad_query_
        else:
            grad_query = _partial_update(
                grad_query,
                grad_query_,
                dim=seq_dim,
                n_chunks=ROUND_ROBIN_CYCLE,
                idx=1,
                add=True,
            )

    if grad_key_ is None:
        raise AssertionError
    if grad_value_ is None:
        raise AssertionError
    grad_query = grad_query.to(query.dtype)
    next_grad_kv = dkv_rotater.next_buffer().to(key.dtype)
    grad_key = next_grad_kv[: grad_key.numel()].reshape(grad_key.shape)
    grad_value = next_grad_kv[grad_key.numel() :].reshape(grad_value.shape)
    return (
        grad_query,
        grad_key,
        grad_value,
        # pyrefly: ignore [unbound-name]
        *rest,
    )


def _scaled_dot_product_ring_flash_attention(
    mesh: DeviceMesh,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: float | None = None,
) -> tuple[torch.Tensor, ...]:
    if return_debug_mask:
        raise NotImplementedError("return_debug_mask is not supported yet")

    # TODO: remove this hardcoding
    seq_dim = 2
    group = mesh.get_group()
    return _templated_ring_attention(
        group,
        seq_dim,
        aten._scaled_dot_product_flash_attention,
        query=query,
        key=key,
        value=value,
        is_causal=is_causal,
        dropout_p=dropout_p,
        scale=scale,
    )


def _scaled_dot_product_ring_efficient_attention(
    mesh: DeviceMesh,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor | None = None,
    compute_log_sumexp: bool = True,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: float | None = None,
) -> tuple[torch.Tensor, ...]:
    if attn_bias is not None:
        raise NotImplementedError("attn_bias is not supported yet")

    if not compute_log_sumexp:
        # CP requires compute_log_sumexp to be True because it always merges LSE
        compute_log_sumexp = True

    # TODO: remove this hardcoding
    seq_dim = 2
    group = mesh.get_group()
    return _templated_ring_attention(
        group,
        seq_dim,
        aten._scaled_dot_product_efficient_attention,
        query=query,
        key=key,
        value=value,
        is_causal=is_causal,
        attn_bias=attn_bias,
        dropout_p=dropout_p,
        scale=scale,
        compute_log_sumexp=compute_log_sumexp,
    )


def _scaled_dot_product_ring_cudnn_attention(
    mesh: DeviceMesh,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor | None = None,
    compute_log_sumexp: bool = True,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: float | None = None,
) -> tuple[torch.Tensor, ...]:
    if attn_bias is not None:
        raise NotImplementedError("attn_bias is not supported yet")

    if not compute_log_sumexp:
        # CP requires compute_log_sumexp to be True because it always merges LSE
        compute_log_sumexp = True

    # TODO: remove this hardcoding
    seq_dim = 2
    group = mesh.get_group()
    return _templated_ring_attention(
        group,
        seq_dim,
        aten._scaled_dot_product_cudnn_attention,
        query=query,
        key=key,
        value=value,
        attn_bias=attn_bias,
        compute_log_sumexp=compute_log_sumexp,
        dropout_p=dropout_p,
        is_causal=is_causal,
        return_debug_mask=return_debug_mask,
        scale=scale,
    )


def _scaled_dot_product_ring_flash_attention_backward(
    mesh: DeviceMesh,
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    cum_seq_q: torch.Tensor,
    cum_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    *,
    scale: float | None = None,
) -> tuple[torch.Tensor, ...]:
    # TODO: remove this hardcoding
    seq_dim = 2
    group = mesh.get_group()
    return _templated_ring_attention_backward(
        group,
        seq_dim,
        aten._scaled_dot_product_flash_attention_backward.default,
        grad_out=grad_out,
        grad_out_name="grad_out",
        query=query,
        key=key,
        value=value,
        out=out,
        logsumexp=logsumexp,
        is_causal=is_causal,
        cum_seq_q=cum_seq_q,
        cum_seq_k=cum_seq_k,
        max_q=max_q,
        max_k=max_k,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        scale=scale,
    )


def _scaled_dot_product_ring_efficient_attention_backward(
    mesh: DeviceMesh,
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    dropout_p: float,
    grad_input_mask: tuple[bool, ...],
    is_causal: bool = False,
    *,
    scale: float | None = None,
) -> tuple[torch.Tensor, ...]:
    # TODO: remove this hardcoding
    seq_dim = 2
    group = mesh.get_group()
    return _templated_ring_attention_backward(
        group,
        seq_dim,
        aten._scaled_dot_product_efficient_attention_backward.default,
        grad_out=grad_out,
        grad_out_name="grad_out_",
        query=query,
        key=key,
        value=value,
        attn_bias=bias,
        out=out,
        logsumexp=logsumexp,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        dropout_p=dropout_p,
        grad_input_mask=grad_input_mask,
        is_causal=is_causal,
        scale=scale,
    )


def _scaled_dot_product_ring_cudnn_attention_backward(
    mesh: DeviceMesh,
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    attn_bias: torch.Tensor,
    cum_seq_q: torch.Tensor,
    cum_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    *,
    scale: float | None = None,
) -> tuple[torch.Tensor, ...]:
    # TODO: remove this hardcoding
    seq_dim = 2
    group = mesh.get_group()
    return _templated_ring_attention_backward(
        group,
        seq_dim,
        aten._scaled_dot_product_cudnn_attention_backward.default,
        grad_out=grad_out,
        grad_out_name="grad_out",
        query=query,
        key=key,
        value=value,
        out=out,
        logsumexp=logsumexp,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        attn_bias=attn_bias,
        cum_seq_q=cum_seq_q,
        cum_seq_k=cum_seq_k,
        max_q=max_q,
        max_k=max_k,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


def _sdpa_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    # extract local tensor and sharding infos to a OpInfo
    op_info = DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)
    logger.debug("Dispatching op_call: %s", op_info.schema or op_call)

    # sharding propagation
    # TODO: remove the context parallel strategy from the default propagation
    # rule. Either figure out how to dynamically enable it or just don't call
    # propagate.
    DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    if output_sharding is None:
        raise AssertionError("output sharding should not be None")
    if output_sharding.needs_redistribute:
        raise AssertionError("inputs need to be redistributed")

    call_maps: dict[torch._ops.OpOverload, Callable] = {
        aten._scaled_dot_product_flash_attention.default: _scaled_dot_product_ring_flash_attention,
        aten._scaled_dot_product_efficient_attention.default: _scaled_dot_product_ring_efficient_attention,
        aten._scaled_dot_product_cudnn_attention.default: _scaled_dot_product_ring_cudnn_attention,
        aten._scaled_dot_product_flash_attention_backward.default: _scaled_dot_product_ring_flash_attention_backward,
        aten._scaled_dot_product_efficient_attention_backward.default: _scaled_dot_product_ring_efficient_attention_backward,
        aten._scaled_dot_product_cudnn_attention_backward.default: _scaled_dot_product_ring_cudnn_attention_backward,
    }
    if op_call in call_maps:
        local_results = call_maps[op_call](
            op_info.compute_mesh,
            *op_info.local_args,  # type: ignore[arg-type]
            **op_info.local_kwargs,  # type: ignore[arg-type]
        )
    else:
        raise NotImplementedError(
            "CP only supports flash attention and memory efficient attention now."
        )

    return DTensor._op_dispatcher.wrap(local_results, output_sharding.output_spec)


custom_ops = {
    aten._scaled_dot_product_flash_attention.default: _sdpa_handler,
    aten._scaled_dot_product_flash_attention_backward.default: _sdpa_handler,
    aten._scaled_dot_product_efficient_attention.default: _sdpa_handler,
    aten._scaled_dot_product_efficient_attention_backward.default: _sdpa_handler,
    aten._scaled_dot_product_cudnn_attention.default: _sdpa_handler,
    aten._scaled_dot_product_cudnn_attention_backward.default: _sdpa_handler,
}
existing_custom_ops = DTensor._op_dispatcher._custom_op_handlers


ArgsType = tuple[Any, ...]
KwargsType = dict[str, Any]
InputFnType = Callable[[nn.Module | None, ArgsType, KwargsType, DeviceMesh], Any]
OutputFnType = Callable[[nn.Module | None, Any, Any, DeviceMesh], Any]

_replaced_functions: dict[Callable, tuple[str, Callable]] = {}


def _distribute_function(
    fn: Callable,
    fn_module: types.ModuleType,
    device_mesh: DeviceMesh,
    input_fn: InputFnType,
    output_fn: OutputFnType,
) -> None:
    """
    A helper function to replace a function with a distributed version by
    using the monkey patching approach.

    This function is for the CP internal usage only.
    """

    def wrapper(
        target_fn: Callable, input_fn: InputFnType, output_fn: OutputFnType
    ) -> Callable:
        def inner_fn(*args: ArgsType, **kwargs: KwargsType) -> Any:
            args, kwargs = input_fn(None, args, kwargs, device_mesh)
            outputs = target_fn(*args, **kwargs)
            return output_fn(None, (args, kwargs), outputs, device_mesh)

        return inner_fn

    global _replaced_functions

    if fn in _replaced_functions:
        return

    wrapper_fn = wrapper(fn, input_fn, output_fn)
    setattr(fn_module, fn.__name__, wrapper_fn)
    _replaced_functions[wrapper_fn] = (fn.__name__, fn)


def _restore_function(fn: Callable, fn_module: types.ModuleType) -> None:
    """Restore the function that is replaced by _distribute_function."""
    if fn not in _replaced_functions:
        return

    original_name, original_fn = _replaced_functions[fn]
    setattr(fn_module, original_name, original_fn)


def _enable_cp_dtensor_dispatcher() -> None:
    """Enables DTensor dispatcher to dispatch SDPA to CP."""
    # Enable custom op handlers for CP
    DTensor._op_dispatcher._custom_op_handlers = {
        **existing_custom_ops,
        **custom_ops,
    }
    # Register CP-specific sharding rules
    from ._sharding_rules import register_cp_sharding_rules

    register_cp_sharding_rules()


def _disable_cp_dtensor_dispatcher() -> None:
    """Disables DTensor dispatcher to dispatch SDPA to CP."""
    # Restore original custom op handlers
    DTensor._op_dispatcher._custom_op_handlers = existing_custom_ops

    # TODO: unregister_cp_sharding_rules(clear_the_cache=True) will cause
    # all DTensor sharding propagation cache being invalidated. It is not
    # easy to achieve selectively invalidating lru cache without rewriting
    # the sharding propagation wrapper.

    from ._sharding_rules import unregister_cp_sharding_rules

    unregister_cp_sharding_rules(clear_the_cache=False)


def _enable_context_parallel_dispatcher_impl(seq_dim: int, mesh: DeviceMesh) -> None:
    sdpa_cp = _ContextParallel(
        seq_dim=seq_dim,
        attention_type=_ContextParallel.AttentionType.SDPA,
    )

    if _dispatch_mode == _DispatchMode.MONKEY_PATCH:
        _distribute_function(
            F.scaled_dot_product_attention,
            F,
            mesh,
            sdpa_cp.sdpa_input_fn,
            sdpa_cp.sdpa_output_fn,
        )
        _enable_cp_dtensor_dispatcher()
    elif _dispatch_mode == _DispatchMode.MODULE_WRAPPER:
        _enable_cp_dtensor_dispatcher()
    else:
        raise ValueError(f"Unknown dispatch mode: {_dispatch_mode}")


def _disable_context_parallel_dispatcher_impl() -> None:
    if _dispatch_mode == _DispatchMode.MONKEY_PATCH:
        _restore_function(F.scaled_dot_product_attention, F)
    elif _dispatch_mode == _DispatchMode.MODULE_WRAPPER:
        pass
    else:
        raise NotImplementedError(f"Unknown dispatch mode: {_dispatch_mode}")

    _disable_cp_dtensor_dispatcher()


_compiled_create_block_mask = None


def _context_parallel_buffers(
    mesh: DeviceMesh,
    buffers: list[torch.Tensor | BlockMask],
    buffer_seq_dims: list[int],
    load_balancer: _LoadBalancer | None = None,
) -> list[torch.Tensor | BlockMask]:
    """
    Shard the buffers along the sequence dimensions according to CP rules.
    Args:
        mesh (:class:`DeviceMesh`): the device mesh for the context parallelism.
        buffers (List[torch.Tensor]): the buffers to be sharded.
        seq_dims (List[int]): the sequence dimensions of ``buffers``. This list
            must have the same length as ``buffers``.
        load_balancer (Optional[:class:`_LoadBalancer`]): an optional `_LoadBalancer`
            object. If this argument is `None`, it means the `buffers` need no
            rearrangement before being sharded. If this argument is a `_LoadBalancer`
            object, call its `_generate_indices(restore=False)` to generate the
            rearrangement indices such that each shard of `buffer[rearrange_idx]` is
            well-balanced (i.e., having close sparsities).

    Returns:
        List[torch.Tensor]: the sharded buffers.

    Note:
        For `_context_parallel_shard` we require a non-None `load_balancer` object to be
        explicitly passed if load-balancing is needed.
    """
    # generate the index tensor for rearranging the buffer if a load-balance
    # is available
    load_balance_indices = load_balancer._generate_indices() if load_balancer else None
    if not (load_balance_indices is None or load_balance_indices.ndim == 2):
        raise AssertionError(
            "load balance index expects shape (1, seq_len) or (B, seq_len) "
            f"but got {load_balance_indices.shape}."
        )

    new_buffers = []
    sharded_buffer: torch.Tensor | BlockMask
    for buffer, seq_dim in zip(buffers, buffer_seq_dims):
        if isinstance(buffer, torch.Tensor):
            # NOTE: assuming batch dim is 0

            if load_balance_indices is not None:
                # TODO: we should explicitly ask users to unsqueeze the batch dim.
                # But this is a BC breaking ask.
                # However, what we have done today is also not very safe.
                idx_batch_size = load_balance_indices.size(0)
                data_batch_size = buffer.size(0) if seq_dim > 0 else 1

                if idx_batch_size != 1 and idx_batch_size != data_batch_size:
                    raise ValueError(
                        "Cannot rearrange buffer: "
                        f"load_balance_indices has shape {load_balance_indices.shape}, "
                        f"but buffer has shape {buffer.shape}."
                    )

                if seq_dim == 0:
                    # buffer has shape [seq_len] or [seq_len, ...]
                    # Just use the first (and only) batch of indices
                    buffer = torch.index_select(
                        buffer, dim=0, index=load_balance_indices[0]
                    )
                else:
                    indices = load_balance_indices
                    if idx_batch_size == 1:
                        size = [data_batch_size] + list(indices.size())[1:]
                        indices = indices.expand(*size)

                    # load_balance_indices that has shape [B, seq_len] where:
                    #   - dim 0 corresponds to buffer dim 0 (batch)
                    #   - dim 1 corresponds to buffer dim seq_dim
                    # Need to insert dimensions for all dims between 0 and seq_dim,
                    # and all dims after seq_dim.

                    # Insert dimensions between batch (dim 0) and seq_dim
                    for i in range(1, seq_dim):
                        indices = indices.unsqueeze(i)

                    # Insert dimensions after seq_dim
                    for _ in range(seq_dim + 1, buffer.ndim):
                        indices = indices.unsqueeze(-1)

                    # Expand to match buffer's shape
                    indices = indices.expand(buffer.shape)

                    buffer = torch.gather(buffer, dim=seq_dim, index=indices)

            # use DTensor to shard the buffer on sequence dimension,
            # retain the local tensor
            sharded_buffer = distribute_tensor(
                buffer, mesh, [Shard(seq_dim)], src_data_rank=None
            ).to_local()
        elif isinstance(buffer, BlockMask):
            sharded_buffer = _create_cp_block_mask(
                mask_mod=buffer.mask_mod,
                B=buffer.kv_num_blocks.shape[0],
                H=buffer.kv_num_blocks.shape[1],
                Q_LEN=buffer.seq_lengths[0],
                KV_LEN=buffer.seq_lengths[1],
                device_mesh=mesh,
                load_balancer=load_balancer,
            )
        else:
            raise ValueError(f"Unknown buffer type: {type(buffer)}")

        new_buffers.append(sharded_buffer)

    return new_buffers


def _create_cp_block_mask(
    mask_mod: _mask_mod_signature,
    B: int,
    H: int,
    Q_LEN: int,
    KV_LEN: int,
    device_mesh: DeviceMesh,
    load_balancer: _LoadBalancer | None = None,
) -> BlockMask:
    """
    Creates a specialized BlockMask for Context Parallel FlexAttention.

    This function creates a BlockMask that enables computation of attention results
    for sharded Q attending to global KV. The mask appropriately handles the query
    index offset required when each rank operates on a shard of the query sequence
    while accessing the full key-value sequence.

    The function internally rewrites the provided mask_mod function to translate local
    query indices to global query indices, ensuring that the masking logic is applied
    correctly across the distributed computation.

    Args:
        mask_mod (Callable): Mask function that operates on global attention indices.
        B (int): Batch size.
        H (int): Number of query heads.
        Q_LEN (int): Global sequence length of the query.
        KV_LEN (int): Global sequence length of the key/value.
        device_mesh (DeviceMesh): Device mesh used for context parallelism.
        load_balancer (Optional[:class:`_LoadBalancer`]): The load-balancer used to rearrange
            QKV before sharding. This will be used to modify the block_mask generated.

    Returns:
        BlockMask: A block mask configured for the local query shard that can be used
            with flex_attention() for the given cp_mesh.

    Raises:
        NotImplementedError: If Q_LEN is not divisible by (CP world size * BLOCK_SIZE).

    Warning:
        Currently requires Q_LEN to be divisible by CP mesh world size * BLOCK_SIZE
        (BLOCK_SIZE defaults to 128). This constraint exists because the BlockMask
        must handle both padding and offsets correctly. For example, if Q_LEN is 384,
        CP world size is 2, and BLOCK_SIZE is 128, the local Q_LEN would be 192. In
        such cases, both rank0 and rank1 would have paddings in their local BlockMasks.
        Support for padding in this scenario is planned for future work.

    """

    from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE

    if Q_LEN % (device_mesh.size() * _DEFAULT_SPARSE_BLOCK_SIZE) != 0:
        raise NotImplementedError(
            f"Q_LEN {Q_LEN} is not divisible by CP mesh world size {device_mesh.size()} * "
            f"BLOCK_SIZE {_DEFAULT_SPARSE_BLOCK_SIZE}. This is not supported yet. "
        )

    global _compiled_create_block_mask
    if _compiled_create_block_mask is None:
        _compiled_create_block_mask = torch.compile(
            create_block_mask, dynamic=False, fullgraph=True
        )
    compiled_create_block_mask = _compiled_create_block_mask

    def _rewrite_mask_mod(
        mask_mod: _mask_mod_signature,
        rank: int,
        block_size: int,
        local_q_size: int,
        qkv_rearrange_indices: torch.Tensor | None = None,
    ) -> _mask_mod_signature:
        if not (qkv_rearrange_indices is None or qkv_rearrange_indices.ndim == 2):
            raise AssertionError(
                "load balance index expects shape (1, seq_len) or (B, seq_len) "
                f"but got {qkv_rearrange_indices.shape}."
            )

        def qkv_idx_restore(
            b: torch.Tensor, idx_post_rearrange: torch.Tensor
        ) -> torch.Tensor:
            if qkv_rearrange_indices is not None:
                if (
                    qkv_rearrange_indices.size(0) == 1
                ):  # identical load-balance in batch
                    idx_pre_rearrange = qkv_rearrange_indices[0][idx_post_rearrange]
                else:
                    idx_pre_rearrange = qkv_rearrange_indices[b][idx_post_rearrange]
            else:
                idx_pre_rearrange = idx_post_rearrange

            return idx_pre_rearrange

        def local_q_idx_to_q_idx(local_q_idx: torch.Tensor) -> torch.Tensor:
            # calculate local block_idx and block_offset
            local_blk_idx, local_blk_offset = (
                local_q_idx // block_size,
                local_q_idx % block_size,
            )
            # NOTE: load balancing is not used
            local_num_blocks = local_q_size // block_size
            blk_idx = local_num_blocks * rank + local_blk_idx
            return blk_idx * block_size + local_blk_offset

        return lambda b, h, q_idx, kv_idx: mask_mod(
            b,
            h,
            qkv_idx_restore(b, local_q_idx_to_q_idx(q_idx)),
            qkv_idx_restore(b, kv_idx),
        )

    cp_rank = device_mesh.get_local_rank()
    cp_group_size = device_mesh.size()
    load_balancer = load_balancer or _create_default_load_balancer(
        Q_LEN, cp_group_size, device_mesh.device_type
    )
    Q_SHARD_LEN = Q_LEN // cp_group_size
    block_size = _DEFAULT_SPARSE_BLOCK_SIZE

    rearrange_indices = (
        load_balancer._generate_indices(restore=False) if load_balancer else None
    )
    block_mask = compiled_create_block_mask(
        _rewrite_mask_mod(
            mask_mod,
            cp_rank,
            block_size,
            Q_SHARD_LEN,
            qkv_rearrange_indices=rearrange_indices,
        ),
        B,
        H,
        Q_SHARD_LEN,
        KV_LEN,
        device=device_mesh.device_type,
        BLOCK_SIZE=(block_size, block_size),
    )
    return block_mask


#####################
# Experimental APIs
#####################


class _ContextParallel(ParallelStyle):
    class AttentionType(Enum):
        FLEX = "flex_attention"
        SDPA = "scaled_dot_product_attention"

    def __init__(
        self,
        seq_dim: int,
        attention_type: AttentionType,
    ) -> None:
        super().__init__()
        self.seq_dim = seq_dim
        self.attention_type = attention_type

    def _apply(self, module: nn.Module, mesh: DeviceMesh) -> nn.Module:
        if self.attention_type == self.AttentionType.FLEX:
            module.register_forward_pre_hook(
                partial(self.flex_input_fn, mesh=mesh), with_kwargs=True
            )
            return module
        elif self.attention_type == self.AttentionType.SDPA:
            module.register_forward_pre_hook(
                partial(self.sdpa_input_fn, mesh=mesh),
                with_kwargs=True,
            )
            module.register_forward_hook(partial(self.sdpa_output_fn, mesh=mesh))
            return module
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")

    def flex_input_fn(
        self, module: nn.Module | None, args: Any, kwargs: Any, mesh: DeviceMesh
    ) -> Any:
        # We don't care about other args, and these argument order must be consistent
        # with the signature of flex_attention.
        expected_arg_names = ("query", "key", "value")
        args_list = list(args)
        for idx, name in enumerate(expected_arg_names):
            if idx >= len(args):
                args_list.append(kwargs.pop(name, None))

        query, key, value = args_list[: len(expected_arg_names)]
        if not isinstance(query, torch.Tensor):
            raise AssertionError
        if not isinstance(key, torch.Tensor):
            raise AssertionError
        if not isinstance(value, torch.Tensor):
            raise AssertionError

        key = key.contiguous()
        value = value.contiguous()

        global_key, global_value = flex_cp_allgather(
            key, value, self.seq_dim, c10d._get_process_group_name(mesh.get_group())
        )
        args_list[1] = global_key
        args_list[2] = global_value

        for idx in range(len(args), len(expected_arg_names)):
            kwargs[expected_arg_names[idx]] = args_list[idx]
        args_list = args_list[: len(args)]
        return tuple(args_list), kwargs

    def sdpa_input_fn(
        self,
        module: nn.Module | None,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        mesh: DeviceMesh,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        placement = [Shard(self.seq_dim)]
        all_args = []

        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, torch.Tensor):
                if isinstance(arg, DTensor):
                    if arg._spec.placements != placement:
                        raise AssertionError
                else:
                    arg = DTensor.from_local(arg, mesh, placement, run_check=False)

            all_args.append(arg)

        new_args = tuple(all_args[0 : len(args)])
        new_kwargs = dict(zip(kwargs.keys(), all_args[len(args) :]))
        return new_args, new_kwargs

    def sdpa_output_fn(
        self, module: nn.Module | None, inputs: Any, outputs: Any, mesh: DeviceMesh
    ) -> Any:
        new_outputs = []
        for output in [outputs] if isinstance(outputs, torch.Tensor) else outputs:
            output = output.to_local() if isinstance(output, DTensor) else output
            new_outputs.append(output)

        if isinstance(outputs, torch.Tensor):
            return new_outputs[0]

        return tuple(new_outputs)


CPBuffer: TypeAlias = torch.Tensor | BlockMask
CPBufferContainer: TypeAlias = Sequence[CPBuffer] | Mapping[str, CPBuffer]
CPBufferSeqDims: TypeAlias = Sequence[int] | Mapping[str, int]


def _context_parallel_shard(
    mesh: DeviceMesh,
    buffers: CPBufferContainer,
    seq_dims: CPBufferSeqDims,
    load_balancer: _LoadBalancer | None = None,
) -> list[torch.Tensor | BlockMask]:
    """
    Shard the buffers along the specified sequence dimensions (`seq_dims`), so that each
    rank retains only its corresponding shard according to the provided `mesh`. If a
    `load_balancer` is provided, the buffers will be rearranged by the load balancer
    before sharding to improve load balance. Buffers can be either tensors or `BlockMask`
    objects. If a buffer is a `BlockMask`, its sharding dimension is determined by the
    `BlockMask` implementation, and the corresponding `seq_dim` is ignored.

    Note:
        For `_context_parallel_shard`, a non-None `load_balancer` must be explicitly passed
        if load balancing is required.

    Args:
        mesh (DeviceMesh): The device mesh used for context parallelism.
        buffers (List[torch.Tensor | BlockMask]): Buffers whose usage depends on the sequence
            dimension. Examples include input batches, labels, and positional embedding buffers.
            These buffers must be sharded along the sequence dimension to ensure correctness.
        seq_dims (List[int]): The sequence dimensions for each buffer in `buffers`. Must have
            the same length as `buffers`.
        load_balancer (Optional[_LoadBalancer]): An optional load balancer object. If provided,
            it rearranges the buffers before sharding to achieve better load balance. If not
            provided, no rearrangement is performed.

    Returns:
