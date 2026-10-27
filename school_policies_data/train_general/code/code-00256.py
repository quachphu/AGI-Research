        self,
        *,
        output_layouts: Placement | tuple[Placement | None, ...],
        desired_output_layouts: Placement | tuple[Placement, ...],
        use_local_output: bool = True,
    ):
        self.output_layouts = (
            (output_layouts,)
            if isinstance(output_layouts, Placement)
            else output_layouts
        )
        self.desired_output_layouts = (
            (desired_output_layouts,)
            if isinstance(desired_output_layouts, Placement)
            else desired_output_layouts
        )
        self.use_local_output = use_local_output
        if len(self.output_layouts) != len(self.desired_output_layouts):
            raise AssertionError(
                "output_layouts and desired_output_layouts should have same length!"
            )

    def _prepare_out_fn(self, outputs, device_mesh):
        prepared_outputs = []
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        if len(outputs) != len(self.output_layouts):
            raise ValueError(
                "module outputs and output_layouts should have same length!"
            )

        for out, out_layout, desired_out_layout in zip(
            outputs, self.output_layouts, self.desired_output_layouts
        ):
            if out_layout is not None:
                if isinstance(out, DTensor):
                    # TODO: re-enable the check once we fix the compile path
                    # assert out.placements[0] == out_layout
                    dt_out = out
                else:
                    dt_out = DTensor.from_local(
                        out, device_mesh, (out_layout,), run_check=False
                    )

                if out_layout != desired_out_layout:
                    dt_out = dt_out.redistribute(placements=(desired_out_layout,))
                prepared_outputs.append(
                    dt_out.to_local() if self.use_local_output else dt_out
                )
            else:
                prepared_outputs.append(out)
        if len(prepared_outputs) == 1:
            return prepared_outputs[0]
        else:
            return tuple(prepared_outputs)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        module.register_forward_hook(
            lambda _, inputs, outputs: self._prepare_out_fn(outputs, device_mesh)
        )  # type: ignore[misc, call-arg]
        return module

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        tmpstr += f"output_layouts={self.output_layouts}, "
        tmpstr += f"desired_output_layouts={self.desired_output_layouts}, "
        tmpstr += f"use_local_output={self.use_local_output}"
        tmpstr += ")"
        return tmpstr


class PrepareModuleInputOutput(ParallelStyle):
    """
    Configure the nn.Module's inputs (and outputs) to convert the input tensors (and output tensors, respectively) of the nn.Module
    to DTensors at runtime according to ``input_layouts`` (and output_layouts, respectively), and perform layout redistribution
    according to the ``desired_input_layouts`` (and ``desired_output_layouts``, respectively). This is a combination of
    :class:`PrepareModuleInput` and :class:`PrepareModuleOutput`.

    Keyword Args:
        input_layouts (Union[Placement, Tuple[Optional[Placement]]]):
            The DTensor layouts of input tensors for the nn.Module, this is used to convert the input tensors to
            DTensors. If some inputs are not torch.Tensor or no need to convert to DTensors, ``None`` need to be specified
            as a placeholder. default: None.
        desired_input_layouts (Union[Placement, Tuple[Optional[Placement]]]):
            The desired DTensor layout of input tensors for the nn.Module, this is used to ensure the inputs of the nn.Module
            have the desired DTensor layouts. This argument needs to have the same length with ``input_layouts``. default: None.
        input_kwarg_layouts (Dict[str, Placement]):
            The DTensor layouts of input kwargs for the nn.Module, this is used to convert the input kwarg tensors to DTensors.
            default: None
        desired_input_kwarg_layouts: (Dict[str, Placement]):
            The desired DTensor layout of input kwargs for the nn.Module, this is used to ensure the inputs of the nn.Module
            have the desired DTensor layouts. default: None.
        use_local_input (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module inputs, default: False.
        output_layouts (Union[Placement, Tuple[Placement]]):
            The DTensor layouts of output tensors for the nn.Module, this is used to convert the output tensors to
            DTensors if they are :class:`torch.Tensor`. If some outputs are not torch.Tensor or no need to convert to DTensors,
            ``None`` need to be specified as a placeholder.
        desired_output_layouts (Union[Placement, Tuple[Placement]]):
            The desired DTensor layouts of output tensors for the nn.Module, this is used to ensure the outputs of the nn.Module
            have the desired DTensor layouts.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module outputs, default: True.
    Returns:
        A :class:`ParallelStyle` object that prepares the sharding layouts of the nn.Module's inputs and outputs.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, PrepareModuleInputOutput
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> block = TransformerBlock(...)  # block is a nn.Module that contains an "attn" Attention submodule
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>>
        >>> # According to the style specified below, the first input of attn will be annotated as Sharded DTensor
        >>> # and then redistributed to Replicated DTensor, and the output of the TransformerBlock will be annotated
        >>> # as Replicated DTensor and then redistributed to Sharded DTensor.
        >>> parallelize_module(
        >>>     block, # this can be a submodule or module
        >>>     tp_mesh,
        >>>     parallelize_plan={
        >>>         "attn": PrepareModuleInputOutput(
        >>>             input_layouts=(Shard(0), None, None, ...),
        >>>             desired_input_layouts=(Replicate(), None, None, ...),
        >>>             output_layouts=Replicate(),
        >>>             desired_output_layouts=Shard(0),
        >>>         ),
        >>>     }
        >>> )
    """

    def __init__(
        self,
        *,
        input_layouts: Placement | tuple[Placement | None, ...] | None = None,
        desired_input_layouts: Placement | tuple[Placement | None, ...] | None = None,
        input_kwarg_layouts: dict[str, Placement] | None = None,
        desired_input_kwarg_layouts: dict[str, Placement] | None = None,
        use_local_input: bool = False,
        output_layouts: Placement | tuple[Placement | None, ...],
        desired_output_layouts: Placement | tuple[Placement, ...],
        use_local_output: bool = True,
    ):
        self.prepare_module_input = PrepareModuleInput(
            input_layouts=input_layouts,
            desired_input_layouts=desired_input_layouts,
            input_kwarg_layouts=input_kwarg_layouts,
            desired_input_kwarg_layouts=desired_input_kwarg_layouts,
            use_local_output=use_local_input,
        )
        self.prepare_module_output = PrepareModuleOutput(
            output_layouts=output_layouts,
            desired_output_layouts=desired_output_layouts,
            use_local_output=use_local_output,
        )

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        self.prepare_module_input._apply(module, device_mesh)
        self.prepare_module_output._apply(module, device_mesh)

        return module

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        tmpstr += f"input_layouts={self.prepare_module_input.input_layouts}, "
        tmpstr += (
            f"desired_input_layouts={self.prepare_module_input.desired_input_layouts}, "
        )
        tmpstr += (
            f"input_kwarg_layouts={self.prepare_module_input.input_kwarg_layouts}, "
        )
        tmpstr += f"desired_input_kwarg_layouts={self.prepare_module_input.desired_input_kwarg_layouts}, "
        tmpstr += f"use_local_input={self.prepare_module_input.use_local_output}, "
        tmpstr += f"output_layouts={self.prepare_module_output.output_layouts}, "
        tmpstr += f"desired_output_layouts={self.prepare_module_output.desired_output_layouts}, "
        tmpstr += f"use_local_output={self.prepare_module_output.use_local_output}"
        tmpstr += ")"
        return tmpstr
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor


import copy

import torch
from torch._ops import OpOverload
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import (
    ArgsType,
    KwargsType,
    OpSchema,
    OpStrategy,
    PlacementList,
    RuntimeSchemaInfo,
)
from torch.distributed.tensor._ops.single_dim_strategy import (
    _ShardingPlaceholder,
    register_single_dim_strategy,
)
from torch.distributed.tensor._ops.utils import prod
from torch.distributed.tensor._utils import (
    compute_local_shape_and_global_offset,
    compute_local_stride,
)
from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.fx.experimental.symbolic_shapes import guard_or_false


aten = torch.ops.aten


@register_single_dim_strategy(aten.t.default, allow_unbacked_sharding=False)
def transpose_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    if ndim <= 1:
        for dim in range(ndim):
            strategies.append([_ShardingPlaceholder(dim), _ShardingPlaceholder(dim)])
    else:
        strategies.append([_ShardingPlaceholder(1), _ShardingPlaceholder(0)])
        strategies.append([_ShardingPlaceholder(0), _ShardingPlaceholder(1)])

    for reduce_op in ("sum", "avg", "max", "min"):
        strategies.append([Partial(reduce_op), Partial(reduce_op)])
    return strategies


def _scaled_mm_scale_placement(
    data_placement: Placement | _ShardingPlaceholder,
    scale_shape: torch.Size,
    contracting_dim: int,
) -> Placement | _ShardingPlaceholder | None:
    """
    Derive scale placement from data operand placement for _scaled_mm.

    Handles three cases:

    1. Tensor-wise scale (single element): always Replicate.
    2. 2D (or higher) scale, e.g. row-wise [M,1]: copy data placement directly.
    3. 1D blockwise scale, e.g. MX format [M*K/block_size]: map
       non-contracting shard to Shard(0)/_ShardingPlaceholder(0), and reject
       contracting-dim shards (returns None).
    """
    if prod(scale_shape) == 1:
        return Replicate()

    if len(scale_shape) != 1:
        return data_placement

    # 1D blockwise scale: Shard(>=1) is invalid on a 1D tensor, so we need
    # to map the data operand's placement to a valid 1D placement.
    if isinstance(data_placement, _ShardingPlaceholder):
        if data_placement.dim == contracting_dim:
            return None
        return _ShardingPlaceholder(0)
    # NOTE: isinstance(_, Shard) does not match _StridedShard; see _is_shard_like().
    elif isinstance(data_placement, Shard):
        if data_placement.dim == contracting_dim:
            return None
        return Shard(0)
    elif isinstance(data_placement, (Replicate, Partial)):
        return Replicate()
    return data_placement


from ._einsum_strategy import EinsumDims


def gen_single_dim_einsum_strategies(
    equation: str,
    *,
    bias_shape: torch.Size | None = None,
) -> list[list[Placement | _ShardingPlaceholder]]:
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

    3. Per-input linearity (Partial): Since matmul is linear in each input
       independently, one input can remain Partial while others are Replicate,
       producing a Partial output.

    4. Batch-dimension linearity (all-Partial): When all dims are batch dims
       (no contracting or free dims), the operation is element-wise and linear
       in all inputs simultaneously, so all inputs can be Partial.

    5. Bias input (optional): If bias_shape is provided, a bias placement
       is inserted after the output placement. The bias placement is derived from
       the output placement, accounting for broadcast semantics (based on ndim
       difference between output and bias). This is used for addmm-like ops
       (addmm, baddbmm) where bias + mat1 @ mat2.
    """
    # parse einop equation and extract dims
    input_dims, output_dim = EinsumDims.parse_equation(equation)
    edims = EinsumDims.parse_dims(input_dims, output_dim)

    # Compute broadcast dims map for bias if provided
    # Maps output dims to bias dims, -1 for broadcast dims (dims that don't exist in bias
    # or have size 1)
    broadcast_dims_map: list[int] | None = None
    if bias_shape is not None:
        output_ndim = len(output_dim)
        bias_ndim = len(bias_shape)
        pad_size = output_ndim - bias_ndim
        broadcast_dims_map = []
        for i in range(output_ndim):
            if i < pad_size:
                # Padded dimension (not in bias)
                broadcast_dims_map.append(-1)
            else:
                bias_dim_idx = i - pad_size
                if bias_shape[bias_dim_idx] == 1:
                    # Size-1 dimension (broadcasts)
                    broadcast_dims_map.append(-1)
                else:
                    broadcast_dims_map.append(bias_dim_idx)

    def _derive_bias_placement(
        output_placement: Placement | _ShardingPlaceholder,
    ) -> Placement | _ShardingPlaceholder:
        """Derive bias placement from output placement, accounting for broadcast."""
        if broadcast_dims_map is None:
            return copy.copy(output_placement)
        if isinstance(output_placement, _ShardingPlaceholder):
            output_dim_idx = output_placement.dim
            bias_dim = broadcast_dims_map[output_dim_idx]
            if bias_dim == -1:
                # Dim doesn't exist in bias (broadcast), replicate
                return Replicate()
            else:
                return _ShardingPlaceholder(bias_dim)
        else:
            # Clone Partial, Replicate, or other placements
            return copy.copy(output_placement)

    def _maybe_add_bias(
        placement_list: list[Placement | _ShardingPlaceholder],
    ) -> list[Placement | _ShardingPlaceholder]:
        """Insert bias placement after output if bias_shape is provided."""
        if bias_shape is None:
            return placement_list
        output_placement = placement_list[0]
        bias_placement = _derive_bias_placement(output_placement)
        return [placement_list[0], bias_placement] + placement_list[1:]

    # generate strategies for each mesh dim and do cartesian product for final strategy. E.g., for a 2D mesh, we can have [P(),R,R]
    strategies_over_one_mesh_dim: list[list[Placement | _ShardingPlaceholder]] = []
    placement_list: list[Placement | _ShardingPlaceholder]
    # split batch dim
    for batch_dim in edims.batch_dims:
        output_batch_dim = output_dim.index(batch_dim)
        placement_list = [_ShardingPlaceholder(output_batch_dim)]
        for input_dim in input_dims:
            input_batch_dim = input_dim.index(batch_dim)
            placement_list.append(_ShardingPlaceholder(input_batch_dim))

        strategies_over_one_mesh_dim.append(_maybe_add_bias(placement_list))

    # split contracting dim
    for contracting_dim in edims.contracting_dims:
        # Contracting dim can shard on same device axis for both inputs. This
        # results in the output being Partial on that device axis. For example:
        # bmk_{x},k_{x}n -> bmn{Ux} (becomes partial over device axis x)
        placement_list = [Partial()]
        for input_dim in input_dims:
            input_contracting_dim = input_dim.index(contracting_dim)
            placement_list.append(_ShardingPlaceholder(input_contracting_dim))

        strategies_over_one_mesh_dim.append(_maybe_add_bias(placement_list))

    # split lhs free dim
    for lhs_dim in edims.lhs_out_only_dims:
        lhs_free_dim_output = output_dim.index(lhs_dim)
        lhs_free_dim_input = input_dims[0].index(lhs_dim)
        # this means split the lhs input and output
        # i.e. S(0), R -> S(0)
        lhs_placement_list: list[Placement | _ShardingPlaceholder] = [
            _ShardingPlaceholder(lhs_free_dim_output),
            _ShardingPlaceholder(lhs_free_dim_input),
            Replicate(),
        ]
        strategies_over_one_mesh_dim.append(_maybe_add_bias(lhs_placement_list))

    # split rhs free dim
    for rhs_dim in edims.rhs_out_only_dims:
        rhs_free_dim_output = output_dim.index(rhs_dim)
        rhs_free_dim_input = input_dims[1].index(rhs_dim)
        rhs_placement_list: list[Placement | _ShardingPlaceholder] = [
            _ShardingPlaceholder(rhs_free_dim_output),
            Replicate(),
            _ShardingPlaceholder(rhs_free_dim_input),
        ]
        strategies_over_one_mesh_dim.append(_maybe_add_bias(rhs_placement_list))

    # Per-input linearity: matmul is linear in each input independently.
    # One input Partial, the other Replicate → output Partial.
    for reduce_op in Partial.LINEAR_REDUCE_OPS:
        output_placement = Partial(reduce_op)
        strategies_over_one_mesh_dim.append(
            _maybe_add_bias([output_placement, Partial(reduce_op), Replicate()])
        )
        strategies_over_one_mesh_dim.append(
            _maybe_add_bias([output_placement, Replicate(), Partial(reduce_op)])
        )

    # Batch-dimension linearity: when the einsum has no contracting dims and
    # no free dims (all dims are batch dims), the operation is element-wise
    # and linear in all inputs simultaneously. Add all-Partial strategies.
    if (
        not edims.contracting_dims
        and not edims.lhs_out_only_dims
        and not edims.rhs_out_only_dims
    ):
        for reduce_op in Partial.LINEAR_REDUCE_OPS:
            linearity_placements: list[Placement | _ShardingPlaceholder] = [
                Partial(reduce_op)
            ] + [Partial(reduce_op) for _ in input_dims]
            strategies_over_one_mesh_dim.append(_maybe_add_bias(linearity_placements))

    return strategies_over_one_mesh_dim


@register_single_dim_strategy(aten.dot.default, allow_unbacked_sharding=True)
def dot_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    return gen_single_dim_einsum_strategies("i,i->")


@register_single_dim_strategy(aten.mm.default, allow_unbacked_sharding=True)
def mm_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    return gen_single_dim_einsum_strategies("mk,kn->mn")


@register_single_dim_strategy(aten.addmm.default, allow_unbacked_sharding=True)
def addmm_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    bias_meta = args_schema[0]
    if not isinstance(bias_meta, TensorMeta):
        raise AssertionError
    return gen_single_dim_einsum_strategies("mk,kn->mn", bias_shape=bias_meta.shape)


@register_single_dim_strategy(aten.bmm.default, allow_unbacked_sharding=True)
def bmm_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    return gen_single_dim_einsum_strategies("bmk,bkn->bmn")


@register_single_dim_strategy(aten.baddbmm.default, allow_unbacked_sharding=True)
def baddbmm_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    bias_meta = args_schema[0]
    if not isinstance(bias_meta, TensorMeta):
        raise AssertionError
    return gen_single_dim_einsum_strategies("bmk,bkn->bmn", bias_shape=bias_meta.shape)


@register_single_dim_strategy(aten._scaled_mm.default, allow_unbacked_sharding=True)
def scaled_mm_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    scale_self_meta = args_schema[2]
    scale_mat2_meta = args_schema[3]
    if not isinstance(scale_self_meta, TensorMeta):
        raise AssertionError
    if not isinstance(scale_mat2_meta, TensorMeta):
        raise AssertionError
    if args_schema[4] is not None:
        raise AssertionError("_scaled_mm on DTensors doesn't support bias")
    if args_schema[5] is not None:
        raise AssertionError("_scaled_mm on DTensors doesn't support scale_result")

    # "mk,kn->mn": self_contracting_dim=1, mat2_contracting_dim=0
    base_strategies = gen_single_dim_einsum_strategies("mk,kn->mn")
    result = []
    for strat in base_strategies:
        # strat is [output, self, mat2]; derive scale placements
        scale_self_p = _scaled_mm_scale_placement(
            strat[1], scale_self_meta.shape, contracting_dim=1
        )
        scale_mat2_p = _scaled_mm_scale_placement(
            strat[2], scale_mat2_meta.shape, contracting_dim=0
        )
        if scale_self_p is None or scale_mat2_p is None:
            continue
        result.append(strat + [scale_self_p, scale_mat2_p])
    return result


def _scaled_dot_product_flash_attention_base_strategies(
    op_schema: OpSchema,
) -> list[PlacementList]:
    """Helper that returns list of base placement strategies (without CP)."""
    return_debug_mask = len(op_schema.args_schema) >= 6 and op_schema.args_schema[5]
    q_input_strategy = op_schema.args_schema[0]
    if not isinstance(q_input_strategy, OpStrategy):
        raise AssertionError(f"Expected OpStrategy, got {type(q_input_strategy)}")
    # assuming q/k/v have the same shape

    single_mesh_dim_strategies = []

    # placement list stores placements of [outputs, inputs]
    # in the sdpa case, we have 3 valid tensor outputs and 3 tensor inputs
    # first we can always accept full replication for both inputs and outputs
    all_replicate: PlacementList = [
        Replicate(),
        Replicate(),
        None,  # cum_seq_q
        None,  # cum_seq_k
        None,  # max_q
        None,  # max_k
        Replicate(),  # rng_state
        None,  # unused
        Replicate(),
        Replicate(),
        Replicate(),
        Replicate(),
    ]
    single_mesh_dim_strategies.append(all_replicate)

    # second we can accept the sharding pattern of tensor parallelism, which
    # shard on the num of head dim
    qkv_sharding = Shard(1)  # num head dim
    output_sharding = Shard(1)  # num head dim
    logsumexp_sharding = Shard(1)  # num head dim
    if return_debug_mask:
        debug_attn_mask_sharding: Placement = Shard(1)  # num head dim
    else:
        # empty debug mask, replicated
        debug_attn_mask_sharding = Replicate()

    num_heads_dim_sharding: PlacementList = [
        output_sharding,
        logsumexp_sharding,
        None,  # cum_seq_q
        None,  # cum_seq_k
        None,  # max_q
        None,  # max_k
        Replicate(),  # rng_state
        None,  # unused
        debug_attn_mask_sharding,
        qkv_sharding,
        qkv_sharding,
        qkv_sharding,
    ]
    single_mesh_dim_strategies.append(num_heads_dim_sharding)

    # Shard on the batch dimension
    debug_attn_mask_sharding = Shard(0) if return_debug_mask else Replicate()
    single_mesh_dim_strategies.append(
        [
            Shard(0),  # output
            Shard(0),  # logsumexp
            None,  # cum_seq_q
            None,  # cum_seq_k
            None,  # max_q
            None,  # max_k
            Replicate(),  # rng_state
            None,  # unused
            debug_attn_mask_sharding,  # debugattn
            Shard(0),  # q
            Shard(0),  # k
            Shard(0),  # v
        ]
    )
    return single_mesh_dim_strategies


@register_single_dim_strategy(
    aten._scaled_dot_product_flash_attention.default, schema_info=RuntimeSchemaInfo(5)
)
def scaled_dot_product_flash_attention_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder | None]]:
    q_meta = args_schema[0]
    if not isinstance(q_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(q_meta)}")

    return_debug_mask = len(args_schema) >= 6 and args_schema[5]
    debug_attn_mask_head: Placement | _ShardingPlaceholder = (
        _ShardingPlaceholder(1) if return_debug_mask else Replicate()
    )
    debug_attn_mask_batch: Placement | _ShardingPlaceholder = (
        _ShardingPlaceholder(0) if return_debug_mask else Replicate()
    )

    return [
        [
            _ShardingPlaceholder(1),  # output
            _ShardingPlaceholder(1),  # logsumexp
            None,  # cum_seq_q
            None,  # cum_seq_k
            None,  # max_q
            None,  # max_k
            Replicate(),  # rng_state
            None,  # unused
            debug_attn_mask_head,
            _ShardingPlaceholder(1),  # q
            _ShardingPlaceholder(1),  # k
            _ShardingPlaceholder(1),  # v
        ],
        [
            _ShardingPlaceholder(0),  # output
            _ShardingPlaceholder(0),  # logsumexp
            None,  # cum_seq_q
            None,  # cum_seq_k
            None,  # max_q
            None,  # max_k
            Replicate(),  # rng_state
            None,  # unused
            debug_attn_mask_batch,
            _ShardingPlaceholder(0),  # q
            _ShardingPlaceholder(0),  # k
            _ShardingPlaceholder(0),  # v
        ],
    ]


def _scaled_dot_product_flash_attention_backward_base_strategies(
    op_schema: OpSchema,
) -> list[PlacementList]:
    """Helper that returns list of base placement strategies (without CP)."""
    q_input_strategy = op_schema.args_schema[1]
    if not isinstance(q_input_strategy, OpStrategy):
        raise AssertionError(f"Expected OpStrategy, got {type(q_input_strategy)}")
    # assuming q/k/v have the same shape

    tensor_input_indices = [
        i
        for i, arg_spec in enumerate(op_schema.args_schema)
        if isinstance(arg_spec, OpStrategy)
    ]
    num_tensor_inputs = len(tensor_input_indices)

    single_mesh_dim_strategies = []

    # placement list stores placements of [outputs, inputs]
    # in the sdpa backward case, we have 3 tensor outputs and 6 to 10 tensor inputs
    # first we can always accept full replication for both inputs and outputs
    all_replicate: PlacementList = [Replicate()] * (3 + num_tensor_inputs)

    single_mesh_dim_strategies.append(all_replicate)

    # second we can accept the sharding pattern of tensor parallelism, which
    # shard on the num of head dim
    grad_output_sharding = Shard(1)  # num head dim
    qkv_sharding = Shard(1)  # num head dim
    output_sharding = Shard(1)  # num head dim
    logsumexp_sharding = Shard(1)  # num head dim
    grad_qkv_sharding = Shard(1)  # num head dim

    num_heads_dim_sharding: PlacementList = [
        grad_qkv_sharding,
        grad_qkv_sharding,
        grad_qkv_sharding,
        grad_output_sharding,
        qkv_sharding,
        qkv_sharding,
        qkv_sharding,
        output_sharding,
        logsumexp_sharding,
    ]
    # accept replicate on the rest tensor inputs, potentially
    # cum_seq_q, cum_seq_k, philox_seed, philox_offset
    # at indices 6, 7, 12, 13, respectively
    num_heads_dim_sharding.extend([Replicate()] * (num_tensor_inputs - 6))
    single_mesh_dim_strategies.append(num_heads_dim_sharding)

    # Batch sharding
    batch_dim_sharding: PlacementList = [
        Shard(0),  # grad_q
        Shard(0),  # grad_k
        Shard(0),  # grad_v
        Shard(0),  # grad_output
        Shard(0),  # q
        Shard(0),  # k
        Shard(0),  # v
        Shard(0),  # output
        Shard(0),  # logsumexp
    ]
    # accept replicate on the rest tensor inputs, potentially
    # cum_seq_q, cum_seq_k, philox_seed, philox_offset
    # at indices 6, 7, 12, 13, respectively
    batch_dim_sharding.extend([Replicate()] * (num_tensor_inputs - 6))
    single_mesh_dim_strategies.append(batch_dim_sharding)

    return single_mesh_dim_strategies


@register_single_dim_strategy(aten._scaled_dot_product_flash_attention_backward.default)
def scaled_dot_product_flash_attention_backward_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    num_tensor_inputs = sum(isinstance(arg, TensorMeta) for arg in args_schema)
    if num_tensor_inputs < 6:
        raise AssertionError(
            f"Expected at least 6 tensor inputs, got {num_tensor_inputs}"
        )

    num_heads_dim_sharding: list[Placement | _ShardingPlaceholder] = [
        _ShardingPlaceholder(1),  # grad_q
        _ShardingPlaceholder(1),  # grad_k
        _ShardingPlaceholder(1),  # grad_v
        _ShardingPlaceholder(1),  # grad_out
        _ShardingPlaceholder(1),  # q
        _ShardingPlaceholder(1),  # k
        _ShardingPlaceholder(1),  # v
        _ShardingPlaceholder(1),  # output
        _ShardingPlaceholder(1),  # logsumexp
    ]
    num_heads_dim_sharding.extend([Replicate()] * (num_tensor_inputs - 6))

    batch_dim_sharding: list[Placement | _ShardingPlaceholder] = [
        _ShardingPlaceholder(0),  # grad_q
        _ShardingPlaceholder(0),  # grad_k
        _ShardingPlaceholder(0),  # grad_v
        _ShardingPlaceholder(0),  # grad_out
        _ShardingPlaceholder(0),  # q
        _ShardingPlaceholder(0),  # k
        _ShardingPlaceholder(0),  # v
        _ShardingPlaceholder(0),  # output
        _ShardingPlaceholder(0),  # logsumexp
    ]
    batch_dim_sharding.extend([Replicate()] * (num_tensor_inputs - 6))

    return [num_heads_dim_sharding, batch_dim_sharding]


@register_single_dim_strategy(
    aten.constant_pad_nd.default, schema_info=RuntimeSchemaInfo(1)
)
def constant_pad_nd_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    # Allow sharding on non-padded dimensions; ban sharding on dims
    # that have non-zero padding (where the pad value must be inserted).
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)
    pad = args_schema[1]
    if not isinstance(pad, (list, tuple)):
        raise AssertionError(f"Expected list or tuple, got {type(pad)}")

    # pad is [dim_{n-1}_left, dim_{n-1}_right, dim_{n-2}_left, ...] from
    # the last dim backwards. Determine which dims have non-zero padding.
    padded_dims = set()
    for i in range(len(pad) // 2):
        if not (
            guard_or_false(pad[i * 2] == 0) and guard_or_false(pad[i * 2 + 1] == 0)
        ):
            padded_dims.add(ndim - 1 - i)

    # Shard on any non-padded dim: output and input share the same placement.
    # All-Replicate is added automatically by the framework.
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in range(ndim):
        if dim not in padded_dims:
            strategies.append([_ShardingPlaceholder(dim), _ShardingPlaceholder(dim)])

    # Partial rules: at padded positions every rank writes the same constant v,
    # so reduce(v, v, ..., v) = v for avg/max/min (idempotent). P(sum) only
    # works when v=0 since sum(v, ..., v) = N*v != v otherwise.
    # When all pad amounts are zero the op is a no-op, so all reduce ops hold.
    value = args_schema[2] if len(args_schema) > 2 else 0
    no_padding = all(guard_or_false(pad[i] == 0) for i in range(len(pad)))
    if no_padding or guard_or_false(value == 0):
        reduce_ops = ("sum", "avg", "max", "min")
    else:
        reduce_ops = ("avg", "max", "min")
    for reduce_op in reduce_ops:
        strategies.append([Partial(reduce_op), Partial(reduce_op)])

    return strategies


def _scaled_dot_product_efficient_attention_base_strategies(
    op_schema: OpSchema,
) -> list[PlacementList]:
    """Helper that returns list of base placement strategies (without CP)."""
    q_input_strategy = op_schema.args_schema[0]
    if not isinstance(q_input_strategy, OpStrategy):
        raise AssertionError(f"Expected OpStrategy, got {type(q_input_strategy)}")
    # assuming q/k/v have the same shape

    has_attn_bias = op_schema.args_schema[3] is not None
    compute_log_sumexp = op_schema.args_schema[4]

    single_mesh_dim_strategies: list[PlacementList] = []

    # placement list stores placements of [outputs, inputs]
    # in the sdpa case, we have 2 valid tensor outputs and 3 or 4 tensor inputs
    # first we can always accept full replication for both inputs and outputs
    all_replicate: PlacementList = [
        Replicate(),
        Replicate(),
        None,
        None,
        Replicate(),
        Replicate(),
        Replicate(),
    ]
    if has_attn_bias:
        all_replicate.append(Replicate())  # attn bias

    single_mesh_dim_strategies.append(all_replicate)

    # second we can accept the sharding pattern of tensor parallelism, which
    # shard on the heads dimension
    qkv_sharding = Shard(1)
    output_sharding = Shard(1)
    if compute_log_sumexp:
        logsumexp_sharding: Placement = Shard(1)
    else:
        # empty logsumexp, replicated
        logsumexp_sharding = Replicate()

    num_heads_dim_sharding = [
        output_sharding,
        logsumexp_sharding,
        None,
        None,
        qkv_sharding,
        qkv_sharding,
        qkv_sharding,
    ]
    if has_attn_bias:
        num_heads_dim_sharding.append(Shard(1))
    single_mesh_dim_strategies.append(num_heads_dim_sharding)

    # batch sharding
    if compute_log_sumexp:
        logsumexp_sharding_dp: Placement = Shard(0)
    else:
        # empty logsumexp, replicated
        logsumexp_sharding_dp = Replicate()
    batch_sharding = [
        Shard(0),  # output
        logsumexp_sharding_dp,  # logsumexp
        None,  # philox_seed
        None,  # philox_offset
        Shard(0),  # q
        Shard(0),  # k
        Shard(0),  # v
    ]
    if has_attn_bias:
        batch_sharding.append(Shard(0))

    single_mesh_dim_strategies.append(batch_sharding)

    return single_mesh_dim_strategies


@register_single_dim_strategy(
    aten._scaled_dot_product_efficient_attention.default,
    schema_info=RuntimeSchemaInfo(4),
)
def scaled_dot_product_efficient_attention_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder | None]]:
    q_meta = args_schema[0]
    if not isinstance(q_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(q_meta)}")

    has_attn_bias = args_schema[3] is not None
    compute_log_sumexp = args_schema[4]

    logsumexp_head: Placement | _ShardingPlaceholder = (
        _ShardingPlaceholder(1) if compute_log_sumexp else Replicate()
    )
    logsumexp_batch: Placement | _ShardingPlaceholder = (
        _ShardingPlaceholder(0) if compute_log_sumexp else Replicate()
    )

    num_heads_dim_sharding: list[Placement | _ShardingPlaceholder | None] = [
        _ShardingPlaceholder(1),  # output
        logsumexp_head,
        None,  # philox_seed
        None,  # philox_offset
        _ShardingPlaceholder(1),  # q
        _ShardingPlaceholder(1),  # k
        _ShardingPlaceholder(1),  # v
    ]
    if has_attn_bias:
        num_heads_dim_sharding.append(_ShardingPlaceholder(1))

    batch_dim_sharding: list[Placement | _ShardingPlaceholder | None] = [
        _ShardingPlaceholder(0),  # output
        logsumexp_batch,
        None,  # philox_seed
        None,  # philox_offset
        _ShardingPlaceholder(0),  # q
        _ShardingPlaceholder(0),  # k
        _ShardingPlaceholder(0),  # v
    ]
    if has_attn_bias:
        batch_dim_sharding.append(_ShardingPlaceholder(0))

    return [num_heads_dim_sharding, batch_dim_sharding]


def _scaled_dot_product_efficient_attention_backward_base_strategies(
    op_schema: OpSchema,
) -> list[PlacementList]:
    """Helper that returns list of base placement strategies (without CP)."""
    q_input_strategy = op_schema.args_schema[1]
    if not isinstance(q_input_strategy, OpStrategy):
        raise AssertionError(f"Expected OpStrategy, got {type(q_input_strategy)}")
    # assuming q/k/v have the same shape
    has_attn_bias = op_schema.args_schema[4] is not None

    single_mesh_dim_strategies = []

    # placement list stores placements of [outputs, inputs]
    # in the sdpa backward case, we have 4 tensor outputs and 8 or 9 tensor inputs
    # NOTE: Output sharding of grad_bias on heads dim if attn_bias is present;
    #       otherwise grad_bias will be empty and its DTensorSpec will be removed.
    # first we can always accept full replication for both inputs and outputs
    all_replicate: PlacementList = [Replicate()] * (12 + has_attn_bias)

    if not has_attn_bias:
        all_replicate[3] = None  # grad bias is None if attn_bias is not present

    single_mesh_dim_strategies.append(all_replicate)

    # second we can accept the sharding pattern of tensor parallelism, which
    # shard on the heads dimension
    grad_output_sharding = Shard(1)
    qkv_sharding = Shard(1)
    output_sharding = Shard(1)
    logsumexp_sharding = Shard(1)
    grad_qkv_sharding = Shard(1)
    grad_bias_sharding = Shard(1) if has_attn_bias else None

    num_heads_dim_sharding: PlacementList = [
        grad_qkv_sharding,
        grad_qkv_sharding,
        grad_qkv_sharding,
        grad_bias_sharding,
        grad_output_sharding,
        qkv_sharding,
        qkv_sharding,
        qkv_sharding,
        # the place for optional input attn_bias,
        output_sharding,
        logsumexp_sharding,
    ]
    # input sharding of attn_bias on heads dim if present
    if has_attn_bias:
        num_heads_dim_sharding.insert(8, Shard(1))
    # accept replicate on the rest scalar tensor inputs
    # namely philox_seed and philox_offset
    num_heads_dim_sharding.extend([Replicate(), Replicate()])
    single_mesh_dim_strategies.append(num_heads_dim_sharding)

    # Shards on batch dim
    batch_dim_sharding: PlacementList = [
        Shard(0),  # grad_q
        Shard(0),  # grad_k
        Shard(0),  # grad_v
        Shard(0) if has_attn_bias else None,  # grad_bias
        Shard(0),  # grad_output
        Shard(0),  # q
        Shard(0),  # k
        Shard(0),  # v
        Shard(0),  # output
        Shard(0),  # logsumexp
    ]
    # accept replicate on the rest tensor inputs, potentially
    # cum_seq_q, cum_seq_k, philox_seed, philox_offset
    # at indices 6, 7, 12, 13, respectively
    if has_attn_bias:
        batch_dim_sharding.insert(8, Shard(0))
    batch_dim_sharding.extend([Replicate(), Replicate()])
    single_mesh_dim_strategies.append(batch_dim_sharding)

    return single_mesh_dim_strategies


@register_single_dim_strategy(
    aten._scaled_dot_product_efficient_attention_backward.default
)
def scaled_dot_product_efficient_attention_backward_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder | None]]:
    has_attn_bias = args_schema[4] is not None

    num_heads_dim_sharding: list[Placement | _ShardingPlaceholder | None] = [
        _ShardingPlaceholder(1),  # grad_q
        _ShardingPlaceholder(1),  # grad_k
        _ShardingPlaceholder(1),  # grad_v
        _ShardingPlaceholder(1) if has_attn_bias else None,  # grad_bias
        _ShardingPlaceholder(1),  # grad_out
        _ShardingPlaceholder(1),  # q
        _ShardingPlaceholder(1),  # k
        _ShardingPlaceholder(1),  # v
    ]
    if has_attn_bias:
        num_heads_dim_sharding.append(_ShardingPlaceholder(1))
    num_heads_dim_sharding.extend(
        [
            _ShardingPlaceholder(1),  # output
            _ShardingPlaceholder(1),  # logsumexp
            Replicate(),  # philox_seed
            Replicate(),  # philox_offset
        ]
    )

    batch_dim_sharding: list[Placement | _ShardingPlaceholder | None] = [
        _ShardingPlaceholder(0),  # grad_q
        _ShardingPlaceholder(0),  # grad_k
        _ShardingPlaceholder(0),  # grad_v
        _ShardingPlaceholder(0) if has_attn_bias else None,  # grad_bias
        _ShardingPlaceholder(0),  # grad_out
        _ShardingPlaceholder(0),  # q
        _ShardingPlaceholder(0),  # k
        _ShardingPlaceholder(0),  # v
    ]
    if has_attn_bias:
        batch_dim_sharding.append(_ShardingPlaceholder(0))
    batch_dim_sharding.extend(
        [
            _ShardingPlaceholder(0),  # output
            _ShardingPlaceholder(0),  # logsumexp
            Replicate(),  # philox_seed
            Replicate(),  # philox_offset
        ]
    )

    return [num_heads_dim_sharding, batch_dim_sharding]


def _scaled_dot_product_cudnn_attention_base_strategies(
    op_schema: OpSchema,
) -> list[PlacementList]:
    """Helper that returns list of base placement strategies (without CP)."""
    (
        query_strategy,  # query
        _,  # key
        _,  # value
        attn_bias_strategy,
        compute_log_sumexp,  # compute_log_sumexp
        *rest_args,  # optional args: dropout_p, is_causal, return_debug_mask, scale
    ) = op_schema.args_schema
    return_debug_mask = len(op_schema.args_schema) >= 8 and rest_args[2]
    has_attn_bias = attn_bias_strategy is not None
    debug_attn_mask_sharding: Placement | None = (
        Replicate() if return_debug_mask else None
    )

    if not isinstance(query_strategy, OpStrategy):
        raise AssertionError(f"Expected OpStrategy, got {type(query_strategy)}")
    # assuming q/k/v have the same shape

    single_mesh_dim_strategies = []

    # placement list stores placements of [outputs, inputs]
    # in the sdpa case, we have 2 valid tensor outputs and 3 tensor inputs
    # first we can always accept full replication for both inputs and outputs
    all_replicate: PlacementList = [
        Replicate(),  # output
        Replicate(),  # logsumexp
        None,  # cum_seq_q
        None,  # cum_seq_k
        None,  # max_q
        None,  # max_k
        None,  # philox_seed
        None,  # philox_offset
        # NOTE: debug_attn_mask is not supported by pytorch and is always an empty tensor
        # https://github.com/pytorch/pytorch/blob/60205b0eb2602317856312a66d955c88334ade0b/aten/src/ATen/native/transformers/cuda/attention.cu#L839-L840
        debug_attn_mask_sharding,  # debug_attn_mask
        Replicate(),  # q
        Replicate(),  # k
        Replicate(),  # v
    ]
    if has_attn_bias:
        all_replicate.append(Replicate())  # attn bias

    single_mesh_dim_strategies.append(all_replicate)

    # second we can accept the sharding pattern of tensor parallelism, which
    # shard on the num of head dim
    tp_sharding = Shard(1)  # num head dim
    qkv_sharding = tp_sharding
    output_sharding = tp_sharding
    logsumexp_sharding = tp_sharding if compute_log_sumexp else Replicate()
    debug_attn_mask_sharding = tp_sharding if return_debug_mask else None

    num_heads_dim_sharding: PlacementList = [
        output_sharding,
        logsumexp_sharding,
        None,  # cum_seq_q
        None,  # cum_seq_k
        None,  # max_q
        None,  # max_k
        None,  # philox_seed
        None,  # philox_offset
        debug_attn_mask_sharding,
        qkv_sharding,
        qkv_sharding,
        qkv_sharding,
    ]
    single_mesh_dim_strategies.append(num_heads_dim_sharding)

    # batch parallelism
    logsumexp_sharding = Shard(0) if compute_log_sumexp else Replicate()
    debug_attn_mask_sharding = Shard(0) if return_debug_mask else None
    batch_dim_sharding: PlacementList = [
        Shard(0),  # output
        logsumexp_sharding,
        None,  # cum_seq_q
        None,  # cum_seq_k
        None,  # max_q
        None,  # max_k
        None,  # philox_seed
        None,  # philox_offset
        debug_attn_mask_sharding,
        Shard(0),  # q
        Shard(0),  # k
        Shard(0),  # v
    ]
    single_mesh_dim_strategies.append(batch_dim_sharding)

    return single_mesh_dim_strategies


@register_single_dim_strategy(
    aten._scaled_dot_product_cudnn_attention.default,
    schema_info=RuntimeSchemaInfo(4),
)
def scaled_dot_product_cudnn_attention_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder | None]]:
    query_meta = args_schema[0]
    if not isinstance(query_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(query_meta)}")

    has_attn_bias = args_schema[3] is not None
    compute_log_sumexp = args_schema[4]
    return_debug_mask = len(args_schema) >= 8 and args_schema[7]

    logsumexp_head: Placement | _ShardingPlaceholder = (
        _ShardingPlaceholder(1) if compute_log_sumexp else Replicate()
    )
    logsumexp_batch: Placement | _ShardingPlaceholder = (
        _ShardingPlaceholder(0) if compute_log_sumexp else Replicate()
    )
    debug_attn_mask_head: Placement | _ShardingPlaceholder | None = (
        _ShardingPlaceholder(1) if return_debug_mask else None
    )
    debug_attn_mask_batch: Placement | _ShardingPlaceholder | None = (
        _ShardingPlaceholder(0) if return_debug_mask else None
    )

    num_heads_dim_sharding: list[Placement | _ShardingPlaceholder | None] = [
        _ShardingPlaceholder(1),  # output
        logsumexp_head,
        None,  # cum_seq_q
        None,  # cum_seq_k
        None,  # max_q
        None,  # max_k
        None,  # philox_seed
        None,  # philox_offset
        debug_attn_mask_head,
        _ShardingPlaceholder(1),  # q
        _ShardingPlaceholder(1),  # k
        _ShardingPlaceholder(1),  # v
    ]
    if has_attn_bias:
        num_heads_dim_sharding.append(_ShardingPlaceholder(1))

    batch_dim_sharding: list[Placement | _ShardingPlaceholder | None] = [
        _ShardingPlaceholder(0),  # output
        logsumexp_batch,
        None,  # cum_seq_q
        None,  # cum_seq_k
        None,  # max_q
        None,  # max_k
        None,  # philox_seed
        None,  # philox_offset
        debug_attn_mask_batch,
        _ShardingPlaceholder(0),  # q
        _ShardingPlaceholder(0),  # k
        _ShardingPlaceholder(0),  # v
    ]
    if has_attn_bias:
        batch_dim_sharding.append(_ShardingPlaceholder(0))

    return [num_heads_dim_sharding, batch_dim_sharding]


def _scaled_dot_product_cudnn_attention_backward_base_strategies(
    op_schema: OpSchema,
) -> list[PlacementList]:
    """Helper that returns list of base placement strategies (without CP)."""
    if len(op_schema.args_schema) < 15:
        raise AssertionError(
            f"Expected at least 15 args_schema, got {len(op_schema.args_schema)}"
        )
    has_attn_bias = op_schema.args_schema[8] is not None
    has_scale = len(op_schema.args_schema) >= 16 and False

    query_strategy = op_schema.args_schema[1]
    if not isinstance(query_strategy, OpStrategy):
        raise AssertionError(f"Expected OpStrategy, got {type(query_strategy)}")
    # assuming q/k/v have the same shape

    single_mesh_dim_strategies = []

    # placement list stores placements of [outputs, inputs]
    # cudnn outputs: (Tensor dq, Tensor dk, Tensor dv)
    # cudnn inputs: (
    #   Tensor grad_out,
    #   Tensor query,
    #   Tensor key,
    #   Tensor value,
    #   Tensor out,
    #   Tensor logsumexp,
    #   Tensor philox_seed,
    #   Tensor philox_offset,
    #   Tensor attn_bias,
    #   Tensor cum_seq_q,
    #   Tensor cum_seq_k,
    #   SymInt max_q,
    #   SymInt max_k,
    #   float dropout_p,
    #   bool is_causal,
    #   int? scale,
    # )

    # case 1: we can always accept full replication for both inputs and outputs
    all_replicate_out: PlacementList = [
        Replicate(),  # dq
        Replicate(),  # dk
        Replicate(),  # dv
    ]
    all_replicate_inp: PlacementList = [Replicate()] * 6
    all_replicate_inp += [
        Replicate()
    ] * 2  # philox_seed, philox_offset is casted to Replicate() in DTensor
    all_replicate_inp += [Replicate() if has_attn_bias else None]
    all_replicate_inp += [None] * 6
    if has_scale:
        all_replicate_inp.append(None)

    all_replicate: PlacementList = all_replicate_out + all_replicate_inp
    single_mesh_dim_strategies.append(all_replicate)

    # case 2: we can accept the sharding pattern of tensor parallelism, which
    #   shards on the num of head dim
    qkv_sharding = Shard(1)  # num head dim
    output_sharding = Shard(1)  # num head dim
    logsumexp_sharding = Shard(1)  # num head dim

    num_heads_dim_sharding_out: PlacementList = [qkv_sharding] * 3
    num_heads_dim_sharding_inp: PlacementList = [qkv_sharding] * 4
    num_heads_dim_sharding_inp += [output_sharding]
    num_heads_dim_sharding_inp += [logsumexp_sharding]
    num_heads_dim_sharding_inp += [
        Replicate()
    ] * 2  # philox_seed, philox_offset is casted to Replicate() in DTensor
    num_heads_dim_sharding_inp += [Shard(1) if has_attn_bias else None]
    num_heads_dim_sharding_inp += [None] * 6
    if has_scale:
        num_heads_dim_sharding_inp.append(None)

    num_heads_dim_sharding = num_heads_dim_sharding_out + num_heads_dim_sharding_inp
    single_mesh_dim_strategies.append(num_heads_dim_sharding)

    # case 3: we can accept the sharding pattern of batch parallelism, which
    #   shards on the batch dimension
    qkv_sharding = Shard(0)
    output_sharding = Shard(0)
    logsumexp_sharding = Shard(0)

    batch_dim_sharding_out: PlacementList = [qkv_sharding] * 3
    batch_dim_sharding_inp: PlacementList = [qkv_sharding] * 4
    batch_dim_sharding_inp += [output_sharding]
    batch_dim_sharding_inp += [logsumexp_sharding]
    batch_dim_sharding_inp += [
        Replicate()
    ] * 2  # philox_seed, philox_offset is casted to Replicate() in DTensor
    batch_dim_sharding_inp += [Shard(0) if has_attn_bias else None]
    batch_dim_sharding_inp += [None] * 6
    if has_scale:
        batch_dim_sharding_inp.append(None)

    batch_dim_sharding = batch_dim_sharding_out + batch_dim_sharding_inp
    single_mesh_dim_strategies.append(batch_dim_sharding)

    return single_mesh_dim_strategies


@register_single_dim_strategy(aten._scaled_dot_product_cudnn_attention_backward.default)
def scaled_dot_product_cudnn_attention_backward_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder | None]]:
    if len(args_schema) < 15:
        raise AssertionError(f"Expected at least 15 args, got {len(args_schema)}")

    for arg_index in range(6):
        arg = args_schema[arg_index]
        if not isinstance(arg, TensorMeta):
            raise AssertionError(f"Expected TensorMeta, got {type(arg)}")

    philox_placements: list[Placement] = []
    for arg_index in (6, 7):
        arg = args_schema[arg_index]
        if isinstance(arg, TensorMeta):
            philox_placements.append(Replicate())
        elif not isinstance(arg, torch.Tensor):
            raise AssertionError(f"Expected TensorMeta or Tensor, got {type(arg)}")

    has_attn_bias = args_schema[8] is not None
    if has_attn_bias and not isinstance(args_schema[8], (TensorMeta, torch.Tensor)):
        raise AssertionError(
            f"Expected TensorMeta or Tensor, got {type(args_schema[8])}"
        )

    cum_seq_placements: list[None] = []
    for arg_index in (9, 10):
        arg = args_schema[arg_index]
        if isinstance(arg, TensorMeta):
            cum_seq_placements.append(None)
        elif arg is None or isinstance(arg, torch.Tensor):
            pass
        else:
            raise AssertionError(f"Expected TensorMeta or Tensor, got {type(arg)}")

    num_heads_dim_sharding: list[Placement | _ShardingPlaceholder | None] = [
        _ShardingPlaceholder(1),  # grad_q
        _ShardingPlaceholder(1),  # grad_k
        _ShardingPlaceholder(1),  # grad_v
        _ShardingPlaceholder(1),  # grad_out
        _ShardingPlaceholder(1),  # q
        _ShardingPlaceholder(1),  # k
        _ShardingPlaceholder(1),  # v
        _ShardingPlaceholder(1),  # output
        _ShardingPlaceholder(1),  # logsumexp
    ]
    num_heads_dim_sharding.extend(philox_placements)
    if has_attn_bias and isinstance(args_schema[8], TensorMeta):
        num_heads_dim_sharding.append(_ShardingPlaceholder(1))
    num_heads_dim_sharding.extend(cum_seq_placements)

    batch_dim_sharding: list[Placement | _ShardingPlaceholder | None] = [
        _ShardingPlaceholder(0),  # grad_q
        _ShardingPlaceholder(0),  # grad_k
        _ShardingPlaceholder(0),  # grad_v
        _ShardingPlaceholder(0),  # grad_out
        _ShardingPlaceholder(0),  # q
        _ShardingPlaceholder(0),  # k
        _ShardingPlaceholder(0),  # v
        _ShardingPlaceholder(0),  # output
        _ShardingPlaceholder(0),  # logsumexp
    ]
    batch_dim_sharding.extend(philox_placements)
    if has_attn_bias and isinstance(args_schema[8], TensorMeta):
        batch_dim_sharding.append(_ShardingPlaceholder(0))
    batch_dim_sharding.extend(cum_seq_placements)

    return [num_heads_dim_sharding, batch_dim_sharding]


def _valid_grouped_mm_strides(
    mesh: DeviceMesh,
    op_schema: OpSchema,
    input_specs: list[DTensorSpec],
    output_specs: DTensorSpec | tuple[DTensorSpec | None, ...],
) -> bool:
    def local_meta(spec: DTensorSpec) -> TensorMeta:
        if not isinstance(spec.tensor_meta, TensorMeta):
            raise AssertionError(f"Expected TensorMeta, got {type(spec.tensor_meta)}")
        meta: TensorMeta = spec.tensor_meta
        local_shape, _ = compute_local_shape_and_global_offset(
            meta.shape, mesh, spec.placements, skip_offset=True
        )
        local_stride = compute_local_stride(meta.stride, local_shape)
        return TensorMeta(torch.Size(local_shape), local_stride, meta.dtype)

    def check_valid_strides(meta: TensorMeta) -> bool:
        # copied from `_meta_grouped_mm_common` in meta_registrations.py
        end_dim = len(meta.shape) - 1
        alignment = 16 // meta.dtype.itemsize
        if meta.stride[end_dim - 1] == 1 and meta.stride[end_dim] >= max(
            1, meta.shape[end_dim - 1]
        ):
            return meta.stride[end_dim] % alignment == 0
        elif meta.stride[end_dim] == 1 and meta.stride[end_dim - 1] >= max(
            1, meta.shape[end_dim]
        ):
            return meta.stride[end_dim - 1] % alignment == 0
        else:
            return False

    mat1_meta = local_meta(input_specs[0])
    mat2_meta = local_meta(input_specs[1])
    return check_valid_strides(mat1_meta) and check_valid_strides(mat2_meta)


@register_single_dim_strategy(
    aten._grouped_mm.default,
    full_mesh_strategy_filter=_valid_grouped_mm_strides,
)
def grouped_mm_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder | None]]:
    mat1_meta = args_schema[0]
    if not isinstance(mat1_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(mat1_meta)}")
    mat2_meta = args_schema[1]
    if not isinstance(mat2_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(mat2_meta)}")

    if len(args_schema) > 3 and args_schema[3] is not None:
        raise AssertionError("grouped_mm doesn't support bias yet")

    tensor_inputs_tail: list[Placement | None] = []
    if len(args_schema) > 2 and args_schema[2] is not None:
        if not isinstance(args_schema[2], TensorMeta):
            raise AssertionError(f"Expected TensorMeta, got {type(args_schema[2])}")
        tensor_inputs_tail.append(Replicate())

    strategies: list[list[Placement | _ShardingPlaceholder | None]] = [
        [Partial(), Partial(), Replicate(), *tensor_inputs_tail],
        [Partial(), Replicate(), Partial(), *tensor_inputs_tail],
    ]

    mat1_ndim = len(mat1_meta.shape)
    mat2_ndim = len(mat2_meta.shape)

    if mat1_ndim == 2 and mat2_ndim == 3:
        strategies.extend(
            [
                [
                    _ShardingPlaceholder(1),
                    Replicate(),
                    _ShardingPlaceholder(2),
                    *tensor_inputs_tail,
                ],
                [
                    Partial(),
                    _ShardingPlaceholder(1),
                    _ShardingPlaceholder(1),
                    *tensor_inputs_tail,
                ],
            ]
        )

    if mat1_ndim == 3 and mat2_ndim == 2:
        strategies.extend(
            [
                [
                    Partial(),
                    _ShardingPlaceholder(2),
                    _ShardingPlaceholder(0),
                    *tensor_inputs_tail,
                ],
                [
                    _ShardingPlaceholder(0),
                    _ShardingPlaceholder(1),
                    Replicate(),
                    *tensor_inputs_tail,
                ],
            ]
        )

    if mat1_ndim == 2 and mat2_ndim == 2:
        strategies.extend(
            [
                [
                    _ShardingPlaceholder(2),
                    Replicate(),
                    _ShardingPlaceholder(1),
                    *tensor_inputs_tail,
                ],
                [
                    _ShardingPlaceholder(1),
                    _ShardingPlaceholder(0),
                    Replicate(),
                    *tensor_inputs_tail,
                ],
            ]
        )

    if mat1_ndim == 3 and mat2_ndim == 3:
        strategies.extend(
            [
                [
                    _ShardingPlaceholder(2),
                    Replicate(),
                    _ShardingPlaceholder(2),
                    *tensor_inputs_tail,
                ],
                [
                    _ShardingPlaceholder(1),
                    _ShardingPlaceholder(1),
                    Replicate(),
                    *tensor_inputs_tail,
                ],
                [
                    Partial(),
                    _ShardingPlaceholder(2),
                    _ShardingPlaceholder(1),
                    *tensor_inputs_tail,
                ],
                [
                    _ShardingPlaceholder(0),
                    _ShardingPlaceholder(0),
                    _ShardingPlaceholder(0),
                    *tensor_inputs_tail,
                ],
            ]
        )

    return strategies
<|endoftext|>#  Copyright (c) Meta Platforms, Inc. and affiliates
"""Schema-discovered registration of single-dim strategy variants.

Manual single-dim registrations target the semantic base op.
``auto_register_op_variants`` mechanically covers the related ATen overloads
(in-place, out, functional, and foreach) by reusing the base op's strategy.
"""

from collections.abc import Sequence
from typing import Any

from torch._ops import OpOverload
from torch.distributed.tensor._dtensor_spec import TensorMeta
from torch.distributed.tensor._op_schema import ArgsType, KwargsType, RuntimeSchemaInfo
from torch.distributed.tensor._ops.single_dim_strategy import (
    _get_overload_packet,
    _get_packet_overload,
    _normalize_schema_type,
    _op_namespace_and_base_name,
    _resolve_foreach_elementwise_overload,
    _schema_args_match,
    _ShardingPlaceholder,
    _SingleDimStrategyFunc,
    _SingleDimStrategyInfo,
)
from torch.distributed.tensor.placement_types import Placement


def _tensor_schema_count(schema_values: Sequence[Any]) -> int:
    """Count Tensor-valued entries in an ATen schema sequence.

    Variant discovery uses this to reject overloads with incompatible arity.
    """
    return sum(1 for value in schema_values if "Tensor" in str(value.type))


def _schema_tensor_output_count(op: OpOverload) -> int:
    """Count Tensor-valued returns for an ATen overload."""
    return _tensor_schema_count(op._schema.returns)


def _is_write_arg(arg: Any) -> bool:
    """Return whether a schema argument is mutated by the op.

    Schema alias info is the reliable way to identify explicit out args and
    mutable inputs whose placements must be preserved.
    """
    return arg.alias_info is not None and arg.alias_info.is_write


def _is_foreach_like_op_name(op_name: str) -> bool:
    """Return whether the op name uses foreach/fused naming conventions.

    These ops are already aggregate/list variants, so they should not be used
    as scalar bases for another round of automatic variant registration.
    """
    return op_name.startswith(
        ("aten::_foreach_", "aten::_amp_foreach_", "aten::_fused_")
    )


def _schema_args_are_same(
    base_args: Sequence[Any], candidate_args: Sequence[Any]
) -> bool:
    """Return whether two schema arg lists are identical for strategy reuse.

    Out variants require this stricter check so only the written output args are
    allowed to differ from the base op.
    """
    if len(base_args) != len(candidate_args):
        return False
    return all(
        base_arg.name == candidate_arg.name
        and base_arg.kwarg_only == candidate_arg.kwarg_only
        and _normalize_schema_type(base_arg) == _normalize_schema_type(candidate_arg)
        for base_arg, candidate_arg in zip(base_args, candidate_args)
    )


def _is_explicit_out_arg(arg: Any) -> bool:
    """Return whether a schema argument is an explicit output argument."""
    return bool(getattr(arg, "is_out", False))


def _schema_non_alias_tensor_output_indices(op: OpOverload) -> list[int]:
    """Return tensor return indices that are not aliases of mutable inputs."""
    indices: list[int] = []
    for idx, ret in enumerate(op._schema.returns):
        if "Tensor" in str(ret.type) and ret.alias_info is None:
            indices.append(idx)
    return indices


def _schema_written_tensor_arg_count(op: OpOverload) -> int:
    """Count Tensor-valued arguments written by an op."""
    return sum(
        1
        for arg in op._schema.arguments
        if _is_write_arg(arg) and "Tensor" in str(arg.type)
    )


def _functional_variant_tensor_output_count(base_op: OpOverload) -> int:
    """Return expected tensor outputs for a functionalization variant.

    Mutable ops can have real tensor returns and/or alias returns for mutated
    inputs. Functionalization returns the real tensor returns plus updated
    copies of written tensor inputs.
    """
    return len(_schema_non_alias_tensor_output_indices(base_op)) + (
        _schema_written_tensor_arg_count(base_op)
    )


def _iter_packet_overloads(packet: Any | None) -> list[OpOverload]:
    """Return overloads from an overload packet, skipping lookup misses."""
    if packet is None:
        return []
    overloads: list[OpOverload] = []
    for overload_name in packet.overloads():
        overload = _get_packet_overload(packet, overload_name)
        if overload is not None:
            overloads.append(overload)
    return overloads


def _count_tensor_meta_values(value: object) -> int:
    """Count TensorMeta leaves in a value that may contain nested tuples/lists.

    Strategy rule indices are based on tensor leaves, not raw schema argument
    count.
    """
    if isinstance(value, TensorMeta):
        return 1
    if isinstance(value, (list, tuple)):
        return sum(_count_tensor_meta_values(item) for item in value)
    return 0


def _clone_schema_info(
    schema_info: RuntimeSchemaInfo | None, *, needs_pytree: bool | None = None
) -> RuntimeSchemaInfo | None:
    """Copy schema cache metadata, optionally overriding pytree flattening.

    Auto-registered variants should inherit the base op cache behavior unless
    their argument structure changes, as foreach does with Tensor lists.
    """
    if schema_info is None:
        if needs_pytree is None:
            return None
        return RuntimeSchemaInfo(needs_pytree=needs_pytree)

    return RuntimeSchemaInfo(
        static_argnum=schema_info.static_argnum,
        static_kwargkey=(
            list(schema_info.static_kwargkey)
            if schema_info.static_kwargkey is not None
            else None
        ),
        needs_pytree=schema_info.needs_pytree if needs_pytree is None else needs_pytree,
    )


def _clone_strategy_info(
    info: _SingleDimStrategyInfo, func: _SingleDimStrategyFunc
) -> _SingleDimStrategyInfo:
    """Copy strategy options while replacing the underlying strategy function.

    Variant wrappers must preserve base-op flags such as uneven sharding support.
    """
    return _SingleDimStrategyInfo(
        func=func,
        allow_unbacked_sharding=info.allow_unbacked_sharding,
        allow_uneven_sharding=info.allow_uneven_sharding,
        different_mesh_args=(
            list(info.different_mesh_args)
            if info.different_mesh_args is not None
            else None
        ),
    )


def _canonical_variant_base_name(op: OpOverload) -> tuple[str, str]:
    """Return the namespace and base name shared by related ATen variants."""
    namespace, base_name = _op_namespace_and_base_name(op)
    if base_name.endswith("_functional"):
        return namespace, base_name.removesuffix("_functional")
    if op._schema.is_mutable and base_name.endswith("_"):
        return namespace, base_name.removesuffix("_")
    return namespace, base_name


def _strip_output_args_for_base_call(
    out_op: OpOverload,
    output_arg_names: set[str],
    args_schema: ArgsType,
    kwargs_schema: KwargsType,
) -> tuple[ArgsType, KwargsType]:
    """Drop explicit out args before invoking the functional base strategy.

    Base strategy functions do not expect the extra writable tensors from the
    out overload schema.
    """
    base_args: list[object] = []
    positional_idx = 0
    for arg in out_op._schema.arguments:
        if arg.kwarg_only:
            continue
        if positional_idx >= len(args_schema):
            break
        value = args_schema[positional_idx]
        positional_idx += 1
        if arg.name not in output_arg_names:
            base_args.append(value)

    base_kwargs = {
        name: value
        for name, value in kwargs_schema.items()
        if name not in output_arg_names
    }
    return tuple(base_args), base_kwargs


def _find_inplace_variant_overloads(base_op: OpOverload) -> list[OpOverload]:
    """Find an inplace overload that has the same tensor schema as base_op.

    Inplace variants can reuse the base strategy only when inputs and outputs
    have the same tensor structure.
    """
    namespace, base_name = _canonical_variant_base_name(base_op)
    if base_op._schema.is_mutable:
        return []
    if _is_foreach_like_op_name(base_op.name()):
        return []
    if any(_is_explicit_out_arg(arg) for arg in base_op._schema.arguments):
        return []

    packet = _get_overload_packet(namespace, f"{base_name}_")
    if packet is None:
        return []

    variants: list[OpOverload] = []
    for candidate in _iter_packet_overloads(packet):
        if not candidate._schema.is_mutable:
            continue
        if _schema_tensor_output_count(candidate) != _schema_tensor_output_count(
            base_op
        ):
            continue
        if _schema_args_match(base_op._schema.arguments, candidate._schema.arguments):
            variants.append(candidate)
    return variants


def _find_out_variant_overloads(
    base_op: OpOverload,
) -> list[tuple[OpOverload, tuple[str, ...]]]:
    """Find an out overload and return its written output arg names.

    Out variants need extra strategy inputs for the provided output tensors.
    """
    namespace, base_name = _canonical_variant_base_name(base_op)
    if _is_foreach_like_op_name(base_op.name()):
        return []
    if base_name.endswith("_functional"):
        return []
    if any(_is_explicit_out_arg(arg) for arg in base_op._schema.arguments):
        return []

    base_num_outputs = _schema_tensor_output_count(base_op)
    if base_num_outputs == 0:
        return []

    packet = _get_overload_packet(namespace, base_name)
    if packet is None:
        return []

    variants: list[tuple[OpOverload, tuple[str, ...]]] = []
    for candidate in _iter_packet_overloads(packet):
        if candidate is base_op:
            continue
        if not candidate._schema.is_mutable:
            continue
        if _schema_tensor_output_count(candidate) != base_num_outputs:
            continue

        output_args = [
            arg for arg in candidate._schema.arguments if _is_explicit_out_arg(arg)
        ]
        if len(output_args) != base_num_outputs:
            continue
        if not all("Tensor" in str(arg.type) for arg in output_args):
            continue

        non_output_args = [
            arg for arg in candidate._schema.arguments if not _is_explicit_out_arg(arg)
        ]
        if _schema_args_are_same(base_op._schema.arguments, non_output_args):
            variants.append((candidate, tuple(arg.name for arg in output_args)))

    return variants


def _find_functional_variant_overloads(base_op: OpOverload) -> list[OpOverload]:
    """Find the functional variant generated from a mutable base overload.

    Functional variants return updated copies of mutable inputs, so their extra
    outputs must follow the mutated input placements.
    """
    if not any(_is_write_arg(arg) for arg in base_op._schema.arguments):
        return []
    namespace, base_name = _canonical_variant_base_name(base_op)
    if _is_foreach_like_op_name(base_op.name()):
        return []
    if any(_is_explicit_out_arg(arg) for arg in base_op._schema.arguments):
        return []

    packet = _get_overload_packet(namespace, f"{base_name}_functional")
    if packet is None:
        return []

    variants: list[OpOverload] = []
    expected_outputs = _functional_variant_tensor_output_count(base_op)
    for candidate in _iter_packet_overloads(packet):
        if candidate._schema.is_mutable:
            continue
        if not _schema_args_match(
            base_op._schema.arguments, candidate._schema.arguments
        ):
            continue
        if _schema_tensor_output_count(candidate) != expected_outputs:
            continue
        variants.append(candidate)
    return variants


def _find_foreach_variants(base_op: OpOverload) -> list[OpOverload]:
    """Find elementwise foreach overloads that can reuse base_op's strategy.

    Foreach variants are registered in one pass because newly registered
    variants are not revisited by auto_register_op_variants.
    """
    namespace, base_name = _op_namespace_and_base_name(base_op)
    if namespace != "aten":
        return []
    if base_op._schema.is_mutable or _is_foreach_like_op_name(base_op.name()):
        return []
    if _schema_tensor_output_count(base_op) == 0:
        return []

    variants: list[OpOverload] = []
    for foreach_name in (f"_foreach_{base_name}", f"_foreach_{base_name}_"):
        packet = _get_overload_packet(namespace, foreach_name)
        if packet is None:
            continue
        for overload_name in packet.overloads():
            if "out" in overload_name:
                continue
            candidate = _get_packet_overload(packet, overload_name)
            if candidate is None:
                continue
            if _resolve_foreach_elementwise_overload(candidate) == base_op:
                variants.append(candidate)
    return variants


def _make_same_schema_variant_strategy_fn(
    base_fn: _SingleDimStrategyFunc, base_op: OpOverload
) -> _SingleDimStrategyFunc:
    """Wrap base_fn so a same-schema variant uses base-op strategy logic.

    Passing base_op keeps op-sensitive strategy functions on their original
    lookup paths.
    """

    def strategy(
        op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
    ) -> list[list[Placement | _ShardingPlaceholder]]:
        return base_fn(base_op, args_schema, kwargs_schema)

    return strategy


def _make_out_variant_strategy_fn(
    base_fn: _SingleDimStrategyFunc,
    base_op: OpOverload,
    out_op: OpOverload,
    output_arg_names: tuple[str, ...],
) -> _SingleDimStrategyFunc:
    """Wrap base_fn so out tensors appear as extra strategy inputs.

    DTensor dispatch cannot redistribute out kwargs, so generated rules must
    make each output placement agree with its corresponding out tensor.
    """
    output_arg_name_set = set(output_arg_names)
    output_arg_to_index = {
        name: output_idx for output_idx, name in enumerate(output_arg_names)
    }
    base_num_outputs = _schema_tensor_output_count(base_op)

    def _output_arg_placements(
        rule: list[Placement | _ShardingPlaceholder],
        arg_name: str,
        value: object,
    ) -> list[Placement | _ShardingPlaceholder]:
        """Return placements to append for one explicit out argument."""
        tensor_count = _count_tensor_meta_values(value)
        if tensor_count == 0:
            return []
        if tensor_count > 1 and len(output_arg_names) == 1:
            return [rule[i] for i in range(base_num_outputs)]
        return [rule[output_arg_to_index[arg_name]]]

    def strategy(
        op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
    ) -> list[list[Placement | _ShardingPlaceholder]]:
        base_args, base_kwargs = _strip_output_args_for_base_call(
            out_op, output_arg_name_set, args_schema, kwargs_schema
        )
        base_rules = base_fn(base_op, base_args, base_kwargs)
        rules: list[list[Placement | _ShardingPlaceholder]] = []

        for rule in base_rules:
            output_arg_placements: list[Placement | _ShardingPlaceholder] = []
            positional_idx = 0
            for arg in out_op._schema.arguments:
                if arg.kwarg_only:
                    continue
                if positional_idx >= len(args_schema):
                    break
                value = args_schema[positional_idx]
                positional_idx += 1
                if arg.name in output_arg_to_index:
                    output_arg_placements.extend(
                        _output_arg_placements(rule, arg.name, value)
                    )

            for name, value in kwargs_schema.items():
                if name in output_arg_to_index:
                    output_arg_placements.extend(
                        _output_arg_placements(rule, name, value)
                    )

            rules.append([*rule, *output_arg_placements])

        return rules

    return strategy


def _make_functional_variant_strategy_fn(
    base_fn: _SingleDimStrategyFunc, base_op: OpOverload
) -> _SingleDimStrategyFunc:
    """Wrap mutable base_fn for functional variants with extra mutated outputs.

    The added functional outputs represent updated mutable inputs and therefore
    inherit those input placements.
    """
    mutable_arg_names = {
        arg.name for arg in base_op._schema.arguments if _is_write_arg(arg)
    }
    base_num_outputs = _schema_tensor_output_count(base_op)
    non_alias_output_indices = _schema_non_alias_tensor_output_indices(base_op)

    def _mutable_input_rule_indices(
        args_schema: ArgsType, kwargs_schema: KwargsType
    ) -> list[int]:
        """Map mutable schema args to their positions in a base strategy rule."""
        mutable_indices: list[int] = []
        tensor_input_idx = 0
        positional_idx = 0
        for arg in base_op._schema.arguments:
            value = None
            if arg.kwarg_only:
                value = kwargs_schema.get(arg.name)
            elif positional_idx < len(args_schema):
                value = args_schema[positional_idx]
                positional_idx += 1
            else:
                value = kwargs_schema.get(arg.name)

            tensor_count = _count_tensor_meta_values(value)
            if tensor_count == 0:
                continue
            if arg.name in mutable_arg_names:
                mutable_indices.append(base_num_outputs + tensor_input_idx)
            tensor_input_idx += tensor_count
        return mutable_indices

    def strategy(
        op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
    ) -> list[list[Placement | _ShardingPlaceholder]]:
        mutable_input_indices = _mutable_input_rule_indices(args_schema, kwargs_schema)
        base_rules = base_fn(base_op, args_schema, kwargs_schema)
        rules: list[list[Placement | _ShardingPlaceholder]] = []
        for rule in base_rules:
            output_placements = [
                rule[output_idx] for output_idx in non_alias_output_indices
            ]
            output_placements.extend(
                rule[input_idx] for input_idx in mutable_input_indices
            )
            rules.append([*output_placements, *rule[base_num_outputs:]])
        return rules

    return strategy


def auto_register_op_variants() -> None:
    """Register schema-discovered variants for existing single-dim strategies.

    This keeps manual registrations focused on the semantic base op while
    covering mechanically related ATen overloads (in-place, out, functional,
    and foreach).
    """
    from torch.distributed.tensor._api import DTensor

    propagator = DTensor._op_dispatcher.sharding_propagator
    registry = propagator.op_single_dim_strategy_funcs
    already_registered = set(registry)

    for base_op, info in list(registry.items()):
        base_schema_info = propagator.op_to_schema_info_for_single_dim_strategy.get(
            base_op
        )

        for inplace_op in _find_inplace_variant_overloads(base_op):
            if inplace_op in already_registered:
                continue
            propagator.register_single_dim_op_strategy(
                inplace_op,
                _clone_strategy_info(
                    info, _make_same_schema_variant_strategy_fn(info.func, base_op)
                ),
                _clone_schema_info(base_schema_info),
            )
            already_registered.add(inplace_op)

        for out_variant in _find_out_variant_overloads(base_op):
            out_op, output_arg_names = out_variant
            if out_op in already_registered:
                continue
            propagator.register_single_dim_op_strategy(
                out_op,
                _clone_strategy_info(
                    info,
                    _make_out_variant_strategy_fn(
                        info.func, base_op, out_op, output_arg_names
                    ),
                ),
                _clone_schema_info(base_schema_info),
            )
            already_registered.add(out_op)

        for functional_op in _find_functional_variant_overloads(base_op):
            if functional_op in already_registered:
                continue
            propagator.register_single_dim_op_strategy(
                functional_op,
                _clone_strategy_info(
                    info, _make_functional_variant_strategy_fn(info.func, base_op)
                ),
                _clone_schema_info(base_schema_info),
            )
            already_registered.add(functional_op)

        for foreach_op in _find_foreach_variants(base_op):
            if foreach_op in already_registered:
                continue
            propagator.register_single_dim_op_strategy(
                foreach_op,
                info,
                RuntimeSchemaInfo(needs_pytree=True),
            )
            already_registered.add(foreach_op)
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates
"""
Strategy validation for DTensor sharding rules.

This module provides utilities to validate DTensor's sharding strategies by:
1. Running operators on full tensors to get ground truth
2. Simulating sharding with various placement combinations
3. Comparing redistributed outputs against ground truth
4. Reporting incorrect rules (DTensor claims valid but wrong) and
   missing rules (ground truth valid but DTensor has no rule)

Run as a module to compare DTensor rules against ground truth:
    python -m torch.distributed.tensor._ops.strategy_validation --op add
    python -m torch.distributed.tensor._ops.strategy_validation --op relu,mul,add
    python -m torch.distributed.tensor._ops.strategy_validation --op "nn.functional.*"
    python -m torch.distributed.tensor._ops.strategy_validation --all-registered
    python -m torch.distributed.tensor._ops.strategy_validation --op div --incorrect-only
    python -m torch.distributed.tensor._ops.strategy_validation --op add --show-repro
"""

import argparse
import fnmatch
import itertools
import re
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
import torch.testing._internal.common_methods_invocations as common_ops
from torch._ops import OpOverload
from torch.distributed._local_tensor import LocalTensor, LocalTensorMode
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate
from torch.distributed.tensor._decompositions import DecompShardingStrategy
from torch.distributed.tensor._dtensor_spec import TensorMeta
from torch.distributed.tensor._op_schema import (
    DTensorSpec,
    OpSchema,
    OpSpec,
    OpStrategy,
)
from torch.distributed.tensor._ops.single_dim_strategy import _ShardingPlaceholder
from torch.distributed.tensor.placement_types import Partial, Placement, Shard
from torch.testing._internal.common_methods_invocations import op_db, SampleInput
from torch.testing._internal.opinfo import core as opinfo_core
from torch.utils import _pytree as pytree


# A combo key is (input_placement_strs, output_placement_strs)
# For single-output ops: (("S(0)",), ("P(min)",))
# For multi-output ops:  (("S(0)",), ("P(min)", "P(min)"))
ComboKey = tuple[tuple[str, ...], tuple[str, ...]]

# Partial reduce ops to enumerate
PARTIAL_REDUCE_OPS = ["sum", "avg", "min", "max"]

SKIP_OPS: dict[str, str] = {
    "bernoulli": "non-deterministic (random sampling)",
    "empty_like": "uninitialized memory",
    "exponential": "non-deterministic (random sampling)",
    "new_empty": "uninitialized memory",
    "new_empty_strided": "uninitialized memory",
    "nn.functional.dropout": "non-deterministic (random masking)",
    "normal": "non-deterministic (random sampling)",
    "rand_like": "non-deterministic (random sampling)",
    "randint_like": "non-deterministic (random sampling)",
    "randn_like": "non-deterministic (random sampling)",
    "uniform": "non-deterministic (random sampling)",
}


PlacementCombination = tuple[tuple[Placement, ...], tuple[Placement, ...]]


@dataclass
class Discrepancy:
    """Represents a discrepancy between ground truth and DTensor's rules."""

    input_placements: tuple[str, ...]
    output_placements: tuple[str, ...]
    sample_idx: int
    input_shapes: tuple[tuple[int, ...], ...]
    discrepancy_type: str  # "false_positive" or "false_negative"
    error_msg: str = ""
    scalar_args: tuple[Any, ...] = ()
    scalar_kwargs: dict[str, Any] = field(default_factory=dict)
    aten_op: OpOverload | None = None
    variant: str = ""
    sample: SampleInput | None = None


@dataclass
class ComparisonStats:
    """Statistics for comparing ground truth vs DTensor rules."""

    true_positives: int = 0
    true_negatives: int = 0
    false_positives: list[Discrepancy] = field(
        default_factory=list
    )  # DTensor has rule, ground truth says invalid
    false_negatives: list[Discrepancy] = field(
        default_factory=list
    )  # Ground truth valid, DTensor has no rule
    total_samples: int = 0
    total_combinations: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)
    no_dtensor_support: bool = False
    # Per aten op variant breakdown (e.g. "aten.min.dim" -> 5)
    true_positives_by_op: dict[str, int] = field(default_factory=dict)


@dataclass
class _FalsePositiveMitigations:
    """Bundle of sample variants used to detect false positive validations.

    Contains negated and non-rounded variants of a sample. These are used to
    re-test combinations that pass the primary validation, catching cases where
    sign patterns or rounding modes mask real differences.
    """

    negated_sample: SampleInput | None = None
    negated_tensors: list[tuple[str, torch.Tensor]] | None = None
    negated_ground_truth: torch.Tensor | list[torch.Tensor] | None = None
    non_rounded_sample: SampleInput | None = None
    non_rounded_ground_truth: torch.Tensor | list[torch.Tensor] | None = None
    non_rounded_negated_sample: SampleInput | None = None
    non_rounded_negated_tensors: list[tuple[str, torch.Tensor]] | None = None
    non_rounded_negated_ground_truth: torch.Tensor | list[torch.Tensor] | None = None


def parse_placement(s: str) -> Placement | None:
    """
    Parse a placement string back to a placement object.
    Placement strings are: R, S(dim), P(reduce_op)
    """
    s = s.strip()
    if s == "R":
        return Replicate()
    elif s.startswith("S("):
        m = re.match(r"S\((\d+)\)", s)
        if m:
            return Shard(int(m.group(1)))
    elif s.startswith("P("):
        m = re.match(r"P\((\w+)\)", s)
        if m:
            return Partial(m.group(1))
    return None


def is_fully_replicated(placements: tuple[Placement, ...]) -> bool:
    """Check if all placements are Replicate."""
    return all(isinstance(p, Replicate) for p in placements)


def is_trivial_shard(p: Placement, tensor_shape: tuple[int, ...]) -> bool:
    """Check if placement is a Shard on a size-1 dimension."""
    # NOTE: isinstance(_, Shard) does not match _StridedShard; see _is_shard_like().
    return (
        isinstance(p, Shard) and p.dim < len(tensor_shape) and tensor_shape[p.dim] == 1
    )


def normalize_placement(p: Placement, tensor_shape: tuple[int, ...]) -> Placement:
    """
    Normalize a placement for a given tensor shape.

    Converts Shard on a size-1 dimension to Replicate for deduplication.
    Shard(0) on a [1, 4] tensor puts all data on rank 0 and an empty [0, 4]
    on rank 1. Rank 1's empty computation is vacuous (contributes nothing
    after redistribution), so the validation outcome is determined entirely
    by rank 0, which has the full data — same as Replicate. We gain no
    signal from testing S(0) on a size-1 dim beyond what R already provides,
    so we normalize to R to avoid spurious "missing rule" noise when ground
    truth and DTensor use different forms for size-1 dims.
    """
    if is_trivial_shard(p, tensor_shape):
        return Replicate()
    return p


def normalize_placement_str(p_str: str, shape: tuple[int, ...]) -> str:
    """Normalize a placement string, converting trivial shards to Replicate."""
    p = parse_placement(p_str)
    if p is None:
        return p_str
    normalized = normalize_placement(p, shape)
    if isinstance(normalized, Replicate):
        return "R"
    return p_str


def normalize_combo_key(
    combo_key: ComboKey,
    input_shapes: tuple[tuple[int, ...], ...],
    output_shapes: tuple[tuple[int, ...], ...],
) -> ComboKey:
    """
    Normalize a combo_key by converting trivial shards to Replicate.

    This deduplicates equivalent placement combinations, e.g.:
    - P(max) -> S(0) on output [1,1,1] becomes P(max) -> R
    - S(0), R -> R on input [1,4] becomes R, R -> R

    Args:
        combo_key: (input_placement_strs, output_placement_strs) tuple
        input_shapes: Shapes of input tensors
        output_shapes: Shapes of output tensors

    Returns:
        Normalized combo_key with trivial shards converted to Replicate
    """
    input_placement_strs, output_placement_strs = combo_key

    normalized_inputs = tuple(
        normalize_placement_str(p_str, shape)
        for p_str, shape in zip(input_placement_strs, input_shapes)
    )

    normalized_outputs = tuple(
        normalize_placement_str(p_str, shape)
        for p_str, shape in zip(output_placement_strs, output_shapes)
    )

    return (normalized_inputs, normalized_outputs)


def get_1d_input_placements_for_tensor(
    t: torch.Tensor, include_partial: bool = False
) -> list[Placement]:
    """
    Get all possible 1-D mesh placements for an INPUT tensor.

    Args:
        t: The tensor to get placements for
        include_partial: If True, include Partial placements for inputs.
    """
    placements: list[Placement] = [Replicate()]
    for dim in range(t.ndim):
        placements.append(Shard(dim))
    if include_partial and t.dtype != torch.bool:
        for reduce_op in PARTIAL_REDUCE_OPS:
            placements.append(Partial(reduce_op))
    return placements


def get_1d_output_placements_for_tensor(t: torch.Tensor) -> list[Placement]:
    """
    Get all possible 1-D mesh placements for an OUTPUT tensor.
    """
    placements: list[Placement] = [Replicate()]
    for dim in range(t.ndim):
        placements.append(Shard(dim))

    if t.dtype != torch.bool:
        for reduce_op in PARTIAL_REDUCE_OPS:
            placements.append(Partial(reduce_op))
    return placements


def extract_tensors_from_sample(
    sample_input: SampleInput,
) -> list[tuple[str, torch.Tensor]]:
    """
    Extract all tensor arguments from a SampleInput.
    Returns a list of (name, tensor) pairs.

    Uses pytree traversal to match the same order as _replace_with_local
    in validate_combination, which uses pytree.tree_map on the same structures.
    """
    tensors: list[tuple[str, torch.Tensor]] = []
    idx = 0

    def _collect(x):
        nonlocal idx
        if isinstance(x, torch.Tensor):
            tensors.append((f"tensor_{idx}", x))
            idx += 1
        return x

    pytree.tree_map(_collect, sample_input.input)
    pytree.tree_map(_collect, sample_input.args)
    pytree.tree_map(_collect, sample_input.kwargs)

    return tensors


def _checkerboard_mask(
    tensor: torch.Tensor, tensor_idx: int = 0, mask_shift: int = 0
) -> torch.Tensor:
    """Checkerboard mask that alternates in every dimension.

    Unlike flat-index % 2, which can be uniform along even-stride dimensions
    (causing all elements in a reduction group to get the same offset), the
    checkerboard uses sum-of-coordinates mod 2 so adjacent elements differ
    along every axis.

    Returns a flat bool tensor of shape (numel,).
    """
    if tensor.ndim == 0:
        return torch.tensor([(tensor_idx + mask_shift) % 2 == 0], device=tensor.device)
    coords = [torch.arange(s, device=tensor.device) for s in tensor.shape]
    grids = torch.meshgrid(*coords, indexing="ij")
    coord_sum = grids[0].clone()
    for g in grids[1:]:
        coord_sum += g
    return ((coord_sum + tensor_idx + mask_shift) % 2 == 0).flatten()


def _create_partial_input(
    tensor: torch.Tensor,
    placement: Partial,
    world_size: int,
    tensor_idx: int = 0,
    mask_shift: int = 0,
) -> LocalTensor:
    """
    Create a LocalTensor with values that reduce to the original tensor.

    Uses asymmetric splits to avoid coincidental matches when combining
    different Partial types.

    For each placement combination, it creates local tensors that would
    reduce to the original (e.g., for P(sum), splits values across ranks so
    they sum back), runs the op on those local tensors, wraps the output as
    a DTensor, redistributes to Replicate, and compares against ground
    truth.

    The main challenge is avoiding false positives where a rule appears
    valid on a specific input but is actually incorrect. Several techniques
    are used:

    Asymmetric splits for P(sum)/P(avg): instead of splitting evenly
    (tensor/2 per rank), uses a 60/40 ratio (varied by tensor index) so
    that ops which are not truly linear don't accidentally produce
    matching outputs.

    Sign-varying offsets for P(sum)/P(avg): adds an offset that
    alternates sign across elements, so local tensors have mixed positive
    and negative values. Without this, proportional splits preserve the
    sign pattern of the original tensor, causing non-linear ops like abs
    to falsely validate P(sum)->P(sum).

    Distinct magnitudes for P(min) vs P(max): P(min) offsets non-holding
    ranks by +(range*2+1) while P(max) offsets by -(range*2+1), where
    range is the tensor's value range. Using adaptive offsets that exceed
    the value range ensures that index-returning ops (argmin/argmax)
    produce different results on different ranks, correctly rejecting
    P(min)/P(max) inputs for those ops. Using different signs for min vs
    max prevents accidental cancellation when both appear in the same
    combination.

    Alternating rank ownership for P(min)/P(max): a multi-dimensional
    checkerboard mask (sum of coordinates mod 2) controls which rank holds
    the true value vs the offset value. Unlike a flat-index mask which can
    have uniform parity along an even-stride dimension, the checkerboard
    guarantees alternation along EVERY dimension. The mask_shift parameter
    allows re-validation with the complementary mask to catch ops where
    the result coincidentally matches.

    """
    reduce_op = placement.reduce_op
    local_tensors: dict[int, torch.Tensor] = {}

    if reduce_op in ("sum", "avg"):
        base_ratio = 0.6 + 0.1 * (tensor_idx % 3)

        # See docstring above: "Sign-varying offsets"
        flat = tensor.flatten()
        offset_mag = flat.abs() + 1.0
        signs = torch.ones_like(flat)
        # Use checkerboard mask so offset sign alternates in every dimension,
        # not just along flat index (which can be uniform along even-stride dims).
        signs[_checkerboard_mask(tensor, tensor_idx, mask_shift)] = -1.0
        offset = (offset_mag * signs).reshape(tensor.shape)

        scale = world_size if reduce_op == "avg" else 1
        for r in range(world_size):
            if r == 0:
                local_tensors[r] = tensor.clone() * base_ratio * scale + offset
            else:
                local_tensors[r] = tensor.clone() * (
                    (1 - base_ratio) / (world_size - 1)
                ) * scale - offset / (world_size - 1)

    elif reduce_op == "min":
        # See docstring above: "Distinct Magnitudes" and "Alternating Rank Ownership"
        flat = tensor.flatten()
        # Offset must exceed the tensor's value range so that the mask pattern
        # determines argmin/argmax, not the original values.
        value_range = (flat.max() - flat.min()).item()
        min_offset = value_range * 2 + 1
        mask = _checkerboard_mask(tensor, tensor_idx, mask_shift)
        for r in range(world_size):
            if r == 0:
                r_offset = torch.where(
                    mask, torch.zeros_like(flat), torch.full_like(flat, min_offset)
                )
            else:
                r_offset = torch.where(
                    mask, torch.full_like(flat, min_offset), torch.zeros_like(flat)
                )
            local_tensors[r] = (flat + r_offset).reshape(tensor.shape)

    elif reduce_op == "max":
        # See docstring above: "Distinct Magnitudes" and "Alternating Rank Ownership"
        flat = tensor.flatten()
        value_range = (flat.max() - flat.min()).item()
        max_offset = -(value_range * 2 + 1)
        mask = _checkerboard_mask(tensor, tensor_idx, mask_shift)
        for r in range(world_size):
            if r == 0:
                r_offset = torch.where(
                    mask, torch.zeros_like(flat), torch.full_like(flat, max_offset)
                )
            else:
                r_offset = torch.where(
                    mask, torch.full_like(flat, max_offset), torch.zeros_like(flat)
                )
            local_tensors[r] = (flat + r_offset).reshape(tensor.shape)

    else:
        for r in range(world_size):
            local_tensors[r] = tensor.clone()

    # pyrefly: ignore [bad-argument-type, bad-argument-count]
    return LocalTensor(local_tensors)


def _shard_tensors(
    tensors: list[tuple[str, torch.Tensor]],
    input_placements: tuple[Placement, ...],
    world_size: int,
    mesh: DeviceMesh,
    mask_shift: int = 0,
) -> list[LocalTensor | torch.Tensor]:
    """Create sharded LocalTensors from tensors according to placements."""
    local_tensors: list[LocalTensor | torch.Tensor] = []
    for tensor_idx, ((name, tensor), placement) in enumerate(
        zip(tensors, input_placements)
    ):
        if isinstance(placement, Partial):
            local_tensor = _create_partial_input(
                tensor, placement, world_size, tensor_idx, mask_shift
            )
        elif isinstance(placement, Replicate):
            _tmp = {r: tensor.clone() for r in range(world_size)}
            # pyrefly: ignore [bad-argument-type, bad-argument-count]
            local_tensor = LocalTensor(_tmp)
        elif isinstance(placement, Shard):
            shard_dim = placement.dim
            chunks = tensor.tensor_split(world_size, dim=shard_dim)
            _tmp = {
                r: chunks[r].clone(memory_format=torch.contiguous_format)
                for r in range(world_size)
            }
            # pyrefly: ignore [bad-argument-type, bad-argument-count]
            local_tensor = LocalTensor(_tmp)
        else:
            dt = distribute_tensor(tensor.clone(), mesh, (placement,))
            local_tensor = dt.to_local()
        local_tensors.append(local_tensor)
    return local_tensors


def _compare_outputs(
    local_output: Any,
    ground_truth: torch.Tensor | list[torch.Tensor],
    output_placements: tuple[Placement, ...],
    mesh: DeviceMesh,
    world_size: int,
) -> tuple[bool, str]:
    """Compare op output (wrapped as DTensor) against ground truth."""
    if isinstance(local_output, (list, tuple)):
        local_outputs = list(local_output)
    else:
        local_outputs = [local_output]

    if isinstance(ground_truth, list):
        ground_truths = ground_truth
    else:
        ground_truths = [ground_truth]

    if len(local_outputs) != len(ground_truths):
        return (
            False,
            f"Output count mismatch: got {len(local_outputs)}, "
            f"expected {len(ground_truths)}",
        )

    if len(local_outputs) != len(output_placements):
        return (
            False,
            f"Output count mismatch with placements: "
            f"got {len(local_outputs)}, expected {len(output_placements)}",
        )

    for i, (local_out, gt, out_plc) in enumerate(
        zip(local_outputs, ground_truths, output_placements)
    ):
        if not isinstance(local_out, torch.Tensor):
            return False, f"Local output[{i}] is not a tensor: {type(local_out)}"

        if not isinstance(local_out, LocalTensor):
            return False, f"LocalTensor inputs produced non-LocalTensor output[{i}]"

        output_dt = DTensor.from_local(
            local_out,
            mesh,
            (out_plc,),
            shape=gt.shape,
            stride=gt.stride(),
        )

        if isinstance(out_plc, Replicate):
            local_values = [local_out._local_tensors[r] for r in range(world_size)]
            all_same = all(
                torch.allclose(local_values[0], lv, atol=1e-5, rtol=1e-5)
                for lv in local_values[1:]
            )
            if not all_same:
                return (
                    False,
                    f"Replicate output[{i}] but local values differ across ranks",
                )

        full_output = output_dt.redistribute(mesh, (Replicate(),)).to_local()

        if isinstance(full_output, LocalTensor):
            full_output = full_output._local_tensors[0]

        if gt.shape != full_output.shape:
            return (
                False,
                f"Shape mismatch[{i}]: expected {gt.shape}, got {full_output.shape}",
            )

        if not torch.allclose(gt, full_output, atol=1e-5, rtol=1e-5, equal_nan=True):
            max_diff = (gt - full_output).abs().max().item()
            return False, f"Value mismatch[{i}]: max_diff={max_diff:.6f}"

    return True, ""


def validate_combination(
    op: Callable[..., Any],
    sample_input: SampleInput,
    tensors: list[tuple[str, torch.Tensor]],
    combination: PlacementCombination,
    ground_truth: torch.Tensor | list[torch.Tensor],
    world_size: int = 2,
    mesh: DeviceMesh | None = None,
    mask_shift: int = 0,
) -> tuple[bool | None, str]:
    """
    Validate a single placement combination against ground truth.

    Returns (True, "") if valid, (False, error_msg) if invalid, or
    (None, reason) if the combination cannot be tested (e.g. uneven shards).

    The validation logic:
    1. Shard inputs according to input placements to get local tensors
    2. Run the raw op on local tensors (bypassing DTensor dispatch)
    3. Wrap the local output in a DTensor with the claimed output placement
    4. Redistribute to Replicate and compare with ground truth

    Args:
        op: The operator function
        sample_input: The SampleInput with original arguments
        tensors: List of (name, tensor) pairs extracted from sample
        combination: The placement combination to validate
        ground_truth: Expected output tensor(s). For multi-output ops, a list
            of tensors where each element is validated independently against
            the same output placement.
        world_size: Number of simulated ranks
        mesh: Optional pre-created device mesh (for performance)

    Returns:
        (is_valid, error_message)
    """
    try:
        if mesh is None:
            device = tensors[0][1].device.type if tensors else "cpu"
            mesh = init_device_mesh(device, (world_size,))

        # Uneven shards produce SymInt in LocalTensor's wrapper shape,
        # which breaks C++ overload resolution before __torch_dispatch__
        # can intercept. Return None to signal "untestable".
        for (name, tensor), placement in zip(tensors, combination[0]):
            if isinstance(placement, Shard):
                if tensor.size(placement.dim) % world_size != 0:
                    return None, "uneven shard"

        local_tensors = _shard_tensors(
            tensors, combination[0], world_size, mesh, mask_shift
        )

        local_idx = 0

        def _replace_with_local(a):
            nonlocal local_idx
            if isinstance(a, torch.Tensor):
                local = local_tensors[local_idx]
                local_idx += 1
                return local
            return a

        if isinstance(sample_input.input, torch.Tensor):
            local_input = _replace_with_local(sample_input.input)
        else:
            local_input = pytree.tree_map(_replace_with_local, sample_input.input)

        local_args = pytree.tree_map(_replace_with_local, sample_input.args)
        local_kwargs = pytree.tree_map(_replace_with_local, sample_input.kwargs)

        local_output = op(local_input, *local_args, **local_kwargs)

        return _compare_outputs(
            local_output, ground_truth, combination[1], mesh, world_size
        )

    except Exception as e:
        # TODO: This is too broad. Consider: (1) explicit checks for shard dim
        # validity and shape compatibility before calling tensor_split/from_local,
        # (2) scoped try/except around op() and redistribute() that raise specific
        # exceptions (e.g., UnsupportedRedistribute, OpError), and (3) only
        # catching those here, letting real bugs propagate.
        return False, f"Exception: {type(e).__name__}: {e}"


def extract_tensors_from_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> list[tuple[str, torch.Tensor]]:
    """Extract tensor arguments from captured aten args/kwargs.

    Unlike extract_tensors_from_sample which walks SampleInput pytrees,
    this walks the flat aten-level args and kwargs directly.
    """
    tensors: list[tuple[str, torch.Tensor]] = []
    idx = 0

    def _collect(x: Any) -> Any:
        nonlocal idx
        if isinstance(x, torch.Tensor):
            tensors.append((f"tensor_{idx}", x))
            idx += 1
        return x

    pytree.tree_map(_collect, args)
    pytree.tree_map(_collect, kwargs)
    return tensors


def validate_aten_combination(
    aten_op: OpOverload,
    captured_args: tuple[Any, ...],
    captured_kwargs: dict[str, Any],
    ground_truth: torch.Tensor | list[torch.Tensor],
    combination: PlacementCombination,
    world_size: int,
    mesh: DeviceMesh,
    mask_shift: int = 0,
) -> tuple[bool | None, str]:
    """Validate a placement combination using aten-level captured args.

    Works directly with aten op args/kwargs instead of SampleInput pytrees.
    Replaces tensors in the flat args/kwargs with sharded LocalTensors,
    calls the aten op, and compares output.

    Returns (True, ""), (False, error_msg), or (None, reason) if untestable.
    """
    try:
        tensors = extract_tensors_from_args(captured_args, captured_kwargs)
        if not tensors:
            return False, "No tensor args in captured aten call"

        for (name, tensor), placement in zip(tensors, combination[0]):
            if isinstance(placement, Shard):
                if tensor.size(placement.dim) % world_size != 0:
                    return None, "uneven shard"

        local_tensors = _shard_tensors(
            tensors, combination[0], world_size, mesh, mask_shift
        )

        local_idx = 0

        def _replace_with_local(a: Any) -> Any:
            nonlocal local_idx
            if isinstance(a, torch.Tensor):
                local = local_tensors[local_idx]
                local_idx += 1
                return local
            return a

        local_args = pytree.tree_map(_replace_with_local, captured_args)
        local_kwargs = pytree.tree_map(_replace_with_local, captured_kwargs)

        local_output = aten_op(*local_args, **local_kwargs)

        return _compare_outputs(
            local_output, ground_truth, combination[1], mesh, world_size
        )

    except Exception as e:
        return False, f"Exception: {type(e).__name__}: {e}"


def has_pmin_pmax(
    input_placements: tuple[Placement, ...],
    output_placements: tuple[Placement, ...],
) -> bool:
    """Check if any placement is Partial(min) or Partial(max)."""
    for p in (*input_placements, *output_placements):
        if isinstance(p, Partial) and p.reduce_op in ("min", "max"):
            return True
    return False


def has_any_partial(
    input_placements: tuple[Placement, ...],
    output_placements: tuple[Placement, ...],
) -> bool:
    """Check if any placement is Partial (any reduce op)."""
    for p in (*input_placements, *output_placements):
        if isinstance(p, Partial):
            return True
    return False


def negate_all_tensors(
    tensors: list[tuple[str, torch.Tensor]],
) -> list[tuple[str, torch.Tensor]]:
    """Return a new list with all tensors negated."""
    return [(name, -t) for name, t in tensors]


def create_fully_negated_sample(sample: SampleInput) -> SampleInput:
    """Create a sample with ALL tensors negated (for P(min)/P(max) sign testing)."""

    def negate_tensor(x):
        if isinstance(x, torch.Tensor):
            return -x
        return x

    new_input = pytree.tree_map(negate_tensor, sample.input)
    new_args = pytree.tree_map(negate_tensor, sample.args)
    new_kwargs = pytree.tree_map(negate_tensor, sample.kwargs)

    return SampleInput(new_input, args=new_args, kwargs=new_kwargs)


def _run_op_on_sample(op: Callable[..., Any], sample: SampleInput) -> Any:
    """Run an operator on a SampleInput, handling both tensor and tuple inputs."""
    if isinstance(sample.input, torch.Tensor):
        return op(sample.input, *sample.args, **sample.kwargs)
    return op(*sample.input, *sample.args, **sample.kwargs)


def _extract_rules_from_op_strategy(
    op_strategy: Any,
    input_shapes: tuple[tuple[int, ...], ...],
    output_shapes: tuple[tuple[int, ...], ...],
) -> set[ComboKey]:
    """Extract normalized sharding rules from an OpStrategy.

    Called during rule comparison to collect DTensor's claimed-valid placement
    combinations. These are compared against ground truth (brute-force
    validation) to find false positives (DTensor claims valid but wrong) and
    false negatives (valid but DTensor has no rule).
    """
    rules: set[ComboKey] = set()
    if not isinstance(op_strategy, OpStrategy):
        return rules
    for spec in op_strategy.strategies:
        if spec.input_specs is None:
            continue
        if isinstance(spec.output_specs, tuple):
            output_plcs: list[Placement] = []
            has_none = False
            for out_spec in spec.output_specs:
                if out_spec is None:
                    # None means the output placement is undefined for this
                    # strategy (e.g. indices under P(max) reduction). Skip it.
                    has_none = True
                    break
                output_plcs.append(out_spec.placements[0])
            if has_none:
                continue
        else:
            # Single DTensorSpec — the propagator duplicates it for all
            # outputs of multi-output ops, so we do the same here.
            output_plcs = [spec.output_spec.placements[0]] * len(output_shapes)
        input_plcs = tuple(s.placements[0] for s in spec.input_specs)
        rule_key: ComboKey = (
            tuple(str(p) for p in input_plcs),
            tuple(str(p) for p in output_plcs),
        )
        normalized_rule = normalize_combo_key(rule_key, input_shapes, output_shapes)
        if not is_fully_replicated(
            tuple(parse_placement(p) or Replicate() for p in normalized_rule[0])
        ):
            rules.add(normalized_rule)
    return rules


class _CaptureAtenOp(torch.utils._python_dispatch.TorchDispatchMode):
    """Dispatch mode that captures aten ops called, their args, and return values."""

    def __init__(self, target_op_name: str = ""):
        self.target_op_name = target_op_name.lower()
        self.all_ops: list[tuple[OpOverload, tuple[Any, ...], dict[str, Any], Any]] = []
        self.best_match: OpOverload | None = None
        self.best_match_args: tuple[Any, ...] | None = None
        self.best_match_kwargs: dict[str, Any] | None = None
        self.best_match_result: Any = None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        result = func(*args, **kwargs)
        if func.namespace == "aten":
            self.all_ops.append((func, args, kwargs, result))
            op_name = func.name().split("::")[1].split(".")[0].lower()
            if self.target_op_name and self.target_op_name in op_name:
                if self.best_match is None:
                    self.best_match = func
                    self.best_match_args = args
                    self.best_match_kwargs = kwargs
                    self.best_match_result = result
        return result


def get_aten_op_for_sample(
    op: Callable[..., Any], sample: SampleInput, op_name: str = ""
) -> _CaptureAtenOp:
    """
    Capture aten ops dispatched for a given sample.

    Returns the _CaptureAtenOp object containing all captured ops with their
    args, kwargs, and return values. Use best_match for the primary op or
    all_ops for exhaustive iteration.
    """
    with _CaptureAtenOp(op_name) as capture:
        try:
            if isinstance(sample.input, torch.Tensor):
                op(sample.input, *sample.args, **sample.kwargs)
            else:
                op(*sample.input, *sample.args, **sample.kwargs)
        except Exception:
            pass

    # Populate best_match from first op if target match wasn't found
    if capture.best_match is None and capture.all_ops:
        first_op, first_args, first_kwargs, first_result = capture.all_ops[0]
        capture.best_match = first_op
        capture.best_match_args = first_args
        capture.best_match_kwargs = first_kwargs
        capture.best_match_result = first_result

    return capture


def query_single_dim_strategy(
    op_overload: OpOverload,
    captured_args: tuple[Any, ...],
    captured_kwargs: dict[str, Any],
) -> list[list[Placement]] | None:
    """
    Query DTensor's single-dim strategy for given input tensors.
    Returns list of [output_placement, *input_placements] rules.
    """
    propagator = DTensor._op_dispatcher.sharding_propagator

    if op_overload not in propagator.op_single_dim_strategy_funcs:
        return None

    strategy_func = propagator.op_single_dim_strategy_funcs[op_overload]

    args_meta = tuple(
        TensorMeta(shape=a.shape, stride=a.stride(), dtype=a.dtype)
        if isinstance(a, torch.Tensor)
        else a
        for a in captured_args
    )
    kwargs_meta = {
        k: TensorMeta(shape=v.shape, stride=v.stride(), dtype=v.dtype)
        if isinstance(v, torch.Tensor)
        else v
        for k, v in captured_kwargs.items()
    }

    try:
        result = strategy_func(op_overload, args_meta, kwargs_meta)

        expanded_result: list[list[Placement]] = []
        for combo in result:
            expanded_combo: list[Placement] = []
            for p in combo:
                if isinstance(p, _ShardingPlaceholder):
                    expanded_combo.append(Shard(p.dim))
                else:
                    expanded_combo.append(p)
            expanded_result.append(expanded_combo)

        return expanded_result
    except Exception:
        return None


def get_opinfo_by_name(name: str) -> list[opinfo_core.OpInfo]:
    """Find OpInfo entries by exact operator name."""
    matches = [op for op in op_db if op.name == name]
    if matches:
        return matches

    # Suggest alternatives
    candidates = _find_opinfo_candidates(name)
    if candidates:
        suggestions = ", ".join(f'"{c}"' for c in candidates)
        raise ValueError(f'No OpInfo found for "{name}", did you mean: {suggestions}?')
    raise ValueError(
        f'No OpInfo found for "{name}". OpInfo is required as it provides '
        f"sample inputs for the operator."
    )


def _find_opinfo_candidates(name: str) -> list[str]:
    """Find OpInfo names that plausibly match a short/incorrect name."""
    candidates: list[str] = []
    seen: set[str] = set()
    # Match on aten_name (e.g., "relu" -> OpInfo with aten_name="relu")
    for op in op_db:
        if op.aten_name == name and op.name not in seen:
            candidates.append(op.name)
            seen.add(op.name)
    # Suffix match: "relu" matches "nn.functional.relu"
    suffix = "." + name
    for op in op_db:
        if op.name.endswith(suffix) and op.name not in seen:
            candidates.append(op.name)
            seen.add(op.name)
    return candidates


def resolve_op_names(patterns: list[str]) -> list[str]:
    """Resolve user-provided op patterns to exact OpInfo names.

    Supports exact names, comma separation, and glob patterns (e.g.,
    "nn.functional.*"). Short names like "relu" are resolved unambiguously
    or an error is raised with suggestions.
    """
    all_opinfo_names = sorted({op.name for op in op_db})
    resolved: list[str] = []
    seen: set[str] = set()

    for pattern in patterns:
        # Glob pattern
        if "*" in pattern or "?" in pattern:
            matches = fnmatch.filter(all_opinfo_names, pattern)
            if not matches:
                raise ValueError(f'No OpInfo names match pattern "{pattern}".')
            for m in matches:
                if m not in seen:
                    resolved.append(m)
                    seen.add(m)
            continue

        # Exact match
        if pattern in {op.name for op in op_db}:
            if pattern not in seen:
                resolved.append(pattern)
                seen.add(pattern)
            continue

        # Try to resolve shorthand
        candidates = _find_opinfo_candidates(pattern)
        if len(candidates) == 1:
            name = candidates[0]
            if name not in seen:
                resolved.append(name)
                seen.add(name)
        elif len(candidates) > 1:
            suggestions = ", ".join(f'"{c}"' for c in candidates)
            raise ValueError(
                f'"{pattern}" is ambiguous, matching: {suggestions}. '
                f"Use the fully qualified name."
            )
        else:
            raise ValueError(f'No OpInfo found for "{pattern}".')

    return resolved


def _is_tensor_output(result: Any) -> bool:
    """Check if a result is a tensor or list/tuple of tensors."""
    if isinstance(result, torch.Tensor):
        return True
    if isinstance(result, (list, tuple)):
        has_tensor = any(isinstance(t, torch.Tensor) for t in result)
        all_tensor = all(isinstance(t, torch.Tensor) for t in result)
        if has_tensor and not all_tensor:
            raise NotImplementedError(
                f"Mixed tensor/non-tensor tuple outputs are not supported by the "
                f"validator. Got types: {[type(t).__name__ for t in result]}"
            )
        return all_tensor
    return False


def _to_ground_truth(result: Any) -> torch.Tensor | list[torch.Tensor]:
    """Convert an op result to the ground truth format (tensor or list of tensors)."""
    if isinstance(result, torch.Tensor):
        return result
    return list(result)


def _prepare_false_positive_mitigations(
    op: Callable[..., Any],
    sample: SampleInput,
    tensors: list[tuple[str, torch.Tensor]],
) -> _FalsePositiveMitigations:
    """Create negated and non-rounded sample variants for false positive detection."""
    m = _FalsePositiveMitigations()

    try:
        m.negated_sample = create_fully_negated_sample(sample)
        m.negated_tensors = negate_all_tensors(tensors)
        result = _run_op_on_sample(op, m.negated_sample)
        if _is_tensor_output(result):
            m.negated_ground_truth = _to_ground_truth(result)
        else:
            m.negated_sample = None
    except Exception:
        m.negated_sample = None
        m.negated_tensors = None

    if "rounding_mode" not in sample.kwargs:
        return m

    try:
        non_rounded_kwargs = {
            k: v for k, v in sample.kwargs.items() if k != "rounding_mode"
        }
        m.non_rounded_sample = SampleInput(
            sample.input, args=sample.args, kwargs=non_rounded_kwargs
        )
        result = _run_op_on_sample(op, m.non_rounded_sample)
        if not _is_tensor_output(result):
            m.non_rounded_sample = None
        else:
            m.non_rounded_ground_truth = _to_ground_truth(result)
            m.non_rounded_negated_sample = create_fully_negated_sample(
                m.non_rounded_sample
            )
            m.non_rounded_negated_tensors = negate_all_tensors(tensors)
            nr_neg_result = _run_op_on_sample(op, m.non_rounded_negated_sample)
            if _is_tensor_output(nr_neg_result):
                m.non_rounded_negated_ground_truth = _to_ground_truth(nr_neg_result)
            else:
                m.non_rounded_negated_sample = None
    except Exception:
        m.non_rounded_sample = None
        m.non_rounded_ground_truth = None
        m.non_rounded_negated_sample = None

    return m


def _query_dtensor_rules(
    aten_op: OpOverload | None,
    tensors: list[tuple[str, torch.Tensor]],
    captured_args: tuple[Any, ...],
    captured_kwargs: dict[str, Any],
    input_shapes: tuple[tuple[int, ...], ...],
    output_shapes: tuple[tuple[int, ...], ...],
    world_size: int,
    verbose: bool,
) -> set[ComboKey]:
    """Query DTensor's claimed sharding rules via single-dim, op_strategy, or decomp paths.

    TODO: This reimplements strategy resolution logic from ShardingPropagator.
    Refactor ShardingPropagator to expose a public API for querying sharding
    rules given an op and tensor metadata, so this function can be replaced
    with a single call.
    """
    if not aten_op:
        return set()

    num_tensors = len(tensors)
    non_tensor_kwargs = {
        k: v for k, v in captured_kwargs.items() if not isinstance(v, torch.Tensor)
    }
    n_outputs = len(output_shapes)
    propagator = DTensor._op_dispatcher.sharding_propagator
    rules: set[ComboKey] = set()

    if aten_op in propagator.op_single_dim_strategy_funcs:
        strategy_result = query_single_dim_strategy(
            aten_op, captured_args, captured_kwargs
        )
        if strategy_result:
            for combo in strategy_result:
                if len(combo) >= n_outputs + num_tensors:
                    output_plcs = combo[:n_outputs]
                    input_plcs = tuple(combo[n_outputs : n_outputs + num_tensors])
                    rule_key: ComboKey = (
                        tuple(str(p) for p in input_plcs),
                        tuple(str(p) for p in output_plcs),
                    )
                    normalized_rule = normalize_combo_key(
                        rule_key, input_shapes, output_shapes
                    )
                    if not is_fully_replicated(
                        tuple(
                            parse_placement(p) or Replicate()
                            for p in normalized_rule[0]
                        )
                    ):
                        rules.add(normalized_rule)

    elif aten_op in propagator.op_strategy_funcs:
        try:
            mesh = init_device_mesh("cpu", (world_size,))
            # Build OpStrategy objects for each tensor, keyed by identity
            tensor_to_strategy: dict[int, OpStrategy] = {}
            for _, t in tensors:
                input_placements = get_1d_input_placements_for_tensor(
                    t, include_partial=True
                )
                specs = []
                for p in input_placements:
                    spec = DTensorSpec(
                        mesh=mesh,
                        placements=(p,),
                        tensor_meta=TensorMeta(
                            shape=t.shape, stride=t.stride(), dtype=t.dtype
                        ),
                    )
                    specs.append(OpSpec(output_specs=spec, input_specs=tuple()))
                tensor_to_strategy[id(t)] = OpStrategy(specs)
            # Interleave strategies and non-tensor args at original positions
            args_schema = [
                tensor_to_strategy[id(a)] if isinstance(a, torch.Tensor) else a
                for a in captured_args
            ]
            op_schema = OpSchema(aten_op, tuple(args_schema), non_tensor_kwargs)
            strategy_func = propagator.op_strategy_funcs[aten_op]
            output_strategy = strategy_func(op_schema)
            rules |= _extract_rules_from_op_strategy(
                output_strategy, input_shapes, output_shapes
            )
        except Exception as e:
            if verbose:
                print(f"        Error querying op_strategy: {e}")

    else:
        # Decomp-based strategy: only discovers rules reachable from a single
        # seed (Shard(0) on the first input). Rules requiring other input
        # placements (e.g., Shard(1), Partial, or sharding on non-first inputs)
        # will not be found, so this under-reports DTensor's capabilities.
        if DecompShardingStrategy.has_decomp(aten_op):
            try:
                mesh = init_device_mesh("cpu", (world_size,))
                # Interleave DTensorSpec and non-tensor args at original positions
                tensor_idx = 0
                args_schema: list[Any] = []
                for a in captured_args:
                    if isinstance(a, torch.Tensor):
                        # First tensor gets Shard(0) to seed candidate
                        # placement generation in _get_candidate_placements
                        plc = Shard(0) if tensor_idx == 0 else Replicate()
                        spec = DTensorSpec(
                            mesh=mesh,
                            placements=(plc,),
                            tensor_meta=TensorMeta(
                                shape=a.shape, stride=a.stride(), dtype=a.dtype
                            ),
                        )
                        args_schema.append(spec)
                        tensor_idx += 1
                    else:
                        args_schema.append(a)
                op_schema = OpSchema(aten_op, tuple(args_schema), non_tensor_kwargs)
                propagator.decomp_strategy.ensure_schema_info(aten_op)
                output_strategy = propagator.decomp_strategy.propagate_strategy(
                    op_schema,
                )
                if output_strategy is not None:
                    rules |= _extract_rules_from_op_strategy(
                        output_strategy, input_shapes, output_shapes
                    )
            except Exception as e:
                if verbose:
                    print(f"        Error querying decomp strategy: {e}")

    return rules


def _validate_with_mitigations(
    op: Callable[..., Any],
    sample: SampleInput,
    tensors: list[tuple[str, torch.Tensor]],
    input_placements: tuple[Placement, ...],
    output_placements: tuple[Placement, ...],
    ground_truth: torch.Tensor | list[torch.Tensor],
    world_size: int,
    mesh: DeviceMesh,
    mitigations: _FalsePositiveMitigations,
) -> bool | None:
    """Validate a combination, including false positive mitigation re-checks.

    Returns True (valid), False (invalid), or None (untestable).
    """
    combo: PlacementCombination = (input_placements, output_placements)
    is_valid, _ = validate_combination(
        op, sample, tensors, combo, ground_truth, world_size, mesh
    )
    if is_valid is None:
        return None

    # Flipped-mask mitigation: the checkerboard mask that controls offset
    # signs (for P(sum)/P(avg)) or rank ownership (for P(min)/P(max)) is
    # deterministic per tensor_idx. Re-validate with the complementary mask
    # to catch index-returning ops (argmin/argmax) where the result
    # coincidentally matches because the dominant value happens to land on
    # a position where both mask orientations preserve argmin/argmax.
    if is_valid and has_any_partial(input_placements, output_placements):
        is_valid, _ = validate_combination(
            op,
            sample,
            tensors,
            combo,
            ground_truth,
            world_size,
            mesh,
            mask_shift=1,
        )

    if (
        is_valid
        and mitigations.negated_sample
        and has_pmin_pmax(input_placements, output_placements)
    ):
        if mitigations.negated_tensors is None:
            raise AssertionError("negated_tensors is None")
        if mitigations.negated_ground_truth is None:
            raise AssertionError("negated_ground_truth is None")
        is_valid, _ = validate_combination(
            op,
            mitigations.negated_sample,
            mitigations.negated_tensors,
            combo,
            mitigations.negated_ground_truth,
            world_size,
            mesh,
        )

    if (
        is_valid
        and mitigations.non_rounded_sample
        and has_any_partial(input_placements, output_placements)
    ):
        if mitigations.non_rounded_ground_truth is None:
            raise AssertionError("non_rounded_ground_truth is None")
        is_valid, _ = validate_combination(
            op,
            mitigations.non_rounded_sample,
            tensors,
            combo,
            mitigations.non_rounded_ground_truth,
            world_size,
            mesh,
        )

    if (
        is_valid
        and mitigations.non_rounded_negated_sample
        and has_pmin_pmax(input_placements, output_placements)
    ):
        if mitigations.non_rounded_negated_tensors is None:
            raise AssertionError("non_rounded_negated_tensors is None")
        if mitigations.non_rounded_negated_ground_truth is None:
            raise AssertionError("non_rounded_negated_ground_truth is None")
        is_valid, _ = validate_combination(
            op,
            mitigations.non_rounded_negated_sample,
            mitigations.non_rounded_negated_tensors,
            combo,
            mitigations.non_rounded_negated_ground_truth,
            world_size,
            mesh,
        )

    return is_valid


@dataclass
class _AtenFalsePositiveMitigations:
    """Bundle of negated variants for aten-level false positive detection.

    Unlike _FalsePositiveMitigations, this works with captured aten args/kwargs
    directly, without SampleInput. The rounding_mode mitigation is skipped
    since it's an OpInfo-level concept that doesn't appear in captured aten kwargs.
    """

    negated_args: tuple[Any, ...] | None = None
    negated_kwargs: dict[str, Any] | None = None
    negated_ground_truth: torch.Tensor | list[torch.Tensor] | None = None


def _negate_tensors_in_tree(tree: Any) -> Any:
    """Negate all tensors in a pytree structure."""

    def _negate(x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return -x
        return x

    return pytree.tree_map(_negate, tree)


def _prepare_aten_mitigations(
    aten_op: OpOverload,
    captured_args: tuple[Any, ...],
    captured_kwargs: dict[str, Any],
) -> _AtenFalsePositiveMitigations:
    """Create negated variants for aten-level false positive detection."""
    m = _AtenFalsePositiveMitigations()
    try:
        m.negated_args = _negate_tensors_in_tree(captured_args)
        m.negated_kwargs = _negate_tensors_in_tree(captured_kwargs)
        result = aten_op(*m.negated_args, **m.negated_kwargs)
        if _is_tensor_output(result):
            m.negated_ground_truth = _to_ground_truth(result)
        else:
            m.negated_args = None
            m.negated_kwargs = None
    except Exception:
        m.negated_args = None
        m.negated_kwargs = None
    return m


def _validate_aten_with_mitigations(
    aten_op: OpOverload,
    captured_args: tuple[Any, ...],
    captured_kwargs: dict[str, Any],
    input_placements: tuple[Placement, ...],
    output_placements: tuple[Placement, ...],
    ground_truth: torch.Tensor | list[torch.Tensor],
    world_size: int,
    mesh: DeviceMesh,
    mitigations: _AtenFalsePositiveMitigations,
) -> bool | None:
    """Validate an aten-level combination with false positive mitigations.

    Returns True (valid), False (invalid), or None (untestable).
    """
    combo: PlacementCombination = (input_placements, output_placements)
    is_valid, _ = validate_aten_combination(
        aten_op,
        captured_args,
        captured_kwargs,
        ground_truth,
        combo,
        world_size,
        mesh,
    )
    if is_valid is None:
        return None

    if is_valid and has_any_partial(input_placements, output_placements):
        is_valid, _ = validate_aten_combination(
            aten_op,
            captured_args,
            captured_kwargs,
            ground_truth,
            combo,
            world_size,
            mesh,
            mask_shift=1,
        )

    if (
        is_valid
        and mitigations.negated_args is not None
        and has_pmin_pmax(input_placements, output_placements)
    ):
        if mitigations.negated_kwargs is None:
            raise AssertionError("negated_kwargs must not be None")
        if mitigations.negated_ground_truth is None:
            raise AssertionError("negated_ground_truth must not be None")
        is_valid, _ = validate_aten_combination(
            aten_op,
            mitigations.negated_args,
            mitigations.negated_kwargs,
            mitigations.negated_ground_truth,
            combo,
            world_size,
            mesh,
        )

    return is_valid


def _assert_keys_normalized(
    keys: set[ComboKey],
    input_shapes: tuple[tuple[int, ...], ...],
    output_shapes: tuple[tuple[int, ...], ...],
) -> None:
    """Assert all combo keys have trivial shards already normalized to Replicate."""
    for key in keys:
        if key != normalize_combo_key(key, input_shapes, output_shapes):
            raise AssertionError(
                f"Key {key} contains un-normalized trivial shards; "
                f"call normalize_combo_key before _compare_rules"
            )


def _compare_rules(
    ground_truth_valid: set[ComboKey],
    dtensor_rules: set[ComboKey],
    input_shapes: tuple[tuple[int, ...], ...],
    output_shapes: tuple[tuple[int, ...], ...],
    sample_idx: int,
    scalar_args: tuple[Any, ...],
    scalar_kwargs: dict[str, Any],
    aten_op: OpOverload | None,
    variant: str,
    stats: ComparisonStats,
    sample: SampleInput | None = None,
    untestable: set[ComboKey] | None = None,
) -> None:
    """Compare ground truth valid rules against DTensor claimed rules, updating stats."""
    if not dtensor_rules:
        return
    if untestable is None:
        untestable = set()

    _assert_keys_normalized(ground_truth_valid, input_shapes, output_shapes)
    _assert_keys_normalized(dtensor_rules, input_shapes, output_shapes)

    op_str = str(aten_op)
    for combo_key in ground_truth_valid:
        if combo_key in dtensor_rules:
            stats.true_positives += 1
            stats.true_positives_by_op[op_str] = (
                stats.true_positives_by_op.get(op_str, 0) + 1
            )
        elif combo_key not in untestable:
            stats.false_negatives.append(
                Discrepancy(
                    input_placements=combo_key[0],
                    output_placements=combo_key[1],
                    sample_idx=sample_idx,
                    input_shapes=input_shapes,
                    discrepancy_type="false_negative",
                    scalar_args=scalar_args,
                    scalar_kwargs=scalar_kwargs,
                    aten_op=aten_op,
                    variant=variant,
                    sample=sample,
                )
            )

    for combo_key in dtensor_rules:
        if combo_key not in ground_truth_valid and combo_key not in untestable:
            stats.false_positives.append(
                Discrepancy(
                    input_placements=combo_key[0],
                    output_placements=combo_key[1],
                    sample_idx=sample_idx,
                    input_shapes=input_shapes,
                    discrepancy_type="false_positive",
                    scalar_args=scalar_args,
                    scalar_kwargs=scalar_kwargs,
                    aten_op=aten_op,
                    variant=variant,
                    sample=sample,
                )
            )


def _format_op_name(opinfo_name: str) -> str:
    """Format an OpInfo name for display, adding aten. prefix for simple names."""
    if "." not in opinfo_name:
        return f"aten.{opinfo_name}"
    return opinfo_name


def _format_sample_repro(sample: SampleInput, aten_op: OpOverload | None = None) -> str:
    """Format a SampleInput's values for repro output, using schema arg names."""
    # Get argument names from the aten op schema if available.
    # Schema args are ordered: the first is the input, then positional args,
    # then keyword-only args (which appear in sample.kwargs).
    arg_names: list[str] = []
    if aten_op is not None:
        try:
            arg_names = [a.name for a in aten_op._schema.arguments]
        except Exception:
            pass

    parts = []
    # input is the first positional arg
    name = arg_names[0] if arg_names else "input"
    parts.append(f"{name}={sample.input!r}")
    # remaining positional args
    for i, arg in enumerate(sample.args):
        name = arg_names[1 + i] if 1 + i < len(arg_names) else f"args[{i}]"
        parts.append(f"{name}={arg!r}")
    # kwargs — use their actual key names (already named)
    for k, v in sample.kwargs.items():
        parts.append(f"{k}={v!r}")
    return ", ".join(parts)


def _print_discrepancy_section(
    title: str, discrepancies: list[Discrepancy], show_repro: int = 0
) -> None:
    """Print grouped discrepancies for a section (incorrect or missing)."""
    if not discrepancies:
        return
    print(f"\n{title}")
    by_op: dict[str, dict[ComboKey, list[Discrepancy]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for d in discrepancies:
        op_str = str(d.aten_op)
        key = (d.input_placements, d.output_placements)
        by_op[op_str][key].append(d)

    for op_str in sorted(by_op.keys()):
        print(f"\n  [{op_str}]")
        for (inp, out), discs in sorted(by_op[op_str].items(), key=str):
            inp_str = ", ".join(inp)
            out_str = out[0] if len(out) == 1 else "(" + ", ".join(out) + ")"
            print(f"    {inp_str} -> {out_str}")
            if show_repro:
                limit = len(discs) if show_repro < 0 else show_repro
                for d in discs[:limit]:
                    if d.sample is not None:
                        print(
                            f"      Repro: {_format_sample_repro(d.sample, d.aten_op)}"
                        )
                if len(discs) > limit:
                    print(f"      ... and {len(discs) - limit} more")


def _print_comparison_summary(
    stats: ComparisonStats,
    show_repro: int = 0,
) -> None:
    """Print discrepancy details for an operator."""
    # Per aten op variant breakdown
    fp_by_op: dict[str, set[ComboKey]] = defaultdict(set)
    for d in stats.false_positives:
        op_str = str(d.aten_op)
        fp_by_op[op_str].add((d.input_placements, d.output_placements))
    fn_by_op: dict[str, set[ComboKey]] = defaultdict(set)
    for d in stats.false_negatives:
        op_str = str(d.aten_op)
        fn_by_op[op_str].add((d.input_placements, d.output_placements))

    all_ops = sorted(set(stats.true_positives_by_op) | set(fp_by_op) | set(fn_by_op))
    if len(all_ops) > 1:
        for op_str in all_ops:
            tp = stats.true_positives_by_op.get(op_str, 0)
            fp = len(fp_by_op.get(op_str, set()))
            fn = len(fn_by_op.get(op_str, set()))
            print(f"  {op_str}: {tp} correct, {fp} incorrect, {fn} missing")

    _print_discrepancy_section(
        "Incorrect (has rule but ground truth invalid)",
        stats.false_positives,
        show_repro,
    )
    _print_discrepancy_section(
        "Possibly missing (valid in ground truth but no DTensor rule)",
        stats.false_negatives,
        show_repro,
    )


def _has_dtensor_support(aten_op: OpOverload) -> bool:
    """Check if an aten op has any DTensor sharding strategy registered."""
    propagator = DTensor._op_dispatcher.sharding_propagator
    if aten_op in propagator.op_single_dim_strategy_funcs:
        return True
    if aten_op in propagator.op_strategy_funcs:
        return True
    return DecompShardingStrategy.has_decomp(aten_op)


def _discover_aten_op(
    opinfos: list[opinfo_core.OpInfo],
    device: str,
    dtype: torch.dtype,
) -> OpOverload | None:
    """Discover the aten op dispatched by the first valid sample."""
    for opinfo in opinfos:
        try:
            samples = list(opinfo.sample_inputs(device, dtype))
        except Exception:
            continue
        for sample in samples:
            tensors = extract_tensors_from_sample(sample)
            if not tensors or any(0 in t.shape for _, t in tensors):
                continue
            capture = get_aten_op_for_sample(opinfo.op, sample, opinfo.name)
            aten_op = capture.best_match
            if aten_op is not None:
                return aten_op
    return None


def _check_ground_truth(
    result: Any,
) -> torch.Tensor | list[torch.Tensor] | None:
    """Validate an op result is suitable as ground truth.

    Returns the ground truth tensor(s) or None if the result should be skipped.
    """
    if isinstance(result, (list, tuple)):
        if not all(isinstance(t, torch.Tensor) for t in result):
            return None
        gt = list(result)
    elif isinstance(result, torch.Tensor):
        gt = result
    else:
        return None

    first_gt = gt[0] if isinstance(gt, list) else gt
    if first_gt.numel() == 0:
        return None
    if (first_gt == 0).all():
        return None
    if first_gt.isnan().all():
        return None
    return gt


def _validate_aten_op_for_sample(
    aten_op: OpOverload,
    captured_args: tuple[Any, ...],
    captured_kwargs: dict[str, Any],
    ground_truth: torch.Tensor | list[torch.Tensor],
    world_size: int,
    incorrect_only: bool,
    verbose: bool,
    sample_idx: int,
    variant: str,
    stats: ComparisonStats,
    sample: SampleInput | None = None,
) -> tuple[int, int]:
    """Validate a single aten op with captured args against ground truth.

    Shared logic used by both default (1:1) and allow_composite modes in
    compare_operator. Returns (samples_counted, combinations_counted).
    """
    tensors = extract_tensors_from_args(captured_args, captured_kwargs)
    if not tensors:
        return 0, 0
    if any(0 in t.shape for _, t in tensors):
        return 0, 0

    input_shapes = tuple(t.shape for _, t in tensors)
    gt_list = ground_truth if isinstance(ground_truth, list) else [ground_truth]
    output_shapes = tuple(tuple(gt.shape) for gt in gt_list)
    n_outputs = len(gt_list)
    first_gt = gt_list[0]

    scalar_args = tuple(a for a in captured_args if not isinstance(a, torch.Tensor))
    scalar_kwargs = {
        k: v for k, v in captured_kwargs.items() if not isinstance(v, torch.Tensor)
    }

    mitigations = _prepare_aten_mitigations(aten_op, captured_args, captured_kwargs)

    input_placement_options = [
        get_1d_input_placements_for_tensor(t, include_partial=True) for _, t in tensors
    ]
    output_placement_options = get_1d_output_placements_for_tensor(first_gt)

    dtensor_rules = _query_dtensor_rules(
        aten_op,
        tensors,
        captured_args,
        captured_kwargs,
        input_shapes,
        output_shapes,
        world_size,
        verbose,
    )

    ground_truth_valid: set[ComboKey] = set()
    total_combinations = 0

    tensor_device = tensors[0][1].device.type if tensors else "cpu"
    with LocalTensorMode(frozenset(range(world_size))):
        mesh = init_device_mesh(tensor_device, (world_size,))

        if incorrect_only:
            combinations_to_test = []
            for combo_key in dtensor_rules:
                input_plc_strs, output_plc_strs = combo_key
                input_plcs_list: list[Placement] = []
                all_valid = True
                for s in input_plc_strs:
                    p = parse_placement(s)
                    if p is None:
                        all_valid = False
                        break
                    input_plcs_list.append(p)
                output_plcs_list: list[Placement] = []
                for s in output_plc_strs:
                    p = parse_placement(s)
                    if p is None:
                        all_valid = False
                        break
                    output_plcs_list.append(p)
                if not all_valid:
                    continue
                combinations_to_test.append(
                    (
                        tuple(input_plcs_list),
                        tuple(output_plcs_list),
                        combo_key,
                    )
                )
        else:
            combinations_to_test = []
            for input_placements in itertools.product(*input_placement_options):
                if is_fully_replicated(input_placements):
                    continue
                for output_placement in output_placement_options:
                    output_placements = tuple(
                        output_placement for _ in range(n_outputs)
                    )
                    combo_key = (
                        tuple(str(p) for p in input_placements),
                        tuple(str(p) for p in output_placements),
                    )
                    combinations_to_test.append(
                        (input_placements, output_placements, combo_key)
                    )

        untestable: set[ComboKey] = set()

        for (
            input_placements,
            output_placements,
            combo_key,
        ) in combinations_to_test:
            total_combinations += 1
            is_valid = _validate_aten_with_mitigations(
                aten_op,
                captured_args,
                captured_kwargs,
                input_placements,
                output_placements,
                ground_truth,
                world_size,
                mesh,
                mitigations,
            )

            if is_valid is None:
                normalized_key = normalize_combo_key(
                    combo_key, input_shapes, output_shapes
                )
                untestable.add(normalized_key)
            elif is_valid:
                normalized_key = normalize_combo_key(
                    combo_key, input_shapes, output_shapes
                )
                if not is_fully_replicated(
                    tuple(parse_placement(p) or Replicate() for p in normalized_key[0])
                ):
                    ground_truth_valid.add(normalized_key)

    _compare_rules(
        ground_truth_valid,
        dtensor_rules,
        input_shapes,
        output_shapes,
        sample_idx,
        scalar_args,
        scalar_kwargs,
        aten_op,
        variant,
        stats,
        sample,
        untestable,
    )

    if verbose:
        print(f"      Sample {sample_idx} [{aten_op}]: shapes={input_shapes}")
        print(f"        Ground truth valid: {len(ground_truth_valid)}")
        print(f"        DTensor rules: {len(dtensor_rules)}")

    return 1, total_combinations


def compare_operator(
    op_name: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    world_size: int = 2,
    max_samples: int | None = None,
    verbose: bool = False,
    incorrect_only: bool = False,
    allow_composite: bool = False,
) -> ComparisonStats:
    """
    Compare DTensor's sharding rules against ground truth for an operator.

    Args:
        op_name: Name of the operator to test
        device: Device to run on
        dtype: Data type for tensors
        world_size: Simulated world size
        max_samples: Maximum number of samples to test per OpInfo
        verbose: Print detailed output
        incorrect_only: If True, only test DTensor's claimed rules for correctness.
            Skips search for missing rules (much faster).
        allow_composite: If True, validate each supported aten op individually for
            samples that decompose into multiple aten calls. Default (False)
            skips samples where the OpInfo doesn't map 1:1 to a single aten op.
    """
    if op_name in SKIP_OPS:
        return ComparisonStats()

    opinfos = get_opinfo_by_name(op_name)

    stats = ComparisonStats()

    if not allow_composite:
        aten_op = _discover_aten_op(opinfos, device, dtype)
        if aten_op is None or not _has_dtensor_support(aten_op):
            if verbose:
                print(f"  ATEN_OP_MAP: {op_name} -> {aten_op} [no_support]")
            stats.no_dtensor_support = True
            return stats
        if verbose:
            print(f"  ATEN_OP_MAP: {op_name} -> {aten_op} [supported]")

    total_samples = 0
    total_combinations = 0
    skip_reasons: dict[str, int] = defaultdict(int)

    for opinfo in opinfos:
        variant = opinfo.variant_test_name
        if variant and verbose:
            print(f"\n  OpInfo variant: {variant}")

        op = opinfo.op

        try:
            samples = list(opinfo.sample_inputs(device, dtype))
        except Exception as e:
            if verbose:
                print(f"    Error generating samples: {e}")
            continue

        if max_samples:
            samples = samples[:max_samples]

        for sample_idx, sample in enumerate(samples):
            # Check that SampleInput has tensor inputs and no zero-sized tensors
            sample_tensors = extract_tensors_from_sample(sample)
            if len(sample_tensors) == 0:
                skip_reasons["no tensor inputs"] += 1
                continue
            if any(0 in t.shape for _, t in sample_tensors):
                skip_reasons["zero-sized tensor"] += 1
                continue

            # Capture all aten ops dispatched for this sample
            capture = get_aten_op_for_sample(op, sample, opinfo.name)
            if capture.best_match is None:
                skip_reasons["no aten op captured"] += 1
                continue

            # Count supported aten ops in the capture
            supported_ops = [
                (func, args, kwargs, result)
                for func, args, kwargs, result in capture.all_ops
                if _has_dtensor_support(func)
            ]
            num_supported = len(supported_ops)

            if allow_composite:
                # Validate each supported aten op individually
                if num_supported == 0:
                    skip_reasons["no supported aten ops"] += 1
                    continue

                for func, args, kwargs, result in supported_ops:
                    gt = _check_ground_truth(result)
                    if gt is None:
                        skip_reasons["non-tensor/degenerate aten output"] += 1
                        continue
                    n_samples, n_combos = _validate_aten_op_for_sample(
                        func,
                        args,
                        kwargs,
                        gt,
                        world_size,
                        incorrect_only,
                        verbose,
                        sample_idx,
                        variant,
                        stats,
                        sample,
                    )
                    total_samples += n_samples
                    total_combinations += n_combos
            else:
                # Default: only validate samples with a single supported aten op
                if num_supported != 1:
                    skip_reasons["non-1:1 aten mapping"] += 1
                    continue

                func, args, kwargs, result = supported_ops[0]
                gt = _check_ground_truth(result)
                if gt is None:
                    skip_reasons["non-tensor/degenerate aten output"] += 1
                    continue

                n_samples, n_combos = _validate_aten_op_for_sample(
                    func,
                    args,
                    kwargs,
                    gt,
                    world_size,
                    incorrect_only,
                    verbose,
                    sample_idx,
                    variant,
                    stats,
                    sample,
                )
                total_samples += n_samples
                total_combinations += n_combos

    stats.total_samples = total_samples
    stats.total_combinations = total_combinations
    stats.skip_reasons = dict(skip_reasons)

    # In allow_composite mode, check DTensor support after processing
    if allow_composite and total_samples == 0 and not skip_reasons:
        stats.no_dtensor_support = True

    return stats


def get_registered_op_names() -> list[str]:
    """Get all op names that have DTensor sharding rules and also have OpInfo.

    Returns OpInfo names (which may differ from aten base names, e.g.,
    "nn.functional.relu" instead of "relu").
    """

    propagator = DTensor._op_dispatcher.sharding_propagator

    # Get all registered aten ops
    all_registered = set(propagator.op_single_dim_strategy_funcs.keys()) | set(
        propagator.op_strategy_funcs.keys()
    )

    # Extract base names (aten.mul.Tensor -> mul)
    base_names = set()
    for op in all_registered:
        parts = str(op).split(".")
        if len(parts) >= 2:
            base_names.add(parts[1])

    # Build mappings from OpInfo: both by name and by aten_name
    opinfo_by_name = {}
    opinfo_by_aten_name: dict[str, list[str]] = {}
    for op in op_db:
        opinfo_by_name[op.name] = True
        opinfo_by_aten_name.setdefault(op.aten_name, []).append(op.name)

    result = set()
    for base_name in base_names:
        if base_name in opinfo_by_name:
            # Direct match (e.g., "add" -> OpInfo named "add")
            result.add(base_name)
        elif base_name in opinfo_by_aten_name:
            # Match via aten_name (e.g., "relu" -> OpInfo "nn.functional.relu"
            # which has aten_name="relu")
            result.update(opinfo_by_aten_name[base_name])

    return sorted(result)


def report_registrations(verbose: bool = False) -> None:
    """Report the number (and optionally full list) of ops by registration method.

    Directly registered categories (mutually exclusive):
      - rule: ops registered via register_prop_rule
      - op_strategy: ops registered via register_op_strategy
      - single_dim_strategy: ops registered via register_single_dim_strategy

    Ops not in any of the above may still be supported at runtime via
    DecompShardingStrategy, which traces through the op's decomposition and
    propagates placements through the decomposed sub-ops.  Whether this
    actually works depends on every sub-op having a registered strategy.
    We report the decomposition_table entries as a separate (untested) count.
    """
    from torch._decomp import decomposition_table

    propagator = DTensor._op_dispatcher.sharding_propagator

    rule_ops = sorted(propagator.op_to_rules.keys(), key=str)
    strategy_ops = sorted(propagator.op_strategy_funcs.keys(), key=str)
    single_dim_ops = sorted(propagator.op_single_dim_strategy_funcs.keys(), key=str)

    directly_registered = (
        set(propagator.op_to_rules.keys())
        | set(propagator.op_strategy_funcs.keys())
        | set(propagator.op_single_dim_strategy_funcs.keys())
    )

    # Ops from the explicit decomposition table that aren't directly registered.
    # These *may* work via DecompShardingStrategy if all their sub-ops are
    # supported, but we can't verify that without tracing each one.
    decomp_only_ops = sorted(
        (op for op in decomposition_table if op not in directly_registered),
        key=str,
    )

    print("=" * 70)
    print("DTensor operator registration report")
    print("=" * 70)

    print("\nDirectly registered:")
    print(f"  rule (register_prop_rule):            {len(rule_ops):>4}")
    print(f"  op_strategy (register_op_strategy):   {len(strategy_ops):>4}")
    print(f"  single_dim_strategy:                  {len(single_dim_ops):>4}")
    print(f"  total:                                {len(directly_registered):>4}")

    print(f"\nDecomposition table (not directly registered): {len(decomp_only_ops)}")
    print(
        "  These ops have entries in torch._decomp.decomposition_table but no\n"
        "  direct DTensor strategy. They may work at runtime via\n"
        "  DecompShardingStrategy if all decomposed sub-ops are supported.\n"
        "  Additional ops beyond this count may also be reachable via CIA\n"
        "  (CompositeImplicitAutograd) decompositions."
    )

    if verbose:

        def _print_ops(label: str, ops: list) -> None:
            print(f"\n{label} ({len(ops)}):")
            for op in ops:
                print(f"  {op}")

        _print_ops("rule", rule_ops)
        _print_ops("op_strategy", strategy_ops)
        _print_ops("single_dim_strategy", single_dim_ops)
        _print_ops("decomp table (not directly registered)", decomp_only_ops)


if __name__ == "__main__":
    # Override common size variables to ensure even sharding across world_size=2.
    # These are process-global mutations, but this is a CLI entry point so the
    # process exits after running.
    opinfo_core.L = 24  # pyrefly: ignore[bad-assignment]
    opinfo_core.M = 12  # pyrefly: ignore[bad-assignment]
    opinfo_core.S = 4  # pyrefly: ignore[bad-assignment]
    opinfo_core.XS = 2  # pyrefly: ignore[bad-assignment]

    common_ops.L = 24  # pyrefly: ignore[bad-assignment]
    common_ops.M = 12  # pyrefly: ignore[bad-assignment]
    common_ops.S = 4  # pyrefly: ignore[bad-assignment]
    common_ops.XS = 2  # pyrefly: ignore[bad-assignment]

    parser = argparse.ArgumentParser(
        description="Compare DTensor rules against ground truth"
    )
    parser.add_argument(
        "--op",
        default=None,
        help="Operator name(s) to compare (comma-separated, supports glob "
        'patterns, e.g., "relu,add" or "nn.functional.*")',
    )
    parser.add_argument(
        "--all-registered",
        action="store_true",
        help="Test all ops with DTensor sharding rules registered",
    )
    parser.add_argument(
        "--incorrect-only",
        action="store_true",
        help="Only test DTensor's claimed rules (faster, skips missing detection)",
    )
    parser.add_argument(
        "--allow-composite",
        action="store_true",
        help="Validate each supported aten op individually for decomposed ops "
        "(default skips non-1:1 aten mappings)",
    )
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--dtype", default="float32", help="Dtype to use")
    parser.add_argument(
        "--world-size", type=int, default=2, help="Simulated world size"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Max samples to test"
    )
    parser.add_argument(
        "--show-repro",
        nargs="?",
        const=1,
        type=int,
        default=0,
        metavar="N",
        help="Show N sample repros per rule (default 1 if flag given, -1 for all)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Show registration statistics (op counts per registration method)",
    )
    parser.add_argument(
        "--report-full",
        action="store_true",
        help="Show full op lists with registration statistics (implies --report)",
    )
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(args.dtype, torch.float32)

    dist.init_process_group("fake", rank=0, world_size=args.world_size)

    if args.report or args.report_full:
        report_registrations(verbose=args.report_full)
        if not args.all_registered and not args.op:
            dist.destroy_process_group()
            sys.exit(0)
        print()

    try:
        if args.all_registered:
            op_names = get_registered_op_names()
        elif args.op:
            patterns = [p.strip() for p in args.op.split(",")]
            op_names = resolve_op_names(patterns)
        else:
            op_names = ["add"]

        # Preamble
        display_names = [_format_op_name(n) for n in op_names]
        print(f"Testing ops: {', '.join(display_names)}")
        if args.allow_composite:
            print(
                "Mode: allow-composite (validates each aten op in decomposed samples)"
            )
        if args.incorrect_only:
            print("Mode: incorrect-only (fast)")
        print(f"Device: {args.device}, Dtype: {dtype}, World size: {args.world_size}")

        op_results: list[tuple[str, ComparisonStats, float]] = []

        for i, op_name in enumerate(op_names):
            display = _format_op_name(op_name)

            if op_name in SKIP_OPS:
                print(
                    f"\n[{i + 1}/{len(op_names)}] {display}"
                    f" — skipped: {SKIP_OPS[op_name]}"
                )
                continue

            try:
                op_start = time.time()
                stats = compare_operator(
                    op_name,
                    args.device,
                    dtype,
                    args.world_size,
                    args.max_samples,
                    verbose=True,
                    incorrect_only=args.incorrect_only,
                    allow_composite=args.allow_composite,
                )
                elapsed = time.time() - op_start

                if stats.no_dtensor_support:
                    print(
                        f"\n[{i + 1}/{len(op_names)}] {display}"
                        f" — skipped: no DTensor support"
                    )
                    continue

                if stats.total_samples == 0 and stats.skip_reasons:
                    reasons = ", ".join(
                        f"{r} ({n})" for r, n in stats.skip_reasons.items()
                    )
                    print(f"\n[{i + 1}/{len(op_names)}] {display} — skipped: {reasons}")
                    continue

                op_results.append((op_name, stats, elapsed))

                skipped = sum(stats.skip_reasons.values())
                skipped_str = f" ({skipped} skipped)" if skipped else ""
                print(
                    f"\n[{i + 1}/{len(op_names)}] {display}"
                    f" — Samples: {stats.total_samples}{skipped_str},"
                    f" Combinations: {stats.total_combinations}"
                )
                print("-" * 70)
                _print_comparison_summary(stats, show_repro=args.show_repro)
            except Exception as e:
                print(f"\n[{i + 1}/{len(op_names)}] {display} — Error: {e}")

        if len(op_results) >= 1:
            # Summary table
            print("\n" + "=" * 70)
            print("Summary")
            print("=" * 70)

            formatted_results = [
                (_format_op_name(name), stats, elapsed)
                for name, stats, elapsed in op_results
            ]

            col_widths = {
                "op": max(len(name) for name, _, _ in formatted_results),
                "tp": 7,
                "fp": 9,
                "fn": 7,
                "time": 6,
            }
            col_widths["op"] = max(col_widths["op"], 2)  # min width

            header = (
                f"{'Op':<{col_widths['op']}}  "
                f"{'Correct':>{col_widths['tp']}}  "
                f"{'Incorrect':>{col_widths['fp']}}  "
                f"{'Missing':>{col_widths['fn']}}  "
                f"{'Time':>{col_widths['time']}}"
            )
            print(header)
            print("-" * len(header))

            total_tp = 0
            total_fp = 0
            total_fn = 0
            total_time = 0.0

            for name, stats, elapsed in formatted_results:
                fp_rules = {
                    (d.input_placements, d.output_placements)
                    for d in stats.false_positives
                }
                fn_rules = {
                    (d.input_placements, d.output_placements)
                    for d in stats.false_negatives
                }
                n_fp = len(fp_rules)
                n_fn = len(fn_rules)
                total_tp += stats.true_positives
                total_fp += n_fp
                total_fn += n_fn
                total_time += elapsed

                fp_str = str(n_fp) if n_fp else "0"
                fn_str = str(n_fn) if n_fn else "0"

                print(
                    f"{name:<{col_widths['op']}}  "
                    f"{stats.true_positives:>{col_widths['tp']}}  "
                    f"{fp_str:>{col_widths['fp']}}  "
                    f"{fn_str:>{col_widths['fn']}}  "
                    f"{elapsed:>{col_widths['time']}.1f}s"
                )

            print("-" * len(header))
            fp_str = str(total_fp) if total_fp else "0"
            fn_str = str(total_fn) if total_fn else "0"
            print(
                f"{'Total':<{col_widths['op']}}  "
                f"{total_tp:>{col_widths['tp']}}  "
                f"{fp_str:>{col_widths['fp']}}  "
                f"{fn_str:>{col_widths['fn']}}  "
                f"{total_time:>{col_widths['time']}.1f}s"
            )
    finally:
        dist.destroy_process_group()
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates
from collections.abc import Callable

import torch
from torch._ops import OpOverload
from torch.distributed.tensor._dtensor_spec import TensorMeta
from torch.distributed.tensor._op_schema import ArgsType, KwargsType, RuntimeSchemaInfo
from torch.distributed.tensor._ops.single_dim_strategy import (
    _ShardingPlaceholder,
    register_single_dim_strategy,
)
from torch.distributed.tensor._ops.utils import infer_broadcast_dims_map
from torch.distributed.tensor.placement_types import Partial, Placement, Replicate


aten = torch.ops.aten
prims = torch.ops.prims

# Linear pointwise ops, split by linearity type.
unary_linear_ops = [aten.to.dtype]


def _common_pointwise_single_dim_strategy(
    partial_extra_rules: list[list[Placement | _ShardingPlaceholder]] | None = None,
) -> Callable[
    [OpOverload, ArgsType, KwargsType], list[list[Placement | _ShardingPlaceholder]]
]:
    """Factory for single-dim strategies that add partial placement rules.

    Returns strategies shaped [output, *args] only.  Tensor kwarg placements
    (e.g. ``out``, ``lr``) are appended by the wrapper in
    ``_register_single_dim_pointwise``.
    """

    def strategy(
        op: OpOverload,
        args_schema: ArgsType,
        kwargs_schema: KwargsType,
    ) -> list[list[Placement | _ShardingPlaceholder]]:
        tensor_arg_metas: list[TensorMeta] = [
            arg for arg in args_schema if isinstance(arg, TensorMeta)
        ]
        common_shape = torch.broadcast_shapes(
            *[arg.shape for arg in args_schema if isinstance(arg, TensorMeta)]
        )
        # For multi-output ops (e.g. frexp), all outputs share the same
        # pointwise sharding, so replicate the output placement.
        num_outputs = sum(1 for r in op._schema.returns if "Tensor" in str(r.type))
        placements: list[list[Placement | _ShardingPlaceholder]] = []
        for i in range(len(common_shape)):
            shard_placements: list[Placement | _ShardingPlaceholder] = [
                _ShardingPlaceholder(i)
            ] * num_outputs
            for arg in tensor_arg_metas:
                common_dim_to_arg_dim = infer_broadcast_dims_map(
                    common_shape, arg.shape
                )
                # If the output shard dim maps to an input dim, shard that
                # input dim; otherwise it was broadcast, so replicate.
                if common_dim_to_arg_dim[i] >= 0:
                    shard_placements.append(
                        _ShardingPlaceholder(common_dim_to_arg_dim[i])
                    )
                else:
                    shard_placements.append(Replicate())
            placements.append(shard_placements)
        if partial_extra_rules:
            n_tensors = len(tensor_arg_metas)
            expected_len = num_outputs + n_tensors
            for rule in partial_extra_rules:
                # Filter rather than assert: some ops (e.g. mul.Tensor) mix
                # unary rules (len 2, for scalar promotion) and binary rules
                # (len 3, for tensor-tensor), so mismatched lengths are expected.
                # see _MUL_RULES to see how _UNARY_LINEAR_RULES handles the
                # scalar promotion case
                if len(rule) == expected_len:
                    placements.append(rule)
        return placements

    return strategy


def _is_list_op(op: OpOverload) -> bool:
    """Returns True if op is a foreach, amp_foreach, or fused op."""
    name = op.name()
    return name.startswith(("aten::_foreach_", "aten::_amp_foreach_", "aten::_fused_"))


# The state_steps arg of fused adam / adamw is a Replicate scalar tensor, which will be put on
# the compute_mesh of an op across all parameter groups, even when not all parameter groups
# are on the same device mesh. This idx will help avoid hitting exceptions or unnecessary
# redistribute during sharding propagation.
_FUSED_OP_SCALAR_IDX = 5

# Ops registered with extra Partial rules; populated by _register_single_dim_pointwise
# when partial_extra_rules is not None, to avoid double-registration from tag discovery.
_specially_registered_ops: set[OpOverload] = set()


def _register_single_dim_pointwise(
    op: OpOverload,
    partial_extra_rules: list[list[Placement]] | None = None,
    static_argnum: int = 0,
) -> None:
    if partial_extra_rules is not None:
        _specially_registered_ops.add(op)
    inner_fn = _common_pointwise_single_dim_strategy(
        partial_extra_rules=partial_extra_rules  # pyrefly: ignore[bad-argument-type]
    )

    # Wrap to append tensor kwarg placements in schema declaration order.
    # out = output placement (s[0]); everything else (e.g. lr) = Replicate.
    # TODO: move kwargs handling upstream if this works
    def strategy_fn(
        op: OpOverload,
        args: ArgsType,
        kwargs: KwargsType,
        _fn: Callable = inner_fn,
    ) -> list[list[Placement | _ShardingPlaceholder]]:
        strategies = _fn(op, args, kwargs)
        kw_names = [k for k, v in kwargs.items() if isinstance(v, TensorMeta)]
        if not kw_names:
            return strategies
        return [
            s + [s[0] if name == "out" else Replicate() for name in kw_names]
            for s in strategies
        ]

    if _is_list_op(op):
        schema_info = RuntimeSchemaInfo(needs_pytree=True)
    else:
        schema_info = RuntimeSchemaInfo(static_argnum, static_kwargkey=["out"])
    # Fused ops (e.g. _fused_adam_) have state_steps on a potentially different
    # mesh; see the note in expand_to_full_mesh_op_strategy for details.
    different_mesh_args: list[int] | None = None
    if op.name().startswith("aten::_fused_"):
        different_mesh_args = [_FUSED_OP_SCALAR_IDX]
    register_single_dim_strategy(
        op,
        schema_info=schema_info,
        allow_uneven_sharding=True,
        allow_unbacked_sharding=True,
        different_mesh_args=different_mesh_args,
    )(strategy_fn)


_UNARY_LINEAR_RULES: list[list[Placement]] = [
    [Partial("sum"), Partial("sum")],
    [Partial("avg"), Partial("avg")],
]

binary_additive_ops = [
    aten.add.Tensor,
    aten.add_.Tensor,
    aten.add.out,
    aten.sub.Tensor,
    aten.sub_.Tensor,
    aten.sub.out,
    # foreach variants
    aten._foreach_add.List,
    aten._foreach_add_.List,
    aten._foreach_sub.List,
    aten._foreach_sub_.List,
]

_BINARY_ADDITIVE_RULES: list[list[Placement]] = [
    [Partial("sum"), Partial("sum"), Partial("sum")],
    [Partial("avg"), Partial("avg"), Partial("avg")],
    # P(x), R -> P(x): adding/subtracting a replicated value preserves partial types
    # avg, max, min. sum would result in R being added n times, n = num_ranks
    # (the replicated value is constant across ranks, so reduce order is unaffected)
    [Partial("avg"), Partial("avg"), Replicate()],
    [Partial("max"), Partial("max"), Replicate()],
    [Partial("min"), Partial("min"), Replicate()],
    # R, P(avg) -> P(avg): avg is linear so this holds for any alpha
    # (R, P(max/min) excluded: negative alpha would flip the ordering)
    [Partial("avg"), Replicate(), Partial("avg")],
]

for op in binary_additive_ops:
    _register_single_dim_pointwise(op, _BINARY_ADDITIVE_RULES)

# mul: partials propagate through either arg. div: only through numerator.
binary_mul_ops = [
    aten.mul.Tensor,
    aten.mul_.Tensor,
    aten.mul.out,
    # foreach variants
    aten._foreach_mul.List,
    aten._foreach_mul_.List,
    aten._foreach_mul.Tensor,
    aten._foreach_mul_.Tensor,
]
binary_div_ops = [
    aten.div.Tensor,
    aten.div_.Tensor,
    aten.div.out,
    # foreach variants
    aten._foreach_div.List,
    aten._foreach_div_.List,
    aten._foreach_div.Tensor,
    aten._foreach_div_.Tensor,
]

# _UNARY_LINEAR_RULES handles the scalar promotion case: Python's __mul__/__truediv__
# promote scalars to 0-dim tensors, so aten.mul.Scalar dispatches as aten.mul.Tensor
# with n_tensors=1, matching the length-2 unary rules.
_MUL_RULES: list[list[Placement]] = [
    [Partial("sum"), Partial("sum"), Replicate()],
    [Partial("avg"), Partial("avg"), Replicate()],
    [Partial("sum"), Replicate(), Partial("sum")],
    [Partial("avg"), Replicate(), Partial("avg")],
]

_DIV_RULES: list[list[Placement]] = [
    [Partial("sum"), Partial("sum"), Replicate()],
    [Partial("avg"), Partial("avg"), Replicate()],
]

for op in binary_mul_ops:
    _register_single_dim_pointwise(op, _UNARY_LINEAR_RULES + _MUL_RULES)

for op in binary_div_ops:
    _register_single_dim_pointwise(op, _UNARY_LINEAR_RULES + _DIV_RULES)

scalar_linear_ops = [
    aten.div.Scalar,
    aten.div_.Scalar,
    aten.mul.Scalar,
    aten.mul_.Scalar,
    # foreach variants
    aten._foreach_div.Scalar,
    aten._foreach_div_.Scalar,
    aten._foreach_mul.Scalar,
    aten._foreach_mul_.Scalar,
    aten._foreach_div.ScalarList,
    aten._foreach_div_.ScalarList,
    aten._foreach_mul.ScalarList,
    aten._foreach_mul_.ScalarList,
]

for op in scalar_linear_ops:
    _register_single_dim_pointwise(op, _UNARY_LINEAR_RULES, static_argnum=1)


# Non-decreasing unary ops: f(max(a,b)) = max(f(a),f(b)).
# Only ops that are non-decreasing on their ENTIRE domain belong here.
# Ops with restricted domains (e.g. log on (0,∞), asin on [-1,1]) do NOT qualify
# because P(max) offsets can push inputs outside the valid domain.
non_decreasing_unary_ops = [
    aten.asinh.default,
    aten.asinh_.default,
    aten.asinh.out,
    aten.atan.default,
    aten.atan_.default,
    aten.atan.out,
    aten.ceil.default,
    aten.ceil_.default,
    aten.ceil.out,
    aten.deg2rad.default,
    aten.deg2rad_.default,
    aten.deg2rad.out,
    aten.erf.default,
    aten.erf_.default,
    aten.erf.out,
    aten.exp.default,
    aten.exp_.default,
    aten.exp.out,
    aten.exp2.default,
    aten.exp2_.default,
    aten.exp2.out,
    aten.expm1.default,
    aten.expm1_.default,
    aten.expm1.out,
    aten.floor.default,
    aten.floor_.default,
    aten.floor.out,
    aten.rad2deg.default,
    aten.rad2deg_.default,
    aten.rad2deg.out,
    aten.relu.default,
    aten.relu_.default,
    aten.round.decimals,
    aten.round.default,
    aten.round_.decimals,
    aten.round_.default,
    aten.round.decimals_out,
    aten.round.out,
    aten.sgn.default,
    aten.sgn_.default,
    aten.sgn.out,
    aten.sigmoid.default,
    aten.sigmoid_.default,
    aten.sigmoid.out,
    aten.sign.default,
    aten.sign_.default,
    aten.sign.out,
    aten.sinh.default,
    aten.sinh_.default,
    aten.sinh.out,
    aten.tanh.default,
    aten.tanh_.default,
    aten.tanh.out,
    aten.trunc.default,
    aten.trunc_.default,
    aten.trunc.out,
    # nan_to_num is non-decreasing on its entire domain (including nan/inf):
    # it maps -inf→min, nan→0, inf→max, and is identity elsewhere.
    aten.nan_to_num.default,
    aten.nan_to_num_.default,
    aten.nan_to_num.out,
    # hardshrink: x if |x|>lambd else 0. Non-decreasing on entire domain.
    aten.hardshrink.default,
    # I1(x) is monotonically non-decreasing for all real x.
    aten.special_modified_bessel_i1.default,
    # threshold(x, t, v): x if x > t else v. Non-decreasing for v <= t (the
    # common case, including the default v=0, t=0).
    aten.threshold.default,
    # foreach variants
    aten._foreach_exp.default,
    aten._foreach_exp_.default,
    aten._foreach_clamp_max_.Scalar,
    aten._foreach_clamp_min_.Scalar,
]

_NON_DECREASING_RULES: list[list[Placement]] = [
    [Partial("max"), Partial("max")],
    [Partial("min"), Partial("min")],
]

for op in non_decreasing_unary_ops:
    _register_single_dim_pointwise(op, _NON_DECREASING_RULES)

# Non-increasing unary ops: f(max(a,b)) = min(f(a),f(b)).
# Note: acos excluded due to domain constraints [-1,1] causing validation failures
non_increasing_unary_ops: list[OpOverload] = [
    aten.erfc.default,
    aten.erfc_.default,
    aten.erfc.out,
    aten.special_erfcx.default,
    aten.special_erfcx.out,
]

_NON_INCREASING_RULES: list[list[Placement]] = [
    [Partial("min"), Partial("max")],
    [Partial("max"), Partial("min")],
]

for op in non_increasing_unary_ops:
    _register_single_dim_pointwise(op, _NON_INCREASING_RULES)

# Bessel K functions are strictly decreasing for x > 0 but undefined at x <= 0.
# Only P(min)->P(max) is safe: P(min) offsets add positive values to the
# non-holding rank, keeping all inputs positive. P(max) offsets subtract,
# which can push inputs to x <= 0 producing NaN.
_POSITIVE_DOMAIN_NON_INCREASING_RULES: list[list[Placement]] = [
    [Partial("max"), Partial("min")],
]

for op in [
    aten.special_modified_bessel_k0.default,
    aten.special_modified_bessel_k1.default,
    aten.special_scaled_modified_bessel_k0.default,
    aten.special_scaled_modified_bessel_k1.default,
]:
    _register_single_dim_pointwise(op, _POSITIVE_DOMAIN_NON_INCREASING_RULES)

# neg is linear: -(A1 + A2) = -A1 + -A2
neg_ops = [
    aten.neg.default,
    aten.neg_.default,
    aten.neg.out,
    # foreach variants
    aten._foreach_neg.default,
    aten._foreach_neg_.default,
]

_NEG_RULES: list[list[Placement]] = _UNARY_LINEAR_RULES + _NON_INCREASING_RULES

for op in neg_ops:
    _register_single_dim_pointwise(op, _NEG_RULES)

# xlog1py(x, y) = x * log1p(y). Linear in x with y replicated:
# (a+b)*log1p(y) = a*log1p(y) + b*log1p(y).
_XLOG1PY_RULES: list[list[Placement]] = [
    [Partial("sum"), Partial("sum"), Replicate()],
    [Partial("avg"), Partial("avg"), Replicate()],
]

for op in [aten.special_xlog1py.default, aten.special_xlog1py.other_scalar]:
    _register_single_dim_pointwise(op, _XLOG1PY_RULES)


# All-partial-preserving unary ops: P(x)->P(x) for all x.
# TODO: positive should be removed once CIA (Copy Is All) optimizes it away.
all_partial_preserving_unary_ops = [
    aten.to.dtype,
    aten.positive.default,
]

_ALL_PARTIAL_PRESERVING_RULES: list[list[Placement]] = [
    [Partial(r), Partial(r)] for r in ("sum", "avg", "max", "min")
]

for op in all_partial_preserving_unary_ops:
    _register_single_dim_pointwise(op, _ALL_PARTIAL_PRESERVING_RULES)

all_partial_preserving_binary_ops = [
    aten.copy_.default,
    prims.copy_to.default,
]

_ALL_PARTIAL_BINARY_PRESERVING_RULES: list[list[Placement]] = [
    [Partial(r), Partial(r), Partial(r)] for r in ("sum", "avg", "max", "min")
]

for op in all_partial_preserving_binary_ops:
    _register_single_dim_pointwise(op, _ALL_PARTIAL_BINARY_PRESERVING_RULES)

# Monotonic increasing in both args but don't preserve any specific partial type.
monotonic_binary_ops = [
    aten.logaddexp.default,
    aten.logaddexp.out,
    aten.logaddexp2.default,
    aten.logaddexp2.out,
]

_MONOTONE_BINARY_BASE_RULES: list[list[Placement]] = [
    [Partial("max"), Partial("max"), Replicate()],
    [Partial("max"), Replicate(), Partial("max")],
    [Partial("min"), Partial("min"), Replicate()],
    [Partial("min"), Replicate(), Partial("min")],
]

for op in monotonic_binary_ops:
    _register_single_dim_pointwise(op, _MONOTONE_BINARY_BASE_RULES)

# Binary ops monotonically increasing in both arguments.
# max-preserving: P(max)+P(max)->P(max) because max(max(a),max(b)) = max(a,b)
monotonic_max_preserving_binary_ops = [
    aten.clamp_min.Tensor,
    aten.fmax.default,
    aten.fmax.out,
    aten.maximum.default,
    aten.maximum.out,
    prims.fmax.default,
    # foreach variants
    aten._foreach_maximum_.List,
]

_MONOTONE_MAX_PRESERVING_BINARY_BASE_RULES: list[list[Placement]] = [
    *_MONOTONE_BINARY_BASE_RULES,
    [Partial("max"), Partial("max"), Partial("max")],
]

for op in monotonic_max_preserving_binary_ops:
    _register_single_dim_pointwise(op, _MONOTONE_MAX_PRESERVING_BINARY_BASE_RULES)

# min-preserving: P(min)+P(min)->P(min) because min(min(a),min(b)) = min(a,b)
monotonic_min_preserving_binary_ops = [
    aten.clamp_max.Tensor,
    aten.fmin.default,
    aten.fmin.out,
    aten.minimum.default,
    aten.minimum.out,
    prims.fmin.default,
]

_MONOTONE_MIN_PRESERVING_BINARY_BASE_RULES: list[list[Placement]] = [
    *_MONOTONE_BINARY_BASE_RULES,
    [Partial("min"), Partial("min"), Partial("min")],
]

for op in monotonic_min_preserving_binary_ops:
    _register_single_dim_pointwise(op, _MONOTONE_MIN_PRESERVING_BINARY_BASE_RULES)


# Ops that are pointwise for DTensor purposes but lack torch.Tag.pointwise.
# TODO(pianpwk): add torch.Tag.pointwise to these ops in native_functions.yaml
# so this list can be removed.
_extra_pointwise_ops: list[OpOverload] = [
    aten.__irshift__.Scalar,
    aten.__irshift__.Tensor,
    aten._conj.default,
    aten.abs_.default,
    aten.copysign_.Scalar,
    aten.copysign_.Tensor,
    aten.exponential_.default,
    aten.float_power.Scalar,
    aten.float_power.Scalar_out,
    aten.float_power.Tensor_Scalar,
    aten.float_power.Tensor_Scalar_out,
    aten.float_power.Tensor_Tensor,
    aten.float_power.Tensor_Tensor_out,
    aten.masked_fill_.Scalar,
    aten.native_dropout_backward.out,
    aten.polygamma_.default,
    aten.rrelu_with_noise.default,
    aten.where.self_out,
    aten.xlogy_.Scalar_Other,
    prims.bessel_i0e.default,
    prims.bessel_i1.default,
    prims.bessel_i1e.default,
    prims.bessel_j0.default,
    prims.bessel_j1.default,
    prims.div.default,
    prims.erfcx.default,
    prims.frexp.default,
    prims.gcd.default,
    prims.ndtri.default,
    prims.ne.default,
    prims.spherical_bessel_j0.default,
    prims.zeta.default,
    # foreach variants
    aten._foreach_abs.default,
    aten._foreach_abs_.default,
    aten._foreach_addcdiv_.Scalar,
    aten._foreach_addcdiv_.ScalarList,
    aten._foreach_addcdiv_.Tensor,
    aten._foreach_addcmul.Scalar,
    aten._foreach_addcmul_.Scalar,
    aten._foreach_addcmul_.ScalarList,
    aten._foreach_addcmul_.Tensor,
    aten._foreach_lerp_.Scalar,
    aten._foreach_pow.List,
    aten._foreach_pow.ScalarList,
    aten._foreach_reciprocal_.default,
    aten._foreach_sub.Scalar,
    aten._foreach_sub_.Scalar,
    aten._foreach_sub.ScalarList,
    aten._foreach_sub_.ScalarList,
    aten._foreach_sqrt.default,
    aten._foreach_sqrt_.default,
    aten._foreach_zero_.default,
    aten._foreach_cos.default,
    aten._foreach_cos_.default,
    aten._foreach_log.default,
    aten._foreach_log_.default,
    aten._amp_foreach_non_finite_check_and_unscale_.default,
    # foreach linearity variants
    aten._foreach_add.Scalar,
    aten._foreach_add_.Scalar,
    aten._foreach_add_.ScalarList,
    # fused optimizer ops
    aten._fused_adam_.default,
    aten._fused_adam.default,
    aten._fused_adam.tensor_lr,
    aten._fused_adam_.tensor_lr,
    aten._fused_adamw_.default,
    aten._fused_adamw.default,
    aten._fused_adamw.tensor_lr,
    aten._fused_adamw_.tensor_lr,
]


def _get_pointwise_ops_from_tag() -> list[OpOverload]:
    """
    Auto-discover pointwise ops via torch.Tag.pointwise, from ops.aten, ops.prims.
    """
    ops = []
    for ns in [torch.ops.aten, torch.ops.prims]:
        for attr_name in dir(ns):
            attr = getattr(ns, attr_name)
            if isinstance(attr, torch._ops.OpOverloadPacket):
                for overload_name in attr.overloads():
                    op = getattr(attr, overload_name)
                    if torch.Tag.pointwise in op.tags:
                        ops.append(op)
    return ops


pointwise_ops = [
    op
    for op in _get_pointwise_ops_from_tag() + _extra_pointwise_ops
    if op not in _specially_registered_ops
]


for op in pointwise_ops:
    _register_single_dim_pointwise(op)


def register_inductor_prims() -> None:
    """Register DTensor sharding strategies for inductor prims ops.

    Called lazily because inductor prims are created via make_prim() in
    torch._inductor.inductor_prims, which is imported after this module.
    """
    # TODO: handle other inductor prims ops that may need DTensor sharding
    # strategies (e.g. mul_rn, div_rn). Those are more complicated and not
    # necessarily pointwise.
    _register_single_dim_pointwise(prims.fma.default)
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast, NamedTuple

import torch
from torch import Tensor
from torch._prims_common import DimsType
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpSpec,
    OpStrategy,
    RuntimeSchemaInfo,
    StrategyType,
)
from torch.distributed.tensor._ops.single_dim_strategy import (
    _ShardingPlaceholder,
    register_single_dim_strategy,
)
from torch.distributed.tensor._ops.utils import (
    generate_redistribute_costs,
    normalize_dim,
    normalize_dims,
    prod,
    register_op_strategy,
)
from torch.distributed.tensor.placement_types import (
    _defer_symbolic_equality_assert,
    _StridedShard,
    Partial,
    Placement,
    Replicate,
    Shard,
)


aten = torch.ops.aten

Shape = tuple[int, ...]


def _explicit_unbacked_hint(value: object) -> int | None:
    from torch.fx.experimental.symbolic_shapes import (
        free_unbacked_symbols,
        optimization_hint,
    )

    if not isinstance(value, torch.SymInt):
        return None
    shape_env = getattr(value.node, "shape_env", None)
    if shape_env is None:
        return None
    unbacked_symbols = free_unbacked_symbols(value.node.expr)
    if not unbacked_symbols or not unbacked_symbols.issubset(
        shape_env.var_to_hint_override.keys()
    ):
        return None
    return int(optimization_hint(value, fallback=None))


def _split_factor_hint(value: object) -> int | None:
    if isinstance(value, int):
        return int(value)
    return _explicit_unbacked_hint(value)


def _split_factor_matches(expected: object, actual: object) -> bool:
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    if expected is None:
        return False
    if guard_or_false(expected == actual):
        return True
    expected_hint = _split_factor_hint(expected)
    actual_hint = _split_factor_hint(actual)
    if expected_hint is None or actual_hint is None or expected_hint != actual_hint:
        return False
    if isinstance(expected, torch.SymInt):
        _defer_symbolic_equality_assert(
            expected, actual_hint, reason="_StridedShard split_factor match"
        )
    elif isinstance(actual, torch.SymInt):
        _defer_symbolic_equality_assert(
            actual, expected_hint, reason="_StridedShard split_factor match"
        )
    return True


class ClaimedDim(NamedTuple):
    """An (input_dim, output_dim) pair claimed by a mesh dim's _StridedShard rewrite."""

    input_dim: int
    output_dim: int


@dataclass
class DimSpec:
    """Specifies how an output dimension maps to an input dimension."""

    def inputs(self) -> Iterable["DimSpec"]:
        return ()


# Rules that map each dimension of the output to dimensions of the input tensor
DimMap = tuple[DimSpec, ...]


@dataclass
class Singleton(DimSpec):
    """Output dimension is a singleton."""


@dataclass(eq=False)
class InputDim(DimSpec):
    """Output dimension maps directly to an input dimension."""

    input_dim: int

    def __eq__(self, other: object) -> bool:
        """Raises TypeError for non-DimSpec comparisons to catch accidental
        ``shard.dim == input_dim`` bugs where ``.input_dim`` was intended."""
        if isinstance(other, InputDim):
            return self.input_dim == other.input_dim
        if not isinstance(other, DimSpec):
            raise TypeError(
                f"Cannot compare InputDim with {type(other).__name__}. "
                f"Did you mean to use .input_dim?"
            )
        return NotImplemented

    def __hash__(self) -> int:
        return hash((InputDim, self.input_dim))


@dataclass
class Broadcast(DimSpec):
    """Output is the broadcast of a singleton input dimension."""

    dim: DimSpec
    dim_size: int

    @classmethod
    def new(cls, dim: DimSpec, dim_size: int) -> DimSpec:
        return Broadcast(dim, dim_size)

    def inputs(self) -> Iterable[DimSpec]:
        return (self.dim,)


@dataclass
class NewDim(DimSpec):
    """This is a new dimension created by the op."""

    size: int

    @classmethod
    def new(cls, size: int) -> DimSpec:
        from torch.fx.experimental.symbolic_shapes import guard_or_false

        return Singleton() if guard_or_false(size == 1) else NewDim(size)


@dataclass
class Repeat(DimSpec):
    """Output dimension is the input dimension repeated n-times."""

    input_dim: DimSpec
    times: int

    @classmethod
    def new(cls, dim: DimSpec, times: int) -> DimSpec:
        from torch.fx.experimental.symbolic_shapes import guard_or_false

        if guard_or_false(times == 1):
            return dim
        elif isinstance(dim, Singleton):
            # repeating a singleton is the same as broadcasting it
            return Broadcast(dim, times)
        else:
            return Repeat(dim, times)

    def inputs(self) -> Iterable[DimSpec]:
        return (self.input_dim,)


@dataclass
class Flatten(DimSpec):
    """Flatten a set of input dimensions, ensuring right-most adjacent elements remain adjacent in the output."""

    input_dims: Sequence[DimSpec]

    @classmethod
    def new(cls, dims: Sequence[DimSpec]) -> DimSpec:
        if len(dims) == 0:
            # flattening a scalar leads to a singleton
            return Singleton()
        elif len(dims) == 1:
            # flattening a single dimension is no-op
            return dims[0]
        else:
            return Flatten(dims)

    def inputs(self) -> Iterable[DimSpec]:
        return self.input_dims


@dataclass
class Split(DimSpec):
    """
    This dimension is a member of a decomposition of the input dim.

    Note that input_dim itself could be a Flattened set of input dims.
    """

    input_dim: DimSpec
    group_shape: Shape
    split_id: int

    @classmethod
    def new(cls, dim: DimSpec, group_shape: tuple[int, ...], idx: int) -> DimSpec:
        from torch.fx.experimental.symbolic_shapes import guard_or_false, guard_or_true

        if not len(group_shape) > 0:
            raise AssertionError(
                f"Expected group_shape length > 0, got {len(group_shape)}"
            )
        if len(group_shape) == 1:
            # not really a group, just return the input dim back
            if not idx == 0:
                raise AssertionError(f"Expected idx == 0, got {idx}")
            return dim
        elif guard_or_false(group_shape[idx] == 1):
            return Singleton()
        else:
            # remove singletons from group
            # group_mapping = [(new_index, (shape, old_index)) ...]
            group_mapping = list(
                enumerate(
                    (s, i) for i, s in enumerate(group_shape) if guard_or_true(s != 1)
                )
            )
            new_group_shape = tuple(m[1][0] for m in group_mapping)
            new_idx = next(filter(lambda x: x[1][1] == idx, group_mapping))[0]
            return Split(dim, new_group_shape, new_idx)

    def inputs(self) -> Iterable[DimSpec]:
        return (self.input_dim,)


def dim_pad_left(ndim: int, min_dims: int) -> DimMap:
    return (Singleton(),) * max(0, min_dims - ndim) + tuple(
        InputDim(i) for i in range(ndim)
    )


def dim_atleast_3d(ndim: int) -> DimMap:
    if ndim == 0:
        return (Singleton(), Singleton(), Singleton())
    elif ndim == 1:
        return (Singleton(), InputDim(0), Singleton())
    elif ndim == 2:
        return (InputDim(0), InputDim(1), Singleton())
    else:
        return tuple(InputDim(i) for i in range(ndim))


def expand(input_shape: Shape, shape: Shape) -> DimMap:
    """Implement broadcast on multiple dimensions."""
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    if not len(shape) >= len(input_shape):
        raise AssertionError(
            f"Expected len(shape) >= len(input_shape), got {len(shape)} < {len(input_shape)}"
        )

    # 1. create padded input dimensions
    padded_input = dim_pad_left(len(input_shape), len(shape))
    # 2. check that input shapes are compatible
    mapping = []
    for p, desired_s in zip(padded_input, shape):
        if isinstance(p, Singleton):
            actual_s = 1
            if not desired_s >= 0:
                raise AssertionError(f"Expected desired_s >= 0, got {desired_s}")
        else:
            if not isinstance(p, InputDim):
                raise AssertionError(f"DimSpec not supported in expand: {p}")
            actual_s = input_shape[p.input_dim]
            if not (
                guard_or_false(actual_s == 1)
                or guard_or_false(desired_s == -1)
                or guard_or_false(desired_s == actual_s)
            ):
                raise AssertionError(
                    f"Expected actual_s == 1 or desired_s == -1 or "
                    f"desired_s == actual_s, got actual_s={actual_s}, desired_s={desired_s}"
                )
        mapping.append(
            p
            if (
                guard_or_false(desired_s == 1)
                or guard_or_false(desired_s == -1)
                or guard_or_false(desired_s == actual_s)
            )
            else Broadcast.new(p, desired_s)
        )
    return tuple(mapping)


def normalize_sizes(sizes: Shape | tuple[Shape]) -> Shape:
    if isinstance(sizes[0], (int, torch.SymInt)):
        return cast(Shape, sizes)
    elif len(sizes) == 1:
        return sizes[0]
    else:
        raise RuntimeError("Size must be int... or tuple")


def dim_flatten(ndim: int, start_dim=0, end_dim=-1) -> DimMap:
    if ndim == 0:
        return (Singleton(),)
    elif ndim == 1:
        return (InputDim(0),)
    else:
        # only flattening dims from start_dim to end_dim (inclusive)
        # other dims are passed through
        if end_dim < 0:
            end_dim += ndim
        results: list[DimSpec] = [InputDim(i) for i in range(start_dim)]
        results.append(
            Flatten.new(tuple(InputDim(i) for i in range(start_dim, end_dim + 1)))
        )
        results.extend([InputDim(i) for i in range(end_dim + 1, ndim)])
        return tuple(results)


def dim_movedim(
    ndim: int,
    input: DimsType,
    destination: DimsType,
) -> DimMap:
    input = normalize_dims(input, ndim)
    destination = normalize_dims(destination, ndim)

    if not len(input) == len(destination):
        raise AssertionError(
            f"Expected len(input) == len(destination), got {len(input)} != {len(destination)}"
        )
    input_set = set(input)
    if not len(input_set) == len(input):
        raise AssertionError("Found repeated input dims")
    if not len(set(destination)) == len(destination):
        raise AssertionError("Found repeated output dims")
    if not max(input) < ndim:
        raise AssertionError(f"Expected max(input) < ndim, got {max(input)} >= {ndim}")
    if not max(destination) < ndim:
        raise AssertionError(
            f"Expected max(destination) < ndim, got {max(destination)} >= {ndim}"
        )

    dest = [-1] * ndim
    for i, d in zip(input, destination):
        dest[d] = i

    unused_inputs_iter = iter(i for i in range(ndim) if i not in input_set)
    for i in range(ndim):
        if dest[i] == -1:
            dest[i] = next(unused_inputs_iter)

    return tuple(InputDim(i) for i in dest)


def dim_repeat(ndim: int, sizes: Shape) -> DimMap:
    sizes = normalize_sizes(sizes)
    if not len(sizes) >= ndim:
        raise AssertionError(
            f"Number of dimensions of repeat dims {sizes} can not be smaller than number of dimensions of tensor {ndim}."
        )
    pad = len(sizes) - ndim
    return tuple(Repeat.new(Singleton(), s) for s in sizes[:pad]) + tuple(
        Repeat.new(InputDim(i), s) for i, s in enumerate(sizes[pad:])
    )


def infer_size(total_size: int, sizes: Shape) -> Shape:
    """
    One dimension input to view may be "-1".

    Infer the size of this dimension given the total_size.
    """
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    infers = [i for i, s in enumerate(sizes) if guard_or_false(s == -1)]
    size = prod(sizes)
    if not len(infers) <= 1:
        raise AssertionError("can only infer one size")
    if infers:
        size = -size
        missing_size = total_size // size
        torch._check(
            total_size % size == 0,
            lambda: f"size inferred for -1 is not integral {sizes} should have {total_size} elements.",
        )
        return tuple(s if not guard_or_false(s == -1) else missing_size for s in sizes)
    torch._check(
        size == total_size,
        lambda: f"sizes do not match {total_size} vs {size}",
    )
    return sizes


def view_groups(from_size: Shape, to_size: Shape) -> DimMap:
    """
    Decompose a reshape operation into forwarding, flattening, or splitting dimensions for each output dimension.

    A view or reshape operation can be decomposed into a set of 3 types of smaller operations:
    1) Forward a dimension from input to output
    2) Flatten a set of dimensions into a single dimension
    3) Split one dimension into multiple dimensions

    view_groups identifies these operations and returns, for each output dimension, what
    operation was performed in the input dimension. For example:

        view_groups([2, 3, 4], [2, 12]) -> (
            InputDim(0),
            Flatten((InputDim(1), InputDim(2)))
        )

    - output dimension 0 maps to input dimension 0
    - output dimension 1 maps to a flattened input dimensions 1 and 2


        view_groups([2, 3], [3, 2]) -> (
            Split(Flatten((InputDim(0), InputDim(1))), (3, 2), 0),
            Split(Flatten((InputDim(0), InputDim(1))), (3, 2), 1),
        )

    - in the above, input is flattened into a single dimension and then split
      into two separate dimensions with different sizes from the input.
    """
    from torch.fx.experimental.symbolic_shapes import (
        free_symbols,
        guard_or_false,
