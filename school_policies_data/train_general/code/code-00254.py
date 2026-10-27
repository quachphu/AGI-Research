                    aten.view.default(shard, shard.shape, **kwargs)
                    for shard in args[0].local_shards()
                ]
            else:
                raise NotImplementedError("No support for view on tensors ndim > 2")
        else:
            # view is called per shard
            res_shards_list = [
                aten.view.default(shard, args[1], **kwargs)
                for shard in args[0].local_shards()
            ]
        return LocalShardsWrapper(res_shards_list, args[0].local_offsets())

    @staticmethod
    def handle_equal(args, kwargs) -> bool:
        """
        LocalShardsWrapper equal impl also checks for equality of storage metadata
        and the order of shards
        """
        a, b = args[0], args[1]
        if len(a.local_shards()) != len(b.local_shards()):
            return False
        if not all(
            aten.equal.default(x, y) for x, y in zip(a.local_shards(), b.local_shards())
        ):
            return False
        if a.storage_metadata() != b.storage_metadata():
            return False
        return True

    @staticmethod
    def handle_detach(args, kwargs) -> "LocalShardsWrapper":
        self_ls = args[0]
        deatched_local_shards = [
            aten.detach.default(shard) for shard in self_ls.local_shards()
        ]
        self_ls._local_shards = deatched_local_shards
        self_ls._storage_meta.properties.requires_grad = False
        return self_ls

    @staticmethod
    def handle_clone(args, kwargs) -> "LocalShardsWrapper":
        self_ls = args[0]
        desired_memory_format = kwargs.get("memory_format", None)
        if desired_memory_format and desired_memory_format != torch.preserve_format:
            raise NotImplementedError(
                f"{desired_memory_format} is not supported for LocalShardsWrapper!"
            )
        cloned_local_shards = [
            shard.clone(memory_format=desired_memory_format)
            for shard in self_ls._local_shards
        ]
        return LocalShardsWrapper(cloned_local_shards, self_ls.local_offsets())

    @staticmethod
    def handle_new_empty(args, kwargs) -> "LocalShardsWrapper":
        self_ls = args[0]
        return LocalShardsWrapper(
            [torch.empty_like(shard) for shard in self_ls._local_shards],
            self_ls.local_offsets(),
        )

    @property
    def device(self) -> torch._C.device:  # type: ignore[override]
        return (
            self._local_shards[0].device if self._local_shards else torch.device("meta")
        )

    @property
    def is_meta(self) -> bool:  # type: ignore[override]
        return self._local_shards[0].is_meta if self._local_shards else True

    def is_pinned(self) -> bool:  # type: ignore[override]
        return self._storage_meta.properties.pin_memory

    def requires_grad_(self, requires_grad: bool = True) -> "LocalShardsWrapper":
        self._storage_meta.properties.requires_grad = requires_grad
        [shard.requires_grad_(requires_grad) for shard in self._local_shards]
        return self

    def local_shards(self) -> list[torch.Tensor]:
        """
        Returns a list of :class:`torch.Tensor' corresponding to the
        local shards for this rank. Returns an empty list if the current rank
        does not host any shards for this Tensor.
        """
        return self._local_shards

    def local_sizes(self) -> list[torch.Size]:
        """
        Returns a list of :class:`torch.Size' corresponding to the
        local sizes for the shards on this rank. Returns an empty list if the current rank
        does not host any shards for this Tensor.
        """
        return [chunk.sizes for chunk in self._storage_meta.chunks]

    def local_offsets(self) -> list[torch.Size]:
        """
        Returns a list of :class:`torch.Size' corresponding to the
        local offsets for the shards on this rank. Returns an empty list if the current rank
        does not host any shards for this Tensor.
        """
        return [chunk.offsets for chunk in self._storage_meta.chunks]

    @property
    def local_chunks(self) -> list[ChunkStorageMetadata]:
        """
        Returns a :class:`list[ChunkStorageMetadata]` object corresponding to the
        metadata for each tensor shard
        """
        return self._storage_meta.chunks

    def storage_metadata(self) -> TensorStorageMetadata:
        """
        Returns a :class:`TensorStorageMetadata` object corresponding to the
        metadata for the local tensor on current rank
        """
        return self._storage_meta

    def is_empty_shard(self) -> bool:
        """
        Returns a :class:`bool` object indicating if the local tensor on current rank
        is an empty tensor
        """
        return self._storage_meta.size[0] == 0 and self._storage_meta.size[1] == 0

    def __create_write_items__(self, fqn: str, object: Any) -> list[WriteItem]:
        """
        For compatibility with DCP, we support creation of WriteItems
        such that they can be saved properly.
        """
        return [
            WriteItem(
                index=MetadataIndex(fqn, chunks.offsets),
                type=WriteItemType.SHARD,
                tensor_data=TensorWriteData(
                    chunk=ChunkStorageMetadata(
                        offsets=chunks.offsets,
                        sizes=chunks.sizes,
                    ),
                    properties=self._storage_meta.properties,
                    size=object.size(),
                ),
            )
            for tensor, chunks in zip(self.local_shards(), self.local_chunks)
        ]

    def __create_chunk_list__(self) -> list[ChunkStorageMetadata]:
        """
        For compatibility with DCP, we support creation of chunk lists
        such that they can be saved properly.
        """
        return self._storage_meta.chunks

    def __get_tensor_shard__(self, index: MetadataIndex) -> torch.Tensor:
        """
        For compatibility with DCP, we support finding shard based on index
        Return a 'torch.Tensor' shard based on 'MetadataIndex'.
        """
        # Fast lookup path
        if index.index is not None:
            if (
                len(self._local_shards) > index.index
                and self._storage_meta.chunks[index.index].offsets == index.offset
            ):
                return self._local_shards[index.index]

        if index.offset is not None:
            for shard, chunk in zip(self._local_shards, self._storage_meta.chunks):
                if chunk.offsets == index.offset:
                    return shard

        # Empty shard case
        if len(self._local_shards) == 0 and self._storage_meta.chunks[
            0
        ].sizes == torch.Size([0, 0]):
            return torch.empty(0)

        raise ValueError(
            f"Could not find shard at '{index.offset}' for FQN: '{index.fqn}'"
        )

    def _get_tensor_size_bytes(self) -> int:
        object_size = 0
        for shard in self.local_shards():
            object_size += shard.nelement() * shard.element_size()
        return object_size

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:  # type: ignore[override]
        return f"LocalShardsWrapper:{self._local_shards} {self._storage_meta}"

    def __str__(self) -> str:
        return f"LocalShardsWrapper:{self._local_shards} {self._storage_meta}"
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates

import torch
import torch.distributed.tensor._ops  # force import all built-in dtensor ops
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor._api import (
    distribute_module,
    distribute_tensor,
    DTensor,
    empty,
    full,
    linspace,
    logspace,
    ones,
    rand,
    randn,
    zeros,
)
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.optim.optimizer import (
    _foreach_supported_types as _optim_foreach_supported_types,
)
from torch.utils._foreach_utils import (
    _foreach_supported_types as _util_foreach_supported_types,
)


# All public APIs from dtensor package
__all__ = [
    "DTensor",
    "distribute_tensor",
    "distribute_module",
    "Shard",
    "Replicate",
    "Partial",
    "Placement",
    "ones",
    "empty",
    "full",
    "linspace",
    "logspace",
    "rand",
    "randn",
    "zeros",
]

# For weights_only torch.load
from ._dtensor_spec import (
    DTensorSpec as _DTensorSpec,
    ShardOrderEntry as _ShardOrderEntry,
    TensorMeta as _TensorMeta,
)


torch.serialization.add_safe_globals(
    [
        DeviceMesh,
        _DTensorSpec,
        _TensorMeta,
        _ShardOrderEntry,
        DTensor,
        Partial,
        Replicate,
        Shard,
        _StridedShard,
    ]
)


# Append DTensor to the list of supported types for foreach implementation for optimizer
# and clip_grad_norm_ so that we will try to use foreach over the for-loop implementation on CUDA.
if DTensor not in _optim_foreach_supported_types:
    _optim_foreach_supported_types.append(DTensor)

if DTensor not in _util_foreach_supported_types:
    _util_foreach_supported_types.append(DTensor)  # type: ignore[arg-type]


# Set namespace for exposed private names
DTensor.__module__ = "torch.distributed.tensor"
distribute_tensor.__module__ = "torch.distributed.tensor"
distribute_module.__module__ = "torch.distributed.tensor"
ones.__module__ = "torch.distributed.tensor"
empty.__module__ = "torch.distributed.tensor"
full.__module__ = "torch.distributed.tensor"
linspace.__module__ = "torch.distributed.tensor"
logspace.__module__ = "torch.distributed.tensor"
rand.__module__ = "torch.distributed.tensor"
randn.__module__ = "torch.distributed.tensor"
zeros.__module__ = "torch.distributed.tensor"

# Register DTensor dispatch for higher order operators
from torch._higher_order_ops.print import _register_dtensor_impl


_register_dtensor_impl()
<|endoftext|># mypy: allow-untyped-defs
"""
DTensor operator schema definitions and utilities.

This module defines the core data structures and utilities for describing and managing
distributed tensor operations in PyTorch's DTensor system. It provides the foundational
schema types used for sharding propagation, operator strategy selection, and distributed
execution planning.

Key components:
- OpSpec: Describes acceptable sharding placements for operations
- OpStrategy: Represents the possible sharding strategies for an operator
- TupleStrategy: Container for multiple strategies when ops have tuple/list of tensors input
- OpSchema: Describes operator input/output schemas with DTensorSpecs
- OutputSharding: Manages output sharding specifications and redistribution
- RuntimeSchemaInfo: Runtime execution metadata for operators
- OpInfo: Complete runtime operator execution information

These schema definitions enable the DTensor system to:
1. Propagate tensor sharding information to the operator outputs
2. Greedily select sharding strategies for distributed operations
3. Plan and execute tensor redistributions when needed
4. Cache sharding decisions for performance optimization
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any
from typing_extensions import deprecated

import torch
from torch._C import (
    _DTensor_OpSchema_post_init,
    _DTensor_OpSchema_recompute_comparison_key,
)
from torch._ops import OpOverload
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor.placement_types import Placement


try:
    from torch.utils._cxx_pytree import (
        register_pytree_node,
        tree_leaves,
        tree_map_only,
        TreeSpec,
    )
except ImportError:
    from torch.utils._pytree import (  # type: ignore[no-redef, assignment]
        register_pytree_node,
        tree_leaves,
        tree_map_only,
        TreeSpec,
    )


# Common type aliases
ArgsType = tuple[object, ...]
KwargsType = dict[str, object]

PlacementList = list[Placement | None]

# ATen op schemas could have Tensor, Tuple[Tensor] and List[Tensor], so output type should
# be the same set of possibilities.
OutputSpecType = DTensorSpec | Sequence[DTensorSpec | None] | None


def _rebuild_tensor_from_dtensor_meta(arg) -> object:
    """
    This is used to propagate tensor metadata, must be under fake mode
    """
    if arg.tensor_meta is None:
        raise AssertionError("DTensorSpec does not contain tensor_meta.")
    return torch.empty_strided(
        arg.tensor_meta.shape,
        arg.tensor_meta.stride,
        dtype=arg.tensor_meta.dtype,
        device=arg.mesh.device_type,
    )


def _pretty_print_spec(spec: object) -> str:
    if spec is None:
        return "None"
    elif isinstance(spec, DTensorSpec):
        return "".join([str(p) for p in spec.placements])
    elif isinstance(spec, Sequence):
        return "(" + ", ".join([_pretty_print_spec(s) for s in spec]) + ")"
    else:
        raise RuntimeError(f"Unknown spec type to print: spec={spec}")


@dataclass
class OpSpec:
    """
    An OpSpec describes an acceptable sharding placements of an operation, with the
    specified DTensorSpecs for both the output and the inputs.

    note: when the op return value is a single DTensor object, output_specs is
    DTensorSpec; when the return value is a tuple of Optional[DTensor],
    output_specs is a tuple of Optional[DTensorSpec].

    note: we MUST produce an DTensorSpec for every output that is a Tensor.  None
    entries only occur for non-Tensor outputs (e.g., operators that return Optional[Tensor],
    or non-Tensor outputs.)

    invariant: the DeviceMesh on all DTensorSpec must be the same
    """

    # output_specs and input_specs are related: for this op, given these input_specs,
    # this is the way the output would look
    # Note: output_specs can be None for ops that don't return tensors (e.g., _linalg_check_errors)
    output_specs: DTensorSpec | tuple[DTensorSpec | None, ...] | None
    input_specs: Sequence[DTensorSpec] | None = None

    """
    redistribute_cost tells how expensive it is to redistribute a given input into the
    placement specified in this OpSpec.

    outer list: one entry (list) per (tensor) input in the op's arg schema
    inner list: one entry (cost value) per possible sharding spec for that input

    Example:
    -------
    another_op() -> tensor_a   # another_op produces the output that becomes our first input
    my_op(tensor_a)

    Let's assume this OpSpec's input_specs are [Replicate()],
    but another_op() supports 2 strategies (OpSpecs) which produce outputs of
       Replicate()
       Shard(0)

    In this example, redistribute_costs would look like this
    [
        # one row representing "my_op's first input" (tensor_a)
        [
            # two entries, one for each strategies supported by another_op
            0.0,  # cost of redistributing tensor_a from 'Replicate()'
            K,    # cost of redistributing tensor_a from 'Shard(0)'
        ],
    """
    redistribute_cost: list[list[float]] | None = None

    @cached_property
    def output_spec(self) -> DTensorSpec:
        """
        This function requires that the strategy have exactly one DTensorSpec as the
        output spec. If the output_specs is a tuple, we throw an exception.
        """
        if isinstance(self.output_specs, DTensorSpec):
            return self.output_specs
        else:
            raise ValueError(
                f"function output_spec expects a single DTensorSpec but got: {self.output_specs}"
            )

    @cached_property
    def mesh(self):
        if isinstance(self.output_specs, DTensorSpec):
            return self.output_specs.mesh
        elif isinstance(self.output_specs, tuple):
            for out_spec in self.output_specs:
                if isinstance(out_spec, DTensorSpec):
                    return out_spec.mesh
            # all entries are None — fall through to input_specs
        elif self.output_specs is not None:
            raise ValueError(
                f"output_specs expects DTensorSpec, tuple of DTensorSpec, or None but got: {self.output_specs}"
            )
        # output_specs is None or tuple-of-all-None — try input_specs
        if self.input_specs is not None:
            for spec in self.input_specs:
                if spec is not None:
                    return spec.mesh
        raise AssertionError(
            "Cannot determine mesh: no non-None DTensorSpec in output_specs or input_specs"
        )

    def input_spec(self, index: int = 0) -> DTensorSpec:
        if self.input_specs is None:
            raise AssertionError("input_specs of OpSpec is None!")
        if len(self.input_specs) <= index:
            raise AssertionError(
                f"Invalid index {index} for input_specs of length "
                f"{len(self.input_specs)}: {self.input_specs}"
            )
        return self.input_specs[index]

    def __str__(self) -> str:
        if self.input_specs is not None:
            input_specs_str = f"{_pretty_print_spec(self.input_specs)} -> "
        else:
            input_specs_str = ""
        output_spec_str = _pretty_print_spec(self.output_specs)
        return f"{input_specs_str}{output_spec_str}"

    def __hash__(self) -> int:
        if self.output_specs is None:
            output_hash = hash(None)
        elif isinstance(self.output_specs, DTensorSpec):
            output_hash = hash(self.output_specs)
        else:
            output_hash = hash(tuple(self.output_specs))
        input_hash = hash(tuple(self.input_specs)) if self.input_specs else 0
        return hash((output_hash, input_hash))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OpSpec):
            return False
        return (
            self.output_specs == other.output_specs
            and self.input_specs == other.input_specs
        )


class StrategyType:
    """
    Base class type for op strategy, We have two StrategyType:
        OpStrategy and TupleStrategy
    """


class OpStrategy(StrategyType):
    """
    OpStrategy that consists of a list of sharding strategies associated with the op,
    where each strategy is an OpSpec that describes the acceptable input/output sharding.

    invariant: the DeviceMesh on all OpSpec must be the same
    """

    def __init__(self, strategies: list[OpSpec]) -> None:
        super().__init__()
        self.strategies: list[OpSpec] = strategies

    def __str__(self) -> str:
        strategy_list_str = ", ".join([str(strategy) for strategy in self.strategies])
        try:
            mesh_shape = self.mesh_shape
        except AssertionError:
            return f"OpStrategy[{strategy_list_str}]"
        return f"OpStrategy[{strategy_list_str}] @ mesh: {mesh_shape}"

    def max_num_shards(self) -> int:
        """
        Returns the max number of shards across all OpSpecs
        """
        return max(strategy.output_spec.num_shards for strategy in self.strategies)

    @property
    def mesh(self):
        return self.strategies[0].mesh

    @property
    def mesh_shape(self):
        return self.strategies[0].mesh.shape

    @property
    def ndim(self):
        return self.strategies[0].output_spec.ndim

    @property
    def shape(self):
        return self.strategies[0].output_spec.shape

    @property
    def tensor_meta(self) -> TensorMeta:
        # TODO upstream this assert to DTensorSpec itself and fill any missing TensorMetas
        if self.strategies[0].output_spec.tensor_meta is None:
            raise AssertionError
        return self.strategies[0].output_spec.tensor_meta

    def __hash__(self) -> int:
        return hash(tuple(self.strategies))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, OpStrategy) and self.strategies == other.strategies


class TupleStrategy(StrategyType):
    """
    TupleStrategy is a special case for operators that are fundamentally compound or batched such that some subset
    of the inputs and outputs are completely unrelated to some other subset.

    Generally, foreach_* ops are the most common use-case for TupleStrategy, because they accept lists of inputs,
    but operate independently on each input or tuple of zipped inputs.

    For example, [out_a, out_b] = torch.foreach_add([a,  b], scalar): input a's sharding only affects out_a's sharding,
    independent of b and out_b.

    An example of an operator that should NOT use TupleStrategy is torch.split.  It produces a List[Tensor]
    as its output, but the sharding decision of one output is bound together with the decision
    of each other output and the common input.
    """

    def __init__(
        self,
        children: Sequence[StrategyType],
    ) -> None:
        super().__init__()
        self.children: Sequence[StrategyType] = children

    @property
    @deprecated(
        "TupleStrategy.childs is deprecated, use TupleStrategy.children instead.",  # codespell:ignore childs
        category=FutureWarning,
    )
    def childs(self) -> Sequence[StrategyType]:  # codespell:ignore childs
        """
        Alias for children, to maintain backward compatibility.
        """
        return self.children

    def child_mesh(self, index: int) -> DeviceMesh:
        op_strategy = self.children[index]
        if not isinstance(op_strategy, OpStrategy):
            raise AssertionError
        return op_strategy.mesh

    def __str__(self) -> str:
        child_strategies_str = ", ".join(
            [f"{str(strat)}" for idx, strat in enumerate(self.children)]
        )
        return f"TupleStrategy({child_strategies_str})"

    def __hash__(self) -> int:
        return hash(tuple(self.children))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TupleStrategy) and self.children == other.children


try:
    register_pytree_node(
        TupleStrategy,
        lambda node: (node.children, None),
        lambda children, _: TupleStrategy(tuple(children)),
    )
except ValueError:
    # already registered TupleStrategy, skip
    pass


@dataclass
class RuntimeSchemaInfo:
    """
    RuntimeSchemaInfo stores the operator schema related information for runtime (eager)
    execution. This is mainly used for two ways: 1. to generate hash for args to determine
    whether to re-run sharding prop or not 2. to determine if we need pytree
    """

    # This static_argnum records static arg "starting index" for ops that have non-tensor
    # args/kwargs which would affect sharding propagation results. All args starting from
    # this index would be hashed to our sharding cache.
    # Note that only a few ops need this information, e.g. view, transpose, var.dim, etc.
    static_argnum: int = 100
    # This static_kwargkey records static kwarg names which would affect sharding prop
    static_kwargkey: list[str] | None = None
    # each op can decide if it wants to use pytree flatten/unflatten during operator
    # eager execution, by default we don't need to do flatten/unflatten, only if the
    # op indicate it needs to, this is to accelerate eager performance.
    needs_pytree: bool = False


@dataclass
class OpSchema:
    """
    OpSchema is a data class that describes an operator input schemas, it includes
    DTensorSpecs/OpStrategies (instead of DTensor) and non-tensor args/kwargs (positional
    order preserved). It is mainly used by the DTensor's dispatching logic to perform various
    actions (i.e. sharding propagation, caching sharding decisions, redistribute, etc.)

    NOTE: this must be used as a read only data class
    TODO: make this a frozen dataclass

    Args:
        op: the operator overload we are intercepting
        args_schema: contains args except that the DTensor args have been replaced
            with its DTensorSpec or OpStrategy
        kwargs_schema: contains kwargs except that the DTensor kwargs have been replaced
            with its DTensorSpec or OpStrategy
    """

    op: OpOverload
    args_schema: ArgsType
    kwargs_schema: KwargsType

    schema_info: RuntimeSchemaInfo | None = None

    _comparison_key: tuple[object, ...] | None = None

    @property
    def args_spec(self) -> tuple[DTensorSpec, ...]:
        """
        args_spec: Tuple[DTensorSpec, ...]: contains a clean list of args spec list
            with NO non-DTensor positional arguments (i.e. int/float/tuple, etc)
            mainly used by sharding propagation to propagate the output spec
        """
        args = (
            tree_leaves(self.args_schema)
            if self.schema_info is not None and self.schema_info.needs_pytree
            else self.args_schema
        )
        return tuple(item for item in args if isinstance(item, DTensorSpec))

    @property
    def args_strategy(self) -> tuple[OpStrategy, ...]:
        # filter out non-relevant values from args schema to get a clean OpStrategy list
        # separate with args_spec for the ease of type annotation
        # TODO: see if we should merge this with args_spec
        args = (
            tree_leaves(self.args_schema)
            if self.schema_info is not None and self.schema_info.needs_pytree
            else self.args_schema
        )
        return tuple(item for item in args if isinstance(item, OpStrategy))

    @property
    def kwargs_strategy(self) -> tuple[OpStrategy, ...]:
        # returns OpStrategy items from kwargs_schema.
        kwargs_vals = (
            tree_leaves(self.kwargs_schema)
            if self.schema_info is not None and self.schema_info.needs_pytree
            else self.kwargs_schema.values()
        )
        return tuple(item for item in kwargs_vals if isinstance(item, OpStrategy))

    @property
    def args_meta(self) -> tuple[TensorMeta | Any, ...]:
        # Used for calling single_dim strategy functions, which aren't allowed to see DTensorSpecs/Meshes
        # like args_spec, but has OpStrategy replaced with corresponding TensorMeta,
        # and TupleStrategy replaced with tuple of TensorMeta
        # preserves the original pytree structure
        # example:
        # args_schema = (OpStrategy1, TupleStrategy([OpStrategy2, OpStrategy3]), OpStrategy4)
        # args_meta: (TensorMeta1, (TensorMeta2, TensorMeta3), TensorMeta4)

        def convert_to_meta(item):
            if isinstance(item, OpStrategy):
                return item.tensor_meta
            elif isinstance(item, TupleStrategy):
                return tuple(convert_to_meta(child) for child in item.children)
            elif isinstance(item, (list, tuple)):
                converted = [convert_to_meta(child) for child in item]
                return type(item)(converted)
            else:
                return item

        return tuple(convert_to_meta(arg) for arg in self.args_schema)

    @property
    def kwargs_meta(self) -> dict[str, object]:
        # like args_meta, but for kwargs

        def convert_to_meta(item):
            if isinstance(item, OpStrategy):
                return item.tensor_meta
            elif isinstance(item, TupleStrategy):
                return tuple(convert_to_meta(child) for child in item.children)
            elif isinstance(item, (list, tuple)):
                converted = [convert_to_meta(child) for child in item]
                return type(item)(converted)
            else:
                return item

        return {
            key: convert_to_meta(value) for key, value in self.kwargs_schema.items()
        }

    def __repr__(self) -> str:
        args_schema = ", ".join([str(arg_schema) for arg_schema in self.args_schema])
        return (
            f"OpSchema(op={self.op},"
            f" args_schema=({args_schema}),"
            f" kwargs_schema={self.kwargs_schema})"
        )

    def __str__(self) -> str:
        args_schema: list[str] = []
        device_mesh = None

        for arg in self.args_schema:
            if isinstance(arg, DTensorSpec):
                args_schema.append(str(arg))
                device_mesh = arg.mesh
            elif isinstance(arg, OpStrategy):
                if len(arg.strategies) != 1:
                    raise AssertionError
                args_schema.append(_pretty_print_spec(arg.strategies[0].output_specs))
                device_mesh = arg.mesh
            elif isinstance(arg, TupleStrategy):
                first_op_strategy = arg.children[0]
                if not isinstance(first_op_strategy, OpStrategy):
                    raise AssertionError
                device_mesh = first_op_strategy.mesh
                args_schema.append(str(arg))
            else:
                args_schema.append(str(arg))

        return f"{self.op}({', '.join(args_schema)}) on {device_mesh})"

    def __post_init__(self) -> None:
        _DTensor_OpSchema_post_init(self)

    def arg_type_tensor_or_tensor_list_like(self, arg: object) -> bool:
        is_tensor = isinstance(arg, DTensorSpec)
        if is_tensor:
            return True

        if not isinstance(arg, list):
            return False

        return all(isinstance(e, DTensorSpec) or e is None for e in arg)

    def return_type_tuple_tensor_like(self) -> bool:
        # all dispatch ops could only return Tuple[Tensor] or have None/ints/floats
        # in the tuple, but the first element must be a Tensor, so this check is enough
        return_types = self.op._schema.returns
        return len(return_types) > 1 and isinstance(
            return_types[0].type, torch.TensorType
        )

    def return_type_list_tensor_like(self) -> bool:
        # returns True if the return type is a List
        return_types = self.op._schema.returns
        return len(return_types) == 1 and isinstance(
            return_types[0].type, torch.ListType
        )

    def return_type_tensor(self) -> bool:
        return_types = self.op._schema.returns
        # all dispatch ops only return Tensor or Tuple[Tensor] for tensor like
        # return types, so this check is enough for tensor like types
        return len(return_types) > 0 and isinstance(
            return_types[0].type, torch.TensorType
        )

    def get_mesh_from_args(self, validate: bool = True) -> DeviceMesh:
        """
        This util can be used to get a mesh from the OpSchema that contains multiple
        DTensors as arguments. When `validate` is True, it will try to validate that all the
        arguments have the same mesh to avoid unexpected cross mesh errors.

        NOTE: this util currently does not handle TupleStrategy when `validate=True`,
        this is because for TupleStrategy there could be different types of checks, i.e.:
            - for stack and cat like op, we need to check within a TupleStrategy is every
              input is on the same mesh
            - for foreach like ops we need to check "zipped" inputs are on the same mesh
              for each index.
        """
        mesh = None
        # Scan all args to find the first DTensorSpec/OpStrategy (not just the first arg)
        for arg in self.args_schema:
            if isinstance(arg, (DTensorSpec, OpStrategy)):
                mesh = arg.mesh
                break
            elif isinstance(arg, (list, tuple, TupleStrategy)):
                # Scan all elements in the list/tuple, not just the first one,
                # to handle cases like List[Optional[Tensor]] where first elem may be None
                elems = arg.children if isinstance(arg, TupleStrategy) else arg
                for elem in elems:
                    if isinstance(elem, (DTensorSpec, OpStrategy)):
                        mesh = elem.mesh
                        break
                if mesh is not None:
                    break
        if mesh is None:
            raise ValueError(f"Cannot find device mesh from args for op : {self.op}.")

        if validate:
            for arg in self.args_schema[1:]:
                if isinstance(arg, (DTensorSpec, OpStrategy)) and arg.mesh != mesh:
                    raise RuntimeError(
                        f"DTensor does not support cross-mesh operation on {self.op}! "
                        f"Got meshes: {mesh} {arg.mesh}. "
                        f"Please make sure all the arguments have the same DeviceMesh."
                    )

        return mesh

    def is_inplace_op(self) -> bool:
        # simple analysis of function schema to determine
        # if this is an inplace variant, it might not
        # be entirely correct, but it's good enough for now.
        return self.op._schema.name[-1] == "_"

    def is_out_variant_op(self) -> bool:
        # simple analysis of function schema to determine
        # if this is an out variant.
        return any(argument.is_out for argument in self.op._schema.arguments)

    def is_view_op(self) -> bool:
        return self.op._schema._is_view_op()

    def _recompute_comparison_key(self) -> None:
        _DTensor_OpSchema_recompute_comparison_key(self)

    def __hash__(self) -> int:
        return hash(self._comparison_key)

    def __eq__(self, other: object) -> bool:
        # early return checks
        if not isinstance(other, OpSchema):
            return False

        if self.op != other.op:
            return False

        if len(self.args_schema) != len(other.args_schema):
            return False

        return self._comparison_key == other._comparison_key

    def gen_fake_args(self) -> ArgsType:
        """
        gen_fake_args: generate fake args for the operator, this is mainly used
            by sharding propagation rules to generate fake args for the operator
            to run the local tensor operator and get the output spec.
        """
        return tree_map_only(
            DTensorSpec,
            _rebuild_tensor_from_dtensor_meta,
            self.args_schema,
            is_leaf=lambda x: isinstance(x, DTensorSpec),
        )

    def gen_fake_kwargs(self) -> KwargsType:
        """
        gen_fake_kwargs: generate fake kwargs for the operator, this is mainly used
            by sharding propagation rules to generate fake kwargs for the operator
            to run the local tensor operator and get the output spec.
        """
        return tree_map_only(
            DTensorSpec,
            _rebuild_tensor_from_dtensor_meta,
            self.kwargs_schema,
            is_leaf=lambda x: isinstance(x, DTensorSpec),
        )

    def _inplace_rewrap_schema_suggestion(self, origin_schema: "OpSchema") -> None:
        suggestion_args_spec = self.args_spec
        new_arg_schema: list[object] = []
        idx_of_args_spec = 0
        if (
            origin_schema.schema_info is not None
            and origin_schema.schema_info.needs_pytree
        ):
            args_schema: Sequence[Any] = tree_leaves(origin_schema.args_schema)
        else:
            args_schema = origin_schema.args_schema
        for arg in args_schema:
            if isinstance(arg, DTensorSpec):
                new_arg_schema.append(suggestion_args_spec[idx_of_args_spec])
                idx_of_args_spec += 1
            else:
                new_arg_schema.append(arg)
        self.args_schema = tuple(new_arg_schema)
        self.kwargs_schema = origin_schema.kwargs_schema
        self._recompute_comparison_key()


@dataclass
class OutputSharding:
    """
    OutputSharding is a data class that is used by the sharding propagation,
    it could set the output_spec upon successful propagation. If needs_redistribute
    is set to True, a redistribute_schema would be returned together to indicate
    the input arguments needs to be redistributed before the op execution.

    NOTE: the redistribute_schema generated by sharding propagation should be
    exactly the same as the operator OpSchema, except the DTensorSpecs
    """

    # specifies the output sharding pattern
    output_spec: OutputSpecType
    # schema for redistribution if needed
    redistribute_schema: OpSchema | None = None
    # flag indicating if inputs need redistribution
    needs_redistribute: bool = False
    # flag to use values from `redistribute_schema`
    use_val_from_redistribute_schema: bool = False

    @cached_property
    def mesh(self):
        if isinstance(self.output_spec, DTensorSpec):
            return self.output_spec.mesh
        elif isinstance(self.output_spec, tuple):
            out_spec = self.output_spec[0]
            if isinstance(out_spec, DTensorSpec):
                return out_spec.mesh
            else:
                raise ValueError(f"Unknown output spec type: {type(out_spec)}")
        else:
            raise ValueError(f"Unknown output spec type: {type(self.output_spec)}")


@dataclass
class OpInfo:
    """
    All Runtime Op execution info are packed here
    """

    # The first compute device mesh recorded from args
    # NOTE: one op could have multiple meshes from its args. We just record the first
    # mesh here to check if current rank should participate in computation or not.
    compute_mesh: DeviceMesh

    # compete runtime operator infos
    # NOTE: schema can be None due to C++ fast path optimization. When the C++
    # dispatch layer (dispatchDTensorOp in python_variable.cpp) finds a cached
    # sharding decision, it skips creating the full OpSchema to reduce CPU overhead.
    # In this case, OpInfo is created with create_schema=False, setting schema to None.
    # The operator information is still available through output_sharding.redistribute_schema
    # when redistribution is needed.
    schema: OpSchema | None
    flat_args_schema: list[object]
    local_args: Sequence[object]
    local_kwargs: dict[str, object]
    args_tree_spec: TreeSpec | None = None

    # the output sharding info
    output_sharding: OutputSharding | None = None
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import contextlib
import dataclasses
import itertools
import logging
import math
import weakref
from collections import defaultdict
from collections.abc import Sequence
from functools import cache
from typing import Any, cast, TypedDict

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.tensor._api as dtensor
from torch.distributed._functional_collectives import _are_we_tracing
from torch.distributed._mesh_layout import _MeshLayout
from torch.distributed.tensor._collective_utils import one_step_redistribute_cost
from torch.distributed.tensor._dtensor_spec import (
    _StridedShardNotDecodableError,
    DTensorSpec,
    ShardOrder,
    ShardOrderEntry,
    TensorMeta,
)
from torch.distributed.tensor._utils import assert_no_mixed_partial_types
from torch.distributed.tensor.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import (
    _is_shard_like,
    _StridedShard,
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.types import FloatLikeType, IntLikeType
from torch.utils._debug_mode import get_active_debug_mode
from torch.utils._typing_utils import not_none


logger = logging.getLogger(__name__)

# Global configuration flag to control the redistribution planning strategy.
# When True, forces the graph-based algorithm using Dijkstra's shortest path.
# When False, prefers the greedy algorithm for faster planning. Uses the graph-based algorithm
# only when necessary to support strided-shard redistribution
_FORCE_MIN_COST_REDISTRIBUTION_PLAN: bool | None = None

# Global kill switch to disable the transform optimization pass in
# _optimize_transform_infos.  When True, the optimization that merges
# consecutive same-type collectives into flattened operations is skipped,
# and the unmodified transform_infos list is returned as-is.
_DISABLE_REDISTRIBUTE_TRANSFORM_OPTIMIZATION: bool = False


def _redistribute_cost_sort_key(cost: FloatLikeType) -> float:
    if isinstance(cost, float):
        return cost
    if isinstance(cost, torch.SymFloat):
        expr = cost.node.expr
    else:
        return float(cost)

    free_symbols = expr.free_symbols
    if not free_symbols:
        return float(expr)

    shape_env = not_none(cost.node.shape_env)
    hint_overrides = shape_env.var_to_hint_override
    if free_symbols.issubset(hint_overrides):
        return float(expr.xreplace(hint_overrides))

    return float(shape_env.optimization_hint(expr, fallback=None))


@contextlib.contextmanager
def use_min_cost_redistribution_plan(enabled: bool = True):
    """
    Context manager to control the redistribution planning strategy for DTensor operations.

    This context manager allows you to choose between two algorithms for computing the
    sequence of collective operations needed to redistribute a DTensor from one placement
    to another:

    - **Graph-based**: Uses Dijkstra's algorithm to find the minimum-cost path
      through all possible placement transformations. This approach considers the global
      cost of all collective operations and finds the optimal sequence. Best for complex
      redistribution patterns where reducing communication cost and memory overhead is critical.

    - **Greedy**: Uses a heuristic approach that makes locally optimal choices
      at each step. This is faster to compute but may not produce the globally optimal
      transformation sequence. Best for simple redistribution patterns or when planning
      speed is more important than optimal communication.

    **Default Behavior (without this context manager):**

    When this context manager is NOT used, the algorithm selection follows this priority:

    1. **Non-default shard orders**
       → Always use graph-based algorithm (required for correctness)

    2. **Explicit `use_graph_based_transform` parameter** to `_gen_transform_infos_non_cached`
       → Use the specified algorithm (True = graph-based, False = greedy)

    3. **No explicit parameter** (default case)
       → Use greedy algorithm for faster planning

    **Behavior with this context manager:**

    This context manager overrides the default selection by setting the global flag
    `_FORCE_MIN_COST_REDISTRIBUTION_PLAN`, which takes precedence over the explicit
    `use_graph_based_transform` parameter (but not over non-default shard order requirements).

    **Cache Considerations:**

    The redistribution planner caches transform info for performance via the `@cache`
    decorator on `_gen_transform_infos`. If you need to change the algorithm selection
    for the same input specs, clear the cache using `_gen_transform_infos.cache_clear()`
    to ensure the new setting takes effect and doesn't reuse cached results from a
    previous run.

    Args:
        enabled (bool): If True, forces the use of the graph-based algorithm.
                       If False, forces the use of the greedy algorithm.
                       Default: True
    """
    global _FORCE_MIN_COST_REDISTRIBUTION_PLAN

    old_value = _FORCE_MIN_COST_REDISTRIBUTION_PLAN
    _FORCE_MIN_COST_REDISTRIBUTION_PLAN = enabled
    try:
        yield
    finally:
        _FORCE_MIN_COST_REDISTRIBUTION_PLAN = old_value


@contextlib.contextmanager
def disable_redistribute_transform_optimization(disabled: bool = True):
    """
    Context manager to disable the transform optimization pass that merges
    consecutive same-type collectives into single flattened operations.

    When the optimization is disabled, ``_optimize_transform_infos`` becomes a
    no-op and returns the original list of ``_TransformInfo`` objects unchanged.
    This is useful for debugging or isolating issues related to the flattened
    collective merging logic.

    The flag can also be set directly::

        torch.distributed.tensor._redistribute._DISABLE_REDISTRIBUTE_TRANSFORM_OPTIMIZATION = True

    Args:
        disabled (bool): If True (default), disables the optimization.
                         If False, explicitly enables it (the normal default).
    """
    global _DISABLE_REDISTRIBUTE_TRANSFORM_OPTIMIZATION

    old_value = _DISABLE_REDISTRIBUTE_TRANSFORM_OPTIMIZATION
    _DISABLE_REDISTRIBUTE_TRANSFORM_OPTIMIZATION = disabled
    try:
        yield
    finally:
        _DISABLE_REDISTRIBUTE_TRANSFORM_OPTIMIZATION = old_value


@dataclasses.dataclass(frozen=True, slots=True)
class _TransformInfo:
    mesh_dim: int
    src_dst_placements: tuple[Placement, Placement]
    # logical_shape on this mesh dimension
    logical_shape: Sequence[IntLikeType]

    def __post_init__(self):
        if self.mesh_dim < 0:
            raise AssertionError
        if self.src_dst_placements[0] == self.src_dst_placements[1]:
            raise AssertionError(
                "TransformInfo should only be created if it is an op with some effect, not a no-op"
            )

    def _comm_type_key(self) -> str | None:
        """
        Return a key for grouping transforms by communication type.

        Returns None for local ops (no communication needed), or a string
        that identifies the collective type for potential grouping/merging.
        """
        src, dst = self.src_dst_placements
        if src.is_partial() and dst.is_replicate():
            return "all_reduce"
        elif src.is_partial() and _is_shard_like(dst):
            return "reduce_scatter"
        elif _is_shard_like(src) and dst.is_replicate():
            return "all_gather"
        elif _is_shard_like(src) and _is_shard_like(dst):
            return "all_to_all"
        else:
            # Local ops (Replicate->Shard, Replicate->Partial, noop, etc.)
            return None


@dataclasses.dataclass(frozen=True, slots=True)
class _FlattenedTransformInfo(_TransformInfo):
    """
    Represents a flattened transform that combines multiple mesh dimensions
    into a single collective operation using a flattened DeviceMesh.

    Note: inherits the fields from _TransformInfo. Gets an __init__ with parent fields, followed by child fields,
    and runs parent validation (post_init)
    """

    # The flattened DeviceMesh to use for the collective operation
    mesh: DeviceMesh
    # The mesh dimensions from the original mesh that are being flattened (for debugging)
    original_mesh_dims: tuple[int, ...]
    # Scale factor for merged sum/avg partials (product of avg mesh dim sizes)
    # When merging sum/avg partials, we use sum for the collective and divide by this afterward
    avg_scale: int | None = None

    def __post_init__(self) -> None:
        _TransformInfo.__post_init__(self)
        if self.avg_scale is not None:
            if not self.avg_scale > 1:
                raise AssertionError(
                    f"avg_scale must be > 1 if set, got {self.avg_scale}"
                )


def _update_shard_order_and_placements(
    transform_info: _TransformInfo,
    current_placements: list[Placement],
    shard_order_dict: dict[int, list[int]],
) -> None:
    """
    Update current_placements and shard_order_dict in-place to reflect the
    effect of a single transform step.
    """
    src_placement, dst_placement = transform_info.src_dst_placements

    if isinstance(transform_info, _FlattenedTransformInfo):
        mesh_dims = transform_info.original_mesh_dims
    else:
        mesh_dims = (transform_info.mesh_dim,)

    if isinstance(src_placement, Shard | _StridedShard):
        src_dim = src_placement.dim  # type: ignore[attr-defined]
        removed_dim = set()
        for _ in mesh_dims:
            if len(shard_order_dict[src_dim]) == 0:
                raise ValueError(
                    "Invalid shard_order update. No entries left to pop for src_dim "
                    f"{src_dim}. transform_info={transform_info}, "
                    f"current_placements={current_placements}, "
                    f"shard_order={shard_order_dict}"
                )
            removed_dim.add(shard_order_dict[src_dim].pop())

        if not set(mesh_dims) == removed_dim:
            raise ValueError(
                "Mismatch between expected and removed mesh dims during shard_order "
                "update. Expected to remove "
                f"{set(mesh_dims)}, but removed {removed_dim}. "
                f"transform_info={transform_info}, "
                f"current_placements={current_placements}, "
                f"shard_order={shard_order_dict}"
            )
    if isinstance(dst_placement, Shard | _StridedShard):
        dst_dim = dst_placement.dim  # type: ignore[attr-defined]
        if dst_dim not in shard_order_dict:
            shard_order_dict[dst_dim] = []
        for mesh_dim in mesh_dims:
            shard_order_dict[dst_dim].append(mesh_dim)

    for mesh_dim in mesh_dims:
        current_placements[mesh_dim] = dst_placement


def _get_flattened_mesh_by_layout_impl(
    mesh: DeviceMesh, mesh_dims: tuple[int, ...]
) -> DeviceMesh | None:
    """
    Query for an explicitly created flattened mesh using layout comparison.

    Searches root_mesh._flatten_mapping for a mesh whose layout matches
    the expected flattened layout for the given dims. Pure Python layout math.
    """
    root_mesh = mesh._get_root_mesh()
    mesh_dim_names = mesh.mesh_dim_names

    if mesh_dim_names is None:
        return None

    # Convert mesh dim indices to dim names
    dim_names = tuple(mesh_dim_names[i] for i in mesh_dims)

    # Compute expected layout WITHOUT creating a submesh (avoids tracing issues)
    # _get_slice_mesh_layout does pure layout math, no tensor operations
    sliced_layout = mesh._get_slice_mesh_layout(dim_names)
    expected_layout = _MeshLayout([sliced_layout.collapse()])

    # Search existing flattened meshes by comparing layouts
    for flattened_mesh in root_mesh._flatten_mapping.values():
        if flattened_mesh._layout == expected_layout:
            return flattened_mesh

    return None


def _get_flattened_mesh_by_layout(
    mesh: DeviceMesh, mesh_dims: tuple[int, ...]
) -> DeviceMesh | None:
    """
    Query for an explicitly created flattened mesh using layout comparison.

    When tracing with compile_on_one_rank, delegates to a custom op so the
    flattened mesh appears as a call_function node derived from mesh (a graph
    input) rather than as a get_attr constant holding an unpicklable
    ProcessGroup.
    """
    import torch.compiler.config

    if _are_we_tracing() and torch.compiler.config.compile_on_one_rank:
        # Pre-check: the custom op can't return None (torch.library doesn't
        # support Optional opaque return types), so guard here first.
        if _get_flattened_mesh_by_layout_impl(mesh, mesh_dims) is None:
            return None
        from torch.distributed._ops import device_mesh as _  # noqa: F401

        return torch.ops.device_mesh._get_flattened_submesh(mesh, list(mesh_dims))

    return _get_flattened_mesh_by_layout_impl(mesh, mesh_dims)


# Track (mesh_hash, mesh_dims, reason) we've already warned about to avoid repeated warnings
_warned_flatten_issues: set[tuple[int, tuple[int, ...], str]] = set()


def _warn_flatten_optimization_not_possible(
    device_mesh: DeviceMesh,
    mesh_dims: tuple[int, ...],
    src_placements: tuple[Placement, ...],
    dst_placements: tuple[Placement, ...],
    num_ops: int,
    comm_type: str,
    reason: str,
) -> None:
    """
    Warn once per (mesh, dims, reason) about inability to flatten operations.

    Args:
        device_mesh: The device mesh being used
        mesh_dims: Tuple of mesh dimensions that could not be flattened
        src_placements: Source placements for the redistribution
        dst_placements: Target placements for the redistribution
        num_ops: Number of sequential operations that will be performed
        comm_type: Type of collective operation (e.g., "reduce_scatter")
        reason: Either "no_flattened_mesh" or "uneven_tensor_shape"
    """
    cache_key = (hash(device_mesh), mesh_dims, reason)
    if cache_key in _warned_flatten_issues:
        return
    _warned_flatten_issues.add(cache_key)

    mesh_dim_names = device_mesh.mesh_dim_names
    if mesh_dim_names is not None:
        dim_names = [mesh_dim_names[d] for d in mesh_dims]
        dims_str = ", ".join(f'"{name}"' for name in dim_names)
    else:
        dims_str = f"dims {', '.join(str(d) for d in mesh_dims)} of {device_mesh}"

    common_warning = (
        "While redistributing from %s to %s, %d sequential %s "
        "operations will be performed. This is suboptimal: "
        "multiple collective operations have higher latency "
        "(separate kernel launches and synchronization points) "
        "and may give inconsistent results between ranks due to different reduction orders. %s"
    )

    if reason == "no_flattened_mesh":
        reason_msg = f"To optimize, flatten mesh dimensions [{dims_str}] so DTensor can use a single operation instead."
    elif reason == "uneven_tensor_shape":
        reason_msg = (
            " Unfortunately, because the tensor dimension is not evenly divisible by the product of "
            "the mesh dim sizes that would need to be flattened for the optimization to work, it can not be optimized.",
        )
    elif reason == "non_ascending_mesh_dims":
        reason_msg = (
            f"it is not possible to merge non-ascending order {comm_type} operations."
        )
    else:
        raise AssertionError(f"Unexpected reason: {reason}")

    logger.warning(
        common_warning, src_placements, dst_placements, num_ops, comm_type, reason_msg
    )


def _optimize_transform_infos(
    transform_infos: list[_TransformInfo],
    device_mesh: DeviceMesh,
    src_placements: tuple[Placement, ...],
    dst_placements: tuple[Placement, ...],
) -> list[_TransformInfo | _FlattenedTransformInfo]:
    """
    Optimize transform infos by merging consecutive same-type collectives into
    a single flattened operation when a matching flattened DeviceMesh exists.

    Merging requirements:
    - Operations must be consecutive in the transform list (no reordering).
      Notably, redistributing from P, P, P -> R, S, R is not optimized here and cannot be optimized due to
      optimization needing to fuse non-contiguous reductions, leaving this pattern vulnerable to numerics issues and
      suboptimal perf
    - Operations must have the same comm type (e.g., all allgather or all reduce_scatter)
    - Operations must have identical src_dst_placements (e.g., can't merge
      Partial->Shard(0) with Partial->Shard(1))
    - A flattened mesh covering the relevant dimensions must exist
    - For reduce_scatter, tensor dim must be evenly divisible by flattened mesh size

    For nested sharding, the merged operation uses the logical_shape from the
    outermost mesh dimension (smallest mesh_dim index) which represents the
    global tensor shape needed for correct padding/unpadding.

    TODO:
    - all_to_all operations are excluded from merging, but it may be possible to merge them in some cases.

    """
    if len(transform_infos) < 2:
        return transform_infos

    if _DISABLE_REDISTRIBUTE_TRANSFORM_OPTIMIZATION:
        return transform_infos

    # Comm types that are safe to merge (all_to_all excluded for now)
    MERGEABLE_COMM_TYPES = frozenset({"all_gather", "all_reduce", "reduce_scatter"})

    def is_mergeable(key: str | None) -> bool:
        """Check if a comm type key represents a mergeable operation."""
        return key in MERGEABLE_COMM_TYPES

    def are_placements_mergeable(
        p1: tuple[Placement, Placement], p2: tuple[Placement, Placement]
    ) -> bool:
        """
        Check if two src_dst_placements can be merged.

        Allows merging of Partial("sum") and Partial("avg") since they can be
        combined: perform sum reduction, then scale by avg mesh dims afterward.
        """
        if p1 == p2:
            return True

        src1, dst1 = p1
        src2, dst2 = p2

        # Destinations must match exactly
        if dst1 != dst2:
            return False

        # Both sources must be partial
        if not (src1.is_partial() and src2.is_partial()):
            return False

        # Only sum and avg can be merged (both use sum reduction, avg just scales)
        partial1 = cast(Partial, src1)
        partial2 = cast(Partial, src2)
        mergeable_reduce_ops = {"sum", "avg"}
        return (
            partial1.reduce_op in mergeable_reduce_ops
            and partial2.reduce_op in mergeable_reduce_ops
        )

    def try_create_flattened(
        infos: list[_TransformInfo],
    ) -> tuple[_FlattenedTransformInfo | None, str | None]:
        """
        Try to create a flattened transform from 2+ same-type transforms.

        Returns (result, failure_reason) where:
        - result is the FlattenedTransformInfo if successful, None otherwise
        - failure_reason is None if successful, or one of:
          - "too_few_transforms": Less than 2 transforms provided
          - "no_flattened_mesh": No flattened mesh exists for the required dimensions
          - "uneven_tensor_shape": For reduce_scatter, tensor dim not evenly divisible
        """
        if len(infos) < 2:
            return None, "too_few_transforms"

        # All transforms must have mergeable src_dst_placements
        # (e.g., can't merge Partial->Shard(0) with Partial->Shard(1))
        first_placements = infos[0].src_dst_placements
        comm_type = infos[0]._comm_type_key()
        if not all(
            are_placements_mergeable(info.src_dst_placements, first_placements)
            for info in infos
        ):
            raise AssertionError(
                "All transforms must have mergeable src_dst_placements"
            )
        mesh_dims = tuple(info.mesh_dim for info in infos)
        sorted_mesh_dims = tuple(sorted(mesh_dims))

        # For reduce_scatter and all_gather, order matters for correctness.
        # Flattened meshes only exist for ascending dim order.
        if comm_type == "reduce_scatter":
            # For reduce_scatter: the transform order determines the operation sequence.
            # If transforms are in order (1, 0) but flattened mesh is (0, 1), we can't flatten.
            if mesh_dims != sorted_mesh_dims:
                return None, "non_ascending_mesh_dims"
        elif comm_type == "all_gather":
            # For all_gather: transforms come from planner in innermost-to-outermost order
            # (descending mesh dims for ascending shard order). If transforms are not in
            # descending order, the shard order isn't ascending and we can't flatten.
            if mesh_dims != sorted_mesh_dims[::-1]:
                return None, "non_ascending_mesh_dims"
        # Use sorted dims for mesh lookup (required by DeviceMesh API)
        flattened_mesh = _get_flattened_mesh_by_layout(device_mesh, sorted_mesh_dims)
        if flattened_mesh is None:
            return None, "no_flattened_mesh"

        # For nested sharding, each transform has a different logical_shape.
        # We need the outermost transform's logical_shape, which represents the
        # tensor shape before any of the transforms in this group are applied.
        # The outermost transform has the largest logical_shape on the affected
        # tensor dimension (least divided by prior shards).
        src, dst = first_placements
        if comm_type == "all_gather":
            # S->R (all_gather): affected dim is the source shard dim
            affected_dim = cast(Shard, src).dim
            outermost_info = max(infos, key=lambda x: x.logical_shape[affected_dim])
        elif comm_type == "reduce_scatter":
            affected_dim = cast(Shard, dst).dim
            outermost_info = max(infos, key=lambda x: x.logical_shape[affected_dim])
            tensor_dim_size = outermost_info.logical_shape[affected_dim]
            effective_shard_mesh_size = math.prod(
                device_mesh.size(info.mesh_dim) for info in infos
            )
            # For reduce_scatter (Partial -> Shard), we cannot flatten if the tensor
            # dimension is not evenly divisible by the flattened mesh size.
            # The effective size is the product of mesh sizes for dims being transformed
            # (not all dims with matching placement - intervening shards are already
            # accounted for in logical_shape).
            if tensor_dim_size % effective_shard_mesh_size != 0:
                return None, "uneven_tensor_shape"
        elif comm_type == "all_reduce":
            # no shape change, any info works
            outermost_info = infos[0]
        else:
            raise NotImplementedError(
                f"Unsupported comm type for try_create_flattened: {comm_type}"
            )

        # For mixed sum/avg partials: use sum for the collective, compute avg scale
        avg_scale = None
        merged_src = src
        if src.is_partial():
            scale = math.prod(
                device_mesh.size(info.mesh_dim)
                for info in infos
                if cast(Partial, info.src_dst_placements[0]).reduce_op == "avg"
            )
            if scale > 1:
                avg_scale = scale
                merged_src = Partial("sum")

        merged_placements = (merged_src, dst)

        return (
            _FlattenedTransformInfo(
                mesh_dim=0,
                src_dst_placements=merged_placements,
                logical_shape=outermost_info.logical_shape,
                mesh=flattened_mesh,
                original_mesh_dims=sorted_mesh_dims,
                avg_scale=avg_scale,
            ),
            None,
        )

    # Merge consecutive same-type operations (without reordering)
    result: list[_TransformInfo | _FlattenedTransformInfo] = []
    i = 0

    while i < len(transform_infos):
        info = transform_infos[i]
        current_key = info._comm_type_key()

        # Only try to merge if this is a mergeable comm type
        if not is_mergeable(current_key):
            result.append(info)
            i += 1
            continue

        # Collect consecutive transforms with mergeable src_dst_placements
        # (not just same comm type - e.g., Partial->Shard(0) vs Partial->Shard(1) can't merge)
        # Note: sum/avg partials can be merged since they use the same reduction
        current_placements = info.src_dst_placements
        group: list[_TransformInfo] = [info]
        j = i + 1
        while (
            j < len(transform_infos)
            and is_mergeable(transform_infos[j]._comm_type_key())
            and are_placements_mergeable(
                transform_infos[j].src_dst_placements, current_placements
            )
        ):
            group.append(transform_infos[j])
            j += 1

        # Try to flatten the group
        flattened, failure_reason = try_create_flattened(group)
        if flattened is not None:
            result.append(flattened)
        else:
            # Can't flatten - add individually and warn once if applicable
            result.extend(group)
            # Warn for reasons that indicate a real optimization opportunity was missed
            if failure_reason in (
                "no_flattened_mesh",
                "uneven_tensor_shape",
                "non_ascending_mesh_dims",
            ):
                mesh_dims = tuple(sorted(g.mesh_dim for g in group))
                _warn_flatten_optimization_not_possible(
                    device_mesh,
                    mesh_dims,
                    src_placements,
                    dst_placements,
                    len(group),
                    current_key,  # type: ignore[arg-type]
                    failure_reason,
                )

        i = j
    logger.debug(
        "_optimize_transform_infos original: %s, optimized: %s", transform_infos, result
    )

    return result


# Global cache for DTensorRedistributePlanner instances
_planner_cache: dict[
    tuple[weakref.ReferenceType[DeviceMesh], TensorMeta],
    "DTensorRedistributePlanner",
] = {}


def get_redistribute_planner(
    device_mesh: DeviceMesh,
    dtensor_meta: TensorMeta,
) -> "DTensorRedistributePlanner":
    """
    Factory function to get or create a DTensorRedistributePlanner instance.
    This function provides transparent caching of planner instances based on
    device mesh and dtensor meta. Multiple calls with the same parameters
    will return the same cached instance for better performance.
    Args:
        device_mesh: The device mesh for the planner
        dtensor_meta: TensorMeta of the DTensor to redistribute
    Returns:
        A DTensorRedistributePlanner instance (potentially cached)
    """
    if _are_we_tracing():
        return DTensorRedistributePlanner(device_mesh, dtensor_meta)

    cache_key = (weakref.ref(device_mesh), dtensor_meta)
    if cache_key not in _planner_cache:
        planner = DTensorRedistributePlanner(device_mesh, dtensor_meta)
        _planner_cache[cache_key] = planner

    return _planner_cache[cache_key]


def clear_redistribute_planner_cache() -> None:
    """Clear the cache of DTensorRedistributePlanner instances."""
    _planner_cache.clear()


class DTensorRedistributePlanner:
    """
    This class is used to plan the collective calls to transform the local shard
    of the DTensor from its current spec to the target spec.
    Suppose there are N tensor dimensions and M mesh dimensions, the total
    possible state size will be (N+2)*M*M!.
    Note: Use get_redistribute_planner() factory function instead of direct
    instantiation for automatic caching.
    """

    @dataclasses.dataclass(frozen=True, slots=True)
    class DistState:
        placements: tuple[Placement, ...]
        tensor_dim_to_mesh_dim: ShardOrder
        _hash: int | None = dataclasses.field(
            default=None, init=False, repr=False, compare=False
        )

        def __str__(self):
            return DTensorSpec.format_shard_order_str(
                self.placements,
                self.tensor_dim_to_mesh_dim,
            )

        def __repr__(self):
            return self.__str__()

        def __post_init__(self):
            # precompute hash after all attributes are set
            object.__setattr__(
                self,
                "_hash",
                self._compute_hash(),
            )

        def __hash__(self) -> int:
            return self._hash if self._hash is not None else self._compute_hash()

        def _compute_hash(self) -> int:
            return hash(
                (
                    self.placements,
                    self.tensor_dim_to_mesh_dim,
                )
            )

        def __eq__(self, other: object) -> bool:
            if not isinstance(other, DTensorRedistributePlanner.DistState):
                return False
            if self._hash != other._hash:
                return False
            return (
                self.placements,
                self.tensor_dim_to_mesh_dim,
            ) == (
                other.placements,
                other.tensor_dim_to_mesh_dim,
            )

    def _to_tuple(self, x):
        """Convert a nested list structure to a nested tuple structure."""
        if isinstance(x, list | tuple):
            return tuple(self._to_tuple(item) for item in x)
        return x

    @staticmethod
    def _dict_to_ShardOrder(x: dict[int, list[int]]) -> ShardOrder:
        """Convert dict to ShardOrder"""
        return tuple(
            ShardOrderEntry(tensor_dim=key, mesh_dims=tuple(value))
            for key, value in sorted(x.items())
            if value
        )

    @staticmethod
    def _ShardOrder_to_dict(x: ShardOrder) -> dict[int, list[int]]:
        """Convert ShardOrder to dict with tensor dim as key"""
        tensor_mesh_dim_dict = defaultdict(list)
        for entry in x:
            tensor_mesh_dim_dict[entry.tensor_dim] = list(entry.mesh_dims)
        return tensor_mesh_dim_dict

    @staticmethod
    def stringify_transform_infos(
        mesh: DeviceMesh,
        transform_infos: Sequence[_TransformInfo],
        src_placement: tuple[Placement, ...],
        src_shard_order: ShardOrder | None = None,
        use_strided_shard_as_shard_order: bool = False,
    ) -> str:
        """
        Generate a string representation of the sequence of state transitions
        (placements and shard orders) as described by the given transform_info.

        Args:
            mesh: The DeviceMesh used for the redistribution.
            transform_infos: A sequence of _TransformInfo objects describing each
                transformation step.
            src_placement: The initial tuple of Placement objects.
            src_shard_order: (Optional) The initial ShardOrder representing
                the mapping of tensor dimensions to mesh dimensions. If None,
                the default shard order is computed from src_placement and mesh.
            use_strided_shard_as_shard_order: If True, normalize _StridedShard
                placements into regular Shard placements with an explicit
                shard_order before stringifying.

        Returns:
            A string showing the sequence of DistState transitions, separated by '->'.
        """
        if len(src_placement) != mesh.ndim:
            raise AssertionError(
                f"dimensions mismatch {len(src_placement)} vs {mesh.ndim}"
            )
        if use_strided_shard_as_shard_order:
            src_placement, src_shard_order = (
                DTensorSpec._normalize_placements_into_shard_order(
                    src_placement, mesh, use_strided_shard_as_shard_order=True
                )
            )
        if src_shard_order is None:
            src_shard_order = DTensorSpec.compute_default_shard_order(src_placement)
        cur_placement = list(src_placement)
        shard_order_dict = DTensorRedistributePlanner._ShardOrder_to_dict(
            src_shard_order
        )
        cur_state = DTensorRedistributePlanner.DistState(
            tuple(cur_placement), src_shard_order
        )
        state_list = [
            cur_state,
        ]
        # Track whether each transition is flattened (for visualization)
        is_flattened_list: list[bool] = []

        for transform_info in transform_infos:
            is_flattened = isinstance(transform_info, _FlattenedTransformInfo)

            _update_shard_order_and_placements(
                transform_info, cur_placement, shard_order_dict
            )

            new_state = DTensorRedistributePlanner.DistState(
                tuple(cur_placement),
                DTensorRedistributePlanner._dict_to_ShardOrder(shard_order_dict),
            )
            state_list.append(new_state)
            is_flattened_list.append(is_flattened)

        # Build the trace string using '-->' for flattened transforms, '->' for regular
        trace_parts = [str(state_list[0])]
        for i, is_flattened in enumerate(is_flattened_list):
            separator = "-->" if is_flattened else "->"
            trace_parts.append(separator)
            trace_parts.append(str(state_list[i + 1]))
        return "".join(trace_parts)

    def __init__(
        self,
        device_mesh: DeviceMesh,
        dtensor_meta: TensorMeta,
    ) -> None:
        """
        Initialize DTensorRedistributePlanner.

        Args:
            device_mesh: The device mesh for this planner
            dtensor_meta: TensorMeta of the DTensor to redistribute
        """
        self.device_mesh = device_mesh
        if not device_mesh._is_current_rank_part_of_mesh():
            raise AssertionError
        if dtensor_meta is None:
            raise AssertionError
        self.dtensor_meta = dtensor_meta
        self.tensor_dimension = len(dtensor_meta.shape)
        self.strided_shard_placements_in_target: set[_StridedShard] = set()
        self.partial_reduce_ops_in_target: set[str] = set()
        self.setup_cost_callbacks()

    def setup_cost_callbacks(
        self,
    ) -> None:
        """
        Set up the cost function for different collective operations.
        Uses communication time estimation based on actual tensor sizes and
        mesh topology for accurate cost modeling.
        """

        def state_to_spec(
            state: DTensorRedistributePlanner.DistState,
        ) -> DTensorSpec:
            return DTensorSpec(
                mesh=self.device_mesh,
                placements=state.placements,
                tensor_meta=self.dtensor_meta,
                shard_order=state.tensor_dim_to_mesh_dim,
                use_strided_shard_as_shard_order=False,
            )

        def cost_function(src_state, dst_state):
            return one_step_redistribute_cost(
                state_to_spec(src_state), state_to_spec(dst_state)
            )

        self.cost_function = cost_function

    def get_next_state(
        self,
        placements: tuple[Placement, ...],
        tensor_mesh_dim_tuple: ShardOrder,
    ) -> dict["DTensorRedistributePlanner.DistState", float]:
        # We map tensor dimensions to device mesh axes, similar to JAX-style
        # sharding representation. Notation:
        # S(<tensor_dim>)[<list_of_device_dims>] means tensor dimension
        # <tensor_dim> is sharded on the listed device mesh axes, where
        # <list_of_device_dims> is sorted by device order.
        #
        # To generalize to arbitrary dimensionality, we use the following notation:
        #   S(a)[x, ...]   : tensor dimension 'a' is sharded on device mesh axes x, ... (variadic, possibly empty)
        #   SS(a)[x, ...]  : _StridedShard on tensor dimension 'a' on device mesh axes x, ... (variadic, possibly empty)
        #   R[...]         : replicated on the listed device mesh axes (possibly empty)
        #   P[...]         : partial on the listed device mesh axes (possibly empty)
        # The ellipsis '...' denotes a variadic wildcard, i.e., zero or more device mesh axes.
        #
        # Below are possible transitions from one sharding state to another.
        # We use `S` for Shard, `SS` for _StridedShard, `R` for Replicate, and `P` for Partial.
        #
        # Case 1. Shard(a) -> Shard(b), use all-to-all (a2a), applies to:
        #   S(a)[..., x] -> S(b)[..., x]
        #   or
        #   S(a)[..., x, y]S(b)[..., z, k] -> S(a)[..., x]S(b)[..., z, k, y]
        #   where device order of 'y' > device order of 'z' and 'k'
        #
        # Case 2. Shard() -> Replicate(), use all-gather, applies to:
        #   S(a)[..., x, y, z] -> S(a)[..., x, y]
        #
        # Case 3. Partial() -> Replicate(), use all-reduce, applies to:
        #   P[..., x, y] -> P[..., y] or P[..., x]
        #   Note: this case can be disabled because all-reduce technically is not
        #   a primitive since it combines a reduce-scatter + all-gather.
        #
        # Case 4. Replicate() -> Shard(), use chunk, applies to:
        #   S(a)[..., z] -> S(a)[..., z, y] (`a` can be any tensor dim). Note that
        #   'y' must be after 'z'.
        #
        # Case 5. Partial() -> Shard(), use reduce-scatter, applies to:
        #  P[..., x, y] -> P[..., x]S(a)[..., y] or P[..., x, y] -> P[..., y]S(a)[..., x]
        #
        # Case 6. Replicate() -> Partial(), local math op, applies to:
        #   R* -> P[..., x]
        #
        # Case 7. Shard() -> Partial("sum"), local zero-fill, generated only as
        # a destination-directed transition in find_min_cost_path(), applies to:
        #   S(a)[..., x] -> P[..., x]
        # This transition is available only for explicit redistribution.
        #
        # Strategy planning assigns it infinite cost. A zero communication cost
        # would ignore the logical-shape allocation, zero-fill, and shard copy,
        # causing the planner to over-prefer it.
        #
        # Transform planning assigns an explicit request zero search cost because
        # it requires no communication. This difference is intentional: the
        # transform planner supports strategy-selected transitions plus additional
        # explicit-only transitions.

        # (TODO) Case 8. _StridedShard(a) -> Shard(b), use all-to-all (a2a), applies to:
        #   SS(a)[..., x] -> S(b)[..., x]
        #
        # Case 9. _StridedShard() -> Replicate(), use all-gather, applies to:
        #   SS(a)[..., x, y, z] -> SS(a)[..., x, y]
        #
        # (TODO) Case 10. Shard(a) -> _StridedShard(b), use all-to-all (a2a), applies to:
        #   S(a)[..., x] -> SS(b)[..., x]
        #
        # (TODO) Case 11. Partial() -> _StridedShard(), use reduce-scatter, applies to:
        #   P[..., x, y] -> P[..., x]SS(a)[..., y] or P[..., x, y] -> P[..., y]SS(a)[..., x]
        #
        # Case 12. Replicate() -> _StridedShard(), use chunk, applies to:
        #   R* -> SS(a)[..., x]
        #
        # NB: Regarding `_StridedShard``, we only allow changing `Replicate` into
        # `_StridedShard` with the same tensor dim and split_factor that occurs in the
        # target placement.
        #
        # (TODO) Verify device order impact in Partial placement. We may need to handle
        # device ordering for Partial also.

        # list of [DistState, cost]
        all_next_state: dict[DTensorRedistributePlanner.DistState, float] = {}

        tensor_mesh_dim_dict = DTensorRedistributePlanner._ShardOrder_to_dict(
            tensor_mesh_dim_tuple
        )
        cur_dist_state = self.DistState(
            self._to_tuple(placements),
            tensor_mesh_dim_tuple,
        )
        ######################################################################
        # handle case 1: Shard(a) -> Shard(b)
        # For S(a), S(b), only the last device order of S(a) and S(b) can be a2a
        # interchangeably.

        # convert sparse tuple
        for entry in tensor_mesh_dim_tuple:
            src_tensor_dim = entry.tensor_dim
            src_mesh_dim = tensor_mesh_dim_dict[src_tensor_dim][-1]
            if not isinstance(placements[src_mesh_dim], Shard):
                # skip special case like `_StridedShard`
                continue
            for dst_tensor_dim in range(self.tensor_dimension):
                if src_tensor_dim == dst_tensor_dim:
                    continue
                # try move the last sharded device dim from
                # Shard(src_tensor_dim) to Shard(dst_tensor_dim)
                move_mesh_dim = tensor_mesh_dim_dict[src_tensor_dim].pop()
                tensor_mesh_dim_dict[dst_tensor_dim].append(move_mesh_dim)
                new_placements = list(placements)
                new_placements[move_mesh_dim] = Shard(dst_tensor_dim)
                dist_state = self.DistState(
                    self._to_tuple(new_placements),
                    DTensorRedistributePlanner._dict_to_ShardOrder(
                        tensor_mesh_dim_dict
                    ),
                )
                all_next_state[dist_state] = self.cost_function(
                    cur_dist_state,
                    dist_state,
                )
                # reset content for next iteration
                tensor_mesh_dim_dict[src_tensor_dim].append(move_mesh_dim)
                tensor_mesh_dim_dict[dst_tensor_dim].pop()
        # TODO(zpcore): support discovering submesh to prevent padding when
        # tensor dim is not divisible by the mesh dim.

        ######################################################################
        # handle case 2: Shard() -> Replicate()
        for entry in tensor_mesh_dim_tuple:
            src_tensor_dim = entry.tensor_dim
            src_mesh_dim = tensor_mesh_dim_dict[src_tensor_dim][-1]
            if not isinstance(placements[src_mesh_dim], Shard):
                # skip special case like `_StridedShard`
                continue
            move_mesh_dim = tensor_mesh_dim_dict[src_tensor_dim].pop()
            new_placements = list(placements)
            new_placements[move_mesh_dim] = Replicate()
            dist_state = self.DistState(
                self._to_tuple(new_placements),
                DTensorRedistributePlanner._dict_to_ShardOrder(tensor_mesh_dim_dict),
            )
            tensor_mesh_dim_dict[src_tensor_dim].append(move_mesh_dim)
            all_next_state[dist_state] = self.cost_function(
                cur_dist_state,
                dist_state,
            )

        ######################################################################
        # handle case 3: Partial() -> Replicate()
        for src_mesh_dim, placement in enumerate(placements):
            if not isinstance(placement, Partial):
                continue
            new_placements = list(placements)
            new_placements[src_mesh_dim] = Replicate()
            dist_state = self.DistState(
                self._to_tuple(new_placements), tensor_mesh_dim_tuple
            )
            all_next_state[dist_state] = self.cost_function(
                cur_dist_state,
                dist_state,
            )

        ######################################################################
        # handle case 4: Replicate() -> Shard()
        for mesh_dim, placement in enumerate(placements):
            if not isinstance(placement, Replicate):
                continue
            for dst_tensor_dim in range(self.tensor_dimension):
                # try convert placement[mesh_dim] to Shard(dst_tensor_dim)
                new_placements = list(placements)
                new_placements[mesh_dim] = Shard(dst_tensor_dim)
                tensor_mesh_dim_dict[dst_tensor_dim].append(mesh_dim)
                dist_state = self.DistState(
                    self._to_tuple(new_placements),
                    DTensorRedistributePlanner._dict_to_ShardOrder(
                        tensor_mesh_dim_dict
                    ),
                )
                all_next_state[dist_state] = self.cost_function(
                    cur_dist_state,
                    dist_state,
                )
                tensor_mesh_dim_dict[dst_tensor_dim].pop()

        ######################################################################
        # handle case 5: Partial() -> Shard()
        for mesh_dim, placement in enumerate(placements):
            if not isinstance(placement, Partial):
                continue
            for dst_tensor_dim in range(self.tensor_dimension):
                # try convert placement[mesh_dim] to Shard(dst_tensor_dim)
                new_placements = list(placements)
                new_placements[mesh_dim] = Shard(dst_tensor_dim)
                tensor_mesh_dim_dict[dst_tensor_dim].append(mesh_dim)
                dist_state = self.DistState(
                    self._to_tuple(new_placements),
                    DTensorRedistributePlanner._dict_to_ShardOrder(
                        tensor_mesh_dim_dict
                    ),
                )
                all_next_state[dist_state] = self.cost_function(
                    cur_dist_state,
                    dist_state,
                )
                tensor_mesh_dim_dict[dst_tensor_dim].pop()

        ######################################################################
        # handle case 6: Replicate() -> Partial()
        # Generate transitions only for reduce_ops that are present in the src/dst
        # placements for this redistribution, avoiding unnecessary graph expansion.
        for mesh_dim, placement in enumerate(placements):
            if not isinstance(placement, Replicate):
                continue
            for reduce_op in self.partial_reduce_ops_in_target:
                new_placements = list(placements)
                new_placements[mesh_dim] = Partial(reduce_op)

                # Skip if this would create mixed partial types (except sum+avg which commute)
                partial_reduce_ops = {
                    p.reduce_op for p in new_placements if isinstance(p, Partial)
                }
                if len(partial_reduce_ops) > 1 and partial_reduce_ops != {"sum", "avg"}:
                    continue

                dist_state = self.DistState(
                    self._to_tuple(new_placements), tensor_mesh_dim_tuple
                )
                all_next_state[dist_state] = self.cost_function(
                    cur_dist_state,
                    dist_state,
                )

        # Additional cases handling for _StridedShard

        ######################################################################
        # TODO(zpcore): handle case 8: _StridedShard() -> Shard() on the same dim

        ######################################################################
        # handle case 9: _StridedShard() -> Replicate()
        for entry in tensor_mesh_dim_tuple:
            src_tensor_dim = entry.tensor_dim
            src_mesh_dim = tensor_mesh_dim_dict[src_tensor_dim][-1]
            if not isinstance(placements[src_mesh_dim], _StridedShard):
                continue
            move_mesh_dim = tensor_mesh_dim_dict[src_tensor_dim].pop()
            new_placements = list(placements)
            new_placements[move_mesh_dim] = Replicate()
            dist_state = self.DistState(
                self._to_tuple(new_placements),
                DTensorRedistributePlanner._dict_to_ShardOrder(tensor_mesh_dim_dict),
            )
            tensor_mesh_dim_dict[src_tensor_dim].append(move_mesh_dim)
            all_next_state[dist_state] = self.cost_function(
                cur_dist_state,
                dist_state,
            )

        # Early exit if no StridedShard in target
        if not self.strided_shard_placements_in_target:
            return all_next_state

        ######################################################################
        # TODO(zpcore): handle case 10: Shard() -> _StridedShard()

        ######################################################################
        # TODO(zpcore): handle case 11: Partial() -> _StridedShard()

        ######################################################################
        # handle case 12: Replicate() -> _StridedShard()
        for mesh_dim, placement in enumerate(placements):
            if not isinstance(placement, Replicate):
                continue
            for strided_shard_obj in self.strided_shard_placements_in_target:
                dst_tensor_dim = strided_shard_obj.dim
                # try convert placement[mesh_dim] to strided_shard_obj
                new_placements = list(placements)
                new_placements[mesh_dim] = strided_shard_obj
                tensor_mesh_dim_dict[dst_tensor_dim].append(mesh_dim)
                dist_state = self.DistState(
                    self._to_tuple(new_placements),
                    DTensorRedistributePlanner._dict_to_ShardOrder(
                        tensor_mesh_dim_dict
                    ),
                )
                all_next_state[dist_state] = self.cost_function(
                    cur_dist_state,
                    dist_state,
                )
                tensor_mesh_dim_dict[dst_tensor_dim].pop()

        return all_next_state

    def _get_shard_to_partial_target_states(
        self, current_state: DistState, dst_state: DistState
    ) -> dict[DistState, float]:
        """Generate explicit-only Shard -> Partial("sum") target steps.

        Zero weight means no communication; allocation and copy costs are omitted.
        This edge is generated only when Partial("sum") is the requested target,
        so path search cannot use it as an intermediate step to another layout.
        """
        next_states: dict[DTensorRedistributePlanner.DistState, float] = {}
        shard_order = self._ShardOrder_to_dict(current_state.tensor_dim_to_mesh_dim)
        for entry in current_state.tensor_dim_to_mesh_dim:
            tensor_dim = entry.tensor_dim
            mesh_dim = shard_order[tensor_dim][-1]
            current = current_state.placements[mesh_dim]
            target = dst_state.placements[mesh_dim]
            if not (
                isinstance(current, Shard)
                and type(target) is Partial
                and target.reduce_op == "sum"
            ):
                continue

            shard_order[tensor_dim].pop()
            placements = list(current_state.placements)
            placements[mesh_dim] = target
            next_state = self.DistState(
                tuple(placements), self._dict_to_ShardOrder(shard_order)
            )
            shard_order[tensor_dim].append(mesh_dim)
            next_states[next_state] = 0.0
        return next_states

    # TODO(zpcore): if the dst_state contains special placement like
    # `_MaskPartial`, we will never reach that state. Need to support this case.
    def find_min_cost_path(
        self, src_state: DistState, dst_state: DistState
    ) -> list["DTensorRedistributePlanner.DistState"]:
        """
        Find the min cost path from src_state to dst_state using Dijkstra's
        algorithm.

        Args:
            src_state: The source state
            dst_state: The destination state

        Returns:
            A list of states representing the min cost path from src_state to
            dst_state
        """
        import heapq

        # priority queue (cost_key, counter, cost, state, path) for Dijkstra's
        # algorithm. The sort key is a non-guarding optimization hint so that
        # symbolic communication costs do not force heapq to evaluate guards.
        # Counter breaks ties and avoids comparing DistState objects.
        counter = 0
        pq: list[
            tuple[
                float,
                int,
                FloatLikeType,
                DTensorRedistributePlanner.DistState,
                list[DTensorRedistributePlanner.DistState],
            ]
        ] = [(0.0, counter, 0.0, src_state, [src_state])]
        visited = set()
        while pq:
            _, _, cost, current_state, path = heapq.heappop(pq)
            if current_state == dst_state:
                return path
            if current_state in visited:
                continue
            visited.add(current_state)
            # get all possible next states and their costs
            next_states = self.get_next_state(
                current_state.placements,
                current_state.tensor_dim_to_mesh_dim,
            )
            next_states.update(
                self._get_shard_to_partial_target_states(current_state, dst_state)
            )
            for next_state, transition_cost in next_states.items():
                if next_state not in visited:
                    new_cost = cast(Any, cost) + transition_cost
                    new_path = path + [next_state]
                    counter += 1
                    heapq.heappush(
                        pq,
                        (
                            _redistribute_cost_sort_key(new_cost),
                            counter,
                            new_cost,
                            next_state,
                            new_path,
                        ),
                    )
        raise AssertionError(
            f"No path found from src_state {src_state} to dst_state {dst_state}"
        )

    def get_logical_shape(
        self,
        src_state: "DTensorRedistributePlanner.DistState",
        mesh_dim: int,
        full_tensor_shape: tuple[int, ...],
    ) -> list[IntLikeType]:
        new_logical_shape: list[IntLikeType] = list(full_tensor_shape)
        for entry in src_state.tensor_dim_to_mesh_dim:
            tensor_dim = entry.tensor_dim
            mesh_dims = entry.mesh_dims
            if len(mesh_dims) <= 0:
                raise AssertionError
            for mdim in mesh_dims:
                if mdim == mesh_dim:
                    continue
                placement = src_state.placements[mdim]
                if isinstance(placement, Shard):
                    new_size, _ = placement.local_shard_size_and_offset(
                        new_logical_shape[tensor_dim],
                        self.device_mesh.size(mesh_dim=mdim),
                        self.device_mesh._sym_get_coordinate(mdim),
                    )
                elif isinstance(placement, _StridedShard):
                    new_size, _ = placement.local_shard_size_and_offset(
                        new_logical_shape[tensor_dim],
                        self.device_mesh.size(mesh_dim=mdim),
                        self.device_mesh._sym_get_coordinate(mdim),
                    )
                else:
                    raise ValueError(f"Unsupported placement type: {placement}")
                new_logical_shape[tensor_dim] = new_size
        return new_logical_shape

    def generate_graph_based_transform_infos(
        self,
        src_spec: DTensorSpec,
        dst_spec: DTensorSpec,
        full_tensor_shape: tuple[int, ...],
    ) -> list[_TransformInfo]:
        # TODO(zpcore): Temporary workaround for backward compatibility where
        # _StridedShard was used to encode device shard order. We should migrate
        # to explicit `shard_order` instead.
        def _try_normalize_spec(
            spec: DTensorSpec,
        ) -> tuple[tuple[Placement, ...], ShardOrder]:
            if spec.use_strided_shard_as_shard_order:
                new_placements, shard_order = (
                    DTensorSpec._normalize_placements_into_shard_order(
                        spec.placements,
                        spec.mesh,
                        use_strided_shard_as_shard_order=True,
                    )
                )
                return new_placements, shard_order
            else:
                if spec.shard_order is None:
                    raise ValueError(f"Missing shard_order field in {spec}")
                return spec.placements, spec.shard_order

        src_placements, src_shard_order = _try_normalize_spec(src_spec)
        dst_placements, dst_shard_order = _try_normalize_spec(dst_spec)

        # In case _StridedShard still exists in placements, collect possible
        # split_factor values in the target placements. Need those values to
        # redistribute from Shard into _StridedShard.
        for placement in dst_placements:
            if isinstance(placement, _StridedShard):
                self.strided_shard_placements_in_target.add(placement)

        # Collect Partial reduce ops from src and dst placements. These are used
        # to generate R->P transitions only for reduce ops that are actually
        # present in the redistribution, avoiding unnecessary graph expansion.
        for placement in itertools.chain(src_placements, dst_placements):
            if isinstance(placement, Partial):
                self.partial_reduce_ops_in_target.add(placement.reduce_op)

        src_state = self.DistState(src_placements, src_shard_order)
        dst_state = self.DistState(dst_placements, dst_shard_order)
        transform_infos: list[_TransformInfo] = []
        state_path = self.find_min_cost_path(src_state, dst_state)
        for cur_state, nxt_state in itertools.pairwise(state_path):
            # find the mesh_dim that is different between cur_state and nxt_state
            if cur_state.placements != nxt_state.placements:
                update_mesh_dim = -1
                for mesh_dim, (cur_placement, nxt_placement) in enumerate(
                    zip(cur_state.placements, nxt_state.placements)
                ):
                    if cur_placement != nxt_placement:
                        if update_mesh_dim != -1:
                            raise AssertionError(
                                "Multiple mesh_dims are different between cur_state and nxt_state"
                            )
                        update_mesh_dim = mesh_dim
                        logical_shape = self.get_logical_shape(
                            cur_state, mesh_dim, full_tensor_shape
                        )
                        transform_infos.append(
                            _TransformInfo(
                                mesh_dim=update_mesh_dim,
                                src_dst_placements=(cur_placement, nxt_placement),
                                logical_shape=logical_shape,
                            )
                        )

        return transform_infos

    def generate_greedy_transform_infos(
        self,
        src_spec: DTensorSpec,
        dst_spec: DTensorSpec,
    ) -> list[_TransformInfo]:
        """
        Generate the transform infos from the source placements to the target placements.

        To transform from source to target placement it might have multiple steps, i.e. it
        might decompose Si -> Sj into Si -> R -> Sj.
        This would detect if there're mis-aligned/nested shardings between src/dst placements.
        E.g. Suppose the redistribution to perform is (Shard(0), Shard(0)) -> (Replicate(), Shard(0)),
        in this case Shard(0) -> Shard(0) for mesh dimension 1 actually needs resharding, because in
        the former is a nested-sharding of a tensor already sharded dimension 0, whereas
        the latter is the first sharding on tensor dimension 0.
        """
        # logical shape records the logic tensor shape on the mesh dimension
        # this is useful to ensure uneven sharding gets correct output shape
        initial_logical_shape = list(src_spec.shape)
        mesh_dims_to_logical_shape = [initial_logical_shape]
        transform_infos: list[_TransformInfo] = []
        if self.device_mesh.ndim == 1:
            # if device_mesh is 1D, redistribute is a simple direct
            # transformation (skip if src == dst)
            if src_spec.placements[0] != dst_spec.placements[0]:
                transform_infos.append(
                    _TransformInfo(
                        mesh_dim=0,
                        src_dst_placements=(
                            src_spec.placements[0],
                            dst_spec.placements[0],
                        ),
                        logical_shape=initial_logical_shape,
                    )
                )
            return transform_infos

        # Handle multi-dim device mesh placement redistribution First, we need
        # to build the logical shape for each mesh dim for correct allgather
        # uneven shards on each mesh dim (with dynamic padding)
        for i, src in enumerate(src_spec.placements):
            current_logical_shape = mesh_dims_to_logical_shape[i]
            if isinstance(src, Shard):
                if i < self.device_mesh.ndim - 1:
                    # calculate and save the logical shape for this sharding
                    mesh_dim_size = self.device_mesh.size(mesh_dim=i)
                    local_shard_size, _ = src._local_shard_size_and_offset(
                        current_logical_shape[src.dim],
                        mesh_dim_size,
                        self.device_mesh._sym_get_coordinate(i),
                    )
                    new_logical_shape = list(current_logical_shape)
                    new_logical_shape[src.dim] = local_shard_size
                    mesh_dims_to_logical_shape.append(new_logical_shape)
            else:
                mesh_dims_to_logical_shape.append(current_logical_shape)

        # Next, we need to derive the transform infos from src to dst
        # placements, here we use a greedy search with step by step state
        # transformations
        current_placements = list(src_spec.placements)
        target_placements = list(dst_spec.placements)

        if src_spec.num_shards > 1:
            # If src_spec have sharding, it could potentially have sharding that
            # is misaligned with dst_spec a common case of this is nested
            # sharding (i.e. (S(0), S(0)) -> (R, S(0))). In those cases, we
            # first traverse from inner placement to outer placement to detect
            # misaligned shardings and properly replicate nested sharding first.
            for mesh_dim in reversed(range(len(current_placements))):
                current = current_placements[mesh_dim]
                target = target_placements[mesh_dim]
                # If target is not Shard, we can directly redistribute since we
                # are traversing from inner to outer placements here
                # TODO: extend nested sharding detection to _StridedShard
                # (isinstance check and is_shard() below miss it).
                # Safe today: strategies convert _StridedShard to Replicate
                # on ALL mesh dims for a given reduction dim, so misaligned
                # nested _StridedShard targets can't arise.
                if isinstance(target, Shard):
                    # If target is Shard, check for nested sharding on the
                    # tensor dim BEFORE the current mesh_dim
                    shard_dim = target.dim
                    current_mesh_sharding, target_mesh_sharding = [], []
                    for i, (s, p) in enumerate(
                        zip(current_placements, target_placements)
                    ):
                        if i >= mesh_dim:
                            break
                        if s.is_shard(shard_dim):
                            current_mesh_sharding.append(i)
                        if p.is_shard(shard_dim):
                            target_mesh_sharding.append(i)

                    if current_mesh_sharding != target_mesh_sharding:
                        # if current/target_placements have misaligned sharding
                        # on the tensor dim BEFORE the current mesh_dim, we need
                        # to replicate the tensor on the mesh dim first to clear
                        # the nested sharding
                        target = Replicate()

                if current != target:
                    transform_infos.append(
                        _TransformInfo(
                            mesh_dim=mesh_dim,
                            src_dst_placements=(current, target),
                            logical_shape=mesh_dims_to_logical_shape[mesh_dim],
                        )
                    )
                    current_placements[mesh_dim] = target

        # We always traverse from outer placement to inner placement to collect
        # the remaining needed transform infos (i.e. the replication from nested
        # sharding might need to further perform resharding to Shard again)
        for mesh_dim, (current, target) in enumerate(
            zip(current_placements, target_placements)
        ):
            if current != target:
                transform_infos.append(
                    _TransformInfo(
                        mesh_dim=mesh_dim,
                        src_dst_placements=(current, target),
                        logical_shape=mesh_dims_to_logical_shape[mesh_dim],
                    )
                )
                current_placements[mesh_dim] = target
        return transform_infos


def _gen_transform_infos_non_cached(
    src_spec: DTensorSpec,
    dst_spec: DTensorSpec,
    use_graph_based_transform: bool | None = None,
) -> list[_TransformInfo]:
    device_mesh = src_spec.device_mesh
    src_shard_order = src_spec.shard_order
    dst_shard_order = dst_spec.shard_order
    # DTensorSpec should automatically generate shard_order, and it can be () if
    # no shard.
    has_non_default_order = not all(
        DTensorSpec.is_default_device_order(order)
        for order in (src_shard_order, dst_shard_order)
    )
    has_strided_shard = any(
        isinstance(p, _StridedShard)
        for p in (*src_spec.placements, *dst_spec.placements)
    )

    # Determine which transform strategy to use:
    # 1. Non-standard device order or contains _StridedShard → always use graph-based
    # 2. Global flag or explicit parameter True → use graph-based
    # 3. Otherwise → use greedy
    if has_non_default_order or has_strided_shard:
        use_graph_based_transform = True
    elif _FORCE_MIN_COST_REDISTRIBUTION_PLAN is not None:
        use_graph_based_transform = _FORCE_MIN_COST_REDISTRIBUTION_PLAN
    elif use_graph_based_transform is None:
        use_graph_based_transform = False
    if src_spec.tensor_meta is None:
        raise AssertionError
    drp = get_redistribute_planner(
        device_mesh,
        src_spec.tensor_meta,
    )
    if use_graph_based_transform:
        # TODO(zpcore): Temporary workaround for the case where _StridedShard
        # cannot be decoded into shard order. This happens when
        # use_strided_shard_as_shard_order defaults to True (e.g. in
        # Redistribute.forward where the target DTensorSpec is constructed from
        # raw placements without the flag), but the split_factor doesn't
        # correspond to any valid product of mesh dimension sizes (e.g. sf=2
        # on a 1D mesh). A proper fix is to either pass
        # use_strided_shard_as_shard_order through the Redistribute API, or
        # migrate to explicit shard_order so _StridedShard is no longer
        # overloaded for two purposes.
        try:
            transform_infos = drp.generate_graph_based_transform_infos(
                src_spec,
                dst_spec,
                src_spec.shape,
            )
        except _StridedShardNotDecodableError:
            transform_infos = drp.generate_greedy_transform_infos(src_spec, dst_spec)
    else:
        transform_infos = drp.generate_greedy_transform_infos(src_spec, dst_spec)
    return transform_infos


@cache
def _gen_transform_infos(
    src_spec: DTensorSpec,
    dst_spec: DTensorSpec,
    use_graph_based_transform: bool | None = None,
) -> list[_TransformInfo]:
    return _gen_transform_infos_non_cached(
        src_spec,
        dst_spec,
        use_graph_based_transform,
    )


def redistribute_local_tensor(
    local_tensor: torch.Tensor,
    current_spec: DTensorSpec,
    target_spec: DTensorSpec,
    *,
    async_op: bool = False,
    use_graph_based_transform: bool | None = None,
    # True if user explicitly called DTensor.redistribute()
    is_explicit: bool = False,
) -> torch.Tensor:
    """
    This redistribute the local tensor (torch.Tensor) from the current DTensorSpec to
    the target DTensorSpec, which involves the necessary collective calls to transform
    the local shard of the DTensor from its current spec to the target spec.
    """

    if current_spec.mesh != target_spec.mesh:
        # TODO: alltoall/permute reshuffling to change device_mesh if they are not the same
        raise NotImplementedError("Cross device mesh comm not supported yet!")

    if current_spec.use_strided_shard_as_shard_order is None:
        raise ValueError(
            "use_strided_shard_as_shard_order should be initialized in DTensorSpec.__post_init__()"
        )

    # We do not see a valid use case for mixing different partial types in the same DTensor.
    # in principle it could be supported, but since nonlinear reductions (e.g. max) exist, relative ordering
    # of different partials would become semantically critical.  Without a motivating use case, we prohibit this.
    assert_no_mixed_partial_types(current_spec.placements)
    assert_no_mixed_partial_types(target_spec.placements)

    new_local_tensor = local_tensor
    device_mesh = current_spec.mesh

    if not device_mesh._is_current_rank_part_of_mesh():
        # if rank is not part of mesh, we skip redistribute and simply return local_tensor,
        # which should be an empty tensor
        return local_tensor

    if _are_we_tracing():
        transform_infos = _gen_transform_infos_non_cached(
            current_spec,
            target_spec,
            use_graph_based_transform,
        )
    else:
        transform_infos = _gen_transform_infos(
            current_spec,
            target_spec,
            use_graph_based_transform,
        )

    # Optimize by grouping same-type collectives into flattened operations
    optimized_transform_infos = _optimize_transform_infos(
        transform_infos,
        device_mesh,
        current_spec.placements,
        target_spec.placements,
    )

    debug_mode = get_active_debug_mode()

    redistribute_context = (
        debug_mode.record_redistribute_calls(  # type: ignore[union-attr]
            local_tensor,
            current_spec.placements,
            target_spec.placements,
            DTensorRedistributePlanner.stringify_transform_infos(
                device_mesh,
                optimized_transform_infos,
                current_spec.placements,
                current_spec.shard_order,
                current_spec.use_strided_shard_as_shard_order,
            ),
            is_explicit=is_explicit,
        )
        if debug_mode is not None
        else contextlib.nullcontext()
    )

    with redistribute_context:
        for transform_info in optimized_transform_infos:
            # Determine which mesh to use: flattened transforms have their own mesh
            if isinstance(transform_info, _FlattenedTransformInfo):
                mesh_to_use = transform_info.mesh
            else:
                mesh_to_use = device_mesh
            i = transform_info.mesh_dim
            current, target = transform_info.src_dst_placements

            # _StridedShard methods use device_mesh directly, not mesh_to_use.
            # This is safe because _StridedShard.is_shard() returns False, so
            # _comm_type_key() returns None and flattening is never attempted.
            if isinstance(current, _StridedShard) or isinstance(target, _StridedShard):
                if mesh_to_use is not device_mesh:
                    raise AssertionError(
                        "_StridedShard redistribute assumes no flattened transforms"
                    )

            num_chunks = mesh_to_use.size(mesh_dim=i)

            if current == target:
                # short cut, just use the original local tensor
                new_local_tensor = local_tensor
                continue

            if num_chunks == 1:
                # short cut, if there's only one shard, we don't need to do any collective
                # comm, just use the original local tensor
                new_local_tensor = local_tensor
                continue

            if target.is_replicate():
                # Case 1: target is Replicate
                if current.is_partial():
                    partial_spec = cast(Partial, current)
                    new_local_tensor = partial_spec._reduce_value(
                        local_tensor, mesh_to_use, i
                    )
                    # For merged sum/avg partials, apply avg scaling
                    if (
                        isinstance(transform_info, _FlattenedTransformInfo)
                        and transform_info.avg_scale is not None
                    ):
                        new_local_tensor = new_local_tensor / transform_info.avg_scale
                elif current.is_shard():
                    current_placement = cast(Shard, current)
                    new_local_tensor = current_placement._to_replicate_tensor(
                        local_tensor, mesh_to_use, i, transform_info.logical_shape
                    )
                elif isinstance(current, _StridedShard):
                    new_local_tensor = current._to_replicate_tensor(
                        local_tensor, device_mesh, i, transform_info.logical_shape
                    )
                else:
                    raise RuntimeError(
                        f"redistribute from {current} to {target} not supported yet"
                    )

            elif target.is_shard():
                # Case 2: target is Shard
                target_placement = cast(Shard, target)
                if current.is_partial():
                    partial_spec = cast(Partial, current)
                    new_local_tensor = partial_spec._reduce_shard_value(
                        local_tensor, mesh_to_use, i, target_placement
                    )
                    # For merged sum/avg partials, apply avg scaling
                    if (
                        isinstance(transform_info, _FlattenedTransformInfo)
                        and transform_info.avg_scale is not None
                    ):
                        new_local_tensor = new_local_tensor / transform_info.avg_scale
                elif current.is_replicate():
                    # split the tensor and return the corresponding cloned local shard
                    new_local_tensor = target_placement._replicate_to_shard(
                        local_tensor, mesh_to_use, i, mesh_to_use._sym_get_coordinate(i)
                    )
                elif current.is_shard():
                    shard_spec = cast(Shard, current)
                    if shard_spec.dim != target_placement.dim:
                        new_local_tensor = shard_spec._to_new_shard_dim(
                            local_tensor,
                            mesh_to_use,
                            i,
                            transform_info.logical_shape,
                            target_placement.dim,
                        )
                elif isinstance(current, _StridedShard):
                    # _StridedShard -> Shard: go via Replicate as intermediate
                    replicated = current._to_replicate_tensor(
                        local_tensor, device_mesh, i, transform_info.logical_shape
                    )
                    new_local_tensor = target_placement._replicate_to_shard(
                        replicated,
                        mesh_to_use,
                        i,
                        mesh_to_use._sym_get_coordinate(i),
                    )
                else:
                    raise ValueError(
                        f"Unexpected placement {current} for redistribute to target placement {target}"
                    )
            elif target.is_partial():
                if current.is_replicate():
                    partial_spec = cast(Partial, target)
                    new_local_tensor = partial_spec._partition_value(
                        local_tensor, mesh_to_use, i
                    )
                elif _is_shard_like(current):
                    if (
                        not isinstance(current, Shard)
                        or type(target) is not Partial
                        or target.reduce_op != "sum"
                    ):
                        raise RuntimeError(
                            f"redistribute from {current} to {target} not supported yet"
                        )
                    new_local_tensor = current._to_partial_tensor(
                        local_tensor,
                        mesh_to_use,
                        i,
                        transform_info.logical_shape,
                    )
                else:
                    if current != target:
                        raise AssertionError(
                            f"Redistribution from one partial type ({current}) to another ({target}) is unsupported."
                        )
                    # partial -> partial no op, should never hit
                    new_local_tensor = local_tensor
            elif isinstance(target, _StridedShard):
                # Case 4: target is _StridedShard
                if current.is_partial():
                    # Partial -> _StridedShard: reduce to Replicate, then strided shard
                    partial_spec = cast(Partial, current)
                    replicated = partial_spec._reduce_value(
                        local_tensor, mesh_to_use, i
                    )
                    new_local_tensor = target._replicate_to_strided_shard(
                        replicated, device_mesh, i, device_mesh._sym_get_coordinate(i)
                    )
                elif current.is_replicate():
                    # split the tensor and return the corresponding local strided shard
                    new_local_tensor = target._replicate_to_strided_shard(
                        local_tensor, device_mesh, i, device_mesh._sym_get_coordinate(i)
                    )
                elif current.is_shard():
                    # Shard -> _StridedShard: all-gather to Replicate, then strided shard
                    current_placement = cast(Shard, current)
                    replicated = current_placement._to_replicate_tensor(
                        local_tensor, mesh_to_use, i, transform_info.logical_shape
                    )
                    new_local_tensor = target._replicate_to_strided_shard(
                        replicated, device_mesh, i, device_mesh._sym_get_coordinate(i)
                    )
                elif isinstance(current, _StridedShard):
                    # _StridedShard -> _StridedShard: go through Replicate
                    # First convert to Replicate, then to _StridedShard
                    replicated = current._to_replicate_tensor(
                        local_tensor, device_mesh, i, transform_info.logical_shape
                    )
                    new_local_tensor = target._replicate_to_strided_shard(
                        replicated, device_mesh, i, device_mesh._sym_get_coordinate(i)
                    )
                else:
                    raise ValueError(
                        f"Unexpected placement {current} for redistribute to target placement {target}"
                    )

            if not async_op and isinstance(
                new_local_tensor, funcol.AsyncCollectiveTensor
            ):
                new_local_tensor = new_local_tensor.wait()
            local_tensor = new_local_tensor
    return new_local_tensor


def _redistribute_backward(
    grad_output: "dtensor.DTensor",
    previous_spec: DTensorSpec,
    *,
    out_dtype: torch.dtype,
    op_dtype: torch.dtype,
    async_op: bool = False,
):
    """
    Common function for redistributing a distributed tensor during backward
    and twice-backward backpropagation steps.

    Args:
        grad_output: The output gradient tensor.
        previous_spec: DTensorSpec prior to redistribution.
        out_dtype: dtype to cast the returned gradient to. Pinned by autograd
                to match the dtype of the input of the node above this one.
        op_dtype: dtype to run the backward collective at.
        async_op: whether to perform the DTensor redistribute operation
                asynchronously or not. Default: False

    Returns:
        A :class:`torch.Tensor` object.
        A :class:`DTensorSpec` object.
    """
    if op_dtype != grad_output._local_tensor.dtype:
        local_tensor = grad_output._local_tensor.to(dtype=op_dtype)
        current_spec = DTensorSpec(
            mesh=grad_output._spec.device_mesh,
            placements=grad_output._spec.placements,
            tensor_meta=TensorMeta(
                shape=grad_output.shape,
                stride=grad_output.stride(),
                dtype=op_dtype,
            ),
            use_strided_shard_as_shard_order=grad_output._spec.use_strided_shard_as_shard_order,
        )
        previous_spec = DTensorSpec(
            mesh=previous_spec.device_mesh,
            placements=previous_spec.placements,
            tensor_meta=current_spec.tensor_meta,
            use_strided_shard_as_shard_order=previous_spec.use_strided_shard_as_shard_order,
        )
    else:
        local_tensor = grad_output._local_tensor
        current_spec = grad_output._spec
    # skip the replicate to partial transformation when we are in backward pass
    # In this case we keep the grad as replicate, this is because we don't
    # want to convert the replicated gradients back to partial, although
    # that's logically conform with the same layout, converting the gradients
    # back to partial is actually useless as you would have to do reduce later
    # which would be more expensive than keeping it replicate!

    # for backward shard -> partial, we just do shard -> replicate
    # for backward replicate -> partial, we skip the transformation
    # NOTE: _is_shard_like covers _StridedShard defensively; currently
    # unreachable because Partial -> _StridedShard is not implemented.
    normalized_placements: list[Placement] = []
    for current, target in zip(current_spec.placements, previous_spec.placements):
        if (_is_shard_like(current) or current.is_replicate()) and target.is_partial():
            normalized_placements.append(Replicate())
        else:
            normalized_placements.append(target)

    previous_spec = DTensorSpec(
        previous_spec.device_mesh,
        placements=tuple(normalized_placements),
        tensor_meta=previous_spec.tensor_meta,
        use_strided_shard_as_shard_order=previous_spec.use_strided_shard_as_shard_order,
    )

    output = redistribute_local_tensor(
        local_tensor,
        current_spec,
        previous_spec,
        async_op=async_op,
    )

    if output.dtype != out_dtype:
        output = output.to(out_dtype)

    spec = DTensorSpec(
        previous_spec.device_mesh,
        tuple(normalized_placements),
        tensor_meta=TensorMeta(
            shape=grad_output.shape,
            stride=grad_output.stride(),
            dtype=output.dtype,
        ),
        use_strided_shard_as_shard_order=previous_spec.use_strided_shard_as_shard_order,
    )
    return output, spec


class _BackwardDtypeConfig(TypedDict):
    op_dtype: torch.dtype  # dtype the backward collective runs at
    out_dtype: torch.dtype  # dtype the returned gradient is cast to


class _DtypeConfig(TypedDict):
    op_dtype: torch.dtype  # forward: dtype the collective runs at
    out_dtype: torch.dtype  # forward: dtype of the output tensor
    backward_options: _BackwardDtypeConfig


class Redistribute(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        # pyre-fixme[2]: Parameter must be annotated.
        ctx,
        input: "dtensor.DTensor",
        device_mesh: DeviceMesh,
        placements: tuple[Placement, ...],
        async_op: bool,
        dtype_config: _DtypeConfig,
    ):
        ctx.async_op = async_op
        bwd = dtype_config["backward_options"]
        ctx.bwd_op_dtype = bwd["op_dtype"]
        ctx.bwd_out_dtype = bwd["out_dtype"]

        op_dtype = dtype_config["op_dtype"]
        out_dtype = dtype_config["out_dtype"]

        if op_dtype != input._local_tensor.dtype:
            local_tensor = input._local_tensor.to(dtype=op_dtype)
            current_spec = DTensorSpec(
                mesh=device_mesh,
                placements=input._spec.placements,
                tensor_meta=TensorMeta(
                    shape=input.shape,
                    stride=input.stride(),
                    dtype=op_dtype,
                ),
                use_strided_shard_as_shard_order=input._spec.use_strided_shard_as_shard_order,
            )
        else:
            local_tensor = input._local_tensor
            current_spec = input._spec

        ctx.current_spec = current_spec

        if current_spec.placements != placements:
            target_spec = DTensorSpec(
                device_mesh, placements, tensor_meta=current_spec.tensor_meta
            )

            output = redistribute_local_tensor(
                local_tensor,
                current_spec,
                target_spec,
                async_op=async_op,
                is_explicit=True,
            )
        else:
            # use the same local tensor if placements are the same.
            output = local_tensor
            target_spec = current_spec

        if output.dtype != out_dtype:
            output = output.to(out_dtype)
            target_spec = DTensorSpec(
                device_mesh,
                target_spec.placements,
                tensor_meta=TensorMeta(
                    shape=input.shape,
                    stride=input.stride(),
                    dtype=out_dtype,
                ),
                use_strided_shard_as_shard_order=target_spec.use_strided_shard_as_shard_order,
            )

        # pyrefly: ignore [bad-argument-type]
        return dtensor.DTensor(
            # pyrefly: ignore [bad-argument-count]
            output,
            target_spec,
            # pyrefly: ignore [unexpected-keyword]
            requires_grad=input.requires_grad,
        )

    @staticmethod
    def backward(ctx, grad_output: "dtensor.DTensor"):  # type: ignore[override]
        output_dtensor = NestedRedistribute.apply(
            grad_output,
            ctx.current_spec,
            ctx.async_op,
            ctx.bwd_op_dtype,
            ctx.bwd_out_dtype,
        )
        return (
            output_dtensor,
            None,
            None,
            None,
            None,
        )


class NestedRedistribute(torch.autograd.Function):
    """
    This class is used to make the redistribution of a DTensor twice-differentiable.
    This is called during the `Redistribute.forward`.
    Therefore, `NestedRedistribute.forward` is called during the first backward pass,
    and `NestedRedistribute.backward` is called during the second backward pass.

    Note: `NestedRedistribute.backward` is not differentiable, and therefore triple
    backward is not yet supported.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        # pyre-fixme[2]: Parameter must be annotated.
        ctx,
        grad_output: "dtensor.DTensor",
        previous_spec: DTensorSpec,
        async_op: bool,
        op_dtype: torch.dtype,
        out_dtype: torch.dtype,
    ):
        # Persist op_dtype so the double-backward reuses the same collective
        # dtype one level down.
        ctx.async_op = async_op
        ctx.original_dtype = grad_output._local_tensor.dtype
        ctx.op_dtype = op_dtype

        output, spec = _redistribute_backward(
            grad_output,
            previous_spec,
            out_dtype=out_dtype,
            op_dtype=op_dtype,
            async_op=async_op,
        )

        ctx.current_spec = spec

        # pyrefly: ignore [bad-argument-type]
        return dtensor.DTensor(
            # pyrefly: ignore [bad-argument-count]
            output,
            spec,
            # pyrefly: ignore [unexpected-keyword]
            requires_grad=grad_output.requires_grad,
        )

    @staticmethod
    def backward(ctx, grad2_output: "dtensor.DTensor"):  # type: ignore[override]
        previous_spec = ctx.current_spec
        # Reuse the same op_dtype one level down; pin out_dtype to the dtype
        # of the grad we received at this level, since that's what the node
        # above us expects back.
        output_dtensor = NestedRedistribute.apply(
            grad2_output,
            previous_spec,
            ctx.async_op,
            ctx.op_dtype,
            ctx.original_dtype,
        )

        return (
            output_dtensor,
            None,
            None,
            None,
            None,
            None,
        )
<|endoftext|>import logging
import threading
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.tensor._api as dtensor
from torch._logging import LazyString
from torch._prims_common import ShapeType
from torch.distributed import RankType
from torch.distributed._local_tensor import maybe_run_for_local_tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._collective_utils import redistribute_cost
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import OpSchema
from torch.distributed.tensor.placement_types import (
    _is_shard_like,
    _StridedShard,
    _StridedShardOffsetMode,
    Partial,
    Placement,
    Replicate,
    Shard,
)


logger = logging.getLogger(__name__)


def _format_implicit_redistribution_msg(schema: OpSchema) -> str:
    return f"Implicit redistribution occurred for {schema} while ExplicitRedistributionContext was active"


class ExplicitRedistributionContext:
    """
    Within this context manager, DTensor will refuse to perform implicit redistribution,
    instead raising an error.  Manual calls to ``redistribute()`` are required wherever a redistribution
    must occur to avoid erroring.  This can be used to ensure that the user is aware of all redistribution.

    Note: it is easier to use this mode on just the forward pass of a typical DTensor program, as the backwards pass
    may contain implicit redistribution calls that are not visible to the user and difficult to replace with manual
    calls.  Redistribution during backward can be made explicit by writing `autograd.Function`s that are no-op
    during forward and perform a manual redistribution during backwards.

    enable (bool) if False, disables the context manager. Can be used nested inside an enabled region.

    strict (bool) if True, triggers on any redistribution.  If False, only triggers on redistributions that perform
    communication.

    mode (str) Determines what happens when ExplicitRedistributionContext triggers:
    "raise": raises an exception, "warn" issues a warning
    """

    _local = threading.local()

    def __init__(self, enable: bool = True, strict: bool = False, mode="raise"):
        self._enable = enable
        self._strict = strict
        if mode not in ("raise", "warn"):
            raise RuntimeError(f"Invalid mode {mode}")
        self._raise_on_redistribution = mode == "raise"

    @classmethod
    def observe_redistribution(
        cls,
        src_spec: DTensorSpec,
        dst_spec: DTensorSpec,
        redistribution_msg: LazyString,
    ):
        if instance := getattr(cls._local, "_active", None):
            allowed = True
            if instance._enable:
                if instance._strict:
                    allowed = False
                else:
                    allowed = redistribute_cost(src_spec, dst_spec) <= 0
            if not allowed:
                if instance._raise_on_redistribution:
                    raise RuntimeError(redistribution_msg)
                else:
                    logger.warning(redistribution_msg)

    def __enter__(self):
        self._prev = getattr(ExplicitRedistributionContext._local, "_active", None)
        ExplicitRedistributionContext._local._active = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ExplicitRedistributionContext._local._active = self._prev


def compute_local_shape_and_global_offset(
    global_shape: ShapeType,
    mesh: DeviceMesh,
    placements: Sequence[Placement],
    skip_offset: bool = False,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Compute the local tensor shape and the global offsets into the original tensor
    of a DTensor on its current global rank. This is useful for checkpointing purpose.

    Example:
    global_tensor = [[0,  1,  2,  3,  4], sharded on mesh (DP=2, TP=2) with (Shard(1), Shard(1))
                     [10, 11, 12, 13, 14]]

    This table shows the return value of local_shape and global_offset for each rank.
    (`local_tensor` is for illustration only).

    Note how the first coordinate of global_offset is always 0, corresponding to tensor dim 0 being replicated.

    Rank        local_tensor        local_shape     global_offset
    -------------------------------------------------------------
    0           [[0, 1],            (2, 2)          (0, 0)
                 [10, 11]]

    1           [[2],               (2, 1)          (0, 2)
                 [12]]

    2           [[3],               (2, 1)          (0, 3)
                 [13]]

    3           [[4],               (2, 1)          (0, 4)
                 [14]]

    Args:
        global_shape (ShapeType): The global shape of the DTensor.
        mesh (:class:`DeviceMesh`): The device mesh this DTensor is distributed on.
        placements (Sequence[:class:`Placement`]]): The placements of the DTensor.
        skip_offset (bool): If True, skip computing the global offsets and return an empty
            tuple for global_offset. This can improve performance when only the local shape
            is needed. Defaults to False.

    Return:
        local_shape: the shape of the DTensor's _local_tensor on the current rank.
        global_offset: a tuple of offsets for each dimension of the global tensor shape,
        identifying how this shard fits into the global tensor in each dimension. If
        skip_offset is True, this will be an empty tuple.

    """
    empty_offset = ()
    if not mesh._is_current_rank_part_of_mesh():
        # if rank not in the mesh, return empty offset
        return ((0,), empty_offset)

    return _compute_local_shape_and_global_offset(
        global_shape, mesh.shape, mesh._sym_get_coordinate, placements, skip_offset
    )


@maybe_run_for_local_tensor
def _get_shard_size_and_offsets(
    curr_local_size: int,
    mesh_dim_size: int,
    rank: RankType,
    placement: Shard | _StridedShard,
    previous_offsets,
    zero_global_offset: int,
    skip_offset: bool,
) -> tuple[int, torch.Tensor | None]:
    kwargs: dict[str, Any] = {
        "curr_local_size": curr_local_size,
        "num_chunks": mesh_dim_size,
        "rank": rank,
    }
    if isinstance(placement, _StridedShard):
        kwargs["offset_mode"] = (
            _StridedShardOffsetMode.NONE if skip_offset else _StridedShardOffsetMode.ALL
        )
        # _StridedShard.local_shard_size_and_offset materializes the offsets list
        # via .tolist() on a (potentially fake) index tensor; under FakeTensorMode
        # that allocates one unbacked SymInt per element. Skip when the caller
        # discards the offsets anyway.
    shard_size, shard_offsets = placement._local_shard_size_and_offset(**kwargs)
    if skip_offset:
        return shard_size, None
    if shard_size == 0:
        return shard_size, torch.arange(zero_global_offset, zero_global_offset + 1)
    if isinstance(placement, Shard) and not isinstance(placement, _StridedShard):
        if not isinstance(shard_offsets, int):
            raise AssertionError
        index = torch.arange(shard_offsets, shard_offsets + shard_size)
    else:
        if not isinstance(shard_offsets, list):
            raise AssertionError
        index = torch.tensor(shard_offsets)
    if previous_offsets is None:
        return shard_size, index
    else:
        return shard_size, previous_offsets[index]


@maybe_run_for_local_tensor
def _get_first_offset(offsets: torch.Tensor) -> int:
    return int(offsets[0])


# accept 'plain data types' to enable simpler unit testing without creating device mesh
def _compute_local_shape_and_global_offset(
    global_shape: ShapeType,
    mesh_shape: ShapeType,
    my_coordinate: list[int] | Callable[[int], RankType] | None,
    placements: Sequence[Placement],
    skip_offset: bool = False,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Suppose you have a full tensor with size global_shape, and you have sharded
    it according to placements for mesh_shape.  This function returns, for a
    specific coordinate my_coordinate in the device mesh:

        - The size of your local shard WITHOUT padding (i.e., if you have
          an uneven split, your size might be smaller than the other entries
          in your dim), and

        - Where the data for your shard begins, in the full tensor.

    This function is fairly simple if your tensor is evenly sharded; the complication
    is around uneven splits.  There is also some complication for handling StridedShard,
    which changes the order you should apply sharding.

    Args:
        global_shape (ShapeType): The global shape of the tensor.
        mesh_shape (ShapeType): The shape of the device mesh.
        my_coordinate (Optional[list[int]]): The coordinate of the current rank in the device mesh.
        placements (Sequence[Placement]): The placements of the DTensor.
        skip_offset (bool): If True, skip computing the global offsets and return an empty
            tuple for global_offset. This can improve performance when only the local shape
            is needed. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - local_shape (tuple[int, ...]): The shape of the local shard on the current rank.
            - global_offset (tuple[int, ...]): The offsets for each dimension identifying where
              this shard begins in the global tensor. If skip_offset is True, this will be an
              empty tuple.
    """

    if isinstance(my_coordinate, (list, tuple)):
        _coord: list | tuple = my_coordinate

        def coordinate_lookup(dim: int) -> RankType:
            return _coord[dim]
    else:
        if my_coordinate is None:
            raise AssertionError
        coordinate_lookup = my_coordinate

    local_shape = list(global_shape)
    # Perform shard from left to right. For example,
    #   global tensor: [0, 1, 2, 3, 4, 5, 6, 7]
    #   placements: S(0), SS(0, split_factor=2)
    #   mesh_shape: (2, 2)
    # After S(0), shard_dim_to_global_offsets are
    #   {0: [0, 1, 2, 3]} on my_coordinate [0, 0] [0, 1]
    #   {0: [4, 5, 6, 7]} on my_coordinate [1, 0] [1, 1]
    # After SS(0, split_factor=2), shard_dim_to_global_offsets are
    #   {0: [0, 2]} on my_coordinate [0, 0]
    #   {0: [1, 3]} on my_coordinate [0, 1]
    #   {0: [4, 6]} on my_coordinate [1, 0]
    #   {0: [5, 7]} on my_coordinate [1, 1]
    shard_dim_to_global_offsets = {}
    for mesh_dim, placement in enumerate(placements):
        if not isinstance(placement, (Shard, _StridedShard)):
            continue
        shard_dim = placement.dim
        zero_global_offset = global_shape[shard_dim]
        if shard_dim >= len(local_shape):
            raise AssertionError(
                f"Sharding dim {shard_dim} greater than tensor ndim {len(local_shape)}"
            )
        previous_offsets = shard_dim_to_global_offsets.get(shard_dim)
        shard_size, shard_offsets = _get_shard_size_and_offsets(
            local_shape[shard_dim],
            mesh_shape[mesh_dim],
            coordinate_lookup(mesh_dim),
            placement,
            previous_offsets,
            zero_global_offset,
            skip_offset,
        )
        local_shape[shard_dim] = shard_size
        shard_dim_to_global_offsets[shard_dim] = shard_offsets
    if skip_offset:
        return tuple(local_shape), ()
    global_offset = [0] * len(global_shape)
    for shard_dim, global_offsets in shard_dim_to_global_offsets.items():
        global_offset[shard_dim] = _get_first_offset(global_offsets)
    return tuple(local_shape), tuple(global_offset)


compute_global_tensor_info = torch._C._DTensor_compute_global_tensor_info


def compute_local_tensor_info(
    global_tensor: torch.Tensor,
    mesh: DeviceMesh,
    placements: Sequence[Placement],
) -> tuple[list[int], list[int]]:
    """
    Compute the local size and stride of a DTensor from the given global tensor info.

    For example, if we have a global tensor with size (4, 8, 4) and stride (32, 1, 8).
    If the DTensor placements are [Shard(2)] and world_size is 2;
    then the local size is (4, 8, 2) and stride is (16, 1, 8).

    Args:
        tensor (:class:`torch.Tensor`):
            Global tensor which DTensor will distribute
        mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology
            of devices for the DTensor.
        placements (Sequence[:class:`Placement`]):
            The attribute of the DTensor that describes its layout
            on the mesh topology.

    Returns:
        local_shape: A List of int which specifies the size of the local tensor.
        local_stride: A List of int which specifies the stride of the local tensor.
    """
    local_shape = list(global_tensor.size())
    local_stride = list(global_tensor.stride())

    for idx, placement in enumerate(placements):
        mesh_dim_size = mesh.size(idx)
        if _is_shard_like(placement):
            if placement.dim < 0:
                raise AssertionError(
                    "Shard placements should have negative dims normalized in "
                    f"the user-facing APIs: {placement}"
                )
            shard_dim = placement.dim
            if shard_dim >= len(local_shape):
                raise AssertionError(
                    f"Sharding dim {shard_dim} greater than tensor ndim {len(local_shape)} "
                    f"for placement number {idx}."
                )

            global_dim_size = local_shape[shard_dim]
            if global_dim_size % mesh_dim_size != 0:
                raise AssertionError(
                    f"Global dim {global_dim_size} not divisible by mesh size {mesh_dim_size}"
                )
            local_shape[shard_dim] = global_dim_size // mesh_dim_size

            # shrink strides that were scaled up globally
            for i in range(len(local_stride)):
                if (
                    i != shard_dim
                    and local_stride[i] >= local_stride[shard_dim] * mesh_dim_size
                ):
                    local_stride[i] = local_stride[i] // mesh_dim_size

        elif not isinstance(placement, (Replicate, Partial)):
            raise RuntimeError(f"placement type {type(placement)} not supported!")

    return local_shape, local_stride


def compute_global_tensor_shape(
    shape: torch.Size, mesh: DeviceMesh, placements: Sequence[Placement]
) -> torch.Size:
    """
    Compute the global size of a DTensor from the given local tensor shape,
    the mesh and placements. Different from `compute_global_tensor_info`,
    which assumes sharding is even, this util allgathers local shards' shapes
    from all ranks and thus can support uneven sharding.
    NOTE: Currently this function only supports 1D mesh.

    Args:
        shape (:class:`torch.Size`):
            Shape of the local tensor
        mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology
            of devices for the DTensor.
        placements (Sequence[:class:`Placement`]]):
            The attribute of the DTensor that describes its layout
            on the mesh topology.

    Return:
        tensor_shape: Shape of the global DTensor.
    """
    if len(placements) != 1:
        raise NotImplementedError(
            "compute_global_tensor_shape only supports 1 placement for now."
        )

    if len(placements) != mesh.ndim:
        raise RuntimeError(
            "Expected one placement per mesh dim, "
            f"but found {len(placements)} placements and {mesh.ndim} mesh dims."
        )

    if isinstance(placements[0], Replicate):
        return shape
    # NOTE: isinstance(_, Shard) does not match _StridedShard; see _is_shard_like().
    elif isinstance(placements[0], Shard):

        @maybe_run_for_local_tensor
        def _create_local_shape_tensor(shape):
            return torch.tensor(list(shape), device=mesh.device_type)

        local_shape = _create_local_shape_tensor(shape)
        gathered_shaped_tensors = [
            torch.empty_like(local_shape, device=local_shape.device)
            for _ in range(mesh.size())
        ]
        funcol.all_gather_inplace(gathered_shaped_tensors, local_shape, mesh)

        @maybe_run_for_local_tensor
        def _validate_and_compute_global_shape(local_shape, gathered_shaped_tensors):
            sharded_dim_sum = 0
            shard_dim = placements[0].dim  # type: ignore[union-attr]
            other_dims = [d for d in range(len(shape)) if d != shard_dim]
            for shape_tensor in gathered_shaped_tensors:
                if not torch.equal(local_shape[other_dims], shape_tensor[other_dims]):
                    raise RuntimeError(
                        "Non-sharded dimensions should have identical size across ranks."
                    )
                shape_tensor_list = shape_tensor.tolist()
                sharded_dim_sum += shape_tensor_list[shard_dim]
            return sharded_dim_sum

        sharded_dim_sum = _validate_and_compute_global_shape(
            local_shape, gathered_shaped_tensors
        )
        global_shape = list(shape)
        global_shape[placements[0].dim] = sharded_dim_sum
        return torch.Size(global_shape)
    else:
        raise NotImplementedError(
            f"Placement type {type(placements[0])} not supported."
        )


def try_find_mesh_from_args(
    op_call: torch._ops.OpOverload, args: Sequence[object]
) -> DeviceMesh:
    """
    Find the device mesh object from args.
    It returns None if no mesh is found.
    NOTE: we can optimize this search if needed
    """
    for arg in args:
        if isinstance(arg, (dtensor.DTensor, DTensorSpec)):
            return arg.device_mesh
        elif (
            isinstance(arg, (list, tuple))
            and len(arg) > 0
            and isinstance(arg[0], (dtensor.DTensor, DTensorSpec))
        ):
            return arg[0].device_mesh

    raise ValueError(f"Cannot find device mesh from args for op : {op_call}.")


def compute_local_stride(
    global_stride: ShapeType, local_shape: ShapeType
) -> tuple[int, ...]:
    """
    Compute the stride of a local tensor shard, given the global stride and local shape.

    Derives strides by preserving the memory layout (dimension ordering) implied
    by the global strides, then computing contiguous strides for the local shape
    in that order.  Assumes the global tensor is non-overlapping and dense.
    """
    ndim = len(global_stride)
    # Sort dims by global stride descending to recover memory layout order.
    # Stable sort preserves original dim order for ties, which only occur
    # on size-1 dims where the stride value is semantically irrelevant.
    perm = sorted(range(ndim), key=lambda d: global_stride[d], reverse=True)
    local_strides = [0] * ndim
    s = 1
    for d in reversed(perm):
        local_strides[d] = s
        s *= local_shape[d]
    return tuple(local_strides)


def normalize_to_torch_size(size) -> torch.Size:  # type: ignore[no-untyped-def]
    """
    Unify variable types of size argument to torch.Size
    Acceptable types include:
        int, Sequence[int], Tuple[int], Tuple[Sequence[int]],
        or torch.Size
    """
    if isinstance(size, torch.Size):
        return size

    if isinstance(size, int):
        torch_size = [size]
    elif len(size) == 1 and isinstance(size[0], Sequence):
        torch_size = list(size[0])
    else:
        torch_size = list(size)
    return torch.Size(torch_size)


def assert_no_mixed_partial_types(placements: Sequence[Placement]) -> None:
    """
    Assert that a placement list doesn't contain mixed Partial reduce types.

    Mixed Partial types (e.g., ``Partial("sum")`` and ``Partial("max")`` together in the
    same placement list) are not supported and will raise a ``ValueError``. This restriction
    exists because nonlinear reductions (e.g., max) don't commute with linear reductions
    (e.g., sum), which means the relative ordering of different partial types would be
    semantically critical during redistribution. Rather than introducing complex ordering
    constraints, we prohibit mixing different Partial reduce types.

    Note: Partial("sum") and Partial("avg") DO commute with each other, so they can be ordered
    arbitrarily, and we allow this.

    This function is called internally by public APIs like :meth:`DTensor.from_local` and
    :func:`distribute_tensor` to validate placements early, before DTensor construction.

    Args:
        placements (Sequence[:class:`Placement`]): A sequence of placement specifications
            to validate.

    Raises:
        ValueError: If the placements contain more than one distinct Partial reduce type.
    """
    partial_reduce_ops: set[str] = set()
    for p in placements:
        if isinstance(p, Partial):
            partial_reduce_ops.add(p.reduce_op)

    if len(partial_reduce_ops) > 1 and partial_reduce_ops != {"sum", "avg"}:
        raise ValueError(
            f"Mixed Partial reduce types are not supported in the same placement list. "
            f"Found reduce ops: {partial_reduce_ops}. "
            f"Please ensure all Partial placements use the same reduce operation."
        )
<|endoftext|>import hashlib
import itertools
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, cast, NamedTuple

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import (
    _is_shard_like,
    _MaskPartial,
    _StridedShard,
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.utils._debug_mode import _stringify_shape
from torch.utils._dtype_abbrs import dtype_abbrs


# Defined here (not in placement_types.py) because decoding split_factor into
# a shard order is a DTensorSpec concern — placement_types doesn't know about
# shard orders.
class _StridedShardNotDecodableError(ValueError):
    """Raised when _StridedShard split_factor cannot be decoded into a shard order."""


class ShardOrderEntry(NamedTuple):
    """
    Represents how a single tensor dimension is sharded across mesh dimensions.

    Attributes:
        tensor_dim: The tensor dimension being sharded (e.g., 0, 1, 2 for a 3D tensor).
        mesh_dims: Tuple of mesh dimensions across which this tensor dimension is sharded,
                   in execution order. The first mesh dim is applied first, second is applied
                   second, etc. This tuple is guaranteed to be non-empty.

    Examples:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_DISTRIBUTED)
        >>> # Tensor dim 1 sharded across mesh dim 2, then mesh dim 0
        >>> ShardOrderEntry(tensor_dim=1, mesh_dims=(2, 0))

        >>> # Tensor dim 0 sharded only on mesh dim 1
        >>> ShardOrderEntry(tensor_dim=0, mesh_dims=(1,))
    """

    tensor_dim: int
    mesh_dims: tuple[int, ...]  # guaranteed to be non-empty


# Type alias for the complete shard order specification
# A tuple of ShardOrderEntry, one per sharded tensor dimension
#
# Example:
#   shard_order = (
#       ShardOrderEntry(tensor_dim=0, mesh_dims=(1,)),
#       ShardOrderEntry(tensor_dim=2, mesh_dims=(0, 3)),
#   )
#   This means:
#     - Tensor dimension 0 is sharded on mesh dimension 1
#     - Tensor dimension 2 is sharded on mesh dimension 0 first, then mesh dimension 3
ShardOrder = tuple[ShardOrderEntry, ...]


class TensorMeta(NamedTuple):
    # simple named tuple to represent tensor metadata
    # intentionally to stay simple only for sharding
    # propagation purposes.
    shape: torch.Size
    stride: tuple[int, ...]
    dtype: torch.dtype


# used internally to propagate the placements
@dataclass
class DTensorSpec:
    mesh: DeviceMesh
    placements: tuple[Placement, ...]

    # tensor meta will only be set during sharding propagation
    tensor_meta: TensorMeta | None = None

    # When a tensor dimension is sharded across multiple mesh axes,
    # `shard_order` specifies the sequence in which these shardings are applied.
    # This order determines how tensor shards are mapped and distributed across
    # devices.
    #
    # Example:
    #   For a tensor of shape [8, 16] and a 3D device mesh, if dim 0 is sharded over
    #   mesh dim 1, and dim 1 is sharded over mesh dim 0 and then mesh dim 2,
    #   the shard_order would be:
    #     shard_order = (
    #         ShardOrderEntry(tensor_dim=0, mesh_dims=(1,)),
    #         ShardOrderEntry(tensor_dim=1, mesh_dims=(0, 2)),
    #     )
    shard_order: ShardOrder = None  # type: ignore[assignment]

    # When True, _StridedShard placements encode the shard order, and the
    # shard_order field must be left as None (it will be derived on demand).
    # Set explicitly to False to treat _StridedShard as a regular Shard,
    # e.g., in view propagation.
    use_strided_shard_as_shard_order: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.placements, tuple):
            self.placements = tuple(self.placements)
        if self.use_strided_shard_as_shard_order is None:
            if any(isinstance(p, _StridedShard) for p in self.placements):
                self.use_strided_shard_as_shard_order = True
            else:
                self.use_strided_shard_as_shard_order = False
        if self.use_strided_shard_as_shard_order:
            if self.shard_order is not None:
                raise ValueError(
                    "DTensorSpec doesn't allow specify shard_order when "
                    "use_strided_shard_as_shard_order is True. This may result "
                    "in conflicting shard order."
                )
        else:
            if self.shard_order is None:
                self.shard_order = self.compute_default_shard_order(self.placements)

        self._hash: int | None = None

    @staticmethod
    def _normalize_placements_into_shard_order(
        placements: tuple[Placement, ...],
        mesh: DeviceMesh,
        use_strided_shard_as_shard_order: bool = True,
    ) -> tuple[tuple[Placement, ...], ShardOrder]:
        # If use_strided_shard_as_shard_order, it means the StridedShard/Shard
        # combinations should be interpreted as shard order.
        if use_strided_shard_as_shard_order:
            # _StridedShard in placements, try check if it can be decoded as shard order
            shard_order = DTensorSpec._maybe_convert_StridedShard_to_shard_order(
                placements, mesh
            )
            if shard_order is None:
                raise _StridedShardNotDecodableError(
                    f"_StridedShard placements {placements} cannot be decoded "
                    "into a corresponding shard_order"
                )
            normalized_placements = tuple(
                [
                    p if not isinstance(p, _StridedShard) else Shard(p.dim)
                    for p in placements
                ]
            )
            return normalized_placements, shard_order
        else:
            return placements, DTensorSpec.compute_default_shard_order(placements)

    @staticmethod
    def compute_default_shard_order(
        placements: tuple[Placement, ...],
    ) -> ShardOrder:
        """
        Compute the default shard order from placements.

        Returns a ShardOrder where each ShardOrderEntry maps a tensor dimension
        to the mesh dimensions it's sharded on, in left-to-right order.

        Args:
            placements: Tuple of Placement objects representing how a tensor is
                distributed across mesh dimensions.
        """
        # follow default left-to-right device order if shard_order is not specified
        tensor_dim_to_mesh_dims: defaultdict[int, list[int]] = defaultdict(list)
        mesh_ndim = len(placements)
        for mesh_dim in range(mesh_ndim):
            if _is_shard_like(placements[mesh_dim]):
                placement = placements[mesh_dim]
                shard_dim = placement.dim  # pyrefly: ignore [missing-attribute]
                if shard_dim < 0:
                    raise AssertionError(
                        f"Shard dim {shard_dim} in placements {placements} must be normalized"
                    )
                tensor_dim_to_mesh_dims[shard_dim].append(mesh_dim)

        # Convert dict into ShardOrderEntry tuples
        default_shard_order = tuple(
            ShardOrderEntry(tensor_dim=key, mesh_dims=tuple(value))
            for key, value in sorted(tensor_dim_to_mesh_dims.items())
            if value
        )
        return default_shard_order

    @staticmethod
    def _convert_shard_order_to_StridedShard(
        shard_order: ShardOrder, placements: tuple[Placement, ...], mesh: DeviceMesh
    ) -> tuple[Placement, ...]:
        """
        Convert ShardOrder to placements with _StridedShard.

        This function converts a ShardOrder specification into a tuple of Placement objects,
        using _StridedShard when a tensor dimension is sharded across multiple mesh dimensions
        in a non-default order. The split_factor of each _StridedShard is determined by the
        product of mesh dimension sizes that appear earlier in the shard order but later in
        the placement tuple.

        Args:
            shard_order: ShardOrder specification indicating which tensor dimensions are
                sharded on which mesh dimensions and in what execution order.
            placements: Tuple of Placement objects that does not contain _StridedShard.
            mesh: DeviceMesh containing the size information for each mesh dimension.

        Returns:
            Updated tuple of Placement objects with Shard or _StridedShard placements.

        Algorithm:
            For each ShardOrderEntry in shard_order:
              - For each mesh dimension in the entry's mesh_dims (in order):
                - Calculate split_factor as the product of mesh sizes for all mesh dimensions
                  that appear:
                  1. Earlier in the shard order (lower index in mesh_dims), and
                  2. Later in the placement tuple (higher mesh dimension index)
                - If split_factor == 1: use normal Shard
                - Otherwise: use _StridedShard with the calculated split_factor

        Example:
            >>> # xdoctest: +SKIP("Requires DeviceMesh")
            >>> # Tensor dimension 0 sharded on mesh dims [2, 0, 1] in that order
            >>> # mesh = DeviceMesh([4, 3, 2])  # sizes: mesh[0]=4, mesh[1]=3, mesh[2]=2
            >>> shard_order = (ShardOrderEntry(tensor_dim=0, mesh_dims=(2, 0, 1)),)
            >>> placements = (Shard(0), Shard(0), Shard(0))
            >>> # For mesh_dim=2 (index 0 in mesh_dims): no earlier dims, split_factor=1
            >>> #   -> placements[2] = Shard(0)
            >>> # For mesh_dim=0 (index 1 in mesh_dims): mesh_dim=2 is earlier and has index 2>0
            >>> #   -> split_factor = mesh.size(2) = 2
            >>> #   -> placements[0] = _StridedShard(0, split_factor=2)
            >>> # For mesh_dim=1 (index 2 in mesh_dims): mesh_dim=2 is earlier and has index 2>1
            >>> #   -> split_factor = mesh.size(2) = 2
            >>> #   -> placements[1] = _StridedShard(0, split_factor=2)
            >>> # Result: (_StridedShard(0, sf=2), _StridedShard(0, sf=2), Shard(0))
        """
        placements_list = list(placements)
        for entry in shard_order:
            tensor_dim = entry.tensor_dim
            mesh_dims = entry.mesh_dims
            for idx in range(len(mesh_dims)):
                # TODO(zpcore): split_factor from `view` and `shard order`
                # should be able to be multiplied into one. Need to loosen the
                # condition here.
                mesh_dim = mesh_dims[idx]
                if type(placements[mesh_dim]) is not Shard:
                    raise ValueError(
                        f"Only Shard placement can be converted to _StridedShard, "
                        f"found {placements[mesh_dim]} in {placements=}."
                    )
                split_factor = math.prod(
                    mesh.size(i) for i in mesh_dims[:idx] if i > mesh_dim
                )
                if split_factor == 1:
                    # use normal Shard
                    placements_list[mesh_dim] = Shard(tensor_dim)
                else:
                    placements_list[mesh_dim] = _StridedShard(
                        tensor_dim, split_factor=split_factor
                    )
        return tuple(placements_list)

    @staticmethod
    def _maybe_convert_StridedShard_to_shard_order(
        placements: tuple[Placement, ...], mesh: DeviceMesh
    ) -> ShardOrder | None:
        """
        Try to convert _StridedShard placements to ShardOrder.

        This is the inverse of `_convert_shard_order_to_StridedShard`. It reconstructs the shard
        order by examining the split_factor of each _StridedShard and determining its position
        in the execution order. If the _StridedShard configuration cannot be represented as a
        valid ShardOrder (i.e., there's no shard order that produces the observed split_factors),
        this function returns None.

        Args:
            placements: Tuple of Placement objects that may contain _StridedShard.
            mesh: DeviceMesh containing the size information for each mesh dimension.

        Returns:
            ShardOrder if conversion is possible, None otherwise. For placements without
            _StridedShard, returns the default shard order.

          Algorithm:
              1. If no _StridedShard in placements, return default shard order
              2. Create an empty list for each tensor dimension to represent mesh dim ordering
              3. Iterate through placements in reverse order (right to left):
                 - For each Shard/_StridedShard on a tensor dimension:
                   - Extract its split_factor (1 for Shard, split_factor for _StridedShard)
                   - Find the position in mesh_dims_order where accumulated_sf equals split_factor
                   - accumulated_sf is the product of mesh sizes of mesh dimensions that appear
                     earlier in mesh_dims_order (lower indices)
                   - Insert mesh_dim at the found position
              4. If no valid position found for any split_factor, return None (unable to convert)
              5. Construct ShardOrderEntry for each tensor dimension from mesh_dims_order

        Example:
            >>> # xdoctest: +SKIP("Requires DeviceMesh")
            >>> # mesh = DeviceMesh([4, 3, 2])  # sizes: mesh[0]=4, mesh[1]=3, mesh[2]=2
            >>> # placements = (_StridedShard(0, sf=2), _StridedShard(0, sf=2), Shard(0))
            >>> # Process tensor_dim=0 from right to left:
            >>> #   - mesh_dim=2: Shard(0) with sf=1
            >>> #     Try position 0: accumulated_sf=1, matches! Insert at position 0
            >>> #     Current mesh_dims_order order: [2]
            >>> #   - mesh_dim=1: _StridedShard(0, sf=2) with sf=2
            >>> #     Try position 0: accumulated_sf=1, no match
            >>> #     Try position 1: accumulated_sf=1*mesh.size(2)=2, matches! Insert at position 1
            >>> #     Current mesh_dims_order order: [2, 1]
            >>> #   - mesh_dim=0: _StridedShard(0, sf=2) with sf=2
            >>> #     Try position 0: accumulated_sf=1, no match
            >>> #     Try position 1: accumulated_sf=1*mesh.size(2)=2, matches! Insert at position 1
            >>> #     Final mesh_dims_order order: [2, 0, 1]
            >>> # Result: ShardOrder((ShardOrderEntry(tensor_dim=0, mesh_dims=(2, 0, 1)),))
            >>> # This means: first shard on mesh_dim=2, then mesh_dim=0, then mesh_dim=1

        Note:
            This function validates that _StridedShard can be represented as a ShardOrder.
            Not all _StridedShard configurations are valid - the split_factor must match
            the product of mesh sizes in some execution order.
        """
        if not any(isinstance(p, _StridedShard) for p in placements):
            return DTensorSpec.compute_default_shard_order(placements)
        max_tensor_dim = max([i.dim for i in placements if _is_shard_like(i)]) + 1
        shard_order = []

        tensor_dim_to_mesh_dims_order: list[list[int]] = [
            [] for i in range(max_tensor_dim)
        ]
        for mesh_dim in reversed(range(len(placements))):
            cur_placement = placements[mesh_dim]
            if _is_shard_like(cur_placement):
                tensor_dim = cur_placement.dim
                mesh_dims_order = tensor_dim_to_mesh_dims_order[tensor_dim]
                cur_sf = 1
                if isinstance(cur_placement, _StridedShard):
                    cur_sf = cur_placement.split_factor
                accumulated_sf = 1
                find_order = False
                for i in range(len(mesh_dims_order) + 1):
                    if accumulated_sf == cur_sf:
                        mesh_dims_order.insert(i, mesh_dim)
                        find_order = True
                        break
                    if i < len(mesh_dims_order):
                        accumulated_sf *= mesh.size(mesh_dims_order[i])
                if not find_order:
                    # _StridedShard is not convertible to ShardOrder
                    return None
            else:
                if not isinstance(cur_placement, Replicate | Partial | _MaskPartial):
                    raise ValueError(
                        f"Unsupported placement type {type(cur_placement)} encountered in "
                        f"{placements}; expected Replicate, Partial, or _MaskPartial."
                    )
        for tensor_dim in range(max_tensor_dim):
            if len(tensor_dim_to_mesh_dims_order[tensor_dim]) > 0:
                shard_order.append(
                    ShardOrderEntry(
                        tensor_dim=tensor_dim,
                        mesh_dims=tuple(tensor_dim_to_mesh_dims_order[tensor_dim]),
                    )
                )
        return tuple(shard_order)

    def _verify_shard_order(self, shard_order: ShardOrder) -> None:
        """Verify that the shard_order is valid and matches the placements."""
        total_shard = 0
        if any(isinstance(p, _StridedShard) for p in self.placements):
            # _StridedShard shard_order validation not yet supported;
            # the Shard-only checks below (line 390, 394) would fail.
            return
        prev_tensor_dim = -1
        for entry in shard_order:
            tensor_dim = entry.tensor_dim
            mesh_dims = entry.mesh_dims
            if len(mesh_dims) <= 0:
                raise AssertionError(f"shard_order {shard_order} has empty mesh dim")
            if tensor_dim < 0:
                raise AssertionError(
                    f"shard_order {shard_order} has invalid tensor dim {tensor_dim}"
                )
            if tensor_dim <= prev_tensor_dim:
                raise AssertionError("tensor dim should be sorted in shard_order")
            prev_tensor_dim = tensor_dim
            total_shard += len(mesh_dims)
            for mesh_dim in mesh_dims:
                if not (0 <= mesh_dim < len(self.placements)):
                    raise AssertionError(
                        f"shard_order {shard_order} has invalid mesh dim {mesh_dims}"
                    )
                if self.placements[mesh_dim] != Shard(tensor_dim):
                    raise AssertionError(
                        f"placement[{mesh_dim}] doesn't have a matching shard in shard_order"
                    )
        if total_shard != sum(1 for p in self.placements if isinstance(p, Shard)):
            raise AssertionError

    def __setattr__(self, attr: str, value: Any) -> None:
        if attr == "shard_order" and value is not None:
            self._verify_shard_order(value)
        super().__setattr__(attr, value)
        # Make sure to recompute the hash in case any of the hashed attributes
        # change (though we do not expect `mesh`, `placements` or `shard_order`
        # to change)
        if hasattr(self, "_hash") and attr in (
            "mesh",
            "placements",
            "tensor_meta",
            "shard_order",
        ):
            self._hash = None
        # This assert was triggered by buggy handling for dict outputs in some
        # FX passes, where you accidentally iterate over a dict and try to put
        # keys into TensorMeta.  See https://github.com/pytorch/pytorch/issues/157919
        if attr == "tensor_meta" and value is not None:
            from torch.fx.passes.shape_prop import TensorMetadata

            # TODO: the TensorMetadata arises from
            # test/distributed/tensor/experimental/test_tp_transform.py::TensorParallelTest::test_tp_transform_e2e
            # but I actually can't reproduce it, maybe it is also a bug!
            if not isinstance(value, TensorMeta | TensorMetadata):
                raise AssertionError(repr(value))

    def _hash_key(self) -> tuple[Any, ...]:
        """Return the tuple used for hashing. Used by both __hash__ and _stable_hash."""
        if self.tensor_meta is not None:
            return (
                self.mesh,
                self.placements,
                self.shard_order,
                self.tensor_meta.shape,
                self.tensor_meta.stride,
                self.tensor_meta.dtype,
            )
        return (self.mesh, self.placements, self.shard_order)

    def _hash_impl(self) -> int:
        # hashing and equality check for DTensorSpec are used to cache the sharding
        # propagation results. We only need to consider the mesh, placements, shape
        # dtype and stride.
        # Caveat: we need to keep this in mind and sync hash and eq if we add more
        # fields to them.
        return hash(self._hash_key())

    def __hash__(self) -> int:
        # We lazily cache the spec to avoid recomputing the hash upon each
        # use, where we make sure to update the hash when the `tensor_meta`
        # changes by overriding `__setattr__`. This must be lazy so that Dynamo
        # does not try to hash non-singleton `SymInt`s for the stride.
        if self._hash is None:
            self._hash = self._hash_impl()
        return self._hash

    def _stable_hash(self) -> str:
        """
        Return a stable hash for AOT autograd caching.
        [See note: Tensor subclass stable hashing for AOT autograd cache]
        """
        from torch._library.fake_class_registry import maybe_unwrap_fake_script_object

        # Get hash key, but replace mesh with its stable hash
        key = self._hash_key()
        # First element is mesh, replace with its stable hash
        mesh = maybe_unwrap_fake_script_object(self.mesh)
        stable_key = (mesh._stable_hash(),) + key[1:]
        return hashlib.blake2b(repr(stable_key).encode(), digest_size=16).hexdigest()

    def _check_equals(self, other: object, skip_shapes: bool = False) -> bool:
        if not (
            isinstance(other, DTensorSpec)
            and self.mesh == other.mesh
            and self.placements == other.placements
            and self.shard_order == other.shard_order
        ):
            return False
        if self.tensor_meta is None or other.tensor_meta is None:
            return self.tensor_meta == other.tensor_meta

        if skip_shapes:
            return self.tensor_meta.dtype == other.tensor_meta.dtype
        return (
            self.tensor_meta.shape == other.tensor_meta.shape  # type: ignore[union-attr]
            and self.tensor_meta.stride == other.tensor_meta.stride  # type: ignore[union-attr]
            and self.tensor_meta.dtype == other.tensor_meta.dtype  # type: ignore[union-attr]
        )

    def __eq__(self, other: object, /) -> bool:
        return self._check_equals(other)

    def __str__(self) -> str:
        """
        human readable representation of the DTensorSpec
        """
        placement_str = self.format_shard_order_str(self.placements, self.shard_order)
        if self.tensor_meta is not None:
            tensor_shape = _stringify_shape(self.tensor_meta.shape)
            tensor_dtype = dtype_abbrs[self.tensor_meta.dtype]
        else:
            tensor_shape = "unknown shape"
            tensor_dtype = "unknown dtype"

        return f"Spec({tensor_dtype}{tensor_shape}({placement_str}))"

    @staticmethod
    def is_default_device_order(shard_order: ShardOrder) -> bool:
        """
        Check if the device order is the default left-to-right order.
        """
        if shard_order is None:
            # Missing `shard_order` attribute, possibly due to _StridedShard in
            # placements.
            return False
        for entry in shard_order:
            mesh_dims = entry.mesh_dims
            is_increasing = all(
                prev < nxt for prev, nxt in itertools.pairwise(mesh_dims)
            )
            if not is_increasing:
                return False
        return True

    @staticmethod
    def format_shard_order_str(
        placements: tuple[Placement, ...],
        shard_order: ShardOrder | None = None,
    ) -> str:
        """
        Format DTensor sharding information as a human-readable string.

        This method formats the sharding pattern in mesh-centric order, showing the placement
        for each mesh dimension sequentially. When a tensor dimension is sharded across multiple
        mesh dimensions, the order index indicates the execution sequence of the sharding operations.

        Args:
            placements: Tuple of placement objects for each mesh dimension.
            shard_order: Optional ShardOrder specifying the sharding order.

        Returns:
            String representation of the sharding pattern in mesh-centric format.

        Example:
            For a 3D tensor on a 2x2x2x2 mesh (16 devices) with::

                placements = [Partial(), Shard(1), Shard(1), Replicate()]
                shard_order = (ShardOrderEntry(tensor_dim=1, mesh_dims=(2, 1)),)

            Mesh configuration:
                - mesh_dim_0: Partial reduction (sum)
                - mesh_dim_1: Shard tensor dimension 1 (executed second, order index 1)
                - mesh_dim_2: Shard tensor dimension 1 (executed first, order index 0)
                - mesh_dim_3: Replicate

            Output: ``"PS(1)[1]S(1)[0]R"``

            Explanation:
                - ``P``: mesh dimension 0 has partial reduction
                - ``S(1)[1]``: mesh dimension 1 shards tensor dimension 1 (order index 1 means second)
                - ``S(1)[0]``: mesh dimension 2 shards tensor dimension 1 (order index 0 means first)
                - ``R``: mesh dimension 3 replicates

            The format follows mesh dimension order (0, 1, 2, 3), and when a tensor dimension
            is sharded across multiple mesh dimensions, the bracketed index shows the execution
            order: ``[0]`` is executed first, ``[1]`` is executed second, etc.
        """
        out_str = ""
        # native dtensor-style sharding representation: map from mesh
        # dim to tensor dim
        for mesh_dim, placement in enumerate(placements):
            if _is_shard_like(placement):
                if shard_order is not None:
                    for entry in shard_order:
                        tensor_dim = entry.tensor_dim
                        mesh_dims = entry.mesh_dims

                        if placement.dim == tensor_dim:
                            if mesh_dim not in mesh_dims:
                                raise AssertionError
                            if len(mesh_dims) > 1:
                                out_str += f"{placement}[{mesh_dims.index(mesh_dim)}]"
                            else:
                                # no need to show device order if the tensor dim is
                                # only sharded in one mesh dim
                                out_str += str(placement)
                            break
                else:
                    out_str += str(placement)
            else:
                out_str += str(placement)
        return out_str

    @property
    def shape(self) -> torch.Size:
        if self.tensor_meta is None:
            raise ValueError("tensor_meta is not set")
        return self.tensor_meta.shape

    @property
    def stride(self) -> tuple[int, ...]:
        if self.tensor_meta is None:
            raise ValueError("tensor_meta is not set")
        return self.tensor_meta.stride

    @property
    def ndim(self) -> int:
        if self.tensor_meta is None:
            raise ValueError("tensor_meta is not set")
        return len(self.tensor_meta.shape)

    @property
    def num_shards(self) -> int:
        num_shards = 1
        for i, placement in enumerate(self.placements):
            if _is_shard_like(placement):
                num_shards *= self.mesh.size(i)
        return num_shards

    @property
    def device_mesh(self) -> DeviceMesh:
        # simple aliasing for the mesh field, make some
        # checks that mixes DTensor/DTensorSpec easier
        return self.mesh

    @property
    def dim_map(self) -> list[int]:
        """
        dim_map is a property we derive from `placements` of
        the distributed tensor. It simply return a list of ints
        where dim_map[i] denotes the sharding mapping to the mesh
        dimension, and len(dim_map) == dist_tensor.ndim
        dim_map[i] = -1: means tensor dim i replicate on mesh
        dim_map[i] = j: means tensor dim i shard on mesh dim j

        For example, we have a dist tensor that have the shape of
        [18, 20, 30], and device_mesh([0, 1, 2, 3]), placements:
        [Shard(1)], the dim_map of this placement would be:
        [-1, 0, -1]. This representation is pretty helpful during
        sharding propagation where we could know exactly each
        tensor dimension is sharded or not.

        Note that if placements contains `_Partial`, we have to
        explicitly deal with it, so that when we create a DTensorSpec
        with dim_map, we could properly record the pending sums.
        """
        # dims mapping of dist tensor sharding
        # return size of tensor ndim, -1 represent replicate
        # and int >=0 represent shard on that device mesh dim
        r = [-1] * self.ndim
        for i, placement in enumerate(self.placements):
            if _is_shard_like(placement):
                shard_dim = placement.dim
                if r[shard_dim] > -1:
                    raise ValueError(
                        f"Tensor dim {shard_dim} is already sharded on mesh dim {r[shard_dim]},"
                        " DTensor operator implementation does not support things like hybrid"
                        " sharding strategies yet (i.e. [Shard(0), Shard(0)])"
                    )
                r[shard_dim] = i
        return r

    @property
    def num_shards_map(self) -> list[int]:
        """
        dim_map is a property we derive from `placements` of
        the distributed tensor. Unlike `dim_map`, `num_shards_map`
        denotes how many shards each tensor dim has. Like `dim_map`:
            len(num_shards_map) == dist_tensor.ndim
            num_shards_map[i] = 1: means tensor dim i is not sharded
            num_shards_map[i] = j: means tensor dim i has j shards in total

        For example, we have a dist tensor of shape [18, 20, 30],
        a device_mesh ([[0, 1, 2, 3], [4, 5, 6, 7]]), and placements
        ([Shard(1), Shard(0)]), the num_shards_map of this distributed tensor
        would be: [4, 2, 1].
        """
        r = [1] * self.ndim
        for i, placement in enumerate(self.placements):
            if _is_shard_like(placement):
                r[placement.dim] *= self.mesh.size(i)

        return r

    @property
    def sums(self) -> list[int]:
        """
        sums is a property we derive from `placements` of the
        distributed tensor. It simply return a list of ints where
        sums[i] denotes the pending sum (partial) on mesh dim i
        """
        return [
            idx
            for idx, placement in enumerate(self.placements)
            if placement.is_partial()
        ]

    @classmethod
    def from_dim_map(
        cls,
        mesh: DeviceMesh,
        dim_map: list[int],
        sums: list[int],
        tensor_meta: TensorMeta | None = None,
    ) -> "DTensorSpec":
        """
        Construct a DTensorSpec from dim_map list and pending sum.

        Args:
            mesh (class:`DeviceMesh`): device mesh to be used in the DTensorSpec
            dim_map (List[int]): a list of integer that represents sharding on each
                tensor dimension, see `dim_map` property doc for details
            sums (List[int]): a list of integer that represents the dist tensor have
                pending sum on which device mesh dimension.
            tensor meta (TensorMeta): DTensor metadata

        Return:
            a class:`DTensorSpec` object
        """
        # by default replicate on device mesh dims
        placements: list[Placement] = [Replicate() for _ in range(mesh.ndim)]

        # find all mesh dims that need pending reductions
        for s in sums:
            placements[s] = Partial()

        for i, m in enumerate(dim_map):
            if m >= 0:
                placement = placements[m]
                if placement.is_shard():  # dim_map only produces Shard placements
                    placement = cast(Shard, placement)
                    raise RuntimeError(
                        f"DeviceMesh dimension can't be mapped to two dimension of the same tensor: {i} and {placement.dim}"
                    )
                elif placement.is_partial():
                    raise RuntimeError(
                        f"DeviceMesh dimension {m} cannot be both shard and partial!"
                    )
                placements[m] = Shard(i)

        return cls(mesh, tuple(placements), tensor_meta=tensor_meta)

    def is_replicated(self) -> bool:
        """
        return True if the current DTensorSpec replicates on all mesh dims (devices)
        """
        return all(placement.is_replicate() for placement in self.placements)

    def is_sharded(self) -> bool:
        """
        return True if the current DTensorSpec uses Shard() or _StridedShard() placement on any mesh dims (devices)
        """
        return any(_is_shard_like(placement) for placement in self.placements)

    def shallow_copy_with_tensor_meta(
        self, tensor_meta: TensorMeta | None
    ) -> "DTensorSpec":
        """
        Shallow copy the DTensorSpec with a new tensor_meta.
        """
        if tensor_meta is None:
            raise AssertionError("shallow copy with no tensor_meta!")
        return DTensorSpec(
            self.mesh,
            self.placements,
            tensor_meta=tensor_meta,
            use_strided_shard_as_shard_order=self.use_strided_shard_as_shard_order,
        )
<|endoftext|># mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import copy
import hashlib
import inspect
import warnings
from collections.abc import Callable, Sequence
from typing import Any
from typing_extensions import deprecated

import torch
import torch.distributed.tensor._dispatch as op_dispatch
import torch.distributed.tensor._random as random
import torch.nn as nn
from torch._export.wrappers import mark_subclass_constructor_exportable_experimental
from torch._logging import LazyString
from torch.distributed._local_tensor import maybe_run_for_local_tensor
from torch.distributed.device_mesh import (
    _mesh_resources,
    _register_distributed_opaque_types,
    DeviceMesh,
)
from torch.distributed.tensor._collective_utils import check_tensor_meta, mesh_broadcast
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._redistribute import (
    Redistribute,
    redistribute_local_tensor,
)
from torch.distributed.tensor._utils import (
    assert_no_mixed_partial_types,
    compute_global_tensor_info,
    compute_local_shape_and_global_offset,
    ExplicitRedistributionContext,
    normalize_to_torch_size,
)
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Partial,
    Placement,
    Replicate,
    Shard,
)


__all__ = [
    "DTensor",
    "distribute_tensor",
    "distribute_module",
    "ones",
    "empty",
    "full",
    "linspace",
    "logspace",
    "rand",
    "randn",
    "zeros",
]

aten = torch.ops.aten


def _normalize_placements_for_grad(
    placements: tuple[Placement, ...],
) -> tuple[Placement, ...]:
    """
    Normalize gradient placements by converting Partial to Replicate.

    See the gradient placement guarantees documented in DTensor.from_local's docstring
    for why Partial forward placements map to Replicate gradient placements. We do this
    for both from_local and to_local backward.
    """
    normalized: list[Placement] = []
    for p in placements:
        if p.is_partial():
            normalized.append(Replicate())
        else:
            normalized.append(p)
    return tuple(normalized)


# NOTE [Autograd interaction between torch.Tensor]
#
# The autograd functions defined below are being used by the public
# facing APIs (i.e. from_local, to_local) to ensure DTensor to work
# together with torch.Tensor within the autograd engine. This
# allows DTensor to only exist on part of the module hierarchy.
#
# As an example, we have a module that consists of submodules
# A, B, and C, the execution flow would be like:
#  input(torch.Tensor) -> Module A -> Module B -> Module C -> output (torch.Tensor)
#
# Suppose I only want to make Module B be a sharded module with
# DTensor params, the following forward/backward should work:
#
#  input(torch.Tensor) -> Module A
#       -> DTensor input (from_local) -> Sharded Module B -> DTensor output
#           -> torch.Tensor output (to_local) -> Module C
#
# So from_local/to_local must be Autograd functions.
#
class _ToTorchTensor(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        input: "DTensor",
        grad_placements: Sequence[Placement] | None,
    ):
        ctx.dtensor_spec = input._spec
        ctx.grad_placements = grad_placements
        ctx.set_materialize_grads(False)
        local_tensor = input._local_tensor
        ctx.local_tensor_stride = local_tensor.stride()

        # We need to return a fresh Tensor object there as autograd metadata
        # will be inplaced into it. So we don't want to pollute the Tensor
        # object stored in the _local_tensor of this DTensor.
        return local_tensor.view_as(local_tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor | None):  # type: ignore[override]
        if grad_output is None:
            return None, None

        dtensor_spec = ctx.dtensor_spec
        mesh = dtensor_spec.mesh
        grad_placements = ctx.grad_placements
        dtensor_meta = dtensor_spec.tensor_meta

        # user should provide grad_placements as there's no guarantee on input gradient placement
        # if grad_placement is None, we provide default placement
        if grad_placements is None:
            # See DTensor.from_local docstring for gradient placement guarantees
            grad_placements = _normalize_placements_for_grad(dtensor_spec.placements)

        from torch._prims_common import check_contiguous_sizes_strides
        from torch.fx.experimental.symbolic_shapes import guard_or_false

        def strides_provably_match(stride, other):
            return len(stride) == len(other) and all(
                guard_or_false(a == b) for a, b in zip(stride, other)
            )

        def strides_provably_differ(stride, other):
            return len(stride) != len(other) or any(
                guard_or_false(a != b) for a, b in zip(stride, other)
            )

        def strides_are_contiguous_for_shape(shape, stride, other):
            if not (len(shape) == len(stride) == len(other)):
                return False
            return check_contiguous_sizes_strides(
                shape, stride, false_if_dde=True
            ) and check_contiguous_sizes_strides(shape, other, false_if_dde=True)

        def strides_may_match(shape, stride, other):
            return (
                strides_provably_match(stride, other)
                or strides_are_contiguous_for_shape(shape, stride, other)
                or not strides_provably_differ(stride, other)
            )

        def check_stride_matches(shape, stride, other):
            if strides_provably_match(
                stride, other
            ) or strides_are_contiguous_for_shape(shape, stride, other):
                return
            for a, b in zip(stride, other):
                torch._check(
                    a == b,
                    lambda: (
                        f"Expected matching DTensor local strides, got {stride} and {other}"
                    ),
                )

        # Same-placement to_local() backward preserves the forward DTensor
        # layout. With symbolic shapes, the local grad stride can be equivalent
        # to the saved stride while carrying different symbol provenance.
        if (
            grad_placements == dtensor_spec.placements
            and strides_may_match(
                grad_output.shape, grad_output.stride(), ctx.local_tensor_stride
            )
            and strides_may_match(
                dtensor_meta.shape, ctx.local_tensor_stride, dtensor_meta.stride
            )
        ):
            check_stride_matches(
                grad_output.shape, grad_output.stride(), ctx.local_tensor_stride
            )
            check_stride_matches(
                dtensor_meta.shape, ctx.local_tensor_stride, dtensor_meta.stride
            )
            # Avoid actual sharing of specs in case they're modified during (e.g.)
            # sharding propagation.
            grad_spec = copy.copy(dtensor_spec)
        else:
            _, tensor_stride = compute_global_tensor_info(
                grad_output, mesh, dtensor_spec.placements
            )
            tensor_stride = tuple(tensor_stride)
            grad_spec = DTensorSpec(
                mesh,
                grad_placements,
                tensor_meta=TensorMeta(
                    shape=dtensor_meta.shape,
                    stride=tensor_stride,
                    dtype=dtensor_meta.dtype,
                ),
            )
        return (
            DTensor.from_local(
                grad_output,
                grad_spec.device_mesh,
                grad_spec.placements,
                shape=grad_spec.shape,
                stride=grad_spec.stride,
            ),
            None,
        )


class _FromTorchTensor(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,  # pyre-ignore[2]: Parameter must be annotated.
        input: torch.Tensor,
        device_mesh: DeviceMesh,
        placements: tuple[Placement, ...],
        run_check: bool,
        shape: torch.Size | None = None,
        stride: tuple[int, ...] | None = None,
        grad_placements: tuple[Placement, ...] | None = None,
    ) -> "DTensor":
        ctx.forward_input_placements = placements
        ctx.forward_input_device_mesh = device_mesh
        ctx.grad_placements = grad_placements
        ctx.set_materialize_grads(False)

        if shape and stride:
            tensor_shape, tensor_stride = shape, stride
        elif not shape and not stride:
            # if it's not by default run_check, we assume user is certain that each
            # rank has the same tensor shape, and we just use that to calculate the
            # global shape
            global_shape, global_stride = compute_global_tensor_info(
                input, device_mesh, placements
            )
            tensor_shape, tensor_stride = torch.Size(global_shape), tuple(global_stride)
        else:
            raise RuntimeError(
                f"Found shape:{shape}, stride:{stride}.",
                "Please pass both shape and stride at the same time.",
            )

        if not device_mesh._is_current_rank_part_of_mesh():
            # if the global rank is not participating in the device mesh, we
            # simply set the local tensor to an empty tensor
            input = input.new_empty(0, requires_grad=input.requires_grad)
        elif run_check:
            # TODO: support uneven sharding when global shape/stride not passed, by
            # building the global TensorMeta during check_tensor_meta
            check_shape_stride = not shape and not stride
            check_tensor_meta(input, check_shape_stride=check_shape_stride)
            # TODO: See if we need to make this run_check logic
            # have a corresponding backward.
            for idx, placement in enumerate(placements):
                if placement.is_replicate():
                    # broadcast rank 0 tensor to all ranks
                    # only broadcast if run_check is True
                    input = input.contiguous()
                    mesh_broadcast(input, device_mesh, mesh_dim=idx)

        dist_spec = DTensorSpec(
            device_mesh,
            placements,
            tensor_meta=TensorMeta(
                tensor_shape,
                tensor_stride,
                input.dtype,
            ),
        )

        # We want a fresh Tensor object that shares memory with the input tensor
        # pyrefly: ignore [bad-argument-type]
        dist_tensor = DTensor(
            # pyrefly: ignore [bad-argument-count]
            input.view_as(input),
            dist_spec,
            # requires_grad of the dist tensor depends on if input
            # requires_grad or not
            # pyrefly: ignore [unexpected-keyword]
            requires_grad=input.requires_grad,
        )
        return dist_tensor

    @staticmethod
    def backward(ctx, grad_output: "DTensor | None"):  # type: ignore[override]
        forward_input_placements = ctx.forward_input_placements
        forward_input_device_mesh = ctx.forward_input_device_mesh
        grad_placements = ctx.grad_placements

        if grad_output is None:
            return None, None, None, None, None, None, None

        # reshard to the placement when creating DistributedTensor
        # so that the gradient layout matches, and we could return
        # local gradients directly
        # if forward placement is partial, we always redistribute grad_input to Replicate:
        # Partial(fwd) - Replicate(grad_output) - Replicate(grad_input): no redistribution needed
        # Partial(fwd) - Partial(grad_output) - Replicate(grad_input): redistribute Partial to Replicate

        # The second case is BC breaking:
        # was: Partial(fwd) - Partial(grad_output) - Partial(grad_input)
        # now: Partial(fwd) - Partial(grad_output) - Replicate(grad_input)
        # See DTensor.from_local docstring for gradient placement guarantees

        # user should provide grad_placements as there's no guarantee on input gradient placement
        # if grad_placements is None, we provide default placement
        if grad_placements is not None:
            normalized_placements = grad_placements
        else:
            normalized_placements = _normalize_placements_for_grad(
                forward_input_placements
            )

        if grad_output.placements != normalized_placements:
            current_spec: DTensorSpec = grad_output._spec
            target_spec = DTensorSpec(
                forward_input_device_mesh,
                normalized_placements,
                tensor_meta=grad_output._spec.tensor_meta,
            )
            ExplicitRedistributionContext.observe_redistribution(
                current_spec,
                target_spec,
                LazyString(
                    lambda: "Implicit redistribution occurred in DTensor.from_local backward"
                ),
            )
            local_tensor = grad_output._local_tensor
            output = redistribute_local_tensor(
                local_tensor,
                current_spec,
                target_spec,
            )
            # TODO: return the redistributed local tensor directly without
            # differentiable backward. see if this make sense for all cases.
            return output, None, None, None, None, None, None

        # TODO: backward is also differentiable now, add a test
        # to test higher level gradients.
        return grad_output.to_local(), None, None, None, None, None, None


class DTensor(torch.Tensor):
    """
    ``DTensor`` (Distributed Tensor) is a subclass of ``torch.Tensor`` that provides single-device like
    abstraction to program with multi-device ``torch.Tensor``. It describes the distributed tensor sharding
    layout (DTensor Layout) through the :class:`DeviceMesh` and following types of :class:`Placement`:

    * :class:`Shard`: Tensor sharded on the tensor dimension ``dim`` on the devices of the ``DeviceMesh`` dimension
    * :class:`Replicate`: Tensor replicated on the devices of the ``DeviceMesh`` dimension
    * :class:`Partial`: Tensor is pending reduction on the devices of the ``DeviceMesh`` dimension

    When calling PyTorch operators, ``DTensor`` overrides the PyTorch operators to perform sharded computation and issue
    communications whenever necessary. Along with the operator computation, ``DTensor`` will transform or propagate the
    placements (DTensor Layout) properly (based on the operator semantic itself) and generate new ``DTensor`` outputs.

    To ensure numerical correctness of the ``DTensor`` sharded computation when calling PyTorch operators, ``DTensor``
    requires every Tensor argument of the operator be DTensor.

    .. note:: Directly using the Tensor subclass constructor here is not the recommended way to create a ``DTensor``
        (i.e. it does not handle autograd correctly hence is not the public API). Please refer to the `create_dtensor`_
        section to see how to create a ``DTensor``.
    """

    _local_tensor: torch.Tensor
    _spec: DTensorSpec
    __slots__ = ["_local_tensor", "_spec"]

    # _op_dispatcher instance as a class attribute to handle runtime dispatching logic
    _op_dispatcher: op_dispatch.OpDispatcher = op_dispatch.OpDispatcher()

    # This implementation is just to convince mypy _spec and _local_tensor are
    # initialized; it is immediately overridden below.
    def __new__(
        cls,
        local_tensor: torch.Tensor,
        spec: DTensorSpec,
        *,
        requires_grad: bool,
    ) -> "DTensor":
        r = torch.Tensor._dtensor__new__(
            cls, local_tensor, spec, requires_grad=requires_grad
        )
        r._spec = spec
        r._local_tensor = local_tensor
        return r

    __new__ = torch.Tensor._dtensor__new__  # type: ignore[assignment] # noqa: F811

    @torch._disable_dynamo
    @mark_subclass_constructor_exportable_experimental
    def __init__(self, *args, **kwargs):
        """
        Construct a DTensor from a local tensor, device mesh, and placement and
        other tensor properties (i.e. shape, requires_grad, strides, etc).
        .. note:: This is not a public API and it's only supposed to be used by the
            operator implementations and internals. If you want to construct a
            DTensor from a local tensor, consider using ``DTensor.from_local``, if
            you want to construct a DTensor from a "global" tensor (where you
            already have tensor initialized and want to shard this tensor),
            consider using ``distribute_tensor``.
        """
        super().__init__()

    # pyre-fixme[14]: `__repr__` overrides method defined in `DTensor` inconsistently.
    # pyre-fixme[3]: Return type must be annotated.
    def __repr__(self):  # type: ignore[override]
        # TODO: consider all_gather the local tensors for better debugging
        return f"DTensor(local_tensor={self._local_tensor}, device_mesh={self._spec.mesh}, placements={self._spec.placements})"

    def __tensor_flatten__(self):
        """
        protocol to inform how to flatten a DTensor to local tensor
        for PT2 tracing
        """
        return ["_local_tensor", "device_mesh"], (
            self._spec.placements,
            self._spec.tensor_meta,
            self._spec.shard_order,
            self.requires_grad,
        )

    @staticmethod
