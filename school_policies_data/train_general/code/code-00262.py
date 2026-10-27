            if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
                # TODO(voz): Don't graph break on this - dynamo hates the i1 != i2
                # tensor comparison control flow.
                # https://github.com/pytorch/pytorch/issues/107055
                for (r1, i1), (r2, i2) in itertools.combinations(
                    (
                        (
                            rank,
                            world_indices[
                                rank * num_valid_indices : (rank + 1)
                                * num_valid_indices
                            ],
                        )
                        for rank in range(self.world_size)
                    ),
                    2,
                ):
                    if i1 != i2:
                        r1_param_names = self._get_names_from_handle_indices(i1)
                        r2_param_names = self._get_names_from_handle_indices(i2)
                        raise RuntimeError(
                            f"{msg_prefix} rank {r1} is all-gathering parameters "
                            f"for {r1_param_names} while rank {r2} is all-gathering "
                            f"parameters for {r2_param_names}"
                        )
        else:
            # Only issue warnings on the first deviating iteration and stop
            # checking thereafter to avoid flooding the console
            if self.warn_status == _ExecOrderWarnStatus.WARNED:
                return
            msg_prefix = None  # non-`None` means we should warn
            if self.current_order_index >= len(self.handles_pre_forward_order):
                # This iteration sees extra all-gather(s) compared to the first
                msg_prefix = (
                    "Expected to not all-gather any more parameters in the "
                    "forward but trying to all-gather parameters for "
                )
            else:
                expected_handle = self.handles_pre_forward_order[
                    self.current_order_index
                ]
                if expected_handle != handle:
                    expected_param_names = self._get_names_from_handles(expected_handle)
                    msg_prefix = (
                        f"Expected to all-gather for {expected_param_names} "
                        "but trying to all-gather parameters for "
                    )
            if msg_prefix is not None:
                param_names = self._get_names_from_handles(handle)
                msg_suffix = (
                    f"{param_names}"
                    if param_names
                    else "a newly-added parameter since construction time"
                )
                warnings.warn(
                    "Forward order differs from that of the first iteration "
                    f"on rank {self.rank}. Collectives are unchecked and may "
                    f"give incorrect results or hang.\n{msg_prefix}{msg_suffix}",
                    stacklevel=2,
                )
                self.warn_status = _ExecOrderWarnStatus.WARNING
            self.current_order_index += 1

    def _get_handle_indices(
        self,
        handle: FlatParamHandle,
    ) -> tuple[int | None, ...]:
        """
        Returns the handle indices (i.e. indices into ``self.all_handles``)
        corresponding to the handles in ``handle``. An entry in the
        returned tuple is ``None`` if the handle is invalid.
        """
        indices: list[int | None] = []
        if handle:
            indices.append(handle._handle_index)
        return tuple(indices)

    def _get_names_from_handle_indices(
        self,
        handle_indices: tuple[int, ...],
    ) -> list[list[str]]:
        """
        Returns a list of FQNs for each handle in ``handle_indices``. If a
        handle index is invalid, then its FQNs are omitted from the returned
        list.
        """
        fqns: list[list[str]] = []
        for index in handle_indices:
            if index is None or index < 0 or index >= len(self.all_handles):
                continue
            handle = self.all_handles[index]
            flat_param = handle.flat_param
            fqns.append(self.param_to_fqn[flat_param])
        return fqns

    def _get_names_from_handles(
        self,
        handle: FlatParamHandle,
    ) -> list[list[str]]:
        """
        Returns a list of FQNs for each handle in ``handles_key``. If a handle
        is invalid, then its FQNs are omitted from the returned list.
        """
        fqns: list[list[str]] = []
        if handle:
            flat_param = handle.flat_param
            if flat_param in self.param_to_fqn:
                fqns.append(self.param_to_fqn[flat_param])
        return fqns

    def next_iter(self):
        """
        Advances the internal data structures per iteration. This should be
        called in the post-backward callback since that marks the true end of
        an iteration.
        """
        self._iter += 1
        self.handles_post_forward_order.clear()
        if self._checking_order:
            self.current_order_index = 0
            if self.warn_status == _ExecOrderWarnStatus.WARNING:
                self.warn_status = _ExecOrderWarnStatus.WARNED
<|endoftext|>from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed.fsdp._shard_utils import (
    _all_gather_dtensor,
    _create_chunk_dtensor,
    _create_chunk_sharded_tensor,
)
from torch.distributed.tensor import DeviceMesh, DTensor


class FSDPExtensions(ABC):
    """
    This enables some customizable hooks to enable composability with tensor
    parallelism. To activate these hooks, use :func:`_set_fsdp_extensions` to
    set a custom :class:`FSDPExtensions` that implements the hooks.
    """

    @abstractmethod
    def pre_flatten_transform(
        self,
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, Any | None]:
        """E.g. converting ``DistributedTensor`` to local tensor."""
        ...

    @abstractmethod
    def post_unflatten_transform(
        self,
        tensor: torch.Tensor,
        param_extension: Any,
    ) -> torch.Tensor:
        """E.g. converting local tensor to ``DistributedTensor``."""
        ...

    @abstractmethod
    def chunk_tensor(
        self,
        tensor: torch.Tensor,
        rank: int,
        world_size: int,
        num_devices_per_node: int,
        pg: dist.ProcessGroup,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Shards a tensor to chunks and returns the local chunk."""
        ...

    @abstractmethod
    def chunk_dtensor(
        self,
        tensor: torch.Tensor,
        rank: int,
        device_mesh: DeviceMesh,
    ) -> torch.Tensor:
        """Shards a tensor/DTensor to DTensor and returns the local DTensor."""
        ...

    @abstractmethod
    def pre_load_state_dict_transform(
        self,
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, list[Shard]]:
        """
        This is to be called before loading a *sharded* model state dict and
        should return the tensor and list of shards from which to load data.
        """
        ...

    @abstractmethod
    def all_gather_dtensor(
        self,
        tensor: DTensor,
        parent_mesh: DeviceMesh | None,
    ) -> torch.Tensor:
        """
        This is to be called before loading a *sharded* DTensor state dict.
        This gathers tensor in FSDP dimension and returns local tensor of
        TP DTensor.
        """
        ...


_extensions: FSDPExtensions | None = None


def _set_fsdp_extensions(flattener: FSDPExtensions) -> None:
    global _extensions
    _extensions = flattener


def _ext_pre_flatten_transform(
    tensor: torch.Tensor,
    fsdp_extension: FSDPExtensions | None = None,
) -> tuple[torch.Tensor, Any | None]:
    if fsdp_extension is not None:
        new_tensor, param_extension = fsdp_extension.pre_flatten_transform(tensor)
        if param_extension is not None:
            return new_tensor, param_extension
    return tensor, None


def _ext_post_unflatten_transform(
    tensor: torch.Tensor,
    param_extension: Any,
    fsdp_extension: FSDPExtensions | None = None,
) -> torch.Tensor:
    if fsdp_extension is not None and param_extension is not None:
        return fsdp_extension.post_unflatten_transform(tensor, param_extension)
    return tensor


def _ext_chunk_tensor(
    tensor: torch.Tensor,
    rank: int,
    world_size: int,
    num_devices_per_node: int,
    pg: dist.ProcessGroup,
    fsdp_extension: FSDPExtensions | None = None,
) -> torch.Tensor:
    chunk_tensor_fn = (
        fsdp_extension.chunk_tensor
        if fsdp_extension is not None
        else _create_chunk_sharded_tensor
    )
    return chunk_tensor_fn(
        tensor,
        rank,
        world_size,
        num_devices_per_node,
        pg,
    )


def _ext_chunk_dtensor(
    tensor: torch.Tensor,
    rank: int,
    device_mesh: DeviceMesh,
    fsdp_extension: FSDPExtensions | None = None,
) -> torch.Tensor:
    chunk_dtensor_fn = (
        fsdp_extension.chunk_dtensor
        if fsdp_extension is not None
        else _create_chunk_dtensor
    )
    return chunk_dtensor_fn(
        tensor,
        rank,
        device_mesh,
    )


def _ext_pre_load_state_dict_transform(
    tensor: torch.Tensor,
    fsdp_extension: FSDPExtensions | None = None,
) -> tuple[torch.Tensor, list[Shard]]:
    if fsdp_extension is not None:
        return fsdp_extension.pre_load_state_dict_transform(tensor)

    if type(tensor) is not ShardedTensor:
        raise AssertionError(f"Expected ShardedTensor, got {type(tensor)}")
    shards = tensor.local_shards()
    return (tensor, shards)


def _ext_all_gather_dtensor(
    tensor: DTensor,
    parent_mesh: DeviceMesh | None,
    fsdp_extension: FSDPExtensions | None = None,
) -> torch.Tensor:
    all_gather_dtensor_fn = (
        fsdp_extension.all_gather_dtensor
        if fsdp_extension is not None
        else _all_gather_dtensor
    )
    return all_gather_dtensor_fn(tensor, parent_mesh)
<|endoftext|>"""
This file includes public APIs for FSDP such as the classes used for the
constructor arguments.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import auto, Enum

import torch
from torch.nn.modules.batchnorm import _BatchNorm


__all__ = [
    "ShardingStrategy",
    "BackwardPrefetch",
    "MixedPrecision",
    "CPUOffload",
    "StateDictType",
    "StateDictConfig",
    "FullStateDictConfig",
    "LocalStateDictConfig",
    "ShardedStateDictConfig",
    "OptimStateDictConfig",
    "FullOptimStateDictConfig",
    "LocalOptimStateDictConfig",
    "ShardedOptimStateDictConfig",
    "StateDictSettings",
]


class ShardingStrategy(Enum):
    """
    This specifies the sharding strategy to be used for distributed training by
    :class:`FullyShardedDataParallel`.

    - ``FULL_SHARD``: Parameters, gradients, and optimizer states are sharded.
      For the parameters, this strategy unshards (via all-gather) before the
      forward, reshards after the forward, unshards before the backward
      computation, and reshards after the backward computation. For gradients,
      it synchronizes and shards them (via reduce-scatter) after the backward
      computation. The sharded optimizer states are updated locally per rank.
    - ``SHARD_GRAD_OP``: Gradients and optimizer states are sharded during
      computation, and additionally, parameters are sharded outside
      computation. For the parameters, this strategy unshards before the
      forward, does not reshard them after the forward, and only reshards them
      after the backward computation. The sharded optimizer states are updated
      locally per rank. Inside ``no_sync()``, the parameters are not resharded
      after the backward computation.
    - ``NO_SHARD``: Parameters, gradients, and optimizer states are not sharded
      but instead replicated across ranks similar to PyTorch's
      :class:`DistributedDataParallel` API. For gradients, this strategy
      synchronizes them (via all-reduce) after the backward computation. The
      unsharded optimizer states are updated locally per rank.
    - ``HYBRID_SHARD``: Apply ``FULL_SHARD`` within a node, and replicate parameters across
      nodes. This results in reduced communication volume as expensive all-gathers and
      reduce-scatters are only done within a node, which can be more performant for medium
      -sized models.
    - ``_HYBRID_SHARD_ZERO2``: Apply ``SHARD_GRAD_OP`` within a node, and replicate parameters across
      nodes. This is like ``HYBRID_SHARD``, except this may provide even higher throughput
      since the unsharded parameters are not freed after the forward pass, saving the
      all-gathers in the pre-backward.
    """

    FULL_SHARD = auto()
    SHARD_GRAD_OP = auto()
    NO_SHARD = auto()
    HYBRID_SHARD = auto()
    _HYBRID_SHARD_ZERO2 = auto()


class BackwardPrefetch(Enum):
    """
    This configures explicit backward prefetching, which improves throughput by
    enabling communication and computation overlap in the backward pass at the
    cost of slightly increased memory usage.

    - ``BACKWARD_PRE``: This enables the most overlap but increases memory
      usage the most. This prefetches the next set of parameters *before* the
      current set of parameters' gradient computation. This overlaps the *next
      all-gather* and the *current gradient computation*, and at the peak, it
      holds the current set of parameters, next set of parameters, and current
      set of gradients in memory.
    - ``BACKWARD_POST``: This enables less overlap but requires less memory
      usage. This prefetches the next set of parameters *after* the current
      set of parameters' gradient computation. This overlaps the *current
      reduce-scatter* and the *next gradient computation*, and it frees the
      current set of parameters before allocating memory for the next set of
      parameters, only holding the next set of parameters and current set of
      gradients in memory at the peak.
    - FSDP's ``backward_prefetch`` argument accepts ``None``, which disables
      the backward prefetching altogether. This has no overlap and does not
      increase memory usage. In general, we do not recommend this setting since
      it may degrade throughput significantly.

    For more technical context: For a single process group using NCCL backend,
    any collectives, even if issued from different streams, contend for the
    same per-device NCCL stream, which implies that the relative order in which
    the collectives are issued matters for overlapping. The two backward
    prefetching values correspond to different issue orders.
    """

    # NOTE: For both modes, the ordering that defines "current" and "next" is
    # not always exact in the current implementation. A mistargeted prefetch
    # simply means that the parameter memory is allocated earlier than needed,
    # possibly increasing peak memory usage, but does not affect correctness.
    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()


@dataclass
class MixedPrecision:
    """
    This configures FSDP-native mixed precision training.

    Attributes:
        param_dtype (Optional[torch.dtype]): This specifies the dtype for model
            parameters during forward and backward and thus the dtype for
            forward and backward computation. Outside forward and backward, the
            *sharded* parameters are kept in full precision (e.g. for the
            optimizer step), and for model checkpointing, the parameters are
            always saved in full precision. (Default: ``None``)
        reduce_dtype (Optional[torch.dtype]): This specifies the dtype for
            gradient reduction (i.e. reduce-scatter or all-reduce). If this is
            ``None`` but ``param_dtype`` is not ``None``, then this takes on
            the ``param_dtype`` value, still running gradient reduction in low
            precision. This is permitted to differ from ``param_dtype``, e.g.
            to force gradient reduction to run in full precision. (Default:
            ``None``)
        buffer_dtype (Optional[torch.dtype]): This specifies the dtype for
            buffers. FSDP does not shard buffers. Rather, FSDP casts them to
            ``buffer_dtype`` in the first forward pass and keeps them in that
            dtype thereafter. For model checkpointing, the buffers are saved
            in full precision except for ``LOCAL_STATE_DICT``. (Default:
            ``None``)
        keep_low_precision_grads (bool): If ``False``, then FSDP upcasts
            gradients to full precision after the backward pass in preparation
            for the optimizer step. If ``True``, then FSDP keeps the gradients
            in the dtype used for gradient reduction, which can save memory if
            using a custom optimizer that supports running in low precision.
            (Default: ``False``)
        cast_forward_inputs (bool): If ``True``, then this FSDP module casts
            its forward args and kwargs to ``param_dtype``. This is to ensure
            that parameter and input dtypes match for forward computation, as
            required by many ops. This may need to be set to ``True`` when only
            applying mixed precision to some but not all FSDP modules, in which
            case a mixed-precision FSDP submodule needs to recast its inputs.
            (Default: ``False``)
        cast_root_forward_inputs (bool): If ``True``, then the root FSDP module
            casts its forward args and kwargs to ``param_dtype``, overriding
            the value of ``cast_forward_inputs``. For non-root FSDP modules,
            this does not do anything. (Default: ``True``)
        _module_classes_to_ignore: (Sequence[Type[nn.Module]]): This specifies
            module classes to ignore for mixed precision when using an
            ``auto_wrap_policy``: Modules of these classes will have FSDP
            applied to them separately with mixed precision disabled (meaning
            that the final FSDP construction would deviate from the specified
            policy). If ``auto_wrap_policy`` is not specified, then this does
            not do anything. This API is experimental and subject to change.
            (Default: ``(_BatchNorm,)``)

    .. note:: This API is experimental and subject to change.

    .. note:: Only floating point tensors are cast to their specified dtypes.

    .. note:: In ``summon_full_params``, parameters are forced to full
        precision, but buffers are not.

    .. note:: Layer norm and batch norm accumulate in ``float32`` even when
        their inputs are in a low precision like ``float16`` or ``bfloat16``.
        Disabling FSDP's mixed precision for those norm modules only means that
        the affine parameters are kept in ``float32``. However, this incurs
        separate all-gathers and reduce-scatters for those norm modules, which
        may be inefficient, so if the workload permits, the user should prefer
        to still apply mixed precision to those modules.

    .. note:: By default, if the user passes a model with any ``_BatchNorm``
        modules and specifies an ``auto_wrap_policy``, then the batch norm
        modules will have FSDP applied to them separately with mixed precision
        disabled. See the ``_module_classes_to_ignore`` argument.

    .. note:: ``MixedPrecision`` has ``cast_root_forward_inputs=True`` and
        ``cast_forward_inputs=False`` by default. For the root FSDP instance,
        its ``cast_root_forward_inputs`` takes precedence over its
        ``cast_forward_inputs``. For non-root FSDP instances, their
        ``cast_root_forward_inputs`` values are ignored. The default setting is
        sufficient for the typical case where each FSDP instance has the same
        ``MixedPrecision`` configuration and only needs to cast inputs to the
        ``param_dtype`` at the beginning of the model's forward pass.

    .. note:: For nested FSDP instances with different ``MixedPrecision``
        configurations, we recommend setting individual ``cast_forward_inputs``
        values to configure casting inputs or not before each instance's
        forward. In such a case, since the casts happen before each FSDP
        instance's forward, a parent FSDP instance should have its non-FSDP
        submodules run before its FSDP submodules to avoid the activation dtype
        being changed due to a different ``MixedPrecision`` configuration.

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> model = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))
            >>> model[1] = FSDP(
            >>>     model[1],
            >>>     mixed_precision=MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True),
            >>> )
            >>> model = FSDP(
            >>>     model,
            >>>     mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, cast_forward_inputs=True),
            >>> )

        The above shows a working example. On the other hand, if ``model[1]``
        were replaced with ``model[0]``, meaning that the submodule using
        different ``MixedPrecision`` ran its forward first, then ``model[1]``
        would incorrectly see ``float16`` activations instead of ``bfloat16``
        ones.

    """

    param_dtype: torch.dtype | None = None
    reduce_dtype: torch.dtype | None = None
    buffer_dtype: torch.dtype | None = None
    keep_low_precision_grads: bool = False
    cast_forward_inputs: bool = False
    cast_root_forward_inputs: bool = True
    _module_classes_to_ignore: Sequence[type[torch.nn.Module]] = (_BatchNorm,)


@dataclass
class CPUOffload:
    """
    This configures CPU offloading.

    Attributes:
        offload_params (bool): This specifies whether to offload parameters to
            CPU when not involved in computation. If ``True``, then this
            offloads gradients to CPU as well, meaning that the optimizer step
            runs on CPU.
    """

    offload_params: bool = False


class StateDictType(Enum):
    """
    This enum indicates that which type of ``state_dict`` the FSDP module is
    currently processing (returning or loading).
    The default value is FULL_STATE_DICT to comply the PyTorch convention.

    .. note::
        FSDP currently supports three types of ``state_dict``:
            1. ``state_dict/load_state_dict`: this pair of APIs return and load
               the non-sharded, unflattened parameters. The semantics is the
               same as using DDP.
            2. ``_local_state_dict/_load_local_state_dict``: this pair of APIs return
               and load local sharded, flattened parameters. The values returned
               by ``_local_state_dict`` can be directly used by FSDP and is only
               meaningful to FSDP (because parameters are flattened). Note that
               these APIs are meant for use via the :func:`state_dict_type`
               context manager as follows:
                   >>> # xdoctest: +SKIP("undefined variables")
                   >>> with fsdp.state_dict_type(StateDictType.LOCAL_STATE_DICT):
                   ...     state = fsdp.state_dict()  # loads local state dict
            3. ``_sharded_state_dict/_load_sharded_state_dict``: this pair of APIs
               return and load sharded, unflattened parameters. The ``state_dict``
               return by ``sharded_state_dict`` can be used by all other parallel
               schemes (resharding may be required).
    """

    FULL_STATE_DICT = auto()
    LOCAL_STATE_DICT = auto()
    SHARDED_STATE_DICT = auto()


@dataclass
class StateDictConfig:
    """
    ``StateDictConfig`` is the base class for all ``state_dict`` configuration
    classes. Users should instantiate a child class (e.g.
    ``FullStateDictConfig``) in order to configure settings for the
    corresponding ``state_dict`` type supported by FSDP.

    Attributes:
        offload_to_cpu (bool): If ``True``, then FSDP offloads the state dict
            values to CPU, and if ``False``, then FSDP keeps them on GPU.
            (Default: ``False``)
    """

    offload_to_cpu: bool = False


@dataclass
class FullStateDictConfig(StateDictConfig):
    """
    ``FullStateDictConfig`` is a config class meant to be used with
    ``StateDictType.FULL_STATE_DICT``. We recommend enabling both
    ``offload_to_cpu=True`` and ``rank0_only=True`` when saving full state
    dicts to save GPU memory and CPU memory, respectively. This config class
    is meant to be used via the :func:`state_dict_type` context manager as
    follows:

        >>> # xdoctest: +SKIP("undefined variables")
        >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        >>> fsdp = FSDP(model, auto_wrap_policy=...)
        >>> cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        >>> with FSDP.state_dict_type(fsdp, StateDictType.FULL_STATE_DICT, cfg):
        >>>     state = fsdp.state_dict()
        >>> # `state` will be empty on non rank 0 and contain CPU tensors on rank 0.
        >>> # To reload checkpoint for inference, finetuning, transfer learning, etc:
        >>> model = model_fn()  # Initialize model in preparation for wrapping with FSDP
        >>> if dist.get_rank() == 0:
        >>> # Load checkpoint only on rank 0 to avoid memory redundancy
        >>>     state_dict = torch.load("my_checkpoint.pt")
        >>>     model.load_state_dict(state_dict)
        >>> # All ranks initialize FSDP module as usual. `sync_module_states` argument
        >>> # communicates loaded checkpoint states from rank 0 to rest of the world.
        >>> fsdp = FSDP(
        ...     model,
        ...     device_id=torch.cuda.current_device(),
        ...     auto_wrap_policy=...,
        ...     sync_module_states=True,
        ... )
        >>> # After this point, all ranks have FSDP model with loaded checkpoint.

    Attributes:
        rank0_only (bool): If ``True``, then only rank 0 saves the full state
            dict, and nonzero ranks save an empty dict. If ``False``, then all
            ranks save the full state dict. (Default: ``False``)
    """

    rank0_only: bool = False


@dataclass
class LocalStateDictConfig(StateDictConfig):
    pass


@dataclass
class ShardedStateDictConfig(StateDictConfig):
    """
    ``ShardedStateDictConfig`` is a config class meant to be used with
    ``StateDictType.SHARDED_STATE_DICT``.

    Attributes:
        _use_dtensor (bool): If ``True``, then FSDP saves the state dict values
            as ``DTensor``, and if ``False``, then FSDP saves them as
            ``ShardedTensor``. (Default: ``False``)

    .. warning:: ``_use_dtensor`` is a private field of :class:`ShardedStateDictConfig`
      and it is used by FSDP to determine the type of state dict values. Users should not
      manually modify ``_use_dtensor``.
    """

    _use_dtensor: bool = False


@dataclass
class OptimStateDictConfig:
    """
    ``OptimStateDictConfig`` is the base class for all ``optim_state_dict``
    configuration classes.  Users should instantiate a child class (e.g.
    ``FullOptimStateDictConfig``) in order to configure settings for the
    corresponding ``optim_state_dict`` type supported by FSDP.

    Attributes:
        offload_to_cpu (bool): If ``True``, then FSDP offloads the state dict's
            tensor values to CPU, and if ``False``, then FSDP keeps them on the
            original device (which is GPU unless parameter CPU offloading is
            enabled). (Default: ``True``)
    """

    offload_to_cpu: bool = True


@dataclass
class FullOptimStateDictConfig(OptimStateDictConfig):
    """
    Attributes:
        rank0_only (bool): If ``True``, then only rank 0 saves the full state
            dict, and nonzero ranks save an empty dict. If ``False``, then all
            ranks save the full state dict. (Default: ``False``)
    """

    rank0_only: bool = False


@dataclass
class LocalOptimStateDictConfig(OptimStateDictConfig):
    offload_to_cpu: bool = False


@dataclass
class ShardedOptimStateDictConfig(OptimStateDictConfig):
    """
    ``ShardedOptimStateDictConfig`` is a config class meant to be used with
    ``StateDictType.SHARDED_STATE_DICT``.

    Attributes:
        _use_dtensor (bool): If ``True``, then FSDP saves the state dict values
            as ``DTensor``, and if ``False``, then FSDP saves them as
            ``ShardedTensor``. (Default: ``False``)

    .. warning:: ``_use_dtensor`` is a private field of :class:`ShardedOptimStateDictConfig`
      and it is used by FSDP to determine the type of state dict values. Users should not
      manually modify ``_use_dtensor``.
    """

    _use_dtensor: bool = False


@dataclass
class StateDictSettings:
    state_dict_type: StateDictType
    state_dict_config: StateDictConfig
    optim_state_dict_config: OptimStateDictConfig
<|endoftext|># mypy: allow-untyped-defs
import contextlib
import functools
import logging
import os
import warnings
from collections.abc import Callable, Generator, Iterator, Sequence
from enum import auto, Enum
from itertools import accumulate, chain
from typing import Any, cast, NamedTuple, no_type_check

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.fsdp._common_utils import (
    _FSDPDeviceHandle,
    _named_parameters_with_duplicates,
    _no_dispatch_record_stream,
    _set_fsdp_flattened,
    HandleTrainingState,
)
from torch.distributed.utils import (
    _alloc_storage,
    _data_ptr_allocated,
    _free_storage,
    _p_assert,
)
from torch.nn.parameter import _ParameterMeta  # type: ignore[attr-defined]
from torch.testing._internal.distributed.fake_pg import FakeProcessGroup

from ._fsdp_extensions import (
    _ext_post_unflatten_transform,
    _ext_pre_flatten_transform,
    FSDPExtensions,
)


__all__ = [
    "FlatParameter",
    "FlatParamHandle",
    "FlatParamShardMetadata",
    "ParamInfo",
    "SharedParamInfo",
    "HandleShardingStrategy",
]

logger = logging.getLogger(__name__)


"""
[Note: Fully Sharded Module]
We define the "fully sharded module" to be the original ``nn.Module`` that owns
a ``FlatParamHandle``. It is the *single* module logically responsible for the
*single* unshard/reshard pair for the handle's ``FlatParameter`` for a given
forward or backward pass. The fully sharded module should be passed to the
``FlatParamHandle`` constructor.

For the wrapper code path:
- The ``FullyShardedDataParallel`` module wrapping the fully sharded module
runs the unshard/reshard on behalf of the fully sharded module by overriding
``nn.Module.forward``.
- The fully sharded module is exactly the module passed to the
``FullyShardedDataParallel`` constructor's ``module`` argument.

For the non-wrapper code path:
- Hooks registered on the fully sharded module run the unshard/reshard.
- The fully sharded module may either be the direct argument to ``fully_shard``
or a submodule chosen by the provided wrapping policy.
"""

# Environment variable toggling whether to use unsafe `setattr()` for view
# setting in `_use_sharded_views()` and `_use_unsharded_views()`
# We should use 'safe' by default since it respects method overrides, but for
# special cases such as for high CPU overhead or for intentionally bypassing
# checks in the overrides, we may use 'unsafe'.
_FSDP_USE_UNSAFE_SETATTR = "FSDP_USE_UNSAFE_SETATTR"

# Environment variable toggling whether to check for parameter/gradient
# writeback in case their storages change after FSDP initialization
# We should check by default since it prevents silent correctness errors, but
# since such changes are atypical, we may want to skip the check to save CPU
# overhead, especially since the check happens in the pre-forward and
# pre-backward each iteration.
_FSDP_SKIP_WRITEBACK_CHECK = "FSDP_SKIP_WRITEBACK_CHECK"

# Env var toggling whether when model is in .eval() mode, should we run in fp32
# or the reduced precision.
_FSDP_USE_FULL_PREC_IN_EVAL = "FSDP_USE_FULL_PREC_IN_EVAL"

# Some value to set padding in tensors to for debuggability
_FLAT_PARAM_PADDING_VALUE = 42

# Environment variables for disabling the all-gather and reduce-scatter
# communication ops for ablation studies. Note that without these communication
# ops the training won't converge, and you probably need to disable correctness
# checks in your model.
_FSDP_USE_FAKE_ALL_GATHER = "FSDP_USE_FAKE_ALL_GATHER"
_FSDP_USE_FAKE_REDUCE = "FSDP_USE_FAKE_REDUCE"


# TODO: Define this for now to avoid circular imports. See if we can remove.
class HandleShardingStrategy(Enum):
    FULL_SHARD = auto()
    SHARD_GRAD_OP = auto()
    NO_SHARD = auto()
    HYBRID_SHARD = auto()
    _HYBRID_SHARD_ZERO2 = auto()


RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES = (
    HandleShardingStrategy.FULL_SHARD,
    HandleShardingStrategy.HYBRID_SHARD,
)
NO_RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES = (
    HandleShardingStrategy.SHARD_GRAD_OP,
    HandleShardingStrategy._HYBRID_SHARD_ZERO2,
)


class ParamInfo(NamedTuple):
    """Information for an original parameter."""

    param_name: str  # unprefixed
    module: nn.Module
    module_name: str


class SharedParamInfo(NamedTuple):
    """
    Additional information for a shared parameter.

    For each shared parameter, we designate one module and its parameter
    variable to be the primary owner, determined as the first one encountered
    in the parameter walk. These are prefixed with "prim". The primary module
    and parameter do not have their own :class:`SharedParamInfo` instance.
    """

    param_name: str  # unprefixed
    module: nn.Module
    module_name: str
    prim_param_name: str  # unprefixed
    prim_module: nn.Module
    prim_module_name: str


class _ShardParamInfo(NamedTuple):
    """Shard-related information for an original parameter."""

    in_shard: bool
    # Use to index into the sharded flat parameter, e.g.
    # `flat_param[offset_in_shard : offset_in_shard + numel_in_shard]`
    offset_in_shard: int | None
    numel_in_shard: int | None
    # Use to get part of the parameter in the local shard from a flattened
    # version of the unsharded parameter, e.g. either
    # `param.flatten()[intra_param_start_idx : intra_param_end_idx + 1]` or
    # `param.as_strided((param.numel(),), (1,))[intra_param_start_idx : intra_param_end_idx + 1]`
    intra_param_start_idx: int | None
    intra_param_end_idx: int | None  # inclusive


class FlatParamShardMetadata(NamedTuple):
    """
    This holds metadata specific to this rank's shard of the flat parameter.

    Attributes:
        param_names (Tuple[str, ...]): Prefixed parameter names of this rank's
            shard of the parameters; see :class:`FlatParameter`.
        param_shapes (Tuple[torch.Size, ...]): Parameter shapes of this rank's
            shard of the parameters; see :class:`FlatParameter`.
        param_strides (Tuple[torch.Size, ...]): Parameter strides of this rank's
            shard of the parameters; see :class:`FlatParameter`.
        param_contiguities (Tuple[bool, ...]): Parameter `.contiguous` call results
            of this rank's shard of the parameters; see :class:`FlatParameter`.
        param_numels (Tuple[int, ...]): Parameter numels of this rank's shard
            of the parameters; see :class:`FlatParameter`.
        param_offsets (Tuple[Tuple[int, int], ...]): [start, end] offsets (in
            units of numels) giving this rank's part of each flattened
            original parameter.
    """

    param_names: tuple[str, ...]
    param_shapes: tuple[torch.Size, ...]
    param_strides: tuple[tuple[int, ...], ...]
    param_contiguities: tuple[bool, ...]
    param_numels: tuple[int, ...]
    param_offsets: tuple[tuple[int, int], ...]


class _FlatParameterMeta(_ParameterMeta):
    # Make `isinstance(t, FlatParameter)` return True for custom tensor
    # instances that have the _is_flat_param flag for BC
    def __instancecheck__(self, instance):
        # NB: do NOT test the super implementation
        return isinstance(instance, torch.Tensor) and getattr(
            instance, "_is_flat_param", False
        )


class FlatParameter(nn.Parameter, metaclass=_FlatParameterMeta):
    """
    This is the flat parameter used by :class:`FullyShardedDataParallel`.

    It is comprised of one or more original parameters, which are flattened and
    concatenated to construct the flat parameter.

    Under the current design, this parameter logically represents both the
    unsharded and sharded flat parameter, and its data changes storages
    dynamically.
        - In the :class:`FullyShardedDataParallel` constructor, the parameter
        is initialized as unsharded and then sharded in-place.
        - At runtime, the parameter is lazily (re)-initialized. The sharded
        parameter data is saved in ``self._local_shard``, and a new ``Tensor``
        ``self._full_param_padded`` is created, which is the all-gather
        destination and owns the unsharded parameter storage thereafter. (See
        :meth:`FlatParamHandle.init_flat_param_attributes`.)
        - Throughout runtime, the parameter data changes storages as needed,
        e.g. to the sharded flat parameter, low precision sharded flat
        parameter, or the unsharded flat parameter.

    NOTE: Since ``use_orig_params=True`` supports intra-``FlatParameter``
    padding, we have two versions of the per-parameter numels, one that
    includes the padding (``_numels_with_padding``) and one that does not
    (``_numels``). The former may have length longer than the other data
    structures, while the latter has the same length as the number of actual
    original parameters like the other per-parameter data structures.

    NOTE: This is not a real class; instead, you will always get a Parameter
    back out if you try to create one of these.  This is similar to the trick
    we implemented for Parameter to get it to work with subclasses; this
    is primarily so that FlatParameter supports combination with FakeTensor.

    Attributes:
        _unpadded_unsharded_size (torch.Size): Unsharded flat parameter's size
            without right-hand-side padding for divisibility by the world size.
            For ``use_orig_params=True``, this includes alignment padding.
        _padded_unsharded_size (torch.Size): Unsharded flat parameter's size
            with right-hand-side padding for divisibility by the world size.
            For ``use_orig_params=True``, this includes alignment padding. This
            is only set for sharded strategies since they require padding for
            the all-gather.
        _sharded_size (torch.Size): Sharded flat parameter's size with padding.
            This is also set for ``NO_SHARD``, in which case it is the same as
            the unsharded sizes. (We omit "padded" because there is no
            analogous unpadded one.)

        _num_params (int): Number of original parameters flattened into this
            flat parameter. This is the length of the per-parameter data
            structures.
        _param_infos (Tuple[ParamInfo, ...]): Each parameter's parameter info
            entry; see :class:`ParamInfo` for details.
        _shapes (Tuple[torch.Size, ...]): Each parameter's original shape.
        _strides (Tuple[torch.Size, ...]): Each parameter's original stride.
        _contiguities (Tuple[bool, ...]): Each parameter's ``contiguous()``
            call result.
        _fqns (Tuple[str, ...]): Each parameter's fully-qualified name (FQN)
            prefixed from the ``_fully_sharded_module``. The names are
            guaranteed to be unique in the subtree rooted at that module.
        _param_extensions (Tuple[Optional[Any], ...]): Each parameter's
            extension (i.e. some per-parameter state) used to customize
            pre-flatten and post-unflatten behavior or ``None``. This is
            experimental, and users should not depend on its existence in the
            future.
        _numels_with_padding (Tuple[int, ...]): Each parameter's numel
            including entries for the padding. This is used to construct views
            into the flat parameter via ``torch.split()``. This may have length
            longer than ``_num_params``.
        _numels (Tuple[int, ...]): Each parameter's numel excluding entries for
            padding. This has length equal to ``_num_params``.
        _shard_param_infos (Tuple[_ShardParamInfo, ...]): Each parameter's
            shard parameter info; see :class:`_ShardParamInfo` for details.
        _shared_param_infos (Tuple[SharedParamInfo, ...]): Shared parameter
            info entries; see :class:`SharedParamInfo` for details.
        _modules (set[nn.Module]): Modules that contain some original parameter
            that is flattened into the flat parameter.

        _shard_numel_padded (int): Numel padded for this rank's sharded flat
            parameter.
        _local_shard (Tensor): Sharded flat parameter with padding if using a
            sharded strategy. If using ``NO_SHARD``, then this is the unpadded
            unsharded flat parameter, and there is no notion of a sharded flat
            parameter or padded unsharded flat parameter.
        _full_param_padded (Tensor): Unsharded flat parameter with padding.
            This is not defined for ``NO_SHARD``. When using mixed precision
            for parameters, this has the low precision.
        _full_prec_full_param_padded (Tensor): Full precision unsharded flat
            parameter with padding. This is used for unsharding outside of
            computation when using mixed precision for parameters. This is
            never defined for ``NO_SHARD``.
        _post_backward_hook_handle (RemovableHandle):
            Flat parameter's post-backward hook handle. (Compile only)
        _post_backward_hook_state (Tuple[AccumulateGrad, RemovableHandle]):
            Flat parameter's :class:`AccumulateGrad` object and post-backward
            hook handle. (Eager only)
        _mp_shard (Tensor): Low precision sharded flat parameter with padding.
            This is only defined when parameter mixed precision is enabled. For
            ``NO_SHARD``, this is used for computation.
        _cpu_grad (Tensor): Sharded gradient with padding stored on CPU.
            This is only defined when offloading parameters is enabled.
        _saved_grad_shard (Tensor): Sharded gradient with padding from previous
            iterations for gradient accumulation without :meth:`no_sync`.

        _params (Optional[List[nn.Parameter]]): If ``use_orig_params=True``,
            then each original parameter variable; otherwise, ``None``. This
            does not include any padding tensors.
        _shared_params (Optional[List[nn.Parameter]]): The original shared
            parameter variables if ``use_orig_params=True`` and ``None``
            otherwise.
        _tensors (Optional[List[Optional[Tensor]]]): This saves the ``Tensor``
            views created in the forward and tracked by autograd when
            ``use_orig_params=True`` and is ``None`` otherwise. This is to
            preserve those ``Tensor`` variables for the backward to ensure that
            the ``FlatParameter`` 's ``AccumulateGrad`` object does not change
            in which case the post-backward hook does not run. This is relevant
            for cases like reentrant activation checkpointing.
        _is_grad_none_mask (Optional[List[bool]]): If ``use_orig_params=True``,
            a mask over the original parameters' gradients indicating if it is
            logically ``None`` or not; otherwise, ``None``. This does not
            include entries for padding. This mask is needed because only some
            of the parameters may have ``None`` gradient, in which case the
            flat gradient must be non-``None`` and must use zeros to
            approximate those original ``None`` gradients. This mask informs
            FSDP to set the original parameter gradients to ``None`` (instead
            of zeros) as needed.
    """

    _unpadded_unsharded_size: torch.Size
    _padded_unsharded_size: torch.Size
    _sharded_size: torch.Size
    _num_params: int
    _param_infos: tuple[ParamInfo, ...]
    _shapes: tuple[torch.Size, ...]
    _strides: tuple[tuple[int, ...], ...]
    _contiguities: tuple[bool, ...]
    _fqns: tuple[str, ...]
    _param_extensions: tuple[Any | None, ...]
    _numels_with_padding: tuple[int, ...]
    _numels: tuple[int, ...]
    _shard_param_infos: tuple[_ShardParamInfo, ...]
    _shared_param_infos: tuple[SharedParamInfo, ...]
    _modules: set[nn.Module]
    _shard_numel_padded: int
    _local_shard: Tensor
    _full_param_padded: Tensor
    _full_prec_full_param_padded: Tensor
    # Eager only
    _post_backward_hook_state: tuple[Any, Any]
    # Compile only
    _post_backward_hook_handle: Any
    _mp_shard: Tensor
    _cpu_grad: Tensor
    _saved_grad_shard: Tensor
    _params: list[nn.Parameter] | None
    _shared_params: list[nn.Parameter] | None
    _tensors: list[Tensor | None] | None
    _is_grad_none_mask: list[bool] | None

    _is_padding_mask: list[bool]

    def __new__(cls, data=None, requires_grad=True):
        if cls is not FlatParameter:
            raise AssertionError("subclasses FlatParameter not supported")
        r = nn.Parameter.__new__(nn.Parameter, data, requires_grad)  # type: ignore[call-arg]
        r._is_flat_param = True  # type: ignore[attr-defined]
        return r

    # NB: This is not a regular method, because FlatParameters are not actually
    # instances of this class (see __new__ above).  So you must indirectly
    # call this directly through the classmethod.
    @classmethod
    def _init_metadata(
        cls,
        self,
        param_infos: list[ParamInfo],
        numels: list[int],
        shapes: list[torch.Size],
        strides: list[tuple[int, ...]],
        contiguities: list[bool],
        fqns: list[str],
        shared_param_infos: list[SharedParamInfo],
        param_extensions: list[Any | None],
        params: list[nn.Parameter] | None,
        shared_params: list[nn.Parameter] | None,
        is_padding_mask: list[bool],
    ) -> None:
        """
        Initialize attributes holding metadata about the original parameters comprising the flat parameter.

        We expose this method separate from the constructor to keep the
        constructor only responsible for the flat parameter's tensor data. This
        method should only be called once per model, while the constructor may
        be called multiple times, e.g. when reloading from a checkpoint, in
        which case only the tensor data needs to be passed to the constructor.
        Since :meth:`load_state_dict` is implemented via :meth:`copy_`, the
        metadata is correctly assumed to be unchanged.

        Args:
            See the Attributes in the class docstring.
        """
        if len(param_infos) != len(shapes):
            raise AssertionError(
                f"Expected param_infos length {len(param_infos)} to match shapes length {len(shapes)}"
            )
        if len(param_infos) != len(strides):
            raise AssertionError(
                f"Expected param_infos length {len(param_infos)} to match strides length {len(strides)}"
            )
        if len(param_infos) != len(contiguities):
            raise AssertionError(
                f"Expected param_infos length {len(param_infos)} to match contiguities length {len(contiguities)}"
            )
        if len(param_infos) != len(fqns):
            raise AssertionError(
                f"Expected param_infos length {len(param_infos)} to match fqns length {len(fqns)}"
            )
        if len(param_infos) != len(param_extensions):
            raise AssertionError(
                f"Expected param_infos length {len(param_infos)} to match param_extensions length {len(param_extensions)}"
            )
        self._num_params = len(param_infos)
        self._param_infos = param_infos
        self._shapes = shapes
        self._strides = strides
        self._contiguities = contiguities
        self._fqns = fqns
        self._param_extensions = param_extensions
        self._is_padding_mask = is_padding_mask

        numels_without_padding: list[int] = []
        for numel, is_padding in zip(numels, is_padding_mask):
            if not is_padding:
                numels_without_padding.append(numel)
        self._numels = tuple(numels_without_padding)
        self._numels_with_padding = tuple(numels)
        if len(self._numels) != self._num_params:
            raise AssertionError(
                f"Expected _numels length {len(self._numels)} to equal _num_params {self._num_params}"
            )

        self._shared_param_infos = tuple(shared_param_infos)
        self._modules = {pi.module for pi in self._param_infos}.union(
            {spi.module for spi in self._shared_param_infos}
        )
        if (params is None) != (shared_params is None):
            raise AssertionError(
                "Expected params and shared_params to both be None or both be not None"
            )
        if params is not None:
            if shared_params is None or len(shared_params) != len(shared_param_infos):
                raise AssertionError(
                    f"Expected shared_params to be not None and have length {len(shared_param_infos)}, got {shared_params}"
                )
            self._params = []
            for param, is_padding in zip(params, is_padding_mask):
                if not is_padding:
                    self._params.append(param)
            if shared_params is not None:
                self._shared_params = shared_params
            else:
                self._shared_params = []
            # Mark the original parameters to avoid flattening them into
            # another `FlatParameter` during recursive construction
            for param in chain(self._params, self._shared_params):
                _set_fsdp_flattened(param)
            self._is_grad_none_mask = [False for _ in range(self._num_params)]
            self._tensors = [None for _ in range(self._num_params)]
        else:
            self._params = None
            self._shared_params = None
            self._is_grad_none_mask = None
            self._tensors = None
        self._unpadded_unsharded_size = self.size()
        _set_fsdp_flattened(self)
        # Tracks whether the `FlatParameter`'s post-backward hook has been
        # called to modify the behavior of the post-backward callback
        self._post_backward_called = False


class FlatParamHandle:
    """
    A handle that manages a flat parameter (:class:`FlatParameter`).

    This includes sharding and view management.

    Args:
        params (Sequence[nn.Parameter]): The parameters to flatten into the
            flat parameter.
        fully_sharded_module (nn.Module): See [Note: Fully Sharded Module].
        device (torch.device): The compute and communication device, which
            should be a non-CPU device. We refer to it as the compute device.
        sharding_strategy (ShardingStrategy): Sharding strategy to apply to
            this handle's ``FlatParameter``.
        offload_params (bool): Whether to offload the handle's
            ``FlatParameter`` to CPU.
        mp_param_dtype (Optional[torch.dtype]): Parameter mixed precision
            setting passed to the FSDP constructor.
        mp_reduce_dtype (Optional[torch.dtype]): Gradient reduction mixed
            precision setting passed to the FSDP constructor.
        keep_low_precision_grads (bool): Whether to keep gradients in low
            precision.
        use_orig_params (bool): If ``True``, then FSDP preserves the original
            parameter variables and returns them from ``named_parameters()``
            (e.g. to support different optimizer hyperparameters within one
            :class:`FlatParameter`). If ``False``, then FSDP reconstructs the
            parameters every iteration and returns the :class:`FlatParameter` s
            from ``named_parameters()``.
    """

    ##################
    # INITIALIZATION #
    ##################
    def __init__(
        self,
        params: Sequence[nn.Parameter | Tensor],
        fully_sharded_module: nn.Module,
        device: torch.device,
        sharding_strategy: HandleShardingStrategy,
        offload_params: bool,
        mp_param_dtype: torch.dtype | None,
        mp_reduce_dtype: torch.dtype | None,
        keep_low_precision_grads: bool,
        process_group: dist.ProcessGroup,
        use_orig_params: bool,
        *,
        fsdp_extension: FSDPExtensions | None = None,
    ):
        super().__init__()
        params = list(params)
        if len(params) == 0:
            raise ValueError(
                f"Cannot construct a {self.__class__.__name__} with an empty parameter list"
            )
        self._init_setattr_fns()
        self._skip_writeback_check = (
            os.environ.get(_FSDP_SKIP_WRITEBACK_CHECK, "") == "1"
        )
        self._use_full_prec_in_eval = (
            os.environ.get(_FSDP_USE_FULL_PREC_IN_EVAL, "") == "1"
        )
        self._use_fake_all_gather = os.environ.get(_FSDP_USE_FAKE_ALL_GATHER, "") == "1"
        self._use_fake_reduce = os.environ.get(_FSDP_USE_FAKE_REDUCE, "") == "1"
        if self._skip_writeback_check:
            _warn_skip_writeback_check(
                logger,
                f"Since {_FSDP_SKIP_WRITEBACK_CHECK}=1, FSDP will not check "
                "for parameter or gradient writeback. Changing parameter or "
                "gradient storages may lead to silent correctness errors.",
            )
        if self._use_fake_all_gather:
            _warn_use_fake_all_gather(
                logger,
                f"Since {_FSDP_USE_FAKE_ALL_GATHER}=1, FSDP will not execute "
                "all-gather ops. Your training will be incorrect, but "
                "can reveal how much time spent on all-gather ops.",
            )
        if self._use_fake_reduce:
            _warn_use_fake_reduce(
                logger,
                f"Since {_FSDP_USE_FAKE_REDUCE}=1, FSDP will not execute "
                "reduce-scatter ops. Your training will be incorrect, but "
                "can reveal how much time spent on reduce-scatter ops.",
            )
        # Only align addresses for `use_orig_params=True` (for now)
        align_addresses = use_orig_params
        self._init_get_unflat_views_fn(align_addresses)
        self.device = device
        self._device_handle = _FSDPDeviceHandle.from_device(self.device)
        self.process_group = process_group
        if self._use_fake_all_gather or self._use_fake_reduce:
            self._fake_process_group = FakeProcessGroup._create_internal(
                rank=process_group.rank(), world_size=process_group.size()
            )
        self.rank = process_group.rank()
        self.world_size = process_group.size()
        self._sharding_strategy = sharding_strategy
        self._offload_params = offload_params
        self._use_orig_params = use_orig_params
        self._keep_low_precision_grads = keep_low_precision_grads
        self._training_state = HandleTrainingState.IDLE
        self._debug_level = dist.get_debug_level()
        self._fully_sharded_module = fully_sharded_module
        # For strategies that do not free after forward, we skip using sharded
        # views after forward since the unsharded data exists. We still switch
        # `self.flat_param` to point to the sharded flat parameter since what
        # it points to parameterizes behavior. We use the following attribute
        # to track which tensor data the parameters are unsharded views into.
        self._unsharded_flat_param_for_skipped_views: Tensor | None = None
        # The index in the state's `all_handles`, which must be the
        # same across ranks for the execution order validation to work
        self._handle_index: int | None = None
        # Index in handles_to_pre_forward_order
        self._pre_forward_order_index: int | None = None
        # Index in `handles_post_forward_order`
        self._post_forward_index: int | None = None
        # Used for guarding against mistargeted forward prefetches
        self._needs_pre_forward_unshard = False
        # Used for guarding against mistargeted backward prefetches
        self._needs_pre_backward_unshard = False
        # Was the handle prefetched? Set on successful _prefetch_handle and unshard
        self._prefetched = False
        # Optimistically assume a valid input `params` and set dtype attributes
        # before `_init_flat_param()`, which performs the actual validation
        self._orig_param_dtype = params[0].dtype
        self._init_param_reduce_dtypes(mp_param_dtype, mp_reduce_dtype)
        if self._fwd_bwd_param_dtype is None:
            raise AssertionError("Expected _fwd_bwd_param_dtype to be not None")  # mypy
        self._aligned_numel = (
            _get_aligned_numel(unsharded_dtype=self._fwd_bwd_param_dtype)
            if align_addresses
            else 0
        )
        self._fsdp_extension = fsdp_extension
        self._init_flat_param_and_metadata(
            params,
            fully_sharded_module,
            self._aligned_numel,
            use_orig_params,  # type: ignore[arg-type]
        )
        self._use_unsharded_views(as_params=False)

    def __repr__(self):
        return f"FlatParamHandle(flat_param.fqns={self.flat_param._fqns})"

    def _init_setattr_fns(self):
        use_unsafe_setattr = os.environ.get(_FSDP_USE_UNSAFE_SETATTR, "") == "1"
        self._setattr_tensor: Callable[[nn.Module, str, Tensor], None]
        self._setattr_param: Callable[[nn.Module, str, nn.Parameter], None]
        if use_unsafe_setattr:
            self._setattr_tensor = _unsafe_setattr_tensor
            self._setattr_param = _unsafe_setattr_param
        else:
            self._setattr_tensor = _safe_setattr_tensor_or_param
            self._setattr_param = _safe_setattr_tensor_or_param

    def _init_get_unflat_views_fn(self, align_addresses: bool):
        self._get_unflat_views = (
            self._get_unflat_views_aligned
            if align_addresses
            else self._get_unflat_views_unaligned
        )

    def _init_flat_param_and_metadata(
        self,
        params: list[Tensor | nn.Parameter],
        module: nn.Module,
        aligned_numel: int,
        use_orig_params: bool,
    ) -> None:
        """
        Initialize the ``FlatParameter`` and its metadata.

        NOTE: This should only be called once at construction time, after which
        the ``FlatParameter`` metadata is assumed to be static.

        NOTE: The elements of ``params`` should only be ``Tensor`` s when
        composing with ``DTensor`` -based tensor parallelism, in which case the
        elements may be ``DTensor`` local shards.
        """
        if len(params) == 0:
            raise ValueError("Expects non-empty `params`")
        if aligned_numel < 0:
            raise ValueError(
                f"Expects non-negative `aligned_numel` but got {aligned_numel}"
            )
        (
            dtype,
            flat_param_requires_grad,
            device,
        ) = self._validate_tensors_to_flatten(params)
        params_set = set(params)
        # For alignment padding, only `numels` gets strictly non-`None`
        # elements, and all other lists get `None` elements for padding.
        param_infos: list[ParamInfo] = []
        numels: list[int] = []
        shapes: list[torch.Size] = []
        strides: list[tuple[int, ...]] = []
        contiguities: list[bool] = []
        fqns: list[str] = []
        shared_param_infos: list[SharedParamInfo] = []
        shared_param_memo: dict[Tensor | nn.Parameter, tuple[nn.Module, str, str]] = {}
        params_to_flatten: list[Tensor | nn.Parameter] = []
        shared_params: list[Tensor | nn.Parameter] = []
        param_extensions: list[Any] = []
        is_padding_mask: list[bool] = []
        total_numel = total_numel_without_padding = 0
        for submodule_name, submodule in module.named_modules(remove_duplicate=False):
            for param_name, param in _named_parameters_with_duplicates(
                submodule, recurse=False
            ):
                if param not in params_set:
                    continue
                if param in shared_param_memo:  # shared reference
                    prim_module, prim_module_name, prim_param_name = shared_param_memo[
                        param
                    ]
                    shared_params.append(param)
                    shared_param_infos.append(
                        SharedParamInfo(
                            param_name,
                            submodule,
                            submodule_name,
                            prim_param_name,
                            prim_module,
                            prim_module_name,
                        )
                    )
                else:
                    if aligned_numel > 0:
                        numel_to_pad = aligned_numel - (total_numel % aligned_numel)
                        if numel_to_pad > 0 and numel_to_pad < aligned_numel:
                            padding_tensor = _construct_padding_tensor(
                                numel_to_pad, dtype, False, device
                            )
                            params_to_flatten.append(padding_tensor)
                            is_padding_mask.append(True)
                            numels.append(numel_to_pad)
                            total_numel += numel_to_pad
                    transform_t, extension = _ext_pre_flatten_transform(
                        param,
                        self._fsdp_extension,
                    )
                    param = cast(nn.Parameter, transform_t)
                    param_extensions.append(extension)
                    shared_param_memo[param] = (submodule, submodule_name, param_name)
                    params_to_flatten.append(param)
                    is_padding_mask.append(False)
                    param_infos.append(ParamInfo(param_name, submodule, submodule_name))
                    numels.append(param.numel())
                    shapes.append(param.shape)
                    strides.append(param.stride())
                    contiguities.append(_is_truly_contiguous(param))
                    fqn = (
                        submodule_name + "." + param_name
                        if submodule_name
                        else param_name
                    )
                    fqns.append(fqn)
                    total_numel += param.numel()
                    total_numel_without_padding += param.numel()
        if len(params_to_flatten) == 0:
            raise ValueError(
                f"`params` were not found in `module`'s tree"
                f"params: {params}\nmodule: {module}"
            )
        if (
            self.rank == 0
            and aligned_numel > 0
            and total_numel != total_numel_without_padding
        ):
            logger.debug(
                "FSDP FlatParameter address alignment created "
                "%s numel of padding (%s vs. %s)",
                total_numel - total_numel_without_padding,
                total_numel,
                total_numel_without_padding,
            )
        if aligned_numel > 0:
            # Pad to be divisible by world size to avoid a copy for the
            # post-backward reduce-scatter
            numel_to_pad = self.world_size - (total_numel % self.world_size)
            if numel_to_pad > 0 and numel_to_pad < self.world_size:
                if self.rank == 0:
                    logger.info(
                        "FSDP FlatParameter world size divisibility created "
                        "%s numel of padding",
                        numel_to_pad,
                    )
                padding_tensor = _construct_padding_tensor(
                    numel_to_pad, dtype, False, device
                )
                params_to_flatten.append(padding_tensor)
                is_padding_mask.append(True)
                numels.append(numel_to_pad)
                total_numel += numel_to_pad
        # Pass `aligned_numel=0` since we already included padding tensors
        self.flat_param: FlatParameter = self.flatten_tensors_into_flat_param(
            params_to_flatten,
            aligned_numel=0,
            requires_grad=flat_param_requires_grad,
        )
        FlatParameter._init_metadata(
            self.flat_param,
            param_infos,
            numels,
            shapes,
            strides,
            contiguities,
            fqns,
            shared_param_infos,
            param_extensions,
            _convert_to_params(params_to_flatten) if use_orig_params else None,
            _convert_to_params(shared_params) if use_orig_params else None,
            is_padding_mask,
        )

    def _validate_tensors_to_flatten(
        self, tensors: list[Tensor | nn.Parameter]
    ) -> tuple:
        """Validate the tensors to flatten and returns any necessary metadata."""
        dtype: torch.dtype | None = None
        # Return as the logical OR over each tensor's value
        flat_param_requires_grad: bool | None = None
        device: torch.device | None = None
        # For `use_orig_params=True`, permit non-uniform `requires_grad`
        for tensor in tensors:
            if isinstance(tensor, FlatParameter):
                raise ValueError("Cannot flatten a `FlatParameter`")
            if dtype is None and not tensor.is_floating_point():
                raise ValueError("Cannot flatten integer dtype tensors")
            if dtype is not None and tensor.dtype != dtype:
                raise ValueError(
                    f"Must flatten tensors with uniform dtype but got {dtype} "
                    f"and {tensor.dtype}"
                )
            if (
                not self._use_orig_params
                and flat_param_requires_grad is not None
                and tensor.requires_grad != flat_param_requires_grad
            ):
                raise ValueError(
                    "Must flatten tensors with uniform `requires_grad` when "
                    "`use_orig_params=False`"
                )
            if device is not None and tensor.device != device:
                raise ValueError(
                    "Must flatten tensors on the same device but got both "
                    f"{device} and {tensor.device}"
                )
            dtype = tensor.dtype
            flat_param_requires_grad = flat_param_requires_grad or tensor.requires_grad
            device = tensor.device
        if flat_param_requires_grad is None:
            raise AssertionError("Requires non-empty `tensors` list")
        return dtype, flat_param_requires_grad, device

    def flatten_tensors(
        self,
        tensors: list[Tensor],
        aligned_numel: int,
    ) -> Tensor:
        """
        Flatten ``tensors`` into a single flat tensor.

        The flattening optionally includes
        padding if ``aligned_numel`` is greater than 0, where ``aligned_numel``
        gives the numel required to have address alignment.

        NOTE: The padding alignment algorithm must be kept in sync with
        :meth:`_init_flat_param_metadata`. We separate the two methods because
        the initialization happens once, whereas this method may be called
        multiple times throughout training (e.g. for checkpointing).
        """
        if len(tensors) == 0:
            raise ValueError("Expects non-empty `tensors`")
        if aligned_numel < 0:
            raise ValueError(
                f"Expects non-negative `aligned_numel` but got {aligned_numel}"
            )
        dtype, _, device = self._validate_tensors_to_flatten(tensors)
        flat_tensors: list[Tensor] = []
        if aligned_numel > 0:
            total_numel = 0
            for tensor in tensors:
                numel_to_pad = aligned_numel - (total_numel % aligned_numel)
                if numel_to_pad > 0 and numel_to_pad < aligned_numel:
                    padding_tensor = _construct_padding_tensor(
                        numel_to_pad, dtype, False, device
                    )
                    flat_tensors.append(padding_tensor)
                    total_numel += numel_to_pad
                flat_tensors.append(
                    torch.flatten(_detach_if_needed(tensor))
                    if _is_truly_contiguous(tensor)
                    else _detach_if_needed(tensor).as_strided((tensor.numel(),), (1,))
                )
                total_numel += tensor.numel()
            numel_to_pad = self.world_size - (total_numel % self.world_size)
            if numel_to_pad > 0 and numel_to_pad < self.world_size:
                padding_tensor = _construct_padding_tensor(
                    numel_to_pad, dtype, False, device
                )
                flat_tensors.append(padding_tensor)
                total_numel += numel_to_pad
        else:
            flat_tensors = [
                torch.flatten(_detach_if_needed(tensor))
                if _is_truly_contiguous(tensor)
                else _detach_if_needed(tensor).as_strided((tensor.numel(),), (1,))
                for tensor in tensors
            ]
        return torch.cat(flat_tensors, dim=0)

    def flatten_tensors_into_flat_param(
        self,
        tensors: list[Tensor],
        aligned_numel: int,
        requires_grad: bool,
    ) -> FlatParameter:
        flat_param_data = self.flatten_tensors(tensors, aligned_numel)
        return FlatParameter(flat_param_data, requires_grad=requires_grad)

    def _init_param_reduce_dtypes(
        self,
        mp_param_dtype: torch.dtype | None,
        mp_reduce_dtype: torch.dtype | None,
    ) -> None:
        """
        Initialize param and reduce dtypes.

        Precondition: ``self.flat_param`` is set. This ensures that this
        handle's parameters have a single dtype.

        Postcondition: This sets ``self._fwd_bwd_param_dtype`` and
        ``self._reduce_dtype``. If ``mp_param_dtype`` or ``mp_reduce_dtype``
        is ``None``, then we assume the original parameter dtype. One special
        case is if ``mp_param_dtype`` is not ``None`` and ``mp_reduce_dtype``
        is ``None``, in which case we assume the gradient reduction dtype
        matches the forward/backward parameter dtype.
        """
        # Save whether these dtypes were specified so that we permit the
        # parameter dtype to change up until the lazy initialization
        self._low_prec_param_dtype_specified = mp_param_dtype is not None
        self._low_prec_reduce_dtype_specified = mp_reduce_dtype is not None
        if (
            self._low_prec_param_dtype_specified
            and not self._low_prec_reduce_dtype_specified
        ):
            # Special case: infer gradient reduction mixed precision
            self._fwd_bwd_param_dtype = mp_param_dtype
            self._reduce_dtype = self._fwd_bwd_param_dtype
        else:
            self._fwd_bwd_param_dtype = mp_param_dtype or self._orig_param_dtype
            self._reduce_dtype = mp_reduce_dtype or self._orig_param_dtype
        if self._fwd_bwd_param_dtype is None:
            raise AssertionError("Expected _fwd_bwd_param_dtype to be not None")
        if self._reduce_dtype is None:
            raise AssertionError("Expected _reduce_dtype to be not None")

    ###################################
    # SHARD INITIALIZATION & METADATA #
    ###################################
    @torch.no_grad()
    def shard(self):
        """
        Shard the handle's ``FlatParameter``.

        This allocates new memory for
        the sharded flat parameter and frees the unsharded flat parameter's
        storage.

        Postcondition: ``self.flat_param`` is the sharded flat parameter. Shard
        metadata attributes are set for all sharding strategies.
        """
        flat_param = self.flat_param
        if not self.uses_sharded_strategy:
            self._init_shard_metadata(0, 0, flat_param.numel() - 1)
        else:
            _p_assert(
                flat_param.storage_offset() == 0,
                "The `FlatParameter` is not the sole occupant of its storage",
            )
            sharded_flat_param, numel_padded = FlatParamHandle._get_shard(
                flat_param, self.rank, self.world_size
            )
            if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
                allocated = flat_param._typed_storage()._size() > 0
                if allocated:
                    flat_param._typed_storage()._resize_(0)
            flat_param.set_(sharded_flat_param)  # type: ignore[call-overload]
            start_idx = sharded_flat_param.numel() * self.rank
            end_idx = sharded_flat_param.numel() * (self.rank + 1) - 1  # inclusive
            self._init_shard_metadata(numel_padded, start_idx, end_idx)
        if self._use_orig_params:
            self._use_sharded_views()

    def _init_shard_metadata(
        self,
        numel_padded: int,
        unsharded_start_idx: int,
        unsharded_end_idx: int,
    ) -> None:
        """
        Initialize shard-related metadata for this rank's shard of the flat parameter.

        This includes ``_sharded_size``, ``_shard_param_infos``, and ``_shard_numel_padded``.

        Args:
            numel_padded (int): Numel padded for this rank's sharded flat
                parameter.
            unsharded_start_idx (int): Start index in the unsharded flat
            parameter assigned to this rank.
            unsharded_end_idx (int): End index (inclusive) in the unsharded
                flat parameter assigned to this rank.

        Precondition: ``self.flat_param`` 's data is the sharded flat
        parameter.
        """
        flat_param = self.flat_param
        flat_param._sharded_size = flat_param.size()  # type: ignore[attr-defined]
        sharded_flat_param_numel = flat_param.numel()  # includes `numel_padded`
        _p_assert(
            unsharded_start_idx >= 0 and unsharded_start_idx <= unsharded_end_idx,
            f"unsharded_start_idx: {unsharded_start_idx} unsharded_end_idx: {unsharded_end_idx}",
        )
        _p_assert(
            numel_padded <= sharded_flat_param_numel,
            f"numel_padded: {numel_padded} "
            f"sharded_flat_param_numel: {sharded_flat_param_numel}",
        )
        shard_param_infos = self._get_shard_metadata(
            unsharded_start_idx, unsharded_end_idx
        )
        if len(shard_param_infos) != flat_param._num_params:
            raise AssertionError(
                f"Expects length {flat_param._num_params} but got {len(shard_param_infos)}"
            )
        flat_param._shard_param_infos = shard_param_infos  # type: ignore[attr-defined]
        flat_param._shard_numel_padded = numel_padded  # type: ignore[attr-defined]

    def _get_shard_metadata(
        self,
        unsharded_start_idx: int,
        unsharded_end_idx: int,
    ) -> tuple[_ShardParamInfo, ...]:
        """
        Compute the shard metadata based on ``unsharded_start_idx`` and ``unsharded_end_idx`` (inclusive).

        ``unsharded_start_idx`` and ``unsharded_end_idx`` give the interval of the
        unsharded flat parameter specifying the shard.
        """
        flat_param_offsets = self._get_flat_param_offsets()
        if len(flat_param_offsets) != len(self.flat_param._numels_with_padding):
            raise AssertionError(
                f"Expected {len(self.flat_param._numels_with_padding)} but got {len(flat_param_offsets)}"
            )
        shard_param_infos: list[_ShardParamInfo] = []
        sharded_flat_param_numel = unsharded_end_idx - unsharded_start_idx + 1
        # `unsharded_param_start_idx` and `unsharded_param_end_idx` are indices
        # into the unsharded flat parameter (inclusive) of the given parameter
        for (
            (unsharded_param_start_idx, unsharded_param_end_idx),
            is_padding,
        ) in zip(flat_param_offsets, self.flat_param._is_padding_mask):
            if is_padding:
                continue
            in_sharded_flat_param = (
                unsharded_start_idx <= unsharded_param_end_idx
                and unsharded_end_idx >= unsharded_param_start_idx
            )
            if not in_sharded_flat_param:
                shard_param_info = _ShardParamInfo(False, None, None, None, None)
            else:
                if unsharded_start_idx <= unsharded_param_start_idx:
                    # This branch can only happen once since the rank's
                    # unsharded start index can only intersect one parameter
                    intra_param_start_idx = 0
                    offset_in_shard = unsharded_param_start_idx - unsharded_start_idx
                else:
                    intra_param_start_idx = (
                        unsharded_start_idx - unsharded_param_start_idx
                    )
                    offset_in_shard = 0
                if not (
                    offset_in_shard >= 0 and offset_in_shard < sharded_flat_param_numel
                ):
                    raise AssertionError(
                        f"Invalid `offset_in_shard` of {offset_in_shard} for "
                        f"sharded flat parameter with {sharded_flat_param_numel} numel"
                    )
                intra_param_end_idx = (
                    min(unsharded_param_end_idx, unsharded_end_idx)
                    - unsharded_param_start_idx
                )
                numel_in_shard = intra_param_end_idx - intra_param_start_idx + 1
                shard_param_info = _ShardParamInfo(
                    True,
                    offset_in_shard,
                    numel_in_shard,
                    intra_param_start_idx,
                    intra_param_end_idx,
                )
            shard_param_infos.append(shard_param_info)
        return tuple(shard_param_infos)

    @staticmethod
    def _get_unpadded_shard(
        tensor: Tensor,
        rank: int,
        world_size: int,
    ) -> tuple[Tensor, int]:
        """
        Return the unpadded shard of ``tensor`` for the given ``rank`` and ``world_size``.

        The returned value is a tuple of the shard of ``tensor`` without any
        padding and the numel to pad for that shard.

        If ``tensor`` is already flattened or may be viewed in the flattened
        shape (which is true in the expected usage), then this method does not
        allocate any new tensor memory.
        """
        chunks = (
            torch.flatten(tensor).chunk(world_size)
            if _is_truly_contiguous(tensor)
            else tensor.as_strided((tensor.numel(),), (1,)).chunk(world_size)
        )
        if len(chunks) < (rank + 1):
            # This rank gets an empty chunk fully padded with zeros since there
            # are not enough chunks across ranks
            chunk = chunks[0].new_empty(0)
        else:
            chunk = chunks[rank]
        numel_to_pad = chunks[0].numel() - chunk.numel()
        if numel_to_pad < 0:
            raise AssertionError(
                "Chunk's size should be at most the first chunk's size"
            )
        return chunk, numel_to_pad

    @staticmethod
    def _get_shard(
        tensor: Tensor,
        rank: int,
        world_size: int,
    ) -> tuple[Tensor, int]:
        """
        Return the shard of ``tensor`` with padding for the given ``rank`` and ``world_size`` and the numel padded for that shard.

        This method allocates new memory (via :meth:`clone`) since the
        unsharded ``tensor`` may be deallocated after this method returns.
        """
        chunk, numel_to_pad = FlatParamHandle._get_unpadded_shard(
            tensor, rank, world_size
        )
        shard = chunk.clone()
        if numel_to_pad > 0:
            shard = F.pad(shard, [0, numel_to_pad])
        return shard, numel_to_pad

    @staticmethod
    def _get_sharded_size(tensor: Tensor, rank: int, world_size: int) -> torch.Size:
        """
        Return the shape of ``tensor`` after sharding including padding.

        This requires ``tensor`` to have 1D shape and ensures that the returned
        shape is 1D.
        """
        if len(tensor.shape) != 1:
            raise AssertionError(f"Expected 1D tensor shape, got {tensor.shape}")
        unpadded_sharded_tensor, numel_to_pad = FlatParamHandle._get_unpadded_shard(
            tensor, rank, world_size
        )
        unpadded_sharded_size = unpadded_sharded_tensor.size()
        if len(unpadded_sharded_size) != 1:
            raise AssertionError(
                f"Expected 1D unpadded_sharded_size, got {unpadded_sharded_size}"
            )
        return torch.Size([unpadded_sharded_size[0] + numel_to_pad])

    def _get_flat_param_offsets(self) -> list[tuple[int, int]]:
        """
        Return [start, end] offsets of each original parameter's flattened data in the unsharded flat parameter (without padding).

        NOTE: The returned list includes elements for alignment padding.
        """
        cumulative_sum = list(accumulate(self.flat_param._numels_with_padding))
        starts = [0] + cumulative_sum[:-1]
        ends = [end - 1 for end in cumulative_sum]  # inclusive
        param_offsets = list(zip(starts, ends))
        return param_offsets

    @no_type_check
    def shard_metadata(
        self,
    ) -> FlatParamShardMetadata:
        """
        Return the shard-related metadata specific to this rank's shard of the flat parameter.

        NOTE: The returned tuple does not include elements for alignment
        padding but does account for the padding.
        """
        fqns_list = []
        shapes_list = []
        strides_list = []
        contiguities_list = []
        numels_list = []
        shard_param_offsets = []
        for fqn, shape, stride, contiguous, numel, shard_param_info in zip(
            self.flat_param._fqns,
            self.flat_param._shapes,
            self.flat_param._strides,
            self.flat_param._contiguities,
            self.flat_param._numels,
            self.flat_param._shard_param_infos,
        ):
            if not shard_param_info.in_shard:
                continue
            fqns_list.append(fqn)
            shapes_list.append(shape)
            strides_list.append(stride)
            contiguities_list.append(contiguous)
            numels_list.append(numel)
            shard_param_offsets.append(
                (
                    shard_param_info.intra_param_start_idx,
                    shard_param_info.intra_param_end_idx,
                )
            )
        return FlatParamShardMetadata(
            tuple(fqns_list),
            tuple(shapes_list),
            tuple(strides_list),
            tuple(contiguities_list),
            tuple(numels_list),
            tuple(shard_param_offsets),
        )

    @no_type_check
    @torch.no_grad()
    def init_flat_param_attributes(self) -> None:
        """
        This initializes some attributes on the handle's ``FlatParameter``.
        This should be called during lazy initialization since it requires the
        parameter to be on the compute device if not offloading to CPU and we
        want to give users the chance to move the parameter appropriately after
        the FSDP constructor.

        For each tensor attribute on the ``FlatParameter``, see the unshard and
        reshard methods in this class for the allocation and free pattern.
        """
        flat_param = self.flat_param
        if flat_param.dtype != self._orig_param_dtype:
            # Entering this branch means that the user changed the parameter
            # dtype after FSDP initialization, in which case we may need to
            # refresh some saved dtype attributes (dtypes specified as a part
            # of mixed precision take precedence).
            if not self._low_prec_param_dtype_specified:
                self._fwd_bwd_param_dtype = flat_param.dtype
            # For `reduce_dtype`, require `param_dtype` was not specified since
            # then we infer the `reduce_dtype` from the specified `param_dtype`
            if (
                not self._low_prec_reduce_dtype_specified
                and not self._low_prec_param_dtype_specified
            ):
                self._reduce_dtype = flat_param.dtype
            self._orig_param_dtype = flat_param.dtype
        cpu_device = torch.device("cpu")
        if self._offload_params:
            _p_assert(
                flat_param.device == cpu_device,
                f"Expects the `FlatParameter` to be on CPU when parameter CPU "
                f"offloading is enabled, not {flat_param.device}",
            )
        else:
            self._check_on_compute_device(self.flat_param)
        flat_param._local_shard = flat_param.data
        if self._offload_params:
            # Pin the memory for faster H2D transfer
            flat_param._local_shard = flat_param._local_shard.pin_memory()
            # Pre-allocate the sharded gradient on CPU to enable non-blocking
            # D2H transfer during the backward pass
            flat_param._cpu_grad = torch.zeros_like(
                flat_param._local_shard, device=cpu_device
            ).pin_memory()
        if self._uses_param_mixed_precision:
            # For parameter mixed precision, we maintain a low precision
            # sharded tensor on the compute device to be all-gathered (for
            # sharded strategies) or directly used (for `NO_SHARD`) for
            # computation.
            flat_param._mp_shard = torch.empty_like(
                flat_param._local_shard,
                device=self.device,
                dtype=self._fwd_bwd_param_dtype,
            )
            _free_storage(flat_param._mp_shard)
        if self.uses_sharded_strategy:
            # We maintain a padded unsharded tensor that serves as the
            # all-gather destination and owns the original parameter storages.
            unsharded_param_dtype = (
                self._fwd_bwd_param_dtype
                if self._uses_param_mixed_precision
                else flat_param.dtype
            )  # use low precision if parameter mixed precision is enabled
            padded_unsharded_numel = flat_param.numel() * self.world_size
            flat_param._full_param_padded = torch.empty(
                padded_unsharded_numel,
                device=self.device,
                dtype=unsharded_param_dtype,
            )
            flat_param._padded_unsharded_size = flat_param._full_param_padded.size()
            _free_storage(flat_param._full_param_padded)

            if self._uses_param_mixed_precision:
                # For parameter mixed precision, we maintain a full precision
                # padded unsharded tensor for when we force full precision.
                flat_param._full_prec_full_param_padded = torch.empty(
                    padded_unsharded_numel,
                    device=self.device,
                    dtype=flat_param.dtype,  # full precision
                )
                _free_storage(flat_param._full_prec_full_param_padded)

    ###################
    # UNSHARD/RESHARD #
    ###################
    def pre_unshard(self) -> bool:
        """
        Return ``False`` if this is a no-op and ``True`` otherwise.

        Postcondition: ``self.flat_param`` 's data is on the device for
        communication and is what should be all-gathered. This means that it
        matches the dtype of the expected unsharded parameter.
        """
        if (
            self._training_state == HandleTrainingState.SUMMON_FULL_PARAMS
            and self._skipped_use_sharded_views
        ):
            # Since this path imposes special semantics for the unsharded flat
            # parameter (e.g. forcing full precision), use sharded views to
            # reuse the existing logic for that special handling
            self._use_sharded_views()
        ret = False
        if self._use_orig_params and not self._skip_writeback_check:
            ret = self._writeback_orig_params()
        if (
            self.uses_sharded_strategy
            and not self._offload_params
            and not self.needs_unshard()
        ):
            pass  # no-op
        elif self._uses_param_mixed_precision and not self._force_full_precision:
            self._use_low_precision_shard()
            ret = True
        elif self._offload_params and self.flat_param.device != self.device:
            # NOTE: This creates a new tensor distinct from any attributes.
            self.flat_param_to(self.device, non_blocking=True)
            ret = True
        self._check_on_compute_device(self.flat_param)
        return ret

    def _use_low_precision_shard(self):
        """Allocate on the compute device and switch to using the low precision sharded flat parameter."""
        self._check_low_precision_shard()
        flat_param = self.flat_param
        _alloc_storage(
            flat_param._mp_shard,
            flat_param._local_shard.size(),  # type: ignore[attr-defined]
        )
        # `copy_()` implicitly casts to the low precision
        flat_param._mp_shard.copy_(  # type: ignore[attr-defined]
            flat_param._local_shard.to(  # type: ignore[attr-defined]
                self.device, non_blocking=True
            )
        )
        # Invariant: `_mp_shard` is always on the compute device.
        flat_param.data = flat_param._mp_shard  # type: ignore[attr-defined]

    def unshard(self):
        """
        Run the unshard logic.

        This includes all-gathering the flat parameter
        and switching to using the unsharded flat parameter. If the handle does
        not need unsharding, then this only switches to using the unsharded
        flat parameter. For ``NO_SHARD``, this is a no-op.

        If FSDP is in :meth:`summon_full_params` and the handle uses parameter
        mixed precision, then the parameter is forced to full precision.
        """
        if not self.needs_unshard():
            # Even when not needing an unshard, we should switch to using
            # the unsharded flat parameter
            unsharded_flat_param = (
                self._get_padded_unsharded_flat_param()
                if self.uses_sharded_strategy
                else self.flat_param
            )
            self._use_unsharded_flat_param(unsharded_flat_param)
            return
        unsharded_flat_param = self._alloc_padded_unsharded_flat_param()
        padded_unsharded_flat_param = self._all_gather_flat_param(unsharded_flat_param)
        self._use_unsharded_flat_param(padded_unsharded_flat_param)

    def needs_unshard(self) -> bool:
        """Return if the handle's flat parameter needs to be unsharded."""
        if not self.uses_sharded_strategy:
            return False
        unsharded_flat_param = self._get_padded_unsharded_flat_param()
        already_unsharded = _same_storage_size(
            unsharded_flat_param, unsharded_flat_param.numel()
        )
        return not already_unsharded

    def _alloc_padded_unsharded_flat_param(self):
        """
        Allocate the *padded* unsharded flat parameter.

        The unpadded unsharded
        flat parameter is always a view into the padded one. This padded
        parameter is saved to a different attribute on the ``FlatParameter``
        depending on if we force full precision.
        """
        self._check_sharded_strategy()
        flat_param = self.flat_param
        unsharded_flat_param = self._get_padded_unsharded_flat_param()
        self._check_storage_freed(unsharded_flat_param)
        _alloc_storage(unsharded_flat_param, flat_param._padded_unsharded_size)  # type: ignore[attr-defined]
        return unsharded_flat_param

    def _get_padded_unsharded_flat_param(self) -> torch.Tensor:
        """
        Return a reference to the padded unsharded flat parameter depending on the calling context.

        This should only be called if using a sharded strategy.
        """
        self._check_sharded_strategy()
        flat_param = self.flat_param
        if self._force_full_precision and self._uses_param_mixed_precision:
            # When parameter mixed precision is enabled, we use a different
            # tensor as the all-gather destination to preserve the invariant
            # that  `_full_param_padded` is in the low precision
            unsharded_flat_param = flat_param._full_prec_full_param_padded  # type: ignore[attr-defined]
            _p_assert(
                unsharded_flat_param.dtype != self._fwd_bwd_param_dtype,
                f"Expects full precision but got {self._fwd_bwd_param_dtype}",
            )
            # For no-reshard-after-forward strategies, `_full_param_padded` may
            # still be allocated from a previous forward. As we are forcing
            # full precision here, the full-precision unsharded copy may be
            # modified, invalidating the existing low-precision unsharded copy,
            # so we should free it here to ensure a new all-gather for the next
            # forward/backward computation to persist the modifications.
            if flat_param._full_param_padded.untyped_storage().size() > 0:
                _free_storage(flat_param._full_param_padded)
        else:
            unsharded_flat_param = flat_param._full_param_padded  # type: ignore[attr-defined]
        return unsharded_flat_param

    def _all_gather_flat_param(
        self,
        padded_unsharded_flat_param: Tensor,
    ) -> Tensor:
        """
        All-gather the handle's flat parameter to the destination ``padded_unsharded_flat_param``.

        Then switch to use the all-gathered tensor.
        """
        _p_assert(
            hasattr(self, "process_group") and hasattr(self, "world_size"),
            "Expects a process group and world size to have been set via `shard()`",
        )
        sharded_flat_param = self.flat_param.data
        expected_numel = sharded_flat_param.numel() * self.world_size
        _p_assert(
            padded_unsharded_flat_param.numel() == expected_numel,
            f"Expects {expected_numel} numel but got {padded_unsharded_flat_param.numel()}",
        )

        pg = (
            self._fake_process_group
            if self._use_fake_all_gather
            else self.process_group
        )

        # HACK this should be handled by C10D
        if sharded_flat_param.is_cpu:  # type: ignore[attr-defined]
            tensor_list = list(
                torch.chunk(
                    padded_unsharded_flat_param,
                    dist.get_world_size(pg),  # type: ignore[arg-type]
                )
            )
            dist.all_gather(tensor_list, sharded_flat_param, group=pg)
        else:
            dist.all_gather_single(
                padded_unsharded_flat_param,
                sharded_flat_param,
                pg,
            )

        if self._offload_params:
            # In case of offloading, `flat_param.data` (i.e. sharded param) is
            # created on the pre-unshard stream. We need to hand it over to the
            # unshard stream for all-gather
            _no_dispatch_record_stream(
                sharded_flat_param,
                self._device_handle.current_stream(),  # unshard_stream
            )
        return padded_unsharded_flat_param

    def _use_unsharded_flat_param(
        self,
        padded_unsharded_flat_param: torch.Tensor,
    ) -> None:
        """
        Switch to use the *unpadded* unsharded flat parameter.

        This is a view into the *padded* unsharded flat parameter.
        """
        unsharded_size = self.flat_param._unpadded_unsharded_size
        flat_param_part = padded_unsharded_flat_param[: unsharded_size.numel()]
        # slicing [:] is not visible to autograd because of .data
        self.flat_param.data = flat_param_part
        in_forward = self._training_state == HandleTrainingState.FORWARD
        in_pre_backward = self._training_state == HandleTrainingState.BACKWARD_PRE
        if self._use_orig_params:
            if self._skipped_use_sharded_views and in_pre_backward:
                # This call corresponds to the complementary pre-backward
                # `_use_unsharded_views()` to the skipped pre-forward
                # `_use_sharded_views()`, so we should skip this one too.
                return
            # We use `Tensor` views in the forward so that they are tracked by
            # autograd. We use them in the pre-backward as well to support
            # reentrant activation checkpointing, which needs the views to be
            # tracked by autograd in the backward pass's recomputed forward.
            self._use_unsharded_views(
                as_params=(not in_forward and not in_pre_backward)
            )
        elif in_forward:
            self._use_unsharded_views(as_params=False)

    def post_unshard(self):
        """
        Run the post-unshard logic.

        This includes freeing the low precision shard if needed.
        """
        if self._uses_param_mixed_precision and self.uses_sharded_strategy:
            self._free_low_precision_sharded_param()
        self._check_on_compute_device(self.flat_param)

    def _free_low_precision_sharded_param(self):
        """Frees the low precision sharded flat parameter."""
        self._check_low_precision_shard()
        # `_mp_shard` is allocated in the pre-unshard stream, consumed in the
        # unshard stream for sharded strategies, and consumed in both the
        # unshard and default streams for `NO_SHARD`. For sharded strategies,
        # the current stream here is the unshard stream, and for `NO_SHARD`,
        # it is the default stream. For `NO_SHARD`, only recording for the
        # default stream suffices since the default stream waits for the
        # unshard stream.
        _no_dispatch_record_stream(
            self.flat_param._mp_shard,
            self._device_handle.current_stream(),  # type: ignore[attr-defined]
        )
        _free_storage(self.flat_param._mp_shard)  # type: ignore[attr-defined]

    @torch.no_grad()
    def unshard_grad(self):
        """
        Unshard the handle's ``FlatParameter``'s gradient.

        If all ranks have
        ``None`` gradient, then all original parameters will as well. This
        method performs an all-reduce and an all-gather. The additional
        all-reduce is tolerable since this method is not meant to be used on
        the computation critical path.

        Postcondition: ``_saved_grad_shard`` is defined and contains the value
        to set ``flat_param.grad`` after gradients are resharded.
        """
        if not self.uses_sharded_strategy:
            self._use_unsharded_grad_views()
            return
        flat_param = self.flat_param
        self._check_unsharded(flat_param)

        # Check if all ranks have a `None` gradient
        num_grad_none = torch.zeros(1, dtype=torch.int32, device=self.device)
        num_grad_none[0] = flat_param.grad is None
        dist.all_reduce(num_grad_none, group=self.process_group)
        if num_grad_none[0] == self.world_size:
            flat_param._saved_grad_shard = None  # type: ignore[assignment]
            self._use_unsharded_grad_views()
            return

        if flat_param.grad is None:
            # In the case that only some ranks have `None` gradient, we use
            # zeros to approximate as a best effort attempt
            if self._debug_level == dist.DebugLevel.INFO:
                warnings.warn(
                    f"[Rank {self.rank}] Only some but not all ranks have a "
                    "`None` `FlatParameter` gradient, so FSDP is using zeros to "
                    "approximate those ranks' sharded gradients being `None`",
                    stacklevel=2,
                )
            flat_param._saved_grad_shard = None  # type: ignore[assignment]
            sharded_grad = torch.zeros(flat_param._sharded_size, device=self.device)  # type: ignore[attr-defined]
        else:
            self._check_sharded(flat_param.grad)
            flat_param._saved_grad_shard = flat_param.grad  # type: ignore[attr-defined]
            sharded_grad = flat_param._saved_grad_shard  # type: ignore[attr-defined]
        padded_unsharded_grad = torch.empty(
            flat_param._padded_unsharded_size,  # type: ignore[attr-defined]
            device=self.device,
            dtype=sharded_grad.dtype,
        )
        dist.all_gather_single(padded_unsharded_grad, sharded_grad, self.process_group)
        unsharded_size = self.flat_param._unpadded_unsharded_size
        flat_param.grad = padded_unsharded_grad[: unsharded_size.numel()].view(
            unsharded_size
        )
        self._use_unsharded_grad_views()

    def reshard_grad(self):
        if self._use_orig_params:
            self._use_sharded_grad_views()
        if not self.uses_sharded_strategy:
            return
        self.flat_param.grad = self.flat_param._saved_grad_shard  # type: ignore[attr-defined]
        delattr(self.flat_param, "_saved_grad_shard")

    def prepare_gradient_for_backward(self):
        """
        Prepare the gradient for the backward computation.

        This is done by saving and clearing any existing sharded gradient
        in ``.grad`` to enable computing a new unsharded gradient.
        """
        _p_assert(
            self._training_state
            in (HandleTrainingState.BACKWARD_PRE, HandleTrainingState.IDLE),
            "Expects to be in `BACKWARD_PRE` or `IDLE` (if prefetching)",
        )
        flat_param = self.flat_param
        if flat_param.grad is not None and (
            flat_param.grad.size() != flat_param._unpadded_unsharded_size
            or flat_param.grad.device != flat_param.device  # grad on CPU
        ):
            self._check_on_compute_device(self.flat_param)
            grad_offloaded = flat_param.grad.device != self.device
            _p_assert(
                not grad_offloaded or self._offload_params,
                f"Expects the sharded gradient to be on {self.device} "
                f"but got {flat_param.grad.device}",
            )
            prev_iter_synced_gradients = (
                flat_param.grad.size() == flat_param._local_shard.size()  # type: ignore[attr-defined]
            )
            if prev_iter_synced_gradients:
                # TODO (awgu): Gradient accumulation outside `no_sync()`
                # does not work with CPU offloading. The issue should be
                # that, in the post-backward hook, we cannot do an addition
                # between a CPU tensor (the existing sharded gradient) and
                # a GPU tensor (the new sharded gradient).
                if not grad_offloaded:
                    flat_param._saved_grad_shard = flat_param.grad.data  # type: ignore[attr-defined]
                    sharded_grad = flat_param._saved_grad_shard  # type: ignore[attr-defined]
                else:
                    _p_assert(
                        hasattr(flat_param, "_cpu_grad"),
                        "`_cpu_grad` should be defined if the gradient is on CPU",
                    )
                    sharded_grad = flat_param._cpu_grad  # type: ignore[attr-defined]
                # If user specified to keep the gradient in low precision, then
                # the gradient may still be of the low precision dtype if the
                # user did not set the gradient to `None` after the previous
                # backward, in which case FSDP should cast back to the full
                # precision dtype so that FSDP can accumulate in that dtype in
                # the post-backward hook and assign to `.grad` in that dtype in
                # the post-backward callback.
                local_shard_dtype = flat_param._local_shard.dtype  # type: ignore[attr-defined]
                if (
                    self._keep_low_precision_grads
                    and sharded_grad.dtype != local_shard_dtype
                ):
                    sharded_grad.data = sharded_grad.to(local_shard_dtype)
            else:
                padded_unsharded_size = flat_param._padded_unsharded_size  # type: ignore[attr-defined]
                _p_assert(
                    flat_param.grad.size() == padded_unsharded_size,
                    "Expects `.grad` to be the unsharded gradient in "
                    f"`no_sync()` with size {padded_unsharded_size} "
                    f"but got size {flat_param.grad.size()}",
                )
            flat_param.grad = None

    def prepare_gradient_for_optim(self):
        """Prepare the gradient for optimizer computation by moving the sharded gradient to the ``.grad`` attribute."""

        def cast_grad_to_param_dtype_if_needed(flat_param):
            # TODO (rohan-varma): test for full precision with keep_low_precision_grads
            if not self._force_full_precision and self._keep_low_precision_grads:
                _p_assert(flat_param.grad is not None, "Unexpected None grad!")
                if flat_param.grad.dtype != self._fwd_bwd_param_dtype:
                    flat_param.grad.data = flat_param.grad.to(self._fwd_bwd_param_dtype)
                    if self._use_orig_params:
                        self._use_sharded_grad_views()

        flat_param = self.flat_param
        # TODO (awgu): We should replace these conditional checks to encode
        # the logical intention more directly.
        if hasattr(flat_param, "_cpu_grad"):
            # NOTE: This branch includes `NO_SHARD`.
            self._check_sharded(flat_param)
            self._check_on_cpu(flat_param)
            flat_param.grad = flat_param._cpu_grad  # type: ignore[attr-defined]
            cast_grad_to_param_dtype_if_needed(flat_param)
        elif hasattr(flat_param, "_saved_grad_shard"):
            self._check_sharded(flat_param)
            self._check_on_compute_device(flat_param)
            if flat_param._saved_grad_shard is not None:
                self._check_on_compute_device(flat_param._saved_grad_shard)  # type: ignore[attr-defined]
            # If no sharded gradient was computed this iteration, then there is
            # no need to forward `_saved_grad_shard` to `grad`
            if flat_param._post_backward_called:  # type: ignore[attr-defined]
                flat_param.grad = flat_param._saved_grad_shard  # type: ignore[attr-defined]
                if flat_param.grad is not None:
                    cast_grad_to_param_dtype_if_needed(flat_param)
        else:
            _p_assert(
                not self.uses_sharded_strategy or not flat_param._post_backward_called,  # type: ignore[attr-defined]
                "All sharded parameters that received a gradient in the "
                "post-backward should use `_saved_grad_shard`",
            )
        # Delete `_saved_grad_shard` since its existence indicates a previous
        # gradient to accumulate with in the post-backward hook
        if hasattr(flat_param, "_saved_grad_shard"):
            delattr(flat_param, "_saved_grad_shard")

    @contextlib.contextmanager
    def to_cpu(self):
        """
        Move the unpadded unsharded flat parameter to CPU while in the context and moves it back to the previous device upon exit.

        For now, this assumes the ``FlatParameter`` is the unpadded unsharded flat parameter
        since (1) there is no reason to include the padding in the copy and (2)
        there is no use case for the sharded flat parameter.

        Precondition: ``self.flat_param`` 's data is the unpadded unsharded
        flat parameter on the compute device, and the handle uses a sharded
        strategy.
        Postcondition: Same as the precondition.
        """
        self._check_sharded_strategy()
        _p_assert(
            self.flat_param.size() == self.flat_param._unpadded_unsharded_size,
            f"Expects size {self.flat_param._unpadded_unsharded_size} but got {self.flat_param.size()}",
        )
        self._check_on_compute_device(self.flat_param)
        # Check that the unpadded unsharded flat parameter is a view into the
        # padded unsharded flat parameter as expected
        # NOTE: This check is not strictly needed for correctness but is a
        # useful sanity check since the tensor should only be used internally.
        _p_assert(
            _same_storage(self.flat_param, self._get_padded_unsharded_flat_param()),
            "Expects the unpadded parameter to be a view into the padded parameter",
        )
        cpu_offload_ctx = (
            contextlib.nullcontext()
            if self.flat_param.device.type == "cpu"
            else self._offload_to_cpu()
        )
        with cpu_offload_ctx:
            yield

    @contextlib.contextmanager
    def _offload_to_cpu(self):
        self.flat_param_to(torch.device("cpu"))
        self._free_unsharded_flat_param()
        try:
            yield
        finally:
            self._restore_unsharded_flat_param_from_cpu()

    def _restore_unsharded_flat_param_from_cpu(self):
        _p_assert(
            self.flat_param.size() == self.flat_param._unpadded_unsharded_size,
            f"Expects size {self.flat_param._unpadded_unsharded_size} but got {self.flat_param.size()}",
        )
        padded_unsharded_flat_param = self._alloc_padded_unsharded_flat_param()
        # Copy from CPU to the compute device
        padded_unsharded_flat_param[: self.flat_param.numel()].copy_(self.flat_param)
        self._use_unsharded_flat_param(padded_unsharded_flat_param)

    def reshard(self, free_unsharded_flat_param: bool):
        """
        Run the reshard logic.

        This includes freeing the unsharded flat
        parameter if ``free_unsharded_flat_param`` and switching to using the
        sharded flat parameter. Note that this also implicitly offloads
        the sharded flat parameter (if CPU offload is enabled) by pointing
        it to the ``_local_shard`` attribute which resides on CPU.
        """
        # Switch to the sharded `FlatParameter` before freeing to prevent
        # "use-after-free"-type bugs with external profiling tools, where for
        # `use_orig_params=True`, the `param` does not point to valid memory
        # when setting `param.data = ...` in `_use_sharded_views()`.
        self._use_sharded_flat_param()
        if free_unsharded_flat_param:
            self._free_unsharded_flat_param()

    def post_reshard(self):
        """
        Run the post-reshard logic.

        This includes freeing any memory that
        can now be freed given that the ``FlatParameter`` points to the full
        precision sharded flat parameter.

        Precondition: ``self.flat_param`` 's data points to the full precision
        sharded flat parameter.
        """
        # For `NO_SHARD`, `_mp_shard` is not freed in the post-unshard since it
        # is also the low precision *unsharded* flat parameter. Hence, we delay
        # the free until the reshard.
        if (
            self._uses_param_mixed_precision
            and not self.uses_sharded_strategy
            and not self._force_full_precision  # did not use the low precision shard
        ):
            self._free_low_precision_sharded_param()

    def _free_unsharded_flat_param(self):
        """
        Free the padded unsharded flat parameter. We allow this
        function to be called even when storage is not allocated

        The tensor to free depends
        on the calling context since the unshard may have forced full
        precision, in which case a different tensor is used.
        """
        self._check_sharded_strategy()
        unsharded_flat_param = self._get_padded_unsharded_flat_param()
        self._check_on_compute_device(unsharded_flat_param)
        # Do not free the memory until all ops in the current stream finish
        _no_dispatch_record_stream(
            unsharded_flat_param, self._device_handle.current_stream()
        )
        _free_storage(unsharded_flat_param)

    def _use_sharded_flat_param(self) -> None:
        """Switches to using the sharded flat parameter."""
        flat_param = self.flat_param
        if self._use_orig_params:
            in_forward = self._training_state == HandleTrainingState.FORWARD
            skip_use_sharded_views = (
                torch.is_grad_enabled()
                and in_forward
                and self._sharding_strategy
                in NO_RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES
            )
            # Only incur the extra `.data` call if needed
            if skip_use_sharded_views:
                unsharded_flat_param = flat_param.data
        if self._offload_params:
            device = flat_param._local_shard.device  # type: ignore[attr-defined]
            _p_assert(
                device == torch.device("cpu"),
                f"Expects the local shard to be on CPU but got {device}",
            )
        flat_param.data = flat_param._local_shard  # type: ignore[attr-defined]
        if self._use_orig_params:
            if skip_use_sharded_views:  # type: ignore[possibly-undefined]
                self._unsharded_flat_param_for_skipped_views = unsharded_flat_param  # type: ignore[possibly-undefined]
            else:
                self._use_sharded_views()
            # For the post-forward reshard, we may try to use sharded gradient
            # views (or unsharded gradient views if a gradient was accumulated
            # in `no_sync()`), but for the post-backward reshard, we delay the
            # call to after the reduce-scatter.
            if (
                in_forward  # type: ignore[possibly-undefined]
                # Skip using gradient views if skipped using sharded views
                # since exposing unsharded parameters with sharded gradients
                # may be confusing to the user
                and not self._skipped_use_sharded_views
            ):
                # TODO: Change `_unpadded_unsharded_size` if we change the
                # gradient to be computed directly with padding.
                accumulated_grad_in_no_sync = (
                    flat_param.grad is not None
                    and self.uses_sharded_strategy
                    and flat_param.grad.shape == flat_param._unpadded_unsharded_size
                )
                if accumulated_grad_in_no_sync:
                    self._use_unsharded_grad_views()
                else:
                    self._use_sharded_grad_views()

    #########
    # VIEWS #
    #########
    @no_type_check
    def _get_unflat_views_unaligned(
        self,
        tensor: torch.Tensor | None = None,
    ) -> Iterator[Tensor]:
        """
        Return unflattened ``Tensor`` views into ``tensor``.

        If `tensor`` is ``None``,  ``flat_param`` is used. The unflattening is based
        on ``flat_param`` 's metadata.

        Examples for ``tensor`` include ``flat_param.grad`` or unsharded
        tensor optimizer state.
        """
        flat_param = self.flat_param
        if tensor is None:
            tensor = flat_param
        views = (
            _ext_post_unflatten_transform(
                subtensor.view(shape)
                if contiguous
                else subtensor.as_strided(shape, stride),
                param_extension,
                self._fsdp_extension,
            )
            for (subtensor, shape, stride, contiguous, param_extension) in zip(
                torch.split(tensor, flat_param._numels, dim=0),
                flat_param._shapes,
                flat_param._strides,
                flat_param._contiguities,
                flat_param._param_extensions,
            )
        )
        return views

    @no_type_check
    def _get_unflat_views_aligned(
        self,
        tensor: Tensor | None = None,
    ) -> list[Tensor]:
        """
        Return unflattened ``Tensor`` views into ``tensor`` with handling for padding.

        This method has the same contract as :meth:`_get_unflat_views_unaligned`
        except it checks for ``None`` placeholders representing padding for
        alignment, which may incur slightly more CPU overhead.
        """
        flat_param = self.flat_param
        if tensor is None:
            tensor = flat_param
        splits: list[Tensor] = torch.split(
            tensor, flat_param._numels_with_padding, dim=0
        )
        idx = 0
        views: list[Tensor] = []
        for split, is_padding in zip(splits, flat_param._is_padding_mask):
            if is_padding:
                continue
            views.append(
                _ext_post_unflatten_transform(
                    split.view(flat_param._shapes[idx])
                    if flat_param._contiguities[idx]
                    else split.as_strided(
                        flat_param._shapes[idx], flat_param._strides[idx]
                    ),
                    flat_param._param_extensions[idx],
                    self._fsdp_extension,
                )
            )
            idx += 1
        return views

    @no_type_check
    @torch.enable_grad()
    def _use_unsharded_views(self, as_params: bool) -> None:
        """
        Unflatten the unsharded flat parameter by setting the original parameter variables to be views into it.

        Args:
            as_params (bool): If ``True``, then registers the original
                parameters as ``nn.Parameter`` s; if ``False``, then registers
                the original parameters only as ``Tensor`` s. ``False`` should
                be used during forward/backward computation and when hiding the
                original parameters from :meth:`nn.Module.named_parameters`.

        Note:
            when prefetching for next forward, current forward may be
            annotated with `@torch.no_grad()`
            `@torch.enable_grad()` ensures non-empty `view.grad_fn`
            otherwise `_post_backward_hook` will not get called
        """
        flat_param = self.flat_param
        self._check_unsharded(flat_param)
        views = self._get_unflat_views()
        from torch.distributed.tensor import DTensor

        for i, (view, (param_name, module, _)) in enumerate(
            zip(views, flat_param._param_infos)
        ):
            if self._use_orig_params and as_params:
                if type(view) is DTensor:
                    # A `DTensor` `view` is not compatible with assigning
                    # `param.data = view`, so we cannot preserve the parameter
                    # variable.
                    self._setattr_param(
                        module,
                        param_name,
                        nn.Parameter(view, requires_grad=flat_param.requires_grad),
                    )
                    continue
                param = self.flat_param._params[i]
                self._setattr_param(module, param_name, param)
                param.data = view
            elif as_params:
                self._setattr_param(
                    module,
                    param_name,
                    nn.Parameter(view, requires_grad=flat_param.requires_grad),
                )
            else:  # `as_params=False`
                param_var: Tensor = view
                if self._use_orig_params:
                    if self._training_state == HandleTrainingState.FORWARD:
                        # Save the `Tensor` for the pre-backward
                        self.flat_param._tensors[i] = view  # save for pre-backward
                    elif self._training_state == HandleTrainingState.BACKWARD_PRE:
                        # Use the saved `Tensor` variable from the forward to
                        # preserve the autograd graph so that the post-backward
                        # hook fires (e.g. for reentrant AC)
                        tensor = self.flat_param._tensors[i]
                        tensor.data = view
                        param_var = tensor
                self._setattr_tensor(module, param_name, param_var)
                if (
                    self._use_orig_params
                    and self._training_state == HandleTrainingState.FORWARD
                ):
                    module._parameters[param_name] = param_var
        for i, (
            param_name,
            module,
            _,
            prim_param_name,
            prim_module,
            _,
        ) in enumerate(self.flat_param._shared_param_infos):
            prim_param: Tensor | nn.Parameter = getattr(prim_module, prim_param_name)
            _p_assert(
                not as_params or isinstance(prim_param, nn.Parameter),
                f"as_params={as_params} type(prim_param)={type(prim_param)}",
            )
            if self._use_orig_params and as_params:
                shared_param = self.flat_param._shared_params[i]
                self._setattr_param(module, param_name, shared_param)
                shared_param.data = prim_param
            elif as_params:
                self._setattr_param(module, param_name, prim_param)
            else:
                self._setattr_tensor(module, param_name, prim_param)
                if (
                    self._use_orig_params
                    and self._training_state == HandleTrainingState.FORWARD
                ):
                    module._parameters[param_name] = prim_param

    @no_type_check
    def _use_unsharded_grad_views(self) -> None:
        """
        Unflatten the unsharded flat parameter's gradient.

        The original parameter variables' gradients are set to be views into
        the unsharded flat parameter's gradient.
        """
        # Expects the gradient to be in `flat_param.grad`
        if self.flat_param.grad is None:
            for param in chain(self.flat_param._params, self.flat_param._shared_params):
                param.grad = None
            return
        self._check_unsharded(self.flat_param.grad)
        views = self._get_unflat_views(self.flat_param.grad)
        for i, (view, (param_name, module, _)) in enumerate(
            zip(views, self.flat_param._param_infos)
        ):
            _p_assert(
                hasattr(module, param_name),
                f"{self.flat_param._fqns[i]} is missing",
            )
            param = getattr(module, param_name)
            if (
                param.shape != view.shape
                or param.dtype != view.dtype
                or param.device != view.device
            ):
                # NOTE: This is a hack using `.data` to side step the check
                # that parameter/gradient sizes/dtypes/devices match. From
                # calling `reshard()`, `param` has the sharded size, has the
                # full precision dtype, and if CPU offloading is enabled, is on
                # CPU. Thus, one or more of the following cases can hold when
                # in `no_sync()`, where `view` is the original parameter's
                # gradient:
                # 1. `view` can have the unsharded size.
                # 2. `view` can have the parameter low precision dtype.
                # 3. `view` can be on GPU.
                if param.grad is None:
                    param.grad = torch.empty_like(param)
                param.grad.data = view
            else:
                param.grad = view
        for (
            param_name,
            module,
            module_name,
            prim_param_name,
            prim_module,
            _,
        ) in self.flat_param._shared_param_infos:
            _p_assert(
                hasattr(module, param_name),
                f"{module_name + '.' + param_name if module_name else param_name} is missing",
            )
            param = getattr(module, param_name)
            prim_param = getattr(prim_module, prim_param_name)
            if (
                param.shape != prim_param.grad.shape
                or param.dtype != prim_param.grad.dtype
                or param.device != prim_param.grad.device
            ):
                # NOTE: This is the same hack to use `.data` to side step the
                # size check.
                if param.grad is None:
                    param.grad = torch.empty_like(param)
                param.grad.data = prim_param.grad
            else:
                param.grad = prim_param.grad

    @contextlib.contextmanager
    def unflatten_as_params(self) -> Generator:
        """
        Unflatten the original parameters.

        The function assumes that the flat parameter is unsharded. When in the context,
        unflattens the original parameters as ``nn.Parameter`` views into the
        flat parameter, and after the context, restores the original parameters
        as ``Tensor`` views into the flat parameter.
        """
        self._use_unsharded_views(as_params=True)
        try:
            yield
        finally:
            self._use_unsharded_views(as_params=False)

    @no_type_check
    @torch.no_grad()
    def _use_sharded_views(self) -> None:
        """
        Set the original parameter variables' data to be flattened views into the sharded flat parameter.

        The views are kept as flattened to simplify the case where a parameter
        is sharded across ranks. Parameters whose data is not present in the
        sharded flat parameter have their data set to a size-0 empty tensor. We
        do not delete them to ensure to preserve expected behaviors like model
        printability. Parameters whose data is present must preserve their
        variables to be passable to an optimizer.
        """
        self._unsharded_flat_param_for_skipped_views = None
        if not self.uses_sharded_strategy:
            # For `NO_SHARD`, use the *unflattened* unsharded views since we
            # have the unsharded parameter
            self._use_unsharded_views(as_params=True)
            return
        flat_param = self.flat_param
        self._check_sharded(flat_param)
        # Construct once and reuse for all parameters not in the local shard
        size_0_empty_tensor = torch.empty(
            0,
            dtype=self.flat_param.dtype,  # in case `flat_param` changed dtype
            device=self.flat_param.device,
            requires_grad=False,
        )
        for param, shard_param_info, (param_name, module, _) in zip(
            flat_param._params, flat_param._shard_param_infos, flat_param._param_infos
        ):
            self._setattr_param(module, param_name, param)
            if not shard_param_info.in_shard:
                # Allow the original data to be freed via garbage collection
                param.data = size_0_empty_tensor
            else:
                offset = shard_param_info.offset_in_shard
                numel_in_shard = shard_param_info.numel_in_shard
                param.data = flat_param[offset : offset + numel_in_shard]
        if self.flat_param._shared_params is None:
            raise AssertionError("Expected _shared_params to be not None")
        for param, (param_name, module, _, prim_param_name, prim_module, _) in zip(
            self.flat_param._shared_params, self.flat_param._shared_param_infos
        ):
            self._setattr_param(module, param_name, param)
            prim_param = getattr(prim_module, prim_param_name)
            param.data = prim_param  # could be both empty and non-empty
        if self._training_state == HandleTrainingState.BACKWARD_POST:
            # Clear the saved `Tensor`s since they are unneeded now
            for i in range(len(self.flat_param._tensors)):
                self.flat_param._tensors[i] = None

    @no_type_check
    @torch.no_grad()
    def _use_sharded_grad_views(self) -> None:
        """
        Set the original parameter variables' gradients to be flattened views into the sharded flat parameter's gradient.

        This is a no-op if there is no gradient.

        Parameters whose data is not present in the sharded flat parameter and
        parameters with ``requires_grad=False`` have their gradients set to
        ``None``. Since the gradient variables do not need to be preserved,
        this method does not manipulate existing ``Tensor`` data directly and
        creates new ``Tensor`` variables instead.
        """
        flat_param = self.flat_param
        self._check_sharded(flat_param)
        grad = self.sharded_grad
        if grad is None:
            for param in chain(flat_param._params, flat_param._shared_params):
                param.grad = None
            return
        self._check_sharded(grad)
        for param, shard_param_info, is_grad_none in zip(
            flat_param._params,
            flat_param._shard_param_infos,
            flat_param._is_grad_none_mask,
        ):
            if not shard_param_info.in_shard:
                param.grad = None
            else:
                numel_in_shard = shard_param_info.numel_in_shard
                if param.requires_grad and not is_grad_none:
                    offset = shard_param_info.offset_in_shard
                    if self._keep_low_precision_grads or param.dtype != grad.dtype:
                        # NOTE: This is a hack using `.data` to side step the
                        # check that parameter/gradient dtypes match. Here,
                        # `param` has full precision; `grad` has low precision.
                        if param.grad is None:
                            # `.grad` must have the same shape as `param`
                            param.grad = torch.empty_like(param)
                        param.grad.data = grad[
                            offset : offset + numel_in_shard
                        ].reshape(param.shape)
                    else:
                        param.grad = grad[offset : offset + numel_in_shard].reshape(
                            param.shape
                        )
                else:
                    param.grad = None
        if flat_param._shared_params is None:
            raise AssertionError("Expected _shared_params to be not None")
        for param, (_, _, _, prim_param_name, prim_module, _) in zip(
            flat_param._shared_params, flat_param._shared_param_infos
        ):
            in_sharded_flat_param = hasattr(prim_module, prim_param_name)
            if in_sharded_flat_param and param.requires_grad:
                prim_param = getattr(prim_module, prim_param_name)
                param.grad = prim_param.grad  # share the same reference
            else:
                param.grad = None

    @no_type_check
    @torch.no_grad()
    def _writeback_orig_params(self) -> bool:
        """
        Write back any parameters that changed storage to the handle's ``FlatParameter``.

        Iterates over the original parameters and writes back any parameters
        that changed storages (due to a non-inplace operator) to the handle's
        ``FlatParameter``. This method preserves the ``FlatParameter` 's
        device even if an original parameter's device changes.

        Raises:
            RuntimeError: If an original parameter or gradient changes storages
            but no longer has the expected flattened shape.
        Returns: ``True`` if some writeback happened, and ``False`` otherwise.
        """
        if (
            self.uses_sharded_strategy
            and not self.is_sharded(self.flat_param)
            and not self._skipped_use_sharded_views
        ):
            # For `NO_SHARD`, we may still need to writeback
            return False
        flat_param = self.flat_param
        wroteback = False
        if self._skipped_use_sharded_views and self.uses_sharded_strategy:
            # NOTE: We must use the unsharded flat parameter from which the
            # unsharded views were computed, not the one from the current
            # calling context (`_get_padded_unsharded_flat_param()`) since that
            # may be different (e.g. the model changed from train to eval).
            flat_param_tensor = self._unsharded_flat_param_for_skipped_views
            _p_assert(
                _data_ptr_allocated(flat_param_tensor),
                "If skipped using sharded views, the unsharded flat parameter "
                "should be allocated",
            )
        else:
            flat_param_tensor = flat_param
        # NOTE: Since this method is called in the pre-unshard, which is only
        # called during computation in the pre-forward or pre-backward, the
        # sharded gradient should be guaranteed to be in `.grad`, not in
        # `._saved_grad_shard`.
        flat_param_grad = (
            flat_param.grad
            if self.uses_sharded_strategy or not self._offload_params
            else flat_param._cpu_grad
        )
        for i, (
            param,
            (in_shard, offset_in_shard, numel_in_shard, _, _),
            (param_name, module, _),
        ) in enumerate(
            zip(
                flat_param._params,
                flat_param._shard_param_infos,
                flat_param._param_infos,
            )
        ):
            if not in_shard:
                continue
            if not hasattr(module, param_name):
                # Do not writeback if original parameters are deregistered
                # (e.g. during model checkpointing)
                continue

            # Check for parameter writeback
            if self._skipped_use_sharded_views:
                param = flat_param._tensors[i]
                _p_assert(
                    param is not None,
                    f"Expects to have saved tensor for {flat_param._fqns[i]}",
                )
            param_changed = getattr(module, param_name) is not param
            needs_param_writeback = (
                param_changed  # changed parameter variable itself
                or not _same_storage(param, flat_param_tensor)
            )
            if self._skipped_use_sharded_views and (
                param_changed or needs_param_writeback
            ):
                raise AssertionError(
                    "FSDP does not support changing the parameters between "
                    f"forward and backward for {self._sharding_strategy}"
                )
            if param_changed:
                # NOTE: The gradient is not preserved after a parameter change.
                param = getattr(module, param_name)
                flat_param._params[i] = param
            if needs_param_writeback:
                expected_shape = torch.Size([numel_in_shard])
                src = param if self.uses_sharded_strategy else param.view(-1)
                self._writeback_tensor(
                    src, flat_param, i, expected_shape, offset_in_shard, True
                )
                wroteback = True

            # Check for gradient writeback
            if self._skipped_use_sharded_views:
                # Skip the writeback check because we do not expose gradients
                # when we skipped using sharded views
                continue
            if param.grad is None and flat_param.grad is not None:
                expected_shape = torch.Size([numel_in_shard])
                self._writeback_tensor(
                    None, flat_param.grad, i, expected_shape, offset_in_shard, False
                )
            elif param.grad is not None:
                # For `NO_SHARD` + CPU offloading, `_cpu_grad` is always in
                # memory and owns the gradient storage, so it will never
                # require gradient writeback.
                if not self.uses_sharded_strategy and self._offload_params:
                    # Explicitly continue to handle the case of `no_sync()`,
                    # where `param.grad` is a view into the GPU gradient
                    # referenced by `flat_param.grad`, while `flat_param_grad`
                    # is `flat_param._cpu_grad`, which is on CPU
                    continue

                needs_grad_writeback = flat_param_grad is None or not _same_storage(
                    param.grad, flat_param_grad
                )
                if needs_grad_writeback:
                    if flat_param_grad is None:
                        flat_param_grad = torch.zeros_like(flat_param)
                    expected_shape = torch.Size([numel_in_shard])
                    src = (
                        param.grad
                        if self.uses_sharded_strategy
                        else param.grad.view(-1)
                    )
                    self._writeback_tensor(
                        src,
                        flat_param_grad,
                        i,
                        expected_shape,
                        offset_in_shard,
                        False,
                    )
                    flat_param.grad = flat_param_grad
                    flat_param_grad = flat_param.grad

        # TODO: If we want to handle shared parameters, we need to re-generate
        # the shared parameter data structures in case sharedness changed.
        for (
            param_name,
            module,
            _,
            prim_param_name,
            prim_module,
            _,
        ) in flat_param._shared_param_infos:
            if getattr(module, param_name) is not getattr(prim_module, prim_param_name):
                raise NotImplementedError(
                    "Changing shared parameters is not supported yet"
                )
        return wroteback

    def _writeback_tensor(
        self,
        src_tensor: Tensor | None,
        dst_tensor: Tensor,
        tensor_index: int,
        expected_shape: torch.Size,
        offset: int,
        is_param: bool,  # else gradient
    ) -> None:
        """
        Write back ``src_tensor`` to ``dst_tensor`` at offset ``offset``, where ``src_tensor`` should have shape ``expected_shape``.

        ``is_param`` indicates if the tensor is the parameter (if ``True``) or gradient (if
        ``False``). If ``src_tensor`` is ``None``, then the effect is zeroing
        instead of copying. ``tensor_index`` gives the index of ``src_tensor``
        in the metadata structures.

        Raises:
            RuntimeError: If the ``src_tensor`` does not have the expected
            shape.
        """
        _p_assert(
            len(expected_shape) == 1,
            f"Expects a 1D expected shape but got {expected_shape}",
        )
        if self._debug_level == dist.DebugLevel.INFO:
            rank = self.rank if hasattr(self, "rank") else dist.get_rank()
            src_shape = src_tensor.shape if src_tensor is not None else None
            src_device = src_tensor.device if src_tensor is not None else None
            warnings.warn(
                f"[Rank {rank}] {'Parameter' if is_param else 'Gradient'} needs "
                f"writeback in {self._training_state}\n"
                f"expected shape={expected_shape} shape={src_shape} "
                f"expected device={dst_tensor.device} device={src_device}",
                stacklevel=2,
            )
        if src_tensor is not None and src_tensor.shape != expected_shape:
            # NOTE: Gradient shape mismatch is not possible in practice since
            # the gradient shape is enforced to match that of the parameter and
            # we already check for parameter shape mismatch.
            raise RuntimeError(
                f"Cannot writeback when the {'parameter' if is_param else 'gradient'} "
                f"shape changes\nExpects {expected_shape} but got {src_tensor.shape}"
            )
        if src_tensor is not None:
            dst_tensor[offset : offset + expected_shape.numel()].copy_(src_tensor)
        else:
            dst_tensor[offset : offset + expected_shape.numel()].zero_()
            if self.flat_param._is_grad_none_mask is None:
                raise AssertionError("Expected _is_grad_none_mask to be not None")
            self.flat_param._is_grad_none_mask[tensor_index] = True

    def _reset_flat_param_grad_info_if_needed(self):
        """
        Reset ``flat_param.grad`` if needed.

        When ``use_orig_params=True``:
        (1) sets the underlying ``flat_param.grad`` to ``None`` if *all* of the
        original parameters' ``.grad`` are ``None``, and
        (2) sets ``flat_param.requires_grad=False`` if *none* of the original
        parameters require gradient.
        For (1), this is targeting ``optim.zero_grad(set_to_none=True)``, in
        which case we want to free the gradients as soon after the
        ``zero_grad()`` call as possible.
        """
        if not self._use_orig_params:
            return
        flat_param = self.flat_param
        if flat_param._params is None:
            raise AssertionError("Expected _params to be not None")  # mypy
        all_grad_none = True
        requires_grad = False
        for param in flat_param._params:
            all_grad_none &= param.grad is None
            requires_grad |= param.requires_grad
        if all_grad_none:
            flat_param.grad = None
        # As long as one parameter requires gradient, then the flat parameter
        # must require gradient
        flat_param.requires_grad = requires_grad

    def _deregister_orig_params(self):
        for param_info in self.flat_param._param_infos:
            param_name, module, _ = param_info
            if hasattr(module, param_name):
                delattr(module, param_name)
        for param_name, module, _, _, _, _ in self.flat_param._shared_param_infos:
            if hasattr(module, param_name):
                delattr(module, param_name)

    ###########
    # HELPERS #
    ###########
    def flat_param_to(self, *args, **kwargs):
        """Wrap an in-place call to ``.to()`` for ``self.flat_param``."""

        self.flat_param.data = self.flat_param.to(*args, **kwargs)
        if self._use_orig_params:
            # Refresh the views because their storage may have changed
            if self.is_sharded(self.flat_param):
                self._use_sharded_views()
            else:
                self._use_unsharded_views(as_params=True)

    def _get_modules(self) -> set[nn.Module]:
        """Return a :class:`set` of the modules whose parameters are included in this handle's flat parameter."""
        return {pi.module for pi in self.flat_param._param_infos}.union(
            {spi.module for spi in self.flat_param._shared_param_infos}
        )

    def is_sharded(self, tensor: Tensor) -> bool:
        """
        Return whether ``tensor`` is *currently* sharded.

        For ``NO_SHARD``, we choose to have this always return ``False`` for clarity.
        """
        if (
            not hasattr(self.flat_param, "_sharded_size")
            or not self.uses_sharded_strategy
        ):
            # `_sharded_size` is defined iff `handle.shard()` has been called
            return False
        sharded_size = self.flat_param._sharded_size  # type: ignore[attr-defined]
        return tensor.size() == sharded_size

    def param_module_names(self) -> Iterator[tuple[str, str]]:
        shared_param_infos = [
            ParamInfo(param_name, module, module_name)
            for (
                param_name,
                module,
                module_name,
                _,
                _,
                _,
            ) in self.flat_param._shared_param_infos
        ]
        for param_info in chain(self.flat_param._param_infos, shared_param_infos):
            param_name, _, module_name = param_info  # type: ignore[misc]
            yield (param_name, module_name)

    def shared_param_module_names(self) -> Iterator[tuple[str, str]]:
        for param_name, _, module_name in [
            ParamInfo(param_name, module, module_name)
            for (
                param_name,
                module,
                module_name,
                _,
                _,
                _,
            ) in self.flat_param._shared_param_infos
        ]:
            yield (param_name, module_name)

    @property
    def _fqns_in_shard(self) -> list[str]:
        """Return the FQNs of the parameters present in this rank's shard."""
        fqns_in_shard: list[str] = []
        for fqn, shard_param_info in zip(
            self.flat_param._fqns,
            self.flat_param._shard_param_infos,  # type: ignore[attr-defined]
        ):
            if shard_param_info.in_shard:
                fqns_in_shard.append(fqn)
        return fqns_in_shard

    @property
    def sharded_grad(self) -> Tensor | None:
        """Return the handle's sharded gradient."""
        flat_param = self.flat_param
        # Priority for non-`None`: `_cpu_grad` > `_saved_grad_shard` > `grad`
        # - CPU offloading: `_cpu_grad`
        # - No CPU offloading + sharded strategies: `_saved_grad_shard`
        # - No CPU offloading + `NO_SHARD`: `grad`
        grad: Tensor | None
        if hasattr(flat_param, "_cpu_grad"):
            grad = flat_param._cpu_grad  # type: ignore[attr-defined]
        elif hasattr(flat_param, "_saved_grad_shard"):
            # In the post-backward hook, the sharded gradient is still in
            # `_saved_grad_shard`.
            grad = flat_param._saved_grad_shard  # type: ignore[attr-defined]
        else:
            # If in IDLE or in FORWARD states, then there may be an
            # (accumulated) gradient. If accessed in IDLE, then this should
            # be due to re-registering the original parameters (e.g. in state
            # dict load).
            _p_assert(
                flat_param.grad is None
                or not self.uses_sharded_strategy
                or self._training_state
                in (HandleTrainingState.FORWARD, HandleTrainingState.IDLE),
                "Sharded strategies should use `_cpu_grad` or `_saved_grad_shard` "
                "unless in IDLE or FORWARD",
            )
            grad = flat_param.grad
        return grad

    def _reset_is_grad_none(self) -> None:
        """
        Reset ``_is_grad_none_mask`` as needed.

        This method should only be
        called in the post-backward after gradient computation, in which case
        if a parameter requires gradient, then it will surely receive a
        gradient and we may reset its mask entry to ``False``.
        """
        if not self._use_orig_params:
            return
        _p_assert(
            self._training_state == HandleTrainingState.BACKWARD_POST,
            "Expects to only be called in the post-backward after gradient computation",
        )
        flat_param = self.flat_param
        if flat_param._params is None:
            raise AssertionError("Expected _params to be not None")  # mypy
        for i, param in enumerate(flat_param._params):  # type: ignore[arg-type]
            # As long as the parameter requires gradient, it should receive a
            # meaningful gradient (even if the gradient happens to be zeros)
            if param.requires_grad:
                if flat_param._is_grad_none_mask is None:
                    raise AssertionError(
                        "Expected _is_grad_none_mask to be not None"
                    )  # mypy
                flat_param._is_grad_none_mask[i] = False

    #######################
    # CHECKS & INVARIANTS #
    #######################
    def _check_sharded_strategy(self):
        _p_assert(self.uses_sharded_strategy, "Expects sharded strategy")

    def _check_on_compute_device(self, tensor: Tensor):
        _p_assert(
            tensor.device == self.device,
            f"Expects tensor to be on the compute device {self.device}, was on {tensor.device}",
        )

    def _check_on_cpu(self, tensor: Tensor):
        _p_assert(
            tensor.device == torch.device("cpu"),
            f"Expects tensor to be on CPU but got {tensor.device}",
        )

    @staticmethod
    def _check_storage_freed(tensor: Tensor):
        # Compile does not resize during trace
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            _p_assert(
                _same_storage_size(tensor, 0),
                "Expects storage to be freed but got storage with size > 0",
            )

    @staticmethod
    def _check_storage_allocated(tensor: Tensor):
        _p_assert(_storage_size_allocated(tensor), "Expects storage to be allocated")

    def _check_low_precision_shard(self):
        _p_assert(
            self._uses_param_mixed_precision,
            "Not using low precision for parameters",
        )
        _p_assert(
            getattr(self.flat_param, "_mp_shard", None) is not None,
            "Expects `_mp_shard` to exist",
        )
        device = self.flat_param._mp_shard.device  # type: ignore[attr-defined]
        _p_assert(
            device == self.device,
            f"Expects the low precision shard to be on {self.device} but got {device}",
        )

    def _check_unsharded(self, tensor: Tensor):
        msg_prefix = "Expects tensor to be unsharded "
        _p_assert(tensor is not None, msg_prefix + "but got `None`")
        unsharded_size = self.flat_param._unpadded_unsharded_size
        _p_assert(
            tensor.size() == unsharded_size,
            msg_prefix + f"with size {unsharded_size} but got {tensor.size()}",
        )

    def _check_sharded(self, tensor: Tensor):
        msg_prefix = "Expects tensor to be sharded "
        _p_assert(tensor is not None, msg_prefix + "but got `None`")
        sharded_size = self.flat_param._sharded_size  # type: ignore[attr-defined]
        _p_assert(
            tensor.size() == sharded_size,
            msg_prefix + f"with size {sharded_size} but got {tensor.size()}",
        )

    ##############
    # PROPERTIES #
    ##############
    @property
    def uses_sharded_strategy(self) -> bool:
        return self._sharding_strategy != HandleShardingStrategy.NO_SHARD

    @property
    def _uses_param_mixed_precision(self) -> bool:
        return self._fwd_bwd_param_dtype != self._orig_param_dtype

    @property
    def _uses_reduce_mixed_precision(self) -> bool:
        return self._reduce_dtype != self._orig_param_dtype

    @property
    def _force_full_precision(self) -> bool:
        return (
            self._uses_param_mixed_precision or self._uses_reduce_mixed_precision
        ) and (
            self._training_state == HandleTrainingState.SUMMON_FULL_PARAMS
            or
            # Also disable mixed precision in model eval mode, if configured
            (not self._fully_sharded_module.training and self._use_full_prec_in_eval)
        )

    @property
    def _skipped_use_sharded_views(self) -> bool:
        """
        This property is used for sharding strategies that do not free after forward with ``use_orig_params=True``.

        This returns if this handle is
        currently in a state where it has skipped using sharded views, in which
        case it can restore view invariants via ``_use_sharded_views()``.
        """
        return self._unsharded_flat_param_for_skipped_views is not None


# NOTE: These are hacks to bypass `nn.Module.__setattr__` checks.
def _unsafe_setattr_param(
    module: nn.Module, param_name: str, param: nn.Parameter
) -> None:
    module._parameters[param_name] = param
    # This bypasses any overrides in case `module` is an instance of an
    # `nn.Module` subclass
    super(nn.Module, module).__setattr__(param_name, param)


def _unsafe_setattr_tensor(module: nn.Module, param_name: str, tensor: Tensor) -> None:
    module._parameters.pop(param_name, None)
    # This bypasses any overrides in case `module` is an instance of an
    # `nn.Module` subclass
    super(nn.Module, module).__setattr__(param_name, tensor)


def _safe_setattr_tensor_or_param(
    module: nn.Module, param_name: str, tensor_or_param: Tensor | nn.Parameter
):
    # Call `delattr()` and `setattr()` to go through `nn.Module` checks
    if hasattr(module, param_name):
        delattr(module, param_name)
    setattr(module, param_name, tensor_or_param)


def _convert_to_params(
    tensors: list[torch.Tensor | nn.Parameter],
) -> list[nn.Parameter]:
    return [t if isinstance(t, nn.Parameter) else nn.Parameter(t) for t in tensors]


def _is_truly_contiguous(x: Tensor) -> bool:
    # Special case: Pytorch thinks that 1x1 channels_last convolution weights are
    # both contiguous and channels_last contiguous at the same time.
    # CuDNN does not agree though and refuses to select faster kernels.
    # It is the reason of having the extra check here.
    return x.stride(-1) == 1 and x.is_contiguous()


def _detach_if_needed(param_or_tensor: nn.Parameter | Tensor) -> Tensor:
    return (
        param_or_tensor.detach()
        if isinstance(param_or_tensor, nn.Parameter)
        else param_or_tensor
    )


def _get_aligned_numel(unsharded_dtype: torch.dtype):
    # NOTE: This alignment constraint comes from TorchInductor.
    ALIGNMENT = 16  # bytes
    unsharded_dtype_size = _get_dtype_size(unsharded_dtype)
    aligned_numel = ALIGNMENT // unsharded_dtype_size
    return aligned_numel


@functools.lru_cache(8)
def _get_dtype_size(dtype):
    return torch.empty((), dtype=dtype).element_size()


def _construct_padding_tensor(
    padding_numel: int, dtype: torch.dtype, requires_grad: bool, device: torch.device
):
    # NOTE: Set the padding value as a magic number for debuggability. The
    # value itself should never be used in any user-facing computation.
    return (
        torch.ones(
            (padding_numel,), dtype=dtype, requires_grad=requires_grad, device=device
        )
        * _FLAT_PARAM_PADDING_VALUE
    )


# Use `lru_cache(1)` to only log the warning once (assuming the fixed warning
# message is passed in)
@functools.lru_cache(1)
def _warn_skip_writeback_check(log: logging.Logger, warning: str):
    logger.warning(warning)


# Use `lru_cache(1)` to only log the warning once
@functools.lru_cache(1)
def _warn_use_fake_all_gather(log: logging.Logger, warning: str):
    logger.warning(warning)


# Use `lru_cache(1)` to only log the warning once
@functools.lru_cache(1)
def _warn_use_fake_reduce(log: logging.Logger, warning: str):
    logger.warning(warning)


def _same_storage(a, b):
    # Params are DTensors in backward
    # with SHARD_GRAD_OP + TP
    from torch.distributed.tensor import DTensor

    if isinstance(a, DTensor):
        a = a._local_tensor
    if isinstance(b, DTensor):
        b = b._local_tensor
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def _same_storage_size(a: torch.Tensor, b: int):
    return a.untyped_storage().size() // a.element_size() == b


def _storage_size_allocated(tensor: Tensor):
    storage_size: int = tensor.untyped_storage().size()
    return storage_size > 0
<|endoftext|>import collections

import torch


class _FreeEventQueue:
    """
    This tracks all pending frees corresponding to inflight all-gathers. The
    queueing pattern is iterative enqueues with a single dequeue per iteration
    once the limit ``_max_num_inflight_all_gathers`` is reached.
    """

    def __init__(self) -> None:
        self._queue: collections.deque[torch.Event] = collections.deque()
        self._max_num_inflight_all_gathers = 2  # empirically chosen

    def enqueue(self, free_event: torch.Event) -> None:
        """Enqueues a free event."""
        self._queue.append(free_event)

    def dequeue_if_needed(self) -> torch.Event | None:
        """Dequeues a single event if the limit is reached."""
        if len(self._queue) >= self._max_num_inflight_all_gathers:
            return self._dequeue()
        return None

    def _dequeue(self) -> torch.Event | None:
        """Dequeues a free event if possible."""
        if self._queue:
            event = self._queue.popleft()
            return event
        return None
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterable, Sequence
from typing import Any, cast

import torch.nn as nn


__all__ = [
    "always_wrap_policy",
    "lambda_auto_wrap_policy",
    "transformer_auto_wrap_policy",
    "size_based_auto_wrap_policy",
    "enable_wrap",
    "wrap",
    "CustomPolicy",
    "ModuleWrapPolicy",
]


# NOTE: We intentionally keep this function simple and isolate the complexity
# to `fn` to enable using this function generically. We may move this to a
# non-FSDP-specific folder and/or make it public in the future.
def _post_order_apply(
    root_module: nn.Module,
    fn: Callable[[nn.Module], nn.Module | None],
):
    """
    This applies ``fn`` to every module in the module tree of ``root_module``
    following a post-order traversal. If ``fn`` returns an :class:`nn.Module`,
    then this replaces the original module with the newly returned one in the
    tree. Otherwise, ``fn`` should return ``None``, in which case the module is
    not changed.
    """
    # Track visited modules to avoid visiting shared modules multiple times
    visited_modules: set[nn.Module] = {root_module}

    def _post_order_apply_inner(
        module: nn.Module,
        module_name: str,
        parent_module: nn.Module | None,
    ):
        for child_module_name, child_module in module.named_children():
            if child_module not in visited_modules:
                visited_modules.add(child_module)
                _post_order_apply_inner(child_module, child_module_name, module)
        optional_module = fn(module)
        if optional_module is not None:
            if not isinstance(parent_module, nn.Module):
                raise AssertionError(
                    "Non-root modules should have their parent module set but got "
                    f"{parent_module} for {module}"
                )
            if not module_name:
                raise AssertionError(
                    "Non-root modules should have their module name set but got "
                    f"an empty module name for {module}"
                )
            if not isinstance(optional_module, nn.Module):
                raise AssertionError(
                    f"fn should return None or an nn.Module but got {optional_module}"
                )
            setattr(parent_module, module_name, optional_module)

    _post_order_apply_inner(root_module, "", None)


def _construct_wrap_fn(
    root_module: nn.Module,
    target_module_to_kwargs: dict[nn.Module, dict[str, Any]],
    fsdp_fn: Callable,
) -> Callable[[nn.Module], nn.Module | None]:
    """
    This constructs the "wrap" function to pass to :func:`_post_order_apply`
    based on ``target_module_to_kwargs``, which should be constructed from the
    wrapping policy.
    """

    def fn(module: nn.Module) -> nn.Module | None:
        # Explicitly avoid wrapping the root module since for FSDP, it is
        # handled by the caller
        if module in target_module_to_kwargs and module is not root_module:
            kwargs = target_module_to_kwargs[module]
            return fsdp_fn(module, **kwargs)
        return None

    return fn


def _run_mixed_precision_override_policy(
    root_module: nn.Module,
    module_classes: Iterable[type[nn.Module]],
    ignored_modules: set[nn.Module],
    root_kwargs: dict[str, Any],
    target_module_to_kwargs: dict[nn.Module, dict[str, Any]],
):
    module_classes_tuple = tuple(set(module_classes))
    for module in root_module.modules():
        if module in ignored_modules:
            continue
        elif isinstance(module, module_classes_tuple):
            # This policy overrides any existing policy
            if module not in target_module_to_kwargs:
                # Only inherit from the root kwargs if not already specified
                target_module_to_kwargs[module] = root_kwargs
            target_module_to_kwargs[module]["mixed_precision"] = None
    return target_module_to_kwargs


def always_wrap_policy(*args, **kwargs) -> bool:
    """
    A simple recursive wrap policy that always returns ``True``. This means
    that every submodule is wrapped by the wrapper class in
    :func:`_recursive_wrap`.
    """
    return True


class _Policy(ABC):
    """
    This defines an abstract base class that represents a policy for applying
    a module-level API.
    """

    @abstractmethod
    def _run_policy(
        self,
        root_module: nn.Module,
        ignored_modules: set[nn.Module],
        root_kwargs: dict[str, Any],
    ) -> dict[nn.Module, dict[str, Any]]:
        """
        This should return a dict ``target_module_to_kwargs`` that maps from
        each target module to wrap to its kwargs.
        """
        ...


def _module_wrap_policy(
    module: nn.Module,
    recurse: bool,
    nonwrapped_numel: int,
    module_classes: set[type[nn.Module]],
) -> bool:
    """
    This auto wrap policy wraps every module that is an instance of any type in
    ``module_classes`` as its own FSDP instance. The root module given by
    ``module`` is always wrapped as an FSDP instance regardless. Since the
    wrapping proceeds bottom up, each FSDP instance manages the parameters in
    its subtree excluding any already managed by a child FSDP instance.

    Args:
        module (nn.Module): Current module being considered.
        recurse (bool): If ``False``, then this function must decide whether
            ``module`` should be wrapped as an FSDP instance or not. If
            ``True``, then the function is still recursing down the module
            tree as a part of the DFS.
        nonwrapped_numel (int): Parameter numel not yet wrapped.
        module_classes (Set[Type[nn.Module]]): Set of module classes that are
            wrapped as FSDP instances.

    Returns:
        ``True`` if ``recurse=True``, and whether ``module`` should be wrapped
        if ``recurse=False``.
    """
    if recurse:
        return True  # always recurse
    return isinstance(module, tuple(module_classes))


class ModuleWrapPolicy(_Policy):
    """
    This policy applies to every module of the specified module classes,
    passing in the kwargs given to the root.
    """

    def __init__(self, module_classes: Iterable[type[nn.Module]]):
        module_classes_set = set(module_classes)
        self._module_classes = module_classes_set
        self._module_classes_str = str(module_classes_set)

    def _run_policy(
        self,
        root_module: nn.Module,
        ignored_modules: set[nn.Module],
        root_kwargs: dict[str, Any],
    ) -> dict[nn.Module, dict[str, Any]]:
        module_classes = tuple(self._module_classes)
        target_module_to_kwargs: dict[nn.Module, dict[str, Any]] = {}
        for module in root_module.modules():
            if module in ignored_modules:
                continue
            elif isinstance(module, module_classes):
                # Shallow copy to avoid coupling changes across modules
                target_module_to_kwargs[module] = copy.copy(root_kwargs)
        return target_module_to_kwargs

    def __call__(self, module, recurse, *args, **kwargs):
        # nonwrapped_numel is not used.
        return _module_wrap_policy(
            module, recurse, nonwrapped_numel=-1, module_classes=self._module_classes
        )

    def __repr__(self) -> str:
        return super().__repr__() + f"({self._module_classes_str})"


class CustomPolicy(_Policy):
    """
    This policy takes in a lambda function that maps a given ``nn.Module`` to
    either ``False``, ``True``, or a kwarg dictionary.
    - If the function returns ``False`` or an empty dictionary, then the module
      does not have the API applied.
    - If the function returns ``True``, then the module has the API applied
      with the root's kwargs.
    - If the function returns a non-empty dictionary, then the module has the
      API applied, and the dictionary overrides the root's kwargs.

    Example::

        >>> # xdoctest: +SKIP("undefined variables")
        >>> model = init_transformer_model(...)
        >>> def lambda_fn(module: nn.Module):
        >>>     if module is model.lm_head:
        >>>         return {"sharding_strategy": ShardingStrategy.SHARD_GRAD_OP}
        >>>     elif isinstance(module, TransformerBlock):
        >>>         return True
        >>>     return False
        >>> policy = CustomPolicy(lambda_fn)
        >>> fsdp_model = FSDP(model, auto_wrap_policy=policy)
    """

    def __init__(self, lambda_fn: Callable[[nn.Module], bool | dict[str, Any]]):
        self._lambda_fn = lambda_fn

    def _run_policy(
        self,
        root_module: nn.Module,
        ignored_modules: set[nn.Module],
        root_kwargs: dict[str, Any],
    ) -> dict[nn.Module, dict[str, Any]]:
        target_module_to_kwargs: dict[nn.Module, dict[str, Any]] = {}
        for module in root_module.modules():
            if module in ignored_modules:
                continue
            res = self._lambda_fn(module)
            if not isinstance(res, (dict, bool)):
                raise ValueError(
                    "The lambda_fn passed to CustomPolicy should return "
                    f"False/True or a kwarg dict, but it returned {res}"
                )
            if not res:
                continue
            kwargs = copy.copy(root_kwargs)
            if isinstance(res, dict):
                # Override the root kwargs with the ones specified by the
                # lambda function
                kwargs.update(res)
            target_module_to_kwargs[module] = kwargs
        return target_module_to_kwargs


def lambda_auto_wrap_policy(
    module: nn.Module, recurse: bool, nonwrapped_numel: int, lambda_fn: Callable
) -> bool:
    """
    A convenient auto wrap policy to wrap submodules based on an arbitrary user
    function. If `lambda_fn(submodule) == True``, the submodule will be wrapped as
    a `wrapper_cls` unit.

    Return if a module should be wrapped during auto wrapping.

    The first three parameters are required by :func:`_recursive_wrap`.

    Args:
        module (nn.Module): Current module being considered.
        recurse (bool): If ``False``, then this function must decide whether
            ``module`` should be wrapped as an FSDP instance or not. If
            ``True``, then the function is still recursing down the module
            tree as a part of the DFS.
        nonwrapped_numel (int): Parameter numel not yet wrapped.

        lambda_fn (Callable[[nn.Module], bool]): If this returns ``True``, then
            this module will be wrapped.
    """
    if recurse:
        return True  # always recurse
    return lambda_fn(module)


def transformer_auto_wrap_policy(
    module: nn.Module,
    recurse: bool,
    nonwrapped_numel: int,
    transformer_layer_cls: set[type[nn.Module]],
) -> bool:
    """
    See :func:`_module_wrap_policy`, where ``transformer_layer_cls`` is the
    same as ``module_classes``. Note that shared parameters must be wrapped in
    the same FSDP instance, so this auto wrap policy can help wrap shared
    embeddings into the same FSDP instance for transformer models.
    """
    return _module_wrap_policy(module, recurse, nonwrapped_numel, transformer_layer_cls)


def _wrap_module_cls_individually(
    module: nn.Module, module_classes: Sequence[type], recurse: bool, *args, **kwargs
):
    if recurse:
        # always recurse
        return True
    else:
        # if not recursing, decide whether we should wrap based on whether the type of module
        # is in `module_classes`.
        return isinstance(module, tuple(module_classes))


def _or_policy(
    module: nn.Module,
    recurse: bool,
    nonwrapped_numel: int,
    policies,
) -> bool:
    """
    A policy that wraps ``module`` if any policy in the passed in iterable of
    ``policies`` returns ``True``.
    """
    return any(
        policy(module=module, recurse=recurse, nonwrapped_numel=nonwrapped_numel)
        for policy in policies
    )


def size_based_auto_wrap_policy(
    module: nn.Module,
    recurse: bool,
    nonwrapped_numel: int,
    # Additional custom arguments
    min_num_params: int = int(1e8),
    force_leaf_modules: set[type[nn.Module]] | None = None,
    exclude_wrap_modules: set[type[nn.Module]] | None = None,
) -> bool:
    """
    A size-based auto wrap policy.

    Args:
        module (nn.Module): Current module being considered.
        recurse (bool): If ``False``, then this function must decide whether
            ``module`` should be wrapped as an FSDP instance or not. If
            ``True``, then the function is still recursing down the module
            tree as a part of the DFS.
        nonwrapped_numel (int): Parameter numel not yet wrapped.

        min_num_params (int): Customizable policy input that controls the size
            threshold over which a module is ready to be wrapped. This is in
            units of numel.
        force_leaf_modules (Optional[set[type[nn.Module]]]): Set of module types to keep
            as leaves, i.e. their children will never be wrapped.
        exclude_wrap_modules (Optional[set[type[nn.Module]]]): Set of module types to be
            excluded in wrapping.

    Returns:
        Whether ``module`` should be wrapped.
    """
    force_leaf_modules = (
        size_based_auto_wrap_policy.FORCE_LEAF_MODULES  # type: ignore[attr-defined]
        if force_leaf_modules is None
        else force_leaf_modules
    )
    exclude_wrap_modules = (
        size_based_auto_wrap_policy.EXCLUDE_WRAP_MODULES  # type: ignore[attr-defined]
        if exclude_wrap_modules is None
        else exclude_wrap_modules
    )

    # Keep the argument `min_num_params` for BC for now, but it represents the
    # minimum non-wrapped *numel* before triggering a wrapping
    min_nonwrapped_numel = min_num_params
    is_large = nonwrapped_numel >= min_nonwrapped_numel
    if recurse:
        # We should recurse if the module is big enough but not in force_leaf_modules list.
        return is_large and not isinstance(module, tuple(force_leaf_modules))
    else:
        # If we are not recursing, determine if we should wrap.
        return is_large and not isinstance(module, tuple(exclude_wrap_modules))


# Set those defaults to the size_based_auto_wrap_policy function. Make them easy to be imported.
size_based_auto_wrap_policy.EXCLUDE_WRAP_MODULES = {nn.ModuleList, nn.ModuleDict}  # type: ignore[attr-defined]
size_based_auto_wrap_policy.FORCE_LEAF_MODULES = {nn.MultiheadAttention}  # type: ignore[attr-defined]


@contextlib.contextmanager
def enable_wrap(
    *, wrapper_cls: Any, **wrapper_kwargs: Any
) -> Generator[None, None, None]:
    """
    Context manager to wrap modules using a wrapper.

    Useful for when you'd like to apply the same configuration arguments to all
    child modules that you wrap. A particularly important use case is wrapping
    large layers so that they get sharded (in-place) during initialization, to
    avoid running out of system memory. Large layers can indicate that they
    should be sharded via the ``wrap`` annotation and this context manager can
    provide the exact configuration for these nested instances.

    Usage::

        with enable_wrap(wrapper_cls, **params):
            # Wraps layer in FSDP by default if within context
            self.l1 = wrap(torch.nn.Linear(5, 5))

    Args:
        wrapper_cls:
            Class that `wrap` annotation will `wrap` modules with, such as
            `FullyShardedDataParallel`.
        **wrapper_kwargs:
            Configuration settings that will be passed to all ``wrap``
            instances inside the context
    """
    kwargs = {
        "wrapper_cls": wrapper_cls,
        **wrapper_kwargs,
    }
    with _ConfigAutoWrap(**kwargs):
        yield


def wrap(module: nn.Module, **wrap_overrides: Any) -> nn.Module:
    """
    Annotate that a module should be wrapped. Annotated modules will only be
    wrapped if inside of an :func:`enable_wrap` context manager. This allows
    a module to be initialized both with and without a wrapper without code
    change.

    The class that this function wraps the passed in ``nn.Module`` with is the
    passed in ``wrapper_cls`` argument into ``enable_wrap``. Both
    ``enable_wrap`` and ``wrap`` can take in kwargs specifying how to construct
    the ``wrapper_cls`` instance. In the case of duplicate kwargs in
    ``enable_wrap`` and ``wrap``, the argument passed into ``wrap`` will be
    respected.

    Usage::

        with enable_wrap(wrapper_cls=FSDP, **fsdp_config):
            # Wraps layer in FSDP by default if within context
            self.l1 = wrap(torch.nn.Linear(5, 5))

    Args:
        module (nn.Module): module to wrap (if in :func:`enable_wrap` context)
        **wrap_overrides: configuration overrides that will take priority over
            the values provided by the :func:`enable_wrap` context
    """
    if _ConfigAutoWrap.in_autowrap_context:
        if _ConfigAutoWrap.wrapper_cls is None:
            raise AssertionError("Expected _ConfigAutoWrap.wrapper_cls to be set")

        wrap_overrides = {**_ConfigAutoWrap.kwargs, **wrap_overrides}
        return _wrap(
            module,
            _ConfigAutoWrap.wrapper_cls,
            **wrap_overrides,
        )
    return module


def _wrap(module: nn.Module, wrapper_cls: Callable, **kwargs) -> nn.Module:
    if wrapper_cls is None:
        raise AssertionError("Expected wrapper_cls to be set")
    if hasattr(module, "_wrap_overrides"):
        # If module has a _wrap_overrides attribute, we force overriding the
        # FSDP config with these attributes for this module. Currently this
        # is only used to disable mixed precision for BatchNorm when
        # auto_wrapping.
        overrides = {**kwargs, **module._wrap_overrides}  # type: ignore[arg-type, dict-item]
        return wrapper_cls(module, **overrides)

    return wrapper_cls(module, **kwargs)


def _recursive_wrap(
    module: nn.Module,
    auto_wrap_policy: Callable,
    wrapper_cls: Callable,
    ignored_modules: set[nn.Module],
    ignored_params: set[nn.Parameter],
    only_wrap_children: bool = False,
    **kwargs: Any,
) -> tuple[nn.Module, int]:
    """
    Wraps submodules of ``module`` for which ``auto_wrap_policy`` returns
    ``True`` with ``wrapper_cls``.

    Args:
        module (nn.Module): Module to recursively wrap.
        auto_wrap_policy (Callable): A callable representing a policy that
            determines which modules to recursively wrap with ``wrapper_cls``.
        ignored_modules (set[torch.nn.Module]): Modules to ignore when
            wrapping.
        ignored_params (set[torch.nn.Parameter]): Parameters to ignore when
            wrapping; these should be the parameters contained in the modules
            in ``ignored_modules``.
    Returns:
        (nn.Module, int):
            ``module`` after wrapping and the numel recursively wrapped.
    """
    if auto_wrap_policy is None:
        raise AssertionError("Must specify auto_wrap_policy.")
    if wrapper_cls is None:
        raise AssertionError("Must specify wrapper_cls")
    # Make sure no child is already wrapped.
    for _, child in module.named_modules():
        if child in ignored_modules:
            continue
        try:
            if isinstance(child, cast(type, wrapper_cls)):
                raise AssertionError(
                    f"Child module {child} is already wrapped by {wrapper_cls}"
                )
        except TypeError:
            # wrapper_cls is a function as opposed to a class type, just bypass above check.
            pass

    # We count all params, assuming none of them are already wrapped.
    nonwrapped_numel = sum(
        p.numel() for p in module.parameters() if p not in ignored_params
    )

    if auto_wrap_policy is None:
        raise AssertionError("Expected auto_wrap_policy to be set")
    if auto_wrap_policy(module=module, recurse=True, nonwrapped_numel=nonwrapped_numel):
        total_wrapped_numel = 0
        # Iterate through the children, recursively wrap if necessary
        for name, child in module.named_children():
            if child in ignored_modules:
                continue
            wrapped_child, num_wrapped_params = _recursive_wrap(
                module=child,
                auto_wrap_policy=auto_wrap_policy,
                wrapper_cls=wrapper_cls,
                ignored_modules=ignored_modules,
                ignored_params=ignored_params,
                **kwargs,
            )
            setattr(module, name, wrapped_child)
            # Keep track of how many parameters have been wrapped
            total_wrapped_numel += num_wrapped_params
        # decide if we need to wrap the current module,
        # since the left over parameters exceed the number of params to wrap
        remainder = nonwrapped_numel - total_wrapped_numel
        if not only_wrap_children and auto_wrap_policy(
            module=module, recurse=False, nonwrapped_numel=remainder
        ):
            # Leaf node or final wrapping of the remainder both happen here.
            return _wrap(module, wrapper_cls, **kwargs), nonwrapped_numel
        else:
            return module, total_wrapped_numel
    return module, 0


class _ConfigAutoWrap:
    """
    Helper class to wrap modules based on default config args via a context manager.
    See :func:`enable_wrap` for more information.
    """

    in_autowrap_context: bool = False  # Context flag
    wrapper_cls: Callable | None = None  # The wrapper class
    kwargs: dict[str, Any] = {}  # Wrapper's args

    def __init__(self, **kwargs: dict[str, Any]):
        self.kwargs = kwargs

    @staticmethod
    def enable_autowrap_context(kwargs: Any) -> None:
        if _ConfigAutoWrap.in_autowrap_context:
            raise NotImplementedError(
                "You are already within an autowrap context and we currently do not supported nested autowrap."
            )
        _ConfigAutoWrap.in_autowrap_context = True
        # Get and save the wrapper cls for the context.
        if "wrapper_cls" not in kwargs:
            raise AssertionError(
                "Expected to pass in wrapper_cls arg into _ConfigAutoWrap."
            )
        _ConfigAutoWrap.wrapper_cls = cast(Callable, kwargs["wrapper_cls"])
        del kwargs["wrapper_cls"]
        # Save the rest.
        _ConfigAutoWrap.kwargs = kwargs

    @staticmethod
    def disable_autowrap_context() -> None:
        _ConfigAutoWrap.in_autowrap_context = False
        _ConfigAutoWrap.wrapper_cls = None
        _ConfigAutoWrap.kwargs = {}

    def __enter__(self) -> None:
        self.enable_autowrap_context(self.kwargs)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.disable_autowrap_context()
<|endoftext|># mypy: ignore-errors

import contextlib
import copy
import functools
import math
import traceback
import warnings
from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import contextmanager
from enum import auto, Enum
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    _CHECKPOINT_WRAPPED_MODULE,
    ActivationWrapper,
)
from torch.distributed.algorithms._comm_hooks import LOW_PRECISION_HOOKS
from torch.distributed.fsdp._common_utils import (
    _FSDPState,
    _get_param_to_fqns,
    FSDP_PREFIX,
    FSDP_WRAPPED_MODULE,
    HandleTrainingState,
    TrainingState,
)
from torch.distributed.fsdp._dynamo_utils import _annotate_modules_for_dynamo
from torch.distributed.fsdp._init_utils import (
    _check_orig_params_flattened,
    _init_buffer_state,
    _init_core_state,
    _init_device_handle,
    _init_extension,
    _init_ignored_module_states,
    _init_param_handle_from_module,
    _init_prefetching_state,
    _init_process_group_state,
    _init_runtime_state,
    _init_state_dict_state,
    HYBRID_SHARDING_STRATEGIES,
    ProcessGroupType,
)
from torch.distributed.fsdp._runtime_utils import (
    _get_fsdp_root_states,
    _is_fsdp_root,
    _lazy_init,
    _post_forward,
    _post_forward_reshard,
    _pre_forward,
    _pre_forward_unshard,
    _root_pre_forward,
    _unshard,
    _wait_for_computation_stream,
)
from torch.distributed.fsdp._wrap_utils import _auto_wrap
from torch.distributed.fsdp.api import (
    BackwardPrefetch,
    CPUOffload,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    LocalOptimStateDictConfig,
    LocalStateDictConfig,
    MixedPrecision,
    OptimStateDictConfig,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictConfig,
    StateDictSettings,
    StateDictType,
)
from torch.distributed.tensor import DeviceMesh
from torch.distributed.utils import _p_assert
from torch.utils._typing_utils import copy_method_params, copy_method_sig

from ._flat_param import FlatParameter, FlatParamHandle
from ._optim_utils import (
    _flatten_optim_state_dict,
    _get_param_id_to_param_from_optim_input,
    _get_param_key_to_param,
    _get_param_to_param_id_from_optim_input,
    _get_param_to_param_key,
    _optim_state_dict,
    _rekey_sharded_optim_state_dict,
    _set_optim_use_dtensor,
)
from ._state_dict_utils import _register_all_state_dict_hooks
from ._unshard_param_utils import (
    _deregister_orig_params,
    _register_flat_param,
    _register_orig_params,
    _unshard_params,
    _unshard_params_for_summon,
)
from .wrap import CustomPolicy, ModuleWrapPolicy


__all__ = [
    "FullyShardedDataParallel",
    "OptimStateKeyType",
]


FLAT_PARAM = "_flat_param"


class OptimStateKeyType(Enum):
    """Represents the type of key in an optimizer state-dict."""

    PARAM_NAME = auto()
    PARAM_ID = auto()


class FullyShardedDataParallel(nn.Module, _FSDPState):
    """A wrapper for sharding module parameters across data parallel workers.

    This is inspired by `Xu et al. <https://arxiv.org/abs/2004.13336>`_ as
    well as the ZeRO Stage 3 from `DeepSpeed <https://www.deepspeed.ai/>`_.
    FullyShardedDataParallel is commonly shortened to FSDP.

    Example::

        >>> # xdoctest: +SKIP("undefined variables")
        >>> import torch
        >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        >>> torch.cuda.set_device(device_id)
        >>> sharded_module = FSDP(my_module)
        >>> optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)
        >>> x = sharded_module(x, y=3, z=torch.Tensor([1]))
        >>> loss = x.sum()
        >>> loss.backward()
        >>> optim.step()

    Using FSDP involves wrapping your module and then initializing your
    optimizer after. This is required since FSDP changes the parameter
    variables.

    When setting up FSDP, you need to consider the destination CUDA
    device. If the device has an ID (``dev_id``), you have three options:

    * Place the module on that device
    * Set the device using ``torch.cuda.set_device(dev_id)``
    * Pass ``dev_id`` into the ``device_id`` constructor argument.

    This ensures that the FSDP instance's compute device is the
    destination device. For option 1 and 3, the FSDP initialization
    always occurs on GPU. For option 2, the FSDP initialization
    happens on module's current device, which may be a CPU.

    If you're using the ``sync_module_states=True`` flag, you need to
    ensure that the module is on a GPU or use the ``device_id``
    argument to specify a CUDA device that FSDP will move the module
    to in the FSDP constructor. This is necessary because
    ``sync_module_states=True`` requires GPU communication.

    FSDP also takes care of moving input tensors to the forward method
    to the GPU compute device, so you don't need to manually move them
    from CPU.

    For ``use_orig_params=True``,
    ``ShardingStrategy.SHARD_GRAD_OP`` exposes the unsharded
    parameters, not the sharded parameters after forward, unlike
    ``ShardingStrategy.FULL_SHARD``. If you want
    to inspect the gradients, you can use the ``summon_full_params``
    method with ``with_grads=True``.

    With ``limit_all_gathers=True``, you may see a gap in the FSDP
    pre-forward where the CPU thread is not issuing any kernels. This is
    intentional and shows the rate limiter in effect. Synchronizing the CPU
    thread in that way prevents over-allocating memory for subsequent
    all-gathers, and it should not actually delay GPU kernel execution.

    FSDP replaces managed modules' parameters with ``torch.Tensor``
    views during forward and backward computation for autograd-related
    reasons. If your module's forward relies on saved references to
    the parameters instead of reacquiring the references each
    iteration, then it will not see FSDP's newly created views,
    and autograd will not work correctly.

    Finally, when using ``sharding_strategy=ShardingStrategy.HYBRID_SHARD``
    with the sharding process group being intra-node and the
    replication process group being inter-node, setting
    ``NCCL_CROSS_NIC=1`` can help improve the all-reduce times over
    the replication process group for some cluster setups.

    **Limitations**

    There are several limitations to be aware of when using FSDP:

    * FSDP currently does not support gradient accumulation outside
      ``no_sync()`` when using CPU offloading. This is because FSDP
      uses the newly-reduced gradient instead of accumulating with any
      existing gradient, which can lead to incorrect results.

    * FSDP does not support running the forward pass of a submodule
      that is contained in an FSDP instance. This is because the
      submodule's parameters will be sharded, but the submodule itself
      is not an FSDP instance, so its forward pass will not all-gather
      the full parameters appropriately.

    * FSDP does not work with double backwards due to the way it
      registers backward hooks.

    * FSDP has some constraints when freezing parameters.
      For ``use_orig_params=False``, each FSDP instance must manage
      parameters that are all frozen or all non-frozen. For
      ``use_orig_params=True``, FSDP supports mixing frozen and
      non-frozen parameters, but it's recommended to avoid doing so to
      prevent higher than expected gradient memory usage.

    * As of PyTorch 1.12, FSDP offers limited support for shared
      parameters. If enhanced shared parameter support is needed for
      your use case, please post in
      `this issue <https://github.com/pytorch/pytorch/issues/77724>`__.

    * You should avoid modifying the parameters between forward and
      backward without using the ``summon_full_params`` context, as
      the modifications may not persist.

    Args:
        module (nn.Module):
            This is the module to be wrapped with FSDP.
        process_group (Optional[Union[ProcessGroup, Tuple[ProcessGroup, ProcessGroup]]]):
            This is the process group over which the model is sharded and thus
            the one used for FSDP's all-gather and reduce-scatter collective
            communications. If ``None``, then FSDP uses the default process
            group. For hybrid sharding strategies such as
            ``ShardingStrategy.HYBRID_SHARD``, users can pass in a tuple of
            process groups, representing the groups over which to shard and
            replicate, respectively. If ``None``, then FSDP constructs process
            groups for the user to shard intra-node and replicate inter-node.
            (Default: ``None``)
        sharding_strategy (Optional[ShardingStrategy]):
            This configures the sharding strategy, which may trade off memory
            saving and communication overhead. See :class:`ShardingStrategy`
            for details. (Default: ``FULL_SHARD``)
        cpu_offload (Optional[CPUOffload]):
            This configures CPU offloading. If this is set to ``None``, then
            no CPU offloading happens. See :class:`CPUOffload` for details.
            (Default: ``None``)
        auto_wrap_policy (Optional[Union[Callable[[nn.Module, bool, int], bool], ModuleWrapPolicy, CustomPolicy]]):
            This specifies a policy to apply FSDP to submodules of ``module``,
            which is needed for communication and computation overlap and thus
            affects performance. If ``None``, then FSDP only applies to
            ``module``, and users should manually apply FSDP to parent modules
            themselves (proceeding bottom-up). For convenience, this accepts
            ``ModuleWrapPolicy`` directly, which allows users to specify the
            module classes to wrap (e.g. the transformer block). Otherwise,
            this should be a callable that takes in three arguments
            ``module: nn.Module``, ``recurse: bool``, and
            ``nonwrapped_numel: int`` and should return a ``bool`` specifying
            whether the passed-in ``module`` should have FSDP applied if
            ``recurse=False`` or if the traversal should continue into the
            module's subtree if ``recurse=True``. Users may add additional
            arguments to the callable. The ``size_based_auto_wrap_policy`` in
            ``torch.distributed.fsdp.wrap.py`` gives an example callable that
            applies FSDP to a module if the parameters in its subtree exceed
            100M numel. We recommend printing the model after applying FSDP
            and adjusting as needed.

            Example::

                >>> def custom_auto_wrap_policy(
                >>>     module: nn.Module,
                >>>     recurse: bool,
                >>>     nonwrapped_numel: int,
                >>>     # Additional custom arguments
                >>>     min_num_params: int = int(1e8),
                >>> ) -> bool:
                >>>     return nonwrapped_numel >= min_num_params
                >>> # Configure a custom `min_num_params`
                >>> my_auto_wrap_policy = functools.partial(custom_auto_wrap_policy, min_num_params=int(1e5))

        backward_prefetch (Optional[BackwardPrefetch]):
            This configures explicit backward prefetching of all-gathers. If
            ``None``, then FSDP does not backward prefetch, and there is no
            communication and computation overlap in the backward pass. See
            :class:`BackwardPrefetch` for details. (Default: ``BACKWARD_PRE``)
        mixed_precision (Optional[MixedPrecision]):
            This configures native mixed precision for FSDP. If this is set to
            ``None``, then no mixed precision is used. Otherwise, parameter,
            buffer, and gradient reduction dtypes can be set. See
            :class:`MixedPrecision` for details. (Default: ``None``)
        ignored_modules (Optional[Iterable[torch.nn.Module]]): Modules whose
            own parameters and child modules' parameters and buffers are
            ignored by this instance. None of the modules directly in
            ``ignored_modules`` should be :class:`FullyShardedDataParallel`
            instances, and any child modules that are already-constructed
            :class:`FullyShardedDataParallel` instances will not be ignored if
            they are nested under this instance. This argument may be used to
            avoid sharding specific parameters at module granularity when using an
            ``auto_wrap_policy`` or if parameters' sharding is not managed by
            FSDP. (Default: ``None``)
        param_init_fn (Optional[Callable[[nn.Module], None]]):
            A ``Callable[torch.nn.Module] -> None`` that
            specifies how modules that are currently on the meta device should
            be initialized onto an actual device. As of v1.12, FSDP detects
            modules with parameters or buffers on meta device via ``is_meta``
            and either applies ``param_init_fn`` if specified or calls
            ``nn.Module.reset_parameters()`` otherwise. For both cases, the
            implementation should *only* initialize the parameters/buffers of
            the module, not those of its submodules. This is to avoid
            re-initialization. In addition, FSDP also supports deferred
            initialization via torchdistX's (https://github.com/pytorch/torchdistX)
            ``deferred_init()`` API, where the deferred modules are initialized
            by calling ``param_init_fn`` if specified or torchdistX's default
            ``materialize_module()`` otherwise. If ``param_init_fn`` is
            specified, then it is applied to all meta-device modules, meaning
            that it should probably case on the module type. FSDP calls the
            initialization function before parameter flattening and sharding.

            Example::

                >>> # xdoctest: +SKIP("undefined variables")
                >>> module = MyModule(device="meta")
                >>> def my_init_fn(module: nn.Module):
                >>>     # E.g. initialize depending on the module type
                >>>     ...
                >>> fsdp_model = FSDP(module, param_init_fn=my_init_fn, auto_wrap_policy=size_based_auto_wrap_policy)
                >>> print(next(fsdp_model.parameters()).device) # current CUDA device
                >>> # With torchdistX
                >>> module = deferred_init.deferred_init(MyModule, device="cuda")
                >>> # Will initialize via deferred_init.materialize_module().
                >>> fsdp_model = FSDP(module, auto_wrap_policy=size_based_auto_wrap_policy)

        device_id (Optional[Union[int, torch.device]]): An ``int`` or
            ``torch.device`` giving the CUDA device on which FSDP
            initialization takes place, including the module initialization
            if needed and the parameter sharding. This should be specified to
            improve initialization speed if ``module`` is on CPU. If the
            default CUDA device was set (e.g. via ``torch.cuda.set_device``),
            then the user may pass ``torch.cuda.current_device`` to this.
            (Default: ``None``)
        sync_module_states (bool): If ``True``, then each FSDP module will
            broadcast module parameters and buffers from rank 0 to ensure that
            they are replicated across ranks (adding communication overhead to
            this constructor). This can help load ``state_dict`` checkpoints
            via ``load_state_dict`` in a memory efficient way. See
            :class:`FullStateDictConfig` for an example of this. (Default:
            ``False``)
        forward_prefetch (bool): If ``True``, then FSDP *explicitly* prefetches
            the next forward-pass all-gather before the current forward
            computation. This is only useful for CPU-bound workloads, in which
            case issuing the next all-gather earlier may improve overlap. This
            should only be used for static-graph models since the prefetching
            follows the first iteration's execution order. (Default: ``False``)
        limit_all_gathers (bool): If ``True``, then FSDP explicitly
            synchronizes the CPU thread to ensure GPU memory usage from only
            *two* consecutive FSDP instances (the current instance running
            computation and the next instance whose all-gather is prefetched).
            If ``False``, then FSDP allows the CPU thread to issue all-gathers
            without any extra synchronization. (Default: ``True``) We often
            refer to this feature as the "rate limiter". This flag should only
            be set to ``False`` for specific CPU-bound workloads with low
            memory pressure in which case the CPU thread can aggressively issue
            all kernels without concern for the GPU memory usage.
        use_orig_params (bool): Setting this to ``True`` has FSDP use
            ``module`` 's original parameters. FSDP exposes those original
            parameters to the user via :meth:`nn.Module.named_parameters`
            instead of FSDP's internal :class:`FlatParameter` s. This means
            that the optimizer step runs on the original parameters, enabling
            per-original-parameter hyperparameters. FSDP preserves the original
            parameter variables and manipulates their data between unsharded
            and sharded forms, where they are always views into the underlying
            unsharded or sharded :class:`FlatParameter`, respectively. With the
            current algorithm, the sharded form is always 1D, losing the
            original tensor structure. An original parameter may have all,
            some, or none of its data present for a given rank. In the none
            case, its data will be like a size-0 empty tensor. Users should not
            author programs relying on what data is present for a given
            original parameter in its sharded form. ``True`` is required to
            use ``torch.compile()``. Setting this to ``False`` exposes FSDP's
            internal :class:`FlatParameter` s to the user via
            :meth:`nn.Module.named_parameters`. (Default: ``False``)
        ignored_states (Optional[Iterable[torch.nn.Parameter]], Optional[Iterable[torch.nn.Module]]):
            Ignored parameters or modules that will not be managed by this FSDP
            instance, meaning that the parameters are not sharded and their
            gradients are not reduced across ranks. This argument unifies with
            the existing ``ignored_modules`` argument, and we may deprecate
            ``ignored_modules`` soon. For backward compatibility, we keep both
            ``ignored_states`` and `ignored_modules``, but FSDP only allows one
            of them to be specified as not ``None``.
        device_mesh (Optional[DeviceMesh]): DeviceMesh can be used as an alternative to
            process_group. When device_mesh is passed, FSDP will use the underlying process
            groups for all-gather and reduce-scatter collective communications. Therefore,
            these two args need to be mutually exclusive. For hybrid sharding strategies such as
            ``ShardingStrategy.HYBRID_SHARD``, users can pass in a 2D DeviceMesh instead
            of a tuple of process groups. For 2D FSDP + TP, users are required to pass in
            device_mesh instead of process_group. For more DeviceMesh info, please visit:
            https://pytorch.org/tutorials/recipes/distributed_device_mesh.html
    """

    def __init__(
        self,
        module: nn.Module,
        process_group: ProcessGroupType = None,
        sharding_strategy: ShardingStrategy | None = None,
        cpu_offload: CPUOffload | None = None,
        auto_wrap_policy: Callable | ModuleWrapPolicy | CustomPolicy | None = None,
        backward_prefetch: BackwardPrefetch | None = BackwardPrefetch.BACKWARD_PRE,
        mixed_precision: MixedPrecision | None = None,
        ignored_modules: Iterable[torch.nn.Module] | None = None,
