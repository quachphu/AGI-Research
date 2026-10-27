        """
        response = saver.async_save(
            state_dict,
            storage_writer=self.storage_writer,
            process_group=self.process_group,
            planner=self.save_planner,
        )
        if not isinstance(response, Future):
            raise AssertionError("response should be a Future instance")
        return response

    def load(self, state_dict: dict[str, Any]) -> None:
        """Calls :py:meth: `torch.distributed.state_dict_loader.load`. Utilizing values passed during initialization."""
        loader.load(
            state_dict,
            storage_reader=self.storage_reader,
            process_group=self.process_group,
            planner=self.load_planner,
        )
<|endoftext|>import traceback as tb
from typing import Any


WRAPPED_EXCEPTION = tuple[BaseException, tb.StackSummary]

__all__ = ["CheckpointException"]


def _wrap_exception(exc: BaseException) -> WRAPPED_EXCEPTION:
    summary = tb.extract_tb(exc.__traceback__)
    # Python 3.13+ stores bytecode objects in FrameSummary._code,
    # which cannot be pickled. Clear them so gather_object succeeds
    # and the real exception is reported instead of a misleading
    # "cannot pickle code objects" TypeError.
    for frame in summary:
        if hasattr(frame, "_code"):
            object.__setattr__(frame, "_code", None)
    return (exc, summary)


def _is_wrapped_exception(obj: Any) -> bool:
    if not isinstance(obj, tuple):
        return False
    if len(obj) != 2:
        return False
    return isinstance(obj[0], BaseException) and isinstance(obj[1], tb.StackSummary)


class CheckpointException(BaseException):
    """Exception raised if failure was detected as part of a checkpoint load or save."""

    def __init__(self, msg: str, failures: dict[int, WRAPPED_EXCEPTION]):
        super().__init__(msg, failures)
        self._failures = failures

    @property
    def failures(self) -> dict[int, WRAPPED_EXCEPTION]:
        """Return a dictionary mapping node ranks to their associated exceptions in case of failure."""
        return self._failures

    def __str__(self) -> str:
        str = f"CheckpointException ranks:{self._failures.keys()}\n"
        for rank, exc_pair in self._failures.items():
            exc, trace = exc_pair
            str += f"Traceback (most recent call last): (RANK {rank})\n"
            if trace is not None:
                str += "".join(tb.format_list(trace))
            str += "".join(tb.format_exception_only(type(exc), value=exc))
        return str
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates
import dataclasses
import logging
from typing import TYPE_CHECKING

from torch.distributed.checkpoint.planner import SavePlan


if TYPE_CHECKING:
    from torch.distributed.checkpoint.metadata import MetadataIndex

__all__ = ["dedup_tensors"]


def init_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    level = logging.INFO
    logger.setLevel(level)
    console = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)s %(levelname)s p:%(processName)s t:%(threadName)s: %(message)s"
    )
    console.setFormatter(formatter)
    console.setLevel(level)
    logger.addHandler(console)
    logger.propagate = False
    return logger


logger = init_logger()


# TODO add docstring for dedup_tensors
def dedup_tensors(all_plans: list[SavePlan]) -> list[SavePlan]:
    all_plans = list(all_plans)
    key_to_plan: dict[MetadataIndex, list[int]] = {}
    for plan_idx, plan in enumerate(all_plans):
        for write_item in plan.items:
            key_to_plan.setdefault(write_item.index, []).append(plan_idx)

    replicated_items = {k: v for k, v in key_to_plan.items() if len(v) > 1}

    # Remove duplicates by always keeping the first entry.
    # Compute the per-rank remove set.
    plan_to_keys: dict[int, list[MetadataIndex]] = {}
    for key, plans in replicated_items.items():
        for plan_idx in plans[1:]:
            plan_to_keys.setdefault(plan_idx, []).append(key)
    if len(plan_to_keys) > 0:
        logger.info("Duplicate keys to remove: %s", plan_to_keys)

    for plan_idx, keys in plan_to_keys.items():
        key_set = set(keys)
        # rewrite items and remove elements
        new_items = [
            write_item
            for write_item in all_plans[plan_idx].items
            if write_item.index not in key_set
        ]
        all_plans[plan_idx] = dataclasses.replace(all_plans[plan_idx], items=new_items)

    return all_plans
<|endoftext|># mypy: allow-untyped-defs
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from torch.distributed.checkpoint.stateful import StatefulT


__all__ = [
    "ChunkStorageMetadata",
    "TensorStorageMetadata",
    "BytesStorageMetadata",
    "Metadata",
    "MetadataIndex",
    "TensorProperties",
    "StorageMeta",
]


@dataclass
class ChunkStorageMetadata:
    """
    Each chunk is expected to have the same properties of the TensorStorageMetadata
    that includes it.
    """

    offsets: torch.Size
    sizes: torch.Size


class _MEM_FORMAT_ENCODING(Enum):
    """Describe the memory format of a tensor."""

    TORCH_CONTIGUOUS_FORMAT = 0
    TORCH_CHANNELS_LAST = 1
    TORCH_PRESERVE_FORMAT = 2


@dataclass
class TensorProperties:
    """Properties used to create :class:`Tensor`"""

    # Regular tensor fields
    dtype: torch.dtype = field(default_factory=torch.get_default_dtype)
    # This field is deprecated.
    layout: torch.layout = field(default=torch.strided)
    # This field is deprecated.
    requires_grad: bool = False
    # This field is deprecated.
    memory_format: torch.memory_format = field(default=torch.contiguous_format)
    # This field is deprecated.
    pin_memory: bool = False

    def __getstate__(self):
        # Since torch.memory_format cannot be pickled!
        memory_format = self.memory_format
        if memory_format == torch.contiguous_format:
            mem_format_encoding = _MEM_FORMAT_ENCODING.TORCH_CONTIGUOUS_FORMAT
        elif memory_format == torch.channels_last:
            mem_format_encoding = _MEM_FORMAT_ENCODING.TORCH_CHANNELS_LAST
        elif memory_format == torch.preserve_format:
            mem_format_encoding = _MEM_FORMAT_ENCODING.TORCH_PRESERVE_FORMAT
        else:
            raise RuntimeError(f"Invalid torch.memory_format: {memory_format}")

        return (
            self.dtype,
            self.layout,
            self.requires_grad,
            mem_format_encoding,
            self.pin_memory,
        )

    def __setstate__(
        self,
        state,
    ):
        (
            self.dtype,
            self.layout,
            self.requires_grad,
            mem_format_encoding,
            self.pin_memory,
        ) = state

        if mem_format_encoding == _MEM_FORMAT_ENCODING.TORCH_CONTIGUOUS_FORMAT:
            memory_format = torch.contiguous_format
        elif mem_format_encoding == _MEM_FORMAT_ENCODING.TORCH_CHANNELS_LAST:
            memory_format = torch.channels_last
        elif mem_format_encoding == _MEM_FORMAT_ENCODING.TORCH_PRESERVE_FORMAT:
            memory_format = torch.preserve_format
        else:
            raise RuntimeError(
                f"Invalid torch.memory_format encoding: {mem_format_encoding}"
            )

        self.memory_format = memory_format

    @staticmethod
    def create_from_tensor(tensor: torch.Tensor) -> "TensorProperties":
        return TensorProperties(
            dtype=tensor.dtype,
            layout=tensor.layout,
            requires_grad=tensor.requires_grad,
            memory_format=torch.contiguous_format,
            pin_memory=tensor.is_pinned(),
        )


@dataclass
class TensorStorageMetadata:
    properties: TensorProperties
    size: torch.Size
    chunks: list[ChunkStorageMetadata]


@dataclass
class BytesStorageMetadata:
    pass


STORAGE_TYPES = TensorStorageMetadata | BytesStorageMetadata
STATE_DICT_TYPE = dict[str, StatefulT | Any]


@dataclass
class StorageMeta:
    checkpoint_id: str | os.PathLike | None = None
    save_id: str | None = None
    load_id: str | None = None
    modules: list[str] = field(default_factory=list)


@dataclass
class Metadata:
    """This class represents the metadata of the checkpoint."""

    # Keys are the same from the `state_dict` used.
    state_dict_metadata: dict[str, STORAGE_TYPES]
    # It is the responsibility of the planner and storage plugins to ensure
    # backward compatibility of the planner_data and storage_data. DCP will
    # also ensure the backward compatibility of the metadata in this file and
    # the metadata of the built-in planner and storage plugins.
    planner_data: Any = None
    storage_data: Any = None
    storage_meta: StorageMeta | None = None
    version: str | None = None


@dataclass(frozen=True)
class MetadataIndex:
    """This class represents a lookup key for items in a state dict or Metadata."""

    fqn: str
    """Fully Qualified Name of the object"""

    offset: torch.Size | None = None
    """If the object is a tensor, offset into the tensor we're looking for"""

    index: int | None = field(hash=False, compare=False, default=None)
    """
    Index hint when searching for tensor chunk to speedup lookups (optional)

    A common representation of a sharded tensor is as a list of chunks so to
    find the index in such a list you need to linear search it.

    When constructing an instance of MetadataIndex that points to that list,
    one can provide the index as a hint and it will be probed first before
    the linear search and thus making it significantly faster.
    """

    def __init__(
        self,
        fqn: str,
        offset: Sequence[int] | None = None,
        index: int | None = None,
    ):
        # We must use object.__setattr__ due to frozen=True
        object.__setattr__(self, "fqn", fqn)
        object.__setattr__(self, "index", index)
        if offset is not None:
            object.__setattr__(self, "offset", torch.Size(offset))
<|endoftext|>import abc
import os
from dataclasses import dataclass
from typing import Any

from torch.distributed.checkpoint.metadata import Metadata, MetadataIndex, StorageMeta
from torch.distributed.checkpoint.planner import (
    LoadPlan,
    LoadPlanner,
    SavePlan,
    SavePlanner,
)
from torch.futures import Future


__all__ = ["WriteResult", "StorageWriter", "StorageReader"]


@dataclass(frozen=True)
class WriteResult:
    index: MetadataIndex

    size_in_bytes: int
    storage_data: Any


class StorageWriter(abc.ABC):
    """
    Interface used by ``save_state_dict`` to write to storage.

    One StorageWriter instance acts as both the coordinator and the follower
    in a distributed checkpoint. As part of initialization, each instance
    is told its role.

    A subclass should expect the following sequence of calls.

    0) (all ranks) set checkpoint_id if users pass a valid checkpoint_id.
    1) (all ranks) set_up_storage_writer()
    2) (all ranks) prepare_local_plan()
    3) (coordinator) prepare_global_plan()
    4) (all ranks) write_data()
    5) (coordinator) finish()
    """

    @abc.abstractmethod
    def reset(self, checkpoint_id: str | os.PathLike | None = None) -> None:
        """
        Calls to indicates a brand new checkpoint write is going to happen.
        A checkpoint_id may be present if users set the checkpoint_id for
        this checkpoint write. The meaning of the checkpoint_id is
        storage-dependent. It can be a path to a folder/file or a key for
        a key-value storage.

        Args:
            checkpoint_id (Union[str, os.PathLike, None]):
                The ID of this checkpoint instance. The meaning of the checkpoint_id
                depends on the storage. It can be a path to a folder or to a file.
                It can also be a key if the storage is a key-value store.
                (Default: ``None``)
        """
        ...

    @abc.abstractmethod
    def set_up_storage_writer(
        self, is_coordinator: bool, *args: Any, **kwargs: Any
    ) -> None:
        """
        Initialize this instance.

        Args:
            is_coordinator (bool): Whether this instance is responsible for coordinating
              the checkpoint.
        """

    @abc.abstractmethod
    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        """
        Perform storage-specific local planning.

        While this method can produce a completely different plan, the recommended
        way is to store storage specific data in SavePlan::storage_data.

        Args:
            plan (SavePlan): The local plan from the ``SavePlanner`` in use.

        Returns:
            A transformed ``SavePlan`` after storage local planning
        """

    @abc.abstractmethod
    def prepare_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]:
        """
        Perform centralized planning of storage.

        This method is only called on the coordinator instance.

        While this method can produce a completely different plan, the preferred
        way is to store storage specific data in SavePlan::storage_data.

        Args:
            plans: A list of ``SavePlan`` instances, one for each rank.

        Returns:
            A list of transformed ``SavePlan`` after storage global planning
        """

    @abc.abstractmethod
    def write_data(
        self, plan: SavePlan, planner: SavePlanner
    ) -> Future[list[WriteResult]]:
        """
        Write all items from ``plan`` using ``planner`` to resolve the data.

        A subclass should call ``SavePlanner::resolve_data`` on each item
        from the plan to get access to the underlying object to write.

        Subclasses should lazily call `resolve_data` as it can allocate memory.
        In case of tensors, make following assumptions:

        - They might be on any device, including not matching the one on ``WriteItem::tensor_data``
        - They might be views or not contiguous. Only the projection needs to be saved.

        Args:
            plan (SavePlan): The save plan to execute.
            planner (SavePlanner): Planner object to be used to resolve items to data.

        Returns:
            A future that completes to a list of WriteResult
        """

    @abc.abstractmethod
    def finish(self, metadata: Metadata, results: list[list[WriteResult]]) -> None:
        """
        Write the metadata and marks the current checkpoint as successful.

        The actual format/schema used for serializing `metadata` is an
        implementation detail. The only requirement is that it's recoverable
        in to the same object graph.

        Args:
            metadata (Metadata): metadata for the new checkpoint
            results: A list of WriteResults from all ranks.

        Returns:
            None
        """

    @classmethod
    @abc.abstractmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        """
        Check if the given checkpoint_id is supported by the storage. This allows
        us to enable automatic storage selection.
        """
        ...

    def storage_meta(self) -> StorageMeta | None:
        """
        Return the storage-specific metadata. This is used to store additional information
        in a checkpoint that can be useful for providing request-level observability. StorageMeta
        is passed to the ``SavePlanner`` during save calls. Returns None by default.

        TODO: provide an example
        """
        return None


class StorageReader(abc.ABC):
    """
    Interface used by ``load_state_dict`` to read from storage.

    One StorageReader instance acts as both the coordinator and the follower
    in a distributed checkpoint. As part of initialization, each instance
    is told its role.

    A subclass should expect the following sequence of calls by ``load_state_dict``:

    0) (all ranks) set checkpoint_id if users pass a valid checkpoint_id.
    1) (all ranks) read_metadata()
    2) (all ranks) set_up_storage_reader()
    3) (all ranks) prepare_local_plan()
    4) (coordinator) prepare_global_plan()
    5) (all ranks) read_data()
    """

    @abc.abstractmethod
    def reset(self, checkpoint_id: str | os.PathLike | None = None) -> None:
        """
        Calls to indicates a brand new checkpoint read is going to happen.
        A checkpoint_id may be present if users set the checkpoint_id for
        this checkpoint read. The meaning of the checkpoint_id is
        storage-dependent. It can be a path to a folder/file or a key for
        a key-value storage.

        Args:
            checkpoint_id (Union[str, os.PathLike, None]):
                The ID of this checkpoint instance. The meaning of the checkpoint_id
                depends on the storage. It can be a path to a folder or to a file.
                It can also be a key if the storage is more like a key-value store.
                (Default: ``None``)
        """
        ...

    @abc.abstractmethod
    def read_metadata(self, *args: Any, **kwargs: Any) -> Metadata:
        """
        Read the checkpoint metadata.

        Returns:
            The metadata object associated with the checkpoint being loaded.

        """

    @abc.abstractmethod
    def set_up_storage_reader(
        self, metadata: Metadata, is_coordinator: bool, *args: Any, **kwargs: Any
    ) -> None:
        """
        Initialize this instance.

        Args:
            metadata (Metadata): The metadata schema to use.
            is_coordinator (bool): Whether this instance is responsible for coordinating
              the checkpoint.
        """

    @abc.abstractmethod
    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        """
        Perform storage-specific local planning.

        While this method can produce a completely different plan, the recommended
        way is to store storage specific data in LoadPlan::storage_data.

        Args:
            plan (LoadPlan): The local plan from the ``LoadPlan`` in use.

        Returns:
            A transformed ``LoadPlan`` after storage local planning
        """

    @abc.abstractmethod
    def prepare_global_plan(self, plans: list[LoadPlan]) -> list[LoadPlan]:
        """
        Perform centralized planning of storage loading.

        This method is only called on the coordinator instance.

        While this method can produce a completely different plan, the preferred
        way is to store storage specific data in LoadPlan::storage_data.

        Args:
            plans: A list of ``LoadPlan`` instances, one for each rank.

        Returns:
            A list of transformed ``LoadPlan`` after storage global planning
        """

    @abc.abstractmethod
    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        """
        Read all items from ``plan`` using ``planner`` to resolve the data.

        A subclass should call ``LoadPlanner::load_bytes`` to deserialize a BytesIO
        object into the right place.

        A subclass should call ``LoadPlanner::resolve_tensor`` to get access to the
        tensors that in should load data into.

        It's the StorageLayer responsibility to properly schedule any cross device copies
        required.

        Args:
            plan (LoadPlan): The local plan to execute on
            planner (LoadPlanner): The planner object to use to resolve items.

        Returns:
            A future that completes once all reads are finished.
        """

    @classmethod
    @abc.abstractmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        """
        Check if the given checkpoint_id is supported by the storage. This allows
        us to enable automatic storage selection.
        """
        ...
<|endoftext|># pyre-strict
# mypy: allow-untyped-defs
import abc
import os
from concurrent.futures import Future

import torch.distributed as dist
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.checkpoint.planner import SavePlanner
from torch.distributed.checkpoint.storage import StorageWriter


class _AsyncCheckpointExecutor(abc.ABC):
    @abc.abstractmethod
    def execute_save(
        self,
        staging_future_or_state_dict: STATE_DICT_TYPE | Future[STATE_DICT_TYPE],
        *,
        checkpoint_id: str | os.PathLike | None = None,
        storage_writer: StorageWriter | None = None,
        planner: SavePlanner | None = None,
        process_group: dist.ProcessGroup | None = None,
        no_dist: bool = False,
        use_collectives: bool = True,
    ) -> Future:
        """
        Execute the checkpoint save request asynchronously.

        This method is intended to be used as an abstraction for
        implementing async checkpointing. The actual checkpoint save
        operation is executed in a separate thread or process depending
        on the implementation of this interface.
        """
<|endoftext|>from __future__ import annotations

from typing import Protocol, runtime_checkable, TypeGuard

import torch

from .metadata import ChunkStorageMetadata, MetadataIndex


__all__ = ["CheckpointableTensor"]


@runtime_checkable
class CheckpointableTensor(Protocol):
    """Protocol fields for checkpointing a local tensor as global tensor shards.

    A tensor does not need to be wrapped or subclassed for DCP to checkpoint it
    as a shard. It can stay a regular local ``torch.Tensor``; implementing
    these fields is enough for DCP to map one or more slices of that tensor
    into a logical global tensor.

    Attributes:
        global_shape: Full logical tensor shape to write into checkpoint
            metadata; needed because ``tensor.size()`` is only the local buffer
            size.
        global_offsets: Global start coordinate for each local shard; needed to
            name checkpoint chunks and match load requests by global offset.
        local_offsets: Start coordinate for each local shard inside the local
            tensor; needed when one tensor stores multiple shards or includes
            padding.
        local_sizes: Shape of each local shard; needed to build checkpoint
            chunks and slice the local tensor during load.
    """

    global_shape: tuple[int, ...]
    global_offsets: tuple[tuple[int, ...], ...]
    local_offsets: tuple[tuple[int, ...], ...]
    local_sizes: tuple[tuple[int, ...], ...]


def _is_checkpointable_tensor(obj: object) -> TypeGuard[CheckpointableTensor]:
    return isinstance(obj, torch.Tensor) and isinstance(obj, CheckpointableTensor)


def _copy_checkpointable_tensor_metadata(
    src: CheckpointableTensor, dst: torch.Tensor
) -> None:
    setattr(dst, "global_shape", src.global_shape)  # noqa: B010
    setattr(dst, "global_offsets", src.global_offsets)  # noqa: B010
    setattr(dst, "local_offsets", src.local_offsets)  # noqa: B010
    setattr(dst, "local_sizes", src.local_sizes)  # noqa: B010


def _get_checkpointable_tensor_chunks(
    tensor: CheckpointableTensor,
) -> list[ChunkStorageMetadata]:
    _validate_checkpointable_tensor_metadata(tensor)
    return [
        ChunkStorageMetadata(
            offsets=torch.Size(global_offset),
            sizes=torch.Size(local_size),
        )
        for global_offset, local_size in zip(
            tensor.global_offsets,
            tensor.local_sizes,
            strict=True,
        )
    ]


def _get_checkpointable_tensor_shard(
    tensor: CheckpointableTensor,
    index: MetadataIndex,
) -> torch.Tensor:
    _validate_checkpointable_tensor_metadata(tensor)

    if index.offset is None:
        if len(tensor.global_offsets) == 1:
            shard_idx = 0
        else:
            raise ValueError(
                f"Cannot lookup {index.fqn} with multiple checkpointable shards and no offset"
            )
    elif (
        index.index is not None
        and index.index < len(tensor.global_offsets)
        and torch.Size(tensor.global_offsets[index.index]) == index.offset
    ):
        shard_idx = index.index
    else:
        shard_idx = -1
        for idx, global_offset in enumerate(tensor.global_offsets):
            if torch.Size(global_offset) == index.offset:
                shard_idx = idx
                break
        if shard_idx < 0:
            raise ValueError(
                f"Could not find checkpointable tensor shard at '{index.offset}' "
                f"for FQN: '{index.fqn}'"
            )

    local_offset = tensor.local_offsets[shard_idx]
    local_size = tensor.local_sizes[shard_idx]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("CheckpointableTensor must also be a torch.Tensor")
    local_tensor = tensor
    if not local_offset:
        return local_tensor
    return local_tensor[
        tuple(
            slice(offset, offset + size)
            for offset, size in zip(local_offset, local_size, strict=True)
        )
    ]


def _validate_checkpointable_tensor_metadata(tensor: CheckpointableTensor) -> None:
    num_shards = len(tensor.global_offsets)
    if len(tensor.local_offsets) != num_shards:
        raise ValueError("global_offsets and local_offsets must have the same length")
    if len(tensor.local_sizes) != num_shards:
        raise ValueError("global_offsets and local_sizes must have the same length")

    global_shape = tensor.global_shape
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("CheckpointableTensor must also be a torch.Tensor")
    tensor_shape = tuple(tensor.size())
    for idx, (global_offset, local_offset, local_size) in enumerate(
        zip(
            tensor.global_offsets,
            tensor.local_offsets,
            tensor.local_sizes,
            strict=True,
        )
    ):
        if len(global_offset) != len(global_shape):
            raise ValueError(
                f"global_offsets[{idx}] must have {len(global_shape)} dimensions"
            )
        if len(local_offset) != len(tensor_shape):
            raise ValueError(
                f"local_offsets[{idx}] must have {len(tensor_shape)} dimensions"
            )
        if len(local_size) != len(global_shape):
            raise ValueError(
                f"local_sizes[{idx}] must have {len(global_shape)} dimensions"
            )
        if len(local_size) != len(tensor_shape):
            raise ValueError(
                f"local_sizes[{idx}] must have {len(tensor_shape)} local dimensions"
            )

        for dim, (offset, size, global_dim) in enumerate(
            zip(global_offset, local_size, global_shape, strict=True)
        ):
            if offset < 0 or size < 0 or offset + size > global_dim:
                raise ValueError(
                    f"global shard {idx} dimension {dim} is outside global_shape"
                )

        for dim, (offset, size, local_dim) in enumerate(
            zip(local_offset, local_size, tensor_shape, strict=True)
        ):
            if offset < 0 or size < 0 or offset + size > local_dim:
                raise ValueError(
                    f"local shard {idx} dimension {dim} is outside tensor shape"
                )
<|endoftext|># pyre-strict
# mypy: allow-untyped-defs
import gc
import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any
from uuid import uuid4

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import PrefixStore, TCPStore
from torch.distributed.checkpoint._async_executor import _AsyncCheckpointExecutor
from torch.distributed.checkpoint.logger import _dcp_method_logger, _init_logger
from torch.distributed.checkpoint.metadata import Metadata, STATE_DICT_TYPE
from torch.distributed.checkpoint.planner import SavePlanner
from torch.distributed.checkpoint.storage import StorageWriter
from torch.distributed.checkpoint.utils import _DistWrapper
from torch.distributed.elastic.agent.server.api import _get_fq_hostname
from torch.distributed.elastic.utils.distributed import get_free_port


logger = logging.getLogger()


class _CheckpointSaveProcessControlOpts(Enum):
    INIT_COMPLETE = "init_complete"
    TERMINATE = "terminate"


@dataclass(init=False, unsafe_hash=True)
class _CheckpointRequestIdentifier:
    checkpoint_id: str | os.PathLike | None
    uuid: str

    def __init__(self, checkpoint_id: str | os.PathLike | None):
        self.checkpoint_id = checkpoint_id
        self.uuid = str(uuid4())


@dataclass
class _AsyncCheckpointRequest:
    staged_state_dict: STATE_DICT_TYPE
    checkpoint_request_id: _CheckpointRequestIdentifier
    storage_writer: StorageWriter | None = None
    planner: SavePlanner | None = None
    no_dist: bool = False
    use_collectives: bool = True


@dataclass(init=False)
class _ProcessGroupInitInfo:
    local_rank: int
    global_rank: int
    world_size: int
    tcp_store_master_addr: str
    tcp_store_master_port: int
    use_prefix_store: bool
    disable_automatic_gc: bool
    disable_manual_gc: bool

    def __init__(self, process_group: dist.ProcessGroup | None = None):
        self.local_rank = dist.get_node_local_rank(fallback_rank=0)
        self.global_rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        self.use_prefix_store = os.environ.get("DCP_USE_PREFIX_STORE", "0") == "1"
        self.disable_automatic_gc = (
            os.environ.get("DCP_DISABLE_AUTOMATIC_GC", "0") == "1"
        )
        self.disable_manual_gc = os.environ.get("DCP_DISABLE_MANUAL_GC", "0") == "1"

        # Let coordinator rank find a port on the localhost.
        # Broadcast the (master_addr, port) to all ranks; each rank in the
        # checkpoint daemon process will use TCPStore (master_addr, port)
        # for collective communication.
        dist_wrapper: _DistWrapper = _DistWrapper(
            group=process_group,
            use_dist=True,
            coordinator_rank=0,
        )

        def get_master_addr_and_port() -> tuple[str, int]:
            if self.use_prefix_store:
                master_addr = os.environ.get("MASTER_ADDR")
                master_port = os.environ.get("MASTER_PORT")
                if master_addr is None:
                    raise AssertionError("DCP needs MASTER_ADDR to use prefix store")
                if master_port is None:
                    raise AssertionError("DCP needs MASTER_PORT to use prefix store")
                master_port = int(master_port)
            else:
                master_addr = os.environ.get("MASTER_ADDR")
                if master_addr is None:
                    master_addr = _get_fq_hostname()
                master_port = get_free_port()

            return master_addr, master_port

        self.tcp_store_master_addr, self.tcp_store_master_port = dist_wrapper.broadcast(
            step="get_master_addr_and_port",
            map_fun=get_master_addr_and_port,
        )


class _AsyncCheckpointProcess:
    def __init__(
        self,
        pg_init_info: _ProcessGroupInitInfo,
    ):
        self.ctx = mp.get_context("spawn")
        self._process_pipe, child_end = self.ctx.Pipe()

        self._save_process = self.ctx.Process(
            target=self._checkpointing_subprocess,
            args=(
                pg_init_info,
                child_end,
            ),
            daemon=True,
        )

        self._save_process.start()

        # Close the parent's copy of child end after we pass it into the child,
        # so the recv()s on it will fail-fast if the child process dies.
        child_end.close()

        # Wait for the checkpoint background process to initialize.
        # Using default GLOO init timeout.
        response = self._wait_for_response(timeout=1800)
        if not response == _CheckpointSaveProcessControlOpts.INIT_COMPLETE:
            raise AssertionError(f"Expected INIT_COMPLETE response, got {response}")

    def __del__(self) -> None:
        if self._save_process.is_alive():
            try:
                logger.info("Terminating the checkpoint background process.")
                self._send(_CheckpointSaveProcessControlOpts.TERMINATE)
                self._save_process.join(timeout=5)
            finally:
                if self._save_process.is_alive():
                    logger.warning(
                        "Checkpoint background process is still alive after termination request. Sending SIGTERM."
                    )
                    self._save_process.terminate()

    def _send(self, data: Any) -> None:
        self._process_pipe.send(data)

    def _wait_for_response(self, timeout: float | None = None) -> Any:
        if not self._save_process.is_alive():
            logger.info("Checkpoint background process is dead calling join()...")
            self._save_process.join()
            raise RuntimeError(
                f"Checkpoint background process is dead. Exit code: {self._save_process.exitcode}"
            )

        if timeout is not None and not self._process_pipe.poll(timeout=timeout):
            raise RuntimeError(
                f"Timed out after {timeout}s while waiting for response from checkpointer process pid: {self._save_process.pid}"
            )

        try:
            response = self._process_pipe.recv()
        except EOFError:
            raise RuntimeError(  # noqa: B904
                f"Checkpoint background process is dead. Exit code: {self._save_process.exitcode}"
            )

        if isinstance(response, BaseException):
            raise response

        return response

    def save(
        self,
        staged_state_dict: STATE_DICT_TYPE,
        *,
        checkpoint_id: str | os.PathLike | None = None,
        storage_writer: StorageWriter | None = None,
        planner: SavePlanner | None = None,
        no_dist: bool = False,
        use_collectives: bool = True,
    ) -> Metadata:
        # Create a unique identifier to locate requests/responses
        # from the checkpoint daemon process.
        checkpoint_request_id = _CheckpointRequestIdentifier(checkpoint_id)
        async_cp_request = _AsyncCheckpointRequest(
            staged_state_dict=staged_state_dict,
            checkpoint_request_id=checkpoint_request_id,
            storage_writer=storage_writer,
            planner=planner,
            no_dist=no_dist,
            use_collectives=use_collectives,
        )
        self._send(async_cp_request)
        result = self._wait_for_response()
        if not isinstance(result, Metadata):
            raise AssertionError(f"Expected Metadata response, got {type(result)}")
        return result

    @staticmethod
    def _execute_save(
        state_dict: STATE_DICT_TYPE,
        *,
        checkpoint_request_id: _CheckpointRequestIdentifier,
        storage_writer: StorageWriter | None = None,
        planner: SavePlanner | None = None,
        no_dist: bool = False,
        use_collectives: bool = True,
    ) -> Metadata:
        from torch.distributed.checkpoint.state_dict_saver import save

        metadata = save(
            state_dict,
            checkpoint_id=checkpoint_request_id.checkpoint_id,
            storage_writer=storage_writer,
            planner=planner,
            no_dist=no_dist,
            use_collectives=use_collectives,
        )
        return metadata

    @staticmethod
    def _checkpointing_subprocess(
        pg_init_info: _ProcessGroupInitInfo,
        parent_conn,
    ) -> None:
        # Phase 1: Process Group Initialization
        # Only needs to execute once during the lifetime of the checkpoint background process.
        try:
            _init_logger(pg_init_info.global_rank)

            # Setup environment variables for process group initialization.
            os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"
            os.environ["MASTER_ADDR"] = pg_init_info.tcp_store_master_addr
            os.environ["MASTER_PORT"] = str(pg_init_info.tcp_store_master_port)
            os.environ["LOCAL_RANK"] = str(pg_init_info.local_rank)
            os.environ["RANK"] = str(pg_init_info.global_rank)
            os.environ["WORLD_SIZE"] = str(pg_init_info.world_size)

            logger.info(
                "Initializing dist.ProcessGroup in checkpoint background process on port %s",
                pg_init_info.tcp_store_master_port,
            )
            # NOTE: GLOO backend is enforced here.
            if pg_init_info.use_prefix_store:
                logger.info(
                    "Initializing dist.ProcessGroup in checkpoint background process with prefix store"
                )
                store = PrefixStore(
                    "AsyncCheckpointProcess/",
                    TCPStore(
                        pg_init_info.tcp_store_master_addr,
                        pg_init_info.tcp_store_master_port,
                    ),
                )
                dist.init_process_group(
                    backend=dist.Backend.GLOO,
                    store=store,
                    world_size=pg_init_info.world_size,
                    rank=pg_init_info.global_rank,
                )
            else:
                dist.init_process_group(backend=dist.Backend.GLOO)
            dist.barrier()

            logger.info("Checkpoint background process is running...")
            parent_conn.send(_CheckpointSaveProcessControlOpts.INIT_COMPLETE)

            if pg_init_info.disable_automatic_gc:
                # Disable automatic garbage collection
                # GC can optionally be called manually after each checkpoint
                gc.disable()
                logger.info("Disabled automatic garbage collection")
        except BaseException as e:
            logger.error(
                f"Checkpoint background process failed during initialization: {e}"  # noqa: G004
            )
            parent_conn.send(e)
            return

        # Phase 2: Serving Loop
        try:
            first_request = True
            while True:
                logger.info("Waiting for checkpoint save request...")
                obj = parent_conn.recv()
                if (
                    isinstance(obj, _CheckpointSaveProcessControlOpts)
                    and obj == _CheckpointSaveProcessControlOpts.TERMINATE
                ):
                    logger.info("Terminating the checkpoint background process.")
                    return
                if not isinstance(obj, _AsyncCheckpointRequest):
                    raise AssertionError(
                        f"Expected _AsyncCheckpointRequest, got {type(obj)}"
                    )
                logger.info(
                    f"Received async checkpoint request with id={obj.checkpoint_request_id.checkpoint_id}"  # noqa: G004
                )

                try:
                    response = _AsyncCheckpointProcess._execute_save(
                        obj.staged_state_dict,
                        checkpoint_request_id=obj.checkpoint_request_id,
                        storage_writer=obj.storage_writer,
                        planner=obj.planner,
                        no_dist=obj.no_dist,
                        use_collectives=obj.use_collectives,
                    )
                    parent_conn.send(response)
                    logger.info(
                        f"Completed checkpoint save request for checkpoint_id={obj.checkpoint_request_id}"  # noqa: G004
                    )

                    # in theory this manual gc should not be needed as we shouldn't be leaking anything from checkpointing process
                    if (
                        pg_init_info.disable_automatic_gc
                        and not pg_init_info.disable_manual_gc
                    ):
                        del obj

                        collected_objects = gc.collect()

                        logger.info(
                            f"Manual garbage collection completed - collected {collected_objects} objects."  # noqa: G004
                        )
                        if first_request:
                            # Freeze GC to not check GC for large checkpoint save plans
                            # After freezing, subsequent gc.collect() calls will only scan
                            # NEW objects created after this point, not the frozen save plan
                            logger.info(
                                "First checkpoint request completed - freezing gc"
                            )
                            gc.freeze()
                    first_request = False
                except BaseException as e:
                    logger.error(
                        f"Checkpoint save failed for checkpoint_id={obj.checkpoint_request_id.checkpoint_id}: {e}"  # noqa: G004
                    )
                    parent_conn.send(e)
                    # Continue serving loop - don't exit process
        finally:
            logger.info("Checkpoint background process is shutting down...")
            dist.destroy_process_group()
            parent_conn.close()


_CHECKPOINT_PROCESS: _AsyncCheckpointProcess | None = None


class _ProcessBasedAsyncCheckpointExecutor(_AsyncCheckpointExecutor):
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1)

    @staticmethod
    def _execute_save_impl(
        *,
        pg_init_info: _ProcessGroupInitInfo | None,
        staging_future_or_state_dict: Future[STATE_DICT_TYPE] | STATE_DICT_TYPE,
        checkpoint_id: str | os.PathLike | None = None,
        storage_writer: StorageWriter | None = None,
        planner: SavePlanner | None = None,
        process_group: dist.ProcessGroup | None = None,
        no_dist: bool = False,
        use_collectives: bool = True,
    ) -> Metadata:
        global _CHECKPOINT_PROCESS
        if _CHECKPOINT_PROCESS is None:
            if pg_init_info is None:
                raise AssertionError(
                    "pg_init_info must not be None when _CHECKPOINT_PROCESS is None"
                )
            ckpt_kwargs = {}
            if (ckpt_id := getattr(storage_writer, "checkpoint_id", None)) is not None:
                ckpt_kwargs["checkpoint_id"] = ckpt_id
                ckpt_kwargs["process_group"] = process_group

            @_dcp_method_logger(**ckpt_kwargs)
            def create_checkpoint_daemon_process() -> None:
                global _CHECKPOINT_PROCESS
                # pyrefly: ignore [bad-argument-type]
                _CHECKPOINT_PROCESS = _AsyncCheckpointProcess(pg_init_info=pg_init_info)

            create_checkpoint_daemon_process()

        if _CHECKPOINT_PROCESS is None:
            raise AssertionError(
                "_CHECKPOINT_PROCESS must not be None after initialization"
            )
        staged_state_dict = (
            staging_future_or_state_dict.result()
            if isinstance(staging_future_or_state_dict, Future)
            else staging_future_or_state_dict
        )
        return _CHECKPOINT_PROCESS.save(
            staged_state_dict=staged_state_dict,
            checkpoint_id=checkpoint_id,
            storage_writer=storage_writer,
            planner=planner,
            no_dist=no_dist,
            use_collectives=use_collectives,
        )

    def execute_save(
        self,
        staging_future_or_state_dict: Future[STATE_DICT_TYPE] | STATE_DICT_TYPE,
        *,
        checkpoint_id: str | os.PathLike | None = None,
        storage_writer: StorageWriter | None = None,
        planner: SavePlanner | None = None,
        process_group: dist.ProcessGroup | None = None,
        no_dist: bool = False,
        use_collectives: bool = True,
    ) -> Future:
        """
        NOTE:

        - Checkpoint process is implemented as a daemon process.
        The AsyncCheckpointProcess' lifetime is tied to the lifetime of the
        main process (e.g. trainer process).

        - The first call to execute_save_in_process() will initialize the checkpoint
        daemon process. Subsequent async checkpoint requests will not need process
        initialization. Therefore, the first async checkpoint request will take longer to complete.

        - Process initialization can have significant overhead, dominated by latency for all ranks to spawn
        a background process + process group initialization in the background process.
        """

        global _CHECKPOINT_PROCESS
        pg_init_info: _ProcessGroupInitInfo | None = None
        if _CHECKPOINT_PROCESS is None:
            # Find a port on coordinator rank and broadcast
            # to all ranks.
            pg_init_info = _ProcessGroupInitInfo(process_group)

        f: Future = self._executor.submit(
            self._execute_save_impl,
            pg_init_info=pg_init_info,
            staging_future_or_state_dict=staging_future_or_state_dict,
            checkpoint_id=checkpoint_id,
            storage_writer=storage_writer,
            planner=planner,
            no_dist=no_dist,
            use_collectives=use_collectives,
        )
        f.add_done_callback(lambda f: self._executor.shutdown(wait=False))

        return f
<|endoftext|># mypy: allow-untyped-defs
import collections
import dataclasses
import io
import json
import operator
import os
import pickle
import queue
import threading
import uuid
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from io import UnsupportedOperation
from pathlib import Path
from typing import Any, cast, Final, IO

# introduced as collections.abc.Buffer in Python 3.12
from typing_extensions import Buffer

import torch
from torch import Tensor
from torch._utils import _get_available_device_type, _get_device_module
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint._extension import (
    ExtensionRegistry,
    StreamTransformExtension,
)
from torch.distributed.checkpoint._hf_utils import (
    CUSTOM_METADATA_KEY,
    DCP_VERSION_KEY,
    FORMAT_KEY,
    FORMAT_VALUE,
    HF_DCP_VERSION,
)
from torch.distributed.checkpoint.metadata import Metadata, STATE_DICT_TYPE, StorageMeta
from torch.distributed.checkpoint.planner import (
    LoadItemType,
    LoadPlan,
    LoadPlanner,
    ReadItem,
    SavePlan,
    SavePlanner,
    WriteItem,
    WriteItemType,
)
from torch.distributed.checkpoint.staging import BlockingAsyncStager
from torch.distributed.checkpoint.storage import (
    StorageReader,
    StorageWriter,
    WriteResult,
)
from torch.distributed.checkpoint.utils import _create_file_view
from torch.futures import Future


__all__ = [
    "FileSystemWriter",
    "FileSystemReader",
    "FileSystem",
    "FileSystemBase",
    "SerializationFormat",
]

_metadata_fn: str = ".metadata"

CURRENT_DCP_VERSION: Final[str] = "1.0.0"


@dataclass
class _StorageInfo:
    """This is the per entry storage info."""

    relative_path: str
    offset: int
    length: int
    transform_descriptors: Sequence[str] | None = None

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class _StoragePrefix:
    prefix: str


class SerializationFormat(Enum):
    TORCH_SAVE = "torch_save"
    SAFETENSORS = "safetensors"


DEFAULT_SUFFIX = ".distcp"


def _generate_uuid() -> str:
    return str(uuid.uuid4())


class _TensorLoader(ABC):
    @abstractmethod
    def add(self, size: int, obj: object) -> None:
        pass

    @abstractmethod
    def start_loading(self) -> None:
        pass

    @abstractmethod
    def values(self) -> Iterator[tuple[torch.Tensor, object]]:
        pass


class _SerialCpuLoader(_TensorLoader):
    def __init__(self, resolve_fun: Callable) -> None:
        self.resolve_fun = resolve_fun
        self.items: list[tuple[int, object]] = []

    def add(self, size: int, obj: object) -> None:
        self.items.append((size, obj))

    def start_loading(self) -> None:
        pass

    def values(self) -> Iterator[tuple[torch.Tensor, object]]:
        for _, obj in self.items:
            tensor = self.resolve_fun(obj).detach()
            tensor = tensor.cpu()
            if tensor.untyped_storage().size() != tensor.nbytes:
                # creates a new tensor with minimal storage while preserving memory format.
                tensor = tensor.clone()
            yield (
                tensor,
                obj,
            )


class _OverlappingCpuLoader(_TensorLoader):
    def __init__(
        self,
        resolve_fun: Callable,
        stream: torch.Stream | None = None,
        inflight_threshhold: int = 1_000_000,
    ) -> None:
        self.resolve_fun = resolve_fun
        self.items: list[tuple[int, object]] = []
        self.inflight_threshhold = inflight_threshhold
        self.in_flight_data = 0
        self.current_items: collections.deque = collections.deque()
        self.idx = 0
        self.started = False
        self.device_type = (
            stream.device_type if stream else _get_available_device_type()
        )
        self.device_module = _get_device_module(self.device_type)
        self.stream = cast(
            torch.cuda.Stream, stream or self.device_module.current_stream()
        )
        if self.stream != self.device_module.current_stream():
            self.stream.wait_stream(self.device_module.current_stream())

    @property
    def _done(self) -> bool:
        return self.idx >= len(self.items)

    def _drain(self) -> list[tuple[torch.Tensor, object]]:
        drained = []
        if self.in_flight_data >= self.inflight_threshhold:
            self.stream.synchronize()
        while self.in_flight_data >= self.inflight_threshhold:
            val = self.current_items.popleft()
            self.in_flight_data -= val[0].numel() * val[0].element_size()
            drained.append(val)
        return drained

    def _refill(self) -> None:
        with self.device_module.stream(self.stream):
            while not self._done and self.in_flight_data < self.inflight_threshhold:
                _, obj = self.items[self.idx]
                self.idx += 1
                tensor = self.resolve_fun(obj).detach()
                if tensor.device.type == self.device_type:
                    tensor = tensor.to(device="cpu", non_blocking=True)
                elif tensor.device == torch.device("cpu"):
                    if (
                        tensor.untyped_storage().size()
                        != tensor.numel() * tensor.itemsize
                    ):
                        # creates a new tensor with minimal storage while preserving memory format.
                        tensor = tensor.clone()

                self.current_items.append(
                    (
                        tensor,
                        obj,
                    )
                )
                self.in_flight_data += tensor.numel() * tensor.element_size()

    def _finish(self) -> Iterable[tuple[torch.Tensor, object]]:
        if not self._done:
            raise AssertionError("_finish called before all items were processed")
        if len(self.current_items) > 0:
            self.stream.synchronize()
        return self.current_items

    def add(self, size: int, obj: object) -> None:
        if self.started:
            raise RuntimeError("cannot add items after loading started")
        self.items.append((size, obj))

    def start_loading(self) -> None:
        if self.started:
            return
        self.started = True
        self.items.sort(key=operator.itemgetter(0))
        self._refill()

    def values(self) -> Iterator[tuple[torch.Tensor, object]]:
        self.start_loading()
        while not self._done:
            drained = self._drain()
            self._refill()
            yield from drained

        yield from self._finish()


class _StorageWriterTransforms:
    """
    This is experimental, and will likely move elsewhere in the
    future.  It lives here to minimize changes while we are still
    learning and gathering feedback.
    """

    def __init__(
        self, extensions: Sequence[StreamTransformExtension] | None = None
    ) -> None:
        """
        If the extensions arg is None, this means the implementation
        should provide whatever defaults it chooses.  An empty
        sequence indicates no extensions should be used.  At this
        time, the default extensions sequence is empty.
        """
        self.extensions = () if extensions is None else extensions

    def transform_save_stream(
        self, write_item: WriteItem, raw_stream: io.IOBase
    ) -> tuple[IO[bytes], list[str]]:
        # In order to avoid leaking fds, transformers' close must
        # cascade to wrapped streams, but since this function can
        # append to the raw stream, we can't close the actual stream.
        # So, we use this to put a wrapper around the raw stream's
        # close() to make it a noop, and it gets closed once all files
        # are appended.

        class NoCloseWriter(io.IOBase):
            def __init__(self, raw: io.IOBase):
                self.raw = raw

            def writeable(self) -> bool:
                return True

            def write(self, b: Buffer) -> int:
                return self.raw.write(b)

            def close(self):
                self.flush()
                self.raw.flush()
                # but not close.

        transform_to = cast(IO[bytes], NoCloseWriter(raw_stream))

        for ex in self.extensions:
            transform_to = ex.transform_to(transform_to)

        return (transform_to, [ex.get_descriptor() for ex in reversed(self.extensions)])


def _item_size(item: WriteItem) -> int:
    size = 1
    if item.tensor_data is None:
        raise AssertionError("WriteItem tensor_data must not be None")
    # can't use math.prod as PT needs to support older python
    for s in item.tensor_data.size:
        size *= s

    dtype = item.tensor_data.properties.dtype
    return size * torch._utils._element_size(dtype)


def _split_by_size_and_type(bins: int, items: list[WriteItem]) -> list[list[WriteItem]]:
    if bins == 1:
        return [items]

    bytes_w = [wi for wi in items if wi.type == WriteItemType.BYTE_IO]
    tensor_w = [wi for wi in items if wi.type != WriteItemType.BYTE_IO]

    buckets: list[list[WriteItem]] = [[] for _ in range(bins)]
    bucket_sizes = [0 for _ in range(bins)]

    tensor_w.sort(key=_item_size, reverse=True)

    for i, wi in enumerate(bytes_w):
        buckets[i % bins].append(wi)

    for wi in tensor_w:
        # TODO replace with heapq
        idx = min(enumerate(bucket_sizes), key=operator.itemgetter(1))[0]
        buckets[idx].append(wi)
        bucket_sizes[idx] += _item_size(wi)

    return buckets


def _write_item(
    transforms: _StorageWriterTransforms,
    stream: io.IOBase,
    data: io.BytesIO | torch.Tensor,
    write_item: WriteItem,
    storage_key: str,
    serialization_format: SerializationFormat,
) -> WriteResult:
    offset = stream.tell()

    (transform_to, transform_descriptors) = transforms.transform_save_stream(
        write_item, stream
    )

    if write_item.type == WriteItemType.BYTE_IO:
        if not isinstance(data, io.BytesIO):
            raise AssertionError("Data must be io.BytesIO for BYTE_IO write items")
        transform_to.write(data.getbuffer())
    else:
        if not isinstance(data, torch.Tensor):
            raise AssertionError(
                "Data must be torch.Tensor for non-BYTE_IO write items"
            )
        if data.device != torch.device("cpu"):
            raise AssertionError("Tensor must be on CPU device")
        if serialization_format == SerializationFormat.TORCH_SAVE:
            torch.save(data, transform_to)

    transform_to.close()

    if serialization_format == SerializationFormat.TORCH_SAVE or isinstance(
        data, io.BytesIO
    ):
        length = stream.tell() - offset
    else:
        length = data.numel() * data.element_size()

    # For consistency with earlier versions, leave this field out of the
    # metadata if there are no extensions.
    info_transform_descriptors = (
        None if len(transform_descriptors) == 0 else transform_descriptors
    )

    return WriteResult(
        index=write_item.index,
        size_in_bytes=length,
        storage_data=_StorageInfo(
            storage_key,
            offset,
            length,
            transform_descriptors=info_transform_descriptors,
        ),
    )


def _write_files_from_queue(
    create_stream: Callable,
    file_queue: queue.Queue,
    result_queue: queue.Queue,
    planner: SavePlanner,
    transforms: _StorageWriterTransforms,
    inflight_threshhold: int,
    use_fsync: bool,
    thread_count: int,
    serialization_format: SerializationFormat,
) -> None:
    try:
        while True:
            file_name, storage_key, write_items = file_queue.get_nowait()
            loader: _TensorLoader

            custom_backend_name = torch._C._get_privateuse1_backend_name()
            custom_device_mod = getattr(torch, custom_backend_name, None)

            # TODO: Using the OverlappingCpuLoader with multiple threads creates significant
            # performance degradation, observed as being related to cuda stream syncs. We
            # should try to fix this and use _OverlappingCpuLoader for all threaded cases
            if (
                thread_count == 1
                and (
                    torch.cuda.is_available()
                    or (custom_device_mod and custom_device_mod.is_available())
                )
                and inflight_threshhold > 0
            ):
                loader = _OverlappingCpuLoader(
                    planner.resolve_data,
                    inflight_threshhold=inflight_threshhold,
                )
            else:
                loader = _SerialCpuLoader(
                    planner.resolve_data,
                )

            tensor_w = [wi for wi in write_items if wi.type != WriteItemType.BYTE_IO]
            for write_item in tensor_w:
                loader.add(_item_size(write_item), write_item)
            loader.start_loading()

            bytes_w = [wi for wi in write_items if wi.type == WriteItemType.BYTE_IO]
            write_results = []

            with create_stream(file_name, "wb") as stream:
                for write_item in bytes_w:
                    data = planner.resolve_data(write_item)
                    write_results.append(
                        _write_item(
                            transforms,
                            stream,
                            data,
                            write_item,
                            storage_key,
                            serialization_format,
                        )
                    )

                tensor_dict = {}
                metadata_dict = {}
                for tensor, write_item in loader.values():
                    if not tensor.is_cpu:
                        raise AssertionError("Tensor must be on CPU")
                    write_results.append(
                        _write_item(
                            transforms,
                            stream,
                            tensor,
                            write_item,  # type: ignore[arg-type]
                            storage_key,
                            serialization_format,
                        )
                    )
                    tensor_dict[write_item.index.fqn] = tensor  # type: ignore[attr-defined]
                    metadata_dict[write_item.index.fqn] = {  # type: ignore[attr-defined]
                        "saved_offsets": write_item.tensor_data.chunk.offsets  # type: ignore[attr-defined]
                    }

                if serialization_format == SerializationFormat.SAFETENSORS:
                    from safetensors.torch import save  # type: ignore[import-not-found]

                    stream.write(
                        save(
                            tensor_dict,
                            metadata={
                                CUSTOM_METADATA_KEY: json.dumps(metadata_dict),
                                DCP_VERSION_KEY: str(HF_DCP_VERSION),
                                FORMAT_KEY: FORMAT_VALUE,
                            },
                        )
                    )

                if use_fsync:
                    # Flush Python-level buffers (OS for local files, network for cloud storage) before fsync.
                    stream.flush()
                    try:
                        os.fsync(stream.fileno())
                    except (AttributeError, UnsupportedOperation) as e:
                        warnings.warn(
                            f"fsync not supported for this stream, relying on flush(): {e}"
                        )
                stream.close()
            result_queue.put(write_results)
    except queue.Empty:
        pass


class FileSystemBase(ABC):
    @contextmanager
    @abstractmethod
    def create_stream(
        self, path: str | os.PathLike, mode: str
    ) -> Generator[io.IOBase, None, None]: ...

    @abstractmethod
    def concat_path(
        self, path: str | os.PathLike, suffix: str
    ) -> str | os.PathLike: ...

    @abstractmethod
    def rename(self, path: str | os.PathLike, new_path: str | os.PathLike) -> None: ...

    @abstractmethod
    def init_path(self, path: str | os.PathLike) -> str | os.PathLike: ...

    @abstractmethod
    def mkdir(self, path: str | os.PathLike) -> None: ...

    @classmethod
    @abstractmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool: ...

    @abstractmethod
    def exists(self, path: str | os.PathLike) -> bool: ...

    @abstractmethod
    def rm_file(self, path: str | os.PathLike) -> None: ...


class FileSystem(FileSystemBase):
    @contextmanager
    def create_stream(
        self, path: str | os.PathLike, mode: str
    ) -> Generator[io.IOBase, None, None]:
        if not isinstance(path, Path):
            path = Path(path)
        with path.open(mode) as stream:
            yield cast(io.IOBase, stream)

    def concat_path(self, path: str | os.PathLike, suffix: str) -> str | os.PathLike:
        if not isinstance(path, Path):
            path = Path(path)
        return path / suffix

    def init_path(self, path: str | os.PathLike) -> str | os.PathLike:
        if not isinstance(path, Path):
            path = Path(path)
        return path

    def rename(self, path: str | os.PathLike, new_path: str | os.PathLike) -> None:
        if not isinstance(path, Path):
            path = Path(path)

        path.rename(cast(Path, new_path))

    def mkdir(self, path: str | os.PathLike) -> None:
        if not isinstance(path, Path):
            path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        if isinstance(checkpoint_id, Path):
            return True

        if "://" in str(checkpoint_id):
            return False

        for p in Path(checkpoint_id).parents:
            if p.exists() and os.access(str(p), os.W_OK):
                return True

        return False

    def exists(self, path: str | os.PathLike) -> bool:
        if not isinstance(path, Path):
            path = Path(path)
        return path.exists()

    def rm_file(self, path: str | os.PathLike) -> None:
        if not isinstance(path, Path):
            path = Path(path)
        path.unlink()

    def ls(self, path: str | os.PathLike) -> list[str]:
        if not isinstance(path, Path):
            path = Path(path)
        return [str(p) for p in path.iterdir()]


class _FileSystemWriter(StorageWriter):
    """
    Basic implementation of StorageWriter using file IO.

    This implementation makes the following assumptions and simplifications:

    * The checkpoint path is an empty or non-existing directory.
    * File creation is atomic

    The checkpoint consist of one file per write request plus
    a `.metadata` file with the serialized metadata.

    """

    def __init__(
        self,
        path: str | os.PathLike,
        single_file_per_rank: bool = True,
        sync_files: bool = True,
        thread_count: int = 1,
        per_thread_copy_ahead: int = 10_000_000,
        overwrite: bool = True,
        _extensions: Sequence[StreamTransformExtension] | None = None,
        serialization_format: SerializationFormat = SerializationFormat.TORCH_SAVE,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the writer pointing to `path`.

        Args:
            path: directory where the checkpoint will be written to.
            single_file_per_rank: Produce one file per rank instead of one file per tensor/blob. Default to True.
            sync_files : force files to be synced to permanent storage. Default to True.
            thread_count: Number of IO threads to use to write. Default to 1.
            per_thread_copy_ahead: How many bytes to copy from the GPU ahead of saving them. Default 10Mb.
            overwrite: Whether to allow overwriting existing checkpoints. Defaults to True.
            _extensions: Extensions to apply to output streams (EXPERIMENTAL)

        N. B. If sync_files is disabled, there's no guarantee that the checkpoint will be consistent in the case of a failure.
        """
        super().__init__()
        self.fs = FileSystem()
        self.path = self.fs.init_path(path)
        self.single_file_per_rank = single_file_per_rank
        self.sync_files = sync_files
        self.thread_count = thread_count
        self.per_thread_copy_ahead = per_thread_copy_ahead
        self.save_id = _generate_uuid()
        self.overwrite = overwrite
        self.transforms = _StorageWriterTransforms(_extensions)
        self.serialization_format = serialization_format
        self.rank: int | None = None
        self.use_collectives: bool = True

    def reset(self, checkpoint_id: str | os.PathLike | None = None) -> None:
        if checkpoint_id:
            self.path = self.fs.init_path(checkpoint_id)
        self.save_id = _generate_uuid()

    def set_up_storage_writer(
        self, is_coordinator: bool, *args: Any, **kwargs: Any
    ) -> None:
        self.rank = kwargs.get("rank")
        self.use_collectives = kwargs.get("use_collectives", True)

    def _metadata_exists(self) -> bool:
        if self.use_collectives:
            # A global checkpoint metadata file
            metadata_path = self._get_metadata_path(rank=None)
        else:
            # A rank 0 specific metadata file if every rank has written its own metadata
            # Just looking for lowest rank metadata file is sufficient
            metadata_path = self._get_metadata_path(rank=0)

        return self.fs.exists(metadata_path)

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        self.fs.mkdir(self.path)
        if self._metadata_exists():
            if self.overwrite:
                warnings.warn(
                    f"Detected an existing checkpoint in {self.path}, overwriting since {self.overwrite=}."
                    " Past version 2.5 of PyTorch, `overwrite` will default to False. Set this variable to True to"
                    " maintain this functionality or False to raise when an existing checkpoint is found.",
                    stacklevel=2,
                )
            else:
                raise RuntimeError(f"Checkpoint already exists and {self.overwrite=}.")

        if self.rank is not None and not self.use_collectives:
            plan = dataclasses.replace(
                plan, storage_data=_StoragePrefix(f"__{self.rank}_")
            )

        return plan

    def prepare_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]:
        new_plans = [
            dataclasses.replace(plan, storage_data=_StoragePrefix(f"__{i}_"))
            if plan.storage_data is None
            else plan
            for i, plan in enumerate(plans)
        ]
        return new_plans

    def write_data(
        self,
        plan: SavePlan,
        planner: SavePlanner,
    ) -> Future[list[WriteResult]]:
        storage_plan: _StoragePrefix = plan.storage_data
        file_count = 0

        def gen_file():
            nonlocal file_count
            file_name = f"{storage_plan.prefix}{file_count}{DEFAULT_SUFFIX}"
            file_count += 1
            return file_name

        file_queue: queue.Queue = queue.Queue()
        if self.single_file_per_rank:
            for bucket in _split_by_size_and_type(self.thread_count, plan.items):
                file_name = gen_file()
                path = self.fs.concat_path(self.path, file_name)
                file_queue.put((path, file_name, bucket))
        else:
            for item in plan.items:
                file_name = gen_file()
                path = self.fs.concat_path(self.path, file_name)
                file_queue.put((path, file_name, [item]))

        return self._write_data(planner, file_queue)

    def _write_data(
        self,
        planner: SavePlanner,
        file_queue: queue.Queue,
    ) -> Future[list[WriteResult]]:
        result_queue: queue.Queue = queue.Queue()

        threads = []
        for _ in range(1, self.thread_count):
            t = threading.Thread(
                target=_write_files_from_queue,
                args=(
                    self.fs.create_stream,
                    file_queue,
                    result_queue,
                    planner,
                    self.transforms,
                    self.per_thread_copy_ahead,
                    self.sync_files,
                    self.thread_count,
                    self.serialization_format,
                ),
            )
            t.start()
            threads.append(t)

        _write_files_from_queue(
            create_stream=self.fs.create_stream,
            file_queue=file_queue,
            result_queue=result_queue,
            planner=planner,
            transforms=self.transforms,
            inflight_threshhold=self.per_thread_copy_ahead,
            use_fsync=self.sync_files,
            thread_count=self.thread_count,
            serialization_format=self.serialization_format,
        )

        for t in threads:
            t.join()

        res = []
        try:
            while True:
                res += result_queue.get_nowait()
        except queue.Empty:
            fut: Future[list[WriteResult]] = Future()
            fut.set_result(res)
            return fut

    def finish(self, metadata: Metadata, results: list[list[WriteResult]]) -> None:
        metadata.version = CURRENT_DCP_VERSION

        storage_md = {}
        for wr_list in results:
            storage_md.update({wr.index: wr.storage_data for wr in wr_list})
        metadata.storage_data = storage_md

        metadata.storage_meta = self.storage_meta()
        tmp_filename = (
            f"__{self.rank}{_metadata_fn}.tmp"
            if not self.use_collectives and self.rank is not None
            else f"{_metadata_fn}.tmp"
        )
        tmp_path = cast(Path, self.fs.concat_path(self.path, tmp_filename))
        with self.fs.create_stream(tmp_path, "wb") as metadata_file:
            pickle.dump(metadata, metadata_file)
            if self.sync_files:
                # Flush Python-level buffers (OS for local files, network for cloud storage) before fsync.
                metadata_file.flush()
                try:
                    os.fsync(metadata_file.fileno())
                except (AttributeError, UnsupportedOperation) as e:
                    warnings.warn(
                        f"fsync not supported for this stream, relying on flush(): {e}"
                    )

        # delete in-case other checkpoints were present.
        if not self.use_collectives and self.rank is not None:
            metadata_path = self._get_metadata_path(self.rank)
        else:
            metadata_path = self._get_metadata_path()

        if self.fs.exists(metadata_path):
            self.fs.rm_file(metadata_path)

        self.fs.rename(tmp_path, metadata_path)

    def storage_meta(self) -> StorageMeta | None:
        return StorageMeta(checkpoint_id=self.checkpoint_id, save_id=self.save_id)

    def _get_metadata_path(self, rank: int | None = None) -> os.PathLike:
        filename = f"{_metadata_fn}" if rank is None else f"__{rank}{_metadata_fn}"
        return cast(Path, self.fs.concat_path(self.path, filename))

    @property
    def checkpoint_id(self) -> str | os.PathLike:
        """
        return the checkpoint_id that will be used to save the checkpoint.
        """
        return self.path

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        return FileSystem.validate_checkpoint_id(checkpoint_id)


class _StorageReaderTransforms:
    """
    This is experimental, and will likely move elsewhere in the
    future.  It lives here to minimize changes while we are still
    learning and gathering feedback.
    """

    def __init__(self, extension_registry: ExtensionRegistry | None = None) -> None:
        self.extension_registry = (
            ExtensionRegistry() if extension_registry is None else extension_registry
        )

    def transform_load_stream(
        self,
        read_item: ReadItem,
        transform_descriptors: Sequence[str],
        raw_stream: IO[bytes],
    ) -> IO[bytes]:
        extensions = self.extension_registry.from_descriptor_list(transform_descriptors)
        transform_from = raw_stream
        for ex in extensions:
            if isinstance(ex, StreamTransformExtension):
                transform_from = ex.transform_from(transform_from)
        return transform_from


class FileSystemReader(StorageReader):
    def __init__(
        self,
        path: str | os.PathLike,
        _extension_registry: ExtensionRegistry | None = None,  # EXPERIMENTAL
    ) -> None:
        super().__init__()
        self.fs = FileSystem()
        self.path = self.fs.init_path(path)
        self.storage_data: dict[Any, Any] = {}
        self.load_id = _generate_uuid()
        self.transforms = _StorageReaderTransforms(_extension_registry)
        self.rank = None
        self.use_collectives = True

    def _slice_file(self, file, sinfo: _StorageInfo) -> IO[bytes]:
        return cast(IO[bytes], _create_file_view(file, sinfo.offset, sinfo.length))

    def reset(self, checkpoint_id: str | os.PathLike | None = None) -> None:
        self.storage_data = {}
        if checkpoint_id:
            self.path = self.fs.init_path(checkpoint_id)
        self.load_id = _generate_uuid()

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        # group requests by file
        per_file: dict[str, list[ReadItem]] = {}
        for read_item in plan.items:
            item_md: _StorageInfo = self.storage_data[read_item.storage_index]
            path = item_md.relative_path
            per_file.setdefault(path, []).append(read_item)

        for relative_path, reqs in per_file.items():
            new_path = self.fs.concat_path(self.path, relative_path)
            with self.fs.create_stream(new_path, "rb") as stream:
                # TODO sort by offset and cache the reading
                for req in reqs:
                    item_md = self.storage_data[req.storage_index]
                    file_slice = self._slice_file(stream, item_md)
                    transform_from = self.transforms.transform_load_stream(
                        req,
                        # This field wasn't present in older
                        # implementations so provide a fallback.
                        item_md.transform_descriptors or (),
                        file_slice,
                    )

                    if req.type == LoadItemType.BYTE_IO:
                        read_bytes = io.BytesIO(transform_from.read(-1))
                        read_bytes.seek(0)
                        planner.load_bytes(req, read_bytes)
                    else:
                        if transform_from.seekable():
                            seekable = transform_from
                        else:
                            # torch.load requires a seekable input, so read the transform
                            # stream now and store the output if needed
                            seekable = io.BytesIO(transform_from.read(-1))
                            seekable.seek(0)

                        tensor = cast(
                            Tensor,
                            torch.load(
                                seekable,
                                map_location="cpu",
                                weights_only=True,
                            ),
                        )
                        tensor = narrow_tensor_by_index(
                            tensor, req.storage_offsets, req.lengths
                        )
                        target_tensor = planner.resolve_tensor(req).detach()

                        if target_tensor.size() != tensor.size():
                            raise AssertionError(
                                f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
                            )
                        target_tensor.copy_(tensor)
                        planner.commit_tensor(req, target_tensor)

        fut: Future = Future()
        fut.set_result(None)
        return fut

    def _get_metadata_path(self, rank: int | None = None) -> os.PathLike:
        filename = f"{_metadata_fn}" if rank is None else f"__{rank}{_metadata_fn}"
        return cast(Path, self.fs.concat_path(self.path, filename))

    # Implementing the abstract function in StorageReader
    def read_metadata(self, *args: Any, **kwargs: Any) -> Metadata:
        rank = kwargs.get("rank")
        path = self._get_metadata_path(rank)
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = pickle.load(metadata_file)

        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = StorageMeta()
        metadata.storage_meta.load_id = self.load_id

        return metadata

    def set_up_storage_reader(
        self, metadata: Metadata, is_coordinator: bool, *args: Any, **kwargs: Any
    ) -> None:
        self.storage_data = metadata.storage_data
        self.rank = kwargs.get("rank")
        self.use_collectives = kwargs.get("use_collectives", True)
        if self.storage_data is None:
            raise AssertionError("storage_data must not be None in metadata")

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        return plan

    def prepare_global_plan(self, plans: list[LoadPlan]) -> list[LoadPlan]:
        return plans

    @property
    def checkpoint_id(self) -> str | os.PathLike:
        """
        return the checkpoint_id that will be used to load the checkpoint.
        """
        return self.path

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        return FileSystem.validate_checkpoint_id(checkpoint_id)


class FileSystemWriter(_FileSystemWriter, BlockingAsyncStager):
    """
    Basic implementation of StorageWriter using file IO.

    This implementation makes the following assumptions and simplifications:

    * The checkpoint path is an empty or non-existing directory.
    * File creation is atomic

    The checkpoint consist of one file per write request plus
    a global `.metadata` file with the serialized metadata if rank coordination is enabled.
    a rank local `__{rank}.metadata` file with the serialized metadata if rank coordination is NOT enabled.

    """

    def __init__(
        self,
        path: str | os.PathLike,
        single_file_per_rank: bool = True,
        sync_files: bool = True,
        thread_count: int = 1,
        per_thread_copy_ahead: int = 10_000_000,
        cache_staged_state_dict: bool = False,
        overwrite: bool = True,
        _extensions: Sequence[StreamTransformExtension] | None = None,
        serialization_format: SerializationFormat = SerializationFormat.TORCH_SAVE,
    ) -> None:
        """
        Initialize the writer pointing to `path`.

        Args:
            path: directory where the checkpoint will be written to.
            single_file_per_rank: Produce one file per rank instead of one file per tensor/blob. Default to True.
            sync_files : force files to be synced to permanent storage. Default to True.
            thread_count: Number of IO threads to use to write. Default to 1.
            per_thread_copy_ahead: How many bytes to copy from the GPU ahead of saving them. Default 10Mb.
            cache_staged_state_dict: Whether to cache the staged state_dict. This option decreases staging latency
                at the cost of increased memory usage. Additionally, if this parameter is set to True, it's the expectation
                that the stager is maintained and reused for multiple dcp.async_save calls. Default to False.
            overwrite: Whether to allow overwriting existing checkpoints. Defaults to True.
            _extensions: Extensions to apply to output streams (EXPERIMENTAL)

        N. B. If sync_files is disabled, there's no guarantee that the checkpoint will be consistent in the case of a failure.
        """
        _FileSystemWriter.__init__(
            self,
            path=path,
            single_file_per_rank=single_file_per_rank,
            sync_files=sync_files,
            thread_count=thread_count,
            per_thread_copy_ahead=per_thread_copy_ahead,
            overwrite=overwrite,
            _extensions=_extensions,
            serialization_format=serialization_format,
        )
        BlockingAsyncStager.__init__(
            self,
            cache_staged_state_dict=cache_staged_state_dict,
        )

    def stage(self, state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
        """Override of AsyncStager.stage"""
        # in the async case, the state dict is already on CPU, so maintaining this
        # buffer makes no sense
        self.per_thread_copy_ahead = 0
        return super().stage(state_dict)
<|endoftext|># mypy: allow-untyped-defs
import argparse
import os
from enum import Enum
from typing import cast

import torch
import torch.distributed as dist
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torch.distributed.checkpoint.default_planner import (
    _EmptyStateDictLoadPlanner,
    DefaultLoadPlanner,
)
from torch.distributed.checkpoint.metadata import (
    Metadata,
    STATE_DICT_TYPE,
    STORAGE_TYPES,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import LoadItemType, LoadPlan, LoadPlanner
from torch.distributed.checkpoint.planner_helpers import _create_chunk_list
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from torch.distributed.checkpoint.state_dict_saver import _save_state_dict
from torch.distributed.checkpoint.storage import StorageReader
from torch.futures import Future


__all__ = [
    "dcp_to_torch_save",
    "torch_save_to_dcp",
    "BroadcastingTorchSaveReader",
    "DynamicMetaLoadPlanner",
]


class BroadcastingTorchSaveReader(StorageReader):
    """
    StorageReader for reading a Torch Save file. This reader will read the entire checkpoint
    on the coordinator rank, and then broadcast and shard each tensor to all ranks.

    . N.B. Intended to be used with DynamicMetaLoadPlanner

    .. warning::
        Current implementation only supports loading Tensors.

    >>> # xdoctest: +SKIP("undefined vars")
    >>> sd = {"mode": model}
    >>> dcp.load(
    >>>    sd,
    >>>    storage_reader=BroadcastingTorchSaveReader(),
    >>>    planner=DynamicMetaLoadPlanner(),
    >>>    checkpoint_id="path_to_model.pt"
    >>> )
    """

    def __init__(
        self,
        checkpoint_id: str | os.PathLike | None = None,
        coordinator_rank: int = 0,
    ) -> None:
        self.checkpoint_id = checkpoint_id
        self.coordinator_rank = coordinator_rank

    # pyrefly: ignore [bad-override]
    def read_metadata(self) -> Metadata:
        """Extends the default StorageReader to support building the metadata file"""
        # Metadata is built in planner.set_up_planner, since we are not actually reading metadata from
        # the disk
        return Metadata(state_dict_metadata={})

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        """
        Reads torch save data on the coordinator rank, and broadcast afterwards
        this incurrs a communication cost, but avoids having to load
        the entire checkpoint on each rank, hopefully preventing OOM issues
        """
        planner = cast(DefaultLoadPlanner, planner)

        # data is read in on the coordinator rank, and broadcast afterwards
        # this incurs a communication cost, but it avoids having to load
        # the entire checkpoint on each rank, hopefully preventing OOM issues
        # TODO: read on each host, instead of only the coordinator
        if self.is_coordinator:
            if self.checkpoint_id is None:
                raise AssertionError("checkpoint_id must be set before reading data")
            torch_state_dict = torch.load(
                self.checkpoint_id, map_location="cpu", weights_only=False
            )
            if planner.flatten_state_dict:
                torch_state_dict, _ = flatten_state_dict(torch_state_dict)
        else:
            torch_state_dict = None

        for req in plan.items:
            if req.type == LoadItemType.BYTE_IO:
                raise RuntimeError(
                    f"Non-tensor value identified at {req.storage_index.fqn}. "
                    f"At this time {type(self).__name__} only supports loading Tensors."
                )

            #  Broadcast the tensor from the coordinator rank
            if self.is_coordinator:
                pg_device = dist.distributed_c10d._get_pg_default_device()
                # pyrefly: ignore [unsupported-operation]
                tensor = torch_state_dict[req.storage_index.fqn].to(pg_device)
            else:
                tensor = torch.empty_like(planner.state_dict[req.storage_index.fqn])

            dist.broadcast(tensor, src=self.coordinator_rank, async_op=False)

            tensor = narrow_tensor_by_index(tensor, req.storage_offsets, req.lengths)
            target_tensor = planner.resolve_tensor(req).detach()
            if not target_tensor.size() == tensor.size():
                raise AssertionError(
                    f"req {req.storage_index} mismatch sizes, "
                    f"{target_tensor.size()} vs {tensor.size()}"
                )
            target_tensor.copy_(tensor)
            planner.commit_tensor(req, target_tensor)

        fut: Future = Future()
        fut.set_result(None)
        return fut

    # pyrefly: ignore [bad-override]
    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool) -> None:
        """Implementation of the StorageReader method"""
        self.is_coordinator = is_coordinator
        if self.is_coordinator:
            if not dist.get_rank() == self.coordinator_rank:
                raise AssertionError(
                    f"Coordinator rank mismatch: expected {self.coordinator_rank}, "
                    f"got {dist.get_rank()}"
                )

        if self.checkpoint_id is None:
            raise AssertionError(
                "checkpoint_id must be set before setting up storage reader"
            )

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        """Implementation of the StorageReader method"""
        return plan

    def prepare_global_plan(self, global_plan: list[LoadPlan]) -> list[LoadPlan]:
        """Implementation of the StorageReader method"""
        return global_plan

    def reset(self, checkpoint_id: str | os.PathLike | None = None) -> None:
        """Implementation of the StorageReader method"""
        self.checkpoint_id = checkpoint_id

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        """Implementation of the StorageReader method"""
        return os.path.isfile(checkpoint_id)


class DynamicMetaLoadPlanner(DefaultLoadPlanner):
    """
    Extension of DefaultLoadPlanner, which creates a new Metadata object based on the passed in state dict,
    avoiding the need to read metadata from disk. This is useful when reading formats which don't have a
    metadata file, like Torch Save files.

    . N.B. Intended to be used with BroadcastingTorchSaveReader

    .. warning::
        Current implementation only supports loading Tensors.

    >>> # xdoctest: +SKIP("undefined vars")
    >>> sd = {"mode": model}
    >>> dcp.load(
    >>>    sd,
    >>>    storage_reader=BroadcastingTorchSaveReader(),
    >>>    planner=DynamicMetaLoadPlanner(),
    >>>    checkpoint_id="path_to_model.pt"
    >>> )
    """

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        """Setups of the planner, extnding default behavior by creating the Metadata object from the state dict"""
        super().set_up_planner(state_dict, metadata, is_coordinator)

        state_dict_metadata: dict[str, STORAGE_TYPES] = {}
        for key, tensor in self.state_dict.items():
            if not torch.is_tensor(tensor):
                raise RuntimeError(
                    f"Non-tensor value identified at {key}. "
                    f"At this time {type(self).__name__} only supports loading Tensors."
                )

            state_dict_metadata[key] = TensorStorageMetadata(
                TensorProperties(dtype=tensor.dtype),
                tensor.size(),
                _create_chunk_list(tensor),
            )
        self.metadata = Metadata(state_dict_metadata=state_dict_metadata)


def dcp_to_torch_save(
    dcp_checkpoint_dir: str | os.PathLike,
    torch_save_path: str | os.PathLike,
):
    """
    Given a directory containing a DCP checkpoint, this function will convert it into a
    Torch save file.

    Args:
        dcp_checkpoint_dir: Directory containing the DCP checkpoint.
        torch_save_path: Filename to store the converted Torch save file.

    .. warning::
        To avoid OOM, it's recommended to only run this function on a single rank.
    """
    sd: STATE_DICT_TYPE = {}
    _load_state_dict(
        sd,
        storage_reader=FileSystemReader(dcp_checkpoint_dir),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    torch.save(sd, torch_save_path)


def torch_save_to_dcp(
    torch_save_path: str | os.PathLike,
    dcp_checkpoint_dir: str | os.PathLike,
):
    """
    Given the location of a torch save file, converts it into a DCP checkpoint.

    Args:
        torch_save_path: Filename of the Torch save file.
        dcp_checkpoint_dir: Directory to store the DCP checkpoint.

    .. warning::
        To avoid OOM, it's recommended to only run this function on a single rank.
    """

    state_dict = torch.load(torch_save_path, weights_only=False)
    # we don't need stateful behavior here because the expectation is anything loaded by
    # torch.load would not contain stateful objects.
    _save_state_dict(
        state_dict, storage_writer=FileSystemWriter(dcp_checkpoint_dir), no_dist=True
    )


if __name__ == "__main__":

    class FormatMode(Enum):
        TORCH_TO_DCP = "torch_to_dcp"
        DCP_TO_TORCH = "dcp_to_torch"

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        help="Conversion mode",
        choices=[m.value for m in FormatMode],
        default=FormatMode.TORCH_TO_DCP,
    )
    parser.add_argument("src", type=str, help="Path to the source model")
    parser.add_argument("dst", type=str, help="Path to the destination model")
    args = parser.parse_args()

    print(
        f"Converting checkpoint from {args.src} to {args.dst} using method: '{args.mode}'"
    )
    checkpoint_missing_warning = (
        f"No checkpoint found at {args.src}. Skipping conversion."
    )
    if args.mode == FormatMode.TORCH_TO_DCP.value:
        if os.path.isfile(args.src):
            torch_save_to_dcp(args.src, args.dst)
        else:
            print(checkpoint_missing_warning)
    elif args.mode == FormatMode.DCP_TO_TORCH.value:
        if os.path.isdir(args.src):
            dcp_to_torch_save(args.src, args.dst)
        else:
            print(checkpoint_missing_warning)
    else:
        raise ValueError(f"Unknown conversion mode: {args.mode}")
<|endoftext|># mypy: allow-untyped-defs
import json
import logging
import math
from pathlib import Path
from typing import Any

import torch
from torch.distributed.checkpoint._hf_utils import _metadata_fn
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torch.distributed.checkpoint.planner import LoadPlanner, ReadItem

from .hf_storage import HuggingFaceStorageReader


logger: logging.Logger = logging.getLogger(__name__)

__all__ = ["QuantizedHuggingFaceStorageReader"]


class QuantizedHuggingFaceStorageReader(HuggingFaceStorageReader):
    """
    Extension of HuggingFaceStorageReader that handles quantized tensors.
    Checkpoint should have the full tensor in a SafeTensor file. The quantized
    tensor should not be sharded across multiple files.

    This reader handles the dequantization of tensors during the read process,
    converting them from quantized blocks to full dequantized tensors before
    copying to the target tensor.
    """

    def __init__(
        self,
        path: str,
        thread_count: int = 1,
        target_dtype: torch.dtype = torch.float32,
        block_size: int = 128,
    ):
        """
        Initialize the HuggingFace storage reader to load quantized checkpoints

        Args:
            path: directory where the checkpoint will be read from.
            thread_count: Number of threads to use to read distributed checkpoint. Defaults to 1.
            target_dtype: Target dtype for dequantized tensor. Defaults to torch.float32.
            block_size: Fixed block size for dequantization. Defaults to 128.
        """
        super().__init__(path=path, thread_count=thread_count)

        self.target_dtype: torch.dtype = target_dtype
        self.block_size: int = block_size
        self._weight_scale_mapping: dict[str, str] = {}
        # Track which file contains each tensor
        self._weight_map: dict[str, str] = {}
        # Cache for full tensor shapes (fqn -> shape)
        self._tensor_full_shapes: dict[str, torch.Size] = {}

    def read_metadata(self) -> Any:
        metadata = super().read_metadata()

        # Load quantization metadata first.
        self._load_quantization_metadata()

        # Build a cache of FQN -> full tensor shape, correcting for quantized tensors.
        for fqn, tensor_metadata in metadata.state_dict_metadata.items():
            # Only process TensorStorageMetadata which has size attribute.
            if isinstance(tensor_metadata, TensorStorageMetadata):
                # Check if this is a MXFP4 quantized tensor that needs shape correction.
                if fqn.endswith("_blocks"):
                    # Save the quantized tensor shapes for lookup when dequantization.
                    self._tensor_full_shapes[fqn + "_quantized"] = tensor_metadata.size
                    *prefix_shape, G, B = tensor_metadata.size
                    dequantized_size = torch.Size([*prefix_shape, G * B * 2])

                    # Update the metadata with the size after dequantization.
                    # Metadata used by planner to slice state dict.
                    tensor_metadata.size = dequantized_size
                    self._tensor_full_shapes[fqn] = dequantized_size
                else:
                    self._tensor_full_shapes[fqn] = tensor_metadata.size

        return metadata

    def _load_quantization_metadata(self):
        """Load quantization metadata from the checkpoint."""
        checkpoint_path = Path(self.path)
        # Load weight mapping from index file
        index_file = checkpoint_path / _metadata_fn

        with open(index_file) as f:
            index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
            self._build_weight_scale_mapping(weight_map)

    def _build_weight_scale_mapping(self, weight_map: dict[str, str]):
        """Analyze and build weight-scale tensor pairs from weight mapping."""
        # Store the complete weight map for file location lookups.
        self._weight_map = weight_map

        for tensor_name in weight_map:
            if tensor_name.endswith(".weight_scale_inv"):
                weight_name = tensor_name.replace(".weight_scale_inv", ".weight")
                if weight_name in weight_map:
                    self._weight_scale_mapping[weight_name] = tensor_name
            # Handle MXFP4 format: _blocks and _scales.
            elif tensor_name.endswith("_scales"):
                blocks_name = tensor_name.replace("_scales", "_blocks")
                if blocks_name in weight_map:
                    self._weight_scale_mapping[blocks_name] = tensor_name

    def _process_read_request(
        self, f: Any, req: ReadItem, planner: LoadPlanner
    ) -> None:
        """Override the Helper function that processes a single read request."""
        tensor_fqn = req.storage_index.fqn

        # Check if this is a quantized tensor that needs dequantization
        if self._is_tensor_quantized(tensor_fqn):
            tensor = self._read_quantized_tensor_with_block_alignment(req, f)
        else:
            # Standard tensor reading
            slices = tuple(
                slice(offset, offset + length)
                for offset, length in zip(req.storage_offsets, req.lengths)
            )
            tensor = f.get_slice(tensor_fqn)[slices]

        target_tensor = planner.resolve_tensor(req).detach()

        if target_tensor.size() != tensor.size():
            raise AssertionError(
                f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
            )

        target_tensor.copy_(tensor)
        planner.commit_tensor(req, target_tensor)

    def _get_slice_to_block_mapping(
        self, req: ReadItem
    ) -> tuple[tuple[int, int], tuple[int, int], slice, slice]:
        """
        Calculate which blocks correspond to the requested slice.

        Args:
            req: Read request containing tensor info and required slices

        Returns:
            Tuple of (row_block_range, col_block_range, row_slice, col_slice)
        """
        # Get the slice information
        row_slice = slice(
            req.storage_offsets[0], req.storage_offsets[0] + req.lengths[0]
        )
        col_slice = slice(
            req.storage_offsets[1], req.storage_offsets[1] + req.lengths[1]
        )

        # Calculate which blocks this slice spans
        row_start_block = row_slice.start // self.block_size
        row_end_block = (row_slice.stop - 1) // self.block_size + 1  # Inclusive end

        col_start_block = col_slice.start // self.block_size
        col_end_block = (col_slice.stop - 1) // self.block_size + 1  # Inclusive end

        return (
            (row_start_block, row_end_block),
            (col_start_block, col_end_block),
            row_slice,
            col_slice,
        )

    def _dequantize_tensor_mxfp4(
        self,
        blocks: torch.Tensor,
        scales: torch.Tensor,
        req: ReadItem,
        group_start: int,
        offset_in_first_group: int,
    ) -> torch.Tensor:
        """
        Dequantize a 4D tensor using MXFP4 format.
        Adapted from openai's implementation:
        https://github.com/openai/gpt-oss/blob/8890e95919f975a490fc0ba09ffb10890ec7319d/gpt_oss/torch/weights.py#L68

        Args:
            blocks: Sliced quantized weight tensor of shape [a_slice, b_slice, groups_slice, B] in uint8
            scales: FULL scale tensor of shape [a, b, c] in uint8 (will be converted to exponents)
            req: Read request containing slice information
            group_start: The starting group index in the checkpoint
            offset_in_first_group: Offset in values within the first group

        Returns:
            Dequantized tensor matching the requested shape
        """
        # FP4 lookup table
        FP4_VALUES = [
            +0.0,
            +0.5,
            +1.0,
            +1.5,
            +2.0,
            +3.0,
            +4.0,
            +6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ]

        # blocks: [a_slice, b_slice, groups_slice, B] uint8.
        # Read slightly more groups than needed, and slice at the end.

        # Slice the scales to match the blocks dimensions.
        # [a_full, b_full, c_full] -> [a_slice, b_slice, groups_slice]
        dim0_start = req.storage_offsets[0]
        dim0_end = dim0_start + req.lengths[0]
        dim1_start = req.storage_offsets[1]
        dim1_end = dim1_start + req.lengths[1]
        num_groups = blocks.shape[2]
        scales = scales[
            dim0_start:dim0_end,
            dim1_start:dim1_end,
            group_start : group_start + num_groups,
        ]

        scales = scales.to(torch.int32) - 127

        if blocks.shape[:-1] != scales.shape:
            raise AssertionError(f"{blocks.shape=} does not match {scales.shape=}")

        lut = torch.tensor(FP4_VALUES, dtype=self.target_dtype, device=blocks.device)

        *prefix_shape, G, B = blocks.shape
        rows_total = math.prod(prefix_shape) * G

        blocks = blocks.reshape(rows_total, B)
        scales = scales.reshape(rows_total, 1)

        out = torch.empty(
            rows_total, B * 2, dtype=self.target_dtype, device=blocks.device
        )

        rows_per_chunk = 16384 * 512

        for r0 in range(0, rows_total, rows_per_chunk):
            r1 = min(r0 + rows_per_chunk, rows_total)

            blk = blocks[r0:r1]
            exp = scales[r0:r1]

            # nibble indices -> int64
            idx_lo = (blk & 0x0F).to(torch.long)
            idx_hi = (blk >> 4).to(torch.long)

            sub = out[r0:r1]
            sub[:, 0::2] = lut[idx_lo]
            sub[:, 1::2] = lut[idx_hi]

            torch.ldexp(sub, exp, out=sub)

            del idx_lo, idx_hi, blk, exp

        result = out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)

        # Slice the last dimension to match the requested range.
        if offset_in_first_group > 0 or result.shape[-1] > req.lengths[2]:
            end_offset = offset_in_first_group + req.lengths[2]
            result = result[..., offset_in_first_group:end_offset]

        return result

    def _dequantize_tensor(
        self,
        weight: torch.Tensor,
        scale_inv: torch.Tensor,
        full_tensor_shape: torch.Size,
        slice_info: tuple[tuple[int, int], tuple[int, int], slice, slice],
    ) -> torch.Tensor:
        """
        Dequantize a sliced tensor using the appropriate portion of the scale tensor.

        Args:
            weight: Sliced quantized weight tensor
            scale_inv: Full scale inverse tensor for dequantization
            full_tensor_shape: Shape of the original full tensor
            slice_info: Block mapping information from _get_slice_to_block_mapping

        Returns:
            Dequantized tensor
        """
        (row_block_range, col_block_range, row_slice, col_slice) = slice_info

        # Convert to float32 for computation
        # Certain quantized dtypes like Float8_e4m3fn
        # don't support multiplication on CPU yet in PyTorch.
        upcasted_weight = weight.to(torch.float32)

        # Create output tensor in target dtype
        dequantized = weight.detach().to(dtype=self.target_dtype, copy=True)

        # Get the actual slice boundaries
        row_start_global = row_slice.start
        row_end_global = row_slice.stop
        col_start_global = col_slice.start
        col_end_global = col_slice.stop

        # Apply scaling factors to each block that intersects with our slice
        for block_i in range(row_block_range[0], row_block_range[1]):
            for block_j in range(col_block_range[0], col_block_range[1]):
                # Calculate the block boundaries in global coordinates
                block_row_start_global = block_i * self.block_size
                block_row_end_global = min(
                    block_row_start_global + self.block_size, full_tensor_shape[0]
                )
                block_col_start_global = block_j * self.block_size
                block_col_end_global = min(
                    block_col_start_global + self.block_size, full_tensor_shape[1]
                )

                # Find the intersection of the block with our slice
                intersect_row_start = max(block_row_start_global, row_start_global)
                intersect_row_end = min(block_row_end_global, row_end_global)
                intersect_col_start = max(block_col_start_global, col_start_global)
                intersect_col_end = min(block_col_end_global, col_end_global)

                # Skip if no intersection
                if (
                    intersect_row_start >= intersect_row_end
                    or intersect_col_start >= intersect_col_end
                ):
                    continue

                # Convert global coordinates to local coordinates in the sliced tensor
                local_row_start = intersect_row_start - row_start_global
                local_row_end = intersect_row_end - row_start_global
                local_col_start = intersect_col_start - col_start_global
                local_col_end = intersect_col_end - col_start_global

                # Get the block from the sliced tensor
                block = upcasted_weight[
                    local_row_start:local_row_end, local_col_start:local_col_end
                ]

                # Apply the scale factor
                scale = scale_inv[block_i, block_j]
                block = block * scale

                # Convert block to target dtype and store
                block_converted = block.to(dtype=self.target_dtype)
                dequantized[
                    local_row_start:local_row_end, local_col_start:local_col_end
                ] = block_converted

        return dequantized

    def _is_tensor_quantized(self, tensor_fqn: str) -> bool:
        """
        Check if a tensor is a quantized.

        Args:
            tensor_fqn: Fully qualified name of the tensor

        Returns:
            True if tensor is quantized and has a corresponding scale tensor,
            False otherwise
        """
        # Skip scale tensors themselves
        if tensor_fqn.endswith((".weight_scale_inv", "_scales")):
            return False

        # Check if this weight tensor has a corresponding scale tensor
        if tensor_fqn not in self._weight_scale_mapping:
            return False

        return True

    def _read_quantized_tensor_with_block_alignment(
        self, req: ReadItem, safetensor_file: Any
    ) -> torch.Tensor:
        """
        Read a quantized tensor with block alignment.

        Args:
            req: Read request containing tensor info and required slices
            safetensor_file: Open safetensors file handle

        Returns:
            Dequantized tensor ready for use
        """
        tensor_fqn = req.storage_index.fqn
        scale_fqn = self._weight_scale_mapping[tensor_fqn]

        try:
            group_start = 0
            offset_in_first_group = 0
            if tensor_fqn.endswith("_blocks"):
                # Full tensor is a 4D MXFP4 quantized tensor: [..., G, B].
                # Each group G produces B * 2 dequantized values.
                # Checkpoint [..., G, B] -> dequantized [..., G*B*2].

                # The planner gives 3D requests based on the dequantized shape.
                # Need to figure out which groups (dimension 2 in checkpoint) to read.

                # Use the quantized checkpoint shape to get the correct B.
                *prefix_shape, B = self._tensor_full_shapes[tensor_fqn + "_quantized"]
                values_per_group = B * 2  # Each byte has 2 nibbles (4-bit values).

                # Calculate which groups we need based on the requested range in dim 2.
                # Ensure the request is in 3D.
                if len(req.storage_offsets) != 3:
                    raise AssertionError

                # Positions in dequantized space.
                dim2_start_deq = req.storage_offsets[2]
                dim2_length_deq = req.lengths[2]
                dim2_end_deq = dim2_start_deq + dim2_length_deq

                # Convert to group indices.
                group_start = dim2_start_deq // values_per_group
                group_end = (dim2_end_deq + values_per_group - 1) // values_per_group

                # Read only the necessary groups from checkpoint.
                weight_slices_4d = (
                    slice(
                        req.storage_offsets[0], req.storage_offsets[0] + req.lengths[0]
                    ),
                    slice(
                        req.storage_offsets[1], req.storage_offsets[1] + req.lengths[1]
                    ),
                    slice(group_start, group_end),
                    slice(None),  # Read all B values for each group.
                )
                quantized_tensor = safetensor_file.get_slice(tensor_fqn)[
                    weight_slices_4d
                ]

                # Also track the offset within the first group
                offset_in_first_group = dim2_start_deq - (
                    group_start * values_per_group
                )
            else:
                # 2D quantized tensor, use 2d block partition.
                weight_slices = tuple(
                    slice(offset, offset + length)
                    for offset, length in zip(req.storage_offsets, req.lengths)
                )
                quantized_tensor = safetensor_file.get_slice(tensor_fqn)[weight_slices]

            # Load the corresponding scale inverse tensor (full tensor)
            scale_file_name = self._weight_map.get(scale_fqn)
            if scale_file_name is None:
                raise ValueError(f"Scale tensor {scale_fqn} not found in weight_map")

            # Check if scale tensor is in the same file as the weight tensor
            weight_file_name = self._weight_map.get(tensor_fqn)

            if scale_file_name == weight_file_name:
                # Scale tensor is in the same file, use current handle
                scale_inv = safetensor_file.get_tensor(scale_fqn)
            else:
                # Scale tensor is in a different file, need to open it
                from safetensors import safe_open  # type: ignore[import]

                scale_file_path = Path(self.path) / scale_file_name
                with safe_open(
                    scale_file_path, framework="pt", device="cpu"
                ) as scale_file:
                    scale_inv = scale_file.get_tensor(scale_fqn)

            # Get the full tensor shape from our O(1) lookup cache
            full_tensor_shape = self._tensor_full_shapes.get(tensor_fqn)
            if full_tensor_shape is None:
                raise ValueError(f"Could not find full tensor shape for {tensor_fqn}")

            # Determine which dequantization function to use.
            if len(full_tensor_shape) == 2:
                # 2D block-wise quantization, e.g., used in deepseek v3.1
                slice_info = self._get_slice_to_block_mapping(req)
                dequantized_tensor = self._dequantize_tensor(
                    weight=quantized_tensor,
                    scale_inv=scale_inv,
                    full_tensor_shape=full_tensor_shape,
                    slice_info=slice_info,
                )
            elif tensor_fqn.endswith("_blocks"):
                # 4D with blocks along dimension 2, used in MXFP4, e.g. gpt-oss
                dequantized_tensor = self._dequantize_tensor_mxfp4(
                    blocks=quantized_tensor,
                    scales=scale_inv,
                    req=req,
                    group_start=group_start,
                    offset_in_first_group=offset_in_first_group,
                )
            else:
                raise ValueError("Unsupported quantization types")

            return dequantized_tensor

        except Exception as e:
            logger.error("Failed to read the quantized tensor!!")
            raise e
<|endoftext|>import abc
import io
import operator
from dataclasses import dataclass
from enum import auto, Enum
from functools import reduce
from typing import Any

import torch
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    STATE_DICT_TYPE,
    StorageMeta,
    TensorProperties,
)


__all__ = [
    "WriteItemType",
    "LoadItemType",
    "BytesIOWriteData",
    "TensorWriteData",
    "WriteItem",
    "ReadItem",
    "SavePlan",
    "LoadPlan",
    "SavePlanner",
    "LoadPlanner",
]


class WriteItemType(Enum):
    TENSOR = auto()
    SHARD = auto()
    BYTE_IO = auto()


class LoadItemType(Enum):
    TENSOR = auto()
    BYTE_IO = auto()


@dataclass(frozen=True)
class BytesIOWriteData:
    nbytes: int


@dataclass(frozen=True)
class TensorWriteData:
    chunk: ChunkStorageMetadata
    properties: TensorProperties
    size: torch.Size


@dataclass(frozen=True)
class WriteItem:
    """Dataclass which holds information about what needs to be written to storage."""

    index: MetadataIndex
    type: WriteItemType

    # Size of bytesIO data to be written.
    bytes_io_data: BytesIOWriteData | None = None

    # Value present if it's a tensor write
    tensor_data: TensorWriteData | None = None

    def tensor_storage_size(self) -> int | None:
        """
        Calculates the storage size of the underlying tensor, or None if this is not a tensor write.

        Returns:
            Optional[int] storage size, in bytes of underlying tensor if any.
        """
        if self.tensor_data is None:
            return None

        numels = reduce(operator.mul, self.tensor_data.size, 1)
        dtype_size = torch._utils._element_size(self.tensor_data.properties.dtype)
        return numels * dtype_size


@dataclass(frozen=True)
class ReadItem:
    # Read Item
    type: LoadItemType

    # Index into the state_dict
    dest_index: MetadataIndex
    # Offsets into destination tensor
    dest_offsets: torch.Size

    # Index into the checkpoint
    storage_index: MetadataIndex
    # Offset into the checkpoint data
    storage_offsets: torch.Size

    # Size of the hypercube to copy
    lengths: torch.Size


@dataclass(frozen=True)
class SavePlan:
    items: list[WriteItem]
    storage_data: Any = None
    planner_data: Any = None
    # This is used to indicate that the ranks should
    # use the cached plans to write data instead.
    usable: bool = True


@dataclass
class LoadPlan:
    items: list[ReadItem]
    storage_data: Any = None
    planner_data: Any = None


class SavePlanner(abc.ABC):
    """
    Abstract class defining the protocol used by save_state_dict to plan the save process.

    SavePlanners are stateful objects that can be used to customize the whole save process.

    SavePlanner acts as an access proxy to the state_dict, so any transformation done to it
    will be visible to the whole process.

    A planner subclass can expect the following sequence of calls during save_state_dict:

    1) set_up_planner - called on all ranks.
        Signals the start of a checkpoint save.

    2) create_local_plan - called on all ranks.
        Process the state_dict and produces a `SavePlan` that will be sent for global planning.

    3) create_global_plan - called on the coordinator rank only.
        Takes the SavePlan from all ranks and make any global decision.

    4) finish_plan - called on all ranks.
        This gives each rank a chance to adjust to global planning decisions.

    5) resolve_data - called multiple times on each rank
        Lookups a value on the `state_dict` for the storage layer to write.

    Users are recommended to extend DefaultSavePlanner instead of this interface directly as
    most changes can be expressed by changes in a single method.

    There are 3 usual patterns of extension:

    Rewriting state_dict. This is the simplest way to extend the save process as it
    doesn't require understanding the intricacies of how SavePlan works:

    >>> # xdoctest: +SKIP("undefined vars")
    >>> class RenamePlanner(DefaultSavePlanner):
    >>>     def set_up_planner(
    >>>         self,
    >>>         state_dict: STATE_DICT_TYPE,
    >>>         storage_meta: Optional[StorageMeta],
    >>>         is_coordinator: bool,
    >>>     ) -> None:
    >>> # prefix all keys with `foo_``
    >>>         super().set_up_planner({"foo_" + k: v for k, v in state_dict.items()}, storage_meta, is_coordinator)

    Modifying local plan and lookup in tandem. This is useful when fine control of how data is persisted

    >>> # xdoctest: +SKIP("undefined vars")
    >>> class FP16Planner(DefaultSavePlanner):
    >>>     def create_local_plan(self):
    >>>         plan = super().create_local_plan()
    >>>         for p in plan:
    >>>             if p.tensor_data is not None:
    >>>                 p.tensor_data.properties.dtype = torch.float16
    >>>         return plan
    >>>
    >>>     def resolve_data(self, write_item):
    >>>         item = super().resolve_data(write_item)
    >>>         return item if write_item.type == WriteItemType.BYTE_IO else item.to(torch.float16)

    Using the global planning step to make central decisions that can't be made individually by each rank

    >>> # xdoctest: +SKIP("undefined vars")
    >>> from itertools import zip_longest
    >>> from dataclasses import replace
    >>> class DDPLoadBalancingPlanner(DefaultSavePlanner):
    >>> # This uses the default local plan behavior of having all non-sharded writes in rank 0
    >>> # This sample doesn't handle ShardedTensors
    >>>     def create_global_plan(self, all_plans):
    >>>         iters = [iter(all_plans[0].items)] * len(all_plans)
    >>>         items_per_rank = [
    >>>             [item for item in items if item is not None]
    >>>             for items in zip(*zip_longest(*iters), strict=True)
    >>>         ]
    >>>         all_plans = [
    >>>             replace(plan, items=items)
    >>>             for plan, items in zip(all_plans, items_per_rank, strict=True)
    >>>         ]
    >>>         return super().create_global_plan(all_plans)

    Finally, some planners need to save additional metadata in the checkpoint, this is
    accomplished by having each rank contribute their data items in the local plan and
    the global planner aggregate them:

    >>> # xdoctest: +SKIP("undefined vars")
    >>> class SaveExtraDataPlanner(DefaultSavePlanner):
    >>>     def create_local_plan(self) -> SavePlan:
    >>>         plan = super().create_local_plan()
    >>>         return replace(plan, planner_data="per-rank-data")
    >>>
    >>>     def create_global_plan(self, all_plans: List[SavePlan]) -> Tuple[List[SavePlan], Metadata]:
    >>>         global_plan, metadata = super().create_global_plan(all_plans)
    >>>         merged_data = [p.planner_data for p in global_plan]
    >>>         metadata = replace(metadata, planner_data=merged_data)
    >>>         return global_plan, metadata
    """

    # Save plan for the current rank as computed by `create_local_plan` API
    # Cached on the local rank.
    _cached_save_plan: dict[str, SavePlan] = {}
    # Final save plan for the current rank.
    # This is created by merging the plan created by `create_local_plan` API
    # and the result of `create_global_plan` for the given rank.
    # This is the final plan computed by the `finish_plan` API that gets
    # sent to the `write_data`.
    # Cached on the local rank.
    _cached_final_save_plan: dict[str, SavePlan] = {}
    # Collection of all the local plans from all the ranks.
    # This is the input to the `create_global_plan` API.
    # Cached on the coordinator rank.
    _cached_all_plans: dict[str, list[SavePlan]] = {}
    # Global checkpoint plan as computed by `create_global_plan` API.
    # Cached on the coordinator rank.
    _cached_global_plan: dict[str, list[SavePlan]] = {}
    # Metadata for the global checkpoint plan as computed by `create_global_plan` API.
    # Cached on the coordinator rank.
    _cached_metadata: dict[str, Metadata] = {}

    @abc.abstractmethod
    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        storage_meta: StorageMeta | None = None,
        is_coordinator: bool = False,
    ) -> None:
        """
        Initialize this planner to save ``state_dict``.

        Implementations should save those values as they won't be provided later in the save process.

        This is called on all ranks.
        """

    @abc.abstractmethod
    def create_local_plan(self) -> SavePlan:
        """
        Compute the save plan for the current rank.

        This will be aggregated and passed to create_global_plan.
        Planner specific data can be passed through SavePlan::planner_data.

        This is called on all ranks.
        """

    @abc.abstractmethod
    def create_global_plan(
        self, all_plans: list[SavePlan]
    ) -> tuple[list[SavePlan], Metadata]:
        """
        Compute the global checkpoint plan and return the local plan of each rank.

        This is called on the coordinator rank only.
        """

    @abc.abstractmethod
    def finish_plan(self, new_plan: SavePlan) -> SavePlan:
        """
        Merge the plan created by `create_local_plan` and the result of `create_global_plan`.

        This is called on all ranks.
        """

    @abc.abstractmethod
    def resolve_data(self, write_item: WriteItem) -> torch.Tensor | io.BytesIO:
        """
        Transform and prepare ``write_item`` from ``state_dict`` for storage, ensuring idempotency and thread-safety.

        Lookup the object associated with ``write_item`` in ``state_dict`` and apply any
        transformation (such as serialization) prior to the storage layer consuming it.

        Called on each rank multiple times, at least once per WriteItem in the final SavePlan.

        This method should be idempotent and thread-save. StorageWriter implementations
        are free to call it as frequently as they need.

        Any transformation that allocates memory should be lazily done when his method
        is called in order to reduce peak memory required by checkpointing.

        When returning tensors, they can be on any device or format, they can be views too.
        It's the storage layer responsibility to figure out how to save them.
        """


class LoadPlanner:
    """
    Abstract class defining the protocol used by load_state_dict to plan the load process.

    LoadPlanner are stateful objects that can be used to customize the whole load process.

    LoadPlanner acts as an access proxy to the state_dict, so any transformation done to it
    will be visible to the whole process.

    A planner subclass can expect the following sequence of calls during load_state_dict:

    1) set_up_planner - called on all ranks.
        Signals the start of loading a checkpoint.

    2) create_local_plan - called on all ranks.
        Process the state_dict and produces a `LoadPlan` that will be sent for global planning.

    3) create_global_plan - called on the coordinator rank only.
        Takes the LoadPlan from all ranks and make any global decision.

    4) load_bytes - called multiple times on each rank
        This is called once per non-tensor value in state_dict.

    5) resolve_tensor and commit_tensor - called multiple times on each rank
        They are called in pair for each Tensor value in state_dict.

    Users are recommended to extend DefaultLoadPlanner instead of this interface directly as
    most changes can be expressed by changes in a single method.

    There are two usual patterns of extension:

    Rewriting state_dict. This is the simplest way to extend the load process as it
    doesn't require understanding the intricacies of how LoadPlan works. We need
    to keep a reference to the original state_dict as load happens in place so
    we need to be able to perform it in place

    >>> # xdoctest: +SKIP("undefined vars")
    >>> class RenamePlanner(DefaultLoadPlanner):
    >>>     def set_up_planner(
    >>>         self,
    >>>         state_dict: STATE_DICT_TYPE,
    >>>         metadata: Metadata,
    >>>         is_coordinator: bool,
    >>>     ) -> None:
    >>>         self.original_state_dict = state_dict
    >>>         state_dict = {"foo_" + k: v for k, v in state_dict.items()}
    >>>
    >>>         if self.flatten_sharded_tensors:
    >>>             state_dict = _flatten_sharded_tensors(state_dict)
    >>>
    >>>         if self.flatten_state_dict:
    >>>             state_dict, self.mappings = flatten_state_dict(state_dict)
    >>>
    >>>         self.state_dict = state_dict
    >>>         self.metadata = metadata
    >>>         self.is_coordinator = is_coordinator
    >>>
    >>>     def load_bytes(self, read_item, value):
    >>> # Remove the "foo_" prefix
    >>>         self.original_state_dict[read_item.dest_index.fqn[4:]] = torch.load(value, weights_only=False)


    Modifying resolve_tensor and commit_tensor to handle load time transformation.

    >>> # xdoctest: +SKIP("undefined vars")
    >>> class MetaModelMaterialize(DefaultSavePlanner):
    >>>     def resolve_tensor(self, read_item):
    >>>         tensor = super().resolve_tensor(read_item)
    >>>         return torch.empty_like(tensor, device="cpu")
    >>>
    >>>     def commit_tensor(self, read_item, tensor):
    >>>         self.state_dict[read_item.dest_index.fqn] = tensor
    """

    @abc.abstractmethod
    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        """
        Initialize this instance to load data into ``state_dict``.

        . N.B. This is called on every rank.
        """

    @abc.abstractmethod
    def create_local_plan(self) -> LoadPlan:
        """
        Create a LoadPlan based on state_dict and metadata provided by set_up_planner.

        . N.B. This is called on every rank.
        """

    @abc.abstractmethod
    def create_global_plan(self, global_plan: list[LoadPlan]) -> list[LoadPlan]:
        """
        Compute the global load plan and return plans for each rank.

        . N.B. This is called on the coordinator rank only
        """

    @abc.abstractmethod
    def finish_plan(self, central_plan: LoadPlan) -> LoadPlan:
        """Accept the plan from coordinator and return final LoadPlan."""

    @abc.abstractmethod
    def load_bytes(self, read_item: ReadItem, value: io.BytesIO) -> None:
        """
        Load the item described by ``read_item``and ``value``.

        This method is expected to modify in-place the underlying state_dict.

        The contents of ``value`` are defined by the SavePlanner used to produce
        the checkpoint being loaded.
        """

    def resolve_bytes(self, read_item: ReadItem) -> io.BytesIO:
        """
        Return the BytesIO to be used by the StorageReader to load `read_item`.

        The BytesIO should alias with one on the underlying state_dict as StorageReader will replace its contents.
        """
        raise NotImplementedError("LoadPlanner.resolve_bytes is not implemented")

    @abc.abstractmethod
    def resolve_tensor(self, read_item: ReadItem) -> torch.Tensor:
        """
        Return the tensor described by ``read_item`` to be used by the StorageReader to load `read_item`.

        The tensor should alias with one on the underlying state_dict as StorageReader will replace its contents.
        If, for any reason, that's not possible, the planner can use the ``commit_tensor`` method to copy the data
        back to the one in state_dict.
        """

    @abc.abstractmethod
    def commit_tensor(self, read_item: ReadItem, tensor: torch.Tensor) -> None:
        """
        Call once the StorageReader finished loading data into ``tensor``.

        The provided tensor is the same one returned by the call to ``resolve_tensor``.
        This method is only needed if this LoadPlanner needs to post process ``tensor`` prior to
        copying it back to the one in the state_dict.

        The contents of tensor will follow its device synchronization model.
        """
<|endoftext|>from torch.distributed.checkpoint.metadata import ChunkStorageMetadata


__all__: list[str] = []


def _check_shard_metadata_pair_overlap(
    shard1: ChunkStorageMetadata, shard2: ChunkStorageMetadata
) -> bool:
    """Check if two shards overlap."""
    # For each dim of each shard, check if one shard resides on the other
    # end of second shard with respect to that dim. As an example for a 2D
    # shard, we would check if one shard is above or on the left of the
    # other shard.
    ndims = len(shard1.offsets)
    for i in range(ndims):
        if shard1.offsets[i] >= shard2.offsets[i] + shard2.sizes[i]:
            return False
        if shard2.offsets[i] >= shard1.offsets[i] + shard1.sizes[i]:
            return False

    return True


def _shards_get_overlap_region_wrt_saved_tensor(
    saved_shard: ChunkStorageMetadata, current_shard: ChunkStorageMetadata
) -> list[tuple[int, int, int, int]]:
    """
    Return the overlapping region between saved_shard and current_shard.

    The returned list has the same number of elements as the tensor's dimension.
    For each element, we produce a tuple with the following contents:
        (dimension, `saved_shard` offset, `current_shard` offset, length)

    Offsets are relative to each shard.
    """
    narrows = []
    for dim, (
        saved_shard_offset,
        current_shard_offset,
        saved_shard_size,
        current_shard_size,
    ) in enumerate(
        zip(
            saved_shard.offsets,
            current_shard.offsets,
            saved_shard.sizes,
            current_shard.sizes,
        )
    ):
        min_range_end = min(
            saved_shard_offset + saved_shard_size,
            current_shard_offset + current_shard_size,
        )

        length = min_range_end - max(current_shard_offset, saved_shard_offset)

        if saved_shard_offset > current_shard_offset:
            offset_for_saved_tensor = 0
            offset_for_current_tensor = saved_shard_offset - current_shard_offset
        else:
            offset_for_saved_tensor = current_shard_offset - saved_shard_offset
            offset_for_current_tensor = 0

        narrows.append(
            (dim, offset_for_saved_tensor, offset_for_current_tensor, length)
        )

    return narrows
<|endoftext|>"""
Factory functions for creating checkpointer instances with sensible defaults.

This module provides high-level factory functions that simplify the creation
of checkpointer instances by automatically handling component initialization
and configuration with reasonable defaults.
"""

from collections.abc import Callable
from typing import Any

import torch.distributed as dist

from .barriers import create_barrier_from_config
from .checkpoint_process import CheckpointProcess
from .checkpoint_reader import CheckpointReader
from .checkpoint_writer import CheckpointWriter, CheckpointWriterConfig, WriterHook
from .checkpointer import AsyncCheckpointer, SyncCheckpointer
from .config import CheckpointerConfig
from .staging import DefaultStager
from .types import RankInfo


def _get_default_rank_info() -> RankInfo:
    """
    Get default rank information from the current distributed environment.

    Returns:
        RankInfo: Rank information from the default process group if initialized,
                 otherwise single-rank fallback.
    """
    if dist.is_initialized():
        return RankInfo(
            global_world_size=dist.get_world_size(),
            global_rank=dist.get_rank(),
        )
    else:
        # Single-rank fallback
        return RankInfo(global_world_size=1, global_rank=0)


def default_subprocess_init_fn(*_: Any) -> None:
    """Default subprocess initialization function (no-op)."""


def default_writer_init_fn(rank_info: RankInfo) -> CheckpointWriter:
    """Default checkpoint writer initialization function."""
    return CheckpointWriter(
        config=CheckpointWriterConfig(),
        rank_info=rank_info,
    )


def make_sync_checkpointer(
    config: CheckpointerConfig = CheckpointerConfig(),
    rank_info: RankInfo | None = None,
    commit_hook: WriterHook | None = None,
) -> SyncCheckpointer:
    """
    Factory function to create a SyncCheckpointer instance with sensible defaults.

    This function creates a synchronous checkpointer with default components, automatically
    detecting rank information from the default process group if available, and using the
    provided component configurations.

    Args:
        config: CheckpointerConfig containing component-specific configurations
               (writer_config, staging_config, process_config). Defaults to CheckpointerConfig().
        rank_info: RankInfo for distributed training. Defaults to auto-detection from
                  the default PyTorch distributed process group if initialized, otherwise
                  falls back to single-rank (world_size=1, rank=0).
        commit_hook: Optional hook for custom actions before and after checkpoint commits.

    Returns:
        SyncCheckpointer: A configured synchronous checkpointer instance.

    Examples:
        # Simplest usage - auto-detect rank, default config
        checkpointer = make_sync_checkpointer()

        # Explicit rank configuration
        checkpointer = make_sync_checkpointer(
            rank_info=RankInfo(global_world_size=4, global_rank=0)
        )

        # Disable barrier
        from .barriers import BarrierConfig
        config = CheckpointerConfig(barrier_config=BarrierConfig(barrier_type=None))
        checkpointer = make_sync_checkpointer(config=config)
    """
    if rank_info is None:
        rank_info = _get_default_rank_info()

    reader = CheckpointReader(
        rank_info=rank_info,
    )

    barrier = create_barrier_from_config(config.barrier_config)

    writer = CheckpointWriter(
        config=config.writer_config,
        rank_info=rank_info,
        barrier=barrier,
        commit_hook=commit_hook,
    )

    return SyncCheckpointer(
        writer=writer,
        reader=reader,
    )


def make_async_checkpointer(
    config: CheckpointerConfig = CheckpointerConfig(),
    rank_info: RankInfo | None = None,
    subprocess_init_fn: Callable[..., None] = default_subprocess_init_fn,
    subprocess_init_args: tuple[Any, ...] = (),
    checkpoint_writer_init_fn: Callable[..., CheckpointWriter] = default_writer_init_fn,
    checkpoint_writer_init_args: dict[str, Any] | None = None,
) -> AsyncCheckpointer:
    """
    Factory function to create an AsyncCheckpointer instance with sensible defaults.

    This function creates an asynchronous checkpointer using the provided configuration,
    automatically detecting rank information if not provided.

    Args:
        config: CheckpointerConfig containing component-specific configurations.
        rank_info: RankInfo for distributed training. Defaults to auto-detection.
        subprocess_init_fn: Function to initialize the subprocess. Defaults to no-op.
        subprocess_init_args: Arguments to pass to subprocess_init_fn.
        checkpoint_writer_init_fn: Function to create CheckpointWriter instance.
        checkpoint_writer_init_args: Arguments to pass to checkpoint_writer_init_fn.

    Returns:
        AsyncCheckpointer: A configured asynchronous checkpointer instance.

    Examples:
        # Create with default config
        checkpointer = make_async_checkpointer()

        # Create with custom init functions
        checkpointer = make_async_checkpointer(
            subprocess_init_fn=my_subprocess_init_fn,
            checkpoint_writer_init_fn=my_writer_init_fn
        )
    """
    if rank_info is None:
        rank_info = _get_default_rank_info()

    reader = CheckpointReader(
        rank_info=rank_info,
    )

    checkpoint_stager = DefaultStager(
        config=config.staging_config,
    )

    checkpoint_writer_init_args = checkpoint_writer_init_args or {}

    checkpoint_process = CheckpointProcess(
        rank_info=rank_info,
        config=config.process_config,
        subprocess_init_fn=subprocess_init_fn,
        subprocess_init_args=subprocess_init_args,
        checkpoint_writer_init_fn=checkpoint_writer_init_fn,
        checkpoint_writer_init_args=checkpoint_writer_init_args,
    )

    return AsyncCheckpointer(
        checkpoint_stager=checkpoint_stager,
        checkpoint_process=checkpoint_process,
        reader=reader,
    )
<|endoftext|>import abc
import logging
from concurrent.futures import Future
from typing import Any, TypeVar

from .checkpoint_process import CheckpointProcess
from .checkpoint_reader import CheckpointReader
from .checkpoint_writer import CheckpointWriter
from .staging import CheckpointStager
from .types import STATE_DICT
from .utils import wrap_future


logger = logging.getLogger(__name__)

LOG_INTERVAL = 60
T = TypeVar("T")


class Checkpointer(abc.ABC):
    """
    WARNING: This class is experimental, and is created to validate certain ideas,
    and is subjected to change or deprecation and we strong discourage any usages at
    this time.

    Abstract base class that defines the API for checkpointing.

    This class defines the interface for coordinating the writing and loading of model
    state dictionaries to and from storage. It provides abstract methods to save and load model states
    with support for both synchronous and asynchronous operations.

    Concrete implementations of this class must implement all the abstract methods.
    """

    @abc.abstractmethod
    def save(
        self,
        path: str,
        state_dict: STATE_DICT,
        **kwargs: dict[str, Any],
    ) -> tuple[Future, Future] | None:
        """
        Save a state dictionary to storage.

        Args:
            path: The path where the checkpoint should be saved.
            state_dict: The state dictionary to save.
            **kwargs: Additional keyword arguments to pass to the writer.

        Returns:
            For synchronous implementations: None
            For asynchronous implementations: tuple of (stage_future, write_future)
                                            representing the staging and writing operations.
        """

    @abc.abstractmethod
    def load(
        self,
        path: str,
        state_dict: STATE_DICT | None = None,
        *,
        default_map_location: Any = None,
        strict: bool = False,
        **kwargs: dict[str, Any],
    ) -> STATE_DICT:
        """
        Load a state dictionary from storage.

        Args:
            path: The path from which to load the checkpoint.
            state_dict: Optional state dictionary to update with loaded values.
                        If provided, only keys in this dictionary will be loaded.
            default_map_location: Device mapping function or device name for relocating tensors.
            strict: If True, raises an error when there are missing keys in the checkpoint.
            **kwargs: Additional keyword arguments to pass to the reader.

        Returns:
            The loaded state dictionary.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """
        Close the checkpointer and release any resources.

        This method should be called when the checkpointer is no longer needed to ensure
        proper cleanup of resources.
        """


class SyncCheckpointer(Checkpointer):
    """
    Synchronous implementation of Checkpointer.

    This class coordinates the writing and loading of model state dictionaries to and from storage
    using only synchronous operations. It provides a simple, efficient interface for checkpoint
    operations without async overhead.

    Attributes:
        _writer: CheckpointWriter for writing state dictionaries to storage.
        _reader: CheckpointReader for reading state dictionaries from storage.

    Example:
        checkpointer = SyncCheckpointer(writer=writer, reader=reader)
        checkpointer.save(state_dict, path)
        loaded_state_dict = checkpointer.load(path)
    """

    def __init__(
        self,
        writer: CheckpointWriter,
        reader: CheckpointReader,
    ):
        """
        Initialize a synchronous checkpointer.

        Args:
            writer: CheckpointWriter for writing checkpoints to storage.
            reader: CheckpointReader for reading checkpoints from storage.
        """
        self._writer = writer
        self._reader = reader

    def save(
        self,
        path: str,
        state_dict: STATE_DICT,
        **kwargs: dict[str, Any],
    ) -> tuple[Future, Future] | None:
        """
        Save a state dictionary to storage synchronously.

        Args:
            path: The path where the checkpoint should be saved.
            state_dict: The state dictionary to save.
            **kwargs: Additional keyword arguments to pass to the writer.

        Returns:
            Always returns None as operations are synchronous.

        Example:
            checkpointer.save("/path/to/checkpoint", state_dict)
        """
        logger.debug("Saving checkpoint synchronously to %s", path)
        self._writer.write(path, state_dict, **kwargs)
        return None

    def load(
        self,
        path: str,
        state_dict: STATE_DICT | None = None,
        *,
        default_map_location: Any = None,
        strict: bool = False,
        **kwargs: dict[str, Any],
    ) -> STATE_DICT:
        """
        Load a state dictionary from storage.

        Args:
            path: The path from which to load the checkpoint.
            state_dict: Optional state dictionary to update with loaded values.
                        If provided, only keys in this dictionary will be loaded.
            default_map_location: Device mapping function or device name for relocating tensors.
            strict: If True, raises an error when there are missing keys in the checkpoint.
            **kwargs: Additional keyword arguments to pass to the reader.

        Returns:
            The loaded state dictionary.

        Raises:
            RuntimeError: If strict=True and there are missing keys in the checkpoint.
            FileNotFoundError: If the checkpoint file is not found.
        """
        logger.info("Loading checkpoint from %s", path)

        loaded_state_dict, missing_keys = self._reader.read(
            path=path,
            state_dict=state_dict,
            map_location=default_map_location,
            **kwargs,
        )
        if strict and missing_keys is not None and missing_keys != []:
            raise RuntimeError(f"Checkpoint at {path} is missing keys: {missing_keys}")
        return loaded_state_dict

    def close(self) -> None:
        """
        Close the checkpointer and release any resources.

        This method should be called when the checkpointer is no longer needed to ensure
        proper cleanup of resources.
        """
        self._writer.close()
        logger.info("SyncCheckpointer closed")


class AsyncCheckpointer(Checkpointer):
    """
    Asynchronous implementation of Checkpointer.

    This class coordinates the writing and loading of model state dictionaries to and from storage
    using asynchronous operations for saving. It provides efficient async checkpoint operations
    with staging and background writing capabilities.

    Attributes:
        _reader: CheckpointReader for reading state dictionaries from storage.
        _checkpoint_stager: Stager for async operations.
        _checkpoint_process: Process for async operations.
        _write_future: Future representing the ongoing async write operation.

    Example:
        checkpointer = AsyncCheckpointer(
            reader=reader,
            checkpoint_stager=stager,
            checkpoint_process=process
        )
        stage_future, write_future = checkpointer.save(state_dict, path)
        # ... do other work ...
        write_future.result()  # Wait for completion
    """

    def __init__(
        self,
        checkpoint_stager: CheckpointStager,
        checkpoint_process: CheckpointProcess,
        reader: CheckpointReader,
    ):
        """
        Initialize an asynchronous checkpointer.

        Args:
            checkpoint_stager: Stager for async operations.
            checkpoint_process: Process for async operations.
            reader: CheckpointReader for reading checkpoints from storage.
        """
        self._reader = reader
        self._checkpoint_stager = checkpoint_stager
        self._checkpoint_process = checkpoint_process
        self._write_future: Future[Any] | None = None

    def save(
        self,
        path: str,
        state_dict: STATE_DICT,
        **kwargs: Any,
    ) -> tuple[Future, Future] | None:
        """
        Save a state dictionary to storage asynchronously.

        Args:
            path: The path where the checkpoint should be saved.
            state_dict: The state dictionary to save.
            **kwargs: Additional keyword arguments to pass to the stager and writer.

        Returns:
            A tuple of (stage_future, write_future) representing the staging and writing operations.

        Example:
            stage_future, write_future = checkpointer.save("/path/to/checkpoint", state_dict)
            # ... do other work ...
            write_future.result()  # Wait for completion
        """
        logger.info(
            "Initiating checkpoint save to %s. Will wait for prev checkpoints to complete.",
            path,
        )
        # Wait for previous checkpoint ops to finish and verify they are successful
        if self._write_future is not None:
            self._write_future.result()

        logger.debug("Starting state dictionary staging")
        staging_result = self._checkpoint_stager.stage(
            state_dict=state_dict,
            **kwargs,
        )

        logger.debug("Starting checkpoint write to %s", path)
        self._write_future = self._checkpoint_process.write(
            staging_result, path, **kwargs
        )
        logger.info("Checkpoint save to %s initiated", path)

        # Return futures for the staging and writing operations
        if self._write_future is not None:
            return wrap_future(staging_result), self._write_future
        else:
            # This should not happen since we just assigned _write_future above
            raise RuntimeError("Write future is unexpectedly None")

    def load(
        self,
        path: str,
        state_dict: STATE_DICT | None = None,
        *,
        default_map_location: Any = None,
        strict: bool = False,
        **kwargs: Any,
    ) -> STATE_DICT:
        """
        Load a state dictionary from storage.

        Loading is always performed synchronously, even in AsyncCheckpointer.

        Args:
            path: The path from which to load the checkpoint.
            state_dict: Optional state dictionary to update with loaded values.
                        If provided, only keys in this dictionary will be loaded.
            default_map_location: Device mapping function or device name for relocating tensors.
            strict: If True, raises an error when there are missing keys in the checkpoint.
            **kwargs: Additional keyword arguments to pass to the reader.

        Returns:
            The loaded state dictionary.

        Raises:
            RuntimeError: If strict=True and there are missing keys in the checkpoint.
            FileNotFoundError: If the checkpoint file is not found.
        """
        logger.info("Loading checkpoint from %s", path)

        loaded_state_dict, missing_keys = self._reader.read(
            path=path,
            state_dict=state_dict,
            map_location=default_map_location,
            **kwargs,
        )
        if strict and missing_keys is not None and missing_keys != []:
            raise RuntimeError(f"Checkpoint at {path} is missing keys: {missing_keys}")
        return loaded_state_dict

    def close(self) -> None:
        """
        Close the checkpointer and release any resources.

        This method should be called when the checkpointer is no longer needed to ensure
        proper cleanup of async resources.
        """
        self._checkpoint_stager.close()
        self._checkpoint_process.close()
        logger.info("AsyncCheckpointer closed")
<|endoftext|>import logging
import os
import traceback
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from multiprocessing.connection import Connection
from typing import Any

import torch.multiprocessing as mp
from torch.multiprocessing.spawn import ProcessExitedException

from .checkpoint_writer import CheckpointWriter
from .types import RankInfo, STATE_DICT


logger = logging.getLogger(__name__)


@dataclass
class CheckpointProcessConfig:
    """
    Configuration options for the CheckpointProcess.

    This class provides configuration options for the checkpoint process,
    including initialization functions, timeouts, and writer configuration.

    Attributes:
        subprocess_init_timeout_secs: Maximum time in seconds to wait for subprocess initialization.
        subprocess_shutdown_timeout_secs: Maximum time in seconds to wait for subprocess shutdown.
    """

    subprocess_init_timeout_secs: int = 30
    subprocess_shutdown_timeout_secs: int = 60


class RequestType(Enum):
    PING = "ping"
    WRITE_CHECKPOINT = "write_checkpoint"
    TERMINATE_PROCESS = "exit"


@dataclass
class WorkerRequest:
    """
    A dataclass for storing the command to be sent to the worker process.
    Note: This relies on pickling to send the command to the worker process. Handle
    backward compatibility accordingly.
    """

    request_type: RequestType
    payload: dict[str, Any]


@dataclass
class WorkerResponse:
    request_type: RequestType
    success: bool
    error_msg: str | None = None
    payload: dict[str, Any] | None = None


class CheckpointProcess:
    """
    A checkpoint writer that writes checkpoints to a remote process.
    """

    def __init__(
        self,
        rank_info: RankInfo,
        config: CheckpointProcessConfig,
        subprocess_init_fn: Callable[[Any], None],
        subprocess_init_args: tuple[Any, ...],
        checkpoint_writer_init_fn: Callable[..., CheckpointWriter],
        checkpoint_writer_init_args: dict[str, Any],
    ):
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._rank_info = rank_info
        self._config = config
        self._subprocess_init_fn = subprocess_init_fn
        self._subprocess_init_args = subprocess_init_args
        self._checkpoint_writer_init_fn = checkpoint_writer_init_fn
        self._checkpoint_writer_init_args = checkpoint_writer_init_args
        self.process = None
        self._parent_end: Connection | None = None
        self._child_end: Connection | None = None

        self.process_creation_future = self._executor.submit(
            self._create_subprocess,
            config,
        )

    def _create_subprocess(
        self,
        config: CheckpointProcessConfig,
    ) -> None:
        logger.info(
            "Creating checkpoint subprocess for rank %d", self._rank_info.global_rank
        )

        spawn_context = mp.get_context("spawn")
        self._parent_end, child_end = spawn_context.Pipe()

        # Known workaround for https://github.com/pytorch/pytorch/issues/37377
        os.environ["MKL_SERVICE_FORCE_INTEL"] = "GNU"

        logger.debug("Spawning subprocess for rank_info=%s", self._rank_info)
        self.process = mp.spawn(
            fn=CheckpointProcess._subprocess,
            args=(
                self._rank_info,
                child_end,
                self._subprocess_init_fn,
                self._subprocess_init_args,
                self._checkpoint_writer_init_fn,
                self._checkpoint_writer_init_args,
            ),
            nprocs=1,
            join=False,
            daemon=True,
        )

        # close the child end of the pipe so recv on it will fail
        # fast when the child process is terminated unexpectedly.
        child_end.close()
        self._send(
            request_type=RequestType.PING,
            payload={},
        )

        logger.debug(
            "Waiting for checkpoint subprocess to initialize (timeout: %ds)",
            config.subprocess_init_timeout_secs,
        )

        # wait for the timeout or a response from subprocess
        if self._parent_end is None:
            raise AssertionError("Parent end of pipe should be initialized")
        if not self._parent_end.poll(timeout=config.subprocess_init_timeout_secs):
            msg = f"Timed out after {config.subprocess_init_timeout_secs}s waiting for checkpoint subprocess to initialize"
            logger.error(msg)
            raise TimeoutError(msg)

        self._recv()
        logger.info("Checkpoint subprocess initialized successfully")

    @staticmethod
    def _subprocess(
        sub_rank: int,
        rank_info: RankInfo,
        parent_pipe: Connection,
        subprocess_init_fn: Callable[[Any], None],
        subprocess_init_args: tuple[Any, ...],
        checkpoint_writer_init_fn: Callable[..., CheckpointWriter],
        checkpoint_writer_init_args: dict[str, Any],
    ) -> None:
        logger.debug(
            "Checkpoint subprocess started for rank %d/%d (PID: %d)",
            rank_info.global_rank,
            rank_info.global_world_size,
            os.getpid(),
        )

        if sub_rank != 0:
            raise AssertionError("We need only one checkpointer per parent training")
        request = WorkerRequest(request_type=RequestType.PING, payload={})

        try:
            # Calling initialize callback, so we can perform app-specific initialization of the subprocess.
            subprocess_init_fn(*subprocess_init_args)

            # Initialize checkpoint writer - automatically include rank_info in init_args
            writer_init_args = dict(checkpoint_writer_init_args)
            if "rank_info" not in writer_init_args:
                writer_init_args["rank_info"] = rank_info
            checkpoint_writer = checkpoint_writer_init_fn(**writer_init_args)

            while True:
                request = parent_pipe.recv()

                if request.request_type == RequestType.PING:
                    parent_pipe.send(
                        WorkerResponse(request_type=RequestType.PING, success=True)
                    )
                elif request.request_type == RequestType.WRITE_CHECKPOINT:
                    path = request.payload["path"]
                    logger.info("Writing checkpoint to %s", path)

                    checkpoint_writer.write(
                        path=path,
                        state_dict=request.payload["state_dict"],
                        **request.payload["kwargs"],
                    )

                    logger.info("Checkpoint written successfully to %s", path)
                    parent_pipe.send(
                        WorkerResponse(RequestType.WRITE_CHECKPOINT, success=True)
                    )
                elif request.request_type == RequestType.TERMINATE_PROCESS:
                    logger.debug("Received termination request.")
                    parent_pipe.send(
                        WorkerResponse(RequestType.TERMINATE_PROCESS, success=True)
                    )
                    logger.info("Subprocess terminated gracefully")
                    break
                else:
                    error_msg = f"Unknown request type: {request.request_type}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

        except Exception as e:
            error_text = traceback.format_exc()
            logger.error(
                "Exception in subprocess  (%s): %s", type(e).__name__, error_text
            )

            # Communicating exception via the queue to the main process
            parent_pipe.send(
                WorkerResponse(
                    request_type=request.request_type,
                    success=False,
                    error_msg=error_text,
                )
            )
            parent_pipe.close()
            logger.exception("Subprocess terminated due to exception")

    def _send(self, request_type: RequestType, payload: dict[str, Any]) -> None:
        try:
            if self._parent_end is None:
                raise AssertionError("Parent end of pipe should be initialized")
            self._parent_end.send(
                WorkerRequest(
                    request_type=request_type,
                    payload=payload,
                )
            )
        except OSError as e:
            error_msg = "Child process terminated unexpectedly"
            logger.exception(
                "Communication failed during %s request", request_type.value
            )
            raise RuntimeError(error_msg) from e

    def _recv(self) -> dict[str, Any] | None:
        try:
            if self._parent_end is None:
                raise AssertionError("Parent end of pipe should be initialized")
            response = self._parent_end.recv()
            if response.success is False:
                error_msg = (
                    f"Unexpected response from worker process: {response.error_msg}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            return response.payload
        except (EOFError, BrokenPipeError, ConnectionResetError) as e:
            error_msg = f"Child process terminated unexpectedly: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def write(
        self,
        state_dict: STATE_DICT | Future[STATE_DICT],
        path: str,
        **kwargs: Any,
    ) -> Future[None] | None:
        logger.debug("Waiting for subprocess initialization to complete")

        # wait until the process is started
        self.process_creation_future.result()

        return self._executor.submit(
            self._write,
            state_dict,
            path,
            **kwargs,
        )

    def _write(
        self,
        state_dict: STATE_DICT | Future[STATE_DICT],
        path: str,
        **kwargs: Any,
    ) -> None:
        logger.debug("Starting checkpoint write to %s", path)

        # wait for staging state_dict to be available
        if isinstance(state_dict, Future):
            logger.debug("Waiting for state_dict Future to resolve")
            sd = state_dict.result()
        else:
            sd = state_dict

        # Log state_dict info only if debug logging is enabled (performance-conscious)
        if logger.isEnabledFor(logging.DEBUG):
            if hasattr(sd, "keys"):
                logger.debug("State_dict contains %d keys", len(sd.keys()))

        self._send(
            request_type=RequestType.WRITE_CHECKPOINT,
            payload={
                "state_dict": sd,
                "path": path,
                "kwargs": kwargs,
            },
        )

        logger.debug("Waiting for write completion response")
        # wait for response
        self._recv()
        logger.debug("Checkpoint write to %s completed successfully", path)

    def close(self) -> None:
        logger.debug(
            "Closing CheckpointProcess for rank %d", self._rank_info.global_rank
        )
        self._executor.shutdown(wait=True, cancel_futures=True)

        if self.process and self.process.processes[0].is_alive():
            subprocess_pid = self.process.processes[0].pid
            # send graceful termination to sub process
            try:
                # pyrefly: ignore [missing-attribute]
                self._parent_end.send(
                    WorkerRequest(
                        request_type=RequestType.TERMINATE_PROCESS,
                        payload={},
                    )
                )
            except BrokenPipeError:
                logger.warning(
                    "BrokenPipeError when sending termination request - subprocess (PID: %d) may have already terminated",
                    subprocess_pid,
                )
                # subprocess terminated unexpectedly and below code will raise a
                # ProcessExitedException.

            logger.debug(
                "Waiting for subprocess to terminate gracefully (timeout: %ds)",
                self._config.subprocess_shutdown_timeout_secs,
            )

            try:
                if not self.process.join(
                    timeout=self._config.subprocess_shutdown_timeout_secs
                ):
                    # graceful shutdown failed, kill the process.
                    logger.warning(
                        "Subprocess (PID: %d) did not terminate gracefully within %ds, killing it",
                        subprocess_pid,
                        self._config.subprocess_shutdown_timeout_secs,
                    )
                    self.process.processes[0].kill()
                    logger.info("Subprocess killed forcefully")
            except ProcessExitedException:
                logger.exception("ProcessExitedException during subprocess termination")
                raise

        logger.debug("CheckpointProcess closed successfully")
<|endoftext|>"""
Checkpoint writer functionality for machine learning models.

This module provides classes for writing checkpoints to storage, including
determining checkpoint layout, configuring the writer, and defining hooks
for custom actions during the checkpoint writing process.
"""

import abc
import logging
import os
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .barriers import Barrier
from .types import RankInfo, STATE_DICT


logger = logging.getLogger(__name__)


class WriterHook(abc.ABC):
    """
    Abstract base class for checkpoint commit hooks.

    A commit hook provides callbacks that are executed before and after a checkpoint
    is committed to storage. This allows for custom actions to be performed at specific
    points in the checkpoint writing process, such as metadata updates, cleanup operations,
    or notifications.
    """

    @abc.abstractmethod
    def pre_commit(self, path: str, **kwargs: dict[str, Any]) -> None:
        """
        Performs actions before committing the checkpoint.
        """

    @abc.abstractmethod
    def post_commit(self, path: str, **kwargs: dict[str, Any]) -> None:
        """
        Performs actions after committing the checkpoint.
        """


@dataclass
class CheckpointWriterConfig:
    """
    Configuration options for the CheckpointWriter.

    Attributes:
        write_barrier_timeout_secs: Maximum time in seconds to wait for all ranks
            to reach the checkpoint barrier before timing out. Default is 600 seconds.
    """

    write_barrier_timeout_secs: int = 600


class CheckpointWriter:
    """
    Handles writing state dictionaries to storage.

    This class is responsible for writing model state dictionaries to storage according
    to the specified checkpoint layout. It supports synchronization barriers to ensure
    all ranks in a distributed setting complete their checkpoint operations.
    """

    def __init__(
        self,
        config: CheckpointWriterConfig,
        rank_info: RankInfo,
        barrier: Barrier | None = None,
        commit_hook: WriterHook | None = None,
    ):
        """
        Initialize a CheckpointWriter.

        Args:
            config: Configuration options for the checkpoint writer.
            rank_info: Information about the current rank in a distributed setting.
            barrier: Optional synchronization barrier for distributed checkpointing.
                    Note: The barrier should be initialized with the appropriate barrier_prefix
                    and timeout_secs parameters.
            commit_hook: Optional hook for custom actions before and after checkpoint commits.
        """

        self._config = config
        self._rank_info = rank_info
        self._commit_hook = commit_hook
        self._barrier = barrier

    def write(
        self,
        path: str,
        state_dict: STATE_DICT,
        **kwargs: dict[str, Any],
    ) -> Future[None] | None:
        """
        Writes the state_dict to storage.

        Args:
            path (str): The path to write the checkpoint to.
            state_dict (STATE_DICT): The state_dict to write.
            **kwargs: Additional keyword arguments passed to hooks.

        Returns:
            Optional[Future[None]]: A future for tracking the write operation, if applicable.
        """
        logger.debug(
            "Writing checkpoint to %s for rank %s",
            path,
            self._rank_info.global_rank,
        )
        dir_path = Path(path)
        full_path = dir_path / f"checkpoint_{self._rank_info.global_rank}.pt"
        os.makedirs(
            os.path.dirname(full_path),
            exist_ok=True,
        )
        torch.save(state_dict, full_path)
        logger.debug("Successfully saved checkpoint file to %s", full_path)

        # Execute pre-commit hook if available
        commit_hook = self._commit_hook
        if commit_hook is not None:
            logger.debug("Executing pre-commit hook for %s", path)
            commit_hook.pre_commit(path, **kwargs)

        # Wait for all ranks to finish writing if barrier is available
        barrier = self._barrier
        if barrier is not None:
            logger.info(
                "Waiting for all ranks at barrier with timeout %ss",
                self._config.write_barrier_timeout_secs,
            )
            barrier.execute_barrier()
            logger.info("All ranks passed barrier")
        else:
            logger.info("No barrier configured, skipping synchronization")

        # Execute commit hook if available
        if commit_hook is not None:
            logger.debug("Executing commit hook for %s", path)
            commit_hook.post_commit(path, **kwargs)

        logger.info(
            "Successfully wrote checkpoint to %s for rank %s",
            path,
            self._rank_info.global_rank,
        )
        return None

    def close(self) -> None:
        """
        Close the writer and release any resources.

        This is a no-op for the base CheckpointWriter but may be overridden
        by subclasses that need to perform cleanup.
        """
        logger.debug("Closing checkpoint writer")
<|endoftext|>"""
Experimental staging module for PyTorch Distributed Checkpointing.

This module provides advanced staging capabilities for checkpoints including:
- Asynchronous staging using ThreadPoolExecutor
- Pinned memory allocation for faster CPU-GPU transfers
- Shared memory support for multi-process scenarios
- Non-blocking CUDA operations with stream synchronization
- Caching of frequently used storages for efficient memory management
- Automatic resource cleanup and memory management

Classes:
    CheckpointStager: Abstract base class defining the staging interface
    StagingOptions: Configuration dataclass for staging behavior
    DefaultStager: Default implementation with comprehensive staging features
"""

import abc
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, TypeVar

import torch
from torch.distributed.checkpoint._state_dict_stager import StateDictStager

from .types import STATE_DICT


T = TypeVar("T")


class CheckpointStager(abc.ABC):
    """
    Abstract base class for checkpoint staging implementations.

    CheckpointStager defines the interface that all staging implementations
    must follow. Staging is the process of offloading state dictionaries
    for async checkpointing.
    """

    @abc.abstractmethod
    def stage(
        self,
        state_dict: STATE_DICT,
        **kwargs: Any,
    ) -> STATE_DICT | Future[STATE_DICT]:
        """
        Stage a state dictionary for checkpointing.

        Args:
            state_dict: The state dictionary to stage
            **kwargs: Additional staging parameters

        Returns:
            Either a staged state dictionary (synchronous) or a Future
            that will resolve to the staged state dictionary (asynchronous)
        """

    @abc.abstractmethod
    def close(self) -> None:
        """
        Clean up all resources used by the stager.
        """


@dataclass
class CheckpointStagerConfig:
    """
    Configuration options for checkpoint staging behavior.

    Attributes:
        use_pinned_memory (bool): Enable pinned memory allocation for faster
            CPU-GPU transfers. Requires CUDA to be available. Default: True
        use_shared_memory (bool): Enable shared memory for multi-process
            scenarios. Useful when multiple processes need access to the
            same staged data. Default: True
        use_async_staging (bool): Enable asynchronous staging using a
            background thread pool. Allows overlapping computation with
            staging operations. Requires CUDA. Default: True
        use_non_blocking_copy (bool): Use non-blocking device memory
            copies with stream synchronization. Improves performance by
            allowing CPU work to continue during GPU transfers. Default: True

    Note:
        CUDA-dependent features will raise exception if CUDA is not available.
    """

    use_pinned_memory: bool = True
    use_shared_memory: bool = True
    use_async_staging: bool = True
    use_non_blocking_copy: bool = True


class DefaultStager(CheckpointStager):
    """
    DefaultStager provides a full-featured staging implementation that combines
    multiple optimization techniques for efficient checkpoint preparation.

    The staging process works as follows:
    1. State dictionary is submitted for staging (sync or async)
    2. Tensors are copied from GPU to optimized CPU storage
    3. CUDA operations are synchronized if non-blocking copies are used
    4. Staged state dictionary is returned or made available via Future

    NOTE: state_dict should be deep-copyable object as staging will create a
    copy of it.

    Usage Patterns:
        # Synchronous staging
        stager = DefaultStager(CheckpointStagerConfig(use_async_staging=False))
        staged_dict = stager.stage(state_dict)
        stager.close()

        # Asynchronous staging
        stager = DefaultStager(CheckpointStagerConfig(use_async_staging=True))
        future = stager.stage(state_dict)
        # ... do other work ...
        staged_dict = future.result()
        stager.close()

        # Context manager pattern (recommended)
        with DefaultStager(config) as stager:
            result = stager.stage(state_dict)
            # Automatic cleanup on exit

    Performance Considerations:
        - Async staging provides best performance when model computation
          can overlap with staging operations
        - Pinned memory improves CPU-GPU transfer speeds but uses more memory
        - Shared memory allows efficient IPC to checkpoint process
        - Non-blocking copies reduce GPU idle time during memory transfers

    Thread Safety:
        DefaultStager is not thread-safe. Each thread should use its own
        instance, or external synchronization should be provided.
    """

    def __init__(
        self,
        config: CheckpointStagerConfig = CheckpointStagerConfig(),
    ):
        self._config = config
        self._state_dict_stager = StateDictStager(
            pin_memory=config.use_pinned_memory, share_memory=config.use_shared_memory
        )
        self._staging_executor = None
        self._staging_stream = None

        if self._config.use_async_staging:
            self._staging_executor = ThreadPoolExecutor(max_workers=1)
            if torch.accelerator.is_available():
                # Note: stream needs to be initialized on the main thread after default cuda
                # stream is setup/used to avoid the risk of accidentally reusing the main
                # compute stream or in other cases kernels actually launching from the
                # main thread.
                self._staging_stream = torch.Stream()

        if self._config.use_non_blocking_copy:
            if not torch.accelerator.is_available():
                raise AssertionError(
                    "Non-blocking copy requires that the current accelerator is available."
                )

    def stage(
        self,
        state_dict: STATE_DICT,
        **kwargs: Any,
    ) -> STATE_DICT | Future[STATE_DICT]:
        if self._config.use_async_staging:
            if self._staging_executor is None:
                raise AssertionError(
                    "Staging executor should be initialized for async staging"
                )
            return self._staging_executor.submit(
                self._stage,
                state_dict,
                **kwargs,
            )
        else:
            return self._stage(state_dict, **kwargs)

    def _stage(self, state_dict: STATE_DICT, **kwargs: Any) -> STATE_DICT:
        state_dict = self._state_dict_stager.stage(
            state_dict, non_blocking=self._config.use_non_blocking_copy, **kwargs
        )

        if self._config.use_non_blocking_copy:
            if not (self._staging_stream or not self._config.use_async_staging):
                raise AssertionError(
                    "Non-blocking copy in a background thread for async staging needs staging_stream to be initialized."
                )

            # waits for the enqueued copy operations to finish.
            self._staging_stream.synchronize() if self._staging_stream else torch.accelerator.synchronize()

        return state_dict

    def close(self) -> None:
        """
        Clean up all resources used by the DefaultStager. Shuts down the ThreadPoolExecutor
        used for async staging operations and cleans up the underlying StateDictStager's
        cached storages. Should be called when the stager is no longer needed to prevent
        resource leaks, especially in long-running applications. After calling close(),
        the stager should not be used for further staging operations.

        state_dict should be deep-copyable object.

        Example:
            stager = DefaultStager(CheckpointStagerConfig(use_async_staging=True))
            # ... do staging operations ...
            stager.close()  # Clean up all resources
        """
        if self._staging_executor:
            self._staging_executor.shutdown(wait=True)

        self._state_dict_stager.close()
<|endoftext|>"""
Type definitions for distributed training and checkpointing.

This module provides type definitions and classes for managing rank information
in distributed training environments, which is essential for proper checkpoint
saving and loading.
"""

from dataclasses import dataclass
from typing import Any, TypeAlias


# Type alias for state dictionaries used in checkpointing
STATE_DICT: TypeAlias = dict[str, Any]


@dataclass
class RankInfo:
    """
    Information about the current rank in a distributed training environment.

    Attributes:
        global_rank: The global rank ID of the current process.
        global_world_size: The total number of processes in the distributed environment.
    """

    global_rank: int
    global_world_size: int
<|endoftext|>"""
Utility functions for the experimental checkpoint module.

This module contains helper functions and utilities used across the experimental
checkpoint functionality.
"""

from concurrent.futures import Future
from typing import Any


def wrap_future(original_result: Any) -> Future[None]:
    """
    Wraps a result (Future or not) to return a Future with None result.

    If the input is a Future, returns a new Future that completes with None when
    the original Future completes successfully, or propagates any exception.
    If the input is not a Future, returns a completed Future with None result.

    Args:
        original_result: The result to wrap (Future or any other value).

    Returns:
        A Future that completes with None on success or propagates exceptions.
    """
    masked_future: Future[None] = Future()

    if isinstance(original_result, Future):

        def on_complete(_: Future[Any]) -> None:
            try:
                original_result.result()
                masked_future.set_result(None)
            except Exception as e:
                masked_future.set_exception(e)

        original_result.add_done_callback(on_complete)
    else:
        # Return a completed future with None result
        masked_future.set_result(None)

    return masked_future
<|endoftext|>"""
Configuration classes for checkpointer construction.

This module provides configuration dataclasses that consolidate all
configuration options needed to construct checkpointers.
"""

from dataclasses import dataclass, field

from .barriers import BarrierConfig
from .checkpoint_process import CheckpointProcessConfig
from .checkpoint_writer import CheckpointWriterConfig
from .staging import CheckpointStagerConfig


@dataclass
class CheckpointerConfig:
    """
    Configuration class for checkpointer construction.

    This class consolidates the core component configuration options needed to construct
    a checkpointer, providing a clean separation of concerns where each component
    manages its own configuration.

    Attributes:
        writer_config: Configuration options for the checkpoint writer component.
        barrier_config: Configuration for barrier construction and arguments.
        staging_config: Configuration options for the async staging component.
        process_config: Configuration options for the async checkpoint process component.

    """

    writer_config: CheckpointWriterConfig = field(
        default_factory=CheckpointWriterConfig
    )
    barrier_config: BarrierConfig = field(default_factory=BarrierConfig)

    # Below configs are used for async checkpointing
    staging_config: CheckpointStagerConfig = field(
        default_factory=CheckpointStagerConfig
    )
    process_config: CheckpointProcessConfig = field(
        default_factory=CheckpointProcessConfig
    )
<|endoftext|>"""
Checkpoint reader functionality for machine learning models.

This module provides classes for reading checkpoints from storage, including
determining checkpoint layout and configuring the reader.
"""

import logging
import os
from itertools import zip_longest
from pathlib import Path
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from .types import RankInfo, STATE_DICT


logger = logging.getLogger(__name__)


class CheckpointReader:
    """
    Handles reading state dictionaries from storage.

    This class is responsible for reading model state dictionaries from storage according
    to the specified checkpoint layout. It supports synchronization barriers to ensure
    all ranks in a distributed setting complete their checkpoint operations.
    """

    def __init__(
        self,
        rank_info: RankInfo,
    ):
        """
        Initialize a CheckpointReader.

        Args:
            rank_info: Information about the current rank in a distributed setting.
        """

        self._rank_info = rank_info

    def read(
        self,
        path: str,
        state_dict: STATE_DICT | None = None,
        *,
        map_location: Any = None,
        **kwargs: dict[str, Any],
    ) -> tuple[STATE_DICT, list[str]]:
        """
        Reads a state dictionary from storage.

        Args:
            path (str): The path from which to read the checkpoint.
            map_location (Any): Device mapping function or device name for relocating tensors.
            **kwargs: Additional keyword arguments passed to torch.load.

        Returns:
            STATE_DICT: The loaded state dictionary.
            list[str]: List of missing keys.
        """
        logger.debug(
            "Reading checkpoint from %s for rank %s",
            path,
            self._rank_info.global_rank,
        )

        dir_path = Path(path)
        file_path = dir_path / f"checkpoint_{self._rank_info.global_rank}.pt"

        # Check if the file exists
        if not os.path.exists(file_path):
            logger.error("Checkpoint file not found at %s", file_path)
            raise FileNotFoundError(f"Checkpoint file not found at {file_path}")

        if state_dict is None:
            result: tuple[STATE_DICT, list[str]] = (
                torch.load(file_path, map_location=map_location),
                [],
            )
        else:
            result = self._partial_read(
                file_path, state_dict, map_location=map_location, **kwargs
            )
        logger.debug("Successfully read checkpoint file from %s", file_path)
        return result

    def _partial_read(
        self,
        file_path: Path,
        state_dict: STATE_DICT,
        *,
        map_location: Any = None,
        **kwargs: dict[str, Any],
    ) -> tuple[STATE_DICT, list[str]]:
        """
        Reads only the keys present in state_dict from the checkpoint file.

        This method optimizes checkpoint loading by only loading the tensors that
        are actually needed, based on the keys present in the input state_dict.
        This can significantly reduce memory usage and loading time for large checkpoints
        when only a subset of the model needs to be loaded.

        Args:
            file_path (str): The path to the checkpoint file.
            state_dict (STATE_DICT): The state dictionary containing keys to load.
            map_location (Any): Device mapping function or device name for relocating tensors.
            **kwargs: Additional keyword arguments passed to torch.load.

        Returns:
            tuple[STATE_DICT, list[str]]: The updated state dictionary with loaded values and a list of missing keys.
        """

        with FakeTensorMode():
            metadata_dict = torch.load(file_path, map_location=map_location)

        missing_keys = []

        with open(file_path, "rb") as file:
            # Helper function to load tensor data from file
            def load_tensor(
                target: torch.Tensor | None, source: torch.Tensor, full_key: str
            ) -> torch.Tensor:
                if target is not None and (
                    target.size() != source.size() or target.dtype != source.dtype
                ):
                    raise RuntimeError(
                        f"Target tensor size={target.size()} dtype={target.dtype} does not match "
                        f"source tensor size={source.size()} dtype={source.dtype} for key {full_key}"
                    )

                tensor_offset = source.untyped_storage()._checkpoint_offset

                if tensor_offset is None:
                    raise AssertionError(
                        "checkpoint_offset for tensor in torch serialized file is not set. This could "
                        "happen if the checkpoint was saved with an older version of PyTorch. "
                        "Please make sure that the checkpoint was saved with Pytorch 2.7 or later."
                    )

                tensor_len = source.nelement() * source.element_size()
                file.seek(
                    tensor_offset + source.element_size() * int(source.storage_offset())
                )
                if target is None:
                    target = torch.empty(
                        source.size(), dtype=source.dtype, device=source.device
                    )

                buffer = file.read(tensor_len)
                cpu_tensor = torch.frombuffer(buffer, dtype=source.dtype)
                tensor = cpu_tensor.view(source.size())
                target.copy_(tensor)
                return target

            # Helper function to recursively process nested structures
            def process_value(
                target_value: Any, source_value: Any, key_path: str
            ) -> Any:
                source_type = type(source_value)
                if source_type is torch._subclasses.fake_tensor.FakeTensor:
                    source_type = torch.Tensor
                if target_value is not None and not isinstance(
                    target_value, source_type
                ):
                    raise RuntimeError(
                        f"Target value {key_path} is set to {type(target_value)}, but source value is {type(source_value)}"
                    )
                if isinstance(source_value, torch.Tensor):
                    return load_tensor(target_value, source_value, key_path)
                elif isinstance(source_value, dict):
                    if target_value is None:
                        # create a new map with all the keys present in source_value
                        target_value = dict.fromkeys(source_value.keys())

                    # pyrefly: ignore [missing-attribute]
                    for key in list(target_value.keys()):
                        current_path = f"{key_path}.{key}" if key_path else key
                        if key in source_value:
                            target_value[key] = process_value(
                                target_value[key], source_value[key], current_path
                            )
                        else:
                            missing_keys.append(current_path)

                    return target_value
                elif isinstance(source_value, list):
                    if target_value is None:
                        target_value = [None] * len(source_value)
                    result = []
                    for i, (target_item, source_item) in enumerate(
                        zip_longest(target_value, source_value, fillvalue=None)
                    ):
                        current_path = f"{key_path}[{i}]" if key_path else f"[{i}]"
                        result.append(
                            process_value(target_item, source_item, current_path)
                        )
                    return result
                else:
                    return source_value

            # Start recursive processing from the root of the state dictionary
            updated_state_dict = process_value(state_dict, metadata_dict, "")

        if missing_keys:
            if len(missing_keys) > 10:
                logger.warning(
                    "Missing %s keys from checkpoint: %s... (and %s more)",
                    len(missing_keys),
                    missing_keys[:10],
                    len(missing_keys) - 10,
                )
            else:
                logger.warning(
                    "Missing %s keys from checkpoint: %s",
                    len(missing_keys),
                    missing_keys,
                )

        return updated_state_dict, missing_keys
<|endoftext|>"""
Checkpoint functionality for machine learning models.

This module provides classes for saving and loading model checkpoints in a distributed
training environment. It includes functionality for coordinating checkpoint operations
across multiple processes and customizing the checkpoint process through hooks.

Key components:
- Checkpointer: Main class for orchestrating checkpoint operations (save, load)
- CheckpointWriter: Handles writing state dictionaries to storage
- CheckpointReader: Handles reading state dictionaries from storage read
- Barrier: Synchronization mechanism for distributed checkpointing
- RankInfo: Information about the current rank in a distributed environment
"""

from .barriers import (
    Barrier,
    BarrierConfig,
    create_barrier_from_config,
    TCPStoreBarrier,
)
from .builder import make_async_checkpointer, make_sync_checkpointer
from .checkpoint_reader import CheckpointReader
from .checkpoint_writer import CheckpointWriter, CheckpointWriterConfig, WriterHook
from .checkpointer import AsyncCheckpointer, Checkpointer, SyncCheckpointer
from .config import CheckpointerConfig
from .staging import CheckpointStager, CheckpointStagerConfig, DefaultStager
from .types import RankInfo, STATE_DICT
from .utils import wrap_future


__all__ = [
    "Barrier",
    "TCPStoreBarrier",
    "CheckpointReader",
    "CheckpointWriter",
    "CheckpointWriterConfig",
    "WriterHook",
    "Checkpointer",
    "SyncCheckpointer",
    "AsyncCheckpointer",
    "CheckpointerConfig",
    "BarrierConfig",
    "create_barrier_from_config",
    "CheckpointStager",
    "CheckpointStagerConfig",
    "DefaultStager",
    "RankInfo",
    "STATE_DICT",
    "wrap_future",
    "make_sync_checkpointer",
    "make_async_checkpointer",
]
<|endoftext|>"""
Barrier implementations for synchronizing distributed checkpoint operations.

This module provides abstract and concrete barrier implementations that ensure
all ranks in a distributed training environment complete their checkpoint operations
before proceeding, which is essential for data consistency.
"""

import abc
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Optional

import torch.distributed as dist
import torch.distributed.elastic.utils.store as store_util


logger = logging.getLogger()


# Registry of barrier types
BARRIER_REGISTRY: dict[str, type] = {}


def register_barrier(barrier_class: type) -> type:
    """Register a barrier class in the global registry."""
    if hasattr(barrier_class, "barrier_type"):
        BARRIER_REGISTRY[barrier_class.barrier_type] = barrier_class
    return barrier_class


@dataclass
class BarrierConfig:
    """
    Configuration for barrier construction.

    This class provides a flexible way to configure different barrier implementations
    with their specific constructor arguments. The barrier type will be looked up
    from a registry and instantiated with rank_info and barrier_args.

    Attributes:
        barrier_type: A string identifying the barrier type (e.g., "tcp_store").
                     If None, no barrier will be used.
        barrier_args: Dictionary of arguments to pass to the barrier constructor.
                     rank_info will be automatically injected as the first argument.

    Examples:
        # No barrier
        BarrierConfig()

        # TCPStore barrier
        BarrierConfig(
            barrier_type="tcp_store",
            barrier_args={
                'timeout_barrier_init_secs': 30,
                'barrier_prefix_list': ['checkpoint'],
                'use_checkpoint_barrier_tcpstore_libuv': False,
                'tcpstore_port': 12345,
                'master_address': 'localhost'
            }
        )
    """

    barrier_type: str | None = None
    barrier_args: dict[str, Any] = field(default_factory=dict)


def create_barrier_from_config(
    barrier_config: BarrierConfig,
) -> Optional["Barrier"]:
    """
    Create a barrier instance from BarrierConfig.

    Args:
        barrier_config: Configuration for barrier construction.

    Returns:
        Barrier instance or None if no barrier type is configured.

    Raises:
        ValueError: If the barrier_type is not found in the registry.
    """
    if barrier_config.barrier_type is None:
        return None

    if barrier_config.barrier_type not in BARRIER_REGISTRY:
        raise ValueError(
            f"Unknown barrier type: {barrier_config.barrier_type}. "
            f"Available types: {list(BARRIER_REGISTRY.keys())}"
        )

    barrier_class = BARRIER_REGISTRY[barrier_config.barrier_type]
    return barrier_class(**barrier_config.barrier_args)


class Barrier(abc.ABC):
    """
    Abstract base class for synchronization barriers.

    A barrier ensures that all ranks in a distributed environment reach a certain
    point in execution before any rank proceeds further, which is essential for
    coordinating operations like checkpointing across multiple processes.
    """

    @abc.abstractmethod
    def __init__(self, **kwargs: dict[str, Any]):
        """
        Initialize a barrier.

        Args:
            **kwargs: Keyword arguments for specific barrier implementations.
                     Common arguments may include rank information, barrier prefixes,
                     timeout settings, and other barrier-specific configuration.
        """
        # No implementation needed in the abstract base class

    @abc.abstractmethod
    def execute_barrier(self) -> None:
        """
        Execute a synchronization barrier.

        This method uses the barrier_prefix provided during initialization to
        coordinate synchronization across processes.
        """


@register_barrier
class DistBarrier(Barrier):
    """
    A barrier implementation using PyTorch's distributed barrier for synchronization.

    This barrier uses the built-in torch.distributed.barrier() function to coordinate
    synchronization across multiple processes. It's simpler than TCPStoreBarrier but
    requires an initialized process group.
    """

    barrier_type = "dist_barrier"

    def __init__(
        self,
    ) -> None:
        """
        Initialize a DistBarrier.

        This barrier requires an initialized PyTorch distributed process group.
        No additional arguments are needed as it uses the current process group.

        Raises:
            AssertionError: If the distributed process group is not initialized.
        """
        if not dist.is_initialized():
            raise AssertionError("DistBarrier requires an initialized process group.")

    def execute_barrier(self) -> None:
        """
        Execute a synchronization barrier using the prefix provided during initialization.
        """
        # Note: dist.barrier() doesn't support explicit timeouts
        # The timeout is handled by the underlying implementation
        dist.barrier()


@register_barrier
class TCPStoreBarrier(Barrier):
    """
    A barrier implementation using PyTorch's TCPStore for synchronization.

    This barrier uses a TCP-based distributed key-value store to coordinate
    synchronization across multiple processes. It uses a single TCP store
    for all barrier operations, with different prefixes to distinguish between
    different barrier types.
    """

    barrier_type = "tcp_store"

    def __init__(
        self,
        global_rank: int,
        global_world_size: int,
        barrier_prefix: str,
        timeout_barrier_init_secs: int,
        use_checkpoint_barrier_tcpstore_libuv: bool,
        tcpstore_port: int,
        master_address: str,
        timeout_secs: int,
    ):
        """
        Initialize a TCPStoreBarrier.

        Args:
            global_rank: The rank of the current process in the distributed environment.
            global_world_size: The total number of processes in the distributed environment.
            barrier_prefix: A string prefix to identify this specific barrier.
            timeout_barrier_init_secs: Timeout in seconds for initializing the TCPStore.
            use_checkpoint_barrier_tcpstore_libuv: Whether to use libuv for the TCPStore.
            tcpstore_port: Port number for the TCPStore.
            master_address: Address of the master node for the TCPStore.
            timeout_secs: Maximum time in seconds to wait for all ranks to reach the barrier.
        """
        logger.info(
            "Initializing TCPStore master_address=%s tcpstore_port=%s rank=%s "
            "world_size=%s barrier_prefix=%s timeout_barrier_init_secs=%s "
            "use_checkpoint_barrier_tcpstore_libuv=%s timeout_secs=%s",
            master_address,
            tcpstore_port,
            global_rank,
            global_world_size,
            barrier_prefix,
            timeout_barrier_init_secs,
            use_checkpoint_barrier_tcpstore_libuv,
            timeout_secs,
        )

        # Counter collection to track barrier seq on a per barrier prefix basis.
        self._tcp_store_barrier_seq: Counter = Counter()
        self._barrier_prefix = barrier_prefix

        # Store rank and world size for barrier operations
        self._global_rank = global_rank
        self._global_world_size = global_world_size
        self._timeout_secs = timeout_secs

        # Create a single TCP store for all barrier operations
        self._tcp_store = dist.TCPStore(
            master_address,
            int(tcpstore_port),
            world_size=self._global_world_size,
            timeout=timedelta(seconds=timeout_barrier_init_secs),
            is_master=(self._global_rank == 0),
        )

    def execute_barrier(self) -> None:
        """
        Execute a synchronization barrier using the prefix provided during initialization.

        The implementation uses a sequence number that is incremented every time
        a barrier is reached. The sequence number is per barrier prefix to allow
        different barriers to operate concurrently.
        """
        barrier_prefix = self._barrier_prefix

        logger.info(
            "Executing barrier barrier_prefix=%s timeout_secs=%s",
            barrier_prefix,
            self._timeout_secs,
        )

        def _rank_key(rank: int) -> str:
            return f"rank{rank}"

        # Track which barrier sequence this rank is joining.
        self._tcp_store.set(
            _rank_key(self._global_rank),
            str(self._tcp_store_barrier_seq[barrier_prefix]),
        )

        # Execute barrier for that sequence number (for the specific prefix).
        store_util.barrier(
            store=self._tcp_store,
            world_size=self._global_world_size,
            key_prefix=(
                barrier_prefix + str(self._tcp_store_barrier_seq[barrier_prefix])
            ),
        )
        self._tcp_store_barrier_seq[barrier_prefix] += 1
<|endoftext|>import functools
import math
import operator
from collections.abc import Callable, Sequence
from datetime import timedelta

import torch
from torch._C import ScriptObject
