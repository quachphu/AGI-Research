import torch
import torch.nn as nn
from torch.utils.checkpoint import (
    _checkpoint_without_reentrant_generator_impl,
    _DEFAULT_DETERMINISM_MODE,
)

from .contract import _State, contract


@contextmanager
def _no_hook(module: nn.Module, user_ctx: AbstractContextManager | None = None):
    r"""
    Disable hooks installed by checkpoint to avoid unintentional recursion
    during backward recomputation.
    """

    with user_ctx if user_ctx else nullcontext():
        orig_enable_hook = checkpoint.state(module).enable_hook
        checkpoint.state(module).enable_hook = False
        try:
            yield
        finally:
            checkpoint.state(module).enable_hook = orig_enable_hook


class _CheckpointState(_State):
    enable_hook: bool = False
    _ac_generator: Generator[None, None, None] | None


@contract(_CheckpointState)
def checkpoint(module: nn.Module, **kwargs) -> nn.Module:
    r"""
    This is a composable activation checkpointing API. Unlike functional
    activation checkpointing APIs, this one does not require changing model
    source code. Unlike ``nn.Module`` wrapper activation checkpointing APIs,
    this one does not modify model structure or fully-qualified names either.
    Under the hood, it registers activation checkpointing logic as pre- and
    post-forward hooks. Hence, this API can be easily applied to any model or
    sub-modules in the model.

    Args:
        module (nn.Module): the target model or sub-module to apply activation
            checkpointing.

    Example::
        >>> # xdoctest: +SKIP
        >>> import torch.nn as nn
        >>>
        >>> class MyModel(nn.Module):
        >>>     def __init__(self) -> None:
        >>>         super().__init__()
        >>>         self.l1 = nn.Linear(10, 10)
        >>>         self.l2 = nn.Linear(10, 10)
        >>>
        >>>     def forward(self, x):
        >>>         return self.l2(self.l1(x))
        >>>
        >>> model = MyModel()
        >>> checkpoint(model.l1)  # apply activation checkpointing only to l1
        >>> model(torch.zeros(2, 10)).sum().backward()

    """
    torch._C._log_api_usage_once("torch.distributed.checkpoint")

    use_reentrant = kwargs.pop("use_reentrant", False)
    if use_reentrant:
        raise NotImplementedError(
            "use_reentrant=True is not supported in composable checkpoint. "
            "Please use torch.utils.checkpoint.checkpoint instead."
        )
    preserve_rng_state = kwargs.pop("preserve_rng_state", True)
    user_context_fns = kwargs.pop("context_fn", None)
    determinism_check = kwargs.pop("determinism_check", _DEFAULT_DETERMINISM_MODE)
    debug = kwargs.pop("debug", False)
    early_stop = kwargs.pop("early_stop", True)

    if kwargs:
        raise ValueError(
            "Unexpected keyword arguments: " + ",".join(arg for arg in kwargs)
        )

    def forward_pre_hook(
        module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        if checkpoint.state(module).enable_hook:

            def context_fns():
                if user_context_fns is not None:
                    ctx1, ctx2 = user_context_fns()
                    return ctx1, _no_hook(module, ctx2)
                else:
                    return nullcontext(), _no_hook(module)

            gen = _checkpoint_without_reentrant_generator_impl(
                module,
                args,
                kwargs,
                preserve_rng_state=preserve_rng_state,
                context_fn=context_fns,
                determinism_check=determinism_check,
                debug=debug,
                early_stop=early_stop,
            )
            checkpoint.state(module)._ac_generator = gen
            next(gen)

    def forward_hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
        if checkpoint.state(module).enable_hook:
            try:
                gen = checkpoint.state(module)._ac_generator
                if gen is None:
                    raise AssertionError
                next(gen)
            except StopIteration:
                pass
            else:
                raise RuntimeError(
                    "Expected non-reentrant activation checkpoint generator to be exhausted, but it was not!"
                )

        #  Ensure that we no longer hold on to the generator. always_call=True helps ensure we
        # clear this even in the case of exception in fwd pass.
        checkpoint.state(module)._ac_generator = None

    checkpoint.state(module).enable_hook = True
    module.register_forward_pre_hook(forward_pre_hook, with_kwargs=True)
    module.register_forward_hook(forward_hook, prepend=True, always_call=True)
    return module
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import logging
from typing import overload, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.distributed._composable_state import _get_module_state
from torch.distributed.fsdp._fully_shard._fsdp_api import (
    MixedPrecisionPolicy,
    OffloadPolicy,
)
from torch.distributed.fsdp._fully_shard._fsdp_common import DDPMeshInfo
from torch.distributed.fsdp._fully_shard._fsdp_init import (
    _apply_to_module,
    _get_device_from_mesh,
    _get_mesh_info,
    _get_modules_and_states,
    _init_default_mesh,
    _init_param_group,
    _validate_mesh as _validate_mesh_common,
    _validate_module as _validate_module_common,
)
from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState, FSDPStateContext
from torch.distributed.fsdp._fully_shard._fully_shard import (
    _unimplemented_deepcopy,
    FSDPModule,
)

from .contract import _get_registry, contract


if TYPE_CHECKING:
    from torch.distributed.fsdp._fully_shard._fsdp_api import DataParallelMeshDims
    from torch.distributed.tensor import DeviceMesh


cls_to_replicate_cls: dict[type, type] = {}

logger = logging.getLogger("torch.distributed._composable.replicate_with_fsdp")


class _ReplicateStateContext(FSDPStateContext["_ReplicateState"]):
    """
    State shared across Replicate states.

    This is a typed subclass of FSDPStateContext parameterized with _ReplicateState,
    providing correct type annotations (e.g., all_states: list[_ReplicateState]).
    It also allows call sites to differentiate between Replicate and FSDP contexts
    via isinstance checks if needed.
    """


def _get_module_replicate_state(module: nn.Module) -> _ReplicateState | None:
    state = _get_module_state(module)
    if isinstance(state, _ReplicateState):
        return state
    return None


class _ReplicateState(FSDPState):
    _state_name: str = "Replicate"

    def __init__(self) -> None:
        super().__init__()
        self._state_ctx = _ReplicateStateContext()

    def _get_state_for_module(self, module: nn.Module) -> FSDPState | None:
        return _get_module_replicate_state(module)

    def init(
        self,
        modules: tuple[nn.Module, ...],
        device: torch.device,
        mp_policy: MixedPrecisionPolicy,
        auto_reshard_after_forward: bool = False,
    ) -> None:
        super().init(modules, device, mp_policy, auto_reshard_after_forward)


@overload
# pyrefly: ignore [inconsistent-overload]
def replicate(
    module: nn.Module,
    *,
    mesh: DeviceMesh | None = ...,
    mp_policy: MixedPrecisionPolicy = ...,
    offload_policy: OffloadPolicy = ...,
    ignored_params: set[nn.Parameter] | None = ...,
    dp_mesh_dims: DataParallelMeshDims | None = ...,
) -> ReplicateModule: ...


@overload
# pyrefly: ignore [inconsistent-overload]
def replicate(
    module: list[nn.Module],
    *,
    mesh: DeviceMesh | None = ...,
    mp_policy: MixedPrecisionPolicy = ...,
    offload_policy: OffloadPolicy = ...,
    ignored_params: set[nn.Parameter] | None = ...,
    dp_mesh_dims: DataParallelMeshDims | None = ...,
) -> list[ReplicateModule]: ...


@contract(state_cls=_ReplicateState)  # type: ignore[misc]
def replicate(
    module: nn.Module,
    *,
    mesh: DeviceMesh | None = None,
    mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
    offload_policy: OffloadPolicy = OffloadPolicy(),
    ignored_params: set[nn.Parameter] | None = None,
    dp_mesh_dims: DataParallelMeshDims | None = None,
):
    r"""Replicates a module

    Args:
        module (torch.nn.Module): module to replicate

    Example::
        >>> # xdoctest: +REQUIRES(module:torch._C._distributed_c10d)
        >>> module = nn.Linear(3, 3)
        >>> replicate(module)
    """
    torch._C._log_api_usage_once("torch.distributed._composable.replicate_with_fsdp")
    _validate_module(module)
    mesh = mesh or _init_default_mesh(mesh_dim_names=("replicate",))
    if dp_mesh_dims is not None:
        _validate_mesh_common(mesh, dp_mesh_dims)
        mesh_info = _get_mesh_info(mesh, dp_mesh_dims)
        if not isinstance(mesh_info, DDPMeshInfo):
            raise ValueError(
                "replicate() with dp_mesh_dims requires replicate-only "
                "dims (no shard dims). Use fully_shard() for sharding."
            )
    else:
        _validate_mesh(mesh)
        mesh_info = DDPMeshInfo(mesh, replicate_mesh_dim=0)
    device = _get_device_from_mesh(mesh)
    # managed_modules (3rd return) and buffers (5th return) are unused:
    # - managed_modules: FSDP uses this to set Dynamo-specific attributes
    #   (_is_fsdp_managed_module, _fsdp_use_orig_params), which replicate doesn't need
    # - buffers: already moved to device by _get_modules_and_states; replicate
    #   doesn't need to track them separately
    arg_module, modules, _, params, _ = _get_modules_and_states(
        module,
        device,
        ignored_params,
        is_composable_fn=is_composable_with_replicate,
        get_state_fn=_get_module_replicate_state,
    )
    state = replicate.state(modules[0])  # type: ignore[attr-defined]
    state.init(modules, device, mp_policy)

    _init_param_group(
        state,
        params,
        modules,
        mesh_info,
        None,  # post_forward_mesh_info
        device,
        None,  # shard_placement_fn
        mp_policy,
        offload_policy,
    )

    # Place Replicate leftmost for highest priority in the method resolution order
    _apply_to_module(
        modules,
        cls_to_replicate_cls,
        ReplicateModule,
        "Replicate",
        _unimplemented_deepcopy,
    )
    return arg_module


class ReplicateModule(FSDPModule):
    # Index in MRO where the original class is found.
    # For Replicate: [Replicate<Orig>, ReplicateModule, FSDPModule, Orig, ...] -> index 3
    _orig_cls_mro_index: int = 3


def is_composable_with_replicate(module: nn.Module) -> bool:
    registry = _get_registry(module)
    if registry is None:
        return True
    return "fully_shard" not in registry


def _validate_module(module: nn.Module) -> None:
    if not is_composable_with_replicate(module):
        raise RuntimeError(
            "Cannot apply `replicate()` on a Module already managed by `fully_shard`"
        )
    _validate_module_common(module, "replicate")


def _validate_mesh(mesh: DeviceMesh) -> None:
    if mesh.ndim != 1:
        raise ValueError(f"replicate expects a 1D DeviceMesh but got {mesh}")
<|endoftext|># mypy: allow-untyped-defs
import weakref
from collections.abc import Iterable
from typing import Any, NoReturn

import torch
import torch.nn as nn
from torch.distributed._composable_state import _State
from torch.nn.parallel import DistributedDataParallel

from .contract import _get_registry, contract


_ROOT_MODULE_PREFIX = ""


class _ReplicateState(_State):
    _ddp_weakref: weakref.ref

    def __init__(self) -> None:
        super().__init__()
        self.module: nn.Module = nn.ParameterList()
        self.has_initialized: bool = False
        self._param_list: nn.ParameterList = nn.ParameterList()
        # TODO(@fegin): this variable is originally created for testing, we
        # should remove this if possible.
        self._orig_module = self.module
        self._param_names: list[str] = []
        self._no_sync: bool = False
        self._init_args: tuple[Any, ...] | None = None
        self._init_kwargs: dict[str, Any] = {}
        self._comm_hook_args: list[Any] = []

    def _collect_params(
        self,
        module: nn.Module,
        ignored_modules: set[nn.Module],
        ignored_params: set[nn.Parameter],
        prefix: str = _ROOT_MODULE_PREFIX,
    ) -> None:
        # skip if managed by fully_sharded API
        if _is_fully_sharded(module):
            return

        # if a module is ignored, all descendants of the module are ignored.
        if module in ignored_modules:
            return

        recurse_prefix = (
            f"{prefix}." if prefix != _ROOT_MODULE_PREFIX else _ROOT_MODULE_PREFIX
        )

        for n, p in module.named_parameters(recurse=False):
            if p not in ignored_params:
                self._param_list.append(p)
                self._param_names.append(f"{recurse_prefix}{n}")

        for name, child_module in module.named_children():
            self._collect_params(
                child_module,
                ignored_modules,
                ignored_params,
                prefix=f"{recurse_prefix}{name}",
            )

    def lazy_init(self) -> None:
        @torch._disable_dynamo(recursive=True)
        def _lazy_init():
            if self._init_args is None:
                raise AssertionError
            self.init(*self._init_args, **self._init_kwargs)
            self.register_comm_hook()
            self._init_args = ()
            self._init_kwargs = {}

        _lazy_init()

    def init(
        self,
        module: nn.Module,
        ignored_modules: set[nn.Module],
        **kwargs,
    ) -> None:
        if self.has_initialized:
            return

        self.has_initialized = True
        self.module = module
        ignored_params = {p for m in ignored_modules for p in m.parameters()}
        for submodule in module.modules():
            if _is_fully_sharded(submodule):
                ignored_params.update(submodule.parameters())
        from torch.distributed.tensor.parallel.ddp import _localize_dtensor

        _localize_dtensor(module, ignored_params=ignored_params)
        self._collect_params(module, ignored_modules, ignored_params)

        if "device_id" in kwargs:
            # replicate() supports a small usability enhancement where
            # user can pass in device_id as a Union[int, torch.device] even for
            # CPU devices so users don't have to change code for CPU/GPU runs.
            # We derive the right device_ids to feed into DDP to support this.
            if kwargs["device_id"] is not None:
                device_id = kwargs["device_id"]
                # Convert to device_ids that DDP expects.
                if isinstance(device_id, torch.device) and device_id.type == "cpu":
                    # CPU modules receive device_ids None
                    kwargs["device_ids"] = None
                else:
                    # GPU modules expect device_ids=[cuda_device]
                    kwargs["device_ids"] = [device_id]
            else:
                kwargs["device_ids"] = None
            kwargs.pop("device_id")

        self._ddp = DistributedDataParallel(self._param_list, **kwargs)
        # Weakref to the DDP instance is currently only used for testing.
        replicate.state(self.module)._ddp_weakref = weakref.ref(self._ddp)

    def register_comm_hook(self) -> None:
        for comm_args, comm_kwargs in self._comm_hook_args:
            self._ddp.register_comm_hook(*comm_args, **comm_kwargs)
        self._comm_hook_args.clear()

    def record_init_args(self, *args, **kwargs) -> None:
        self._init_args = args
        self._init_kwargs = kwargs

    def forward_pre_hook(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        if self._init_args or self._init_kwargs:
            self.lazy_init()
        self._ddp.require_backward_grad_sync = not self._no_sync
        DistributedDataParallel._active_ddp_module = self._ddp
        return self._ddp._pre_forward(*args, **kwargs)

    def forward_post_hook(
        self,
        module: nn.Module,
        input: tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> torch.Tensor:
        DistributedDataParallel._active_ddp_module = None
        return self._ddp._post_forward(output)


def unimplemented_deepcopy(*args: Any, **kwargs: Any) -> NoReturn:
    raise AssertionError(
        "DDP does not support deepcopy. Please use state dict for serialization."
    )


# Follow the same pattern as FSDP/fully_shard
class DDP:
    def __new__(cls, *args, **kwargs):
        """
        Override ``__new__`` to remove the DDP class and directly construct
        the original class for cases like indexing into a container module.
        """
        # Use index 2 since 0 is the dynamically constructed `DDP<...>` class
        # and index 1 is the `DDP` class itself
        orig_cls = cls.__mro__[2]
        return orig_cls.__new__(orig_cls, *args, **kwargs)

    def set_requires_gradient_sync(self, requires_gradient_sync: bool) -> None:
        """
        Sets if the module should sync gradients. This can be used to implement
        gradient accumulation without communication.

        Args:
            requires_gradient_sync (bool): Whether to reduce gradients for the
                module's parameters.
        """
        replicate.state(self)._no_sync = not requires_gradient_sync  # type: ignore[arg-type]

    def register_comm_hook(self, *args, **kwargs) -> None:
        replicate.state(self)._comm_hook_args.append((args, kwargs))  # type: ignore[arg-type]


@contract(state_cls=_ReplicateState)
def replicate(
    module: nn.Module,
    ignored_modules: Iterable[torch.nn.Module] | None = None,
    **kwargs,
) -> nn.Module:
    r"""Replicates a module

    Args:
        module (torch.nn.Module): module to replicate

    Example::
        >>> # xdoctest: +REQUIRES(module:torch._C._distributed_c10d)
        >>> module = nn.Linear(3, 3)
        >>> replicate(module)
    """
    torch._C._log_api_usage_once("torch.distributed.replicate")

    # TODO(fegin): using kwargs is not a good idea if we would like to make
    # replicate a formal API to replace DDP.
    if "device_id" in kwargs:
        if not isinstance(kwargs["device_id"], (int, torch.device)):
            raise RuntimeError(
                "Expected device_id to be int or torch.device, "
                f"but got {type(kwargs['device_id'])}"
            )

    if _is_fully_sharded(module):
        raise RuntimeError(
            "Cannot apply `replicate()` on a Module already managed by `fully_shard`"
        )

    if ignored_modules is None:
        ignored_modules = {}
    else:
        ignored_modules = set(ignored_modules)

    state = replicate.state(module)
    module.register_forward_pre_hook(state.forward_pre_hook, with_kwargs=True)
    device_mesh = kwargs.get("device_mesh")
    if device_mesh is not None:
        root_mesh = device_mesh._get_root_mesh()
        # if a root mesh is not the same as device_mesh,
        # meaning the device_mesh is sliced out from the root mesh.
        if root_mesh != device_mesh:
            # TODO: This is a temporary work around to enable DDP + TP.
            # We should do the logic in DDP so that the 2D implementation is
            # sound and the state_dict works out of the box.
            #
            # This won't conflict with what is done in DDP class as the module
            # replicate is going to pass is NOT the original module.
            from torch.distributed.tensor.parallel.ddp import (
                _localize_dtensor,
                _reconstruct_dtensor,
            )

            module.register_forward_pre_hook(_reconstruct_dtensor)
            module.register_forward_hook(_localize_dtensor)

    module.register_forward_hook(state.forward_post_hook)  # type: ignore[arg-type]

    state.record_init_args(module, ignored_modules, **kwargs)

    # Place DDP leftmost for highest priority in the method resolution order
    cls = module.__class__
    dct = {"__deepcopy__": unimplemented_deepcopy}
    new_cls = type(f"DDP{cls.__name__}", (DDP, cls), dct)
    module.__class__ = new_cls
    return module


def _is_fully_sharded(module: nn.Module) -> bool:
    r"""Check if module is marked with fully_shard."""
    registry = _get_registry(module)
    if registry is None:
        return False
    return "fully_shard" in registry
<|endoftext|>"""
NOTICE: DTensor has moved to torch.distributed.tensor

This file is a shim to redirect to the new location, and
we keep the old import path starts with `_tensor` for
backward compatibility. We will remove this folder once
we resolve all the BC issues.
"""

import sys
from importlib import import_module


submodules = [
    # TODO: _shards_wrapper/_utils here mainly for checkpoint BC, remove them
    "_shards_wrapper",
    "_utils",
    "experimental",
    "device_mesh",
]

# Redirect imports
for submodule in submodules:
    full_module_name = f"torch.distributed.tensor.{submodule}"
    sys.modules[f"torch.distributed._tensor.{submodule}"] = import_module(
        full_module_name
    )

from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    empty,
    full,
    init_device_mesh,
    ones,
    Partial,
    Placement,
    rand,
    randn,
    Replicate,
    Shard,
    zeros,
)
<|endoftext|>import os
from collections.abc import Generator
from contextlib import contextmanager, ExitStack

from torch.distributed.elastic.multiprocessing.errors import record


__all__ = [
    "worker_main",
]

TORCH_WORKER_SERVER_SOCKET = "TORCH_WORKER_SERVER_SOCKET"


@contextmanager
def _worker_server(socket_path: str) -> Generator[None, None, None]:
    from torch._C._distributed_c10d import _WorkerServer

    server = _WorkerServer(socket_path)
    try:
        yield
    finally:
        server.shutdown()


@record
@contextmanager
def worker_main() -> Generator[None, None, None]:
    """
    This is a context manager that wraps your main entry function. This combines
    the existing ``errors.record`` logic as well as a new ``_WorkerServer`` that
    exposes handlers via a unix socket specified by
    ``TORCH_WORKER_SERVER_SOCKET``.

    Example

    ::

     @worker_main()
     def main():
         pass


     if __name__ == "__main__":
         main()

    """
    with ExitStack() as stack:
        socket_path = os.environ.get(TORCH_WORKER_SERVER_SOCKET)
        if socket_path is not None:
            stack.enter_context(_worker_server(socket_path))

        yield
<|endoftext|>#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import logging
from typing import cast

from torch.distributed import PrefixStore, Store, TCPStore
from torch.distributed.elastic.rendezvous import (
    RendezvousHandler,
    RendezvousInfo,
    RendezvousParameters,
    RendezvousStoreInfo,
)
from torch.distributed.elastic.rendezvous.utils import parse_rendezvous_endpoint


__all__ = ["StaticTCPRendezvous", "create_rdzv_handler"]

logger = logging.getLogger(__name__)

_default_timeout_seconds = 600


class StaticTCPRendezvous(RendezvousHandler):
    """
    Static rendezvous that is a wrapper around the TCPStore.

    Creates TCPStore based on the input parameters with the
    listener on the agent with group_rank=0
    """

    def __init__(
        self,
        master_addr: str,
        master_port: int,
        rank: int,
        world_size: int,
        run_id: str,
        timeout: int,
    ):
        self.master_addr = master_addr
        self.master_port = master_port
        self.rank = rank
        self.world_size = world_size
        self.run_id = run_id
        self.timeout = datetime.timedelta(seconds=timeout)
        self._store: Store | None = None

    def get_backend(self) -> str:
        return "static"

    @property
    def use_agent_store(self) -> bool:
        return True

    def next_rendezvous(self) -> RendezvousInfo:
        logger.info("Creating TCPStore as the c10d::Store implementation")
        is_master = self.rank == 0
        if not self._store:
            self._store = TCPStore(  # type: ignore[call-arg]
                self.master_addr,
                self.master_port,
                self.world_size,
                is_master,
                self.timeout,
                multi_tenant=True,
            )
        store = PrefixStore(self.run_id, self._store)
        # TCPStore server instance is used by trainer code
        bootstrap_store_info = RendezvousStoreInfo(self.master_addr, self.master_port)
        return RendezvousInfo(
            store,
            self.rank,
            self.world_size,
            bootstrap_store_info,
        )

    def is_closed(self):
        return False

    def set_closed(self):
        pass

    def num_nodes_waiting(self):
        return 0

    def get_run_id(self) -> str:
        return self.run_id

    def shutdown(self) -> bool:
        return True


def create_rdzv_handler(params: RendezvousParameters) -> RendezvousHandler:
    if "rank" not in params.config:
        raise ValueError(
            "rank is absent in RendezvousParameters."
            "Try add --node-rank to the cmd request"
        )
    endpoint = params.endpoint.strip()
    if not endpoint:
        raise ValueError(
            "endpoint is absent in RendezvousParameters"
            "Try add --master-port and --master-addr to the cmd request"
        )
    master_addr, master_port = parse_rendezvous_endpoint(endpoint, -1)
    if master_port == -1:
        raise ValueError(
            f"Port is absent in endpoint: {endpoint}. Try launching with --master-port"
        )
    world_size = params.max_nodes
    rank = cast(int, params.config.get("rank"))
    run_id = params.run_id
    if "timeout" in params.config:
        timeout = int(params.config["timeout"])
    else:
        timeout = _default_timeout_seconds

    return StaticTCPRendezvous(
        master_addr, master_port, rank, world_size, run_id, timeout
    )
<|endoftext|># Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from importlib.metadata import entry_points

from .api import (
    rendezvous_handler_registry as handler_registry,
    RendezvousHandler,
    RendezvousParameters,
)
from .dynamic_rendezvous import create_handler


log = logging.getLogger(__name__)

__all__ = ["get_rendezvous_handler"]


def _create_static_handler(params: RendezvousParameters) -> RendezvousHandler:
    from . import static_tcp_rendezvous

    return static_tcp_rendezvous.create_rdzv_handler(params)


def _create_etcd_handler(params: RendezvousParameters) -> RendezvousHandler:
    from . import etcd_rendezvous

    return etcd_rendezvous.create_rdzv_handler(params)


def _create_etcd_v2_handler(params: RendezvousParameters) -> RendezvousHandler:
    from .etcd_rendezvous_backend import create_backend

    backend, store = create_backend(params)

    return create_handler(store, backend, params)


def _create_c10d_handler(params: RendezvousParameters) -> RendezvousHandler:
    from .c10d_rendezvous_backend import create_backend

    backend, store = create_backend(params)

    return create_handler(store, backend, params)


def _register_default_handlers() -> None:
    handler_registry.register("etcd", _create_etcd_handler)
    handler_registry.register("etcd-v2", _create_etcd_v2_handler)
    handler_registry.register("c10d", _create_c10d_handler)
    handler_registry.register("static", _create_static_handler)


def _register_out_of_tree_handlers() -> None:
    discovered_handler_generators = entry_points(group="torchrun.handlers")

    for handler_generator in discovered_handler_generators:
        try:
            # pyrefly: ignore [bad-index]
            get_handler = discovered_handler_generators[handler_generator.name].load()
            handler_registry.register(handler_generator.name, get_handler())
        except Exception:
            log.warning(
                "Exception while registering out of tree plugin %s: ",
                handler_generator.name,
                exc_info=True,
            )


def get_rendezvous_handler(params: RendezvousParameters) -> RendezvousHandler:
    """
    Obtain a reference to a :py:class`RendezvousHandler`.

    Custom rendezvous handlers can be registered by

    ::

      from torch.distributed.elastic.rendezvous import rendezvous_handler_registry
      from torch.distributed.elastic.rendezvous.registry import get_rendezvous_handler


      def create_my_rdzv(params: RendezvousParameters):
          return MyCustomRdzv(params)


      rendezvous_handler_registry.register("my_rdzv_backend_name", create_my_rdzv)

      my_rdzv_handler = get_rendezvous_handler(
          "my_rdzv_backend_name", RendezvousParameters
      )
    """
    return handler_registry.create_handler(params)
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import logging
import os
import pickle
import socket
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

import torch.distributed as dist
from torch.distributed import Store
from torch.distributed.elastic.events import construct_and_record_rdzv_event, NodeState

from .api import (
    RendezvousClosedError,
    RendezvousError,
    RendezvousGracefulExitError,
    RendezvousHandler,
    RendezvousInfo,
    RendezvousParameters,
    RendezvousStateError,
    RendezvousStoreInfo,
    RendezvousTimeoutError,
)
from .utils import _delay, _PeriodicTimer


__all__ = [
    "RendezvousBackend",
    "RendezvousTimeout",
    "RendezvousSettings",
    "DynamicRendezvousHandler",
    "create_handler",
]

logger = logging.getLogger(__name__)


def get_method_name(depth=2):
    if len(inspect.stack()) > depth:
        return inspect.stack()[depth].function
    return "no_method_name"


Token = Any
"""Represent an opaque fencing token used by the rendezvous backend."""


class RendezvousBackend(ABC):
    """Represent a backend that holds the rendezvous state."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the backend."""

    @abstractmethod
    def get_state(self) -> tuple[bytes, Token] | None:
        """Get the rendezvous state.

        Returns:
            A tuple of the encoded rendezvous state and its fencing token or
            ``None`` if no state is found in the backend.

        Raises:
            RendezvousConnectionError:
                The connection to the backend has failed.
            RendezvousStateError:
                The rendezvous state is corrupt.
        """

    @abstractmethod
    def set_state(
        self, state: bytes, token: Token | None = None
    ) -> tuple[bytes, Token, bool] | None:
        """Set the rendezvous state.

        The new rendezvous state is set conditionally:

          - If the specified ``token`` matches the fencing token stored in the
            backend, the state will be updated. The new state will be returned
            to the caller along with its fencing token.
          - If the specified ``token`` does not match the fencing token stored
            in the backend, the state won't be updated; instead the existing
            state along with its fencing token will be returned to the caller.
          - If the specified ``token`` is ``None``, the new state will be set
            only if there is no existing state in the backend. Either the new
            state or the existing state along with its fencing token will be
            returned to the caller.

        Args:
            state:
                The encoded rendezvous state.
            token:
                An optional fencing token that was retrieved by a previous call
                to :py:meth:`get_state` or ``set_state()``.

        Returns:
            A tuple of the serialized rendezvous state, its fencing token, and
            a boolean value indicating whether our set attempt succeeded.

        Raises:
            RendezvousConnectionError:
                The connection to the backend has failed.
            RendezvousStateError:
                The rendezvous state is corrupt.
        """


class RendezvousTimeout:
    """Hold the timeout configuration of a rendezvous.

    Args:
        join:
            The time within which the rendezvous is expected to complete.
        last_call:
            An additional wait amount before completing the rendezvous once the
            rendezvous has the minimum number of required participants.
        close:
            The time within which the rendezvous is expected to close after a
            call to :py:meth:`RendezvousHandler.set_closed` or
            :py:meth:`RendezvousHandler.shutdown`.
        heartbeat:
            The time within which a keep-alive heartbeat is expected to
            complete.
    """

    _ZERO = timedelta(0)

    _DEFAULT_TIMEOUTS = {
        "join": timedelta(seconds=600),
        "last_call": timedelta(seconds=30),
        "close": timedelta(seconds=30),
        "heartbeat": timedelta(seconds=5),
    }

    _join: timedelta
    _last_call: timedelta
    _close: timedelta
    _heartbeat: timedelta

    def __init__(
        self,
        join: timedelta | None = None,
        last_call: timedelta | None = None,
        close: timedelta | None = None,
        heartbeat: timedelta | None = None,
    ) -> None:
        self._set_timeouts(
            join=join, last_call=last_call, close=close, heartbeat=heartbeat
        )

    @property
    def join(self) -> timedelta:
        """Get the join timeout."""
        return self._join

    @property
    def last_call(self) -> timedelta:
        """Get the last call timeout."""
        return self._last_call

    @property
    def close(self) -> timedelta:
        """Get the close timeout."""
        return self._close

    @property
    def heartbeat(self) -> timedelta:
        """Get the keep-alive heartbeat timeout."""
        return self._heartbeat

    def _set_timeouts(self, **timeouts: timedelta | None):
        for name, timeout in timeouts.items():
            if timeout is None:
                timeout = self._DEFAULT_TIMEOUTS[name]
            if timeout <= self._ZERO:
                raise ValueError(f"The {name} timeout ({timeout}) must be positive.")
            setattr(self, "_" + name, timeout)


@dataclass(repr=False, eq=False, frozen=True)
class RendezvousSettings:
    """Hold the settings of the rendezvous.

    Attributes:
        run_id:
            The run id of the rendezvous.
        min_nodes:
            The minimum number of nodes to admit to the rendezvous.
        max_nodes:
            The maximum number of nodes to admit to the rendezvous.
        timeout:
            The timeout configuration of the rendezvous.
        keep_alive_interval:
            The amount of time a node waits before sending a heartbeat to keep
            it alive in the rendezvous.
        keep_alive_max_attempt:
            The maximum number of failed heartbeat attempts after which a node
            is considered dead.
    """

    run_id: str
    min_nodes: int
    max_nodes: int
    timeout: RendezvousTimeout
    keep_alive_interval: timedelta
    keep_alive_max_attempt: int


@dataclass(eq=True, order=True, frozen=True)
class _NodeDesc:
    """Describe a node in the rendezvous.

    Attributes:
        addr:
            The FQDN of the node or user specified local node address.
        pid:
            The id of the process in which the rendezvous handler runs.
        local_id:
            A process-wide unique id.
    """

    addr: str
    pid: int
    local_id: int

    def __repr__(self) -> str:
        return f"{self.addr}_{self.pid}_{self.local_id}"


class _NodeDescGenerator:
    """Generate node descriptors.

    A node descriptor is a combination of an FQDN, a process id, and an auto-
    incremented integer that uniquely identifies a node in the rendezvous.
    """

    _lock: threading.Lock
    _local_id: int

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # An integer that is incremented with each call to generate().
        self._local_id = 0

    def generate(self, local_addr: str | None = None) -> _NodeDesc:
        # This method can be called by multiple threads concurrently; therefore,
        # we must increment the integer atomically.
        with self._lock:
            local_id = self._local_id

            self._local_id += 1

        return _NodeDesc(local_addr or socket.getfqdn(), os.getpid(), local_id)


class _RendezvousState:
    """Hold the state of a rendezvous.

    Attributes:
        round:
            The current round of the rendezvous.
        complete:
            A boolean value indicating whether the current round of the
            rendezvous is complete.
        deadline:
            The time at which the current round of the rendezvous will be
            considered complete if it is still waiting for nodes to join.
        closed:
            A boolean value indicating whether the rendezvous is closed.
        participants:
            A dictionary of the participants and their corresponding ranks.
        wait_list:
            A set of nodes that are waiting to participate in the next round of
            the rendezvous.
        redundancy_list:
            A set of nodes that are redundant in the current round and can join
            the next rendezvous without triggering re-rendezvous.
        last_heartbeats:
            A dictionary containing each node's last heartbeat time.
    """

    round: int
    complete: bool
    deadline: datetime | None
    closed: bool
    participants: dict[_NodeDesc, int]
    wait_list: set[_NodeDesc]
    redundancy_list: set[_NodeDesc]
    last_heartbeats: dict[_NodeDesc, datetime]

    def __init__(self) -> None:
        self.round = 0
        self.complete = False
        self.deadline = None
        self.closed = False
        self.participants = {}
        self.wait_list = set()
        self.redundancy_list = set()
        self.last_heartbeats = {}


def _remove_participant_epilogue(
    state: _RendezvousState, settings: RendezvousSettings
) -> None:
    if state.complete:
        # If we do not have any participants left, move to the next round.
        if not state.participants:
            msg = "No participants left in the rendezvous, marking rendezvous as incomplete"
            logger.debug(msg)
            state.complete = False

            state.round += 1
    else:
        if len(state.participants) < settings.min_nodes:
            msg = (
                f"Number of participants {len(state.participants)} less than "
                f"min_nodes {settings.min_nodes}, clearing deadline in state"
            )
            logger.debug(msg)
            state.deadline = None


class _RendezvousStateHolder(ABC):
    """Hold the shared rendezvous state synced with other nodes."""

    @property
    @abstractmethod
    def state(self) -> _RendezvousState:
        """Get the local state."""

    @abstractmethod
    def sync(self) -> bool | None:
        """Read or writes the latest state.

        Returns:
            A boolean value indicating whether the local state, in case marked
            as dirty, was successfully synced with other nodes.
        """

    @abstractmethod
    def mark_dirty(self) -> None:
        """Mark the local state as dirty."""


class _BackendRendezvousStateHolder(_RendezvousStateHolder):
    """Hold the rendezvous state synced with other nodes via a backend.

    Args:
        backend:
            The rendezvous backend to use.
        settings:
            The rendezvous settings.
        cache_duration:
            The amount of time, in seconds, to cache the last rendezvous state
            before requesting it from the backend again.
    """

    _backend: RendezvousBackend
    _state: _RendezvousState
    _settings: RendezvousSettings
    _cache_duration: int
    _token: Token
    _dirty: bool
    _last_sync_time: float
    _dead_nodes: list[_NodeDesc]

    def __init__(
        self,
        backend: RendezvousBackend,
        settings: RendezvousSettings,
        cache_duration: int = 1,
    ) -> None:
        self._backend = backend
        self._state = _RendezvousState()
        self._settings = settings
        self._cache_duration = cache_duration
        self._token = None
        self._dirty = False
        self._last_sync_time = -1
        self._dead_nodes = []

    def _record(self, message: str, node_state: NodeState = NodeState.RUNNING):
        construct_and_record_rdzv_event(
            name=f"{self.__class__.__name__}.{get_method_name()}",
            run_id=self._settings.run_id,
            message=message,
            node_state=node_state,
        )

    @property
    def state(self) -> _RendezvousState:
        """See base class."""
        return self._state

    def sync(self) -> bool | None:
        """See base class."""
        state_bits: bytes | None = None

        token = None

        has_set: bool | None

        if self._dirty:
            has_set = False

            state_bits = pickle.dumps(self._state)

            set_response = self._backend.set_state(state_bits, self._token)
            if set_response is not None:
                state_bits, token, has_set = set_response
        else:
            has_set = None

            if self._cache_duration > 0:
                # Avoid overloading the backend if we are asked to retrieve the
                # state repeatedly. Try to serve the cached state.
                if self._last_sync_time >= max(
                    time.monotonic() - self._cache_duration, 0
                ):
                    return None

            get_response = self._backend.get_state()
            if get_response is not None:
                state_bits, token = get_response

        if state_bits is not None:
            try:
                self._state = pickle.loads(state_bits)
            except pickle.PickleError as exc:
                raise RendezvousStateError(
                    "The rendezvous state is corrupt. See inner exception for details."
                ) from exc
        else:
            self._state = _RendezvousState()

        if has_set and self._dead_nodes and logger.isEnabledFor(logging.DEBUG):
            node_list = ", ".join(f"'{dead_node}'" for dead_node in self._dead_nodes)

            msg = (
                f"As part of the sync operation the node(s) {node_list} have been removed from the "
                f"rendezvous '{self._settings.run_id}' since they had no heartbeat."
            )
            self._record(message=msg)
            logger.debug(msg)

        self._token = token

        self._dirty = False

        self._last_sync_time = time.monotonic()

        self._sanitize()

        return has_set

    def _sanitize(self) -> None:
        state = self._state

        expire_time = datetime.now(timezone.utc) - (
            self._settings.keep_alive_interval * self._settings.keep_alive_max_attempt
        )

        # Filter out the dead nodes.
        self._dead_nodes = [
            node
            for node, last_heartbeat in state.last_heartbeats.items()
            if last_heartbeat < expire_time
        ]

        participant_removed = False

        for dead_node in self._dead_nodes:
            msg = f"Detected dead node '{dead_node}', removing it from the rendezvous"
            logger.debug(msg)
            del state.last_heartbeats[dead_node]

            try:
                del state.participants[dead_node]

                participant_removed = True
            except KeyError:
                pass

            try:
                state.wait_list.remove(dead_node)
            except KeyError:
                pass

            try:
                state.redundancy_list.remove(dead_node)
            except KeyError:
                pass

        if participant_removed:
            # Common epilogue shared with the _remove_from_participants()
            # function of _DistributedRendezvousOpExecutor.
            _remove_participant_epilogue(state, self._settings)

    def mark_dirty(self) -> None:
        """See base class.

        If the local rendezvous state is dirty, the next sync call will try to
        write the changes back to the backend. However this attempt might fail
        if another node, which had the same state, also made changes and wrote
        them before us.
        """
        self._dirty = True


class _Action(Enum):
    """Specifies the possible actions based on the state of the rendezvous."""

    KEEP_ALIVE = 1
    ADD_TO_PARTICIPANTS = 2
    ADD_TO_WAIT_LIST = 3
    ADD_TO_REDUNDANCY_LIST = 4
    REMOVE_FROM_PARTICIPANTS = 5
    REMOVE_FROM_WAIT_LIST = 6
    REMOVE_FROM_REDUNDANCY_LIST = 7
    MARK_RENDEZVOUS_COMPLETE = 8
    MARK_RENDEZVOUS_CLOSED = 9
    SYNC = 10
    ERROR_CLOSED = 11
    ERROR_TIMEOUT = 12
    FINISH = 13


class _RendezvousContext:
    """Holds the context of the rendezvous.

    Attributes:
        node:
            The node descriptor associated with the current rendezvous handler
            instance.
        state:
            The current state of the rendezvous.
        settings:
            The rendezvous settings.
    """

    node: _NodeDesc
    state: _RendezvousState
    settings: RendezvousSettings

    def __init__(
        self, node: _NodeDesc, state: _RendezvousState, settings: RendezvousSettings
    ) -> None:
        self.node = node
        self.state = state
        self.settings = settings


class _RendezvousOpExecutor(ABC):
    """Execute rendezvous operations."""

    @abstractmethod
    def run(
        self,
        state_handler: Callable[[_RendezvousContext, float], _Action],
        deadline: float,
        update_deadline: Callable[[timedelta], float] | None = None,
    ) -> None:
        """Execute a rendezvous operation.

        An operation is run inside a state machine and is expected to transition
        the rendezvous from one state to another.

        Args:
            state_handler:
                A callable that is expected to return the next state transition
                action based on the current state of the rendezvous.
            deadline:
                The time, in seconds, at which the operation will be considered
                timed-out.
            update_deadline:
                Function to generate a new operation deadline if the current
                node may participate in the next rendezvous.
        """


class _DistributedRendezvousOpExecutor(_RendezvousOpExecutor):
    """Execute rendezvous operations using a shared state.

    Args:
        node:
            The node descriptor associated with the current rendezvous handler
            instance.
        state_holder:
            The ``RendezvousStateHolder`` to use to sync the rendezvous state
            with other nodes.
        settings:
            The rendezvous settings.
    """

    _node: _NodeDesc
    _state: _RendezvousState
    _state_holder: _RendezvousStateHolder
    _settings: RendezvousSettings

    def __init__(
        self,
        node: _NodeDesc,
        state_holder: _RendezvousStateHolder,
        settings: RendezvousSettings,
    ) -> None:
        self._node = node
        self._state_holder = state_holder
        self._settings = settings

    def _record(self, message: str, node_state: NodeState = NodeState.RUNNING) -> None:
        construct_and_record_rdzv_event(
            name=f"{self.__class__.__name__}.{get_method_name()}",
            run_id=self._settings.run_id,
            message=message,
            node_state=node_state,
            hostname=self._node.addr,
            pid=self._node.pid,
            local_id=self._node.local_id,
        )

    def run(
        self,
        state_handler: Callable[[_RendezvousContext, float], _Action],
        deadline: float,
        update_deadline: Callable[[timedelta], float] | None = None,
    ) -> None:
        """See base class."""
        action = None
        while action != _Action.FINISH:
            # Reads or writes the latest rendezvous state shared by all nodes in
            # the rendezvous. Note that our local changes might get overridden
            # by another node if that node synced its changes before us.
            has_set = self._state_holder.sync()
            if has_set is not None:
                if has_set:
                    msg = (
                        f"The node '{self._node}' has successfully synced its local changes with "
                        f"other nodes in the rendezvous '{self._settings.run_id}'."
                    )
                else:
                    msg = (
                        f"The node '{self._node}' has a stale state and failed to sync its local "
                        f"changes with other nodes in the rendezvous '{self._settings.run_id}'."
                    )

                self._record(message=msg)
                logger.debug(msg)

            self._state = self._state_holder.state

            ctx = _RendezvousContext(self._node, self._state, self._settings)

            # Determine the next action to take based on the current state of
            # the rendezvous.
            action = state_handler(ctx, deadline)

            if action == _Action.FINISH:
                continue

            if action == _Action.ERROR_CLOSED:
                raise RendezvousClosedError

            if action == _Action.ERROR_TIMEOUT:
                raise RendezvousTimeoutError

            if action == _Action.SYNC:
                # Delay the execution by one second to avoid overloading the
                # backend if we are asked to poll for state changes.
                _delay(seconds=1)
            else:
                if action == _Action.KEEP_ALIVE:
                    self._keep_alive()
                elif action == _Action.ADD_TO_PARTICIPANTS:
                    self._add_to_participants()
                elif action == _Action.ADD_TO_WAIT_LIST:
                    self._add_to_wait_list()
                elif action == _Action.ADD_TO_REDUNDANCY_LIST:
                    self._add_to_redundancy_list()
                elif action == _Action.REMOVE_FROM_PARTICIPANTS:
                    self._remove_from_participants()
                elif action == _Action.REMOVE_FROM_WAIT_LIST:
                    self._remove_from_wait_list()
                elif action == _Action.REMOVE_FROM_REDUNDANCY_LIST:
                    self._remove_from_redundancy_list()
                    # update deadline since the node may participate in rendezvous process
                    if update_deadline:
                        deadline = update_deadline(self._settings.timeout.join)
                elif action == _Action.MARK_RENDEZVOUS_COMPLETE:
                    self._mark_rendezvous_complete()
                elif action == _Action.MARK_RENDEZVOUS_CLOSED:
                    self._mark_rendezvous_closed()

                # Attempt to sync our changes back to other nodes.
                self._state_holder.mark_dirty()

    def _keep_alive(self) -> None:
        msg = (
            f"The node '{self._node}' updated its keep-alive heartbeat time for the rendezvous "
            f"'{self._settings.run_id}'. Pending sync."
        )
        self._record(message=msg)
        logger.debug(msg)

        self._state.last_heartbeats[self._node] = datetime.now(timezone.utc)

    def _add_to_participants(self) -> None:
        msg = (
            f"The node '{self._node}' added itself to the participants of round "
            f"{self._state.round} of the rendezvous '{self._settings.run_id}'. Pending sync."
        )
        self._record(message=msg)
        logger.debug(msg)

        state = self._state

        try:
            state.wait_list.remove(self._node)
        except KeyError:
            pass

        # The ranks of the participants will be set once the rendezvous is
        # complete.
        state.participants[self._node] = 0

        self._keep_alive()

        if len(state.participants) == self._settings.min_nodes:
            state.deadline = (
                datetime.now(timezone.utc) + self._settings.timeout.last_call
            )

        if len(state.participants) == self._settings.max_nodes:
            self._mark_rendezvous_complete()

    def _add_to_wait_list(self) -> None:
        msg = (
            f"The node '{self._node}' added itself to the wait list of round "
            f"{self._state.round + 1} of the rendezvous '{self._settings.run_id}'. Pending sync."
        )
        self._record(message=msg)
        logger.debug(msg)

        if self._node in self._state.redundancy_list:
            self._state.redundancy_list.remove(self._node)
        self._state.wait_list.add(self._node)

        self._keep_alive()

    def _add_to_redundancy_list(self) -> None:
        msg = (
            f"The node '{self._node}' added itself to the redundancy list of round "
            f"{self._state.round + 1} of the rendezvous '{self._settings.run_id}'. Pending sync."
        )
        self._record(message=msg)
        logger.debug(msg)

        self._state.redundancy_list.add(self._node)

        self._keep_alive()

    def _remove_from_participants(self) -> None:
        msg = (
            f"The node '{self._node}' removed itself from the participants of round "
            f"{self._state.round} of the rendezvous '{self._settings.run_id}'. Pending sync."
        )
        self._record(message=msg)
        logger.debug(msg)

        state = self._state

        del state.participants[self._node]

        del state.last_heartbeats[self._node]

        # Common epilogue shared with the sanitizer() function of
        # _BackendRendezvousStateHolder.
        _remove_participant_epilogue(state, self._settings)

    def _remove_from_wait_list(self) -> None:
        msg = (
            f"The node '{self._node}' removed itself from the wait list of round "
            f"{self._state.round + 1} of the rendezvous '{self._settings.run_id}'. Pending sync."
        )
        self._record(message=msg)
        logger.debug(msg)

        self._state.wait_list.remove(self._node)

        del self._state.last_heartbeats[self._node]

    def _remove_from_redundancy_list(self) -> None:
        msg = (
            f"The node '{self._node}' removed itself from the redundant list of round "
            f"{self._state.round + 1} of the rendezvous '{self._settings.run_id}'. Pending sync."
        )
        self._record(message=msg)
        logger.debug(msg)

        self._state.redundancy_list.remove(self._node)

        del self._state.last_heartbeats[self._node]

    def _mark_rendezvous_complete(self) -> None:
        msg = (
            f"The node '{self._node}' marked round {self._state.round} of the rendezvous "
            f"'{self._settings.run_id}' as complete. Pending sync."
        )
        self._record(message=msg, node_state=NodeState.SUCCEEDED)
        logger.debug(msg)

        state = self._state

        state.complete = True
        state.deadline = None

        # Assign the ranks.
        for rank, node in enumerate(sorted(state.participants)):
            state.participants[node] = rank

    def _mark_rendezvous_closed(self) -> None:
        msg = (
            f"The node '{self._node}' marked the rendezvous '{self._settings.run_id}' as closed. "
            "Pending sync."
        )
        self._record(message=msg, node_state=NodeState.SUCCEEDED)
        logger.debug(msg)

        self._state.closed = True


def _should_keep_alive(ctx: _RendezvousContext) -> bool:
    """Determine whether a keep-alive heartbeat should be sent."""
    try:
        last_heartbeat = ctx.state.last_heartbeats[ctx.node]
    except KeyError:
        return False

    return (
        last_heartbeat <= datetime.now(timezone.utc) - ctx.settings.keep_alive_interval
    )


class _RendezvousExitOp:
    """Represent a rendezvous exit operation."""

    def __call__(self, ctx: _RendezvousContext, deadline: float) -> _Action:
        if ctx.node in ctx.state.participants:
            if time.monotonic() > deadline:
                return _Action.ERROR_TIMEOUT
            return _Action.REMOVE_FROM_PARTICIPANTS
        return _Action.FINISH


class _RendezvousJoinOp:
    """Represent a rendezvous join operation."""

    def __call__(self, ctx: _RendezvousContext, deadline: float) -> _Action:
        state = ctx.state

        # A closed rendezvous means that it no longer accepts new nodes.
        if state.closed:
            if ctx.node in state.redundancy_list:
                msg = f"The rendezvous '{ctx.settings.run_id}' is closed, terminating pending rendezvous."
                raise RendezvousGracefulExitError(msg)
            return _Action.ERROR_CLOSED

        if ctx.node in state.redundancy_list:
            msg = f"The node {ctx.node} is in redundancy list"
            logger.debug(msg)
            # don't apply the timeout logic here, since we want to allow the node to rejoin
            if len(state.participants) == ctx.settings.max_nodes:
                if _should_keep_alive(ctx):
                    return _Action.KEEP_ALIVE
                else:
                    return _Action.SYNC
            else:
                # transition to waiting state that will respect timeouts.
                msg = f"The node {ctx.node} is removed from redundancy list"
                logger.debug(msg)
                return _Action.REMOVE_FROM_REDUNDANCY_LIST

        is_participant = ctx.node in state.participants

        # If we are part of the rendezvous and it is already complete there is
        # no further action to take.
        if state.complete and is_participant:
            return _Action.FINISH

        now = time.monotonic()
        if now > deadline:
            rollback_period = 5  # 5 seconds

            # If we still have time to rollback (a short period on top of the
            # operation deadline), try to remove ourself from the rendezvous.
            # It is okay if we can't though as our keep-alive will eventually
            # expire.
            if now <= deadline + rollback_period:
                # If we are part of the rendezvous, it means we couldn't find
                # enough participants to complete it on time.
                if is_participant:
                    return _Action.REMOVE_FROM_PARTICIPANTS
                # If we are in the wait list, it means we couldn't wait till the
                # next round of the rendezvous.
                if ctx.node in state.wait_list:
                    return _Action.REMOVE_FROM_WAIT_LIST
            return _Action.ERROR_TIMEOUT

        if state.complete:
            # If we are here, it means we are not part of the rendezvous. In
            # case the rendezvous has capacity for additional participants add
            # ourself to the wait list for the next round.
            if len(state.participants) < ctx.settings.max_nodes:
                if ctx.node not in state.wait_list:
                    return _Action.ADD_TO_WAIT_LIST
            elif len(state.participants) >= ctx.settings.max_nodes:
                if (
                    ctx.node not in state.redundancy_list
                    and ctx.node not in state.wait_list
                ):
                    return _Action.ADD_TO_REDUNDANCY_LIST
        elif is_participant:
            # If the rendezvous has enough number of participants including us,
            # check whether we have passed the rendezvous deadline. If yes,
            # complete it.
            if (
                len(state.participants) >= ctx.settings.min_nodes
                and len(state.participants) <= ctx.settings.max_nodes
                and state.deadline is not None
            ):
                if state.deadline < datetime.now(timezone.utc):
                    msg = (
                        f"The node '{ctx.node}' marking the rendezvous complete, "
                        f"quorum established within deadline"
                    )
                    logger.debug(msg)
                    return _Action.MARK_RENDEZVOUS_COMPLETE
                else:
                    msg = f"The node '{ctx.node}' can't complete rendezvous: deadline reached"
                    logger.debug(msg)
            else:
                msg = f"The node '{ctx.node}' can't complete rendezvous: not enough participants"
                logger.debug(msg)
        else:
            # The rendezvous is not complete yet and we are not part of it. Try
            # to join.
            return _Action.ADD_TO_PARTICIPANTS

        if _should_keep_alive(ctx):
            return _Action.KEEP_ALIVE

        # At this point either the rendezvous is not complete, but we are part
        # of it, which means we have to wait for other participants to join; or
        # the rendezvous is complete, but we are not part of it, which means we
        # have to wait for the next round.
        return _Action.SYNC


class _RendezvousCloseOp:
    """Represent a rendezvous close operation."""

    def __call__(self, ctx: _RendezvousContext, deadline: float) -> _Action:
        if ctx.state.closed:
            return _Action.FINISH
        if time.monotonic() > deadline:
            return _Action.ERROR_TIMEOUT
        return _Action.MARK_RENDEZVOUS_CLOSED


class _RendezvousKeepAliveOp:
    """Represent a rendezvous keep-alive update operation."""

    def __call__(self, ctx: _RendezvousContext, deadline: float) -> _Action:
        if _should_keep_alive(ctx):
            if time.monotonic() > deadline:
                return _Action.ERROR_TIMEOUT
            return _Action.KEEP_ALIVE
        return _Action.FINISH


class DynamicRendezvousHandler(RendezvousHandler):
    """Represent a handler that sets up a rendezvous among a set of nodes."""

    # Static
    _node_desc_generator = _NodeDescGenerator()

    _this_node: _NodeDesc
    _settings: RendezvousSettings
    _backend_name: str
    _store: Store
    _state_holder: _RendezvousStateHolder
    _op_executor: _RendezvousOpExecutor
    _heartbeat_lock: threading.Lock
    _keep_alive_timer: _PeriodicTimer | None

    @classmethod
    def from_backend(
        cls,
        run_id: str,
        store: Store,
        backend: RendezvousBackend,
        min_nodes: int,
        max_nodes: int,
        local_addr: str | None = None,
        timeout: RendezvousTimeout | None = None,
        keep_alive_interval: int = 5,
        keep_alive_max_attempt: int = 3,
    ):
        """Create a new :py:class:`DynamicRendezvousHandler`.

        Args:
            run_id:
                The run id of the rendezvous.
            store:
                The C10d store to return as part of the rendezvous.
            backend:
                The backend to use to hold the rendezvous state.
            min_nodes:
                The minimum number of nodes to admit to the rendezvous.
            max_nodes:
                The maximum number of nodes to admit to the rendezvous.
            local_addr:
                The local node address.
            timeout:
                The timeout configuration of the rendezvous.
            keep_alive_interval:
                The amount of time a node waits before sending a heartbeat to keep
                it alive in the rendezvous.
            keep_alive_max_attempt:
                The maximum number of failed heartbeat attempts after which a node
                is considered dead.
        """
        # We associate each handler instance with a unique node descriptor.
        node = cls._node_desc_generator.generate(local_addr)

        settings = RendezvousSettings(
            run_id,
            min_nodes,
            max_nodes,
            timeout or RendezvousTimeout(),
            keep_alive_interval=timedelta(seconds=keep_alive_interval),
            keep_alive_max_attempt=keep_alive_max_attempt,
        )

        state_holder = _BackendRendezvousStateHolder(backend, settings)

        return cls(node, settings, backend.name, store, state_holder)

    def __init__(
        self,
        node: _NodeDesc,
        settings: RendezvousSettings,
        backend_name: str,
        store: Store,
        state_holder: _RendezvousStateHolder,
    ) -> None:
        if not settings.run_id:
            raise ValueError("The run id must be a non-empty string.")

        if settings.min_nodes < 1:
            raise ValueError(
                f"The minimum number of nodes ({settings.min_nodes}) must be greater than zero."
            )

        if settings.max_nodes < settings.min_nodes:
            raise ValueError(
                f"The maximum number of nodes ({settings.max_nodes}) must be greater than or equal "
                f"to the minimum number of nodes ({settings.min_nodes})."
            )

        self._this_node = node

        self._settings = settings

        self._backend_name = backend_name

        self._store = store

        self._state_holder = state_holder

        self._op_executor = _DistributedRendezvousOpExecutor(
            self._this_node, self._state_holder, self._settings
        )

        self._heartbeat_lock = threading.Lock()

        self._keep_alive_timer = None

        # Cached shared store server reference
        self._shared_tcp_store_server: dist.Store | None = None

        self._bootstrap_store_info: RendezvousStoreInfo | None = None

    def _record(
        self,
        message: str,
        node_state: NodeState = NodeState.RUNNING,
        rank: int | None = None,
    ) -> None:
        construct_and_record_rdzv_event(
            name=f"{self.__class__.__name__}.{get_method_name()}",
            run_id=self._settings.run_id,
            message=message,
            node_state=node_state,
            hostname=self._this_node.addr,
            pid=self._this_node.pid,
            local_id=self._this_node.local_id,
            rank=rank,
        )

    def _create_tcp_store_server(self, master_addr, master_port) -> dist.TCPStore:
        return dist.TCPStore(
            host_name=master_addr,
            port=master_port,
            is_master=True,
            multi_tenant=True,
        )

    @property
    def settings(self) -> RendezvousSettings:
        """Get the settings of the rendezvous."""
        return self._settings

    def get_backend(self) -> str:
        """See base class."""
        return self._backend_name

    @property
    def use_agent_store(self) -> bool:
        """See base class."""
        return os.getenv("TORCH_DISABLE_SHARE_RDZV_TCP_STORE", "0") != "1"

    def next_rendezvous(self) -> RendezvousInfo:
        """See base class."""
        msg = (
            f"The node '{self._this_node}' attempts to join the next round of the rendezvous "
            f"'{self._settings.run_id}'."
        )
        self._record(message=msg)
        logger.info(msg)

        try:
            self._stop_heartbeats()

            # Delay the execution for a small random amount of time if this is our
            # first run. This will slightly skew the rendezvous attempts across the
            # nodes and reduce the load on the backend.
            if self._state_holder.state.round == 0:
                _delay(seconds=(0, 0.3))

            exit_op = _RendezvousExitOp()
            join_op = _RendezvousJoinOp()

            deadline = self._get_deadline(self._settings.timeout.join)
            self._op_executor.run(exit_op, deadline)
            self._op_executor.run(join_op, deadline, self._get_deadline)

            self._start_heartbeats()

            rank, world_size = self._get_world()
            store = self._get_store()

        except Exception as e:
            self._record(
                message=f"{type(e).__name__}: {str(e)}",
                node_state=NodeState.FAILED,
            )
            raise

        msg = (
            f"The node '{self._this_node}' has joined round {self._state_holder.state.round} of "
            f"the rendezvous '{self._settings.run_id}' as rank {rank} in a world of size "
            f"{world_size}."
        )
        self._record(message=msg, rank=rank)
        logger.info(msg)

        # opt-out option of TCPStore sharing
        if os.getenv("TORCH_DISABLE_SHARE_RDZV_TCP_STORE", "0") == "1":
            bootstrap_store_info = RendezvousStoreInfo.build(
                rank, store, local_addr=self._this_node.addr
            )
            return RendezvousInfo(
                store,
                rank,
                world_size,
                bootstrap_store_info,
            )

        # This will only be hit when TCPStore sharing is enabled.
        # Rebuild bootstrap store info when this is the first rendezvous or
        # when the node became rank 0 but hasn't created a TCP store server
        # yet (rank changed between rounds due to elastic membership changes).
        if self._bootstrap_store_info is None or (
            rank == 0 and self._shared_tcp_store_server is None
        ):
            server_port = 0
            if rank == 0:
                self._shared_tcp_store_server = self._create_tcp_store_server(
                    self._this_node.addr, server_port
                )
                server_port = self._shared_tcp_store_server.port
            self._bootstrap_store_info = RendezvousStoreInfo.build(
                rank,
                store,
                local_addr=self._this_node.addr,
                server_port=server_port,  # For non-0 rank, this is a no-op
            )

        if self._bootstrap_store_info is None:
            raise AssertionError
        if rank == 0:
            if self._shared_tcp_store_server is None:
                raise AssertionError

        return RendezvousInfo(
            store,
            rank,
            world_size,
            self._bootstrap_store_info,  # type: ignore[assignment]
        )

    def is_closed(self) -> bool:
        """See base class."""
        try:
            with self._heartbeat_lock:
                self._state_holder.sync()

                return self._state_holder.state.closed

        except Exception as e:
            self._record(
                message=f"{type(e).__name__}: {str(e)}",
                node_state=NodeState.FAILED,
            )
            raise

    def set_closed(self) -> None:
        """See base class."""
        try:
            with self._heartbeat_lock:
                self._close()
        except Exception as e:
            self._record(
                message=f"{type(e).__name__}: {str(e)}",
                node_state=NodeState.FAILED,
            )
            raise

    def num_nodes_waiting(self) -> int:
        """See base class."""
        try:
            with self._heartbeat_lock:
                self._state_holder.sync()

                return len(self._state_holder.state.wait_list)

        except Exception as e:
            self._record(
                message=f"{type(e).__name__}: {str(e)}",
                node_state=NodeState.FAILED,
            )
            raise

    def get_run_id(self) -> str:
        """See base class."""
        return self._settings.run_id

    def shutdown(self) -> bool:
        """See base class."""
        self._stop_heartbeats()

        try:
            self._close()

            return True
        except RendezvousError as ex:
            msg = (
                f"The node '{self._this_node}' has failed to shutdown the rendezvous "
                f"'{self._settings.run_id}' due to an error of type {type(ex).__name__}."
            )
            self._record(message=msg, node_state=NodeState.FAILED)
            logger.warning(msg)

            return False
        except Exception as e:
            self._record(
                message=f"{type(e).__name__}: {str(e)}",
                node_state=NodeState.FAILED,
            )
            raise

    def _close(self) -> None:
        op = _RendezvousCloseOp()

        deadline = self._get_deadline(self._settings.timeout.close)

        self._op_executor.run(op, deadline)

        msg = f"The node '{self._this_node}' has closed the rendezvous '{self._settings.run_id}'."
        self._record(message=msg, node_state=NodeState.SUCCEEDED)
        logger.info(msg)

    @staticmethod
    def _keep_alive_weak(weak_self) -> None:
        self = weak_self()
        if self is not None:
            self._keep_alive()

    def _keep_alive(self) -> None:
        with self._heartbeat_lock:
            op = _RendezvousKeepAliveOp()

            deadline = self._get_deadline(self._settings.timeout.heartbeat)

            try:
                self._op_executor.run(op, deadline)

                msg = (
                    f"The node '{self._this_node}' has sent a keep-alive heartbeat to the rendezvous "
                    f"'{self._settings.run_id}'."
                )
                self._record(message=msg)
                logger.debug(msg)
            except RendezvousError as ex:
                msg = (
                    f"The node '{self._this_node}' has failed to send a keep-alive heartbeat to the "
                    f"rendezvous '{self._settings.run_id}' due to an error of type {type(ex).__name__}."
                )
                self._record(message=msg, node_state=NodeState.FAILED)
                logger.warning(msg)

    def _start_heartbeats(self) -> None:
        self._keep_alive_timer = _PeriodicTimer(
            self._settings.keep_alive_interval, self._keep_alive_weak, weakref.ref(self)
        )

        self._keep_alive_timer.set_name(
            f"RendezvousKeepAliveTimer_{self._this_node.local_id}"
        )

        self._keep_alive_timer.start()

    def _stop_heartbeats(self) -> None:
        if self._keep_alive_timer is None:
            return

        self._keep_alive_timer.cancel()

    def _get_world(self) -> tuple[int, int]:
        state = self._state_holder.state

        return state.participants[self._this_node], len(state.participants)

    def _wrap_store(self, store: Store) -> Store:
        key_prefix = (
            f"torch.rendezvous.{self._settings.run_id}.{self._state_holder.state.round}"
        )

        return dist.PrefixStore(key_prefix, store)

    def _get_store(self) -> Store:
        return self._wrap_store(self._store)

    def _get_deadline(self, timeout: timedelta) -> float:
        return time.monotonic() + timeout.total_seconds()


def _get_timeout(params: RendezvousParameters, key: str) -> timedelta | None:
    timeout = params.get_as_int(key + "_timeout")
    if timeout is None:
        return None
    return timedelta(seconds=timeout)


def create_handler(
    store: Store, backend: RendezvousBackend, params: RendezvousParameters
) -> DynamicRendezvousHandler:
    """Create a new :py:class:`DynamicRendezvousHandler` from the specified parameters.

    Args:
        store:
            The C10d store to return as part of the rendezvous.
        backend:
            The backend to use to hold the rendezvous state.

    +-------------------+------------------------------------------------------+
    | Parameter         | Description                                          |
    +===================+======================================================+
    | join_timeout      | The total time, in seconds, within which the         |
    |                   | rendezvous is expected to complete. Defaults to 600  |
    |                   | seconds.                                             |
    +-------------------+------------------------------------------------------+
    | last_call_timeout | An additional wait amount, in seconds, before        |
    |                   | completing the rendezvous once the minimum number of |
    |                   | nodes has been reached. Defaults to 30 seconds.      |
    +-------------------+------------------------------------------------------+
    | close_timeout     | The time, in seconds, within which the rendezvous is |
    |                   | expected to close after a call to                    |
    |                   | :py:meth:`RendezvousHandler.set_closed` or           |
    |                   | :py:meth:`RendezvousHandler.shutdown`. Defaults to   |
    |                   | 30 seconds.                                          |
    +-------------------+------------------------------------------------------+
    | heartbeat         | The time, in seconds, within which a keep-alive      |
    |                   | heartbeat is expected to complete                    |
    +-------------------+------------------------------------------------------+
    """
    try:
        timeout = RendezvousTimeout(
            _get_timeout(params, "join"),
            _get_timeout(params, "last_call"),
            _get_timeout(params, "close"),
            _get_timeout(params, "heartbeat"),
        )
        keep_alive_interval = params.get_as_int("keep_alive_interval", 5)
        if keep_alive_interval is None:
            raise TypeError(
                "You passed 'keep_alive_interval=None' as a rendezvous configuration option"
            )
        keep_alive_max_attempt = params.get_as_int("keep_alive_max_attempt", 3)
        if keep_alive_max_attempt is None:
            raise TypeError(
                "You passed 'keep_alive_max_attempt=None' as a rendezvous configuration option"
            )

        return DynamicRendezvousHandler.from_backend(
            params.run_id,
            store,
            backend,
            params.min_nodes,
            params.max_nodes,
            params.local_addr,
            timeout,
            keep_alive_interval=keep_alive_interval,
            keep_alive_max_attempt=keep_alive_max_attempt,
        )
    except Exception as e:
        construct_and_record_rdzv_event(
            message=f"{type(e).__name__}: {str(e)}",
            run_id=params.run_id,
            node_state=NodeState.FAILED,
        )
        raise
<|endoftext|>#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import atexit
import logging
import os
import shlex
import shutil
import socket
import subprocess
import tempfile
import time
from typing import TextIO


try:
    import etcd  # type: ignore[import]
except ModuleNotFoundError:
    pass


logger = logging.getLogger(__name__)


def find_free_port():
    """
    Find a free port and binds a temporary socket to it so that the port can be "reserved" until used.

    .. note:: the returned socket must be closed before using the port,
              otherwise an ``address already in use`` error will happen.
              The socket should be held and closed as close to the
              consumer of the port as possible since otherwise, there
              is a greater chance of race-condition where a different
              process may see the port as being free and take it.

    Returns: a socket bound to the reserved free port

    Usage::

    sock = find_free_port()
    port = sock.getsockname()[1]
    sock.close()
    use_port(port)
    """
    addrs = socket.getaddrinfo(
        host="localhost", port=None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
    )

    for addr in addrs:
        family, type, proto, _, _ = addr
        try:
            s = socket.socket(family, type, proto)
            s.bind(("localhost", 0))
            s.listen(0)
            return s
        except OSError as e:
            s.close()  # type: ignore[possibly-undefined]
            print(f"Socket creation attempt failed: {e}")
    raise RuntimeError("Failed to create a socket")


def stop_etcd(subprocess, data_dir: str | None = None):
    if subprocess and subprocess.poll() is None:
        logger.info("stopping etcd server")
        subprocess.terminate()
        subprocess.wait()

    if data_dir:
        logger.info("deleting etcd data dir: %s", data_dir)
        shutil.rmtree(data_dir, ignore_errors=True)


class EtcdServer:
    """
    .. note:: tested on etcd server v3.4.3.

    Starts and stops a local standalone etcd server on a random free
    port. Useful for single node, multi-worker launches or testing,
    where a sidecar etcd server is more convenient than having to
    separately setup an etcd server.

    This class registers a termination handler to shutdown the etcd
    subprocess on exit. This termination handler is NOT a substitute for
    calling the ``stop()`` method.

    The following fallback mechanism is used to find the etcd binary:

    1. Uses env var TORCHELASTIC_ETCD_BINARY_PATH
    2. Uses ``<this file root>/bin/etcd`` if one exists
    3. Uses ``etcd`` from ``PATH``

    Usage
    ::

     server = EtcdServer("/usr/bin/etcd", 2379, "/tmp/default.etcd")
     server.start()
     client = server.get_client()
     # use client
     server.stop()

    Args:
        etcd_binary_path: path of etcd server binary (see above for fallback path)
    """

    def __init__(self, data_dir: str | None = None):
        self._port = -1
        self._host = "localhost"

        root = os.path.dirname(__file__)
        default_etcd_bin = os.path.join(root, "bin/etcd")
        self._etcd_binary_path = os.environ.get(
            "TORCHELASTIC_ETCD_BINARY_PATH", default_etcd_bin
        )
        if not os.path.isfile(self._etcd_binary_path):
            self._etcd_binary_path = "etcd"

        self._base_data_dir = (
            data_dir if data_dir else tempfile.mkdtemp(prefix="torchelastic_etcd_data")
        )
        self._etcd_cmd = None
        self._etcd_proc: subprocess.Popen | None = None

    def _get_etcd_server_process(self) -> subprocess.Popen:
        if not self._etcd_proc:
            raise RuntimeError(
                "No etcd server process started. Call etcd_server.start() first"
            )
        else:
            return self._etcd_proc

    def get_port(self) -> int:
        """Return the port the server is running on."""
        return self._port

    def get_host(self) -> str:
        """Return the host the server is running on."""
        return self._host

    def get_endpoint(self) -> str:
        """Return the etcd server endpoint (host:port)."""
        return f"{self._host}:{self._port}"

    def start(
        self,
        timeout: int = 60,
        num_retries: int = 3,
        stderr: int | TextIO | None = None,
    ) -> None:
        """
        Start the server, and waits for it to be ready. When this function returns the server is ready to take requests.

        Args:
            timeout: time (in seconds) to wait for the server to be ready
                before giving up.
            num_retries: number of retries to start the server. Each retry
                will wait for max ``timeout`` before considering it as failed.
            stderr: the standard error file handle. Valid values are
                `subprocess.PIPE`, `subprocess.DEVNULL`, an existing file
                descriptor (a positive integer), an existing file object, and
                `None`.

        Raises:
            TimeoutError: if the server is not ready within the specified timeout
        """
        curr_retries = 0
        while True:
            try:
                data_dir = os.path.join(self._base_data_dir, str(curr_retries))
                os.makedirs(data_dir, exist_ok=True)
                return self._start(data_dir, timeout, stderr)
            except Exception as e:
                curr_retries += 1
                stop_etcd(self._etcd_proc)
                logger.warning(
                    "Failed to start etcd server, got error: %s, retrying", e
                )
                if curr_retries >= num_retries:
                    shutil.rmtree(self._base_data_dir, ignore_errors=True)
                    raise
        atexit.register(stop_etcd, self._etcd_proc, self._base_data_dir)

    def _start(
        self, data_dir: str, timeout: int = 60, stderr: int | TextIO | None = None
    ) -> None:
        sock = find_free_port()
        sock_peer = find_free_port()
        self._port = sock.getsockname()[1]
        peer_port = sock_peer.getsockname()[1]

        etcd_cmd = shlex.split(
            " ".join(
                [
                    self._etcd_binary_path,
                    "--enable-v2",
                    "--data-dir",
                    data_dir,
                    "--listen-client-urls",
                    f"http://{self._host}:{self._port}",
                    "--advertise-client-urls",
                    f"http://{self._host}:{self._port}",
                    "--listen-peer-urls",
                    f"http://{self._host}:{peer_port}",
                ]
            )
        )

        logger.info("Starting etcd server: [%s]", etcd_cmd)

        sock.close()
        sock_peer.close()
        self._etcd_proc = subprocess.Popen(etcd_cmd, close_fds=True, stderr=stderr)
        self._wait_for_ready(timeout)

    def get_client(self):
        """Return an etcd client object that can be used to make requests to this server."""
        return etcd.Client(
            host=self._host, port=self._port, version_prefix="/v2", read_timeout=10
        )

    def _wait_for_ready(self, timeout: int = 60) -> None:
        client = etcd.Client(
            host=f"{self._host}", port=self._port, version_prefix="/v2", read_timeout=5
        )
        max_time = time.time() + timeout

        while time.time() < max_time:
            if self._get_etcd_server_process().poll() is not None:
                # etcd server process finished
                exitcode = self._get_etcd_server_process().returncode
                raise RuntimeError(
                    f"Etcd server process exited with the code: {exitcode}"
                )
            try:
                logger.info("etcd server ready. version: %s", client.version)
                return
            except Exception:
                time.sleep(1)
        raise TimeoutError("Timed out waiting for etcd server to be ready!")

    def stop(self) -> None:
        """Stop the server and cleans up auto generated resources (e.g. data dir)."""
        logger.info("EtcdServer stop method called")
        stop_etcd(self._etcd_proc, self._base_data_dir)
<|endoftext|># Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any


"""
This file is not meant to be used directly. It serves as a stub to allow
other files to be safely imported without requiring the installation of
the 'etcd' library. The classes and methods here raise exceptions to
indicate that the real 'etcd' module is needed.
"""


class EtcdStubError(ImportError):
    """Custom exception to indicate that the real etcd module is required."""

    def __init__(self) -> None:
        super().__init__("The 'etcd' module is required but not installed.")


class EtcdAlreadyExist(Exception):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise EtcdStubError


class EtcdCompareFailed(Exception):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise EtcdStubError


class EtcdKeyNotFound(Exception):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise EtcdStubError


class EtcdWatchTimedOut(Exception):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise EtcdStubError


class EtcdEventIndexCleared(Exception):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise EtcdStubError


class EtcdException(Exception):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise EtcdStubError


class EtcdResult:
    def __init__(self) -> None:
        raise EtcdStubError


class Client:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise EtcdStubError

    def read(self, key: str) -> None:
        raise EtcdStubError

    def write(
        self, key: str, value: Any, ttl: int | None = None, **kwargs: Any
    ) -> None:
        raise EtcdStubError

    def test_and_set(
        self, key: str, value: Any, prev_value: Any, ttl: int | None = None
    ) -> None:
        raise EtcdStubError
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import random
import time
from base64 import b64decode, b64encode

# pyre-ignore[21]: Could not find name `Store` in `torch.distributed`.
from torch.distributed import Store


try:
    import etcd  # type: ignore[import]
except ModuleNotFoundError:
    from . import _etcd_stub as etcd


# Delay (sleep) for a small random amount to reduce CAS failures.
# This does not affect correctness, but will reduce requests to etcd server.
def cas_delay():
    time.sleep(random.uniform(0, 0.1))


# pyre-fixme[11]: Annotation `Store` is not defined as a type.
class EtcdStore(Store):
    """
    Implement a c10 Store interface by piggybacking on the rendezvous etcd instance.

    This is the store object returned by ``EtcdRendezvous``.
    """

    def __init__(
        self,
        etcd_client,
        etcd_store_prefix,
        # Default timeout same as in c10d/Store.hpp
        timeout: datetime.timedelta | None = None,
    ):
        super().__init__()  # required for pybind trampoline.

        self.client = etcd_client
        self.prefix = etcd_store_prefix

        if timeout is not None:
            self.set_timeout(timeout)

        if not self.prefix.endswith("/"):
            self.prefix += "/"

    def set(self, key, value):
        """
        Write a key/value pair into ``EtcdStore``.

        Both key and value may be either Python ``str`` or ``bytes``.
        """
        self.client.set(key=self.prefix + self._encode(key), value=self._encode(value))

    def get(self, key) -> bytes:
        """
        Get a value by key, possibly doing a blocking wait.

        If key is not immediately present, will do a blocking wait
        for at most ``timeout`` duration or until the key is published.


        Returns:
            value ``(bytes)``

        Raises:
            LookupError - If key still not published after timeout
        """
        b64_key = self.prefix + self._encode(key)
        kvs = self._try_wait_get([b64_key])

        if kvs is None:
            raise LookupError(f"Key {key} not found in EtcdStore")

        return self._decode(kvs[b64_key])

    def add(self, key, num: int) -> int:
        """
        Atomically increment a value by an integer amount.

        The integer is represented as a string using base 10. If key is not present,
        a default value of ``0`` will be assumed.

        Returns:
             the new (incremented) value


        """
        b64_key = self._encode(key)
        # c10d Store assumes value is an integer represented as a decimal string
        try:
            # Assume default value "0", if this key didn't yet:
            node = self.client.write(
                key=self.prefix + b64_key,
                value=self._encode(str(num)),  # i.e. 0 + num
                prevExist=False,
            )
            return int(self._decode(node.value))
        except etcd.EtcdAlreadyExist:
            pass

        while True:
            # Note: c10d Store does not have a method to delete keys, so we
            # can be sure it's still there.
            node = self.client.get(key=self.prefix + b64_key)
            new_value = self._encode(str(int(self._decode(node.value)) + num))
            try:
                node = self.client.test_and_set(
                    key=node.key, value=new_value, prev_value=node.value
                )
                return int(self._decode(node.value))
            except etcd.EtcdCompareFailed:
                cas_delay()

    # pyrefly: ignore [bad-override]
    def wait(self, keys, override_timeout: datetime.timedelta | None = None):
        """
        Wait until all of the keys are published, or until timeout.

        Raises:
            LookupError - if timeout occurs
        """
        b64_keys = [self.prefix + self._encode(key) for key in keys]
        kvs = self._try_wait_get(b64_keys, override_timeout)
        if kvs is None:
            raise LookupError("Timeout while waiting for keys in EtcdStore")
        # No return value on success

    def check(self, keys) -> bool:
        """Check if all of the keys are immediately present (without waiting)."""
        b64_keys = [self.prefix + self._encode(key) for key in keys]
        kvs = self._try_wait_get(
            b64_keys,
            override_timeout=datetime.timedelta(microseconds=1),  # as if no wait
        )
        return kvs is not None

    #
    # Encode key/value data in base64, so we can store arbitrary binary data
    # in EtcdStore. Input can be `str` or `bytes`.
    # In case of `str`, utf-8 encoding is assumed.
    #
    def _encode(self, value) -> str:
        if type(value) is bytes:
            return b64encode(value).decode()
        elif type(value) is str:
            return b64encode(value.encode()).decode()
        raise ValueError("Value must be of type str or bytes")

    #
    # Decode a base64 string (of type `str` or `bytes`).
    # Return type is `bytes`, which is more convenient with the Store interface.
    #
    def _decode(self, value) -> bytes:
        if type(value) is bytes:
            return b64decode(value)
        elif type(value) is str:
            return b64decode(value.encode())
        raise ValueError("Value must be of type str or bytes")

    #
    # Get all of the (base64-encoded) etcd keys at once, or wait until all the keys
    # are published or timeout occurs.
    # This is a helper method for the public interface methods.
    #
    # On success, a dictionary of {etcd key -> etcd value} is returned.
    # On timeout, None is returned.
    #
    def _try_wait_get(self, b64_keys, override_timeout=None):
        timeout = self.timeout if override_timeout is None else override_timeout  # type: ignore[attr-defined]
        deadline = time.time() + timeout.total_seconds()

        while True:
            # Read whole directory (of keys), filter only the ones waited for
            all_nodes = None
            try:
                all_nodes = self.client.get(key=self.prefix)
                req_nodes = {
                    node.key: node.value
                    for node in all_nodes.children
                    if node.key in b64_keys
                }

                if len(req_nodes) == len(b64_keys):
                    # All keys are available
                    return req_nodes
            except etcd.EtcdKeyNotFound:
                pass

            watch_timeout = deadline - time.time()
            if watch_timeout <= 0:
                return None

            try:
                index = all_nodes.etcd_index + 1 if all_nodes else 0
                self.client.watch(
                    key=self.prefix,
                    recursive=True,
                    timeout=watch_timeout,
                    index=index,
                )
            except etcd.EtcdWatchTimedOut:
                if time.time() >= deadline:
                    return None
                else:
                    continue
            except etcd.EtcdEventIndexCleared:
                continue
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import binascii
from base64 import b64decode, b64encode
from typing import cast

import urllib3.exceptions  # type: ignore[import]


try:
    import etcd  # type: ignore[import]
except ModuleNotFoundError:
    from . import _etcd_stub as etcd

from torch.distributed import Store

from .api import RendezvousConnectionError, RendezvousParameters, RendezvousStateError
from .dynamic_rendezvous import RendezvousBackend, Token
from .etcd_store import EtcdStore
from .utils import parse_rendezvous_endpoint


class EtcdRendezvousBackend(RendezvousBackend):
    """Represents an etcd-based rendezvous backend.

    Args:
        client:
            The ``etcd.Client`` instance to use to communicate with etcd.
        run_id:
            The run id of the rendezvous.
        key_prefix:
            The path under which to store the rendezvous state in etcd.
        ttl:
            The TTL of the rendezvous state. If not specified, defaults to two hours.
    """

    _DEFAULT_TTL = 7200  # 2 hours

    _client: etcd.Client
    _key: str
    _ttl: int

    def __init__(
        self,
        client: etcd.Client,
        run_id: str,
        key_prefix: str | None = None,
        ttl: int | None = None,
    ) -> None:
        if not run_id:
            raise ValueError("The run id must be a non-empty string.")

        self._client = client

        if key_prefix:
            self._key = key_prefix + "/" + run_id
        else:
            self._key = run_id

        if ttl and ttl > 0:
            self._ttl = ttl
        else:
            self._ttl = self._DEFAULT_TTL

    @property
    def name(self) -> str:
        """See base class."""
        return "etcd-v2"

    def get_state(self) -> tuple[bytes, Token] | None:
        """See base class."""
        try:
            result = self._client.read(self._key)
        except etcd.EtcdKeyNotFound:
            return None
        except (etcd.EtcdException, urllib3.exceptions.TimeoutError) as exc:
            raise RendezvousConnectionError(
                "The connection to etcd has failed. See inner exception for details."
            ) from exc

        return self._decode_state(result)

    def set_state(
        self, state: bytes, token: Token | None = None
    ) -> tuple[bytes, Token, bool] | None:
        """See base class."""
        base64_state = b64encode(state).decode()

        kwargs = {}

        def get_state():
            result = self.get_state()
            if result is not None:
                return *result, False
            return None

        if token:
            try:
                token = int(token)
            except ValueError:
                return get_state()

        if token:
            kwargs["prevIndex"] = token
        else:
            kwargs["prevExist"] = False

        try:
            result = self._client.write(self._key, base64_state, self._ttl, **kwargs)
        except (etcd.EtcdAlreadyExist, etcd.EtcdCompareFailed):
            result = None
        except (etcd.EtcdException, urllib3.exceptions.TimeoutError) as exc:
            raise RendezvousConnectionError(
                "The connection to etcd has failed. See inner exception for details."
            ) from exc

        if result is None:
            return get_state()

        tmp = *self._decode_state(result), True
        return tmp

    def _decode_state(self, result: etcd.EtcdResult) -> tuple[bytes, Token]:
        # pyrefly: ignore [missing-attribute]
        base64_state = result.value.encode()

        try:
            state = b64decode(base64_state)
        except binascii.Error as exc:
            raise RendezvousStateError(
                "The state object is corrupt. See inner exception for details."
            ) from exc

        # pyrefly: ignore [missing-attribute]
        return state, result.modifiedIndex


def _create_etcd_client(params: RendezvousParameters) -> etcd.Client:
    host, port = parse_rendezvous_endpoint(params.endpoint, default_port=2379)

    # The timeout
    read_timeout = cast(int, params.get_as_int("read_timeout", 60))
    if read_timeout <= 0:
        raise ValueError("The read timeout must be a positive integer.")

    # The communication protocol
    protocol = params.get("protocol", "http").strip().lower()
    if protocol != "http" and protocol != "https":
        raise ValueError("The protocol must be HTTP or HTTPS.")

    # The SSL client certificate
    ssl_cert = params.get("ssl_cert")
    if ssl_cert:
        ssl_cert_key = params.get("ssl_cert_key")
        if ssl_cert_key:
            # The etcd client expects the certificate key as the second element
            # of the `cert` tuple.
            ssl_cert = (ssl_cert, ssl_cert_key)

    # The root certificate
    ca_cert = params.get("ca_cert")

    try:
        return etcd.Client(
            host,
            port,
            read_timeout=read_timeout,
            protocol=protocol,
            cert=ssl_cert,
            ca_cert=ca_cert,
            allow_reconnect=True,
        )
    except (etcd.EtcdException, urllib3.exceptions.TimeoutError) as exc:
        raise RendezvousConnectionError(
            "The connection to etcd has failed. See inner exception for details."
        ) from exc


def create_backend(params: RendezvousParameters) -> tuple[EtcdRendezvousBackend, Store]:
    """Create a new :py:class:`EtcdRendezvousBackend` from the specified parameters.

    +--------------+-----------------------------------------------------------+
    | Parameter    | Description                                               |
    +==============+===========================================================+
    | read_timeout | The read timeout, in seconds, for etcd operations.        |
    |              | Defaults to 60 seconds.                                   |
    +--------------+-----------------------------------------------------------+
    | protocol     | The protocol to use to communicate with etcd. Valid       |
    |              | values are "http" and "https". Defaults to "http".        |
    +--------------+-----------------------------------------------------------+
    | ssl_cert     | The path to the SSL client certificate to use along with  |
    |              | HTTPS. Defaults to ``None``.                              |
    +--------------+-----------------------------------------------------------+
    | ssl_cert_key | The path to the private key of the SSL client certificate |
    |              | to use along with HTTPS. Defaults to ``None``.            |
    +--------------+-----------------------------------------------------------+
    | ca_cert      | The path to the root SSL authority certificate. Defaults  |
    |              | to ``None``.                                              |
    +--------------+-----------------------------------------------------------+
    """
    client = _create_etcd_client(params)

    backend = EtcdRendezvousBackend(
        client, params.run_id, key_prefix="/torch/elastic/rendezvous"
    )

    store = EtcdStore(client, "/torch/elastic/store")

    return backend, store
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ipaddress
import random
import re
import socket
import time
import weakref
from collections.abc import Callable
from datetime import timedelta
from threading import Event, Thread
from typing import Any


__all__ = ["parse_rendezvous_endpoint"]


def _parse_rendezvous_config(config_str: str) -> dict[str, str]:
    """Extract key-value pairs from a rendezvous configuration string.

    Args:
        config_str:
            A string in format <key1>=<value1>,...,<keyN>=<valueN>.
    """
    config: dict[str, str] = {}

    config_str = config_str.strip()
    if not config_str:
        return config

    key_values = config_str.split(",")
    for kv in key_values:
        key, *values = kv.split("=", 1)

        key = key.strip()
        if not key:
            raise ValueError(
                "The rendezvous configuration string must be in format "
                "<key1>=<value1>,...,<keyN>=<valueN>."
            )

        value: str | None
        if values:
            value = values[0].strip()
        else:
            value = None
        if not value:
            raise ValueError(
                f"The rendezvous configuration option '{key}' must have a value specified."
            )

        config[key] = value
    return config


def _try_parse_port(port_str: str) -> int | None:
    """Try to extract the port number from ``port_str``."""
    if port_str and re.match(r"^[0-9]{1,5}$", port_str):
        return int(port_str)
    return None


def parse_rendezvous_endpoint(
    endpoint: str | None, default_port: int
) -> tuple[str, int]:
    """Extract the hostname and the port number from a rendezvous endpoint.

    Args:
        endpoint:
            A string in format <hostname>[:<port>].
        default_port:
            The port number to use if the endpoint does not include one.

    Returns:
        A tuple of hostname and port number.
    """
    if endpoint is not None:
        endpoint = endpoint.strip()

    if not endpoint:
        return ("localhost", default_port)

    # An endpoint that starts and ends with brackets represents an IPv6 address.
    if endpoint[0] == "[" and endpoint[-1] == "]":
        host, *rest = endpoint, *[]
    else:
        host, *rest = endpoint.rsplit(":", 1)

    # Sanitize the IPv6 address.
    if len(host) > 1 and host[0] == "[" and host[-1] == "]":
        host = host[1:-1]

    if len(rest) == 1:
        port = _try_parse_port(rest[0])
        if port is None or port >= 2**16:
            raise ValueError(
                f"The port number of the rendezvous endpoint '{endpoint}' must be an integer "
                "between 0 and 65536."
            )
    else:
        port = default_port

    if not re.match(r"^[\w\.:-]+$", host):
        raise ValueError(
            f"The hostname of the rendezvous endpoint '{endpoint}' must be a dot-separated list of "
            "labels, an IPv4 address, or an IPv6 address."
        )

    return host, port


def _matches_machine_hostname(host: str) -> bool:
    """Indicate whether ``host`` matches the hostname of this machine.

    This function compares ``host`` to the hostname as well as to the IP
    addresses of this machine. Note that it may return a false negative if this
    machine has CNAME records beyond its FQDN or IP addresses assigned to
    secondary NICs.
    """
    if host == "localhost":
        return True

    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        addr = None

    if addr and addr.is_loopback:
        return True

    try:
        host_addr_list = socket.getaddrinfo(
            host, None, proto=socket.IPPROTO_TCP, flags=socket.AI_CANONNAME
        )
    except (ValueError, socket.gaierror) as _:
        host_addr_list = []

    host_ip_list = [host_addr_info[4][0] for host_addr_info in host_addr_list]

    this_host = socket.gethostname()
    if host == this_host:
        return True

    addr_list = socket.getaddrinfo(
        this_host, None, proto=socket.IPPROTO_TCP, flags=socket.AI_CANONNAME
    )
    for addr_info in addr_list:
        # If we have an FQDN in the addr_info, compare it to `host`.
        if addr_info[3] and addr_info[3] == host:
            return True

        # Otherwise if `host` represents an IP address, compare it to our IP
        # address.
        if addr and addr_info[4][0] == str(addr):
            return True

        # If the IP address matches one of the provided host's IP addresses
        if addr_info[4][0] in host_ip_list:
            return True

    return False


def _delay(seconds: float | tuple[float, float]) -> None:
    """Suspend the current thread for ``seconds``.

    Args:
        seconds:
            Either the delay, in seconds, or a tuple of a lower and an upper
            bound within which a random delay will be picked.
    """
    if isinstance(seconds, tuple):
        seconds = random.uniform(*seconds)
    # Ignore delay requests that are less than 10 milliseconds.
    if seconds >= 0.01:
        time.sleep(seconds)


class _PeriodicTimer:
    """Represent a timer that periodically runs a specified function.

    Args:
        interval:
            The interval, in seconds, between each run.
        function:
            The function to run.
    """

    # The state of the timer is held in a separate context object to avoid a
    # reference cycle between the timer and the background thread.
    class _Context:
        interval: float
        function: Callable[..., None]
        args: tuple[Any, ...]
        kwargs: dict[str, Any]
        stop_event: Event

    _name: str | None
    _thread: Thread | None
    _finalizer: weakref.finalize | None

    # The context that is shared between the timer and the background thread.
    _ctx: _Context

    def __init__(
        self,
        interval: timedelta,
        function: Callable[..., None],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._name = None

        self._ctx = self._Context()
        self._ctx.interval = interval.total_seconds()
        self._ctx.function = function  # type: ignore[assignment]
        self._ctx.args = args or ()
        self._ctx.kwargs = kwargs or {}
        self._ctx.stop_event = Event()

        self._thread = None
        self._finalizer = None

    @property
    def name(self) -> str | None:
        """Get the name of the timer."""
        return self._name

    def set_name(self, name: str) -> None:
        """Set the name of the timer.

        The specified name will be assigned to the background thread and serves
        for debugging and troubleshooting purposes.
        """
        if self._thread:
            raise RuntimeError("The timer has already started.")

        self._name = name

    def start(self) -> None:
        """Start the timer."""
        if self._thread:
            raise RuntimeError("The timer has already started.")

        self._thread = Thread(
            target=self._run,
            name=self._name or "PeriodicTimer",
            args=(self._ctx,),
            daemon=True,
        )

        # We avoid using a regular finalizer (a.k.a. __del__) for stopping the
        # timer as joining a daemon thread during the interpreter shutdown can
        # cause deadlocks. The weakref.finalize is a superior alternative that
        # provides a consistent behavior regardless of the GC implementation.
        self._finalizer = weakref.finalize(
            self, self._stop_thread, self._thread, self._ctx.stop_event
        )

        # We do not attempt to stop our background thread during the interpreter
        # shutdown. At that point we do not even know whether it still exists.
        self._finalizer.atexit = False

        self._thread.start()

    def cancel(self) -> None:
        """Stop the timer at the next opportunity."""
        if self._finalizer:
            self._finalizer()

    @staticmethod
    def _run(ctx) -> None:
        while not ctx.stop_event.wait(ctx.interval):
            ctx.function(*ctx.args, **ctx.kwargs)

    @staticmethod
    def _stop_thread(thread, stop_event):
        stop_event.set()

        thread.join()
<|endoftext|>#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import sys
import threading
import time


try:
    import etcd  # type: ignore[import]
except ModuleNotFoundError:
    from . import _etcd_stub as etcd

from torch.distributed.elastic.rendezvous import (
    RendezvousClosedError,
    RendezvousError,
    RendezvousHandler,
    RendezvousInfo,
    RendezvousParameters,
    RendezvousStoreInfo,
    RendezvousTimeoutError,
)

from .etcd_store import cas_delay, EtcdStore
from .utils import parse_rendezvous_endpoint


__all__ = [
    "EtcdRendezvousRetryableFailure",
    "EtcdRendezvousRetryImmediately",
    "EtcdRendezvousHandler",
    "EtcdRendezvous",
    "create_rdzv_handler",
]

_log_fmt = logging.Formatter("%(levelname)s %(asctime)s %(message)s")
_log_handler = logging.StreamHandler(sys.stderr)
_log_handler.setFormatter(_log_fmt)

logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)
logger.addHandler(_log_handler)


# Retryable failure exception means we were too late to make
# a desired state transition (e.g. because of a race condition),
# and should now restart from the beginning.
# A small delay is recommended to avoid spamming Etcd.
class EtcdRendezvousRetryableFailure(Exception):
    pass


# Similar to retryable failure, but the new state we observed suggests we
# can re-try immediately, i.e. without a need for "safety delay".
class EtcdRendezvousRetryImmediately(Exception):
    pass


# Default timeout for the rendezvous.
_DEFAULT_TIMEOUT: int = 600  # 10 minutes

# Additional waiting time after reaching the minimum number of nodes
# in case the rendezvous is elastic (min != max).
_DEFAULT_LAST_CALL_TIMEOUT: int = 30  # 30 seconds

# Various constants used internally in EtcdRendezvous
CONST_ETCD_SETUP_TTL = 5
CONST_ETCD_FROZEN_TTL = 10
CONST_ETCD_JOINABLE_EPHEMERAL_TTL = 10

# Ephemeral node TTL for worker's keep-alive key:
CONST_WORKER_KEEPALIVE_TTL = 10

# TTL for the ephemeral run_id-specific directory. All rendezvous state data
# for a specific run_id (job instance) is contained within directory.
# Its only role is to clean-up rendezvous data from old runs (for the case when
# etcd server is persistent), and has no affect on correctness, but should be
# larger than any timeouts that a worker process is expected to survive:
CONST_RUNID_SUBROOT_TTL = 7200  # 2 hours


class EtcdRendezvousHandler(RendezvousHandler):
    """
    Implements a
    :py:class:`torch.distributed.elastic.rendezvous.RendezvousHandler` interface
    backed by
    :py:class:`torch.distributed.elastic.rendezvous.etcd_rendezvous.EtcdRendezvous`.
    ``EtcdRendezvousHandler`` uses a URL to configure the type of rendezvous to
    use and to pass implementation specific configurations to the rendezvous
    module. The basic etcd rendezvous configuration URL looks like the following
    ::

     etcd://<etcd_address>:<port>/<job_id>?min_workers=<min_workers>&max_workers=<max_workers>  # noqa: W605

     -- example --

     etcd://localhost:2379/1234?min_workers=1&max_workers=3

    The URL above is interpreted as follows:

    1. Use the rendezvous handler that is registered with the ``etcd``
       scheme
    2. The ``etcd`` endpoint to use is ``localhost:2379``
    3. ``job_id == 1234`` is used as the prefix in etcd (this allows one to
       share a common etcd server for multiple jobs so long as the
       ``job_ids`` are guaranteed to be unique). Note that the job id can be
       any string (e.g. does not need to be a number) as long as it is
       unique.
    4. ``min_workers=1`` and ``max_workers=3`` specifies a range for
       membership size - Torch Distributed Elastic starts running the job as
       long as the cluster size is greater than or equal to ``min_workers``
       and admits up to ``max_workers`` into the cluster.

    Below are a full list of the parameters that can be passed to etcd
    rendezvous:

    +--------------------------------------------+--------------------------+
    | Parameter                                  | Description              |
    +============================================+==========================+
    | min_workers                                | minimum number of        |
    |                                            | workers for the          |
    |                                            | rendezvous to be valid   |
    +--------------------------------------------+--------------------------+
    | max_workers                                | maximum number of        |
    |                                            | workers to admit         |
    +--------------------------------------------+--------------------------+
    | timeout                                    | total timeout within     |
    |                                            | which next_rendezvous is |
    |                                            | expected to succeed      |
    |                                            | (default 600s)           |
    +--------------------------------------------+--------------------------+
    | last_call_timeout                          | additional wait amount   |
    |                                            | ("last call") after min  |
    |                                            | number of workers has    |
    |                                            | been reached (defaults   |
    |                                            | to 30s)                  |
    +--------------------------------------------+--------------------------+
    | etcd_prefix                                | path prefix (from etcd   |
    |                                            | root), inside which all  |
    |                                            | etcd nodes will be       |
    |                                            | created (defaults to     |
    |                                            | ``/torchelastic/p2p``)   |
    +--------------------------------------------+--------------------------+
    """

    def __init__(self, rdzv_impl: "EtcdRendezvous", local_addr: str | None):
        """
        Args:
            rdzv_impl: the implementation of the rendezvous
            local_addr: the local address of the current node
        """

        self._rdzv_impl = rdzv_impl
        self._local_addr = local_addr

    def __del__(self):
        # TODO: look into using weakref here instead.
        del self._rdzv_impl

    def get_backend(self) -> str:
        return "etcd"

    def next_rendezvous(self):
        rdzv_version, rank, world_size = self._rdzv_impl.rendezvous_barrier()

        logger.info("Creating EtcdStore as the c10d::Store implementation")
        store = self._rdzv_impl.setup_kv_store(rdzv_version)

        bootstrap_store_info = RendezvousStoreInfo.build(
            rank, store, local_addr=self._local_addr
        )
        return RendezvousInfo(store, rank, world_size, bootstrap_store_info)

    def is_closed(self):
        try:
            _, state = self._rdzv_impl.get_rdzv_state()
            return state["status"] == "closed"
        except etcd.EtcdKeyNotFound:
            # No rendezvous state, so it cannot be closed.
            return False

    def set_closed(self):
        self._rdzv_impl.set_closed()

    def num_nodes_waiting(self):
        try:
            _, state = self._rdzv_impl.get_rdzv_state()
            if state["status"] == "final":
                return state["num_workers_waiting"]
        except etcd.EtcdKeyNotFound:
            pass
        return 0

    def get_run_id(self) -> str:
        return self._rdzv_impl._run_id

    def shutdown(self) -> bool:
        try:
            self.set_closed()
            return True
        except BaseException:
            logger.warning("Shutdown failed", exc_info=True)
            return False


# TODO: we should probably handle a few additional errors,
# like EtcdLeaderElectionInProgress and EtcdWatcherCleared. These are
# only relevant for multi-node Etcd ensemble. A simple retry would work,
# but is verbose to add everywhere. Consider wrapping the client calls
# into auto-retry for these errors?
#
class EtcdRendezvous:
    """A rendezvous implementation that uses `etcd <https://etcd.io/>`__ as the backend store."""

    def __init__(
        self,
        client,
        prefix,
        run_id,
        num_min_workers,
        num_max_workers,
        timeout,
        last_call_timeout,
    ):
        self.client = client
        logger.info("Etcd machines: %s", self.client.machines)

        self._prefix = prefix
        self._run_id = run_id
        self._num_min_workers = num_min_workers
        self._num_max_workers = num_max_workers
        self._timeout = timeout
        self._last_call_timeout = last_call_timeout

        # For cleaning up TTL refresher threads (for ephemeral keys)
        self._lease_run_id_stop = None
        self._lease_this_rank_stop = None

        if not self._prefix.endswith("/"):
            self._prefix += "/"

        # Setup a permanent prefix dir, if didn't exist
        if self._prefix != "/":
            self.create_path_if_not_exists(self._prefix)

        # Lease a "sub-root" node specific to this job instance (run_id)
        self.create_path_if_not_exists(self.get_path(""), ttl=CONST_RUNID_SUBROOT_TTL)
        self._lease_run_id_stop = self.setup_lease_renewal(
            self.get_path(""), ttl=CONST_RUNID_SUBROOT_TTL
        )

        # Subdir for all rendezvous work
        self.create_path_if_not_exists(self.get_path("/rdzv"))

        # Create a rendezvous version counter, if doesn't exist
        try:
            self.client.write(
                key=self.get_path("/rdzv/version_counter"), value="0", prevExist=False
            )
        except etcd.EtcdAlreadyExist:
            pass

    def __del__(self):
        # TODO: look into using weakref here instead.
        if self._lease_run_id_stop is not None:
            self._lease_run_id_stop.set()

        if self._lease_this_rank_stop is not None:
            self._lease_this_rank_stop.set()

    def rendezvous_barrier(self):
        """
        Main entry point for next rendezvous.

        This method is blocking until rendezvous succeeds or a timeout occurs.

        Returns:
             ``(rdzv_version, rank, world_size)``

        Raises:
            RendezvousTimeoutError - timeout waiting for rendezvous
            RendezvousClosedError - rendezvous is or was closed while waiting
            RendezvousError - other persistent errors that
             render the rendezvous non-retryable
        """
        self._rendezvous_deadline = time.time() + self._timeout
        while True:
            if time.time() > self._rendezvous_deadline:
                raise RendezvousTimeoutError

            logger.info("Attempting to join next rendezvous")
            try:
                # Dis-own our lease in the previous rendezvous, if exists
                if self._lease_this_rank_stop is not None:
                    self._lease_this_rank_stop.set()

                return self.init_phase()

            except EtcdRendezvousRetryImmediately:
                # The type of failure suggests we can retry without delay
                pass

            except EtcdRendezvousRetryableFailure:
                # In case of retryable failure, wait a small delay
                # to avoid spamming etcd
                time.sleep(1)

            except RendezvousTimeoutError:
                logger.info("Rendezvous timeout occurred in EtcdRendezvousHandler")
                raise

            except RendezvousClosedError:
                logger.info(
                    "Rendezvous for run_id=%s was observed to be closed", self._run_id
                )
                raise

            except RendezvousError:
                raise

            except Exception as e:
                # In case of a general exception, wait a small delay
                # to avoid spamming etcd
                # FIXME: there are a few things that fall under this like
                # etcd.EtcdKeyNotFound, etc, which could be handled more explicitly.
                logger.info("Rendezvous attempt failed, will retry. Reason: %s", e)
                time.sleep(1)

    def init_phase(self):
        """
        Initially, the rendezvous state is expected to be one of:

        1. empty (non-existent) - in this case we try to create a new one.
        2. joinable - we try to join it.
        3. final - we announce ourselves as waiting, and go into monitoring mode

        Any other state is considered transitional, and will be retried after
        a short delay.

        Returns:
            ``(rdzv_version, rank, world_size)``

        Raises:
            RendezvousClosedError - current rendezvous was/is closed
            EtcdRendezvousRetryableFailure - observed some intermediate
             state, which is best handled by retrying later
        """
        try:
            active_version = self.try_create_rendezvous()
            state = json.loads(active_version.value)
            logger.info("New rendezvous state created: %s", state)
        except etcd.EtcdAlreadyExist:
            active_version, state = self.get_rdzv_state()
            # Note: it is possible for above query to fail (etcd.EtcdKeyNotFound),
            # but this is ok for us - just means we'll restart from beginning.
            logger.info("Observed existing rendezvous state: %s", state)

        if state["status"] == "closed":
            raise RendezvousClosedError

        if state["status"] == "joinable":
            return self.join_phase(state["version"])

        if state["status"] == "final":
            self.handle_existing_rendezvous(state["version"])
            raise EtcdRendezvousRetryImmediately

        self.try_wait_for_state_change(etcd_index=active_version.etcd_index + 1)
        raise EtcdRendezvousRetryableFailure

    def join_phase(self, expected_version):
        """
        We observed a rendezvous state in 'joinable' state, and attempt to join this
        particular version, and then wait for all other peers to join.
        """
        # Failure to join will propagate an exception, causing a re-entry.
        active_version, this_rank = self.join_rendezvous(expected_version)
        state = json.loads(active_version.value)
        logger.info(
            "Joined rendezvous version %s as rank %s. Full state: %s",
            state["version"],
            this_rank,
            state,
        )

        # If this worker was first to reach num_min_workers requirement,
        # and rendezvous is still joinable (therefore it is elastic),
        # then this worker will be responsible for waiting out the "last call"
        # timeout and closing (i.e. transitioning to 'frozen') the rendezvous
        # afterwards.
        # As a safety against a potential failure of this worker (during the
        # last call timeout), the rendezvous state is made ephemeral
        # when min_num_workers is reached.

        if this_rank == self._num_min_workers - 1 and state["status"] == "joinable":
            logger.info("Rank %s is responsible for join last call.", this_rank)
            last_call_deadline = time.time() + self._last_call_timeout
            self.handle_join_last_call(expected_version, last_call_deadline)
            logger.info("Rank %s finished join last call.", this_rank)

        # Wait for rendezvous state to be frozen, which means a fixed set of peers
        logger.info("Waiting for remaining peers.")
        active_version = self.wait_for_peers(expected_version)
        state = json.loads(active_version.value)

        if state["version"] != expected_version:
            raise AssertionError("Logic error: failed to observe version mismatch")

        return self.confirm_phase(expected_version, this_rank)

    def confirm_phase(self, expected_version, this_rank):
        """
        Once the rendezvous state transitions from 'joinable' to 'frozen',
        we have every participant confirm their membership and setup per-member
        keep-alive TTL keys, and then wait for all other participants to confirm,
        which would then successfully conclude this rendezvous.
        """
        logger.info("All peers arrived. Confirming membership.")
        self.confirm_membership(expected_version, this_rank)

        logger.info("Waiting for confirmations from all peers.")
        active_version = self.wait_for_final(expected_version)
        state = json.loads(active_version.value)

        logger.info(
            "Rendezvous version %s is complete. Final state: %s",
            state["version"],
            state,
        )

        # Rendezvous version number; our rank in it; world size
        return state["version"], this_rank, len(state["participants"])

    def handle_existing_rendezvous(self, expected_version):
        """
        Handle the case when there's an existing (state 'final) rendezvous already
        in place, and we have to announce ourselves waiting, and wait until
        the next rendezvous opportunity.
        """
        # If state is 'final' -> increment num_workers_waiting
        # Then, observe state changes:
        #   1. if it's no longer final -> bail out and re-try
        #   2. if keep alives are missing, destroy it and bail out.
        active_state = self.announce_self_waiting(expected_version)
        logger.info(
            "Added self to waiting list. Rendezvous full state: %s", active_state.value
        )

        self.wait_for_rendezvous_to_free(expected_version)
        logger.info(
            "Previously existing rendezvous state changed. Will re-try joining."
        )

    def try_create_rendezvous(self):
        """
        Create new rendezvous state or raise an exception that indicates an unexpected state (e.g. already exists).

        Raises:
             RendezvousError - on unexpected state
        """
        # Initially active_version is ephemeral - this is to handle the
        # possibility that might fail to complete the setup transaction,
        # i.e. the transition "setup" -> "joinable".
        active_version = self.client.write(
            key=self.get_path("/rdzv/active_version"),
            value=json.dumps({"status": "setup"}),
            prevExist=False,
            ttl=CONST_ETCD_SETUP_TTL,
        )

        try:
            version_counter = self.client.get(self.get_path("/rdzv/version_counter"))
            version_counter.value = str(int(version_counter.value) + 1)
            self.client.update(version_counter)
        except (etcd.EtcdKeyNotFound, etcd.EtcdCompareFailed) as e:
            raise RendezvousError(
                "Unexpected state of EtcdRendezvousHandler, worker needs to die."
            ) from e

        # Any failure below results in declaring a retryable rendezvous failure.
        # The ephemeral /rdzv/active_version will expire and someone can then
        # re-try the setup process.

        # Create directory node for participant data
        self.client.write(
            key=self.get_path(f"/rdzv/v_{version_counter.value}"),
            value=None,
            dir=True,
            prevExist=False,
        )

        # Publish rendezvous version and signal it is ready-to-be-joined.
        # If rendezvous was set closed just before this, a retry will happen,
        # where the closed condition will be handled.
        return self.client.test_and_set(
            key=self.get_path("/rdzv/active_version"),
            value=json.dumps(
                {
                    "status": "joinable",
                    "version": version_counter.value,
                    "participants": [],
                }
            ),
            prev_value=active_version.value,
        )

    def join_rendezvous(self, expected_version):
        """Helper method for the join phase."""
        # Use compare-and-swap to add self to rendezvous state:
        while True:
            cas_delay()
            active_version, state = self.get_rdzv_state()

            if state["status"] != "joinable":
                raise EtcdRendezvousRetryableFailure(
                    "Rendezvous state became non-joinable before we could join. "
                    "Must join next one."
                )

            if state["version"] != expected_version:
                raise EtcdRendezvousRetryImmediately(
                    "Rendezvous version changed. Must try join the new one."
                )

            if len(state["participants"]) >= self._num_max_workers:
                raise AssertionError(
                    "Logic error: joinable rendezvous should always have space left"
                )

            this_rank = len(state["participants"])
            state["participants"].append(this_rank)

            # When reaching min workers, or changing state to frozen, we'll set
            # the active_version node to be ephemeral.
            set_ttl: int | None = None
            if len(state["participants"]) == self._num_max_workers:
                state["status"] = "frozen"
                state["keep_alives"] = []
                set_ttl = CONST_ETCD_FROZEN_TTL
            elif len(state["participants"]) >= self._num_min_workers:
                set_ttl = CONST_ETCD_JOINABLE_EPHEMERAL_TTL

            try:
                # Compare-and-swap.
                active_version = self.client.test_and_set(
                    key=self.get_path("/rdzv/active_version"),
                    value=json.dumps(state),
                    prev_value=active_version.value,
                    ttl=set_ttl,
                )
                # We succeeded joining.
                return active_version, this_rank

            except etcd.EtcdCompareFailed:
                logger.info("Join rendezvous CAS unsuccessful, retrying")

    def wait_for_peers(self, expected_version):
        """Helper method for the join phase."""
        active_version, state = self.get_rdzv_state()
        while True:
            if state["status"] == "frozen" and state["version"] == expected_version:
                # Success, all peers arrived.
                return active_version

            elif state["status"] == "joinable" and state["version"] == expected_version:
                # Continue waiting for any interesting events.
                active_version, state = self.try_wait_for_state_change(
                    etcd_index=active_version.etcd_index + 1
                )

            else:
                # No valid transition possible at this point
                raise EtcdRendezvousRetryableFailure(
                    "Rendezvous state transition no longer possible. Must re-enter."
                )

    def confirm_membership(self, expected_version, this_rank):
        """Helper method for the confirm phase."""
        # Compare-and-swap loop
        while True:
            cas_delay()
            active_version, state = self.get_rdzv_state()

            if state["status"] != "frozen":
                raise EtcdRendezvousRetryImmediately(
                    "Rendezvous no longer frozen, before we confirmed. "
                    "Must join next one"
                )
            if state["version"] != expected_version:
                raise EtcdRendezvousRetryImmediately(
                    "Rendezvous version changed. Must try join the new one."
                )

            this_lease_key = self.get_path(
                f"/rdzv/v_{expected_version}/rank_{this_rank}"
            )
            self.client.set(this_lease_key, value=None, ttl=CONST_WORKER_KEEPALIVE_TTL)

            state["keep_alives"].append(this_lease_key)
            if len(state["keep_alives"]) == len(state["participants"]):
                # Everyone confirmed (this rank is last to do so)
                state["status"] = "final"
                state["num_workers_waiting"] = 0
                finalize = True
            else:
                finalize = False

            try:
                # Compare-and-swap. If new state is still frozen, keep it ephemeral.
                active_version = self.client.test_and_set(
                    key=self.get_path("/rdzv/active_version"),
                    value=json.dumps(state),
                    prev_value=active_version.value,
                    ttl=None if finalize else CONST_ETCD_FROZEN_TTL,
                )

                self._lease_this_rank_stop = self.setup_lease_renewal(
                    this_lease_key, ttl=CONST_WORKER_KEEPALIVE_TTL
                )
                return active_version

            except etcd.EtcdCompareFailed:
                logger.info("Confirm membership CAS unsuccessful, retrying")

    def wait_for_final(self, expected_version):
        """Helper method for the confirm phase."""
        active_version, state = self.get_rdzv_state()
        while True:
            if state["status"] == "final" and state["version"] == expected_version:
                # Success. This rendezvous is final, and we accept it.
                return active_version

            elif state["status"] == "frozen" and state["version"] == expected_version:
                # Continue waiting for any interesting events.
                active_version, state = self.try_wait_for_state_change(
                    etcd_index=active_version.etcd_index + 1
                )

            else:
                # No valid transition possible at this point
                raise EtcdRendezvousRetryableFailure(
                    "Rendezvous state transition no longer possible. Must re-enter."
                )

    def announce_self_waiting(self, expected_version):
        """
        Announce this worker is waiting (via num_workers_waiting counter) to join next
        rendezvous, but only if state and version match.
        """
        while True:
            cas_delay()
            active_version, state = self.get_rdzv_state()

            if state["status"] != "final" or state["version"] != expected_version:
                raise EtcdRendezvousRetryImmediately

            # Increment counter to signal an additional waiting worker.
            state["num_workers_waiting"] += 1

            try:
                active_version = self.client.test_and_set(
                    key=self.get_path("/rdzv/active_version"),
                    value=json.dumps(state),
                    prev_value=active_version.value,
                )
                return active_version

            except etcd.EtcdCompareFailed:
                logger.info("Announce self as waiting CAS unsuccessful, retrying")

    def wait_for_rendezvous_to_free(self, expected_version):
        """
        When there's an existing valid rendezvous in state 'final', we have to wait until the next opportunity to join.

        Such opportunity may come from:

        1. rendezvous state changed by someone else, in which case we unblock and retry.
        2. rendezvous becomes invalid because at least one member failed to renew their
           leased keep_alive node. We detect this, and destroy the rendezvous.
        """
        active_version, state = self.get_rdzv_state()
        while True:
            if state["status"] != "final" or state["version"] != expected_version:
                return

            # Check if current rendezvous state is valid, in the sense that all
            # its members are alive (renewing their lease).
            # If not, try destroy this rendezvous, so a new one can be created.
            alive_members = self.client.get(
                self.get_path(f"/rdzv/v_{expected_version}")
            )
            keep_alive_keys = [ch.key for ch in alive_members.children]

            for key in state["keep_alives"]:
                if key not in keep_alive_keys:
                    # This participant didn't renew their lease. We'll declare this
                    # rendezvous version as dead (but only if it hadn't changed)
                    logger.info("Keep-alive key %s is not renewed.", key)
                    logger.info(
                        "Rendezvous version %s is incomplete. ", expected_version
                    )
                    logger.info("Attempting to destroy it.")

                    # Compare-and-delete operation. Throws if compare failed,
                    # which means rendezvous was already destroyed/re-created/closed,
                    # and we can try to re-enter the barrier.
                    self.client.delete(
                        key=self.get_path("/rdzv/active_version"),
                        prevValue=active_version.value,
                    )

                    logger.info(
                        "Destroyed rendezvous version %s successfully.",
                        expected_version,
                    )

                    # We can return (and retry) immediately
                    return

            # Existing rendezvous seems valid, no reason to destroy it.
            # We just have to wait until something changes and re-check.
            try:
                overall_timeout = (
                    max(self._rendezvous_deadline - time.time(), 0.0) + 1.0
                )
                self.client.watch(
                    key=self.get_path("/rdzv"),
                    index=active_version.etcd_index + 1,
                    recursive=True,
                    timeout=overall_timeout,
                )
            except (etcd.EtcdEventIndexCleared, etcd.EtcdWatchTimedOut):
                pass

            if time.time() > self._rendezvous_deadline:
                raise RendezvousTimeoutError
            active_version, state = self.get_rdzv_state()

    def handle_join_last_call(self, expected_version, deadline):
        """
        After we reach min number of workers, one particular worker takes on the
        responsibility of waiting an additional timeout before closing the join window.
        If the worker responsible for this fails, the rendezvous will be destroyed due
        to expiring TTL, and the other participants will re-rendezvous.

        Here we expect to see state <joinable, expected_version>
        Exit gracefully if either:

        1. state becomes <frozen, expected_version>
        2. timeout happens (reaching deadline), in which case
           we try the transition to <frozen, expected_version>

        Exit with exception otherwise.
        """
        active_version, state = self.get_rdzv_state()
        while True:
            if state["status"] == "frozen" and state["version"] == expected_version:
                # Worker set became frozen before last-call timeout. This is possible
                # when num_max_workers is reached before the timeout.
                return

            if state["status"] != "joinable" or state["version"] != expected_version:
                raise EtcdRendezvousRetryableFailure(
                    "Rendezvous state transition no longer possible. Must re-enter."
                )

            # If timeout occurred, attempt a state transition (joinable -> frozen)
            if time.time() >= deadline:
                state["status"] = "frozen"
                state["keep_alives"] = []
                try:
                    active_version = self.client.test_and_set(
                        key=self.get_path("/rdzv/active_version"),
                        value=json.dumps(state),
                        prev_value=active_version.value,
                        ttl=CONST_ETCD_FROZEN_TTL,
                    )
                    # We successfully made this rendezvous frozen.
                    return
                except etcd.EtcdCompareFailed:
                    logger.info(
                        "Join last-call transition CAS unsuccessful. Will retry"
                    )
                    cas_delay()
                    active_version, state = self.get_rdzv_state()
                    continue

            # Timeout did not occur, so we must refresh TTL, and wait for
            # further changes. Note: we only want TTL to be refreshed if
            # state is still joinable, hence we use CAS for that here,
            # even though we don't change any of the data.
            try:
                active_version = self.client.test_and_set(
                    key=self.get_path("/rdzv/active_version"),
                    value=active_version.value,
                    prev_value=active_version.value,
                    ttl=CONST_ETCD_JOINABLE_EPHEMERAL_TTL,
                )

                # Minimize "oversleeping":
                timeout = min(
                    CONST_ETCD_JOINABLE_EPHEMERAL_TTL / 2,
                    deadline - time.time() + 1.0,  # Oversleeping by 1s is ok.
                )
                active_version, state = self.try_wait_for_state_change(
                    etcd_index=active_version.etcd_index + 1, timeout=timeout
                )
            except etcd.EtcdCompareFailed:
                logger.info("Join last-call TTL refresh CAS unsuccessful, will retry")
                cas_delay()
                active_version, state = self.get_rdzv_state()

    def set_closed(self):
        """
        Mark rendezvous 'closed' for current run_id, which is used to signal other
        participants to not attempt to perform (re-)rendezvous. This is useful
        when one of the workers decides the job is complete.
        """
        while True:
            active_version, state = self.get_rdzv_state()

            if state["status"] == "closed":
                # Already closed by someone else.
                return

            state["status"] = "closed"
            try:
                self.client.test_and_set(
                    key=self.get_path("/rdzv/active_version"),
                    value=json.dumps(state),
                    prev_value=active_version.value,
                )
                return

            except etcd.EtcdCompareFailed:
                logger.info("Set closed CAS unsuccessful, retrying")
                cas_delay()

    def get_rdzv_state(self):
        active_version = self.client.get(key=self.get_path("/rdzv/active_version"))
        return active_version, json.loads(active_version.value)

    def try_wait_for_state_change(self, etcd_index, timeout=None):
        # Don't sleep past the overall deadline (at least more than by 1s)
        overall_timeout = max(self._rendezvous_deadline - time.time(), 0.0) + 1.0
        timeout = overall_timeout if timeout is None else min(timeout, overall_timeout)

        try:
            self.client.watch(
                self.get_path("/rdzv/active_version"), index=etcd_index, timeout=timeout
            )
        except (etcd.EtcdEventIndexCleared, etcd.EtcdWatchTimedOut):
            pass

        if time.time() > self._rendezvous_deadline:
            raise RendezvousTimeoutError

        # Unfortunately, we have to do another fetch in order to get last etcd_index.
        return self.get_rdzv_state()

    def get_path(self, path):
        if not path.startswith("/"):
            path = "/" + path

        return f"{self._prefix}run_{self._run_id}{path}"

    def create_path_if_not_exists(self, full_path, ttl=None):
        try:
            self.client.write(
                key=full_path, value=None, dir=True, prevExist=False, ttl=ttl
            )
        except etcd.EtcdAlreadyExist:
            pass

    def setup_lease_renewal(self, full_path, ttl):
        # NOTE: For ephemeral key TTL renewal (~lease) to work correctly,
        # make sure you don't call any long-blocking methods that do not
        # release the Python's GIL! An example of this is calling a pybind11
        # extension function that is blocking / long-running, but is not
        # doing a scoped release of the GIL.
        def lease_worker(client, path, ttl, stop_event):
            while True:
                try:
                    client.refresh(path, ttl=ttl)
                except etcd.EtcdKeyNotFound:
                    break
                except ConnectionRefusedError:
                    # This error usually occurs during test when the server already got terminated but the
                    # python garbage collector have not yet invoked the __del__ method.
                    break

                if stop_event.wait(timeout=ttl / 2):
                    break

        lease_stop_event = threading.Event()
        lease_thread = threading.Thread(
            target=lease_worker, args=(self.client, full_path, ttl, lease_stop_event)
        )

        lease_thread.daemon = True
        lease_thread.start()

        return lease_stop_event

    def store_extra_data(self, rdzv_version, key, value):
        node = self.get_path(f"/rdzv/v_{rdzv_version}/extra_data")
        try:
            # If first time we are storing anything:
            extra_data = self.client.write(
                key=node, value=json.dumps({key: value}), prevExist=False
            )
            return
        except etcd.EtcdAlreadyExist:
            pass

        # CAS loop, to make sure we don't lose concurrent stores.
        while True:
            # We never delete extra_data. Failure here should be fatal, no special handling.
            extra_data = self.client.get(node)

            new_extra_data_value = json.loads(extra_data.value)
            new_extra_data_value[key] = value

            try:
                extra_data = self.client.test_and_set(
                    key=node,
                    value=json.dumps(new_extra_data_value),
                    prev_value=extra_data.value,
                )
                return
            except etcd.EtcdCompareFailed:
                logger.info("Store extra_data CAS unsuccessful, retrying")
                time.sleep(0.1)

    def load_extra_data(self, rdzv_version, key, timeout=None):
        # 'extra_data' node itself, and the directory it is located in:
        node = self.get_path(f"/rdzv/v_{rdzv_version}/extra_data")
        node_dir = self.get_path(f"/rdzv/v_{rdzv_version}")

        # TODO: implement timeout
        # https://github.com/pytorch/elastic/issues/12
        while True:
            # Combined wait for the node itself, and the key inside it.
            root = self.client.get(node_dir)

            # Find the extra_data node, if it exists
            extra_data = [n for n in root.children if n.key == node]
            if len(extra_data) > 1:
                raise AssertionError

            # Node for extra_data exists, check the desired key inside it.
            if len(extra_data) == 1:
                extra_data_dict = json.loads(extra_data[0].value)
                if key in extra_data_dict:
                    return extra_data_dict[key]

            # The 'extra_data' node doesn't exist, or they key isn't published yet.
            # Wait for interesting events on the extra_data node and retry.
            try:
                self.client.watch(node, index=root.etcd_index + 1)
            except (etcd.EtcdEventIndexCleared, etcd.EtcdWatchTimedOut):
                pass

    def setup_kv_store(self, rdzv_version):
        store_path = self.get_path(f"/rdzv/v_{rdzv_version}/kv")
        self.create_path_if_not_exists(store_path)
        return EtcdStore(etcd_client=self.client, etcd_store_prefix=store_path)


def _create_etcd_client(params: RendezvousParameters) -> etcd.Client:
    """Create a new ``etcd.Client`` from the specified ``RendezvousParameters``."""
    hostname, port = parse_rendezvous_endpoint(params.endpoint, 2379)

    # The communication protocol
    protocol = params.config.get("protocol")
    if protocol is None:
        protocol = "http"
    else:
        if protocol != "http" and protocol != "https":
            raise ValueError("The etcd protocol must be HTTP or HTTPS.")

    # The SSL client certificate
    ssl_cert = params.config.get("cert")
    if ssl_cert is not None:
        cert_key = params.config.get("key")
        if cert_key is not None:
            # The etcd client expects the certificate key as the second element
            # of the `cert` tuple.
            ssl_cert = (ssl_cert, cert_key)

    # The root certificate
    ca_cert = params.config.get("cacert")

    return etcd.Client(
        hostname,
        port,
        protocol=protocol,
        cert=ssl_cert,
        ca_cert=ca_cert,
        allow_reconnect=True,
    )


# Handler for torch.distributed "static" registration
def create_rdzv_handler(params: RendezvousParameters) -> RendezvousHandler:
    """
    Usage:

    ::

    rdzv_params = RendezvousParameters(
                        backend="etcd",
                        endpoint="192.168.0.42:2379",
                        run_id="123",
                        min_nodes=4,
                        max_nodes=8,
                        timeout=300,
                        last_call_timeout=30,
                        etcd_prefix="custom_prefix",
                        protocol="https",
                        cacert="/etc/kubernetes/certs/ca.crt",
                        cert="/etc/kubernetes/certs/client.crt",
                        key="/etc/kubernetes/certs/client.key")
    # -- or --
    rdzv_params = RendezvousParameters(
                        backend="etcd",
                        endpoint="192.168.0.42:2379",
                        run_id="123",
                        min_nodes=4,
                        max_nodes=8)

    etcd_rdzv_handler = create_etcd_rendezvous_handler(rdzv_params)


    Where:
        run_id - unique id for this training job instance,
        min_nodes - min number of workers expected to join the rendezvous,
        max_nodes - max number of workers allowed to join the rendezvous,
                        defaults to min_workers is not specified.
        timeout - total timeout within which next_rendezvous is expected to
                      succeed; a RendezvousTimeoutError is raised otherwise;
                      Defaults is 600 (10 minutes).
        last_call_timeout - additional wait amount ("last call") after
                            min number of workers has been reached.
                            Defaults to 30 seconds.
        etcd_prefix - path prefix (from etcd root), inside which all
                      etcd nodes will be created.
                      Default is "/torchelastic/p2p".
        protocol - http (default) or https to access etcd.
        cacert - CA cert to access etcd, only makes sense with https.
        cert - client cert to access etcd, only makes sense with https.
        key - client key to access etcd, only makes sense with https.
    """
    client = _create_etcd_client(params)

    etcd_prefix = params.get("etcd_prefix", "/torchelastic/p2p")

    rdzv = EtcdRendezvous(
        client=client,
        prefix=etcd_prefix,
        run_id=params.run_id,
        num_min_workers=params.min_nodes,
        num_max_workers=params.max_nodes,
        timeout=params.get_as_int("timeout", _DEFAULT_TIMEOUT),
        last_call_timeout=params.get_as_int(
            "last_call_timeout", _DEFAULT_LAST_CALL_TIMEOUT
        ),
    )
    return EtcdRendezvousHandler(
        rdzv_impl=rdzv,
        local_addr=params.local_addr,
    )
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import binascii
import logging
import os
import tempfile
from base64 import b64decode, b64encode
from datetime import timedelta
from typing import Any, cast

from torch.distributed import FileStore, Store, TCPStore
from torch.distributed.elastic.events import construct_and_record_rdzv_event, NodeState

from .api import (
    RendezvousConnectionError,
    RendezvousError,
    RendezvousParameters,
    RendezvousStateError,
)
from .dynamic_rendezvous import RendezvousBackend, Token
from .utils import _matches_machine_hostname, parse_rendezvous_endpoint


logger = logging.getLogger(__name__)

# default port for the TCP store
DEFAULT_PORT = 29400


class C10dRendezvousBackend(RendezvousBackend):
    """Represents a C10d-backed rendezvous backend.

    Args:
        store:
            The :py:class:`torch.distributed.Store` instance to use to
            communicate with the C10d store.
        run_id:
            The run id of the rendezvous.
    """

    # See the explanation in the __init__ method.
    _NULL_SENTINEL = "Y2FuaW1hZGFt"

    _store: Store
    _key: str

    def __init__(self, store: Store, run_id: str) -> None:
        if not run_id:
            raise ValueError("The run id must be a non-empty string.")

        self._store = store

        self._key = "torch.rendezvous." + run_id

        # The read operation of a store blocks the caller until the specified
        # key becomes available. This behavior makes it tricky to use a store
        # as a regular key-value dictionary.
        #
        # As a workaround we initially set a sentinel value as the rendezvous
        # state. Whenever this value gets returned we treat it as a None.
        self._call_store("compare_set", self._key, "", self._NULL_SENTINEL)

    @property
    def name(self) -> str:
        """See base class."""
        return "c10d"

    def get_state(self) -> tuple[bytes, Token] | None:
        """See base class."""
        base64_state: bytes = self._call_store("get", self._key)

        return self._decode_state(base64_state)

    def set_state(
        self, state: bytes, token: Token | None = None
    ) -> tuple[bytes, Token, bool] | None:
        """See base class."""
        base64_state_str: str = b64encode(state).decode()

        if token:
            # Shortcut if we know for sure that the token is not valid.
            if not isinstance(token, bytes):
                result = self.get_state()
                if result is not None:
                    return *result, False
                return None

            token = token.decode()
        else:
            token = self._NULL_SENTINEL

        base64_state: bytes = self._call_store(
            "compare_set", self._key, token, base64_state_str
        )

        state_token_pair = self._decode_state(base64_state)
        if state_token_pair is None:
            return None

        new_state, new_token = state_token_pair

        # C10d Store's compare_set method does not offer an easy way to find out
        # whether our write attempt was successful. As a brute-force solution we
        # perform a bitwise comparison of our local state and the remote state.
        return new_state, new_token, new_state == state

    def _call_store(self, store_op: str, *args, **kwargs) -> Any:
        try:
            return getattr(self._store, store_op)(*args, **kwargs)
        except (ValueError, RuntimeError, TimeoutError) as exc:
            raise RendezvousConnectionError(
                "The connection to the C10d store has failed. See inner exception for details."
            ) from exc

    def _decode_state(self, base64_state: bytes) -> tuple[bytes, Token] | None:
        if base64_state == self._NULL_SENTINEL.encode():
            return None

        try:
            state = b64decode(base64_state)
        except binascii.Error as exc:
            raise RendezvousStateError(
                "The state object is corrupt. See inner exception for details."
            ) from exc

        return state, base64_state


def _create_tcp_store(params: RendezvousParameters) -> TCPStore:
    host, port = parse_rendezvous_endpoint(params.endpoint, default_port=DEFAULT_PORT)

    cfg_is_host = params.get_as_bool("is_host")
    # If the user has explicitly specified whether our process should host
    # the store, respect it.
    if cfg_is_host is not None:
        is_host = cfg_is_host
    # Otherwise try to determine whether we are the host based on our hostname
    # and IP address.
    else:
        is_host = _matches_machine_hostname(host)

    # The timeout
    read_timeout = cast(int, params.get_as_int("read_timeout", 60))
    if read_timeout <= 0:
        raise ValueError("The read timeout must be a positive integer.")

    # In specific cases we attempt to instantiate the store twice. For details
    # see the explanation in the except clause below.
    for is_server in [is_host, False]:
        try:
            store = TCPStore(
                host,
                port,
                is_master=is_server,
                multi_tenant=True,
                timeout=timedelta(seconds=read_timeout),
            )

            if is_server:
                msg = f"Process {os.getpid()} hosts the TCP store for the C10d rendezvous backend."
                construct_and_record_rdzv_event(
                    run_id=params.run_id, message=msg, node_state=NodeState.INIT
                )
                logger.info(msg)

            break
        except (ValueError, RuntimeError, TimeoutError) as exc:
            # If we heuristically inferred the value of is_host as True and our
            # first attempt to instantiate the TCP store has failed, try it one
            # more time with is_host set to False. As an edge case there can be
            # more than one process that is part of the same rendezvous on this
            # machine and only one of them will eventually host the store.

            if not is_server or cfg_is_host is not None:
                raise RendezvousConnectionError(
                    "The connection to the C10d store has failed. See inner exception for details."
                ) from exc

    return store  # type: ignore[possibly-undefined]


def _create_file_store(params: RendezvousParameters) -> FileStore:
    # If a user specifies an endpoint, we treat it as a path to a file.
    if params.endpoint:
        path = params.endpoint
    else:
        try:
            # The temporary file is readable and writable only by the user of
            # this process.
            _, path = tempfile.mkstemp()
        except OSError as exc:
            raise RendezvousError(
                "The file creation for C10d store has failed. See inner exception for details."
            ) from exc

    try:
        store = FileStore(path)
    except (ValueError, RuntimeError) as exc:
        raise RendezvousConnectionError(
            "The connection to the C10d store has failed. See inner exception for details."
        ) from exc

    return store


def create_backend(params: RendezvousParameters) -> tuple[C10dRendezvousBackend, Store]:
    """Create a new :py:class:`C10dRendezvousBackend` from the specified parameters.

    +--------------+-----------------------------------------------------------+
    | Parameter    | Description                                               |
    +==============+===========================================================+
    | store_type   | The type of the C10d store. The currently supported types |
    |              | are "tcp" and "file" which correspond to                  |
    |              | :py:class:`torch.distributed.TCPStore` and                |
    |              | :py:class:`torch.distributed.FileStore`, respectively.    |
    |              | Defaults to "tcp".                                        |
    +--------------+-----------------------------------------------------------+
    | read_timeout | The read timeout, in seconds, for store operations.       |
    |              | Defaults to 60 seconds.                                   |
    |              |                                                           |
    |              | Note this only applies to                                 |
    |              | :py:class:`torch.distributed.TCPStore`. It is not relevant|
    |              | to :py:class:`torch.distributed.FileStore` which does not |
    |              | take in timeout as a parameter.                           |
    +--------------+-----------------------------------------------------------+
    | is_host      | A boolean value indicating whether this backend instance  |
    |              | will host the C10d store. If not specified it will be     |
    |              | inferred heuristically by matching the hostname or the IP |
    |              | address of this machine against the specified rendezvous  |
    |              | endpoint. Defaults to ``None``.                           |
    |              |                                                           |
    |              | Note that this configuration option only applies to       |
    |              | :py:class:`torch.distributed.TCPStore`. In normal         |
    |              | circumstances you can safely skip it; the only time when  |
    |              | it is needed is if its value cannot be correctly          |
    |              | determined (e.g. the rendezvous endpoint has a CNAME as   |
    |              | the hostname or does not match the FQDN of the machine).  |
    +--------------+-----------------------------------------------------------+
    """
    # As of today we only support TCPStore and FileStore. Other store types do
    # not have the required functionality (e.g. compare_set) yet.
    store_type = params.get("store_type", "tcp").strip().lower()
    store: Store

    try:
        if store_type == "file":
            store = _create_file_store(params)
        elif store_type == "tcp":
            store = _create_tcp_store(params)
        else:
            raise ValueError(
                "Invalid store type given. Currently only supports file and tcp."
            )

        backend = C10dRendezvousBackend(store, params.run_id)

    except Exception as e:
        construct_and_record_rdzv_event(
            message=f"{type(e).__name__}: {str(e)}",
            run_id=params.run_id,
            node_state=NodeState.FAILED,
        )
        raise

    return backend, store
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import socket
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from torch.distributed import Store
from torch.distributed.elastic.utils.distributed import get_free_port


__all__ = [
    "RendezvousClosedError",
    "RendezvousConnectionError",
    "RendezvousError",
    "RendezvousGracefulExitError",
    "RendezvousHandler",
    "RendezvousHandlerCreator",
    "RendezvousHandlerRegistry",
    "RendezvousInfo",
    "RendezvousParameters",
    "RendezvousStateError",
    "RendezvousStoreInfo",
    "RendezvousTimeoutError",
    "rendezvous_handler_registry",
]


class RendezvousError(Exception):
    """Represents the base type for rendezvous errors."""


class RendezvousClosedError(RendezvousError):
    """Raised when a rendezvous is closed."""


class RendezvousTimeoutError(RendezvousError):
    """Raised when a rendezvous did not complete on time."""


class RendezvousConnectionError(RendezvousError):
    """Raised when the connection to a rendezvous backend has failed."""


class RendezvousStateError(RendezvousError):
    """Raised when the state of a rendezvous is corrupt."""


class RendezvousGracefulExitError(RendezvousError):
    """Raised when node wasn't included in rendezvous and gracefully exits.

    Exception is a mechanism to exit the stack, however does not mean a failure.
    """


@dataclass
class RendezvousStoreInfo:
    """Store address and port that can be used to bootstrap trainer distributed comms"""

    MASTER_ADDR_KEY: ClassVar[str] = "MASTER_ADDR"
    MASTER_PORT_KEY: ClassVar[str] = "MASTER_PORT"
    master_addr: str
    master_port: int

    @staticmethod
    def build(
        rank: int,
        store: Store,
        local_addr: str | None,
        server_port: int | None = None,
    ) -> "RendezvousStoreInfo":
        """Factory method, finds unused new port on rank0 host and addr/port info with all ranks.

        If master_addr/master_port is known (useful when sharing existing tcp store server) use the constructor.

        Args:
            rank: rank of the current node
            store: store to use for rendezvous
            local_addr: address of the current node, if not provided will be resolved from hostname
            server_port: port of the TCPStore server, when the TCPStore is shared.
        """
        # TODO swap to collectives comms API
        if rank == 0:
            addr = local_addr or socket.getfqdn()
            # When TCPStore is not shared, we fallback to get_free_port.
            port = server_port or get_free_port()
            store.set(
                RendezvousStoreInfo.MASTER_ADDR_KEY,
                addr.encode(encoding="UTF-8"),  # type: ignore[arg-type]
            )
            store.set(
                RendezvousStoreInfo.MASTER_PORT_KEY,
                str(port).encode(encoding="UTF-8"),  # type: ignore[arg-type]
            )

        addr = store.get(RendezvousStoreInfo.MASTER_ADDR_KEY).decode(encoding="UTF-8")
        port = int(
            store.get(RendezvousStoreInfo.MASTER_PORT_KEY).decode(encoding="UTF-8")
        )
        return RendezvousStoreInfo(master_addr=addr, master_port=port)


class RendezvousInfo:
    """Holds the information about the rendezvous."""

    def __init__(
        self,
        store: Store,
        rank: int,
        world_size: int,
        bootstrap_store_info: RendezvousStoreInfo,
    ):
        self._store = store
        self._rank = rank
        self._world_size = world_size
        self._bootstrap_store_info = bootstrap_store_info

    @property
    def store(self) -> Store:
        """Store used by torchelastic control plane"""
        return self._store

    @property
    def rank(self) -> int:
        """Rank within a group"""
        return self._rank

    @property
    def world_size(self) -> int:
        """Global group size"""
        return self._world_size

    @property
    def bootstrap_store_info(self) -> RendezvousStoreInfo | None:
        """Store information that can used by trainer code to bootstrap distributed comms."""
        return self._bootstrap_store_info


class RendezvousHandler(ABC):
    """Main rendezvous interface.

    Note:
        Distributed Torch users normally **do not** need to implement their own
        ``RendezvousHandler``. An implementation based on C10d Store is already
        provided, and is recommended for most users.
    """

    @abstractmethod
    def get_backend(self) -> str:
        """Return the name of the rendezvous backend."""

    @property
    def use_agent_store(self) -> bool:
        """Indicates that store reference returned by :py:meth:`next_rendezvous` can be shared with user
        applications and will be available during application lifecycle.

        Rendezvous handler impl will share store details as instance of :py:class:`RendezvousStoreInfo`.
        Applications as a convention use `MASTER_ADDR`/`MASTER_PORT` env variables to lookup the store.
        """
        return False

    @abstractmethod
    def next_rendezvous(self) -> RendezvousInfo:
        """Main entry-point into the rendezvous barrier.

        Blocks until the rendezvous is complete and the current process is
        included in the formed worker group, or a timeout occurs, or the
        rendezvous was marked closed.

        Returns:
            Instance of :py:class:`RendezvousInfo`.

        Raises:
            RendezvousClosedError:
                The rendezvous is closed.
            RendezvousConnectionError:
                The connection to the rendezvous backend has failed.
            RendezvousStateError:
                The rendezvous state is corrupt.
            RendezvousTimeoutError:
                The rendezvous did not complete on time.
        """

    @abstractmethod
    def is_closed(self) -> bool:
        """Check whether the rendezvous has been closed.

        A closed rendezvous means all future attempts to re-rendezvous within
        same job will fail.

        ``is_closed()`` and :py:meth:`set_closed` have semantics of eventual
        propagation and should not be used for synchronization. The intention is
        that if at least one node decides the job is finished, it will close the
        rendezvous, and other nodes will soon observe this and stop running as
        well.
        """

    @abstractmethod
    def set_closed(self):
        """Mark the rendezvous as closed."""

    @abstractmethod
    def num_nodes_waiting(self) -> int:
        """Return the number of nodes who arrived late at the rendezvous
        barrier, hence were not included in the current worker group.

        Callers should periodically call this method to check whether new
        nodes are waiting to join the job and if so admit them by calling
        :py:meth:`next_rendezvous()` (re-rendezvous).
        """

    @abstractmethod
    def get_run_id(self) -> str:
        """Return the run id of the rendezvous.

        The run id is a user-defined id that uniquely identifies an instance of
        a distributed application. It typically maps to a job id and is used to
        allow nodes to join the correct distributed application.
        """

    @abstractmethod
    def shutdown(self) -> bool:
        """Close all resources that were open for the rendezvous.

        Example::

            rdzv_handler = ...
            try:
                store, rank, world_size = rdzv_handler.next_rendezvous()
            finally:
                rdzv_handler.shutdown()
        """


class RendezvousParameters:
    """Hold the parameters to construct a :py:class:`RendezvousHandler`.

    Args:
        backend:
            The name of the backend to use to handle the rendezvous.
        endpoint:
            The endpoint of the rendezvous, usually in form <hostname>[:<port>].
        run_id:
            The id of the rendezvous.
        min_nodes:
            The minimum number of nodes to admit to the rendezvous.
        max_nodes:
            The maximum number of nodes to admit to the rendezvous.
        local_addr:
            The address of the local node.
        **kwargs:
            Additional parameters for the specified backend.
    """

    def __init__(
        self,
        backend: str,
        endpoint: str,
        run_id: str,
        min_nodes: int,
        max_nodes: int,
        local_addr: str | None = None,
        **kwargs,
    ):
        if not backend:
            raise ValueError("The rendezvous backend name must be a non-empty string.")

        if min_nodes < 1:
            raise ValueError(
                f"The minimum number of rendezvous nodes ({min_nodes}) must be greater than zero."
            )
        if max_nodes < min_nodes:
            raise ValueError(
                f"The maximum number of rendezvous nodes ({max_nodes}) must be greater than or "
                f"equal to the minimum number of rendezvous nodes ({min_nodes})."
            )

        self.backend = backend
        self.endpoint = endpoint
        self.run_id = run_id
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.config = kwargs
        self.local_addr = local_addr

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for ``key`` if ``key`` exists, else ``default``."""
        return self.config.get(key, default)

    def get_as_bool(self, key: str, default: bool | None = None) -> bool | None:
        """Return the value for ``key`` as a ``bool``."""
        value = self.get(key, default)
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, int):
            if value == 1:
                return True
            if value == 0:
                return False
        elif isinstance(value, str):
            if value.lower() in ["1", "true", "t", "yes", "y"]:
                return True
            if value.lower() in ["0", "false", "f", "no", "n"]:
                return False
        raise ValueError(
            f"The rendezvous configuration option '{key}' does not represent a valid boolean value."
        )

    def get_as_int(self, key: str, default: int | None = None) -> int | None:
        """Return the value for ``key`` as an ``int``."""
        value = self.get(key, default)
        if value is None:
            return value
        try:
            return int(value)
        except ValueError as e:
            raise ValueError(
                f"The rendezvous configuration option '{key}' does not represent a valid integer "
                "value."
            ) from e


RendezvousHandlerCreator = Callable[[RendezvousParameters], RendezvousHandler]


class RendezvousHandlerRegistry:
    """Represent a registry of :py:class:`RendezvousHandler` backends."""

    _registry: dict[str, RendezvousHandlerCreator]

    def __init__(self) -> None:
        self._registry = {}

    def register(self, backend: str, creator: RendezvousHandlerCreator) -> None:
        """Register a new rendezvous backend.

        Args:
            backend:
                The name of the backend.
            creator:
                The callback to invoke to construct the
                :py:class:`RendezvousHandler`.
        """
        if not backend:
            raise ValueError("The rendezvous backend name must be a non-empty string.")

        current_creator: RendezvousHandlerCreator | None
        try:
            current_creator = self._registry[backend]
        except KeyError:
            current_creator = None

        if current_creator is not None and current_creator != creator:
            raise ValueError(
                f"The rendezvous backend '{backend}' cannot be registered with '{creator}' as it "
                f"is already registered with '{current_creator}'."
            )

        self._registry[backend] = creator

    def create_handler(self, params: RendezvousParameters) -> RendezvousHandler:
        """Create a new :py:class:`RendezvousHandler`."""
        try:
            creator = self._registry[params.backend]
        except KeyError as e:
            raise ValueError(
                f"The rendezvous backend '{params.backend}' is not registered. Did you forget "
                f"to call `{self.register.__name__}`?"
            ) from e

        handler = creator(params)

        # Do some sanity check.
        if handler.get_backend() != params.backend:
            raise RuntimeError(
                f"The rendezvous backend '{handler.get_backend()}' does not match the requested "
                f"backend '{params.backend}'."
            )

        return handler


# The default global registry instance used by launcher scripts to instantiate
# rendezvous handlers.
rendezvous_handler_registry = RendezvousHandlerRegistry()
<|endoftext|>#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Library that launches and manages ``n`` copies of worker subprocesses either specified by a function or a binary.

For functions, it uses ``torch.multiprocessing`` (and therefore python
``multiprocessing``) to spawn/fork worker processes. For binaries it uses python
``subprocessing.Popen`` to create worker processes.


Usage 1: Launching two trainers as a function

::

 from torch.distributed.elastic.multiprocessing import Std, start_processes


 def trainer(a, b, c):
     pass  # train


 # runs two trainers
 # LOCAL_RANK=0 trainer(1,2,3)
 # LOCAL_RANK=1 trainer(4,5,6)
 ctx = start_processes(
     name="trainer",
     entrypoint=trainer,
     args={0: (1, 2, 3), 1: (4, 5, 6)},
     envs={0: {"LOCAL_RANK": 0}, 1: {"LOCAL_RANK": 1}},
     log_dir="/tmp/foobar",
     redirects=Std.ALL,  # write all worker stdout/stderr to a log file
     tee={0: Std.ERR},  # tee only local rank 0's stderr to console
 )

 # waits for all copies of trainer to finish
 ctx.wait()

Usage 2: Launching 2 echo workers as a binary

::

 # same as invoking
 # echo hello
 # echo world > stdout.log
 ctx = start_processes(
         name="echo"
         entrypoint="echo",
         log_dir="/tmp/foobar",
         args={0: "hello", 1: "world"},
         redirects={1: Std.OUT},
        )

Just like ``torch.multiprocessing``, the return value of the function
:func:`start_processes` is a process context (:class:`api.PContext`). If a function
was launched, a :class:`api.MultiprocessContext` is returned and if a binary
was launched a :class:`api.SubprocessContext` is returned. Both are specific
implementations of the parent :class:`api.PContext` class.
"""

from collections.abc import Callable
from typing import Optional, Union

from torch.distributed.elastic.multiprocessing.api import (
    _validate_full_rank,
    DefaultLogsSpecs,
    LogsDest,
    LogsSpecs,
    MultiprocessContext,
    PContext,
    ProcessFailure,
    RunProcsResult,
    SignalException,
    Std,
    SubprocessContext,
    to_map,
)
from torch.distributed.elastic.utils.logging import get_logger
from torch.numa.binding import NumaOptions


__all__ = [
    "start_processes",
    "MultiprocessContext",
    "PContext",
    "ProcessFailure",
    "RunProcsResult",
    "SignalException",
    "Std",
    "LogsDest",
    "LogsSpecs",
    "DefaultLogsSpecs",
    "SubprocessContext",
    "to_map",
]


def start_processes(
    name: str,
    entrypoint: Callable | str,
    args: dict[int, tuple],
    envs: dict[int, dict[str, str]],
    logs_specs: LogsSpecs,
    log_line_prefixes: dict[int, str] | None = None,
    start_method: str = "spawn",
    numa_options: NumaOptions | None = None,
    duplicate_stdout_filters: list[str] | None = None,
    duplicate_stderr_filters: list[str] | None = None,
) -> PContext:
    """
    Start ``n`` copies of ``entrypoint`` processes with the provided options.

    ``entrypoint`` is either a ``Callable`` (function) or a ``str`` (binary).
    The number of copies is determined by the number of entries for ``args`` and
    ``envs`` arguments, which need to have the same key set.

    ``args`` and ``env`` parameters are the arguments and environment variables
    to pass down to the entrypoint mapped by the replica index (local rank).
    All local ranks must be accounted for.
    That is, the keyset should be ``{0,1,...,(nprocs-1)}``.

    .. note:: When the ``entrypoint`` is a binary (``str``), ``args`` can only be strings.
              If any other type is given, then it is casted to a string representation
              (e.g. ``str(arg1)``). Furthermore, a binary failure will only write
              an ``error.json`` error file if the main function is annotated with
              ``torch.distributed.elastic.multiprocessing.errors.record``. For function launches,
              this is done by default and there is no need to manually annotate
              with the ``@record`` annotation.

    Inside ``logs_specs``, ``redirects`` and ``tee`` are bitmasks specifying which std
    stream(s) to redirect to a log file in the ``log_dir``. Valid mask values are defined
    in ``Std``.  To redirect/tee only certain local ranks, pass ``redirects`` as a map
    with the key as the local rank to specify the redirect behavior for.
    Any missing local ranks will default to ``Std.NONE``.

    ``duplicate_stdout_filters`` and ``duplicate_stderr_filters``, if non-empty,
    duplicate stdouts and stderrs respectively specified in ``logs_specs``'s ``tee``
    to a file containing only lines that match _any_ of the filter strings. The log
    file is aggregated across all ranks selected by ``tee``.

    ``tee`` acts like the unix "tee" command in that it redirects + prints to console.
    To avoid worker stdout/stderr from printing to console, use the ``redirects`` parameter.

    For each process, the ``log_dir`` will contain:

    #. ``{local_rank}/error.json``: if the process failed, a file with the error info
    #. ``{local_rank}/stdout.log``: if ``redirect & STDOUT == STDOUT``
    #. ``{local_rank}/stderr.log``: if ``redirect & STDERR == STDERR``
    #. ``filtered_stdout.log``: if ``duplicate_stdout_filters`` is non-empty
    #. ``filtered_stderr.log``: if ``duplicate_stderr_filters`` is non-empty

    .. note:: It is expected that the ``log_dir`` exists, is empty, and is a directory.

    Example:
    ::

     log_dir = "/tmp/test"

     # ok; two copies of foo: foo("bar0"), foo("bar1")
     start_processes(
        name="trainer",
        entrypoint=foo,
        args:{0:("bar0",), 1:("bar1",),
        envs:{0:{}, 1:{}},
        log_dir=log_dir
     )

     # invalid; envs missing for local rank 1
     start_processes(
        name="trainer",
        entrypoint=foo,
        args:{0:("bar0",), 1:("bar1",),
        envs:{0:{}},
        log_dir=log_dir
     )

     # ok; two copies of /usr/bin/touch: touch file1, touch file2
     start_processes(
        name="trainer",
        entrypoint="/usr/bin/touch",
        args:{0:("file1",), 1:("file2",),
        envs:{0:{}, 1:{}},
        log_dir=log_dir
      )

     # caution; arguments casted to string, runs:
     # echo "1" "2" "3" and echo "[1, 2, 3]"
     start_processes(
        name="trainer",
        entrypoint="/usr/bin/echo",
        args:{0:(1,2,3), 1:([1,2,3],),
        envs:{0:{}, 1:{}},
        log_dir=log_dir
      )

    Args:
        name: a human readable short name that describes what the processes are
              (used as header when tee'ing stdout/stderr outputs)
        entrypoint: either a ``Callable`` (function) or ``cmd`` (binary)
        args: arguments to each replica
        envs: env vars to each replica
        log_dir: directory used to write log files
        start_method: multiprocessing start method (spawn, fork, forkserver)
                      ignored for binaries
        logs_specs: defines ``log_dir``, ``redirects``, and ``tee``.
                    inside ``logs_specs``:
                    - redirects: which std streams to redirect to a log file
                    - tee: which std streams to redirect + print to console
        local_ranks_filter: which ranks' logs to print to console
        duplicate_stdout_filters: filters for the duplicated stdout logs
        duplicate_stderr_filters: filters for the duplicated stderr logs

    """

    nprocs = len(args)
    _validate_full_rank(args, nprocs, "args")
    _validate_full_rank(envs, nprocs, "envs")

    context: PContext
    if isinstance(entrypoint, str):
        context = SubprocessContext(
            name=name,
            entrypoint=entrypoint,
            args=args,
            envs=envs,
            duplicate_stdout_filters=duplicate_stdout_filters,
            duplicate_stderr_filters=duplicate_stderr_filters,
            logs_specs=logs_specs,
            log_line_prefixes=log_line_prefixes,
            numa_options=numa_options,
        )
    else:
        context = MultiprocessContext(
            name=name,
            entrypoint=entrypoint,
            args=args,
            envs=envs,
            duplicate_stdout_filters=duplicate_stdout_filters,
            duplicate_stderr_filters=duplicate_stderr_filters,
            log_line_prefixes=log_line_prefixes,
            start_method=start_method,
            logs_specs=logs_specs,
            numa_options=numa_options,
        )

    try:
        context.start()
        return context
    except Exception:
        context.close()
        raise
<|endoftext|># mypy: allow-untyped-defs
# !/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Taken and modified from original source:
# https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
import ctypes
import logging
import os
import sys
from contextlib import contextmanager
from functools import partial


IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"


logger = logging.getLogger(__name__)


_WIN32_STD_HANDLE = {
    "stdout": -11,  # STD_OUTPUT_HANDLE
    "stderr": -12,  # STD_ERROR_HANDLE
}


def get_libc():
    if IS_MACOS:
        logger.warning("NOTE: Redirects are currently not supported in MacOs.")
        return None
    elif IS_WINDOWS:
        for lib_name in ("ucrtbase", "msvcrt", "msvcr110", "msvcr100"):
            try:
                lib = ctypes.CDLL(lib_name)
                logger.debug("Loaded Windows C runtime: %s", lib_name)
                return lib
            except OSError:
                continue
        raise RuntimeError(
            "Could not load a C runtime DLL on Windows (tried: ucrtbase, msvcrt, "
            "msvcr110, msvcr100). Redirects cannot function without a CRT."
        )
    else:
        return ctypes.CDLL("libc.so.6")


libc = get_libc()


def _c_std(stream: str):
    if IS_WINDOWS:
        stream_index = 2 if stream == "stderr" else 1
        try:
            iob_func = libc.__acrt_iob_func
            iob_func.restype = ctypes.POINTER(ctypes.c_void_p)
            iob_func.argtypes = [ctypes.c_uint]
            return iob_func(stream_index)
        except AttributeError:
            pass
        try:
            legacy_index = 2 if stream == "stderr" else 1
            iob = (ctypes.POINTER(ctypes.c_void_p) * 3).in_dll(libc, "_iob")
            return iob[legacy_index]
        except (AttributeError, OSError) as err:
            raise RuntimeError(
                f"Could not resolve C-runtime FILE* for '{stream}'. "
                "Neither __acrt_iob_func nor _iob are available in the loaded CRT."
            ) from err
    return ctypes.c_void_p.in_dll(libc, stream)


def _python_std(stream: str):
    return {"stdout": sys.stdout, "stderr": sys.stderr}[stream]


_VALID_STD = {"stdout", "stderr"}


if IS_WINDOWS:  # libc is None on macOS; all of the below is Windows-only
    import io as _io
    import msvcrt as _msvcrt

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]

    _crt_dup = libc._dup
    _crt_dup2 = libc._dup2
    _crt_dup.restype = ctypes.c_int
    _crt_dup.argtypes = [ctypes.c_int]
    _crt_dup2.restype = ctypes.c_int
    _crt_dup2.argtypes = [ctypes.c_int, ctypes.c_int]

    @contextmanager
    def redirect(std: str, to_file: str):
        """
        Redirect ``std`` (one of ``"stdout"`` or ``"stderr"``) to a file at ``to_file``.

        On Windows this performs a four-layer redirect:

        1. ``sys.stdout``/``sys.stderr`` -- rewired to a new TextIOWrapper so
           Python's ``print()`` writes to the destination file.
        2. CRT fd (``_dup2``) -- captures C ``printf`` and UCRT ``FILE*`` writers.
        3. Win32 ``SetStdHandle`` -- captures native code using ``WriteFile``/
           ``WriteConsole`` directly, including HIP/ROCm.
        4. ``fflush`` before each switch -- prevents lost output from CRT buffering.

        .. note:: If ROCm/HIP caches the Win32 HANDLE before this redirect runs
                  (e.g. at ``import torch`` time), set up the redirect *before*
                  importing torch/ROCm to capture all output.

        Directory of ``to_file`` is assumed to exist. The destination file is
        overwritten if it already exists.
        """
        if std not in _VALID_STD:
            raise ValueError(
                f"unknown standard stream <{std}>, must be one of {_VALID_STD}"
            )

        std_fd = 1 if std == "stdout" else 2
        win32_handle_id = _WIN32_STD_HANDLE[std]
        orig_sys_std = getattr(sys, std)
        orig_fd_dup = _crt_dup(std_fd)
        if orig_fd_dup == -1:
            raise OSError(f"CRT _dup failed for {std} (fd={std_fd})")
        orig_win32_handle = _kernel32.GetStdHandle(win32_handle_id)

        with open(to_file, mode="w+b") as dst:
            dst_fd = dst.fileno()

            try:
                libc.fflush(_c_std(std))
            except Exception:
                pass
            try:
                orig_sys_std.flush()
            except Exception:
                pass

            _kernel32.SetStdHandle(
                win32_handle_id,
                _msvcrt.get_osfhandle(dst_fd),  # pyrefly: ignore [missing-attribute]
            )

            if _crt_dup2(dst_fd, std_fd) == -1:
                raise OSError(f"CRT _dup2 failed redirecting {std}")

            new_sys_std = _io.TextIOWrapper(
                open(dst_fd, mode="wb", closefd=False),  # noqa: SIM115
                encoding=orig_sys_std.encoding or "utf-8",
                errors="replace",
                line_buffering=True,
            )
            setattr(sys, std, new_sys_std)

            try:
                yield
            finally:
                try:
                    new_sys_std.flush()
                except Exception:
                    pass
                try:
                    libc.fflush(_c_std(std))
                except Exception:
                    pass

                setattr(sys, std, orig_sys_std)
                _crt_dup2(orig_fd_dup, std_fd)
                os.close(orig_fd_dup)
                _kernel32.SetStdHandle(win32_handle_id, orig_win32_handle)

else:

    @contextmanager
    def redirect(std: str, to_file: str):
        """
        Redirect ``std`` (one of ``"stdout"`` or ``"stderr"``) to a file in the path specified by ``to_file``.

        This method redirects the underlying std file descriptor (not just python's ``sys.stdout|stderr``).
        See usage for details.

        Directory of ``to_file`` is assumed to exist and the destination file
        is overwritten if it already exists.

        .. note:: Due to buffering cross source writes are not guaranteed to
                  appear in wall-clock order. For instance in the example below
                  it is possible for the C-outputs to appear before the python
                  outputs in the log file.

        Usage::

            # syntactic-sugar for redirect("stdout", "tmp/stdout.log")
            with redirect_stdout("/tmp/stdout.log"):
                print("python stdouts are redirected")
                libc = ctypes.CDLL("libc.so.6")
                libc.printf(b"c stdouts are also redirected")
                os.system("echo system stdouts are also redirected")

            print("stdout restored")
        """
        if std not in _VALID_STD:
            raise ValueError(
                f"unknown standard stream <{std}>, must be one of {_VALID_STD}"
            )

        c_std = _c_std(std)
        python_std = _python_std(std)
        std_fd = python_std.fileno()

        def _redirect(dst):
            libc.fflush(c_std)
            python_std.flush()
            os.dup2(dst.fileno(), std_fd)

        with os.fdopen(os.dup(std_fd)) as orig_std, open(to_file, mode="w+b") as dst:
            _redirect(dst)
            try:
                yield
            finally:
                _redirect(orig_std)


redirect_stdout = partial(redirect, "stdout")
redirect_stderr = partial(redirect, "stderr")
<|endoftext|>#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import IntFlag
from multiprocessing import synchronize
from types import FrameType
from typing import Any, TextIO, Union

import torch.multiprocessing as mp
from torch.distributed.elastic.multiprocessing.errors import ProcessFailure, record
from torch.distributed.elastic.multiprocessing.redirects import (
    redirect_stderr,
    redirect_stdout,
)
from torch.distributed.elastic.multiprocessing.subprocess_handler import (
    get_subprocess_handler,
    SubprocessHandler,
)
from torch.distributed.elastic.multiprocessing.tail_log import TailLog
from torch.numa.binding import _maybe_wrap_with_numa_binding, NumaOptions


IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"


logger = logging.getLogger(__name__)

__all__ = [
    "DefaultLogsSpecs",
    "SignalException",
    "Std",
    "to_map",
    "RunProcsResult",
    "PContext",
    "get_std_cm",
    "MultiprocessContext",
    "SubprocessContext",
    "LogsDest",
    "LogsSpecs",
]


class SignalException(Exception):
    """
    Exception is raised inside the torchelastic agent process by the termination handler
    if the death signal got received by the process.
    """

    def __init__(self, msg: str, sigval: signal.Signals) -> None:
        super().__init__(msg)
        self.sigval = sigval


def _terminate_process_handler(signum: int, frame: FrameType | None) -> None:
    """Termination handler that raises exceptions on the main process.

    When the process receives death signal(SIGTERM, SIGINT), this termination handler will
    be invoked. It raises the ``SignalException`` exception that should be processed by the
    user code. Python does not terminate process after the termination handler is finished,
    so the exception should not be silently ignored, otherwise the process will never
    be terminated.
    """
    sigval = signal.Signals(signum)
    raise SignalException(f"Process {os.getpid()} got signal: {sigval}", sigval=sigval)


def _get_kill_signal() -> signal.Signals:
    """Get the kill signal. SIGKILL for unix, CTRL_C_EVENT for windows."""
    if IS_WINDOWS:
        return signal.CTRL_C_EVENT  # type: ignore[attr-defined]
    else:
        return signal.SIGKILL


def _get_default_signal() -> signal.Signals:
    """Get the default termination signal. SIGTERM for unix, CTRL_C_EVENT for windows."""
    if IS_WINDOWS:
        return signal.CTRL_C_EVENT  # type: ignore[attr-defined]
    else:
        return signal.SIGTERM


def _validate_full_rank(d: dict[int, Any], nprocs: int, what: str):
    actual_keys = set(d.keys())
    expected_keys = set(range(nprocs))

    if actual_keys != expected_keys:
        raise RuntimeError(
            f"{what}, local rank mapping mismatch,"
            f" expected: {expected_keys}, actual: {actual_keys}"
        )


_MAPPING_REGEX = r"^(\d:[0123],)*(\d:[0123])$"
_VALUE_REGEX = r"^[0123]$"


class Std(IntFlag):
    NONE = 0
    OUT = 1
    ERR = 2
    ALL = OUT | ERR

    @classmethod
    def from_str(cls, vm: str) -> Union["Std", dict[int, "Std"]]:
        """
        Example:
        ::

         from_str("0") -> Std.NONE
         from_str("1") -> Std.OUT
         from_str("0:3,1:0,2:1,3:2") -> {0: Std.ALL, 1: Std.NONE, 2: Std.OUT, 3: Std.ERR}

