        bufs_to_move = [p for p in buffers if p not in ignored_buffers]
        _move_states_to_device(params_to_move, bufs_to_move, device_from_device_id)
        return
    param = next(_get_orig_params(module, ignored_params), None)
    if param is not None and param.device == cpu_device:
        _warn_cpu_init()


def _move_states_to_device(
    params: list[nn.Parameter],
    buffers: list[torch.Tensor],
    device_from_device_id: torch.device | None,
) -> None:
    """
    Move states to the specified device.

    Precondition: ``_check_single_device_module()`` and module's parameters and
    buffers have been materialized if needed.
    """
    if len(params) == 0 and len(buffers) == 0:
        return
    if len(params) > 0:
        current_device = params[0].device
    elif len(buffers) > 0:
        current_device = buffers[0].device
    cpu_device = torch.device("cpu")
    if device_from_device_id is not None:
        # Move the parameters and buffers like the `.data` code path in
        # `nn.Module._apply()`, which underlies `nn.Module.to()`
        for param in params:
            with torch.no_grad():
                param.data = param.to(device_from_device_id)
                if param.grad is not None:
                    param.grad.data = param.grad.to(device_from_device_id)
        for buffer in buffers:
            buffer.data = buffer.to(device_from_device_id)
    elif current_device == cpu_device:  # type: ignore[possibly-undefined]
        _warn_cpu_init()


def _warn_cpu_init():
    warnings.warn(
        "The passed-in `module` is on CPU and will thus have FSDP's sharding "
        "initialization run on CPU, which may be slower than on GPU. We "
        "recommend passing in the `device_id` argument for FSDP to move "
        "`module` to GPU for the sharding initialization. `module` must also "
        "be on GPU device to work with the `sync_module_states=True` flag "
        "since that requires GPU communication.",
        stacklevel=2,
    )


def _get_compute_device(
    module: nn.Module,
    ignored_params: set[nn.Parameter],
    device_from_device_id: torch.device | None,
    rank: int,
    device_handle: _FSDPDeviceHandle,
) -> torch.device:
    """
    Determine and return this FSDP instance's compute device.

    If the module is already on a non-CPU device, then the compute device is that non-CPU
    device. If the module is on CPU, then the compute device is the current
    device.

    Since this method should be called after materializing the module, any
    non-CPU device should not be meta device. For now, the compute device is
    always a CUDA or CUDA-like device with its explicit index.

    Precondition: ``_check_single_device_module()`` and
    ``_move_module_to_device()``.
    """
    param = next(_get_orig_params(module, ignored_params), None)
    if param is not None and param.device.type != "cpu":
        compute_device = param.device  # Determined by model param placement
    else:
        compute_device = torch.device(device_handle.current_device())
    if device_from_device_id is not None and compute_device != device_from_device_id:
        raise ValueError(
            f"Inconsistent compute device and `device_id` on rank {rank}: "
            f"{compute_device} vs {device_from_device_id}"
        )
    return compute_device


# TODO: See how to deprecate!
def _sync_module_params_and_buffers(
    module: nn.Module,
    params: list[nn.Parameter],
    process_group: dist.ProcessGroup,
) -> None:
    """
    Synchronize module states (i.e. parameters ``params`` and all not-yet-synced buffers) by broadcasting from rank 0 to all ranks.

    Precondition: ``sync_module_states == True`` and ``self.process_group`` has
    been set.
    """
    module_states: list[torch.Tensor] = []
    for buffer in module.buffers():
        # Avoid re-synchronizing buffers in case of nested wrapping
        if not getattr(buffer, FSDP_SYNCED, False):
            setattr(buffer, FSDP_SYNCED, True)
            detached_buffer = buffer.detach()
            if is_traceable_wrapper_subclass(detached_buffer):
                # NOTE: Here we assume no nested subclasses, at most one level of subclass
                # in both model's buffers and params
                attrs, _ = detached_buffer.__tensor_flatten__()  # type: ignore[attr-defined]
                for attr in attrs:
                    match getattr(detached_buffer, attr):
                        case torch.Tensor() as v:
                            module_states.append(v)
                        case CustomClassBase():
                            pass
                        case unexpected:
                            raise AssertionError(
                                f"expected Tensor or CustomClassBase, got {type(unexpected)}"
                            )
            else:
                module_states.append(detached_buffer)

    for param in params:
        detached_param = param.detach()
        if is_traceable_wrapper_subclass(detached_param):
            attrs, _ = detached_param.__tensor_flatten__()  # type: ignore[attr-defined]
            for attr in attrs:
                match getattr(detached_param, attr):
                    case torch.Tensor() as v:
                        module_states.append(v)
                    case CustomClassBase():
                        pass
                    case unexpected:
                        raise AssertionError(
                            f"expected Tensor or CustomClassBase, got {type(unexpected)}"
                        )
        else:
            module_states.append(detached_param)

    _check_module_states_for_sync_module_states(module_states)
    _sync_params_and_buffers(
        process_group,
        module_states,
        PARAM_BROADCAST_BUCKET_SIZE,
        src=0,
    )


def _check_module_states_for_sync_module_states(
    module_states: list[torch.Tensor],
) -> None:
    if module_states and any(
        tensor.device == torch.device("cpu") for tensor in module_states
    ):
        raise ValueError(
            "The module has CPU parameters or buffers when `sync_module_states=True`, "
            "which requires them to be on GPU. Please specify the `device_id` argument "
            "or move the module to GPU before passing it to FSDP."
        )


def _get_orig_params(
    module: nn.Module,
    ignored_params: set[nn.Parameter],
) -> Iterator[nn.Parameter]:
    """
    Return an iterator over the original parameters in ``module``.

    The iterator does not return
    the parameters in ``ignored_params``, any ``FlatParameter`` s (which may be
    present due to nested FSDP wrapping), or any original parameters already
    flattened (only relevant when ``use_orig_params=True``).
    """
    param_gen = module.parameters()
    try:
        while True:
            param = next(param_gen)
            if param not in ignored_params and not _is_fsdp_flattened(param):
                yield param
    except StopIteration:
        pass


def _check_orig_params_flattened(
    fsdp_module,
    ignored_params: set[nn.Parameter],
) -> None:
    """
    Check that original parameters in ``fsdp_module`` have been flattened.

    The flattened parameters are made
    invisible to ``named_parameters()`` for the module hierarchy rooted at
    ``fsdp_module``. This should be called as a sanity check after flattening
    the wrapped module's parameters.
    """
    for param_name, param in _named_parameters_with_duplicates(fsdp_module):
        if param not in ignored_params and not _is_fsdp_flattened(param):
            raise RuntimeError(
                f"Found an unflattened parameter: {param_name}; "
                f"{param.size()} {param.__class__}"
            )


def _get_default_comm_hook(sharding_strategy: ShardingStrategy):
    return (
        default_hooks.allreduce_hook
        if sharding_strategy == ShardingStrategy.NO_SHARD
        else default_hooks.reduce_scatter_hook
    )


def _get_default_comm_hook_state(
    process_group: dist.ProcessGroup,
) -> default_hooks.DefaultState:
    return default_hooks.DefaultState(process_group=process_group)
<|endoftext|># mypy: allow-untyped-defs
import logging
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum

import torch
import torch.distributed as dist
import torch.distributed.fsdp._flat_param as flat_param_file
from torch.distributed.fsdp._common_utils import (
    _apply_to_modules,
    _get_module_fsdp_state,
    clean_tensor_name,
)


logger = logging.getLogger(__name__)


class SimpleProfiler:
    class Type(str, Enum):
        ALL = "all"
        ALLGATHER = "all_gather"
        ALLGATHER_OBJ = "all_gather_object"
        RESHARDING = "resharding"
        H2D = "H2D"
        D2H = "D2H"

    results: dict[str, float] = defaultdict(float)
    profiling: set[str] = set()

    @classmethod
    def reset(cls) -> None:
        cls.results.clear()
        cls.profiling.clear()

    @classmethod
    @contextmanager
    def profile(cls, profile_type: str) -> Iterator[None]:
        if profile_type in cls.profiling:
            raise AssertionError(
                f"{profile_type} is already being profiled. "
                "SimpleProfiler does not support profiling multiple instances at "
                "the same time. "
            )

        cls.profiling.add(profile_type)
        begin = time.monotonic()
        try:
            yield
        finally:
            end = time.monotonic()
            cls.results[profile_type] += end - begin
            cls.profiling.remove(profile_type)

    @classmethod
    def dump_and_reset(cls, msg: str) -> None:
        # This cannot be combined with DETAIL distributed log
        # as the profiling will be very incorrect.
        if dist.get_rank() == 0 and dist.get_debug_level() == dist.DebugLevel.INFO:
            logger.info("%s %s", msg, cls.results)
        cls.reset()


def _get_sharded_module_tree_with_module_name_to_fqns(
    model: torch.nn.Module,
) -> tuple[str, dict[str, list[str]]]:
    """
    It is used for composable fully_shard() code path, it returns
      1. sharded module tree info: each line represents a submodule name that contains the
    submodule's FQN and its submodule class name, if the submodule is sharded by `fully_shard`,
    the submodule name will add a postfix with ' FULLY SHARDED'. Each increased tree
    level adds 4 spaces before the printed name. A printed sharded module tree info for a toy model
    is like this:
        [CompositeModel] FULLY SHARDED
            l1[Linear]
            u1[UnitModule] FULLY SHARDED
                u1.l1[Linear]
                u1.seq[Sequential]
                    u1.seq.0[ReLU]
                    u1.seq.1[Linear]
                    u1.seq.2[ReLU]
                u1.l2[Linear]
            u2[UnitModule] FULLY SHARDED
                u2.l1[Linear]
                u2.seq[Sequential]
                    u2.seq.0[ReLU]
                    u2.seq.1[Linear]
                    u2.seq.2[ReLU]
                u2.l2[Linear]
            l2[Linear]
      2. a dict mapping from the concated module FQN and class name to a list of its managed
    original parameters' FQNs. An example of the dict for the above toy sharded model is like this:
            {'[CompositeModel]': ['l1.weight', 'l1.bias', 'l2.weight', 'l2.bias'],
             'u1[UnitModule]': ['u1.l1.weight', 'u1.l1.bias', 'u1.seq.1.weight', 'u1.seq.1.bias', 'u1.l2.weight', 'u1.l2.bias'],
             'u2[UnitModule]': ['u2.l1.weight', 'u2.l1.bias', 'u2.seq.1.weight', 'u2.seq.1.bias', 'u2.l2.weight', 'u2.l2.bias']
            }
    All FQNs are prefixed starting from ``model``.

    Args:
        model (torch.nn.Module): Root module (which may or may not be passed to
                                 composable `fully_shard()`).
    """

    def module_fn(
        module, prefix, tree_level, sharded_tree_info, sharded_module_name_to_fqns
    ):
        num_spaces = tree_level * 4
        trimed_prefix = (
            prefix[:-1] if (len(prefix) > 0 and prefix[-1] == ".") else prefix
        )
        prefixed_module_name = trimed_prefix + "[" + module.__class__.__name__ + "]"
        printed_prefixed_module_name = " " * num_spaces + prefixed_module_name

        state = _get_module_fsdp_state(module)
        if state is None:
            sharded_tree_info[0] += printed_prefixed_module_name + "\n"
            return

        handle = state._fully_sharded_module_to_handle.get(module, None)

        if handle:
            sharded_tree_info[0] += (
                printed_prefixed_module_name + " FULLY SHARDED" + "\n"
            )
        else:
            sharded_tree_info[0] += printed_prefixed_module_name + "\n"

        if handle:
            param = handle.flat_param
            if not isinstance(param, flat_param_file.FlatParameter):
                raise AssertionError(f"Expected FlatParameter, got {type(param)}")
            global_fqns = [
                clean_tensor_name(prefix + name) for name in param._fqns
            ]  # prefixed from the top level `model` (i.e. including `prefix`)

            if prefixed_module_name in sharded_module_name_to_fqns:
                sharded_module_name_to_fqns[prefixed_module_name].extend(global_fqns)
            else:
                sharded_module_name_to_fqns[prefixed_module_name] = global_fqns

    def return_fn(sharded_tree_info, sharded_module_name_to_fqns):
        return sharded_tree_info[0], sharded_module_name_to_fqns

    # Use List to mutate its value in place while running the recursive functions
    sharded_tree_info: list[str] = [
        "",
    ]
    sharded_module_name_to_fqns: dict[str, list[str]] = {}
    return _apply_to_modules(
        model,
        module_fn,
        return_fn,
        [key for key, _ in model.named_parameters()],
        sharded_tree_info,
        sharded_module_name_to_fqns,
    )
<|endoftext|># mypy: allow-untyped-defs
import contextlib
import warnings
from collections.abc import Generator
from typing import cast

import torch
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed.fsdp._common_utils import (
    _FSDPState,
    _get_module_fsdp_state,
    _has_fsdp_params,
    _module_handle,
    HandleTrainingState,
    TrainingState,
)
from torch.distributed.fsdp._runtime_utils import (
    _lazy_init,
    _reset_flat_param_grad_info_if_needed,
    _reshard,
    _reshard_grads,
    _unshard,
    _unshard_grads,
)
from torch.distributed.utils import _p_assert

from ._flat_param import FlatParamHandle


FLAT_PARAM = "_flat_param"


@torch.no_grad()
def _writeback_to_local_shard(
    handle: FlatParamHandle,
    writeback_grad: bool,
):
    """
    For the handle, writes back this rank's shard of the unsharded
    flattened parameter to the sharded flattened parameter. If
    ``writeback_grad=True``, then writes back to the sharded gradient as
    well.

    Precondition: The handle's ``FlatParameter`` 's data points to the
    padded unsharded flattened parameter.
    """

    def _get_shard(flat_param_or_grad: torch.Tensor) -> torch.Tensor:
        if handle.uses_sharded_strategy:
            # For sharded strategies, get the *unpadded* shard instead of
            # the *padded* shard to persist user changes to the padding
            # (though FSDP does not explicitly support this)
            shard, _ = FlatParamHandle._get_unpadded_shard(
                flat_param_or_grad,
                handle.rank,
                handle.world_size,
            )
            return shard
        # For `NO_SHARD`, the `flat_param` or its gradient may be modified,
        # so we write it back directly
        return flat_param_or_grad

    param_shard = _get_shard(handle.flat_param)
    handle.flat_param._local_shard[: param_shard.numel()].copy_(param_shard)  # type: ignore[attr-defined]
    if writeback_grad:
        existing_grad = handle.sharded_grad
        if existing_grad is not None:
            if handle.flat_param.grad is None:
                raise AssertionError("Expected handle.flat_param.grad to not be None")
            grad_shard = _get_shard(handle.flat_param.grad)
            existing_grad[: grad_shard.numel()].copy_(grad_shard)


def _deregister_flat_param(state: _FSDPState, module: nn.Module) -> None:
    """
    De-registers the flattened parameter from the wrapped module, hiding it
    from ``nn.Module`` methods.

    We do not use ``del`` because we want ``FLAT_PARAM`` to always be an
    attribute but dynamically change whether it is visible to ``nn.Module``
    methods.
    """
    if _has_fsdp_params(state, module):
        # TODO: figure out the case for the composable APIs.
        cast(nn.Module, module.module)._parameters.pop(FLAT_PARAM, None)


def _register_flat_param(state: _FSDPState, module: nn.Module) -> None:
    """
    Registers the flattened parameter to the wrapped module, making it
    visible to ``nn.Module`` methods.

    We do not use :meth:`nn.Module.register_parameter` because we want
    ``FLAT_PARAM`` to always be an attribute but dynamically change whether
    it is visible to ``nn.Module`` methods.
    """
    handle = _module_handle(state, module)
    if _has_fsdp_params(state, module):
        # TODO: figure out the case for the composable APIs.
        cast(nn.Module, module.module)._parameters[FLAT_PARAM] = handle.flat_param


@contextlib.contextmanager
def _unflatten_as_params(state: _FSDPState, module: nn.Module) -> Generator:
    """
    Assumes that the flattened parameter is unsharded. When in the context,
    de-registers the flattened parameter and unflattens the original
    parameters as ``nn.Parameter`` views into the flattened parameter.
    After the context, re-registers the flattened parameter and restores
    the original parameters as ``Tensor`` views into the flattened
    parameter.
    """
    handle = _module_handle(state, module)
    if not handle:
        yield
    else:
        _deregister_flat_param(state, module)
        try:
            with handle.unflatten_as_params():
                yield
        finally:
            if not handle._use_orig_params:
                _register_flat_param(state, module)


def _validate_unshard_params_args(
    state: _FSDPState,
    writeback: bool,
    rank0_only: bool,
    offload_to_cpu: bool,
    with_grads: bool,
) -> None:
    if with_grads and (offload_to_cpu or not state._use_orig_params):
        raise NotImplementedError(
            f"with_grads={with_grads}, "
            f"use_orig_params={state._use_orig_params}, "
            f"offload_to_cpu={offload_to_cpu} "
            f"is not supported yet"
        )
    if offload_to_cpu and state._handle and (not state._handle.uses_sharded_strategy):
        raise NotImplementedError(
            "offload_to_cpu=True and NO_SHARD is not supported yet"
        )
    if writeback and rank0_only:
        # TODO: Rank 0 can broadcast the `FlatParameter` to allow all ranks to
        # persist the changes.
        raise NotImplementedError(
            "writeback=True and rank0_only=True is not supported yet"
        )
    if offload_to_cpu and not rank0_only:
        warnings.warn(
            "offload_to_cpu=True and rank0_only=False may result in the"
            "unsharded parameters being redundantly copied to CPU memory for "
            "GPUs sharing the same CPU memory, which risks CPU OOM. We "
            "recommend using offload_to_cpu=True with rank0_only=True.",
            stacklevel=2,
        )


@contextlib.contextmanager
def _unshard_fsdp_state_params(
    module: nn.Module,
    state: _FSDPState,
    writeback: bool,
    rank0_only: bool,
    offload_to_cpu: bool,
    with_grads: bool,
):
    """
    This unshards the parameters for a single FSDP state ``state`` that
    corresponds to ``module``.
    """
    _validate_unshard_params_args(
        state, writeback, rank0_only, offload_to_cpu, with_grads
    )
    state._device_handle.synchronize()
    # If handles are shared by other module(s), the handle may be already unsharded.
    maybe_handle = _module_handle(state, module)
    handle = None
    if (
        maybe_handle
        and maybe_handle._training_state != HandleTrainingState.SUMMON_FULL_PARAMS
    ):
        handle = maybe_handle
    if not handle:
        yield
        return

    if handle._training_state != HandleTrainingState.IDLE:
        raise AssertionError(
            f"Expects the handle training to be IDLE but got {handle._training_state}"
        )

    handle._training_state = HandleTrainingState.SUMMON_FULL_PARAMS

    _reset_flat_param_grad_info_if_needed(handle)
    free_unsharded_flat_param = handle.needs_unshard()
    # No need to call `wait_stream()` since we unshard in the computation
    # stream directly
    computation_stream = state._device_handle.current_stream()
    _unshard(state, handle, computation_stream, computation_stream)
    if with_grads:
        _unshard_grads(handle)

    if rank0_only and state.rank != 0:
        # Free the unsharded flattened parameter early
        _reshard(state, handle, free_unsharded_flat_param)
        if with_grads:
            _reshard_grads(handle)
        try:
            yield
        finally:
            handle._training_state = HandleTrainingState.IDLE
    else:
        # Unflatten the unsharded flattened parameters
        with contextlib.ExitStack() as stack:
            # Invariant: rank == 0 or !rank0_only
            if offload_to_cpu and handle.uses_sharded_strategy:
                stack.enter_context(handle.to_cpu())
                # NOTE: Since PyTorch enforces that a parameter and its
                # gradients need to match metadata (e.g. device), we must
                # move gradients to CPU *after* we move parameters.
            # NOTE: This assumes 1 `FlatParameter`
            if not state._use_orig_params:
                stack.enter_context(_unflatten_as_params(state, module))
            try:
                yield
            finally:
                stack.close()
                if writeback:
                    _writeback_to_local_shard(handle, with_grads)
                _reshard(state, handle, free_unsharded_flat_param)
                if with_grads:
                    _reshard_grads(handle)
                handle._training_state = HandleTrainingState.IDLE


@contextlib.contextmanager
def _unshard_params_for_summon(
    module: nn.Module,
    state: _FSDPState,
    writeback: bool,
    rank0_only: bool,
    offload_to_cpu: bool,
    with_grads: bool,
):
    _validate_unshard_params_args(
        state, writeback, rank0_only, offload_to_cpu, with_grads
    )
    _lazy_init(state, module)
    if state.training_state == TrainingState.FORWARD_BACKWARD:
        raise AssertionError(
            "Cannot manually unshard parameters during forward/backward"
        )
    elif state.training_state == TrainingState.SUMMON_FULL_PARAMS:
        raise AssertionError(
            "Cannot manually unshard parameters when already unsharding parameters"
        )
    with _unshard_fsdp_state_params(
        module=module,
        state=state,
        writeback=writeback,
        rank0_only=rank0_only,
        offload_to_cpu=offload_to_cpu,
        with_grads=with_grads,
    ):
        try:
            state.training_state = TrainingState.SUMMON_FULL_PARAMS
            yield
        finally:
            state.training_state = TrainingState.IDLE


@contextlib.contextmanager
def _unshard_params(
    module: nn.Module,
    recurse: bool,
    writeback: bool,
    rank0_only: bool,
    offload_to_cpu: bool,
    with_grads: bool,
):
    """
    This unshards FSDP-managed parameters for all modules with FSDP applied in
    the module tree rooted at ``module``.
    """
    if not recurse:
        optional_state = _get_module_fsdp_state(module)
        if optional_state is None:
            with contextlib.nullcontext():
                yield
            return
        states_and_modules = ([optional_state], [module])
    else:
        states_and_modules = traversal_utils._get_fsdp_states_with_modules(module)
    with contextlib.ExitStack() as stack:
        for state, module in zip(*states_and_modules):
            stack.enter_context(
                _unshard_params_for_summon(
                    module=module,
                    state=state,
                    writeback=writeback,
                    rank0_only=rank0_only,
                    offload_to_cpu=offload_to_cpu,
                    with_grads=with_grads,
                )
            )
        yield


def _deregister_orig_params(state: _FSDPState, module: nn.Module) -> None:
    """
    Deregisters the original parameters; registers the ``FlatParameter``.
    """
    handle = _module_handle(state, module)
    if not handle:
        return
    _p_assert(
        handle._use_orig_params,
        f"Inconsistent `_use_orig_params` -- FSDP: {state._use_orig_params} "
        f"handle: {handle._use_orig_params}",
    )
    handle._deregister_orig_params()
    _register_flat_param(state, module)


def _register_orig_params(state: _FSDPState, module: nn.Module) -> None:
    """
    Deregisters the ``FlatParameter``; registers the original parameters.
    """
    handle = _module_handle(state, module)
    if not handle:
        return
    _deregister_flat_param(state, module)
    if handle.is_sharded(handle.flat_param):
        handle._use_sharded_views()
        handle._use_sharded_grad_views()
    else:
        handle._use_unsharded_views(as_params=True)
<|endoftext|># mypy: allow-untyped-defs
import functools
import logging
from collections.abc import Callable
from enum import auto, Enum
from typing import Any, no_type_check

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd.graph import register_multi_grad_hook
from torch.distributed.algorithms._comm_hooks import LOW_PRECISION_HOOKS
from torch.distributed.fsdp._common_utils import (
    _assert_in_training_states,
    _FSDPState,
    _get_module_fsdp_state,
    _is_composable,
    _log_post_backward_hook,
    _no_dispatch_record_stream,
    clean_tensor_name,
    TrainingState,
)
from torch.distributed.fsdp._flat_param import (
    FlatParameter,
    FlatParamHandle,
    HandleShardingStrategy,
    HandleTrainingState,
    RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES,
)
from torch.distributed.fsdp._init_utils import HYBRID_SHARDING_STRATEGIES
from torch.distributed.fsdp.api import BackwardPrefetch
from torch.distributed.utils import (
    _apply_to_tensors,
    _cast_forward_inputs,
    _p_assert,
    _to_kwargs,
)
from torch.utils import _pytree as pytree


logger = logging.getLogger(__name__)

# Do not include "process_group" to enable hybrid shard and MoE cases
HOMOGENEOUS_ATTR_NAMES = (
    "_use_orig_params",
    "limit_all_gathers",
    "_use_full_prec_in_eval",
)


class _PrefetchMode(Enum):
    BACKWARD = auto()
    FORWARD = auto()


def _get_fsdp_root_states_with_modules(
    module: nn.Module,
) -> tuple[list[_FSDPState], list[nn.Module]]:
    """
    Returns a tuple containing:
    1. A list of the root ``_FSDPState`` instances in the module tree rooted at
    ``module`` without any duplicates and following the ``module.modules()``
    traversal order (which is assumed to be depth-first).
    2. A corresponding list of the root modules owning the states in the first
    list.

    This is similar to :func:`_get_fsdp_states_with_modules` except that we
    must call :func:`_is_fsdp_root` to force a lazy initialization to determine
    the FSDP root in case lazy initialization has not yet happened.
    """
    fsdp_root_states: list[_FSDPState] = []
    fsdp_root_modules: list[nn.Module] = []
    visited_fsdp_states: set[_FSDPState] = set()
    # NOTE: This function assumes that `module.modules()` proceeds top-down.
    for submodule in module.modules():
        optional_state = _get_module_fsdp_state(submodule)
        if (
            optional_state is not None
            and optional_state not in visited_fsdp_states
            and _is_fsdp_root(optional_state, submodule)
        ):
            visited_fsdp_states.add(optional_state)
            fsdp_root_states.append(optional_state)
            fsdp_root_modules.append(submodule)
    return fsdp_root_states, fsdp_root_modules


def _get_fsdp_root_states(module: nn.Module) -> list[_FSDPState]:
    """See :func:`_get_fsdp_root_states_with_modules`."""
    fsdp_root_states, _ = _get_fsdp_root_states_with_modules(module)
    return fsdp_root_states


def _is_fsdp_root(state: _FSDPState, module: nn.Module) -> bool:
    """
    Returns if ``state`` corresponds to that of an FSDP root.

    For the wrapper code path, ``state`` and ``module`` should be the same. For
    the non-wrapper code path, ``state`` should be ``module`` 's state.
    """
    # Force a lazy initialization to determine the FSDP root
    _lazy_init(state, module)
    if state._is_root is None:
        raise AssertionError("Expected _is_root to be set after lazy init")
    return state._is_root


@no_type_check
def _lazy_init(
    state: _FSDPState,
    root_module: nn.Module,
) -> _FSDPState:
    """
    Performs initialization lazily, typically right before the first forward
    pass. The laziness is needed to ensure that the parameter device/dtype and
    the FSDP hierarchy have finalized. This method's actual logic only runs on
    the root FSDP instance, which performs initialization for all non-root FSDP
    instances to avoid partial initialization.

    For the non-composable code path, ``state`` and ``root_module`` should be
    the same, namely the FSDP instance itself.
    """
    if state._is_root is not None:
        return  # no-op: already lazily initialized
    if not state._device_handle.is_available():
        # Allow the FSDP constructor to run even without CUDA but check this
        # once we start real execution
        raise RuntimeError("FSDP does not support CPU only execution")
    # The following logic is only run on the root FSDP instance since it will
    # set `_is_root=False` for the non-root instances
    state._is_root = True
    _assert_in_training_states(state, [TrainingState.IDLE])
    _check_flat_params_on_expected_device(state, root_module)
    state._all_fsdp_states = traversal_utils._get_fsdp_states(root_module)
    _init_streams(state)
    buffers, buffer_dtypes = _get_buffers_and_dtypes_for_computation(state, root_module)
    _cast_buffers_to_dtype_and_device(buffers, buffer_dtypes, state.compute_device)
    state._exec_order_data.init(state, root_module, state.process_group)
    _share_state_and_init_handle_attrs(state, root_module)
    return state


def _check_flat_params_on_expected_device(state: _FSDPState, module: nn.Module):
    """
    Checks that all ``FlatParameter``s in ``module`` 's tree managed by
    ``state`` are on the expected device for *lazy initialization*.
    """
    cpu_device = torch.device("cpu")
    for handle in traversal_utils._get_fsdp_handles(module):
        if (
            not handle._offload_params
            and handle.flat_param.device != state.compute_device
        ):
            raise RuntimeError(
                "An FSDP-managed module unexpectedly has parameters on "
                f"{handle.flat_param.device}. Make sure to move the module to "
                f"{state.compute_device} before training."
            )
        elif handle._offload_params and handle.flat_param.device != cpu_device:
            raise RuntimeError(
                "An FSDP-managed module with parameter CPU offloading enabled "
                f"has parameters on {handle.flat_param.device}. Make sure to "
                f"not move the module from CPU when offloading parameters."
            )


@no_type_check
def _share_state_and_init_handle_attrs(
    root_state: _FSDPState,
    root_module: nn.Module,
) -> None:
    """
    Shares data structure state from the ``root_state`` to all FSDP states in
    ``root_module`` 's module tree, and initializes handle attributes. These
    are done together to require a single loop over the states.
    """
    handle = root_state._handle
    if handle:
        handle.init_flat_param_attributes()
    attr_name_to_values: dict[str, set[Any]] = {}
    for attr_name in HOMOGENEOUS_ATTR_NAMES:
        attr_name_to_values[attr_name] = set()
    root_state._all_handles = root_state._exec_order_data.all_handles  # share reference
    # Update _has_optim_in_backward for each handle.
    for handle in root_state._all_handles:
        flat_param = handle.flat_param
        if hasattr(flat_param, "_in_backward_optimizers"):
            raise RuntimeError(
                "FSDP optimizer in backward only supported with use_orig_params=True!"
            )
        handle._has_optim_in_backward = flat_param._params is not None and any(
            hasattr(param, "_in_backward_optimizers") for param in flat_param._params
        )
        if handle._has_optim_in_backward:
            torch._C._log_api_usage_once("fsdp.optimizer_in_backward")
    for fsdp_state in root_state._all_fsdp_states:
        for attr_name in HOMOGENEOUS_ATTR_NAMES:
            _p_assert(
                hasattr(fsdp_state, attr_name),
                f"FSDP state missing attribute {attr_name}",
            )
            attr_name_to_values[attr_name].add(getattr(fsdp_state, attr_name))
        if fsdp_state is root_state:
            continue
        # Relax the assert for non-root FSDP instances in case the nested
        # initialized module is wrapped again in FSDP later (e.g. after
        # training to run inference)
        _p_assert(
            fsdp_state._is_root is None or not fsdp_state._is_root,
            "Non-root FSDP instance's `_is_root` should not have been "
            "set yet or should have been set to `False`",
        )
        fsdp_state._is_root = False
        fsdp_state._unshard_stream = root_state._unshard_stream
        fsdp_state._post_backward_stream = root_state._post_backward_stream
        fsdp_state._pre_unshard_stream = root_state._pre_unshard_stream
        fsdp_state._all_reduce_stream = root_state._all_reduce_stream
        fsdp_state._default_stream = root_state._default_stream
        fsdp_state._exec_order_data = root_state._exec_order_data
        fsdp_state._free_event_queue = root_state._free_event_queue
        if fsdp_state._fsdp_extension is not None:
            fsdp_state._fsdp_extension.compute_stream = root_state._default_stream
        handle = fsdp_state._handle
        if handle:
            handle.init_flat_param_attributes()
    for attr_name, attr_values in attr_name_to_values.items():
        if len(attr_values) != 1:
            raise ValueError(
                f"Expects one homogeneous value for {attr_name} but got {attr_values}"
            )


@no_type_check
def _init_streams(
    state: _FSDPState,
) -> None:
    """
    Initializes CUDA streams for overlapping communication, computation, and
    data transfers. The streams should be shared across FSDP instances.
    """
    if not state._is_root:
        raise AssertionError("Expected state to be root")
    if not state._device_handle.is_available():
        raise AssertionError("Expected device handle to be available")
    uses_hybrid_sharding = any(
        fsdp_state.sharding_strategy in HYBRID_SHARDING_STRATEGIES
        for fsdp_state in state._all_fsdp_states
    )
    # Prioritize all-gathers/reduce-scatters over async all-reduce for HSDP and
    # preserve the default priority of 0 otherwise
    high_priority = -1 if state.limit_all_gathers and uses_hybrid_sharding else 0
    # Default stream for computation
    state._default_stream = state._device_handle.current_stream()
    if state._fsdp_extension is not None:
        # set the compute stream to the FSDP extension
        state._fsdp_extension.compute_stream = state._default_stream

    # Stream for unshard logic, including allocating the all-gather destination
    # tensors and the all-gathers themselves
    state._unshard_stream = state._device_handle.Stream(priority=high_priority)
    # Stream for overlapping gradient reduction with the backward pass gradient
    # computation
    state._post_backward_stream = state._device_handle.Stream(priority=high_priority)
    # Stream for pre-unshard logic, namely allocations and writes for CPU
    # offloading (H2D copy) and mixed precision (low precision cast)
    state._pre_unshard_stream = state._device_handle.Stream(priority=high_priority)
    # Stream to run HSDP's all-reduce as async (if using HSDP)
    state._all_reduce_stream = (
        state._device_handle.Stream() if uses_hybrid_sharding else state._default_stream
    )


@no_type_check
def _unshard(
    state: _FSDPState,
    handle: FlatParamHandle,
    unshard_stream: torch.Stream,
    pre_unshard_stream: torch.Stream,
) -> None:
    """
    Unshards the handles in ``handles``. If the handles are in
    :meth:`summon_full_params` and are using mixed precision, then they are
    forced to full precision.

    Postcondition: handle's ``FlatParameter`` 's data is the padded
    unsharded flat parameter on the compute device.
    """
    if not handle:
        return
    with state._device_handle.stream(pre_unshard_stream):
        ran_pre_unshard = handle.pre_unshard()
    if ran_pre_unshard:
        unshard_stream.wait_stream(pre_unshard_stream)
    if state.limit_all_gathers:
        event = state._free_event_queue.dequeue_if_needed()
        if event:
            with torch.profiler.record_function(
                "FullyShardedDataParallel.rate_limiter"
            ):
                event.synchronize()
    with state._device_handle.stream(unshard_stream):
        handle.unshard()
        handle.post_unshard()


@no_type_check
def _reshard(
    state: _FSDPState,
    handle: FlatParamHandle,
    free_unsharded_flat_param: bool,
):
    """
    Reshards the handle. ``free_unsharded_flat_param`` indicates whether to
    free the handle's padded unsharded flat parameter.
    """
    handle.reshard(free_unsharded_flat_param)
    if state.limit_all_gathers and free_unsharded_flat_param:
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            # We don't run an event queue for freeing under torch compile atm
            # But maybe we need to? TODO(voz): Look into this
            free_event = state._device_handle.Event()
            free_event.record()
            state._free_event_queue.enqueue(free_event)
    handle.post_reshard()
    # Flat parameter freed or not, we always have to "unshard" the parameter
    # upon next access to get its shape correct.
    handle._prefetched = False


def _unshard_grads(
    handle: FlatParamHandle | None,
) -> None:
    if handle:
        handle.unshard_grad()


def _reshard_grads(
    handle: FlatParamHandle | None,
) -> None:
    if handle:
        handle.reshard_grad()


@no_type_check
def _pre_forward(
    state: _FSDPState,
    handle: FlatParamHandle | None,
    unshard_fn: Callable,
    module: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """
    Runs the pre-forward logic. This includes an opportunity to unshard
    currently sharded parameters such as those for the current forward and
    registering post-backward hooks for these current parameters. This function
    also converts forward ``args`` and ``kwargs`` to the given precision.

    Args:
        handles (List[FlatParamHandle]): Handles giving the parameters used in
            the current forward.
        unshard_fn (Optional[Callable]): A callable to unshard any currently
            sharded parameters or ``None`` to not do any unsharding.
        module (nn.Module): Module whose forward this method runs right before;
            expected by the hook signature.
        args (Tuple[Any, ...]): Module forward ``args``.
        kwargs (Dict[str, Any]): Module forward ``kwargs``.
    """
    with torch.profiler.record_function("FullyShardedDataParallel._pre_forward"):
        # For `fully_shard` + `checkpoint`, skip pre-forward logic in the
        # recomputed forward
        if handle and handle._training_state == HandleTrainingState.BACKWARD_PRE:
            # For both checkpoint implementations, we do not need to re-cast
            # inputs here since they will be checkpointed in the low precision
            # either by AC or normally by autograd as long as the AC region is
            # nested within FSDP
            return args, kwargs
        state.training_state = TrainingState.FORWARD_BACKWARD
        state._exec_order_data.record_pre_forward(handle, module.training)
        if handle:
            handle._training_state = HandleTrainingState.FORWARD
        if unshard_fn is not None:
            unshard_fn(state, handle)
        # Register post-backward hooks to reshard the parameters and reduce-scatter
        # their gradients. They must be re-registered every forward pass in case
        # the `grad_fn` is mutated.
        _register_post_backward_hook(state, handle)
        # We have to reallocate the _cpu_grad if optimizer overlap
        # set the grad to None in the backward pass.
        if handle and handle._offload_params and handle.flat_param._cpu_grad is None:
            handle.flat_param._cpu_grad = torch.zeros_like(
                handle.flat_param._local_shard, device=torch.device("cpu")
            ).pin_memory()

        should_cast_forward_inputs = (
            state._handle and not state._handle._force_full_precision
        )

        if should_cast_forward_inputs and state.mixed_precision.cast_forward_inputs:
            # Recursively convert args and kwargs to specified precision.
            input_dtype: torch.dtype | None = state.mixed_precision.param_dtype
            args, kwargs = _cast_forward_inputs(input_dtype, *args, **kwargs)
        _register_post_backward_reshard_only_hook(state, handle, args, kwargs)
        return args, kwargs


@no_type_check
def _pre_forward_unshard(
    state: _FSDPState,
    handle: FlatParamHandle | None,
) -> None:
    """Unshards parameters in the pre-forward."""
    if not handle:
        return
    # If the handles have been prefetched, then there is no need to call
    # `_unshard()` again
    if not handle._prefetched:
        _unshard(state, handle, state._unshard_stream, state._pre_unshard_stream)
    handle._needs_pre_forward_unshard = False
    # Don't wait during trace
    if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
        current_stream = state._device_handle.current_stream()
        if state._unshard_event is not None:
            current_stream.wait_event(state._unshard_event)
            state._unshard_event = None
        else:
            current_stream.wait_stream(state._unshard_stream)
    with torch.profiler.record_function(
        "FullyShardedDataParallel._pre_forward_prefetch"
    ):
        _prefetch_handle(state, handle, _PrefetchMode.FORWARD)


@no_type_check
def _post_forward(
    state: _FSDPState,
    handle: FlatParamHandle | None,
    reshard_fn: Callable,
    module: nn.Module,
    input: Any,
    output: Any,
) -> Any:
    """
    Runs the post-forward logic. This includes an opportunity to reshard
    currently unsharded parameters such as those used in the current forward
    and registering pre-backward hooks on the forward outputs.

    Args:
        handles (List[FlatParamHandle]): Handles giving the parameters used in
            the current forward.
        reshard_fn (Optional[Callable]): A callable to reshard any currently
            unsharded parameters (e.g. from the current forward) or ``None`` to
            not do any resharding.
        module (nn.Module): Module whose forward just ran, which should be a
            fully sharded module (see [Note: Fully Sharded Module]); expected
            by the hook signature.
        input (Any): Unused; expected by the hook signature.
        output (Any): Forward pass output; pre-backward hooks are registered on
            the tensors that require gradients in this output.

    Postcondition: Each ``FlatParameter`` 's data points to the sharded flat
    parameter.
    """
    with torch.profiler.record_function("FullyShardedDataParallel._post_forward"):
        # For `fully_shard` + `checkpoint`, skip post-forward logic in the
        # recomputed forward
        if handle and handle._training_state == HandleTrainingState.BACKWARD_PRE:
            return output

        state._exec_order_data.record_post_forward(handle)
        if reshard_fn is not None:
            reshard_fn(state, handle)
        # Register pre-backward hooks to unshard the flat parameters for the
        # gradient computation (if needed)
        output = _register_pre_backward_hooks(state, module, output, handle)
        state.training_state = TrainingState.IDLE
        if handle:
            handle._training_state = HandleTrainingState.IDLE
        return output


@no_type_check
def _post_forward_reshard(
    state: _FSDPState,
    handle: FlatParamHandle,
) -> None:
    """Reshards parameters in the post-forward."""
    if not handle:
        return
    # Do not free the root's parameters in the post-forward for `FULL_SHARD`
    # with the intention that they are immediately used for backward
    # computation (though this may not be true)
    free_unsharded_flat_param = (
        not state._is_root
        and handle._sharding_strategy in RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES
    )
    _reshard(state, handle, free_unsharded_flat_param)


@no_type_check
def _root_pre_forward(
    state: _FSDPState,
    module: nn.Module,
    args,
    kwargs,
) -> None:
    """
    Runs pre-forward logic specific to the root FSDP instance, which should run
    before any individual module's pre-forward. This starts with an attempt at
    lazy initialization (which only runs non-vacuously once). Otherwise, if
    this is called on a non-root FSDP instance, then it returns directly.

    Args:
        module (nn.Module): Module for which this logic tries to run. It may or
            may not be the root. If not, then this method does not do anything.
    """
    with torch.profiler.record_function("FullyShardedDataParallel._root_pre_forward"):
        _lazy_init(state, module)
        _p_assert(state._is_root is not None, "Expects a root FSDP to have been set")
        if not state._is_root:
            # Always cast forward inputs in the root of this local FSDP unit for mixed
            # precision, as this is where mixed precision could be configured.
            # This is more useful for auto wrapping that is recommended in composable path.
            # For manual wrapping, cast forward inputs on each local FSDP unit root will
            # increase some overhead, so not turned on for model wrapper path right now where
            # manual wrapping is more broadly used.
            if _is_composable(state):
                return _root_cast_forward_input(state, module, args, kwargs)
            return args, kwargs

        # We cast buffers back to full precision if we're forcing full precision. Disjointly, we check if buffers
        # are in full precision and if we should cast them back to lower precision, which happens when
        # exiting eval() mode.
        handle = state._handle
        if handle:
            should_cast_buffers_to_full_prec = handle._force_full_precision
        else:
            # If the root has no handle (no managed parameters), then we fall
            # back to checking if any child wants to force full precision as a
            # workaround
            handles = traversal_utils._get_fsdp_handles(module)
            should_cast_buffers_to_full_prec = any(
                handle._force_full_precision for handle in handles
            )

        if should_cast_buffers_to_full_prec:
            _cast_buffers_to_dtype_and_device(
                buffers=dict(module.named_buffers()).values(),
                buffer_dtypes=list(state._buffer_name_to_orig_dtype.values()),
                device=state.compute_device,
            )
            # This flag is only set when we cast buffers to full precision, to avoid the
            # CPU overhead that can stem from retrieving all buffers and their types in the
            # following else branch.
            state._needs_buffer_dtype_restore_check = True
        elif getattr(state, "_needs_buffer_dtype_restore_check", False):
            # Check if buffers are in full precision and we need to cast them
            # back down.
            (
                buffers,
                buffer_dtypes_for_computation,
            ) = _get_buffers_and_dtypes_for_computation(state, module)
            if len(buffers) > 0 and len(buffer_dtypes_for_computation) > 0:
                if any(
                    buffer.dtype != buffer_dtype_for_computation
                    for buffer, buffer_dtype_for_computation in zip(
                        buffers, buffer_dtypes_for_computation
                    )
                ):
                    # Assume we have to cast everything if there is one mismatch
                    _cast_buffers_to_dtype_and_device(
                        buffers, buffer_dtypes_for_computation, state.compute_device
                    )
            # We don't have to check this again until we cast buffers to full precision again.
            state._needs_buffer_dtype_restore_check = False

        if state.forward_prefetch:
            handles = [
                fsdp_state._handle
                for fsdp_state in state._all_fsdp_states
                if fsdp_state._handle
            ]
            for handle in handles:
                handle._needs_pre_forward_unshard = True
                handle._prefetched = False
        _wait_for_computation_stream(
            state._device_handle.current_stream(),
            state._unshard_stream,
            state._pre_unshard_stream,
        )
        _reset_flat_param_grad_info_if_needed(state._all_handles)

        # Prepares the forward inputs by moving them to ``compute_device``
        # TODO: Do not use the side stream for tensor copies for now; investigate
        # the perf with/without it.
        with torch.profiler.record_function("FullyShardedDataParallel._to_kwargs"):
            args_tuple, kwargs_tuple = _to_kwargs(
                args, kwargs, state.compute_device, False
            )
        args = args_tuple[0] if args_tuple else ()
        kwargs = kwargs_tuple[0] if kwargs_tuple else {}

        return _root_cast_forward_input(state, module, args, kwargs)


@no_type_check
def _root_cast_forward_input(
    state: _FSDPState, module: torch.nn.Module, args, kwargs
) -> tuple[Any, Any]:
    if state._handle:
        force_full_precision = not state._handle._force_full_precision
    else:
        force_full_precision = True

    should_cast_forward_inputs = (
        (module.training or not state._use_full_prec_in_eval) and force_full_precision
    ) and state.mixed_precision.cast_root_forward_inputs

    if should_cast_forward_inputs:
        input_dtype: torch.dtype | None = state.mixed_precision.param_dtype
        args, kwargs = _cast_forward_inputs(input_dtype, *args, **kwargs)

    return args, kwargs


@no_type_check
def _pre_backward_hook(
    state: _FSDPState,
    module: nn.Module,
    handle: FlatParamHandle,
    grad,
    *unused: Any,
) -> Any:
    """
    Prepares ``_handle`` 's ``FlatParameter`` s for gradient computation.

    Args:
        module (nn.Module): Fully sharded module (see [Note: Fully Sharded
            Module]).
    """
    # Only run the pre-backward hook once per group of handles involved in the
    # same module forward computation
    if (
        handle
        and hasattr(handle, "_ran_pre_backward_hook")
        and handle._ran_pre_backward_hook
    ):
        return grad

    with torch.profiler.record_function("FullyShardedDataParallel._pre_backward_hook"):
        # Queue the post-backward callback once for the root FSDP instance to
        # attach it to the outermost backward graph task so that it is called
        # after all backward calls complete
        if state._is_root and not state._post_backward_callback_queued:
            _register_post_backward_final_callback(state, module)
            _reset_flat_param_grad_info_if_needed(state._all_handles)
        elif handle:
            allowed_states = [TrainingState.IDLE]
            if _is_composable(state):
                allowed_states.append(TrainingState.FORWARD_BACKWARD)
            _assert_in_training_states(state, allowed_states)
        state.training_state = TrainingState.FORWARD_BACKWARD
        # Queueing the post-backward callback is the only logic that is not
        # per-handle in the pre-backward hook, so we can return early here if
        # there are no handles.
        if not handle:
            return grad
        handle._training_state = HandleTrainingState.BACKWARD_PRE

        if handle._needs_pre_backward_unshard:
            # If the handles have been prefetched, then there is no need to
            # call `_unshard()` again
            if not handle._prefetched:
                _unshard(
                    state,
                    handle,
                    state._unshard_stream,
                    state._pre_unshard_stream,
                )
            # Don't wait during trace
            if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
                state._device_handle.current_stream().wait_stream(state._unshard_stream)

        # Set this to `False` to ensure that a mistargeted prefetch does not
        # actually unshard these handles
        handle._needs_pre_backward_unshard = False
        with torch.profiler.record_function(
            "FullyShardedDataParallel._pre_backward_prefetch"
        ):
            _prefetch_handle(state, handle, _PrefetchMode.BACKWARD)
        handle.prepare_gradient_for_backward()
        handle._ran_pre_backward_hook = True
        return grad


@no_type_check
@torch.no_grad()
def _post_backward_hook(
    state: _FSDPState,
    handle: FlatParamHandle,
    flat_param,
    *unused: Any,
):
    """
    Reduce-scatters the gradient of ``handle`` 's ``FlatParameter``.

    Precondition: The ``FlatParameter`` 's ``.grad`` attribute contains the
    unsharded gradient for the local batch.

    Postcondition:
    - If using ``NO_SHARD``, then the ``.grad`` attribute is the reduced
    unsharded gradient.
    - Otherwise, the ``_saved_grad_shard`` attribute is the reduced sharded
    gradient (accumulating with any existing gradient).
    """
    _log_post_backward_hook(state, handle, logger)
    flat_param = handle.flat_param
    flat_param._post_backward_called = True
    with torch.autograd.profiler.record_function(
        "FullyShardedDataParallel._post_backward_hook"
    ):
        _assert_in_training_states(state, [TrainingState.FORWARD_BACKWARD])
        # For multiple applications of reentrant AC across submodules sharing
        # the same `FlatParameter`, the post-backward hook may run multiple
        # times in one backward, in which case we permit the state to already
        # be in `BACKWARD_POST`.
        _p_assert(
            handle._training_state
            in (HandleTrainingState.BACKWARD_PRE, HandleTrainingState.BACKWARD_POST),
            f"Expects `BACKWARD_PRE` or `BACKWARD_POST` state but got {handle._training_state}",
        )
        handle._training_state = HandleTrainingState.BACKWARD_POST

        if flat_param.grad is None:
            return
        if flat_param.grad.requires_grad:
            raise RuntimeError("FSDP does not support gradients of gradients")

        _post_backward_reshard(state, handle)
        if not state._sync_gradients:
            if handle._use_orig_params:
                handle._use_unsharded_grad_views()
            return

        # Wait for all ops in the current stream (e.g. gradient computation) to
        # finish before reduce-scattering the gradient
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            state._post_backward_stream.wait_stream(
                state._device_handle.current_stream()
            )

        with state._device_handle.stream(state._post_backward_stream):
            autograd_computed_grad = flat_param.grad.data
            if (
                not _low_precision_hook_enabled(state)
                and flat_param.grad.dtype != handle._reduce_dtype
                # If we are forcing full precision but communicating grads
                # (i.e. model.eval() + full precision in eval was configured), don't downcast gradient.
                and not handle._force_full_precision
            ):
                flat_param.grad.data = flat_param.grad.to(handle._reduce_dtype)
            if handle.uses_sharded_strategy:
                _reduce_grad(state, handle)
            else:
                _reduce_grad_no_shard(state, handle)
            # Since the unsharded gradient is produced in the computation
            # stream and consumed in the post-backward stream, inform the
            # caching allocator (before it goes out of scope)
            _no_dispatch_record_stream(
                autograd_computed_grad, state._post_backward_stream
            )


def _post_backward_reshard_only_hook(
    state: _FSDPState,
    handle: FlatParamHandle,
    *unused: Any,
) -> None:
    with torch.profiler.record_function(
        "FullyShardedDataParallel._post_backward_hook_reshard_only"
    ):
        # `_pre_backward_hook` may not get executed
        # if forward output does not require grad
        # overwrite IDLE state for post-backward prefetching
        state.training_state = TrainingState.FORWARD_BACKWARD
        handle._training_state = HandleTrainingState.BACKWARD_POST
        _post_backward_reshard(state, handle)


def _post_backward_reshard(
    state: _FSDPState,
    handle: FlatParamHandle,
    *unused: Any,
) -> None:
    free_unsharded_flat_param = _should_free_in_backward(state, handle)
    _reshard(state, handle, free_unsharded_flat_param)

    # TODO: Post-backward prefetching does not support the multiple handles
    # per module case since the post-backward hook runs per handle, not per
    # group of handles.
    with torch.profiler.record_function(
        "FullyShardedDataParallel._post_backward_prefetch"
    ):
        _prefetch_handle(state, handle, _PrefetchMode.BACKWARD)


@no_type_check
def _should_free_in_backward(
    state: _FSDPState,
    handle: FlatParamHandle,
) -> bool:
    """
    Returns whether FSDP should free the unsharded flat parameter in the
    post-backward or not.
    """
    if not handle.uses_sharded_strategy:
        return False
    # If not syncing gradients, then we do not free for strategies that do not
    # reshard after forward as a *heuristic* to tradeoff higher memory for
    # higher throughput.
    return (
        state._sync_gradients
        or handle._sharding_strategy in RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES
    )


@no_type_check
def _reduce_grad(state: _FSDPState, handle: FlatParamHandle) -> None:
    """
    For sharded strategies, this runs gradient reduction, sharded gradient
    accumulation if needed, and the post-reduction callback.
    """
    flat_param = handle.flat_param
    uses_hybrid_sharded_strategy = handle._sharding_strategy in (
        HandleShardingStrategy.HYBRID_SHARD,
        HandleShardingStrategy._HYBRID_SHARD_ZERO2,
    )
    # We clear `.grad` to permit multiple backwards. This avoids a race where
    # the second backward pass computation precedes ahead of the first backward
    # pass reduction, which is possible since the reduction is issued in a
    # separate stream and is async and would result in reducing the wrong
    # gradient.
    unsharded_grad = flat_param.grad.data
    flat_param.grad = None
    padded_unsharded_grad, new_sharded_grad = _get_reduce_scatter_tensors(
        state, unsharded_grad
    )
    if state._comm_hook is None:  # default path
        _div_if_needed(padded_unsharded_grad, state._gradient_predivide_factor)
        pg = (
            handle._fake_process_group
            if handle._use_fake_reduce
            else state.process_group
        )
        dist.reduce_scatter_single(
            new_sharded_grad,
            padded_unsharded_grad,
            group=pg,
        )
        if uses_hybrid_sharded_strategy:
            # Don't wait during trace
            if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
                state._all_reduce_stream.wait_stream(state._post_backward_stream)
            with state._device_handle.stream(state._all_reduce_stream):
                # Since the new sharded gradient is produced in the post-
                # backward stream and consumed in the all-reduce stream,
                # inform the caching allocator
                _no_dispatch_record_stream(new_sharded_grad, state._all_reduce_stream)
                dist.all_reduce(new_sharded_grad, group=state._inter_node_pg)
                _div_if_needed(new_sharded_grad, state._gradient_postdivide_factor)
                grad_to_offload = _accumulate_sharded_grad(
                    state, handle, new_sharded_grad
                )
                _post_reduce_grad_callback(state, handle, grad_to_offload)
                return
        _div_if_needed(new_sharded_grad, state._gradient_postdivide_factor)
    else:
        state._comm_hook(
            state._comm_hook_state, padded_unsharded_grad, new_sharded_grad
        )
        # NOTE: HSDP variants do not support communication hook.
    grad_to_offload = _accumulate_sharded_grad(state, handle, new_sharded_grad)
    _post_reduce_grad_callback(state, handle, grad_to_offload)


@no_type_check
def _get_reduce_scatter_tensors(
    state: _FSDPState, unsharded_grad: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the input and output tensors to reduce-scatter, respectively.
    """
    chunks = list(unsharded_grad.chunk(state.world_size))
    numel_to_pad = state.world_size * chunks[0].numel() - unsharded_grad.numel()
    padded_unsharded_grad = (
        F.pad(unsharded_grad, [0, numel_to_pad]) if numel_to_pad > 0 else unsharded_grad
    )
    new_sharded_grad = torch.empty_like(chunks[0])  # padded
    return padded_unsharded_grad, new_sharded_grad


@no_type_check
def _accumulate_sharded_grad(
    state: _FSDPState,
    handle: FlatParamHandle,
    sharded_grad: torch.Tensor,
) -> torch.Tensor:
    """
    Accumulates the reduce-scattered sharded gradient with any existing sharded
    gradient if needed, returning the gradient to offload (if CPU offloading is
    enabled).
    """
    flat_param = handle.flat_param
    _cast_grad_to_param_dtype(state, sharded_grad, flat_param)
    # Save the sharded gradient in `_saved_grad_shard` to support gradient
    # accumulation -- for multiple backwards, the gradient reductions may
    # happen in arbitrary order
    accumulate_grad = hasattr(flat_param, "_saved_grad_shard")
    if accumulate_grad:
        _check_grad_to_accumulate(sharded_grad, flat_param._saved_grad_shard)
        flat_param._saved_grad_shard += sharded_grad
    else:
        flat_param._saved_grad_shard = sharded_grad
    grad_to_offload = flat_param._saved_grad_shard
    return grad_to_offload


@no_type_check
def _reduce_grad_no_shard(state: _FSDPState, handle: FlatParamHandle) -> None:
    """
    For no-shard, this runs gradient reduction (which directly covers any
    gradient accumulation implicitly) and the post-reduction callback.
    """
    flat_param = handle.flat_param
    if state._comm_hook is None:  # default path
        _div_if_needed(flat_param.grad, state._gradient_predivide_factor)
        dist.all_reduce(flat_param.grad, group=state.process_group)
        _div_if_needed(flat_param.grad, state._gradient_postdivide_factor)
    else:
        state._comm_hook(state._comm_hook_state, flat_param.grad)
    # For `NO_SHARD`, we can keep the low precision gradients by simply
    # omitting the cast altogether
    if not handle._keep_low_precision_grads:
        _cast_grad_to_param_dtype(state, flat_param.grad, flat_param)
    grad_to_offload = flat_param.grad.data
    _post_reduce_grad_callback(state, handle, grad_to_offload)


@no_type_check
def _post_reduce_grad_callback(
    state: _FSDPState,
    handle: FlatParamHandle,
    # Additional arguments needed for the callback logic
    grad_to_offload: torch.Tensor,
):
    """
    This callback captures any logic to run after the gradient reduction
    finishes. Currently, this offloads the gradient to CPU if CPU offloading is
    enabled and uses sharded gradient views if ``use_orig_params=True``.
    """
    _offload_grad(state, handle, grad_to_offload)
    _post_backward_use_sharded_grad_views(handle)


@no_type_check
def _offload_grad(
    state: _FSDPState,
    handle: FlatParamHandle,
    grad_to_offload: torch.Tensor,
):
    if not handle._offload_params:
        return
    # Offload the gradient to CPU to ensure parameters and gradients are on the
    # same device as required by the optimizer
    # TODO: Investigate why `NO_SHARD` breaks correctness when using
    # `non_blocking=True` here.
    # TODO (rohan-varma): When CPU offload and optimizer overlap,
    # non_blocking=True won't work since the copy may have not finished before
    # the optimizer step executes on CPU. If we want to use non-blocking=True
    # here, we'll have to synchronize before using result on CPU.
    non_blocking = handle.uses_sharded_strategy and not handle._has_optim_in_backward
    handle.flat_param._cpu_grad.copy_(
        grad_to_offload.detach(), non_blocking=non_blocking
    )  # synchronized in the post-backward callback
    # Since the gradient being offloaded may have been produced in the
    # computation stream and is being consumed here in the post-backward
    # stream, inform the caching allocator
    _no_dispatch_record_stream(grad_to_offload.data, state._post_backward_stream)


@no_type_check
def _post_backward_use_sharded_grad_views(handle: FlatParamHandle):
    if not handle._use_orig_params:
        return
    # Since the handle's `FlatParameter` completed its gradient computation, we
    # should reset the gradient noneness mask
    handle._reset_is_grad_none()
    # Delay using sharded gradient views until after the reduce-scatter instead
    # of immediately after resharding
    handle._use_sharded_grad_views()
    if handle._has_optim_in_backward:
        handle.prepare_gradient_for_optim()
        for orig_param in handle.flat_param._params:
            # Check for `None` gradient to filter parameters not in the rank
            if orig_param.grad is not None and hasattr(
                orig_param, "_in_backward_optimizers"
            ):
                # TODO (rohan-varma): For CPU offload, this unfortunately
                # operates on CPU because the parameters and gradients have
                # already been offloaded. We should run this on GPU after
                # refactoring.
                for optim in orig_param._in_backward_optimizers:
                    optim.step()

                optim.zero_grad(set_to_none=True)
        handle._reset_flat_param_grad_info_if_needed()
        if handle._offload_params:
            handle.flat_param._cpu_grad = None


def _div_if_needed(tensor: torch.Tensor, div_factor: float) -> None:
    if div_factor > 1:
        tensor.div_(div_factor)


@no_type_check
def _cast_grad_to_param_dtype(
    state: _FSDPState,
    sharded_grad: torch.Tensor,
    param: FlatParameter,
):
    """
    Casts ``sharded_grad`` back to the full parameter dtype so that the
    optimizer step runs with that dtype. This performs an actual cast if
    1. parameters were in reduced precision during the forward since then
    gradients would be in that reduced precision, or
    2. parameters were not in reduced precision but gradients were in
    reduced precision for communication.
    However, if a low precision communication hook is registered, then this
    dtype cast happens in the hook instead.
    """
    _assert_in_training_states(state, [TrainingState.FORWARD_BACKWARD])
    if not _low_precision_hook_enabled(state) and sharded_grad.dtype != param.dtype:
        low_prec_grad_data = sharded_grad.data
        sharded_grad.data = sharded_grad.data.to(dtype=param.dtype)
        # Since for `NO_SHARD`, the gradient is produced in the computation
        # stream and consumed here in the post-backward stream, inform the
        # caching allocator; for the sharded strategies, the gradient is
        # produced in the post-backward stream, so this `record_stream()`
        # should be a no-op
        _no_dispatch_record_stream(
            low_prec_grad_data, state._device_handle.current_stream()
        )


def _check_grad_to_accumulate(
    new_sharded_grad: torch.Tensor,
    accumulated_grad: torch.Tensor,
) -> None:
    _p_assert(
        accumulated_grad.shape == new_sharded_grad.shape,
        "Shape mismatch when accumulating gradients: "
        f"existing gradient shape={accumulated_grad.shape} "
        f"new gradient shape={new_sharded_grad.shape}",
    )
    _p_assert(
        accumulated_grad.device == new_sharded_grad.device,
        "Device mismatch when accumulating gradients: "
        f"existing gradient device={accumulated_grad.device} "
        f"new gradient device={new_sharded_grad.device}",
    )


@no_type_check
def _low_precision_hook_enabled(state: _FSDPState) -> bool:
    return state._comm_hook in LOW_PRECISION_HOOKS


@no_type_check
@torch.no_grad()
def _post_backward_final_callback(
    state: _FSDPState,
    module: nn.Module,
):
    """
    This waits for the post-backward to finish and performs some final cleanup.
    This runs at the end of the entire backward pass and should only be called
    on the root FSDP instance.
    """
    _p_assert(
        state._is_root,
        "The post-backward callback should only be called on the root FSDP instance",
    )
    root_state = state

    if root_state._sync_gradients:
        current_stream = state._device_handle.current_stream()
        # TODO (rohan-varma): this also waits for the overlapped optimizer step to finish
        # since it currently runs in the post-backward stream. That can be
        # pushed to the next forward if run in a different stream
        current_stream.wait_stream(root_state._post_backward_stream)
        if root_state._all_reduce_stream is not current_stream:  # uses HSDP
            current_stream.wait_stream(root_state._all_reduce_stream)
        if root_state.cpu_offload.offload_params:
            # Wait for non-blocking GPU -> CPU sharded gradient copies from the
            # post-backward hooks to finish explicitly since CPU gradients do
            # not automatically synchronize with the GPU
            state._device_handle.current_stream().synchronize()
    root_state._exec_order_data.next_iter()

    for fsdp_state in state._all_fsdp_states:
        _catch_all_reshard(fsdp_state)
        _finalize_params(fsdp_state)
        fsdp_state.training_state = TrainingState.IDLE
        handle = fsdp_state._handle
        if handle:
            handle._ran_pre_backward_hook = False
            handle._needs_pre_backward_unshard = False
            handle._post_forward_index = None
            handle._training_state = HandleTrainingState.IDLE
            handle._prefetched = False
    # Reset for cases like one forward and multiple backwards
    root_state._post_backward_callback_queued = False


@no_type_check
def _catch_all_reshard(
    state: _FSDPState,
) -> None:
    """
    Reshards the parameters that may not have been resharded in the
    post-backward hook. This can happen when a module's output is used in the
    forward pass, meaning that its pre-backward hook runs (unsharding the
    parameter), but the post-backward hook does not run because the output was
    not used in the loss computation corresponding to this backward pass.
    """
    # Wrap with a try-except to provide a more informative traceback if an
    # error is raised
    try:
        if state._handle:
            # TODO: This already-resharded check is brittle:
            # https://github.com/pytorch/pytorch/issues/83956
            already_resharded = (
                state._handle.flat_param.data_ptr()
                == state._handle.flat_param._local_shard.data_ptr()
                # If FSDP skipped using sharded views, then the flat parameter
                # still points to the sharded data, so we need to reshard to
                # use sharded views
                and not state._handle._skipped_use_sharded_views
            )
            if already_resharded:
                return
            free_unsharded_flat_param = _should_free_in_backward(state, state._handle)
            _reshard(state, state._handle, free_unsharded_flat_param)
    except Exception as e:
        _p_assert(
            False,
            f"Got exception in the catch-all reshard for {state}: {str(e)}",
            raise_assertion_error=False,
        )
        raise e


@no_type_check
def _finalize_params(
    state: _FSDPState,
) -> None:
    """Finalizes the parameters before the next iteration."""
    handle = state._handle
    if not handle:
        return
    flat_param = handle.flat_param
    if torch.distributed._functional_collectives.is_torchdynamo_compiling():
        if hasattr(flat_param, "_post_backward_hook_handle"):
            pbhs_handle = flat_param._post_backward_hook_handle
            pbhs_handle.remove()
            del flat_param._post_backward_hook_handle
    else:
        if hasattr(flat_param, "_post_backward_hook_state"):
            post_backward_hook_state_len = len(flat_param._post_backward_hook_state)
            expected_post_backward_hook_state_len = int(flat_param.requires_grad) + 1
            _p_assert(
                post_backward_hook_state_len == expected_post_backward_hook_state_len,
                f"Invalid: ``_post_backward_hook_state``: {flat_param._post_backward_hook_state}",
            )
            flat_param._post_backward_hook_state[-1].remove()
            delattr(flat_param, "_post_backward_hook_state")
    if flat_param.requires_grad:
        if not state._sync_gradients:
            # Preserve the gradient accumulation state if not synchronizing
            # gradients: `.grad` remains the unsharded gradient  from prior
            # `no_sync()` iterations, and `_saved_grad_shard` remains the
            # sharded gradient from the last synchronized iteration
            return
        if not handle._has_optim_in_backward:
            handle.prepare_gradient_for_optim()
        _p_assert(
            hasattr(flat_param, "_post_backward_called"),
            "Expects `_post_backward_called` to be set on the `FlatParameter`",
        )
        flat_param._post_backward_called = False


@no_type_check
def _prefetch_handle(
    state: _FSDPState,
    current_handle: FlatParamHandle | None,
    prefetch_mode: _PrefetchMode,
) -> None:
    """
    Prefetches the next handles if needed (without synchronization). An empty
    handles key cannot prefetch.
    """
    if not current_handle:
        return
    handle = _get_handle_to_prefetch(state, current_handle)
    if not handle:
        return
    # Temporarily emulate the training state while calling `_unshard` to
    # ensure the correct `as_params` for `_use_unsharded_views()`
    prev_training_state = handle._training_state
    if prefetch_mode == _PrefetchMode.BACKWARD:
        handle._training_state = HandleTrainingState.BACKWARD_PRE
    elif prefetch_mode == _PrefetchMode.FORWARD:
        handle._training_state = HandleTrainingState.FORWARD
    else:
        raise ValueError(f"Invalid prefetch mode on rank {state.rank}: {prefetch_mode}")
    # Prefetch the next set of handles without synchronizing to allow
    # the sync to happen as late as possible to maximize overlap
    _unshard(state, handle, state._unshard_stream, state._pre_unshard_stream)
    handle._training_state = prev_training_state
    handle._prefetched = True


@no_type_check
def _get_handle_to_prefetch(
    state: _FSDPState,
    current_handle: FlatParamHandle,
) -> FlatParamHandle:
    """
    Returns a :class:`list` of the handles keys to prefetch for the next
    module(s), where ``current_handle`` represents the current module.

    "Prefetching" refers to running the unshard logic early (without
    synchronization), and the "next" modules depend on the recorded execution
    order and the current training state.
    """
    training_state = _get_training_state(current_handle)
    valid_training_states = (
        HandleTrainingState.BACKWARD_PRE,
        HandleTrainingState.BACKWARD_POST,
        HandleTrainingState.FORWARD,
    )
    _p_assert(
        training_state in valid_training_states,
        f"Prefetching is only supported in {valid_training_states} but "
        f"currently in {training_state}",
    )
    eod = state._exec_order_data
    target_handle: FlatParamHandle | None = None
    if (
        training_state == HandleTrainingState.BACKWARD_PRE
        and state.backward_prefetch == BackwardPrefetch.BACKWARD_PRE
    ) or (
        training_state == HandleTrainingState.BACKWARD_POST
        and state.backward_prefetch == BackwardPrefetch.BACKWARD_POST
    ):
        target_handle_candidate = eod.get_handle_to_backward_prefetch(current_handle)
        if (
            target_handle_candidate
            and target_handle_candidate._needs_pre_backward_unshard
            and not target_handle_candidate._prefetched
        ):
            target_handle = target_handle_candidate
        else:
            target_handle = None
    elif training_state == HandleTrainingState.FORWARD and state.forward_prefetch:
        target_handle_candidate = eod.get_handle_to_forward_prefetch(current_handle)
        if (
            target_handle_candidate
            and target_handle_candidate._needs_pre_forward_unshard
            and not target_handle_candidate._prefetched
        ):
            target_handle = target_handle_candidate
        else:
            target_handle = None

    return target_handle


def _get_training_state(
    handle: FlatParamHandle,
) -> HandleTrainingState:
    """Returns the training state of the handles in ``handle``."""
    _p_assert(handle, "Expects a non-empty handle")
    return handle._training_state


@no_type_check
def _register_pre_forward_hook(
    state: _FSDPState,
    module: nn.Module,
) -> None:
    """
    Registers a pre-forward hook on ``module``.
    """
    for forward_handle in state._pre_forward_handles:
        forward_handle.remove()
    state._pre_forward_handles.clear()
    module_param_handle = state._fully_sharded_module_to_handle.get(module, None)
    hook = functools.partial(
        _pre_forward, state, module_param_handle, _pre_forward_unshard
    )
    state._pre_forward_handles.append(
        module.register_forward_pre_hook(hook, prepend=True, with_kwargs=True)
    )


@no_type_check
def _register_post_forward_hook(
    state: _FSDPState,
    module: nn.Module,
) -> None:
    """
    Registers a post-forward hook on ``module``. Even if the module has no
    handles, we should register the hook since it will register the module's
    pre-backward hook.
    """
    for forward_handle in state._post_forward_handles:
        forward_handle.remove()
    state._post_forward_handles.clear()
    module_param_handle = state._fully_sharded_module_to_handle.get(module, None)
    hook = functools.partial(
        _post_forward,
        state,
        module_param_handle,
        _post_forward_reshard,
    )
    state._post_forward_handles.append(module.register_forward_hook(hook))


@no_type_check
def _register_root_pre_forward_hook(
    state: _FSDPState,
    module: nn.Module,
):
    """
    Registers root pre-forward hook on ``module``, which should be the local
    FSDP root.

    NOTE: For the current composable FSDP design, we have each application of
    ``fully_shard()`` to a module to indicate that that module is the local
    FSDP root. We may remove this assumption in the future, in which case we
    will need to register this root pre-forward hook on any candidate module
    that may be the local FSDP root.
    """
    for forward_handle in state._root_pre_forward_handles:
        forward_handle.remove()
    state._root_pre_forward_handles.clear()
    hook = functools.partial(_root_pre_forward, state)
    state._root_pre_forward_handles.append(
        module.register_forward_pre_hook(hook, prepend=True, with_kwargs=True)
    )


@no_type_check
def _register_pre_backward_hooks(
    state: _FSDPState,
    module: nn.Module,
    outputs: Any,
    handle: FlatParamHandle,
) -> None:
    """
    Registers pre-backward hooks on the tensors that require gradients in the
    forward pass outputs ``outputs``, which were computed using the
    ``FlatParameter`` s of ``handles``.

    Args:
        module (nn.Module): Fully sharded module (see [Note: Fully Sharded
            Module]).

    Returns:
        Forward pass outputs with pre-backward hooks registered to tensors that
        require gradients.
    """
    # If there is no gradient computation, then there is no need for
    # pre-backward logic
    if not torch.is_grad_enabled():
        return outputs
    if state._is_root:
        state._post_backward_callback_queued = False  # only defined on the root

    if handle:
        handle._needs_pre_backward_unshard = False
        # Since these handles' `FlatParameter`s participated in a forward, we
        # conservatively assume that they will be used in the backward
        handle._ran_pre_backward_hook = False

    def _register_hook(t: torch.Tensor) -> torch.Tensor:
        if t.requires_grad:
            t.register_hook(
                torch.utils.hooks.unserializable_hook(
                    functools.partial(_pre_backward_hook, state, module, handle)
                )
            )
            if handle:
                handle._needs_pre_backward_unshard = True
        return t

    return _apply_to_tensors(_register_hook, outputs)


def _register_post_backward_hook(
    state: _FSDPState,
    handle: FlatParamHandle | None,
) -> None:
    """
    Registers post-backward hooks on the ``FlatParameter`` s'
    ``AccumulateGrad`` objects to reshard and to reduce-scatter gradients.

    The ``AccumulateGrad`` object represents the last function that finalizes
    the ``FlatParameter`` 's gradient, so it only runs after its entire
    gradient computation has finished.

    We register the post-backward hook only once in the *first* forward that a
    ``FlatParameter`` participates in. This relies on the ``AccumulateGrad``
    object being preserved through multiple forwards.

    NOTE: We follow this heuristic to prefer the *first* forward to target the
    parameter mixed precision case, where there are *separate*
    ``AccumulateGrad`` objects across the different forwards. (Without
    parameter mixed precision, the ``AccumulateGrad`` objects are the same.) If
    we instead prefer the *last* forward, then the hook runs early.
    """
    # If there is no gradient computation, then there is no need for
    # post-backward logic
    if not torch.is_grad_enabled():
        return
    if not handle:
        return
    flat_param = handle.flat_param

    if torch.distributed._functional_collectives.is_torchdynamo_compiling():
        already_registered = hasattr(flat_param, "_post_backward_hook_handle")
        if already_registered or not flat_param.requires_grad:
            return
        hook = functools.partial(_post_backward_hook, state, handle)
        hook_handle = flat_param.register_post_accumulate_grad_hook(hook)
        flat_param._post_backward_hook_handle = hook_handle  # type: ignore[attr-defined]
    else:
        already_registered = hasattr(flat_param, "_post_backward_hook_state")
        if already_registered or not flat_param.requires_grad:
            return
        # Get the `AccumulateGrad` object
        temp_flat_param = flat_param.expand_as(flat_param)
        _p_assert(
            temp_flat_param.grad_fn is not None,
            "The `grad_fn` is needed to access the `AccumulateGrad` and "
            "register the post-backward hook",
        )
        acc_grad = temp_flat_param.grad_fn.next_functions[0][0]  # type: ignore[union-attr]
        if acc_grad is None:
            raise AssertionError("Expected acc_grad to be set")
        hook_handle = acc_grad.register_hook(
            functools.partial(_post_backward_hook, state, handle)
        )
        flat_param._post_backward_hook_state = (acc_grad, hook_handle)  # type: ignore[attr-defined]


def _register_post_backward_reshard_only_hook(
    state: _FSDPState,
    handle: FlatParamHandle | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """
    Registers post-backward hooks to reshard flat parameters that do not
    require gradient. We register these using multi-post-grad hooks on the
    input activations to ensure that all gradients that may depend on the
    parameters have been computed before resharding.
    """
    # If there is no gradient computation, then there is no need for
    # post-backward logic
    if not torch.is_grad_enabled():
        return
    # Construct `inp_tensors` lazily to avoid CPU overhead in typical case
    # where each flat parameter requires gradient
    inp_tensors: list[torch.Tensor] | None = None
    if not handle:
        return
    flat_param = handle.flat_param

    if torch.distributed._functional_collectives.is_torchdynamo_compiling():
        already_registered = hasattr(flat_param, "_post_backward_hook_handle")
    else:
        already_registered = hasattr(flat_param, "_post_backward_hook_state")

    if already_registered or flat_param.requires_grad:
        return
    if inp_tensors is None:
        args_flat = pytree.arg_tree_leaves(*args, **kwargs)
        inp_tensors = [
            obj for obj in args_flat if torch.is_tensor(obj) and obj.requires_grad
        ]
    if inp_tensors is None:
        raise AssertionError("Expected inp_tensors to be set")
    hook_handle = register_multi_grad_hook(
        inp_tensors, functools.partial(_post_backward_reshard_only_hook, state, handle)
    )
    if torch.distributed._functional_collectives.is_torchdynamo_compiling():
        flat_param._post_backward_hook_handle = hook_handle  # type: ignore[attr-defined, assignment]
    else:
        flat_param._post_backward_hook_state = (hook_handle,)  # type: ignore[attr-defined, assignment]


@no_type_check
def _register_post_backward_final_callback(
    state: _FSDPState, module: nn.Module
) -> None:
    """
    Registers the post-backward final callback that runs at the end of the
    backward pass. This should be called from the root FSDP instance at the
    beginning of the pre-backward.
    """
    _p_assert(
        state._is_root,
        "Only the root FSDP instance should register the post-backward callback",
    )
    if state._post_backward_callback_queued:
        return
    _assert_in_training_states(state, [TrainingState.IDLE])
    # Trace does not need this callback
    if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
        state._post_backward_callback_queued = True
        Variable._execution_engine.queue_callback(
            functools.partial(_post_backward_final_callback, state, module)
        )


def _wait_for_computation_stream(
    computation_stream: torch.Stream,
    unshard_stream: torch.Stream,
    pre_unshard_stream: torch.Stream,
):
    """
    Has the unshard and pre-unshard streams wait for the computation stream.
    For example, this should be called in the FSDP root's pre-forward to
    respect optimizer step computation.
    """
    # Tracing does not need to wait
    if torch.distributed._functional_collectives.is_torchdynamo_compiling():
        return
    unshard_stream.wait_stream(computation_stream)  # type: ignore[attr-defined]
    # Having the pre-all-gather stream wait for the current stream even if we
    # do not leverage the pre-all-gather stream is tolerable since this only
    # runs once per iteration
    pre_unshard_stream.wait_stream(computation_stream)  # type: ignore[attr-defined]


def _reset_flat_param_grad_info_if_needed(
    handles: list[FlatParamHandle],
):
    """
    Clears the original parameters' gradients if needed. This method's CPU
    overhead is minimal, so we may call it throughout FSDP methods, which serve
    as callsites to free the gradient memory earlier.
    """
    if not isinstance(handles, list):
        handles = [handles]
    for handle in handles:
        if handle._use_orig_params:
            handle._reset_flat_param_grad_info_if_needed()


@no_type_check
def _get_buffers_and_dtypes_for_computation(
    state: _FSDPState,
    root_module: nn.Module,
) -> tuple[list[torch.Tensor], list[torch.dtype | None]]:
    """
    Returns all buffers in the module tree rooted at ``root_module`` and a
    corresponding list of the buffer dtypes for computation. Each buffer dtype
    is either ``None`` if buffer mixed precision is not enabled or the buffer
    low precision dtype otherwise.
    """
    _p_assert(state._is_root, "Expects the root to cast buffers")
    buffers: list[torch.Tensor] = []
    buffer_dtypes: list[torch.dtype | None] = []
    visited_buffers: set[torch.Tensor] = set()
    # Traverse the FSDP states bottom-up so that we prefer the owning FSDP
    # instance's mixed precision setting for each buffer
    fsdp_states, fsdp_modules = traversal_utils._get_fsdp_states_with_modules(
        root_module
    )
    for fsdp_state, fsdp_module in zip(reversed(fsdp_states), reversed(fsdp_modules)):
        for buffer_name, buffer in fsdp_module.named_buffers():
            if buffer in visited_buffers:
                continue
            visited_buffers.add(buffer)
            if clean_tensor_name(buffer_name) in fsdp_state._ignored_buffer_names:
                continue
            buffers.append(buffer)
            buffer_dtypes.append(fsdp_state.mixed_precision.buffer_dtype)
    if len(buffers) != len(buffer_dtypes):
        raise AssertionError(
            f"Expected buffers and buffer_dtypes to have the same length, got {len(buffers)} and {len(buffer_dtypes)}"
        )
    return buffers, buffer_dtypes


@no_type_check
def _get_orig_buffer_dtypes(
    state: _FSDPState,
    buffer_names: list[str],
) -> list[torch.dtype]:
    """
    Returns the original buffer types of the given buffer names.
    """
    buffer_dtypes: list[torch.dtype] = []
    for buffer_name in buffer_names:
        _p_assert(
            buffer_name in state._buffer_name_to_orig_dtype,
            f"{buffer_name} is missing from pre-computed dict on rank "
            f"{state.rank}, which only has keys "
            f"{state._buffer_name_to_orig_dtype.keys()}",
        )
        buffer_dtypes.append(state._buffer_name_to_orig_dtype[buffer_name])
    return buffer_dtypes


def _cast_buffers_to_dtype_and_device(
    buffers: list[torch.Tensor],
    buffer_dtypes: list[torch.dtype | None],
    device: torch.device,
) -> None:
    """
    Casts ``buffers`` to the dtypes given by ``buffer_dtypes`` and moves them
    to ``device``. If an element in ``buffer_dtypes`` is ``None``, then the
    corresponding buffer is only moved to ``device``.
    """
    _p_assert(
        buffer_dtypes is None or len(buffers) == len(buffer_dtypes),
        f"Expects `buffers` and `buffer_dtypes` to have the same length if "
        f"`buffer_dtypes` is specified but got {len(buffers)} and "
        f"{len(buffer_dtypes)}",
    )
    for buffer, buffer_dtype in zip(buffers, buffer_dtypes):
        if not torch.is_floating_point(buffer) or buffer_dtype is None:
            buffer.data = buffer.to(device=device)
        else:
            buffer.data = buffer.to(device=device, dtype=buffer_dtype)
<|endoftext|># mypy: allow-untyped-defs
import copy
import itertools
import math

import torch
import torch.distributed as dist
from torch._utils import _get_device_module
from torch.distributed import distributed_c10d
from torch.distributed._shard.sharded_tensor import (
    Shard,
    ShardedTensor,
    ShardedTensorMetadata,
    TensorProperties,
)
from torch.distributed._shard.sharding_spec import ShardMetadata
from torch.distributed.tensor import DeviceMesh, DTensor, Replicate, Shard as DShard


def _get_remote_device_str(rank, device_type, num_devices_per_node):
    if device_type.lower() == "cpu":
        return f"rank:{rank}/{device_type}"
    elif device_type.lower() == "hpu":
        return f"rank:{rank}/{device_type}:{_get_device_module(device_type).current_device()}"
    else:
        return f"rank:{rank}/{device_type}:{rank % num_devices_per_node}"


def _create_chunk_sharded_tensor(
    tensor: torch.Tensor,
    rank: int,
    world_size: int,
    num_devices_per_node: int,
    pg: dist.ProcessGroup,
    device: torch.device | None = None,
) -> ShardedTensor:
    """
    Shard a tensor to chunks along the first dimension. The local rank will get its
    corresponding chunk as the local shard to create a ShardedTensor.
    """
    chunks = tensor.chunk(world_size, dim=0)
    if len(chunks) > rank:
        local_shard = chunks[rank].clone()
        offsets = [0 for _ in tensor.size()]
        offsets[0] = math.ceil(tensor.size()[0] / world_size) * rank
        local_shards = [Shard.from_tensor_and_offsets(local_shard, offsets, rank)]
    else:
        local_shards = []

    # Create a ShardedTensor without invoking communication.
    chunk_sizes = [list(chunk.size()) for chunk in chunks]
    dim0_offsets = [0] + list(
        itertools.accumulate([chunk_size[0] for chunk_size in chunk_sizes])
    )[:-1]
    offsets = [0] * (len(chunk_sizes[0]) - 1)
    chunk_offsets = [[d0] + offsets for d0 in dim0_offsets]
    device_type = (
        distributed_c10d._get_pg_default_device(pg).type
        if device is None
        else device.type
    )
    placements = [
        _get_remote_device_str(
            dist.get_global_rank(pg, r),
            device_type,
            num_devices_per_node,
        )
        for r in range(len(chunk_sizes))
    ]
    if len(chunk_sizes) != len(chunk_offsets) or len(chunk_sizes) != len(placements):
        raise AssertionError(
            f"Expected chunk_sizes, chunk_offsets, and placements to have the same length, "
            f"got {len(chunk_sizes)}, {len(chunk_offsets)}, {len(placements)}"
        )
    shard_metadata = [
        ShardMetadata(offset, size, placement)
        for offset, size, placement in zip(chunk_offsets, chunk_sizes, placements)
    ]
    sharded_tensor_metadata = ShardedTensorMetadata(
        shards_metadata=shard_metadata,
        size=tensor.size(),
        tensor_properties=TensorProperties(
            dtype=tensor.dtype,
            layout=tensor.layout,
            requires_grad=False,
            memory_format=torch.contiguous_format,
            pin_memory=tensor.is_pinned(),
        ),
    )
    return ShardedTensor._init_from_local_shards_and_global_metadata(
        local_shards, sharded_tensor_metadata=sharded_tensor_metadata, process_group=pg
    )


def _create_chunk_dtensor(
    tensor: torch.Tensor,
    rank: int,
    device_mesh: DeviceMesh,
) -> DTensor:
    """
    Shard a tensor to chunks along the first dimension. The local rank will get its
    corresponding chunk as the local tensor to create a DTensor.
    """
    # We need to explicitly call .detach() to return a new tensor detached from the current graph.
    tensor = tensor.detach().clone()

    # FSDP placements: [Shard(0)]
    # HSDP placements: [Replicate(), Shard(0)]
    replicate_placements = [Replicate() for _ in range(device_mesh.ndim)]
    shard_placements = [Replicate() for _ in range(device_mesh.ndim)]
    shard_placements[-1] = DShard(0)  # type: ignore[call-overload]

    return DTensor.from_local(
        tensor, device_mesh, replicate_placements, run_check=False
    ).redistribute(
        placements=shard_placements,
    )


def _all_gather_dtensor(
    tensor: DTensor,
    root_mesh: DeviceMesh | None,
) -> torch.Tensor:
    """
    All gather a DTensor in its sharded dimension and return the local tensor.
    """
    if root_mesh != tensor.device_mesh:
        raise AssertionError("The device mesh of a tensor should be a root mesh.")

    placements = list(copy.deepcopy(tensor.placements))
    # FSDP placements: [Shard(0)] -> [Replicate()]
    # HSDP placements: [Replicate(), Shard(0)] -> [Replicate(), Replicate()]
    placements[-1] = Replicate()
    tensor = tensor.redistribute(
        device_mesh=tensor.device_mesh,
        placements=placements,
    )

    return tensor.to_local()
<|endoftext|>from ._flat_param import FlatParameter as FlatParameter
from ._fully_shard import (
    CPUOffloadPolicy,
    DataParallelMeshDims,
    FSDPModule,
    fully_shard,
    MixedPrecisionPolicy,
    OffloadPolicy,
    register_fsdp_forward_method,
    share_comm_ctx,
    UnshardHandle,
)
from .fully_sharded_data_parallel import (
    BackwardPrefetch,
    CPUOffload,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel,
    LocalOptimStateDictConfig,
    LocalStateDictConfig,
    MixedPrecision,
    OptimStateDictConfig,
    OptimStateKeyType,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictConfig,
    StateDictSettings,
    StateDictType,
)


__all__ = [
    # FSDP1
    "BackwardPrefetch",
    "CPUOffload",
    "FullOptimStateDictConfig",
    "FullStateDictConfig",
    "FullyShardedDataParallel",
    "LocalOptimStateDictConfig",
    "LocalStateDictConfig",
    "MixedPrecision",
    "OptimStateDictConfig",
    "OptimStateKeyType",
    "ShardedOptimStateDictConfig",
    "ShardedStateDictConfig",
    "ShardingStrategy",
    "StateDictConfig",
    "StateDictSettings",
    "StateDictType",
    # FSDP2
    "CPUOffloadPolicy",
    "DataParallelMeshDims",
    "FSDPModule",
    "fully_shard",
    "MixedPrecisionPolicy",
    "OffloadPolicy",
    "register_fsdp_forward_method",
    "UnshardHandle",
    "share_comm_ctx",
]

# Set namespace for exposed private names
CPUOffloadPolicy.__module__ = "torch.distributed.fsdp"
DataParallelMeshDims.__module__ = "torch.distributed.fsdp"
FSDPModule.__module__ = "torch.distributed.fsdp"
fully_shard.__module__ = "torch.distributed.fsdp"
MixedPrecisionPolicy.__module__ = "torch.distributed.fsdp"
OffloadPolicy.__module__ = "torch.distributed.fsdp"
register_fsdp_forward_method.__module__ = "torch.distributed.fsdp"
UnshardHandle.__module__ = "torch.distributed.fsdp"
share_comm_ctx.__module__ = "torch.distributed.fsdp"
<|endoftext|># mypy: allow-untyped-defs
import copy
import functools
import logging
import warnings
from collections.abc import Iterable, Iterator, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, cast, NamedTuple, no_type_check, TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed._state_dict_utils import _gather_state_dict
from torch.distributed.distributed_c10d import _get_pg_default_device
from torch.distributed.fsdp._common_utils import (
    _apply_to_modules,
    _FSDPState,
    _get_module_fsdp_state_if_fully_sharded_module,
    _get_param_to_fqns,
    _module_handle,
    _named_parameters_with_duplicates,
    clean_tensor_name,
)
from torch.distributed.fsdp._debug_utils import SimpleProfiler
from torch.distributed.fsdp._flat_param import FlatParameter, FlatParamHandle
from torch.distributed.fsdp._fsdp_extensions import (
    _ext_chunk_dtensor,
    _ext_chunk_tensor,
)
from torch.distributed.fsdp._runtime_utils import (
    _lazy_init,
    _reset_flat_param_grad_info_if_needed,
)
from torch.distributed.fsdp.api import (
    ShardingStrategy,
    StateDictSettings,
    StateDictType,
)
from torch.distributed.tensor import DTensor, Replicate
from torch.utils._pytree import tree_map_only


if TYPE_CHECKING:
    from torch.distributed._shard.sharded_tensor import ShardedTensor


logger = logging.getLogger(__name__)


@dataclass
class FSDPParamInfo:
    state: _FSDPState
    handle: FlatParamHandle
    param_indices: dict[str, int]
    param_requires_grad: list[bool]


def sorted_items(dictionary: dict[str, Any]) -> Iterator[tuple[str, Any]]:
    keys = sorted(dictionary.keys())
    for k in keys:
        yield k, dictionary[k]


@dataclass
class _ConsolidatedOptimState:
    """
    This holds the consolidated optimizer state on the target rank. Positive-
    dimension tensor state is communicated across ranks, while zero-dimension
    tensor state and non-tensor state is taken directly from the target rank.

    PyTorch version 1.12 moved to using zero-dimension tensors for scalar
    values, but user implemented optimizers may still use float (i.e. a
    non-tensor). Thus, we support both and handle them identically.

    Attributes:
        tensor_state (Dict[str, torch.Tensor]): Mapping from positive-dimension
            tensor state name to the unsharded flat tensor representing the
            state.
        zero_dim_tensor_state (Dict[str, torch.Tensor]): Mapping from zero-
            dimension tensor state name to its value.
        non_tensor_state (Dict[str, Any]): Mapping from non-tensor state
            name to its value.
    """

    tensor_state: dict[str, torch.Tensor] = field(default_factory=dict)
    zero_dim_tensor_state: dict[str, torch.Tensor] = field(default_factory=dict)
    non_tensor_state: dict[str, Any] = field(default_factory=dict)


class _PosDimTensorInfo(NamedTuple):
    """
    Metadata for positive-dimension tensors used internally for
    :meth:`scatter_full_optim_state_dict`.

    Attributes:
        shape (torch.Size): Sharded tensor shape (which is equal to the
            unsharded tensor shape if the tensor is optimizer state for a
            non-FSDP parameter and is hence not sharded).
        dtype (torch.dtype): Data type of the tensor.
    """

    shape: torch.Size
    dtype: torch.dtype


class _OptimStateKey(NamedTuple):
    """
    This represents an optimizer state key that may be used commonly across
    ranks. It is based on the unflattened parameter names rather than parameter
    IDs to make it independent of each rank's own optimizer construction.
    """

    unflat_param_names: tuple[str, ...]
    is_fsdp_managed: bool


def _unflatten_optim_state(
    fsdp_param_info: FSDPParamInfo,
    flat_param_state: dict[str, Any],
    to_save: bool,
    shard_state: bool,
    cpu_offload: bool,
) -> list[dict[str, Any]]:
    """
    Unflattens the optimizer state, consisting of the "state" part and the
    "param_groups" part. Unflattening the "state" part involves consolidating
    the state on the target rank and remapping from flattened to unflattened
    parameter IDs, and the "param_groups" part only involves remapping from
    flattened to unflattened parameter IDs.

    Args:
        fsdp_param_info (FSDPParamInfo): The FSDP state, the handle, and a
            mapping from FQN to original parameter index.
        flat_param_state (Dict[str, Any]): Entry for the flat parameter in the
            "state" part of the optimizer state dict.
        to_save (bool): Whether to save the state on this rank.

    Returns:
        List[Dict[str, Any]]: A :class:`list` holding the entries in the
        "state" part of the optimizer state dict corresponding to the
        unflattened parameters comprising the flat parameter if on the target
        rank or an empty :class:`list` otherwise. The final optimizer state
        dict will need to map these entries using the proper unflattened
        parameter IDs.
    """
    if shard_state and not to_save:
        raise AssertionError("If ``shard_state`` is True, ``to_save`` has to be True.")
    consolidated_state = _communicate_optim_state(
        fsdp_param_info,
        flat_param_state,
    )
    if to_save:
        unflat_param_state = _unflatten_communicated_optim_state(
            fsdp_param_info,
            consolidated_state,
            shard_state,
        )
        for optim_state in unflat_param_state:
            # We can't use .items() below cuz we'd run into a concurrent modification error
            if cpu_offload:
                for key in list(optim_state.keys()):
                    state = optim_state[key]
                    if not isinstance(state, torch.Tensor):
                        continue
                    optim_state[key] = state.cpu()
        return unflat_param_state
    else:
        return []


def _is_zero_dim_tensor(x: Any) -> bool:
    return torch.is_tensor(x) and x.dim() == 0


def _communicate_optim_state(
    fsdp_param_info: FSDPParamInfo,
    flat_param_state: dict[str, Any],
) -> _ConsolidatedOptimState:
    """
    Communicates the optimizer state for a flat parameter across ranks. All
    ranks will hold the entire non-sharded optimizer state on GPU.

    If ``N`` is the number of tensor optimizer states in the optimizer state
    dict, then the communication complexity is 0 if ``N = 0`` and ``N + 1``
    otherwise (where the plus 1 comes from all-gathering the padding per rank).

    Args:
        fsdp_param_info (FSDPParamInfo): The FSDP state, the handle, and a
            mapping from FQN to original parameter index.
        flat_param_state (Dict[str, Any]): The entry in the "state" part of the
            optimizer state dict corresponding to the flat parameter.

    Returns:
        ConsolidatedOptimState: Consolidated optimizer state for the target
        flat parameter.
    """
    fsdp_state = fsdp_param_info.state
    flat_param = fsdp_param_info.handle.flat_param
    state = _ConsolidatedOptimState()
    tensor_state, zero_dim_tensor_state, non_tensor_state = (
        state.tensor_state,
        state.zero_dim_tensor_state,
        state.non_tensor_state,
    )

    for state_name, value in sorted_items(flat_param_state):
        # Positive-dimension tensor state: communicate across ranks
        if torch.is_tensor(value) and value.dim() > 0:
            # If the parameter is not sharded, then neither is the
            # positive-dimension tensor state, so no need to communicate it --
            # we take the target rank's value
            if (
                fsdp_state.world_size == 1
                or fsdp_state.sharding_strategy == ShardingStrategy.NO_SHARD
            ):
                tensor_state[state_name] = value
                continue
            if fsdp_state.compute_device is None:
                raise AssertionError("compute_device has not been initialized")
            if value.device.type != fsdp_state.compute_device.type:
                value = value.to(fsdp_state.compute_device)
            # Assume that positive-dimension tensor optimizer state
            # has the same shape as the sharded flat parameter
            buffer_size = flat_param._full_param_padded.size()  # type: ignore[attr-defined]
            tensor_buffer = value.new_zeros(*buffer_size)
            dist.all_gather_single(tensor_buffer, value, group=fsdp_state.process_group)
            fsdp_state._device_handle.synchronize()
            unpadded_numel = cast(
                nn.Parameter, flat_param._unpadded_unsharded_size
            ).numel()
            tensor_state[state_name] = tensor_buffer[:unpadded_numel]
        # Zero-dimension tensor state and non-tensor state: take this rank's
        # value directly
        else:
            if _is_zero_dim_tensor(value):
                zero_dim_tensor_state[state_name] = value.detach().clone()
            else:
                non_tensor_state[state_name] = value
    return state


def _unflatten_communicated_optim_state(
    fsdp_param_info: FSDPParamInfo,
    state: _ConsolidatedOptimState,
    shard_state: bool,
) -> list[dict[str, Any]]:
    """
    Unflattens the communicated optimizer state (given by ``tensor_state``,
    ``non_tensor_state``, and ``zero_dim_tensor_state``) for a single flat
    parameter. This should only be called on the target rank.

    Args:
        fsdp_param_info (FSDPParamInfo): The FSDP state, the handle, and a
            mapping from FQN to original parameter index.
        state (_ConsolidatedOptimState): Consolidated optimizer state.

    Returns:
        List[Dict[str, Any]]: A :class:`list` holding the entries in the
        "state" part of the optimizer state dict corresponding to the
        unflattened parameters comprising the flat parameter. The final
        optimizer state dict will need to map these entries using the proper
        unflattened parameter IDs.
    """
    fsdp_state = fsdp_param_info.state
    handle = fsdp_param_info.handle
    flat_param = handle.flat_param
    unflat_param_state: list[dict[str, Any]] = []
    flat_param_views: dict[str, Iterator] = {}
    num_unflat_params = flat_param._num_params
    tensor_state, zero_dim_tensor_state, non_tensor_state = (
        state.tensor_state,
        state.zero_dim_tensor_state,
        state.non_tensor_state,
    )

    for _ in range(num_unflat_params):
        unflat_state_param = {}
        # Add positive-dimension tensor state: unflatten with views
        for state_name, flat_tensor in sorted_items(tensor_state):
            views_generated = state_name in flat_param_views
            if not views_generated:
                views = handle._get_unflat_views(flat_tensor)
                flat_param_views[state_name] = views
            else:
                views = flat_param_views[state_name]
            optim_state: torch.Tensor | ShardedTensor | DTensor = next(views)
            if shard_state:
                osd_config = fsdp_state._optim_state_dict_config
                if getattr(osd_config, "_use_dtensor", False):
                    if fsdp_state._device_mesh is None:
                        raise AssertionError(
                            f"Expected _device_mesh to be not None, got {fsdp_state._device_mesh}"
                        )
                    optim_state = _ext_chunk_dtensor(
                        optim_state,
                        fsdp_state.rank,
                        fsdp_state._device_mesh,
                        fsdp_state._fsdp_extension,
                    )
                else:
                    if fsdp_state.process_group is None:
                        raise AssertionError(
                            f"Expected process_group to be not None, got {fsdp_state.process_group}"
                        )
                    optim_state = _ext_chunk_tensor(
                        optim_state,
                        fsdp_state.rank,
                        fsdp_state.world_size,
                        fsdp_state._device_handle.device_count(),
                        fsdp_state.process_group,
                        fsdp_state._fsdp_extension,
                    )
            unflat_state_param[state_name] = optim_state

        # Add zero-dimension tensor state: take the target rank's value
        unflat_state_param.update(sorted_items(zero_dim_tensor_state))
        # Add non-tensor state: take the target rank's value
        unflat_state_param.update(sorted_items(non_tensor_state))
        unflat_param_state.append(unflat_state_param)
    return unflat_param_state


def _broadcast_processed_state(
    fsdp_state: _FSDPState,
    optim_state: dict[str, Any],
    group: dist.ProcessGroup | None,
) -> dict[str, Any]:
    objects: list[Any] = [None]
    if dist.get_rank(group) == 0:
        objects[0] = tree_map_only(
            torch.Tensor,
            lambda v: v.cpu() if v.dim() == 0 else _PosDimTensorInfo(v.shape, v.dtype),  # type: ignore[union-attr]
            optim_state,
        )
    dist.broadcast_object_list(objects, src=0, group=group)
    if dist.get_rank(group) == 0:
        return optim_state
    else:
        return objects[0]


def _broadcast_state(
    fsdp_state: _FSDPState, state: Any, group: dist.ProcessGroup | None
) -> Any:
    if dist.get_rank(group) == 0:
        if not isinstance(state, torch.Tensor) or state.dim() == 0:
            return state
        tensor = state.to(fsdp_state.compute_device)
    else:
        if isinstance(state, torch.Tensor):
            if state.dim() != 0:
                raise AssertionError(
                    "For non-zero ranks, a tensor state should have zero dimension, "
                    f"but got the state with shape {state.shape}."
                )
            return state
        elif not isinstance(state, _PosDimTensorInfo):
            return state
        tensor = torch.zeros(
            state.shape, dtype=state.dtype, device=fsdp_state.compute_device
        )
    dist.broadcast(tensor, src=0, group=group)
    return tensor


def _shard_orig_param_state(
    fsdp_param_info: FSDPParamInfo,
    fqn: str,
    optim_state: dict[str, Any],
) -> dict[str, Any]:
    """
    Shard the optimizer state for the original parameter with the name ``fqn``.
    This API should only be used when ``use_orig_params`` is True.
    """
    if not optim_state:
        return {}
    fsdp_state = fsdp_param_info.state
    flat_param = fsdp_param_info.handle.flat_param
    param_idx = fsdp_param_info.param_indices[fqn]
    shard_param_info = flat_param._shard_param_infos[param_idx]  # type: ignore[attr-defined]
    optim_state = _gather_state_dict(
        optim_state, pg=fsdp_state.process_group, device=fsdp_state.compute_device
    )
    if not shard_param_info.in_shard:
        return {}
    # Flatten and shard the state.
    new_optim_state: dict[str, Any] = {}
    intra_param_start_idx = shard_param_info.intra_param_start_idx
    intra_param_end_idx = shard_param_info.intra_param_end_idx
    for state_name, value in optim_state.items():
        if (
            torch.is_tensor(value)
            and value.dim() > 0
            and fsdp_state.sharding_strategy != ShardingStrategy.NO_SHARD
        ):
            value = value.flatten()[
                intra_param_start_idx : intra_param_end_idx  # type: ignore[operator]
                + 1
            ].clone()
        new_optim_state[state_name] = value
    return new_optim_state


def _flatten_optim_state_dict(
    optim_state_dict: dict[str, Any],
    model: nn.Module,
    use_orig_params: bool = False,
    optim: torch.optim.Optimizer | None = None,
    rank0_only: bool = False,
    group: dist.ProcessGroup | None = None,
) -> dict[str, Any]:
    """
    Flattens the full optimizer state dict, still keying by unflattened parameter
    names.

    If ``use_orig_params`` is True, each rank will have all FSDP-managed
    parameters but some of these parameters may be empty due to the sharding.
    For a regular optim.Optimizer, states for those empty parameters will
    not be initialized. So, when aggregating the FQNs across ranks, no assert
    will be raised on a rank even if it does not have all the states -- it is
    valid and FSDP know how to aggregate them. However, FSDP has to ignore
    handling those parameters that are not managed by FSDP and do not exist on
    the local rank -- it is managed by other parallelism and FSDP does not
    know ho to handle/aggregate them.

    Note that ``_flatten_tensor_optim_state`` does not need ``optim`` to
    flatten/shard the state. However, NamedOptimizer and KeyedOptimizer require
    all the states even if the corresponding parameters are empty. To this end,
    ``optim`` will be used to get the initial state of the empty parameters.
    ``optim`` should only be non-None if the ``optim` is KeyedOptimizer or
    NamedOptimizer.

    Returns:
        Dict[str, Any]: The flattened optimizer state dict.
    """
    SimpleProfiler.reset()

    unflat_osd = optim_state_dict
    if "state" not in unflat_osd and not rank0_only:
        raise ValueError(
            '`optim_state_dict` must have the keys "state"'
            "to be a valid optimizer state dict"
        )
    param_to_fqns = _get_param_to_fqns(model)
    fqn_to_fsdp_param_info = _get_fqn_to_fsdp_param_info(model)
    fsdp_state = next(iter(fqn_to_fsdp_param_info.values())).state

    # Broadcast unflat_osd without non-scalar tensor if rank0_only is True.
    if rank0_only:
        unflat_osd = _broadcast_processed_state(fsdp_state, unflat_osd, group=group)

    # Construct the "state" part
    flat_osd_state: dict[_OptimStateKey | str, Any] = {}
    unflat_osd_state = unflat_osd["state"]
    all_state_keys = set(unflat_osd_state.keys())

    for param, fqns in param_to_fqns.items():
        fqn = fqns[0]
        if fqn not in unflat_osd_state:
            continue
        all_state_keys.difference_update(fqns)

        if rank0_only:
            for fqn in fqns:
                if not unflat_osd_state[fqn]:
                    continue
                for state_name in unflat_osd_state[fqn]:
                    unflat_osd_state[fqn][state_name] = _broadcast_state(
                        fsdp_state, unflat_osd_state[fqn][state_name], group=group
                    )
            fqn = fqns[0]
        if fqn in fqn_to_fsdp_param_info:
            fsdp_param_info = fqn_to_fsdp_param_info[fqn]
            if use_orig_params:
                with SimpleProfiler.profile(SimpleProfiler.Type.RESHARDING):
                    flat_state = _shard_orig_param_state(
                        fsdp_param_info,
                        fqn,
                        unflat_osd_state[fqn],
                    )
            else:
                flat_state = _flatten_optim_state(
                    fsdp_param_info,
                    unflat_osd_state,
                    fqns,
                )
            key = _OptimStateKey(tuple(fqns), True)
            # Only include non-empty states since as expected by
            # `torch.optim.Optimizer` s unless the optimizer is KeyedOptimizer
            # or NamedOptimizer.
            if flat_state:
                flat_osd_state[key] = flat_state
            elif use_orig_params:
                if len(fqns) != 1:
                    raise AssertionError(
                        f"use_orig_params is True but there are multiple FQNs, {fqns}."
                    )
                if optim is not None:  # NamedOptimizer or KeyedOptimizer case.
                    state = optim.state.get(param, None)  # type: ignore[call-overload]
                    if state is not None:
                        flat_osd_state[key] = copy.deepcopy(state)
                    else:
                        warnings.warn(
                            f"optim_state[{key}] is not on rank{fsdp_state.rank}.",
                            stacklevel=2,
                        )

            else:
                raise RuntimeError(
                    f"The state of {key} is empty. This should happen when "
                    "use_orig_params=True."
                )
        else:  # do not flatten non-FSDP parameters' states
            if len(fqns) != 1:
                raise AssertionError(f"Expected len(fqns) == 1, got {len(fqns)}")
            key = _OptimStateKey(tuple(fqns), False)
            flat_osd_state[key] = copy.copy(unflat_osd_state[fqn])

        if rank0_only:
            for fqn in fqns:
                if not unflat_osd_state[fqn]:
                    continue
                for state_name, param_state in list(unflat_osd_state[fqn].items()):
                    if fsdp_state.rank > 0:
                        # Deference the tensor so that PyTorch can collect the memory.
                        del unflat_osd_state[fqn][state_name]
                    else:
                        # Move the tensor in the original osd back to CPU to make the
                        # original osd unaffected.
                        unflat_osd_state[fqn][state_name] = param_state.cpu()

    # Handle user-defined state, states that are not associated with parameters.
    for key in all_state_keys:
        user_state = unflat_osd_state[key]
        if isinstance(user_state, torch.Tensor) and rank0_only and use_orig_params:
            user_state = _broadcast_state(fsdp_state, user_state, group=group)
        flat_osd_state[key] = copy.copy(user_state)

    SimpleProfiler.dump_and_reset("FSDP _flatten_optim_state_dict() profiling: ")
    # Construct the "param_groups" part -- copy as is since it will be
    # rekeyed later according to the target rank's optimizer
    # Only copy param_groups if it exists in unflat_osd
    if "param_groups" in unflat_osd:
        flat_osd_param_groups = copy.deepcopy(unflat_osd["param_groups"])
        return {"state": flat_osd_state, "param_groups": flat_osd_param_groups}
    else:
        return {"state": flat_osd_state}


def _flatten_optim_state(
    fsdp_param_info: FSDPParamInfo,
    unflat_osd_state: dict[str, dict[str, Any]],
    unflat_param_names: list[str],
) -> dict[str, Any]:
    """
    Flattens the optimizer state in ``full_optim_state_dict`` for a single
    flat parameter in ``fsdp_param_info`` corresponding to the unflattened
    parameter names in ``unflat_param_names``.

    Args:
        fsdp_param_info (FSDPParamInfo): The FSDP state, the handle, and a
            mapping from FQN to original parameter index.
        unflat_osd_state (Dict[str, Dict[str, Any]]): The "state" part of the
            optimizer state dict corresponding to the unflattened parameters.
        unflat_param_names (List[str]): A :class:`list` of unflattened
            parameter names corresponding to the flat parameter ``flat_param``.

    Returns:
        Dict[str, Any]: A :class:`dict` mapping state names to their values for
        a particular flat parameter. The sharded optimizer state dict's "state"
        part will map a key to this returned value.
    """
    fsdp_state = fsdp_param_info.state
    handle = fsdp_param_info.handle
    flat_param = handle.flat_param
    num_unflat_params = len(unflat_param_names)
    if num_unflat_params <= 0:
        raise AssertionError(
            "Expects at least one unflattened parameter corresponding to the flat parameter"
        )
    unflat_param_shapes = flat_param._shapes
    num_unflat_param_shapes = len(unflat_param_shapes)
    if num_unflat_params != num_unflat_param_shapes:
        raise AssertionError(
            f"Expects {num_unflat_params} shapes but got {num_unflat_param_shapes}"
        )

    # Check if these unflattened parameters have any optimizer state
    has_state = [
        bool(unflat_param_name in unflat_osd_state)
        for unflat_param_name in unflat_param_names
    ]
    # If none of the unflattened parameters comprising this flat parameter have
    # any state, then we do not want an entry in the optimizer state dict
    if not any(has_state):
        return {}  # no need to flatten any state
    # There may still be some unflattened parameters with state and some
    # without
    unflat_param_states = [
        _gather_state_dict(
            unflat_osd_state[unflat_param_name],
            pg=fsdp_state.process_group,
            device=fsdp_state.compute_device,
        )
        if unflat_param_name in unflat_osd_state
        else None
        for unflat_param_name in unflat_param_names
    ]
    # Check that the unflattened parameters have the same state names
    state_names = None
    # pyrefly: ignore [bad-assignment]
    for unflat_param_state in unflat_param_states:
        if unflat_param_state is None:
            continue
        if state_names is None:
            state_names = set(unflat_param_state.keys())
        else:
            if state_names != set(unflat_param_state.keys()):
                raise ValueError(
                    "Differing optimizer state names for the unflattened "
                    f"parameters: {unflat_param_names}"
                )
    if state_names is None:
        raise AssertionError(f"Expected state_names to be not None, got {state_names}")

    # Flatten the state
    flat_state: dict[str, torch.Tensor | None] = {}
    for state_name in state_names:
        state_values = [
            unflat_param_state[state_name] if unflat_param_state is not None else None
            for unflat_param_state in unflat_param_states
        ]
        non_none_state_values = [v for v in state_values if v is not None]
        # If all ranks have None, this is a None value
        if not non_none_state_values:
            flat_state[state_name] = None
            continue
        are_pos_dim_tensors = are_zero_dim_tensors = are_non_tensors = True
        for v in non_none_state_values:
            are_pos_dim_tensors &= torch.is_tensor(v) and v.dim() > 0
            are_zero_dim_tensors &= _is_zero_dim_tensor(v)
            are_non_tensors &= not torch.is_tensor(v)
        types = {type(v) for v in non_none_state_values}
        if len(types) != 1 or not (
            are_pos_dim_tensors or are_zero_dim_tensors or are_non_tensors
        ):
            raise ValueError(
                f"Differing optimizer state types for state {state_name}, "
                f"values {non_none_state_values}, and unflattened parameter "
                f"names {unflat_param_names}"
            )
        if are_pos_dim_tensors:
            flat_tensor = _flatten_tensor_optim_state(
                state_name,
                state_values,  # type: ignore[arg-type]
                unflat_param_names,
                unflat_param_shapes,
                handle,
            )
            # Shard the flattened tensor immediately to minimize max memory
            # usage
            if (
                fsdp_state.world_size != 1
                and fsdp_state.sharding_strategy != ShardingStrategy.NO_SHARD
            ):
                sharded_flat_tensor, _ = FlatParamHandle._get_shard(
                    flat_tensor,
                    fsdp_state.rank,
                    fsdp_state.world_size,
                )
            else:
                sharded_flat_tensor = flat_tensor
            flat_state[state_name] = sharded_flat_tensor
        elif are_zero_dim_tensors:
            flat_state[state_name] = _flatten_zero_dim_tensor_optim_state(
                state_name,
                state_values,  # type: ignore[arg-type]
                unflat_param_names,
            )
        else:
            if not are_non_tensors:
                raise AssertionError(
                    f"Expected are_non_tensors to be True, got {are_non_tensors}"
                )
            flat_state[state_name] = _flatten_non_tensor_optim_state(
                state_name,
                state_values,
                unflat_param_names,
            )

    return flat_state


def _flatten_tensor_optim_state(
    state_name: str,
    pos_dim_tensors: list[torch.Tensor],
    unflat_param_names: list[str],
    unflat_param_shapes: Sequence[torch.Size],
    handle: FlatParamHandle,
) -> torch.Tensor:
    """
    Flattens the positive-dimension tensor optimizer state given by the values
    ``tensors`` for the state ``state_name`` for a single flat parameter
    from ``handle`` corresponding to the unflattened parameter names
    ``unflat_param_names`` and unflatted parameter shapes
    ``unflat_param_shapes``. This flattens each unflattened parameter's tensor
    state into one tensor.

    NOTE: We use zero tensors for any unflattened parameters without state
    since some value is required to fill those entries. This assumes that the
    zero tensor is mathematically equivalent to having no state, which is true
    for Adam's "exp_avg" and "exp_avg_sq" but may not be true for all
    optimizers.

    Args:
        state_name (str): Optimizer state name.
        pos_dim_tensors (List[torch.Tensor]): Positive-dimension tensor
            optimizer state values for the unflattened parameters corresponding
            to the single flat parameter.
        unflat_param_names (List[str]): A :class:`list` of unflattened
            parameter names corresponding to the single flat parameter.
        unflat_param_shapes (List[torch.Size]): Unflattened parameter shapes
            corresponding to the single flat parameter.
        handle (FlatParamHandle): The flat parameter's handle.

    Returns:
        torch.Tensor: A flat tensor containing the optimizer state
        corresponding to ``state_name`` constructed by concatenating the
        unflattened parameter tensor states in ``pos_dim_tensors`` (using zero
        tensors for any unflattened parameters without the state).
    """
    flat_param = handle.flat_param
    non_none_tensors = [t for t in pos_dim_tensors if t is not None]
    # Check that all are tensors with the same dtype
    dtypes = {t.dtype for t in non_none_tensors}
    if len(dtypes) != 1:
        raise ValueError(
            "All unflattened parameters comprising a single flat "
            "parameter must have positive-dimension tensor state with the "
            f"same dtype but got dtypes {dtypes} for state {state_name} and "
            f"unflattened parameter names {unflat_param_names}"
        )
    dtype = next(iter(dtypes))
    # Check that each tensor state matches its parameter's shape
    for tensor, shape in zip(pos_dim_tensors, unflat_param_shapes):
        if tensor is None and len(shape) == 0:
            raise ValueError("Flattening a zero-dimension parameter is not supported")
        elif tensor is not None and tensor.shape != shape:
            raise ValueError(
                "Tensor optimizer state does not have same shape as its "
                f"parameter: {tensor.shape} {shape}"
            )
    # Flatten the tensor states: we do not need to add any right-hand-side
    # padding since the flat optimizer state tensor is sharded via
    # `_get_shard()`, which pads the shard as needed (just like for the flat
    # parameter)
    cpu_device = torch.device("cpu")
    tensors_to_flatten = [
        torch.flatten(state_value.to(cpu_device))
        if state_value is not None
        else torch.flatten(
            torch.zeros(
                size=shape,
                dtype=dtype,
                device=cpu_device,
            )
        )
        for state_value, shape in zip(pos_dim_tensors, unflat_param_shapes)
    ]
    flat_tensor = handle.flatten_tensors(tensors_to_flatten, handle._aligned_numel)
    flat_param_shape = flat_param._unpadded_unsharded_size  # type: ignore[attr-defined]
    if flat_tensor.shape != flat_param_shape:
        raise AssertionError(
            f"tensor optim state: {flat_tensor.shape} flat parameter: {flat_param_shape}"
        )
    return flat_tensor


def _flatten_zero_dim_tensor_optim_state(
    state_name: str,
    zero_dim_tensors: list[torch.Tensor],
    unflat_param_names: list[str],
) -> torch.Tensor:
    """
    Flattens the zero-dimension tensor optimizer state given by the values
    ``zero_dim_tensors`` for the state ``state_name`` for a single flat
    parameter corresponding to the unflattened parameter names
    ``unflat_param_names`` by enforcing that all tensors are the same and using
    that common value.

    NOTE: The requirement that the tensors are the same across all unflattened
    parameters comprising the flat parameter is needed to maintain the
    invariant that FSDP performs the same computation as its non-sharded
    equivalent. This means that none of the unflattened parameters can be
    missing this state since imposing a value may differ from having no value.
    For example, for Adam's "step", no value means maximum bias correction,
    while having some positive value means less bias correction.

    Args:
        state_name (str): Optimizer state name.
        zero_dim_tensors (List[torch.Tensor]): Zero-dimension optimizer state
            for the unflattened parameters corresponding to the single
            flat parameter.
        unflat_param_names (List[str]): A :class:`list` of unflattened
            parameter names corresponding to the single flat parameter.

    Returns:
        torch.Tensor: A zero-dimensional tensor giving the value of the state
        ``state_name`` for all unflattened parameters corresponding to the
        names ``unflat_param_names``.
    """
    non_none_tensors = [t for t in zero_dim_tensors if t is not None]
    # Enforce that all have the same value and dtype
    values_set = {t.item() if t is not None else None for t in zero_dim_tensors}
    dtypes = {t.dtype if t is not None else None for t in zero_dim_tensors}
    if (
        len(non_none_tensors) != len(zero_dim_tensors)
        or len(values_set) != 1
        or len(dtypes) != 1
    ):
        raise ValueError(
            "All unflattened parameters comprising a single flat "
            "parameter must have scalar state with the same value and dtype "
            f"but got values {values_set} and dtypes {dtypes} for state "
            f"{state_name} and unflattened parameter names "
            f"{unflat_param_names}"
        )
    value = next(iter(values_set))
    dtype = next(iter(dtypes))
    return torch.tensor(value, dtype=dtype, device=torch.device("cpu"))


def _flatten_non_tensor_optim_state(
    state_name: str,
    non_tensors: list[Any],
    unflat_param_names: list[str],
) -> Any:
    """
    Flattens the non-tensor optimizer state given by the values ``non_tensors``
    for the state ``state_name`` for a single flat parameter corresponding
    to the unflattened parameter names ``unflat_param_names`` by enforcing that
    all values are the same and using that common value.

    See the note in :func:`_flatten_zero_dim_tensor_optim_state`.

    Args:
        state_name (str): Optimizer state name.
        non_tensors (List[Any]): Non-tensor optimizer state for the unflattened
            parameters corresponding to the single flat parameter.
        unflat_param_names (List[str]): A :class:`list` of unflattened
            parameter names corresponding to the single flat parameter.

    Returns:
        Any: A non-tensor giving the value of the state ``state_name`` for all
        unflattened parameters corresponding to the names
        ``unflat_param_names``.
    """
    non_none_non_tensors = [nt for nt in non_tensors if nt is not None]
    # Enforce that all have the same value (same type already checked)
    non_tensor_set = set(non_tensors)
    if len(non_none_non_tensors) != len(non_tensors) or len(non_tensor_set) != 1:
        raise ValueError(
            "All unflattened parameters comprising a single flat "
            "parameter must have scalar state with the same value and dtype "
            f"but got values {non_tensor_set} for state {state_name} and  "
            f"unflattened parameter names {unflat_param_names}"
        )
    non_tensor = next(iter(non_tensor_set))
    return non_tensor


def _rekey_sharded_optim_state_dict(
    sharded_osd: dict[str, Any],
    model: nn.Module,
    optim: torch.optim.Optimizer,
    optim_input: list[dict[str, Any]] | Iterable[nn.Parameter] | None,
    using_optim_input: bool,
    is_named_optimizer: bool = False,
) -> dict[str, Any]:
    """
    Rekeys the optimizer state dict from unflattened parameter names to flat
    parameter IDs according to the calling rank's ``optim``, which may be
    different across ranks. In particular, the unflattened parameter names are
    represented as :class:`_OptimStateKey` s.
    """
    param_to_fqns = _get_param_to_fqns(model)
    flat_param_to_fqn = _get_flat_param_to_fqn(model)
    param_to_param_key: dict[nn.Parameter, int | str] = cast(
        dict[nn.Parameter, int | str],
        (
            _get_param_to_param_id_from_optim_input(model, optim_input)
            if using_optim_input
            else _get_param_to_param_key(
                optim, model, is_named_optimizer, param_to_fqns, flat_param_to_fqn
            )
        ),
    )
    # All parameter keys in `param_to_param_key` should be in
    # `param_to_fqns` -- strict inequality follows when not all parameters are
    # passed to the optimizer
    if len(param_to_param_key) > len(param_to_fqns):
        raise AssertionError(
            f"Expected len(param_to_param_key) <= len(param_to_fqns), got {len(param_to_param_key)} > {len(param_to_fqns)}"
        )

    unflat_param_names_to_flat_param_key: dict[
        tuple[str, ...], int | str
    ] = {}  # for "state"
    unflat_param_name_to_flat_param_key: dict[str, int | str] = {}  # for "param_groups"
    for param, unflat_param_names in param_to_fqns.items():
        if param not in param_to_param_key:
            # This parameter was not passed to the optimizer
            continue
        flat_param_key = param_to_param_key[param]
        unflat_param_names_to_flat_param_key[tuple(unflat_param_names)] = flat_param_key
        for unflat_param_name in unflat_param_names:
            unflat_param_name_to_flat_param_key[unflat_param_name] = flat_param_key

    sharded_osd_state = sharded_osd["state"]
    rekeyed_osd_state: dict[str | int, Any] = {}
    for key, param_state in sharded_osd_state.items():
        if isinstance(key, str):
            rekeyed_osd_state[key] = param_state
            continue
        flat_param_key = unflat_param_names_to_flat_param_key.get(
            key.unflat_param_names, key.unflat_param_names
        )

        rekeyed_osd_state[flat_param_key] = param_state

    # Only process param_groups if it exists in sharded_osd
    if "param_groups" in sharded_osd:
        rekeyed_osd_param_groups: list[dict[str, Any]] = []
        for unflat_param_group in sharded_osd["param_groups"]:
            flat_param_group = copy.deepcopy(unflat_param_group)
            flat_param_keys = sorted(
                {
                    unflat_param_name_to_flat_param_key[unflat_param_name]
                    for unflat_param_name in unflat_param_group["params"]
                }
            )
            flat_param_group["params"] = flat_param_keys
            rekeyed_osd_param_groups.append(flat_param_group)
        return {"state": rekeyed_osd_state, "param_groups": rekeyed_osd_param_groups}
    else:
        return {"state": rekeyed_osd_state}


def _get_param_id_to_param_from_optim_input(
    model: nn.Module,
    optim_input: list[dict[str, Any]] | Iterable[nn.Parameter] | None = None,
) -> dict[int, nn.Parameter]:
    """
    Constructs a mapping from parameter IDs to parameters. This may be used
    both for models with ``FlatParameter`` s and without.

    NOTE: This method is only preserved for backward compatibility. The method
    :meth:`_get_param_key_to_param` is the preferred code path that does not
    rely on ``optim_input``.

    NOTE: We critically assume that, whether the optimizer input is a list of
    parameters or a list of parameter groups, :class:`torch.optim.Optimizer`
    enumerates the parameter IDs in order. In other words, for a parameter list
    input, the parameter IDs should be in that list order, and for a parameter
    groups input, the parameter IDs should be in order within each parameter
    group and in order across parameter groups.

    Args:
        model (nn.Module): Model whose parameters are passed into the
            optimizer.
        optim_input (Optional[Union[List[Dict[str, Any]],
        Iterable[nn.Parameter]]]): Input passed into the optimizer
            representing either a :class:`list` of parameter groups or an
            iterable of parameters; if ``None``, then this method assumes the
            input was ``model.parameters()``. (Default: ``None``)

    Returns:
        List[nn.Parameter]: Mapping from parameter IDs to parameters,
        where the parameter ID is implicitly the index in the :class:`list`.
    """
    # Assume the standard case of passing `model.parameters()` to the optimizer
    # if `optim_input` is not specified
    if optim_input is None:
        return dict(enumerate(model.parameters()))
    try:
        # pyrefly: ignore [redundant-cast]
        params = cast(list[nn.Parameter], list(optim_input))
    except TypeError as e:
        raise TypeError(
            "Optimizer input should be an iterable of Tensors or dicts, "
            f"but got {optim_input}"
        ) from e
    if len(params) == 0:
        raise ValueError("Optimizer input should not be empty")

    # Check if the optimizer input represents tensors or parameter groups
    all_tensors = True
    all_dicts = True
    for param in params:
        all_tensors &= isinstance(param, torch.Tensor)
        all_dicts &= isinstance(param, dict)
    if not all_tensors and not all_dicts:
        raise TypeError("Optimizer input should be an iterable of Tensors or dicts")
    if all_tensors:
        return dict(enumerate(params))
    if not all_dicts:
        raise AssertionError(f"Expected all_dicts to be True, got {all_dicts}")
    param_id_to_param: list[nn.Parameter] = []
    for param_group in params:
        has_params_key = "params" in param_group  # type: ignore[operator]
        if not has_params_key:
            raise AssertionError(
                'A parameter group should map "params" to a list of the parameters in the group'
            )
        # Implicitly map `flat_param_id` (current length of the list) to
        # `param`
        param_id_to_param.extend(param_group["params"])  # type: ignore[index]
    return dict(enumerate(param_id_to_param))


def _get_flat_param_to_fqn(model: torch.nn.Module) -> dict[FlatParameter, str]:
    """
    Constructs a mapping from ``FlatParameter`` to a cleaned (devoid of prefixes
    from wrappers) fully qualified name (FQN). Note that this FQN is "non-canonical"
    because ``FlatParameter``  s do not come from the original module but are
    registered only after FSDP has been applied. This function returns the FSDP-given
    name for the ``FlatParameter`` (usually module._flat_param) as opposed to the
    canonical FQNs returned for ``FlatParameter`` s in ``_common_utils._get_param_to_fqns(...)``).

    Consequently, this function will only return a non-empty mapping if FSDP was
    applied with ``use_orig_params=False`` as, otherwise, the original parameters
    are used within the module and there would be no ``FlatParameter`` s in the module.

    """

    def module_fn(module, prefix, tree_level, flat_param_to_fqn):
        for param_name, param in _named_parameters_with_duplicates(
            module, recurse=False
        ):
            if not isinstance(param, FlatParameter):
                continue
            fqn = clean_tensor_name(prefix + param_name)
            flat_param_to_fqn[param] = fqn

    def return_fn(flat_param_to_fqn):
        return flat_param_to_fqn

    flat_param_to_fqn_ret: dict[FlatParameter, str] = {}
    return _apply_to_modules(
        model,
        module_fn,
        return_fn,
        [fqn for fqn, _ in _named_parameters_with_duplicates(model)],
        flat_param_to_fqn_ret,
    )


def _get_param_key_to_param(
    optim: torch.optim.Optimizer,
    model: nn.Module | None = None,
    is_named_optimizer: bool = False,
    param_to_fqns: dict[nn.Parameter, list[str]] | None = None,
    flat_param_to_fqn: dict[FlatParameter, str] | None = None,
) -> dict[int | str, nn.Parameter]:
    """
    Constructs a mapping from parameter keys to parameters. For the regular
    optimizers, the keys are parameter IDs. For NamedOptimizer, the keys
    are FQNs. This API may be used both for models with ``FlatParameter`` s and
    without.
    """
    clean_fqn_to_curr_fqn: dict[str, str] = {}
    if is_named_optimizer:
        if param_to_fqns is None or flat_param_to_fqn is None:
            raise AssertionError(
                "The optimizer is a NamedOptimizer, `param_to_fqns` must not be None."
            )
        if model is None:
            raise AssertionError(f"Expected model to be not None, got {model}")
        for key, _ in _named_parameters_with_duplicates(model):
            clean_fqn_to_curr_fqn[clean_tensor_name(key)] = key

    param_key_to_param: dict[str | int, nn.Parameter] = {}
    pid = 0
    for param_group in optim.param_groups:
        if is_named_optimizer:
            for param in param_group["params"]:
                if flat_param_to_fqn is None:
                    raise AssertionError(
                        f"Expected flat_param_to_fqn to be not None, got {flat_param_to_fqn}"
                    )
                if param in flat_param_to_fqn:
                    # FlatParameter case
                    key = flat_param_to_fqn[param]
                else:
                    if param_to_fqns is None:
                        raise AssertionError(
                            f"Expected param_to_fqns to be not None, got {param_to_fqns}"
                        )
                    # use_orig_params case
                    if len(param_to_fqns[param]) != 1:
                        raise AssertionError(
                            f"Expected len(param_to_fqns[param]) == 1, got {len(param_to_fqns[param])}"
                        )
                    key = param_to_fqns[param][0]
                try:
                    key = clean_fqn_to_curr_fqn[key]
                except KeyError as e:
                    raise KeyError(
                        f"Can't find {key} from {list(clean_fqn_to_curr_fqn.keys())}."
                    ) from e
                param_key_to_param[key] = param
        else:
            for param in param_group["params"]:
                param_key_to_param[pid] = param
                pid += 1

    return param_key_to_param


def _get_param_to_param_key(
    optim: torch.optim.Optimizer,
    model: nn.Module | None = None,
    is_named_optimizer: bool = False,
    param_to_fqns: dict[nn.Parameter, list[str]] | None = None,
    flat_param_to_fqn: dict[FlatParameter, str] | None = None,
) -> dict[nn.Parameter, int | str]:
    """
    Constructs the inverse mapping of :func:`_get_param_key_to_param`. This API
    only supports the case where `optim` is a regular optimizer, not NamedOptimizer.
    So the parameter keys will be parameter ids.
    """
    param_id_to_param = _get_param_key_to_param(
        optim, model, is_named_optimizer, param_to_fqns, flat_param_to_fqn
    )
    return {param: param_id for param_id, param in param_id_to_param.items()}


def _get_param_to_param_id_from_optim_input(
    model: nn.Module,
    optim_input: list[dict[str, Any]] | Iterable[nn.Parameter] | None = None,
) -> dict[nn.Parameter, int]:
    """Constructs the inverse mapping of :func:`_get_param_id_to_param_from_optim_input`."""
    param_id_to_param = _get_param_id_to_param_from_optim_input(model, optim_input)
    return {param: param_id for param_id, param in param_id_to_param.items()}


def _check_missing_keys_on_rank(
    r0_optim_state_keys: list[_OptimStateKey],
    optim_state_key_to_param_key: dict[_OptimStateKey, str | int],
    param_key_to_param: dict[str | int, nn.Parameter],
    group: dist.ProcessGroup | None,
) -> None:
    # Ensure that all ranks have at least the optimizer states needed by
    # rank 0's optimizer
    missing_keys: list[_OptimStateKey] = []
    for r0_optim_state_key in r0_optim_state_keys:
        if r0_optim_state_key not in optim_state_key_to_param_key:
            # A parameter from rank 0's optimizer does not exist for this
            # rank's optimizer
            missing_keys.append(r0_optim_state_key)
            continue
        param_key = optim_state_key_to_param_key[r0_optim_state_key]
        if isinstance(param_key, int):
            if not (param_key >= 0 and param_key < len(param_key_to_param)):
                raise AssertionError("Check the `param_key_to_param` construction")
    # We cannot use FSDPState.compute_device as this API is a global view.
    device = _get_pg_default_device(group)
    num_missing = torch.tensor([len(missing_keys)], dtype=torch.int32, device=device)
    dist.all_reduce(num_missing, group=group)
    if num_missing.item() > 0:
        obj_list = [None for _ in range(dist.get_world_size(group))]
        dist.all_gather_object(obj_list, missing_keys, group=group)
        error_msg = (
            "FSDP currently requires each rank to have at least the "
            "optimizer states needed by rank 0's optimizer but some ranks "
            "are missing some of those states"
        )
        for rank, keys in enumerate(obj_list):
            keys = cast(list[_OptimStateKey], keys)
            if len(keys) > 0:
                error_msg += (
                    f"\nRank {rank} is missing states for the parameters: "
                    f"{[key.unflat_param_names for key in keys]}"
                )
        raise RuntimeError(error_msg)


def _map_param_key_to_optim_keys(
    optim_state_dict: dict[str, Any],
    group: dist.ProcessGroup | None,
    param_key_to_param: dict[int | str, nn.Parameter],
    param_to_fqns: dict[nn.Parameter, list[str]],
    fqn_to_fsdp_param_info: dict[str, FSDPParamInfo],
    merge_keys: bool = False,
) -> tuple[list[_OptimStateKey], dict[_OptimStateKey, int | str]]:
    """
    Construct the local mapping between the ``_OptimStateKey`` and parameter keys
    and all the ``_OptimStateKey`` across ranks. If ``merge_keys`` is False, rank0
    must contain all the ``_OptimStateKey``, an exception will be raised otherwise.
    Note that ``merge_keys`` should equal to ``use_orig_params``.
    """
    rank = dist.get_rank(group)
    optim_state_key_to_param_key: dict[_OptimStateKey, int | str] = {}  # local
    all_optim_state_keys: list[_OptimStateKey] = []

    for param_key, param in param_key_to_param.items():
        # Do not include parameters without state to avoid empty mappings
        # just like in normal `torch.optim.Optimizer.state_dict()`
        if param_key not in optim_state_dict["state"]:
            continue
        fqns = param_to_fqns[param]
        is_fsdp_managed = isinstance(param, FlatParameter)
        if is_fsdp_managed:
            if fqns[0] not in fqn_to_fsdp_param_info:
                raise AssertionError(
                    f"Expected {fqns[0]} to be in fqn_to_fsdp_param_info, got keys: {list(fqn_to_fsdp_param_info.keys())}"
                )
        is_fsdp_managed = fqns[0] in fqn_to_fsdp_param_info
        optim_state_key = _OptimStateKey(
            unflat_param_names=tuple(fqns),
            is_fsdp_managed=is_fsdp_managed,
        )
        if rank == 0 or merge_keys:
            all_optim_state_keys.append(optim_state_key)
        optim_state_key_to_param_key[optim_state_key] = param_key

    if merge_keys:
        all_keys: list[list[_OptimStateKey]] = [
            [] for _ in range(dist.get_world_size(group))
        ]
        dist.all_gather_object(all_keys, all_optim_state_keys, group=group)
        merge_all_optim_state_keys = [*chain.from_iterable(all_keys)]
        all_optim_state_keys = sorted(set(merge_all_optim_state_keys))
    else:
        key_obj_list: list[list[_OptimStateKey] | None] = (
            [all_optim_state_keys] if rank == 0 else [None]
        )
        dist.broadcast_object_list(key_obj_list, src=0, group=group)
        if key_obj_list[0] is None:
            raise AssertionError(
                f"Expected key_obj_list[0] to be not None, got {key_obj_list[0]}"
            )
        all_optim_state_keys = key_obj_list[0]
        _check_missing_keys_on_rank(
            all_optim_state_keys,
            optim_state_key_to_param_key,
            param_key_to_param,
            group,
        )

    return all_optim_state_keys, optim_state_key_to_param_key


def _unflatten_param_groups(
    state_dict: dict[str, Any],
    param_key_to_param: dict[int | str, nn.Parameter],
    param_to_fqns: dict[nn.Parameter, list[str]],
) -> list[dict[str, Any]]:
    param_groups: list[dict[str, Any]] = []
    for flat_param_group in state_dict["param_groups"]:
        unflat_param_group = copy.deepcopy(flat_param_group)
        param_group_params = [
            param_key_to_param[flat_param_key]
            for flat_param_key in flat_param_group["params"]
        ]
        nested_unflat_param_names = [
            param_to_fqns[param] for param in param_group_params
        ]
        unflat_param_group["params"] = [
            *chain.from_iterable(nested_unflat_param_names)
        ]  # flatten the list of lists
        param_groups.append(unflat_param_group)
    return param_groups


def _is_named_optimizer(optim_state_dict: dict[str, Any]) -> bool:
    """
    Returns whether the state_dict is from a NamedOptimizer.
    This function checks that the keys in the state_dict['state'] are strings
    (which usually are FQNs) versus integers (which usually refer to param_ids
    from a vanilla torch.optim.Optimizer).
    """
    state = optim_state_dict.get("state")
    if not state:
        # If we cannot find a state, assume it is not NamedOptimizer as
        # NamedOptimizer has eager initialization.
        return False
    try:
        key = next(iter(state.keys()))
    except Exception as e:
        raise Exception(optim_state_dict) from e  # noqa: TRY002
    return isinstance(key, str)


@dataclass
class StateInfo:
    # The key of these dictionaries are the state name, e.g., `exp_avg`.
    tensors: dict[str, _PosDimTensorInfo]
    scalar_tensors: dict[str, torch.Tensor]
    non_tensors: dict[str, Any]


def _allgather_state_info(
    fsdp_state: _FSDPState,
    input_states: dict[str, Any],
) -> list[dict[str, StateInfo]]:
    """
    Given the ``input_states``, allgather StateInfo for each state. The function
    uses all_gather_object to gather StateInfo so no GPU tensors are sent.
    """

    processed_state_dict: dict[str, StateInfo] = {}
    gathered_state_info: list[dict[str, StateInfo]] = [
        {} for _ in range(fsdp_state.world_size)
    ]

    for fqn, optim_state in input_states.items():
        # Allgather the scalar tensor state, non-tensor states and tensors metadata.
        processed_state = StateInfo({}, {}, {})
        for state_name, value in sorted_items(optim_state):
            if torch.is_tensor(value):
                if value.dim() == 0:
                    # Ensure that `step` is on CPU.
                    processed_state.scalar_tensors[state_name] = value.cpu()
                else:
                    processed_state.tensors[state_name] = _PosDimTensorInfo(
                        value.shape, value.dtype
                    )
            else:
                processed_state.non_tensors[state_name] = value
        processed_state_dict[fqn] = processed_state
    dist.all_gather_object(
        gathered_state_info,
        processed_state_dict,
        group=fsdp_state.process_group,
    )
    return gathered_state_info


def _convert_all_state_info(
    fsdp_param_info: FSDPParamInfo,
    gathered_state_info: list[dict[str, StateInfo]],
    input_states: dict[str, Any],
    output_states: dict[str, dict[str, Any]],
) -> tuple[torch.dtype | None, dict[str, list[torch.Tensor | None]]]:
    """
    Given the ``gathered_state_info`` and ``input_states``, the API converted
    the StateInfo into the original state if the state is not a non-scalar
    tensor. For a multi-dimensional tensor, the local state will be stored in
    ``state_buffer`` in a correct order for later allgather purpose.
    """

    state_buffers: dict[str, list[torch.Tensor | None]] = {}

    for fqn, gathered_state in output_states.items():
        state_info = [s[fqn] for s in gathered_state_info]
        all_tensor_states = sorted({n for state in state_info for n in state.tensors})
        empty_ranks: set[int] = set()
        dtype: torch.dtype | None = None
        # First check all the non-scalar states and get the information of
        # states on each rank.
        for state_name in all_tensor_states:
            numels = []
            _empty_ranks: set[int] = set()
            for rank, object_state in enumerate(state_info):
                numels.append(0)
                info = object_state.tensors.get(state_name, None)
                if info is not None:
                    numels[-1] = info.shape.numel()
                    if not dtype:
                        dtype = info.dtype
                    else:
                        if dtype != info.dtype:
                            raise AssertionError(
                                f"Expected dtype == info.dtype, got {dtype} != {info.dtype}"
                            )
                if numels[-1] == 0:
                    _empty_ranks.add(rank)

            if not (not empty_ranks or empty_ranks == _empty_ranks):
                raise AssertionError(
                    f"Expected empty_ranks to be empty or equal to _empty_ranks, got {empty_ranks} vs {_empty_ranks}"
                )
            empty_ranks = _empty_ranks
            if state_name not in state_buffers:
                state_buffers[state_name] = [
                    None for _ in fsdp_param_info.param_indices
                ]
            local_state = input_states[fqn].get(state_name, None)
            # N.B. We need to move the state to compute_device. The reason is
            # not yet clear and we need to figure out why the state may be on a
            # different device.
            if local_state is not None:
                local_state = local_state.to(fsdp_param_info.state.compute_device)
            state_buffers[state_name][fsdp_param_info.param_indices[fqn]] = local_state

        # Restoring the scalar and non-tensor states. If the corresponding
        # non-scalar states do not exist on the rank, we also skip the scalar
        # non-tensor states on that rank.
        for rank, object_state in enumerate(state_info):
            if rank in empty_ranks:
                continue
            for name, non_tensor_value in object_state.non_tensors.items():
                curr_non_tensor_value = gathered_state.get(name, None)
                if not (
                    curr_non_tensor_value is None
                    or curr_non_tensor_value == non_tensor_value
                ):
                    raise AssertionError(
                        f"Rank {rank} has different values for {name}: {non_tensor_value}."
                        + f" Other ranks: {curr_non_tensor_value}"
                    )
                gathered_state[name] = non_tensor_value

            for name, scalar_tensor_value in object_state.scalar_tensors.items():
                curr_scalar_tensor_value = gathered_state.get(name, None)
                if not (
                    curr_scalar_tensor_value is None
                    or torch.equal(scalar_tensor_value, curr_scalar_tensor_value)
                ):
                    raise AssertionError(
                        f"Rank {rank} has different values for {name}: {scalar_tensor_value}."
                        + f" Other ranks: {curr_scalar_tensor_value}"
                    )
                gathered_state[name] = scalar_tensor_value

    return dtype, state_buffers  # type: ignore[possibly-undefined]


def _unflatten_orig_param_states(
    fsdp_param_info: FSDPParamInfo,
    output_states: dict[str, dict[str, Any]],
    state_name: str,
    shard_state: bool,
    to_save: bool,
    cpu_offload: bool,
) -> None:
    """
    Given an output state dict, ``output_states``, which the keys are FQNs to the
    original parameters (not FlatParameters nor parameter ID), and the values
    are gathered states, unflatten the states to the original dimensions.

    This function performs the unflattening process in-place.
    """
    if not to_save:
        return
    flat_param = fsdp_param_info.handle.flat_param
    fsdp_state = fsdp_param_info.state
    for fqn, gathered_state in output_states.items():
        value = gathered_state[state_name]
        param_idx = fsdp_param_info.param_indices[fqn]

        # TODO: This solution is not general and only apply to PTD TP solution.
        if isinstance(value, DTensor):
            placement = value.placements[0]
            # If gathered state is a DTensor and its TP placement is not Replicate(), we need to
            # gather the tensor on its TP dimension before chunking them into DTensor again.
            if placement != Replicate():
                placement_dim = placement.dim  # type: ignore[attr-defined]
                value.redistribute(placements=(Replicate(),))
                reshape_size = list(flat_param._shapes[param_idx])
                reshape_size[placement_dim] *= value.device_mesh.size(0)
                reshape_size = torch.Size(reshape_size)
                value = value.reshape(reshape_size)
            # If gathered state is a replicate DTensor, we directly reshape it.
            else:
                value = value.reshape(flat_param._shapes[param_idx])
        else:
            # If gathered state is a tensor, we directly reshape it into unflatten state.
            value = value.reshape(flat_param._shapes[param_idx])

        if shard_state:
            osd_config = fsdp_state._optim_state_dict_config
            if getattr(osd_config, "_use_dtensor", False):
                if fsdp_state._device_mesh is None:
                    raise AssertionError(
                        f"Expected _device_mesh to be not None, got {fsdp_state._device_mesh}"
                    )
                value = _ext_chunk_dtensor(
                    value,
                    fsdp_state.rank,
                    fsdp_state._device_mesh,
                    fsdp_state._fsdp_extension,
                )
            else:
                if fsdp_state.process_group is None:
                    raise AssertionError(
                        f"Expected process_group to be not None, got {fsdp_state.process_group}"
                    )
                value = _ext_chunk_tensor(
                    value,
                    fsdp_state.rank,
                    fsdp_state.world_size,
                    fsdp_state._device_handle.device_count(),
                    fsdp_state.process_group,
                    fsdp_state._fsdp_extension,
                )
        elif not cpu_offload:
            with SimpleProfiler.profile("clone"):
                value = value.detach().clone()

        if cpu_offload:
            with SimpleProfiler.profile(SimpleProfiler.Type.D2H):
                value = value.cpu()
        gathered_state[state_name] = value


def _allgather_orig_param_states(
    fsdp_param_info: FSDPParamInfo,
    gathered_state_info: list[dict[str, StateInfo]],
    input_states: dict[str, Any],
    shard_state: bool,
    to_save: bool,
    cpu_offload: bool,
) -> dict[str, dict[str, Any]]:
    """
    Given the ``gathered_state_info`` and ``input_states``, the API allgathers
    all tensor states and restore non-tensor states from ``gathered_state_info``.
    """
    fsdp_state = fsdp_param_info.state
    if fsdp_state.rank == 0 and dist.get_debug_level() == dist.DebugLevel.DETAIL:
        logger.info(
            "Memory Summary before calling to _allgather_orig_param_states %s",
            fsdp_state._device_handle.memory_summary(),
        )

    output_states: dict[str, dict[str, Any]] = {fqn: {} for fqn in input_states}

    dtype, state_buffers = _convert_all_state_info(
        fsdp_param_info, gathered_state_info, input_states, output_states
    )

    if len(state_buffers) == 0:
        return output_states

    has_state_params: list[bool] = [
        fqn in output_states for fqn in fsdp_param_info.param_indices
    ]

    # Loop through the ``state_buffers`` and construct the flattened, concatenated,
    # sharded states. The size of the constructed state will be the same size as
    # flat_param (also sharded).
    # Then we perform an allgather_into_tensor to get the full flat_param state.
    # The full flat_param state is the result of concatenation of multiple states
    # the order of flat_param._fqns.
    # The final step is to split the flat_param state into original param states
    # and return the result.
    flat_param = fsdp_param_info.handle.flat_param
    empty_func = functools.partial(
        torch.empty, dtype=dtype, device=fsdp_state.compute_device
    )
    gathered_tensor = empty_func(flat_param._padded_unsharded_size)
    # Synchronize can be slow but this will be easier for us to debug.
    fsdp_state._device_handle.synchronize()
    for state_name, buffers in state_buffers.items():
        local_buffers: list[torch.Tensor] = []
        begin = fsdp_state.rank * flat_param._sharded_size.numel()
        # End is inclusive.
        end = begin + flat_param._sharded_size.numel() - 1
        # param_idx corresponds to the parameter index in the FlatParameter.
        mem_offset, param_idx = 0, 0
        for numel, is_padding in zip(
            flat_param._numels_with_padding, flat_param._is_padding_mask
        ):
            frozen_and_no_state = not is_padding and (
                not fsdp_param_info.param_requires_grad[param_idx]
                and not has_state_params[param_idx]
            )

            if is_padding or frozen_and_no_state:
                # This memory range is a padding or the param is frozen and does
                # not require gradient. For the later case, we treat it as a
                # padding and add empty values to the local_buffers.

                padding_begin, padding_end = mem_offset, mem_offset + numel - 1
                if padding_begin <= begin <= padding_end:
                    # The range is an align padding before the first parameter in
                    # the shard. The shard includes parts of this align padding.
                    padding_len = (
                        padding_end - begin + 1
                        if end >= padding_end
                        else end - begin + 1
                    )
                elif padding_begin <= end <= padding_end:
                    # The range is an align padding after the last parameter in
                    # the shard. The shard includes parts of this align padding.
                    padding_len = (
                        end - padding_begin + 1
                        if begin <= padding_begin
                        else end - begin + 1
                    )
                elif begin < padding_begin <= padding_end < end:
                    # The range is an align padding that is completely in the
                    # shard.
                    padding_len = numel
                else:
                    padding_len = 0
                if padding_len:
                    local_buffers.append(empty_func(padding_len))

            if not is_padding:
                # This memory range is a parameter in FlatParameter. So there
                # should be an corresponding state in the optimizer unless the
                # parameter is frozen, which we treat it as a padding above.

                # We need to check if this rank owns the buffer. If this is None:
                # 1.) the rank does not own any part of the original parameter.
                #     As a result, there is no corresponding optimizer state on
                #     the rank as well.
                # 2.) the parameter is frozen AND no optimizer state for the
                #     parameter. If a parameter is frozen, there can still be
                #     optimizer state if the parameter is not frozen in the
                #     previous steps.
                if buffers[param_idx] is not None:
                    local_buffers.append(cast(torch.Tensor, buffers[param_idx]))
                param_idx += 1

            mem_offset += numel

        shard_numel_padded = flat_param._sharded_size.numel() - (
            sum(t.numel() for t in local_buffers)
        )

        if flat_param._shard_numel_padded != shard_numel_padded:
            raise AssertionError(
                "Manually calculated _sharded_numel_padded is incorrect. "
                f"_shard_numel_padded={flat_param._shard_numel_padded}, "
                f"shard_numel_padded={shard_numel_padded}, "
                f"_sharded_size.numel={flat_param._sharded_size.numel()}, "
                f"_numels_with_padding={flat_param._numels_with_padding}, "
                f"begin={begin}, end={end},"
            )
        if shard_numel_padded > 0:
            # Add right-handed padding.
            local_buffers.append(empty_func(shard_numel_padded))
        local_shard = torch.cat(local_buffers)
        if local_shard.numel() * fsdp_state.world_size != gathered_tensor.numel():
            raise AssertionError(
                "The size of local shard times the world size should equal to the "
                "gathered tensor size. The inconsistency may be from a bug of "
                "FlatParameter's metadata or the reconstruction logic in optimizer "
                "state dict."
            )
        fsdp_state._device_handle.synchronize()
        with SimpleProfiler.profile(SimpleProfiler.Type.ALLGATHER):
            dist.all_gather_single(
                gathered_tensor, local_shard, group=fsdp_state.process_group
            )
            # Synchronize can be slow but this will be easier for us to debug.
            fsdp_state._device_handle.synchronize()

        unpadded_tensor = gathered_tensor[: flat_param._unpadded_unsharded_size.numel()]
        flat_param_handle = fsdp_param_info.handle
        orig_states = flat_param_handle._get_unflat_views_aligned(unpadded_tensor)
        if len(orig_states) != len(fsdp_param_info.param_indices):
            raise AssertionError(
                "The number of parameters from FlatParameter is not consistent to "
                "the number of states used by optimizer state dict reconstruction "
                "logic."
            )
        for fqn, idx in fsdp_param_info.param_indices.items():
            if fsdp_param_info.param_requires_grad[idx] or fqn in output_states:
                output_states[fqn][state_name] = orig_states[idx]

        _unflatten_orig_param_states(
            fsdp_param_info,
            output_states,
            state_name,
            shard_state,
            to_save,
            cpu_offload,
        )

    del gathered_tensor
    return output_states


def _gather_all_orig_param_state(
    fsdp_param_info: FSDPParamInfo,
    input_states: dict[str, Any],
    shard_state: bool,
    to_save: bool,
    cpu_offload: bool,
) -> dict[str, Any]:
    """
    Given a optimizer state dict, ``input_states``, which the keys are FQNs to the
    original parameters (not FlatParameters nor parameter ID), gather all the
    states and unflatten them to the original dimensions. Note that all the
    params referred by the ``input_states`` must be managed by FSDP.
    """
    fsdp_state = fsdp_param_info.state
    if (
        fsdp_state.world_size == 1
        or fsdp_state.sharding_strategy == ShardingStrategy.NO_SHARD
    ):
        return input_states if to_save else {}

    with SimpleProfiler.profile(SimpleProfiler.Type.RESHARDING):
        with SimpleProfiler.profile(SimpleProfiler.Type.ALLGATHER_OBJ):
            gathered_state_info = _allgather_state_info(fsdp_state, input_states)
        output_states = _allgather_orig_param_states(
            fsdp_param_info,
            gathered_state_info,
            input_states,
            shard_state,
            to_save,
            cpu_offload,
        )
    if to_save:
        for key, idx in fsdp_param_info.param_indices.items():
            if key in output_states:
                continue
            if not fsdp_param_info.param_requires_grad[idx]:
                continue

            raise RuntimeError(
                f"{key} is not in the output state. "
                "The FSDPParamInfo has the param keys "
                f"{sorted(fsdp_param_info.param_indices.keys())} while "
                "the output_states has the param keys "
                f"{sorted(output_states.keys())}."
            )
        return output_states
    else:
        return {}


def _convert_state_with_orig_params(
    all_optim_state_keys: list[_OptimStateKey],
    optim_state_key_to_param_key: dict[_OptimStateKey, int | str],
    fqn_to_fsdp_param_info: dict[str, FSDPParamInfo],
    optim_state_dict: dict[str | int, Any],
    to_save: bool,
    shard_state: bool,
    cpu_offload: bool = True,
) -> dict[str, Any]:
    fsdp_osd_state: dict[str, Any] = {}
    # This variable is used to deduplicate the FSDPParamInfo as one FSDPParamInfo
    # usually corresponds to multiple parameters. We could not use FSDPParamInfo
    # as the key because FSDPParamInfo is not hashable. As a result, we fall back
    # to `id(FSDPParamInfo)`, which the type is an integer.
    all_states: dict[int, dict[str, Any]] = {}
    # Iterate in rank 0's flat parameter ID order to ensure aligned all-gathers
    # across ranks
    for optim_state_key in all_optim_state_keys:
        param_key: str | int | None = optim_state_key_to_param_key.get(optim_state_key)

        if param_key is None and not optim_state_key.is_fsdp_managed:
            continue

        if optim_state_key.is_fsdp_managed:
            fqn = optim_state_key.unflat_param_names[0]
            fsdp_param_info = fqn_to_fsdp_param_info.get(fqn)
            if fsdp_param_info is None:
                # This can happen if the not all FSDP instances have all the
                # parameters. This can happen with FSDP + some MPMD style
                # parallelism.

                # TODO: it is unclear if we need to do the same check with
                # non-FSDP managed keys.
                continue
            state = {} if param_key is None else optim_state_dict[param_key]
            if id(fsdp_param_info) not in all_states:
                all_states[id(fsdp_param_info)] = {}
            all_states[id(fsdp_param_info)][fqn] = state

        elif to_save:
            if len(optim_state_key.unflat_param_names) != 1:
                raise AssertionError(
                    f"Expected len(optim_state_key.unflat_param_names) == 1, got {len(optim_state_key.unflat_param_names)}"
                )
            unflat_param_name = optim_state_key.unflat_param_names[0]
            with SimpleProfiler.profile("none_fsdp_managed_copy"):
                param_key = cast(str | int, param_key)
                fsdp_osd_state[unflat_param_name] = copy.copy(
                    optim_state_dict[param_key]
                )
                if cpu_offload:
                    for state_name, value in sorted_items(
                        fsdp_osd_state[unflat_param_name]
                    ):
                        if not torch.is_tensor(value):
                            continue
                        fsdp_osd_state[unflat_param_name][state_name] = value.cpu()

    # Instead of gathering the state of each parameter individually, we perform
    # the gathering  all at once to speed up the process.
    for _all_states in all_states.values():
        fqn = next(iter(_all_states.keys()))
        fsdp_param_info = fqn_to_fsdp_param_info[fqn]
        if len(fsdp_param_info.param_requires_grad) <= 0:
            raise AssertionError(
                "With use_orig_params, FSDPParamInfo should have requires_grad "
                "information. However, the length is zero."
            )
        for key, idx in fsdp_param_info.param_indices.items():
            if key in _all_states:
                continue
            if not fsdp_param_info.param_requires_grad[idx]:
                continue
            raise RuntimeError(
                f"{key} is not in the optimizer state. "
                "The FSDPParamInfo has the param keys "
                f"{sorted(fsdp_param_info.param_indices.keys())} while "
                "the optimizer has the param keys "
                f"{sorted(_all_states.keys())}."
            )
        fsdp_osd_state.update(
            _gather_all_orig_param_state(
                fsdp_param_info,
                _all_states,
                shard_state,
                to_save,
                cpu_offload,
            )
        )

    return fsdp_osd_state


def _convert_state_with_flat_params(
    all_optim_state_keys: list[_OptimStateKey],
    optim_state_key_to_param_key: dict[_OptimStateKey, int | str],
    fqn_to_fsdp_param_info: dict[str, FSDPParamInfo],
    optim_state_dict: dict[str | int, Any],
    to_save: bool,
    shard_state: bool,
    cpu_offload: bool = True,
) -> dict[str, Any]:
    fsdp_osd_state: dict[str, Any] = {}
    # Iterate in rank 0's flat parameter ID order to ensure aligned all-gathers
    # across ranks
    for optim_state_key in all_optim_state_keys:
        param_key: str | int | None = optim_state_key_to_param_key.get(optim_state_key)

        if param_key is None:
            raise AssertionError(
                "If use_orig_params is False, we must be able to find the "
                f"corresponding param id. {optim_state_key} {param_key}"
            )

        if optim_state_key.is_fsdp_managed:
            # If there are multiple unflat_param_names (not use_orig_params),
            # they share the same FSDPParamInfo. So the first unflat_param_name
            # is sufficient to fetch the FSDPParamInfo.
            fqn = optim_state_key.unflat_param_names[0]
            fsdp_param_info = fqn_to_fsdp_param_info[fqn]
            unflat_state = _unflatten_optim_state(
                fsdp_param_info,
                optim_state_dict[param_key],
                to_save,
                shard_state,
                cpu_offload,
            )
            if to_save:
                if len(unflat_state) != len(optim_state_key.unflat_param_names):
                    raise AssertionError(
                        f"Expected len(unflat_state) == len(optim_state_key.unflat_param_names), "
                        f"got {len(unflat_state)} != {len(optim_state_key.unflat_param_names)}"
                    )
                fsdp_osd_state.update(
                    zip(
                        optim_state_key.unflat_param_names,
                        unflat_state,
                    )
                )
        elif to_save:
            if len(optim_state_key.unflat_param_names) != 1:
                raise AssertionError(
                    f"Expected len(optim_state_key.unflat_param_names) == 1, got {len(optim_state_key.unflat_param_names)}"
                )
            unflat_param_name = optim_state_key.unflat_param_names[0]
            fsdp_osd_state[unflat_param_name] = copy.copy(optim_state_dict[param_key])
            if cpu_offload:
                for state_name, value in sorted_items(
                    fsdp_osd_state[unflat_param_name]
                ):
                    if not torch.is_tensor(value):
                        continue
                    fsdp_osd_state[unflat_param_name][state_name] = value.cpu()

    return fsdp_osd_state


@torch.no_grad()
def _optim_state_dict(
    model: nn.Module,
    optim: torch.optim.Optimizer,
    optim_state_dict: dict[str, Any],
    optim_input: list[dict[str, Any]] | Iterable[nn.Parameter] | None,
    rank0_only: bool,
    shard_state: bool,
    group: dist.ProcessGroup | None,
    using_optim_input: bool,
    use_orig_params: bool = False,
    cpu_offload: bool = True,
) -> dict[str, Any]:
    """
    Consolidates the optimizer state and returns it as a :class:`dict`
    following the convention of :meth:`torch.optim.Optimizer.state_dict`,
    i.e. with keys ``"state"`` and ``"param_groups"``.
    The flat parameters in ``FSDP`` modules contained in ``model`` are mapped
    back to their unflattened parameters.

    Parameter keys are not well-defined. For a regular optimizer, the optimizer
    state_dict contains a mapping from parameter IDs to parameter states.
    Parameter IDs are the order of parameters in ``optim.param_groups()`` across
    all the groups. This API also allows user to pass ``optim_input`` for the
    mapping between parameters and parameter IDs. Using ``optim_input`` is being
    deprecated.

    If the optimizer is a ``NamedOptimizer``, the optimizer state_dict does not
    contain parameter IDs mapping but a mapping from parameter FQNs to parameter
    states. This API finds the mapping from FQNs to parameters if the optimizer
    is a ``NamedOptimizer``.

    If ``use_orig_params`` is True, each rank will have all FSDP-managed
    parameters but some of these parameters may be empty due to the sharding.
    For a regular optim.Optimizer, states for those empty parameters will
    not be initialized. So, when aggregating the FQNs across ranks, no assert
    will be raised on a rank even if it does not have all the states -- it is
    valid and FSDP knows how to aggregate them. However, FSDP has to ignore
    handling those parameters that are not managed by FSDP and do not exist on
    the local rank -- those are managed by other parallelisms and FSDP does not
    know how to handle/aggregate them.

    Args:
        model (nn.Module): Root module (which may or may not be a
            :class:`FullyShardedDataParallel` instance) whose parameters
            were passed into the optimizer ``optim``.
        optim (torch.optim.Optimizer): Optimizer for ``model`` 's
            parameters.
        rank0_only (bool): If ``True``, saves the populated :class:`dict`
            only on rank 0; if ``False``, saves it on all ranks. (Default:
            ``True``)
        shard_state (bool): If ``True``, shard and distribute all
            non-zero-dimension states.

    Returns:
        Dict[str, Any]: A :class:`dict` containing the optimizer state for
        ``model`` 's original unflattened parameters and including keys
        "state" and "param_groups" following the convention of
        :meth:`torch.optim.Optimizer.state_dict`. If ``rank0_only=False``,
        then nonzero ranks return an empty :class:`dict`.
    """
    SimpleProfiler.reset()
    cm = ExitStack()
    cm.enter_context(SimpleProfiler.profile(SimpleProfiler.Type.ALL))
    _reset_flat_param_grad_info_if_needed(traversal_utils._get_fsdp_handles(model))
    to_save = not rank0_only or dist.get_rank(group) == 0 or shard_state

    with SimpleProfiler.profile("preprocessing"):
        param_to_fqns = _get_param_to_fqns(model)
        flat_param_to_fqn = _get_flat_param_to_fqn(model)
        is_named_optimizer = _is_named_optimizer(optim_state_dict)

        param_key_to_param = cast(
            dict[int | str, nn.Parameter],
            (
                _get_param_id_to_param_from_optim_input(model, optim_input)
                if using_optim_input
                else _get_param_key_to_param(
                    optim, model, is_named_optimizer, param_to_fqns, flat_param_to_fqn
                )
            ),
        )
        fqn_to_fsdp_param_info = _get_fqn_to_fsdp_param_info(model)

    with SimpleProfiler.profile("preprocessing_with_comm"):
        (
            all_optim_state_keys,
            optim_state_key_to_param_key,
        ) = _map_param_key_to_optim_keys(
            optim_state_dict,
            group,
            param_key_to_param,
            param_to_fqns,
            fqn_to_fsdp_param_info,
            merge_keys=use_orig_params,
        )

    with SimpleProfiler.profile("state_converting"):
        convert_fn = (
            _convert_state_with_orig_params
            if use_orig_params
            else _convert_state_with_flat_params
        )
        fsdp_osd_state = convert_fn(
            all_optim_state_keys,
            optim_state_key_to_param_key,
            fqn_to_fsdp_param_info,
            optim_state_dict["state"],
            to_save,
            shard_state,
            cpu_offload,
        )

    # At this point, communication is complete and ranks can return early if nothing
    # will be saved on that rank.
    if not to_save:
        return {}

    fsdp_osd: dict[str, Any] = {"state": fsdp_osd_state}

    flat_param_fqns = set(flat_param_to_fqn.values())
    for key, value in optim_state_dict["state"].items():
        if key in fsdp_osd_state:
            continue
        if key in flat_param_fqns:
            continue
        if key in param_key_to_param:
            continue
        # This key is not recognized by FSDP. It may be a user-defined state
        # or some parameters state that FSDP is unable to map from
        # ``optim.param_groups``.
        warnings.warn(
            f"Found a optim state, {key}, that FSDP cannot process. FSDP "
            "will directly copy everything to the returned state_dict. In "
            "most cases, this is a user-defined state that is not "
            "associated with any particular parameter. Another possible "
            "case is this state is managed by TorchRec. Otherwise, there may "
            " be a mismatched assumption of optim_state_dict of this mode.",
            stacklevel=2,
        )
        fsdp_osd_state[key] = value

    if "param_groups" in optim_state_dict:
        fsdp_osd["param_groups"] = _unflatten_param_groups(
            optim_state_dict, param_key_to_param, param_to_fqns
        )

    cm.close()
    SimpleProfiler.dump_and_reset("FSDP _optim_state_dict() profiling: ")

    return fsdp_osd


def _get_fqn_to_fsdp_param_info(model: nn.Module) -> dict[str, FSDPParamInfo]:
    """
    Construct the mapping from a param's fqn to its corresponding ``FSDPParamInfo``
    if the param is managed by FSDP. Shared parameters, or original parameters that
    are shared across multiple nn.Modules, are required to belong to one and only
    one FSDP instance and thus correspond to one ``FlatParameter``. Within the one
    ``FlatParameter``, ``FlatParameter._fqns`` only stores the first FQN of a shared
    parameter. Thus, the keys in the mapping are guaranteed to map to unique parameters.
    """

    def module_fn(module, prefix, tree_level, fqn_to_param_info):
        fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(module)
        if fsdp_state is None:
            return
        _lazy_init(fsdp_state, module)
        handle = _module_handle(fsdp_state, module)
        if not handle:
            return
        flat_param = handle.flat_param
        fsdp_param_info = FSDPParamInfo(fsdp_state, handle, {}, [])
        # NOTE: `idx` indexes into the data structures *without* padding
        # elements
        for idx, local_fqn in enumerate(flat_param._fqns):
            fqn = clean_tensor_name(prefix + local_fqn)
            if fqn in fqn_to_param_info:
                if fqn_to_param_info[fqn].handle.flat_param is not flat_param:
                    raise AssertionError(
                        f"Expected fqn_to_param_info[fqn].handle.flat_param is flat_param for {fqn}"
                    )
            fqn_to_param_info[fqn] = fsdp_param_info
            fsdp_param_info.param_indices[fqn] = idx
            if flat_param._params is not None:
                fsdp_param_info.param_requires_grad.append(
                    flat_param._params[idx].requires_grad
                )

    def return_fn(fqn_to_param_info):
        return fqn_to_param_info

    fqn_to_param_info: dict[str, FSDPParamInfo] = {}
    # FlatParameter._fqns stores the local fqn, starting from the root of the
    # FSDP. Using _apply_to_modules() with model (may not be the FSDP root
    # module) allows us to construct the global fqn.
    return _apply_to_modules(
        model,
        module_fn,
        return_fn,
        [fqn for fqn, _ in _named_parameters_with_duplicates(model)],
        fqn_to_param_info,
    )


@no_type_check
def _set_optim_use_dtensor(
    fsdp_state: _FSDPState,
    state_dict_settings: StateDictSettings,
) -> None:
    # If device_mesh is passed in when initializing FSDP, we automatically turn the
    # _use_dtensor flag to be true for ShardedOptimStateDictConfig() if state_dict_type
    # has to be set to SHARDED_STATE_DICT.
    if getattr(fsdp_state, "_device_mesh", None):
        state_dict_type = state_dict_settings.state_dict_type
        if state_dict_type == StateDictType.LOCAL_STATE_DICT:
            raise RuntimeError(
                "Found state_dict_type LOCAL_STATE_DICT. "
                "DeviceMesh is not compatible with LOCAL_STATE_DICT. "
                "Please set state_dict_type to SHARDED_STATE_DICT to get DTensor state_dict."
            )
        else:
            state_dict_settings.optim_state_dict_config._use_dtensor = True
<|endoftext|># mypy: allow-untyped-defs
import itertools
import warnings
from enum import auto, Enum

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed.fsdp._common_utils import _FSDPState, _get_param_to_fqns
from torch.distributed.fsdp._flat_param import FlatParamHandle


class _ExecOrderWarnStatus(Enum):
    """Used internally for execution order validation."""

    NONE = auto()  # no deviation yet
    WARNING = auto()  # deviated this iteration; currently issuing warnings
    WARNED = auto()  # deviated in a previous iteration


class _ExecOrderData:
    """
    This contains the data structures to track the execution order. We track
    the pre-forward order on the *first* iteration for forward prefetching
    (which thus assumes static graph) and the post-forward order on *every*
    iteration for backward prefetching (which thus does not assume static
    graph but may be provide an incorrect order).
    """

    def __init__(
        self,
        debug_level: dist.DebugLevel,
        backward_prefetch_limit: int,
        forward_prefetch_limit: int,
    ) -> None:
        # Tracks the (static) pre-forward order for execution order validation
        # and forward prefetching
        self.handles_pre_forward_order: list[FlatParamHandle] = []
        # Tracks the post-forward order for pre-backward prefetching
        self.handles_post_forward_order: list[FlatParamHandle | None] = []
        self._iter = 0

        # Gives the max number of backward/forward prefetched all-gathers by a
        # single module
        self._backward_prefetch_limit = backward_prefetch_limit
        self._forward_prefetch_limit = forward_prefetch_limit

        # Data structures for execution order validation
        self._checking_order: bool = debug_level == dist.DebugLevel.DETAIL
        self.process_group: dist.ProcessGroup | None = None
        self.world_size: int | None = None
        self.all_handles: list[FlatParamHandle] = []
        # Names are prefixed from the root module
        self.param_to_fqn: dict[nn.Parameter, list[str]] = {}
        # Current index in the pre-forward execution order
        self.current_order_index = 0
        self.warn_status = _ExecOrderWarnStatus.NONE

    def init(
        self,
        state: _FSDPState,
        root_module: nn.Module,
        process_group: dist.ProcessGroup,
    ) -> None:
        """
        Initializes the data structures needed for checking the forward order.
        This should be called after a root FSDP instance has been set during
        lazy initialization.
        """
        self.process_group = process_group
        self.rank = process_group.rank()
        self.world_size = process_group.size()
        # Fix an order over the handles, which should be the same across ranks
        for handle in traversal_utils._get_fsdp_handles(root_module):
            index = len(self.all_handles)
            self.all_handles.append(handle)
            handle._handle_index = index
        self.param_to_fqn = _get_param_to_fqns(root_module)
        # TODO (awgu): We can broadcast the metadata of rank 0's `all_handles`
        # to check that all ranks have the same handles in the same order.
        # https://github.com/pytorch/pytorch/issues/79620

    @property
    def is_first_iter(self) -> bool:
        return self._iter == 0

    def get_handle_to_backward_prefetch(
        self,
        current_handle: FlatParamHandle,
    ) -> FlatParamHandle | None:
        """
        Returns a :class:`list` of the handles keys of the handles to backward
        prefetch given the current handles key. If there are no valid handles
        keys to prefetch, then this returns an empty :class:`list`.
        """
        current_index = current_handle._post_forward_index
        if current_index is None:
            return None
        target_index = current_index - 1
        target_handle: FlatParamHandle | None = None
        for _ in range(self._backward_prefetch_limit):
            if target_index < 0:
                break
            target_handle = self.handles_post_forward_order[target_index]
            target_index -= 1
        return target_handle

    def get_handle_to_forward_prefetch(
        self,
        current_handle: FlatParamHandle,
    ) -> FlatParamHandle | None:
        """
        Returns a :class:`list` of the handles keys of the handles to forward
        prefetch given the current handles key. If there are no valid handles
        keys to prefetch, then this returns an empty :class:`list`.
        """
        current_index = current_handle._pre_forward_order_index
        if current_index is None:
            return None
        target_index = current_index + 1
        target_handle: FlatParamHandle | None = None
        for _ in range(self._forward_prefetch_limit):
            if target_index >= len(self.handles_pre_forward_order):
                break
            target_handle = self.handles_pre_forward_order[target_index]
            target_index += 1
        return target_handle

    def record_post_forward(self, handle: FlatParamHandle | None) -> None:
        """
        Records ``handles`` in the post-forward order, where ``handles`` should
        be a group of handles used in the same module's forward. If ``handles``
        is empty, then it is omitted.

        Unlike :meth:`record_pre_forward`, this records the order *every*
        iteration with the expectation that the recorded order is reset in
        :meth:`next_iter`.
        """
        if not handle:
            return
        # Only record the first usage of a handles key
        if handle._post_forward_index:
            self.handles_post_forward_order.append(handle)
            return
        index = len(self.handles_post_forward_order)
        handle._post_forward_index = index
        self.handles_post_forward_order.append(handle)

    def record_pre_forward(
        self, handle: FlatParamHandle | None, is_training: bool
    ) -> None:
        """
        Records ``handles`` in the pre-forward order, where ``handles`` should
        be a group of handles used in the same module's forward. If ``handles``
        is empty, then it is omitted.

        On the first iteration, this checks the execution order across ranks.
        See :meth:`_check_order` for details.
        """
        if not handle:
            return
        self._check_order(handle, is_training)
        # Fix the order after the first iteration and only record the first
        # usage of a handles key
        if not self.is_first_iter or handle._pre_forward_order_index is not None:
            return
        index = len(self.handles_pre_forward_order)
        handle._pre_forward_order_index = index
        self.handles_pre_forward_order.append(handle)

    def _check_order(self, handle: FlatParamHandle, is_training: bool) -> None:
        """
        Checks the forward execution order as long as ``is_training`` is
        ``True`` since checking in eval mode is not supported. This only checks
        if the distributed debug level is DETAIL.

        - On the first iteration, this uses all-gathers to check that all ranks
        are all-gathering the same handles and hence ``FlatParameter`` s,
        raising an error if not.
        - On subsequent iterations, this checks that each rank is locally
        consistent with its own forward order from the first iteration, issuing
        a warning if not. This issues a warning on the first deviating
        iteration and stops warning thereafter.
        """
        # Do not check order in eval mode since the post-backward callback does
        # not run so it cannot be used to mark the end of an iteration
        if not is_training or not self._checking_order:
            return
        if self.is_first_iter:
            msg_prefix = "Forward order differs across ranks:"
            optional_local_indices: tuple[int | None, ...] = self._get_handle_indices(
                handle
            )
            device = handle.device  # guaranteed to be non-CPU
            num_valid_indices = sum(
                (index is not None) for index in optional_local_indices
            )
            tensor_kwargs: dict[str, torch.dtype | torch.device] = {
                "dtype": torch.int32,
                "device": device,
            }
            world_num_valid_indices = torch.zeros(self.world_size, **tensor_kwargs)  # type: ignore[arg-type, call-overload]
            local_num_valid_indices = torch.tensor([num_valid_indices], **tensor_kwargs)  # type: ignore[arg-type, call-overload]
            dist.all_gather_single(
                world_num_valid_indices,
                local_num_valid_indices,
                group=self.process_group,
            )
            # Copy entire tensor from D2H once to avoid per element D2H copies
            world_num_valid_indices = world_num_valid_indices.cpu()
            # Check that all ranks plan to all-gather the same number of
            # parameters
            # TODO (awgu): Since every module has at most one handle in the
            # current implementation, this should never raise the error.
            if self.world_size is None:
                raise AssertionError("Expected world_size to not be None")
            if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
                # TODO(voz): Don't graph break on this - dynamo hates the n1 != n2
                # tensor comparison control flow.
                # https://github.com/pytorch/pytorch/issues/107055
                for (r1, n1), (r2, n2) in itertools.combinations(
                    (
                        (rank, world_num_valid_indices[rank])
                        for rank in range(self.world_size)
                    ),
                    2,
                ):
                    if n1 != n2:
                        raise RuntimeError(
                            f"{msg_prefix} rank {r1} is all-gathering {n1} parameters "
                            f"while rank {r2} is all-gathering {n2} parameters"
                        )
            world_indices = torch.zeros(  # type: ignore[call-overload]
                self.world_size * num_valid_indices, **tensor_kwargs
            )
            local_indices = torch.tensor(optional_local_indices, **tensor_kwargs)  # type: ignore[arg-type]
            dist.all_gather_single(
                world_indices, local_indices, group=self.process_group
            )
            # Copy entire tensor from D2H once to avoid per element D2H copies
            world_indices = world_indices.cpu()
            # Check that all ranks plan to all-gather the same index parameters
