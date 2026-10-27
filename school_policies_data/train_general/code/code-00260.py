    ACT = "Activation"
    TEMP = "Temp"
    ALL_GATHER = "All Gather"
    REDUCE_SCATTER = "Reduce Scatter"
    OPT = "OptState"
    INP = "Inputs"


class _SavedFSDPMethods(NamedTuple):
    pre_backward: Callable
    post_backward: Callable


class _FSDPModState(_State):
    """
    Enumerates the states of FSDP modules during the forward and backward passes.
    """

    BEF_PRE_FW = "Before Pre-Forward"
    AFT_PRE_FW = "After Pre-Forward"
    BEF_POST_FW = "Before Post-Forward"
    AFT_POST_FW = "After Post-Forward"
    BEF_PRE_BW = "Before Pre-Backward"
    AFT_PRE_BW = "After Pre-Backward"
    BEF_POST_BW = "Before Post-Backward"
    AFT_POST_BW = "After Post-Backward"
    PRE_FW_AC = "Pre-Forward AC"
    POST_FW_AC = "Post-Forward AC"
    PEAK_FW = "Peak Forward"
    PEAK_BW = "Peak Backward"


class _FSDPModMemStats:
    """
    A class to store the memory statistics of an FSDP module.

    Args:
        mod_fqn (str): The fully qualified name of the FSDP module.

    Attributes:
        snapshots (Dict[_FSDPModState, Dict[torch.device, Dict[str, int]]]): A dictionary of memory snapshots
        of the module at different states as defined by ``_FSDPModState``. Each key is a device, and
        each value is another dictionary with keys as memory reference types defined by ``_FSDPRefType`` and
        values as the memory consumed in bytes.

    """

    def __init__(self, mod_fqn: str) -> None:
        self.mod_fqn = mod_fqn
        self.local_peak: dict[torch.device, int] = {}
        self.snapshots: dict[
            _FSDPModState, list[dict[torch.device, dict[str, int]]]
        ] = {}


class _FSDPState(Enum):
    PRE_FW = auto()
    FW = auto()
    POST_FW = auto()
    PRE_BW = auto()
    BW = auto()
    POST_BW = auto()


class FSDPMemTracker(MemTracker):
    """
    A ``TorchDispatchMode`` based context manager that extends ``torch.distributed._tools.mem_tracker.MemTracker`` to track
    and categorize the peak memory and module-wise memory usage of FSDP modules.

    It tracks the peak memory usage across all the devices of all the FSDP modules in the module tree and categorizes
    the tensor memory usage as defined by ``_FSDPRefType``. Further, it captures memory `snapshots` at different stages of
    the module execution defined by ``_FSDPModState``.

    Attributes:
        memory_tracking: A weakref key dictionary to store the memory statistics of each module. Each key is a reference
        to a module, and each value is a ``_FSDPModMemStats`` object that stores the memory statistics of the module.

    Args:
        mod (torch.nn.Module): The root FSDP module to be tracked.
        optm (torch.optim.Optimizer, optional): The optimizer to be tracked.

    Note: Please refer to ``torch.distributed._tools.mem_tracker.MemTracker`` to learn about the limitations.

    Example usage

    .. code-block:: python

        module = ...
        optimizer = ...
        inp = ...
        fmt = FSDPMemTracker(module, optimizer)
        fmt.track_inputs((inp,))
        with fmt:
            optimizer.zero_grad()
            loss = module(inp)
            print("After Forward:")
            fmt.display_snapshot("current")
            loss.backward()
            optimizer.step()
        fmt.display_snapshot("peak")
        fmt.display_modulewise_snapshots(depth=3, units="MB")

    """

    def __init__(
        self,
        mod: torch.nn.Module,
        optm: torch.optim.Optimizer | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(mod, FSDPModule):
            raise AssertionError("FSDPMemTracker only supports FSDP modules")
        self._root_mod = mod
        self._optm = optm
        self._fsdp_mod_to_saved_methods: WeakIdKeyDictionary = WeakIdKeyDictionary()
        self._fsdp_state: _FSDPState = _FSDPState.PRE_FW
        self._ref_class: type[_RefType] = _FSDPRefType

    def _instrument_fsdp_sharded_params_grads(
        self, fsdp_param_group: FSDPParamGroup
    ) -> None:
        # Track sharded params and grads after initialization
        for fsdp_param in fsdp_param_group.fsdp_params:
            self._update_and_maybe_create_winfos(
                fsdp_param.sharded_param,
                _FSDPRefType.SHARDED_PARAM,
            )
            sharded_grad = fsdp_param.sharded_param.grad
            if sharded_grad is not None:
                self._update_and_maybe_create_winfos(
                    sharded_grad,
                    _FSDPRefType.SHARDED_GRAD,
                )

    def _fsdp_state_pre_forward(
        self,
        fsdp_mod: FSDPModule,
        orig_fsdp_state_pre_fw: Callable[_P, tuple[tuple[Unpack[_Ts]], dict[str, Any]]],
    ) -> Callable[_P, tuple[tuple[Unpack[_Ts]], dict[str, Any]]]:
        # We capture memory snapshots before and after ``FSDPState._pre_forward`` to attribute the `unsharded` params
        # and `all_gather` buffers.  There are three cases:
        # Case 1: If the module is not in the ``memory_tracking`` dictionary, create a new ``_FSDPModMemStats``
        #         instance for the module and add it to the ``memory_tracking`` dictionary.
        # Case 2: If the module is already in the ``memory_tracking`` dictionary and we are in backward, this means
        #         we are in the AC region. We check if this is the top most module in the AC region. If it is,
        #         we store a weak reference and set the flag ``_in_ac`` to True.
        # Case 3: If the module is already in the ``memory_tracking`` dictionary and we are in forward, this means
        #         this module is called for the second time. If it is a root module, that means we are in the next
        #         iteration and we error out. If it is not a root module, that means it's a submodule that is being
        #         used multiple times in the same iteration, which we allow and track.
        # For Case 1 and 3, we also initialize the ``local_peak`` and ``PEAK_FW`` snapshot for the module.
        # For Case 2 we only capture 1 snapshot after ``FSDPState._pre_forward`` runs because it is a no-op.
        @wraps(orig_fsdp_state_pre_fw)
        def inner(
            *args: _P.args, **kwargs: _P.kwargs
        ) -> tuple[tuple[Unpack[_Ts]], dict[str, Any]]:
            self._fsdp_state = _FSDPState.PRE_FW
            mod_fqn = self._mod_tracker.get_known_fqn(fsdp_mod)
            if mod_fqn is None:
                raise AssertionError
            if fsdp_mod not in self.memory_tracking:
                mod_stat = _FSDPModMemStats(mod_fqn)
                self.memory_tracking[fsdp_mod] = mod_stat
                snapshot = self.get_tracker_snapshot()
                mod_stat.local_peak = {
                    dev: dev_snap[_TOTAL_KEY] for dev, dev_snap in snapshot.items()
                }
                mod_stat.snapshots.setdefault(_FSDPModState.PEAK_FW, []).append(
                    snapshot
                )
                mod_stat.snapshots.setdefault(_FSDPModState.BEF_PRE_FW, []).append(
                    deepcopy(snapshot)
                )
            elif not self._mod_tracker.is_bw:
                parents = self._mod_tracker.parents - {mod_fqn}
                if len(parents) == 1 and "Global" in parents:
                    raise NotImplementedError(
                        "FSDPMemTracker does not support memory tracking for multiple iterative calls."
                        " Either use ``reset_mod_stats`` to clear module memory stats for the previous iteration"
                        " or file a github issue if you need this feature."
                    )

            # pyrefly: ignore [bad-assignment]
            args, kwargs = orig_fsdp_state_pre_fw(*args, **kwargs)

            fsdp_state = fsdp_mod._get_fsdp_state()
            if fsdp_param_group := fsdp_state._fsdp_param_group:
                for fsdp_param in fsdp_param_group.fsdp_params:
                    self._update_and_maybe_create_winfos(
                        fsdp_param.unsharded_param,
                        _FSDPRefType.UNSHARDED_PARAM,
                    )
            mod_stat = self.memory_tracking[fsdp_mod]
            if self._mod_tracker.is_bw:
                state = _FSDPModState.PRE_FW_AC
                if self._ac_mod is None:
                    self._ac_mod = weakref.ref(fsdp_mod)
                    self._in_ac = True
            else:
                state = _FSDPModState.AFT_PRE_FW
            mod_stat.snapshots.setdefault(state, []).append(self.get_tracker_snapshot())
            self._fsdp_state = _FSDPState.FW
            # pyrefly: ignore [bad-return]
            return args, kwargs

        return inner

    def _fsdp_state_post_forward(
        self,
        fsdp_mod: FSDPModule,
        orig_fsdp_state_post_fw: Callable[_P, _R],
    ) -> Callable[_P, _R]:
        # We capture memory snapshots before and after ``FSDPState._post_forward`` to capture the resharded state
        # if ``reshard_after_forward`` is not ``False``. There are two cases:
        # Case 1: This is called in backward, which means we are in the AC region. If this is the top most module
        #         in the AC region, we set the flag ``_in_ac`` to False.
        # Case 2: This is called in forward.
        @wraps(orig_fsdp_state_post_fw)
        def inner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            mod_stat = self.memory_tracking[fsdp_mod]
            if self._mod_tracker.is_bw:
                state = _FSDPModState.POST_FW_AC
                if self._ac_mod is not None and self._ac_mod() is fsdp_mod:
                    self._ac_mod = None
                    self._in_ac = False
            else:
                state = _FSDPModState.BEF_POST_FW
            mod_stat.snapshots.setdefault(state, []).append(self.get_tracker_snapshot())
            self._fsdp_state = _FSDPState.POST_FW

            output = orig_fsdp_state_post_fw(*args, **kwargs)

            if not self._mod_tracker.is_bw:
                mod_stat.snapshots.setdefault(_FSDPModState.AFT_POST_FW, []).append(
                    self.get_tracker_snapshot()
                )
            return output

        return inner

    def _fsdp_param_group_pre_backward(
        self,
        fsdp_mod: FSDPModule,
        orig_fsdp_param_group_pre_backward: Callable[_P, Any],
    ) -> Callable[_P, None]:
        # We capture memory snapshots before and after ``FSDPParamGroup.pre_backward`` to capture the pre-fetching
        # and unsharding of params. We also initialize ``local_peak`` and ``PEAK_BW`` snapshot for the module.
        @wraps(orig_fsdp_param_group_pre_backward)
        def inner(*args: _P.args, **kwargs: _P.kwargs) -> None:
            self._fsdp_state = _FSDPState.PRE_BW
            mod_stat = self.memory_tracking[fsdp_mod]
            snapshot = self.get_tracker_snapshot()
            mod_stat.local_peak = {
                dev: dev_snap[_TOTAL_KEY] for dev, dev_snap in snapshot.items()
            }
            mod_stat.snapshots.setdefault(_FSDPModState.PEAK_BW, []).append(snapshot)
            mod_stat.snapshots.setdefault(_FSDPModState.BEF_PRE_BW, []).append(
                deepcopy(snapshot)
            )
            orig_fsdp_param_group_pre_backward(*args, **kwargs)

            mod_stat.snapshots.setdefault(_FSDPModState.AFT_PRE_BW, []).append(
                self.get_tracker_snapshot()
            )
            self._fsdp_state = _FSDPState.BW

        return inner

    def _fsdp_param_group_post_backward(
        self,
        fsdp_mod: FSDPModule,
        orig_fsdp_param_group_post_backward: Callable[_P, Any],
    ) -> Callable[_P, None]:
        # We capture the memory snapshots before and after ``FSDPParamGroup.post_backward`` to track and attribute
        # the `unsharded` grads before the post backward and then `sharded` grads and `reduce_scatter`  buffers
        # after the post backward.
        @wraps(orig_fsdp_param_group_post_backward)
        def inner(*args: _P.args, **kwargs: _P.kwargs) -> None:
            fsdp_state = fsdp_mod._get_fsdp_state()
            if fsdp_param_group := fsdp_state._fsdp_param_group:
                for fsdp_param in fsdp_param_group.fsdp_params:
                    unsharded_grad = fsdp_param._unsharded_param.grad
                    if unsharded_grad is not None:
                        self._update_and_maybe_create_winfos(
                            unsharded_grad,
                            _FSDPRefType.UNSHARDED_GRAD,
                            update_existing=True,
                        )

            mod_stat = self.memory_tracking[fsdp_mod]
            mod_stat.snapshots.setdefault(_FSDPModState.BEF_POST_BW, []).append(
                self.get_tracker_snapshot()
            )
            self._fsdp_state = _FSDPState.POST_BW
            orig_fsdp_param_group_post_backward(*args, **kwargs)

            if fsdp_param_group := fsdp_state._fsdp_param_group:
                for fsdp_param in fsdp_param_group.fsdp_params:
                    sharded_grad = fsdp_param.sharded_param.grad
                    if sharded_grad is not None:
                        self._update_and_maybe_create_winfos(
                            sharded_grad,
                            _FSDPRefType.SHARDED_GRAD,
                        )

            mod_stat.snapshots.setdefault(_FSDPModState.AFT_POST_BW, []).append(
                self.get_tracker_snapshot()
            )

        return inner

    def _instrument_fsdp_module(self) -> None:
        # We uninstall the existing `FSDPState._pre_forward` and `FSDPState._post_forward` hooks and install
        # our own hooks that wrap them. We choose this over monkey-patching `FSDPParamGroup.pre_forward` and
        # `FSDPParamGroup.post_forward` because during AC these won't be called.
        # TODO(@sanketpurandare): This will need to be modified after this PR (https://github.com/pytorch/pytorch/pull/127786)
        # lands. For backward we monkey-patch the `FSDPParamGroup.pre_backward` and `FSDPParamGroup.post_backward`.

        # get the unique _MultiHandlers/RemoveHandlers and store in dictionary
        # the _MultiHandlers object will only need to be grabbed once.
        unique_handlers: dict[RemovableHandle, bool] = {}

        for module in self._root_mod.modules():
            if isinstance(module, FSDPModule):
                fsdp_state = module._get_fsdp_state()
                if fsdp_param_group := fsdp_state._fsdp_param_group:
                    if not unique_handlers.get(fsdp_state._pre_forward_hook_handle):
                        unique_handlers[fsdp_state._pre_forward_hook_handle] = True
                    if not unique_handlers.get(fsdp_state._post_forward_hook_handle):
                        unique_handlers[fsdp_state._post_forward_hook_handle] = True
        # call remove on the handles once
        for f_hook_handle in unique_handlers:
            f_hook_handle.remove()

        for module in self._root_mod.modules():
            if isinstance(module, FSDPModule):
                fsdp_state = module._get_fsdp_state()
                if fsdp_param_group := fsdp_state._fsdp_param_group:
                    self._instrument_fsdp_sharded_params_grads(fsdp_param_group)
                    fsdp_state._pre_forward_hook_handle = (
                        module.register_forward_pre_hook(
                            self._fsdp_state_pre_forward(
                                module, fsdp_state._pre_forward
                            ),
                            prepend=True,
                            with_kwargs=True,
                        )
                    )

                    fsdp_state._post_forward_hook_handle = module.register_forward_hook(
                        self._fsdp_state_post_forward(module, fsdp_state._post_forward),
                        prepend=False,
                        always_call=True,
                    )
                    self._fsdp_mod_to_saved_methods[module] = _SavedFSDPMethods(
                        fsdp_param_group.pre_backward,
                        fsdp_param_group.post_backward,
                    )
                    fsdp_param_group.pre_backward = self._fsdp_param_group_pre_backward(  # type: ignore[assignment]
                        module, fsdp_param_group.pre_backward
                    )
                    fsdp_param_group.post_backward = (  # type: ignore[assignment]
                        self._fsdp_param_group_post_backward(
                            module, fsdp_param_group.post_backward
                        )
                    )

        for buffer in self._root_mod.buffers():
            self._update_and_maybe_create_winfos(
                buffer,
                _FSDPRefType.BUFFER,
            )

    def _instrument_optimizer(self) -> None:
        # Register a hook on the optimizer step to track the optimizer states.
        # The pre-hook is to set the flag ``_in_opt`` to True. The post-hook unsets the flag,
        # and also tracks any optimizer states that are created during the optimizer step.
        if self._optm is not None:
            self._track_optimizer_states(_FSDPRefType.OPT, self._optm)

            def _opt_step_pre_hook(
                optimizer: optim.Optimizer, args: Any, kwargs: Any
            ) -> None:
                self._in_opt = True

            def _opt_step_post_hook(
                optimizer: optim.Optimizer, args: Any, kwargs: Any
            ) -> None:
                self._track_optimizer_states(_FSDPRefType.OPT, optimizer)
                self._in_opt = False

            self._optimizer_hook_handles = (
                self._optm.register_step_pre_hook(_opt_step_pre_hook),
                self._optm.register_step_post_hook(_opt_step_post_hook),
            )

    def _register_module_and_optimizer_hooks(self) -> None:
        self._instrument_fsdp_module()
        self._instrument_optimizer()

    def _deregister_module_and_optimizer_hooks(self) -> None:
        for (
            fsdp_mod,
            saved_methods,
        ) in self._fsdp_mod_to_saved_methods.items():
            fsdp_state = fsdp_mod._get_fsdp_state()
            fsdp_state._pre_forward_hook_handle.remove()
            fsdp_state._post_forward_hook_handle.remove()
            fsdp_state._pre_forward_hook_handle = fsdp_mod.register_forward_pre_hook(
                fsdp_state._pre_forward, prepend=True, with_kwargs=True
            )
            fsdp_state._post_forward_hook_handle = fsdp_mod.register_forward_hook(
                fsdp_state._post_forward, prepend=False
            )
            if fsdp_param_group := fsdp_state._fsdp_param_group:
                fsdp_param_group.pre_backward = saved_methods.pre_backward
                fsdp_param_group.post_backward = saved_methods.post_backward
        self._fsdp_mod_to_saved_methods.clear()

        if self._optimizer_hook_handles is not None:
            for handle in self._optimizer_hook_handles:
                handle.remove()
            self._optimizer_hook_handles = None

    def track_inputs(self, inputs: tuple[Any, ...]) -> None:
        """
        This is used to track the input tensors to the model and annotate them as ``Inputs``.
        Args:
            inputs (Tuple[Any]): A tuple containing the input data. This can include tensors
                        as well as other data types. Only tensors will be tracked.
        """

        def _track_inputs(t: torch.Tensor) -> None:
            self._update_and_maybe_create_winfos(
                t,
                _FSDPRefType.INP,
            )

        tree_map_only(torch.Tensor, _track_inputs, inputs)

    def track_external(
        self, *external: nn.Module | optim.Optimizer | torch.Tensor
    ) -> None:
        """This is no-op for ``FSDPMemTracker``"""

    def __enter__(self) -> "FSDPMemTracker":
        if self._depth == 0:
            # None in eager, a FakeTensorMode instance in SAC.  Used in
            # __torch_dispatch__ to skip DTensor propagation ops.
            self._fake_mode_on_entry = active_fake_mode()
            self._register_module_and_optimizer_hooks()
            self._track_resize()
            self._peak_mem_snap = self.get_tracker_snapshot()
            self._peak_mem = {
                dev: dev_snap[_TOTAL_KEY]
                for dev, dev_snap in self._peak_mem_snap.items()
            }
            self._mod_tracker.__enter__()
        TorchDispatchMode.__enter__(self)
        self._depth += 1
        return self

    def __exit__(self, *args: Any) -> None:
        self._depth -= 1
        if self._depth == 0:
            self._deregister_module_and_optimizer_hooks()
            self._restore_resize()
            self._mod_tracker.__exit__(*args)
        TorchDispatchMode.__exit__(self, *args)

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):  # type: ignore[no-untyped-def]
        # When running this mode with DTensor, ordinarily all modes will
        # run **before** subclasses get a chance to run.
        # Returning NotImplemented here gives us a chance to let DTensor
        # run and desugar into local tensor ops, before `MemTracker` sees them.
        if any(t == DTensor for t in types):
            return NotImplemented
        if (
            func is torch.ops._c10d_functional.wait_tensor.default
            and active_fake_mode()
        ):
            # N.B: This is a hacky way to override the Meta IMPL of wait_tensor. The original impl returns
            # a new tensor which does not happen in eager mode, when a wait_tensor is called.
            # pyrefly: ignore [unsupported-operation]
            res = args[0]
        else:
            res = func(*args, **kwargs or {})
        # DTensor sharding propagation may use a nested FakeTensorMode to
        # compute output metadata (shapes, placements).  The real memory
        # usage comes from the local op on shard data, which
        # runs outside DTensor's fake mode.  We use identity comparison
        # against the fake mode at tracker entry to skip propagation ops
        # while still tracking local ops in both eager and SAC contexts.
        #   - Eager (entry mode is None): real ops have no fake mode
        #     (None is None), DTensor propagation ops are skipped (B is not None).
        #   - SAC (entry mode is A): SAC ops run under mode A (A is A),
        #     DTensor propagation ops are skipped (B is not A).
        if active_fake_mode() is self._fake_mode_on_entry:
            # If we are tracking an optimizer state, we use the optimizer reference type.
            # If we are in backward region and not in AC region, we use the backward reference type.
            # Else we use the forward reference type.
            if self._in_opt:
                reftype = _FSDPRefType.OPT
            elif self._mod_tracker.is_bw and not self._in_ac:
                reftype = _FSDPRefType.TEMP
            else:
                reftype = _FSDPRefType.ACT
            if func is c10d._allgather_base_.default and self._fsdp_state in [
                _FSDPState.PRE_FW,
                _FSDPState.PRE_BW,
            ]:
                # pyrefly: ignore [unsupported-operation]
                output_tensor = args[0]
                self._update_and_maybe_create_winfos(
                    output_tensor,
                    _FSDPRefType.ALL_GATHER,
                    update_existing=True,
                )
            if (
                func is c10d._reduce_scatter_base_.default
                and self._fsdp_state == _FSDPState.POST_BW
            ):
                # pyrefly: ignore [unsupported-operation]
                input_tensor = args[1]
                self._update_and_maybe_create_winfos(
                    input_tensor,
                    _FSDPRefType.REDUCE_SCATTER,
                    update_existing=True,
                )

            tree_map_only(torch.Tensor, partial(self._track, reftype), res)
            peak_state = (
                _FSDPModState.PEAK_BW
                if self._mod_tracker.is_bw
                else _FSDPModState.PEAK_FW
            )
            self._update_peak_stats(peak_state)
        return res
<|endoftext|>import copy
from collections import OrderedDict
from typing import cast, TypedDict

import numpy as np

import torch
from torch.distributed._tools.mem_tracker import (
    _MemRefType,
    _ModMemStats,
    _ModState,
    MemTracker,
)
from torch.distributed._tools.runtime_estimator import RuntimeEstimator
from torch.distributed._tools.sac_estimator import SACEstimator, SACTradeOffStats


class ModOrder(TypedDict):
    fw_pre_order: list[str]
    bw_pre_order: list[str]
    fw_post_order: list[str]
    bw_post_order: list[str]


class ModRuntime(TypedDict):
    fw: float
    bw: float


class ModStats(TypedDict):
    fqn: str
    # per-module params
    param_per_module: int
    # per-module grads
    grad_per_module: int
    # total accumulated gradients up to and including this module
    grad_total: int
    # per module fw activation size (excluding input and output)
    act_fw_per_module: int
    # per module bw activation size during peak_bw
    act_bw_per_module: int
    # per module activation grad size during peak_bw
    act_grad_per_module: int
    # total activation size up to but excluding the current module
    # includes input of the current module (i.e., output of previous module)
    act_total: int
    # Inputs to the module
    input_per_module: int
    # Outputs of the module
    output_per_module: int
    # Total fw run-time of the module
    fw_runtime_per_module: float
    # Total bw run-time of the module
    bw_runtime_per_module: float
    # Is this module a leaf module
    is_leaf: bool
    # Total ac run-time of the module
    sac_runtime: float
    # Total ac_memory for the module
    sac_memory: int
    # Number of piecewise-linear functions used for approximating ac tradeoff curve
    n_segments: int
    # Slopes of the of piecewise-linear functions
    slopes: list[float]
    # Intercepts of the of piecewise-linear functions
    intercepts: list[float]
    # X breakpoints of the of piecewise-linear functions
    breakpoints: list[float]
    # Original trade-off curves
    tradeoff_curve: OrderedDict[float, float]


class ModuleInfo(TypedDict):
    mod_order: ModOrder
    mod_stats: list[ModStats]


def aggregate_stats(
    model: torch.nn.Module,
    mem_tracker: MemTracker,
    runtime_estimator: RuntimeEstimator,
    sac_estimator: SACEstimator,
    dev: torch.device,
) -> ModuleInfo:
    """
    Collect modulewise stats for a given model, including memory, runtime, and AC tradeoff stats.

    Args:
        model: nn.Module object
        runtime_estimator: RuntimeEstimator object with runtime stats
        mem_tracker: MemTracker object with memory stats
        sac_estimator: SACEstimator object with AC tradeoff stats
        dev: device the model was run on (used to extract memory stats from MemTracker)

    Returns:
        ModuleInfo: A dictionary with module order and module stats.
    """

    # Memory stats
    mod_mem_stats: dict[torch.nn.Module, _ModMemStats] = dict(
        copy.deepcopy(mem_tracker.memory_tracking)
    )

    # Runtime stats
    mod_runtime_stats: dict[str, ModRuntime] = {
        fqn: {"fw": v["fw"], "bw": v["bw"]}
        for fqn, v in runtime_estimator.mod_runtimes.items()
    }

    # Module order
    mod_order: ModOrder = {
        "fw_pre_order": list(runtime_estimator.mod_fw_pre_order),
        "bw_pre_order": list(runtime_estimator.mod_bw_pre_order),
        "fw_post_order": list(runtime_estimator.mod_fw_post_order),
        "bw_post_order": list(runtime_estimator.mod_bw_post_order),
    }

    # Selective Activation Checkpointing stats
    sac_estimator.pwlf_sac_tradeoff_curve()
    mod_sac_tradeoff_stats: dict[str, SACTradeOffStats] = copy.deepcopy(
        sac_estimator.sac_mod_tradeoff_stats
    )

    module_info: ModuleInfo = {
        "mod_order": mod_order,
        "mod_stats": [],
    }

    for mod in model.modules():
        if mod_mem_stat := mod_mem_stats.get(mod):
            if tradeoff_stats := mod_sac_tradeoff_stats.get(mod_mem_stat.mod_fqn):
                sac_runtime = tradeoff_stats.sac_runtime
                sac_memory = tradeoff_stats.sac_memory
                n_segments = tradeoff_stats.n_segments
                slopes = tradeoff_stats.slopes
                intercepts = tradeoff_stats.intercepts
                breakpoints = tradeoff_stats.fit_breaks
                tradeoff_curve = tradeoff_stats.tradeoff_curve
                is_leaf = False
            else:
                sac_runtime = sac_memory = n_segments = 0
                slopes = intercepts = breakpoints = []
                tradeoff_curve: OrderedDict[float, float] = OrderedDict()  # type: ignore[no-redef]
                is_leaf = True
            mod_stat: ModStats = {
                "fqn": mod_mem_stat.mod_fqn,
                "param_per_module": mod_mem_stat.parameter_mem,
                "grad_per_module": mod_mem_stat.parameter_mem,
                "grad_total": mod_mem_stat.snapshots[_ModState.PRE_BW][-1][dev][
                    _MemRefType.GRAD
                ],
                "act_fw_per_module": max(
                    0,
                    mod_mem_stat.snapshots[_ModState.POST_FW][-1][dev][_MemRefType.ACT]
                    - mod_mem_stat.snapshots[_ModState.PRE_FW][-1][dev][_MemRefType.ACT]
                    - mod_mem_stat.output_mem,
                ),
                "act_bw_per_module": max(
                    0,
                    mod_mem_stat.snapshots[_ModState.PEAK_BW][-1][dev][_MemRefType.ACT],
                ),
                "act_grad_per_module": (
                    mod_mem_stat.snapshots[_ModState.PEAK_BW][-1][dev][_MemRefType.TEMP]
                    - mod_mem_stat.snapshots[_ModState.PRE_BW][-1][dev][
                        _MemRefType.TEMP
                    ]
                ),
                "act_total": mod_mem_stat.snapshots[_ModState.POST_FW][-1][dev][
                    _MemRefType.ACT
                ],
                "input_per_module": mod_mem_stat.input_mem,
                "output_per_module": mod_mem_stat.output_mem,
                "fw_runtime_per_module": mod_runtime_stats[mod_mem_stat.mod_fqn]["fw"],
                "bw_runtime_per_module": mod_runtime_stats[mod_mem_stat.mod_fqn]["bw"],
                "is_leaf": is_leaf,
                "sac_runtime": sac_runtime,
                "sac_memory": sac_memory,
                "n_segments": n_segments,
                "slopes": slopes,
                "intercepts": intercepts,
                "breakpoints": breakpoints,
                "tradeoff_curve": tradeoff_curve,
            }
            module_info["mod_stats"].append(mod_stat)

    return module_info


class Node(ModStats):
    index: int  # index according to forward pre-order
    pos_fw_post_order: int  # index according to forward post-order


class Graph:
    def __init__(self, n: int) -> None:
        self.nodes: list[Node] = []
        self.name2node: dict[str, Node] = {}
        self.ad_matrix = np.zeros((n, n))
        self.fw_post_order: list[str] = []

    def add_node(self, node: Node) -> None:
        self.nodes.append(node)
        self.name2node[node["fqn"]] = node


def parse_module_info(module_info: ModuleInfo) -> Graph:
    """
    Parse module info and create a graph (tree) of modules. The graph will be
    used by MILP solver to find optimal SAC and/or FSDP configurations.
    """
    mod_stats = module_info["mod_stats"]
    fw_pre_order = module_info["mod_order"]["fw_pre_order"]
    # assertion and number of nodes
    if len(mod_stats) != len(fw_pre_order):
        raise AssertionError
    n_nodes = len(mod_stats)

    # create graph
    g = Graph(n_nodes)
    g.fw_post_order = module_info["mod_order"]["fw_post_order"]

    # sort the modules by pre-order and add them to the graph
    module_info["mod_stats"] = sorted(
        mod_stats, key=lambda x: fw_pre_order.index(x["fqn"])
    )
    for i, one_mod_stats in enumerate(mod_stats):
        node: Node = cast(Node, one_mod_stats)
        node["index"] = i
        node["pos_fw_post_order"] = g.fw_post_order.index(node["fqn"])
        g.add_node(node)

    # set up ancestor-descendant matrix
    for i in range(n_nodes):
        for j in range(i, n_nodes):
            if is_self_or_submodule(g.nodes[j]["fqn"], g.nodes[i]["fqn"]):
                g.ad_matrix[i][j] = 1
            else:
                break

    return g


def is_self_or_submodule(name_descendant: str, name_ancestor: str) -> bool:
    """
    check if name_descendant is a submodule of name_ancestor, or if they are the same
    """
    return name_descendant == name_ancestor or name_ancestor + "." in name_descendant


def is_submodule(name_descendant: str, name_ancestor: str) -> bool:
    """
    if name_descendant is a submodule of name_ancestor, but not the same
    """
    return name_ancestor + "." in name_descendant


def display_bytes(b: int, unit: str = "MiB") -> str:
    """
    return a string that represent the number of bytes in a desired unit
    """
    if unit == "KiB":
        return f"{b / 2**10:.2f} KiB"
    if unit == "MiB":
        return f"{b / 2**20:.2f} MiB"
    if unit == "GiB":
        return f"{b / 2**30:.2f} GiB"
    return f"{b:.2f} bytes"


def get_peak_memory_runtime_baseline(graph: Graph) -> tuple[int, float]:
    """
    Get the baseline peak memory and runtime.
    Baseline here means there is no FSDP or AC.
    Memory includes the parameters, gradients, activations, and activation gradients.
    Memory does not include e.g., optimizer states, embedding tables, etc.

    Returns:
        int: peak memory in bytes
        float: compute time in ms
    """
    P_1 = graph.nodes[0]["param_per_module"]
    num_nodes = len(graph.nodes)
    peak_mem = 0
    for i in range(num_nodes):
        TG_i = graph.nodes[i]["grad_total"]
        AG_i = graph.nodes[i]["act_grad_per_module"]
        TA_i = graph.nodes[i]["act_total"]
        peak_mem = max(peak_mem, P_1 + TG_i + AG_i + TA_i)
    compute_time = (
        graph.nodes[0]["fw_runtime_per_module"]
        + graph.nodes[0]["bw_runtime_per_module"]
    )
    return (peak_mem, compute_time)
<|endoftext|>import math
import os
import sys
from collections import OrderedDict
from dataclasses import astuple, dataclass
from typing import Any, NamedTuple
from typing_extensions import Self

import torch
from torch import nan, nn, UntypedStorage
from torch._guards import active_fake_mode
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.common_utils import get_untyped_storages
from torch.distributed._tools.mod_tracker import ModTracker
from torch.distributed._tools.runtime_estimator import RuntimeEstimator
from torch.testing._internal.composite_compliance import (
    is_inplace,
    is_inplace_view_fn,
    is_view_fn,
)
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from torch.utils.checkpoint import SAC_IGNORED_OPS


__all__ = ["SACEstimator", "SACStats", "MSPS", "SACTradeOffStats", "SACGreedyOrderMeta"]
aten = torch.ops.aten

_ADDITIONAL_IGNORED_OPS = {
    aten.lift_fresh.default,  # type: ignore[attr-defined]
    torch.ops.profiler._record_function_exit._RecordFunction,  # type: ignore[attr-defined]
    aten.clone.default,  # type: ignore[attr-defined] # seems needed for torch.compile
}
OPS_TO_ALWAYS_SKIP = SAC_IGNORED_OPS | _ADDITIONAL_IGNORED_OPS
# This value is hard-coded here:
# https://github.com/pytorch/pytorch/blob/5fba5d83f0703ff8077ab65448a998e9ad6598fd/c10/cuda/CUDACachingAllocator.cpp#L117
_PYTORCH_MIN_ALLOCATE = (
    2**9 if int(os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING", 0)) == 0 else 1
)


def _display_stats_tabular(headers: list[str], table_data: list[list[Any]]) -> None:
    try:
        from tabulate import tabulate
    except ImportError as err:
        raise ImportError("Please install tabulate.") from err

    # Use tabulate to print the table
    print(tabulate(table_data, headers=headers, tablefmt="rst"))


# Based on:
# https://github.com/facebookresearch/xformers/blob/main/xformers/checkpoint.py#L71
@dataclass
class _SACMetadata:
    """
    Stores metadata for a single operator for SAC.

    Attributes:
        func (Any): The operator function.
        time_taken (float): The time taken by the operator.
        memory_used (float): The memory used by the operator.
        curr_idx (int): The current operator index.
        output_ids (Tuple[int, ...]): The storage IDs of the operator's outputs.
        inplace_info (Tuple[int, ...]): Tuple of self and parent operator for in-place operator.
        is_view_like (bool): Whether the operator is view-like.
        is_rand_op (bool): Whether the operator is a random operator.
    """

    func: Any
    time_taken: float
    memory_used: float
    curr_idx: int
    output_ids: tuple[int, ...]
    inplace_info: tuple[int, ...]
    is_view_like: bool
    is_rand_op: bool


@dataclass
class _SACModMetadata:
    """
    Stores metadata for a module for SAC.

    Attributes:
        start_idx (int): The starting index of the module's operators.
        force_store_random (bool): Whether to force store random operators in the module.
        sac_metadata (List[_SACMetadata]): List of metadata for each operator in the module.
    """

    start_idx: int
    force_store_random: bool
    sac_metadata: list[_SACMetadata]


@dataclass
class SACStats:
    """
    A class for storing Activation Checkpointing statistics corresponding to a module.

    Attributes:
        func_names (List[str]): List of operator names.
        runtimes (List[float]): List of operator runtimes in millliseconds.
        memory (List[int]): List of operator memory usage in bytes.
        view_like_ops (List[int]): Indices of view-like operators.
        rand_ops (List[int]): Indices of random operators.
        saved_autograd_ops (List[int]): Indices of operator results saved by autograd engine.
        inplace_ops (List[Tuple[int, int]]): Tuple of indices of op and its first parent for Inplace operators.
        force_store_random (bool): Whether to force store random operator results.
    """

    func_names: list[str]
    runtimes: list[float]
    memory: list[int]
    view_like_ops: list[int]
    rand_ops: list[int]
    saved_autograd_ops: list[int]
    inplace_ops: list[tuple[int, int]]
    force_store_random: bool


class MSPS(NamedTuple):
    """
    Represents Memory and Runtime Statistics for an operator/operator group.

    Attributes:
        func_names (set[str]): Set of operator/operator group names.
        op_idx (int): Operator index (group head index in case of operator groups).
        memory (int): Memory usage in bytes.
        runtime (float): Runtime in milliseconds.
        msps (float): Memory per second calculated as memory/runtime.
    """

    func_names: set[str]
    op_idx: int
    memory: int
    runtime: float
    msps: float


@dataclass
class SACTradeOffStats:
    """
    Stores statistics for activation-checkpointing trade-off.

    Attributes:
        n_segments (int): Number of piecewise linear segments fitted to the trade-off curve.
        slopes (List[float]): Slopes of the pieces of linear segments fitted to the trade-off curve.
        intercepts (List[float]): Intercepts of the of the pieces of linear segments fitted to the trade-off curve.
        fit_breaks (List[float]): Breakpoints of the of the pieces of linear segments fitted to the trade-off curve.
        tradeoff_curve (OrderedDict[float, float]): Trade-off curve data of memory discarded vs recomputation time.
        sac_memory (int): Total memory of operations available for activation checkpointing in bytes.
        sac_runtime (float): Total runtime of operations available for activation checkpointing in milliseconds.
    """

    n_segments: int
    slopes: list[float]
    intercepts: list[float]
    fit_breaks: list[float]
    tradeoff_curve: OrderedDict[float, float]
    sac_memory: int
    sac_runtime: float


@dataclass
class SACGreedyOrderMeta:
    """
    Stores metadata for Greedy-order SAC.

    Attributes:
        recomputed_ops (set[int]): Set of operator indices to be recomputed.
        stored_ops (set[int]): Set of operator indices to be stored.
        inplace_op_groups (dict[int, set[int]]): Dictionary of inplace operator groups from group-head to operators.
        random_ops_group (dict[int, set[int]]): Dictionary of random op group head to random ops.
        msps_meta (list[MSPS]): List of Memory and Runtime Statistics for operators.
    """

    recomputed_ops: set[int]
    stored_ops: set[int]
    inplace_op_groups: dict[int, set[int]]
    random_ops_group: dict[int, set[int]]
    msps_meta: list[MSPS]


class SACEstimator(TorchDispatchMode):
    """
    Estimates the memory and recomputation time trade-offs for applying Selective Activation Checkpointing (SAC).

    This class provides a ``TorchDispatchMode`` based context manager that can be used to estimate the memory and
    runtime trade-offs of functions or ``torch.nn.Module``s for Selective Activation Checkpointing (SAC). It provides
    detailed statistics and metadata information for operators of each module and provides a greedy order for selecting
    the operators to be recomputed/checkpointed.  It also constructs the per-module trade-off graph of discarded memory
    vs recomputation time for the obtained greedy order. Using ``RuntimeEstimator`` under the hood, it supports two
    estimation modes, `operator-level-benchmark` and (`operator-level-cost-model` (roofline model).

    Attributes:
        sac_mod_stats (Dict[str, SACStats]): Dictionary from module FQN (fully qualified name) to ``SACStats``.
        sac_mod_tradeoff_stats (Dict[str, SACTradeOffStats]): Dictionary from module FQN to ``SACTradeOffStats``.
        sac_mod_greedy_order_meta (Dict[str, SACGreedyOrderMeta]): Dictionary from module FQN to ``SACGreedyOrderMeta``.

    Note:
        1) This class is designed to be used under ``FakeTensorMode``.
        2) Currently, it only supports estimation of compute time and memory usage, and does not consider communication.

    Example usage:

        .. code-block:: python

            sac_estimator = SACEstimator()
            with FakeTensorMode():
                module = ...
                inp = ...
                with sac_estimator("operator-level-cost-model"):
                    output = module(inp)
                sac_estimator.display_modulewise_sac_stats(depth=4, print_tabular=True)
    """

    def __init__(self, gpu_type: str | None = None) -> None:
        """
        Args:
            gpu_type (str | None): Optional datasheet device name (e.g.
                ``"NVIDIA H100"``) to pin the roofline estimates to instead of
                querying the current device. See ``RuntimeEstimator``.
        """
        self.sac_mod_stats: dict[str, SACStats] = {}
        self.sac_mod_tradeoff_stats: dict[str, SACTradeOffStats] = {}
        self.sac_mod_greedy_order_meta: dict[str, SACGreedyOrderMeta] = {}
        self._mod_tracker = ModTracker()
        self._sac_metadata: list[_SACMetadata] = []
        self._sac_mod_metadata: dict[str, _SACModMetadata] = {}
        self._leaf_modules: set[str] = set()
        self._saved_tensor_hook_ctx = torch.autograd.graph.saved_tensors_hooks(
            self._pack_hook, lambda x: x
        )
        self._saved_tensor_ids: set[int] = set()
        self._gpu_type = gpu_type
        self._estimate_runtime = RuntimeEstimator._roofline_estimate

    def _pack_hook(self, x: torch.Tensor) -> torch.Tensor:
        # Hook function to track underlying storage IDs of tensors
        # Updates the _saved_tensor_ids set with the IDs of the tensor's storages
        # Used in conjunction with torch.autograd.graph.saved_tensors_hooks
        untyped_storages = get_untyped_storages(x)
        storage_ids = (hash(st) for st in untyped_storages)
        self._saved_tensor_ids.update(storage_ids)
        return x

    def _pre_fw_hook(self, mod: nn.Module, inputs: Any) -> None:
        # Pre-forward hook function to prepare module metadata
        # Tracks module FQN, force store random flag, and ``SACModMetadata``
        # Initializes metadata for non-leaf modules, marks leaf modules
        mod_fqn = self._mod_tracker.get_known_fqn(mod)
        if mod_fqn is None:
            raise AssertionError
        num_children = sum(1 for _ in mod.children())
        if num_children > 0:
            force_store_random = self._get_force_store_random(inputs)
            self._sac_mod_metadata[mod_fqn] = _SACModMetadata(
                start_idx=len(self._sac_metadata),
                force_store_random=force_store_random,
                sac_metadata=[],
            )
        else:
            self._leaf_modules.add(mod_fqn)

    def _post_fw_hook(self, mod: nn.Module, inputs: Any, outputs: Any) -> None:
        # 1. Retrieves the module's FQN and checks if it's a leaf module
        # 2. If not a leaf module, computes:
        #    - ``SACStats`` using the module's metadata and force store random flag
        #    - ``SACGreedyOrderMeta`` using the computed SAC statistics
        mod_fqn = self._mod_tracker.get_known_fqn(mod)
        if mod_fqn is None:
            raise AssertionError
        if mod_fqn in self._leaf_modules:
            return
        else:
            self.sac_mod_stats[mod_fqn] = self._get_sac_stats(
                data=self._sac_mod_metadata[mod_fqn].sac_metadata,
                force_store_random=self._sac_mod_metadata[mod_fqn].force_store_random,
            )
            self.sac_mod_greedy_order_meta[mod_fqn] = self._get_greedy_order_meta(
                self.sac_mod_stats[mod_fqn]
            )

    def _get_force_store_random(self, inputs: Any) -> bool:
        flat_inputs, _ = tree_flatten(inputs)
        return all(not isinstance(x, torch.Tensor) for x in flat_inputs)

    def _get_sac_stats(
        self, data: list[_SACMetadata], force_store_random: bool
    ) -> SACStats:
        # 1. Ignore the operations that should be skipped by SAC such as aten.detach.default because autograd
        # inserts those during backward and it breaks the fwd-bwd alignment
        filtered_data = [x for x in data if x.func not in OPS_TO_ALWAYS_SKIP]

        (
            ops,
            runtimes_,
            memory_,
            new_ids,
            output_ids,
            inplace_ops_,
            view_like_ops_,
            rand_ops_,
        ) = zip(*[astuple(x) for x in filtered_data], strict=True)

        # 2. Extract the metadata information
        runtimes = list(runtimes_)
        memory = list(memory_)
        func_names = [op._overloadpacket.__name__ for op in ops]
        view_like_ops = [i for i, x in enumerate(view_like_ops_) if x]
        rand_ops = [i for i, x in enumerate(rand_ops_) if x]
        saved_autograd_ops = [
            i
            for i, out_ids in enumerate(output_ids)
            if set(out_ids).issubset(self._saved_tensor_ids)
        ]

        # 3. Remap the inplace indices as we have removed OPS_TO_ALWAYS_SKIP
        # FIXME @sanketpurandare: Fix this by changing the parent of the inplace-op
        # to itself if the original parent is in OPS_TO_ALWAYS_SKIP.
        try:
            inplace_ops = [tuple(map(new_ids.index, x)) for x in inplace_ops_ if x]
        except ValueError as err:
            raise ValueError(
                f"The remapping of inplace ops failed since one of the inplace op parents"
                f" must have been present in {OPS_TO_ALWAYS_SKIP}"
            ) from err

        # 4. The last operation is always stored as the output of the checkpoint
        # block, so we can avoid recomputing it. We set the memory to zero
        # instead of adding a new constraint because we want both the 0 and 1
        # endpoints for memory_budget to be valid
        # FIXME @sanketpurandare: this heuristic for finding the last non-view non-inplace op
        # might not always be correct, which would yield suboptimal policies
        last_op = len(ops) - 1
        skip_ops_ = set(view_like_ops) | set({x[0] for x in inplace_ops})
        reversed_skip_ops = sorted(skip_ops_, reverse=True)
        for op in reversed_skip_ops:
            if op == last_op:
                last_op -= 1

        memory[last_op] = 0

        # 5. Create a single ``SACStats`` object for the entire block of ``_SACMetadata``.
        return SACStats(
            func_names=func_names,
            runtimes=runtimes,
            memory=memory,
            view_like_ops=view_like_ops,
            rand_ops=rand_ops,
            saved_autograd_ops=saved_autograd_ops,
            inplace_ops=inplace_ops,  # type: ignore[arg-type]
            force_store_random=force_store_random,
        )

    def _get_inplace_metadata(
        self, func: Any, out_storages: set[UntypedStorage]
    ) -> tuple[int, tuple[int, ...], dict[str, tuple[int, ...]]]:
        # 1. Get the current index of the metadata obtained so far
        curr_idx = len(self._sac_metadata)
        # 2. Get the set of active modules that are not leaf
        active_mod_fqns: set[str] = {
            par for par in self._mod_tracker.parents if par not in self._leaf_modules
        }
        # 3. Output ids are the identifies of the storage objects corresponding to the tensors
        output_ids = tuple(hash(st) for st in out_storages)
        # 4. If the function is not inplace, return
        if not is_inplace(func):
            return curr_idx, output_ids, dict.fromkeys(active_mod_fqns, ())

        op_idx = curr_idx
        # 5. Initialize the parent op ids of the inplace op for each of the active modules
        mod_op_parent_idxs: dict[str, int] = dict.fromkeys(active_mod_fqns, -1)
        for i, d in enumerate(self._sac_metadata):
            # 6. Find the first occurrence of a tensor corresponding to each module that
            # shares the same storage as the current tensor
            past_output_ids = d.output_ids
            if set(output_ids).issubset(set(past_output_ids)):
                for mod_fqn, op_parent_idx in mod_op_parent_idxs.items():
                    if op_parent_idx == -1:
                        if acm_stats := self._sac_mod_metadata.get(mod_fqn, None):
                            if i >= acm_stats.start_idx:
                                mod_op_parent_idxs[mod_fqn] = i
                        else:
                            if mod_fqn != "Global":
                                raise AssertionError
                            mod_op_parent_idxs[mod_fqn] = i
        # 7. If no parent tensor is found, then it's probably an inplace op on the arguments
        # so one can just store the current-op idx as parent idx
        for mod_fqn, op_parent_idx in mod_op_parent_idxs.items():
            if op_parent_idx < 0:
                mod_op_parent_idxs[mod_fqn] = op_idx
        mod_inplace_info = {
            mod_fqn: (op_idx, mod_op_parent_idxs[mod_fqn])
            for mod_fqn in active_mod_fqns
        }
        return curr_idx, output_ids, mod_inplace_info  # type: ignore[return-value]

    def __torch_dispatch__(  # type: ignore[no-untyped-def]
        self, func, types, args=..., kwargs=None
    ):
        # 1. Get the runtime estimate
        out, op_time = self._estimate_runtime(func, args, kwargs)
        flat_outs, _ = tree_flatten(out)
        out_storages_cuda: set[UntypedStorage] = set()
        out_storages_cpu: set[UntypedStorage] = set()
        cuda_devices: set[torch.device] = set()
        for o in flat_outs:
            if isinstance(o, torch.Tensor):
                if o.device.type == "cuda":
                    out_storages_cuda.update(get_untyped_storages(o))
                    cuda_devices.add(o.device)
                else:
                    out_storages_cpu.update(get_untyped_storages(o))

        # Check if there's more than 1 CUDA device
        if len(cuda_devices) > 1:
            raise AssertionError(
                f"{func.__name__}'s output has more than 1 CUDA devices {cuda_devices}"
            )

        # 2. Get the memory consumed by output
        nbytes_cuda = sum(
            math.ceil(st.nbytes() / _PYTORCH_MIN_ALLOCATE) * _PYTORCH_MIN_ALLOCATE
            for st in out_storages_cuda
        )
        nbytes_cpu = sum(st.nbytes() for st in out_storages_cpu)
        nbytes = nbytes_cuda + nbytes_cpu
        # 3. Get the current operator index, output storage identifiers and inplace metadata
        out_storages = out_storages_cuda | out_storages_cpu
        curr_idx, output_ids, mod_inplace_info = self._get_inplace_metadata(
            func, out_storages
        )
        # 4. Determine if the function is in-place, random-op or a view-like
        is_view_like = is_view_fn(func) or is_inplace_view_fn(func)
        is_rand_op = torch.Tag.nondeterministic_seeded in func.tags
        if is_view_like:
            nbytes = 0
        # sdpa has non-deterministic seed, but might be deterministic
        # if no dropout is applied
        if func.overloadpacket.__name__ == "_scaled_dot_product_flash_attention":
            # pyrefly: ignore [missing-attribute]
            is_rand_op = kwargs.get("dropout_p", 0) != 0
        # 5. Create metadata information per active non-leaf module
        for mod_fqn in self._mod_tracker.parents:
            if mod_fqn in self._leaf_modules:
                continue
            acm = _SACMetadata(
                func=func,
                time_taken=op_time,
                memory_used=nbytes,
                curr_idx=curr_idx,
                output_ids=output_ids,
                inplace_info=mod_inplace_info[mod_fqn],
                is_view_like=is_view_like,
                is_rand_op=is_rand_op,
            )
            if acm_stats := self._sac_mod_metadata.get(mod_fqn, None):
                acm_stats.sac_metadata.append(acm)
            else:
                if mod_fqn != "Global":
                    raise AssertionError(f"Module {mod_fqn} not found in AC Mod Stats")
                self._sac_metadata.append(acm)

        return out

    def _get_greedy_order_meta(self, sac_stats: SACStats) -> SACGreedyOrderMeta:
        # An inplace-op group is a set of inplace-ops that operate on the same underlying tensor storage.
        # 1. inplace_op_groups: A dictionary from the top-most parent of inplace-ops to the inplace-ops in the group
        #   The top-most op can itself be an inplace-op or can be a non-inplace op.
        # 2. inplace_op_to_group_head: A dictionary that maps all the inplace-ops to their respective group heads.
        inplace_op_groups: dict[int, set[int]] = {}
        inplace_op_to_group_head: dict[int, int] = dict(sac_stats.inplace_ops)

        # Initialize inplace_op_groups using inplace_op_to_group_head
        for op_idx, group_head_idx in inplace_op_to_group_head.items():
            op_group = inplace_op_groups.setdefault(group_head_idx, {group_head_idx})
            op_group.add(op_idx)

        # Like inplace ops, all of the random ops in the function/module should all be either recomputed or saved
        # as a group. This is because, they affect the random seed generator. If force_store_random is set True,
        # all of the random ops will be stored by default. For ease of manageability, we store the top-most random op
        # as the leader of the random_ops_group.
        random_ops_group: dict[int, set[int]] = {}
        random_group_head_idx = min(sac_stats.rand_ops, default=-1)
        has_rand_ops = bool(sac_stats.rand_ops)
        if has_rand_ops:
            random_ops_group[random_group_head_idx] = set(sac_stats.rand_ops)

        # 1. Random ops are stored if force_store_random is set
        # 2. View-like ops are recomputed by default
        # 3. For inplace_op_groups:
        #   a) If the head of this group is an inplace op, then we have to store the entire group.
        #   b) If any op in the group is random and force_store_random is set, then entire group will be stored.
        #   c) If none of ops in the group are random and the head of the group is not an in-place op, then
        #       this group can be considered for recomputation in its entirety
        stored_ops: set[int] = set()
        recomputed_ops: set[int] = set()
        # Case 1:
        if has_rand_ops and sac_stats.force_store_random:
            stored_ops.add(random_group_head_idx)
        # Case 2:
        recomputed_ops.update(set(sac_stats.view_like_ops))

        for group_head_idx, op_group in inplace_op_groups.items():
            # Case 3a:
            if group_head_idx in inplace_op_to_group_head:
                stored_ops.add(group_head_idx)
            # Case 3b:
            if (
                sac_stats.force_store_random & len(op_group & set(sac_stats.rand_ops))
                > 0
            ):
                stored_ops.add(group_head_idx)

        # The potential recompute candidates are populated as:
        recompute_candidates: set[int] = set()
        # 1) The random group head if it is not stored
        if has_rand_ops and random_group_head_idx not in stored_ops:
            recompute_candidates.add(random_group_head_idx)
        # 2) The in-place op group heads that are not stored
        recompute_candidates.update(set(inplace_op_groups.keys()) - stored_ops)
        # 3) The non-inplace and non-random ops that are neither stored nor recomputed by default
        recompute_candidates.update(
            set(range(len(sac_stats.memory)))
            - recomputed_ops
            - stored_ops
            - set(inplace_op_to_group_head.keys())
            - set(sac_stats.rand_ops)
        )

        # We define msps for a recomp candidate as the ratio of memory/runtime aka memory savings per second
        msps_meta: list[MSPS] = []
        for cand_idx in recompute_candidates:
            op_indices = {cand_idx}
            if cand_idx in inplace_op_groups:
                op_indices.update(inplace_op_groups[cand_idx])
            if has_rand_ops and cand_idx == random_group_head_idx:
                op_indices.update(sac_stats.rand_ops)

            mem = sum(sac_stats.memory[op_idx] for op_idx in op_indices)
            runtime = sum(sac_stats.runtimes[op_idx] for op_idx in op_indices)
            func_names = {sac_stats.func_names[op_idx] for op_idx in op_indices}
            msps = (mem / runtime) if runtime > 0 else sys.float_info.max
            msps_meta.append(MSPS(func_names, cand_idx, mem, runtime, msps))
        # We choose candidates to be recomputed based on increasing msps
        msps_meta.sort(key=lambda x: x.msps, reverse=True)
        return SACGreedyOrderMeta(
            recomputed_ops, stored_ops, inplace_op_groups, random_ops_group, msps_meta
        )

    def _get_sac_tradeoff_pwlf_stats(
        self,
        sac_stats: SACStats,
        greedy_order_meta: SACGreedyOrderMeta,
        n_segments: int = 2,
        save_tradeoff_graph: bool = False,
        filename: str = "ac_tradeoff",
    ) -> SACTradeOffStats:
        try:
            import numpy as np  # type: ignore[import-not-found]
            import pwlf  # type: ignore[import-untyped, import-not-found]
        except ImportError as err:
            raise ImportError("Please install pwlf and numpy package.") from err

        stored_ops, recomputed_ops, inplace_op_groups, random_ops_group, msps_meta = (
            greedy_order_meta.stored_ops,
            greedy_order_meta.recomputed_ops,
            greedy_order_meta.inplace_op_groups,
            greedy_order_meta.random_ops_group,
            greedy_order_meta.msps_meta,
        )
        # 1. Initialize the discarded memory and recomputation runtime to sum of already chosen recomputed_ops
        recomp_indices: set[int] = set()
        for r_idx in recomputed_ops:
            recomp_indices.add(r_idx)
            if r_idx in inplace_op_groups:
                recomp_indices.update(inplace_op_groups[r_idx])
            if r_idx in random_ops_group:
                recomp_indices.update(random_ops_group[r_idx])

        discarded_mem = sum(sac_stats.memory[op_idx] for op_idx in recomp_indices)
        recomp_runtime = sum(sac_stats.runtimes[op_idx] for op_idx in recomp_indices)
        # 2. Initialize the max recomputation time and total recomputation memory
        sac_runtime = sum(sac_stats.runtimes)
        sac_memory = sum(sac_stats.memory)
        # 3. Tradeoff curve stores the KV pair of the discarded memory to total memory and,
        # recomputation time to total runtime incurred.
        delta = 1e-2
        tradeoff_curve = OrderedDict()
        # 4. Initialize the trade-off curve with the stats of already chosen recomputed_ops
        tradeoff_curve[(discarded_mem / sac_memory) + delta] = (
            recomp_runtime / sac_runtime
        )
        # 5. Update the trade-off curve with memory and runtime stats of SAC candidates in the
        # greedy order of their ``MSPS``.
        for cand in msps_meta:
            discarded_mem += cand.memory
            recomp_runtime += cand.runtime
            tradeoff_curve[(discarded_mem / sac_memory) + delta] = (
                recomp_runtime / sac_runtime
            )
        # 6. Finally, we add the memory and recomputation time of the always stored ops.
        stored_indices: set[int] = set()
        for s_idx in stored_ops:
            stored_indices.add(s_idx)
            if s_idx in inplace_op_groups:
                stored_indices.update(inplace_op_groups[s_idx])
            if s_idx in random_ops_group:
                stored_indices.update(random_ops_group[s_idx])
        discarded_mem += sum(sac_stats.memory[op_idx] for op_idx in stored_indices)
        recomp_runtime += sum(sac_stats.runtimes[op_idx] for op_idx in stored_indices)
        tradeoff_curve[(discarded_mem / sac_memory) + delta] = (
            recomp_runtime / sac_runtime
        )
        x_ = list(tradeoff_curve.keys())
        y_ = list(tradeoff_curve.values())
        # 7. We shift the y values to left and x values to right to upperbound the trade-off function
        # TODO: Write a better explanation why this needs to be done
        x = x_[: len(x_) - 1]
        y = y_[1:]
        tradeoff_pwlf = pwlf.PiecewiseLinFit(x, y)
        # 8. Fit a piecewise linear function with the specified number of segments to the trade-off curve.
        n_segments = max(min(len(x) - 2, n_segments), 1)
        tradeoff_pwlf.fit(n_segments=n_segments)

        # save prediction graph
        def save_prediction_graph(
            pwlf_: pwlf.PiecewiseLinFit, x: list[float], y: list[float], filename: str
        ) -> None:
            try:
                import matplotlib.pyplot as plt  # type: ignore[import-not-found]
                import numpy as np  # type: ignore[import-not-found]
            except ImportError as err:
                raise ImportError(
                    "Install matplotlib and numpy using pip: pip install matplotlib numpy"
                ) from err
            # predict for the determined points
            xHat = np.linspace(min(x), max(x), num=10000)
            yHat = pwlf_.predict(xHat)

            # plot the results
            plt.figure()
            plt.plot(x, y, "o", label="Shifted")
            plt.plot(xHat, yHat, "-", label="Predicted")
            plt.plot(x_, y_, "x", label="Original")
            plt.ylabel("Recomp time / Total recomp time")
            plt.xlabel("Memory discarded / Total memory")
            plt.legend()
            plt.title(f"{filename}")
            plt.suptitle(
                f"Total Memory = {sac_memory} B Total Runtime = {sac_runtime:.4f} ms",
                fontsize=10,
            )
            folder_name = "tradeoff_graphs"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            # Save the plots in the folder
            plt.savefig(os.path.join(folder_name, f"{filename}.png"))

        if save_tradeoff_graph:
            save_prediction_graph(tradeoff_pwlf, x, y, filename)
        # 9. Obtain the slopes, intercepts and breakpoints of the fitted piecewise linear functions
        slopes = tradeoff_pwlf.calc_slopes().tolist()
        if not (
            isinstance(tradeoff_pwlf.intercepts, np.ndarray)
            and isinstance(tradeoff_pwlf.fit_breaks, np.ndarray)
        ):
            raise AssertionError
        intercepts = tradeoff_pwlf.intercepts.tolist()
        fit_breaks = tradeoff_pwlf.fit_breaks.tolist()
        return SACTradeOffStats(
            n_segments=n_segments,
            slopes=slopes,
            intercepts=intercepts,  # type: ignore[arg-type]
            fit_breaks=fit_breaks,  # type: ignore[arg-type]
            tradeoff_curve=tradeoff_curve,
            sac_memory=sac_memory,
            sac_runtime=sac_runtime,
        )

    def display_sac_stats(
        self, sac_stats: SACStats, print_tabular: bool = False
    ) -> None:
        """
        Displays the SAC statistics.

        Args:
            sac_stats (SACStats): The SAC statistics to display.
            print_tabular (bool, optional): Whether to print the statistics in a tabular format. Defaults to False.

        Prints:
            1. Total Memory: The total memory usage in bytes.
            2. Total Runtime: The total runtime in milliseconds.
            3. Store Random: A flag indicating whether to force store random operator results.

            Followed by a table with the following columns:
            1. Op Idx: The operator index.
            2. Op Name: The operator name.
            3. Runtimes (ms): The operator runtime in milliseconds.
            4. Memory (B): The operator memory usage in bytes.
            5. View-like: A flag indicating whether the operator is view-like.
            6. Random: A flag indicating whether the operator is random.
            7. Saved Autograd: A flag indicating whether the operator's result is saved by autograd engine.
            8. In-place: The index of the operator's first parent, or None if not in-place.

        If print_tabular is True, the table is printed in a tabular format.
        Otherwise, the table is printed in a plain text format.
        """
        print(
            f"Total Memory: {sum(sac_stats.memory)} B Total Runtime: {sum(sac_stats.runtimes)} ms"
            f" Store Random: {sac_stats.force_store_random}"
        )
        table_data = []
        op_parent = dict(sac_stats.inplace_ops)
        for i, fn_name in enumerate(sac_stats.func_names):
            row = [
                str(i),
                fn_name,
                f"{sac_stats.runtimes[i]:.4f}",
                str(sac_stats.memory[i]),
                str(i in sac_stats.view_like_ops),
                str(i in sac_stats.rand_ops),
                str(i in sac_stats.saved_autograd_ops),
                str(op_parent.get(i)),
            ]
            table_data.append(row)
        # Define headers
        headers = [
            "Op Idx",
            "Op Name",
            "Runtimes(ms)",
            "Memory (B)",
            "View-like",
            "Random",
            "Saved Autograd",
            "In-place",
        ]
        if print_tabular:
            _display_stats_tabular(headers, table_data)
        else:
            max_widths = [0 for _ in range(len(headers))]
            table_data.insert(0, headers)
            for row in table_data:
                for i, elem in enumerate(row):
                    max_widths[i] = max(max_widths[i], len(elem))
            for row in table_data:
                print(
                    "\t".join(
                        [f"{elem:<{max_widths[i]}}" for i, elem in enumerate(row)]
                    )
                )

    def display_sac_tradeoff_stats(
        self,
        greedy_order_meta: SACGreedyOrderMeta,
        sac_stats: SACStats,
        print_tabular: bool = False,
    ) -> None:
        """
        Displays the SAC trade-off statistics.

        Args:
            greedy_order_meta (SACGreedyOrderMeta): The SAC greedy order metadata.
            sac_stats (SACStats): The SAC statistics.
            print_tabular (bool, optional): Whether to print the statistics in a tabular format. Defaults to False.

        Prints:
            A table with the following columns:
            1. Op Id(s): The operator index(es).
            2. Op Name(s): The operator name(s).
            3. Discarded Mem (%): The percentage of discarded memory.
            4. Discarded Mem (B): The discarded memory in bytes.
            5. Recomp time (%): The percentage of recomputed time.
            6. Recomp time (ms): The recomputed time in milliseconds.
            7. MSPS: The memory per second.
            8. Always Stored: A flag indicating whether the operator is always stored.
            9. Always Recomputed: A flag indicating whether the operator is always recomputed.

        If print_tabular is True, the table is printed in a tabular format.
        Otherwise, the table is printed in a plain text format.
        """
        table_data = []
        total_memory, total_runtime = sum(sac_stats.memory), sum(sac_stats.runtimes)
        discarded_mem: int = 0
        recomp_runtime: float = 0.0

        def append_row(
            op_indices: set[int],
            func_names: set[str],
            msps: float | None = None,
            stored: bool | None = False,
            recomputed: bool | None = False,
        ) -> None:
            row = [
                str(op_indices),
                str(func_names),
                f"{discarded_mem / total_memory:.4f}",
                str(discarded_mem),
                f"{recomp_runtime / total_runtime:.4f}",
                str(recomp_runtime),
                f"{msps:.2e}" if msps is not None else str(nan),
                str(stored),
                str(recomputed),
            ]
            table_data.append(row)

        stored_ops, recomputed_ops, inplace_op_groups, random_ops_group, msps_meta = (
            greedy_order_meta.stored_ops,
            greedy_order_meta.recomputed_ops,
            greedy_order_meta.inplace_op_groups,
            greedy_order_meta.random_ops_group,
            greedy_order_meta.msps_meta,
        )

        for op_idx in recomputed_ops:
            op_indices: set[int] = {op_idx}
            if op_idx in inplace_op_groups:
                op_indices.update(inplace_op_groups[op_idx])
            if op_idx in random_ops_group:
                op_indices.update(random_ops_group[op_idx])
            discarded_mem += sum(sac_stats.memory[i] for i in op_indices)
            recomp_runtime += sum(sac_stats.runtimes[i] for i in op_indices)
            func_names = {sac_stats.func_names[i] for i in op_indices}
            append_row(op_indices, func_names, recomputed=True)

        for cand in msps_meta:
            discarded_mem += cand.memory
            recomp_runtime += cand.runtime
            op_indices = {cand.op_idx}
            if cand.op_idx in inplace_op_groups:
                op_indices.update(inplace_op_groups[cand.op_idx])
            if cand.op_idx in random_ops_group:
                op_indices.update(random_ops_group[cand.op_idx])
            append_row(op_indices, cand.func_names, msps=cand.msps)

        for op_idx in stored_ops:
            op_indices = {op_idx}
            if op_idx in inplace_op_groups:
                op_indices.update(inplace_op_groups[op_idx])
            if op_idx in random_ops_group:
                op_indices.update(random_ops_group[op_idx])
            discarded_mem += sum(sac_stats.memory[i] for i in op_indices)
            recomp_runtime += sum(sac_stats.runtimes[i] for i in op_indices)
            func_names = {sac_stats.func_names[i] for i in op_indices}
            append_row(op_indices, func_names, stored=True)

        headers = [
            "Op Id(s)",
            "Op Name(s)",
            "Discarded Mem (%)",
            "Discarded Mem (B)",
            "Recomp time (%)",
            "Recomp time (ms)",
            "MSPS",
            "Always Stored",
            "Always Recomputed",
        ]
        if print_tabular:
            _display_stats_tabular(headers, table_data)
        else:
            max_widths = [0 for _ in range(len(headers))]
            table_data.insert(0, headers)
            for row in table_data:
                for i, elem in enumerate(row):
                    max_widths[i] = max(max_widths[i], len(elem))
            for row in table_data:
                print(
                    "\t".join(
                        [f"{elem:<{max_widths[i]}}" for i, elem in enumerate(row)]
                    )
                )

    def pwlf_sac_tradeoff_curve(
        self,
        n_segments: int = 2,
        save_tradeoff_graphs: bool = False,
    ) -> None:
        """
        Fits a piecewise linear function with the specified sumber of segments to the SAC trade-off curve of
        discarded memory vs recomputation time.

        Args:
            n_segments (int, optional): The number of segments to be used for fitting the piecewise linear function to
                the trade-off curve. Defaults to 2.
            save_tradeoff_graphs (bool, optional): Whether to save the trade-off graphs to file. Defaults to False.

        If save_tradeoff_graphs is True, the trade-off graphs are saved to file using the module FQN as the filename.
        """
        for mod_fqn, sac_stats in self.sac_mod_stats.items():
            self.sac_mod_tradeoff_stats[mod_fqn] = self._get_sac_tradeoff_pwlf_stats(
                sac_stats=sac_stats,
                greedy_order_meta=self.sac_mod_greedy_order_meta[mod_fqn],
                n_segments=n_segments,
                save_tradeoff_graph=save_tradeoff_graphs,
                filename=mod_fqn,
            )

    def display_modulewise_sac_stats(
        self, depth: int = 2, print_tabular: bool = False
    ) -> None:
        """
        Displays the SAC and trade-off statistics for each module.

        Args:
            depth (int, optional): The maximum depth of modules to display. Defaults to 2.
            print_tabular (bool, optional): Whether to print the statistics in a tabular format. Defaults to False.

        Prints:
            For each module with depth less than or equal to the specified depth:
            1. The SAC statistics for the module (using display_sac_stats).
            2. The SAC trade-off statistics for the module (using display_sac_tradeoff_stats).

        If print_tabular is True, the statistics are printed in a tabular format.
        Otherwise, the statistics are printed in a plain text format.
        """
        for mod_fqn, sac_stats in self.sac_mod_stats.items():
            mod_depth = mod_fqn.count(".") + 1
            if mod_depth > depth:
                continue
            print(f"Module: {mod_fqn}")
            self.display_sac_stats(sac_stats, print_tabular)
            print(f"AC Trade-off for Module: {mod_fqn} MSPS = Memory/Runtime")
            self.display_sac_tradeoff_stats(
                self.sac_mod_greedy_order_meta[mod_fqn], sac_stats, print_tabular
            )

    def __call__(self, estimate_mode_type: str) -> Self:
        """
        Sets the estimate mode type.

        Currently supported modes:
            - "operator-level-benchmark": Estimates runtime using operator benchmarking.
            - "operator-level-cost-model": Estimates runtime using roofline cost model.

        Args:
            estimate_mode_type (str): The type of estimate mode to use.

        Returns:
            SACEstimator: The SAC estimator instance.

        Raises:
            NotImplementedError: If the estimate mode type is not supported.
        """
        if estimate_mode_type == "operator-level-benchmark":
            self._estimate_runtime = RuntimeEstimator._benchmark_estimate
        elif estimate_mode_type == "operator-level-cost-model":
            self._estimate_runtime = RuntimeEstimator._roofline_estimate
        else:
            raise NotImplementedError(
                f"estimate_mode_type {estimate_mode_type} not supported"
            )
        return self

    def __enter__(self) -> Self:  # type: ignore[no-untyped-def]
        fake_mode = active_fake_mode()
        if not isinstance(fake_mode, FakeTensorMode):
            raise AssertionError("SAC Estimator should be called in FakeTensorMode")
        RuntimeEstimator.fake_mode = fake_mode
        RuntimeEstimator.gpu_type = self._gpu_type
        self._mod_tracker.register_user_hooks(
            pre_fw_hook=self._pre_fw_hook,
            post_fw_hook=self._post_fw_hook,
        )
        self._mod_tracker.__enter__()
        self._saved_tensor_hook_ctx.__enter__()
        return super().__enter__()

    def __exit__(self, *args: Any) -> None:  # type: ignore[no-untyped-def]
        self._saved_tensor_hook_ctx.__exit__()
        self._mod_tracker.__exit__(*args)
        super().__exit__(*args)
<|endoftext|>import warnings

import torch
from torch._custom_class_base import CustomClassBase
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


def get_untyped_storages(t: torch.Tensor) -> set[torch.UntypedStorage]:
    """
    Recursively extracts untyped storages from a tensor or its subclasses.

    Args:
        t (torch.Tensor): The tensor to extract storages from.

    Returns:
        Set[torch.UntypedStorage]: A set of untyped storages.
    """
    unflattened_tensors = [t]
    flattened_tensor_storages = set()
    while len(unflattened_tensors) > 0:
        obj = unflattened_tensors.pop()
        if is_traceable_wrapper_subclass(obj):
            attrs, _ = obj.__tensor_flatten__()
            for attr in attrs:
                match getattr(obj, attr):
                    case torch.Tensor() as v:
                        unflattened_tensors.append(v)
                    case CustomClassBase():
                        pass
                    case unexpected:
                        raise AssertionError(
                            f"expected Tensor or CustomClassBase, got {type(unexpected)}"
                        )
        else:
            if not hasattr(obj, "untyped_storage"):
                warnings.warn(
                    f"Expected a tensor or a traceable wrapper-subclass of tensor, but got {type(obj)}",
                    category=UserWarning,
                    stacklevel=2,
                )
            else:
                flattened_tensor_storages.add(obj.untyped_storage())
    return flattened_tensor_storages
<|endoftext|># mypy: allow-untyped-defs
import contextlib
import logging
import math
import warnings
from collections.abc import Callable, Generator, Iterator
from typing import Any, cast, no_type_check

import torch
import torch.distributed as dist
import torch.distributed.algorithms._checkpoint.checkpoint_wrapper as checkpoint_wrapper
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._shard.sharded_tensor import (
    init_from_local_shards,
    Shard,
    ShardedTensor,
)
from torch.distributed.fsdp._common_utils import (
    _FSDPState,
    _get_module_fsdp_state_if_fully_sharded_module,
    _has_fsdp_params,
    _is_composable,
    _module_handle,
    clean_tensor_name,
    FSDP_PREFIX,
    FSDP_WRAPPED_MODULE,
)
from torch.distributed.fsdp._debug_utils import SimpleProfiler
from torch.distributed.fsdp._runtime_utils import (
    _cast_buffers_to_dtype_and_device,
    _get_orig_buffer_dtypes,
    _lazy_init,
    _reset_flat_param_grad_info_if_needed,
)
from torch.distributed.fsdp.api import (
    FullStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.tensor import DTensor
from torch.distributed.utils import _replace_by_prefix

from ._fsdp_extensions import (
    _ext_all_gather_dtensor,
    _ext_chunk_dtensor,
    _ext_chunk_tensor,
    _ext_post_unflatten_transform,
    _ext_pre_load_state_dict_transform,
)
from ._unshard_param_utils import _unshard_fsdp_state_params, FLAT_PARAM


logger = logging.getLogger(__name__)


def _should_unshard_params(fsdp_state: _FSDPState) -> bool:
    return not (
        fsdp_state.sharding_strategy == ShardingStrategy.NO_SHARD
        and (_is_composable(fsdp_state) or fsdp_state._use_orig_params)
    )


def _convert_to_wrapped_module_name(module_name: str) -> str:
    module_name = module_name.replace(f"{FSDP_PREFIX}", "")
    module_name = module_name.replace(f"{FSDP_WRAPPED_MODULE}", "")
    if module_name:
        module_name = f"{module_name}."
    # `CheckpointWrapper` adds a prefix that has to be removed as well.
    module_name = module_name.replace(checkpoint_wrapper._CHECKPOINT_PREFIX, "")
    return module_name


def _param_name_infos(
    module: nn.Module, fsdp_state: _FSDPState
) -> Iterator[tuple[str, str, str]]:
    if not _has_fsdp_params(fsdp_state, module):
        return
    for param_name, module_name in _module_handle(
        fsdp_state, module
    ).param_module_names():
        module_name = _convert_to_wrapped_module_name(module_name)
        fqn = f"{module_name}{param_name}"
        yield fqn, param_name, module_name


def _shared_param_name_infos(
    module: nn.Module, fsdp_state
) -> Iterator[tuple[str, str, str]]:
    for param_name, module_name in _module_handle(
        fsdp_state, module
    ).shared_param_module_names():
        module_name = _convert_to_wrapped_module_name(module_name)
        fqn = f"{module_name}{param_name}"
        yield fqn, param_name, module_name


@no_type_check
def _enter_unshard_params_ctx(
    module: nn.Module,
    fsdp_state: _FSDPState,
    writeback: bool = False,
    rank0_only: bool = False,
    offload_to_cpu: bool = False,
    with_grads: bool = False,
) -> None:
    """
    state_dict hooks cannot use the pure context call as the checkpoint flow
    requires to enter the context in the pre-hook but leave the context in the
    post-hook. This API enters the context of ``_unshard_fsdp_state_params``.
    """
    if module in fsdp_state._unshard_params_ctx:
        raise AssertionError(
            "Entering the ``_unshard_fsdp_state_params`` context but _unshard_params_ctx[module] "
            "is not None."
        )
    fsdp_state._unshard_params_ctx[module] = _unshard_fsdp_state_params(
        module,
        fsdp_state,
        writeback=writeback,
        rank0_only=rank0_only,
        offload_to_cpu=offload_to_cpu,
        with_grads=with_grads,
    )
    fsdp_state._unshard_params_ctx[module].__enter__()


@no_type_check
def _exit_unshard_params_ctx(module: nn.Module, fsdp_state: _FSDPState) -> None:
    """A helper function to exit ``_unshard_fsdp_state_params`` context."""
    fsdp_state._unshard_params_ctx[module].__exit__(None, None, None)
    fsdp_state._unshard_params_ctx.pop(module)


def _common_pre_state_dict_hook(
    module: nn.Module,
    fsdp_state: _FSDPState,
) -> None:
    """Performs the pre-state_dict tasks shared by all state_dict types."""
    if fsdp_state._device_handle.is_available():
        fsdp_state._device_handle.synchronize()
    # TODO: need to check if this is always correct for composable FSDP.
    _lazy_init(fsdp_state, module)
    if fsdp_state._is_root:
        _reset_flat_param_grad_info_if_needed(fsdp_state._all_handles)


def _common_unshard_pre_state_dict_hook(
    module: nn.Module,
    fsdp_state: _FSDPState,
    offload_to_cpu: bool,
    rank0_only: bool,
) -> None:
    """
    Performs the pre-state_dict tasks shared by all state_dict types that require
    ``_unshard_fsdp_state_params()``. FULL_STATE_DICT and SHARDED_STATE_DICT use this hook.
    """
    # For composable `fully_shard`, it does not need to unshard parameters for `NO_SHARD` cases.
    if not _should_unshard_params(fsdp_state):
        return
    _enter_unshard_params_ctx(
        module,
        fsdp_state,
        writeback=False,
        offload_to_cpu=offload_to_cpu,
        rank0_only=rank0_only,
    )


@no_type_check
def _common_unshard_post_state_dict_hook(
    module: nn.Module,
    fsdp_state: _FSDPState,
    state_dict: dict[str, Any],
    prefix: str,
    param_hook: Callable,
) -> dict[str, Any]:
    """
    The post-state_dict flow that shared by all state_dict types that require
    ``_unshard_fsdp_state_params()``. FULL_STATE_DICT and SHARDED_STATE_DICT use this
    hook.
    """
    _replace_by_prefix(state_dict, prefix + f"{FSDP_PREFIX}", prefix)
    # Return early for trivial cases
    if not state_dict or not _has_fsdp_params(fsdp_state, module):
        if _should_unshard_params(fsdp_state):
            _exit_unshard_params_ctx(module, fsdp_state)
        return state_dict

    # If a rank does not have unsharded parameters(when `rank0_only=True`
    # and `rank != 0`), then the rank only needed to participate in the
    # all-gather and does not need to save the # state dict. We simply check
    # rank0_only to ensure this issue.
    rank0_only = (
        fsdp_state._state_dict_type == StateDictType.FULL_STATE_DICT
        and cast(FullStateDictConfig, fsdp_state._state_dict_config).rank0_only
    )
    # no_fsdp_return means the state_dict returned by this rank should contain
    # only non-FSDP controlled parameters and buffers.
    no_fsdp_return = rank0_only and fsdp_state.rank != 0
    if no_fsdp_return and not fsdp_state._use_orig_params:
        for clean_key in fsdp_state._buffer_names:
            # This is a hack to support activation checkpoint.
            clean_key = clean_key.replace(
                f"{checkpoint_wrapper._CHECKPOINT_PREFIX}.", ""
            )
            state_dict.pop(f"{prefix}{clean_key}", None)
        # Non-zero ranks have flat_param key when rank0_only=True, because rank0_only=True is
        # passed in to unshard context, but nonzero ranks reshard early, causing this flat_param
        # to appear in state_dict.
        state_dict.pop(f"{prefix}{FLAT_PARAM}")
        _exit_unshard_params_ctx(module, fsdp_state)
        return state_dict

    # Loop only the parameters saved in this instance's wrapped module to
    # avoid processing buffers.
    for fqn, param_name, module_name in _param_name_infos(module, fsdp_state):
        fqn = f"{prefix}{fqn}"
        if no_fsdp_return:
            state_dict.pop(fqn)
            continue
        if fqn not in state_dict:
            raise AssertionError(
                f"FSDP assumes {fqn} is in the state_dict but the state_dict only "
                f"has {state_dict.keys()}. "
                f"prefix={prefix}, module_name={module_name}, "
                f"param_name={param_name} rank={fsdp_state.rank}."
            )

        param_hook(state_dict, prefix, fqn)

    if _should_unshard_params(fsdp_state):
        _exit_unshard_params_ctx(module, fsdp_state)

    cpu_device = torch.device("cpu")
    buffer_clean_fqns = []
    buffers = []
    for clean_key in fsdp_state._buffer_names:
        # This is a hack to support activation checkpoint.
        clean_key = clean_tensor_name(clean_key)
        fqn = f"{prefix}{clean_key}"
        if fqn not in state_dict:
            # A buffer can be registered as non-persistent.
            continue
        if no_fsdp_return:
            state_dict.pop(fqn)
        else:
            buffer = state_dict[fqn]
            if (
                fsdp_state._state_dict_config.offload_to_cpu
                and buffer.device != cpu_device
            ):
                state_dict[fqn] = buffer.to(cpu_device)
            # skip upcasting for ignored buffers
            if clean_key not in fsdp_state._ignored_buffer_names:
                buffer_clean_fqns.append(clean_key)
                buffers.append(state_dict[fqn])

    if buffers:
        mixed_precision_enabled_for_buffers = (
            fsdp_state._mixed_precision_enabled_for_buffers()
            if not _is_composable(fsdp_state)
            else (fsdp_state.mixed_precision.buffer_dtype is not None)
        )
        if mixed_precision_enabled_for_buffers:
            buffer_dtypes = _get_orig_buffer_dtypes(fsdp_state, buffer_clean_fqns)
            _cast_buffers_to_dtype_and_device(
                buffers, buffer_dtypes, fsdp_state.compute_device
            )
            for buffer, clean_fqn in zip(buffers, buffer_clean_fqns):
                fqn = f"{prefix}{clean_fqn}"
                logger.info("FSDP is casting the dtype of %s to %s", fqn, buffer.dtype)
                state_dict[fqn] = buffer.clone()
    return state_dict


@no_type_check
def _full_pre_state_dict_hook(
    fsdp_state: _FSDPState,
    module: nn.Module,
    *args,
    **kwargs,
) -> None:
    """
    Hook that runs before model.state_dict() is called. pre-state_dict hook is
    not actually supported by ``nn.Module``. As a result, this API is called
    from ``_full_post_state_dict_hook()`` to simulate the case. Once pre-state_dict
    is supported in ``nn.Module``, this hook will be registered as a hook in
    ``nn.Module``.
    """
    if getattr(fsdp_state, "_device_mesh", False):
        fsdp_state._device_mesh._get_root_mesh()

    _common_pre_state_dict_hook(module, fsdp_state)
    _common_unshard_pre_state_dict_hook(
        module,
        fsdp_state,
        offload_to_cpu=fsdp_state._state_dict_config.offload_to_cpu,
        rank0_only=cast(FullStateDictConfig, fsdp_state._state_dict_config).rank0_only,
    )


@no_type_check
def _full_post_state_dict_hook(
    module: nn.Module,
    fsdp_state: _FSDPState,
    state_dict: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    """
    Hook that runs after model.state_dict() is called before returning result to
    user. For FSDP, we may have to clone the tensors in state_dict as params go
    back to sharded version after _unshard_fsdp_state_params ends, and also remove
    the ``FSDP_WRAPPED_MODULE`` prefix.
    """

    def param_hook(
        state_dict: dict[str, Any],
        prefix: str,
        fqn: str,
    ) -> None:
        clean_key = fqn
        clean_prefix = clean_tensor_name(prefix)
        # Strip prefix out of key if needed as buffer names and param names
        # do not have prefix considered as they are not computed in `state_dict`
        # call.
        clean_key = clean_key.removeprefix(clean_prefix)

        # Clone parameters before exiting the `_unshard_fsdp_state_params()` context.
        if not getattr(state_dict[fqn], "_has_been_cloned", False):
            try:
                state_dict[fqn] = state_dict[fqn].detach().clone()
                state_dict[fqn]._has_been_cloned = True  # type: ignore[attr-defined]
            except BaseException as e:
                warnings.warn(
                    f"Failed to clone() tensor with name {fqn} on rank {fsdp_state.rank}. "
                    "This may mean that this state_dict entry could point to invalid "
                    "memory regions after returning from state_dict() call if this "
                    "parameter is managed by FSDP. Please check clone "
                    f"implementation of {fqn}. Error: {str(e)}",
                    stacklevel=2,
                )

    return _common_unshard_post_state_dict_hook(
        module, fsdp_state, state_dict, prefix, param_hook
    )


def _full_pre_load_state_dict_hook(
    module: nn.Module,
    fsdp_state: _FSDPState,
    state_dict: dict[str, Any],
    prefix: str,
) -> None:
    _lazy_init(fsdp_state, module)
    if _should_unshard_params(fsdp_state):
        with SimpleProfiler.profile("_enter_unshard_params_ctx"):
            _enter_unshard_params_ctx(module, fsdp_state, writeback=True)
    # Add FSDP_PREFIX only for wrapper-based FSDP.
    if not _is_composable(fsdp_state):
        _replace_by_prefix(state_dict, prefix, prefix + f"{FSDP_PREFIX}")


def _full_post_load_state_dict_hook(
    module: nn.Module, fsdp_state: _FSDPState, *args, **kwargs
) -> None:
    if _should_unshard_params(fsdp_state):
        with SimpleProfiler.profile("_exit_unshard_params_ctx"):
            _exit_unshard_params_ctx(module, fsdp_state)


def _local_pre_state_dict_hook(
    fsdp_state: _FSDPState,
    module: nn.Module,
    *args,
    **kwargs,
) -> None:
    """
    Hook that runs before model.state_dict() is called. Right now, pre-state_dict
    hook is not supported by the PyTorch core. So this API is called from
    `_local_post_state_dict_hook()` to simulate the case.
    """
    if (
        _has_fsdp_params(fsdp_state, module)
        and not _module_handle(fsdp_state, module).uses_sharded_strategy
    ):
        raise RuntimeError(
            "``local_state_dict`` can only be used when parameters are flatten "
            "and sharded."
        )
    _common_pre_state_dict_hook(module, fsdp_state)


@no_type_check
def _local_post_state_dict_hook(
    module: nn.Module,
    fsdp_state: _FSDPState,
    state_dict: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    """
    This hook create a ShardedTensor from the local flat_param and replace
    the state_dict[f"{prefix}{FLAT_PARAM}] with the ShardedTensor. No copy
    will happen. The underlying storage is the same.
    """

    _replace_by_prefix(state_dict, f"{prefix}{FSDP_PREFIX}", prefix)
    if not _has_fsdp_params(fsdp_state, module):
        return state_dict

    # state_dict[f"{prefix}{FLAT_PARAM}"] exists and has the same tensor
    # value as the flat_param but it is a pure Tensor because
    # nn.Module.state_dict() will detach the parameter. Therefore, we need
    # to get flat_param to get the metadata.
    if not _module_handle(fsdp_state, module):
        raise AssertionError("Should have returned early")
    flat_param = _module_handle(fsdp_state, module).flat_param
    # Constructs a ShardedTensor from the flat_param "without" padding.
    # Removing the padding allows users to change the number of ranks
    # when loading the local_state_dict.
    full_numel = flat_param._unpadded_unsharded_size.numel()  # type: ignore[attr-defined]
    shard_offset = flat_param.numel() * fsdp_state.rank
    valid_data_size = flat_param.numel() - flat_param._shard_numel_padded
    if valid_data_size > 0:
        # If FlatParameter is returned, FlatParameter._local_shard cause a
        # pickling issue (can be torch.save but not torch.load). Since there
        # is no benefit for state_dict to return the actual FlatParameter class,
        # a view (which is a tensor) of the FlatParameter will be returned.
        flat_param = flat_param[:valid_data_size].view(valid_data_size)
        local_shards = [
            Shard.from_tensor_and_offsets(flat_param, [shard_offset], fsdp_state.rank)
        ]
    else:
        local_shards = []
    sharded_tensor = init_from_local_shards(
        local_shards, full_numel, process_group=fsdp_state.process_group
    )  # type: ignore[assignment]
    # TODO: Add DTensor state_dict support for LOCAL_STATE_DICT.
    if fsdp_state._state_dict_config.offload_to_cpu:
        sharded_tensor = sharded_tensor.cpu()
    state_dict[f"{prefix}{FLAT_PARAM}"] = sharded_tensor
    return state_dict


def _local_post_load_state_dict_hook(
    module: nn.Module, fsdp_state: _FSDPState, *args, **kwargs
) -> None:
    pass


def _local_pre_load_state_dict_hook(
    module: nn.Module,
    fsdp_state: _FSDPState,
    state_dict: dict[str, Any],
    prefix: str,
) -> None:
    """
    This hook finds the local flat_param for this FSDP module from the
    state_dict. The flat_param should be a ShardedTensor. This hook converts
    the ShardedTensor to a tensor. No copy happen unless padding is required.
    """
    _lazy_init(fsdp_state, module)
    _replace_by_prefix(state_dict, prefix, f"{prefix}{FSDP_PREFIX}")
    fqn = f"{prefix}{FSDP_PREFIX}{FLAT_PARAM}"
    if fqn not in state_dict:
        if _has_fsdp_params(fsdp_state, module):
            raise AssertionError(
                "No `FlatParameter` in `state_dict` for this FSDP instance "
                "but it has parameters"
            )
        return
    load_tensor = state_dict[fqn]
    if not isinstance(load_tensor, ShardedTensor):
        raise AssertionError("Tensors in local_state_dict should be ShardedTensor.")

    # Convert the ShardedTensor to a Tensor.
    flat_param = _module_handle(fsdp_state, module).flat_param
    if flat_param is None:
        raise AssertionError("Expected flat_param to be set")
    valid_data_size = flat_param.numel() - flat_param._shard_numel_padded
    shards = load_tensor.local_shards()
    if valid_data_size > 0:
        if not len(shards):
            raise AssertionError(
                "load_local_state_dict assume one shard per ShardedTensor."
            )
        load_tensor = shards[0].tensor

        # Get the metadata of the flat_param to decide whether to pad the loaded
        # tensor.
        if flat_param._shard_numel_padded > 0:
            if load_tensor.numel() >= flat_param.numel():
                raise AssertionError(
                    f"Local shard size = {flat_param.numel()} and the tensor in "
                    f"the state_dict is {load_tensor.numel()}."
                )
            load_tensor = F.pad(load_tensor, [0, flat_param._shard_numel_padded])
    else:
        load_tensor = flat_param
    # TODO: Add DTensor state_dict support for LOCAL_STATE_DICT.
    state_dict[fqn] = load_tensor


def _sharded_pre_state_dict_hook(
    fsdp_state: _FSDPState,
    module: nn.Module,
    *args,
    **kwargs,
) -> None:
    """
    Hook that runs before model.state_dict() is called. Check
    ``_full_pre_load_state_dict_hook`` for the detail.
    """
    if (
        _has_fsdp_params(fsdp_state, module)
        and not _module_handle(fsdp_state, module).uses_sharded_strategy
    ):
        raise RuntimeError(
            "``sharded_state_dict`` can only be used when parameters are flatten "
            "and sharded."
        )
    _common_pre_state_dict_hook(module, fsdp_state)
    # Setting offload_to_cpu here does not work even if offload_to_cpu is True.
    # We have to create ShardedTensor first then move it to CPU.
    _common_unshard_pre_state_dict_hook(
        module,
        fsdp_state,
        offload_to_cpu=False,
        rank0_only=False,
    )


@no_type_check
def _sharded_post_state_dict_hook(
    module: nn.Module,
    fsdp_state: _FSDPState,
    state_dict: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    """
    The hook replaces the unflattened, unsharded parameter in the state_dict
    with a unflattened, sharded parameter (a ShardedTensor).
    """

    def param_hook(state_dict: dict[str, Any], prefix: str, fqn: str):
        param = state_dict[fqn]
        if not fsdp_state._state_dict_config._use_dtensor:
            sharded_tensor = _ext_chunk_tensor(
                tensor=param,
                rank=fsdp_state.rank,
                world_size=fsdp_state.world_size,
                num_devices_per_node=fsdp_state._device_handle.device_count(),
                pg=fsdp_state.process_group,
                fsdp_extension=fsdp_state._fsdp_extension,
            )
        else:
            sharded_tensor = _ext_chunk_dtensor(
                tensor=param,
                rank=fsdp_state.rank,
                device_mesh=fsdp_state._device_mesh,
                fsdp_extension=fsdp_state._fsdp_extension,
            )
        if fsdp_state._state_dict_config.offload_to_cpu:
            sharded_tensor = sharded_tensor.cpu()
        state_dict[fqn] = sharded_tensor

    return _common_unshard_post_state_dict_hook(
        module, fsdp_state, state_dict, prefix, param_hook
    )


@no_type_check
def _sharded_post_load_state_dict_hook(
    module: nn.Module, fsdp_state: _FSDPState, *args, **kwargs
) -> None:
    if _has_fsdp_params(fsdp_state, module):
        with SimpleProfiler.profile("_exit_unshard_params_ctx"):
            _exit_unshard_params_ctx(module, fsdp_state)


@no_type_check
def _sharded_pre_load_state_dict_hook(
    module: nn.Module,
    fsdp_state: _FSDPState,
    state_dict: dict[str, Any],
    prefix: str,
) -> None:
    """
    The hook combines the unflattened, sharded parameters (ShardedTensor) to
    a new FlatParameter and shards the new FlatParameter to the local chunk.
    """
    _lazy_init(fsdp_state, module)
    if not _is_composable(fsdp_state):
        _replace_by_prefix(state_dict, prefix, prefix + f"{FSDP_PREFIX}")
    if not _has_fsdp_params(fsdp_state, module):
        return

    handle = _module_handle(fsdp_state, module)
    if not handle.uses_sharded_strategy:
        raise RuntimeError(
            "load_sharded_state_dict can only be called when parameters "
            "are flattened and sharded."
        )
    fqn_to_param_ext = dict(
        zip(handle.flat_param._fqns, handle.flat_param._param_extensions)
    )

    for fqn, _, _ in _param_name_infos(module, fsdp_state):
        if not _is_composable(fsdp_state):
            fqn_from_global_root = f"{prefix}{FSDP_PREFIX}{fqn}"
        else:
            fqn_from_global_root = f"{prefix}{fqn}"
        try:
            param = state_dict.pop(fqn_from_global_root)
        except KeyError:
            logger.warning(
                f"Did not find param with FQN {fqn_from_global_root}, skipping it. "  # noqa: G004
                "The weight will not be filled if you expect it to be."
            )
            continue  # TODO: Improve unittesting for state_dict finetuning
            # cases: https://github.com/pytorch/pytorch/issues/109134

        if not fsdp_state._state_dict_config._use_dtensor:
            # All-gather the param (ShardedTensor)
            param, shards = _ext_pre_load_state_dict_transform(
                param, fsdp_state._fsdp_extension
            )

            if len(shards) >= 2:
                raise AssertionError(
                    "Expects 0 or 1 shard per rank "
                    f"but got {len(shards)} shards on rank {fsdp_state.rank}."
                )
            param_numel = param.size().numel()
            dim_0_size = param.size()[0]
            chunk_size = (
                math.ceil(dim_0_size / fsdp_state.world_size)
                * param_numel
                // dim_0_size
            )
            if len(shards) == 1:
                local_tensor = shards[0].tensor.flatten()
                with SimpleProfiler.profile(SimpleProfiler.Type.H2D):
                    local_tensor = local_tensor.to(fsdp_state.compute_device)
                num_padding = chunk_size - local_tensor.numel()
                if num_padding > 0:
                    local_tensor = F.pad(local_tensor, [0, num_padding])
            else:
                local_tensor = torch.zeros(
                    chunk_size, dtype=param.dtype, device=fsdp_state.compute_device
                )
            tensor = torch.empty(
                chunk_size * fsdp_state.world_size,
                dtype=local_tensor.dtype,
                device=fsdp_state.compute_device,
            )
            with SimpleProfiler.profile(SimpleProfiler.Type.ALLGATHER):
                dist.all_gather_single(
                    tensor, local_tensor, group=fsdp_state.process_group
                )
            tensor = tensor.narrow(0, 0, param_numel).reshape(param.size())
            state_dict[fqn_from_global_root] = tensor
        else:
            if param.device != fsdp_state._device_mesh.device_type:
                param = param.to(fsdp_state._device_mesh.device_type)

            root_mesh = fsdp_state._device_mesh._get_root_mesh()
            local_tensor = _ext_all_gather_dtensor(
                param, root_mesh, fsdp_state._fsdp_extension
            )

            if fqn_to_param_ext.get(fqn) is not None:
                ext = fqn_to_param_ext[fqn]
                local_tensor = _ext_post_unflatten_transform(
                    local_tensor, ext, fsdp_state._fsdp_extension
                )
            state_dict[fqn_from_global_root] = local_tensor

    with SimpleProfiler.profile("_enter_unshard_params_ctx"):
        _enter_unshard_params_ctx(module, fsdp_state, writeback=True)


@contextlib.contextmanager
def _replace_with_full_state_dict_type(fsdp_state: _FSDPState) -> Generator:
    old_state_dict_config = fsdp_state._state_dict_config
    old_state_dict_type = fsdp_state._state_dict_type
    fsdp_state._state_dict_config = FullStateDictConfig()
    fsdp_state._state_dict_type = StateDictType.FULL_STATE_DICT
    yield
    fsdp_state._state_dict_config = old_state_dict_config
    fsdp_state._state_dict_type = old_state_dict_type


@no_type_check
@torch.no_grad()
def _post_state_dict_hook(
    module: nn.Module,
    state_dict: dict[str, Any],
    prefix: str,
    *args: Any,
) -> dict[str, Any]:
    """
    _post_state_dict_hook() is called after the state_dict() of this
    FSDP module is executed. ``fsdp_state._state_dict_type`` is used to decide
    what postprocessing will be done.
    """
    fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(module)
    if fsdp_state.sharding_strategy == ShardingStrategy.NO_SHARD:
        context = _replace_with_full_state_dict_type(fsdp_state)
        warnings.warn(
            "When using ``NO_SHARD`` for ``ShardingStrategy``, full_state_dict will "
            "be returned.",
            stacklevel=2,
        )
    else:
        context = contextlib.nullcontext()

    with context:
        _post_state_dict_hook_fn = {
            StateDictType.FULL_STATE_DICT: _full_post_state_dict_hook,
            StateDictType.LOCAL_STATE_DICT: _local_post_state_dict_hook,
            StateDictType.SHARDED_STATE_DICT: _sharded_post_state_dict_hook,
        }
        processed_state_dict = _post_state_dict_hook_fn[fsdp_state._state_dict_type](
            module, fsdp_state, state_dict, prefix
        )

    if fsdp_state._is_root:
        logger.info("FSDP finished processing state_dict(), prefix=%s", prefix)
        for key, tensor in sorted(processed_state_dict.items()):
            if key.startswith(prefix) and isinstance(tensor, torch.Tensor):
                local_shape = tensor.shape
                device = None
                if isinstance(tensor, ShardedTensor):
                    local_shape = None
                    shards = tensor.local_shards()
                    if shards:
                        local_shape = shards[0].tensor.shape
                        device = shards[0].tensor.device
                elif isinstance(tensor, DTensor):
                    local_shape = tensor.to_local().shape
                    device = tensor.device
                else:
                    device = tensor.device
                logger.info(
                    "FQN=%s: type=%s, shape=%s, local_shape=%s, dtype=%s, device=%s",
                    key,
                    type(tensor),
                    tensor.shape,
                    local_shape,
                    tensor.dtype,
                    device,
                )

    return processed_state_dict


@no_type_check
@torch.no_grad()
def _pre_state_dict_hook(
    module: nn.Module,
    *args,
    **kwargs,
) -> None:
    """
    This is called before the core state dict saving logic of ``module``.
    ``fsdp_state._state_dict_type`` is used to decide what postprocessing will
    be done.
    """
    fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(module)
    if fsdp_state.sharding_strategy == ShardingStrategy.NO_SHARD:
        context = _replace_with_full_state_dict_type(fsdp_state)
        warnings.warn(
            "When using ``NO_SHARD`` for ``ShardingStrategy``, full_state_dict will "
            "be returned.",
            stacklevel=2,
        )
    else:
        _set_use_dtensor(fsdp_state)
        context = contextlib.nullcontext()

    with context:
        _pre_state_dict_hook_fn = {
            StateDictType.FULL_STATE_DICT: _full_pre_state_dict_hook,
            StateDictType.LOCAL_STATE_DICT: _local_pre_state_dict_hook,
            StateDictType.SHARDED_STATE_DICT: _sharded_pre_state_dict_hook,
        }
        _pre_state_dict_hook_fn[fsdp_state._state_dict_type](
            fsdp_state,
            module,
            *args,
            **kwargs,
        )


@no_type_check
def _set_use_dtensor(fsdp_state: _FSDPState) -> None:
    # If device_mesh is passed in when initializing FSDP, we automatically turn the
    # _use_dtensor flag to be true for ShardedStateDictConfig().
    if getattr(fsdp_state, "_device_mesh", None):
        state_dict_type = fsdp_state._state_dict_type
        if state_dict_type == StateDictType.LOCAL_STATE_DICT:
            raise RuntimeError(
                "Found state_dict_type LOCAL_STATE_DICT. "
                "DeviceMesh is not compatible with LOCAL_STATE_DICT. "
                "Please set state_dict_type to SHARDED_STATE_DICT to get DTensor state_dict."
            )
        else:
            fsdp_state._state_dict_config._use_dtensor = True


@no_type_check
@torch.no_grad()
def _pre_load_state_dict_hook(
    module: nn.Module,
    state_dict: dict[str, Any],
    prefix: str,
    *args: Any,
) -> None:
    """
    This is called before ``module._load_from_state_dict()``.
    ``fsdp_state._state_dict_type`` is used to decide what preprocessing will
    be done.
    """
    fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(module)
    if fsdp_state.sharding_strategy == ShardingStrategy.NO_SHARD:
        context = _replace_with_full_state_dict_type(fsdp_state)
        warnings.warn(
            "When using ``NO_SHARD`` for ``ShardingStrategy``, full_state_dict will"
            "be returned.",
            stacklevel=2,
        )
    else:
        _set_use_dtensor(fsdp_state)
        context = contextlib.nullcontext()

    _lazy_init(fsdp_state, module)
    if fsdp_state._is_root:
        SimpleProfiler.reset()

    with context:
        _pre_load_state_dict_hook_fn = {
            StateDictType.FULL_STATE_DICT: _full_pre_load_state_dict_hook,
            StateDictType.LOCAL_STATE_DICT: _local_pre_load_state_dict_hook,
            StateDictType.SHARDED_STATE_DICT: _sharded_pre_load_state_dict_hook,
        }
        # Code that is common for all state_dict impls
        if fsdp_state._device_handle.is_available():
            fsdp_state._device_handle.synchronize()
        # Dispatch into state_dict specific implementation of pre-hook.
        _pre_load_state_dict_hook_fn[fsdp_state._state_dict_type](
            module, fsdp_state, state_dict, prefix
        )


@no_type_check
@torch.no_grad()
def _post_load_state_dict_hook(
    module: nn.Module,
    incompatible_keys: tuple[list[str], list[str]],
    *args: Any,
) -> None:
    fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(module)
    if fsdp_state.sharding_strategy == ShardingStrategy.NO_SHARD:
        context = _replace_with_full_state_dict_type(fsdp_state)
        warnings.warn(
            "When using ``NO_SHARD`` for ``ShardingStrategy``, full_state_dict will"
            "be returned.",
            stacklevel=2,
        )
    else:
        context = contextlib.nullcontext()

    with context:
        _post_load_state_dict_hook_fn = {
            StateDictType.FULL_STATE_DICT: _full_post_load_state_dict_hook,
            StateDictType.LOCAL_STATE_DICT: _local_post_load_state_dict_hook,
            StateDictType.SHARDED_STATE_DICT: _sharded_post_load_state_dict_hook,
        }
        # Code that is common for all state_dict impls
        # Dispatch into state_dict type specific implementation of post-hook for
        # loading state_dict.
        _post_load_state_dict_hook_fn[fsdp_state._state_dict_type](module, fsdp_state)

    # When reporting incompatible keys, trim FSDP prefixes.
    missing_keys = incompatible_keys[0]
    unexpected_keys = incompatible_keys[1]
    for i in range(len(missing_keys)):
        missing_keys[i] = clean_tensor_name(missing_keys[i])

    for i in range(len(unexpected_keys)):
        unexpected_keys[i] = clean_tensor_name(unexpected_keys[i])

    if fsdp_state._is_root:
        SimpleProfiler.dump_and_reset("FSDP model load_state_dict profiling: ")


def _register_all_state_dict_hooks(state: _FSDPState):
    """
    Registers pre-save, post-save, pre-load, and post-load state dict hooks.
    """
    for hook_registration_fn_str, hook, hook_registration_fn_kwargs in (
        ("register_state_dict_pre_hook", _pre_state_dict_hook, {}),
        ("_register_state_dict_hook", _post_state_dict_hook, {}),
        (
            "_register_load_state_dict_pre_hook",
            _pre_load_state_dict_hook,
            {"with_module": True},
        ),
        ("register_load_state_dict_post_hook", _post_load_state_dict_hook, {}),
    ):
        _register_state_dict_hooks_base(
            state, hook_registration_fn_str, hook, hook_registration_fn_kwargs
        )


@no_type_check
def _register_state_dict_hooks_base(
    state: _FSDPState,
    hook_registration_fn_name: str,
    hook: Callable,
    hook_registration_fn_kwargs: dict[str, Any],
) -> None:
    """Registers ``hook`` using ``hook_registration_fn``."""
    if not _is_composable(state):
        getattr(state, hook_registration_fn_name)(hook, **hook_registration_fn_kwargs)
    else:
        handle = state._handle
        if handle:
            getattr(handle._fully_sharded_module, hook_registration_fn_name)(
                hook, **hook_registration_fn_kwargs
            )
<|endoftext|>import torch.nn as nn


def _annotate_modules_for_dynamo(
    module: nn.Module,
    ignored_modules: set[nn.Module],
    use_orig_params: bool,
) -> None:
    """
    Annotates the submodules in ``module`` 's tree, except those in
    ``ignored_modules``, indicating that the submodules are FSDP-managed and
    saving the ``use_orig_params`` setting passed to the FSDP constructor.
    """
    for submodule in module.modules():
        if submodule not in ignored_modules:
            """[note: Dynamo treats FSDP wrapped modules as UnspecializedNNModule]

            Dynamo doesn't get to see this instance (FullyShardedDataParallel) during tracing, since
            it skips tracing all the torch.distributed.fsdp code.
                - Why? Running the FSDP code eagerly avoids lots of issues trying to trace complex hooks, and also
                gets us graph-breaks on FSDP module boundaries which we want anyway for comm ops.
                - However, we _also_ want dynamo to treat the wrapped module inside FSDP 'unspecially' (*),
                and we need a way to indicate to dynamo which modules are wrapped by FSDP.

            (*) UnspecializedNNModules in dynamo are traced-through without any assumptions, and with thorough
            guards.  NNModules otherwise are 'specialized', meaning there is less overhead due to assuming
            their code is well-behaved.

            One particular issue with specialized NNModules for FSDP is that the
            views created for orig_params are captured into the compiled graph on the first iteration, and while
            they are always going to point to the correct flatparameter and give correct results, their order
            of creation influences the order of backward execution, preventing overlap of comm and computation
            during backward.  We need to _use_ the new parameter views created on each forward iteration, in
            order for backward to interleave hooks with compute per layer.  UnspecializedNNModule lets us achieve
            this by capturing the module code more 'functionally' and passing parameters in as inputs each time.
            """
            submodule._is_fsdp_managed_module = True  # type: ignore[assignment]

            # Dynamo only supports FSDP with use_orig_params=True.
            # This is hacky, but I could not think of another way to add an assertion to dynamo
            # for this, since Dynamo skips all the FSDP code frames and thus can't inspect the
            # FSDP module directly
            submodule._fsdp_use_orig_params = use_orig_params  # type: ignore[assignment]
<|endoftext|># mypy: allow-untyped-defs
import logging
from collections import abc, defaultdict
from collections.abc import Iterable
from typing import Any, overload

import torch
import torch.distributed as dist
from torch.amp.grad_scaler import _MultiDeviceReplicator, GradScaler, OptState
from torch.distributed.distributed_c10d import ProcessGroup


logger = logging.getLogger(__name__)


def _refresh_per_optimizer_state() -> dict[str, Any]:
    return {"stage": OptState.READY, "found_inf_per_device": {}}


def _is_supported_device(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda or tensor.device.type in (
        "xla",
        "cpu",
        "hpu",
        "mtia",
        "xpu",
        torch._C._get_privateuse1_backend_name(),
    )


class _GeneralMultiDeviceReplicator(_MultiDeviceReplicator):
    """
    Lazily serves tensor to request device. This class extends
    _MultiDeviceReplicator to allow support for "cpu" as a device.
    """

    def __init__(self, master_tensor: torch.Tensor) -> None:
        if not _is_supported_device(master_tensor):
            raise AssertionError(
                f"Expected supported device, got {master_tensor.device}"
            )
        self.master = master_tensor
        self._per_device_tensors: dict[torch.device, torch.Tensor] = {}


class ShardedGradScaler(GradScaler):
    """
    ShardedGradScaler helps perform gradient scaling in a shard aware manner. It extends
    functionality from GradScaler:
    * Supports PyTorch DDP and FSDP implementations
    * Support CPU offloaded tensors (as used in fully sharded data parallel[FSDP])
    * Supports the custom Mixed Precision loss dtype (fp16, bf16) that FSDP returns
    * Sync inf/nan for scaled gradient tensors on any torch.device (where tensors are placed) across
    nodes

    Example::

        # Creates a ShardedGradScaler once at the beginning of training.
        scaler = ShardedGradScaler()

        for epoch in epochs:
            for input, target in data:
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)

                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                scaler.scale(loss).backward()

                # scaler.step() first unscales gradients of the optimizer's params.
                # If gradients don't contain infs/NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)

                # Updates the scale for next iteration.
                scaler.update()

    See :class:`GradScaler` for explanation of scaling/unscaling and more use cases.

    Args:
        init_scale (float, optional, default=2.**16):  Initial scale factor.
        growth_factor (float, optional, default=2.0):  Factor by which the scale is multiplied during
            :meth:`update` if no inf/NaN gradients occur for ``growth_interval`` consecutive iterations.
        backoff_factor (float, optional, default=0.5):  Factor by which the scale is multiplied during
            :meth:`update` if inf/NaN gradients occur in an iteration.
        growth_interval (int, optional, default=2000):  Number of consecutive iterations without inf/NaN gradients
            that must occur for the scale to be multiplied by ``growth_factor``.
        enabled (bool, optional):  If ``False``, disables gradient scaling. :meth:`step` simply
            invokes the underlying ``optimizer.step()``, and other methods become no-ops.
            Default: ``True``
        process_group (ProcessGroup, optional, default=torch.distributed.group.WORLD):
            process group for sharding
    """

    def __init__(
        self,
        device: str = "cuda",
        init_scale: float = 2.0**16,
        backoff_factor: float = 0.5,
        growth_factor: float = 2.0,
        growth_interval: int = 2000,
        enabled: bool = True,
        process_group: ProcessGroup | None = dist.group.WORLD,
    ) -> None:
        super().__init__(
            device,
            init_scale=init_scale,
            backoff_factor=backoff_factor,
            growth_factor=growth_factor,
            growth_interval=growth_interval,
            enabled=enabled,
        )
        if self._enabled:
            self.process_group = process_group
            self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)

    @overload
    def scale(self, outputs: torch.Tensor) -> torch.Tensor: ...

    @overload
    def scale(self, outputs: list[torch.Tensor]) -> list[torch.Tensor]: ...

    @overload
    def scale(self, outputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]: ...

    @overload
    def scale(self, outputs: Iterable[torch.Tensor]) -> Iterable[torch.Tensor]: ...

    def scale(
        self, outputs: torch.Tensor | Iterable[torch.Tensor]
    ) -> torch.Tensor | Iterable[torch.Tensor]:
        if not self._enabled:
            return outputs

        if isinstance(outputs, torch.Tensor):
            if not _is_supported_device(outputs):
                raise AssertionError(f"Expected supported device, got {outputs.device}")
            if self._scale is None:
                self._lazy_init_scale_growth_tracker(outputs.device)
            if self._scale is None:
                raise AssertionError("Expected _scale to be initialized, got None")
            scaled_output = outputs * self._scale.to(
                device=outputs.device, non_blocking=True
            )
            # Here we ensure the return dtype is the same as the outputs dtype.
            # For the FSDP + Mixed Precision use case, the loss output is in the Mixed Precision
            # format (fp16, bf16) and so the scaled loss should be of the same dtype.
            return scaled_output.type(outputs.dtype)

        stash: list[_GeneralMultiDeviceReplicator] = []

        def apply_scale(val: torch.Tensor | Iterable[torch.Tensor]):
            if isinstance(val, torch.Tensor):
                if not _is_supported_device(val):
                    raise AssertionError(f"Expected supported device, got {val.device}")
                if len(stash) == 0:
                    if self._scale is None:
                        self._lazy_init_scale_growth_tracker(val.device)
                    if self._scale is None:
                        raise AssertionError(
                            "Expected _scale to be initialized, got None"
                        )
                    stash.append(_GeneralMultiDeviceReplicator(self._scale))
                scaled_val = val * stash[0].get(val.device)
                # Here we ensure the return dtype is the same as the outputs dtype.
                # For the FSDP + Mixed Precision use case, the loss output is in the Mixed Precision
                # format (fp16, bf16) and so the scaled loss should be of the same dtype.
                return scaled_val.type(val.dtype)
            if isinstance(val, abc.Iterable):
                iterator = map(apply_scale, val)
                if isinstance(val, (list, tuple)):
                    return type(val)(iterator)
                return iterator
            raise ValueError("outputs must be a Tensor or an iterable of Tensors")

        return apply_scale(outputs)

    def _unscale_grads_(
        self,
        optimizer: torch.optim.Optimizer,
        inv_scale: torch.Tensor,
        found_inf: torch.Tensor,
        allow_fp16: bool = True,
    ) -> dict[torch.device, torch.Tensor]:
        per_device_inv_scale = _GeneralMultiDeviceReplicator(inv_scale)
        per_device_found_inf = _GeneralMultiDeviceReplicator(found_inf)

        # To set up _amp_foreach_non_finite_check_and_unscale_, split grads by device and dtype.
        # There could be thousands of grads, so we'd like to iterate through them just once.
        # However, we don't know their devices or dtypes in advance.

        # https://stackoverflow.com/questions/5029934/defaultdict-of-defaultdict
        # Google says mypy struggles with defaultdicts type annotations.
        per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))  # type: ignore[var-annotated]
        with torch.no_grad():
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    if (not allow_fp16) and param.grad.dtype == torch.float16:
                        raise ValueError("Attempting to unscale FP16 gradients.")
                    if param.grad.is_sparse:
                        # is_coalesced() == False means the sparse grad has values with duplicate indices.
                        # coalesce() deduplicates indices and adds all values that have the same index.
                        # For scaled fp16 values, there's a good chance coalescing will cause overflow,
                        # so we should check the coalesced _values().
                        if param.grad.dtype is torch.float16:
                            # coalesce is not supported in torch.float16
                            param_grad_fp32 = param.grad.type(torch.float32).coalesce()
                            param.grad = param_grad_fp32.type(torch.float16)
                        to_unscale = param.grad._values()
                    else:
                        to_unscale = param.grad

                    per_device_and_dtype_grads[to_unscale.device][
                        to_unscale.dtype
                    ].append(to_unscale)

            for device, per_dtype_grads in per_device_and_dtype_grads.items():
                for grads in per_dtype_grads.values():
                    torch._amp_foreach_non_finite_check_and_unscale_(
                        grads,
                        per_device_found_inf.get(device),
                        per_device_inv_scale.get(device),
                    )
        # There exist contexts (e.g. w/ `use_orig_params=True`) wherein some
        # ranks may have no (non-zero sized) parameter shards, necessitating the
        # initialization of `per_device_found_inf._per_device_tensors` here
        if not per_device_found_inf._per_device_tensors:
            if self._scale is None:
                raise AssertionError("Expected _scale to be initialized, got None")
            per_device_found_inf.get(self._scale.device)
        return per_device_found_inf._per_device_tensors

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        if not self._enabled:
            return

        self._check_scale_growth_tracker("unscale_")

        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if optimizer_state["stage"] is OptState.UNSCALED:
            raise RuntimeError(
                "unscale_() has already been called on this optimizer since the last update()."
            )
        elif optimizer_state["stage"] is OptState.STEPPED:
            raise RuntimeError("unscale_() is being called after step().")

        # FP32 division can be imprecise for certain compile options, so we carry out the reciprocal in FP64.
        if self._scale is None:
            raise AssertionError("Expected _scale to be initialized, got None")
        inv_scale = self._scale.double().reciprocal().float()
        found_inf = torch.full(
            (1,), 0.0, dtype=torch.float32, device=self._scale.device
        )

        optimizer_state["found_inf_per_device"] = self._unscale_grads_(
            optimizer, inv_scale, found_inf, True
        )
        optimizer_state["stage"] = OptState.UNSCALED

        # Synchronize the detected inf across the ranks
        optimizer_state = self._per_optimizer_states[id(optimizer)]
        works = []
        found_inf_on_cpus = []
        found_inf_on_devices = []

        for found_inf in optimizer_state["found_inf_per_device"].values():
            if self._device != "cpu" and found_inf.device.type == "cpu":
                found_inf_on_cpus.append(found_inf)
                found_inf_on_device = found_inf.to(self._device)
                found_inf_on_devices.append(found_inf_on_device)
                works.append(
                    dist.all_reduce(
                        found_inf_on_device, async_op=True, group=self.process_group
                    )
                )
            else:
                works.append(
                    dist.all_reduce(found_inf, async_op=True, group=self.process_group)
                )
        for work in works:
            work.wait()
        if found_inf_on_cpus:
            torch._foreach_copy_(found_inf_on_cpus, found_inf_on_devices)

    def _amp_update_scale_cpu_(self, found_inf: torch.Tensor) -> None:
        """
        If found_inf is 1.0 (True), then scale is multiplied by backoff_factor and growth_tracker is set to zero.
        Otherwise, scale is multiplied by the growth factor when the growth interval is reached.
        """
        if self._scale is None or self._growth_tracker is None:
            raise AssertionError(
                "Expected _scale and _growth_tracker to be initialized, got None"
            )

        if found_inf.item() >= 1.0:
            self._scale *= self._backoff_factor
            self._growth_tracker.fill_(0)
        else:
            successful = self._growth_tracker + 1
            if successful == self._growth_interval:
                self._scale *= self._growth_factor
                self._growth_tracker.fill_(0)
            else:
                self._growth_tracker = successful

    def update(self, new_scale: float | torch.Tensor | None = None) -> None:
        """
        Updates the scale factor.
        If any optimizer steps were skipped the scale is multiplied by ``backoff_factor``
        to reduce it. If ``growth_interval`` unskipped iterations occurred consecutively,
        the scale is multiplied by ``growth_factor`` to increase it.
        Passing ``new_scale`` sets the new scale value manually. (``new_scale`` is not
        used directly, it's used to fill GradScaler's internal scale tensor. So if
        ``new_scale`` was a tensor, later in-place changes to that tensor will not further
        affect the scale GradScaler uses internally.)
        Args:
            new_scale (float or :class:`torch.Tensor`, optional, default=None):  New scale factor.
        .. warning::
            :meth:`update` should only be called at the end of the iteration, after ``scaler.step(optimizer)`` has
            been invoked for all optimizers used this iteration.
        """

        if not self._enabled:
            return

        _scale, _growth_tracker = self._check_scale_growth_tracker("update")  # type: ignore[var-annotated]

        if new_scale is not None:
            # Accept a new user-defined scale.
            if isinstance(new_scale, float):
                self._scale.fill_(new_scale)  # type: ignore[union-attr]
            else:
                reason = (
                    "new_scale should be a float or a 1-element torch.cuda.FloatTensor or "
                    "torch.FloatTensor with requires_grad=False."
                )
                if new_scale.device.type != self._device:
                    raise AssertionError(reason)
                if new_scale.numel() != 1:
                    raise AssertionError(reason)
                if new_scale.requires_grad is not False:
                    raise AssertionError(reason)
                self._scale.copy_(new_scale)  # type: ignore[union-attr]
        else:
            # Consume shared inf/nan data collected from optimizers to update the scale.
            # If all found_inf tensors are on the same device as self._scale, this operation is asynchronous.
            found_infs = [
                found_inf.to(device=_scale.device, non_blocking=True)
                for state in self._per_optimizer_states.values()
                for found_inf in state["found_inf_per_device"].values()
            ]

            if len(found_infs) == 0:
                raise AssertionError("No inf checks were recorded prior to update.")

            found_inf_combined = found_infs[0]
            if len(found_infs) > 1:
                for i in range(1, len(found_infs)):
                    found_inf_combined += found_infs[i]

            if _scale.device.type == "cpu":
                self._amp_update_scale_cpu_(found_inf_combined)
            else:
                torch._amp_update_scale_(
                    self._scale,  # type: ignore[arg-type]
                    self._growth_tracker,  # type: ignore[arg-type]
                    found_inf_combined,
                    self._growth_factor,  # type: ignore[arg-type]
                    self._backoff_factor,  # type: ignore[arg-type]
                    self._growth_interval,  # type: ignore[arg-type]
                )

        # To prepare for next iteration, clear the data collected from optimizers this iteration.
        self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)
<|endoftext|># mypy: allow-untyped-defs
import functools
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import torch
import torch.nn as nn


@dataclass
class TracingConfig:
    """
    This represents a symbolic tracing configuration.

    Args:
        tracer (torch.fx.Tracer): An instance of :class:`torch.fx.Tracer` to
            use for symbolic tracing. The default value is the native
            :class:`torch.fx.Tracer` constructed with default arguments.
            However, the user may want to pass a different value such as the
            ``HFTracer`` for models in the HuggingFace Transformers_ library.
            .. _Transformers: https://huggingface.co/docs/transformers/index
        concrete_args (Optional[Dict[str, Any]]): Concrete arguments that
            should not be treated as ``torch.fx.Proxy`` when tracing the
            module ``forward()``. Passing ``concrete_args`` allows partially
            specializing the forward, e.g. to remove control flow or data
            structures. This ``concrete_args`` here is the same argument used
            in :meth:`~torch.fx.Tracer.trace`.
    """

    tracer: torch.fx.Tracer = field(default_factory=torch.fx.Tracer)
    concrete_args: dict[str, Any] | None = None


class _ParamUsageInfo(NamedTuple):
    """
    This is used for ``_ExecutionInfo.module_to_param_usage_infos`` to record
    execution information. The ``dict`` maps modules to a list of these
    ``_ParamUsageInfo`` instances, where each instance represents a group of
    parameters used together.

    Specifically, for each module key in the ``dict``, each instance of this
    class represents either:
    (1) the module and some sublist of its ``named_parameters()`` used
    together in execution (see ``_patched_create_proxy()``), or
    (2) a submodule and all of ``submodule.named_parameters()`` (see
    ``_patched_call_module()``).

    Type (1) corresponds to directly using parameters in ops without calling
    ``forward()``, and type (2) corresponds to calling ``forward()``. The
    mapped-to lists in the ``dict`` follow the execution order.
    """

    module: nn.Module
    named_params: list[tuple[str, nn.Parameter]]


class _ExecutionInfo:
    """
    This represents the execution order information from the forward pass.

    Attributes:
        curr_module (nn.Module): Current module being traced.
        module_forward_order (List[nn.Module]): The modules in (pre-)forward
            order, i.e. the order in which their ``forward()`` methods are
            called. Each call to a module's ``forward()`` corresponds to one
            element in the list.
        module_to_param_usage_infos (Dict[nn.Module, List[_ParamUsageInfo]]):
            Maps a module to a list of module execution infos. See
            :class:`_ParamUsageInfo` for details.
        param_forward_order (List[nn.Parameter]): The parameters in forward
            execution order, where only a parameter's first participation is
            included.
        visited_params (Set[nn.Parameter]): The parameters visited so far
            during the trace. This is only used during tracing for fast
            membership check. Invariant: The parameters in
            ``param_forward_order`` are exactly those in ``visited_params``.
    """

    def __init__(self, root_module: nn.Module) -> None:
        self.curr_module: nn.Module = root_module
        self.module_forward_order: list[nn.Module] = [root_module]
        self.module_to_param_usage_infos: dict[nn.Module, list[_ParamUsageInfo]] = {
            root_module: []
        }
        self.param_forward_order: list[nn.Parameter] = []
        self.visited_params: set[nn.Parameter] = set()


class _ExecOrderTracer:
    def __init__(self) -> None:
        self.exec_info: _ExecutionInfo | None = None

    @contextmanager
    def patch_tracer(self, tracer: torch.fx.Tracer, root_module: nn.Module):
        self.exec_info = _ExecutionInfo(root_module)
        orig_call_module = tracer.call_module
        orig_create_proxy = tracer.create_proxy
        tracer.call_module = functools.partial(  # type: ignore[method-assign]
            self._patched_call_module, orig_call_module, self.exec_info
        )
        fqn_to_param = dict(root_module.named_parameters())
        tracer.create_proxy = functools.partial(  # type: ignore[method-assign]
            self._patched_create_proxy,
            orig_create_proxy,
            self.exec_info,
            fqn_to_param,
        )
        try:
            yield
        finally:
            tracer.call_module = orig_call_module  # type: ignore[method-assign]
            tracer.create_proxy = orig_create_proxy  # type: ignore[method-assign]

    def _patched_call_module(
        self,
        call_module: Callable,
        exec_info: _ExecutionInfo,
        # Below are the expected arguments to `call_module()`
        module: nn.Module,
        forward: Callable,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """
        Overrides ``call_module`` to save execution information to
        ``exec_info``. Note that ``call_module`` is called during symbolic
        tracing for each non-root module.

        Args:
            call_module (Callable): Original ``call_module`` to override.
            exec_info (_ExecutionInfo): Used to record execution information.
            module (nn.Module): Module corresponding to this ``call_module``.
            forward (Callable): ``forward()`` method of ``module`` to be called
                for this ``call_module``.
            args (Tuple[Any, ...]): Positional arguments for ``forward``.
            kwargs (Dict[str, Any]): Keyword arguments for ``forward``.

        Returns:
            Same return value as ``call_module``.
        """
        exec_info.module_forward_order.append(module)
        named_params = list(module.named_parameters())
        curr_module = exec_info.curr_module
        if named_params:
            if curr_module not in exec_info.module_to_param_usage_infos:
                raise AssertionError(
                    "The current module should have already been processed by a patched `call_module`"
                )
            exec_info.module_to_param_usage_infos[exec_info.curr_module].append(
                _ParamUsageInfo(module, named_params)
            )
        prev_curr_module = curr_module
        exec_info.curr_module = module
        exec_info.module_to_param_usage_infos[module] = []
        output = call_module(module, forward, args, kwargs)
        exec_info.curr_module = prev_curr_module
        return output

    def _patched_create_proxy(
        self,
        create_proxy: Callable,
        exec_info: _ExecutionInfo,
        fqn_to_param: dict[str, nn.Parameter],
        # Below are the expected arguments to `create_proxy()`
        kind: str,
        target: torch.fx.node.Target,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        name: str | None = None,
        type_expr: Any | None = None,
        proxy_factory_fn: Callable[[torch.fx.Node], torch.fx.Proxy] | None = None,
    ) -> torch.fx.Proxy:
        """
        Overrides ``create_proxy`` to save execution information to
        ``exec_info``. Note that ``create_proxy`` is called during symbolic
        tracing for each leaf function/method/module.

        Args:
            create_proxy (Callable): Original ``create_proxy`` to override.
            exec_info (_ExecutionInfo): Used to record execution information.
            fqn_to_param (Dict[str, nn.Parameter]): ``dict`` version of the
                root module's ``named_parameters()`` with FQN as key and
                parameter as value.
            kind (str): Kind of the target method ('call_function',
                'call_method', 'get_attr', 'call_module', 'placeholder', or
                'output'). See :class:`torch.fx.Graph` for details. This is
                passed to ``create_proxy``.
            target (torch.fx.node.Target): Contains the string name of the
                function/method/module. This is passed to ``create_proxy``.
            args (Tuple[Any, ...]): Positional arguments for the function/
                method/module. This is passed to ``create_proxy``.
            kwargs (Dict[str, Any]): Keyword arguments for the function/method/
                module. This is passed to ``create_proxy``
            name (Optional[str]): An optional string name for the ``Node``
                created in ``create_proxy``. This is passed to
                ``create_proxy``.
            type_expr (Optional[Any]): An optional type annotation representing
                the Python type that the output of the node has. This is passed
                to ``create_proxy``.
            proxy_factory_fn (Callable[[torch.fx.Node], torch.fx.Proxy]):
                An alternative proxy constructor used in ``create_proxy``. This
                is passed to ``create_proxy``.

        Returns:
            torch.fx.Proxy: Created ``Node`` wrapped in a ``Proxy`` object.
        """
        proxy = create_proxy(
            kind, target, args, kwargs, name, type_expr, proxy_factory_fn
        )
        curr_module = exec_info.curr_module
        if kind in ("call_function", "call_method"):
            if args is not None:
                named_params: list[tuple[str, nn.Parameter]] = []
                for arg in args:
                    if (
                        isinstance(arg, torch.fx.Proxy)
                        and arg.node.target in fqn_to_param
                    ):
                        param = fqn_to_param[arg.node.target]  # type: ignore[index]
                        named_params.append((arg.node.target, param))  # type: ignore[arg-type]
                        if param not in exec_info.visited_params:
                            exec_info.visited_params.add(param)
                            exec_info.param_forward_order.append(param)
                if named_params:
                    exec_info.module_to_param_usage_infos[curr_module].append(
                        _ParamUsageInfo(curr_module, named_params)
                    )
        elif kind == "call_module":
            named_params = list(curr_module.named_parameters())
            if named_params:
                exec_info.module_to_param_usage_infos[curr_module].append(
                    _ParamUsageInfo(curr_module, named_params)
                )
            for _, param in named_params:
                if param not in exec_info.visited_params:
                    exec_info.visited_params.add(param)
                    exec_info.param_forward_order.append(param)
        return proxy
<|endoftext|>"""
NOTE: This file must be imported like
``import torch.distributed.fsdp._traversal_utils`` and not like
``from torch.distributed.fsdp._traversal_utils import ...`` to avoid circular
imports. For brevity, we may import the file as ``traversal_utils``.
"""

import collections

import torch.nn as nn
from torch.distributed._composable.contract import _get_registry
from torch.distributed.fsdp._common_utils import _FSDPState, _get_module_fsdp_state


"""
[Note: FSDP State Traversal]
For the wrapper code path, ``_FSDPState`` is the ``FullyShardedDataParallel``
module wrapping a fully sharded module, and for the non-wrapper code path,
``_FSDPState`` is an object that gets embedded on a fully sharded module.
See [Note: Fully Sharded Module] for the definition.

There are three common traversal idioms: Given a root module,
- ``_get_fsdp_states()`` returns all ``_FSDPState`` s in the tree.
- ``get_fsdp_root_states()`` returns all local root ``_FSDPState`` s in the
tree (i.e. those with ``_is_root == True``).
- ``_get_fsdp_handles()``returns all ``FlatParamHandle`` s in the tree.

All of these methods must take in the root module (i.e. an ``nn.Module``) and
not a general ``_FSDPState`` because ``_FSDPState`` does not support a graph
traversal, whereas ``nn.Module`` has ``nn.Module.modules()`` for traversal.
"""


def _composable(module: nn.Module) -> bool:
    """
    Returns if ``module`` can compose with ``fully_shard``.
    """
    # TODO: Add any other composable APIs that are mutually exclusive.
    registry = _get_registry(module)
    if registry is None:
        return True
    return "replicate" not in registry


# TODO (awgu): We may be able to remove this function if we retired the
# `use_orig_params=False` code path since so far we only need the module for
# `FlatParameter` registration, which is not needed for `use_orig_params=True`.
def _get_fsdp_states_with_modules(
    module: nn.Module,
) -> tuple[list[_FSDPState], list[nn.Module]]:
    """
    Returns a tuple containing:
    1. A list of the ``_FSDPState`` instances in the module tree rooted at
    ``module`` without any duplicates and following the ``module.modules()``
    traversal order (which is assumed to be depth-first).
    2. A corresponding list of the modules owning the states in the first list.

    For the wrapper code path, both returned lists are the same, each
    containing all ``FullyShardedDataParallel`` instances. For the composable
    code path, this returns a list of all composable state instances and a list
    of the corresponding fully sharded modules. See [Note: Fully Sharded
    Module].

    NOTE: The traversal does not proceed into any module annotated by an
    incompatible API (e.g. ``replicate``).
    """
    fsdp_states: list[_FSDPState] = []
    fsdp_modules: list[nn.Module] = []
    # Track the visited FSDP states since multiple modules may share the same
    # one and we want to return a de-duplicated list
    visited_fsdp_states: set[_FSDPState] = set()
    # Track the visited modules in case of shared modules, which implies the
    # module graph is no longer a tree
    visited_modules: set[nn.Module] = set()

    # Perform depth-first search from `module` to ensure that we do not
    # traverse into an incompatible API's subtree (use DFS instead of BFS to
    # match `.modules()` order)
    deque: collections.deque[nn.Module] = collections.deque([module])
    while deque:
        submodule = deque.popleft()
        visited_modules.add(submodule)
        if not _composable(submodule):
            continue
        for child_module in reversed(list(submodule.children())):
            if child_module not in visited_modules:
                deque.appendleft(child_module)
        optional_state = _get_module_fsdp_state(submodule)
        if optional_state is not None and optional_state not in visited_fsdp_states:
            visited_fsdp_states.add(optional_state)
            fsdp_states.append(optional_state)
            fsdp_modules.append(submodule)
    return fsdp_states, fsdp_modules


def _get_fsdp_states(module: nn.Module) -> list[_FSDPState]:
    """See :func:`_get_fsdp_states_with_modules`."""
    fsdp_states, _ = _get_fsdp_states_with_modules(module)
    return fsdp_states


def _get_fsdp_handles(module: nn.Module) -> list:
    """
    Returns all ``FlatParamHandle`` s in the module tree rooted at ``module``
    following the rules in :func:`_get_fsdp_state`.
    """
    handles = [
        fsdp_state._handle
        for fsdp_state in _get_fsdp_states(module)
        if fsdp_state._handle is not None
    ]
    return handles
<|endoftext|># mypy: allow-untyped-defs
import collections
import functools
import inspect
import warnings
from collections.abc import Callable
from functools import partial
from typing import Any

import torch.nn as nn
from torch.distributed.fsdp._common_utils import (
    _get_module_fsdp_state,
    _override_module_mixed_precision,
)
from torch.distributed.fsdp.wrap import (
    _construct_wrap_fn,
    _or_policy,
    _Policy,
    _post_order_apply,
    _recursive_wrap,
    _run_mixed_precision_override_policy,
    _wrap_module_cls_individually,
)


def _auto_wrap(
    root_module: nn.Module,
    policy: Callable | _Policy,
    ignored_modules: set[nn.Module],
    ignored_params: set[nn.Parameter],
    root_kwargs: dict[str, Any],
    fsdp_fn: Callable,  # e.g. `FullyShardedDataParallel` or `fully_shard`
):
    """
    Auto wraps modules in ``root_module`` 's tree according to ``policy``
    following a post-order traversal.

    Precondition: ``root_kwargs`` should contain all arguments except
    ``module``. This function accepts the kwargs dict directly since it gets
    forwarded into the post-order traversal function.
    """
    mixed_precision = root_kwargs["mixed_precision"]
    is_wrapper = inspect.isclass(fsdp_fn)
    # TODO: We may relax this no-nested-wrapping constraint to support manual
    # wrapping followed by auto wrapping.
    _check_nested_wrapping(root_module)

    if isinstance(policy, _Policy):
        root_kwargs["auto_wrap_policy" if is_wrapper else "policy"] = None
        target_module_to_kwargs = policy._run_policy(
            root_module, ignored_modules, root_kwargs
        )
        if mixed_precision is not None:
            target_module_to_kwargs = _run_mixed_precision_override_policy(
                root_module,
                mixed_precision._module_classes_to_ignore,
                ignored_modules,
                root_kwargs,
                target_module_to_kwargs,
            )
            overridden_module_classes = _override_module_mixed_precision(
                root_module, mixed_precision._module_classes_to_ignore
            )
            _warn_on_overridden_mixed_precision(overridden_module_classes)
        use_orig_params = root_kwargs.get("use_orig_params", False)
        _validate_frozen_params(
            root_module,
            set(target_module_to_kwargs.keys()),
            ignored_params,
            use_orig_params,
        )
        wrap_fn = _construct_wrap_fn(root_module, target_module_to_kwargs, fsdp_fn)
        _post_order_apply(root_module, wrap_fn)
        return

    recursive_wrap_kwargs = {
        "module": root_module,
        "auto_wrap_policy": policy,
        "wrapper_cls": fsdp_fn,
        "ignored_modules": ignored_modules,
        "ignored_params": ignored_params,
        "only_wrap_children": True,
    }
    if mixed_precision is not None:
        # Wrap modules of the ignored types separately and register forward
        # hooks to cast to fp32 and back to the original dtype, respectively
        overridden_module_classes = _override_module_mixed_precision(
            root_module, mixed_precision._module_classes_to_ignore
        )
        policy = functools.partial(
            _or_policy,
            policies=[
                policy,
                partial(
                    _wrap_module_cls_individually,
                    module_classes=mixed_precision._module_classes_to_ignore,
                ),
            ],
        )
        recursive_wrap_kwargs["auto_wrap_policy"] = policy
        _warn_on_overridden_mixed_precision(overridden_module_classes)
    _recursive_wrap(**recursive_wrap_kwargs, **root_kwargs)  # type: ignore[arg-type]


def _check_nested_wrapping(root_module: nn.Module):
    for module_name, module in root_module.named_modules():
        if _get_module_fsdp_state(module) is not None:
            raise ValueError(
                "FSDP auto wrapping requires modules to not already have "
                f"FSDP applied but found {module_name} in\n{root_module}"
            )


def _warn_on_overridden_mixed_precision(
    overridden_module_classes: set[type[nn.Module]],
):
    if len(overridden_module_classes) == 0:
        return
    warnings.warn(
        "Both mixed precision and an auto_wrap_policy were specified to FSDP, "
        f"where the wrapped module has submodules of type:\n{overridden_module_classes}\n"
        "These modules will be wrapped as separate FSDP instances with mixed "
        "precision disabled.",
        stacklevel=2,
    )


def _validate_frozen_params(
    root_module: nn.Module,
    modules_to_wrap: set[nn.Module],
    ignored_params: set[nn.Parameter],
    use_orig_params: bool,
):
    """
    This checks that, given ``modules_to_wrap``, each module would manage
    parameters that are uniformly frozen or non-frozen. This uniformity
    requirement is strict for ``use_orig_params=False`` (hard error) and highly
    recommended for ``use_orig_params=True`` (user warning).
    """
    post_order_named_modules = _get_post_order_named_modules(root_module)
    visited_modules: set[nn.Module] = set()
    for module_name, module in post_order_named_modules:
        if module in modules_to_wrap:
            param_to_fqn = _get_managed_param_to_fqn(
                module, ignored_params, visited_modules, module_name
            )
            frozen_param_fqns: list[str] = []
            frozen_param_numel = 0
            nonfrozen_param_fqns: list[str] = []
            nonfrozen_param_numel = 0
            for param, fqn in param_to_fqn.items():
                if param.requires_grad:
                    nonfrozen_param_fqns.append(fqn)
                    nonfrozen_param_numel += param.numel()
                else:
                    frozen_param_fqns.append(fqn)
                    frozen_param_numel += param.numel()
            if len(frozen_param_fqns) > 0 and len(nonfrozen_param_fqns) > 0:
                msg = f"{module_name} has both parameters with requires_grad=True and False."
                if use_orig_params:
                    total_param_numel = frozen_param_numel + nonfrozen_param_numel
                    msg += (
                        " We do not recommend wrapping such modules since "
                        "the gradient memory usage will be higher than expected "
                        f"({total_param_numel} numel instead of {nonfrozen_param_numel} numel "
                        "before sharding via reduce-scatter). "
                    )
                else:
                    msg += " FSDP does not support wrapping such modules when use_orig_params=False. "
                msg += "If possible, wrap the frozen parameters with FSDP separately.\n"
                msg += (
                    f"The following parameters have requires_grad=True:\n{nonfrozen_param_fqns}\n"
                    f"The following parameters have requires_grad=False:\n{frozen_param_fqns}"
                )
                if use_orig_params:
                    warnings.warn(msg, stacklevel=2)
                else:
                    raise ValueError(msg)


def _get_post_order_named_modules(
    root_module: nn.Module,
) -> list[tuple[str, nn.Module]]:
    """
    This returns the named modules following a post-order traversal, which is a
    valid reverse topological sort. We achieve this using the reverse of a
    stack-based DFS order instead of reversing ``root_module.named_modules()``
    since the former gives the modules in registration order at each level in
    the module tree (as opposed to the reverse), which allows us to error/warn
    on the first registered module that violates the condition.

    For example, consider the following module structure:
        M(
          S1(),
          S2(
            SS1(),
            SS2(),
          ),
          S3(),
        )
    The reverse DFS order is [S1, SS1, SS2, S2, S3, M], while the reverse
    ``named_modules()`` order is [S3, SS2, SS1, S2, S1, M].
    """
    visited_modules = {root_module}
    stack = [("", root_module)]
    # Append and reverse at the end for linear-time algorithm
    reverse_post_order_named_modules: list[tuple[str, nn.Module]] = []
    while stack:
        module_name, module = stack.pop()
        reverse_post_order_named_modules.append((module_name, module))
        for child_module_name, child_module in module.named_children():
            if child_module is None:  # only for overrides of `named_children()`
                continue
            if child_module not in visited_modules:
                visited_modules.add(child_module)
                if module_name != "":
                    child_module_name = module_name + "." + child_module_name
                stack.append((child_module_name, child_module))
    post_order_named_modules = list(reversed(reverse_post_order_named_modules))
    return post_order_named_modules


def _get_managed_param_to_fqn(
    module_to_wrap: nn.Module,
    ignored_params: set[nn.Parameter],
    visited_modules: set[nn.Module],
    root_prefix: str,
) -> dict[nn.Parameter, str]:
    """
    This returns a dict that maps managed parameter to its FQN for the given
    ``module_to_wrap``. The dict's keys are exactly the parameters that would
    be managed by the module, where this is achieved by calling this function
    on the modules to wrap in reverse topological order, destructively updating
    ``visited_modules``, and not traversing into those modules. The FQNs are
    prefixed from the root (via ``root_prefix``) to be more informative.

    NOTE: This function is meant to be called pre-wrapping and iteratively in
    reverse topological order to cover the full module tree. This differs from
    the ``_get_param_to_fqn()`` function meant to be called post-wrapping and
    on the full module tree in one shot. Given those differences, we do not try
    to unify the two.
    """
    param_to_fqn: dict[nn.Parameter, str] = {}
    # Run BFS (or any tree traversal works)
    queue = collections.deque([(module_to_wrap, root_prefix)])
    visited_modules.add(module_to_wrap)
    while queue:
        module, prefix = queue.popleft()
        for param_name, param in module.named_parameters(recurse=False):
            if param not in ignored_params:
                fqn = param_name if prefix == "" else prefix + "." + param_name
                param_to_fqn[param] = fqn
        for child_module_name, child_module in module.named_children():
            if child_module is None:  # only for overrides of `named_children()`
                continue
            if child_module not in visited_modules:
                visited_modules.add(child_module)
                child_prefix = (
                    child_module_name
                    if prefix == ""
                    else prefix + "." + child_module_name
                )
                queue.append((child_module, child_prefix))
    return param_to_fqn
<|endoftext|># mypy: allow-untyped-defs
import collections
import itertools
import os
import warnings
from collections.abc import Callable, Generator, Iterable, Iterator
from typing import Any, no_type_check, TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.distributed.fsdp._exec_order_utils as exec_order_utils
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.distributed.fsdp.fully_sharded_data_parallel as fsdp_file
import torch.nn as nn
from torch._custom_class_base import CustomClassBase
from torch.distributed.algorithms._comm_hooks import default_hooks
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.fsdp._common_utils import (
    _FSDPDeviceHandle,
    _FSDPState,
    _get_module_fsdp_state,
    _is_fsdp_flattened,
    _named_parameters_with_duplicates,
    clean_tensor_name,
    TrainingState,
)
from torch.distributed.fsdp._flat_param import (
    _FSDP_USE_FULL_PREC_IN_EVAL,
    FlatParameter,
    FlatParamHandle,
    HandleShardingStrategy,
)
from torch.distributed.fsdp._limiter_utils import _FreeEventQueue
from torch.distributed.fsdp.api import (
    BackwardPrefetch,
    CPUOffload,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import _Policy
from torch.distributed.tensor.parallel.fsdp import DTensorExtensions
from torch.distributed.utils import _sync_params_and_buffers
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle

_TORCHDISTX_AVAIL = True
try:
    from torchdistx import deferred_init, fake  # type: ignore[import]
except ImportError:
    _TORCHDISTX_AVAIL = False

PARAM_BROADCAST_BUCKET_SIZE = 250 * 1024 * 1024
FSDP_SYNCED = "_fsdp_synced"
# Specification of process groups for hybrid sharding strategies.
HybridShardProcessGroupType = tuple[dist.ProcessGroup, dist.ProcessGroup]
# Overall specification of process group.
ProcessGroupType = dist.ProcessGroup | HybridShardProcessGroupType | None


# TODO (awgu): Refactor this later
SHARDING_STRATEGY_MAP = {
    ShardingStrategy.NO_SHARD: HandleShardingStrategy.NO_SHARD,
    ShardingStrategy.FULL_SHARD: HandleShardingStrategy.FULL_SHARD,
    ShardingStrategy.SHARD_GRAD_OP: HandleShardingStrategy.SHARD_GRAD_OP,
    ShardingStrategy.HYBRID_SHARD: HandleShardingStrategy.HYBRID_SHARD,
    ShardingStrategy._HYBRID_SHARD_ZERO2: HandleShardingStrategy._HYBRID_SHARD_ZERO2,
}
HYBRID_SHARDING_STRATEGIES = [
    ShardingStrategy.HYBRID_SHARD,
    ShardingStrategy._HYBRID_SHARD_ZERO2,
]
NO_RESHARD_AFTER_FORWARD_STRATEGIES = (
    ShardingStrategy.SHARD_GRAD_OP,
    ShardingStrategy._HYBRID_SHARD_ZERO2,
)


# NOTE: Since non-self attributes cannot be type annotated, several attributes
# on `state` are defined first as local variables before being assigned.


@no_type_check
def _init_process_group_state(
    state: _FSDPState,
    process_group: ProcessGroupType,
    sharding_strategy: ShardingStrategy,
    policy: _Policy | None,
    device_mesh: DeviceMesh | None = None,
) -> _FSDPState:
    if process_group is not None and device_mesh is not None:
        raise ValueError(
            "Cannot pass both process_group and device_mesh at the "
            "same time. Please just pass only one of them."
        )
    is_hybrid_strategy = sharding_strategy in HYBRID_SHARDING_STRATEGIES
    if is_hybrid_strategy:
        if process_group is None and policy is None and device_mesh is None:
            # Raise an error here, since this is manual wrapping with no process group
            # passed in, there is no way to ensure all wrapped FSDP instances use the same
            # process groups.
            raise ValueError(
                f"Manual wrapping with {sharding_strategy} "
                "requires explicit specification of process group or device_mesh."
            )
        else:
            state = _init_process_group_state_for_hybrid_shard(
                state, process_group, device_mesh
            )
    else:
        if device_mesh:
            state._device_mesh = device_mesh
            state.process_group = device_mesh.get_group(mesh_dim=0)
        else:
            state.process_group = (
                process_group if process_group is not None else _get_default_group()
            )

    state.rank = state.process_group.rank()
    state.world_size = state.process_group.size()
    data_parallel_world_size = state.world_size
    if is_hybrid_strategy:
        data_parallel_world_size *= state._inter_node_pg.size()
    state._gradient_predivide_factor = (
        default_hooks.DefaultState._get_gradient_predivide_factor(
            data_parallel_world_size
        )
    )
    state._gradient_postdivide_factor = (
        data_parallel_world_size / state._gradient_predivide_factor
    )
    return state


@no_type_check
def _init_process_group_state_for_hybrid_shard(
    state: _FSDPState,
    process_group: ProcessGroupType,
    device_mesh: DeviceMesh,
) -> _FSDPState:
    if device_mesh:
        if _is_valid_hybrid_shard_device_mesh(device_mesh):
            state._device_mesh = device_mesh
            # We currently only allow _inter_node_pg to be the outermost dimension, and the
            # process_group(intra_node) to be the innermost dimension.
            state._inter_node_pg = device_mesh.get_group(mesh_dim=0)
            state.process_group = device_mesh.get_group(mesh_dim=1)
        else:
            raise ValueError(
                f"Expected device_mesh to have ndim=2 but got {device_mesh.ndim}"
            )
    elif process_group is None:
        default_group = _get_default_group()
        intra_node_group, inter_node_group = _init_intra_and_inter_node_groups(
            default_group, state._device_handle.device_count()
        )
        # we shard across intra-node
        state.process_group = intra_node_group
        # save _inter_node_pg to allreduce across.
        state._inter_node_pg = inter_node_group
    else:
        # Check type and assign state.process_group and state._inter_node_pg.
        if _is_valid_hybrid_shard_pg_type(process_group):
            # Assuming that user passed in as intra node group and inter node group
            # as documented.
            state.process_group, state._inter_node_pg = process_group
        else:
            raise ValueError(
                "Expected process_group to be passed in as either None or "
                f"Tuple[dist.ProcessGroup, dist.ProcessGroup] but got {type(process_group)}"
            )
    # Create state for allreduce
    state._inter_node_state = _get_default_comm_hook_state(
        process_group=state._inter_node_pg,
    )
    return state


@no_type_check
def _is_valid_hybrid_shard_pg_type(process_group: Any) -> bool:
    return (
        isinstance(process_group, tuple)
        and len(process_group) == 2
        and all(isinstance(pg, dist.ProcessGroup) for pg in process_group)
    )


@no_type_check
def _is_valid_hybrid_shard_device_mesh(device_mesh: DeviceMesh) -> bool:
    return isinstance(device_mesh, DeviceMesh) and device_mesh.ndim == 2


@no_type_check
def _init_intra_node_process_group(num_devices_per_node: int) -> dist.ProcessGroup:
    """
    Return a process group across the current node.

    For example, given each row is a distinct node:
    0  1  2  3  4  5  6  7
    8  9 10 11 12 13 14 15
    This API would return an intra-node subgroup across
    [0, 1, ..., 7] or [8, 9, ..., 15] depending on the process's rank.
    For example, rank 3 would get [0, 1, ..., 7].
    """
    intra_node_subgroup, _ = dist.new_subgroups(num_devices_per_node)
    return intra_node_subgroup


@no_type_check
def _init_inter_node_process_group(
    global_process_group: dist.ProcessGroup,
    num_devices_per_node: int,
) -> dist.ProcessGroup:
    """
    Return an inter-node process group where each contained rank has the same local rank.

    For example, given each row is a distinct node:
    0  1  2  3  4  5  6  7
    8  9 10 11 12 13 14 15
    This API would return inter-node process group [0, 8], [1, 9], [2, 10], and so forth
    depending on the process's rank. For example, rank 1 would get [1, 9], rank 5
    would get [5, 13].
    """
    # the inter-node pg that is returned
    inter_node_pg = None
    sharding_backend = dist.get_backend(global_process_group)
    world_size = dist.get_world_size(global_process_group)
    # Assuming fully homogeneous setup
    num_nodes = world_size // num_devices_per_node
    my_local_rank = dist.get_rank(global_process_group) % num_devices_per_node
    for local_rank in range(num_devices_per_node):
        ranks_for_inter_group = [
            local_rank + (i * num_devices_per_node) for i in range(num_nodes)
        ]
        # every rank always needs to call dist.new_group
        grp = dist.new_group(ranks=ranks_for_inter_group, backend=sharding_backend)
        if local_rank == my_local_rank:
            inter_node_pg = grp

    if inter_node_pg is None:
        raise AssertionError(
            f"{my_local_rank} expected to assign inter-node pg, but did not"
        )
    return inter_node_pg


def _init_intra_and_inter_node_groups(
    global_process_group: dist.ProcessGroup,
    num_devices_per_node: int,
) -> tuple[dist.ProcessGroup, dist.ProcessGroup]:
    """
    Initialize intra and inter-node process groups and return the ones corresponding to this process's rank.

    This function can be used to initialize process groups for ``HYBRID_SHARD`` or
    ``_HYBRID_SHARD_ZERO2`` in FSDP.
    This function assumes each node has an equal number of CUDA-enabled devices.
    Returns:
        Tuple[dist.ProcessGroup, dist.ProcessGroup]: Intra and inter-node process group.
    """
    return (
        _init_intra_node_process_group(num_devices_per_node),
        _init_inter_node_process_group(global_process_group, num_devices_per_node),
    )


@no_type_check
def _init_ignored_module_states(
    state: _FSDPState,
    module: nn.Module,
    ignored_modules: Iterable[torch.nn.Module] | None,
    ignored_states: Iterable[torch.nn.Parameter]
    | Iterable[torch.nn.Module]
    | None = None,
) -> _FSDPState:
    if ignored_modules is not None and ignored_states is not None:
        raise ValueError(
            "Cannot pass both ignored_modules and ignored_states at the "
            "same time. Please just pass ignored_states."
        )
    ignored_parameters = None
    passed_as_ignored_states = ignored_states is not None
    if passed_as_ignored_states:
        ignored_states_list = list(ignored_states)
        _check_ignored_states(ignored_states_list, True)
    else:
        ignored_states_list = []
        _check_ignored_states(
            list(ignored_modules) if ignored_modules is not None else [], False
        )
    if len(ignored_states_list) > 0:
        if isinstance(ignored_states_list[0], nn.Parameter):
            ignored_parameters = ignored_states_list
        else:
            ignored_modules = ignored_states_list
    state._ignored_modules = _get_ignored_modules(module, ignored_modules)
    state._ignored_params = _get_ignored_params(
        module,
        state._ignored_modules,
        ignored_parameters,
    )
    state._ignored_buffer_names = _get_ignored_buffer_names(
        module,
        state._ignored_modules,
    )
    # TODO: FSDP's contract for buffers is not well-defined. They are
    # implicitly ignored for most functionality since they are not sharded;
    # however, FSDP still imposes some semantics on buffers (e.g. buffer mixed
    # precision). We should formalize this contract and decide if we need to
    # compute and store `_ignored_buffers`.
    return state


def _check_ignored_states(
    ignored_states: list[Any], passed_as_ignored_states: bool
) -> None:
    """
    Check that the ignored states are uniformly parameters or uniformly modules.

    We may remove this check in the future if we permit mixing.
    """
    if len(ignored_states) == 0:
        return
    if passed_as_ignored_states:
        all_params = all(isinstance(state, nn.Parameter) for state in ignored_states)
        all_modules = all(isinstance(state, nn.Module) for state in ignored_states)
        if not all_params and not all_modules:
            # Sort for consistent ordering for unit test regex matching
            sorted_types = sorted({type(state) for state in ignored_states}, key=repr)
            raise ValueError(
                "ignored_states expects all nn.Parameter or all nn.Module list "
                f"elements but got types {sorted_types}"
            )
    else:
        if not all(isinstance(state, nn.Module) for state in ignored_states):
            sorted_types = sorted({type(state) for state in ignored_states}, key=repr)
            raise ValueError(
                "ignored_modules expects nn.Module list elements but got "
                f"types {sorted_types}"
            )


@no_type_check
def _init_device_handle(
    state: _FSDPState,
    module: nn.Module,
    ignored_params: set[nn.Parameter],
    device_id: int | torch.device | None,
) -> _FSDPState:
    """
    Determine device handle used for initializing FSDP.

    If a device is specified by ``device_id``,
    then returns device handle corresponds to that device type. Otherwise, If the
    module is already on a non-CPU device, then the device type is that non-CPU device type.
    If the module is on CPU or meta, then the device type is the current accelerator device.
    See the :ref:`Accelerators<accelerators>` for details.


    This method will be called once ignored parameters was determined, as the device handle maybe needed
    for other initialization.
    """
    determined_device = None
    if device_id is not None:
        determined_device = (
            device_id
            if isinstance(device_id, torch.device)
            else torch.device(device_id)
        )
    if determined_device is None:
        for param in _get_orig_params(module, ignored_params):
            if param.device.type in {"cpu", "meta"}:
                continue
            if determined_device is None:
                determined_device = param.device
            else:
                if param.device.type != determined_device.type:
                    raise RuntimeError(
                        f"FSDP does not support modules with different device types "
                        f"but got params on {determined_device.type} and {param.device.type}"
                    )
        determined_device = determined_device or torch._C._get_accelerator()
        if determined_device.type == "cpu":
            raise RuntimeError(
                "FSDP needs a non-CPU accelerator device, but no accelerator device is detected."
            )

    state._device_handle = _FSDPDeviceHandle.from_device(determined_device)
    return state


@no_type_check
def _init_buffer_state(
    state: _FSDPState,
    module: nn.Module,
) -> _FSDPState:
    state._buffer_names = _get_buffer_names(module)
    # Save a mapping from clean fully-qualified buffer name (starting from
    # `module`) to its original dtype for restoring that dtype during model
    # checkpointing when buffer mixed precision is enabled. The names should
    # be clean since the casting happens in a `summon_full_params()` context.
    _buffer_name_to_orig_dtype: dict[str, torch.dtype] = {}
    for buffer_name, buffer in module.named_buffers():
        buffer_name = clean_tensor_name(buffer_name)
        _buffer_name_to_orig_dtype[buffer_name] = buffer.dtype
    state._buffer_name_to_orig_dtype = _buffer_name_to_orig_dtype
    return state


@no_type_check
def _init_core_state(
    state: _FSDPState,
    sharding_strategy: ShardingStrategy | None,
    mixed_precision: MixedPrecision | None,
    cpu_offload: CPUOffload | None,
    limit_all_gathers: bool,
    use_orig_params: bool,
    backward_prefetch_limit: int,
    forward_prefetch_limit: int,
) -> _FSDPState:
    # We clamp the strategy to `NO_SHARD` for world size of 1 since they are
    # currently functionally equivalent. This may change if/when we integrate
    # FSDP with MoE.
    if state.world_size == 1:
        if sharding_strategy != ShardingStrategy.NO_SHARD:
            warnings.warn(
                "FSDP is switching to use `NO_SHARD` instead of "
                f"{sharding_strategy or ShardingStrategy.FULL_SHARD} since "
                "the world size is 1.",
                stacklevel=2,
            )
        sharding_strategy = ShardingStrategy.NO_SHARD
    elif sharding_strategy == ShardingStrategy.NO_SHARD:
        warnings.warn(
            "The `NO_SHARD` sharding strategy is deprecated. If having issues, "
            "please use `DistributedDataParallel` instead.",
            FutureWarning,
            # Level 1 is here, level 2 is from `FullyShardedDataParallel`, and
            # level 3 is from the true caller
            stacklevel=3,
        )
    state.sharding_strategy = sharding_strategy or ShardingStrategy.FULL_SHARD
    state.mixed_precision = mixed_precision or MixedPrecision()
    if mixed_precision is not None:
        torch._C._log_api_usage_once(
            f"torch.distributed.fsdp.mixed_precision.{str(state.mixed_precision)}"
        )
    state._use_full_prec_in_eval = (
        os.environ.get(_FSDP_USE_FULL_PREC_IN_EVAL, "") == "1"
    )
    state.cpu_offload = cpu_offload or CPUOffload()
    state.limit_all_gathers = limit_all_gathers
    state._use_orig_params = use_orig_params
    state.training_state = TrainingState.IDLE
    state._is_root = None
    state._free_event_queue = _FreeEventQueue()
    state._debug_level = dist.get_debug_level()
    state._exec_order_data = exec_order_utils._ExecOrderData(
        state._debug_level,
        backward_prefetch_limit,
        forward_prefetch_limit,
    )
    state._unshard_event = None
    # Mapping from fully sharded module to the handles it is responsible to
    # unshard and reshard (see [Note: Fully Sharded Module])
    _fully_sharded_module_to_handle: dict[nn.Module, FlatParamHandle] = {}
    state._fully_sharded_module_to_handle = _fully_sharded_module_to_handle
    # Invariant: `state.params` contains exactly the `FlatParameter`s of the
    # handles in `state._handle`
    _handle: FlatParamHandle | None = None
    state._handle = _handle
    params: list[FlatParameter] = []
    state.params = params
    return state


@no_type_check
def _init_runtime_state(
    state: _FSDPState,
) -> _FSDPState:
    _root_pre_forward_handles: list[RemovableHandle] = []
    state._root_pre_forward_handles = _root_pre_forward_handles
    _pre_forward_handles: list[RemovableHandle] = []
    state._pre_forward_handles = _pre_forward_handles
    _post_forward_handles: list[RemovableHandle] = []
    state._post_forward_handles = _post_forward_handles
    state._sync_gradients = True
    state._comm_hook = None
    state._comm_hook_state = None
    # Used to prevent running the pre-backward hook multiple times
    return state


@no_type_check
def _init_prefetching_state(
    state: _FSDPState,
    backward_prefetch: BackwardPrefetch,
    forward_prefetch: bool,
) -> _FSDPState:
    state.backward_prefetch = backward_prefetch
    state.forward_prefetch = forward_prefetch
    # The data structures use tuples of handles to generalize over the case
    # where a module's forward involves multiple handles.
    return state


@no_type_check
# pyrefly: ignore [bad-function-definition]
def _init_extension(state: _FSDPState, device_mesh: DeviceMesh = None) -> _FSDPState:
    # TODO: we need to add additional check once we support FSDP + PiPPy.
    # This check is currently sufficient, since we only support FSDP + TP.
    root_mesh = device_mesh._get_root_mesh() if device_mesh is not None else None
    # if a root mesh is not the same as device_mesh,
    # meaning the device_mesh is sliced out from the root mesh.
    if device_mesh and root_mesh != state._device_mesh:
        state._fsdp_extension = DTensorExtensions(state._device_handle)
    else:
        # We need to explicitly set _fsdp_extension to None.
        # Otherwise, we will run into an infinite recursion when getting the attribute.
        state._fsdp_extension = None
    return state


@no_type_check
def _init_state_dict_state(state: _FSDPState) -> _FSDPState:
    state._state_dict_type = StateDictType.FULL_STATE_DICT
    state_dict_config: StateDictConfig = FullStateDictConfig()
    state._optim_state_dict_config = FullOptimStateDictConfig()
    state._state_dict_config = state_dict_config
    unshard_params_ctx: dict[nn.Module, Generator] = {}
    state._unshard_params_ctx = unshard_params_ctx

    return state


def _verify_managed_params(module: nn.Module, params: list[nn.Parameter]) -> None:
    """
    Verify if the parameters are accepted by FSDP. The only restriction now
    is that the parameter cannot be a scalar tensor (param.shape == []).
    """
    for param in params:
        if len(param.shape) == 0:
            param_name = ""
            for name, param_ in module.named_parameters():
                if param is param_:
                    param_name = name
                    break
            if not param_name:
                raise AssertionError("Expected param_name to be set")
            raise ValueError(
                "FSDP doesn't support scalar parameters. "
                f"Change {param_name} to a 1D tensor with numel equal to 1."
            )


@no_type_check
def _init_param_handle_from_module(
    state: _FSDPState,
    fully_sharded_module: nn.Module,
    device_id: int | torch.device | None,
    param_init_fn: Callable[[nn.Module], None] | None,
    sync_module_states: bool,
) -> _FSDPState:
    """Initialize a ``FlatParamHandle`` from a module ``fully_sharded_module``."""
    _check_single_device_module(fully_sharded_module, state._ignored_params, device_id)
    device_from_device_id = _get_device_from_device_id(
        device_id, state.rank, state._device_handle
    )
    is_meta_module, is_torchdistX_deferred_init = _need_to_materialize_module(
        fully_sharded_module, state._ignored_params, state._ignored_modules
    )
    # Materialize the module if needed
    if (is_meta_module or is_torchdistX_deferred_init) and param_init_fn is not None:
        _materialize_with_param_init_fn(
            fully_sharded_module, param_init_fn, state._ignored_modules
        )
    elif is_meta_module:
        _materialize_meta_module(
            fully_sharded_module,
            device_id,
            state._ignored_modules,
            state._device_handle,
        )
    elif is_torchdistX_deferred_init:
        deferred_init.materialize_module(
            fully_sharded_module,
            check_fn=lambda submodule: _get_module_fsdp_state(submodule) is None
            and submodule not in state._ignored_modules,
        )

    ignored_buffers = {
        buffer
        for ignored_module in state._ignored_modules
        for buffer in ignored_module.buffers()
    }

    _move_module_to_device(
        fully_sharded_module,
        state._ignored_params,
        ignored_buffers,
        device_from_device_id,
    )
    state.compute_device = _get_compute_device(
        fully_sharded_module,
        state._ignored_params,
        device_from_device_id,
        state.rank,
        state._device_handle,
    )

    managed_params = list(_get_orig_params(fully_sharded_module, state._ignored_params))
    _verify_managed_params(fully_sharded_module, managed_params)
    if sync_module_states:
        if state.sharding_strategy in HYBRID_SHARDING_STRATEGIES:
            # Broadcast inter-node first, then intra-node. The inter-node
            # broadcast propagates rank 0's states to each node's local
            # rank 0, so the subsequent intra-node broadcast has the
            # correct source values on every node. Reversing this order
            # causes local rank 0 on non-source nodes to broadcast
            # uninitialized states (e.g. from meta-device materialization).
            _sync_module_params_and_buffers(
                fully_sharded_module, managed_params, state._inter_node_pg
            )
            # _sync_module_params_and_buffers marks each buffer with
            # FSDP_SYNCED=True to avoid redundant syncs in nested
            # wrapping. Reset the flag here so the intra-node broadcast
            # below also includes buffers.
            for buffer in fully_sharded_module.buffers():
                if hasattr(buffer, FSDP_SYNCED):
                    setattr(buffer, FSDP_SYNCED, False)
        _sync_module_params_and_buffers(
            fully_sharded_module, managed_params, state.process_group
        )
    _init_param_handle_from_params(state, managed_params, fully_sharded_module)
    return state


@no_type_check
def _init_param_handle_from_params(
    state: _FSDPState,
    params: list[nn.Parameter],
    fully_sharded_module: nn.Module,
):
    if len(params) == 0:
        return
    handle = FlatParamHandle(
        params,
        fully_sharded_module,
        state.compute_device,
        SHARDING_STRATEGY_MAP[state.sharding_strategy],
        state.cpu_offload.offload_params,
        state.mixed_precision.param_dtype,
        state.mixed_precision.reduce_dtype,
        state.mixed_precision.keep_low_precision_grads,
        state.process_group,
        state._use_orig_params,
        fsdp_extension=state._fsdp_extension,
    )
    handle.shard()
    if state._handle:
        raise AssertionError("Expected state._handle to be None")
    state.params.append(handle.flat_param)
    state._handle = handle
    state._fully_sharded_module_to_handle[handle._fully_sharded_module] = handle
    cpu_device = torch.device("cpu")
    if state.cpu_offload.offload_params and handle.flat_param.device != cpu_device:
        handle.flat_param_to(cpu_device)


def _get_ignored_modules(
    root_module: nn.Module,
    _ignored_modules: Iterable[torch.nn.Module] | None,
) -> set[nn.Module]:
    """
    Check that ``_ignored_modules`` is an iterable of ``nn.Module`` s without any FSDP instances.

    Return the modules contained in their module
    subtrees as a :class:`set`. Nested FSDP instances are excluded, but their
    already-computed ignored modules are included.

    ``_ignored_modules`` represents the argument passed by the user to FSDP.
    """
    msg_prefix = "`ignored_modules` should be an iterable of `torch.nn.Module`s "
    try:
        ignored_root_modules = (
            set(_ignored_modules) if _ignored_modules is not None else set()
        )
    except TypeError as e:
        raise TypeError(msg_prefix + f"but got {type(_ignored_modules)}") from e
    for module in ignored_root_modules:
        if not isinstance(module, torch.nn.Module):
            raise TypeError(msg_prefix + f"but got an iterable with {type(module)}")
        if _get_module_fsdp_state(module):
            # TODO: We may relax this by taking the FSDP instance's wrapped
            # module to provide more flexibility to the user.
            raise ValueError("`ignored_modules` should not include FSDP modules")
    # Treat modules that cannot compose with `fully_shard` as ignored modules,
    # meaning that their subtrees are ignored
    for module in root_module.modules():
        if not traversal_utils._composable(module):
            ignored_root_modules.add(module)
    # NOTE: Even if `ignored_root_modules` is empty, do not return early so
    # that this FSDP instance can get any ignored modules from its children.

    # Include child modules and exclude nested FSDP modules themselves
    ignored_modules = {
        child
        for module in ignored_root_modules
        for child in module.modules()
        if not isinstance(child, fsdp_file.FullyShardedDataParallel)
    }
    if root_module in ignored_modules:
        warnings.warn(
            "Trying to ignore the top-level module passed into the FSDP "
            "constructor itself will result in all parameters being "
            f"ignored and is not well-supported: {module}",
            stacklevel=2,
        )
    # Include nested FSDP modules' ignored modules
    for submodule in root_module.modules():
        optional_fsdp_state = _get_module_fsdp_state(submodule)
        if optional_fsdp_state is not None:
            if not hasattr(optional_fsdp_state, "_ignored_modules"):
                raise AssertionError(
                    "Expected optional_fsdp_state to have _ignored_modules attribute"
                )
            ignored_modules.update(optional_fsdp_state._ignored_modules)
    return ignored_modules


def _get_ignored_params(
    root_module: torch.nn.Module,
    ignored_modules: set[torch.nn.Module],
    ignored_parameters: Iterable[torch.nn.Parameter] | None = None,
) -> set[torch.nn.Parameter]:
    """
    Return the parameters of the modules in ``ignored_modules`` and the parameters in ``ignored_parameters``.

    :class:`FlatParameter` s are excluded from the result.
    """
    all_ignored_params: set[torch.nn.Parameter] = set()

    params_in_ignored_modules = {
        p for m in ignored_modules for p in m.parameters() if not _is_fsdp_flattened(p)
    }

    all_ignored_params.update(params_in_ignored_modules)

    if ignored_parameters is not None:
        params_in_ignored_parameters = {
            p for p in ignored_parameters if not _is_fsdp_flattened(p)
        }
        all_ignored_params.update(params_in_ignored_parameters)

    # Always include nested FSDP modules' ignored parameters
    for submodule in root_module.modules():
        optional_fsdp_state = _get_module_fsdp_state(submodule)
        if optional_fsdp_state is not None:
            if not hasattr(optional_fsdp_state, "_ignored_params"):
                raise AssertionError(
                    "Expected optional_fsdp_state to have _ignored_params attribute"
                )
            all_ignored_params.update(optional_fsdp_state._ignored_params)

    return all_ignored_params


def _get_ignored_buffer_names(
    root_module: torch.nn.Module,
    ignored_modules: set[torch.nn.Module],
) -> set[str]:
    """Return the cleaned buffer FQNs in ``ignored_modules``."""
    all_ignored_buffer_names: set[str] = set()

    buffers_in_ignored_modules = {
        buffer for m in ignored_modules for buffer in m.buffers()
    }

    all_ignored_buffer_names.update(
        {
            clean_tensor_name(buffer_name)
            for buffer_name, buffer in root_module.named_buffers()
            if buffer in buffers_in_ignored_modules
        }
    )

    # Always include nested FSDP modules' ignored buffer names
    for submodule in root_module.modules():
        optional_fsdp_state = _get_module_fsdp_state(submodule)
        if optional_fsdp_state is not None:
            if not hasattr(optional_fsdp_state, "_ignored_buffer_names"):
                raise AssertionError(
                    "Expected optional_fsdp_state to have _ignored_buffer_names attribute"
                )
            all_ignored_buffer_names.update(optional_fsdp_state._ignored_buffer_names)

    return all_ignored_buffer_names


def _get_buffer_names(root_module: nn.Module) -> set[str]:
    """Return the fully prefixed names of all buffers in the module hierarchy rooted at ``root_module`` as a class:`set`."""
    return {
        clean_tensor_name(buffer_name) for buffer_name, _ in root_module.named_buffers()
    }


def _check_single_device_module(
    module: nn.Module,
    ignored_params: set[nn.Parameter],
    device_id: int | torch.device | None,
) -> None:
    """
    Raise an error if ``module`` has original parameters on multiple devices, ignoring the parameters in ``ignored_params``.

    Thus, after this method, the
    module must be either fully on the CPU or fully on a non-CPU device.
    """
    devices = {param.device for param in _get_orig_params(module, ignored_params)}
    # We allow module to be partially on CPU and partially on GPU if device_id is not
    # None, since the device_id arg will result in the CPU portion being moved to
    # GPU. This is useful in cases where part of the module may be parallelized
    # by another algorithm and may already be on GPU. We'd like to enforce device_id
    # to not be None, otherwise we'd flatten parameters in a mixed module which is
    # not supported.
    if len(devices) == 2 and torch.device("cpu") in devices:
        if device_id is None:
            raise RuntimeError(
                "To support a module with both CPU and GPU params, "
                "please pass in device_id argument."
            )
    elif len(devices) > 1:
        raise RuntimeError(
            f"FSDP only supports single device modules but got params on {devices}"
        )


def _get_device_from_device_id(
    device_id: int | torch.device | None,
    rank: int,
    device_handle: _FSDPDeviceHandle,
) -> torch.device | None:
    """
    Return a ``torch.device`` for the specified ``device_id``.

    Processes ``device_id`` and returns either the corresponding device or
    ``None`` if ``device_id`` is ``None``.
    """
    if device_id is None:
        return None
    device = (
        device_id if isinstance(device_id, torch.device) else torch.device(device_id)
    )
    if device.type != "cpu" and device.index is None:
        warnings.warn(
            f"FSDP got the argument `device_id` {device_id} on rank "
            f"{rank}, which does not have an explicit index. "
            f"FSDP will use the current device {device_handle.current_device()}. "
            f"If this is incorrect, please explicitly call `torch.{device.type}.set_device()` "
            "before FSDP initialization or pass in the explicit device "
            "index as the `device_id` argument.",
            stacklevel=2,
        )
        device = torch.device(device_handle.current_device())
    return device


def _need_to_materialize_module(
    module: nn.Module,
    ignored_params: set[nn.Parameter],
    ignored_modules: set[nn.Module],
) -> tuple[bool, bool]:
    """
    Return if ``module`` has parameters on meta device and if ``module`` is using torchdistX deferred initialization.

    At most of the returned bools can
    be ``True``. If either is ``True``, then ``module`` needs to be
    materialized.
    """
    managed_params = list(_get_orig_params(module, ignored_params))
    is_meta_module = any(param.is_meta for param in managed_params)
    # TODO: We need to establish a contract for FSDP and buffers. For now, we
    # skip checking for meta buffers from ignored modules. We should consider
    # refactoring the initialization holistically to avoid so many traversals.
    for submodule in module.modules():
        if submodule in ignored_modules:
            continue
        for buf in submodule.buffers(recurse=False):
            is_meta_module |= buf.is_meta
    is_torchdistX_deferred_init = (
        not is_meta_module
        and _TORCHDISTX_AVAIL
        and any(fake.is_fake(param) for param in managed_params)
    )
    return is_meta_module, is_torchdistX_deferred_init


def _materialize_with_param_init_fn(
    root_module: nn.Module,
    param_init_fn: Callable[[nn.Module], None],
    ignored_modules: set[nn.Module],
) -> None:
    if not callable(param_init_fn):
        raise ValueError(
            f"Expected {param_init_fn} to be callable but got {type(param_init_fn)}"
        )
    modules_to_materialize = _get_modules_to_materialize(root_module, ignored_modules)
    for module in modules_to_materialize:
        param_init_fn(module)


def _materialize_meta_module(
    root_module: nn.Module,
    device_from_device_id: torch.device | None,
    ignored_modules: set[nn.Module],
    device_handle: _FSDPDeviceHandle,
):
    # Run default meta device initialization
    materialization_device = device_from_device_id or torch.device(
        device_handle.current_device()
    )
    modules_to_materialize = _get_modules_to_materialize(root_module, ignored_modules)
    module = None
    try:
        # Assume that each module's `reset_parameters()` only initializes its
        # own parameters and not those of its children
        with torch.no_grad():
            for module in modules_to_materialize:
                # As a contract to the user, only call `reset_parameters()` if
                # the module has directly managed parameters/buffers
                module_state_iter = itertools.chain(
                    module.parameters(recurse=False),
                    # pyrefly: ignore [bad-argument-type]
                    module.buffers(recurse=False),
                )
                has_module_states = len(list(module_state_iter)) > 0
                if has_module_states:
                    module.to_empty(device=materialization_device, recurse=False)
                    module.reset_parameters()  # type: ignore[operator]
    except BaseException as e:
        warnings.warn(
            "Unable to call `reset_parameters()` for module on meta "
            f"device with error {str(e)}. Please ensure that your module of"
            f"type {type(module)} implements a `reset_parameters()` method.",
            stacklevel=2,  # type: ignore[possibly-undefined]
        )
        raise e


def _get_modules_to_materialize(
    root_module: nn.Module, ignored_modules: set[nn.Module]
) -> list[nn.Module]:
    # Run BFS to collect the modules to materialize via `reset_parameters()`,
    # stopping at any module with FSDP already applied or at ignored modules.
    modules_to_materialize: list[nn.Module] = []
    queue = collections.deque([root_module])
    visited_modules: set[nn.Module] = {root_module}
    while queue:
        module = queue.popleft()
        modules_to_materialize.append(module)
        for child_module in module.children():
            if (
                child_module not in visited_modules
                and _get_module_fsdp_state(child_module) is None
                and child_module not in ignored_modules
            ):
                visited_modules.add(child_module)
                queue.append(child_module)
    return modules_to_materialize


def _move_module_to_device(
    module: nn.Module,
    ignored_params: set[nn.Parameter],
    ignored_buffers: set[torch.Tensor],
    device_from_device_id: torch.device | None,
) -> None:
    """
    Move ``module`` depending on ``device_from_device_id`` and its current device.

    This includes moving ignored modules' parameters.

    - If ``device_from_device_id`` is not ``None``, then this moves
    ``module`` to the device.
    - If ``device_from_device_id`` is ``None``, then this does not move
    ``module`` but warns the user if it is on CPU.

    Precondition: ``_check_single_device_module()``.
    """
    cpu_device = torch.device("cpu")
    if device_from_device_id is not None:
        # BFS from `module` without traversing any nested FSDP instances to
        # collect the parameters/buffers that have not yet been managed
        queue: collections.deque[nn.Module] = collections.deque()
        queue.append(module)
        params: list[nn.Parameter] = []
        buffers: list[torch.Tensor] = []
        while queue:
            curr_module = queue.popleft()
            # NOTE: We include a check to only move parameters/buffers that are
            # on CPU device. If they are on a CUDA device different from the
            # one specified by `device_id`, then this does NOT move them. This
            # is so that we can raise an error in `_get_compute_device()`.
            params.extend(
                param
                for param in curr_module.parameters(recurse=False)
                if param.device == cpu_device
            )
            buffers.extend(
                buffer
                for buffer in curr_module.buffers(recurse=False)
                if buffer.device == cpu_device
            )
            for submodule in curr_module.children():
                if not isinstance(submodule, fsdp_file.FullyShardedDataParallel):
                    queue.append(submodule)
        params_to_move = [p for p in params if p not in ignored_params]
