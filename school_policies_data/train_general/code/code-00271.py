    info: _StateDictInfo,
) -> None:
    for module in info.fsdp_modules:
        fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(module)
        if fsdp_state is None:
            raise AssertionError("Expected a fsdp_state with a fsdp module.")

    # Verify if the model_state_dict and optim_state_dict are valid. This API
    # should give the users an explicit error message to debug or report.
    if (
        info.handle_model
        and not model_state_dict
        and not info.submodule_prefixes
        and not info.ignore_frozen_params
        and not (info.cpu_offload and info.full_state_dict)
        and info.strict
        and not info.broadcast_from_rank0
    ):
        raise RuntimeError(
            "The option indicates that model state_dict is required to save "
            "or load, but model state_dict is empty."
            f"rank = {dist.get_rank()=}."
        )

    if info.handle_optim:
        if (
            not optim_state_dict
            and not (info.cpu_offload and info.full_state_dict)
            and (not info.broadcast_from_rank0)
        ):
            raise RuntimeError(
                "The option indicates that model state_dict is required to save, "
                f"or load but optim state_dict is empty. {optim_state_dict}"
            )

    for key in model_state_dict:
        if _FLAT_PARAM in key:
            raise RuntimeError(
                f"{key} contains {_FLAT_PARAM}. This can happen if the model "
                "is not the root module."
            )


def _state_dict_fn(obj: nn.Module | torch.optim.Optimizer, api: str) -> Callable:
    call = getattr(obj, api)
    if call in _patched_state_dict:
        call = functools.partial(getattr(obj.__class__, api), self=obj)
    return call


def _get_fsdp_process_group(
    model: nn.Module, info: _StateDictInfo
) -> dist.ProcessGroup | None:
    if isinstance(model, FSDP):
        fsdp_module = model
    elif info.fsdp_modules:
        fsdp_module = cast(FSDP, info.fsdp_modules[0])
    else:
        return None

    if fsdp_module.sharding_strategy in HYBRID_SHARDING_STRATEGIES:
        return None

    process_group = fsdp_module.process_group
    if isinstance(process_group, tuple):
        return None
    return process_group


def _maybe_full_or_cpu_state_dict(
    state_dict: dict[str, Any], info: _StateDictInfo
) -> dict[str, Any]:
    if info.full_state_dict:
        ranks_only = (
            ()
            if (not info.cpu_offload or not torch.distributed.is_initialized())
            else (0,)
        )
        return _gather_state_dict(
            state_dict, cpu_offload=info.cpu_offload, ranks_only=ranks_only
        )
    elif info.cpu_offload:
        return _offload_state_dict_to_cpu(state_dict)
    else:
        return state_dict


@torch.no_grad()
def _get_model_state_dict(
    model: nn.Module, info: _StateDictInfo
) -> dict[str, ValueType]:
    if not info.handle_model:
        return {}

    with info.fsdp_context():
        state_dict = _state_dict_fn(model, "state_dict")()

    for key in list(state_dict.keys()):
        fqns = _get_fqns(model, key)
        if len(fqns) != 1:
            raise AssertionError(
                f"Expected 1 FQN for key '{key}', got {len(fqns)}: {fqns}"
            )
        fqn = next(iter(fqns))
        if fqn != key:
            # As we only support FSDP, DDP, and TP, the only cases are
            # wrapper-based DDP and compiler. Verify if the assumption
            # is correct.
            def verify(key, fqn) -> bool:
                if len(fqn) >= len(key):
                    return False
                fqn_split = fqn.split(".")
                key_split = key.split(".")
                fqn_idx = 0
                for key_idx, key_name in enumerate(key_split):
                    if key_name == fqn_split[fqn_idx]:
                        fqn_idx += 1
                        if fqn_idx == len(fqn_split):
                            return key_idx == len(key_split) - 1
                    elif key_name in ("module", "_orig_mod"):
                        continue
                    else:
                        return False
                return True

            if not verify(key, fqn):
                raise RuntimeError(f"An unexpected key, {key}, exists. FQN is {fqn}")
            state_dict[fqn] = state_dict.pop(key)

    if info.submodule_prefixes:
        new_state_dict: dict[str, ValueType] = {}
        # TODO: make this faster.
        for fqn in state_dict:
            for prefix in info.submodule_prefixes:
                if not fqn.startswith(prefix):
                    continue
                if info.keep_submodule_prefixes:
                    new_state_dict[fqn] = state_dict[fqn]
                else:
                    new_fqn = fqn[len(prefix) :]
                    new_state_dict[new_fqn] = state_dict[fqn]
        state_dict = new_state_dict

    if info.ignore_frozen_params:
        for key, param in model.named_parameters():
            if param.requires_grad:
                continue
            fqns = _get_fqns(model, key)
            for fqn in fqns:
                state_dict.pop(fqn)

    return _maybe_full_or_cpu_state_dict(state_dict, info)


@torch.no_grad()
def _load_model_state_dict(
    model: nn.Module,
    state_dict: dict[str, ValueType],
    info: _StateDictInfo,
) -> _IncompatibleKeys:
    if not info.handle_model or (not state_dict and not info.broadcast_from_rank0):
        return _IncompatibleKeys({}, {})

    local_state_dict = {}
    for key, value in _iterate_valid_model_state(model, info.dsd_fqn_modifiers):
        fqns = _get_fqns(model, key, info.dsd_fqn_modifiers)
        fqns_with_prefix = _get_fqns(
            model,
            key,
            info.dsd_fqn_modifiers,
            skip_ddp_prefix=False,
            skip_compiler_prefix=False,
        )

        for fqn, fqn_with_prefix in zip(fqns, fqns_with_prefix):
            if (
                not info.broadcast_from_rank0 or dist.get_rank() == 0
            ) and fqn != fqn_with_prefix:
                load_value = state_dict.pop(fqn, None)
                if load_value is None:
                    if info.strict:
                        raise RuntimeError(f"Missing key: {fqn}.")
                else:
                    state_dict[fqn_with_prefix] = load_value
            local_state_dict[fqn_with_prefix] = value

    assign = False
    if info.broadcast_from_rank0 or info.full_state_dict:
        devices = set()
        for value in local_state_dict.values():
            if torch.is_tensor(value) and value.dim() > 0:
                devices.add(value.device)
        # In lora state_dict, there could be multiple devices, with meta device inside.
        # Take the other device in the broadcast/distribute, and set assign to True
        if torch.device("meta") in devices:
            devices.remove(torch.device("meta"))
            assign = True
        if len(devices) == 0:
            devices.add(dist.distributed_c10d._get_pg_default_device())
        elif len(devices) > 1:
            raise ValueError("Multiple devices found")

        if info.broadcast_from_rank0:
            _broadcast_state_dict(
                state_dict,
                local_state_dict,
                device=devices.pop(),
                strict=info.strict,
                cpu_offload=info.cpu_offload,
            )
        elif info.full_state_dict:
            _distribute_state_dict(state_dict, local_state_dict, device=devices.pop())
        state_dict.update(local_state_dict)

    with info.fsdp_context():
        return cast(
            _IncompatibleKeys,
            _state_dict_fn(model, "load_state_dict")(
                state_dict=state_dict, strict=info.strict, assign=assign
            ),
        )


def _init_optim_state(optim: torch.optim.Optimizer) -> None:
    """
    Initialize optim states by calling the step() with zero grads.
    """
    if optim.state:
        # The optimizer state is initialized.
        return

    # There are some stateless optimizers like SGD. These optimizer will
    # not return in the above condition. So if gradients exist, we should also
    # return. If gradients do not exist, the following initialization should
    # not disturb SGD because the gradients and lr are both zero.
    for param_group in optim.param_groups:
        for param in param_group[_PARAMS]:
            if param.grad is not None:
                return

    for param_group in optim.param_groups:
        for param in param_group[_PARAMS]:
            if param.requires_grad:
                param.grad = torch.zeros_like(param)

    # Some optimizers will update parameters regardless of grads due to lr, so
    # make lr to zero when calling `step()`.
    lrs = []
    for param_group in optim.param_groups:
        if "lr" in param_group:
            lrs.append(param_group["lr"])
            param_group["lr"] = (
                torch.tensor(0.0)
                if isinstance(param_group["lr"], torch.Tensor)
                else 0.0
            )
    optim.step(closure=None)
    # Whether to recover the "lr" should not matter too much as we will
    # restore checkpointing later.
    for param_group in optim.param_groups:
        if "lr" in param_group:
            param_group["lr"] = lrs.pop(0)
    optim.zero_grad(set_to_none=True)


def _flatten_optim_state_dict(state_dict: OptimizerStateType) -> dict[str, ValueType]:
    """
    This API flattens the optimizer state_dict to support optimizer resharding for
    MPMD, e.g., pipeline parallelism.

    Without the API, the original optimizer state_dict looks like:
    {
        "state": {
            "layer1.weight": {
                "step": 10, "exp_avg": SomeTensor, "exp_avg_sq": SomeTensor
            },
            "layer2.weight": {
                "step": 10, "exp_avg": SomeTensor, "exp_avg_sq": SomeTensor
            },
        },
        "param_groups": [
            {
                "lr": 0.0,
                "betas": (0.9, 0.95), ...,
                "params": ["layer1.weight", "layer2.weight"]
            }
        ]
    }

    With this API, the optimizer state_dict looks like:
    {
        "state.layer1.weight.step": 10,
        "state.layer2.weight.step": 10,
        "state.layer1.weight.exp_avg": SomeTensor,
        "state.layer2.weight.exp_avg": SomeTensor,
        "state.layer1.weight.exp_avg_sq": SomeTensor,
        "state.layer2.weight.exp_avg_sq": SomeTensor,
        "param_groups.layer1.weight.lr": 0.1,
        "param_groups.layer2.weight.lr": 0.1,
        "param_groups.layer1.weight.betas": (0.9, 0.95),
        "param_groups.layer2.weight.betas": (0.9, 0.95),
    }

    The "state" section supports arbitrary levels of nesting for optimizers like Shampoo.
    """

    def _flatten_state_nested_dict(
        nested_dict: dict[str, Any], prefix: str
    ) -> dict[str, ValueType]:
        """
        Recursively flatten a nested dictionary with dot-separated keys.

        Args:
            nested_dict: The dictionary to flatten
            prefix: The prefix to prepend to all keys

        Returns:
            Flattened dictionary with dot-separated keys
        """
        flattened: dict[str, ValueType] = {}

        for key, value in nested_dict.items():
            # Convert all keys to strings for flattening
            str_key = str(key)
            full_key = f"{prefix}.{str_key}" if prefix else str_key

            if isinstance(value, dict):
                # Recursively flatten nested dictionaries
                flattened.update(_flatten_state_nested_dict(value, full_key))
            else:
                # Base case: store the value with the flattened key
                _raise_if_type_not_supported(value)
                flattened[full_key] = value

        return flattened

    def _raise_if_type_not_supported(v):
        if not isinstance(v, (torch.Tensor, int, float, dict)):
            raise NotImplementedError(
                "Flattening optimizer state_dict only supports "
                "tensor, int, float, dict states now. "
                f"Type is {type(v)}."
            )

    ret: dict[str, ValueType] = {}

    # Handle the "state" section with recursive flattening
    for fqn, state in cast(DictValueType, state_dict[_STATE]).items():
        state_prefix = f"{_STATE}.{fqn}"
        ret.update(
            _flatten_state_nested_dict(cast(dict[str, Any], state), state_prefix)
        )

    # Handle the "param_groups" section with two-level flattening
    for param_group in cast(ListDictValueType, state_dict[_PG]):
        fqns = param_group.pop(_PARAMS)
        for fqn in cast(list[str], fqns):
            for k, v in param_group.items():
                ret[f"{_PG}.{fqn}.{k}"] = v

    return ret


def _unflatten_optim_state_dict(
    optim: torch.optim.Optimizer,
    state_dict: dict[str, ValueType],
    info: _StateDictInfo,
) -> OptimizerStateType:
    """
    This API unflattens the state_dict generated by _flatten_optim_state_dict().
    Supports arbitrary levels of nesting in the state section through recursive reconstruction.

    See the docstring of _flatten_optim_state_dict() for more detail.
    """

    def _reconstruct_nested_dict(
        flattened_key: str, flattened_dict: dict[str, ValueType]
    ) -> dict[str, ValueType]:
        """
        Reconstructs a potentially nested value from flattened keys.
        For non-nested values, returns the value directly.
        For nested values, reconstructs the nested structure with string keys.
        """

        # Create the prefix to search for nested keys
        # e.g., if flattened_key is "state.layer1.weight", prefix becomes "state.layer1.weight."
        prefix = f"{flattened_key}."
        # Initialize an empty dictionary to build our nested structure
        nested_dict: dict[str, Any] = {}

        # Iterate through all keys in the flattened dictionary
        for key, value in flattened_dict.items():
            # Check if this key is nested under our target key
            # e.g., "state.layer1.weight.exp_avg" starts with "state.layer1.weight."
            if not key.startswith(prefix):
                # Skip keys that don't belong to this nested structure
                continue

            # Remove the prefix to get just the nested part
            # e.g., "state.layer1.weight.exp_avg" -> "exp_avg"
            remaining_key = key[len(prefix) :]
            # Split the remaining key into parts to build the nested structure
            # e.g., "step" -> ["step"] or "momentum_buffer" -> ["momentum_buffer"]
            parts = remaining_key.split(".")
            # Start at the root of our new nested dictionary
            current = nested_dict

            # Navigate through or create the nested dictionary structure
            # For each part except the last one (which will hold the value)
            for part in parts[:-1]:
                # Create the nested dictionary if it doesn't exist yet
                if part not in current:
                    current[part] = {}
                # Move deeper into the nested structure
                if not isinstance(current[part], dict):
                    raise AssertionError
                current = current[part]

            # Set the value at the final level using the last part as the key
            # e.g., current["exp_avg"] = tensor(...)
            current[parts[-1]] = value

        # Return the reconstructed nested dictionary (empty dict if no keys matched at all)
        return nested_dict

    state: DictValueType = {}
    pg_state: ListDictValueType = []
    return_osd: OptimizerStateType = {_STATE: state, _PG: pg_state}

    for param_group in optim.param_groups:
        pg_state.append({_PARAMS: []})
        for param in param_group[_PARAMS]:
            for fqn in info.fqn_param_mapping[param]:
                # If a parameter is shared, only one of the FQN will be used.
                # So we need to verify which if this fqn is actually used in
                # the state_dict.
                if fqn in info.shared_params_mapping:
                    in_params = False
                    for k in param_group:
                        if k == _PARAMS:
                            continue
                        flatten_key = f"{_PG}.{fqn}.{k}"
                        if flatten_key in state_dict:
                            in_params = True
                        break
                else:
                    in_params = True

                if not in_params:
                    continue

                params = pg_state[-1][_PARAMS]
                if not isinstance(params, list):
                    raise AssertionError(f"Expected list, got {type(params)}")
                params.append(fqn)

                # Only add state if param requires grad
                if not param.requires_grad:
                    continue

                # Reconstruct state for this parameter
                # pyrefly: ignore [unsupported-operation]
                state[fqn] = {}
                for state_name in optim.state[param]:
                    flattened_state_key = f"{_STATE}.{fqn}.{state_name}"

                    if flattened_state_key not in state_dict:
                        # Try to reconstruct the value
                        reconstructed_value = _reconstruct_nested_dict(
                            flattened_state_key, state_dict
                        )
                        # pyrefly: ignore [bad-index]
                        cast(DictValueType, state[fqn])[state_name] = (
                            reconstructed_value
                        )
                    else:
                        # Existing keys mean no nesting, directly use the value.
                        # pyrefly: ignore [bad-index]
                        cast(DictValueType, state[fqn])[state_name] = state_dict[
                            flattened_state_key
                        ]

        first_param_fqn = cast(list[str], pg_state[-1][_PARAMS])[0]
        for k in param_group:
            if k == _PARAMS:
                continue
            value = state_dict[f"{_PG}.{first_param_fqn}.{k}"]
            if k not in pg_state[-1]:
                pg_state[-1][k] = value
            elif pg_state[-1][k] != value:
                raise RuntimeError(
                    "All the parameters in the same parameter group should have "
                    f"the same saved param_group value. But {first_param_fqn}.{k} "
                    f"is {value} while other(s) is {pg_state[-1][k]}."
                )

    return return_osd


@torch.no_grad()
def _get_optim_state_dict(
    model: nn.Module,
    optimizers: tuple[torch.optim.Optimizer, ...],
    info: _StateDictInfo,
) -> OptimizerStateType:
    if not info.handle_optim:
        return {}

    optim_state_dict: OptimizerStateType = {_STATE: {}, _PG: []}
    for optim in optimizers:
        _init_optim_state(optim)
        osd = _state_dict_fn(optim, "state_dict")()
        if info.fsdp_modules:
            with info.fsdp_context():
                osd = FSDP.optim_state_dict(
                    model,
                    optim,
                    osd,
                    group=_get_fsdp_process_group(model, info),
                )

            # We need to specially handle FlatParameter FSDP as
            # FlatParameter FSDP converts the FQNs.
            # There are no easy ways to do this conversion systematically.
            # We can only use a string replacement without correctness check.
            if not osd:
                continue
            for k in list(osd[_STATE].keys()):
                if "_orig_mod" in k:
                    osd[_STATE][k.replace("_orig_mod.", "")] = osd[_STATE].pop(k)
            for g in osd[_PG]:
                params = [k.replace("_orig_mod.", "") for k in g[_PARAMS]]
                g[_PARAMS] = params
        else:
            params = list(chain.from_iterable(g[_PARAMS] for g in optim.param_groups))
            param_pid_mapping = dict(zip(params, range(len(params))))
            fqn_pid_mapping = {}
            for key, param in model.named_parameters():
                fqns = _get_fqns(model, key)
                if len(fqns) != 1:
                    raise AssertionError(
                        f"Expected 1 FQN for key '{key}', got {len(fqns)}"
                    )
                fqn = next(iter(fqns))
                if param not in param_pid_mapping:
                    continue
                # pyrefly: ignore [bad-index]
                pid = param_pid_mapping[param]
                fqn_pid_mapping[fqn] = pid
                # pyrefly: ignore [unsupported-operation]
                fqn_pid_mapping[pid] = fqn

            # Only convert top-level parameter IDs to FQNs, preserve nested key types
            for key in list(osd[_STATE].keys()):
                fqn = fqn_pid_mapping[key]
                # Move the entire state dict value (which may contain nested integer keys)
                # without modifying its internal structure
                osd[_STATE][fqn] = osd[_STATE].pop(key)

            for group in osd[_PG]:
                group[_PARAMS] = [fqn_pid_mapping[pid] for pid in group[_PARAMS]]

        if not osd:
            continue

        cast(DictValueType, optim_state_dict[_STATE]).update(osd[_STATE])
        cast(ListDictValueType, optim_state_dict[_PG]).extend(osd[_PG])

    if info.flatten_optimizer_state_dict:
        optim_state_dict = cast(
            OptimizerStateType, _flatten_optim_state_dict(optim_state_dict)
        )

    return _maybe_full_or_cpu_state_dict(optim_state_dict, info)


def _split_optim_state_dict(
    model: nn.Module,
    optim: torch.optim.Optimizer,
    optim_state_dict: OptimizerStateType,
    info: _StateDictInfo,
) -> OptimizerStateType:
    """
    Extract the corresponding optim state_dict from ``optim_state_dict`` for
    ``optim`` and return the result optim state_dict.

    Args:
        model (nn.Module): the root model.
        optim (torch.optim.Optimizer): the optimizer.
        optim_state_dict (Dict[str, ValueType]): the superset optim state_dict that
            contains the optim state_dict of ``optim``.
        info (_StateDictInfo): state dict information.

    Returns:
        The optim state_dict of ``optim``.
    """

    state: DictValueType = {}
    pg_state: ListDictValueType = []
    return_osd: OptimizerStateType = {_STATE: state, _PG: pg_state}
    pg_mapping: dict[int, int] = {}

    if all(isinstance(k, int) for k in cast(DictValueType, optim_state_dict[_STATE])):
        return optim_state_dict

    for param_group in optim.param_groups:
        pg_state.append({_PARAMS: []})
        for param in param_group[_PARAMS]:
            for fqn in info.fqn_param_mapping[param]:
                if fqn in info.shared_params_mapping:
                    in_params = False
                    for loaded_param_group in cast(
                        ListDictValueType, optim_state_dict[_PG]
                    ):
                        if fqn in cast(list[str], loaded_param_group[_PARAMS]):
                            in_params = True
                            break
                else:
                    in_params = True
                if not in_params:
                    continue

                params = pg_state[-1][_PARAMS]
                if not isinstance(params, list):
                    raise AssertionError(f"Expected list, got {type(params)}")
                params.append(fqn)
                if param.requires_grad:
                    if fqn in cast(DictValueType, optim_state_dict[_STATE]):
                        state[fqn] = cast(DictValueType, optim_state_dict[_STATE])[fqn]
                    elif info.strict:
                        raise RuntimeError(
                            f"Missing optimizer state for parameter '{fqn}' in checkpoint. "
                            "The parameter requires gradients but has no saved optimizer state. "
                            "To load anyway, use StateDictOptions(strict=False)."
                        )
                for loaded_param_group in cast(
                    ListDictValueType, optim_state_dict[_PG]
                ):
                    if fqn in cast(list[str], loaded_param_group[_PARAMS]):
                        pg_mapping[id(loaded_param_group)] = len(return_osd[_PG]) - 1

        if len(param_group[_PARAMS]) == 0:
            # Param_group with empty params.
            ret = []
            for loaded_param_group in cast(ListDictValueType, optim_state_dict[_PG]):
                if len(cast(list[str], loaded_param_group[_PARAMS])) == 0:
                    ret.append(loaded_param_group)
            if len(ret) != 1:
                raise ValueError(
                    "There are param groups that have zero parameters. "
                    "In such a case, DSD only support exactly one param group "
                    "with zero parameters."
                    "But the loaded state_dict has zero or more than one param groups "
                    "that have zero parameters."
                )
            if len(optim_state_dict[_PG]) != len(optim.param_groups):
                raise ValueError(
                    "When there is a parameter group that has zero parameters, "
                    "multiple optimizers are not supported."
                )
            pg_mapping[id(loaded_param_group)] = len(return_osd[_PG]) - 1

    for param_group in cast(ListDictValueType, optim_state_dict[_PG]):
        pg_idx = pg_mapping.get(id(param_group), -1)
        if pg_idx == -1:
            continue

        for key, value in param_group.items():
            if key == _PARAMS:
                continue
            # TODO: check if value is the same if exists.
            pg_state[pg_idx][key] = value

    return return_osd


@torch.no_grad()
def _load_optim_state_dict(
    model: nn.Module,
    optimizers: tuple[torch.optim.Optimizer, ...],
    state_dict: OptimizerStateType,
    info: _StateDictInfo,
) -> None:
    if not info.handle_optim:
        return

    for optim in optimizers:
        _init_optim_state(optim)
        if state_dict:
            if _STATE in state_dict:
                optim_state_dict = _split_optim_state_dict(
                    model, optim, state_dict, info
                )
            else:
                optim_state_dict = _unflatten_optim_state_dict(
                    optim, cast(dict[str, ValueType], state_dict), info
                )
        else:
            optim_state_dict = {}
        if info.fsdp_modules:
            # We need to specially handle FlatParameter FSDP as
            # FlatParameter FSDP converts the FQNs.
            for original_fqn, _ in model.named_parameters():
                fqns = _get_fqns(model, original_fqn)
                fqns_with_compiler = _get_fqns(
                    model, original_fqn, skip_compiler_prefix=False
                )
                if fqns == fqns_with_compiler:
                    continue

                if len(fqns) != 1:
                    raise AssertionError(
                        f"Expected 1 FQN for '{original_fqn}', got {len(fqns)}"
                    )
                fqn = fqns.pop()
                fqn_with_compiler = fqns_with_compiler.pop()
                for g in optim_state_dict[_PG]:
                    val = cast(dict[str, Any], g)
                    params = [
                        key.replace(fqn, fqn_with_compiler) for key in val[_PARAMS]
                    ]
                    val[_PARAMS] = params
                osd_state = cast(DictValueType, optim_state_dict[_STATE])
                for k in list(osd_state.keys()):
                    if fqn in k:
                        osd_state[k.replace(fqn, fqn_with_compiler)] = osd_state.pop(k)

            with info.fsdp_context():
                optim_state_dict = FSDP.optim_state_dict_to_load(
                    model,
                    optim,
                    optim_state_dict,
                    group=_get_fsdp_process_group(model, info),
                )
        elif info.full_state_dict:
            info.full_state_dict = False
            local_state_dict = _get_optim_state_dict(model, (optim,), info)
            info.full_state_dict = True
            device = None

            def _device(t):
                if t.dim() > 0:
                    nonlocal device
                    if device is None:
                        device = t.device
                    elif device != t.device:
                        raise ValueError("Device mismatch")
                return t

            _ = tree_map_only(torch.Tensor, _device, local_state_dict)
            if device is None:
                raise AssertionError("Expected device to be set")
            flatten_osd, osd_mapping = _flatten_state_dict(optim_state_dict)
            flatten_local_osd, local_osd_mapping = _flatten_state_dict(local_state_dict)
            if info.broadcast_from_rank0:
                _broadcast_state_dict(flatten_osd, flatten_local_osd, device=device)
            else:
                _distribute_state_dict(flatten_osd, flatten_local_osd, device=device)
            # The modifications listed seek to address the problem where optim might possess
            # dissimilar parameters in comparison to optim_state_dict. This is achieved by
            # incorporating differential parameters within local, which may result in optim
            # having additional parameters ultimately.
            for optim_key in flatten_osd:
                if optim_key not in flatten_local_osd:
                    if optim_key not in osd_mapping:
                        raise AssertionError(
                            f"Expected key '{optim_key}' in osd_mapping"
                        )
                    flatten_local_osd[optim_key] = flatten_osd[optim_key]
                    local_osd_mapping[optim_key] = osd_mapping[optim_key]
            optim_state_dict = _unflatten_state_dict(
                flatten_local_osd, local_osd_mapping
            )
            for pg in optim_state_dict[_PG]:
                if _PARAMS not in pg:
                    cast(dict[str, ValueType], pg)[_PARAMS] = []

        # Note that we do not have to convert the FQN back to param id here if
        # order in optim.param_groups[idx][_PARAMS] is the same as the one in
        # optim_state_dict[_PG][idx][_PARAMS].
        _state_dict_fn(optim, "load_state_dict")(state_dict=optim_state_dict)


def get_model_state_dict(
    model: nn.Module,
    *,
    submodules: set[nn.Module] | None = None,
    options: StateDictOptions | None = None,
) -> dict[str, ValueType]:
    """
    Return the model state_dict of ``model``.

    See ``get_state_dict`` for the detail usage.

    Args:
        model (nn.Module): the nn.Module to the model.
        submodules (deprecated): Optional[set[nn.Module]]: only return the model parameters
            that belong to the submodules.
        options (StateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be returned. See
            `StateDictOptions` for the details.

    Returns:
        The state_dict for ``model``.

    :rtype: typing.Dict[str, ValueType]
    """
    with _gc_context():
        info = _verify_options(
            model,
            (),
            optim_only=False,
            submodules=submodules,
            options=options,
        )
        model_state_dict = _get_model_state_dict(model, info)
        _verify_state_dict(model_state_dict, {}, info)
        return model_state_dict


def get_optimizer_state_dict(
    model: nn.Module,
    optimizers: torch.optim.Optimizer | Iterable[torch.optim.Optimizer],
    *,
    submodules: set[nn.Module] | None = None,
    options: StateDictOptions | None = None,
) -> OptimizerStateType:
    """
    Return the combined state_dict for optimizers.

    See ``get_state_dict`` for the detail usage.

    Args:
        model (nn.Module): the nn.Module to the model.
        optimizers (Union[None, Optimizer, Iterable[Optimizer]]):
            The optimizers that are used to optimize ``model``.
        submodules (deprecated): Optional[set[nn.Module]]: only return the model parameters
            that belong to the submodules.
        options (StateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be returned. See
            `StateDictOptions` for the details.

    Returns:
        The state_dict for ``optimizers``.

    :rtype: OptimizerStateType
    """
    with _gc_context():
        optimizers = (
            (optimizers,)
            if isinstance(optimizers, torch.optim.Optimizer)
            else tuple(optimizers)
        )
        info = _verify_options(
            model,
            optimizers,
            optim_only=True,
            submodules=submodules,
            options=options,
        )
        optim_state_dict = _get_optim_state_dict(model, optimizers, info)
        _verify_state_dict({}, optim_state_dict, info)
        return optim_state_dict


def get_state_dict(
    model: nn.Module,
    optimizers: torch.optim.Optimizer | Iterable[torch.optim.Optimizer],
    *,
    submodules: set[nn.Module] | None = None,
    options: StateDictOptions | None = None,
) -> tuple[dict[str, ValueType], OptimizerStateType]:
    """
    Return the model state_dict and optimizers state_dict.

    ``get_state_dict`` can process any module that is parallelized by PyTorch
    FSDP/fully_shard, DDP/replicate, tensor_parallel/parallelize_module, and any
    combination of these parallelisms. The main functions of ``get_state_dict``
    are: 1.) returning a model and optimizer state_dict that can be resharded
    with a different number of trainers and/or different parallelisms.
    2.) hiding the parallelism-specific state_dict APIs. Users don't have to call
    these APIs.
    3.) sanity checking the result state_dict.

    The keys of the result state dictionary are the canonical FQNs (Fully
    Qualified Names).  A canonical FQN refers to the FQN based on a parameter's
    position in an nn.Module hierarchy. More specifically, a canonical FQN to a
    parameter is the FQN returned by ``module.named_parameters()`` or
    ``module.named_buffers()`` when the module is not distributed by any
    parallelisms. Since the optimizer internally uses parameter IDs to represent
    a parameter, there will be a conversion from the parameter IDs to the
    canonical FQNs when calling this API.

    ``get_state_dict`` can also process a module that is not parallelized. In
    such a case, ``get_state_dict`` only performs one function -- converting the
    optimizer parameter IDs to the canonical FQNs.

    Example:
        >>> # xdoctest: +SKIP
        >>> import torch
        >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        >>> from torch.nn.parallel import DistributedDataParallel as DDP
        >>> from torch.distributed.checkpoint.state_dict import get_state_dict

        >>> fsdp_model = FSDP(copy.deepcopy(model))
        >>> fsdp_optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> ddp_model = DDP(copy.deepcopy(model))
        >>> ddp_optim = torch.optim.Adam(model.parameters(), lr=1e-3)


        >>> ddp_state_dict, ddp_optim_state_dict = get_state_dict(ddp_model, ddp_optim)
        >>> fsdp_state_dict, fsdp_optim_state_dict = get_state_dict(
        ...     fsdp_model, fsdp_optim
        ... )

        >>> # if we simply call ddp_model.state_dict() and fsdp_model.state_dict(),
        >>> # the asserts will fail.
        >>> assert ddp_state_dict == fsdp_state_dict
        >>> assert ddp_optim_state == fsdp_optim_state_dict


    Args:
        model (nn.Module): the nn.Module to the model.
        optimizers (Union[None, Optimizer, Iterable[Optimizer]]):
            The optimizers that are used to optimize ``model``.
        submodules (deprecated): Optional[set[nn.Module]]: only return the model parameters
            that belong to the submodules.
        options (StateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be returned. See
            `StateDictOptions` for the details.

    Returns:
        ``Tuple`` that contain model state_dict and optimizer state_dict.

    :rtype: typing.Tuple[typing.Dict[str, ValueType], OptimizerStateType]
    """

    with _gc_context():
        optimizers = (
            (optimizers,)
            if isinstance(optimizers, torch.optim.Optimizer)
            else tuple(optimizers)
        )
        info = _verify_options(
            model,
            optimizers,
            optim_only=False,
            submodules=submodules,
            options=options,
        )
        model_state_dict = _get_model_state_dict(model, info)
        optim_state_dict = _get_optim_state_dict(model, optimizers, info)
        _verify_state_dict(model_state_dict, optim_state_dict, info)
        return model_state_dict, optim_state_dict


def _unflatten_model_state_dict(
    model: nn.Module,
    state_dict: dict[nn.Module, dict[str, ValueType]] | dict[str, ValueType],
) -> dict[str, ValueType]:
    if not state_dict:
        return {}

    if isinstance(next(iter(state_dict.keys())), nn.Module):
        warnings.warn(
            "Passing model_state_dict as a ``Dict[nn.Module, Dict[str, Any]]``"
            "is deprecated and will be removed in 2.5. If you need this "
            "feature, please preprocess the model_state_dict to achieve the "
            "same functionality.",
            FutureWarning,
            stacklevel=2,
        )
        cast_state_dict = cast(dict[nn.Module, dict[str, ValueType]], state_dict)
        new_state_dict: dict[str, ValueType] = {}
        for submodule, sub_state_dict in cast_state_dict.items():
            for name, m in model.named_modules():
                if m != submodule:
                    continue

                fqns = _get_fqns(model, name)
                if len(fqns) != 1:
                    raise AssertionError(
                        "FQNs for a submodule should only have 1 element"
                    )
                prefix = f"{next(iter(fqns))}."
                new_state_dict.update(
                    {prefix + subfqn: value for subfqn, value in sub_state_dict.items()}
                )
        return new_state_dict
    else:
        return cast(dict[str, ValueType], state_dict)


def set_model_state_dict(
    model: nn.Module,
    model_state_dict: dict[str, ValueType],
    *,
    options: StateDictOptions | None = None,
) -> _IncompatibleKeys:
    """Load the model state_dict.

    The counterpart of ``get_model_state_dict`` to set the state_dict to the
    model. See ``set_state_dict`` for the detail usage.

    Args:
        model (nn.Module): the nn.Module to the model.
        model_state_dict: (Dict[str, ValueType]):
           the model state_dict to load. If the key of the ``model_state_dict``
           is nn.Module, the key is a submodule of ``model`` and the value should
           be the state_dict of the submodule. When loading the state_dict,
           the prefix of the submodule will be append to the state_dict.
        options (StateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be loaded. See
            `StateDictOptions` for the details.

    Returns:
        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing the missing keys
            * **unexpected_keys** is a list of str containing the unexpected keys

    :type model_state_dict: typing.Dict[str, ValueType]
    """
    model_state_dict: dict[str, ValueType] = _unflatten_model_state_dict(
        model, model_state_dict
    )
    with _gc_context():
        info = _verify_options(model, (), optim_only=False, options=options)

        _verify_state_dict(model_state_dict, {}, info)
        return _load_model_state_dict(model, model_state_dict, info)


def set_optimizer_state_dict(
    model: nn.Module,
    optimizers: torch.optim.Optimizer | Iterable[torch.optim.Optimizer],
    optim_state_dict: OptimizerStateType,
    *,
    options: StateDictOptions | None = None,
) -> None:
    """Load the optimizers state_dict.

    The counterpart of ``get_optimizer_state_dict`` to set the state_dict to the
    optimizers. See ``set_state_dict`` for the detail usage.

    WARN: ``set_optimizer_state_dict`` can only be called before ``backward()`` or after
        ``step()`` is called on the optimizers. Otherwise, the optimizer states won't be
        initialized correctly.

    Args:
        model (nn.Module): the nn.Module to the model.
        optimizers (Union[Optimizer, Iterable[Optimizer]]):
            The optimizers that are used to optimize ``model``.
        optim_state_dict: OptimizerStateType:
            the optimizer state_dict to load.
        options (StateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be loaded. See
            `StateDictOptions` for the details.

    Returns:
        None

    :type optim_state_dict: typing.OptimizerStateType
    """
    with _gc_context():
        optimizers = (
            (optimizers,)
            if isinstance(optimizers, torch.optim.Optimizer)
            else tuple(optimizers)
        )
        info = _verify_options(model, optimizers, optim_only=True, options=options)

        _verify_state_dict({}, optim_state_dict, info)
        _load_optim_state_dict(model, optimizers, optim_state_dict, info)


def set_state_dict(
    model: nn.Module,
    optimizers: torch.optim.Optimizer | Iterable[torch.optim.Optimizer],
    *,
    model_state_dict: dict[str, ValueType],
    optim_state_dict: OptimizerStateType,
    options: StateDictOptions | None = None,
) -> _IncompatibleKeys:
    """Load the model state_dict and optimizers state_dict.

    The counterpart of ``get_state_dict`` to set the state_dict to the model and
    optimizers.  The given ``model_state_dict`` and ``optim_state_dict`` do not
    have to be returned by ``get_state_dict`` but must meet the following
    requirements: 1) all FQNs are canonical FQNs as defined in ``get_state_dict``,
    2) if a tensor is sharded, it must be either a ShardedTensor or DTensor,
    3) optimizer state_dict cannot contain the parameter IDs; the keys should be
    the canonical FQNs.

    WARN: ``set_state_dict`` can only be called before ``backward()`` or after ``step()``
        is called on the optimizers. Otherwise, the optimizer states won't be initialized
        correctly.

    Args:
        model (nn.Module): the nn.Module to the model.
        optimizers (Union[Optimizer, Iterable[Optimizer]]):
            The optimizers that are used to optimize ``model``.
        model_state_dict: (Union[Dict[nn.Module, Dict[str, ValueType]], Dict[str, ValueType]]):
           the model state_dict to load. If the key of the ``model_state_dict``
           is nn.Module, the key is a submodule of ``model`` and the value should
           be the state_dict of the submodule. When loading the state_dict,
           the prefix of the submodule will be append to the state_dict.
        optim_state_dict: OptimizerStateType:
            the optimizer state_dict to load.
        options (StateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be loaded. See
            `StateDictOptions` for the details.

    Returns:
        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing the missing keys of the model state_dict.
            * **unexpected_keys** is a list of str containing the unexpected keys of the model state_dict.

    :type model_state_dict: typing.Dict[str, ValueType]
    :type optim_state_dict: typing.OptimizerStateType
    """

    model_state_dict: dict[str, ValueType] = _unflatten_model_state_dict(
        model, model_state_dict
    )
    with _gc_context():
        optimizers = (
            (optimizers,)
            if isinstance(optimizers, torch.optim.Optimizer)
            else tuple(optimizers)
        )
        info = _verify_options(
            model, optimizers, optim_only=not model_state_dict, options=options
        )

        _verify_state_dict(model_state_dict, optim_state_dict, info)
        _load_optim_state_dict(model, optimizers, optim_state_dict, info)
        return _load_model_state_dict(model, model_state_dict, info)


# TODO: correct the state_dict function signature.
# TODO: this API is not yet fully tested. Make it private
@no_type_check
def _patch_model_state_dict(
    model: nn.Module,
    *,
    options: StateDictOptions | None = None,
) -> None:
    """Patch the ``state_dict`` and ``load_state_dict`` attributes of ``model``.

    Patch the ``state_dict`` and ``load_state_dict`` attributes of ``model`` to
    be a partial function to call ``get_state_dict`` and ``set_state_dict``.

    Example:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.checkpoint.state_dict import patch_model_state_dict

        model = fsdp(model)
        patch_model_state_dict(model)

    Args:
        model (nn.Module): the nn.Module to the model.
        options (StateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be loaded. See
            `StateDictOptions` for the details.
    Returns:
        None
    """

    _state_dict_call = functools.partial(
        get_model_state_dict,
        model=model,
        options=options,
    )

    def state_dict_call():
        return _state_dict_call()

    model.state_dict = state_dict_call

    _load_state_dict_call = functools.partial(
        set_model_state_dict,
        model=model,
        options=options,
    )

    def load_state_dict_call(state_dict: dict[str, Any]):
        _load_state_dict_call(model_state_dict=state_dict)

    model.load_state_dict = load_state_dict_call

    _patched_state_dict.add(state_dict_call)
    _patched_state_dict.add(load_state_dict_call)


# TODO: correct the load_state_dict function signature.
# TODO: this API is not yet fully tested. Make it private
@no_type_check
def _patch_optimizer_state_dict(
    model: nn.Module,
    *,
    optimizers: tuple[torch.optim.Optimizer, ...],
    options: StateDictOptions | None = None,
) -> None:
    """Patch the ``state_dict`` and ``load_state_dict`` attributes of ``optimizers``.

    Patch the ``state_dict`` and ``load_state_dict`` attributes of ``optimizers`` to
    be a partial function to call ``get_state_dict`` and ``set_state_dict``.

    Note that if there are multiple optimizers, all of the optimizers will be patched.
    So users only need to call one of the state_dict() to get the full result.

    Example:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.checkpoint.state_dict import patch_model_state_dict

        model = fsdp(model)
        patch_model_state_dict(model)

    Args:
        model (nn.Module): the nn.Module to the model.
        options (StateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be loaded. See
            `StateDictOptions` for the details.
    Returns:
        None
    """

    _state_dict_call = functools.partial(
        get_optimizer_state_dict,
        model=model,
        optimizers=optimizers,
        options=options,
    )

    def state_dict_call():
        return _state_dict_call()

    _load_state_dict_call = functools.partial(
        set_optimizer_state_dict,
        model=model,
        optimizers=optimizers,
        options=options,
    )

    def load_state_dict_call(state_dict: dict[str, Any]):
        _load_state_dict_call(optim_state_dict=state_dict)

    _patched_state_dict.add(state_dict_call)
    _patched_state_dict.add(load_state_dict_call)
    optimizers = (
        (optimizers,)
        if isinstance(optimizers, torch.optim.Optimizer)
        else tuple(optimizers)
    )
    for optim in optimizers:
        optim.state_dict = state_dict_call
        optim.load_state_dict = load_state_dict_call
<|endoftext|>import os
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, cast
from typing_extensions import deprecated, Protocol, runtime_checkable

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed._state_dict_utils import _copy_state_dict, _create_cpu_state_dict
from torch.distributed.checkpoint._pg_transport import PGTransport
from torch.distributed.checkpoint._state_dict_stager import StateDictStager
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE


__all__ = ["AsyncStager", "BlockingAsyncStager", "DefaultStager", "StagingOptions"]

"""
Experimental staging module for PyTorch Distributed Checkpointing.
This module provides advanced staging capabilities for checkpoints including:
- Asynchronous staging using ThreadPoolExecutor
- Pinned memory allocation for faster CPU-GPU transfers
- Shared memory support for multi-process scenarios
- Non-blocking CUDA operations with stream synchronization
- Caching of frequently used storages for efficient memory management
- Automatic resource cleanup and memory management
Classes:
    AsyncStager: Protocol defining the staging interface
    StagingOptions: Configuration dataclass for staging behavior
    DefaultStager: Default implementation with comprehensive staging features
    BlockingAsyncStager: Implementation of AsyncStager which stages the state_dict
    on CPU RAM and blocks until the copy is complete. Please use DefaultStager instead.
"""


@runtime_checkable
class AsyncStager(Protocol):
    """
    This protocol is meant to provide customization and extensibility for dcp.async_save, allowing users
    to customize how data is staged previous to executing the usual dcp.save path in parallel.
    The expected order of operations (concretely defined in `torch.distributed.state_dict_saver.async_save`)
    is the following:

    1. AsyncStager.stage_data(state_dict):
        This call gives the AsyncStager the opportunity to 'stage'
        the state_dict. The expectation and purpose of staging in this context is to create a "training-safe"
        representation of the state dict, meaning that any updates to module data after staging is complete
        should not be reflected in the state dict returned from this method. For example, in the default
        case a copy of the entire state dict is created on CPU RAM and returned here, allowing users
        to continue training without risking changes to data which is being serialized.

    2. dcp.save is called on the state_dict returned from stage in parallel. This call is responsible
        for serializing the state_dict and writing it to storage.

    3. If AsyncStager.should_synchronize_after_execute is True, this method will be called immediately after
        the serialization thread starts and before returning from dcp.async_save. If this is set to False,
        the assumption is the user has defined a custom synchronization point for the purpose of further
        optimizing save latency in the training loop (for example, by overlapping staging with the
        forward/backward pass), and it is the respondsibility of the user to call `AsyncStager.synchronize_staging`
        at the appropriate time.

    """

    # default to True since the common case is to stage synchronously
    _synchronize_after_execute: bool = True

    @property
    def should_synchronize_after_execute(self) -> bool:
        """
        Whether to synchronize after executing the stage.
        """
        return self._synchronize_after_execute

    def stage(
        self, state_dict: STATE_DICT_TYPE
    ) -> Future[STATE_DICT_TYPE] | STATE_DICT_TYPE:
        """
        Returns a "staged" copy of `state_dict`. The expectation of the staged copy is that it is
        inoculated from any updates incurred after the stage call is complete.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement stage method"
        )

    @deprecated(
        "`synchronize_staging` is deprecated and will be removed in future versions."
        "Please use staging_future from AsyncSaveResponse instead.",
        category=FutureWarning,
    )
    def synchronize_staging(self) -> None:
        """
        In the case `stage` is async in some way, this method should be called to ensure staging
        is complete and it is safe to begin modifying the original `state_dict`
        """

    def close(self) -> None:
        """
        Clean up all resources used by the stager.
        """


@dataclass
class StagingOptions:
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


class DefaultStager(AsyncStager):
    """
    DefaultStager provides a full-featured staging implementation that combines
    multiple optimization techniques for efficient checkpoint preparation.

    The staging process works as follows:
    1. State dictionary is submitted for staging (sync or async)
    2. Tensors are copied from GPU to optimized CPU storage
    3. CUDA operations are synchronized if non-blocking copies are used
    4. Staged state dictionary is returned or made available via Future

    Usage Patterns:
        # Synchronous staging
        stager = DefaultStager(StagingOptions(use_async_staging=False))
        staged_dict = stager.stage(state_dict)
        stager.close()

        # Asynchronous staging
        stager = DefaultStager(StagingOptions(use_async_staging=True))
        future = stager.stage(state_dict)
        # ... do other work ...
        staged_dict = future.result()
        stager.close()

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
        config: StagingOptions = StagingOptions(),
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

        self._staging_future: Future[STATE_DICT_TYPE] | None = None

    def stage(
        self,
        state_dict: STATE_DICT_TYPE,
        **kwargs: Any,
    ) -> STATE_DICT_TYPE | Future[STATE_DICT_TYPE]:
        """
        This function is responsible for staging staging the state_dict.
        See class docstring for more details on staging.
        If use_async_staging is True, it will return a Future object that will be
        fulfilled when staging is complete.
        If use_async_staging is False, it will return the fully staged state_dict.

        Args:
            state_dict (STATE_DICT_TYPE): The state_dict to be staged.
        """
        if self._config.use_async_staging:
            if self._staging_executor is None:
                raise AssertionError(
                    "staging_executor should not be None for async staging"
                )
            self._staging_future = self._staging_executor.submit(
                self._stage,
                state_dict,
                **kwargs,
            )
            return self._staging_future
        else:
            return self._stage(state_dict, **kwargs)

    def _stage(self, state_dict: STATE_DICT_TYPE, **kwargs: Any) -> STATE_DICT_TYPE:
        if self._config.use_non_blocking_copy:
            if not (self._staging_stream or not self._config.use_async_staging):
                raise AssertionError(
                    "Non-blocking copy in a background thread for async staging needs staging_stream to be initialized."
                )
            with (
                self._staging_stream
                if self._staging_stream is not None
                else nullcontext()
            ):
                state_dict = self._state_dict_stager.stage(
                    state_dict, non_blocking=self._config.use_non_blocking_copy
                )
            # waits for the enqued copy operations to finish.
            self._staging_stream.synchronize() if self._staging_stream else torch.accelerator.synchronize()
        else:
            state_dict = self._state_dict_stager.stage(state_dict, non_blocking=False)

        return state_dict

    def close(self) -> None:
        """
        Clean up all resources used by the DefaultStager. Shuts down the ThreadPoolExecutor
        used for async staging operations and cleans up the underlying StateDictStager's
        cached storages. Should be called when the stager is no longer needed to prevent
        resource leaks, especially in long-running applications. After calling close(),
        the stager should not be used for further staging operations.

        Example Usage:
            stager = DefaultStager(StagingOptions(use_async_staging=True))
            future = stager.stage(state_dict)
            result = future.result()
            stager.close()  # Clean up all resources
        """
        if self._staging_executor:
            self._staging_executor.shutdown(wait=True)
        self._state_dict_stager.close()

    def synchronize_staging(self) -> None:
        """
        When use_async_staging is True, this method will wait until staging is complete.
        If use_async_staging is False, this method is a no-op.
        """
        if self._staging_future is not None:
            self._staging_future.result()


class BlockingAsyncStager(AsyncStager):
    """
    An implementation of AsyncStager which stages the state_dict on CPU RAM and blocks until the copy is complete.
    This implementation also provides an option to optimize stage latency using pinned memory.

    N.B. synchronize_staging is a no-op in this case.


    """

    # default to True since the common case is to stage synchronously
    _synchronize_after_execute: bool = False

    def __init__(
        self,
        cache_staged_state_dict: bool = False,
        type_check: bool = False,
    ):
        """
        Initializes the BlockingAsyncStager.

        Args:
            cache_staged_state_dict: Whether to cache the staged state_dict. This option decreases staging latency
                at the cost of increased memory usage. Additionally, if this parameter is set to True, it's the expectation
                that the stager is maintained and reused for multiple dcp.async_save calls. Default to False.
            type_check: Whether to perform a type check during cpu_offload. Defaults to False.

        """
        self.cache_staged_state_dict = cache_staged_state_dict
        self.type_check = type_check
        self.state_dict_cache: STATE_DICT_TYPE | None = None

    def stage(self, state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
        """
        Returns a copy of `state_dict` on the CPU.
        """

        if not self.cache_staged_state_dict:
            staged_state_dict = _create_cpu_state_dict(state_dict)
            _copy_state_dict(state_dict, staged_state_dict, type_check=self.type_check)
            return staged_state_dict

        if self.state_dict_cache is None:
            self.state_dict_cache = _create_cpu_state_dict(state_dict, pin_memory=True)
        return _copy_state_dict(state_dict, self.state_dict_cache)

    def synchronize_staging(self) -> None:
        """
        No-op function, since staging is blocking.
        """

    def close(self) -> None:
        pass


class _ReplicationStager(AsyncStager):
    """
    An AsyncStager implementation that replicates state_dict across training ranks
    using PGTransport.

    Args:
        pg: ProcessGroup for distributed communication
        timeout: Timeout for communication operations
        device: Device to use for tensor operations
        storage_dir: Directory to store persisted state_dicts

    Warning: This is experimental and subject to change.
    """

    _synchronize_after_execute: bool = False

    def __init__(
        self,
        pg: ProcessGroup,
        timeout: timedelta = timedelta(minutes=30),
        device: torch.device = torch.device("cpu"),
        storage_dir: str | None = None,
    ):
        self._pg = pg
        self._timeout = timeout
        self._device = device
        self._transport = PGTransport(pg, timeout, device, None)

        # Set up storage directory for persisting exchanged state_dicts
        if storage_dir is None:
            self._storage_dir = tempfile.mkdtemp(prefix="replication_stager_")
        else:
            self._storage_dir = storage_dir
        os.makedirs(self._storage_dir, exist_ok=True)

    def stage(
        self, state_dict: STATE_DICT_TYPE
    ) -> Future[STATE_DICT_TYPE] | STATE_DICT_TYPE:
        """
        Stage the state_dict by replicating it across ranks. Returns a state_dict representing
        the received replica.

        Perform the actual replication logic. Creates bidirectional pairs where each rank exchanges
        state_dict with its partner at (rank + world_size//2) % world_size.
        Uses simple rank-based ordering to prevent deadlocks.

        Assumes world_size is always even.
        """
        if not dist.is_initialized():
            return state_dict

        world_size = dist.get_world_size()

        current_rank = dist.get_rank()

        # Calculate partner rank using half-world offset
        # creates bidirectional pairs for replication.
        offset = world_size // 2
        partner_rank = (current_rank + offset) % world_size

        # Use simple rank-based ordering to prevent deadlocks.
        # Lower-numbered rank sends first, higher-numbered rank receives first.
        if current_rank < partner_rank:
            # Send first, then receive
            self._transport.send_checkpoint([partner_rank], state_dict)
            received_state_dict = self._transport.recv_checkpoint(partner_rank)
        else:
            # Receive first, then send
            received_state_dict = self._transport.recv_checkpoint(partner_rank)
            self._transport.send_checkpoint([partner_rank], state_dict)

        # Persist the received state_dict for future discoverability
        received_state_dict = cast(STATE_DICT_TYPE, received_state_dict)
        self._persist_state_dict(received_state_dict, current_rank, partner_rank)

        return received_state_dict

    def _persist_state_dict(
        self, state_dict: STATE_DICT_TYPE, current_rank: int, partner_rank: int
    ) -> None:
        """
        Persist the received state_dict to disk for future discoverability.
        Only keeps one replica per rank, overwriting any previous replica.
        Uses atomic write pattern (temp file + rename).

        Args:
            state_dict: The state_dict received from partner rank
            current_rank: Current rank that received the state_dict
            partner_rank: Rank that sent the state_dict
        """
        final_path = self._get_persisted_path(current_rank, partner_rank)
        temp_path = final_path + ".tmp"

        try:
            # Ensure parent directory exists and is writable
            os.makedirs(os.path.dirname(final_path), exist_ok=True)

            # Write to temporary file with explicit flushing
            with open(temp_path, "wb") as f:
                torch.save(state_dict, f)
                # Flush application buffers to OS buffers
                f.flush()
                # Force OS buffers to disk for durability
                os.fsync(f.fileno())

            # Atomic rename to final location
            os.rename(temp_path, final_path)
        except Exception as e:
            # Clean up temp file if it exists
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass  # Ignore cleanup errors
            # Re-raise the original exception with more context
            raise RuntimeError(
                f"Failed to persist state_dict from rank {partner_rank} to rank {current_rank}: {e}"
            ) from e

    def _get_persisted_path(self, current_rank: int, partner_rank: int) -> str:
        """
        Get the file path where a state_dict would be persisted.

        Args:
            current_rank: Current rank

        Returns:
            File path for the persisted state_dict
        """
        filename = f"rank_{current_rank}_replica_partner_{partner_rank}.pt"
        return os.path.join(self._storage_dir, filename)

    def synchronize_staging(self) -> None:
        """
        No-op function, since staging is blocking.
        """

    def close(self) -> None:
        """
        Clean up resources. Persisted files are intentionally left for future discovery.
        """
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates

import copy
from typing import TYPE_CHECKING

import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import Shard, ShardedTensor, ShardMetadata
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.remote_device import _remote_device

from ._traverse import OBJ_PATH, set_element, STATE_DICT_ITEM, traverse_state_dict
from .utils import _element_wise_add, _normalize_device_info


if TYPE_CHECKING:
    from torch.distributed._shard.sharded_tensor.metadata import ShardedTensorMetadata


# TODO: We need to refactor this code.
def _flatten_sharded_tensors(state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
    r"""
    Transform ``state_dict`` by flattening all nested ShardedTensor instances found.

    The resulting ShardedTensor instances are only correct regarding the local shard and
    MUST not be used for any other purpose but checkpointing, as no operator will work with them.

    This function should be used in conjunction with a state_dict produced by FSDP's
    StateDictType.SHARDED_STATE_DICT methods.
    """
    new_state_dict: STATE_DICT_TYPE = {}

    def rewrite_dict(path: OBJ_PATH, value: STATE_DICT_ITEM) -> None:
        if not isinstance(value, ShardedTensor):
            set_element(new_state_dict, path, value)
            return
        shards = value.local_shards()

        if len(shards) == 0:
            return
        if len(shards) != 1:
            set_element(new_state_dict, path, value)
            return

        outer_shard = shards[0]

        inner_st = outer_shard.tensor
        if not isinstance(inner_st, ShardedTensor):
            set_element(new_state_dict, path, value)
            return

        if len(inner_st.local_shards()) != 1:
            raise ValueError("Cannot handle inner tensor with more than 1 shard")
        inner_shard = inner_st.local_shards()[0]

        local_shards = [
            Shard(
                tensor=inner_shard.tensor,
                metadata=ShardMetadata(
                    shard_offsets=_element_wise_add(
                        outer_shard.metadata.shard_offsets,
                        inner_shard.metadata.shard_offsets,
                    ),
                    shard_sizes=inner_shard.metadata.shard_sizes,
                    placement=f"rank:{dist.get_rank()}/{inner_shard.tensor.device}",
                ),
            )
        ]

        st_meta: ShardedTensorMetadata = copy.deepcopy(value.metadata())
        other_rank = 0 if dist.get_rank() > 0 else 1
        device_info = _normalize_device_info(inner_shard.tensor.device.type, 0)

        # Remove the outer ST shard the inner ST covers
        for i, shard_md in enumerate(st_meta.shards_metadata):
            if shard_md.shard_offsets == outer_shard.metadata.shard_offsets:
                st_meta.shards_metadata.pop(i)
                break

        # Attribute other rank for the other shards
        for shard_md in st_meta.shards_metadata:
            shard_md.placement = _remote_device(f"rank:{other_rank}/{device_info}")

        # Add other inner shards from the inner tensor
        for inner_md in inner_st.metadata().shards_metadata:
            if inner_md.shard_offsets != inner_shard.metadata.shard_offsets:
                st_meta.shards_metadata.append(
                    ShardMetadata(
                        shard_offsets=_element_wise_add(
                            outer_shard.metadata.shard_offsets,
                            inner_md.shard_offsets,
                        ),
                        shard_sizes=inner_md.shard_sizes,
                        placement=f"rank:{other_rank}/{device_info}",
                    )
                )

        # Finally add this shard
        st_meta.shards_metadata.append(local_shards[0].metadata)

        st = ShardedTensor._init_from_local_shards_and_global_metadata(
            local_shards=local_shards,
            sharded_tensor_metadata=st_meta,
        )
        set_element(new_state_dict, path, st)

    traverse_state_dict(state_dict, rewrite_dict)
    return new_state_dict
<|endoftext|>import io
import json
import struct
from dataclasses import dataclass
from typing import Any

import torch


_metadata_fn: str = "model.safetensors.index.json"

FILE_NAME = "model-{cpt_idx}-of-{num_files}"
SHARDED_FILE_NAME = "shard-{shard_idx}-model-{cpt_idx}-of-{num_files}"
SUFFIX = ".safetensors"

# metadata keys
CUSTOM_METADATA_KEY = "DCP_SHARDING_INFO"
DEFAULT_EXTRA_METADATA_KEY = "__metadata__"
SAVED_OFFSETS_KEY = "saved_offsets"
SHAPE_KEY = "shape"
DATA_KEY = "data"
DTYPE_KEY = "dtype"
DATA_OFFSETS_KEY = "data_offsets"

DTYPE_MAP = {
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I8": torch.int8,
    "U8": torch.uint8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "BF16": torch.bfloat16,
}

HF_DCP_VERSION: float = 1.0
DCP_VERSION_KEY = "DCP_VERSION"
DCP_SHARDING_INFO_KEY = "DCP_SHARDING_INFO"

FORMAT_KEY = "format"
FORMAT_VALUE = "pt"

NUM_BYTES_FOR_HEADER_LEN = 8

SHARDED_DIR_NAME = "sharded"


@dataclass
class _HFStorageInfo:
    """This is the per entry storage info."""

    relative_path: str
    shape: torch.Size
    dtype: torch.dtype


def _gen_file_name(
    index: int, largest_index: int, shard_index: int | None = None
) -> str:
    if shard_index is not None:
        return (
            SHARDED_FILE_NAME.format(
                shard_idx=f"{shard_index}".zfill(5),
                cpt_idx=f"{index}".zfill(5),
                num_files=f"{largest_index}".zfill(5),
            )
            + SUFFIX
        )
    else:
        return (
            FILE_NAME.format(
                cpt_idx=f"{index}".zfill(5), num_files=f"{largest_index}".zfill(5)
            )
            + SUFFIX
        )


def _get_safetensors_file_metadata(file_bytes: io.IOBase) -> tuple[Any, int]:
    # this uses the same logic that's done in HF code base
    # https://github.com/2404589803/huggingface_hub/blob/main/src/huggingface_hub/hf_api.py#L5308
    # and follows their documentation on how their files are serialized
    # https://huggingface.co/docs/safetensors/index#format

    header_len_bytes = file_bytes.read(NUM_BYTES_FOR_HEADER_LEN)
    header_len = struct.unpack("<Q", header_len_bytes)[0]
    header_json = file_bytes.read(header_len)
    metadata = json.loads(header_json)
    return (metadata, header_len + NUM_BYTES_FOR_HEADER_LEN)


def _get_dtype(dtype_str: str) -> torch.dtype:
    try:
        dtype = DTYPE_MAP[dtype_str]
    except KeyError:
        dtype = torch.get_default_dtype()

    return dtype


def _get_dcp_custom_metadata(metadata: Any) -> Any | None:
    if DEFAULT_EXTRA_METADATA_KEY in metadata:
        custom_metadata = metadata[DEFAULT_EXTRA_METADATA_KEY]
        if CUSTOM_METADATA_KEY in custom_metadata:
            return json.loads(custom_metadata[CUSTOM_METADATA_KEY])
    return None
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

import dataclasses
import io
import logging
import math
import sys
from bisect import bisect_right, insort
from collections import ChainMap
from typing import Any, cast

import torch
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint._dedup_save_plans import dedup_save_plans
from torch.distributed.checkpoint._nested_dict import (
    FLATTEN_MAPPING,
    flatten_state_dict,
)
from torch.distributed.checkpoint._sharded_tensor_utils import _flatten_sharded_tensors
from torch.distributed.checkpoint._traverse import set_element
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    STATE_DICT_TYPE,
    STORAGE_TYPES,
    StorageMeta,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import (
    LoadPlan,
    LoadPlanner,
    ReadItem,
    SavePlan,
    SavePlanner,
    WriteItem,
    WriteItemType,
)
from torch.distributed.checkpoint.planner_helpers import (
    _compare_save_plans,
    _contains_usable_plan,
    _create_default_metadata_only_plan,
    _create_read_items,
    _create_write_items,
    _init_state_dict,
    _merge_delta_local_plans,
)
from torch.distributed.checkpoint.protocol import _is_checkpointable_tensor
from torch.distributed.checkpoint.utils import find_state_dict_object
from torch.distributed.tensor import DTensor

from . import _version


logger: logging.Logger = logging.getLogger(__name__)


__all__ = [
    "DefaultSavePlanner",
    "DefaultLoadPlanner",
    "create_default_local_load_plan",
    "create_default_global_load_plan",
    "create_default_local_save_plan",
    "create_default_global_save_plan",
]


# TODO: Update docstrings for default_planner.py
class DefaultSavePlanner(SavePlanner):
    mappings: FLATTEN_MAPPING

    def __init__(
        self,
        flatten_state_dict: bool = True,
        flatten_sharded_tensors: bool = True,
        dedup_replicated_tensors: bool | None = None,
        dedup_save_to_lowest_rank: bool = False,
        enable_plan_caching: bool = False,
    ) -> None:
        self.flatten_state_dict = flatten_state_dict
        self.flatten_sharded_tensors = flatten_sharded_tensors
        self.mappings = {}
        self.dedup_save_to_lowest_rank = dedup_save_to_lowest_rank
        if dedup_replicated_tensors is not None:
            logger.warning(
                "DefaultSavePlanner's `dedup_replicated_tensors` argument is being "
                "deprecated, and no longer has any effect. Please remove this argument "
                "from your call."
            )
        self._cached_plans_key: str = self.__class__.__name__
        self._enable_plan_caching = enable_plan_caching

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        storage_meta: StorageMeta | None = None,
        is_coordinator: bool = False,
    ) -> None:
        if self.flatten_state_dict:
            state_dict, self.mappings = flatten_state_dict(state_dict)
        if self.flatten_sharded_tensors:
            state_dict = _flatten_sharded_tensors(state_dict)
        self.state_dict = state_dict
        self.is_coordinator = is_coordinator

    def create_local_plan(self) -> SavePlan:
        plan = create_default_local_save_plan(self.state_dict, self.is_coordinator)
        if self.flatten_state_dict:
            plan = dataclasses.replace(plan, planner_data=self.mappings)
        self.plan = plan

        if self._enable_plan_caching:
            # If plans are equal, we can skip sending the plan to the coordinator.
            if (
                self._cached_plans_key in SavePlanner._cached_save_plan
                and _compare_save_plans(
                    plan, SavePlanner._cached_save_plan[self._cached_plans_key]
                )
            ):
                logger.info(
                    "No change in the local plan. Skipping sending the plan to the coordinator"
                )
                return SavePlan([], usable=False)
            else:
                # Store the plan as pending. It will be promoted to the
                # class-level cache in finish_plan after the global plan
                # has succeeded. This avoids a stale local cache when
                # the global plan fails (e.g. validation error) but the
                # local cache was already populated.
                self._pending_local_plan = plan

        return self.plan

    def _dedup_save_plans(self, all_plans: list[SavePlan]) -> list[SavePlan]:
        return dedup_save_plans(all_plans, self.dedup_save_to_lowest_rank)

    def _create_global_plan(
        self, all_plans: list[SavePlan]
    ) -> tuple[list[SavePlan], Metadata]:
        deduped_plans = self._dedup_save_plans(all_plans)

        global_plan, metadata = create_default_global_save_plan(deduped_plans)

        if self.flatten_state_dict:
            # | does not work for Python 3.8 or older version.
            # merged_mappings = reduce(
            #     lambda x, y: x | y, (p.planner_data for p in global_plan)
            # )
            planner_data_dict = [p.planner_data for p in global_plan]
            merged_mappings = dict(ChainMap(*planner_data_dict))
            metadata = dataclasses.replace(metadata, planner_data=merged_mappings)

        validation_errors = _validate_global_plan(global_plan, metadata)
        if validation_errors:
            error_summary = "; ".join(validation_errors)
            if len(error_summary) > 500:
                error_summary = error_summary[:500] + "... (truncated)"
            raise ValueError(f"Failed to validate global plan: {error_summary}")

        return global_plan, metadata

    def _create_global_plan_with_caching(
        self, all_plans: list[SavePlan]
    ) -> tuple[list[SavePlan], list[SavePlan], Metadata]:
        """
        Create global plan with caching.
        Returns a tuple of global_plan_delta, global_plan, metadata.
        """
        global_plan_delta: list[SavePlan] = []

        if self._cached_plans_key not in SavePlanner._cached_all_plans:
            # Case 1: If the plans are not cached, the cache will be hydrated with the
            # all_plans, global_plans (Deduped), and metadata.

            # First create and validate the global plan. Only cache everything
            # after success to avoid partial cache state
            global_plan, metadata = self._create_global_plan(all_plans)

            # Cache all plans atomically after successful validation
            SavePlanner._cached_all_plans[self._cached_plans_key] = all_plans
            SavePlanner._cached_global_plan[self._cached_plans_key] = global_plan
            SavePlanner._cached_metadata[self._cached_plans_key] = metadata
            # If plans are not cached, global_plan delta will be the same as global plan.
            return global_plan, global_plan, metadata

        # Case 2: Plans are cached
        if not _contains_usable_plan(all_plans):
            # Case 2.1: Plans are cached and the local plans have NOT changed (No usable plans).
            # Global plan delta will be empty plans to avoid the collective overhead.
            # We can reuse the deduped global plan and metadata from the cache directly.
            global_plan_delta = [SavePlan([], usable=False)] * len(all_plans)
            global_plan = SavePlanner._cached_global_plan[self._cached_plans_key]
            metadata = SavePlanner._cached_metadata[self._cached_plans_key]
        else:
            # Case 2.2: Plans are cached but the local plans have changed.
            # We will merge the changed local plans with the cached local plans.
            # Updated plans will overwrite the cached plans. New global plan and metadata will be created and cached.
            # Global plan delta will be created by comparing the new global plan with the cached global plan.
            # Only the global plan delta (updated ones) will be sent to the coordinator to avoid the collective overhead.
            merged_plans = _merge_delta_local_plans(
                SavePlanner._cached_all_plans[self._cached_plans_key], all_plans
            )
            # Cache the updated local plans
            SavePlanner._cached_all_plans[self._cached_plans_key] = merged_plans
            global_plan, metadata = self._create_global_plan(merged_plans)

            if self._cached_plans_key in self._cached_global_plan:
                for cached_plan, new_plan in zip(
                    SavePlanner._cached_global_plan[self._cached_plans_key], global_plan
                ):
                    if _compare_save_plans(cached_plan, new_plan):
                        global_plan_delta.append(SavePlan([], usable=False))
                    else:
                        global_plan_delta.append(new_plan)

            # Cache the new global plan and the metadata
            SavePlanner._cached_global_plan[self._cached_plans_key] = global_plan
            SavePlanner._cached_metadata[self._cached_plans_key] = metadata

        return global_plan_delta, global_plan, metadata

    def create_global_plan(
        self, all_plans: list[SavePlan]
    ) -> tuple[list[SavePlan], Metadata]:
        global_plan_delta: list[SavePlan] = []
        if self._enable_plan_caching:
            # If the plans are cached, we only need to send the global plan delta to be scattered
            # across ranks. Ranks will use the cached final plans instead.
            (
                global_plan_delta,
                global_plan,
                metadata,
            ) = self._create_global_plan_with_caching(all_plans)
        else:
            global_plan, metadata = self._create_global_plan(all_plans)
            # If the caching is not enabled, global delta plan will always be same as the new global plan.
            global_plan_delta = global_plan

        self.global_plan = global_plan
        self.metadata = metadata

        return global_plan_delta, self.metadata

    def _finish_plan_with_caching(self, new_plan: SavePlan) -> SavePlan:
        finished_plan: SavePlan = new_plan

        if not new_plan.usable:
            finished_plan = SavePlanner._cached_final_save_plan[self._cached_plans_key]
        else:
            finished_plan = new_plan
            SavePlanner._cached_final_save_plan[self._cached_plans_key] = new_plan
        return finished_plan

    def finish_plan(self, new_plan: SavePlan) -> SavePlan:
        finished_plan: SavePlan = new_plan

        if self._enable_plan_caching:
            finished_plan = self._finish_plan_with_caching(new_plan)

            # Promote the pending local plan to the class-level cache now
            # that the global plan has succeeded and we are finalizing.
            # This ensures the local cache is only populated after a
            # successful end-to-end checkpoint plan creation.
            if hasattr(self, "_pending_local_plan"):
                SavePlanner._cached_save_plan[self._cached_plans_key] = (
                    self._pending_local_plan
                )
                del self._pending_local_plan

        self.plan = finished_plan
        return self.plan

    def resolve_data(self, write_item: WriteItem) -> torch.Tensor | io.BytesIO:
        object = self.lookup_object(write_item.index)
        return self.transform_object(write_item, object)

    def lookup_object(self, index: MetadataIndex) -> Any:
        """Extension from the planner interface to make it easy to extend the default planner."""
        return find_state_dict_object(self.state_dict, index)

    def transform_object(self, write_item: WriteItem, object: Any):
        """Extension from the planner interface to make it easy to extend the default planner."""
        if write_item.type == WriteItemType.BYTE_IO:
            bytes = io.BytesIO()
            torch.save(object, bytes)
            object = bytes
        return object


class DefaultLoadPlanner(LoadPlanner):
    """
    DefaultLoadPlanner that adds multiple features on top of LoadPlanner.

    In particular it adds the following:

    flatten_state_dict: Handle state_dict with nested dicts
    flatten_sharded_tensors: For FSDP in 2D parallel mode
    allow_partial_load: If False, will raise a runtime error if a key is present in state_dict, but not in the checkpoint.
    """

    original_state_dict: STATE_DICT_TYPE
    mappings: FLATTEN_MAPPING

    def __init__(
        self,
        flatten_state_dict: bool = True,
        flatten_sharded_tensors: bool = True,
        allow_partial_load: bool = False,
    ) -> None:
        self.flatten_state_dict = flatten_state_dict
        self.flatten_sharded_tensors = flatten_sharded_tensors
        self.original_state_dict = {}
        self.mappings = {}
        self.allow_partial_load = allow_partial_load

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        _init_state_dict(state_dict)
        self.original_state_dict = state_dict

        if self.flatten_sharded_tensors:
            state_dict = _flatten_sharded_tensors(state_dict)

        if self.flatten_state_dict:
            state_dict, self.mappings = flatten_state_dict(state_dict)

        self.state_dict = state_dict
        self.metadata = metadata
        self.is_coordinator = is_coordinator

    def create_local_plan(self) -> LoadPlan:
        if self.metadata is None:
            raise AssertionError("self.metadata is not None")
        if self.flatten_state_dict:
            # To support checkpoints that are saved before v2.4, we have to
            # differentiate if the missing keys are due to old checkpoints.
            # The contracts are:
            # 1. There are 3 cases when we found a missing key.
            #    1.1 Actual missing key, but allow_partial_load is False
            #    1.2 Actual missing key, but allow_partial load is True
            #    1.3 Old checkpoint, but allow_partial_load is False
            #    1.4 Old checkpoint, but allow_partial_load is True
            # 2. If we found a missing key, we first convert the keys back to
            #    the key format of v2.3
            # 3. If the previous missing keys are in the v2.3 keys, we assume
            #    this is an old checkpoint.
            # 4. Pass the state_dict to `create_default_local_load_plan()`,
            #    which has the logic to check missing for allow_partial_load.
            # So for 1.2 and 1.4 cases, we delegate allow_partial_load check to
            # `create_default_local_load_plan()`. The logic here is to determine
            # whether the checkpoint belong to 2.3 (or before) or 2.4 (or after).
            current_keys = set(self.state_dict.keys())
            load_keys = set(self.metadata.state_dict_metadata.keys())
            missing_keys = load_keys - current_keys
            if missing_keys:
                _version._derived_version = "2_3"
                old_state_dict, old_mappings = flatten_state_dict(
                    self.original_state_dict
                )
                old_keys = set(old_state_dict.keys())
                if old_keys & missing_keys:
                    self.state_dict, self.mappings = old_state_dict, old_mappings
                # _derived_version is only used by flatten_state_dict now.
                # Set it back to None so that later we can save to a new version.
                _version._derived_version = None

        return create_default_local_load_plan(
            self.state_dict, self.metadata, not self.allow_partial_load
        )

    def create_global_plan(self, global_plan: list[LoadPlan]) -> list[LoadPlan]:
        return create_default_global_load_plan(global_plan)

    def finish_plan(self, new_plan: LoadPlan) -> LoadPlan:
        return new_plan

    def load_bytes(self, read_item: ReadItem, value: io.BytesIO) -> None:
        if self.flatten_state_dict:
            set_element(
                self.original_state_dict,
                self.mappings[read_item.dest_index.fqn],
                torch.load(value, weights_only=False),
            )
        else:
            self.state_dict[read_item.dest_index.fqn] = torch.load(
                value, weights_only=False
            )

    def resolve_tensor(self, read_item: ReadItem):
        tensor = self.lookup_tensor(read_item.dest_index)
        return self.transform_tensor(read_item, tensor)

    def commit_tensor(self, read_item: ReadItem, tensor: torch.Tensor) -> None:
        pass

    def lookup_tensor(self, index: MetadataIndex) -> torch.Tensor:
        """Extension from the planner interface to make it easy to extend the default planner."""
        return find_state_dict_object(self.state_dict, index)

    def transform_tensor(self, read_item: ReadItem, tensor: torch.Tensor):
        """Extension from the planner interface to make it easy to extend the default planner."""
        return narrow_tensor_by_index(tensor, read_item.dest_offsets, read_item.lengths)


class _EmptyStateDictLoadPlanner(DefaultLoadPlanner):
    """
    Extension of DefaultLoadPlanner, which rebuilds state_dict from the saved metadata.
    Useful for loading in state_dict without first initializing a model, such as
    when converting a DCP checkpoint into a Torch save file.

    . N.B. `state_dict` must be an empty dictionary when used with this LoadPlanner

    .. warning::
        Because the entire state dict is initialized, It's recommended to only utilize
        this LoadPlanner on a single rank or process to avoid OOM.

    """

    def __init__(self, keys=None, *args, **kwargs):
        self.keys = keys
        super().__init__(*args, **kwargs)

    def _should_include_key(self, key: str, metadata: Metadata) -> bool:
        if self.keys is None:
            return True

        if key in self.keys:
            return True

        unflattened_keys: list[str] = []
        planner_data = metadata.planner_data.get(key)
        for unflattened_key in planner_data:
            if unflattened_keys:
                unflattened_keys.append(
                    ".".join([unflattened_keys[-1], str(unflattened_key)])
                )

            else:
                unflattened_keys.append(unflattened_key)

        if any(unflattened_key in self.keys for unflattened_key in unflattened_keys):
            return True

        return False

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        if state_dict:
            raise AssertionError("not state_dict")
        if metadata is None:
            raise AssertionError("metadata is not None")

        # rebuild the state dict from the metadata
        for k, v in metadata.state_dict_metadata.items():
            if not self._should_include_key(k, metadata):
                continue

            if isinstance(v, TensorStorageMetadata):
                v = torch.empty(v.size, dtype=v.properties.dtype)  # type: ignore[assignment]
            if metadata.planner_data is not None and k in metadata.planner_data:
                set_element(state_dict, metadata.planner_data[k], v)
            else:
                state_dict[k] = v

        super().set_up_planner(state_dict, metadata, is_coordinator)


def create_default_local_load_plan(
    state_dict: dict[str, Any], metadata: Metadata, strict: bool = True
) -> LoadPlan:
    requests = []
    """
    Create the ``LoadPlan`` used by DefaultLoadPlanner.

    It produces one read item per value in ``state_dict`` using the metadata in ``metadata``.

    The default behavior is to match key exactly between state_dict and metadata.
    It handles resharding by issuing multiple read requests against storage in order to match
    load requirements.
    """

    for fqn, obj in state_dict.items():
        # ignore state_dict keys which do not exist in `state_dict` if strict=False
        if fqn not in metadata.state_dict_metadata:
            if strict:
                raise RuntimeError(f"Missing key in checkpoint state_dict: {fqn}.")
            else:
                continue

        md = metadata.state_dict_metadata[fqn]
        if (
            isinstance(md, TensorStorageMetadata)
            and getattr(obj, "size", None) is not None
        ):
            obj_size = (
                torch.Size(obj.global_shape)
                if _is_checkpointable_tensor(obj)
                else obj.size()
            )
            if md.size != obj_size:
                raise ValueError(
                    f"Size mismatch between saved {md.size} and current: {obj_size} for {fqn}",
                )
        # Since DTensor supports submesh, adding extra check to ensure _create_read_items()
        # gets called only when the current rank is part of the mesh for the corresponding DTensor.
        if isinstance(obj, DTensor):
            if obj.device_mesh.get_coordinate() is not None:
                requests += _create_read_items(fqn, md, obj)
        else:
            requests += _create_read_items(fqn, md, obj)

    return LoadPlan(requests)


def create_default_global_load_plan(
    all_plans: list[LoadPlan],
) -> list[LoadPlan]:
    """
    Create global load plan used by DefaultLoadPlanner.

    The default load behavior involved no global coordination and this function
    currently doesn't change the local plans.
    """
    return all_plans


def create_default_local_save_plan(
    state_dict: dict[str, Any], is_coordinator: bool
) -> SavePlan:
    """
    Create the ``SavePlan`` used by DefaultSavePlanner.

    On non-coordinator ranks, this function ignores tensors and non-tensor objects,
    only producing writes for ShardedTensor objects.

    On the coordinator rank, produce writes for all values.
    """
    requests = []
    for fqn, obj in state_dict.items():
        # Since DTensor supports submesh, adding extra check to ensure _create_write_items()
        # gets called only when the current rank is part of the mesh for the corresponding DTensor.
        if isinstance(obj, DTensor):
            if obj.device_mesh.get_coordinate() is not None:
                requests += _create_write_items(fqn, obj)
        else:
            # For the plain tensor and non-tensor values, add the request for all
            # the ranks. Coordinator will decide whether to deduplicate the
            # values based on the keys.
            requests += _create_write_items(fqn, obj)

    return SavePlan(requests)


def create_default_global_save_plan(
    all_plans: list[SavePlan],
    rewrite_index_hints: bool = True,
) -> tuple[list[SavePlan], Metadata]:
    """
    Create the global plan and metadata used by DefaultSavePlanner.

    Metadata is produced by concatenating the metadata of all ``WriteItem`` from the supplied plans.

    The only global planning change is to update index hints in all ``MetadataIndex`` objects if
    ``rewrite_index_hints`` is True.
    """
    md: dict[str, STORAGE_TYPES] = {}
    new_plans = []
    for plan in all_plans:
        new_items = []
        for item in plan.items:
            if item.type != WriteItemType.SHARD:
                if item.index.fqn in md:
                    raise AssertionError("item.index.fqn not in md")

            if item.type == WriteItemType.BYTE_IO:
                md[item.index.fqn] = BytesStorageMetadata()
                new_items.append(item)
            else:
                if item.tensor_data is None:
                    raise AssertionError("item.tensor_data is not None")
                tensor_md = cast(
                    TensorStorageMetadata,
                    md.setdefault(
                        item.index.fqn,
                        TensorStorageMetadata(
                            properties=item.tensor_data.properties,
                            size=item.tensor_data.size,
                            chunks=[],
                        ),
                    ),
                )
                new_item = item
                if rewrite_index_hints:
                    new_index = dataclasses.replace(
                        item.index, index=len(tensor_md.chunks)
                    )
                    new_item = dataclasses.replace(item, index=new_index)
                new_items.append(new_item)

                if item.tensor_data.chunk is None:
                    raise AssertionError(f"""
                    Cannot create MD for tensor without bounds.
                    FQN: {item.index.fqn}
                """)
                tensor_md.chunks.append(item.tensor_data.chunk)
        new_plans.append(dataclasses.replace(plan, items=new_items))
    return (new_plans, Metadata(md))


def _create_default_local_metadata(state_dict: STATE_DICT_TYPE) -> Metadata:
    """Return the ``Metadata`` if DefaultSavePlanner was used to checkpoint ``state_dict``."""
    plan = _create_default_metadata_only_plan(state_dict)
    _, md = create_default_global_save_plan([plan])
    return md


def _check_box_overlap(box0: ChunkStorageMetadata, box1: ChunkStorageMetadata) -> bool:
    """Check if two boxes overlap. Tuples are (offset, lengths)."""
    # For each dim of each shard, check if one shard resides on the other
    # end of second shard with respect to that dim. As an example for a 2D
    # shard, we would check if one shard is above or on the left of the
    # other shard.
    ndims = len(box0.offsets)
    for i in range(ndims):
        if box0.offsets[i] >= box1.offsets[i] + box1.sizes[i]:
            return False
        if box1.offsets[i] >= box0.offsets[i] + box0.sizes[i]:
            return False

    return True


def _check_box_bounds(
    outer_box_size: torch.Size, inner_box: ChunkStorageMetadata
) -> bool:
    for i in range(len(outer_box_size)):
        if inner_box.offsets[i] < 0:
            return False
        if inner_box.sizes[i] < 0:
            return False
        if inner_box.offsets[i] + inner_box.sizes[i] > outer_box_size[i]:
            return False

    return True


def _validate_global_plan(global_plan: list[SavePlan], metadata: Metadata) -> list[str]:
    """Validate the global plan and return a list of error messages (empty if valid)."""
    errors: list[str] = []
    for key, value in metadata.state_dict_metadata.items():
        if isinstance(value, BytesStorageMetadata):
            continue
        if len(value.size) == 0:
            continue
        chunks = value.chunks
        chunks_volume = 0
        for chunk in chunks:
            # Compute the volume
            if not _check_box_bounds(value.size, chunk):
                msg = (
                    f"key:{key} has out of bounds chunk: "
                    f"tensor-size:{value.size} chunk: {chunk}"
                )
                logger.warning(msg)
                errors.append(msg)
            chunks_volume += math.prod(chunk.sizes)

        if len(chunks) > 1:
            dims = len(value.size)
            sweep_dim = max(range(dims), default=0, key=lambda d: value.size[d])
            sorted_indices = sorted(
                range(len(chunks)),
                key=lambda idx: (
                    chunks[idx].offsets[sweep_dim],
                    *(chunks[idx].offsets[d] for d in range(dims)),
                ),
            )
            active: list[tuple[int, int]] = []
            for idx in sorted_indices:
                current = chunks[idx]
                start = current.offsets[sweep_dim]
                end = start + current.sizes[sweep_dim]

                cutoff = bisect_right(active, (start, sys.maxsize))
                if cutoff:
                    del active[:cutoff]

                for _, other_idx in active:
                    other = chunks[other_idx]
                    if _check_box_overlap(current, other):
                        msg = f"key:{key} has overlapping chunks: {current} {other}"
                        logger.warning(msg)
                        errors.append(msg)

                insort(active, (end, idx))

        # Check whether combined chunk cover the whole tensor
        tensor_volume = math.prod(value.size)
        if len(global_plan) > 1 and chunks_volume != tensor_volume:
            msg = (
                f"key:{key} invalid fill tensor-volume: "
                f"{tensor_volume} chunks-volume: {chunks_volume}"
            )
            logger.warning(msg)
            errors.append(msg)

    return errors
<|endoftext|># mypy: allow-untyped-defs
import cProfile
import inspect
import io
import itertools
import os
import warnings
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from functools import wraps
from pstats import Stats
from typing import Any, cast, TypeVar

import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharded_tensor.shard import Shard

from .api import (
    _is_wrapped_exception,
    _wrap_exception,
    CheckpointException,
    WRAPPED_EXCEPTION,
)
from .metadata import MetadataIndex, STATE_DICT_TYPE
from .protocol import _get_checkpointable_tensor_shard, _is_checkpointable_tensor


__all__ = ["find_tensor_shard", "find_state_dict_object"]

T = TypeVar("T")
R = TypeVar("R")


def _get_failure_dict(
    results: list[T | WRAPPED_EXCEPTION],
) -> dict[int, WRAPPED_EXCEPTION]:
    return cast(
        dict[int, WRAPPED_EXCEPTION],
        {i: err for i, err in enumerate(results) if _is_wrapped_exception(err)},
    )


def _all_gather_keys(
    local_dict: dict[str, Any], group: dist.ProcessGroup | None = None
) -> set[str]:
    """Gathers all keys, and returns them sorted."""
    keys = list(local_dict.keys())
    gathered_keys: list[list[str]] = [None] * dist.get_world_size(group)  # type: ignore[list-item]

    dist.all_gather_object(gathered_keys, keys, group=group)
    return set(itertools.chain.from_iterable(gathered_keys))


def _assert_same_keys(
    state_dict: dict[str, Any], process_group: dist.ProcessGroup | None = None
) -> None:
    """
    Asserts that all ranks have the same keys in their state dict.
    This is a collective call which requires all ranks in ``process_group`` to
    join. It will also induce cross-rank communication and block CPU.
    """

    if dist.get_world_size(process_group) == 1:
        return

    all_keys = _all_gather_keys(state_dict, process_group)
    my_keys = set(state_dict.keys())
    diff = all_keys - my_keys
    if len(diff) > 0:
        raise AssertionError(
            f"Key(s) present in other ranks but not this one, difference: {diff}"
        )


class _DistWrapper:
    """
    This is a wrapper around PG that provides a series of features around object collectives.

    It works without distributed initialized, where most collectives turn into nops.

    All variants that take functions are exception robust, meaning that if one or more
    ranks raise errors, all ranks will observe those.
    """

    def __init__(
        self,
        group: dist.ProcessGroup | None,
        use_dist: bool,
        coordinator_rank: int,
    ):
        self.group = group
        self.use_dist = use_dist
        self.coordinator_rank = coordinator_rank
        if self.use_dist:
            self.global_coordinator_rank = (
                dist.get_global_rank(group, coordinator_rank)
                if group is not None
                else coordinator_rank
            )
            self.rank = dist.get_rank(group)
            self.is_coordinator = self.rank == coordinator_rank
        else:
            self.global_coordinator_rank = 0
            self.rank = 0
            self.is_coordinator = True

    def get_rank(self) -> int:
        return self.rank

    def get_world_size(self) -> int:
        if self.use_dist:
            return dist.get_world_size(self.group)
        return 1

    def broadcast_object(self, object: T | None) -> T:
        """Implement functionality similar to c10d::broadcast_object_list but without distributed enabled."""
        object_list = [object]
        if self.use_dist:
            dist.broadcast_object_list(
                object_list=object_list,
                group=self.group,
                src=self.global_coordinator_rank,
            )
        return cast(T, object_list[0])

    def gather_object(self, object: T) -> list[T] | None:
        """Implement functionality similar to c10d::gather_object but without distributed enabled."""
        if self.use_dist:
            gather_objs = (
                cast(list[T], [None] * dist.get_world_size(self.group))
                if self.is_coordinator
                else None
            )

            dist.gather_object(
                obj=object,
                object_gather_list=gather_objs if self.is_coordinator else None,
                dst=self.global_coordinator_rank,
                group=self.group,
            )
            result = gather_objs
        else:
            result = [object]
        return result

    def all_gather_object(self, object: T) -> list[T]:
        """Implement functionality similar to c10d::all_gather_object but without distributed enabled."""
        if self.use_dist:
            gather_objs = cast(list[T], [None] * dist.get_world_size(self.group))

            dist.all_gather_object(
                object_list=gather_objs, obj=object, group=self.group
            )
        else:
            gather_objs = [object]
        return gather_objs

    def scatter_object(self, object_list: list[T] | None) -> T:
        """Implement functionality similar to c10d::scatter_object but without distributed enabled."""
        if self.use_dist:
            gather_result = cast(list[T], [None])
            dist.scatter_object_list(
                scatter_object_output_list=gather_result,
                scatter_object_input_list=object_list if self.is_coordinator else None,
                src=self.global_coordinator_rank,
                group=self.group,
            )

            local_reply = gather_result[0]
        else:
            if object_list is None:
                raise AssertionError("object_list is None")
            local_reply = object_list[0]
        return local_reply

    def reduce_scatter(
        self,
        step: str,
        map_fun: Callable[[], T],
        reduce_fun: Callable[[list[T]], list[R]],
    ) -> R:
        """
        Compute a value on each rank, then do centralized reduce on a single rank, followed by a scatter.

        This method operates in the following way:
            Run ``map_fun`` on all ranks
            Gather results on rank 0
            Call ``reduce_fun`` on all those values
            Scatter to each rank part of the result.
        """
        local_data: WRAPPED_EXCEPTION | T
        try:
            local_data = map_fun()
        except BaseException as e:
            local_data = _wrap_exception(e)

        all_data = self.gather_object(local_data)
        all_results: list[R | CheckpointException] | None = None
        if self.is_coordinator:
            if all_data is None:
                raise AssertionError("all_data is None")
            node_failures = _get_failure_dict(all_data)

            if len(node_failures) == 0:
                try:
                    # N.B. why can't mypy cast List[R] to List[Union[R, WRAPPED_EXCEPTION]]?
                    all_results = cast(
                        list[R | CheckpointException],
                        reduce_fun(cast(list[T], all_data)),
                    )
                except BaseException as e:
                    node_failures[self.rank] = _wrap_exception(e)

            if len(node_failures) > 0:
                all_results = [
                    CheckpointException(step, node_failures)
                ] * self.get_world_size()

        result = self.scatter_object(all_results)
        if isinstance(result, CheckpointException):
            raise result
        return result

    def all_reduce(
        self,
        step: str,
        map_fun: Callable[[], T],
        reduce_fun: Callable[[list[T]], R],
    ) -> R:
        """
        Compute a value on each rank, then do centralized reduce on a single rank, followed by a broadcast.

        This method operates in the following way:
            Run ``map_fun`` on all ranks
            Gather results on rank 0
            Call ``reduce_fun`` on all those values
            Broadcast the reduced value to all ranks.
        """
        local_data: T | WRAPPED_EXCEPTION
        try:
            local_data = map_fun()
        except BaseException as e:
            local_data = _wrap_exception(e)

        all_data = self.gather_object(local_data)
        result: R | CheckpointException | None = None
        if self.is_coordinator:
            if all_data is None:
                raise AssertionError("all_data is None")
            node_failures = _get_failure_dict(all_data)
            if len(node_failures) == 0:
                try:
                    result = reduce_fun(cast(list[T], all_data))
                except BaseException as e:
                    node_failures[self.rank] = _wrap_exception(e)

            if len(node_failures) > 0:
                result = CheckpointException(step, node_failures)

        # pyrefly: ignore [bad-argument-type]
        final_result = self.broadcast_object(result)
        if isinstance(final_result, CheckpointException):
            raise final_result
        # pyrefly: ignore [redundant-cast]
        return cast(R, final_result)

    def all_gather(
        self,
        step: str,
        map_fun: Callable[[], T],
    ) -> list[T]:
        """
        Compute a value on each rank, then all_gather them.

        This method operates in the following way:
            Run ``map_cp`` on all ranks
            all_gather the values to all ranks
        """
        result: T | WRAPPED_EXCEPTION
        try:
            result = map_fun()
        except BaseException as e:
            result = _wrap_exception(e)

        all_results = self.all_gather_object(result)

        node_failures = _get_failure_dict(all_results)
        if len(node_failures) > 0:
            raise CheckpointException(step, node_failures)
        return cast(list[T], all_results)

    def broadcast(
        self,
        step: str,
        map_fun: Callable[[], T],
    ) -> T:
        """
        Compute a value on rank 0 and broadcast it.

        This method operates in the following way:
            Run ``map_cp`` on rank 0
            broadcast the value
        """
        result: T | CheckpointException | None = None
        if self.is_coordinator:
            try:
                result = map_fun()
            except BaseException as e:
                result = CheckpointException(step, {self.rank: _wrap_exception(e)})
        # pyrefly: ignore [bad-argument-type]
        final_result = self.broadcast_object(result)
        if isinstance(final_result, CheckpointException):
            raise final_result
        # pyrefly: ignore [redundant-cast]
        return cast(T, final_result)

    def barrier(self) -> None:
        """
        Add a synchronization point across all processes when using distributed.
        If torch.distributed is initialized, this function will invoke a barrier across the global process group.
        If torch.distributed is not initialized, this function is a no-op.
        """
        if not self.use_dist:
            return
        dist.barrier(group=self.group)


def _find_shard(tensor: ShardedTensor, index: MetadataIndex) -> Shard:
    if index.offset is None:
        raise ValueError(
            f"Cannot lookup {index.fqn} since its a ShardedTensor and no offset was provided"
        )

    shards = tensor.local_shards()
    # index fast path
    if index.index is not None:
        if (
            len(shards) > index.index
            and torch.Size(shards[index.index].metadata.shard_offsets) == index.offset
        ):
            return shards[index.index]

    for shard in shards:
        if torch.Size(shard.metadata.shard_offsets) == index.offset:
            return shard
    raise ValueError(f"Could not find shard at '{index.offset}' for FQN: '{index.fqn}'")


def find_tensor_shard(tensor: torch.Tensor, index: MetadataIndex) -> torch.Tensor:
    if hasattr(tensor, "__get_tensor_shard__"):
        # DTensor implements _Checkpointable
        return tensor.__get_tensor_shard__(index)  # type: ignore[attr-defined]
    if _is_checkpointable_tensor(tensor):
        return _get_checkpointable_tensor_shard(tensor, index)
    if isinstance(tensor, ShardedTensor):
        return _find_shard(tensor, index).tensor
    if index.offset is not None:
        # special case looking up a tensor by origin
        if index.offset == torch.Size([0] * len(tensor.size())):
            return tensor
        raise ValueError(
            f"FQN: '{index.fqn}' is not a ShardedTensor, can't find by offset: '{index.offset}'"
        )
    return tensor


def find_state_dict_object(state_dict: STATE_DICT_TYPE, index: MetadataIndex) -> Any:
    if index.fqn not in state_dict:
        raise ValueError(f"Could not find FQN: '{index.fqn}'")
    obj = state_dict[index.fqn]

    if isinstance(obj, torch.Tensor):
        return find_tensor_shard(obj, index)
    elif index.offset is not None:
        raise ValueError(
            f"FQN: '{index.fqn}' is not a ShardedTensor, can't find by offset: '{index.offset}'"
        )
    return obj


def _element_wise_add(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return [i_a + i_b for i_a, i_b in zip(a, b)]


def _element_wise_sub(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return [i_a - i_b for i_a, i_b in zip(a, b)]


class _ReaderView(io.IOBase):
    def __init__(self, base_stream: io.IOBase, offset: int, len: int):
        super().__init__()
        self.offset = offset
        self.len = len
        self.base_stream = base_stream
        self.seek(0)

    def seek(self, offset: int, whence: int = os.SEEK_SET, /) -> int:
        if whence == os.SEEK_SET:
            offset = self.offset + offset
        elif whence == os.SEEK_END:
            whence = os.SEEK_SET
            offset = (self.offset + self.len) - offset
        return self.base_stream.seek(offset, whence)

    def tell(self) -> int:
        return self.base_stream.tell() - self.offset

    def readable(self) -> bool:
        return self.base_stream.readable()

    def seekable(self) -> bool:
        return self.base_stream.seekable()

    def readinto(self, b):
        max_size = self.len - self.tell()
        if max_size == 0:
            return 0
        if len(b) > max_size:
            b = memoryview(b)[:max_size]
        return self.base_stream.readinto(b)  # type: ignore[attr-defined]

    def read(self, size=-1):
        max_size = self.len - self.tell()
        if size == -1 or size > max_size:
            size = max_size
        return self.base_stream.read(size)


def _create_file_view(file: io.IOBase, offset: int, length: int) -> io.IOBase:
    # FIXME (kumpera) torch.load fails if we wrap with io.BufferedReader
    return _ReaderView(file, offset, length)


def _normalize_device_info(device_type: str, device_id: int) -> str:
    """Device info normalization."""
    if device_type == "cpu":
        return "cpu"
    return f"{device_type}:{device_id}"


# TODO: integrate with distributed logging flag
ENABLE_PROFILE = False


@contextmanager
def _profile():
    # Only log the profiling when it is enable and is on rank0  or dist is not
    # available.
    if ENABLE_PROFILE and (not dist.is_available() or dist.get_rank() == 0):
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            yield
        finally:
            profiler.disable()
            stats = Stats(profiler)
            stats.sort_stats("time").print_stats(10)
    else:
        yield


def _api_bc_check(func):
    @wraps(func)
    def inner_func(*args, **kwargs) -> Any:
        if len(args) == 2:
            warnings.warn(
                f"The argument order of {func.__name__} has been changed. "
                "Please check the document to avoid future breakages.",
                stacklevel=2,
            )
            sig = inspect.signature(func)
            kwonlyargs = [
                p.name for p in sig.parameters.values() if p.kind == p.KEYWORD_ONLY
            ]
            if "storage_writer" in kwonlyargs:
                if "storage_writer" in kwargs:
                    raise AssertionError(f"storage_writer in kwargs: {(args, kwargs)}")
                kwargs["storage_writer"] = args[1]
            elif "storage_reader" in kwonlyargs:
                if "storage_reader" in kwargs:
                    raise AssertionError(f"storage_reader in kwargs: {(args, kwargs)}")
                kwargs["storage_reader"] = args[1]
            else:
                raise RuntimeError(f"Unexpected kwonlyargs = {kwonlyargs}")
            return func(args[0], **kwargs)
        else:
            return func(*args, **kwargs)

    return inner_func
<|endoftext|># mypy: allow-untyped-defs
import types
import warnings
import weakref
from copyreg import dispatch_table
from typing import Any

import torch
import torch.cuda._pin_memory_utils as pin_memory_utils
from torch.storage import UntypedStorage
from torch.utils.weak import WeakIdKeyDictionary


class StateDictStager:
    """
    A class for optimizing storage objects during staging for async checkpointing.

    StateDictStager stages the state_dict to CPU DRAM while applying optimizations
    like memory sharing and pinning to improve performance. It caches storage objects
    to avoid redundant copies and can be configured to automatically share memory
    (for multi-process usage) and pin memory (for faster CPU-GPU transfers).

    Attributes:
        pin_memory (bool): Whether to pin CPU memory for faster CPU-GPU transfers
        share_memory (bool): Whether to share memory across processes
        pin_memory_min_bytes (int): Minimum tensor size in bytes to pin memory (default: 5)
        _cached_storage_mapping (WeakIdKeyDictionary): Maps storage objects to optimized CPU storages using weak references
    """

    def __init__(
        self,
        pin_memory: bool = False,
        share_memory: bool = False,
        pin_memory_min_bytes: int = 5,
    ):
        if pin_memory and not torch.cuda.is_available():
            warnings.warn(
                "Ignoring pin_memory flag for checkpoint staging as pinning memory"
                "requires CUDA, but CUDA is not available. ",
                stacklevel=2,
            )
            self.pin_memory = False
        else:
            self.pin_memory = pin_memory
        self.share_memory = share_memory
        # Mapping from original storage objects to CPU storages using weak references
        self._cached_storage_mapping = WeakIdKeyDictionary()
        self.pin_memory_min_bytes = pin_memory_min_bytes

        def _deepcopy_atomic(x, _):
            return x

        def _deepcopy_list(x, memo, non_blocking=False):
            y: list = []
            memo[id(x)] = y
            append = y.append
            for a in x:
                append(
                    self.deepcopy_with_tensor_offload(
                        a, memo, non_blocking=non_blocking
                    )
                )
            return y

        def _deepcopy_tuple(x, memo, non_blocking=False):
            y = [
                self.deepcopy_with_tensor_offload(a, memo, non_blocking=non_blocking)
                for a in x
            ]
            # We're not going to put the tuple in the memo, but it's still important we
            # check for it, in case the tuple contains recursive mutable structures.
            try:
                return memo[id(x)]
            except KeyError:
                pass

            # Check if any elements changed during deepcopy
            for k, j in zip(x, y):
                if k is not j:
                    # At least one element changed, create new tuple
                    return tuple(y)

            # No elements changed, return original tuple
            return x

        def _deepcopy_dict(x, memo, non_blocking=False):
            y: dict = {}
            memo[id(x)] = y
            for key, value in x.items():
                y[
                    self.deepcopy_with_tensor_offload(
                        key, memo, non_blocking=non_blocking
                    )
                ] = self.deepcopy_with_tensor_offload(
                    value, memo, non_blocking=non_blocking
                )
            return y

        def _deepcopy_method(x, memo, non_blocking=False):  # Copy instance methods
            return type(x)(
                x.__func__,
                self.deepcopy_with_tensor_offload(
                    x.__self__, memo, non_blocking=non_blocking
                ),
            )

        d: dict[Any, Any] = {}
        self._deepcopy_dispatch = d
        d[type(None)] = _deepcopy_atomic
        d[int] = _deepcopy_atomic
        d[float] = _deepcopy_atomic
        d[bool] = _deepcopy_atomic
        d[complex] = _deepcopy_atomic
        d[bytes] = _deepcopy_atomic
        d[str] = _deepcopy_atomic
        d[types.CodeType] = _deepcopy_atomic
        d[type] = _deepcopy_atomic
        d[range] = _deepcopy_atomic
        d[types.BuiltinFunctionType] = _deepcopy_atomic
        d[types.FunctionType] = _deepcopy_atomic
        d[weakref.ref] = _deepcopy_atomic
        d[property] = _deepcopy_atomic
        d[types.MethodType] = _deepcopy_method
        d[dict] = _deepcopy_dict
        d[tuple] = _deepcopy_tuple
        d[list] = _deepcopy_list

    def _stage_untyped_storage(
        self,
        storage: UntypedStorage,
        non_blocking: bool = False,
    ):
        """
        Called from the hooked storage_deepcopy function in torch.Tensor.__deepcopy__.

        This method handles the storage optimization logic for the StagingStateDict class.
        It checks if the storage has already been cached, and if so, reuses it.
        Otherwise, it creates a new CPU storage and applies memory optimizations.

        Args:
            storage: The storage to optimize

        Returns:
            The optimized storage
        """
        # Check if we've already cached this storage
        if storage in self._cached_storage_mapping:
            cached_storage = self._cached_storage_mapping[storage]
            if cached_storage.size() != storage.size():
                raise AssertionError(
                    "For async checkpointing,  We cache storages in DRAM and reuse them."
                    "Cached storage size does not match original storage size."
                    "This should never happen as we track the original storage weakref "
                    "and clean up the cache storage. Please report this to PyTorch Distributed Checkpointing."
                )
            # Reuse cached storage but update with new data
            cached_storage.copy_(storage, non_blocking=non_blocking)
            return cached_storage

        # Create new CPU storage
        if self.share_memory:
            new_storage = type(storage)._new_shared(storage.size(), device="cpu")
        else:
            new_storage = type(storage)(storage.size(), device="cpu")

        # Skip pinning for tensors below the minimum size threshold
        # Small tensors (e.g., optimizer step counters, scalars) have negligible
        # transfer time improvement from pinning, but pinning overhead is significant
        if self.pin_memory and new_storage.nbytes() >= self.pin_memory_min_bytes:
            pin_memory_utils.pin_memory(new_storage.data_ptr(), new_storage.nbytes())
            # Set up a weak reference to unpin when cpu storage is garbage collected
            f = weakref.finalize(
                new_storage, pin_memory_utils.unpin_memory, new_storage.data_ptr()
            )
            # This makes sure that the finalizer is not called after
            # cuda context is destroyed.
            f.atexit = False

        new_storage.copy_(storage, non_blocking=non_blocking)

        # Cache the storage - WeakIdKeyDictionary will automatically clean up when storage is garbage collected
        self._cached_storage_mapping[storage] = new_storage
        return new_storage

    @torch.no_grad()
    def stage(
        self,
        state_dict: Any,
        non_blocking: bool = False,
    ) -> Any:
        return self.deepcopy_with_tensor_offload(state_dict, None, [], non_blocking)

    def _offload_tensor(self, x, memo, non_blocking=False):
        """
        Deep copy a PyTorch tensor with optimized storage handling.

        This method creates a CPU copy of a tensor while applying memory optimizations
        like sharing and pinning based on the StateDictStager configuration.

        Args:
            x: The tensor to copy
            memo: Memo dictionary for tracking already copied objects
            non_blocking: Whether to perform non-blocking copies where possible

        Returns:
            A CPU copy of the tensor with optimized storage
        """
        # if data_ptr is not 0, we allocate a new storage below. so we can skip
        # memory allocation by using [] for size.
        y = x.new_empty([] if x.data_ptr() != 0 else x.size(), device="cpu")

        # Store in memo dict early to handle recursive references
        d = id(x)
        memo[d] = y

        if type(x) is torch.Tensor or x.data_ptr() != 0:
            # Get the untyped storage
            untyped_storage = x.untyped_storage()
            storage_id = id(untyped_storage)

            # Check if this storage has already been staged in this deepcopy operation
            # This handles the case where different tensors share the same storage
            # (e.g., FSDP state_dict where norm.weight and norm_weight reference same storage)
            # PyTorch caches untyped_storage() calls, so same storage -> same id
            if storage_id in memo:
                copied_storage = memo[storage_id]
            else:
                # Storage not seen before in this operation, stage it
                copied_storage = self._stage_untyped_storage(
                    untyped_storage, non_blocking=non_blocking
                )
                # Add to memo to avoid re-staging if we see this storage again
                memo[storage_id] = copied_storage

            # Set the tensor data using the staged storage
            y.set_(copied_storage, x.storage_offset(), x.size(), x.stride())

        # Copy any attributes the tensor might have
        if hasattr(x, "__dict__"):
            for attr_name, attr_value in x.__dict__.items():
                setattr(
                    y,
                    attr_name,
                    self.deepcopy_with_tensor_offload(
                        attr_value, memo, non_blocking=non_blocking
                    ),
                )

        if hasattr(x, "__slots__"):
            for slot in x.__slots__:
                if hasattr(x, slot):
                    setattr(
                        y,
                        slot,
                        self.deepcopy_with_tensor_offload(
                            getattr(x, slot), memo, non_blocking=non_blocking
                        ),
                    )

        return y

    def close(self):
        """
        Clean up all cached storages and release associated resources.

        This method clears the internal storage cache, allowing garbage collection
        of cached CPU storages. Any pinned memory associated with cached storages
        will be automatically unpinned through weak reference finalizers.

        It also clears the _deepcopy_dispatch dict to break the reference cycle
        created by closures that capture self. Without this, it may
        cause memory leaks.
        """
        self._cached_storage_mapping.clear()
        self._deepcopy_dispatch.clear()

    @torch.no_grad()
    def deepcopy_with_tensor_offload(self, x, memo=None, _nil=[], non_blocking=False):  # noqa: B006
        """Deep copy operation on arbitrary Python objects with special handling for PyTorch tensors.

        This implementation extends the standard deepcopy functionality to handle PyTorch tensors
        and their storages in a way that optimizes memory usage and performance, similar to the
        stage method. It applies memory sharing and pinning optimizations based on the StateDictStager
        configuration.

        Args:
            x: The object to deep copy
            memo: Memo dictionary for tracking already copied objects
            _nil: Sentinel value for memo dictionary
            non_blocking: Whether to perform non-blocking copies where possible

        Returns:
            A deep copy of the input object with optimized tensor storage handling
        """
        if memo is None:
            memo = {}

        d = id(x)
        y = memo.get(d, _nil)
        if y is not _nil:
            return y

        cls = type(x)

        # tensors and subclasses of tensors are handled separately
        if isinstance(x, torch.Tensor):
            y = self._offload_tensor(x, memo, non_blocking=non_blocking)
        else:
            # Use the dispatch table for standard types
            copier = self._deepcopy_dispatch.get(cls)
            if copier is not None:
                # Check if this is an atomic copier (only accepts x and memo)
                if copier.__name__ == "_deepcopy_atomic":
                    y = copier(x, memo)
                else:
                    y = copier(x, memo, non_blocking=non_blocking)
            else:
                if issubclass(cls, type):
                    # type copier is also atomic
                    y = self._deepcopy_dispatch[type](x, memo)
                else:
                    copier = getattr(x, "__deepcopy__", None)
                    if copier is not None:
                        y = copier(memo)
                    else:
                        reductor = dispatch_table.get(cls)
                        if reductor:
                            rv = reductor(x)
                        else:
                            reductor = getattr(x, "__reduce_ex__", None)
                            if reductor is not None:
                                rv = reductor(4)
                            else:
                                reductor = getattr(x, "__reduce__", None)
                                if reductor:
                                    rv = reductor()
                                else:
                                    raise RuntimeError(
                                        f"un(deep)copyable object of type {cls}"
                                    )
                        if isinstance(rv, str):
                            y = x
                        else:
                            # Unpack rv tuple elements (up to 5 from pickle protocol)
                            # and explicitly pass non_blocking as keyword arg
                            if len(rv) == 2:
                                func, args = rv
                                y = self._reconstruct(
                                    x, memo, func, args, non_blocking=non_blocking
                                )
                            elif len(rv) == 3:
                                func, args, state = rv
                                y = self._reconstruct(
                                    x,
                                    memo,
                                    func,
                                    args,
                                    state,
                                    non_blocking=non_blocking,
                                )
                            elif len(rv) == 4:
                                func, args, state, listiter = rv
                                y = self._reconstruct(
                                    x,
                                    memo,
                                    func,
                                    args,
                                    state,
                                    listiter,
                                    non_blocking=non_blocking,
                                )
                            elif len(rv) == 5:
                                func, args, state, listiter, dictiter = rv
                                y = self._reconstruct(
                                    x,
                                    memo,
                                    func,
                                    args,
                                    state,
                                    listiter,
                                    dictiter,
                                    non_blocking=non_blocking,
                                )
                            else:
                                raise RuntimeError(
                                    f"Unexpected pickle protocol return value length: {len(rv)}"
                                )

        # If it is its own copy, don't memoize.
        if y is not x:
            memo[d] = y
            self._keep_alive(x, memo)  # Make sure x lives at least as long as d
        return y

    def _keep_alive(self, x, memo):
        """Keeps a reference to the object x in the memo.

        Because we remember objects by their id, we have
        to assure that possibly temporary objects are kept
        alive by referencing them.
        We store a reference at the id of the memo, which should
        normally not be used unless someone tries to deepcopy
        the memo itself...
        """
        try:
            memo[id(memo)].append(x)
        except KeyError:
            # aha, this is the first one :-)
            memo[id(memo)] = [x]

    def _reconstruct(
        self,
        x,
        memo,
        func,
        args,
        state=None,
        listiter=None,
        dictiter=None,
        non_blocking=False,
    ):
        deep = memo is not None
        if deep and args:
            args = tuple(
                self.deepcopy_with_tensor_offload(arg, memo, non_blocking=non_blocking)
                for arg in args
            )
        y = func(*args)
        if deep:
            memo[id(x)] = y

        if state is not None:
            if deep:
                state = self.deepcopy_with_tensor_offload(
                    state, memo, non_blocking=non_blocking
                )
            if hasattr(y, "__setstate__"):
                y.__setstate__(state)
            else:
                if isinstance(state, tuple) and len(state) == 2:
                    state, slotstate = state
                else:
                    slotstate = None
                if state is not None:
                    y.__dict__.update(state)
                if slotstate is not None:
                    for key, value in slotstate.items():
                        setattr(y, key, value)

        if listiter is not None:
            if deep:
                for item in listiter:
                    item = self.deepcopy_with_tensor_offload(
                        item, memo, non_blocking=non_blocking
                    )
                    y.append(item)
            else:
                for item in listiter:
                    y.append(item)
        if dictiter is not None:
            if deep:
                for key, value in dictiter:
                    key = self.deepcopy_with_tensor_offload(
                        key, memo, non_blocking=non_blocking
                    )
                    value = self.deepcopy_with_tensor_offload(
                        value, memo, non_blocking=non_blocking
                    )
                    y[key] = value
            else:
                for key, value in dictiter:
                    y[key] = value
        return y
<|endoftext|># mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import inspect
import logging
import os
import warnings
from typing import Any, cast, TYPE_CHECKING
from typing_extensions import deprecated

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.logger import _dcp_method_logger
from torch.distributed.checkpoint.stateful import Stateful

from ._storage_utils import _storage_setup
from .default_planner import DefaultLoadPlanner
from .planner import LoadPlan, LoadPlanner
from .storage import StorageReader
from .utils import _api_bc_check, _DistWrapper, _profile


if TYPE_CHECKING:
    from torch.distributed.checkpoint.metadata import Metadata

__all__ = ["load_state_dict", "load"]

logger = logging.getLogger()


@deprecated(
    "`load_state_dict` is deprecated and will be removed in future versions. "
    "Please use `load` instead.",
    category=FutureWarning,
)
def load_state_dict(
    state_dict: dict[str, Any],
    storage_reader: StorageReader,
    process_group: dist.ProcessGroup | None = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: LoadPlanner | None = None,
) -> None:
    """This method is deprecated. Please switch to 'load'."""
    storage_reader.reset()
    with _profile():
        # TODO: test returning `load` here instead.
        return _load_state_dict(
            state_dict,
            storage_reader,
            process_group,
            coordinator_rank,
            no_dist,
            planner,
        )


@_dcp_method_logger(log_exceptions=True)
@_api_bc_check
def load(
    state_dict: dict[str, Any],
    *,
    checkpoint_id: str | os.PathLike | None = None,
    storage_reader: StorageReader | None = None,
    planner: LoadPlanner | None = None,
    process_group: dist.ProcessGroup | None = None,
    no_dist: bool = False,
) -> None:
    """
    Load a checkpoint into a distributed state dict in SPMD style.

    Each rank must have the same keys in their ``state_dict`` provided to this
    API. Mismatched keys may result in hangs or errors. If unsure, you can use
    the ``utils._assert_same_keys`` API to check (but may incur communication
    costs).

    Each rank will try to read the least amount of data necessary
    to fulfill the requested `state_dict`. When loading :class:`ShardedTensor`
    or :class:`DTensor` instances, each rank only reads data for their local shards.

    For each ``Stateful`` object (having both a ``state_dict`` and a ``load_state_dict``),
    load will first call ``state_dict`` before attempting deserialization, followed by
    ``load_state_dict`` once the deserialization is complete.
    For each non-``Stateful`` object, load will deserialize the object, and then replace
    it in the ``state_dict`` with the deserialized object.

    .. warning::
        All tensors in ``state_dict`` must be allocated on their
        destination device *prior to* calling this function.

        All non-tensor data is loaded using `torch.load()` and modified in place
        on state_dict.

    .. warning::
        Users must call `load_state_dict` on the root module to ensure load
        post-processing and non-tensor data properly propagates.

    .. note:
        If no process group is initialized, this function will assume the intent
        is to load a checkpoint into the local process. This can be useful in the
        case of local inference, and when using regular Tensors (as opposed to DTensor
         or ShardedTensor)

    .. note:
        Rank 0 is assumed to be the coordinator rank.

    Args:
        state_dict (Dict[str, Any]): The state_dict to load the checkpoint into.
        checkpoint_id (Union[str, os.PathLike, None]):
            The ID of this checkpoint instance. The meaning of the checkpoint_id
            depends on the storage. It can be a path to a folder or to a file.
            It can also be a key if the storage is a key-value store.
            (Default: ``None``)
        storage_reader (Optional[StorageReader]):
            Instance of StorageWriter used to perform reads. If this is not
            specified, DCP will automatically infer the reader based on the
            checkpoint_id. If checkpoint_id is also None, an exception will
            be raised. (Default: ``None``)
        planner (Optional[LoadPlanner]):
            Instance of LoadPlanner. If this is not specified, the default
            planner will be used. (Default: ``None``)
        process_group (Optional[ProcessGroup]):
            ProcessGroup to be used for cross-rank synchronization.
            (Default: ``None``)
        no_dist (bool): If ``True``, this function will assume the intent is to load
            a checkpoint without using cross-rank synchronization. (Default: ``False``)
    Returns:
        None.

    Examples
        >>> # xdoctest: +SKIP
        >>> my_model = MyModule()
        >>> optimizer = Adagrad(my_model.parameters())
        >>> model_state_dict = my_model.state_dict()
        >>> fs_storage_reader = torch.distributed.checkpoint.FileSystemReader(
        ...     "/checkpoint/1"
        ... )

        >>> torch.distributed.checkpoint.load_state_dict(
        >>>     state_dict=model_state_dict,
        >>>     storage_reader=fs_storage_reader,
        >>> )

        >>> # module.load_state_dict() function might have customized steps
        >>> # to flush the state_dict, must call it to
        >>> # ensure correct behavior.
        >>> my_model.load_state_dict(model_state_dict)

    .. note::
        load_state_dict uses collectives to coordinate reads across ranks.
        For NCCL-based process groups, internal tensor representations of
        objects must be moved to the GPU device before communication takes place.
        In this case, the device used is given by ``torch.cuda.current_device()``
        and it is the user's responsibility to ensure that this is set so that each
        rank has an individual GPU, via ``torch.cuda.set_device()``.
    """

    no_dist = no_dist or (not dist.is_available()) or (not dist.is_initialized())
    if no_dist:
        warnings.warn(
            "torch.distributed is disabled, unavailable or uninitialized, assuming the intent is to load in a single process.",
            stacklevel=2,
        )

    with _profile():
        storage_reader = cast(
            StorageReader, _storage_setup(storage_reader, checkpoint_id, reader=True)
        )

        # All ranks must have the same keys in their `state_dict` provided to
        # this API.  See documentation for more details.
        # Here we simply sort the keys to ensure that all ranks load values in
        # the same order.
        keys = sorted(state_dict.keys())

        stateful_sd = {}
        for key in keys:
            if key not in state_dict:
                continue
            elem = state_dict[key]
            stateful_sd[key] = elem.state_dict() if isinstance(elem, Stateful) else elem

        _load_state_dict(
            state_dict=stateful_sd,
            storage_reader=storage_reader,
            process_group=process_group,
            no_dist=no_dist,
            planner=planner,
        )
        for key in keys:
            if key not in state_dict:
                continue
            elem = state_dict[key]
            if isinstance(elem, Stateful):
                # If the state_dict is a Stateful object,
                # DCP does an in-place load in the original state dict.
                elem.load_state_dict(stateful_sd[key])
            else:
                # Otherwise, replace the state_dict with the loaded state_dict.
                state_dict[key] = stateful_sd[key]


def _load_state_dict(
    state_dict: dict[str, Any],
    storage_reader: StorageReader,
    process_group: dist.ProcessGroup | None = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: LoadPlanner | None = None,
) -> None:
    torch._C._log_api_usage_once("torch.distributed.checkpoint.load_state_dict")

    distW = _DistWrapper(process_group, not no_dist, coordinator_rank)
    if planner is None:
        planner = DefaultLoadPlanner()

    ckpt_kwargs = {}
    if (ckpt_id := getattr(storage_reader, "checkpoint_id", None)) is not None:
        ckpt_kwargs["checkpoint_id"] = ckpt_id
        ckpt_kwargs["process_group"] = distW.group

    use_collectives = True
    metadata: Metadata | None = None

    @_dcp_method_logger(**ckpt_kwargs)
    def local_step():
        nonlocal use_collectives
        nonlocal metadata

        # Use global metadata if available, otherwise fallback to rank local metadata
        global_metadata_exc: Exception | None = None
        rank_metadata_exc: Exception | None = None
        try:
            metadata = storage_reader.read_metadata()
        except Exception as e:
            global_metadata_exc = e
            logger.warning(
                "Global metadata is not found. Falling back to rank local metadata.",
                exc_info=True,
            )

        if (
            not metadata
            and "kwargs" in inspect.signature(storage_reader.read_metadata).parameters
        ):
            try:
                metadata = storage_reader.read_metadata(rank=distW.rank)
                use_collectives = False
            except Exception as e:
                rank_metadata_exc = e
                logger.warning("Rank local metadata is not found.", exc_info=True)

        if planner is None:
            raise AssertionError("planner is None")
        if metadata is None:
            error_parts = ["metadata is None"]
            if global_metadata_exc is not None:
                error_parts.append(
                    f"global metadata read failed: {global_metadata_exc}"
                )
            if rank_metadata_exc is not None:
                error_parts.append(
                    f"rank local metadata read failed: {rank_metadata_exc}"
                )
            raise AssertionError("; ".join(error_parts))
        planner.set_up_planner(state_dict, metadata, distW.is_coordinator)

        if (
            "kwargs"
            in inspect.signature(storage_reader.set_up_storage_reader).parameters
        ):
            storage_reader.set_up_storage_reader(
                metadata,
                distW.is_coordinator,
                rank=distW.rank,
                use_collectives=use_collectives,
            )
        else:
            storage_reader.set_up_storage_reader(metadata, distW.is_coordinator)

        local_plan = planner.create_local_plan()
        local_plan = storage_reader.prepare_local_plan(local_plan)
        return local_plan

    @_dcp_method_logger(**ckpt_kwargs)
    def global_step(all_local_plans):
        if planner is None:
            raise AssertionError("planner is None")
        all_local_plans = planner.create_global_plan(all_local_plans)
        all_local_plans = storage_reader.prepare_global_plan(all_local_plans)
        return all_local_plans

    central_plan: LoadPlan | None = None
    if use_collectives:
        central_plan = distW.reduce_scatter("plan", local_step, global_step)
    else:
        local_plan: LoadPlan = local_step()
        global_plan: list[LoadPlan] = global_step([local_plan])
        central_plan = global_plan[0]

    @_dcp_method_logger(**ckpt_kwargs)
    def read_data():
        if planner is None:
            raise AssertionError("planner is None")
        if central_plan is None:
            raise AssertionError("central_plan is None")
        final_local_plan = planner.finish_plan(central_plan)
        all_reads = storage_reader.read_data(final_local_plan, planner)

        all_reads.wait()
        return None

    if use_collectives:
        _ = distW.all_gather("read", read_data)
    else:
        read_data()
        distW.barrier()


def _load_state_dict_from_keys(
    keys: set[str] | str | None = None,
    *,
    checkpoint_id: str | os.PathLike | None = None,
    storage_reader: StorageReader | None = None,
    process_group: dist.ProcessGroup | None = None,
) -> dict[str, Any]:
    """
    Load only the specified keys from the checkpoint, if no keys are specified, the entire
    checkpoint will be loaded. Note, this method completely loads the checkpoint into the
    current process and is not distributed.

    .. warning::


    .. warning::

        All non-tensor data is loaded using `torch.load()`

    .. note:
        As opposed to the usual pattern, this function does not take a state dict as input
        and does not load inplace. Instead, a new state dict is directly initialized and read
        from file.

    .. note:
        If no process group is initialized, this function will assume the intent
        is to load a checkpoint into the local process. This can be useful in the
        case of local inference, and when using regular Tensors (as opposed to DTensor
         or ShardedTensor)

    .. note:
        Rank 0 is assumed to be the coordinator rank.

    Args:
        keys (Optional[Union[set[str], str]]):
            Loads any key specified in this set. If no keys are specified, the entire checkpoint
            is loaded.
        checkpoint_id (Union[str, os.PathLike, None]):
            The ID of this checkpoint instance. The meaning of the checkpoint_id
            depends on the storage. It can be a path to a folder or to a file.
            It can also be a key if the storage is a key-value store.
            (Default: ``None``)
        storage_reader (Optional[StorageReader]):
            Instance of StorageWriter used to perform reads. If this is not
            specified, DCP will automatically infer the reader based on the
            checkpoint_id. If checkpoint_id is also None, an exception will
            be raised. (Default: ``None``)
        process_group (Optional[ProcessGroup]):
            ProcessGroup to be used for cross-rank synchronization.
            (Default: ``None``)

    Returns:
        State dict from specified keys
    """
    torch._C._log_api_usage_once(
        "torch.distributed.checkpoint._load_state_dict_from_keys"
    )

    no_dist = not (dist.is_available() and dist.is_initialized())
    if no_dist:
        warnings.warn(
            "torch.distributed is unavailable or uninitialized, assuming the intent is to load in a single process.",
            stacklevel=2,
        )

    storage_reader = cast(
        StorageReader, _storage_setup(storage_reader, checkpoint_id, reader=True)
    )

    if isinstance(keys, str):
        keys = {keys}

    sd: dict[str, Any] = {}
    _load_state_dict(
        state_dict=sd,
        storage_reader=storage_reader,
        process_group=process_group,
        no_dist=no_dist,
        planner=_EmptyStateDictLoadPlanner(keys=keys),
    )

    return sd
<|endoftext|>from . import _extension
from .api import CheckpointException
from .default_planner import DefaultLoadPlanner, DefaultSavePlanner
from .filesystem import FileSystemReader, FileSystemWriter
from .hf_storage import HuggingFaceStorageReader, HuggingFaceStorageWriter
from .metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    TensorStorageMetadata,
)
from .optimizer import load_sharded_optimizer_state_dict
from .planner import LoadPlan, LoadPlanner, ReadItem, SavePlan, SavePlanner, WriteItem
from .protocol import CheckpointableTensor
from .quantized_hf_storage import QuantizedHuggingFaceStorageReader

# pyrefly: ignore [deprecated]
from .state_dict_loader import load, load_state_dict

# pyrefly: ignore [deprecated]
from .state_dict_saver import async_save, save, save_state_dict
from .storage import StorageReader, StorageWriter
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates

import abc
import io
from collections.abc import Sequence
from typing import cast, IO

# introduced as collections.abc.Buffer in Python 3.12
from typing_extensions import Buffer

from torch._utils import try_import


# NOTE: everything in this file is experimental, and subject to
# change.  Feedback and bug fixes are always welcome.

pyzstd_module_name = "pyzstd"
pyzstd = try_import(pyzstd_module_name)
zstandard_module_name = "zstandard"
zstandard = try_import(zstandard_module_name)


__all__ = [
    "Extension",
    "StreamTransformExtension",
    "ZStandard",
    "ExtensionRegistry",
]


class Extension(abc.ABC):
    """
    Extensions provide modular additions to functionality within distributed checkpointing,
    which affect the layout or format of the written artifacts.  Extensions may be
    built into pytorch, or provided externally.

    When writing, the caller provides a list of extension instances of the appropriate
    type.  Each extension can output a descriptor which is used to reconstitute the
    extension at read-time.
    """

    @staticmethod
    @abc.abstractmethod
    def registry_name() -> str:
        """
        See ExtensionRegistry.from_descriptor_list
        """

    @staticmethod
    @abc.abstractmethod
    def from_descriptor(version: str) -> "Extension":
        """
        See ExtensionRegistry.from_descriptor_list
        """

    @abc.abstractmethod
    def get_descriptor(self) -> str:
        """
        Return descriptor name to be included in metadata.  The form should be
        "extension_name[@local-domain][/version]".
        """


class StreamTransformExtension(Extension):
    """
    An extension which performs transformation on a byte stream, such as compression
    or encryption.

    Implementations should try to be memory friendly and performant.  For example, don't
    read the whole input, then transform it, and write it back.  If at all possible, do it in
    chunks.  But, don't read/transform/write one byte at a time, either.
    """

    @abc.abstractmethod
    def transform_to(self, output: IO[bytes]) -> IO[bytes]:
        """
        Takes a writeable output stream, and generates a new stream which implements the
        output transform.  Input data written to the returned stream will be transformed
        and written to the `output` argument stream.
        """

    @abc.abstractmethod
    def transform_from(self, input: IO[bytes]) -> IO[bytes]:
        """
        Takes a readable input stream, and generates a new stream which implements the
        input transform.  When the returned stream is read, data will be read from the
        'input' stream, transformed, and returned.
        """


class ZStandard(StreamTransformExtension):
    @staticmethod
    def is_available() -> bool:
        return zstandard is not None or pyzstd is not None

    @staticmethod
    def from_descriptor(version: str) -> "ZStandard":
        if version.partition(".")[0] != "1":
            raise ValueError(f"Unknown extension {version=}")
        if not ZStandard.is_available():
            raise ValueError(
                f"Stream with ZStandard compression cannot be processed because "
                f"no module named '{zstandard_module_name}' or '{pyzstd_module_name}'"
            )
        return ZStandard()

    @staticmethod
    def registry_name() -> str:
        return "stream.zstd"

    def __init__(self) -> None:
        super().__init__()
        if not ZStandard.is_available():
            raise ValueError(
                f"ZStandard extension is unavailable because no module named '{zstandard_module_name}' or '{pyzstd_module_name}'"
            )

    def get_descriptor(self) -> str:
        return f"{self.registry_name()}/1"

    def transform_to(self, output: IO[bytes]) -> IO[bytes]:
        if zstandard is not None:
            compressor = zstandard.ZstdCompressor()  # type: ignore[union-attr]
            return compressor.stream_writer(output)

        class Writer(io.RawIOBase):
            def __init__(self, output: IO[bytes]) -> None:
                self.output = output
                self.compressor = pyzstd.ZstdCompressor()  # type: ignore[union-attr]

            def writeable(self) -> bool:
                return True

            def write(self, b: Buffer) -> int | None:
                outdata = self.compressor.compress(b)
                if outdata:
                    self.output.write(outdata)
                return len(memoryview(b))

            def flush(self) -> None:
                outdata = self.compressor.flush()
                if outdata:
                    self.output.write(outdata)
                self.output.flush()

        return cast(IO[bytes], Writer(output))

    def transform_from(self, input: IO[bytes]) -> IO[bytes]:
        if zstandard is not None:
            decompressor = zstandard.ZstdDecompressor()  # type: ignore[union-attr]
            return decompressor.stream_reader(input)

        class Reader(io.RawIOBase):
            def __init__(self, input: IO[bytes]) -> None:
                self.input = input
                self.decompressor = pyzstd.EndlessZstdDecompressor()  # type: ignore[union-attr]

            def readable(self) -> bool:
                return True

            def readinto(self, b: Buffer) -> int | None:
                # This needs to read enough so it can decompress
                # something so the output doesn't look like EOF.  This
                # means reading at least one block.  The max block
                # size is 128KB, so we read that plus some
                # overhead to be sure.

                if self.decompressor.needs_input:
                    indata = self.input.read((128 + 6) * 1024)
                else:
                    indata = b""

                bview = memoryview(b)
                blen = len(bview)
                outdata = self.decompressor.decompress(indata, blen)
                if outdata is None:
                    return None

                count = len(outdata)
                bview[:count] = outdata
                return count

            def seekable(self) -> bool:
                return False

        return cast(IO[bytes], Reader(input))


class ExtensionRegistry:
    def __init__(self) -> None:
        # Populate default registry contents
        self.extensions: dict[str, type[Extension]] = {
            cls.registry_name(): cls for cls in (ZStandard,)
        }

    def register(self, cls: type[Extension]) -> None:
        self.extensions[cls.registry_name()] = cls

    def from_descriptor_list(self, descriptors: Sequence[str]) -> Sequence[Extension]:
        """
        Given a sequence of descriptor strings as returned by
        Extension.get_descriptor at save time, creates a sequence of
        Extension instances.  The name[@local-domain] preceding the
        version number is used to look up an implementation class in
        the registry, and the version is passed to the class's
        from_descriptor static method.  If the registry contains no
        match, this will throw ValueError.  If the from_descriptor
        method raises an exception, that will pass through to the
        caller.
        """

        def from_descriptor(desc: str) -> Extension:
            name, _, version = desc.partition("/")
            if version is None:
                version = 0
            ext = self.extensions.get(name)
            if not ext:
                raise ValueError(f"Unknown extension {name=}")
            # pyrefly: ignore [bad-argument-type]
            return ext.from_descriptor(version)

        return [from_descriptor(desc) for desc in descriptors]
<|endoftext|>import logging
import pickle
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import cast, TypeVar

import torch
from torch.distributed import ProcessGroup, Work
from torch.distributed._shard.sharded_tensor import (
    Shard as ShardedTensorShard,
    ShardedTensor,
    ShardMetadata,
)
from torch.distributed._shard.sharded_tensor.metadata import ShardedTensorMetadata
from torch.distributed.tensor import _DTensorSpec, DTensor
from torch.utils._pytree import (
    KeyPath,
    tree_flatten_with_path,
    tree_unflatten,
    TreeSpec,
)


logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class _TensorMeta:
    """
    This is the metadata for a tensor that is used to transfer checkpoints.
    It contains the shape, the dtype, the storage offset and the stride of the
    tensor.

    This must be pickleable so that it can be sent over the wire.
    """

    shape: torch.Size
    dtype: torch.dtype
    storage_offset: int
    stride: tuple[int, ...]
    nbytes: int


@dataclass
class _DTensorMeta:
    """
    This is the metadata for a DTensor that is used to transfer checkpoints.
    It contains the metadata for the local tensor and the spec of the DTensor.

    This must be pickleable so that it can be sent over the wire.
    """

    local: _TensorMeta
    spec: _DTensorSpec


@dataclass
class _ShardedTensorMeta:
    """
    This is the metadata for a ShardedTensor that is used to transfer checkpoints.
    It contains the metadata for all local shards and the global tensor metadata.

    This must be pickleable so that it can be sent over the wire.
    """

    local_shards_meta: list[_TensorMeta]
    local_shards_shard_metadata: list[
        ShardMetadata
    ]  # Original shard metadata for each local shard
    sharded_tensor_metadata: ShardedTensorMetadata


@dataclass
class _StateDictMeta:
    """
    This is the metadata for a state dict that is used to transfer checkpoints.
    It contains the step, the pytree spec of the state dict and the metadata for
    each tensor in the state dict.

    This must be pickleable so that it can be sent over the wire.

    Args:
        step: the step of the checkpoint to verify consistency
        treespec: the pytree spec of the state dict
        paths: the path of each leaf in the state dict
        non_tensor_leaves: the metadata for each tensor in the state dict and any
            non-tensor leaves in the state dict
    """

    treespec: TreeSpec
    paths: list[KeyPath]
    non_tensor_leaves: list[object | _TensorMeta | _DTensorMeta | _ShardedTensorMeta]


@contextmanager
def _timeit(name: str) -> Generator[None, None, None]:
    start = time.perf_counter()
    yield
    dur = time.perf_counter() - start
    logger.info("%s took %ss", name, dur)


def _prepare_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, _TensorMeta]:
    return (
        _cast_tensor(tensor, torch.uint8),
        _TensorMeta(
            shape=tensor.shape,
            dtype=tensor.dtype,
            storage_offset=cast(int, tensor.storage_offset()),
            stride=tensor.stride(),
            nbytes=tensor.untyped_storage().nbytes(),
        ),
    )


def _prepare_state_dict(
    state_dict: object,
    device: torch.device,
) -> tuple[_StateDictMeta, list[torch.Tensor]]:
    leaves: list[tuple[KeyPath, object]]
    leaves, treespec = tree_flatten_with_path(state_dict)

    paths: list[KeyPath] = []
    non_tensor_leaves: list[
        object | _TensorMeta | _DTensorMeta | _ShardedTensorMeta
    ] = []
    tensors: list[torch.Tensor] = []
    for key_path, v in leaves:
        paths.append(key_path)

        if isinstance(v, DTensor):
            tensor, tensor_meta = _prepare_tensor(v._local_tensor)

            tensors.append(tensor)

            non_tensor_leaves.append(
                _DTensorMeta(
                    local=tensor_meta,
                    spec=v._spec,
                )
            )
        elif isinstance(v, ShardedTensor):
            # Handle ShardedTensor by extracting all local shards
            local_shards = v.local_shards()

            # Prepare metadata for all local shards
            local_shards_meta = []
            local_shards_shard_metadata = []
            for shard in local_shards:
                tensor, tensor_meta = _prepare_tensor(shard.tensor)
                tensors.append(tensor)
                local_shards_meta.append(tensor_meta)
                local_shards_shard_metadata.append(shard.metadata)

            non_tensor_leaves.append(
                _ShardedTensorMeta(
                    local_shards_meta=local_shards_meta,
                    local_shards_shard_metadata=local_shards_shard_metadata,
                    sharded_tensor_metadata=v.metadata(),  # Complete metadata
                )
            )
        elif isinstance(v, torch.Tensor):
            tensor, tensor_meta = _prepare_tensor(v)
            tensors.append(tensor)
            non_tensor_leaves.append(tensor_meta)
        else:
            non_tensor_leaves.append(v)

    return (
        _StateDictMeta(
            treespec=treespec,
            paths=paths,
            non_tensor_leaves=non_tensor_leaves,
        ),
        tensors,
    )


def _cast_tensor(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Casts the underlying storage to a tensor of the given dtype.

    The returned tensor will be of size ``storage.nbytes``.

    This works for all datatypes and supports strided/offset tensors with the
    caveat that the cast tensor may be larger than the original tensor due to
    the differences in striding.
    """
    if type(tensor) is not torch.Tensor:
        raise AssertionError(f"can only cast standard tensors not {type(tensor)}")
    storage = tensor.untyped_storage()
    ret = torch.tensor(storage, dtype=dtype, device=tensor.device)
    if ret.untyped_storage() is not storage:
        raise AssertionError("storage should be the same")
    return ret


class PGTransport:
    """
    This is a checkpoint transport that uses the process group to transfer checkpoints.
    This allows for fast recovery of workers by fetching the current weights
    from an existing worker.

    Args:
        pg: the process group to use for communication
        timeout: the timeout for communication
        device: the device to use for tensors
        state_dict: if specified this function will be called to do an inplace
            receive into the returned state_dict. This is much faster than
            having to allocate new tensors and transferring them to the CPU.
    """

    def __init__(
        self,
        pg: ProcessGroup,
        timeout: timedelta,
        device: torch.device,
        state_dict: Callable[[], object] | None = None,
    ) -> None:
        self._work: list[Work] = []
        self._pg = pg
        self._timeout = timeout
        self._device = device
        self._state_dict = state_dict

    def send_checkpoint(self, dst_ranks: list[int], state_dict: object) -> None:
        """
        Send a checkpoint to multiple destination ranks.

        The process:
        1. Prepares the state dict by converting tensors to a serializable format
        2. Sends metadata as pickled data
        3. Sends each tensor sequentially to all destination ranks

        Args:
            dst_ranks: List of destination ranks to send the checkpoint to
            state_dict: The state dictionary containing model parameters
        """
        with _timeit("preparing state_dict"):
            meta, tensors = _prepare_state_dict(state_dict, device=self._device)

        work = []

        with _timeit("send meta"):
            buf = pickle.dumps(meta)
            len_t = torch.tensor([len(buf)], dtype=torch.int64, device=self._device)
            buf_t = torch.frombuffer(buf, dtype=torch.uint8).to(self._device)
            for dst_rank in dst_ranks:
                work.append(self._pg.send([len_t], dst_rank, tag=1))
                work.append(self._pg.send([buf_t], dst_rank, tag=2))

        with _timeit("send tensors"):
            for i, t in enumerate(tensors):
                original_device = t.device
                t = t.to(self._device)
                for dst_rank in dst_ranks:
                    work.append(self._pg.send([t], dst_rank, tag=3 + i))

                # if we did a copy we should wait for the work to complete so we
                # can free the memory to avoid OOMs
                if original_device == torch.device("cpu"):
                    for w in work:
                        w.wait()
                    work = []

            for w in work:
                w.wait()

    def recv_checkpoint(self, src_rank: int) -> object:
        """
        Receive a checkpoint from a source rank.

        The process:
        1. Receives metadata about the checkpoint structure
        2. Receives each tensor, potentially reusing existing tensors for in-place updates
        3. Reconstructs the original state dict structure

        Args:
            src_rank: The source rank to receive the checkpoint from

        Returns:
            The reconstructed state dictionary with model parameters
        """
        state_dict = self._state_dict() if self._state_dict else {}
        state_dict_leaves, _ = tree_flatten_with_path(state_dict)

        dst_tensors: dict[KeyPath, object] = dict(state_dict_leaves)

        len_t = torch.zeros(1, dtype=torch.int64, device=self._device)
        self._pg.recv([len_t], src_rank, tag=1).wait()
        length = cast(int, len_t.item())

        buf = torch.empty(length, dtype=torch.uint8, device=self._device)
        self._pg.recv([buf], src_rank, tag=2).wait()

        meta: _StateDictMeta = pickle.loads(buf.cpu().numpy().tobytes())

        i: int = 0
        works: list[Work] = []

        def recv(path: KeyPath, v: _TensorMeta) -> torch.Tensor:
            nonlocal i

            inplace = dst_tensors.get(path)
            if (
                isinstance(inplace, torch.Tensor)
                and inplace.device.type == self._device.type
            ):
                if isinstance(inplace, DTensor):
                    inplace = inplace._local_tensor
                t = _cast_tensor(inplace, torch.uint8)
                if t.nbytes != v.nbytes:
                    raise AssertionError("inplace tensor storage must be the same size")
            else:
                t = torch.empty(v.nbytes, dtype=torch.uint8, device=self._device)

            work = self._pg.recv([t], src_rank, tag=3 + i)
            i += 1

            if inplace is None:
                # if not inplace we need to copy it to CPU to avoid OOMing
                work.wait()
                t = t.cpu()
            else:
                works.append(work)

            return torch.as_strided(
                t.view(v.dtype),
                size=v.shape,
                stride=v.stride,
                storage_offset=v.storage_offset,
            )

        values: list[object] = []
        for path, v in zip(meta.paths, meta.non_tensor_leaves):
            if isinstance(v, _TensorMeta):
                values.append(recv(path, v))
            elif isinstance(v, _DTensorMeta):
                tensor = recv(path, v.local)
                # pyrefly: ignore [bad-argument-type, bad-argument-count, unexpected-keyword]
                values.append(DTensor(tensor, v.spec, requires_grad=False))
            elif isinstance(v, _ShardedTensorMeta):
                # Receive all local shards that were sent to us
                local_shards = []
                current_rank = self._pg.rank()

                # Receive tensors for each local shard that was sent
                for j, shard_meta in enumerate(v.local_shards_meta):
                    tensor = recv(path, shard_meta)

                    # Use the original shard metadata that was stored during preparation
                    # but update the placement to reflect the current rank/device
                    original_shard_metadata = v.local_shards_shard_metadata[j]
                    updated_shard_metadata = ShardMetadata(
                        shard_offsets=original_shard_metadata.shard_offsets,
                        shard_sizes=original_shard_metadata.shard_sizes,
                        placement=f"rank:{current_rank}/{tensor.device.type}",
                    )

                    local_shard = ShardedTensorShard(
                        tensor=tensor, metadata=updated_shard_metadata
                    )
                    local_shards.append(local_shard)

                # Use complete metadata to reconstruct ShardedTensor
                sharded_tensor = (
                    ShardedTensor._init_from_local_shards_and_global_metadata(
                        local_shards=local_shards,
                        sharded_tensor_metadata=v.sharded_tensor_metadata,
                    )
                )
                values.append(sharded_tensor)
            else:
                values.append(v)

        for work in works:
            work.wait()

        return tree_unflatten(values, meta.treespec)
<|endoftext|>from typing import Any, TypeVar
from typing_extensions import Protocol, runtime_checkable


__all__ = ["Stateful", "StatefulT"]


@runtime_checkable
class Stateful(Protocol):
    """
    Stateful protocol for objects that can be checkpointed and restored.
    """

    def state_dict(self) -> dict[str, Any]:
        """
        Objects should return their state_dict representation as a dictionary.
        The output of this function will be checkpointed, and later restored in
        `load_state_dict()`.

        .. warning::
            Because of the inplace nature of restoring a checkpoint, this function
            is also called during `torch.distributed.checkpoint.load`.


        Returns:
            Dict: The objects state dict
        """

        ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Restore the object's state from the provided state_dict.

        Args:
            state_dict: The state dict to restore from
        """

        ...


StatefulT = TypeVar("StatefulT", bound=Stateful)
<|endoftext|># mypy: allow-untyped-defs
import io
import itertools
from bisect import bisect_right, insort
from collections.abc import Callable
from typing import Any, cast

import torch
import torch.distributed as dist
from torch._utils import _get_device_module
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from .metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    MetadataIndex,
    STATE_DICT_TYPE,
    STORAGE_TYPES,
    TensorProperties,
    TensorStorageMetadata,
)
from .planner import (
    LoadItemType,
    ReadItem,
    SavePlan,
    TensorWriteData,
    WriteItem,
    WriteItemType,
)
from .protocol import (
    _copy_checkpointable_tensor_metadata,
    _get_checkpointable_tensor_chunks,
    _is_checkpointable_tensor,
    CheckpointableTensor,
)
from .resharding import (
    _check_shard_metadata_pair_overlap,
    _shards_get_overlap_region_wrt_saved_tensor,
)


__all__: list[str] = ["create_read_items_for_chunk_list"]


def _compare_save_plans(plan: SavePlan, other_plan: SavePlan) -> bool:
    """
    Compare the two Save plans and return True if they are equal.

    Args:
        plan (SavePlan): First SavePlan to compare.
        other_plan (SavePlan): Second SavePlan to compare.

    Returns:
       True if the two plans are equal, False otherwise.
    """
    if plan.usable != other_plan.usable:
        return False

    # Both the plans should have the same number of items
    if len(plan.items) != len(other_plan.items):
        return False

    # Both the plans should have the same write items.
    for plan_item, other_plan_item in zip(plan.items, other_plan.items):
        # Write item type should be same
        if plan_item.type != other_plan_item.type:
            return False

        plan_metadata_index = plan_item.index
        other_plan_metadata_index = other_plan_item.index

        # Write item metadata_index should be same
        if (
            plan_metadata_index.fqn != other_plan_metadata_index.fqn
            or plan_metadata_index.offset != other_plan_metadata_index.offset
            or plan_metadata_index.index != other_plan_metadata_index.index
        ):
            return False

        # Write item tensor_data should be present in both the write items plans, if it exists in either of them.
        tensor_data = plan_item.tensor_data
        other_tensor_data = other_plan_item.tensor_data
        if (tensor_data and not other_tensor_data) or (
            not tensor_data and other_tensor_data
        ):
            return False

        if tensor_data and other_tensor_data:
            # Write item tensor_data size should be same
            if tensor_data.size != other_tensor_data.size:
                return False

            # Write item tensor_data chunk should be present in both the write items, if it exists in either of them.
            chunk = tensor_data.chunk
            other_chunk = other_tensor_data.chunk
            if (chunk and not other_chunk) or (not chunk and other_chunk):
                return False

            # Write item tensor_data chunk offsets and sizes should be same
            if chunk and other_chunk:
                if (
                    chunk.offsets != other_chunk.offsets
                    or chunk.sizes != other_chunk.sizes
                ):
                    return False

    return True


def _contains_usable_plan(delta_plans: list[SavePlan]) -> bool:
    """
    Check if any delta plan is usable, indicating the plan has changed.

    Args:
        delta_plans (List[SavePlan]): A list of delta plans to check.
    Returns:
        True if any delta plan is usable, False otherwise.
    """
    return any(delta_plan and delta_plan.usable for delta_plan in delta_plans)


def _merge_delta_local_plans(
    cached_plans: list[SavePlan],
    delta_plans: list[SavePlan],
) -> list[SavePlan]:
    """
    Merge a list of delta plans into a single plan.

    Args:
        cached_plans (List[SavePlan]): A list of cached plans.
        delta_plans (List[SavePlan]): A list of delta plans to merge. It can contain empty plans

    Returns:
        A single merged plan. If a delta plan is not usable, use the cached plan. Otherwise, use the delta plan.
    """
    merged_plans = []

    for cached_plan, delta_plan in zip(cached_plans, delta_plans):
        if delta_plan and not delta_plan.usable:
            merged_plans.append(cached_plan)
        else:
            merged_plans.append(delta_plan)

    return merged_plans


def _create_chunk_from_tensor(tensor: torch.Tensor) -> ChunkStorageMetadata:
    return ChunkStorageMetadata(
        offsets=torch.Size([0] * len(tensor.size())), sizes=tensor.size()
    )


def _chunk_for_shard(shard_md: ShardMetadata) -> ChunkStorageMetadata:
    return ChunkStorageMetadata(
        offsets=torch.Size(shard_md.shard_offsets),
        sizes=torch.Size(shard_md.shard_sizes),
    )


def _sharded_tensor_metadata(
    sharded_tensor: ShardedTensor, shard_md: ShardMetadata
) -> TensorWriteData:
    shard_properties = sharded_tensor.metadata().tensor_properties

    properties = TensorProperties(
        dtype=shard_properties.dtype,
        layout=shard_properties.layout,
        requires_grad=shard_properties.requires_grad,
        memory_format=shard_properties.memory_format,
        pin_memory=shard_properties.pin_memory,
    )

    return TensorWriteData(
        chunk=_chunk_for_shard(shard_md),
        properties=properties,
        size=sharded_tensor.metadata().size,
    )


def _create_write_items_for_dtensor(fqn: str, tensor: DTensor) -> WriteItem:
    sizes, offsets = compute_local_shape_and_global_offset(
        tensor.shape, tensor.device_mesh, tensor.placements
    )
    sizes, offsets = torch.Size(sizes), torch.Size(offsets)

    return WriteItem(
        index=MetadataIndex(fqn, offsets),
        type=WriteItemType.SHARD,
        tensor_data=TensorWriteData(
            chunk=ChunkStorageMetadata(
                offsets=offsets,
                sizes=sizes,
            ),
            properties=TensorProperties.create_from_tensor(tensor.to_local()),
            size=tensor.size(),
        ),
    )


def _create_write_item_for_shard(
    fqn: str, sharded_tensor: ShardedTensor, shard_md: ShardMetadata
) -> WriteItem:
    offsets = torch.Size(shard_md.shard_offsets)
    return WriteItem(
        index=MetadataIndex(fqn, offsets),
        type=WriteItemType.SHARD,
        tensor_data=_sharded_tensor_metadata(sharded_tensor, shard_md),
    )


def _create_write_item_for_tensor(fqn: str, tensor: torch.Tensor) -> WriteItem:
    offsets = torch.Size([0] * len(tensor.size()))
    return WriteItem(
        index=MetadataIndex(fqn, offsets),
        type=WriteItemType.TENSOR,
        tensor_data=TensorWriteData(
            chunk=ChunkStorageMetadata(offsets=offsets, sizes=tensor.size()),
            properties=TensorProperties.create_from_tensor(tensor),
            size=tensor.size(),
        ),
    )


def _get_checkpointable_tensor_write_items(
    fqn: str, tensor: CheckpointableTensor
) -> list[WriteItem]:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("CheckpointableTensor must also be a torch.Tensor")
    properties = TensorProperties.create_from_tensor(tensor)
    return [
        WriteItem(
            index=MetadataIndex(fqn, chunk.offsets, idx),
            type=WriteItemType.SHARD,
            tensor_data=TensorWriteData(
                chunk=chunk,
                properties=properties,
                size=torch.Size(tensor.global_shape),
            ),
        )
        for idx, chunk in enumerate(_get_checkpointable_tensor_chunks(tensor))
    ]


def _create_write_item_for_bytesio(fqn: str, bytes: Any):
    return WriteItem(
        index=MetadataIndex(fqn),
        type=WriteItemType.BYTE_IO,
    )


def _create_read_item_for_byteio(
    dest_index, dest_offset, storage_index, storage_offset, length
):
    return ReadItem(
        type=LoadItemType.BYTE_IO,
        dest_index=dest_index,
        dest_offsets=torch.Size((dest_offset,)),
        storage_index=storage_index,
        storage_offsets=torch.Size((storage_offset,)),
        lengths=torch.Size((length,)),
    )


def _create_read_item_for_tensor(
    dest_index, dest_offsets, storage_index, storage_offsets, lengths
):
    return ReadItem(
        type=LoadItemType.TENSOR,
        dest_index=dest_index,
        dest_offsets=torch.Size(dest_offsets),
        storage_index=storage_index,
        storage_offsets=torch.Size(storage_offsets),
        lengths=torch.Size(lengths),
    )


def create_read_items_for_chunk_list(
    fqn: str,
    checkpoint_md: TensorStorageMetadata,
    local_chunks: list[ChunkStorageMetadata],
) -> list[ReadItem]:
    """
    Create a list of ``ReadItem`` based on the checkpoint and local chunks.

    This applies the resharding algorithm and computes the reads needed
    to satisfy ``local_chunks`` with a checkpoint described by ``checkpoint_md``.

    Args:
        fqn (str) : The state_dict FQN to pass to ``ReadItem``.
        checkpoint_md (TensorStorageMetadata): metadata for a given tensor
            from a checkpoint.
        local_chunks (List[ChunkStorageMetadata]): Local chunks that needs to be
            loaded.

    Returns:
        A list of ``ReadItem`` that will satisfy all input chunks.
    """
    read_items: list[ReadItem] = []
    saved_chunks = checkpoint_md.chunks

    if not local_chunks or not saved_chunks:
        return read_items

    num_dims = len(local_chunks[0].offsets)

    # Find sweep dimension (dimension with largest extent for better pruning)
    sweep_dim = 0
    if num_dims > 1:
        max_size = 0
        for dim in range(num_dims):
            dim_size = max(
                chunk.offsets[dim] + chunk.sizes[dim]
                for chunk in itertools.chain(local_chunks, saved_chunks)
            )
            if dim_size > max_size:
                max_size = dim_size
                sweep_dim = dim

    # Pre-compute bounds: (start, end) for each chunk in sweep dimension
    # For 0-d tensors, use (0, 1) so all chunks overlap in the sweep line
    if num_dims == 0:
        saved_bounds = [(0, 1)] * len(saved_chunks)
        local_bounds = [(0, 1)] * len(local_chunks)
    else:
        saved_bounds = [
            (c.offsets[sweep_dim], c.offsets[sweep_dim] + c.sizes[sweep_dim])
            for c in saved_chunks
        ]
        local_bounds = [
            (c.offsets[sweep_dim], c.offsets[sweep_dim] + c.sizes[sweep_dim])
            for c in local_chunks
        ]

    saved_sorted_indices = sorted(
        range(len(saved_chunks)),
        key=lambda idx: saved_bounds[idx][0],
    )
    local_sorted_indices = sorted(
        range(len(local_chunks)),
        key=lambda idx: local_bounds[idx][0],
    )

    active_saved: list[tuple[int, int]] = []
    saved_ptr = 0
    num_saved = len(saved_sorted_indices)

    for local_idx in local_sorted_indices:
        local_chunk = local_chunks[local_idx]
        local_start, local_end = local_bounds[local_idx]

        cutoff = bisect_right(active_saved, (local_start, -1))
        if cutoff:
            del active_saved[:cutoff]

        while saved_ptr < num_saved:
            storage_idx = saved_sorted_indices[saved_ptr]
            storage_chunk = saved_chunks[storage_idx]
            saved_start, saved_end = saved_bounds[storage_idx]

            if saved_start >= local_end:
                break

            insort(active_saved, (saved_end, storage_idx))
            saved_ptr += 1

        for _, storage_idx in active_saved:
            storage_chunk = saved_chunks[storage_idx]
            if not _check_shard_metadata_pair_overlap(local_chunk, storage_chunk):
                continue

            storage_offsets = []
            dest_offsets = []
            lengths = []
            for (
                _dim,
                offset_for_saved_tensor,
                offset_for_current_tensor,
                length,
            ) in _shards_get_overlap_region_wrt_saved_tensor(
                saved_shard=storage_chunk, current_shard=local_chunk
            ):
                storage_offsets.append(offset_for_saved_tensor)
                dest_offsets.append(offset_for_current_tensor)
                lengths.append(length)

            read_items.append(
                _create_read_item_for_tensor(
                    dest_index=MetadataIndex(fqn, local_chunk.offsets, local_idx),
                    dest_offsets=dest_offsets,
                    storage_index=MetadataIndex(
                        fqn, storage_chunk.offsets, storage_idx
                    ),
                    storage_offsets=storage_offsets,
                    lengths=lengths,
                )
            )
    return read_items


def _create_default_metadata_only_plan(state_dict: STATE_DICT_TYPE) -> SavePlan:
    requests = []
    for fqn, obj in state_dict.items():
        if isinstance(obj, DTensor):
            requests.append(_create_write_items_for_dtensor(fqn, obj))
        elif isinstance(obj, ShardedTensor):
            requests.extend(
                _create_write_item_for_shard(fqn, obj, shard_md)
                for shard_md in obj.metadata().shards_metadata
            )
        elif _is_checkpointable_tensor(obj):
            # Avoid treating a local shard as a plain Tensor: the WriteItems
            # carry both local chunk metadata and the global tensor size.
            requests.extend(_get_checkpointable_tensor_write_items(fqn, obj))
        elif isinstance(obj, torch.Tensor):
            requests.append(_create_write_item_for_tensor(fqn, obj))
        else:
            requests.append(_create_write_item_for_bytesio(fqn, obj))
    return SavePlan(requests)


def _create_write_items(fqn: str, object: Any) -> list[WriteItem]:
    if hasattr(object, "__create_write_items__"):
        # DTensor implements _Checkpointable
        return object.__create_write_items__(fqn, object)  # type: ignore[attr-defined]
    elif isinstance(object, ShardedTensor):
        return [
            _create_write_item_for_shard(fqn, object, shard.metadata)
            for shard in object.local_shards()
        ]
    elif _is_checkpointable_tensor(object):
        return _get_checkpointable_tensor_write_items(fqn, object)
    elif isinstance(object, torch.Tensor):
        return [_create_write_item_for_tensor(fqn, object)]
    else:
        return [_create_write_item_for_bytesio(fqn, object)]


def _create_chunk_from_dtensor(tensor: DTensor) -> ChunkStorageMetadata:
    sizes, offsets = compute_local_shape_and_global_offset(
        tensor.shape, tensor.device_mesh, tensor.placements
    )
    sizes, offsets = torch.Size(sizes), torch.Size(offsets)
    return ChunkStorageMetadata(
        offsets=offsets,
        sizes=sizes,
    )


def _create_chunk_list(tensor: torch.Tensor) -> list[ChunkStorageMetadata]:
    if hasattr(tensor, "__create_chunk_list__"):
        # DTensor implements _Checkpointable
        local_chunks = tensor.__create_chunk_list__()  # type: ignore[attr-defined]
    elif _is_checkpointable_tensor(tensor):
        local_chunks = _get_checkpointable_tensor_chunks(tensor)
    elif isinstance(tensor, ShardedTensor):
        local_chunks = [
            _chunk_for_shard(shard.metadata) for shard in tensor.local_shards()
        ]
    elif isinstance(tensor, torch.Tensor):
        local_chunks = [_create_chunk_from_tensor(tensor)]
    else:
        raise ValueError(
            "Unsupported Type, expecting one of [Tensor, DTensor, ShardedTensor] "
            f",but got {type(tensor)}"
        )

    return local_chunks


def _create_read_items(fqn: str, md: STORAGE_TYPES, obj: Any) -> list[ReadItem]:
    if not isinstance(md, BytesStorageMetadata):
        try:
            local_chunks = _create_chunk_list(obj)
        except ValueError as ex:
            raise ValueError(
                f"Invalid checkpoint metadata for {fqn}, "
                + f"expected BytesStorageMetadata but found {type(md)}",
            ) from ex

        return create_read_items_for_chunk_list(fqn, md, local_chunks)
    else:
        return [
            _create_read_item_for_byteio(
                dest_index=MetadataIndex(fqn),
                dest_offset=0,
                storage_index=MetadataIndex(fqn),
                storage_offset=0,
                length=0,
            )
        ]


def _init_state_dict(state_dict: dict[str, Any]) -> Any:
    """
    Initializes meta tensor if the meta tensor is DTensor or torch.Tensor.
    """

    def dtensor_func(value: DTensor):
        device = getattr(value, "device", None)
        if device == torch.device("meta"):
            device_type = dist.distributed_c10d._get_pg_default_device().type
            device = cast(
                torch.device, _get_device_module(device_type).current_device()
            )
            new_local_tensor = torch.empty_like(value.to_local(), device=device)
            # We need to pass shape and stride explicitly, since DTensor might be
            # sharded unevenly.
            dtensor = DTensor.from_local(
                new_local_tensor,
                device_mesh=value.device_mesh,
                placements=value.placements,
                shape=value.size(),
                stride=value.stride(),
            )
            return dtensor
        else:
            return value

    def sharded_tensor_func(value: Any):
        device = getattr(value, "device", None)
        if device == torch.device("meta"):
            raise RuntimeError(
                f"Found unsupported type {type(value)} for meta device loading."
            )
        else:
            return value

    def tensor_func(value: torch.Tensor):
        device = getattr(value, "device", None)
        if device == torch.device("meta"):
            device_type = dist.distributed_c10d._get_pg_default_device().type
            device = cast(
                torch.device, _get_device_module(device_type).current_device()
            )
            tensor = torch.empty_like(value, device=device)
            if _is_checkpointable_tensor(value):
                _copy_checkpointable_tensor_metadata(value, tensor)
            return tensor
        else:
            return value

    _iterate_state_dict(
        state_dict,
        dtensor_func,
        sharded_tensor_func,
        tensor_func,
    )


def _iterate_state_dict(
    iter_object: Any,
    dtensor_func: Callable,
    sharded_tensor_func: Callable,
    tensor_func: Callable,
):
    """
    Iterate through the state dict, applying the given functions to each tensor type
    and update the state dict in place.

    Args:
        iter_object (Any): the target state_dict.
        sharded_tensor_func (Callable): the function to apply to ShardedTensor
        dtensor_func (Callable): the function to apply to DTensor
        tensor_func (Callable): the function to apply to Tensor

    # TODO: let state_dict_util._iterate_state_dict() to support in place option
    so we don't need to have two versions of _iterate_state_dict.
    """

    if isinstance(iter_object, DTensor):
        return dtensor_func(iter_object)
    elif isinstance(iter_object, ShardedTensor):
        return sharded_tensor_func(iter_object)
    elif isinstance(iter_object, torch.Tensor):
        return tensor_func(iter_object)
    elif (
        isinstance(iter_object, (int, float, str, bytes, io.BytesIO))
        or iter_object is None
    ):
        return iter_object
    elif isinstance(iter_object, dict):
        for key, value in iter_object.items():
            iter_object[key] = _iterate_state_dict(
                value, dtensor_func, sharded_tensor_func, tensor_func
            )
        return iter_object
    elif isinstance(iter_object, (list, tuple)):
        ret = [
            _iterate_state_dict(v, dtensor_func, sharded_tensor_func, tensor_func)
            for v in iter_object
        ]
        if isinstance(iter_object, tuple):
            ret = tuple(ret)  # type: ignore[assignment]
        return ret
<|endoftext|>from concurrent.futures import Future
from typing import Any

import torch.distributed as dist
import torch.distributed.checkpoint.state_dict_loader as loader
import torch.distributed.checkpoint.state_dict_saver as saver
from torch.distributed.checkpoint.metadata import Metadata, STATE_DICT_TYPE
from torch.distributed.checkpoint.storage import (
    LoadPlanner,
    SavePlanner,
    StorageReader,
    StorageWriter,
)


__all__: list[str] = []


class _Checkpointer:
    """This base class specifies a high level API for saving and loading
    distributed `state_dict` 's. It provides an abstraction over the low-level APIs
    provided by :py:mod:`torch.distributed.checkpoint.storage`, essentially calling
    :py:meth: `torch.distributed.state_dict_saver.save` and
    :py:meth: `torch.distributed.state_dict_loader.load` with the provided storage
    readers and writers.

    .. warning::
        This feature is experimental and subject to removal/change.

    """

    def __init__(
        self,
        storage_writer: StorageWriter,
        storage_reader: StorageReader,
        *,
        process_group: dist.ProcessGroup | None = None,
        coordinator_rank: int = 0,
        no_dist: bool = False,
        load_planner: LoadPlanner | None = None,
        save_planner: SavePlanner | None = None,
    ):
        """Initializes the Checkpointer instance.

        Args:
            storage_writer: Instance of StorageWrite used to perform writes.
            storage_reader: StorageReader used to load data from.
            process_group: ProcessGroup to be used for cross-rank synchronization.
            coordinator_rank: Rank to use to coordinate the checkpoint. rank0 is used by default.
            no_dist: If ``True``, distributed checkpoint will not load in SPMD style. (Default: ``False``)
            loader_planner: Instance of LoadPlanner to use when loading.
            save_planner: Instance of SavePlanner to use when saving.
        """
        self.storage_writer = storage_writer
        self.storage_reader = storage_reader
        self.process_group = process_group
        self.coordinator_rank = coordinator_rank
        self.no_dist = no_dist
        self.load_planner = load_planner
        self.save_planner = save_planner

    def save(
        self,
        state_dict: STATE_DICT_TYPE,
    ) -> Metadata:
        """Calls :py:meth: `torch.distributed.state_dict_saver.save`. Utilizing values passed during initialization."""
        return saver.save(
            state_dict,
            self.storage_writer,
            process_group=self.process_group,
            coordinator_rank=self.coordinator_rank,
            no_dist=self.no_dist,
            planner=self.save_planner,
        )

    def async_save(
        self,
        state_dict: STATE_DICT_TYPE,
    ) -> Future:
        """
        Calls :py:meth: `torch.distributed.state_dict_saver._async_save`. Utilizing values passed during initialization.

        Returns:
            Future: A future holding the resultant Metadata object from `save`.
