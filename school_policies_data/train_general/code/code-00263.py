        param_init_fn: Callable[[nn.Module], None] | None = None,
        device_id: int | torch.device | None = None,
        sync_module_states: bool = False,
        forward_prefetch: bool = False,
        limit_all_gathers: bool = True,
        use_orig_params: bool = False,
        ignored_states: Iterable[torch.nn.Parameter]
        | Iterable[torch.nn.Module]
        | None = None,
        device_mesh: DeviceMesh | None = None,
    ):
        torch._C._log_api_usage_once("torch.distributed.fsdp")
        super().__init__()
        if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
            warnings.warn(
                "FSDP will not all-gather parameters for containers that do "
                f"not implement forward: {module}",
                stacklevel=2,
            )
        _init_ignored_module_states(self, module, ignored_modules, ignored_states)
        _init_device_handle(self, module, self._ignored_params, device_id)

        # Add module annotations for Dynamo support (see function for details)
        _annotate_modules_for_dynamo(module, self._ignored_modules, use_orig_params)

        # Initializes self.process_group, along with rank and world size. This will
        # also set another attribute, _inter_node_pg, to control the process group
        # over which sharding occurs, if sharding_strategy is {HYBRID_SHARD, _HYBRID_SHARD_ZERO2}.
        # Note that this is done before auto_wrapping, so that child FSDP modules simply pick up
        # the same process group state as the root FSDP module.
        self._device_mesh = device_mesh
        _init_process_group_state(
            self,
            process_group,
            sharding_strategy,
            auto_wrap_policy,
            device_mesh,
        )
        if auto_wrap_policy is not None:
            root_kwargs = {
                "process_group": process_group,
                "sharding_strategy": sharding_strategy,
                "cpu_offload": cpu_offload,
                "backward_prefetch": backward_prefetch,
                "mixed_precision": mixed_precision,
                "param_init_fn": param_init_fn,
                "device_id": device_id,
                "sync_module_states": sync_module_states,
                "forward_prefetch": forward_prefetch,
                "limit_all_gathers": limit_all_gathers,
                "use_orig_params": use_orig_params,
                "ignored_states": self._ignored_params,
                "device_mesh": device_mesh,
            }
            if sharding_strategy in HYBRID_SHARDING_STRATEGIES and device_mesh is None:
                # Share root process groups with children to maintain
                # the invariant that all FSDP modules will have the same
                # process groups.
                root_kwargs["process_group"] = (self.process_group, self._inter_node_pg)

            _auto_wrap(
                module,
                auto_wrap_policy,
                self._ignored_modules,
                self._ignored_params,
                root_kwargs,
                FullyShardedDataParallel,
            )

        backward_prefetch_limit = 1
        forward_prefetch_limit = 1
        _init_core_state(
            self,
            sharding_strategy,
            mixed_precision,
            cpu_offload,
            limit_all_gathers,
            use_orig_params,
            backward_prefetch_limit,
            forward_prefetch_limit,
        )
        _init_runtime_state(self)
        _init_prefetching_state(self, backward_prefetch, forward_prefetch)
        _init_buffer_state(self, module)
        # extension needs to be set before `_init_param_handle_from_module()`
        _init_extension(self, device_mesh)
        _init_param_handle_from_module(
            self,
            module,
            device_id,
            param_init_fn,
            sync_module_states,
        )
        self._fsdp_wrapped_module = module
        if not use_orig_params:
            _check_orig_params_flattened(self, self._ignored_params)
            _register_flat_param(self, self)

        # `_state_dict_type` controls the `state_dict()` behavior, which is
        # implemented using post-save and pre-load hooks
        _init_state_dict_state(self)
        _register_all_state_dict_hooks(self)
        self._zero_scalar = None

    @property
    def module(self) -> nn.Module:
        """Return the wrapped module."""
        # FSDP's `.module` must refer to the innermost wrapped module when
        # composing with other module wrappers in order for state dict to work
        if isinstance(self._fsdp_wrapped_module, ActivationWrapper):
            return getattr(self._fsdp_wrapped_module, _CHECKPOINT_WRAPPED_MODULE)
        return self._fsdp_wrapped_module

    @property
    def _has_params(self) -> bool:
        """Returns whether this FSDP instance manages any parameters."""
        return hasattr(self, "_handle") and self._handle is not None

    @property
    def _flat_param(self) -> FlatParameter | None:
        return self._handle.flat_param if self._handle else None

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self._fsdp_wrapped_module, name)

    def __getitem__(self, key: int) -> Any:
        """Forward indexing calls in case the module is an ``nn.Sequential``."""
        if hasattr(self, FSDP_WRAPPED_MODULE):
            return self._fsdp_wrapped_module.__getitem__(key)  # type: ignore[operator]
        return super().__getitem__(key)

    def check_is_root(self) -> bool:
        """Check if this instance is a root FSDP module."""
        return _is_fsdp_root(self, self)

    @staticmethod
    def fsdp_modules(
        module: nn.Module,
        root_only: bool = False,
    ) -> list["FullyShardedDataParallel"]:
        """Return all nested FSDP instances.

        This possibly includes ``module`` itself and only includes FSDP root modules if ``root_only=True``.

        Args:
            module (torch.nn.Module): Root module, which may or may not be an
                ``FSDP`` module.
            root_only (bool): Whether to return only FSDP root modules.
                (Default: ``False``)

        Returns:
            List[FullyShardedDataParallel]: FSDP modules that are nested in
            the input ``module``.
        """
        if root_only:
            return _get_fsdp_root_states(module)
        return traversal_utils._get_fsdp_states(module)

    def apply(self, fn: Callable[[nn.Module], None]) -> "FullyShardedDataParallel":
        r"""Apply ``fn`` recursively to every submodule (as returned by ``.children()``) as well as self.

        Typical use includes initializing the parameters of a model (see also :ref:`nn-init-doc`).

        Compared to ``torch.nn.Module.apply``, this version additionally gathers
        the full parameters before applying ``fn``. It should not be called from
        within another ``summon_full_params`` context.

        Args:
            fn (:class:`Module` -> None): function to be applied to each submodule

        Returns:
            Module: self
        """
        uninitialized = self._is_root is None
        self._assert_state(TrainingState.IDLE)
        # Use `_unshard_params_for_summon()` with `recurse=False` instead of
        # `_unshard_fsdp_state_params()` directly to perform lazy
        # initialization, which is needed to initialize `FlatParameter`
        # parameter attributes as required by the unshard logic
        with _unshard_params_for_summon(
            self,
            self,
            writeback=True,
            rank0_only=False,
            offload_to_cpu=False,
            with_grads=False,
        ):
            ret = super().apply(fn)

        # Reset lazy init called in `_unshard_params_for_summon()` since
        # `apply()` may have been called on FSDP instance that is not truly a
        # root, in which case it will be incorrectly marked as one.
        if uninitialized and self._is_root:
            for module in traversal_utils._get_fsdp_states(self):
                module._reset_lazy_init()

        return ret

    def _mixed_precision_enabled_for_buffers(self) -> bool:
        """Return whether the user explicitly enabled buffer mixed precision.

        NOTE: Unlike parameters and gradient reduction, buffer mixed precision
        is applied at the FSDP instance level, not the ``FlatParameter`` level,
        which may be different for the composable code path.
        """
        return self.mixed_precision.buffer_dtype is not None

    def _low_precision_hook_enabled(self) -> bool:
        """Whether a low precision hook is registered or not."""
        return self._comm_hook is not None and self._comm_hook in LOW_PRECISION_HOOKS

    def _reset_lazy_init(self) -> None:
        """Reset instance so :func:`_lazy_init` will run on the next forward."""
        self._is_root: bool | None = None

    @staticmethod
    def set_state_dict_type(
        module: nn.Module,
        state_dict_type: StateDictType,
        state_dict_config: StateDictConfig | None = None,
        optim_state_dict_config: OptimStateDictConfig | None = None,
    ) -> StateDictSettings:
        """Set the ``state_dict_type`` of all the descendant FSDP modules of the target module.

        Also takes (optional) configuration for the model's and optimizer's state dict.
        The target module does not have to be a FSDP module. If the target
        module is a FSDP module, its ``state_dict_type`` will also be changed.

        .. note:: This API should be called for only the top-level (root)
            module.

        .. note:: This API enables users to transparently use the conventional
            ``state_dict`` API to take model checkpoints in cases where the
            root FSDP module is wrapped by another ``nn.Module``. For example,
            the following will ensure ``state_dict`` is called on all non-FSDP
            instances, while dispatching into `sharded_state_dict` implementation
            for FSDP:

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> model = DDP(FSDP(...))
            >>> FSDP.set_state_dict_type(
            >>>     model,
            >>>     StateDictType.SHARDED_STATE_DICT,
            >>>     state_dict_config = ShardedStateDictConfig(offload_to_cpu=True),
            >>>     optim_state_dict_config = OptimStateDictConfig(offload_to_cpu=True),
            >>> )
            >>> param_state_dict = model.state_dict()
            >>> optim_state_dict = FSDP.optim_state_dict(model, optim)

        Args:
            module (torch.nn.Module): Root module.
            state_dict_type (StateDictType): the desired ``state_dict_type`` to set.
            state_dict_config (Optional[StateDictConfig]): the configuration for the
                target ``state_dict_type``.
            optim_state_dict_config (Optional[OptimStateDictConfig]): the configuration
                for the optimizer state dict.

        Returns:
            A StateDictSettings that include the previous state_dict type and
            configuration for the module.
        """
        warnings.warn(
            "FSDP.state_dict_type() and FSDP.set_state_dict_type() are being "
            "deprecated. Please use APIs, get_state_dict() and set_state_dict(), "
            "which can support different parallelisms, FSDP1, FSDP2, DDP. "
            "API doc: https://pytorch.org/docs/stable/distributed.checkpoint.html"
            "#torch.distributed.checkpoint.state_dict.get_state_dict ."
            "Tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html .",
            FutureWarning,
            stacklevel=2,
        )
        _state_dict_type_to_config = {
            StateDictType.FULL_STATE_DICT: FullStateDictConfig,
            StateDictType.LOCAL_STATE_DICT: LocalStateDictConfig,
            StateDictType.SHARDED_STATE_DICT: ShardedStateDictConfig,
        }
        _optim_state_dict_type_to_config = {
            StateDictType.FULL_STATE_DICT: FullOptimStateDictConfig,
            StateDictType.LOCAL_STATE_DICT: LocalOptimStateDictConfig,
            StateDictType.SHARDED_STATE_DICT: ShardedOptimStateDictConfig,
        }

        # Use the default config if a state_dict config is not set.
        state_dict_config_type = _state_dict_type_to_config[state_dict_type]
        optim_state_dict_config_type = _optim_state_dict_type_to_config[state_dict_type]
        if state_dict_config is None:
            state_dict_config = state_dict_config_type()
        if optim_state_dict_config is None:
            optim_state_dict_config = optim_state_dict_config_type()
        if state_dict_config_type is not type(state_dict_config):
            raise RuntimeError(
                f"Expected state_dict_config of type {state_dict_config_type} "
                f"but got {type(state_dict_config)}"
            )
        if optim_state_dict_config_type is not type(optim_state_dict_config):
            raise RuntimeError(
                f"Expected optim_state_dict_config of type {optim_state_dict_config_type} "
                f"but got {type(optim_state_dict_config)}"
            )

        # Set the state_dict type and configurations.
        prev_state_dict_type = None
        prev_state_dict_config = None
        prev_optim_state_dict_config = None
        for submodule in traversal_utils._get_fsdp_states(module):
            if prev_state_dict_type is None:
                prev_state_dict_type = submodule._state_dict_type
            else:
                if prev_state_dict_type != submodule._state_dict_type:
                    raise AssertionError(
                        "All FSDP modules should have the same state_dict_type."
                    )
            if prev_state_dict_config is None:
                prev_state_dict_config = submodule._state_dict_config
            else:
                if not isinstance(
                    submodule._state_dict_config, type(prev_state_dict_config)
                ):
                    raise AssertionError(
                        "All FSDP modules must have the same type of state_dict_config."
                    )
            if prev_optim_state_dict_config is None:
                prev_optim_state_dict_config = submodule._optim_state_dict_config
            else:
                if not isinstance(
                    submodule._optim_state_dict_config,
                    type(prev_optim_state_dict_config),
                ):
                    raise AssertionError(
                        "All FSDP modules must have the same type of optim_state_dict_config."
                    )

            submodule._state_dict_type = state_dict_type
            submodule._state_dict_config = state_dict_config
            submodule._optim_state_dict_config = optim_state_dict_config

        return StateDictSettings(
            prev_state_dict_type, prev_state_dict_config, prev_optim_state_dict_config
        )

    @staticmethod
    def get_state_dict_type(module: nn.Module) -> StateDictSettings:
        """Get the state_dict_type and the corresponding configurations for the FSDP modules rooted at ``module``.

        The target module does not have to be an FSDP module.

        Returns:
            A ``StateDictSettings`` containing the state_dict_type and
            state_dict / optim_state_dict configs that are currently set.

        Raises:
            ``AssertionError`` if the ``StateDictSettings`` for different
            FSDP submodules differ.
        """
        state_dict_settings: StateDictSettings | None = None
        for submodule in FullyShardedDataParallel.fsdp_modules(module):
            if state_dict_settings is None:
                state_dict_settings = StateDictSettings(
                    state_dict_type=submodule._state_dict_type,
                    state_dict_config=submodule._state_dict_config,
                    optim_state_dict_config=submodule._optim_state_dict_config,
                )
                _set_optim_use_dtensor(submodule, state_dict_settings)
            else:
                submodule_settings = StateDictSettings(
                    submodule._state_dict_type,
                    submodule._state_dict_config,
                    submodule._optim_state_dict_config,
                )
                if state_dict_settings != submodule_settings:
                    raise AssertionError(
                        "All FSDP modules must have the same state dict settings."
                        f"Got {submodule_settings} and {state_dict_settings}."
                    )
                _set_optim_use_dtensor(submodule, submodule_settings)
        return state_dict_settings

    @staticmethod
    @contextlib.contextmanager
    def state_dict_type(
        module: nn.Module,
        state_dict_type: StateDictType,
        state_dict_config: StateDictConfig | None = None,
        optim_state_dict_config: OptimStateDictConfig | None = None,
    ) -> Generator:
        """Set the ``state_dict_type`` of all the descendant FSDP modules of the target module.

        This context manager has the same functions as :meth:`set_state_dict_type`. Read the document of
        :meth:`set_state_dict_type` for the detail.

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> model = DDP(FSDP(...))
            >>> with FSDP.state_dict_type(
            >>>     model,
            >>>     StateDictType.SHARDED_STATE_DICT,
            >>> ):
            >>>     checkpoint = model.state_dict()

        Args:
            module (torch.nn.Module): Root module.
            state_dict_type (StateDictType): the desired ``state_dict_type`` to set.
            state_dict_config (Optional[StateDictConfig]): the model ``state_dict``
                configuration for the target ``state_dict_type``.
            optim_state_dict_config (Optional[OptimStateDictConfig]): the optimizer
               ``state_dict`` configuration for the target ``state_dict_type``.
        """
        prev_state_dict_settings = FullyShardedDataParallel.set_state_dict_type(
            module,
            state_dict_type,
            state_dict_config,
            optim_state_dict_config,
        )
        yield
        FullyShardedDataParallel.set_state_dict_type(
            module,
            prev_state_dict_settings.state_dict_type,
            prev_state_dict_settings.state_dict_config,
            prev_state_dict_settings.optim_state_dict_config,
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the forward pass for the wrapped module, inserting FSDP-specific pre- and post-forward sharding logic."""
        handle = self._handle
        with torch.autograd.profiler.record_function(
            "FullyShardedDataParallel.forward"
        ):
            args, kwargs = _root_pre_forward(self, self, args, kwargs)
            unused = None
            args, kwargs = _pre_forward(
                self,
                handle,
                _pre_forward_unshard,
                self._fsdp_wrapped_module,
                args,
                kwargs,
            )
            if handle:
                _p_assert(
                    handle.flat_param.device == self.compute_device,
                    "Expected `FlatParameter` to be on the compute device "
                    f"{self.compute_device} but got {handle.flat_param.device}",
                )
            output = self._fsdp_wrapped_module(*args, **kwargs)
            return _post_forward(
                self, handle, _post_forward_reshard, self, unused, output
            )

    @staticmethod
    @contextlib.contextmanager
    def summon_full_params(
        module: nn.Module,
        recurse: bool = True,
        writeback: bool = True,
        rank0_only: bool = False,
        offload_to_cpu: bool = False,
        with_grads: bool = False,
    ) -> Generator:
        r"""Expose full params for FSDP instances with this context manager.

        Can be useful *after* forward/backward for a model to get
        the params for additional processing or checking. It can take a non-FSDP
        module and will summon full params for all contained FSDP modules as
        well as their children, depending on the ``recurse`` argument.

        .. note:: This can be used on inner FSDPs.
        .. note:: This can *not* be used within a forward or backward pass. Nor
            can forward and backward be started from within this context.
        .. note:: Parameters will revert to their local shards after the context
            manager exits, storage behavior is the same as forward.
        .. note:: The full parameters can be modified, but only the portion
            corresponding to the local param shard will persist after the
            context manager exits (unless ``writeback=False``, in which case
            changes will be discarded). In the case where FSDP does not shard
            the parameters, currently only when ``world_size == 1``, or ``NO_SHARD``
            config, the modification is persisted regardless of ``writeback``.
        .. note:: This method works on modules which are not FSDP themselves but
            may contain multiple independent FSDP units. In that case, the given
            arguments will apply to all contained FSDP units.

        .. warning:: Note that ``rank0_only=True`` in conjunction with
            ``writeback=True`` is not currently supported and will raise an
            error. This is because model parameter shapes would be different
            across ranks within the context, and writing to them can lead to
            inconsistency across ranks when the context is exited.

        .. warning:: Note that ``offload_to_cpu`` and ``rank0_only=False`` will
            result in full parameters being redundantly copied to CPU memory for
            GPUs that reside on the same machine, which may incur the risk of
            CPU OOM. It is recommended to use ``offload_to_cpu`` with
            ``rank0_only=True``.

        Args:
            recurse (bool, Optional): recursively summon all params for nested
                FSDP instances (default: True).
            writeback (bool, Optional): if ``False``, modifications to params are
                discarded after the context manager exits;
                disabling this can be slightly more efficient (default: True)
            rank0_only (bool, Optional): if ``True``, full parameters are
                materialized on only global rank 0. This means that within the
                context, only rank 0 will have full parameters and the other
                ranks will have sharded parameters. Note that setting
                ``rank0_only=True`` with ``writeback=True`` is not supported,
                as model parameter shapes will be different across ranks
                within the context, and writing to them can lead to
                inconsistency across ranks when the context is exited.
            offload_to_cpu (bool, Optional): If ``True``, full parameters are
                offloaded to CPU. Note that this offloading currently only
                occurs if the parameter is sharded (which is only not the case
                for world_size = 1 or ``NO_SHARD`` config). It is recommended
                to use ``offload_to_cpu`` with ``rank0_only=True`` to avoid
                redundant copies of model parameters being offloaded to the same CPU memory.
            with_grads (bool, Optional): If ``True``, gradients are also
                unsharded with the parameters. Currently, this is only
                supported when passing ``use_orig_params=True`` to the FSDP
                constructor and ``offload_to_cpu=False`` to this method.
                (Default: ``False``)
        """
        with _unshard_params(
            module, recurse, writeback, rank0_only, offload_to_cpu, with_grads
        ):
            yield

    @contextlib.contextmanager
    def _deregister_orig_params_ctx(self):
        """Deregister the original parameters and expose the :class:`FlatParameter`.

        If a :class:`FlatParameter` is sharded, then
        this refreshes the sharded views before exiting. This method should
        only be called when using the original parameters.
        """
        _p_assert(
            self._use_orig_params,
            "`_deregister_orig_params_ctx()` should only be called when "
            "`_use_orig_params=True`",
        )
        for fsdp_module in traversal_utils._get_fsdp_states(self):
            _deregister_orig_params(fsdp_module, fsdp_module)
        try:
            yield
        finally:
            for fsdp_module in traversal_utils._get_fsdp_states(self):
                _register_orig_params(fsdp_module, fsdp_module)

    @copy_method_sig(nn.Module._apply)
    def _apply(self, *args, **kwargs):
        """Deregister the original parameters and expose the :class:`FlatParameter` s before calling ``_apply()``."""
        # When using the original parameters: Since (1) the `FlatParameter`s
        # own the storage and (2) `_apply()` is the subroutine underlying the
        # most common storage-changing ops like `to()` and `cuda()`, we
        # override `_apply()` to have the storage change directly performed on
        # the `FlatParameter`s instead of applying to the original parameters
        # and then writing back to the `FlatParameter`s.
        context = (
            self._deregister_orig_params_ctx()
            if self._use_orig_params
            else contextlib.nullcontext()
        )
        with context:
            return super()._apply(*args, **kwargs)

    @copy_method_params(nn.Module.named_buffers)
    def named_buffers(
        self,
        *args,
        **kwargs,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Return an iterator over module buffers, yielding both the name of the buffer and the buffer itself.

        Intercepts buffer names and removes all occurrences of the FSDP-specific flattened buffer prefix
        when inside the :meth:`summon_full_params` context manager.
        """
        should_clean_name = self.training_state == TrainingState.SUMMON_FULL_PARAMS
        for buffer_name, buffer in super().named_buffers(*args, **kwargs):
            if should_clean_name:
                # Remove any instances of the FSDP-specific prefix; there can
                # be multiple in the case of nested FSDP modules
                buffer_name = buffer_name.replace(FSDP_PREFIX, "")
            yield (buffer_name, buffer)

    @copy_method_params(nn.Module.named_parameters)
    def named_parameters(
        self,
        *args,
        **kwargs,
    ) -> Iterator[tuple[str, torch.nn.Parameter]]:
        """Return an iterator over module parameters, yielding both the name of the parameter and the parameter itself.

        Intercepts parameter names and removes all occurrences of the FSDP-specific flattened parameter prefix
        when inside the :meth:`summon_full_params` context manager.
        """
        should_clean_name = self.training_state == TrainingState.SUMMON_FULL_PARAMS
        for param_name, param in super().named_parameters(*args, **kwargs):
            if should_clean_name:
                # Remove any instances of the FSDP-specific prefix; there can
                # be multiple in the case of nested FSDP modules
                param_name = param_name.replace(FSDP_PREFIX, "")
            yield (param_name, param)

    def _assert_state(self, state: TrainingState | list[TrainingState]) -> None:
        """Assert we are in the given state."""
        # Since assert can be turned off and this error checking
        # is really important, we use explicit error checking
        # and raise a ValueError if needed.
        if isinstance(state, TrainingState):
            state = [state]
        if self.training_state not in state:
            msg = (
                f"expected to be in states {state} but current state "
                f"is {self.training_state}"
            )
            # In case we are failing in the context of autograd hook, asserting
            # may not generate useful msg. So, let's print it to be sure.
            if self.rank == 0:
                print(f"Asserting FSDP instance is: {self}")
                print(f"ERROR: {msg}")
                traceback.print_stack()
            raise ValueError(msg)

    @contextmanager
    def no_sync(self) -> Generator:
        """Disable gradient synchronizations across FSDP instances.

        Within this context, gradients will be accumulated in module
        variables, which will later be synchronized in the first
        forward-backward pass after exiting the context. This should only be
        used on the root FSDP instance and will recursively apply to all
        children FSDP instances.

        .. note:: This likely results in higher memory usage because FSDP will
            accumulate the full model gradients (instead of gradient shards)
            until the eventual sync.

        .. note:: When used with CPU offloading, the gradients will not be
            offloaded to CPU when inside the context manager. Instead, they
            will only be offloaded right after the eventual sync.
        """
        _lazy_init(self, self)
        if not self._is_root:
            raise RuntimeError(
                "`no_sync()` on inner FSDP instances is not supported. Please call `no_sync()` on root FSDP module."
            )
        self._assert_state(TrainingState.IDLE)
        old_flags = []
        for m in self.modules():
            if isinstance(m, FullyShardedDataParallel):
                old_flags.append((m, m._sync_gradients))
                m._sync_gradients = False
        try:
            yield
        finally:
            for m, old_flag in old_flags:
                if m._sync_gradients:
                    raise AssertionError(
                        "`_sync_gradients` was incorrectly set to "
                        "`True` while in the `no_sync()` context manager"
                    )
                m._sync_gradients = old_flag

    @torch.no_grad()
    def clip_grad_norm_(
        self, max_norm: float | int, norm_type: float | int = 2.0
    ) -> torch.Tensor:
        """Clip the gradient norm of all parameters.

        The norm is computed over all parameters' gradients as viewed as a single vector, and the
        gradients are modified in-place.

        Args:
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'``
                for infinity norm.

        Returns:
            Total norm of the parameters (viewed as a single vector).

        If every FSDP instance uses ``NO_SHARD``, meaning that no
        gradients are sharded across ranks, then you may directly use
        :func:`torch.nn.utils.clip_grad_norm_`.

        If at least some FSDP instance uses a sharded strategy (i.e.
        one other than ``NO_SHARD``), then you should use this method
        instead of :func:`torch.nn.utils.clip_grad_norm_` since this method
        handles the fact that gradients are sharded across ranks.

        The total norm returned will have the "largest" dtype across
        all parameters/gradients as defined by PyTorch's type promotion
        semantics. For example, if *all* parameters/gradients use a low
        precision dtype, then the returned norm's dtype will be that low
        precision dtype, but if there exists at least one parameter/
        gradient using FP32, then the returned norm's dtype will be FP32.

        .. warning:: This needs to be called on all ranks since it uses
            collective communications.
        """
        _lazy_init(self, self)
        if not self._is_root:
            raise RuntimeError(
                "`clip_grad_norm_()` should only be called on the root FSDP instance"
            )
        if self._zero_scalar is None:
            self._zero_scalar = torch.tensor(0.0, device=self.compute_device)
        self._assert_state(TrainingState.IDLE)
        # If every FSDP instance uses `NO_SHARD`, then we can directly use
        # the normal `nn.utils` one targeting local gradients
        all_no_shard = all(
            not handle.uses_sharded_strategy for handle in self._all_handles
        )
        if all_no_shard:
            return torch.nn.utils.clip_grad_norm_(
                self.parameters(), max_norm, norm_type
            )
        # Otherwise, there exists some FSDP instance using a sharded strategy,
        # where sharded and non-sharded parameters must be handled separately
        max_norm = float(max_norm)
        norm_type = float(norm_type)
        sharded_params_set = set()
        nonsharded_params_set = set()  # `NO_SHARD` or not FSDP-managed
        # Make sure to compute the local norm using lists for deterministic
        # iteration order and hence deterministic total norm computation
        sharded_params = []
        nonsharded_params = []
        grads: list[torch.Tensor] = []
        for handle in self._all_handles:
            if handle.uses_sharded_strategy:
                target_set = sharded_params_set
                target_list = sharded_params
            else:
                target_set = nonsharded_params_set
                target_list = nonsharded_params
            if handle._use_orig_params:
                for param in handle.flat_param._params:
                    if param not in target_set:
                        target_set.add(param)
                        target_list.append(param)
                        if param.grad is not None:
                            grads.append(param.grad)
            else:
                if handle.flat_param not in target_set:
                    target_set.add(handle.flat_param)
                    target_list.append(handle.flat_param)
                    if handle.flat_param.grad is not None:
                        grads.append(handle.flat_param.grad)
        for param in self.parameters():
            not_fsdp_managed = (
                param not in sharded_params_set and param not in nonsharded_params_set
            )
            if not_fsdp_managed:
                nonsharded_params_set.add(param)
                nonsharded_params.append(param)
                if param.grad is not None:
                    grads.append(param.grad)
        # Compute local norms (forced to be in FP32)
        local_sharded_norm = _get_grad_norm(
            sharded_params, norm_type, self._zero_scalar, self.compute_device
        )
        local_nonsharded_norm = (
            _get_grad_norm(
                nonsharded_params, norm_type, self._zero_scalar, self.compute_device
            )
            if nonsharded_params
            else None
        )
        # Reconstruct the total gradient norm depending on the norm type
        if norm_type == math.inf:
            total_norm = (
                torch.maximum(local_sharded_norm, local_nonsharded_norm)
                if local_nonsharded_norm is not None
                else local_sharded_norm
            )
            dist.all_reduce(
                total_norm, op=torch.distributed.ReduceOp.MAX, group=self.process_group
            )
        else:
            total_norm = local_sharded_norm**norm_type
            dist.all_reduce(total_norm, group=self.process_group)
            # All-reducing the local non-sharded norm would count it an extra
            # world-size-many times
            if local_nonsharded_norm is not None:
                total_norm += local_nonsharded_norm**norm_type
            total_norm = total_norm ** (1.0 / norm_type)
        if self.cpu_offload.offload_params:
            total_norm = total_norm.cpu()

        clip_coef = max_norm / (total_norm + 1e-6)
        # Multiplying by the clamped coefficient is meaningless when it is
        # equal to 1, but it avoids the host-device sync that would result from
        # `if clip_coef < 1`
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for grad in grads:
            grad.mul_(clip_coef_clamped.to(grad.device, grad.dtype))
        # Use the "largest" dtype by type promotion semantics to use the same
        # dtype as if we did not force local norm computation to be in FP32
        if len(grads) == 0:
            # If this rank has no gradients, then we must default to FP32
            # unless we use additional communication, which we prefer to avoid
            # since `clip_grad_norm_()` is called in the training loop
            warnings.warn(
                f"Called FSDP.clip_grad_norm_() on rank {self.rank} with no "
                "gradients -- returning the total norm in the default dtype "
                f"{total_norm.dtype}",
                stacklevel=2,
            )  # warn since this is generally unexpected
            return total_norm
        total_norm_dtype = functools.reduce(
            torch.promote_types,
            [grad.dtype for grad in grads],
        )
        return total_norm.to(total_norm_dtype)

    @staticmethod
    def _warn_optim_input(optim_input, *, stacklevel: int = 1):
        if optim_input is not None:
            warnings.warn(
                "The `optim_input` argument is deprecated and will be removed after PyTorch 1.13. "
                "You may remove it from your code without changing its functionality.",
                FutureWarning,
                stacklevel=stacklevel + 1,
            )

    @staticmethod
    def _is_using_optim_input(optim_input, optim) -> bool:
        if optim_input is None and optim is None:
            # Use the default behavior of `optim_input``
            return True
        if optim_input is not None:
            # Use the `optim_input` code path
            return True
        # Use the `optim` code path
        return False

    @staticmethod
    def _warn_legacy_optim_state_dict(curr: str, new: str, *, stacklevel: int = 1):
        warnings.warn(
            f"``FullyShardedDataParallel.{curr}``is being deprecated and is "
            f"replaced by ``FullyShardedDataParallel.{new}``. "
            f"``FullyShardedDataParallel.{curr}`` may be removed after PyTorch 2.2.",
            FutureWarning,
            stacklevel=stacklevel + 1,
        )

    @staticmethod
    def _optim_state_dict_impl(
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        optim_state_dict: dict[str, Any],
        optim_input: list[dict[str, Any]] | Iterable[torch.nn.Parameter] | None = None,
        rank0_only: bool = True,
        full_state_dict: bool = True,
        group: dist.ProcessGroup | None = None,
        cpu_offload: bool = True,
        *,
        _stacklevel: int = 1,
    ) -> dict[str, Any]:
        """Transform the state-dict of an optimizer corresponding to a sharded model.

        This is the internal API that is used by all the optim_state_dict implementations.
        Given model, optim, the original optim_state_dict, this API removes the
        FSDP internal information and internal sharding from the optim_state_dict.
        """
        if full_state_dict:
            FullyShardedDataParallel._warn_optim_input(
                optim_input, stacklevel=_stacklevel + 1
            )
            using_optim_input = FullyShardedDataParallel._is_using_optim_input(
                optim_input,
                optim,
            )
        else:
            using_optim_input = False
            if optim_input is not None or rank0_only:
                raise AssertionError(
                    f"Expected optim_input to be None and rank0_only to be False, "
                    f"got optim_input={optim_input}, rank0_only={rank0_only}"
                )

        use_orig_params = FullyShardedDataParallel.fsdp_modules(model)[
            0
        ]._use_orig_params
        if not all(
            use_orig_params == m._use_orig_params
            for m in FullyShardedDataParallel.fsdp_modules(model)
        ):
            raise AssertionError(
                "Not all FSDP modules have the same _use_orig_params value"
            )

        return _optim_state_dict(
            model=model,
            optim=optim,
            optim_state_dict=optim_state_dict,
            optim_input=optim_input,
            rank0_only=rank0_only,
            shard_state=not full_state_dict,
            group=group,
            using_optim_input=using_optim_input,
            use_orig_params=use_orig_params,
            cpu_offload=cpu_offload,
        )

    @staticmethod
    def _optim_state_dict_to_load_impl(
        optim_state_dict: dict[str, Any],
        model: torch.nn.Module,
        optim_input: list[dict[str, Any]] | Iterable[torch.nn.Parameter] | None = None,
        optim: torch.optim.Optimizer | None = None,
        full_state_dict: bool = True,
        rank0_only: bool = False,
        is_named_optimizer: bool = False,
        group: dist.ProcessGroup | None = None,
    ) -> dict[str, Any]:
        """
        Convert an optimizer state-dict so that it can be loaded into the optimizer associated with the FSDP model.

        This is the internal API that is used by all the load optim_state_dict implementations.
        Given model, optim, and the saved optim_state_dict, this API adds the FSDP
        internal information and internal sharding to the optim_state_dict.
        """
        if full_state_dict:
            FullyShardedDataParallel._warn_optim_input(optim_input)
            using_optim_input = FullyShardedDataParallel._is_using_optim_input(
                optim_input,
                optim,
            )
        else:
            using_optim_input = False
            if optim_input is not None or rank0_only:
                raise AssertionError(
                    f"Expected optim_input to be None and rank0_only to be False, "
                    f"got optim_input={optim_input}, rank0_only={rank0_only}"
                )

        use_orig_params = FullyShardedDataParallel.fsdp_modules(model)[
            0
        ]._use_orig_params
        if not all(
            use_orig_params == m._use_orig_params
            for m in FullyShardedDataParallel.fsdp_modules(model)
        ):
            raise AssertionError(
                "Not all FSDP modules have the same _use_orig_params value"
            )

        if rank0_only and dist.get_rank(group) > 0:
            optim_state_dict = {}
        sharded_osd = _flatten_optim_state_dict(
            optim_state_dict,
            model=model,
            use_orig_params=use_orig_params,
            optim=(optim if is_named_optimizer else None),
            rank0_only=rank0_only,
            group=group,
        )
        return _rekey_sharded_optim_state_dict(
            sharded_osd,
            model=model,
            optim=optim,
            optim_input=optim_input,
            using_optim_input=using_optim_input,
            is_named_optimizer=is_named_optimizer,
        )

    @staticmethod
    def full_optim_state_dict(
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        optim_input: list[dict[str, Any]] | Iterable[torch.nn.Parameter] | None = None,
        rank0_only: bool = True,
        group: dist.ProcessGroup | None = None,
    ) -> dict[str, Any]:
        """Return the full optimizer state-dict.

        Consolidates the full optimizer state on rank 0 and returns it
        as a :class:`dict` following the convention of
        :meth:`torch.optim.Optimizer.state_dict`, i.e. with keys ``"state"``
        and ``"param_groups"``. The flattened parameters in ``FSDP`` modules
        contained in ``model`` are mapped back to their unflattened parameters.

        This needs to be called on all ranks since it uses
        collective communications. However, if ``rank0_only=True``, then
        the state dict is only populated on rank 0, and all other ranks
        return an empty :class:`dict`.

        Unlike ``torch.optim.Optimizer.state_dict()``, this method
        uses full parameter names as keys instead of parameter IDs.

        Like in :meth:`torch.optim.Optimizer.state_dict`, the tensors
        contained in the optimizer state dict are not cloned, so there may
        be aliasing surprises. For best practices, consider saving the
        returned optimizer state dict immediately, e.g. using
        ``torch.save()``.

        Args:
            model (torch.nn.Module): Root module (which may or may not be a
                :class:`FullyShardedDataParallel` instance) whose parameters
                were passed into the optimizer ``optim``.
            optim (torch.optim.Optimizer): Optimizer for ``model`` 's
                parameters.
            optim_input (Optional[Union[List[Dict[str, Any]], Iterable[torch.nn.Parameter]]]):
                Input passed into the optimizer ``optim`` representing either a
                :class:`list` of parameter groups or an iterable of parameters;
                if ``None``, then this method assumes the input was
                ``model.parameters()``. This argument is deprecated, and there
                is no need to pass it in anymore. (Default: ``None``)
            rank0_only (bool): If ``True``, saves the populated :class:`dict`
                only on rank 0; if ``False``, saves it on all ranks. (Default:
                ``True``)
            group (dist.ProcessGroup): Model's process group or ``None`` if using
                the default process group. (Default: ``None``)

        Returns:
            Dict[str, Any]: A :class:`dict` containing the optimizer state for
            ``model`` 's original unflattened parameters and including keys
            "state" and "param_groups" following the convention of
            :meth:`torch.optim.Optimizer.state_dict`. If ``rank0_only=True``,
            then nonzero ranks return an empty :class:`dict`.
        """
        FullyShardedDataParallel._warn_legacy_optim_state_dict(
            "full_optim_state_dict",
            "optim_state_dict",
            stacklevel=2,
        )
        return FullyShardedDataParallel._optim_state_dict_impl(
            model=model,
            optim=optim,
            optim_state_dict=optim.state_dict(),
            optim_input=optim_input,
            rank0_only=rank0_only,
            group=group,
            full_state_dict=True,
            _stacklevel=2,
        )

    @staticmethod
    def sharded_optim_state_dict(
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        group: dist.ProcessGroup | None = None,
    ) -> dict[str, Any]:
        """Return the optimizer state-dict in its sharded form.

        The API is similar to :meth:`full_optim_state_dict` but this API chunks
        all non-zero-dimension states to :class:`ShardedTensor` to save memory.
        This API should only be used when the model ``state_dict`` is derived
        with the context manager ``with state_dict_type(SHARDED_STATE_DICT):``.

        For the detailed usage, refer to :meth:`full_optim_state_dict`.

        .. warning:: The returned state dict contains ``ShardedTensor`` and
            cannot be directly used by the regular ``optim.load_state_dict``.
        """
        FullyShardedDataParallel._warn_legacy_optim_state_dict(
            "sharded_optim_state_dict",
            "optim_state_dict",
            stacklevel=2,
        )
        return FullyShardedDataParallel._optim_state_dict_impl(
            model=model,
            optim=optim,
            optim_state_dict=optim.state_dict(),
            optim_input=None,
            rank0_only=False,
            full_state_dict=False,
            group=group,
            _stacklevel=2,
        )

    @staticmethod
    def shard_full_optim_state_dict(
        full_optim_state_dict: dict[str, Any],
        model: torch.nn.Module,
        optim_input: list[dict[str, Any]] | Iterable[torch.nn.Parameter] | None = None,
        optim: torch.optim.Optimizer | None = None,
    ) -> dict[str, Any]:
        """Shard a full optimizer state-dict.

        Remaps the state in ``full_optim_state_dict`` to flattened parameters instead of unflattened
        parameters and restricts to only this rank's part of the optimizer state.
        The first argument should be the return value of :meth:`full_optim_state_dict`.

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            >>> model, optim = ...
            >>> full_osd = FSDP.full_optim_state_dict(model, optim)
            >>> torch.save(full_osd, PATH)
            >>> # Define new model with possibly different world size
            >>> new_model, new_optim = ...
            >>> full_osd = torch.load(PATH)
            >>> sharded_osd = FSDP.shard_full_optim_state_dict(full_osd, new_model)
            >>> new_optim.load_state_dict(sharded_osd)

        .. note:: Both :meth:`shard_full_optim_state_dict` and
            :meth:`scatter_full_optim_state_dict` may be used to get the
            sharded optimizer state dict to load. Assuming that the full
            optimizer state dict resides in CPU memory, the former requires
            each rank to have the full dict in CPU memory, where each rank
            individually shards the dict without any communication, while the
            latter requires only rank 0 to have the full dict in CPU memory,
            where rank 0 moves each shard to GPU memory (for NCCL) and
            communicates it to ranks appropriately. Hence, the former has
            higher aggregate CPU memory cost, while the latter has higher
            communication cost.

        Args:
            full_optim_state_dict (Dict[str, Any]): Optimizer state dict
                corresponding to the unflattened parameters and holding the
                full non-sharded optimizer state.
            model (torch.nn.Module): Root module (which may or may not be a
                :class:`FullyShardedDataParallel` instance) whose parameters
                correspond to the optimizer state in ``full_optim_state_dict``.
            optim_input (Optional[Union[List[Dict[str, Any]], Iterable[torch.nn.Parameter]]]):
                Input passed into the optimizer representing either a
                :class:`list` of parameter groups or an iterable of parameters;
                if ``None``, then this method assumes the input was
                ``model.parameters()``. This argument is deprecated, and there
                is no need to pass it in anymore. (Default: ``None``)
            optim (Optional[torch.optim.Optimizer]): Optimizer that will load
                the state dict returned by this method. This is the preferred
                argument to use over ``optim_input``. (Default: ``None``)

        Returns:
            Dict[str, Any]: The full optimizer state dict now remapped to
            flattened parameters instead of unflattened parameters and
            restricted to only include this rank's part of the optimizer state.
        """
        FullyShardedDataParallel._warn_legacy_optim_state_dict(
            "shard_full_optim_state_dict",
            "optim_state_dict_to_load",
            stacklevel=2,
        )
        return FullyShardedDataParallel._optim_state_dict_to_load_impl(
            optim_state_dict=full_optim_state_dict,
            model=model,
            optim_input=optim_input,
            optim=optim,
            full_state_dict=True,
            is_named_optimizer=False,
        )

    @staticmethod
    def flatten_sharded_optim_state_dict(
        sharded_optim_state_dict: dict[str, Any],
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
    ) -> dict[str, Any]:
        """Flatten a sharded optimizer state-dict.

        The API is similar to :meth:`shard_full_optim_state_dict`. The only
        difference is that the input ``sharded_optim_state_dict`` should be
        returned from :meth:`sharded_optim_state_dict`. Therefore, there will
        be all-gather calls on each rank to gather ``ShardedTensor`` s.

        Args:
            sharded_optim_state_dict (Dict[str, Any]): Optimizer state dict
                corresponding to the unflattened parameters and holding the
                sharded optimizer state.
            model (torch.nn.Module):
                Refer to :meth:`shard_full_optim_state_dict`.
            optim (torch.optim.Optimizer): Optimizer for ``model`` 's
                parameters.

        Returns:
            Refer to :meth:`shard_full_optim_state_dict`.
        """
        FullyShardedDataParallel._warn_legacy_optim_state_dict(
            "flatten_sharded_optim_state_dict",
            "optim_state_dict_to_load",
            stacklevel=2,
        )
        return FullyShardedDataParallel._optim_state_dict_to_load_impl(
            optim_state_dict=sharded_optim_state_dict,
            model=model,
            optim_input=None,
            optim=optim,
            full_state_dict=False,
            is_named_optimizer=False,
        )

    @staticmethod
    def scatter_full_optim_state_dict(
        full_optim_state_dict: dict[str, Any] | None,
        model: torch.nn.Module,
        optim_input: list[dict[str, Any]] | Iterable[torch.nn.Parameter] | None = None,
        optim: torch.optim.Optimizer | None = None,
        group: Any | None = None,
    ) -> dict[str, Any]:
        """Scatter the full optimizer state dict from rank 0 to all other ranks.

        Returns the sharded optimizer state dict on each rank.
        The return value is the same as :meth:`shard_full_optim_state_dict`, and on rank
        0, the first argument should be the return value of
        :meth:`full_optim_state_dict`.

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            >>> model, optim = ...
            >>> full_osd = FSDP.full_optim_state_dict(model, optim)  # only non-empty on rank 0
            >>> # Define new model with possibly different world size
            >>> new_model, new_optim, new_group = ...
            >>> sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, new_model, group=new_group)
            >>> new_optim.load_state_dict(sharded_osd)

        .. note:: Both :meth:`shard_full_optim_state_dict` and
            :meth:`scatter_full_optim_state_dict` may be used to get the
            sharded optimizer state dict to load. Assuming that the full
            optimizer state dict resides in CPU memory, the former requires
            each rank to have the full dict in CPU memory, where each rank
            individually shards the dict without any communication, while the
            latter requires only rank 0 to have the full dict in CPU memory,
            where rank 0 moves each shard to GPU memory (for NCCL) and
            communicates it to ranks appropriately. Hence, the former has
            higher aggregate CPU memory cost, while the latter has higher
            communication cost.

        Args:
            full_optim_state_dict (Optional[Dict[str, Any]]): Optimizer state
                dict corresponding to the unflattened parameters and holding
                the full non-sharded optimizer state if on rank 0; the argument
                is ignored on nonzero ranks.
            model (torch.nn.Module): Root module (which may or may not be a
                :class:`FullyShardedDataParallel` instance) whose parameters
                correspond to the optimizer state in ``full_optim_state_dict``.
            optim_input (Optional[Union[List[Dict[str, Any]], Iterable[torch.nn.Parameter]]]):
                Input passed into the optimizer representing either a
                :class:`list` of parameter groups or an iterable of parameters;
                if ``None``, then this method assumes the input was
                ``model.parameters()``. This argument is deprecated, and there
                is no need to pass it in anymore. (Default: ``None``)
            optim (Optional[torch.optim.Optimizer]): Optimizer that will load
                the state dict returned by this method. This is the preferred
                argument to use over ``optim_input``. (Default: ``None``)
            group (dist.ProcessGroup): Model's process group or ``None`` if
                using the default process group. (Default: ``None``)

        Returns:
            Dict[str, Any]: The full optimizer state dict now remapped to
            flattened parameters instead of unflattened parameters and
            restricted to only include this rank's part of the optimizer state.
        """
        FullyShardedDataParallel._warn_legacy_optim_state_dict(
            "scatter_full_optim_state_dict",
            "optim_state_dict_to_load",
            stacklevel=2,
        )
        return FullyShardedDataParallel._optim_state_dict_to_load_impl(
            optim_state_dict=full_optim_state_dict,
            model=model,
            optim_input=optim_input,
            optim=optim,
            full_state_dict=True,
            rank0_only=True,
            is_named_optimizer=False,
            group=group,
        )

    @staticmethod
    def rekey_optim_state_dict(
        optim_state_dict: dict[str, Any],
        optim_state_key_type: OptimStateKeyType,
        model: torch.nn.Module,
        optim_input: list[dict[str, Any]] | Iterable[torch.nn.Parameter] | None = None,
        optim: torch.optim.Optimizer | None = None,
    ) -> dict[str, Any]:
        """Re-keys the optimizer state dict ``optim_state_dict`` to use the key type ``optim_state_key_type``.

        This can be used to achieve compatibility between optimizer state dicts from models with FSDP
        instances and ones without.

        To re-key an FSDP full optimizer state dict (i.e. from
        :meth:`full_optim_state_dict`) to use parameter IDs and be loadable to
        a non-wrapped model::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> wrapped_model, wrapped_optim = ...
            >>> full_osd = FSDP.full_optim_state_dict(wrapped_model, wrapped_optim)
            >>> nonwrapped_model, nonwrapped_optim = ...
            >>> rekeyed_osd = FSDP.rekey_optim_state_dict(full_osd, OptimStateKeyType.PARAM_ID, nonwrapped_model)
            >>> nonwrapped_optim.load_state_dict(rekeyed_osd)

        To re-key a normal optimizer state dict from a non-wrapped model to be
        loadable to a wrapped model::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> nonwrapped_model, nonwrapped_optim = ...
            >>> osd = nonwrapped_optim.state_dict()
            >>> rekeyed_osd = FSDP.rekey_optim_state_dict(osd, OptimStateKeyType.PARAM_NAME, nonwrapped_model)
            >>> wrapped_model, wrapped_optim = ...
            >>> sharded_osd = FSDP.shard_full_optim_state_dict(rekeyed_osd, wrapped_model)
            >>> wrapped_optim.load_state_dict(sharded_osd)

        Returns:
            Dict[str, Any]: The optimizer state dict re-keyed using the
            parameter keys specified by ``optim_state_key_type``.
        """
        FullyShardedDataParallel._warn_optim_input(optim_input)
        using_optim_input = FullyShardedDataParallel._is_using_optim_input(
            optim_input,
            optim,
        )
        if optim_state_key_type not in (
            OptimStateKeyType.PARAM_NAME,
            OptimStateKeyType.PARAM_ID,
        ):
            raise AssertionError(
                f"Expected optim_state_key_type to be PARAM_NAME or PARAM_ID, got {optim_state_key_type}"
            )
        osd = optim_state_dict  # alias
        # Validate that the existing parameter keys are uniformly typed
        uses_param_name_mask = [type(param_key) is str for param_key in osd["state"]]
        uses_param_id_mask = [type(param_key) is int for param_key in osd["state"]]
        if (any(uses_param_name_mask) and not all(uses_param_name_mask)) or (
            any(uses_param_id_mask) and not all(uses_param_id_mask)
        ):
            error_msg = f"Invalid parameter keys: {osd['state'].keys()}"
            raise ValueError(error_msg)
        # Return directly if the existing key type matches the target key type
        if (
            optim_state_key_type == OptimStateKeyType.PARAM_NAME
            and all(uses_param_name_mask)
        ) or (
            optim_state_key_type == OptimStateKeyType.PARAM_ID
            and all(uses_param_id_mask)
        ):
            return osd
        # Otherwise, actually perform the re-keying
        new_osd = {}
        if optim_state_key_type == OptimStateKeyType.PARAM_NAME:  # ID -> name
            param_id_to_param = (
                _get_param_id_to_param_from_optim_input(model, optim_input)
                if using_optim_input
                else _get_param_key_to_param(optim)
            )
            param_to_param_name = _get_param_to_fqn(model)
            param_id_to_param_name: list[str] = [
                param_to_param_name[param] for param in param_id_to_param.values()
            ]
            new_osd["state"] = {
                param_id_to_param_name[param_id]: param_state
                for param_id, param_state in osd["state"].items()
            }
            new_osd["param_groups"] = copy.deepcopy(osd["param_groups"])
            for param_group in new_osd["param_groups"]:
                param_group["params"] = sorted(
                    [
                        param_id_to_param_name[param_id]
                        for param_id in param_group["params"]
                    ]
                )
            return new_osd
        elif optim_state_key_type == OptimStateKeyType.PARAM_ID:  # name -> ID
            param_name_to_param = _get_fqn_to_param(model)
            param_to_param_id = (
                _get_param_to_param_id_from_optim_input(model, optim_input)
                if using_optim_input
                else _get_param_to_param_key(optim)
            )
            # Because not all model parameters may be passed as the optimizer
            # input, we may need to drop some parameters from this mapping
            param_name_to_param_id = {
                param_name: param_to_param_id[param]
                for param_name, param in param_name_to_param.items()
                if param in param_to_param_id
            }
            new_osd["state"] = {
                param_name_to_param_id[param_name]: param_state
                for param_name, param_state in osd["state"].items()
            }
            new_osd["param_groups"] = copy.deepcopy(osd["param_groups"])
            for param_group in new_osd["param_groups"]:
                param_group["params"] = sorted(
                    [
                        param_name_to_param_id[param_name]
                        for param_name in param_group["params"]
                    ]
                )
            return new_osd
        return new_osd  # should never reach here

    @staticmethod
    def optim_state_dict(
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        optim_state_dict: dict[str, Any] | None = None,
        group: dist.ProcessGroup | None = None,
    ) -> dict[str, Any]:
        """
        Transform the state-dict of an optimizer corresponding to a sharded model.

        The given state-dict can be transformed to one of three types:
        1) full optimizer state_dict, 2) sharded optimizer state_dict, 3) local optimizer state_dict.

        For full optimizer state_dict, all states are unflattened and not sharded.
        Rank0 only and CPU only can be specified via :meth:`state_dict_type` to
        avoid OOM.

        For sharded optimizer state_dict, all states are unflattened but sharded.
        CPU only can be specified via :meth:`state_dict_type` to further save
        memory.

        For local state_dict, no transformation will be performed. But a state
        will be converted from nn.Tensor to ShardedTensor to represent its sharding
        nature (this is not supported yet).

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            >>> from torch.distributed.fsdp import StateDictType
            >>> from torch.distributed.fsdp import FullStateDictConfig
            >>> from torch.distributed.fsdp import FullOptimStateDictConfig
            >>> # Save a checkpoint
            >>> model, optim = ...
            >>> FSDP.set_state_dict_type(
            >>>     model,
            >>>     StateDictType.FULL_STATE_DICT,
            >>>     FullStateDictConfig(rank0_only=False),
            >>>     FullOptimStateDictConfig(rank0_only=False),
            >>> )
            >>> state_dict = model.state_dict()
            >>> optim_state_dict = FSDP.optim_state_dict(model, optim)
            >>> save_a_checkpoint(state_dict, optim_state_dict)
            >>> # Load a checkpoint
            >>> model, optim = ...
            >>> state_dict, optim_state_dict = load_a_checkpoint()
            >>> FSDP.set_state_dict_type(
            >>>     model,
            >>>     StateDictType.FULL_STATE_DICT,
            >>>     FullStateDictConfig(rank0_only=False),
            >>>     FullOptimStateDictConfig(rank0_only=False),
            >>> )
            >>> model.load_state_dict(state_dict)
            >>> optim_state_dict = FSDP.optim_state_dict_to_load(
            >>>     model, optim, optim_state_dict
            >>> )
            >>> optim.load_state_dict(optim_state_dict)

        Args:
            model (torch.nn.Module): Root module (which may or may not be a
                :class:`FullyShardedDataParallel` instance) whose parameters
                were passed into the optimizer ``optim``.
            optim (torch.optim.Optimizer): Optimizer for ``model`` 's
                parameters.
            optim_state_dict (Dict[str, Any]): the target optimizer state_dict to
                transform. If the value is None, optim.state_dict() will be used. (
                Default: ``None``)
            group (dist.ProcessGroup): Model's process group across which parameters
                are sharded or ``None`` if using the default process group. (
                Default: ``None``)

        Returns:
            Dict[str, Any]: A :class:`dict` containing the optimizer state for
            ``model``. The sharding of the optimizer state is based on
            ``state_dict_type``.
        """
        state_dict_settings = FullyShardedDataParallel.get_state_dict_type(model)
        if optim_state_dict is None:
            optim_state_dict = optim.state_dict()
        return FullyShardedDataParallel._optim_state_dict_impl(
            model=model,
            optim=optim,
            optim_state_dict=optim_state_dict,
            optim_input=None,
            rank0_only=getattr(
                state_dict_settings.optim_state_dict_config, "rank0_only", False
            ),
            full_state_dict=state_dict_settings.state_dict_type
            == StateDictType.FULL_STATE_DICT,
            group=group,
            cpu_offload=getattr(
                state_dict_settings.optim_state_dict_config, "offload_to_cpu", True
            ),
            _stacklevel=2,
        )

    @staticmethod
    def optim_state_dict_to_load(
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        optim_state_dict: dict[str, Any],
        is_named_optimizer: bool = False,
        load_directly: bool = False,
        group: dist.ProcessGroup | None = None,
    ) -> dict[str, Any]:
        """
        Convert an optimizer state-dict so that it can be loaded into the optimizer associated with the FSDP model.

        Given a ``optim_state_dict`` that is transformed through
        :meth:`optim_state_dict`, it gets converted to the flattened optimizer
        state_dict that can be loaded to ``optim`` which is the optimizer for
        ``model``. ``model`` must be sharded by FullyShardedDataParallel.

            >>> # xdoctest: +SKIP("undefined variables")
            >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            >>> from torch.distributed.fsdp import StateDictType
            >>> from torch.distributed.fsdp import FullStateDictConfig
            >>> from torch.distributed.fsdp import FullOptimStateDictConfig
            >>> # Save a checkpoint
            >>> model, optim = ...
            >>> FSDP.set_state_dict_type(
            >>>     model,
            >>>     StateDictType.FULL_STATE_DICT,
            >>>     FullStateDictConfig(rank0_only=False),
            >>>     FullOptimStateDictConfig(rank0_only=False),
            >>> )
            >>> state_dict = model.state_dict()
            >>> original_osd = optim.state_dict()
            >>> optim_state_dict = FSDP.optim_state_dict(
            >>>     model,
            >>>     optim,
            >>>     optim_state_dict=original_osd
            >>> )
            >>> save_a_checkpoint(state_dict, optim_state_dict)
            >>> # Load a checkpoint
            >>> model, optim = ...
            >>> state_dict, optim_state_dict = load_a_checkpoint()
            >>> FSDP.set_state_dict_type(
            >>>     model,
            >>>     StateDictType.FULL_STATE_DICT,
            >>>     FullStateDictConfig(rank0_only=False),
            >>>     FullOptimStateDictConfig(rank0_only=False),
            >>> )
            >>> model.load_state_dict(state_dict)
            >>> optim_state_dict = FSDP.optim_state_dict_to_load(
            >>>     model, optim, optim_state_dict
            >>> )
            >>> optim.load_state_dict(optim_state_dict)

        Args:
            model (torch.nn.Module): Root module (which may or may not be a
                :class:`FullyShardedDataParallel` instance) whose parameters
                were passed into the optimizer ``optim``.
            optim (torch.optim.Optimizer): Optimizer for ``model`` 's
                parameters.
            optim_state_dict (Dict[str, Any]): The optimizer states to be loaded.
            is_named_optimizer (bool): Is this optimizer a NamedOptimizer or
                KeyedOptimizer. Only set to True if ``optim`` is TorchRec's
                KeyedOptimizer or torch.distributed's NamedOptimizer.
            load_directly (bool): If this is set to True, this API will also
                call optim.load_state_dict(result) before returning the result.
                Otherwise, users are responsible to call ``optim.load_state_dict()``
                (Default: ``False``)
            group (dist.ProcessGroup): Model's process group across which parameters
                are sharded or ``None`` if using the default process group. (
                Default: ``None``)
        """
        state_dict_settings = FullyShardedDataParallel.get_state_dict_type(model)
        result = FullyShardedDataParallel._optim_state_dict_to_load_impl(
            optim_state_dict=optim_state_dict,
            model=model,
            optim_input=None,
            optim=optim,
            full_state_dict=(
                state_dict_settings.state_dict_type == StateDictType.FULL_STATE_DICT
            ),
            rank0_only=getattr(
                state_dict_settings.optim_state_dict_config, "rank0_only", False
            ),
            is_named_optimizer=is_named_optimizer,
            group=group,
        )
        if load_directly:
            optim.load_state_dict(result)
        return result

    def register_comm_hook(self, state: object, hook: callable) -> None:
        """Register a communication hook.

        This is an enhancement that provides a flexible hook to users where they can specify how FSDP aggregates
        gradients across multiple workers.
        This hook can be used to implement several algorithms like
        `GossipGrad <https://arxiv.org/abs/1803.05880>`_ and gradient compression
        which involve different communication strategies for
        parameter syncs while training with :class:`FullyShardedDataParallel`.

        .. warning ::
            FSDP communication hook should be registered before running an initial forward pass
            and only once.

        Args:
            state (object): Passed to the hook to maintain any state information during the training process.
                            Examples include error feedback in gradient compression,
                            peers to communicate with next in `GossipGrad <https://arxiv.org/abs/1803.05880>`_, etc.
                            It is locally stored by each worker
                            and shared by all the gradient tensors on the worker.
            hook (Callable): Callable, which has one of the following signatures:
                            1) ``hook: Callable[torch.Tensor] -> None``:
                            This function takes in a Python tensor, which represents
                            the full, flattened, unsharded gradient with respect to all variables
                            corresponding to the model this FSDP unit is wrapping
                            (that are not wrapped by other FSDP sub-units).
                            It then performs all necessary processing and returns ``None``;
                            2) ``hook: Callable[torch.Tensor, torch.Tensor] -> None``:
                            This function takes in two Python tensors, the first one represents
                            the full, flattened, unsharded gradient with respect to all variables
                            corresponding to the model this FSDP unit is wrapping
                            (that are not wrapped by other FSDP sub-units). The latter
                            represents a pre-sized tensor to store a chunk of a sharded gradient after
                            reduction.
                            In both cases, callable performs all necessary processing and returns ``None``.
                            Callables with signature 1 are expected to handle gradient communication for a `NO_SHARD` case.
                            Callables with signature 2 are expected to handle gradient communication for sharded cases.

        """
        if not self.check_is_root():
            raise AssertionError(
                "register_comm_hook can only be called on a root instance."
            )
        for fsdp_state in traversal_utils._get_fsdp_states(self):
            if fsdp_state.sharding_strategy in HYBRID_SHARDING_STRATEGIES:
                raise AssertionError(
                    f"Communication hook is not supported for hybrid strategies: {fsdp_state.sharding_strategy}"
                )
            if fsdp_state._comm_hook is not None:
                raise AssertionError("A communication hook is already registered")
            if not callable(hook):
                raise ValueError(
                    f"The communication hook must be callable but got {hook}"
                )
            fsdp_state._comm_hook = hook
            fsdp_state._comm_hook_state = state

    def _unshard(self, async_op: bool = False):
        class UnshardHandle:
            def __init__(
                self,
                flat_param_handle: FlatParamHandle | None,
                unshard_event: torch.Event,
            ):
                self._flat_param_handle = flat_param_handle
                self._unshard_event = unshard_event

            def wait(self):
                if self._flat_param_handle is not None:
                    current_stream = (
                        self._flat_param_handle._device_handle.current_stream()
                    )
                    current_stream.wait_event(self._unshard_event)
                    self._flat_param_handle = None

        if self._handle:
            with self._use_training_state(
                TrainingState.FORWARD_BACKWARD, HandleTrainingState.FORWARD
            ):
                _unshard(
                    self, self._handle, self._unshard_stream, self._pre_unshard_stream
                )
                self._unshard_event = self._unshard_stream.record_event()
            self._handle._prefetched = True
        unshard_handle = UnshardHandle(self._handle, self._unshard_event)
        if async_op:
            return unshard_handle
        unshard_handle.wait()
        return None

    def _wait_unshard_streams_on_current_stream(self):
        _wait_for_computation_stream(
            self._device_handle.current_stream(),
            self._unshard_stream,
            self._pre_unshard_stream,
        )

    @contextlib.contextmanager
    def _use_training_state(
        self, training_state: TrainingState, handle_training_state: HandleTrainingState
    ):
        prev_training_state = self.training_state
        self.training_state = training_state
        if self._handle:
            prev_handle_training_state = self._handle._training_state
            self._handle._training_state = handle_training_state
        try:
            yield
        finally:
            self.training_state = prev_training_state
            if self._handle:
                self._handle._training_state = prev_handle_training_state


def _get_grad_norm(
    params: Iterable[nn.Parameter],
    norm_type: float,
    zero: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Return the gradient norm of parameters ``param`` s, where the gradients are viewed as a single vector.

    The returned norm is in FP32 even if parameters/gradients are in a low precision. This is because the downstream
    use of this return value is a reduction across ranks.
    """
    params_with_grad = [param for param in params if param.grad is not None]
    if len(params_with_grad) == 0:
        # Reuse a tensor for zero to avoid a GPU sync
        return zero
    grads = [param.grad for param in params_with_grad]
    # Compute the gradient norm in FP32, where we treat the gradients as a
    # single vector. This naturally handles mixed-dtype gradients since we
    # cast each gradient to FP32 during the norm computation.
    grad_norm = torch.linalg.vector_norm(
        torch.stack(
            [
                torch.linalg.vector_norm(grad.detach(), norm_type, dtype=torch.float32)
                for grad in grads
            ],
        ),
        norm_type,
        dtype=torch.float32,
    )
    return grad_norm.to(device=device)


def _get_param_to_fqn(
    model: torch.nn.Module,
) -> dict[torch.nn.Parameter, str]:
    """
    Construct a mapping from parameters to their parameter names.

    The ``model`` should not contain any :class:`FullyShardedDataParallel` instances, which
    means that none of the parameters should be ``FlatParameter`` s. As a
    result, compared to :meth:`_get_param_to_fqns`, the mapped
    values may be flattened from singleton :class:`list` s to the contained
    names themselves.

    Args:
        model (torch.nn.Module): Root module, which should not contain any
            :class:`FullyShardedDataParallel` instances.
    """
    param_to_param_names = _get_param_to_fqns(model)
    for param_names in param_to_param_names.values():
        if len(param_names) == 0:
            raise AssertionError(
                "`_get_param_to_fqns()` should not construct empty lists"
            )
        if len(param_names) > 1:
            raise RuntimeError(
                "Each parameter should only map to one parameter name but got "
                f"{len(param_names)}: {param_names}"
            )
    param_to_param_name = {
        param: param_names[0] for param, param_names in param_to_param_names.items()
    }
    return param_to_param_name


def _get_fqn_to_param(
    model: torch.nn.Module,
) -> dict[str, torch.nn.Parameter]:
    """Construct the inverse mapping of :meth:`_get_param_to_fqn`."""
    param_to_param_name = _get_param_to_fqn(model)
    return dict(zip(param_to_param_name.values(), param_to_param_name.keys()))
<|endoftext|># mypy: allow-untyped-defs
import functools
import math
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import auto, Enum
from typing import Any, TypeVar
from typing_extensions import ParamSpec

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.contract import _get_registry
from torch.distributed.tensor import DeviceMesh, DTensor, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec

from ._fsdp_api import DataParallelMeshDims


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _dynamo_disable(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Disable dynamo tracing for FSDP hooks."""

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs):
        return torch._dynamo.disable(
            func, recursive=True, reason="skipping FSDP hooks"
        )(*args, **kwargs)

    return wrapper


def _disable_functorch_if_active(func: Callable[_P, _R]) -> Callable[_P, _R]:
    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if torch._C._are_functorch_transforms_active():
            with torch._C._DisableFuncTorch():
                return func(*args, **kwargs)
        return func(*args, **kwargs)

    return wrapper


@dataclass
class DataParallelMeshInfo:
    mesh: DeviceMesh
    shard_mesh_dim: int | None = None
    replicate_mesh_dim: int | None = None
    dp_mesh_dims: DataParallelMeshDims | None = None
    # The full SPMD mesh (excluding PP dims) that params are distributed on.
    # Must include all non-PP SPMD dims (e.g. DP + TP); passing a submesh
    # that omits dims like TP will lead to incorrect behavior.
    spmd_mesh: DeviceMesh | None = field(default=None, repr=False)
    is_spmd_mesh: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if self.shard_mesh_dim is None and self.replicate_mesh_dim is None:
            raise AssertionError(
                "At least one of shard_mesh_dim and replicate_mesh_dim must not be None"
            )
        self.is_spmd_mesh = self.dp_mesh_dims is not None


@dataclass
class FSDPMeshInfo(DataParallelMeshInfo):
    def __post_init__(self):
        super().__post_init__()
        if self.shard_mesh_dim is None:
            raise AssertionError("Expects non-None shard_mesh_dim")
        self.shard_mesh_size: int = self.mesh.size(self.shard_mesh_dim)
        self.shard_process_group = self.mesh.get_group(self.shard_mesh_dim)
        self.shard_mesh_rank: int = self.shard_process_group.rank()
        # Reduce-scatter shares the shard PG (one NCCL communicator) by default;
        # FSDPModule.set_separate_reduce_scatter_group assigns a dedicated PG
        # here so reduce-scatter can overlap all-gather in backward.
        self.reduce_scatter_process_group: dist.ProcessGroup | None = None


@dataclass
class DDPMeshInfo(DataParallelMeshInfo):
    def __post_init__(self):
        super().__post_init__()
        if self.replicate_mesh_dim is None:
            raise AssertionError("Expects non-None replicate_mesh_dim")
        self.replicate_mesh_size: int = self.mesh.size(self.replicate_mesh_dim)
        self.replicate_process_group = self.mesh.get_group(self.replicate_mesh_dim)
        self.replicate_mesh_rank: int = self.replicate_process_group.rank()


@dataclass
class HSDPMeshInfo(FSDPMeshInfo, DDPMeshInfo):
    def __post_init__(self):  # pylint:disable=useless-parent-delegation
        # Calls `FSDPMeshInfo` -> `DDPMeshInfo` -> `DataParallelMeshInfo`
        super().__post_init__()


class TrainingState(Enum):
    """Describes the training state of one FSDP state / parameter group."""

    # Transition to forward starting pre-forward until post-forward
    FORWARD = auto()
    # Transition to pre-backward when unsharding in backward
    PRE_BACKWARD = auto()
    # Transition to post-backward when resharding and reducing gradients
    POST_BACKWARD = auto()
    # Idle before/after forward or before pre-backward/after post-backward
    IDLE = auto()


def _raise_assert_with_print(*args: Any, **kwargs: Any):
    print(f"[Rank {dist.get_rank()}] ", end="")
    print(*args, **kwargs)
    traceback.print_stack()
    raise AssertionError(*args, **kwargs)


def _is_composable_with_fsdp(module: nn.Module) -> bool:
    registry = _get_registry(module)
    if registry is None:
        return True
    # Registry keys by function name
    return "replicate" not in registry


def _get_dim0_padded_size(tensor_size: torch.Size, dim0_factor: int) -> torch.Size:
    padded_dim0 = math.ceil(tensor_size[0] / dim0_factor) * dim0_factor
    return torch.Size([padded_dim0]) + tensor_size[1:]


def _chunk_with_empty(
    tensor: torch.Tensor, num_chunks: int, dim: int
) -> list[torch.Tensor]:
    chunks = list(torch.chunk(tensor, num_chunks, dim=dim))
    while len(chunks) < num_chunks:
        chunks.append(chunks[0].new_empty(0))
    return chunks


def _get_dim_chunked_size(
    chunk: torch.Tensor, unchunked_size: torch.Size, dim: int
) -> torch.Size:
    if chunk.numel() > 0:
        return chunk.size()
    # For 0 numel, we need to preserve nonzero-sized dims for DTensor APIs
    # pyrefly: ignore [bad-return]
    return unchunked_size[:dim] + torch.Size([0]) + unchunked_size[dim + 1 :]


def _from_local_no_grad(
    local_tensor: torch.Tensor,
    sharding_spec: DTensorSpec,
) -> DTensor:
    """
    This method is similar to ``DTensor.from_local()`` except that in eager mode
    it avoids some CPU overhead by avoiding default args and not being differentiable.
    """
    # pyrefly: ignore [bad-argument-type]
    return DTensor(
        # Use the local tensor directly instead of constructing a new tensor
        # variable, e.g. with `view_as()`, since this is not differentiable
        # pyrefly: ignore [bad-argument-count]
        local_tensor,
        sharding_spec,
        # pyrefly: ignore [unexpected-keyword]
        requires_grad=local_tensor.requires_grad,
    )


def _to_dtype_if_needed(
    tensor: torch.Tensor, dtype: torch.dtype | None
) -> torch.Tensor:
    if dtype is not None and tensor.dtype != dtype:
        return tensor.to(dtype)
    return tensor


def _cast_fp_tensor(dtype: torch.dtype, x: torch.Tensor) -> torch.Tensor:
    if (
        not isinstance(x, torch.Tensor)
        or not torch.is_floating_point(x)
        or x.dtype == dtype
    ):
        return x
    return x.to(dtype)


def is_bw() -> bool:
    return torch._C._current_graph_task_id() != -1


@dataclass
class ShardPlacementResult:
    placement: Shard | None
    mesh_info: FSDPMeshInfo


ShardPlacementFnResult = Shard | ShardPlacementResult | None


def resolve_shard_placement(
    result: ShardPlacementFnResult,
    default_mesh_info: FSDPMeshInfo,
) -> ShardPlacementResult:
    """Resolve the shard_placement_fn result to a ShardPlacementResult.

    Handles different input types and applies defaults:
    - None: Use default sharding (Shard(0)) on default mesh
    - Shard: Use specified shard dimension on default mesh
    - ShardPlacementResult: Use as-is

    Args:
        result: The return value from shard_placement_fn, or None if no fn provided.
        default_mesh_info: The default FSDPMeshInfo to use if not specified.

    Returns:
        A ShardPlacementResult with placement and mesh_info.
    """
    if result is None:
        return ShardPlacementResult(placement=None, mesh_info=default_mesh_info)
    if isinstance(result, Shard):
        return ShardPlacementResult(placement=result, mesh_info=default_mesh_info)
    if isinstance(result, ShardPlacementResult):
        return result
    raise ValueError(f"Invalid shard_placement_fn result: {result}")
<|endoftext|>import math
from collections.abc import Callable, Sequence
from itertools import chain
from typing import Any, cast, Literal, NamedTuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.distributed_c10d import ReduceOp
from torch.distributed.fsdp._fully_shard._fsdp_api import AllGather, ReduceScatter
from torch.distributed.tensor import DTensor

from ._fsdp_api import _ReduceOp
from ._fsdp_common import (
    _get_dim0_padded_size,
    _raise_assert_with_print,
    _to_dtype_if_needed,
)
from ._fsdp_param import FSDPParam, ShardedState


class AllGatherResult(NamedTuple):
    all_gather_output: torch.Tensor
    all_gather_event: torch.Event | None
    all_gather_work: dist.distributed_c10d.Work | None
    # For each parameter, the all-gather input dtype for each input
    param_all_gather_input_dtypes: list[list[torch.dtype]]
    # For each parameter, the all-gather input numel for each input
    param_all_gather_input_numels: list[list[int]]
    # 1D flattened version of `param_all_gather_input_numels` saved to avoid
    # CPU overhead from recomputing
    all_gather_input_split_sizes: list[int]


lib = torch.library.Library("fsdp", "FRAGMENT")

lib.define(
    """
    all_gather_copy_in(
        Tensor[] all_gather_inputs,
        Tensor all_gather_output,
        SymInt[] inp_split_sizes,
        SymInt all_gather_input_numel,
        SymInt rank
    ) -> (Tensor, Tensor)
    """
)


class DefaultAllocMixin:
    def allocate(
        self,
        size: Sequence[int | torch.SymInt],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.empty(*size, dtype=dtype, device=device)


class ProcessGroupAllocMixin:
    def __init__(self, group: dist.ProcessGroup, *args: Any, **kwargs: Any):
        self._group = group
        super().__init__(*args, **kwargs)

    def allocate(
        self,
        size: Sequence[int | torch.SymInt],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        backend = self._group._get_backend(device)
        if backend.supports_tensor_alloc(device):
            size_1d = math.prod(int(s) for s in size)
            return backend.allocate_tensor(size_1d, dtype=dtype, device=device)
        return torch.empty(*size, dtype=dtype, device=device)


class SymmMemAllocMixin:
    def __init__(
        self,
        group: dist.ProcessGroup,
        backend: Literal["NCCL"] = "NCCL",
        *args: Any,
        **kwargs: Any,
    ):
        self._group = group
        symm_mem.set_backend(backend)
        # Force initialization of communicator; otherwise, the rendezvous may
        # see empty communicator.
        # TODO: Remove this, maybe by warning user to perform eager dist init.
        # For now, it is okay since it is just a one-time cost at init.
        dist.barrier(group=group)

    def allocate(
        self,
        size: Sequence[int | torch.SymInt],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        # Leverage MemPool to reuse the symmetric buffer, avoiding allocation
        # and rendezvous overhead
        mempool = symm_mem.get_mem_pool(device)
        with torch.cuda.use_mem_pool(mempool):
            return torch.empty(size, dtype=dtype, device=device)


class DefaultAllGather(DefaultAllocMixin, AllGather):
    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = False,
    ) -> dist.Work | None:
        return dist.all_gather_single(
            output_tensor,
            input_tensor,
            group=group,
            async_op=async_op,
        )


class ProcessGroupAllocAllGather(ProcessGroupAllocMixin, AllGather):
    def __init__(self, group: dist.ProcessGroup) -> None:
        super().__init__(group)

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = False,
    ) -> dist.Work | None:
        return dist.all_gather_single(
            output_tensor,
            input_tensor,
            group=group,
            async_op=async_op,
        )


class SymmMemAllGather(SymmMemAllocMixin, AllGather):
    def __init__(
        self,
        group: dist.ProcessGroup,
        backend: Literal["NCCL"] = "NCCL",
    ) -> None:
        super().__init__(group, backend)

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = False,
    ) -> dist.Work | None:
        # We are doing inplace all-gather, so we need to rendezvous the output tensor only
        symm_mem.rendezvous(output_tensor, group=group.group_name)
        # Calling regular all-gather would already cause libraries like NCCL to
        # use its optimized all-gather implementation for symmetric memory:
        # - Copy Engine All-Gather (when zero-CTA policy is enabled)
        # - Symmetric Kernel All-Gather (when zero-CTA policy is not enabled)
        return dist.all_gather_single(
            output_tensor,
            input_tensor,
            group=group,
            async_op=async_op,
        )


class DefaultReduceScatter(DefaultAllocMixin, ReduceScatter):
    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        op: _ReduceOp,
        async_op: bool = False,
    ) -> dist.Work | None:
        return dist.reduce_scatter_single(
            output=output_tensor,
            input=input_tensor,
            group=group,
            op=op,
            async_op=async_op,
        )


class ProcessGroupAllocReduceScatter(ProcessGroupAllocMixin, ReduceScatter):
    def __init__(self, group: dist.ProcessGroup) -> None:
        super().__init__(group)

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        op: _ReduceOp,
        async_op: bool = False,
    ) -> dist.Work | None:
        return dist.reduce_scatter_single(
            output=output_tensor,
            input=input_tensor,
            group=group,
            op=op,
            async_op=async_op,
        )


class SymmMemReduceScatter(SymmMemAllocMixin, ReduceScatter):
    def __init__(
        self,
        group: dist.ProcessGroup,
        backend: Literal["NCCL"] = "NCCL",
    ) -> None:
        super().__init__(group, backend)

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        op: _ReduceOp,
        async_op: bool = False,
    ) -> dist.Work | None:
        symm_mem.rendezvous(input_tensor, group=group.group_name)
        symm_mem.rendezvous(output_tensor, group=group.group_name)
        # Calling regular reduce-scatter would already cause libraries like NCCL to
        # use its optimized reduce-scatter implementation for symmetric memory
        return dist.reduce_scatter_single(
            output=output_tensor,
            input=input_tensor,
            group=group,
            op=op,
            async_op=async_op,
        )


@torch.library.impl(lib, "all_gather_copy_in", "Meta")
def all_gather_copy_in_meta(
    all_gather_inputs: list[torch.Tensor],
    all_gather_output: torch.Tensor,
    inp_split_sizes: list[int],
    all_gather_input_numel: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_gather_input = all_gather_output.narrow(
        0, all_gather_input_numel * rank, all_gather_input_numel
    )
    return all_gather_input, all_gather_output


@torch.library.impl(lib, "all_gather_copy_in", "CUDA")
@torch.library.impl(lib, "all_gather_copy_in", "XPU")
@torch.library.impl(lib, "all_gather_copy_in", "HPU")
@torch.library.impl(lib, "all_gather_copy_in", "CPU")
@torch.library.impl(lib, "all_gather_copy_in", "MTIA")
@torch.library.impl(lib, "all_gather_copy_in", "PrivateUse1")
def all_gather_copy_in_cuda(
    all_gather_inputs: list[torch.Tensor],
    all_gather_output: torch.Tensor,
    inp_split_sizes: list[int],
    all_gather_input_numel: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_gather_input = all_gather_output.narrow(
        0, all_gather_input_numel * rank, all_gather_input_numel
    )
    foreach_copy_dsts = torch.split(all_gather_input, inp_split_sizes)
    with torch.no_grad():
        torch._foreach_copy_(foreach_copy_dsts, all_gather_inputs)
    return all_gather_input, all_gather_output


lib.define(
    "split_with_sizes_copy(Tensor all_gather_output, SymInt[] all_gather_input_split_sizes, int dim=0, *, Tensor(a!)[] out) -> ()"
)


@torch.library.impl(lib, "split_with_sizes_copy", "Meta")
@torch.library.impl(lib, "split_with_sizes_copy", "CUDA")
@torch.library.impl(lib, "split_with_sizes_copy", "XPU")
@torch.library.impl(lib, "split_with_sizes_copy", "HPU")
@torch.library.impl(lib, "split_with_sizes_copy", "CPU")
@torch.library.impl(lib, "split_with_sizes_copy", "MTIA")
@torch.library.impl(lib, "split_with_sizes_copy", "PrivateUse1")
def split_with_sizes_copy(
    all_gather_output: torch.Tensor,
    all_gather_input_split_sizes: list[int],
    dim: int = 0,
    *,
    out: list[torch.Tensor],
) -> None:
    torch.split_with_sizes_copy(
        all_gather_output, all_gather_input_split_sizes, dim=dim, out=out
    )


lib.define(
    "chunk_cat(Tensor[] tensors, int dim, int num_chunks, *, Tensor(a!) out) -> ()"
)


@torch.library.impl(lib, "chunk_cat", "Meta")
@torch.library.impl(lib, "chunk_cat", "CUDA")
@torch.library.impl(lib, "chunk_cat", "XPU")
@torch.library.impl(lib, "chunk_cat", "HPU")
@torch.library.impl(lib, "chunk_cat", "CPU")
@torch.library.impl(lib, "chunk_cat", "MTIA")
@torch.library.impl(lib, "chunk_cat", "PrivateUse1")
def chunk_cat(
    tensors: list[torch.Tensor],
    dim: int,
    num_chunks: int,
    out: torch.Tensor,
) -> None:
    torch._chunk_cat(tensors, dim, num_chunks, out=out)


@torch.no_grad()
def foreach_all_gather(
    fsdp_params: list[FSDPParam],
    group: dist.ProcessGroup,
    async_op: bool,
    all_gather_copy_in_stream: torch.Stream,
    all_gather_stream: torch.Stream,
    device: torch.device,
    all_gather_comm: AllGather,
) -> AllGatherResult | None:
    world_size, rank = group.size(), group.rank()
    device_handle = _get_device_handle(device.type)
    with device_handle.stream(all_gather_copy_in_stream):
        param_all_gather_inputs = _get_param_all_gather_inputs(fsdp_params)
        (
            param_all_gather_input_dtypes,
            param_all_gather_input_numels,
            dtype,
        ) = _get_all_gather_input_metadatas(param_all_gather_inputs)
        if dtype == torch.uint8:
            all_gather_inputs = [
                t.view(torch.uint8) for ts in param_all_gather_inputs for t in ts
            ]
        else:
            all_gather_inputs = [*chain.from_iterable(param_all_gather_inputs)]
        inp_split_sizes = [t.numel() for t in all_gather_inputs]
        all_gather_input_numel = sum(inp_split_sizes)
        all_gather_output = all_gather_comm.allocate(
            (all_gather_input_numel * world_size,), dtype=dtype, device=device
        )
        all_gather_input, all_gather_output = torch.ops.fsdp.all_gather_copy_in(
            all_gather_inputs,
            all_gather_output,
            inp_split_sizes,
            all_gather_input_numel,
            rank,
        )
        del param_all_gather_inputs
    all_gather_stream.wait_stream(all_gather_copy_in_stream)
    with device_handle.stream(all_gather_stream):
        all_gather_work = all_gather_comm(
            output_tensor=all_gather_output,
            input_tensor=all_gather_input,
            group=group,
            async_op=async_op,
        )
        all_gather_event = all_gather_stream.record_event()
        return AllGatherResult(
            all_gather_output,
            all_gather_event,
            all_gather_work,
            param_all_gather_input_dtypes,
            param_all_gather_input_numels,
            inp_split_sizes,
        )


@torch.no_grad()
def _get_param_all_gather_inputs(
    fsdp_params: list[FSDPParam],
) -> list[list[torch.Tensor]]:
    # Intentionally try to run a fast-path that bypasses abstractions for the
    # common FSDP case of bf16/fp32 mixed precision in order to use foreach
    # copy for lower CPU overhead and more efficient copying in eager
    def use_foreach_copy(fsdp_param: FSDPParam) -> bool:
        return (
            fsdp_param.param_dtype is not None
            and not fsdp_param.offload_to_cpu
            and not hasattr(fsdp_param._sharded_local_tensor, "fsdp_pre_all_gather")
        )

    param_all_gather_inputs: list[list[torch.Tensor]] = [[] for _ in fsdp_params]
    foreach_copy_indices: list[int] = []
    foreach_copy_inputs: list[torch.Tensor] = []
    foreach_copy_input_numels: list[int] = []

    # 1st pass: for foreach-copy parameters, get inputs and metadata for the
    # foreach copy, and for the others, actually get their all-gather inputs
    for i, fsdp_param in enumerate(fsdp_params):
        if use_foreach_copy(fsdp_param):
            foreach_copy_indices.append(i)
            all_gather_input = (
                fsdp_param._sharded_param_data
                if fsdp_param.sharded_state == ShardedState.SHARDED
                else cast(torch.Tensor, fsdp_param._sharded_post_forward_param_data)
            )
            foreach_copy_inputs.append(all_gather_input)
            foreach_copy_input_numels.append(all_gather_input.numel())
        else:
            param_all_gather_inputs[i] = fsdp_param.all_gather_inputs

    # 2nd pass: use foreach copy to compute the remaining all-gather inputs
    if foreach_copy_inputs:
        fsdp_param_0 = fsdp_params[foreach_copy_indices[0]]
        param_dtype, device = fsdp_param_0.param_dtype, fsdp_param_0.device
        flat_foreach_copy_input = torch.empty(
            (sum(foreach_copy_input_numels),), device=device, dtype=param_dtype
        )
        splits = torch.split(flat_foreach_copy_input, foreach_copy_input_numels)
        torch._foreach_copy_(splits, foreach_copy_inputs)
        for i, split in zip(foreach_copy_indices, splits):
            param_all_gather_inputs[i] = [split]

    return param_all_gather_inputs


@torch.no_grad()
def foreach_all_gather_copy_out(
    all_gather_result: AllGatherResult,
    fsdp_params: list[FSDPParam],
    group: dist.ProcessGroup,
) -> None:
    (
        all_gather_output,
        all_gather_event,
        all_gather_work,
        param_all_gather_input_dtypes,
        param_all_gather_input_numels,
        all_gather_input_split_sizes,
    ) = all_gather_result
    _dtype, device = all_gather_output.dtype, all_gather_output.device
    device_handle = _get_device_handle(device.type)
    if all_gather_event is not None:  # sync op
        device_handle.current_stream().wait_event(all_gather_event)
    if isinstance(all_gather_work, dist.distributed_c10d.Work):  # async op
        all_gather_work.wait()
    world_size, device = group.size(), all_gather_output.device

    split_with_sizes_out: list[torch.Tensor] = []
    shard_i_copy_infos: list[tuple[FSDPParam, list[torch.Tensor]]] = []
    for all_gather_input_numels, all_gather_input_dtypes, fsdp_param in zip(
        param_all_gather_input_numels, param_all_gather_input_dtypes, fsdp_params
    ):
        fsdp_param.init_all_gather_outputs(
            all_gather_input_numels,
            all_gather_input_dtypes,
            world_size,
            device,
        )
        fsdp_param.alloc_all_gather_outputs()
        param_all_gather_outputs = fsdp_param.all_gather_outputs
        if fsdp_param.fsdp_placement.dim != 0:
            # Copy to a temporary and then chunk-cat into the final all-gather
            # output tensors
            param_all_gather_outputs = [
                torch.empty_like(t) for t in param_all_gather_outputs
            ]
            shard_i_copy_infos.append((fsdp_param, param_all_gather_outputs))
        split_with_sizes_out.extend(param_all_gather_outputs)

    all_gather_output = all_gather_output.view(world_size, -1)
    if all_gather_output.dtype == torch.uint8:
        out = [t.view(world_size, -1).view(torch.uint8) for t in split_with_sizes_out]
    else:
        out = [t.view(world_size, -1) for t in split_with_sizes_out]

    # only avoid VC bump if we are not in inference mode
    non_inference_outs = [o for o in out if not o.is_inference()]

    if len(non_inference_outs) > 0:
        with torch.autograd._unsafe_preserve_version_counter(tuple(non_inference_outs)):
            torch.ops.fsdp.split_with_sizes_copy(
                all_gather_output, all_gather_input_split_sizes, dim=1, out=out
            )
    else:
        torch.ops.fsdp.split_with_sizes_copy(
            all_gather_output, all_gather_input_split_sizes, dim=1, out=out
        )

    for fsdp_param, param_all_gather_outputs in shard_i_copy_infos:
        # Chunk-cat from the temporary to the final all-gather output tensors
        shard_dim = fsdp_param.fsdp_placement.dim

        with torch.autograd._unsafe_preserve_version_counter(
            tuple(fsdp_param.all_gather_outputs)
        ):
            for param_all_gather_output, target_all_gather_output in zip(
                param_all_gather_outputs, fsdp_param.all_gather_outputs
            ):
                padded_sharded_size = (
                    fsdp_param.padded_sharded_param_size
                    if fsdp_param.sharded_state == ShardedState.SHARDED
                    else cast(
                        torch.Tensor, fsdp_param._sharded_post_forward_param_data
                    ).size()
                )
                pre_param_size = list(padded_sharded_size)
                pre_param_size[0] *= world_size
                chunks = torch.chunk(
                    param_all_gather_output.view(pre_param_size), world_size, dim=0
                )
                post_param_size = list(padded_sharded_size)
                post_param_size[shard_dim] *= world_size
                cat_out = target_all_gather_output.view(post_param_size)
                torch.cat(chunks, dim=shard_dim, out=cat_out)


@torch.no_grad()
def foreach_reduce(
    fsdp_params: list[FSDPParam],
    unsharded_grads: list[torch.Tensor],
    reduce_scatter_group: dist.ProcessGroup,
    reduce_scatter_stream: torch.Stream,
    reduce_scatter_comm: ReduceScatter,
    orig_dtype: torch.dtype | None,
    reduce_dtype: torch.dtype | None,
    device: torch.device,
    gradient_divide_factor: float | None,
    all_reduce_group: dist.ProcessGroup | None,  # not `None` iff HSDP
    all_reduce_stream: torch.Stream,
    all_reduce_grads: bool,
    partial_reduce_output: torch.Tensor | None,  # only used for HSDP
    all_reduce_hook: Callable[[torch.Tensor], None] | None,
    force_sum_reduction_for_comms: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Event,
    torch.Stream,
    torch.Event,
    torch.Tensor | None,
    torch.Event | None,
    torch.Tensor | None,
]:
    """
    ``unsharded_grads`` owns the references to the gradients computed by
    autograd, so clearing the list frees the gradients.
    """

    grad_dtypes = {grad.dtype for grad in unsharded_grads}
    if len(grad_dtypes) != 1:
        # Check this at runtime since it could be a real runtime error if e.g.
        # fp8 weights do not produce the correct higher precision gradients
        _raise_assert_with_print(
            f"FSDP reduce-scatter expects uniform gradient dtype but got {grad_dtypes}"
        )
    grad_dtype = unsharded_grads[0].dtype
    reduce_dtype = reduce_dtype or grad_dtype
    (predivide_factor, postdivide_factor, reduce_scatter_op, all_reduce_op) = (
        _get_gradient_divide_factors(
            reduce_scatter_group,
            all_reduce_group,
            reduce_dtype,
            device.type,
            gradient_divide_factor,
            force_sum_reduction_for_comms,
        )
    )

    if reduce_scatter_group is None:
        world_size = 1
    else:
        world_size = reduce_scatter_group.size()
    device_handle = _get_device_handle(device.type)
    current_stream = device_handle.current_stream()

    if world_size > 1:
        for i, (fsdp_param, unsharded_grad) in enumerate(
            zip(fsdp_params, unsharded_grads)
        ):
            if (shard_dim := fsdp_param.fsdp_placement.dim) == 0:
                continue
            if unsharded_grad.size(shard_dim) % world_size != 0:
                raise AssertionError(
                    f"Shard({shard_dim}) requires even sharding: {unsharded_grad.size()=} {world_size=}"
                )
            chunks = torch.chunk(unsharded_grad, world_size, dim=shard_dim)
            unsharded_grads[i] = torch.cat(chunks, dim=0)

    padded_unsharded_sizes = tuple(
        _get_dim0_padded_size(grad.size(), world_size) for grad in unsharded_grads
    )
    reduce_scatter_input_numel = sum(s.numel() for s in padded_unsharded_sizes)
    reduce_scatter_output_numel = reduce_scatter_input_numel // world_size
    reduce_scatter_input = reduce_scatter_comm.allocate(
        (reduce_scatter_input_numel,),
        dtype=reduce_dtype,
        device=device,
    )

    foreach_reduce_scatter_copy_in(unsharded_grads, reduce_scatter_input, world_size)

    # Only after the copy-in finishes can we free the gradients
    unsharded_grads.clear()
    reduce_scatter_stream.wait_stream(current_stream)
    all_reduce_input = None
    all_reduce_event = None

    with device_handle.stream(reduce_scatter_stream):
        reduce_output = reduce_scatter_comm.allocate(
            (reduce_scatter_output_numel,),
            dtype=reduce_dtype,
            device=device,
        )
        _div_if_needed(reduce_scatter_input, predivide_factor)
        if world_size > 1:
            reduce_scatter_comm(
                output_tensor=reduce_output,
                input_tensor=reduce_scatter_input,
                group=reduce_scatter_group,
                op=reduce_scatter_op,
            )
        else:
            # For single GPU, just copy the input to output (no actual reduce-scatter needed), and
            # account for a possible gradient_divide_factor.
            if gradient_divide_factor is not None:
                reduce_output.copy_(reduce_scatter_input / gradient_divide_factor)
            else:
                reduce_output.copy_(reduce_scatter_input)
        reduce_scatter_event = reduce_scatter_stream.record_event()
        post_reduce_stream = reduce_scatter_stream
        if all_reduce_group is not None:  # HSDP or DDP/replicate
            # Accumulations must run in the reduce-scatter stream
            if not all_reduce_grads:
                if partial_reduce_output is not None:
                    partial_reduce_output += reduce_output
                else:
                    partial_reduce_output = reduce_output
                return (
                    reduce_scatter_input,
                    reduce_scatter_event,
                    post_reduce_stream,
                    post_reduce_stream.record_event(),
                    all_reduce_input,
                    all_reduce_event,
                    partial_reduce_output,
                )
            if partial_reduce_output is not None:
                reduce_output += partial_reduce_output
            post_reduce_stream = all_reduce_stream
            if world_size >= 1:
                all_reduce_stream.wait_stream(reduce_scatter_stream)
            else:
                all_reduce_stream.wait_stream(current_stream)
            with device_handle.stream(all_reduce_stream):
                dist.all_reduce(
                    reduce_output,
                    group=all_reduce_group,
                    op=all_reduce_op,
                )
                # Keep refs to the reduce-dtype AR buffer + completion
                # event so FSDPParamGroup._all_reduce_state can hold them
                # across layers. This keeps the buffer off the caching
                # allocator's free list; otherwise the next layer's
                # reduce-scatter can reuse the same physical block while
                # this layer's AR is still in flight, causing cross-layer
                # gradient aliasing under slow AR. See PR #140044,
                # regression test PR #180900.
                all_reduce_input = reduce_output
                all_reduce_event = all_reduce_stream.record_event()
    # -- END: ops in reduce_scatter stream

    if all_reduce_hook is not None:
        # Execute user-specified all reduce hook.
        # If native HSDP is used, this is executed after the HSDP all reduce.
        # If 1-d FSDP is used, this is executed post reduce-scatter.
        post_reduce_stream = all_reduce_stream
        all_reduce_stream.wait_stream(reduce_scatter_stream)
        with device_handle.stream(all_reduce_stream):
            all_reduce_hook(reduce_output)
    # -- END: ops post reduce_scatter

    with device_handle.stream(post_reduce_stream):
        _div_if_needed(reduce_output, postdivide_factor)
        # Rebinds to a new orig_dtype tensor when reduce_dtype !=
        # orig_dtype. Do NOT rely on this stream-scoped rebind to manage
        # the old reduce-dtype buffer's lifetime: the rebind orders the
        # cast before the free-event on AR stream, but the freed block
        # lands on the caching allocator's free list and the next layer's
        # RS on RS stream can reuse it without waiting for this layer's
        # AR to finish. The reduce-dtype buffer is held across layers by
        # FSDPParamGroup._all_reduce_state (captured above) to prevent
        # this. See PR #140044, regression test PR #180900.
        reduce_output = _to_dtype_if_needed(reduce_output, orig_dtype)
        # View out and accumulate sharded gradients
        flat_grad_offset = 0  # [0, reduce_scatter_output_numel - 1]
        for padded_unsharded_size, fsdp_param in zip(
            padded_unsharded_sizes, fsdp_params
        ):
            # Assume even sharding for Shard(i), i > 0; otherwise would require
            # copy-out for contiguous strides
            new_sharded_grad = torch.as_strided(
                reduce_output,
                size=fsdp_param.sharded_size,
                stride=fsdp_param.contiguous_sharded_stride,
                storage_offset=flat_grad_offset,
            )
            to_accumulate_grad = fsdp_param.sharded_param.grad is not None
            if fsdp_param.offload_to_cpu:
                # Only overlap the D2H copy (copying to pinned memory) when no
                # in-backward CPU consumer of the grad exists. Two such
                # consumers suppress the overlap:
                #   - Accumulating grads: the CPU add kernel depends on the
                #     copy result and we cannot run the add as a callback.
                #   - Post-accumulate-grad hooks: user code (e.g.
                #     optimizer-in-backward) reads ``param.grad`` on CPU
                #     synchronously. With ``non_blocking=True`` the hook would
                #     observe in-flight pinned memory — silently wrong
                #     optimizer updates.
                has_post_acc_grad_hook = bool(
                    getattr(
                        fsdp_param.sharded_param,
                        "_post_accumulate_grad_hooks",
                        None,
                    )
                )
                non_blocking = (
                    fsdp_param.pin_memory
                    and not to_accumulate_grad
                    and not has_post_acc_grad_hook
                )
                # Since the GPU sharded gradient is allocated in the RS stream,
                # we can free it here by not keeping a ref without waiting for
                # the D2H copy since future RS-stream ops run after the copy
                new_sharded_grad = new_sharded_grad.to(
                    torch.device("cpu"), non_blocking=non_blocking
                )
                if non_blocking:
                    # Record an event on which to block the CPU thread to
                    # ensure that the D2H copy finishes before the optimizer
                    fsdp_param.grad_offload_event = post_reduce_stream.record_event()
            if to_accumulate_grad:
                if not isinstance(fsdp_param.sharded_param.grad, DTensor):
                    raise AssertionError(
                        f"Expected fsdp_param.sharded_param.grad to be DTensor, got {type(fsdp_param.sharded_param.grad)}"
                    )
                fsdp_param.sharded_param.grad._local_tensor += new_sharded_grad
            else:
                new_sharded_dtensor_grad = fsdp_param.to_sharded_dtensor(
                    new_sharded_grad
                )
                fsdp_param.sharded_param.grad = new_sharded_dtensor_grad
            for hook in (
                getattr(fsdp_param.sharded_param, "_post_accumulate_grad_hooks", {})
                or {}
            ).values():
                hook(fsdp_param.sharded_param)
            padded_sharded_numel = padded_unsharded_size.numel() // world_size
            flat_grad_offset += padded_sharded_numel
        post_reduce_event = post_reduce_stream.record_event()
    # The RS output is allocated in the RS stream and used in the default
    # stream (for optimizer). To ensure its memory is not reused for later
    # RSs, we do not need extra synchronization since the sharded parameters
    # hold refs through the end of backward.
    return (
        reduce_scatter_input,
        reduce_scatter_event,
        post_reduce_stream,
        post_reduce_event,
        all_reduce_input,
        all_reduce_event,
        None,
    )


def foreach_reduce_scatter_copy_in(
    unsharded_grads: list[torch.Tensor],
    reduce_scatter_input: torch.Tensor,
    world_size: int,
) -> None:
    reduce_scatter_input = reduce_scatter_input.view(world_size, -1)
    torch.ops.fsdp.chunk_cat(
        unsharded_grads, dim=0, num_chunks=world_size, out=reduce_scatter_input
    )


def _get_all_gather_input_metadatas(
    param_all_gather_inputs: list[list[torch.Tensor]],
) -> tuple[list[list[torch.dtype]], list[list[int]], torch.dtype]:
    param_all_gather_input_dtypes: list[list[torch.dtype]] = []
    param_all_gather_input_numels: list[list[int]] = []
    all_gather_dtype = param_all_gather_inputs[0][0].dtype
    for all_gather_inputs in param_all_gather_inputs:
        input_dtypes: list[torch.dtype] = []
        input_numels: list[int] = []
        for all_gather_input in all_gather_inputs:
            if all_gather_input.dtype != all_gather_dtype:
                all_gather_dtype = torch.uint8
            input_dtypes.append(all_gather_input.dtype)
            input_numels.append(all_gather_input.numel())
        param_all_gather_input_dtypes.append(input_dtypes)
        param_all_gather_input_numels.append(input_numels)
    return (
        param_all_gather_input_dtypes,
        param_all_gather_input_numels,
        all_gather_dtype,
    )


def _get_gradient_divide_factors(
    reduce_scatter_group: dist.ProcessGroup | None,
    all_reduce_group: dist.ProcessGroup | None,
    reduce_dtype: torch.dtype,
    device_type: str = "",
    factor: float | None = None,
    force_sum_reduction_for_comms: bool = False,
) -> tuple[
    float | None,
    float | None,
    dist.ReduceOp | dist.ReduceOp.RedOpType,
    dist.ReduceOp | dist.ReduceOp.RedOpType,
]:
    # MTIA appears to only support SUM reduction, hence we force it implicitly
    if device_type == "mtia":
        force_sum_reduction_for_comms = True

    # For fp32/bf16, we do not need to worry about overflow/underflow, so we
    # use NCCL's built-in division to avoid separate div kernels
    overflow_risk = reduce_dtype not in (torch.float32, torch.bfloat16)
    if reduce_scatter_group is not None:
        data_parallel_size = reduce_scatter_group.size()
    else:
        data_parallel_size = 1

    if all_reduce_group is not None:
        data_parallel_size *= all_reduce_group.size()

    if not overflow_risk and not force_sum_reduction_for_comms:
        if factor is None:
            # Warning: NCCL ReduceOp.AVG may produce incorrect results with
            # world size 1.
            if data_parallel_size == 1:
                return None, None, ReduceOp.SUM, ReduceOp.SUM
            return None, None, ReduceOp.AVG, ReduceOp.AVG
        if reduce_scatter_group is not None and factor == reduce_scatter_group.size():
            reduce_scatter_op = ReduceOp.AVG
        else:
            reduce_scatter_op = ReduceOp.PREMUL_SUM(1 / factor)
        return None, None, reduce_scatter_op, ReduceOp.SUM

    if factor is None:
        factor = float(data_parallel_size)
    pre_factor: float | None
    if overflow_risk:
        # Since fp16 has smaller dynamic range than fp32/bf16, we want to avoid
        # overflow/underflow. For N data parallel workers, each worker computes
        # g_i, and they collectively reduce (g_1 + ... + g_N) / N. To avoid
        # overflow/underflow, we divide by ~sqrt(N) before/after the reduction.
        pre_factor = 1
        while factor % pre_factor == 0 and factor / pre_factor > pre_factor:
            pre_factor *= 2
        post_factor = factor / pre_factor
    else:
        # Prefer post-multiplying as it operates on less data and is thus faster
        pre_factor, post_factor = None, factor

    return pre_factor, post_factor, ReduceOp.SUM, ReduceOp.SUM


def _div_if_needed(tensor: torch.Tensor, div_factor: float | None) -> None:
    if div_factor is not None and div_factor != 1:
        tensor.div_(div_factor)
<|endoftext|># mypy: allow-untyped-defs
import inspect
import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from enum import auto, Enum
from typing import Any, cast, TYPE_CHECKING


if TYPE_CHECKING:
    from ._fsdp_api import DataParallelMeshDims

import torch
import torch.distributed as dist
import torch.nn as nn


if dist._is_spmd_types_available():
    import spmd_types as spmd
    from spmd_types._mesh_axis import flatten_axes
    from spmd_types.runtime import get_partition_spec
    from spmd_types.types import partition_spec_to_shard_types

from torch._prims_common import make_contiguous_strides_for
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp._fully_shard._fsdp_common import DDPMeshInfo
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor.placement_types import _StridedShard, Placement

from ._fsdp_api import CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy
from ._fsdp_common import (
    _chunk_with_empty,
    _from_local_no_grad,
    _get_dim_chunked_size,
    _raise_assert_with_print,
    _to_dtype_if_needed,
    DataParallelMeshInfo,
    FSDPMeshInfo,
    HSDPMeshInfo,
    resolve_shard_placement,
    ShardPlacementFnResult,
)


_orig_param_uid_counter = itertools.count()


def _get_orig_param_uid(param: nn.Parameter) -> int:
    if not hasattr(param, "_fsdp_orig_uid"):
        uid = next(_orig_param_uid_counter)
        param._fsdp_orig_uid = uid  # pyrefly: ignore[missing-attribute]
    return param._fsdp_orig_uid  # pyrefly: ignore[missing-attribute]


def _spans_same_mesh(lhs_axes: tuple[Any, ...], rhs_axes: tuple[Any, ...]) -> bool:
    """Return whether two axis sets span the same normalized rank mesh."""
    lhs_mesh = spmd.normalize_mesh(frozenset(lhs_axes))
    rhs_mesh = spmd.normalize_mesh(frozenset(rhs_axes))
    if not lhs_mesh or not rhs_mesh:
        return lhs_mesh == rhs_mesh
    return flatten_axes(tuple(lhs_mesh)) == flatten_axes(tuple(rhs_mesh))


"""
[Note: FSDP tensors]
FSDP considers the following tensors:
- Original parameter: parameter passed to :class:`FSDPParam`, i.e. the one
  on the module when applying FSDP
- Sharded parameter: sharding the original parameter on dim-0 (or a
  user-specified dim) as a DTensor over the main mesh
- All-gather inputs: the ``torch.Tensor`` or ``Tensor`` s passed to all-gather,
  derived from the sharded parameter
- All-gather output: the ``torch.Tensor`` or ``Tensor`` s resulting from
  all-gathering the all-gather inputs
- Unsharded parameter: parameter used for forward/backward computation, derived
  from the all-gather output; autograd leaf

We define these tensors to describe the general framework that can accommodate
extensions, where:
- all-gather-inputs = pre-all-gather-transform(sharded-parameter)
- unsharded-parameter = post-all-gather-transform(all-gather-outputs)

For the default ``torch.Tensor`` case, there is only one all-gather input, and
it shares the same underlying tensor data as the sharded parameter, meaning
that they can be thought of as the same tensors. The same applies for the
all-gather output and unsharded parameter. For non-``torch.Tensor`` extensions,
these equivalences may no longer hold due to the pre/post-all-gather
transforms, and some may have multiple all-gather inputs/outputs (e.g.
quantized data and scales).

[Note: FSDP and autograd]
FSDP dynamically frees and allocates the unsharded parameter. Since autograd
can pack a reference to it or a view to save for backward, we use storage
resizing to implement the freeing/allocation since that preserves the aliasing.
This implies that we construct the unsharded parameter object once and write to
it in-place thereafter. For the default ``torch.Tensor` original parameter
case, the all-gather output and unsharded parameter share the same
data, so we use storage resizing on the all-gather output.

[Note: FSDP and spmd_types]
Refer to https://github.com/meta-pytorch/spmd_types/blob/main/docs/local_spmd_types.md
"""

lib = torch.library.Library("fsdp", "FRAGMENT")

lib.define("copy_(Tensor(a!) tensor, Tensor data) -> ()")


@torch.library.impl(lib, "copy_", "Meta")
@torch.library.impl(lib, "copy_", "CUDA")
@torch.library.impl(lib, "copy_", "XPU")
@torch.library.impl(lib, "copy_", "HPU")
@torch.library.impl(lib, "copy_", "CPU")
@torch.library.impl(lib, "copy_", "MTIA")
def copy_(tensor, data):
    tensor.copy_(data)


@torch.library.impl(lib, "copy_", "Functionalize")
def copy__functionalize(tensor, data):
    torch._sync(tensor)
    torch._sync(data)
    tensor_inner = torch._from_functional_tensor(tensor)
    data_inner = torch._from_functional_tensor(data)
    with torch._C._ExcludeDispatchKeyGuard(
        torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
    ):
        torch.ops.fsdp.copy_.default(tensor_inner, data_inner)


torch.fx.node.has_side_effect(torch.ops.fsdp.copy_.default)


class ShardedState(Enum):
    """
    - ``SHARDED``: The sharded parameter is registered to the module. It is the
      only contributor to parameter memory.
    - ``SHARDED_POST_FORWARD``: The unsharded parameter is resharded to a
      smaller world size. Since this data should not be used for computation,
      we do not register it to the module. Users should reshard the module
      before any in-place modifications. Both it and the sharded parameter
      contribute to parameter memory.
    - ``UNSHARDED``: The unsharded parameter is registered to the module. Both
      it and the sharded parameter contribute to parameter memory.
    """

    SHARDED = auto()
    SHARDED_POST_FORWARD = auto()
    UNSHARDED = auto()


@dataclass
class ParamModuleInfo:
    """
    For a parameter, this stores the module and the parameter name to be able
    to do a parameter swap via ``setattr(module, param_name, ...)`` or to get
    the parameter via ``getattr(module, param_name)``. We additionally save
    shared modules and shared parameter names to update them accordingly.
    """

    # Parameter names are unprefixed, e.g. "weight", not "lin.weight"
    module: nn.Module
    param_name: str
    shared_modules: list[nn.Module] = field(default_factory=list)
    shared_param_names: list[str] = field(default_factory=list)


@dataclass
class ExtensionsData:
    # User-defined metadata passed from pre to post-all-gather
    all_gather_metadata: Any | None = None
    # Save the all-gather input sizes to unflatten the all-gather outputs to ND
    all_gather_input_sizes: Sequence[torch.Size] = ()  # ND

    def clear(self):
        self.all_gather_metadata = None
        self.all_gather_input_sizes = ()


class FSDPParam:
    """
    This class manages a parameter with FSDP or FSDP variants applied,
    implementing dim-0 per-parameter sharding.
    """

    orig_dtype: torch.dtype
    param_dtype: torch.dtype | None
    reduce_dtype: torch.dtype | None
    _orig_size: torch.Size  # ND
    sharded_size: torch.Size  # ND
    contiguous_sharded_stride: tuple[int, ...]
    padded_sharded_param_size: torch.Size  # ND
    sharded_post_forward_size: torch.Size  # ND
    contiguous_sharded_post_forward_stride: tuple[int, ...]
    _sharded_param_data: torch.Tensor  # 1D
    sharded_param: nn.Parameter  # ND
    _sharded_post_forward_param_data: torch.Tensor | None  # 1D
    _sharded_post_forward_param: nn.Parameter | None  # ND
    _unsharded_param: nn.Parameter  # ND
    unsharded_accumulated_grad: torch.Tensor | None  # ND
    _sharding_spec: DTensorSpec
    _unsharded_dtensor_spec: (
        DTensorSpec | None
    )  # set for DTensor params (SPMD or TP/EP)
    all_gather_outputs: list[torch.Tensor]  # 1D
    # All-gather extension attributes
    _extensions_data: ExtensionsData
    _unsharded_inner_tensors: list[torch.Tensor]
    _orig_param_uid: int

    def __init__(
        self,
        param: nn.Parameter,
        module_info: ParamModuleInfo,
        mesh_info: DataParallelMeshInfo,
        post_forward_mesh_info: FSDPMeshInfo | None,
        device: torch.device,
        shard_placement_fn: Callable[[nn.Parameter], ShardPlacementFnResult] | None,
        mp_policy: MixedPrecisionPolicy,
        offload_policy: OffloadPolicy,
    ):
        self._module_info: ParamModuleInfo = module_info
        self.post_forward_mesh_info = post_forward_mesh_info
        self.device = device
        self.mp_policy = mp_policy
        self.offload_to_cpu: bool = isinstance(offload_policy, CPUOffloadPolicy)
        self.pin_memory = (
            self.offload_to_cpu and cast(CPUOffloadPolicy, offload_policy).pin_memory
        )
        self.grad_offload_event: torch.Event | None = None
        self._init_sharded_param(param, device, shard_placement_fn, mesh_info)
        if self.post_forward_mesh_info:
            self._init_sharded_post_forward_param_metadata(param)
        self._init_extensions()
        self.all_gather_outputs: list[torch.Tensor] = []
        self.unsharded_accumulated_grad = None
        self._param_fqn: str | None = None  # prefixed from root module
        # TODO: Remove this padding logic once DTensor pads the local tensor:
        # https://github.com/pytorch/pytorch/issues/113045
        self._post_load_hook_handle = (
            module_info.module.register_load_state_dict_post_hook(
                lambda *args, **kwargs: self.reset_sharded_param()
            )
        )

    @torch.no_grad()
    def _init_sharded_param(
        self,
        param: nn.Parameter,
        device: torch.device,
        shard_placement_fn: Callable[[nn.Parameter], ShardPlacementFnResult] | None,
        mesh_info: DataParallelMeshInfo,
    ):
        shard_placement_result = None
        if callable(shard_placement_fn):
            shard_placement_result = resolve_shard_placement(
                shard_placement_fn(param),
                cast(FSDPMeshInfo, mesh_info),
            )
            self.mesh_info = shard_placement_result.mesh_info
            fsdp_placement = shard_placement_result.placement
        else:
            self.mesh_info = mesh_info  # pyrefly: ignore[bad-assignment]
            fsdp_placement = None
        self._shard_mesh = self._init_shard_mesh()
        if param.device != device and param.device.type != "meta":
            raise AssertionError(
                f"Expects the parameter to already be moved to device {device} but got {param.device}"
            )
        if not param.is_contiguous():
            raise NotImplementedError(
                f"FSDP does not support non-contiguous parameters yet: {param.shape=} {param.stride()=}"
            )
        if fsdp_placement is None:
            fsdp_placement = Shard(0)
        elif fsdp_placement.dim < 0:
            fsdp_placement = Shard(fsdp_placement.dim + param.ndim)
        if not isinstance(fsdp_placement, Shard):
            raise AssertionError(
                f"Expected Shard, got {type(fsdp_placement)}: {fsdp_placement}"
            )
        self.fsdp_placement = fsdp_placement
        shard_dim = fsdp_placement.dim
        # TODO: Replace the sharded DTensor parameter construction logic with
        # `distribute_tensor` after https://github.com/pytorch/pytorch/issues/116101
        # TODO: Simplify the following sharded parameter padding logic after
        # https://github.com/pytorch/pytorch/issues/113045
        # Tracks that this parameter contained FSDP-init-time spmd_types annotations;
        # this remains true even if spmd typechecking is disabled at runtime.
        self.is_spmd_types = (
            dist._is_spmd_types_available()
            and bool(spmd_local_type := spmd.get_local_type(param))
            and not isinstance(param, DTensor)
        )
        if self.is_spmd_types:
            param = self._resolve_spmd_types_for_storage(
                param, get_partition_spec(param), spmd_local_type, self.mesh_info
            )
        self.is_dtensor = isinstance(param, DTensor)
        self._orig_param_uid = _get_orig_param_uid(param)
        param_data = self._init_sharding_spec(param, fsdp_placement, shard_dim)
        if not param_data.is_contiguous():
            raise AssertionError(
                f"Expected contiguous tensor, got {param_data.shape=} {param_data.stride()=}"
            )
        shard_dim = fsdp_placement.dim
        if shard_dim >= param_data.ndim:
            raise AssertionError(
                f"Shard dim {shard_dim} is invalid for {param_data.ndim}D tensor: {param.shape}"
            )
        self._orig_size = param_data.size()
        self._contiguous_orig_stride = make_contiguous_strides_for(self._orig_size)
        if isinstance(self.mesh_info, FSDPMeshInfo):  # FSDP or HSDP
            shard_rank = self.mesh_info.shard_mesh_rank
            shard_world_size = self.mesh_info.shard_mesh_size
        else:  # DDP
            shard_rank = 0
            shard_world_size = 1

        if shard_dim > 0 and param_data.size(shard_dim) % shard_world_size != 0:
            # If sharding on nonzero dim, require even sharding for now because
            # the uneven sharding (1) requires extra copies before/after FSDP
            # collectives and (2) introduces extra complexity to handle padding
            # and unpadding
            raise NotImplementedError(
                f"FSDP does not support uneven sharding on dim {shard_dim}: "
                f"{param_data.size()} (world size: {shard_world_size})"
            )
        chunks = _chunk_with_empty(param_data, shard_world_size, dim=shard_dim)
        sharded_param = chunks[shard_rank]
        self.sharded_size = _get_dim_chunked_size(
            sharded_param, param_data.size(), dim=shard_dim
        )
        self.contiguous_sharded_stride = make_contiguous_strides_for(self.sharded_size)
        padded_sharded_size = chunks[0].size()  # 0th always padded
        self.padded_sharded_param_size = padded_sharded_size
        # Pre-pad the sharded parameter to avoid padding before all-gather
        padded_sharded_param = param_data.new_zeros(padded_sharded_size)
        if sharded_param.numel() > 0:
            padded_sharded_param.narrow(
                dim=shard_dim, start=0, length=sharded_param.size(shard_dim)
            ).copy_(sharded_param)
        if self.offload_to_cpu and not padded_sharded_param.is_meta:
            padded_sharded_param = padded_sharded_param.cpu()
            if self.pin_memory:
                padded_sharded_param = padded_sharded_param.pin_memory()
        self._sharded_param_data = padded_sharded_param.view(-1)
        length = sharded_param.size(shard_dim) if sharded_param.numel() > 0 else 0
        sharded_param = padded_sharded_param.narrow(
            dim=shard_dim, start=0, length=length
        )
        if not sharded_param.is_contiguous():
            raise AssertionError(
                f"Expected contiguous tensor with {self.fsdp_placement=}"
            )
        self.sharded_param = nn.Parameter(
            self.to_sharded_dtensor(sharded_param),
            requires_grad=param.requires_grad,
        )
        # Let `param_data` be freed normally when its ref count reaches 0 when
        # the `fully_shard` call returns to allow provided parameters to alias
        self._setattr_on_modules(self.sharded_param)
        self.sharded_state = ShardedState.SHARDED

    def _resolve_spmd_types_for_storage(
        self,
        param: nn.Parameter,
        partition_spec: "spmd.PartitionSpec | None",
        init_local_type: "spmd.LocalSpmdType",
        mesh_info: DataParallelMeshInfo,
    ) -> nn.Parameter:
        """
        Resolve spmd_types metadata and translate it into FSDP storage.

        The original parameter is a plain tensor annotated for module compute
        on the typechecking mesh. FSDP stores parameters as DTensors on
        ``mesh_info.spmd_mesh``, so init time only needs enough type information
        to translate into DTensor placements on the storage mesh axes. Axes
        named by ``dp_mesh_dims`` are FSDP-managed DP axes and are treated as
        ``spmd.R``; non-FSDP storage axes must have a direct spmd_types
        annotation.

        The original compute annotations and PartitionSpec are saved. At
        compute time, FSDP all-gathers the DTensor storage, exposes a plain
        tensor parameter, and restores spmd_types metadata on the init-time
        compute mesh. Missing compute axes are filled in as ``spmd.R``.

        Gradient storage placements are derived from the backward spmd_type.
        FSDP-managed "DP" axes require forward ``spmd.R`` since FSDP owns
        their gradient reduction. Consequently, the backward type is ``spmd.P``,
        so the incoming plain grad is wrapped as a ``Partial()`` DTensor on those
        axes, pending the gradient all-reduce/reduce-scatter to storage-time
        placement.

        For non-FSDP axes (e.g. TP), annotations are flexible. For a parameter
        annotated with ``spmd.I`` on TP, the grad is wrapped as ``Replicate()`` and no
        reduction is pending. ``spmd.S(dim)`` stays ``Shard(dim)``.

        This follows tied FWD/BWD typing for spmd_types: ``R`` -> ``P``,
        ``P`` -> ``R``, ``I`` -> ``I``, and ``S(dim)`` -> ``S(dim)``.
        """
        storage_mesh = mesh_info.spmd_mesh
        if storage_mesh is None or storage_mesh.mesh_dim_names is None:
            raise ValueError(
                "spmd_types parameters require fully_shard() to be called with both "
                "a named full DeviceMesh for FSDP storage and "
                "dp_mesh_dims=DataParallelMeshDims(shard=..., replicate=...), "
                "where shard names the mesh axis or axes FSDP shards parameters "
                "across and replicate names the HSDP/DDP mesh axis or axes FSDP "
                "replicates across."
            )

        storage_mesh_dim_names = storage_mesh.mesh_dim_names
        storage_mesh_axes = tuple(
            (name, spmd.MeshAxis.of(storage_mesh.get_group(name)))
            for name in storage_mesh_dim_names
        )
        storage_axes = tuple(axis for _, axis in storage_mesh_axes)
        local_type: spmd.PerMeshAxisSpmdTypes = dict(init_local_type)
        if partition_spec is not None:
            local_type.update(  # pyrefly: ignore [no-matching-overload]
                partition_spec_to_shard_types(partition_spec)
            )
        annotated_axes = tuple(local_type.keys())

        # resolve typechecking mesh on which parameters are annotated
        if _spans_same_mesh(annotated_axes, storage_axes):
            restore_mesh = annotated_axes
        elif (current_mesh := spmd.current_mesh()) is not None:
            restore_mesh = tuple(current_mesh)
        else:
            raise ValueError(
                f"Parameter '{self._module_info.param_name}' has partial "
                "spmd_types annotations that cannot be restored from its FSDP "
                "storage mesh. Fully annotate the parameter on its storage "
                "mesh, or wrap fully_shard() in spmd.set_current_mesh(...) "
                "when all parameters in the FSDP unit share one typechecking "
                "mesh. Mixed per-parameter typechecking meshes with partial "
                "annotations are not supported. "
                f"Annotated axes: {annotated_axes}. "
                f"Storage mesh axes: {storage_axes}."
            )

        # Validate the typechecking mesh with FSDP storage mesh.
        unknown_axes = tuple(
            axis for axis in init_local_type if axis not in restore_mesh
        )
        if unknown_axes:
            raise ValueError(
                f"FSDP found spmd_types annotations on parameter "
                f"'{self._module_info.param_name}' and will restore runtime "
                f"annotations on compute mesh axes {restore_mesh}, but the "
                f"parameter is annotated on axes {unknown_axes} that are not in "
                "that mesh. Annotate only axes in the compute mesh."
            )
        if not _spans_same_mesh(restore_mesh, storage_axes):
            raise ValueError(
                f"Parameter '{self._module_info.param_name}' uses spmd_types "
                "annotations on a typechecking mesh that does not span the "
                "same ranks as its FSDP storage mesh. FSDP can fill omitted "
                "FSDP-managed axes only when these meshes span the same "
                f"rank set. Typechecking mesh axes: {restore_mesh}. "
                f"Storage mesh axes: {storage_axes}."
            )
        dp_names = set(
            itertools.chain(
                mesh_info.dp_mesh_dims.shard_names,  # pyrefly: ignore [missing-attribute]
                mesh_info.dp_mesh_dims.replicate_names,  # pyrefly: ignore [missing-attribute]
            )
        )
        fsdp_axis = flatten_axes(
            tuple(axis for name, axis in storage_mesh_axes if name in dp_names)
        )
        non_fsdp_storage_mesh_axes = {
            axis for name, axis in storage_mesh_axes if name not in dp_names
        }
        storage_axis_types: spmd.PerMeshAxisSpmdTypes = {
            axis: spmd.R for name, axis in storage_mesh_axes if name in dp_names
        }
        # Param annotations are either optional spmd.R on FSDP axes or explicit
        # annotations on non-FSDP storage axes.
        for axis, axis_type in local_type.items():
            # MeshAxis ordering implements the FSDP-submesh containment check.
            if axis <= fsdp_axis:  # pyrefly: ignore [unsupported-operation]
                if axis_type is not spmd.R:
                    raise ValueError(
                        f"Expected spmd.R on FSDP DP axis {axis} for parameter "
                        f"'{self._module_info.param_name}' but got {axis_type}. "
                        "FSDP requires DP parameters to be R since it handles "
                        "the DP gradient reduction."
                    )
            else:
                if axis not in non_fsdp_storage_mesh_axes:
                    raise ValueError(
                        f"Parameter '{self._module_info.param_name}' has spmd_types "
                        f"annotation on axis {axis}, which is neither a non-FSDP "
                        "storage-mesh axis nor contained in the FSDP DP mesh."
                    )
                storage_axis_types[axis] = axis_type

        if set(storage_axes) != set(storage_axis_types.keys()):
            raise ValueError(
                f"Parameter '{self._module_info.param_name}' has incomplete "
                "spmd_types annotations for FSDP storage. "
                f"Annotated axes: {tuple(local_type.keys())}. "
                f"Storage mesh axes: {storage_axes}. "
                f"FSDP mesh dims: {mesh_info.dp_mesh_dims}. "
                "Missing non-FSDP storage axes: "
                f"{tuple(axis for _, axis in storage_mesh_axes if axis not in storage_axis_types)}."
            )

        # Translate to DTensor FWD/BWD placements; R@FSDP axes are optional.
        restore_type: spmd.PerMeshAxisSpmdTypes = {
            axis: init_local_type.get(axis, spmd.R) for axis in restore_mesh
        }
        placements = []
        grad_placements = []
        for axis_type in storage_axis_types.values():
            placements.append(spmd.spmd_type_to_dtensor_placement(axis_type))
            grad_placements.append(
                spmd.spmd_type_to_dtensor_placement(axis_type.backward_type())
            )
        dtensor_param = nn.Parameter(
            DTensor.from_local(param.data, storage_mesh, placements, run_check=False),
            requires_grad=param.requires_grad,
        )

        # Assign spmd_types state after validation succeeds.
        self._spmd_partition_spec: spmd.PartitionSpec | None = partition_spec
        self._spmd_init_local_type: spmd.LocalSpmdType = init_local_type
        self._spmd_restore_mesh: tuple[Any, ...] = restore_mesh
        self._spmd_restore_type: spmd.PerMeshAxisSpmdTypes = restore_type
        self._spmd_grad_placements: tuple[Placement, ...] = tuple(grad_placements)
        return dtensor_param

    def _restore_spmd_types(self, tensor: torch.Tensor) -> None:
        """Restore the saved spmd_types annotation onto a tensor."""
        if not self.is_spmd_types or not spmd.is_type_checking():
            return
        if self._spmd_restore_type:
            spmd.assert_type(
                tensor,
                self._spmd_restore_type,
                partition_spec=self._spmd_partition_spec,
            )

    def _init_sharding_spec(
        self,
        param: nn.Parameter,
        fsdp_placement: Shard,
        shard_dim: int,
    ) -> torch.Tensor:
        """
        Build ``_sharding_spec``, ``_spmd_mesh``, and ``_spmd_placements`` and
        return the local tensor data to be sharded.
        """
        self._unsharded_dtensor_spec = None
        if self.mesh_info.is_spmd_mesh and not self.is_dtensor:
            raise ValueError(
                "When dp_mesh_dims is provided, all parameters must be "
                "DTensors on the full SPMD mesh (e.g. via distribute_module). "
                f"Got plain tensor for parameter '{self._module_info.param_name}'."
            )
        if self.is_dtensor and self.mesh_info.is_spmd_mesh:
            return self._init_sharding_spec_spmd(param, fsdp_placement, shard_dim)
        if self.is_dtensor:
            return self._init_sharding_spec_tp(param, fsdp_placement, shard_dim)
        return self._init_sharding_spec_plain(param, fsdp_placement)

    def _init_sharding_spec_spmd(
        self,
        param: nn.Parameter,
        fsdp_placement: Shard,
        shard_dim: int,
    ) -> torch.Tensor:
        """SPMD path: param is a DTensor on the full SPMD mesh."""
        self._unsharded_dtensor_spec = cast(DTensor, param)._spec
        spmd_mesh = self._unsharded_dtensor_spec.mesh
        dp_dim_names = self.mesh_info.dp_mesh_dims
        if dp_dim_names is None:
            raise AssertionError("dp_dim_names must not be None for SPMD mesh")
        if spmd_mesh.mesh_dim_names is None:
            raise AssertionError("spmd_mesh.mesh_dim_names must not be None")
        if (
            self.mesh_info.spmd_mesh is not None
            and spmd_mesh is not self.mesh_info.spmd_mesh
        ):
            raise ValueError(
                "Expected param's DTensor mesh to be the same mesh passed "
                "to fully_shard, but got different mesh objects"
            )

        dp_shard_indices = [
            spmd_mesh.mesh_dim_names.index(n) for n in dp_dim_names.shard_names
        ]

        orig_placements = self._unsharded_dtensor_spec.placements
        for idx in dp_shard_indices:
            if not isinstance(orig_placements[idx], Replicate):
                raise ValueError(
                    f"Expected Replicate() on DP shard dim "
                    f"'{spmd_mesh.mesh_dim_names[idx]}' (index {idx}) "
                    f"but got {orig_placements[idx]}"
                )
        dp_replicate_indices = []
        for rep_name in dp_dim_names.replicate_names:
            rep_idx = spmd_mesh.mesh_dim_names.index(rep_name)
            dp_replicate_indices.append(rep_idx)
            if not isinstance(orig_placements[rep_idx], Replicate):
                raise ValueError(
                    f"Expected Replicate() on DP replicate dim "
                    f"'{spmd_mesh.mesh_dim_names[rep_idx]}' (index {rep_idx}) "
                    f"but got {orig_placements[rep_idx]}"
                )

        # Cache DP dim indices so _get_grad_inner_tensor can skip
        # redistribution on DP dims and let FSDP's reduce-scatter handle them.
        self._dp_dim_indices: frozenset[int] = frozenset(
            dp_shard_indices + dp_replicate_indices
        )

        new_placements = list(orig_placements)
        for dp_idx in dp_shard_indices:
            # split_factor = number of non-DP shards on shard_dim from
            # mesh dims with higher index (the "right-side" dims that
            # _StridedShard needs to interleave with)
            sf = 1
            for j in range(dp_idx + 1, spmd_mesh.ndim):
                p = orig_placements[j]
                if isinstance(p, (Shard, _StridedShard)) and p.dim == shard_dim:
                    sf *= spmd_mesh.size(j)
            new_placements[dp_idx] = (
                _StridedShard(shard_dim, split_factor=sf) if sf > 1 else fsdp_placement
            )

        self._spmd_mesh = spmd_mesh
        self._spmd_placements: tuple[Placement, ...] = tuple(new_placements)
        self._sharding_spec = self._build_spmd_sharding_spec(
            dp_dim_names,
            dp_shard_indices,
            fsdp_placement,
        )
        return cast(DTensor, param)._local_tensor

    def _build_spmd_sharding_spec(
        self,
        dp_dim_names: "DataParallelMeshDims",
        dp_shard_indices: list[int],
        fsdp_placement: Shard,
    ) -> DTensorSpec:
        """Build the DTensorSpec for the sharded parameter/gradient.

        When multiple DP shard dims exist (e.g. dp_shard + cp), flatten
        them into one axis so the parameter and gradient DTensor have one
        Shard axis for DP.
        """
        spmd_mesh = self._spmd_mesh
        spmd_placements = self._spmd_placements
        if self._unsharded_dtensor_spec is None:
            raise AssertionError("_unsharded_dtensor_spec cannot be None")
        tensor_meta = self._unsharded_dtensor_spec.tensor_meta

        if len(dp_shard_indices) <= 1:
            return DTensorSpec(spmd_mesh, spmd_placements, tensor_meta=tensor_meta)

        shard_names_set = set(dp_dim_names.shard_names)
        replicate_names_set = set(dp_dim_names.replicate_names)

        # Walk spmd_mesh.mesh_dim_names in order. Replace consecutive DP
        # shard dims with the flattened DP mesh; keep non-DP dims as-is.
        if spmd_mesh.mesh_dim_names is None:
            raise AssertionError("mesh_dim_names cannot be None")
        submeshes: list[DeviceMesh] = []
        spec_placements: list[Placement] = []
        skip = 0
        for i, name in enumerate(spmd_mesh.mesh_dim_names):
            if skip > 0:
                skip -= 1
                continue
            if name in shard_names_set:
                submeshes.append(self.mesh_info.mesh)
                if isinstance(self.mesh_info, HSDPMeshInfo):
                    spec_placements.append(Replicate())
                # Reuse the placement already computed for this DP shard dim
                # so that we don't loss _StridedShard.
                spec_placements.append(spmd_placements[i])
                skip = len(dp_dim_names.shard_names) - 1
            elif name in replicate_names_set and isinstance(
                self.mesh_info, HSDPMeshInfo
            ):
                # HSDP replicate is inserted by the shard branch above.
                continue
            else:
                submeshes.append(spmd_mesh[name])
                spec_placements.append(spmd_placements[i])

        spec_mesh = DeviceMesh._concatenate(submeshes)
        return DTensorSpec(spec_mesh, tuple(spec_placements), tensor_meta=tensor_meta)

    def _init_sharding_spec_tp(
        self,
        param: nn.Parameter,
        fsdp_placement: Shard,
        shard_dim: int,
    ) -> torch.Tensor:
        """TP/EP path: param is a DTensor, DP mesh is separate from TP mesh."""
        self._unsharded_dtensor_spec = cast(DTensor, param)._spec
        dp_mesh, tp_mesh = (self.mesh_info.mesh, self._unsharded_dtensor_spec.mesh)
        if dp_mesh is None or tp_mesh is None:
            raise AssertionError(
                "FSDP requires the DP and model parallel TP/EP mesh to be not None but got: \n"
                f"DP's mesh: {dp_mesh}\nTP/EP's mesh: {tp_mesh}"
            )
        self._spmd_mesh = DeviceMesh._concatenate([dp_mesh, tp_mesh])
        if len(self._unsharded_dtensor_spec.placements) > 2:
            raise NotImplementedError(
                f"FSDP only supports 1D TP/EP or 2D EP+TP, not {self._unsharded_dtensor_spec.placements}"
            )
        split_factor = self._unsharded_dtensor_spec.num_shards_map[shard_dim]
        if not (2 <= self._spmd_mesh.ndim <= 4):
            raise AssertionError(
                "_spmd_mesh.ndim can only be 2 (FSDP+TP/EP), 3 (FSDP+EP+TP, HSDP+TP/EP), "
                f"or 4 (HSDP+EP+TP) but got {self._spmd_mesh.ndim}."
            )
        if isinstance(self.mesh_info, FSDPMeshInfo):
            dp_shard_tp_placement = (
                (
                    _StridedShard(shard_dim, split_factor=split_factor)
                    if split_factor > 1
                    else fsdp_placement
                ),
                *self._unsharded_dtensor_spec.placements,
            )
        else:  # DDP
            dp_shard_tp_placement = (
                Replicate(),
                *self._unsharded_dtensor_spec.placements,
            )
        self._spmd_placements: tuple[Placement, ...]
        if isinstance(self.mesh_info, HSDPMeshInfo):
            if self.mesh_info.replicate_mesh_dim != 0:
                raise AssertionError(
                    f"Expected replicate_mesh_dim to be 0, got {self.mesh_info.replicate_mesh_dim}"
                )
            self._spmd_placements = (Replicate(),) + dp_shard_tp_placement
        else:
            self._spmd_placements = dp_shard_tp_placement

        self._sharding_spec = DTensorSpec(
            self._spmd_mesh,
            self._spmd_placements,
            tensor_meta=self._unsharded_dtensor_spec.tensor_meta,
        )
        return cast(DTensor, param)._local_tensor

    def _init_sharding_spec_plain(
        self,
        param: nn.Parameter,
        fsdp_placement: Shard,
    ) -> torch.Tensor:
        """Plain tensor path: param is not a DTensor."""
        self._spmd_mesh = self.mesh_info.mesh
        if isinstance(self.mesh_info, HSDPMeshInfo):
            self._spmd_placements = (Replicate(), fsdp_placement)
        elif isinstance(self.mesh_info, FSDPMeshInfo):
            self._spmd_placements = (fsdp_placement,)
        elif isinstance(self.mesh_info, DDPMeshInfo):
            self._spmd_placements = (Replicate(),)
        self._sharding_spec = DTensorSpec(
            self._spmd_mesh,
            self._spmd_placements,
            tensor_meta=TensorMeta(param.size(), param.stride(), param.dtype),
        )
        return param

    def _init_sharded_post_forward_param_metadata(self, param: torch.Tensor) -> None:
        mesh_info = self.post_forward_mesh_info
        if mesh_info is None:
            raise AssertionError("Expected post_forward_mesh_info to not be None")
        param_data = param._local_tensor if isinstance(param, DTensor) else param
        if isinstance(mesh_info, FSDPMeshInfo):
            chunks = _chunk_with_empty(param_data, mesh_info.shard_mesh_size, dim=0)
            self.sharded_post_forward_size = _get_dim_chunked_size(
                chunks[mesh_info.shard_mesh_rank],
                param_data.size(),
                dim=self.fsdp_placement.dim,
            )
        else:  # DDP
            chunks = _chunk_with_empty(param_data, 1, dim=0)
            self.sharded_post_forward_size = _get_dim_chunked_size(
                chunks[0],
                param_data.size(),
                dim=self.fsdp_placement.dim,
            )
        self.contiguous_sharded_post_forward_stride = make_contiguous_strides_for(
            self.sharded_post_forward_size
        )

    def init_dtype_attrs(self, mp_policy: MixedPrecisionPolicy):
        param_dtype, reduce_dtype = (mp_policy.param_dtype, mp_policy.reduce_dtype)
        self.orig_dtype = self.sharded_param.dtype
        # Clamp `reduce_dtype` to `None` if no casting is required: since
        # gradients are computed in `param_dtype`, if `reduce_dtype` matches,
        # then we do not need extra casting
        if reduce_dtype == param_dtype:
            reduce_dtype = None
        # Clamp `param_dtype` to `None` if no casting is required or if the
        # parameter is non-floating-point (mixed precision is only meaningful
        # for floating-point parameters)
        if param_dtype == self.orig_dtype or not self.orig_dtype.is_floating_point:
            param_dtype = None
        self.param_dtype = param_dtype
        self.reduce_dtype = reduce_dtype
        # None indicates that the mixed precision is not enabled

    def _init_extensions(self) -> None:
        inner_tensor = self._sharded_local_tensor
        has_fsdp_pre_all_gather = hasattr(inner_tensor, "fsdp_pre_all_gather")
        has_fsdp_post_all_gather = hasattr(inner_tensor, "fsdp_post_all_gather")
        if has_fsdp_pre_all_gather != has_fsdp_post_all_gather:
            raise AssertionError(
                "Both fsdp_pre_all_gather and fsdp_post_all_gather should be defined "
                f"if using all-gather extensions: {inner_tensor}"
            )
        if has_fsdp_pre_all_gather:
            self._extensions_data = ExtensionsData()
        self._unsharded_inner_tensors: list[torch.Tensor] = []

    def init_all_gather_outputs(
        self,
        all_gather_input_numels: list[int],
        all_gather_input_dtypes: list[torch.dtype],
        world_size: int,
        device: torch.device,
    ):
        if len(self.all_gather_outputs) > 0:
            return  # already initialized
        self.all_gather_outputs = [
            torch.empty(torch.Size([numel * world_size]), dtype=dtype, device=device)
            for numel, dtype in zip(all_gather_input_numels, all_gather_input_dtypes)
        ]

    def init_unsharded_param(self):
        if hasattr(self, "_unsharded_param"):  # after the 1st all-gather
            inner_tensor = self._sharded_local_tensor
            if not hasattr(inner_tensor, "fsdp_post_all_gather"):
                return  # already initialized
            for tensor in self._unsharded_inner_tensors:
                alloc_storage(tensor)
            all_gather_outputs = self._unflatten_all_gather_outputs()
            inner_tensor.fsdp_post_all_gather(
                all_gather_outputs,
                self._extensions_data.all_gather_metadata,
                self.param_dtype or self.orig_dtype,
                out=self._unsharded_param,
            )
            self._extensions_data.clear()
            return
        inner_tensor = self._sharded_local_tensor
        if hasattr(inner_tensor, "fsdp_post_all_gather"):
            all_gather_outputs = self._unflatten_all_gather_outputs()
            (
                unsharded_tensor,
                self._unsharded_inner_tensors,
            ) = inner_tensor.fsdp_post_all_gather(
                all_gather_outputs,
                self._extensions_data.all_gather_metadata,
                self.param_dtype or self.orig_dtype,
            )
            self._extensions_data.clear()
        else:
            # For the default path (no post-all-gather), the all-gather output
            # gives the unsharded parameter data directly
            if len(self.all_gather_outputs) != 1:
                raise AssertionError(
                    f"Expected 1 all_gather_output, got {len(self.all_gather_outputs)}"
                )
            unsharded_tensor = self.all_gather_outputs[0]
        unsharded_param = torch.as_strided(
            unsharded_tensor,
            self._orig_size,
            self._contiguous_orig_stride,
            storage_offset=0,
        )
        if self.is_spmd_types:
            pass  # keep as plain tensor; spmd_types restored before module compute
        elif self._unsharded_dtensor_spec is not None:
            unsharded_dtensor_spec = self._get_unsharded_dtensor_spec(unsharded_param)
            unsharded_param = _from_local_no_grad(
                unsharded_param, unsharded_dtensor_spec
            )
        self._unsharded_param = nn.Parameter(
            unsharded_param, requires_grad=self.sharded_param.requires_grad
        )

    def _get_unsharded_dtensor_spec(self, unsharded_param: torch.Tensor) -> DTensorSpec:
        if self._unsharded_dtensor_spec is None:
            raise AssertionError("Expected _unsharded_dtensor_spec for DTensor param")
        tensor_meta = self._unsharded_dtensor_spec.tensor_meta
        if tensor_meta is None or tensor_meta.dtype == unsharded_param.dtype:
            return self._unsharded_dtensor_spec
        return replace(
            self._unsharded_dtensor_spec,
            tensor_meta=TensorMeta(
                tensor_meta.shape,
                tensor_meta.stride,
                unsharded_param.dtype,
            ),
        )

    def _unflatten_all_gather_outputs(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            t.view(-1, *s[1:])
            for t, s in zip(
                self.all_gather_outputs, self._extensions_data.all_gather_input_sizes
            )
        )

    def to_sharded(self) -> None:
        self._setattr_on_modules(self.sharded_param)
        self.free_unsharded_param()
        self.sharded_state = ShardedState.SHARDED

    def to_sharded_post_forward(self) -> None:
        if self.is_dtensor:
            raise NotImplementedError(
                "Resharding to smaller mesh is not supported for DTensor parameters yet"
            )
        self._assert_in_states(ShardedState.UNSHARDED)
        if self.post_forward_mesh_info is None:
            raise AssertionError("Expected post_forward_mesh_info to not be None")
        if len(self.all_gather_outputs) != 1:
            raise AssertionError(
                f"Expected 1 all_gather_output, got {len(self.all_gather_outputs)}"
            )
        shard_world_size = self.post_forward_mesh_info.shard_mesh_size
        if (numel := self.all_gather_outputs[0].numel()) % shard_world_size != 0:
            _raise_assert_with_print(
                f"All-gather output size ({numel}) must be divisible by the shard "
                f"world size ({shard_world_size})"
            )
        shard_rank = self.post_forward_mesh_info.shard_mesh_rank
        sharded_numel = numel // shard_world_size
        self._sharded_post_forward_param_data = (
            self.all_gather_outputs[0].narrow(
                0, sharded_numel * shard_rank, sharded_numel
            )
        ).clone()  # clone to be able to free all-gather output
        sharded_post_forward_tensor = torch.as_strided(
            self._sharded_post_forward_param_data,
            size=self.sharded_post_forward_size,
            stride=self.contiguous_sharded_post_forward_stride,
            storage_offset=0,
        )
        self._sharded_post_forward_param = nn.Parameter(
            self.to_sharded_post_forward_dtensor(sharded_post_forward_tensor),
            requires_grad=self.sharded_param.requires_grad,
        )
        self._setattr_on_modules(self._sharded_post_forward_param)
        self.free_unsharded_param()
        self.sharded_state = ShardedState.SHARDED_POST_FORWARD

    def to_unsharded(self) -> None:
        # Assume that the data has been allocated and all-gathered
        set_requires_grad_if_needed(self.sharded_param, self._unsharded_param)
        self._setattr_on_modules(self._unsharded_param)
        if self.sharded_state == ShardedState.SHARDED_POST_FORWARD:
            # The data is allocated in the default stream via the post-forward
            # reshard and must be kept alive for the next all-gather copy-in.
            # Since we call this method after the copy-out, the data's lifetime
            # is ensured without further synchronization.
            self._sharded_post_forward_param = None
            self._sharded_post_forward_param_data = None  # free
        self.sharded_state = ShardedState.UNSHARDED

    def _setattr_on_modules(self, param: nn.Parameter) -> None:
        unsafe_setattr_param(
            self._module_info.module, self._module_info.param_name, param
        )
        for shared_module, shared_param_name in zip(
            self._module_info.shared_modules, self._module_info.shared_param_names
        ):
            unsafe_setattr_param(shared_module, shared_param_name, param)

    def to_sharded_dtensor(self, tensor: torch.Tensor) -> DTensor:
        """
        Converts a local tensor representing either the sharded parameter or
        sharded gradient to DTensor.
        """
        if tensor.shape != self.sharded_size:
            _raise_assert_with_print(
                f"Expects size {self.sharded_size} but got {tensor.shape}"
            )
        return _from_local_no_grad(
            tensor,
            self._sharding_spec,
        )

    def to_sharded_post_forward_dtensor(self, tensor: torch.Tensor) -> DTensor:
        if tensor.shape != self.sharded_post_forward_size:
            _raise_assert_with_print(
                f"Expects size {self.sharded_post_forward_size} but got {tensor.shape}"
            )
        if not isinstance(self.post_forward_mesh_info, HSDPMeshInfo):
            raise AssertionError(
                f"Expected HSDPMeshInfo, got {type(self.post_forward_mesh_info)}"
            )
        # TODO: Prefer this DTensor to be read-only and generalize the
        # placement once we support TP.
        post_forward_sharding_spec = DTensorSpec(
            self.post_forward_mesh_info.mesh,
            (Replicate(), Shard(0)),
            tensor_meta=self._sharding_spec.tensor_meta,
        )
        return _from_local_no_grad(tensor, post_forward_sharding_spec)

    def to_accumulated_grad_if_needed(self) -> None:
        # Access `_unsharded_param` to bypass the sharded state check since we
        # prefer to reshard before upcasting the gradient to save memory
        if (
            self.reduce_dtype is None
            or self._unsharded_param.grad is None
            or self._unsharded_param.grad.dtype == self.reduce_dtype
        ):
            return
        unsharded_grad = self._unsharded_param.grad
        self._unsharded_param.grad = None
        self.unsharded_accumulated_grad = unsharded_grad.to(self.reduce_dtype)

    def accumulate_unsharded_grad_if_needed(self) -> None:
        if (
            self.unsharded_accumulated_grad is not None
            and self.unsharded_param.grad is not None
        ):
            self.unsharded_accumulated_grad += self.unsharded_param.grad
            self.unsharded_param.grad = None

    def alloc_all_gather_outputs(self) -> None:
        for tensor in self.all_gather_outputs:
            alloc_storage(tensor)

    def free_unsharded_param(self) -> None:
        for tensor in itertools.chain(
            self.all_gather_outputs, self._unsharded_inner_tensors
        ):
            free_storage(tensor)

    @property
    def all_gather_inputs(self) -> list[torch.Tensor]:  # 1D
        self._assert_in_states(ShardedState.SHARDED, ShardedState.SHARDED_POST_FORWARD)
        if self.sharded_state == ShardedState.SHARDED:
            if hasattr(self._sharded_local_tensor, "fsdp_pre_all_gather"):
                sharded_local_tensor = self._sharded_local_tensor
                if self.offload_to_cpu:
                    sharded_local_tensor = sharded_local_tensor.to(
                        self.device, non_blocking=True
                    )
                pre_all_gather_signature = inspect.signature(
                    # pyrefly: ignore [missing-attribute]
                    sharded_local_tensor.fsdp_pre_all_gather
                )
                num_fn_params = len(pre_all_gather_signature.parameters)
                # Old signature only passes mesh; keep for BC for now
                if num_fn_params not in (1, 5):
                    raise AssertionError(
                        f"Invalid fsdp_pre_all_gather: {pre_all_gather_signature}\n"
                        "Expects fsdp_pre_all_gather(self, mesh: DeviceMesh, "
                        "outer_size: torch.Size, outer_stride: tuple[int, ...], "
                        "module: nn.Module, mp_policy: MixedPrecisionPolicy)"
                    )
                if num_fn_params == 1:
                    (
                        all_gather_inputs,
                        self._extensions_data.all_gather_metadata,
                        # pyrefly: ignore [missing-attribute]
                    ) = sharded_local_tensor.fsdp_pre_all_gather(
                        self.shard_mesh_from_root
                    )
                else:
                    (
                        all_gather_inputs,
                        self._extensions_data.all_gather_metadata,
                        # pyrefly: ignore [missing-attribute]
                    ) = sharded_local_tensor.fsdp_pre_all_gather(
                        self.shard_mesh_from_root,
                        self._orig_size,
                        self._contiguous_orig_stride,
                        self._module_info.module,
                        self.mp_policy,
                    )
                    if (
                        sharded_local_tensor.size() != self.padded_sharded_param_size
                        and any(
                            all_gather_input.size() != self.padded_sharded_param_size
                            for all_gather_input in all_gather_inputs
                        )
                    ):
                        # NOTE: Since this error can only be raised on the
                        # ranks that have padding, this can manifest as a NCCL
                        # watchdog timeout, as the other ranks will not error.
                        raise AssertionError(
                            "When a parameter is unevenly sharded by FSDP "
                            f"(orig size={self._orig_size}, FSDP world size={self.mesh_info.mesh.size()}), "
                            "fsdp_pre_all_gather must return all-gather inputs with the padded sharded size "
                            f"{self.padded_sharded_param_size} but got {[t.size() for t in all_gather_inputs]}"
                        )
                self._extensions_data.all_gather_input_sizes = [
                    t.size() for t in all_gather_inputs
                ]
                return [t.view(-1) for t in all_gather_inputs]
            sharded_param_data = self._sharded_param_data
            if self.offload_to_cpu:
                sharded_param_data = sharded_param_data.to(
                    self.device, non_blocking=True
                )
            return [_to_dtype_if_needed(sharded_param_data, self.param_dtype)]
        elif self.sharded_state == ShardedState.SHARDED_POST_FORWARD:
            if hasattr(self._sharded_local_tensor, "fsdp_pre_all_gather"):
                raise NotImplementedError
            all_gather_input = _to_dtype_if_needed(
                cast(torch.Tensor, self._sharded_post_forward_param_data),
                self.param_dtype,
            )
            return [all_gather_input]
        return [torch.empty(0)]  # mypy

    @property
    def unsharded_param(self) -> nn.Parameter:  # ND
        return self._unsharded_param

    @property
    def unsharded_grad_data(self) -> torch.Tensor:
        grad = self.unsharded_param.grad
        if grad is None:
            raise AssertionError("Expects unsharded_param.grad to not be None")
        return self._get_grad_inner_tensor(grad)

    @property
    def unsharded_accumulated_grad_data(self) -> torch.Tensor:
        grad = self.unsharded_accumulated_grad
        if grad is None:
            raise AssertionError("Expects unsharded_accumulated_grad to not be None")
        return self._get_grad_inner_tensor(grad)

    @property
    def unsharded_zero_grad_data(self) -> torch.Tensor:
        return self._get_grad_inner_tensor(torch.zeros_like(self.unsharded_param))

    def _get_grad_inner_tensor(self, grad: torch.Tensor) -> torch.Tensor:
        if self.is_spmd_types:
            if self._unsharded_dtensor_spec is None:
                raise AssertionError(
                    "Expected _unsharded_dtensor_spec for spmd_types param"
                )
            grad = DTensor.from_local(
                grad,
                self._unsharded_dtensor_spec.mesh,
                tuple(self._spmd_grad_placements),
                run_check=False,
            )
            if not self.is_dtensor:
                raise AssertionError(
                    "Expected spmd_types -> DTensor wrapping to use the DTensor gradient "
                    "path for correct gradient redistribution."
                )

        if self.is_dtensor:
            if isinstance(grad, AsyncCollectiveTensor):
                grad = grad.wait()
            if not isinstance(grad, DTensor):
                raise AssertionError(f"Expected DTensor, got {type(grad)}")
            if self._unsharded_dtensor_spec is None:
                raise AssertionError(
                    "Expected _unsharded_dtensor_spec for DTensor param"
                )
            placements = self._unsharded_dtensor_spec.placements
            if self.mesh_info.is_spmd_mesh:
                # Only redistribute non-DP dims; keep Partial on DP dims
                # so FSDP's reduce-scatter handles them directly, avoiding
                # a redundant all-reduce on the DP dimensions.
                target_placements = tuple(
                    grad.placements[i] if i in self._dp_dim_indices else placements[i]
                    for i in range(len(placements))
                )
                if target_placements != grad.placements:
                    if len(placements) != len(grad.placements):
                        raise AssertionError(
                            f"Expected same placement length: {placements=} {grad.placements=}"
                        )
                    grad = grad.redistribute(placements=target_placements)
            else:
                if placements != grad.placements:
                    if len(placements) != len(grad.placements):
                        raise AssertionError(
                            f"Expected same placement length: {placements=} {grad.placements=}"
                        )
                    grad = grad.redistribute(placements=placements)
            grad = grad._local_tensor
        return grad

    @property
    def _sharded_local_tensor(self) -> torch.Tensor:
        return cast(DTensor, self.sharded_param)._local_tensor

    def _init_shard_mesh(self) -> DeviceMesh:
        mesh = self.mesh_info.mesh
        if mesh.ndim == 1:
            return mesh
        if mesh.mesh_dim_names is None:
            raise AssertionError("Expected mesh_dim_names to not be None")
        return mesh[mesh.mesh_dim_names[-1]]

    @property
    def shard_mesh(self):
        return self._shard_mesh

    @property
    def shard_mesh_from_root(self):
        return self.shard_mesh

    def _assert_in_states(self, *states: ShardedState) -> None:
        if self.sharded_state not in states:
            _raise_assert_with_print(
                f"Expects to be in one of {states}, not {self.sharded_state}"
            )

    def reset_sharded_param(self):
        # For ops like `nn.Module._apply` or `load_state_dict(assign=True)`
        # that change the sharded parameter tensor, we may need to re-pad the
        # sharded local tensor and re-save the reference.
        module_info = self._module_info
        new_param = getattr(module_info.module, module_info.param_name)
        if new_param is not self.sharded_param:
            if torch.__future__.get_swap_module_params_on_conversion():
                raise AssertionError(
                    f"Expects swap_tensors to preserve object but got {new_param} "
                    f"instead of {self.sharded_param}"
                )
            self.sharded_param = new_param

        local_tensor = new_param._local_tensor
        if local_tensor.is_meta:
            return
        updated_local_tensor = False
        # local_tensor can be padded twice
        # 1st time in fully_shard(model)
        # 2nd time in model(input) lazy_init
        # 2nd time should be no-op if parameters remain unchanged
        # 2nd time shouldn't be no-op if people call model.load_state_dict(...) before lazy_init
        # this makes it possible for trainer to call `sd = model.state_dict()` before the training loop
        # and use `sd` without calling .state_dict() per iteration
        same_local_tensor = False
        # TODO: need to support tensor subclass
        if type(self._sharded_param_data) is torch.Tensor:
            same_local_tensor = (
                # when sharding param with shape (1, ...) over 2 ranks
                # local_tensor on rank 1 can be size 0, data_ptr() can be 0
                self._sharded_param_data.untyped_storage().data_ptr() > 0
                and self._sharded_param_data.untyped_storage().data_ptr()
                == local_tensor.untyped_storage().data_ptr()
            )
        padded_sharded_size = self.padded_sharded_param_size
        shard_dim = self.fsdp_placement.dim
        length = local_tensor.size(shard_dim) if local_tensor.numel() > 0 else 0
        if local_tensor.size() != padded_sharded_size and not same_local_tensor:
            if shard_dim != 0:
                raise AssertionError(
                    f"Shard({shard_dim}) requires even sharding: {local_tensor.size()=}"
                )
            padded_local_tensor = local_tensor.new_zeros(padded_sharded_size)
            padded_local_tensor.narrow(dim=shard_dim, start=0, length=length).copy_(
                local_tensor
            )
            local_tensor = padded_local_tensor
            updated_local_tensor = True
        if self.pin_memory and not local_tensor.is_pinned():
            local_tensor = local_tensor.cpu().pin_memory()
            updated_local_tensor = True
        if not same_local_tensor:
            self._sharded_param_data = local_tensor.view(-1)
        if not isinstance(self.sharded_param, DTensor):
            raise AssertionError(f"Expected DTensor, got {type(self.sharded_param)}")
        if updated_local_tensor:
            # Only change the local tensor object if needed
            self.sharded_param._local_tensor = local_tensor.narrow(
                dim=shard_dim, start=0, length=length
            )
            if not self.sharded_param._local_tensor.is_contiguous():
                raise AssertionError(
                    "Expected sharded_param._local_tensor to be contiguous"
                )
        self._sharding_spec = self.sharded_param._spec

    def __repr__(self):
        return f"FSDPParam(fqn={self._param_fqn}, orig_size={self._orig_size})"


def alloc_storage(tensor: torch.Tensor) -> None:
    size = tensor.numel() * tensor.itemsize
    if (storage := tensor.untyped_storage()).size() != size:
        storage.resize_(size)


def free_storage(tensor: torch.Tensor) -> None:
    if (storage := tensor.untyped_storage()).size() != 0:
        storage.resize_(0)


# NOTE: These bypass `nn.Module.__setattr__` checks, which incur non-trivial
# CPU overhead, if the module did not override it. For FSDP, we know we do not
# need those checks when transitioning between sharded/unsharded parameters.
def unsafe_setattr_param(
    module: nn.Module, param_name: str, param: nn.Parameter
) -> None:
    if getattr(module.__setattr__, "__func__", None) is nn.Module.__setattr__:
        module._parameters[param_name] = param
    else:  # slow path
        setattr(module, param_name, param)


def set_requires_grad_if_needed(
    src_tensor: torch.Tensor, dst_tensor: torch.Tensor
) -> None:
    # Only call `requires_grad_` if needed to avoid the Python <> C++ context
    # switch overhead
    if src_tensor.requires_grad != dst_tensor.requires_grad:
        dst_tensor.requires_grad_(src_tensor.requires_grad)
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import contextlib
import logging
from typing import Any, cast, Literal, NamedTuple, TYPE_CHECKING
from typing_extensions import TypeVarTuple, Unpack

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import _spmd_no_typecheck
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.fsdp._common_utils import (
    _named_parameters_with_duplicates,
    collect_grad_tensors,
    replace_grad_tensors,
)
from torch.profiler import record_function
from torch.utils.hooks import RemovableHandle

from ._fsdp_api import CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy
from ._fsdp_collectives import (
    AllGather,
    AllGatherResult,
    DefaultAllGather,
    DefaultReduceScatter,
    foreach_all_gather,
    foreach_all_gather_copy_out,
    foreach_reduce,
    ProcessGroupAllocAllGather,
    ProcessGroupAllocReduceScatter,
    ReduceScatter,
    SymmMemAllGather,
    SymmMemReduceScatter,
)
from ._fsdp_common import (
    _disable_functorch_if_active,
    _dynamo_disable,
    DataParallelMeshInfo,
    DDPMeshInfo,
    FSDPMeshInfo,
    HSDPMeshInfo,
    is_bw,
    ShardPlacementFnResult,
    TrainingState,
)
from ._fsdp_param import alloc_storage, FSDPParam, ParamModuleInfo, ShardedState


if TYPE_CHECKING:
    from collections.abc import Callable


logger = logging.getLogger("torch.distributed.fsdp.fully_shard")

_ModuleToHandleDict = dict[nn.Module, RemovableHandle]  # for state dict
_GradInputs = TypeVarTuple("_GradInputs")


"""
[Note: Overlapping all-gather copy-in and all-gather]
For implicit forward prefetching, we want to overlap the next copy-in with the
current all-gather. We do so using a separate copy-in stream. However, since
we have the all-gather input as a view into the output, we must make sure to
copy into different memory from the current all-gather's output. Thus, we keep
a reference to the current all-gather's output and have the next FSDP parameter
group free it after its copy-in. Finally, we have the last FSDP state flush the
reference to avoid holding onto memory after forward.
"""


class FSDPCommContext:
    """This has the communication state shared across FSDP states/parameter groups."""

    def lazy_init(self, device: torch.device):
        self.device_handle = _get_device_handle(device.type)
        # Setting the all-gather/reduce-scatter streams to be higher priority
        # can help avoid some issues where their copies in/out are delayed and
        # block computation (this is different from high-pri NCCL streams)
        high_priority = -1
        # All-gather state and copy-in stream allow overlapping the next
        # copy-in with the current all-gather in forward; copy-in overlaps with
        # reduce-scatter in backward without the separate copy-in stream
        self.all_gather_copy_in_stream = self.device_handle.Stream(
            priority=high_priority
        )
        # All-gather stream allows overlapping next all-gather with current
        # forward compute
        self.all_gather_stream = self.device_handle.Stream(priority=high_priority)
        # Reduce-scatter stream gives separate execution "thread" for post-
        # backward logic like pre/post-gradient division and reduce-scatter
        self.reduce_scatter_stream = self.device_handle.Stream(priority=high_priority)
        # The most recent post-reduce event across all param groups, recorded
        # on reduce_scatter_stream (FSDP) or all_reduce_stream (HSDP). Since
        # later events subsume earlier ones on the same stream, a single wait
        # on the last event recorded on each stream used in the post backward
        # hook covers all groups.
        self._last_post_reduce_events: dict[torch.Stream, torch.Event] = dict()
        # Run the HSDP all-reduces concurrently with all-gather/reduce-scatter
        # since collectives use different network resources and can overlap
        # in the typical intra-node sharding / inter-node replication case
        self.all_reduce_stream = self.device_handle.Stream()
        # All-gather/reduce-scatter states keep references to collective
        # tensors produced in one stream and used in another and accompanying
        # CUDA events for synchronization
        self.all_gather_state: AllGatherState | None = None
        self.reduce_scatter_states: list[ReduceScatterState] = []
        # Effective cap on retained reduce_scatter_states, resolved from the
        # per-group reduce_scatter_max_input_buffers in
        # FSDPState._init_shared_state. It lives here, not per group, because
        # reduce_scatter_states is a single shared list governed by one cap.
        self.reduce_scatter_max_input_buffers: int = 1
        # Post-forward order for explicit backward prefetching
        self.post_forward_order: list[FSDPParamGroup] = []  # will cause ref cycles

    def get_all_gather_streams(
        self, async_op: bool, training_state: TrainingState
    ) -> tuple[torch.Stream, torch.Stream]:
        if not async_op and training_state in (
            TrainingState.FORWARD,
            TrainingState.PRE_BACKWARD,
        ):
            # Use separate streams for implicit prefetching
            return self.all_gather_copy_in_stream, self.all_gather_stream
        current_stream = self.device_handle.current_stream()
        return current_stream, current_stream


# See [Note: Overlapping all-gather copy-in and all-gather]
class AllGatherState(NamedTuple):
    all_gather_result: AllGatherResult
    event: torch.Event | None  # all-gather copy-out


class ReduceScatterState(NamedTuple):
    reduce_scatter_input: torch.Tensor
    event: torch.Event | None  # reduce-scatter event


class AllReduceState(NamedTuple):
    # Holding all_reduce_input (the reduce-dtype AR buffer) keeps the
    # caching allocator from reusing the block across layers. This is a
    # structural invariant, not bookkeeping: without it, the next layer's
    # RS can reuse the same physical block before this layer's AR finishes
    # under slow AR, causing gradient aliasing. See PR #140044, PR #180900.
    all_reduce_input: torch.Tensor
    event: torch.Event | None  # all-reduce event


class FSDPParamGroup:
    """This class represents a parameter group to communicate together."""

    _orig_dtype: torch.dtype | None
    _reduce_dtype: torch.dtype | None

    def __init__(
        self,
        params: list[nn.Parameter],
        modules: tuple[nn.Module, ...],
        mesh_info: DataParallelMeshInfo,
        post_forward_mesh_info: FSDPMeshInfo | None,
        device: torch.device,
        shard_placement_fn: Callable[[nn.Parameter], ShardPlacementFnResult] | None,
        mp_policy: MixedPrecisionPolicy,
        offload_policy: OffloadPolicy,
    ):
        self.modules = modules  # permit ref cycle because 1:1 lifetime
        param_module_infos = _get_param_module_infos(params, modules)

        self.fsdp_params = [
            FSDPParam(
                param,
                module_info,
                mesh_info,
                post_forward_mesh_info,
                device,
                shard_placement_fn,
                mp_policy,
                offload_policy,
            )
            for param, module_info in zip(params, param_module_infos)
        ]
        self.mesh_info = mesh_info
        self.post_forward_mesh_info = post_forward_mesh_info
        self.device = device
        self.device_handle = _get_device_handle(device.type)
        self.mp_policy = mp_policy
        self.offload_policy = offload_policy
        self._training_state = TrainingState.IDLE
        # Group's sharded state always matches its parameters' sharded states
        self._sharded_state = ShardedState.SHARDED
        self._module_fqn: str | None = None  # prefixed from root module
        # Only consider resetting sharded parameters once in lazy init since it
        # can incur nontrivial overhead to reset them
        self._reset_sharded_params: bool = False

        # - Hook state
        self._module_to_pre_save_state_dict_hook_handle: _ModuleToHandleDict = {}
        self._module_to_pre_load_state_dict_hook_handle: _ModuleToHandleDict = {}
        self._all_reduce_hook: Callable[[torch.Tensor], None] | None = None
        self._all_gather_comm: AllGather = DefaultAllGather()
        self._all_gather_output = torch.empty(0, device=self.device)
        self._reduce_scatter_comm: ReduceScatter = DefaultReduceScatter()
        # Optional stream to run the user-defined all-reduce hook in
        # Saved here and not in the comm. context because we allow the user to
        # specify it, possibly at construction time before lazy init
        self._all_reduce_hook_stream: torch.cuda.Stream | None = None

        # - Communication and communication/computation overlap
        self.comm_ctx = FSDPCommContext()
        self._param_group_index: int = 0
        self._num_param_groups: int = 1
        # Group's indices in the shared post-forward order
        self._post_forward_indices: list[int] = []
        # Whether to reduce gradients at all (whether for FSDP or HSDP)
        self.reduce_grads: bool = True
        # Whether to all-reduce gradients for HSDP; only used if
        # `self.reduce_grads` is true, in which case setting this to false
        # means reduce-scatter but no all-reduce
        self.all_reduce_grads: bool = True
        # Whether to reshard parameters after backward (only useful for
        # gradient accumulation)
        self.reshard_after_backward: bool = True
        # Per-group input for the reduce-scatter copy-in (chunk_cat) buffer
        # cap-K, set via FSDPModule.set_reduce_scatter_max_input_buffers (see
        # its docstring). FSDP keeps 1 by default. These per-group values are
        # resolved into the single shared
        # comm_ctx.reduce_scatter_max_input_buffers that post_backward reads.
        self.reduce_scatter_max_input_buffers: int = 1
        # Optional custom factor for the gradient reduction op (e.g. to divide
        # by a factor other than the world size)
        self.gradient_divide_factor: float | None = None
        # Whether to include zero gradients for parameters that did not
        # receive a gradient in backward (e.g. due to conditional parameter
        # usage across ranks). This ensures all ranks participate in the same
        # reduce-scatter collectives, avoiding collective mismatch errors.
        self.reduce_scatter_unused_params: bool = False
        # Whether reduce-scatter and all-reduce should be issued using only
        # summations, potentially with separate pre-/post-scaling.
        self.force_sum_reduction_for_comms: bool = False
        # `async_op` arg used for pre-forward/pre-backward unshard; can be
        # overridden to only do explicit prefetching and avoid inter-stream
        # fragmentation from using separate unshard streams
        self.unshard_async_op: bool = False
        # Whether to unshard in backward: can be overridden by the user if the
        # parameters in this group are not needed for backward (e.g. embedding)
        self.unshard_in_backward: bool = True

        # - CUDA events for stream synchronization
        # Holds the all-gather output buffer, sync objects, and metadata
        self._all_gather_result: AllGatherResult | None = None
        # Holds the reduce-scatter/all-reduce view-out CUDA event that marks the end of
        # the group's post-backward (e.g. reduce-scatter, all-reduce and div), which
        # should be waited on at the end of backward
        self._post_reduce_event: torch.Event | None = None
        # Holds the reshard-after-forward CUDA event when resharding to a
        # different world size, which should be waited on in the next unshard
        self._reshard_after_forward_event: torch.Event | None = None

        # Only for HSDP, if accumulating gradients without all-reduce, save the
        # partial reduce output (only reduce-scattered but not all-reduced)
        self._partial_reduce_output: torch.Tensor | None = None
        # Holds the reduce-dtype AR buffer + completion event across
        # layers in HSDP+AR with reduce_dtype != orig_dtype (e.g., bf16
        # reduce + fp32 params). Structural invariant: the live Python
        # ref keeps the buffer off the caching allocator's free list,
        # preventing the next layer's RS from reusing the same physical
        # block while this layer's AR is still in flight. See
        # AllReduceState docstring and regression test PR #180900.
        self._all_reduce_state: AllReduceState | None = None

    # Initialization #
    def _init_mp_dtypes(self) -> None:
        for fsdp_param in self.fsdp_params:
            fsdp_param.init_dtype_attrs(self.mp_policy)
        trainable_params: list[FSDPParam] = [
            p for p in self.fsdp_params if p.sharded_param.requires_grad
        ]
        if trainable_params:
            params_for_dtype = trainable_params
        else:
            params_for_dtype = [
                p for p in self.fsdp_params if p.orig_dtype.is_floating_point
            ]
        orig_dtypes = {p.orig_dtype for p in params_for_dtype}
        reduce_dtypes = {p.reduce_dtype for p in params_for_dtype}
        if len(trainable_params) > 0 and len(orig_dtypes) != 1:
            # Models may have no grad params
            raise AssertionError(
                f"FSDP expects uniform original parameter dtype but got {orig_dtypes}"
            )
        if len(trainable_params) > 0 and len(reduce_dtypes) != 1:
            # This can be relaxed if we issue one reduce-scatter per reduce
            # dtype (but we would need a way for users to specify multiple
            # reduce dtypes)
            raise AssertionError(
                f"FSDP expects uniform reduce dtype but got {reduce_dtypes}"
            )
        dtype_sets_are_uniform = len(orig_dtypes) == 1 and len(reduce_dtypes) == 1
        self._orig_dtype = next(iter(orig_dtypes)) if dtype_sets_are_uniform else None
        self._reduce_dtype = (
            next(iter(reduce_dtypes)) if dtype_sets_are_uniform else None
        )

    def lazy_init(self):
        # Lazy init should be idempotent
        # Users may change or register parameters after construction time.
        # For example, DoRA (https://arxiv.org/abs/2402.09353) initializes linear magnitudes based on
        # other parameters (e.g. loaded from the state dict).
        if not hasattr(self.comm_ctx, "device_handle"):
            self.comm_ctx.device_handle = _get_device_handle(self.device.type)
        if self.is_sharded and not self._reset_sharded_params:
            for fsdp_param in self.fsdp_params:
                fsdp_param.reset_sharded_param()
                fsdp_param._init_extensions()  # allow monkey patch after init
            self._reset_sharded_params = True
        self._validate_no_meta_params()
        self._validate_cpu_offload_params()
        self._validate_reduce_scatter_max_input_buffers()
        # Initialize mixed precision attributes lazily in case the user changes
        # the parameter dtypes after construction time but before forward
        self._init_mp_dtypes()
        self._register_state_dict_hooks()

    def set_symm_mem(self, backend: Literal["NCCL"] = "NCCL") -> None:
        if not isinstance(self._all_gather_comm, (DefaultAllGather | SymmMemAllGather)):
            raise AssertionError(
                "cannot call set_symm_mem() "
                f"when all gather comm is custom: {self._all_gather_comm.__class__.__name__}"
            )
        self._all_gather_comm = SymmMemAllGather(
            self._all_gather_process_group, backend
        )
        if not isinstance(
            self._reduce_scatter_comm, (DefaultReduceScatter | SymmMemReduceScatter)
        ):
            raise AssertionError(
                "cannot call set_symm_mem() "
                f"when reduce scatter comm is custom: {self._reduce_scatter_comm.__class__.__name__}"
            )
        if self.force_sum_reduction_for_comms:
            # As of NCCL 2.29.3, NCCL symmetric reduce-scatter only supports SUM reduction
            self._reduce_scatter_comm = SymmMemReduceScatter(
                self._reduce_scatter_process_group, backend
            )

    def set_allocate_memory_from_process_group(self, enable: bool) -> None:
        """
        Whether to (try to) use the ProcessGroup's allocate_tensor method for
        the staging buffers for collective comms.
        """
        if not isinstance(
            self._all_gather_comm, (DefaultAllGather | ProcessGroupAllocAllGather)
        ):
            raise AssertionError(
                "cannot call set_allocate_memory_from_process_group() "
                f"when all gather comm is custom: {self._all_gather_comm.__class__.__name__}"
            )
        self._all_gather_comm = (
            ProcessGroupAllocAllGather(self._all_gather_process_group)
            if enable
            else DefaultAllGather()
        )

        if not isinstance(
            self._reduce_scatter_comm,
            (DefaultReduceScatter | ProcessGroupAllocReduceScatter),
        ):
            raise AssertionError(
                "cannot call set_allocate_memory_from_process_group() "
                f"when reduce scatter comm is custom: {self._reduce_scatter_comm.__class__.__name__}"
            )
        self._reduce_scatter_comm = (
            ProcessGroupAllocReduceScatter(self._reduce_scatter_process_group)
            if enable
