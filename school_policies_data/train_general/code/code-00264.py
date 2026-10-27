            else DefaultReduceScatter()
        )

    # Runtime #
    @_disable_functorch_if_active
    def unshard(self, async_op: bool = False):
        if self._all_gather_result is not None:  # already called, pending wait
            return
        if self.is_unsharded:
            return  # no-op
        if (
            not self.unshard_in_backward
            and self._training_state == TrainingState.PRE_BACKWARD
        ):
            return
        if self._reshard_after_forward_event is not None:
            # Resharded parameter data is allocated in the default stream and
            # used in the all-gather streams
            self._wait_all_gather_streams_on_event(self._reshard_after_forward_event)
            self._reshard_after_forward_event = None

        if isinstance(self.mesh_info, FSDPMeshInfo):
            world_size = self._all_gather_process_group.size()
        else:
            world_size = 1
        if world_size == 1:
            # can't skip due to early return in wait_for_unshard if
            # no self._all_gather_result
            self._all_gather_result = AllGatherResult(
                all_gather_output=self._all_gather_output,
                all_gather_event=self.device_handle.Event().record(),
                all_gather_work=None,
                param_all_gather_input_dtypes=[],
                param_all_gather_input_numels=[],
                all_gather_input_split_sizes=[],
            )

            return

        with record_function(self._with_fqn("FSDP::all_gather")):
            self._all_gather_result = foreach_all_gather(
                self.fsdp_params,
                self._all_gather_process_group,
                async_op,
                *self.comm_ctx.get_all_gather_streams(async_op, self._training_state),
                self.device,
                self._all_gather_comm,
            )

    @_disable_functorch_if_active
    def wait_for_unshard(self):
        """
        1. In forward with implicit prefetching, to overlap the current copy-out
        with the next all-gather, we save a reference to the current all-gather
        result to free after the next copy-out.
        2. Otherwise (explicit prefetching or in backward), we free the
        all-gather result immediately after the current copy-out since we can
        already overlap the current copy-out with the previous reduce-scatter.
        """
        if not self._all_gather_result:
            return  # no preceding unshard
        async_op = self._all_gather_result.all_gather_work is not None
        if self._training_state == TrainingState.FORWARD:  # implicit prefetch
            if prev_all_gather_state := self.comm_ctx.all_gather_state:
                self._wait_all_gather_streams_on_event(prev_all_gather_state.event)
                self.comm_ctx.all_gather_state = None  # free the all-gather result
        if isinstance(self.mesh_info, FSDPMeshInfo):
            world_size = self._all_gather_process_group.size()
        else:
            world_size = 1
        if world_size == 1:
            # directly initialize unsharded parameters from sharded parameters

            for fsdp_param in self.fsdp_params:
                # Use all_gather_inputs which already handles conversion to param_dtype
                # This is consistent with the world_size > 1 path
                all_gather_input = fsdp_param.all_gather_inputs[0]

                # Make sure the all_gather_outputs has proper storage size before using it
                # First ensure we have at least one tensor in all_gather_outputs
                fsdp_param.init_all_gather_outputs(
                    [all_gather_input.numel()],
                    [all_gather_input.dtype],
                    world_size,
                    self.device,
                )

                tensor = fsdp_param.all_gather_outputs[0]
                alloc_storage(tensor)

                # find alternative way to check if tensor.is_inference
                with torch.autograd._unsafe_preserve_version_counter(tensor):
                    tensor.copy_(all_gather_input)

        else:
            with record_function(self._with_fqn("FSDP::all_gather_copy_out")):
                foreach_all_gather_copy_out(
                    self._all_gather_result,
                    self.fsdp_params,
                    self._all_gather_process_group,
                )

        for fsdp_param in self.fsdp_params:
            fsdp_param.init_unsharded_param()

        self._to_unsharded()
        all_gather_copy_out_event = self.device_handle.Event()
        all_gather_copy_out_event.record()

        if (
            not async_op
            and self._training_state == TrainingState.FORWARD
            and world_size > 1
        ):
            # Defer free to allow for overlap of this copy-out with next
            # all-gather collective
            self.comm_ctx.all_gather_state = AllGatherState(
                self._all_gather_result, all_gather_copy_out_event
            )
        else:
            self._wait_all_gather_streams_on_event(all_gather_copy_out_event)

        self._all_gather_result = None  # free unless saved in `all_gather_state`

    def _wait_all_gather_streams_on_event(self, event: torch.Event | None):
        # Calling `unshard` before lazy init means streams are not initialized
        if hasattr(self.comm_ctx, "all_gather_copy_in_stream") and event is not None:
            self.comm_ctx.all_gather_copy_in_stream.wait_event(event)
        if hasattr(self.comm_ctx, "all_gather_stream") and event is not None:
            self.comm_ctx.all_gather_stream.wait_event(event)

    @_disable_functorch_if_active
    def reshard(self):
        if self._training_state == TrainingState.FORWARD:
            if not self._reshard_after_forward:
                return
            if self._use_post_forward_mesh:
                self._to_sharded_post_forward()
                self._reshard_after_forward_event = self.device_handle.Event()
                if self._reshard_after_forward_event is not None:
                    self._reshard_after_forward_event.record()
                return
        self._to_sharded()

    def _reset_iter_state(self) -> None:
        # See FSDPState._reset_iter_state for semantics. Waits on any
        # in-flight collective events owned by this group, discards
        # accumulated grad-reduction state, and restores sharded params.
        current_stream = self.device_handle.current_stream()
        if self._all_gather_result is not None:
            if (event := self._all_gather_result.all_gather_event) is not None:
                current_stream.wait_event(event)
            work = self._all_gather_result.all_gather_work
            if isinstance(work, dist.distributed_c10d.Work):
                work.wait()
            self._all_gather_result = None
        if self._post_reduce_event is not None:
            current_stream.wait_event(self._post_reduce_event)
            self._post_reduce_event = None
        if (
            self._all_reduce_state is not None
            and self._all_reduce_state.event is not None
        ):
            current_stream.wait_event(self._all_reduce_state.event)
        self._all_reduce_state = None
        if self._reshard_after_forward_event is not None:
            self._wait_all_gather_streams_on_event(self._reshard_after_forward_event)
            self._reshard_after_forward_event = None
        self._partial_reduce_output = None
        self._post_forward_indices.clear()
        self._training_state = TrainingState.IDLE
        self._to_sharded()

    def pre_forward(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        logger.debug("%s", self._with_fqn("FSDP::pre_forward"))
        with record_function(self._with_fqn("FSDP::pre_forward")):
            # FORWARD at entry means another module in the grouped
            # ``fully_shard([a, b])`` already registered post_backward this
            # pass; skip to avoid duplicate ``RegisterPostBackwardFunction``
            # autograd nodes.
            with _spmd_no_typecheck():
                entering_forward_pass = self._training_state != TrainingState.FORWARD
                self._training_state = TrainingState.FORWARD
                self.unshard(self.unshard_async_op)
                self.wait_for_unshard()
            for fsdp_param in self.fsdp_params:
                fsdp_param._restore_spmd_types(fsdp_param.unsharded_param)
            if entering_forward_pass:
                args, kwargs = self._register_post_backward_hook(args, kwargs)
            return args, kwargs

    def post_forward(self, module: nn.Module, input: Any, output: Any):
        logger.debug("%s", self._with_fqn("FSDP::post_forward"))
        with (
            _spmd_no_typecheck(),
            record_function(self._with_fqn("FSDP::post_forward")),
        ):
            # for AC(fully_shard(model)), AC runs fsdp's _pre_forward
            # it shouldn't change post_forward_order
            if not is_bw():
                self.reshard()
                self._record_post_forward()
            self._training_state = TrainingState.IDLE
            return output

    def _record_post_forward(self) -> None:
        # Since a group has one pre-backward unshard for each forward call
        # before the backward, we record each usage (with multiplicity)
        post_forward_index = len(self.comm_ctx.post_forward_order)
        self.comm_ctx.post_forward_order.append(self)
        self._post_forward_indices.append(post_forward_index)

    @_dynamo_disable
    def pre_backward(self, default_prefetch: bool, *unused: Any):
        if self._training_state == TrainingState.PRE_BACKWARD:
            return
        logger.debug("%s", self._with_fqn("FSDP::pre_backward"))
        with (
            _spmd_no_typecheck(),
            record_function(self._with_fqn("FSDP::pre_backward")),
        ):
            self._training_state = TrainingState.PRE_BACKWARD
            self.unshard(self.unshard_async_op)  # no-op if prefetched
            self.wait_for_unshard()
            if default_prefetch:
                self._backward_prefetch()

    @_dynamo_disable
    def post_backward(self, *unused: Any):
        with _spmd_no_typecheck():
            # This method should be idempotent and safe to call even when this
            # FSDP parameter group was not used in backward (should be a no-op)
            logger.debug("%s", self._with_fqn("FSDP::post_backward"))
            # Partial-group-forward detection: grouped ``fully_shard([a, b, ...])``
            # where the forward ran only a subset of the group's modules (chunked
            # loss, 1F1B, etc.). The wrapped post-hook took its partial path, so
            # ``state._post_forward`` never ran and no ``_pre_backward`` hook was
            # registered — ``pre_backward`` therefore never fired and state stays
            # at FORWARD instead of transitioning IDLE → PRE_BACKWARD.
            is_partial_group_backward = (
                len(self.modules) > 1  # grouped (structural)
                and self._training_state == TrainingState.FORWARD  # partial path taken
            )
            self._training_state = TrainingState.POST_BACKWARD
            with record_function(self._with_fqn("FSDP::post_backward_accumulate")):
                for fsdp_param in self.fsdp_params:
                    fsdp_param.accumulate_unsharded_grad_if_needed()
            with record_function(self._with_fqn("FSDP::post_backward_reshard")):
                if not self.reduce_grads:
                    if self.reshard_after_backward:
                        self.reshard()
                    for fsdp_param in self.fsdp_params:
                        fsdp_param.to_accumulated_grad_if_needed()
                    return
                # Save the autograd-computed gradients before resharding to only
                # access the unsharded parameters when their data is present
                fsdp_params_with_grad: list[FSDPParam] = []
                unsharded_grads: list[torch.Tensor] = []

                for fsdp_param in self.fsdp_params:
                    if not hasattr(fsdp_param, "_unsharded_param"):
                        continue
                    # May have an accumulated gradient of the reduce dtype if the
                    # previous backward did not reduce-scatter
                    if fsdp_param.unsharded_accumulated_grad is not None:
                        fsdp_params_with_grad.append(fsdp_param)
                        unsharded_grads.append(
                            fsdp_param.unsharded_accumulated_grad_data
                        )
                        fsdp_param.unsharded_accumulated_grad = None
                    elif fsdp_param.unsharded_param.grad is not None:
                        fsdp_params_with_grad.append(fsdp_param)
                        unsharded_grads.append(fsdp_param.unsharded_grad_data)
                        fsdp_param.unsharded_param.grad = None
                    elif (
                        self.reduce_scatter_unused_params
                        and fsdp_param.unsharded_param.requires_grad
                    ):
                        fsdp_params_with_grad.append(fsdp_param)
                        unsharded_grads.append(fsdp_param.unsharded_zero_grad_data)
                if self.reshard_after_backward:
                    self.reshard()
            # Recycle prior modules' reduce-scatter input buffers, keeping at most
            # `max_input_buffers` in flight: reclaim the oldest (wait on its
            # reduce-scatter, then drop the keepalive ref that was deferring the
            # allocator's reuse of its memory) until fewer than that remain, before
            # this module's reduce appends one more. See
            # set_reduce_scatter_max_input_buffers for the memory/overlap tradeoff.
            # (Assumes backward fires groups N-1 first; if not, overlap degrades but
            # correctness is preserved.)
            max_input_buffers = self.comm_ctx.reduce_scatter_max_input_buffers
            states = self.comm_ctx.reduce_scatter_states
            if (
                self._param_group_index == self._num_param_groups - 1
                and len(states) >= max_input_buffers
            ):
                with record_function(
                    f"FSDP::post_backward_rs_wait ({self._module_fqn})"
                ):
                    while len(states) >= max_input_buffers:
                        oldest = states.pop(0)
                        if oldest.event is not None:
                            self.device_handle.current_stream().wait_event(oldest.event)
                        del oldest
            if len(fsdp_params_with_grad) == 0:
                return
            with record_function(self._with_fqn("FSDP::post_backward_reduce")):
                all_reduce_pg = (
                    self._all_reduce_process_group
                    if isinstance(self.mesh_info, DDPMeshInfo)
                    else None
                )
                all_reduce_stream: torch.cuda.Stream
                if all_reduce_pg is None and self._all_reduce_hook_stream is not None:
                    # this means the native HSDP is not enabled,
                    # but user may want to have a custom HSDP setup
                    if self._all_reduce_hook is None:
                        raise AssertionError(
                            "all reduce hook stream is specified but hook itself is missing."
                        )
                    all_reduce_stream = self._all_reduce_hook_stream
                else:
                    all_reduce_stream = self.comm_ctx.all_reduce_stream

                self._wait_for_post_backward()
                (
                    reduce_scatter_input,
                    reduce_scatter_event,
                    post_reduce_stream,
                    self._post_reduce_event,
                    all_reduce_input,
                    all_reduce_event,
                    self._partial_reduce_output,
                ) = foreach_reduce(
                    fsdp_params_with_grad,
                    unsharded_grads,
                    (
                        # pyrefly: ignore [bad-argument-type]
                        self._reduce_scatter_process_group
                        if isinstance(self.mesh_info, FSDPMeshInfo)
                        else None  # pyre-fixme[6]
                    ),
                    self.comm_ctx.reduce_scatter_stream,
                    self._reduce_scatter_comm,
                    self._orig_dtype,
                    self._reduce_dtype,
                    self.device,
                    self.gradient_divide_factor,
                    (
                        self._all_reduce_process_group
                        if isinstance(self.mesh_info, DDPMeshInfo)
                        else None
                    ),
                    all_reduce_stream,
                    self.all_reduce_grads,
                    self._partial_reduce_output,
                    self._all_reduce_hook,
                    self.force_sum_reduction_for_comms,
                )
                self.comm_ctx._last_post_reduce_events[post_reduce_stream] = (
                    self._post_reduce_event
                )
                self.comm_ctx.reduce_scatter_states.append(
                    ReduceScatterState(reduce_scatter_input, reduce_scatter_event)
                )
                if is_partial_group_backward:
                    # Serialize the default stream on this invocation's
                    # post-accumulate event before returning to autograd.
                    # Otherwise the next partial-group invocation's autograd
                    # kernels queue on the default stream concurrently with
                    # this one's accumulate on the RS stream; the caching
                    # allocator then hands an autograd kernel a block that is
                    # a live input/output of this accumulate, corrupting grads
                    # on non-zero ranks. The post-accumulate event is load-
                    # bearing (pre-accumulate reduce_scatter_event is NOT, per
                    # MI350X); waiting at the next post_backward's entry is
                    # too late because autograd runs between post_backwards.
                    # No CPU sync.
                    #
                    # TODO(#181218): open questions on scope.
                    #   1. Conditional vs unconditional. The same cross-stream
                    #      hazard exists structurally in regular FSDP but has
                    #      not been observed to fire, plausibly because
                    #      ``reduce_scatter_states.append(...)`` ref-holds
                    #      ``reduce_scatter_input`` and cross-layer allocator
                    #      pressure is looser than within a group. Dropping
                    #      the gate needs a real-model overlap-loss measurement.
                    #   2. ROCm-specific vs cross-platform. Only observed on
                    #      ROCm/RCCL/MI350X; CUDA FSDP passes without this fix.
                    #      The standalone repro shows the stream-ordering
                    #      hazard (vector 1) is cross-platform, but FSDP's
                    #      ref-hold closes it on both. The residual FSDP-side
                    #      race on ROCm (vector 2) is on a non-Python-reachable
                    #      buffer — RCCL workspace or allocator fragment.
                    #      Whether vector 2 exists on CUDA FSDP but is timing-
                    #      masked is unresolved. See
                    #      ``fsdp2_chunked_loss_rocm_race.md``.
                    self.device_handle.current_stream().wait_event(
                        self._post_reduce_event
                    )
                if all_reduce_input is not None:
                    if self.device.type != "cpu":
                        if all_reduce_event is None:
                            raise AssertionError(
                                "Expected all_reduce_event to be set for non-CPU device"
                            )
                    self._all_reduce_state = AllReduceState(
                        all_reduce_input, all_reduce_event
                    )

    def finalize_backward(self):
        for event in self.comm_ctx._last_post_reduce_events.values():
            self.device_handle.current_stream().wait_event(event)
        self.comm_ctx._last_post_reduce_events = dict()
        self._post_reduce_event = None
        self._all_reduce_state = None
        for fsdp_param in self.fsdp_params:
            if fsdp_param.grad_offload_event is not None:
                fsdp_param.grad_offload_event.synchronize()
                fsdp_param.grad_offload_event = None
        if self._all_gather_result is not None:
            # If there was a mistargeted unshard without a corresponding wait,
            # then we wait here and clear the unshard
            if (event := self._all_gather_result.all_gather_event) is not None:
                torch.accelerator.current_stream().wait_event(event)
            work = self._all_gather_result.all_gather_work
            if isinstance(work, dist.distributed_c10d.Work):
                work.wait()
            self._all_gather_result = None
        self._post_forward_indices.clear()

    def _wait_for_post_backward(self):
        if self._post_reduce_event is not None:
            self.device_handle.current_stream().wait_event(self._post_reduce_event)
            self._post_reduce_event = None
        if (
            self._all_reduce_state is not None
            and self._all_reduce_state.event is not None
        ):
            self.device_handle.current_stream().wait_event(self._all_reduce_state.event)
        self._all_reduce_state = None

    def _backward_prefetch(self) -> None:
        if self._training_state == TrainingState.PRE_BACKWARD:
            if not self._post_forward_indices:
                # Can be cleared if running multiple `backward`s
                return
            curr_index = self._post_forward_indices.pop()
            if self._num_param_groups > 1:
                # Backward fires groups in reverse forward order:
                # N-1, N-2, ..., 1, 0.  Index 1 is always the
                # penultimate group regardless of N.  Prefetching here
                # lets the next module's AG overlap with group 0's RS
                # without holding unsharded params too long (as would
                # happen if we prefetched from N-1).
                if self._param_group_index != 1:
                    return
                # E.g. fully_shard(block, shard_placement_fn=...) creates two
                # param groups per block (dense + moe), giving
                # post_forward_order = [block0, block0.moe, block1, block1.moe].
                # block1.moe walks back past block1 to prefetch block0.moe then block0.
                curr_modules = self.modules
                target_modules: tuple[nn.Module, ...] | None = None
                for step in range(1, curr_index + 1):
                    target = self.comm_ctx.post_forward_order[curr_index - step]
                    if target.modules is curr_modules:
                        continue
                    if target_modules is None:
                        target_modules = target.modules
                    elif target.modules is not target_modules:
                        break
                    # Prefetch all groups of the target module in
                    # reverse forward order (highest index first),
                    # matching the explicit path in _pre_backward.
                    self._prefetch_unshard(target, "backward")
            elif curr_index > 0:
                target = self.comm_ctx.post_forward_order[curr_index - 1]
                self._prefetch_unshard(target, "backward")

    @staticmethod
    def _prefetch_unshard(
        target_fsdp_param_group: FSDPParamGroup, pass_type: str
    ) -> None:
        if pass_type == "backward":
            training_state = TrainingState.PRE_BACKWARD
        elif pass_type == "forward":
            training_state = TrainingState.FORWARD
        else:
            raise ValueError(f"Unknown pass type: {pass_type}")
        target_fqn = target_fsdp_param_group._module_fqn
        with (
            record_function(f"FSDP::{pass_type}_prefetch for {target_fqn}"),
            target_fsdp_param_group.use_training_state(training_state),
        ):
            async_op = target_fsdp_param_group.unshard_async_op
            target_fsdp_param_group.unshard(async_op)

    # Utilities #
    def _to_sharded(self):
        if not self.is_sharded:
            for fsdp_param in self.fsdp_params:
                fsdp_param.to_sharded()
            self._sharded_state = ShardedState.SHARDED

    def _to_sharded_post_forward(self):
        if not self.is_sharded_post_forward:
            for fsdp_param in self.fsdp_params:
                fsdp_param.to_sharded_post_forward()
            self._sharded_state = ShardedState.SHARDED_POST_FORWARD

    def _to_unsharded(self):
        if not self.is_unsharded:
            for fsdp_param in self.fsdp_params:
                fsdp_param.to_unsharded()
            self._sharded_state = ShardedState.UNSHARDED

    @property
    def is_sharded(self) -> bool:
        return self._sharded_state == ShardedState.SHARDED

    @property
    def is_sharded_post_forward(self) -> bool:
        return self._sharded_state == ShardedState.SHARDED_POST_FORWARD

    @property
    def is_unsharded(self) -> bool:
        return self._sharded_state == ShardedState.UNSHARDED

    @contextlib.contextmanager
    def use_training_state(self, training_state: TrainingState):
        old_training_state = self._training_state
        self._training_state = training_state
        try:
            yield
        finally:
            self._training_state = old_training_state

    # Hook Registration #
    def _register_post_backward_hook(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if not torch.is_grad_enabled():
            return args, kwargs

        # Collect all tensors that require gradients (including from dataclasses)
        inp_tensors = collect_grad_tensors((args, kwargs))
        if not inp_tensors:
            return args, kwargs

        # Apply RegisterPostBackwardFunction to all tensors at once
        out_tensors = RegisterPostBackwardFunction.apply(self, *inp_tensors)

        # Replace tensors in the structure (iterator order matches collect order)
        new_args, new_kwargs = replace_grad_tensors((args, kwargs), iter(out_tensors))
        return new_args, new_kwargs

    def _register_state_dict_hooks(self) -> None:
        num_pre_save_hooks = len(self._module_to_pre_save_state_dict_hook_handle)
        num_pre_load_hooks = len(self._module_to_pre_load_state_dict_hook_handle)
        if num_pre_save_hooks != num_pre_load_hooks:
            raise AssertionError(
                f"Pre-save: {num_pre_save_hooks} pre-load: {num_pre_load_hooks}"
            )
        if num_pre_save_hooks > 0:
            return  # already registered
        modules_with_fsdp_params: set[nn.Module] = {
            fsdp_param._module_info.module for fsdp_param in self.fsdp_params
        }

        def to_sharded_hook(*args: Any, **kwargs: Any) -> None:
            self._to_sharded()

        for module in modules_with_fsdp_params:
            self._module_to_pre_save_state_dict_hook_handle[module] = (
                module.register_state_dict_pre_hook(to_sharded_hook)
            )
            self._module_to_pre_load_state_dict_hook_handle[module] = (
                module._register_load_state_dict_pre_hook(to_sharded_hook)
            )

    # Properties #
    @property
    def _reshard_after_forward(self) -> bool:
        return self.post_forward_mesh_info is not None

    @property
    def _use_post_forward_mesh(self) -> bool:
        return (
            self._reshard_after_forward
            and self.mesh_info != self.post_forward_mesh_info
        )

    @property
    def _is_hsdp(self) -> bool:
        return isinstance(self.mesh_info, HSDPMeshInfo)

    @property
    def _all_gather_process_group(self) -> dist.ProcessGroup:
        mesh_info = (
            cast(FSDPMeshInfo, self.post_forward_mesh_info)
            if self.is_sharded_post_forward
            else self.mesh_info
        )
        if not isinstance(mesh_info, FSDPMeshInfo):
            raise AssertionError(
                f"Expected mesh_info to be FSDPMeshInfo, got {type(mesh_info)}"
            )
        return mesh_info.shard_process_group

    @property
    def _reduce_scatter_process_group(self) -> dist.ProcessGroup:
        if not isinstance(self.mesh_info, FSDPMeshInfo):
            raise AssertionError(
                f"Expected mesh_info to be FSDPMeshInfo, got {type(self.mesh_info)}"
            )
        # Falls back to the shard PG (shared with all-gather) unless
        # set_separate_reduce_scatter_group opted in to a dedicated PG.
        return (
            self.mesh_info.reduce_scatter_process_group
            or self.mesh_info.shard_process_group
        )

    def _set_separate_reduce_scatter_group(
        self, enable: bool, new_groups: dict[tuple[int, ...], dist.ProcessGroup]
    ) -> None:
        # Give reduce-scatter its own process group (separate from the shared
        # all-gather/shard PG) so the two can overlap in backward, or reset to
        # the shared group. ``new_groups`` caches groups by shard ranks within
        # one set_separate_reduce_scatter_group call so meshes sharding over the
        # same ranks share one communicator (typically one total) rather than one
        # per mesh. HSDP ranks may create only their local shard subgroup, so
        # use local synchronization to avoid waiting for nonmember ranks in
        # new_group's post-init barrier.
        mesh_info = self.mesh_info
        if not isinstance(mesh_info, FSDPMeshInfo):
            raise AssertionError(
                f"Expected mesh_info to be FSDPMeshInfo, got {type(mesh_info)}"
            )
        if not enable:
            mesh_info.reduce_scatter_process_group = None
            return
        ranks = tuple(dist.get_process_group_ranks(mesh_info.shard_process_group))
        if ranks not in new_groups:
            # Reuse the mesh's existing group if already enabled (idempotent),
            # else create one for this rank set.
            existing = mesh_info.reduce_scatter_process_group
            if existing is not None:
                new_groups[ranks] = existing
            else:
                new_group = dist.new_group(
                    list(ranks),
                    use_local_synchronization=True,
                    group_desc="fsdp_reduce_scatter",
                )
                if new_group == dist.GroupMember.NON_GROUP_MEMBER:
                    raise AssertionError(
                        f"Current rank was not included in process group {ranks}"
                    )
                new_groups[ranks] = new_group
        mesh_info.reduce_scatter_process_group = new_groups[ranks]

    @property
    def _all_reduce_process_group(self) -> dist.ProcessGroup:
        if not isinstance(self.mesh_info, DDPMeshInfo):
            raise AssertionError(
                f"Expected mesh_info to be DDPMeshInfo or HSDPMeshInfo, got {type(self.mesh_info)}"
            )
        return self.mesh_info.replicate_process_group

    def _with_fqn(self, label: str) -> str:
        if self._module_fqn:
            label = f"{label} ({self._module_fqn})"
        if self._num_param_groups > 1 and isinstance(self.mesh_info, FSDPMeshInfo):
            label = f"{label} [pg={self.mesh_info.shard_mesh_size}]"
        return label

    def __repr__(self):
        return f"FSDPParamGroup(fqn={self._module_fqn})"

    def _validate_no_meta_params(self):
        param_names_on_meta = [
            fsdp_param._param_fqn
            for fsdp_param in self.fsdp_params
            if fsdp_param.sharded_param.device.type == "meta"
        ]
        if param_names_on_meta:
            raise RuntimeError(
                "FSDP parameters should be materialized from meta device before training, "
                f"but the following were still on meta device: {param_names_on_meta}\n"
                "For example, call module.to_empty(device) to materialize to device and "
                "call module.reset_parameters() on each module to initialize values."
            )

    def _validate_cpu_offload_params(self):
        if not isinstance(self.offload_policy, CPUOffloadPolicy):
            return
        fsdp_params_not_on_cpu = [
            fsdp_param
            for fsdp_param in self.fsdp_params
            if fsdp_param.sharded_param.device.type != "cpu"
        ]
        if fsdp_params_not_on_cpu:
            raise RuntimeError(
                "FSDP parameters should be materialized on CPU when enabling CPU offloading. "
                'For example, load a CPU state dict or call module.to_empty(device="cpu"). '
                "Found following parameters on non-CPU device: "
                f"{[(fsdp_param._param_fqn, fsdp_param.sharded_param.device) for fsdp_param in fsdp_params_not_on_cpu]}\n"
            )

    def _validate_reduce_scatter_max_input_buffers(self):
        # Reject max_input_buffers > 1 with the symmetric-memory reduce-scatter
        # comm: retaining >1 input buffers relies on the allocator handing out a
        # fresh buffer while prior ones are ref-held, but the symmetric-memory
        # pool reuses a single buffer, so retaining K would force K symmetric
        # segments and can exhaust symmetric memory. Read the per-group cap (set
        # in __init__) so this is safe to call before the comm context is lazily
        # initialized (e.g. unshard before the first forward).
        if self.reduce_scatter_max_input_buffers > 1 and isinstance(
            self._reduce_scatter_comm, SymmMemReduceScatter
        ):
            raise ValueError(
                "set_reduce_scatter_max_input_buffers(>1) is not supported with "
                "the symmetric-memory reduce-scatter comm, but got "
                f"{self.reduce_scatter_max_input_buffers} for module "
                f"'{self._module_fqn}' using "
                f"{self._reduce_scatter_comm.__class__.__name__}. Keep "
                "max_input_buffers=1 with symmetric memory, or use the default "
                "reduce-scatter comm."
            )


def _get_param_module_infos(
    params: list[nn.Parameter], modules: tuple[nn.Module, ...]
) -> list[ParamModuleInfo]:
    """
    Shared parameter: lin1.weight = lin2.weight
    Shared module: mlp.lin1 = mlp.lin2
    We do not remove duplicates when traversing both modules and parameters to
    find shared modules' parameters and shared parameters within a module.
    """
    params_set = set(params)
    param_to_module_info: dict[nn.Parameter, ParamModuleInfo] = {}
    for module in modules:
        for _, submodule in module.named_modules(remove_duplicate=False):
            for param_name, param in _named_parameters_with_duplicates(
                submodule, recurse=False
            ):
                if param in params_set:
                    if param not in param_to_module_info:
                        param_to_module_info[param] = ParamModuleInfo(
                            submodule, param_name
                        )
                    else:
                        param_to_module_info[param].shared_modules.append(submodule)
                        param_to_module_info[param].shared_param_names.append(
                            param_name
                        )
    if len(param_to_module_info) != len(params):
        raise AssertionError(f"Some parameters are not in the module tree of {modules}")
    return [param_to_module_info[param] for param in params]


class RegisterPostBackwardFunction(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(param_group: FSDPParamGroup, *inputs: torch.Tensor):
        # All tensors in `inputs` should require gradient
        return inputs

    @staticmethod
    def setup_context(ctx, inputs: tuple[Any, ...], output: Any) -> None:
        param_group, *_ = inputs
        ctx.param_group = param_group

    @staticmethod
    def backward(ctx, *grads: torch.Tensor):
        ctx.param_group.post_backward()
        return (None,) + grads

    @staticmethod
    # pyrefly: ignore [bad-override]
    def jvp(
        ctx: Any,
        param_group_tangent: object,
        *grad_inputs: Unpack[_GradInputs],
    ) -> tuple[Unpack[_GradInputs]]:
        # Drop the non-tensor param_group tangent. The output pre-backward hook
        # queues final post-backward after all primal/tangent paths finish.
        return grad_inputs


if dist._is_spmd_types_available():
    import spmd_types

    spmd_types.register_local_autograd_function(RegisterPostBackwardFunction)
<|endoftext|># mypy: allow-untyped-decorators
# mypy: allow-untyped-defs

from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import Any, cast, Literal, NoReturn, overload, TYPE_CHECKING
from typing_extensions import deprecated

import torch
import torch.nn as nn
from torch.distributed._composable import contract

from ._fsdp_api import (
    AllGather,
    DataParallelMeshDims,
    MixedPrecisionPolicy,
    OffloadPolicy,
    ReduceScatter,
)
from ._fsdp_common import _dynamo_disable, FSDPMeshInfo, ShardPlacementFnResult
from ._fsdp_init import (
    _apply_to_module,
    _get_device_from_mesh,
    _get_mesh_info,
    _get_modules_and_states,
    _get_post_forward_mesh_info,
    _init_default_mesh,
    _init_param_group,
    _validate_mesh,
    _validate_module,
)
from ._fsdp_state import _get_module_fsdp_state, FSDPState


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from torch.distributed.tensor import DeviceMesh

    from ._fsdp_param_group import FSDPParamGroup

__all__ = [
    "fully_shard",
    "FSDPModule",
    "UnshardHandle",
    "register_fsdp_forward_method",
    "get_cls_to_fsdp_cls",
    "disable_fsdp_module_new_init",
    "share_comm_ctx",
]


cls_to_fsdp_cls: dict[type, type] = {}


def get_cls_to_fsdp_cls() -> dict[type, type]:
    return cls_to_fsdp_cls


@overload
# pyrefly: ignore [inconsistent-overload]
def fully_shard(
    module: nn.Module,
    *,
    mesh: DeviceMesh | None = ...,
    reshard_after_forward: bool | int | None = ...,
    shard_placement_fn: Callable[[nn.Parameter], ShardPlacementFnResult] | None = ...,
    mp_policy: MixedPrecisionPolicy = ...,
    offload_policy: OffloadPolicy = ...,
    ignored_params: set[nn.Parameter] | None = ...,
    dp_mesh_dims: DataParallelMeshDims | None = ...,
) -> FSDPModule: ...


@overload
# pyrefly: ignore [inconsistent-overload]
def fully_shard(
    module: list[nn.Module],
    *,
    mesh: DeviceMesh | None = ...,
    reshard_after_forward: bool | int | None = ...,
    shard_placement_fn: Callable[[nn.Parameter], ShardPlacementFnResult] | None = ...,
    mp_policy: MixedPrecisionPolicy = ...,
    offload_policy: OffloadPolicy = ...,
    ignored_params: set[nn.Parameter] | None = ...,
    dp_mesh_dims: DataParallelMeshDims | None = ...,
) -> list[FSDPModule]: ...


# The decorator adds a state object to `module` that can be accessed via
# `fully_shard.state(module)`. The state object and module are 1:1.
# [1] Python runtime decorator does not play well with static type checking
# so suppressing some type checks to support type overloads
# such that caller can still get correct return types based on input type
@contract(state_cls=FSDPState)  # type: ignore[misc] # see [1]
def fully_shard(
    module,
    *,
    mesh: DeviceMesh | None = None,
    reshard_after_forward: bool | int | None = None,
    shard_placement_fn: Callable[[nn.Parameter], ShardPlacementFnResult] | None = None,
    mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
    offload_policy: OffloadPolicy = OffloadPolicy(),
    ignored_params: set[nn.Parameter] | None = None,
    dp_mesh_dims: DataParallelMeshDims | None = None,
):
    """
    Apply fully sharded data parallelism (FSDP) to ``module``, where FSDP
    shards module parameters, gradients, and optimizer states across data
    parallel workers to save memory at the cost of communication.

    At initialization, FSDP shards the module's parameters across the data
    parallel workers given by ``mesh``. Before forward, FSDP all-gathers the
    sharded parameters across the data-parallel workers to get the unsharded
    parameters for forward computation. If ``reshard_after_forward`` is
    ``True``, then FSDP frees the unsharded parameters after forward and
    re-all-gathers them in backward before gradient computation. After gradient
    computation, FSDP frees the unsharded parameters and reduce-scatters the
    unsharded gradients across data-parallel workers.

    This implementation represents the sharded parameters as :class:`DTensor` s
    sharded on dim-0, while the unsharded parameters will be like the original
    parameters on ``module`` (e.g. :class:`torch.Tensor` if originally
    :class:`torch.Tensor`). A module
    `forward pre-hook <https://pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook>`_
    on ``module`` all-gathers the parameters, and a module
    `forward hook <https://pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook>`_
    on ``module`` frees them (if needed). Similar backward hooks all-gather
    parameters and later free parameters and reduce-scatter gradients.

    Since grouping multiple tensors together for one collective is critical for
    communication efficiency, this implementation makes this grouping first
    class. Calling :meth:`fully_shard` on ``module`` constructs one group that
    includes the parameters in ``module.parameters()`` except those already
    assigned to a group from an earlier call on a submodule. This means that
    :meth:`fully_shard` should be called bottom-up on your model. Each group's
    parameters are all-gathered in one collective, and its gradients are
    reduce-scattered in one collective. Partitioning the model into multiple
    groups ("layer by layer") allows for peak memory savings and communication/computation
    overlap. Users generally should *not* call :meth:`fully_shard` only on the
    topmost root module.

    When called with a list (``fully_shard([a, b, ...])``), the model's
    forward may run only a subset of the grouped modules, with the rest
    called later in the same iteration. Chunked-loss training with
    ``fully_shard([norm, head])`` is the motivating case: the main forward
    runs norm only, then head is invoked per chunk. Caveats:

    - Each standalone per-chunk invocation registers its own post_backward
      autograd node, so N chunk calls produce N reduce-scatters for that
      group.
    - ``mp_policy.cast_forward_inputs`` and ``mp_policy.output_dtype``
      both apply per module in the group — every invocation (including
      each standalone per-chunk call) casts its inputs to ``param_dtype``
      and its output to ``output_dtype``.

    .. note::
       If ``forward()`` or ``backward()`` raises, FSDP's per-iteration
       state (iteration forward-root marker, grouped-module run
       trackers, in-flight collective state, per-group training states)
       is left in an undefined condition. To recover and run another
       iteration, call :meth:`FSDPModule.reset_iter_state` on the root
       FSDP module. The failed iteration's gradients are discarded,
       including any ``no_sync`` / HSDP partial-reduce accumulation
       state.

    Args:
        module (Union[nn.Module, List[nn.Module]]): The module or modules to
            shard with FSDP and group together for communication.
        mesh (Optional[DeviceMesh]): This data parallel mesh defines the
            sharding and device. If 1D, then parameters are fully sharded
            across the 1D mesh (FSDP) with ``(Shard(0),)`` placement. If 2D,
            then parameters are sharded across the 1st dim and replicated
            across the 0th dim (HSDP) with ``(Replicate(), Shard(0))``
            placement. The mesh's device type gives the device type used for
            communication; if a CUDA or CUDA-like device type, then we use the
            current device.
        reshard_after_forward (Optional[Union[bool, int]]): This controls the parameter
            behavior after forward and can trade off memory and communication:

            - If ``True``, then this reshards parameters after forward and
              re-all-gathers in backward.
            - If ``False``, then this keeps the unsharded parameters in memory
              after forward and avoids the all-gather in backward. For best performance,
              we usually set ``False`` for the root module, because the root module
              is typically required immediately when the backward pass begins.
            - If ``None``, it is set to ``True`` for non-root modules and ``False``
              for root modules.
            - If an ``int``, then this represents the world size to reshard to
              after forward. It should be a non-trivial divisor of the ``mesh``
              shard dim size (i.e. excluding 1 and the dim size itself). A
              choice may be the intra-node size (e.g. ``torch.cuda.device_count()``).
              This allows the all-gather in backward to be over a smaller world
              size at the cost of higher memory usage than setting to ``True``.
            - After forward, the parameters registered to the module depend on
              to this: The registered parameters are the sharded parameters if
              ``True``; unsharded parameters if ``False``; and the parameters
              resharded to the smaller mesh otherwise. To modify the parameters
              between forward and backward, the registered parameters must be
              the sharded parameters. For ``False`` or an ``int``, this can be
              done by manually resharding via :meth:`reshard`.
        shard_placement_fn (Optional[Callable[[nn.Parameter], Optional[Shard | ShardPlacementResult]]]):
            This callable can be used to override the sharding placement and/or
            mesh for a parameter. It can return:

            - ``None``: Use default sharding (Shard(0)) on the mesh passed to
              ``fully_shard``.
            - :class:`Shard`: Shard the parameter on the specified dimension
              using the mesh passed to ``fully_shard``.
            - :class:`ShardPlacementResult`: Specify both the shard placement
              and a custom :class:`FSDPMeshInfo`. This allows different
              parameters to be sharded across different process groups, enabling
              use cases like Mixture of Experts where expert params use a
              different mesh than regular params.

            If sharding on a nonzero dim, we currently require even sharding,
            i.e. the tensor dim size on that dim must be divisible by the FSDP
            shard mesh size.
        mp_policy (MixedPrecisionPolicy): This controls the mixed precision
            policy, which offers parameter/reduction mixed precision for this
            module. See :class:`MixedPrecisionPolicy` for details.
        offload_policy (OffloadPolicy): This controls the offloading policy,
            which offers parameter/gradient/optimizer state offloading. See
            :class:`OffloadPolicy` and its subclasses for details.
        ignored_params: Optional(Set[nn.Parameter]): The set of parameters to be
            ignored by FSDP. They will not be sharded, nor moved to the device
            during init, nor have their gradients reduced in backward.
        dp_mesh_dims (Optional[DataParallelMeshDims]): When provided,
            ``mesh`` is treated as the full SPMD mesh, and parameters should be
            DTensors on this mesh with ``Replicate()`` on all DP dimensions.
            The ``shard`` field names which dim(s) FSDP shards on (multiple
            dims are flattened). The ``replicate`` field names the HSDP
            replication dim(s) (multiple dims are flattened).

    Returns:
        FSDPModule: The module with FSDP applied (in-place).
    """
    torch._C._log_api_usage_once("torch.distributed.fsdp.fully_shard")
    _validate_module(module, "fully_shard")
    mesh = mesh or _init_default_mesh()
    _validate_mesh(mesh, dp_mesh_dims)
    mesh_info = _get_mesh_info(mesh, dp_mesh_dims)
    device = _get_device_from_mesh(mesh)
    auto_reshard_after_forward = reshard_after_forward is None
    # If the user does not provide ``reshard_after_forward``, we set it to True.
    # During lazy_init, we identify which module is the root and override its value to False
    if isinstance(mesh_info, FSDPMeshInfo):
        if (
            mesh_info.is_spmd_mesh
            and not isinstance(reshard_after_forward, bool)
            and isinstance(reshard_after_forward, int)
        ):
            raise NotImplementedError(
                "reshard_after_forward as int is not yet supported with "
                "SPMD mesh (dp_mesh_dims)"
            )
        post_forward_mesh_info = _get_post_forward_mesh_info(
            reshard_after_forward if not auto_reshard_after_forward else True,  # type: ignore[arg-type]
            mesh_info,
        )
    else:
        # DDPMeshInfo: no sharding, so no post-forward resharding needed
        post_forward_mesh_info = None
    arg_module, modules, managed_modules, params, buffers = _get_modules_and_states(
        module, device, ignored_params
    )
    state = fully_shard.state(modules[0])  # type: ignore[attr-defined]
    state.init(modules, device, mp_policy, auto_reshard_after_forward)

    _init_param_group(
        state,
        params,
        modules,
        mesh_info,
        post_forward_mesh_info,
        device,
        shard_placement_fn,
        mp_policy,
        offload_policy,
        reshard_after_forward=reshard_after_forward
        if not auto_reshard_after_forward
        else True,
    )

    # For Dynamo
    for managed_module in managed_modules:
        managed_module._is_fsdp_managed_module = True  # type: ignore[assignment]
        managed_module._fsdp_use_orig_params = True  # type: ignore[assignment]

    # Place FSDP leftmost for highest priority in the method resolution order
    _apply_to_module(
        modules, cls_to_fsdp_cls, FSDPModule, "FSDP", _unimplemented_deepcopy
    )
    return arg_module


def _unimplemented_deepcopy(*args: Any, **kwargs: Any) -> NoReturn:
    raise AssertionError(
        "FSDP does not support deepcopy. Please use state dict for serialization."
    )


_enable_fsdp_module_new_init: bool = True


@contextmanager
def disable_fsdp_module_new_init() -> Iterator[None]:
    global _enable_fsdp_module_new_init
    prev, _enable_fsdp_module_new_init = _enable_fsdp_module_new_init, False
    try:
        yield
    finally:
        _enable_fsdp_module_new_init = prev


class FSDPModule:
    # Index in MRO where the original class is found.
    # For FSDP: [FSDP<Orig>, FSDPModule, Orig, ...] -> index 2
    # Subclasses like ReplicateModule override this.
    _orig_cls_mro_index: int = 2

    def __new__(cls, *args, **kwargs):
        """
        Override ``__new__`` to remove the FSDP class and directly construct
        the original class for cases like indexing into a container module.
        """
        orig_cls = cls.__mro__[cls._orig_cls_mro_index]
        self = orig_cls.__new__(orig_cls, *args, **kwargs)
        if _enable_fsdp_module_new_init:
            self.__init__(*args, **kwargs)
        return self

    def reshard(self) -> None:
        """
        Reshards the module's parameters, freeing the unsharded parameters if
        they are allocated and registering the sharded parameters to the
        module. This method is *not* recursive.
        """
        state = self._get_fsdp_state()
        for fsdp_param_group in state._fsdp_param_groups:
            fsdp_param_group.reshard()

    def unshard(self, async_op: bool = False) -> UnshardHandle | None:
        """
        Unshards the module's parameters by allocating memory and all-gathering
        the parameters. This method is *not* recursive. The unshard follows the
        :class:`MixedPrecisionPolicy`, so it will all-gather following
        ``param_dtype`` if set.

        Args:
            async_op (bool): If ``True``, then returns a :class:`UnshardHandle`
                that has a :meth:`wait` method to wait on the unshard op. If
                ``False``, then returns ``None`` and waits on the handle inside
                this function.

        .. note:: If ``async_op=True``, then FSDP will wait on the pending
            unshard in the module's pre-forward for the user. The user only
            needs to call :meth:`wait` explicitly if the wait should happen
            before pre-forward.
        """
        state = self._get_fsdp_state()
        for fsdp_param_group in state._fsdp_param_groups:
            fsdp_param_group.lazy_init()
            fsdp_param_group.unshard(async_op=async_op)
        handle = _UnshardHandleImpl(
            list(state._fsdp_param_groups) if state._fsdp_param_groups else None
        )
        if async_op:
            return handle
        handle.wait()
        return None

    @_dynamo_disable
    def reset_iter_state(self) -> None:
        """
        Resets FSDP's per-iteration state after an exception aborted a
        forward or backward mid-flight. The supported recovery workflow is:

        1. Catch the exception from ``forward()`` or ``backward()``.
        2. Call ``reset_iter_state()`` on the *root* FSDP module.
        3. Run the next iteration normally.

        The reset waits on any in-flight all-gather/reduce-scatter events,
        reshards every parameter group, and clears iteration trackers
        (``iter_forward_root``, ``_modules_to_run_forward``, post-forward
        order, per-group training states). Any in-flight gradient
        reductions are discarded: the failed iteration's gradients are
        lost, including HSDP partial-reduce-accumulation state and
        ``no_sync`` grad-accumulation state. Callers doing gradient
        accumulation should treat the microbatch sequence as invalidated
        and restart it.

        Must be called on the root FSDP module — i.e. the module the
        top-level ``fully_shard`` was applied to, equivalently the
        module first forwarded. Calling on a non-root module raises
        ``RuntimeError``.
        """
        state = self._get_fsdp_state()
        state._reset_iter_state()

    def set_is_last_backward(self, is_last_backward: bool) -> None:
        """
        Sets whether the next backward is the last one. On the last backward,
        FSDP waits on pending gradient reduction and clears internal data
        data structures for backward prefetching. This can be useful for
        microbatching.
        """
        state = self._get_fsdp_state()
        state._state_ctx.is_last_backward = is_last_backward

    def set_requires_gradient_sync(
        self, requires_gradient_sync: bool, *, recurse: bool = True
    ) -> None:
        """
        Sets if the module should sync gradients. This can be used to implement
        gradient accumulation *without communication*. For HSDP, this controls
        both reduce-scatter and all-reduce together. This is the equivalence of
        `no_sync` in FSDP1.

        Args:
            requires_gradient_sync (bool): Whether to reduce gradients for the
                module's parameters.
            recurse (bool): Whether to set for all FSDP submodules or just the
                passed-in module.
        """
        self_module = cast(nn.Module, self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, FSDPModule):
                state = module._get_fsdp_state()
                for fsdp_param_group in state._fsdp_param_groups:
                    fsdp_param_group.reduce_grads = requires_gradient_sync
                    fsdp_param_group.all_reduce_grads = requires_gradient_sync

    def set_requires_all_reduce(
        self, requires_all_reduce: bool, *, recurse: bool = True
    ) -> None:
        """
        Sets if the module should all-reduce gradients. This can be used to
        implement gradient accumulation with only reduce-scatter but not
        all-reduce for HSDP.
        """
        self_module = cast(nn.Module, self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, FSDPModule):
                state = module._get_fsdp_state()
                for fsdp_param_group in state._fsdp_param_groups:
                    fsdp_param_group.all_reduce_grads = requires_all_reduce

    def set_reshard_after_forward(
        self, reshard_after_forward: bool, recurse: bool = True
    ) -> None:
        """
        Sets if the module should reshard parameters after forward. This can be
        used to change the ``reshard_after_forward`` FSDP arg at runtime. For
        example, this can be used to set the FSDP root module's value to
        ``True`` (since it is otherwise specially set to ``False``), or it can
        set an FSDP module's value to ``False`` for running evals and set back
        to ``True`` for training.

        Args:
            reshard_after_forward (bool): Whether to reshard parameters after
                forward.
            recurse (bool): Whether to set for all FSDP submodules or just the
                passed-in module.
        """
        if not isinstance(reshard_after_forward, bool):
            raise ValueError(
                f"reshard_after_forward should be a bool, got {type(reshard_after_forward)}"
            )
        self_module = cast(nn.Module, self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, FSDPModule):
                state = module._get_fsdp_state()
                state._auto_reshard_after_forward = False
                for fsdp_param_group in state._fsdp_param_groups:
                    if not isinstance(fsdp_param_group.mesh_info, FSDPMeshInfo):
                        raise AssertionError
                    fsdp_param_group.post_forward_mesh_info = (
                        _get_post_forward_mesh_info(
                            reshard_after_forward,
                            fsdp_param_group.mesh_info,
                        )
                    )

    def set_reshard_after_backward(
        self, reshard_after_backward: bool, *, recurse: bool = True
    ) -> None:
        """
        Sets if the module should reshard parameters after backward. This can
        be used during gradient accumulation to trade off higher memory for
        reduced communication since the unsharded parameters do not need to be
        re-all-gathered before the next forward.

        Args:
            reshard_after_backward (bool): Whether to reshard parameters after
                backward.
            recurse (bool): Whether to set for all FSDP submodules or just the
                passed-in module.
        """
        self_module = cast(nn.Module, self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, FSDPModule):
                state = module._get_fsdp_state()
                for fsdp_param_group in state._fsdp_param_groups:
                    fsdp_param_group.reshard_after_backward = reshard_after_backward

    def set_modules_to_forward_prefetch(self, modules: list[FSDPModule]) -> None:
        """
        Sets the FSDP modules for which this FSDP module should explicitly
        prefetch all-gathers in forward. The prefetching runs after this
        module's all-gather copy-out.

        Passing a singleton list containing the next FSDP module gives the same
        all-gather overlap behavior as the default overlap behavior, except the
        prefetched all-gather is issued earlier from the CPU. Passing a list
        with at least length two is required for more aggressive overlap and
        will use more reserved memory.

        Args:
            modules (List[FSDPModule]): FSDP modules to prefetch.
        """
        _assert_all_fsdp_modules(modules)
        self._get_fsdp_state()._states_to_forward_prefetch = [
            module._get_fsdp_state() for module in modules
        ]

    def set_modules_to_backward_prefetch(self, modules: list[FSDPModule]) -> None:
        """
        Sets the FSDP modules for which this FSDP module should explicitly
        prefetch all-gathers in backward. This overrides the default backward
        pretching implementation that prefetches the next FSDP module based on
        the reverse post-forward order.

        Passing a singleton list containing the previous FSDP module gives the
        same all-gather overlap behavior as the default overlap behavior.
        Passing a list with at least length two is required for more aggressive
        overlap and will use more reserved memory.

        Args:
            modules (List[FSDPModule]): FSDP modules to prefetch.
        """
        _assert_all_fsdp_modules(modules)
        self._get_fsdp_state()._states_to_backward_prefetch = [
            module._get_fsdp_state() for module in modules
        ]

    def set_custom_all_gather(self, comm: AllGather) -> None:
        """
        Overrides the default ``all_gather`` communication behavior,
        to have better control over the communication and memory usage.
        See `Comm` and `ReduceScatter` for details.

        Args:
            comm (AllGather): Custom all-gather communication.
        """
        state = self._get_fsdp_state()
        if len(state._fsdp_param_groups) > 1:
            raise ValueError(
                "set_custom_all_gather is not supported with multiple param "
                "groups (from per-param mesh via shard_placement_fn). "
                "The custom comm would be ambiguous across groups with different meshes."
            )
        for fsdp_param_group in state._fsdp_param_groups:
            fsdp_param_group._all_gather_comm = comm

    def set_custom_reduce_scatter(self, comm: ReduceScatter) -> None:
        """
        Overrides the default ``reduce_scatter`` communication behavior,
        to have better control over the communication and memory usage.
        See `Comm` and `ReduceScatter` for details.

        Args:
            comm (ReduceScatter): Custom reduce_scatter communication.
        """
        state = self._get_fsdp_state()
        if len(state._fsdp_param_groups) > 1:
            raise ValueError(
                "set_custom_reduce_scatter is not supported with multiple param "
                "groups (from per-param mesh via shard_placement_fn). "
                "The custom comm would be ambiguous across groups with different meshes."
            )
        for fsdp_param_group in state._fsdp_param_groups:
            fsdp_param_group._reduce_scatter_comm = comm

    def set_all_reduce_hook(
        self,
        hook: Callable[[torch.Tensor], None],
        *,
        stream: torch.cuda.Stream | None = None,
    ):
        """
        Args:
            hook (Callable[[torch.Tensor], None]): User-defined all-reduce hook
                with expected signature ``hook(reduce_output: torch.Tensor) -> None``
                where ``reduce_output`` is the reduce-scatter output if only
                using FSDP or the all-reduce output if using native HSDP.
            stream (Optional[torch.cuda.Stream]): Stream to run the all-reduce
                hook in. This should only be set if not using native HSDP. If
                using native HSDP, the hook will run in the internally defined
                all-reduce stream used by the native HSDP all-reduce.
        """
        state = self._get_fsdp_state()
        if len(state._fsdp_param_groups) > 1:
            raise ValueError(
                "set_all_reduce_hook is not supported with multiple param "
                "groups (from per-param mesh via shard_placement_fn). "
                "The hook would be ambiguous across groups with different meshes."
            )
        for fsdp_param_group in state._fsdp_param_groups:
            fsdp_param_group._all_reduce_hook = hook
            if stream is not None:
                if fsdp_param_group._is_hsdp:
                    raise ValueError("stream cannot be set when using native HSDP")
                fsdp_param_group._all_reduce_hook_stream = stream

    def set_post_optim_event(self, event: torch.Event) -> None:
        """
        Sets a post-optimizer-step event for the root FSDP module to wait the
        all-gather streams on.

        By default, the root FSDP module waits the all-gather streams on the
        current stream to ensure that the optimizer step has finished before
        all-gathering. However, this may introduce false dependencies if
        there is unrelated computation after the optimizer step. This API
        allows the user to provide their own event to wait on. After the root
        waits on the event, the event is discarded, so this API should be
        called with a new event each iteration.

        Args:
            event (torch.Event): Event recorded after the optimizer step
                to wait all-gather streams on.
        """
        self._get_fsdp_state()._state_ctx.post_optim_event = event

    @deprecated("Use `set_gradient_divide_factor` instead")
    def set_reduce_scatter_divide_factor(self, factor: float) -> None:
        """Use :py:meth:`set_gradient_divide_factor` instead"""
        self.set_gradient_divide_factor(factor)

    def set_gradient_divide_factor(self, factor: float) -> None:
        """
        Sets a custom divide factor for the gradient reduction. This might use
        a custom reduce op using NCCL's PreMulSum, which allows multiplying by
        the factor before reduction.

        Args:
            factor (float): Custom divide factor.
        """
        state = self._get_fsdp_state()
        for fsdp_param_group in state._fsdp_param_groups:
            fsdp_param_group.gradient_divide_factor = factor

    def set_force_sum_reduction_for_comms(self, enable: bool) -> None:
        """
        Sets whether to require the low-level collective communication
        primitives to exclusively use "sum"-type reductions, even if it comes
        at the cost of separate additional pre- or post-scaling operations.
        This is needed for example because NCCL currently supports zero-copy
        transfers only for this kind of collectives.

        NB: for MTIA devices, this is always implicitly enabled.

        NB: if `set_all_reduce_hook` is used under FSDP setup, the caller needs
        to ensure the custom all-reduce across FSDP units follow this strategy
        as well, as FSDP can no longer automatically handle that.

        Args:
            enable (bool): Whether to only ever use ReduceOp.SUM for comms.
        """
        state = self._get_fsdp_state()
        for fsdp_param_group in state._fsdp_param_groups:
            fsdp_param_group.force_sum_reduction_for_comms = enable

    def set_reduce_scatter_unused_params(
        self, reduce_scatter_unused_params: bool, *, recurse: bool = True
    ) -> None:
        """
        Sets whether to include zero gradients for parameters that did not
        receive a gradient in backward. This is needed when different ranks
        use different parameters due to conditional control flow (e.g.
        multi-modal models, mixture of experts), causing mismatched
        reduce-scatter collectives. Similar to DDP's
        ``find_unused_parameters``.

        Args:
            reduce_scatter_unused_params (bool): Whether to include zero
                gradients for unused parameters in gradient reduction.
            recurse (bool): Whether to set for all FSDP submodules or just
                the passed-in module.
        """
        self_module = cast(nn.Module, self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, FSDPModule):
                state = module._get_fsdp_state()
                for fsdp_param_group in state._fsdp_param_groups:
                    fsdp_param_group.reduce_scatter_unused_params = (
                        reduce_scatter_unused_params
                    )

    def set_reduce_scatter_max_input_buffers(
        self, max_input_buffers: int, *, recurse: bool = True
    ) -> None:
        """
        Sets how many gradient reduce-scatter input buffers may be in flight at
        once -- the copy-in (``chunk_cat``) buffer cap-K (experimental).

        FSDP keeps **1** such buffer in flight by default, so the compute stream
        must wait on the previous reduce-scatter before the next copy-in can
        reuse that buffer. When the reduce-scatter is exposed (communication
        slower than the backward compute meant to hide it), that recycle wait
        stalls the compute stream. Raising the cap lets the next copy-in write a
        **fresh** buffer instead of waiting -- removing the stall -- at the cost
        of extra peak memory for the retained buffers. The copy-in stays on the
        compute stream; there is no extra stream and no ``record_stream``. This
        helps only when the reduce-scatter is exposed.

        Args:
            max_input_buffers (int): Max reduce-scatter input buffers retained in
                flight (the memory<->overlap dial); must be ``>= 1``. ``1`` is
                FSDP's default behavior (single buffer; the exposed-RS stall). A
                small value (e.g. ``2``) bounds peak memory and adds no stall as
                long as it is ``>=`` the reduce-scatter pipeline depth (otherwise
                an exposed reduce-scatter trades back a tail stall); a larger
                value retains more buffers for deeper overlap at higher peak
                memory.
            recurse (bool): Whether to set for all FSDP submodules or just the
                passed-in module.
        """
        # bool is an int subclass; reject it so True does not silently mean 1.
        if isinstance(max_input_buffers, bool) or not isinstance(
            max_input_buffers, int
        ):
            raise TypeError(
                "max_input_buffers must be an int, got "
                f"{type(max_input_buffers).__name__}"
            )
        if max_input_buffers < 1:
            raise ValueError(
                f"max_input_buffers must be a positive int, got {max_input_buffers}"
            )
        self_module = cast(nn.Module, self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, FSDPModule):
                state = module._get_fsdp_state()
                for fsdp_param_group in state._fsdp_param_groups:
                    fsdp_param_group.reduce_scatter_max_input_buffers = (
                        max_input_buffers
                    )

    def set_separate_reduce_scatter_group(
        self, enable: bool = True, *, recurse: bool = True
    ) -> None:
        """
        Enables (or disables) running gradient reduce-scatter on its own process
        group so it can overlap with all-gather in the backward pass
        (experimental).

        By default FSDP runs all-gather and reduce-scatter on separate CUDA
        streams but through the **same** process group -- one NCCL communicator,
        which processes one collective at a time and so serializes them on the
        wire. When enabled, FSDP creates a dedicated process group over the shard
        ranks (``dist.new_group(..., use_local_synchronization=True)``) -- one
        per distinct set of shard ranks, typically a single communicator -- so
        the two collectives can progress concurrently when the network can
        sustain it. This is collective for each shard rank set: like other FSDP
        comm setup, call it consistently across ranks using this FSDP mesh.

        Args:
            enable (bool): ``True`` (default) gives reduce-scatter its own
                process group; ``False`` resets it to the shared shard/all-gather
                group.
            recurse (bool): Whether to set for all FSDP submodules or just the
                passed-in module.
        """
        self_module = cast(nn.Module, self)
        modules = list(self_module.modules()) if recurse else [self_module]
        # Cache created groups by shard ranks so meshes that shard over the same
        # ranks share one communicator (typically one total), not one per mesh.
        new_groups: dict = {}
        for module in modules:
            if isinstance(module, FSDPModule):
                state = module._get_fsdp_state()
                for fsdp_param_group in state._fsdp_param_groups:
                    fsdp_param_group._set_separate_reduce_scatter_group(
                        enable, new_groups
                    )

    def set_unshard_in_backward(self, unshard_in_backward: bool) -> None:
        """
        Sets whether the FSDP module's parameters need to be unsharded in
        backward. This can be used in expert cases when the user knows that all
        parameters in this FSDP module's parameter group are not needed for
        backward computation (e.g. embedding).
        """
        state = self._get_fsdp_state()
        for fsdp_param_group in state._fsdp_param_groups:
            fsdp_param_group.unshard_in_backward = unshard_in_backward

    def set_allocate_memory_from_process_group_for_comm(self, enable: bool) -> None:
        """
        Sets whether the temporary staging buffers used to send and receive data
        over collective communications should be allocated using the custom
        optimized allocator provided by the ProcessGroup itself (if any). This
        might allow the ProcessGroup to be more efficient. For example, when
        using NCCL, this enables it to leverage zero-copy transfers over SHARP
        (for NVLink and/or InfiniBand).

        This cannot be used together with :meth:`set_custom_all_gather` or
        :meth:`set_custom_reduce_scatter` as those APIs allow for
        finer-grained control over each communication, and this method cannot
        determine their staging buffer allocation strategy.

        Args:
            enable (bool): Whether to turn on ProcessGroup allocation.
        """
        state = self._get_fsdp_state()
        for fsdp_param_group in state._fsdp_param_groups:
            fsdp_param_group.set_allocate_memory_from_process_group(enable)

    def set_symm_mem_for_comm(self, backend: Literal["NCCL"] = "NCCL") -> None:
        """
        Sets the symmetric memory (``symm_mem``) backend for allocating the
        staging buffers used in all-gather collectives. This allows NCCL to use
        optimized all-gather implementations via symmetric memory. Such
        optimization may depend on the topology of the system.  For single node,
        Copy Engine All-Gather may be used. For multi-node, Symmetric Kernel
        All-Gather may be used.

        To enable Copy Engine All-Gather, you need to set the NCCL process group
        with the zero-CTA policy.
        ```python
        opts = dist.ProcessGroupNCCL.Options()
        opts.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO
        dist.init_process_group(backend="nccl", pg_options=opts, device_id=device)
        ```
        Alternatively, you can set the environment variable `NCCL_CTA_POLICY` to 2.
        ```bash
        export NCCL_CTA_POLICY=2
        ```
        For more details, see [Copy Engine
        Collectives](https://docs.pytorch.org/docs/2.11/symmetric_memory.html#copy-engine-collectives).

        This cannot be used together with :meth:`set_custom_all_gather` or
        :meth:`set_custom_reduce_scatter`.

        Args:
            backend (str): The symmetric memory backend to use. Defaults to
                ``"NCCL"``. Currently, only ``"NCCL"`` is supported.
        """
        state = self._get_fsdp_state()
        for fsdp_param_group in state._fsdp_param_groups:
            fsdp_param_group.set_symm_mem(backend)

    def _set_unshard_async_op(self, async_op: bool):
        """
        Sets whether to use ``async_op=True`` or ``False`` for the pre-forward
        and pre-backward unshard op. This defaults to ``False`` but can be set
        to ``True`` with this method.

        Setting this to ``True`` allows the all-gather allocations to happen in
        the default stream, avoiding inter-stream memory fragmentation.
        However, you must use explicit prefetching (e.g. via :meth:`unshard`)
        in forward to still get overlap, and the pre-all-gather ops like dtype
        casting and copy-in will not overlap with compute.
        """
        self_module = cast(nn.Module, self)
        for module in self_module.modules():
            if isinstance(module, FSDPModule):
                state = module._get_fsdp_state()
                for fsdp_param_group in state._fsdp_param_groups:
                    fsdp_param_group.unshard_async_op = async_op

    def _get_fsdp_state(self) -> FSDPState:
        if (state := _get_module_fsdp_state(cast(nn.Module, self))) is None:
            raise AssertionError(f"No FSDP state found on {self}")
        return state

    def _apply(self, *args: Any, **kwargs: Any) -> Any:
        # Reshard to ensure that sharded parameters are registered
        self.reshard()
        ret = super()._apply(*args, **kwargs)  # type: ignore[misc]
        state = self._get_fsdp_state()
        if not state._fsdp_param_groups:
            return ret
        # TODO: Remove this padding logic once DTensor pads the local tensor:
        # https://github.com/pytorch/pytorch/issues/113045
        with torch.no_grad():
            for fsdp_param_group in state._fsdp_param_groups:
                for fsdp_param in fsdp_param_group.fsdp_params:
                    fsdp_param.reset_sharded_param()
        return ret


class UnshardHandle:
    """
    A handle to wait on a :meth:`FSDPModule.unshard` op.
    """

    def wait(self) -> None:
        """
        Waits on the unshard op. This ensures that the current stream can use
        the unsharded parameters, which are now registered to the module.
        """
        return


class _UnshardHandleImpl(UnshardHandle):
    def __init__(self, fsdp_param_groups: list[FSDPParamGroup] | None):
        self._fsdp_param_groups = fsdp_param_groups

    def wait(self):
        if self._fsdp_param_groups is not None:
            for fsdp_param_group in self._fsdp_param_groups:
                fsdp_param_group.wait_for_unshard()
            # Avoid keeping a reference
            self._fsdp_param_groups = None


def register_fsdp_forward_method(module: nn.Module, method_name: str) -> None:
    """
    Registers a method on ``module`` to be considered a forward method for
    FSDP.

    FSDP all-gathers parameters pre-forward and optionally frees parameters
    post-forward (depending on ``reshard_after_forward``). FSDP only knows to
    do this for :meth:`nn.Module.forward` by default. This function patches a
    user-specified method to run the pre/post-forward hooks before/after the
    method, respectively. If ``module`` is not an :class:`FSDPModule`, then
    this is a no-op.

    Args:
        module (nn.Module): Module to register the forward method on.
        method_name (str): Name of the forward method.
    """
    if not isinstance(module, FSDPModule):
        # Make no-op to allow including both when using/not using FSDP
        return
    if not hasattr(module, method_name):
        raise ValueError(f"{type(module)} does not have a method {method_name}")
    orig_method = getattr(module, method_name)

    @functools.wraps(orig_method)
    def wrapped_method(self, *args, **kwargs):
        fsdp_state = self._get_fsdp_state()
        args, kwargs = fsdp_state._pre_forward(self, args, kwargs)
        out = orig_method(*args, **kwargs)
        return fsdp_state._post_forward(self, args, out)

    # Use `__get__` to make `wrapped_method` an instance method
    setattr(
        module,
        method_name,
        wrapped_method.__get__(module, type(module)),  # type:ignore[attr-defined]
    )


def share_comm_ctx(modules: list[FSDPModule]) -> None:
    """
    Share cuda streams for multiple FSDPModules

    Example usage:
        from torch.distributed.fsdp import share_comm_ctx
        share_comm_ctx([fsdp_model_1, fsdp_model_2, ...])

    For Pipeline Parallelism (PP), each model chunk is a FSDP root. We want
    to share cuda streams for all-gather, reduce-scatter, and all-reduce.
    This avoids allocating inter-stream memory framgmentation

    Args:
        modules (List[FSDPModule]): modules to share cuda streams
    """
    if len(modules) == 0:
        return
    for module in modules:
        if not isinstance(module, FSDPModule):
            raise ValueError(f"Expects list of FSDPModules but got {module}")
    fsdp_states = [module._get_fsdp_state() for module in modules]
    comm_ctx = fsdp_states[0]._comm_ctx
    for fsdp_state in fsdp_states[1:]:
        fsdp_state._comm_ctx = comm_ctx
        for fsdp_param_group in fsdp_state._fsdp_param_groups:
            fsdp_param_group.comm_ctx = comm_ctx


def _assert_all_fsdp_modules(modules: Iterable[Any]) -> None:
    for module in modules:
        if not isinstance(module, FSDPModule):
            raise ValueError(f"Expects FSDPModule but got {type(module)}: {module}")
<|endoftext|># mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import functools
import logging
import warnings
from collections.abc import Callable, Sequence
from typing import Any, Generic, TYPE_CHECKING, TypeVar

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd.graph import _MultiHandle
from torch.distributed import _spmd_no_typecheck
from torch.distributed._composable_state import (
    _get_module_state,
    _insert_module_state,
    _State,
)
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.fsdp._common_utils import collect_grad_tensors
from torch.distributed.utils import _apply_to_tensors, _to_kwargs

from ._fsdp_api import MixedPrecisionPolicy
from ._fsdp_common import _cast_fp_tensor, _dynamo_disable, TrainingState
from ._fsdp_param_group import FSDPCommContext, FSDPParamGroup


if TYPE_CHECKING:
    from ._fsdp_param import FSDPParam

logger = logging.getLogger("torch.distributed.fsdp.fully_shard")

_StateType = TypeVar("_StateType", bound="FSDPState")


class FSDPStateContext(Generic[_StateType]):
    """This has state shared across FSDP states."""

    def __init__(self) -> None:
        # All FSDP states in the root state's module tree, in
        # ``named_modules()`` pre-order. ``_force_complete_incomplete_states``
        # iterates ``reversed`` so within a nesting chain the deepest
        # state's ``output_dtype`` cast applies first, matching the order
        # ``_post_forward`` would unwind in the non-partial case.
        self.all_states: list[_StateType] = []
        # Iteration's forward root runs the once-per-forward logic; this root
        # may not be the overall root set by lazy initialization in cases where
        # only a submodule runs forward (e.g. encoder-only for eval)
        self.iter_forward_root: _StateType | None = None
        # Final callback should only be queued once per backward
        self.post_backward_final_callback_queued: bool = False
        # Whether to finalize backward in this backward's final callback
        self.is_last_backward: bool = True
        # Optional user-provided event recorded after optimizer for the
        # all-gather streams to wait on in the root pre-forward
        self.post_optim_event: torch.Event | None = None


class FSDPState(_State):
    # Name used in error messages; subclasses can override
    _state_name: str = "FSDP"

    def __init__(self) -> None:
        super().__init__()
        # Support multiple param groups for per-param mesh support.
        # Each group has params with the same mesh_info.
        self._fsdp_param_groups: list[FSDPParamGroup] = []
        self._is_root: bool | None = None  # root set during lazy init
        self._state_ctx = FSDPStateContext()
        self._comm_ctx = FSDPCommContext()
        self._training_state: TrainingState = TrainingState.IDLE
        self._states_to_forward_prefetch: list[FSDPState] = []
        self._states_to_backward_prefetch: list[FSDPState] = []
        self._modules_to_run_forward: set[nn.Module] = set()
        # ``False`` when user set reshard_after_forward
        # through ``fully_shard`` or ``set_reshard_after_forward``
        self._auto_reshard_after_forward: bool | None = True

    def _get_state_for_module(self, module: nn.Module) -> "FSDPState | None":
        """Get the state for a module. Subclasses can override to use different state getters."""
        return _get_module_fsdp_state(module)

    @property
    def _fsdp_param_group(self) -> FSDPParamGroup | None:
        """
        Returns the param group for backward compatibility.
        This property is only valid when there is at most one param group.
        For per-param mesh support with multiple param groups, use
        ``_fsdp_param_groups`` instead.
        """
        if len(self._fsdp_param_groups) > 1:
            group_fqns = [g._module_fqn for g in self._fsdp_param_groups]
            raise AssertionError(
                f"Expected at most 1 param group for backward compatibility, "
                f"but got {len(self._fsdp_param_groups)} (fqns: {group_fqns}). "
                f"Use `_fsdp_param_groups` (plural) to access all param groups "
                f"when using per-param mesh via shard_placement_fn returning "
                f"ShardPlacementResult."
            )
        if self._fsdp_param_groups:
            return self._fsdp_param_groups[0]
        return None

    # Define a separate init since `__init__` is called in the contract
    def init(
        self,
        modules: tuple[nn.Module, ...],
        device: torch.device,
        mp_policy: MixedPrecisionPolicy,
        auto_reshard_after_forward: bool,
    ) -> None:
        for module in modules:
            _insert_module_state(module, self)
        self._modules = modules
        self._device = device
        self._device_handle = _get_device_handle(device.type)
        self._mp_policy = mp_policy
        self._auto_reshard_after_forward = auto_reshard_after_forward
        if len(modules) == 1:
            self._pre_forward_hook_handle = modules[0].register_forward_pre_hook(
                self._pre_forward, prepend=True, with_kwargs=True
            )
            self._post_forward_hook_handle = modules[0].register_forward_hook(
                self._post_forward, prepend=False
            )
        else:
            hook_handle = _register_group_forward_hooks(
                modules,
                self._pre_forward,
                self._post_forward,
                self._modules_to_run_forward,
                self._cast_output_dtype,
            )
            self._pre_forward_hook_handle = hook_handle
            self._post_forward_hook_handle = hook_handle

    def _root_pre_forward(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        self._lazy_init()
        if self._state_ctx.iter_forward_root is not None:
            return args, kwargs
        logger.debug("FSDP::root_pre_forward")
        self._state_ctx.iter_forward_root = self
        with torch.profiler.record_function("FSDP::root_pre_forward"):
            # Wait for optimizer before implicitly prefetched all-gathers
            if (event := self._state_ctx.post_optim_event) is not None:
                self._comm_ctx.all_gather_copy_in_stream.wait_event(event)
                self._comm_ctx.all_gather_stream.wait_event(event)
                self._state_ctx.post_optim_event = None
            else:
                current_stream = self._device_handle.current_stream()
                self._comm_ctx.all_gather_copy_in_stream.wait_stream(current_stream)
                self._comm_ctx.all_gather_stream.wait_stream(current_stream)
            if self._device.type in [
                "cuda",
                "hpu",
                "xpu",
                "mtia",
                torch._C._get_privateuse1_backend_name(),
            ]:
                if args or kwargs:
                    with torch.profiler.record_function("FSDP::inputs_to_device"):
                        args_tuple, kwargs_tuple = _to_kwargs(
                            args, kwargs, self._device, False
                        )  # same as DDP
                    args = args_tuple[0]
                    kwargs = kwargs_tuple[0]
        return args, kwargs

    def _lazy_init(self) -> None:
        """
        Lazy initialization represents when all modules' parallelisms have
        finalized (e.g. FSDP has been applied to all desired modules). This
        means that we can determine which state is the root, and we do so by
        the 1st state to run forward.
        """
        if self._is_root is not None:
            return  # no-op: already initialized
        self._is_root = True
        if len(self._modules) > 1:
            raise RuntimeError(
                f"{self._state_name} requires a single root module but got {self._modules}"
            )
        root_module = self._modules[0]
        visited_states: set[FSDPState] = set()
        for module_name, module in root_module.named_modules():
            if (state := self._get_state_for_module(module)) is None:
                continue
            if module is not root_module:
                if state not in visited_states and state._is_root is not None:
                    raise RuntimeError(
                        f"{self._state_name} state has already been lazily initialized for "
                        f"{module_name}\n{self._state_name} requires running forward through "
                        "the root module first"
                    )
                state._is_root = False
            # A single state can map to multiple modules (e.g.
            # fully_shard([mod_a, mod_b, mod_c])), so dedup here.
            if state not in visited_states:
                self._state_ctx.all_states.append(state)
            visited_states.add(state)
        # For the root, do not reshard after forward since for training,
        # the parameters would be freed and all-gathered immediately
        if self._auto_reshard_after_forward:
            for fsdp_param_group in self._fsdp_param_groups:
                fsdp_param_group.post_forward_mesh_info = None
        self._init_fqns()
        self._init_shared_state()
        self._validate_no_duplicate_params()
        # Run parameter group lazy inits after initializing FQNs for improved
        # error messages
        for state in self._state_ctx.all_states:
            for fsdp_param_group in state._fsdp_param_groups:
                fsdp_param_group.lazy_init()

    def _validate_no_duplicate_params(self) -> None:
        seen: set[int] = set()
        for state in self._state_ctx.all_states:
            for fsdp_param_group in state._fsdp_param_groups:
                for fsdp_param in fsdp_param_group.fsdp_params:
                    if fsdp_param._orig_param_uid in seen:
                        raise ValueError(
                            f"Parameter '{fsdp_param._param_fqn}' is shared with a "
                            f"parameter already managed by another FSDP group. "
                            f"For shared/tied parameters, use "
                            f"fully_shard([module_a, module_b]) to place them in "
                            f"the same FSDP group."
                        )
                    seen.add(fsdp_param._orig_param_uid)

    def _init_shared_state(self) -> None:
        self._comm_ctx.lazy_init(self._device)
        for state in self._state_ctx.all_states:
            state._state_ctx = self._state_ctx
            state._comm_ctx = self._comm_ctx
            num_groups = len(state._fsdp_param_groups)
            for i, fsdp_param_group in enumerate(state._fsdp_param_groups):
                fsdp_param_group.comm_ctx = self._comm_ctx
                fsdp_param_group._param_group_index = i
                fsdp_param_group._num_param_groups = num_groups
        # The reduce-scatter input-buffer cap governs the single shared
        # reduce_scatter_states list, so resolve the per-group caps into one
        # effective value (the most aggressive); a lower-cap group must not
        # silently undo a higher-cap group's retention.
        self._comm_ctx.reduce_scatter_max_input_buffers = max(
            (
                fsdp_param_group.reduce_scatter_max_input_buffers
                for state in self._state_ctx.all_states
                for fsdp_param_group in state._fsdp_param_groups
            ),
            default=1,
        )

    def _init_fqns(self) -> None:
        """Sets module and parameter FQN attributes for debugging."""
        if not self._is_root:
            raise AssertionError("Expected _is_root to be True")
        root_module = self._modules[0]
        param_to_fsdp_param: dict[nn.Parameter, FSDPParam] = {}
        # Build a mapping from module to all its FSDPParamGroups (not just one)
        module_to_fsdp_param_groups: dict[nn.Module, list[FSDPParamGroup]] = {}
        for state in self._state_ctx.all_states:
            for fsdp_param_group in state._fsdp_param_groups:
                for fsdp_param in fsdp_param_group.fsdp_params:
                    param_to_fsdp_param[fsdp_param.sharded_param] = fsdp_param
                for module in fsdp_param_group.modules:
                    if module not in module_to_fsdp_param_groups:
                        module_to_fsdp_param_groups[module] = []
                    module_to_fsdp_param_groups[module].append(fsdp_param_group)
        for param_name, param in root_module.named_parameters():
            if param in param_to_fsdp_param:
                param_to_fsdp_param[param]._param_fqn = param_name
        for module_name, module in root_module.named_modules():
            if module in module_to_fsdp_param_groups:
                # Set FQN for all param groups associated with this module
                for fsdp_param_group in module_to_fsdp_param_groups[module]:
                    module_fqn = fsdp_param_group._module_fqn
                    if module_fqn is None:
                        fsdp_param_group._module_fqn = module_name
                    else:
                        if not isinstance(module_fqn, str):
                            raise AssertionError(
                                f"Expected module_fqn to be str, got {type(module_fqn)}: {module_fqn}"
                            )
                        module_fqn += f", {module_name}"
                        fsdp_param_group._module_fqn = module_fqn

    @_dynamo_disable
    def _pre_forward(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        # When composing with module-hook-based activation checkpointing, the
        # pre-backward hook is responsible for the unshard
        if self._training_state == TrainingState.PRE_BACKWARD:
            with _spmd_no_typecheck():
                # With nested FSDP and multiple forward passes before backward,
                # the params might have been resharded by a previous post_backward.
                # We need to ensure params are unsharded for AC recomputation.
                for fsdp_param_group in self._fsdp_param_groups:
                    if not fsdp_param_group.is_unsharded:
                        fsdp_param_group.unshard()
                        fsdp_param_group.wait_for_unshard()
            return self._cast_forward_inputs(args, kwargs)

        # With grouped ``fully_shard([a, b, ...])`` the pre-hook fires per
        # module (so ``cast_forward_inputs`` and ``fsdp_param_group.pre_forward``
        # run for each). Root setup and forward prefetch are one-shot, gated
        # on the first module's entry.
        with _spmd_no_typecheck():
            state_first_in_pass = self._training_state != TrainingState.FORWARD
            self._training_state = TrainingState.FORWARD
            if state_first_in_pass:
                args, kwargs = self._root_pre_forward(module, args, kwargs)
        args, kwargs = self._cast_forward_inputs(args, kwargs)
        for fsdp_param_group in self._fsdp_param_groups:
            args, kwargs = fsdp_param_group.pre_forward(module, args, kwargs)
        with _spmd_no_typecheck():
            if state_first_in_pass:
                for fsdp_state in self._states_to_forward_prefetch:
                    # Forward order (not reversed) to match forward execution
                    # order; contrast with reversed() in _pre_backward.
                    for target_param_group in fsdp_state._fsdp_param_groups:
                        FSDPParamGroup._prefetch_unshard(target_param_group, "forward")
            return args, kwargs

    @_dynamo_disable
    def _post_forward(self, module: nn.Module, input: Any, output: Any) -> Any:
        with _spmd_no_typecheck():
            # When composing with module-hook-based activation checkpointing, the
            # post-backward hook is responsible for the reshard
            if self._training_state == TrainingState.PRE_BACKWARD:
                return self._cast_output_dtype(output)
            for fsdp_param_group in self._fsdp_param_groups:
                output = fsdp_param_group.post_forward(module, input, output)
            output = self._register_pre_backward_hook(output)
            self._training_state = TrainingState.IDLE
            if self._state_ctx.iter_forward_root is self:
                output = self._force_complete_incomplete_states(output)
                if all_gather_state := self._comm_ctx.all_gather_state:
                    # Free the last all-gather result if needed; refer to
                    # [Note: Overlapping all-gather copy-in and all-gather]
                    self._comm_ctx.all_gather_copy_in_stream.wait_event(
                        all_gather_state.event
                    )
                    self._comm_ctx.all_gather_stream.wait_event(all_gather_state.event)
                    self._comm_ctx.all_gather_state = None  # free the all-gather result
                self._state_ctx.iter_forward_root = None
            return self._cast_output_dtype(output)

    def _cast_forward_inputs(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if not (self._mp_policy.cast_forward_inputs and self._mp_policy.param_dtype):
            return args, kwargs
        with torch.profiler.record_function("FSDP::cast_forward_inputs"):
            cast_fn = functools.partial(_cast_fp_tensor, self._mp_policy.param_dtype)
            return _apply_to_tensors(cast_fn, args), _apply_to_tensors(cast_fn, kwargs)

    def _cast_output_dtype(self, output: Any) -> Any:
        if self._mp_policy.output_dtype is None:
            return output
        with torch.profiler.record_function("FSDP::cast_forward_outputs"):
            return _apply_to_tensors(
                functools.partial(_cast_fp_tensor, self._mp_policy.output_dtype),
                output,
            )

    def _force_complete_incomplete_states(self, output: Any) -> Any:
        # Complete post-forward for any state whose group forward did not
        # run all modules (e.g. chunked loss where model.forward skips
        # head). See ``all_states`` init for why we iterate ``reversed``.
        for state in reversed(self._state_ctx.all_states):
            if state is self or not state._modules_to_run_forward:
                continue
            logger.debug("FSDP::force_complete_post_forward")
            for fsdp_param_group in state._fsdp_param_groups:
                output = fsdp_param_group.post_forward(None, None, output)
            output = state._register_pre_backward_hook(output)
            state._training_state = TrainingState.IDLE
            output = state._cast_output_dtype(output)
            state._modules_to_run_forward.clear()
        return output

    @_dynamo_disable
    def _pre_backward(self, grad: torch.Tensor) -> torch.Tensor:
        with _spmd_no_typecheck():
            self._training_state = TrainingState.PRE_BACKWARD
            self._register_root_post_backward_final_callback()
            default_prefetch = len(self._states_to_backward_prefetch) == 0
            for fsdp_param_group in self._fsdp_param_groups:
                fsdp_param_group.pre_backward(default_prefetch)
            for fsdp_state in self._states_to_backward_prefetch:
                # Reverse so higher-indexed groups are prefetched first,
                # matching backward execution order (reverse of forward).
                for target_param_group in reversed(fsdp_state._fsdp_param_groups):
                    FSDPParamGroup._prefetch_unshard(target_param_group, "backward")
            return grad

    @_dynamo_disable
    def _root_post_backward_final_callback(self) -> None:
        logger.debug("FSDP::root_post_backward")
        with torch.profiler.record_function("FSDP::root_post_backward_callback"):
            # Reset per-iteration state. With chunked loss, each standalone
            # per-chunk call repopulates an inner state's
            # ``_modules_to_run_forward`` (and claims ``iter_forward_root``)
            # but the group post-hook never empties it because not all
            # modules run; clear both here so the next iteration starts
            # fresh.
            self._state_ctx.iter_forward_root = None
            for state in self._state_ctx.all_states:
                state._modules_to_run_forward.clear()
                # Reverse so that the last param group (which gates the
                # reduce-scatter wait/clear) fires first, matching the
                # autograd backward order and preserving RS overlap for
                # per-param-mesh modules whose inputs lack gradients.
                for fsdp_param_group in reversed(state._fsdp_param_groups):
                    if fsdp_param_group._training_state != TrainingState.POST_BACKWARD:
                        # Run post-backward in case forward inputs did not require
                        # gradient so the autograd backward did not run
                        fsdp_param_group.post_backward()
                    fsdp_param_group._training_state = TrainingState.IDLE
                state._training_state = TrainingState.IDLE
                if self._state_ctx.is_last_backward:
                    for fsdp_param_group in state._fsdp_param_groups:
                        fsdp_param_group.finalize_backward()
            if self._state_ctx.is_last_backward:
                self._comm_ctx.post_forward_order.clear()
                # Wait on and release any retained reduce-scatter input buffers:
                # the last module's (which no later module's rs_wait clears) and,
                # when set_reduce_scatter_max_input_buffers retains more than
                # one in flight, the rest. The compute stream (which reuses the
                # memory) is ordered past each reduce-scatter first.
                for rs_state in self._comm_ctx.reduce_scatter_states:
                    if rs_state.event is not None:
                        self._device_handle.current_stream().wait_event(rs_state.event)
                self._comm_ctx.reduce_scatter_states.clear()
            self._state_ctx.post_backward_final_callback_queued = False

    def _register_pre_backward_hook(self, output: Any) -> Any:
        if not torch.is_grad_enabled():
            return output
        # In eager, register hooks directly on grad-requiring outputs.
        tensors = collect_grad_tensors(output)
        for t in tensors:
            if t._base is not None:
                cls = ", ".join(type(m).__name__ for m in self._modules) or "?"
                warnings.warn(
                    f"FSDP2-wrapped module ({cls}) returned a view tensor. "
                    "An in-place op on this view (e.g., `x += y`) will silently "
                    "drop the pre-backward hook and skip the all-gather, which "
                    "can cause backward to fail or produce wrong gradients. "
                    "Use out-of-place ops (`out = out + y`, not `out += y`) or "
                    "`.clone()` the output before any in-place op.",
                    UserWarning,
                    stacklevel=2,
                )
            t.register_hook(self._pre_backward)
        if torch._C._are_functorch_transforms_active():
            # Under functorch, some differentiable outputs report
            # requires_grad=False inside the transform even when returned
            # primals participate in outer autograd.
            return _apply_to_tensors(
                lambda t: RegisterPreBackwardFunction.apply(self, t)
                if not t.requires_grad
                and (torch.is_floating_point(t) or torch.is_complex(t))
                else t,
                output,
            )
        return output

    def _register_root_post_backward_final_callback(self):
        if self._state_ctx.post_backward_final_callback_queued:
            return
        self._state_ctx.post_backward_final_callback_queued = True
        Variable._execution_engine.queue_callback(
            self._root_post_backward_final_callback
        )

    def _reset_iter_state(self) -> None:
        # Iteration-wide recovery after a mid-forward or mid-backward
        # exception. Waits on in-flight collectives, reshards every param
        # group, and clears per-iteration trackers so the next forward can
        # start from a clean state. Any in-flight gradients (reduce-scatter
        # results, HSDP partial reduce outputs, grad-accum state) are
        # discarded: the failed iteration is treated as lost.
        if self._is_root is False:
            raise RuntimeError(
                "reset_iter_state must be called on the root FSDP module"
            )
        current_stream = self._device_handle.current_stream()
        if ag_state := self._comm_ctx.all_gather_state:
            if ag_state.event is not None:
                current_stream.wait_event(ag_state.event)
            self._comm_ctx.all_gather_state = None
        for rs_state in self._comm_ctx.reduce_scatter_states:
            if rs_state.event is not None:
                current_stream.wait_event(rs_state.event)
        self._comm_ctx.reduce_scatter_states.clear()
        for event in self._comm_ctx._last_post_reduce_events.values():
            current_stream.wait_event(event)
        self._comm_ctx._last_post_reduce_events.clear()
        self._comm_ctx.post_forward_order.clear()
        for state in self._state_ctx.all_states:
            state._modules_to_run_forward.clear()
            state._training_state = TrainingState.IDLE
            for fsdp_param_group in state._fsdp_param_groups:
                fsdp_param_group._reset_iter_state()
        self._state_ctx.iter_forward_root = None
        self._state_ctx.post_backward_final_callback_queued = False


def _get_module_fsdp_state(module: nn.Module) -> FSDPState | None:
    state = _get_module_state(module)
    if isinstance(state, FSDPState):
        return state
    return None


class RegisterPreBackwardFunction(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(state: FSDPState, output: torch.Tensor):
        return output

    @staticmethod
    def setup_context(ctx, inputs: tuple[Any, ...], output: Any) -> None:
        state, _ = inputs
        ctx.state = state

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        (grad,) = grad_outputs
        return None, ctx.state._pre_backward(grad)

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        # Drop the non-tensor FSDP state tangent; keep tangent backward on the
        # FSDP pre-backward path.
        tangent = grad_inputs[1]
        if tangent is None:
            return None
        return RegisterPreBackwardFunction.apply(ctx.state, tangent)


def _register_group_forward_hooks(
    modules: Sequence[nn.Module],
    pre_hook: Callable,
    post_hook: Callable,
    modules_to_run: set[nn.Module],
    cast_output_dtype: Callable[[Any], Any],
) -> _MultiHandle:
    """
    Registers group forward pre and post-hooks. The pre-hook runs on every
    module pre-forward; downstream state gating ensures one-shot work
    (root setup, post_backward hook registration) fires once per group
    pass. The post-hook runs on every module's post-forward: on the
    partial path (group not yet complete) it applies only the
    ``mp_policy.output_dtype`` cast so standalone per-module callers
    observe the same output dtype semantics as the non-grouped case; on
    the last module it runs the full ``post_hook`` (which reshards,
    registers the pre-backward hook, and itself casts output_dtype). If
    a module never runs forward, the post-hook does not fire for it and
    ``_force_complete_incomplete_states`` finishes the group from the
    root's post-forward.
    """
    modules_set = set(modules)

    @_dynamo_disable
    @functools.wraps(pre_hook)
    def wrapped_pre_hook(*args: Any, **kwargs: Any):
        if len(modules_to_run) == 0:
            modules_to_run.update(modules_set)
        return pre_hook(*args, **kwargs)

    def get_wrapped_post_hook(module: nn.Module):
        @_dynamo_disable
        @functools.wraps(post_hook)
        def wrapped_post_hook(hook_module: nn.Module, input: Any, output: Any) -> Any:
            # Full path fires once, when this invocation completes the
            # group. Otherwise apply only the output_dtype cast so every
            # module in the group (including repeat invocations such as
            # per-chunk standalone head calls) produces output in the
            # mp_policy's output_dtype.
            if module in modules_to_run:
                modules_to_run.remove(module)
                if len(modules_to_run) == 0:
                    return post_hook(hook_module, input, output)
            return cast_output_dtype(output)

        return wrapped_post_hook

    pre_handles = [
        module.register_forward_pre_hook(
            wrapped_pre_hook, prepend=True, with_kwargs=True
        )
        for module in modules
    ]
    post_handles = [
        module.register_forward_hook(
            get_wrapped_post_hook(module), prepend=False, always_call=True
        )
        for module in modules
    ]
    return _MultiHandle(tuple(pre_handles + post_handles))
<|endoftext|># mypy: allow-untyped-defs
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist


_ReduceOp = dist.ReduceOp | dist.ReduceOp.RedOpType


@dataclass(frozen=True)
class MixedPrecisionPolicy:
    """
    This configures FSDP's mixed precision. Unlike autocast, this applies mixed
    precision at the module level, not op level, which means low-precision
    activations are saved for backward and high-to-low-precision casts are
    incurred only at module boundaries.

    FSDP works well with module-level mixed precision since it keeps the
    high-precision sharded parameters in memory anyway. In other words, FSDP
    does not require any extra memory to keep a high-precision copy of the
    parameters for the optimizer step.

    Attributes:
        param_dtype (Optional[torch.dtype]): This specifies the dtype for
            the unsharded parameter and hence the dtype for forward/backward
            computation and the parameter all-gather. If this is ``None``, then
            the unsharded parameter uses the original dtype. The optimizer step
            uses the sharded parameter in the original dtype. (Default:
            ``None``)
        reduce_dtype (Optional[torch.dtype]): This specifies the dtype for
            gradient reduction (i.e. reduce-scatter or all-reduce). If this is
            ``None`` but ``param_dtype`` is not ``None``, then the reduction
            uses the compute dtype. This can be used to run gradient reduction
            in full precision while using low precision for compute. If also
            gradient reduction is disabled via :meth:`set_requires_gradient_sync`,
            then FSDP will accumulate gradients using ``reduce_dtype``.
            (Default: ``None``)
        output_dtype (Optional[torch.dtype]): This specifies the dtype for
            casting floating-point forward outputs. This can be used to
            help implement cases where different modules have different mixed
            precision policies. (Default: ``None``)
        cast_forward_inputs (bool): This specifies whether FSDP should cast the
            forward's floating-point input tensors to ``param_dtype`` or not.
            For grouped ``fully_shard([a, b, ...])``, the cast is applied per
            module, before each module's forward.
    """

    param_dtype: torch.dtype | None = None
    reduce_dtype: torch.dtype | None = None
    output_dtype: torch.dtype | None = None
    cast_forward_inputs: bool = True


class Comm(ABC):
    """
    Interface for communication primitives.
    A primitive primarily needs to handle 3 tasks, namely:

    1. How to allocate memory for communication
       Depending on the goal, an implementation can choose to:
       a. associate each call to a temporary buffer
          (best for flexibility and simplicity)
       b. reuse a persistent buffer for efficiency reasons

    2. Where to allocate memory
       (e.g. NCCL mem pool or regular cuda caching allocator)

    3. What to do/call upon the comm is called
       (see `AllGather` interface as an example)
    """

    @abstractmethod
    def allocate(
        self,
        size: Sequence[int | torch.SymInt],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        This handles the "how to allocate memory" part.

        A default implementation could be simply:

        .. code-block:: python
            with self.mem_pool:
                torch.empty(...)

        Args:
            size (Sequence[Union[int, torch.SymInt]]): size of the tensor buffer
            dtype (torch.dtype): dtype of the tensor buffer
            device (torch.device): which device to allocate the tensor onto
        """
        ...


class AllGather(Comm):
    """
    Interface for all_gather comm primitive
    """

    @abstractmethod
    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = False,
    ) -> dist.Work | None: ...


class ReduceScatter(Comm):
    """
    Interface for reduce_scatter comm primitive
    """

    @abstractmethod
    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        op: _ReduceOp,
        async_op: bool = False,
    ) -> dist.Work | None: ...


@dataclass
class DataParallelMeshDims:
    """
    Specifies which dimensions of a full SPMD :class:`DeviceMesh` correspond to
    data parallelism when using :func:`fully_shard` whose parameters are already
    DTensors on that mesh.

    Attributes:
        shard (Optional[Union[str, tuple[str, ...]]]): Mesh dimension name(s)
            that FSDP shards parameters on. If a tuple of names, those dims
            are flattened into a single shard dimension. At least one of
            ``shard`` and ``replicate`` must be set.
        replicate (Optional[Union[str, tuple[str, ...]]]): Mesh dimension
            name(s) for HSDP or DDP replication. If a tuple of names, those
            dims are flattened into a single replicate dimension.
    """

    shard: str | tuple[str, ...] | None = None
    replicate: str | tuple[str, ...] | None = None

    def __post_init__(self):
        if self.shard is None and self.replicate is None:
            raise ValueError(
                "At least one of shard or replicate must be set in DataParallelMeshDims"
            )

    @property
    def shard_names(self) -> tuple[str, ...]:
        if self.shard is None:
            return ()
        if isinstance(self.shard, str):
            return (self.shard,)
        return tuple(self.shard)

    @property
    def replicate_names(self) -> tuple[str, ...]:
        if self.replicate is None:
            return ()
        if isinstance(self.replicate, str):
            return (self.replicate,)
        return tuple(self.replicate)


@dataclass
class OffloadPolicy:
    """
    This base class represents the policy of no offloading and is only used as
    the default value for the ``offload_policy`` arg.
    """


@dataclass
class CPUOffloadPolicy(OffloadPolicy):
    """
    This offload policy offloads parameters, gradients, and optimizer states to
    CPU. Sharded parameters are copied host-to-device before all-gather. The
    all-gathered parameters are freed according to ``reshard_after_forward``.
    Sharded gradients are copied device-to-host in backward, and the optimizer
    step runs on CPU with CPU optimizer states.

    Attributes:
        pin_memory (bool): Whether to pin sharded parameter and gradient
            memory. Pinning memory allows both more efficient H2D/D2H copies
            and for the copies to overlap with compute. However, the pinned
            memory cannot be used by other processes. Set this to ``False`` if
            you have insufficient CPU memory. (Default: ``True``)
    """

    pin_memory: bool = True
<|endoftext|>import itertools
import logging
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._logging import warning_once
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.tensor import DeviceMesh, DTensor, init_device_mesh
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from ._fsdp_common import (
    _is_composable_with_fsdp,
    DataParallelMeshInfo,
    DDPMeshInfo,
    FSDPMeshInfo,
    HSDPMeshInfo,
)
from ._fsdp_state import _get_module_fsdp_state


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from ._fsdp_api import DataParallelMeshDims, MixedPrecisionPolicy, OffloadPolicy
    from ._fsdp_common import ShardPlacementFnResult
    from ._fsdp_state import FSDPState


logger = logging.getLogger("torch.distributed.fsdp.fully_shard")


def _validate_module(module: nn.Module, func_name: str) -> None:
    """
    Validate that the module can be used with fully_shard or replicate.

    Raises ValueError if the module is a container that doesn't implement forward.
    """
    if (
        isinstance(module, (nn.ModuleList, nn.ModuleDict))
        and module.__class__.forward is nn.Module.forward
    ):
        raise ValueError(
            f"{func_name} does not support containers that do not implement forward: {module}"
        )


def _validate_mesh(
    mesh: "DeviceMesh",
    dp_mesh_dims: "DataParallelMeshDims | None" = None,
) -> None:
    """
    Validate that the mesh can be used with fully_shard.

    When ``dp_mesh_dims`` is provided, validates that the named dims
    exist in the mesh and at least one of shard/replicate is set.
    Otherwise raises ValueError if the mesh is not 1D or 2D.
    """
    if dp_mesh_dims is not None:
        if dp_mesh_dims.shard is None and dp_mesh_dims.replicate is None:
            raise ValueError(
                "At least one of shard or replicate must be set in dp_mesh_dims"
            )
        if mesh.mesh_dim_names is None:
            raise ValueError(
                "mesh must have mesh_dim_names when dp_mesh_dims is provided"
            )
        names_to_check: list[str] = list(dp_mesh_dims.shard_names)
        names_to_check.extend(dp_mesh_dims.replicate_names)
        for name in names_to_check:
            if name not in mesh.mesh_dim_names:
                raise ValueError(
                    f"Mesh dim name '{name}' not found in mesh.mesh_dim_names "
                    f"{mesh.mesh_dim_names}"
                )
        return
    if mesh.ndim not in (1, 2):
        raise ValueError(f"fully_shard expects a 1D or 2D DeviceMesh but got {mesh}")
    if mesh.ndim == 2 and mesh.mesh_dim_names is None:
        raise AssertionError(
            "Please init the 2D mesh for HSDP with mesh_dim_names specified"
        )


def _get_mesh_info(
    mesh: "DeviceMesh",
    dp_mesh_dims: "DataParallelMeshDims | None" = None,
) -> "DataParallelMeshInfo":
    """
    Get the appropriate mesh info for the given mesh.

    When ``dp_mesh_dims`` is provided, extracts the DP submesh from the
    full SPMD mesh and returns FSDPMeshInfo, HSDPMeshInfo, or DDPMeshInfo
    with ``dp_mesh_dims`` set and ``is_spmd_mesh`` as True.

    Returns FSDPMeshInfo for 1D mesh, HSDPMeshInfo for 2D mesh.
    """
    if dp_mesh_dims is not None:
        return _get_mesh_info_from_named_dims(mesh, dp_mesh_dims)
    if mesh.ndim == 1:
        return FSDPMeshInfo(mesh, shard_mesh_dim=0)
    else:
        return HSDPMeshInfo(mesh, shard_mesh_dim=1, replicate_mesh_dim=0)


def _get_mesh_info_from_named_dims(
    mesh: "DeviceMesh",
    dp_mesh_dims: "DataParallelMeshDims",
) -> "DataParallelMeshInfo":
    shard_names = dp_mesh_dims.shard_names
    replicate_names = dp_mesh_dims.replicate_names

    def _get_submesh(names: tuple[str, ...]) -> "DeviceMesh":
        if len(names) == 1:
            return mesh[names[0]]
        # Flatten multi-dim submesh into a single dim so FSDP's internal
        # logic (which expects one shard and/or one replicate dim) works
        # unchanged. This creates a new 1D DeviceMesh and ProcessGroup.
        return mesh[names]._flatten("_".join(names))

    if len(shard_names) == 0:  # DDP
        dp_mesh = _get_submesh(replicate_names)
        return DDPMeshInfo(
            dp_mesh,
            replicate_mesh_dim=0,
            dp_mesh_dims=dp_mesh_dims,
            spmd_mesh=mesh,
        )
    if len(replicate_names) == 0:  # FSDP
        dp_mesh = _get_submesh(shard_names)
        return FSDPMeshInfo(
            dp_mesh,
            shard_mesh_dim=0,
            dp_mesh_dims=dp_mesh_dims,
            spmd_mesh=mesh,
        )
    # HSDP
    shard_mesh = _get_submesh(shard_names)
    replicate_mesh = _get_submesh(replicate_names)
    dp_mesh = DeviceMesh._concatenate([replicate_mesh, shard_mesh])
    return HSDPMeshInfo(
        dp_mesh,
        shard_mesh_dim=1,
        replicate_mesh_dim=0,
        dp_mesh_dims=dp_mesh_dims,
        spmd_mesh=mesh,
    )


def _get_post_forward_mesh_info(
    reshard_after_forward: bool | int, mesh_info: FSDPMeshInfo
) -> FSDPMeshInfo | None:
    shard_mesh_size = mesh_info.shard_mesh_size
    if not isinstance(reshard_after_forward, (bool, int)):
        raise ValueError(
            "reshard_after_forward should be a bool or an int representing the "
            f"group size to reshard to, not {reshard_after_forward}"
        )
    # NOTE: `isinstance(False, int)` returns `True`.
    if not isinstance(reshard_after_forward, bool) and isinstance(
        reshard_after_forward, int
    ):
        if (
            reshard_after_forward < 1
            or reshard_after_forward > shard_mesh_size
            or shard_mesh_size % reshard_after_forward != 0
        ):
            raise ValueError(
                "If passing reshard_after_forward as an int, it should be a "
                f"factor of {shard_mesh_size}, not {reshard_after_forward}"
            )
        elif reshard_after_forward == 1:
            msg = (
                "reshard_after_forward=1 (int) means resharding parameters to world size 1, "
                "instead of reshard_after_forward=True (bool)"
            )
            warning_once(logger, msg, stacklevel=2)
            reshard_after_forward = False
        elif reshard_after_forward == shard_mesh_size:
            reshard_after_forward = True
    post_forward_mesh_info = None
    if reshard_after_forward is True:
        post_forward_mesh_info = mesh_info
    elif reshard_after_forward is not False:  # int case
        # For HSDP, we can flatten the two replicate dims into the 0th dim
        post_forward_mesh_tensor = mesh_info.mesh.mesh.view(-1, reshard_after_forward)
        post_forward_mesh = DeviceMesh(
            mesh_info.mesh.device_type, post_forward_mesh_tensor
        )
        post_forward_mesh_info = HSDPMeshInfo(
            post_forward_mesh, shard_mesh_dim=1, replicate_mesh_dim=0
        )
    return post_forward_mesh_info


def _init_default_mesh(
    mesh_dim_names: tuple[str, ...] | None = None,
) -> DeviceMesh:
    """Default to global CUDA mesh if possible else global CPU mesh."""
    if not dist.distributed_c10d.is_initialized():
        dist.distributed_c10d.init_process_group()
    default_pg = dist.distributed_c10d._get_default_group()
    device = torch._C._get_accelerator()
    mesh = init_device_mesh(
        device.type,
        mesh_shape=(default_pg.size(),),
        mesh_dim_names=mesh_dim_names,
    )
    return mesh


def _init_default_fully_shard_mesh() -> DeviceMesh:
    """Default to global CUDA mesh if possible else global CPU mesh."""
    return _init_default_mesh()


def _get_device_from_mesh(mesh: DeviceMesh) -> torch.device:
    if mesh.device_type == "cpu":
        return torch.device("cpu")
    device_handle = _get_device_handle(mesh.device_type)
    return torch.device(mesh.device_type, device_handle.current_device())


def _ignore_module(
    module: nn.Module,
    ignored_params: set[nn.Parameter],
    ignore_decision: dict[nn.Module, bool],
) -> bool:
    """
    Decide if it is safe to ignore a module for applying fully_shard.
    """
    if module in ignore_decision:
        return ignore_decision[module]

    if len(list(module.buffers(recurse=False))) > 0:
        # Cannot ignore a module with any buffer
        ignore_decision[module] = False
        return False

    for _, param in module.named_parameters(recurse=False):
        if param not in ignored_params:
            # at least one param is not ignored. So this module shouldn't be.
            ignore_decision[module] = False
            return False

    # Need to consider descendants of module
    for child in list(module.children()):
        ignore_child = _ignore_module(child, ignored_params, ignore_decision)
        if not ignore_child:
            # Cannot ignore module if one of its children is not ignored
            ignore_decision[module] = False
            return False

    # Safe to ignore module
    ignore_decision[module] = True
    return True


def _adjust_managed_modules(
    modules: list[nn.Module], ignored_params: set[nn.Parameter]
) -> list[nn.Module]:
    """
    Adjust the given list of managed modules by removing those with all parameters ignored.
    """
    ignore_decision: dict[nn.Module, bool] = {}
    new_modules = []
    for module in modules:
        ignored = _ignore_module(module, ignored_params, ignore_decision)
        if not ignored:
            new_modules.append(module)
    return new_modules


def _get_managed_modules(
    root_modules: tuple[nn.Module, ...],
    ignored_params: set[nn.Parameter] | None = None,
    is_composable_fn: "Callable[[nn.Module], bool] | None" = None,
    get_state_fn: "Callable[[nn.Module], Any] | None" = None,
) -> list[nn.Module]:
    """
    Get the list of managed modules for FSDP/replicate.

    Args:
        root_modules: The root modules to start the search from.
        ignored_params: Parameters to ignore.
        is_composable_fn: Callable to check if a module is composable.
            Defaults to ``_is_composable_with_fsdp``.
        get_state_fn: Callable to get the state of a module.
            Defaults to ``_get_module_fsdp_state``.
    """
    if is_composable_fn is None:
        is_composable_fn = _is_composable_with_fsdp
    if get_state_fn is None:
        get_state_fn = _get_module_fsdp_state

    modules: list[nn.Module] = []
    root_modules_set = set(root_modules)
    # Track visited modules to avoid visiting shared modules multiple times
    visited_modules: set[nn.Module] = set()

    def dfs(module: nn.Module) -> None:
        """
        Runs a DFS to collect managed modules, not recursing into modules with
        a non-composable API or ``fully_shard`` already applied.
        """
        if not is_composable_fn(module):
            return
        elif module not in root_modules_set and get_state_fn(module) is not None:
            return  # nested `fully_shard` module
        visited_modules.add(module)
        for submodule in module.children():
            if submodule not in visited_modules:
                dfs(submodule)
        modules.append(module)

    for root_module in root_modules:
        dfs(root_module)

    if ignored_params is None:
        return modules

    adjusted_modules = _adjust_managed_modules(modules, ignored_params)
    return adjusted_modules


def _verify_managed_param(name: str, param: nn.Parameter) -> None:
    """
    Verify if the parameter is accepted by fully_shard. The only restriction now
    is that the parameter cannot be a scalar tensor (param.numel == 0) since we
    need at least one dim to shard.
    """
    if len(param.shape) == 0:
        raise ValueError(
            "fully_shard doesn't support scalar parameters. "
            f"Change {name} to a 1D tensor with numel equal to 1."
        )


def _get_managed_states(
    modules: list[nn.Module], ignored_params: set[nn.Parameter] | None = None
) -> tuple[list[nn.Parameter], list[torch.Tensor]]:
    params: list[nn.Parameter] = []
    buffers: list[torch.Tensor] = []
    # Track visited parameters/buffers to avoid visiting shared parameters and
    # buffers multiple times
    visited_params: set[nn.Parameter] = set()
    visited_buffers: set[torch.Tensor] = set()
    if ignored_params is None:
        ignored_params = set()

    for module in modules:
        for name, param in module.named_parameters(recurse=False):
            if param in ignored_params:
                # do not include an ignored parameters
                continue
            if param not in visited_params:
                _verify_managed_param(name, param)
                params.append(param)
                visited_params.add(param)
        for buffer in module.buffers(recurse=False):
            if buffer not in visited_buffers:
                buffers.append(buffer)
                visited_buffers.add(buffer)
    return params, buffers


def _move_states_to_device(
    params: list[nn.Parameter],
    buffers: list[torch.Tensor],
    device: torch.device,
) -> None:
    """
    We have FSDP move states to device for simpler and faster initialization
    since FSDP almost always uses CUDA for training. We move parameters/buffers
    rather than modules since modules to support ignoring parameters/buffers in
    the future.
    """
    # Follow the logic in `nn.Module._apply`
    # pyrefly: ignore [bad-argument-type]
    for tensor in itertools.chain(params, buffers):
        if tensor.device == device or tensor.device.type == "meta":
            # Keep meta-device tensors on meta device for deferred init
            continue
        if isinstance(tensor, DTensor):
            if (dtensor_mesh_type := tensor.device_mesh.device_type) != device.type:
                raise ValueError(
                    "Requires DTensor to have mesh of the same type as the FSDP mesh "
                    f"but got {dtensor_mesh_type} for DTensor and {device.type} for FSDP"
                )
            raise AssertionError(
                f"Expects DTensor to be moved to {dtensor_mesh_type} but got {tensor.device}"
            )
        tensor_ = tensor
        if is_traceable_wrapper_subclass(tensor_):
            with torch.no_grad():  # avoid autograd increasing C++ refcount by 1
                tensor_on_device = nn.Parameter(tensor.to(device))
            torch.utils.swap_tensors(tensor, tensor_on_device)
        else:
            tensor.data = tensor.to(device)


def _apply_to_module(
    modules: tuple[nn.Module, ...],
    cls_to_wrapper_cls: dict[type, type],
    wrapper_module_cls: type,
    wrapper_cls_prefix: str,
    unimplemented_deepcopy: "Callable",
) -> None:
    """
    Modify module classes to include the wrapper class in their MRO.

    Args:
        modules: The modules to apply the wrapper to.
        cls_to_wrapper_cls: Cache dict mapping original class to wrapper class.
        wrapper_module_cls: The wrapper module class (e.g., FSDPModule, ReplicateModule).
        wrapper_cls_prefix: Prefix for the dynamically created class name (e.g., "FSDP", "Replicate").
        unimplemented_deepcopy: The deepcopy function to use for the wrapper class.
    """
    for module in modules:
        cls = module.__class__
        new_cls = cls_to_wrapper_cls.get(cls)
        if not new_cls:
            dct = {"__deepcopy__": unimplemented_deepcopy}
            new_cls = type(
                f"{wrapper_cls_prefix}{cls.__name__}", (wrapper_module_cls, cls), dct
            )
            cls_to_wrapper_cls[cls] = new_cls
        module.__class__ = new_cls


def _init_param_group(
    state: "FSDPState",
    params: list[nn.Parameter],
    modules: tuple[nn.Module, ...],
    mesh_info: DataParallelMeshInfo,
    post_forward_mesh_info: FSDPMeshInfo | None,
    device: torch.device,
    shard_placement_fn: "Callable[[nn.Parameter], ShardPlacementFnResult] | None",
    mp_policy: "MixedPrecisionPolicy",
    offload_policy: "OffloadPolicy",
    reshard_after_forward: bool | int = True,
) -> None:
    """
    Initialize FSDP param groups for the given state.

    Params are grouped by their process group (derived from ``mesh_info`` via
    ``shard_placement_fn``). Each group becomes a separate ``FSDPParamGroup``.
    When ``shard_placement_fn`` is ``None`` or returns the same mesh for all
    params, this creates a single group.
    """
    # Import here to avoid circular imports
    from ._fsdp_common import FSDPMeshInfo, resolve_shard_placement
    from ._fsdp_param_group import FSDPParamGroup

    if not params:
        return

    if shard_placement_fn is None:
        # No shard_placement_fn means all params use the same mesh_info,
        # so no grouping is needed. This also handles DDPMeshInfo from
        # replicate_with_fsdp, which doesn't have shard_process_group.
        state._fsdp_param_groups.append(
            FSDPParamGroup(
                params,
                modules,
                mesh_info,
                post_forward_mesh_info,
                device,
                shard_placement_fn,
                mp_policy,
                offload_policy,
            )
        )
        return

    # Group params by their process group to support per-param mesh,
    # e.g., expert params using ep_mesh vs regular params using dp_mesh.
    # For HSDP, also key by replicate_process_group to avoid grouping
    # FSDPMeshInfo params with HSDPMeshInfo params that share the same
    # shard_process_group but require different gradient reduction behavior.
    if not isinstance(mesh_info, FSDPMeshInfo):
        raise ValueError(
            "Per-param mesh via shard_placement_fn is not supported with "
            f"{type(mesh_info).__name__}; it requires FSDPMeshInfo (or subclass)"
        )
    pg_to_group: dict[
        tuple[dist.ProcessGroup, dist.ProcessGroup | None],
        tuple[FSDPMeshInfo, list[nn.Parameter]],
    ] = {}
    for param in params:
        param_mesh_info = resolve_shard_placement(
            shard_placement_fn(param),
            mesh_info,
        ).mesh_info
        shard_pg = param_mesh_info.shard_process_group
        replicate_pg: dist.ProcessGroup | None = None
        if isinstance(param_mesh_info, HSDPMeshInfo):
            replicate_pg = param_mesh_info.replicate_process_group
        key = (shard_pg, replicate_pg)
        if key not in pg_to_group:
            pg_to_group[key] = (param_mesh_info, [param])
        else:
            existing_mesh_info = pg_to_group[key][0]
            if existing_mesh_info is not param_mesh_info:
                raise ValueError(
                    f"Params sharing the same process group must use the same "
                    f"FSDPMeshInfo object, but got different objects: "
                    f"{existing_mesh_info} vs {param_mesh_info}"
                )
            pg_to_group[key][1].append(param)

    # Create a FSDPParamGroup per process group
    for group_mesh_info, group_params in pg_to_group.values():
        if group_mesh_info is not mesh_info:
            group_post_forward = _get_post_forward_mesh_info(
                reshard_after_forward, group_mesh_info
            )
        else:
            group_post_forward = post_forward_mesh_info
        state._fsdp_param_groups.append(
            FSDPParamGroup(
                group_params,
                modules,
                group_mesh_info,
                group_post_forward,
                device,
                shard_placement_fn,
                mp_policy,
                offload_policy,
            )
        )


def _get_modules_and_states(
    module: nn.Module,
    device: torch.device,
    ignored_params: set[nn.Parameter] | None,
    is_composable_fn: "Callable[[nn.Module], bool] | None" = None,
    get_state_fn: "Callable[[nn.Module], Any] | None" = None,
) -> tuple[
    nn.Module,
    tuple[nn.Module, ...],
    list[nn.Module],
    list[nn.Parameter],
    list[torch.Tensor],
]:
    """
    Get modules tuple, managed modules, params, and buffers for FSDP/replicate initialization.

    Returns:
        Tuple of (arg_module, modules, managed_modules, params, buffers)
    """
    from torch.distributed.utils import _get_root_modules

    arg_module = module
    modules = (
        (module,) if isinstance(module, nn.Module) else tuple(_get_root_modules(module))
    )

    managed_modules = _get_managed_modules(
        modules, ignored_params, is_composable_fn, get_state_fn
    )
    params, buffers = _get_managed_states(managed_modules, ignored_params)

    _move_states_to_device(params, buffers, device)

    return arg_module, modules, managed_modules, params, buffers
<|endoftext|>#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed.elastic.rendezvous.registry as rdzv_registry
from torch._utils_internal import get_default_numa_options, justknobs_check
from torch.distributed.elastic import events, metrics
from torch.distributed.elastic.agent.server.api import WorkerSpec
from torch.distributed.elastic.agent.server.health_check_server import (
    create_healthcheck_server,
    HealthCheckServer,
)
from torch.distributed.elastic.agent.server.local_elastic_agent import (
    _AliveCallbackProxy,
    LocalElasticAgent,
    TORCHELASTIC_HEALTH_CHECK_PORT,
)
from torch.distributed.elastic.multiprocessing import (
    DefaultLogsSpecs,
    LogsSpecs,
    SignalException,
)
from torch.distributed.elastic.multiprocessing.errors import ChildFailedError
from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.rendezvous.utils import parse_rendezvous_endpoint
from torch.distributed.elastic.utils.logging import get_logger
from torch.numa.binding import NumaOptions


__all__ = ["LaunchConfig", "elastic_launch", "launch_agent"]

logger = get_logger(__name__)


@dataclass
class LaunchConfig:
    """
    Creates a rendezvous config.

    Args:
        min_nodes: Minimum amount of nodes that the user function will
                        be launched on. Elastic agent ensures that the user
                        function start only when the min_nodes amount enters
                        the rendezvous.
        max_nodes: Maximum amount of nodes that the user function
                        will be launched on.
        nproc_per_node: On each node the elastic agent will launch
                            this amount of workers that will execute user
                            defined function.
        rdzv_backend: rdzv_backend to use in the rendezvous (zeus-adapter, etcd).
        rdzv_endpoint: The endpoint of the rdzv sync. storage.
        rdzv_configs: Key, value pair that specifies rendezvous specific configuration.
        rdzv_timeout: Legacy argument that specifies timeout for the rendezvous. It is going
            to be removed in future versions, see the note below. The default timeout is 900 seconds.
        run_id: The unique run id of the job (if not passed a unique one will be
                deduced from run environment - flow workflow id in flow - or auto generated).
        role: User defined role of the worker (defaults to "trainer").
        max_restarts: The maximum amount of restarts that elastic agent will conduct
                    on workers before failure.
        monitor_interval: The interval in seconds that is used by the elastic_agent
                        as a period of monitoring workers.
        start_method: The method is used by the elastic agent to start the
                    workers (spawn, fork, forkserver).
        metrics_cfg: configuration to initialize metrics.
        local_addr: address of the local node if any. If not set, a lookup on the local
                machine's FQDN will be performed.
        local_ranks_filter: ranks for which to show logs in console. If not set, show from all.
        event_log_handler: name of the event logging handler as registered in
          `elastic/events/handlers.py <https://docs.pytorch.org/docs/stable/elastic/events.html>`_.
        duplicate_stdout_filters: If non-empty, duplicates stdout to a file containing only lines
                                that match _any_ of the filter strings.
        duplicate_stderr_filters: If non-empty, duplicates stderr to a file containing only lines
                                that match _any_ of the filter strings.
        virtual_local_rank: Enable virtual local rank mode for workers (defaults to False).
                           When enabled, LOCAL_RANK is set to 0 for all workers and
                           CUDA_VISIBLE_DEVICES is adjusted so each worker accesses its
                           assigned GPU at device index 0.
        shutdown_timeout: Time in seconds to wait for graceful shutdown of workers before
                        sending SIGKILL. Can also be set via TORCH_ELASTIC_SHUTDOWN_TIMEOUT
                        environment variable. Defaults to 30 seconds.


    .. note::
        `rdzv_timeout` is a legacy argument that will be removed in future.
        Set the timeout via `rdzv_configs['timeout']`

    """

    min_nodes: int
    max_nodes: int
    nproc_per_node: int
    logs_specs: LogsSpecs | None = None
    run_id: str = ""
    role: str = "default_role"
    rdzv_endpoint: str = ""
    rdzv_backend: str = "etcd"
    rdzv_configs: dict[str, Any] = field(default_factory=dict)
    rdzv_timeout: int = -1
    max_restarts: int = 3
    monitor_interval: float = 0.1
    start_method: str = "spawn"
    log_line_prefix_template: str | None = None
    metrics_cfg: dict[str, str] = field(default_factory=dict)
    local_addr: str | None = None
    event_log_handler: str = "null"
    numa_options: NumaOptions | None = None
    signals_to_handle: str = "SIGTERM,SIGINT,SIGHUP,SIGQUIT"
    duplicate_stdout_filters: list[str] | None = None
    duplicate_stderr_filters: list[str] | None = None
    virtual_local_rank: bool = False
    shutdown_timeout: int | None = None

    def __post_init__(self):
        default_timeout = 900
        if self.rdzv_timeout != -1:
            self.rdzv_configs["timeout"] = self.rdzv_timeout
        elif "timeout" not in self.rdzv_configs:
            self.rdzv_configs["timeout"] = default_timeout

        # Post-processing to enable refactoring to introduce logs_specs due to non-torchrun API usage
        if self.logs_specs is None:
            self.logs_specs = DefaultLogsSpecs()

        if self.numa_options is None:
            try:
                should_use_default_numa = (
                    torch.accelerator.is_available()
                    and torch.accelerator.device_count() == self.nproc_per_node
                )
            except RuntimeError:
                should_use_default_numa = False
            if should_use_default_numa:
                self.numa_options = get_default_numa_options()
                logger.info("Using default numa options = %r", self.numa_options)

        # Set shutdown_timeout from environment variable if not explicitly set
        if self.shutdown_timeout is None:
            self.shutdown_timeout = int(
                os.environ.get("TORCH_ELASTIC_SHUTDOWN_TIMEOUT", "30")
            )
        elif self.shutdown_timeout < 0:
            raise ValueError(
                f"shutdown_timeout must be non-negative, got {self.shutdown_timeout}"
            )


class elastic_launch:
    """
    Launches a torchelastic agent on the container that invoked the entrypoint.

        1. Pass the ``entrypoint`` arguments as non ``kwargs`` (e.g. no named parameters)/
           ``entrypoint`` can be a function or a command.
        2. The return value is a map of each worker's output mapped
           by their respective global rank.

    Usage

    ::

    def worker_fn(foo):
        # ...

    def main():
        # entrypoint is a function.
        outputs = elastic_launch(LaunchConfig, worker_fn)(foo)
        # return rank 0's output
        return outputs[0]

        # entrypoint is a command and ``script.py`` is the python module.
        outputs = elastic_launch(LaunchConfig, "script.py")(args)
        outputs = elastic_launch(LaunchConfig, "python")("script.py")
    """

    def __init__(
        self,
        config: LaunchConfig,
        entrypoint: Callable | str | None,
        health_check_server: HealthCheckServer | None = None,
    ):
        self._config = config
        self._entrypoint = entrypoint
        self._health_check_server = health_check_server

    def __call__(self, *args):
        return launch_agent(
            self._config,
            self._entrypoint,
            list(args),
            health_check_server=self._health_check_server,
        )


def _get_entrypoint_name(entrypoint: Callable | str | None, args: list[Any]) -> str:
    """Retrieve entrypoint name with the rule:
    1. If entrypoint is a function, use ``entrypoint.__qualname__``.
    2. If entrypoint is a string, check its value:
        2.1 if entrypoint equals to ``sys.executable`` (like "python"), use the first element from ``args``
            which does not start with hyphen letter (for example, "-u" will be skipped).
        2.2 otherwise, use ``entrypoint`` value.
    3. Otherwise, return empty string.
    """
    if isinstance(entrypoint, Callable):  # type: ignore[arg-type]
        return entrypoint.__name__  # type: ignore[union-attr]
    elif isinstance(entrypoint, str):
        if entrypoint == sys.executable:
            return next((arg for arg in args if arg[0] != "-"), "")
        else:
            return entrypoint
    else:
        return ""


def _get_addr_and_port(
    rdzv_parameters: RendezvousParameters,
) -> tuple[str | None, int | None]:
    if rdzv_parameters.backend != "static":
        return (None, None)
    endpoint = rdzv_parameters.endpoint
    endpoint = endpoint.strip()
    if not endpoint:
        raise ValueError(
            "Endpoint is missing in endpoint. Try to add --master-addr and --master-port"
        )
    master_addr, master_port = parse_rendezvous_endpoint(endpoint, default_port=-1)
    if master_port == -1:
        raise ValueError(
            f"port is missing in endpoint: {endpoint}. Try to specify --master-port"
        )
    return (master_addr, master_port)


def launch_agent(
    config: LaunchConfig,
    entrypoint: Callable | str | None,
    args: list[Any],
    health_check_server: HealthCheckServer | None = None,
) -> dict[int, Any]:
    if not config.run_id:
        run_id = str(uuid.uuid4().int)
        logger.warning("config has no run_id, generated a random run_id: %s", run_id)
        config.run_id = run_id

    entrypoint_name = _get_entrypoint_name(entrypoint, args)

    logger.info(
        "Starting elastic_operator with launch configs:\n"
        "  entrypoint               : %(entrypoint)s\n"
        "  min_nodes                : %(min_nodes)s\n"
        "  max_nodes                : %(max_nodes)s\n"
        "  nproc_per_node           : %(nproc_per_node)s\n"
        "  run_id                   : %(run_id)s\n"
        "  rdzv_backend             : %(rdzv_backend)s\n"
        "  rdzv_endpoint            : %(rdzv_endpoint)s\n"
        "  rdzv_configs             : %(rdzv_configs)s\n"
        "  max_restarts             : %(max_restarts)s\n"
        "  monitor_interval         : %(monitor_interval)s\n"
        "  log_dir                  : %(log_dir)s\n"
        "  metrics_cfg              : %(metrics_cfg)s\n"
        "  event_log_handler        : %(event_log_handler)s\n"
        "  numa_options             : %(numa_options)s\n"
        "  signals_to_handle        : %(signals_to_handle)s\n"
        "  duplicate_stdout_filters : %(duplicate_stdout_filters)s\n"
        "  duplicate_stderr_filters : %(duplicate_stderr_filters)s\n",
        {
            "entrypoint": entrypoint_name,
            "min_nodes": config.min_nodes,
            "max_nodes": config.max_nodes,
            "nproc_per_node": config.nproc_per_node,
            "run_id": config.run_id,
            "rdzv_backend": config.rdzv_backend,
            "rdzv_endpoint": config.rdzv_endpoint,
            "rdzv_configs": config.rdzv_configs,
            "max_restarts": config.max_restarts,
            "monitor_interval": config.monitor_interval,
            "log_dir": config.logs_specs.root_log_dir,  # type: ignore[union-attr]
            "metrics_cfg": config.metrics_cfg,
            "event_log_handler": config.event_log_handler,
            "numa_options": config.numa_options,
            "signals_to_handle": config.signals_to_handle,
            "duplicate_stdout_filters": config.duplicate_stdout_filters,
            "duplicate_stderr_filters": config.duplicate_stderr_filters,
        },
    )

    rdzv_parameters = RendezvousParameters(
        backend=config.rdzv_backend,
        endpoint=config.rdzv_endpoint,
        run_id=config.run_id,
        min_nodes=config.min_nodes,
        max_nodes=config.max_nodes,
        local_addr=config.local_addr,
        **config.rdzv_configs,
    )

    master_addr, master_port = _get_addr_and_port(rdzv_parameters)

    # Set the signals to handle in the environment variable
    os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"] = config.signals_to_handle

    # Start health check server before rendezvous so TW sees a healthy
    # thrift port during the potentially long MAST rendezvous store barrier
    # (10-22+ min for large jobs).  The _AliveCallbackProxy returns
    # time.time() until wired to the agent after construction.
    # Skip if a server was already provided by the caller (e.g. started
    # before remote_pre_launch in the APF executor).
    if health_check_server is None:
        healthcheck_port = os.getenv(TORCHELASTIC_HEALTH_CHECK_PORT)
        if healthcheck_port is not None and justknobs_check(
            "ai_infra/pytorch_distributed:torchelastic_enable_healthcheck_before_rendezvous",
            default=False,
        ):
            try:
                health_check_server = create_healthcheck_server(
                    alive_callback=_AliveCallbackProxy(),
                    port=int(healthcheck_port),
                    timeout=60,
                )
                health_check_server.start()
                logger.info(
                    "Started early health check server on port %s before rendezvous",
                    healthcheck_port,
                )
            except Exception:
                logger.warning(
                    "Failed to start early health check server", exc_info=True
                )
                health_check_server = None

    spec = WorkerSpec(
        role=config.role,
        local_world_size=config.nproc_per_node,
        entrypoint=entrypoint,
        args=tuple(args),
        rdzv_handler=rdzv_registry.get_rendezvous_handler(rdzv_parameters),
        max_restarts=config.max_restarts,
        monitor_interval=config.monitor_interval,
        master_addr=master_addr,
        master_port=master_port,
        local_addr=config.local_addr,
        event_log_handler=config.event_log_handler,
        numa_options=config.numa_options,
        duplicate_stdout_filters=config.duplicate_stdout_filters,
        duplicate_stderr_filters=config.duplicate_stderr_filters,
        virtual_local_rank=config.virtual_local_rank,
    )

    agent = LocalElasticAgent(
        spec=spec,
        logs_specs=config.logs_specs,  # type: ignore[arg-type]
        start_method=config.start_method,
        log_line_prefix_template=config.log_line_prefix_template,
        shutdown_timeout=config.shutdown_timeout,  # type: ignore[arg-type]
        health_check_server=health_check_server,
    )

    if health_check_server is not None:
        cb = health_check_server.alive_callback
        if isinstance(cb, _AliveCallbackProxy):
            cb.set_delegate(agent._get_alive_time)

    shutdown_rdzv = True
    try:
        metrics.initialize_metrics(metrics.MetricsConfig(config.metrics_cfg))

        result = agent.run()
        # records that agent.run() has succeeded NOT that workers have succeeded
        events.record(agent.get_event_succeeded(), config.event_log_handler)

        if result.is_failed():
            # ChildFailedError is treated specially by @record
            # if the error files for the failed children exist
            # @record will copy the first error (root cause)
            # to the error file of the launcher process.
            raise ChildFailedError(
                name=entrypoint_name,
                failures=result.failures,
            )

        return result.return_values
    except ChildFailedError:
        raise
    except SignalException:
        # when the agent dies with a signal do NOT shutdown the rdzv_handler
        # since this closes the rendezvous on this rdzv_id permanently and
        # prevents any additional scaling events
        shutdown_rdzv = False
        events.record(agent.get_event_failed(), config.event_log_handler)
        raise
    except Exception:
        events.record(agent.get_event_failed(), config.event_log_handler)
        raise
    finally:
        if shutdown_rdzv:
            spec.rdzv_handler.shutdown()
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates

import torch
from torch import SymInt

from ..device_mesh import DeviceMesh


# Register custom operator
torch.library.define(
    "device_mesh::_runtime_compute_coordinate_on_dim",
    "(Tensor full_mesh, int index) -> SymInt",
    tags=torch.Tag.pt2_compliant_tag,
)


@torch.library.register_fake("device_mesh::_runtime_compute_coordinate_on_dim")
def _runtime_compute_coordinate_on_dim_fake(
    full_mesh: torch.Tensor, index: int
) -> SymInt:
    from torch.fx.experimental.symbolic_shapes import _constrain_range_for_size

    ctx = torch._custom_op.impl.get_ctx()
    shape_env = ctx._shape_env

    # Bypass allow_dynamic_output_shape_ops check by directly creating the symint.
    # This is intentional - the coordinate is always valid and bounded.
    sz = shape_env.create_unbacked_symint()

    # Apply size constraints - coordinate is bounded by mesh size on the given dimension.
    # The full_mesh tensor has an extra batch dimension at the front, so the actual
    # mesh dimensions start at index 1. mesh.size(index) = full_mesh.size(index + 1)
    mesh_size = full_mesh.size(index + 1)
    _constrain_range_for_size(
        sz, min=0, max=mesh_size - 1 if isinstance(mesh_size, int) else None
    )

    try:
        # Check if we're currently tracing in dynamo (as opposed to AOT or export).
        in_dynamo = torch._dynamo.symbolic_convert.InstructionTranslator.current_tx()
    except AttributeError:
        in_dynamo = False

    if in_dynamo:
        # During dynamo tracing, distributed ops are treated as atomic - so the
        # rank SymInt may be computed but not traced into the graph (e.g., it
        # affects tensor values but not shapes). Mark it as ignorable here;
        # when we decompose these ops later (after dynamo), we'll create fresh
        # SymInts that do get traced.
        shape_env.ignorable_fresh_unbacked_symbols.append(sz.node._expr)

    return sz


@torch.library.impl(
    "device_mesh::_runtime_compute_coordinate_on_dim", "CompositeExplicitAutograd"
)
def _runtime_compute_coordinate_on_dim_impl(full_mesh: torch.Tensor, index: int) -> int:
    rank = torch.distributed.get_rank()
    mesh = DeviceMesh._get_mesh_tensor_from_full_mesh(full_mesh)
    mesh_coords = DeviceMesh._compute_coordinates_from_mesh(mesh, rank)
    if mesh_coords is None:
        raise AssertionError
    return mesh_coords[index]


def _get_flattened_submesh_impl(mesh: DeviceMesh, mesh_dims: list[int]) -> DeviceMesh:
    from torch.distributed.tensor._redistribute import (
        _get_flattened_mesh_by_layout_impl,
    )

    result = _get_flattened_mesh_by_layout_impl(mesh, tuple(mesh_dims))
    if result is None:
        raise ValueError(f"No flattened mesh found for mesh_dims={mesh_dims} on {mesh}")
    return result


@torch.library.custom_op("device_mesh::_get_flattened_submesh", mutates_args=())
def _get_flattened_submesh(mesh: DeviceMesh, mesh_dims: list[int]) -> DeviceMesh:
    return _get_flattened_submesh_impl(mesh, mesh_dims)


@_get_flattened_submesh.register_fake
def _get_flattened_submesh_fake(mesh: DeviceMesh, mesh_dims: list[int]) -> DeviceMesh:
    return _get_flattened_submesh_impl(mesh, mesh_dims)


def _get_submesh_impl(mesh: DeviceMesh, mesh_dims: list[int]) -> DeviceMesh:
    all_dim_names = mesh._mesh_dim_names
    if all_dim_names is None:
        raise ValueError(f"Cannot slice mesh without dim names: {mesh}")
    dim_names = tuple(all_dim_names[i] for i in mesh_dims)
    if len(dim_names) == 1:
        return mesh[dim_names[0]]
    return mesh[dim_names]


@torch.library.custom_op("device_mesh::_get_submesh", mutates_args=())
def _get_submesh(mesh: DeviceMesh, mesh_dims: list[int]) -> DeviceMesh:
    return _get_submesh_impl(mesh, mesh_dims)


@_get_submesh.register_fake
def _get_submesh_fake(mesh: DeviceMesh, mesh_dims: list[int]) -> DeviceMesh:
    return _get_submesh_impl(mesh, mesh_dims)
<|endoftext|>import logging
import os
import subprocess
import sysconfig
from typing import Any

import torch.distributed as dist
from torch.utils._triton import has_triton


logger = logging.getLogger(__name__)


class NvshmemLibFinder:
    """
    A class to find path to the NVSHMEM device library.

    Environment variable:

    `NVSHMEM_LIB_DIR` (Optional[str]): The directory where the NVSHMEM device
    library is located. If not provided, it will use the default path where
    NVSHMEM wheel is installed, or search for the library in common system
    paths.
    """

    # Class variable to store the found library path for reuse
    found_device_lib_path: str | None = None

    @classmethod
    def find_device_library(cls) -> str:
        """
        Find the path to the NVSHMEM device library.

        Returns:
            str: The path to libnvshmem_device.bc (included).
        """
        if cls.found_device_lib_path is not None:
            # Return the cached path if it exists
            return cls.found_device_lib_path

        # First, check if the user has specified a custom library path
        user_lib_dir = os.environ.get("NVSHMEM_LIB_DIR", None)
        if user_lib_dir is not None:
            lib_path = os.path.join(user_lib_dir, "libnvshmem_device.bc")
            if not os.path.exists(lib_path):
                raise RuntimeError(
                    f"NVSHMEM device library not found at specified path: {user_lib_dir}"
                )
            cls.found_device_lib_path = lib_path
            return lib_path

        # Otherwise, search for the library in the default installation paths
        paths = [
            os.path.join(sysconfig.get_path("purelib"), "nvidia", "nvshmem", "lib")
        ]

        # Add common system installation paths
        common_paths = [
            "/usr/local/lib",
            "/usr/lib",
            "/opt/nvidia/nvshmem/lib",
        ]
        paths.extend(common_paths)

        try:
            import torch

            torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            so_path = os.path.join(torch_lib, "libtorch_nvshmem.so")

            if os.path.exists(so_path):
                try:
                    result = subprocess.run(
                        ["readelf", "-d", so_path],
                        capture_output=True,
                        text=True,
                        check=True,
                    )

                    for line in result.stdout.splitlines():
                        if ("RPATH" in line or "RUNPATH" in line) and "[" in line:
                            rpath = line.split("[", 1)[1].split("]", 1)[0]
                            for p in rpath.split(":"):
                                p = p.strip().replace("$ORIGIN", torch_lib)
                                if p and p not in paths:
                                    paths.append(p)
                except subprocess.CalledProcessError:
                    pass

        except ImportError:
            pass

        for path in paths:
            device_lib = os.path.join(path, "libnvshmem_device.bc")
            if os.path.exists(device_lib):
                cls.found_device_lib_path = device_lib
                return device_lib

        raise RuntimeError(f"NVSHMEM device library not found. Searched: {paths}")


def enable_triton(lib_dir: str | None = None) -> dict[str, str]:
    raise NotImplementedError(
        "`enable_triton` is deprecated. "
        "If you need NVSHMEM device function support for Triton, "
        "please use `@requires_nvshmem` to decorate your Triton kernel. ",
    )


class NvshmemKernelRegistry:
    """
    A class to register kernel functions that ** require NVSHMEM initialization **
    """

    # Class variable to store the functions to be initialized
    _to_init: dict[str, Any] = {}

    @classmethod
    def register(cls, name: str) -> None:
        """
        Register a kernel function with the given name.

        Args:
            name (str): The name of the kernel function.
        """
        cls._to_init.setdefault(name)

    @classmethod
    def deregister(cls, name: str) -> None:
        """
        Deregister a kernel function with the given name.

        Args:
            name (str): The name of the kernel function.
        """
        cls._to_init.pop(name, None)

    @classmethod
    def has(cls, name: str) -> bool:
        """
        Check if a kernel function with the given name is registered.

        Args:
            name (str): The name of the kernel function.

        Returns:
            bool: True if the kernel function is registered, False otherwise.
        """
        return name in cls._to_init


def _nvshmem_init_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    """
    A hook function to initialize the CUModule created by `triton.jit` with
    NVSHMEM device context
    """
    from torch._C._distributed_c10d import _nvshmemx_cumodule_init

    jit_function = kwargs["fn"].jit_function
    fn_name = jit_function.fn.__name__

    # Only initialize NVSHMEM module for kernels registered via @requires_nvshmem
    if NvshmemKernelRegistry.has(fn_name):
        key = kwargs["key"]
        device = kwargs["compile"]["device"]
        jit_function = kwargs["fn"].jit_function
        kernel_cache = jit_function.device_caches[device][0]
        kernel = kernel_cache.get(key, None)
        if kernel is not None:
            _ = kernel.run
            # Initialize NVSHMEM for the CU module
            _nvshmemx_cumodule_init(kernel.module)
        else:
            logger.warning(
                f"It seems Triton hasn't created a kernel for function {fn_name}. "  # noqa: G004
                "Please report this issue to Triton."
            )


if has_triton():
    from triton.runtime.jit import JITFunction, KernelInterface

    # Create a new Callable class that follows the KernelInterface protocol so
    # that the Callable works with the subscript operator, e.g. `foo[(1, 1)]`
    class GridCallableWithExtern(KernelInterface):
        """
        `KernelInterface` invokes `self.run` in `__getitem__`, i.e. [].  We
        implement a `run` method by directing the call to `JITFunction.run`,
        with added extern_libs kwarg, so that users don't have to pass it
        """

        def __init__(self, jit_func: JITFunction, extern_libs: dict[str, str]) -> None:
            self.jit_func = jit_func
            self.extern_libs = extern_libs

        def run(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            # Call the JITFunction.run with added extern_libs kwarg
            return self.jit_func.run(*args, **kwargs, extern_libs=self.extern_libs)


def requires_nvshmem(  # type: ignore[no-untyped-def]
    jit_func,  # JITFunction created by triton.jit
):
    """
    A decorator to register a Triton kernel function that requires NVSHMEM initialization.

    Example usage:
    ```
        @requires_nvshmem
        @triton.jit
        def foo(...):
            ...
    ```

    If you would like to specify a path to the NVSHMEM device library other
    than standard search locations, you can use the following environment
    variable:
    ```
        export NVSHMEM_LIB_DIR=/path/to/nvshmem/lib
    ```
    """

    import triton
    from triton.runtime.jit import JITFunction

    if not isinstance(jit_func, JITFunction):
        raise TypeError(f"Expected a JITFunction, but got {type(jit_func)}")

    # Find the NVSHMEM device library
    lib_path = NvshmemLibFinder.find_device_library()
    extern_libs = {"libnvshmem_device": lib_path}

    # Register the JITFunction with the kernel registry as "to be initialized"
    NvshmemKernelRegistry.register(jit_func.fn.__name__)

    # Register the NVSHMEM init function as a post-compile hook.
    # [Note] This is a global setting (due to lack of Triton API exposure). To
    # avoid initializing Triton kernels that do not require NVSHMEM, filtering
    # is performed in the hook function itself by checking against
    # NvshmemKernelRegistry.
    triton.knobs.runtime.jit_post_compile_hook = _nvshmem_init_hook

    return GridCallableWithExtern(jit_func, extern_libs)


if has_triton():
    import triton
    import triton.language as tl
    from triton.language import core

    @triton.jit  # type: ignore[misc]
    def put(dest, source, nelems, pe):  # type: ignore[no-untyped-def]
        """
        Put tensor data from local PE to a remote PE.

        This high-level function provides a tensor-aware interface for NVSHMEM put
        operations. It automatically handles type checking and size calculations, making
        the API more ergonomic and type-safe.

        Args:
            dest: Destination tensor on the remote PE. Type must match source.
            source: Source tensor on the local PE containing data to be copied.
            nelems: Number of elements to transfer.
            pe: PE number of the remote PE (0 ≤ pe < nvshmem_n_pes()).

        Notes:
            - Performs compile-time type checking between dest and source tensors.
            - Automatically calculates byte size from tensor type and element count.
            - This is a blocking operation that returns after data has been copied out
              of the source array on the local PE.
            - The operation does not guarantee delivery to the destination PE.
              Use nvshmem_fence() for ordering or nvshmem_quiet() for completion.

        Example:
            ```
            # Transfer 100 elements to PE 1
            nvshmem.put(dest_tensor, src_tensor, 100, 1)
            ```
        """
        tl.static_assert(dest.type == source.type)
        nbytes = nelems * dest.type.element_ty.itemsize
        return putmem_block_extern_wrapper(
            dest.to(tl.int64), source.to(tl.int64), nbytes.to(tl.int64), pe
        )

    @core.extern
    def putmem_block_extern_wrapper(dest, source, size_bytes, pe, _semantic=None):  # type: ignore[no-untyped-def]
        """Low-level extern wrapper for NVSHMEM put"""
        return core.extern_elementwise(
            "",
            "",
            [dest, source, size_bytes, pe],
            {
                (
                    core.dtype("int64"),  # dest ptr
                    core.dtype("int64"),  # source ptr
                    core.dtype("int64"),  # size in bytes
                    core.dtype("int32"),  # pe number
                ): ("nvshmemx_putmem_block", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @triton.jit  # type: ignore[misc]
    def get(dest, source, nelems, pe):  # type: ignore[no-untyped-def]
        """
        Get tensor data from a remote PE to local PE.

        This high-level function provides a tensor-aware interface for NVSHMEM get
        operations. It automatically handles type checking and size calculations, making
        the API more ergonomic and type-safe.

        Args:
            dest: Destination tensor on the local PE. Type must match source.
            source: Source tensor on the remote PE containing data to be copied.
            nelems: Number of elements to transfer.
            pe: PE number of the remote PE (0 ≤ pe < nvshmem_n_pes()).

        Notes:
            - Performs compile-time type checking between dest and source tensors.
            - Automatically calculates byte size from tensor type and element count.
            - This is a blocking operation that returns after data has been delivered
              to the destination array on the local PE.
            - The destination data is guaranteed to be available for use after the call returns.

        Example:
            ```
            # Get 100 elements from PE 0
            nvshmem.get(dest_tensor, src_tensor, 100, 0)
            ```
        """
        tl.static_assert(dest.type == source.type)
        nbytes = nelems * dest.type.element_ty.itemsize
        return getmem_block_extern_wrapper(
            dest.to(tl.int64), source.to(tl.int64), nbytes.to(tl.int64), pe
        )

    @core.extern
    def getmem_block_extern_wrapper(dest, source, size_bytes, pe, _semantic=None):  # type: ignore[no-untyped-def]
        """Low-level extern wrapper for NVSHMEM get"""
        return core.extern_elementwise(
            "",
            "",
            [dest, source, size_bytes, pe],
            {
                (
                    core.dtype("int64"),  # dest ptr
                    core.dtype("int64"),  # source ptr
                    core.dtype("int64"),  # size in bytes
                    core.dtype("int32"),  # pe number
                ): ("nvshmemx_getmem_block", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @triton.jit  # type: ignore[misc]
    def get_nbi(dest, source, nelems, pe):  # type: ignore[no-untyped-def]
        """
        Get tensor data from a remote PE to local PE, non-blocking.

        Different from the `get` function, this function returns after
        initiating the operation. The operation is considered complete after a
        subsequent call to `quiet`.

        Args:
            dest: Destination tensor on the local PE. Type must match source.
            source: Source tensor on the remote PE containing data to be copied.
            nelems: Number of elements to transfer.
            pe: PE number of the remote PE (0 ≤ pe < nvshmem_n_pes()).

        Notes:
            - Performs compile-time type checking between dest and source tensors.
            - Automatically calculates byte size from tensor type and element count.

        Example:
            ```
            # Get 100 elements from PE 0
            nvshmem.get_nbi(dest, src, 100, 0)
            # Some independent computation which overlaps with the get operation
            ...
            # Wait for completion of the get operation
            nvshmem.quiet()
            ```
        """
        tl.static_assert(dest.type == source.type)
        nbytes = nelems * dest.type.element_ty.itemsize
        return getmem_block_extern_wrapper(
            dest.to(tl.int64), source.to(tl.int64), nbytes.to(tl.int64), pe
        )

    @core.extern
    def getmem_nbi_block_extern_wrapper(dest, source, size_bytes, pe, _semantic=None):  # type: ignore[no-untyped-def]
        """Low-level extern wrapper for NVSHMEM get"""
        return core.extern_elementwise(
            "",
            "",
            [dest, source, size_bytes, pe],
            {
                (
                    core.dtype("int64"),  # dest ptr
                    core.dtype("int64"),  # source ptr
                    core.dtype("int64"),  # size in bytes
                    core.dtype("int32"),  # pe number
                ): ("nvshmemx_getmem_nbi_block", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @triton.jit  # type: ignore[misc]
    def putmem_signal_block(  # type: ignore[no-untyped-def]
        dst,
        src,
        size_bytes,
        signal,
        sig_val,
        sig_op,
        pe,
    ):  # type: ignore[no-untyped-def]
        """
        Put data to remote PE with atomic signal operation using block-scoped operation.

        This function copies data from the local PE to the remote PE and then
        atomically updates a signal variable on the remote PE to indicate completion.
        This enables efficient point-to-point synchronization between PEs.

        Args:
            dst (tensor): A tensor on calling PE symmetric to the destination tensor on remote PE.
            src (tensor): Local tensor containing the source data.
            size_bytes (int64): Number of bytes to transfer. Must be positive.
            signal (tensor): Symmetric signal pad with remote PE.
                             Must be 8-byte aligned symmetric memory.
            signal (int64): Value to be used in the signal operation.
            sig_op (int32): Signal operation type. Common values:
                           - NVSHMEM_SIGNAL_SET (0): Atomic set operation
                           - NVSHMEM_SIGNAL_ADD (5): Atomic add operation
            pe (int32): PE number of the remote PE (0 ≤ pe < nvshmem_n_pes()).

        Returns:
            int32: Status code (0 for success).

        Notes:
            - This is a blocking operation that returns after data has been copied out
              of the source array and the signal has been updated on the remote PE.
            - The signal update is performed atomically with respect to other signal
              operations and synchronization routines.
            - The signal variable must be of type uint64_t in symmetric memory.
            - Use with nvshmem_signal_wait_until() for synchronization.

        Example:
            ```
            # Transfer data and set completion flag to 1
            NVSHMEM_SIGNAL_SET = 0
            nvshmem.putmem_signal_block(
                dst_ptr, src_ptr, 1024, sig_ptr, 1, NVSHMEM_SIGNAL_SET, target_pe
            )
            ```
        """
        # Ensure sig_val is 64 bits
        sig_val = 0 << 32 | sig_val
        return putmem_signal_block_extern_wrapper(
            dst.to(tl.int64),
            src.to(tl.int64),
            size_bytes.to(tl.int64),
            signal.to(tl.int64),
            sig_val.to(tl.uint64),
            sig_op,
            pe,
        )

    @core.extern
    def putmem_signal_block_extern_wrapper(  # type: ignore[no-untyped-def]
        dst,
        src,
        size_bytes,
        signal,
        sig_val,
        sig_op,
        pe,
        _semantic=None,
    ):  # type: ignore[no-untyped-def]
        return core.extern_elementwise(
            "",
            "",
            [dst, src, size_bytes, signal, sig_val, sig_op, pe],
            {
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("uint64"),
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("nvshmemx_putmem_signal_block", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    # Wait and Signal Operations

    @triton.jit  # type: ignore[misc]
    def wait_until(ivar, cmp_op, cmp_val):  # type: ignore[no-untyped-def]
        """
        Wait until a tensor variable meets a specified condition.

        This high-level function provides a tensor-aware interface for NVSHMEM wait_until
        operations. It automatically handles tensor address extraction, making
        the API more ergonomic and type-safe.

        Args:
            ivar_tensor: Tensor to monitor (typically int64/uint64) in symmetric memory.
            cmp: Comparison operator. Common values:
                 - NVSHMEM_CMP_EQ (0): Wait until ivar == cmp_val
                 - NVSHMEM_CMP_NE (1): Wait until ivar != cmp_val
                 - NVSHMEM_CMP_GT (2): Wait until ivar > cmp_val
                 - NVSHMEM_CMP_GE (3): Wait until ivar >= cmp_val
                 - NVSHMEM_CMP_LT (4): Wait until ivar < cmp_val
                 - NVSHMEM_CMP_LE (5): Wait until ivar <= cmp_val
            cmp_val: Value to compare against.

        Notes:
            - This is a blocking operation that will wait indefinitely until the
              condition is satisfied.
            - The tensor must be in symmetric memory and accessible from other PEs.

        Example:
            ```
            # Wait until flag tensor becomes 1 (set by another PE)
            NVSHMEM_CMP_EQ = 0
            nvshmem.wait_until_tensor(flag_tensor, NVSHMEM_CMP_EQ, 1)
            ```
        """
        tl.static_assert(
            ivar.type.element_ty.itemsize == 4,
            "wait_until expects a 32-bit type for the synchronization variable",
        )
        return wait_until_extern_wrapper(ivar.to(tl.int64), cmp_op, cmp_val)

    @core.extern
    def wait_until_extern_wrapper(ivar, cmp, cmp_val, _semantic=None):  # type: ignore[no-untyped-def]
        return core.extern_elementwise(
            "",
            "",
            [ivar, cmp, cmp_val],
            {
                (
                    core.dtype("int64"),
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("nvshmem_int_wait_until", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @triton.jit  # type: ignore[misc]
    def signal_wait_until(signal, cmp, cmp_val):  # type: ignore[no-untyped-def]
        """
        Wait until a signal variable meets a specified condition.

        This function blocks the calling thread until the value at the specified
        signal variable satisfies the given comparison condition. Signal variables
        are special uint64_t symmetric objects used for efficient synchronization
        with signal operations.

        Args:
            signal (tensor): Symmetric signal tensor with remote PE.
                             Must be 8-byte aligned symmetric memory.
            cmp (int32): Comparison operator. Common values:
                        - NVSHMEM_CMP_EQ (0): Wait until signal == cmp_val
                        - NVSHMEM_CMP_NE (1): Wait until signal != cmp_val
                        - NVSHMEM_CMP_GT (2): Wait until signal > cmp_val
                        - NVSHMEM_CMP_GE (3): Wait until signal >= cmp_val
                        - NVSHMEM_CMP_LT (4): Wait until signal < cmp_val
                        - NVSHMEM_CMP_LE (5): Wait until signal <= cmp_val
            cmp_val (int64): Value to compare against.

        Returns:
            int32: Status code (0 for success).

        Notes:
            - This is a blocking operation designed specifically for signal variables.
            - Signal variables are updated atomically by putmem_signal operations.
            - More efficient than wait_until for signal-based synchronization patterns.
            - Ensures the signal update is fully complete before returning.
            - Commonly used with putmem_signal_block for producer-consumer patterns.

        Example:
            ```
            # Wait for signal to be set to completion value
            NVSHMEM_CMP_EQ = 0
            nvshmem.signal_wait_until(signal_ptr, NVSHMEM_CMP_EQ, 42)
            ```
        """
        cmp_val = 0 << 32 | cmp_val
        return signal_wait_until_extern_wrapper(
            signal.to(tl.int64), cmp, cmp_val.to(tl.uint64)
        )

    @core.extern
    def signal_wait_until_extern_wrapper(signal, cmp, cmp_val, _semantic=None):  # type: ignore[no-untyped-def]
        return core.extern_elementwise(
            "",
            "",
            [signal, cmp, cmp_val],
            {
                (
                    core.dtype("int64"),
                    core.dtype("int32"),
                    core.dtype("uint64"),
                ): ("nvshmem_signal_wait_until", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @core.extern
    def signal_op(sig_addr, signal, sig_op, pe, _semantic=None):  # type: ignore[no-untyped-def]
        """
        Perform an atomic signal operation on a remote PE.

        This function atomically updates a signal variable on the specified remote PE
        using the given operation and value. This enables efficient point-to-point
        synchronization and notification between PEs.

        Args:
            sig_addr (int64): Symmetric address of the signal variable (uint64_t) on the remote PE.
                             Must be 8-byte aligned symmetric memory.
            signal (int64): Value to be used in the signal operation.
            sig_op (int32): Signal operation type. Common values:
                           - NVSHMEM_SIGNAL_SET (0): Atomically set sig_addr = signal
                           - NVSHMEM_SIGNAL_ADD (5): Atomically set sig_addr += signal
            pe (int32): PE number of the remote PE (0 ≤ pe < nvshmem_n_pes()).
            _semantic: Optional semantic information for Triton compilation.

        Returns:
            int32: Status code (0 for success).

        Notes:
            - This is a one-sided operation - the remote PE does not need to participate.
            - The signal operation is performed atomically on the remote PE.
            - Can be used with signal_wait_until() on the remote PE for synchronization.
            - Provides low-overhead notification mechanism between PEs.
            - The signal variable must be of type uint64_t in symmetric memory.

        Example:
            ```python
            # Atomically set remote signal to 1 to notify completion
            NVSHMEM_SIGNAL_SET = 0
            nvshmem.signal_op(remote_signal_ptr, 1, NVSHMEM_SIGNAL_SET, target_pe)
            ```
        """
        return core.extern_elementwise(
            "",
            "",
            [sig_addr, signal, sig_op, pe],
            {
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("nvshmemx_signal_op", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    # Memory Ordering Operations
    @core.extern
    def fence(_semantic=None):  # type: ignore[no-untyped-def]
        """
        Ensure ordering of put operations to each remote PE.

        This function provides a memory fence that ensures point-to-point ordering
        of remote memory operations. Put operations issued before the fence are
        guaranteed to be ordered before put operations issued after the fence,
        when targeting the same remote PE.

        Args:
            _semantic: Optional semantic information for Triton compilation.

        Returns:
            int32: Status code (0 for success).

        Notes:
            - This provides weaker ordering guarantees than quiet().
            - Operations to each PE are ordered, but operations to different PEs
              may still be reordered relative to each other.
            - Does not guarantee completion of operations, only ordering.
            - Non-blocking operations are not ordered by fence - use quiet() instead.
            - Essential for ensuring correct ordering in communication patterns.

        Memory Ordering Guarantees:
            - Put operations before fence() → ordered before → Put operations after fence()
            - Ordering is maintained per-destination-PE basis
            - Remote PEs can observe the enforced ordering

        Example:
            ```
            # Ensure first put completes before second put to same PE
            nvshmem.put(dst, src, nelems, target_pe)
            nvshmem.fence()  # Enforce ordering
            nvshmem.put(dst2, src2, nelems, target_pe)
            ```
        """
        return core.extern_elementwise(
            "",
            "",
            [],
            {
                (): ("nvshmem_fence", core.dtype("int32")),
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @core.extern
    def quiet(_semantic=None):  # type: ignore[no-untyped-def]
        """
        Wait for completion of all outstanding put operations.

        This function blocks until all outstanding remote memory operations issued
        by the calling PE have completed. It provides stronger guarantees than
        fence() by ensuring both ordering and completion of all operations.

        Args:
            _semantic: Optional semantic information for Triton compilation.

        Returns:
            int32: Status code (0 for success).

        Notes:
            - This is a blocking operation that waits for completion.
            - Ensures all previous put operations have been delivered to their destinations.
            - Provides global ordering - operations to ALL PEs are ordered.
            - Required to complete non-blocking operations.
            - More expensive than fence() but provides stronger guarantees.

        Memory Ordering Guarantees:
            - All put operations before quiet() are completed before any operations after quiet()
            - Operations are visible to all PEs as having occurred before subsequent operations
            - Both blocking and non-blocking operations are completed

        Example:
            ```
            # Ensure all data transfers complete before setting completion flag
            nvshmem.putmem_block(data_ptr, src_ptr, data_size, target_pe)
            nvshmem.quiet()  # Wait for data transfer completion
            nvshmem.putmem_block(
                flag_ptr, flag_src_ptr, 8, target_pe
            )  # Signal completion
            ```
        """
        return core.extern_elementwise(
            "",
            "",
            [],
            {
                (): ("nvshmem_quiet", core.dtype("int32")),
            },
            is_pure=False,
            _semantic=_semantic,
        )

    # PE Information Operations
    @core.extern
    def my_pe(_semantic=None):  # type: ignore[no-untyped-def]
        """
        Get the PE number of the calling PE.

        This function returns the unique identifier (PE number) of the current
        processing element within the NVSHMEM job. PE numbers range from 0 to
        nvshmem_n_pes() - 1.

        Args:
            _semantic: Optional semantic information for Triton compilation.

        Returns:
            int32: PE number of the calling PE (0 ≤ pe < nvshmem_n_pes()).

        Notes:
            - This is a pure function that returns the same value throughout execution.
            - PE numbering starts from 0 and is contiguous.
            - Each PE has a unique identifier within the NVSHMEM job.
            - Can be called from both host and device code.
            - Essential for implementing PE-specific logic and communication patterns.

        Example:
            ```
            # Get current PE number for conditional logic
            pe = nvshmem.my_pe()
            if pe == 0:
                # Root PE logic
                pass
            else:
                # Non-root PE logic
                pass
            ```
        """
        return core.extern_elementwise(
            "",
            "",
            [],
            {(): ("nvshmem_my_pe", core.dtype("int32"))},
            is_pure=True,
            _semantic=_semantic,
        )

    @core.extern
    def n_pes(_semantic=None):  # type: ignore[no-untyped-def]
        """
        Get the total number of PEs in the NVSHMEM job.

        This function returns the total count of processing elements (PEs)
        participating in the current NVSHMEM job. This value remains constant
        throughout the execution of the program.

        Args:
            _semantic: Optional semantic information for Triton compilation.

        Returns:
            int32: Total number of PEs in the job (always ≥ 1).

        Notes:
            - This is a pure function that returns the same value throughout execution.
            - The value is determined at NVSHMEM initialization and never changes.
            - Valid PE numbers range from 0 to n_pes() - 1.
            - Can be called from both host and device code.
            - Essential for implementing collective operations and communication patterns.

        Example:
            ```
            # Broadcast from root to all other PEs
            total_pes = nvshmem.n_pes()
            my_rank = nvshmem.my_pe()

            if my_rank == 0:
                # Send to all other PEs
                for peer in range(1, total_pes):
                    nvshmem.putmem_block(dst_ptr, src_ptr, size, peer)
            ```
        """
        return core.extern_elementwise(
            "",
            "",
            [],
            {(): ("nvshmem_n_pes", core.dtype("int32"))},
            is_pure=True,
            _semantic=_semantic,
        )

    # Synchronization Operations
    @core.extern
    def barrier_all(_semantic=None):  # type: ignore[no-untyped-def]
        """
        Synchronize all PEs with completion guarantee.

        This function creates a barrier across all PEs in the NVSHMEM job. It ensures
        that all local and remote memory updates issued before the barrier by any PE
        are completed before any PE exits the barrier. This provides both
        synchronization and memory consistency.

        Args:
            _semantic: Optional semantic information for Triton compilation.

        Returns:
            int32: Status code (0 for success).

        Notes:
            - This is a collective operation - all PEs must participate.
            - Stronger guarantee than sync_all() - ensures completion of remote operations.
            - Blocks until all PEs reach the barrier AND all memory operations complete.
            - Must be called from kernels launched with cooperative launch.
            - Provides full memory consistency across all PEs.
            - More expensive than sync_all() due to completion guarantees.

        Memory Consistency Guarantees:
            - All memory updates before barrier_all() are visible to all PEs
            - All remote memory operations are completed before any PE continues
            - Provides a global synchronization point with memory ordering

        Example:
            ```
            # Ensure all PEs complete their work before proceeding
            # All PEs execute this - it's a collective operation
            nvshmem.barrier_all()
            # At this point, all previous operations are complete on all PEs
            ```
        """
        return core.extern_elementwise(
            "",
            "",
            [],
            {(): ("nvshmem_barrier_all", core.dtype("int32"))},
            is_pure=False,
            _semantic=_semantic,
        )

    @core.extern
    def sync_all(_semantic=None):  # type: ignore[no-untyped-def]
        """
        Synchronize all PEs with local completion guarantee.

        This function creates a lightweight synchronization barrier across all PEs.
        It ensures that all local store operations issued before the sync are
        visible to other PEs, but does not guarantee completion of remote memory
        operations initiated by the calling PE.

        Args:
            _semantic: Optional semantic information for Triton compilation.

        Returns:
            int32: Status code (0 for success).

        Notes:
            - This is a collective operation - all PEs must participate.
            - Lighter weight than barrier_all() - only ensures local store visibility.
            - Does not guarantee completion of remote memory updates initiated locally.
            - Must be called from kernels launched with cooperative launch.
            - Suitable when only synchronization (not completion) is needed.
            - More efficient than barrier_all() for synchronization-only patterns.

        Memory Consistency Guarantees:
            - Local store operations are visible to other PEs
            - Does NOT ensure completion of outgoing remote operations
            - Provides synchronization point without full completion overhead

        Example:
            ```
            # Lightweight synchronization between PEs
            # All PEs execute this - it's a collective operation
            nvshmem.sync_all()
            # Local stores are visible, but remote ops may still be in flight
            ```
        """
        return core.extern_elementwise(
            "",
            "",
            [],
            {(): ("nvshmem_sync_all", core.dtype("int32"))},
            is_pure=False,
            _semantic=_semantic,
        )

    # Collective Operations (mem-based APIs - sizes in bytes)
    @triton.jit  # type: ignore[misc]
    def alltoall(team, dest, source, nelems_per_pe):  # type: ignore[no-untyped-def]
        """
        All-to-all tensor exchange between PEs in a team.

        This high-level function provides a tensor-aware interface for NVSHMEM alltoall
        operations. Each PE sends nelems_per_pe elements to every other PE and receives
        the same amount from every other PE.

        Args:
            team: Team handle for the collective operation. Use 0 for NVSHMEM_TEAM_WORLD.
            dest: Destination tensor. Must be large enough for nelems_per_pe * n_pes elements.
            source: Source tensor containing data for all PEs. Must contain nelems_per_pe * n_pes elements.
            nelems_per_pe: Number of elements to exchange with each PE.

        Notes:
            - Performs compile-time type checking between dest and source tensors.
            - Automatically calculates byte size from tensor type and element count.
            - This is a collective operation - all PEs in the team must participate.
            - Data layout: source=[data_for_pe0, data_for_pe1, ...], dest=[data_from_pe0, data_from_pe1, ...]

        Example:
            ```
            # Each PE exchanges 10 elements with every other PE
            nvshmem.alltoall(0, dest_tensor, src_tensor, 10)
            ```
        """
        tl.static_assert(dest.type == source.type)
        size_bytes_per_pe = nelems_per_pe * dest.type.element_ty.itemsize
        return alltoallmem_block_extern_wrapper(
            team, dest.to(tl.int64), source.to(tl.int64), size_bytes_per_pe.to(tl.int64)
        )

    @core.extern  # type: ignore[misc]
    def alltoallmem_block_extern_wrapper(
        team: Any, dest: Any, source: Any, size_bytes: Any, _semantic: Any = None
    ) -> None:
        """Low-level extern wrapper for NVSHMEM alltoall"""
        return core.extern_elementwise(
            "",
            "",
            [team, dest, source, size_bytes],
            {
                (
                    core.dtype("int32"),  # team handle
                    core.dtype("int64"),  # dest ptr
                    core.dtype("int64"),  # source ptr
                    core.dtype("int64"),  # size in bytes
                ): ("nvshmemx_alltoallmem_block", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @triton.jit  # type: ignore[misc]
    def broadcast(team, dest, source, nelems, pe_root):  # type: ignore[no-untyped-def]
        """
        Broadcast tensor data from a root PE to all other PEs in a team.

        This high-level function provides a tensor-aware interface for NVSHMEM broadcast
        operations. It automatically handles type checking and size calculations, making
        the API more ergonomic and type-safe.

        Args:
            team: Team handle for the collective operation. Use 0 for NVSHMEM_TEAM_WORLD.
            dest: Destination tensor with type information. All PEs receive data here.
            source: Source tensor on the root PE. Type must match dest.
