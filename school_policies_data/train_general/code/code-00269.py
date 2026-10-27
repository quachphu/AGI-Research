                # SEND/RECV op are avoided for special case with 2 adjacent stages on same rank
                # see [Note: V-schedule special case]
                if is_prev_stage_on_this_rank:
                    stage_index_to_stage[stage_idx - 1].set_local_bwd_input(
                        stage.get_local_bwd_output(mb_index), mb_index
                    )
            elif comp_type == BACKWARD_WEIGHT:
                self._assert_unsharded(stage)
                self.backward_counter[stage_idx] += 1
                last_backward = self.backward_counter[stage_idx] == self._n_microbatches
                stage.backward_weight_one_chunk(
                    mb_index,
                    last_backward=last_backward,
                )
            elif comp_type == REDUCE_GRAD:
                grad_scale_factor = self._n_microbatches if self.scale_grads else 1
                stage.perform_reduce_grad(grad_scale_factor)
            else:
                raise ValueError(f"{action=} is unknown or unsupported")

        # count either full_backward or backward_weight together, to determine when to sync DP grads
        self.backward_counter.clear()
        for time_step, action in enumerate(self.pipeline_order_with_comms[self.rank]):
            logger.debug(
                "_PipelineScheduleRuntime running time_step %d, action %s",
                time_step,
                action,
            )
            try:
                with record_function(_get_profiler_function_name(action)):
                    if action.computation_type in self._comp_type_to_function_map:
                        ctx = _PipelineContext(
                            self,
                            arg_mbs,
                            kwarg_mbs,
                            target_mbs,
                            losses,
                        )
                        self._comp_type_to_function_map[action.computation_type](
                            action, ctx
                        )
                    elif action.computation_type == OVERLAP_F_B:
                        if action.sub_actions is None:
                            raise AssertionError("sub_actions must be set")
                        for sub_a in action.sub_actions:
                            _perform_action(sub_a)
                    else:
                        _perform_action(action)
            except Exception as e:
                logger.error(
                    "_PipelineScheduleRuntime caught exception at step %s when running action %s.  Full Schedule:",
                    time_step,
                    action,
                )
                logger.error(
                    _format_pipeline_order(
                        self.pipeline_order_with_comms,  # type: ignore[arg-type]
                        error_step_number=time_step,
                    )
                )
                raise e

        # Mostly these operations should have finished long ago, but there isn't an obvious time when to wait for them
        while send_ops:
            _wait_batch_p2p(send_ops.pop())

        if len(self.unshard_ops) != 0:
            raise AssertionError("Unused unshard operations")

        # Return losses if there is a container passed in
        self._update_losses(self._stages, losses)


class ScheduleLoopedBFS(_PipelineScheduleRuntime):
    """
    Breadth-First Pipeline Parallelism.
    See https://arxiv.org/abs/2211.05953 for details.
    Similar to Interleaved 1F1B, Looped BFS supports multiple stages per rank.
    What is different is that when microbatches are ready for multiple local
    stages, Loops BFS will prioritizes the earlier stage, running all available
    microbatches at once.
    """

    def __init__(
        self,
        stages: list[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Callable | _Loss | None = None,
        output_merge_spec: dict[str, Any] | tuple[Any] | None = None,
        scale_grads: bool = True,
        backward_requires_autograd: bool = True,
        defer_pp_recv: bool = False,
    ):
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            output_merge_spec=output_merge_spec,
            scale_grads=scale_grads,
            backward_requires_autograd=backward_requires_autograd,
            defer_pp_recv=defer_pp_recv,
        )

        # 1. Create the pipeline_order (all ranks do this calculation)
        # This will be used to keep track of the current state of the entire pipeline
        # pipeline_order[rank] = [Action(computation_type, microbatch_index, stage_index), ...]
        self.pipeline_order: dict[int, list[_Action | None]] = {}
        # ========================================================================
        for rank in range(self.pp_group_size):
            rank_ops = self._calculate_single_rank_operations(rank)
            self.pipeline_order[rank] = rank_ops

        # Initialize the pipeline order with communication necessary to run with _PipelineScheduleRuntime
        self._prepare_schedule_with_comms(self.pipeline_order)

    def _calculate_single_rank_operations(self, rank):
        n_local_stages = len(self._stages)
        stage_indices = range(
            rank, self.pp_group_size * n_local_stages, self.pp_group_size
        )

        # Store the list of operations used for that rank
        # Pre-padding, rank starts with no-ops based on the warmup.
        rank_ops: list[_Action | None] = [None for _ in range(rank)]

        for stage_index in stage_indices:
            rank_ops.extend(
                _Action(stage_index, _ComputationType.FORWARD, mb_index)
                for mb_index in range(self._n_microbatches)
            )

        # wait for the first backward to trickle up
        # which is 2 for every hop away
        post_warmup_ops = 2 * (self.pp_group_size - 1 - rank)
        rank_ops.extend([None] * post_warmup_ops)

        for stage_index in reversed(stage_indices):
            rank_ops.extend(
                _Action(stage_index, _ComputationType.FULL_BACKWARD, mb_index)
                for mb_index in reversed(range(self._n_microbatches))
            )
        return rank_ops


def _get_1f1b_rank_ops(
    n_local_stages,
    pp_group_size,
    warmup_ops,
    fwd_bwd_ops,
    cooldown_ops,
    rank,
    forward_stage_index,
    backward_stage_index,
    num_1f1b_microbatches=0,
    enable_zero_bubble=False,
):
    # All stages start with handling microbatch 0
    fwd_stage_mb_index: dict[int, int] = defaultdict(int)
    bwd_stage_mb_index: dict[int, int] = defaultdict(int)
    weight_stage_mb_index: dict[int, int] = defaultdict(int)

    # Store the list of operations used for that rank
    # Pre-padding, rank starts with no-ops based on the warmup.
    rank_ops: list[_Action | None] = [None for _ in range(rank)]
    # These are used to calculate the number of slots to fill with no-ops, to account for the delay in warmup
    # when we want to wait for the backward to trickle back up and start 1f1b to align all ranks.
    # Formula:
    # pre-padding + warmup_ops + post_warmup_ops = earliest time step of first backward
    # post_warmup_ops = [earliest time step of first backward] - (warmup_ops + pre-padding)
    # earliest time step of first backward = [local_stages * group_size + 2 * (group_size - 1 - rank)]
    # warmup_ops = calculated above
    post_warmup_ops = (
        n_local_stages * pp_group_size + 2 * (pp_group_size - 1 - rank)
    ) - (warmup_ops + rank)

    if enable_zero_bubble:
        post_warmup_ops = pp_group_size - rank - 1

    total_ops = warmup_ops + fwd_bwd_ops + cooldown_ops

    backward_op_ids = []
    weight_op_count = 0

    FULL_BACKWARD_OR_BACKWARD_INPUT = (
        BACKWARD_INPUT if enable_zero_bubble else FULL_BACKWARD
    )

    for op in range(total_ops):
        # Warmup phase
        if op < warmup_ops:
            fwd_stage_index = forward_stage_index(op)
            # This will assign the current microbatch index and update it as well
            fwd_stage_mb_index[fwd_stage_index] = (
                mb_index := fwd_stage_mb_index[fwd_stage_index]
            ) + 1
            rank_ops.append(
                _Action(fwd_stage_index, _ComputationType.FORWARD, mb_index)
            )
            if op == warmup_ops - 1:
                # This is the last step in the warmup phase, so we need to wait for the backward to trickle back up
                rank_ops.extend([None] * post_warmup_ops)
        # 1F1B Phase (forward and backward)
        elif warmup_ops <= op < warmup_ops + fwd_bwd_ops:
            fwd_stage_index = forward_stage_index(op)
            fwd_stage_mb_index[fwd_stage_index] = (
                fwd_mb_index := fwd_stage_mb_index[fwd_stage_index]
            ) + 1
            rank_ops.append(
                _Action(fwd_stage_index, _ComputationType.FORWARD, fwd_mb_index)
            )
            bwd_stage_index = backward_stage_index(op)
            bwd_stage_mb_index[bwd_stage_index] = (
                bwd_mb_index := bwd_stage_mb_index[bwd_stage_index]
            ) + 1
            rank_ops.append(
                _Action(bwd_stage_index, FULL_BACKWARD_OR_BACKWARD_INPUT, bwd_mb_index)
            )
            backward_op_ids.append(op)

            if enable_zero_bubble and op - warmup_ops >= num_1f1b_microbatches:
                weight_stage_index = backward_stage_index(
                    backward_op_ids[weight_op_count]
                )
                weight_stage_mb_index[weight_stage_index] = (
                    weight_mb_index := weight_stage_mb_index[weight_stage_index]
                ) + 1
                rank_ops.append(
                    _Action(
                        weight_stage_index,
                        _ComputationType.BACKWARD_WEIGHT,
                        weight_mb_index,
                    )
                )
                weight_op_count += 1
        # Cooldown phase
        else:
            # During cooldown phase, we need steps to align with 1f1b happening in other ranks
            # TODO: we don't need to always append, after all 1f1b are finished we can stop appending None
            if not enable_zero_bubble:
                rank_ops.append(None)

            bwd_stage_index = backward_stage_index(op)
            bwd_stage_mb_index[bwd_stage_index] = (
                bwd_mb_index := bwd_stage_mb_index[bwd_stage_index]
            ) + 1
            rank_ops.append(
                _Action(bwd_stage_index, FULL_BACKWARD_OR_BACKWARD_INPUT, bwd_mb_index)
            )
            backward_op_ids.append(op)

            if enable_zero_bubble and op - warmup_ops >= num_1f1b_microbatches:
                weight_stage_index = backward_stage_index(
                    backward_op_ids[weight_op_count]
                )
                weight_stage_mb_index[weight_stage_index] = (
                    weight_mb_index := weight_stage_mb_index[weight_stage_index]
                ) + 1
                rank_ops.append(
                    _Action(
                        weight_stage_index,
                        _ComputationType.BACKWARD_WEIGHT,
                        weight_mb_index,
                    )
                )
                weight_op_count += 1

    while enable_zero_bubble and weight_op_count < len(backward_op_ids):
        weight_stage_index = backward_stage_index(backward_op_ids[weight_op_count])
        weight_stage_mb_index[weight_stage_index] = (
            weight_mb_index := weight_stage_mb_index[weight_stage_index]
        ) + 1
        rank_ops.append(
            _Action(
                weight_stage_index, _ComputationType.BACKWARD_WEIGHT, weight_mb_index
            )
        )
        weight_op_count += 1

    return rank_ops


def _get_warmup_ops(
    rank: int,
    n_local_stages: int,
    microbatches_per_round: int,
    pp_group_size: int,
    n_microbatches: int,
    multiply_factor: int = 2,
) -> int:
    """
    Calculate the number of warmup operations for interleaved schedules.
    """
    # Warmup operations for last stage
    warmups_ops_last_stage = (n_local_stages - 1) * microbatches_per_round
    # Increment warmup operations by multiply_factor for each hop away from the last stage
    warmup_ops = warmups_ops_last_stage + multiply_factor * ((pp_group_size - 1) - rank)
    # We cannot have more warmup operations than there are number of microbatches, so cap it there
    return min(warmup_ops, n_microbatches * n_local_stages)


class ScheduleInterleaved1F1B(_PipelineScheduleRuntime):
    """
    The Interleaved 1F1B schedule.
    See https://arxiv.org/pdf/2104.04473 for details.
    Will perform one forward and one backward on the microbatches in steady
    state and supports multiple stages per rank. When microbatches are ready for
    multiple local stages, Interleaved 1F1B prioritizes the earlier microbatch
    (also called "depth first").

    This schedule is mostly similar to the original paper.
    It differs by being relaxing the requirement of num_microbatch % pp_size == 0.
    Using the flex_pp schedule, we will have num_rounds = max(1, n_microbatches // pp_group_size) and
    it works as long as n_microbatches % num_rounds is 0. As a few examples, support

    1. pp_group_size = 4, n_microbatches = 10. We will have num_rounds = 2 and n_microbatches % 2 is 0.
    2. pp_group_size = 4, n_microbatches = 3. We will have num_rounds = 1 and n_microbatches % 1 is 0.
    """

    def __init__(
        self,
        stages: list[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Callable | None = None,
        args_chunk_spec: tuple[TensorChunkSpec, ...] | None = None,
        kwargs_chunk_spec: dict[str, TensorChunkSpec] | None = None,
        output_merge_spec: dict[str, Any] | tuple[Any] | None = None,
        scale_grads: bool = True,
        backward_requires_autograd: bool = True,
        defer_pp_recv: bool = False,
    ):
        self.pp_group_size = stages[0].group_size
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
            scale_grads=scale_grads,
            backward_requires_autograd=backward_requires_autograd,
            defer_pp_recv=defer_pp_recv,
        )
        self.n_local_stages = len(stages)
        self.rank = stages[0].group_rank
        self.number_of_rounds = max(1, n_microbatches // self.pp_group_size)
        self.microbatches_per_round = n_microbatches // self.number_of_rounds
        if n_microbatches % self.number_of_rounds != 0:
            raise ValueError(
                "Interleaved 1F1B requires the number of microbatches to be a "
                f"multiple of the number of rounds ({self.number_of_rounds}), "
                f"but got {n_microbatches}."
            )
        # 1. Create the pipeline_order (all ranks do this calculation)
        # This will be used to keep track of the current state of the entire pipeline
        # pipeline_order[rank] = [Action(computation_type, microbatch_index, stage_index), ...]
        self.pipeline_order: dict[int, list[_Action | None]] = {}
        for rank in range(self.pp_group_size):
            rank_ops = self._calculate_single_rank_operations(rank)
            self.pipeline_order[rank] = rank_ops

        # Initialize the pipeline order with communication necessary to run with _PipelineScheduleRuntime
        self._prepare_schedule_with_comms(self.pipeline_order)

    def _calculate_single_rank_operations(self, rank) -> list[_Action | None]:
        warmup_ops = _get_warmup_ops(
            rank,
            self.n_local_stages,
            self.microbatches_per_round,
            self.pp_group_size,
            self._n_microbatches,
            multiply_factor=2,
        )
        microbatch_ops = self.n_local_stages * self._n_microbatches
        # fwd_bwd_ops should encompass the remaining forwards
        fwd_bwd_ops = microbatch_ops - warmup_ops
        # cooldown_ops should encompass the remaining backwards
        cooldown_ops = microbatch_ops - fwd_bwd_ops
        # total ops encompass both forward and backward ops
        total_ops = warmup_ops + fwd_bwd_ops + cooldown_ops
        # warmup_ops + fwd_bwd_ops * 2 + cooldown_ops == microbatch_ops * 2
        logger.debug(
            "rank %s, warmup_ops %s, 1f1b %s, cooldown_ops %s total_ops %s",
            rank,
            warmup_ops,
            fwd_bwd_ops,
            cooldown_ops,
            total_ops,
        )

        # Calculates the stage index based on step and pp_group_size
        def forward_stage_index(step):
            # Get the local index from 0 to n_local_stages-1
            local_index = (step // self.microbatches_per_round) % self.n_local_stages
            return (local_index * self.pp_group_size) + rank

        def backward_stage_index(step):
            local_index = (
                self.n_local_stages
                - 1
                - ((step - warmup_ops) // self.microbatches_per_round)
                % self.n_local_stages
            )
            return (local_index * self.pp_group_size) + rank

        return _get_1f1b_rank_ops(
            self.n_local_stages,
            self.pp_group_size,
            warmup_ops,
            fwd_bwd_ops,
            cooldown_ops,
            rank,
            forward_stage_index,
            backward_stage_index,
        )


class ScheduleInterleavedZeroBubble(_PipelineScheduleRuntime):
    """
    The Interleaved Zero Bubble schedule.
    See https://arxiv.org/pdf/2401.10241 for details.
    Will perform one forward and one backward on inputs for the microbatches in steady
    state and supports multiple stages per rank. Uses the backward for weights to fill in
    the pipeline bubble.

    In particular this is implementing the ZB1P schedule in the paper.
    """

    def __init__(
        self,
        stages: list[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Callable | None = None,
        args_chunk_spec: tuple[TensorChunkSpec, ...] | None = None,
        kwargs_chunk_spec: dict[str, TensorChunkSpec] | None = None,
        output_merge_spec: dict[str, Any] | tuple[Any] | None = None,
        scale_grads: bool = True,
        backward_requires_autograd: bool = True,
        defer_pp_recv: bool = False,
    ):
        # TODO: we don't support input/weight backward split with torch.compile
        _check_torch_compile_compatibility(stages, self.__class__.__name__)
        self.pp_group_size = stages[0].group_size
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
            scale_grads=scale_grads,
            backward_requires_autograd=backward_requires_autograd,
            defer_pp_recv=defer_pp_recv,
        )
        self.n_local_stages = len(stages)
        self.rank = stages[0].group_rank
        self.number_of_rounds = max(1, n_microbatches // self.pp_group_size)
        self.microbatches_per_round = n_microbatches // self.number_of_rounds
        if n_microbatches % self.number_of_rounds != 0:
            raise ValueError(
                "Zero bubble requires the number of microbatches to be a "
                f"multiple of the number of rounds ({self.number_of_rounds}), "
                f"but got {n_microbatches}."
            )
        # 1. Create the pipeline_order (all ranks do this calculation)
        # This will be used to keep track of the current state of the entire pipeline
        # pipeline_order[rank] = [Action(computation_type, microbatch_index, stage_index), ...]
        self.pipeline_order: dict[int, list[_Action | None]] = {}
        for rank in range(self.pp_group_size):
            rank_ops = self._calculate_single_rank_operations(rank)
            self.pipeline_order[rank] = rank_ops

        # This function add bubbles to the generated schedule based on dependencies of actions
        # Note that the ZB1P schedule will not require bubbles to be manually added and it is
        # only useful when n_microbatches <= microbatches_per_round
        self.pipeline_order = self._add_bubbles_to_actions(
            self.n_local_stages * self.pp_group_size,
        )

        # Initialize the pipeline order with communication necessary to run with _PipelineScheduleRuntime
        self._prepare_schedule_with_comms(self.pipeline_order)

    def _calculate_single_rank_operations(self, rank) -> list[_Action | None]:
        warmup_ops = _get_warmup_ops(
            rank,
            self.n_local_stages,
            self.microbatches_per_round,
            self.pp_group_size,
            self._n_microbatches,
            multiply_factor=1,
        )
        microbatch_ops = self.n_local_stages * self._n_microbatches
        # fwd_bwd_ops should encompass the remaining forwards
        fwd_bwd_ops = microbatch_ops - warmup_ops
        # cooldown_ops should encompass the remaining backwards
        cooldown_ops = microbatch_ops - fwd_bwd_ops
        # total ops encompass both forward and backward ops
        total_ops = warmup_ops + fwd_bwd_ops + cooldown_ops
        # warmup_ops + fwd_bwd_ops * 2 + cooldown_ops == microbatch_ops * 2
        logger.debug(
            "rank %s, warmup_ops %s, 1f1b %s, cooldown_ops %s total_ops %s",
            rank,
            warmup_ops,
            fwd_bwd_ops,
            cooldown_ops,
            total_ops,
        )

        # Calculates the stage index based on step and pp_group_size

        def forward_stage_index(step):
            # Get the local index from 0 to n_local_stages-1
            local_index = (step // self.microbatches_per_round) % self.n_local_stages
            return (local_index * self.pp_group_size) + rank

        def backward_stage_index(step):
            local_index = (
                self.n_local_stages
                - 1
                - ((step - warmup_ops) // self.microbatches_per_round)
                % self.n_local_stages
            )
            return (local_index * self.pp_group_size) + rank

        num_1f1b_microbatches = rank

        return _get_1f1b_rank_ops(
            self.n_local_stages,
            self.pp_group_size,
            warmup_ops,
            fwd_bwd_ops,
            cooldown_ops,
            rank,
            forward_stage_index,
            backward_stage_index,
            num_1f1b_microbatches,
            enable_zero_bubble=True,
        )

    def _add_bubbles_to_actions(self, num_stages_global):
        actions = self.pipeline_order

        def need_bubble(stage, op, microbatch, num_stages_global, seen_ops):
            if op == _ComputationType.FORWARD:
                if stage != 0 and (stage - 1, op, microbatch) not in seen_ops:
                    return True
            elif op == _ComputationType.FULL_BACKWARD:
                if stage == num_stages_global - 1:
                    return (stage, _ComputationType.FORWARD, microbatch) not in seen_ops
                return (stage + 1, op, microbatch) not in seen_ops
            return False

        seen_ops: set[tuple[int, _ComputationType, int]] = set()
        result: dict[int, list[_Action | None]] = {}
        next_pointer: dict[int, int] = {}
        bubbles_added: dict[int, int] = {}
        total_bubbles_added = 0

        for rank in range(self.pp_group_size):
            result[rank] = []
            next_pointer[rank] = 0
            bubbles_added[rank] = 0

        while True:
            should_stop = True

            temp_seen_ops: set[tuple[int, _ComputationType, int]] = set()

            for rank in range(self.pp_group_size):
                timestamp = next_pointer[rank]
                if timestamp >= len(actions[rank]):
                    continue

                should_stop = False

                if actions[rank][timestamp] is not None:
                    temp_action = actions[rank][timestamp]
                    if temp_action is None:
                        raise AssertionError(
                            f"Expected temp_action to be not None, got {type(temp_action)}"
                        )
                    stage_index, op, microbatch, _ = temp_action
                    if not need_bubble(
                        stage_index, op, microbatch, num_stages_global, seen_ops
                    ):
                        result[rank].append(actions[rank][timestamp])
                        if microbatch is not None:
                            temp_seen_ops.add((stage_index, op, microbatch))
                        next_pointer[rank] += 1
                    else:
                        result[rank].append(None)
                        bubbles_added[rank] += 1
                else:
                    next_pointer[rank] += 1
                    result[rank].append(None)

            seen_ops.update(temp_seen_ops)
            if should_stop:
                break

        if total_bubbles_added > 0:
            logger.warning(
                "Non zero bubbles added: total_bubbles_added=%s bubbles_added=%s",
                total_bubbles_added,
                bubbles_added,
            )
        return result


class ScheduleZBVZeroBubble(_PipelineScheduleRuntime):
    """
    The Zero Bubble schedule (ZBV variant).
    See https://arxiv.org/pdf/2401.10241 Section 6 for details.

    This schedules requires exactly two stages per rank.

    This schedule will perform one forward and one backward on inputs for the microbatches in steady
    state and supports multiple stages per rank. Uses backward with respect to weights to fill in
    the pipeline bubble.

    This ZB-V schedule would have the "zero bubble" property only if time forward == time backward input == time backward weights.
    In practice, this is not likely true for real models so alternatively
    a greedy scheduler could be implemented for unequal/unbalanced time.
    """

    def __init__(
        self,
        stages: list[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Callable | None = None,
        args_chunk_spec: tuple[TensorChunkSpec, ...] | None = None,
        kwargs_chunk_spec: dict[str, TensorChunkSpec] | None = None,
        output_merge_spec: dict[str, Any] | tuple[Any] | None = None,
        scale_grads: bool = True,
        backward_requires_autograd: bool = True,
        defer_pp_recv: bool = False,
    ):
        # TODO: we don't support input/weight backward split with torch.compile
        _check_torch_compile_compatibility(stages, self.__class__.__name__)
        self.pp_group_size = stages[0].group_size
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
            scale_grads=scale_grads,
            backward_requires_autograd=backward_requires_autograd,
            defer_pp_recv=defer_pp_recv,
        )
        self.stage_index_to_group_rank = generate_stage_to_rank_mapping(
            self.pp_group_size, self._num_stages, style="v"
        )
        for stage in self._stages:
            stage.stage_index_to_group_rank = self.stage_index_to_group_rank

        self.n_local_stages = len(stages)
        if self.n_local_stages != 2:
            raise ValueError(
                "ZBV requires exactly 2 stages per rank, but got "
                f"{self.n_local_stages}."
            )

        self.rank = stages[0].group_rank
        self.num_stages = stages[0].num_stages

        # 1. Create the pipeline_order (all ranks do this calculation)
        # This will be used to keep track of the current state of the entire pipeline
        # pipeline_order[rank] = [Action(computation_type, microbatch_index, stage_index), ...]
        self.pipeline_order: dict[int, list[_Action | None]] = {}
        for rank in range(self.pp_group_size):
            rank_ops = self._calculate_single_rank_operations(rank)
            self.pipeline_order[rank] = rank_ops

        # Initialize the pipeline order with communication necessary to run with _PipelineScheduleRuntime
        self._prepare_schedule_with_comms(self.pipeline_order)

    def _calculate_single_rank_operations(self, rank) -> list[_Action | None]:
        # max(2 * self.pp_group_size - 1, ...) ensure the number of microbatches is at least
        # as large of the number of microbatches needed to fully utilize the pipeline
        n_micro = max(2 * self.pp_group_size - 1, self._n_microbatches)
        rank_ops: list[_Action | None] = [None for _ in range(rank)]

        # Forward and backward action counts for stage chunk 0 and chunk 1
        f0_cnt, f1_cnt, b0_cnt, b1_cnt = 0, 0, 0, 0
        # warm-up phase
        warmup_n1 = 2 * (self.pp_group_size - rank) - 1
        stage_id_chunk0 = rank
        stage_id_chunk1 = self.num_stages - 1 - rank

        for _ in range(warmup_n1):
            rank_ops.append(
                _Action(stage_id_chunk0, computation_type=F, microbatch_index=f0_cnt)
            )
            f0_cnt += 1
        warmup_n2 = rank
        for _ in range(warmup_n2):
            rank_ops.append(
                _Action(stage_id_chunk1, computation_type=F, microbatch_index=f1_cnt)
            )
            f1_cnt += 1
            rank_ops.append(
                _Action(stage_id_chunk0, computation_type=F, microbatch_index=f0_cnt)
            )
            f0_cnt += 1
        warmup_n3 = self.pp_group_size - rank
        for _ in range(warmup_n3):
            rank_ops.append(
                _Action(stage_id_chunk1, computation_type=F, microbatch_index=f1_cnt)
            )
            f1_cnt += 1
            rank_ops.append(
                _Action(stage_id_chunk1, computation_type=I, microbatch_index=b1_cnt)
            )
            rank_ops.append(
                _Action(stage_id_chunk1, computation_type=W, microbatch_index=b1_cnt)
            )
            b1_cnt += 1
        # stable phase
        while f1_cnt < f0_cnt or f0_cnt < n_micro:
            if f0_cnt < n_micro:
                rank_ops.append(
                    _Action(
                        stage_id_chunk0, computation_type=F, microbatch_index=f0_cnt
                    )
                )
                f0_cnt += 1
            rank_ops.append(
                _Action(stage_id_chunk0, computation_type=I, microbatch_index=b0_cnt)
            )
            rank_ops.append(
                _Action(stage_id_chunk0, computation_type=W, microbatch_index=b0_cnt)
            )
            b0_cnt += 1

            rank_ops.append(
                _Action(stage_id_chunk1, computation_type=F, microbatch_index=f1_cnt)
            )
            f1_cnt += 1
            rank_ops.append(
                _Action(stage_id_chunk1, computation_type=I, microbatch_index=b1_cnt)
            )
            rank_ops.append(
                _Action(stage_id_chunk1, computation_type=W, microbatch_index=b1_cnt)
            )
            b1_cnt += 1
        # cool-down phase
        w0_cnt, w1_cnt = b0_cnt, b1_cnt
        cooldown_n1 = rank
        for _ in range(cooldown_n1):
            rank_ops.append(
                _Action(stage_id_chunk0, computation_type=I, microbatch_index=b0_cnt)
            )
            b0_cnt += 1
            rank_ops.append(
                _Action(stage_id_chunk1, computation_type=I, microbatch_index=b1_cnt)
            )
            b1_cnt += 1
        cooldown_n2 = self.pp_group_size - rank
        for _ in range(cooldown_n2):
            rank_ops.append(
                _Action(stage_id_chunk0, computation_type=I, microbatch_index=b0_cnt)
            )
            b0_cnt += 1
            rank_ops.append(
                _Action(stage_id_chunk0, computation_type=W, microbatch_index=w0_cnt)
            )
            w0_cnt += 1
        while w1_cnt < b1_cnt:
            rank_ops.append(
                _Action(stage_id_chunk1, computation_type=W, microbatch_index=w1_cnt)
            )
            w1_cnt += 1
        while w0_cnt < b0_cnt:
            rank_ops.append(
                _Action(stage_id_chunk0, computation_type=W, microbatch_index=w0_cnt)
            )
            w0_cnt += 1

        if not (w0_cnt == b0_cnt and b0_cnt == f0_cnt):
            raise AssertionError(
                f"Expected w0_cnt == b0_cnt == f0_cnt, got w0_cnt={w0_cnt}, b0_cnt={b0_cnt}, f0_cnt={f0_cnt}"
            )
        if not (w1_cnt == b1_cnt and b1_cnt == f1_cnt):
            raise AssertionError(
                f"Expected w1_cnt == b1_cnt == f1_cnt, got w1_cnt={w1_cnt}, b1_cnt={b1_cnt}, f1_cnt={f1_cnt}"
            )
        # We use max() in the n_micro computation above, so we may need to
        # remove redundant microbatches
        rank_ops = [
            (
                action
                if action is not None
                and action.microbatch_index is not None
                and action.microbatch_index < self._n_microbatches
                else None
            )
            for action in rank_ops
        ]
        return rank_ops


class ScheduleDualPipeV(_PipelineScheduleRuntime):
    """
    The DualPipeV schedule. A more efficient schedule variant based on the
    DualPipe schedule introduced by DeepSeek in https://arxiv.org/pdf/2412.19437

    Based on the open sourced code from https://github.com/deepseek-ai/DualPipe
    """

    def __init__(
        self,
        stages: list[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Callable | None = None,
        args_chunk_spec: tuple[TensorChunkSpec, ...] | None = None,
        kwargs_chunk_spec: dict[str, TensorChunkSpec] | None = None,
        output_merge_spec: dict[str, Any] | tuple[Any] | None = None,
        scale_grads: bool = True,
        backward_requires_autograd: bool = True,
        defer_pp_recv: bool = False,
    ):
        # TODO: we don't support input/weight backward split with torch.compile
        _check_torch_compile_compatibility(stages, self.__class__.__name__)
        self.pp_group_size = stages[0].group_size
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
            scale_grads=scale_grads,
            backward_requires_autograd=backward_requires_autograd,
            defer_pp_recv=defer_pp_recv,
        )
        self.stage_index_to_group_rank = generate_stage_to_rank_mapping(
            self.pp_group_size, self._num_stages, style="v"
        )
        for stage in self._stages:
            stage.stage_index_to_group_rank = self.stage_index_to_group_rank

        self.n_local_stages = len(stages)
        if self.n_local_stages != 2:
            raise ValueError(
                "ZBV requires exactly 2 stages per rank, but got "
                f"{self.n_local_stages}."
            )
        if n_microbatches < self._num_stages:
            raise ValueError(
                "DualPipeV requires at least as many microbatches as stages, but got "
                f"{n_microbatches} microbatches and {self._num_stages} stages."
            )

        self.rank = stages[0].group_rank
        self.num_stages = stages[0].num_stages

        # 1. Create the pipeline_order (all ranks do this calculation)
        # This will be used to keep track of the current state of the entire pipeline
        # pipeline_order[rank] = [Action(computation_type, microbatch_index, stage_index), ...]
        self.pipeline_order: dict[int, list[_Action | None]] = {}
        for rank in range(self.pp_group_size):
            rank_ops = self._calculate_single_rank_operations(rank)
            self.pipeline_order[rank] = rank_ops

        # Initialize the pipeline order with communication necessary to run with _PipelineScheduleRuntime
        self._prepare_schedule_with_comms(self.pipeline_order)

    def _calculate_single_rank_operations(self, rank) -> list[_Action | None]:
        actions: list[_Action | None] = []
        counters: dict[
            tuple[int, _ComputationType], int
        ] = {}  # (stage_index, computation_type) -> mb_index
        weight_queue = []  # Queue of (stage_index, mb_index) for pending weight actions

        num_ranks = self.pp_group_size
        num_chunks = self._n_microbatches

        rank_to_stages = generate_rank_to_stage_mapping(
            num_ranks, num_ranks * 2, style="v"
        )
        stage0_index, stage1_index = rank_to_stages[rank]

        def increment_backward_counts(stage_index: int):
            """Helper method to increment BACKWARD_INPUT and BACKWARD_WEIGHT counters when FULL_BACKWARD is used."""
            input_key = (stage_index, BACKWARD_INPUT)
            weight_key = (stage_index, BACKWARD_WEIGHT)
            counters[input_key] = counters.get(input_key, 0) + 1
            counters[weight_key] = counters.get(weight_key, 0) + 1

        def add_overlap_f_b(
            actions: list,
            forward_stage: int,
            backward_stage: int,
        ):
            """Helper method to add an overlapped forward+backward action which tracks microbatch index."""
            # Create new overlapped forward+backward action with sub_actions
            forward_key = (forward_stage, FORWARD)
            backward_key = (backward_stage, BACKWARD_INPUT)

            forward_mb = counters.get(forward_key, 0)
            backward_mb = counters.get(backward_key, 0)

            sub_actions = (
                _Action(forward_stage, FORWARD, forward_mb),
                _Action(backward_stage, FULL_BACKWARD, backward_mb),
            )
            actions.append(_Action(-1, OVERLAP_F_B, None, sub_actions))

            # Update counters for sub_actions
            counters[forward_key] = forward_mb + 1
            increment_backward_counts(backward_stage)

        def add_action(
            actions: list,
            stage_index: int,
            computation_type: _ComputationType,
        ):
            # Regular single action, for FULL_BACKWARD we only use the BACKWARD_INPUT counter
            key = (
                (stage_index, computation_type)
                if computation_type != FULL_BACKWARD
                else (stage_index, BACKWARD_INPUT)
            )
            mb_index = counters.get(key, 0)
            actions.append(_Action(stage_index, computation_type, mb_index))

            # If FULL_BACKWARD is used, just increment the separate BACKWARD_INPUT and BACKWARD_WEIGHT counters
            if computation_type == FULL_BACKWARD:
                increment_backward_counts(stage_index)
            else:
                # If BACKWARD_INPUT is updated, add corresponding weight action to queue
                if computation_type == BACKWARD_INPUT:
                    # Add weight action to queue for later processing
                    weight_queue.append((stage_index, mb_index))
                counters[key] = mb_index + 1

        def add_weight_action_if_pending(actions: list):
            """Helper method to add a weight action from the queue."""
            if not weight_queue:
                return  # No pending weight actions, skip
            # Pop the oldest weight action from the queue
            actual_stage_index, weight_mb_index = weight_queue.pop(0)
            actions.append(
                _Action(
                    actual_stage_index,
                    BACKWARD_WEIGHT,
                    weight_mb_index,
                )
            )
            # Update the counter for the actual stage that was processed
            weight_key = (actual_stage_index, BACKWARD_WEIGHT)
            counters[weight_key] = counters.get(weight_key, 0) + 1

        # Step 1: F0
        step_1 = (num_ranks - rank - 1) * 2
        for _ in range(step_1):
            add_action(actions, stage0_index, FORWARD)

        # Step 2: F0F1
        step_2 = rank + 1
        for _ in range(step_2):
            add_action(actions, stage0_index, FORWARD)
            add_action(actions, stage1_index, FORWARD)

        # Step 3: I1W1F1 (Use zero bubble)
        step_3 = num_ranks - rank - 1
        for _ in range(step_3):
            add_action(actions, stage1_index, BACKWARD_INPUT)
            add_weight_action_if_pending(actions)
            add_action(actions, stage1_index, FORWARD)

        # Step 4 (Main step): F0B1-F1B0 (combined, overlapped forward+backward)
        step_4 = num_chunks - num_ranks * 2 + rank + 1
        for i in range(step_4):
            if i == 0 and rank == num_ranks - 1:
                # NOTE: We don't overlap these two chunks to further reduce bubble size.
                add_action(actions, stage0_index, FORWARD)
                add_action(actions, stage1_index, FULL_BACKWARD)
            else:
                add_overlap_f_b(
                    actions,
                    forward_stage=stage0_index,
                    backward_stage=stage1_index,
                )
            add_overlap_f_b(
                actions,
                forward_stage=stage1_index,
                backward_stage=stage0_index,
            )

        # Step 5: B1-F1B0
        step_5 = num_ranks - rank - 1
        for _ in range(step_5):
            add_action(actions, stage1_index, FULL_BACKWARD)
            add_overlap_f_b(
                actions,
                forward_stage=stage1_index,
                backward_stage=stage0_index,
            )

        # Step 6: B1B0 (The second half of the chunks use zero bubble)
        step_6 = rank + 1
        enable_zb = False
        for i in range(step_6):
            if i == step_6 // 2 and rank % 2 == 1:
                enable_zb = True
            comp_type = BACKWARD_INPUT if enable_zb else FULL_BACKWARD
            add_action(actions, stage1_index, comp_type)
            if i == step_6 // 2 and rank % 2 == 0:
                enable_zb = True
            comp_type = BACKWARD_INPUT if enable_zb else FULL_BACKWARD
            add_action(actions, stage0_index, comp_type)

        # Step 7: W0B0
        step_7 = num_ranks - rank - 1
        for _ in range(step_7):
            add_weight_action_if_pending(actions)
            comp_type = BACKWARD_INPUT if enable_zb else FULL_BACKWARD
            add_action(actions, stage0_index, comp_type)

        # Step 8: W0
        step_8 = rank + 1
        for _ in range(step_8):
            add_weight_action_if_pending(actions)

        return actions


def get_schedule_class(schedule_name: str):
    """
    Maps a schedule name (case insensitive) to its corresponding class object.

    Args:
        schedule_name (str): The name of the schedule.
    """
    schedule_map = {
        "1F1B": Schedule1F1B,
        "Interleaved1F1B": ScheduleInterleaved1F1B,
        "GPipe": ScheduleGPipe,
        "LoopedBFS": ScheduleLoopedBFS,
        "InterleavedZeroBubble": ScheduleInterleavedZeroBubble,
        "PipelineScheduleSingle": PipelineScheduleSingle,
        "PipelineScheduleMulti": PipelineScheduleMulti,
        "ZBVZeroBubble": ScheduleZBVZeroBubble,
        "DualPipeV": ScheduleDualPipeV,
    }
    lowercase_keys = {k.lower(): k for k in schedule_map}
    lowercase_schedule_name = schedule_name.lower()
    if lowercase_schedule_name not in lowercase_keys:
        raise ValueError(
            f"Unknown schedule name '{schedule_name}'. The valid options are {list(schedule_map.keys())}"
        )
    return schedule_map[lowercase_keys[lowercase_schedule_name]]


def _simulate_comms_compute(
    pipeline_order, stage_to_rank: Callable[[int], int], num_stages: int
):
    """This function dry-run simulates the actions in the schedule from the perspective of all ranks, and flags
    any deadlocks caused by missing or misordered communications.  It also simulates any bubbles in time where a rank
    can not execute any action due to waiting for unmet dependencies.  The total number of simulator steps can be used
    as a metric for unit tests involving IR optimization passes as reordering and merging of IR can reduce the number
    of simulated steps.

    The simulation is not high-fidelity and does not model overlapping of compute and communication, or cuda streams.
    Future work may be to enhance this and model the compute time, comms overlap, and even memory.
    """
    pipeline_order = {
        rank: [a for a in pipeline_order[rank] if a is not None]
        for rank in sorted(pipeline_order)
    }
    _schedule: dict[int, list[_Action | None]] = {
        rank: [] for rank in sorted(pipeline_order)
    }

    _prev_ops_rank: dict[int, set[_Action]] = {rank: set() for rank in _schedule}

    def add_to_schedule(rank: int, action: _Action | None):
        _schedule[rank].append(action)
        if action is not None:
            _prev_ops_rank[rank].add(action)

    def _ready_to_schedule(action: _Action | None) -> bool:
        if action is None:
            return True

        stage_idx = action.stage_index
        prev_ops = _prev_ops_rank[stage_to_rank(stage_idx)]
        if action.computation_type == F:
            if action.stage_index == 0:
                return True
            elif (
                _Action(action.stage_index, RECV_F, action.microbatch_index) in prev_ops
            ):
                return True
            elif (
                _Action(action.stage_index - 1, F, action.microbatch_index) in prev_ops
            ):
                return True
            return False
        elif action.computation_type in (BACKWARD_INPUT, FULL_BACKWARD):
            if action.stage_index == num_stages - 1:
                return True
            if _Action(action.stage_index, RECV_B, action.microbatch_index) in prev_ops:
                return True
            if (
                _Action(action.stage_index + 1, BACKWARD_INPUT, action.microbatch_index)
                in prev_ops
            ):
                return True
            if (
                _Action(action.stage_index + 1, FULL_BACKWARD, action.microbatch_index)
                in prev_ops
            ):
                return True
            return False
        elif action.computation_type == BACKWARD_WEIGHT:
            return True
        elif action.computation_type == SEND_F:
            expected_f = _Action(action.stage_index, F, action.microbatch_index)
            return expected_f in prev_ops
        elif action.computation_type == RECV_F:
            peer_stage_idx = stage_idx - 1
            expected_send = _Action(peer_stage_idx, SEND_F, action.microbatch_index)
            return expected_send in _prev_ops_rank[stage_to_rank(peer_stage_idx)]
        elif action.computation_type == SEND_B:
            expected_b = _Action(
                action.stage_index, BACKWARD_INPUT, action.microbatch_index
            )
            expected_bw = _Action(
                action.stage_index, FULL_BACKWARD, action.microbatch_index
            )
            return expected_b in prev_ops or expected_bw in prev_ops
        elif action.computation_type == RECV_B:
            peer_stage_idx = stage_idx + 1
            expected_send = _Action(peer_stage_idx, SEND_B, action.microbatch_index)
            return expected_send in _prev_ops_rank[stage_to_rank(peer_stage_idx)]
        else:
            raise ValueError(f"Unsupported action type {action}")

    while pipeline_order:
        progress = False
        for rank in sorted(pipeline_order):
            if len(pipeline_order[rank]) == 0:
                continue

            action = pipeline_order[rank][0]
            if _ready_to_schedule(action):
                if action is not None:
                    add_to_schedule(rank, action)
                pipeline_order[rank].pop(0)
                progress = True
            else:
                add_to_schedule(rank, None)

        for i in sorted(pipeline_order, reverse=True):
            if len(pipeline_order[i]) == 0:
                del pipeline_order[i]

        # hacky, but do a second pass to replace any 'none' at this timestep with a real action, if it got unblocked
        # by one of the later ranks
        for rank in sorted(pipeline_order):
            if len(pipeline_order[rank]) == 0:
                continue

            if _schedule[rank][-1] is not None:
                continue

            action = pipeline_order[rank][0]
            if _ready_to_schedule(action):
                if action is not None:
                    _schedule[rank][-1] = action
                    _prev_ops_rank[rank].add(action)
                pipeline_order[rank].pop(0)

        for i in sorted(pipeline_order, reverse=True):
            if len(pipeline_order[i]) == 0:
                del pipeline_order[i]

        if not progress:
            print("WIP comms schedule:\n", _format_pipeline_order(_schedule))
            for rank in pipeline_order:
                print(f"{rank=} next action= {pipeline_order[rank][0]}")
            raise ValueError("Schedule is not progressing")

    return _schedule


def _dump_chrometrace(schedule, filename):
    """
    This function dumps a schedule IR into a chrometrace format so it can be visualized.

    It is currently very basic and only serves as a graphical alternative to dumping the schedule IR as text.

    As future work we may extend this to include more accurate heuristics for durations, or let users input durations,
    add 'flow events' to let the UI show the connection between sends and recvs, and model cuda streams for comm/compute
    as separate streams on the chrometrace view.
    """
    events = []
    for rank in sorted(schedule):
        for timestep, action in enumerate(schedule[rank]):
            if action is None:
                continue
            events.append(
                {
                    "name": str(action),
                    "cat": (
                        "computation"
                        if action.computation_type in (F, B, W)
                        else "communication"
                    ),
                    "ph": "X",
                    "pid": rank,
                    "tid": rank,
                    "ts": timestep,
                    "dur": 1,
                }
            )
    import json

    with open(filename, "w") as f:
        json.dump({"traceEvents": events}, f)


def _check_torch_compile_compatibility(
    stages: list[_PipelineStageBase], schedule_name: str
):
    """
    Check if the schedule is compatible with torch.compile.

    Args:
        stages: List of pipeline stages to check
        schedule_name: Name of the schedule for error message

    Raises:
        RuntimeError: If any stage uses torch.compile
    """
    for stage in stages:
        if not isinstance(stage.submod, torch.nn.Module):
            continue

        for module in stage.submod.modules():
            if isinstance(module, OptimizedModule):
                raise RuntimeError(
                    f"The {schedule_name} schedule is not supported with "
                    "stage modules that have used torch.compile. "
                    f"Found OptimizedModule in {type(module).__name__}"
                )
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import operator
from collections.abc import Sequence
from typing import Any, cast

import torch
from torch.distributed.tensor import DTensor
from torch.fx.node import map_aggregate
from torch.nn.attention.flex_attention import BlockMask
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten


__all__ = [
    "TensorChunkSpec",
    "split_args_kwargs_into_chunks",
    "merge_chunks",
]

logger = logging.getLogger(__name__)

"""
_debug_mask_minibatches specifies to send masked versions of the mini-batch
through instead of micro-batch slices--this can be used for more stable
numerical testing (see [A Note About Correctness Testing])
"""
_debug_mask_minibatches = False


class _CustomReducer:
    """
    Custom reducer class that can be used to specify a custom operation that
    reduces losses of multiple microbatches into one value.

    Example:
    >>> # xdoctest: +SKIP
    >>> sum_reducer = _CustomReducer(
    >>>     torch.tensor(0.0),
    >>>     lambda a, b: a + b
    >>> )
    """

    def __init__(self, init_value, reduce_fn):
        self.init_value = init_value
        self.reduce_fn = reduce_fn


class _LossReducer(_CustomReducer):
    pass


sum_reducer = _LossReducer(torch.tensor(0.0), operator.add)

# Default chunking dimension is 0. This is used for the case where the user did
# not specify a chunking dimension.
DEFAULT_CHUNK_DIM = 0


class TensorChunkSpec:
    """
    Class used to specify chunking of inputs
    """

    def __init__(self, split_dim):
        self.split_dim = split_dim

    split_dim: int

    def __repr__(self):
        return (
            f"{self.__class__.__module__}.{self.__class__.__name__}({self.split_dim})"
        )

    def __str__(self):
        return f"TensorChunkSpec({self.split_dim})"

    @staticmethod
    def from_tuple(
        chunk_dims: tuple[int, ...],
    ):
        """
        A helper for creating a tuple of `TensorChunkSpec` from a tuple of chunk
        dimensions (int's).
        Example:
            >>> # xdoctest: +SKIP
            >>> # There are three positional arguments to the model, and
            >>> # we are chunking them along dimension 0, 0 and 1, respectively
            >>> args_chunk_spec = TensorChunkSpec.from_tuple((0, 0, 1))
        """
        args_chunk_spec = map_aggregate(
            chunk_dims,
            lambda dim: TensorChunkSpec(dim),  # type: ignore[arg-type,return-value]
        )
        return args_chunk_spec

    @staticmethod
    def from_dict(
        chunk_dims: dict[str, int],
    ):
        """
        A helper for creating a dictionary of `TensorChunkSpec` from a
        dictionary of chunk dimensions (int's).
        Example:
            >>> # xdoctest: +SKIP
            >>> # Chunk dimension 0 for the "id" argument, 1 for the "mask" argument
            >>> kwargs_chunk_spec = TensorChunkSpec.from_dict({"id": 0, "mask": 1})
        """
        kwargs_chunk_spec = map_aggregate(
            chunk_dims,
            lambda dim: TensorChunkSpec(dim),  # type: ignore[arg-type,return-value]
        )
        return kwargs_chunk_spec


# Class used to specify replication of inputs
class _Replicate:
    pass


def _split_block_mask(
    block_mask: BlockMask,
    num_chunks: int,
) -> list[BlockMask]:
    """Given a block mask, split the block mask along the batch dimension (dim0).

    Args:
        block_mask: Block mask to split
        num_chunks: Number of chunks to split the block mask into

    Returns:
        chunk_block_masks: List of chunked block masks
    """

    # BlockMask will broadcast if B is 1.
    if block_mask.kv_num_blocks.size(0) == 1:
        return [block_mask] * num_chunks

    if not block_mask.kv_num_blocks.size(0) >= num_chunks:
        raise AssertionError(
            "Block mask has fewer batch size than the number of chunks. "
        )

    batch_dim = 0
    kv_num_blocks_chunks = torch.tensor_split(
        block_mask.kv_num_blocks, num_chunks, batch_dim
    )
    kv_indices_chunks = torch.tensor_split(block_mask.kv_indices, num_chunks, batch_dim)
    full_kv_num_blocks_chunks = (
        torch.tensor_split(block_mask.full_kv_num_blocks, num_chunks, batch_dim)
        if block_mask.full_kv_num_blocks is not None
        else [None] * num_chunks
    )
    full_kv_indices_chunks = (
        torch.tensor_split(block_mask.full_kv_indices, num_chunks, batch_dim)
        if block_mask.full_kv_indices is not None
        else [None] * num_chunks
    )

    chunk_block_masks = []
    batch_offset = 0
    for chunk_idx in range(num_chunks):

        def create_mask_mod(idx):
            def batch_offset_mask_mod(b, h, q_idx, kv_idx):
                b_offset = torch.full_like(b, idx)
                return block_mask.mask_mod(b + b_offset, h, q_idx, kv_idx)

            return batch_offset_mask_mod

        chunk_block_masks.append(
            BlockMask.from_kv_blocks(
                kv_num_blocks=kv_num_blocks_chunks[chunk_idx],
                kv_indices=kv_indices_chunks[chunk_idx],
                full_kv_num_blocks=full_kv_num_blocks_chunks[chunk_idx],
                full_kv_indices=full_kv_indices_chunks[chunk_idx],
                BLOCK_SIZE=block_mask.BLOCK_SIZE,
                mask_mod=create_mask_mod(batch_offset),
                seq_lengths=block_mask.seq_lengths,
            )
        )
        batch_offset += kv_num_blocks_chunks[chunk_idx].size(0)
    return chunk_block_masks


def _split_tensor(
    tensor: torch.Tensor,
    spec: TensorChunkSpec,
    num_chunks: int,
) -> Sequence[torch.Tensor]:
    """Given a tensor, and a chunking spec, split the tensor.
    Args:

        tensor: Tensor to split
        spec: Chunking spec
        num_chunks: Number of chunks to split the tensor into

    Returns:
        chunk_tensors: List of chunked tensors
    """

    if not tensor.size(spec.split_dim) >= num_chunks:
        raise AssertionError(
            f"Tensor size {tensor.size(spec.split_dim)} is smaller than num_chunks"
        )

    _is_dtensor = isinstance(tensor, DTensor)

    if _is_dtensor:
        # Split locally and wrap each chunk as a DTensor with the correct
        # global shape.  We avoid DTensor dispatch (which would all-gather
        # Shard(split_dim) to Replicate) and avoid local_map (whose
        # from_local call assumes even sharding and computes wrong global
        # sizes when local shards differ across ranks).
        placements = tensor.placements
        mesh = tensor.device_mesh
        local_tensor = tensor.to_local()
        local_chunks = torch.tensor_split(local_tensor, num_chunks, spec.split_dim)
        # Compute the correct global chunk sizes along split_dim.
        global_split_size = tensor.shape[spec.split_dim]
        quotient, remainder = divmod(global_split_size, num_chunks)
        global_stride = tensor.stride()
        chunk_tensors_list: list[torch.Tensor] = []
        for i, local_chunk in enumerate(local_chunks):
            chunk_shape = list(tensor.shape)
            chunk_shape[spec.split_dim] = quotient + (1 if i < remainder else 0)
            chunk_tensors_list.append(
                DTensor.from_local(
                    local_chunk,
                    mesh,
                    placements,
                    shape=torch.Size(chunk_shape),
                    stride=global_stride,
                    run_check=False,
                )
            )
        chunk_tensors: Sequence[torch.Tensor] = chunk_tensors_list  # type: ignore[assignment]
    else:
        chunk_tensors = torch.tensor_split(tensor, num_chunks, spec.split_dim)

    # tensor_split on a leaf tensor produces non-leaf views that won't
    # accumulate .grad during torch.autograd.backward().  Call retain_grad()
    # on those views so that stage_backward() can read .grad from them.
    if tensor.requires_grad and tensor.is_leaf:
        for chunk in chunk_tensors:
            chunk.retain_grad()

    if not _debug_mask_minibatches:
        return chunk_tensors

    def _expand_chunks(
        orig: torch.Tensor, *chunks: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        expanded = []
        idx = 0
        for chunk in chunks:
            new_val = torch.zeros_like(orig)
            upper = idx + chunk.size(spec.split_dim)
            slices: list[slice] = [slice(None)] * new_val.ndim
            slices[spec.split_dim] = slice(idx, upper)
            new_val[slices] = chunk
            expanded.append(new_val)
            idx += chunk.size(spec.split_dim)
        return tuple(expanded)

    if _is_dtensor:
        placements = tensor.placements
        mesh = tensor.device_mesh
        global_shape = tensor.shape
        global_stride = tensor.stride()
        local_expanded = _expand_chunks(
            tensor.to_local(),
            *(cast(DTensor, c).to_local() for c in chunk_tensors),
        )
        return [
            DTensor.from_local(
                t,
                mesh,
                placements,
                shape=global_shape,
                stride=global_stride,
                run_check=False,
            )
            for t in local_expanded
        ]
    else:
        return list(_expand_chunks(tensor, *chunk_tensors))


def _shard_dict_of_args(
    args_dict,
    args_chunk_spec,
    num_chunks,
):
    """
    Given a dictionary of args, and a dictionary of chunking specs, shard the
    args according to the chunking specs.

    Args:
        args_dict: Dictionary of args
        args_chunk_spec: Dictionary of chunking specs
        num_chunks: Number of chunks to shard the args into

    Returns:
        args_split: List of sharded args
    """

    if not args_dict:
        return [{} for _ in range(num_chunks)]

    if not len(args_dict) == len(args_chunk_spec):
        raise AssertionError(
            f"args_dict.keys() = {list(args_dict.keys())} "
            f"args_chunk_spec.keys() = {list(args_chunk_spec.keys())}"
        )
    if args_chunk_spec is None:
        raise AssertionError("args_chunk_spec should have been set by caller")

    values, tree_spec = tree_flatten(
        args_dict, is_leaf=lambda x: isinstance(x, BlockMask)
    )
    chunk_specs, _ = tree_flatten(
        args_chunk_spec, is_leaf=lambda x: isinstance(x, BlockMask)
    )

    # First check and find the actual number of chunks
    split_sizes = []
    for v, spec in zip(values, chunk_specs, strict=True):
        # The original logic is "spec is _Replicate". This doesn't seem to be
        # correct. But we keep it for backward compatibility.
        if spec is _Replicate or isinstance(spec, _Replicate):
            split_sizes.append(num_chunks)
        elif isinstance(v, torch.Tensor):
            if not isinstance(spec, TensorChunkSpec):
                raise AssertionError(f"Expected TensorChunkSpec, got {type(spec)}")
            split_sizes.append(v.size(spec.split_dim))
        elif isinstance(v, BlockMask):
            if not isinstance(spec, TensorChunkSpec):
                raise AssertionError(f"Expected TensorChunkSpec, got {type(spec)}")
            if not spec.split_dim == 0:
                raise AssertionError("BlockMask only supports split_dim=0")
            # BlockMask will broadcast if B is 1.
            if v.kv_num_blocks.size(0) == 1:
                split_sizes.append(num_chunks)
            else:
                split_sizes.append(v.kv_num_blocks.size(0))
        else:
            raise ValueError(
                f"Unsupported chunk spec: {spec} and value: {v} combination."
            )
    result_num_chunks = min(*split_sizes, num_chunks)

    flat_split_results: list[Any] = [[] for _ in range(result_num_chunks)]
    for v, spec in zip(values, chunk_specs, strict=True):
        v_splits: Sequence[Any] = []
        if spec is _Replicate or isinstance(spec, _Replicate):
            v_splits = [v] * result_num_chunks
        elif isinstance(v, torch.Tensor):
            v_splits = _split_tensor(v, spec, result_num_chunks)
        elif isinstance(v, BlockMask):
            v_splits = _split_block_mask(v, result_num_chunks)
        else:
            raise ValueError(
                f"Unsupported chunk spec: {spec} and value: {v} combination."
            )

        for _flat_split_result, _v_split in zip(
            flat_split_results, v_splits, strict=True
        ):
            _flat_split_result.append(_v_split)

    return [
        tree_unflatten(_flat_split_result, tree_spec)
        for _flat_split_result in flat_split_results
    ]


def split_args_kwargs_into_chunks(
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
    chunks: int,
    args_chunk_spec: tuple[TensorChunkSpec, ...] | None = None,
    kwargs_chunk_spec: dict[str, TensorChunkSpec] | None = None,
) -> tuple[list[tuple], list[dict]]:
    """
    Given a sequence of args and kwargs, split them into a number of chunks
    according to  their respective chunking specs.

    Args:
        args: Tuple of args
        kwargs: Dict of kwargs
        chunks: Number of chunks to split the args and kwargs into
        args_chunk_spec: chunking specs for args, in same shape as args
        kwargs_chunk_spec: chunking specs for kwargs, in same shape as kwargs

    Returns:
        args_split: List of sharded args
        kwargs_split: List of sharded kwargs
    """
    # Given `args` and `kwargs`, we want to yield a set of `chunks` args and kwargs such that
    # the constituent Tensor values have been sharded/replicated according to the `args_chunk_spec`
    # and `kwargs_chunk_spec` specifications. The steps are as follows:
    #
    # 1. Use pytree.tree_flatten to flatten each arg and its spec into a 1d array of values.
    #    To use a running example: suppose our inputs look like
    #
    #       args = ([A, [B, C]], D) args_spec = ([None, [None, TensorChunkSpec]], None)
    #       (kwargs not shown but it's a similar process)
    #
    #    Then for this step we would end up with
    #
    #       args = ([A, B, C], D) args_spec = ([None, None, TensorChunkSpec], None)
    #
    # 2. Shard or replicate the arguments subject to the policy in the spec. Suppose chunks = 2
    #
    #       args = ([[A, A], [B, B], [C_1, C_2]], [D, D])
    #
    # 3. Rotate the nesting order such that chunks are the outer dimension
    #
    #       args_chunks = [
    #           ([A, B, C_1], D),
    #           ([A, B, C_2], D),
    #       ]
    #
    # 4. Unflatten each chunk according to the spec
    #
    #       args_chunks = [
    #           ([A, [B, C_1]], D),
    #           ([A, [B, C_2]], D),
    #       ]

    # TODO: _debug_mask_minibatches
    # Handle the case where kwargs is None
    if kwargs is None:
        kwargs = {}

    # If user did not provide args_chunk_spec or kwargs_chunk_spec, we extend
    # their format and use default chunking along dim 0
    def default_spec(v):
        if isinstance(v, torch.Tensor | BlockMask):
            return TensorChunkSpec(DEFAULT_CHUNK_DIM)
        else:
            return _Replicate()

    if args_chunk_spec is None:
        args_chunk_spec = tree_map(
            default_spec, args, is_leaf=lambda v: isinstance(v, BlockMask)
        )

    if kwargs_chunk_spec is None:
        kwargs_chunk_spec = tree_map(
            default_spec, kwargs, is_leaf=lambda v: isinstance(v, BlockMask)
        )

    args_split_dict = _shard_dict_of_args(
        dict(enumerate(args)),
        dict(enumerate(args_chunk_spec)),
        chunks,
    )
    real_num_chunks = len(args_split_dict)

    kwargs_split = _shard_dict_of_args(
        kwargs,
        kwargs_chunk_spec,
        real_num_chunks,
    )

    if len(kwargs_split) < real_num_chunks:
        # In case kwargs are sharded into less chunks
        # e.g. when `args` has no tensor, just values
        real_num_chunks = len(kwargs_split)
        # Re-shard args
        args_split_dict = _shard_dict_of_args(
            dict(enumerate(args)),
            dict(enumerate(args_chunk_spec)),
            real_num_chunks,
        )

    if len(args_split_dict) != len(kwargs_split):
        raise RuntimeError(
            "args and kwargs are split into different number of chunks: "
            f"{len(args_split_dict)}, {len(kwargs_split)}"
        )

    args_split = [
        tuple(chunk_args[i] for i in range(len(chunk_args)))
        for chunk_args in args_split_dict
    ]

    return args_split, kwargs_split


def merge_chunks(
    chunks: list[Any],
    chunk_spec,
):
    """
    Given a list of chunks, merge them into a single value according to
    the chunk spec.

    Args:
        chunks: list of chunks
        chunk_spec: Chunking spec for the chunks

    Returns:
        value: Merged value
    """
    # This is essentially the inverse of `split_args_kwargs_into_chunks`, so the
    # steps are similar to the steps in that function but in reverse. Given the
    # input values:
    #
    #       chunks = [
    #           ([A, [B, C_1]], D),
    #           ([A, [B, C_2]], D),
    #       ]
    #       args_spec = ([None, [None, TensorChunkSpec]], None)
    #
    # 1. Flatten the chunks according to the chunk_spec
    #
    #       chunks_flat = [
    #           ([A, B, C_1], D),
    #           ([A, B, C_2], D),
    #       ]
    #
    # 2. Rotate the nesting order such that chunks are the inner dimension
    #
    #       value_inner = ([A, B, [C_1, C_2]], D)
    #
    # 3. Concatenate sharded arguments
    #
    #       value_combined = ([A, B, C], D)
    #
    # 4. Unflatten the combined args given the spec
    #
    #       value = ([A, [B, C]], D)

    # Preliminary: flatten the chunk spec
    if chunk_spec is not None:
        spec_flattened, flatten_spec = tree_flatten(chunk_spec)
    else:
        # If chunk_spec is not provided, we will merge chunks along the default dimension (0), for all output fields
        # We obtain the output structure by flattening chunk 0 and generate the chunk_spec
        chunk0_flat, flatten_spec = tree_flatten(chunks[0])
        spec_flattened = [TensorChunkSpec(DEFAULT_CHUNK_DIM)] * len(chunk0_flat)

    # Stage 1: flatten chunks
    # chunks_flattened : [num chunks, num args]
    chunks_flattened = []

    for chunk in chunks:
        chunk_flattened, _ = tree_flatten(chunk)
        if len(chunk_flattened) != len(spec_flattened):
            raise ValueError(f"Chunk {chunk} did not match chunk spec {chunk_spec}")

        chunks_flattened.append(chunk_flattened)

    # Stage 2 and 3: Rotate nesting order s.t. chunks are inner dimension and
    #                concatenate sharded operands
    # args_flattened : [num args]
    args_flattened: list[Any] = []
    for arg_idx, arg in enumerate(spec_flattened):
        if isinstance(arg, TensorChunkSpec):
            partial_values = [
                chunks_flattened[chunk_idx][arg_idx]
                for chunk_idx in range(len(chunks_flattened))
            ]

            if _debug_mask_minibatches:
                # Infer size of individual chunks by running `tensor_split` again
                overall_shape = partial_values[0].shape
                for val in partial_values[1:]:
                    if not val.shape == overall_shape:
                        raise AssertionError(
                            f"Expected shape {overall_shape}, got {val.shape}"
                        )
                meta_chunks = torch.tensor_split(
                    torch.empty(*overall_shape, device="meta"),
                    sections=len(partial_values),
                    dim=arg.split_dim,
                )

                values_to_cat = []
                chunk_start_idx = 0
                if not len(partial_values) == len(meta_chunks):
                    raise AssertionError(
                        f"Expected len(partial_values) == len(meta_chunks), got {len(partial_values)} != {len(meta_chunks)}"
                    )

                for partial_value, meta_chunk in zip(
                    partial_values, meta_chunks, strict=True
                ):
                    chunk_end_idx = chunk_start_idx + meta_chunk.size(arg.split_dim)

                    slice_indices = [slice(None, None, None)] * partial_value.ndim
                    slice_indices[arg.split_dim] = slice(chunk_start_idx, chunk_end_idx)
                    sliced = partial_value[slice_indices]
                    values_to_cat.append(sliced)

                    chunk_start_idx = chunk_end_idx

            else:
                values_to_cat = partial_values

            # Validate DTensor consistency: either all values are DTensors
            # or none are. A mix indicates a bug in the pipeline stage.
            dtensor_flags = [isinstance(v, DTensor) for v in values_to_cat]
            if any(dtensor_flags):
                if not all(dtensor_flags):
                    raise AssertionError(
                        "merge_chunks: expected all values to be DTensors or "
                        "none to be DTensors, got a mix"
                    )
                # All DTensors must have matching placements.
                placements = values_to_cat[0].placements
                for i, v in enumerate(values_to_cat[1:], 1):
                    if v.placements != placements:
                        raise AssertionError(
                            f"merge_chunks: placement mismatch at chunk {i}: "
                            f"expected {placements}, got {v.placements}"
                        )
                mesh = values_to_cat[0].device_mesh
                local_cat = torch.cat(
                    [v.to_local() for v in values_to_cat],
                    dim=arg.split_dim,
                )
                cat_shape = list(values_to_cat[0].shape)
                cat_shape[arg.split_dim] = sum(
                    v.shape[arg.split_dim] for v in values_to_cat
                )
                cat_shape = torch.Size(cat_shape)
                cat_stride = values_to_cat[0].stride()
                args_flattened.append(
                    DTensor.from_local(
                        local_cat,
                        mesh,
                        placements,
                        shape=cat_shape,
                        stride=cat_stride,
                        run_check=False,
                    )
                )
            else:
                args_flattened.append(torch.cat(values_to_cat, dim=arg.split_dim))
        elif isinstance(arg, _CustomReducer):
            reduced_val = arg.init_value

            for chunk_idx in range(len(chunks_flattened)):
                reduced_val = arg.reduce_fn(
                    reduced_val, chunks_flattened[chunk_idx][arg_idx]
                )

            args_flattened.append(reduced_val)
        else:
            value = chunks_flattened[0][arg_idx]
            for chunk_idx in range(1, len(chunks_flattened)):
                if not chunks_flattened[chunk_idx][arg_idx] == value:
                    raise AssertionError(
                        f"Expected {value}, got {chunks_flattened[chunk_idx][arg_idx]}"
                    )
            args_flattened.append(value)

    # Stage 4: Unflatten combined args
    return tree_unflatten(args_flattened, flatten_spec)
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates
from ._IR import Pipe, pipe_split, pipeline, SplitPoint
from .schedules import (
    _ScheduleForwardOnly,
    Schedule1F1B,
    ScheduleDualPipeV,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
    ScheduleInterleavedZeroBubble,
    ScheduleLoopedBFS,
    ScheduleZBVZeroBubble,
)
from .stage import build_stage, PipelineStage


__all__ = [
    "Pipe",
    "pipe_split",
    "SplitPoint",
    "pipeline",
    "PipelineStage",
    "build_stage",
    "Schedule1F1B",
    "ScheduleGPipe",
    "ScheduleInterleaved1F1B",
    "ScheduleLoopedBFS",
    "ScheduleInterleavedZeroBubble",
    "ScheduleZBVZeroBubble",
    "ScheduleDualPipeV",
]
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import cast, Literal, overload, Protocol, TYPE_CHECKING, TypeAlias

import torch
from torch import fx
from torch.distributed._mesh_layout import _MeshLayout
from torch.distributed.tensor import DTensor
from torch.utils._pytree import tree_flatten, tree_unflatten


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor.placement_types import Placement


logger = logging.getLogger(__name__)


class GetMeshCallback(Protocol):
    """Callback to create/retrieve a DeviceMesh from its cache key components."""

    def __call__(
        self,
        mesh_dim_names: tuple[str, ...],
        mesh_layout: _MeshLayout | None,
    ) -> DeviceMesh: ...


# Key for mesh cache: (mesh_dim_names, mesh_layout)
# mesh_layout is the _MeshLayout object containing shape and stride (not actual ranks).
# This uniquely identifies a mesh within the same "universe" where all stages share
# the same rank tensor.
MeshCacheKey: TypeAlias = tuple[tuple[str, ...], _MeshLayout | None]


class PipeliningMetadataError(RuntimeError):
    """Raised on metadata mismatches during pipeline communication."""


@dataclass(frozen=True, slots=True)
class _TensorMeta:
    """Tensor metadata for recv buffer allocation and validation.

    For plain tensors, these are the tensor's actual attributes.
    For DTensors, these are LOCAL shard attributes; global attributes
    are stored in :class:`_DTensorMeta`.
    """

    shape: torch.Size
    stride: tuple[int, ...]
    dtype: torch.dtype
    requires_grad: bool

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> _TensorMeta:
        """Create metadata from a plain tensor.

        Args:
            tensor: A plain ``torch.Tensor`` (not DTensor).

        Returns:
            Metadata capturing shape, stride, dtype, and requires_grad.

        Raises:
            TypeError: If ``tensor`` is a DTensor.
        """
        if isinstance(tensor, DTensor):
            raise PipeliningMetadataError(
                "Expected plain tensor, got DTensor. Use _DTensorMeta.from_dtensor instead."
            )
        return _TensorMeta(
            shape=tensor.shape,
            stride=tensor.stride(),
            dtype=tensor.dtype,
            requires_grad=tensor.requires_grad,
        )

    def to_tensor(self, device: torch.device | str) -> torch.Tensor:
        """Reconstruct a tensor on ``device`` from this metadata.

        Args:
            device: Target device for the tensor.

        Returns:
            An empty strided tensor on ``device``.
        """
        t = _make_tensor_from_meta(self, device)
        if t.is_floating_point():
            t.requires_grad_(self.requires_grad)
        return t

    def get_diff(self, other: _TensorMeta) -> list[str]:
        """Return field-by-field differences with ``other``.

        Args:
            other: Metadata to compare against.

        Returns:
            List of human-readable difference strings (empty if equal).
        """
        if self == other:
            return []

        diffs = []
        if self.shape != other.shape:
            diffs.append(f"shape mismatch: {self.shape} vs {other.shape}")
        if self.stride != other.stride:
            diffs.append(f"stride mismatch: {self.stride} vs {other.stride}")
        if self.dtype != other.dtype:
            diffs.append(f"dtype mismatch: {self.dtype} vs {other.dtype}")
        # requires_grad is intentionally excluded: it is a runtime concern
        # determined by has_backward and grad context, not a metadata invariant.
        return diffs


@dataclass(frozen=True, slots=True)
class _DTensorMeta(_TensorMeta):
    """DTensor metadata extending :class:`_TensorMeta` with distribution info.

    Inherited fields (shape, stride, etc.) are LOCAL shard attributes.
    Additional fields capture global shape and placement information
    needed to reconstruct a :class:`DTensor` via ``DTensor.from_local()``.

    The :class:`DeviceMesh` is **not** stored (not serializable for P2P);
    it is looked up from :class:`_MeshCache` using
    ``(mesh_dim_names, mesh_layout)`` as the key.
    """

    # Global DTensor properties (for reconstruction)
    global_shape: torch.Size = field(default_factory=lambda: torch.Size([]))
    global_stride: tuple[int, ...] = field(default=())

    # DTensor distribution properties
    placements: tuple[Placement, ...] = field(
        default=()
    )  # e.g., (Shard(0), Replicate())

    # Mesh identification - used to look up the correct DeviceMesh from cache
    mesh_dim_names: tuple[str, ...] = field(default=())  # e.g., ("tp",) or ("dp", "tp")
    mesh_layout: _MeshLayout | None = field(
        default=None
    )  # _MeshLayout with shape/stride - uniquely identifies mesh within the same universe

    @staticmethod
    def from_dtensor(dtensor: DTensor) -> _DTensorMeta:
        """Create metadata from a DTensor.

        Args:
            dtensor: The DTensor to extract metadata from.

        Returns:
            Metadata capturing both local and global attributes.
        """
        device_mesh = dtensor.device_mesh

        return _DTensorMeta(
            # Local tensor attributes (for recv buffer allocation)
            shape=dtensor._local_tensor.shape,
            stride=dtensor._local_tensor.stride(),
            dtype=dtensor.dtype,
            requires_grad=dtensor.requires_grad,
            # Global DTensor attributes (for reconstruction)
            global_shape=dtensor.shape,
            global_stride=dtensor.stride(),
            # Distribution info
            placements=dtensor._spec.placements,
            mesh_dim_names=(
                tuple(device_mesh.mesh_dim_names) if device_mesh.mesh_dim_names else ()
            ),
            mesh_layout=device_mesh._layout,
        )

    @property
    def mesh_cache_key(self) -> MeshCacheKey:
        """Cache key ``(mesh_dim_names, mesh_layout)`` for mesh lookup."""
        return (self.mesh_dim_names, self.mesh_layout)

    def to_dtensor(self, device: torch.device | str, mesh: DeviceMesh) -> DTensor:
        """Reconstruct a DTensor on ``device`` with placements.

        Args:
            device: Target device for the local tensor.
            mesh: The ``DeviceMesh`` to attach.

        Returns:
            A DTensor on ``device``.
        """
        local_tensor = _make_tensor_from_meta(self, device)
        # Set requires_grad after from_local() so that the from_local
        # operation itself is not recorded in the autograd graph.
        dt = DTensor.from_local(
            local_tensor,
            device_mesh=mesh,
            placements=self.placements,
            shape=self.global_shape,
            stride=self.global_stride,
            run_check=False,
        )
        if self.requires_grad and dt.is_floating_point():
            dt = dt.requires_grad_(True)
        return cast(DTensor, dt)

    def get_diff(self, other: _TensorMeta) -> list[str]:
        """Return field-by-field differences, including DTensor-specific fields.

        Args:
            other: Metadata to compare against.

        Returns:
            List of human-readable difference strings (empty if equal).
        """
        if self == other:
            return []

        # Get base class differences (compares local shape/stride/dtype/requires_grad)
        # NOTE: Use explicit class call instead of super() because
        # @dataclass(slots=True) on both parent and child can break super().
        diffs = _TensorMeta.get_diff(self, other)

        # Add DTensor-specific comparisons if other is also _DTensorMeta
        if isinstance(other, _DTensorMeta):
            if self.global_shape != other.global_shape:
                diffs.append(
                    f"global_shape mismatch: {self.global_shape} vs {other.global_shape}"
                )
            if self.global_stride != other.global_stride:
                diffs.append(
                    f"global_stride mismatch: {self.global_stride} vs {other.global_stride}"
                )
            if self.placements != other.placements:
                diffs.append(
                    f"placements mismatch: {self.placements} vs {other.placements}"
                )
            if self.mesh_dim_names != other.mesh_dim_names:
                diffs.append(
                    f"mesh_dim_names mismatch: {self.mesh_dim_names} vs {other.mesh_dim_names}"
                )
            if self.mesh_layout != other.mesh_layout:
                diffs.append(
                    f"mesh_layout mismatch: {self.mesh_layout} vs {other.mesh_layout}"
                )
        else:
            diffs.append("type: _DTensorMeta vs _TensorMeta")

        return diffs


# Type alias for union of tensor metadata types
TensorMeta: TypeAlias = _TensorMeta | _DTensorMeta


# Not frozen: fields are populated incrementally during forward and
# backward metadata inference or from user provided static metadata
@dataclass(slots=True)
class _StageMeta:
    """Consolidated tensor metadata for a pipeline stage's forward and backward passes."""

    inputs: tuple[TensorMeta, ...] | None = None
    outputs: tuple[TensorMeta, ...] | None = None
    input_grads: tuple[TensorMeta | None, ...] | None = None
    output_grads: tuple[TensorMeta | None, ...] | None = None

    def has_any(self) -> bool:
        """Check if any metadata field is populated."""
        return any(
            v is not None
            for v in [self.inputs, self.outputs, self.input_grads, self.output_grads]
        )

    def has_dtensors(self) -> bool:
        """Check if any input/output metadata is DTensor type."""
        for metas in [self.inputs, self.outputs]:
            if metas and any(isinstance(m, _DTensorMeta) for m in metas if m):
                return True
        return False

    def is_complete_for_forward(self) -> bool:
        """Check if forward metadata is fully populated."""
        return self.inputs is not None and self.outputs is not None


@dataclass(frozen=True, slots=True)
class _StageForwardMeta:
    """Forward metadata transmitted from stage *i* to stage *i+1* during inference."""

    forward_metas: tuple[TensorMeta, ...]  # Stage i's outputs → Stage i+1's inputs


@dataclass(frozen=True, slots=True)
class _StageBackwardMeta:
    """Backward metadata transmitted from stage *i* to stage *i-1* during inference.

    Gradient placements may differ from forward activations
    (e.g., ``Replicate`` → ``Partial``).
    """

    backward_metas: tuple[
        TensorMeta | None, ...
    ]  # Stage i's input_grads → Stage i-1's output_grads


def _make_tensor_from_meta(
    meta: _TensorMeta,
    device: torch.device | str,
) -> torch.Tensor:
    """Create a tensor from metadata.

    Args:
        meta: Metadata with shape, stride, and dtype.
        device: Target device for the tensor.

    Returns:
        Empty tensor preserving the exact memory layout.
    """
    return torch.empty_strided(
        size=meta.shape,
        stride=meta.stride,
        dtype=meta.dtype,
        device=device,
    )


def _derive_grad_metas(
    tensor_metas: tuple[TensorMeta, ...],
) -> tuple[TensorMeta | None, ...]:
    """Derive gradient metadata from tensor metadata.

    Returns metadata with the same shape/stride/dtype but ``requires_grad=False``.
    Entries where the source has ``requires_grad=False`` become ``None``.
    """

    def derive_one(m: TensorMeta) -> TensorMeta | None:
        if not m.requires_grad:
            return None
        if isinstance(m, _DTensorMeta):
            return None
        return _TensorMeta(
            shape=m.shape,
            stride=m.stride,
            dtype=m.dtype,
            requires_grad=False,
        )

    return tuple(derive_one(m) for m in tensor_metas)


class _MeshCache:
    """Cache for :class:`DeviceMesh` objects keyed by ``(mesh_dim_names, mesh_layout)``.

    Assumes all pipeline stages share the same rank tensor (true for
    TorchTitan-style frameworks where meshes derive from a common world).
    """

    def __init__(self, get_mesh_cb: GetMeshCallback | None = None) -> None:
        self._cache: dict[MeshCacheKey, DeviceMesh] = {}
        self._get_mesh_cb = get_mesh_cb

    def get_mesh(self, key: MeshCacheKey) -> DeviceMesh:
        """Return a cached mesh, or create one via the callback.

        Args:
            key: Cache key ``(mesh_dim_names, mesh_layout)``.

        Returns:
            The ``DeviceMesh``.

        Raises:
            PipeliningMetadataError: If not cached and no callback provided.
        """
        if key in self._cache:
            return self._cache[key]

        mesh_dim_names, mesh_layout = key

        if self._get_mesh_cb is None:
            raise PipeliningMetadataError(
                f"Mesh not found in cache for mesh_dim_names={mesh_dim_names}, "
                f"mesh_layout={mesh_layout}, and no get_mesh callback provided. "
                f"Provide a get_mesh callback or use DTensors in static mode."
            )

        mesh = self._get_mesh_cb(mesh_dim_names, mesh_layout)
        if mesh is None:
            raise PipeliningMetadataError(
                f"Mesh lookup failed: callback returned None for "
                f"mesh_dim_names={mesh_dim_names}, mesh_layout={mesh_layout}. "
                f"Ensure all stages use meshes from the same universe."
            )
        self._cache[key] = mesh
        return mesh

    def put(self, key: MeshCacheKey, mesh: DeviceMesh) -> None:
        """Add a mesh to the cache."""
        self._cache[key] = mesh

    def update_from_tensors(self, tensors: tuple[torch.Tensor | None, ...]) -> None:
        """Extract and cache meshes from any :class:`DTensor` instances in *tensors*."""
        for tensor in tensors:
            if isinstance(tensor, DTensor):
                mesh = tensor.device_mesh
                dim_names = tuple(mesh.mesh_dim_names) if mesh.mesh_dim_names else ()
                mesh_layout = mesh._layout
                key = (dim_names, mesh_layout)
                if key not in self._cache:
                    self._cache[key] = mesh

    def __contains__(self, key: MeshCacheKey) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)


# ============================================================================
# Inference mode enum
# ============================================================================


class InferenceMode(Enum):
    """Pipeline-level metadata inference mode, determined collectively across all PP ranks.

    The mode is set by the schedule (not individual stages) because
    ``has_backward`` is only known at schedule creation time and all
    stages must agree to avoid P2P hangs.

    .. attribute:: STATIC

        All stages have sufficient metadata; runtime inference is skipped.

    .. attribute:: DYNAMIC

        At least one stage requires runtime metadata inference.
    """

    STATIC = "static"
    DYNAMIC = "dynamic"

    @classmethod
    def needs_dynamic(cls, meta: _StageMeta, stage_has_backward: bool) -> bool:
        """Determine whether dynamic metadata inference is needed for a stage.

        Args:
            meta: Stage metadata from user-provided args.
            stage_has_backward: Whether a backward pass will be performed.

        Returns:
            ``True`` if dynamic inference is needed.
        """
        # Case 1: Forward metadata incomplete → needs DYNAMIC
        if not meta.is_complete_for_forward():
            return True

        # Case 2: No DTensors → STATIC is fine (bwd metadata derivable from fwd metadata)
        if not meta.has_dtensors():
            return False

        # Case 3: No backward needed → STATIC is fine (don't need grad metadata)
        if not stage_has_backward:
            return False

        # Case 4: DTensors with backward but missing ANY grad metadata → needs DYNAMIC
        # Both input_grads AND output_grads are required for static mode with DTensors
        if meta.input_grads is None or meta.output_grads is None:
            return True

        # Case 5: DTensors with complete grads → STATIC is fine
        return False


# ============================================================================
# Utility functions
# ============================================================================


def flatten_args(args, *, detach: bool = False):
    """Flatten ``args`` into a list, optionally detaching tensors.

    Args:
        args: Nested arguments to flatten.
        detach: If ``True``, detach tensors while preserving ``requires_grad``.

    Returns:
        ``(new_args, flat_detached_args)`` when ``detach=True``;
        ``flat_args`` list otherwise.
    """
    flat_args, treespec = tree_flatten(args)

    if detach:
        flat_detached = [
            a.detach().requires_grad_(a.requires_grad)
            if isinstance(a, torch.Tensor)
            else a
            for a in flat_args
        ]
        new_args = tree_unflatten(flat_detached, treespec)
        return new_args, flat_detached

    return flat_args


# Backward compatibility alias
def flatten_args_detach(args):
    """Flatten and detach. Deprecated: use ``flatten_args(args, detach=True)``."""
    return flatten_args(args, detach=True)


def generate_stage_to_rank_mapping(
    pp_size: int, num_stages: int, style: str = "loop"
) -> dict[int, int]:
    """
    Compute the stage id to rank mapping for either a looped or V-style schedule.

    Most commonly num_stages == pp_size * 2, but this function can be used to
    compute the mapping for any number of stages per rank.
    """
    mapping = {}
    if style == "loop":
        for stage_index in range(num_stages):
            mapping[stage_index] = stage_index % pp_size
    elif style == "v":
        if num_stages % pp_size != 0:
            raise ValueError(
                f"num_stages {num_stages} must be evenly divisible by pp_size {pp_size} for V schedules"
            )

        rank_index = 0
        for stage_index in range(num_stages):
            mapping[stage_index] = rank_index
            # don't change rank if we are on the border (to keep v shape)
            if (stage_index + 1) % pp_size == 0:
                continue
            if (stage_index // pp_size) % 2 == 0:
                rank_index += 1
            else:
                rank_index -= 1
    else:
        raise ValueError(f"Style {style} is not supported.")
    return mapping


def generate_rank_to_stage_mapping(
    pp_size: int, num_stages: int, style: str = "loop"
) -> dict[int, list[int]]:
    """
    Compute the rank to stage id mapping for either a looped or V-style schedule.

    This function inverts the stage_to_rank_mapping to get which stages are assigned to each rank.

    Returns a dictionary mapping rank -> list of stage indices assigned to that rank.
    """
    stage_to_rank = generate_stage_to_rank_mapping(pp_size, num_stages, style)

    # Invert the mapping: rank -> list of stages
    rank_to_stages: dict[int, list[int]] = {}
    for stage_id, rank in stage_to_rank.items():
        if rank not in rank_to_stages:
            rank_to_stages[rank] = []
        rank_to_stages[rank].append(stage_id)

    # Sort the stage lists for each rank to ensure consistent ordering
    for stages in rank_to_stages.values():
        stages.sort()

    return rank_to_stages


@dataclass(slots=True)
class PipeInfo:
    """
    Captures information for a pipeline (`Pipe` object).
    """

    graph: fx.Graph
    num_stages: int
    has_loss_and_backward: bool


# ============================================================================
# Metadata extraction helpers
# ============================================================================


def extract_tensor_meta(tensor: torch.Tensor) -> TensorMeta:
    """Extract metadata from a tensor.

    Handles both plain Tensor and DTensor correctly: DTensors are
    dispatched to ``_DTensorMeta.from_dtensor`` which captures local
    shard attributes plus global shape/placement info, while plain
    tensors use ``_TensorMeta.from_tensor``.

    Args:
        tensor: A plain tensor or DTensor.

    Returns:
        ``_TensorMeta`` for plain tensors, ``_DTensorMeta`` for DTensors.
    """
    if isinstance(tensor, DTensor):
        return _DTensorMeta.from_dtensor(tensor)
    else:
        return _TensorMeta.from_tensor(tensor)


@overload
def extract_tensor_metas(
    tensors: tuple[torch.Tensor, ...] | None,
    *,
    allow_none: Literal[False] = ...,
) -> tuple[TensorMeta, ...] | None: ...


@overload
def extract_tensor_metas(
    tensors: tuple[torch.Tensor | None, ...] | None,
    *,
    allow_none: Literal[True],
) -> tuple[TensorMeta | None, ...] | None: ...


def extract_tensor_metas(
    tensors: tuple[torch.Tensor | None, ...] | tuple[torch.Tensor, ...] | None,
    *,
    allow_none: bool = False,
) -> tuple[TensorMeta | None, ...] | None:
    """Extract metadata from a tuple of tensors.

    Args:
        tensors: Tuple of tensors (may include ``None`` when ``allow_none=True``).
        allow_none: If ``True``, preserve ``None`` elements (for gradients).

    Returns:
        Tuple of ``TensorMeta``, or ``None`` if ``tensors`` is ``None``.

    Raises:
        PipeliningMetadataError: If ``None`` found and ``allow_none=False``.
    """
    if tensors is None:
        return None

    metas_with_none: list[TensorMeta | None] = []
    has_none = False
    for t in tensors:
        if isinstance(t, torch.Tensor):
            metas_with_none.append(extract_tensor_meta(t))
        else:
            has_none = True
            metas_with_none.append(None)
    if not allow_none and has_none:
        raise PipeliningMetadataError(
            "None values are not allowed in tensor metadata tuples. "
            "Use allow_none=True for optional values."
        )
    return tuple(metas_with_none)


def to_local_if_dtensor(tensor: torch.Tensor, detach: bool = False) -> torch.Tensor:
    """Convert a DTensor to its local shard, or return a plain tensor as-is.

    When ``detach=True``, the tensor is detached before conversion —
    this applies to both DTensors and plain tensors.

    Args:
        tensor: A tensor that may be a DTensor.
        detach: If ``True``, detach before ``to_local()`` to avoid
            redistribution during backward.

    Returns:
        The local tensor component.
    """
    maybe_detached_tensor = tensor.detach() if detach else tensor
    if isinstance(maybe_detached_tensor, DTensor):
        return maybe_detached_tensor.to_local()
    return maybe_detached_tensor


@overload
def validate_and_normalize_to_tuple(
    args: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor] | None,
    allow_none: Literal[False] = ...,
) -> tuple[torch.Tensor, ...] | None: ...


@overload
def validate_and_normalize_to_tuple(
    args: torch.Tensor
    | tuple[torch.Tensor | None, ...]
    | list[torch.Tensor | None]
    | None,
    allow_none: Literal[True] = ...,
) -> tuple[torch.Tensor | None, ...] | None: ...


def validate_and_normalize_to_tuple(
    args: torch.Tensor
    | tuple[torch.Tensor, ...]
    | tuple[torch.Tensor | None, ...]
    | list[torch.Tensor]
    | list[torch.Tensor | None]
    | None,
    allow_none: bool = False,
) -> tuple[torch.Tensor | None, ...] | tuple[torch.Tensor, ...] | None:
    """Normalize ``args`` to a tuple and validate that all elements are tensors.

    Args:
        args: A single tensor, tuple/list of tensors, or ``None``.
        allow_none: If ``True``, permit ``None`` elements (for gradients).

    Returns:
        Tuple of tensors, or ``None`` if ``args`` is ``None``.

    Raises:
        PipeliningMetadataError: On non-tensor values
            (or ``None`` when ``allow_none=False``).
    """
    if args is None:
        return None
    elif isinstance(args, torch.Tensor):
        return (args,)
    elif isinstance(args, (tuple, list)):
        for i, arg in enumerate(args):
            if arg is None:
                if not allow_none:
                    raise PipeliningMetadataError(
                        f"Stage arg[{i}] is None. "
                        f"Stage args must be tensors. Use kwargs for optional values."
                    )
                continue
            if not isinstance(arg, torch.Tensor):
                raise PipeliningMetadataError(
                    f"Stage arg[{i}] has type {type(arg).__name__}. "
                    f"All stage args must be tensors. Use kwargs for non-tensor inputs."
                )
        # Normalize list to tuple
        return tuple(args) if isinstance(args, list) else args
    else:
        raise PipeliningMetadataError(
            f"Stage args must be a tensor, tuple, or list of tensors, got {type(args).__name__}."
        )


# ============================================================================
# Validation functions
# ============================================================================


def validate_metadata(
    desc: str,
    expected: TensorMeta,
    actual: torch.Tensor | TensorMeta,
    *,
    raise_on_mismatch: bool = False,
    warn_on_mismatch: bool = False,
) -> list[str]:
    """
    Compare expected metadata against actual tensor or metadata.

    This is the unified validation/comparison function that uses get_diff() from
    metadata classes. Works with both plain tensors and DTensors.

    For plain tensors: compares shape/stride/dtype/requires_grad.
    For DTensors: compares all properties including global shape and placements.

    Args:
        desc: Description for error/warning messages.
        expected: Expected tensor metadata (_TensorMeta or _DTensorMeta).
        actual: Actual tensor or metadata to compare against.
        raise_on_mismatch: If True, raise PipeliningMetadataError on mismatch.
        warn_on_mismatch: If True, issue a warning on mismatch.

    Returns:
        List of differences (empty if metadata matches).

    Raises:
        PipeliningMetadataError: If raise_on_mismatch=True and differences exist.
    """
    # Extract metadata if actual is a tensor
    if isinstance(actual, torch.Tensor):
        actual_meta = extract_tensor_meta(actual)
    else:
        actual_meta = actual

    # Type check: ensure both are same type for meaningful comparison
    if type(expected) is not type(actual_meta):
        type_diff = [
            f"type: expected {type(expected).__name__}, got {type(actual_meta).__name__}"
        ]
        if raise_on_mismatch:
            raise PipeliningMetadataError(f"{desc}: {type_diff[0]}")
        if warn_on_mismatch:
            warnings.warn(
                f"{desc}: Metadata type mismatch. {type_diff[0]}. "
                f"Using dynamically inferred metadata instead.",
                UserWarning,
                stacklevel=2,
            )
        return type_diff

    # Use get_diff() from the metadata class
    diffs = expected.get_diff(actual_meta)

    if diffs:
        if raise_on_mismatch:
            raise PipeliningMetadataError(f"{desc}: {'; '.join(diffs)}")
        if warn_on_mismatch:
            warnings.warn(
                f"{desc}: Metadata mismatch. {'; '.join(diffs)}. "
                f"Using dynamically inferred metadata instead.",
                UserWarning,
                stacklevel=2,
            )

    return diffs


def validate_tensors_metadata(
    desc: str,
    expected: tuple[TensorMeta | None, ...],
    actual: tuple[torch.Tensor | TensorMeta | None, ...],
    *,
    raise_on_mismatch: bool = True,
    warn_on_mismatch: bool = False,
) -> list[str]:
    """Validate metadata for a tuple of tensors element-wise.

    Args:
        desc: Description prefix for error/warning messages.
        expected: Tuple of expected metadata (may include ``None`` for grads).
        actual: Tuple of actual tensors or metadata to compare against.
        raise_on_mismatch: If ``True``, raise on the first mismatch.
        warn_on_mismatch: If ``True``, issue warnings for mismatches.

    Returns:
        Aggregated list of difference strings.

    Raises:
        PipeliningMetadataError: If lengths differ or on mismatch.
    """
    if len(expected) != len(actual):
        msg = f"{desc}: expected {len(expected)} tensors, got {len(actual)}"
        if raise_on_mismatch:
            raise PipeliningMetadataError(msg)
        if warn_on_mismatch:
            warnings.warn(msg, UserWarning, stacklevel=2)
        return [msg]

    all_diffs: list[str] = []
    for i, (exp, act) in enumerate(zip(expected, actual, strict=True)):
        if exp is None and act is None:
            continue
        if exp is None or act is None:
            msg = (
                f"{desc}[{i}]: expected {'None' if exp is None else 'metadata'}, "
                f"got {'None' if act is None else 'metadata'}"
            )
            if raise_on_mismatch:
                raise PipeliningMetadataError(msg)
            if warn_on_mismatch:
                warnings.warn(msg, UserWarning, stacklevel=2)
            all_diffs.append(msg)
            continue
        diffs = validate_metadata(
            f"{desc}[{i}]",
            exp,
            act,
            raise_on_mismatch=raise_on_mismatch,
            warn_on_mismatch=warn_on_mismatch,
        )
        all_diffs.extend(diffs)
    return all_diffs


def validate_static_arg_grad_correspondence(
    stage_index: int,
    args: tuple[torch.Tensor, ...],
    grads: tuple[torch.Tensor | None, ...],
    is_input: bool,
) -> None:
    """
    Validate the args↔grads contract for static mode.

    Enforces four rules for each (arg, grad) pair:
      1. len(args) must equal len(grads).
      2. If arg.requires_grad is False, grad must be None.
      3. If arg.requires_grad is True and grad is None, emit a warning
         (this is legal at pipeline boundaries but may indicate a bug).
      4. If arg is a DTensor with requires_grad=True and grad is not None,
         grad must also be a DTensor.

    Args:
        stage_index: The stage index for error messages.
        args: Tuple of forward tensors.
        grads: Tuple of gradient tensors (can include None).
        is_input: True for input_args/input_grads, False for output_args/output_grads.

    Raises:
        PipeliningMetadataError: If any hard rule (1, 2, or 4) is violated.
    """
    kind = "input" if is_input else "output"
    args_name = f"{kind}_args"
    grads_name = f"{kind}_grads"

    # Rule 1: lengths must match
    if len(args) != len(grads):
        raise PipeliningMetadataError(
            f"Stage {stage_index}: {grads_name} length ({len(grads)}) does not match "
            f"{args_name} length ({len(args)}). Each forward tensor must have a "
            f"corresponding gradient entry (use None for tensors that don't require grad)."
        )

    for i, (arg, grad) in enumerate(zip(args, grads, strict=True)):
        # Rule 2: no grad for a non-differentiable arg
        if not arg.requires_grad and grad is not None:
            raise PipeliningMetadataError(
                f"Stage {stage_index}: {args_name}[{i}] has requires_grad=False, "
                f"but {grads_name}[{i}] is not None ({type(grad).__name__}). "
                f"Non-differentiable tensors must have None as their gradient entry."
            )

        # Rule 3: missing grad for a differentiable arg (warn, don't raise)
        if arg.requires_grad and grad is None:
            warnings.warn(
                f"Stage {stage_index}: {args_name}[{i}] has requires_grad=True, "
                f"but {grads_name}[{i}] is None. This is legal at pipeline boundaries "
                f"but may indicate a missing gradient.",
                UserWarning,
                stacklevel=2,
            )

        # Rule 4: DTensor arg must have DTensor grad
        if (
            isinstance(arg, DTensor)
            and arg.requires_grad
            and grad is not None
            and not isinstance(grad, DTensor)
        ):
            raise PipeliningMetadataError(
                f"Stage {stage_index}: {args_name}[{i}] is a DTensor with requires_grad=True, "
                f"but {grads_name}[{i}] is {type(grad).__name__}, expected DTensor or None. "
                f"DTensor gradients may have different placements than forward tensors."
            )
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import operator
import warnings
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.distributed.config as dist_config
import torch.fx as fx
import torch.nn as nn
from torch._subclasses.fake_tensor import is_fake_tensor
from torch.distributed._composable.replicate_with_fsdp import replicate, ReplicateModule
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.pipelining._utils import (
    _derive_grad_metas,
    _DTensorMeta,
    _make_tensor_from_meta,
    _MeshCache,
    _StageBackwardMeta,
    _StageForwardMeta,
    _StageMeta,
    _TensorMeta,
    extract_tensor_meta,
    extract_tensor_metas,
    flatten_args,
    GetMeshCallback,
    InferenceMode,
    PipeInfo,
    PipeliningMetadataError,
    TensorMeta,
    to_local_if_dtensor,
    validate_and_normalize_to_tuple,
    validate_static_arg_grad_correspondence,
    validate_tensors_metadata,
)
from torch.distributed.tensor import DTensor
from torch.fx.node import Argument, map_aggregate
from torch.nn.parallel import DistributedDataParallel

from ._backward import (
    _autograd_grad_for_inputs,
    stage_backward,
    stage_backward_input,
    stage_backward_weight,
)
from ._debug import map_debug_info


__all__ = [
    "PipelineStage",
    "build_stage",
]

logger = logging.getLogger(__name__)


def _normalize_model_output_as_tuple(output: Any) -> tuple[Any]:
    """[Note: pipeline model output type]

    The output of the model passed to pipelining can be any type, controlled by the user.

    However, there are 2 API surfaces that complicate this.
    (1) the outputs of intermediate stages are passed via Send/Recv ops to subsequent stages. The implicit assumption
    is that each element of the outputs is a tensor.  Otherwise, Send/Recv would not be supported.  The exception
    is the last layer of the model, which can output anything any which won't be communicated via Send/Recv.
    (2) the outputs of the last layer of the model are returned to the user, or, passed to the loss function.
    The loss function can be written in any way, such that its inputs match the outputs of the model.

    It would be convenient if we could strictly type the output signature of the pipeline stage wrapping the model,
    but we do not want to impose an unnecessary constraint on user provided models.

    Currently, we let user provided models return either a Tensor or a tuple of Tensors from each stage. Due to
    torch.export tracing, compiled models may also return a list instead of a Tuple, which we will normalize back to a
    tuple for consistency.

    TODO: should we be stricter about asserting that stage modules (intermediate and output) all return only Tensor
    values?
    """
    if type(output) is list:
        # HACK: this is a hacky workaround for the fact that export creates
        # output in list format
        output = tuple(output)

    # Unify output form to tuple for easy correspondence with
    # `act_send_info`
    output_tuple = output if type(output) is tuple else (output,)
    return output_tuple


class _RecvInfo:
    """Input tensor descriptor for a pipeline stage.

    Handles both received activations from a previous stage
    (``is_root_arg=False``) and root-level model inputs provided
    by the user (``is_root_arg=True``).
    """

    def __init__(
        self,
        input_name: str,
        source: int | None,
        buffer: torch.Tensor | None,
        tensor_meta: TensorMeta | None,
        *,
        is_root_arg: bool = False,
    ):
        # Name of this input
        self.input_name = input_name
        # Stage index of the source of this input (None for root args)
        self.source = source
        # Buffer to receive the input into (None for root args)
        self.buffer = buffer
        # Tensor metadata for validation and DTensor reconstruction
        self.tensor_meta = tensor_meta
        # Whether this is a root-level model input (no recv needed)
        self.is_root_arg = is_root_arg

    def __repr__(self):
        if self.is_root_arg:
            return f"_RecvInfo(input={self.input_name}, root_arg=True)"
        meta_type = type(self.tensor_meta).__name__ if self.tensor_meta else "None"
        buffer_shape = self.buffer.size() if self.buffer is not None else "None"
        return f"_RecvInfo(input={self.input_name}, source={self.source}, shape={buffer_shape}, meta={meta_type})"


# Cache of per-direction P2P communicators, keyed (weakly) by the PP process
# group they are derived from. Looped/V schedules construct several stage chunks
# per rank that share one PP group; they must share the same forward/backward
# comms (and issue the split_group collective only once) so creation stays
# consistent and cheap across all ranks. Stage chunks are constructed serially
# within a rank, so the first miss performs the split and the rest hit the cache.
# The key is a weakref, so entries are
# dropped automatically once the parent group is destroyed (e.g. via
# destroy_process_group / reinitialization), avoiding stale dead communicators.
_PP_DIRECTION_GROUP_CACHE: "weakref.WeakKeyDictionary[dist.ProcessGroup, tuple[dist.ProcessGroup, dist.ProcessGroup]]" = weakref.WeakKeyDictionary()


def _build_p2p_direction_groups(
    group: dist.ProcessGroup | None,
) -> tuple[dist.ProcessGroup, dist.ProcessGroup]:
    """Create two communicators over the same ranks as ``group``, one per data-flow
    direction: ``downstream`` carries traffic flowing ``r -> r+1`` (forward
    activations) and ``upstream`` carries ``r -> r-1`` (backward gradients).

    Pipeline P2P normally shares a single communicator for both directions, which
    serializes every send/recv in one FIFO. Coalescing makes a single mixed
    send+recv batch deadlock-free, but across *separate* batches (pipeline skew,
    looped / V schedules, skip connections) the shared FIFO can still form a
    dependency cycle and deadlock. Routing the two directions onto separate
    communicators / streams removes that cross-batch coupling and restores
    full-duplex bandwidth.

    Uses ``split_group``, which is collective over ``group``'s own ranks (not the
    whole world), so it composes with PP as a sub-axis of a larger device mesh.
    Requires the default process group to be device-bound (e.g.
    ``init_process_group(..., device_id=...)``), which ``split_group`` needs for
    NCCL; torchcomms binds the device automatically.
    """
    parent = group if group is not None else dist.distributed_c10d._get_default_group()
    cached = _PP_DIRECTION_GROUP_CACHE.get(parent)
    if cached is not None:
        return cached

    split_ranks = [list(range(dist.get_world_size(parent)))]
    # split_group splits the parent's communicator, so the default process group
    # must be device-bound (NCCL) -- torchcomms binds the device automatically. If
    # it is not, split_group raises its own device error.
    downstream = dist.split_group(
        parent_pg=group, split_ranks=split_ranks, group_desc="pp_p2p_downstream"
    )
    upstream = dist.split_group(
        parent_pg=group, split_ranks=split_ranks, group_desc="pp_p2p_upstream"
    )
    # All parent ranks are members of the single split.
    if not isinstance(downstream, dist.ProcessGroup):
        raise AssertionError(f"expected dist.ProcessGroup, got {type(downstream)}")
    if not isinstance(upstream, dist.ProcessGroup):
        raise AssertionError(f"expected dist.ProcessGroup, got {type(upstream)}")
    logger.info("Pipeline P2P: using per-direction (downstream/upstream) communicators")
    _PP_DIRECTION_GROUP_CACHE[parent] = (downstream, upstream)
    return downstream, upstream


class _PipelineStageBase(ABC):
    """Base class for pipeline stages.

    Defines common methods used by ``_PipelineStage`` (tracing frontend)
    and ``PipelineStage`` (manual frontend).
    """

    def __init__(
        self,
        submodule: torch.nn.Module,
        stage_index: int,
        num_stages: int,
        device: torch.device,
        group: dist.ProcessGroup | None = None,
        dw_builder: Callable[[], Callable[..., None]] | None = None,
    ):
        """
        Args:
            submodule: The module to be executed in this stage.
            stage_index: The index of this stage.
            num_stages: The total number of stages in this pipeline.
            device: The device to run this stage on.
            group: Process group for communication. Defaults to the
                default process group if ``None``.
            dw_builder: Builder function that produces a ``dw_runner``
                for deferred weight updates in F/I/W zero-bubble
                schedules. If ``None``, a runner is generated
                automatically via autograd graph traversal.
        """
        super().__init__()
        if stage_index >= num_stages:
            raise ValueError(
                f"Stage index {stage_index} is out of range of {num_stages}"
            )

        self.submod = submodule
        self.stage_index = stage_index
        self.num_stages = num_stages
        self.device = device
        self.group = group

        # Downstream (data flowing r -> r+1: forward activations) and upstream
        # (r -> r-1: backward gradients) P2P communicators. Auto-enabled when
        # TorchComms is in use (its split path is always available and the
        # single-comm FIFO deadlock is most acute there); the config flag
        # torch.distributed.config.pipeline_per_direction_p2p (env
        # TORCH_DISTRIBUTED_PIPELINE_PER_DIRECTION_P2P) force-enables it on other
        # backends. When disabled both alias ``self.group`` so behavior is
        # byte-for-byte unchanged.
        self.p2p_per_direction = (
            dist_config.pipeline_per_direction_p2p
            or dist.distributed_c10d._use_torchcomms_enabled()
        )
        self._downstream_group: dist.ProcessGroup | None
        self._upstream_group: dist.ProcessGroup | None
        if self.p2p_per_direction:
            self._downstream_group, self._upstream_group = _build_p2p_direction_groups(
                group
            )
        else:
            self._downstream_group = group
            self._upstream_group = group

        self.dw_builder = dw_builder

        # backward state
        self.backward_state: dict[int, tuple[Any, ...]] = {}

        # store dw_runner per microbatch_id
        self.dw_runner: dict[int, Callable[..., None]] = {}

        # `group_rank` is rank in process group `group`.
        self.group_rank = dist.get_rank(self.group)
        self.group_size = dist.get_world_size(self.group)
        if self.group_size > self.num_stages:
            raise RuntimeError(
                f"Pipeline group size {self.group_size} cannot be larger than number of stages {self.num_stages}"
            )

        # Run time states
        # map microbatch ID to list of forward tensor args
        self.fwd_cache: dict[int, tuple[Any, list[torch.Tensor]]] = {}
        # map microbatch ID to list of backward grad tensor args
        self.bwd_cache: dict[int, tuple[torch.Tensor | None, ...]] = {}
        # Caching chunk outputs for final output merge or reduction
        self.output_chunks: list[Any] = []

        # Initialize has_backward to false; this will be set to true if loss
        # function is passed to pipeline schedule
        self.has_backward = False
        # Log prefix
        self.log_prefix = f"[Stage {self.stage_index}]"

        # Forward infra
        self.args_recv_info: dict[int, tuple[_RecvInfo, ...]] = {}
        self.act_send_info: dict[int, list] = {}

        # Backward infra will be created lazily
        self.grad_recv_info: dict = {}
        self.grad_send_info: list | None = None

        # To be populated later by the Schedule
        self.chunks: int | None = None
        self.stage_index_to_group_rank: dict[int, int] = {
            i: i % self.group_size for i in range(self.num_stages)
        }

        # DTensor support: mesh cache for looking up DeviceMesh by (dim_names, layout)
        self._mesh_cache = _MeshCache()

        # Per-chunk runtime validation is expensive; only enable under
        # TORCH_DISTRIBUTED_DEBUG=DETAIL for debugging shape/dtype mismatches.
        self._runtime_validate = dist.get_debug_level() == dist.DebugLevel.DETAIL

        # DTensor support: consolidated stage metadata container
        # Contains inputs, outputs, input_grads, output_grads metadata
        self._stage_meta = _StageMeta()

    @property
    def has_backward(self) -> bool:
        """
        Returns true if this stage has a backward pass.
        """
        return self._has_backward

    @has_backward.setter
    def has_backward(self, has_backward: bool):
        self._has_backward = has_backward

    @property
    def is_first(self):
        """
        Returns true if this stage is the first stage in the pipeline.
        """
        return self.stage_index == 0

    @property
    def is_last(self):
        """
        Returns true if this stage is the last stage in the pipeline.
        """
        return self.stage_index == self.num_stages - 1

    def _validate_stage_tensors(
        self,
        desc: str,
        expected: tuple[TensorMeta | None, ...] | None,
        actual: tuple[torch.Tensor | None, ...],
    ) -> None:
        """Validate actual tensors against expected metadata.

        Raises:
            PipeliningMetadataError: If metadata is missing or mismatched.
        """
        if expected is None:
            raise PipeliningMetadataError(f"{desc}: no metadata available")
        validate_tensors_metadata(desc, expected, actual)

    def _check_chunk_id(self, chunk_id: int):
        if self.chunks is None:
            raise RuntimeError(
                "Attempted to access chunk_id before chunks have been configured."
            )
        if chunk_id >= self.chunks:
            raise RuntimeError(
                f"Chunk id {chunk_id} is out of range [0, {self.chunks})"
            )

    def _create_grad_send_info(
        self,
        args_recv_info: tuple,
    ) -> list[int | None]:
        """
        Create a list of stage indices to send gradients to.
        """
        grad_send_info: list[int | None] = []

        def map_recv_to_send(a):
            # Note: we send gradients back to previous stage as long as in
            # forward it is a received input, regardless of whether it requires
            # grad. It is up to the previous stage to discard this gradient.
            if a.is_root_arg:
                # Root args don't have a source stage to send gradients to
                grad_send_info.append(None)
                return None
            else:
                grad_send_info.append(a.source)
                return a.source

        map_aggregate(args_recv_info, map_recv_to_send)

        logger.debug("%s Grad send info: %s", self.log_prefix, grad_send_info)
        return grad_send_info

    @abstractmethod
    def _prepare_forward_infra(
        self,
        num_microbatches: int,
        args: tuple[Any, ...] | _StageForwardMeta | None,
        kwargs: dict[str, Any] | None = None,
        has_backward: bool = False,
    ) -> _StageForwardMeta | None:
        raise NotImplementedError

    @abstractmethod
    def _prepare_backward_infra(
        self,
        num_microbatches: int,
        loss_fn: Callable[..., torch.Tensor] | None = None,
        target: torch.Tensor | None = None,
        received_grad_meta: _StageBackwardMeta | None = None,
        loss_kwargs: dict[str, Any] | None = None,
    ) -> _StageBackwardMeta | None:
        raise NotImplementedError

    def _setup_backward_recv_info(self, num_microbatches: int):
        # TODO: this is needed for backward_maybe_with_nosync
        self.chunks = num_microbatches

        # IMPORTANT: _create_grad_recv_info reads self._stage_meta.output_grads
        # to attach DTensor metadata to _RecvInfo objects. The clear below MUST
        # happen after all _create_grad_recv_info calls complete.
        for mb_index in range(num_microbatches):
            self.grad_recv_info[mb_index] = self._create_grad_recv_info(
                self.act_send_info
            )

    @abstractmethod
    def _create_grad_recv_info(
        self,
        act_send_info: dict,
    ) -> tuple[_RecvInfo, ...]:
        raise NotImplementedError

    def _resolve_peer_global_rank(self, stage_idx: int) -> int:
        """Map a pipeline stage index to the corresponding global rank for P2P communication."""
        peer_rank = self.stage_index_to_group_rank[stage_idx]
        return dist.get_global_rank(
            self.group or dist.distributed_c10d._get_default_group(),
            peer_rank,
        )

    def _get_recv_ops(
        self,
        recv_infos: tuple[_RecvInfo, ...],
        group: dist.ProcessGroup | None,
    ) -> list[dist.P2POp]:
        """
        Helper function shared by `get_fwd_recv_ops` and `get_bwd_recv_ops`.
        Returns a list of ops that correspond to the recv infos. ``group`` is the
        direction-specific communicator (downstream vs upstream); it equals
        ``self.group`` unless per-direction P2P is enabled.
        """
        ops: list[dist.P2POp] = []
        for info in recv_infos:
            if info.is_root_arg:
                # Root args don't need recv operations
                continue
            # Skip entries with None buffer (None gradients)
            if info.buffer is None:
                if info.tensor_meta is not None:
                    raise AssertionError("expected info.tensor_meta to be None")
                continue
            # At this point, source and buffer are guaranteed non-None
            if info.source is None:
                raise AssertionError("expected info.source to be not None")
            peer_global_rank = self._resolve_peer_global_rank(info.source)
            ops.append(dist.P2POp(dist.irecv, info.buffer, peer_global_rank, group))

        return ops

    """[Note: V-schedule special case]

    V-Schedules have a special case where 2 stages with adjacent stage_id
    are on the same rank.

    Example: 2 ranks, 4 stages forms a simple V::

        rank0:  stage 0                   stage 3
        rank1:          stage 1  stage 2

    Stages 0/1 and 2/3 communicate via send/recv, but stages 1/2 pass
    tensors directly via function call, avoiding communication ops.
    """

    def set_local_fwd_input(self, prev_stage_outputs: Any, mb_index: int) -> None:
        """Pass outputs from a same-rank stage as forward inputs (V-schedule).

        Detaches tensors and sets ``requires_grad`` so they serve as autograd
        leaves. Handles DTensor activations transparently.
        """
        recv_infos: tuple[_RecvInfo, ...] = self.args_recv_info[mb_index]

        # See [Note: pipeline model output type]
        prev_stage_outputs = _normalize_model_output_as_tuple(prev_stage_outputs)

        for info, tensor in zip(recv_infos, prev_stage_outputs, strict=True):
            if not isinstance(tensor, torch.Tensor):
                raise AssertionError(
                    f"expected tensor values as outputs from prev stage, got {type(tensor)}"
                )
            if info.is_root_arg:
                raise AssertionError(
                    "set_local_fwd_input should only be called on non-first stage, which should always have non-root RecvInfo"
                )

            # Pass the activation tensor directly (same rank for local execution).
            # Detach to create a new autograd leaf for the fresh autograd graph.
            info.buffer = to_local_if_dtensor(tensor, detach=True)

    def get_local_bwd_output(self, mb_index):
        """
        Returns the input grad tensors for this stage, which correspond to the stage inputs during forward.
        """
        if not self.has_backward:
            raise AssertionError(
                "can't steal_bwd_input if this stage doesn't have backward"
            )
        if self.is_first:
            raise AssertionError("can't get bwd output if this stage is first")

        self._check_chunk_id(mb_index)
        return self.bwd_cache.pop(mb_index)

    def set_local_bwd_input(
        self, next_stage_bwd_outputs: tuple[torch.Tensor | None, ...], mb_index: int
    ) -> None:
        """
        Moves 'grad input' tensors from the next stage to 'grad_output' on this stage, avoiding a copy or send/recv.
        Does not detach or set '_requires_grad'.
        Handles DTensor gradients for V-schedule local passing.
        """
        if not isinstance(next_stage_bwd_outputs, tuple):
            raise AssertionError(f"Expected tuple, got {type(next_stage_bwd_outputs)}")

        if not self.has_backward:
            raise AssertionError(
                "can't set bwd input if this stage doesn't have backward"
            )
        if self.is_last:
            raise AssertionError("can't set bwd input if this stage is last")
        recv_infos = self.grad_recv_info[mb_index]
        for info, tensor in zip(recv_infos, next_stage_bwd_outputs, strict=True):
            if tensor is None:
                if info.buffer is not None:
                    info.buffer.zero_()
                continue
            if not isinstance(tensor, torch.Tensor):
                raise AssertionError(
                    f"expected tensor values as outputs from prev stage, got {type(tensor)}"
                )
            if info.is_root_arg:
                raise AssertionError(
                    "set_local_bwd_input should only be called with non-root RecvInfo"
                )

            # Extract local tensor for the buffer (handles DTensor or plain tensor)
            info.buffer = to_local_if_dtensor(tensor)

    def get_fwd_recv_ops(self, fwd_chunk_id: int) -> list[dist.P2POp]:
        """
        Returns a list of ops that are needed to receive the input arguments
        for this stage.
        """
        recv_infos: tuple[_RecvInfo, ...] = self.args_recv_info[fwd_chunk_id]

        return self._get_recv_ops(recv_infos, self._downstream_group)

    def get_bwd_recv_ops(self, bwd_chunk_id: int) -> list[dist.P2POp]:
        """
        Returns a list of ops that are needed to receive the gradients
        for this stage.
        """
        if not self.has_backward or self.is_last:
            return []

        recv_infos = self.grad_recv_info[bwd_chunk_id]
        return self._get_recv_ops(recv_infos, self._upstream_group)

    def get_fwd_send_ops(self, fwd_chunk_id: int) -> list[dist.P2POp]:
        """
        Get the activation send ops for current stage's forward.
        Handles DTensor outputs by extracting local tensors.
        """
        output_tuple, _ = self.fwd_cache[fwd_chunk_id]

        ops: list[dist.P2POp] = []

        for idx, out in enumerate(output_tuple):
            dst_stages = self.act_send_info[idx]
            for dst in dst_stages:
                if dst is None:
                    continue
                # Extract local tensor if DTensor
                send_tensor = to_local_if_dtensor(out, detach=True)
                logger.debug(
                    "%s Sending tensor to Stage %s: %s",
                    self.log_prefix,
                    dst,
                    send_tensor.size(),
                )
                peer_global_rank = self._resolve_peer_global_rank(dst)
                ops.append(
                    dist.P2POp(
                        dist.isend,
                        send_tensor,
                        peer_global_rank,
                        self._downstream_group,
                    )
                )

        return ops

    def _get_grad_send_meta(self, input_idx: int) -> TensorMeta | None:
        """Return gradient metadata for ``input_idx`` if a recv was allocated."""
        input_grads = self._stage_meta.input_grads
        if input_grads is not None and input_idx < len(input_grads):
            return input_grads[input_idx]

        inputs = self._stage_meta.inputs
        if inputs is not None and input_idx < len(inputs):
            meta = inputs[input_idx]
            if meta is not None:
                return _derive_grad_metas((meta,))[0]

        raise PipeliningMetadataError(
            f"Stage {self.stage_index}: backward produced gradient for input "
            f"{input_idx}, but no metadata is available for the gradient send."
        )

    def get_bwd_send_ops(self, bwd_chunk_id: int) -> list[dist.P2POp]:
        """
        Get the gradient send ops for current stage's backward.
        Handles DTensor gradients by extracting local tensors.
        """
        if not self.has_backward or self.is_first:
            return []

        self._check_chunk_id(bwd_chunk_id)
        # Create bwd send infra lazily
        if self.grad_send_info is None:
            # Send info for input grads during backward:
            # List of destinations corresponding to input grads
            # Can be None if an input has no grad
            # `grad_send_info` is a mirror of `args_recv_info`
            self.grad_send_info = self._create_grad_send_info(self.args_recv_info[0])

        ops: list[dist.P2POp] = []
        grads_input = self.bwd_cache.pop(bwd_chunk_id)

        for idx, (grad, grad_recv_stage) in enumerate(
            zip(grads_input, self.grad_send_info, strict=True)
        ):
            if grad_recv_stage is None:
                if grad is not None:
                    raise PipeliningMetadataError(
                        f"[{self.stage_index}] for chunk {bwd_chunk_id} has gradient {grad} "
                        f"but no stage to send it to"
                    )
                continue

            grad_meta = self._get_grad_send_meta(idx)
            if grad_meta is None:
                if grad is not None:
                    raise PipeliningMetadataError(
                        f"Stage {self.stage_index} chunk {bwd_chunk_id} input {idx}: "
                        f"backward produced a {type(grad).__name__} gradient, but "
                        "backward metadata for this grad slot is None, so the "
                        "previous stage did not allocate a recv buffer. DTensor "
                        "grad metadata cannot be derived from forward DTensor "
                        "metadata because gradient placements may differ from "
                        "activation placements. Provide static input_grads "
                        "metadata or run metadata inference with a microbatch "
                        "that produces this grad."
                    )
                continue

            if isinstance(grad, torch.Tensor):
                # Extract local tensor if DTensor
                send_tensor = to_local_if_dtensor(grad)
                logger.debug(
                    "%s Sending gradient to Stage %s: %s",
                    self.log_prefix,
                    grad_recv_stage,
                    send_tensor.size(),
                )
                peer_global_rank = self._resolve_peer_global_rank(grad_recv_stage)
                ops.append(
                    dist.P2POp(
                        dist.isend, send_tensor, peer_global_rank, self._upstream_group
                    )
                )
            elif grad is None:
                zero_grad = _make_tensor_from_meta(grad_meta, self.device).zero_()
                send_tensor = to_local_if_dtensor(zero_grad)
                logger.debug(
                    "%s Sending zero gradient to Stage %s: %s",
                    self.log_prefix,
                    grad_recv_stage,
                    send_tensor.size(),
                )
                peer_global_rank = self._resolve_peer_global_rank(grad_recv_stage)
                ops.append(
                    dist.P2POp(dist.isend, send_tensor, peer_global_rank, self.group)
                )
        return ops

    def clear_runtime_states(self) -> None:
        """
        Clear runtime states of the stage.
        """
        # map microbatch ID to list of forward tensor args
        self.fwd_cache.clear()
        # Caching chunk outputs for final output merge or reduction
        self.output_chunks.clear()

        # Clear grad of input buffers in between schedule steps. This is because
        # `torch.autograd.backward()` will accumulate gradients into leaf
        # tensors by default. For gradients to pass back to previous stages, we
        # don't want such accumulation.
        for recv_tuple in self.args_recv_info.values():  # iterate over all chunks
            for a in recv_tuple:  # iterate over all input args
                if not a.is_root_arg and a.buffer is not None:
                    # Set to None is the newer and recommended way to clear grads, compared to `zero_()`.
                    # See https://github.com/pytorch/pytorch/pull/92731
                    a.buffer.grad = None

    def _map_tensor_from_recv_info(
        self,
        recv_infos: tuple[_RecvInfo, ...],
    ):
        """
        Map tensors from recv infos to a list.
        """

        def get_recv_tensor(info):
            if info.is_root_arg:
                raise PipeliningMetadataError("Cannot get recv tensor from root arg")
            return info.buffer

        return map_aggregate(cast(Argument, recv_infos), get_recv_tensor)

    def _retrieve_recv_activations(
        self,
        fwd_chunk_id: int,
    ):
        """
        Retrieve the activations received for the current stage during forward.
        Reconstructs DTensors if the inputs were DTensors.
        Also validates DTensor metadata against expected values.
        """
        recv_infos = self.args_recv_info[fwd_chunk_id]

        activations = []
        for i, info in enumerate(recv_infos):
            if not info.is_root_arg:
                # Non-root args have valid buffer and tensor_meta
                if info.buffer is None or info.tensor_meta is None:
                    raise PipeliningMetadataError(
                        f"Non-root arg '{info.input_name}' has None buffer or tensor_meta"
                    )
                # Effective requires_grad: metadata captures what the model
                # produced, but the runtime context (has_backward, grad mode)
                # determines whether we actually need gradients.
                effective_requires_grad = (
                    info.tensor_meta.requires_grad
                    and self.has_backward
                    and torch.is_grad_enabled()
                )
                if isinstance(info.tensor_meta, _DTensorMeta):
                    # Buffer must not require grad so from_local stays out
                    # of the autograd graph (no grad_placements needed).
                    if info.buffer.requires_grad:
                        raise PipeliningMetadataError(
                            f"Stage {self.stage_index}: recv buffer "
                            f"'{info.input_name}' unexpectedly requires grad "
                            f"before DTensor reconstruction"
                        )
                    mesh = self._mesh_cache.get_mesh(info.tensor_meta.mesh_cache_key)
                    activation = DTensor.from_local(
                        info.buffer,
                        device_mesh=mesh,
                        placements=info.tensor_meta.placements,
                        shape=info.tensor_meta.global_shape,
                        stride=info.tensor_meta.global_stride,
                        run_check=False,
                    ).requires_grad_(effective_requires_grad)
                else:
                    activation = info.buffer.requires_grad_(effective_requires_grad)
                # Activation must be a leaf so backward terminates here.
                if effective_requires_grad and not activation.is_leaf:
                    warnings.warn(
                        f"Stage {self.stage_index}: activation "
                        f"'{info.input_name}' is not a leaf "
                        f"(grad_fn={activation.grad_fn}); using "
                        f"retain_grad() as fallback",
                        stacklevel=2,
                    )
                    activation.retain_grad()
                activations.append(activation)
            else:
                raise PipeliningMetadataError(
                    f"_retrieve_recv_activations expected non-root _RecvInfo but got root arg at index {i}"
                )

        return tuple(activations)

    def _retrieve_recv_grads(
        self,
        bwd_chunk_id: int,
    ):
        """
        Retrieve the gradients received for the current stage during backward.

        Handles None gradients gracefully (for inputs that don't require grad).
        """
        recv_infos = self.grad_recv_info[bwd_chunk_id]

        grads: list[torch.Tensor | None] = []
        for i, info in enumerate(recv_infos):
            if not isinstance(info, _RecvInfo):
                raise PipeliningMetadataError(
                    f"Expected _RecvInfo but got {type(info)}"
                )
            if not info.is_root_arg:
                # Gradients can be None for non-differentiable outputs
                if info.buffer is None:
                    if info.tensor_meta is not None:
                        raise PipeliningMetadataError(
                            f"Grad recv '{info.input_name}': buffer is None but tensor_meta is not None"
                        )
                    grads.append(None)
                    continue
                if info.tensor_meta is None:
                    raise PipeliningMetadataError(
                        f"Grad recv '{info.input_name}': buffer is not None but tensor_meta is None"
                    )
                if isinstance(info.tensor_meta, _DTensorMeta):
                    # Reconstruct DTensor gradient from local tensor + metadata
                    mesh = self._mesh_cache.get_mesh(info.tensor_meta.mesh_cache_key)
                    grad = DTensor.from_local(
                        info.buffer,
                        device_mesh=mesh,
                        placements=info.tensor_meta.placements,
                        shape=info.tensor_meta.global_shape,
                        stride=info.tensor_meta.global_stride,
                        run_check=False,
                    )
                else:
                    grad = info.buffer
                grads.append(grad)
            else:
                raise PipeliningMetadataError(
                    f"grad_recv_info should not contain root args, but found one at index {i}"
                )

        return tuple(grads)

    def forward_maybe_with_nosync(self, *args, **kwargs):
        # If submod is wrapped with DDP, we use the `no_sync` context manager to
        # avoid gradient all-reduce per microbatch
        if isinstance(self.submod, DistributedDataParallel):
            with self.submod.no_sync():  # type: ignore[operator]
                out_val = self.submod(*args, **kwargs)
        else:
            out_val = self.submod(*args, **kwargs)
        return out_val

    def scale_grads(self, grad_scale_factor: int) -> None:
        """Scale gradients model gradients by `grad_scale_factor`, which should be specified in coordination with the
        loss function used with pipelining.  For loss functions which perform 'mean' loss reduction, `grad_scale_factor`
        should be set to num_microbatches.  For loss functions that use `sum` reduction, `grad_scale_factor` should
        be set to 1.

        Should only be called once per pipeline schedule step, after all backwards passes have completed.
        """

        # PP scales only for its own contribution (microbatches), but relies on DP to scale further
        # for DP degree.
        if grad_scale_factor != 1:
            for p in self.submod.parameters():
                if p.grad is not None:
                    p.grad.div_(grad_scale_factor)

    def backward_maybe_with_nosync(
        self,
        backward_type,
        bwd_kwargs: dict,
        last_backward: bool = False,
    ) -> tuple[tuple[torch.Tensor | None, ...], list[dict[str, Any]] | None]:
        """
        Whether using PP with FSDP, DDP, or replicate there are some runtime differences between the last backward step and the
        other steps.  Namely, we need to accumulate gradients on previous steps and reduce them on the last step, but
        there are additional state-variables and performance considerations depending on the data parallelism used.
        This helper should adapt any pipeline parallel schedule to work with common/supported data parallel libraries.
        """

        def perform_backward(
            backward_type,
        ) -> Callable[
            [],
            tuple[tuple[torch.Tensor | None, ...], list[dict[str, Any]] | None],
        ]:
            if backward_type == "full":
                return lambda: (
                    stage_backward(
                        bwd_kwargs["stage_output"],
                        bwd_kwargs["output_grads"],
                        bwd_kwargs["input_values"],
                    ),
                    None,
                )
            elif backward_type == "input":
                return lambda: stage_backward_input(
                    bwd_kwargs["stage_output"],
                    bwd_kwargs["output_grads"],
                    bwd_kwargs["input_values"],
                    self.submod.parameters(),
                )
            elif backward_type == "weight":
                return lambda: (
                    stage_backward_weight(
                        self.submod.parameters(), bwd_kwargs["param_groups"]
                    ),
                    None,
                )
            else:
                raise RuntimeError(f"Unknown backward type: {backward_type}")

        # If submod is wrapped by DDP
        if isinstance(self.submod, DistributedDataParallel):
            if last_backward:
                # Last chunk, prepare for gradient reduction
                # HACK: reaching into DDP implementation details here. Is there a better way?
                self.submod.reducer.prepare_for_backward(  # type: ignore[union-attr, operator]
                    list(
                        torch.nn.parallel.distributed._find_tensors(  # type: ignore[attr-defined]
                            bwd_kwargs["stage_output"]
                        )
                    )
                )
                result = perform_backward(backward_type)()
            else:
                with self.submod.no_sync():  # type: ignore[operator]
                    result = perform_backward(backward_type)()

        # If submod is a FSDP or replicate module
        elif isinstance(self.submod, FSDPModule):
            self.submod.set_is_last_backward(False)
            self.submod.set_reshard_after_backward(False)
            self.submod.set_requires_gradient_sync(False)
            result = perform_backward(backward_type)()

        else:
            # Non-DP submodule, regular backward
            result = perform_backward(backward_type)()

        grads, param_groups = result
        return grads, param_groups

    def forward_one_chunk(
        self,
        fwd_chunk_id: int,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
        save_forward_output: bool = True,
    ):
        """
        Perform forward pass on the stage with one microbatch.
        `args` and `kwargs` are the inputs from *external* to this stage.
        As of Sept 2024:
        - `args` applies to the first stage only, other stages receives args
          through activation transmission.
        - `kwargs` can be passed to all stages via respective `step` calls.
        """

        if self.is_first:
            # First stage doesn't need to receive anything
            composite_args = args
        else:
            # Receive activations for this chunk
            # Activations only come in args form
            composite_args = self._retrieve_recv_activations(fwd_chunk_id)

        composite_kwargs = kwargs or {}

        if self._runtime_validate:
            self._validate_stage_tensors(
                f"Stage {self.stage_index} forward inputs",
                self._stage_meta.inputs,
                composite_args,
            )

        # Compute forward
        try:
            output = self.forward_maybe_with_nosync(*composite_args, **composite_kwargs)

        except Exception as e:
            exc_msg = f"""
            {self.log_prefix} failed to run forward:
            args: {map_debug_info(composite_args)}
            kwargs: {map_debug_info(composite_kwargs)}
            """
            raise RuntimeError(exc_msg) from e

        # See [Note: pipeline model output type]
        output_tuple = _normalize_model_output_as_tuple(output)

        # Prepare for final output merge or reduction
        # Output chunks is only used for the last stage since we only merge the output of the last stage
        if self.is_last and save_forward_output:
            self.output_chunks.append(output)
        # Save activations and inputs for backward
        flat_args = flatten_args(composite_args)
        flat_kwargs = flatten_args(composite_kwargs)
        flatten_input_tensors = flat_args + flat_kwargs
        self.fwd_cache[fwd_chunk_id] = (
            output_tuple,  # stage_output
            flatten_input_tensors,  # input_values
        )

        logger.debug(
            "%s Forwarded chunk %s, outputs: %s",
            self.log_prefix,
            fwd_chunk_id,
            map_debug_info(output),
        )
        # Validate outputs before P2P send; skipped for last stage (outputs
        # go to loss/user, not via send/recv).
        if self._runtime_validate and not self.is_last:
            self._validate_stage_tensors(
                f"Stage {self.stage_index} forward outputs",
                self._stage_meta.outputs,
                output_tuple,
            )

        # We return the original user-provided output, not normalized to tuple.
        # See [Note: pipeline model output type]
        return output

    def backward_one_chunk(
        self,
        bwd_chunk_id: int,
        loss=None,
        full_backward: bool = True,
        last_backward=False,
    ):
        """
        Perform backward pass on the module.
        This should only be called once per microbatch.

        If full_backward is True (the default), the full backward pass including weight and input gradients will be run,
        and it is an error to call `backward_weight_one_chunk` for this bwd_chunk_id.

        If full_backward is False, it is optional that `dw_runner` was provided to the PipelineStage at __init__ time,
        and a subsequent call to `backward_weight_one_chunk` is required to invoke dw_runner and complete the backward.

        last_backward is controlled by the schedule and signals synchronization of gradients across DP groups
        after the last backward.
        """
        # skip backward computation if backward is not enabled
        if not self.has_backward:
            return

        self._check_chunk_id(bwd_chunk_id)

        (
            stage_output,
            input_values,
        ) = self.fwd_cache.pop(bwd_chunk_id)

        # Compute backward
        if self.is_last:
            # Last stage computes gradients from loss and has no gradients from
            # next stage
            bwd_kwargs = {
                "stage_output": loss,
                "output_grads": None,
                "input_values": input_values,
            }
        else:
            # Otherwise, receive gradients from next stage
            grads_output = self._retrieve_recv_grads(bwd_chunk_id)
            if self._runtime_validate:
                # Validate backward input (output gradients) for DTensor metadata
                self._validate_stage_tensors(
                    f"Stage {self.stage_index} backward input (output_grads)",
                    self._stage_meta.output_grads,
                    grads_output,
                )
            # If an input to the pipeline requires gradient,
            # `torch.autograd.backward` will accumulate the gradient into the
            # `.grad` field of such input
            bwd_kwargs = {
                "stage_output": stage_output,
                "output_grads": grads_output,
                "input_values": input_values,
            }

        grads_input: tuple[torch.Tensor | None, ...] = ()

        # Custom backward function
        if self.dw_builder:
            # TODO: We may want to change our semantics so we are allowed to ignore
            # the 'dw_builder' and call full_backward directly when it is a full_backward op.
            grads_input, _ = self.backward_maybe_with_nosync(
                "full",
                bwd_kwargs,
                last_backward=last_backward,
            )
            if full_backward:
                self.dw_builder()()
            else:
                self.dw_runner[bwd_chunk_id] = self.dw_builder()
        else:
            if full_backward:
                grads_input, _ = self.backward_maybe_with_nosync(
                    "full", bwd_kwargs, last_backward=last_backward
                )
            else:
                param_groups: list[dict[str, Any]] | None = None
                # Skip the backward for the first stage since we will perform the weight update with
                # autograd.backward in backward_weight_one_chunk
                if not self.is_first:
                    if isinstance(bwd_kwargs["stage_output"], torch.Tensor):
                        bwd_kwargs["stage_output"] = (bwd_kwargs["stage_output"],)

                    # perform the partial backwards for the inputs with a custom backward function
                    # when the "stage_ouput" is a loss, then it is a tensor, otherwise it is a tuple of tensors
                    grads_input, param_groups = self.backward_maybe_with_nosync(
                        "input", bwd_kwargs, last_backward=last_backward
                    )

                # TODO: we don't need to save this, add to dw_runner?
                self.backward_state[bwd_chunk_id] = (
                    bwd_kwargs["input_values"],
                    param_groups,
                    bwd_kwargs["stage_output"],
                    bwd_kwargs["output_grads"],
                )
                # Save a placeholder for the dw_runner
                self.dw_runner[bwd_chunk_id] = lambda: None
        # Note: grads_input may contain gradients for both args and kwargs (from fwd_cache),
        # Kwargs are local to each stage and don't need gradient transmission.
        # Validate backward output (input gradients) for DTensor metadata
        if self._stage_meta.inputs is None:
            raise AssertionError("expected self._stage_meta.inputs to be not None")
        num_fwd_args = len(self._stage_meta.inputs)
        if self._runtime_validate and not self.is_first:
            self._validate_stage_tensors(
                f"Stage {self.stage_index} backward output (input_grads)",
                self._stage_meta.input_grads,
                grads_input[:num_fwd_args],
            )
        self.bwd_cache[bwd_chunk_id] = grads_input[:num_fwd_args]

        if self.is_last and not self.is_first:
            # Autograd dependencies:
            #    rest_of_autograd_graph -> stage_output -> loss
            # stage_output is no longer used in the last stage for backward and only needed
            # to return to the user in merge_output_chunks, therefore
            # this should be detached to release autograd graph context and free memory earlier
            for t in stage_output:
                if not t._is_view():  # views are not detachable in-place
                    t.detach_()

        logger.debug("%s Backwarded chunk %s", self.log_prefix, bwd_chunk_id)

    def backward_weight_one_chunk(self, bwd_chunk_id: int, last_backward=False):
        # skip backward computation if backward is not enabled
        if not self.has_backward:
            return

        if bwd_chunk_id not in self.dw_runner:
            raise AssertionError(
                f"{self.log_prefix} Attempted to run backward_weight_one_chunk for chunk {bwd_chunk_id}"
                " without first calling `backward_one_chunk(full_backward=False)`"
            )

        if self.dw_builder is not None:
            self.dw_runner.pop(bwd_chunk_id)()
        else:
            (
                input_values,
                param_groups,
                stage_output,
                output_grads,
            ) = self.backward_state.pop(bwd_chunk_id)

            if self.stage_index != 0:
                bwd_kwargs = {
                    "stage_output": stage_output,
                    "param_groups": param_groups,
                }
                self.backward_maybe_with_nosync(
                    "weight", bwd_kwargs, last_backward=last_backward
                )
            else:
                # TODO: figure out a better way to do this:
                # if inputs does not require gradient,
                # then the parameter group will not be fully captured during stage_backward_input
                # in this case, we need call grad directly on the parameters
                # To solve: make input fn do the intersect compute and then finish it off during W
                bwd_kwargs = {
                    "stage_output": stage_output,
                    "output_grads": output_grads,
                    "input_values": input_values,
                }
                self.backward_maybe_with_nosync(
                    "full", bwd_kwargs, last_backward=last_backward
                )

    def _get_init_p2p_neighbors_ops(self) -> list[dist.P2POp]:
        """
        Get the operations to initialize the p2p communicators between previous and next stages.
        This is done so by creating a dummy tensor and sending it to the next stage and receiving
        from the previous stage.
        """
        ops: list[dist.P2POp] = []
        next_stage_peer_rank = self.stage_index_to_group_rank.get(self.stage_index + 1)
        prev_stage_peer_rank = self.stage_index_to_group_rank.get(self.stage_index - 1)

        # Separate recv buffers per direction: with per-direction P2P the
        # downstream and upstream recvs run concurrently on different
        # communicators/streams, so they must not share a buffer (concurrent
        # writes = data race). The send buffer is only read, so it can be shared.
        downstream_recv_tensor = torch.zeros(1, device=self.device, dtype=torch.float32)
        upstream_recv_tensor = torch.zeros(1, device=self.device, dtype=torch.float32)
        send_tensor = torch.tensor(
            self.stage_index, device=self.device, dtype=torch.float32
        )
        # downstream traffic (r -> r+1: forward activations) -> downstream comm
        if not self.is_first:
            ops.append(
                dist.P2POp(
                    dist.irecv,
                    downstream_recv_tensor,
                    group_peer=prev_stage_peer_rank,
                    group=self._downstream_group,
                )
            )
        if not self.is_last:
            ops.append(
                dist.P2POp(
                    dist.isend,
                    send_tensor,
                    group_peer=next_stage_peer_rank,
                    group=self._downstream_group,
                )
            )

        # upstream traffic (r -> r-1: backward gradients) -> upstream comm
        if not self.is_first:
            ops.append(
                dist.P2POp(
                    dist.isend,
                    send_tensor,
                    group_peer=prev_stage_peer_rank,
                    group=self._upstream_group,
                )
            )
        if not self.is_last:
            ops.append(
                dist.P2POp(
                    dist.irecv,
                    upstream_recv_tensor,
                    group_peer=next_stage_peer_rank,
                    group=self._upstream_group,
                )
            )

        return ops

    def perform_reduce_grad(self, grad_scale_factor: int):
        """
        Called as a part of schedule IR.
        REDUCE_GRAD action is scheduled after all microbatches W, B actions.

        Currently contains "post_backward" functionality for FSDP.
        We can try to extract post_backward in a separate IR action in future.
        """
        # Manually call post backward for FSDP
        if isinstance(self.submod, FSDPModule):
            fsdp_module = self.submod
            fsdp_module.set_is_last_backward(True)
            fsdp_module.set_reshard_after_backward(True)
            fsdp_module.set_requires_gradient_sync(True)

            if isinstance(fsdp_module, ReplicateModule):
                distributed_state = replicate.state(fsdp_module)  # type: ignore[arg-type]
            else:
                distributed_state = fully_shard.state(fsdp_module)  # type: ignore[attr-defined]

            for state in distributed_state._state_ctx.all_states:
                for fsdp_param_group in state._fsdp_param_groups:
                    fsdp_param_group.post_backward()

            # it would be much better if pipelining backward invoked .backward so autograd hooks
            # worked and modules like DDP/FSDP behaved as expected.  Working around this for the time being,
            # we need to call this too to ensure FSDP syncs its grad reduction ops back to the default stream.
            distributed_state._root_post_backward_final_callback()
        # Call gradient scaling at the end of the backward pass
        # NOTE: this must happen after FSDP post_backward is FSDP is enabled
        if grad_scale_factor != 1:
            self.scale_grads(grad_scale_factor)


class _PipelineStage(_PipelineStageBase):
    def __init__(
        self,
        stage_module: torch.nn.Module,
        stage_index: int,
        pipe_info: PipeInfo,
        device: torch.device,
        group: dist.ProcessGroup | None = None,
    ):
        """
        Create a pipeline stage given a stage_module to be wrapped by this stage
        and a `pipe_info` describing the stage relationship of the pipeline.

        Args:
            stage_module (torch.nn.Module): the module to be wrapped by this stage
            stage_index (int): the index of this stage in the pipeline
            pipe_info (PipeInfo): information about the pipeline, can be retrieved by `pipe.info()`
            device (torch.device): the device to be used by this stage
            group (Optional[dist.ProcessGroup]): the process group to be used by this stage
        """
        _PipelineStageBase.__init__(
            self,
            stage_module,
            stage_index,
            pipe_info.num_stages,
            device,
            group,
        )
        self.pipe_info = pipe_info

        # Find stage nodes in graph
        submod_nodes = [
            node for node in pipe_info.graph.nodes if node.op == "call_module"
        ]
        if len(submod_nodes) != self.num_stages:
            raise PipeliningMetadataError(
                f"Number of submodules in pipe graph {len(submod_nodes)} does not match number of stages {self.num_stages}"
            )

        # Find my stage node in graph
        self.node = submod_nodes[self.stage_index]
        self.name = self.node.name
        logger.info(
            "[%s] Creating PipelineStage %s for %s",
            self.group_rank,
            stage_index,
            self.name,
        )

        # Create mapping from stage name to stage index
        self.submod_to_stage_index: dict[str, int] = {}
        for i, node in enumerate(submod_nodes):
            self.submod_to_stage_index.setdefault(node.name, i)

        # Cast submodule to device
        self._move_submod_to_device()

    def _move_submod_to_device(self):
        # Move submodule to indicated device if possible
        # Note: we cannot move meta module to real devices because meta tensors
        # do not support to() method. One needs to do an in-place tensor swap in
        # that case.
        has_meta_param = any(
            is_fake_tensor(p) or p.is_meta for p in self.submod.parameters()
        )
        if has_meta_param:
            logger.debug("%s Found meta parameters!", self.log_prefix)
        else:
            self.submod.to(self.device)

    def _prepare_forward_infra(
        self,
        num_microbatches: int,
        args: tuple[Any, ...] | _StageForwardMeta | None,
        kwargs: dict[str, Any] | None = None,
        has_backward: bool = False,
    ) -> _StageForwardMeta | None:
        """
        Prepare forward infrastructure for traced pipeline.

        Metadata is created directly from graph placeholders with correct
        ``requires_grad`` — received activations get ``requires_grad=True``
        when ``has_backward`` is set, fixing the fact that ``torch.export``
        traces under ``no_grad()``.

        ``_stage_meta.inputs`` is derived from recv infos and aligned with
        ``forward_one_chunk``'s ``composite_args``: positional root inputs
        on the first stage, received activations only on subsequent stages.
        """
        # Step 1: Create recv info for each microbatch.
        # _create_act_recv_info is self-contained: it creates _TensorMeta
        # directly from graph placeholder values with correct requires_grad.
        for chunk in range(num_microbatches):
            self.args_recv_info[chunk] = self._create_act_recv_info()

        # Step 2: Derive _stage_meta.inputs from recv infos.
        # forward_one_chunk builds composite_args as:
        #   - First stage:     args (positional root inputs, excludes kwargs)
        #   - Non-first stages: received activations only (no root kwargs)
        # _stage_meta.inputs must match composite_args for validation.
        recv_infos = self.args_recv_info[0]
        if self.is_first:
            # All placeholders are root args.  Only the first len(args)
            # correspond to positional inputs (composite_args); the rest
            # are kwargs passed separately via composite_kwargs.
            # First stage always receives real tensor args, never _StageForwardMeta.
            if not isinstance(args, tuple):
                raise AssertionError("First stage requires real tensor args")
            n_positional = len(args)
            self._stage_meta.inputs = tuple(
                info.tensor_meta  # type: ignore[misc]
                for info in recv_infos[:n_positional]
            )
        else:
            self._stage_meta.inputs = tuple(
                info.tensor_meta  # type: ignore[misc]
                for info in recv_infos
                if not info.is_root_arg
            )

        # Step 3: Create send info and output metadata.
        self.act_send_info = self._create_act_send_info()

        return None

    def _prepare_backward_infra(
        self,
        num_microbatches: int,
        loss_fn: Callable[..., torch.Tensor] | None = None,
        target: torch.Tensor | None = None,
        received_grad_meta: _StageBackwardMeta | None = None,
        loss_kwargs: dict[str, Any] | None = None,
    ) -> _StageBackwardMeta | None:
        """
        Prepare backward infrastructure for traced pipeline.
        Derives input_grads metadata from inputs (plain tensors only).

        Note: DTensors are NOT supported in the traced frontend.
        """
        # Derive input_grads from inputs (for plain tensors, grad shape == input shape)
        if self._stage_meta.inputs is None:
            raise PipeliningMetadataError(
                f"Stage {self.stage_index}: inputs metadata required for backward inference."
            )

        self._stage_meta.input_grads = _derive_grad_metas(self._stage_meta.inputs)

        # Setup backward recv info (calls _create_grad_recv_info which sets output_grads).
        # Note: grad_send_info is created lazily in get_bwd_send_ops() since
        # it mirrors args_recv_info (already populated during forward).
        self._setup_backward_recv_info(num_microbatches)

        return None

    def get_stage_index_of_submod(
        self,
        submod_name: str,
    ):
        """
        Given a submodule name, return the stage index of the submodule.
        """
        if submod_name not in self.submod_to_stage_index:
            raise PipeliningMetadataError(f"Stage id of {submod_name} not found")

        return self.submod_to_stage_index[submod_name]

    def _create_act_recv_info(
        self,
    ):
        """
        Create a tuple of `_RecvInfo` for inputs to the stage.

        Self-contained: creates ``_TensorMeta`` directly from graph
        placeholder values with correct ``requires_grad``.
        ``torch.export`` traces under ``no_grad()`` so traced metadata
        always has ``requires_grad=False``; for received activations we
        set ``requires_grad=True`` when ``has_backward`` is set.

        Note: DTensors are NOT supported in the traced frontend.
        """

        def create_recv_tensor(placeholder, arg_node):
            example_value = placeholder.meta["val"]

            # Reject DTensors in traced frontend
            if isinstance(example_value, DTensor):
                raise PipeliningMetadataError(
                    f"{self.log_prefix} DTensor detected in traced pipeline input "
                    f"'{placeholder.name}'. DTensor metadata propagation is NOT "
                    f"supported for the traced frontend (_PipelineStage). "
                    f"Use the manual PipelineStage frontend for full DTensor support."
                )

            if arg_node.op == "placeholder":
                # Root-level placeholder: an input argument to the entire
                # model.  Keep original metadata from the trace.
                return _RecvInfo(
                    input_name=f"root_input_{placeholder.name}",
                    source=None,
                    buffer=None,
                    tensor_meta=_TensorMeta.from_tensor(example_value),
                    is_root_arg=True,
                )

            # Received activation from a previous stage.
            while arg_node.target is operator.getitem:
                arg_node = arg_node.args[0]

            if arg_node.op != "call_module":
                raise PipeliningMetadataError(
                    f"Expecting call_module, got {arg_node.op}"
                )
            src_stage = self.get_stage_index_of_submod(arg_node.name)

            # Create metadata directly with correct requires_grad.
            # Guard for non-float tensors.
            tensor_meta = _TensorMeta(
                shape=example_value.shape,
                stride=example_value.stride(),
                dtype=example_value.dtype,
                requires_grad=self.has_backward and example_value.is_floating_point(),
            )

            logger.debug(
                "%s Creating recv buffer for input '%s' : %s, %s",
                self.log_prefix,
                placeholder.name,
                tensor_meta.shape,
                tensor_meta.dtype,
            )
            buffer = _make_tensor_from_meta(tensor_meta, self.device)
            if self.has_backward and example_value.is_floating_point():
                buffer.requires_grad_(True)

            return _RecvInfo(
                arg_node.name,
                src_stage,
                buffer,
                tensor_meta,
            )

        args_recv_info: list[_RecvInfo] = []
        placeholders = filter(  # type: ignore[var-annotated]
            lambda node: node.op == "placeholder",  # type: ignore[arg-type]
            self.submod.graph.nodes,  # type: ignore[arg-type,union-attr]
        )
        # `placeholders` are nodes internal to submod.
        # `self.node.args` are dependency nodes in the outer graph.
        # The two are 1:1.
        for placeholder, arg_node in zip(placeholders, self.node.args, strict=True):
            args_recv_info.append(create_recv_tensor(placeholder, arg_node))

        logger.debug(
            "%s Activation recv / args info: %s", self.log_prefix, args_recv_info
        )
        return tuple(args_recv_info)

    def find_dst_rank(
        self,
        user: fx.Node,
    ) -> int | None:
        """
        Find the destination rank of a `user` node.
        If the `user` is not a submod, `None` may be returned.
        """
        if user.op == "call_module":
            # User is a stage (`call_module`)
            return self.get_stage_index_of_submod(user.name)
        else:
            # - If user.op == "output":
            #   No need to send back to rank 0
            # - If user.target is stage_backward:
            #   No need to send assuming submod output is stored locally or
            #   should be re-calculated in case of activation checkpointing
            return None

    def _create_act_send_info(self):
        """
        Create a dict of send info for activations and output metadata.

        Output metadata is created directly with correct ``requires_grad``
        (``torch.export`` traces under ``no_grad()``, so traced values
        always have ``requires_grad=False``; at runtime, stage outputs
        carry ``requires_grad=True`` when training).
        """
        # Output index: List of receiver ranks
        act_send_info: dict[int, list] = {}
        out_idx = 0

        for user in self.node.users:
            if user.target is operator.getitem:
                gi_dsts = act_send_info.setdefault(out_idx, [])
                for gi_user in user.users:
                    dst_rank = self.find_dst_rank(gi_user)
                    if dst_rank is not None:
                        gi_dsts.append(dst_rank)
                out_idx += 1
            else:
                dsts = act_send_info.setdefault(out_idx, [])
                dst_rank = self.find_dst_rank(user)
                if dst_rank is not None:
                    dsts.append(dst_rank)

        output_node = self._get_output_node()
        output_vals: tuple[torch.Tensor] = tuple(
            v.meta["val"] for v in flatten_args(output_node.args)
        )
        # Reject DTensors and create output metadata directly with
        # correct requires_grad.
        output_metas: list[_TensorMeta] = []
        for i, val in enumerate(output_vals):
            if isinstance(val, DTensor):
                raise PipeliningMetadataError(
                    f"{self.log_prefix} DTensor detected in traced pipeline output index {i}. "
                    f"DTensor metadata propagation is NOT supported for the traced frontend "
                    f"(_PipelineStage). Use the manual PipelineStage frontend for full DTensor support."
                )
            output_metas.append(
                _TensorMeta(
                    shape=val.shape,
                    stride=val.stride(),
                    dtype=val.dtype,
                    requires_grad=self.has_backward and val.is_floating_point(),
                )
            )
        self._stage_meta.outputs = tuple(output_metas)

        logger.debug("%s Send info: %s", self.log_prefix, act_send_info)
        return act_send_info

    def _get_output_node(self):
        output_nodes = [node for node in self.submod.graph.nodes if node.op == "output"]  # type: ignore[union-attr]
        if len(output_nodes) != 1:
            raise PipeliningMetadataError(
                f"Expected 1 output node, got {len(output_nodes)}"
            )
        output_node = output_nodes[0]
        return output_node

    def _create_grad_recv_info(self, act_send_info: dict) -> tuple[_RecvInfo, ...]:
        """
        Create a tuple of `_RecvInfo` for gradients.
        Reuses output metadata from _stage_meta.outputs (populated by _create_act_send_info).
        """
        if self._stage_meta.outputs is None:
            raise PipeliningMetadataError(
                f"Stage {self.stage_index}: outputs metadata required for grad recv info. "
                f"Ensure _create_act_send_info is called first."
            )

        outputs_meta = self._stage_meta.outputs
        output_grads_metas: list[TensorMeta | None] = []
        grad_recv_infos: list[_RecvInfo] = []

        for out_idx, out_meta in enumerate(outputs_meta):
            dst_list = act_send_info.get(out_idx, [])

            # Determine the source stage for gradients
            grad_src = dst_list[0] if dst_list else self.stage_index + 1

            # Check if this output needs gradients
            if not dst_list or not out_meta.requires_grad:
                output_grads_metas.append(None)
                grad_recv_infos.append(
                    _RecvInfo(
                        input_name=f"recv_grad_for_{self.stage_index}_none_{out_idx}",
                        source=grad_src,
                        buffer=None,
                        tensor_meta=None,
                    )
                )
            else:
                # Derive grad metadata from output metadata (same shape, requires_grad=False)
                grad_meta = _TensorMeta(
                    shape=out_meta.shape,
                    stride=out_meta.stride,
                    dtype=out_meta.dtype,
                    requires_grad=False,
                )
                output_grads_metas.append(grad_meta)

                if len(dst_list) != 1:
                    raise PipeliningMetadataError(
                        "Backward of skip connections not supported yet"
                    )

                logger.debug(
                    "%s Creating grad recv buffer for output %s : %s, %s",
                    self.log_prefix,
                    out_idx,
                    grad_meta.shape,
                    grad_meta.dtype,
                )

                grad_recv_infos.append(
                    _RecvInfo(
                        input_name=f"recv_grad_for_{self.stage_index}_from_{grad_src}",
                        source=grad_src,
                        buffer=_make_tensor_from_meta(grad_meta, self.device),
                        tensor_meta=grad_meta,
                    )
                )

        self._stage_meta.output_grads = tuple(output_grads_metas)
        if self._stage_meta.inputs is not None:
            self._stage_meta.input_grads = _derive_grad_metas(self._stage_meta.inputs)
        logger.debug("%s Grad recv info: %s", self.log_prefix, grad_recv_infos)
        return tuple(grad_recv_infos)


# A helper function to create a pipeline stage based on traced pipeline information
def build_stage(
    stage_module: torch.nn.Module,
    stage_index: int,
    pipe_info: PipeInfo,
    device: torch.device,
    group: dist.ProcessGroup | None = None,
) -> _PipelineStage:
    """
    Create a pipeline stage given a stage_module to be wrapped by this stage
    and pipeline information.

    Args:
        stage_module (torch.nn.Module): the module to be wrapped by this stage
        stage_index (int): the index of this stage in the pipeline
        pipe_info (PipeInfo): information about the pipeline, can be retrieved by `pipe.info()`
        device (torch.device): the device to be used by this stage
        group (Optional[dist.ProcessGroup]): the process group to be used by this stage

    Returns:
        _PipelineStage: a pipeline stage that can run with `PipelineSchedules`.
    """
    return _PipelineStage(
        stage_module,
        stage_index,
        pipe_info,
        device,
        group,
    )


class PipelineStage(_PipelineStageBase):
    """A pipeline stage for pipeline parallelism with sequential model partitioning.

    Supports both **static** and **dynamic** metadata inference:

    Static mode:
        All of ``input_args``, ``output_args`` (and ``input_grads``/``output_grads``
        when DTensors are present) are provided at construction time.

    Dynamic mode:
        Metadata is inferred from the first microbatch at runtime; any
        statically provided args are used for validation only.

    Args:
        submodule: The ``nn.Module`` wrapped by this stage.
        stage_index: Zero-based stage ID.
        num_stages: Total number of stages in the pipeline.
        device: Device this stage runs on.
        input_args: Example input tensors (single tensor or tuple). Optional.
        output_args: Example output tensors. Optional.
        output_grads: Example output gradients (received from next stage). Optional.
        input_grads: Example input gradients (sent to previous stage). Optional.
        group: Process group for P2P communication. Defaults to the
            world process group.
        dw_builder: Builder for deferred weight-update runners used by
            zero-bubble (F/I/W) schedules.
        get_mesh: `GetMeshCallback` used during
            dynamic DTensor inference. Ignored in fully static DTensor mode.
    """

    def __init__(
        self,
        submodule: nn.Module,
        stage_index: int,
        num_stages: int,
        device: torch.device,
        input_args: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        output_args: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        output_grads: torch.Tensor | tuple[torch.Tensor | None, ...] | None = None,
        input_grads: torch.Tensor | tuple[torch.Tensor | None, ...] | None = None,
        group: dist.ProcessGroup | None = None,
        dw_builder: Callable[[], Callable[..., None]] | None = None,
        get_mesh: GetMeshCallback | None = None,
    ):
        super().__init__(submodule, stage_index, num_stages, device, group, dw_builder)

        self._mesh_cache = _MeshCache(get_mesh_cb=get_mesh)
        self._inference_mode: InferenceMode | None = None
        self._fwd_outputs_for_bwd_meta: tuple[torch.Tensor, ...] | None = None
        self._fwd_inputs_for_bwd_meta: tuple[torch.Tensor, ...] | None = None
        self._fwd_kwargs_tensors_for_bwd_meta: tuple[torch.Tensor, ...] | None = None
        self._metadata_inference_buffer_backup: (
            list[tuple[torch.Tensor, torch.Tensor]] | None
        ) = None

        # Validate and normalize args to tuples
        inputs = validate_and_normalize_to_tuple(input_args)
        outputs = validate_and_normalize_to_tuple(output_args)
        in_grads = validate_and_normalize_to_tuple(input_grads, allow_none=True)
        out_grads = validate_and_normalize_to_tuple(output_grads, allow_none=True)

        self._user_meta = _StageMeta(
            inputs=extract_tensor_metas(inputs),
            outputs=extract_tensor_metas(outputs),
            input_grads=extract_tensor_metas(in_grads, allow_none=True),
            output_grads=extract_tensor_metas(out_grads, allow_none=True),
        )

        # Cache meshes from user-provided DTensors
        for args in (inputs, outputs, in_grads, out_grads):
            if args is not None:
                self._mesh_cache.update_from_tensors(args)

        # Validate DTensor↔grad correspondence independently for inputs and outputs
        if self._user_meta.has_dtensors():
            if inputs and in_grads:
                validate_static_arg_grad_correspondence(
                    self.stage_index, inputs, in_grads, is_input=True
                )
            if outputs and out_grads:
                validate_static_arg_grad_correspondence(
                    self.stage_index, outputs, out_grads, is_input=False
                )

    def _recv_meta(self, src_stage: int) -> Any:
        """Receive metadata object from a stage on a different rank via P2P."""
        objects: list[Any] = [None]
        dist.recv_object_list(
            objects,
            src=self._resolve_peer_global_rank(src_stage),
            group=self.group,
            device=self.device,
            use_batch=True,
        )
        if len(objects) != 1:
            raise PipeliningMetadataError(
                f"Expected exactly one object to be received but got: {len(objects)}"
            )
        return objects[0]

    def _send_meta(self, meta: Any, dst_stage: int) -> None:
        """Send metadata object to a stage on a different rank via P2P."""
        dist.send_object_list(
            [meta],
            dst=self._resolve_peer_global_rank(dst_stage),
            group=self.group,
            device=self.device,
            use_batch=True,
        )

    def _is_same_rank(self, other_stage: int) -> bool:
        """Check if another stage is on the same rank as this stage."""
        return self.stage_index_to_group_rank[other_stage] == self.group_rank

    def _warmup_forward_vote(
        self, has_backward: bool, received_acc: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward phase of the warm-up vote protocol (stage 0 → N−1).

        Each stage computes a vote (1 = STATIC, 0 = DYNAMIC) based on
        ``InferenceMode.needs_dynamic``, multiplies it with the accumulated
        product from the previous stage, and forwards the result to the next
        stage.  The final product at stage N−1 is 1 iff *every* stage voted
        STATIC.

        Args:
            has_backward: Whether the schedule includes a backward pass.
            received_acc: Accumulated product tensor from the previous
                same-rank stage (V-schedule), or ``None`` for the first
                stage / cross-rank.

        Returns:
            The accumulated product tensor after this stage's vote.
        """
        my_vote = 0 if InferenceMode.needs_dynamic(self._user_meta, has_backward) else 1

        my_vote_t = torch.tensor([my_vote], dtype=torch.int32, device=self.device)

        if self.is_first:
            acc = my_vote_t
        elif self._is_same_rank(self.stage_index - 1):
            if received_acc is None:
                raise AssertionError("expected received_acc to be not None")
            acc = received_acc * my_vote_t
        else:
            peer_global = self._resolve_peer_global_rank(self.stage_index - 1)
            acc = torch.zeros(1, dtype=torch.int32, device=self.device)
            dist.recv(acc, src=peer_global, group=self.group)
            acc = acc * my_vote_t

        if not self.is_last and not self._is_same_rank(self.stage_index + 1):
            peer_global = self._resolve_peer_global_rank(self.stage_index + 1)
            dist.send(acc, dst=peer_global, group=self.group)

        return acc

    def _warmup_backward_result(
        self, received_result: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Backward phase of the warm-up vote protocol (stage N−1 → 0).

        Propagates the final accumulated product (computed in the forward
        phase) back through the pipeline so every stage learns the global
        inference mode.

        Args:
            received_result: Result tensor from the next same-rank stage
                (V-schedule), or ``None`` for the last stage / cross-rank.

        Returns:
            The global vote result tensor for this stage.
        """
        if self.is_last or self._is_same_rank(self.stage_index + 1):
            if received_result is None:
                raise AssertionError("expected received_result to be not None")
            result = received_result
        else:
            peer_global = self._resolve_peer_global_rank(self.stage_index + 1)
            result = torch.zeros(1, dtype=torch.int32, device=self.device)
            dist.recv(result, src=peer_global, group=self.group)

        if not self.is_first and not self._is_same_rank(self.stage_index - 1):
            peer_global = self._resolve_peer_global_rank(self.stage_index - 1)
            dist.send(result, dst=peer_global, group=self.group)

        return result

    def _compute_outputs(
        self,
        *args: torch.Tensor,
        module: torch.nn.Module,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor] | None:
        """Compute outputs of the submodule."""
        return module(*args, **kwargs)

    def _compute_input_grads(
        self,
        outputs: list[torch.Tensor],
        all_fwd_inputs: list[torch.Tensor],
        grad_outputs: list[torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor | None, ...]:
        """Compute input gradients via :func:`_autograd_grad_for_inputs`."""
        return _autograd_grad_for_inputs(
            outputs,
            all_fwd_inputs,
            grad_outputs,
            allow_unused=True,
        )

    def _to_tensor(self, arg: torch.Tensor | TensorMeta) -> torch.Tensor:
        """Convert a tensor or metadata to a real tensor on ``self.device``.

        Real tensors are detached and re-set requires_grad to create a fresh
        autograd leaf, isolating metadata inference from the user's graph.
        TensorMeta is materialized as an empty tensor (or DTensor via mesh cache).
        """
        if isinstance(arg, torch.Tensor):
            return arg.detach().requires_grad_(arg.requires_grad)
        elif isinstance(arg, TensorMeta):
            if isinstance(arg, _DTensorMeta):
                mesh = self._mesh_cache.get_mesh(arg.mesh_cache_key)
                return arg.to_dtensor(self.device, mesh)
            else:
                return arg.to_tensor(self.device)
        else:
            raise PipeliningMetadataError(
                f"Unsupported type {type(arg)} for _to_tensor: {arg}"
            )

    def _ones_from_metadata(self, meta: TensorMeta) -> torch.Tensor:
        """Create a ones tensor from metadata for backward inference grad_outputs."""
        local_ones = torch.ones(
            meta.shape,
            dtype=meta.dtype,
            device=self.device,
        )
        if isinstance(meta, _DTensorMeta):
            mesh = self._mesh_cache.get_mesh(meta.mesh_cache_key)
            return DTensor.from_local(
                local_ones,
                device_mesh=mesh,
                placements=meta.placements,
                shape=meta.global_shape,
                stride=meta.global_stride,
                run_check=False,
            )
        return local_ones

    def _pre_metadata_inference_backup(self) -> None:
        """Back up state that dynamic metadata inference must not publish.

        Dynamic metadata inference executes representative real forwards and
        backwards before the first runtime microbatch. Those executions are
        allowed to observe module buffers, but they must not publish training
        state updates such as running counters or routing statistics.
        """
        if self._inference_mode != InferenceMode.DYNAMIC:
            return
        if self._metadata_inference_buffer_backup is not None:
            raise RuntimeError(
                "Metadata inference buffer backup already exists. "
                "Run _post_metadata_inference_cleanup before backing up again."
            )
        self._metadata_inference_buffer_backup = [
            (buffer, buffer.detach().clone())
            for _, buffer in self.submod.named_buffers(remove_duplicate=False)
        ]

    def _forward_metadata_inference(
        self,
        args: tuple[torch.Tensor, ...] | _StageForwardMeta | None,
        kwargs: dict[str, Any] | None = None,
        has_backward: bool = False,
    ) -> _StageForwardMeta | None:
        """Run forward metadata inference (Stage 0 → N).

        Args:
            args: Real tensors (first stage), ``_StageForwardMeta``
                (same-rank), or ``None`` (cross-rank P2P).
            kwargs: Keyword arguments forwarded to the submodule.
            has_backward: Whether backward inference follows.

        Returns:
            ``_StageForwardMeta`` for the next stage, or ``None`` if sent via P2P.
        """
        kwargs = kwargs or {}

        # === RECEIVE: Get input metadata and create meta tensors ===
        if self.is_first:
            # First stage: extract metadata from real tensors
            if args is None or isinstance(args, _StageForwardMeta):
                raise PipeliningMetadataError(
                    f"Stage {self.stage_index}: First stage requires real tensors, "
                    f"got {type(args).__name__}."
                )
            tensor_args = validate_and_normalize_to_tuple(args)
            if tensor_args is None:
                raise AssertionError("expected tensor_args to be not None")
            self._stage_meta.inputs = extract_tensor_metas(tensor_args)
            inference_args = tuple(self._to_tensor(a) for a in tensor_args)
        elif self._is_same_rank(self.stage_index - 1):
            # Same-rank: _StageForwardMeta passed via argument
            if not isinstance(args, _StageForwardMeta):
                raise PipeliningMetadataError(
                    f"Stage {self.stage_index}: Expected _StageForwardMeta from same-rank "
                    f"previous stage, got {type(args).__name__}."
                )
            self._stage_meta.inputs = args.forward_metas
            inference_args = tuple(self._to_tensor(m) for m in args.forward_metas)
        else:
            # Cross-rank: receive _StageForwardMeta via P2P
            recv_meta = self._recv_meta(self.stage_index - 1)
            if not isinstance(recv_meta, _StageForwardMeta):
                raise PipeliningMetadataError(
                    f"Stage {self.stage_index}: Expected _StageForwardMeta from P2P, "
                    f"got {type(recv_meta).__name__}."
                )
            self._stage_meta.inputs = recv_meta.forward_metas
            inference_args = tuple(self._to_tensor(m) for m in recv_meta.forward_metas)

        inference_kwargs = {
            k: self._to_tensor(v) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }

        # Isolate metadata inference from user's grad context.
        # has_backward → enable_grad() so backward tracing sees grad_fn;
        # no backward → no_grad() for cross-rank consistency.
        ctx = torch.enable_grad() if has_backward else torch.no_grad()
        with ctx:
            outputs = self._compute_outputs(
                *inference_args, module=self.submod, **inference_kwargs
            )

        # Normalize outputs to tuple
        outputs = validate_and_normalize_to_tuple(outputs)

        self._stage_meta.outputs = extract_tensor_metas(outputs)

        # Store for backward metadata inference (always, even during eval)
        fwd_kwargs_tensors = tuple(
            v for v in flatten_args(inference_kwargs) if isinstance(v, torch.Tensor)
        )
        self._fwd_outputs_for_bwd_meta = outputs
        self._fwd_inputs_for_bwd_meta = inference_args
        self._fwd_kwargs_tensors_for_bwd_meta = fwd_kwargs_tensors

        # === SEND: Pass output metadata to next stage ===
        if self._stage_meta.outputs is None:
            raise PipeliningMetadataError(
                f"Stage {self.stage_index}: output metadata is required for forward inference."
            )
        fwd_meta = _StageForwardMeta(forward_metas=self._stage_meta.outputs)

        if self.is_last or self._is_same_rank(self.stage_index + 1):
            # Same-rank or last: return for caller to pass
            return fwd_meta
        else:
            # Cross-rank: send via P2P
            self._send_meta(fwd_meta, self.stage_index + 1)
            return None

    def _backward_metadata_inference(
        self,
        loss_fn: Callable[..., torch.Tensor] | None = None,
        target: torch.Tensor | None = None,
        received_grad_meta: _StageBackwardMeta | None = None,
        loss_kwargs: dict[str, Any] | None = None,
    ) -> _StageBackwardMeta | None:
        """Run backward metadata inference (Stage N → 0).

        Args:
            loss_fn: Loss function (required for the last stage).
            target: Target tensor (required for the last stage).
            received_grad_meta: Grad metadata from next same-rank stage
                (V-schedule only).
            loss_kwargs: Extra keyword arguments forwarded to the loss
                function (e.g. ``global_valid_tokens``).

        Returns:
            ``_StageBackwardMeta`` for the previous stage, or ``None`` if sent via P2P.
        """
        fwd_outputs = self._fwd_outputs_for_bwd_meta
        fwd_inputs = self._fwd_inputs_for_bwd_meta
        if fwd_outputs is None or fwd_inputs is None:
            raise PipeliningMetadataError(
                "Backward metadata inference requires forward metadata inference to run first"
            )
        kwargs_tensors = self._fwd_kwargs_tensors_for_bwd_meta or ()
        all_fwd_inputs = list(fwd_inputs) + list(kwargs_tensors)
        # Clear temporary storage early — local refs are sufficient from here
        self._fwd_outputs_for_bwd_meta = None
        self._fwd_inputs_for_bwd_meta = None
        self._fwd_kwargs_tensors_for_bwd_meta = None
        # === RECEIVE: Get output grad metadata (except last stage) ===
        if self.is_last:
            if loss_fn is None or target is None:
                raise PipeliningMetadataError(
                    f"Stage {self.stage_index}: loss_fn and target required for last stage"
                )
            inference_target = self._to_tensor(target)
            loss = loss_fn(
                fwd_outputs[0] if len(fwd_outputs) == 1 else fwd_outputs,
                inference_target,
                **(loss_kwargs or {}),
            )
            self._stage_meta.output_grads = None
            all_input_grads = self._compute_input_grads(
                [loss],
                all_fwd_inputs,
            )
        else:
            # Non-last stage: receive grad metadata from next stage
            if self._is_same_rank(self.stage_index + 1):
                # Same-rank: _StageBackwardMeta passed via argument
                if not isinstance(received_grad_meta, _StageBackwardMeta):
                    raise PipeliningMetadataError(
                        f"Stage {self.stage_index}: Expected _StageBackwardMeta from same-rank "
                        f"next stage, got {type(received_grad_meta).__name__}."
                    )
                self._stage_meta.output_grads = received_grad_meta.backward_metas
            else:
                # Cross-rank: receive _StageBackwardMeta via P2P
                recv_meta = self._recv_meta(self.stage_index + 1)
                if not isinstance(recv_meta, _StageBackwardMeta):
                    raise PipeliningMetadataError(
                        f"Stage {self.stage_index}: Expected _StageBackwardMeta from P2P, "
                        f"got {type(recv_meta).__name__}."
                    )
                self._stage_meta.output_grads = recv_meta.backward_metas

            # === COMPUTE: Build grad_outputs and compute input grads ===
            # Extract output tensors and corresponding grad_outputs from metadata
            # Must iterate together to maintain alignment
            if self._stage_meta.output_grads is None:
                raise PipeliningMetadataError(
