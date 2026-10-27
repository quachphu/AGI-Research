        work = Join.notify_join_context(self)
        if work:
            self.reducer._set_forward_pass_work_handle(
                work,
                self._divide_by_initial_world_size,  # type: ignore[arg-type]
            )

        # Calling _rebuild_buckets before forward computation,
        # It may allocate new buckets before deallocating old buckets
        # inside _rebuild_buckets. To save peak memory usage,
        # call _rebuild_buckets before the peak memory usage increases
        # during forward computation.
        # This should be called only once during whole training period.
        if torch.is_grad_enabled() and self.reducer._rebuild_buckets():
            logger.info("Reducer buckets have been rebuilt in this iteration.")
            self._has_rebuilt_buckets = True

        # sync params according to location (before/after forward) user
        # specified as part of hook, if hook was specified.
        if self._check_sync_bufs_pre_fwd():
            self._sync_buffers()

        if self._join_config.enable:
            # Notify joined ranks whether they should sync in backwards pass or not.
            self._check_global_requires_backward_grad_sync(is_joined_rank=False)

        if self.device_ids:
            moved_inputs, moved_kwargs = _to_kwargs(
                inputs,
                kwargs,
                torch.device(self.device_type, self.device_ids[0]),
                self.use_side_stream_for_tensor_copies,
            )
            args, kwargs = moved_inputs[0], moved_kwargs[0]
            # Cast inputs to reduced precision if needed.
            if self.mixed_precision is not None:
                args, kwargs = _cast_forward_inputs(
                    self.mixed_precision.param_dtype,
                    *args,
                    **kwargs,
                )
            return args, kwargs
        else:
            # Cast inputs to reduced precision if needed.
            # TODO (rohan-varma) test this codepath.
            if self.mixed_precision is not None:
                inputs, kwargs = _cast_forward_inputs(
                    self.mixed_precision.param_dtype,
                    *inputs,
                    **kwargs,
                )
            return inputs, kwargs

    def _post_forward(self, output):
        if self._use_python_reducer:
            return output

        if self._delay_all_reduce_all_params:
            self._clear_grad_buffer()
            return output

        # sync params according to location (before/after forward) user
        # specified as part of hook, if hook was specified.
        if self._check_sync_bufs_post_fwd():
            self._sync_buffers()

        if torch.is_grad_enabled() and self.require_backward_grad_sync:
            self.require_forward_param_sync = True
            # We'll return the output object verbatim since it is a freeform
            # object. We need to find any tensors in this object, though,
            # because we need to figure out which parameters were used during
            # this forward pass, to ensure we short circuit reduction for any
            # unused parameters. Only if `find_unused_parameters` is set.
            if self.find_unused_parameters and not self.static_graph:
                # Do not need to populate this for static graph.
                self.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        else:
            self.require_forward_param_sync = False

        # TODO: DDPSink is currently enabled for unused parameter detection and
        # static graph training for first iteration.
        if (self.find_unused_parameters and not self.static_graph) or (
            self.static_graph and not self._static_graph_delay_allreduce_enqueued
        ):
            (
                output_tensor_list,
                treespec,
                output_is_rref,
            ) = _tree_flatten_with_rref(output)
            output_placeholders: list[torch.Tensor | None] = [
                None for _ in range(len(output_tensor_list))
            ]
            # Do not touch tensors that have no grad_fn, which can cause issues
            # such as https://github.com/pytorch/pytorch/issues/60733
            for i, output in enumerate(output_tensor_list):
                if torch.is_tensor(output) and output.grad_fn is None:
                    output_placeholders[i] = output

            # When find_unused_parameters=True, makes tensors which require grad
            # run through the DDPSink backward pass. When not all outputs are
            # used in loss, this makes those corresponding tensors receive
            # undefined gradient which the reducer then handles to ensure
            # param.grad field is not touched and we don't error out.
            passthrough_tensor_list = _DDPSink.apply(
                weakref.ref(self),
                *output_tensor_list,
            )
            for i in range(len(output_placeholders)):
                if output_placeholders[i] is None:
                    output_placeholders[i] = passthrough_tensor_list[i]

            # Reconstruct output data structure.
            output = _tree_unflatten_with_rref(
                output_placeholders, treespec, output_is_rref
            )

        # At the end of the forward pass, reset the grad buffer and grad views
        self._clear_grad_buffer()
        return output

    def forward(self, *inputs, **kwargs):
        with torch.autograd.profiler.record_function("DistributedDataParallel.forward"):
            inputs, kwargs = self._pre_forward(*inputs, **kwargs)
            output = (
                self.module.forward(*inputs, **kwargs)
                if self._delay_all_reduce_all_params
                else self._run_ddp_forward(*inputs, **kwargs)
            )
            return self._post_forward(output)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def to_kwargs(self, inputs, kwargs, device_id):
        # Kept for BC
        return _to_kwargs(
            inputs,
            kwargs,
            torch.device(self.device_type, device_id),
            self.use_side_stream_for_tensor_copies,
        )

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)

    def train(self, mode=True):
        super().train(mode)
        return self

    # When running in join mode, schedules an allreduce to notify joined ranks
    # of whether backwards pass synchronization will run this iteration or not.
    def _check_global_requires_backward_grad_sync(self, is_joined_rank):
        if not is_joined_rank and self.require_backward_grad_sync:
            requires_sync_tensor = torch.ones(1, device=self.device)
        else:
            requires_sync_tensor = torch.zeros(1, device=self.device)

        work = dist.all_reduce(
            requires_sync_tensor, group=self.process_group, async_op=True
        )

        # (kwen2501) This if condition is a plain translation of previous
        # behavior, i.e. in the `is_joined_rank=False` case, `work.wait()`
        # is not called and it doesn't care about the result. I am guessing
        # that it just wants to fire a matching all-reduce and does not want
        # the main stream to wait.
        if is_joined_rank:
            work.wait()
            should_sync_backwards = requires_sync_tensor.item() != 0
            return should_sync_backwards
        else:
            return None  # Return value is not/should not be used.

    # When running in join mode, checks and performs sync of module buffers if
    # the models have buffers that should be synchronized in the forward pass.
    def _check_and_sync_module_buffers(self):
        if self._check_sync_bufs_pre_fwd():
            authoritative_rank = self._find_common_rank(self._distributed_rank, False)
            self._sync_module_buffers(authoritative_rank)

    # When running in join mode, agrees upon a common rank and broadcast model
    # parameters to all other ranks.
    def _sync_final_model(self, is_last_joiner):
        # Agree upon the process that will be the authoritative model copy.
        # The current rank is a candidate for being the authoritative copy if
        # is_last_joiner=True. We break ties via picking the larger rank.
        self._authoritative_rank = self._find_common_rank(
            self._distributed_rank, is_last_joiner
        )
        _sync_module_states(
            module=self.module,
            process_group=self.process_group,
            broadcast_bucket_size=self.broadcast_bucket_size,
            src=self._authoritative_rank,
            params_and_buffers_to_ignore=self.parameters_to_ignore,
            broadcast_buffers=self.forward_sync_buffers,
        )

    # Schedule comm ops to match those scheduled in the reducer's backward
    # pass.
    def _match_all_reduce_for_bwd_pass(self):
        comm_work = []
        # Schedule comm in the same order as Reducer schedules them, i.e.
        # the order of the buckets. Retrieving the bucket order from the reducer
        # ensures that we keep the same order in join mode, such as when bucket
        # order is rebuilt dynamically.

        # Returns grad_buckets in order, but real tensors are substituted with
        # zero tensors of the same shape.
        grad_buckets = self.reducer._get_zeros_like_grad_buckets()
        for grad_bucket in grad_buckets:
            # Joined processes contribute zero gradient. In the case that
            # divide_by_initial_world_size=True, we divide grads by the static
            # world size, if not, the dividing factor is reduced by the number
            # of joined processes.
            work = self.reducer._run_comm_hook(grad_bucket)
            comm_work.append(work)
        for work in comm_work:
            work.wait()

    # Allreduces the used parameter mapping across ranks.
    def _match_unused_params_allreduce(self):
        locally_used_param_map = self.reducer._get_local_used_map()
        self.process_group.allreduce(locally_used_param_map)

    def join(
        self,
        divide_by_initial_world_size: bool = True,
        enable: bool = True,
        throw_on_early_termination: bool = False,
    ):
        r"""
        Context manager for training with uneven inputs across processes in DDP.

        This context manager will keep track of already-joined DDP processes,
        and "shadow" the forward and backward passes by inserting collective
        communication operations to match with the ones created by non-joined
        DDP processes. This will ensure each collective call has a corresponding
        call by already-joined DDP processes, preventing hangs or errors that
        would otherwise happen when training with uneven inputs across
        processes. Alternatively, if the flag ``throw_on_early_termination`` is
        specified to be ``True``, all trainers will throw an error once one rank
        runs out of inputs, allowing these errors to be caught and handled
        according to application logic.

        Once all DDP processes have joined, the context manager will broadcast
        the model corresponding to the last joined process to all processes to
        ensure the model is the same across all processes
        (which is guaranteed by DDP).

        To use this to enable training with uneven inputs across processes,
        simply wrap this context manager around your training loop. No further
        modifications to the model or data loading is required.

        .. warning::
            If the model or training loop this context manager is wrapped around
            has additional distributed collective operations, such as
            ``SyncBatchNorm`` in the model's forward pass, then the flag
            ``throw_on_early_termination`` must be enabled. This is because this
            context manager is not aware of non-DDP collective communication.
            This flag will cause all ranks to throw when any one rank
            exhausts inputs, allowing these errors to be caught and recovered
            from across all ranks.

        Args:
            divide_by_initial_world_size (bool): If ``True``, will divide
                gradients by the initial ``world_size`` DDP training was launched
                with. If ``False``, will compute the effective world size
                (number of ranks that have not depleted their inputs yet) and
                divide gradients by that during allreduce. Set
                ``divide_by_initial_world_size=True`` to ensure every input
                sample including the uneven inputs have equal weight in terms of
                how much they contribute to the global gradient. This is
                achieved by always dividing the gradient by the initial
                ``world_size`` even when we encounter uneven inputs. If you set
                this to ``False``, we divide the gradient by the remaining
                number of nodes. This ensures parity with training on a smaller
                ``world_size`` although it also means the uneven inputs would
                contribute more towards the global gradient. Typically, you
                would want to set this to ``True`` for cases where the last few
                inputs of your training job are uneven. In extreme cases, where
                there is a large discrepancy in the number of inputs, setting
                this to ``False`` might provide better results.
            enable (bool): Whether to enable uneven input detection or not. Pass
                in ``enable=False`` to disable in cases where you know that
                inputs are even across participating processes. Default is
                ``True``.
            throw_on_early_termination (bool): Whether to throw an error
                or continue training when at least one rank has exhausted
                inputs. If ``True``, will throw upon the first rank reaching end
                of data. If ``False``, will continue training with a smaller
                effective world size until all ranks are joined. Note that if
                this flag is specified, then the flag
                ``divide_by_initial_world_size`` would be ignored. Default
                is ``False``.


        Example::

            >>> # xdoctest: +SKIP("Distributed")
            >>> import torch
            >>> import torch.distributed as dist
            >>> import os
            >>> import torch.multiprocessing as mp
            >>> import torch.nn as nn
            >>> # On each spawned worker
            >>> def worker(rank):
            >>>     dist.init_process_group("nccl", rank=rank, world_size=2)
            >>>     torch.cuda.set_device(rank)
            >>>     model = nn.Linear(1, 1, bias=False).to(rank)
            >>>     model = torch.nn.parallel.DistributedDataParallel(
            >>>         model, device_ids=[rank], output_device=rank
            >>>     )
            >>>     # Rank 1 gets one more input than rank 0.
            >>>     inputs = [torch.tensor([1]).float() for _ in range(10 + rank)]
            >>>     with model.join():
            >>>         for _ in range(5):
            >>>             for inp in inputs:
            >>>                 loss = model(inp).sum()
            >>>                 loss.backward()
            >>>     # Without the join() API, the below synchronization will hang
            >>>     # blocking for rank 1's allreduce to complete.
            >>>     torch.cuda.synchronize(device=rank)
        """
        return Join(
            [self],
            enable,
            throw_on_early_termination,
            divide_by_initial_world_size=divide_by_initial_world_size,
        )

    def join_hook(
        self,
        **kwargs,
    ):
        r"""
        DDP join hook enables training on uneven inputs by mirroring communications in forward and backward passes.

        Arguments:
            kwargs (dict): a :class:`dict` containing any keyword arguments
                to modify the behavior of the join hook at run time; all
                :class:`Joinable` instances sharing the same join context
                manager are forwarded the same value for ``kwargs``.

        The hook supports the following keyword arguments:
            divide_by_initial_world_size (bool, optional):
                If ``True``, then gradients are divided by the initial world
                size that DDP was launched with.
                If ``False``, then gradients are divided by the effective world
                size (i.e. the number of non-joined processes), meaning that
                the uneven inputs contribute more toward the global gradient.
                Typically, this should be set to ``True`` if the degree of
                unevenness is small but can be set to ``False`` in extreme
                cases for possibly better results.
                Default is ``True``.
        """
        divide_by_initial_world_size = kwargs.get("divide_by_initial_world_size", True)
        return _DDPJoinHook(
            self, divide_by_initial_world_size=divide_by_initial_world_size
        )

    @property
    def join_device(self):
        return self.device

    @property
    def join_process_group(self):
        return self.process_group

    def _register_buffer_comm_hook(
        self,
        state,
        hook: Callable,
        comm_hook_location=_BufferCommHookLocation.POST_FORWARD,
    ):
        r"""
        Allow custom registration of hooks that define how buffer are synchronized across ranks.

        The hook takes in an optional state and is passed in a Dict[str, Tensor]
        corresponding to buffer names and the buffers, and can run arbitrary reductions
        on buffers as opposed to DDP's default broadcast from rank 0. This is useful for
        example if a counter needs to be summed or averaged across ranks every iteration.

        Args:
            state (Any): Optional state that is passed to the hook.
            hook (Callable): Callable with the following signature:
                         ``hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]``
            comm_hook_location (_BufferCommHookLocation): Enum value indicating
                            where to run the hook.
                            _BufferCommHookLocation.PRE_FORWARD means that the
                            hook will run _before_ the forward pass, and
                            _BufferCommHookLocation.POST_FORWARD means that the
                            hook will run _after_ the forward pass.

            NOTE: To maximize performance, users can return a
                list[torch.futures.Future] from their hook, and DDP will
                install and await these hooks appropriately at the end of
                the backward pass. This will ensure all buffers are
                synchronized by the end of the backward pass. If this
                setting is used, it is recommended to pass
                comm_hook_location=_BufferCommHookLocation.POST_FORWARD,
                which will trigger the hook after the forward pass.
                If _BufferCommHookLocation.PRE_FORWARD is used, users must
                ensure appropriate synchronization when manipulating GPU
                buffers in the forward pass.
        """
        if not callable(hook):
            raise AssertionError(f"hook must be callable, got {type(hook).__name__}")
        self.buffer_hook = _BufferCommHook(
            buffer_comm_hook=hook,
            buffer_comm_hook_state=state,
            buffer_comm_hook_location=comm_hook_location,
        )

    def register_comm_hook(self, state: object, hook: Callable) -> None:
        r"""
        Register communication hook for user-defined DDP aggregation of gradients across multiple workers.

        This hook would be very useful for researchers to try out new ideas. For
        example, this hook can be used to implement several algorithms like GossipGrad
        and gradient compression which involve different communication strategies for
        parameter syncs while running Distributed DataParallel training.

        Args:
            state (object): Passed to the hook to maintain any state information during the training process.
                            Examples include error feedback in gradient compression,
                            peers to communicate with next in GossipGrad, etc.

                            It is locally stored by each worker
                            and shared by all the gradient tensors on the worker.
            hook (Callable): Callable with the following signature:
                             ``hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]``:

                             This function is called once the bucket is ready. The
                             hook can perform whatever processing is needed and return
                             a Future indicating completion of any async work (ex: allreduce).
                             If the hook doesn't perform any communication, it still
                             must return a completed Future. The Future should hold the
                             new value of grad bucket's tensors. Once a bucket is ready,
                             c10d reducer would call this hook and use the tensors returned
                             by the Future and copy grads to individual parameters.
                             Note that the future's return type must be a single tensor.

                             We also provide an API called ``get_future`` to retrieve a
                             Future associated with the completion of ``c10d.ProcessGroup.Work``.
                             ``get_future`` is currently supported for NCCL and also supported for most
                             operations on GLOO and MPI, except for peer to peer operations (send/recv).

        .. warning ::
            Grad bucket's tensors will not be predivided by world_size. User is responsible
            to divide by the world_size in case of operations like allreduce.

        .. warning ::
            DDP communication hook can only be registered once and should be registered
            before calling backward.

        .. warning ::
            The Future object that hook returns should contain a single tensor
            that has the same shape with the tensors inside grad bucket.

        .. warning ::
            ``get_future`` API supports NCCL, and partially GLOO and MPI backends (no support
            for peer-to-peer operations like send/recv) and will return a ``torch.futures.Future``.

        Example::
            Below is an example of a noop hook that returns the same tensor.

            >>> # xdoctest: +SKIP('undefined name')
            >>> def noop(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
            >>>     fut = torch.futures.Future()
            >>>     fut.set_result(bucket.buffer())
            >>>     return fut
            >>> ddp.register_comm_hook(state=None, hook=noop)

        Example::
            Below is an example of a Parallel SGD algorithm where gradients are encoded before
            allreduce, and then decoded after allreduce.

            >>> # xdoctest: +SKIP('undefined name')
            >>> def encode_and_decode(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
            >>>     encoded_tensor = encode(bucket.buffer())  # encode gradients
            >>>     fut = torch.distributed.all_reduce(encoded_tensor).get_future()
            >>>     # Define the then callback to decode.
            >>>     def decode(fut):
            >>>         decoded_tensor = decode(fut.value()[0])  # decode gradients
            >>>         return decoded_tensor
            >>>     return fut.then(decode)
            >>> ddp.register_comm_hook(state=None, hook=encode_and_decode)
        """
        self._check_comm_hook(hook)
        if self.logger is None:
            raise AssertionError("self.logger must not be None")
        self.logger._set_comm_hook_name(hook.__qualname__)
        self._comm_hooks.append((hook, state))
        dist._register_comm_hook(self.reducer, state, hook)

    def _register_builtin_comm_hook(self, comm_hook_type):
        r"""
        Register a built-in communication hook that specifies how DDP aggregates gradients across multiple workers.

        The built-in hooks aim to provide efficient C++ implementations for certain hooks,
        which might not be as efficient if implemented in Python using a Python communication hook.

        Args:
            comm_hook_type (dist.BuiltinCommHookType): type of communication hook, such as ALLREDUCE, FP16_COMPRESS, etc.

        .. warning ::
            DDP communication hook can only be registered once and should be registered
            before calling backward.

        Example::
            Below is an example of a FP16 compression where gradients are
            compressed into 16-bit floating-point numbers before allreduce, and
            then decompressed after allreduce.

            >>> # xdoctest: +SKIP('undefined name')
            >>> ddp._register_builtin_comm_hook(dist.BuiltinCommHookType.FP16_COMPRESS)

        """
        if self.logger is None:
            raise AssertionError("self.logger must not be None")
        self.logger._set_comm_hook_name(str(comm_hook_type))
        dist._register_builtin_comm_hook(self.reducer, comm_hook_type)

    def _register_fused_optim(self, optim: type, *args, optim_params=None, **kwargs):
        r"""
        Register an optimizer in DDP to optimize parameter immediately after its gradient reduction.

        Registers an optimizer with DDP such that the optimization for a
        parameter will run immediately when that parameter's gradient is
        finished with reduction, instead of waiting for all parameters'
        gradients to finish reduction. This can result in a training speedup
        depending on your workload since the optimizer can run while gradient
        reduction for other parameters are still ongoing. In addition, this has
        the potential to reduce peak memory consumption during training, as it
        only needs to load the per-parameter optimizer states of a single
        parameter at a time, instead of loading all per-parameter optimizer
        states at once.

        Args:
            optim (Type): a ``torch.optim.Optimizer`` class to be registered
            as a fused optimizer.
            *args (Sequence[Any]): Arguments to forward to `optim`.
            optim_params (Optional[Iterable[torch.Tensor]]): Set of parameters
            to optimize, similar to `params` argument of traditional `torch.optim`
            Optimizers. If this is omitted, all DDP model parameters will be
            optimized.
            **kwargs: (Dict[str, Any]): Keyword arguments to forward to `optim`.

        .. warning ::
            _register_fused_optim should only be called once on a DDP instance,
            and registering multiple fused optimizers for the same DDP model
            is not currently supported. Please ping
            https://github.com/pytorch/pytorch/issues/71595 if this is necessary
            for your use case.

        .. warning ::
            _register_fused_optim and register_comm_hook currently do not
            compose together, meaning that custom DDP communication hooks are
            not supported with overlapped optimizers. Please ping
            https://github.com/pytorch/pytorch/issues/71595 if this is necessary
            for your use case.

        .. warning ::
            Gradient accumulation and DDP `no_sync` are currently not supported
            with overlapped optimizer. Please ping
            https://github.com/pytorch/pytorch/issues/71595 if this is necessary
            for your use case.

        Example::

            >>> # xdoctest: +SKIP("No rendezvous handler")
            >>> torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
            >>> net = torch.nn.parallel.DistributedDataParallel(model, pg)
            >>> lr = 1e-2
            >>> betas = (0.9, 0.99)
            >>> eps = 1e-6
            >>> net._register_fused_optim(torch.optim.Adam, lr, betas=betas, eps=eps)
            >>> # Example with subset of parameters
            >>> params_to_opt = [list(net.parameters())[0]]
            >>> net._register_fused_optim(
            ...   torch.optim.Adam, lr, optim_params=params_to_opt,  betas=betas, eps=eps
            ... )
        """
        # Note: importing in function, otherwise this will cause a circular
        # import as optimizer_overlap module needs to import DistributedDataParallel.
        from torch.distributed.algorithms._optimizer_overlap import _as_overlapped_optim

        overlapped_optim = _as_overlapped_optim(optim, optim_params, *args, **kwargs)
        try:
            overlapped_optim.register_ddp(self)
        except NotImplementedError as e:
            raise RuntimeError(
                f"{optim} does not support overlapped DDP. Please file an issue to PyTorch or the respective owner of {optim}."
            ) from e

    def _distributed_broadcast_coalesced(
        self, tensors, buffer_size, authoritative_rank=0
    ):
        dist._broadcast_coalesced(
            self.process_group, tensors, buffer_size, authoritative_rank
        )

    def _check_sync_bufs_post_fwd(self):
        return (
            self.will_sync_module_buffers()
            and hasattr(self, "buffer_hook")
            and self.buffer_hook.buffer_comm_hook_location
            == _BufferCommHookLocation.POST_FORWARD
        )

    def _check_sync_bufs_pre_fwd(self):
        return self.will_sync_module_buffers() and (
            not hasattr(self, "buffer_hook")
            or self.buffer_hook.buffer_comm_hook_location
            == _BufferCommHookLocation.PRE_FORWARD
        )

    def will_sync_module_buffers(self):
        return (
            self.require_forward_param_sync
            and self.forward_sync_buffers
            and len(self.modules_buffers) > 0
        )

    def _find_common_rank(self, input_rank, rank_cond):
        # -1 indicates that this rank is not under consideration to be the
        # common_rank
        rank_to_use = torch.tensor(
            [input_rank if rank_cond else -1],
            device=self.device,
        )
        dist.all_reduce(rank_to_use, op=ReduceOp.MAX, group=self.process_group)
        if rank_to_use.item() == -1:
            self._log_and_throw(
                ValueError,
                "BUG! Expected rank_cond to be true for at least one process."
                " This indicates a bug in PyTorch, please report an issue.",
            )
        return rank_to_use.item()

    def _sync_buffers(self):
        with torch.no_grad():
            # module buffer sync
            # Synchronize buffers across processes.
            # If we are running DDP with the join manager, we have to agree
            # upon a rank to sync module buffers from, since rank 0 may
            # already have been joined and have stale module buffers.
            if self._join_config.enable:
                authoritative_rank = self._find_common_rank(
                    self._distributed_rank, True
                )
            else:
                # The process with rank 0 is considered the authoritative copy.
                authoritative_rank = 0
            # Update self.modules_buffers in case any buffers were
            # reassigned.
            self._assign_modules_buffers()
            with torch.autograd._unsafe_preserve_version_counter(
                tuple(self.modules_buffers)
            ):
                self._sync_module_buffers(authoritative_rank)

    def _sync_module_buffers(self, authoritative_rank):
        if not hasattr(self, "buffer_hook"):
            self._default_broadcast_coalesced(authoritative_rank=authoritative_rank)
        else:
            hook = self.buffer_hook.buffer_comm_hook
            state = self.buffer_hook.buffer_comm_hook_state
            futs = hook(state, self.named_module_buffers)
            if futs is not None:
                self.reducer._install_post_backward_futures(futs)

    def _default_broadcast_coalesced(
        self, bufs=None, bucket_size=None, authoritative_rank=0
    ):
        """
        Broadcasts buffers from rank 0 to rest of workers.

        If bufs, bucket_size are None, default values self.modules_buffers
        and self.broadcast_bucket_size are used instead.
        """
        if bufs is None:
            bufs = self.modules_buffers
        if bucket_size is None:
            bucket_size = self.broadcast_bucket_size

        self._distributed_broadcast_coalesced(bufs, bucket_size, authoritative_rank)

    def _passing_sync_batchnorm_handle(self, module):
        for layer in module.modules():
            if isinstance(layer, torch.nn.modules.SyncBatchNorm):
                if self.device_type == "cpu":
                    self._log_and_throw(
                        ValueError,
                        "SyncBatchNorm layers only work with GPU modules",
                    )

    def _check_comm_hook(self, hook):
        if not callable(hook):
            self._log_and_throw(TypeError, "Communication hook must be callable.")

        sig = inspect.signature(hook)
        if (
            sig.parameters["bucket"].annotation != inspect._empty
            and sig.parameters["bucket"].annotation != dist.GradBucket
        ):
            self._log_and_throw(
                ValueError,
                "Communication hook: bucket annotation should be dist.GradBucket.",
            )

        if (
            sig.return_annotation != inspect._empty
            and sig.return_annotation != torch.futures.Future[torch.Tensor]
        ):
            self._log_and_throw(
                ValueError,
                "Communication hook: return annotation should be torch.futures.Future[torch.Tensor].",
            )

        if hook.__name__ in ["bf16_compress_hook", "bf16_compress_wrapper_hook"]:
            cuda_supported = (
                torch.version.cuda is not None
            ) or torch.version.hip is not None
            nccl_supported = (
                dist.is_available()
                and dist.is_nccl_available()
                and torch.cuda.nccl.version() >= (2, 10)
            )
            xpu_xccl_supported = (
                dist.is_available()
                and dist.is_xccl_available()
                and torch.xpu.is_available()
            )

            if not ((cuda_supported and nccl_supported) or xpu_xccl_supported):
                self._log_and_throw(
                    TypeError,
                    "BF16 all reduce communication hook required CUDA 11+ and NCCL 2.10+ or XPU and XCCL",
                )

    @property
    def _distributed_rank(self):
        return dist.get_rank(self.process_group)

    @staticmethod
    def _get_data_parallel_params(module, named_params=False):
        """Return a generator of parameters managed by a given DDP unit."""
        for param in (
            module.parameters() if not named_params else module.named_parameters()
        ):
            if not hasattr(param, "_ddp_ignored"):
                yield param

    @staticmethod
    def _set_params_and_buffers_to_ignore_for_model(
        module, params_and_buffers_to_ignore
    ):
        """
        Set parameters and buffers to be ignored by DDP.

        Expected format for parameters is the fully qualified name: {module_name}.{param_name}, and
        similarly, {module_name}.{buffer_name} for buffers. For example:
        params_to_ignore = []
        # NB: model here is vanilla PyTorch module, not yet wrapped with DDP.
        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if should_ignore(param):
                    # Create expected format
                    fqn = f"{module_name}.{param_name}"
                    params_to_ignore.append(fqn)
        torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
            model,
            params_to_ignore
        )
        """
        # This is a workaround to set parameters and buffers DDP should ignore
        # during synchronization. It will be removed when the API is finalized
        # as part of addressing https://github.com/pytorch/pytorch/issues/43690.
        module._ddp_params_and_buffers_to_ignore = params_and_buffers_to_ignore
        for name, param in module.named_parameters():
            if name in params_and_buffers_to_ignore:
                param._ddp_ignored = True
        for name, buffer in module.named_buffers():
            if name in params_and_buffers_to_ignore:
                buffer._ddp_ignored = True

    def _get_ddp_logging_data(self):
        r"""
        Return a dictionary of logging data for debugging and analysis.

        This interface can be called after DistributedDataParallel() is
        constructed. It returns a dictionary of logging data. It could help
        for debugging and analysis. The logging data includes DistributedDataParallel
        constructor input parameters, some internal states of DistributedDataParallel
        and performance metrics. Simply print the dictionary and see what
        these metrics are.
        This is a prototype interface and subject to change in the future.
        """
        if self.logger is None:
            raise AssertionError("self.logger must not be None")
        ddp_logging_data = self.logger._get_ddp_logging_data()
        return {**ddp_logging_data.strs_map, **ddp_logging_data.ints_map}

    def _set_ddp_runtime_logging_sample_rate(self, sample_rate):
        r"""
        Set sample_rate of collecting runtime stats.

        This interface allows users to set sample_rate of collecting
        runtime stats. The runtime stats will be recorded for the
        first 10 iterations, after 10 iterations runtime stats will be
        recorded once every "sample_rate" training iterations. In
        default, runtime stats are recorded for the first 10 iterations,
        after 10 iterations runtime stats are recorded once every
        "kDDPRuntimeLoggingSampleRate=100" training iterations.
        This is a prototype interface and subject to change in the future.
        """
        if sample_rate < 1:
            self._log_and_throw(
                ValueError,
                "DDP runtime logging sample rate should be equal or greater than 1",
            )
        self.reducer._set_ddp_runtime_logging_sample_rate(sample_rate)

    def _set_static_graph(self):
        """
        Set static graph for DDP.

        It is recommended to set static graph in the DDP constructor, which will
        call this private API internally.
        """
        # If self.static_graph has been set, no need to set it again
        if self.static_graph:
            warnings.warn(
                "You've set static_graph to be True, no need to set it again.",
                stacklevel=2,
            )
            return
        self.static_graph = True
        self._static_graph_delay_allreduce_enqueued = False
        self.reducer._set_static_graph()
        if self.logger is None:
            raise AssertionError("self.logger must not be None")
        self.logger._set_static_graph()
        if self.find_unused_parameters:
            warnings.warn(
                "You passed find_unused_parameters=true to DistributedDataParallel, "
                "`_set_static_graph` will detect unused parameters automatically, so "
                "you do not need to set find_unused_parameters=true, just be sure these "
                "unused parameters will not change during training loop while calling "
                "`_set_static_graph`.",
                stacklevel=2,
            )

    def _remove_autograd_hooks(self):
        """Remove autograd hooks registered by the reducer on the model parameters."""
        self.reducer._remove_autograd_hooks()

    def _check_reducer_finalized(self):
        """
        Check if the reducer has processed all buckets and finalized the backward appropriately.

        It is useful to call this method after calling .backward() in your training loop
        in order to avoid subsequent hard to debug errors down the road due to the
        reducer not finalizing backward.
        """
        self.reducer._check_reducer_finalized()

    def _set_sparse_metadata(self, global_unique_ids):
        self.reducer._set_sparse_metadata(global_unique_ids)

    def _update_process_group(self, new_process_group):
        """
        Dynamically updates the process group for DDP so that we can shrink/expand DDP
        world size without having to reinitialize DDP.

        NOTE: If you are using custom communications hooks via, register_comm_hook,
        you need to update the process groups for those hooks separately.
        """
        # Force a rebuild of buckets for a new process group. This ensures all ranks
        # are synchronized in terms of when they will rebuild buckets and also
        # re-evaluates previous assumptions of buckets given the world size might have
        # changed.
        self._has_rebuilt_buckets = False
        self.reducer._reset_state()

        if not _rank_not_in_group(new_process_group):
            self.process_group = new_process_group
            self.reducer._update_process_group(new_process_group)

    def _set_ddp_sink_clone(self, val: bool):
        """
        Sets whether or not DDPSink should clone the output tensors or not.
        The default is True since if the loss is modified in place we run
        into the view is modified in-place error.

        Although, cloning the tensors can add significant memory and
        performance hit if the number and size of tensors are large. As
        a result, this can be set to False if you are not modifying the
        loss in place.
        """
        self._ddp_sink_clone = val
<|endoftext|>import warnings
from itertools import chain

import torch
from torch._utils import _get_device_index
from torch.autograd import Function
from torch.nn.parallel import comm


class Broadcast(Function):
    @staticmethod
    def forward(ctx, target_gpus, *inputs):
        if not all(i.device.type != "cpu" for i in inputs):
            raise AssertionError("Broadcast function not implemented for CPU tensors")
        target_gpus = [_get_device_index(x, True) for x in target_gpus]
        ctx.target_gpus = target_gpus
        if len(inputs) == 0:
            return ()
        ctx.num_inputs = len(inputs)
        ctx.input_device = inputs[0].get_device()

        ctx.complex_mask = [inp.is_complex() for inp in inputs]

        outputs = comm.broadcast_coalesced(inputs, ctx.target_gpus)

        for device_outputs in outputs:
            for i, is_complex in enumerate(ctx.complex_mask):
                if is_complex:
                    device_outputs[i] = torch.view_as_complex(device_outputs[i])

        non_differentiables = []
        for idx, input_requires_grad in enumerate(ctx.needs_input_grad[1:]):
            if not input_requires_grad:
                non_differentiables.extend(output[idx] for output in outputs)
        ctx.mark_non_differentiable(*non_differentiables)
        return tuple(chain.from_iterable(outputs))

    @staticmethod
    def backward(ctx, *grad_outputs):
        grads = ReduceAddCoalesced.apply(
            ctx.input_device, ctx.num_inputs, *grad_outputs
        )

        return (None,) + grads


class ReduceAddCoalesced(Function):
    @staticmethod
    def forward(ctx, destination, num_inputs, *grads):
        ctx.target_gpus = [
            grads[i].get_device() for i in range(0, len(grads), num_inputs)
        ]

        complex_mask = [grads[i].is_complex() for i in range(num_inputs)]
        ctx.complex_mask = complex_mask

        grads_converted = tuple(
            torch.view_as_real(g) if g.is_complex() else g for g in grads
        )

        grads_ = [
            grads_converted[i : i + num_inputs]
            for i in range(0, len(grads_converted), num_inputs)
        ]
        results = comm.reduce_add_coalesced(grads_, destination)

        results = tuple(
            torch.view_as_complex(r) if is_complex else r
            for r, is_complex in zip(results, complex_mask)
        )

        return results

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (
            None,
            None,
        ) + Broadcast.apply(ctx.target_gpus, *grad_outputs)


class Gather(Function):
    @staticmethod
    def forward(ctx, target_device, dim, *inputs):
        if not all(i.device.type != "cpu" for i in inputs):
            raise AssertionError("Gather function not implemented for CPU tensors")
        if target_device == "cpu":
            ctx.target_device = "cpu"
        else:
            target_device = _get_device_index(target_device, True)
            ctx.target_device = target_device
        ctx.dim = dim
        ctx.input_gpus = tuple(i.get_device() for i in inputs)
        if all(t.dim() == 0 for t in inputs) and dim == 0:
            inputs = tuple(t.view(1) for t in inputs)
            warnings.warn(
                "Was asked to gather along dimension 0, but all "
                "input tensors were scalars; will instead unsqueeze "
                "and return a vector.",
                stacklevel=2,
            )
            ctx.unsqueezed_scalar = True
        else:
            ctx.unsqueezed_scalar = False
        ctx.input_sizes = tuple(i.size(ctx.dim) for i in inputs)

        is_complex = len(inputs) > 0 and inputs[0].is_complex()

        output = comm.gather(inputs, ctx.dim, ctx.target_device)

        if is_complex:
            output = torch.view_as_complex(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        scattered_grads = Scatter.apply(
            ctx.input_gpus, ctx.input_sizes, ctx.dim, grad_output
        )
        if ctx.unsqueezed_scalar:
            scattered_grads = tuple(g[0] for g in scattered_grads)
        return (None, None) + scattered_grads


class Scatter(Function):
    @staticmethod
    def forward(ctx, target_gpus, chunk_sizes, dim, input):
        target_gpus = [_get_device_index(x, True) for x in target_gpus]
        ctx.dim = dim
        ctx.input_device = input.get_device() if input.device.type != "cpu" else -1
        streams = None
        if torch.accelerator.is_available() and ctx.input_device == -1:
            # Perform CPU to GPU copies in a background stream
            streams = [_get_stream(torch.device(device)) for device in target_gpus]

        is_complex = input.is_complex()

        outputs = comm.scatter(input, target_gpus, chunk_sizes, ctx.dim, streams)

        if is_complex:
            outputs = tuple(torch.view_as_complex(o) for o in outputs)

        # Synchronize with the copy stream
        if streams is not None:
            for i, output in enumerate(outputs):
                with torch.accelerator.device_index(target_gpus[i]):
                    main_stream = torch.accelerator.current_stream()
                    main_stream.wait_stream(streams[i])
                    output.record_stream(main_stream)
        return outputs

    @staticmethod
    def backward(ctx, *grad_output):
        return None, None, None, Gather.apply(ctx.input_device, ctx.dim, *grad_output)


# background streams used for copying
_streams: list[torch.Stream | None] | None = None


def _get_stream(device: torch.device):
    """Get a background stream for copying between CPU and target device."""
    global _streams
    if device.type == "cpu" or not torch.accelerator.is_available():
        return None
    if torch.accelerator.current_accelerator().type != device.type:
        raise AssertionError(
            f"Expected current accelerator type {torch.accelerator.current_accelerator().type} "
            f"to match device type {device.type}"
        )
    if _streams is None:
        _streams = [None] * torch.accelerator.device_count()
    if _streams[device.index] is None:
        _streams[device.index] = torch.Stream(device.index)
    return _streams[device.index]
<|endoftext|>from collections import OrderedDict
from collections.abc import Iterator, Sequence
from typing import cast, TYPE_CHECKING, TypeVar
from typing_extensions import TypeIs

import torch
from torch._utils import _get_device_index
from torch.nn.modules import Module
from torch.nn.parallel import comm


if TYPE_CHECKING:
    from torch._C import ScriptMethod
    from torch.jit import ScriptModule
    from torch.jit._state import EnabledProxy


__all__ = ["replicate"]


def _is_script_module(module: Module) -> TypeIs["ScriptModule"]:
    import torch.jit

    return isinstance(module, torch.jit.ScriptModule)


def _is_script_method(module: object) -> TypeIs["ScriptMethod"]:
    import torch.jit

    return isinstance(module, torch._C.ScriptMethod)


def _init_script_module() -> "ScriptModule":
    import torch.jit

    return torch.jit.ScriptModule()


def _is_jit_enabled() -> "EnabledProxy":
    import torch.jit._state

    return torch.jit._state._enabled


# Check if we can safely replicate the module.
# there are two types of module:
# 1. python modules
# 2. ScriptModule
#
# currently a module cannot be replicated properly if the descendants of
# any ScriptModule contains python module (type 1 above)
def _replicatable_module(module: Module, memo: set[Module] | None = None) -> bool:
    # module.modules() contains module itself as the first element
    def descendant_modules(module: Module) -> Iterator[Module]:
        gen = module.modules()
        next(gen)
        return gen

    if not _is_jit_enabled():
        return True
    if memo is None:
        memo = set()

    # memoize visited modules
    memo.add(module)
    if _is_script_module(module):
        memo.update(descendant_modules(module))
        return all(
            _is_script_module(descendant) for descendant in descendant_modules(module)
        )

    for child in module.children():
        # since any unreplicatable module will cause the check to return
        # False early, visited modules here can be safely ignored.
        if child in memo:
            continue
        if not _replicatable_module(child, memo):
            return False

    return True


def _broadcast_coalesced_reshape(
    tensors: Sequence[torch.Tensor],
    devices: Sequence[int | torch.device],
    detach: bool = False,
) -> list[list[torch.Tensor]]:
    from torch.nn.parallel._functions import Broadcast

    if len(tensors) == 0:
        return []

    if detach:
        complex_mask = [
            not isinstance(t, torch.nn.UninitializedParameter) and t.is_complex()
            for t in tensors
        ]

        outputs = comm.broadcast_coalesced(tensors, devices)

        for device_outputs in outputs:
            for i, is_complex in enumerate(complex_mask):
                if is_complex:
                    device_outputs[i] = torch.view_as_complex(device_outputs[i])

        return outputs
    else:
        tensor_copies = Broadcast.apply(devices, *tensors)
        return [
            list(tensor_copies[i : i + len(tensors)])
            for i in range(0, len(tensor_copies), len(tensors))
        ]


T = TypeVar("T", bound=Module)


def replicate(
    network: T,
    devices: Sequence[int | torch.device],
    detach: bool = False,
) -> list[T]:
    if not _replicatable_module(network):
        raise RuntimeError(
            "Cannot replicate network where python modules are children of ScriptModule"
        )

    if not devices:
        return []

    devices = [_get_device_index(x, True) for x in devices]
    num_replicas = len(devices)

    params = list(network.parameters())
    param_indices = {param: idx for idx, param in enumerate(params)}
    param_copies = _broadcast_coalesced_reshape(params, devices, detach)

    buffers = list(network.buffers())
    buffers_rg: list[torch.Tensor] = []
    buffers_not_rg: list[torch.Tensor] = []
    for buf in buffers:
        if buf.requires_grad and not detach:
            buffers_rg.append(buf)
        else:
            buffers_not_rg.append(buf)

    buffer_indices_rg = {buf: idx for idx, buf in enumerate(buffers_rg)}
    buffer_indices_not_rg = {buf: idx for idx, buf in enumerate(buffers_not_rg)}

    buffer_copies_rg = _broadcast_coalesced_reshape(buffers_rg, devices, detach=detach)
    buffer_copies_not_rg = _broadcast_coalesced_reshape(
        buffers_not_rg, devices, detach=True
    )

    modules = list(network.modules())
    module_copies: list[list[Module]] = [[] for _ in devices]
    module_indices: dict[Module, int] = {}

    for i, module in enumerate(modules):
        module_indices[module] = i
        for j in range(num_replicas):
            replica = module._replicate_for_data_parallel()
            # This is a temporary fix for DDP. DDP needs to access the
            # replicated model parameters. It used to do so through
            # `module.parameters()`. The fix added in #33907 for DP stops the
            # `parameters()` API from exposing the replicated parameters.
            # Hence, we add a `_former_parameters` dict here to support DDP.
            replica._former_parameters = OrderedDict()

            module_copies[j].append(replica)

    for i, module in enumerate(modules):
        for key, child in module._modules.items():
            if child is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._modules[key] = None
            else:
                module_idx = module_indices[child]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    setattr(replica, key, module_copies[j][module_idx])
        for key, param in module._parameters.items():
            if param is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._parameters[key] = None
            else:
                param_idx = param_indices[param]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    param_copy = param_copies[j][param_idx]
                    # parameters in replicas are no longer leaves,
                    # so setattr them as non-parameter attributes
                    setattr(replica, key, param_copy)
                    # expose the parameter for DDP
                    replica._former_parameters[key] = param_copy  # type: ignore[operator, index]
        for key, buf in module._buffers.items():  # type: ignore[assignment]
            if buf is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._buffers[key] = None
            else:
                if buf.requires_grad and not detach:
                    buffer_copies = buffer_copies_rg
                    buffer_idx = buffer_indices_rg[buf]
                else:
                    buffer_copies = buffer_copies_not_rg
                    buffer_idx = buffer_indices_not_rg[buf]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    setattr(replica, key, buffer_copies[j][buffer_idx])

    return [cast(T, module_copies[j][0]) for j in range(num_replicas)]
<|endoftext|># mypy: allow-untyped-defs
import warnings

import torch
from torch._utils import (
    _flatten_dense_tensors,
    _get_device_index,
    _handle_complex,
    _reorder_tensors_as,
    _take_tensors,
    _unflatten_dense_tensors,
)
from torch.cuda import nccl


def broadcast(tensor, devices=None, *, out=None):
    r"""Broadcasts a tensor to specified GPU devices.

    Args:
        tensor (Tensor): tensor to broadcast. Can be on CPU or GPU.
        devices (Iterable[torch.device, str or int], optional): an iterable of
          GPU devices, among which to broadcast.
        out (Sequence[Tensor], optional, keyword-only): the GPU tensors to
          store output results.

    .. note::
        Exactly one of :attr:`devices` and :attr:`out` must be specified.

    Returns:
        - If :attr:`devices` is specified,
            a tuple containing copies of :attr:`tensor`, placed on
            :attr:`devices`.
        - If :attr:`out` is specified,
            a tuple containing :attr:`out` tensors, each containing a copy of
            :attr:`tensor`.
    """
    tensor = _handle_complex(tensor)
    if not ((devices is None) ^ (out is None)):
        raise RuntimeError(
            f"Exactly one of 'devices' and 'out' must be specified, but got devices={devices} and out={out}"
        )
    if devices is not None:
        devices = [_get_device_index(d) for d in devices]
        return torch._C._broadcast(tensor, devices)
    else:
        # pyrefly: ignore [bad-argument-type]
        return torch._C._broadcast_out(tensor, out)


def broadcast_coalesced(tensors, devices, buffer_size=10485760):
    """Broadcast a sequence of tensors to the specified GPUs.

    Small tensors are first coalesced into a buffer to reduce the number of synchronizations.

    Args:
        tensors (sequence): tensors to broadcast. Must be on the same device,
          either CPU or GPU.
        devices (Iterable[torch.device, str or int]): an iterable of GPU
          devices, among which to broadcast.
        buffer_size (int): maximum size of the buffer used for coalescing

    Returns:
        A tuple containing copies of :attr:`tensor`, placed on :attr:`devices`.
    """
    devices = [_get_device_index(d) for d in devices]
    tensors = [_handle_complex(t) for t in tensors]
    return torch._C._broadcast_coalesced(tensors, devices, buffer_size)


def reduce_add(inputs, destination=None):
    """Sum tensors from multiple GPUs.

    All inputs should have matching shapes, dtype, and layout. The output tensor
    will be of the same shape, dtype, and layout.

    Args:
        inputs (Iterable[Tensor]): an iterable of tensors to add.
        destination (int, optional): a device on which the output will be
            placed (default: current device).

    Returns:
        A tensor containing an elementwise sum of all inputs, placed on the
        :attr:`destination` device.
    """
    destination = _get_device_index(destination, optional=True)
    input_size = inputs[0].size()
    root_index = None  # index of input tensor that already is on the correct device
    for i, inp in enumerate(inputs):
        if inp.device.type == "cpu":
            raise AssertionError(
                f"reduce_add expects all inputs to be on GPUs, but input {i} is on CPU"
            )
        if inp.get_device() == destination:
            root_index = i
        if inp.size() != input_size:
            got = "x".join(str(x) for x in inp.size())
            expected = "x".join(str(x) for x in input_size)
            raise ValueError(
                f"input {i} has invalid size: got {got}, but expected {expected}"
            )
    if root_index is None:
        raise RuntimeError(
            "reduce_add expects destination to be on the same GPU with one of the tensors"
        )

    if len(inputs) == 1:
        return inputs[0]

    if nccl.is_available(inputs):
        result = torch.empty_like(inputs[root_index])
        nccl.reduce(inputs, output=result, root=root_index)
    else:
        destination_device = torch.device(inputs[root_index].device.type, destination)
        nonroot = [t for i, t in enumerate(inputs) if i != root_index]
        # make a new tensor w/o clone
        result = inputs[root_index] + nonroot[0].to(
            device=destination_device, non_blocking=True
        )
        for other in nonroot[1:]:
            result.add_(other.to(device=destination_device, non_blocking=True))
    return result


def reduce_add_coalesced(inputs, destination=None, buffer_size=10485760):
    """Sum tensors from multiple GPUs.

    Small tensors are first coalesced into a buffer to reduce the number
    of synchronizations.

    Args:
        inputs (Iterable[Iterable[Tensor]]): iterable of iterables that
            contain tensors from a single device.
        destination (int, optional): a device on which the output will be
            placed (default: current device).
        buffer_size (int): maximum size of the buffer used for coalescing

    Returns:
        A tuple of tensors containing an elementwise sum of each group of
        inputs, placed on the ``destination`` device.
    """
    # TODO: When `len(inputs) == 1` and all inputs are on `destination`, just
    #       return `inputs`.
    dense_tensors: list[list] = [[] for _ in inputs]  # shape (num_gpus, num_tensors)
    output = []
    ref_order = []
    # process sparse ones first since they may have different sizes on different gpus
    for tensor_at_gpus in zip(*inputs, strict=True):
        if all(t.is_sparse for t in tensor_at_gpus):
            result = reduce_add(tensor_at_gpus, destination)  # this will be sparse too
            output.append(result)
            ref_order.append(tensor_at_gpus[0])
        else:
            for coll, t in zip(dense_tensors, tensor_at_gpus, strict=True):
                coll.append(t.to_dense() if t.is_sparse else t)
            ref_order.append(dense_tensors[0][-1])
    itrs = [_take_tensors(tensors, buffer_size) for tensors in dense_tensors]
    # now the dense ones, which have consistent sizes
    for chunks in zip(*itrs, strict=True):
        flat_tensors = [
            _flatten_dense_tensors(chunk) for chunk in chunks
        ]  # (num_gpus,)
        flat_result = reduce_add(flat_tensors, destination)
        for t in _unflatten_dense_tensors(flat_result, chunks[0]):
            # The unflattened tensors do not share storage, and we don't expose
            # base flat tensor anyways, so give them different version counters.
            # See NOTE [ Version Counter in comm.*_coalesced ]
            output.append(t.data)
    return tuple(_reorder_tensors_as(output, ref_order))


def scatter(tensor, devices=None, chunk_sizes=None, dim=0, streams=None, *, out=None):
    """Scatters tensor across multiple GPUs.

    Args:
        tensor (Tensor): tensor to scatter. Can be on CPU or GPU.
        devices (Iterable[torch.device, str or int], optional): an iterable of
          GPU devices, among which to scatter.
        chunk_sizes (Iterable[int], optional): sizes of chunks to be placed on
          each device. It should match :attr:`devices` in length and sums to
          ``tensor.size(dim)``. If not specified, :attr:`tensor` will be divided
          into equal chunks.
        dim (int, optional): A dimension along which to chunk :attr:`tensor`.
          Default: ``0``.
        streams (Iterable[torch.cuda.Stream], optional): an iterable of Streams, among
          which to execute the scatter. If not specified, the default stream will
          be utilized.
        out (Sequence[Tensor], optional, keyword-only): the GPU tensors to
          store output results. Sizes of these tensors must match that of
          :attr:`tensor`, except for :attr:`dim`, where the total size must
          sum to ``tensor.size(dim)``.

    .. note::
        Exactly one of :attr:`devices` and :attr:`out` must be specified. When
        :attr:`out` is specified, :attr:`chunk_sizes` must not be specified and
        will be inferred from sizes of :attr:`out`.

    Returns:
        - If :attr:`devices` is specified,
            a tuple containing chunks of :attr:`tensor`, placed on
            :attr:`devices`.
        - If :attr:`out` is specified,
            a tuple containing :attr:`out` tensors, each containing a chunk of
            :attr:`tensor`.
    """
    tensor = _handle_complex(tensor)
    if out is None:
        # pyrefly: ignore [not-iterable]
        devices = [_get_device_index(d) for d in devices]
        return tuple(torch._C._scatter(tensor, devices, chunk_sizes, dim, streams))
    else:
        if devices is not None:
            raise RuntimeError(
                f"'devices' must not be specified when 'out' is specified, but got devices={devices}"
            )
        if chunk_sizes is not None:
            raise RuntimeError(
                f"'chunk_sizes' must not be specified when 'out' is specified, but got chunk_sizes={chunk_sizes}"
            )
        return tuple(torch._C._scatter_out(tensor, out, dim, streams))


def gather(tensors, dim=0, destination=None, *, out=None):
    r"""Gathers tensors from multiple GPU devices.

    Args:
        tensors (Iterable[Tensor]): an iterable of tensors to gather.
          Tensor sizes in all dimensions other than :attr:`dim` have to match.
        dim (int, optional): a dimension along which the tensors will be
          concatenated. Default: ``0``.
        destination (torch.device, str, or int, optional): the output device.
          Can be CPU or CUDA. Default: the current CUDA device.
        out (Tensor, optional, keyword-only): the tensor to store gather result.
          Its sizes must match those of :attr:`tensors`, except for :attr:`dim`,
          where the size must equal ``sum(tensor.size(dim) for tensor in tensors)``.
          Can be on CPU or CUDA.

    .. note::
        :attr:`destination` must not be specified when :attr:`out` is specified.

    Returns:
        - If :attr:`destination` is specified,
            a tensor located on :attr:`destination` device, that is a result of
            concatenating :attr:`tensors` along :attr:`dim`.
        - If :attr:`out` is specified,
            the :attr:`out` tensor, now containing results of concatenating
            :attr:`tensors` along :attr:`dim`.
    """
    tensors = [_handle_complex(t) for t in tensors]
    if out is None:
        if destination == -1:
            warnings.warn(
                "Using -1 to represent CPU tensor is deprecated. Please use a "
                'device object or string instead, e.g., "cpu".',
                FutureWarning,
                stacklevel=2,
            )
        destination = _get_device_index(destination, allow_cpu=True, optional=True)
        return torch._C._gather(tensors, dim, destination)
    else:
        if destination is not None:
            raise RuntimeError(
                f"'destination' must not be specified when 'out' is specified, but got destination={destination}"
            )
        return torch._C._gather_out(tensors, out, dim)
<|endoftext|>r"""QAT Modules.

This package is in the process of being deprecated.
Please, use `torch.ao.nn.qat.modules` instead.
"""

from torch.ao.nn.qat.modules.conv import Conv1d, Conv2d, Conv3d
from torch.ao.nn.qat.modules.embedding_ops import Embedding, EmbeddingBag
from torch.ao.nn.qat.modules.linear import Linear
from torch.nn.qat.modules import conv, embedding_ops, linear


__all__ = [
    "Linear",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "Embedding",
    "EmbeddingBag",
]
<|endoftext|>"""
In NUMA (Non-Uniform Memory Access) systems, accessing memory on remote NUMA
nodes incurs additional latency. PyTorch provides NUMA binding utilities to
promote memory locality by binding worker processes to CPUs near their assigned accelerator devices.

In practice, NUMA binding typically results in 1-10% overall performance improvements,
but some workloads may obtain much greater benefits or none at all.

To enable NUMA binding, use the ``--numa-binding`` flag with :ref:`torchrun <launcher-api>`, e.g.:

.. code-block:: bash

    torchrun --numa-binding=node --nproc_per_node=8 train.py

Alternatively, pass :class:`NumaOptions` to ``LaunchConfig``
when using ``elastic_launch``.

See :class:`AffinityMode` for available binding modes.
"""

import os
import shutil
import traceback
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from enum import Enum
from functools import wraps
from logging import getLogger
from typing import ParamSpec, TypeVar

import torch
from torch._utils_internal import signpost_event


__all__ = [
    "AffinityMode",
    "NumaOptions",
]

logger = getLogger(__name__)


class AffinityMode(str, Enum):
    NODE = "node"
    """
    Each worker process and its threads will be bound to all the CPUs
    on the NUMA node containing the accelerator device whose local index equals the worker's
    local rank. If in doubt, use this option rather than the others.

    **Ex.:** If device 3 lives on NUMA node 1, then the worker whose local rank is 3 will
    only be able to run on the CPUs of NUMA node 1.
    """

    SOCKET = "socket"
    """
    Each worker process and its threads will be bound to all the CPUs on all the NUMA nodes of the
    socket containing the accelerator device whose local index equals the worker's local rank.

    **Ex.:** If socket 0 contains device 3 and NUMA nodes 0-1, then the worker whose
    local rank is 3 will be bound to the CPUs of NUMA nodes 0-1.

    For cases where there is only one NUMA node per socket anyway, this is equivalent to NODE.
    """

    EXCLUSIVE = "exclusive"
    """
    Each worker process and its threads will be bound to an exclusive subset of CPUs
    on the NUMA node containing the accelerator device whose local index equals the worker's
    local rank. The CPUs on the NUMA node are divided evenly among all devices on that node,
    so no two workers share the same CPU cores.

    **Ex.:** If NUMA node 1 has 16 physical cores and devices 2 and 3, then the worker whose
    local rank is 2 will be bound to cores 0-7, and the worker whose local rank is 3 will
    be bound to cores 8-15.
    """

    CORE_COMPLEX = "core-complex"
    """
    Each worker process and its threads will be bound to a single core complex (a group of cores
    sharing the same L3 cache) on the NUMA node containing the accelerator device whose local
    index equals the worker's local rank. Each worker is bound to a different core complex when
    possible.

    **Ex.:** If NUMA node 1 has two core complexes (cores 0-7 sharing one L3 cache, cores 8-15
    sharing another) and devices 2 and 3, then the worker whose local rank is 2 will be bound to
    cores 0-7, and the worker whose local rank is 3 will be bound to cores 8-15.
    """


@dataclass(frozen=True)
class NumaOptions:
    affinity_mode: AffinityMode

    should_fall_back_if_binding_fails: bool = False
    """
    If ``True``, we will silence any exceptions that occur during NUMA binding itself
    rather than raising them.

    There are no expected exceptions, so avoid using this option. Its purpose is simply
    to mitigate crash risk while conducting mass rollouts of NUMA binding.
    """


def _maybe_wrap_command_args_with_numa_binding(
    command_args: tuple[str, ...],
    *,
    device_index: int,
    numa_options: NumaOptions | None,
) -> tuple[str, ...]:
    """
    Wraps command arguments with numactl to apply NUMA CPU binding.

    This function prepends numactl with appropriate CPU affinity flags to the
    provided command arguments, binding the process to CPUs associated with
    the specified accelerator device's NUMA node.

    Args:
        command_args: The original command arguments to wrap.
        device_index: The index of the accelerator device that will be used by the subprocess.
        numa_options: Configuration for NUMA binding behavior. If None, returns
            the original command_args unchanged.

    Returns:
        Tuple of command arguments, potentially wrapped with numactl for NUMA binding.
        Returns the original command_args if numa_options is None or if binding fails
        and fallback is enabled.
    """
    if numa_options is None:
        return command_args

    kwargs = {
        "command_args": command_args,
        "device_index": device_index,
        "numa_options": asdict(numa_options),
    }

    try:
        logical_cpu_indices = _get_validated_logical_cpus_to_bind_to(
            device_index=device_index,
            numa_options=numa_options,
        )

        wrapped_command_args = _assemble_numactl_command_args(
            original_command_args=command_args,
            logical_cpu_indices=logical_cpu_indices,
        )
        signpost_event(
            category="numa_binding",
            name="apply_success",
            parameters={
                **kwargs,
                "wrapped_command": wrapped_command_args,
            },
        )
        return wrapped_command_args
    except Exception:
        # pyrefly: ignore [bad-argument-type]
        _handle_exception(numa_options=numa_options, logger_kwargs=kwargs)
        return command_args


_TParams = ParamSpec("_TParams")
_TReturn = TypeVar("_TReturn")


def _maybe_wrap_with_numa_binding(
    func: Callable[_TParams, _TReturn],
    *,
    device_index: int,
    numa_options: NumaOptions | None,
) -> Callable[_TParams, _TReturn]:
    """
    Wraps a function to apply NUMA CPU binding before execution.

    This decorator applies NUMA CPU affinity to all threads in the current process
    before calling the wrapped function, binding them to CPUs associated with the
    specified accelerator device's NUMA node.

    Args:
        func: The function to wrap with NUMA binding.
        device_index: The index of the accelerator device that will be used.
        numa_options: Configuration for NUMA binding behavior. If None, returns
            the original function unchanged.

    Returns:
        A wrapped function that applies NUMA binding before execution, or the
        original function if numa_options is None.
    """
    if numa_options is None:
        return func

    @wraps(func)
    def wrapped(*args: _TParams.args, **kwargs: _TParams.kwargs) -> _TReturn:
        _maybe_apply_numa_binding_to_current_process(
            device_index=device_index,
            # pyrefly: ignore [bad-argument-type]
            numa_options=numa_options,
        )
        return func(*args, **kwargs)

    return wrapped


def _maybe_apply_numa_binding_to_current_process(
    *, device_index: int, numa_options: NumaOptions
) -> None:
    kwargs = {
        "device_index": device_index,
        "numa_options": asdict(numa_options),
    }

    try:
        logical_cpu_indices = _get_validated_logical_cpus_to_bind_to(
            device_index=device_index,
            numa_options=numa_options,
        )

        _bind_all_threads_in_current_process_to_logical_cpus(
            logical_cpu_indices=logical_cpu_indices
        )

        signpost_event(
            category="numa_binding",
            name="apply_success",
            parameters={
                **kwargs,
                "logical_cpu_indices": _get_ranges_str_from_ints(logical_cpu_indices),
            },
        )
    except Exception:
        # pyrefly: ignore [bad-argument-type]
        _handle_exception(numa_options=numa_options, logger_kwargs=kwargs)


def _assemble_numactl_command_args(
    *, original_command_args: tuple[str, ...], logical_cpu_indices: set[int]
) -> tuple[str, ...]:
    return (
        "numactl",
        f"--physcpubind={_get_ranges_str_from_ints(logical_cpu_indices)}",
        *original_command_args,
    )


def _handle_exception(
    *, numa_options: NumaOptions, logger_kwargs: dict[str, object]
) -> None:
    signpost_event(
        category="numa_binding",
        name="apply_exception",
        parameters={
            **logger_kwargs,
            "traceback": traceback.format_exc(),
        },
    )
    logger.exception("Failed to apply NUMA binding for input=%r", logger_kwargs)
    if numa_options.should_fall_back_if_binding_fails:
        logger.warning(
            "Continuing executing without applying NUMA binding, despite exception %s",
            traceback.format_exc(),
        )
        return
    # This function is called within an except block, so silence the warning
    # about raise without an exception.
    raise  # noqa: PLE0704


def _get_validated_logical_cpus_to_bind_to(
    *,
    device_index: int,
    numa_options: NumaOptions,
) -> set[int]:
    logical_cpu_indices = _get_logical_cpus_to_bind_to(
        device_index=device_index, numa_options=numa_options
    )
    _raise_if_binding_invalid(logical_cpu_indices=logical_cpu_indices)

    return logical_cpu_indices


def _raise_if_binding_invalid(*, logical_cpu_indices: set[int]) -> None:
    # NOTE: numactl CLI is only actually necessary for the str entrypoint path,
    # but for simplicity we will just check it no matter what.
    if shutil.which("numactl") is None:
        raise RuntimeError("numactl CLI is required for NUMA binding")

    if not logical_cpu_indices:
        raise RuntimeError("Must bind to a non-empty set of CPU indices")


def _bind_all_threads_in_current_process_to_logical_cpus(
    *, logical_cpu_indices: set[int]
) -> None:
    # Save the original affinity of the main thread before changing it
    # pyrefly: ignore [missing-attribute]
    original_main_thread_affinity = os.sched_getaffinity(0)  # type: ignore[attr-defined]

    # 0 represents the current thread.
    # This is outside the try/except because the main thread should always bind successfully.
    # pyrefly: ignore [missing-attribute]
    os.sched_setaffinity(0, logical_cpu_indices)  # type: ignore[attr-defined]

    for tid_str in os.listdir("/proc/self/task"):
        try:
            tid = int(tid_str)
            # pyrefly: ignore [missing-attribute]
            tid_affinity = os.sched_getaffinity(tid)  # type: ignore[attr-defined]

            # Defensive check to ensure we do not overwrite affinity on any threads
            # that have already had their affinity set elsewhere.
            if tid_affinity == original_main_thread_affinity:
                # pyrefly: ignore [missing-attribute]
                os.sched_setaffinity(tid, logical_cpu_indices)  # type: ignore[attr-defined]
        except Exception:
            # Thread may have exited or otherwise become invalid
            pass


def _get_logical_cpus_to_bind_to(
    *,
    device_index: int,
    numa_options: NumaOptions,
) -> set[int]:
    """
    Args:
        device_index: The index of the accelerator device that will be used by the subprocess.
            Example: 0
        numa_options: See NumaOptions for details.

    Returns:
        Set of logical CPU indices to bind to.
    """
    if numa_options.affinity_mode == AffinityMode.NODE:
        logical_cpus = _node_get_logical_cpus_to_bind_to(device_index=device_index)
    elif numa_options.affinity_mode == AffinityMode.SOCKET:
        logical_cpus = _socket_get_logical_cpus_to_bind_to(device_index=device_index)
    elif numa_options.affinity_mode == AffinityMode.EXCLUSIVE:
        logical_cpus = _exclusive_get_logical_cpus_to_bind_to(device_index=device_index)
    elif numa_options.affinity_mode == AffinityMode.CORE_COMPLEX:
        logical_cpus = _core_complex_get_logical_cpus_to_bind_to(
            device_index=device_index
        )
    else:
        raise ValueError(f"Affinity mode {numa_options.affinity_mode} not supported.")

    return logical_cpus


def _node_get_logical_cpus_to_bind_to(*, device_index: int) -> set[int]:
    """
    Core logic of 'node' numa strategy.
    """
    numa_node_index = _get_numa_node_index_for_device_index(device_index=device_index)

    return _get_allowed_logical_cpu_indices_for_numa_node(
        numa_node_index=numa_node_index
    )


def _socket_get_logical_cpus_to_bind_to(*, device_index: int) -> set[int]:
    """
    Core logic of 'socket' numa strategy.
    """
    numa_node_index_of_device = _get_numa_node_index_for_device_index(
        device_index=device_index
    )
    socket_index = _get_socket_index_for_numa_node(
        numa_node_index=numa_node_index_of_device
    )
    numa_node_indices = _get_numa_node_indices_for_socket_index(
        socket_index=socket_index
    )

    logical_cpus = set()
    for numa_node_index in numa_node_indices:
        logical_cpus.update(
            _get_allowed_logical_cpu_indices_for_numa_node(
                numa_node_index=numa_node_index
            )
        )

    return logical_cpus


def _exclusive_get_logical_cpus_to_bind_to(*, device_index: int) -> set[int]:
    """
    Core logic of 'exclusive' numa strategy.
    """
    numa_node_index = _get_numa_node_index_for_device_index(device_index=device_index)

    dev_indices = _get_device_indices_for_numa_node(numa_node_index=numa_node_index)
    dev_indices = sorted(dev_indices)
    original_device_relative_index = dev_indices.index(device_index)

    allowed_logical_cpu_indices = _get_allowed_logical_cpu_indices_for_numa_node(
        numa_node_index=numa_node_index
    )

    # Arbitrarily use the min logical cpu index on the physical core to
    # represent the physical core.
    physical_core_to_allowed_logical_cpu_indices = _group_by(
        allowed_logical_cpu_indices,
        lambda logical_cpu_index: min(
            _get_logical_cpu_indices_sharing_same_physical_core_as(
                logical_cpu_index=logical_cpu_index
            )
        ),
    )
    # Sort the dict for consistency (dicts maintain order in Python)
    physical_core_to_allowed_logical_cpu_indices = dict(
        sorted(physical_core_to_allowed_logical_cpu_indices.items())
    )

    num_physical_cores_per_device = len(
        physical_core_to_allowed_logical_cpu_indices
    ) // len(dev_indices)
    num_devices_to_give_one_extra_physical_core = len(
        physical_core_to_allowed_logical_cpu_indices
    ) % len(dev_indices)

    if num_physical_cores_per_device < 1:
        raise RuntimeError(
            f"There are only {len(physical_core_to_allowed_logical_cpu_indices)} physical cores on {numa_node_index=},"
            + f" but there are {len(dev_indices)} devices associated with this NUMA node."
        )

    start = original_device_relative_index * num_physical_cores_per_device + min(
        original_device_relative_index, num_devices_to_give_one_extra_physical_core
    )
    end = (
        start
        + num_physical_cores_per_device
        + (
            1
            if original_device_relative_index
            < num_devices_to_give_one_extra_physical_core
            else 0
        )
    )

    # Slice and flatten the logical CPUs from the selected physical cores
    logical_cpu_indices_for_device = {
        logical_cpu_index
        for logical_cpu_indices in list(
            physical_core_to_allowed_logical_cpu_indices.values()
        )[start:end]
        for logical_cpu_index in logical_cpu_indices
    }

    return logical_cpu_indices_for_device


def _core_complex_get_logical_cpus_to_bind_to(*, device_index: int) -> set[int]:
    """
    Core logic of 'core-complex' numa strategy.

    Each accelerator device is assigned a full core complex (group of cores sharing
    L3 cache) within its affined NUMA node.
    """
    numa_node_index = _get_numa_node_index_for_device_index(device_index=device_index)

    dev_indices = _get_device_indices_for_numa_node(numa_node_index=numa_node_index)
    dev_indices = sorted(dev_indices)
    original_device_relative_index = dev_indices.index(device_index)

    allowed_logical_cpu_indices = _get_allowed_logical_cpu_indices_for_numa_node(
        numa_node_index=numa_node_index
    )

    # Arbitrarily use the min logical cpu index on the max level cache
    # to represent the max level cache.
    max_level_cache_to_allowed_logical_cpu_indices = _group_by(
        allowed_logical_cpu_indices,
        lambda logical_cpu_index: min(
            _get_logical_cpus_sharing_same_max_level_cache_as(
                logical_cpu_index=logical_cpu_index
            )
        ),
    )

    max_level_cache_to_allowed_logical_cpu_indices = dict(
        sorted(
            max_level_cache_to_allowed_logical_cpu_indices.items(),
            # First, prioritize caches with more available cpus
            # Second, prioritize lower index cpus (just for clarity/consistency)
            key=lambda item: (-len(item[1]), item[0]),
        )
    )

    cache_index_for_device = original_device_relative_index % len(
        max_level_cache_to_allowed_logical_cpu_indices
    )
    logical_cpu_indices_for_device = list(
        max_level_cache_to_allowed_logical_cpu_indices.values()
    )[cache_index_for_device]

    return logical_cpu_indices_for_device


K = TypeVar("K")
V = TypeVar("V")


def _group_by(values: Iterable[V], get_key: Callable[[V], K]) -> dict[K, set[V]]:
    """
    Groups elements with same key into sets.
    """
    key_to_values: defaultdict[K, set[V]] = defaultdict(set)
    for value in values:
        key = get_key(value)
        key_to_values[key].add(value)
    return key_to_values


def _get_logical_cpu_indices_sharing_same_physical_core_as(
    *, logical_cpu_index: int
) -> set[int]:
    thread_siblings_list_absolute_path = (
        f"/sys/devices/system/cpu/cpu{logical_cpu_index}/topology/thread_siblings_list"
    )
    with open(thread_siblings_list_absolute_path) as f:
        return _get_set_of_int_from_ranges_str(f.read())


def _get_logical_cpus_sharing_same_max_level_cache_as(
    *, logical_cpu_index: int
) -> set[int]:
    cpu_cache_dir_absolute_path = (
        f"/sys/devices/system/cpu/cpu{logical_cpu_index}/cache"
    )

    max_level = -1
    logical_cpus_sharing_max_level_cache = set()
    for entry in os.listdir(cpu_cache_dir_absolute_path):
        if not entry.startswith("index") or not entry[5:].isdecimal():
            continue
        cache_index_absolute_path = os.path.join(cpu_cache_dir_absolute_path, entry)

        # Filter out other cache types like Instruction
        type_absolute_path = os.path.join(cache_index_absolute_path, "type")
        with open(type_absolute_path) as type_file:
            if type_file.read().strip() not in {"Unified", "Data"}:
                continue

        level_absolute_path = os.path.join(cache_index_absolute_path, "level")
        with open(level_absolute_path) as level_file:
            level = int(level_file.read())
        if level <= max_level:
            continue

        max_level = level
        shared_cpu_list_absolute_path = os.path.join(
            cache_index_absolute_path, "shared_cpu_list"
        )
        with open(shared_cpu_list_absolute_path) as share_cpu_list_file:
            logical_cpus_sharing_max_level_cache = _get_set_of_int_from_ranges_str(
                share_cpu_list_file.read()
            )

    return logical_cpus_sharing_max_level_cache


def _get_allowed_logical_cpu_indices_for_numa_node(*, numa_node_index: int) -> set[int]:
    all_cpu_indices = _get_cpu_indices_for_numa_node_MAYBE_NOT_ALLOWED(
        numa_node_index=numa_node_index
    )
    allowed_cpu_indices = _get_allowed_cpu_indices_for_current_thread()
    return all_cpu_indices & allowed_cpu_indices


def _get_cpu_indices_for_numa_node_MAYBE_NOT_ALLOWED(
    *, numa_node_index: int
) -> set[int]:
    """
    Returns:
        Indices of all CPUs associated with numa_node_index. However, the list
        is not filtered based on whether the thread is allowed to use them.
    """
    cpulist_absolute_path = f"/sys/devices/system/node/node{numa_node_index}/cpulist"
    try:
        with open(cpulist_absolute_path) as f:
            cpu_range_str = f.read()
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Could not determine CPUs corresponding to {numa_node_index=}."
        ) from e
    return _get_set_of_int_from_ranges_str(cpu_range_str)


def _get_device_count() -> int:
    return torch.accelerator.device_count()


def _get_numa_node_index_for_device_index(*, device_index: int) -> int:
    if not torch.accelerator.is_available():
        raise RuntimeError("No accelerator available for NUMA binding")
    acc = torch.accelerator.current_accelerator()
    device_module = torch.get_device_module(acc)
    device_properties = device_module.get_device_properties(device_index)

    domain = device_properties.pci_domain_id  # type: ignore[attr-defined]
    bus = device_properties.pci_bus_id  # type: ignore[attr-defined]
    device = device_properties.pci_device_id  # type: ignore[attr-defined]

    # Format to sysfs PCI address: "0000:dc:00.0"
    pci_addr = f"{domain:04x}:{bus:02x}:{device:02x}.0"

    pci_numa_node_absolute_path = f"/sys/bus/pci/devices/{pci_addr}/numa_node"
    with open(pci_numa_node_absolute_path) as f:
        # In systems with only one NUMA node, this will
        # often be saved as -1. In those cases, there is obviously
        # at least one numa node, 0, so we use that.
        return max(int(f.read().strip()), 0)


def _get_device_indices_for_numa_node(*, numa_node_index: int) -> set[int]:
    return {
        device_index
        for device_index in range(_get_device_count())
        if _get_numa_node_index_for_device_index(device_index=device_index)
        == numa_node_index
    }


def _get_socket_index_for_numa_node(*, numa_node_index: int) -> int:
    arbitrary_cpu_index = _get_arbitrary_allowed_cpu_index_for_numa_node(
        numa_node_index=numa_node_index
    )

    return _get_socket_index_for_cpu(cpu_index=arbitrary_cpu_index)


def _get_socket_index_for_cpu(*, cpu_index: int) -> int:
    package_id_absolute_path = (
        f"/sys/devices/system/cpu/cpu{cpu_index}/topology/physical_package_id"
    )
    try:
        with open(package_id_absolute_path) as f:
            return int(f.read().strip())
    except FileNotFoundError as e:
        raise RuntimeError(f"Could not determine socket for {cpu_index=}") from e


def _get_arbitrary_allowed_cpu_index_for_numa_node(*, numa_node_index: int) -> int:
    return min(
        _get_allowed_logical_cpu_indices_for_numa_node(numa_node_index=numa_node_index)
    )


def _get_set_of_int_from_ranges_str(ranges_str: str) -> set[int]:
    """
    Util for parsing a string of int ranges, as in a sysfs file.

    Args:
        ranges_str: E.g., "0-2,4,6-7"

    Returns:
        E.g., {0, 1, 2, 4, 6, 7}
    """
    ints: set[int] = set()
    for range_str in ranges_str.split(","):
        range_str = range_str.strip()
        if not range_str:
            continue
        if "-" in range_str:
            start_str, end_str = range_str.split("-")
            start, end = int(start_str), int(end_str)
            ints.update(range(start, end + 1))
        else:
            ints.add(int(range_str))
    return ints


def _get_ranges_str_from_ints(ints: Iterable[int]) -> str:
    """
    Convert a set of integers to a compact string with ranges.

    Args:
        ints: E.g., {0, 1, 2, 4, 6, 7}

    Returns:
        E.g., "0-2,4,6-7"
    """
    if not ints:
        return ""

    sorted_ints = sorted(ints)
    ranges = []
    start = prev = sorted_ints[0]

    for num in sorted_ints[1:]:
        if num == prev + 1:
            prev = num
        else:
            if start == prev:
                ranges.append(f"{start}")
            else:
                ranges.append(f"{start}-{prev}")
            start = prev = num

    # Append the last range
    if start == prev:
        ranges.append(f"{start}")
    else:
        ranges.append(f"{start}-{prev}")

    return ",".join(ranges)


def _get_systemwide_numa_node_indices() -> set[int]:
    with open("/sys/devices/system/node/possible") as f:
        possible_nodes_str = f.read()

    return _get_set_of_int_from_ranges_str(possible_nodes_str)


def _get_numa_node_indices_for_socket_index(*, socket_index: int) -> set[int]:
    systemwide_numa_node_indices = _get_systemwide_numa_node_indices()

    matching_numa_node_indices = set()
    for numa_node_index in systemwide_numa_node_indices:
        arbitrary_cpu_index = _get_arbitrary_allowed_cpu_index_for_numa_node(
            numa_node_index=numa_node_index
        )
        if socket_index == _get_socket_index_for_cpu(cpu_index=arbitrary_cpu_index):
            matching_numa_node_indices.add(numa_node_index)

    return matching_numa_node_indices


def _get_allowed_cpu_indices_for_current_thread() -> set[int]:
    # 0 denotes current thread
    # pyrefly: ignore [missing-attribute]
    return os.sched_getaffinity(0)  # type:ignore[attr-defined]
<|endoftext|># mypy: allow-untyped-defs
import collections
import inspect
import logging
import warnings
import weakref
from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from typing import Any, overload, TypeAlias, Union

import torch
from torch import _C, _ops, Tensor
from torch.types import _dtype
from torch.utils._exposed_in import exposed_in

from . import autograd, utils
from .effects import EffectType


device_types_t: TypeAlias = str | Sequence[str] | None  # noqa: PYI042
tags_t: TypeAlias = _C.Tag | Sequence[_C.Tag] | None  # noqa: PYI042
log = logging.getLogger(__name__)


def _normalize_tags(tags: tags_t) -> tuple[_C.Tag, ...]:
    if tags is None:
        return ()
    if isinstance(tags, _C.Tag):
        tags = (tags,)
    return tuple(dict.fromkeys(tags))


def _with_pt2_compliant_tag(tags: Sequence[_C.Tag]) -> list[_C.Tag]:
    return [
        _C.Tag.pt2_compliant_tag,
        *(tag for tag in tags if tag != _C.Tag.pt2_compliant_tag),
    ]


@overload
def custom_op(
    name: str,
    fn: None = None,
    /,
    *,
    mutates_args: str | Iterable[str],
    device_types: device_types_t = None,
    schema: str | None = None,
    tags: tags_t = None,
) -> Callable[[Callable[..., object]], "CustomOpDef"]: ...


@overload
def custom_op(
    name: str,
    fn: Callable[..., object],
    /,
    *,
    mutates_args: str | Iterable[str],
    device_types: device_types_t = None,
    schema: str | None = None,
    tags: tags_t = None,
) -> "CustomOpDef": ...


@exposed_in("torch.library")
def custom_op(
    name: str,
    fn: Callable | None = None,
    /,
    *,
    mutates_args: str | Iterable[str],
    device_types: device_types_t = None,
    schema: str | None = None,
    tags: tags_t = None,
) -> Union[Callable[[Callable[..., object]], "CustomOpDef"], "CustomOpDef"]:
    """Wraps a function into custom operator.

    Reasons why you may want to create a custom op include:
    - Wrapping a third-party library or custom kernel to work with PyTorch
    subsystems like Autograd.
    - Preventing torch.compile/export/FX tracing from peeking inside your function.

    This API is used as a decorator around a function (please see examples).
    The provided function must have type hints; these are needed to interface
    with PyTorch's various subsystems.

    Args:
        name (str): A name for the custom op that looks like "{namespace}::{name}",
            e.g. "mylib::my_linear". The name is used as the op's stable identifier
            in PyTorch subsystems (e.g. torch.export, FX graphs).
            To avoid name collisions, please use your project name as the namespace;
            e.g. all custom ops in pytorch/fbgemm use "fbgemm" as the namespace.
        mutates_args (Iterable[str] or "unknown"): The names of args that the function mutates.
            This MUST be accurate, otherwise, the behavior is undefined. If "unknown",
            it pessimistically assumes that all inputs to the operator are being mutated.
        device_types (str | None | Sequence[str]): The device type(s) the function
            is valid for. If no device type is provided, then the function
            is used as the default implementation for all device types.
            Examples: "cpu", "cuda".
            When registering a device-specific implementation for an operator that accepts no Tensors,
            we require the operator to have a "device: torch.device argument".
        schema (str | None): A schema string for the operator. If None
            (recommended) we'll infer a schema for the operator from its type
            annotations. We recommend letting us infer a schema unless you
            have a specific reason not to.
            Example: "(Tensor x, int y) -> (Tensor, Tensor)".
        tags (Tag | Sequence[Tag] | None): one or more tags to apply to the
            operator. Use ``torch.Tag.inplace`` for operators that mutate their
            first Tensor argument and return it. Use ``torch.Tag.out`` for
            operators that mutate keyword-only output tensors and return them.
            Like PyTorch's built-in ``out=`` operators, ``torch.Tag.out``
            custom ops do not support autograd.

    The following types are supported for the wrapped function's input parameters:

        - Scalars: ``int``, ``float``, ``bool``, ``str``, ``torch.types.Number``
        - Tensors: ``torch.Tensor``
        - Enums/devices: ``torch.dtype``, ``torch.device``
        - Flat list of the same type: ``list[torch.Tensor]``,
          ``list[int]``, ``list[float]``, ``list[bool]``,
          ``list[torch.types.Number]``
        - Optionals: ``Optional`` of any of the above scalar/tensor types
        - Types registered via :func:`torch.library.register_custom_class`

    The following types are supported for the return value:

        ``torch.Tensor``, ``list[torch.Tensor]``, ``int``, ``float``,
        ``bool``, ``torch.types.Number``.

    .. note::
        We recommend not passing in a ``schema`` arg and instead letting us infer
        it from the type annotations. It is error-prone to write your own schema.
        You may wish to provide your own schema if our interpretation of
        the type annotation is not what you want.
        For more info on how to write a schema string, see
        `here <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#func>`_

    Examples::
        >>> import torch
        >>> from torch import Tensor
        >>> from torch.library import custom_op
        >>> import numpy as np
        >>>
        >>> @custom_op("mylib::numpy_sin", mutates_args=())
        >>> def numpy_sin(x: Tensor) -> Tensor:
        >>>     x_np = x.cpu().numpy()
        >>>     y_np = np.sin(x_np)
        >>>     return torch.from_numpy(y_np).to(device=x.device)
        >>>
        >>> x = torch.randn(3)
        >>> y = numpy_sin(x)
        >>> assert torch.allclose(y, x.sin())
        >>>
        >>> # Example of a custom op that only works for one device type.
        >>> @custom_op("mylib::numpy_sin_cpu", mutates_args=(), device_types="cpu")
        >>> def numpy_sin_cpu(x: Tensor) -> Tensor:
        >>>     x_np = x.numpy()
        >>>     y_np = np.sin(x_np)
        >>>     return torch.from_numpy(y_np)
        >>>
        >>> x = torch.randn(3)
        >>> y = numpy_sin_cpu(x)
        >>> assert torch.allclose(y, x.sin())
        >>>
        >>> # Example of a custom op that mutates an input
        >>> @custom_op("mylib::numpy_sin_inplace", mutates_args={"x"}, device_types="cpu")
        >>> def numpy_sin_inplace(x: Tensor) -> None:
        >>>     x_np = x.numpy()
        >>>     np.sin(x_np, out=x_np)
        >>>
        >>> x = torch.randn(3)
        >>> expected = x.sin()
        >>> numpy_sin_inplace(x)
        >>> assert torch.allclose(x, expected)
        >>>
        >>> # Example of a custom op with inplace semantics
        >>> @custom_op(
        >>>     "mylib::numpy_sin_",
        >>>     mutates_args={"x"},
        >>>     device_types="cpu",
        >>>     tags=torch.Tag.inplace,
        >>> )
        >>> def numpy_sin_(x: Tensor) -> Tensor:
        >>>     x_np = x.numpy()
        >>>     np.sin(x_np, out=x_np)
        >>>     return x
        >>>
        >>> x = torch.randn(3)
        >>> expected = x.sin()
        >>> result = numpy_sin_(x)
        >>> assert result is x
        >>> assert torch.allclose(x, expected)
        >>>
        >>> # Example of a custom op with out= semantics
        >>> @custom_op(
        >>>     "mylib::numpy_sin_out",
        >>>     mutates_args={"out"},
        >>>     device_types="cpu",
        >>>     tags=torch.Tag.out,
        >>> )
        >>> def numpy_sin_out(x: Tensor, *, out: Tensor) -> Tensor:
        >>>     x_np = x.numpy()
        >>>     out_np = out.numpy()
        >>>     np.sin(x_np, out=out_np)
        >>>     return out
        >>>
        >>> x = torch.randn(3)
        >>> out = torch.empty_like(x)
        >>> result = numpy_sin_out(x, out=out)
        >>> assert result is out
        >>> assert torch.allclose(out, x.sin())
        >>>
        >>> # Example of a factory function
        >>> @torch.library.custom_op("mylib::bar", mutates_args={}, device_types="cpu")
        >>> def bar(device: torch.device) -> Tensor:
        >>>     return torch.ones(3)
        >>>
        >>> bar("cpu")
        >>>
        >>> # Example of a custom op with list inputs
        >>> @custom_op("mylib::weighted_sum", mutates_args=())
        >>> def weighted_sum(
        >>>     tensors: list[Tensor],
        >>>     weights: list[float],
        >>> ) -> Tensor:
        >>>     return sum(t * w for t, w in zip(tensors, weights))
        >>>
        >>> x = torch.randn(3)
        >>> y = torch.randn(3)
        >>> out = weighted_sum([x, y], [0.3, 0.7])

    """

    def inner(fn: Callable[..., object]) -> CustomOpDef:
        import torch

        normalized_tags = _normalize_tags(tags)
        if schema is None:
            schema_str = torch.library.infer_schema(
                fn,
                mutates_args=mutates_args,
                tags=normalized_tags,
            )
        else:
            schema_str = schema

        namespace, opname = name.split("::")
        result = CustomOpDef(namespace, opname, schema_str, fn, normalized_tags)
        if schema is not None:
            # Check that schema's alias annotations match those of `mutates_args`.
            expected = set()
            for arg in result._opoverload._schema.arguments:
                if arg.alias_info is not None and arg.alias_info.is_write:
                    expected.add(arg.name)
            if expected != set(mutates_args):
                raise ValueError(
                    f"Attempted to create a custom op with `mutates_args={mutates_args}` "
                    f"and `schema={schema}. The schema suggests that the op mutates {expected}"
                    f"which is different from what was provided to us in `mutates_args`. "
                    f"Please make these consistent."
                )
        result.register_kernel(device_types)(fn)
        result._opoverload._enable_pyobj_dispatch(True)
        return result

    if fn is None:
        return inner
    return inner(fn)


class CustomOpDef:
    """CustomOpDef is a wrapper around a function that turns it into a custom op.

    It has various methods for registering additional behavior for this
    custom op.

    You should not instantiate CustomOpDef directly; instead, use the
    :func:`torch.library.custom_op` API.
    """

    def __init__(
        self,
        namespace: str,
        name: str,
        schema: str,
        fn: Callable,
        tags: tags_t = None,
    ) -> None:
        # Fields used to interface with the PyTorch dispatcher
        self._namespace = namespace
        self._name = name
        self._schema = schema
        self._tags = _normalize_tags(tags)

        self._init_fn = fn

        self._backend_fns: dict[str | None, Callable] = {}
        self._abstract_fn: Callable | None = None
        self._setup_context_fn: Callable | None = None
        self._backward_fn: Callable | None = None
        self._torch_dispatch_fns: dict[type, Callable] = {}
        self._vmap_fn: Callable | None = None
        self._autocast_dtype: dict[str, _dtype | None] = {}
        self._is_inplace = False
        self._is_out = False
        self._out_kwarg_names: tuple[str, ...] = ()

        self._lib = get_library_allowing_overwrite(self._namespace, self._name)
        self._register_to_dispatcher(self._tags)
        self._disabled_kernel: set = set()
        self._used_triton_kernels: list[Any] = list()
        OPDEFS[self._qualname] = self

    @property
    def _qualname(self) -> str:
        return f"{self._namespace}::{self._name}"

    def __repr__(self) -> str:
        return f"<CustomOpDef({self._qualname})>"

    @contextmanager
    def set_kernel_enabled(self, device_type: str, enabled: bool = True):
        """
        Disable or re-enable an already registered kernel for this custom operator.

        If the kernel is already disabled/enabled, this is a no-op.

        Note:
            If a kernel is first disabled and then registered, it is disabled until enabled again.

        Args:
            device_type (str): The device type to disable/enable the kernel for.
            disable (bool): Whether to disable or enable the kernel.

        Example:
            >>> inp = torch.randn(1)
            >>>
            >>> # define custom op `f`.
            >>> @custom_op("mylib::f", mutates_args=())
            >>> def f(x: Tensor) -> Tensor:
            >>>     return torch.zeros(1)
            >>>
            >>> print(f(inp))  # tensor([0.]), default kernel
            >>>
            >>> @f.register_kernel("cpu")
            >>> def _(x):
            >>>     return torch.ones(1)
            >>>
            >>> print(f(inp))  # tensor([1.]), CPU kernel
            >>>
            >>> # temporarily disable the CPU kernel
            >>> with f.set_kernel_enabled("cpu", enabled = False):
            >>>     print(f(inp))  # tensor([0.]) with CPU kernel disabled

        """
        action = "enable" if enabled else "disable"
        originally_disabled = device_type in self._disabled_kernel
        if device_type not in self._backend_fns:
            log.warning(
                "Attempted to %s kernel for %s but no kernel was registered for this device type.",
                action,
                device_type,
            )

        if not enabled:
            if originally_disabled:
                log.warning(
                    "Attempted to disable kernel for %s but it was already disabled.",
                    device_type,
                )
            else:
                self._disabled_kernel.add(device_type)
        else:  # enable the kernel
            if not originally_disabled:
                log.warning(
                    "Attempted to enable kernel for  %s but it was already enabled.",
                    device_type,
                )
            else:
                self._disabled_kernel.remove(device_type)

        try:
            yield
        finally:
            # restore original state
            if originally_disabled:
                self._disabled_kernel.add(device_type)
            else:
                self._disabled_kernel.discard(device_type)

    def register_kernel(
        self, device_types: device_types_t, fn: Callable | None = None, /
    ) -> Callable:
        """Register an implementation for a device type for this operator.

        Some valid device_types are: "cpu", "cuda", "xla", "mps", "ipu", "xpu".
        This API may be used as a decorator.

        Args:
            fn (Callable): The function to register as the implementation for
                the given device types.
            device_types (str | Sequence[str]): The device device_types to register an impl to.

        Examples::
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
            >>> import torch
            >>> from torch import Tensor
            >>> from torch.library import custom_op
            >>> import numpy as np
            >>>
            >>> # Create a custom op that works on cpu
            >>> @custom_op("mylib::numpy_sin", mutates_args=(), device_types="cpu")
            >>> def numpy_sin(x: Tensor) -> Tensor:
            >>>     x_np = x.numpy()
            >>>     y_np = np.sin(x_np)
            >>>     return torch.from_numpy(y_np)
            >>>
            >>> # Add implementations for the cuda device
            >>> @numpy_sin.register_kernel("cuda")
            >>> def _(x):
            >>>     x_np = x.cpu().numpy()
            >>>     y_np = np.sin(x_np)
            >>>     return torch.from_numpy(y_np).to(device=x.device)
            >>>
            >>> x_cpu = torch.randn(3)
            >>> x_cuda = x_cpu.cuda()
            >>> assert torch.allclose(numpy_sin(x_cpu), x_cpu.sin())
            >>> assert torch.allclose(numpy_sin(x_cuda), x_cuda.sin())

        """

        def inner(fn):
            if device_types is None or isinstance(device_types, str):
                dtypes: list[str | None] = [device_types]
            else:
                dtypes = list(device_types)
            for device_type in dtypes:
                if device_type not in self._backend_fns:

                    def backend_impl(*args, **kwargs):
                        result = self._backend_fns[device_type](*args, **kwargs)

                        def get_module():
                            fn = self._backend_fns[device_type]
                            return inspect.getmodule(fn)

                        schema = self._opoverload._schema
                        if self._is_inplace:
                            if result is not args[0]:
                                raise RuntimeError(
                                    f"{self._name} (with implementation in {get_module()}): "
                                    "An operator tagged with torch.Tag.inplace must "
                                    "return its first argument."
                                )
                        elif self._is_out:
                            out_kwarg_names = self._out_kwarg_names
                            if len(out_kwarg_names) == 1:
                                returns_out_args = result is kwargs[out_kwarg_names[0]]
                            else:
                                returns_out_args = (
                                    isinstance(result, tuple)
                                    and len(result) == len(out_kwarg_names)
                                    and all(
                                        result[i] is kwargs[name]
                                        for i, name in enumerate(out_kwarg_names)
                                    )
                                )
                            if not returns_out_args:
                                raise RuntimeError(
                                    f"{self._name} (with implementation in {get_module()}): "
                                    "An operator tagged with torch.Tag.out must "
                                    "return its mutable keyword-only arguments."
                                )
                        elif not schema._is_view_op():
                            utils._c_check_aliasing_constraint(
                                self._name,
                                args,
                                kwargs,
                                result,
                                get_module,
                            )
                        return result

                    if device_type is None:
                        self._lib.impl(
                            self._name, backend_impl, "CompositeExplicitAutograd"
                        )
                    else:
                        self._lib.impl(
                            self._name,
                            backend_impl,
                            _C._dispatch_key_for_device(device_type),
                        )

                # Wrap function to choose between the default implementation or the device-specific
                # implementation depending on if the kernel is disabled.
                @torch._disable_dynamo
                def wrapped_fn(*args, **kwargs):
                    if device_type in self._disabled_kernel:
                        return self._init_fn(*args, **kwargs)
                    else:
                        return fn(*args, **kwargs)

                self._backend_fns[device_type] = wrapped_fn
            return fn

        if device_types is not None and not utils.has_tensor_arg(
            self._opoverload._schema
        ):
            device_arg_index = utils.get_device_arg_index(self._opoverload._schema)
            if device_arg_index is None:
                raise ValueError(
                    "Functions without tensor inputs are required to have a `device: torch.device` argument"
                )
            self._register_backend_select_dispatcher(device_arg_index)

        # See NOTE: [Supporting decorator and non-decorator usage]
        if fn is None:
            return inner
        return inner(fn)

    def register_fake(self, fn: Callable, /) -> Callable:
        r"""Register a FakeTensor implementation for this custom op.

        This is necessary to get the operator to work efficiently with torch.compile.

        The Fake impl (sometimes also known as a meta kernel or abstract impl)
        specifies the behavior of this operator on Tensors that carry no data.
        Given some input Tensors with certain properties
        (sizes/strides/storage_offset/device), it specifies what the properties of
        the output Tensors are.

        Please see :func:`torch.library.register_fake` for more details.

        Args:
            fn (Callable): The function to register as the FakeTensor
                implementation.

        Examples:
            >>> import torch
            >>> import numpy as np
            >>> from torch import Tensor
            >>>
            >>> # Example 1: an operator without data-dependent output shape
            >>> @torch.library.custom_op("mylib::linear", mutates_args=())
            >>> def linear(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
            >>>     return (x @ weight.t()) + bias
            >>>
            >>> @linear.register_fake
            >>> def _(x, weight, bias):
            >>>     assert x.dim() == 2
            >>>     assert weight.dim() == 2
            >>>     assert bias.dim() == 1
            >>>     assert x.shape[1] == weight.shape[1]
            >>>     assert weight.shape[0] == bias.shape[0]
            >>>     assert x.device == weight.device
            >>>     return x.new_empty(x.size(0), weight.size(0))
            >>>
            >>> x = torch.randn(2, 2)
            >>> weight = torch.randn(2, 2)
            >>> bias = torch.randn(2)
            >>> # xdoctest: +SKIP("Requires Python <= 3.11")
            >>> out = torch.compile(linear, fullgraph=True)(x, weight, bias)
            >>> # xdoctest: +SKIP("Requires Python <= 3.11")
            >>> assert torch.allclose(out, torch.nn.functional.linear(x, weight, bias))
            >>>
            >>> # Example 2: an operator with data-dependent output shape
            >>> @torch.library.custom_op("mylib::nonzero", mutates_args=())
            >>> def nonzero(x: Tensor) -> Tensor:
            >>>     x_np = x.cpu().numpy()
            >>>     res = np.stack(np.nonzero(x_np), axis=1)
            >>>     return torch.tensor(res, device=x.device)
            >>>
            >>> @nonzero.register_fake
            >>> def _(x):
            >>> # Number of nonzero-elements is data-dependent.
            >>> # Since we cannot peek at the data in an abstract impl,
            >>> # we use the ctx object to construct a new symint that
            >>> # represents the data-dependent size.
            >>>     ctx = torch.library.get_ctx()
            >>>     nnz = ctx.new_dynamic_size()
            >>>     shape = [nnz, x.dim()]
            >>>     result = x.new_empty(shape, dtype=torch.int64)
            >>>     return result
            >>>
            >>> x = torch.tensor([0, 1, 2, 0, 0, 1])
            >>> # xdoctest: +SKIP("Requires Python <= 3.11")
            >>> out = torch.compile(nonzero, fullgraph=True)(x)
            >>> # xdoctest: +SKIP("Requires Python <= 3.11")
            >>> assert torch.allclose(out, x.nonzero())

        """
        if self._is_inplace or self._is_out:
            tag = "torch.Tag.inplace" if self._is_inplace else "torch.Tag.out"
            warnings.warn(
                f"The fake registration for {self} is unnecessary. Custom ops "
                f"tagged with {tag} already get an autogenerated fake kernel. "
                "The provided fake impl will be used instead.",
                stacklevel=2,
            )
        self._abstract_fn = fn
        return fn

    def register_effect(self, effect: EffectType | None) -> None:
        self._lib._register_effectful_op(self._qualname, effect)

    def register_torch_dispatch(
        self, torch_dispatch_class: Any, fn: Callable | None = None, /
    ) -> Callable:
        r"""Registers a torch_dispatch rule for the given operator and ``torch_dispatch_class``.

        This allows for open registration to specify the behavior between the operator
        and the ``torch_dispatch_class`` without needing to modify the ``torch_dispatch_class``
        or the operator directly.

        Please see :func:`torch.library.register_torch_dispatch` for examples and more details.
        """

        def register(fn):
            if torch_dispatch_class not in self._torch_dispatch_fns:

                def inner(*args, **kwargs):
                    return self._torch_dispatch_fns[torch_dispatch_class](
                        *args, **kwargs
                    )

                self._lib._register_torch_dispatch_rule(
                    self._name, torch_dispatch_class, inner
                )
            self._torch_dispatch_fns[torch_dispatch_class] = fn
            return fn

        if fn is None:
            return register
        else:
            return register(fn)

    def register_autograd(
        self,
        backward: Callable,
        /,
        *,
        setup_context: Callable | None = None,
    ) -> None:
        r"""Register a backward formula for this custom op.

        In order for an operator to work with autograd, you need to register
        a backward formula:
        1. You must tell us how to compute gradients during the backward pass
        by providing us a "backward" function.
        2. If you need any values from the forward to compute gradients, you can
        use `setup_context` to save values for backward.

        ``backward_fn`` runs during the backward pass. It accepts ``(ctx, *grads)``:
        - ``grads`` is one or more gradients. The number of gradients matches
        the number of outputs of the operator.
        The ``ctx`` object is `the same ctx object <context_method_mixins>`_ used by
        :class:`torch.autograd.Function`. The semantics of ``backward_fn`` are the
        same as :meth:`torch.autograd.Function.backward`.

        ``setup_context(ctx, inputs, output)`` runs during the forward pass.
        Please save quantities needed for backward onto the ``ctx`` object via
        either :meth:`torch.autograd.function.FunctionCtx.save_for_backward`
        or assigning them as attributes of ``ctx``. If your custom op has
        kwarg-only arguments, we expect the signature of ``setup_context``
        to be ``setup_context(ctx, inputs, keyword_only_inputs, output)``.

        Both ``setup_context_fn`` and ``backward_fn`` must be traceable. That is,
        they may not directly access :meth:`torch.Tensor.data_ptr` and they must
        not depend on or mutate global state. If you need a non-traceable backward,
        you can make it a separate custom_op that you call inside ``backward_fn``.

        If you need different autograd behavior on different devices, then we
        recommend creating two different custom operators, one for each device
        that needs different behavior, and switching between them at runtime.

        Examples:
            >>> import torch
            >>> import numpy as np
            >>> from torch import Tensor
            >>>
            >>> @torch.library.custom_op("mylib::numpy_sin", mutates_args=())
            >>> def numpy_sin(x: Tensor) -> Tensor:
            >>>     x_np = x.cpu().numpy()
            >>>     y_np = np.sin(x_np)
            >>>     return torch.from_numpy(y_np).to(device=x.device)
            >>>
            >>> def setup_context(ctx, inputs, output) -> Tensor:
            >>>     x, = inputs
            >>>     ctx.save_for_backward(x)
            >>>
            >>> def backward(ctx, grad):
            >>>     x, = ctx.saved_tensors
            >>>     return grad * x.cos()
            >>>
            >>> numpy_sin.register_autograd(backward, setup_context=setup_context)
            >>>
            >>> x = torch.randn(3, requires_grad=True)
            >>> y = numpy_sin(x)
            >>> (grad_x,) = torch.autograd.grad(y, x, torch.ones_like(y))
            >>> assert torch.allclose(grad_x, x.cos())
            >>>
            >>> # Example with a keyword-only arg
            >>> @torch.library.custom_op("mylib::numpy_mul", mutates_args=())
            >>> def numpy_mul(x: Tensor, *, val: float) -> Tensor:
            >>>     x_np = x.cpu().numpy()
            >>>     y_np = x_np * val
            >>>     return torch.from_numpy(y_np).to(device=x.device)
            >>>
            >>> def setup_context(ctx, inputs, keyword_only_inputs, output) -> Tensor:
            >>>     ctx.val = keyword_only_inputs["val"]
            >>>
            >>> def backward(ctx, grad):
            >>>     return grad * ctx.val
            >>>
            >>> numpy_mul.register_autograd(backward, setup_context=setup_context)
            >>>
            >>> x = torch.randn(3, requires_grad=True)
            >>> y = numpy_mul(x, val=3.14)
            >>> (grad_x,) = torch.autograd.grad(y, x, torch.ones_like(y))
            >>> assert torch.allclose(grad_x, torch.full_like(x, 3.14))

        """
        schema = self._opoverload._schema
        if utils.is_out(self._opoverload):
            raise RuntimeError(
                f"Cannot register autograd formula for operator tagged with "
                f"torch.Tag.out: {self}. Out variants do not support autograd."
            )
        if not utils.is_functional_schema(schema, allow_valid_view=True):
            raise RuntimeError(
                f"Cannot register autograd formula for non-functional operator "
                f"{self} with schema {schema}. Please create "
                f"a functional operator and register an autograd formula for that."
            )

        self._backward_fn = backward
        self._setup_context_fn = setup_context

    def _register_to_dispatcher(self, tags: Sequence[_C.Tag]) -> None:
        schema_str = self._name + self._schema
        cpp_schema = _C.parse_schema(schema_str)
        self._validate_schema(cpp_schema, schema_str)
        self._define_dispatcher_op(schema_str, tags)
        self._is_inplace = utils.is_inplace(self._opoverload)
        self._is_out = utils.is_out(self._opoverload)
        if self._is_out:
            _, out_kwarg_names = utils.mutated_args_kwargs(self._opoverload._schema)
            self._out_kwarg_names = tuple(out_kwarg_names)
        self._register_fake_dispatcher_impl()
        self._register_autograd_dispatcher_impl()
        self._register_adinplaceorview_dispatcher_impl()

    def _validate_schema(self, schema: _C.FunctionSchema, schema_str: str) -> None:
        if utils.has_kwarg_only_tensors(schema) and torch.Tag.out not in self._tags:
            # If you want to support this, the progression is:
            # - supporting kwarg-only Tensors that are non-differentiable
            # - supporting kwarg-only Tensors (regardless of differentiability)
            raise NotImplementedError(
                f"custom_op with kwarg-only Tensor args. Please make your "
                f"tensors not kwarg-only. Got: {schema_str}"
            )

    def _define_dispatcher_op(self, schema_str: str, tags: Sequence[_C.Tag]) -> None:
        self._lib.define(
            schema_str,
            tags=_with_pt2_compliant_tag(tags),
        )
        self._opoverload = utils.lookup_op(self._qualname)

    def _register_fake_dispatcher_impl(self) -> None:
        def fake_impl(*args, **kwargs):
            if self._abstract_fn is None:
                if utils.can_generate_trivial_fake_impl(self._opoverload):
                    return utils.generate_trivial_fake_impl(
                        self._opoverload, *args, **kwargs
                    )
                raise RuntimeError(
                    f"There was no fake impl registered for {self}. "
                    f"This is necessary for torch.compile/export/fx tracing to work. "
                    f"Please use `{self._init_fn.__name__}.register_fake` to add an "
                    f"fake impl."
                )
            return self._abstract_fn(*args, **kwargs)

        self._lib._register_fake(self._name, fake_impl, _stacklevel=5)

    def _register_autograd_dispatcher_impl(self) -> None:
        autograd_impl = autograd.make_autograd_impl(self._opoverload, self)
        self._lib.impl(self._name, autograd_impl, "Autograd", with_keyset=True)

    def _register_adinplaceorview_dispatcher_impl(self) -> None:
        schema = self._opoverload._schema
        if not (schema._is_view_op() or schema.is_mutable):
            return

        self._lib.m.register_ad_inplace_or_view_fallback(self._name)  # type: ignore[union-attr]

        if schema.is_mutable:
            self._register_mutation_version_bump(schema)

    def _register_mutation_version_bump(self, schema: _C.FunctionSchema) -> None:
        mutated_idxs, mutated_keys = utils.mutated_args_kwargs(schema)

        is_view = schema._is_view_op()
        if is_view:
            original_kernel = torch._C._dispatch_get_computed_kernel_for_dispatch_key(
                f"{self._lib.ns}::{self._name}", "ADInplaceOrView"
            )
        else:
            op = self._opoverload
            after_ADInplaceOrView_keyset = _C._after_ADInplaceOrView_keyset

        def adinplaceorview_impl(keyset, *args, **kwargs):
            # The dispatcher normalizes non-keyword-only arguments into args.
            # Avoid filling defaults for every schema argument on this hot path.
            for idx in mutated_idxs:
                if idx < len(args):
                    increment_version(args[idx])
            for key in mutated_keys:
                if key in kwargs:
                    increment_version(kwargs[key])
            if is_view:
                # View ops need the C++ fallback for aliasing tracking
                return original_kernel.call_boxed(keyset, *args, **kwargs)
            # Non-view mutable ops: increment_version is sufficient,
            # redispatch directly past ADInplaceOrView to the backend.
            return op.redispatch(keyset & after_ADInplaceOrView_keyset, *args, **kwargs)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Warning only once for all operators",
                category=UserWarning,
            )
            self._lib.impl(
                self._name,
                adinplaceorview_impl,
                "ADInplaceOrView",
                with_keyset=True,
            )

    def _register_backend_select_dispatcher(self, device_arg_index: int):
        """
        Switch on the device argument to select the correct backend to dispatch to.
        """

        def backend_select(keyset, *args, **kwargs):
            device = args[device_arg_index].type
            if device not in self._backend_fns:
                raise RuntimeError(
                    f"{self._name} does not have a kernel registered for {device}. "
                    "Please use register_kernel to do so."
                )
            # pyrefly: ignore [bad-argument-type]
            dispatch_key = _C._dispatch_key_for_device(device)
            dispatch_key = getattr(_C.DispatchKey, dispatch_key)
            return self._opoverload.redispatch(
                _C.DispatchKeySet(dispatch_key), *args, **kwargs
            )

        self._lib.impl(self._name, backend_select, "BackendSelect", with_keyset=True)

    def __call__(self, *args, **kwargs):
        return self._opoverload(*args, **kwargs)

    def register_vmap(
        self,
        func: Callable | None = None,
    ):
        r"""Register a vmap implementation to support :func:`torch.vmap` for this custom op.

        This API may be used as a decorator.

        In order for an operator to work with :func:`torch.vmap`, you may need to register a
        vmap implementation in the following signature:

            ``vmap_func(info, in_dims: Tuple[Optional[int]], *args, **kwargs)``,

        where ``*args`` and ``**kwargs`` are the arguments and kwargs for ``op``.

        It specifies how do we compute the batched version of ``op`` given inputs with an additional
        dimension (specified by ``in_dims``).

        For each arg in ``args``, ``in_dims`` has a corresponding ``Optional[int]``. It is ``None``
        if the arg is not a Tensor or if the arg is not being vmapped over, otherwise, it is an integer
        specifying what dimension of the Tensor is being vmapped over.

        ``info`` is a collection of additional metadata that may be helpful:
        ``info.batch_size`` specifies the size of the dimension being vmapped over, while
        ``info.randomness`` is the ``randomness`` option that was passed to :func:`torch.vmap`.

        The return of the function ``func`` is a tuple of ``(output, out_dims)``. Similar to ``in_dims``,
        ``out_dims`` should be of the same structure as ``output`` and contain one ``out_dim``
        per output that specifies if the output has the vmapped dimension and what index it is in.

        Examples:
            >>> import torch
            >>> import numpy as np
            >>> from torch import Tensor
            >>> from typing import Tuple
            >>>
            >>> def to_numpy(tensor):
            >>>     return tensor.cpu().numpy()
            >>>
            >>> lib = torch.library.Library("mylib", "FRAGMENT")
            >>> @torch.library.custom_op("mylib::numpy_cube", mutates_args=())
            >>> def numpy_cube(x: Tensor) -> Tuple[Tensor, Tensor]:
            >>>     x_np = to_numpy(x)
            >>>     dx = torch.tensor(3 * x_np ** 2, device=x.device)
            >>>     return torch.tensor(x_np ** 3, device=x.device), dx
            >>>
            >>> def numpy_cube_vmap(info, in_dims, x):
            >>>     result = numpy_cube(x)
            >>>     return result, (in_dims[0], in_dims[0])
            >>>
            >>> numpy_cube.register_vmap(numpy_cube_vmap)
            >>>
            >>> x = torch.randn(3)
            >>> torch.vmap(numpy_cube)(x)
            >>>
            >>> @torch.library.custom_op("mylib::numpy_mul", mutates_args=())
            >>> def numpy_mul(x: Tensor, y: Tensor) -> Tensor:
            >>>     return torch.tensor(to_numpy(x) * to_numpy(y), device=x.device)
            >>>
            >>> @numpy_mul.register_vmap
            >>> def numpy_mul_vmap(info, in_dims, x, y):
            >>>     x_bdim, y_bdim = in_dims
            >>>     x = x.movedim(x_bdim, -1) if x_bdim is not None else x.unsqueeze(-1)
            >>>     y = y.movedim(y_bdim, -1) if y_bdim is not None else y.unsqueeze(-1)
            >>>     result = x * y
            >>>     result = result.movedim(-1, 0)
            >>>     return result, 0
            >>>
            >>>
            >>> x = torch.randn(3)
            >>> y = torch.randn(3)
            >>> torch.vmap(numpy_mul)(x, y)
        """
        from torch._functorch.autograd_function import custom_function_call_vmap_helper
        from torch._functorch.pyfunctorch import retrieve_current_functorch_interpreter

        def register(func):
            need_register = self._vmap_fn is None
            self._vmap_fn = func

            if need_register:

                def wrapped_func(keyset, *args, **kwargs):
                    interpreter = retrieve_current_functorch_interpreter()
                    return custom_function_call_vmap_helper(
                        # pyrefly: ignore[bad-argument-type]
                        interpreter,
                        # pyrefly: ignore[bad-argument-type]
                        self._vmap_fn,
                        self._opoverload,
                        *args,
                        **kwargs,
                    )

                self._lib.impl(
                    self._name, wrapped_func, "FuncTorchBatched", with_keyset=True
                )

        if func is None:
            return register
        else:
            return register(func)

    def register_autocast(
        self,
        device_type: str,
        cast_inputs: _dtype,
    ):
        r"""Register an autocast dispatch rule for this custom op.

        Valid ``device_type`` values include any device type that supports autocast.
        See :func:`torch.amp.is_autocast_available` for details.

        Args:
            op (str | OpOverload): The operator to register an autocast dispatch rule to.
            device_type(str):  Device type to use. 'cuda', 'cpu', 'xpu', or any other device type that supports autocast.
                The type is the same as the `type` attribute of a :class:`torch.device`.
                Thus, you may obtain the device type of a tensor using `Tensor.device.type`.
            cast_inputs (:class:`torch.dtype`): When custom op runs in an autocast-enabled region,
                casts incoming floating-point Tensors to the target dtype (non-floating-point Tensors
                are not affected), then executes custom op with autocast disabled.
            lib (Optional[Library]): If provided, the lifetime of this registration

        Examples::
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
            >>> import torch
            >>> from torch import Tensor
            >>> from torch.library import custom_op
            >>>
            >>> # Create a custom op that works on cuda
            >>> @torch.library.custom_op("mylib::my_sin", mutates_args=())
            >>> def my_sin(x: Tensor) -> Tensor:
            >>>     return torch.sin(x)
            >>>
            >>> # Register autocast dispatch rule for the cuda device
            >>> torch.library.register_autocast("mylib::my_sin", "cuda", torch.float16)
            >>>
            >>> x = torch.randn(3, dtype=torch.float32, device="cuda")
            >>> with torch.autocast("cuda", dtype=torch.float16):
            >>>     y = torch.ops.mylib.my_sin(x)
            >>> assert y.dtype == torch.float16

        """
        if not isinstance(device_type, str):
            raise ValueError(
                f"Expected `device_type` of type `str`, got: `{type(device_type)}`"
            )
        if not torch._C._is_autocast_available(device_type):
            raise ValueError(f"Device type '{device_type}' does not support autocast.")

        need_register = self._autocast_dtype.get(device_type) is None
        if need_register:
            self._autocast_dtype[device_type] = cast_inputs
        autocast_key = "Autocast" + torch._C._dispatch_key_for_device(device_type)
        autocast_dispatch_key = getattr(torch._C.DispatchKey, autocast_key)

        def kernel(_, *args, **kwargs):
            if len(kwargs) != 0:
                raise AssertionError(
                    f"Custom ops do not support kwargs yet, got {list(kwargs.keys())}"
                )
            with torch._C._ExcludeDispatchKeyGuard(
                torch._C.DispatchKeySet(autocast_dispatch_key)
            ):
                return self._opoverload(*_cast(args, device_type, cast_inputs))

        if need_register:
            self._lib.impl(self._name, kernel, autocast_key, with_keyset=True)

        return kernel


# TODO: Merge this function with torch.amp.autocast_mode._cast, and refactor it
# into a utility function once custom ops support arbitrary input types.
def _cast(value, device_type: str, dtype: _dtype):
    if isinstance(value, torch.Tensor):
        is_eligible = (
            value.is_floating_point()
            and value.device.type == device_type
            and (value.dtype is not torch.float64)
        )
        return value.to(dtype) if is_eligible else value
    elif isinstance(value, (str, bytes)):
        return value
    elif isinstance(value, collections.abc.Iterable):
        iterable = (_cast(v, device_type, dtype) for v in value)
        if isinstance(value, (list, tuple)):
            return type(value)(iterable)
        else:
            return iterable
    else:
        return value


def increment_version(val: Any) -> None:
    if isinstance(val, Tensor):
        torch.autograd.graph.increment_version(val)
    elif isinstance(val, (tuple, list)):
        for v in val:
            if isinstance(v, Tensor):
                torch.autograd.graph.increment_version(v)


# NOTE: [Supporting decorator and non-decorator usage]
#
# Some APIs may be both used as a decorator and not as a decorator.
# For example:
#
# >>> def fn(x):
# >>>     return x.sin()
# >>>
# >>> # Usage 1: not as a decorator
# >>> numpy_sin.register_kernel("cuda", fn)
# >>>
# >>> # Usage 2: as a decorator
# >>> @numpy_sin.register_kernel("cuda")
# >>> def fn2(x):
# >>>     return x.sin
#
# The way we support this is that `register_kernel` accepts an optional `fn`.
# If `fn` is provided (Usage 1), then we know that the user is using it not
# as a decorator.
# If `fn` is not provided (Usage 2), then `register_kernel` needs to return a
# decorator.


OPDEF_TO_LIB: dict[str, "torch.library.Library"] = {}
OPDEFS: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


def get_library_allowing_overwrite(
    namespace: str, name: str
) -> "torch.library.Library":
    qualname = f"{namespace}::{name}"

    if qualname in OPDEF_TO_LIB:
        OPDEF_TO_LIB[qualname]._destroy()
        del OPDEF_TO_LIB[qualname]

    lib = torch.library.Library(namespace, "FRAGMENT")
    OPDEF_TO_LIB[qualname] = lib
    return lib


def _maybe_get_opdef(
    op: CustomOpDef | _ops.OpOverload | str,
) -> CustomOpDef | None:
    if isinstance(op, CustomOpDef):
        return op
    if isinstance(op, _ops.OpOverload):
        op = op._name
    if not isinstance(op, str):
        raise AssertionError(f"op must be str, got {type(op)}")
    if op in OPDEFS:
        return OPDEFS[op]
    return None
<|endoftext|># mypy: allow-untyped-defs
import dataclasses
import inspect
import sys
from collections.abc import Callable, Iterable, Iterator
from typing import Any, Literal, overload

import torch
import torch.utils._pytree as pytree
import torchgen
from torch import _C, _utils_internal
from torch._ops import OpOverload


@dataclasses.dataclass
class Kernel:
    """Models a (function, source location)"""

    func: Callable
    source: str

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class RegistrationHandle:
    """Does something when someone calls .destroy() on it"""

    def __init__(self, on_destroy: Callable):
        self._on_destroy = on_destroy

    def destroy(self) -> None:
        self._on_destroy()


def get_source(stacklevel: int) -> str:
    """Get a string that represents the caller.

    Example: "/path/to/foo.py:42"

    Use stacklevel=1 to get the caller's source
    Use stacklevel=2 to get the caller's caller's source
    etc.
    """
    frame = inspect.getframeinfo(sys._getframe(stacklevel))
    source = f"{frame.filename}:{frame.lineno}"
    return source


def parse_namespace(qualname: str) -> tuple[str, str]:
    splits = qualname.split("::")
    if len(splits) != 2:
        raise ValueError(
            f"Expected `qualname` to be of the form "
            f'"namespace::name", but got {qualname}. '
            f"The qualname passed to the torch.library APIs must consist "
            f"of a namespace and a name, e.g. aten::sin"
        )
    return splits[0], splits[1]


def lookup_op(qualname: str) -> OpOverload:
    namespace, name = parse_namespace(qualname)
    if "." in name:
        name, overload = name.split(".")
    else:
        overload = "default"
    ns = getattr(torch.ops, namespace)
    packet = getattr(ns, name)
    return getattr(packet, overload)


def is_builtin(op: OpOverload) -> bool:
    if not isinstance(op, OpOverload):
        raise AssertionError(f"op must be OpOverload, got {type(op)}")
    return op.namespace in {"aten", "prim", "prims"}


def is_out(op: OpOverload) -> bool:
    """Returns True if the operator has "out" semantics: its mutable arguments
    are write-only output buffers that are not read from."""
    return torch.Tag.out in op.tags


def is_inplace(op: OpOverload) -> bool:
    """Returns True if the operator has inplace semantics: it mutates its first
    positional argument and returns it."""
    return torch.Tag.inplace in op.tags


def is_functional_schema(schema: Any, *, allow_valid_view: bool = False) -> bool:
    """Check if the schema is functional.

    An operator is functional if:
    - it does not mutate any of its inputs
    - If no view are allowed
        - it does not return a view on any of its inputs
    - If valid views are allowed
        - it is not a view or a view with a single input Tensor and single output Tensor
    - it has at least one return
    """

    def is_functional(schema):
        if schema.is_mutable:
            return False
        rets = schema.returns
        is_non_mutating_view = len(rets) > 0 and any(
            r.alias_info is not None and not r.alias_info.is_write for r in rets
        )
        num_tensor_inputs = 0
        num_tensor_outputs = 0

        if isinstance(schema, torch.FunctionSchema):
            for arg in schema.arguments:
                if isinstance(arg.type, torch.TensorType):
                    num_tensor_inputs += 1

            for ret in schema.returns:
                if isinstance(ret.type, torch.TensorType):
                    num_tensor_outputs += 1

        elif isinstance(schema, torchgen.model.FunctionSchema):
            for argument in schema.arguments.flat_non_out:
                if argument.type.is_tensor_like():
                    num_tensor_inputs += 1

            for ret_arg in schema.returns:
                if ret_arg.type.is_tensor_like():
                    num_tensor_outputs += 1

        if is_non_mutating_view:
            return allow_valid_view and (
                num_tensor_inputs == 1 and num_tensor_outputs == 1
            )
        if not schema.returns:
            return False
        return True

    if isinstance(schema, torch._C.FunctionSchema):
        return is_functional(schema)

    # Lazy import because not all PyTorch builds have torchgen
    from torchgen.model import FunctionSchema

    if isinstance(schema, str):
        schema = FunctionSchema.parse(schema)
    if not isinstance(schema, FunctionSchema):
        raise AssertionError(f"schema must be FunctionSchema, got {type(schema)}")
    return is_functional(schema)


# should be torch._C.JitType but that annotation is busted
def is_tensorlist_like_type(typ: Any) -> bool:
    return (
        typ == _C.ListType(_C.TensorType.get())
        or typ == _C.ListType(_C.OptionalType(_C.TensorType.get()))
        or typ == _C.OptionalType(_C.ListType(_C.TensorType.get()))
        or typ == _C.OptionalType(_C.ListType(_C.OptionalType(_C.TensorType.get())))
    )


# should be torch._C.JitType but that annotation is busted
def is_tensor_like_type(typ: Any) -> bool:
    return typ == _C.TensorType.get() or typ == _C.OptionalType(_C.TensorType.get())


def mutates_and_returns_first_arg(op: OpOverload):
    """Check if an op is an inplace aten op, i.e. it mutates and returns the first arg.

    TODO: torchgen/model.py's FunctionSchema.parse is the source of truth for this,
    but not all PyTorch builds have torchgen (due to the yaml dependency being weird).
    Figure this out.

    Example: add_(Tensor(a!) x, Tensor y) -> Tensor(a)
    """
    if op.namespace != "aten":
        return False
    schema = op._schema
    if len(schema.returns) != 1:
        return False
    if schema.returns[0].alias_info is None:
        return False
    alias_set = schema.returns[0].alias_info.after_set
    if len(alias_set) != 1:
        return False
    loc = next(iter(alias_set))
    if len(schema.arguments) < 1:
        return False
    first_arg = schema.arguments[0]
    if first_arg.alias_info is None:
        return False
    if not first_arg.alias_info.is_write:
        return False
    alias_set = first_arg.alias_info.after_set
    if len(alias_set) != 1:
        return False
    if loc != next(iter(alias_set)):
        return False
    for arg in schema.arguments[1:]:
        if arg.alias_info is not None:
            return False
    return True


def fill_defaults(schema, args, kwargs):
    new_args = []
    new_kwargs = {}
    for i in range(len(schema.arguments)):
        info = schema.arguments[i]
        if info.kwarg_only:
            if info.name in kwargs:
                new_kwargs[info.name] = kwargs[info.name]
            else:
                new_kwargs[info.name] = info.default_value
        else:
            if i < len(args):
                new_args.append(args[i])
            else:
                new_args.append(info.default_value)
    return tuple(new_args), new_kwargs


def zip_schema(
    schema: _C.FunctionSchema, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Iterable[tuple[_C.Argument, Any]]:
    """zips schema.arguments and (args, kwargs) together.

    Assumes that (args, kwargs) were the inputs to some torch._ops.OpOverload:
    that is, (args, kwargs) must be bindable to the schema (args, kwargs).
    """
    if len(schema.arguments) < len(args) + len(kwargs):
        raise AssertionError(
            f"schema has {len(schema.arguments)} arguments but got {len(args)} args and {len(kwargs)} kwargs"
        )
    for i in range(len(schema.arguments)):
        info = schema.arguments[i]
        if info.kwarg_only:
            if info.name in kwargs:
                yield info, kwargs[info.name]
            continue
        if i >= len(args):
            if not info.kwarg_only and info.name in kwargs:
                yield info, kwargs[info.name]
            # args that are equal to their default values are not populated
            # if they are followed by args that are equal to their defaults.
            # Skip these.
            continue
        yield info, args[i]
    return


def hop_schema_from_fx_node(node):
    from torchgen.gen_schema_utils import FunctionSchemaGen

    hop = node.target
    if not isinstance(hop, torch._ops.HigherOrderOperator):
        raise RuntimeError("fx_node's target must be a hop.")

    def _collect_example_val(node):
        meta_val = node.meta.get("val", None)
        if meta_val is None:
            if node.op != "get_attr":
                raise AssertionError(
                    f"node.op must be 'get_attr' when val is None, got {node.op!r}"
                )
            meta_val = getattr(node.graph.owning_module, node.target)
        return meta_val

    example_inputs = []
    for arg in node.args:
        if isinstance(arg, (torch.fx.Node, torch.fx.node.Node)):
            example_inputs.append(_collect_example_val(arg))
        elif isinstance(
            arg, (torch.fx.immutable_collections.immutable_list, list, tuple)
        ):
            example_inputs.append([_collect_example_val(x) for x in arg])
        else:
            raise RuntimeError(f"Unsupported arg type {type(arg)}")

    # Bound the arguments to make sure number of inputs are correct
    bound_args: inspect.BoundArguments = inspect.signature(hop.__call__).bind(
        *example_inputs
    )

    # We treat example_output as a single value in return. This is to differentiate 1. return a single val
    # vs 2. return a tuple with one element.
    example_output = _collect_example_val(node)
    return FunctionSchemaGen.from_example(
        hop._name, tuple(bound_args.arguments.items()), (list(example_output),)
    )


def can_generate_trivial_fake_impl(op: OpOverload) -> bool:
    if not isinstance(op, OpOverload):
        raise AssertionError(f"op must be OpOverload, got {type(op)}")
    if is_builtin(op):
        # We control the built-ins. These may (in rare cases)
        # do input metadata mutation (which we have banned on custom ops)
        return False
    schema = op._schema
    if is_out(op):
        # Tag.out ops have a trivial fake impl: return the out= args in order.
        return True
    if is_inplace(op):
        # Tag.inplace ops have a trivial fake impl: return the mutated first arg.
        return True
    # It's suspicious if the op is not mutable but returns nothing, so we return False out of an abundance of caution
    if not schema.is_mutable:
        return False
    if len(schema.returns) > 0:
        return False
    # If the op returns nothing, then it has a trivial fake impl.
    return True


def generate_trivial_fake_impl(op: OpOverload, *args, **kwargs):
    """Generate the result of a trivial fake impl for the given op.

    For ops with no returns: returns None.
    For Tag.out ops: returns the out= kwargs in declaration order.
    For Tag.inplace ops: returns the first positional arg.
    """
    if is_out(op):
        schema = op._schema
        _, out_kwarg_names = mutated_args_kwargs(schema)
        out_args = tuple(kwargs[name] for name in out_kwarg_names)
        if len(out_args) == 1:
            return out_args[0]
        return out_args
    if is_inplace(op):
        return args[0]
    return None


def requires_set_python_module() -> bool:
    """If an op was defined in C++ and extended from Python using the
    torch.library APIs, returns if we require that there have been a
    m.set_python_module("mylib.ops") call from C++ that associates
    the C++ op with a python module.
    """
    return getattr(_utils_internal, "REQUIRES_SET_PYTHON_MODULE", True)


def handle_dispatch_mode(curr_mode, op_overload, *args, **kwargs):
    if not isinstance(curr_mode, torch.utils._python_dispatch.TorchDispatchMode):
        raise AssertionError(
            f"curr_mode must be TorchDispatchMode, got {type(curr_mode)}"
        )
    args_flattened, _ = torch.utils._pytree.tree_flatten((args, kwargs.values()))
    # TODO: need to double check the semantics of the "types" argument to torch_dispatch.
    # It's generated in PyInterpreter.cpp, but seems to be generated in two places,
    # where in one case we only include tensors with the python key, and in another
    # we include **all** tensors.
    overload_types = [
        type(a)
        for a in args_flattened
        if isinstance(a, torch.Tensor)
        and torch._C._dispatch_keys(a).has(torch._C.DispatchKey.Python)
    ]
    # TODO: check that I got these args correct (in C++, we pass in "0000"??)

    return curr_mode.__torch_dispatch__(op_overload, overload_types, args, kwargs)


def has_kwarg_only_args(schema: _C.FunctionSchema):
    return any(a.kwarg_only for a in schema.arguments)


def has_kwarg_only_tensors(schema: _C.FunctionSchema):
    for a in schema.arguments:
        if not (is_tensor_like_type(a.type) or is_tensorlist_like_type(a.type)):
            continue
        if not a.kwarg_only:
            continue
        return True
    return False


def has_tensor_arg(schema: _C.FunctionSchema) -> bool:
    """
    Given a schema, returns True if the schema has a Tensor arg.
    A Tensor arg is any arg with a type annotation that might involve Tensor.
    """
    return any(
        (is_tensor_like_type(a.type) or is_tensorlist_like_type(a.type))
        for a in schema.arguments
    )


def get_device_arg_index(schema: _C.FunctionSchema) -> int | None:
    """
    Given a schema, returns the id of the `device: torch.device` argument.
    If it does not exist, returns None.
    """
    for index, arg in enumerate(schema.arguments):
        if arg.type is _C.DeviceObjType.get() and arg.name == "device":
            return index
    return None


def iter_tensors(
    args: tuple[Any], kwargs: dict[str, Any], allowed_nesting: int = 1
) -> Iterator[torch.Tensor]:
    def check(arg):
        if isinstance(arg, torch.Tensor):
            yield arg
        elif allowed_nesting > 0 and isinstance(arg, (tuple, list)):
            yield from iter_tensors(tuple(arg), {}, allowed_nesting - 1)

    for arg in args:
        yield from check(arg)
    for kwarg in kwargs.values():
        yield from check(kwarg)


def check_aliasing_constraint(name, prev, result, get_module=lambda: "???"):
    """
    custom operators' outputs must not alias any inputs or other outputs.
    """
    storages = {t.untyped_storage()._cdata for t in prev if isinstance(t, torch.Tensor)}
    tuple_result = result
    if not isinstance(result, tuple):
        tuple_result = (result,)
    for tensor in iter_tensors(tuple_result, {}):
        key = tensor.untyped_storage()._cdata
        if tensor.untyped_storage()._cdata in storages:
            raise RuntimeError(
                f"{name} (with implementation in {get_module()}): "
                f"The output of this custom operator (1) must not "
                f"also be an input to this custom operator and "
                f"(2) may not alias any inputs to this custom operator "
                f"or other returns. "
                f"The most common way to trigger this error is if "
                f"we have y = custom_op(x) and y and x are the same Tensor. "
                f"Please instead return a clone of the offending output "
                f"tensor(s) (e.g. return x.clone()) or refactor the custom "
                f"operator to not return y."
            )
        storages.add(key)


def _c_check_aliasing_constraint(name, args, kwargs, result, get_module=lambda: "???"):
    """
    custom operators' outputs must not have any aliases
    This version uses C++ implementation for perf.
    Only List container is supported.
    Tensors in Lists with not only Tensors are checked.
    """
    tuple_result = result
    if not isinstance(result, tuple):
        tuple_result = (result,)
    if _C._any_output_is_alias_to_input_or_output(args, kwargs, tuple_result):
        raise RuntimeError(
            f"{name} (with implementation in {get_module()}): "
            f"The output of this custom operator (1) must not "
            f"also be an input to this custom operator and "
            f"(2) may not alias any inputs to this custom operator "
            f"or other returns. "
            f"The most common way to trigger this error is if "
            f"we have y = custom_op(x) and y and x are the same Tensor. "
            f"Please instead return a clone of the offending output "
            f"tensor(s) (e.g. return x.clone()) or refactor the custom "
            f"operator to not return y."
        )


class MutationChecker:
    """
    Check if an operator mutated its arguments.
    Usage:

    checker = MutationChecker(op, flat_args, args_spec)
    op(*args, **kwargs)
    checker.check()
    """

    def __init__(self, op, flat_args, args_spec):
        self.op = op
        self.args_spec = args_spec
        self.flat_args = flat_args
        self.real_pre_hashes = [
            hash_tensor(a) if isinstance(a, torch.Tensor) else None for a in flat_args
        ]

    def check(self):
        real_post_hashes = [
            hash_tensor(a) if isinstance(a, torch.Tensor) else None
            for a in self.flat_args
        ]
        was_mutated = [
            not torch.equal(pre, post)
            and not (pre.isnan().all() and post.isnan().all())
            if isinstance(pre, torch.Tensor) and isinstance(post, torch.Tensor)
            else None
            for pre, post in zip(self.real_pre_hashes, real_post_hashes)
        ]
        was_mutated_args, was_mutated_kwargs = pytree.tree_unflatten(
            was_mutated, self.args_spec
        )
        for info, was_mutated in zip_schema(
            self.op._schema, was_mutated_args, was_mutated_kwargs
        ):

            def check_one(info, was_mutated):
                if info.is_write == was_mutated:
                    return
                raise RuntimeError(
                    f"{self.op._name}: for argument '{info.name}': the operator's schema "
                    f"{self.op._schema} specified that "
                    f"the operator {'mutates' if info.is_write else 'does not mutate'} "
                    f"the argument, but this seems to be empirically wrong. "
                    f"Please make the schema and operator behavior consistent. "
                    f"You can specify that an operator mutates a Tensor by "
                    f"e.g. changing its schema type from 'Tensor name' to 'Tensor(a!) name'"
                    f"(use different identifiers (a, b, c, ...) for different Tensors)"
                )

            if is_tensor_like_type(info.type):
                check_one(info, was_mutated)
            elif is_tensorlist_like_type(info.type):
                was_any_mutated = False if was_mutated is None else any(was_mutated)
                check_one(info, was_any_mutated)


def hash_tensor(t: torch.Tensor) -> torch.Tensor:
    """Some inexpensive hash. Used as a quick and dirty indicator for tensor mutation"""
    return t.detach().float().mean()


def has_fake_kernel(op: torch._ops.OpOverload) -> bool:
    """If an operator (that stays alive until FakeTensorMode) has a Fake kernel.
    Don't use this if the operator decomposes before FakeTensorMode.
    """
    if can_generate_trivial_fake_impl(op):
        return True
    name = op._name
    if torch._C._dispatch_has_kernel_for_dispatch_key(
        name, "CompositeImplicitAutograd"
    ):
        return True
    opdef = torch._library.custom_ops._maybe_get_opdef(name)
    if opdef is None:
        # the non-torch.library.custom_op path
        if torch._C._dispatch_has_kernel_for_dispatch_key(
            name, "CompositeExplicitAutograd"
        ):
            return True
        entry = torch._library.simple_registry.singleton.find(name)
        if entry.fake_impl.kernel is not None:
            return True
        if torch._C._dispatch_has_kernel_for_dispatch_key(name, "Meta"):
            return True
    else:
        # the torch.library.custom_op path
        if opdef._abstract_fn is not None:
            return True
    return False


def mutated_args_kwargs(schema: _C.FunctionSchema) -> tuple[list[int], list[str]]:
    idxs = []
    keys = []
    for i, info in enumerate(schema.arguments):
        if info.alias_info is not None and info.alias_info.is_write:
            if info.kwarg_only:
                keys.append(info.name)
            else:
                idxs.append(i)
    return idxs, keys


tags_by_priority = [
    _C.Tag.needs_exact_strides,
    _C.Tag.needs_contiguous_strides,
    _C.Tag.needs_fixed_stride_order,
    _C.Tag.flexible_layout,
]


# Case 1: with_default=True (or omitted). Return type is guaranteed to be a Tag.
@overload
def get_layout_constraint_tag(
    fn: Any, *, with_default: Literal[True] = True
) -> _C.Tag: ...


# Case 2: with_default=False. Return type can be a Tag or None.
@overload
def get_layout_constraint_tag(
    fn: Any, *, with_default: Literal[False]
) -> _C.Tag | None: ...


def get_layout_constraint_tag(fn, *, with_default=True):
    for tag in tags_by_priority:
        if tag in fn.tags:
            return tag
    if with_default:
        if is_builtin(fn):
            return _C.Tag.flexible_layout
        import torch._functorch
        from torch._functorch import config

        return getattr(torch._C.Tag, config.custom_op_default_layout_constraint)
    return None


# List of random functions that should be treated as impure
_RANDOM_FUNCTIONS = {
    torch.rand,
    torch.randn,
    torch.randint,
    torch.randperm,
    torch.rand_like,
    torch.randn_like,
    torch.randint_like,
    torch.normal,
    torch.poisson,
    torch.bernoulli,
    torch.multinomial,
}


def is_impure(
    op: Callable,
    *,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    impure_random: bool = True,
) -> bool:
    """
    An operator is impure if it:
    - Mutates its inputs (has a mutable schema)
    - Has nondeterministic/random behavior that mutates RNG state
    - Is explicitly marked as effectful via torch.library._register_effectful_op

    Args:
        op: The operator to check (function, OpOverload, HigherOrderOperator, etc.)
        args: Optional arguments that would be passed to the callable
        kwargs: Optional keyword arguments that would be passed to the callable
        impure_random: Whether to treat random operations as impure (default: True)

    Returns:
        bool: True if the callable has side effects, False otherwise
    """
    # Import here to avoid circular dependencies
    from torch._higher_order_ops.effects import _get_effect
    from torch.fx.node import _side_effectful_functions

    if isinstance(op, torch._ops.OpOverloadPacket):
        if op in _side_effectful_functions:
            return True
        default = getattr(op, "default", None)
        if default is not None:
            return is_impure(
                default, args=args, kwargs=kwargs, impure_random=impure_random
            )

    if isinstance(op, torch._ops.OpOverload):
        schema = getattr(op, "_schema", None)
        if schema is not None and schema.is_mutable:
            return True

        if op in _side_effectful_functions:
            return True

        if _get_effect(op) is not None:
            return True

    if isinstance(op, torch._ops.HigherOrderOperator):
        if op in (
            torch.ops.higher_order.auto_functionalized,
            torch.ops.higher_order.auto_functionalized_v2,
        ):
            # Check if the auto-functionalized operator (the first argument) is
            # side-effectful
            if args and len(args) > 0:
                return args[0] in _side_effectful_functions

        if _get_effect(op) is not None:
            return True

        if op in _side_effectful_functions:
            return True

        return False

    # Impure since it mutates RNG state
    if impure_random and getattr(op, "_nondeterministic_seeded", False):
        return True

    # Handle Python random functions that don't have _nondeterministic_seeded
    # but still affect global RNG state (issue #151524)
    # These should be impure regardless of impure_random setting to maintain
    # consistency between eager and compiled execution
    # All random operations are impure to ensure consistent behavior
    # between eager and compiled execution, regardless of generator usage
    if op in _RANDOM_FUNCTIONS:
        return True

    schema = getattr(op, "_schema", None)
    if schema is not None and schema.is_mutable:
        return True

    if op in _side_effectful_functions:
        return True

    return False
<|endoftext|>from enum import Enum

import torch


class EffectType(Enum):
    ORDERED = "Ordered"


from torch._library.utils import RegistrationHandle


# These classes do not have side effects as they just store quantization
# params, so we don't need to mark them as ordered
skip_classes = (
    "__torch__.torch.classes.quantized.Conv2dPackedParamsBase",
    "__torch__.torch.classes.quantized.Conv3dPackedParamsBase",
    "__torch__.torch.classes.quantized.EmbeddingPackedParamsBase",
    "__torch__.torch.classes.quantized.LinearPackedParamsBase",
    "__torch__.torch.classes.xnnpack.Conv2dOpContext",
    "__torch__.torch.classes.xnnpack.LinearOpContext",
    "__torch__.torch.classes.xnnpack.TransposeConv2dOpContext",
)


class EffectHolder:
    """A holder where one can register an effect impl to."""

    def __init__(self, qualname: str):
        self.qualname: str = qualname
        self._set_default_effect()

    def _set_default_effect(self) -> None:
        self._effect: EffectType | None = None

        # If the op contains a ScriptObject input, we want to mark it as having effects
        namespace, opname = torch._library.utils.parse_namespace(self.qualname)
        split = opname.split(".")
        if len(split) > 1:
            if len(split) != 2:
                raise AssertionError(
                    f"Tried to split {opname} based on '.' but found more than 1 '.'"
                )
            opname, overload = split
        else:
            overload = ""

        if namespace == "higher_order":
            return

        opname = f"{namespace}::{opname}"
        if torch._C._get_operation_overload(opname, overload) is not None:
            # Since we call this when destroying the library, sometimes the
            # schema will be gone already at that time.
            schema = torch._C._get_schema(opname, overload)
            for arg in schema.arguments:
                if isinstance(arg.type, torch.ClassType):
                    type_str = arg.type.str()  # pyrefly: ignore[missing-attribute]
                    if type_str in skip_classes:
                        continue
                    self._effect = EffectType.ORDERED
                    return

    @property
    def effect(self) -> EffectType | None:
        return self._effect

    @effect.setter
    def effect(self, _):
        raise RuntimeError("Unable to directly set kernel.")

    def register(self, effect: EffectType | None) -> RegistrationHandle:
        """Register an effect

        Returns a RegistrationHandle that one can use to de-register this
        effect.
        """
        self._effect = effect

        def deregister_effect():
            self._set_default_effect()

        handle = RegistrationHandle(deregister_effect)
        return handle
<|endoftext|>import torch._library.autograd
import torch._library.fake_impl
import torch._library.simple_registry
import torch._library.utils
from torch._library.fake_class_registry import register_fake_class
from torch._library.triton import capture_triton, triton_op, wrap_triton
<|endoftext|># mypy: allow-untyped-defs
import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from torch import _C, _ops, autograd, Tensor
from torch.utils import _pytree

from . import utils


class InfoProtocol(Protocol):
    _backward_fn: Callable | None
    _setup_context_fn: Callable | None


@dataclasses.dataclass
class Info:
    _backward_fn: Callable | None
    _setup_context_fn: Callable | None


def make_autograd_impl(op: _ops.OpOverload, info: InfoProtocol) -> Callable:
    name: str = f"GeneratedBackwardFor_{op._namespace}_{op._opname}_{op._overloadname}"

    schema = op._schema
    is_out_op = utils.is_out(op)
    has_kwarg_only_args = utils.has_kwarg_only_args(op._schema)
    num_positional_args = sum(not a.kwarg_only for a in schema.arguments)
    has_tensorlist_like_args = any(
        utils.is_tensorlist_like_type(a.type)
        for a in (*schema.arguments, *schema.returns)
    )

    @dataclass
    class Metadata:
        keyset: _C.DispatchKeySet
        keyword_only_args: dict[str, Any]

    def redispatch_no_grad(keyset, args, kwargs):
        with _C._AutoDispatchBelowAutograd():
            result = op.redispatch(keyset & _C._after_autograd_keyset, *args, **kwargs)
            return result

    def forward(ctx, *args):
        metadata = args[-1]
        args = args[:-1]

        with _C._AutoDispatchBelowAutograd():
            keyset = metadata.keyset
            kwargs = metadata.keyword_only_args
            result = op.redispatch(keyset & _C._after_autograd_keyset, *args, **kwargs)
            if info._setup_context_fn:
                # The Dispatcher will remove args that are equal to their default
                # values from (args, kwargs). We're going to add it back so that
                # the user can access them.
                #
                # This is OK to do: The Dispatcher removed the args for serialization
                # FC/BC reasons (that is, a graph will not store args that are equal
                # to their default values), but that doesn't matter here. If the user
                # adds a new default arg, then they must update
                # their setup_context (along with the rest of their operator
                # registrations)
                args, kwargs = utils.fill_defaults(op._schema, args, kwargs)

                if has_kwarg_only_args:
                    info._setup_context_fn(
                        ctx=ctx, inputs=args, keyword_only_inputs=kwargs, output=result
                    )
                else:
                    info._setup_context_fn(ctx=ctx, inputs=args, output=result)
            return result

    def backward(ctx, *grads):
        if info._backward_fn:
            try:
                prev_needs_input_grad = ctx.needs_input_grad
                ctx.needs_input_grad = prev_needs_input_grad[:-1]
                result = info._backward_fn(ctx, *grads)
            finally:
                ctx.needs_input_grad = prev_needs_input_grad
            num_actual_inputs = len(prev_needs_input_grad) - 1
            valid_return_counts = {num_actual_inputs, num_positional_args}
            actual = len(result) if isinstance(result, tuple) else 1
            if actual not in valid_return_counts:
                expected = (
                    str(num_actual_inputs)
                    if num_actual_inputs == num_positional_args
                    else f"{num_actual_inputs} or {num_positional_args}"
                )
                raise RuntimeError(
                    f"The backward formula for {op} returned an incorrect "
                    f"number of gradients (expected {expected}, got {actual}). "
                    f"Expected one gradient for each forward input, or for "
                    f"each positional input to the operator. Use None for "
                    f"inputs that do not require a gradient."
                )
            if isinstance(result, tuple):
                extra_grads = result[num_actual_inputs:]
                if any(grad is not None for grad in extra_grads):
                    raise RuntimeError(
                        f"The backward formula for {op} returned a non-None "
                        f"gradient for an input that was not passed to the "
                        f"operator. Defaulted inputs that were not passed "
                        f"through autograd must return None."
                    )
                if has_tensorlist_like_args:
                    result = result[:num_actual_inputs]
                return (*result, None)
            return result, None
        raise RuntimeError(
            f"Trying to backward through {op} but no autograd "
            f"formula was registered. "
            f"Please use register_autograd to add one."
        )

    Generated = type(
        name,
        (autograd.Function,),
        {
            "forward": staticmethod(forward),
            "backward": staticmethod(backward),
        },
    )

    if has_tensorlist_like_args:
        Generated = supports_tensorlist(Generated)

    # The dispatcher passes any keyword-only-args as kwargs and the
    # rest of the args (even if specified as kwargs) as args.
    def autograd_impl(keyset, *args, **keyword_only_args):
        if is_out_op:
            if _C.is_grad_enabled() and _C._any_requires_grad(
                *args, **keyword_only_args
            ):
                raise RuntimeError(
                    f"{op._opname}(): functions with out=... arguments don't "
                    "support automatic differentiation, but one of the arguments "
                    "requires grad."
                )
            return redispatch_no_grad(keyset, args, keyword_only_args)

        if _C.is_grad_enabled() and _C._any_requires_grad(*args):
            result = Generated.apply(*args, Metadata(keyset, keyword_only_args))  # type: ignore[attr-defined]
        else:
            result = redispatch_no_grad(keyset, args, keyword_only_args)
        return result

    return autograd_impl


def supports_tensorlist(cls: Any) -> Any:
    """Allows a given autograd.Function class to support List[Tensor] inputs/outputs.

    Regular autograd.Function has a constraint that it only directly supports autograd for
    Tensors. Applying @supports_tensorlist enables an autograd.Function to support
    autograd for List[Tensor] inputs and outputs.
    """
    orig_forward = cls.forward
    orig_backward = cls.backward
    orig_apply = cls.apply

    @dataclass
    class TensorListMetadata:
        input_spec: _pytree.TreeSpec
        output_spec: _pytree.TreeSpec | None = None
        result_is_tuple: bool | None = None

    def new_forward(ctx, *args):
        metadata = args[-1]
        args = args[:-1]
        if not isinstance(metadata, TensorListMetadata):
            raise NotImplementedError(
                "NYI: calling supports_tensorlist autograd.Function.forward directly. "
                "You should probably be calling .apply instead. "
                "Please file an issue if not."
            )
        args = _pytree.tree_unflatten(list(args), metadata.input_spec)
        result = orig_forward(ctx, *args)
        metadata.result_is_tuple = isinstance(result, tuple)
        if not metadata.result_is_tuple:
            result = (result,)
        flat_result, output_spec = _pytree.tree_flatten(result, not_list_of_tensor)
        metadata.output_spec = output_spec

        if hasattr(ctx, "_pt_metadata"):
            raise RuntimeError(
                "Please don't set ctx._pt_metadata; PyTorch uses it to store info"
            )
        ctx._pt_metadata = metadata

        return tuple(flat_result)

    def new_backward(ctx, *grads):
        if not hasattr(ctx, "_pt_metadata"):
            raise NotImplementedError(
                "NYI: calling supports_tensorlist autograd.Function.backward directly. "
                "This will automatically get called by PyTorch autograd. "
                "Please file an issue if you need this."
            )

        metadata = ctx._pt_metadata
        grads = _pytree.tree_unflatten(list(grads), metadata.output_spec)

        # If the user's input is ([x, y, z], w),
        # then needs_input_grad is (bool, bool, bool, bool, bool).
        # We need to
        # 1. get rid of the additional bool (which comes from the extra
        # `metadata input`)
        # 2. _pytree.tree_unflatten to get the right structure.
        prev_needs_input_grad = ctx.needs_input_grad
        try:
            ctx.needs_input_grad = _pytree.tree_unflatten(
                list(ctx.needs_input_grad[:-1]), metadata.input_spec
            )
            grad_inputs = orig_backward(ctx, *grads)
        finally:
            ctx.needs_input_grad = prev_needs_input_grad

        if not isinstance(grad_inputs, tuple):
            grad_inputs = (grad_inputs,)
        # Assume that any Nones in the backward are Tensors.
        # If the forward has an arg that is [1, 2, 3], the backward should
        # return None as the grad.
        # If the forward has an arg that is [tensor, tensor], the backward
        # may return [None, None], [grad, None], [None, grad], or [grad, grad].
        flat_grad_inputs, grad_inputs_spec = _pytree.tree_flatten(
            grad_inputs, not_list_of_optional_tensor
        )
        if grad_inputs_spec != metadata.input_spec:
            raise RuntimeError(
                f"Expected the return from backward to be of the same structure "
                f"as the inputs. Got: {grad_inputs_spec} (return from backward), "
                f"{metadata.input_spec} (inputs)"
            )
        return tuple(flat_grad_inputs + [None])

    def new_apply(*args):
        flat_args, input_spec = _pytree.tree_flatten(args, is_leaf=not_list_of_tensor)
        metadata = TensorListMetadata(input_spec)
        result = orig_apply(*flat_args, metadata)  # type: ignore[misc]
        if metadata.output_spec is None:
            raise AssertionError("metadata.output_spec must not be None")
        result = _pytree.tree_unflatten(list(result), metadata.output_spec)
        if not metadata.result_is_tuple:
            if not isinstance(result, tuple):
                raise AssertionError(f"result must be tuple, got {type(result)}")
            if len(result) != 1:
                raise AssertionError(
                    f"result tuple must have length 1, got {len(result)}"
                )
            return result[0]
        return result

    cls.forward = new_forward
    cls.backward = new_backward
    cls.apply = new_apply
    return cls


def not_list_of_tensor(tree):
    if isinstance(tree, tuple):
        return False
    if isinstance(tree, list):
        return any(not isinstance(l, Tensor) for l in tree)
    return True


def not_list_of_optional_tensor(tree):
    if isinstance(tree, tuple):
        return False
    if isinstance(tree, list):
        return any(l is not None and not isinstance(l, Tensor) for l in tree)
    return True
<|endoftext|># mypy: allow-untyped-defs
import copy
import logging
from typing import Any, Protocol

import torch
from torch._library.utils import parse_namespace
from torch.utils._python_dispatch import _disable_current_modes


log = logging.getLogger(__name__)


class FakeScriptObject:
    def __init__(
        self, wrapped_obj: Any, script_class_name: str, x: torch.ScriptObject | None
    ):
        # Use object.__setattr__ to bypass our custom __setattr__ during initialization
        object.__setattr__(self, "wrapped_obj", wrapped_obj)
        object.__setattr__(self, "script_class_name", script_class_name)

        from torch._library.opaque_object import is_custom_class

        # We don't want to deepcopy when tracing with opaque objects because
        # if a mutation happens intentionally (Ex. caching in device mesh)
        # then we want it to be recorded on the real object
        real_obj = x
        if not is_custom_class(type(x)):
            try:
                with _disable_current_modes():
                    real_obj = copy.deepcopy(x)
            except (RuntimeError, TypeError) as e:
                log.warning(
                    "Unable to deepcopy the custom object %s due to %s. "
                    "Defaulting to the user given object. This might be "
                    "dangerous as side effects may be directly applied "
                    "to the object.",
                    script_class_name,
                    e,
                )

        object.__setattr__(self, "real_obj", real_obj)

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError as e:
            raise AttributeError(
                f"Tried to call __getattr__ with attr '{name}' on a FakeScriptObject, "
                "implying that you are calling this inside of a fake kernel. "
                "The fake kernel should not depend on the contents of the "
                "OpaqueObject at all, so we're erroring out. If this attr is "
                "a method or constant attribute, you can allow this member access by "
                "registering it via `register_custom_class(members=...)`."
            ) from e

    def __setattr__(self, name, value):
        raise AttributeError(
            f"Tried to call __setattr__ with attr '{name}' on a FakeScriptObject, "
            "implying that you are calling this inside of a fake kernel. "
            "The fake kernel should not depend on the contents of the "
            "OpaqueObject at all, so we're erroring out. If you need this"
            "functionality, consider creating a custom TorchBind Object instead"
            "(but note that this is more difficult)."
        )

    def __getitem__(self, key):
        # This is needed for DeviceMesh support
        return self.real_obj[key]

    def __eq__(self, other):
        if self is other:
            return True
        # Get real_obj without triggering custom __getattribute__
        self_real = object.__getattribute__(self, "real_obj")
        if isinstance(other, FakeScriptObject):
            other_real = object.__getattribute__(other, "real_obj")
            # For reference types, identity check first
            if self_real is other_real:
                return True
            # Fall back to equality check
            return self_real == other_real
        # Compare with the real object directly
        return self_real == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        # Use real_obj's hash if available, otherwise use object id
        real_obj = object.__getattribute__(self, "real_obj")
        try:
            return hash(real_obj)
        except TypeError:
            # Object is not hashable, use identity-based hash
            return id(real_obj)

    def __deepcopy__(self, memo: dict[int, Any]) -> "FakeScriptObject":
        if id(self) in memo:
            return memo[id(self)]
        new_obj = FakeScriptObject.__new__(FakeScriptObject)
        memo[id(self)] = new_obj
        object.__setattr__(
            new_obj, "wrapped_obj", copy.deepcopy(self.wrapped_obj, memo)
        )
        object.__setattr__(new_obj, "script_class_name", self.script_class_name)
        # Disable dispatch modes during deepcopy of real_obj and attribute
        # access to prevent tensor operations (e.g. storage cloning, property
        # access on DeviceMesh) from going through proxy tracing or
        # functionalization.
        with _disable_current_modes():
            new_real_obj = copy.deepcopy(self.real_obj, memo)
            object.__setattr__(new_obj, "real_obj", new_real_obj)
            for name, value in self.__dict__.items():
                if name not in ("wrapped_obj", "script_class_name", "real_obj"):
                    if isinstance(value, FakeScriptMethod):
                        object.__setattr__(
                            new_obj,
                            name,
                            FakeScriptMethod(new_obj, value.method_name, value.schema),
                        )
                    else:
                        if hasattr(new_real_obj, name):
                            object.__setattr__(
                                new_obj, name, getattr(new_real_obj, name)
                            )
                        else:
                            object.__setattr__(new_obj, name, value)
        return new_obj


def maybe_unwrap_fake_script_object(obj: Any) -> Any:
    """If obj is a FakeScriptObject, return the underlying real object."""
    if isinstance(obj, FakeScriptObject):
        return obj.real_obj
    return obj


class FakeScriptMethod:
    def __init__(
        self,
        self_fake_obj: FakeScriptObject,
        method_name: str,
        schema: torch.FunctionSchema | None,
    ):
        self.self_fake_obj = self_fake_obj
        self.method_name = method_name
        self.schema = schema

    def __call__(self, *args, **kwargs):
        from torch._higher_order_ops.torchbind import call_torchbind

        return call_torchbind(self.self_fake_obj, self.method_name, *args, **kwargs)


class HasStaticMethodFromReal(Protocol):
    @classmethod
    def from_real(cls, real_obj: torch.ScriptObject):
        pass


class FakeClassRegistry:
    def __init__(self) -> None:
        self._registered_class: dict[str, Any] = {}

    def has_impl(self, full_qualname: str) -> bool:
        return full_qualname in self._registered_class

    def get_impl(self, full_qualname: str) -> Any:
        self._check_registered(full_qualname)
        return self._registered_class[full_qualname]

    def register(self, full_qualname: str, fake_class=None) -> None:
        if self.has_impl(full_qualname):
            log.warning(
                "%s is already registered. Previous fake class is overridden with  %s.",
                full_qualname,
                fake_class,
            )
        self._registered_class[full_qualname] = fake_class

    def deregister(self, full_qualname: str) -> Any:
        if not self.has_impl(full_qualname):
            log.warning(
                "Cannot deregister %s. Please use register_fake_class to register it first."
                " Or do you dereigster it twice?",
                full_qualname,
            )
        else:
            return self._registered_class.pop(full_qualname)

    def clear(self) -> None:
        self._registered_class.clear()

    def _check_registered(self, full_qualname: str) -> None:
        if full_qualname not in self._registered_class:
            raise RuntimeError(
                f"{full_qualname} is not registered. Please use register_fake_class to register it first."
            )


global_fake_class_registry = FakeClassRegistry()


# TODO: add this check at compile time for __obj_flatten__.
def _check_valid_flat_script_obj(flat_x):
    if not isinstance(flat_x, tuple):
        raise RuntimeError("Expect flat x to be a tuple.")

    for tp in flat_x:
        if not isinstance(tp, tuple):
            raise RuntimeError("Expect flat x to be a tuple of tuples.")

        if not len(tp) == 2 or not isinstance(tp[0], str):
            raise RuntimeError(
                "Expect element of flat x to be a tuple of two elements with first element being a string"
            )


def tracing_with_real(x: torch.ScriptObject) -> bool:
    if not hasattr(x, "tracing_mode"):
        return False

    if x.tracing_mode() not in ["real", "fake"]:
        raise AssertionError(
            f"tracing_mode can be either real or fake but got {x.tracing_mode()}"
        )
    return x.tracing_mode() == "real"


def maybe_to_fake_obj(
    fake_mode,
    x: Any,
) -> FakeScriptObject | torch.ScriptObject:
    import torch.utils._pytree as pytree

    # When tracing with real mode, people should implement meta kernels that can
    # handle the case of real script object + fake tensor inputs.
    if tracing_with_real(x):
        return x

    from torch._library.opaque_object import (
        FakeOpaqueObject,
        get_opaque_obj_info,
        get_opaque_type_name,
        is_custom_class,
        OpaqueTypeStr,
    )

    x_type = type(x)
    if is_custom_class(x_type):
        type_name = OpaqueTypeStr if x is None else get_opaque_type_name(x_type)
        fake_x_wrapped = FakeScriptObject(FakeOpaqueObject(), type_name, x)

        # Set specified members onto the fake object
        opaque_info = get_opaque_obj_info(x_type)
        if opaque_info is None:
            raise AssertionError(f"opaque_info for type {x_type} must not be None")
        for attr_name in opaque_info.members:
            with _disable_current_modes():
                if not hasattr(x, attr_name):
                    raise TypeError(
                        f"Opaque object of type '{type_name}' was specified to have member "
                        f"'{attr_name}', but this doesn't actually exist in the object."
                    )
                object.__setattr__(fake_x_wrapped, attr_name, getattr(x, attr_name))

        return fake_x_wrapped
    else:
        # x.__obj_flatten__() could be calling some tensor operations inside but we don't
        # want to call these ops in surrounding dispatch modes when executing it.
        # Otherwise, for example, the fake tensor modes will error out when the tensors inside
        # script object execute some operations like clone if allow_non_fake_input flag is set.
        with _disable_current_modes():
            flat_x = x.__obj_flatten__()  # type: ignore[attr-defined]

        _check_valid_flat_script_obj(flat_x)

        with fake_mode:
            from torch._higher_order_ops.utils import _tensor_storage

            storage_map = {
                _tensor_storage(inp): i
                for i, inp in enumerate(flat_x)
                if isinstance(inp, torch.Tensor)
            }
            alias_map = {
                i: storage_map[_tensor_storage(inp)]
                for i, inp in enumerate(flat_x)
                if isinstance(inp, torch.Tensor)
                and storage_map[_tensor_storage(inp)] != i
            }
            if len(alias_map) > 0:
                log.warning(
                    "Detected script object %s has aliasing relationship among its tensors. "
                    "Flattened obj: %s. Aliasing tensor indices: %s. "
                    "This is not supported and may cause unexpected behavior.",
                    x,
                    flat_x,
                    alias_map,
                )

            # This breaks the aliasing relationship among the tensors inside the torchbind object
            # This is bad but since we don't need to preserve the aliasing relationship anyway and
            # we state clearly that aliasing relationship is not preserved in the doc so this might be OK.
            fake_flattened = pytree.tree_map_only(
                torch.Tensor,
                lambda t: torch.empty_strided(
                    t.size(),
                    t.stride(),
                    device=t.device,
                    dtype=t.dtype,
                    requires_grad=t.requires_grad,
                    layout=t.layout,
                ),
                flat_x,
            )

        fake_x = _find_fake_class_for_script_object(x).__obj_unflatten__(fake_flattened)

    fake_x_wrapped = FakeScriptObject(fake_x, x._type().qualified_name(), x)  # type: ignore[attr-defined]

    for name in x._method_names():  # type: ignore[attr-defined]
        attr = getattr(fake_x, name, None)
        if attr is not None:
            if not callable(attr):
                raise RuntimeError(f"Expect {name} to be a callable but got {attr}.")

            real_attr = getattr(x, name)  # type: ignore[attr-defined]

            # real attr sometimes is not torch.ScriptMethod thus doesn't have schema e.g. __init___ or __eq__
            method_schema: torch.FunctionSchema | None = None
            if isinstance(real_attr, torch.ScriptMethod):
                method_schema = real_attr.schema  # type: ignore[attr-defined]

            # Bypasses our custom setattr function
            object.__setattr__(
                fake_x_wrapped,
                name,
                FakeScriptMethod(fake_x_wrapped, name, method_schema),
            )
        else:
            override_skip_list = {"__obj_flatten__", "__getstate__", "__setstate__"}
            if name not in override_skip_list:
                log.warning("fake object of %s doesn't implement method %s.", x, name)
    return fake_x_wrapped


def register_fake_class(qualname, fake_class: HasStaticMethodFromReal | None = None):
    r"""Register a fake implementation for this class.

    It's in the same spirit of registering a fake implementation for
    an operator but with the difference that it
    associates a fake class with the original torch bind class (registered
    with torch::class_). In this way, torch.compile can handle them properly
    in components such as Dynamo and AOTAutograd.

    This API may be used as a decorator (see example). For the fake class, users
    are required to provide a from_real classmethod that takes a real object and
    returns an instance of the fake class. All tensors in the fake object should also
    be properly fakified with to_fake_tensor() in from_real.


    Examples:
        # For a custom class Foo defined in test_custom_class_registration.cpp:

        TORCH_LIBRARY(_TorchScriptTesting, m) {
          m.class_<TensorQueue>("_TensorQueue")
            .def(torch::init<at::Tensor>())
            .def("push", &TensorQueue::push)
            .def("pop", &TensorQueue::pop)
            .def("top", &TensorQueue::top)
            .def("size", &TensorQueue::size)
            .def("clone_queue", &TensorQueue::clone_queue)
            .def("__obj_flatten__", &TensorQueue::__obj_flatten__)
            .def_pickle(
                // __getstate__
                [](const c10::intrusive_ptr<TensorQueue>& self)
                    -> c10::Dict<std::string, at::Tensor> {
                  return self->serialize();
                },
                // __setstate__
                [](c10::Dict<std::string, at::Tensor> data)
                    -> c10::intrusive_ptr<TensorQueue> {
                  return c10::make_intrusive<TensorQueue>(std::move(data));
                });
            };
        # We could register a fake class FakeTensorQueue in Python as follows:
        import torch

        @torch._library.register_fake_class("_TorchScriptTesting::_TensorQueue")
        class FakeTensorQueue:
            def __init__(self, queue):
                self.queue = queue

            @classmethod
            def __obj_unflatten__(cls, flattened_ctx):
                return cls(**dict(ctx))

            def push(self, x):
                self.queue.append(x)

            def pop(self):
                return self.queue.pop(0)

            def size(self):
                return len(self.queue)

    In this example, the original TensorQueue need to add a __obj_flatten__ method
    to the class TensorQueue and the flattened result is passed into FakeTensorQueue's
    __obj_unflatten__ as inputs to create a fake class. This protocol allows pytorch to look
    at the contents of the script object and properly handle them in the subsystems
    like dynamo, aot_autograd or more.
    """

    def inner(fake_class: HasStaticMethodFromReal):
        ns, name = parse_namespace(qualname)

        # This also checks whether the referred torch::class_ exists.
        torch._C._get_custom_class_python_wrapper(ns, name)

        from_method = getattr(fake_class, _CONVERT_FROM_REAL_NAME, None)
        if not from_method:
            raise RuntimeError(
                f"{fake_class} doesn't define a classmethod {_CONVERT_FROM_REAL_NAME}."
            )

        if not isinstance(fake_class.__dict__[_CONVERT_FROM_REAL_NAME], classmethod):
            raise RuntimeError(
                f"{_CONVERT_FROM_REAL_NAME} method is not a classmethod."
            )

        global_fake_class_registry.register(_full_qual_class_name(qualname), fake_class)
        return fake_class

    if fake_class is None:
        return inner
    return inner(fake_class)


def deregister_fake_class(qualname):
    return global_fake_class_registry.deregister(_full_qual_class_name(qualname))


def has_fake_class(full_qualname) -> bool:
    return global_fake_class_registry.has_impl(full_qualname)


def find_fake_class(full_qualname) -> Any | None:
    if not has_fake_class(full_qualname):
        return None
    return global_fake_class_registry.get_impl(full_qualname)


def _full_qual_class_name(qualname: str) -> str:
    ns, name = parse_namespace(qualname)
    return "__torch__.torch.classes." + ns + "." + name


def _is_script_object(obj: Any) -> bool:
    return isinstance(
        obj, torch.ScriptObject
    ) and obj._type().qualified_name().startswith(  # type: ignore[attr-defined]
        "__torch__.torch.classes"
    )


# Return the namespace and class name from fully qualified name.
def _ns_and_class_name(full_qualname: str) -> tuple[str, str]:
    splits = full_qualname.split(".")
    if len(splits) != 5:
        raise AssertionError(f"Could not split {full_qualname=}, expected 5 parts")
    _torch, _torch_ns, _classes, ns, class_name = splits
    return ns, class_name


def _find_fake_class_for_script_object(x: torch.ScriptObject) -> Any:
    full_qualname = x._type().qualified_name()  # type: ignore[attr-defined]
    ns, class_name = _ns_and_class_name(full_qualname)
    fake_class = find_fake_class(full_qualname)
    if fake_class is None:
        raise RuntimeError(
            f" ScriptObject's {full_qualname} haven't registered a fake class."
            f" Please use register_fake_class({ns}::{class_name}) to annotate a fake class for the script obj."
            f" Specifically, create a python class that implements a fake version for all the methods"
            f" that're used in the program and put annotated class in the program e.g. after loading the library."
            f" The fake methods can be written in the same way as a meta kernel for an operator but need to additionally"
            f" simulate the object's states. Be sure to add a {_CONVERT_FROM_REAL_NAME} classmethod"
            f" to enable creating a fake obj from a real one."
        )
    return fake_class


_CONVERT_FROM_REAL_NAME = "__obj_unflatten__"


def _fake_obj_from_real(fake_mode, x) -> Any:
    fake_class = _find_fake_class_for_script_object(x)

    from_real_method = getattr(fake_class, _CONVERT_FROM_REAL_NAME, None)
    if not from_real_method:
        raise RuntimeError(
            f"{fake_class} must define a classmethod {_CONVERT_FROM_REAL_NAME}"
            f" that converts the real object to the fake object."
        )

    # from_real defined by user need the ctx to fakify the tensor states.
    ctx = torch._library.fake_impl.FakeImplCtx(fake_mode, None)
    with torch._library.fake_impl.set_ctx_getter(lambda: ctx):
        return fake_class.from_real(x)
<|endoftext|>import ast
import contextlib
import inspect
import logging
import threading
from collections.abc import Callable, Generator, Iterable
from typing import Any

from torch.utils._exposed_in import exposed_in

from .custom_ops import custom_op, CustomOpDef
from .infer_schema import infer_schema


logger = logging.getLogger(__name__)

triton_ops_to_kernels: dict[str, list[object]] = {}


def get_triton_kernels_for_op(name: str) -> list[object]:
    return triton_ops_to_kernels.get(name, [])


def get_inner_triton_kernels(fn: Callable[..., Any]) -> list[object]:
    """
    Inspect the source of an arbitrary callable passed to torch._library.triton_op,
    and grab all of the triton kernels that are wrapped inside of it.

    This function traces local variable assignments to handle patterns like:
        kernel_fn = _my_kernel  # global JITFunction
        wrapped = some_wrapper(kernel_fn)
        capture_triton(wrapped)[grid](...)

    It also recursively analyzes called functions to find triton kernels hidden
    behind helper function calls.

    That said, it is best effort. There are cases (e.g., recursion > MAX_RECURSION_DEPTH)
    that are not accounted for, so keep that in mind.
    """

    # prevent infinite recursion
    MAX_RECURSION_DEPTH = 5

    def find_triton_kernels(
        fn: Callable[..., Any],
        visited_fns: set[int] | None = None,
        depth: int = 0,
    ) -> list[object]:
        try:
            from triton.runtime.autotuner import Autotuner
            from triton.runtime.jit import JITFunction
        except ImportError:
            logger.warning("Triton not available, find_triton_kernels = []")
            return []

        # unwrap decorated fn's (e.g., @lru_cache) to get the original
        fn = inspect.unwrap(fn)

        # init visited set and check for cycles/depth limit
