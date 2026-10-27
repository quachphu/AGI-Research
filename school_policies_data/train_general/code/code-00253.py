    local_tensor_args = cast(tuple[object, ...], local_tensor_args)
    op_call(*local_tensor_args, **op_info.local_kwargs)

    grad_dtensor = cast(list[dtensor.DTensor], args[0])[0]
    grad_placements = grad_dtensor.placements
    mesh = grad_dtensor.device_mesh

    found_inf_placements: list[Placement] = []
    for placement in grad_placements:
        if isinstance(placement, Replicate):
            found_inf_placements.append(placement)
        else:
            found_inf_placements.append(Partial("max"))

    target_tensor = cast(torch.Tensor, args[1])
    spec = DTensorSpec(
        mesh=mesh,
        placements=tuple(found_inf_placements),
        tensor_meta=TensorMeta(
            shape=target_tensor.size(),
            stride=target_tensor.stride(),
            dtype=target_tensor.dtype,
        ),
    )
    # pyrefly: ignore [bad-argument-type]
    found_inf_dtensor = dtensor.DTensor(
        local_tensor=target_tensor,  # pyrefly: ignore [unexpected-keyword]
        spec=spec,  # pyrefly: ignore [unexpected-keyword]
        requires_grad=False,  # pyrefly: ignore [unexpected-keyword]
    )
    found_inf = found_inf_dtensor.full_tensor()
    target_tensor.copy_(found_inf)


class OpDispatcher:
    """
    Op dispatching class instance to handle args/kwargs pre-processing (un-wrapping), sharding
    propagation, redistribute local args, local compute, and post-processing (re-wrapping). It
    also handles any op specific logic if necessary.

    NOTE: Given the runtime overhead of Tensor subclass (__torch_dispatch__), the OpDispatcher
    is designed to minimize the CPU overhead by using the tricks of proper unflattening, faster
    pytree if needed, and leveraging various caching mechanisms implemented in the sharding
    propagation and redistribute modules. The CPU overhead is critical to eager mode performance,
    one need to carefully measure the CPU overhead when making significant changes to the
    OpDispatcher and ShardingPropagator.
    """

    def __init__(self) -> None:
        self.sharding_propagator = ShardingPropagator()
        # NOTE: must stay in sync with is_random_op in
        # torch/csrc/autograd/python_variable.cpp
        self._random_ops = {
            aten.native_dropout.default,
            aten.normal_.default,
            aten.rand.default,
            aten.rand_like.default,
            aten.randn.default,
            aten.randn_like.default,
            aten.randint_like.default,
            aten.randint_like.low_dtype,
            aten.randint_like.low_dtype_out,
            aten.uniform_.default,
            aten.bernoulli.default,
            aten.bernoulli_.float,
        }
        self._custom_op_handlers = {
            aten.is_same_size.default: is_same_size_handler,
            aten.is_pinned.default: is_pinned_handler,
            aten.convolution.default: convolution_handler,
            aten.convolution_backward.default: convolution_backward_handler,
            aten._amp_foreach_non_finite_check_and_unscale_.default: found_inf_reduce_handler,
            aten.as_strided.default: as_strided_handler,
            aten.argmin.default: argminmax_handler,
            aten.argmax.default: argminmax_handler,
        }

    # ********************************************************************************************
    # def dispatch(...)
    #
    # NOTE: this class no longer contains the top-level dispatch entrypoint!
    # See #167051 for details
    #
    # The entrypoint has been moved to C++, and it handles common cases and then calls back into
    # OpDispatcher python to handle corner cases.
    # See dispatchDTensorOp() defined in python_variable.cpp and called from python_arg_parser.cpp
    # ********************************************************************************************

    # This flag is used internally to control whether we treat the torch.Tensor(non-DTensor)
    # as implicitly replicated or we throw error to user.
    # NOTE: It is EXTREMELY UNSAFE to turn this flag on by default so we intentionally leave
    # it as False by default.
    @property
    def _allow_implicit_replication(self) -> bool:
        return torch._C._get_dtensor_allow_implicit_replication()

    @_allow_implicit_replication.setter
    def _allow_implicit_replication(self, value: bool) -> None:
        return torch._C._set_dtensor_allow_implicit_replication(value)

    def _propagate_op_sharding_dispatch_slow_path(
        self,
        op_call: torch._ops.OpOverload,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        op_info: OpInfo,
        # The logic here is a bit messy.  There are several reasons why the
        # C++ fastpath may have bailed out.  If we just cache missed, we will
        # come here because we need to actually calculate the real thing.
        # There's no need to have a SECOND Python cache lookup; the C++ native
        # cache completely subsumes it.  But sometimes, we will have failed
        # to compute the cache key in C++ entirely.  In this case, we DO need
        # to do a cache lookup in Python, as the missing cache key in C++
        # means we don't have access to it all.  Furthermore, without duping
        # this function, we need to do the try_cache test inside of the
        # try-except block so that either case hits the inference mode /
        # exception rewrapping case.
        #
        # This should be cleaned up.  First, ensuring the C++ codepath can
        # always compute a key will be a big help.  Second, we should properly
        # fastpath inference mode composite implicit autograd so that you
        # don't have to throw an exception even in "fastpath".
        try_cache: bool,
    ) -> object:
        # NOTE: schema should always be populated when calling this function,
        # as it's only called from C++ after unwrap_to_op_info (create_schema=True).
        # See dispatchDTensorOp in python_variable.cpp line 1453-1460.
        if op_info.schema is None:
            raise AssertionError(
                "op_info.schema should not be None in sharding propagation. "
                "This function should only be called after unwrap_to_op_info."
            )
        try:
            # We have basically inlined propagate() here, but WITHOUT the
            # output_sharding assignment
            if try_cache and not _are_we_tracing():
                result = self.sharding_propagator.propagate_op_sharding(op_info.schema)
            else:
                result = self.sharding_propagator.propagate_op_sharding_non_cached(
                    op_info.schema
                )
            if logger.handlers and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "sharding_prop MISS (C++ fast path): %s -> %s",
                    op_info.schema,
                    # pyrefly: ignore [missing-attribute]
                    result.output_spec,
                )
            return result
        except NotImplementedError:
            if torch._C._dispatch_has_kernel_for_dispatch_key(
                op_call.name(), torch._C.DispatchKey.CompositeImplicitAutograd
            ):
                # When running under inference mode, CompositeImplicitAutograd ops show up in __torch_dispatch__,
                # so we manually decompose them, here
                out = op_call.decompose(*args, **kwargs)
                if out is NotImplemented:
                    raise AssertionError from None
                return out
            else:
                raise
        except Exception as e:
            raise RuntimeError(
                f"{e}\n\nSharding propagation failed for {op_info.schema or op_call}"
            ) from e

    def _dispatch_get_local_results_slow_path(
        self,
        op_call: torch._ops.OpOverload,
        args: tuple[object, ...],
        op_info: OpInfo,
    ) -> object:
        output_sharding = op_info.output_sharding
        if output_sharding is None:
            raise AssertionError("output sharding should not be None")
        if op_info is None:
            raise AssertionError("op_info should never be None")

        # Record output placements for debugging
        debug_mode = get_active_debug_mode()
        if debug_mode is not None and output_sharding.output_spec is not None:
            debug_mode.record_output_placements(output_sharding.output_spec)

        mesh = op_info.compute_mesh
        participating = mesh._is_current_rank_part_of_mesh()
        local_results = None
        if participating:
            # computation that happens in the current rank of the mesh, normal case
            if output_sharding.needs_redistribute:
                # If sharding propagation decision needs redistribute, perform redistribute
                # on args first, which could potentially modify args (i.e. allgather certain arg)
                if output_sharding.redistribute_schema is None:
                    raise AssertionError
                with _ignore_fresh_unbacked_symbols_for_dtensor_tracing(
                    output_sharding.output_spec
                ):
                    self.redistribute_local_args(
                        op_info,
                        output_sharding.redistribute_schema,
                        output_sharding.use_val_from_redistribute_schema,
                    )

            local_tensor_args = (
                pytree.tree_unflatten(
                    cast(list[object], op_info.local_args),
                    # pyrefly: ignore [bad-argument-type]
                    op_info.args_tree_spec,
                )
                if op_info.args_tree_spec
                else op_info.local_args
            )

            # run local op computation with potentially modified args/kwargs
            local_tensor_args = cast(tuple[object, ...], local_tensor_args)
            if op_call in self._random_ops:
                if not random._rng_tracker and is_rng_supported_mesh(mesh):
                    # Default to `OffsetBasedRNGTracker` if the parallelism API did not already construct one
                    # Skip RNG state sync during tracing to avoid lazily initializing real RNG state under fake mode.
                    run_state_sync = not _are_we_tracing()
                    if not run_state_sync:
                        logger.info(
                            "DTensor RNG tracker is being lazily initialized during tracing. "
                            "RNG states may not be synchronized across ranks, which can lead "
                            "to silent incorrectness. Please call `torch.manual_seed()` with "
                            "the same seed on all ranks before compiling DTensor random ops.",
                            stacklevel=2,
                        )
                    random._rng_tracker = random.OffsetBasedRNGTracker(
                        mesh, run_state_sync
                    )

                first_arg, first_local_arg = (
                    cast(dtensor.DTensor, args[0]),
                    cast(torch.Tensor, local_tensor_args[0]),
                )

                # If the user provided a generator, we hook it up to our RNG manager, but we also pop it from kwargs
                # so the op_call does not directly use it (we want op_call to fall back to the 'default' which is
                # our RNG manager)
                maybe_user_generator = op_info.local_kwargs.pop("generator", None)
                if not (
                    maybe_user_generator is None
                    or isinstance(maybe_user_generator, torch.Generator)
                ):
                    raise AssertionError

                if (
                    random._rng_tracker
                    and not first_local_arg.is_meta
                    and random._rng_tracker.distribute_region_enabled
                ):
                    if (
                        maybe_user_generator is not None
                        or first_local_arg.device.type != "cuda"
                        or (
                            not _are_we_tracing()
                            and type(first_local_arg) is not torch.Tensor
                        )
                    ):
                        with random._rng_tracker._distribute_region(
                            first_arg._spec, generator=maybe_user_generator
                        ):
                            with _ignore_fresh_unbacked_symbols_for_dtensor_tracing(
                                output_sharding.output_spec
                            ):
                                local_results = op_call(
                                    *local_tensor_args, **op_info.local_kwargs
                                )
                    else:
                        # CUDA device without user generator, use HOP for traceability
                        if not isinstance(
                            random._rng_tracker, random.OffsetBasedRNGTracker
                        ):
                            raise AssertionError
                        start_offset_incr, end_offset_incr = (
                            random._rng_tracker._compute_rng_offsets(first_arg._spec)
                        )
                        with _ignore_fresh_unbacked_symbols_for_dtensor_tracing(
                            output_sharding.output_spec
                        ):
                            local_results = run_dtensor_rng_op(
                                start_offset_incr,
                                end_offset_incr,
                                op_call,
                                *local_tensor_args,
                                **op_info.local_kwargs,
                            )
                else:
                    # No rng_tracker, meta tensor, or distribute_region disabled
                    with _ignore_fresh_unbacked_symbols_for_dtensor_tracing(
                        output_sharding.output_spec
                    ):
                        local_results = op_call(
                            *local_tensor_args, **op_info.local_kwargs
                        )
            else:
                # normal case, run local sharded op computation
                if (
                    output_sharding.needs_redistribute
                    and output_sharding.redistribute_schema is not None
                    and output_sharding.redistribute_schema.op != op_call
                ):
                    # Op was rewritten (e.g., squeeze.default → squeeze.dims)
                    with _ignore_fresh_unbacked_symbols_for_dtensor_tracing(
                        output_sharding.output_spec
                    ):
                        local_results = output_sharding.redistribute_schema.op(
                            *local_tensor_args, **op_info.local_kwargs
                        )
                else:
                    with _ignore_fresh_unbacked_symbols_for_dtensor_tracing(
                        output_sharding.output_spec
                    ):
                        local_results = op_call(
                            *local_tensor_args, **op_info.local_kwargs
                        )

        else:
            # For a non-participating device (happens on rank that does not belong to
            # the device mesh), we do:
            #   1. if the return type is scalar, set the local result to None.
            #   2. if the return type is Tensor or List[Tensor], return empty
            #   tensor(s) with correct dtype.
            spec = output_sharding.output_spec
            ret_list = op_call._schema.returns

            if spec is None:
                # For a scalar return type, the non-participating device has None
                # as its local result
                local_results = None
            else:

                def default_tensor(spec: DTensorSpec) -> torch.Tensor:
                    if spec.tensor_meta is not None:
                        shape = spec.tensor_meta.shape
                        dtype = spec.tensor_meta.dtype
                        if len(shape) == 0:
                            # scalar tensor
                            return torch.zeros((), dtype=dtype)
                        else:
                            # non-scalar tensor
                            return torch.tensor([], dtype=dtype)
                    else:
                        raise RuntimeError(f"{spec} has no tensor metadata.")

                if isinstance(spec, DTensorSpec):
                    # return a Tensor value
                    local_results = default_tensor(spec)
                elif isinstance(spec, Sequence):
                    # return a List[Tensor] value
                    local_results = [
                        default_tensor(s) if s is not None else None for s in spec
                    ]
                    if not isinstance(local_results, list):
                        raise AssertionError
                    if None in local_results:
                        ret_type = str(ret_list[0].type)
                        raise NotImplementedError(
                            f"return type {ret_type} in DTensor op is not supported"
                        )
        return local_results

    def _dispatch_fast_path_python_tail(
        self,
        op_call: torch._ops.OpOverload,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        compute_mesh: DeviceMesh,
        output_sharding: OutputSharding,
        local_results: object,
        participating: bool,
        is_inplace_op: bool,
        is_out_variant_op: bool,
    ) -> object:
        """
        Tail of main dispatching logic, called from C++ fast path.
        """

        # Record output placements for debugging
        debug_mode = get_active_debug_mode()
        if debug_mode is not None and output_sharding.output_spec is not None:
            debug_mode.record_output_placements(output_sharding.output_spec)

        if output_sharding.output_spec is None:
            if op_call == aten.equal.default:
                # The output of the equal op is a bool, by converting it into
                # a single value tensor, we can use all-reduce with min reduce op
                # to simulate logical and.
                if not (local_results is None or isinstance(local_results, bool)):
                    raise AssertionError
                r = torch.tensor(
                    int(local_results) if local_results is not None else 1,
                    device=compute_mesh.device_type,
                )
                dist.all_reduce(r, op=dist.ReduceOp.MIN)
                local_results = bool(r.item())

        if is_inplace_op:
            # inplace op should return self instead of re-wrapping
            if output_sharding.output_spec is not None:
                output_spec = output_sharding.output_spec
                if not isinstance(output_spec, DTensorSpec):
                    raise AssertionError
                if not isinstance(args[0], dtensor.DTensor):
                    raise AssertionError

                # Fast path: placements unchanged (common case: add_, mul_, etc.)
                if args[0]._spec.placements == output_spec.placements:
                    return args[0]

                # Placement reindexed (e.g. squeeze_ removing a non-sharded
                # dim: Shard(1) → Shard(0)). No redistribution — the local
                # tensor data is unchanged, only dim indices shift.
                # strict_view=True in sharding prop prevents the illegal
                # case (squeezing a sharded dim) from reaching here.
                args[0]._spec = output_spec
                return return_and_correct_aliasing(op_call, args, kwargs, args[0])
            else:
                return None
        elif is_out_variant_op:
            # out variant could possibly have multiple out args (i.e. lu_unpack.out)
            output_specs = (
                (output_sharding.output_spec,)
                if not isinstance(output_sharding.output_spec, tuple)
                else output_sharding.output_spec
            )
            out_dts = []
            spec_idx = 0
            for argument in op_call._schema.arguments:
                if argument.is_out:
                    out_dt = cast(dtensor.DTensor, kwargs[argument.name])
                    out_dt._spec = cast(DTensorSpec, output_specs[spec_idx])
                    out_dts.append(out_dt)
                    spec_idx += 1

            if len(out_dts) < 1:
                raise AssertionError("out variant should have at least one out arg")
            return tuple(out_dts) if len(out_dts) > 1 else out_dts[0]
        else:
            if op_call != aten.equal.default:
                raise AssertionError(op_call)
            ret = self.wrap(local_results, output_sharding.output_spec)  # type: ignore[possibly-undefined]
            if participating and op_call._schema._is_view_op():
                return return_and_correct_aliasing(op_call, args, kwargs, ret)
            else:
                return ret

    @staticmethod
    def redistribute_local_args(
        op_info: OpInfo,
        suggested_input_schema: OpSchema,
        use_val_from_redistribute_schema: bool,
    ) -> None:
        debug_mode = get_active_debug_mode()

        # NOTE: it's very rare that we need to reshard kwargs so we intentionally skip it
        if op_info.args_tree_spec is not None:
            flatten_args_schema_to_reshard = tuple(
                pytree.tree_leaves(suggested_input_schema.args_schema)
            )
        else:
            flatten_args_schema_to_reshard = suggested_input_schema.args_schema

        new_local_args: list[object] = []
        for i, arg_spec in enumerate(op_info.flat_args_schema):
            reshard_arg_spec = flatten_args_schema_to_reshard[i]
            if isinstance(arg_spec, DTensorSpec):
                local_tensor = cast(torch.Tensor, op_info.local_args[i])
                if arg_spec != reshard_arg_spec:
                    redistribute_context = (
                        debug_mode.record_redistribute_calls(  # type: ignore[union-attr]
                            i, arg_spec, reshard_arg_spec
                        )
                        if debug_mode is not None
                        else contextlib.nullcontext()
                    )

                    ExplicitRedistributionContext.observe_redistribution(
                        arg_spec,
                        # pyrefly: ignore [bad-argument-type]
                        reshard_arg_spec,
                        LazyString(
                            _format_implicit_redistribution_msg,
                            op_info.schema or suggested_input_schema.op,
                        ),
                    )
                    with redistribute_context:
                        resharded_local_tensor = redistribute_local_tensor(
                            local_tensor,
                            arg_spec,
                            # pyrefly: ignore [bad-argument-type]
                            reshard_arg_spec,
                        )
                    new_local_args.append(resharded_local_tensor)
                else:
                    new_local_args.append(local_tensor)
            else:
                if use_val_from_redistribute_schema:
                    # args can be updated for view related ops, we refer to the
                    # update in redistribute_schema.
                    new_local_args.append(reshard_arg_spec)
                else:
                    new_local_args.append(arg_spec)

        # Append extra non-tensor args from rewritten schema (e.g., dims tuple).
        if use_val_from_redistribute_schema:
            for i in range(
                len(op_info.flat_args_schema), len(flatten_args_schema_to_reshard)
            ):
                new_local_args.append(flatten_args_schema_to_reshard[i])

        op_info.local_args = tuple(new_local_args)

    def unwrap_to_op_info(
        self,
        op_call: torch._ops.OpOverload,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> OpInfo:
        return self._unwrap_to_op_info_impl(op_call, args, kwargs, True)

    def _unwrap_to_op_info_impl(
        self,
        op_call: torch._ops.OpOverload,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        create_schema: bool,
    ) -> OpInfo:
        # get runtime schema info to determine whether to use pytree to flatten inputs
        runtime_schema_info = self.sharding_propagator.op_to_schema_info.get(
            op_call, None
        )
        if runtime_schema_info is None:
            runtime_schema_info = (
                self.sharding_propagator.op_to_schema_info_for_single_dim_strategy.get(
                    op_call, None
                )
            )

        # Auto-detect needs_pytree if any arg is a list/tuple containing tensors
        def _contains_tensor(arg: object) -> bool:
            if isinstance(arg, (list, tuple)):
                return any(isinstance(item, torch.Tensor) for item in arg)
            return False

        needs_pytree = (
            runtime_schema_info is not None and runtime_schema_info.needs_pytree
        ) or any(_contains_tensor(arg) for arg in args)

        if needs_pytree:
            # flatten args/kwargs when op says necessary or args contain lists/tuples
            tree_args, args_spec = pytree.tree_flatten(args)
            args_list: Sequence[object] = tree_args
        else:
            args_list, args_spec = args, None

        args_schema: list[object] = []
        kwargs_schema: dict[str, object] = {}
        local_args: list[object] = []
        local_kwargs: dict[str, object] = {}
        compute_mesh: DeviceMesh | None = None

        for arg in args_list:
            if isinstance(arg, dtensor.DTensor):
                local_args.append(arg._local_tensor)
                args_schema.append(arg._spec)
                if compute_mesh is None:
                    # record the first compute device mesh from args
                    compute_mesh = arg.device_mesh
            elif isinstance(arg, torch.Tensor):
                compute_mesh = compute_mesh or try_find_mesh_from_args(
                    op_call, args_list
                )
                args_schema.append(
                    self._try_replicate_spec_for_scalar_tensor(
                        op_call, arg, compute_mesh
                    )
                )
                local_args.append(arg)
            else:
                # non DTensor/Tensor args (i.e. int/float/bool), just add to args_schema/local_args
                args_schema.append(arg)
                local_args.append(arg)

        for k, v in kwargs.items():
            if isinstance(v, dtensor.DTensor):
                local_kwargs[k] = v._local_tensor
                kwargs_schema[k] = v._spec
                if compute_mesh is None:
                    # record the first compute device mesh from kwargs
                    compute_mesh = v.device_mesh
            elif isinstance(v, torch.Tensor):
                compute_mesh = compute_mesh or try_find_mesh_from_args(
                    op_call, args_list
                )
                kwargs_schema[k] = self._try_replicate_spec_for_scalar_tensor(
                    op_call,
                    v,
                    compute_mesh,
                )
                local_kwargs[k] = v
            else:
                # non DTensor/Tensor args (i.e. int/float/bool), just add to args_schema/local_args
                kwargs_schema[k] = v
                local_kwargs[k] = v

        if compute_mesh is None:
            raise AssertionError(
                f"found no DeviceMesh from dtensor args for {op_call}!"
            )
        op_info = OpInfo(
            compute_mesh,
            OpSchema(
                op_call,
                (
                    # pyrefly: ignore [bad-argument-type]
                    pytree.tree_unflatten(args_schema, args_spec)
                    if args_spec
                    else tuple(args_schema)
                ),
                kwargs_schema,
                schema_info=runtime_schema_info,
            )
            if create_schema
            else None,  # type: ignore[arg-type]
            args_schema,
            tuple(local_args),
            local_kwargs,
            args_spec,
        )
        return op_info

    @staticmethod
    def wrap(res: object, spec: OutputSpecType) -> object:
        if isinstance(res, torch.Tensor):
            if spec is not None:
                if not isinstance(spec, DTensorSpec):
                    raise AssertionError(
                        f"output spec does not match with output! Expected DTensorSpec, got {spec}."
                    )
                # pyrefly: ignore [bad-argument-type, bad-argument-count, unexpected-keyword]
                return dtensor.DTensor(res, spec, requires_grad=res.requires_grad)
            else:
                # if output does not have a DTensorSpec due to specific ops, it must be a scalar tensor
                if res.ndim != 0:
                    raise AssertionError("output tensor should be scalar!")
                return res
        elif isinstance(res, (list, tuple)):
            if not (spec is not None and isinstance(spec, (list, tuple))):
                raise AssertionError(
                    f"output spec does not match with output! Expected list/tuple, got {spec}."
                )
            res_list = []
            for e, s in zip(res, spec):
                # pyrefly: ignore [bad-argument-type]
                res_list.append(OpDispatcher.wrap(e, s))

            return tuple(res_list) if isinstance(res, tuple) else res_list
        else:
            # if the res contains only non tensor values (i.e. int/float/none), we simply return it
            # without rewrapping to DTensor.
            return res

    def _try_replicate_spec_for_scalar_tensor(
        self,
        op_call: torch._ops.OpOverload,
        tensor_arg: torch.Tensor,
        compute_mesh: DeviceMesh,
    ) -> DTensorSpec:
        # util function to produce a replicate spec for a scalar tensor arg/kwarg
        if tensor_arg.numel() == 1 and tensor_arg.ndim == 1:
            warnings.warn(
                "Found a non-scalar tensor with numel=1 and ndim!=0, "
                "we are implicitly creating a replicated DTensor for it. "
                "However, please consider changing it to a scalar tensor "
                "or explicitly create a DTensor under distributed environment.",
                stacklevel=2,
            )

        if tensor_arg.numel() == 1 or self._allow_implicit_replication:
            # scalar tensor can be safely treated as replicated
            replication_spec = DTensorSpec(
                compute_mesh,
                (Replicate(),) * compute_mesh.ndim,
                tensor_meta=TensorMeta(
                    shape=tensor_arg.shape,
                    stride=tensor_arg.stride(),
                    dtype=tensor_arg.dtype,
                ),
            )
        else:
            raise RuntimeError(
                f"{op_call}: got mixed torch.Tensor and DTensor, need to convert all"
                " torch.Tensor to DTensor before calling distributed operators!"
                " Please see https://docs.pytorch.org/docs/main/distributed.tensor.html#mixed-tensor-and-dtensor-operations"
                " for more details."
            )
        return replication_spec
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor
from typing import cast

import torch
import torch.distributed as dist
import torch.distributed.tensor._api as dtensor


aten = torch.ops.aten


def _requires_data_exchange(padding, dim_map) -> bool:
    # Data exchange is not need if only sharded across batch dim
    if all(x == -1 for x in dim_map[1:]):
        return False
    # TODO: whether there requires data exchange is currently determined by padding
    return padding[-1] != 0


def _is_supported(input_size, kernel_size, stride, padding, dilation):
    if dilation[-1] != 1:
        raise RuntimeError("Dilation must be 1 for tensor parallel convolution.")
    if padding[-1] != 0:
        if stride[-1] != 1:
            raise RuntimeError(
                "Stride must be 1 when there is padding for tensor parallel convolution."
            )
        if kernel_size[-1] // 2 > input_size[-1]:
            raise RuntimeError(
                "kernel_size[-1] // 2 should be less than or equal to input_size[-1] for tensor parallel convolution."
            )
    else:
        if not (input_size[-1] % stride[-1] == 0 and stride[-1] == kernel_size[-1]):
            raise RuntimeError(
                "It requires that input_size[-1] is divisible by stride[-1] and stride[-1] equals kernel_size[-1] "
                "when there is padding for tensor parallel convolution."
            )
    return True


def _ring_send_recv_construct(in_tensor, d1, d2, left, right, rank, size):
    # dist comms and reconstruct local input tensor
    send_to_right = in_tensor[..., -d1:].contiguous()
    send_to_left = in_tensor[..., :d2].contiguous()
    recv_from_right = torch.zeros_like(send_to_left)
    recv_from_left = torch.zeros_like(send_to_right)

    send_op_right = dist.P2POp(dist.isend, send_to_right, right)
    send_op_left = dist.P2POp(dist.isend, send_to_left, left)
    recv_op_right = dist.P2POp(dist.irecv, recv_from_right, right)
    recv_op_left = dist.P2POp(dist.irecv, recv_from_left, left)

    reqs = dist.batch_isend_irecv(
        [send_op_right, send_op_left, recv_op_left, recv_op_right]
    )
    for req in reqs:
        req.wait()

    if rank == 0:
        in_tensor = torch.cat([in_tensor, recv_from_right], dim=-1)
    elif rank == size - 1:
        in_tensor = torch.cat([recv_from_left, in_tensor], dim=-1)
    else:
        in_tensor = torch.cat([recv_from_left, in_tensor, recv_from_right], dim=-1)

    return in_tensor


def _ring_send_recv_aggregate(grad_in_tensor, d1, d2, left, right, rank, size):
    # dist comms and aggregate gradients for edge pixels
    send_to_right = grad_in_tensor[:, :, :, -d2:].contiguous()
    send_to_left = grad_in_tensor[:, :, :, :d1].contiguous()
    recv_from_right = torch.zeros_like(send_to_left)
    recv_from_left = torch.zeros_like(send_to_right)

    send_op_right = dist.P2POp(dist.isend, send_to_right, right)
    send_op_left = dist.P2POp(dist.isend, send_to_left, left)
    recv_op_right = dist.P2POp(dist.irecv, recv_from_right, right)
    recv_op_left = dist.P2POp(dist.irecv, recv_from_left, left)

    reqs = dist.batch_isend_irecv(
        [send_op_right, send_op_left, recv_op_left, recv_op_right]
    )
    for req in reqs:
        req.wait()

    if rank == 0:
        grad_in_tensor = grad_in_tensor[:, :, :, :-d2]
        grad_in_tensor[:, :, :, -d1:] = torch.add(
            grad_in_tensor[:, :, :, -d1:], recv_from_right
        )
    elif rank == size - 1:
        grad_in_tensor = grad_in_tensor[:, :, :, d1:]
        grad_in_tensor[:, :, :, :d2] = torch.add(
            grad_in_tensor[:, :, :, :d2], recv_from_left
        )
    else:
        grad_in_tensor = grad_in_tensor[:, :, :, d1:-d2]
        grad_in_tensor[:, :, :, -d1:] = torch.add(
            grad_in_tensor[:, :, :, -d1:], recv_from_right
        )
        grad_in_tensor[:, :, :, :d2] = torch.add(
            grad_in_tensor[:, :, :, :d2], recv_from_left
        )


def tp_convolution(
    op_call: torch._ops.OpOverload,
    local_tensor_args: tuple[object, ...],
    local_tensor_kwargs: dict[str, object],
    dim_map: list[int],
) -> object:
    if op_call != aten.convolution.default:
        raise AssertionError
    if len(local_tensor_args) != 9:
        raise AssertionError

    rank = dist.get_rank()
    size = dist.get_world_size()
    in_tensor = cast(torch.Tensor, local_tensor_args[0])
    weight = cast(torch.Tensor, local_tensor_args[1])
    stride, padding, dilation = local_tensor_args[3:6]

    if not isinstance(padding, list):
        raise AssertionError

    if not _requires_data_exchange(padding, dim_map):
        local_results = op_call(*local_tensor_args, **local_tensor_kwargs)
        return local_results
    else:
        if not _is_supported(in_tensor.shape, weight.shape, stride, padding, dilation):
            raise AssertionError(
                "tp_convolution data exchange requires supported stride/padding/dilation"
            )
        # step 0 compute the overlap pixels of the input tensor
        d = weight.shape[-1] - 1
        d1 = d // 2
        d2 = d - d1
        if d1 + d2 != d:
            raise AssertionError
        right = (rank + 1) % size
        left = (rank - 1 + size) % size

        # step1 reconstruct local input tensor
        in_tensor = _ring_send_recv_construct(
            in_tensor, d1, d2, left, right, rank, size
        )

        # step2 feed local input tensor to op_call
        local_tensor_args_list = list(local_tensor_args)
        local_tensor_args_list[0] = in_tensor
        local_tensor_args = cast(tuple[object, ...], local_tensor_args_list)
        local_results = op_call(*local_tensor_args, **local_tensor_kwargs)

        # step3 remove extra outputs from the results
        padding_w = padding[-1]
        w = local_results.size(-1)
        if rank == 0:
            local_results = local_results[..., : w - padding_w]
        elif rank == size - 1:
            local_results = local_results[..., padding_w:]
        else:
            local_results = local_results[..., padding_w : w - padding_w]

        return local_results


def tp_convolution_backward(
    op_call: torch._ops.OpOverload,
    local_tensor_args: tuple[object, ...],
    local_tensor_kwargs: dict[str, object],
    dim_map: list[int],
) -> object:
    if op_call != aten.convolution_backward.default:
        raise AssertionError
    if len(local_tensor_args) != 11:
        raise AssertionError

    rank = dist.get_rank()
    size = dist.get_world_size()
    grad_out_tensor = cast(torch.Tensor, local_tensor_args[0])
    in_tensor = cast(torch.Tensor, local_tensor_args[1])
    weight = cast(torch.Tensor, local_tensor_args[2])
    stride, padding, dilation = local_tensor_args[4:7]

    if not isinstance(padding, list):
        raise AssertionError

    if not _requires_data_exchange(padding, dim_map):
        local_results = op_call(*local_tensor_args, **local_tensor_kwargs)
        return local_results
    else:
        if not _is_supported(in_tensor.shape, weight.shape, stride, padding, dilation):
            raise AssertionError(
                "tp_convolution_backward data exchange requires supported stride/padding/dilation"
            )
        # step 0 compute the overlap pixels of the input tensor
        d = weight.shape[3] - 1
        d1 = d // 2
        d2 = d - d1
        if d1 + d2 != d:
            raise AssertionError
        right = (rank + 1) % size
        left = (rank - 1 + size) % size

        # step1 reconstruct local input tensor
        in_tensor = _ring_send_recv_construct(
            in_tensor, d1, d2, left, right, rank, size
        )

        # step2 reconstruct local gradient output tensor
        padding_w = padding[1]
        if rank == 0:
            grad_out_tensor = torch.nn.functional.pad(
                grad_out_tensor, (0, padding_w), "constant", 0
            )
        elif rank == size - 1:
            grad_out_tensor = torch.nn.functional.pad(
                grad_out_tensor, (padding_w, 0), "constant", 0
            )
        else:
            grad_out_tensor = torch.nn.functional.pad(
                grad_out_tensor, (padding_w, padding_w), "constant", 0
            )

        # step3 feed local input tensor to op_call
        local_tensor_args_list = list(local_tensor_args)
        local_tensor_args_list[0] = grad_out_tensor
        local_tensor_args_list[1] = in_tensor
        local_tensor_args = cast(tuple[object, ...], local_tensor_args_list)
        local_results = op_call(*local_tensor_args, **local_tensor_kwargs)

        # step4 aggregate gradients for edge pixels
        grad_in_tensor = local_results[0]
        if grad_in_tensor is not None:
            grad_in_tensor = _ring_send_recv_aggregate(
                grad_in_tensor, d1, d2, left, right, rank, size
            )
            local_results = list(local_results)
            local_results[0] = grad_in_tensor

        local_results = cast(tuple[object, ...], local_results)

        return local_results


def convolution_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    # extract local tensor and sharding infos to an OpInfo
    op_info = dtensor.DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)

    # sharding propagation
    dtensor.DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    if output_sharding is None:
        raise AssertionError("output sharding should not be None")
    output_spec = output_sharding.output_spec
    if not isinstance(output_spec, dtensor.DTensorSpec):
        raise AssertionError

    # local propagation
    local_results = tp_convolution(
        op_call,
        tuple(op_info.local_args),
        op_info.local_kwargs,
        output_spec.dim_map,
    )

    return dtensor.DTensor._op_dispatcher.wrap(local_results, output_spec)


def convolution_backward_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    # Redistribute grad_output tensor to the same placement as input tensor
    # pyrefly: ignore [bad-assignment]
    args = list(args)
    if not (
        isinstance(args[0], dtensor.DTensor) and isinstance(args[1], dtensor.DTensor)
    ):
        raise AssertionError
    # pyrefly: ignore [unsupported-operation]
    args[0] = args[0].redistribute(args[1].device_mesh, args[1].placements)
    args = tuple(args)

    # extract local tensor and sharding infos to an OpInfo
    op_info = dtensor.DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)

    # sharding propagation
    dtensor.DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    if output_sharding is None:
        raise AssertionError("output sharding should not be None")
    if not isinstance(op_info.flat_args_schema[0], dtensor.DTensorSpec):
        raise AssertionError

    # local propagation
    local_results = tp_convolution_backward(
        op_call,
        tuple(op_info.local_args),
        op_info.local_kwargs,
        op_info.flat_args_schema[0].dim_map,
    )

    return dtensor.DTensor._op_dispatcher.wrap(
        local_results, output_sharding.output_spec
    )
<|endoftext|># mypy: allow-untyped-defs
import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.tensor._dtensor_spec as dtensor_spec
from torch._C._distributed_c10d import _resolve_process_group
from torch._logging import warning_once
from torch.distributed._functional_collectives import _are_we_tracing
from torch.distributed._local_tensor import (
    local_tensor_mode,
    maybe_run_for_local_tensor,
)
from torch.distributed.device_mesh import _mesh_resources, DeviceMesh
from torch.distributed.distributed_c10d import (
    broadcast,
    get_group_rank,
    get_rank,
    GroupName,
    ProcessGroup,
    scatter,
    Work,
)
from torch.fx.experimental.symbolic_shapes import guard_or_false
from torch.types import IntLikeType


logger = logging.getLogger(__name__)

# Opaque types must be registered before defining schemas that reference them,
# so the schema parser recognizes the type names and uses PyObjectType (which
# wraps Python objects as ConcretePyObjectHolder) instead of AnyType (which
# calls toTypeInferredIValue and fails for Python-only opaque types).
from torch.distributed.device_mesh import _register_distributed_opaque_types


_register_distributed_opaque_types()

_dtensor_lib = torch.library.Library("_dtensor", "FRAGMENT")
_dtensor_lib.define(
    "mesh_get_process_group("
    "torch.distributed.device_mesh.DeviceMesh mesh, int dim"
    ") -> torch.distributed.distributed_c10d.ProcessGroup"
)


@torch.library.impl("_dtensor::mesh_get_process_group", "CompositeExplicitAutograd")
def _mesh_get_process_group_impl(mesh, dim):
    return mesh.get_group(dim)


@torch.library.register_fake("_dtensor::mesh_get_process_group")
def _mesh_get_process_group_fake(mesh, dim):
    from torch._library.fake_class_registry import maybe_unwrap_fake_script_object

    real_mesh = maybe_unwrap_fake_script_object(mesh)
    return real_mesh.get_group(dim)


@torch.library.register_fake("_dtensor::shard_dim_alltoall")
def _shard_dim_alltoall_meta(
    input, gather_dim, shard_dim, group_name: GroupName | ProcessGroup
):
    if isinstance(group_name, str):
        # pyrefly: ignore[bad-argument-type]  # pyrefly bug
        group_name = _resolve_process_group(group_name)
    group_size = group_name.size()
    stacked_list = [torch.empty_like(input) for _ in range(group_size)]
    group_rank = get_group_rank(group_name, get_rank())

    cat_tensor = torch.cat(stacked_list, dim=gather_dim)
    # pyrefly: ignore [unsupported-operation]
    chunk_size = cat_tensor.size(shard_dim) // group_size
    chunk = torch.narrow(cat_tensor, shard_dim, group_rank * chunk_size, chunk_size)
    return chunk.contiguous()


def _shard_dim_alltoall_backward(ctx, grad_output):
    return (
        torch.ops._dtensor.shard_dim_alltoall.default(
            grad_output.contiguous(), ctx.shard_dim, ctx.gather_dim, ctx.group_name
        ),
        None,
        None,
        None,
    )


def _shard_dim_alltoall_setup_context(ctx, inputs, output):
    input, gather_dim, shard_dim, group_name = inputs
    ctx.gather_dim = gather_dim
    ctx.shard_dim = shard_dim
    ctx.group_name = group_name


torch.library.register_autograd(
    "_dtensor::shard_dim_alltoall",
    _shard_dim_alltoall_backward,
    setup_context=_shard_dim_alltoall_setup_context,
)


def shard_dim_alltoall(input, gather_dim, shard_dim, mesh, mesh_dim):
    if mesh.device_type == "cpu" and local_tensor_mode() is None:
        # Gloo does not support alltoall, so falling back to allgather + chunk
        warning_once(
            logger,
            "CPU process group does not support alltoall yet, falling back with allgather + chunk!",
        )
        out = funcol.all_gather_single(input, gather_dim, (mesh, mesh_dim))
        if isinstance(out, funcol.AsyncCollectiveTensor):
            # stick to the same behavior for the alltoall case, remove this once we enable alltoall async
            out = out.wait()
        from torch.distributed.tensor.placement_types import Shard

        out = Shard._custom_chunk(out, mesh.size(mesh_dim), dim=shard_dim)[
            mesh.get_local_rank(mesh_dim)
        ]
        return out.contiguous()

    group = funcol._resolve_group((mesh, mesh_dim))
    # TODO: enable async op for shard_dim_alltoall
    return torch.ops._dtensor.shard_dim_alltoall(
        input, gather_dim, shard_dim, funcol._group_or_group_name(group)
    )


def mesh_scatter(
    output: torch.Tensor,
    scatter_list: list[torch.Tensor],
    mesh: DeviceMesh,
    mesh_dim: int = 0,
    async_op: bool = False,
    *,
    group_src: int = 0,
) -> Work | None:
    """
    scatter a list of tensors to a device mesh dimension. We by default
    use the first rank of the mesh dimension as the source of truth, i.e
    for a 2d mesh [[0, 1], [2, 3]], if we scatter on mesh_dim = 1, we will
    scatter the tensor list on rank 0 to rank 0/1, and tensor list on rank
    2 to rank 2/3.

    Args:
        output (torch.Tensor): the tensor to receive the scattered list.
        scatter_list (List[torch.Tensor]): the tensor list to be scattered.
        mesh_dim (int, optional): indicate which mesh dimension we want
            to scatter on, we by default choose the first rank on the
            mesh dimension as source of truth.

    Keyword args:
        group_src (int, optional): the group rank of the source data for the
        logical/global tensor, on the specific mesh dimension. By default, we
        use ``group_rank=0`` on each DeviceMesh dimension as the source data
        to preserve the single-device semantic. If passing ``None`` explicitly,
        this method simply uses its local data with no communication.

    Returns:
        A :class:`Work` object
    """
    # TODO: Ideally we should use the meta tensor way
    # (to register a meta kernel for the collective op)
    # so that it would avoid the communication. Need to
    # remove the check below once that is done.
    if output.is_meta:
        return None
    dim_group = mesh.get_group(mesh_dim)
    if not isinstance(dim_group, ProcessGroup):
        raise AssertionError

    if group_src == get_rank(dim_group):
        fut = scatter(
            output,
            scatter_list=scatter_list,
            group=dim_group,
            async_op=async_op,
            group_src=group_src,
        )
    else:
        fut = scatter(
            output,
            scatter_list=None,
            group=dim_group,
            async_op=async_op,
            group_src=group_src,
        )

    return fut


def mesh_broadcast(
    tensor: torch.Tensor,
    mesh: DeviceMesh,
    mesh_dim: int = 0,
    async_op: bool = False,
    *,
    group_src: int = 0,
) -> Work | None:
    """
    broadcast the tensor to a device mesh dimension. We by default
    use the first rank of the mesh dimension as the source of truth, i.e
    for a 2d mesh [[0, 1], [2, 3]], if we broadcast on mesh_dim = 1, we will
    broadcast the tensor on rank 0 to rank 0/1, and tensor on rank 2
    to rank 2/3.

    Args:
        tensor (torch.Tensor): tensor to broadcast.
        mesh_dim (int, optional): indicate which mesh dimension we want
            to scatter on, we by default choose the first rank on the
            mesh dimension as source of truth.

    Keyword args:
        group_src (int, optional): the group rank of the source data for the
        logical/global tensor, on the specific mesh dimension. By default, we
        use ``group_rank=0`` on each DeviceMesh dimension as the source data
        to preserve the single-device semantic. If passing ``None`` explicitly,
        this method simply uses its local data with no communication.

    Returns:
        A :class:`Work` object
    """
    # TODO: Ideally we should use the meta tensor way
    # (to register a meta kernel for the collective op)
    # so that it would avoid the communication. Need to
    # remove the check below once that is done.
    if tensor.is_meta:
        return None
    dim_group = mesh.get_group(mesh_dim)
    if not isinstance(dim_group, ProcessGroup):
        raise AssertionError

    return broadcast(tensor, group=dim_group, async_op=async_op, group_src=group_src)


@maybe_run_for_local_tensor
def pad_tensor(
    tensor: torch.Tensor, pad_dim: int, pad_size: IntLikeType
) -> torch.Tensor:
    # During tracing, always emit the pad op even when pad_size=0 so all
    # ranks produce identical FX graph structure (SPMD).
    # In eager with concrete pad_size=0, guard_or_false returns True and we
    # skip the no-op pad. Check _are_we_tracing() first to avoid
    # guard_or_false creating a guard that concretizes symbolic pad sizes
    # during make_fx tracing.
    if isinstance(pad_size, int):
        # Fast path: avoids _are_we_tracing() which is costly at compile
        # time due to multiple C++ dispatch mode checks.
        if pad_size == 0:
            return tensor
    elif not _are_we_tracing() and guard_or_false(pad_size == 0):
        return tensor
    if _are_we_tracing():
        from torch.fx.experimental.symbolic_shapes import has_free_unbacked_symbols

        if has_free_unbacked_symbols(tensor):
            pad_shape: list[IntLikeType] = list(tensor.shape)
            pad_shape[pad_dim] = pad_size
            return torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=pad_dim)

    pad = [0, 0] * (tensor.ndim - pad_dim)
    pad[-1] = pad_size  # pyrefly: ignore[unsupported-operation]
    return torch.nn.functional.pad(tensor, pad)


@maybe_run_for_local_tensor
def unpad_tensor(
    tensor: torch.Tensor, pad_dim: int, pad_size: IntLikeType
) -> torch.Tensor:
    # During tracing, always emit the narrow op even when pad_size=0 so all
    # ranks produce identical FX graph structure (SPMD).
    # In eager with concrete pad_size=0, guard_or_false returns True and we
    # skip the no-op narrow. Check _are_we_tracing() first to avoid
    # guard_or_false creating a guard that concretizes symbolic pad sizes
    # during make_fx tracing.
    if isinstance(pad_size, int):
        # Fast path: avoids _are_we_tracing() which is costly at compile
        # time due to multiple C++ dispatch mode checks.
        if pad_size == 0:
            return tensor
    elif not _are_we_tracing() and guard_or_false(pad_size == 0):
        return tensor
    return tensor.narrow(
        pad_dim,
        start=0,
        length=tensor.size(pad_dim) - pad_size,
    )


def fill_empty_tensor_to_shards(
    shards: list[torch.Tensor], shard_dim: int, num_empty_tensors: int
) -> list[torch.Tensor]:
    if num_empty_tensors == 0:
        return shards
    tensor_size = list(shards[0].size())
    tensor_size[shard_dim] = 0
    tensor = shards[0].new_zeros(tensor_size)
    shards.extend(tensor for _ in range(num_empty_tensors))
    return shards


def check_tensor_meta(
    local_tensor, check_shape_stride=False
) -> Optional["dtensor_spec.TensorMeta"]:
    local_metadata = {
        "dtype": local_tensor.dtype,
        "requires_grad": local_tensor.requires_grad,
    }

    if check_shape_stride:
        local_metadata.update(
            {"shape": local_tensor.shape, "stride": local_tensor.stride()}
        )

    gathered_metadata = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(gathered_metadata, local_metadata)

    # Check if metadata is consistent across ranks
    if not all(meta == local_metadata for meta in gathered_metadata):
        raise ValueError(
            "Inconsistent tensor metadata (including shape and stride) across ranks."
        )
    return None


def spec_to_bytes(spec: "dtensor_spec.DTensorSpec") -> int:
    if spec.tensor_meta is None:
        raise AssertionError("spec should have tensor meta defined!")
    return spec.tensor_meta.dtype.itemsize * math.prod(spec.shape)


@dataclass
class MeshTopoInfo:
    """
    Mesh information for collective cost estimation
    """

    mesh: DeviceMesh
    mesh_dim_devices: list[int]
    mesh_dim_bandwidth: list[float]
    mesh_dim_latency: list[float]

    @staticmethod
    @lru_cache(None)
    def build_from_mesh(mesh: DeviceMesh) -> "MeshTopoInfo":
        # Generate mesh topology info for intra-host/inter-host communication pattern
        # Note that we made bunch of assumptions for simplicity:
        # 1. we assume the mesh is homogeneous, and it's gpu/nccl model
        # 2. we assume gpu arch is Ampere or Hopper
        # 3. we assume collectives are all ring base algo for now
        num_devices_per_host = _mesh_resources.num_devices_per_host(mesh.device_type)
        # the base bw number (intra-node), GB/s
        base_bw = 87.7
        mesh_dim_bandwidth = [base_bw] * mesh.ndim
        # the latency in terms of us (intra-node, nv-link)
        mesh_dim_latency = [0.6] * mesh.ndim
        mesh_dim_devices = [1] * mesh.ndim

        total_num_devices = 1
        for mesh_dim in reversed(range(mesh.ndim)):
            num_devices = mesh.size(mesh_dim)
            mesh_dim_devices[mesh_dim] = num_devices
            total_num_devices *= num_devices
            if total_num_devices > num_devices_per_host:
                # magic number for inter-host communication bandwidth/latency factor
                # This number assumes latest GPU arch, i.e. Ampere or Hopper
                # TODO: see if we need to tweak this or offer a way for user
                # to specify the bandwidths/latency
                mesh_dim_bandwidth[mesh_dim] *= 0.22
                # set to ethernet latency for inter-host
                mesh_dim_latency[mesh_dim] = 2.7

        return MeshTopoInfo(
            mesh, mesh_dim_devices, mesh_dim_bandwidth, mesh_dim_latency
        )


def allgather_cost(bytes_gb: float, mesh_topo: MeshTopoInfo, mesh_dim: int) -> float:
    num_devices_on_mesh_dim = mesh_topo.mesh_dim_devices[mesh_dim]
    mesh_dim_bandwidth = mesh_topo.mesh_dim_bandwidth[mesh_dim]
    num_hops = num_devices_on_mesh_dim - 1
    # base latency + comm latency
    latency = 6.6 + num_hops * mesh_topo.mesh_dim_latency[mesh_dim]  # us
    bw = (bytes_gb * num_hops / num_devices_on_mesh_dim) / mesh_dim_bandwidth  # s
    return latency + bw * 1e6  # rescale to us


def allreduce_cost(bytes_gb: float, mesh_topo: MeshTopoInfo, mesh_dim: int) -> float:
    num_devices_on_mesh_dim = mesh_topo.mesh_dim_devices[mesh_dim]
    mesh_dim_bandwidth = mesh_topo.mesh_dim_bandwidth[mesh_dim]
    # allreduce have almost 2x comm bytes compare to allgather/reduce_scatter
    num_hops = 2 * (num_devices_on_mesh_dim - 1)

    latency = 6.6 + num_hops * mesh_topo.mesh_dim_latency[mesh_dim]
    bw = (bytes_gb * num_hops / num_devices_on_mesh_dim) / mesh_dim_bandwidth
    return latency + bw * 1e6


def reduce_scatter_cost(
    bytes_gb: float,
    mesh_topo: MeshTopoInfo,
    mesh_dim: int,
) -> float:
    num_devices_on_mesh_dim = mesh_topo.mesh_dim_devices[mesh_dim]
    mesh_dim_bandwidth = mesh_topo.mesh_dim_bandwidth[mesh_dim]
    num_hops = num_devices_on_mesh_dim - 1
    # base latency + comm latency
    latency = 6.6 + num_hops * mesh_topo.mesh_dim_latency[mesh_dim]
    bw = (bytes_gb * num_hops / num_devices_on_mesh_dim) / mesh_dim_bandwidth
    return latency + bw * 1e6


def _compute_placement_transition_cost(
    current_placement: "dtensor_spec.Placement",
    target_placement: "dtensor_spec.Placement",
    mesh_topo: MeshTopoInfo,
    mesh_dim: int,
    comm_bytes_gb: float,
) -> tuple[float, float]:
    """
    Compute the cost of transitioning from one placement to another on a single mesh dimension.

    Args:
        current_placement: The current placement on the mesh dimension.
        target_placement: The target placement on the mesh dimension.
        mesh_topo: Mesh topology information for cost estimation.
        mesh_dim: The mesh dimension where the transition happens.
        comm_bytes_gb: The communication bytes in GB for this step.

    Returns:
        A tuple of (cost, updated_comm_bytes_gb):
            - cost: The communication cost for this transition. ``float("inf")``
              means that the strategy planner must not select the transition
              implicitly; the explicit redistribution API may still support it.
            - updated_comm_bytes_gb: The updated communication bytes after this step.
    """
    if current_placement == target_placement:
        return 0.0, comm_bytes_gb

    num_devices_on_mesh_dim = mesh_topo.mesh_dim_devices[mesh_dim]

    # NOTE: is_shard() does not match _StridedShard; see _is_shard_like().
    # Safe today: redistribute_cost bails with inf when shard_order is None.
    if current_placement.is_shard() and target_placement.is_replicate():
        # allgather gives larger comm bytes
        comm_bytes_gb *= num_devices_on_mesh_dim
        return allgather_cost(comm_bytes_gb, mesh_topo, mesh_dim), comm_bytes_gb
    elif current_placement.is_shard() and target_placement.is_shard():
        # should be alltoall comm, since we haven't implement it yet, add 1.0 as penalty
        # to favor allgather instead
        # TODO: add alltoall_cost
        return allgather_cost(comm_bytes_gb, mesh_topo, mesh_dim) + 1.0, comm_bytes_gb
    elif current_placement.is_partial() and target_placement.is_replicate():
        return allreduce_cost(comm_bytes_gb, mesh_topo, mesh_dim), comm_bytes_gb
    elif current_placement.is_partial() and target_placement.is_shard():
        cost = reduce_scatter_cost(comm_bytes_gb, mesh_topo, mesh_dim)
        # after reduce_scatter the comm bytes for further collectives halved.
        comm_bytes_gb /= num_devices_on_mesh_dim
        return cost, comm_bytes_gb
    elif current_placement.is_shard() and target_placement.is_partial():
        # Shard -> Partial("sum") is a valid explicit redistribution, but it
        # materializes a logical-shape tensor and copies the local shard into it.
        # The cost model accounts only for communication, so assigning zero cost
        # would make the strategy planner over-prefer this memory- and copy-heavy
        # transition. Exclude it from implicit strategy selection instead.
        return float("inf"), comm_bytes_gb
    elif current_placement.is_partial() and target_placement.is_partial():
        # we already handled the == case at the top, and we ban converting between partial types.
        return float("inf"), comm_bytes_gb
    elif current_placement.is_replicate() and target_placement.is_shard():
        comm_bytes_gb /= num_devices_on_mesh_dim
        return 0.0, comm_bytes_gb

    return 0.0, comm_bytes_gb


def one_step_redistribute_cost(
    current_spec: "dtensor_spec.DTensorSpec",
    target_spec: "dtensor_spec.DTensorSpec",
) -> float:
    """
    Calculate the cost of a single redistribution step between two DTensorSpecs.

    This function computes the communication cost for a one-step redistribution
    where the current and target specs differ by exactly one placement on one
    mesh dimension.

    Args:
        current_spec: The current DTensorSpec.
        target_spec: The target DTensorSpec.

    Returns:
        The communication cost for this step (float("inf") if invalid).
    """
    if current_spec.mesh != target_spec.mesh:
        return float("inf")

    if current_spec.placements == target_spec.placements:
        return 0.0

    # Find the mesh dimension that differs
    mesh_dim = -1
    current_placement = None
    target_placement = None
    for i, (cur, tgt) in enumerate(
        zip(current_spec.placements, target_spec.placements)
    ):
        if cur != tgt:
            if mesh_dim != -1:
                # More than one dimension differs - not a single step
                raise ValueError(
                    "one_step_redistribute_cost expects specs that differ by exactly one placement"
                )
            mesh_dim = i
            current_placement = cur
            target_placement = tgt

    if mesh_dim == -1:
        return 0.0

    if current_placement is None or target_placement is None:
        raise AssertionError

    mesh_topo = MeshTopoInfo.build_from_mesh(current_spec.mesh)
    comm_bytes_gb = (
        spec_to_bytes(current_spec) / current_spec.num_shards / 1024 / 1024 / 1024
    )

    cost, _ = _compute_placement_transition_cost(
        current_placement, target_placement, mesh_topo, mesh_dim, comm_bytes_gb
    )
    return cost


def redistribute_cost(
    current_spec: "dtensor_spec.DTensorSpec",
    target_spec: "dtensor_spec.DTensorSpec",
) -> float:
    """
    This function returns the cost of redistribute from current to target DTensorSpec.

    NOTE:
    1. Only consider communication cost here, since computation costs for redistribute
       are quite trivial (i.e. we only need to narrow or simple division)
    2. Only consider redistribute cost on same mesh, cross mesh communication cost is
       not quite needed for operator strategy estimation/selection.
    """
    if current_spec.mesh != target_spec.mesh:
        # make infinite cost if meshes are not same
        # TODO: see if we want to support this once there's cross mesh communication
        return float("inf")
    if current_spec.is_replicated():
        # short-cut: comm cost is 0 if current spec is already full replication
        return 0.0

    # TODO(zpcore): test placements with _StridedShard if we replace shard_order
    # with _StridedShard.
    if (
        current_spec.placements == target_spec.placements
        and current_spec.shard_order == target_spec.shard_order
    ):
        return 0.0

    # For sub-meshes, ranks not participating in the mesh should not compute
    # redistribution costs. Return 0 since they won't actually participate.
    if not current_spec.mesh._is_current_rank_part_of_mesh():
        return 0.0

    mesh_topo = MeshTopoInfo.build_from_mesh(current_spec.mesh)
    cost = 0.0
    comm_bytes_gb = (
        spec_to_bytes(current_spec) / current_spec.num_shards / 1024 / 1024 / 1024
    )
    # Transformation that considered for redistribute cost:
    # 1. allgather 2. alltoall
    # 3. allreduce 4. reduce_scatter
    from torch.distributed._functional_collectives import _are_we_tracing
    from torch.distributed.tensor._redistribute import (
        _gen_transform_infos,
        _gen_transform_infos_non_cached,
    )

    # TODO(zpcore): Support _StridedShard redistribution. Remove the temporary
    # fix, which is to prevent StridedShard erroring out.
    if current_spec.shard_order is None or target_spec.shard_order is None:
        return float("inf")

    # No redistribution needed when placements are already identical.
    # This also prevents potential failures in _gen_transform_infos for certain configurations
    # (e.g., sub-meshes) where finding a transform path between identical states may error out.
    # TODO(zpcore): test placements with _StridedShard if we replace shard_order
    # with _StridedShard.
    if (
        current_spec.placements == target_spec.placements
        and current_spec.shard_order == target_spec.shard_order
    ):
        return cost

    if _are_we_tracing():
        transform_infos = _gen_transform_infos_non_cached(current_spec, target_spec)
    else:
        transform_infos = _gen_transform_infos(current_spec, target_spec)
    for transform_info in transform_infos:
        if current_spec.tensor_meta is None:
            raise AssertionError("spec should have tensor meta defined!")
        current = transform_info.src_dst_placements[0]
        target = transform_info.src_dst_placements[1]
        mesh_dim = transform_info.mesh_dim
        step_cost, comm_bytes_gb = _compute_placement_transition_cost(
            current, target, mesh_topo, mesh_dim, comm_bytes_gb
        )
        if step_cost == float("inf"):
            return float("inf")
        cost += step_cost

    return cost
<|endoftext|># mypy: allow-untyped-defs
"""
Decomposition-based sharding propagation for DTensor.

When an operator doesn't have a registered sharding strategy, we derive one by
tracing through its decomposition. The decomposed ops (which do have strategies)
determine how placements propagate through the original op.
"""

from __future__ import annotations

import itertools
from typing import Any, TYPE_CHECKING

import torch
from torch._decomp import decomposition_table
from torch.distributed._functional_collectives import _are_we_tracing
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import OpSchema, OpStrategy, RuntimeSchemaInfo
from torch.distributed.tensor._utils import try_find_mesh_from_args
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Placement,
    Replicate,
    Shard,
)
from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode
from torch.utils._python_dispatch import TorchDispatchMode


def _infer_schema_info_from_op(op: OpOverload) -> RuntimeSchemaInfo:
    """Infer RuntimeSchemaInfo from an operator's schema for decomposition ops"""
    schema = op._schema

    # Find first non-tensor positional arg index
    static_argnum = None
    for i, arg in enumerate(schema.arguments):
        if arg.kwarg_only:
            break
        if arg.type.kind() != "TensorType" and static_argnum is None:
            static_argnum = i
            break

    # Find keyword-only args that aren't tensors
    kwarg_only_names = []
    for arg in schema.arguments:
        if arg.kwarg_only and arg.type.kind() != "TensorType":
            kwarg_only_names.append(arg.name)

    kwargs = {}
    if static_argnum is not None:
        kwargs["static_argnum"] = static_argnum
    if kwarg_only_names:
        # pyrefly: ignore [unsupported-operation]
        kwargs["static_kwargkey"] = kwarg_only_names

    # pyrefly: ignore [bad-argument-type]
    return RuntimeSchemaInfo(**kwargs)


from torch.utils._pytree import tree_any, tree_flatten, tree_map, tree_map_only


if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch._ops import OpOverload
    from torch.distributed.tensor._sharding_prop import ShardingPropagator


def _extract_input_specs(op_schema: OpSchema) -> tuple[DTensorSpec | object, ...]:
    return op_schema.args_schema + tuple(op_schema.kwargs_schema.values())


class PlacementTrackingMode(TorchDispatchMode):
    """
    TorchDispatchMode that tracks DTensor placements through op execution.

    Used during decomposition tracing: intercepts each op, propagates sharding
    via the ShardingPropagator, and records output placements on the result tensors.
    """

    def __init__(self, sharding_prop: ShardingPropagator, mesh: DeviceMesh):
        super().__init__()
        self.sharding_prop = sharding_prop
        self.mesh = mesh

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        args_schema, kwargs_schema = tree_map(
            lambda x: getattr(x, "_spec", x) if isinstance(x, torch.Tensor) else x,
            (args, kwargs or {}),
        )

        if not tree_any(
            lambda x: isinstance(x, DTensorSpec), (args_schema, kwargs_schema)
        ):
            raise NotImplementedError(f"No DTensorSpec found in args/kwargs for {func}")

        # Set schema_info so the LRU cache key includes static args
        op_schema = OpSchema(func, args_schema, kwargs_schema)
        schema_info = self.sharding_prop.op_to_schema_info.get(func)
        if schema_info is None:
            schema_info = (
                self.sharding_prop.op_to_schema_info_for_single_dim_strategy.get(func)
            )
        if schema_info is not None:
            op_schema.schema_info = schema_info
            op_schema._recompute_comparison_key()

        if _are_we_tracing():
            output_sharding = self.sharding_prop.propagate_op_sharding_non_cached(
                op_schema
            )
        else:
            output_sharding = self.sharding_prop.propagate_op_sharding(op_schema)

        if (
            output_sharding.needs_redistribute  # pyrefly: ignore [missing-attribute]
            and (
                redistribute_schema
                := output_sharding.redistribute_schema  # pyrefly: ignore [missing-attribute]
            )
            is not None
        ):
            # a pure .needs_redistribute check is too broad; we want to ban redistribution,
            # but this flag is set for view ops that convert global shape -> local shape args.
            # During decomposition tracing on meta tensors at global shape, the shape adjustment
            # is irrelevant — only reject true redistribution.
            for orig, desired in zip(
                op_schema.args_spec,
                redistribute_schema.args_spec,  # pyrefly: ignore [missing-attribute]
            ):
                if orig.placements != desired.placements:
                    raise RuntimeError(
                        f"Decomposition requires redistribution for {func}"
                    )

        out = func(*args, **kwargs)
        # pyrefly: ignore [missing-attribute]
        self._record_output_specs(out, output_sharding.output_spec)
        return out

    def _record_output_specs(self, output: Any, output_spec: DTensorSpec | Any) -> None:
        if isinstance(output, torch.Tensor) and output_spec is not None:
            output._spec = output_spec  # pyrefly: ignore [missing-attribute]
        elif isinstance(output, (tuple, list)) and isinstance(
            output_spec, (tuple, list)
        ):
            for t, s in zip(output, output_spec):
                self._record_output_specs(t, s)


class DecompShardingStrategy:
    """
    Generates sharding strategies for ops by tracing through their decompositions.

    For each candidate input placement combination, runs the decomposition on meta
    tensors under PlacementTrackingMode to determine the output placement. These
    single-dimension strategies are then expanded to the full mesh.
    """

    def __init__(self, sharding_prop: ShardingPropagator):
        self.sharding_prop = sharding_prop
        # Cache fake meshes per device type to avoid repeated allocation.
        # A fake size-1 mesh ensures identical strategy computation across all ranks
        # during decomposition tracing, avoiding potential SPMD divergence.
        # False negatives are avoided (all sizes % 1 == 0), while false positives
        # are caught on expansion to the real, multi-dim device mesh.
        self._fake_meshes: dict[str, DeviceMesh] = {}

    def _get_fake_mesh(self, device_type: str) -> DeviceMesh:
        fake_mesh = self._fake_meshes.get(device_type)
        if fake_mesh is None:
            fake_mesh = DeviceMesh(device_type, [0], _init_backend=False, _rank=0)
            self._fake_meshes[device_type] = fake_mesh
        return fake_mesh

    @staticmethod
    def has_decomp(op: OpOverload) -> bool:
        # Check if op has a decomposition (explicit or CIA)
        return op in decomposition_table or op._can_decompose()

    def ensure_schema_info(self, op: OpOverload) -> None:
        """
        Register schema_info for decomposition op on first invocation.
        Needed for correct shard prop cache key.
        """
        if op not in self.sharding_prop.op_to_schema_info:
            schema_info = _infer_schema_info_from_op(op)
            self.sharding_prop.op_to_schema_info[op] = schema_info

    def propagate_strategy(
        self,
        op_schema: OpSchema,
        output_tensor_meta: TensorMeta | Sequence[TensorMeta | None] | None = None,
    ) -> OpStrategy | None:
        if not tree_any(
            lambda x: isinstance(x, DTensorSpec),
            (op_schema.args_schema, op_schema.kwargs_schema),
        ):
            return None

        candidate_placements = self._get_candidate_placements(op_schema)
        mesh = try_find_mesh_from_args(
            op_schema.op,
            op_schema.args_schema + tuple(op_schema.kwargs_schema.values()),
        )

        fake_mesh = self._get_fake_mesh(mesh.device_type)
        single_dim_strategies = []
        output_placements: list[Placement | tuple[Placement, ...]] = []
        for input_placements in candidate_placements:
            try:
                output = self._propagate_through_decomp(
                    op_schema,
                    input_placements,
                    fake_mesh,
                )
            except NotImplementedError:
                return None
            except GuardOnDataDependentSymNode:
                return None
            except (RuntimeError, KeyError, IndexError):
                # TODO(pianpwk): RuntimeError is raised when redistribution is detected; switch to a custom error type
                # Runtime/KeyError/IndexError can also occur in view ops
                continue

            output_placements = (
                [output] if not isinstance(output, tuple) else list(output)
            )
            single_dim_strategies.append(output_placements + list(input_placements))

        if not single_dim_strategies:
            raise AssertionError(
                "Sharding propagation should have produced at least Replicate() strategy"
            )

        n_outputs = len(output_placements)
        strategy_schema = self.sharding_prop._wrap_with_op_strategy(op_schema)
        # Import here to avoid circular import at module load time
        from torch.distributed.tensor._ops.utils import expand_to_full_mesh_op_strategy

        return expand_to_full_mesh_op_strategy(
            mesh,
            strategy_schema,
            single_dim_strategies,
            input_index=n_outputs,
            output_tensor_meta=output_tensor_meta,
        )

    def _propagate_through_decomp(
        self,
        op_schema: OpSchema,
        placement: tuple[Placement | None],
        mesh: DeviceMesh,
    ) -> Placement | tuple[Placement, ...]:
        op = op_schema.op
        if op in decomposition_table:
            decomp_fn = decomposition_table[op]
        elif op._can_decompose():
            decomp_fn = op.decompose
        else:
            raise NotImplementedError(f"No decomposition found for {op}")

        placement_iter = iter(placement)

        def to_meta(x):
            p = next(placement_iter)
            if isinstance(x, DTensorSpec):
                # pyrefly: ignore [missing-attribute]
                meta = torch.empty(x.shape, dtype=x.tensor_meta.dtype, device="meta")
                # pyrefly: ignore [missing-attribute]
                meta._spec = DTensorSpec(mesh, (p,), tensor_meta=x.tensor_meta)
                return meta
            return x

        # Disable LocalTensorMode during decomposition tracing to prevent
        # interference with meta tensor operations
        from torch.distributed._local_tensor import maybe_disable_local_tensor_mode

        with maybe_disable_local_tensor_mode():
            # Create meta tensors and run decomposition outside LocalTensorMode
            args_meta = tree_map(to_meta, op_schema.args_schema)
            kwargs_meta = tree_map(to_meta, op_schema.kwargs_schema)

            with PlacementTrackingMode(self.sharding_prop, mesh):
                output = decomp_fn(*args_meta, **kwargs_meta)

        def get_placement(t):
            if isinstance(t, torch.Tensor):
                spec = getattr(t, "_spec", None)
                return spec.placements[0] if spec else None
            return None

        result = tree_map(get_placement, output)
        if isinstance(result, (tuple, list)):
            flat = [p for p in result if p is not None]
            return flat[0] if len(flat) == 1 else tuple(flat)
        return result

    @staticmethod
    def _get_candidate_placements(
        op_schema: OpSchema,
    ) -> list[tuple[Placement | None]]:
        tensor_specs = _extract_input_specs(op_schema)
        flat_specs, _ = tree_flatten(list(tensor_specs))

        # Step 1: Collect unique placements across all DTensorSpec inputs
        all_placements: set[Placement] = {Replicate()}
        tree_map_only(
            DTensorSpec,
            lambda spec: all_placements.update(spec.placements),
            flat_specs,
        )

        # Step 2: For each input, use the placement set, but expand Shard/StridedShard to all tensor dims
        candidates: list[list[Placement | None]] = []
        for spec in flat_specs:
            if not isinstance(spec, DTensorSpec):
                candidates.append([None])
            else:
                options = set(all_placements)
                for p in all_placements:
                    if isinstance(p, _StridedShard):
                        options |= {
                            _StridedShard(i, split_factor=p.split_factor)
                            for i in range(spec.ndim)
                        }
                    elif isinstance(p, Shard):
                        options |= {Shard(i) for i in range(spec.ndim)}
                candidates.append(list(options))

        # pyrefly: ignore [bad-argument-type, no-matching-overload]
        return list(itertools.product(*candidates))
<|endoftext|>import operator
from functools import reduce
from typing import cast

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.tensor._api as dtensor
from torch.distributed.tensor._op_schema import OutputSharding
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Partial,
    Placement,
    Replicate,
    Shard,
)


# Mapping from argmin/argmax ops to their corresponding value ops (min/max)
_ARGMINMAX_REDUCTION_OPS = {
    torch.ops.aten.argmax.default: torch.max,
    torch.ops.aten.argmin.default: torch.min,
}


def _get_output_sharding(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> OutputSharding:
    """Get the output sharding for the given op."""
    op_info = dtensor.DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)
    dtensor.DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    if output_sharding is None:
        raise AssertionError("output sharding should not be None")
    return output_sharding


def _prep_arguments(
    op_call_repr: str,
    args: tuple[object, ...],
    kwargs: dict[str, object] | None,
) -> tuple[
    torch.Tensor,
    torch.Size,
    "torch.distributed.device_mesh.DeviceMesh",
    tuple[Placement, ...],
    int | None,
    bool,
]:
    """
    Prepare arguments for nonlinear reduction ops.

    Returns:
        local_tensor: The local tensor to operate on
        global_shape: The global shape of the DTensor
        device_mesh: The device mesh
        placements: The placements tuple
        dim: The reduction dimension (can be None)
        keepdim: Whether to keep the reduced dimension
    """
    input_dtensor = cast(dtensor.DTensor, args[0])
    dim: int | None = None
    keepdim: bool = False

    if not isinstance(input_dtensor, dtensor.DTensor):
        raise NotImplementedError
    if len(args) > 1:
        dim = cast(int, args[1])
    if len(args) > 2:
        keepdim = cast(bool, args[2])
    if kwargs:
        if "dim" in kwargs:
            dim = cast(int, kwargs["dim"])
        if "keepdim" in kwargs:
            keepdim = cast(bool, kwargs["keepdim"])
    device_mesh = input_dtensor.device_mesh
    placements = input_dtensor.placements

    # check for partial placements and handle it as a replicate.
    if any(isinstance(p, Partial) for p in placements):
        target_placements = [
            Replicate() if isinstance(p, Partial) else p for p in placements
        ]
        input_dtensor = input_dtensor.redistribute(
            device_mesh=device_mesh, placements=target_placements
        )
        placements = input_dtensor.placements
    local_tensor = input_dtensor.to_local()
    global_shape = input_dtensor.shape

    return local_tensor, global_shape, device_mesh, placements, dim, keepdim


def _get_expected_shape(
    local_tensor: torch.Tensor, dim: int | None, keepdim: bool
) -> torch.Size:
    """Compute the expected output shape after reduction."""
    input_shape = list(local_tensor.shape)
    if dim is None:
        expected_shape = (
            torch.Size([1] * len(input_shape)) if keepdim else torch.Size([])
        )
    elif keepdim:
        if input_shape:
            input_shape[dim] = 1
        expected_shape = torch.Size(input_shape)
    else:
        if input_shape:
            input_shape.pop(dim)
        expected_shape = torch.Size(input_shape)

    return expected_shape


def _collect_shard_mesh_dims(
    op_call_repr: str,
    local_tensor: torch.Tensor,
    placements: tuple[Placement, ...],
    dim: int | None,
) -> list[int]:
    """Collect mesh dimensions that are sharded along the reduction dimension."""
    shard_mesh_dims: list[int] = []
    for mesh_dim, p in enumerate(placements):
        if isinstance(p, Shard):
            if dim is None or p.dim == (dim if dim >= 0 else local_tensor.ndim + dim):
                shard_mesh_dims.append(mesh_dim)
        elif isinstance(p, _StridedShard):
            raise NotImplementedError(f"{op_call_repr} does not support _StridedShard!")
    return shard_mesh_dims


def _convert_to_global_idxs(
    local_idx: torch.Tensor,
    global_shape: torch.Size,
    device_mesh: "torch.distributed.device_mesh.DeviceMesh",
    placements: tuple[Placement, ...],
    dim: int | None,
) -> tuple[int, torch.Tensor]:
    """Convert local indices to global indices."""
    local_shape, global_offset = compute_local_shape_and_global_offset(
        global_shape, device_mesh, placements
    )

    if dim is None:
        # Convert flat local index → flat global index using arithmetic ops
        # instead of torch.unravel_index, which doesn't support SymInt shapes.
        gathered_idxs = torch.zeros_like(local_idx)
        remaining = local_idx
        for i in range(len(local_shape)):
            local_stride = reduce(operator.mul, local_shape[i + 1 :], 1)
            global_stride = reduce(operator.mul, global_shape[i + 1 :], 1)
            coord = remaining // local_stride
            remaining = remaining % local_stride
            gathered_idxs = gathered_idxs + (coord + global_offset[i]) * global_stride
        gather_dim = 0
    else:
        gather_dim = dim
        gathered_idxs = local_idx + global_offset[dim]
    return gather_dim, gathered_idxs


def _gather_tensors(
    gather_dim: int,
    gathered_idxs: torch.Tensor,
    local_redux: torch.Tensor,
    device_mesh: "torch.distributed.device_mesh.DeviceMesh",
    shard_mesh_dims: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gather the min or max of the tensors and their corresponding indices.

    Args:
        gather_dim: The dim to stack the collected min/max tensors.
        gathered_idxs: The local tensor holding the corresponding indices.
        local_redux: The local tensor holding the operator's value i.e. min/max.
        device_mesh: Device mesh of the DTensor.
        shard_mesh_dims: List of mesh dimensions that are sharded.

    Returns:
        All gathered tensors (gathered_redux, gathered_idxs) of the reducing operator.
    """
    gathered_redux = local_redux
    for mesh_dim in shard_mesh_dims:
        gathered_redux = funcol.all_gather_single(
            gathered_redux,
            gather_dim=gather_dim,
            group=(device_mesh, mesh_dim),
        )
        gathered_idxs = funcol.all_gather_single(
            gathered_idxs,
            gather_dim=gather_dim,
            group=(device_mesh, mesh_dim),
        )
    return gathered_redux, gathered_idxs


def argminmax_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    """
    Handler for aten.argmin.default and aten.argmax.default ops.

    This is a pure function handler that doesn't require instantiation.
    """
    if op_call not in _ARGMINMAX_REDUCTION_OPS:
        raise NotImplementedError(f"Unsupported reduction op: {op_call}")

    local_tensor, global_shape, device_mesh, placements, dim, keepdim = _prep_arguments(
        str(op_call), args, kwargs
    )
    output_sharding = _get_output_sharding(op_call, args, kwargs)

    expected_shape = _get_expected_shape(local_tensor, dim, keepdim)
    shard_mesh_dims = _collect_shard_mesh_dims(
        str(op_call), local_tensor, placements, dim
    )

    # Compute local reduction
    if dim is None:
        val_op = _ARGMINMAX_REDUCTION_OPS[op_call]
        # unsqueeze scalars to 1-d so they can be allgathered
        local_redux = val_op(local_tensor).unsqueeze(0)
        local_idx = op_call(local_tensor).unsqueeze(0)
    else:
        val_op = _ARGMINMAX_REDUCTION_OPS[op_call]
        local_redux, local_idx = val_op(local_tensor, dim=dim, keepdim=True)

    if not shard_mesh_dims:
        return dtensor.DTensor._op_dispatcher.wrap(
            local_idx.reshape(expected_shape), output_sharding.output_spec
        )

    gather_dim, gathered_idxs = _convert_to_global_idxs(
        local_idx, global_shape, device_mesh, placements, dim
    )
    gathered_redux, gather_idxs = _gather_tensors(
        gather_dim, gathered_idxs, local_redux, device_mesh, shard_mesh_dims
    )
    # Select the rank with the best value; use dim=0 when dim was None since
    # the scalars were unsqueezed to 1-d for gathering
    select_dim = 0 if dim is None else dim
    rank_winner = op_call(gathered_redux, select_dim, True)
    final_idx = torch.gather(gather_idxs, dim=gather_dim, index=rank_winner)

    return dtensor.DTensor._op_dispatcher.wrap(
        final_idx.reshape(expected_shape), output_sharding.output_spec
    )


def minmax_dim_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    """
    Handler for aten.min.dim and aten.max.dim ops.

    This is a pure function handler that doesn't require instantiation.
    """
    local_tensor, global_shape, device_mesh, placements, dim, keepdim = _prep_arguments(
        str(op_call), args, kwargs
    )
    output_sharding = _get_output_sharding(op_call, args, kwargs)

    expected_shape = _get_expected_shape(local_tensor, dim, keepdim)
    shard_mesh_dims = _collect_shard_mesh_dims(
        str(op_call), local_tensor, placements, dim
    )

    # Compute local reduction - min/max with dim always requires dim
    if dim is None:
        raise AssertionError
    local_redux, local_idx = op_call(local_tensor, dim=dim, keepdim=True)

    if not shard_mesh_dims:
        return dtensor.DTensor._op_dispatcher.wrap(
            (
                local_redux.reshape(expected_shape),
                local_idx.reshape(expected_shape),
            ),
            output_sharding.output_spec,
        )

    gather_dim, gathered_idxs = _convert_to_global_idxs(
        local_idx, global_shape, device_mesh, placements, dim
    )

    gathered_redux, gather_idxs = _gather_tensors(
        gather_dim, gathered_idxs, local_redux, device_mesh, shard_mesh_dims
    )
    # The op_call here is min/max with dim which returns (values, indices)
    final_redux, rank_winner = op_call(gathered_redux, dim, True)
    final_idx = torch.gather(gather_idxs, dim=gather_dim, index=rank_winner)

    return dtensor.DTensor._op_dispatcher.wrap(
        (
            final_redux.reshape(expected_shape),
            final_idx.reshape(expected_shape),
        ),
        output_sharding.output_spec,
    )
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import contextlib
import warnings
from collections.abc import Sequence
from logging import getLogger
from typing import Optional

import torch
from torch.distributed._local_tensor import maybe_run_for_local_tensor
from torch.distributed.device_mesh import _get_device_handle, DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor.placement_types import _StridedShard, Shard
from torch.types import IntLikeType


logger = getLogger(__name__)

__all__ = [
    "is_rng_supported_mesh",
    "manual_seed",
    "OffsetBasedRNGTracker",
]

_rng_tracker: Optional["_RNGStateTracker"] = None


def is_rng_supported_mesh(device_mesh: DeviceMesh) -> bool:
    """Checks if the current device of ``device_mesh`` supports DTensor's random APIs.
    Currently DTensor Random APIs only supports cuda/cuda-like devices. We suggest
    users call this API to test the availability before using our random APIs.

    Args:
        device_mesh (:class:`DeviceMesh`): The device mesh on which we check if the
            random ops APIs are supported.

    Returns:
        A bool value. True if ``device_mesh`` supports DTensor Random APIs; False otherwise.

    .. warning::
        Currently we only support correct RNG on cuda/cuda-like devices.
    """
    device_handle = _get_device_handle(device_mesh.device_type)
    if device_handle and hasattr(device_handle, "set_rng_state"):
        return True
    else:
        # TODO: Logs way too much
        warnings.warn(
            f"DTensor random operators may not have complete support on {device_mesh.device_type} device mesh",
            stacklevel=2,
        )
        return False


def manual_seed(seed: int, device_mesh: DeviceMesh) -> None:
    """Sets the seed for generating random numbers for the calling rank.

    Args:
        seed (int): The desired seed.
        device_mesh (:class:`DeviceMesh`): The device mesh to set the seed. It is
            required that the ``device_mesh`` include the calling rank. This is
            to ensure that the SPMD region maintains a synchronous RNG state, which
            means no ranks should be initialized with values other than ``seed``.

    Returns:
        None

    .. warning::
        :func:`manual_seed` does not check the ``seed`` value correctness. Users must
        ensure on their own that the value passed in is the desired ``seed`` for ranks
        within ``device_mesh``.
        If ``device_mesh`` is a sub-mesh and the calling rank is not a part of it,
        ``manual_seed`` will throw an error.
        Current implementation only supports a GPU device mesh.
    """
    if not is_rng_supported_mesh(device_mesh):
        warnings.warn(
            "DTensor manual_seed() may not have complete support "
            f"on {device_mesh.device_type} device mesh",
            stacklevel=2,
        )
        return

    # TODO: deprecate this API, but also need to ensure we disable broadcast for PP case, and that's currently
    # bundled together with this API.  See torchtitan/distributed/utils.py:set_determinism
    # warnings.warn(
    #     "DTensor manual_seed() is deprecated, since DTensor no longer maintains a separate copy of generator state. "
    #     "Use `torch.manual_seed` instead"
    # )
    # Note: we still need to ensure setting `run_state_sync=False` to support the pp case

    # instantiate a RNG tracker if haven't. By default DTensor uses an
    # OffsetBasedRNGTracker to perform random operators.
    global _rng_tracker
    if not _rng_tracker:
        _rng_tracker = OffsetBasedRNGTracker(device_mesh, run_state_sync=False)

    if device_mesh.get_coordinate() is None:
        raise RuntimeError(
            "manual_seed requires the current rank to be a part of the device mesh "
            "otherwise DTensor RNG state on the rank will not be initialized and "
            "the behavior of DTensor random ops is undefined."
        )

    # DTensor no longer maintains a copy of rng state. manual seed on dtensor is the same thing
    # as manual seed on torch.
    #
    # torch.manual_seed will handle LocalTensor mode correctly by
    # iterating through all ranks if seed is a LocalIntNode.
    torch.manual_seed(seed)


class _PhiloxState:
    """
    Convenience accessor for interpreting the packed bits of (seed: uint64, offset: uint64) in the philox state,
    which for some reason is actually exposed as a size-16 uint8 tensor.

    The state is always moved to .cpu since it is necessary for it to be on CPU before applying it back to a generator.
    """

    def __init__(self, state: torch.Tensor):
        self._state = state.to("cpu")

    @property
    def state(self):
        return self._state

    @property
    def offset(self) -> torch.Tensor:
        return self._state[8:].view(dtype=torch.int64)

    @offset.setter
    def offset(self, offset: torch.Tensor) -> None:
        if offset.numel() != 1:
            raise AssertionError
        self._state[8:] = offset.view(torch.uint8)

    @property
    def seed(self) -> torch.Tensor:
        return self._state[:8].view(dtype=torch.uint64)

    @seed.setter
    def seed(self, seed: torch.Tensor) -> None:
        if seed.numel() != 1:
            raise AssertionError
        self._state[:8] = seed.view(torch.uint8)


class _RNGStateTracker:
    """
    _RNGStateTracker stores Random Number Generator (RNG) state (a ByteTensor object)
    in a dict, mapping from a corresponding tag to each state tensor. It also provides
    a set of convenient utility methods to help access/modify the state tensors. The most
    important interface is _distribute_region which will be used when DTensor executes
    a random op (an operator that calls RNG).
    """

    def __init__(self, device: torch.device):
        self._device = device
        self._device_handle = _get_device_handle(self._device.type)
        if not (self._device_handle and self._device_handle.is_available()):
            raise RuntimeError(
                f"{self.__class__.__name__} instantiation requires the presence of "
                f"{device.type} device but couldn't find."
            )
        self._use_distribute_region = True

    @property
    def distribute_region_enabled(self) -> bool:
        return self._use_distribute_region

    @distribute_region_enabled.setter
    def distribute_region_enabled(self, value) -> None:
        self._use_distribute_region = value

    def _distribute_region(
        self, spec: DTensorSpec, generator: torch.Generator | None = None
    ):
        pass

    def _manual_seed(self, parallel_seed: int) -> None:
        pass


class OffsetBasedRNGTracker(_RNGStateTracker):
    """
    This subclass of ``_RNGStateTracker`` defines the default policy of how RNG states
    should be shared and synchronized among all ranks to respect the semantics of DTensor
    random operators.

    note: _RNGStateTracker only supports cuda/cuda-like device.
    """

    def __init__(
        self,
        device_mesh: DeviceMesh,
        run_state_sync: bool = True,
    ):
        super().__init__(_resolve_device(device_mesh=device_mesh))
        if self._device_handle is None:
            raise AssertionError
        # DTensor RNG tracker so far only supports CUDA/CUDA-like devices
        if self._device.type == "cpu":
            raise RuntimeError(
                f"{self.__class__.__name__} instantiation requires the presence of "
                f"CUDA/CUDA-like/XPU device. Got {self._device.type} instead."
            )

        if run_state_sync:
            rng_state = self._get_device_state()
            # synchronize RNG state using rank 0's current one
            torch.distributed.broadcast(rng_state, 0)
            my_rng_state = self._get_device_state()
            if not all(my_rng_state == rng_state):
                logger.warning(
                    "DTensor is synchronizing RNG states of every rank with the state from rank 0. "
                    "This behavior is deprecated. "
                    "Please call `torch.manual_seed()` on every rank that participates in SPMD DTensor Operations with "
                    "the same seed. If using Pipeline Parallelism, each pipeline state would use a different seed, "
                    "but all ranks belonging to one pipeline stage would use the same seed."
                )
            self._set_device_state(rng_state)

    def _get_device_state(self) -> torch.Tensor:
        if self._device.type == "hpu":
            self._device_handle.set_rng_ctx("philox")
        rng_state = self._device_handle.get_rng_state().to(self._device)
        if self._device.type == "hpu":
            self._device_handle.unset_rng_ctx("philox")
        return rng_state

    def _set_device_state(self, state: torch.Tensor):
        # It seems that the underlying generator wants a cpu tensor but the dtensor code expects `_get_device_state`
        # to convert to a 'device' tensor, probably because we may use it with our backend comms for sync/debug
        # for now, we just convert back to cpu here to make sure it always works.
        if self._device.type == "hpu":
            self._device_handle.set_rng_ctx("philox")
        self._device_handle.set_rng_state(state.to("cpu"))
        if self._device.type == "hpu":
            self._device_handle.unset_rng_ctx("philox")

    @contextlib.contextmanager
    def _distribute_region(
        self, spec: DTensorSpec, generator: torch.Generator | None = None
    ):
        from torch.distributed._local_tensor import maybe_enable_local_tracker

        if local_tracker_context := maybe_enable_local_tracker(
            self._device.type, self.distribute_region_enabled, spec, generator
        ):
            with local_tracker_context:
                yield
            return

        # regular (non-LocalTensor) mode
        if generator is not None:
            # This is a little hacky, but for any user-passed generator, we store its state under a unique key,
            # not because we need to keep a copy of it but because its the easiest way to make it work with the
            # existing set/get APIs. We also ensure we remove it from rng_states after each _distribute_region.
            state = _PhiloxState(generator.get_state())
        else:
            state = _PhiloxState(self._get_device_state())

        if self.distribute_region_enabled:
            if self._device.type == "hpu":
                self._device_handle.set_rng_ctx("philox")
            old_offset = state.offset.clone()
            self._set_pre_op_offset(state, spec)
            with torch.random.fork_rng(
                devices=[self._device], device_type=self._device.type
            ):
                if self._device_handle is None:
                    raise AssertionError
                self._device_handle.set_rng_state(state.state)
                try:
                    yield  # execute the region code
                finally:
                    # update offset to synchronize among ranks
                    self._set_post_op_offset(state, spec, old_offset)
            if self._device.type == "hpu":
                self._device_handle.unset_rng_ctx("philox")
        else:
            yield

        if generator is not None:
            # ensure we (a) propagate the state advancement back to the user's RNG so its visible and impacts any future
            # usage of that RNG (dtensor or non-dtensor), (b) drop it from our own cache so that if the user updates
            # the seed value in their rng and uses it with DTensor again, we always use the latest value
            generator.set_state(state.state)
        else:
            self._set_device_state(state.state)

    def _set_pre_op_offset(self, state: _PhiloxState, spec: DTensorSpec) -> None:
        """Set the starting RNG offset for current device's local shard before actual
        op execution. The pre_op_offset value should start from the current RNG offset
        and increment by the size of local shard until it reaches the size of the whole
        DTensor. For different ranks that hold the same DTensor shard, their pre_op_offset
        will be the same.

        Args:
            state (:class:`Tensor`): The generator state to modify
            spec (:class:`DTensorSpec`): the spec of the DTensor object on which
                we prepare the offset for running random ops.

        Returns:
            None

        .. warning::
            Note that, current implementation does not consider DTensor's continguity.

        Example:
            take a DTensor of shape [8, 16] as an example. Assume that the DTensor
            is placed on a device mesh with placements ([Shard(1), Replicate(), Shard(0)]),
            and the mesh is:
                [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
            ``spec.mesh.get_coordinate()`` provides the coordinate of the current rank
            in the mesh. For example, the coordinate of rank 5 is (1, 0, 1).

            Another concept to introduce besides rank coordinate is shard coordinate.
            Each rank holds a local shard of the DTensor. In the example, the DTensor
            is partitioned into 4 [4, 8] shards. The first shard has 2 replicas and
            rank 0 (coord (0, 0, 0)) and rank 2 (coord (0, 1, 0)) have 1 replica each.
            That being said, the local shard on rank 0 and rank 2 correspond to the same
            shard of the DTensor. To denote each DTensor shard, we use a shard coordinate
            (in the example, it will be a tuple (i, j) where shard (i, j) has the slice
            DTensor[4 * i : 4 * (i + 1), 8 * j : 8 * (j + 1)], 0 <= i < 2, 0 <= j < 2).

            Once we have rank coordinate and shard coordinate, we can calculate on each rank
            what shard of the DTensor the rank holds, with the help of dim_map. The dim_map
            of the above DTensor is [2, 0] so the shard coordinate of a rank with rank coord
            (x, y, z) is simply (z, x) by taking(rank_coord[dim_map[0]],rank_coord[dim_map[1]]).
            Following this calculation,
            rank 0 and rank 2 holds the shard of coord (0, 0);
            rank 1 and rank 3 holds the shard of coord (0, 1);
            rank 4 and rank 6 holds the shard of coord (1, 0);
            rank 5 and rank 7 holds the shard of coord (1, 1);

            The last value to calculate before obtaining the starting offset is the shard linear index.
            The starting offset for each rank will be its shard_linear_index * local_tensor_numel.
        """
        start_offset_incr, _ = self._compute_rng_offsets(spec)
        state.offset = state.offset + start_offset_incr

    def _set_post_op_offset(
        self, state: _PhiloxState, spec: DTensorSpec, old_offset: torch.Tensor
    ) -> None:
        """Sets the RNG to a synchronized state after running the local random op. Every
        rank should set its RNG offset to `old_offset + DTensor.numel()` where old_offset is
        the offset before calling `set_pre_op_offset` i.e. the offset before running DTensor
        random ops.

        Args:
            state (:class:`Tensor`): The generator state to modify.
            spec (:class:`DTensorSpec`): the spec of the DTensor object on which
                we post-process the offset for running random ops.

        Returns:
            None
        """
        _, end_offset_incr = self._compute_rng_offsets(spec)
        state.offset = old_offset + end_offset_incr

    def _compute_rng_offsets(self, spec: DTensorSpec) -> tuple[int, int]:
        """Compute the RNG offset increments for a distributed random op.

        These values are derived from mesh topology, placements, and tensor shape,
        and are static for a given compiled graph. They can be burned into the graph
        as integer constants rather than keeping the DTensorSpec around at runtime.

        Returns:
            (start_offset_incr, end_offset_incr) — both aligned to multiples of 4.
        """
        from torch.distributed.tensor._ops.utils import prod

        mesh = spec.mesh
        mesh_coordinate = [mesh._sym_get_coordinate(i) for i in range(mesh.ndim)]

        shard_idx_by_dim, total_num_shards_by_dim = _calc_shard_info(
            mesh_coordinate, spec
        )
        shard_linear_idx = self._calc_shard_linear_idx(
            shard_idx_by_dim, total_num_shards_by_dim
        )
        local_size = prod(_calc_first_shard_size(spec))
        # pytorch: offset must be multiple of 4
        # source: aten/src/ATen/cuda/CUDAGeneratorImpl.cpp
        start_offset_incr = (shard_linear_idx * local_size + 3) // 4 * 4
        end_offset_incr = (prod(spec.shape) + 3) // 4 * 4

        return start_offset_incr, end_offset_incr  # type: ignore[bad-return]

    def _calc_shard_linear_idx(
        self, shard_coord: Sequence[IntLikeType], shard_size: Sequence[IntLikeType]
    ) -> IntLikeType:
        return _calc_shard_linear_idx(shard_coord, shard_size)


def _calc_first_shard_size(spec: DTensorSpec) -> list[int]:
    local_size_on_rank_0 = list(spec.shape)
    for idx, placement in enumerate(spec.placements):
        if isinstance(placement, Shard | _StridedShard):
            mesh_dim_size = spec.mesh.size(idx)
            shard_dim = placement.dim
            local_size_on_rank_0[shard_dim], _ = placement._local_shard_size_and_offset(
                spec.shape[shard_dim],
                mesh_dim_size,
                0,
            )
    return local_size_on_rank_0


def _calc_shard_info(
    mesh_coordinate: Sequence[IntLikeType], spec: DTensorSpec
) -> tuple[list[IntLikeType], list[IntLikeType]]:
    mesh = spec.mesh
    # note: dim_map does not allow double sharding which is the FSDP(fully_shard)+TP
    # case. Replace the custom logic with dim_map once we support it.
    dim_map: list[int | list[int]] = [-1] * spec.ndim
    for i, placement in enumerate(spec.placements):
        if isinstance(placement, Shard | _StridedShard):
            shard_dim = placement.dim
            if dim_map[shard_dim] == -1:
                dim_map[shard_dim] = [i]
            else:
                mesh_dim_list = dim_map[shard_dim]
                if not isinstance(mesh_dim_list, list):
                    raise AssertionError
                mesh_dim_list.append(i)

    # Compute shard coordinate:
    # The coordinate on each tensor dim is a tuple (idx, range)
    # If a DTensor is partitioned on its dim i into n shards, and the current rank
    # holds the j-th, then its shard coordinate will be (idx=j, range=n) on dim i
    mesh_size = mesh.shape
    shard_idx_by_dim = []
    total_num_shards_by_dim: list[
        IntLikeType
    ] = []  # total number of shards on each tensor dim
    for mesh_dim in dim_map:
        shard_idx: IntLikeType = 0
        total_num_shards: IntLikeType = 1
        # the tensor dim is sharded on more than 1 mesh dim
        if isinstance(mesh_dim, list):
            rank_coord = [mesh_coordinate[d] for d in mesh_dim]
            num_shards = [mesh_size[d] for d in mesh_dim]
            # compute the shard idx and total number of shards
            for idx, size in zip(rank_coord, num_shards):
                shard_idx = shard_idx * size + idx
                total_num_shards *= size

        shard_idx_by_dim.append(shard_idx)
        total_num_shards_by_dim.append(total_num_shards)
    return shard_idx_by_dim, total_num_shards_by_dim


def _calc_shard_linear_idx(
    shard_coord: Sequence[IntLikeType], shard_size: Sequence[IntLikeType]
) -> IntLikeType:
    # compute shard linear index
    shard_linear_idx: IntLikeType = 0
    shard_coord_stride: IntLikeType = 1
    for idx, size in zip(reversed(shard_coord), reversed(shard_size)):
        shard_linear_idx += idx * shard_coord_stride
        shard_coord_stride *= size

    return shard_linear_idx


def _resolve_device(device_mesh: DeviceMesh) -> torch.device:
    device_type = device_mesh.device_type
    device_handle = _get_device_handle(device_type)
    if device_handle is None:
        raise AssertionError
    device_idx = device_mesh.get_rank() % device_handle.device_count()

    @maybe_run_for_local_tensor
    def get_device(device_idx):
        return torch.device(f"{device_type}:{device_idx:d}")

    return get_device(device_idx)
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

import functools
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, cast, TypeGuard, TypeVar

import torch
import torch._C
import torch.distributed._functional_collectives as funcol
from torch._C._distributed import Placement
from torch.distributed import RankType
from torch.distributed._local_tensor import maybe_run_for_local_tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._collective_utils import (
    fill_empty_tensor_to_shards,
    mesh_broadcast,
    mesh_scatter,
    pad_tensor,
    shard_dim_alltoall,
    unpad_tensor,
)
from torch.distributed.tensor._ops._mask_buffer import MaskBuffer
from torch.types import IntLikeType


__all__ = ["Placement", "Shard", "Replicate", "Partial"]

_RankTypeT = TypeVar("_RankTypeT", bound=RankType)
_STRIDED_SHARD_BASE_SPLIT_FACTOR: Any = torch._C._distributed.StridedShard.split_factor


# Appease TestPublicBindings.test_correct_module_names
Placement.__module__ = "torch.distributed.tensor.placement_types"


def _raise_error(method_name, _):
    raise RuntimeError(
        f"Placement method '{method_name}' should not be called as "
        "it should be overridden by the subclass"
    )


Placement.__eq__ = functools.partial(_raise_error, method_name="__eq__")
Placement.__hash__ = functools.partial(_raise_error, method_name="__hash__")
Placement.__fx_repr__ = functools.partial(_raise_error, method_name="__fx_repr__")


class _StridedShardOffsetMode(IntEnum):
    FIRST = 0
    ALL = 1
    NONE = 2


def _explicit_or_backed_hint(value: IntLikeType) -> int | None:
    from torch.fx.experimental.symbolic_shapes import (
        free_unbacked_symbols,
        optimization_hint,
    )

    if isinstance(value, int):
        return int(value)
    if not isinstance(value, torch.SymInt):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    shape_env = getattr(value.node, "shape_env", None)
    if shape_env is None:
        return None
    unbacked_symbols = free_unbacked_symbols(value.node.expr)
    if unbacked_symbols and not unbacked_symbols.issubset(
        shape_env.var_to_hint_override.keys()
    ):
        return None
    return int(optimization_hint(value, fallback=None))


def _guarded_hint_int(value: IntLikeType, *, reason: str) -> int:
    if isinstance(value, int):
        return int(value)
    if not isinstance(value, torch.SymInt):
        return int(value)
    hint = _explicit_or_backed_hint(value)
    if hint is None:
        raise RuntimeError(
            f"Cannot specialize symbolic {reason} without a hint: {value}"
        )
    _defer_symbolic_equality_assert(value, hint, reason=reason)
    return hint


def _defer_symbolic_equality_assert(
    value: IntLikeType, hint: int, *, reason: str
) -> None:
    if not isinstance(value, torch.SymInt):
        return
    shape_env = getattr(value.node, "shape_env", None)
    if shape_env is None:
        return
    import sympy

    from torch.fx.experimental.symbolic_shapes import (
        free_unbacked_symbols,
        RuntimeAssert,
    )
    from torch.utils._traceback import CapturedTraceback

    expr = sympy.Eq(value.node.expr, hint)
    cands = sorted(
        free_unbacked_symbols(expr),
        key=lambda symbol: int(symbol.name[1:]),
    )
    key = cands[-1] if cands else None
    runtime_asserts = shape_env.deferred_runtime_asserts.setdefault(key, [])
    if any(runtime_assert.expr == expr for runtime_assert in runtime_asserts):
        return
    runtime_asserts.append(
        RuntimeAssert(
            expr,
            f"Runtime assertion failed: symbolic {reason} {value.node.expr} == {hint}",
            CapturedTraceback.extract(skip=1),
        )
    )
    shape_env.num_deferred_runtime_asserts += 1
    shape_env._update_version_counter()


def _split_factor_key(value: IntLikeType) -> object:
    if isinstance(value, torch.SymInt):
        return value.node.expr
    return int(value)


def _sym_mod(left: IntLikeType, right: IntLikeType) -> Any:
    return cast(Any, left) % cast(Any, right)


def _hint_proves_even_shard(size: IntLikeType, num_chunks: IntLikeType) -> bool:
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    if guard_or_false(_sym_mod(size, num_chunks) == 0):
        return True
    size_hint = _explicit_or_backed_hint(size)
    chunk_hint = _explicit_or_backed_hint(num_chunks)
    if (
        size_hint is None
        or chunk_hint is None
        or chunk_hint == 0
        or size_hint % chunk_hint != 0
    ):
        return False

    # Padding decisions require a Python branch. Only explicit hint overrides
    # can justify skipping padding for otherwise data-dependent sharding.
    torch._check(_sym_mod(size, num_chunks) == 0)
    return True


class Shard(torch._C._distributed.Shard):
    """
    The ``Shard(dim)`` placement describes the DTensor sharding on tensor dimension
    ``dim`` over a corresponding ``DeviceMesh`` dimension, where each rank on the
    DeviceMesh dimension only holds a shard/piece of the global Tensor. The
    ``Shard(dim)`` placement follows the ``torch.chunk(dim)`` semantic, where the
    last few shards on the DeviceMesh dimension might be empty when the tensor dimension
    is not evenly divisible on the DeviceMesh dimension. The ``Shard`` placement can be
    used by all DTensor APIs (i.e. distribute_tensor, from_local, etc.)

    Args:
        dim (int): The tensor dimension that describes the DTensor is sharded over its
            corresponding DeviceMesh dimension.

    .. warning:: sharding on a tensor dimension where the tensor dimension size is not
        evenly divisible on a DeviceMesh dimension is currently experimental and subject to change.

    .. note:: When checking whether a placement is shard-like, use
        :func:`_is_shard_like` instead of ``isinstance(p, Shard)`` to also
        match :class:`_StridedShard`.
    """

    def _split_tensor(
        self,
        tensor: torch.Tensor,
        num_chunks: int,
        *,
        with_padding: bool = True,
        contiguous: bool = True,
    ) -> tuple[list[torch.Tensor], list[int]]:
        """
        This function uses torch.chunk to split a tensor into num_chunks shards along
        the Shard placement dimension, and return a list of shards with their pad sizes.

        Keyword args:
            with_padding (bool, optional): when True, we pad the tensor on the last
            few ranks before calling the collectives (i.e. scatter/all_gather, etc.).
            This is because collectives usually require equal size tensor inputs
        """
        return self._split_tensor_helper(
            tensor, num_chunks, with_padding, contiguous, self.dim
        )

    @staticmethod
    def _split_tensor_helper(
        tensor: torch.Tensor,
        num_chunks: int,
        with_padding: bool,
        contiguous: bool,
        dim: int,
    ) -> tuple[list[torch.Tensor], list[int]]:
        if dim > tensor.ndim:
            raise AssertionError(
                f"Sharding dim {dim} greater than tensor ndim {tensor.ndim}"
            )

        # chunk tensor over dimension `dim` into n slices
        tensor_list = Shard._custom_chunk(tensor, num_chunks, dim=dim)

        # compute the chunk size inline with ``torch.chunk`` to calculate padding
        full_chunk_size = (tensor.size(dim) + num_chunks - 1) // num_chunks

        shard_list: list[torch.Tensor] = []
        pad_sizes: list[int] = []
        for shard in tensor_list:
            if with_padding:
                pad_size = Shard._get_shard_pad_size(full_chunk_size, shard, dim)
                shard = pad_tensor(shard, dim, pad_size)
                pad_sizes.append(pad_size)
            if contiguous:
                shard = shard.contiguous()
            shard_list.append(shard)
        return shard_list, pad_sizes

    @maybe_run_for_local_tensor
    def _select_split_tensor(
        self,
        tensor: torch.Tensor,
        num_chunks: int,
        index: RankType,
        *,
        with_padding: bool = True,
        contiguous: bool = True,
        clone: bool = True,
    ) -> torch.Tensor:
        """
        Like _split_tensor() but only returns a single shard at the given index.

        This function splits a tensor into num_chunks shards along the Shard placement
        dimension and returns only the shard at the specified index.

        Keyword args:
            with_padding (bool, optional): when True, we pad the tensor on the last
                few ranks before calling the collectives (i.e. scatter/all_gather, etc.).
                This is because collectives usually require equal size tensor inputs.
            contiguous (bool, optional): when True, the returned shard is made contiguous.
            clone (bool, optional): when True, the returned shard is cloned.
        """
        # We don't handle SymInt with_padding yet (because that requires extra
        # work based on the shard)
        if isinstance(index, int) or with_padding:
            shards, _ = self._split_tensor(
                tensor, num_chunks, with_padding=with_padding, contiguous=False
            )
            result = shards[index]
            if clone:
                result = result.clone()
            elif contiguous:
                result = result.contiguous()
            return result

        # For the SymInt implementation just compute the value for the tensor we
        # want rather than computing all of them.

        if self.dim > tensor.ndim:
            raise AssertionError(
                f"Sharding dim {self.dim} greater than tensor ndim {tensor.ndim}"
            )

        # chunk tensor over dimension `dim` into n slices
        dim_size = tensor.size(self.dim)
        split_size = (dim_size + num_chunks - 1) // num_chunks
        # each split is split_size except (maybe) the last one...
        last_split = dim_size - split_size * (num_chunks - 1)

        start = split_size * index
        length = torch.sym_ite(index == num_chunks - 1, last_split, split_size)
        result = torch.narrow(tensor, self.dim, start, length)
        if clone:
            result = result.clone()
        elif contiguous:
            result = result.contiguous()
        return result

    @staticmethod
    def _custom_chunk(
        tensor: torch.Tensor, num_chunks: IntLikeType, dim: int
    ) -> list[torch.Tensor]:
        """
        Returns list of tensor chunks along dim.
        Uses torch.chunk in eager mode, but torch.narrow under tracing to be unbacked-symint safe.
        Also handles uneven/zero-sharding cases.
        """
        from torch.distributed._functional_collectives import _are_we_tracing
        from torch.fx.experimental.symbolic_shapes import has_free_unbacked_symbols

        if tensor.dim() <= 0:
            raise AssertionError(f"Expected tensor.dim() > 0, got {tensor.dim()}")
        num_chunks_int = _guarded_hint_int(num_chunks, reason="chunk count")
        if num_chunks_int <= 0:
            raise AssertionError(f"Expected num_chunks > 0, got {num_chunks}")

        # TODO(pianpwk): remove the unbacked symbols check and fix AsyncTP pattern matching
        # for test_micro_pipeline_tp.py.
        if not _are_we_tracing() or not has_free_unbacked_symbols(tensor):
            tensor_list = list(torch.chunk(tensor, num_chunks_int, dim=dim))
            return fill_empty_tensor_to_shards(
                tensor_list, dim, num_chunks_int - len(tensor_list)
            )
        else:
            dim_size = tensor.size(dim)
            if _hint_proves_even_shard(dim_size, num_chunks):
                chunk_size = dim_size // num_chunks
                return [
                    tensor.narrow(dim, chunk_size * i, chunk_size)
                    for i in range(num_chunks_int)
                ]

            split_size = (dim_size + num_chunks - 1) // num_chunks
            chunks = []
            for i in range(num_chunks_int):
                start = torch.sym_min(split_size * i, dim_size)
                end = torch.sym_min(split_size * (i + 1), dim_size)
                chunks.append(tensor.narrow(dim, start, end - start))
            return chunks

    @staticmethod
    @maybe_run_for_local_tensor
    def local_shard_size_and_offset(
        curr_local_size: IntLikeType,
        num_chunks: int,
        rank: _RankTypeT,
    ) -> tuple[_RankTypeT, _RankTypeT]:
        """
        Given the size of the current local tensor (which may already be sharded on some dimensions),
        computes the new local shard size and offset given the desired number of chunks
        (num_chunks is generally equal to the size of the current sharding dim).

        Note: new local shard offset is relative to the current sharded tensor, not the global tensor.
        See `_utils.compute_local_shape_and_global_offset` for computing global offset.

        Returns (new local shard size, offset)

        """
        from torch.fx.experimental.symbolic_shapes import guard_or_false

        # Compute the chunk size inline with ``torch.chunk``
        # Assume uneven sharding as general case for unbacked sizes.
        if guard_or_false(curr_local_size % num_chunks == 0):
            full_chunk_size = curr_local_size // num_chunks
            # pyrefly: ignore[bad-assignment] # pyrefly bug?
            shard_starting_idx: _RankTypeT = full_chunk_size * rank
            return full_chunk_size, shard_starting_idx  # pyrefly: ignore[bad-return]

        # uneven sharding case
        full_chunk_size = (curr_local_size + num_chunks - 1) // num_chunks
        # pyrefly: ignore[bad-assignment] # pyrefly bug?
        shard_starting_idx: _RankTypeT = full_chunk_size * rank

        shard_end_idx = torch.sym_min(
            curr_local_size, shard_starting_idx + full_chunk_size
        )
        local_shard_size = torch.sym_max(0, shard_end_idx - shard_starting_idx)
        return local_shard_size, torch.sym_min(curr_local_size, shard_starting_idx)  # type: ignore[return-value]

    def _local_shard_size_and_offset(
        self,
        curr_local_size: int,
        num_chunks: int,
        rank: RankType,
    ) -> tuple[int, RankType]:
        # pyrefly: ignore [bad-argument-type, bad-return]
        return Shard.local_shard_size_and_offset(curr_local_size, num_chunks, rank)

    @staticmethod
    @maybe_run_for_local_tensor
    def _maybe_unpad_tensor_with_sizes(
        dim, local_tensor, pad_sizes, mesh_dim_local_rank, make_contiguous
    ) -> torch.Tensor:
        from torch.fx.experimental.symbolic_shapes import guard_or_true

        # Only unpad if the local_tensor was padded on the dimension.
        # Assume padding (uneven sharding) as general case for unbacked sizes.
        if guard_or_true(pad_sizes[mesh_dim_local_rank] > 0):
            local_tensor = unpad_tensor(
                local_tensor, dim, pad_sizes[mesh_dim_local_rank]
            )
            if make_contiguous:
                local_tensor = local_tensor.contiguous()
        return local_tensor

    def _shard_tensor(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        src_data_rank: int | None = 0,
    ) -> torch.Tensor:
        """
        Shard and scatter a tensor on a mesh dimension (use coordinate 0 on the
        mesh dimension as source of truth).

        Create the local tensor for this rank following the given Shard
        placement. If src_data_rank is None, perform only local splitting.
        Otherwise, additionally scatter data from src_data_rank. Unlike
        ``_split_tensor``, which supports uneven sharding via padding, this
        method requires the tensor dimension to be evenly divisible by the
        number of chunks (mesh dimension size).
        """
        my_coordinate = mesh.get_coordinate()
        num_chunks = mesh.size(mesh_dim=mesh_dim)

        if my_coordinate is None:
            # if rank is not part of mesh, we simply return an empty tensor
            return tensor.new_empty(0, requires_grad=tensor.requires_grad)

        mesh_dim_local_rank = my_coordinate[mesh_dim]

        if src_data_rank is None:
            # src_data_rank specified as None explicitly means to skip the
            # communications, simply split
            return self._select_split_tensor(
                tensor,
                num_chunks,
                mesh_dim_local_rank,
                with_padding=False,
                contiguous=True,
            )

        scatter_list, pad_sizes = self._split_tensor(
            tensor, num_chunks, with_padding=True, contiguous=True
        )

        it = iter(scatter_list)
        first = next(it)
        # Tensors in the scatter list are expected to have the same shape because
        # split is requested with padding.
        if not all(first.shape == v.shape for v in it):
            raise AssertionError

        output = torch.empty_like(first)

        # perform scatter from the src_data_rank as data source when it is not None
        mesh_scatter(
            output, scatter_list, mesh, mesh_dim=mesh_dim, group_src=src_data_rank
        )

        return Shard._maybe_unpad_tensor_with_sizes(
            self.dim, output, pad_sizes, mesh_dim_local_rank, True
        )

    @classmethod
    def _make_shard_tensor(
        cls,
        dim: int,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        src_data_rank: int | None = 0,
    ) -> torch.Tensor:
        shard_placement = cls(dim)
        return shard_placement._shard_tensor(tensor, mesh, mesh_dim, src_data_rank)

    def _reduce_shard_tensor(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        reduce_op: str,
        mesh_dim: int,
    ) -> torch.Tensor:
        """
        reduce and scatter a tensor on a mesh dimension
        """
        from torch.fx.experimental.symbolic_shapes import guard_or_true

        my_coordinate = mesh.get_coordinate()
        num_chunks = mesh.size(mesh_dim=mesh_dim)

        if my_coordinate is None:
            # if rank is not part of mesh, we simply return local_tensor,
            # which should be an empty tensor
            return tensor

        # Assume padding (uneven sharding) as general case for unbacked sizes,
        # unless an optimization hint can prove the even-shard branch.
        is_padded = not _hint_proves_even_shard(
            tensor.size(self.dim), num_chunks
        ) and guard_or_true(tensor.size(self.dim) % num_chunks != 0)
        pad_sizes = None
        if is_padded:
            scattered_list, pad_sizes = self._split_tensor(
                tensor, num_chunks, with_padding=True, contiguous=True
            )
            tensor = torch.cat(scattered_list, dim=self.dim)
        elif not tensor.is_contiguous():
            tensor = tensor.contiguous()

        output = funcol.reduce_scatter_single(
            tensor, reduce_op, scatter_dim=self.dim, group=(mesh, mesh_dim)
        )

        if is_padded:
            if pad_sizes is None:
                raise AssertionError
            output = Shard._maybe_unpad_tensor_with_sizes(
                self.dim, output, pad_sizes, mesh._sym_get_coordinate(mesh_dim), False
            )
        return output

    @maybe_run_for_local_tensor
    def _maybe_pad_tensor(
        self,
        local_tensor: torch.Tensor,
        logical_dim_size: IntLikeType,
        num_chunks: int,
    ) -> torch.Tensor:
        from torch.fx.experimental.symbolic_shapes import guard_or_true

        # Assume padding (uneven sharding) as general case for unbacked sizes,
        # unless an optimization hint can prove the even-shard branch.
        is_padded = not _hint_proves_even_shard(
            logical_dim_size, num_chunks
        ) and guard_or_true(logical_dim_size % num_chunks != 0)

        if is_padded:
            full_chunk_size = (logical_dim_size + num_chunks - 1) // num_chunks
            pad_size = full_chunk_size - local_tensor.size(self.dim)
            local_tensor = pad_tensor(local_tensor, self.dim, pad_size)  # type: ignore[bad-argument-type]

        if not local_tensor.is_contiguous():
            local_tensor = local_tensor.contiguous()

        return local_tensor

    @maybe_run_for_local_tensor
    def _maybe_unpad_tensor(
        self,
        local_tensor: torch.Tensor,
        logical_dim_size: IntLikeType,
        num_chunks: int,
    ) -> torch.Tensor:
        from torch.fx.experimental.symbolic_shapes import guard_or_false, guard_or_true

        # Assume padding (uneven sharding) as general case for unbacked sizes,
        # unless an optimization hint can prove the even-shard branch.
        is_padded = not _hint_proves_even_shard(
            logical_dim_size, num_chunks
        ) and guard_or_true(logical_dim_size % num_chunks != 0)

        if is_padded:
            full_chunk_size = (logical_dim_size + num_chunks - 1) // num_chunks
            unpad_size = full_chunk_size * num_chunks - logical_dim_size  # type: ignore[possibly-undefined]
            local_tensor = unpad_tensor(local_tensor, self.dim, unpad_size)  # type: ignore[bad-argument-type]

        # Bind derived symbolic sizes (e.g. 2*(s//2)) back to the original
        # symbol - needed for correct shape propagation and dynamo generation
        if local_tensor.size(self.dim) is not logical_dim_size:
            orig_size = local_tensor.size(self.dim)
            torch._check(orig_size >= logical_dim_size)
            local_tensor = local_tensor.narrow(self.dim, 0, logical_dim_size)

            # Safety check: the narrow should never change the concrete size.
            # Use guard_or_false so we don't trigger data-dependent guards
            # on unbacked symints.
            if guard_or_false(local_tensor.size(self.dim) != orig_size):
                raise RuntimeError(
                    f"narrow unexpectedly changed concrete size on dim {self.dim}: "
                    f"{orig_size} -> {local_tensor.size(self.dim)}"
                )

        # narrow/unpad on a non-leading dim leaves the tensor non-contiguous,
        # which leaks out to callers that expect a Replicate() result to have
        # a contiguous local tensor (e.g. downstream view ops in TP linear).
        # Use unconditional .contiguous() to avoid guarding on symbolic strides
        # under torch.compile with unbacked symints.
        local_tensor = local_tensor.contiguous()

        return local_tensor

    def _to_replicate_tensor(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: Sequence[IntLikeType],
    ) -> torch.Tensor:
        """
        This function all_gather all shards and return a tensor that
        is replicated on the previously sharded mesh dimension
        """
        num_chunks = mesh.size(mesh_dim=mesh_dim)
        logical_dim_size = current_logical_shape[self.dim]

        local_tensor = self._maybe_pad_tensor(
            local_tensor, logical_dim_size, num_chunks
        )

        result = funcol.all_gather_single(
            local_tensor,
            gather_dim=self.dim,
            group=(mesh, mesh_dim),
        )

        result = self._maybe_unpad_tensor(result, logical_dim_size, num_chunks)

        return result

    def _replicate_to_shard(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        shard_index: IntLikeType,
    ) -> torch.Tensor:
        """
        transform from replicated tensor to a sharded tensor on
        the current rank, which would perform a local chunk
        """
        num_chunks = mesh.size(mesh_dim=mesh_dim)
        return self._select_split_tensor(
            local_tensor,
            num_chunks,
            shard_index,
            with_padding=False,
            clone=True,
        )

    def _to_partial_tensor(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: Sequence[IntLikeType],
    ) -> torch.Tensor:
        """Embed a local shard in a zero-filled tensor pending sum reduction."""
        num_chunks = mesh.size(mesh_dim=mesh_dim)
        logical_dim_size = current_logical_shape[self.dim]
        local_shard_size, local_offset = self.local_shard_size_and_offset(
            logical_dim_size,
            num_chunks,
            mesh._sym_get_coordinate(mesh_dim),
        )
        torch._check(local_tensor.size(self.dim) == local_shard_size)
        output_shape: list[IntLikeType] = list(local_tensor.shape)
        output_shape[self.dim] = logical_dim_size
        output = local_tensor.new_zeros(output_shape)
        offset = torch.scalar_tensor(
            local_offset, dtype=torch.int64, device=local_tensor.device
        )
        indices = torch.arange(
            local_shard_size, dtype=torch.int64, device=local_tensor.device
        )
        return output.index_add(self.dim, indices + offset, local_tensor)

    @staticmethod
    @maybe_run_for_local_tensor
    def _get_shard_pad_size(
        full_size: int, local_tensor: torch.Tensor, dim: int
    ) -> int:
        """
        Get the padding size of the local tensor on the shard dimension.
        """
        return full_size - local_tensor.size(dim)

    @staticmethod
    def _compute_padding_info(
        current_logical_shape: Sequence[IntLikeType],
        num_chunks: int,
        old_shard_dim: int,
        new_shard_dim: int,
    ) -> tuple[bool, IntLikeType, int, bool, IntLikeType, int]:
        from torch.fx.experimental.symbolic_shapes import guard_or_true

        results = []
        for shard_dim in [old_shard_dim, new_shard_dim]:
            dim_logical_size = current_logical_shape[shard_dim]
            dim_padding = guard_or_true(dim_logical_size % num_chunks != 0)
            dim_full_chunk_size = (dim_logical_size + num_chunks - 1) // num_chunks
            results.append((dim_padding, dim_logical_size, dim_full_chunk_size))

        return results[0] + results[1]  # type: ignore[bad-return]

    @staticmethod
    @maybe_run_for_local_tensor
    def _pad_for_new_shard_dim(
        current_logical_shape: Sequence[IntLikeType],
        local_tensor: torch.Tensor,
        num_chunks: int,
        old_shard_dim: int,
        new_shard_dim: int,
    ) -> torch.Tensor:
        (
            old_dim_padding,
            _,
            old_dim_full_chunk_size,
            new_dim_padding,
            _,
            new_dim_full_chunk_size,
        ) = Shard._compute_padding_info(
            current_logical_shape, num_chunks, old_shard_dim, new_shard_dim
        )

        if old_dim_padding:
            old_dim_pad_size = Shard._get_shard_pad_size(
                old_dim_full_chunk_size, local_tensor, old_shard_dim
            )
            local_tensor = pad_tensor(local_tensor, old_shard_dim, old_dim_pad_size)
        if new_dim_padding:
            new_dim_pad_size = Shard._get_shard_pad_size(
                new_dim_full_chunk_size * num_chunks, local_tensor, new_shard_dim
            )
            local_tensor = pad_tensor(local_tensor, new_shard_dim, new_dim_pad_size)

        if not local_tensor.is_contiguous():
            local_tensor = local_tensor.contiguous()
        return local_tensor

    @staticmethod
    @maybe_run_for_local_tensor
    def _unpad_for_new_shard_dim(
        current_logical_shape: Sequence[IntLikeType],
        local_tensor: torch.Tensor,
        num_chunks: int,
        old_shard_dim: int,
        new_shard_dim: int,
        local_rank: int,
    ) -> torch.Tensor:
        (
            old_dim_padding,
            _,
            old_dim_full_chunk_size,
            new_dim_padding,
            new_dim_logical_size,
            new_dim_full_chunk_size,
        ) = Shard._compute_padding_info(
            current_logical_shape, num_chunks, old_shard_dim, new_shard_dim
        )

        if old_dim_padding:
            old_dim_unpad_size = (
                old_dim_full_chunk_size * num_chunks
                - current_logical_shape[old_shard_dim]  # type: ignore[possibly-undefined]
            )
            local_tensor = unpad_tensor(local_tensor, old_shard_dim, old_dim_unpad_size)  # type: ignore[possibly-undefined]

        if new_dim_padding:
            local_shard_size_on_new_dim = Shard.local_shard_size_and_offset(
                new_dim_logical_size, num_chunks, local_rank
            )[0]
            new_dim_unpad_size = new_dim_full_chunk_size - local_shard_size_on_new_dim  # type: ignore[possibly-undefined]
            local_tensor = unpad_tensor(local_tensor, new_shard_dim, new_dim_unpad_size)  # type: ignore[possibly-undefined]

        return local_tensor

    def _to_new_shard_dim(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: Sequence[IntLikeType],
        new_shard_dim: int,
    ) -> torch.Tensor:
        """
        transform from existing sharded tensor to a new sharded tensor on
        that shard on a new dimension, which performs an alltoall
        """
        my_coordinate = mesh.get_coordinate()
        if my_coordinate is None:
            # if rank is not part of mesh, we simply return local_tensor,
            # which should be an empty tensor
            return local_tensor

        num_chunks = mesh.size(mesh_dim=mesh_dim)

        local_tensor = Shard._pad_for_new_shard_dim(
            current_logical_shape, local_tensor, num_chunks, self.dim, new_shard_dim
        )

        new_tensor = shard_dim_alltoall(
            local_tensor, self.dim, new_shard_dim, mesh, mesh_dim
        )

        new_tensor = Shard._unpad_for_new_shard_dim(
            current_logical_shape,
            new_tensor,
            num_chunks,
            self.dim,
            new_shard_dim,
            my_coordinate[mesh_dim],
        )

        return new_tensor

    def __hash__(self) -> int:
        return hash(self.dim)

    def __repr__(self) -> str:
        """
        machine readable representation of the Shard placement
        """
        return f"Shard(dim={self.dim})"

    def __fx_repr__(self):
        """
        Returns FX-evaluable repr and required globals for Shard placement.
        Needed for passing this type as an opaque object input to a custom op.
        """
        return f"torch.distributed.tensor.placement_types.Shard(dim={self.dim})", {}

    def __str__(self) -> str:
        """human readable representation of the Shard placement"""
        return f"S({self.dim})"


class _StridedShard(torch._C._distributed.StridedShard):
    """
    _StridedShard was originally introduced to support 2D FSDP2 + TP sharding where the tensor
    is sharded on the TP mesh dimension first, then sharded on the FSDP mesh dimension.
    We call this right-to-left sharding which is the opposite of the default
    left-to-right sharding. See the example below::

        tensor shape: [8, 8]
        mesh: [[0, 1], [2, 3]], names=("dp", "tp")
        placements: [Shard(0), Shard(0)]

    The default sharding behavior shards the tensor on "dp" mesh dimension first then
    "tp" dimension. The sharding result will be::

        Rank    |   Mesh Coordinate |   Shard Index
        ------------------------------------------------
        0       |   (0, 0)          |   0 (row 0-1)
        1       |   (0, 1)          |   1 (row 2-3)
        2       |   (1, 0)          |   2 (row 4-5)
        3       |   (1, 1)          |   3 (row 6-7)

    While the FSDP2 + TP sharding behavior does the opposite: it shards the tensor on
    "tp" mesh dim first then "dp" dim. This right-to-left sharding will produce the
    result::

        Rank    |   Mesh Coordinate |   Shard Index
        ------------------------------------------------
        0       |   (0, 0)          |   0 (row 0-1)
        1       |   (0, 1)          |   2 (row 4-5)
        2       |   (1, 0)          |   1 (row 2-3)
        3       |   (1, 1)          |   3 (row 6-7)

    The consequence is, any attempt to redistribute this DTensor to a full replica will
    produce a wrong result because the shard-to-replicate redistribution always happens
    right-to-left, regardless it's left-to-right sharding or right-to-left. To address
    this, we use _StridedShard placement to make this right-to-left sharding compatible
    with our left-to-right convention on both tensor distribution and redistribution.

    Now with _StridedShard, the right-to-left sharding above can be represented as::

        tensor shape: [8, 8]
        mesh: [[0, 1], [2, 3]], names=("dp", "tp")
        placements: [_StridedShard(0, split_factor=2), Shard(0)]

    And a left-to-right processing of `placements` will produce the same result, which is
    different from using the `Shard` placement::

        Rank    |   Mesh Coordinate |   Shard Index
        ------------------------------------------------
        0       |   (0, 0)          |   0 (row 0-1)
        1       |   (0, 1)          |   2 (row 4-5)
        2       |   (1, 0)          |   1 (row 2-3)
        3       |   (1, 1)          |   3 (row 6-7)

    The argument `split_factor` is the number of existing shards over the tensor sharding
    dimension before processing the _StridedShard placement, as if the sharding happened
    right-to-left. In the example above, the tensor should first be sharded on the "tp"
    dimension into 2 shards before being sharded on the "dp" dimension. Therefore, the
    `split_factor` of the _StridedShard placement on "dp" dim is 2.
    """

    def __init__(
        self,
        dim: int,
        *,
        split_factor: IntLikeType = 1,
        sf: IntLikeType | None = None,
    ) -> None:
        # The C++ placement stores the unroll count as int64_t, but view
        # propagation may derive the split factor from symbolic shape metadata.
        # Preserve that symbolic value on the Python wrapper and guard the
        # concrete unroll count used by list-based lowering.
        if sf is not None:
            if split_factor != 1:
                raise TypeError("_StridedShard received both split_factor and sf")
            split_factor = sf
        split_factor_int = _guarded_hint_int(
            split_factor, reason="_StridedShard split_factor"
        )
        super().__init__(dim, split_factor=split_factor_int)
        if isinstance(split_factor, torch.SymInt):
            self._symbolic_split_factor = split_factor

    @property
    def split_factor(self) -> IntLikeType:  # pyrefly: ignore [bad-override]
        symbolic_split_factor = getattr(self, "_symbolic_split_factor", None)
        if symbolic_split_factor is not None:
            return symbolic_split_factor
        return int(_STRIDED_SHARD_BASE_SPLIT_FACTOR.__get__(self, type(self)))

    def _split_factor_int(self) -> int:
        return int(_STRIDED_SHARD_BASE_SPLIT_FACTOR.__get__(self, type(self)))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _StridedShard)
            and self.dim == other.dim
            and _split_factor_key(self.split_factor)
            == _split_factor_key(other.split_factor)
        )

    def __hash__(self) -> int:
        return hash((self.dim, _split_factor_key(self.split_factor)))

    def __repr__(self) -> str:
        """
        machine readable representation of the _StridedShard placement
        """
        return f"_StridedShard(dim={self.dim}, sf={self.split_factor})"

    def __str__(self) -> str:
        """human readable representation of the _StridedShard placement"""
        return f"_S({self.dim}, {self.split_factor})"

    @staticmethod
    @maybe_run_for_local_tensor
    def _select_shard(shards: list[torch.Tensor], shard_index) -> torch.Tensor:
        return shards[shard_index].clone()

    def __fx_repr__(self):
        """
        Returns FX-evaluable repr and required globals for Shard placement.
        Needed for passing this type as an opaque object input to a custom op.
        """
        # FX repr cannot refer to this placement's private SymInt object. The
        # constructor emitted a guard tying the symbolic split factor to this
        # concrete unroll count before the placement can reach codegen.
        return (
            f"torch.distributed.tensor.placement_types._StridedShard(dim={self.dim}, sf={self._split_factor_int()})",
            {},
        )

    @classmethod
    def _make_shard_tensor(
        cls,
        dim: int,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        src_data_rank: int | None = 0,
        split_factor: IntLikeType = 1,
    ) -> torch.Tensor:
        strided_shard_placement = cls(dim=dim, split_factor=split_factor)
        return strided_shard_placement._shard_tensor(
            tensor, mesh, mesh_dim, src_data_rank
        )

    def _shard_tensor(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        src_data_rank: int | None = 0,
    ) -> torch.Tensor:
        """
        Shard and scatter a tensor on a mesh dimension (use coordinate 0 on the
        mesh dimension as source of truth).

        Create the local tensor for this rank following the given StridedShard
        placement. If src_data_rank is None, perform only local splitting.
        Otherwise, additionally scatter data from src_data_rank. Unlike
        ``_split_tensor``, which supports uneven sharding via padding, this
        method requires the tensor dimension to be evenly divisible by the
        number of chunks (mesh dimension size).
        """
        my_coordinate = mesh.get_coordinate()
        num_chunks = mesh.size(mesh_dim=mesh_dim)

        if my_coordinate is None:
            # if rank is not part of mesh, we simply return an empty tensor
            return tensor.new_empty(0, requires_grad=tensor.requires_grad)

        mesh_dim_local_rank = my_coordinate[mesh_dim]

        if src_data_rank is None:
            # src_data_rank specified as None explicitly means to skip the
            # communications, simply split
            scatter_list, _ = self._split_tensor(
                tensor, num_chunks, with_padding=False, contiguous=True
            )

            return self._select_shard(scatter_list, mesh_dim_local_rank)

        scatter_list, pad_sizes = self._split_tensor(
            tensor, num_chunks, with_padding=True, contiguous=True
        )

        it = iter(scatter_list)
        first = next(it)
        # Tensors in the scatter list are expected to have the same shape because
        # split is requested with padding.
        if not all(first.shape == v.shape for v in it):
            raise AssertionError

        output = torch.empty_like(first)

        # perform scatter from the src_data_rank as data source when it is not None
        mesh_scatter(
            output, scatter_list, mesh, mesh_dim=mesh_dim, group_src=src_data_rank
        )

        return Shard._maybe_unpad_tensor_with_sizes(
            self.dim, output, pad_sizes, mesh_dim_local_rank, True
        )

    def _split_tensor(
        self,
        tensor: torch.Tensor,
        num_chunks: int,
        *,
        with_padding: bool = True,
        contiguous: bool = True,
    ) -> tuple[list[torch.Tensor], list[int]]:
        if self.dim > tensor.ndim:
            raise AssertionError(
                f"Sharding dim {self.dim} greater than tensor ndim {tensor.ndim}"
            )

        # Essentially _StridedShard express the right-to-left sharding in the
        # reversed order. Here we perform first_split as the virtual "right" sharding,
        # and then second_split as the virtual "left" sharding, and finally assemble
        # results in the transposed left-first order.

        # First split: chunk into split_factor pieces
        split_factor = self.split_factor
        split_factor_int = self._split_factor_int()
        first_split = Shard._custom_chunk(tensor, split_factor, dim=self.dim)

        # Second split: chunk each piece into num_chunks pieces
        second_split = []
        for s in first_split:
            second_split.append(Shard._custom_chunk(s, num_chunks, dim=self.dim))

        shard_list: list[torch.Tensor] = []
        for i in range(num_chunks):
            shard = torch.cat(
                [second_split[j][i] for j in range(split_factor_int)],
                dim=self.dim,
            )
            if contiguous:
                shard = shard.contiguous()
            shard_list.append(shard)

        pad_sizes: list[int] = []
        if with_padding:
            # The amount of padding is determined by the local chunk with the
            # largest size. Avoid this Python max when padding is disabled so
            # tracing paths can keep unbacked sizes symbolic.
            max_chunk_size = max([shard.size(self.dim) for shard in shard_list])
            pad_sizes = [max_chunk_size - shard.size(self.dim) for shard in shard_list]

        return shard_list, pad_sizes

    @maybe_run_for_local_tensor
    def _select_split_tensor(
        self,
        tensor: torch.Tensor,
        num_chunks: int,
        index: IntLikeType,
        *,
        with_padding: bool = True,
        contiguous: bool = True,
        clone: bool = True,
    ) -> torch.Tensor:
        """
        Like _split_tensor() but only returns a single shard at the given index.

        This function splits a tensor into num_chunks shards along the _StridedShard
        placement dimension and returns only the shard at the specified index.

        Keyword args:
            with_padding (bool, optional): when True, we pad the tensor on the last
                few ranks before calling the collectives (i.e. scatter/all_gather, etc.).
                This is because collectives usually require equal size tensor inputs.
            contiguous (bool, optional): when True, the returned shard is made contiguous.
            clone (bool, optional): when True, the returned shard is cloned.
        """
        shards, _ = self._split_tensor(
            tensor, num_chunks, with_padding=with_padding, contiguous=False
        )
        result = shards[index]
        if clone:
            result = result.clone()
        elif contiguous:
            result = result.contiguous()
        return result

    def _to_replicate_tensor(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: Sequence[IntLikeType],
    ) -> torch.Tensor:
        """
        Replay the replicate-to-shard process to understand how to stitch shards back.

        This method performs all_gather to collect all shards and then reconstructs
        the original replicated tensor by handling padding, unpadding, and reordering.

        Example:
            Consider a 1D input tensor [0, 1, 2, 3, 4, 5, 6, 7, 8] with 9 elements.
            Using _StridedShard(dim=0, split_factor=2) and num_chunks=4:

            Preparation (via _split_tensor, before _to_replicate_tensor is called):
                _split_tensor produces 4 shards with strided indices:
                - First split (split_factor=2): [0,1,2,3,4] and [5,6,7,8]
                - Second split (num_chunks=4) on each piece:
                    [0,1,2,3,4] (5 elements) -> [[0,1], [2,3], [4], []]
                    [5,6,7,8]   (4 elements) -> [[5], [6], [7], [8]]
                - Transpose and concatenate for each chunk:
                    Chunk 0: [0,1] + [5] = [0,1,5]  (indices)
                    Chunk 1: [2,3] + [6] = [2,3,6]  (indices)
                    Chunk 2: [4] + [7]   = [4,7]    (indices)
                    Chunk 3: [] + [8]    = [8]      (indices)

                So we get shards with values:
                    Rank 0: [0, 1, 5]  (size=3)
                    Rank 1: [2, 3, 6]  (size=3)
                    Rank 2: [4, 7]     (size=2)
                    Rank 3: [8]        (size=1)

                These shards are the `local_tensor` input to _to_replicate_tensor
                on each rank. Each rank only has its own shard when this function
                is called.

            Step 1: Pad all shards to max_chunk_size=3:
                    Rank 0: [0, 1, 5]     (no padding needed)
                    Rank 1: [2, 3, 6]     (no padding needed)
                    Rank 2: [4, 7, P]     (padded with 1 element)
                    Rank 3: [8, P, P]     (padded with 2 elements)

            Step 2: all_gather produces concatenated padded tensor:
                [0, 1, 5, | 2, 3, 6, | 4, 7, P, | 8, P, P]
                 chunk 0    chunk 1    chunk 2    chunk 3
                (pos 0-2)  (pos 3-5)  (pos 6-8)  (pos 9-11)

            Step 3: Compute select_indices to extract valid elements and reorder:
                sharded_indices = [[0,1,5], [2,3,6], [4,7], [8]]
                padded_positions:
                    chunk 0: base=0 -> [0, 1, 2]  (positions of [0,1,5] in gathered)
                    chunk 1: base=3 -> [3, 4, 5]  (positions of [2,3,6] in gathered)
                    chunk 2: base=6 -> [6, 7]     (positions of [4,7] in gathered)
                    chunk 3: base=9 -> [9]        (position of [8] in gathered)

                permutation = cat(sharded_indices) = [0, 1, 5, 2, 3, 6, 4, 7, 8]
                select_positions = cat(padded_positions) = [0, 1, 2, 3, 4, 5, 6, 7, 9]

                inv_permutation = argsort(permutation)
                    permutation[0]=0 -> inv_permutation[0]=0
                    permutation[1]=1 -> inv_permutation[1]=1
                    permutation[2]=5 -> inv_permutation[5]=2
                    permutation[3]=2 -> inv_permutation[2]=3
                    permutation[4]=3 -> inv_permutation[3]=4
                    permutation[5]=6 -> inv_permutation[6]=5
                    permutation[6]=4 -> inv_permutation[4]=6
                    permutation[7]=7 -> inv_permutation[7]=7
                    permutation[8]=8 -> inv_permutation[8]=8
                    => inv_permutation = [0, 1, 3, 4, 6, 2, 5, 7, 8]

                select_indices = select_positions[inv_permutation]
                               = [0, 1, 2, 3, 4, 5, 6, 7, 9][inv_permutation]
                    For original position 0: select_indices[0] = select_positions[0] = 0
                    For original position 1: select_indices[1] = select_positions[1] = 1
                    For original position 2: select_indices[2] = select_positions[3] = 3
                    For original position 3: select_indices[3] = select_positions[4] = 4
                    For original position 4: select_indices[4] = select_positions[6] = 6
                    For original position 5: select_indices[5] = select_positions[2] = 2
                    For original position 6: select_indices[6] = select_positions[5] = 5
                    For original position 7: select_indices[7] = select_positions[7] = 7
                    For original position 8: select_indices[8] = select_positions[8] = 9
                    => select_indices = [0, 1, 3, 4, 6, 2, 5, 7, 9]

            Step 4: index_select from gathered tensor using select_indices:
                gathered = [0, 1, 5, 2, 3, 6, 4, 7, P, 8, P, P]
                result = gathered[select_indices]
                       = gathered[[0, 1, 3, 4, 6, 2, 5, 7, 9]]
                       = [0, 1, 2, 3, 4, 5, 6, 7, 8]

            The result is the original replicated tensor [0, 1, 2, 3, 4, 5, 6, 7, 8].
        """
        num_chunks = mesh.size(mesh_dim=mesh_dim)
        logical_dim_size = current_logical_shape[self.dim]

        # indices_tensor is 1D torch.arange(logical_dim_size) unsqueezed
        # so that we can reuse self._split_tensor which splits on self.dim
        shape = [1] * self.dim + [logical_dim_size]
        # pyrefly: ignore [no-matching-overload]
        indices_tensor = torch.arange(
            logical_dim_size, device=local_tensor.device
        ).view(shape)

        sharded_indices, _ = self._split_tensor(
            indices_tensor,
            num_chunks,
            with_padding=False,
            contiguous=False,
        )
        # squeeze back to 1D indices tensor
        sharded_indices = [shard.view(-1) for shard in sharded_indices]

        # First chunk should be one of those biggest chunks. Compute the size
        # from the logical shape rather than the fake arange split so padding
        # depends only on the original symbolic dimension and its hint.
        max_chunk_size, _ = self.local_shard_size_and_offset(
            logical_dim_size,
            num_chunks,
            rank=0,
            offset_mode=_StridedShardOffsetMode.ALL,
        )
        local_pad_size = max_chunk_size - local_tensor.size(self.dim)
        local_tensor_padded = pad_tensor(local_tensor, self.dim, local_pad_size)

        if not local_tensor_padded.is_contiguous():
            local_tensor_padded = local_tensor_padded.contiguous()

        replicate_tensor_permuted_padded = funcol.all_gather_single(
            local_tensor_padded,
            gather_dim=self.dim,
            group=(mesh, mesh_dim),
        )
        if isinstance(replicate_tensor_permuted_padded, funcol.AsyncCollectiveTensor):
            replicate_tensor_permuted_padded = replicate_tensor_permuted_padded.wait()

        # After all_gather, the tensor is [chunk0, chunk1 (may padded), ...].
        # Each chunk may have padding at the end. Use a single index_select to
        # both extract non-padding data and reorder to the original positions.
        #
        # Build select_indices where select_indices[original_pos] = position in
        # the padded tensor that holds the element for original_pos.
        padded_positions: list[torch.Tensor] = []
        for i, shard in enumerate(sharded_indices):
            base_offset = i * max_chunk_size
            positions = cast(
                torch.Tensor,
                base_offset
                + torch.arange(
                    shard.numel(),
                    device=local_tensor.device,
                ),
            )
            padded_positions.append(positions)

        # Permutation ends up containing strided indices because we create it by
        # chunking over a particular dimension of an N-D shaped arange tensor.
        permutation = torch.cat(sharded_indices)
        # Choose the position by skipping padding indices from
        # replicate_tensor_permuted_padded.
        select_positions = torch.cat(padded_positions)

        inv_permutation = torch.argsort(permutation)
        select_indices = select_positions.index_select(0, inv_permutation)

        replicate_tensor = torch.index_select(
            replicate_tensor_permuted_padded, self.dim, select_indices
        )
        logical_shape = cast(list[IntLikeType], list(replicate_tensor.shape))
        logical_shape[self.dim] = current_logical_shape[self.dim]
        replicate_tensor = replicate_tensor.as_strided(
            logical_shape,
            torch._prims_common.make_contiguous_strides_for(
                cast(list[int], logical_shape)
            ),
        )

        return replicate_tensor.contiguous()

    def _replicate_to_strided_shard(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        shard_index: IntLikeType,
    ) -> torch.Tensor:
        """
        Transform from replicated tensor to a strided-sharded tensor on the current rank.

        This performs a local chunking operation using the _StridedShard pattern,
        where the tensor is split according to the strided sharding semantics
        (interleaved pieces based on split_factor).

        Args:
            local_tensor: The replicated tensor on this rank.
            mesh: The device mesh over which the tensor is distributed.
            mesh_dim: The mesh dimension for the sharding.
            shard_index: The index of the shard to select (typically the rank's
                coordinate on the mesh dimension).

        Returns:
            The local strided shard for this rank.
        """
        num_chunks = mesh.size(mesh_dim=mesh_dim)
        return self._select_split_tensor(
            local_tensor,
            num_chunks,
            shard_index,
            with_padding=False,
            clone=True,
        )

    @staticmethod
    @maybe_run_for_local_tensor
    def _local_shard_size(sharded_indices: list[torch.Tensor], rank: RankType) -> int:
        return len(sharded_indices[rank])

    def _local_shard_size_and_offset(
        self,
        curr_local_size: int,
        num_chunks: int,
        rank: RankType,
        offset_mode: _StridedShardOffsetMode = _StridedShardOffsetMode.FIRST,
    ) -> tuple[int, int | list[int] | None]:
        return self.local_shard_size_and_offset(
            curr_local_size, num_chunks, rank, offset_mode
        )

    @maybe_run_for_local_tensor
    def local_shard_size_and_offset(
        self,
        curr_local_size: IntLikeType,
        num_chunks: int,
        rank: RankType,
        offset_mode: _StridedShardOffsetMode = _StridedShardOffsetMode.FIRST,
    ) -> tuple[int, list[int] | int | None]:
        """
        Compute the local shard size and offset(s) for a _StridedShard placement.

        Unlike the regular Shard placement which produces contiguous offsets, _StridedShard
        produces non-contiguous (strided) offsets due to the right-to-left sharding semantics.
        This method computes the actual indices that belong to the local shard.

        Args:
            self (_StridedShard): The _StridedShard placement instance.
            curr_local_size (int): The current size of the tensor dimension to be sharded.
            num_chunks (int): Number of chunks to split the dimension into (typically the mesh dimension size).
            rank (RankType): The rank index to compute the shard for.
            offset_mode (_StridedShardOffsetMode): Controls offset materialization.
                FIRST returns only the first offset as an int, ALL returns all offsets
                as a list, and NONE skips offset materialization. Defaults to FIRST.

        Returns:
            tuple: A tuple containing:
                - local_shard_size (int): The number of elements in the local shard
                  for this rank.
                - offset (int | list[int] | None): Determined by offset_mode. FIRST
                  returns the first offset as an int, ALL returns a list of all offsets,
                  and NONE returns None.
        """
        offset_mode = _StridedShardOffsetMode(offset_mode)
        from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

        if (
            offset_mode is not _StridedShardOffsetMode.FIRST
            and isinstance(curr_local_size, torch.SymInt)
            and free_unbacked_symbols(curr_local_size.node.expr)
        ):
            split_factor = self.split_factor
            split_factor_int = self._split_factor_int()
            offset = None if offset_mode is _StridedShardOffsetMode.NONE else []
            if _hint_proves_even_shard(curr_local_size, split_factor * num_chunks):
                # pyrefly: ignore[bad-return]
                return (
                    curr_local_size // num_chunks,
                    offset,
                )

            local_shard_size: IntLikeType = 0
            first_split_size = (curr_local_size + split_factor - 1) // split_factor
            for split_idx in range(split_factor_int):
                first_start = torch.sym_min(
                    first_split_size * split_idx, curr_local_size
                )
                first_end = torch.sym_min(
                    first_split_size * (split_idx + 1), curr_local_size
                )
                first_len = torch.sym_max(0, first_end - first_start)
                second_split_size = (first_len + num_chunks - 1) // num_chunks
                second_start = torch.sym_min(second_split_size * rank, first_len)
                second_end = torch.sym_min(second_split_size * (rank + 1), first_len)
                local_shard_size = local_shard_size + torch.sym_max(
                    0, second_end - second_start
                )
            return local_shard_size, offset  # pyrefly: ignore[bad-return]

        # indices_tensor is 1D torch.arange(logical_dim_size) unsqueezed
        # so that we can reuse self._split_tensor which splits on self.dim
        shape = [1] * self.dim + [curr_local_size]
        # pyrefly: ignore [no-matching-overload]
        indices_tensor = torch.arange(
            curr_local_size,
        ).view(shape)

        sharded_indices, _ = self._split_tensor(
            indices_tensor,
            num_chunks,
            with_padding=False,
            contiguous=False,
        )
        # squeeze back to 1D indices tensor
        sharded_indices = [shard.view(-1) for shard in sharded_indices]

        local_shard_size = _StridedShard._local_shard_size(sharded_indices, rank)
        if offset_mode is _StridedShardOffsetMode.NONE:
            # Avoid .tolist() which creates unbacked SymInts under FakeTensorMode.
            return local_shard_size, None
        if local_shard_size > 0:
            offsets = sharded_indices[rank].tolist()
        else:
            offsets = []

        if offset_mode is _StridedShardOffsetMode.FIRST:
            # Always return an int for consistency across ranks.
            # For empty shards, return -1 as an invalid offset indicator.
            offsets = offsets[0] if len(offsets) > 0 else -1

        return local_shard_size, offsets

    @staticmethod
    def _compute_padding_info(
        logical_size_on_dim: int,
        num_chunks: int,
        shard_dim: int,
        split_factor: int = 1,
    ) -> tuple[bool, int]:
        """
        Compute padding information for _StridedShard collective operations.

        This method calculates whether padding is needed and the maximum chunk size
        for collective operations (e.g., all-to-all) involving _StridedShard tensors.

        For _StridedShard with split_factor > 1, the tensor undergoes two-level splitting:
        1. First level: split into ``split_factor`` pieces
        2. Second level: each piece is split into ``num_chunks`` pieces

        The resulting shards are interleaved, so padding must account for both levels.
        Unlike ``Shard._compute_padding_info`` where at most one partition (the last)
        requires padding, _StridedShard can have multiple trailing partitions that
        need padding due to the interleaved structure.

        When split_factor=1, this behaves identically to regular Shard padding logic.

        Args:
            logical_size_on_dim: The logical shape size of the tensor ``shard_dim``.
            num_chunks: Number of chunks to split into (typically the mesh dim size).
            shard_dim: The dimension along which the tensor is sharded.
            split_factor: The number of pre-existing splits from right-to-left sharding.
                Defaults to 1 (no strided sharding effect).

        Returns:
            A tuple of (needs_padding_on_dim, max_chunk_size):
                - needs_padding_on_dim: Whether padding is required on the shard dimension.
                - max_chunk_size: The maximum chunk size per rank after both levels of splitting.
        """

        def _ceil_div(a: int, b: int) -> int:
            return (a + b - 1) // b

        if split_factor != 1:
            # Computing padding info for StridedShard tensor dim
            # First level: split into split_factor pieces
            first_chunk_size = _ceil_div(logical_size_on_dim, split_factor)
            num_full_first_chunks = logical_size_on_dim // first_chunk_size
            remainder = logical_size_on_dim - num_full_first_chunks * first_chunk_size

            # Determine if padding is needed:
            # - remainder > 0 means one chunk has partial size
            # - empty first-level chunks means some ranks get nothing at each level
            # - first_chunk_size % num_chunks != 0 means uneven second-level split
            #   (e.g., 6 rows, sf=2, 8 ranks: first_chunk=3, but only 3 of 8 ranks
            #   get data per first-level chunk, so ranks 3-7 have size-0 tensors)
            has_partial_chunk = remainder > 0
            num_empty_chunks = (
                split_factor - num_full_first_chunks - (1 if has_partial_chunk else 0)
            )
            has_uneven_second_level = first_chunk_size % num_chunks != 0
            needs_padding_on_dim = (
                has_partial_chunk or num_empty_chunks > 0 or has_uneven_second_level
            )

            # Second level: each first-level chunk is split into num_chunks pieces
            # Calculate the per-rank chunk size after both levels of splitting
            max_chunk_size = (
                _ceil_div(first_chunk_size, num_chunks) * num_full_first_chunks
            )
            if has_partial_chunk:
                max_chunk_size += _ceil_div(remainder, num_chunks)
        else:
            # Compute padding info for normal shard, no split_factor impact.
            needs_padding_on_dim = logical_size_on_dim % num_chunks != 0
            max_chunk_size = _ceil_div(logical_size_on_dim, num_chunks)

        return (
            needs_padding_on_dim,
            max_chunk_size,
        )

    @maybe_run_for_local_tensor
    def _pad_for_new_shard_dim(
        self,
        current_logical_shape: list[int],
        local_tensor: torch.Tensor,
        num_chunks: int,
        old_shard_dim: int,
        new_shard_dim: int,
        split_factor: int,
    ) -> torch.Tensor:
        """
        Pad the local tensor before alltoall for _StridedShard's _to_new_shard_dim operation.

        For _StridedShard, the padding must account for the split_factor since each rank
        holds split_factor interleaved pieces. The full chunk size per rank is
        split_factor * single_chunk_size.
        """
        (
            old_dim_padding,
            old_dim_max_chunk_size,
        ) = _StridedShard._compute_padding_info(
            current_logical_shape[old_shard_dim],
            num_chunks,
            old_shard_dim,
            split_factor,
        )
        (
            new_dim_padding,
            new_dim_max_chunk_size,
        ) = _StridedShard._compute_padding_info(
            current_logical_shape[new_shard_dim],
            num_chunks,
            new_shard_dim,
        )

        if old_dim_padding:
            target_size = old_dim_max_chunk_size
            old_dim_pad_size = target_size - local_tensor.size(old_shard_dim)
            if old_dim_pad_size > 0:
                local_tensor = pad_tensor(local_tensor, old_shard_dim, old_dim_pad_size)

        if new_dim_padding:
            target_total_size = new_dim_max_chunk_size * num_chunks
            new_dim_pad_size = target_total_size - local_tensor.size(new_shard_dim)
            if new_dim_pad_size > 0:
                local_tensor = pad_tensor(local_tensor, new_shard_dim, new_dim_pad_size)

        if not local_tensor.is_contiguous():
            local_tensor = local_tensor.contiguous()
        return local_tensor

    @maybe_run_for_local_tensor
    def _unpad_for_new_shard_dim(
        self,
        current_logical_shape: list[int],
        local_tensor: torch.Tensor,
        num_chunks: int,
        old_shard_dim: int,
        new_shard_dim: int,
        split_factor: int,
        local_rank: int,
    ) -> torch.Tensor:
        """
        Unpad the local tensor after alltoall for _StridedShard's
        _to_new_shard_dim operation.

        This uses the same index_select approach as _to_replicate_tensor to both
        extract correct elements (removing padding) and reorder them. The key insight
        is that padding can only be at the end, so the first shard has the max size.
        """
        old_dim_logical_size = current_logical_shape[old_shard_dim]
        new_dim_logical_size = current_logical_shape[new_shard_dim]
        (
            new_dim_padding,
            new_dim_max_chunk_size,
        ) = _StridedShard._compute_padding_info(
            current_logical_shape[new_shard_dim],
            num_chunks,
            new_shard_dim,
        )

        # Build sharded indices to understand the strided pattern
        shape = [1] * old_shard_dim + [old_dim_logical_size]
        indices_tensor = torch.arange(
            old_dim_logical_size, device=local_tensor.device
        ).view(shape)
        sharded_indices, _ = self._split_tensor(
            indices_tensor, num_chunks, with_padding=False, contiguous=False
        )
        sharded_indices = [shard.view(-1) for shard in sharded_indices]

        max_chunk_size = len(sharded_indices[0])

        # Compute positions in the padded local_tensor for each actual element.
        # Each shard i starts at position i * max_chunk_size in local_tensor.
        padded_positions = []
        for i, shard in enumerate(sharded_indices):
            base_offset = i * max_chunk_size
            positions = base_offset + torch.arange(
                len(shard), device=local_tensor.device
            )
            padded_positions.append(positions)

        permutation = torch.cat(sharded_indices)
        select_positions = torch.cat(padded_positions)

        inv_permutation = torch.argsort(permutation)
        select_indices = select_positions.index_select(0, inv_permutation)
        local_tensor = torch.index_select(local_tensor, old_shard_dim, select_indices)

        if new_dim_padding:
            # copy from Shard._unpad_for_new_shard_dim
            local_shard_size_on_new_dim = Shard.local_shard_size_and_offset(
                new_dim_logical_size, num_chunks, local_rank
            )[0]
            new_dim_unpad_size = new_dim_max_chunk_size - local_shard_size_on_new_dim  # type: ignore[possibly-undefined]
            local_tensor = unpad_tensor(local_tensor, new_shard_dim, new_dim_unpad_size)  # type: ignore[possibly-undefined]
        return local_tensor

    def _to_new_shard_dim(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: list[int],
        new_shard_dim: int,
    ) -> torch.Tensor:
        """
        Transform from existing _StridedShard tensor to a new _StridedShard tensor that
        shards on a new dimension, which performs an alltoall.

        This is analogous to Shard._to_new_shard_dim but accounts for the split_factor
        in padding calculations since _StridedShard represents interleaved sharding.
        """
        my_coordinate = mesh.get_coordinate()
        if my_coordinate is None:
            # if rank is not part of mesh, we simply return local_tensor,
            # which should be an empty tensor
            return local_tensor

        num_chunks = mesh.size(mesh_dim=mesh_dim)

        local_tensor = self._pad_for_new_shard_dim(
            current_logical_shape,
            local_tensor,
            num_chunks,
            self.dim,
            new_shard_dim,
            self._split_factor_int(),
        )

        new_tensor = shard_dim_alltoall(
            local_tensor, self.dim, new_shard_dim, mesh, mesh_dim
        )
        new_tensor = self._unpad_for_new_shard_dim(
            current_logical_shape,
            new_tensor,
            num_chunks,
            self.dim,
            new_shard_dim,
            self._split_factor_int(),
            my_coordinate[mesh_dim],
        )

        return new_tensor


def _is_shard_like(p: "Placement") -> TypeGuard[Shard | _StridedShard]:
    """Check if a placement is Shard or _StridedShard.

    Use this instead of ``isinstance(p, Shard)`` to avoid silently missing
    ``_StridedShard``.  When ``_StridedShard`` is unified with ``Shard``
    (see TODO on the class), this helper can be collapsed to a single
    ``isinstance`` check.
    """
    return isinstance(p, Shard | _StridedShard)


class Replicate(torch._C._distributed.Replicate):
    """
    The ``Replicate()`` placement describes the DTensor replicating on a corresponding
    ``DeviceMesh`` dimension, where each rank on the DeviceMesh dimension holds a
    replica of the global Tensor. The ``Replicate`` placement can be used by all
    DTensor APIs (i.e. ``distribute_tensor``, ``DTensor.from_local``, etc.)
    """

    def __hash__(self) -> int:
        # every replicate placement is the same
        return -1

    def __repr__(self) -> str:
        """
        machine readable representation of the Replicate placement
        """
        return "Replicate()"

    def __fx_repr__(self):
        """
        Returns FX-evaluable repr and required globals for Replicate placement.
        Needed for passing this type as an opaque object input to a custom op.
        """
        return "torch.distributed.tensor.placement_types.Replicate()", {}

    def __str__(self) -> str:
        """
        human readable representation of the Replicate placement
        """
        return "R"

    @classmethod
    def _make_replicate_tensor(
        cls,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        src_data_rank: int | None = 0,
    ) -> torch.Tensor:
        """
        Replicate (broadcast) a torch.Tensor on a mesh dimension (use
        the first coordinate on the mesh dimension as source of truth)
        """
        my_coordinate = mesh.get_coordinate()
        if my_coordinate is None:
            # if rank is not part of mesh, we simply return an empty tensor
            return tensor.new_empty(0, requires_grad=tensor.requires_grad)

        tensor = tensor.contiguous()

        if src_data_rank is not None:
            # perform broadcast from the src_data_rank as data source when it is not None
            mesh_broadcast(tensor, mesh, mesh_dim=mesh_dim, group_src=src_data_rank)
        return tensor

    def _replicate_tensor(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        src_data_rank: int | None = 0,
    ) -> torch.Tensor:
        return Replicate._make_replicate_tensor(tensor, mesh, mesh_dim, src_data_rank)


class Partial(torch._C._distributed.Partial):
    # reduce_ops that distribute over addition, enabling per-input linearity
    # for bilinear ops like mm: reduce_op(A_i @ B) = reduce_op(A_i) @ B
    LINEAR_REDUCE_OPS: tuple[str, ...] = ("sum", "avg")
    ALL_REDUCE_OPS: tuple[str, ...] = ("sum", "avg", "min", "max", "product")

    """
    The ``Partial(reduce_op)`` placement describes the DTensor that is pending
    reduction on a specified ``DeviceMesh`` dimension, where each rank on the
    DeviceMesh dimension holds the partial value of the global Tensor. User can
    redistribute the ``Partial`` DTensor to a ``Replicate`` or ``Shard(dim)``
    placement on the specified ``DeviceMesh`` dimension using ``redistribute``,
    which would trigger necessary communication operations under the hood (i.e.
    ``allreduce``, ``reduce_scatter``).

    Args:
        reduce_op (str, optional): The reduction op to be used for the partial DTensor
            to produce Replicated/Sharded DTensor. Corresponds to the reduce operations
            supported by ``torch.distributed.ReduceOp``. Default: "sum".

            Supported values:

            * ``"sum"``: Element-wise sum across all ranks.
            * ``"avg"``: Element-wise average across all ranks.
            * ``"min"``: Element-wise minimum across all ranks.
            * ``"max"``: Element-wise maximum across all ranks.
            * ``"product"``: Element-wise product across all ranks.
            * ``"band"``: Bitwise AND across all ranks (integer tensors only).
            * ``"bor"``: Bitwise OR across all ranks (integer tensors only).
            * ``"bxor"``: Bitwise XOR across all ranks (integer tensors only).

    .. note:: The ``Partial`` placement can be generated as a result of DTensor
        operators, by ``DTensor.from_local``, or by redistributing a ``Shard``
        placement to ``Partial("sum")``.
    """

    def _reduce_value(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        # Partial placement contract #1:
        # _reduce_value: reduce the value of the tensor on the mesh dimension
        return funcol.all_reduce(
            tensor, reduceOp=self.reduce_op, group=(mesh, mesh_dim)
        )

    def _reduce_shard_value(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        shard_spec: Placement,
    ) -> torch.Tensor:
        # Partial placement contract #2:
        # _reduce_shard_value: reduce_scatter the value of the tensor over the mesh dimension
        shard_spec = cast(Shard, shard_spec)
        return shard_spec._reduce_shard_tensor(tensor, mesh, self.reduce_op, mesh_dim)

    def _partition_value(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        """
        Partition a replicated tensor to create partial values for Replicate → Partial.

        This is the conjugate operation of _reduce_value. The partition operation
        must satisfy the invariant that applying _reduce_value to the partitioned
        values recovers the original replicated value (modulo floating-point error).

        Mathematical analysis by reduce_op:

        * "sum": partition(v) = v / n, then sum([v/n] * n) = v
          Introduces floating-point error from the division and summation.
          Error grows with n (the number of ranks).

        * "avg": partition(v) = v, then avg([v] * n) = v
          Numerically exact (averaging identical values).

        * "min": partition(v) = v, then min([v] * n) = v
          Numerically exact.

        * "max": partition(v) = v, then max([v] * n) = v
          Numerically exact.

        * "product": Would need partition(v) = v^(1/n), but n-th root is not exact
          for general values and undefined for negative values with even n.
          NOT SUPPORTED.

        * "band"/"bor"/"bxor": Bitwise operations have no well-defined inverse
          that partitions a value such that reducing recovers the original.
          NOT SUPPORTED.
        """
        num_chunks = mesh.size(mesh_dim=mesh_dim)
        if self.reduce_op == "sum":
            return tensor / num_chunks
        elif self.reduce_op in ("avg", "min", "max"):
            return tensor
        else:
            raise ValueError(
                f"Replicate to Partial({self.reduce_op}) conversion is not supported."
            )

    def __hash__(self) -> int:
        return 1 + hash(self.reduce_op)

    def __repr__(self) -> str:
        """
        machine readable representation of the Partial placement
        """
        return f"Partial({self.reduce_op})"

    def __fx_repr__(self):
        """
        Returns FX-evaluable repr and required globals for Partial placement.
        Needed for passing this type as an input to a custom op.
        """
        return (
            f"torch.distributed.tensor.placement_types.Partial({self.reduce_op!r})",
            {},
        )

    def __str__(self) -> str:
        """
        human readable representation of the Partial placement
        """
        return f"P({self.reduce_op})"


# We keep the old _Partial name for a while for BC reason
_Partial = Partial


@dataclass(frozen=True)
class _MaskPartial(Partial):
    """
    A partial mask placement devised for rowwise sharded embedding op, where we need
    to mask and adjust the indices to the local embedding shard, embedding masking
    is a special type of the Partial placement

    NOTE: the lifecycle of this _MaskPartial placement follows the corresponding DTensor
    lifecycle, i.e. the indices_mask would only be alive during the lifetime of the DTensor.
    """

    mask_buffer: MaskBuffer = field(default_factory=MaskBuffer)

    # required fields for computing the local offset and deriving the mask
    offset_shape: torch.Size | None = None
    offset_dim: int = 0

    def __init__(
        self,
        reduce_op=None,
        mask_buffer=None,
        offset_shape=None,
        offset_dim=0,
        *args,
        **kwargs,
    ):
        super().__init__(reduce_op)
        if mask_buffer is None:
            mask_buffer = MaskBuffer()
        object.__setattr__(self, "mask_buffer", mask_buffer)
        object.__setattr__(self, "offset_shape", offset_shape)
        object.__setattr__(self, "offset_dim", offset_dim)

    @staticmethod
    @maybe_run_for_local_tensor
    def _mask_tensor(
        tensor: torch.Tensor,
        local_offset_on_dim: IntLikeType,
        local_shard_size: IntLikeType,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Build the input mask and save it for the current partial placement
        # this is so that the output of embedding op can reuse the same partial
        # placement saved mask to perform mask + reduction
        mask = (tensor < local_offset_on_dim) | (
            tensor >= local_offset_on_dim + local_shard_size
        )
        # mask the input tensor
        # pyrefly: ignore [unsupported-operation]
        masked_tensor = tensor.clone() - local_offset_on_dim
        masked_tensor[mask] = 0
        # pyrefly: ignore [bad-return]
        return mask, masked_tensor

    def _partition_value(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        if not mesh._is_current_rank_part_of_mesh():
            raise AssertionError("rank is not part of mesh")
        # override parent logic to perform partial mask for embedding
        num_chunks = mesh.size(mesh_dim)
        # get local shard size and offset on the embedding_dim
        if self.offset_shape is None:
            raise AssertionError("offset_shape needs to be set for _MaskPartial")
        local_shard_size, local_offset_on_dim = Shard.local_shard_size_and_offset(
            self.offset_shape[self.offset_dim],
            num_chunks,
            mesh._sym_get_coordinate(mesh_dim),
        )
        mask, masked_tensor = _MaskPartial._mask_tensor(
            tensor, local_offset_on_dim, local_shard_size
        )
        # materialize the mask buffer to be used for reduction
        self.mask_buffer.materialize_mask(mask)
        return masked_tensor

    def _reduce_value(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        # by the time we need reduction, we should have already saved the mask
        if self.mask_buffer.data is None:
            raise AssertionError

        # apply the mask to the tensor that pending reduction
        self.mask_buffer.apply_mask(tensor)

        # clear the mask buffer
        self.mask_buffer.release_mask()

        # perform sum reduction
        return funcol.all_reduce(
            tensor, reduceOp=self.reduce_op, group=(mesh, mesh_dim)
        )

    def _reduce_shard_value(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        shard_spec: Placement,
    ) -> torch.Tensor:
        # by the time we need reduction, we should have already saved the mask
        if self.mask_buffer.data is None:
            raise AssertionError

        # apply the mask to the tensor that pending reduction
        self.mask_buffer.apply_mask(tensor)

        # clear the mask buffer
        self.mask_buffer.release_mask()

        # call reduce_shard_tensor of the shard_spec.
        shard_spec = cast(Shard, shard_spec)
        return shard_spec._reduce_shard_tensor(tensor, mesh, self.reduce_op, mesh_dim)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _MaskPartial):
            return False

        return (
            self.reduce_op == other.reduce_op
            and self.offset_shape == other.offset_shape
            and self.offset_dim == other.offset_dim
            and self.mask_buffer is other.mask_buffer
        )

    def __hash__(self) -> int:
        return 1 + hash(
            (
                self.reduce_op,
                self.offset_shape,
                self.offset_dim,
                id(self.mask_buffer),
            )
        )

    def __repr__(self) -> str:
        """
        machine readable representation of the _MaskPartial placement
        """
        return f"_MaskPartial(reduce_op={self.reduce_op}, offset_shape={self.offset_shape}, offset_dim={self.offset_dim})"

    def __str__(self) -> str:
        """
        human readable representation of the _MaskPartial placement
        """
        return f"MaskP({self.reduce_op}, {self.offset_shape}, {self.offset_dim})"

    def __fx_repr__(self):
        """
        Returns FX-evaluable repr and required globals for Partial placement.
        Needed for passing this type as an input to a custom op.
        """
        return (
            f"torch.distributed.tensor.placement_types.MaskPartial(reduce_op={self.reduce_op}, offset_shape={self.offset_shape}, offset_dim={self.offset_dim})",
            {},
        )


def _register_placements_as_opaque():
    from torch._library.opaque_object import MemberType, register_custom_class

    allowed_members = {
        "is_shard": MemberType.USE_REAL,
        "is_partial": MemberType.USE_REAL,
        "is_replicate": MemberType.USE_REAL,
        "__eq__": MemberType.USE_REAL,
    }
    register_custom_class(Placement, typ="constant", members=allowed_members)
    register_custom_class(
        Shard, typ="constant", members=allowed_members | {"dim": MemberType.USE_REAL}
    )
    register_custom_class(Replicate, typ="constant", members=allowed_members)
    register_custom_class(
        Partial,
        typ="constant",
        members=allowed_members | {"reduce_op": MemberType.USE_REAL},
    )
    register_custom_class(
        _StridedShard,
        typ="constant",
        members=allowed_members | {"dim": MemberType.USE_REAL},
    )
    register_custom_class(
        _MaskPartial,
        typ="constant",
        members=allowed_members | {"reduce_op": MemberType.USE_REAL},
    )


_register_placements_as_opaque()
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import (
    TensorWriteData,
    WriteItem,
    WriteItemType,
)


aten = torch.ops.aten


class LocalShardsWrapper(torch.Tensor):
    """
    A wrapper class to hold local shards of a DTensor.
    This class is used largely for checkpointing purposes and implicitly subtypes
    the _Checkpointable protocol.
    """

    __slots__ = ["_local_shards", "_storage_meta"]
    _local_shards: list[torch.Tensor]
    _storage_meta: TensorStorageMetadata

    @staticmethod
    def __new__(
        cls, local_shards: list[torch.Tensor], local_offsets: list[tuple[int, ...]]
    ) -> "LocalShardsWrapper":
        if not all(
            tensor.device == local_shards[0].device for tensor in local_shards[1:]
        ):
            raise AssertionError

        # if empty shard, we create an empty tensor
        if len(local_shards) == 0:
            r = torch.Tensor._make_wrapper_subclass(
                cls,
                torch.Size([0, 0]),
            )
            r._local_shards = []
            r._storage_meta = TensorStorageMetadata(
                properties=TensorProperties(),
                size=torch.Size([0, 0]),
                chunks=[
                    ChunkStorageMetadata(
                        offsets=torch.Size([0, 0]), sizes=torch.Size([0, 0])
                    )
                ],
            )
            return r

        # we calculate the total tensor size by "concat" on second tensor dimension
        cat_tensor_shape = list(local_shards[0].size())
        if len(local_shards) > 1 and local_shards[0].ndim == 2:  # column-wise sharding
            for shard in local_shards[1:]:
                cat_tensor_shape[1] += shard.size()[1]

        # in cases of sharding optimizer rowwise, we calculate total tensor size by "concat" on first tensor dimension
        if len(local_shards) > 1 and local_shards[0].ndim == 1:  # column-wise sharding
            for shard in local_shards[1:]:
                cat_tensor_shape[0] += shard.size()[0]

        wrapper_properties = TensorProperties.create_from_tensor(local_shards[0])
        wrapper_shape = torch.Size(cat_tensor_shape)
        chunks_meta = [
            ChunkStorageMetadata(
                offsets=torch.Size(offset),
                sizes=shard.size(),
            )
            for shard, offset in zip(local_shards, local_offsets)
        ]

        r = torch.Tensor._make_wrapper_subclass(
            cls,
            torch.Size(cat_tensor_shape),
        )
        r._local_shards = local_shards
        r._storage_meta = TensorStorageMetadata(
            properties=wrapper_properties,
            size=wrapper_shape,
            chunks=chunks_meta,
        )

        return r

    # necessary for ops dispatching from this subclass to its local shards
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore[override]
        kwargs = kwargs or {}

        dispatcher = {
            torch.ops._c10d_functional.all_gather_into_tensor.default: cls.handle_all_gather_into_tensor,
            torch.ops._c10d_functional.wait_tensor.default: cls.handle_wait_tensor,
            aten._to_copy.default: cls.handle_to_copy,
            aten.view.default: cls.handle_view,
            aten.equal.default: cls.handle_equal,
            aten.detach.default: cls.handle_detach,
            aten.clone.default: cls.handle_clone,
            aten.new_empty.default: cls.handle_new_empty,
        }

        if func in dispatcher:
            return dispatcher[func](args, kwargs)
        else:
            raise NotImplementedError(
                f"{func} is not supported for LocalShardsWrapper!"
            )

    @staticmethod
    def handle_all_gather_into_tensor(args, kwargs) -> torch.Tensor:
        dim = args[0].local_sizes()[0][1]
        cat_tensor = torch.cat(
            [t.view(-1) for t in args[0].local_shards()], dim=0
        ).view(-1, dim)
        return torch.ops._c10d_functional.all_gather_into_tensor.default(
            cat_tensor, *args[1:], **kwargs
        )

    @staticmethod
    def handle_wait_tensor(args, kwargs) -> torch.Tensor:
        return torch.ops._c10d_functional.wait_tensor(args[0])

    @staticmethod
    def handle_to_copy(args, kwargs) -> torch.Tensor:
        res_shards_list = [
            aten._to_copy.default(shard, *args[1:], **kwargs)
            for shard in args[0].local_shards()
        ]
        return LocalShardsWrapper(res_shards_list, args[0].local_offsets())

    @staticmethod
    def handle_view(args, kwargs) -> "LocalShardsWrapper":
        view_shape = args[1]
        res_shards_list = []
        if len(args[0].local_shards()) > 1:
            if args[0].local_shards()[0].ndim == 2:
                if not (
                    args[0].storage_metadata().size[0] == view_shape[0]
                    and args[0].storage_metadata().size[1] == view_shape[1]
                ):
                    raise AssertionError
                # This accounts for a DTensor quirk, when multiple shards are present on a rank, DTensor on
                # init calls view_as() on the global tensor shape
                # will fail because the view shape is not applicable to individual shards.
                res_shards_list = [
                    aten.view.default(shard, shard.shape, **kwargs)
                    for shard in args[0].local_shards()
                ]
            elif args[0].local_shards()[0].ndim == 1:
                if args[0].storage_metadata().size[0] != view_shape[0]:
                    raise AssertionError
                # This case is for optimizer sharding as regardless of sharding type, optimizer state is row wise sharded
                res_shards_list = [
