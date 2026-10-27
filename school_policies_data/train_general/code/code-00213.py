                if node.op == "placeholder":
                    continue
                new_n = fw_g.node_copy(node, lambda n: env[n])
                fw_g_names.add(new_n.name)
                env[node] = new_n
                # Output node is temporarily copied to have remapped arguments.
                # Removed in the end.
                if node.op == "output":
                    fw_pack_out_args = new_n.args[0]
                    fw_g.erase_node(new_n)

        env.clear()
        if not fw_pack_out_args:
            raise AssertionError("fw_pack_out_args must not be empty")
        fw_outs_bw_ins_node_names = []
        for out_idx, _n in enumerate(pytree.tree_leaves(fw_pack_out_args)):
            if not isinstance(_n, torch.fx.Node):
                fw_outs_bw_ins_node_names.append("")
                continue

            # This happens when hook is noop and it is either user input or user output.
            # Do not do anything with this node.
            if _n.op == "placeholder" or _n in fw_outs_inner_set:
                # This means the hook returned input primals unchanged
                # Do not rename in this case.
                n = _n
                new_node_name = _n.name
                fw_outs_bw_ins_node_names.append(new_node_name)
            else:
                # We can not specify desired name in node_copy.
                # Copying node manually to set specific name,
                # to have matching fw_outs, bw_inputs names.
                new_node_name = _gen_unused_name(f"{saved.name}_hook_{out_idx}")  # type: ignore[union-attr]
                with fw_g.inserting_before(_n):
                    n = fw_g.create_node(
                        _n.op,
                        _n.target,
                        _n.args,
                        _n.kwargs,
                        name=new_node_name,
                    )
                if n.name != new_node_name:
                    raise AssertionError(
                        f"expected n.name == {new_node_name}, got {n.name}"
                    )
                fw_outs_bw_ins_node_names.append(new_node_name)
                n.meta = copy.copy(_n.meta)
                _n.replace_all_uses_with(n)
                fw_g.erase_node(_n)
            if isinstance(n.meta["val"], torch.Tensor):
                fw_outs_packed_tensors.append(n)
            elif is_sym_node(n):
                fw_outs_packed_syms.append(n)

        # Install unpack hook graph as a prologue of backward graph
        # Saved tensors inputs are replaced with packed tensors and packed sym scalars.
        # The saved tensors inputs usages in the graph are replaced with unpack hook graph outputs.
        with _saved_tensor_hook_context(_get_extra_info()):
            unpack_gm = prepare_hook_gm(aot_config, unpack_hook_gm, (pack_out_val,))
            unpack_g = unpack_gm.graph
            maybe_log_graph(
                unpack_gm,
                f"saved_tensors_unpack_hook {saved.name}",  # type: ignore[union-attr]
                aot_config,
                lambda: f"aot_saved_tensors_hooks_unpack {saved.name}",  # type: ignore[union-attr]
                structured_logs,
            )

        def find_saved_in_bw_inputs(
            bw_inputs: list[torch.fx.Node],
        ) -> torch.fx.Node | None:
            for n in bw_inputs:
                if n.name == saved.name:  # type: ignore[union-attr]
                    return n

        bw_g_input = find_saved_in_bw_inputs(bw_g_inputs)
        if not bw_g_input:
            raise AssertionError(
                f"could not find saved tensor {saved.name} in bw_g_inputs"  # type: ignore[union-attr]
            )
        original_bw_g_input_users = list(bw_g_input.users.keys())
        bw_g_input_used_directly = False

        # Replace backward graph saved tensor input with copy of pack graph outputs
        # All non-Tensor, non-symscalars outputs are constanted.

        unpack_g_inputs = unpack_g.find_nodes(op="placeholder")
        env = {}
        for out_idx, (unp_in_n, out_n, val) in enumerate(
            zip(
                unpack_g_inputs,
                pytree.tree_leaves(fw_pack_out_args),
                pytree.tree_leaves(pack_out_val),
            )
        ):
            is_sym = isinstance(val, py_sym_types)
            if isinstance(val, torch.Tensor) or is_sym:
                # We want forward_outputs names to match backward_inputs,
                # Potentially backward may already have "{saved.name}_hook_{idx}",
                # In this case fx.Graph will add suffix.
                new_node_name = fw_outs_bw_ins_node_names[out_idx]
                if bw_g_input.name == new_node_name:
                    env[unp_in_n] = bw_g_input
                    bw_g_input_used_directly = True
                else:
                    # Backward calling convention: ctx_symints,ctx_saved_tensors
                    # Inserting packed sym scalars before first saved tensor input.
                    # Inserting packed tensors before last saved tensor input.
                    # Saved tensor inputs between them will be removed.
                    with (
                        bw_g.inserting_before(bw_g_inputs[0])
                        if is_sym
                        else bw_g.inserting_before(bw_g_input)
                    ):
                        new_n = bw_g.placeholder(new_node_name)
                        if new_n.name != new_node_name:
                            raise AssertionError(
                                f"expected new_n.name == {new_node_name}, got {new_n.name}"
                            )
                    new_n.meta = copy.copy(out_n.meta)
                    env[unp_in_n] = new_n
            else:
                # Inline values of non-Tensor, non-SymScalars
                env[unp_in_n] = val

        # Inserting unpack hook after placeholders.
        bw_unpack_out_n = None
        with bw_g.inserting_before(bw_g_inputs[-1].next):
            for node in unpack_g.nodes:
                if node.op == "placeholder":
                    continue
                new_n = bw_g.node_copy(node, lambda n: env[n])
                bw_g_names.add(new_n.name)
                env[node] = new_n
                # Temporary insert output, to have remapped by node_copy args.
                # Removed in the end.
                if node.op == "output":
                    bw_unpack_out_n = new_n

        if not bw_unpack_out_n:
            raise AssertionError("bw_unpack_out_n must not be None")
        _leaves = pytree.tree_leaves(bw_unpack_out_n.args)
        if len(_leaves) != 1:
            raise AssertionError(f"expected exactly 1 leaf, got {len(_leaves)}")
        unpack_saved_tensor_n = _leaves[0]

        if not bw_g_input_used_directly:
            bw_g_input.replace_all_uses_with(unpack_saved_tensor_n)
            bw_g.erase_node(bw_g_input)
        else:
            # Keep usages of bw_g_input in inserted unpacked hook graph.
            # Replace other usages of bw_g_input with unpack_saved_tensor_n.
            for use_node in original_bw_g_input_users:
                use_node._replace_input_with(bw_g_input, unpack_saved_tensor_n)
        bw_g.erase_node(bw_unpack_out_n)

    # Changing forward graph outputs,
    # Inserting packed_tensors and packed_syms on the place of saved tensors.
    # Packed sym_scalars are together with saved symints
    symint_outs_saved_for_bw = [n for n in fw_outs_saved_for_bw if is_sym_node(n)]  # type: ignore[arg-type]
    fw_new_outs = pytree.tree_leaves(
        (
            fw_outs[:num_inner_fwd_outputs],  # type: ignore[index]
            fw_outs_packed_tensors,
            fw_outs_packed_syms,
            symint_outs_saved_for_bw,
        )
    )
    fw_out_n.args = (tuple(fw_new_outs),)

    # Assert that saved tensors and symints in forward outputs are aligned with backward inputs
    _fw_n = num_inner_fwd_outputs
    _fw_num_t = len(fw_outs_packed_tensors)
    _fw_num_s = len(fw_outs_packed_syms) + len(symint_outs_saved_for_bw)
    fw_outs_saved_tensors = fw_new_outs[_fw_n : _fw_n + _fw_num_t]
    fw_outs_saved_syms = fw_new_outs[_fw_n + _fw_num_t :]
    bw_new_ins = list(bw_g.find_nodes(op="placeholder"))
    bw_ins_saved_syms = bw_new_ins[:_fw_num_s]
    bw_ins_saved_tensors = bw_new_ins[_fw_num_s : _fw_num_s + _fw_num_t]

    fw_t_names = [n.name for n in fw_outs_saved_tensors]
    bw_t_names = [n.name for n in bw_ins_saved_tensors]
    fw_s_names = [n.name for n in fw_outs_saved_syms]
    bw_s_names = [n.name for n in bw_ins_saved_syms]

    def _log_structured_logs() -> None:
        if not aot_config.enable_log:
            return

        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "aot_saved_tensors_hooks_graphs",
                "encoding": "string",
            },
            payload_fn=lambda: "\n".join(structured_logs),
        )

    if aot_config.enable_log:
        structured_logs.append(
            f"fw_outs[:num_inner_fwd_outputs]:{fw_outs[:num_inner_fwd_outputs]}"  # type: ignore[index]
        )
        structured_logs.append(f"fw_outs_packed_tensors:{fw_outs_packed_tensors}")
        structured_logs.append(f"fw_t_names:{fw_t_names}")
        structured_logs.append(f"bw_t_names:{bw_t_names}")
        structured_logs.append(f"fw_s_names:{fw_s_names}")
        structured_logs.append(f"bw_s_names:{bw_s_names}")
        structured_logs.append(f"\nfw_g_pre_assert:{fw_g}")
        structured_logs.append(f"\nbw_g_pre_assert:{bw_g}")
        maybe_log_graph(
            fw_module,
            "Forward graph after transform pre-assert",
            aot_config,
            lambda: "aot_forward_graph_pre_assert_saved_tensors_hooks",
            structured_logs,
        )
        maybe_log_graph(
            bw_module,
            "Backward graph after transform pre-assert",
            aot_config,
            lambda: "aot_backward_graph_pre_assert_saved_tensors_hooks",
            structured_logs,
        )
        _log_structured_logs()

    if fw_t_names != bw_t_names:
        raise AssertionError(
            f"expected fw_t_names == bw_t_names, got {fw_t_names} != {bw_t_names}"
        )
    if fw_s_names != bw_s_names:
        raise AssertionError(
            f"expected fw_s_names == bw_s_names, got {fw_s_names} != {bw_s_names}"
        )

    fw_g.lint()
    bw_g.lint()
    fw_module.recompile()
    bw_module.recompile()


def _log_joint_graph(
    fx_g: torch.fx.GraphModule,
    aot_config: AOTConfig,
) -> str | None:
    """
    Log the joint graph to the structured logger.
    Return a str representation of the graph.
    """
    joint_graph_str = None
    if aot_config.enable_log:
        aot_joint_log.info(
            "%s",
            lazy_format_graph_code(
                "Joint graph",
                fx_g,
                aot_config.aot_id,
                include_stride=True,
                include_device=True,
                colored=True,
            ),
        )
        joint_graph_str = fx_g.print_readable(
            print_output=False,
            include_stride=True,
            include_device=True,
            expanded_def=True,
        )
        trace_structured(
            "aot_joint_graph",
            payload_fn=lambda: joint_graph_str,
        )
    return joint_graph_str


def _log_fw_bw_graphs(
    fw_module: torch.fx.GraphModule,
    bw_module: torch.fx.GraphModule,
    maybe_subclass_meta: SubclassMeta | None,
    fw_metadata: ViewAndMutationMeta,
    aot_config: AOTConfig,
) -> tuple[str | None, str | None]:
    """
    Log the fw and bw graphs to the structured logger.
    Return str representations of the graphs.
    """
    fw_module_str = None
    bw_module_str = None
    if aot_config.enable_log:
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "torch._functorch.config",
                "encoding": "string",
            },
            payload_fn=lambda: torch._functorch.config.get_serializable_config_copy(),
        )
        aot_graphs_log.info(
            "aot_config id: %s, fw_metadata=%s, inner_meta=%s",
            aot_config.aot_id,
            fw_metadata,
            _get_inner_meta(maybe_subclass_meta, fw_metadata),
        )

        aot_graphs_log.info(
            "%s",
            lazy_format_graph_code(
                "Forward graph",
                fw_module,
                aot_config.aot_id,
                include_stride=True,
                include_device=True,
                colored=True,
            ),
        )
        aot_graphs_log.info(
            "%s",
            lazy_format_graph_code(
                "Backward graph",
                bw_module,
                aot_config.aot_id,
                include_stride=True,
                include_device=True,
                colored=True,
            ),
        )
        fw_module_str = fw_module.print_readable(
            print_output=False,
            include_stride=True,
            include_device=True,
            expanded_def=True,
        )
        bw_module_str = bw_module.print_readable(
            print_output=False,
            include_stride=True,
            include_device=True,
            expanded_def=True,
        )

        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "aot_forward_graph_fw_metadata",
                "encoding": "string",
            },
            payload_fn=lambda: dataclass_repr(fw_metadata),
        )
        if maybe_subclass_meta is not None:
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "aot_forward_graph_fw_subclass_metadata",
                    "encoding": "string",
                },
                payload_fn=lambda: dataclass_repr(maybe_subclass_meta),
            )

        trace_structured(
            "aot_forward_graph",
            payload_fn=lambda: fw_module_str,
        )
        trace_structured(
            "aot_backward_graph",
            payload_fn=lambda: bw_module_str,
        )
    return fw_module_str, bw_module_str


def _partition_joint_graph_into_fw_bw(
    fx_g: torch.fx.GraphModule,
    joint_inputs: list[Any] | tuple[list[Any], list[Any]],
    inner_meta: ViewAndMutationMeta,
    fw_metadata: ViewAndMutationMeta,
    aot_config: AOTConfig,
    # pyrefly: ignore [implicit-any]
    partition_fn: Callable,
) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule, int]:
    # See Note: [Partitioner handling for Subclasses, Part 1]
    # See Note: [Recomputing subclass mutation handling]
    mutated_inp_runtime_indices = compute_inner_mutated_inp_indices_from_subclass_meta(
        fw_metadata, inner_meta
    )
    num_tokens = len(fw_metadata.tokens)
    num_inner_fwd_outputs = (
        len(mutated_inp_runtime_indices)
        + inner_meta.num_outputs
        + inner_meta.num_intermediate_bases
        + inner_meta.num_outputs_rng_offset
        + num_tokens  # See Note [Side-Effectful Tokens in AOTAutograd]
    )

    fx_g = run_joint_graph_passes_on_hops(
        fx_g, joint_inputs, aot_config, default_partition_fn=partition_fn
    )

    # apply joint_gm callback here
    if callable(torch._functorch.config.joint_custom_pass):
        # pyrefly: ignore [bad-assignment]
        fx_g = torch._functorch.config.joint_custom_pass(fx_g, joint_inputs)

    fw_module, bw_module = partition_fn(
        fx_g,
        joint_inputs,
        num_fwd_outputs=num_inner_fwd_outputs,
        static_lifetime_input_indices=fw_metadata.static_input_indices,
    )

    rng_states = [
        n
        for n in fw_module.graph.find_nodes(op="placeholder")
        if "fwd_rng_state" in n.name
    ]
    fw_metadata.num_graphsafe_rng_states = len(rng_states)
    if rng_states:
        rng_device = rng_states[0].meta["val"].device
        fw_metadata.graphsafe_rng_state_index = rng_device.index
        fw_metadata.graphsafe_rng_device = rng_device

    return fw_module, bw_module, num_inner_fwd_outputs


def _joint_inputs_for_forward(
    joint_inputs: list[Any] | tuple[list[Any], list[Any]],
) -> list[Any]:
    return joint_inputs[0] if isinstance(joint_inputs, tuple) else joint_inputs


def _maybe_unlift_partitioned_effect_tokens(
    fw_module: torch.fx.GraphModule,
    bw_module: torch.fx.GraphModule,
    joint_inputs: list[Any] | tuple[list[Any], list[Any]],
    fw_metadata: ViewAndMutationMeta,
    aot_config: AOTConfig,
    num_inner_fwd_outputs: int,
) -> tuple[int, list[Any] | tuple[list[Any], list[Any]]]:
    num_tokens = len(fw_metadata.tokens)

    # See Note [Side-Effectful Tokens in AOTAutograd]
    if config.unlift_effect_tokens and (
        num_tokens > 0 or fw_metadata.num_backward_tokens > 0
    ):
        unlift_tokens(fw_module, fw_metadata, aot_config, bw_module)
        num_inner_fwd_outputs -= num_tokens
        if isinstance(joint_inputs, tuple):
            joint_inputs = (
                _joint_inputs_for_forward(joint_inputs)[num_tokens:],
                joint_inputs[1],
            )
        else:
            joint_inputs = joint_inputs[num_tokens:]

    return num_inner_fwd_outputs, joint_inputs


def _categorize_saved_tensors_for_backward(
    fw_module: torch.fx.GraphModule,
    bw_module: torch.fx.GraphModule,
    inner_meta: ViewAndMutationMeta,
    fw_metadata: ViewAndMutationMeta,
    num_inner_fwd_outputs: int,
) -> tuple[int, int]:
    fw_outs = next(iter(fw_module.graph.find_nodes(op="output"))).args[0]
    # we only need to bookkeep the symints that are saved for bw, not any symints
    # the user forward might have returned in its own output
    fw_outs_saved_for_bw = fw_outs[num_inner_fwd_outputs:]
    num_fw_outs_saved_for_bw = len(fw_outs_saved_for_bw)

    num_symints_saved_for_bw = 0
    num_opaque_objects_saved_for_bw = 0
    saved_tensor_is_graph_input: list[bool] = []
    for idx, node in enumerate(fw_outs_saved_for_bw):
        if is_sym_node(node):
            num_symints_saved_for_bw += 1
        elif is_opaque_node(node):
            num_opaque_objects_saved_for_bw += 1
        elif isinstance(node, torch.fx.Node) and "val" in getattr(node, "meta", {}):
            if is_fake_tensor(node.meta["val"]):
                # If the saved_tensor is a view, a graph intermediate,
                # and returned from the autograd.Function output, we need to
                # detach() it to prevent a reference cycle. Record
                # if the saved_tensor is a graph input here to help.
                saved_tensor_is_graph_input.append(node.op == "placeholder")
                # record dynamic tensor activations
                dynamic_dims: set[int] = {
                    dim
                    for dim, size in enumerate(node.meta["val"].shape)
                    if not isinstance(size, int)
                }
                if dynamic_dims:
                    fw_metadata.dynamic_saved_tensors_idxs[idx] = dynamic_dims
            elif isinstance(node.meta["val"], (FakeScriptObject, CustomClassBase)):
                num_opaque_objects_saved_for_bw += 1
        else:
            saved_tensor_is_graph_input.append(False)

    fw_metadata.num_symints_saved_for_bw = num_symints_saved_for_bw
    fw_metadata.num_opaque_objects_saved_for_bw = num_opaque_objects_saved_for_bw
    num_tensors_saved_for_bw = (
        num_fw_outs_saved_for_bw
        - num_symints_saved_for_bw
        - num_opaque_objects_saved_for_bw
    )
    if len(saved_tensor_is_graph_input) != num_tensors_saved_for_bw:
        raise AssertionError(
            "expected one saved_tensor_is_graph_input entry per saved tensor, "
            f"got {len(saved_tensor_is_graph_input)} != {num_tensors_saved_for_bw}"
        )
    fw_metadata.saved_tensor_is_graph_input = saved_tensor_is_graph_input
    inner_meta.num_symints_saved_for_bw = num_symints_saved_for_bw
    inner_meta.num_opaque_objects_saved_for_bw = num_opaque_objects_saved_for_bw
    inner_meta.saved_tensor_is_graph_input = saved_tensor_is_graph_input

    # See Note [Activations with no version counter checks in eager]
    # Count tensors saved with no version counter check.
    # These are tensors that were stashed on ctx (e.g., ctx.x = x) rather than
    # via save_for_backward in an autograd.Function.
    # The partitioner sorts these to be at the end of saved_values.
    num_tensors_saved_with_no_vc_check = sum(
        1
        for node in fw_outs_saved_for_bw
        if isinstance(node, torch.fx.Node)
        and node.meta.get("saved_tensor_with_no_vc_check", False)
    )
    fw_metadata.num_tensors_saved_with_no_vc_check = num_tensors_saved_with_no_vc_check
    inner_meta.num_tensors_saved_with_no_vc_check = num_tensors_saved_with_no_vc_check

    if torch._functorch.config.donated_buffer:
        fw_metadata.bw_donated_idxs = collect_bw_donated_buffer_idxs(
            fw_module,
            bw_module,
            inner_meta,
        )
        inner_meta.bw_donated_idxs = fw_metadata.bw_donated_idxs

    return num_fw_outs_saved_for_bw, num_symints_saved_for_bw


# Note [Detaching inputs that never need gradients]
# See https://github.com/pytorch/pytorch/issues/97745
# Suppose we have a function like this that we want to compile:
#
# def f(x, y):
#     return torch.mul(x, y.detach())
#
# What gradients should we compute for x and y?
# By default, AOTAutograd will compute a gradient for **every** input that requires gradients,
# and so we'll compute:
#    x_grad_input = y
#    y_grad_input = None
# Does this preserve the semantics of eager mode?
# Unfortunately, no.
# Doing the above will cause autograd to **continue** to backprop the autograd tape
# that was generated from constructing y.
#
# This is **different** from what would have happened in eager mode.
# In eager mode, if we backprop through the output of this function, autograd will only traverse
# the bit of the autograd tape corresponding to "x".
# In particular, if a user had previously backpropped through y's autograd tape,
# And then they try to backprop through the output of the above function,
# then we'll hit the dreaded "Trying to backward through the graph a second time" error.
#
# You might think: If autograd sees that a gradient is None, shouldn't it stop early,
# instead of continuing the backprop through the ancestors of that node in the graph?
#
# Autograd has two passes:
# (1) a first pass that traverses the autograd graph and figures out which nodes need to be executed
# (2) a second pass that actually goes ahead and executes each node when it becomes ready,
#     propagating gradients
# By the time we're executing a node and we see that it produces a None, the set of nodes to execute
# is already locked-in.
#
# The fix: instead, we can recognize statically that the graph we're compiling will never contribute
# gradients to y, and prevent autograd from trying to traverse y's autograd tape at all.
# We can do this by manually detach'ing y before sending it through the `CompiledFunction`.
#
# Note that this solution is not bulletproof.
# It's possible to construct a case where eager may or may not have tried to autograd through y,
# depending on the actual grad_outputs that were passed in during the backward.
# There is no easy fix for this: the simplest fix would be to run with `retain_graph=True`,
# allowing autograd to reuse the graph.
#
# An example of this case is:
# def f(x):
#     return x.detach() * 2, x * 3
# If we were to only backprop through outs[0], in eager, we would stop
# If we backward only on the first output, we shouldn't send a grad through x.
# But the custom autograd function doesn't know that: it will materialize zero grads for x * 3
# and we will end up with a zero grad at x.
# If we later backprop through the second output, this will also require backprop'ing through x.
# Meaning we'll need to use `retain_graph=True` to be able to backprop through x the second time.
def _compute_indices_of_inps_to_detach(
    bw_module: torch.fx.GraphModule,
    maybe_subclass_meta: SubclassMeta | None,
    inner_meta: ViewAndMutationMeta,
    fw_metadata: ViewAndMutationMeta,
) -> list[int]:
    # TODO: we should apply the below "detach inputs if their gradients are statically known to be None"
    # optimization even if we have subclass inputs/outputs (we do not handle this today).
    # Computing which our our inputs get None gradients is a bit more complicated,
    # if any of our inputs are subclasses. Why?
    # (a) we need to make sure that we call .detach() on the input subclasses, since autograd sees subclasses.
    # (b) The grad_outputs that we AOT computed in our backward graph are the desugared tensor tensors,
    #     so we need to figure out which subclass fw inputs they map to.
    if maybe_subclass_meta is not None:
        return []

    indices_of_inps_to_detach: list[int] = []

    # reversed() since we expect output at end of graph
    bw_output = next(reversed(bw_module.graph.find_nodes(op="output")))
    bw_outs = bw_output.args[0]

    num_backward_tokens = inner_meta.num_backward_tokens
    expected_bw_outs = (
        len(fw_metadata.input_info)
        + inner_meta.num_outputs_rng_offset
        + num_backward_tokens
    )
    if len(bw_outs) != expected_bw_outs:
        raise AssertionError(
            f"expected len(bw_outs) == {expected_bw_outs}, got {len(bw_outs)}"
        )

    bw_outs_no_rng_no_tokens = bw_outs
    if (inner_meta.num_outputs_rng_offset + num_backward_tokens) > 0:
        bw_outs_no_rng_no_tokens = bw_outs[
            : -(inner_meta.num_outputs_rng_offset + num_backward_tokens)
        ]
    if len(bw_outs_no_rng_no_tokens) != len(fw_metadata.input_info):
        raise AssertionError(
            f"expected len(bw_outs_no_rng_no_tokens) == {len(fw_metadata.input_info)}, "
            f"got {len(bw_outs_no_rng_no_tokens)}"
        )

    for i, bw_out in enumerate(bw_outs_no_rng_no_tokens):
        # If our input experiences a metadata mutation inside the graph (e.g. set_()),
        # we *must* not detach, otherwise it will be the detach'd input that gets the metadata mutation
        metadata_mutation_in_graph = (
            fw_metadata.input_info[i].mutation_type == MutationType.MUTATED_IN_GRAPH
            and fw_metadata.input_info[i].mutates_storage_metadata
        )
        is_non_leaf = (
            fw_metadata.input_info[i].requires_grad
            and not fw_metadata.input_info[i].is_leaf
        )
        if bw_out is None and not metadata_mutation_in_graph and is_non_leaf:
            indices_of_inps_to_detach.append(i)

    return indices_of_inps_to_detach


def _aot_stage2a_partition(
    fx_g: torch.fx.GraphModule,
    joint_inputs: list[Any] | tuple[list[Any], list[Any]],
    maybe_subclass_meta: SubclassMeta | None,
    fw_metadata: ViewAndMutationMeta,
    aot_config: AOTConfig,
    # pyrefly: ignore [implicit-any]
    partition_fn: Callable,
) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule, int, int, list[int], list[Any]]:
    """
    Partition the joint graph into a forward graph and a backward graph. Returns:
    - the forward and backward graphs
    - the number of forward outputs and the number of symints saved for backward
    - indices of inputs to detach
    - adjusted inputs to forward
    """
    disable_amp = torch._C._is_any_autocast_enabled()
    inner_meta = _get_inner_meta(maybe_subclass_meta, fw_metadata)

    with torch.no_grad():
        context = torch._C._DisableAutocast if disable_amp else nullcontext
        with context(), track_graph_compiling(aot_config, "joint"):
            fw_module, bw_module, num_inner_fwd_outputs = (
                _partition_joint_graph_into_fw_bw(
                    fx_g,
                    joint_inputs,
                    inner_meta,
                    fw_metadata,
                    aot_config,
                    partition_fn,
                )
            )
            num_inner_fwd_outputs, joint_inputs = (
                _maybe_unlift_partitioned_effect_tokens(
                    fw_module,
                    bw_module,
                    joint_inputs,
                    fw_metadata,
                    aot_config,
                    num_inner_fwd_outputs,
                )
            )

            maybe_inline_graph_saved_tensors_hooks(
                fw_module,
                bw_module,
                num_inner_fwd_outputs,
                inner_meta,
                aot_config,
                fw_metadata.static_input_indices,
            )
            num_fw_outs_saved_for_bw, num_symints_saved_for_bw = (
                _categorize_saved_tensors_for_backward(
                    fw_module,
                    bw_module,
                    inner_meta,
                    fw_metadata,
                    num_inner_fwd_outputs,
                )
            )

        _indices_of_inps_to_detach = _compute_indices_of_inps_to_detach(
            bw_module,
            maybe_subclass_meta,
            inner_meta,
            fw_metadata,
        )

    return (
        fw_module,
        bw_module,
        num_fw_outs_saved_for_bw,
        num_symints_saved_for_bw,
        _indices_of_inps_to_detach,
        _joint_inputs_for_forward(joint_inputs),
    )


def _aot_stage2b_fw_compile(
    fw_module: torch.fx.GraphModule,
    adjusted_flat_args: list[Any],
    maybe_subclass_meta: SubclassMeta | None,
    fw_metadata: ViewAndMutationMeta,
    num_fw_outs_saved_for_bw: int,
    aot_config: AOTConfig,
    # pyrefly: ignore [implicit-any]
    fw_compiler: Callable,
    # pyrefly: ignore [implicit-any]
) -> tuple[list[tuple[int, ...] | None] | None, Callable]:
    return _aot_stage2b_compile_forward_or_inference(
        fw_module,
        adjusted_flat_args,
        maybe_subclass_meta,
        fw_metadata,
        aot_config,
        fw_compiler,
        is_inference=False,
        num_fw_outs_saved_for_bw=num_fw_outs_saved_for_bw,
    )


def _aot_stage2b_bw_compile(
    bw_module: torch.fx.GraphModule,
    maybe_subclass_meta: SubclassMeta | None,
    fw_metadata: ViewAndMutationMeta,
    fwd_output_strides: list[tuple[int, ...] | None] | None,
    num_symints_saved_for_bw: int,
    aot_config: AOTConfig,
    # pyrefly: ignore [implicit-any]
    bw_compiler: Callable,
    # pyrefly: ignore [implicit-any]
) -> tuple[AutogradLazyBackwardCompileInfo, Callable | None]:
    """
    Compile the backward graph. Returns:
    - the placeholder list for the backward graph
    - the compiled backward function
    """
    with torch.no_grad():
        # NB: It's important to compile backwards ahead of time, as this may
        # add extra guards which we need to apply to the Dynamo cache at
        # forwards
        with track_graph_compiling(aot_config, "backward"), torch._C._DisableAutocast():
            placeholder_list = fx_placeholder_vals(bw_module)

            forward_saved_for_backwards_strides = None
            if fwd_output_strides is not None:
                inner_meta = _get_inner_meta(maybe_subclass_meta, fw_metadata)
                forward_saved_for_backwards_strides = fwd_output_strides[
                    inner_meta.tensors_saved_for_backwards_slice
                ]

            # saved activations can have different stride to eager if
            # the compiler does layout optimization. We should restride the
            # tensor passed in for compiling the backward graph using the
            # saved tensor's stride.
            for i in range(len(placeholder_list)):
                ph_arg = placeholder_list[i]
                if not isinstance(ph_arg, torch.Tensor):
                    continue

                if forward_saved_for_backwards_strides is None:
                    continue

                real_stride = None
                # Per all_args calling convention
                j = i - num_symints_saved_for_bw
                if 0 <= j < len(forward_saved_for_backwards_strides):
                    real_stride = forward_saved_for_backwards_strides[j]
                if real_stride is None:
                    continue

                # Comparing ph_arg.stride() with real_stride directly may
                # cause dynamic dimensions in ph_arg being specialized to static
                # value. Using suppress_guards and guard_or_true to avoid that.

                stride_different = False
                fake_mode = detect_fake_mode()
                suppress_ctx = (
                    fake_mode.shape_env.suppress_guards()
                    if fake_mode is not None and fake_mode.shape_env is not None
                    else nullcontext()
                )

                # Inductor can choose different strides for activations than
                # what backward graph has. if we can't statically tell that
                # strides are the same, we assume they are not.
                with suppress_ctx:
                    for k in range(len(ph_arg.stride())):
                        # real_stride can't be symbolic.

                        if guard_or_true(ph_arg.stride()[k] != int(real_stride[k])):
                            stride_different = True
                            break

                if stride_different:
                    # Note that here we use the stride of the real tensor to
                    # restride a FakeTensor. This does not cause trouble
                    # for dynamic shape since this code path only get
                    # executed if layout optimization is enabled. And we
                    # disable layout optimization for dynamic shape right
                    # now.
                    #
                    # A solution that decide stride order based on real
                    # tensor's stride and then apply that stride order to
                    # the FakeTensor does not work smoothly since some
                    # tensor's layout is not 'dense'. E.g. mixnet_l has a
                    # tensor with size [8, 64, 112, 112] and strides
                    # (2408448, 1, 21504, 192). The solution mentioned will
                    # decide a stride of (802816, 1, 7168, 64) for this
                    # tensor which is wrong.

                    ph_size = ph_arg.size()

                    placeholder_list[i] = ph_arg.as_strided(ph_size, real_stride)
            compiled_bw_func = None
            if (
                num_symints_saved_for_bw > 0
                or aot_config.force_non_lazy_backward_lowering
            ):
                try:
                    # See Note: [Backward graph lazy lowering]
                    with torch._subclasses.fake_tensor.unset_fake_temporarily():
                        # If bw_module contains lifted constants, they will be real tensors stored as
                        # GraphModule. Deepcopying tensors under fake mode is not supported and will
                        # raise when attempting to set storage.
                        bw_module_copy = copy.deepcopy(bw_module)
                    compiled_bw_func = bw_compiler(bw_module_copy, placeholder_list)
                    del bw_module_copy
                except Exception as e:
                    if aot_config.force_non_lazy_backward_lowering:
                        raise
                    exc = e
                    trace_structured(
                        "artifact",
                        metadata_fn=lambda: {
                            "name": "eager_compile_backwards_failure",
                            "encoding": "string",
                        },
                        payload_fn=lambda: "\n".join(
                            traceback.format_exception(
                                type(exc), exc, exc.__traceback__
                            )
                        ),
                    )
                    log.warning(
                        "failed to eagerly compile backwards for dynamic, suppressing in case backwards not needed",
                        exc_info=True,
                    )
            # Compiled autograd will run the bw_module in the backward pass,
            # so recompilation need happen anyway if the backward pass is ever
            # called.
            #
            # The reason we do the GraphModule recompilation here is because
            # the lazy recompilation will cause issue in the backward pass
            # with compiled autograd.
            #
            # Do the _LazyGraphModule.force_recompile here rather than when
            # bw_module is first generated by the partitioner because the bw_module.recompile
            # may be called in some code path later and cause the _LazyGraphModule.forward
            # becomes the lazy version again. One example is when dynamic shape is enabled
            # upfront, the bw_compiler will be called above which can cause extra
            # graph module recompilation on bw_module.
            if torch._dynamo.compiled_autograd.in_compiled_autograd_region:
                from torch.fx._lazy_graph_module import _LazyGraphModule

                _LazyGraphModule.force_recompile(bw_module)

            saved_context = TracingContext.try_get()
            saved_compile_context = CompileContext.try_get()

            lazy_backward_info = AutogradLazyBackwardCompileInfo(
                # pyrefly: ignore [bad-argument-type]
                bw_module,
                placeholder_list,
                saved_context,
                saved_compile_context,
            )

            return lazy_backward_info, compiled_bw_func


def aot_stage2_autograd(
    aot_state: AOTState,
    aot_graph_capture: AOTGraphCapture,
    # pyrefly: ignore [implicit-any]
    partition_fn: Callable,
    # pyrefly: ignore [implicit-any]
    fw_compiler: Callable,
    # pyrefly: ignore [implicit-any]
    bw_compiler: Callable,
) -> DispatchReturn:
    """
    Autograd logic. Generates a joint graph, partitions it, manipulates the input with various wrappers,
    and returns a wrapped torch.autograd.Function with a forward and backward.
    """

    fx_g = aot_graph_capture.graph_module
    maybe_subclass_meta = aot_graph_capture.maybe_subclass_meta
    fw_metadata = aot_state.fw_metadata
    aot_config = aot_state.aot_config

    CompileEventLogger.try_add_pt2_compile("backend_compile", dispatch_mode="autograd")
    joint_graph_str = _log_joint_graph(fx_g, aot_config)

    _apply_tensorify_python_scalars(fx_g)

    (
        fw_module,
        bw_module,
        num_fw_outs_saved_for_bw,
        num_symints_saved_for_bw,
        _indices_of_inps_to_detach,
        adjusted_flat_args,
    ) = _aot_stage2a_partition(
        fx_g,
        aot_graph_capture.updated_flat_args,
        maybe_subclass_meta,
        fw_metadata,
        aot_config,
        partition_fn,
    )

    min_cut_info_str = getattr(fx_g.graph, "_min_cut_info_str", None)

    fw_module_str, bw_module_str = _log_fw_bw_graphs(
        fw_module, bw_module, maybe_subclass_meta, fw_metadata, aot_config
    )

    fwd_output_strides, compiled_fw_func = _aot_stage2b_fw_compile(
        fw_module,
        adjusted_flat_args,
        maybe_subclass_meta,
        fw_metadata,
        num_fw_outs_saved_for_bw,
        aot_config,
        fw_compiler,
    )

    lazy_backward_info, compiled_bw_func = _aot_stage2b_bw_compile(
        bw_module,
        maybe_subclass_meta,
        fw_metadata,
        fwd_output_strides,
        num_symints_saved_for_bw,
        aot_config,
        bw_compiler,
    )

    try_save_cache_entry, entry = _cache_autograd_info(
        aot_config,
        aot_state.flat_args,
        compiled_fw_func,
        compiled_bw_func,
        fw_module_str,
        bw_module_str,
        joint_graph_str,
        aot_graph_capture.wrappers,
        maybe_subclass_meta,
        fw_metadata,
        num_fw_outs_saved_for_bw,
        _indices_of_inps_to_detach,
        num_symints_saved_for_bw,
        bw_module,
        min_cut_info_str,
    )

    return _aot_stage2c_make_autograd_function(
        aot_config,
        aot_state.flat_args,
        fw_metadata,
        maybe_subclass_meta,
        aot_graph_capture.wrappers,
        compiled_fw_func,
        compiled_bw_func,
        bw_compiler,
        lazy_backward_info,
        try_save_cache_entry,  # type: ignore[arg-type]
        entry,  # type: ignore[arg-type]
        _indices_of_inps_to_detach,
        num_symints_saved_for_bw,
    )


def _aot_stage2c_make_autograd_function(
    aot_config: AOTConfig,
    flat_args: list[Any],
    fw_metadata: ViewAndMutationMeta,
    maybe_subclass_meta: SubclassMeta | None,
    wrappers: list[CompilerWrapper],
    compiled_fw_func: Callable[..., Any],
    compiled_bw_func: Callable[..., Any] | None,
    # pyrefly: ignore [implicit-any]
    bw_compiler: Callable,
    lazy_backward_info: AutogradLazyBackwardCompileInfo | None,
    try_save_cache_entry: Callable[..., Any],
    entry: GenericAOTAutogradResult[Any, Any] | None,
    _indices_of_inps_to_detach: list[int],
    num_symints_saved_for_bw: int,
) -> DispatchReturn:
    backward_state_indices = [
        idx for idx, x in enumerate(flat_args) if isinstance(x, BackwardState)
    ]
    if len(backward_state_indices) > 1:
        raise AssertionError(
            f"expected at most 1 backward_state_index, got {len(backward_state_indices)}"
        )

    disable_amp = torch._C._is_any_autocast_enabled()
    compile_spec = AOTDispatchAutogradCompileSpec(
        compiled_fw_func=compiled_fw_func,
        compiled_bw_func=compiled_bw_func,
        maybe_subclass_meta=maybe_subclass_meta,
        num_symints_saved_for_bw=num_symints_saved_for_bw,
        backward_state_indices=backward_state_indices,
        disable_amp=disable_amp,
        indices_of_inps_to_detach=_indices_of_inps_to_detach,
        lazy_backward_info=lazy_backward_info,
        bw_compiler=bw_compiler,
        aot_config=aot_config,
        fw_metadata=fw_metadata,
        try_save_cache_entry=try_save_cache_entry,
    )
    compiled_fn = AOTDispatchAutograd.post_compile(compile_spec)

    if entry is not None:
        compiled_fn = SerializableCompiledFunction(compiled_fn, lambda: entry)

    if config.debug_assert:
        flat_requires_grad: list[bool | None] = [
            a.requires_grad if isinstance(a, Tensor) else None for a in flat_args
        ]
        compiled_fn = DebugAssertWrapper(
            flat_requires_grad=flat_requires_grad
        ).post_compile(compiled_fn, aot_config, runtime_metadata=fw_metadata)

    compiled_fn = post_compile(
        wrappers,
        compiled_fn,
        aot_config,
        runtime_metadata=fw_metadata,
    )
    return compiled_fn


def _cache_autograd_info(
    aot_config: AOTConfig,
    flat_args: list[Any],
    compiled_fw_func: Callable[..., Any],
    compiled_bw_func: Callable[..., Any] | None,
    fw_module_str: str | None,
    bw_module_str: str | None,
    joint_graph_str: str | None,
    wrappers: list[CompilerWrapper],
    maybe_subclass_meta: SubclassMeta | None,
    fw_metadata: ViewAndMutationMeta,
    num_fw_outs_saved_for_bw: int,
    _indices_of_inps_to_detach: list[int],
    num_symints_saved_for_bw: int,
    bw_module: torch.fx.GraphModule | None,
    min_cut_info_str: str | None,
) -> tuple[
    GenericAOTAutogradResult[Any, Any] | None,
    Callable[..., Any],
]:
    backward_state_indices = [
        idx for idx, x in enumerate(flat_args) if isinstance(x, BackwardState)
    ]
    if len(backward_state_indices) > 1:
        raise AssertionError(
            f"expected at most 1 backward_state_index, got {len(backward_state_indices)}"
        )

    make_runtime_safe(fw_metadata, maybe_subclass_meta)

    try_save_cache_entry: Callable[..., Any] | None = None
    entry: GenericAOTAutogradResult[Any, Any] | None = None

    if aot_config.cache_info is not None:
        forward_time_taken_ns = time.time_ns() - aot_config.cache_info.start_time_ns

        # NB: aot_config here is technically not needed as an argument: we could just
        # close over aot_config.cache_info, since aot_config never changes.
        # But closing over random variables is confusing IMO, so I'm leaving it.
        def try_save_cache_entry(
            compiled_bw_func: Callable[..., Any],
            bw_module: torch.fx.GraphModule,
            _fw_metadata: ViewAndMutationMeta,
            aot_config: AOTConfig,
        ) -> GenericAOTAutogradResult[Any, Any] | None:
            cache_info = aot_config.cache_info

            if cache_info is not None and _should_save_cache(
                compiled_fw_func, compiled_bw_func
            ):
                if forward_time_taken_ns is None:
                    raise AssertionError("forward_time_taken_ns must not be None")
                # TODO: technically, AOTAutograd does a *little* bit of post processing work
                # in the backward that isn't measured here. But it's small enough that it's not worth
                # the complexity of threading a bunch of times through the code, so we
                # use the compiled_bw_func's inductor compile time instead.
                # It's possible this changes in the future, in which case we should
                # update backward_time_taken_ns to be more inclusive
                backward_time_taken_ns = getattr(compiled_bw_func, "_time_taken_ns", 0)

                aot_forward_graph_str: str | None = fw_module_str
                aot_backward_graph_str: str | None = bw_module_str
                aot_joint_graph_str: str | None = joint_graph_str
                guards_expr = AOTAutogradCache.generate_guards_expression(cache_info)

                entry = AOTAutogradCache.make_entry(
                    compiled_fw_func,  # type: ignore[arg-type]
                    compiled_bw_func,  # type: ignore[arg-type]
                    aot_joint_graph_str,
                    aot_forward_graph_str,
                    aot_backward_graph_str,
                    _fw_metadata,
                    wrappers,
                    maybe_subclass_meta,
                    num_fw_outs_saved_for_bw,
                    _indices_of_inps_to_detach,
                    forward_time_taken_ns,
                    backward_time_taken_ns,
                    sanitized_aot_config=aot_config.to_cacheable(),
                    guards_expr=guards_expr,
                    backward_state_indices=backward_state_indices,
                    num_symints_saved_for_bw=num_symints_saved_for_bw,
                    serialized_bw_module=serialize_graph_module(bw_module),
                    min_cut_info_str=min_cut_info_str,
                )
                AOTAutogradCache.save(
                    cache_info.cache_key,
                    entry,
                    remote=should_use_remote_autograd_cache(),
                )
                return entry
            return None

        if compiled_bw_func is not None:
            # If we already compiled the backward, we save its cache entry now
            if bw_module is None:
                raise AssertionError(
                    "bw_module must not be None when compiled_bw_func is not None"
                )
            entry = try_save_cache_entry(
                compiled_bw_func,
                bw_module,
                fw_metadata,
                aot_config,  # type: ignore[arg-type]
            )
            try_save_cache_entry = None

    return try_save_cache_entry, entry  # type: ignore[return-value]


def _aot_stage2b_compile_forward_or_inference(
    fw_module: torch.fx.GraphModule,
    adjusted_flat_args: list[Any],
    maybe_subclass_meta: SubclassMeta | None,
    fw_metadata: ViewAndMutationMeta,
    aot_config: AOTConfig,
    # pyrefly: ignore [implicit-any]
    compiler: Callable,
    *,
    is_inference: bool,
    num_fw_outs_saved_for_bw: int | None = None,
    # pyrefly: ignore [implicit-any]
) -> tuple[list[tuple[int, ...] | None] | None, Callable]:
    """
    Compile the forward or inference graph. Returns:
    - the output strides of the forward graph
    - the compiled forward/inference function

    Args:
        fw_module: The forward graph module to compile
        adjusted_flat_args: Flattened arguments after adjustments
        maybe_subclass_meta: Metadata for tensor subclasses
        fw_metadata: View and mutation metadata
        aot_config: AOT configuration
        is_inference: If True, compile for inference; if False, compile for forward (autograd)
        num_fw_outs_saved_for_bw: Number of forward outputs saved for backward (required if not is_inference)

    Before compiling, we run pre_compile for the following wrappers:
    - FakifiedOutWrapper
    - FunctionalizedRngRuntimeWrapper
    After compiling, we run post_compile for the following wrappers:
    - EffectTokensWrapper
    - AOTDispatchSubclassWrapper
    - FunctionalizedRngRuntimeWrapper
    - FakifiedOutWrapper
    """

    # Validation
    if not is_inference and num_fw_outs_saved_for_bw is None:
        raise ValueError(
            "num_fw_outs_saved_for_bw must be provided when is_inference=False"
        )

    # Determine grad context, autocast context, and tracking mode.
    if is_inference:
        grad_ctx: Any = nullcontext
        autocast_ctx: Any = (
            torch._C._DisableAutocast
            if torch._C._is_any_autocast_enabled()
            else nullcontext
        )
        tracking_mode: str = "inference"
    else:
        grad_ctx = torch.no_grad
        autocast_ctx = torch._C._DisableAutocast
        tracking_mode = "forward"

    with grad_ctx(), autocast_ctx(), track_graph_compiling(aot_config, tracking_mode):
        # Setup wrappers
        fakified_out_wrapper = FakifiedOutWrapper()
        fakified_out_wrapper.pre_compile(
            fw_module, adjusted_flat_args, aot_config, fw_metadata=fw_metadata
        )

        # Initialize RNG wrapper based on mode
        functionalized_rng_wrapper = FunctionalizedRngRuntimeWrapper(
            return_new_outs=is_inference
        )

        # Add RNG states for forward mode only
        if not is_inference and fw_metadata.num_graphsafe_rng_states > 0:
            index = fw_metadata.graphsafe_rng_state_index
            if index is None:
                raise AssertionError(
                    "fw_metadata.graphsafe_rng_state_index must not be None when num_graphsafe_rng_states > 0"
                )
            device = fw_metadata.graphsafe_rng_device
            if device is None:
                raise AssertionError(
                    "fw_metadata.graphsafe_rng_device must not be None when num_graphsafe_rng_states > 0"
                )
            rng_states = [
                get_default_generator(device).clone_state()
                for _ in range(fw_metadata.num_graphsafe_rng_states)
            ]
            adjusted_flat_args.extend(rng_states)  # type: ignore[arg-type]

        functionalized_rng_wrapper.pre_compile(
            fw_module, adjusted_flat_args, aot_config, fw_metadata=fw_metadata
        )

        # Set tracing context
        if tracing_context := torch._guards.TracingContext.try_get():
            tracing_context.fw_metadata = _get_inner_meta(
                maybe_subclass_meta, fw_metadata
            )

        if config.enable_complex_wrapper:
            from .complex_decomposition import decompose_complex_in_graph

            fw_module = decompose_complex_in_graph(
                fw_module, adjusted_flat_args, aot_config.decompositions
            )

        with TracingContext.report_output_strides() as fwd_output_strides:
            # pyrefly: ignore[not-callable]
            compiled_fw_func = compiler(fw_module, adjusted_flat_args)

        # Make boxed if needed
        if not getattr(compiled_fw_func, "_boxed_call", False):
            compiled_fw_func = make_boxed_func(compiled_fw_func)

        # Set forward output strides if needed
        if fakified_out_wrapper.needs_post_compile:
            fakified_out_wrapper.set_fwd_output_strides(fwd_output_strides)  # type: ignore[arg-type]

        # Apply post-compile wrappers
        compiled_fw_func = EffectTokensWrapper().post_compile(
            compiled_fw_func,
            aot_config,
            runtime_metadata=fw_metadata,
        )

        compiled_fw_func = AOTDispatchSubclassWrapper(
            fw_only=None,
            trace_joint=False,
            maybe_subclass_meta=maybe_subclass_meta,
            num_fw_outs_saved_for_bw=num_fw_outs_saved_for_bw,
        ).post_compile(
            compiled_fw_func,
            aot_config,
            runtime_metadata=fw_metadata,
        )

        compiled_fw_func = functionalized_rng_wrapper.post_compile(
            compiled_fw_func, aot_config, runtime_metadata=fw_metadata
        )

        compiled_fw_func = fakified_out_wrapper.post_compile(
            compiled_fw_func,
            aot_config,
            runtime_metadata=fw_metadata,
        )

        return fwd_output_strides, compiled_fw_func
<|endoftext|>"""Activation offloading for memory optimization during compilation.

This module provides functionality to offload activations to CPU during the forward
pass and reload them during the backward pass, reducing GPU memory usage. It can be
applied to graphs produced by both AOT Autograd partitioners and make_fx-based tracing.
"""

import logging
import operator
from dataclasses import dataclass

import torch
import torch.fx as fx
from torch._functorch._activation_offloading.offload_ops import (  # noqa: F401 -- registers ao::offload, ao::reload, ao::wait_tensor ops
    offload,
    reload,
    wait_tensor,
)
from torch._inductor.fx_passes.overlap_scheduling import benchmark_node, is_compute_node
from torch._subclasses.fake_tensor import extract_tensor_metadata
from torch.utils._ordered_set import OrderedSet

from .. import config
from ..partitioners import _size_of, get_default_op_list, OpTypes


log: logging.Logger = logging.getLogger(__name__)


# Node name prefixes for offload/reload operations
# NOTE: right now we are using these prefixes as identifiers for offload/reload
CPU_OFFLOAD_PREFIX = "cpu_offload_"
GPU_RELOAD_PREFIX = "gpu_reload_"


def _find_all_effective_users(node: fx.Node, op_types: OpTypes) -> OrderedSet[fx.Node]:
    """Find all effective users of a node, where view ops extend the lifetime
    of the original node. If a user is a view op, recursively find users of
    the view."""
    effective_users: OrderedSet[fx.Node] = OrderedSet()
    for user in node.users:
        if user.op == "output":
            continue
        effective_users.add(user)
        if op_types.is_view(user):
            effective_users.update(_find_all_effective_users(user, op_types))
    return effective_users


@dataclass
class ReloadNodeInfo:
    """
    Information about backward reload related nodes for each reload operation.

    Pattern: ao.reload → ao.wait_tensor

    - Reload group (ao.reload): Performs the actual asynchronous data transfer.
      Can be moved earlier in the graph to overlap with computation.
    - Wait node (ao.wait_tensor): Synchronization point that blocks until the data
      transfer completes. Must remain at the point where the data is first needed.
    """

    reload_group_nodes: list[fx.Node]
    wait_event_node: fx.Node
    transfer_size_bytes: int
    transfer_time_ms: float


@dataclass
class ReloadQueueEntry:
    """
    Entry in the reload queue for prefetch scheduling.

    Attributes:
        pattern: The reload pattern information
        remaining_time_ms: Remaining overlap time needed in milliseconds
    """

    pattern: ReloadNodeInfo
    remaining_time_ms: float


def offload_activation_fw(graph: fx.Graph) -> None:
    """
    Insert CPU offload operations in the forward pass graph.

    Offload operations are placed after the last effective use of each tensor marked
    for offloading. This ensures the tensor is no longer needed on the GPU before
    transferring it to CPU memory.

    NOTE: An alternative approach would offload tensors immediately after generation
    to maximize compute-communication overlap. However, this requires additional
    synchronization to ensure tensor deletion (which occurs on the default stream)
    waits for the asynchronous offload operation to complete. This would necessitate
    more complex tracking to separate operation scheduling from memory cleanup.

    Args:
        graph: The forward graph to modify
    """

    op_types: OpTypes = get_default_op_list()

    output_node: fx.Node = graph.find_nodes(op="output")[0]
    # pyrefly: ignore [bad-assignment]
    fwd_outputs: tuple[fx.Node, ...] = output_node.args[
        0
    ]  # pyrefly: ignore [bad-assignment]
    node_to_offload: dict[fx.Node, fx.Node] = dict()
    node_to_index: dict[fx.Node, int] = {
        node: idx for idx, node in enumerate(graph.nodes)
    }

    for node in fwd_outputs:
        if node.meta.get("saved_for_offloading", False) is False:
            continue

        # Find insertion point, which is the last use
        if all_effective_users := _find_all_effective_users(node, op_types):
            last_user = max(all_effective_users, key=lambda n: node_to_index[n])
        else:
            last_user: fx.Node = node

        # Insert the CPU offload operation after the last user
        with graph.inserting_after(last_user):
            cpu_node: fx.Node = graph.call_function(
                torch.ops.prims.device_put.default,
                args=(node, torch.device("cpu")),
                kwargs={"non_blocking": True},
                name=CPU_OFFLOAD_PREFIX + str(node.name),
            )
            cpu_node.meta["val"] = node.meta["val"].to(torch.device("cpu"))
            cpu_node.meta["tensor_meta"] = extract_tensor_metadata(cpu_node.meta["val"])

        node_to_offload[node] = cpu_node

    # Update the return node args
    output_node.update_arg(
        0, tuple(node_to_offload.get(node, node) for node in fwd_outputs)
    )


def reload_activation_bw(graph: fx.Graph) -> None:
    """
    Insert GPU reload operations in the backward pass graph.

    Reload operations are placed before the first use of each offloaded tensor,
    transferring it from CPU back to GPU memory before it's needed for computation.

    Args:
        graph: The backward graph to modify
    """

    node_to_index: dict[fx.Node, int] = {
        node: idx for idx, node in enumerate(graph.nodes)
    }
    output_node: fx.Node = graph.find_nodes(op="output")[0]

    for node in graph.find_nodes(op="placeholder"):
        if node.meta.get("saved_for_offloading", False) is False:
            continue

        # Find insertion point, which is the first use or output node if no users
        # The later should not happen, but inserting before output node is safe
        insert_point: fx.Node = (
            min(node.users.keys(), key=lambda n: node_to_index[n])
            if node.users
            else output_node
        )

        # Insert the GPU reload operation before the first user
        original_device: torch.Device = node.meta["original_device"]
        with graph.inserting_before(insert_point):
            gpu_node: fx.Node = graph.call_function(
                torch.ops.prims.device_put.default,
                args=(node, original_device),
                kwargs={"non_blocking": True},
                name=str(node.name).replace(CPU_OFFLOAD_PREFIX, GPU_RELOAD_PREFIX),
            )
            gpu_node.meta["val"] = node.meta["val"].to(original_device)
            gpu_node.meta["tensor_meta"] = extract_tensor_metadata(gpu_node.meta["val"])

        # Replace all uses of the CPU tensor with the GPU tensor
        for user in list(node.users.keys()):
            if user != gpu_node:
                user.replace_input_with(node, gpu_node)


def offload_activation_fw_async(graph: fx.Graph) -> None:
    """Insert async CPU offload operations in the forward pass graph.

    Uses ao.offload + ao.wait_tensor ops which encapsulate stream management
    internally, producing a clean 2-node IR per offloaded tensor.
    """

    op_types: OpTypes = get_default_op_list()

    output_node: fx.Node = graph.find_nodes(op="output")[0]
    # pyrefly: ignore [bad-assignment]
    fwd_outputs: tuple[fx.Node, ...] = output_node.args[
        0
    ]  # pyrefly: ignore [bad-assignment]
    node_to_offload: dict[fx.Node, fx.Node] = dict()
    node_to_index: dict[fx.Node, int] = {
        node: idx for idx, node in enumerate(graph.nodes)
    }

    if not any(n.meta.get("saved_for_offloading", False) for n in fwd_outputs):
        return

    for node in fwd_outputs:
        if node.meta.get("saved_for_offloading", False) is False:
            continue

        if all_effective_users := _find_all_effective_users(node, op_types):
            last_user = max(all_effective_users, key=lambda n: node_to_index[n])
        else:
            last_user: fx.Node = node

        with graph.inserting_after(last_user):
            offload_node: fx.Node = graph.call_function(
                torch.ops.ao.offload.default,
                args=(node,),
                name=f"async_{CPU_OFFLOAD_PREFIX}{node.name}",
            )
            offload_node.meta["val"] = node.meta["val"].to(torch.device("cpu"))
            offload_node.meta["tensor_meta"] = extract_tensor_metadata(
                offload_node.meta["val"]
            )
        # The keepalive=node arg extends the GPU tensor's lifetime in the
        # graph so the allocator doesn't reclaim it before the async D2H
        # copy completes.
        with graph.inserting_after(offload_node):
            wait_node: fx.Node = graph.call_function(
                torch.ops.ao.wait_tensor.default,
                args=(offload_node, node),
                name=CPU_OFFLOAD_PREFIX + str(node.name),
            )
            wait_node.meta["val"] = offload_node.meta["val"]
            wait_node.meta["tensor_meta"] = offload_node.meta["tensor_meta"]

        node_to_offload[node] = wait_node

    output_node.update_arg(
        0, tuple(node_to_offload.get(node, node) for node in fwd_outputs)
    )


def reload_activation_bw_async(graph: fx.Graph) -> None:
    """Insert async GPU reload operations in the backward pass graph.

    Uses ao.reload + ao.wait_tensor ops which encapsulate stream management internally,
    producing a clean 2-node IR per reloaded tensor.
    """

    node_to_index: dict[fx.Node, int] = {
        node: idx for idx, node in enumerate(graph.nodes)
    }

    nodes_to_reload = [
        n
        for n in graph.find_nodes(op="placeholder")
        if n.meta.get("saved_for_offloading", False)
    ]
    if not nodes_to_reload:
        return

    for node in nodes_to_reload:
        if not node.users:
            raise RuntimeError(
                f"Offloaded tensor {node.name} has no users in the backward graph"
            )
        insert_point: fx.Node = min(node.users.keys(), key=lambda n: node_to_index[n])

        original_device: torch.device = node.meta["original_device"]
        original_size: tuple[int, ...] = node.meta["original_size"]
        original_stride: tuple[int, ...] = node.meta["original_stride"]
        with graph.inserting_before(insert_point):
            reload_node: fx.Node = graph.call_function(
                torch.ops.ao.reload.default,
                args=(
                    node,
                    original_device,
                    list(original_size),
                    list(original_stride),
                ),
                name=f"async_{str(node.name).replace(CPU_OFFLOAD_PREFIX, GPU_RELOAD_PREFIX)}",
            )
            reload_node.meta["val"] = torch.empty_strided(
                original_size,
                original_stride,
                dtype=node.meta["val"].dtype,
                device=original_device,
            )
            reload_node.meta["tensor_meta"] = extract_tensor_metadata(
                reload_node.meta["val"]
            )
            wait_node: fx.Node = graph.call_function(
                torch.ops.ao.wait_tensor.default,
                args=(reload_node, node),
                name=str(node.name).replace(CPU_OFFLOAD_PREFIX, GPU_RELOAD_PREFIX),
            )
            wait_node.meta["val"] = reload_node.meta["val"]
            wait_node.meta["tensor_meta"] = reload_node.meta["tensor_meta"]

        for user in list(node.users.keys()):
            if user != reload_node and user != wait_node:
                user.replace_input_with(node, wait_node)


def can_offload(
    node: fx.Node,
    fwd_outputs: OrderedSet[fx.Node],
    model_outputs: OrderedSet[fx.Node],
    static_lifetime_input_nodes: OrderedSet[fx.Node],
) -> bool:
    """
    Determine if a node can be offloaded to CPU.

    Args:
        node: The node to check
        fwd_outputs: Forward module outputs, including model outputs and activations
        model_outputs: Model outputs

    NOTE: Additional context for the logic behind these offloading checks:

    * fwd_outputs: Only saved intermediate tensors should be offloaded.

    * model_outputs / static_lifetime_input_nodes: Tensors that may be accessed outside
      the compiled region (e.g., model outputs, static inputs) cannot be offloaded as
      they must remain accessible beyond the scope of the compiled graph.

    * views / getitems: Offloading such nodes can lead to segmentation faults.

    * contiguous: Offloading non-contiguous tensors causes CPU-side stride changes
      during both forward and backward passes when using the Inductor backend. While
      these stride changes cancel each other out, they introduce significant compute
      overhead. This is due to the contiguity check in ir.py (see link below).
      TODO: This restriction could potentially be bypassed in the future.
      Reference: https://github.com/pytorch/pytorch/blob/44ac69388a4a5eb463dbd2a13f00d1e3b924566c/torch/_inductor/ir.py#L3214

    Additional criteria to consider for offloading optimization:

    * Tensor size: Small tensors may not fully utilize available bandwidth, reducing the
      efficiency gains from offloading.

    * Position in forward/backward graph: Activations generated near the end of the forward
      pass are typically consumed near the beginning of the backward pass. Offloading such
      tensors may be counterproductive since they are quickly reloaded, not having sufficient
      time to overlap the transfer with computation.
    """

    log.debug(f"Checking node {node.name} for offloading...")  # noqa: G004

    op_types: OpTypes = get_default_op_list()

    if node not in fwd_outputs:
        log.debug("\tSkipped! Can only offload nodes in fwd_module_outputs.")
        return False
    if node in model_outputs:
        log.debug("\tSkipped! Cannot offload model outputs.")
        return False
    if node in static_lifetime_input_nodes:
        log.debug("\tSkipped! Cannot offload static input nodes.")
        return False
    if op_types.is_view(node):
        log.debug("\tSkipped! Cannot offload views.")
        return False
    if node.target == operator.getitem:
        log.debug("\tSkipped! Cannot offload getitems.")
        return False
    if hasattr(node, "meta") and "val" in node.meta:
        if (
            isinstance(val := node.meta["val"], torch.Tensor)
            and not val.is_contiguous()
        ):
            log.debug("\tSkipped! Cannot offload non-contiguous tensors.")
            return False

    log.debug("\tGood!")
    return True


def choose_offload_sets(
    fwd_module: fx.GraphModule,
    num_fwd_outputs: int,
    static_lifetime_input_nodes: OrderedSet[fx.Node],
) -> bool:
    """
    Decide which nodes will be offloaded based on the marked nodes and feasibility.
    Marks nodes with "saved_for_offloading" if they should and can be offloaded.

    Args:
        fwd_module: Forward graph module
        bwd_module: Backward graph module
        num_fwd_outputs: Number of forward outputs

    Returns:
        bool: Whether activation offloading should be performed
    """

    fwd_outputs: OrderedSet[fx.Node] = OrderedSet(
        fwd_module.graph.find_nodes(op="output")[0].args[0]
    )
    model_outputs: OrderedSet[fx.Node] = OrderedSet(
        fwd_module.graph.find_nodes(op="output")[0].args[0][:num_fwd_outputs]
    )

    should_perform_offloading = False
    for node in fwd_module.graph.nodes:
        if node.meta.get("should_offload", False) and can_offload(
            node, fwd_outputs, model_outputs, static_lifetime_input_nodes
        ):
            node.meta["saved_for_offloading"] = True
            should_perform_offloading = True

    return should_perform_offloading


def offload_chosen_sets(
    fwd_module: fx.GraphModule,
    bwd_module: fx.GraphModule,
) -> None:
    """
    Add offload and reload nodes to the forward and backward graphs.
    This function adds device_put operations without any stream handling.

    Args:
        fwd_module: Forward module graph
        bwd_module: Backward module graph
    """

    # Add offload nodes in forward graph
    offload_activation_fw(fwd_module.graph)

    # Update backward graph inputs to be offloaded tensors
    bwd_inputs: dict[str, fx.Node] = {
        node.name: node for node in bwd_module.graph.find_nodes(op="placeholder")
    }
    for fwd_node in fwd_module.graph.find_nodes(op="output")[0].args[0]:
        if CPU_OFFLOAD_PREFIX not in fwd_node.name:
            continue

        bwd_node: fx.Node = bwd_inputs[fwd_node.name.replace(CPU_OFFLOAD_PREFIX, "")]
        with bwd_module.graph.inserting_after(bwd_node):
            bwd_offload_node: fx.Node = bwd_module.graph.placeholder(name=fwd_node.name)

        bwd_offload_node.meta.update(fwd_node.meta)
        bwd_offload_node.meta["saved_for_offloading"] = True
        bwd_offload_node.meta["original_device"] = bwd_node.meta["val"].device
        bwd_offload_node.meta["original_size"] = tuple(bwd_node.meta["val"].size())
        bwd_offload_node.meta["original_stride"] = tuple(bwd_node.meta["val"].stride())
        bwd_node.replace_all_uses_with(bwd_offload_node)
        bwd_module.graph.erase_node(bwd_node)

    # Add reload nodes in backward graph
    reload_activation_bw(bwd_module.graph)


def offload_chosen_sets_async(
    fwd_module: fx.GraphModule,
    bwd_module: fx.GraphModule,
) -> None:
    """
    Add async offload and reload nodes using ao ops.

    Uses ao.offload/ao.reload + ao.wait_tensor which encapsulate stream management,
    instead of device_put + explicit stream operations. Can be applied to
    partitioned forward/backward graphs or to a joint graph produced by make_fx.
    """

    offload_activation_fw_async(fwd_module.graph)

    # Replace backward graph placeholders with their offloaded (CPU) counterparts.
    # For each offloaded forward output, find the matching backward input and swap
    # it with a new placeholder carrying the CPU tensor's metadata, then mark it
    # for reloading.
    bwd_inputs: dict[str, fx.Node] = {
        node.name: node for node in bwd_module.graph.find_nodes(op="placeholder")
    }
    for fwd_node in fwd_module.graph.find_nodes(op="output")[0].args[0]:
        if CPU_OFFLOAD_PREFIX not in fwd_node.name:
            continue

        bwd_node: fx.Node = bwd_inputs[fwd_node.name.replace(CPU_OFFLOAD_PREFIX, "")]
        with bwd_module.graph.inserting_after(bwd_node):
            bwd_offload_node: fx.Node = bwd_module.graph.placeholder(name=fwd_node.name)

        bwd_offload_node.meta.update(fwd_node.meta)
        bwd_offload_node.meta["saved_for_offloading"] = True
        bwd_offload_node.meta["original_device"] = bwd_node.meta["val"].device
        bwd_offload_node.meta["original_size"] = tuple(bwd_node.meta["val"].size())
        bwd_offload_node.meta["original_stride"] = tuple(bwd_node.meta["val"].stride())
        bwd_node.replace_all_uses_with(bwd_offload_node)
        bwd_module.graph.erase_node(bwd_node)

    reload_activation_bw_async(bwd_module.graph)


def activation_offload_sink_wait_async(fwd_module: fx.GraphModule) -> None:
    """Sink ao.wait_tensor operations for offload completion to the end of the graph.

    This allows computation to overlap with offload operations.

    NOTE: Sinking waits to the end delays GPU memory release of the source
    tensor (kept alive via the wait's keepalive arg) until the end of the
    compiled graph. For per-layer compile this is fine (one layer's worth of
    memory), but for full-model compile this means offloaded GPU tensors are
    not freed until the entire forward pass completes.
    """
    graph: fx.Graph = fwd_module.graph
    output_node: fx.Node = graph.find_nodes(op="output")[0]

    wait_nodes_to_sink: list[fx.Node] = [
        node
        for node in graph.nodes
        if node.op == "call_function"
        and node.target == torch.ops.ao.wait_tensor.default
        and isinstance(node.args[0], fx.Node)
        and node.args[0].op == "call_function"
        and node.args[0].target == torch.ops.ao.offload.default
    ]

    # prepend moves the node from its current position (no manual removal needed)
    for wait_node in wait_nodes_to_sink:
        output_node.prepend(wait_node)


def activation_reload_prefetch_async(bwd_module: fx.GraphModule) -> None:
    """
    Prefetch backward reload operations by moving ao.reload nodes earlier
    in the graph to overlap data transfer with computation, while keeping
    ao.wait_tensor at its original position.
    """
    graph: fx.Graph = bwd_module.graph
    nodes_list: list[fx.Node] = list(graph.nodes)

    # Identify reload + wait pairs
    reload_patterns: dict[fx.Node, ReloadNodeInfo] = {}
    for node in graph.nodes:
        if not (
            node.op == "call_function" and node.target == torch.ops.ao.reload.default
        ):
            continue
        wait_node = next(
            (u for u in node.users if u.target == torch.ops.ao.wait_tensor.default),
            None,
        )
        if wait_node is None:
            continue
        transfer_size_bytes: int = _calculate_transfer_size(node)
        transfer_time_ms: float = _estimate_transfer_time_in_ms(transfer_size_bytes)
        reload_patterns[node] = ReloadNodeInfo(
            reload_group_nodes=[node],
            wait_event_node=wait_node,
            transfer_size_bytes=transfer_size_bytes,
            transfer_time_ms=transfer_time_ms,
        )

    reorder_for_prefetch(nodes_list, reload_patterns)


def _calculate_transfer_size(device_put_node: fx.Node) -> int:
    """Calculate the size in bytes of data being transferred."""

    # ao.offload(tensor) -> tensor at args[0]
    # ao.reload(tensor, device) -> tensor at args[0]
    if device_put_node.target in (
        torch.ops.ao.offload.default,
        torch.ops.ao.reload.default,
    ):
        return _size_of(device_put_node.args[0])  # pyrefly: ignore [bad-argument-type]
    raise ValueError(f"Unexpected transfer op: {device_put_node.target}")


def _estimate_transfer_time_in_ms(transfer_size_bytes: int) -> float:
    """Estimate transfer time in milliseconds based on size and bandwidth.

    Uses config.activation_offload_cpu_gpu_bw (GB/s) which should be set by
    the user to match their hardware.
    """
    return (
        transfer_size_bytes / (1024**3) * 1_000 / config.activation_offload_cpu_gpu_bw
    )


def identify_reload_patterns(
    graph: fx.Graph, nodes_list: list[fx.Node], node_to_idx: dict[fx.Node, int]
) -> dict[fx.Node, ReloadNodeInfo]:
    """
    Identify backward reload patterns in the graph.

    Pattern: fork → wait_stream → device_put → record_event → join → wait_event

    This uses position-based matching since these nodes are inserted together in
    add_backward_reload_stream_ops() in a specific order. Since stream operations
    do not have data dependencies between them, they are unsuitable for subgroup
    pattern matching type of checks.

    Returns a dict mapping device_put node to ReloadNodeInfo containing:
    - reload_group_nodes: fork → wait_stream → device_put → record_event → join
    - wait_event_node: the wait_event node
    - transfer_size_bytes: size of data being transferred
    - transfer_time_ms: estimated transfer time in milliseconds
    """
    patterns: dict[fx.Node, ReloadNodeInfo] = {}

    # Find all GPU reload device_put nodes whose inputs are placeholder nodes
    reload_nodes: list[fx.Node] = [
        node
        for node in graph.find_nodes(
            op="call_function", target=torch.ops.prims.device_put.default
        )
        if GPU_RELOAD_PREFIX in node.name
        and (
            node.args
            and isinstance(node.args[0], fx.Node)
            and node.args[0].op == "placeholder"
        )
    ]

    # Extract patterns for each reload device_put node
    for reload_node in reload_nodes:
        reload_node_idx: int = node_to_idx[reload_node]

        fork_node: fx.Node = nodes_list[reload_node_idx - 2]
        wait_stream_node: fx.Node = nodes_list[reload_node_idx - 1]
        record_event_node: fx.Node = nodes_list[reload_node_idx + 1]
        join_node: fx.Node = nodes_list[reload_node_idx + 2]
        wait_event_node: fx.Node = nodes_list[reload_node_idx + 3]

        # Validate the nodes are what we expect
        # Removed in follow-up commit
        _validate_pattern_nodes(  # noqa: F821  # pyrefly: ignore [unknown-name]
            fork_node,
            wait_stream_node,
            record_event_node,
            join_node,
            wait_event_node,
        )

        # Calculate transfer size and time
        transfer_size_bytes: int = _calculate_transfer_size(reload_node)
        transfer_time_ms: float = _estimate_transfer_time_in_ms(transfer_size_bytes)

        patterns[reload_node] = ReloadNodeInfo(
            reload_group_nodes=[
                fork_node,
                wait_stream_node,
                reload_node,
                record_event_node,
                join_node,
            ],
            wait_event_node=wait_event_node,
            transfer_size_bytes=transfer_size_bytes,
            transfer_time_ms=transfer_time_ms,
        )

    return patterns


def reorder_for_prefetch(
    nodes_list: list[fx.Node],
    reload_patterns: dict[fx.Node, ReloadNodeInfo],
) -> None:
    """
    Reorder nodes to prefetch reload operations by directly manipulating the graph.

    This follows the algorithm as follows:
    - Go through nodes in reverse order
    - When encountering a reload pattern, add it to a queue with its transfer time
    - When encountering a compute node, use its runtime to satisfy overlap requirements
    - Place reload patterns when their overlap requirement is satisfied
    - When encountering placeholder nodes, flush queue as reloads cannot move before inputs
    """

    # Build a set of all nodes in reload groups for quick lookup
    reload_group_nodes_set: set[fx.Node] = set()
    for pattern in reload_patterns.values():
        reload_group_nodes_set.update(pattern.reload_group_nodes)

    # Queue to hold reload group nodes waiting to be placed (FIFO)
    reload_queue: list[ReloadQueueEntry] = []

    # Loop through nodes in reverse
    for node in reversed(nodes_list):
        if node.op == "output":
            continue
        elif node.op == "placeholder":
            # Flush queue - place all remaining reloads after the last placeholder
            while reload_queue:
                entry: ReloadQueueEntry = reload_queue.pop(0)
                for reload_group_node in reversed(entry.pattern.reload_group_nodes):
                    node.append(reload_group_node)
            break
        elif node in reload_patterns:
            pattern: ReloadNodeInfo = reload_patterns[node]
            reload_queue.append(
                ReloadQueueEntry(
                    pattern=pattern, remaining_time_ms=pattern.transfer_time_ms
                )
            )
        elif node in reload_group_nodes_set:
            continue
        else:
            if not reload_queue:
                continue
            compute_runtime_ms: float = (
                benchmark_node(node) if is_compute_node(node) else 0
            )
            reload_queue[0].remaining_time_ms -= compute_runtime_ms

            # Pop and place reload if its remaining time is satisfied (<= 0)
            if reload_queue[0].remaining_time_ms <= 0:
                entry: ReloadQueueEntry = reload_queue.pop(0)
                for reload_group_node in entry.pattern.reload_group_nodes:
                    node.prepend(reload_group_node)


def activation_offload_sink_wait(fwd_module: fx.GraphModule) -> None:
    """
    Sink wait_event operations for offload completion to the end of the graph.

    This function identifies wait_event nodes for offload completion and moves them
    to the end of the graph, allowing computation to overlap with offload operations.

    Args:
        fwd_module: Forward module graph
    """
    graph: fx.Graph = fwd_module.graph
    nodes_list: list[fx.Node] = list(graph.nodes)
    node_to_idx: dict[fx.Node, int] = {node: idx for idx, node in enumerate(nodes_list)}

    # Find all CPU offload device_put nodes
    offload_nodes: list[fx.Node] = [
        node
        for node in graph.find_nodes(
            op="call_function", target=torch.ops.prims.device_put.default
        )
        if CPU_OFFLOAD_PREFIX in node.name
    ]

    # Collect all wait_event nodes that need to be moved
    wait_nodes_to_sink: list[fx.Node] = []
    for offload_node in offload_nodes:
        offload_idx: int = node_to_idx[offload_node]
        wait_event_node: fx.Node = nodes_list[offload_idx + 3]

        # Validate it's actually a wait_event node
        if not (
            wait_event_node.op == "call_function"
            and wait_event_node.target == torch.ops.streams.wait_event.default
        ):
            raise ValueError(
                f"Expected wait_event node three positions after {offload_node.name}"
            )

        wait_nodes_to_sink.append(wait_event_node)

    # Find the output node, and move all wait_event nodes to just before the output node
    output_node: fx.Node = graph.find_nodes(op="output")[0]
    for wait_node in wait_nodes_to_sink:
        output_node.prepend(wait_node)


def activation_reload_prefetch(bwd_module: fx.GraphModule) -> None:
    """
    Prefetch backward reload operations by moving them earlier in the graph
    to overlap communication with computation.

    This function identifies backward reload patterns (fork → wait_stream → device_put →
    record_event → join) and moves them earlier in the execution order to overlap
    the data transfer with computation, while keeping the wait_event at its original
    position.

    Args:
        bwd_module: Backward module graph
    """
    graph: fx.Graph = bwd_module.graph
    nodes_list: list[fx.Node] = list(graph.nodes)
    node_to_idx: dict[fx.Node, int] = {node: idx for idx, node in enumerate(nodes_list)}

    # Step 1: Identify reload patterns
    reload_patterns: dict[fx.Node, ReloadNodeInfo] = identify_reload_patterns(
        graph, nodes_list, node_to_idx
    )

    # Step 2: Reorder nodes by directly manipulating the graph
    reorder_for_prefetch(nodes_list, reload_patterns)


def enable_activation_offloading(
    fwd_module: fx.GraphModule,
    bwd_module: fx.GraphModule,
    num_fwd_outputs: int,
    static_lifetime_input_nodes: OrderedSet[fx.Node],
) -> None:
    """
    Main entry point for activation offloading.

    Args:
        fwd_module: Forward module graph
        bwd_module: Backward module graph
        num_fwd_outputs: Number of forward outputs
    """

    # Step 1: Decide which nodes to offload and mark them
    should_perform_offloading: bool = choose_offload_sets(
        fwd_module,
        num_fwd_outputs,
        static_lifetime_input_nodes,
    )
    if not should_perform_offloading:
        return

    # Step 2: Add offload and reload nodes to the graphs
    if config.activation_offload_separate_stream:
        # Use async ao ops (2 nodes each: offload/reload + wait_tensor)
        offload_chosen_sets_async(fwd_module, bwd_module)
        if config.activation_offload_sink_wait:
            activation_offload_sink_wait_async(fwd_module)
        if config.activation_reload_prefetch:
            activation_reload_prefetch_async(bwd_module)
    else:
        # Use synchronous device_put (1 node each)
        offload_chosen_sets(fwd_module, bwd_module)

    fwd_module.graph.lint()
    bwd_module.graph.lint()
<|endoftext|>"""Custom ops for async activation offloading between GPU and CPU.

These ops encapsulate stream management internally, producing a clean 2-node
IR pattern (offload/reload + wait_tensor) similar to c10d functional collectives.

A single dedicated transfer stream handles all D2H/H2D copies.
Completion events are keyed by output tensor data_ptr() and stored in a
module-level registry, so ``ao.wait_tensor`` takes only the tensor itself
(plus optional ``keepalive`` and ``last_use_of_storage`` args).

Offload pattern:
    cpu_tensor = ao.offload(gpu_tensor)
    cpu_tensor = ao.wait_tensor(cpu_tensor, keepalive=gpu_tensor)
        keepalive frees the GPU tensor's storage after the D2H copy completes.

Reload pattern:
    gpu_tensor = ao.reload(cpu_tensor, device)
    gpu_tensor = ao.wait_tensor(gpu_tensor, keepalive=cpu_tensor)
        keepalive frees the CPU tensor's storage after the H2D copy completes.

The optional ``last_use_of_storage`` arg creates a dependency on a tensor whose
storage must outlive the async transfer but is not otherwise connected by a
data-flow edge in the graph.
"""

import torch
from torch._library.custom_ops import custom_op
from torch.fx import has_side_effect


# --- Global transfer stream (one per device, lazily created) ---
_transfer_streams: dict[torch.device, torch.Stream] = {}


def _get_or_create_transfer_stream(device: torch.device) -> torch.Stream:
    if device not in _transfer_streams:
        _transfer_streams[device] = torch.Stream(device=device)
    return _transfer_streams[device]


# --- Pinned memory pool (avoids per-offload cudaHostAlloc overhead) ---
# Keyed by (numel, dtype) to prevent cross-dtype reuse.
# Not keyed by device: pinned CPU memory is host-side and accessible from
# any GPU. Cross-device reuse is safe because wait_tensor synchronizes the
# transfer (via wait_event) before _pool_free returns the buffer — the
# buffer is never in-flight when reused.
_pinned_pool: dict[tuple[int, torch.dtype], list[torch.Tensor]] = {}
_pool_enabled: bool = False
_pool_managed_ptrs: set[int] = set()


def _maybe_pool_alloc(numel: int, dtype: torch.dtype) -> torch.Tensor:
    """Get a pinned CPU buffer from the pool (if enabled), or allocate fresh."""
    if _pool_enabled:
        key = (numel, dtype)
        bucket = _pinned_pool.get(key)
        if bucket:
            buf = bucket.pop()
            _pool_managed_ptrs.add(buf.data_ptr())
            return buf
        buf = torch.empty(numel, dtype=dtype, device="cpu", pin_memory=True)
        _pool_managed_ptrs.add(buf.data_ptr())
        return buf
    return torch.empty(numel, dtype=dtype, device="cpu", pin_memory=True)


def _pool_free(buf: torch.Tensor) -> None:
    """Return a pinned buffer to the pool, or free its storage if pool disabled."""
    ptr = buf.data_ptr()
    if ptr in _pool_managed_ptrs:
        key = (buf.nelement(), buf.dtype)
        _pinned_pool.setdefault(key, []).append(buf)
    else:
        storage = buf.untyped_storage()
        if storage.size() > 0:
            storage.resize_(0)


def _pool_clear() -> None:
    """Free all cached pinned buffers."""
    _pinned_pool.clear()
    _pool_managed_ptrs.clear()


import contextlib


@contextlib.contextmanager
def pinned_memory_pool(enabled: bool = True):
    """Context manager that controls pinned memory pooling for offload ops.

    Without this context manager, ``ao.offload`` allocates a fresh pinned
    buffer every call and ``ao.wait_tensor`` does not cache freed buffers.
    Inside the context with ``enabled=True``, buffers are reused across
    calls, avoiding the ~3 ms per-tensor ``cudaHostAlloc`` overhead::

        with pinned_memory_pool():
            for step in range(num_steps):
                train_step()  # ao.offload/reload reuse pooled buffers
        # pinned buffers freed here

    Nestable: inner contexts save and restore the outer state.
    Pass ``enabled=False`` to temporarily disable pooling.
    """
    global _pool_enabled
    saved = _pool_enabled
    _pool_enabled = enabled
    try:
        yield
    finally:
        if not saved:
            _pool_clear()
        _pool_enabled = saved


# --- Wait registry: maps data_ptr() -> (completion_event, device) ---
# Created by ao.offload / ao.reload, consumed (popped) by ao.wait_tensor.
# Not thread-safe — graph execution is single-threaded Python.
_wait_registry: dict[int, tuple[torch.Event, torch.device]] = {}


def _register_wait(tensor: torch.Tensor, device: torch.device) -> torch.Event:
    """Create an event for an async transfer and register it for wait_tensor."""
    event = torch.Event()
    _wait_registry[tensor.data_ptr()] = (event, device)
    return event


def _pop_wait(tensor: torch.Tensor) -> tuple[torch.Event, torch.device]:
    key = tensor.data_ptr()
    try:
        return _wait_registry.pop(key)
    except KeyError:
        raise RuntimeError(
            f"ao.wait_tensor: no pending transfer for tensor with data_ptr={key}. "
            "Every ao.wait_tensor must be paired with a preceding ao.offload or ao.reload."
        ) from None


def _clear_wait_registry() -> None:
    _wait_registry.clear()


@custom_op("ao::offload", mutates_args=())
def offload(tensor: torch.Tensor) -> torch.Tensor:
    """Async offload a GPU tensor to CPU on the dedicated transfer stream.

    Callers MUST pair this with an ``ao.wait_tensor`` that passes the source GPU
    tensor as ``keepalive`` to extend its lifetime past the async D2H copy.
    Do NOT use ``record_stream`` — it causes memory fragmentation and
    unbounded memory growth.

    Uses pinned-memory allocation + copy_ so the transfer is compatible
    with CUDA graph capture.
    """
    device = tensor.device
    transfer_stream = _get_or_create_transfer_stream(device)
    current_stream = torch.accelerator.current_stream(device)

    transfer_stream.wait_stream(current_stream)

    torch.accelerator.set_stream(transfer_stream)
    result = _maybe_pool_alloc(tensor.nelement(), tensor.dtype).view(tensor.shape)
    completion_event = _register_wait(result, device)
    result.copy_(tensor, non_blocking=True)
    transfer_stream.record_event(completion_event)
    torch.accelerator.set_stream(current_stream)

    return result


@offload.register_fake
def _(tensor: torch.Tensor) -> torch.Tensor:
    return torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")


@custom_op("ao::reload", mutates_args=())
def reload(
    tensor: torch.Tensor,
    device: torch.device,
    original_size: list[int] | None = None,
    original_stride: list[int] | None = None,
) -> torch.Tensor:
    """Async reload a CPU tensor to GPU on the dedicated transfer stream.

    The GPU tensor is allocated on the compute stream to avoid cross-stream
    allocator ownership issues. The H2D copy runs on the transfer stream.
    The completion event is keyed by the output tensor's data_ptr.

    ``original_size`` and ``original_stride`` restore the GPU tensor's
    original layout (which may be non-contiguous, e.g. from transpose).
    When None, the tensor's own size/stride are used.
    """
    size = original_size if original_size is not None else list(tensor.shape)
    stride = original_stride if original_stride is not None else list(tensor.stride())
    transfer_stream = _get_or_create_transfer_stream(device)
    current_stream = torch.accelerator.current_stream(device)

    result = torch.empty_strided(size, stride, dtype=tensor.dtype, device=device)
    completion_event = _register_wait(result, device)

    transfer_stream.wait_stream(current_stream)

    torch.accelerator.set_stream(transfer_stream)
    result.copy_(tensor, non_blocking=True)
    transfer_stream.record_event(completion_event)
    torch.accelerator.set_stream(current_stream)

    return result


@reload.register_fake
def _(
    tensor: torch.Tensor,
    device: torch.device,
    original_size: list[int] | None = None,
    original_stride: list[int] | None = None,
) -> torch.Tensor:
    size = original_size if original_size is not None else list(tensor.shape)
    stride = original_stride if original_stride is not None else list(tensor.stride())
    return torch.empty_strided(size, stride, dtype=tensor.dtype, device=device)


# ao::wait_tensor is defined via torch.library with an aliasing schema so the
# output can alias the input (custom_op forbids this).
#
# Uses CompositeExplicitAutograd (single impl for all devices) because the
# offload case has mixed-device args: ``tensor`` is CPU (the offload result)
# while ``keepalive`` is CUDA (the source GPU tensor). A single impl avoids
# relying on device-priority dispatch ordering.
#
# Synchronization details (completion event, device) are looked up from
# ``_wait_registry`` keyed on ``tensor.data_ptr()``.
#
# ``keepalive`` is the source tensor of the async transfer. It creates a
# graph dependency that extends the source tensor's lifetime until the
# compute stream has waited on the transfer completion event. After the
# wait, the op frees the source tensor's storage via ``resize_(0)`` since
# it is no longer needed:
#   - Offload (D2H): keepalive is the GPU tensor; freed after the D2H copy.
#   - Reload (H2D): keepalive is the CPU tensor; freed after the H2D copy.
#
# ``last_use_of_storage`` is an optional tensor whose storage is shared with
# other live tensors (views/aliases). Passing it here tells the scheduler
# that this wait_tensor call is the last consumer of that storage, creating
# a data dependency edge that prevents the storage from being freed or
# reused before the async transfer completes. This is needed when the
# storage is not otherwise kept alive by a direct data-flow edge in the
# graph -- without it, a compiler pass could schedule a storage-freeing op
# before the transfer finishes.
_lib = torch.library.Library("ao", "DEF")
_lib.define(
    "wait_tensor(Tensor(a) tensor, Tensor? keepalive=None, Tensor? last_use_of_storage=None) -> Tensor(a)"
)


@torch.library.impl("ao::wait_tensor", "CompositeExplicitAutograd")
def _ao_wait_tensor(
    tensor: torch.Tensor,
    keepalive: torch.Tensor | None = None,
    last_use_of_storage: torch.Tensor | None = None,
) -> torch.Tensor:
    completion_event, device = _pop_wait(tensor)
    current_stream = torch.accelerator.current_stream(device)

    current_stream.wait_event(completion_event)
    if keepalive is not None:
        if keepalive.is_pinned():
            # Return CPU pinned buffer to pool for reuse
            _pool_free(keepalive.view(-1))
        else:
            storage = keepalive.untyped_storage()
            if storage.size() > 0:
                storage.resize_(0)
    return tensor


@torch.library.register_fake("ao::wait_tensor")
def _ao_wait_tensor_fake(
    tensor: torch.Tensor,
    keepalive: torch.Tensor | None = None,
    last_use_of_storage: torch.Tensor | None = None,
) -> torch.Tensor:
    return tensor


has_side_effect(torch.ops.ao.wait_tensor.default)


def wait_tensor(
    tensor: torch.Tensor,
    keepalive: torch.Tensor | None = None,
    last_use_of_storage: torch.Tensor | None = None,
) -> torch.Tensor:
    """Callable wrapper so ``wait_tensor`` can be imported by name for op registration."""
    return torch.ops.ao.wait_tensor.default(tensor, keepalive, last_use_of_storage)
<|endoftext|># mypy: allow-untyped-defs
import inspect
import types
import typing
import warnings
from collections.abc import Callable, Sequence
from functools import wraps
from types import GenericAlias
from typing import NamedTuple, overload, TypeVar
from typing_extensions import ParamSpec

import torch
import torch._prims_common as utils
from torch._prims_common import (
    CustomOutParamAnnotation,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    Number,
    NumberType,
    ShapeType,
    TensorLike,
    TensorLikeType,
)
from torch.utils import _pytree as pytree
from torch.utils._inspect import _fast_bind
from torch.utils._pytree import tree_flatten, tree_unflatten


_T = TypeVar("_T")
_P = ParamSpec("_P")


@overload
# pyrefly: ignore [bad-return]
def _maybe_convert_to_dtype(a: TensorLikeType, dtype: torch.dtype) -> TensorLikeType:
    pass


@overload
# pyrefly: ignore [bad-return]
def _maybe_convert_to_dtype(a: NumberType, dtype: torch.dtype) -> NumberType:
    pass


@overload
# pyrefly: ignore [bad-return]
def _maybe_convert_to_dtype(a: Sequence, dtype: torch.dtype) -> Sequence:
    pass


@overload
def _maybe_convert_to_dtype(a: None, dtype: torch.dtype) -> None:
    pass


# TODO: implement ref.cast with an option to enforce safe casting
def _maybe_convert_to_dtype(a, dtype):
    if isinstance(a, TensorLike):
        if a.dtype != dtype:
            return a.to(dtype)
        return a
    if isinstance(a, Number):
        return utils.dtype_to_type_ctor(dtype)(a)  # type: ignore[arg-type]
    if isinstance(a, Sequence):
        return tuple(_maybe_convert_to_dtype(x, dtype) for x in a)
    # Passthrough None because some functions wrapped with type promotion
    # wrapper might have optional args
    if a is None:
        return None

    raise ValueError(
        f"Received unsupported type {type(a)}. Expected TensorLike, Number, or Sequence."
    )


def _maybe_convert_to_type(a: NumberType, typ: type) -> NumberType:
    if not isinstance(a, Number):
        msg = f"Found unknown type {type(a)} when trying to convert scalars!"
        raise ValueError(msg)
    if not utils.is_weakly_lesser_type(type(a), typ):
        msg = f"Scalar {a} of type {type(a)} cannot be safely cast to type {typ}!"
        raise ValueError(msg)

    return typ(a)


def _annotation_has_type(*, typ, annotation):
    args = typing.get_args(annotation)
    if args:
        return any(_annotation_has_type(typ=typ, annotation=a) for a in args)
    return typ is annotation


class elementwise_type_promotion_wrapper:
    """
    Adds elementwise type promotion to a Python reference implementation.

    Takes two kwargs, type_promoting_args and type_promotion_kind.

    type_promoting_args must be a string Sequence specifying the argument names of all
    arguments that participate in type promotion (and should be type promoted). If the
    arg specifies a Sequence-type then every element of the Sequence will participate in
    type promotion.

    type_promotion_kind must be one of the kinds specified by ELEMENTWISE_TYPE_PROMOTION_KIND.
    See its documentation for details.

    The return_dtype will be coerced to the wrapped function's dtype arg if it is available and
    not None.

    Other type promotion behavior, like validating the Python type of scalar arguments, must
    be handled separately.
    """

    def __init__(
        self,
        *,
        type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND,
        type_promoting_args: Sequence[str] | None = None,
    ):
        self.type_promoting_arg_names = type_promoting_args
        self.type_promotion_kind = type_promotion_kind

    def __call__(self, fn: Callable) -> Callable:
        sig = inspect.signature(fn)

        # TorchDynamo tracing of inspect causes fake tensor dynamo_wrapped tests to fail
        # PYTORCH_TEST_WITH_DYNAMO=1 python test/test_fake_tensor.py FakeTensorTest.test_basic
        @torch._disable_dynamo
        @wraps(fn)
        def _fn(*args, **kwargs):
            bound = _fast_bind(sig, *args, **kwargs)
            type_promoting_args = tuple(
                bound.arguments[x]
                for x in self.type_promoting_arg_names  # type: ignore[union-attr]
                if x in bound.arguments
            )

            flattened_type_promoting_args = pytree.arg_tree_leaves(*type_promoting_args)
            compute_dtype, result_dtype = utils.elementwise_dtypes(
                *flattened_type_promoting_args,
                type_promotion_kind=self.type_promotion_kind,
            )

            promoted_args = {
                x: _maybe_convert_to_dtype(bound.arguments[x], compute_dtype)
                for x in self.type_promoting_arg_names  # type: ignore[union-attr]
                if x in bound.arguments
            }
            bound.arguments.update(promoted_args)

            result = fn(**bound.arguments)

            # Override the return_dtype if a dtype arg is present and not None
            if "dtype" in bound.arguments:
                maybe_dtype = bound.arguments["dtype"]
                if maybe_dtype:  # dtype cannot be None
                    result_dtype = maybe_dtype

            if isinstance(result, TensorLike):
                return _maybe_convert_to_dtype(result, result_dtype)
            if isinstance(result, Sequence):
                return tuple(_maybe_convert_to_dtype(x, result_dtype) for x in result)
            raise AssertionError(f"Unhandled result type: {type(result)}")

        _fn.__signature__ = sig  # type: ignore[attr-defined]
        return _fn


# Returns True if resize is necessary
def _resize_output_check(out: TensorLikeType, shape: ShapeType):
    # If the shapes are correct there's nothing to do
    if utils.same_shape(out.shape, shape):
        return False
    if out.numel() != 0:
        msg = (
            f"An output with one or more elements was resized since it had shape {str(out.shape)} "
            "which does not match the required output shape {str(shape)}. "
            "This behavior is deprecated, and in a future PyTorch release outputs will not "
            "be resized unless they have zero elements. "
            "You can explicitly reuse an out tensor t by resizing it, inplace, to zero elements with t.resize_(0)."
        )
        warnings.warn(msg, stacklevel=2)
    return True


# TODO: handle tuples of tensors
def _maybe_resize_out(
    out: TensorLikeType,
    shape: ShapeType,
    memory_format: torch.memory_format | None = None,
):
    if _resize_output_check(out, shape):
        return out.resize_(shape, memory_format=memory_format)
    else:
        return out


def is_cpu_scalar(x: TensorLikeType) -> bool:
    return x.dim() == 0 and x.device.type == "cpu"


def check_copy_devices(*, copy_from: TensorLikeType, copy_to: TensorLikeType) -> None:
    if copy_from.device != copy_to.device:
        msg = (
            f"Attempting to copy from device {copy_from.device} "
            f"to device {copy_to.device}, but cross-device copies are not allowed!"
        )
        raise RuntimeError(msg)


def _safe_copy_out(
    *, copy_from: TensorLikeType, copy_to: TensorLikeType, exact_dtype: bool = False
):
    # Checks same device
    if not is_cpu_scalar(copy_from):
        check_copy_devices(copy_from=copy_from, copy_to=copy_to)

    # Checks safe cast
    if exact_dtype:
        torch._check(
            copy_from.dtype == copy_to.dtype,
            lambda: f"Expected out tensor to have dtype {copy_from.dtype} "
            f"but got {copy_to.dtype} instead",
        )
    else:
        torch._check(
            utils.can_safe_cast_to(cast_from=copy_from.dtype, cast_to=copy_to.dtype),
            lambda: f"Attempting to cast from {copy_from.dtype} to out tensor with dtype {copy_to.dtype}, "
            "but this can't be cast because it is not safe!",
        )

    return copy_to.copy_(copy_from)


def out_wrapper(
    *out_names: str,
    exact_dtype: bool = False,
    pass_is_out: bool = False,
    preserve_memory_format: bool = False,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    # The wrapped function needs to convert the output parameters to ensure
    # compatibility between the Python API (which always uses "out" as the
    # parameter name and may be a tuple) and the Aten API (which may have
    # multiple output parameters and use different parameter names such as
    # "grad_input", "indices" or "values".)

    default_out_names = ("out",)
    if len(out_names) == 0:
        # Use default in out name
        out_names = default_out_names

    is_tensor = len(out_names) == 1

    def maybe_compute_memory_format(t):
        return utils.suggest_memory_format(t) if preserve_memory_format else None

    def _out_wrapper(fn: Callable[_P, _T]) -> Callable[_P, _T]:
        """
        Adds the out parameter to a Python reference.
        """
        out_type = (
            TensorLikeType
            if is_tensor
            else GenericAlias(
                tuple, tuple(TensorLikeType for _ in range(len(out_names)))
            )
        )
        # For backward compatibility - should be able to remove once PEP585
        # conversion is complete.
        bc_out_type = (
            TensorLikeType
            if is_tensor
            else types.GenericAlias(
                tuple, tuple(TensorLikeType for _ in range(len(out_names)))
            )
        )
        return_type = (
            TensorLikeType
            if is_tensor
            else NamedTuple(
                f"return_types_{fn.__name__}",
                # pyrefly: ignore [bad-argument-count]
                [(o, TensorLikeType) for o in out_names],
            )
        )

        sig = inspect.signature(fn)
        factory_kwargs = ("device", "dtype")
        is_factory_fn = all(p in sig.parameters for p in factory_kwargs)

        @wraps(fn)
        def _fn(*args: _P.args, **kwargs: _P.kwargs):
            out = kwargs.pop("out", None)
            if is_factory_fn and out is not None:
                for k in factory_kwargs:
                    out_attr = getattr(out, k)
                    if k not in kwargs:
                        kwargs[k] = out_attr

            def maybe_check_copy_devices(out):
                if isinstance(out, TensorLike) and isinstance(args[0], TensorLike):
                    check_copy_devices(copy_from=args[0], copy_to=out)

            if isinstance(out, (tuple, list)):
                for o in out:
                    maybe_check_copy_devices(o)
            else:
                maybe_check_copy_devices(out)

            if pass_is_out:
                result = fn(*args, is_out=(out is not None), **kwargs)  # type: ignore[arg-type]
            else:
                result = fn(*args, **kwargs)
            if result is NotImplemented:
                return NotImplemented
            if not (
                (isinstance(result, TensorLike) and is_tensor)
                or (
                    isinstance(result, tuple)  # type: ignore[arg-type]
                    and len(result) == len(out_names)  # type: ignore[arg-type]
                )
                or (
                    fn.__name__ == "unbind" and isinstance(result, (list, tuple))  # type: ignore[arg-type]
                )
            ):
                raise AssertionError(
                    f"Unexpected result type: {type(result)}, is_tensor={is_tensor}, "
                    f"out_names={out_names}"
                )
            # unbind_copy is a special case: see https://github.com/pytorch/pytorch/issues/130829
            if out is not None:
                # Naively you might expect this assert to be true, but
                # it's not:
                #
                #   assert type(out) is type(result)
                #
                # The reason is that functions under this wrapper can
                # get registered to the Meta dispatch key, and that
                # means they can be executed in a context where tensor
                # subclasses are disabled (with no_dispatch), which is a
                # handy way for an is-a tensor subclass (e.g.,
                # FakeTensor) to have the normal meta backend create a
                # meta tensor, to be wrapped once it gets returned.
                # In this situation, you will get a FakeTensor as
                # the output tensor, but not the result--which will
                # be a normal meta tensor, but this is perfectly
                # harmless.
                if is_tensor and fn.__name__ != "unbind":
                    if not isinstance(out, TensorLike):
                        raise AssertionError(
                            f"out must be TensorLike, got {type(out)}"
                        )  # mypy
                    # These two operations are done in-place
                    _maybe_resize_out(
                        out,
                        result.shape,  # type: ignore[union-attr]
                        maybe_compute_memory_format(result),
                    )
                    _safe_copy_out(
                        copy_from=result,  # type: ignore[arg-type]
                        copy_to=out,
                        exact_dtype=exact_dtype,
                    )
                else:
                    if fn.__name__ != "unbind":
                        if not isinstance(out, tuple):
                            raise AssertionError(f"out must be tuple, got {type(out)}")  # type: ignore[arg-type]  # mypy
                    else:
                        if not isinstance(out, (list, tuple)):
                            raise AssertionError(
                                f"out must be list or tuple, got {type(out)}"
                            )  # type: ignore[arg-type]  # mypy
                    torch._check_type(
                        len(out) == len(result),  # type: ignore[arg-type]
                        lambda: f"expected tuple of {len(result)} elements but got {len(out)}",  # type: ignore[arg-type]
                    )
                    for r, o in zip(result, out):  # type: ignore[arg-type]
                        # These two operations are done in-place
                        _maybe_resize_out(o, r.shape, maybe_compute_memory_format(r))
                        _safe_copy_out(copy_from=r, copy_to=o, exact_dtype=exact_dtype)  # type: ignore[arg-type]
            else:
                out = result
            # mypy does not see through  the definition of out_type given that it's in a different scope
            return out if is_tensor else return_type(*out)  # type: ignore[operator]

        out_param = inspect.Parameter(
            "out",
            kind=inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=out_type,
        )
        # Mark that the function now returns a tuple
        if not (
            isinstance(sig.return_annotation, (str, TypeVar))
            or sig.return_annotation in (sig.empty, out_type, bc_out_type)
        ):
            raise AssertionError(
                f"Unexpected return annotation: {sig.return_annotation}, "
                f"expected str, TypeVar, empty, {out_type}, or {bc_out_type}"
            )
        params = *sig.parameters.values(), out_param

        # If there's a Parameter.VAR_KEYWORD parameter (like **kwds), it must appear
        # after the out= parameter, which is Parameter.KEYWORD_ONLY. Sorting by
        # Parameter.kind guarantees that all the parameters are in legal order.
        params = sorted(params, key=lambda p: p.kind)

        _fn.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            parameters=params,
            return_annotation=return_type,  # type: ignore[arg-type]
        )

        _fn.__annotations__ = dict(getattr(fn, "__annotations__", {}))
        _fn.__annotations__["out"] = out_type
        _fn.__annotations__["return"] = return_type

        # In the special case of having a single tensor out parameter with a
        # name other than out, add a special annotation to name the parameter
        if is_tensor and out_names != default_out_names:
            _fn.__annotations__[CustomOutParamAnnotation] = out_names[0]

        # Add an indicator attribute that can be used in special cases
        # where having a function wrapped by `out_wrapper` is not desirable e.g.
        # jit
        _fn._torch_decompositions_out_wrapper = (  # type: ignore[attr-defined]
            f"This function is wrapped by {out_wrapper.__module__}.out_wrapper"
        )

        return _fn

    return _out_wrapper


def _maybe_remove_out_wrapper(fn: Callable):
    return inspect.unwrap(
        fn,
        stop=lambda f: not hasattr(f, "_torch_decompositions_out_wrapper"),
    )


def backwards_not_supported(prim):
    def redispatch_prim(args, kwargs):
        with torch._C._AutoDispatchBelowAutograd():
            return prim(*args, **kwargs)

    class BackwardsNotSupported(torch.autograd.Function):
        @staticmethod
        # pyrefly: ignore [bad-override]
        def forward(ctx, args_spec, *flat_args):
            args, kwargs = tree_unflatten(flat_args, args_spec)  # type: ignore[arg-type]
            return redispatch_prim(args, kwargs)

        @staticmethod
        def backward(ctx, *args):
            raise RuntimeError("backwards not supported on prim")

    @wraps(prim)
    def _autograd_impl(*args, **kwargs):
        flat_args, args_spec = tree_flatten((args, kwargs))
        if torch.is_grad_enabled() and any(
            a.requires_grad for a in flat_args if isinstance(a, torch.Tensor)
        ):
            # TODO: There is a subtle bug here: prims like copy_to
            # return their input argument after mutating it; and custom
            # autograd function will incorrectly turn the result into
            # a view which will fail test_python_ref_executor tests.
            # At the moment, we sidestep this by observing that the
            # unit tests don't ever try to run the executor with
            # autograd, so we don't exercise the buggy case, but if
            # you ever want to feed autograd through this, be aware
            # of it!  We need a way of properly implementing autograd
            # for mutating operations in Python to do this.
            return BackwardsNotSupported.apply(args_spec, *flat_args)
        else:
            return redispatch_prim(args, kwargs)

    return _autograd_impl


# TODO: when tracing this will add torch tensors and not TensorMeta objects
# to the trace -- we should fix this by adding a tracing context and NumberMeta classes
# TODO: this wrapper is currently untested
def elementwise_unary_scalar_wrapper(
    fn: Callable[_P, _T],
) -> Callable[_P, _T | NumberType]:
    """
    Allows unary operators that accept tensors to work with Python numbers.
    """
    sig = inspect.signature(fn)

    @wraps(fn)
    def _fn(*args, **kwargs):
        if len(args) > 0 and isinstance(args[0], Number):
            dtype = utils.type_to_dtype(type(args[0]))
            args_ = list(args)
            args_[0] = torch.tensor(args[0], dtype=dtype)
            # pyrefly: ignore [invalid-param-spec]
            result = fn(*args_, **kwargs)
            if not isinstance(result, torch.Tensor):
                raise AssertionError(f"Expected torch.Tensor, got {type(result)}")
            return result.item()

        # pyrefly: ignore [invalid-param-spec]
        return fn(*args, **kwargs)

    _fn.__signature__ = sig  # type: ignore[attr-defined]
    # pyrefly: ignore [bad-return]
    return _fn
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import operator
import typing
import warnings
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from enum import Enum
from functools import reduce
from typing import (
    Any,
    cast,
    NamedTuple,
    Optional,
    overload,
    TYPE_CHECKING,
    TypeAlias,
    TypeGuard,
    TypeVar,
    Union,
)
from typing_extensions import deprecated

import torch
from torch import sym_float, sym_int, sym_max, sym_min


if TYPE_CHECKING:
    # Import the following modules during type checking to enable code intelligence features,
    # such as auto-completion in tools like pylance, even when these modules are not explicitly
    # imported in user code.

    import sympy

    from torch.types import BoolLikeType

    class _WorksWithInt(typing.Protocol):
        def __add__(self, other: Any) -> typing.Self: ...

        def __radd__(self, other: Any) -> typing.Self: ...

        def __mul__(self, other: Any) -> typing.Self: ...

        def __rmul__(self, other: Any) -> typing.Self: ...

    _IntLikeT = TypeVar("_IntLikeT", bound=_WorksWithInt)


ShapeType: TypeAlias = torch.Size | list[int] | tuple[int, ...]
StrideType: TypeAlias = list[int] | tuple[int, ...]
DimsType: TypeAlias = int | list[int] | tuple[int, ...]
DimsSequenceType: TypeAlias = list[int] | tuple[int, ...]
# TODO: Type[torch.SymInt], Type[torch.SymFloat]
NumberTypeType: TypeAlias = type[bool] | type[int] | type[float] | type[complex]
# TODO: This needs a lot more type annotations
# NumberType = Union[bool, int, float, complex, torch.SymInt, torch.SymFloat]
NumberType: TypeAlias = bool | int | float | complex
RealNumberType: TypeAlias = bool | int | float

Number = (bool, int, float, complex, torch.SymInt, torch.SymFloat, torch.SymBool)
# I don't call it Integral because numbers.Integral includes bool, but IntLike
# does not
Dim = int
IntLike = (int, torch.SymInt)
FloatLike = (float, torch.SymFloat)
BoolLike = (bool, torch.SymBool)
IntWithoutSymInt = int
FloatWithoutSymFloat = float
DeviceLikeType: TypeAlias = str | torch.device | int
Tensor = torch.Tensor


torch_function_passthrough = {
    torch.device,
    torch.sym_not,
    torch.sym_float,
    torch.sym_int,
    torch.sym_max,
    torch.sym_min,
    torch._sym_sqrt,  # type: ignore[attr-defined]
    torch.sym_ite,
    torch.Tensor.dim,
    torch.Tensor.ndim.__get__,  # type: ignore[attr-defined]
    torch.Tensor.numel,
    torch.Tensor.size,
    torch.Tensor.storage_offset,
    torch.Tensor.stride,
    torch.Tensor.dtype.__get__,  # type: ignore[attr-defined]
    torch.Tensor.is_sparse.__get__,  # type: ignore[attr-defined]
    torch.Tensor.shape.__get__,  # type: ignore[attr-defined]
    torch.Tensor.device.__get__,  # type: ignore[attr-defined]
    torch.Tensor.requires_grad.__get__,  # type: ignore[attr-defined]
    torch.Tensor.layout.__get__,  # type: ignore[attr-defined]
    torch.Tensor.is_contiguous,
    # For TorchRefsMode only
    torch.Tensor.__format__,
    torch.Tensor.__repr__,
    torch.Tensor.requires_grad.__get__,  # type: ignore[attr-defined]
    torch.Tensor.__getitem__,
}


TensorLikeType = torch.Tensor
TensorLike = torch.Tensor
TensorSequenceType: TypeAlias = list[TensorLikeType] | tuple[TensorLikeType, ...]
TensorOrNumberLikeType: TypeAlias = TensorLikeType | NumberType

CustomOutParamAnnotation = "__custom_out_param__"


def same_shape(a: ShapeType, b: ShapeType, *, allow_rhs_unbacked=False) -> bool:
    from torch.fx.experimental.symbolic_shapes import guard_or_true

    if len(a) != len(b):
        return False

    for x, y in zip(a, b):
        if allow_rhs_unbacked:
            if isinstance(y, torch.SymInt):
                continue

        # if we do not know, then they are not the same.
        if guard_or_true(x != y):
            return False

    return True


def _maybe_get_pytype(t):
    if t is torch.SymFloat:
        return float
    elif t is torch.SymInt:
        return int
    elif t is torch.SymBool:
        return bool
    else:
        return t


# TODO: look at using torch.testing.assert_close instead with an option
#   to just compare metadata
def compare_tensor_meta(
    a: TensorLikeType,
    b: TensorLikeType,
    check_sizes=True,
    check_strides=False,
    *,
    allow_rhs_unbacked=False,
    check_conj=True,
):
    """
    Checks that two tensor likes have the same shape,
    dtype and device.

    In the future this will validate additional metadata, like
    strides.
    """
    from torch._subclasses.fake_tensor import MetadataMismatchError

    if not isinstance(a, TensorLike):
        raise AssertionError(f"a must be TensorLike, got {type(a)}")
    if not isinstance(b, TensorLike):
        raise AssertionError(f"b must be TensorLike, got {type(b)}")

    if check_sizes and not same_shape(
        a.shape, b.shape, allow_rhs_unbacked=allow_rhs_unbacked
    ):
        msg = f"Shapes {a.shape} and {b.shape} are not equal!"
        raise MetadataMismatchError(msg)

    if a.dtype != b.dtype:
        msg = f"Dtypes {a.dtype} and {b.dtype} are not equal!"
        raise MetadataMismatchError(msg)

    if a.device != b.device:
        # Handles special cuda:0 vs cuda case
        # TODO: we should review why this happens and see about fixing it
        if (str(a.device) == "cuda:0" or str(a.device) == "cuda") and (
            str(b.device) == "cuda:0" or str(b.device) == "cuda"
        ):
            pass
        else:
            msg = f"Devices {a.device} and {b.device} are not equal!"
            raise MetadataMismatchError(msg)

    # Stride checking is currently disabled, see https://github.com/pytorch/pytorch/issues/78050
    if check_strides:
        same_strides, idx = check_significant_strides(
            a, b, only_cuda=False, allow_rhs_unbacked=allow_rhs_unbacked
        )
        if not same_strides:
            msg = f"Stride mismatch! Strides are {a.stride()} and {b.stride()} (mismatched at {idx})!"
            raise MetadataMismatchError(msg)

        if a.storage_offset() != b.storage_offset():
            msg = f"Storage offset mismatch! Storage offsets are {a.storage_offset()} and {b.storage_offset()}!"
            raise MetadataMismatchError(msg)

    if check_conj:
        if a.is_conj() != b.is_conj():
            raise MetadataMismatchError(
                f"Conj mismatch! is_conj is set to {a.is_conj()} and {b.is_conj()}"
            )

    if a.is_neg() != b.is_neg():
        raise MetadataMismatchError(
            f"Neg mismatch! is_neg is set to {a.is_neg()} and {b.is_neg()}"
        )


def _check_strides_helper(
    a: TensorLikeType,
    b: TensorLikeType,
    *,
    only_cuda=True,
    significant_only=True,
    allow_rhs_unbacked=False,
) -> tuple[bool, int | None]:
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    # NOTE: only on CUDA because CPU elementwise strides are incorrect in PyTorch
    # See https://github.com/pytorch/pytorch/issues/77553
    # Only compares strides that are "meaningful" -- strides for dimensions with length > 1
    # and for tensors with more than one element. Use guard_or_false on the
    # numel gate so unbacked shapes don't trigger a data-dependent guard.
    if (
        not only_cuda or a.device.type == "cuda" or b.device.type == "cuda"
    ) and guard_or_false(a.numel() > 0):
        for idx in range(a.ndim):
            check = not significant_only or a.shape[idx] > 1
            # TODO: Check the symbols are consistent with each other
            if isinstance(b.stride()[idx], torch.SymInt):
                continue
            if a.stride()[idx] != b.stride()[idx] and check:
                return False, idx

    return True, None


def check_significant_strides(
    a: TensorLikeType, b: TensorLikeType, *, only_cuda=True, allow_rhs_unbacked=False
) -> tuple[bool, int | None]:
    return _check_strides_helper(
        a,
        b,
        only_cuda=only_cuda,
        significant_only=True,
        allow_rhs_unbacked=allow_rhs_unbacked,
    )


def check_all_strides(
    a: TensorLikeType, b: TensorLikeType, *, only_cuda=True
) -> tuple[bool, int | None]:
    return _check_strides_helper(a, b, only_cuda=only_cuda, significant_only=False)


def check_contiguous_sizes_strides(sizes, strides, false_if_dde=False):
    """
    Performs an equality check between actual stride & expected stride (based on composed sizes),
    handling contiguous stride representations:
    e.g. torch.empty(u0, u1, u2).contiguous().stride() -> (Max(1, u1) * Max(1, u2), Max(1, u2), 1)
    and we'd like to treat this equal to (u1 * u2, u2, 1) for comparison purposes.
    """

    from torch.fx.experimental.symbolic_shapes import (
        guard_or_false,
        guard_or_true,
        is_nested_int,
    )

    def eval_eager(x):
        return bool(x)

    maybe_guard_or_false = guard_or_false if false_if_dde else eval_eager
    maybe_guard_or_true = guard_or_true if false_if_dde else eval_eager

    expected_stride = 1
    expected_stride_max = 1

    # pyrefly: ignore [bad-assignment]
    for x, y in reversed(tuple(zip(sizes, strides))):
        # Skips checking strides when a dimension has length 1.
        if maybe_guard_or_false(x == 1):
            continue

        if maybe_guard_or_true(y != expected_stride) and maybe_guard_or_true(
            y != expected_stride_max
        ):
            return False

        #  We symbolically check both paths to maximize the cases where this function
        #  returns true. This is because make_contiguous_strides_for adds the max
        #  symbolically, and in some other situations the max might not be there.
        #  And we want to ensure we return true in both cases.
        expected_stride_max *= x if is_nested_int(x) else sym_max(x, 1)  # type:ignore[assignment]

        expected_stride *= x

    return True


# This function is equivalent to compute_contiguous() from TensorImpl.cpp
def is_contiguous(a: TensorLikeType, false_if_dde=False) -> bool:
    """
    Tests whether a tensor is contiguous or not.

    Tensors are contiguous when they have no elements,
    one element, or when they have "nested" strides.
    """
    from torch.fx.experimental.symbolic_shapes import (
        guard_or_false,
        guard_size_oblivious,
    )

    def eval_eager(x):
        return bool(x)

    maybe_guard_or_false = guard_or_false if false_if_dde else eval_eager

    if maybe_guard_or_false(a.numel() < 2):
        return True

    return check_contiguous_sizes_strides(
        a.shape, a.stride(), false_if_dde=false_if_dde
    )


# This function is equivalent to compute_channels_last_contiguous_2d() in TensorImpl.cpp
def is_channels_last_contiguous_2d(a: Tensor, false_if_dde=False) -> bool:
    # NHWC or not channels last 2D contiguous
    if a.ndim != 4:
        return False

    from torch.fx.experimental.symbolic_shapes import guard_or_false, guard_or_true

    def eval_eager(x):
        return bool(x)

    maybe_guard_or_false = guard_or_false if false_if_dde else eval_eager
    maybe_guard_or_true = guard_or_true if false_if_dde else eval_eager

    expected_stride = 1
    for idx in (1, 3, 2, 0):
        length = a.shape[idx]
        if maybe_guard_or_false(length == 1):
            continue

        stride = a.stride()[idx]
        if maybe_guard_or_true(stride != expected_stride):
            return False

        expected_stride *= length

    return True


def is_channels_last_contiguous_3d(a: Tensor, false_if_dde=False) -> bool:
    # NDHWC or not channels last 3D contiguous
    if a.ndim != 5:
        return False

    from torch.fx.experimental.symbolic_shapes import guard_or_false, guard_or_true

    def eval_eager(x):
        return bool(x)

    maybe_guard_or_false = guard_or_false if false_if_dde else eval_eager
    maybe_guard_or_true = guard_or_true if false_if_dde else eval_eager

    expected_stride = 1
    for idx in (1, 4, 3, 2, 0):
        length = a.shape[idx]
        if maybe_guard_or_false(length == 1):
            continue

        stride = a.stride()[idx]
        if maybe_guard_or_true(stride != expected_stride):
            return False

        expected_stride *= length

    return True


_memory_formats = {
    torch.contiguous_format,
    torch.preserve_format,
    torch.channels_last,
    torch.channels_last_3d,
}


def validate_memory_format(memory_format: torch.memory_format):
    torch._check(
        memory_format in _memory_formats,
        lambda: f"Received unknown memory format {memory_format}!",
    )


def is_contiguous_for_memory_format(  # type: ignore[return]
    a: Tensor,
    *,
    memory_format: torch.memory_format,
    false_if_dde=False,
    # pyrefly: ignore [bad-return]
) -> bool:
    validate_memory_format(memory_format)

    if memory_format == torch.contiguous_format:
        return is_contiguous(a, false_if_dde)
    if memory_format == torch.channels_last:
        return is_channels_last_contiguous_2d(a, false_if_dde)
    if memory_format == torch.channels_last_3d:
        return is_channels_last_contiguous_3d(a, false_if_dde)

    torch._check(
        False,
        lambda: f"is_contiguous received unsupported memory format {memory_format}",
    )


def is_contiguous_or_false(a: TensorLikeType) -> bool:
    return is_contiguous(a, false_if_dde=True)


# similar to is_channels_last_contiguous_2d but return false on data dependency.
def is_channels_last_contiguous_or_false_2d(a: Tensor) -> bool:
    return is_channels_last_contiguous_2d(a, false_if_dde=True)


# similar to is_channels_last_contiguous_3d but return false on data dependency.
def is_channels_last_contiguous_or_false_3d(a: Tensor) -> bool:
    return is_channels_last_contiguous_3d(a, false_if_dde=True)


# similar to is_contiguous_for_memory_format but return false on data dependency.
def is_contiguous_for_memory_format_or_false(  # type: ignore[return]
    a: Tensor, *, memory_format: torch.memory_format
) -> bool:
    return is_contiguous_for_memory_format(
        a, memory_format=memory_format, false_if_dde=True
    )


# NOTE: that tensors with no elements and channels last is ???
def is_channels_last_contiguous(a: Tensor) -> bool:
    """
    True when a tensor is channels-last contiguous.

    This requires that:

      - the tensor is conceptually either 4 (NHWC) or 5 (NDHWC) dimensions
      - if we name the tensor's dimensions NCHW or NCDHW, then the strides are such that the
        stride of the 'C' dimension (Cs) is 1 and the strides corresponding to
        each dimension (Xs) can be ordered Cs <= Ws <= Hs <= (Ds) <= Ns and are
        "nested" -- so Ws = Cs * Cl, where Cl is the length of the 'C' dimension,
        for example.
    """
    return is_channels_last_contiguous_2d(a) or is_channels_last_contiguous_3d(a)


# similar to is_channels_last_contiguous but return false on data dependency.
def is_channels_last_contiguous_or_false(a: Tensor) -> bool:
    return is_channels_last_contiguous_or_false_2d(
        a
    ) or is_channels_last_contiguous_or_false_3d(a)


# Defined at module scope so the class is built once, not rebuilt on every call.
class K(NamedTuple):
    size: int
    stride: int

    def __lt__(self, other):
        from torch.fx.experimental.symbolic_shapes import guard_or_false, guard_or_true

        # for backed symbols, this is practically a < operation
        # for unbacked, we return True if < is statically known,
        # then try to answer this symbolically, with stride ordering semantics
        # (e.g. u0 < u0 is False, u0 < u1 is False with no axioms, u0 < 2 * u0 is True)
        return (
            guard_or_false(
                self.stride < other.stride
            )  # checks statically known inequality
            or (
                (
                    guard_or_false(self.stride == 0)
                    or guard_or_false(other.stride % self.stride == 0)
                )
                and guard_or_true(self.stride != other.stride)
            )  # checks symbolic inequality (e.g. u0 < 2048 * u0)
        )


def _is_non_overlapping_and_dense_or_false(sizes, strides) -> bool:
    """
    Helper function for is_non_overlapping_and_dense.
    For unbacked sizes & strides, returns True only if symbolically non-overlapping & dense,
    and False otherwise.

    e.g. sizes: [u0, u1], strides: [u2, u3]
    this may be non-overlapping & dense at runtime, for values {u0: 4, u1: 4, u2: 4, u3: 1},
    but isn't true for all values.
    """
    from torch.fx.experimental.symbolic_shapes import guard_or_false
    from torch.utils._sympy.functions import Max

    # Short-circuits for 0/1-element tensors
    if guard_or_false(prod(sizes) < 2):  # type: ignore[operator]
        return True

    # Short-circuits for tensors of rank one, which are
    # non-overlapping and "dense" if their stride is one
    if len(sizes) == 1:
        return guard_or_false(strides[0] == 1)

    # Checks that there exists a permutation of the strides s.t. the tensor would be contiguous
    # Sorts (length, stride) pairs by stride
    #
    # This sort is done in a size-oblivious way, which helps if we do a
    # comparison like 2048*u0 > u0; we just want this to return True
    # (and not worry about what if u0 is zero).
    lengths_and_strides = sorted(map(K, sizes, strides))

    # verify actual strides match the expected (composed sizes)
    sizes = [x.size for x in lengths_and_strides][::-1]
    strides = [x.stride for x in lengths_and_strides][::-1]
    return check_contiguous_sizes_strides(sizes, strides, false_if_dde=True)


def is_non_overlapping_and_dense_or_false(a: Tensor) -> bool:
    """
    True when a tensor is non-overlapping and dense.

    A tensor is non-overlapping and dense when there exists a permutation of
    its dimensions that is contiguous.
    """
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    if a.is_sparse:
        return False

    return _is_non_overlapping_and_dense_or_false(a.shape, a.stride())


# NOTE: Based on the implementation in TensorIterator.cpp, but note that
# the note [Computing output strides] is incorrect, because it
# says that strides will be preserved even if they are not
# "non overlapping and dense", but this is incorrect. The
# output of elementwise operations are always given
# non overlapping and dense strides.
# This is also INCORRECT because it does not model TensorIterator's
# short-circuit, which can cause different strides.
def compute_elementwise_output_logical_to_physical_perm(
    *tensors, _skip_checks=False, ambiguity_check=False
) -> tuple[list[int], bool]:
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    if not _skip_checks and len(tensors) == 0:
        msg = "Can't compute elementwise output strides for zero tensors!"
        raise ValueError(msg)

    if not _skip_checks:
        check_same_shape(*tensors, allow_cpu_scalar_tensors=True)

    # Filters the tensors to actual tensors
    if not _skip_checks:
        tensors = tuple(
            a
            for a in tensors
            if isinstance(a, TensorLike) and not is_cpu_scalar_tensor(a)
        )

    # Short-circuits for CPU scalar case
    if len(tensors) == 0:
        return [], False

    # Short-circuits for shapes with zero or one dimensions
    # TODO: are these necessary?
    ndim = tensors[0].ndim
    if ndim == 0:
        return [], False
    if ndim == 1:
        return [0], False

    # Short-circuits if contiguous or channels last, following the fake fast path.
    # This reduces the number of guards we end up making
    is_contiguous = True
    is_channels_last = True
    for t in tensors:
        is_contiguous = is_contiguous and is_contiguous_for_memory_format_or_false(
            t, memory_format=torch.contiguous_format
        )
        is_channels_last = (
            is_channels_last
            and is_contiguous_for_memory_format_or_false(
                t, memory_format=torch.channels_last
            )
        )

    if is_contiguous and not is_channels_last:
        return list(range(ndim)), False

    if is_channels_last and not is_contiguous:
        return [0, *list(range(2, ndim)), 1], False

    shape = tensors[0].shape

    def should_swap(idx_a, idx_b):
        def ge(a, b):
            """
            Returns true if a is symbolically greater than or equal to b, assuming a >= 0, b >= 0.
            """
            if guard_or_false(b == 0):
                return True
            elif guard_or_false(a == 0):
                return False
            return guard_or_false(a >= b) or guard_or_false(a % b == 0)

        for tensor in tensors:
            stride_a = tensor.stride()[idx_a]
            stride_b = tensor.stride()[idx_b]

            if guard_or_false(stride_a == 0) or guard_or_false(stride_b == 0):
                continue

            if guard_or_false(stride_a == stride_b):
                if ge(shape[idx_b], shape[idx_a]):
                    continue
                return 1

            if ge(stride_b, stride_a):
                return -1

            if ge(stride_a, stride_b):
                return 1

        # Note: this case is hit if all strides are zero,
        # or all strides are equal and all dimensions have the same length
        return 0

    # The "sort" order for the permutation is back-to-front, but
    # the natural order for permutations is front-to-back.  Do the
    # sorting back-to-front and then reverse it on output.
    #
    # also, note this returns the logical to physical shape permutation
    perm = list(reversed(range(ndim)))

    # insertion sort with support for ambiguous comparisons
    for i in range(1, ndim):
        dim1 = i
        for dim0 in reversed(range(i)):
            comparison = should_swap(perm[dim0], perm[dim1])
            if comparison > 0:
                perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
                dim1 = dim0
            elif comparison < 0:
                break

    # verify we've imposed ordering if ambiguity_check=True
    raise_ambiguous = False
    if ambiguity_check:
        for i, j in zip(range(ndim - 1), range(1, ndim)):
            order = should_swap(perm[i], perm[j])
            if order != -1:
                raise_ambiguous = True
                break

    return list(reversed(perm)), raise_ambiguous


def compute_elementwise_output_strides(*tensors) -> tuple[int, ...]:
    """
    Computes the output strides for elementwise operations.
    """
    if len(tensors) == 0:
        msg = "Can't compute elementwise output strides for zero tensors!"
        raise ValueError(msg)

    check_same_shape(*tensors, allow_cpu_scalar_tensors=True)

    # Filters the tensors to actual tensors
    tensors = tuple(
        a for a in tensors if isinstance(a, TensorLike) and not is_cpu_scalar_tensor(a)
    )

    # Short-circuits for CPU scalar case
    if len(tensors) == 0:
        return ()

    ndim = tensors[0].ndim
    shape = tensors[0].shape

    if ndim == 0:
        return ()
    if ndim == 1:
        return (1,)

    if len(tensors) == 1:
        if torch._prims_common.is_non_overlapping_and_dense_or_false(tensors[0]):
            return tensors[0].stride()
        else:
            empty_like_tensor = torch.empty_like(tensors[0])
            return empty_like_tensor.stride()

    logical_to_physical_perm, _ = compute_elementwise_output_logical_to_physical_perm(
        *tensors, _skip_checks=True
    )
    permuted_shape = apply_perm(shape, logical_to_physical_perm)  # to physical

    new_strides = make_contiguous_strides_for(permuted_shape)
    permuted_strides = apply_perm(
        new_strides, invert_perm(logical_to_physical_perm)
    )  # to logical

    return tuple(permuted_strides)


# Identity permutation is [0, 1, 2]
def apply_perm(inp, perm):
    ndim = len(inp)
    permuted_inp = [-1] * ndim
    for idx, x in enumerate(perm):
        permuted_inp[idx] = inp[x]
    return permuted_inp


def invert_perm(perm):
    ndim = len(perm)
    new_perm = [-1] * ndim
    for idx, x in enumerate(perm):
        new_perm[x] = idx
    return new_perm


#
# Common helper functions
#


def validate_dim_length(length: int):
    """
    Validates that an object represents a valid
    dimension length.
    """

    if isinstance(length, (int, torch.SymInt)):
        torch._check(length >= 0)
    else:
        # sometimes called with sympy expression by inductor
        if length < 0:
            raise AssertionError(f"length must be non-negative, got {length}")


def validate_shape(shape: ShapeType):
    """
    Validates that a sequence represents a valid shape.
    """

    if not isinstance(shape, Sequence):
        raise AssertionError(f"shape must be a Sequence, got {type(shape)}")
    for l in shape:
        validate_dim_length(l)


def validate_strides(strides: StrideType):
    """
    Verifies the object specifies valid strides.
    """

    if not isinstance(strides, Sequence):
        raise AssertionError(f"strides must be a Sequence, got {type(strides)}")
    for stride in strides:
        if stride < 0:
            raise AssertionError(f"stride must be non-negative, got {stride}")


def validate_idx(rank: int, idx: int):
    """
    Validates that idx is a valid index for the given shape.
    Assumes the index is already canonicalized.
    """

    if not isinstance(idx, Dim):
        raise AssertionError(f"idx must be Dim, got {type(idx)}")
    if not isinstance(rank, Dim):
        raise AssertionError(f"rank must be Dim, got {type(rank)}")

    if not (idx >= 0 and idx < rank or idx == 0):
        raise AssertionError(f"idx {idx} is out of bounds for rank {rank}")


def validate_dimension_indices(rank: int, indices: DimsSequenceType):
    for idx in indices:
        validate_idx(rank, idx)


def validate_exclusive_idx(rank: int, ex_idx: int):
    """
    Validates that ex_idx is a valid exclusive index
    for the given shape.
    """

    if not isinstance(ex_idx, Dim):
        raise AssertionError(f"ex_idx must be Dim, got {type(ex_idx)}")
    if not isinstance(rank, Dim):
        raise AssertionError(f"rank must be Dim, got {type(rank)}")
    if not (ex_idx > 0 and ex_idx <= rank):
        raise AssertionError(
            f"ex_idx {ex_idx} is out of bounds for rank {rank} (must be in (0, rank])"
        )


# "Wraps" a dim (up to one time) for the given rank, allowing dims to be
# specified using negative indices. If `wrap_scalar` is true then scalar
# tensors of rank 0 will allow dimensions in the range [-1, 0]. Otherwise,
# idx should be in the range [-rank, rank-1].
def canonicalize_dim(rank: int, idx: int, wrap_scalar: bool = True) -> int:
    if rank < 0:
        msg = f"Rank cannot be negative but got {rank}"
        raise IndexError(msg)

    if rank == 0:
        if not wrap_scalar:
            msg = f"Dimension specified as {idx} but tensor has no dimensions"
            raise IndexError(msg)
        rank = 1

    if idx >= 0 and idx < rank:
        return idx

    if idx < 0:
        _idx = idx + rank
    else:
        _idx = idx

    if _idx < 0 or _idx >= rank:
        # Same error message as in aten/src/ATen/WrapDimUtils.h:49
        msg = f"Dimension out of range (expected to be in range of [{-rank}, {rank - 1}], but got {idx})"
        raise IndexError(msg)

    return _idx


# Takes a dimension or sequence of dimensions and "wraps" them,
# mapping negative offsets to positive ones
@overload
def canonicalize_dims(
    rank: int,
    indices: Sequence[int],
    wrap_scalar: bool = True,
    # pyrefly: ignore [bad-return]
) -> tuple[int, ...]:
    pass


@overload
# pyrefly: ignore [bad-return]
def canonicalize_dims(rank: int, indices: int, wrap_scalar: bool = True) -> int:
    pass


def canonicalize_dims(rank, indices, wrap_scalar=True):
    if isinstance(indices, Dim):
        return canonicalize_dim(rank, indices, wrap_scalar)

    return tuple(canonicalize_dim(rank, x, wrap_scalar) for x in indices)


def is_valid_permutation(rank: int, perm: DimsSequenceType) -> bool:
    """
    Validates that perm is a permutation of length rank.
    """

    return isinstance(perm, Sequence) and sorted(perm) == list(range(rank))


def is_same_shape(a: Sequence, b: Sequence) -> BoolLikeType:
    """
    Compares two shapes a and b, returning True if they are the same
    (their ranks and corresponding lengths match) and False otherwise.

    Uses sym_eq for shape comparison so the result is safe to pass to
    torch._check on tensors with unbacked SymInt dimensions; for backed
    or concrete shapes the behaviour is unchanged.
    """
    from torch.fx.experimental.symbolic_shapes import sym_eq

    return sym_eq(tuple(a), tuple(b))


def is_cpu_scalar_tensor(a: object) -> TypeGuard[TensorLike]:
    return isinstance(a, TensorLike) and a.ndim == 0 and a.device.type == "cpu"


def check_same_device(*args, allow_cpu_scalar_tensors):
    """
    Checks that all Tensors in args have the same device.

    Raises a RuntimeError when:
      - args contains an object whose type is not Tensor or Number
      - two Tensor objects in args have different devices, unless one is a CPU scalar tensor and allow_cpu_scalar_tensors is True
    """
    # Short-circuits if all (one or fewer) arguments are trivially on the same device
    if len(args) <= 1:
        return

    # Note: cannot initialize device to the first arg's device (it may not have one)
    device = None
    # pyrefly: ignore [bad-assignment]
    for arg in args:
        if isinstance(arg, Number):
            continue
        elif isinstance(arg, TensorLike):
            if allow_cpu_scalar_tensors and is_cpu_scalar_tensor(arg):
                continue

            if device is None:
                device = arg.device

            if device != arg.device:
                msg = (
                    "Tensor on device "
                    + str(arg.device)
                    + " is not on the expected device "
                    + str(device)
                    + "!"
                )
                raise RuntimeError(msg)
        else:
            msg = (
                "Unexpected type when checking for same device, " + str(type(arg)) + "!"
            )
            raise RuntimeError(msg)


def canonicalize_device(device: DeviceLikeType) -> torch.device:
    if isinstance(device, torch.device):
        return device

    if not isinstance(device, str):
        raise AssertionError(f"device must be torch.device or str, got {type(device)}")
    return torch.device(device)


# Asserts if any of the following are true:
#   - a non-scalar or non-Tensor is given
#   - the shape of any tensors is distinct
def check_same_shape(*args, allow_cpu_scalar_tensors: bool):
    """
    Checks that all Tensors in args have the same shape.

    Raises a RuntimeError when:
      - args contains an object whose type is not Tensor or Number
      - two Tensor objects in args have different devices
    """
    shape = None

    # pyrefly: ignore [bad-assignment]
    for arg in args:
        if isinstance(arg, Number):
            continue
        elif isinstance(arg, TensorLike):
            if allow_cpu_scalar_tensors and is_cpu_scalar_tensor(arg):
                continue

            if shape is None:
                shape = arg.shape

            if not is_same_shape(shape, arg.shape):
                msg = f"Shape {arg.shape} is not the expected shape {shape}!"
                raise RuntimeError(msg)
        else:
            msg = (
                "Unexpected type when checking for same shape, " + str(type(arg)) + "!"
            )
            raise RuntimeError(msg)


# Acquires a common shape, if it exists, from one or more tensor arguments,
# filtering number arguments
def extract_shape(*args, allow_cpu_scalar_tensors: bool) -> ShapeType | None:
    shape = None
    scalar_shape = None

    # pyrefly: ignore [bad-assignment]
    for arg in args:
        if isinstance(arg, Number):
            continue
        elif isinstance(arg, TensorLike):
            if allow_cpu_scalar_tensors and is_cpu_scalar_tensor(arg):
                scalar_shape = arg.shape
                continue

            if shape is None:
                shape = arg.shape

            if not is_same_shape(shape, arg.shape):
                return None
        else:
            return None

    return shape if shape is not None else scalar_shape


# Extracts dimensions that might be passed either as a list/tuple or as varargs.
# A typical case is Tensor.permute .
def extract_dims_from_varargs(
    dims: DimsSequenceType | tuple[DimsSequenceType, ...],
) -> DimsSequenceType:
    if dims and isinstance(dims[0], Sequence):
        if len(dims) != 1:
            raise AssertionError(
                f"Expected exactly 1 element in dims when first element is a Sequence, got {len(dims)}"
            )
        dims = cast(tuple[DimsSequenceType], dims)
        return dims[0]
    else:
        return cast(DimsSequenceType, dims)


def extract_shape_from_varargs(
    shape: ShapeType | tuple[ShapeType],
    validate=True,
) -> tuple[int, ...]:
    """
    Returns a shape from varargs.

    In PyTorch, operations that accept shapes often accept them as varargs, like
    foo(*shape). However a user can pass the shape as a sequence of integers,
    like this:

      foo(1, 2, 3)

    or as a sequence of integers

      foo((1, 2, 3))

    In the first case shape will be a tuple of integers, and in the second case it's a tuple
    containing a tuple of integers. This validates those inputs and canonicalizes them
    to a tuple of integers.
    """

    # Handles tuple unwrapping
    if len(shape) == 1 and isinstance(shape[0], Sequence):
        # pyrefly: ignore [bad-assignment]
        shape = shape[0]

    if validate:
        validate_shape(shape)  # type: ignore[arg-type]
    return shape  # type: ignore[return-value]


def infer_size_shapes(a: ShapeType, b: ShapeType) -> tuple[int, ...]:
    ndim = max(len(a), len(b))
    expandedSizes = [0] * ndim

    for i in range(ndim - 1, -1, -1):
        offset = ndim - 1 - i
        dimA = len(a) - 1 - offset
        dimB = len(b) - 1 - offset
        sizeA = a[dimA] if dimA >= 0 else 1
        sizeB = b[dimB] if dimB >= 0 else 1

        torch._check(
            (sizeA == sizeB) or (sizeA == 1) or (sizeB == 1),
            lambda: (
                f"The size of tensor a ({sizeA}) must match the size of "
                f"tensor b ({sizeB}) at non-jagged dimension {i}"
            ),
        )

        # 1s map to the other size (even 0)
        expandedSizes[i] = sizeB if sizeA == 1 else sizeA

    return tuple(expandedSizes)


def infer_size(shape: ShapeType, numel: int) -> tuple[int, ...]:
    """
    Infers the size of a dim with size -1, if it exists.
    Also checks that new shape is compatible with the number of elements.
    """
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    dim = None
    newsize = 1
    for i, d in enumerate(shape):
        if guard_or_false(d == -1):
            torch._check(dim is None, lambda: "only one dimension can be inferred")
            dim = i
        else:
            torch._check(
                d >= 0,
                lambda: (
                    f"invalid shape dimension {d}. If this was symbolic, it was assumed to not be -1."
                    "If this was meant to be inferred, please explicitly pass in -1."
                ),
            )
            newsize *= d
    if dim is None:
        torch._check(
            numel == newsize,
            lambda: f"shape '{list(shape)}' is invalid for input of size {numel}",
        )
    else:
        torch._check(
            newsize != 0,
            lambda: (
                f"cannot reshape tensor of 0 elements into shape {list(shape)} because the "
                f"unspecified dimension size -1 can be any value and is ambiguous"
                if guard_or_false(numel == 0)
                else f"shape '{list(shape)}' is invalid for input of size {numel}"
            ),
        )
        torch._check(
            numel % newsize == 0,
            lambda: f"shape '{list(shape)}' is invalid for input of size {numel}",
        )
        # Convert to list to produce a compatible error message with core
        # PyTorch, which prints sequences in square brackets.
        shape = list(shape)
        shape[dim] = numel // newsize
        torch._check(shape[dim] >= 0)
    return tuple(shape)


_integer_dtypes = (
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)
_low_precision_dtypes = (
    torch.float16,
    torch.bfloat16,
    torch.complex32,
    torch.bcomplex32,
)
_complex_dtypes = (torch.complex32, torch.bcomplex32, torch.complex64, torch.complex128)


def is_boolean_dtype(dtype: torch.dtype) -> bool:
    if not isinstance(dtype, torch.dtype):
        raise AssertionError(f"Expected torch.dtype, got {type(dtype)}")
    return dtype is torch.bool


def is_integer_dtype(dtype: torch.dtype) -> bool:
    if not isinstance(dtype, torch.dtype):
        raise AssertionError(f"Expected torch.dtype, got {type(dtype)}")
    return dtype in _integer_dtypes


def is_low_precision_dtype(dtype: torch.dtype) -> bool:
    if not isinstance(dtype, torch.dtype):
        raise AssertionError(f"Expected torch.dtype, got {type(dtype)}")
    return dtype in _low_precision_dtypes


def is_float_dtype(dtype: torch.dtype) -> bool:
    if not isinstance(dtype, torch.dtype):
        raise AssertionError(f"Expected torch.dtype, got {type(dtype)}")
    return dtype.is_floating_point


def is_complex_dtype(dtype: torch.dtype) -> bool:
    if not isinstance(dtype, torch.dtype):
        raise AssertionError(f"Expected torch.dtype, got {type(dtype)}")
    return dtype in _complex_dtypes


def is_grad_dtype(dtype: torch.dtype) -> bool:
    """
    Checks if the dtype can require a gradient.
    """
    return dtype.is_floating_point or is_complex_dtype(dtype)


_complex_to_real_dtype_map = {
    torch.complex128: torch.float64,
    torch.complex64: torch.float32,
    torch.complex32: torch.float16,
    torch.bcomplex32: torch.bfloat16,
}

_real_to_complex_dtype_map = {
    torch.float16: torch.complex32,
    torch.bfloat16: torch.bcomplex32,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}


def corresponding_real_dtype(dtype: torch.dtype) -> torch.dtype:
    return _complex_to_real_dtype_map[dtype]


def corresponding_complex_dtype(dtype: torch.dtype) -> torch.dtype:
    return _real_to_complex_dtype_map[dtype]


def dtype_to_type(dtype: torch.dtype) -> type:
    """
    Computes the corresponding Python type (AKA "type kind") for the
    given dtype.
    """
    if not isinstance(dtype, torch.dtype):
        raise AssertionError(f"Expected torch.dtype, got {type(dtype)}")

    if dtype is torch.bool:
        return bool
    if dtype in _integer_dtypes:
        return int
    if dtype.is_floating_point:
        return float
    if dtype in _complex_dtypes:
        return complex

    raise ValueError("Invalid dtype!")


def dtype_to_type_ctor(dtype: torch.dtype) -> Callable[[NumberType], NumberType]:
    """
    Computes the corresponding Python type constructor for the
    given dtype.
    """
    if not isinstance(dtype, torch.dtype):
        raise AssertionError(f"Expected torch.dtype, got {type(dtype)}")

    if dtype is torch.bool:
        return lambda x: bool(x)
    if dtype in _integer_dtypes:
        return sym_int
    if dtype.is_floating_point:
        # pyrefly: ignore [bad-return]
        return sym_float
    if dtype in _complex_dtypes:
        # TODO: type error here is real, replace with sym_complex
        return lambda x: complex(x)  # type: ignore[arg-type]

    raise ValueError("Invalid dtype!")


def type_to_dtype(typ: type) -> torch.dtype:
    """
    Computes the corresponding dtype for a Number type.
    """

    if not isinstance(typ, type):
        raise AssertionError(f"Expected type, got {type(typ)}")

    if typ in (bool, torch.SymBool):
        return torch.bool
    if typ in (int, torch.SymInt):
        return torch.long
    if typ in (float, torch.SymFloat):
        return torch.get_default_dtype()
    # TODO: sym_complex_float?
    if typ is complex:
        return corresponding_complex_dtype(torch.get_default_dtype())

    raise ValueError(f"Invalid type {typ}!")


def get_dtype(x: torch.Tensor | NumberType):
    if isinstance(x, torch.Tensor):
        return x.dtype
    else:
        return type_to_dtype(type(x))


_ordered_types = (bool, int, float, complex)


def check_fp_or_complex(
    dtype: torch.dtype, fn_name: str, allow_low_precision_dtypes: bool = True
):
    """
    Checks whether the input is floating point or complex.
    If allow_low_precision_dtypes is True, it allows having float16, bfloat16, and [b]complex32
    """
    torch._check(
        is_float_dtype(dtype) or is_complex_dtype(dtype),
        lambda: f"{fn_name}: Expected a floating point or complex tensor as input. Got {dtype}",
    )
    torch._check(
        allow_low_precision_dtypes or not is_low_precision_dtype(dtype),
        lambda: f"{fn_name}: Half precision dtypes not supported. Got {dtype}",
    )


def check_is_matrix(A: TensorLikeType, f_name: str, arg_name: str = "A"):
    torch._check(
        len(A.shape) >= 2,
        lambda: f"{f_name}: The input tensor {arg_name} must have at least 2 dimensions.",
    )


def get_higher_type(a: type, b: type) -> type:
    """
    Returns the higher of the two given Number types.

    The types are ordered bool -> int -> float -> complex.
    """
    a, b = _maybe_get_pytype(a), _maybe_get_pytype(b)
    # Type checking
    if a not in _ordered_types or b not in _ordered_types:
        raise RuntimeError(f"Expected builtin numeric types, found {a}, {b}")

    if a is b:
        return a

    for typ in _ordered_types:
        if a is typ:
            return b
        if b is typ:
            return a

    raise ValueError("Unknown Python scalar type!")


# Returns the higher of two torch datatypes a and b or, if the two
#   are not ordered relative to each other, the next
#   higher datatype
def get_higher_dtype(
    a: torch.dtype | TensorLikeType | NumberType | None,
    b: torch.dtype | TensorLikeType | NumberType | None,
) -> torch.dtype | None:
    """
    Computes the "lowest" datatype that is weakly
    "higher" than both a and b.
    """

    # Type checking
    if a is not None and not isinstance(a, (torch.dtype, TensorLike, Number)):
        raise AssertionError(
            f"a must be None, torch.dtype, TensorLike, or Number, got {type(a)}"
        )
    if b is not None and not isinstance(b, (torch.dtype, TensorLike, Number)):
        raise AssertionError(
            f"b must be None, torch.dtype, TensorLike, or Number, got {type(b)}"
        )

    def _extract_dtype(
        x: torch.dtype | TensorLikeType | NumberType | None,
    ) -> torch.dtype | None:
        if x is None:
            return None
        if isinstance(x, torch.dtype):
            return x
        if isinstance(x, TensorLike):
            return x.dtype
        if isinstance(x, Number):
            return type_to_dtype(type(x))

        raise RuntimeError("Unexpected type given to _extract_dtype!")

    a, b = _extract_dtype(a), _extract_dtype(b)

    if a is b:
        return a

    if a is None:
        return b

    if b is None:
        return a

    ordered_datatypes = (
        (torch.bool,),
        (torch.uint8, torch.int8),
        (torch.int16,),
        (torch.int32,),
        (torch.int64,),
        (torch.float16, torch.bfloat16),
        (torch.float32,),
        (torch.float64,),
        (torch.complex32, torch.bcomplex32),
        (torch.complex64,),
        (torch.complex128,),
    )

    for idx, dtypes in enumerate(ordered_datatypes):
        if a in dtypes and b in dtypes:
            return ordered_datatypes[idx + 1][0]
        if a in dtypes:
            return b
        if b in dtypes:
            return a

    raise RuntimeError("Unexpected termination!")


def check_pin_memory(pin_memory: bool):
    torch._check_not_implemented(
        not pin_memory, lambda: "PrimTorch does not support pinned memory"
    )


def check_layout(layout: torch.layout):
    torch._check_not_implemented(
        layout == torch.strided, lambda: f"PrimTorch doesn't support layout={layout}"
    )


# TODO: maybe unify with can_cast_to?
def is_weakly_lesser_type(a: type, b: type) -> bool:
    """
    Compares two types, a and b, returning True if a is weakly "less" than b.

    The comparison is determined by the following type ordering: bool, int, float, complex.
    """

    a, b = _maybe_get_pytype(a), _maybe_get_pytype(b)

    if a not in _ordered_types or b not in _ordered_types:
        raise RuntimeError(f"Expected builtin numeric types, found {a}, {b}")

    for typ in _ordered_types:
        if a == typ:
            return True
        if b == typ:
            return False

    raise RuntimeError("Unexpected termination!")


def can_safe_cast_to(*, cast_to: torch.dtype, cast_from: torch.dtype) -> bool:
    for fn in (is_complex_dtype, is_float_dtype, is_integer_dtype, is_boolean_dtype):
        if fn(cast_to):
            return True
        if fn(cast_from):
            return False

    raise ValueError(f"Received unknown dtypes {cast_to}, {cast_from}!")


def check_same_dtype(*args):
    """
    Checks that all Tensors in args have the same device and that all Numbers have the
    same corresponding Python type.

    Raises a RuntimeError when:
      - args contains an object whose type is not Tensor or Number
      - two Tensors objects in args have different dtypes
      - two Number objects in args have different types
      - there are Tensors and Numbers in args, and one of those Tensors corresponding
          Python types is different from the type of one of those Numbers
    """
    full_dtype = None
    scalar_type = None

    # pyrefly: ignore [bad-assignment]
    for arg in args:
        if isinstance(arg, Number):
            # Scalar type checking is disabled (and may be removed in the future)
            continue
            # if scalar_type is None:
            #     scalar_type = type(arg)

            # if scalar_type is not type(arg):
            #     msg = (
            #         "Scalar of type "
            #         + str(type(arg))
            #         + " is not the expected type of "
            #         + str(scalar_type)
            #         + "!"
            #     )
            #     raise RuntimeError(msg)
        elif isinstance(arg, TensorLike):
            if full_dtype is None:
                full_dtype = arg.dtype
            if scalar_type is None:
                scalar_type = dtype_to_type(arg.dtype)

            if full_dtype is not arg.dtype:
                msg = (
                    "Tensor with dtype "
                    + str(arg.dtype)
                    + " is not the expected dtype of "
                    + str(full_dtype)
                    + "!"
                )
                raise RuntimeError(msg)

            arg_type = dtype_to_type(arg.dtype)
            if arg_type is not scalar_type:
                msg = (
                    "Tensor with corresponding Python type "
                    + str(arg_type)
                    + " is not the expected type of "
                    + str(scalar_type)
                    + "!"
                )
                raise RuntimeError(msg)
        else:
            msg = (
                "Unexpected type when checking for same dtype, " + str(type(arg)) + "!"
            )
            raise RuntimeError(msg)


# Maps datatypes to their computation types for elementwise operations
_computation_dtype_map = {
    torch.bfloat16: torch.float32,
    torch.float16: torch.float32,
    torch.complex32: torch.complex64,
    torch.bcomplex32: torch.complex64,
}


def get_computation_dtype(dtype: torch.dtype) -> torch.dtype:
    return _computation_dtype_map.get(dtype, dtype)


_cpu_acc_type_map = {
    torch.bfloat16: torch.float64,
    torch.float16: torch.float64,
    torch.float32: torch.float64,
    torch.complex32: torch.complex128,
    torch.complex64: torch.complex128,
}


def get_acc_type(dtype: torch.dtype, device: torch.device) -> torch.dtype:
    # Equivalent to at::toAccumulateType, prefer computation_dtype where possible
    if device.type == "cpu":
        return _cpu_acc_type_map.get(dtype, dtype)
    else:
        return get_computation_dtype(dtype)


class ELEMENTWISE_TYPE_PROMOTION_KIND(Enum):
    DEFAULT = (0,)
    NO_OPMATH = (1,)
    INT_TO_FLOAT = (2,)
    ALWAYS_BOOL = (3,)
    COMPLEX_TO_FLOAT = (4,)
    BOOL_TO_LONG = (5,)


class REDUCTION_OUTPUT_TYPE_KIND(Enum):
    SAME = (0,)
    COMPLEX_TO_FLOAT = (1,)  # for complex types outputs corresponding real type
    KEEP_PROMOTED_TYPE = (2,)  # keep output in opmath type, needed for mean
    ALWAYS_BOOL = (3,)


# Describes the return type of the primitive:
#
#   - NEW, a new tensor is created
#   - VIEW, a view of an input tensor is returned
#   - INPLACE, one or more input tensors is modified
#
# these descriptors are mutually exclusive and exhaustive.
class RETURN_TYPE(Enum):
    NEW = (0,)
    VIEW = (1,)
    INPLACE = (2,)
    NONE = (3,)


# TODO: when NumberType contains the sym types, can simplify this
def number_type(
    x: NumberType | torch.SymInt | torch.SymFloat | torch.SymBool,
) -> type:
    if isinstance(x, torch.SymInt):
        return int
    elif isinstance(x, torch.SymFloat):
        return float
    elif isinstance(x, torch.SymBool):
        return bool
    else:
        return type(x)


def expr_type(x: sympy.Basic) -> type:
    import sympy

    if x.kind is sympy.core.kind.BooleanKind:
        return bool
    elif x.is_integer:  # type: ignore[attr-defined]
        return int
    else:
        # NB: Not strictly correct, but we don't support SymPy complex or bool.
        return float


# TODO: document type promotion kinds
def elementwise_dtypes(
    *_args,
    type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND,
) -> tuple[torch.dtype, torch.dtype]:
    """
    Computes the computation and result dtypes for elementwise type promotion
    on the given arguments and with the given elementwise type promotion kind.

    Note that not all inputs to an elementwise operation necessarily participate in type promotion.
    For example, the "alpha" parameter of torch.add does not participate in type promotion,
    although it may be cast to the Python type corresponding to the computation dtype that
    the type promotion algorithm determines.

    Default elementwise type promotion, which all other type promotion kinds tweak (see below),
    first decides which of four ordered types to use:

    bool -> integer -> floating point -> complex

    The selected type is the "lowest" type in the above list such that all number arguments
    have a weakly "lower" type and all tensor arguments have a weakly lower corresponding
    type for their dtype.

    Once the type is determined, the particular result dtype is found. The dtypes are
    partially ordered as follows:

    bool -> uint8, int8 -> int16 -> int32 -> int64 ->
      float16, bfloat16 -> float32 -> float64 -> complex32, bcomplex32 -> complex64 -> complex128

    The result dtype is selected by:
      - if no tensor's dtype has the same corresponding type as the one selected,
          then the result dtype is the (default) dtype corresponding to the selected type
          (for example, 1.5 + an integer tensor has a result dtype of the default floating point dtype)
      - if the result type is complex then the dtype is:
        -  the default complex dtype if there are no floating point or complex tensors
        -  if there are floating point or complex tensors with one or more dimensions, then
            the complex dtype corresponding to the highest corresponding complex dtype among those tensors
            (for example, double + cfloat -> cdouble)
        -  if there are only floating point or complex tensors with zero dimensions, then
            the complex dtype corresponding to the highest corresponding complex dtype among those tensors
      - if the first two cases do not apply, the result dtype is the highest dtype among
          all tensors with one or more dimensions of the output type, and if there are no such
          tensors then it's the highest dtype among all tensors with zero dimensions of the output type
          (for example, long + half -> half, even if the half tensor has zero dimensions)

    The "corresponding complex dtypes" are:
      float16    -> complex32
      bfloat16   -> bcomplex32
      float32    -> complex64
      float64    -> complex128
      complex32  -> complex32
      complex64  -> complex64
      complex128 -> complex128

    The DEFAULT type promotion kind computes per above, and then uses the result dtype to pick a computation
    dtype by mapping low precision floating point and complex dtypes as follows:

      float16   -> float32
      bfloat16  -> bcomplex32
      complex32 -> complex64

    This is referred to as "op math", and the NO_OPMATH type promotion kind disables this mapping, making the
    computation dtype the same as the result dtype when it's selected. NO_OPMATH is appropriate for kernels
    which perform no mathematical operations on their tensors (see below for examples).

    The INT_TO_FLOAT type promotion kind maps boolean and integer result dtypes to the default floating point dtype,
    and computation dtypes to the appropriate op math dtype.

    The COMPLEX_TO_FLOAT type promotion kind maps complex result dtypes to the corresponding float dtype, following this
    mapping:

        complex32  -> float16
        complex64  -> float32
        complex128 -> float64

    Note that COMPLEX_TO_FLOAT derives the computation dtype as the DEFAULT setting does.

    The BOOL_TO_LONG type promotion kind maps boolean computation and result dtypes to long.

    The ALWAYS_BOOL type promotion kind always sets the result dtype to bool.

    Example operators for each type promotion option:
      DEFAULT                 : add
      NO_OPMATH               : where, nextafter, cat
      INT_TO_FLOAT            : sin
      COMPLEX_TO_FLOAT        : abs
      BOOL_TO_LONG            : pow
      ALWAYS_BOOL             : eq

    """

    args = tuple(x for x in _args if x is not None)

    highest_type: type = bool

    # Import sympy locally, as importing it eagerly at a module level is too slow
    # See https://dev-discuss.pytorch.org/t/delving-into-what-happens-when-you-import-torch/1589
    import sympy

    for x in args:
        if not isinstance(x, (Number, TensorLike, sympy.Basic)):
            msg = f"Unexpected type {str(type(x))} when computing elementwise type promotion!"
            raise ValueError(msg)

        if isinstance(x, Number):
            highest_type = get_higher_type(highest_type, number_type(x))
        elif isinstance(x, sympy.Basic):
            highest_type = get_higher_type(highest_type, expr_type(x))
        else:
            # x is a TensorLike
            highest_type = get_higher_type(highest_type, dtype_to_type(x.dtype))

    result_dtype = None

    def _find_highest_dtype_filtered(
        args, filter, *, float_as_complex=False
    ) -> torch.dtype | None:
        zero_dim_tensor_dtype = None
        one_plus_dim_tensor_dtype = None
        for x in args:
            if isinstance(x, TensorLike) and filter(x.dtype):
                _dtype = x.dtype
                if float_as_complex and is_float_dtype(_dtype):
                    _dtype = corresponding_complex_dtype(_dtype)
                if x.ndim == 0:
                    zero_dim_tensor_dtype = get_higher_dtype(
                        zero_dim_tensor_dtype, _dtype
                    )
                else:
                    # x.ndim > 0
                    one_plus_dim_tensor_dtype = get_higher_dtype(
                        one_plus_dim_tensor_dtype, _dtype
                    )

        # Prefers dtype of tensors with one or more dimensions
        if one_plus_dim_tensor_dtype is not None:
            return one_plus_dim_tensor_dtype

        return zero_dim_tensor_dtype

    if highest_type is float:
        result_dtype = _find_highest_dtype_filtered(args, is_float_dtype)
        result_dtype = (
            torch.get_default_dtype() if result_dtype is None else result_dtype
        )
    elif highest_type is complex:
        result_dtype = _find_highest_dtype_filtered(
            args,
            lambda x: is_float_dtype(x) or is_complex_dtype(x),
            float_as_complex=True,
        )
        if result_dtype is None:
            result_dtype = corresponding_complex_dtype(torch.get_default_dtype())
    elif highest_type is int:
        result_dtype = _find_highest_dtype_filtered(args, is_integer_dtype)
        result_dtype = torch.long if result_dtype is None else result_dtype
    else:
        # highest_type is bool
        result_dtype = torch.bool

    if type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT:
        return get_computation_dtype(result_dtype), result_dtype
    elif type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH:
        return result_dtype, result_dtype
    elif type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT:
        if is_integer_dtype(result_dtype) or is_boolean_dtype(result_dtype):
            result_dtype = torch.get_default_dtype()
        return get_computation_dtype(result_dtype), result_dtype
    elif type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT:
        # NOTE: computation can still occur in a complex dtype
        computation_dtype = get_computation_dtype(result_dtype)
        if is_complex_dtype(result_dtype):
            result_dtype = corresponding_real_dtype(result_dtype)
        return computation_dtype, result_dtype
    elif type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG:
        if is_boolean_dtype(result_dtype):
            return torch.long, torch.long
        return get_computation_dtype(result_dtype), result_dtype
    elif type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL:
        return get_computation_dtype(result_dtype), torch.bool
    else:
        raise ValueError(f"Unknown type promotion kind {str(type_promotion_kind)}")


def reduction_dtypes(
    arg,
    output_dtype_kind: REDUCTION_OUTPUT_TYPE_KIND,
    dtype: torch.dtype | None = None,
) -> tuple[torch.dtype, torch.dtype | None]:
    # even though some reductions, like amin or amax, don't strictly require type promotion,
    # all the math ops (including comparisons) are still defined only for a computation type,
    # so promotion will still happen. We are doing it explicitly here
    inp_dtype = dtype if dtype is not None else arg.dtype
    computation_dtype = get_computation_dtype(inp_dtype)
    if (
        output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.SAME
        or output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT
    ):
        result_dtype = dtype if dtype else arg.dtype
        if (
            output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT
            and is_complex_dtype(result_dtype)
        ):
            result_dtype = corresponding_real_dtype(result_dtype)
    elif output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.KEEP_PROMOTED_TYPE:
        result_dtype = None
    else:  # ALWAYS_BOOL
        result_dtype = torch.bool
    return computation_dtype, result_dtype


# This function's logic is borrowed from the following functions defined in C++:
# batched_matrix_contiguous_strides and contiguous_strides
def make_contiguous_strides_for(
    shape: ShapeType, row_major: bool = True
) -> tuple[_IntLikeT | int, ...]:
    """
    Returns the strides of a contiguous tensor if row_major
    If row_major=True, it returns the strides of a contiguous batch of Fortran-contiguous matrices
    This is often used when calling external libraries like BLAS/LAPACK/cuSolver...
    """
    # contiguous_strides from c10/util/strides.h
    validate_shape(shape)
    if not shape:
        return ()

    from torch.fx.experimental.symbolic_shapes import is_nested_int

    multiplier: _IntLikeT | int = 1
    strides = []
    for l in reversed(shape):
        strides.append(multiplier)
        multiplier *= l if is_nested_int(l) else sym_max(l, 1)  # type:ignore[assignment]

    result = tuple(reversed(strides))

    # batched_matrix_contiguous_strides from aten/src/ATen/native/LinearAlgebraUtils.h
    if row_major:
        return result
    else:
        if len(shape) < 2:
            return result
        # Use sym_max to handle unbacked symbolic dimensions
        return result[:-2] + (1, sym_max(shape[-2], 1))  # type: ignore[return-value]


def make_channels_last_1d_strides_for(
    shape: Sequence[_IntLikeT],
) -> tuple[_IntLikeT | int, ...]:
    torch._check(
        len(shape) == 3,
        lambda: "Only tensors of rank 3 can use the channels_last_1d memory format",
    )

    multiplier: _IntLikeT | int = 1
    strides: list[_IntLikeT | int] = [0] * 3
    for idx in (1, -1, 0):
        # NOTE: intentional divergence from make_contiguous_strides_for
        # This is consistent with eager
        strides[idx] = multiplier
        multiplier *= shape[idx]

    return tuple(strides)


def make_channels_last_2d_strides_for(
    shape: Sequence[_IntLikeT],
) -> tuple[_IntLikeT | int, ...]:
    # TODO: maybe inform the user of channels_last_3d if rank of the tensor is 5?
    torch._check(
        len(shape) == 4,
        lambda: "Only tensors of rank 4 can use the channels_last memory format",
    )

    multiplier: _IntLikeT | int = 1
    strides: list[_IntLikeT | int] = [0] * 4
    for idx in (1, -1, -2, 0):
        # NOTE: intentional divergence from make_contiguous_strides_for
        # This is consistent with eager
        strides[idx] = multiplier
        multiplier *= shape[idx]

    return tuple(strides)


def make_channels_last_3d_strides_for(
    shape: Sequence[_IntLikeT],
) -> tuple[_IntLikeT | int, ...]:
    torch._check(
        len(shape) == 5,
        lambda: "Only tensors of rank 5 can use the channels_last_3d memory format",
    )

    multiplier: _IntLikeT | int = 1
    strides: list[_IntLikeT | int] = [0] * 5
    for idx in (1, -1, -2, -3, 0):
        # NOTE: intentional divergence from make_contiguous_strides_for
        # This is consistent with eager
        strides[idx] = multiplier
        multiplier *= shape[idx]

    return tuple(strides)


def make_channels_last_strides_for(
    shape: Sequence[_IntLikeT],
) -> tuple[_IntLikeT | int, ...]:
    ndim = len(shape) if isinstance(shape, Sequence) else 1
    if ndim == 3:
        return make_channels_last_1d_strides_for(shape)
    elif ndim == 4:
        return make_channels_last_2d_strides_for(shape)
    elif ndim == 5:
        return make_channels_last_3d_strides_for(shape)
    else:
        raise RuntimeError(
            f"no channels last format strides exist in {ndim} dimensions"
        )


def compute_reduction_output_shape(
    shape: ShapeType, dimensions: Sequence
) -> tuple[int, ...]:
    for idx in dimensions:
        validate_idx(len(shape), idx)

    new_shape = []
    for idx in range(len(shape)):
        if idx in dimensions:
            continue

        new_shape.append(shape[idx])

    return tuple(new_shape)


def validate_no_repeating_dims(dims: Sequence):
    if len(dims) != len(set(dims)):
        raise RuntimeError("duplicate value in the list of dims")


def reduction_dims(shape: ShapeType, dims: Sequence | None) -> tuple[int, ...]:
    if dims is None:
        return tuple(range(len(shape)))
    dims = tuple(canonicalize_dim(len(shape), idx) for idx in dims)
    validate_no_repeating_dims(dims)
    return dims


def set_correction(
    unbiased: bool | None = None,
    correction: NumberType | None = None,
) -> float:
    if correction is not None and unbiased is not None:
        raise RuntimeError("cannot specify both correction and unbiased arguments")
    elif correction is None and unbiased is None:
        correction = 1.0
    elif correction is None and unbiased is not None:
        correction = 0.0 if unbiased is False else 1.0
    # NB: we don't actually support symint here, but it's harmless to accept
    if not isinstance(correction, (IntLike, FloatLike)):
        raise ValueError("correction argument should be integer or float")
    # pyrefly: ignore [bad-return]
    return sym_float(correction)


def compute_required_storage_length(
    shape: ShapeType, strides: StrideType, storage_offset: int
) -> int:
    """Computes the minimum storage size to hold the given tensor geometry.

    Example
    =======

    This is the size of a newly allocated tensor's storage, in units of elements

    >>> t = torch.empty((10, 20))
    >>> compute_required_storage_length(t.shape, t.stride(), t.storage_offset())
    200

    >>> # xdoctest: +SKIP(failing)
    >>> t2 = torch.empty_strided((1, 2, 3), (5, 7, 11))
    >>> size = compute_required_storage_length(
    ...     t2.shape, t2.stride(), t2.storage_offset()
    ... )
    >>> size == t.storage().size()
    True

    A valid tensor may have a larger storage size, but never smaller

    >>> slice = torch.empty(100)[20:40]
    >>> slice.storage().size()
    100

    >>> compute_required_storage_length(
    ...     slice.shape, slice.stride(), slice.storage_offset()
    ... )
    40

    """
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    # Short-circuits if the shape has no elements
    # Note: we are unsafely assuming tensor is not empty here, without
    # runtime assertions.
    if guard_or_false(reduce(operator.mul, shape, 1) == 0):
        return 0

    max_offset = sum((x - 1) * y for x, y in zip(shape, strides))
    # +1 to account for the first element which offsets are taken from
    return 1 + storage_offset + max_offset


def check_in_bounds_for_storage(
    a: torch.TypedStorage, shape: ShapeType, strides: StrideType, storage_offset: int
):
    """
    Determines if the given shape, strides, and offset are valid for the given storage.
    """

    required_length = compute_required_storage_length(shape, strides, storage_offset)
    if a.size() < required_length:
        msg = (
            f"Can't view a storage of size {a.size()} with an offset of {storage_offset}, "
            f"shape of {str(shape)}, and strides of {str(strides)}, "
            f"which requires a storage of size {required_length}"
        )
        raise ValueError(msg)


# NOTE: This function should ideally be removed, but some Meta internal models
# packaged with `torch.package` are using it, so it will have to be removed
# at some point in the future when those models no longer use this function.
@deprecated(
    "`torch._prims_common.check` is deprecated and will be removed in the future. "
    "Please use `torch._check*` functions instead.",
    category=FutureWarning,
)
def check(
    b: bool, s: Callable[[], str], exc_type: type[Exception] = RuntimeError
) -> None:
    """
    Helper function for raising an error_type (default: RuntimeError) if a boolean condition fails.
    Error message is a callable producing a string (to avoid wasting time
    string formatting in non-error case, and also to make it easier for torchdynamo
    to trace.)

    .. note:: This function is planned for removal in the future. Please use
        `torch._check*` functions instead.
    """
    torch._check_with(exc_type, b, s)


# This combines is_channels_last_strides_2d and is_channels_last_strides_3d in
# c10/core/MemoryFormat.h into one function
# May return False when input sizes are data-dependent and the property is not
# determined.
def are_strides_like_channels_last_or_false(
    shape: Sequence[int], strides: Sequence[int]
) -> bool:
    from torch.fx.experimental.symbolic_shapes import (
        guard_or_true,
        statically_known_true,
    )

    ndim = len(shape)

    if ndim == 4:
        # Check for channels_last_2d
        dim_order = [1, 3, 2, 0]
    elif ndim == 5:
        # Check for channels_last_3d
        dim_order = [1, 4, 3, 2, 0]
    else:
        return False

    if guard_or_true(strides[1] == 0):
        return False

    min = 0
    for d in dim_order:
        if guard_or_true(shape[d] == 0):
            return False
        if guard_or_true(strides[d] < min):
            return False
        if d == 0 and min == strides[1]:
            return False
        min = strides[d]
        # Only multiply by shape[d] when size >= 1, matching C++ logic
        # shape[d]!=0 hence we know its >=1 here
        min *= shape[d]
    return True


def suggest_memory_format(x: TensorLikeType) -> torch.memory_format:
    """DDE-safe Python equivalent of ``Tensor.suggest_memory_format()``.

    Returns ``torch.channels_last`` for 4D tensors with NHWC-pattern strides,
    ``torch.channels_last_3d`` for 5D NDHWC, otherwise ``torch.contiguous_format``.
    Uses ``are_strides_like_channels_last_or_false`` internally — when
    contiguity can't be statically decided (e.g., unbacked symbolic strides),
    falls back to ``contiguous_format`` instead of raising a data-dependent
    error. Mirrors eager ``Tensor.suggest_memory_format()`` (with the default
    ``channels_last_strides_exact_match=false``) for inputs eager would also
    classify the same way.
    """
    if x.layout != torch.strided:
        return torch.contiguous_format

    if are_strides_like_channels_last_or_false(x.shape, x.stride()):
        return torch.channels_last if x.ndim == 4 else torch.channels_last_3d

    return torch.contiguous_format


def prod(xs: Sequence[NumberType]) -> NumberType:
    """Product of elements in input sequence. Returns 1 for empty sequence"""
    return reduce(operator.mul, xs, 1)


def is_expandable_to(shape: ShapeType, desired: ShapeType) -> bool:
    """Checks if a shape can be expanded to another shape.
    This is equivalent to checking if the two shapes are broadcastable.
    """
    # This is a Python implementation of
    # aten/src/ATen/ExpandUtils.h:is_expandable_to
    if len(shape) > len(desired):
        return False
    for i in range(len(shape)):
        if shape[-i - 1] != desired[-i - 1] and shape[-i - 1] != 1:
            return False
    return True


def mask_tensor(mask: TensorLikeType, t: TensorLikeType):
    """
    Similar to torch.where(mask, t, 0) but if t is boolean,
    result is also boolean and not promoted to int.
    """
    # torch.where(mask, t, False) is equivalent
    # but feels hacky and might break in the future
    if t.dtype is torch.bool:
        return mask.logical_and(t)
    else:
        return torch.where(mask, t, 0)


def get_aten_op(fn: Callable, name: str):
    """
    Given the __module__ of reference and its name, it returns
    (our best guess of) the ATen name of the associated operation

    Note: In ATen, the __name__ of a function within a module often
    starts by the module name. E.g. linalg_eigh, or special_zeta
    """
    module = fn.__module__
    prefix = "torch._refs"
    if not module.startswith(prefix):
        raise AssertionError(f"module must start with {prefix}, got {module}")
    module = module[len(prefix) :]
    # We want to go from .special / .nn.functional
    # to special and special_ / nn_functional_
    if module:
        module = module[1:]
        module = module.replace(".", "_")
        module = module + "_"
    return getattr(torch._ops.ops.aten, f"{module}{name}")


def dtype_or_default(dtype: torch.dtype | None) -> torch.dtype:
    return dtype if dtype is not None else torch.get_default_dtype()


def device_or_default(device: DeviceLikeType | None) -> DeviceLikeType:
    return device if device is not None else torch.device("cpu")


def layout_or_default(layout: torch.layout | None) -> torch.layout:
    return layout if layout is not None else torch.strided


def clone_preserve_strides(x):
    needed_size = compute_required_storage_length(
        x.size(), x.stride(), x.storage_offset()
    )
    # Our eager implementations for *_scatter ops are all primitives w.r.t autograd,
    # so these as_strided() calls are not seen by autograd.
    # We need to mimic this behavior in our ref/prim implementations.
    # TODO: a better way to handle this would be with a new op, "_unsafe_as_strided"
    # We should revisit this when we add a compositional as_strided op,
    # and also as part of https://github.com/pytorch/pytorch/issues/90507
    try:
        old = torch._C._dispatch_tls_is_dispatch_key_excluded(
            torch._C.DispatchKey.ADInplaceOrView
        )
        torch._C._dispatch_tls_set_dispatch_key_excluded(
            torch._C.DispatchKey.ADInplaceOrView, True
        )
        buffer = torch.as_strided(x, (needed_size,), (1,), 0).clone()
        return torch.as_strided(buffer, x.size(), x.stride(), x.storage_offset())
    finally:
        torch._C._dispatch_tls_set_dispatch_key_excluded(
            torch._C.DispatchKey.ADInplaceOrView, old
        )


def alert_not_deterministic(caller: str):
    if torch.are_deterministic_algorithms_enabled():
        if torch.is_deterministic_algorithms_warn_only_enabled():
            warnings.warn(
                f"{caller} does not have a deterministic implementation, but you set "
                f"'torch.use_deterministic_algorithms(True, warn_only=True)'. "
                f"You can file an issue at https://github.com/pytorch/pytorch/issues "
                f"to help us prioritize adding deterministic support for this operation.",
                stacklevel=2,
            )
        else:
            torch._check(
                False,
                lambda: (
                    f"{caller} does not have a deterministic implementation, but you set "
                    f"'torch.use_deterministic_algorithms(True)'. You can turn off "
                    f"determinism just for this operation, or you can use the "
                    f"'warn_only=True' option, if that's acceptable for your application. "
                    f"You can also file an issue at https://github.com/pytorch/pytorch/issues "
                    f"to help us prioritize adding deterministic support for this operation."
                ),
            )


class CUDARngStateHelper:
    @staticmethod
    def get_torch_state_as_tuple(
        fake_mode: AbstractContextManager[Any] = nullcontext(),
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        with fake_mode:
            seed = torch.tensor(torch.cuda.initial_seed())
            offset = torch.tensor(torch.cuda._get_rng_state_offset())
            return seed, offset

    @staticmethod
    def set_torch_state_tensor(seed, offset):
        # Rng state is [64-bit seed, 64-bit offset]
        seed_portion = seed.reshape([1]).view(torch.uint8)
        offset_portion = offset.reshape([1]).view(torch.uint8)
        new_state = torch.cat([seed_portion, offset_portion])
        torch.cuda.set_rng_state(new_state)

    @staticmethod
    def set_new_offset(relative_offset):
        torch.cuda._set_rng_state_offset(relative_offset.item())
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import gzip
import json
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
from typing import Any, TYPE_CHECKING
from typing_extensions import deprecated, Self
from warnings import warn

import torch
import torch.autograd.profiler as prof
from torch._C import _get_privateuse1_backend_name
from torch._C._profiler import (
    _add_execution_trace_observer,
    _disable_execution_trace_observer,
    _enable_execution_trace_observer,
    _ExperimentalConfig,
    _remove_execution_trace_observer,
)
from torch._environment import is_fbcode
from torch._utils_internal import profiler_allow_cudagraph_cupti_lazy_reinit_cuda12
from torch.autograd import kineto_available, ProfilerActivity
from torch.profiler._memory_profiler import MemoryProfile, MemoryProfileTimeline


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


__all__ = [
    "supported_activities",
    "ProfilerAction",
    "schedule",
    "tensorboard_trace_handler",
    "profile",
    "ExecutionTraceObserver",
]
PROFILER_STEP_NAME = "ProfilerStep"

_WARNINGS_SHOWN = set()


def _warn_once(msg, category=UserWarning, stacklevel=2):
    if msg not in _WARNINGS_SHOWN:
        _WARNINGS_SHOWN.add(msg)
        warn(msg, category=category, stacklevel=stacklevel)


def _parse_custom_profiler_config(
    experimental_config: _ExperimentalConfig | None,
) -> dict[str, Any]:
    if experimental_config is None:
        return {}
    raw = getattr(experimental_config, "custom_profiler_config", None)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"backend": raw}
    return parsed if isinstance(parsed, dict) else {}


class _NumpyEncoder(json.JSONEncoder):
    """
    Json encoder for numpy types (np.int, np.float, np.array etc.)
    Returns default encoder if numpy is not available
    """

    def default(self, obj):
        """Encode NumPy types to JSON"""
        try:
            import numpy as np
        except ImportError:
            return json.JSONEncoder.default(self, obj)
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return json.JSONEncoder.default(self, obj)


def supported_activities():
    """
    Returns a set of supported profiler tracing activities.

    Note: profiler uses CUPTI library to trace on-device CUDA kernels.
    In case when CUDA is enabled but CUPTI is not available, passing
    ``ProfilerActivity.CUDA`` to profiler results in using the legacy CUDA
    profiling code (same as in the legacy ``torch.autograd.profiler``).
    This, in turn, results in including CUDA time in the profiler table output,
    but not in the JSON trace.
    """
    return torch.autograd._supported_activities()


class _ITraceObserver(ABC):
    """Abstract interface for a Trace observer.
    This satisfies 3 methods: start, stop and cleanup"""

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def cleanup(self):
        pass


def _parse_activities(
    activities: Iterable[ProfilerActivity | dict[ProfilerActivity, list[str]]],
) -> tuple[set[ProfilerActivity], dict[ProfilerActivity, set[str]]]:
    """Parse a mixed activities list into a set of activities and a filter dict.

    Each item is either a bare ``ProfilerActivity`` (collect all defaults) or a
    ``dict[ProfilerActivity, list[str]]`` (collect only the named subset).
    An empty list value (e.g. ``{CUDA: []}``) means collect nothing for that group.
    """
    parsed_activities: set[ProfilerActivity] = set()
    activity_filters: dict[ProfilerActivity, set[str]] = {}
    for item in activities:
        if isinstance(item, ProfilerActivity):
            if item in parsed_activities:
                raise ValueError(f"Activity {item} specified more than once")
            parsed_activities.add(item)
        elif isinstance(item, dict):
            for key, val in item.items():
                if key in parsed_activities:
                    raise ValueError(f"Activity {key} specified more than once")
                parsed_activities.add(key)
                activity_filters[key] = set(val)
        else:
            raise TypeError(f"Expected ProfilerActivity or dict, got {type(item)}")
    return parsed_activities, activity_filters


class _KinetoProfile:
    """Low-level profiler wrap the autograd profile

    Args:
        activities (iterable): list of activity groups (CPU, CUDA) to use in profiling, supported values:
            ``torch.profiler.ProfilerActivity.CPU``, ``torch.profiler.ProfilerActivity.CUDA``,
            ``torch.profiler.ProfilerActivity.XPU``.
            Default value: ProfilerActivity.CPU and (when available) ProfilerActivity.CUDA
            or (when available) ProfilerActivity.XPU.

            Each item can be a ``ProfilerActivity`` enum (collects all default
            activity types for that group) or a ``dict`` mapping a ``ProfilerActivity``
            to a list of individual activity type names to collect, e.g.
            ``{ProfilerActivity.CUDA: ["GPU_MEMCPY", "CUDA_RUNTIME"]}``.
            An empty list (e.g. ``{ProfilerActivity.CUDA: []}``) means collect
            nothing for that group.
            The same activity group must not appear more than once.
            See :class:`~torch.profiler.ProfilerActivity` for the valid
            activity type names and device-specific behavior.
        record_shapes (bool): save information about operator's input shapes.
        profile_memory (bool): track tensor memory allocation/deallocation (see ``export_memory_timeline``
            for more details).
        with_stack (bool): record source information (file and line number) for the ops.
        with_flops (bool): use formula to estimate the FLOPS of specific operators
            (matrix multiplication and 2D convolution).
        with_modules (bool): record module hierarchy (including function names)
            corresponding to the callstack of the op. e.g. If module A's forward call's
            module B's forward which contains an aten::add op,
            then aten::add's module hierarchy is A.B

            .. deprecated::
                ``with_modules`` is deprecated and will be removed in a future version.
                It only collects data for TorchScript models, which are themselves
                deprecated, and does nothing in eager mode. Use ``with_stack=True``,
                which records ``nn.Module`` events for eager models.
        experimental_config (_ExperimentalConfig) : A set of experimental options
            used by profiler libraries like Kineto. Note, backward compatibility is not guaranteed.
        execution_trace_observer (ExecutionTraceObserver) : A PyTorch Execution Trace Observer object.
            `PyTorch Execution Traces <https://arxiv.org/pdf/2305.14516.pdf>`__ offer a graph based
            representation of AI/ML workloads and enable replay benchmarks, simulators, and emulators.
            When this argument is included the observer start() and stop() will be called for the
            same time window as PyTorch profiler.
        acc_events (bool): Enable the accumulation of FunctionEvents across multiple profiling cycles
        post_processing_timeout_s (float): Optional timeout in seconds for post-processing profiler
            results. In this context, post-processing happens after the profiling itself has finished.
            If specified, event parsing will stop after this duration and return partial results. Useful
            for handling large traces that may take too long to process.


    .. note::
        This API is experimental and subject to change in the future.

        Enabling shape and stack tracing results in additional overhead.
        When record_shapes=True is specified, profiler will temporarily hold references to the tensors;
        that may further prevent certain optimizations that depend on the reference count and introduce
        extra tensor copies.
    """

    def __init__(
        self,
        *,
        activities: Iterable[ProfilerActivity | dict[ProfilerActivity, list[str]]]
        | None = None,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
        with_flops: bool = False,
        with_modules: bool = False,
        experimental_config: _ExperimentalConfig | None = None,
        execution_trace_observer: _ITraceObserver | None = None,
        acc_events: bool = False,
        custom_trace_id_callback: Callable[[], str] | None = None,
