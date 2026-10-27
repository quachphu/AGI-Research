    # default: runtime_optimized_saved_values
    # more aggressive: more_aggressive_saved_values
    # full aggressive: aggressive_recomputation_saved_values

    all_recomputable_banned_nodes = sorted(
        recomputable_banned_nodes, key=_size_of, reverse=True
    )
    if len(all_recomputable_banned_nodes) == 0:
        # Nothing left for the knapsack to trade off, so this is the same cut the
        # knapsack would return with an empty dont_ban.
        return aggressive_recomputation_saved_values
    memories_banned_nodes = [
        get_normalized_size(_size_of(i)) for i in all_recomputable_banned_nodes
    ]
    runtimes_banned_nodes = [
        estimate_runtime(node) for node in all_recomputable_banned_nodes
    ]
    from torch.utils._mode_utils import no_dispatch

    def get_saved_values_knapsack(
        memory_budget: float, node_info: NodeInfo, joint_graph: fx.Graph
    ) -> tuple[list[fx.Node], float]:
        with no_dispatch():
            (
                expected_runtime,
                saved_node_idxs,
                recomputable_node_idxs,
            ) = _optimize_runtime_with_given_memory(
                joint_graph,
                memories_banned_nodes,
                runtimes_banned_nodes,
                max(memory_budget, 0),
                node_info,
                all_recomputable_banned_nodes,
            )
        dont_ban: OrderedSet[fx.Node] = OrderedSet()
        for idx in recomputable_node_idxs:
            # if idx in all_recomputable_banned_nodes:
            try:
                dont_ban.add(all_recomputable_banned_nodes[idx])
            except BaseException:
                pass

        if not dont_ban.issubset(all_recomputable_banned_nodes):
            raise AssertionError(
                "dont_ban must be a subset of all_recomputable_banned_nodes"
            )

        saved_values, _ = solve_min_cut(
            joint_graph,
            node_info,
            aggressive_options,
            dont_ban,
        )
        if AOT_PARTITIONER_DEBUG:
            create_structured_trace_for_min_cut_info(
                joint_graph=joint_graph,
                all_recomputable_banned_nodes=all_recomputable_banned_nodes,
                saved_node_idxs=saved_node_idxs,
                recomputable_node_idxs=recomputable_node_idxs,
                expected_runtime=expected_runtime,
                memories_banned_nodes=[
                    _size_of(i) for i in all_recomputable_banned_nodes
                ],
                normalized_memories_banned_nodes=memories_banned_nodes,
                runtimes_banned_nodes=runtimes_banned_nodes,
                min_cut_saved_values=saved_values,
                memory_budget=memory_budget,
                min_act_size=min_act_size,
                max_act_size=max_act_size,
                saved_values_act_size=estimate_activations_size(saved_values),
                more_aggressive_saved_values_mem_ratio=more_aggressive_saved_values_mem_ratio,
                aggressive_recomputation_saved_values_mem_ratio=aggressive_recomputation_saved_values_mem_ratio,
            )
        return saved_values, expected_runtime

    if config.visualize_memory_budget_pareto:

        def estimate_for_budget(b: float) -> tuple[float, float, float]:
            saved_values, expected_runtime = get_saved_values_knapsack(
                b, node_info=node_info, joint_graph=joint_graph
            )
            return (
                b,
                sum(runtimes_banned_nodes) - expected_runtime,
                get_mem_ratio(saved_values),
            )

        options = [estimate_for_budget(0.0), estimate_for_budget(1.0)]

        if options[0][1:] != options[1][1:]:
            bisects = [(options[0], options[1])]
            while bisects:
                lhs, rhs = bisects.pop()
                if rhs[0] - lhs[0] < 1e-3:
                    options.append(lhs)
                    options.append(rhs)
                    continue
                mid = estimate_for_budget((lhs[0] + rhs[0]) / 2)
                if mid[1:] != lhs[1:]:
                    bisects.append((lhs, mid))
                if mid[1:] != rhs[1:]:
                    bisects.append((mid, rhs))
        options.sort()

        import matplotlib.pyplot as plt

        x_values = [item[2] for item in options]
        y_values = [item[1] for item in options]

        # Plotting the values with updated axis labels and chart title
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, marker="o")

        # Adding labels for each point
        for i, txt in enumerate(x_values):
            plt.annotate(
                f"{txt:.4f}",
                (txt, y_values[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

        plt.xlabel("Memory Budget")
        plt.ylabel("Runtime of Recomputed Components")
        plt.title("Pareto Frontier of Memory Budget vs. Recomputation Runtime")
        plt.grid(True)
        fig = plt.gcf()
        plt.show()
        fig_dir = os.getcwd()
        if config.memory_budget_pareto_dir is not None:
            fig_dir = config.memory_budget_pareto_dir
            os.makedirs(fig_dir, exist_ok=True)
        rank_suffix = ""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank_suffix = f"_rank_{torch.distributed.get_rank()}"
        fig_name = os.path.join(
            fig_dir, f"memory_budget_pareto{rank_suffix}_{get_aot_graph_name()}.svg"
        )
        fig.savefig(fig_name)
        log.warning("Generated Pareto frontier curve at %s", fig_name)

    # todo(chilli): Estimated doesn't align exactly with actual - actual is
    # usually less memory than estimated. i'm guessing (actually quite
    # unsure about this) that's because estimated is just only including
    # tensors we actually banned from recompute, but there may be other
    # tensors that we choose to save.

    return get_saved_values_knapsack(
        memory_budget=memory_budget, node_info=node_info, joint_graph=joint_graph
    )[0]


def _stable_target_str(target: Any) -> str:
    """Stringify a node target stably across processes.

    ``str()`` on a plain Python-function target (e.g. ``torch.sym_not``) renders
    its ``repr`` including the object's memory address (``<function sym_not at
    0x...>``), which differs per process. That poisons cross-rank graph hashing.
    Use FX's qualified name for callables so equal graphs hash equally.
    """
    from torch.fx.node import _get_qualified_name

    if callable(target):
        try:
            return _get_qualified_name(target)
        except Exception:
            return str(target)
    return str(target)


def _cone_hashes(graph: torch.fx.Graph) -> dict[torch.fx.Node, str]:
    """Compute a forward-looking structural hash for each node.

    Each node's hash captures its "role" in the graph toward the output:
    hash(self_key, sorted(cone_hash(user) for user in users)).

    For placeholders, self_key uses tensor metadata instead of the node name
    (which varies across ranks). This makes the hash invariant to the original
    node naming and ordering.
    """
    hashes: dict[torch.fx.Node, str] = {}
    for node in reversed(list(graph.nodes)):
        if node.op == "placeholder":
            val = node.meta.get("val")
            if isinstance(val, torch.Tensor):
                # Exclude shape: different ranks may have different input
                # shapes (e.g., shard sizes) for structurally identical graphs.
                self_key: tuple[Any, ...] = (
                    "placeholder",
                    str(val.dtype),
                    val.requires_grad,
                )
            elif isinstance(val, torch.SymInt):
                self_key = ("placeholder", "symint")
            else:
                self_key = ("placeholder",)
        elif node.op == "output":
            self_key = ("output",)
        else:
            self_key = (node.op, _stable_target_str(node.target))

        user_hashes = tuple(sorted(hashes[u] for u in node.users))
        hashes[node] = hashlib.sha256(
            str((self_key, user_hashes)).encode("utf-8")
        ).hexdigest()

    return hashes


def _canonical_node_names(graph: torch.fx.Graph) -> dict[torch.fx.Node, str]:
    """Build a canonical name mapping for graph nodes using Kahn's algorithm.

    Returns a dict mapping each node to a deterministic name like "node_0",
    "node_1", etc. The mapping is invariant to the original node ordering and
    naming, so structurally equivalent graphs (e.g., traced with different dict
    iteration orders across distributed ranks) produce identical mappings.

    Does NOT modify the graph.
    """
    cone = _cone_hashes(graph)

    indeg: dict[torch.fx.Node, int] = dict.fromkeys(graph.nodes, 0)
    for node in graph.nodes:
        for user in node.users:
            indeg[user] += 1

    canonical_idx: dict[torch.fx.Node, int] = {}

    def _canonical_key(node: torch.fx.Node) -> tuple[Any, ...]:
        if node.op == "placeholder":
            val = node.meta.get("val")
            if isinstance(val, torch.Tensor):
                # Exclude shape: different ranks may have different input
                # shapes (e.g., shard sizes) for structurally identical graphs.
                meta_key: tuple[Any, ...] = (
                    str(val.dtype),
                    val.requires_grad,
                )
            elif isinstance(val, torch.SymInt):
                meta_key = ("symint",)
            else:
                meta_key = ()
            return (0, meta_key, cone[node])
        elif node.op == "get_attr":
            return (1, str(node.target))
        elif node.op == "output":
            return (3,)
        else:
            input_indices = tuple(canonical_idx[n] for n in node.all_input_nodes)
            return (2, _stable_target_str(node.target), input_indices)

    # Seed the heap with nodes that have no dependencies.
    # The counter ensures deterministic ordering when keys are equal.
    counter = 0
    ready: list[tuple[tuple[Any, ...], int, fx.Node]] = []
    for node in graph.nodes:
        if indeg[node] == 0:
            heapq.heappush(ready, (_canonical_key(node), counter, node))
            counter += 1

    canonical_order: list[fx.Node] = []

    while ready:
        _, _, cur = heapq.heappop(ready)
        canonical_order.append(cur)
        canonical_idx[cur] = len(canonical_idx)

        for user in cur.users:
            indeg[user] -= 1
            if indeg[user] == 0:
                heapq.heappush(ready, (_canonical_key(user), counter, user))
                counter += 1

    return {node: f"node_{i}" for i, node in enumerate(canonical_order)}


def _sync_decision_cross_ranks(
    joint_graph: torch.fx.Graph, saved_values: list[torch.fx.Node]
) -> list[torch.fx.Node]:
    # use the same policy across different GPUs
    from torch._subclasses.fake_tensor import unset_fake_temporarily

    def has_collectives(joint_graph: torch.fx.Graph) -> bool:
        for node in joint_graph.nodes:
            if isinstance(
                node.target, torch._ops.OpOverload
            ) and node.target.namespace in {"_c10d_functional", "c10d_functional"}:
                return True
        return False

    if not (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
        and has_collectives(joint_graph)
    ):
        return saved_values

    canonical = _canonical_node_names(joint_graph)
    reverse_canonical = {v: k for k, v in canonical.items()}

    def has_same_nodes(joint_graph: torch.fx.Graph) -> bool:
        # Use canonical names for a hash that is invariant to node ordering
        # and naming. This correctly identifies structurally equivalent graphs
        # even when different ranks trace with different dict iteration orders.
        def _node_hash_str(n: torch.fx.Node) -> str:
            # For placeholders, n.target is the rank-local name (e.g.,
            # primals_1) which may refer to different inputs on different
            # ranks. Use only the canonical name and op for these.
            if n.op == "placeholder":
                return f"{canonical[n]}:{n.op}"
            return f"{canonical[n]}:{n.op}:{_stable_target_str(n.target)}"

        node_str = "/".join(
            _node_hash_str(n)
            for n in sorted(joint_graph.nodes, key=lambda n: canonical[n])
        )
        inputs = hashlib.sha256(node_str.encode("utf-8")).hexdigest()
        all_inputs = [None for _ in range(torch.distributed.get_world_size())]
        with no_dispatch(), unset_fake_temporarily():
            # TODO: maybe use a different process group?
            torch.distributed.all_gather_object(all_inputs, inputs)
            for rank, x in enumerate(all_inputs):
                if all_inputs[0] != x:
                    log.debug(
                        "Skipping sync decision cross rank due to different inputs between rank 0 and rank %s",
                        rank,
                    )
                    return False
        return True

    if has_same_nodes(joint_graph):
        with no_dispatch(), unset_fake_temporarily():
            # Communicate saved values using canonical names so that
            # node names (which may differ across ranks) don't matter.
            objects = [[canonical[x] for x in saved_values]]
            saved_ops_names_all_ranks: list[list[str]] = [
                [] for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather_object(saved_ops_names_all_ranks, objects[0])
            saved_sizes: list[int] = []
            saved_ops_with_sizes: dict[str, int] = {}

            for idx, saved_ops_names in enumerate(saved_ops_names_all_ranks):
                saved_nodes = [
                    reverse_canonical[op_name] for op_name in saved_ops_names
                ]
                saved_size = 0
                for node in saved_nodes:
                    size_of_node = _size_of(node)
                    saved_size += size_of_node
                    if idx == torch.distributed.get_rank():
                        saved_ops_with_sizes[node.name] = size_of_node
                saved_ops_with_sizes["total size"] = saved_size
                saved_sizes.append(saved_size)

            saved_sizes_tensor = torch.tensor(
                saved_sizes,
                device=torch.distributed.distributed_c10d._get_object_coll_device(),
            )
            torch.distributed.all_reduce(
                saved_sizes_tensor, op=torch.distributed.distributed_c10d.ReduceOp.MAX
            )

            picked_rank_idx = int(torch.argmin(saved_sizes_tensor).item())
            sync_decision_cross_ranks_str = f"picked_rank_idx={picked_rank_idx}, saved_nodes of current rank={saved_ops_with_sizes}"
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "aot_joint_graph_sync_decision_cross_ranks",
                    "encoding": "string",
                },
                payload_fn=lambda: sync_decision_cross_ranks_str,
            )

            saved_values = [
                reverse_canonical[n] for n in saved_ops_names_all_ranks[picked_rank_idx]
            ]

    return saved_values


def thread_graphsafe_rng_from_hops(
    module: fx.GraphModule, is_backward: bool
) -> fx.GraphModule:
    """
    Graph-safe RNG lets torch.compile use CUDA Graphs for graphs with RNG ops.
    For graphs without HOPs, the partitioner adds placeholder nodes
    fwd_rng_state_* and bw_rng_state_* to the forward and backward graphs. At
    runtime, the AOTDispatcher retrieves these RNG states and passes them to the
    compiled graphs.

    This works well for no-HOP graphs. With HOPs, the partitioner runs
    recursively: it first partitions the HOP (producing forward/backward HOP
    subgraphs) and then stitches them back into the outer joint graph. For HOPs
    that contain RNG ops, the outer joint graph now includes HOP subgraph
    modules with extra RNG placeholders. We must thread these placeholders
    through the outer module partitioned forward and backward graphs—this
    function does exactly that. It collects the RNG placeholder nodes from the
    HOPs and creates corresponding placeholders in the outer forward and
    backward graphs.

    There is a catch: for a short period, the joint graph is in a “bad” state.
    The HOP subgraphs expect additional inputs (because of the new
    placeholders), but the outer graph call sites don't yet provide them. We
    can't fix this in the joint graph because the joint graph's input signature
    is fixed (primals, tangents). As a compromise, we keep the joint graph in
    somewhat of a bad state for some time and, once the outer forward and
    backward graphs are partitioned, insert the corresponding RNG placeholders
    and wire up the calls.
    """

    rng_count = 0
    rng_string = "bwd_rng_state" if is_backward else "fwd_rng_state"
    last_input = next(reversed(module.graph.find_nodes(op="placeholder")))
    for hop_node in module.graph.find_nodes(
        op="call_function", target=torch.ops.higher_order.invoke_subgraph
    ):
        subgraph = getattr(module, hop_node.args[0].target)
        if isinstance(subgraph, fx.GraphModule):
            new_rng_inputs: list[fx.Node] = []
            for placeholder_node in subgraph.graph.find_nodes(op="placeholder"):
                if rng_string in placeholder_node.name:
                    # Found a rng state placeholder in the hop graph, lets add
                    # the corresponding node in the outer graph
                    with module.graph.inserting_after(last_input):
                        rng_state = module.graph.placeholder(
                            f"{rng_string}_{rng_count}"
                        )
                        rng_count += 1
                        rng_state.meta["val"] = placeholder_node.meta["val"]
                        last_input = rng_state
                        new_rng_inputs.append(rng_state)

            if new_rng_inputs:
                # Pass on the new args that include the new_rng_inputs
                with module.graph.inserting_after(hop_node):
                    new_hop_node_with_fixed_args = module.graph.create_node(
                        "call_function",
                        torch.ops.higher_order.invoke_subgraph,
                        (*hop_node.args, *new_rng_inputs),  # type: ignore[arg-type]
                        {},
                    )
                    hop_node.replace_all_uses_with(
                        new_hop_node_with_fixed_args, propagate_meta=True
                    )

                # Setup the eager_input_vals
                eager_vals = hop_node.meta.get("eager_input_vals")
                if eager_vals:
                    eager_args, eager_kwargs = eager_vals
                    new_eager_args = (
                        *eager_args,
                        *[inp.meta["val"] for inp in new_rng_inputs],
                    )
                    new_hop_node_with_fixed_args.meta["eager_input_vals"] = (
                        new_eager_args,
                        eager_kwargs,
                    )
                module.graph.erase_node(hop_node)

    return module


def classify_nodes(
    joint_module: fx.GraphModule,
    static_lifetime_input_indices: list[int],
    num_fwd_outputs: int,
) -> NodeInfo:
    name_to_node = get_name_to_node(joint_module.graph)
    required_bw_nodes: OrderedSet[fx.Node] = OrderedSet()
    for node in joint_module.graph.nodes:
        if node.op == "placeholder" and "tangents" in node.target:
            required_bw_nodes.add(node)
        elif _must_be_in_backward(node):
            required_bw_nodes.add(node)

        if node in required_bw_nodes:
            if node.op == "call_function" and node.target is control_deps_hop:
                # control_deps mixes forward and backward deps.  Only
                # propagate required_bw to getitem users that extract a
                # backward-only dep.  getitem(cd, k) with k>=1 maps to
                # deps[k-1] (args[k+1]).
                deps = node.args[2:]
                for user in node.users:
                    if (
                        user.target is operator.getitem
                        and isinstance(user.args[1], int)
                        and 1 <= user.args[1] <= len(deps)
                    ):
                        dep = deps[user.args[1] - 1]
                        if isinstance(dep, fx.Node) and dep in required_bw_nodes:
                            required_bw_nodes.add(user)
                    else:
                        # Chained control_deps (e.g. wait/sync cd depends on
                        # the record's cd via additional_deps) -- taint
                        # conservatively.
                        required_bw_nodes.add(user)
            else:
                required_bw_nodes.update(node.users)

    primal_inputs = list(filter(_is_primal, joint_module.graph.nodes))
    fwd_seed_offset_inputs = list(filter(_is_fwd_seed_offset, joint_module.graph.nodes))
    inputs = primal_inputs + fwd_seed_offset_inputs
    fwd_outputs, bwd_outputs, fwd_outputs_descs, bwd_outputs_descs = (
        _extract_fwd_bwd_outputs(joint_module, num_fwd_outputs=num_fwd_outputs)
    )
    # Note: [tangents_closure vs required_bw_nodes]
    #
    # required_bw_nodes is used to determine which nodes need edges to
    # the sink. It is important to also track tangents closure because
    # that determines whether you can save that tensor, i.e., whether you
    # want to connect x_in or x_out to the sink.
    tangents_closure = required_bw_nodes.copy()
    required_bw_nodes.update(
        o for o in bwd_outputs if o is not None and o.op != "output"
    )
    forward_only_graph = _extract_graph_with_inputs_outputs(
        joint_module.graph, inputs, fwd_outputs, fwd_outputs_descs, "forward"
    )
    required_fw_nodes: OrderedSet[fx.Node] = OrderedSet(
        name_to_node[node.name]
        for node in forward_only_graph.nodes
        if node.op != "output" and node.name in name_to_node
    )
    unclaimed_nodes: OrderedSet[fx.Node] = OrderedSet(
        node
        for node in joint_module.graph.nodes
        if node not in required_fw_nodes and node not in required_bw_nodes
    )
    static_lifetime_input_nodes = OrderedSet(
        p for i, p in enumerate(primal_inputs) if i in static_lifetime_input_indices
    )
    fw_cnt = 0
    fw_order = {}
    for node in joint_module.graph.nodes:
        if node in required_fw_nodes:
            fw_order[node] = fw_cnt
            fw_cnt += 1
    return NodeInfo(
        inputs,
        required_fw_nodes,
        required_bw_nodes,
        tangents_closure,
        unclaimed_nodes,
        fw_order,
        static_lifetime_input_nodes,
    )


def min_cut_rematerialization_partition(
    joint_module: fx.GraphModule,
    _joint_inputs: Any,
    compiler: str = "inductor",
    *,
    num_fwd_outputs: int,
    static_lifetime_input_indices: list[int] | None = None,
) -> tuple[fx.GraphModule, fx.GraphModule]:
    """
    Partitions the joint graph such that the backward recomputes the forward.
    Recomputing helps in trading off memory bandwidth with computation.

    To create the fwd and bwd graph, we copy the joint graph, manually set the
    outputs to just original forward or backward outputs. And then we run the
    resulting graphs through dead code elimination.

    .. warning::
        This API is experimental and likely to change.

    Args:
        joint_module(fx.GraphModule): The joint forward and backward graph. This
            is the result of AOT Autograd tracing.
        _joint_inputs: The inputs to the joint graph. This is unused.
        compiler: This option determines the default set of recomputable ops.
            Currently, there are two options: ``nvfuser`` and ``inductor``.
        recomputable_ops: This is an optional set of recomputable ops. If this
            is not None, then this set of ops will be used instead of the
            default set of ops.
        num_fwd_outputs: The number of outputs from the forward graph.

    Returns:
        Returns the generated forward and backward Fx graph modules.
    """

    joint_module.graph.eliminate_dead_code()
    joint_module.recompile()

    fx_g = joint_module.graph

    #  add the CSE pass
    if config.cse:
        # CSE runs before partitioning, so do not merge a forward-only value
        # with a value that is also needed by the backward graph. Otherwise the
        # partitioner may save the merged value as an extra forward output.
        _, bwd_outputs, _, _ = _extract_fwd_bwd_outputs(
            joint_module, num_fwd_outputs=num_fwd_outputs
        )
        backward_dependencies: OrderedSet[fx.Node] = OrderedSet()

        backward_dependency_stack = [
            output
            for output in bwd_outputs
            if output is not None and output.op != "output"
        ]
        while backward_dependency_stack:
            node = backward_dependency_stack.pop()
            if node in backward_dependencies:
                continue
            backward_dependencies.add(node)
            backward_dependency_stack.extend(node.all_input_nodes)

        def partition_key(node: fx.Node) -> bool:
            return node in backward_dependencies

        cse_graph = fx_graph_cse(fx_g, extra_node_key=partition_key)
        joint_module.graph = cse_graph
    joint_graph = joint_module.graph

    graph_has_recomputable_ops = has_recomputable_ops(joint_module)
    graph_has_recomputable_rng_ops = has_recomputable_rng_ops(joint_module)
    if graph_has_recomputable_ops:
        joint_module = cleanup_recompute_tags(joint_module, is_default_partition=False)
    if not config.unsafe_allow_optimization_of_collectives:
        force_save_collectives(joint_module)

    force_save_effectful_ops(joint_module)
    force_save_bw_mutation_src(joint_module)

    if static_lifetime_input_indices is None:
        static_lifetime_input_indices = []
    node_info = classify_nodes(
        joint_module, static_lifetime_input_indices, num_fwd_outputs
    )

    # networkx blows up on graphs with no required backward nodes
    # Since there's nothing to partition anyway, and the default partitioner can "handle"
    # this case, send our graph over to the default partitioner.
    if len(node_info.required_bw_nodes) == 0:
        return default_partition(
            joint_module,
            _joint_inputs,
            num_fwd_outputs=num_fwd_outputs,
            static_lifetime_input_indices=static_lifetime_input_indices,
            static_lifetime_input_nodes=node_info.static_lifetime_input_nodes,
        )

    for node in reversed(joint_module.graph.nodes):
        if node.op == "output":
            node.dist_from_bw = int(1e9)
        elif not node_info.is_required_fw(node):
            node.dist_from_bw = 0
        else:
            node.dist_from_bw = int(1e9)
            for user in node.users:
                node.dist_from_bw = min(node.dist_from_bw, user.dist_from_bw + 1)

    # memory_budget override resolution, in increasing precedence:
    #   1. config.activation_memory_budget (global default).
    #   2. Legacy node.meta["memory_budget"], set by the old
    #      `MemoryBudgetMode` / `set_memory_budget` allow_in_graph marker. First
    #      matching node wins (preserves prior behavior); we may want to remove
    #      this path later.
    #   3. `torch.autograd.graph.region_activation_memory_budget(...)`, read off
    #      node.meta["custom"] via `_get_memory_budget_annotation` and propagated
    #      onto every annotated node via _COPY_META_FIELDS["custom"]; overrides
    #      the legacy reader.
    memory_budget = config.activation_memory_budget
    for node in joint_graph.nodes:
        if isinstance(node.meta.get("memory_budget", None), float):
            memory_budget = node.meta["memory_budget"]
            break

    # The partitioner applies a single budget per joint graph, so all annotated
    # nodes must agree and the annotation must cover every forward op; otherwise
    # the caller mixed budgets or annotated only part of a graph (across a graph
    # break each graph is resolved independently). Collect both in one pass.
    region_budgets: OrderedSet[float] = OrderedSet()
    unannotated_fw_ops: list[fx.Node] = []
    for node in joint_graph.nodes:
        budget = _get_memory_budget_annotation(node)
        if budget is not None:
            region_budgets.add(budget)
        elif node.op == "call_function" and node_info.is_required_fw(node):
            unannotated_fw_ops.append(node)

    # A budget must consistently cover the entire forward, including HOP bodies.
    # Recurse into nested subgraph modules: collect their budgets (for the
    # agreement check) and flag any unannotated call_function (for coverage). In
    # a consistent graph every node in every body carries the budget, so an
    # unannotated body op means the budget did not cover that HOP.
    all_budgets: OrderedSet[float] = OrderedSet(region_budgets)
    for _, sub in joint_module.named_modules():
        if isinstance(sub, fx.GraphModule) and sub.graph is not joint_graph:
            for node in sub.graph.nodes:
                b = _get_memory_budget_annotation(node)
                if b is not None:
                    all_budgets.add(b)
                elif node.op == "call_function":
                    unannotated_fw_ops.append(node)
    if len(all_budgets) > 1:
        raise RuntimeError(
            f"torch.autograd.graph.region_activation_memory_budget: "
            f"conflicting budgets within a single joint graph (including nested "
            f"subgraphs): {sorted(all_budgets)}. Use a graph break to separate "
            f"regions that need different budgets."
        )

    if all_budgets:
        if unannotated_fw_ops:
            raise RuntimeError(
                f"torch.autograd.graph.region_activation_memory_budget: must "
                f"cover the entire forward of a graph (including HOP bodies), but "
                f"{len(unannotated_fw_ops)} forward op(s) are unannotated. Wrap "
                f"the whole forward in a single region; use a graph break to scope "
                f"different budgets to different graphs. Note that a graph break "
                f"inside the annotated region can also cause unannotated ops here. "
                f"Unannotated ops: {[n.name for n in unannotated_fw_ops]}."
            )
        memory_budget = next(iter(all_budgets))
    saved_values = choose_saved_values_set(
        joint_graph,
        node_info,
        memory_budget=memory_budget,
    )
    # pyrefly: ignore [unbound-name]
    if config._sync_decision_cross_ranks:
        saved_values = _sync_decision_cross_ranks(joint_graph, saved_values)

    saved_sym_nodes = list(
        filter(
            lambda n: is_sym_node(n) and not _is_assert_only_symbool(n), saved_values
        )
    )
    saved_opaque_nodes = list(filter(is_opaque_node, saved_values))
    saved_values = list(
        filter(lambda n: not is_sym_node(n) and not is_opaque_node(n), saved_values)
    )

    # NB: saved_sym_nodes will be mutated to reflect the actual saved symbols
    fw_module, bw_module = _extract_fwd_bwd_modules(
        joint_module,
        saved_values,
        # pyrefly: ignore [bad-argument-type]
        saved_sym_nodes=saved_sym_nodes,
        saved_opaque_nodes=saved_opaque_nodes,
        num_fwd_outputs=num_fwd_outputs,
        static_lifetime_input_nodes=node_info.static_lifetime_input_nodes,
    )
    if graph_has_recomputable_ops:
        if graph_has_recomputable_rng_ops:
            fw_module, bw_module = functionalize_rng_ops(
                joint_module, fw_module, bw_module, len(saved_sym_nodes)
            )
    bw_module = reordering_to_mimic_autograd_engine(bw_module)

    # pyrefly: ignore [unbound-name]
    if config.enable_activation_offloading:
        from ._activation_offloading.activation_offloading import (
            enable_activation_offloading,
        )

        enable_activation_offloading(
            fw_module,
            bw_module,
            num_fwd_outputs,
            node_info.static_lifetime_input_nodes,
        )

    # raise all getitem ops to as early as possible
    # this is helpful for memory, especially in the case of aot_eager backend
    fw_module = raise_getitems(fw_module)
    bw_module = raise_getitems(bw_module)

    fw_module = thread_graphsafe_rng_from_hops(fw_module, is_backward=False)
    bw_module = thread_graphsafe_rng_from_hops(bw_module, is_backward=True)

    if AOT_PARTITIONER_DEBUG:
        # Calculate sorted sizes of saved values
        sorted_sizes = sorted([(_size_of(i), str(i)) for i in saved_values])

        # Log total theoretical activations stored
        total_activations_size_gb = sum(_size_of(i) for i in saved_values) / 1e9
        log.info("Theoretical Activations Stored: %.2f GB", total_activations_size_gb)

        # Log theoretical per activation storage sizes
        log.info("Theoretical Per Activation Storage Sizes: %s", sorted_sizes)
        fw_module_nodes = OrderedSet(
            node.name for node in fw_module.graph.nodes if node.op == "call_function"
        )
        bw_module_nodes = OrderedSet(
            node.name for node in bw_module.graph.nodes if node.op == "call_function"
        )
        remat_nodes = fw_module_nodes & bw_module_nodes

        counts: dict[str, int] = defaultdict(int)
        for node in fw_module.graph.nodes:
            if node.name in remat_nodes and hasattr(node.target, "_overloadpacket"):
                counts[str(node.target._overloadpacket)] += 1
        log.info(
            "# remat/fw/bw: %d/%d/%d",
            len(remat_nodes),
            len(fw_module_nodes),
            len(bw_module_nodes),
        )
        rematerialized_ops = sorted(
            counts.items(), key=operator.itemgetter(1), reverse=True
        )
        log.info("Count of Ops Rematerialized: %s", rematerialized_ops)
    return fw_module, bw_module


def draw_graph(
    traced: torch.fx.GraphModule,
    fname: str,
    figname: str = "fx_graph",
    clear_meta: bool = True,
    prog: str | list[str] | None = None,
    parse_stack_trace: bool = False,
    dot_graph_shape: str | None = None,
) -> None:
    if clear_meta:
        new_graph = copy.deepcopy(traced.graph)
        traced = fx.GraphModule(traced, new_graph)
        for node in traced.graph.nodes:
            node.meta = {}  # pyrefly: ignore[implicit-any]
    base, ext = os.path.splitext(fname)
    if not ext:
        ext = "." + config.torch_compile_graph_format
    log.info("Writing FX graph to file: %s%s", base, ext)
    g = graph_drawer.FxGraphDrawer(
        traced,
        figname,
        parse_stack_trace=parse_stack_trace,
        dot_graph_shape=dot_graph_shape,
    )
    x = g.get_main_dot_graph()
    fname = f"{base}{ext}"
    graph_format = ext.lstrip(".")
    if graph_format in {"dot", "raw"}:
        # pydot's write_dot() invokes Graphviz with -Tdot.  For large FX graphs
        # that layout step can dominate compile debugging, so write raw DOT
        # source directly for text graph dumps.
        x.write(fname)
    elif prog is None:
        write_method = getattr(x, "write_" + graph_format)
        write_method(fname)
    else:
        write_method = getattr(x, "write_" + graph_format)
        write_method(fname, prog=prog)
<|endoftext|>from __future__ import annotations

import copy
import math
import os
import sys
from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, TYPE_CHECKING

import torch
import torch.fx as fx
from torch.hub import tqdm
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._content_store import ContentStoreWriter

from .compile_utils import get_outputs, get_placeholders


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


is_tuple = object()


@dataclass
class LoadTensorMeta:
    size: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


class ConcreteProp(torch.fx.Interpreter):
    def __init__(
        self,
        mod: fx.GraphModule,
        *,
        writer: ContentStoreWriter | None = None,
        skip_offload: bool = False,
    ) -> None:
        super().__init__(mod)
        self.writer = writer
        self.skip_offload = skip_offload
        self.seen_storages: set[StorageWeakRef] = set()
        self.pbar: Any = None

    def run_node(self, n: fx.Node) -> Any:
        self.pbar.update(1)
        r = super().run_node(n)
        name = n.name

        if isinstance(r, torch.Tensor):
            if self.writer is None:
                n.meta["concrete_value"] = r
            else:
                if StorageWeakRef(r.untyped_storage()) in self.seen_storages:
                    # Refuse to offload tensors which alias other live
                    # tensors, because this will violate operator contracts
                    n.meta["concrete_value"] = None
                else:
                    if not self.skip_offload:
                        self.writer.write_tensor(os.path.join("eager", name), r)
                    n.meta["concrete_value"] = LoadTensorMeta(
                        r.size(), r.stride(), r.dtype, r.device
                    )
                    self.seen_storages.add(StorageWeakRef(r.untyped_storage()))
        else:
            n.meta["concrete_value"] = is_tuple

        return r

    def propagate(self, *args: Any) -> Any:
        mod = self.module
        if not isinstance(mod, fx.GraphModule):
            raise AssertionError(f"expected fx.GraphModule, got {type(mod)}")
        with tqdm(
            desc="Saving intermediates for delta debugging",
            total=len(mod.graph.nodes),
            disable=self.writer is None,
        ) as pbar:
            self.pbar = pbar
            r = super().run(*args)
            if not self.skip_offload:
                pbar.set_description(
                    "Saved!  To skip next time, run with --skip-saving-eager-intermediates"
                )
            return r


def is_load_tensor_node(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is torch.ops.debugprims.load_tensor.default
    )


# inplace modifies node/inps
def _convert_node_to_placeholder(
    graph: fx.Graph, node: fx.Node, inps: list[torch.Tensor]
) -> bool:
    if node.op == "output" or node.op == "placeholder":
        return False

    if is_load_tensor_node(node):
        return False

    concrete_val = node.meta.get("concrete_value", None)

    if isinstance(concrete_val, torch.Tensor):
        node.op = "placeholder"
        node.target = node.name
        node.args = ()
        node.kwargs = {}

        inps.append(concrete_val)
        return True

    elif concrete_val is None:
        return False

    elif concrete_val is is_tuple:
        r = False
        for tuple_user in list(node.users):
            r = _convert_node_to_placeholder(graph, tuple_user, inps) or r
        # NB: We must not erase the node at this point, because
        # we are iterating over the nodes and this would change
        # the iteration order
        # graph.erase_node(node)
        return r

    elif isinstance(concrete_val, LoadTensorMeta):
        node.op = "call_function"
        node.target = torch.ops.debugprims.load_tensor.default
        node.args = (
            os.path.join("eager", node.name),
            concrete_val.size,
            concrete_val.stride,
        )
        node.kwargs = {
            "device": concrete_val.device,
            "dtype": concrete_val.dtype,
        }
        return True

    return False


def create_minified_hlo_graph(
    minified_fx_graph: fx.GraphModule, inputs: Sequence[torch.Tensor]
) -> None:
    """
    Takes minified FX graph as primary input, and ports it to HLO via StableHLO
    Provides minified HLO graph as output, and archive them to local directory
    """
    hlo_dir = f"{os.getcwd()}/hlo_files"
    os.makedirs(hlo_dir, exist_ok=True)

    from torch_xla.stablehlo import save_torch_model_as_stablehlo

    save_torch_model_as_stablehlo(minified_fx_graph, inputs, hlo_dir)


def dump_state(fx_g: fx.GraphModule, inps: Sequence[torch.Tensor]) -> None:
    print(
        f"""
# Working Repro with {len(fx_g.graph.nodes)} nodes
inps = {[(i.shape, i.dtype, str(i.device)) for i in inps]}
inps = [torch.zeros(())] + [torch.ones(shape, dtype=dtype, device=device) for (shape, dtype, device) in inps]
{fx_g.code}
"""
    )


def is_power_of_two(n: int) -> bool:
    if n == 0:
        return False
    return (n & (n - 1)) == 0


@dataclass
class ReproState:
    graph: fx.Graph
    inps: Sequence[torch.Tensor]

    def __post_init__(self) -> None:
        ph_nodes = get_placeholders(self.graph)
        if len(ph_nodes) != len(self.inps):
            raise AssertionError(
                f"len(ph_nodes)={len(ph_nodes)} != len(self.inps)={len(self.inps)}"
            )


def minifier(
    fail_f: fx.GraphModule,
    inps: Sequence[torch.Tensor],
    module_fails: Callable[[fx.GraphModule, Sequence[torch.Tensor]], bool],
    dump_state: Callable[[fx.GraphModule, Sequence[torch.Tensor]], None] = dump_state,
    *,
    save_dir: str | None = None,
    offload_to_disk: bool = False,
    skip_offload: bool = False,
    skip_sanity: bool = False,
    max_granularity: int | None = None,
) -> tuple[fx.GraphModule, Sequence[torch.Tensor]]:
    """
    Minimizes a FX graph with given inputs, such that the resulting FX graph still returns True for module_fails.

    Does 2 main strategies:
    1. Truncates suffix: Removes some suffix from the graph and sets a new output.
    2. Delta Debugging: Tries replacing half of the graph with inputs. If fails,
        tries replacing quarter of the graph, etc.

    >>> # xdoctest: +SKIP(failing)
    >>> failing_function = fx.symbolic_trace(f)
    >>> minimize(failing_function, [torch.randn(5)], lambda fx_g, inps: fx_g(*inps))

    note: module_fails returns True if it fails.
    """

    failing_graph = fail_f.graph
    cur_size = len(failing_graph.nodes)

    if max_granularity is not None and not is_power_of_two(max_granularity):
        raise RuntimeError(f"max_granularity {max_granularity} not power of two")

    num_queries = 0

    def deepcopy_fx_graph(fx_graph: fx.Graph) -> fx.Graph:
        return fx.GraphModule(fail_f, copy.deepcopy(fx_graph)).graph

    def graph_fails(graph: fx.Graph, inps: Sequence[torch.Tensor]) -> bool:
        nonlocal num_queries
        graph = copy.deepcopy(graph)
        num_queries += 1
        mod = fx.GraphModule(fail_f, graph)
        mod.graph.lint()
        return module_fails(mod, inps)

    writer = None
    if offload_to_disk:
        if save_dir is None:
            raise AssertionError("save_dir must not be None when offload_to_disk=True")
        writer = ContentStoreWriter(save_dir)

    ConcreteProp(fail_f, writer=writer, skip_offload=skip_offload).propagate(*inps)
    if not skip_sanity and not graph_fails(failing_graph, inps):
        raise RuntimeError("Input graph did not fail the tester")
    print(f"Started off with {cur_size} nodes", file=sys.stderr)

    def _register_strategy(
        strategy: Callable[[fx.Graph, Sequence[torch.Tensor], int], ReproState | None],
        name: str,
    ) -> Callable[[ReproState, int], ReproState | None]:
        @wraps(strategy)
        def new_func(old_state: ReproState, granularity: int = 1) -> ReproState | None:
            print(file=sys.stderr)
            print(
                f"Strategy: {name} (G: {granularity}) "
                f"({len(old_state.graph.nodes)} nodes, {len(old_state.inps)} inputs)",
                file=sys.stderr,
            )
            new_state = strategy(
                deepcopy_fx_graph(old_state.graph), list(old_state.inps), granularity
            )
            if new_state is not None:
                new_nodes = len(new_state.graph.nodes)
                old_nodes = len(old_state.graph.nodes)
                new_inps = len(new_state.inps)
                old_inps = len(old_state.inps)
                new_outs = len(get_outputs(new_state.graph))
                old_outs = len(get_outputs(old_state.graph))
                progress_made = False
                if new_nodes < old_nodes:
                    progress_made = True
                    print(
                        f"SUCCESS: Went from {old_nodes} to {new_nodes} nodes",
                        file=sys.stderr,
                    )
                if new_inps > old_inps:
                    progress_made = True
                    print(
                        f"SUCCESS: Went from {old_inps} to {new_inps} inputs",
                        file=sys.stderr,
                    )
                if new_outs < old_outs:
                    progress_made = True
                    print(
                        f"SUCCESS: Went from {old_outs} to {new_outs} outputs",
                        file=sys.stderr,
                    )

                if not progress_made:
                    raise RuntimeError("Success raised but no progress made?")

                if not graph_fails(new_state.graph, new_state.inps):
                    print(
                        "WARNING: Something went wrong, not applying this minification",
                        file=sys.stderr,
                    )
                    return None
                return new_state
            else:
                print(f"FAIL: {name}", file=sys.stderr)
            return None

        return new_func

    def register_strategy(
        name: str,
    ) -> Callable[
        [Callable[[fx.Graph, Sequence[torch.Tensor], int], ReproState | None]],
        Callable[[ReproState, int], ReproState | None],
    ]:
        return partial(_register_strategy, name=name)

    @register_strategy("Truncate suffix")
    def remove_suffix(
        cur_graph: fx.Graph, cur_inps: Sequence[torch.Tensor], granularity: int
    ) -> ReproState | None:
        tested: set[int] = set()
        new_graph = fx.Graph()
        env: dict[fx.Node, fx.Node] = {}
        for idx, node in enumerate(cur_graph.nodes):
            new_node = new_graph.node_copy(node, lambda x: env[x])
            if node.op not in ["placeholder", "output"]:
                # If idx is divisible by (granularity * 2), it would have been checked already.
                if (
                    idx % granularity == 0
                    and (idx % (granularity * 2) != 0)
                    and idx not in tested
                ):
                    output_node = new_graph.output((new_node,))
                    if len(new_graph.nodes) < len(cur_graph.nodes) and graph_fails(
                        new_graph, cur_inps
                    ):
                        return ReproState(new_graph, cur_inps)
                    else:
                        tested.add(idx)
                        new_graph.erase_node(output_node)
            env[node] = new_node
        return None

    @register_strategy("Remove outputs")
    def remove_outputs(
        cur_graph: fx.Graph, cur_inps: Sequence[torch.Tensor], granularity: int
    ) -> ReproState | None:
        granularity = max(1, granularity // 2)
        output: fx.Node | None = None
        for idx, node in enumerate(cur_graph.nodes):
            node.idx = idx  # type: ignore[attr-defined]
            if node.op == "output":
                output = node
                break

        if output is None:
            return None

        if isinstance(output.args[0], fx.Node):
            return None

        # output.args[0] is a tuple/list of nodes when returning multiple outputs
        output_args_raw = output.args[0]
        if not isinstance(output_args_raw, (list, tuple)):
            raise AssertionError(
                f"expected output_args_raw to be list or tuple, got {type(output_args_raw)}"
            )
        output_args = sorted(
            output_args_raw,
            key=lambda x: x.idx if isinstance(x, fx.Node) else int(1e9),  # type: ignore[attr-defined]
        )
        if len(output_args) == 1:
            return None

        for idx in range(0, len(output_args), granularity):
            output.args = (output_args[:idx] + output_args[idx + granularity :],)
            if graph_fails(cur_graph, cur_inps):
                return ReproState(cur_graph, cur_inps)
        return None

    def remove_unused_inputs_unchecked(cur_state: ReproState) -> ReproState | None:
        cur_graph = cur_state.graph
        cur_inps = cur_state.inps
        ph_nodes = list(get_placeholders(cur_graph))
        if len(ph_nodes) != len(cur_inps):
            raise AssertionError(
                f"len(ph_nodes)={len(ph_nodes)} != len(cur_inps)={len(cur_inps)}"
            )

        new_inps: list[torch.Tensor] = []
        for idx in range(len(ph_nodes)):
            if len(ph_nodes[idx].users) == 0:
                cur_graph.erase_node(ph_nodes[idx])
            else:
                new_inps.append(cur_inps[idx])
        if len(new_inps) < len(cur_inps):
            return ReproState(cur_graph, new_inps)
        return None

    def remove_unused_inputs_checked(cur_state: ReproState) -> ReproState | None:
        new_state = remove_unused_inputs_unchecked(cur_state)
        if new_state is not None and graph_fails(new_state.graph, new_state.inps):
            return new_state
        return None

    def _remove_unused_wrapper(
        cur_graph: fx.Graph, cur_inps: Sequence[torch.Tensor], granularity: int
    ) -> ReproState | None:
        return remove_unused_inputs_checked(ReproState(cur_graph, cur_inps))

    remove_unused_inputs = register_strategy("Remove unused inputs")(
        _remove_unused_wrapper
    )

    @register_strategy("Eliminate dead code")
    def eliminate_dead_code(
        cur_graph: fx.Graph, cur_inps: Sequence[torch.Tensor], granularity: int
    ) -> ReproState | None:
        if cur_graph.eliminate_dead_code() and graph_fails(cur_graph, cur_inps):
            return ReproState(cur_graph, cur_inps)
        return None

    def _consolidate_placeholders(
        cur_graph: fx.Graph, inps: list[torch.Tensor]
    ) -> fx.Graph:
        new_graph = fx.Graph()
        env = {}
        seen_non_placeholder = False

        # Move all placeholders to the front; also, if any load_tensor
        # is at the front, convert it into an input (because it can be live
        # all the time)
        for node in cur_graph.nodes:
            if node.op == "placeholder":
                new_node = new_graph.node_copy(node, lambda x: env[x])
                env[node] = new_node
            elif not seen_non_placeholder and is_load_tensor_node(node):
                new_node = new_graph.placeholder(node.name)
                env[node] = new_node
                inps.append(
                    torch.ops.debugprims.load_tensor.default(*node.args, **node.kwargs)
                )
            else:
                seen_non_placeholder = True

        # Move everyone else
        for node in cur_graph.nodes:
            if node not in env:
                new_node = new_graph.node_copy(node, lambda x: env[x])
                env[node] = new_node
        return new_graph

    @register_strategy("Delta Debugging")
    def delta_debugging(
        cur_graph: fx.Graph, cur_inps: Sequence[torch.Tensor], granularity: int
    ) -> ReproState | None:
        num_nodes = len(cur_graph.nodes)
        for start_range in range(0, num_nodes, granularity):
            is_removing = False
            new_graph = deepcopy_fx_graph(cur_graph)
            new_inps = list(cur_inps[:])
            end_range = min(num_nodes, start_range + granularity)
            for idx in range(start_range, end_range):
                new_node = list(new_graph.nodes)[idx]
                if _convert_node_to_placeholder(new_graph, new_node, new_inps):
                    is_removing = True
            if not is_removing:
                continue
            new_graph.eliminate_dead_code()
            new_graph = _consolidate_placeholders(new_graph, new_inps)
            new_state = remove_unused_inputs_unchecked(ReproState(new_graph, new_inps))
            if new_state is None:
                new_state = ReproState(new_graph, new_inps)
            if graph_fails(new_state.graph, new_state.inps):
                return ReproState(new_state.graph, new_state.inps)

        return None

    @register_strategy("Consolidate Inputs")
    def consolidate_inputs(
        cur_graph: fx.Graph, cur_inps: Sequence[torch.Tensor], granularity: int
    ) -> ReproState | None:
        old_len = len(cur_inps)
        new_inps = list(cur_inps[:])
        cur_graph = _consolidate_placeholders(cur_graph, new_inps)
        if len(cur_inps) > old_len and graph_fails(cur_graph, new_inps):
            return ReproState(cur_graph, new_inps)
        return None

    failing_state = ReproState(failing_graph, inps)

    def try_granularity(
        failing_state: ReproState, granularity: int, use_non_granular: bool
    ) -> ReproState | None:
        print(f"Trying granularity {granularity}", file=sys.stderr)

        strategies = []
        num_nodes = len(failing_state.graph.nodes)
        num_outputs = len(get_outputs(failing_state.graph))
        if num_outputs > num_nodes // 2:
            strategies += [remove_outputs]

        if use_non_granular:
            strategies += [
                eliminate_dead_code,
                remove_unused_inputs,
                consolidate_inputs,
            ]

        strategies += [remove_suffix, delta_debugging]

        for strategy in strategies:
            new_state = strategy(failing_state, granularity)
            if new_state is not None:
                return new_state
        return None

    while True:
        dump_state(fx.GraphModule(fail_f, failing_state.graph), failing_state.inps)
        granularity = int(2 ** (math.floor(math.log2(len(failing_state.graph.nodes)))))
        if max_granularity is not None:
            granularity = min(max_granularity, granularity)
        new_state = try_granularity(failing_state, granularity, use_non_granular=True)
        if new_state is not None:
            failing_state = new_state
            continue

        granularity //= 2
        has_progress = False
        while granularity >= 1:
            new_state = try_granularity(
                failing_state, granularity, use_non_granular=False
            )
            if new_state is not None:
                failing_state = new_state
                has_progress = True
                break
            granularity //= 2
        if has_progress:
            continue

        new_state = remove_outputs(failing_state, 1)
        if new_state is not None:
            failing_state = new_state
            continue

        break

    if not graph_fails(failing_state.graph, failing_state.inps):
        raise RuntimeError("Uh oh, something went wrong :( Final graph is not failing")

    print(f"Made {num_queries} queries", file=sys.stderr)
    failing_fx = fx.GraphModule(fail_f, failing_state.graph)

    # If XLA debugging environment is enabled, create minified HLO graph as well
    if "XLA_HLO_DEBUG" in os.environ:
        create_minified_hlo_graph(failing_fx, failing_state.inps)

    dump_state(failing_fx, failing_state.inps)
    print("Wrote minimal repro out to repro.py", file=sys.stderr)
    return failing_fx, failing_state.inps
<|endoftext|># Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextlib
import functools
import itertools
from collections.abc import Callable  # noqa: TC003
from functools import partial
from typing import Any, cast, NoReturn, TYPE_CHECKING
from typing_extensions import ParamSpec, TypeVar

import torch
from torch import Tensor
from torch._C._functorch import is_batchedtensor
from torch._functorch.predispatch import (
    _add_batch_dim,
    _remove_batch_dim,
    _vmap_decrement_nesting,
    _vmap_increment_nesting,
    lazy_load_decompositions,
)
from torch.utils._pytree import (
    _broadcast_to_and_flatten,
    tree_flatten,
    tree_map_,
    tree_unflatten,
    TreeSpec,
)


if TYPE_CHECKING:
    from collections.abc import Generator, Iterable


_P = ParamSpec("_P")
_R = TypeVar("_R")

in_dims_t = int | tuple[Any, ...]
out_dims_t = int | tuple[int, ...] | None


def doesnt_support_saved_tensors_hooks(f: Callable[_P, _R]) -> Callable[_P, _R]:
    message = (
        "torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. "
        "Please open an issue with your use case."
    )

    @functools.wraps(f)
    def fn(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with torch.autograd.graph.disable_saved_tensors_hooks(message):
            return f(*args, **kwargs)

    return fn


# Checks that all args-to-be-batched have the same batch dim size
def _validate_and_get_batch_size(
    flat_in_dims: list[int | None], flat_args: list[Any]
) -> int:
    batch_sizes = [
        arg.size(in_dim)
        for in_dim, arg in zip(flat_in_dims, flat_args)
        if in_dim is not None
    ]
    if len(batch_sizes) == 0:
        raise ValueError("vmap: Expected at least one Tensor to vmap over")
    if batch_sizes and any(size != batch_sizes[0] for size in batch_sizes):
        raise ValueError(
            f"vmap: Expected all tensors to have the same size in the mapped "
            f"dimension, got sizes {batch_sizes} for the mapped dimension"
        )
    return batch_sizes[0]


def _num_outputs(batched_outputs: Tensor | tuple[Tensor, ...]) -> int:
    if isinstance(batched_outputs, tuple):
        return len(batched_outputs)
    return 1


# If value is a tuple, check it has length `num_elements`.
# If value is not a tuple, make a tuple with `value` repeated `num_elements` times


def _as_tuple(
    value: tuple[_R, ...] | _R,
    num_elements: int,
    error_message_lambda: Callable[[], str],
) -> tuple[_R, ...]:
    if not isinstance(value, tuple):
        return (value,) * num_elements
    if len(value) != num_elements:
        raise ValueError(error_message_lambda())
    return value


def _process_batched_inputs(
    in_dims: in_dims_t, args: tuple[Any, ...], func: Callable[..., Any]
) -> tuple[int, list[int | None], list[Any], TreeSpec]:
    if not isinstance(in_dims, int) and not isinstance(in_dims, tuple):
        raise ValueError(
            f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
            f"expected `in_dims` to be int or a (potentially nested) tuple "
            f"matching the structure of inputs, got: {type(in_dims)}."
        )
    if len(args) == 0:
        raise ValueError(
            f"vmap({_get_name(func)})(<inputs>): got no inputs. Maybe you forgot to add "
            f"inputs, or you are trying to vmap over a function with no inputs. "
            f"The latter is unsupported."
        )

    flat_args, args_spec = tree_flatten(args)
    flat_in_dims = _broadcast_to_and_flatten(in_dims, args_spec)
    if flat_in_dims is None:
        raise ValueError(
            f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
            f"in_dims is not compatible with the structure of `inputs`. "
            f"in_dims has structure {tree_flatten(in_dims)[1]} but inputs "
            f"has structure {args_spec}."
        )

    for i, (arg, in_dim) in enumerate(zip(flat_args, flat_in_dims)):
        if not isinstance(in_dim, int) and in_dim is not None:
            raise ValueError(
                f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
                f"Got in_dim={in_dim} for an input but in_dim must be either "
                f"an integer dimension or None."
            )
        if isinstance(in_dim, int) and not isinstance(arg, Tensor):
            raise ValueError(
                f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
                f"Got in_dim={in_dim} for an input but the input is of type "
                f"{type(arg)}. We cannot vmap over non-Tensor arguments, "
                f"please use None as the respective in_dim"
            )
        if in_dim is not None and (in_dim < -arg.dim() or in_dim >= arg.dim()):
            raise ValueError(
                f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
                f"Got in_dim={in_dim} for some input, but that input is a Tensor "
                f"of dimensionality {arg.dim()} so expected in_dim to satisfy "
                f"-{arg.dim()} <= in_dim < {arg.dim()}."
            )
        if in_dim is not None and in_dim < 0:
            flat_in_dims[i] = in_dim % arg.dim()

    return (
        _validate_and_get_batch_size(flat_in_dims, flat_args),
        flat_in_dims,
        flat_args,
        args_spec,
    )


# Creates BatchedTensors for every Tensor in arg that should be batched.
# Returns the (potentially) batched arguments and the batch_size.


# TODO: See if we can explain how flat works to the type checker
def _create_batched_inputs(
    flat_in_dims: list[int | None],
    flat_args: list[Any],
    vmap_level: int,
    args_spec: TreeSpec,
) -> tuple[Any, ...]:
    # See NOTE [Ignored _remove_batch_dim, _add_batch_dim]
    batched_inputs = [
        arg if in_dim is None else _add_batch_dim(arg, in_dim, vmap_level)
        for in_dim, arg in zip(flat_in_dims, flat_args)
    ]
    return tree_unflatten(batched_inputs, args_spec)


def _maybe_remove_batch_dim(
    name: str,
    batched_output: Any,
    vmap_level: int,
    batch_size: int,
    out_dim: int | None,
) -> torch.Tensor:
    if out_dim is None:
        if isinstance(batched_output, torch.Tensor) and is_batchedtensor(
            batched_output
        ):
            raise ValueError(
                f"vmap({name}, ...): `{name}` can not return a "
                f"BatchedTensor when out_dim is None"
            )
        return batched_output

    # out_dim is non None
    if not isinstance(batched_output, torch.Tensor):
        raise ValueError(
            f"vmap({name}, ...): `{name}` must only return "
            f"Tensors, got type {type(batched_output)}. "
            "Did you mean to set out_dims= to None for output?"
        )

    return _remove_batch_dim(batched_output, vmap_level, batch_size, out_dim)


# Undoes the batching (and any batch dimensions) associated with the `vmap_level`.
def _unwrap_batched(
    batched_outputs: Tensor | tuple[Tensor, ...],
    out_dims: out_dims_t,
    vmap_level: int,
    batch_size: int,
    func: Callable[..., Any],
) -> tuple[Any, ...]:
    flat_batched_outputs, output_spec = tree_flatten(batched_outputs)

    def incompatible_error() -> NoReturn:
        raise ValueError(
            f"vmap({_get_name(func)}, ..., out_dims={out_dims})(<inputs>): "
            f"out_dims is not compatible with the structure of `outputs`. "
            f"out_dims has structure {tree_flatten(out_dims)[1]} but outputs "
            f"has structure {output_spec}."
        )

    flat_out_dims: list[int | None] = []
    if isinstance(batched_outputs, torch.Tensor):
        # Some weird edge case requires us to spell out the following
        # see test_out_dims_edge_case
        if isinstance(out_dims, int):
            flat_out_dims = [out_dims]
        elif isinstance(out_dims, tuple) and len(out_dims) == 1:
            flat_out_dims = list(out_dims)
        elif out_dims is None:
            flat_out_dims = [out_dims]
        else:
            incompatible_error()
    else:
        broadcast_result = _broadcast_to_and_flatten(out_dims, output_spec)
        if broadcast_result is None:
            incompatible_error()
        else:
            flat_out_dims = broadcast_result

    flat_outputs = [
        _maybe_remove_batch_dim(
            _get_name(func), batched_output, vmap_level, batch_size, out_dim
        )
        for batched_output, out_dim in zip(flat_batched_outputs, flat_out_dims)
    ]
    return tree_unflatten(flat_outputs, output_spec)


def _check_int_or_none(x: Any, func: Callable[..., Any], out_dims: out_dims_t) -> None:
    if isinstance(x, int):
        return
    if x is None:
        return
    raise ValueError(
        f"vmap({_get_name(func)}, ..., out_dims={out_dims}): `out_dims` must be "
        f"an int, None or a python collection of ints representing where in the outputs the "
        f"vmapped dimension should appear."
    )


def _check_out_dims_is_int_or_int_pytree(
    out_dims: out_dims_t, func: Callable[..., Any]
) -> None:
    if isinstance(out_dims, int):
        return
    tree_map_(partial(_check_int_or_none, func=func, out_dims=out_dims), out_dims)


def _get_name(func: Callable[..., Any]) -> str:
    if hasattr(func, "__name__"):
        return func.__name__

    if isinstance(func, functools.partial):
        return f"functools.partial({_get_name(func.func)}, ...)"

    # Not all callables have __name__, in fact, only static functions/methods
    # do.  A callable created via nn.Module, to name one example, doesn't have a
    # __name__.
    return repr(func)


def vmap_impl(
    func: Callable[_P, Tensor | tuple[Tensor, ...]],
    in_dims: in_dims_t,
    out_dims: out_dims_t,
    randomness: str,
    chunk_size: int | None,
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> Any:
    lazy_load_decompositions()
    _check_out_dims_is_int_or_int_pytree(out_dims, func)
    batch_size, flat_in_dims, flat_args, args_spec = _process_batched_inputs(
        in_dims, args, func
    )

    if chunk_size is not None:
        chunks_flat_args = _get_chunked_inputs(
            flat_args, flat_in_dims, batch_size, chunk_size
        )
        return _chunked_vmap(
            func,
            flat_in_dims,
            chunks_flat_args,
            args_spec,
            out_dims,
            randomness,
            **kwargs,
        )

    # If chunk_size is not specified.
    return _flat_vmap(
        func,
        batch_size,
        flat_in_dims,
        flat_args,
        args_spec,
        out_dims,
        randomness,
        **kwargs,
    )


def get_chunk_sizes(total_elems: int, chunk_size: int) -> list[int]:
    n_chunks = total_elems // chunk_size
    chunk_sizes = [chunk_size] * n_chunks
    # remainder chunk
    remainder = total_elems % chunk_size
    if remainder != 0:
        chunk_sizes.append(remainder)
    return chunk_sizes


def _get_chunked_inputs(
    flat_args: list[Any],
    flat_in_dims: list[int | None],
    batch_size: int,
    chunk_size: int | None,
) -> Iterable[tuple[Any, ...]]:
    split_idxs = (batch_size,)
    if chunk_size is not None:
        chunk_sizes = get_chunk_sizes(batch_size, chunk_size)
        split_idxs = tuple(itertools.accumulate(chunk_sizes))

    flat_args_chunks = tuple(
        (
            t.tensor_split(split_idxs, dim=in_dim)
            if in_dim is not None
            else [
                t,
            ]
            * len(split_idxs)
        )
        for t, in_dim in zip(flat_args, flat_in_dims)
    )

    # transpose chunk dim and flatten structure
    # chunks_flat_args is a list of flatten args
    chunks_flat_args = zip(*flat_args_chunks)
    return chunks_flat_args


def _flatten_chunks_output(
    chunks_output_: list[Any],
) -> tuple[list[tuple[Any, ...]], TreeSpec]:
    # chunks_output is a list of chunked outputs
    # flatten chunked outputs:
    flat_chunks_output: list[list[Any]] = []
    arg_spec: TreeSpec | None = None
    for output in chunks_output_:
        flat_output, arg_specs = tree_flatten(output)
        flat_chunks_output.append(flat_output)
        if arg_spec is None:
            arg_spec = arg_specs

    # transpose chunk dim and flatten structure
    # flat_output_chunks is flat list of chunks
    flat_output_chunks = list(zip(*flat_chunks_output))
    if arg_spec is None:
        raise AssertionError("arg_spec must not be None")
    return flat_output_chunks, arg_spec


def _concat_chunked_outputs(
    out_dims: out_dims_t,
    arg_spec: TreeSpec,
    flat_output_chunks: list[tuple[Any, ...] | None],
) -> list[Tensor]:
    # concat chunks on out_dim
    flat_out_dims = _broadcast_to_and_flatten(out_dims, arg_spec)
    if flat_out_dims is None:
        raise AssertionError("flat_out_dims must not be None")
    if len(flat_out_dims) != len(flat_output_chunks):
        raise AssertionError(
            f"len(flat_out_dims)={len(flat_out_dims)} != len(flat_output_chunks)={len(flat_output_chunks)}"
        )
    flat_output: list[Tensor] = []
    for idx, out_dim in enumerate(flat_out_dims):
        chunk = flat_output_chunks[idx]
        if chunk is None:
            raise AssertionError(f"chunk at index {idx} must not be None")
        flat_output.append(torch.cat(chunk, dim=out_dim))
        # release tensors
        flat_output_chunks[idx] = None

    return flat_output


# Applies vmap on chunked_input and returns concatenated output over the chunks.
def _chunked_vmap(
    func: Callable[_P, Tensor | tuple[Tensor, ...]],
    flat_in_dims: list[int | None],
    chunks_flat_args: Iterable[tuple[Any, ...]],
    args_spec: TreeSpec,
    out_dims: out_dims_t,
    randomness: str,
    **kwargs: Any,
) -> Any:
    chunks_output: list[Any] = []
    rs = torch.get_rng_state() if randomness == "same" else None
    for flat_args_tuple in chunks_flat_args:
        flat_args = list(flat_args_tuple)
        batch_size = _validate_and_get_batch_size(flat_in_dims, flat_args)

        # The way we compute split the input in `_get_chunked_inputs`,
        # we may get a tensor with `0` batch-size. We skip any computation
        # in that case.
        # Eg.
        # >>> chunk_size = 1
        # >>> batch_size = 6
        # >>> t = torch.zeros(batch_size, 1)
        # >>> t.tensor_split([1, 2, 3, 4, 5, 6])
        # (tensor([[0.]]), tensor([[0.]]), tensor([[0.]]), tensor([[0.]]),
        #  tensor([[0.]]), tensor([[0.]]), tensor([], size=(0, 1)))
        if batch_size == 0:
            continue

        if rs is not None:
            torch.set_rng_state(rs)
        chunks_output.append(
            _flat_vmap(
                func,
                batch_size,
                flat_in_dims,
                flat_args,
                args_spec,
                out_dims,
                randomness,
                **kwargs,
            )
        )

    flat_output_chunks, arg_spec = _flatten_chunks_output(chunks_output)

    # chunked output tensors are held by both `flat_output_chunks` and `chunks_output`.
    # eagerly remove the reference from `chunks_output`.
    del chunks_output

    # concat chunks on out_dim
    # Note: We use cast since flat_output_chunks is modified in _concat_chunked_outputs
    # to set elements to None after processing
    flat_output = _concat_chunked_outputs(
        out_dims, arg_spec, cast(list[tuple[Any, ...] | None], flat_output_chunks)
    )

    # finally unflatten the output
    return tree_unflatten(flat_output, arg_spec)


# Vmap refactored helper functions:
def _check_randomness_arg(randomness: str) -> None:
    if randomness not in ["error", "different", "same"]:
        raise RuntimeError(
            f"Only allowed values for randomness are 'error', 'different', or 'same'. Got {randomness}"
        )


@contextlib.contextmanager
def vmap_increment_nesting(
    batch_size: int, randomness: str
) -> Generator[int, None, None]:
    try:
        vmap_level = _vmap_increment_nesting(batch_size, randomness)
        yield vmap_level
    finally:
        _vmap_decrement_nesting()


def _flat_vmap(
    func: Callable[..., Tensor | tuple[Tensor, ...]],
    batch_size: int,
    flat_in_dims: list[int | None],
    flat_args: list[Any],
    args_spec: TreeSpec,
    out_dims: out_dims_t,
    randomness: str,
    **kwargs: Any,
) -> Any:
    with vmap_increment_nesting(batch_size, randomness) as vmap_level:
        batched_inputs = _create_batched_inputs(
            flat_in_dims, flat_args, vmap_level, args_spec
        )
        batched_outputs = func(*batched_inputs, **kwargs)
        return _unwrap_batched(batched_outputs, out_dims, vmap_level, batch_size, func)


# `restore_vmap` is a private helper function. It is vmap but has the following
# differences:
# - instead of returning outputs, it returns an (outputs, out_dims) tuple.
#   out_dims is a pytree of same shape as outputs and contains Optional[int]
#   specifying where the vmapped dimension, if it exists, is in the corresponding output.
# - does no validation on in_dims or inputs (vmap expects at least one Tensor to be vmapped).
#   restore_vmap allows for no inputs to have the vmap dimension
# - does no validation on outputs (vmap expects only Tensor outputs)
#   restore_vmap allows for return of arbitrary outputs (not just Tensors)
#
# The TL;DR is that restore_vmap is more general than vmap and has a slightly
# different API. The relaxations are so that we can "pause" vmap in the middle
# of its execution and then "restore" it later (this is what we do in
# the generate_vmap_rule=True implementation of autograd.Function).
#
# restore_vmap can be technically used in the implementation of vmap, but doing
# that refactor is a bit technically challenging because:
# - vmap couples the tensor-wrapping code with error checking
# - vmap's tensor unwrapping code is in C++; we would need to rewrite part of it
#   in python because it overlaps with unwrap_batched
def restore_vmap(
    func: Callable[..., _R], in_dims: in_dims_t, batch_size: int, randomness: str
) -> Callable[..., tuple[Any, Any]]:
    def inner(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        with vmap_increment_nesting(batch_size, randomness) as vmap_level:
            batched_inputs = wrap_batched(args, in_dims, vmap_level)
            batched_outputs = func(*batched_inputs, **kwargs)
            return unwrap_batched(batched_outputs, vmap_level)

    return inner


def wrap_batched(
    args: tuple[Any, ...], bdims: in_dims_t, level: int
) -> tuple[Any, ...]:
    flat_args, spec = tree_flatten(args)
    flat_bdims = _broadcast_to_and_flatten(bdims, spec)
    if flat_bdims is None:
        raise AssertionError("flat_bdims must not be None")
    result = _create_batched_inputs(flat_bdims, flat_args, level, spec)
    return result


def unwrap_batched(args: Any, level: int) -> tuple[Any, Any]:
    flat_args, spec = tree_flatten(args)
    if len(flat_args) == 0:
        return args, ()
    result = [
        (
            torch._C._functorch._unwrap_batched(arg, level)
            if isinstance(arg, torch.Tensor)
            else (arg, None)
        )
        for arg in flat_args
    ]
    output, bdims = zip(*result)
    return tree_unflatten(output, spec), tree_unflatten(bdims, spec)
<|endoftext|># Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy
from typing import Any, NoReturn, TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils._named_member_accessor import NamedMemberAccessor


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

# Utilities to make nn.Module "functional"
# In particular the goal is to be able to provide a function that takes as input
# the parameters and evaluate the nn.Module using fixed inputs.


def raise_parameter_tying_error() -> NoReturn:
    raise RuntimeError(
        "make_functional(module): we don't yet support models that "
        "do parameter tying (also sometimes known as weight sharing). "
        "Please try to rewrite your model by replacing all instances of the "
        "tied parameter with another and/or comment your support in "
        "https://github.com/pytorch/functorch/issues/446"
    )


def create_names_map(
    named_params: dict[str, Tensor] | Iterable[tuple[str, Tensor]],
    tied_named_params: dict[str, Tensor] | Iterable[tuple[str, Tensor]],
) -> dict[str, list[str]]:
    """
    named_params is a dictionary of tensors: {'A': A, 'B': B}
    tied_named_params is another dictionary of tensors {'A': A, 'B': B, 'B_tied': B}
    with potentially tied (or 'duplicated') tensors

    This function creates a mapping from the names in named_params to the
    names in tied_named_params: {'A': ['A'], 'B': ['B', 'B_tied']}.
    """
    named_params_dict = dict(named_params)
    tied_named_params_dict = dict(tied_named_params)

    tensors_dict_keys = set(named_params_dict.keys())
    tied_tensors_dict_keys = set(tied_named_params_dict.keys())
    if not tensors_dict_keys.issubset(tied_tensors_dict_keys):
        raise AssertionError(
            f"tensors_dict_keys {tensors_dict_keys} is not a subset of "
            f"tied_tensors_dict_keys {tied_tensors_dict_keys}"
        )

    tensor_to_mapping: dict[Tensor, tuple[str, list[str]]] = {}
    for key, tensor in named_params_dict.items():
        tensor_to_mapping[tensor] = (key, [])
    for key, tensor in tied_named_params_dict.items():
        if tensor not in tensor_to_mapping:
            raise AssertionError(
                f"tensor for key '{key}' not found in tensor_to_mapping"
            )
        tensor_to_mapping[tensor][1].append(key)
    return dict(tensor_to_mapping.values())


def _extract_members(
    mod: nn.Module,
    named_members: Callable[..., Iterable[tuple[str, Tensor]]],
    subclass: Callable[[Tensor], Tensor],
) -> tuple[tuple[Tensor, ...], tuple[str, ...], dict[str, list[str]]]:
    all_named_members = tuple(named_members(remove_duplicate=False))
    unique_named_members = tuple(named_members(remove_duplicate=True))
    names_map = create_names_map(unique_named_members, all_named_members)

    # Remove all the members in the model
    # pyrefly: ignore [implicit-any]
    memo = {}
    accessor = NamedMemberAccessor(mod)
    for name, p in all_named_members:
        if p not in memo:
            memo[p] = subclass(torch.empty_like(p, device="meta"))
        replacement = memo[p]
        accessor.set_tensor(name, replacement)

    if len(unique_named_members) == 0:
        names, params = (), ()
    else:
        names, params = zip(*unique_named_members)  # type: ignore[assignment]
    return params, names, names_map


def extract_weights(
    mod: nn.Module,
) -> tuple[tuple[Tensor, ...], tuple[str, ...], dict[str, list[str]]]:
    """
    This function removes all the Parameters from the model and
    return them as a tuple as well as their original attribute names.
    The weights must be re-loaded with `load_weights` before the model
    can be used again.
    Note that this function modifies the model in place and after this
    call, mod.parameters() will be empty.
    """
    return _extract_members(mod, mod.named_parameters, nn.Parameter)


def extract_buffers(
    mod: nn.Module,
) -> tuple[tuple[Tensor, ...], tuple[str, ...], dict[str, list[str]]]:
    return _extract_members(mod, mod.named_buffers, lambda x: x)


def load_weights(
    mod: nn.Module,
    names: Sequence[str],
    params: Sequence[Tensor],
    as_params: bool = False,
) -> None:
    """
    Reload a set of weights so that `mod` can be used again to perform a forward pass.
    Note that the `params` are regular Tensors (that can have history) and so are left
    as Tensors. This means that mod.parameters() will still be empty after this call.
    """
    accessor = NamedMemberAccessor(mod)
    if as_params:
        params = [nn.Parameter(p) for p in params]
    accessor.set_tensors(names, params)


def _swap_state(
    mod: nn.Module, names_map: dict[str, list[str]], elems: Iterable[Tensor]
) -> list[Tensor]:
    result: list[Tensor] = []
    accessor = NamedMemberAccessor(mod)
    for (_, attr_names), elem in zip(names_map.items(), elems):
        for i, attr_name in enumerate(attr_names):
            if i == 0:
                result.append(accessor.swap_tensor(attr_name, elem))
            else:
                accessor.set_tensor(attr_name, elem)
    return result


def load_buffers(
    mod: nn.Module,
    names: Sequence[str],
    buffers: Sequence[Tensor],
    as_params: bool = False,
) -> None:
    accessor = NamedMemberAccessor(mod)
    accessor.set_tensors(names, buffers)


def load_state(
    model: nn.Module,
    weights: Sequence[Tensor],
    weight_names: Sequence[str],
    buffers: Sequence[Tensor] = (),
    buffer_names: Sequence[str] = (),
) -> nn.Module:
    """load_state(model, weights, weight_names, buffers=(), buffer_names=()) -> model

    load_state takes `weights` and `buffers` and assigns them to the model.
    This is the inverse operation of `make_functional_deprecated_v1`.
    """
    if len(weight_names) != len(weights):
        raise AssertionError(
            f"len(weight_names)={len(weight_names)} != len(weights)={len(weights)}"
        )
    load_weights(model, weight_names, weights)
    if len(buffers) > 0:
        if len(buffer_names) != len(buffers):
            raise AssertionError(
                f"len(buffer_names)={len(buffer_names)} != len(buffers)={len(buffers)}"
            )
        load_buffers(model, buffer_names, buffers)
    return model


def make_functional_deprecated_v1(
    model: nn.Module,
) -> tuple[tuple[Tensor, ...], Callable[..., Any], tuple[str, ...]]:
    """make_functional_deprecated_v1(model) -> weights, func, weight_names

    Given an nn.Module, make_functional_deprecated_v1 extracts the state (weights)
    and returns a functional version of the model, `func`. This makes
    it so that it is possible use transforms over the parameters of
    `model`.

    `func` can be invoked as follows:
    ```
    x = torch.randn(4, 3)
    model = nn.Linear(3, 3)
    weights, func, _ = make_functional_deprecated_v1(model)
    func(weights, (x,))
    ```

    And here is an example of applying the grad transform:
    ```
    x = torch.randn(4, 3)
    model = nn.Linear(3, 3)
    weights, _, func = make_functional_deprecated_v1(model)
    grad_weights = grad(func)(weights, (x,))
    ```

    To put the state back into a model, use `load_state`.
    """
    buffers = list(model.buffers())
    if len(buffers) > 0:
        raise RuntimeError(
            "make_functional_deprecated_v1(model): `model` has buffers. Please use "
            "make_functional_with_buffers_deprecated_v1(model) instead."
        )
    weights, descriptors, _ = extract_weights(model)

    def fun(weights: tuple[Tensor, ...], data: tuple[Any, ...]) -> Any:
        mutable_model = copy.deepcopy(model)
        load_weights(mutable_model, descriptors, weights)
        return mutable_model(*data)

    return weights, fun, descriptors


def make_functional_with_buffers_deprecated_v1(
    model: nn.Module,
) -> tuple[
    tuple[Tensor, ...],
    tuple[Tensor, ...],
    Callable[..., Any],
    tuple[str, ...],
    tuple[str, ...],
]:
    """make_functional_with_buffers_deprecated_v1(model) -> weights, buffers, func, weight_names, buffer_names

    Given an nn.Module, make_functional_with_buffers_deprecated_v1 extracts the state (weights and buffers)
    and returns a functional version of the model, `func`.

    `func` can be invoked as follows:
    ```
    x = torch.randn(4, 3)
    model = nn.Linear(3, 3)
    weights, buffers, func, _, _ = make_functional_with_buffers_deprecated_v1(model)
    func(weights, buffers, (x,))
    ```

    And here is an example of applying the grad transform:
    ```
    x = torch.randn(4, 3)
    model = nn.Linear(3, 3)
    weights, buffers, func, _, _ = make_functional_with_buffers_deprecated_v1(model)
    func(weights, buffers, (x,))
    grad_weights = grad(func)(weights, buffers, (x,))
    ```

    To put the state back into a model, use `load_state`.
    """
    weights, weight_descriptors, _ = extract_weights(model)
    buffers, buf_descriptors, _ = extract_buffers(model)

    def fun(
        weights: tuple[Tensor, ...],
        buffers: tuple[Tensor, ...],
        data: tuple[Any, ...],
    ) -> Any:
        mutable_model = copy.deepcopy(model)
        load_weights(mutable_model, weight_descriptors, weights)
        load_buffers(mutable_model, buf_descriptors, buffers)
        return mutable_model(*data)

    return weights, buffers, fun, weight_descriptors, buf_descriptors


class FunctionalModuleWithBuffers(nn.Module):
    """
    This is the callable object returned by :func:`make_functional_with_buffers`.
    """

    def __init__(
        self,
        stateless_model: nn.Module,
        param_names: tuple[str, ...],
        buffer_names: tuple[str, ...],
        param_names_map: dict[str, list[str]],
        buffer_names_map: dict[str, list[str]],
    ) -> None:
        super().__init__()
        self.stateless_model = stateless_model
        self.param_names = param_names
        self.buffer_names = buffer_names

        self.all_names_map = dict(param_names_map)
        self.all_names_map.update(buffer_names_map)

    @staticmethod
    def _create_from(
        model: nn.Module, disable_autograd_tracking: bool = False
    ) -> tuple[FunctionalModuleWithBuffers, tuple[Tensor, ...], tuple[Tensor, ...]]:
        # TODO: We don't need to copy the model to create a stateless copy
        model_copy = copy.deepcopy(model)
        params, param_names, param_names_map = extract_weights(model_copy)
        buffers, buffer_names, buffer_names_map = extract_buffers(model_copy)
        if disable_autograd_tracking:
            for param in params:
                param.requires_grad_(False)
        return (
            FunctionalModuleWithBuffers(
                model_copy, param_names, buffer_names, param_names_map, buffer_names_map
            ),
            params,
            buffers,
        )

    def forward(
        self,
        params: Iterable[Tensor],
        buffers: Iterable[Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # Temporarily load the state back onto self.stateless_model
        old_state = _swap_state(
            self.stateless_model,
            self.all_names_map,
            tuple(params) + tuple(buffers),
        )
        try:
            return self.stateless_model(*args, **kwargs)
        finally:
            # Remove the loaded state on self.stateless_model
            _swap_state(self.stateless_model, self.all_names_map, old_state)


class FunctionalModule(nn.Module):
    """
    This is the callable object returned by :func:`make_functional`.
    """

    def __init__(
        self,
        stateless_model: nn.Module,
        param_names: tuple[str, ...],
        names_map: dict[str, list[str]],
    ) -> None:
        super().__init__()
        self.stateless_model = stateless_model
        self.param_names = param_names
        self.names_map = names_map

    @staticmethod
    def _create_from(
        model: nn.Module, disable_autograd_tracking: bool = False
    ) -> tuple[FunctionalModule, tuple[Tensor, ...]]:
        # TODO: We don't need to copy the model to create a stateless copy
        model_copy = copy.deepcopy(model)
        params, param_names, names_map = extract_weights(model_copy)
        if disable_autograd_tracking:
            for param in params:
                param.requires_grad_(False)
        return FunctionalModule(model_copy, param_names, names_map), params

    def forward(self, params: Iterable[Tensor], *args: Any, **kwargs: Any) -> Any:
        # Temporarily load the state back onto self.stateless_model
        old_state = _swap_state(self.stateless_model, self.names_map, params)
        try:
            return self.stateless_model(*args, **kwargs)
        finally:
            # Remove the loaded state on self.stateless_model
            _swap_state(self.stateless_model, self.names_map, old_state)


def make_functional(
    model: nn.Module, disable_autograd_tracking: bool = False
) -> tuple[FunctionalModule, tuple[Tensor, ...]]:
    """make_functional(model, disable_autograd_tracking=False) -> func, params

    Given a ``torch.nn.Module``, :func:`make_functional` extracts the state
    (params) and returns a functional version of the model, ``func``. This
    makes it so that it is possible use transforms over the parameters of
    ``model``.

    ``func`` can be invoked as follows:

    .. code-block:: python

        import torch
        import torch.nn as nn
        from functorch import make_functional

        x = torch.randn(4, 3)
        model = nn.Linear(3, 3)
        func, params = make_functional(model)
        func(params, x)

    And here is an example of applying the grad transform over the parameters
    of a model.

    .. code-block:: python

        import torch
        import torch.nn as nn
        from functorch import make_functional, grad

        x = torch.randn(4, 3)
        t = torch.randn(4, 3)
        model = nn.Linear(3, 3)
        func, params = make_functional(model)


        def compute_loss(params, x, t):
            y = func(params, x)
            return nn.functional.mse_loss(y, t)


        grad_weights = grad(compute_loss)(params, x, t)

    If the model has any buffers, please use :func:`make_functional_with_buffers` instead.

    Args:
        model (torch.nn.Module): Input model.
        disable_autograd_tracking (bool): Flag to disable gradients tracking for output parameters.
            The returned params are unrelated to the set of params from the original model. If False (default),
            the params will have ``requires_grad=True`` on them (aka they will be trackable with regular
            PyTorch autograd), matching the requires_grad-ness of the params from the original model.
            Otherwise, the returned params will have ``requires_grad=False``. Default, False.
            If you plan on using regular PyTorch autograd (e.g., if you want to call ``.backward()`` or
            ``torch.autograd.grad()``, then set ``disable_autograd_tracking=False``.
            Otherwise, if you're only planning on using functorch's gradient transforms,
            then please set ``disable_autograd_tracking=True`` to avoid unnecessarily tracking
            history with PyTorch autograd.

    """
    buffers = list(model.buffers())
    if len(buffers) > 0:
        raise RuntimeError(
            "make_functional(model): `model` has buffers. Please use "
            "make_functional_with_buffers(model) instead."
        )
    return FunctionalModule._create_from(
        model, disable_autograd_tracking=disable_autograd_tracking
    )


def make_functional_with_buffers(
    model: nn.Module, disable_autograd_tracking: bool = False
) -> tuple[FunctionalModuleWithBuffers, tuple[Tensor, ...], tuple[Tensor, ...]]:
    """make_functional_with_buffers(model, disable_autograd_tracking=False) -> func, params, buffers

    Given a ``torch.nn.Module``, make_functional_with_buffers extracts the
    state (params and buffers) and returns a functional version of the model
    ``func`` that can be invoked like a function.

    ``func`` can be invoked as follows:

    .. code-block:: python

        import torch
        import torch.nn as nn
        from functorch import make_functional_with_buffers

        x = torch.randn(4, 3)
        model = nn.Linear(3, 3)
        func, params, buffers = make_functional_with_buffers(model)
        func(params, buffers, x)

    And here is an example of applying the grad transform over the parameters
    of a model:

    .. code-block:: python

        import torch
        import torch.nn as nn
        from functorch import make_functional_with_buffers, grad

        x = torch.randn(4, 3)
        t = torch.randn(4, 3)
        model = nn.Linear(3, 3)
        func, params, buffers = make_functional_with_buffers(model)


        def compute_loss(params, buffers, x, t):
            y = func(params, buffers, x)
            return nn.functional.mse_loss(y, t)


        grad_weights = grad(compute_loss)(params, buffers, x, t)

    Args:
        model (torch.nn.Module): Input model.
        disable_autograd_tracking (bool): Flag to disable gradients tracking for output parameters.
            The returned params are unrelated to the set of params from the original model. If False (default),
            the params will have ``requires_grad=True`` on them (aka they will be trackable with regular
            PyTorch autograd), matching the requires_grad-ness of the params from the original model.
            Otherwise, the returned params will have ``requires_grad=False``. Default, False.
            If you plan on using regular PyTorch autograd (e.g., if you want to call ``.backward()`` or
            ``torch.autograd.grad()``, then set ``disable_autograd_tracking=False``.
            Otherwise, if you're only planning on using functorch's gradient transforms,
            then please set ``disable_autograd_tracking=True`` to avoid unnecessarily tracking
            history with PyTorch autograd.

    """
    return FunctionalModuleWithBuffers._create_from(
        model, disable_autograd_tracking=disable_autograd_tracking
    )


def transpose_stack(
    tuple_of_tuple_of_tensors: tuple[tuple[Tensor, ...], ...],
) -> tuple[Tensor, ...]:
    tuple_of_tuple_of_tensors = tuple(zip(*tuple_of_tuple_of_tensors))
    results = tuple(
        torch.stack(shards).detach() for shards in tuple_of_tuple_of_tensors
    )
    return results


def combine_state_for_ensemble(
    models: Sequence[nn.Module],
) -> tuple[FunctionalModuleWithBuffers, tuple[Tensor, ...], tuple[Tensor, ...]]:
    """combine_state_for_ensemble(models) -> func, params, buffers

    Prepares a list of torch.nn.Modules for ensembling with :func:`vmap`.

    Given a list of ``M`` ``nn.Modules`` of the same class, stacks all of their
    parameters and buffers together to make ``params`` and ``buffers``.
    Each parameter and buffer in the result will have an additional dimension
    of size ``M``.

    :func:`combine_state_for_ensemble` also returns ``func``, a functional
    version of one of the models in :attr:`models`. One cannot directly run
    ``func(params, buffers, *args, **kwargs)`` directly, you probably want to
    use ``vmap(func, ...)(params, buffers, *args, **kwargs)``

    Here's an example of how to ensemble over a very simple model:

    .. code-block:: python

        num_models = 5
        batch_size = 64
        in_features, out_features = 3, 3
        models = [torch.nn.Linear(in_features, out_features) for i in range(num_models)]
        data = torch.randn(batch_size, 3)

        fmodel, params, buffers = combine_state_for_ensemble(models)
        output = vmap(fmodel, (0, 0, None))(params, buffers, data)

        assert output.shape == (num_models, batch_size, out_features)

    .. warning::
        All of the modules being stacked together must be the same (except for
        the values of their parameters/buffers). For example, they should be in the
        same mode (training vs eval).

        This API is subject to change -- we're investigating better ways to
        create ensembles and would love your feedback how to improve this.
    """
    if len(models) == 0:
        raise RuntimeError(
            "combine_state_for_ensemble: Expected at least one model, got 0."
        )
    if not (all(m.training for m in models) or all(not m.training for m in models)):
        raise RuntimeError(
            "combine_state_for_ensemble: Expected all models to "
            "have the same training/eval mode."
        )
    model0_typ = type(models[0])
    if not all(type(m) is model0_typ for m in models):
        raise RuntimeError(
            "combine_state_for_ensemble: Expected all models to be of the same class."
        )
    funcs, params, buffers = zip(
        *[make_functional_with_buffers(model) for model in models]
    )
    params = transpose_stack(params)
    buffers = transpose_stack(buffers)
    return funcs[0], params, buffers


def functional_init(
    model_class: type[nn.Module],
    ensemble_shape: tuple[()] | tuple[int, ...] = (),
    device: torch.types.Device = "cpu",
) -> Callable[..., tuple[tuple[Tensor, ...], Callable[..., Any], tuple[str, ...]]]:
    def wrapped(
        *args: Any, **kwargs: Any
    ) -> tuple[tuple[Tensor, ...], Callable[..., Any], tuple[str, ...]]:
        if len(ensemble_shape) >= 2:
            raise ValueError("NYI: ensemble_shape with more than 1 element")
        if len(ensemble_shape) == 0:
            model = model_class(*args, **kwargs).to(device)
            return make_functional_deprecated_v1(model)
        num_models = ensemble_shape[0]  # type: ignore[misc]
        if num_models <= 0:
            raise ValueError(f"num_models {num_models} should be > 0")
        # NB: Not very efficient, more of a POC
        models = tuple(
            model_class(*args, **kwargs).to(device) for _ in range(num_models)
        )
        _, fn, names = make_functional_deprecated_v1(model_class(*args, **kwargs))
        weights = tuple(make_functional_deprecated_v1(model)[0] for model in models)
        weights = tuple(zip(*weights))
        weights = tuple(torch.stack(shards).detach() for shards in weights)
        return weights, fn, names

    return wrapped


def functional_init_with_buffers(
    model_class: type[nn.Module],
    ensemble_shape: tuple[()] | tuple[int, ...] = (),
    device: torch.types.Device = "cpu",
) -> Callable[
    ...,
    tuple[
        tuple[Tensor, ...],
        tuple[Tensor, ...],
        Callable[..., Any],
        tuple[str, ...],
        tuple[str, ...],
    ]
    | tuple[tuple[Tensor, ...], Callable[..., Any], tuple[str, ...]],
]:
    def wrapped(
        *args: Any, **kwargs: Any
    ) -> (
        tuple[
            tuple[Tensor, ...],
            tuple[Tensor, ...],
            Callable[..., Any],
            tuple[str, ...],
            tuple[str, ...],
        ]
        | tuple[tuple[Tensor, ...], Callable[..., Any], tuple[str, ...]]
    ):
        if len(ensemble_shape) >= 2:
            raise ValueError("NYI: ensemble_shape with more than 1 element")
        if len(ensemble_shape) == 0:
            model = model_class(*args, **kwargs).to(device)
            return make_functional_deprecated_v1(model)
        num_models = ensemble_shape[0]  # type: ignore[misc]
        if num_models <= 0:
            raise ValueError(f"num_models {num_models} should be > 0")
        # NB: Not very efficient, more of a POC
        models = tuple(
            model_class(*args, **kwargs).to(device) for _ in range(num_models)
        )
        (
            _,
            _,
            fn,
            weight_names,
            buffer_names,
        ) = make_functional_with_buffers_deprecated_v1(model_class(*args, **kwargs))
        weights, buffers = zip(
            *tuple(
                make_functional_with_buffers_deprecated_v1(model)[:2]
                for model in models
            )
        )
        weights = tuple(zip(*weights))
        weights = tuple(torch.stack(shards).detach() for shards in weights)
        buffers = tuple(zip(*buffers))
        buffers = tuple(torch.stack(shards).detach() for shards in buffers)
        return weights, buffers, fn, weight_names, buffer_names

    return wrapped
<|endoftext|># Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextlib
from functools import partial
from typing import Any, overload, TYPE_CHECKING
from typing_extensions import ParamSpec, TypeVar

import torch
import torch.autograd.forward_ad as fwAD
from torch._C._functorch import (
    _assert_wrapped_functional,  # type: ignore[attr-defined]
    _func_decrement_nesting,  # type: ignore[attr-defined]
    _func_increment_nesting,  # type: ignore[attr-defined]
    _grad_decrement_nesting,
    _grad_increment_nesting,
    _propagate_functional_input_mutation,  # type: ignore[attr-defined]
    _unwrap_functional_tensor,
    _wrap_for_grad,
    _wrap_functional_tensor,
    get_inplace_requires_grad_allowed,
    get_unwrapped,
    is_functorch_wrapped_tensor,
    set_inplace_requires_grad_allowed,
)
from torch._functorch.predispatch import (
    _jvp_decrement_nesting,
    _jvp_increment_nesting,
    _unwrap_for_grad,
)
from torch._functorch.utils import argnums_t, exposed_in
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.fx.experimental import const_fold
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils import _pytree as pytree
from torch.utils._pytree import (
    tree_flatten,
    tree_map,
    tree_map_,
    tree_map_only,
    tree_unflatten,
    treespec_pprint,
)

from .apis import _wraps_without_dynamo_attrs, vmap
from .vmap import doesnt_support_saved_tensors_hooks, get_chunk_sizes


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence


_P = ParamSpec("_P")
_R = TypeVar("_R")
_T = TypeVar("_T")


def lazy_dynamo_disallow(func: Callable[_P, _R]) -> Callable[_P, _R]:
    import torch._dynamo

    return torch._dynamo.disallow_in_graph(func)


@contextlib.contextmanager
def enable_inplace_requires_grad(enabled: bool) -> Generator[None, None, None]:
    prev_state = get_inplace_requires_grad_allowed()
    set_inplace_requires_grad_allowed(enabled)
    try:
        yield
    finally:
        set_inplace_requires_grad_allowed(prev_state)


def _set_tensor_requires_grad(x: torch.Tensor) -> torch.Tensor:
    # avoid graph-break on x.requires_grad_()
    # https://github.com/pytorch/pytorch/pull/110053
    return x.requires_grad_()


def _create_differentiable(inps: Any, level: int | None = None) -> Any:
    def create_differentiable(x: torch.Tensor | Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            with enable_inplace_requires_grad(True):
                return _set_tensor_requires_grad(x)
        raise ValueError(f"Thing passed to transform API must be Tensor, got {type(x)}")

    return tree_map(create_differentiable, inps)


_Tensors = torch.Tensor | tuple[torch.Tensor, ...]
_TensorsT = TypeVar("_TensorsT", bound=_Tensors)
_TensorsU = TypeVar("_TensorsU", bound=_Tensors)


def _undo_create_differentiable(inps: _TensorsT, level: int | None = None) -> _TensorsT:
    def unwrap_tensors(x: _TensorsU) -> _TensorsU:
        if isinstance(x, torch.Tensor):
            if level is None:
                raise AssertionError("level must not be None when unwrapping tensors")
            # pyrefly: ignore[bad-return]
            return _unwrap_for_grad(x, level)
        # TODO: Remove the following hack for namedtuples
        if isinstance(x, tuple):
            return tree_map(unwrap_tensors, tuple(x))

        raise RuntimeError(f"Expected tensors, got unsupported type {type(x)}")

    return tree_map(unwrap_tensors, inps)


def _is_differentiable(maybe_tensor: object) -> bool:
    if not isinstance(maybe_tensor, torch.Tensor):
        return False
    return maybe_tensor.requires_grad


def _any_differentiable(
    tensor_or_tuple_of_tensors: _Tensors,
) -> bool:
    flat_args, _ = tree_flatten(tensor_or_tuple_of_tensors)
    return any(tuple(map(_is_differentiable, flat_args)))


def _wrap_tensor_for_grad(maybe_tensor: _T, level: int) -> _T:
    if not isinstance(maybe_tensor, torch.Tensor):
        return maybe_tensor
    # pyrefly: ignore[bad-return]
    return _wrap_for_grad(maybe_tensor, level)


def _wrap_all_tensors(tensor_pytree: _T, level: int) -> _T:
    return tree_map(partial(_wrap_tensor_for_grad, level=level), tensor_pytree)


# TODO: this is more accurate - enable in 3.11
# _Ts = TypeVarTuple("_Ts")
# @overload
# def _as_tuple(val: tuple[*_Ts]) -> tuple[*_Ts]: ...
@overload
def _as_tuple(val: tuple[_R, ...]) -> tuple[_R]: ...
@overload
def _as_tuple(val: _R) -> tuple[_R]: ...
def _as_tuple(val: object) -> object:
    if isinstance(val, tuple):
        return val
    return (val,)


# Version of autograd.grad that handles outputs that don't depend on inputs


def _autograd_grad(
    outputs: Sequence[torch.Tensor],
    inputs: Sequence[torch.Tensor],
    grad_outputs: Sequence[torch.Tensor] | None = None,
    retain_graph: bool = False,
    create_graph: bool = True,
) -> tuple[torch.Tensor, ...]:
    if grad_outputs is None:
        diff_outputs = tuple(out for out in outputs if out.requires_grad)
    else:
        result = tuple(
            (out, go) for out, go in zip(outputs, grad_outputs) if out.requires_grad
        )
        if len(result) == 0:
            diff_outputs, grad_outputs = (), ()
        else:
            diff_outputs, grad_outputs = zip(*result)
    if len(diff_outputs) == 0:
        return tuple(torch.zeros_like(inp) for inp in inputs)
    with torch._dynamo.compiled_autograd._disable():
        grad_inputs = torch.autograd.grad(
            diff_outputs,
            inputs,
            grad_outputs,
            retain_graph=retain_graph,
            create_graph=create_graph,
            allow_unused=True,
        )
    grad_inputs = tuple(
        torch.zeros_like(inp) if gi is None else gi
        for gi, inp in zip(grad_inputs, inputs)
    )
    return grad_inputs


# NOTE [grad and vjp interaction with no_grad]
#
# def f(x):
#   with torch.no_grad():
#     c = x ** 2
#   return x - c
#
# The thing to consider is if enable_grad is on/off before grad gets called.
#
# Case 1: enable_grad is on.
# grad(f)(x)
# In this case, `grad` should respect the inner torch.no_grad.
#
# Case 2: enable_grad is off
# with torch.no_grad():
#   grad(f)(x)
# In this case, `grad` should respect the inner torch.no_grad, but not the
# outer one. This is because `grad` is a "function transform": its result
# should not depend on the result of a context manager outside of `f`.
#
# This gives us the following desired behavior:
# - (nested) grad transforms must obey torch.no_grad inside them
# - (nested) grad transforms should not obey torch.no_grad outside them
#
# To achieve this behavior, upon entering grad/vjp:
# - we save the current ("previous") is_grad_enabled (*)
# - we unconditionally enable grad.
#
# Inside DynamicLayerBackFallback, when we're temporarily popping `grad` layer
# off the stack:
# - if grad_mode is disabled, then we do nothing. (there is a torch.no_grad
#   active, all subsequent grad transforms must obey it).
# - if grad_mode is enabled, and the previous is_grad_enabled (*) is False,
#   then we temporarily restore the previous `is_grad_enabled`. This is
#   because we're crossing the boundary from a `grad` outside the
#   no_grad to a `grad` inside the no_grad.
#
# NB: vjp has some interesting behavior because the vjp's Callable[..., Any] can be called
# under a different grad_mode than the forward computation...
#
# NB: forward-mode AD: forward-mode AD doesn't respect torch.no_grad, but
# it respects c10::AutoFwGradMode. We've implemented the same logic for
# our jvp transform (it will have special handling if FwGradMode is disabled).


# How do we increment and decrement the nesting? I don't think we can.
@exposed_in("torch.func")
def vjp(
    func: Callable[..., Any], *primals: Any, has_aux: bool = False
) -> tuple[Any, Callable[..., Any]] | tuple[Any, Callable[..., Any], Any]:
    """
    Standing for the vector-Jacobian product, returns a tuple containing the
    results of ``func`` applied to ``primals`` and a function that, when
    given ``cotangents``, computes the reverse-mode Jacobian of ``func`` with
    respect to ``primals`` times ``cotangents``.

    Args:
        func (Callable[..., Any]): A Python function that takes one or more arguments. Must
            return one or more Tensors.
        primals (Tensors): Positional arguments to ``func`` that must all be
            Tensors. The returned function will also be computing the
            derivative with respect to these arguments
        has_aux (bool): Flag indicating that ``func`` returns a
            ``(output, aux)`` tuple where the first element is the output of
            the function to be differentiated and the second element is
            other auxiliary objects that will not be differentiated.
            Default: False.

    Returns:
        Returns a ``(output, vjp_fn)`` tuple containing the output of ``func``
        applied to ``primals`` and a function that computes the vjp of
        ``func`` with respect to all ``primals`` using the cotangents passed
        to the returned function. If ``has_aux is True``, then instead returns a
        ``(output, vjp_fn, aux)`` tuple.
        The returned ``vjp_fn`` function will return a tuple of each VJP.

    When used in simple cases, :func:`vjp` behaves the same as :func:`grad`

        >>> x = torch.randn([5])
        >>> f = lambda x: x.sin().sum()
        >>> (_, vjpfunc) = torch.func.vjp(f, x)
        >>> grad = vjpfunc(torch.tensor(1.0))[0]
        >>> assert torch.allclose(grad, torch.func.grad(f)(x))

    However, :func:`vjp` can support functions with multiple outputs by
    passing in the cotangents for each of the outputs

        >>> x = torch.randn([5])
        >>> f = lambda x: (x.sin(), x.cos())
        >>> (_, vjpfunc) = torch.func.vjp(f, x)
        >>> vjps = vjpfunc((torch.ones([5]), torch.ones([5])))
        >>> assert torch.allclose(vjps[0], x.cos() + -x.sin())

    :func:`vjp` can even support outputs being Python structs

        >>> x = torch.randn([5])
        >>> f = lambda x: {"first": x.sin(), "second": x.cos()}
        >>> (_, vjpfunc) = torch.func.vjp(f, x)
        >>> cotangents = {"first": torch.ones([5]), "second": torch.ones([5])}
        >>> vjps = vjpfunc(cotangents)
        >>> assert torch.allclose(vjps[0], x.cos() + -x.sin())

    The function returned by :func:`vjp` will compute the partials with
    respect to each of the ``primals``

        >>> x, y = torch.randn([5, 4]), torch.randn([4, 5])
        >>> (_, vjpfunc) = torch.func.vjp(torch.matmul, x, y)
        >>> cotangents = torch.randn([5, 5])
        >>> vjps = vjpfunc(cotangents)
        >>> assert len(vjps) == 2
        >>> assert torch.allclose(vjps[0], torch.matmul(cotangents, y.transpose(0, 1)))
        >>> assert torch.allclose(vjps[1], torch.matmul(x.transpose(0, 1), cotangents))

    ``primals`` are the positional arguments for ``f``. All kwargs use their
    default value

        >>> x = torch.randn([5])
        >>> def f(x, scale=4.):
        >>>   return x * scale
        >>>
        >>> (_, vjpfunc) = torch.func.vjp(f, x)
        >>> vjps = vjpfunc(torch.ones_like(x))
        >>> assert torch.allclose(vjps[0], torch.full(x.shape, 4.0))

    .. note::
        Using PyTorch ``torch.no_grad`` together with ``vjp``.
        Case 1: Using ``torch.no_grad`` inside a function:

            >>> def f(x):
            >>>     with torch.no_grad():
            >>>         c = x ** 2
            >>>     return x - c

        In this case, ``vjp(f)(x)`` will respect the inner ``torch.no_grad``.

        Case 2: Using ``vjp`` inside ``torch.no_grad`` context manager:

            >>> # xdoctest: +SKIP(failing)
            >>> with torch.no_grad():
            >>>     vjp(f)(x)

        In this case, ``vjp`` will respect the inner ``torch.no_grad``, but not the
        outer one. This is because ``vjp`` is a "function transform": its result
        should not depend on the result of a context manager outside of ``f``.
    """
    return _vjp_with_argnums(func, *primals, has_aux=has_aux)


@contextlib.contextmanager
def _disable_inference_mode() -> Generator[None, None, None]:
    # Disable inference_mode without clobbering grad_mode / fw_grad_mode.
    # torch.inference_mode(False) unconditionally sets grad_mode=True and
    # fw_grad_mode=True; we save and restore those to avoid that.
    # No-op when inference_mode is already off.
    if not torch.is_inference_mode_enabled():
        yield
        return
    prev_grad = torch.is_grad_enabled()
    prev_fw_grad = torch._C._is_fwd_grad_enabled()
    with torch.inference_mode(False):
        torch._C._set_grad_enabled(prev_grad)
        torch._C._set_fwd_grad_enabled(prev_fw_grad)
        yield


@contextlib.contextmanager
def grad_increment_nesting() -> Generator[int, None, None]:
    try:
        grad_level = _grad_increment_nesting()
        yield grad_level
    finally:
        _grad_decrement_nesting()


def enter_jvp_nesting() -> int:
    global JVP_NESTING
    jvp_level = _jvp_increment_nesting()
    JVP_NESTING += 1
    return jvp_level


def exit_jvp_nesting() -> None:
    global JVP_NESTING
    _jvp_decrement_nesting()
    JVP_NESTING -= 1


@contextlib.contextmanager
def jvp_increment_nesting() -> Generator[int, None, None]:
    try:
        yield enter_jvp_nesting()
    finally:
        exit_jvp_nesting()


@doesnt_support_saved_tensors_hooks
def _vjp_with_argnums(
    func: Callable[..., Any],
    *primals: Any,
    argnums: argnums_t | None = None,
    has_aux: bool = False,
) -> tuple[Any, Callable[..., Any]] | tuple[Any, Callable[..., Any], Any]:
    # This is the same function as vjp but also accepts an argnums argument
    # All args are the same as vjp except for the added argument
    # argnums (int or tuple[int,...] | None): Optional, specifies the argument(s) to compute gradients with respect to.
    #         If None, computes the gradients with respect to all inputs (used for vjp). Default: None
    #
    # WARN: Users should NOT call this function directly and should just be calling vjp.
    # It is only separated so that inputs passed to jacrev but not differentiated get the correct wrappers.
    #
    # NOTE: All error messages are produced as if vjp was being called, even if this was called by jacrev
    #
    # Returns the same two elements as :func:`vjp` but the function returned, vjp_fn, returns a tuple of VJPs
    # for only the primal elements given by argnums.
    with grad_increment_nesting() as level:
        # See NOTE [grad and vjp interaction with no_grad]
        with torch.enable_grad():
            primals = _wrap_all_tensors(primals, level)
            if argnums is None:
                diff_primals = _create_differentiable(primals, level)
            else:
                diff_primals = _slice_argnums(primals, argnums, as_tuple=False)
                tree_map_(partial(_create_differentiable, level=level), diff_primals)
            primals_out = func(*primals)

            aux: Any = None
            if has_aux:
                if not (isinstance(primals_out, tuple) and len(primals_out) == 2):
                    raise RuntimeError(
                        "vjp(f, *primals): output of function f should be a tuple: (output, aux) "
                        "if has_aux is True"
                    )
                primals_out, aux = primals_out
                aux = _undo_create_differentiable(aux, level)

            flat_primals_out, primals_out_spec = tree_flatten(primals_out)
            assert_non_empty_tensor_output(flat_primals_out, "vjp(f, *primals)")
            flat_diff_primals, primals_spec = tree_flatten(diff_primals)
            results = _undo_create_differentiable(primals_out, level)

            for primal_out in flat_primals_out:
                if not isinstance(primal_out, torch.Tensor):
                    raise AssertionError(
                        f"expected primal_out to be a Tensor, got {type(primal_out)}"
                    )
                if primal_out.is_floating_point() or primal_out.is_complex():
                    continue
                raise RuntimeError(
                    "vjp(f, ...): All outputs of f must be "
                    "floating-point or complex Tensors, got Tensor "
                    f"with dtype {primal_out.dtype}"
                )

        def wrapper(
            cotangents: Any,
            retain_graph: bool = True,
            create_graph: bool | None = None,
        ) -> Any:
            if create_graph is None:
                create_graph = torch.is_grad_enabled()
            flat_cotangents, cotangents_spec = tree_flatten(cotangents)
            if primals_out_spec != cotangents_spec:
                raise RuntimeError(
                    f"Expected pytree structure of cotangents to be the same "
                    f"as pytree structure of outputs to the function. "
                    f"cotangents: {treespec_pprint(cotangents_spec)}, "
                    f"primal output: {treespec_pprint(primals_out_spec)}"
                )
            # This closure runs after grad_increment_nesting exits, so
            # inference_mode may have been restored. Disable it for autograd.
            # Skip under Dynamo — tracing through the generator CM emits
            # spurious _enter_inference_mode nodes.
            ctx = (
                contextlib.nullcontext()
                if torch.compiler.is_compiling()
                else _disable_inference_mode()
            )
            with ctx:
                result = _autograd_grad(
                    flat_primals_out,
                    flat_diff_primals,
                    flat_cotangents,
                    retain_graph=retain_graph,
                    create_graph=create_graph,
                )
            return tree_unflatten(result, primals_spec)

    if has_aux:
        return results, wrapper, aux  # type: ignore[possibly-unbound]
    else:
        return results, wrapper


def _safe_zero_index(x: tuple[_R, ...]) -> _R:
    if len(x) != 1:
        raise AssertionError(f"expected tuple of length 1, got length {len(x)}")
    return x[0]


# jacrev and jacfwd don't support complex functions
# Helper function to throw appropriate error.
def error_if_complex(func_name: str, args: Any, is_input: bool) -> None:
    flat_args = pytree.tree_leaves(args)
    for idx, arg in enumerate(flat_args):
        if isinstance(arg, torch.Tensor) and arg.dtype.is_complex:
            input_or_output = "inputs" if is_input else "outputs"
            err_msg = (
                f"{func_name}: Expected all {input_or_output} "
                f"to be real but received complex tensor at flattened input idx: {idx}"
            )
            raise RuntimeError(err_msg)


@exposed_in("torch.func")
def jacrev(
    func: Callable[..., Any],
    argnums: int | tuple[int, ...] = 0,
    *,
    has_aux: bool = False,
    chunk_size: int | None = None,
    _preallocate_and_copy: bool = False,
) -> Callable[..., Any]:
    """
    Computes the Jacobian of ``func`` with respect to the arg(s) at index
    ``argnum`` using reverse mode autodiff

    .. note::
        Using :attr:`chunk_size=1` is equivalent to computing the jacobian
        row-by-row with a for-loop i.e. the constraints of :func:`vmap` are
        not applicable.

    Args:
        func (function): A Python function that takes one or more arguments,
            one of which must be a Tensor, and returns one or more Tensors
        argnums (int or tuple[int, ...]): Optional, integer or tuple of integers,
            saying which arguments to get the Jacobian with respect to.
            Default: 0.
        has_aux (bool): Flag indicating that ``func`` returns a
            ``(output, aux)`` tuple where the first element is the output of
            the function to be differentiated and the second element is
            auxiliary objects that will not be differentiated.
            Default: False.
        chunk_size (None or int): If None (default), use the maximum chunk size
            (equivalent to doing a single vmap over vjp to compute the jacobian).
            If 1, then compute the jacobian row-by-row with a for-loop.
            If not None, then compute the jacobian :attr:`chunk_size` rows at a time
            (equivalent to doing multiple vmap over vjp). If you run into memory issues computing
            the jacobian, please try to specify a non-None chunk_size.

    Returns:
        Returns a function that takes in the same inputs as ``func`` and
        returns the Jacobian of ``func`` with respect to the arg(s) at
        ``argnums``. If ``has_aux is True``, then the returned function
        instead returns a ``(jacobian, aux)`` tuple where ``jacobian``
        is the Jacobian and ``aux`` is auxiliary objects returned by ``func``.

    A basic usage with a pointwise, unary operation will give a diagonal array
    as the Jacobian

        >>> from torch.func import jacrev
        >>> x = torch.randn(5)
        >>> jacobian = jacrev(torch.sin)(x)
        >>> expected = torch.diag(torch.cos(x))
        >>> assert torch.allclose(jacobian, expected)

    If you would like to compute the output of the function as well as the
    jacobian of the function, use the ``has_aux`` flag to return the output
    as an auxiliary object:

        >>> from torch.func import jacrev
        >>> x = torch.randn(5)
        >>>
        >>> def f(x):
        >>>   return x.sin()
        >>>
        >>> def g(x):
        >>>   result = f(x)
        >>>   return result, result
        >>>
        >>> jacobian_f, f_x = jacrev(g, has_aux=True)(x)
        >>> assert torch.allclose(f_x, f(x))

    :func:`jacrev` can be composed with vmap to produce batched
    Jacobians:

        >>> from torch.func import jacrev, vmap
        >>> x = torch.randn(64, 5)
        >>> jacobian = vmap(jacrev(torch.sin))(x)
        >>> assert jacobian.shape == (64, 5, 5)

    Additionally, :func:`jacrev` can be composed with itself to produce
    Hessians

        >>> from torch.func import jacrev
        >>> def f(x):
        >>>   return x.sin().sum()
        >>>
        >>> x = torch.randn(5)
        >>> hessian = jacrev(jacrev(f))(x)
        >>> assert torch.allclose(hessian, torch.diag(-x.sin()))

    By default, :func:`jacrev` computes the Jacobian with respect to the first
    input. However, it can compute the Jacobian with respect to a different
    argument by using ``argnums``:

        >>> from torch.func import jacrev
        >>> def f(x, y):
        >>>   return x + y ** 2
        >>>
        >>> x, y = torch.randn(5), torch.randn(5)
        >>> jacobian = jacrev(f, argnums=1)(x, y)
        >>> expected = torch.diag(2 * y)
        >>> assert torch.allclose(jacobian, expected)

    Additionally, passing a tuple to ``argnums`` will compute the Jacobian
    with respect to multiple arguments

        >>> from torch.func import jacrev
        >>> def f(x, y):
        >>>   return x + y ** 2
        >>>
        >>> x, y = torch.randn(5), torch.randn(5)
        >>> jacobian = jacrev(f, argnums=(0, 1))(x, y)
        >>> expectedX = torch.diag(torch.ones_like(x))
        >>> expectedY = torch.diag(2 * y)
        >>> assert torch.allclose(jacobian[0], expectedX)
        >>> assert torch.allclose(jacobian[1], expectedY)

    .. note::
        Using PyTorch ``torch.no_grad`` together with ``jacrev``.
        Case 1: Using ``torch.no_grad`` inside a function:

            >>> def f(x):
            >>>     with torch.no_grad():
            >>>         c = x ** 2
            >>>     return x - c

        In this case, ``jacrev(f)(x)`` will respect the inner ``torch.no_grad``.

        Case 2: Using ``jacrev`` inside ``torch.no_grad`` context manager:

            >>> with torch.no_grad():
            >>>     jacrev(f)(x)

        In this case, ``jacrev`` will respect the inner ``torch.no_grad``, but not the
        outer one. This is because ``jacrev`` is a "function transform": its result
        should not depend on the result of a context manager outside of ``f``.
    """
    if not (chunk_size is None or chunk_size > 0):
        raise ValueError("jacrev: `chunk_size` should be greater than 0.")

    @_wraps_without_dynamo_attrs(func)
    def wrapper_fn(*args: Any) -> Any:
        error_if_complex("jacrev", args, is_input=True)
        vjp_out = _vjp_with_argnums(func, *args, argnums=argnums, has_aux=has_aux)
        aux: Any = None
        if has_aux:
            # pyrefly: ignore[bad-unpacking]
            output, vjp_fn, aux = vjp_out
        else:
            # pyrefly: ignore[bad-unpacking]
            output, vjp_fn = vjp_out

        # See NOTE: [Computing jacobian with vmap and vjp for multiple outputs]
        flat_output, output_spec = tree_flatten(output)

        error_if_complex("jacrev", flat_output, is_input=False)

        # NB: vjp already checks that all outputs are tensors
        # Step 1: Construct grad_outputs by splitting the standard basis
        flat_output_numels = tuple(out.numel() for out in flat_output)

        primals = _slice_argnums(args, argnums)
        flat_primals, primals_spec = tree_flatten(primals)

        def compute_jacobian_stacked() -> list[Any]:
            # Helper function to compute chunked Jacobian
            # The intermediate chunked calculation are only
            # scoped at this function level.
            chunked_results: list[list[Any]] = []
            for flat_basis_chunk in _chunked_standard_basis_for_(
                flat_output, flat_output_numels, chunk_size=chunk_size
            ):
                if chunk_size == 1:
                    # sanity check.
                    for t in flat_basis_chunk:
                        if t.size(0) != 1:
                            raise AssertionError(
                                f"expected t.size(0) to be 1, got {t.size(0)}"
                            )

                    flat_basis_chunk = tree_map(
                        lambda t: torch.squeeze(t, 0), flat_basis_chunk
                    )

                basis = tree_unflatten(flat_basis_chunk, output_spec)

                if chunk_size == 1:
                    # Behaviour with `chunk_size=1` is same as `for-loop`
                    # i.e. user shouldn't deal with the limitations of vmap.
                    chunked_result = vjp_fn(basis)
                else:  # chunk_size is None or chunk_size != 1
                    chunked_result = vmap(vjp_fn)(basis)

                flat_results = pytree.tree_leaves(chunked_result)

                if chunk_size == 1:
                    flat_results = tree_map(
                        lambda t: torch.unsqueeze(t, 0), flat_results
                    )

                chunked_results.append(flat_results)

            if len(chunked_results) == 1:
                # Short-circuit if we used a single chunk
                return chunked_results[0]

            # Concatenate chunks.
            flat_results = []
            # Iterate and concat the jacobians of different
            # inputs.
            for idx in range(len(flat_primals)):
                r = tuple(r_[idx] for r_ in chunked_results)
                flat_results.append(torch.cat(r, 0))

            return flat_results

        def compute_jacobian_preallocate_and_copy() -> list[Any]:
            # Helper function to compute chunked Jacobian
            # The intermediate chunked calculation are only
            # scoped at this function level.
            out_vec_size = sum(flat_output_numels)

            # Don't pre-allocate if we have a single chunk.
            stacked_results: list[torch.Tensor] = []
            if not (chunk_size is None or chunk_size >= out_vec_size):
                stacked_results = [
                    primal.new_zeros(out_vec_size, *primal.shape)
                    for primal in flat_primals
                ]

            for idx, flat_basis_chunk in enumerate(
                _chunked_standard_basis_for_(
                    flat_output, flat_output_numels, chunk_size=chunk_size
                )
            ):
                if chunk_size == 1:
                    # sanity check.
                    for t in flat_basis_chunk:
                        if t.size(0) != 1:
                            raise AssertionError(
                                f"expected t.size(0) to be 1, got {t.size(0)}"
                            )

                    flat_basis_chunk = [torch.squeeze(t, 0) for t in flat_basis_chunk]

                basis = tree_unflatten(flat_basis_chunk, output_spec)

                if chunk_size == 1:
                    # Behaviour with `chunk_size=1` is same as `for-loop`
                    # i.e. user shouldn't deal with the limitations of vmap.
                    chunked_result = vjp_fn(basis)
                else:  # chunk_size is None or chunk_size != 1
                    chunked_result = vmap(vjp_fn)(basis)

                flat_results = pytree.tree_leaves(chunked_result)

                # Short-circuit if we have a single chunk.
                if chunk_size is None or chunk_size >= out_vec_size:
                    if chunk_size == 1:  # and out_vec_size == 1
                        # Since we squeezed the output dim
                        flat_results = tree_map(
                            lambda t: torch.unsqueeze(t, 0), flat_results
                        )
                    return flat_results

                for r, sr in zip(flat_results, stacked_results):  # type: ignore[possibly-unbound]
                    sr[idx * chunk_size : (idx + 1) * chunk_size].copy_(r)

            return stacked_results  # type: ignore[possibly-unbound]

        if _preallocate_and_copy:
            flat_jacobians_per_input = compute_jacobian_preallocate_and_copy()
        else:
            flat_jacobians_per_input = compute_jacobian_stacked()

        # Step 2: The returned jacobian is one big tensor per input. In this step,
        # we split each Tensor by output.
        flat_jacobians_per_input = [
            torch.split(result, list(flat_output_numels), dim=0)
            for result in flat_jacobians_per_input
        ]
        flat_input_flat_output = [
            tuple(
                split.view(out.shape + primal.shape)
                for split, out in zip(splits, flat_output)
            )
            for splits, primal in zip(flat_jacobians_per_input, flat_primals)
        ]

        # Step 3: Right now, `jacobian` is a List[List[Tensor]].
        # The outer List corresponds to the number of primals,
        # the inner List corresponds to the number of outputs.
        # We need to:
        # a. Exchange the order of the outer List and inner List
        # b. tree_unflatten the inner Lists (which correspond to the primals)
        # c. handle the argnums=int case
        # d. tree_unflatten the outer List (which corresponds to the outputs)
        flat_output_flat_input = tuple(zip(*flat_input_flat_output))

        flat_output_input = tuple(
            tree_unflatten(flat_input, primals_spec)
            for flat_input in flat_output_flat_input
        )

        if isinstance(argnums, int):
            flat_output_input = tuple(
                _safe_zero_index(flat_input) for flat_input in flat_output_input
            )
        output_input = tree_unflatten(flat_output_input, output_spec)
        if has_aux:
            return output_input, aux  # type: ignore[possibly-unbound]
        return output_input

    return wrapper_fn


# NOTE: [Computing jacobian with vmap and vjp for multiple outputs]
#
# Let's consider f(x) = (x**2, x.sum()) and let x = torch.randn(3).
# It turns out we can compute the jacobian of this function with a single
# call to autograd.grad by using vmap over the correct grad_outputs.
#
# Firstly, one way to compute the jacobian is to stack x**2 and x.sum()
# into a 4D vector. E.g., use g(x) = torch.stack([x**2, x.sum()])
#
# To get the first row of the jacobian, we call
# >>> autograd.grad(g(x), x, grad_outputs=torch.tensor([1, 0, 0, 0]))
# To get the 2nd row of the jacobian, we call
# >>> autograd.grad(g(x), x, grad_outputs=torch.tensor([0, 1, 0, 0]))
# and so on.
#
# Using vmap, we can vectorize all 4 of these computations into one by
# passing the standard basis for R^4 as the grad_output.
# vmap(partial(autograd.grad, g(x), x))(torch.eye(4)).
#
# Now, how do we compute the jacobian *without stacking the output*?
# We can just split the standard basis across the outputs. So to
# compute the jacobian of f(x), we'd use
# >>> autograd.grad(f(x), x, grad_outputs=_construct_standard_basis_for(...))
# The grad_outputs looks like the following:
# ( torch.tensor([[1, 0, 0],
#                 [0, 1, 0],
#                 [0, 0, 1],
#                 [0, 0, 0]]),
#   torch.tensor([[0],
#                 [0],
#                 [0],
#                 [1]]) )
#
# But we're not done yet!
# >>> vmap(partial(autograd.grad(f(x), x, grad_outputs=...)))
# returns a Tensor of shape [4, 3]. We have to remember to split the
# jacobian of shape [4, 3] into two:
# - one of shape [3, 3] for the first output
# - one of shape [   3] for the second output


def _chunked_standard_basis_for_(
    tensors: Sequence[torch.Tensor],
    tensor_numels: Sequence[int],
    chunk_size: int | None = None,
) -> Generator[tuple[torch.Tensor, ...], None, None]:
    # This function:
    # - constructs a N=sum(tensor_numels) standard basis. i.e. an NxN identity matrix.
    # - Splits the identity matrix into chunks with each chunk size determined by `tensor_numels`.
    # - Each chunk corresponds to one tensor. The chunk has the same dtype and
    #   device as the tensor
    #
    # For example, with tensor_numels = [1, 2, 1], this function returns:
    # ( tensor([[1],     tensor([[0, 0],      tensor([[0],
    #           [0],             [1, 0],              [0],
    #           [0],             [0, 1],              [0],
    #           [0]])  ,         [0, 0]])  ,          [1]])  )
    #
    # Precondition: tensor_numels == tuple(tensor.numel() for tensor in tensors)
    # Precondition: tensors always has at least one element.
    #
    # See NOTE: [Computing jacobian with vmap and grad for multiple tensors]
    # for context behind this function.
    # NOTE: Argument `chunk_size` is used to generate chunked basis instead of
    #       one huge basis matrix. `chunk_size` dictates the maximum size of the
    #       basis matrix along dim=0.
    if len(tensors) != len(tensor_numels):
        raise AssertionError(
            f"len(tensors)={len(tensors)} != len(tensor_numels)={len(tensor_numels)}"
        )
    if len(tensors) == 0:
        raise AssertionError("tensors must have at least one element")
    if chunk_size is not None and chunk_size <= 0:
        raise AssertionError(f"chunk_size must be > 0 or None, got {chunk_size}")
    total_numel = sum(tensor_numels)
    if chunk_size and chunk_size < total_numel:
        chunk_numels = get_chunk_sizes(total_numel, chunk_size)
    else:  # chunk_size is None or chunk_size >= total_numel
        chunk_size = total_numel
        chunk_numels = [total_numel]

    diag_start_indices = []
    diag_start_idx = 0
    for tensor_numel in tensor_numels:
        diag_start_indices.append(diag_start_idx)
        diag_start_idx -= tensor_numel

    for chunk_idx, total_numel in enumerate(chunk_numels):
        chunks = tuple(
            tensor.new_zeros(total_numel, tensor_numel)
            for tensor, tensor_numel in zip(tensors, tensor_numels)
        )

        for chunk, diag_start_idx in zip(chunks, diag_start_indices):
            offset = int(diag_start_idx) + chunk_idx * chunk_size
            chunk.diagonal(offset).fill_(1)
        chunks = tuple(
            chunk.view(total_numel, *tensor.shape)
            for chunk, tensor in zip(chunks, tensors)
        )
        yield chunks


def _construct_standard_basis_for(
    tensors: Sequence[torch.Tensor], tensor_numels: Sequence[int]
) -> tuple[torch.Tensor, ...] | None:
    for basis in _chunked_standard_basis_for_(tensors, tensor_numels, chunk_size=None):
        return basis
    return None


def _validate_and_wrap_argnum(argnum: int, num_args: int) -> int:
    if not isinstance(argnum, int):
        raise RuntimeError(f"argnum must be int, got: {type(argnum)}")
    if argnum >= 0 and argnum < num_args:
        return argnum
    if argnum < 0 and argnum >= -num_args:
        return argnum + num_args
    raise RuntimeError(f"Got argnum={argnum}, but only {num_args} positional inputs")


def _check_unique_non_empty(argnums: argnums_t) -> None:
    if isinstance(argnums, tuple):
        if len(argnums) == 0:
            raise RuntimeError("argnums must be non-empty")
        if len(set(argnums)) != len(argnums):
            raise RuntimeError(f"argnums elements must be unique, got {argnums}")


def _replace_args(
    old_args: tuple[Any, ...], new_args: tuple[Any, ...], argnums: argnums_t
) -> tuple[Any, ...]:
    if isinstance(argnums, int):
        if len(new_args) != 1:
            raise RuntimeError(
                f"new_args should be of size 1, was of size {len(new_args)}"
            )
        return tuple(
            new_args[0] if i == argnums else old_args[i] for i in range(len(old_args))
        )
    if isinstance(argnums, tuple):
        if len(new_args) != len(argnums):
            raise RuntimeError(
                "new_args should have the same size as argnums. "
                f"Argnums size {len(argnums)}, new_args size {len(new_args)}"
            )

        argnums_tuple = argnums

        def get_right_elem(i: int) -> Any:
            return (
                new_args[argnums_tuple.index(i)] if i in argnums_tuple else old_args[i]
            )

        return tuple(get_right_elem(i) for i in range(len(old_args)))
    raise RuntimeError(f"argnums must be int or Tuple[int, ...], got: {type(argnums)}")


def _validate_and_wrap_argnums(argnums: argnums_t, num_args: int) -> argnums_t:
    if isinstance(argnums, int):
        return _validate_and_wrap_argnum(argnums, num_args)
    if isinstance(argnums, tuple):
        return tuple(_validate_and_wrap_argnum(argnum, num_args) for argnum in argnums)
    raise AssertionError("Should never get here")


def _slice_argnums(
    args: tuple[Any, ...], argnums: argnums_t, as_tuple: bool = True
) -> Any:
    if not isinstance(argnums, int) and not isinstance(argnums, tuple):
        raise RuntimeError(
            f"argnums must be int or Tuple[int, ...], got: {type(argnums)}"
        )
    argnums = _validate_and_wrap_argnums(argnums, len(args))
    _check_unique_non_empty(argnums)
    if isinstance(argnums, int):
        if as_tuple:
            return (args[argnums],)
        else:
            return args[argnums]
    return tuple(args[i] for i in argnums)


JVP_NESTING = 0


def assert_flat_tuple_of_tensors(elts: Any, api: str, argname: str) -> None:
    if not isinstance(elts, tuple):
        raise RuntimeError(
            f"{api}: Expected {argname} to be a tuple of Tensors, got {type(elts)}"
        )
    for elt in elts:
        if isinstance(elt, torch.Tensor):
            continue
        raise RuntimeError(
            f"{api}: Expected {argname} to be a tuple of Tensors, got "
            f"a tuple with an element of type {type(elt)}"
        )
    if len(elts) == 0:
        raise RuntimeError(
            f"{api}: Expected {argname} to be a non-empty tuple of Tensors."
        )


def assert_non_empty_tensor_output(output: list[Any], api: str) -> None:
    if (len(output) == 1 and output[0] is None) or len(output) < 1:
        raise RuntimeError(
            f"{api}: Expected f to be a function that has non-empty output (got output = {output})"
        )
    for o in output:
        if not isinstance(o, torch.Tensor):
            raise RuntimeError(
                f"{api}: expected f(*primals) to return only tensors"
                f", got unsupported type {type(o)}"
            )


def assert_output_is_tensor_or_tensors(output: Any, api: str) -> None:
    if isinstance(output, torch.Tensor):
        return
    if not isinstance(output, tuple):
        raise RuntimeError(
            f"{api}: Expected output of f to be a Tensor or Tensors, got {type(output)}"
        )
    if len(output) == 0:
        raise RuntimeError(
            f"{api}: Expected output of f to be a non-empty tuple of Tensors."
        )
    for out in output:
        if isinstance(out, torch.Tensor):
            continue
        raise RuntimeError(
            f"{api}: Expected output of f to be a Tensor or Tensors, got "
            f"{type(out)} as an output"
        )


def assert_non_empty_list_of_tensors(
    output: list[torch.Tensor], api: str, argname: str
) -> None:
    if len(output) == 0:
        raise RuntimeError(f"{api}: Expected {argname} to contain at least one Tensor.")
    for out in output:
        if isinstance(out, torch.Tensor):
            continue
        raise RuntimeError(
            f"{api}: Expected {argname} to only contain Tensors, got {type(out)}"
        )


jvp_str = "jvp(f, primals, tangents)"


def safe_unpack_dual(
    dual: torch.Tensor, strict: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(dual, torch.Tensor):
        raise RuntimeError(
            f"{jvp_str}: expected f(*args) to return only tensors"
            f", got unsupported type {type(dual)}"
        )

    primal, tangent = fwAD.unpack_dual(dual)
    if tangent is None:
        if strict:
            raise RuntimeError(
                "jvp(f, primals, tangents, strict=True): "
                "The output of f is independent of "
                "the inputs. This is not allowed with strict=True."
            )
        tangent = torch.zeros_like(primal)
    return primal, tangent


@exposed_in("torch.func")
def jvp(
    func: Callable[..., Any],
    primals: Any,
    tangents: Any,
    *,
    strict: bool = False,
    has_aux: bool = False,
) -> tuple[Any, Any] | tuple[Any, Any, Any]:
    """
    Standing for the Jacobian-vector product, returns a tuple containing
    the output of `func(*primals)` and the "Jacobian of ``func`` evaluated at
    ``primals``" times ``tangents``. This is also known as forward-mode autodiff.

    Args:
        func (function): A Python function that takes one or more arguments,
            one of which must be a Tensor, and returns one or more Tensors
        primals (Tensors): Positional arguments to ``func`` that must all be
            Tensors. The returned function will also be computing the
            derivative with respect to these arguments
        tangents (Tensors): The "vector" for which Jacobian-vector-product is
            computed. Must be the same structure and sizes as the inputs to
            ``func``.
        has_aux (bool): Flag indicating that ``func`` returns a
            ``(output, aux)`` tuple where the first element is the output of
            the function to be differentiated and the second element is
            other auxiliary objects that will not be differentiated.
            Default: False.

    Returns:
        Returns a ``(output, jvp_out)`` tuple containing the output of ``func``
        evaluated at ``primals`` and the Jacobian-vector product.
        If ``has_aux is True``, then instead returns a ``(output, jvp_out, aux)`` tuple.

    .. note::
        You may see this API error out with "forward-mode AD not implemented
        for operator X". If so, please file a bug report and we will prioritize it.

    jvp is useful when you wish to compute gradients of a function R^1 -> R^N

        >>> from torch.func import jvp
        >>> x = torch.randn([])
        >>> f = lambda x: x * torch.tensor([1.0, 2.0, 3])
        >>> warnings.filterwarnings(
        ...     "ignore", message=".*torch.jit.script"
        ... )  # docs: hide
        >>> value, grad = jvp(f, (x,), (torch.tensor(1.0),))
        >>> assert torch.allclose(value, f(x))
        >>> assert torch.allclose(grad, torch.tensor([1.0, 2, 3]))

    :func:`jvp` can support functions with multiple inputs by passing in the
    tangents for each of the inputs

         >>> from torch.func import jvp
         >>> x = torch.randn(5)
         >>> y = torch.randn(5)
         >>> f = lambda x, y: (x * y)
         >>> _, output = jvp(f, (x, y), (torch.ones(5), torch.ones(5)))
         >>> assert torch.allclose(output, x + y)

    """

    return _jvp_with_argnums(
        func, primals, tangents, argnums=None, strict=strict, has_aux=has_aux
    )


def _jvp_with_argnums(
    func: Callable[..., Any],
    primals: Any,
    tangents: Any,
    argnums: argnums_t | None,
    *,
    strict: bool = False,
    has_aux: bool,
) -> tuple[Any, Any] | tuple[Any, Any, Any]:
    # This is the same function as jvp but also accepts an argnums argument
    # Most args are the same as jvp except for the added argument
    # argnums (int or tuple[int, ...]): Optional, specifies the argument(s) to compute gradients with respect to.
    #         If None, computes the gradients with respect to all inputs (used for jvp). Default: None
    # Because of this, tangents must be of length argnums and matches up to the corresponding primal whose index is
    # given by argnums
    #
    # WARN: Users should NOT call this function directly and should just be calling jvp.
    # It is only separated so that inputs passed to jacfwd but not differentiated get the correct wrappers.
    #
    # NOTE: All error messages are produced as if jvp was being called, even if this was called by jacfwd
    #
    # Returns the same two elements as :func:`jvp` but the returned tuple, ``jvp_out``, only has JVPs with respect to
    # the primals given by argnums
    if not isinstance(primals, tuple):
        raise RuntimeError(
            f"{jvp_str}: Expected primals to be a tuple. "
            f"E.g. it should be valid to call f(*primals)."
        )
    diff_args = primals if argnums is None else _slice_argnums(primals, argnums)
    flat_primals, primals_spec = tree_flatten(diff_args)
    flat_tangents, tangents_spec = tree_flatten(tangents)
    if primals_spec != tangents_spec:
        raise RuntimeError(
            f"{jvp_str}: Expected primals and tangents to have the same python "
            f"structure. For example, if primals is a tuple of 3 tensors, "
            f"tangents also must be. Got primals with structure {primals_spec} "
            f"and tangents with structure {tangents_spec}"
        )
    assert_non_empty_list_of_tensors(flat_primals, jvp_str, "primals")
    assert_non_empty_list_of_tensors(flat_tangents, jvp_str, "tangents")

    global JVP_NESTING

    with jvp_increment_nesting() as level:
        with fwAD._set_fwd_grad_enabled(True):
            ctx = fwAD.dual_level if JVP_NESTING == 1 else contextlib.nullcontext
            with ctx():
                flat_duals = tuple(
                    fwAD.make_dual(p, t) for p, t in zip(flat_primals, flat_tangents)
                )
                duals = tree_unflatten(flat_duals, primals_spec)
                if argnums is not None:
                    primals = _wrap_all_tensors(primals, level)
                    duals = _replace_args(primals, duals, argnums)
                result_duals = func(*duals)
                aux: Any = None
                if has_aux:
                    if not (isinstance(result_duals, tuple) and len(result_duals) == 2):
                        raise RuntimeError(
                            f"{jvp_str}: output of function f should be a tuple: (output, aux) "
                            "if has_aux is True"
                        )
                    result_duals, aux = result_duals
                    aux = _undo_create_differentiable(aux, level)

                result_duals, spec = tree_flatten(result_duals)
                assert_non_empty_tensor_output(result_duals, jvp_str)

                primals_out, tangents_out = zip(
                    *[safe_unpack_dual(dual, strict) for dual in result_duals]
                )
                primals_out = tree_map(
                    partial(_undo_create_differentiable, level=level), primals_out
                )
                tangents_out = tree_map(
                    partial(_undo_create_differentiable, level=level), tangents_out
                )

                primals_out_unflatten = tree_unflatten(primals_out, spec)
                tangents_out_unflatten = tree_unflatten(tangents_out, spec)
                if has_aux:
                    return primals_out_unflatten, tangents_out_unflatten, aux  # type: ignore[possibly-unbound]

                return primals_out_unflatten, tangents_out_unflatten


def safe_unflatten(tensor: torch.Tensor, dim: int, shape: torch.Size) -> torch.Tensor:
    if len(shape) == 0:
        if tensor.shape[dim] != 1:
            raise AssertionError(
                f"expected tensor.shape[{dim}] to be 1, got {tensor.shape[dim]}"
            )
        return tensor.squeeze(dim)
    return tensor.unflatten(dim, shape)


@exposed_in("torch.func")
def jacfwd(
    func: Callable[..., Any],
    argnums: argnums_t = 0,
    has_aux: bool = False,
    *,
    randomness: str = "error",
) -> Callable[..., Any]:
    """
    Computes the Jacobian of ``func`` with respect to the arg(s) at index
    ``argnum`` using forward-mode autodiff

    Args:
        func (function): A Python function that takes one or more arguments,
            one of which must be a Tensor, and returns one or more Tensors
        argnums (int or tuple[int, ...]): Optional, integer or tuple of integers,
            saying which arguments to get the Jacobian with respect to.
            Default: 0.
        has_aux (bool): Flag indicating that ``func`` returns a
            ``(output, aux)`` tuple where the first element is the output of
            the function to be differentiated and the second element is
            auxiliary objects that will not be differentiated.
            Default: False.
        randomness(str): Flag indicating what type of randomness to use.
            See :func:`vmap` for more detail. Allowed: "different", "same", "error".
            Default: "error"

    Returns:
        Returns a function that takes in the same inputs as ``func`` and
        returns the Jacobian of ``func`` with respect to the arg(s) at
        ``argnums``. If ``has_aux is True``, then the returned function
        instead returns a ``(jacobian, aux)`` tuple where ``jacobian``
        is the Jacobian and ``aux`` is auxiliary objects returned by ``func``.

    .. note::
        You may see this API error out with "forward-mode AD not implemented
        for operator X". If so, please file a bug report and we will prioritize it.
        An alternative is to use :func:`jacrev`, which has better operator coverage.

    A basic usage with a pointwise, unary operation will give a diagonal array
    as the Jacobian

        >>> from torch.func import jacfwd
        >>> x = torch.randn(5)
        >>> jacobian = jacfwd(torch.sin)(x)
        >>> expected = torch.diag(torch.cos(x))
        >>> assert torch.allclose(jacobian, expected)

    :func:`jacfwd` can be composed with vmap to produce batched
    Jacobians:

        >>> from torch.func import jacfwd, vmap
        >>> x = torch.randn(64, 5)
        >>> jacobian = vmap(jacfwd(torch.sin))(x)
        >>> assert jacobian.shape == (64, 5, 5)

    If you would like to compute the output of the function as well as the
    jacobian of the function, use the ``has_aux`` flag to return the output
    as an auxiliary object:

        >>> from torch.func import jacfwd
        >>> x = torch.randn(5)
        >>>
        >>> def f(x):
        >>>   return x.sin()
        >>>
        >>> def g(x):
        >>>   result = f(x)
        >>>   return result, result
        >>>
        >>> jacobian_f, f_x = jacfwd(g, has_aux=True)(x)
        >>> assert torch.allclose(f_x, f(x))

    Additionally, :func:`jacrev` can be composed with itself or :func:`jacrev`
    to produce Hessians

        >>> from torch.func import jacfwd, jacrev
        >>> def f(x):
        >>>   return x.sin().sum()
        >>>
        >>> x = torch.randn(5)
        >>> hessian = jacfwd(jacrev(f))(x)
        >>> assert torch.allclose(hessian, torch.diag(-x.sin()))

    By default, :func:`jacfwd` computes the Jacobian with respect to the first
    input. However, it can compute the Jacobian with respect to a different
    argument by using ``argnums``:

        >>> from torch.func import jacfwd
        >>> def f(x, y):
        >>>   return x + y ** 2
        >>>
        >>> x, y = torch.randn(5), torch.randn(5)
        >>> jacobian = jacfwd(f, argnums=1)(x, y)
        >>> expected = torch.diag(2 * y)
        >>> assert torch.allclose(jacobian, expected)

    Additionally, passing a tuple to ``argnums`` will compute the Jacobian
    with respect to multiple arguments

        >>> from torch.func import jacfwd
        >>> def f(x, y):
        >>>   return x + y ** 2
        >>>
        >>> x, y = torch.randn(5), torch.randn(5)
        >>> jacobian = jacfwd(f, argnums=(0, 1))(x, y)
        >>> expectedX = torch.diag(torch.ones_like(x))
        >>> expectedY = torch.diag(2 * y)
        >>> assert torch.allclose(jacobian[0], expectedX)
        >>> assert torch.allclose(jacobian[1], expectedY)

    """

    @_wraps_without_dynamo_attrs(func)
    def wrapper_fn(*args: Any) -> Any:
        error_if_complex("jacfwd", args, is_input=True)
        primals = args if argnums is None else _slice_argnums(args, argnums)
        flat_primals, primals_spec = tree_flatten(primals)
        flat_primals_numels = tuple(p.numel() for p in flat_primals)
        flat_basis = _construct_standard_basis_for(flat_primals, flat_primals_numels)
        if flat_basis is None:
            raise AssertionError("flat_basis must not be None")
        basis = tree_unflatten(flat_basis, primals_spec)

        def push_jvp(basis: Any) -> Any:
            output = _jvp_with_argnums(
                func, args, basis, argnums=argnums, has_aux=has_aux
            )
            # output[0] is the output of `func(*args)`
            error_if_complex("jacfwd", output[0], is_input=False)
            if has_aux:
                _, jvp_out, aux = output  # pyrefly: ignore[bad-unpacking]
                return jvp_out, aux
            _, jvp_out = output  # pyrefly: ignore[bad-unpacking]
            return jvp_out

        results = vmap(push_jvp, randomness=randomness)(basis)
        aux: Any = None
        if has_aux:
            results, aux = results
            # aux is in the standard basis format, e.g. NxN matrix
            # We need to fetch the first element as original `func` output
            flat_aux, aux_spec = tree_flatten(aux)
            flat_aux = [value[0] for value in flat_aux]
            aux = tree_unflatten(flat_aux, aux_spec)

        jac_outs, spec = tree_flatten(results)
        # Most probably below output check can never raise an error
        # as jvp should test the output before
        # assert_non_empty_output(jac_outs, 'jacfwd(f, ...)(*args)')

        jac_outs_ins = tuple(
            tuple(
                safe_unflatten(jac_out_in, -1, primal.shape)
                for primal, jac_out_in in zip(
                    flat_primals,
                    torch.split(
                        jac_out.movedim(0, -1), list(flat_primals_numels), dim=-1
                    ),
                )
            )
            for jac_out in jac_outs
        )
        jac_outs_ins = tuple(
            tree_unflatten(jac_ins, primals_spec) for jac_ins in jac_outs_ins
        )

        if isinstance(argnums, int):
            jac_outs_ins = tuple(jac_ins[0] for jac_ins in jac_outs_ins)
        if has_aux:
            return tree_unflatten(jac_outs_ins, spec), aux  # type: ignore[possibly-unbound]
        return tree_unflatten(jac_outs_ins, spec)

    return wrapper_fn


@exposed_in("torch.func")
def hessian(func: Callable[..., Any], argnums: argnums_t = 0) -> Callable[..., Any]:
    """
    Computes the Hessian of ``func`` with respect to the arg(s) at index
    ``argnum`` via a forward-over-reverse strategy.

    The forward-over-reverse strategy (composing ``jacfwd(jacrev(func))``) is
    a good default for good performance. It is possible to compute Hessians
    through other compositions of :func:`jacfwd` and :func:`jacrev` like
    ``jacfwd(jacfwd(func))`` or ``jacrev(jacrev(func))``.

    Args:
        func (function): A Python function that takes one or more arguments,
            one of which must be a Tensor, and returns one or more Tensors
        argnums (int or tuple[int, ...]): Optional, integer or tuple of integers,
            saying which arguments to get the Hessian with respect to.
            Default: 0.

    Returns:
        Returns a function that takes in the same inputs as ``func`` and
        returns the Hessian of ``func`` with respect to the arg(s) at
        ``argnums``.

    .. note::
        You may see this API error out with "forward-mode AD not implemented
        for operator X". If so, please file a bug report and we will prioritize it.
        An alternative is to use ``jacrev(jacrev(func))``, which has better
        operator coverage.

    A basic usage with a R^N -> R^1 function gives a N x N Hessian:

        >>> from torch.func import hessian
        >>> def f(x):
        >>>   return x.sin().sum()
        >>>
        >>> x = torch.randn(5)
        >>> hess = hessian(f)(x)  # equivalent to jacfwd(jacrev(f))(x)
        >>> assert torch.allclose(hess, torch.diag(-x.sin()))

    """
    return jacfwd(jacrev(func, argnums), argnums)


@doesnt_support_saved_tensors_hooks
def grad_and_value_impl(
    func: Callable[..., Any],
    argnums: argnums_t,
    has_aux: bool,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, Any]:
    with grad_increment_nesting() as level:
        output, aux, grad_input = None, None, None
        # See NOTE [grad and vjp interaction with no_grad]
        with torch.enable_grad():
            args = _wrap_all_tensors(args, level)
            kwargs = _wrap_all_tensors(kwargs, level)
            diff_args = _slice_argnums(args, argnums, as_tuple=False)
            tree_map_(partial(_create_differentiable, level=level), diff_args)

            output = func(*args, **kwargs)
            if has_aux:
                if not (isinstance(output, tuple) and len(output) == 2):
                    raise RuntimeError(
                        "grad_and_value(f)(*args): output of function f should be a tuple: (output, aux) "
                        "if has_aux is True"
                    )
                output, aux = output

            if not isinstance(output, torch.Tensor):
                raise RuntimeError(
                    "grad_and_value(f)(*args): Expected f(*args) "
                    f"to return a Tensor, got {type(output)}"
                )
            if output.dim() != 0:
                raise RuntimeError(
                    "grad_and_value(f)(*args): Expected f(*args) "
                    "to return a scalar Tensor, got tensor with "
                    f"{output.dim()} dims. Maybe you wanted to "
                    "use the vjp or jacrev APIs instead?"
                )

            flat_diff_args, spec = tree_flatten(diff_args)

            # NB: need create_graph so that backward pass isn't run in no_grad mode
            flat_outputs = _as_tuple(output)
            flat_grad_input = _autograd_grad(
                flat_outputs, flat_diff_args, create_graph=True
            )
            grad_input = tree_unflatten(flat_grad_input, spec)

            grad_input = _undo_create_differentiable(grad_input, level)
            output = _undo_create_differentiable(output, level)
            if has_aux:
                # pyrefly: ignore[bad-specialization]
                aux = _undo_create_differentiable(aux, level)

        if has_aux:
            return grad_input, (output, aux)
        return grad_input, output


def grad_impl(
    func: Callable[..., Any],
    argnums: argnums_t,
    has_aux: bool,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    results = grad_and_value_impl(func, argnums, has_aux, args, kwargs)
    if has_aux:
        grad, (_, aux) = results
        return grad, aux
    grad, _ = results
    return grad


def _maybe_wrap_functional_tensor(
    maybe_tensor: Any, level: int, *, _python_functionalize: bool = False
) -> Any:
    if not isinstance(maybe_tensor, torch.Tensor):
        return maybe_tensor
    wrapped = _wrap_functional_tensor(maybe_tensor, level)
    _assert_wrapped_functional(maybe_tensor, wrapped)
    if _python_functionalize:
        # pyrefly: ignore[missing-argument]
        out = FunctionalTensor(wrapped)
        # pyrefly: ignore[missing-attribute]
        torch._mirror_autograd_meta_to(maybe_tensor, out)
        return out
    return wrapped


def _wrap_all_tensors_to_functional(
    tensor_pytree: Any, level: int, *, _python_functionalize: bool = False
) -> Any:
    return tree_map(
        partial(
            lambda x: _maybe_wrap_functional_tensor(
                x, level, _python_functionalize=_python_functionalize
            )
        ),
        tensor_pytree,
    )


def _maybe_unwrap_functional_tensor(maybe_tensor: Any, *, reapply_views: bool) -> Any:
    if not isinstance(maybe_tensor, torch.Tensor):
        return maybe_tensor
    if isinstance(maybe_tensor, FunctionalTensor):
        maybe_tensor = maybe_tensor.elem

    if not torch._is_functional_tensor(maybe_tensor):
        # If it's not a functional tensor, just return it.
        # This can happen if we functionalize a fn that returns a global,
        # which was never wrapped properly.
        return maybe_tensor
    # Sync any pending updates on the output tensor
    torch._sync(maybe_tensor)
    return _unwrap_functional_tensor(maybe_tensor, reapply_views)


def _unwrap_all_tensors_from_functional(
    tensor_pytree: Any, *, reapply_views: bool
) -> Any:
    return tree_map(
        lambda t: _maybe_unwrap_functional_tensor(t, reapply_views=reapply_views),
        tensor_pytree,
    )


@exposed_in("torch.func")
def functionalize(
    func: Callable[..., Any], *, remove: str = "mutations"
) -> Callable[..., Any]:
    """
    functionalize is a transform that can be used to remove (intermediate)
    mutations and aliasing from a function, while preserving the function's
    semantics.

    ``functionalize(func)`` returns a new function with the same semantics
    as ``func``, but with all intermediate mutations removed.
    Every inplace operation performed on an intermediate tensor:
    ``intermediate.foo_()``
    gets replaced by its out-of-place equivalent:
    ``intermediate_updated = intermediate.foo()``.

    functionalize is useful for shipping a pytorch program off to
    backends or compilers that aren't able to easily represent
    mutations or aliasing operators.

    Args:
        func (Callable[..., Any]): A Python function that takes one or more arguments.
        remove (str): An optional string argument, that takes on either
            the value 'mutations' or 'mutations_and_views'.
            If 'mutations' is passed in then all mutating operators
            will be replaced with their non-mutating equivalents.
            If 'mutations_and_views' is passed in, then additionally, all aliasing
            operators will be replaced with their non-aliasing equivalents.
            Default: 'mutations'.

    Returns:
        Returns a new "functionalized" function. It takes the same inputs as
        ``func``, and has the same behavior, but any mutations
        (and optionally aliasing) performed on intermediate tensors
        in the function will be removed.

    functionalize will also remove mutations (and views) that were performed on function inputs.
    However to preserve semantics, functionalize will "fix up" the mutations after
    the transform has finished running, by detecting if any tensor inputs "should have"
    been mutated, and copying the new data back to the inputs if necessary.


    Example::

        >>> # xdoctest: +SKIP
        >>> import torch
        >>> from torch.fx.experimental.proxy_tensor import make_fx
        >>> from torch.func import functionalize
        >>>
        >>> # A function that uses mutations and views, but only on intermediate tensors.
        >>> def f(a):
        ...     b = a + 1
        ...     c = b.view(-1)
        ...     c.add_(1)
        ...     return b
        ...
        >>> inpt = torch.randn(2)
        >>>
        >>> out1 = f(inpt)
        >>> out2 = functionalize(f)(inpt)
        >>>
        >>> # semantics are the same (outputs are equivalent)
        >>> print(torch.allclose(out1, out2))
        True
        >>>
        >>> f_traced = make_fx(f)(inpt)
        >>> f_no_mutations_traced = make_fx(functionalize(f))(inpt)
        >>> f_no_mutations_and_views_traced = make_fx(functionalize(f, remove='mutations_and_views'))(inpt)
        >>>
        >>> print(f_traced.code)



        def forward(self, a_1):
            add = torch.ops.aten.add(a_1, 1);  a_1 = None
            view = torch.ops.aten.view(add, [-1])
            add_ = torch.ops.aten.add_(view, 1);  view = None
            return add

        >>> print(f_no_mutations_traced.code)



        def forward(self, a_1):
            add = torch.ops.aten.add(a_1, 1);  a_1 = None
            view = torch.ops.aten.view(add, [-1]);  add = None
            add_1 = torch.ops.aten.add(view, 1);  view = None
            view_1 = torch.ops.aten.view(add_1, [2]);  add_1 = None
            return view_1

        >>> print(f_no_mutations_and_views_traced.code)



        def forward(self, a_1):
            add = torch.ops.aten.add(a_1, 1);  a_1 = None
            view_copy = torch.ops.aten.view_copy(add, [-1]);  add = None
            add_1 = torch.ops.aten.add(view_copy, 1);  view_copy = None
            view_copy_1 = torch.ops.aten.view_copy(add_1, [2]);  add_1 = None
            return view_copy_1


        >>> # A function that mutates its input tensor
        >>> def f(a):
        ...     b = a.view(-1)
        ...     b.add_(1)
        ...     return a
        ...
        >>> f_no_mutations_and_views_traced = make_fx(functionalize(f, remove='mutations_and_views'))(inpt)
        >>> #
        >>> # All mutations and views have been removed,
        >>> # but there is an extra copy_ in the graph to correctly apply the mutation to the input
        >>> # after the function has completed.
        >>> print(f_no_mutations_and_views_traced.code)



        def forward(self, a_1):
            view_copy = torch.ops.aten.view_copy(a_1, [-1])
            add = torch.ops.aten.add(view_copy, 1);  view_copy = None
            view_copy_1 = torch.ops.aten.view_copy(add, [2]);  add = None
            copy_ = torch.ops.aten.copy_(a_1, view_copy_1);  a_1 = None
            return view_copy_1


    There are a few "failure modes" for functionalize that are worth calling out:
      (1) Like other torch.func transforms, `functionalize()` doesn't work with functions
          that directly use `.backward()`. The same is true for torch.autograd.grad.
          If you want to use autograd, you can compute gradients directly
          with `functionalize(grad(f))`.
      (2) Like other torch.func transforms, `functionalize()` doesn't work with global state.
          If you call `functionalize(f)` on a function that takes views / mutations of
          non-local state, functionalization will simply no-op and pass the view/mutation
          calls directly to the backend.
          One way to work around this is to ensure that any non-local state creation
          is wrapped into a larger function, which you then call functionalize on.
      (3) `resize_()` has some limitations: functionalize will only work on programs
          that use resize_()` as long as the tensor being resized is not a view.
      (4) `as_strided()` has some limitations: functionalize will not work on
          `as_strided()` calls that result in tensors with overlapping memory.


    Finally, a helpful mental model for understanding functionalization is that
    most user pytorch programs are writing with the public torch API.
    When executed, torch operators are generally decomposed into
    our internal C++ "ATen" API.
    The logic for functionalization happens entirely at the level of ATen.
    Functionalization knows how to take every aliasing operator in ATen,
    and map it to its non-aliasing equivalent
    (e.g. ``tensor.view({-1})`` -> ``at::view_copy(tensor, {-1})``),
    and how to take every mutating operator in ATen,
    and map it to its non-mutating equivalent
    (e.g. ``tensor.add_(1)`` -> ``at::add(tensor, -1)``),
    while tracking aliases and mutations out-of-line to know when to fix things up.
    Information about which ATen operators are aliasing or mutating all comes from
    https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml.
    """
    if remove == "mutations":
        reapply_views = True
    elif remove == "mutations_and_views":
        reapply_views = False
    else:
        raise RuntimeError(
            f"functionalize(f, remove='mutations'): received invalid argument for remove={remove}."
            " Valid options are:\n"
            "     remove='mutations': all inplace and out= operators will be removed from the program, and replaced"
            " with their out-of-place equivalents.\n"
            "     remove='mutations_and_views': In addition to the above, all aliasing operators {view} will be"
            " replaced with their non-aliasing counterparts, {view}_copy.\n"
        )

    @_wraps_without_dynamo_attrs(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            func_level = _func_increment_nesting(reapply_views)
            func_args = _wrap_all_tensors_to_functional(args, func_level)
            func_kwargs = _wrap_all_tensors_to_functional(kwargs, func_level)

            flattened_unwrapped_args = pytree.arg_tree_leaves(*args)
            flattened_wrapped_args = pytree.arg_tree_leaves(*func_args)
            flattened_unwrapped_kwargs = pytree.arg_tree_leaves(**kwargs)
            flattened_wrapped_kwargs = pytree.arg_tree_leaves(**func_kwargs)

            func_outputs = func(*func_args, **func_kwargs)
            outputs = _unwrap_all_tensors_from_functional(
                func_outputs, reapply_views=reapply_views
            )

            for a in flattened_wrapped_args + flattened_wrapped_kwargs:
                if isinstance(a, torch.Tensor):
                    # Call sync_() on the inputs, to ensure that any pending mutations have been applied.
                    torch._sync(a)

            # And if any mutations were applied to the inputs, we need to propagate them back to the user.
            for unwrapped, wrapped in zip(
                flattened_unwrapped_args, flattened_wrapped_args
            ):
                if isinstance(unwrapped, torch.Tensor) and isinstance(
                    wrapped, torch.Tensor
                ):
                    _propagate_functional_input_mutation(unwrapped, wrapped)
            for unwrapped, wrapped in zip(
                flattened_unwrapped_kwargs, flattened_wrapped_kwargs
            ):
                if isinstance(unwrapped, torch.Tensor) and isinstance(
                    wrapped, torch.Tensor
                ):
                    _propagate_functional_input_mutation(unwrapped, wrapped)

            return outputs
        finally:
            _func_decrement_nesting()

    return wrapped


@exposed_in("torch.func")
def linearize(
    func: Callable[..., Any], *primals: Any
) -> tuple[Any, Callable[..., Any]]:
    """
    Returns the value of ``func`` at ``primals`` and linear approximation
    at ``primals``.

    Args:
        func (Callable[..., Any]): A Python function that takes one or more arguments.
        primals (Tensors): Positional arguments to ``func`` that must all be
            Tensors. These are the values at which the function is linearly approximated.

    Returns:
        Returns a ``(output, jvp_fn)`` tuple containing the output of ``func``
        applied to ``primals`` and a function that computes the jvp of
        ``func`` evaluated at ``primals``.

    linearize is useful if jvp is to be computed multiple times at ``primals``. However,
    to achieve this, linearize saves intermediate computation and has higher memory requirements
    than directly applying `jvp`. So, if all the ``tangents`` are known, it maybe more efficient
    to compute vmap(jvp) instead of using linearize.

    .. note::
        linearize evaluates ``func`` twice. Please file an issue for an implementation
        with a single evaluation.

    Example::

        >>> import torch
        >>> from torch.func import linearize
        >>> def fn(x):
        ...     return x.sin()
        ...
        >>> warnings.filterwarnings("ignore", message=".*Attempted to insert a get_attr Node with no underlying reference")  # docs: hide
        >>> output, jvp_fn = linearize(fn, torch.zeros(3, 3))
        >>> jvp_fn(torch.ones(3, 3))
        tensor([[1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 1.]])
        >>>

    """
    # Note: We evaluate `fn` twice.
    # Once for returning the output and other while
    # tracing the graph.
    # If this becomes a bottle-neck, we should update
    # make_fx such that it also returns the output.

    output = func(*primals)
    _, output_spec = tree_flatten(output)

    flat_primals, primals_argspec = tree_flatten(primals)

    # tangents for tracing
    flat_tangents = tuple(p.new_empty(()).expand_as(p) for p in flat_primals)

    # function to trace
    def trace_fn(flat_tangents: tuple[torch.Tensor, ...]) -> Any:
        with fwAD.dual_level():
            flat_duals = tuple(
                fwAD.make_dual(p, t) for p, t in zip(flat_primals, flat_tangents)
            )
            duals = tree_unflatten(flat_duals, primals_argspec)
            output = func(*duals)
            tangents = tree_map_only(
                torch.Tensor, lambda dual: safe_unpack_dual(dual, False)[1], output
            )

        return tangents

    jvp_graph = lazy_dynamo_disallow(make_fx)(trace_fn)(flat_tangents)
    const_folded_jvp_graph = lazy_dynamo_disallow(const_fold.split_const_subgraphs)(
        jvp_graph
    )

    # Hold only the meta-data regarding the primals.
    flat_primals_shape = tuple(p.shape for p in flat_primals)
    flat_primals_device = tuple(p.device for p in flat_primals)
    flat_primals_dtype = tuple(p.dtype for p in flat_primals)

    def forward_ad_checks(flat_tangents: Sequence[torch.Tensor]) -> None:
        for idx, t in enumerate(flat_tangents):
            if t.shape != flat_primals_shape[idx]:
                msg = (
                    f"tangent:{idx} with shape {t.shape} in flattened "
                    f"pytree doesn't match the shape {flat_primals_shape[idx]} "
                    "of the corresponding primal."
                )
                raise RuntimeError(msg)

            if t.device != flat_primals_device[idx]:
                msg = (
                    f"tangent:{idx} with device {t.device} in flattened "
                    f"pytree doesn't match the device {flat_primals_device[idx]} "
                    "of the corresponding primal."
                )
                raise RuntimeError(msg)

            if t.dtype != flat_primals_dtype[idx]:
                msg = (
                    f"tangent:{idx} with dtype {t.dtype} in flattened "
                    f"pytree doesn't match the dtype {flat_primals_dtype[idx]} "
                    "of the corresponding primal."
                )
                raise RuntimeError(msg)

    # jvp_fn : Callable[..., Any] to return
    #   It takes care of checking the argspec of tangents,
    #   calling the folded fx graph and unflattening fx graph output
    def jvp_fn(*tangents: Any) -> Any:
        flat_tangents, tangent_argspec = tree_flatten(tangents)
        if tangent_argspec != primals_argspec:
            raise RuntimeError(
                f"Expected the tangents {tangent_argspec} to have "
                f"the same argspec as the primals {primals_argspec}"
            )

        forward_ad_checks(flat_tangents)  # type: ignore[arg-type]

        flat_output = const_folded_jvp_graph(*flat_tangents)
        # const folded graph can return flat output,
        # so transform output.
        return tree_unflatten(flat_output, output_spec)

    return output, jvp_fn


@exposed_in("torch.func")
def debug_unwrap(tensor: torch.Tensor, *, recurse: bool = True) -> torch.Tensor:
    """Unwraps a functorch tensor (e.g. BatchedTensor, GradTrackingTensor) to its underlying tensor.

    This function should only be used in a debug setting (e.g. trying to print the
    value of a Tensor in a debugger). Otherwise, using the result of function
    inside of a function being transformed will lead to undefined behavior.
    """
    if not is_functorch_wrapped_tensor(tensor):
        return tensor
    result = get_unwrapped(tensor)
    if recurse:
        return debug_unwrap(result)
    return result
<|endoftext|>from __future__ import annotations

import contextlib
import dataclasses
import itertools
import time
from contextlib import nullcontext
from functools import wraps
from typing import Any, cast, Literal, TYPE_CHECKING
from typing_extensions import ParamSpec, TypeVar
from unittest.mock import patch

import torch
import torch._dynamo.logging
import torch.nn as nn
import torch.utils._pytree as pytree
import torch.utils.dlpack
from torch import Tensor
from torch._decomp.decompositions_for_rng import PhiloxStateTracker, rng_decompositions
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo import compiled_autograd
from torch._dynamo.utils import (
    CompileEventLogger,
    dynamo_timed,
    preserve_rng_state,
    set_feature_use,
)
from torch._guards import detect_fake_mode
from torch._inductor.codecache import resolve_pre_grad_pass_timing

# Runtime annotation consumers still resolve BoxedBool from module globals.
from torch._subclasses import FakeTensorMode
from torch._subclasses.fake_tensor import maybe_get_fake_mode
from torch.export._tree_utils import reorder_kwargs
from torch.fx.experimental.proxy_tensor import make_fx

from . import config
from ._aot_autograd import autograd_cache
from ._aot_autograd.autograd_cache import (
    AOTAutogradCache,
    should_use_local_autograd_cache,
    should_use_remote_autograd_cache,
)
from ._aot_autograd.collect_metadata_analysis import (
    run_functionalized_fw_and_collect_metadata,
)
from ._aot_autograd.descriptors import (
    AOTInput,
    BufferAOTInput,
    DifferentiableAOTInput,
    ParamAOTInput,
    PlainAOTInput,
)
from ._aot_autograd.frontend_utils import (
    _detect_attribute_assignment,
    _try_get_metadata_from_dynamo,
    construct_fake_mode,
    process_inputs,
)
from ._aot_autograd.functional_utils import (  # noqa: F401
    _check_if_mutation_can_be_in_graph,
    are_all_mutations_hidden_from_autograd,
    are_all_mutations_under_no_grad_or_inference_mode,
    assert_functional_graph,
    from_fun,
    gen_alias_from_base,
    has_data_mutation,
    has_metadata_mutation,
    is_fun,
    sync_functional_tensor,
    to_fun,
)
from ._aot_autograd.graph_capture_wrappers import (  # noqa: F401
    aot_dispatch_subclass,
    create_functional_call,
    create_functionalized_fn,
    create_functionalized_rng_ops_wrapper,
    create_joint,
    fn_input_mutations_to_outputs,
    fn_prepped_for_autograd,
)
from ._aot_autograd.graph_compile import (
    aot_stage1_graph_capture,
    aot_stage2_compile,
    aot_stage2_export,
)
from ._aot_autograd.input_output_analysis import (  # noqa: F401
    compute_overlapping_inputs,
    create_graph_signature,
    create_synthetic_base_metadata,
    remove_dupe_metadata,
)
from ._aot_autograd.logging_utils import (  # noqa: F401
    callback_set,
    describe_input,
    format_guard_bug_msg,
    get_aot_compilation_context,
    get_aot_graph_name,
    get_graph_being_compiled,
    graph_being_compiled,
    model_name,
    nth_graph,
    set_model_name,
    setup_stacktrace_preservation_hooks,
    track_graph_compiling,
)
from ._aot_autograd.runtime_wrappers import (  # noqa: F401
    AOTDedupeWrapper,
    AOTSyntheticBaseWrapper,
    SerializableCompiledFunction,
)
from ._aot_autograd.schemas import (  # noqa: F401
    ActInputPath,
    AOTConfig,
    AOTDispatchCompiler,
    AOTGraphCapture,
    AOTState,
    BackwardSignature,
    FakifiedFlatArgs,
    FQN,
    GraphInputName,
    GraphOutputName,
    GraphSignature,
    InputAliasInfo,
    JointWithDescriptors,
    MutationType,
    OpaqueMeta,
    OutputAliasInfo,
    OutputType,
    SerializableAOTDispatchCompiler,
    SubclassCreationMeta,
    SubclassMeta,
    TensorAlias,
    ViewAndMutationMeta,
)
from ._aot_autograd.subclass_utils import (  # noqa: F401
    requires_subclass_dispatch,
    unwrap_tensor_subclasses,
    unwrap_tensor_subclasses_with_indices_to_original,
    wrap_tensor_subclasses,
    wrap_tensor_subclasses_maybe_joint,
)
from ._aot_autograd.to_standalone_python import (  # noqa: F401
    compile_to_python,
    load_from_python,
)
from ._aot_autograd.utils import (  # noqa: F401
    _get_autocast_states,
    call_func_at_runtime_with_args,
    create_tree_flattened_fn,
    KNOWN_TYPES,
    make_boxed_compiler,
    make_boxed_func,
    maybe_to_fresh_input,
    normalize_as_list,
    partial_flatten_asdict,
    PytreeThunk,
    root_module_when_exporting_non_strict,
    simple_wraps,
    strict_zip,
)
from .partitioners import default_partition


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from torch._inductor.compile_fx import CompilerConfigExtra
    from torch._inductor.output_code import OutputCode
    from torch._inductor.utils import InputType
    from torch._ops import OpOverload
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

_P = ParamSpec("_P")
_R = TypeVar("_R")

zip = strict_zip

# This global counter increments every time we compile a graph with
# AOTAutograd.  You can use this to correlate runtime error messages
# with compile time (e.g., if you get an error at runtime saying
# compiled graph 3 failed, you can set a breakpoint at compile time
# for this graph number to investigate further at compile time.)
#
# NB: this is different from get_aot_compilation_context, which tracks
# each underlying graph that is compiled.  In contrast, AOT_COUNTER
# corresponds to top-level invocations of aot_module/aot_function;
# one counter is allocated per entire compiled block (but this block
# may involve compiling multiple subgraphs; e.g., for forwards/backwards)
AOT_COUNTER = itertools.count()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# AOT Autograd contains a pretty non-trivial amount of logic to handle edge cases around aliasing and mutation
# that are external to the graph (they show up as side effects in some way when you run the graph).
#
# Take a look at `test_aotdispatch.py TestAOTAutograd.test_input_mutation*` tests for some examples functions
# and what they're compiled graphs looks like.
# Below is a very long comment detailing several edge cases, and showing how AOT Autograd handles them.
#
# Note [AOT Autograd: input data mutations]
#
# If we compile a function that mutates inputs, then those input mutations are real side effects
# that a user expects to see after running the compiled graph.
# However, the graph that we want to send to a backend needs to be *entirely* functional.
# The way we reconcile this difference is that we remove the mutations completely from the graph that we compile
# but we update the graph to return (updated_inputs, user_outputs).
# In the epilogue that runs after the compiled graph is executed, we copy the updated inputs back to the originals.
#
# Example: original user code:
# def f(x):
#     x.mul_(2)
#     out = x.mul(3)
#     return out
#
# After AOT Autograd compiles, we end up with a:
# (a) compiled graph
# (b) autograd.Function.forward() method, that executes the compiled graph
# (c) wrapper function, that calls the autograd.Function.forward() and performs the epilogue
#
# The output of (a, b, c) are all written below.
#
# def compiled_forward_graph(x):
#     x_updated = x.mul(2)
#     out = x_updated.mul(3)
#     return x_updated, out
#
# # x_updated gets a gradient in the compiled backward
# def compiled_backward_graph(grad_x_updated, grad_out):
#     grad_x = ...
#     return grad_x
#
# def autograd.Function.forward(x):
#     x_updated, out = compiled_forward_graph(x)
#     return x_updated, out
#
# def compiled_wrapper(x):
#     x_updated, out = autograd.Function.apply(x)
#     x.copy_(x_updated)
#     return out
#
# Another important thing to note is that updated inputs (due to data mutations) *do* participate
# in the compiled backward graph! Since the compiled forward graph gets N extra outputs
# (due to updated inputs showing up as graph outputs),
# The compiled backward gets an additional N inputs.
# That way, during the x.copy_(x_updated) bit in the epilogue, gradients will flow from the updated input
# back to the original input.


# Note [AOT Autograd: input metadata mutations]
#
# For the same reason as input mutations, we also don't put input metadata mutations in the graph.
# Instead, we return the updated version of the input (a view), and mutate the input's metadata outside of the graph
#
# Example: original user code:
# def f(x):
#     x.t_()
#     out = x.mul(3)
#     return out
#
# AOT Autograd output (compiled graph, autograd.Function.forward(), wrapper function):
# def compiled_forward_graph(x):
#     x_updated = x.t()
#     out = x_updated.mul(3)
#     return x_updated, out
#
# # x_updated does *not* get a gradient in the compiled backward
# def compiled_backward_graph(grad_out):
#     grad_x = ...
#     return grad_x
#
# def autograd.Function.forward(x):
#     x_updated, out = compiled_forward_graph(x)
#     return x_updated, out
#
# def compiled_wrapper(x):
#     x_updated, out = autograd.Function.apply(x)
#     x.as_strided_(x_updated)
#     return out


# Note [AOT Autograd: outputs aliasing inputs or intermediates!]
#
# AOT Autograd needs special handling for outputs that alias graph inputs or intermediates!
# Why?
# (1) autograd.Function.forward() has a limitation, where views that returned in the forward cannot later be mutated.
# (2) views don't need to be compiled in the graph anyway - it's cheap to generate them outside of the compiled graph,
#     in an epilogue.
# For outputs that alias inputs, we do the following:
# (a) *still* return the aliased output as a graph output
# (b) In the AOT Autograd wrapper/epilogue, we don't return that aliased output. Instead, we use it to regenerate the output.
#
# For outputs that alias *intermediates*, we do the following:
# (a) Return the output in the compiled forward, **and** return it's ._base (a graph intermediates) as an output in the forward
# (b) Use (output, graph_intermediate) to regenerate the alias, and return that to the user (instead of the compiled fw output).
# You might wonder why we return the aliased output directly in the graph (and making the graph compute it),
# only to not return it and instead generate a fresh alias off of the intermediate,
# instead of (say) just storing metadata about the size/stride of the output somewhere to generate the alias. There are two reasons:
# (1) Getting the actual alias tensor allows us to use view-replay to generate the alias, instead of an as_strided() call
# (2) Inductor (and other backends) are free to change the memory format of graph outputs, if it results in better performance.
#     This can result in problems if a user later tries to .view() that output expecting it to have one set of strides,
#     when it has a different set of strides.
#     By including the view op directly in the graph, inductor takes that into account when deciding what memory format
#     the graph intermediate should be.
#
# Another important thing to note is how our traced backward() graph handles aliases.
# (this applies to outputs aliasing inputs, outputs aliasing intermediates,
#  *and* updated inputs returned in the compiled forward due to metadata-only mutations).
# Any outputs that alias (either inputs or intermediates) do NOT participate in the compiled backward graph
# It would be wasteful to include them in the compiled backward(), because we regenerate them eagerly
# at the end of the forward.
#
# Example: original user code:
# def f(x):
#     out1 = x.t()
#     intermediate = x.mul(2)
#     out2 = intermediate.view(-1)
#     return out1, out2
#
# AOT Autograd output (compiled graph, autograd.Function.forward(), wrapper function):
# def compiled_forward_graph(x):
#     out1 = x.t()
#     intermediate = x.mul(2)
#     out2 = intermediate.view(-1)
#     # the compiled graph also returns the intermediate
#     return out1, out2, intermediate
#
# # intermediate gets a gradient in the compiled backward.
# # both output aliases (out1 and out2) do not.
# def compiled_backward_graph(grad_intermediate):
#     grad_x = ...
#     return grad_x
#
# def autograd.Function.forward(x):
#     out1, out2, intermediate = compiled_forward_graph(x)
#     return out1, out2, intermediate
#
# def compiled_wrapper(x):
#     out1, out2, intermediate = autograd.Function.apply(x)
#     # regenerate out1 from the input
#     out1_regenerated = out1._view_func(x)
#     # regenerate out1 from the intermediate
#     out2_regenerated = out2._view_func(intermediate)
#     return out1_regenerated, out2_regenerated


# Note [AOT Autograd: mutations to inputs that alias other inputs]
#
# Another edge case that is (only partially) handled today is when an input is mutated, but itself aliases another input.
# AOT Autograd needs to **ensure** that functionalization knows that the two inputs are aliased to each other.
# That way, when the aliased input is accessed later in the graph, functionalization knows to "update" the alias
# given the mutation that occurred.
#
# This is handled by updating the calling convention: we create a "synthetic base" that becomes a new input
# in the compiled function, and we regenerate the original (aliased) inputs directly off of the base
# inside of the compiled function.
#
# This logic is fully encapsulated in aot_wrapper_synthetic_base()
#
# Example: original user code:
# def f(x, x_view):
#     x.mul_(2)
#     out = x * x_view
#     return out
# f(x, x.view(-1))
#
# AOT Autograd output (compiled graph, autograd.Function.forward(), wrapper function):
# def compiled_forward_graph(base)
#     x = generate_x(base)
#     x_view = generate_x_view(base)
#     x_updated = x.mul(2)
#     x_view_updated = x_updated.view(-1)
#     out = x_updated * x_view_updated
#     return x_updated, out
#
# # The calling convention change from (aliases) -> (base) happens
# # *outside* of the autograd.Function.forward().
# # That means the forward() only has 1 input (base),
# # and the backward() only has 1 output (grad_base)
# def compiled_backward_graph(grad_out):
#     grad_base = ...
#     return grad_base
#
# def autograd.Function.forward(base):
#     x_updated, out = compiled_forward_graph(base)
#     return x_updated, out
#
# # The compiled wrapper is where we create synthetic bases.
# # The info on which inputs are mutated is also tracked *before* synthetic base creation.
# def compiled_wrapper(x, x_view):
#     base = merge_view_inputs(x, x_view)
#     x_updated, out = autograd.Function.apply(base)
#     # x and x_view are aliased in eager mode, so this mutation to x will automatically affect x_view.
#     x.copy_(x_updated)
#     return out


# Note [AOT Autograd: Views to avoid tangents aliasing inputs]
#
# We view every forward output when creating out tangent tensors to handle the problematic
# case in which a subclass does extra aliasing between graph outputs/inputs in a way that
# is not visible above the subclass.
#
# Ordinarily, when constructing the joint function that we want to trace in AOTAutograd,
# we're guaranteed that the tangent tensors that we pass
# into the joint are distinct tensors from the primals. This is because when
# decide which forward outputs to create tangents for, we only create tangents
# for forward outputs that are not aliases of inputs (See Note
# [AOT Autograd: outputs aliasing inputs or intermediates!]).
#
# However, when wrapper tensor subclasses enter the picture, it is possible
# to have an output of the forward that is a subclass that is not an
# input / alias of an input, but one of its inner tensors is an alias!
# NestedTensor is an example: Performing an out-of-place pointwise op on a
# NestedTensor constructs a fresh NestedTensor that holds onto the input's
# offsets tensor directly.
#
# Having tangent tensors that are the same as the (primal) forward inputs,
# can cause problems during tracing as make_fx() will specialize on our
# duplicate inputs: If we passed in the same tensor for primals_1 and
# tangents_1 during tracing, make_fx() will happily sub out all usages of
# tangents_1 with primals_1 in the graph, which is not what we want.
#
# To work around this, we view every forward output when creating out tangent
# tensors so that tangents can never be the same as forward inputs even if
# forward inputs alias forward outputs.

# Note [Side-Effectful Tokens in AOTAutograd]
#
# We allow some side-effectful operators in
# the post-AOTAutograd (functional) graph, such as prints and torchbind operations.
# To ensure that these side-effects are compatible to future graph passes that
# assume that the graph is functional, we will thread "effect tokens" to show
# data dependence between these side-effectful operators. Practically speaking,
# effect tokens are just dummy values (torch.tensor([])). The graph would look
# like the following:
#
# def gm(self, token0, reader):
#    token1, frame = with_token(ordered_effect_op, (reader,), token0)
#    frame = frame * 2
#    token2, frame2 = with_token(ordered_effect_op, (reader,), token1)
#    frame2 = frame2 * 2
#    return token2, frame, frame2
#
# We will pass the token as an input to the graph, thread it through
# side-effectful operators using the `with_effects` high order operator, and then
# return the updated token as an output.
# So the signature of the graph input would look something like
# (*tokens, *params_buffers, *user_inputs), and the signature of the graph
# output would look something like (*tokens, *outputs).
#
# However, Inductor does not want the concept of tokens in the final generated
# code's input and output. Since changing the graph signature inside of inductor
# is difficult, after generating the forward graph, we will run a pass to
# remove the tokens from the input and generate the following graph for Inductor, where
# the tokens are created and sunk within the graph, rather than as inputs and
# outputs:
#
# def gm(self, reader):
#    token0 = torch.ops.prims._make_token()
#    token1, frame = with_token(ordered_effect_op, (reader,), token0)
#    frame = frame * 2
#    token2, frame2 = with_token(ordered_effect_op, (reader,), token1)
#    frame2 = frame2 * 2
#    sink_token = torch.ops.prims._sink_tokens([token2])
#    return frame, frame2

#
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


aot_autograd_decompositions: dict[OpOverload, Callable[..., Any]] = {}


def create_aot_state(
    stack: contextlib.ExitStack,
    flat_fn: Callable[_P, _R],
    fake_flat_args: FakifiedFlatArgs,
    flat_args_descs: list[AOTInput],
    aot_config: AOTConfig,
    fake_mode: FakeTensorMode,
    shape_env: ShapeEnv | None,
) -> AOTState:
    """
    Traces the forward and backward graphs of the attr:`flat_fn` to generate a
    joint graph. The joint graph is an Fx graph with Aten ops. Please refer to
    the tracing mechanism to understand the graph capturing details.

    The joint graph is then passed through attr:`partition_fn` to isolate the
    forward and backward portions, which are then respectively compiled via the
    provided attr:`fw_compiler` and attr:`bw_compiler`.

    The resulting compiled forward and backward graphs are then wrapped up in a
    ``torch.autograd.Function`` object.

    The calling convention here is that the first aot_config.num_params_buffers
    inputs in flat_args are parameters and buffers, and the rest are inputs.

    We use this to assume that parameters/buffer's shapes don't change.
    """

    # Old name for now to avoid messing with stats.  Also, note this is pushed
    # on the stack, so it extends BEYOND this function
    stack.enter_context(
        dynamo_timed("create_aot_dispatcher_function", log_pt2_compile_event=True)
    )

    # This is the main entry point.
    # TODO: Chillee argues that dynamo itself should pass in fake tensors to
    # the list of arguments when compiling; at the moment we do not do this

    decompositions = aot_config.decompositions or {}
    decompositions = {
        **aot_autograd_decompositions,
        **decompositions,
    }

    if config.functionalize_rng_ops:
        # Update the decompositions with functionalized random decompositions
        decompositions = {
            **rng_decompositions,
            **decompositions,
        }

    aot_config = dataclasses.replace(
        aot_config,
        decompositions=cast("dict[OpOverload, Callable[..., Any]]", decompositions),
    )

    # Check flat_args to see if they're already fake.  If so, use that fake
    # mode instead.

    python_dispatcher_mode = (
        enable_python_dispatcher() if shape_env is not None else nullcontext()
    )

    # See NOTE: [Deferring tensor pack/unpack hooks until runtime]
    # If any saved tensor hooks are active, we **don't** want to trace them.
    # Instead, we'll let them run at runtime, around the custom autograd.Function
    # that we generate in torch.compile.
    stack.enter_context(torch.autograd.set_multithreading_enabled(False))
    stack.enter_context(preserve_rng_state())
    stack.enter_context(fake_mode)
    stack.enter_context(python_dispatcher_mode)
    stack.enter_context(PhiloxStateTracker())
    stack.enter_context(
        torch._dynamo.utils._disable_saved_tensors_hooks_during_tracing()
    )

    from torch._library.fake_class_registry import FakeScriptObject, maybe_to_fake_obj
    from torch._library.opaque_object import is_custom_class

    # Tracing may mutate the states the fake script object,
    # so we need to duplicate the fake script objects so that subsequent tracing
    # won't be affected.
    def _dup_fake_script_obj(fake_flat_args: FakifiedFlatArgs) -> list[Any]:
        return [
            maybe_to_fake_obj(detect_fake_mode(fake_flat_args), arg.real_obj)
            if isinstance(arg, FakeScriptObject) or is_custom_class(type(arg))
            else arg
            for arg in fake_flat_args
        ]

    needs_autograd = any(
        x.requires_grad for x in fake_flat_args if isinstance(x, Tensor)
    )

    with enable_python_dispatcher():
        # Patch set_rng_state as set_rng_state with fake tensors is
