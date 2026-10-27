                    size_bytes=estimate_fx_collective_memory_footprint(start),
                    estimated_time_ms=coll_time_ms,
                    exposed_time_ms=coll_time_ms,  # Initially fully exposed
                )
                self.collective_info[start] = info
                self.unscheduled_collectives.add(start)
                self.all_pgs.add(get_group_name(start))

    def _calculate_domination_index(
        self, target_nodes: list[fx.Node]
    ) -> dict[fx.Node, int]:
        """
        Calculate the topological index of the earliest target node each node dominates.

        target_nodes are assigned indices based on their topological order (0, 1, 2, ...).
        For each node, returns the minimum index of target nodes it blocks/dominates.
        Returns sys.maxsize if the node doesn't block any target nodes.
        """
        target_set = OrderedSet(target_nodes)
        target_node_index: dict[fx.Node, int] = {}
        for node in self.graph.nodes:
            if node in target_set:
                target_node_index[node] = len(target_node_index)

        domination_index: dict[fx.Node, int] = {}
        for node in reversed(self.graph.nodes):
            if node in target_node_index:
                domination_index[node] = target_node_index[node]
            else:
                domination_index[node] = min(
                    (domination_index[succ] for succ in node.users), default=sys.maxsize
                )

        return domination_index

    def _align_compute_nodes_runtime_estimations_across_all_distributed_ranks(
        self,
    ) -> None:
        """Align runtime estimations across ranks (compute + collectives)."""
        log.info(
            "Overlap scheduling: Aligning runtime estimations across all distributed ranks"
        )

        # Benchmark compute nodes
        runtime_estimations_keys: list[str | None] = []
        runtime_estimations: list[float] = []
        compute_key_count = 0

        # Also collect analytical estimations for logging
        runtime_estimations_analytical: list[float] = []

        for n in self.compute_nodes:
            # Compute analytical estimation using roofline model
            val_analytical = estimate_roofline_runtime_ms(n)
            runtime_estimations_analytical.append(val_analytical)

            if self.compute_estimator == "benchmark":
                val, key = benchmark_node_with_cache_key(
                    n, self.custom_runtime_estimation
                )
            else:
                # Use analytical estimation
                val, key = val_analytical, None

            runtime_estimations.append(val)
            runtime_estimations_keys.append(key)
            compute_key_count += 1

        # Log compute estimations
        from torch._inductor.fx_passes.node_runtime_estimation import (
            _log_compute_estimations,
        )

        _log_compute_estimations(
            self.compute_nodes,
            runtime_estimations,
            runtime_estimations_analytical,
        )

        # Benchmark collectives if enabled (only CUDA events - others are deterministic)
        # Skip if custom estimation is provided for collectives
        collective_nodes: list[fx.Node] = []
        benchmarked_collective_nodes: list[
            fx.Node
        ] = []  # Track which were actually benchmarked
        if self.collective_estimator == "benchmark":
            from torch._inductor.fx_passes.node_runtime_estimation import (
                benchmark_collective_with_cuda_events,
            )

            collective_nodes = [
                info.start_node for info in self.collective_info.values()
            ]

            # Benchmark CUDA events (non-deterministic, needs alignment)
            # Skip collectives with custom estimation
            for n in collective_nodes:
                if (
                    get_custom_estimation(n, self.custom_runtime_estimation, None)
                    is not None
                ):
                    continue

                # Benchmark actual size
                cuda_val, cuda_key = benchmark_collective_with_cuda_events(n, nruns=5)
                if cuda_val is not None:
                    runtime_estimations.append(cuda_val)
                    runtime_estimations_keys.append(cuda_key)
                    benchmarked_collective_nodes.append(n)

        # When both estimators are analytical, estimates are deterministic across ranks
        # (same shapes = same estimates), so skip the all_gather to avoid sync.
        import torch.distributed as dist

        world_size = dist.get_world_size()

        if (
            self.compute_estimator == "analytical"
            and self.collective_estimator == "analytical"
        ):
            median_runtime_estimations = runtime_estimations
        else:
            # Single all_gather and compute medians
            from torch._subclasses.fake_tensor import unset_fake_temporarily
            from torch.distributed.distributed_c10d import _get_default_group

            pg = _get_default_group()
            with unset_fake_temporarily():
                gathered_runtime_estimations: list[list[float]] = [
                    [] for _ in range(world_size)
                ]
                dist.all_gather_object(
                    gathered_runtime_estimations, runtime_estimations, pg
                )
                median_runtime_estimations = torch.median(
                    torch.tensor(gathered_runtime_estimations), dim=0
                ).values.tolist()

        # Cache medians
        collective_keys = []
        collective_medians = []
        for idx, (key, median_runtime_estimation) in enumerate(
            zip(runtime_estimations_keys, median_runtime_estimations)
        ):
            if key is None:
                continue
            if idx < compute_key_count:
                # Compute node
                self.node_estimations[self.compute_nodes[idx]] = (
                    median_runtime_estimation
                )
                set_cached_node_time(key, median_runtime_estimation)
            else:
                # Collective CUDA event benchmark
                from torch._inductor.fx_passes.node_runtime_estimation import (
                    set_cached_runtime,
                )

                set_cached_runtime(key, median_runtime_estimation)

                # Update CollectiveInfo with aligned benchmark
                coll_idx = idx - compute_key_count
                coll_node = benchmarked_collective_nodes[coll_idx]
                info = self.collective_info[coll_node]
                info.estimated_time_ms = median_runtime_estimation
                info.exposed_time_ms = median_runtime_estimation
                self.node_estimations[coll_node] = median_runtime_estimation

                collective_keys.append(key)
                collective_medians.append(median_runtime_estimation)

        # Log benchmarks with analytical comparisons
        if collective_keys:
            from torch._inductor.fx_passes.node_runtime_estimation import (
                _log_collective_benchmarks,
            )

            _log_collective_benchmarks(
                benchmarked_collective_nodes,
                collective_keys,
                collective_medians,
                world_size,
                "fx_collectives_node_runtime_estimation",
            )
        else:
            # No benchmarking - log analytical estimations for all collectives
            from torch._inductor.fx_passes.node_runtime_estimation import (
                _log_collective_benchmarks,
            )

            all_collective_nodes = [
                info.start_node for info in self.collective_info.values()
            ]
            if all_collective_nodes:
                _log_collective_benchmarks(
                    all_collective_nodes,
                    artifact_name="fx_collectives_analytical_estimation",
                )

        log.info("Overlap scheduling: Runtime estimations aligned")

    def _get_next_nodes(self) -> list[fx.Node]:
        """
        Get next node(s) to schedule.

        When max_off_bucket_bytes is set, off-path collectives of the same type
        (same bucket_key) are batched together to enable bucketing them in
        overlap_preserving_bucketer. Bucket size is limited by max_off_bucket_bytes.
        """
        if self.off_path_ready:
            _, node = self.off_path_ready[0]

            should_schedule = False
            if not self.on_path_ready or node in self.scheduled:
                should_schedule = True
            elif _schedulable_wait_node(node):
                # Defer exposed waits until hidden or over memory budget
                info = self.collective_info[self.wait_to_start[node]]
                over_budget = (
                    self.memory_tracker.current_memory_bytes
                    > self.allowed_peak_memory_bytes
                )
                should_schedule = not info.is_exposed or over_budget
            elif self.max_off_bucket_bytes is not None and node in self.collective_info:
                # Batch off-path collectives: schedule when bucket threshold is reached
                bucket_key = get_full_bucket_key(node, self.bucket_mode)
                bucket_size = self.off_path_ready_potential_buckets[bucket_key]
                should_schedule = bucket_size >= self.max_off_bucket_bytes
            elif self.dominates_reduce_scatter(node):
                # Only schedule off-path nodes that dominate reduce_scatters after original position
                should_schedule = self.node_idx[node] <= self.last_on_path_node_idx

            if should_schedule:
                heapq.heappop(self.off_path_ready)

                # If batching enabled and this is a collective, gather same-type collectives
                if (
                    self.max_off_bucket_bytes is not None
                    and node in self.collective_info
                ):
                    node_key = get_full_bucket_key(node, self.bucket_mode)
                    if node_key is not None:
                        same_type_nodes = [node]
                        total_bytes = self.collective_info[node].size_bytes
                        indices_to_remove = []

                        # Scan the off_path_ready queue for same-key collectives
                        for i, (_, candidate) in enumerate(self.off_path_ready):
                            if candidate in self.scheduled:
                                continue
                            if candidate not in self.collective_info:
                                continue
                            candidate_key = get_full_bucket_key(
                                candidate, self.bucket_mode
                            )
                            if candidate_key == node_key:
                                candidate_bytes = self.collective_info[
                                    candidate
                                ].size_bytes
                                # Check bucket size limit before adding
                                if (
                                    total_bytes + candidate_bytes
                                    > self.max_off_bucket_bytes
                                ):
                                    continue  # Skip but keep looking for smaller ones
                                same_type_nodes.append(candidate)
                                total_bytes += candidate_bytes
                                indices_to_remove.append(i)

                        # Remove collected nodes from heap (reverse order to preserve indices)
                        for i in reversed(indices_to_remove):
                            self.off_path_ready.pop(i)
                        if indices_to_remove:
                            heapq.heapify(self.off_path_ready)

                        return same_type_nodes

                return [node]

        return [heapq.heappop(self.on_path_ready)[1]]

    def run(self) -> torch.fx.GraphModule:
        """Run the scheduling algorithm."""
        # All ranks must make identical decisions on overlap reordering,
        # Thus we must have identical runtime estimations across ranks.
        # For now we do benchmarking only for compute nodes.
        self._align_compute_nodes_runtime_estimations_across_all_distributed_ranks()

        while self.on_path_ready or self.off_path_ready:
            if self._should_force_wait_for_memory():
                self._force_oldest_wait()
                continue

            nodes = self._get_next_nodes()

            for node in nodes:
                # we don't always remove nodes from the heap when we schedule them
                if node in self.scheduled:
                    continue

                if node.op == "placeholder":
                    self._schedule(node)
                elif node in self.collective_info:
                    self._handle_collective_start(node)
                elif _schedulable_wait_node(node):
                    self._handle_wait(node)
                else:
                    self._handle_compute_or_other(node)

                # Track progress for off-path scheduling - only for nodes from main queue
                if not self.off_compute_path(node):
                    self.last_on_path_node_idx = max(
                        self.last_on_path_node_idx, self.node_idx[node]
                    )
                else:
                    # Decrement off-path bucket bytes when scheduling
                    if node in self.collective_info:
                        bucket_key = get_full_bucket_key(node, self.bucket_mode)
                        node_bytes = self.collective_info[node].size_bytes
                        self.off_path_ready_potential_buckets[bucket_key] -= node_bytes

        self._reorder_graph()

        # Finalize: bucket collectives (if enabled), apply deps
        from torch._inductor.fx_passes.overlap_preserving_bucketer import (
            finalize_overlap_scheduling,
        )

        finalize_overlap_scheduling(
            gm=self.gm,
            collective_info=self.collective_info,
            scheduled=self.scheduled,
            collective_bucketing=self.collective_bucketing,
            insert_overlap_deps=self.insert_overlap_deps,
            max_bucket_memory_gb=2.0,
            max_coll_distance=self.max_node_distance,
            bucket_exposed_first=self.bucket_exposed_first,
            bucket_only_internode_comms=self.bucket_only_internode_comms,
            bucket_mode=self.bucket_mode,
        )

        if self.log_final_collectives_estimations:
            from torch._inductor.fx_passes.node_runtime_estimation import (
                _log_graph_collective_benchmarks,
            )

            _log_graph_collective_benchmarks(
                self.gm, "fx_collectives_estimations_after_overlap_bucketing"
            )

        return self.gm

    def _reduce_exposed_time_of_in_flight_collectives(
        self,
        node: fx.Node,
        available_compute: float,
        exclude_pg: str | None = None,
    ) -> dict[str, float]:
        """
        Reduce exposed time of in-flight collectives using available compute time.

        Collectives on different process groups can overlap simultaneously with the same
        compute, so we track remaining time separately per PG.
        """
        # Initialize all PGs with full available compute (except excluded)
        remaining_time_per_pg: dict[str, float] = {
            pg: available_compute for pg in self.all_pgs if pg != exclude_pg
        }

        for start_node, info in self.in_flight.items():
            if info.exposed_time_ms == 0:
                continue

            pg_name = get_group_name(start_node)
            if pg_name == exclude_pg:
                continue

            pg_remaining = remaining_time_per_pg[pg_name]
            if pg_remaining <= 0:
                continue

            overlap_amount = min(info.exposed_time_ms, pg_remaining)
            info.exposed_time_ms -= overlap_amount
            remaining_time_per_pg[pg_name] -= overlap_amount
            info.hiding_nodes.add(node)

        return remaining_time_per_pg

    def _handle_compute_or_other(self, node: fx.Node) -> None:
        """Handle scheduling compute or other nodes and attempt to overlap with collectives."""
        runtime_estimate = self.node_estimations.get(node)

        # TODO: we could consider skipping overlapping for overlapable, unary chains to collectives.
        # using these nodes for overlap prevents bucketing. potentially if chain time < latency
        if runtime_estimate is None:
            if is_compute_node(node):
                raise AssertionError("should have estimate for compute nodes")
            self._schedule(node)
            return
        if runtime_estimate == 0 and not is_compute_node(node):
            self._schedule(node)
            return

        available_compute = runtime_estimate * self.compute_overlap_multipler

        # First, reduce exposed time of in-flight collectives (per PG)
        remaining_time_per_pg = self._reduce_exposed_time_of_in_flight_collectives(
            node, available_compute
        )
        # Then, schedule new collectives for overlap
        self._schedule_collectives_for_overlap(node, remaining_time_per_pg)
        self._schedule(node)

        if is_compute_node(node):
            self.current_compute_index += 1

    def _schedule(self, node: fx.Node) -> None:
        """Schedule a node."""
        if node in self.scheduled:
            raise AssertionError(f"node {node} already scheduled")
        if not all(n in self.scheduled for n in node.all_input_nodes):
            raise AssertionError(f"node {node} has unscheduled input nodes")
        self.scheduled.add(node)
        self._scheduled_bits |= self.node_ancestors.node_bit(node)
        self.memory_tracker.schedule_node(node)

        log.debug(
            "Scheduled node %s: current_memory=%d bytes, total_scheduled=%d",
            node.name,
            self.memory_tracker.get_current_memory_bytes(),
            len(self.scheduled),
        )

        for user in node.users:
            self.in_degree[user] -= 1
            if self.in_degree[user] == 0:
                self._add_to_ready_queue(user)

    def _compute_on_path_score(self, node: fx.Node) -> object:
        """Compute priority score for on-path nodes (domination-based)."""
        if _schedulable_wait_node(node):
            info = self.collective_info[self.wait_to_start[node]]
            # defer waits locally if they are exposed.
            compute_local_priority = int(info.is_exposed)
        else:
            # if we're scheduling this collective via its queue, then it was not
            # pre-fetched. we might as well maximize overlap for the
            # local, non-mm nodes prior to the next compute node.
            if self.in_overlappable_collective_unary_chain(node):
                compute_local_priority = -1
            else:
                compute_local_priority = 0

        return (
            self.compute_index_domination[node],  # what index compute it blocks
            compute_local_priority,  # collective_start=-1, wait=1, or neither=0
            self.node_idx[node],  # Original order for stability
        )

    def _compute_off_path_score(self, node: fx.Node) -> object:
        """
        Off-path priority scoring.

        Nodes that dominate reduce_scatters are prioritized (lower score = higher priority)
        to ensure they get scheduled eagerly for potential overlap.
        """
        dominates_rs = 0 if self.dominates_reduce_scatter(node) else 1
        return (dominates_rs, self.node_idx[node])

    @staticmethod
    def is_cheap_fn(node: fx.Node) -> bool:
        return getattr(node.target, "is_view", False) or torch.Tag.pointwise in getattr(
            node.target, "tags", ()
        )

    def in_overlappable_collective_unary_chain(self, curr: fx.Node) -> bool:
        while True:
            if len(curr.users) != 1:
                return False

            user = next(iter(curr.users))
            if len(user.all_input_nodes) != 1:
                return False

            if user in self.unscheduled_collectives:
                return True

            if not self.is_cheap_fn(user):
                return False

            curr = user

        return False

    def _should_force_wait_for_memory(self) -> bool:
        """Check if we need to force a wait due to memory pressure"""
        if not self.in_flight:
            return False

        return self.in_flight_bytes >= self.max_in_flight_bytes

    def _force_oldest_wait(self) -> None:
        """Schedule the oldest in flight wait"""
        self._handle_wait(self._get_oldest_wait())

    def _handle_collective_start(self, node: fx.Node) -> None:
        """Handle scheduling a collective start."""
        info = self.collective_info[node]

        if self.should_assume_bucketed(node):
            latency = estimate_collective_time(
                node, 0, custom_runtime_estimation=self.custom_runtime_estimation
            )
            info.exposed_time_ms = max(0, info.exposed_time_ms - latency)

        self.in_flight[node] = info
        self.in_flight_bytes += info.size_bytes
        self.unscheduled_collectives.discard(node)
        self._schedule(node)

    def _handle_wait(self, node: fx.Node) -> None:
        """Handle scheduling a wait."""
        if node not in self.wait_to_start:
            raise AssertionError(f"wait node {node} has no associated start")
        coll_start = self.wait_to_start[node]
        # For coalesced collectives, multiple waits share the same start node.
        # The first wait completes the collective; subsequent waits just schedule.
        if coll_start not in self.in_flight:
            self._schedule(node)
            return

        # Scheduling a wait of a collective also forces the wait
        # of every node enqueued prior to the collective on the
        # same process group
        group_name = get_group_name(coll_start)
        to_schedule: list[fx.Node] = []
        for in_flight_coll in self.in_flight:
            if in_flight_coll == coll_start:
                break
            if get_group_name(in_flight_coll) == group_name:
                to_schedule.append(in_flight_coll)

        for coll_to_schedule in to_schedule:
            self._handle_wait(self.collective_info[coll_to_schedule].wait_node)

        # If we are waiting on an exposed collective, use this time to
        # overlap on other PGs.
        info = self.collective_info[coll_start]
        if info.exposed_time_ms > 0:
            exposed_time = info.exposed_time_ms
            exclude_pg = group_name

            remaining_time_per_pg = self._reduce_exposed_time_of_in_flight_collectives(
                node, exposed_time, exclude_pg=exclude_pg
            )
            self._schedule_collectives_for_overlap(
                node, remaining_time_per_pg, exclude_pg=exclude_pg
            )

        self.in_flight_bytes -= self.in_flight[coll_start].size_bytes
        del self.in_flight[coll_start]
        self._schedule(node)

    def _schedule_collectives_for_overlap(
        self,
        overlap_node: fx.Node,
        remaining_time_per_pg: dict[str, float],
        exclude_pg: str | None = None,
    ) -> None:
        """Opportunistically schedule collectives that can be hidden by available overlap time."""
        if not remaining_time_per_pg or all(
            t <= 0 for t in remaining_time_per_pg.values()
        ):
            return

        # Compile candidates - limit by number of plausible candidates to bound
        # compile time.  Do not stop after seeing max_node_distance raw entries:
        # early entries may be filtered by PG/prefetch distance, while later
        # entries can still use the current overlap window.
        candidates = []
        for collective in self.unscheduled_collectives:
            pg_name = get_group_name(collective)
            if pg_name == exclude_pg:
                continue

            pg_available_time = remaining_time_per_pg.get(pg_name, 0.0)
            if pg_available_time <= 0:
                continue

            if (
                not self.off_compute_path(collective)
                and self.compute_index_domination[collective]
                - self.current_compute_index
                > self.max_compute_pre_fetch
            ):
                continue

            candidates.append(collective)
            if len(candidates) >= self.max_node_distance:
                break

        def get_priority(n: fx.Node) -> int:
            dominates_next_compute = (
                self.compute_index_domination[n] == self.current_compute_index + 1
            )
            if dominates_next_compute:
                return 0  # Dominates next compute layer - most urgent
            elif self.off_compute_path(n) and self.dominates_reduce_scatter(n):
                return 1  # Off-path but blocks reduce_scatter
            elif not self.off_compute_path(n):
                return 2  # On-path but not immediate
            else:
                return 3  # Off-path, doesn't block reduce_scatter

        def overlap_priority(n: fx.Node) -> tuple[int, int, float, int]:
            return (
                get_priority(n),
                self.compute_index_domination[n],
                -min(
                    remaining_time_per_pg.get(get_group_name(n), 0.0),
                    self.collective_info[n].exposed_time_ms,
                ),
                self.node_idx[n],
            )

        candidates.sort(key=overlap_priority)

        if self.prioritize_bucketing_during_scheduling:
            # group candidates by bucket key first so same-bucket
            # collectives are scheduled together, maximizing bucketing opportunities
            bucket_groups: dict[object, list[fx.Node]] = defaultdict(list)
            for coll in candidates:
                key = get_full_bucket_key(coll, self.bucket_mode)
                bucket_groups[key].append(coll)

            # Sort bucket groups by the best candidate priority in each group,
            # with larger groups first as tiebreaker.
            sorted_bucket_keys = sorted(
                bucket_groups.keys(),
                key=lambda k: (
                    min(overlap_priority(c) for c in bucket_groups[k]),
                    -len(bucket_groups[k]),
                ),
            )

            # Flatten back to ordered candidate list
            candidates = []
            for b_key in sorted_bucket_keys:
                group = bucket_groups[b_key]
                group.sort(key=overlap_priority)
                candidates.extend(group)

        for collective in candidates:
            pg_name = get_group_name(collective)
            pg_available_time = remaining_time_per_pg[pg_name]

            if pg_available_time <= 0:
                continue

            why = WhyNoOverlap(overlap_node, collective)
            info = self.collective_info[collective]

            if self.node_ancestors.has_dep(overlap_node, collective):
                why("dependency conflict")
                continue

            # Check if prefetching would exceed memory budget
            if self._prefetch_would_exceed_memory_budget(collective):
                why("prefetch would exceed memory budget")
                continue

            # Try to free memory by forcing hidden waits
            while (
                self.in_flight
                and (self.max_in_flight_bytes - self.in_flight_bytes) < info.size_bytes
                and self._wait_is_hidden(self._get_oldest_wait(), overlap_node)
            ):
                self._force_oldest_wait()

            if (self.max_in_flight_bytes - self.in_flight_bytes) < info.size_bytes:
                why("in-flight memory limit")
                continue

            # Check if we can reach this collective without scheduling compute, other collectives, or waits
            path = self._find_schedulable_path(collective, overlap_node, why)
            if path is None:
                continue

            log.debug(
                "Overlapping collective %s with node %s: coll_domination=%d, current_depth=%d",
                collective.name,
                overlap_node.name,
                self.compute_index_domination[collective],
                self.current_compute_index,
            )

            # TODO: We previously tracked path compute time and added it back to available
            # overlap time. With per-PG tracking this is complex: if there were in-flight
            # collectives on one PG but not another, we can't add path time back to the PG
            # that wasn't in-flight

            # Schedule path and collective
            self._schedule_path_to_collective(path, overlap_node)
            self._handle_collective_start(collective)
            self._update_cumulative_prefetch_memory(collective, info)

            # Update exposed time for this collective
            overlap_amount = min(pg_available_time, info.exposed_time_ms)
            info.exposed_time_ms -= overlap_amount
            info.hiding_nodes.add(overlap_node)

            # Update available time for this PG
            remaining_time_per_pg[pg_name] -= overlap_amount

            if sum(remaining_time_per_pg.values()) == 0:
                break

        if remaining_time_per_pg:
            self.wasted_compute += min(remaining_time_per_pg.values())

    def _find_schedulable_path(
        self, target: fx.Node, curr_overlap_node: fx.Node | None, why: WhyNoOverlap
    ) -> OrderedSet[fx.Node] | None:
        """Find path to target by collecting unscheduled dependencies.

        Called O(C * K) times where C = compute nodes and K = unscheduled
        collectives. Most calls return None because the collective has
        unscheduled compute ancestors. The bitset pre-filter below answers
        this in O(N/64) using three bitwise operations, avoiding a full
        backward BFS in the majority of cases.
        """
        # Bitset pre-filter: O(N/64) check that avoids BFS in most cases.
        ancestor_bits = self.node_ancestors.get_ancestor_bits(target)
        unscheduled_bits = ancestor_bits & ~self._scheduled_bits
        if unscheduled_bits & self._compute_bits:
            why("bitset: unscheduled compute ancestor")
            return None

        # No unscheduled ancestors at all -- path is trivially clear.
        if not unscheduled_bits:
            return OrderedSet()

        # Pre-filter passed (rare): fall through to BFS for exact check
        # (the pre-filter only checks compute nodes; BFS also checks for
        # unscheduled collectives and wait-node conflicts).

        # Backward BFS from target; stops at scheduled nodes.
        target_idx = self.node_idx[target]
        unscheduled_ancestors: OrderedSet[fx.Node] = OrderedSet()
        seen: OrderedSet[fx.Node] = OrderedSet()
        parent_lists = self._parent_lists
        node_idx = self.node_idx
        stack = parent_lists[target_idx][:]
        scheduled = self.scheduled
        while stack:
            n = stack.pop()
            if n in seen or n in scheduled:
                continue
            seen.add(n)
            unscheduled_ancestors.add(n)
            stack.extend(parent_lists[node_idx[n]])

        # only schedule non distributed, non compute nodes
        for node in unscheduled_ancestors:
            if is_compute_node(node):
                why("path blocked by compute node %s", node.name)
                return None

            if node in self.unscheduled_collectives:
                why("path blocked by unscheduled collective %s", node.name)
                return None

            # if we schedule a wait tensor whose start collective is hidden by the
            # current compute node we are scheduling, then we are effectively exposing it.
            # similarly, don't schedule a wait of a collective that could be otherwise hidden,
            # thus forcing it to be exposed.
            # however, if it is already hidden it's fine to schedule it
            if _schedulable_wait_node(node):
                info = self.collective_info[self.wait_to_start[node]]
                if (not info.is_exposed) and (
                    curr_overlap_node not in info.hiding_nodes
                ):
                    continue

                why(
                    "path blocked by wait node %s (exposed=%s, hidden_by_curr_overlap=%s)",
                    node.name,
                    info.is_exposed,
                    curr_overlap_node in info.hiding_nodes,
                )
                return None

            # Skip c10 ops and dtensor shard ops - they should be scheduled via main loop
            target_str = str(node.target)
            if "c10" in target_str or "_dtensor" in target_str:
                log.debug(
                    "Skipping c10/dtensor op %s in path to collective",
                    node.name,
                )
                return None

        return unscheduled_ancestors

    def should_assume_bucketed(self, node: fx.Node) -> bool:
        """
        Check if there's an in-flight collective that can be bucketed with the given node. If so, assume they will bucket.
        This is a optimistic heuristic to account for latency reduction with bucketing. The two nodes may not get bucketed.
        """
        if not torch._inductor.config.test_configs.assume_bucketing_reduces_latency:
            return False

        key = bucket_key(node, mode=self.bucket_mode)
        if key is None:
            return False

        for in_flight_coll in self.in_flight:
            if bucket_key(in_flight_coll, mode=self.bucket_mode) == key:
                return True

        return False

    def _get_oldest_wait(self) -> fx.Node:
        oldest_start = next(iter(self.in_flight))
        return self.collective_info[oldest_start].wait_node

    def _wait_is_hidden(
        self, wait_node: fx.Node, overlap_node: fx.Node | None = None
    ) -> bool:
        if not is_wait_tensor(wait_node):
            raise AssertionError(f"expected a wait tensor node, got {wait_node}")
        info = self.collective_info[self.wait_to_start[wait_node]]
        return not info.is_exposed and overlap_node not in info.hiding_nodes

    def _schedule_path_to_collective(
        self, path: OrderedSet[fx.Node], curr_overlap_node: fx.Node
    ) -> None:
        """Schedule all nodes needed to reach a collective."""

        if not all(n not in self.scheduled for n in path):
            raise AssertionError("path contains already-scheduled nodes")
        for node in sorted(path, key=lambda n: self.node_idx[n]):
            if is_compute_node(node) or node in self.unscheduled_collectives:
                raise AssertionError(
                    f"unexpected compute or unscheduled collective node {node} in path"
                )
            if _schedulable_wait_node(node):
                # When we schedule wait tensors, we also force realization of all
                # collectives enqueued prior to their corresponding collective.
                # It's possible the scheduling of one wait tensor here has forced
                # another in the path. If so, skip scheduling it.
                if node in self.scheduled:
                    continue

                info = self.collective_info[self.wait_to_start[node]]
                if curr_overlap_node in info.hiding_nodes:
                    raise AssertionError(
                        "overlap node unexpectedly in collective hiding_nodes"
                    )
                self._handle_wait(node)
                continue

            self._schedule(node)

    def reorder_graph(self) -> None:
        output_node = self.graph.output_node()
        for node in self.scheduled:
            if node.op == "placeholder":
                continue
            output_node.prepend(node)
        self.graph.lint()

    def _reorder_graph(self) -> None:
        """Reorder graph based on schedule."""
        exposed = [
            c
            for c in self.collective_info.values()
            if c.exposed_time_ms == c.estimated_time_ms
        ]

        potentially_hidden_collectives = self.compute_potential_hidden_collectives()
        bad_exposed = [
            c for c in exposed if c.start_node in potentially_hidden_collectives
        ]

        # Compute total exposed and potential exposed time
        total_exposed = sum(c.exposed_time_ms for c in self.collective_info.values())
        hideable_exposed_ms = sum(
            self.collective_info[c].exposed_time_ms
            for c in potentially_hidden_collectives
        )
        total_potential_exposed = sum(
            c.estimated_time_ms for c in self.collective_info.values()
        )

        counters["inductor"]["overlap_scheduling_exposed"] += len(exposed)
        counters["inductor"]["overlap_scheduling_bad_exposed"] += len(bad_exposed)
        counters["inductor"]["overlap_scheduling_potentially_hidden"] += len(
            potentially_hidden_collectives
        )
        counters["inductor"]["overlap_original_mem"] = self.original_peak_memory
        counters["inductor"]["rescheduled_mem"] = self.memory_tracker.peak_memory

        log.info(
            "Overlap scheduling results: exposed=%d, bad_exposed=%d, potentially_hidden=%d, "
            "original_peak_memory=%d bytes, rescheduled_peak_memory=%d bytes, "
            "total_exposed_ms=%.2f, hideable_exposed_ms=%.2f, total_potential_exposed_ms=%.2f, "
            "wasted_compute_ms=%.2f",
            len(exposed),
            len(bad_exposed),
            len(potentially_hidden_collectives),
            self.original_peak_memory,
            self.memory_tracker.peak_memory,
            total_exposed,
            hideable_exposed_ms,
            total_potential_exposed,
            self.wasted_compute,
        )

        self.reorder_graph()

    def _bucket_collectives(self) -> None:
        from torch._inductor.fx_passes.overlap_preserving_bucketer import (
            OverlapPreservingBucketer,
        )

        bucketer = OverlapPreservingBucketer(
            graph=self.graph,
            collective_info=self.collective_info,
            scheduled=self.scheduled,
            max_bucket_memory_gb=2.0,  # Could make this configurable
            max_coll_distance=self.max_node_distance,
            insert_overlap_deps=self.insert_overlap_deps,
            bucket_mode=self.bucket_mode,
            bucket_exposed_first=self.bucket_exposed_first,
            bucket_only_internode_comms=self.bucket_only_internode_comms,
        )
        bucketer.bucket_collectives()

    def compute_potential_hidden_nodes(
        self, nodes_to_check: Iterable[fx.Node]
    ) -> dict[fx.Node, fx.Node]:
        """
        Returns a dict containing a mapping of nodes which could potentially be hidden to their hiding node
        """

        def could_be_hidden(start: fx.Node) -> fx.Node | None:
            for compute_node in self.compute_nodes:
                if not self.node_ancestors.has_dep(compute_node, start):
                    return compute_node

            return None

        # TODO: We could potentially limit compute nodes per overlap time,
        # today, this is optimistic, and just serves to avoid deferring
        # collectives/waits that have no possible overlap as well as for analysis of how
        # successfully we hid compute
        potentially_hidden = {}
        for node in nodes_to_check:
            if mm := could_be_hidden(node):
                potentially_hidden[node] = mm

        return potentially_hidden

    def compute_potential_hidden_collectives(self) -> dict[fx.Node, fx.Node]:
        """Compute which collective operations could be hidden by compute."""
        return self.compute_potential_hidden_nodes(self.collective_info.keys())

    def compute_potential_hidden_waits(self) -> dict[fx.Node, fx.Node]:
        """Compute which wait operations could be hidden by compute."""
        wait_nodes = [info.wait_node for info in self.collective_info.values()]
        return self.compute_potential_hidden_nodes(wait_nodes)


def gather_node_runtime_estimations(
    gm: torch.fx.GraphModule,
    custom_runtime_estimation: Callable[[fx.Node, int | None], float | None]
    | None = None,
    collective_estimator: Literal["analytical", "benchmark"] = "analytical",
    enable_fusion_regions: bool = False,
    log_estimations: bool = False,
) -> dict[fx.Node, float]:
    """Gather initial runtime estimations for all nodes without scheduling.

    Uses analytical models (roofline) for compute nodes -- the alignment step
    in OverlapScheduler.run() replaces these with benchmarked + cross-rank-aligned
    values. Collectives use bandwidth formulas or CUDA events depending on
    collective_estimator.

    When enable_fusion_regions is True, builds fusion regions and corrects
    per-node I/O costs to account for fusion (no graph mutation).

    Args:
        collective_estimator: "analytical" uses bandwidth formulas,
            "benchmark" uses CUDA events for collectives.
        log_estimations: When True, log compute and collective estimations
            via trace_structured for tlparse.

    Returns estimations mapping fx.Node to runtime in ms.
    """
    fused_costs: dict[fx.Node, float] = {}
    if enable_fusion_regions:
        from torch._inductor.fx_passes.fusion_regions import (
            build_fusion_regions,
            estimate_fused_node_costs,
        )

        raw_regions = build_fusion_regions(gm)
        if raw_regions:
            fused_costs = estimate_fused_node_costs(raw_regions)

    estimations: dict[fx.Node, float] = {}
    nodes = list(gm.graph.nodes)

    # Collectives
    collective_nodes: list[fx.Node] = []
    for node in nodes:
        if _schedulable_wait_node(node):
            start = _get_collective_node_from_wait(node)
            if start is None:
                raise AssertionError("wait node has no associated collective")
            if start in estimations:
                continue
            estimations[start] = estimate_collective_time(
                start,
                custom_runtime_estimation=custom_runtime_estimation,
                collective_estimator=collective_estimator,
            )
            collective_nodes.append(start)

    # Compute nodes (matmul, bmm, etc.) — analytical estimates only.
    # The alignment step in run() replaces these with benchmarked + aligned values.
    compute_nodes: list[fx.Node] = []
    compute_analytical: list[float] = []

    for node in nodes:
        if is_compute_node(node):
            est = estimate_roofline_runtime_ms(node)
            if custom_runtime_estimation is not None:
                custom_est = custom_runtime_estimation(node, None)
                if custom_est is not None:
                    est = custom_est
            estimations[node] = est
            compute_nodes.append(node)
            compute_analytical.append(est)
        elif node.op == "call_function" and node not in estimations:
            if custom_runtime_estimation is not None:
                est = custom_runtime_estimation(node, None)
                if est is not None:
                    estimations[node] = est
            else:
                est = estimate_roofline_runtime_ms(node)
                if est > 0:
                    estimations[node] = est

    # Apply fused costs: replace individual I/O estimates with fusion-aware ones
    estimations.update(fused_costs)

    # Logging
    if log_estimations and compute_nodes:
        from torch._inductor.fx_passes.node_runtime_estimation import (
            _log_compute_estimations,
        )

        _log_compute_estimations(
            compute_nodes,
            compute_analytical,
            compute_analytical,
        )

    # Skip analytical logging when a custom estimator is provided: the
    # analytical estimates weren't used for scheduling, and the logging
    # path calls into NCCL estimation which can crash when group_name
    # is an FX Node (compile-on-one-rank graphs).
    if log_estimations and collective_nodes and custom_runtime_estimation is None:
        from torch._inductor.fx_passes.node_runtime_estimation import (
            _log_collective_benchmarks,
        )

        _log_collective_benchmarks(
            collective_nodes,
            artifact_name="fx_collectives_analytical_estimation",
        )

    return estimations


def align_estimations_across_ranks(
    estimations: dict[fx.Node, float],
) -> dict[fx.Node, float]:
    """Align runtime estimations across distributed ranks using median.

    All ranks must make identical scheduling decisions, so we gather each
    rank's values and take the median. All nodes in estimations are aligned.

    Returns a new estimations dict with aligned values.
    """
    import torch.distributed as dist
    from torch._subclasses.fake_tensor import unset_fake_temporarily
    from torch.distributed.distributed_c10d import _get_default_group

    nodes = list(estimations.keys())
    if not nodes:
        return {}

    local_values = [estimations[n] for n in nodes]

    world_size = dist.get_world_size()
    pg = _get_default_group()

    with unset_fake_temporarily():
        gathered: list[list[float]] = [[] for _ in range(world_size)]
        dist.all_gather_object(gathered, local_values, pg)
        medians = torch.median(torch.tensor(gathered), dim=0).values.tolist()

    return dict(zip(nodes, medians))


def schedule_overlap_bucketing(
    gm: torch.fx.GraphModule,
    max_in_flight_gb: float = 5,
    max_compute_pre_fetch: int = 200,
    collective_bucketing: bool = False,
    insert_overlap_deps: bool = False,
    compute_overlap_multipler: float = 1.0,
    max_coll_distance: int = 200,
    custom_runtime_estimation: Callable[[fx.Node, int | None], float | None]
    | None = None,
    collective_estimator: Literal["analytical", "benchmark"] = "analytical",
    compute_estimator: Literal["analytical", "benchmark"] = "benchmark",
    max_memory_increase_gb: float | None = 1.0,
    max_memory_increase_ratio: float | None = 0.05,
    log_final_collectives_estimations: bool = False,
    bucket_exposed_first: bool | None = None,
    enable_fusion_regions: bool = False,
    bucket_only_internode_comms=False,
    prioritize_bucketing_during_scheduling: bool = True,
    max_off_bucket_gb: float | None = 0.5,
    bucket_mode: BucketMode | None = None,
    pge_profile_path: str | None = None,
    pre_bucketing_fsdp_collectives: bool = True,
    pre_bucketing_fsdp_collectives_bucket_cap_mb: float | None = None,
) -> torch.fx.GraphModule:
    """Schedule nodes to maximize compute-collective overlap.

    Args:
        gm: Input graph module to optimize.
        max_in_flight_gb: Maximum GB of concurrent collective data. Too much in flight memory
            can cause memory fragmentation within the CUDA Caching Allocator.
        max_compute_pre_fetch: Maximum mm nodes to pre fetch. Note: should already be limited by max_in_flight_gb and
            max_memory_increase_gb
        collective_bucketing: Enable overlap-preserving collective bucketing.
        insert_overlap_deps: Insert overlap dependencies using control deps operator. This should only be used if
            compiling with inductor, or for subsequent passes before removing the ops prior to execution.
        compute_overlap_multipler: Scale factor for compute time used to hide collectives. This can be used
            to address over or under aggressive overlapping.
        max_coll_distance: Maximum pre fetch or bucketing candidates. Mainly intended for compile time
        custom_runtime_estimation: Override runtime estimation for specific nodes. Called as
            custom_runtime_estimation(node, override_size) -> float | None. To pass pre-computed
            estimations, wrap a dict: lambda node, _: estimations.get(node).
        collective_estimator: Method for estimating collective runtime. "analytical" uses bandwidth formulas,
            "benchmark" uses CUDA events with power-of-2 rounding and interpolation.
        compute_estimator: Method for estimating compute (ATen op) runtime. "analytical" uses roofline model
            estimates (deterministic, no GPU sync), "benchmark" uses GPU benchmarking (more accurate).
        max_memory_increase_gb: Maximum GB increase above baseline memory (absolute cap). If None, no absolute limit.
        max_memory_increase_ratio: Maximum increase as ratio of baseline peak memory. If None, no ratio limit.
            Uses minimum of absolute and ratio limits when both are specified.
        enable_fusion_regions: Enable fusion region detection and cost estimation for fusible ops.
        bucket_mode: Bucketing mode for grouping collectives.
        pre_bucketing_fsdp_collectives: Pre-bucket FSDP collectives into bandwidth-saturating
            buckets before overlap scheduling.
        pre_bucketing_fsdp_collectives_bucket_cap_mb: Override bucket cap in MB for pre-bucketing.
            When None, auto-computes based on NCCL bandwidth model.
    """
    if not gm.graph.find_nodes(
        op="call_function",
        target=torch.ops._c10d_functional.wait_tensor.default,
    ):
        return gm

    if pre_bucketing_fsdp_collectives:
        from torch._inductor.fx_passes.fsdp import pre_bucket_fsdp_collectives

        pre_bucket_fsdp_collectives(
            gm,
            mode=bucket_mode,
            bucket_cap_mb=pre_bucketing_fsdp_collectives_bucket_cap_mb,
        )

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "overlap_scheduling_graph_before",
            "encoding": "string",
        },
        payload_fn=lambda: gm.print_readable(False),
    )
    ret = OverlapScheduler(
        gm,
        compute_overlap_multipler=compute_overlap_multipler,
        max_in_flight_gb=max_in_flight_gb,
        max_coll_distance=max_coll_distance,
        max_compute_pre_fetch=max_compute_pre_fetch,
        custom_runtime_estimation=custom_runtime_estimation,
        collective_bucketing=collective_bucketing,
        insert_overlap_deps=insert_overlap_deps,
        collective_estimator=collective_estimator,
        compute_estimator=compute_estimator,
        max_memory_increase_gb=max_memory_increase_gb,
        max_memory_increase_ratio=max_memory_increase_ratio,
        log_final_collectives_estimations=log_final_collectives_estimations,
        bucket_exposed_first=bucket_exposed_first,
        enable_fusion_regions=enable_fusion_regions,
        bucket_only_internode_comms=bucket_only_internode_comms,
        prioritize_bucketing_during_scheduling=prioritize_bucketing_during_scheduling,
        max_off_bucket_gb=max_off_bucket_gb,
        bucket_mode=bucket_mode,
        pge_profile_path=pge_profile_path,
    ).run()
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "overlap_scheduling_graph_after",
            "encoding": "string",
        },
        payload_fn=lambda: ret.print_readable(False),
    )

    return ret


def schedule_overlap_bucketing_from_inductor_configs(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """Schedule nodes to maximize compute-collective overlap using inductor configs.

    Reads configuration from torch._inductor.config.aten_distributed_optimizations
    and calls schedule_overlap_bucketing with those settings.
    """
    if not gm.graph.find_nodes(
        op="call_function",
        target=torch.ops._c10d_functional.wait_tensor.default,
    ):
        return gm

    from torch._inductor import config

    dist_opts = config.aten_distributed_optimizations

    kwargs: dict[str, object] = {}

    config_keys = (
        "collective_bucketing",
        "max_compute_pre_fetch",
        "custom_runtime_estimation",
        "insert_overlap_deps",
        "collective_estimator",
        "compute_estimator",
        "max_memory_increase_gb",
        "max_memory_increase_ratio",
        "compute_overlap_multipler",
        "max_in_flight_gb",
        "max_coll_distance",
        "log_final_collectives_estimations",
        "bucket_exposed_first",
        "bucket_only_internode_comms",
        "enable_fusion_regions",
        "prioritize_bucketing_during_scheduling",
        "bucket_mode",
        "pre_bucketing_fsdp_collectives",
        "pre_bucketing_fsdp_collectives_bucket_cap_mb",
    )
    for key in config_keys:
        if (val := getattr(dist_opts, key, None)) is not None:
            kwargs[key] = val

    # Profile-guided latency estimation
    pge_path = dist_opts.profile_guided_estimations_profile_path
    if pge_path and "custom_runtime_estimation" not in kwargs:
        kwargs["pge_profile_path"] = pge_path

    return schedule_overlap_bucketing(gm, **kwargs)  # type: ignore[arg-type]
<|endoftext|># mypy: allow-untyped-defs
import collections
import logging
import operator
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from typing import Any

import torch
from torch._dynamo.utils import counters, is_node_meta_valid
from torch._logging import trace_structured
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.symbolic_shapes import free_symbols
from torch.fx.passes.graph_transform_observer import GraphTransformObserver
from torch.utils._ordered_set import OrderedSet

from .. import config
from ..pattern_matcher import (
    CallFunctionVarArgs,
    CallMethodVarArgs,
    get_arg_value,
    MatchResult,
    stable_topological_sort,
)
from ..utils import OPTIMUS_EXCLUDE_POST_GRAD


try:
    # importing this will register fbgemm lowerings for inductor
    import deeplearning.fbgemm.fbgemm_gpu.fb.inductor_lowerings  # noqa: F401

    has_fbgemm = True
except Exception:
    has_fbgemm = False

aten = torch.ops.aten

log = logging.getLogger(__name__)

DEFAULT_BETA = 1
DEFAULT_ALPHA = 1

MIN_FUSE_SET_SIZE = 5
MAX_FUSE_SET_SIZE = 300
MAX_FUSE_SEARCH_DEPTH = 5
# The maximum tensor size that can go into the fusion group
MAX_FUSE_TENSOR_SIZE_GROUP_LINEAR = 4096
# Whether we only fuse nodes with same parent node
FUSE_NODES_WITH_SAME_PARENT = False
# Whether we enable the add broadcast in batch linear
SHAPE_BROADCAST_BATCH_LINEAR = False
# Whether we enable the fuse nodes with same users
Fuse_NODES_WITH_SAME_USERS = False

# exclude these nodes from BFS
# excluding get item improves optimizer compilation time by 60s
SEARCH_EXCLUSIONS = OrderedSet([operator.getitem])


default_graph_search_options = {
    "min_fuse_set_size": MIN_FUSE_SET_SIZE,
    "max_fuse_set_size": MAX_FUSE_SET_SIZE,
    "max_fuse_search_depth": MAX_FUSE_SEARCH_DEPTH,
    "max_fuse_tensor_size_group_linear": MAX_FUSE_TENSOR_SIZE_GROUP_LINEAR,
    "fuse_nodes_with_same_parent": FUSE_NODES_WITH_SAME_PARENT,
    "shape_broadcast_batch_linear": SHAPE_BROADCAST_BATCH_LINEAR,
    "fuse_nodes_with_same_users": Fuse_NODES_WITH_SAME_USERS,
}

graph_search_options = default_graph_search_options


def update_stack_example_value(node, metadata, dim=0, op=torch.stack):
    """
    Update the example value of the node in the graph to enable followup split cat opt.
    """
    if node is not None and hasattr(node, "meta"):
        if op is torch.stack:
            example_value = torch.stack(metadata, dim=dim)
        elif op is torch.unbind:
            example_value = torch.unbind(metadata, dim=dim)  # type: ignore[assignment]
        else:
            return
        node.meta["example_value"] = example_value


def update_pointwise_example_value(pointwise_node, input, other, op):
    """
    Update the example value of the add node in the graph to enable followup split cat opt.
    """
    if pointwise_node is not None and hasattr(pointwise_node, "meta"):
        if op is torch.add:
            example_value = torch.add(input, other)
        elif op is torch.mul:
            example_value = torch.mul(input, other)
        else:
            return
        pointwise_node.meta["example_value"] = example_value


class GroupBatchFusionBase:
    def __init__(self, **kwargs) -> None:
        self.graph_search_options = kwargs.pop(
            "graph_search_options", default_graph_search_options
        )

    def match(self, node):
        raise NotImplementedError("match called on base")

    def fuse(self, graph, subset):
        raise NotImplementedError("fuse called on base")


PRE_GRAD_FUSIONS: dict[str, GroupBatchFusionBase] = {}
POST_GRAD_FUSIONS: dict[str, GroupBatchFusionBase] = {}


def register_fusion(name: str, pre_grad=True):
    def decorator(fusion_cls: GroupBatchFusionBase):
        if pre_grad:
            PRE_GRAD_FUSIONS[name] = fusion_cls
        else:
            POST_GRAD_FUSIONS[name] = fusion_cls
        return fusion_cls

    return decorator


def list_group_batch_fusions(pre_grad=True) -> list[str]:
    if pre_grad:
        return list(PRE_GRAD_FUSIONS.keys())
    else:
        return list(POST_GRAD_FUSIONS.keys())


def decompose_stack(graph: torch.fx.GraphModule, input_tensors: list[Any]) -> Any:
    unsqueezed_inputs = []
    unsqueezed_inputs_meta = []
    for input_tensor in input_tensors:
        unsqueezed_input = graph.call_function(  # type: ignore[operator]
            aten.unsqueeze, args=(input_tensor,), kwargs={"dim": 0}
        )
        unsqueezed_inputs.append(unsqueezed_input)
        unsqueezed_input.meta["val"] = aten.unsqueeze(input_tensor.meta["val"], dim=0)  # type: ignore[assignment]
        unsqueezed_inputs_meta.append(unsqueezed_input.meta["val"])
    stacked_inputs = graph.call_function(  # type: ignore[operator]
        aten.cat, args=(unsqueezed_inputs,), kwargs={"dim": 0}
    )
    stacked_inputs.meta["val"] = aten.cat(unsqueezed_inputs_meta, dim=0)  # type: ignore[assignment]
    return stacked_inputs


class GroupFusion(GroupBatchFusionBase):
    """
    Fuse ops in a group way, e.g, fuse mm/addmm of arbitrary input shapes with fbgemm.gmm.
    """


class BatchFusion(GroupBatchFusionBase):
    """
    Fuse ops in a batch way, e.g, fuse mm/addmm of same input shapes with bmm.
    """


class BatchPointwiseOpsFusionFactory(BatchFusion):
    def __init__(self, op, **kwargs) -> None:
        super().__init__(**kwargs)
        self.op = op


@register_fusion("batch_linear_post_grad", pre_grad=False)
class PostGradBatchLinearFusion(BatchFusion):
    """
    Fuse ops in a batch way in post grad (aten level).
    """

    def _addmm_node_can_be_fused(self, node: torch.fx.Node) -> bool:
        # pyre-fixme[7]: Incompatible return type
        return (
            # pyrefly: ignore [bad-return]
            node.kwargs.get("beta", DEFAULT_BETA) == DEFAULT_BETA
            and node.kwargs.get("alpha", DEFAULT_ALPHA) == DEFAULT_ALPHA  # type: ignore[return-value]
        )

    def _is_input_2d(self, input: torch.fx.Node) -> bool:
        input_shapes = input.meta["val"].shape
        return (
            len(input_shapes) == 2
            and isinstance(input_shapes[0], int)
            and isinstance(input_shapes[1], int)
        )

    def match(self, node: torch.fx.Node) -> tuple[str, int, int, int, bool, str] | None:
        if CallFunctionVarArgs(aten.mm).match(node):
            input_m, weight_m = node.args
            bias_m = None

        elif CallFunctionVarArgs(aten.addmm.default).match(
            node
        ) and self._addmm_node_can_be_fused(node):
            bias_m, input_m, weight_m = node.args
        else:
            return None
        # get the user of the node
        if self.graph_search_options.get("fuse_nodes_with_same_users", False):
            users = [user.target for user in node.users]
        else:
            users = ""  # type: ignore[assignment]
        # only handle the cases where inputs are 2D tensors
        if not self._is_input_2d(input_m) or not self._is_input_2d(weight_m):  # type: ignore[arg-type]
            return None
        m, k = input_m.meta["val"].shape  # type: ignore[union-attr]
        n = weight_m.meta["val"].shape[1]  # type: ignore[union-attr]
        batch_key = ("batch_linear_post_grad", m, k, n, bias_m is not None, str(users))
        return batch_key

    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]):
        batch_inputs = []
        batch_weights = []
        batch_biases = []
        batch_nodes = []
        batch_inputs_meta = []
        batch_weights_meta = []
        batch_biases_meta = []

        for node in subset:
            if CallFunctionVarArgs(aten.addmm.default).match(node):
                bias, input, weight = node.args
            elif CallFunctionVarArgs(aten.mm.default).match(node):
                input, weight = node.args
                bias = None
            batch_nodes.append(node)
            batch_inputs.append(input)  # type: ignore[possibly-undefined]
            batch_weights.append(weight)  # type: ignore[possibly-undefined]
            batch_biases.append(bias)  # type: ignore[possibly-undefined]
            batch_inputs_meta.append(input.meta)  # type: ignore[possibly-undefined, union-attr]
            batch_weights_meta.append(weight.meta)  # type: ignore[possibly-undefined, union-attr]
            if bias is not None:  # type: ignore[possibly-undefined]
                batch_biases_meta.append(bias.meta)  # type: ignore[possibly-undefined, union-attr]
            else:
                batch_biases_meta.append(None)

        with graph.inserting_before(subset[-1]):  # type: ignore[operator]
            fused_inputs = decompose_stack(graph, batch_inputs)
            fused_weights = decompose_stack(graph, batch_weights)
            fused_inputs_meta_val = torch.stack(
                [input["val"] for input in batch_inputs_meta]
            )
            fused_weights_meta_val = torch.stack(
                [weight["val"] for weight in batch_weights_meta]
            )
            fused_bmm = graph.call_function(  # type: ignore[operator]
                aten.bmm,
                args=(fused_inputs, fused_weights),
            )
            fused_bmm.meta["val"] = aten.bmm(
                fused_inputs_meta_val, fused_weights_meta_val
            )
        for i, original_mm in enumerate(batch_nodes):
            has_bias = False
            with graph.inserting_after(fused_bmm):  # type: ignore[operator]
                new_mm = graph.call_function(aten.select, args=((fused_bmm, 0, i)))  # type: ignore[operator]
                new_mm.meta["val"] = aten.select(fused_bmm.meta["val"], 0, i)
                if batch_biases[i]:
                    has_bias = True
                    # broadcast the bias to the same shape as the mm output
                    if self.graph_search_options.get(
                        "shape_broadcast_batch_linear", False
                    ):
                        broadcast_shape = torch.broadcast_shapes(
                            batch_biases_meta[i]["val"].shape, new_mm.meta["val"].shape
                        )
                        broadcast_bias = graph.call_function(  # type: ignore[operator]
                            aten.broadcast_to.default,
                            args=(batch_biases[i],),
                            kwargs={"size": broadcast_shape},
                        )
                        broadcast_bias.meta["val"] = aten.broadcast_to(
                            batch_biases_meta[i]["val"], broadcast_shape
                        )  # type: ignore[assignment]
                        new_bias_add = graph.call_function(  # type: ignore[operator]
                            aten.add.Tensor, args=((broadcast_bias, new_mm))
                        )
                        new_bias_add.meta["val"] = aten.add.Tensor(
                            broadcast_bias.meta["val"], new_mm.meta["val"]
                        )
                    else:
                        new_bias_add = graph.call_function(  # type: ignore[operator]
                            aten.add, args=((batch_biases[i], new_mm))
                        )
                        new_bias_add.meta["val"] = aten.add.Tensor(
                            batch_biases_meta[i]["val"], new_mm.meta["val"]
                        )
            new_mm_cont = new_bias_add if has_bias else new_mm  # type: ignore[possibly-undefined]
            original_mm.replace_all_uses_with(new_mm_cont)
            new_mm_cont.meta.update(original_mm.meta)
            graph.erase_node(original_mm)  # type: ignore[operator]
        counters["inductor"]["batch_linear_post_grad"] += 1


@register_fusion("group_linear", pre_grad=False)
class GroupLinearFusion(GroupFusion):
    def _addmm_node_can_be_fused(self, node: torch.fx.Node):
        input_shape = node.args[1].meta["val"].shape  # type: ignore[union-attr]
        weight_shape = node.args[2].meta["val"].shape  # type: ignore[union-attr]
        return (
            node.kwargs.get("beta", DEFAULT_BETA) == DEFAULT_BETA
            and node.kwargs.get("alpha", DEFAULT_ALPHA) == DEFAULT_ALPHA
            and len(input_shape) == 2
            and len(weight_shape) == 2
            and all(x % 2 == 0 for x in input_shape + weight_shape)
            and all(
                shape <= self.graph_search_options["max_fuse_tensor_size_group_linear"]
                for shape in input_shape + weight_shape
            )
        )

    def _mm_node_can_be_fused(self, node: torch.fx.Node):
        input_shape = node.args[0].meta["val"].shape  # type: ignore[union-attr]
        weight_shape = node.args[1].meta["val"].shape  # type: ignore[union-attr]
        return (
            len(input_shape) == 2
            and len(weight_shape) == 2
            and all(x % 2 == 0 for x in input_shape + weight_shape)
            and all(
                shape <= self.graph_search_options["max_fuse_tensor_size_group_linear"]
                for shape in input_shape + weight_shape
            )
        )

    def match(self, node: torch.fx.Node) -> tuple[str, bool] | None:
        if CallFunctionVarArgs(aten.mm.default).match(
            node
        ) and self._mm_node_can_be_fused(node):
            group_key = ("group_linear", True)
        elif CallFunctionVarArgs(aten.addmm.default).match(
            node
        ) and self._addmm_node_can_be_fused(node):
            bias = node.args[0]
            group_key = ("group_linear", bias is None)
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]):
        group_inputs = []
        group_weights = []
        group_biases = []
        group_nodes = []
        for node in subset:
            if CallFunctionVarArgs(aten.addmm.default).match(node):
                bias, input, weight = node.args
            else:
                if not CallFunctionVarArgs(aten.mm.default).match(node):
                    raise AssertionError(f"expected aten.mm node, got {node}")
                input, weight = node.args
                bias = None

            group_nodes.append(node)
            group_inputs.append(input)
            group_weights.append(weight)
            group_biases.append(bias)

        if all(bias is None for bias in group_biases):
            group_biases = None  # type: ignore[assignment]

        with graph.inserting_before(subset[0]):  # type: ignore[operator]
            fused_mm = graph.call_function(  # type: ignore[operator]
                torch.ops.fbgemm.gmm.default,
                args=(group_inputs, group_weights, group_biases),
                kwargs={"smart_fused": True},
            )

        for i, original_mm in enumerate(group_nodes):
            with graph.inserting_after(fused_mm):  # type: ignore[operator]
                new_mm = graph.call_function(operator.getitem, args=(fused_mm, i))  # type: ignore[operator]
            original_mm.replace_all_uses_with(new_mm)
            new_mm.meta.update(original_mm.meta)
            graph.erase_node(original_mm)  # type: ignore[operator]
        counters["inductor"]["group_linear"] += 1


class BatchPointwiseMathOpsPostGradFusion(BatchPointwiseOpsFusionFactory):
    """
    Batch pointwise math operator (e.g., add, mul) in post grad pass.
    """

    def __init__(self, op, **kwargs) -> None:
        super().__init__(op, **kwargs)
        self.op = op

    def _pointwise_node_can_be_fused(self, node: torch.fx.Node):
        # note: we only consider the case where the inputs are tensors
        # for mixed precision training, we need to make sure the inputs
        # of the aten.cat when do the stack should be the same dtype
        # otherwise, the output of the aten.cat may be not the same as
        # its inputs, and cause dtype not same error in mm or addmm
        input, other = node.args
        return (
            input.meta["val"].shape == other.meta["val"].shape  # type: ignore[union-attr]
            # input and other can be scalars, where they have no attribute 'meta'
            if hasattr(input, "meta")
            and hasattr(other, "meta")
            and is_node_meta_valid(input)  # type: ignore[arg-type, union-attr]
            and is_node_meta_valid(other)  # type: ignore[arg-type, union-attr]
            # torch.SymInt or torch.SymFloat object has no attribute 'shape'
            and isinstance(input.meta["val"], torch.Tensor)  # type: ignore[union-attr]
            and isinstance(other.meta["val"], torch.Tensor)  # type: ignore[union-attr]
            else False
        )

    def match(self, node: torch.fx.Node):
        if CallFunctionVarArgs(self.op).match(
            node
        ) and self._pointwise_node_can_be_fused(node):
            alpha = node.kwargs.get("alpha", DEFAULT_ALPHA)
            rounding_mode = node.kwargs.get("rounding_mode", None)
            input, other = node.args
            shape = list(input.meta["val"].shape)  # type: ignore[union-attr]
            if self.graph_search_options.get("fuse_nodes_with_same_parent", False):
                # only consider the linear case so far
                # pyre-fixme[16]
                if input.target is aten.select or other.target is aten.select:  # type: ignore[union-attr]
                    parent = (
                        # pyre-fixme[16]
                        input.args[0]  # type: ignore[union-attr]
                        # pyre-fixme[16]
                        if input.target is aten.select  # type: ignore[union-attr]
                        else other.args[0]  # type: ignore[union-attr]
                    )
                else:
                    parent = ""
            else:
                parent = ""
            group_key = (
                "batch_aten_" + self.op.__name__.lower().split(".")[0],
                str(shape),
                str(input.meta["val"].dtype),  # type: ignore[union-attr]
                str(other.meta["val"].dtype),  # type: ignore[union-attr]
                str(alpha),
                str(rounding_mode),
                str(parent),
            )
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]):
        batch_inputs, batch_others = [], []
        alpha = subset[0].kwargs.get("alpha", DEFAULT_ALPHA)
        batch_inputs_meta, batch_others_meta = [], []

        for node in subset:
            input, other = node.args
            batch_inputs.append(input)
            batch_others.append(other)
            batch_inputs_meta.append(input.meta)  # type: ignore[possibly-undefined, union-attr]
            batch_others_meta.append(other.meta)  # type: ignore[possibly-undefined, union-attr]

        with graph.inserting_before(subset[0]):  # type: ignore[operator]
            stack_inputs = decompose_stack(graph, batch_inputs)
            stack_others = decompose_stack(graph, batch_others)
            stack_inputs_meta = torch.stack(
                [input["val"] for input in batch_inputs_meta]
            )
            stack_others_meta = torch.stack(
                [other["val"] for other in batch_others_meta]
            )

            batch_op = graph.call_function(  # type: ignore[operator]
                self.op,
                args=(stack_inputs, stack_others),
                kwargs={"alpha": alpha} if self.op == aten.add.Tensor else {},
            )
            batch_op.meta["val"] = self.op(stack_inputs_meta, stack_others_meta)
            for i, original_add in enumerate(subset):
                with graph.inserting_after(batch_op):  # type: ignore[operator]
                    new_add = graph.call_function(  # type: ignore[operator]
                        torch.ops.aten.select, args=((batch_op, 0, i))
                    )
                original_add.replace_all_uses_with(new_add)
                new_add.meta.update(original_add.meta)
                graph.erase_node(original_add)  # type: ignore[operator]
        counters["inductor"][
            "batch_aten_" + self.op.__name__.lower().split(".")[0]
        ] += 1


@register_fusion("batch_linear_lhs")
class BatchLinearLHSFusion(BatchFusion):
    """
    Batch linear left-hand side fusion. This pass tries to fuse the following patterns:

        torch.nn.functional.linear(x, w1), linear(x, w2),... * linear(x, wn)
        -> torch.mm(x, torch.cat([w1, w2,... * wn]).transpose(0, 1))

    We have a separate pass to eliminate contiguous transpose in a generic way.
    """

    def match(self, node: torch.fx.Node) -> tuple[str, int | None, Any] | None:
        if CallFunctionVarArgs([torch.nn.functional.linear, torch._C._nn.linear]).match(
            node
        ) and is_linear_node_can_be_fused(node):
            input = get_arg_value(node, 0, "input")
            weight = get_arg_value(node, 1, "weight")
            bias = get_arg_value(node, 2, "bias")
            # Skip fusion when weight is a tensor subclass that can't handle aten.cat.
            # fuse() calls torch.cat on the weight example_values, which fails for
            # subclasses lacking aten.cat dispatch.  Probe it here so subclasses that
            # do implement aten.cat (e.g. future torchao versions) still get fused.
            weight_val = weight.meta.get("example_value", weight.meta.get("val"))
            if weight_val is not None and type(weight_val) not in (
                torch.Tensor,
                FakeTensor,
            ):
                try:
                    _ = torch.cat([weight_val[:1], weight_val[:1]], dim=0)
                except (RuntimeError, TypeError):
                    return None
            bias_tensor = None
            if bias is not None:
                bias_tensor = bias.meta.get("val", bias.meta.get("example_value"))
            bias_dim = None if bias_tensor is None else bias_tensor.ndim  # type: ignore[union-attr]
            group_key = ("batch_linear_lhs", bias_dim, input)
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]):
        batch_nodes = []
        batch_input = None
        batch_weights, batch_weights_meta = [], []
        batch_biases, batch_biases_meta = [], []
        split_sections = []
        for node in subset:
            input = get_arg_value(node, 0, "input")
            weight = get_arg_value(node, 1, "weight")
            bias = get_arg_value(node, 2, "bias")
            batch_nodes.append(node)
            if batch_input is None:
                batch_input = input
            else:
                if batch_input is not input:
                    raise AssertionError(
                        f"expected batch_input to be input, got {batch_input}"
                    )
            batch_weights.append(weight)
            batch_weights_meta.append(weight.meta["example_value"])
            if bias:
                batch_biases.append(bias)
                batch_biases_meta.append(bias.meta["example_value"])
            split_sections.append(weight.meta["example_value"].shape[0])

        with graph.inserting_before(subset[0]):  # type: ignore[operator]
            cat_weights = graph.call_function(  # type: ignore[operator]
                torch.cat, args=(batch_weights,), kwargs={"dim": 0}
            )
            cat_weights.meta["example_value"] = torch.cat(batch_weights_meta, dim=0)
            transposed_weights = graph.call_function(  # type: ignore[operator]
                torch.transpose, args=(cat_weights, 0, 1)
            )
            transposed_weights.meta["example_value"] = torch.transpose(
                cat_weights.meta["example_value"], 0, 1
            )
            if len(batch_biases) > 0:
                cat_biases = graph.call_function(  # type: ignore[operator]
                    torch.cat, args=(batch_biases,), kwargs={"dim": 0}
                )
                cat_biases.meta["example_value"] = torch.cat(batch_biases_meta, dim=0)
                fused_lhs = graph.call_function(  # type: ignore[operator]
                    torch.addmm,
                    args=(cat_biases, batch_input, transposed_weights),
                )
                fused_lhs.meta["example_value"] = torch.addmm(
                    cat_biases.meta["example_value"],
                    batch_input.meta["example_value"],  # type: ignore[union-attr]
                    transposed_weights.meta["example_value"],
                )
            else:
                fused_lhs = graph.call_function(  # type: ignore[operator]
                    torch.mm,
                    args=(batch_input, transposed_weights),
                )
                fused_lhs.meta["example_value"] = torch.mm(
                    batch_input.meta["example_value"],  # type: ignore[union-attr]
                    transposed_weights.meta["example_value"],
                )
            fused_lhs_list = graph.call_function(  # type: ignore[operator]
                torch.split, args=(fused_lhs, split_sections), kwargs={"dim": 1}
            )

        for i, node in enumerate(batch_nodes):
            with graph.inserting_after(fused_lhs_list):  # type: ignore[operator]
                new_node = graph.call_function(  # type: ignore[operator]
                    operator.getitem, args=(fused_lhs_list, i)
                )
            node.replace_all_uses_with(new_node)
            new_node.meta.update(node.meta)
            graph.erase_node(node)  # type: ignore[operator]
        counters["inductor"]["batch_linear_lhs"] += 1


# Poor person's check for if a node in the graph mutates its input.
# (the graph is torch IR, so we will see torch fns and python operators)
def _is_mutable_node(tgt):
    if str(tgt).endswith("_"):
        # e.g. torch.mul_, torch.Tensor.mul_
        return True
    if (
        hasattr(tgt, "__module__")
        and tgt.__module__ == "_operator"
        and tgt.__name__.startswith("i")
    ):
        # e.g. operator.iand, operator.imul
        return True
    return False


def is_linear_node_can_be_fused(node: torch.fx.Node):
    input = get_arg_value(node, 0, "input")
    weight = get_arg_value(node, 1, "weight")
    return (
        is_node_meta_valid(node)
        and is_node_meta_valid(input)
        and is_node_meta_valid(weight)
        and len(input.meta["example_value"].shape) == 2
        and len(weight.meta["example_value"].shape) == 2
        # the mm -> bmm transform adds an unbind() op,
        # which is not safe for autograd when the output of the mm is mutated.
        # don't pattern match if any users of the mm mutate the input.
        and not any(_is_mutable_node(user.target) for user in node.users)
    )


@register_fusion("batch_linear")
class PreGradBatchLinearFusion(BatchFusion):
    """
    Batch linear fusion in pre grad pass.
    Fuse linear with same size with torch.baddbmm
    """

    def _getitem_args(self, getitem_node: torch.fx.Node):
        if getitem_node.target != operator.__getitem__ or (
            getitem_node.op != "call_function"
        ):
            return None
        return getitem_node.args[0]

    def match(self, node: torch.fx.Node):
        if CallFunctionVarArgs([torch.nn.functional.linear, torch._C._nn.linear]).match(
            node
        ) and is_linear_node_can_be_fused(node):
            input = get_arg_value(node, 0, "input")
            weight = get_arg_value(node, 1, "weight")
            bias = get_arg_value(node, 2, "bias")
            if self.graph_search_options.get("fuse_nodes_with_same_users", False):
                users = [user.target for user in node.users]
            else:
                users = ""  # type: ignore[assignment]
            group_key = (
                "batch_linear",
                self._getitem_args(input),
                str(input.meta["example_value"].shape),
                str(weight.meta["example_value"].shape),
                bias is None,
                str(users),
            )
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]):
        batch_nodes = []
        batch_inputs = []
        batch_weights = []
        batch_biases = []
        batch_inputs_metadata = []
        batch_weights_metadata = []
        batch_biases_metadata = []
        for node in subset:
            batch_nodes.append(node)
            input = get_arg_value(node, 0, "input")
            batch_inputs.append(input)
            batch_inputs_metadata.append(input.meta["example_value"])
            weight = get_arg_value(node, 1, "weight")
            batch_weights.append(weight)
            batch_weights_metadata.append(weight.meta["example_value"])
            bias = get_arg_value(node, 2, "bias")
            batch_biases.append(bias)
            if bias is not None and hasattr(bias, "meta"):
                batch_biases_metadata.append(bias.meta["example_value"])

        with graph.inserting_before(subset[0]):  # type: ignore[operator]
            stack_inputs = graph.call_function(  # type: ignore[operator]
                torch.stack, args=(batch_inputs,), kwargs={"dim": 0}
            )
            update_stack_example_value(stack_inputs, batch_inputs_metadata)
            stack_weights = graph.call_function(  # type: ignore[operator]
                torch.stack, args=(batch_weights,), kwargs={"dim": 0}
            )
            update_stack_example_value(stack_weights, batch_weights_metadata)
            transpose_weight = graph.call_function(  # type: ignore[operator]
                torch.transpose, args=(stack_weights, 1, 2)
            )
            transpose_weight.meta["example_value"] = torch.transpose(
                stack_weights.meta["example_value"], 1, 2
            )
            if all(bias is None for bias in batch_biases):
                bmm = graph.call_function(  # type: ignore[operator]
                    torch.bmm,
                    args=(stack_inputs, transpose_weight),
                )
                bmm.meta["example_value"] = torch.bmm(
                    stack_inputs.meta["example_value"],
                    transpose_weight.meta["example_value"],
                )
                bmm_meta = bmm.meta["example_value"]
            else:
                stack_biases = graph.call_function(  # type: ignore[operator]
                    torch.stack, args=(batch_biases,), kwargs={"dim": 0}
                )
                update_stack_example_value(stack_biases, batch_biases_metadata)
                unsqueeze_biases = graph.call_function(  # type: ignore[operator]
                    torch.unsqueeze, args=(stack_biases, 1)
                )
                unsqueeze_biases.meta["example_value"] = torch.unsqueeze(
                    stack_biases.meta["example_value"], 1
                )
                bmm = graph.call_function(  # type: ignore[operator]
                    torch.baddbmm,
                    args=(unsqueeze_biases, stack_inputs, transpose_weight),
                )
                try:
                    # it will have runtime error to broadcast when it has dynamic shape included
                    # in the meta data, so we need to skip the update meta data
                    bmm.meta["example_value"] = torch.baddbmm(
                        unsqueeze_biases.meta["example_value"],
                        stack_inputs.meta["example_value"],
                        transpose_weight.meta["example_value"],
                    )
                    bmm_meta = bmm.meta["example_value"]
                except Exception as e:
                    log.debug(
                        f" exception when update bmm meta data with stack error traceback {e}"  # noqa: G004
                    )
                    bmm_meta = None

            bmm = graph.call_function(torch.unbind, args=(bmm,), kwargs={"dim": 0})  # type: ignore[operator]
            if bmm_meta is not None:
                bmm.meta["example_value"] = torch.unbind(bmm_meta, dim=0)
            for i, linear in enumerate(batch_nodes):
                with graph.inserting_after(bmm):  # type: ignore[operator]
                    getitem = graph.call_function(operator.getitem, args=(bmm, i))  # type: ignore[operator]
                linear.replace_all_uses_with(getitem)
                getitem.meta.update(linear.meta)
                graph.erase_node(linear)  # type: ignore[operator]
        counters["inductor"]["batch_linear"] += 1


# Profitability gates for cat_linear (opt-in; numbers come from the sweep in the
# PR description). The split only wins for a narrow output fed by a large,
# memory-bound concat, so gate on output width and concat traffic.
# Widest win output is 96; first loss at N=128.
CAT_LINEAR_MAX_OUTPUT_WIDTH = 96
# Only the concat's M*K_total*dtype_bytes of HBM traffic is removed, so it has to
# cover the per-piece launch + reduce. Bytes, so fp32 trips it at half the bf16
# element count.
CAT_LINEAR_CAT_TRAFFIC_FLOOR_BYTES = 512 * 1024 * 1024
# Skip slivers: a piece this thin is a launch-bound matmul with no upside.
CAT_LINEAR_MIN_PIECE_WIDTH = 8
# More parts means more launches and reduce-adds; >3 lost in the sweep.
CAT_LINEAR_MAX_PARTS = 3


def _weight_hoist_anchor(graph, weight):
    # Insert hoisted slices after the weight's own def, or after the last
    # placeholder when it's a lifted input (can't insert a call between placeholders).
    if weight.op == "placeholder":
        anchor = weight
        for n in graph.nodes:
            if n.op == "placeholder":
                anchor = n
            else:
                break
        return anchor
    return weight


def _meta_shape(node):
    if not isinstance(node, torch.fx.Node) or not is_node_meta_valid(node):
        return None
    val = node.meta.get("example_value")
    if val is None:
        val = node.meta.get("val")
    if val is None or not hasattr(val, "shape"):
        return None
    shape = val.shape
    # static shapes only: bail on symbolic so we never specialize a backed
    # SymInt or trip over an unbacked one (same free_symbols guard split_cat uses)
    if free_symbols(shape):
        return None
    return tuple(int(s) for s in shape)


def match_cat_linear(node: torch.fx.Node):
    """Match linear(cat([parts...], dim=-1), weight, bias).

    Returns (parts, weight, bias, offsets) or None. Conservative on purpose:
    last-dim cat only, total width and part count capped, no thin slivers, and
    a weight whose contraction dim lines up with the cat. The cat must have a
    single user so it can actually be erased once the linear is rewritten.
    """
    cat_in = get_arg_value(node, 0, "input")
    weight = get_arg_value(node, 1, "weight")
    bias = get_arg_value(node, 2, "bias")
    if not isinstance(cat_in, torch.fx.Node) or not CallFunctionVarArgs(
        [torch.cat]
    ).match(cat_in):
        return None
    if len(cat_in.users) != 1:
        return None

    parts = get_arg_value(cat_in, 0, "tensors")
    if not isinstance(parts, (list, tuple)) or not (
        2 <= len(parts) <= CAT_LINEAR_MAX_PARTS
    ):
        return None
    if not all(isinstance(p, torch.fx.Node) for p in parts):
        return None

    cat_shape = _meta_shape(cat_in)
    if cat_shape is None or len(cat_shape) < 2:
        return None
    rank = len(cat_shape)
    cat_dim = get_arg_value(cat_in, 1, "dim")
    if cat_dim is None:
        cat_dim = 0
    if not isinstance(cat_dim, int):
        return None
    if (cat_dim + rank if cat_dim < 0 else cat_dim) != rank - 1:
        return None

    k_total = int(cat_shape[-1])

    offsets = [0]
    for p in parts:
        sh = _meta_shape(p)
        if sh is None or len(sh) != rank:
            return None
        k_i = int(sh[-1])
        if k_i < CAT_LINEAR_MIN_PIECE_WIDTH:
            return None
        offsets.append(offsets[-1] + k_i)
    if offsets[-1] != k_total:
        return None

    w_shape = _meta_shape(weight)
    if w_shape is None or len(w_shape) != 2 or int(w_shape[-1]) != k_total:
        return None

    # N is the profit variable; wide outputs are compute-bound and lose the split.
    n_out = int(w_shape[0])
    if n_out > CAT_LINEAR_MAX_OUTPUT_WIDTH:
        return None

    # traffic = prod(leading dims) * k_total * dtype_bytes.
    cat_val = cat_in.meta.get("example_value")
    if cat_val is None:
        cat_val = cat_in.meta.get("val")
    item_bytes = getattr(getattr(cat_val, "dtype", None), "itemsize", 2)
    m_lead = 1
    for d in cat_shape[:-1]:
        m_lead *= int(d)
    if m_lead * k_total * item_bytes < CAT_LINEAR_CAT_TRAFFIC_FLOOR_BYTES:
        return None

    return parts, weight, bias, offsets


@register_fusion("cat_linear")
class CatLinearFusion(BatchFusion):
    """
    Rewrite linear(cat([x0, x1, ...], dim=-1), weight, bias) into a sum of
    per-piece linears on contiguous last-dim weight slices, so the concatenated
    activation is never materialised in forward or backward.

    Unlike the other fusions here this is a per-node rewrite (one linear(cat)
    splits into several smaller linears) rather than a batch of like-shaped ops,
    so every match forms its own group of one. Off by default; opt in via
    pre_grad_fusion_options["cat_linear"].
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.graph_search_options = {
            **self.graph_search_options,
            "min_fuse_set_size": 1,
        }
        # Dedup hoisted weight slices across the several fuse() calls this rule
        # makes on one graph (weight-tied blocks match at multiple sites). A fresh
        # rule instance is built per compile in generate_fusion_from_config and
        # only ever runs on a single graph, so an instance-level cache is safe.
        self._piece_cache: dict = {}

    def match(self, node: torch.fx.Node):
        if not CallFunctionVarArgs(
            [torch.nn.functional.linear, torch._C._nn.linear]
        ).match(node):
            return None
        if not is_node_meta_valid(node):
            return None
        # match() runs on every linear during the candidate BFS, including nodes
        # that never fuse, so keep it side-effect-free: recompute in fuse() rather
        # than stashing on node.meta (which would leave stale meta behind).
        if match_cat_linear(node) is None:
            return None
        return ("cat_linear", node)

    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]):
        # Slice+clone each weight once and hoist to entry, so a weight matched by
        # several cat_linears (weight-tied blocks) isn't re-sliced at every site.
        cache = self._piece_cache

        for node in subset:
            matched = match_cat_linear(node)
            if matched is None:
                continue
            parts, weight, bias, offsets = matched
            weight_val = weight.meta.get("example_value", weight.meta.get("val"))
            bias_val = None
            if bias is not None:
                bias_val = bias.meta.get("example_value", bias.meta.get("val"))
            anchor = _weight_hoist_anchor(graph, weight)

            with graph.inserting_before(node):  # type: ignore[operator]
                partials = []
                for i, part in enumerate(parts):
                    key = (weight, offsets[i], offsets[i + 1])
                    w_cont = cache.get(key)
                    if w_cont is None:
                        # slice is a differentiable view of the full weight; the
                        # clone makes each GEMM operand contiguous. Folds to a const
                        # when frozen; in training it's a cheap weight-only op.
                        # pyrefly: ignore [not-callable]
                        with graph.inserting_after(anchor):
                            w_slice = graph.call_function(  # type: ignore[operator]
                                aten.slice.Tensor,
                                args=(weight, 1, offsets[i], offsets[i + 1]),
                            )
                        # pyrefly: ignore [not-callable]
                        with graph.inserting_after(w_slice):
                            w_cont = graph.call_function(  # type: ignore[operator]
                                aten.clone.default,
                                args=(w_slice,),
                                kwargs={"memory_format": torch.contiguous_format},
                            )
                        if weight_val is not None:
                            sv = aten.slice.Tensor(  # type: ignore[operator]
                                weight_val, 1, offsets[i], offsets[i + 1]
                            )
                            w_slice.meta["example_value"] = sv
                            w_cont.meta["example_value"] = sv.contiguous()  # type: ignore[operator]
                        cache[key] = w_cont
                    # bias rides on the first piece only; folding it into the
                    # reduce-add would just be an extra node.
                    out_i = graph.call_function(  # type: ignore[operator]
                        torch.nn.functional.linear,
                        args=(part, w_cont, bias if i == 0 else None),
                    )
                    # Each synthesized node gets its own fake val computed from its
                    # own operands, the way BatchLinearLHSFusion does. The old code
                    # aliased the whole linear's fake tensor onto every piece and
                    # every reduce-add, sharing one FakeTensor object across
                    # distinct nodes.
                    part_val = part.meta.get("example_value", part.meta.get("val"))
                    w_cont_val = w_cont.meta.get("example_value")
                    if part_val is not None and w_cont_val is not None:
                        out_i.meta["example_value"] = torch.nn.functional.linear(
                            part_val, w_cont_val, bias_val if i == 0 else None
                        )
                    partials.append(out_i)

                acc = partials[0]
                for out_i in partials[1:]:
                    prev = acc
                    acc = graph.call_function(operator.add, args=(prev, out_i))  # type: ignore[operator]
                    prev_val = prev.meta.get("example_value")
                    out_val = out_i.meta.get("example_value")
                    if prev_val is not None and out_val is not None:
                        acc.meta["example_value"] = torch.add(prev_val, out_val)

            node.replace_all_uses_with(acc)
            graph.erase_node(node)  # type: ignore[operator]
            counters["inductor"]["cat_linear"] += 1


@register_fusion("batch_layernorm")
class BatchLayernormFusion(BatchFusion):
    """
    Batch layer norm fusion in pre grad pass
    """

    def match(self, node: torch.fx.Node):
        if CallFunctionVarArgs(torch.nn.functional.layer_norm).match(node):
            input = get_arg_value(node, 0, "input")
            weight = get_arg_value(node, 2, "weight")
            bias = get_arg_value(node, 3, "bias")
            if self.graph_search_options.get("fuse_nodes_with_same_users", False):
                users = [user.target for user in node.users]
            else:
                users = ""  # type: ignore[assignment]
            group_key = (
                (
                    "batch_layernorm",
                    str(input.meta["example_value"].shape),
                    str(weight.meta["example_value"].shape)
                    if weight is not None
                    else "",
                    str(bias.meta["example_value"].shape) if bias is not None else "",
                    str(get_arg_value(node, 1, "normalized_shape")),
                    str(get_arg_value(node, 4, "eps")),
                    str(users),
                )
                if "example_value" in input.meta
                and is_node_meta_valid(weight)
                and is_node_meta_valid(bias)
                else None
            )
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]):
        group_inputs = []
        group_shapes = []
        group_weights = []
        group_biases = []
        group_epss = []
        group_nodes = []
        group_inputs_metadata = []
        group_biases_metadata = []
        group_weights_metadata = []
        for node in subset:
            group_nodes.append(node)
            input = get_arg_value(node, 0, "input")
            group_inputs.append(input)
            group_inputs_metadata.append(input.meta["example_value"])
            group_shapes.append(get_arg_value(node, 1, "normalized_shape"))
            weight = get_arg_value(node, 2, "weight")
            group_weights.append(weight)
            if weight is not None and hasattr(weight, "meta"):
                group_weights_metadata.append(weight.meta["example_value"])
            bias = get_arg_value(node, 3, "bias")
            group_biases.append(bias)
            if bias is not None and hasattr(bias, "meta"):
                group_biases_metadata.append(bias.meta["example_value"])
            eps = get_arg_value(node, 4, "eps")
            if eps is None:
                eps = 1e-5
            group_epss.append(eps)
        stack_dim = -1 - len(group_shapes[-1])

        if all(bias is None for bias in group_biases):
            group_biases = None  # type: ignore[assignment]
        if all(weight is None for weight in group_weights):
            group_weights = None  # type: ignore[assignment]
        if not all(eps == group_epss[0] for eps in group_epss):
            raise AssertionError("all epsilon values must be equal")

        with graph.inserting_before(subset[0]):  # type: ignore[operator]
            stack_input = graph.call_function(  # type: ignore[operator]
                torch.stack, args=(group_inputs,), kwargs={"dim": stack_dim}
            )
            update_stack_example_value(stack_input, group_inputs_metadata, stack_dim)
            if group_weights is not None:
                stack_weight = graph.call_function(  # type: ignore[operator]
                    torch.stack, args=(group_weights,), kwargs={"dim": 0}
                )
                update_stack_example_value(stack_weight, group_weights_metadata)
            else:
                stack_weight = None
            if group_biases is not None:
                stack_bias = graph.call_function(  # type: ignore[operator]
                    torch.stack, args=(group_biases,), kwargs={"dim": 0}
                )
                update_stack_example_value(stack_bias, group_biases_metadata)
            else:
                stack_bias = None

            batch_layer_norm = graph.call_function(  # type: ignore[operator]
                torch.nn.functional.layer_norm,
                args=(stack_input, group_shapes[-1]),
                kwargs={"eps": group_epss[-1]},
            )
            batch_layer_norm.meta["example_value"] = stack_input.meta["example_value"]

            if group_weights is not None and group_biases is not None:
                previous_batch_layer_norm_meta = batch_layer_norm.meta["example_value"]
                batch_layer_norm = graph.call_function(  # type: ignore[operator]
                    torch.mul, args=(stack_weight, batch_layer_norm)
                )
                update_pointwise_example_value(
                    batch_layer_norm,
                    # pyrefly: ignore [missing-attribute]
                    stack_weight.meta["example_value"],
                    previous_batch_layer_norm_meta,
                    torch.mul,
                )
                previous_batch_layer_norm_meta = batch_layer_norm.meta["example_value"]
                batch_layer_norm = graph.call_function(  # type: ignore[operator]
                    torch.add, args=(stack_bias, batch_layer_norm)
                )
                update_pointwise_example_value(
                    batch_layer_norm,
                    # pyrefly: ignore [missing-attribute]
                    stack_bias.meta["example_value"],
                    previous_batch_layer_norm_meta,
                    torch.add,
                )
            elif group_weights is not None and group_biases is None:
                previous_batch_layer_norm_meta = batch_layer_norm.meta["example_value"]
                # pyrefly: ignore [not-callable]
                batch_layer_norm = graph.call_function(
                    torch.mul, args=(stack_weight, batch_layer_norm)
                )
                update_pointwise_example_value(
                    batch_layer_norm,
                    # pyrefly: ignore [missing-attribute]
                    stack_weight.meta["example_value"],
                    previous_batch_layer_norm_meta,
                    torch.mul,
                )
            elif group_weights is None and group_biases is not None:
                previous_batch_layer_norm_meta = batch_layer_norm.meta["example_value"]
                # pyrefly: ignore [not-callable]
                batch_layer_norm = graph.call_function(
                    torch.add, args=(stack_bias, batch_layer_norm)
                )
                update_pointwise_example_value(
                    batch_layer_norm,
                    # pyrefly: ignore [missing-attribute]
                    stack_bias.meta["example_value"],
                    previous_batch_layer_norm_meta,
                    torch.add,
                )

            batch_layer_norm_unbind = graph.call_function(  # type: ignore[operator]
                torch.unbind,
                args=(batch_layer_norm,),
                kwargs={"dim": stack_dim},
            )
            update_stack_example_value(
                batch_layer_norm_unbind,
                batch_layer_norm.meta["example_value"],
                op=torch.unbind,
                dim=stack_dim,
            )

        for i, node in enumerate(group_nodes):
            with graph.inserting_after(batch_layer_norm_unbind):  # type: ignore[operator]
                new_node = graph.call_function(  # type: ignore[operator]
                    operator.getitem, args=(batch_layer_norm_unbind, i)
                )
            node.replace_all_uses_with(new_node)
            new_node.meta.update(node.meta)
            graph.erase_node(node)  # type: ignore[operator]
        counters["inductor"]["batch_layernorm"] += 1


class BatchPointwiseOpsPreGradFusion(BatchPointwiseOpsFusionFactory):
    """
    Batch pointwise ops (e.g., sigmoid, relu, tanh) fusion in pre grad pass.
    We fuse it in random place, and the introduced stack node may be merged in split cat.
    """

    def __init__(self, op, **kwargs) -> None:
        super().__init__(op, **kwargs)
        self.op = op

    def match(self, node: torch.fx.Node):
        input = get_arg_value(node, 0, "input")
        if CallFunctionVarArgs(self.op).match(node) and is_node_meta_valid(node):
            if self.graph_search_options.get("fuse_nodes_with_same_parent", False):
                # pyre-fixme[16]
                parent = node.args[0]
                parent = parent.target if parent is not None else ""  # type: ignore[union-attr]
            else:
                parent = ""
            # for relu op, we also use the inplace to construct the key
            group_key = (
                "batch_" + self.op.__name__.lower().split(".")[0],
                str(input.meta["example_value"].shape),
                str(node.kwargs.get("inplace", False)),
                str(parent),
            )
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]):
        batch_nodes = []
        batch_inputs = []
        batch_inputs_metadata = []

        for node in subset:
            batch_nodes.append(node)
            input = get_arg_value(node, 0, "input")
            batch_inputs.append(input)
            batch_inputs_metadata.append(input.meta["example_value"])

        with graph.inserting_before(subset[0]):  # type: ignore[operator]
            stack_inputs = graph.call_function(  # type: ignore[operator]
                torch.stack, args=(batch_inputs,), kwargs={"dim": 0}
            )
            update_stack_example_value(stack_inputs, batch_inputs_metadata)
            if self.op is torch.nn.functional.relu:
                batch_op = graph.call_function(  # type: ignore[operator]
                    self.op,
                    args=(stack_inputs,),
                    kwargs={"inplace": subset[0].kwargs.get("inplace", False)},
                )
                batch_op.meta["example_value"] = self.op(
                    stack_inputs.meta["example_value"],
                    # pyrefly: ignore [bad-argument-type]
                    inplace=subset[0].kwargs.get("inplace", False),
                )
            else:
                batch_op = graph.call_function(  # type: ignore[operator]
                    self.op,
                    args=(stack_inputs,),
                )
                batch_op.meta["example_value"] = self.op(
                    stack_inputs.meta["example_value"]
                )
            unbind_op = graph.call_function(  # type: ignore[operator]
                torch.unbind, args=(batch_op,), kwargs={"dim": 0}
            )
            unbind_op.meta["example_value"] = torch.unbind(
                batch_op.meta["example_value"], dim=0
            )
            for i, node in enumerate(batch_nodes):
                with graph.inserting_after(unbind_op):  # type: ignore[operator]
                    getitem = graph.call_function(operator.getitem, args=(unbind_op, i))  # type: ignore[operator]
                node.replace_all_uses_with(getitem)
                getitem.meta.update(node.meta)
                graph.erase_node(node)  # type: ignore[operator]
        counters["inductor"]["batch_" + self.op.__name__.lower().split(".")[0]] += 1


class BatchPointwiseOpsPostGradFusion(BatchPointwiseOpsFusionFactory):
    """
    Batch pointwise ops (e.g., sigmoid, relu, tanh) fusion in post grad pass.
    The introduced stack node may be merged in split cat.
    """

    def __init__(self, op, **kwargs) -> None:
        super().__init__(op, **kwargs)
        self.op = op

    def match(self, node: torch.fx.Node):
        input = get_arg_value(node, 0, "input")
        if CallFunctionVarArgs(self.op).match(node) and is_node_meta_valid(node):
            # for relu op, we also use the inplace to construct the key
            # we batch the ops with same parent to enable followup split cat
            parent = node.args[0]
            parent = (
                parent.target  # type: ignore[union-attr]
                if self.graph_search_options.get("fuse_nodes_with_same_parent", False)
                else ""
            )
            group_key = (
                "batch_aten_" + self.op.__name__.lower().split(".")[0],
                str(input.meta["val"].shape),
                str(node.kwargs.get("inplace", False)),
                # pyre-fixme[16]
                str(parent),
            )
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]):
        batch_nodes = []
        batch_inputs = []
        batch_inputs_metadata = []

        for node in subset:
            batch_nodes.append(node)
            input = get_arg_value(node, 0, "input")
            batch_inputs.append(input)
            batch_inputs_metadata.append(input.meta["val"])

        with graph.inserting_before(subset[0]):  # type: ignore[operator]
            stack_inputs = decompose_stack(graph, batch_inputs)
            update_stack_example_value(stack_inputs, batch_inputs_metadata)
            batch_op = graph.call_function(  # type: ignore[operator]
                self.op,
                args=(stack_inputs,),
            )
            for i, node in enumerate(batch_nodes):
                with graph.inserting_after(batch_op):  # type: ignore[operator]
                    getitem = graph.call_function(aten.select, args=(batch_op, 0, i))  # type: ignore[operator]
                node.replace_all_uses_with(getitem)
                getitem.meta.update(node.meta)
                graph.erase_node(node)  # type: ignore[operator]
        counters["inductor"][
            "batch_aten_" + self.op.__name__.lower().split(".")[0]
        ] += 1


class BatchMathOpsPreGradFusion(BatchPointwiseOpsFusionFactory):
    """
    Batch simple math related ops such as nan_to_num in pre grad pass.
    """

    def __init__(self, op, **kwargs):
        super().__init__(op, **kwargs)
        self.op = op

    def match(self, node: torch.fx.Node):
        input = get_arg_value(node, 0, "input")
        if self._match_op(node) and is_node_meta_valid(node):
            # check the input has the same shape and its users have the same target
            # check all clamp operators have the same min and max values, and
            # nan_to_num operators use the same default value.
            child = next(iter(node.users.keys()))
            group_key = (
                str(input.meta["example_value"].shape)
                + str(node.args[1:])
                + str(node.kwargs)
                + str(child.target)
            )
        else:
            group_key = None
        return group_key

    def _match_op(self, node: torch.fx.Node) -> MatchResult:
        return CallFunctionVarArgs(self.op).match(node)

    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]):
        batch_nodes = []
        batch_inputs = []
        batch_inputs_metadata = []
        args = subset[0].args[1:]
        kwargs = subset[0].kwargs

        for node in subset:
            batch_nodes.append(node)
            input = get_arg_value(node, 0, "input")
            batch_inputs.append(input)
            batch_inputs_metadata.append(input.meta["example_value"])

        with graph.inserting_before(subset[0]):  # type: ignore[operator]
            stack_inputs = graph.call_function(  # type: ignore[operator]
                torch.stack, args=(batch_inputs,), kwargs={"dim": 0}
            )
            update_stack_example_value(stack_inputs, batch_inputs_metadata)
            batch_op = graph.call_function(  # type: ignore[operator]
                self.op,
                args=(stack_inputs, *args),
                kwargs=kwargs,
            )
            batch_op.meta["example_value"] = self.op(
                stack_inputs.meta["example_value"], *args, **kwargs
            )
            unbind_op = graph.call_function(  # type: ignore[operator]
                torch.unbind, args=(batch_op,), kwargs={"dim": 0}
            )
            unbind_op.meta["example_value"] = torch.unbind(
                batch_op.meta["example_value"], dim=0
            )
            for i, node in enumerate(batch_nodes):
                with graph.inserting_after(unbind_op):  # type: ignore[operator]
                    getitem = graph.call_function(operator.getitem, args=(unbind_op, i))  # type: ignore[operator]
                node.replace_all_uses_with(getitem)
                getitem.meta.update(node.meta)
                graph.erase_node(node)  # type: ignore[operator]
        counters["inductor"]["batch_" + self.op.__name__.lower().split(".")[0]] += 1


@register_fusion("batch_tanh")
class BatchTanhPreGradFusion(BatchPointwiseOpsPreGradFusion):
    def __init__(self, **kwargs) -> None:
        super().__init__(torch.tanh, **kwargs)


@register_fusion("batch_sigmoid")
class BatchSigmoidPreGradFusion(BatchPointwiseOpsPreGradFusion):
    def __init__(self, **kwargs) -> None:
        super().__init__(torch.sigmoid, **kwargs)


@register_fusion("batch_relu")
class BatchReLuPreGradFusion(BatchPointwiseOpsPreGradFusion):
    def __init__(self, **kwargs) -> None:
        super().__init__(torch.nn.functional.relu, **kwargs)


@register_fusion("batch_detach")
class BatchDetachPreGradFusion(BatchMathOpsPreGradFusion):
    def __init__(self, **kwargs):
        super().__init__(torch.detach, **kwargs)

    def _match_op(self, node: torch.fx.Node) -> MatchResult:
        return CallFunctionVarArgs(torch.detach).match(node) or CallMethodVarArgs(
            "detach"
        ).match(node)


@register_fusion("batch_nan_to_num")
class BatchNanToNumPreGradFusion(BatchMathOpsPreGradFusion):
    def __init__(self, **kwargs):
        super().__init__(torch.nan_to_num, **kwargs)


@register_fusion("batch_clamp")
class BatchClampPreGradFusion(BatchMathOpsPreGradFusion):
    def __init__(self, **kwargs):
        super().__init__(torch.clamp, **kwargs)


@register_fusion("batch_dropout")
class BatchDropoutPreGradFusion(BatchMathOpsPreGradFusion):
    def __init__(self, **kwargs):
        super().__init__(torch.nn.functional.dropout, **kwargs)


@register_fusion("batch_aten_tanh", pre_grad=False)
class BatchTanhPostGradFusion(BatchPointwiseOpsPostGradFusion):
    def __init__(self, **kwargs) -> None:
        super().__init__(aten.tanh.default, **kwargs)


@register_fusion("batch_aten_sigmoid", pre_grad=False)
class BatchSigmoidPostGradFusion(BatchPointwiseOpsPostGradFusion):
    def __init__(self, **kwargs) -> None:
        super().__init__(aten.sigmoid.default, **kwargs)


@register_fusion("batch_aten_relu", pre_grad=False)
class BatchReLuPostGradFusion(BatchPointwiseOpsPostGradFusion):
    def __init__(self, **kwargs) -> None:
        super().__init__(aten.relu.default, **kwargs)


@register_fusion("batch_aten_add", pre_grad=False)
class BatchAddPostGradFusion(BatchPointwiseMathOpsPostGradFusion):
    def __init__(self, **kwargs) -> None:
        super().__init__(aten.add.Tensor, **kwargs)


@register_fusion("batch_aten_sub", pre_grad=False)
class BatchSubPostGradFusion(BatchPointwiseMathOpsPostGradFusion):
    def __init__(self, **kwargs) -> None:
        super().__init__(aten.sub.Tensor, **kwargs)


@register_fusion("batch_aten_div", pre_grad=False)
class BatchDivPostGradFusion(BatchPointwiseMathOpsPostGradFusion):
    def __init__(self, **kwargs) -> None:
        super().__init__(aten.div.Tensor, **kwargs)


@register_fusion("batch_aten_mul", pre_grad=False)
class BatchMulPostGradFusion(BatchPointwiseMathOpsPostGradFusion):
    def __init__(self, **kwargs) -> None:
        super().__init__(aten.mul.Tensor, **kwargs)


class _OrderedSet:
    def __init__(self, param=None) -> None:
        if param:
            self.rep = OrderedDict(dict.fromkeys(param))
        else:
            self.rep = OrderedDict()

    def __contains__(self, o) -> bool:
        return o in self.rep

    def __len__(self) -> int:
        return self.rep.__len__()

    def append(self, o):
        self.rep[o] = None

    def __iter__(self):
        return self.rep.keys().__iter__()


def find_independent_subset_greedy(
    node_list: Iterable[torch.fx.Node],
    graph_search_options: dict[str, Any],
) -> Iterator[Iterable[torch.fx.Node]]:
    """
    Yields a list of subsets of `node_list` where no element in the subset
    depends on any other element in the subset. This results in a set of
    independent nodes which can be fused together.

    The order of `node_list` is preserved within each subset so we can benefit
    from split-cat elimination in later passes.

    During iteration it is only safe to mutate the graph by changing the nodes
    that have been returned.

    graph_search_options:
      - min_fuse_set_size: Minimum size of the subset to consider. Subsets below
        this size will be ignored.
      - max_fuse_set_size: Maximum size of the subset to consider. Subsets will
        be broken to be at most this size.
    """

    # Compute all the children of `node` which are members of
    # `interesting_nodes`.
    def find_dependent_nodes(node, interesting_nodes):
        visited_node_set = OrderedSet[torch.fx.Node]()
        dep_set = OrderedSet[torch.fx.Node]()

        work = [node]
        while work:
            node = work.pop()
            for input_node in node.all_input_nodes:
                if input_node in interesting_nodes:
                    dep_set.add(input_node)

                if input_node not in visited_node_set:
                    visited_node_set.add(input_node)
                    work.append(input_node)

        return dep_set

    min_fuse_set_size = graph_search_options["min_fuse_set_size"]
    max_fuse_set_size = graph_search_options["max_fuse_set_size"]

    # node_list needs to be a set because we only track the nodes that are left
    # in it (and we want to do the `in` on a set, not a list). But we want to
    # keep the correct order.
    node_list = _OrderedSet(node_list)

    cache: dict[torch.fx.Node, OrderedSet[torch.fx.Node]] = {}
    while node_list:
        subset: list[torch.fx.Node] = []
        subset_deps = OrderedSet[torch.fx.Node]()

        next_round_node_list = _OrderedSet()
        for node in node_list:
            if len(subset) >= max_fuse_set_size or node in subset_deps:
                next_round_node_list.append(node)
                continue

            dep_set = cache.pop(node, None)
            if dep_set is None:
                dep_set = find_dependent_nodes(node, node_list)

            if not dep_set.intersection(subset):
                subset.append(node)
                subset_deps.update(dep_set)
            else:
                next_round_node_list.append(node)
                cache[node] = dep_set

        if len(subset) >= min_fuse_set_size:
            # Careful here - the caller uses the subsets to fuse nodes together
            # so we need to clear any cache entry that contains one of the
            # returned nodes because the dependency list could be different
            # (larger) after the merge.
            cache = {k: v for k, v in cache.items() if v.isdisjoint(subset)}
            yield subset

        node_list = next_round_node_list


def get_fusion_candidates(
    rule: GroupBatchFusionBase,
    root_node: torch.fx.Node,
    fused_set: OrderedSet[torch.fx.Node],
) -> collections.defaultdict[Any, list[torch.fx.Node]]:
    """
    Search fusion candidates for a specific rule using BFS starting from the root node.
    We only search the subgraph within graph_search_options["max_fuse_search_depth"].
    """
    q: collections.deque[tuple[int, torch.fx.Node]] = collections.deque()

    candidate_dict: collections.defaultdict[Any, list[torch.fx.Node]] = (
        collections.defaultdict(list)
    )

    if root_node.target in SEARCH_EXCLUSIONS:
        return candidate_dict

    visited_set = OrderedSet[torch.fx.Node]()

    for next_node in root_node.all_input_nodes:
        q.append((1, next_node))
        visited_set.add(next_node)

    while len(q) > 0:
        depth, node = q.popleft()

        if node in fused_set:
            continue

        key = rule.match(node)
        if key is not None:
            candidate_nodes = candidate_dict[key]
            if node not in candidate_nodes:
                candidate_nodes.append(node)
        else:
            if depth < rule.graph_search_options["max_fuse_search_depth"]:
                for next_node in node.all_input_nodes:
                    if next_node not in visited_set:
                        visited_set.add(next_node)
                        q.append((depth + 1, next_node))

    return candidate_dict


def apply_group_batch_fusion(graph: torch.fx.GraphModule, rule: GroupBatchFusionBase):
    stable_topological_sort(graph)  # type: ignore[arg-type]
    fused_set = OrderedSet[torch.fx.Node]()
    log_to_scuba = False

    for node in reversed(graph.nodes):  # type: ignore[arg-type]
        candidates = get_fusion_candidates(rule, node, fused_set)

        for key, candidate_nodes in candidates.items():
            if len(candidate_nodes) < rule.graph_search_options["min_fuse_set_size"]:
                continue

            for subset in find_independent_subset_greedy(
                candidate_nodes, rule.graph_search_options
            ):
                rule.fuse(graph, subset)
                fused_set.update(subset)
                log.debug(
                    f"{rule.__class__.__name__}: key = {key}; subset size = {len(list(subset))}"  # noqa: G004
                )
                log_to_scuba = True
    if log_to_scuba:
        from torch.fx._lazy_graph_module import _LazyGraphModule

        # Force graph to re-compile otherwise the output python code may be broken
        gm = graph._owning_module
        if isinstance(gm, _LazyGraphModule):
            _LazyGraphModule.recompile()
        else:
            if not isinstance(gm, torch.fx.GraphModule):
                raise AssertionError(f"expected torch.fx.GraphModule, got {type(gm)}")
            gm.recompile()
        graph_str = gm.print_readable(
            print_output=False, include_stride=True, include_device=True
        )

        name = f"optimus_{str(rule.__class__.__name__)}"
        if "MTIA" in name:
            name = f"cff_{str(rule.__class__.__name__)}"
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": name,
                "encoding": "string",
            },
            payload_fn=lambda: graph_str,
        )


def generate_fusion_from_config(config_options: dict[str, Any], pre_grad=True):
    fusions: list[GroupBatchFusionBase] = []
    for name, options in config_options.items():
        # we skip all patterns from pattern_matcher passes (e.g., split_cat)
        if name not in PRE_GRAD_FUSIONS and name not in POST_GRAD_FUSIONS:
            continue
        fusion_cls = PRE_GRAD_FUSIONS[name] if pre_grad else POST_GRAD_FUSIONS[name]
        _options = graph_search_options.copy()
        _options.update({k: v for k, v in options.items() if k != "devices"})
        fusions.append(fusion_cls(graph_search_options=_options))  # type: ignore[operator]
    return fusions


def group_batch_fusion_passes(
    graph: torch.fx.Graph, pre_grad=True, fusion_options=None
):
    fusions: list[GroupBatchFusionBase] = []
    # we keep all current pre grad fusions to keep
    # current implementation, will remove this later
    if pre_grad:
        fusions += generate_fusion_from_config(
            fusion_options
            if fusion_options is not None
            else config.pre_grad_fusion_options,
            pre_grad=True,
        )
    else:
        fbgemm_fusion_keys = [
            x
            for x in config.post_grad_fusion_options
            if (
                x not in OPTIMUS_EXCLUDE_POST_GRAD
                and config.post_grad_fusion_options[x].get("require_fbgemm", False)
            )
        ]
        fbgemm_fusions = {
            fusion: config.post_grad_fusion_options[fusion]
            for fusion in fbgemm_fusion_keys
        }
        non_fbgemm_fusions = {
            fusion: config.post_grad_fusion_options[fusion]
            for fusion in config.post_grad_fusion_options
            if fusion not in fbgemm_fusion_keys
        }
        fusions += generate_fusion_from_config(non_fbgemm_fusions, pre_grad=False)
        if has_fbgemm:
            fusions += generate_fusion_from_config(fbgemm_fusions, pre_grad=False)

    for i, rule in enumerate(fusions):
        with GraphTransformObserver(
            graph.owning_module,  # pyrefly: ignore[bad-argument-type]
            f"group_batch_fusion_{i}",
        ):
            apply_group_batch_fusion(graph, rule)  # type: ignore[arg-type]
<|endoftext|>from __future__ import annotations

import heapq
from collections import Counter, defaultdict
from typing import Any, TYPE_CHECKING

import torch
import torch.fx as fx
from torch._inductor.fx_passes.bucketing import (
    _get_collective_node_from_wait,
    _schedulable_wait_node,
    BucketMode,
    has_mergeable_all_gather_convert_dtype,
    is_all_gather_into_tensor as is_all_gather,
    is_all_reduce_tensor,
    is_fsdp_all_gather,
    is_fsdp_reduce_scatter,
    is_reduce_scatter_tensor as is_reduce_scatter,
    merge_all_gather_bucket,
    merge_all_reduce_bucket,
    merge_reduce_scatter_bucket,
)
from torch._inductor.fx_passes.overlap_preserving_bucketer import (
    get_full_bucket_key,
    OverlapPreservingBucketer,
)
from torch._inductor.fx_passes.overlap_scheduling import (
    CollectiveInfo,
    is_compute_node,
    OverlapScheduler,
)
from torch.utils._ordered_set import OrderedSet

from .graph_view import get_subgraph_by_path, GraphView, make_graph_view


if TYPE_CHECKING:
    from collections.abc import Callable

import logging


logger = logging.getLogger(__name__)


def _collect_nodes_must_be_after(node: fx.Node) -> list[fx.Node]:
    """BFS forward collecting node and its transitive users with no external inputs."""
    result: list[fx.Node] = [node]
    result_set: OrderedSet[fx.Node] = OrderedSet([node])
    i = 0
    while i < len(result):
        for user in result[i].users:
            if user not in result_set and all(
                inp in result_set for inp in user.all_input_nodes
            ):
                result_set.add(user)
                result.append(user)
        i += 1
    return result


def _collect_nodes_must_be_before(
    node: fx.Node, node_positions: dict[fx.Node, int]
) -> list[fx.Node]:
    """BFS backward collecting node and its non-placeholder dependencies, topo-sorted."""
    visited: OrderedSet[fx.Node] = OrderedSet()
    queue = [node]
    while queue:
        cur = queue.pop()
        if cur in visited or cur.op == "placeholder":
            continue
        visited.add(cur)
        queue.extend(cur.all_input_nodes)
    return sorted(visited, key=lambda n: node_positions[n])


def _bucket_trace_inputs(
    coll_node: fx.Node, node_in: object, group_name_arg: int
) -> list[fx.Node]:
    if not isinstance(node_in, fx.Node):
        raise AssertionError(f"expected node input to be a Node, got {type(node_in)}")
    inputs = [node_in]

    group_name = coll_node.args[group_name_arg]
    if isinstance(group_name, fx.Node):
        inputs.append(group_name)
    return inputs


def _all_gather_bucket_trace_inputs(coll_node: fx.Node) -> list[fx.Node]:
    node_in: object = coll_node.args[0]
    # The dtype conversion is erased by the all-gather bucket trace, so anchor
    # insertion on the tensor that remains as a graph input to the bucket.
    if has_mergeable_all_gather_convert_dtype(coll_node):
        if not isinstance(node_in, fx.Node):
            raise AssertionError(
                f"expected node input to be a Node, got {type(node_in)}"
            )
        node_in = node_in.args[0]
    return _bucket_trace_inputs(coll_node, node_in, group_name_arg=2)


def _reduce_scatter_bucket_trace_inputs(coll_node: fx.Node) -> list[fx.Node]:
    return _bucket_trace_inputs(coll_node, coll_node.args[0], group_name_arg=3)


def _all_reduce_bucket_trace_inputs(coll_node: fx.Node) -> list[fx.Node]:
    # all_reduce(tensor, reduce_op, group_name): tensor at 0, group_name at 2.
    return _bucket_trace_inputs(coll_node, coll_node.args[0], group_name_arg=2)


def _move_wait_users_after_latest_inputs(
    graph: fx.Graph,
    replacements: dict[fx.Node, fx.Node],
    replaced_users: dict[fx.Node, list[fx.Node]],
) -> None:
    node_positions = {n: i for i, n in enumerate(graph.nodes)}
    initial_users: OrderedSet[fx.Node] = OrderedSet()
    for old_out, new_out in replacements.items():
        if new_out not in node_positions:
            continue
        for user in replaced_users.get(old_out, []):
            if (
                user in node_positions
                and user.op != "output"
                and node_positions[user] < node_positions[new_out]
            ):
                initial_users.add(user)

    pending = sorted(initial_users, key=lambda n: node_positions[n])
    queued = OrderedSet(pending)
    while pending:
        node = pending.pop(0)
        queued.discard(node)

        node_positions = {n: i for i, n in enumerate(graph.nodes)}
        if node not in node_positions:
            continue

        input_nodes = [inp for inp in node.all_input_nodes if inp in node_positions]
        if not input_nodes:
            continue

        latest_input = max(input_nodes, key=lambda n: node_positions[n])
        if node_positions[node] >= node_positions[latest_input]:
            continue

        # Replacing old waits can leave existing consumers before the new bucket
        # outputs. Pull each affected consumer after its latest input.
        latest_input.append(node)
        node_positions = {n: i for i, n in enumerate(graph.nodes)}
        for user in node.users:
            if user in node_positions and user.op != "output" and user not in queued:
                queued.add(user)
                pending.append(user)


def _move_overlap_nodes(
    graph: fx.Graph,
    overlap_deps: dict[fx.Node, OrderedSet[fx.Node]],
    bucketed_node_types: dict[fx.Node, str],
) -> None:
    if not overlap_deps:
        return

    rs_defer: dict[fx.Node, list[fx.Node]] = defaultdict(list)
    ag_prefetch: dict[fx.Node, list[fx.Node]] = defaultdict(list)

    for target, sources in overlap_deps.items():
        for source in sources:
            source_type = bucketed_node_types.get(source, "")
            if source_type.startswith(
                ("bucketed_reduce_scatter", "bucketed_all_reduce")
            ):
                rs_defer[target].append(source)
            elif source_type.startswith("bucketed_all_gather"):
                ag_prefetch[target].append(source)

    node_positions = {n: i for i, n in enumerate(graph.nodes)}

    for rs_wait, rs_starts in rs_defer.items():
        latest_rs_start = max(rs_starts, key=lambda n: node_positions[n])
        node_insert_after = latest_rs_start
        for node in _collect_nodes_must_be_after(rs_wait):
            node_insert_after.append(node)
            node_insert_after = node

    # Recompute positions after RS moves
    node_positions = {n: i for i, n in enumerate(graph.nodes)}

    for ag_wait, ag_prefetch_starts in ag_prefetch.items():
        ag_wait_pos = node_positions[ag_wait]
        sorted_starts = sorted(ag_prefetch_starts, key=lambda n: node_positions[n])
        for ag_start in sorted_starts:
            if node_positions[ag_start] < ag_wait_pos:
                continue
            for node in _collect_nodes_must_be_before(ag_start, node_positions):
                ag_wait.prepend(node)


class ManualOverlapPreservingBucketer(OverlapPreservingBucketer):
    """
    Buckets collective operations based on user specifications.
    The actual bucket happens in bucket_collectives, where all-gathers/reduce-scatters in
        `nodes` will be buckted one single all-gather/reduce-scatter.
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.node_to_wait_map: dict[fx.Node, fx.Node] = defaultdict()
        # Maps bucketed nodes to their type string, scoped to this bucketer
        # instance so metadata doesn't leak across separate invocations.
        self.bucketed_node_types: dict[fx.Node, str] = {}

    def _bucket_group(self, coll_nodes: list[fx.Node]) -> None:
        if len(coll_nodes) <= 0:
            raise AssertionError("bucketed coll_nodes should have nonzero node")

        # Graph order changes after each bucket, so positions must be fresh.
        node_positions = {n: i for i, n in enumerate(self.graph.nodes)}
        waits = [self.collective_info[n].wait_node for n in coll_nodes]
        first_wait = min(waits, key=lambda w: node_positions[w])
        first = min(coll_nodes, key=lambda n: node_positions[n])
        replaced_users = {wait: list(wait.users) for wait in waits}

        if is_all_gather(first):
            bucket_trace_inputs = _all_gather_bucket_trace_inputs
            merge_bucket = merge_all_gather_bucket
            node_type = "bucketed_all_gather"
        elif is_reduce_scatter(first):
            bucket_trace_inputs = _reduce_scatter_bucket_trace_inputs
            merge_bucket = merge_reduce_scatter_bucket
            node_type = "bucketed_reduce_scatter"
        elif is_all_reduce_tensor(first):
            bucket_trace_inputs = _all_reduce_bucket_trace_inputs
            merge_bucket = merge_all_reduce_bucket
            node_type = "bucketed_all_reduce"
        else:
            raise ValueError(
                "bucket non all_gather/reduce_scatter/all_reduce node is not supported"
            )

        # coll_nodes order is used for tensor packing and may differ from
        # graph order. Insert the bucketed collective after its latest input.
        bucket_inputs = [inp for n in coll_nodes for inp in bucket_trace_inputs(n)]
        anchor = max([first, *bucket_inputs], key=lambda n: node_positions[n])
        next_node = anchor.next
        coll_node_set = OrderedSet(coll_nodes)
        while next_node in coll_node_set:
            next_node = next_node.next
        # Use the earliest old wait unless it precedes the bucket insertion
        # point; otherwise keep wait/output nodes with the traced bucket.
        wait_insertion_point = max(
            (first_wait, next_node), key=lambda n: node_positions[n]
        )

        new_nodes, replacements = merge_bucket(
            self.graph,
            coll_nodes,
            wait_insertion_point=wait_insertion_point,
            insert_before=next_node,
            mode=self.bucket_mode,
        )
        _move_wait_users_after_latest_inputs(self.graph, replacements, replaced_users)

        logger.debug(f"bucketing nodes: {coll_nodes} into {new_nodes}")  # noqa: G004

        # Identify the new wait(s) and their collective start in a single pass
        wait_to_start = {
            n: start
            for n in new_nodes
            if (start := _get_collective_node_from_wait(n)) is not None
        }
        if len(wait_to_start) < 1:
            raise AssertionError(
                f"Expected at least one new wait, got none in {new_nodes}"
            )
        new_waits = list(wait_to_start)
        new_start: fx.Node = wait_to_start[new_waits[0]]
        # Use last wait as the canonical wait for scheduling (same node when len == 1)
        new_wait = new_waits[-1]

        # Track bucketed node types on this bucketer instance so it doesn't leak
        # when the same graph is processed by multiple ManualOverlapScheduler
        # invocations (e.g. separate forward and backward passes).
        wait_set = OrderedSet(new_waits)
        for n in new_nodes:
            if n in wait_set:
                self.bucketed_node_types[n] = node_type + "_wait"
                self.node_to_wait_map[n] = new_wait
            elif n is new_start:
                self.bucketed_node_types[n] = node_type

    def _split_independent_collectives(
        self, coll_nodes: OrderedSet[fx.Node], scope_nodes: list[fx.Node]
    ) -> list[list[fx.Node]]:
        """Partition same-key collectives so no bucket contains a collective that
        depends on another member's result.

        Bucketing fuses collectives into one op: inputs are concatenated before
        the fused collective and outputs split after. If collective B's input
        depends on collective A's output (A's wait), fusing A and B makes the
        merged collective's input depend on its own output -- a graph cycle that
        later fails region topological sort. This arises e.g. in loss-parallel
        cross-entropy, which emits a sumexp and a result sum-all_reduce on the
        same (group, reduce_op, dtype) key where result depends on sumexp.

        Each collective is placed in the bucket equal to the longest chain of
        same-key collectives ending at it (its Mirsky level). Collectives at the
        same level are mutually independent, so grouping by level gives the
        minimum number of dependency-free buckets. Levels are computed with a
        single topological forward pass over ``scope_nodes`` that propagates,
        along real graph edges, the deepest same-key chain reaching each node --
        O(len(scope_nodes)) rather than O(len(coll_nodes)^2) pairwise ancestor
        queries. This is exact because same-key collectives in one bucketing
        scope depend on each other only through in-scope nodes.
        """
        wait_to_start = {self.collective_info[c].wait_node: c for c in coll_nodes}

        # ``reach[node]`` = deepest same-key collective chain in node's cone;
        # ``level[c]`` = that chain length including c (its bucket index + 1).
        reach: dict[fx.Node, int] = {}
        level: dict[fx.Node, int] = {}
        for node in sorted(scope_nodes, key=lambda n: self.node_idx[n]):
            incoming = max(
                (reach.get(inp, 0) for inp in node.all_input_nodes), default=0
            )
            node_reach = incoming
            if node in coll_nodes:
                level[node] = incoming + 1
            # A collective's chain becomes reachable through its wait (its result).
            start = wait_to_start.get(node)
            if start is not None:
                node_reach = max(node_reach, level[start])
            reach[node] = node_reach

        buckets: dict[int, list[fx.Node]] = defaultdict(list)
        for c in coll_nodes:
            buckets[level[c]].append(c)
        return [buckets[lvl] for lvl in sorted(buckets)]

    def manual_bucket_collectives(self, nodes: list[fx.Node]) -> None:
        """
        Bucket all all-gather/reduce-scatter nodes from nodes into one all-gather/reduce-scatter.
        """
        # Filter out valid collectives
        collectives = [n for n in nodes if n in self.collective_info]
        if collectives == []:
            return
        grouped_collectives: dict[object, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        for node in collectives:
            if not (
                is_fsdp_all_gather(node, self.node_ancestors)
                or is_fsdp_reduce_scatter(node)
                or is_all_reduce_tensor(node)
            ):
                continue
            key = get_full_bucket_key(node, "custom_ops")
            if key is not None:
                grouped_collectives[key].add(node)

        # Split each key-group into dependency-free sub-buckets so fusing never
        # makes a bucketed collective's input depend on its own output.
        sub_buckets: list[list[fx.Node]] = []
        for key, key_nodes in grouped_collectives.items():
            sub_buckets.extend(self._split_independent_collectives(key_nodes, nodes))

        for sub_bucket in sub_buckets:
            self._bucket_group(sub_bucket)


class ManualOverlapScheduler(OverlapScheduler):
    """
    Scheduler that manual buckets and reorders collective nodes based on module_bucket_plans
    """

    def __init__(
        self,
        gm: fx.GraphModule,
        module_bucket_plans: list[list[str] | str],
        insert_overlap_deps: bool,
        module_stack_fn: Callable[[fx.Node], list[tuple[str, type[Any]]]] | None = None,
        bucket_mode: BucketMode | None = None,
    ):
        # Manual overlap historically used "custom_ops" mode for bucketing
        bucket_mode = bucket_mode or "custom_ops"
        super().__init__(
            gm,
            max_in_flight_gb=0.0,
            max_compute_pre_fetch=0,
            collective_bucketing=True,
            insert_overlap_deps=insert_overlap_deps,
            compute_overlap_multipler=0.0,
            max_coll_distance=0,
            # ManualOverlapScheduler doesn't use runtime estimates (it
            # hardcodes estimated_time_ms=0 in _identify_collectives and
            # schedules purely from module_bucket_plans). Providing a
            # no-op estimator avoids the analytical NCCL path, which
            # crashes in compile-on-one-rank graphs where group_name is
            # an FX Node and the distributed runtime may not be available.
            custom_runtime_estimation=lambda node, size: 0.0,
            collective_estimator="analytical",
            max_memory_increase_gb=None,
            max_memory_increase_ratio=None,
            bucket_mode=bucket_mode,
        )
        self.module_bucket_plans = module_bucket_plans
        self.nodes_in_subgraph: list[list[fx.Node]] = []

        self.bucketer = ManualOverlapPreservingBucketer(
            graph=self.graph,
            collective_info=self.collective_info,
            scheduled=OrderedSet(self.graph.nodes),
            bucket_mode=bucket_mode,
        )
        self.insert_overlap_deps = insert_overlap_deps

        self.module_stack_fn = module_stack_fn

    def _identify_collectives(self) -> None:
        """Identify all collective operations."""
        for node in self.nodes:
            if _schedulable_wait_node(node):
                start = node.args[0]
                info = CollectiveInfo(
                    start_node=start,
                    wait_node=node,
                    size_bytes=0,
                    estimated_time_ms=0,
                    exposed_time_ms=0,
                )
                self.collective_info[start] = info
                self.wait_to_start[node] = start
                self.unscheduled_collectives.add(start)

    def _add_to_ready_queue(self, node: fx.Node) -> None:
        """Manual scheduling uses single queue ordered by original node index."""
        heapq.heappush(self.on_path_ready, (self.node_idx[node], node))

    def run(self) -> torch.fx.GraphModule:
        """Entry point to run the manual bucket algorithm"""
        # Bucket collectives in each bucket_module
        self._manual_bucket_collectives()

        # Reorder collectives with last/next bucket_module
        self._manual_reorder_graph()

        return self.gm

    def _manual_reorder_graph(self) -> None:
        """
        Reorder nodes in the FX graph to enforce manual overlap dependencies.

        forward graph (all-gathers only):
            modules are processed in order: module 0, 1, 2, ...

            before reordering:
            ag_start_0 -> ag_wait_0 -> compute_0 -> ag_start_1 -> ag_wait_1 -> compute_1 -> ...

            Reordering prefetches module i+1's parameters while computing module i
            It adds dependencies: ag_wait_i should depend on ag_start_(i+1)
            This enforces ag_start_(i+1) to happen before ag_wait_i so it overlaps with module i's compute

            after reordering:
            ag_start_0 -> ag_start_1 -> ag_wait_0 -> compute_0 -> ag_wait_1 -> compute_1 -> ...

        backward graph (all-gathers and reduce-scatters):
            modules are processed in reverse order: module N, N-1, N-2, ...

            before reordering:
            ag_start_N -> ag_wait_N -> compute_N -> rs_start_N -> rs_wait_N -> ...

            For all-gathers, prefetch module i-1's parameters while computing module i
            Adds dependencies: ag_wait_i should depend on ag_start_(i-1)
            So ag_start_(i-1) overlaps with module i's compute

            For reduce-scatters, defer rs_wait_i to happen after rs_start_(i-1)
            Adds dependencies: rs_wait_i should depend on rs_start_(i-1)
            So rs_start_i overlaps with module i-1's compute

        """
        delayed_rs_wait_nodes: list[fx.Node] = []
        current_rs_start_nodes: list[fx.Node] = []
        overlap_deps: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)

        # Re-initialize after graph modification in _manual_bucket_collectives
        self.node_idx = {n: i for i, n in enumerate(self.nodes)}
        self.on_path_ready = []
        self.scheduled = OrderedSet()
        for node in self.nodes:
            if self.in_degree[node] == 0:
                self._add_to_ready_queue(node)

        # schedule reduce scatter normally in self._schedule
        while self.on_path_ready:
            _, node = heapq.heappop(self.on_path_ready)
            node_type = self.bucketer.bucketed_node_types.get(node, "")

            if node in self.scheduled:
                continue

            if node_type in ("bucketed_reduce_scatter", "bucketed_all_reduce"):
                # Collect grad-reduction start nodes (pre_bucket_rs and rs, or
                # the HSDP/DDP replicate all_reduce).
                current_rs_start_nodes.append(node)

            elif node_type in (
                "bucketed_reduce_scatter_wait",
                "bucketed_all_reduce_wait",
            ):
                # When we see a wait node from a new RS/AR, flush delayed waits
                # with dependencies on previously collected start nodes
                if current_rs_start_nodes:
                    for delayed in delayed_rs_wait_nodes:
                        for rs_start in current_rs_start_nodes:
                            overlap_deps[delayed].add(rs_start)
                    delayed_rs_wait_nodes.clear()
                    current_rs_start_nodes.clear()
                delayed_rs_wait_nodes.append(node)

            self._schedule(node)

        self.scheduled = OrderedSet(reversed(list(self.scheduled)))
        picked_ag: list[fx.Node] = []
        last_compute: fx.Node | None = None

        for node in self.scheduled:
            node_type = self.bucketer.bucketed_node_types.get(node, "")
            if node_type == "bucketed_all_gather":
                picked_ag.append(node)
                continue

            if node_type == "bucketed_all_gather_wait":
                # Connect corresponding all_gather_wait -> all_gather edges
                if picked_ag:
                    for ag in picked_ag:
                        overlap_deps[self.bucketer.node_to_wait_map[node]].add(ag)
                picked_ag.clear()
            if is_compute_node(node):
                last_compute = node

        if last_compute is not None:
            if not any(
                self.node_ancestors.is_ancestor(ag, last_compute) for ag in picked_ag
            ):
                for ag in picked_ag:
                    overlap_deps[last_compute].add(ag)

        _move_overlap_nodes(self.graph, overlap_deps, self.bucketer.bucketed_node_types)
        self.graph.lint()

        if self.insert_overlap_deps:
            from torch._inductor.fx_passes.control_dependencies import (
                preserve_node_ordering,
            )

            preserve_node_ordering(self.graph, overlap_deps)

    def _manual_bucket_collectives(self) -> None:
        """Bucket nodes in each module_bucket from module_bucket_plans."""
        self._obtain_nodes_in_subgraph()
        for i, nodes in enumerate(self.nodes_in_subgraph):
            self.bucketer.manual_bucket_collectives(nodes=nodes)

        self.graph.lint()
        self.nodes = list(self.graph.nodes)
        self.in_degree = Counter(user for node in self.nodes for user in node.users)

    def _schedule(self, node: fx.Node) -> None:
        """Schedule a node."""
        if node in self.scheduled:
            raise AssertionError(f"node already scheduled: {node}")
        if not all(n in self.scheduled for n in node.all_input_nodes):
            raise AssertionError(f"all input nodes must be scheduled before {node}")
        self.scheduled.add(node)
        for user in node.users:
            self.in_degree[user] -= 1
            if self.in_degree[user] == 0:
                self._add_to_ready_queue(user)

    def _obtain_nodes_in_subgraph(self) -> None:
        """
        Obtain nodes in each subgraph from module_bucket_plans
        """
        graph_view: GraphView | None = make_graph_view(self.graph, self.module_stack_fn)
        if graph_view is None:
            return

        for module in self.module_bucket_plans:
            subgraph_view = get_subgraph_by_path(graph_view, module)
            self.nodes_in_subgraph.append(subgraph_view)

        all_subgraph_nodes = [
            node for sublist in self.nodes_in_subgraph for node in sublist
        ]
        unique_subgraph_nodes = list(OrderedSet(all_subgraph_nodes))
        if len(all_subgraph_nodes) > len(unique_subgraph_nodes):
            raise AssertionError(
                f"Overlapping FX nodes detected across subgraphs in `module_bucket_plans`. "
                f"Expected disjoint node sets but found "
                f"{len(all_subgraph_nodes) - len(unique_subgraph_nodes)} duplicated node(s)."
            )


def manual_overlap_bucketing(
    gm: torch.fx.GraphModule,
    module_bucket_plans: list[list[str] | str],
    insert_overlap_deps: bool = False,
    module_stack_fn: Callable[[fx.Node], list[tuple[str, type[Any]]]] | None = None,
    bucket_mode: BucketMode | None = None,
) -> torch.fx.GraphModule:
    """Schedule nodes based on user specifications in module_bucket_plans
    The manual overlapping consists of two steps:
    Step 1: bucket all-gather/reduce-scatter in each module in module_bucket_plans
    Step 2: reorder all-gather to overlap with last module_bucket &
        reorder reduce-scatter to overlap with next module_bucket
    TODO(ruisizhang123): allow users to explicitly specify which
        module_bucket they want to overlap.

    Args:
        gm: input graph module to optimize.
        module_bucket_plans: user specified FQNs
        module_stack_fn: Optional callable for extracting module hierarchy from nodes.
            Used to construct a GraphView for identifying nodes in module_bucket_plans.
            The module_class component of the returned tuples is not used by this pass.

            See the `module_stack_fn` parameter in `make_graph_view` (graph_view.py) for
            detailed documentation on signature, return format, and usage examples.
        bucket_mode: Bucket mode for collective bucketing. None uses default.
    """
    # decode abbreviated FQNs to actual FQNs
    overlapped_gm = ManualOverlapScheduler(
        gm,
        module_bucket_plans,
        insert_overlap_deps,
        module_stack_fn,
        bucket_mode=bucket_mode,
    ).run()
    overlapped_gm.recompile()
    return overlapped_gm
<|endoftext|>"""
Node runtime estimation for overlap scheduling.
"""

from __future__ import annotations

import functools
import itertools
import operator
from functools import lru_cache
from typing import Any

import torch
import torch.fx as fx
from torch._inductor import config
from torch._inductor.fx_passes.bucketing import (
    _resolve_group_name,
    _schedulable_wait_node,
)
from torch._inductor.utils import clear_on_fresh_cache
from torch._logging import getArtifactLogger, trace_structured
from torch.fx.experimental.symbolic_shapes import optimization_hint
from torch.fx.operator_schemas import normalize_function


def _format_csv(headers: list[str], rows: list[list[str]]) -> str:
    """Format data as CSV. Human-readable and easily parseable."""
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(str(v) for v in row))
    return "\n".join(lines)


def _get_collective_key(coll_node: fx.Node) -> str:
    """Extract a unique key for a collective node including group info and tensor size."""
    from torch._inductor import fx_utils

    opt_args_kwargs = normalize_function(
        coll_node.target,  # type: ignore[arg-type]
        args=coll_node.args,
        kwargs=coll_node.kwargs,
        normalize_to_only_use_kwargs=True,
    )
    if opt_args_kwargs is None:
        raise AssertionError("normalize_function returned None for collective node")
    _, kwargs = opt_args_kwargs
    raw_group_name = kwargs.get("group_name", None)
    group_name = (
        _resolve_group_name(raw_group_name) if raw_group_name is not None else None
    )
    group_size = kwargs.get("group_size", None)

    tensor_bytes: int | None = None
    success, args, kw = fx_utils.get_fake_args_kwargs(coll_node)
    if success:

        def extract_first_tensor_bytes(t: torch.Tensor) -> torch.Tensor:
            nonlocal tensor_bytes
            if tensor_bytes is None:
                shape = [get_hint(dim) for dim in t.shape]
                if all(s is not None for s in shape):
                    numel = functools.reduce(operator.mul, shape, 1)
                    tensor_bytes = numel * t.dtype.itemsize
            return t

        torch.utils._pytree.tree_map_only(
            torch.Tensor, extract_first_tensor_bytes, (args, kw)
        )

    return f"{coll_node.target} group_size:{group_size} group_name:{group_name} input_bytes:{tensor_bytes}"


def _get_collective_estimations(coll_node: fx.Node) -> tuple[float, float]:
    """Get NCCL and Inductor analytical estimations for a collective node.

    Returns: (nccl_ms, inductor_ms)
    """
    nccl_ms = (
        torch._inductor.comm_analysis.estimate_nccl_collective_runtime_from_fx_node(
            coll_node, None, use_nccl_estimator=True
        )
    )
    inductor_ms = (
        torch._inductor.comm_analysis.estimate_nccl_collective_runtime_from_fx_node(
            coll_node, None, use_nccl_estimator=False
        )
    )
    return nccl_ms, inductor_ms


# Setup logger for artifact logging
log = getArtifactLogger(__name__, "node_runtime_estimation")


# TODO: Consider using a distributed-aware cache or rank-local disk cache
# not using local cache because different ranks might write to it concurrently.
# solvable in future, potentially with workflow to seed cache
@clear_on_fresh_cache
@lru_cache
def _get_collective_cache() -> dict[str, float]:
    """Get process-local cache for collective benchmarks."""
    return {}


def get_cached_runtime(key: str) -> float | None:
    """Get cached runtime from process-local cache."""
    return _get_collective_cache().get(key)


def set_cached_runtime(key: str, value: float) -> None:
    """Set cached runtime in process-local cache."""
    _get_collective_cache()[key] = value


def get_hint(x: int | torch.SymInt) -> int | None:
    if isinstance(x, int):
        return x
    if not isinstance(x, torch.SymInt):
        raise AssertionError(f"expected int or SymInt, got {type(x)}")
    return x.hint if x.has_hint() else None


def can_benchmark_collective() -> bool:
    """Check if we can benchmark collectives (not fake process group)."""
    import torch.distributed as c10d

    if not c10d.is_initialized():
        return False

    pg = c10d.distributed_c10d._get_default_group()
    if (
        torch.distributed.distributed_c10d.get_backend(pg)
        == torch.distributed.distributed_c10d.Backend.FAKE
    ):
        return False

    return True


def _median(lst):
    if len(lst) == 0:
        raise AssertionError("expected non-empty list for median")
    return torch.median(torch.tensor(lst)).item()


def _benchmark_collective_with_cuda_events_impl(
    n: torch.fx.Node,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    nruns: int,
) -> float | None:
    """
    Core benchmarking logic using CUDA events and barriers.
    Returns runtime in ms or None on failure.
    """
    from torch._dynamo.testing import rand_strided

    # Convert FakeTensors to real tensors before benchmarking
    def to_real(t: torch.Tensor) -> torch.Tensor:
        shape = [get_hint(dim) for dim in t.shape]
        stride = [get_hint(s) for s in t.stride()]

        if any(s is None for s in itertools.chain(shape, stride)):
            # This should not happen, as can_benchmark_collective checks for unbacked
            raise ValueError("Cannot convert tensor with symbolic dimensions")

        return rand_strided(shape, stride, device=t.device, dtype=t.dtype)  # type: ignore[arg-type]

    args, kwargs = torch.utils._pytree.tree_map_only(
        torch.Tensor,
        to_real,
        (args, kwargs),
    )

    args, kwargs = torch.utils._pytree.tree_map_only(
        torch.SymInt,
        lambda s: optimization_hint(s, fallback=config.unbacked_symint_fallback),
        (args, kwargs),
    )

    # Warmup: call collective once and wait
    torch.accelerator.synchronize()
    result = n.target(*args, **kwargs)  # type: ignore[operator]
    torch.ops._c10d_functional.wait_tensor(result)
    torch.accelerator.synchronize()

    # Benchmark with CUDA events
    comm_times = []
    for _ in range(nruns):
        start_evt = torch.Event(enable_timing=True)
        end_evt = torch.Event(enable_timing=True)

        start_evt.record()
        result = n.target(*args, **kwargs)  # type: ignore[operator]
        torch.ops._c10d_functional.wait_tensor(result)
        end_evt.record()
        end_evt.synchronize()

        comm_times.append(start_evt.elapsed_time(end_evt))

    return _median(comm_times)


def benchmark_collective_with_cuda_events(
    n: torch.fx.Node,
    nruns: int = 2,
) -> tuple[float | None, str]:
    """
    Benchmark collective with CUDA events. Returns (runtime_ms, cache_key) or (None, "") on failure.
    """
    # context manager not allowed with profiler.
    with torch.utils._python_dispatch._disable_current_modes():
        return benchmark_collective_with_cuda_events_impl(n, nruns)


def benchmark_collective_with_cuda_events_impl(
    n: torch.fx.Node,
    nruns: int = 3,
) -> tuple[float | None, str]:
    """
    Benchmark collective with CUDA events. Returns (runtime_ms, cache_key) or (None, "") on failure.
    """
    from torch._inductor import fx_utils
    from torch.distributed.distributed_c10d import _get_group_size_by_name

    # Early check: can we actually run collectives?
    if not can_benchmark_collective():
        return None, ""

    success, args, kwargs = fx_utils.get_fake_args_kwargs(n)

    opt_args_kwargs = normalize_function(
        n.target,  # type: ignore[arg-type]
        args=n.args,
        kwargs=n.kwargs,
        normalize_to_only_use_kwargs=True,
    )
    if opt_args_kwargs is None:
        raise AssertionError("normalize_function returned None for collective node")
    group_name = _resolve_group_name(opt_args_kwargs[1]["group_name"])
    group_size = _get_group_size_by_name(group_name)

    if not success:
        return None, ""

    # Extract actual input size in BYTES (first tensor argument)
    actual_bytes: int | None = None

    def extract_tensor_info(t: torch.Tensor) -> torch.Tensor:
        nonlocal actual_bytes
        if actual_bytes is None:
            shape = [get_hint(dim) for dim in t.shape]
            if any(s is None for s in shape):
                return t

            total_elems = 1
            for dim in shape:
                if dim is None:
                    raise AssertionError(f"expected non-None dim, got {dim}")
                total_elems *= dim

            actual_bytes = total_elems * t.dtype.itemsize
        else:
            # out-variants (e.g. all_gather_into_tensor_out) can have multiple tensors
            pass
        return t

    torch.utils._pytree.tree_map_only(torch.Tensor, extract_tensor_info, (args, kwargs))

    if actual_bytes is None:
        return None, ""

    # Cache key by BYTES (dtype-agnostic)
    key = f"{n.target}: ({group_size} group size, {actual_bytes} bytes)"

    # Check cache
    if (cached := get_cached_runtime(key)) is not None:
        return cached, key

    # Benchmark using CUDA events with actual args/kwargs
    runtime = _benchmark_collective_with_cuda_events_impl(n, args, kwargs, nruns)

    if runtime is None:
        return None, key

    # Cache the result
    set_cached_runtime(key, runtime)
    return runtime, key


def _log_compute_estimations(
    compute_nodes: list[fx.Node],
    benchmarked_estimations: list[float],
    analytical_estimations: list[float],
) -> None:
    """Log compute node runtime estimations comparing benchmarked vs analytical."""
    import torch.utils._pytree as pytree
    from torch._inductor.fx_utils import count_flops_fx
    from torch.utils._dtype_abbrs import dtype_abbrs

    def _node_summary(n: fx.Node) -> str:
        ret = str(n)
        for arg in pytree.arg_tree_leaves(n.args, n.kwargs):
            if not isinstance(arg, torch.fx.Node):
                continue
            if "val" in arg.meta:
                t = arg.meta["val"]
                if isinstance(t, torch.Tensor):
                    ret += f" {dtype_abbrs[t.dtype]}{tuple(t.shape)}"
                elif isinstance(t, torch.SymInt):
                    ret += f" SymInt({t})"
        return ret

    headers = [
        "Node",
        "Benchmarked Est(us)",
        "Analytical Est(us)",
        "Diff(ratio)",
        "Diff(us)",
        "Flops",
    ]

    def _fmt(v: object) -> str:
        if isinstance(v, torch.types.py_sym_types):
            if v.node.has_hint():
                return f"{v}({v.node.hint:.4f})"
            return str(v)
        return f"{v:.4f}"

    rows = [
        [
            _node_summary(node),
            _fmt(est_b * 1e3),
            _fmt(est_a * 1e3),
            _fmt((est_a / est_b) if est_b > 0 else 0),
            _fmt((est_a - est_b) * 1e3),
            str(count_flops_fx(node)),
        ]
        for node, est_b, est_a in zip(
            compute_nodes, benchmarked_estimations, analytical_estimations
        )
    ]

    log_str = _format_csv(headers, rows)

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "fx_compute_nodes_runtime_estimation",
            "encoding": "string",
        },
        payload_fn=lambda: log_str,
    )


def _log_graph_collective_benchmarks(gm: fx.GraphModule, artifact_name: str) -> None:
    collective_nodes = []
    collective_keys = []
    benchmarked = []

    for node in gm.graph.nodes:
        if _schedulable_wait_node(node):
            start = node.args[0]
            collective_nodes.append(start)
            collective_keys.append(_get_collective_key(start))
            benchmarked_ms, _ = benchmark_collective_with_cuda_events(start, nruns=5)
            benchmarked.append(benchmarked_ms if benchmarked_ms else 0.0)

    if collective_nodes:
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        _log_collective_benchmarks(
            collective_nodes,
            collective_keys,
            benchmarked,
            world_size,
            artifact_name,
        )


def _log_collective_benchmarks(
    collective_nodes: list[fx.Node],
    collective_keys: list[str] | None = None,
    benchmarked_medians: list[float] | None = None,
    world_size: int | None = None,
    artifact_name: str = "fx_collectives_analytical_estimation",
) -> None:
    """Log collective estimations for tlparse. Includes benchmarks if provided."""
    if world_size is None:
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )

    has_benchmarks = benchmarked_medians is not None

    if has_benchmarks:
        headers = [
            "Collective Key",
            "Benchmarked(ms)",
            "NCCL Est(ms)",
            "Inductor Est(ms)",
            "NCCL Diff(ratio)",
            "Inductor Diff(ratio)",
        ]
    else:
        headers = [
            "Collective Key",
            "NCCL Est(ms)",
            "Inductor Est(ms)",
        ]

    rows = []
    for i, coll_node in enumerate(collective_nodes):
        key = collective_keys[i] if collective_keys else _get_collective_key(coll_node)
        nccl_ms, inductor_ms = _get_collective_estimations(coll_node)

        if benchmarked_medians is not None:
            benchmarked_ms = benchmarked_medians[i]
            nccl_diff_pct = (nccl_ms / benchmarked_ms) if benchmarked_ms > 0 else 0
            inductor_diff_pct = (
                (inductor_ms / benchmarked_ms) if benchmarked_ms > 0 else 0
            )
            rows.append(
                [
                    key,
                    f"{benchmarked_ms:.4f}",
                    f"{nccl_ms:.4f}",
                    f"{inductor_ms:.4f}",
                    f"{nccl_diff_pct:.2f}",
                    f"{inductor_diff_pct:.2f}",
                ]
            )
        else:
            rows.append(
                [
                    key,
                    f"{nccl_ms:.4f}",
                    f"{inductor_ms:.4f}",
                ]
            )

    log_str = f"# World size: {world_size}\n"
    log_str += _format_csv(headers, rows)

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": artifact_name,
            "encoding": "string",
        },
        payload_fn=lambda: log_str,
    )
<|endoftext|>import functools
import itertools
import operator
import typing
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch._inductor.runtime.runtime_utils
from torch import Tensor
from torch._dynamo.utils import counters
from torch._inductor import utils
from torch._inductor.autoheuristic.autoheuristic import (
    AHContext,
    AutoHeuristic,
    LocalFeedback,
)
from torch._inductor.autoheuristic.autoheuristic_utils import (
    context_add_strides,
    context_add_using_tf32,
    pad_mm_operations,
    pad_mm_precondition,
)
from torch._inductor.runtime.caching import encoders, memoizers
from torch._subclasses.fake_tensor import is_fake_tensor
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.utils._mode_utils import no_dispatch

from ...utils._triton import has_triton
from ..pattern_matcher import (
    fwd_only,
    gen_register_replacement,
    joint_fwd_bwd,
    Match,
    ReplaceFn,
    SearchFn,
)


aten = torch.ops.aten


# This flag is only used for testing purpose.
# Changing it to True will ignore comparing do_bench times
# between original pattern and padded one.
_skip_do_bench_times = False


def fetch_fake_tensors(match: Match, kwarg_names: Sequence[str]) -> list[Tensor]:
    kwargs = match.kwargs
    return [kwargs[name].meta["val"] for name in kwarg_names]


def unwrap_fake_args(
    *arg_names: str,
) -> Callable[[Callable[..., Any]], Callable[[Match], Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[[Match], Any]:
        def wrapper(match: Match) -> Any:
            fake_tensors = fetch_fake_tensors(match, arg_names)
            return func(*fake_tensors)

        return wrapper

    return decorator


def get_alignment_size(x: Tensor) -> int:
    return get_alignment_size_dtype(x.dtype)


def get_alignment_size_dtype(dtype: torch.dtype) -> int:
    if dtype == torch.float16 or dtype == torch.half or dtype == torch.bfloat16:
        return 8
    elif dtype == torch.float32 or dtype == torch.float:
        return 4
    else:
        return 0


def check_device(a: Tensor, b: Tensor) -> bool:
    return (a.is_cuda and b.is_cuda) or (a.is_xpu and b.is_xpu)


def check_dtype(a: Tensor, b: Tensor) -> bool:
    return a.is_floating_point() and b.is_floating_point()


def hint_symbols(
    ds: Sequence[int | torch.SymInt],
) -> list[int]:
    """Helper to convert symbolic dimensions to their concrete hint values."""
    from torch.fx.experimental.symbolic_shapes import optimization_hint

    return [optimization_hint(d) for d in ds]


def can_pad(
    mat1: Tensor,
    mat2: Tensor,
    op: torch._ops.OpOverloadPacket,
    input: Tensor | None = None,
) -> bool:
    """
    Determines if an operation CAN be padded (safety checks).
    All logic related to whether it's safe to pad should be here.
    """

    # Can't pad if there is no static dims, we pad static dims only.
    def has_one_static_dim(t: Tensor) -> bool:
        """Return False if all dimensions are symbolic — nothing concrete to pad."""
        for x in t.size():
            if isinstance(x, int):
                return True
            elif not isinstance(x, torch.SymInt):
                raise RuntimeError("not expected size")
        return False

    # Basic safety checks
    if not torch._inductor.config.shape_padding:
        return False

    if not check_device(mat1, mat2):
        return False

    if not check_dtype(mat1, mat2):
        return False

    # For padding to be vaible each tensor should have at least one static dim.
    tensors = [t for t in (mat1, mat2, input) if t is not None]
    if not all(has_one_static_dim(t) for t in tensors):
        return False

    # Skip zero-sized dimensions — padding would be wasteful (mm on empty tensors)
    from torch.fx.experimental.symbolic_shapes import optimization_hint

    if any(
        optimization_hint(dim) == 0 for dim in itertools.chain(mat1.shape, mat2.shape)
    ):
        return False

    # Calculate padding lengths to check if padding is needed
    with no_dispatch():
        if op is torch.ops.aten.mm or op is torch.ops.aten.addmm:
            m = mat1.shape[0]
            k = mat1.shape[1]
            n = mat2.shape[1]
        elif op is torch.ops.aten.bmm:
            m = mat1.shape[1]
            k = mat1.shape[2]
            n = mat2.shape[2]
        else:
            return False

        k_padded_length = get_padded_length(k, get_alignment_size(mat1))
        n_padded_length = get_padded_length(n, get_alignment_size(mat2))
        m_padded_length = get_padded_length(m, get_alignment_size(mat1))

        # No padding needed - can't pad if there's nothing to pad
        if m_padded_length == k_padded_length == n_padded_length == 0:
            return False

    # In deterministic mode, we can't safely benchmark - disallow padding
    # Check this after other basic checks so force_shape_pad/autoheuristic can override
    if (
        torch._inductor.config.deterministic
        and not torch._inductor.config.force_shape_pad
        and not torch._inductor.config.use_autoheuristic("pad_mm")
    ):
        return False

    # Triton availability check - required for padding to work
    if not has_triton():
        return False

    return True


def get_padded_length(x: int | torch.SymInt, alignment_size: int) -> int:
    # we don't pad x if it is symbolic
    if isinstance(x, torch.SymInt) or alignment_size == 0 or x % alignment_size == 0:
        return 0

    # ignore dim that can be squeezed away
    if x == 1:
        return 0

    return int((x // alignment_size + 1) * alignment_size) - x


def pad_dim(x: Tensor, padded_length: int, dim: int) -> Tensor:
    if padded_length == 0:
        return x
    pad = x.new_zeros(*x.shape[:dim], padded_length, *x.shape[dim + 1 :])
    return torch.cat([x, pad], dim=dim)


def addmm_pattern(
    input: Tensor, mat1: Tensor, mat2: Tensor, beta: float, alpha: float
) -> Tensor:
    return aten.addmm(input, mat1, mat2, beta=beta, alpha=alpha)


def _is_statically_expandable_to(shape: torch.Size, desired: Sequence[Any]) -> bool:
    if len(shape) > len(desired):
        return False
    return all(
        statically_known_true(dim == desired_dim) or statically_known_true(dim == 1)
        for dim, desired_dim in zip(reversed(shape), reversed(desired))
    )


def should_pad_addmm(match: Match) -> bool:
    mat1, mat2, input = fetch_fake_tensors(match, ("mat1", "mat2", "input"))
    beta = match.kwargs["beta"]
    if (
        beta == 0
        and input.is_cuda
        and not _is_statically_expandable_to(
            input.shape, (mat1.shape[0], mat2.shape[1])
        )
    ):
        return False
    return should_pad(match, mat1, mat2, torch.ops.aten.addmm, input=input)


def pad_addmm(
    input: Tensor | None,
    mat1: Tensor,
    mat2: Tensor,
    m_padded_length: int,
    k_padded_length: int,
    n_padded_length: int,
    beta: float = 1.0,
    alpha: float = 1.0,
    mat1_pre_padded: bool = False,
    mat2_pre_padded: bool = False,
) -> Tensor:
    # for paddings, dim order is reversed for some reasons
    # and for every dim, we need to specify left and right padding
    if not mat1_pre_padded:
        mat1 = pad_mat1(
            mat1, m_padded_length=m_padded_length, k_padded_length=k_padded_length
        )
    if not mat2_pre_padded:
        mat2 = pad_mat2(
            mat2, k_padded_length=k_padded_length, n_padded_length=n_padded_length
        )

    # the add broadcasts, so we only pad if the dimension != 1
    if input is not None:
        if n_padded_length != 0:
            if input.dim() == 2 and input.shape[1] != 1:
                input = pad_dim(input, n_padded_length, 1)
            elif input.dim() == 1 and input.shape[0] != 1:
                input = pad_dim(input, n_padded_length, 0)
        if m_padded_length != 0 and input.dim() == 2 and input.shape[0] != 1:
            input = pad_dim(input, m_padded_length, 0)

    res = aten.addmm(input, mat1, mat2, beta=beta, alpha=alpha)

    if m_padded_length != 0:
        res = res[:-m_padded_length, :]
    if n_padded_length != 0:
        res = res[:, :-n_padded_length]
    return res


def addmm_replace(
    input: Tensor | None,
    mat1: Tensor,
    mat2: Tensor,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> Tensor:
    k_padded_length = get_padded_length(mat1.shape[1], get_alignment_size(mat1))
    n_padded_length = get_padded_length(mat2.shape[1], get_alignment_size(mat2))
    m_padded_length = get_padded_length(mat1.shape[0], get_alignment_size(mat1))
    return pad_addmm(
        input,
        mat1,
        mat2,
        m_padded_length,
        k_padded_length,
        n_padded_length,
        beta,
        alpha,
    )


def is_mm_compute_bound(M: int, K: int, N: int, dtype: torch.dtype) -> bool:
    denominator = M * K + N * K + M * N
    if denominator == 0:
        return False
    arithmetic_intensity = (M * N * K) / denominator

    # we have experienced some large perf hits in this case, even in bandwidth bound regimes
    if (
        dtype is torch.bfloat16
        and K > M
        and K > N
        and (torch.xpu.is_available() or torch.cuda.get_device_capability() < (9, 0))
    ):  # doesn't repro on h100s:
        return True

    # Fails with AMD
    try:
        machine_balance = (
            1000 * utils.get_device_tflops(dtype)
        ) / utils.get_gpu_dram_gbps()
    except Exception:
        return True

    # dram_gbps might be underestimating bandwidth because of cache.
    # if we estimate machine balance too low we might miss some speedups,
    # if we estimate too high there will be unnecessary compilation time increase.
    # TODO - finetune coefficient here. As a reference point, Triton mm model assumes
    # 80% of reads are in cache and cache is 4x faster than dram_gbps
    machine_balance = machine_balance * 0.5

    return arithmetic_intensity > machine_balance


@functools.cache
def get_pad_cache() -> torch._inductor.codecache.LocalCache:
    return torch._inductor.codecache.LocalCache()


def get_cached_should_pad(key: str) -> bool:
    return get_pad_cache().lookup(key)  # type: ignore[return-value]


def set_cached_should_pad(key: str, value: bool) -> None:
    return get_pad_cache().set_value(key, value=value)


def get_cached_base_mm_benchmark_time(key: str) -> float:
    return get_pad_cache().lookup(key)  # type: ignore[return-value]


def set_cached_base_mm_benchmark_time(key: str, value: float) -> None:
    return get_pad_cache().set_value(key, value=value)


def should_pad_bench_key(
    match: Match,
    mat1: Tensor,
    mat2: Tensor,
    op: torch._ops.OpOverloadPacket,
    input: Tensor | None = None,
    is_base_time_key: bool = False,
) -> str:
    def tensor_key(t: Tensor) -> tuple[torch.Size, tuple[int, ...], torch.dtype]:
        return (t.shape, t.stride(), t.dtype)

    tf32_key = (
        None
        if mat1.dtype != torch.float32
        else torch.backends.cuda.matmul.fp32_precision == "tf32"
        or torch.backends.mkldnn.fp32_precision == "tf32"
    )

    def fmt_pad(name: str) -> str | None:
        if is_base_time_key:
            return None
        return f"exclude_pad:{should_exclude_padding_time(match, name)}"

    key = (
        tensor_key(mat1),
        tensor_key(mat2),
        fmt_pad("mat1"),
        fmt_pad("mat2"),
        op,
        input if input is None else tensor_key(input),
        tf32_key,
    )

    key = str(key)
    if is_base_time_key:
        key = f"base mm time: {key}"
    return key


def get_non_view_def(node: torch.fx.Node) -> torch.fx.Node:
    if node.op == "call_function" and node.target is operator.getitem:
        return get_non_view_def(node.args[0])  # type: ignore[arg-type]

    if (
        node.op == "call_function"
        and isinstance(node.target, torch._ops.OpOverload)
        and utils.is_view(node.target)
    ):
        return get_non_view_def(node.all_input_nodes[0])

    return node


def should_exclude_padding_time(match: Match, arg_name: str) -> bool:
    from torch._prims_common import is_contiguous_or_false

    node_def = get_non_view_def(match.kwargs[arg_name])

    # constant padding converts tensors to contiguous so even if the input tensor
    # can be planned layout transform is not free. TODO - way to pad and preserve layout ?
    # Use is_contiguous_or_false to avoid guarding on data-dependent expressions
    # with unbacked symints - returns False instead of raising an error.
    if not is_contiguous_or_false(fetch_fake_tensors(match, (arg_name,))[0]):
        return False

    # TODO - see issue https://github.com/pytorch/pytorch/issues/128889
    # We would only able to completely plan these out if we were only doing
    # first dimension padding. non-first we would still need a copy
    # because these outputs are fixed dense.
    cannot_plan_output = [
        aten.mm.default,
        aten.convolution.default,
        aten.convolution_backward.default,
        aten.bmm.default,
        aten.addmm.default,
        aten._scaled_dot_product_flash_attention.default,
        aten._scaled_dot_product_efficient_attention.default,
    ]

    if node_def.target in cannot_plan_output:
        return False

    if (
        node_def.target is aten.cat.default
        and len(node_def.all_input_nodes)
        > torch._inductor.config.max_pointwise_cat_inputs
    ):
        return False

    # optimistically assume we should be able to memory plan away
    # all non inputs
    return node_def.op != "placeholder"


def is_padded_faster(key: str, ori_time: float, pad_time: float) -> bool:
    """
    Determines if padding is beneficial by comparing benchmark times.
    Helper function that applies a multiplier to account for memory ops overhead.
    """
    multiplier = 1.1
    # Shape padding introduces additional memory ops. Based on microbenchmarks, 1.1x represents a reasonable
    # tradeoff between performance improvement from shape padding and overhead from additional memory ops
    # TODO: Build a learned model which would be better than this heuristic
    if "shape_padding_multiplier" in torch._inductor.config.post_grad_fusion_options:
        multiplier = torch._inductor.config.post_grad_fusion_options[
            "shape_padding_multiplier"
        ].get("value", 1.1)
        counters["inductor"]["shape_padding_multiplier"] += 1
    padded_is_faster = _skip_do_bench_times or ori_time > pad_time * multiplier
    set_cached_should_pad(key, padded_is_faster)
    return padded_is_faster


def should_pad_mm_bf16(dtype: torch.dtype, M: int, N: int, K: int) -> bool:
    # always force pad for mm with bf16 when the following are satisfied to avoid perf regression
    large_k_threshold_to_pad = torch._inductor.config.post_grad_fusion_options[
        "pad_aten_mm_pass"
    ].get("k_threshold_to_pad", 8388608)
    if (
        dtype is torch.bfloat16
        and K > M
        and K > N
        and N % 2 == 1
        and K >= large_k_threshold_to_pad
        and (torch.xpu.is_available() or torch.cuda.get_device_capability() < (9, 0))
    ):  # doesn't repro on h100s:
        return True
    return False


def should_pad(
    match: Match,
    mat1: Tensor,
    mat2: Tensor,
    op: torch._ops.OpOverloadPacket,
    input: Tensor | None = None,
) -> bool:
    _can_pad = can_pad(mat1, mat2, op, input)
    # Note that if you're tempted to insert a dynamo_timed call here, this function can
    # be called enough that the dynamo_timed overhead is not negligible.
    return _can_pad and _should_pad(match, mat1, mat2, op, input)


def get_do_bench() -> Callable[[Callable[[], Any]], float]:
    return functools.partial(
        # pyrefly: ignore [bad-argument-type]
        torch._inductor.runtime.benchmarking.benchmarker.benchmark_gpu,
        warmup=5,
    )


@memoizers.should_pad_memoizer.memoize(
    custom_params_encoder=encoders.should_pad_params_encoder
)
def _should_pad(
    match: Match,
    mat1: Tensor,
    mat2: Tensor,
    op: torch._ops.OpOverloadPacket,
    input: Tensor | None = None,
) -> bool:
    """
    Determines if an operation SHOULD be padded (performance checks).
    All logic related to whether padding would be performant should be here.
    """
    do_bench = get_do_bench()

    with no_dispatch():
        if op is torch.ops.aten.mm or op is torch.ops.aten.addmm:
            m = mat1.shape[0]
            k = mat1.shape[1]
            n = mat2.shape[1]
            k_padded_length = get_padded_length(k, get_alignment_size(mat1))
            n_padded_length = get_padded_length(n, get_alignment_size(mat2))
            m_padded_length = get_padded_length(m, get_alignment_size(mat1))
        elif op is torch.ops.aten.bmm:
            m = mat1.shape[1]
            k = mat1.shape[2]
            n = mat2.shape[2]
            k_padded_length = get_padded_length(k, get_alignment_size(mat1))
            m_padded_length = get_padded_length(m, get_alignment_size(mat1))
            n_padded_length = get_padded_length(n, get_alignment_size(mat2))
        else:
            return False

        # Force padding when explicitly requested - performance override
        if torch._inductor.config.force_shape_pad:
            return True

        # Resolve symbolic dims to concrete hints for heuristic checks below.
        # These are performance decisions, not correctness — optimization_hint is safe.
        m_concrete, k_concrete, n_concrete = hint_symbols((m, k, n))

        # Performance heuristic for bf16 large K scenarios
        if (
            "pad_aten_mm_pass" in torch._inductor.config.post_grad_fusion_options
            and should_pad_mm_bf16(mat1.dtype, m_concrete, n_concrete, k_concrete)
        ):
            return True

        # Check if operation is compute bound (performance check)
        if not is_mm_compute_bound(m_concrete, k_concrete, n_concrete, mat1.dtype):
            return False

        # We don't want to look up the cache for cases that are trivially false
        # since it does file io
        key = should_pad_bench_key(match, mat1, mat2, op, input)

        cached_pad = get_cached_should_pad(key)
        if cached_pad is not None:
            return cached_pad

        def realize_tensor(t):
            if is_fake_tensor(t):
                size_hints = hint_symbols(t.size())
                # pyrefly: ignore [bad-argument-type]
                stride_hint = hint_symbols(t.stride())
                real_size = (
                    sum((d - 1) * s for d, s in zip(size_hints, stride_hint)) + 1
                )
                real_t = torch.randn(real_size, dtype=t.dtype, device=t.device)
                return torch.as_strided(real_t, size_hints, stride_hint)
            else:
                return torch.randn_like(t)

        mat1 = realize_tensor(mat1)
        mat2 = realize_tensor(mat2)

        # since we key on whether or not the inputs can be memory planned, set cache for the
        # original time which is unaffected by whether or not the input can be planned
        ori_time_key = should_pad_bench_key(
            match, mat1, mat2, op, input, is_base_time_key=True
        )
        ori_time = get_cached_base_mm_benchmark_time(ori_time_key)
        if ori_time is None and op is torch.ops.aten.addmm and input is not None:
            # realize bias for addmm
            input = realize_tensor(input)

        mat1_pad = mat1
        mat2_pad = mat2

        is_bmm = op is torch.ops.aten.bmm

        mat1_pre_padded = should_exclude_padding_time(match, "mat1")
        fns = []
        if mat1_pre_padded and (m_padded_length or k_padded_length):
            mat1_pad = pad_mat1(
                mat1_pad,
                m_padded_length=m_padded_length,
                k_padded_length=k_padded_length,
                is_bmm=is_bmm,
            )

            def write_pad():
                if is_bmm:
                    mat1_pad[:, -m_padded_length:, -k_padded_length:].fill_(0)
                else:
                    mat1_pad[-m_padded_length:, -k_padded_length:].fill_(0)

            fns.append(write_pad)

        mat2_pre_padded = should_exclude_padding_time(match, "mat2")
        if mat2_pre_padded and (k_padded_length or n_padded_length):
            mat2_pad = pad_mat2(
                mat2_pad,
                k_padded_length=k_padded_length,
                n_padded_length=n_padded_length,
                is_bmm=is_bmm,
            )

            def write_pad():
                if is_bmm:
                    mat2_pad[:, -k_padded_length:, -n_padded_length:].fill_(0)
                else:
                    mat2_pad[-k_padded_length:, -n_padded_length:].fill_(0)

            fns.append(write_pad)

        if op is torch.ops.aten.addmm:
            input_pad = None
            if input is not None and (input.is_cuda or input.is_xpu):
                input_pad = torch.randn_like(input)
            fns.append(
                lambda: pad_addmm(
                    input_pad,
                    mat1_pad,
                    mat2_pad,
                    m_padded_length,
                    k_padded_length,
                    n_padded_length,
                    mat1_pre_padded=mat1_pre_padded,
                    mat2_pre_padded=mat2_pre_padded,
                )
            )
        elif op is torch.ops.aten.mm:
            fns.append(
                lambda: pad_mm(
                    mat1_pad,
                    mat2_pad,
                    m_padded_length,
                    k_padded_length,
                    n_padded_length,
                    mat1_pre_padded=mat1_pre_padded,
                    mat2_pre_padded=mat2_pre_padded,
                )
            )
        else:
            fns.append(
                lambda: pad_bmm(
                    mat1_pad,
                    mat2_pad,
                    m_padded_length,
                    k_padded_length,
                    n_padded_length,
                    mat1_pre_padded=mat1_pre_padded,
                    mat2_pre_padded=mat2_pre_padded,
                )
            )

        def orig_bench_fn():
            if op is torch.ops.aten.bmm or op is torch.ops.aten.mm:
                op(mat1, mat2)
            else:
                op(input, mat1, mat2)

        def pad_bench_fn():
            for fn in fns:
                fn()

        if (
            torch._inductor.config.run_autoheuristic("pad_mm")
            and op is torch.ops.aten.mm
        ):
            ah_should_pad = run_autoheuristic(
                mat1,
                mat2,
                orig_bench_fn,
                pad_bench_fn,
                m_padded_length,
                k_padded_length,
                n_padded_length,
                do_bench,
                mat1_pre_padded,
                mat2_pre_padded,
                ori_time,
                ori_time_key,
                key,
            )
            if ah_should_pad is not None:
                return ah_should_pad

        # AH didn't make a decision, so if we're in deterministic mode, we should return false
        if torch._inductor.config.deterministic:
            return False

        if ori_time is None:
            ori_time = do_bench(orig_bench_fn)
            set_cached_base_mm_benchmark_time(ori_time_key, ori_time)

        pad_time = do_bench(pad_bench_fn)

        counters["inductor"]["pad_mm_bench"] += 1
        return is_padded_faster(key, ori_time, pad_time)


def get_context(
    mat1: Tensor,
    mat2: Tensor,
    mat1_pre_padded: bool,
    mat2_pre_padded: bool,
    m_padded_length: int,
    k_padded_length: int,
    n_padded_length: int,
) -> AHContext:
    context = AHContext()

    context.add_feature("m", mat1.shape[0])
    context.add_feature("k", mat1.shape[1])
    context.add_feature("n", mat2.shape[1])

    context_add_strides(context, "mat1", mat1.stride())
    context_add_strides(context, "mat2", mat2.stride())

    context.add_feature("m_padded_length", m_padded_length)
    context.add_feature("k_padded_length", k_padded_length)
    context.add_feature("n_padded_length", n_padded_length)

    context.add_feature("mat1_align_size", get_alignment_size(mat1))
    context.add_feature("mat2_align_size", get_alignment_size(mat2))

    context.add_feature("mat1_dtype", mat1.dtype, is_categorical=True)
    context.add_feature("mat2_dtype", mat2.dtype, is_categorical=True)

    context.add_feature("prepadded_mat1", mat1_pre_padded, is_categorical=True)
    context.add_feature("prepadded_mat2", mat2_pre_padded, is_categorical=True)

    context_add_using_tf32(context, mat1.dtype)
    return context


def run_autoheuristic(
    mat1: Tensor,
    mat2: Tensor,
    orig_bench_fn: Callable[[], None],
    pad_bench_fn: Callable[[], None],
    m_padded_length: int,
    k_padded_length: int,
    n_padded_length: int,
    do_bench: Callable[[Callable[[], Any]], float],
    mat1_pre_padded: bool,
    mat2_pre_padded: bool,
    ori_time: float,
    ori_time_key: str,
    key: str,
) -> bool | None:
    def feedback_fn(
        choice: str,
    ) -> float | None:
        if choice == orig_choice:
            return do_bench(orig_bench_fn)
        elif choice == pad_choice:
            return do_bench(pad_bench_fn)
        return None

    def fallback() -> str:
        return "autotune"

    orig_choice = "orig"
    pad_choice = "pad"
    choices = [orig_choice, pad_choice]
    feedback = LocalFeedback(feedback_fn)  # type: ignore[arg-type]
    context = get_context(
        mat1,
        mat2,
        mat1_pre_padded,
        mat2_pre_padded,
        m_padded_length,
        k_padded_length,
        n_padded_length,
    )
    name = "pad_mm"
    autoheuristic = AutoHeuristic(
        fallback=fallback,
        choices=choices,
        feedback=feedback,
        context=context,
        name=name,
        augment_context=pad_mm_operations(),
        precondition=pad_mm_precondition,
    )
    choice = autoheuristic.get_choice()
    choice2should_pad = {orig_choice: False, pad_choice: True, "autotune": None}
    ah_should_pad = choice2should_pad.get(choice)

    if torch._inductor.config.collect_autoheuristic(name):
        ah_ori_time = autoheuristic.get_collected_feedback(orig_choice)
        ah_pad_time = autoheuristic.get_collected_feedback(pad_choice)

        # if precondition is not satisfied, autoheuristic does not collect data
        if ah_ori_time is not None and ah_pad_time is not None:
            if ori_time is None:
                set_cached_base_mm_benchmark_time(ori_time_key, ah_ori_time)
            return is_padded_faster(key, ah_ori_time, ah_pad_time)
    if ah_should_pad is not None:
        set_cached_should_pad(key, ah_should_pad)
    return ah_should_pad


def mm_pattern(mat1: Tensor, mat2: Tensor) -> Tensor:
    return aten.mm(mat1, mat2)


def should_pad_mm(match: Match) -> bool:
    mat1, mat2 = fetch_fake_tensors(match, ("mat1", "mat2"))
    return should_pad(match, mat1, mat2, torch.ops.aten.mm)


def pad_mat1(
    mat1: Tensor, *, m_padded_length: int, k_padded_length: int, is_bmm: bool = False
) -> Tensor:
    if k_padded_length != 0 or m_padded_length != 0:
        # dim order is reversed for constant_pad_nd, for every dim we specify right and left padding
        pad_arg = [0, k_padded_length, 0, m_padded_length]
        if is_bmm:
            pad_arg.extend((0, 0))
        return aten.constant_pad_nd(mat1, pad_arg)
    else:
        return mat1


def pad_mat2(
    mat2: Tensor, *, k_padded_length: int, n_padded_length: int, is_bmm: bool = False
) -> Tensor:
    if k_padded_length != 0 or n_padded_length != 0:
        # dim order is reversed for constant_pad_nd, for every dim we specify right and left padding
        pad_arg = [0, n_padded_length, 0, k_padded_length]
        if is_bmm:
            pad_arg.extend((0, 0))
        return aten.constant_pad_nd(mat2, pad_arg)
    else:
        return mat2


def pad_mm(
    mat1: Tensor,
    mat2: Tensor,
    m_padded_length: int,
    k_padded_length: int,
    n_padded_length: int,
    mat1_pre_padded: bool = False,
    mat2_pre_padded: bool = False,
) -> Tensor:
    if not mat1_pre_padded:
        mat1 = pad_mat1(
            mat1, m_padded_length=m_padded_length, k_padded_length=k_padded_length
        )
    if not mat2_pre_padded:
        mat2 = pad_mat2(
            mat2, k_padded_length=k_padded_length, n_padded_length=n_padded_length
        )
    res = aten.mm(mat1, mat2)
    if m_padded_length != 0:
        res = res[:-m_padded_length, :]
    if n_padded_length != 0:
        res = res[:, :-n_padded_length]
    return res


def mm_replace(mat1: Tensor, mat2: Tensor) -> Tensor:
    k_padded_length = get_padded_length(mat1.shape[1], get_alignment_size(mat1))
    m_padded_length = get_padded_length(mat1.shape[0], get_alignment_size(mat1))
    n_padded_length = get_padded_length(mat2.shape[1], get_alignment_size(mat2))
    return pad_mm(
        mat1,
        mat2,
        m_padded_length,
        k_padded_length,
        n_padded_length,
    )


def bmm_pattern(mat1: Tensor, mat2: Tensor) -> Tensor:
    return aten.bmm(mat1, mat2)


def should_pad_bmm(match: Match) -> bool:
    mat1, mat2 = fetch_fake_tensors(match, ("mat1", "mat2"))
    return should_pad(match, mat1, mat2, torch.ops.aten.bmm)


def pad_bmm(
    mat1: Tensor,
    mat2: Tensor,
    m_padded_length: int,
    k_padded_length: int,
    n_padded_length: int,
    mat1_pre_padded: bool = False,
    mat2_pre_padded: bool = False,
) -> Tensor:
    if not mat1_pre_padded:
        mat1 = pad_mat1(
            mat1,
            m_padded_length=m_padded_length,
            k_padded_length=k_padded_length,
            is_bmm=True,
        )
    if not mat2_pre_padded:
        mat2 = pad_mat2(
            mat2,
            k_padded_length=k_padded_length,
            n_padded_length=n_padded_length,
            is_bmm=True,
        )
    res = aten.bmm(mat1, mat2)
    if m_padded_length != 0:
        res = res[:, :-m_padded_length, :]
    if n_padded_length != 0:
        res = res[:, :, :-n_padded_length]
    return res


def bmm_replace(mat1: Tensor, mat2: Tensor) -> Tensor:
    k_padded_length = get_padded_length(mat1.shape[2], get_alignment_size(mat1))
    n_padded_length = get_padded_length(mat2.shape[2], get_alignment_size(mat2))
    m_padded_length = get_padded_length(mat1.shape[1], get_alignment_size(mat1))
    return pad_bmm(
        mat1,
        mat2,
        m_padded_length,
        k_padded_length,
        n_padded_length,
    )


@functools.cache
def _pad_mm_init(input_device: torch.device | None = None) -> None:
    from .joint_graph import patterns

    if input_device:
        device = str(input_device)
    else:
        if torch.cuda.is_available():
            # workaround https://github.com/pytorch/pytorch/issues/97894
            device = "cuda"
        elif torch.xpu.is_available():
            device = "xpu"
        else:
            device = "cpu"

    # sizes/values don't actually matter for initial trace
    # once we get a possible match we re-trace with the actual values and verify the match still holds

    dim2a = functools.partial(torch.empty, (4, 4), device=device, requires_grad=True)
    dim2b = functools.partial(torch.empty, (4, 4), device=device, requires_grad=True)

    dim3a = functools.partial(torch.empty, (4, 4, 4), device=device, requires_grad=True)
    dim3b = functools.partial(torch.empty, (4, 4, 4), device=device, requires_grad=True)

    dim1a = functools.partial(torch.empty, (4), device=device, requires_grad=True)

    # workaround https://github.com/pytorch/pytorch/issues/97894
    # 0.113377 is a "magic" value that lets us recover the lost input arg relationship
    rep = {"beta": 0.213377, "alpha": 0.113377}

    for pattern, replacement, args, workaround, extra_check in [
        (
            typing.cast(SearchFn, mm_pattern),
            typing.cast(ReplaceFn, mm_replace),
            [dim2a(), dim2b()],
            {},
            should_pad_mm,
        ),
        (
            typing.cast(SearchFn, bmm_pattern),
            typing.cast(ReplaceFn, bmm_replace),
            [dim3a(), dim3b()],
            {},
            should_pad_bmm,
        ),
        (
            typing.cast(SearchFn, addmm_pattern),
            typing.cast(ReplaceFn, addmm_replace),
            [dim1a(), dim2a(), dim2b()],
            rep,
            should_pad_addmm,
        ),
    ]:
        if not isinstance(
            workaround, dict
        ):  # mypy is unable to infer the type properly
            raise AssertionError(
                f"expected workaround to be a dict, got {type(workaround)}"
            )
        name = pattern.__name__

        gen_register_replacement(
            f"{name}_training",
            pattern,
            replacement,
            args,
            # pyrefly: ignore [bad-argument-type]
            joint_fwd_bwd,
            # pyrefly: ignore [bad-argument-type]
            patterns,
            extra_check=extra_check,
            scalar_workaround=workaround,
            skip_duplicates=True,
        )

        gen_register_replacement(
            f"{name}_inference",
            pattern,
            replacement,
            args,
            # pyrefly: ignore [bad-argument-type]
            fwd_only,
            # pyrefly: ignore [bad-argument-type]
            patterns,
            extra_check=extra_check,
            scalar_workaround=workaround,
            skip_duplicates=True,
        )
<|endoftext|># mypy: allow-untyped-defs
import collections
import logging

import torch
from torch.fx.experimental.symbolic_shapes import guard_or_false, statically_known_true
from torch.fx.passes.graph_transform_observer import GraphTransformObserver
from torch.fx.passes.shape_prop import _extract_tensor_metadata

from .. import config, inductor_prims
from ..pattern_matcher import (
    CallFunctionVarArgs,
    Match,
    PatternMatcherPass,
