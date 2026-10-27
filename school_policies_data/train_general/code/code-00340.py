                acc_pipeline.consumer_wait(acc_consumer_state)

                # Group together the EPI_M, EPI_M which are starting at group 3.
                # (T2R, T2R_M, T2R_N, (EPI_M, EPI_M))
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                log(f"group_modes tTR_tAcc = {tTR_tAcc!s}")
                #
                # Store accumulator to global memory in subtiles
                #
                # Use EPI_M*EPI_M to iterate using the 1-D coordinate.
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
                epi_tile_n = cutlass.const_expr(self.epi_tile_n)
                local_reduce_partial: typing.Any = None
                local_reduce_nan_partial: typing.Any = None
                if cutlass.const_expr(
                    (local_reduce_tensor is not None or self.local_reduce_feeds_main)
                    and self.local_reduce_axis == 1
                    and self.local_reduce_group > self.epi_tile_n
                ):
                    local_reduce_partial = cute.make_rmem_tensor(
                        ((1, self.epi_tile_n, 1), 1, 1), self.acc_dtype
                    )
                    if cutlass.const_expr(self.local_reduce_type in ("max", "min")):
                        local_reduce_nan_partial = cute.make_rmem_tensor(
                            ((1, self.epi_tile_n, 1), 1, 1), self.acc_dtype
                        )
                for subtile_idx in cutlass.range(subtile_cnt):
                    #
                    # Load accumulator from tensor memory buffer to register
                    #
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
                    log(f"cute.copy tiled_copy_t2r = {tiled_copy_t2r!s}")
                    log(f"cute.copy tTR_tAcc_mn = {tTR_tAcc_mn!s}")
                    log(f"cute.copy tTR_rAcc = {tTR_rAcc!s}")

                    # Wait for the EFC tensor loads to complete.  The wait is
                    # blocking even if the tx_count is 0 when there is no tensor
                    # to load. So, only wait if there is something to read,
                    # i.e. smem_read is not empty.
                    if cutlass.const_expr(self.efc.kernel.smem_read):
                        c_pipeline.consumer_wait(c_pipeline_consumer_state)

                    # Load supplemental tensors from shared memory to register.
                    self.efc.kernel.load_tensors_from_smem_to_register(
                        c_pipeline_consumer_state.index
                    )

                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    c_pipeline.consumer_release(c_pipeline_consumer_state)

                    # Advance pipeline states
                    c_pipeline_consumer_state.advance()

                    #
                    # Perform epilogue op on accumulator.
                    #
                    tiled_copy_r2s = self.efc.kernel.tiled_copy_r2s[self.d_name_bigger]
                    log(f"tiled_copy_r2s = {tiled_copy_r2s!s}")
                    # Use a SimpleNamespace to pass easily some local content as
                    # an extensible class compatible with CuTe DSL
                    # implementation.
                    epilogue_context = types.SimpleNamespace()
                    # Load the accumulator cast to the epi_dtype used to do all
                    # the computations in the epilogue.
                    # Retile the accumulator subtile to fit the destination
                    # subtile vector TV layout.
                    epilogue_context.acc_vec = (
                        tiled_copy_r2s.retile(tTR_rAcc).load().to(self.epi_dtype)
                    )
                    log(f"before .retile tTR_rAcc = {tTR_rAcc!s}")
                    log(
                        f"tiled_copy_r2s.retile(tTR_rAcc) = {tiled_copy_r2s.retile(tTR_rAcc)!s}"
                    )
                    log(
                        f"tiled_copy_r2s.retile(tTR_rAcc).load() = {tiled_copy_r2s.retile(tTR_rAcc).load()!s}"
                    )
                    log(f"epilogue_context.acc_vec = {epilogue_context.acc_vec!s}")

                    if cutlass.const_expr(self.local_reduce_direct_bool):
                        direct_fragment = cute.make_rmem_tensor_like(
                            epilogue_context.acc_vec, self.acc_dtype
                        )
                        direct_fragment.store(
                            epilogue_context.acc_vec.to(self.acc_dtype)
                        )
                        direct_flt = cute.filter_zeros(direct_fragment)
                        direct_coordinates = partition_for_epilogue(
                            cute.make_identity_tensor(self.cta_tile_shape_mnk[:2]),
                            epi_tile=epi_tile,
                            tiled_copy=tiled_copy_t2r,
                            tidx=epi_tidx,
                            reference_src=False,
                        )[(None, None, None, 0, subtile_idx)]
                        direct_coord_flt = cute.filter_zeros(direct_coordinates)
                        output_m = cute.size(mA_mkl, mode=[0])
                        output_n = cute.size(mB_nkl, mode=[0])
                        for i in cutlass.range(cute.size(direct_flt), unroll_full=True):
                            global_m = (
                                mma_tile_coord_mnl[0] * self.cta_tile_shape_mnk[0]
                                + direct_coord_flt[i][0]
                            )
                            global_n = (
                                mma_tile_coord_mnl[1] * self.cta_tile_shape_mnk[1]
                                + direct_coord_flt[i][1]
                            )
                            if global_m < output_m and global_n < output_n:
                                local_reduce_secondary_feed_tensor[
                                    global_m,
                                    global_n,
                                    mma_tile_coord_mnl[2],
                                ] = (direct_flt[i] > 0.0).to(
                                    local_reduce_secondary_feed_tensor.element_type
                                )

                    if cutlass.const_expr(
                        (
                            local_reduce_tensor is not None
                            or self.local_reduce_feeds_main
                        )
                        and self.local_reduce_axis == 0
                    ):
                        assert self.local_reduce_type in (
                            "sum",
                            "mean",
                            "mean_linear",
                            "prod",
                            "max",
                            "min",
                            "normalize_sum",
                            "normalize_sum_reverse",
                        )
                        assert self.local_reduce_source in (
                            "identity",
                            "square",
                            "abs",
                        )
                        group = cutlass.const_expr(self.local_reduce_group)
                        acc_fragment = cute.make_rmem_tensor_like(
                            epilogue_context.acc_vec, self.acc_dtype
                        )
                        acc_fragment.store(epilogue_context.acc_vec.to(self.acc_dtype))
                        reduced_flt = cute.filter_zeros(acc_fragment)
                        coordinates = partition_for_epilogue(
                            cute.make_identity_tensor(self.cta_tile_shape_mnk[:2]),
                            epi_tile=epi_tile,
                            tiled_copy=tiled_copy_t2r,
                            tidx=epi_tidx,
                            reference_src=False,
                        )
                        coordinates = coordinates[(None, None, None, 0, subtile_idx)]
                        coord_flt = cute.filter_zeros(
                            tiled_copy_r2s.retile(coordinates)
                        )
                        lane_layout_mn, warp_layout_mn = get_lane_warp_layouts(
                            tiled_copy_t2r, reference_src=False
                        )
                        lanes_in_m = cutlass.const_expr(
                            cute.size(lane_layout_mn, mode=[0])
                        )
                        assert group > 1 and group <= self.cta_tile_shape_mnk[0]
                        reduction_group = cutlass.const_expr(min(group, lanes_in_m))
                        assert group % reduction_group == 0
                        assert lanes_in_m % reduction_group == 0
                        for i in cutlass.range(
                            cute.size(reduced_flt), unroll_full=True
                        ):
                            reduced_flt[i] = self.local_reduce_source_op(reduced_flt[i])
                            rows = reduction_group // 2
                            while rows > 0:
                                other = cute.arch.shuffle_sync_bfly(
                                    reduced_flt[i],
                                    offset=cute.crd2idx((rows, 0), lane_layout_mn),
                                )
                                reduced_flt[i] = self.local_reduce_combine(
                                    reduced_flt[i], other
                                )
                                rows //= 2
                            reduced_flt[i] = self.local_reduce_finalize(
                                reduced_flt[i], group
                            )
                        groups_per_cta = cutlass.const_expr(
                            self.cta_tile_shape_mnk[0] // group
                        )
                        g_reduce: typing.Any = None
                        if cutlass.const_expr(local_reduce_tensor is not None):
                            m_reduce = local_reduce_tensor[
                                mma_tile_coord_mnl[2], None, None
                            ]
                            g_reduce = cute.local_tile(
                                m_reduce,
                                (groups_per_cta, self.cta_tile_shape_mnk[1]),
                                mma_tile_coord_mnl[:2],
                            )
                        limit_n = (
                            min(
                                cute.size(local_reduce_tensor, mode=[2])
                                - mma_tile_coord_mnl[1] * self.cta_tile_shape_mnk[1],
                                self.cta_tile_shape_mnk[1],
                            )
                            if cutlass.const_expr(local_reduce_tensor is not None)
                            else self.cta_tile_shape_mnk[1]
                        )
                        limit_groups = (
                            cute.size(local_reduce_tensor, mode=[1])
                            if cutlass.const_expr(local_reduce_tensor is not None)
                            else cute.size(mA_mkl, mode=[0]) // group
                        )
                        group_warps = cutlass.const_expr(group // lanes_in_m)
                        warp_m = warp_layout_mn[0]
                        warps_in_m = cutlass.const_expr(cute.size(warp_m))
                        assert group_warps <= warps_in_m
                        warp_idx = cute.arch.make_warp_uniform(
                            epi_tidx // cute.arch.WARP_SIZE
                        )
                        warp_m_idx = warp_layout_mn.get_hier_coord(warp_idx)[0]
                        feed_fragment = cute.make_rmem_tensor_like(
                            acc_fragment, self.acc_dtype
                        )
                        feed_fragment.store(epilogue_context.acc_vec.to(self.acc_dtype))
                        feed_flt = cute.filter_zeros(feed_fragment)
                        if cutlass.const_expr(group > lanes_in_m):
                            for i in cutlass.range(
                                cute.size(reduced_flt), unroll_full=True
                            ):
                                row_idx = coord_flt[i][0]
                                n_idx = coord_flt[i][1]
                                group_idx = row_idx // group
                                group_warp_start = group_idx * group_warps
                                group_warp_idx = warp_m_idx - group_warp_start
                                if (
                                    row_idx % lanes_in_m == 0
                                    and group_warp_idx > 0
                                    and group_warp_idx < group_warps
                                ):
                                    sLocalReduce[n_idx, warp_m_idx - group_idx - 1] = (
                                        reduced_flt[i]
                                    )
                            epilogue_sync_barrier.arrive_and_wait()
                        for i in cutlass.range(
                            cute.size(reduced_flt), unroll_full=True
                        ):
                            row_idx = coord_flt[i][0]
                            n_idx = coord_flt[i][1]
                            group_idx = row_idx // group
                            global_group_idx = (
                                mma_tile_coord_mnl[0] * groups_per_cta + group_idx
                            )
                            group_value = reduced_flt[i]
                            group_warp_start = group_idx * group_warps
                            if cutlass.const_expr(group > lanes_in_m):
                                if (
                                    row_idx % group == 0
                                    and warp_m_idx == group_warp_start
                                ):
                                    for warp_offset in cutlass.range_constexpr(
                                        1, group_warps
                                    ):
                                        other = sLocalReduce[
                                            n_idx,
                                            group_warp_start
                                            + warp_offset
                                            - group_idx
                                            - 1,
                                        ]
                                        group_value = self.local_reduce_combine(
                                            group_value, other
                                        )
                            if cutlass.const_expr(self.local_reduce_feeds_main):
                                if row_idx % group == 0 and (
                                    group <= lanes_in_m
                                    or warp_m_idx == group_warp_start
                                ):
                                    feed_flt[i] = group_value
                            if cutlass.const_expr(local_reduce_tensor is not None):
                                if (
                                    row_idx % group == 0
                                    and n_idx < limit_n
                                    and global_group_idx < limit_groups
                                ):
                                    g_reduce[group_idx, n_idx] = group_value
                        if cutlass.const_expr(group > lanes_in_m):
                            epilogue_sync_barrier.arrive_and_wait()
                        if cutlass.const_expr(self.local_reduce_feeds_main):
                            primary_acc_vec = epilogue_context.acc_vec
                            for i in cutlass.range(
                                cute.size(reduced_flt), unroll_full=True
                            ):
                                row_idx = coord_flt[i][0]
                                n_idx = coord_flt[i][1]
                                group_idx = row_idx // group
                                group_warp_start = group_idx * group_warps
                                if row_idx % group == 0 and (
                                    group <= lanes_in_m
                                    or warp_m_idx == group_warp_start
                                ):
                                    sLocalReduce[n_idx, group_idx + 3] = feed_flt[i]
                            epilogue_sync_barrier.arrive_and_wait()
                            for i in cutlass.range(
                                cute.size(reduced_flt), unroll_full=True
                            ):
                                row_idx = coord_flt[i][0]
                                n_idx = coord_flt[i][1]
                                reduced_flt[i] = sLocalReduce[
                                    n_idx, row_idx // group + 3
                                ]
                            epilogue_sync_barrier.arrive_and_wait()
                            normalized_fragment = cute.make_rmem_tensor_like(
                                acc_fragment, self.acc_dtype
                            )
                            normalized_fragment.store(
                                epilogue_context.acc_vec.to(self.acc_dtype)
                            )
                            normalized_flt = cute.filter_zeros(normalized_fragment)
                            for i in cutlass.range(
                                cute.size(normalized_flt), unroll_full=True
                            ):
                                if cutlass.const_expr(
                                    self.local_reduce_type == "mean_linear"
                                ):
                                    normalized_flt[i] = (
                                        normalized_flt[i]
                                        * self.local_reduce_input_coefficient
                                        + reduced_flt[i]
                                        * self.local_reduce_result_coefficient
                                        + self.local_reduce_bias
                                    )
                                else:
                                    denominator = (
                                        reduced_flt[i]
                                        * self.local_reduce_denominator_scale
                                        + self.local_reduce_denominator_bias
                                    )
                                    if cutlass.const_expr(
                                        self.local_reduce_type == "normalize_sum"
                                    ):
                                        normalized_flt[i] /= denominator
                                    else:
                                        assert (
                                            self.local_reduce_type
                                            == "normalize_sum_reverse"
                                        )
                                        normalized_flt[i] = (
                                            denominator / normalized_flt[i]
                                        )
                                normalized_flt[i] = (
                                    normalized_flt[i] * self.local_reduce_affine_scale
                                    + self.local_reduce_affine_bias
                                )
                            if cutlass.const_expr(local_reduce_feed_tensor is not None):
                                output_m = cute.size(mA_mkl, mode=[0])
                                output_n = cute.size(mB_nkl, mode=[0])
                                for i in cutlass.range(
                                    cute.size(normalized_flt), unroll_full=True
                                ):
                                    global_m = (
                                        mma_tile_coord_mnl[0]
                                        * self.cta_tile_shape_mnk[0]
                                        + coord_flt[i][0]
                                    )
                                    global_n = (
                                        mma_tile_coord_mnl[1]
                                        * self.cta_tile_shape_mnk[1]
                                        + coord_flt[i][1]
                                    )
                                    if global_m < output_m and global_n < output_n:
                                        local_reduce_feed_tensor[
                                            global_m,
                                            global_n,
                                            mma_tile_coord_mnl[2],
                                        ] = normalized_flt[i].to(
                                            local_reduce_feed_tensor.element_type
                                        )
                            if cutlass.const_expr(
                                local_reduce_secondary_feed_tensor is not None
                            ):
                                output_m = cute.size(mA_mkl, mode=[0])
                                output_n = cute.size(mB_nkl, mode=[0])
                                if cutlass.const_expr(
                                    self.local_reduce_secondary_reuses_primary
                                ):
                                    secondary_flt = normalized_flt
                                else:
                                    secondary_fragment = cute.make_rmem_tensor_like(
                                        acc_fragment, self.acc_dtype
                                    )
                                    secondary_fragment.store(
                                        primary_acc_vec.to(self.acc_dtype)
                                    )
                                    secondary_flt = cute.filter_zeros(
                                        secondary_fragment
                                    )
                                    for i in cutlass.range(
                                        cute.size(secondary_flt), unroll_full=True
                                    ):
                                        secondary_flt[i] *= (
                                            reduced_flt[i]
                                            * self.local_reduce_secondary_scale
                                            + self.local_reduce_secondary_bias
                                        )
                                for i in cutlass.range(
                                    cute.size(secondary_flt), unroll_full=True
                                ):
                                    global_m = (
                                        mma_tile_coord_mnl[0]
                                        * self.cta_tile_shape_mnk[0]
                                        + coord_flt[i][0]
                                    )
                                    global_n = (
                                        mma_tile_coord_mnl[1]
                                        * self.cta_tile_shape_mnk[1]
                                        + coord_flt[i][1]
                                    )
                                    if global_m < output_m and global_n < output_n:
                                        local_reduce_secondary_feed_tensor[
                                            global_m,
                                            global_n,
                                            mma_tile_coord_mnl[2],
                                        ] = secondary_flt[i].to(
                                            local_reduce_secondary_feed_tensor.element_type
                                        )
                            if cutlass.const_expr(local_reduce_feed_tensor is not None):
                                epilogue_context.acc_vec = primary_acc_vec
                            else:
                                epilogue_context.acc_vec = normalized_fragment.load()

                    if cutlass.const_expr(
                        (
                            local_reduce_tensor is not None
                            and self.local_reduce_axis == 1
                        )
                        or (
                            self.local_reduce_feeds_main and self.local_reduce_axis == 1
                        )
                    ):
                        assert self.local_reduce_type in (
                            "sum",
                            "mean",
                            "prod",
                            "max",
                            "min",
                            "logsumexp",
                            "variance",
                            "normalize_sum",
                            "normalize_sum_reverse",
                        )
                        assert self.local_reduce_source in (
                            "identity",
                            "square",
                            "abs",
                        )
                        group = cutlass.const_expr(self.local_reduce_group)
                        fragment_n = epi_tile_n
                        n_subtiles = cutlass.const_expr(
                            self.cta_tile_shape_mnk[1] // fragment_n
                        )
                        fragments_per_group = cutlass.const_expr(
                            max(1, group // fragment_n)
                        )
                        subtile_n_idx = subtile_idx % n_subtiles
                        assert group > 1
                        acc_fragment = cute.make_rmem_tensor_like(
                            epilogue_context.acc_vec, self.acc_dtype
                        )
                        acc_fragment.store(epilogue_context.acc_vec.to(self.acc_dtype))
                        acc_flt = cute.filter_zeros(acc_fragment)
                        wide_reduced = None
                        wide_nan = None
                        if cutlass.const_expr(group > fragment_n):
                            local_reduce_vec = epilogue_context.acc_vec.to(
                                self.acc_dtype
                            )
                            if cutlass.const_expr(
                                self.local_reduce_type in ("max", "min")
                            ):
                                nan_grouped = (
                                    (local_reduce_vec != local_reduce_vec)
                                    .to(self.acc_dtype)
                                    .reshape(((1, fragment_n, 1), 1, 1))
                                )
                                wide_nan = nan_grouped.reduce(
                                    cute.ReductionOp.ADD,
                                    init_val=0.0,
                                    reduction_profile=((None, 1, None), 1, 1),
                                )
                                wide_nan = wide_nan.reshape(
                                    ((1, 1, 1), 1, 1)
                                ).broadcast_to(nan_grouped.shape)
                            local_reduce_vec = self.local_reduce_source_op(
                                local_reduce_vec
                            )
                            fragment_width = cutlass.const_expr(
                                cute.size(local_reduce_vec.shape, mode=[0])
                            )
                            grouped = local_reduce_vec.reshape(
                                ((1, fragment_width, 1), 1, 1)
                            )
                            wide_reduced = grouped.reduce(
                                self.local_reduce_op,
                                init_val=self.local_reduce_init,
                                reduction_profile=((None, 1, None), 1, 1),
                            )
                            wide_reduced = wide_reduced.reshape(
                                ((1, 1, 1), 1, 1)
                            ).broadcast_to(grouped.shape)
                        reduced_fragment = cute.make_rmem_tensor_like(
                            acc_fragment, cutlass.Float32
                        )
                        reduced_flt = cute.filter_zeros(reduced_fragment)
                        wide_partial_fragment = cute.make_rmem_tensor_like(
                            acc_fragment, cutlass.Float32
                        )
                        wide_partial_flt = cute.filter_zeros(wide_partial_fragment)
                        for i in cutlass.range(
                            cute.size(wide_partial_flt), unroll_full=True
                        ):
                            wide_partial_flt[i] = self.local_reduce_init
                        coordinates = partition_for_epilogue(
                            cute.make_identity_tensor(self.cta_tile_shape_mnk[:2]),
                            epi_tile=epi_tile,
                            tiled_copy=tiled_copy_t2r,
                            tidx=epi_tidx,
                            reference_src=False,
                        )
                        coordinates = coordinates[(None, None, None, 0, subtile_idx)]
                        coord_flt = cute.filter_zeros(coordinates)
                        acc_coord_flt = cute.filter_zeros(
                            tiled_copy_r2s.retile(coordinates)
                        )
                        groups_per_cta = cutlass.const_expr(
                            self.cta_tile_shape_mnk[1] // group
                        )
                        g_reduce: typing.Any = None
                        if cutlass.const_expr(local_reduce_tensor is not None):
                            m_reduce = local_reduce_tensor[
                                mma_tile_coord_mnl[2], None, None
                            ]
                            g_reduce = cute.local_tile(
                                m_reduce,
                                (self.cta_tile_shape_mnk[0], groups_per_cta),
                                mma_tile_coord_mnl[:2],
                            )
                        limit_m = min(
                            cute.size(mA_mkl, mode=[0])
                            - mma_tile_coord_mnl[0] * self.cta_tile_shape_mnk[0],
                            self.cta_tile_shape_mnk[0],
                        )
                        limit_groups = (
                            cute.size(local_reduce_tensor, mode=[2])
                            if cutlass.const_expr(local_reduce_tensor is not None)
                            else cute.size(mB_nkl, mode=[0]) // group
                        )
                        for i in cutlass.range(cute.size(acc_flt), unroll_full=True):
                            row_idx = coord_flt[i][0]
                            n_idx = coord_flt[i][1]
                            logical_row = row_idx
                            group_idx = (
                                subtile_n_idx // fragments_per_group
                                if cutlass.const_expr(group > fragment_n)
                                else n_idx // group
                            )
                            global_group_idx = (
                                mma_tile_coord_mnl[1] * groups_per_cta + group_idx
                            )
                            if (
                                (
                                    self.local_reduce_feeds_main
                                    or n_idx
                                    % (
                                        fragment_n
                                        if cutlass.const_expr(group > fragment_n)
                                        else group
                                    )
                                    == 0
                                )
                                and logical_row < limit_m
                                and global_group_idx < limit_groups
                            ):
                                reduced = self.local_reduce_init
                                for j in cutlass.range(
                                    cute.size(acc_flt), unroll_full=True
                                ):
                                    if coord_flt[j][0] == row_idx and (
                                        cutlass.const_expr(group > fragment_n)
                                        or coord_flt[j][1] // group == group_idx
                                    ):
                                        value = acc_flt[j]
                                        value = self.local_reduce_source_op(value)
                                        reduced = self.local_reduce_combine(
                                            reduced, value
                                        )
                                if cutlass.const_expr(
                                    self.local_reduce_type == "logsumexp"
                                    and group <= fragment_n
                                ):
                                    shift = reduced
                                    if cute.math.abs(shift) == cutlass.Float32.inf:
                                        shift = cutlass.Float32(0.0)
                                    exp_sum = cutlass.Float32(0.0)
                                    for j in cutlass.range(
                                        cute.size(acc_flt), unroll_full=True
                                    ):
                                        if (
                                            coord_flt[j][0] == row_idx
                                            and coord_flt[j][1] // group == group_idx
                                        ):
                                            exp_sum += cute.math.exp(acc_flt[j] - shift)
                                    reduced = cute.math.log(exp_sum) + shift
                                elif cutlass.const_expr(
                                    self.local_reduce_type == "variance"
                                    and group <= fragment_n
                                ):
                                    mean = reduced / group
                                    reduced = cutlass.Float32(0.0)
                                    for j in cutlass.range(
                                        cute.size(acc_flt), unroll_full=True
                                    ):
                                        if (
                                            coord_flt[j][0] == row_idx
                                            and coord_flt[j][1] // group == group_idx
                                        ):
                                            delta = acc_flt[j] - mean
                                            reduced += delta * delta
                                    reduced = (
                                        reduced
                                        / group
                                        * self.local_reduce_variance_scale
                                        + self.local_reduce_variance_bias
                                    )
                                elif cutlass.const_expr(group <= fragment_n):
                                    reduced = self.local_reduce_finalize(reduced, group)
                                if cutlass.const_expr(self.local_reduce_feeds_main):
                                    reduced_flt[i] = reduced
                                if cutlass.const_expr(group > fragment_n):
                                    wide_partial_flt[i] = reduced
                                if cutlass.const_expr(local_reduce_tensor is not None):
                                    if cutlass.const_expr(group > fragment_n):
                                        if n_idx % fragment_n == 0:
                                            fragment_idx = (
                                                subtile_n_idx % fragments_per_group
                                            )
                                            sLocalReduce[logical_row, fragment_idx] = (
                                                reduced
                                            )
                                    elif n_idx % group == 0:
                                        g_reduce[logical_row, group_idx] = reduced
                        if cutlass.const_expr(
                            group > fragment_n
                            and local_reduce_tensor is not None
                            and self.local_reduce_type == "sum"
                        ):
                            sum_fragment_idx = subtile_n_idx % fragments_per_group
                            sum_group_idx = subtile_n_idx // fragments_per_group
                            sum_global_group_idx = (
                                mma_tile_coord_mnl[1] * groups_per_cta + sum_group_idx
                            )
                            sum_row_idx = epi_tidx
                            if sum_fragment_idx == 0:
                                if (
                                    sum_row_idx < limit_m
                                    and sum_global_group_idx < limit_groups
                                ):
                                    shared_element = cute.make_tensor(
                                        sLocalReduce.iterator + sum_row_idx,
                                        cute.make_layout(1),
                                    )
                                    shared_element[0] = cutlass.Float32(0.0)
                            epilogue_sync_barrier.arrive_and_wait()
                            for i in cutlass.range(
                                cute.size(acc_flt), unroll_full=True
                            ):
                                if (
                                    sum_row_idx < limit_m
                                    and sum_global_group_idx < limit_groups
                                ):
                                    cute.arch.atomic_add(
                                        sLocalReduce.iterator + sum_row_idx,
                                        acc_flt[i],
                                        scope="cta",
                                    )
                            epilogue_sync_barrier.arrive_and_wait()
                            if sum_fragment_idx == fragments_per_group - 1:
                                if (
                                    sum_row_idx < limit_m
                                    and sum_global_group_idx < limit_groups
                                ):
                                    shared_element = cute.make_tensor(
                                        sLocalReduce.iterator + sum_row_idx,
                                        cute.make_layout(1),
                                    )
                                    output_element = cute.make_tensor(
                                        g_reduce.iterator  # pyrefly: ignore[unbound-name]
                                        + sum_row_idx * limit_groups,
                                        cute.make_layout(1),
                                    )
                                    output_element[0] = shared_element[0]
                        if cutlass.const_expr(
                            group > fragment_n and self.local_reduce_type != "sum"
                        ):
                            fragment_idx = subtile_n_idx % fragments_per_group
                            partial = local_reduce_partial.load()
                            current = wide_reduced
                            if fragment_idx == 0:
                                partial = current
                            else:
                                partial = self.local_reduce_combine(partial, current)
                            local_reduce_partial.store(partial)
                            if cutlass.const_expr(
                                self.local_reduce_type in ("max", "min")
                            ):
                                nan_partial = local_reduce_nan_partial.load()
                                if fragment_idx == 0:
                                    nan_partial = wide_nan
                                else:
                                    nan_partial += wide_nan
                                local_reduce_nan_partial.store(nan_partial)
                            if fragment_idx == fragments_per_group - 1:
                                final_fragment = cute.make_rmem_tensor_like(
                                    local_reduce_partial, cutlass.Float32
                                )
                                final_fragment.store(partial)
                                final_flt = cute.filter_zeros(final_fragment)
                                nan_flt: typing.Any = None
                                if cutlass.const_expr(
                                    self.local_reduce_type in ("max", "min")
                                ):
                                    nan_fragment = cute.make_rmem_tensor_like(
                                        local_reduce_nan_partial, cutlass.Float32
                                    )
                                    nan_fragment.store(local_reduce_nan_partial.load())
                                    nan_flt = cute.filter_zeros(nan_fragment)
                                store_coordinates = partition_for_epilogue(
                                    cute.make_identity_tensor(
                                        self.cta_tile_shape_mnk[:2]
                                    ),
                                    epi_tile=epi_tile,
                                    tiled_copy=tiled_copy_t2r,
                                    tidx=epi_tidx,
                                    reference_src=False,
                                )
                                store_coordinates = store_coordinates[
                                    (
                                        None,
                                        None,
                                        None,
                                        0,
                                        subtile_idx - fragment_idx,
                                    )
                                ]
                                store_coord_flt = cute.filter_zeros(store_coordinates)
                                for i in cutlass.range(
                                    cute.size(final_flt), unroll_full=True
                                ):
                                    row_idx = store_coord_flt[i][0]
                                    n_idx = store_coord_flt[i][1]
                                    group_idx = subtile_n_idx // fragments_per_group
                                    global_group_idx = (
                                        mma_tile_coord_mnl[1] * groups_per_cta
                                        + group_idx
                                    )
                                    if (
                                        n_idx % fragment_n == 0
                                        and row_idx < limit_m
                                        and global_group_idx < limit_groups
                                    ):
                                        value = final_flt[i]
                                        if cutlass.const_expr(
                                            self.local_reduce_type in ("max", "min")
                                        ):
                                            if nan_flt[i] > 0.0:
                                                value = float("nan")
                                        value = self.local_reduce_finalize(value, group)
                                        g_reduce[row_idx, group_idx] = value
                        if cutlass.const_expr(self.local_reduce_feeds_main):
                            primary_acc_vec = epilogue_context.acc_vec
                            for i in cutlass.range(
                                cute.size(acc_flt), unroll_full=True
                            ):
                                denominator = (
                                    reduced_flt[i] * self.local_reduce_denominator_scale
                                    + self.local_reduce_denominator_bias
                                )
                                if cutlass.const_expr(
                                    self.local_reduce_type == "normalize_sum"
                                ):
                                    value = acc_flt[i] / denominator
                                else:
                                    value = denominator / acc_flt[i]
                                acc_flt[i] = (
                                    value * self.local_reduce_affine_scale
                                    + self.local_reduce_affine_bias
                                )
                            if cutlass.const_expr(local_reduce_feed_tensor is not None):
                                output_m = cute.size(mA_mkl, mode=[0])
                                output_n = cute.size(mB_nkl, mode=[0])
                                for i in cutlass.range(
                                    cute.size(acc_flt), unroll_full=True
                                ):
                                    global_m = (
                                        mma_tile_coord_mnl[0]
                                        * self.cta_tile_shape_mnk[0]
                                        + acc_coord_flt[i][0]
                                    )
                                    global_n = (
                                        mma_tile_coord_mnl[1]
                                        * self.cta_tile_shape_mnk[1]
                                        + acc_coord_flt[i][1]
                                    )
                                    if global_m < output_m and global_n < output_n:
                                        local_reduce_feed_tensor[
                                            global_m,
                                            global_n,
                                            mma_tile_coord_mnl[2],
                                        ] = acc_flt[i].to(
                                            local_reduce_feed_tensor.element_type
                                        )
                            if cutlass.const_expr(local_reduce_feed_tensor is not None):
                                epilogue_context.acc_vec = primary_acc_vec
                            else:
                                epilogue_context.acc_vec = acc_fragment.load().to(
                                    self.epi_dtype
                                )

                    # Execute the EFC epilogue.
                    self.efc.kernel.epilogue_computation(epilogue_context)
                    d_buffer = (num_prev_subtiles + subtile_idx) % self.num_d_stage

                    # Store the EFC written tensors to shared memory.
                    self.efc.kernel.store_written_tensors_to_smem(d_buffer)

                    # Fence and barrier to make sure shared memory store is visible to TMA store
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    epilogue_sync_barrier.arrive_and_wait()

                    #
                    # TMA store D to global memory
                    #
                    if warp_idx == self.epilogue_warp_id[0]:
                        # Store with TMA the written EFC tensors to global memory.
                        self.efc.kernel.tma_store_written_tensors_to_gmem(
                            d_buffer, subtile_idx
                        )
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        d_pipeline.producer_commit()
                        d_pipeline.producer_acquire()

                    epilogue_sync_barrier.arrive_and_wait()

                #
                # Async arrive accumulator buffer empty
                #
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            epilogue_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
            #
            # Wait for D store complete
            #
            d_pipeline.producer_tail()

        #
        # Specialized epilogue load warp
        #
        if warp_idx == self.epilogue_load_warp_id:
            # Create the tiled tensors to be loaded in the epilogue.
            self.efc.kernel.create_epilogue_subtile_tensors(tidx, epi_tile)

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            c_pipeline_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_c_stage
            )

            # Setup the pipelines reading the EFC supplemental tensors.

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                # Prepare the EFC tensors to be loaded by the subtiles.
                subtile_cnt = self.efc.kernel.prepare_tensor_load_for_subtiles(
                    mma_tile_coord_mnl,
                )

                # Assume the pipeline can work even in the case there is no
                # tensor to load and so subtile_cnt is 0.
                for subtile_idx in cutlass.range(subtile_cnt):
                    # Load C from global memory to shared memory.
                    c_pipeline.producer_acquire(c_pipeline_producer_state)

                    # Load the subtiles of EFC tensors.
                    self.efc.kernel.load_tensor_subtiles(
                        subtile_idx, c_pipeline, c_pipeline_producer_state
                    )

                    c_pipeline_producer_state.advance()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Wait for the load buffer to be empty.
            #
            c_pipeline.producer_tail(c_pipeline_producer_state)

    def epilogue_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        tCgC: cute.Tensor,
        epi_tile: cute.Tile,
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """Make tiledCopy for tensor memory load, then use it to partition tensor memory (source) and register array (destination)."""
        # Make tiledCopy for tensor memory load
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.d_layout,  # Take this as the reference layout for the epilogue tile.
            self.epi_dtype,  # But we get the accumulator as epi_dtype in the epilogue.
            self.acc_dtype,
            epi_tile,
            self.use_2cta_instrs,
        )

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        tCgC_epi = cute.flat_divide(
            tCgC[((None, None), 0, 0, None, None, None)], epi_tile
        )
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
        tTR_gC = thr_copy_t2r.partition_D(tCgC_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilogue_smem_copy_and_partition_load(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """Make tiledCopy for shared memory load, then use it to partition register array (destination) and shared memory (source).

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rC: The partitioned accumulator tensor
        :type tTR_rC: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor

        :return: A tuple containing (tiled_copy_s2r, tSR_rC, tSR_sC) where:
            - tiled_copy_s2r: The tiled copy operation for smem to register copy(s2r)
            - tSR_rC: The partitioned tensor C (register destination)
            - tSR_sC: The partitioned tensor C (smem source)
        :rtype: tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        copy_atom_s2r = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.c_dtype)
        tiled_copy_s2r = cute.make_tiled_copy_D(copy_atom_s2r, tiled_copy_t2r)
        # (S2R, S2R_M, S2R_N, PIPE_C)
        thr_copy_s2r = tiled_copy_s2r.get_slice(tidx)
        tSR_sC = thr_copy_s2r.partition_D(sC)
        # (S2R, S2R_M, S2R_N)
        tSR_rC = tiled_copy_s2r.retile(tTR_rC)
        return tiled_copy_s2r, tSR_rC, tSR_sC

    def epilogue_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: cute.CopyAtom | cute.TiledCopy,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
        dtype: type[cutlass.Numeric],
    ) -> tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """Make tiledCopy for global memory store, then use it to:
        - partition register array (source) and global memory (destination) for non-TMA store version;
        - partition shared memory (source) and global memory (destination) for TMA store version.

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param atom: The copy_atom_c to be used for TMA store version, or tiled_copy_t2r for non-TMA store version
        :type atom: cute.CopyAtom or cute.TiledCopy
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor

        :return: A tuple containing either:
            - For TMA store: (tma_atom_c, bSG_sC, bSG_gC) where:
                - tma_atom_c: The TMA copy atom
                - bSG_sC: The partitioned shared memory tensor C
                - bSG_gC: The partitioned global tensor C
            - For non-TMA store: (simt_atom, tTR_rC, tTR_gC) where:
                - simt_atom: The SIMT copy atom
                - tTR_rC: The register tensor C
                - tTR_gC: The partitioned global tensor C
        :rtype: tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]
        """
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, tiles_m, tiles_n, tiles_l)
        gC_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )

        tma_atom_c = atom
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, tiles_m, tiles_n, tiles_l)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        return tma_atom_c, bSG_sC, bSG_gC

    def compute_stages(self) -> None:
        """Compute and set the number of stages for A/B/C/D operands.

        Uses instance attributes to compute and assign:
        `self.num_acc_stage`, `self.num_ab_stage`, `self.num_c_stage`,
        and `self.num_d_stage`.
        """
        # Defaults
        self.num_acc_stage = 2
        # To read the tensors needed for the epilogue:
        self.num_c_stage = 2
        # To write the tensors produced by the epilogue:
        self.num_d_stage = 2

        # Calculate smem layout and size for one stage of A, B, C, and D
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            self._tiled_mma, self.mma_tiler, self.a_dtype, 1
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            self._tiled_mma, self.mma_tiler, self.b_dtype, 1
        )

        # Get the contribution from the tensors used in the EFC.
        self.efc.jit.compute_stage()

        ab_bytes_per_stage = cute.size_in_bytes(
            self.a_dtype, a_smem_layout_stage_one
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout_staged_one)
        mbar_helpers_bytes = 2048 + self.local_reduce_smem_bytes
        # Contribution from the tensors loaded in the EFC.
        c_bytes_per_stage = self.efc.jit.smem_size_in_bytes_of_read_tensors()
        c_bytes = c_bytes_per_stage * self.num_c_stage
        # Contribution from the tensors stored in the EFC. There is at least 1
        # written tensor, so the following is strictly positive.
        d_bytes_per_stage = self.efc.jit.smem_size_in_bytes_of_written_tensors()
        d_bytes = d_bytes_per_stage * self.num_d_stage

        # Calculate A/B stages
        self.num_ab_stage = (
            self.smem_capacity // self.occupancy
            - (mbar_helpers_bytes + c_bytes + d_bytes)
        ) // ab_bytes_per_stage
        log(f"{common_efc.TAB}{self.num_ab_stage = }")

        if self.num_ab_stage <= 0:
            raise MemoryError("Not enough smem capacity to allocate all the tensors.")

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B stages and reserved bytes.
        # Add remaining unused smem to epilogue.
        self.num_d_stage += (
            self.smem_capacity
            - self.occupancy * ab_bytes_per_stage * self.num_ab_stage
            - self.occupancy * (mbar_helpers_bytes + c_bytes + d_bytes)
        ) // (self.occupancy * d_bytes_per_stage)
        log(f"{common_efc.TAB}new {self.num_d_stage = }")

    @staticmethod
    def _compute_grid(
        d: cute.Tensor,
        cta_tile_shape_mnk: tuple[int, int, int],
        cluster_shape_mn: tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]:
        """Use persistent tile scheduler to compute the grid size for the output tensor D.

        :param d: The output tensor D
        :type d: cute.Tensor
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr

        :return: A tuple containing:
            - tile_sched_params: Parameters for the persistent tile scheduler.
            - grid: Grid shape for kernel launch.
        :rtype: tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]
        """
        log(f"compute_grid: {max_active_clusters = }")
        d_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gd = cute.zipped_divide(d, tiler=d_shape)
        num_ctas_mnl = gd[(0, (None, None, None))].shape
        common_efc.if_debug(
            lambda: cute.printf("compute_grid: num_ctas_mnl = {}", num_ctas_mnl)
        )
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        common_efc.if_debug(lambda: cute.printf("compute_grid: grid = {}", grid))
        return tile_sched_params, grid

    def compute_num_tmem_alloc_cols(self) -> None:
        """Compute and set the number of tensor memory allocation columns.

        This method uses the instance attributes computed during setup to
        determine the number of tensor memory allocation columns and stores
        the result in `self.num_tmem_alloc_cols`.
        """
        acc_shape = self._tiled_mma.partition_shape_C(self.mma_tiler[:2])
        log(f"compute_num_tmem_alloc_cols: {acc_shape = !s}")
        tCtAcc_fake = self._tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )
        log(f"compute_num_tmem_alloc_cols: {tCtAcc_fake = !s}")
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)
        log(f"compute_num_tmem_alloc_cols: {self.num_tmem_alloc_cols = }")
<|endoftext|># Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: S101

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
from typing import Tuple, Type, Union

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

import torch
from torch._inductor.kernel.vendored_templates.cutedsl.reduction_utils import (
    get_lane_warp_layouts,
    partition_for_epilogue,
)


"""
This example provides an experimental implementation of the SM100 batched dense blockscaled GEMM kernel, please note that the APIs and implementation details related to this kernel may change in future releases.

A high-performance persistent batched dense blockscaled GEMM example for the NVIDIA Blackwell SM100 architecture
using CUTE DSL.
- Matrix A is MxKxL, L is batch dimension, A can be row-major("K") or column-major("M") for MXF8 input type and can only be row-major("K") for MXF4/NVF4 input type
- Matrix B is NxKxL, L is batch dimension, B can be row-major("N") or column-major("K") for MXF8 input type and can only be row-major("K") for MXF4/NVF4 input type
- Matrix C is MxNxL, L is batch dimension, C can be row-major("N") or column-major("M")
- Matrix SFA layout is filled internally according to A shape and BlockScaledBasicChunk, which has M×ceil_div(K, sf_vec_size)×L elements respectively
- Matrix SFB layout is filled internally according to B shape and BlockScaledBasicChunk, which has N×ceil_div(K, sf_vec_size)×L elements respectively

This GEMM kernel supports the following features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations
    - Utilizes Blackwell's tcgen05.mma for matrix multiply-accumulate (MMA) operations (including 2cta mma instructions)
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Support persistent tile scheduling to better overlap memory load/store with mma between tiles
    - Support warp specialization to avoid explicit pipelining between mainloop load and mma

This GEMM works as follows:
1. DMA warp: Load A and B matrices from global memory (GMEM) to shared memory (SMEM) using TMA operations.
2. MMA warp:
    - Load scale factor A/B from shared memory (SMEM) to tensor memory (TMEM) using tcgen05.cp instruction.
    - Perform matrix multiply-accumulate (MMA) operations using tcgen05.mma instruction.
3. EPILOGUE warp:
    - Load completed accumulator from tensor memory (TMEM) to registers (RMEM) using tcgen05.ld.
    - Type convert C matrix to output type.
    - Optionally store C matrix from registers (RMEM) to shared memory (SMEM) to global memory (GMEM) with TMA operations,
      or directly store C matrix from registers (RMEM) to global memory (GMEM) without TMA operations.
    - Optionally accept an elementwise lambda function epilogue_op to apply to the output tensor:
      e.g., relu can set epilogue_op = lambda x: cute.where(x > 0, x, cute.full_like(x, 0))

SM100 tcgen05.mma.kind.block_scale instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Read scalefactor A from TMEM
- Read scalefactor B from TMEM
- Write accumulator to TMEM
The accumulator in TMEM must then be loaded to registers before writing back to GMEM.

Input arguments to this example is shown below:

.. code-block:: bash

    python examples/blackwell/dense_blockscaled_gemm_persistent.py             \
      --ab_dtype Float4E2M1FN --sf_dtype Float8E8M0FNU --sf_vec_size 16        \
      --c_dtype Float16                                                        \
      --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                            \
      --mnkl 8192,8192,1024,1

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/blackwell/dense_blockscaled_gemm_persistent.py         \
      --ab_dtype Float4E2M1FN --sf_dtype Float8E8M0FNU --sf_vec_size 16        \
      --c_dtype Float16                                                        \
      --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                            \
      --mnkl 8192,8192,1024,1                                                  \
      --warmup_iterations 1 --iterations 10 --skip_ref_check


Constraints:
* Supported input data types: mxf8, mxf4, nvf4
  see detailed valid dtype combinations in below Sm100BlockScaledPersistentDenseGemmKernel class documentation
* A/B tensor must have the same data type, mixed data type is not supported (e.g., mxf8 x mxf4)
* Mma tiler M must be 128 or 256(use_2cta_instrs)
* Mma tiler N must be 64/128/192/256
* Cluster shape M/N must be positive and power of 2, total cluster size <= 16
* Cluster shape M must be multiple of 2 if Mma tiler M is 256(use_2cta_instrs)
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 16 and 32 for Float8 and Float4, respectively.
"""


class Sm100BlockScaledPersistentDenseGemmKernel:
    """This class implements batched matrix multiplication (C = A x SFA x B x SFB) with support for various data types
    and architectural features specific to Blackwell GPUs with persistent tile scheduling and warp specialization.

    :param sf_vec_size: Scalefactor vector size.
    :type sf_vec_size: int
    :param mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tile (M,N)
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]

    :note: In current version, A and B tensor must have the same data type
        - i.e., Float8E4M3FN for A and Float8E5M2 for B is not supported

    :note: Supported combinations of A/B data types, SF data typs and SF vector size:
        - MXF8: A/B: Float8E5M2/Float8E4M3FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - MXF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - NVF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU/Float8E4M3FN + sf_vec_size: 16

    :note: Supported accumulator data types:
        - Float32

    :note: Supported C data types:
        - Float32
        - Float16/BFloat16
        - Float8E4M3FN/Float8E5M2
    :note: Constraints:
        - MMA tiler M must be 128 or 256 (use_2cta_instrs)
        - MMA tiler N must be 64/128/192/256
        - Cluster shape M must be multiple of 2 if Mma tiler M is 256
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16
        - Also, Cluster shape M/N must be <= 4 for scale factor multicasts due to limited size of scale factors

    Example:
        >>> # xdoctest: +SKIP(requires cutlass)
        >>> gemm = Sm100BlockScaledPersistentDenseGemmKernel(
        ...     sf_vec_size=16, mma_tiler_mn=(256, 128), cluster_shape_mn=(2, 1)
        ... )
        >>> gemm(
        ...     a_tensor,
        ...     b_tensor,
        ...     sfa_tensor,
        ...     sfb_tensor,
        ...     c_tensor,
        ...     max_active_clusters,
        ...     stream,
        ... )
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
    ):
        """Initializes the configuration for a Blackwell dense GEMM kernel.

        This configuration includes several key aspects:

        1.  MMA Instruction Settings (tcgen05):
            - acc_dtype: Data types for MMA accumulator, always set to Float32
            - sf_vec_size: Scalefactor A/B vector size.
            - mma_tiler_mn: The (M, N) shape of the MMA instruction tiler.

        2.  Cluster Shape:
            - cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster.

        :param sf_vec_size: Scalefactor vector size.
        :type sf_vec_size: int
        :param mma_tiler_mn: Tuple (M, N) shape of the MMA instruction.
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: Tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mn: Tuple[int, int]
        """

        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cluster_shape_mn = cluster_shape_mn
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (*mma_tiler_mn, 1)

        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        # TMA-prefetch tactic: prefetch A/B/SF tiles ahead in the K-loop to
        # hide latency; helps small-M large-K.
        self.use_prefetch = use_prefetch

        self.occupancy = 1
        # Set specialized warp ids
        self.epilog_warp_id = (
            0,
            1,
            2,
            3,
        )
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * len(
            (self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id)
        )
        # Set barrier id for epilogue sync and tmem ptr sync
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=32 * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.num_tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B/SFA/SFB
        - Computing epilogue subtile
        - Setting up A/B/SFA/SFB/C stage counts in shared memory
        - Computing A/B/SFA/SFB/C shared memory layout
        """
        # Compute mma instruction shapes
        # (MMA_Tile_Shape_M, MMA_Tile_Shape_N, MMA_Inst_Shape_K)
        self.mma_inst_shape_mn = (
            self.mma_tiler[0],
            self.mma_tiler[1],
        )
        # (CTA_Tile_Shape_M, Round_Up(MMA_Tile_Shape_N, 128), MMA_Inst_Shape_K)
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )

        # Compute mma/cluster/tile shapes
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_inst_shape_mn[0],
            self.mma_inst_shape_mn[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        # Compute epilogue subtile
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )
        self.epi_tile_n = cute.size(self.epi_tile[1])

        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.smem_capacity,
            self.occupancy,
        )

        # Prefetch depth = AB pipeline depth.
        self.prefetch_dist = self.num_ab_stage

        # Compute A/B/SFA/SFB/C shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage,
        )

        # Overlap and double buffer accumulator when num_acc_stage == 1 for cta_tile_n = 256 case
        self.overlapping_accum = self.num_acc_stage == 1

        # Compute number of TMEM columns for SFA/SFB/Accumulator
        sf_atom_mn = 32
        self.num_sfa_tmem_cols = (
            self.cta_tile_shape_mnk[0] // sf_atom_mn
        ) * mma_inst_tile_k
        self.num_sfb_tmem_cols = (
            self.cta_tile_shape_mnk_sfb[1] // sf_atom_mn
        ) * mma_inst_tile_k
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = (
            self.cta_tile_shape_mnk[1] * self.num_acc_stage
            if not self.overlapping_accum
            else self.cta_tile_shape_mnk[1] * 2 - self.num_sf_tmem_cols
        )

        # Only when overlapping_accum is enabled, we need to release accumulator buffer early in epilogue
        self.iter_acc_early_release_in_epilogue = (
            self.num_sf_tmem_cols // self.epi_tile_n
        )

    @cute.jit
    def __call__(
        self,
        a_tensor: cute.Tensor,
        b_tensor: cute.Tensor,
        sfa_tensor: cute.Tensor,
        sfb_tensor: cute.Tensor,
        c_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
        epilogue_input_dtype: cutlass.Constexpr = cutlass.Float32,
        alpha_tensor: cute.Tensor = None,
        epilogue_tensor0: cute.Tensor = None,
        epilogue_tensor1: cute.Tensor = None,
        epilogue_tensor2: cute.Tensor = None,
        epilogue_tensor3: cute.Tensor = None,
        epilogue_tensor_kind0: cutlass.Constexpr = 0,
        epilogue_tensor_kind1: cutlass.Constexpr = 0,
        epilogue_tensor_kind2: cutlass.Constexpr = 0,
        epilogue_tensor_kind3: cutlass.Constexpr = 0,
        epilogue_output0: cute.Tensor = None,
        epilogue_output1: cute.Tensor = None,
        epilogue_output2: cute.Tensor = None,
        epilogue_output_count: cutlass.Constexpr = 1,
        primary_epilogue_output: cutlass.Constexpr = 0,
        local_reduce_tensor: cute.Tensor = None,
        local_reduce_group: cutlass.Constexpr = 0,
        local_reduce_axis: cutlass.Constexpr = 1,
        local_reduce_type: cutlass.Constexpr = "sum",
        local_reduce_source: cutlass.Constexpr = "identity",
        local_reduce_feeds_main: cutlass.Constexpr = False,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes before smem/grid/tma computation
        - Setup TMA load/store atoms and tensors
        - Compute grid size with regard to hardware constraints
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param a_tensor: Input tensor A
        :type a_tensor: cute.Tensor
        :param b_tensor: Input tensor B
        :type b_tensor: cute.Tensor
        :param sfa_tensor: Scale factor tensor A
        :type sfa_tensor: cute.Tensor
        :param sfb_tensor: Scale factor tensor B
        :type sfb_tensor: cute.Tensor
        :param c_tensor: Output tensor C
        :type c_tensor: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: cutlass.Constexpr
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        :param epilogue_op: Optional elementwise lambda function to apply to the output tensor
        :type epilogue_op: cutlass.Constexpr
        :raises TypeError: If input data types are incompatible with the MMA instruction.
        """

        # Convert from torch convention to CuTe convention.
        # CUTLASS API passes tensors as A:(L,M,K) B:(L,K,N) C:(L,M,N)
        # but CuTe DSL kernels expect A:(M,K,L) B:(N,K,L) C:(M,N,L).
        def add_batch_mode(tensor: cute.Tensor) -> cute.Tensor:
            return cute.make_tensor(
                tensor.iterator,
                cute.prepend(tensor.layout, cute.make_layout(1), up_to_rank=3),
            )

        a_tensor = add_batch_mode(a_tensor)
        b_tensor = add_batch_mode(b_tensor)
        c_tensor = add_batch_mode(c_tensor)
        if cutlass.const_expr(local_reduce_tensor is not None):
            local_reduce_tensor = add_batch_mode(local_reduce_tensor)

        self.local_reduce_group = local_reduce_group
        self.local_reduce_axis = local_reduce_axis
        self.local_reduce_type = local_reduce_type
        self.local_reduce_source = local_reduce_source
        self.local_reduce_feeds_main = local_reduce_feeds_main
        self.has_cross_warp_local_reduce = cutlass.const_expr(
            local_reduce_tensor is not None
            and local_reduce_axis == 0
            and local_reduce_group > 4
        )

        a_tensor = cute.make_tensor(
            a_tensor.iterator, cute.select(a_tensor.layout, [1, 2, 0])
        )
        b_tensor = cute.make_tensor(
            b_tensor.iterator, cute.select(b_tensor.layout, [2, 1, 0])
        )
        c_tensor = cute.make_tensor(
            c_tensor.iterator, cute.select(c_tensor.layout, [1, 2, 0])
        )

        def epilogue_tensor_to_mnl(tensor: cute.Tensor) -> cute.Tensor:
            tensor = add_batch_mode(tensor)
            return cute.make_tensor(
                tensor.iterator, cute.select(tensor.layout, [1, 2, 0])
            )

        if cutlass.const_expr(epilogue_tensor0 is not None):
            epilogue_tensor0 = epilogue_tensor_to_mnl(epilogue_tensor0)
        if cutlass.const_expr(epilogue_tensor1 is not None):
            epilogue_tensor1 = epilogue_tensor_to_mnl(epilogue_tensor1)
        if cutlass.const_expr(epilogue_tensor2 is not None):
            epilogue_tensor2 = epilogue_tensor_to_mnl(epilogue_tensor2)
        if cutlass.const_expr(epilogue_tensor3 is not None):
            epilogue_tensor3 = epilogue_tensor_to_mnl(epilogue_tensor3)
        if cutlass.const_expr(epilogue_output0 is not None):
            epilogue_output0 = epilogue_tensor_to_mnl(epilogue_output0)
        if cutlass.const_expr(epilogue_output1 is not None):
            epilogue_output1 = epilogue_tensor_to_mnl(epilogue_output1)
        if cutlass.const_expr(epilogue_output2 is not None):
            epilogue_output2 = epilogue_tensor_to_mnl(epilogue_output2)

        self.epilogue_output_count = epilogue_output_count
        self.primary_epilogue_output = primary_epilogue_output

        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a_tensor.element_type
        self.b_dtype: Type[cutlass.Numeric] = b_tensor.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa_tensor.element_type
        self.c_dtype: Type[cutlass.Numeric] = c_tensor.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a_tensor).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b_tensor).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c_tensor)

        # Check if input data types are compatible with MMA instruction
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
        # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a_tensor.shape, self.sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa_tensor.iterator, sfa_layout)

        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_tensor.shape, self.sf_vec_size
        )
        sfb_tensor = cute.make_tensor(sfb_tensor.iterator, sfb_layout)

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a_tensor,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for SFA
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa_tensor,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # Setup TMA load for SFB
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
            x = tma_tensor_sfb.stride[0][1]
            y = cute.ceil_div(tma_tensor_sfb.shape[0][1], 4)

            new_shape = (
                (tma_tensor_sfb.shape[0][0], ((2, 2), y)),
                tma_tensor_sfb.shape[1],
                tma_tensor_sfb.shape[2],
            )
            # Use right multiplication for ScaledBasis (3 * x instead of x * 3)
            x_times_3 = 3 * x
            new_stride = (
                (tma_tensor_sfb.stride[0][0], ((x, x), x_times_3)),
                tma_tensor_sfb.stride[1],
                tma_tensor_sfb.stride[2],
            )
            tma_tensor_sfb_new_layout = cute.make_layout(new_shape, stride=new_stride)
            tma_tensor_sfb = cute.make_tensor(
                tma_tensor_sfb.iterator, tma_tensor_sfb_new_layout
            )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        # Setup TMA store for C
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c_tensor,
            epi_smem_layout,
            self.epi_tile,
        )

        # Compute grid size
        self.tile_sched_params, grid = self._compute_grid(
            c_tensor,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
        )

        self.buffer_align_bytes = 1024

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sLocalReduce: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32,
                    self.cta_tile_shape_mnk[1] * 3
                    if self.has_cross_warp_local_reduce
                    else 1,
                ],
                16,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
            epilogue_input_dtype,
            alpha_tensor,
            epilogue_tensor0,
            epilogue_tensor1,
            epilogue_tensor2,
            epilogue_tensor3,
            epilogue_tensor_kind0,
            epilogue_tensor_kind1,
            epilogue_tensor_kind2,
            epilogue_tensor_kind3,
            epilogue_output0,
            epilogue_output1,
            epilogue_output2,
            local_reduce_tensor,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )
        return

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        epilogue_input_dtype: cutlass.Constexpr,
        alpha_tensor: cute.Tensor,
        epilogue_tensor0: cute.Tensor,
        epilogue_tensor1: cute.Tensor,
        epilogue_tensor2: cute.Tensor,
        epilogue_tensor3: cute.Tensor,
        epilogue_tensor_kind0: cutlass.Constexpr,
        epilogue_tensor_kind1: cutlass.Constexpr,
        epilogue_tensor_kind2: cutlass.Constexpr,
        epilogue_tensor_kind3: cutlass.Constexpr,
        epilogue_output0: cute.Tensor,
        epilogue_output1: cute.Tensor,
        epilogue_output2: cute.Tensor,
        local_reduce_tensor: cute.Tensor,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sLocalReduce = storage.sLocalReduce.get_tensor(
            cute.make_layout((self.cta_tile_shape_mnk[1], 3))
            if self.has_cross_warp_local_reduce
            else cute.make_layout(1)
        )

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (
            2 if use_2cta_instrs else 1
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        #
        # Setup smem tensor A/B/SFA/SFB/C
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = storage.sC.get_tensor(
            c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        # (MMA, MMA_N, MMA_K, STAGE)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        #
        # Compute multicast mask for A/B/SFA/SFB buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bK, RestM, RestK, RestL)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgC = thr_mma.partition_C(gC_mnl)

        #
        # Partition global/shared tensor for TMA load A/B
        #
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        #  TMA load SFA partition_S/D
        sfa_cta_layout = a_cta_layout
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            sfa_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        # TMA load SFB partition_S/D
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        if cutlass.const_expr(self.overlapping_accum):
            num_acc_stage_overlapped = 2
            tCtAcc_fake = tiled_mma.make_fragment_C(
                cute.append(acc_shape, num_acc_stage_overlapped)
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_fake = cute.make_tensor(
                tCtAcc_fake.iterator,
                cute.make_layout(
                    tCtAcc_fake.shape,
                    stride=(
                        tCtAcc_fake.stride[0],
                        tCtAcc_fake.stride[1],
                        tCtAcc_fake.stride[2],
                        (256 - self.num_sf_tmem_cols) * tCtAcc_fake.stride[0][1],
                    ),
                ),
            )
        else:
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_fake = tiled_mma.make_fragment_C(
                cute.append(acc_shape, self.num_acc_stage)
            )

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        #
        # Specialized TMA load warp
        #
        if warp_idx == self.tma_warp_id:
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), RestK)
                tAgA_slice = tAgA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                # ((atom_v, rest_v), RestK)
                tBgB_slice = tBgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                # ((atom_v, rest_v), RestK)
                tAgSFA_slice = tAgSFA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]

                slice_n = mma_tile_coord_mnl[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_mnl[1] // 2
                # ((atom_v, rest_v), RestK)
                tBgSFB_slice = tBgSFB[(None, slice_n, None, mma_tile_coord_mnl[2])]

                # TMA-prefetch tactic: issue prefetches for the first
                # prefetch_dist K-tiles to hide latency.
                # Helps small-M large-K where the mainloop is load-bound.
                if cutlass.const_expr(self.use_prefetch):
                    for pf_k_tile in cutlass.range(
                        0, min(self.prefetch_dist, k_tile_cnt), unroll=1
                    ):
                        cute.prefetch(tma_atom_a, tAgA_slice[(None, pf_k_tile)])
                        cute.prefetch(tma_atom_b, tBgB_slice[(None, pf_k_tile)])
                        cute.prefetch(tma_atom_sfa, tAgSFA_slice[(None, pf_k_tile)])
                        cute.prefetch(tma_atom_sfb, tBgSFB_slice[(None, pf_k_tile)])

                # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                        ab_producer_state
                    )
                #
                # Tma load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Conditionally wait for AB buffer empty
                    ab_pipeline.producer_acquire(
                        ab_producer_state, peek_ab_empty_status
                    )

                    # TMA load A/B/SFA/SFB
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, ab_producer_state.count)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, ab_producer_state.count)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=b_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_slice[(None, ab_producer_state.count)],
                        tAsSFA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfa_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_slice[(None, ab_producer_state.count)],
                        tBsSFB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfb_full_mcast_mask,
                    )

                    # Rolling prefetch: stay prefetch_dist tiles ahead each
                    # iteration to hide TMA latency
                    # through the whole K-loop. This is the piece that helps
                    # small-M large-K (the initial pre-loop burst alone does not).
                    if cutlass.const_expr(self.use_prefetch):
                        if k_tile < k_tile_cnt - self.prefetch_dist:
                            future_k_tile = ab_producer_state.count + self.prefetch_dist
                            cute.prefetch(tma_atom_a, tAgA_slice[(None, future_k_tile)])
                            cute.prefetch(tma_atom_b, tBgB_slice[(None, future_k_tile)])
                            cute.prefetch(
                                tma_atom_sfa, tAgSFA_slice[(None, future_k_tile)]
                            )
                            cute.prefetch(
                                tma_atom_sfb, tBgSFB_slice[(None, future_k_tile)]
                            )

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state
                        )

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Wait A/B buffer empty
            #
            ab_pipeline.producer_tail(ab_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator/SFA/SFB tensor
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # Make accumulator tmem tensor
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # Make SFA tmem tensor
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_M, MMA_K)
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            # Make SFB tmem tensor
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_N, MMA_K)
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
            #
            # Partition for S2T copy of SFA/SFB
            #
            (
                tiled_copy_s2t_sfa,
                tCsSFA_compact_s2t,
                tCtSFA_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            (
                tiled_copy_s2t_sfb,
                tCsSFB_compact_s2t,
                tCtSFB_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # Get accumulator stage index
                reverse_subtile = cutlass.Boolean(False)
                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_producer_state.phase ^ 1
                else:
                    acc_stage_index = acc_producer_state.index

                # Set tensor memory buffer for current tile
                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                        ab_consumer_state
                    )

                #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
                    # If this is an ODD tile, shift the TMEM start address for cta_tile_shape_n=192 case by two words (ignores first 64 columns of SFB)
                    offset = (
                        cutlass.Int32(2)
                        if mma_tile_coord_mnl[1] % 2 == 1
                        else cutlass.Int32(0)
                    )
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                elif cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    # Move in increments of 64 columns of SFB
                    offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)

                #
                # Reset the ACCUMULATE field for each tile
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                #
                # Mma mainloop
                #
                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        ab_pipeline.consumer_wait(
                            ab_consumer_state, peek_ab_full_status
                        )

                        #  Copy SFA/SFB from smem to tmem
                        s2t_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            ab_consumer_state.index,
                        )
                        tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                        tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t_staged,
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t_staged,
                            tCtSFB_compact_s2t,
                        )

                        # tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                ab_consumer_state.index,
                            )

                            # Set SFA/SFB tensor to tiled_mma
                            sf_kblock_coord = (None, None, kblock_idx)
                            tiled_mma.set(
                                tcgen05.Field.SFA,
                                tCtSFA[sf_kblock_coord].iterator,
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB,
                                tCtSFB_mma[sf_kblock_coord].iterator,
                            )

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )

                            # Enable accumulate on tCtAcc after first kblock
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Async arrive AB buffer empty
                        ab_pipeline.consumer_release(ab_consumer_state)

                    # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                ab_consumer_state
                            )

                #
                # Async arrive accumulator buffer full
                #
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)
        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            #
            # Alloc tensor memory buffer
            #
            tmem.allocate(self.num_tmem_alloc_cols)

            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            #
            # Partition for epilogue
            #
            epi_tidx = tidx
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs
            )

            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            (
                tma_atom_c,
                bSG_sC,
                bSG_gC_partitioned,
            ) = self.epilog_gmem_copy_and_partition(
                epi_tidx, tma_atom_c, tCgC, epi_tile, sC
            )

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            # Threads/warps participating in tma store pipeline
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                #
                # Slice to per mma tile index
                #
                # ((ATOM_V, REST_V), EPI_M, EPI_N)
                bSG_gC = bSG_gC_partitioned[
                    (
                        None,
                        None,
                        None,
                        *mma_tile_coord_mnl,
                    )
                ]

                # Get accumulator stage index
                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_consumer_state.phase
                    reverse_subtile = (
                        cutlass.Boolean(True)
                        if acc_stage_index == 0
                        else cutlass.Boolean(False)
                    )
                else:
                    acc_stage_index = acc_consumer_state.index

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_stage_index)
                ]

                #
                # Wait for accumulator buffer full
                #
                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                #
                # Store accumulator to global memory in subtiles
                #
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
                local_reduce_partial = None
                if cutlass.const_expr(
                    (local_reduce_tensor is not None or self.local_reduce_feeds_main)
                    and self.local_reduce_axis == 1
                    and self.local_reduce_group > self.epi_tile_n
                ):
                    local_reduce_partial = cute.make_rmem_tensor(
                        ((1, self.epi_tile_n, 1), 1, 1), self.acc_dtype
                    )
                for subtile_idx in cutlass.range(subtile_cnt):
                    real_subtile_idx = subtile_idx
                    if cutlass.const_expr(self.overlapping_accum):
                        if reverse_subtile:
                            real_subtile_idx = (
                                self.cta_tile_shape_mnk[1] // self.epi_tile_n
                                - 1
                                - subtile_idx
                            )
                    #
                    # Load accumulator from tensor memory buffer to register
                    #
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, real_subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    #
                    # Async arrive accumulator buffer empty earlier when overlapping_accum is enabled
                    #
                    if cutlass.const_expr(self.overlapping_accum):
                        if subtile_idx == self.iter_acc_early_release_in_epilogue:
                            # Fence for TMEM load
                            cute.arch.fence_view_async_tmem_load()
                            with cute.arch.elect_one():
                                acc_pipeline.consumer_release(acc_consumer_state)
                            acc_consumer_state.advance()

                    #
                    # Convert to C type
                    #
                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                    # Fused global scale: multiply by runtime alpha[0] (a traced
                    # kernel-arg scalar) when provided. Closure capture cannot
                    # read a runtime tensor here, so alpha must be a kernel arg.
                    # Apply in fp32 (acc_dtype) BEFORE the downcast to c_dtype so
                    # low-range outputs (fp8/fp16) don't overflow/saturate on the
                    # cast before alpha (typically < 1) restores range, and before
                    # epilogue_op so the epilogue sees the true scaled value.
                    # const_expr makes this a compile-time branch (skipped when
                    # alpha_tensor is None) rather than device control flow.
                    if cutlass.const_expr(alpha_tensor is not None):
                        acc_vec = acc_vec * alpha_tensor[0].to(self.acc_dtype)
                    has_epilogue_tensors = cutlass.const_expr(
                        epilogue_tensor0 is not None
                        or epilogue_tensor1 is not None
                        or epilogue_tensor2 is not None
                        or epilogue_tensor3 is not None
                    )
                    has_epilogue_outputs = cutlass.const_expr(
                        self.epilogue_output_count > 1
                    )
                    if cutlass.const_expr(
                        has_epilogue_tensors
                        or has_epilogue_outputs
                        or local_reduce_tensor is not None
                    ):
                        tDcC = partition_for_epilogue(
                            cute.make_identity_tensor(self.cta_tile_shape_mnk[:2]),
                            epi_tile=epi_tile,
                            tiled_copy=tiled_copy_t2r,
                            tidx=epi_tidx,
                            reference_src=False,
                        )
                        tDcC = tDcC[(None, None, None, 0, real_subtile_idx)]
                        coord_flt = cute.filter_zeros(tDcC)

                    epilogue_values = []
                    if cutlass.const_expr(has_epilogue_tensors or has_epilogue_outputs):
                        output_m = cute.size(mC_mnl, mode=[0])
                        output_n = cute.size(mC_mnl, mode=[1])
                    if cutlass.const_expr(has_epilogue_tensors):
                        for tensor, kind in (
                            (epilogue_tensor0, epilogue_tensor_kind0),
                            (epilogue_tensor1, epilogue_tensor_kind1),
                            (epilogue_tensor2, epilogue_tensor_kind2),
                            (epilogue_tensor3, epilogue_tensor_kind3),
                        ):
                            if cutlass.const_expr(tensor is not None):
                                fragment = cute.make_rmem_tensor_like(
                                    acc_vec, tensor.element_type
                                )
                                for i in cutlass.range(
                                    cute.size(fragment), unroll_full=True
                                ):
                                    row_idx = coord_flt[i][0]
                                    col_idx = coord_flt[i][1]
                                    global_m = (
                                        mma_tile_coord_mnl[0]
                                        * self.cta_tile_shape_mnk[0]
                                        + row_idx
                                    )
                                    global_n = (
                                        mma_tile_coord_mnl[1]
                                        * self.cta_tile_shape_mnk[1]
                                        + col_idx
                                    )
                                    tensor_m = (
                                        0 if cutlass.const_expr(kind == 2) else global_m
                                    )
                                    tensor_n = (
                                        0 if cutlass.const_expr(kind == 3) else global_n
                                    )
                                    fragment[i] = tensor[0, 0, 0]
                                    if global_m < output_m and global_n < output_n:
                                        fragment[i] = tensor[
                                            tensor_m,
                                            tensor_n,
                                            mma_tile_coord_mnl[2],
                                        ]
                                epilogue_values.append(fragment.load())
                    if cutlass.const_expr(
                        local_reduce_tensor is not None or self.local_reduce_feeds_main
                    ):
                        local_reduce_vec = acc_vec.to(epilogue_input_dtype).to(
                            self.acc_dtype
                        )
                        if cutlass.const_expr(self.local_reduce_source == "square"):
                            local_reduce_vec = local_reduce_vec * local_reduce_vec
                        else:
                            assert self.local_reduce_source == "identity"
                    if cutlass.const_expr(self.local_reduce_feeds_main):
                        group = cutlass.const_expr(self.local_reduce_group)
                        lane_layout_mn, _ = get_lane_warp_layouts(
                            tiled_copy_t2r, reference_src=False
                        )
                        lanes_in_m = cutlass.const_expr(
                            cute.size(lane_layout_mn, mode=[0])
                        )
                        assert group <= lanes_in_m
                        feed_value = cute.make_rmem_tensor_like(
                            local_reduce_vec, self.acc_dtype
                        )
                        feed_value.store(local_reduce_vec)
                        feed_flt = cute.filter_zeros(feed_value)
                        for i in cutlass.range(cute.size(feed_flt), unroll_full=True):
                            rows = group // 2
                            while rows > 0:
                                feed_flt[i] += cute.arch.shuffle_sync_bfly(
                                    feed_flt[i],
                                    offset=cute.crd2idx((rows, 0), lane_layout_mn),
                                )
                                rows //= 2
                            if cutlass.const_expr(self.local_reduce_type == "mean"):
                                feed_flt[i] /= group
                        epilogue_result = (
                            acc_vec.to(epilogue_input_dtype) - feed_value.load()
                        )
                    else:
                        epilogue_result = epilogue_op(
                            acc_vec.to(epilogue_input_dtype), *tuple(epilogue_values)
                        )
                    if cutlass.const_expr(has_epilogue_outputs):
                        for output_index, tensor in enumerate(
                            (epilogue_output0, epilogue_output1, epilogue_output2)
                        ):
                            if cutlass.const_expr(tensor is not None):
                                result_index = cutlass.const_expr(
                                    output_index
                                    if output_index < self.primary_epilogue_output
                                    else output_index + 1
                                )
                                output_vec = epilogue_result[result_index].to(
                                    tensor.element_type
                                )
                                fragment = cute.make_rmem_tensor_like(
                                    output_vec, tensor.element_type
                                )
                                fragment.store(output_vec)
                                fragment_flt = cute.filter_zeros(fragment)
                                for i in cutlass.range(
                                    cute.size(fragment_flt), unroll_full=True
                                ):
                                    row_idx = coord_flt[i][0]
                                    col_idx = coord_flt[i][1]
                                    global_m = (
                                        mma_tile_coord_mnl[0]
                                        * self.cta_tile_shape_mnk[0]
                                        + row_idx
                                    )
                                    global_n = (
                                        mma_tile_coord_mnl[1]
                                        * self.cta_tile_shape_mnk[1]
                                        + col_idx
                                    )
                                    if global_m < output_m and global_n < output_n:
                                        tensor[
                                            global_m,
                                            global_n,
                                            mma_tile_coord_mnl[2],
                                        ] = fragment_flt[i]
                        acc_vec = epilogue_result[self.primary_epilogue_output].to(
                            self.c_dtype
                        )
                    else:
                        acc_vec = epilogue_result.to(self.c_dtype)

                    if cutlass.const_expr(
                        local_reduce_tensor is not None
                        and not self.local_reduce_feeds_main
                    ):
                        group = cutlass.const_expr(self.local_reduce_group)
                        mReduce = local_reduce_tensor[mma_tile_coord_mnl[2], None, None]
                        if cutlass.const_expr(self.local_reduce_axis == 1):
                            fragment_n = cutlass.const_expr(
                                cute.size(acc_vec.shape, mode=[0])
                            )
                            assert group > 1 and group <= self.cta_tile_shape_mnk[1]
                            fragment_group = cutlass.const_expr(min(group, fragment_n))
                            assert group % fragment_group == 0
                            assert fragment_n % fragment_group == 0
                            repeats = cutlass.const_expr(fragment_n // fragment_group)
                            grouped = local_reduce_vec.reshape(
                                ((1, fragment_group, repeats), 1, 1)
                            )
                            if cutlass.const_expr(
                                self.local_reduce_type in ("sum", "mean")
                            ):
                                reduce_op = cute.ReductionOp.ADD
                                init_val = 0.0
                            elif cutlass.const_expr(self.local_reduce_type == "prod"):
                                reduce_op = cute.ReductionOp.MUL
                                init_val = 1.0
                            elif cutlass.const_expr(self.local_reduce_type == "max"):
                                reduce_op = cute.ReductionOp.MAX
                                init_val = -cutlass.Float32.inf
                            else:
                                assert self.local_reduce_type == "min"
                                reduce_op = cute.ReductionOp.MIN
                                init_val = cutlass.Float32.inf
                            reduced = grouped.reduce(
                                reduce_op,
                                init_val=init_val,
                                reduction_profile=((None, 1, None), 1, 1),
                            )
                            if cutlass.const_expr(
                                self.local_reduce_type == "mean" and group <= fragment_n
                            ):
                                reduced = reduced / group
                            reduced = reduced.reshape(((1, 1, repeats), 1, 1))
                            reduced = reduced.broadcast_to(grouped.shape)
                            should_store = cutlass.Boolean(True)
                            store_offset = cutlass.Int32(0)
                            if cutlass.const_expr(group > fragment_n):
                                fragments_per_group = cutlass.const_expr(
                                    group // fragment_n
                                )
                                partial = local_reduce_partial.load()
                                if subtile_idx % fragments_per_group == 0:
                                    partial = reduced
                                elif cutlass.const_expr(
                                    self.local_reduce_type in ("sum", "mean")
                                ):
                                    partial = partial + reduced
                                elif cutlass.const_expr(
                                    self.local_reduce_type == "prod"
                                ):
                                    partial = partial * reduced
                                elif cutlass.const_expr(
                                    self.local_reduce_type == "max"
                                ):
                                    partial = cute.math.max(partial, reduced)
                                else:
                                    assert self.local_reduce_type == "min"
                                    partial = cute.math.min(partial, reduced)
                                local_reduce_partial.store(partial)
                                should_store = (
                                    subtile_idx % fragments_per_group
                                    == fragments_per_group - 1
                                )
                                if should_store:
                                    reduced = partial
                                    if cutlass.const_expr(
                                        self.local_reduce_type == "mean"
                                    ):
                                        reduced = reduced / group
                                    store_offset = (
                                        real_subtile_idx * self.epi_tile_n
                                    ) % group
                            tDrReduce = cute.make_rmem_tensor(
                                reduced.shape, self.acc_dtype
                            )
                            tDrReduce.store(reduced)
                            reduced_flt = cute.filter_zeros(tDrReduce)
                            groups_per_cta = cutlass.const_expr(
                                self.cta_tile_shape_mnk[1] // group
                            )
                            gReduce = cute.local_tile(
                                mReduce,
                                (self.cta_tile_shape_mnk[0], groups_per_cta),
                                mma_tile_coord_mnl[:2],
                            )
                            limit_m = min(
                                cute.size(local_reduce_tensor, mode=[1])
                                - mma_tile_coord_mnl[0] * self.cta_tile_shape_mnk[0],
                                self.cta_tile_shape_mnk[0],
                            )
                            limit_groups = cute.size(local_reduce_tensor, mode=[2])
                            for i in cutlass.range(
                                cute.size(reduced_flt), unroll_full=True
                            ):
                                row_idx = coord_flt[i][0]
                                n_idx = coord_flt[i][1]
                                group_idx = n_idx // group
                                global_group_idx = (
                                    mma_tile_coord_mnl[1] * groups_per_cta + group_idx
                                )
                                if (
                                    should_store
                                    and n_idx % group == store_offset
                                    and row_idx < limit_m
                                    and global_group_idx < limit_groups
                                ):
                                    gReduce[row_idx, group_idx] = reduced_flt[i]
                        else:
                            tDrReduce = cute.make_rmem_tensor_like(
                                local_reduce_vec, self.acc_dtype
                            )
                            tDrReduce.store(local_reduce_vec)
                            reduced_flt = cute.filter_zeros(tDrReduce)
                            lane_layout_mn, warp_layout_mn = get_lane_warp_layouts(
                                tiled_copy_t2r, reference_src=False
                            )
                            lanes_in_m = cutlass.const_expr(
                                cute.size(lane_layout_mn, mode=[0])
                            )
                            assert group > 1 and group <= self.cta_tile_shape_mnk[0]
                            reduction_group = cutlass.const_expr(min(group, lanes_in_m))
                            assert group % reduction_group == 0
                            assert lanes_in_m % reduction_group == 0
                            for i in cutlass.range(
                                cute.size(reduced_flt), unroll_full=True
                            ):
                                rows = reduction_group // 2
                                while rows > 0:
                                    other = cute.arch.shuffle_sync_bfly(
                                        reduced_flt[i],
                                        offset=cute.crd2idx((rows, 0), lane_layout_mn),
                                    )
                                    if cutlass.const_expr(
                                        self.local_reduce_type in ("sum", "mean")
                                    ):
                                        reduced_flt[i] += other
                                    elif cutlass.const_expr(
                                        self.local_reduce_type == "prod"
                                    ):
                                        reduced_flt[i] *= other
                                    elif cutlass.const_expr(
                                        self.local_reduce_type == "max"
                                    ):
                                        reduced_flt[i] = cute.arch.fmax(
                                            reduced_flt[i], other
                                        )
                                    else:
                                        assert self.local_reduce_type == "min"
                                        reduced_flt[i] = cute.arch.fmin(
                                            reduced_flt[i], other
                                        )
                                    rows = rows // 2
                                if cutlass.const_expr(self.local_reduce_type == "mean"):
                                    reduced_flt[i] /= group
                            groups_per_cta = cutlass.const_expr(
                                self.cta_tile_shape_mnk[0] // group
                            )
                            gReduce = cute.local_tile(
                                mReduce,
                                (groups_per_cta, self.cta_tile_shape_mnk[1]),
                                mma_tile_coord_mnl[:2],
                            )
                            limit_n = min(
                                cute.size(local_reduce_tensor, mode=[2])
                                - mma_tile_coord_mnl[1] * self.cta_tile_shape_mnk[1],
                                self.cta_tile_shape_mnk[1],
                            )
                            limit_groups = cute.size(local_reduce_tensor, mode=[1])
                            group_warps = cutlass.const_expr(group // lanes_in_m)
                            warp_m = warp_layout_mn[0]
                            warps_in_m = cutlass.const_expr(cute.size(warp_m))
                            assert group_warps <= warps_in_m
                            warp_idx = cute.arch.make_warp_uniform(
                                epi_tidx // cute.arch.WARP_SIZE
                            )
                            warp_m_idx = warp_layout_mn.get_hier_coord(warp_idx)[0]
                            if cutlass.const_expr(group > lanes_in_m):
                                for i in cutlass.range(
                                    cute.size(reduced_flt), unroll_full=True
                                ):
                                    row_idx = coord_flt[i][0]
                                    n_idx = coord_flt[i][1]
                                    group_idx = row_idx // group
                                    group_warp_start = group_idx * group_warps
                                    group_warp_idx = warp_m_idx - group_warp_start
                                    if (
                                        row_idx % lanes_in_m == 0
                                        and group_warp_idx > 0
                                        and group_warp_idx < group_warps
                                    ):
                                        sLocalReduce[
                                            n_idx, warp_m_idx - group_idx - 1
                                        ] = reduced_flt[i]
                                self.epilog_sync_barrier.arrive_and_wait()
                            for i in cutlass.range(
                                cute.size(reduced_flt), unroll_full=True
                            ):
                                row_idx = coord_flt[i][0]
                                n_idx = coord_flt[i][1]
                                group_idx = row_idx // group
                                global_group_idx = (
                                    mma_tile_coord_mnl[0] * groups_per_cta + group_idx
                                )
                                group_value = reduced_flt[i]
                                group_warp_start = group_idx * group_warps
                                if cutlass.const_expr(group > lanes_in_m):
                                    if (
                                        row_idx % group == 0
                                        and warp_m_idx == group_warp_start
                                    ):
                                        for warp_offset in cutlass.range_constexpr(
                                            1, group_warps
                                        ):
                                            other = sLocalReduce[
                                                n_idx,
                                                group_warp_start
                                                + warp_offset
                                                - group_idx
                                                - 1,
                                            ]
                                            if cutlass.const_expr(
                                                self.local_reduce_type
                                                in ("sum", "mean")
                                            ):
                                                group_value += other
                                            elif cutlass.const_expr(
                                                self.local_reduce_type == "prod"
                                            ):
                                                group_value *= other
                                            elif cutlass.const_expr(
                                                self.local_reduce_type == "max"
                                            ):
                                                group_value = cute.arch.fmax(
                                                    group_value, other
                                                )
                                            else:
                                                assert self.local_reduce_type == "min"
                                                group_value = cute.arch.fmin(
                                                    group_value, other
                                                )
                                if (
                                    row_idx % group == 0
                                    and n_idx < limit_n
                                    and global_group_idx < limit_groups
                                ):
                                    gReduce[group_idx, n_idx] = group_value
                            if cutlass.const_expr(group > lanes_in_m):
                                self.epilog_sync_barrier.arrive_and_wait()
                    tRS_rC.store(acc_vec)

                    #
                    # Store C to shared memory
                    #
                    c_buffer = (num_prev_subtiles + real_subtile_idx) % self.num_c_stage
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rC,
                        tRS_sC[(None, None, None, c_buffer)],
                    )
                    # Fence and barrier to make sure shared memory store is visible to TMA store
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    self.epilog_sync_barrier.arrive_and_wait()

                    #
                    # TMA store C to global memory
                    #
                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(
                            tma_atom_c,
                            bSG_sC[(None, c_buffer)],
                            bSG_gC[(None, real_subtile_idx)],
                        )
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    self.epilog_sync_barrier.arrive_and_wait()

                #
                # Async arrive accumulator buffer empty
                #
                if cutlass.const_expr(not self.overlapping_accum):
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(acc_tmem_ptr)
            #
            # Wait for C store complete
            #
            c_pipeline.producer_tail()

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for smem to tmem load for scale factor tensor, then use it to partition smem memory (source) and tensor memory (destination).

        :param sSF: The scale factor tensor in smem
        :type sSF: cute.Tensor
        :param tSF: The scale factor tensor in tmem
        :type tSF: cute.Tensor

        :return: A tuple containing (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t) where:
            - tiled_copy_s2t: The tiled copy operation for smem to tmem load for scale factor tensor(s2t)
            - tCsSF_compact_s2t: The partitioned scale factor tensor in smem
            - tSF_compact_s2t: The partitioned scale factor tensor in tmem
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)

        # Make S2T CopyAtom and tiledCopy
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory (source) and register array (destination).

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param tAcc: The accumulator tensor to be copied and partitioned
        :type tAcc: cute.Tensor
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param use_2cta_instrs: Whether use_2cta_instrs is enabled
        :type use_2cta_instrs: bool

        :return: A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc) where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tTR_tAcc: The partitioned accumulator tensor
            - tTR_rAcc: The accumulated tensor in register used to hold t2r results
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # Make tiledCopy for tensor memory load
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition register array (source) and shared memory (destination).

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rC: The partitioned accumulator tensor
        :type tTR_rC: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor
        :type sepi: cute.Tensor

        :return: A tuple containing (tiled_copy_r2s, tRS_rC, tRS_sC) where:
            - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
            - tRS_rC: The partitioned tensor C (register source)
            - tRS_sC: The partitioned tensor C (smem destination)
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        # (R2S, R2S_M, R2S_N)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
    ) -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """Make tiledCopy for global memory store, then use it to:
        partition shared memory (source) and global memory (destination) for TMA store version.

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param atom: The copy_atom_c to be used for TMA store version, or tiled_copy_t2r for none TMA store version
        :type atom: cute.CopyAtom or cute.TiledCopy
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor

        :return: A tuple containing (tma_atom_c, bSG_sC, bSG_gC) where:
            - tma_atom_c: The TMA copy atom
            - bSG_sC: The partitioned shared memory tensor C
            - bSG_gC: The partitioned global tensor C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]
        """
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )

        tma_atom_c = atom
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, RestM, RestN, RestL)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        return tma_atom_c, bSG_sC, bSG_gC

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        smem_capacity: int,
        occupancy: int,
    ) -> Tuple[int, int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: The shape (M, N, K) of the MMA tiler.
        :type mma_tiler_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param epi_tile: The epilogue tile shape.
        :type epi_tile: cute.Tile
        :param c_dtype: Data type of operand C (output).
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: Layout enum of operand C.
        :type c_layout: utils.LayoutEnum
        :param sf_dtype: Data type of Scale factor.
        :type sf_dtype: type[cutlass.Numeric]
        :param sf_vec_size: Scale factor vector size.
        :type sf_vec_size: int
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (ACC stages, A/B operand stages, C stages)
        :rtype: tuple[int, int, int]
        """
        # ACC stages
        num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2

        # Default C stages
        num_c_stage = 2

        # Calculate smem layout and size for one stage of A, B, SFA, SFB and C
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  # a tmp 1 stage is provided
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  # a tmp 1 stage is provided
        )
        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )
        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )

        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage

        # Calculate A/B/SFA/SFB stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial C stages bytes
        # Divide remaining by bytes needed per A/B/SFA/SFB stage
        num_ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B/SFA/SFB stages and reserved bytes
        # Add remaining unused smem to epilogue
        num_c_stage += (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)

        return num_acc_stage, num_ab_stage, num_c_stage

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Use persistent tile scheduler to compute the grid size for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr

        :return: A tuple containing:
            - tile_sched_params: Parameters for the persistent tile scheduler.
            - grid: Grid shape for kernel launch.
        :rtype: Tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]
        """
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @staticmethod
    def is_valid_dtypes_and_scale_factor_vec_size(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """
        Check if the dtypes and sf_vec_size are valid combinations

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size of the scale factor
        :type sf_vec_size: int
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]

        :return: True if the dtypes and sf_vec_size are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        # Check valid ab_dtype
        if ab_dtype not in {
            cutlass.Float4E2M1FN,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            is_valid = False

        # Check valid sf_vec_size
        if sf_vec_size not in {16, 32}:
            is_valid = False

        # Check valid sf_dtype
        if sf_dtype not in {cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN}:
            is_valid = False

        # Check valid sf_dtype and sf_vec_size combinations
        if sf_dtype == cutlass.Float8E4M3FN and sf_vec_size == 32:
            is_valid = False
        if ab_dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN} and sf_vec_size == 16:
            is_valid = False

        # Check valid c_dtype
        if c_dtype not in {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            is_valid = False

        return is_valid

    @staticmethod
    def is_valid_layouts(
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if layouts and dtypes are valid combinations

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major dimension of the A tensor
        :type a_major: str
        :param b_major: The major dimension of the B tensor
        :type b_major: str
        :param c_major: The major dimension of the C tensor
        :type c_major: str

        :return: True if the layouts are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        if ab_dtype is cutlass.Float4E2M1FN and not (a_major == "k" and b_major == "k"):
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
    ) -> bool:
        """
        Check if the mma tiler and cluster shape are valid

        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]

        :return: True if the mma tiler and cluster shape are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        # Skip invalid mma tile shape
        if mma_tiler_mn[0] not in [128, 256]:
            is_valid = False
        if mma_tiler_mn[1] not in [64, 128, 192, 256]:
            is_valid = False
        # Skip illegal cluster shape
        if cluster_shape_mn[0] % (2 if mma_tiler_mn[0] == 256 else 1) != 0:
            is_valid = False
        # Skip invalid cluster shape
        is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            or cluster_shape_mn[0] <= 0
            or cluster_shape_mn[1] <= 0
            # Special cluster shape check for scale factor multicasts.
            # Due to limited size of scale factors, we can't multicast among more than 4 CTAs.
            or cluster_shape_mn[0] > 4
            or cluster_shape_mn[1] > 4
            or not is_power_of_2(cluster_shape_mn[0])
            or not is_power_of_2(cluster_shape_mn[1])
        ):
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_tensor_alignment(
        m: int,
        n: int,
        k: int,
        l: int,
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the tensor alignment is valid

        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int
        :param k: The number of columns in the A tensor
        :type k: int
        :param l: The number of columns in the C tensor
        :type l: int
        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the problem shape is valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        if (
            not check_contigous_16B_alignment(ab_dtype, a_major == "m", (m, k, l))
            or not check_contigous_16B_alignment(ab_dtype, b_major == "n", (n, k, l))
            or not check_contigous_16B_alignment(c_dtype, c_major == "m", (m, n, l))
        ):
            is_valid = False
        return is_valid

    @staticmethod
    def can_implement(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        k: int,
        l: int,
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the gemm can be implemented

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor tensor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size
        :type sf_vec_size: int
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]
        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int
        :param k: The number of columns in the A tensor
        :type k: int
        :param l: The number of columns in the C tensor
        :type l: int
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the gemm can be implemented, False otherwise
        :rtype: bool
        """
        can_implement = True
        # Skip unsupported types
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_dtypes_and_scale_factor_vec_size(
            ab_dtype, sf_dtype, sf_vec_size, c_dtype
        ):
            can_implement = False
        # Skip unsupported layouts
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_layouts(
            ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False
        # Skip invalid mma tile shape and cluster shape
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_mma_tiler_and_cluster_shape(
            mma_tiler_mn, cluster_shape_mn
        ):
            can_implement = False
        # Skip illegal problem shape for load/store alignment
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_tensor_alignment(
            m, n, k, l, ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False
        return can_implement


@cute.jit
def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
    sf_ref_tensor: cute.Tensor,
    sf_mma_tensor: cute.Tensor,
):
    """Convert scale factor tensor from MKL layout to mma specification M(32x4xrest_m)xK(4xrest_k)xL layout"""
    # sf_mma_tensor has flatten shape (32, 4, rest_m, 4, rest_k, l)
    # group to ((32, 4, rest_m), (4, rest_k), l)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
    for i in cutlass.range(cute.size(sf_ref_tensor)):
        mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
        sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]
<|endoftext|># Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import functools
from typing import List, Type, Union
from inspect import isclass

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.torch as cutlass_torch

"""
A grouped GEMM example for the NVIDIA Blackwell SM100 architecture using CUTE DSL

This example demonstrates an implementation of grouped GEMM using a TMA plus Blackwell SM100 TensorCore
warp-specialized persistent kernel.
The grouped GEMM workload computes a batch of GEMM operations with distinct problem sizes. Pointers to matrices
in global memory are passed to the kernel in an array (also held in global memory). Similarly, problem shapes and
strides are also stored in arrays in GMEM.

This differs from "Batched Array" GEMM since the size of each GEMM problem in the grouped GEMM concept may be distinct.

To run this example:

.. code-block:: bash

    python examples/blackwell/grouped_gemm.py                                                 \
      --ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32                                \
      --mma_tiler_mn 128,64 --cluster_shape_mn 1,1                                            \
      --problem_sizes_mnkl "(8192,1280,32,1),(16,384,1536,1),(640,1280,16,1),(640,160,16,1)"  \
      --num_groups 4  --tensormap_update_mode SMEM

The above example command makes 4 groups of different m, n, k sizes. The Blackwell tcgen05 MMA tile shape
is specified as (128, 64) and the cluster shape is (1,1). The input, mma accumulator and output data type
are set as fp16, fp32 and fp16, respectively.

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/blackwell/grouped_gemm.py                                             \
      --ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32                                \
      --mma_tiler_mn 128,64 --cluster_shape_mn 1,1                                            \
      --problem_sizes_mnkl "(8192,1280,32,1),(16,384,1536,1),(640,1280,16,1),(640,160,16,1)"  \
      --num_groups 4  --tensormap_update_mode SMEM                                            \
      --warmup_iterations 1 --iterations 10 --skip_ref_check

There are some constrains for this example. Besides the constrains from the Balckwell dense GEMM persistent example,
there are also the following constrains:
* Only fp16 and bf16 data types are supported as inputs.
* Output data types could be fp16, bf16 or fp32.
* The contiguous dimension of each tensor must be at least 16 bytes aligned.
* The l mode(aka, batch size) for each group must be 1.
* The majorness for A, B and C must be the same across all groups.
"""


class GroupedGemmKernel:
    def __init__(
        self,
        acc_dtype: type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
        tensormap_update_mode: utils.TensorMapUpdateMode = utils.TensorMapUpdateMode.SMEM,
    ):
        """Initializes the configuration for a Blackwell grouped GEMM kernel.

        Besides configurations for dense persistent GEMM, there is an extra config specific to grouped GEMM:

        Tensormap Update Mode:
        - tensormap_update_mode: Specifies whether the tensormap is
            updated in global memory(GMEM) or shared memory(SMEM).
           The 2 modes are functionally equivalent and the difference are:
            - We buffer 3 tensormaps in SMEM for A, B, and C tensors (each TMA descriptor takes 128B) when TMA updates performed on SMEM.
            - Performance varies between modes depending on problem size; optimal choice differs across workloads.

        :param acc_dtype: Data type of the accumulator.
        :type acc_dtype: type[cutlass.Numeric]
        :param use_2cta_instrs: Boolean, True to use cta_group=2 MMA variant.
        :type use_2cta_instrs: bool
        :param mma_tiler_mn: tuple (M, N) shape of the MMA instruction.
        :type mma_tiler_mn: tuple[int, int]
        :param cluster_shape_mn: tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mn: tuple[int, int]
        :param tensormap_update_mode: Mode for updating the tensormap (GMEM or SMEM), defaults to SMEM.
        :type tensormap_update_mode: utils.TensorMapUpdateMode, optional
        """
        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.cta_group = (
            tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.tensormap_update_mode = tensormap_update_mode
        # Delegate tensormap ab initialization to MMA warp when SMEM mode is used for better latency hiding
        self.delegate_tensormap_ab_init = (
            tensormap_update_mode == utils.TensorMapUpdateMode.SMEM
        )

        self.num_mcast_ctas_a = 1
        self.num_mcast_ctas_b = 1
        self.is_a_mcast = False
        self.is_b_mcast = False

        self.occupancy = 1
        # Set specialized warp ids
        self.epilog_warp_id = (
            0,
            1,
            2,
            3,
        )
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * len(
            (self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id)
        )
        # Set barrier for epilog sync, tmem ptr sync and tensormap update sync
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=32 * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        # Barrier used by MMA/TMA warps to signal A/B tensormap initialization completion
        self.tensormap_ab_init_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=32 * (len(self.epilog_warp_id) + 1),
        )
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.num_tma_load_bytes = 0

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        Most of the implementation follows standard dense GEMM patterns,
        with the key difference being additional consideration for SMEM
        buffer needed for tensormap updates.
        """
        # Configure tiled mma
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

        # Compute mma/cluster/tile shapes
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_tile_shape_mnk = tuple(
            x * y for x, y in zip(self.cta_tile_shape_mnk, (*self.cluster_shape_mn, 1))
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Compute epilogue subtile
        self.epi_tile = utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )

        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_epi_stage,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.smem_capacity,
            self.occupancy,
        )

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.epi_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_epi_stage,
        )

        mbar_smem_bytes = self._get_mbar_smem_bytes(
            num_acc_stage=self.num_acc_stage,
            num_ab_stage=self.num_ab_stage,
            num_epi_stage=self.num_epi_stage,
        )
        tensormap_smem_bytes = self._get_tensormap_smem_bytes(
            self.tensormap_update_mode
        )
        if (
            mbar_smem_bytes
            + tensormap_smem_bytes
            + GroupedGemmKernel.tensor_memory_management_bytes
            > self.reserved_smem_bytes
        ):
            raise ValueError(
                f"smem consumption for mbar and tensormap {mbar_smem_bytes + tensormap_smem_bytes} exceeds the "
                f"reserved smem bytes {self.reserved_smem_bytes}"
            )

        # Compute the number of tensor memory allocation columns
        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
            tiled_mma, self.mma_tiler, self.num_acc_stage
        )

    @cute.jit
    def __call__(
        self,
        initial_a: cute.Tensor,
        initial_b: cute.Tensor,
        initial_c: cute.Tensor,
        group_count: cutlass.Constexpr[int],
        problem_shape_mnkl: cute.Tensor,
        strides_abc: cute.Tensor,
        tensor_address_abc: cute.Tensor,
        total_num_clusters: cutlass.Constexpr[int],
        tensormap_cute_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr[int],
        stream: cuda.CUstream,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes before smem/grid/tma computation
        - Setup TMA load/store atoms and tensors
        - Compute grid size with regard to hardware constraints
        - Define shared storage for kernel
        - Launch the kernel synchronously

        For grouped GEMM, tensor shapes, tensor strides, and tensor address are all provided
        by different tensors in global memory. The "initial" tensors only carry data type and
        majorness information.

        :param initial_a: Initial tensor A, used for data type and majorness information.
        :type initial_a: cute.Tensor
        :param initial_b: Initial tensor B, used for data type and majorness information.
        :type initial_b: cute.Tensor
        :param initial_c: Initial tensor C, used for data type and majorness information.
        :type initial_c: cute.Tensor
        :param group_count: The number of GEMM groups.
        :type group_count: cutlass.Constexpr[int]
        :param problem_shape_mnkl: Tensor containing the (M, N, K, L) shape for each group.
        :type problem_shape_mnkl: cute.Tensor
        :param strides_abc: Tensor containing the strides for A, B, and C for each group.
        :type strides_abc: cute.Tensor
        :param tensor_address_abc: Tensor containing the base addresses for A, B, and C for each group.
        :type tensor_address_abc: cute.Tensor
        :param total_num_clusters: Total number of clusters needed for all groups.
        :type total_num_clusters: cutlass.Constexpr[int]
        :param tensormap_cute_tensor: Tensor for storing tensormaps.
        :type tensormap_cute_tensor: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr[int]
        :param stream: CUDA stream for asynchronous execution.
        :type stream: cuda.CUstream
        :raises TypeError: If A and B data types do not match.
        """

        self.a_dtype = initial_a.element_type
        self.b_dtype = initial_b.element_type
        self.c_dtype = initial_c.element_type

        self.a_major_mode = utils.LayoutEnum.from_tensor(initial_a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(initial_b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(initial_c)
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            initial_a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            initial_b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        # Setup TMA store for C
        tma_atom_c = None
        tma_tensor_c = None
        epi_smem_layout = cute.slice_(self.epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            initial_c,
            epi_smem_layout,
            self.epi_tile,
        )

        self.tile_sched_params, grid = self._compute_grid(
            total_num_clusters, self.cluster_shape_mn, max_active_clusters
        )

        self.buffer_align_bytes = 1024
        self.size_tensormap_in_i64 = (
            0
            if self.tensormap_update_mode == utils.TensorMapUpdateMode.GMEM
            else GroupedGemmKernel.num_tensormaps
            * GroupedGemmKernel.bytes_per_tensormap
            // 8
        )

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            tensormap_buffer: cute.struct.MemRange[
                cutlass.Int64, self.size_tensormap_in_i64
            ]
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.epi_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            group_count,
            problem_shape_mnkl,
            strides_abc,
            tensor_address_abc,
            tensormap_cute_tensor,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
