            stream=stream,
            min_blocks_per_mp=1,
        )
        return

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        group_count: cutlass.Constexpr[int],
        problem_sizes_mnkl: cute.Tensor,
        strides_abc: cute.Tensor,
        ptrs_abc: cute.Tensor,
        tensormaps: cute.Tensor,
    ):
        """
        GPU device kernel performing the grouped GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coord inside cluster
        bid = cute.arch.block_idx()
        mma_tile_coord_v = bid[0] % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: tensormap buffer, a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tensormap_a_smem_ptr = None
        tensormap_b_smem_ptr = None
        tensormap_c_smem_ptr = None
        if cutlass.const_expr(
            self.tensormap_update_mode == utils.TensorMapUpdateMode.SMEM
        ):
            tensormap_smem_ptr = storage.tensormap_buffer.data_ptr()
            tensormap_a_smem_ptr = tensormap_smem_ptr
            tensormap_b_smem_ptr = (
                tensormap_a_smem_ptr + GroupedGemmKernel.bytes_per_tensormap // 8
            )
            tensormap_c_smem_ptr = (
                tensormap_b_smem_ptr + GroupedGemmKernel.bytes_per_tensormap // 8
            )

        #  init barrier for loading A, B with TMA
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
        # Accumulator barrier init
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
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        #
        # Setup smem tensor A/B/C
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )

        #
        # Compute multicast mask for A/B buffer full and empty
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        ab_empty_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            ab_empty_mcast_mask = a_full_mcast_mask | b_full_mcast_mask
        acc_full_mcast_mask = None
        if cutlass.const_expr(use_2cta_instrs):
            acc_full_mcast_mask = cute.make_layout_image_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mode=0
            )
            block_in_cluster_coord_vmnk_peer = (
                block_in_cluster_coord_vmnk[0] ^ 1,
                *block_in_cluster_coord_vmnk[1:],
            )
            a_full_mcast_mask_peer = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk_peer, mcast_mode=2
            )
            b_full_mcast_mask_peer = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk_peer, mcast_mode=1
            )
            ab_empty_mcast_mask = (
                a_full_mcast_mask_peer
                | b_full_mcast_mask_peer
                | cutlass.Int16(
                    0 if ab_empty_mcast_mask is None else ab_empty_mcast_mask
                )
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
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgC = thr_mma.partition_C(gC_mnl)

        #
        # Partition global/shared tensor for load A, B with TMA
        #
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
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        #
        # Get tensormap buffer address
        #
        grid_dim = cute.arch.grid_dim()
        tensormap_workspace_idx = (
            bid[2] * grid_dim[1] * grid_dim[0] + bid[1] * grid_dim[0] + bid[0]
        )

        tensormap_manager = utils.TensorMapManager(
            self.tensormap_update_mode, GroupedGemmKernel.bytes_per_tensormap
        )
        tensormap_a_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 0, None)].iterator
        )
        tensormap_b_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 1, None)].iterator
        )
        tensormap_c_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 2, None)].iterator
        )
        # Setup tensormap initialization pointer based on the mode
        if cutlass.const_expr(
            self.tensormap_update_mode == utils.TensorMapUpdateMode.SMEM
        ):
            tensormap_a_init_ptr = tensormap_a_smem_ptr
            tensormap_b_init_ptr = tensormap_b_smem_ptr
            tensormap_c_init_ptr = tensormap_c_smem_ptr
        else:
            tensormap_a_init_ptr = tensormap_a_ptr
            tensormap_b_init_ptr = tensormap_b_ptr
            tensormap_c_init_ptr = tensormap_c_ptr

        #
        # Persistent tile scheduling loop
        #
        # When the problem shapes are on device, we launch one CTA per SM.
        # The if condition later prevents the warps from extra CTAs from doing any work.
        tile_sched = utils.StaticPersistentGroupTileScheduler.create(
            tile_sched_params,
            bid,
            grid_dim,
            self.cluster_tile_shape_mnk,
            utils.create_initial_search_state(),
            group_count,
            problem_sizes_mnkl,
        )
        initial_work_tile_info = tile_sched.initial_work_tile_info()

        #
        # Specialized TMA load warp
        #
        if warp_idx == self.tma_warp_id and initial_work_tile_info.is_valid_tile:
            # Initialize tensormaps for A, B
            if cutlass.const_expr(self.delegate_tensormap_ab_init == False):
                tensormap_manager.init_tensormap_from_atom(
                    tma_atom_a, tensormap_a_init_ptr, self.tma_warp_id
                )
                tensormap_manager.init_tensormap_from_atom(
                    tma_atom_b, tensormap_b_init_ptr, self.tma_warp_id
                )

            tensormap_init_done = cutlass.Boolean(False)
            # group index of last tile
            last_group_idx = cutlass.Int32(-1)

            work_tile = initial_work_tile_info
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            while work_tile.is_valid_tile:
                grouped_gemm_cta_tile_info = work_tile.group_search_result

                cur_k_tile_cnt = grouped_gemm_cta_tile_info.cta_tile_count_k
                is_k_tile_cnt_zero = cur_k_tile_cnt == 0
                cur_group_idx = grouped_gemm_cta_tile_info.group_idx
                # Do not load any data if cur_k_tile_cnt is 0
                if not is_k_tile_cnt_zero:
                    is_group_changed = cur_group_idx != last_group_idx
                    # skip tensormap update if we're working on the same group
                    if is_group_changed:
                        real_tensor_a = self.make_tensor_for_tensormap_update(
                            cur_group_idx,
                            self.a_dtype,
                            (
                                grouped_gemm_cta_tile_info.problem_shape_m,
                                grouped_gemm_cta_tile_info.problem_shape_n,
                                grouped_gemm_cta_tile_info.problem_shape_k,
                            ),
                            strides_abc,
                            ptrs_abc,
                            0,  # 0 for tensor A
                        )
                        real_tensor_b = self.make_tensor_for_tensormap_update(
                            cur_group_idx,
                            self.b_dtype,
                            (
                                grouped_gemm_cta_tile_info.problem_shape_m,
                                grouped_gemm_cta_tile_info.problem_shape_n,
                                grouped_gemm_cta_tile_info.problem_shape_k,
                            ),
                            strides_abc,
                            ptrs_abc,
                            1,  # 1 for tensor B
                        )
                        # wait tensormap initialization complete before update
                        if not tensormap_init_done:
                            if cutlass.const_expr(self.delegate_tensormap_ab_init):
                                self.tensormap_ab_init_barrier.arrive_and_wait()
                            tensormap_manager.fence_tensormap_initialization()
                            tensormap_init_done = True

                        tensormap_manager.update_tensormap(
                            (real_tensor_a, real_tensor_b),
                            (tma_atom_a, tma_atom_b),
                            (tensormap_a_ptr, tensormap_b_ptr),
                            self.tma_warp_id,
                            (tensormap_a_smem_ptr, tensormap_b_smem_ptr),
                        )

                    mma_tile_coord_mnl = (
                        grouped_gemm_cta_tile_info.cta_tile_idx_m
                        // cute.size(tiled_mma.thr_id.shape),
                        grouped_gemm_cta_tile_info.cta_tile_idx_n,
                        0,
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

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                    ab_producer_state.reset_count()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < cur_k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state
                        )
                    # ensure the update to tensormap has completed before using it
                    if is_group_changed:
                        tensormap_manager.fence_tensormap_update(tensormap_a_ptr)
                        tensormap_manager.fence_tensormap_update(tensormap_b_ptr)
                        #
                        # Tma load loop
                        #
                    for k_tile in cutlass.range(0, cur_k_tile_cnt, 1, unroll=1):
                        # Wait for AB buffer empty
                        ab_pipeline.producer_acquire(
                            ab_producer_state, peek_ab_empty_status
                        )

                        # Load A/B with TMA
                        cute.copy(
                            tma_atom_a,
                            tAgA_slice[(None, ab_producer_state.count)],
                            tAsA[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(
                                ab_producer_state
                            ),
                            mcast_mask=a_full_mcast_mask,
                            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                                tensormap_a_ptr,
                                cute.AddressSpace.generic,
                            ),
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB_slice[(None, ab_producer_state.count)],
                            tBsB[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(
                                ab_producer_state
                            ),
                            mcast_mask=b_full_mcast_mask,
                            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                                tensormap_b_ptr,
                                cute.AddressSpace.generic,
                            ),
                        )

                        # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                        ab_producer_state.advance()
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if ab_producer_state.count < cur_k_tile_cnt:
                            peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                                ab_producer_state
                            )
                else:
                    # If tensormap initialization is not done, wait for it to complete
                    if not tensormap_init_done:
                        if cutlass.const_expr(self.delegate_tensormap_ab_init):
                            self.tensormap_ab_init_barrier.arrive_and_wait()
                        tensormap_manager.fence_tensormap_initialization()
                        tensormap_init_done = True
                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                last_group_idx = cur_group_idx

            #
            # Wait A/B buffer empty
            #
            ab_pipeline.producer_tail(ab_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id and initial_work_tile_info.is_valid_tile:
            #  Bar sync for retrieve tmem ptr from shared mem
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            #
            # Persistent tile scheduling loop
            #
            work_tile = initial_work_tile_info
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            # tile count we have searched
            while work_tile.is_valid_tile:
                cur_group_idx = work_tile.group_search_result.group_idx
                problem_shape_k = work_tile.group_search_result.problem_shape_k

                # MMA warp is only interested in number of tiles along K dimension
                cur_k_tile_cnt = (
                    problem_shape_k + self.cluster_tile_shape_mnk[2] - 1
                ) // self.cluster_tile_shape_mnk[2]
                is_k_tile_cnt_zero = cur_k_tile_cnt == 0

                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    if ab_consumer_state.count < cur_k_tile_cnt:
                        peek_ab_full_status = ab_pipeline.consumer_try_wait(
                            ab_consumer_state
                        )

                    #
                    # Wait for accumulator buffer empty
                    #
                    if not is_k_tile_cnt_zero:
                        acc_pipeline.producer_acquire(acc_producer_state)

                    #
                    # Reset the ACCUMULATE field for each tile
                    #
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                    #
                    # Mma mainloop
                    #
                    for k_tile in cutlass.range(0, cur_k_tile_cnt, 1, unroll=1):
                        # Wait for AB buffer full
                        ab_pipeline.consumer_wait(
                            ab_consumer_state, peek_ab_full_status
                        )
                        # tCtAcc += tCrA * tCrB
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                ab_consumer_state.index,
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
                        if ab_consumer_state.count < cur_k_tile_cnt:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                ab_consumer_state
                            )

                    #
                    # Async arrive accumulator buffer full
                    #
                    if not is_k_tile_cnt_zero:
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
        if warp_idx < self.mma_warp_id and initial_work_tile_info.is_valid_tile:
            # initialize tensormap A, B for TMA warp
            if cutlass.const_expr(self.delegate_tensormap_ab_init):
                tensormap_manager.init_tensormap_from_atom(
                    tma_atom_a, tensormap_a_init_ptr, self.epilog_warp_id[0]
                )
                tensormap_manager.init_tensormap_from_atom(
                    tma_atom_b, tensormap_b_init_ptr, self.epilog_warp_id[0]
                )
                # signal tensormap initialization has finished
                self.tensormap_ab_init_barrier.arrive_and_wait()
            # initialize tensorap for C
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_c,
                tensormap_c_init_ptr,
                self.epilog_warp_id[0],
            )
            # Alloc tensor memory buffer
            tmem.allocate(self.num_tmem_alloc_cols)

            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            epi_tidx = tidx
            #
            # Partition for epilogue
            #
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
            ) = self.epilog_gmem_copy_and_partition(tma_atom_c, tCgC, epi_tile, sC)

            #
            # Persistent tile scheduling loop
            #

            work_tile = initial_work_tile_info

            # wait tensormap initialization complete before update
            tensormap_manager.fence_tensormap_initialization()
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            # Threads/warps participating in tma store pipeline
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_epi_stage,
                producer_group=c_producer_group,
            )
            # group index of last tile
            last_group_idx = cutlass.Int32(-1)
            while work_tile.is_valid_tile:
                grouped_gemm_cta_tile_info = work_tile.group_search_result
                cur_group_idx = grouped_gemm_cta_tile_info.group_idx
                cur_k_tile_cnt = grouped_gemm_cta_tile_info.cta_tile_count_k
                is_k_tile_cnt_zero = cur_k_tile_cnt == 0
                is_group_changed = cur_group_idx != last_group_idx
                # We still need to store 0s when k_tile_cnt is 0
                if is_group_changed:
                    # construct tensor C based on real address, shape and stride information
                    real_tensor_c = self.make_tensor_for_tensormap_update(
                        cur_group_idx,
                        self.c_dtype,
                        (
                            grouped_gemm_cta_tile_info.problem_shape_m,
                            grouped_gemm_cta_tile_info.problem_shape_n,
                            grouped_gemm_cta_tile_info.problem_shape_k,
                        ),
                        strides_abc,
                        ptrs_abc,
                        2,  # 2 for tensor C
                    )
                    tensormap_manager.update_tensormap(
                        ((real_tensor_c),),
                        ((tma_atom_c),),
                        ((tensormap_c_ptr),),
                        self.epilog_warp_id[0],
                        (tensormap_c_smem_ptr,),
                    )

                mma_tile_coord_mnl = (
                    grouped_gemm_cta_tile_info.cta_tile_idx_m
                    // cute.size(tiled_mma.thr_id.shape),
                    grouped_gemm_cta_tile_info.cta_tile_idx_n,
                    0,
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
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_consumer_state.index)
                ]
                #
                # Wait for accumulator buffer full
                #
                # Not waiting for accumulator buffer full when k_tile_cnt is 0
                if not is_k_tile_cnt_zero:
                    acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
                # ensure the update to tensormap has completed before using it
                if is_group_changed:
                    if warp_idx == self.epilog_warp_id[0]:
                        tensormap_manager.fence_tensormap_update(tensormap_c_ptr)
                #
                # Store accumulator to global memory in subtiles
                #
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
                for subtile_idx in range(subtile_cnt):
                    #
                    # Store C to shared memory
                    #
                    epi_buffer = (num_prev_subtiles + subtile_idx) % self.num_epi_stage
                    #
                    # Load accumulator from tensor memory buffer to register
                    #
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    if not is_k_tile_cnt_zero:
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                        #
                        # Convert to output type
                        #
                        acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                        tRS_rC.store(acc_vec.to(self.c_dtype))
                    else:
                        tRS_rC.fill(0)
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rC,
                        tRS_sC[(None, None, None, epi_buffer)],
                    )
                    # Fence and barrier to make sure shared memory store is visible to TMA store
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    self.epilog_sync_barrier.arrive_and_wait()
                    #
                    # store C to global memory with TMA
                    #
                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(
                            tma_atom_c,
                            bSG_sC[(None, epi_buffer)],
                            bSG_gC[(None, subtile_idx)],
                            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                                tensormap_c_ptr,
                                cute.AddressSpace.generic,
                            ),
                        )
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    self.epilog_sync_barrier.arrive_and_wait()
                #
                # Async arrive accumulator buffer empty
                #
                if not is_k_tile_cnt_zero:
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                last_group_idx = cur_group_idx

            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

            #
            # Wait for C store complete
            #
            c_pipeline.producer_tail()

    @cute.jit
    def make_tensor_for_tensormap_update(
        self,
        group_idx: cutlass.Int32,
        dtype: Type[cutlass.Numeric],
        problem_shape_mnk: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
        strides_abc: cute.Tensor,
        tensor_address_abc: cute.Tensor,
        tensor_index: int,
    ):
        """Extract stride and tensor address for a given group and construct a global tensor.

        This function is used within the kernel to dynamically create a CUTE tensor
        representing A, B, or C for the current group being processed, using the
        group-specific address, shape, and stride information.

        :param group_idx: The index of the current group within the grouped GEMM.
        :type group_idx: cutlass.Int32
        :param dtype: The data type of the tensor elements (e.g., cutlass.Float16).
        :type dtype: Type[cutlass.Numeric]
        :param problem_shape_mnk: The (M, N, K) problem shape for the current group.
        :type problem_shape_mnk: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]
        :param strides_abc: Tensor containing strides for A, B, C for all groups. Layout: (group_count, 3, 2).
        :type strides_abc: cute.Tensor
        :param tensor_address_abc: Tensor containing global memory addresses for A, B, C for all groups. Layout: (group_count, 3).
        :type tensor_address_abc: cute.Tensor
        :param tensor_index: Specifies which tensor to create: 0 for A, 1 for B, 2 for C.
        :type tensor_index: int
        :return: A CUTE tensor representing the requested global memory tensor (A, B, or C) for the specified group.
        :rtype: cute.Tensor
        :raises TypeError: If the provided dtype is not a subclass of cutlass.Numeric.
        """
        ptr_i64 = tensor_address_abc[(group_idx, tensor_index)]
        if cutlass.const_expr(
            not isclass(dtype) or not issubclass(dtype, cutlass.Numeric)
        ):
            raise TypeError(
                f"dtype must be a type of cutlass.Numeric, got {type(dtype)}"
            )
        tensor_gmem_ptr = cute.make_ptr(
            dtype, ptr_i64, cute.AddressSpace.gmem, assumed_align=16
        )

        strides_tensor_gmem = strides_abc[(group_idx, tensor_index, None)]
        strides_tensor_reg = cute.make_rmem_tensor(
            cute.make_layout(2),
            strides_abc.element_type,
        )
        cute.autovec_copy(strides_tensor_gmem, strides_tensor_reg)
        stride_mn = strides_tensor_reg[0]
        stride_k = strides_tensor_reg[1]
        c1 = cutlass.Int32(1)
        c0 = cutlass.Int32(0)

        if cutlass.const_expr(tensor_index == 0):  # tensor A
            m = problem_shape_mnk[0]
            k = problem_shape_mnk[2]
            return cute.make_tensor(
                tensor_gmem_ptr,
                cute.make_layout((m, k, c1), stride=(stride_mn, stride_k, c0)),
            )
        elif cutlass.const_expr(tensor_index == 1):  # tensor B
            n = problem_shape_mnk[1]
            k = problem_shape_mnk[2]
            return cute.make_tensor(
                tensor_gmem_ptr,
                cute.make_layout((n, k, c1), stride=(stride_mn, stride_k, c0)),
            )
        else:  # tensor C
            m = problem_shape_mnk[0]
            n = problem_shape_mnk[1]
            return cute.make_tensor(
                tensor_gmem_ptr,
                cute.make_layout((m, n, c1), stride=(stride_mn, stride_k, c0)),
            )

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
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
        # Make tiledCopy for tensor memory load(t2r)
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
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
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
        tma_atom_c: cute.CopyAtom,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
    ) -> tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """Make tiledCopy for global memory store, then use it to partition
        shared memory (source) and global memory (destination) for TMA store version.

        :param tma_atom_c: The TMA copy atom configured for storing tensor C.
        :type tma_atom_c: cute.CopyAtom
        :param gC_mnl: The global memory tensor C.
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler defining the granularity of the operation.
        :type epi_tile: cute.Tile
        :param sC: The shared memory epilogue buffer tensor.
        :type sC: cute.Tensor
        :return: A tuple containing:
                 - tma_atom_c: The input TMA copy atom (passed through).
                 - bSG_sC: The source shared memory tensor partitioned for the TMA operation.
                 - tCgC: The destination global memory tensor partitioned for the TMA operation.
        :rtype: tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]
        """
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
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
        mma_tiler_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        smem_capacity: int,
        occupancy: int,
    ) -> tuple[int, int, int]:
        """Computes the number of stages for accumulator, A/B operands, and epilogue based on heuristics.

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
        :param c_layout: Layout enum of operand C in global memory.
        :type c_layout: utils.LayoutEnum
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (accumulator stages, A/B operand stages, epilogue stages)
        :rtype: tuple[int, int, int]
        """
        # Default accumulator and epilogue stages
        num_acc_stage = 2
        num_epi_stage = 2

        # Calculate smem layout and size for one stage of A, B, and Epilogue
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  # stage=1
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  # stage=1
        )
        epi_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,  # stage=1
        )
        ab_bytes_per_stage = cute.size_in_bytes(
            a_dtype, a_smem_layout_stage_one
        ) + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)

        epi_bytes_per_stage = cute.size_in_bytes(c_dtype, epi_smem_layout_staged_one)
        epi_bytes = epi_bytes_per_stage * num_epi_stage

        # Calculate A/B stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial epilogue bytes
        # Divide remaining by bytes needed per A/B stage
        num_ab_stage = (
            smem_capacity // occupancy
            - GroupedGemmKernel.reserved_smem_bytes
            - epi_bytes
        ) // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B stages and reserved bytes
        # Add remaining unused smem to epilogue
        remaining_smem = (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (GroupedGemmKernel.reserved_smem_bytes + epi_bytes)
        )
        num_epi_stage += remaining_smem // (occupancy * epi_bytes_per_stage)
        return num_acc_stage, num_ab_stage, num_epi_stage

    @staticmethod
    def _compute_grid(
        total_num_clusters: int,
        cluster_shape_mn: tuple[int, int],
        max_active_clusters: cutlass.Constexpr[int],
    ) -> tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]:
        """Compute tile scheduler parameters and grid shape for grouped GEMM operations.

        :param total_num_clusters: Total number of clusters to process across all groups.
        :type total_num_clusters: int
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr[int]

        :return: A tuple containing:
            - tile_sched_params: Parameters for the persistent tile scheduler.
            - grid: Grid shape for kernel launch.
        :rtype: tuple[utils.PersistentTileSchedulerParams, tuple[int, ...]]
        """
        # Create problem shape with M, N dimensions from cluster shape
        # and L dimension representing the total number of clusters.
        problem_shape_ntile_mnl = (
            cluster_shape_mn[0],
            cluster_shape_mn[1],
            cutlass.Int32(total_num_clusters),
        )

        tile_sched_params = utils.PersistentTileSchedulerParams(
            problem_shape_ntile_mnl, (*cluster_shape_mn, 1)
        )

        grid = utils.StaticPersistentGroupTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @staticmethod
    def _get_mbar_smem_bytes(**kwargs_stages: int) -> int:
        """Calculate shared memory consumption for memory barriers based on provided stages.

        Each stage requires 2 barriers, and each barrier consumes 8 bytes of shared memory.
        The total consumption is the sum across all provided stages. This function calculates the total
        shared memory needed for these barriers.

        :param kwargs_stages: Variable keyword arguments where each key is a stage name
                              (e.g., num_acc_stage, num_ab_stage) and each value is the
                              number of stages of that type.
        :type kwargs_stages: int
        :return: Total shared memory bytes required for all memory barriers.
        :rtype: int
        """
        num_barriers_per_stage = 2
        num_bytes_per_barrier = 8
        mbar_smem_consumption = sum(
            [
                num_barriers_per_stage * num_bytes_per_barrier * stage
                for stage in kwargs_stages.values()
            ]
        )
        return mbar_smem_consumption

    @staticmethod
    def _get_tensormap_smem_bytes(
        tensormap_update_mode: utils.TensorMapUpdateMode,
    ) -> int:
        """Get the SMEM consumption for the tensormap buffer based on the update mode.

        :param tensormap_update_mode: Specifies whether tensormaps are updated in GMEM or SMEM.
        :type tensormap_update_mode: utils.TensorMapUpdateMode
        :return: The shared memory bytes required for the tensormap buffer. Returns 0 if mode is GMEM.
        :rtype: int
        :raises ValueError: If an invalid tensormap update mode is provided.
        """
        if tensormap_update_mode == utils.TensorMapUpdateMode.GMEM:
            return 0
        elif tensormap_update_mode == utils.TensorMapUpdateMode.SMEM:
            return (
                GroupedGemmKernel.bytes_per_tensormap * GroupedGemmKernel.num_tensormaps
            )
        else:
            raise ValueError(f"Invalid tensormap update mode: {tensormap_update_mode}")

    @staticmethod
    def _compute_num_tmem_alloc_cols(
        tiled_mma: cute.TiledMma,
        mma_tiler: tuple[int, int, int],
        num_acc_stage: int,
    ) -> int:
        """
        Compute the number of tensor memory allocation columns.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler: The shape (M, N, K) of the MMA tile.
        :type mma_tiler: tuple[int, int, int]
        :param acc_stage: The stage of the accumulator tensor.
        :type acc_stage: int

        :return: The number of tensor memory allocation columns.
        :rtype: int
        """
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage))
        num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)

        return num_tmem_alloc_cols

    # Size of smem we reserved for mbarrier, tensor memory management and tensormap update
    reserved_smem_bytes = 1024
    bytes_per_tensormap = 128
    num_tensormaps = 3
    # size of smem used for tensor memory management
    tensor_memory_management_bytes = 12


# Create tensor and return the pointer, tensor, and stride
def create_tensor_and_stride(
    l: int,
    mode0: int,
    mode1: int,
    is_mode0_major: bool,
    dtype: type[cutlass.Numeric],
    is_dynamic_layout: bool = True,
    torch_tensor_cpu: torch.Tensor = None,
) -> tuple[int, torch.Tensor, cute.Tensor, torch.Tensor, tuple[int, int]]:
    """Create GPU tensor from either a new or existing CPU tensor.

    :param torch_tensor_cpu: Optional existing CPU tensor to reuse. If None, creates a new one.
    :type torch_tensor_cpu: torch.Tensor, optional
    """
    if torch_tensor_cpu is None:
        # Create new CPU tensor
        torch_tensor_cpu = cutlass_torch.matrix(l, mode0, mode1, is_mode0_major, dtype)

    # Create GPU tensor from CPU tensor (new or existing)
    cute_tensor, torch_tensor = cutlass_torch.cute_tensor_like(
        torch_tensor_cpu, dtype, is_dynamic_layout, assumed_align=16
    )
    return (
        torch_tensor.data_ptr(),
        torch_tensor,
        cute_tensor,
        torch_tensor_cpu,
        torch_tensor.stride()[:-1],
    )


def create_tensors_for_all_groups(
    problem_sizes_mnkl: List[tuple[int, int, int, int]],
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    torch_fp32_tensors_abc: List[List[torch.Tensor]] = None,
) -> tuple[
    List[List[int]],
    List[List[torch.Tensor]],
    List[tuple],
    List[List[tuple]],
    List[List[torch.Tensor]],
]:
    if torch_fp32_tensors_abc is not None and len(torch_fp32_tensors_abc) != len(
        problem_sizes_mnkl
    ):
        raise ValueError("torch_fp32_tensors_abc must have one entry per group")

    # Initialize lists to store tensors for all groups
    new_torch_fp32_tensors_abc = (
        [] if torch_fp32_tensors_abc is None else torch_fp32_tensors_abc
    )
    torch_tensors_abc = []
    cute_tensors_abc = []
    strides_abc = []
    ptrs_abc = []

    # Iterate through all groups and create tensors for each group
    for group_idx, (m, n, k, l) in enumerate(problem_sizes_mnkl):
        # Get existing CPU tensors if available, otherwise None
        existing_cpu_a = (
            torch_fp32_tensors_abc[group_idx][0] if torch_fp32_tensors_abc else None
        )
        existing_cpu_b = (
            torch_fp32_tensors_abc[group_idx][1] if torch_fp32_tensors_abc else None
        )
        existing_cpu_c = (
            torch_fp32_tensors_abc[group_idx][2] if torch_fp32_tensors_abc else None
        )

        # Create tensors (reusing CPU tensors if provided)
        (
            ptr_a,
            torch_tensor_a,
            cute_tensor_a,
            tensor_fp32_a,
            stride_mk_a,
        ) = create_tensor_and_stride(
            l, m, k, a_major == "m", ab_dtype, torch_tensor_cpu=existing_cpu_a
        )
        (
            ptr_b,
            torch_tensor_b,
            cute_tensor_b,
            tensor_fp32_b,
            stride_nk_b,
        ) = create_tensor_and_stride(
            l, n, k, b_major == "n", ab_dtype, torch_tensor_cpu=existing_cpu_b
        )
        (
            ptr_c,
            torch_tensor_c,
            cute_tensor_c,
            tensor_fp32_c,
            stride_mn_c,
        ) = create_tensor_and_stride(
            l, m, n, c_major == "m", c_dtype, torch_tensor_cpu=existing_cpu_c
        )

        # Only append to new_torch_fp32_tensors_abc if we created new CPU tensors
        if torch_fp32_tensors_abc is None:
            new_torch_fp32_tensors_abc.append(
                [tensor_fp32_a, tensor_fp32_b, tensor_fp32_c]
            )

        ptrs_abc.append([ptr_a, ptr_b, ptr_c])
        torch_tensors_abc.append([torch_tensor_a, torch_tensor_b, torch_tensor_c])
        strides_abc.append([stride_mk_a, stride_nk_b, stride_mn_c])
        cute_tensors_abc.append(
            (
                cute_tensor_a,
                cute_tensor_b,
                cute_tensor_c,
            )
        )

    return (
        ptrs_abc,
        torch_tensors_abc,
        cute_tensors_abc,
        strides_abc,
        new_torch_fp32_tensors_abc,
    )


def run(
    num_groups: int,
    problem_sizes_mnkl: tuple[int, int, int, int],
    host_problem_shape_available: bool,
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    mma_tiler_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
    use_2cta_instrs: bool,
    tensormap_update_mode: utils.TensorMapUpdateMode,
    tolerance: float,
    warmup_iterations: int,
    iterations: int,
    skip_ref_check: bool,
    use_cold_l2: bool = False,
    **kwargs,
):
    """Run grouped GEMM example with specified configurations.

    :param use_cold_l2: Whether to use circular buffer strategy to ensure cold L2 cache, defaults to False
    :type use_cold_l2: bool, optional
    :return: Execution time of the GEMM kernel in microseconds
    :rtype: float
    """
    print("Running Blackwell Grouped GEMM test with:")
    print(f"{num_groups} groups")
    for i, (m, n, k, l) in enumerate(problem_sizes_mnkl):
        print(f"Group {i}: {m}x{n}x{k}x{l}")
    print(f"AB dtype: {ab_dtype}, C dtype: {c_dtype}, Acc dtype: {acc_dtype}")
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"Mma Tiler (M, N): {mma_tiler_mn}, Cluster Shape (M, N): {cluster_shape_mn}")
    print(f"2CTA MMA instructions: {'True' if use_2cta_instrs else 'False'}")
    print(f"Tensor map update mode: {tensormap_update_mode}")
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")
    print(f"Use cold L2: {'True' if use_cold_l2 else 'False'}")

    # Skip unsupported types
    if ab_dtype not in {
        cutlass.Float16,
        cutlass.BFloat16,
    }:
        raise ValueError(f"Skip unsupported ab_dtype {ab_dtype}")
    if c_dtype not in {cutlass.Float16, cutlass.BFloat16, cutlass.Float32}:
        raise ValueError(f"Skip unsupported c_dtype {c_dtype}")
    # Skip unsupported acc dtype
    if acc_dtype not in {cutlass.Float32, cutlass.Float16}:
        raise ValueError(f"Skip unsupported acc_dtype {acc_dtype}")
    # Skip invalid ab_dtype and acc_dtype combination
    if ab_dtype == cutlass.BFloat16 and acc_dtype == cutlass.Float16:
        raise ValueError("Skip invalid ab_dtype and acc_dtype combination")
    # Skip invalid mma tile shape
    if not (
        (not use_2cta_instrs and mma_tiler_mn[0] in [64, 128])
        or (use_2cta_instrs and mma_tiler_mn[0] in [128, 256])
    ):
        raise ValueError(f"Skip invalid mma tiler M {mma_tiler_mn[0]}")
    if mma_tiler_mn[1] not in range(32, 257, 32):
        raise ValueError(f"Skip invalid mma tiler N {mma_tiler_mn[1]}")
    # Skip illegal cluster shape
    if cluster_shape_mn[0] % (2 if use_2cta_instrs else 1) != 0:
        raise ValueError(
            f"cluster_shape_m need align with use_2cta_instrs config {cluster_shape_mn}"
        )
    # Skip invalid cluster shape
    is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
    if (
        cluster_shape_mn[0] * cluster_shape_mn[1] > 16
        or cluster_shape_mn[0] <= 0
        or cluster_shape_mn[1] <= 0
        or not is_power_of_2(cluster_shape_mn[0])
        or not is_power_of_2(cluster_shape_mn[1])
    ):
        raise ValueError(f"Skip invalid cluster shape {cluster_shape_mn}")

    # Skip illegal problem shape for load/store alignment
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
        raise ValueError("Skip invalid problem alignment")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    # Create tensors for all groups using the new function
    (
        ptrs_abc,
        torch_tensors_abc,
        cute_tensors_abc,
        strides_abc,
        torch_fp32_tensors_abc,
    ) = create_tensors_for_all_groups(
        problem_sizes_mnkl,
        ab_dtype,
        c_dtype,
        a_major,
        b_major,
        c_major,
    )

    # Setup inital tensors for TMA of A,B and C
    alignment = 16  # 16 bytes aligned
    min_ab_size = alignment * 8 // ab_dtype.width
    min_c_size = alignment * 8 // c_dtype.width
    initial_cute_tensors_abc = [
        create_tensor_and_stride(1, min_ab_size, min_ab_size, a_major == "m", ab_dtype)[
            2
        ],
        create_tensor_and_stride(1, min_ab_size, min_ab_size, b_major == "n", ab_dtype)[
            2
        ],
        create_tensor_and_stride(1, min_c_size, min_c_size, c_major == "m", c_dtype)[2],
    ]

    hardware_info = utils.HardwareInfo()
    sm_count = hardware_info.get_max_active_clusters(1)
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    # Prepare tensormap buffer for each SM
    num_tensormap_buffers = sm_count
    tensormap_shape = (
        num_tensormap_buffers,
        GroupedGemmKernel.num_tensormaps,
        GroupedGemmKernel.bytes_per_tensormap // 8,
    )
    tensor_of_tensormap, tensor_of_tensormap_torch = cutlass_torch.cute_tensor_like(
        torch.empty(tensormap_shape, dtype=torch.int64),
        cutlass.Int64,
        is_dynamic_layout=False,
    )

    grouped_gemm = GroupedGemmKernel(
        acc_dtype,
        use_2cta_instrs,
        mma_tiler_mn,
        cluster_shape_mn,
        tensormap_update_mode,
    )

    # layout (num_groups, 4):(4, 1)
    (
        tensor_of_dim_size_mnkl,
        tensor_of_dim_size_mnkl_torch,
    ) = cutlass_torch.cute_tensor_like(
        torch.tensor(problem_sizes_mnkl, dtype=torch.int32),
        cutlass.Int32,
        is_dynamic_layout=False,
        assumed_align=16,
    )

    # layout (num_groups, 3, 2):(6, 2, 1)
    tensor_of_strides_abc, tensor_of_strides_abc_torch = cutlass_torch.cute_tensor_like(
        torch.tensor(strides_abc, dtype=torch.int32),
        cutlass.Int32,
        is_dynamic_layout=False,
        assumed_align=16,
    )

    # layout (num_groups,3):(3, 1)
    tensor_of_ptrs_abc, tensor_of_ptrs_abc_torch = cutlass_torch.cute_tensor_like(
        torch.tensor(ptrs_abc, dtype=torch.int64),
        cutlass.Int64,
        is_dynamic_layout=False,
        assumed_align=16,
    )

    # Compute total number of cluster tiles we need to compute for given grouped GEMM problem
    def compute_total_num_clusters(
        problem_sizes_mnkl: List[tuple[int, int, int, int]],
        cluster_tile_shape_mn: tuple[int, int],
    ) -> int:
        total_num_clusters = 0
        for m, n, _, _ in problem_sizes_mnkl:
            num_clusters_mn = tuple(
                (x + y - 1) // y for x, y in zip((m, n), cluster_tile_shape_mn)
            )
            total_num_clusters += functools.reduce(lambda x, y: x * y, num_clusters_mn)
        return total_num_clusters

    # Compute cluster tile shape
    def compute_cluster_tile_shape(
        mma_tiler_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
        use_2cta_instrs: bool,
    ) -> tuple[int, int]:
        cta_tile_shape_mn = list(mma_tiler_mn)
        if use_2cta_instrs:
            cta_tile_shape_mn[0] = cta_tile_shape_mn[0] // 2
        return tuple(x * y for x, y in zip(cta_tile_shape_mn, cluster_shape_mn))

    cluster_tile_shape_mn = compute_cluster_tile_shape(
        mma_tiler_mn, cluster_shape_mn, use_2cta_instrs
    )

    # If the host problem shape is available, we will launch the grid with only
    # the necessary clusters. The function compute_total_num_clusters() does that.
    # If the problem shape only exists on device, we will need to launch all active
    # clusters possible on a device.
    if host_problem_shape_available:
        print("Problem shapes available on host and device")
        total_num_clusters = compute_total_num_clusters(
            problem_sizes_mnkl, cluster_tile_shape_mn
        )
    else:
        print("Problem shapes available only on device")
        total_num_clusters = max_active_clusters

    # Initialize Stream
    current_stream = cutlass_torch.default_stream()

    # try to check CUDA version to decide the opt level
    try:
        from cutlass import CUDA_VERSION

        opt_level = (
            3
            if CUDA_VERSION.major < 13
            or (CUDA_VERSION.major == 13 and CUDA_VERSION.minor < 1)
            else 2
        )
    except ImportError:
        opt_level = 3
    # Compile grouped GEMM kernel
    compiled_grouped_gemm = cute.compile(
        grouped_gemm,
        initial_cute_tensors_abc[0],
        initial_cute_tensors_abc[1],
        initial_cute_tensors_abc[2],
        num_groups,
        tensor_of_dim_size_mnkl,
        tensor_of_strides_abc,
        tensor_of_ptrs_abc,
        total_num_clusters,
        tensor_of_tensormap,
        max_active_clusters,
        current_stream,
        options=f"--opt-level {opt_level}",
    )

    if not skip_ref_check:
        compiled_grouped_gemm(
            initial_cute_tensors_abc[0],
            initial_cute_tensors_abc[1],
            initial_cute_tensors_abc[2],
            tensor_of_dim_size_mnkl,
            tensor_of_strides_abc,
            tensor_of_ptrs_abc,
            tensor_of_tensormap,
            current_stream,
        )

        # Compute reference result
        for i, (a, b, c) in enumerate(torch_tensors_abc):
            ref = torch.einsum(
                "mkl,nkl->mnl",
                a.cpu().to(dtype=torch.float32),
                b.cpu().to(dtype=torch.float32),
            )
            print(f"checking group {i}")
            torch.testing.assert_close(
                c.cpu(),
                ref.to(cutlass_torch.dtype(c_dtype)),
                atol=tolerance,
                rtol=1e-05,
            )

    if iterations <= 0:
        return 0

    def generate_tensors():
        # Reuse existing CPU tensors and create new GPU tensors from them
        (
            ptrs_abc_workspace,
            torch_tensors_abc_workspace,
            cute_tensors_abc_workspace,
            strides_abc_workspace,
            _,
        ) = create_tensors_for_all_groups(
            problem_sizes_mnkl,
            ab_dtype,
            c_dtype,
            a_major,
            b_major,
            c_major,
            torch_fp32_tensors_abc,
        )

        initial_cute_tensors_abc_workspace = [
            create_tensor_and_stride(
                1, min_ab_size, min_ab_size, a_major == "m", ab_dtype
            )[2],
            create_tensor_and_stride(
                1, min_ab_size, min_ab_size, b_major == "n", ab_dtype
            )[2],
            create_tensor_and_stride(
                1, min_c_size, min_c_size, c_major == "m", c_dtype
            )[2],
        ]

        # Create new tensors for this workspace
        tensor_of_strides_abc_workspace, _ = cutlass_torch.cute_tensor_like(
            torch.tensor(strides_abc_workspace, dtype=torch.int32),
            cutlass.Int32,
            is_dynamic_layout=False,
            assumed_align=16,
        )

        tensor_of_ptrs_abc_workspace, _ = cutlass_torch.cute_tensor_like(
            torch.tensor(ptrs_abc_workspace, dtype=torch.int64),
            cutlass.Int64,
            is_dynamic_layout=False,
            assumed_align=16,
        )

        tensormap_workspace, _ = cutlass_torch.cute_tensor_like(
            torch.empty(tensormap_shape, dtype=torch.int64),
            cutlass.Int64,
            is_dynamic_layout=False,
        )

        args = testing.JitArguments(
            initial_cute_tensors_abc_workspace[0],
            initial_cute_tensors_abc_workspace[1],
            initial_cute_tensors_abc_workspace[2],
            tensor_of_dim_size_mnkl,
            tensor_of_strides_abc_workspace,
            tensor_of_ptrs_abc_workspace,
            tensormap_workspace,
            current_stream,
        )
        args.add_to_scope([torch_tensors_abc_workspace])
        return args

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = (
            sum(
                [
                    sum(
                        [
                            torch_tensor.numel() * torch_tensor.element_size()
                            for torch_tensor in group_tensors
                        ]
                    )
                    for group_tensors in torch_tensors_abc
                ]
            )
            +
            # Add size of strides tensor
            tensor_of_strides_abc_torch.numel()
            * tensor_of_strides_abc_torch.element_size()
            +
            # Add size of ptrs tensor
            tensor_of_ptrs_abc_torch.numel() * tensor_of_ptrs_abc_torch.element_size()
            +
            # Add size of tensormap tensor
            tensor_of_tensormap_torch.numel() * tensor_of_tensormap_torch.element_size()
        )
        workspace_count = testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    exec_time = testing.benchmark(
        compiled_grouped_gemm,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        stream=current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    runtime_s = exec_time / 1.0e6
    fmas = 0
    for group in range(num_groups):
        [M, N, K, _] = problem_sizes_mnkl[group]
        fmas += M * N * K
    flop = 2 * fmas
    gflop = flop / 1.0e9
    gflops = gflop / runtime_s

    print("Average Runtime : ", exec_time / 1000, "ms")
    print("GFLOPS          : ", gflops)

    return exec_time  # Return execution time in microseconds


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str) -> tuple[int, ...]:
        try:
            return tuple(int(x.strip()) for x in s.split(","))
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid format. Expected comma-separated integers."
            )

    def parse_comma_separated_tuples(s: str) -> List[tuple[int, ...]]:
        if s.strip().startswith("("):
            # Split on ),( to separate tuples
            tuples = s.strip("()").split("),(")
            result = []
            tuple_len = None

            for t in tuples:
                # Parse individual tuple
                nums = [int(x.strip()) for x in t.split(",")]

                # Validate tuple length consistency
                if tuple_len is None:
                    tuple_len = len(nums)
                elif len(nums) != tuple_len:
                    raise argparse.ArgumentTypeError(
                        "All tuples must have the same length"
                    )

                result.append(tuple(nums))
            return result

        raise argparse.ArgumentTypeError(
            "Invalid format. Expected comma-separated integers or list of tuples"
        )

    parser = argparse.ArgumentParser(
        description="Example of Grouped GEMM on Blackwell."
    )
    parser.add_argument(
        "--num_groups",
        type=int,
        default=3,
        help="Number of groups",
    )
    parser.add_argument(
        "--problem_sizes_mnkl",
        type=parse_comma_separated_tuples,
        default=((128, 128, 128, 1), (512, 128, 128, 1), (128, 256, 128, 1)),
        help="a tuple of problem sizes for each group (comma-separated tuples)",
    )
    parser.add_argument(
        "--host_problem_shape_available",
        action="store_true",
        help="Enable the compute of grid based upon host problem shape",
    )
    parser.add_argument(
        "--mma_tiler_mn",
        type=parse_comma_separated_ints,
        default=(128, 128),
        help="Mma tile shape (comma-separated)",
    )
    parser.add_argument(
        "--cluster_shape_mn",
        type=parse_comma_separated_ints,
        default=(1, 1),
        help="Cluster shape (comma-separated)",
    )
    parser.add_argument(
        "--tensormap_update_mode",
        type=str,
        default="SMEM",
        help="Tensor map update mode",
    )
    parser.add_argument("--ab_dtype", type=cutlass.dtype, default=cutlass.Float16)
    parser.add_argument("--c_dtype", type=cutlass.dtype, default=cutlass.Float16)
    parser.add_argument("--acc_dtype", type=cutlass.dtype, default=cutlass.Float32)
    parser.add_argument(
        "--use_2cta_instrs",
        action="store_true",
        help="Enable 2CTA MMA instructions feature",
    )
    parser.add_argument("--a_major", choices=["k", "m"], type=str, default="k")
    parser.add_argument("--b_major", choices=["k", "n"], type=str, default="k")
    parser.add_argument("--c_major", choices=["n", "m"], type=str, default="n")
    parser.add_argument(
        "--tolerance", type=float, default=1e-01, help="Tolerance for validation"
    )
    parser.add_argument(
        "--warmup_iterations", type=int, default=0, help="Warmup iterations"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run the kernel",
    )
    parser.add_argument(
        "--skip_ref_check", action="store_true", help="Skip reference checking"
    )
    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        default=False,
        help="Use circular buffer tensor sets to ensure L2 cold cache",
    )

    args = parser.parse_args()

    if (
        len(args.problem_sizes_mnkl) != 0
        and len(args.problem_sizes_mnkl) != args.num_groups
    ):
        parser.error("--problem_sizes_mnkl must contain exactly num_groups tuples")

    # l mode must be 1 for all groups
    for _, _, _, l in args.problem_sizes_mnkl:
        if l != 1:
            parser.error("l must be 1 for all groups")

    if len(args.mma_tiler_mn) != 2:
        parser.error("--mma_tiler_mn must contain exactly 2 values")

    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")

    if args.tensormap_update_mode not in ["GMEM", "SMEM"]:
        parser.error("--tensormap_update_mode must be GMEM or SMEM")

    if args.tensormap_update_mode == "GMEM":
        tensormap_update_mode = utils.TensorMapUpdateMode.GMEM
    else:
        tensormap_update_mode = utils.TensorMapUpdateMode.SMEM

    torch.manual_seed(2025)

    run(
        args.num_groups,
        args.problem_sizes_mnkl,
        args.host_problem_shape_available,
        args.ab_dtype,
        args.c_dtype,
        args.acc_dtype,
        args.a_major,
        args.b_major,
        args.c_major,
        args.mma_tiler_mn,
        args.cluster_shape_mn,
        args.use_2cta_instrs,
        tensormap_update_mode,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
    )
    print("PASS")
<|endoftext|>"""Kernel wrapper for dense_blockscaled_gemm_persistent vendored template."""

from __future__ import annotations

import ast
import dataclasses
import functools
import itertools
import logging
from collections.abc import Callable, Generator  # noqa: TC003

import cutlass.operators
from cutlass.operators import ScaleMode, ScaleSwizzleMode
from cutlass.operators.arch import TargetSm
from cutlass.operators.arguments import GemmArguments
from cutlass.operators.artifact import CompiledArtifact  # noqa: TC002
from cutlass.operators.metadata import (
    DenseTensorConstraints,
    GemmOperandsMetadata,
    OperatorMetadata,
    ScaledOperandConstraints,
    Sm100DesignMetadata,
)
from cutlass.operators.mma import BlackwellTcgen05Mma
from cutlass.operators.providers.cutedsl import integration_utils as cutedsl_utils
from cutlass.operators.providers.cutedsl.operator import CuteDslOperator
from cutlass.operators.status import Status
from cutlass.operators.utils.common import tuple_to_string
from cutlass.operators.utils.device import to_cuda_stream
from cutlass.operators.utils.tensor import strides_to_layout_string


log = logging.getLogger(__name__)


_ONES_ALPHA: dict = {}


@dataclasses.dataclass(frozen=True)
class _EpilogueABI:
    """Runtime tensors and metadata derived from a traced epilogue signature."""

    inputs: tuple
    input_kinds: tuple[int, ...]
    outputs: tuple
    output_count: int
    primary_output: int

    @classmethod
    def from_args(cls, args, tensor_attr: str) -> _EpilogueABI:
        outputs, output_count, primary_output = _epilogue_outputs(args, tensor_attr)
        return cls(
            _epilogue_tensors(args, tensor_attr),
            _epilogue_tensor_kinds(args),
            outputs,
            output_count,
            primary_output,
        )


@functools.lru_cache(maxsize=256)
def _epilogue_signature(epilogue_fn) -> tuple[tuple[str, ...], tuple[str, ...]]:
    from cutlass.operators.fusion import trace_in_out

    inputs, outputs = trace_in_out(epilogue_fn)
    return (
        tuple(name for name in inputs if name != "accum"),
        tuple(outputs),
    )


def _epilogue_tensors(args, attr: str) -> tuple:
    epilogue = getattr(args, "epilogue", None)
    tensors = (
        ()
        if epilogue is None
        else tuple(
            getattr(_epilogue_abi_tensor(epilogue.tensors[name]), attr)
            for name in _epilogue_signature(epilogue.epilogue_fn)[0]
        )
    )
    if len(tensors) > 4:
        raise NotImplementedError("NVGEMM scaled epilogues support up to four inputs")
    return tensors + (None,) * (4 - len(tensors))


def _epilogue_abi_tensor(tensor):
    runtime_tensor = tensor.runtime_tensor
    leading = runtime_tensor.shape[0]
    if leading % 8 == 0:
        return tensor
    from cutlass.operators.utils.tensor import TensorWrapper

    padded_shape = (leading + 7) // 8 * 8, *runtime_tensor.shape[1:]
    import torch

    padded = torch.empty(
        padded_shape,
        dtype=runtime_tensor.dtype,
        device=runtime_tensor.device,
    )
    padded[:leading].copy_(runtime_tensor)
    return TensorWrapper(padded)


def _epilogue_tensor_kinds(args) -> tuple[int, ...]:
    epilogue = getattr(args, "epilogue", None)
    if epilogue is None:
        return (0, 0, 0, 0)
    kinds = []
    output_m, output_n = args.out.shape[-2:]
    for name in _epilogue_signature(epilogue.epilogue_fn)[0]:
        shape = epilogue.tensors[name].shape
        if shape[-1] == 1 and (len(shape) == 1 or shape[-2] == 1):
            if output_n == 1:
                kinds.append(2)
            elif output_m == 1:
                kinds.append(3)
            else:
                raise NotImplementedError(
                    "NVGEMM scaled epilogues do not support scalar tensor inputs"
                )
        elif len(shape) == 1 or shape[-2] == 1:
            kinds.append(2)
        elif shape[-1] == 1:
            kinds.append(3)
        else:
            kinds.append(1)
    return tuple(kinds) + (0,) * (4 - len(kinds))


def _epilogue_outputs(args, attr: str) -> tuple[tuple, int, int]:
    epilogue = getattr(args, "epilogue", None)
    if epilogue is None:
        return (None, None, None), 1, 0
    output_names = _epilogue_signature(epilogue.epilogue_fn)[1]
    if not output_names or len(output_names) > 4:
        raise NotImplementedError("NVGEMM scaled epilogues support 1-4 outputs")
    if len(output_names) > 1 and "D" not in output_names:
        raise NotImplementedError("NVGEMM scaled multi-store requires a D output")
    primary_index = output_names.index("D") if "D" in output_names else 0
    tensors = tuple(
        getattr(epilogue.tensors[name], attr)
        for index, name in enumerate(output_names)
        if index != primary_index
    )
    return tensors + (None,) * (3 - len(tensors)), len(output_names), primary_index


def _ones_alpha():
    """Cached per-device (4,)-ones alpha TensorWrapper (identity global scale).

    The kernel always takes an alpha arg so its signature is consistent across
    compile/run paths; when not fusing we pass ones (a no-op *1.0). Len is a
    multiple of 4 (CuTeDSL requires the operand's last dim divisible by 4).
    """
    from cutlass.operators.utils.tensor import TensorWrapper

    import torch

    dev = torch.cuda.current_device()
    tw = _ONES_ALPHA.get(dev)
    if tw is None:
        tw = TensorWrapper(torch.ones(4, dtype=torch.float32, device=f"cuda:{dev}"))
        _ONES_ALPHA[dev] = tw
    return tw


def _epilogue_op_scope(cute):
    def relu(x):
        return cute.math.max(x, cute.full_like(x, 0.0))

    def sigmoid(x):
        return 1.0 / (1.0 + cute.math.exp(-x))

    def gelu(x):
        return 0.5 * x * (1.0 + cute.math.erf(x * 0.7071067811865476))

    return {
        "erf": cute.math.erf,
        "exp": cute.math.exp,
        "gelu": gelu,
        "relu": relu,
        "sigmoid": sigmoid,
        "silu": lambda x: x * sigmoid(x),
        "tanh": cute.math.tanh,
    }


try:
    from ..dense_blockscaled_gemm_persistent import (  # pyrefly: ignore[missing-import]
        Sm100BlockScaledPersistentDenseGemmKernel as BlockScaledGemmKernelImpl,
    )
except ImportError:
    BlockScaledGemmKernelImpl = None  # type: ignore[misc, assignment]


class VendoredDenseBlockScaledGemmKernel(CuteDslOperator):
    """Wrapper for vendored dense blockscaled GEMM template for SM100 GPUs."""

    supported_args_type = GemmArguments
    designed_for_min_cc = 100

    def __init__(self, metadata: OperatorMetadata):
        super().__init__(metadata)

        self.sf_vec_size = metadata.operands.A.mode[-1]
        mma_tiler_mn = (metadata.design.tile_shape[0], metadata.design.tile_shape[1])
        cluster_shape_mn = (
            metadata.design.cluster_shape[0],
            metadata.design.cluster_shape[1],
        )

        import os

        self.impl = BlockScaledGemmKernelImpl(  # pyrefly: ignore[not-callable]
            self.sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch=os.environ.get("TORCHINDUCTOR_NVGEMM_PREFETCH", "0") == "1",
        )
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler_mn = mma_tiler_mn

    @staticmethod
    def _major_modes(args):
        """Extract major modes from arguments or operand metadata."""
        import cutlass.utils as utils
        from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode

        if args.A.stride[-2:].index(1) == 1:
            a_major_mode = (OperandMajorMode.K, "k")
        else:
            a_major_mode = (OperandMajorMode.MN, "m")

        if args.B.stride[-2:].index(1) == 0:
            b_major_mode = (OperandMajorMode.K, "k")
        else:
            b_major_mode = (OperandMajorMode.MN, "n")

        if args.out.stride[-2:].index(1) == 1:
            out_layout = (utils.LayoutEnum.ROW_MAJOR, "n")
        else:
            out_layout = (utils.LayoutEnum.COL_MAJOR, "m")

        return a_major_mode, b_major_mode, out_layout

    def _compile(
        self, args: GemmArguments, target_sm: TargetSm | None = None
    ) -> CompiledArtifact:
        import cutlass.cute as cute

        stream = cute.runtime.make_fake_stream()
        max_active_clusters = cutedsl_utils.mma.get_max_active_clusters(
            self.cluster_shape_mn
        )

        # Fused global scale: alpha is ALWAYS threaded as a trailing kernel arg
        # (ones when not fusing) so the kernel signature is consistent across all
        # compile/run paths -- a None alpha is not reliably dropped from the
        # runtime signature. args.alpha (a TensorWrapper, len multiple-of-4) is
        # applied elementwise in the epilogue; closure capture cannot read a
        # runtime tensor there.
        alpha = getattr(args, "alpha", None)
        if alpha is None:
            alpha = _ones_alpha()

        def epilogue_op(v):
            return v

        if getattr(args, "epilogue", None) is not None:
            epilogue_op = args.epilogue.epilogue_fn
            if isinstance(epilogue_op, str):
                fn_name = next(
                    node.name
                    for node in ast.parse(epilogue_op).body
                    if isinstance(node, ast.FunctionDef)
                )
                scope = _epilogue_op_scope(cute)
                exec(epilogue_op, scope)
                epilogue_op = scope[fn_name]
        epilogue = _EpilogueABI.from_args(args, "compile_time_tensor")
        local_reduce_out = getattr(args, "local_reduce_out", None)
        local_reduce_feeds_main = getattr(args, "local_reduce_feeds_main", False)
        if local_reduce_out is not None or local_reduce_feeds_main:
            return self.cute_compile(
                self.impl,
                args.A.tensor,
                args.B.tensor,
                args.A.scale.tensor,
                args.B.scale.tensor,
                args.out.tensor,
                max_active_clusters,
                stream,
                epilogue_op,
                self.metadata.operands.out.dtype,
                alpha,
                *epilogue.inputs,
                *epilogue.input_kinds,
                *epilogue.outputs,
                epilogue.output_count,
                epilogue.primary_output,
                (
                    local_reduce_out.compile_time_tensor
                    if local_reduce_out is not None
                    else None
                ),
                args.local_reduce_group,
                args.local_reduce_axis,
                args.local_reduce_type,
                args.local_reduce_source,
                local_reduce_feeds_main,
                target_sm=target_sm,
            )
        return self.cute_compile(
            self.impl,
            args.A.tensor,
            args.B.tensor,
            args.A.scale.tensor,
            args.B.scale.tensor,
            args.out.tensor,
            max_active_clusters,
            stream,
            epilogue_op,
            self.metadata.operands.out.dtype,
            alpha,
            *epilogue.inputs,
            *epilogue.input_kinds,
            *epilogue.outputs,
            epilogue.output_count,
            epilogue.primary_output,
            target_sm=target_sm,
        )

    def _run(
        self,
        args: GemmArguments,
        compiled_artifact: CompiledArtifact,
        stream,
        workspace=None,
    ) -> None:
        import torch

        stream = to_cuda_stream(stream)
        compiled_gemm = compiled_artifact.compiled_obj

        # TVM FFI needs a torch.cuda.Stream, not a raw int handle
        if isinstance(stream, int):
            stream = torch.cuda.ExternalStream(stream)

        # Runtime arg list must match _compile: alpha always trails stream.
        alpha = getattr(args, "alpha", None)
        if alpha is None:
            alpha = _ones_alpha()
        with torch.cuda.stream(stream):
            epilogue = _EpilogueABI.from_args(args, "runtime_tensor")

        local_reduce_out = getattr(args, "local_reduce_out", None)
        local_reduce_feeds_main = getattr(args, "local_reduce_feeds_main", False)
        if local_reduce_out is not None or local_reduce_feeds_main:
            self.cute_run(  # pyrefly: ignore[missing-attribute]
                compiled_gemm,
                args.A.tensor,
                args.B.tensor,
                args.A.scale.tensor,
                args.B.scale.tensor,
                args.out.tensor,
                stream,
                alpha,
                *epilogue.inputs,
                *epilogue.outputs,
                local_reduce_out.runtime_tensor
                if local_reduce_out is not None
                else None,
            )
            return

        self.cute_run(  # pyrefly: ignore[missing-attribute]
            compiled_gemm,
            args.A.tensor,
            args.B.tensor,
            args.A.scale.tensor,
            args.B.scale.tensor,
            args.out.tensor,
            stream,
            alpha,
            *epilogue.inputs,
            *epilogue.outputs,
            None,
        )

    def _supports(
        self, args: GemmArguments, target_sm: TargetSm | None = None
    ) -> Status:
        # Match the upstream block-scaled operator: validate scale-factor element
        # counts via ScaledOperand.numel_scale for the operand's mode/swizzle,
        # rather than re-inferring the layout (the old _infer_scale_swizzle_impl
        # check wrongly rejected valid NVFP4 args on the transposed B operand).
        from cutlass.operators.arguments import ScaledOperand

        local_reduce_out = getattr(args, "local_reduce_out", None)
        if local_reduce_out is not None or getattr(
            args, "local_reduce_feeds_main", False
        ):
            group = args.local_reduce_group
            axis = args.local_reduce_axis
            m, n = args.out.shape[-2:]
            selected_size = n if axis == 1 else m
            max_group = self.mma_tiler_mn[axis] if axis == 1 else 16
            if (
                axis not in (0, 1)
                or group <= 1
                or group > max_group
                or selected_size % group != 0
            ):
                return Status.fail(
                    "Grouped reduction requires a supported M- or N-axis group "
                    "that divides the selected dimension."
                )
            if local_reduce_out is not None:
                expected_shape = (m, n // group) if axis == 1 else (m // group, n)
                if local_reduce_out.shape != expected_shape:
                    return Status.fail(
                        "Grouped reduction output shape must be "
                        f"{expected_shape}; got {local_reduce_out.shape}."
                    )
                if local_reduce_out.dtype is not cutlass.Float32 or tuple(
                    local_reduce_out.stride
                ) != (expected_shape[1], 1):
                    return Status.fail(
                        "Grouped reduction output must be contiguous Float32."
                    )

        m, n = args.out.shape[-2:]
        k = args.A.shape[-1]
        L = args.A.shape[0] if len(args.A.shape) == 3 else 1

        expected_sfa = ScaledOperand.numel_scale((L, m, k), args.A.mode, args.A.swizzle)
        expected_sfb = ScaledOperand.numel_scale((L, n, k), args.B.mode, args.B.swizzle)
        if args.A.scale.numel() != expected_sfa:
            return Status.fail(
                f"Scale factor A must have {expected_sfa} elements for mat shape "
                f"{args.A.shape}; got {args.A.scale.numel()}."
            )
        if args.B.scale.numel() != expected_sfb:
            return Status.fail(
                f"Scale factor B must have {expected_sfb} elements for mat shape "
                f"{args.B.shape}; got {args.B.scale.numel()}."
            )
        return Status.success()

    @staticmethod
    def _is_valid_dtype_combo(ab_dtype, sf_dtype, sf_vec_size, out_dtype) -> bool:
        """Validate dtype/scale-factor/vec-size combinations.

        Matches constraints from the upstream kernel:
          MXF8: Float8E5M2/Float8E4M3FN + Float8E8M0FNU, sf_vec_size=32
          MXF4: Float4E2M1FN + Float8E8M0FNU, sf_vec_size=32
          NVF4: Float4E2M1FN + Float8E8M0FNU/Float8E4M3FN, sf_vec_size=16
        """
        import cutlass

        if ab_dtype not in {
            cutlass.Float4E2M1FN,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            return False
        if sf_vec_size not in {16, 32}:
            return False
        if sf_dtype not in {cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN}:
            return False
        # Float8E4M3FN as SF only valid with sf_vec_size=16 (NVF4)
        if sf_dtype == cutlass.Float8E4M3FN and sf_vec_size == 32:
            return False
        # Float8 AB types require sf_vec_size=32 (MXF8)
        if ab_dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN} and sf_vec_size == 16:
            return False
        if out_dtype not in {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            return False
        return True

    @staticmethod
    def _is_valid_layouts(ab_dtype, a_major, b_major) -> bool:
        """Float4E2M1FN (MXF4/NVF4) requires row-major (K-major) A and B."""
        import cutlass

        if ab_dtype is cutlass.Float4E2M1FN and not (a_major == "k" and b_major == "k"):
            return False
        return True

    @staticmethod
    def _valid_operands(operands: GemmOperandsMetadata, sf_vec_size: int) -> bool:
        import cutlass

        if operands.A.dtype != operands.B.dtype:
            return False
        if operands.A.scale.dtype != operands.B.scale.dtype:
            return False
        if operands.accumulator_type != cutlass.Float32:
            return False

        ab_dtype = operands.A.dtype
        sf_dtype = operands.A.scale.dtype
        out_dtype = operands.out.dtype

        if not VendoredDenseBlockScaledGemmKernel._is_valid_dtype_combo(
            ab_dtype, sf_dtype, sf_vec_size, out_dtype
        ):
            return False

        (_, a_major), (_, b_major), _ = VendoredDenseBlockScaledGemmKernel._major_modes(
            operands
        )
        if not VendoredDenseBlockScaledGemmKernel._is_valid_layouts(
            ab_dtype, a_major, b_major
        ):
            return False

        return True

    @staticmethod
    def _metadata_operand_combinations() -> Generator[GemmOperandsMetadata, None, None]:
        import cutlass

        ab_dtypes = [cutlass.Float8E5M2, cutlass.Float8E4M3FN, cutlass.Float4E2M1FN]
        out_dtypes = [
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        ]
        sf_dtypes = [cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN]
        scale_modes = [ScaleMode.Blockwise1x16, ScaleMode.Blockwise1x32]

        row_major = (0, 0, 1)
        col_major = (0, 1, 0)
        alignment_bytes = 16

        def major_str_a(stride):
            return "k" if stride == row_major else "m"

        def major_str_b(stride):
            return "n" if stride == row_major else "k"

        for ab_dtype, sf_dtype, scale_mode, out_dtype in itertools.product(
            ab_dtypes, sf_dtypes, scale_modes, out_dtypes
        ):
            sf_vec_size = scale_mode[-1]
            if not VendoredDenseBlockScaledGemmKernel._is_valid_dtype_combo(
                ab_dtype, sf_dtype, sf_vec_size, out_dtype
            ):
                continue

            for stride_A, stride_B, stride_out in itertools.product(
                [row_major, col_major], repeat=3
            ):
                a_major = major_str_a(stride_A)
                b_major = major_str_b(stride_B)

                if not VendoredDenseBlockScaledGemmKernel._is_valid_layouts(
                    ab_dtype, a_major, b_major
                ):
                    continue

                ab_div = alignment_bytes * 8 // ab_dtype.width
                out_div = alignment_bytes * 8 // out_dtype.width
                sf_div = alignment_bytes * 8 // sf_dtype.width

                yield GemmOperandsMetadata(
                    A=ScaledOperandConstraints(
                        quantized=DenseTensorConstraints(
                            dtype=ab_dtype,
                            stride=stride_A,
                            divisibility=ab_div,
                        ),
                        scale=DenseTensorConstraints(
                            dtype=sf_dtype,
                            stride=None,
                            divisibility=sf_div,
                        ),
                        mode=scale_mode,
                        swizzle=ScaleSwizzleMode.Swizzle32x4x4,
                    ),
                    B=ScaledOperandConstraints(
                        quantized=DenseTensorConstraints(
                            dtype=ab_dtype,
                            stride=stride_B,
                            divisibility=ab_div,
                        ),
                        scale=DenseTensorConstraints(
                            dtype=sf_dtype,
                            stride=None,
                            divisibility=sf_div,
                        ),
                        mode=scale_mode,
                        swizzle=ScaleSwizzleMode.Swizzle32x4x4,
                    ),
                    out=DenseTensorConstraints(
                        dtype=out_dtype,
                        stride=stride_out,
                        divisibility=out_div,
                    ),
                    accumulator_type=cutlass.Float32,
                )

    @classmethod
    def _valid_metadata(cls, metadata: OperatorMetadata) -> bool:
        scale_vec = metadata.operands.A.mode

        if len(scale_vec) > 1:
            for i in range(len(scale_vec) - 1):
                if scale_vec[i] != 1:
                    return False

        sf_vec_size = scale_vec[-1]
        if not VendoredDenseBlockScaledGemmKernel._valid_operands(
            metadata.operands, sf_vec_size
        ):
            return False

        design = metadata.design
        if not isinstance(design, Sm100DesignMetadata):
            return False

        cm, cn, _ = design.cluster_shape
        if cm <= 0 or cn <= 0:
            return False
        if cm * cn > 16:
            return False
        if cm & (cm - 1) != 0 or cn & (cn - 1) != 0:
            return False
        # SF multicast constraint: cluster dims <=4
        if cm > 4 or cn > 4:
            return False

        tile_m, tile_n, _ = design.tile_shape
        if tile_m not in [128, 256]:
            return False
        if tile_n not in [64, 128, 192, 256]:
            return False
        use_2cta = tile_m == 256
        if use_2cta and cm % 2 != 0:
            return False

        if metadata.epilogue is not None and "EFC" not in cls.__name__:
            return False

        return True

    @classmethod
    def _generate_operators(
        cls,
        metadata_filter: Callable[[OperatorMetadata], bool],
        epilogue_args=None,
        target_sm: TargetSm | None = None,
        args=None,
    ) -> list[VendoredDenseBlockScaledGemmKernel]:
        if target_sm is not None and target_sm.cc not in [100, 101, 103]:
            return []
        if epilogue_args is not None:
            return []

        design_params = {
            "mma_instruction_type": [BlackwellTcgen05Mma],
            "use_2cta_mma": [True],
            "tile_shape": [
                (M, N, 256) for M in [128, 256] for N in [64, 128, 192, 256]
            ],
            "cluster_shape": [(M, N, 1) for M in [1, 2, 4] for N in [1, 2, 4]],
            "use_tma_store": [True],
        }

        param_names = list(design_params.keys())
        param_values = [design_params[name] for name in param_names]

        operator_list = []

        for operands in cls._metadata_operand_combinations():
            # pyrefly: ignore[no-matching-overload]
            for values in itertools.product(*param_values):
                design = Sm100DesignMetadata(**dict(zip(param_names, values)))

                operator_name = (
                    f"inductor_vendored.{cls.__name__}_sm100_"
                    "{layout}_A{A}_B{B}_out{out}_SFA{SFA}_SFB{SFB}_"
                    "acc{acc}_scale{scale_mode}_swizzle{scale_swizzle}_"
                    "{num_cta}cta_cluster{cluster}_tile{tile}"
                    "{_tma_store}"
                ).format(
                    layout=strides_to_layout_string(
                        operands.A.stride,
                        operands.B.stride,
                        operands.out.stride,
                    ),
                    A=operands.A.dtype,
                    B=operands.B.dtype,
                    out=operands.out.dtype,
                    SFA=operands.A.scale.dtype,
                    SFB=operands.B.scale.dtype,
                    acc=operands.accumulator_type,
                    scale_mode=operands.A.mode,
                    scale_swizzle=operands.A.swizzle,
                    num_cta="2" if design.use_2cta_mma else "1",
                    cluster=tuple_to_string(design.cluster_shape),
                    tile=tuple_to_string(design.tile_shape),
                    _tma_store="_tma_store" if design.use_tma_store else "",
                )

                metadata = OperatorMetadata(
                    operands=operands,
                    design=design,
                    operator_name=operator_name,
                    operator_class=cls,
                    supported_targets=TargetSm.get_supported_targets(design, operands),
                    epilogue=None,
                )

                if cls._valid_metadata(metadata):
                    if metadata_filter is None or metadata_filter(metadata):
                        operator_list.append(cls(metadata))

        log.debug(
            "Generated %d DenseBlockScaledGemmKernel configurations",
            len(operator_list),
        )
        return operator_list


class VendoredDenseBlockScaledGemmEFC(VendoredDenseBlockScaledGemmKernel):
    """Block-scaled provider variant using the kernel's unary epilogue hook."""

    supported_args_type = GemmArguments
    designed_for_min_cc = 100


# Only register if kernel implementation is available
if BlockScaledGemmKernelImpl is not None:
    cutlass.operators.providers.cutedsl.CuTeDSLProvider.register(
        VendoredDenseBlockScaledGemmKernel
    )
    cutlass.operators.providers.cutedsl.CuTeDSLProvider.register(
        VendoredDenseBlockScaledGemmEFC
    )
<|endoftext|>from __future__ import annotations

import copy
import logging
from functools import lru_cache
from typing import Any, TYPE_CHECKING

import torch
from torch._inductor import config
from torch._inductor.choices import InductorChoices
from torch._inductor.heuristics.template.params import DictKernelTemplateParams
from torch._inductor.kernel_template_choice import KernelTemplateChoice


log = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Generator

    from torch._inductor.codegen.common import KernelTemplate
    from torch._inductor.kernel_inputs import KernelInputs
    from torch._inductor.select_algorithm import ExternKernelChoice


class LookupTableChoices(InductorChoices):
    """
    InductorChoices subclass that uses lookup table when available, otherwise falls back to parent.
    All lookup functionality is contained within this class and can be customized by overriding methods.
    """

    def _get_lookup_table(self) -> dict[str, list[dict[str, Any]]]:
        """
        Get the template lookup table from config.
        Override this method to use custom lookup table sources (database, API, etc.).
        """
        if not torch.cuda.is_available() or config.lookup_table.table is None:
            return {}
        return config.lookup_table.table

    @staticmethod
    @lru_cache
    def _get_device_key(device: torch.device) -> str | None:
        """
        Generate a device key for lookup table indexing.
        For CPU devices, returns None.
        For CUDA devices, returns the props.gcnArchName string.
        """
        if device.type != "cuda":
            # only cuda devices are supported, this indicates that the system is not in use
            # for this device
            return None

        # Get CUDA device properties
        props = torch.cuda.get_device_properties(device.index)
        return props.gcnArchName

    @staticmethod
    def _generate_kernel_inputs_key(kernel_inputs: KernelInputs) -> str:
        """
        Generate a key based on input node properties and scalars.
        The key includes dtype, size, and stride information for each input node,
        plus scalar values as key=value pairs separated by & signs.
        """
        # Get node information using existing methods
        dtypes = kernel_inputs.dtypes()
        shapes = kernel_inputs.shapes_hinted()
        strides = kernel_inputs.strides_hinted()

        # Create tuple of (dtype, shape_list, stride_list) for each node
        node_info = tuple(
            (dtype, list(shape), list(stride))
            for dtype, shape, stride in zip(dtypes, shapes, strides)
        )

        # Create base key from node information
        fmt_key = str(node_info)
        # Add scalar information if present
        if kernel_inputs._scalars:
            # Sort scalars for consistent key generation and join with &
            scalar_parts = [
                f"{key}={value}"
                for key, value in sorted(kernel_inputs._scalars.items())
            ]
            scalars_key = "&".join(scalar_parts)
            fmt_key = f"{fmt_key}+{scalars_key}"

        return f"{fmt_key}"

    def make_lookup_key(
        self, kernel_inputs: KernelInputs, op_name: str, include_device: bool = False
    ) -> str | None:
        """
        Create a flattened lookup key from kernel inputs and operation name.
        Override this method to customize key generation.

        Args:
            kernel_inputs: KernelInputs object containing input nodes and scalars
            op_name: Operation name (e.g., "mm", "addmm")
            include_device: Whether to include device key in the generated key

        Returns:
            A string key combining device (optional), operation, and input information
        """
        device = kernel_inputs.device()
        dev_key = self._get_device_key(device)
        if dev_key is None:
            # The system does not run when dev_key is None, regardless of
            # whether include_device is True or False
            return None
        if not include_device:
            dev_key = None

        # Generate input key using our staticmethod
        input_key = self._generate_kernel_inputs_key(kernel_inputs)

        # Create the flattened lookup key
        if dev_key is not None:
            key_parts = [dev_key, input_key, op_name]
        else:
            key_parts = [input_key, op_name]

        return "+".join(key_parts)

    def make_lookup_key_variants(
        self, kernel_inputs: KernelInputs, op_name: str
    ) -> tuple[str | None, str | None]:
        """
        Generate both device-specific and device-agnostic lookup keys.
        Override this method to customize key variant generation.

        Args:
            kernel_inputs: KernelInputs object containing input nodes and scalars
            op_name: Operation name (e.g., "mm", "addmm")

        Returns:
            Tuple of (device_key, device_agnostic_key). Either may be None if generation fails.
        """
        device_key = self.make_lookup_key(kernel_inputs, op_name, include_device=True)
        device_agnostic_key = self.make_lookup_key(
            kernel_inputs, op_name, include_device=False
        )

        return device_key, device_agnostic_key

    @staticmethod
    def _entry_is_valid(
        cfg: dict[str, Any],
        template_id: str,
        template_hash_map: dict[str, str | None] | None,
    ) -> bool:
        """
        Check if a config entry is valid based on template hash validation.

        Args:
            cfg: Configuration dictionary that may contain a template_hash field
            template_id: The template identifier
            template_hash_map: Optional mapping from template_uid to src_hash for validation

        Returns:
            True if the config is valid and should be kept, False if it should be filtered out
        """
        # If hash checking is disabled or no hash map provided, keep the config
        if not config.lookup_table.check_src_hash or not template_hash_map:
            return True

        template_hash = template_hash_map.get(template_id)
        config_hash = cfg.get("template_hash")

        # Both hashes present - validate they match
        if template_hash is not None and config_hash is not None:
            if config_hash != template_hash:
                log.warning(
                    "Hash validation failed for template '%s': config_hash='%s' != template_hash='%s'. "
                    "Template code may have changed. Filtering out config: %s",
                    template_id,
                    config_hash,
                    template_hash,
                    {k: v for k, v in cfg.items() if k != "template_hash"},
                )
                return False
            else:
                log.debug(
                    "Hash validation passed for template '%s': hash='%s'",
                    template_id,
                    template_hash,
                )
                return True
        # Config has no hash - keep it
        elif config_hash is None:
            log.debug(
                "Config for template '%s' has no hash - keeping it (template_hash='%s')",
                template_id,
                template_hash,
            )
            return True
        # Template has no hash - keep config
        else:
            log.debug(
                "Template '%s' has no src_hash - keeping config with hash '%s'",
                template_id,
                config_hash,
            )
            return True

    def lookup_template_configs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
        template_uids: list[str],
        template_hash_map: dict[str, str | None] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Unified function to look up template configurations for multiple templates.
        Override this method to customize lookup logic.

        Args:
            kernel_inputs: KernelInputs object containing input nodes and scalars
            op_name: Operation name (e.g., "mm", "addmm")
            template_uids: List of template identifiers (e.g., ["mm", "tma", "decompose_k"])
            template_hash_map: Optional mapping from template_uid to src_hash for validation

        Returns:
            {}: No lookup table in use, or no matches found for any template
            {"template_uid1": [config1, config2], ...}: Matches found, filtered configurations
        """
        lookup_table = self._get_lookup_table()
        if not lookup_table:
            log.debug("Lookup table: no table configured or CUDA unavailable")
            return {}

        # Try both key variants: device-specific first, then device-agnostic
        # If both exist, device-specific takes priority
        device_key, device_agnostic_key = self.make_lookup_key_variants(
            kernel_inputs, op_name
        )

        config_list = []

        for key_type, key in [
            ("device-specific", device_key),
            ("device-agnostic", device_agnostic_key),
        ]:
            if key is not None:
                config_list = lookup_table.get(key, [])
                if config_list:
                    log.debug(
                        "Lookup table: found %d configs using %s key '%s' for %s",
                        len(config_list),
                        key_type,
                        key,
                        op_name,
                    )
                    break
        else:
            log.debug(
                "Lookup table: no match for %s (tried keys: %s, %s) (table has %d keys)",
                op_name,
                device_key,
                device_agnostic_key,
                len(lookup_table),
            )
            return {}

        log.debug(
            "Lookup table: found %d configs for %s templates %s",
            len(config_list),
            op_name,
            template_uids,
        )
        # Group configs by template_id
        configs_by_template: dict[str, list[dict[str, Any]]] = {}
        for cfg in config_list:
            if not isinstance(cfg, dict):
                raise ValueError(
                    f"Config for {op_name} operation is not a dictionary: {cfg}"
                )
            if "template_id" not in cfg:
                raise ValueError(
                    f"Config for {op_name} operation missing required 'template_id' field: {cfg}"
                )

            template_id = cfg["template_id"]
            if template_id in template_uids:
                if template_id not in configs_by_template:
                    configs_by_template[template_id] = []
                configs_by_template[template_id].append(cfg)

        # Check template hashes and clean up template_id field
        result = {}
        for template_id, matching_configs in configs_by_template.items():
            filtered_configs = []
            for cfg in matching_configs:
                # Check template hash using helper function
                if not self._entry_is_valid(cfg, template_id, template_hash_map):
                    continue

                # Return a copy of the config, as we don't want to modify the original
                cconfig = copy.deepcopy(cfg)
                # Lastly, we have to throw out the template_id, as it's not a valid kwarg
                # and just used to identify which template the entry belongs to
                del cconfig["template_id"]
                # Similarly, the template_hash is not a valid kwarg
                cconfig.pop("template_hash", None)
                filtered_configs.append(cconfig)

            if filtered_configs:
                result[template_id] = filtered_configs

        return result

    def _finalize_template_configs(
        self,
        template_choices: dict[str, Generator[KernelTemplateChoice, None, None]],
        kernel_inputs: KernelInputs,
        templates: list[KernelTemplate | ExternKernelChoice],
        op_name: str,
        kwarg_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> list[KernelTemplateChoice]:
        """Check lookup table for hits, use those if found, otherwise fall back to parent."""
        # 1. Collect template src_hashes for validation
        template_uids = [template.uid for template in templates]
        template_hash_map = {}
        for template in templates:
            src_hash = getattr(template, "src_hash", None)
            template_hash_map[template.uid] = src_hash

        log.debug(
            "Choices: attempting lookup for %s with %d templates",
            op_name,
            len(template_uids),
        )

        # 2. Single batch lookup for all templates
        lookup_results = self.lookup_template_configs(
            kernel_inputs, op_name, template_uids, template_hash_map
        )

        # 3. Early exit if no lookup table or no matches
        if not lookup_results:  # Empty dict
            log.info("LookupChoices: lookup miss for %s, using fallback", op_name)
            return self._fallback(
                template_choices,
                kernel_inputs,
                templates,
                op_name,
                kwarg_overrides,
            )

        log.info(
            "LookupChoices: lookup hit for %s - found %d/%d templates: %s",
            op_name,
            len(lookup_results),
            len(template_uids),
            list(lookup_results.keys()),
        )

        # 4. Create KTCs only for templates with lookup entries
        return self._create_lookup_choices(
            lookup_results, templates, kernel_inputs, op_name
        )

    def _fallback(
        self,
        template_choices: dict[str, Generator[KernelTemplateChoice, None, None]],
        kernel_inputs: KernelInputs,
        templates: list[KernelTemplate | ExternKernelChoice],
        op_name: str,
        kwarg_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> list[KernelTemplateChoice]:
        """Fallback to parent if no lookup table or no matches."""
        # NOTE: this is broken out, so that subclasses are able to override this
        # to handle explicitly the situations where the lookup table had a miss vs
        # overriding the entire logic
        return super()._finalize_template_configs(
            template_choices,
            kernel_inputs,
            templates,
            op_name,
            kwarg_overrides,
        )

    def _create_lookup_choices(
        self,
        lookup_results: dict[str, list[dict[str, Any]]],
        templates: list[KernelTemplate | ExternKernelChoice],
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> list[KernelTemplateChoice]:
        """Create KernelTemplateChoice objects from lookup results using parent's get_ktc method."""
        templates_by_uid = {template.uid: template for template in templates}
        lookup_choices: list[KernelTemplateChoice] = []

        for template_uid, configs in lookup_results.items():
            template = templates_by_uid[template_uid]

            # Use parent's get_ktc method to get a generator, then get the first base KTC
            ktc_generator = self.get_ktc(kernel_inputs, template, op_name)

            try:
                base_ktc = next(ktc_generator)
            except StopIteration:
                # No configs from heuristic, skip this template
                continue

            # For each lookup config, create a KTC with the override kwargs
            # Start from base params (which include derived options like OUT_DTYPE)
            # and let the lookup config override them
            base_kwargs = base_ktc.params.to_kwargs()
            for c in configs:
                merged = {**base_kwargs, **c}
                lookup_ktc = KernelTemplateChoice(
                    template=base_ktc.template,
                    params=DictKernelTemplateParams(merged),
                    extra_kwargs=base_ktc.extra_kwargs,
                    layout=base_ktc.layout,
                    inputs=base_ktc.inputs,
                )
                lookup_choices.append(lookup_ktc)

        return lookup_choices
<|endoftext|>"""
Template lookup table system for PyTorch Inductor.

This package provides functionality for:
- Loading pre-configured template choices from lookup tables
- Managing template configurations and choices

All functionality is contained within the LookupTableChoices class.
You can customize any aspect by subclassing LookupTableChoices and overriding methods.

Usage:
    # Basic usage
    choices = LookupTableChoices()
    V.set_choices_handler(choices)

    # Custom usage
    class MyCustomChoices(LookupTableChoices):
        def _get_lookup_table(self):
            return my_custom_table

        def make_lookup_key(self, kernel_inputs, op_name, include_device=False):
            return f"custom_{op_name}_{hash(str(kernel_inputs))}"

    V.set_choices_handler(MyCustomChoices())
"""

from .choices import LookupTableChoices


__all__ = [
    "LookupTableChoices",
]
<|endoftext|>import json
import logging
import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch._inductor.analysis.device_info import DeviceInfo, lookup_device_info
from torch._inductor.utils import tabulate_2d, zip_dicts
from torch.utils import _pytree as pytree
from torch.utils._ordered_set import OrderedSet
from torch.utils.flop_counter import flop_registry


log = logging.getLogger(__name__)


ATEN_PREFIX = "aten::"


@dataclass
class ProfileEvent:
    category: str
    key: str
    self_device_time_ms: float
    # the benchmark is run multiple times and we average the count across all the
    # runs. It should be an integer but define a float just in case.
    count: float


# adapters convert the json trace into a format that works with flops_counter
ArgsType = tuple[tuple[Any, ...], dict[Any, Any]]
AdapterType = Callable[[tuple[Any, ...], tuple[Any, ...]], ArgsType]
adapters_map: dict[str, AdapterType] = {}


def parse_list(lst: str) -> list[int]:
    lst = lst.replace("[", "").replace("]", "")
    substrings = lst.split(",")

    return [int(substring.strip()) for substring in substrings]


def register_adapter(
    aten: str | list[str],
) -> Callable[
    [AdapterType],
    AdapterType,
]:
    def decorator(func: AdapterType) -> AdapterType:
        # pyrefly: ignore [unknown-name]
        global _adapters_map

        if isinstance(aten, str):
            adapters_map[aten] = func
        else:
            for at in aten:
                adapters_map[at] = func
        return func

    return decorator


@register_adapter(["_slow_conv2d_forward"])
def _slow_conv2d_adapter(
    shapes: tuple[Any, ...], concrete: tuple[Any, ...]
) -> tuple[tuple[Any], dict[Any, Any]]:
    tmp = list(shapes)
    tmp.append(False)
    tmp2 = list(concrete)
    if len(tmp2) < 5:
        raise ParseException("slow conv2d has less than 5 concrete inputs")
    tmp2[3] = tmp2[4]
    return conv_adapter(tuple(tmp), tuple(tmp2))


@register_adapter(
    ["convolution", "_convolution", "cudnn_convolution", "convolution_overrideable"]
)
def conv_adapter(
    shapes: tuple[Any, ...], concrete: tuple[Any, ...]
) -> tuple[tuple[Any], dict[Any, Any]]:
    tmp = list(shapes)
    if len(tmp) == 4:
        transposed = False
    elif len(tmp) > 6:
        transposed = bool(tmp[6])
        tmp[6] = transposed
    else:
        raise ParseException(f"Convolution has the wrong number of inputs: {len(tmp)}")

    kwargs: dict[Any, Any] = {}
    if not transposed:
        # calculate output shape if not transposed.
        def conv_out_dims(x: int, kernel: int, stride: int) -> int:
            return (x - kernel) // stride + 1

        stride = parse_list(concrete[3])
        inp = shapes[0]
        w = shapes[1]
        out_x_y = [conv_out_dims(*args) for args in zip(inp[2:], w[2:], stride)]
        out = [inp[0], w[0]] + out_x_y  # we only need the xy values
        kwargs["out_val"] = out

    return tuple(tmp), kwargs


def default_adapter(
    shapes: tuple[Any], concrete: tuple[Any]
) -> tuple[tuple[Any], dict[Any, Any]]:
    return shapes, {}


@register_adapter("addmm")
def addmm_adapter(
    shapes: tuple[Any], concrete: tuple[Any]
) -> tuple[tuple[Any], dict[Any, Any]]:
    tmp = list(shapes)[:3]
    return tuple(tmp), {}


@register_adapter("bmm")
def bmm_adapter(
    shapes: tuple[Any], concrete: tuple[Any]
) -> tuple[tuple[Any], dict[Any, Any]]:
    tmp = list(shapes)
    return tuple(tmp[:2]), {}


@register_adapter("baddbmm")
def baddbmm_adapter(
    shapes: tuple[Any], concrete: tuple[Any]
) -> tuple[tuple[Any], dict[Any, Any]]:
    tmp = list(shapes)[:3]
    return tuple(tmp), {}


@register_adapter("mm")
def mm_adapter(
    shapes: tuple[Any], concrete: tuple[Any]
) -> tuple[tuple[Any], dict[Any, Any]]:
    return shapes, {}


def _parse_kernel_name(name: str) -> str | None:
    """
    parse the name of the kernel from the event name.
    """
    if name.startswith(ATEN_PREFIX):
        return name[len(ATEN_PREFIX) :]
    elif "conv" in name:
        return "convolution"
    elif "addmm" in name:
        return "addmm"
    elif "bmm" in name:
        return "bmm"
    elif "baddbmm" in name:
        return "baddbmm"
    elif "_mm" in name:
        return "mm"
    else:
        return None


def _calculate_flops(event: dict[str, Any]) -> int:
    """
    This function has to parse the kernel name, which is error prone. There doesn't seem to be another solution that
    will support all the different backends that can generate kernels, so make sure to update this function when new
    ops and backends are desired.
    """
    name = event["name"]
    if "kernel_flop" in event["args"] and event["args"]["kernel_flop"] != 0:
        return event["args"]["kernel_flop"]
    op_name = _parse_kernel_name(name)
    if op_name is None:
        return 0

    op_obj = getattr(torch.ops.aten, op_name, None)
    if op_obj is None or op_obj not in flop_registry:
        return 0

    flop_function = flop_registry[op_obj]

    if "Input Dims" not in event["args"] or "Concrete Inputs" not in event["args"]:
        return 0
    input_shapes = event["args"]["Input Dims"]
    concrete = event["args"]["Concrete Inputs"]
    if op_name in adapters_map:
        try:
            args, kwargs = adapters_map[op_name](input_shapes, concrete)
        except ParseException as e:
            msg = f"Failed to parse {op_name} with {e}"
            log.warning(msg)
            return 0
    else:
        try:
            args, kwargs = default_adapter(input_shapes, concrete)
        except ParseException as e:
            msg = f"Failed to parse {op_name} with {e}"
            log.warning(msg)
            return 0
    return flop_function(*args, **kwargs)


def _get_size_from_string(type_string: str) -> int:
    if not hasattr(torch, type_string):
        return 1
    else:
        return getattr(torch, type_string).itemsize


def _default_estimate_gb(event: dict[str, Any]) -> float:
    sizes_and_types = zip(event["args"]["Input Dims"], event["args"]["Input type"])
    bw = 0
    for size, typ in sizes_and_types:
        isize = _get_size_from_string(typ)
        bw += isize * math.prod(pytree.tree_flatten(size)[0])
    return bw / 1e9


def _estimate_gb(event: dict[str, Any]) -> float:
    """
    Our best effort to estimate the gb, should be refactored soon with MemoryCounter.
    """
    name = event["name"]
    if "kernel_num_gb" in event["args"] and event["args"]["kernel_num_gb"] != 0:
        return event["args"]["kernel_num_gb"]
    if "Input type" not in event["args"] or "Input Dims" not in event["args"]:
        return 0
    op_name = _parse_kernel_name(name)
    if op_name is None:
        return _default_estimate_gb(event)

    op_obj = getattr(torch.ops.aten, op_name, None)
    if op_obj is None:
        return _default_estimate_gb(event)

    if "Input Dims" not in event["args"] or "Concrete Inputs" not in event["args"]:
        return _default_estimate_gb(event)
    input_shapes = event["args"]["Input Dims"]

    # NOTE these will be refactored into a similar object to FlopCounter soon
    def mm_formula(M: int, N: int, K: int, size: int) -> int:
        return 2 * (M * K + N * K + M * N) * size

    if op_name == "addmm":
        add_in_size = math.prod(pytree.tree_flatten(input_shapes[0])[0])
        add_type_size = _get_size_from_string(event["args"]["Input type"][0])
        M = input_shapes[1][0]
        N = input_shapes[1][1]
        if input_shapes[1][1] != input_shapes[2][0]:
            raise AssertionError(
                f"Shape mismatch for addmm: {input_shapes[1][1]} != {input_shapes[2][0]}"
            )
        K = input_shapes[2][1]
        mul_type_size = _get_size_from_string(event["args"]["Input type"][1])
        return (mm_formula(M, N, K, mul_type_size) + add_in_size * add_type_size) / 1e9
    elif op_name == "mm":
        M = input_shapes[0][0]
        N = input_shapes[0][1]
        if input_shapes[0][1] != input_shapes[1][0]:
            raise AssertionError(
                f"Shape mismatch for mm: {input_shapes[0][1]} != {input_shapes[1][0]}"
            )
        K = input_shapes[1][1]
        type_size = _get_size_from_string(event["args"]["Input type"][0])
        return mm_formula(M, N, K, type_size) / 1e9
    elif op_name == "baddbmm":
        add_in_size = math.prod(pytree.tree_flatten(input_shapes[0])[0])
        add_type_size = _get_size_from_string(event["args"]["Input type"][0])
        B = input_shapes[0][0]
        M = input_shapes[1][1]
        N = input_shapes[1][2]
        K = input_shapes[2][2]
        mul_type_size = _get_size_from_string(event["args"]["Input type"][1])
        return (
            B * mm_formula(M, N, K, mul_type_size) + add_in_size * add_type_size
        ) / 1e9
    elif op_name == "bmm":
        add_in_size = math.prod(pytree.tree_flatten(input_shapes[0])[0])
        add_type_size = _get_size_from_string(event["args"]["Input type"][0])
        B = input_shapes[0][0]
        M = input_shapes[0][1]
        N = input_shapes[0][2]
        K = input_shapes[1][2]
        mul_type_size = _get_size_from_string(event["args"]["Input type"][1])
        return (
            B * mm_formula(M, N, K, mul_type_size) + add_in_size * add_type_size
        ) / 1e9
    elif op_name in [
        "convolution",
        "_convolution",
        "cudnn_convolution",
        "_slow_conv2d_forward",
    ]:
        concrete = event["args"]["Concrete Inputs"]

        def conv_out_dim(x: int, kernel: int, stride: int) -> int:
            return (x - kernel) // stride + 1

        stride = parse_list(
            concrete[3] if op_name != "_slow_conv2d_forward" else concrete[4]
        )
        inp = input_shapes[0]
        w = input_shapes[1]
        out_x_y = [conv_out_dim(*args) for args in zip(inp[2:], w[2:], stride)]
        out = [inp[0], w[0]] + out_x_y
        # each output element reads in * w * w chunk
        input_reads = out[0] * out[1] * out[2] * out[3] * inp[1] * w[2] * w[3]
        # Assume weights are in cache, so only read once
        weight_reads = w[0] * w[1] * w[2] * w[3]
        return (input_reads + weight_reads) / 1e9

    return _default_estimate_gb(event)


def _create_extern_mapping(
    data: dict[str, Any],
) -> defaultdict[int, list[dict[str, Any]]]:
    """
    compute a mapping from external ids to non kernels, which contain the information we need to estimate flops etc
    """
    extern_mapping: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in data["traceEvents"]:
        if (
            "args" not in event
            or "External id" not in event["args"]
            or event["cat"] != "cpu_op"
        ):
            continue
        if len(extern_mapping[event["args"]["External id"]]) > 0:
            raise ParseException("duplicate external id in event")
        extern_mapping[event["args"]["External id"]].append(event)
    return extern_mapping


def _augment_trace_helper(data: dict[str, Any]) -> dict[str, Any]:
    extern_mapping = _create_extern_mapping(data)

    for event in data["traceEvents"]:
        if "cat" not in event or event["cat"] != "kernel":
            continue
        if "args" not in event:
            raise ParseException(f"kernel has no args: {event}")
        if "External id" not in event["args"]:
            event_str = f"kernel has no External id: {event}"
            log.info(event_str)
            continue

        external_op = extern_mapping[event["args"]["External id"]][0]
        flops = _calculate_flops(external_op)
        if flops == 0:
            flops = _calculate_flops(event)
        external_op["args"]["kernel_flop"] = flops
        external_op["args"]["kernel_num_gb"] = _estimate_gb(external_op)
        event["args"]["kernel_flop"] = external_op["args"]["kernel_flop"]
        event["args"]["kernel_num_gb"] = external_op["args"]["kernel_num_gb"]
    return data


_dtype_map: dict[str, torch.dtype] = {
    "float": torch.float,
    "float32": torch.float,
    "double": torch.double,
    "float64": torch.double,
    "int": torch.int,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int,
    "long": torch.long,
    "long int": torch.long,
    "signed char": torch.int8,
    "unsigned char": torch.uint8,
    "bool": torch.bool,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "c10::BFloat16": torch.bfloat16,
    "c10::Half": torch.float16,
    "c10::complex<float>": torch.complex64,
    "c10::complex<double>": torch.complex128,
}


@dataclass(frozen=True)
class KernelStats:
    flops: int
    bw: float
    latency: float  # us
    achieved_flops: float
    achieved_bandwidth: float


KernelNameMap = defaultdict[str, OrderedSet[KernelStats]]


@dataclass(frozen=False)
class Device:
    name: str
    index: int
    info: DeviceInfo | None
    stats: KernelNameMap

    def __repr__(self) -> str:
        return f"Device({self.name}, {self.index}): {self.info}"


DeviceMap = dict[int, Device]
Table = tuple[list[str], dict[str, list[str]]]


class JsonProfile:
    _devices: DeviceMap

    def __init__(
        self,
        path: str,
        benchmark_name: str | None = None,
        dtype: torch.dtype | str | None = None,
    ):
        """
        Convenience class for running common operations on chrome/perfetto json traces.
        """
        self.path = path
        with open(path) as f:
            self.data = json.load(f)
            self.events = self.data["traceEvents"]
        self.benchmark_name = benchmark_name
        if dtype is None:
            self.dtype = None
        elif isinstance(dtype, torch.dtype):
            # pyrefly: ignore [bad-assignment]
            self.dtype = dtype
        else:
            # pyrefly: ignore [bad-assignment]
            self.dtype = _dtype_map.get(dtype)
        self._create_devices()

    def convert_dtype(self, event: dict[str, Any]) -> torch.dtype | None:
        """
        Each op has a list of dtypes for each input arg. We need to convert these into a single dtype for flop estimation.
        Issues:
         - converting the strings to concrete torch.dtypes
         - What if we have float32, float, float16 all in the inputs? Our choice is to use the largest buffer dtype.
        """

        if (
            "Input Dims" not in event["args"]
            or "Input type" not in event["args"]
            or "Concrete Inputs" not in event["args"]
        ):
            if "bfloat16" in event["name"]:
                return torch.bfloat16
            elif "float16" in event["name"]:
                return torch.float16
            else:
                return None

        input_sizes = event["args"]["Input Dims"]
        input_types = event["args"]["Input type"]
        concrete_inputs = event["args"]["Concrete Inputs"]
        if len(input_sizes) != len(input_types):
            raise AssertionError(
                f"input_sizes and input_types length mismatch: "
                f"{len(input_sizes)} != {len(input_types)}"
            )
        if len(input_types) != len(concrete_inputs):
            raise AssertionError(
                f"input_types and concrete_inputs length mismatch: "
                f"{len(input_types)} != {len(concrete_inputs)}"
            )

        if len(input_sizes) == 0:
            raise RuntimeError("Empty input_sizes and input_types")

        biggest_size = 0
        biggest_index = 0
        for i in range(len(input_sizes)):
            if concrete_inputs[i] != "":
                # concrete inputs are usually small tensors, so we can just skip
                continue
            my_size = input_sizes[i]
            total_size = sum(parse_list(my_size))
            if total_size > biggest_size:
                biggest_size = total_size
                biggest_index = i
        ret_type = input_types[biggest_index]
        if ret_type in _dtype_map:
            return _dtype_map[ret_type]
        raise RuntimeError(f"Unknown type: {ret_type}. Please add to _dtype_map.")

    def _create_devices(self) -> None:
        self._devices = {}
        for dev in self.data["deviceProperties"]:
            name = dev["name"]
            device_info = lookup_device_info(name)

            if device_info is None:
                log.info(
                    "Unsupported device in profile: %s, please consider contributing to _device_mapping.",
                    name,
                )
            self._devices[dev["id"]] = Device(
                name, dev["id"], device_info, defaultdict(OrderedSet)
            )

    def calculate_flops(self, event: dict[str, Any]) -> int:
        return _calculate_flops(event)

    def estimate_gb(self, event: dict[str, Any]) -> float:
        return _estimate_gb(event)

    def augment_trace(self) -> None:
        self.data = _augment_trace_helper(self.data)

    def _compute_stats(self) -> None:
        """populates the name -> stats map"""
        for event in self.events:
            if "cat" not in event or "args" not in event or event["cat"] != "kernel":
                continue
            if "device" not in event["args"]:
                continue
            dev_tmp = event["args"]["device"]
            if dev_tmp not in self._devices:
                continue
            dev = self._devices[event["args"]["device"]]

            dur = event["dur"]  # us
            if "kernel_flop" in event["args"]:
                if dur == 0:
                    raise AssertionError(
                        "Event duration is 0 for kernel_flop calculation"
                    )
                # 1,000,000us/s * flop / us
                op_flops = event["args"]["kernel_flop"] / (dur / 1e6)
            else:
                op_flops = 0

            if "kernel_num_gb" in event["args"]:
                if dur == 0:
                    raise AssertionError(
                        "Event duration is 0 for kernel_num_gb calculation"
                    )
                # 1,000,000us/s * gb  = gb/s
                op_gbps = event["args"]["kernel_num_gb"] / (dur / 1e6)
            else:
                op_gbps = 0

            if dev.info is not None:
                dtype = self.convert_dtype(event) or self.dtype
                if dtype is None:
                    raise RuntimeError(
                        "dtype is not found on tensor and default dtype is not set"
                    )
                achieved_flops = 100 * op_flops / (1e12 * dev.info.tops[dtype])
                achieved_bandwidth = 100 * op_gbps / dev.info.dram_bw_gbs
            else:
                achieved_flops = 0
                achieved_bandwidth = 0

            if "name" not in event["args"]:
                continue
            dev.stats[event["name"]].add(
                KernelStats(
                    flops=op_flops,
                    bw=op_gbps,
                    latency=dur,
                    achieved_bandwidth=achieved_bandwidth,
                    achieved_flops=achieved_flops,
                )
            )

    def _create_single_table(self, dev: Device) -> Table:
        """Create a table with the devices mapped to indices."""
        headers = [
            "Kernel Name",
            "Kernel Count",
            "FLOPS",
            "Kernel Reads (GB)",
            "Dur (us)",
            "Achieved FLOPS %",
            "Achieved Bandwidth %",
        ]
        rows: dict[str, list[str]] = {}

        def safe_div_format(x: float, y: float) -> str:
            if y == 0:
                return "0.0"
            return f"{x / y:.4f}"

        for kernel_name, stats_set in dev.stats.items():
            ker_count = 0
            flops = 0
            flops_count = 0
            achieved_flops = 0.0
            bw = 0.0
            bw_count = 0
            achieved_bandwidth = 0.0
            latency = 0.0
            for stats in stats_set:
                if stats.flops != 0:
                    flops += stats.flops
                    achieved_flops += stats.achieved_flops
                    flops_count += 1
                if stats.bw != 0:
                    bw += stats.bw
                    achieved_bandwidth += stats.achieved_bandwidth
                    bw_count += 1
                latency += stats.latency
                ker_count += 1
            if ker_count == 0:
                raise AssertionError("ker_count is 0 when computing kernel stats")
            rows[kernel_name] = [
                str(ker_count),
                safe_div_format(flops, flops_count),
                safe_div_format(bw, bw_count),
                safe_div_format(latency, ker_count),
                safe_div_format(achieved_flops, flops_count),
                safe_div_format(achieved_bandwidth, bw_count),
            ]

        return headers, rows

    def _create_tables(self, devs: DeviceMap) -> dict[int, Table]:
        return {idx: self._create_single_table(dev) for idx, dev in devs.items()}

    def _combine_tables(
        self, table1: Table, table1_name: str, table2: Table, table2_name: str
    ) -> Table:
        new_headers = (
            ["Kernel Name"]
            + [f"{table1_name} {head}" for head in table1[0][1:]]
            + [f"{table2_name} {head}" for head in table2[0][1:]]
        )
        t1_length = len(table1[0][1:])
        t2_length = len(table2[0][1:])
        new_rows = {}

        for key, row1, row2 in zip_dicts(
            table1[1],
            table2[1],
            d1_default=["Empty"] * t1_length,
            d2_default=["Empty"] * t2_length,
        ):
            if row1 is None:
                raise AssertionError("row1 must not be None in table combination")
            if row2 is None:
                raise AssertionError("row2 must not be None in table combination")
            new_rows[key] = row1 + row2
        return new_headers, new_rows

    def report(
        self, other: Optional["JsonProfile"] = None, name_limit: int = 40
    ) -> str:
        def create_ret(
            table_headers: list[str], table_rows: dict[str, list[str]]
        ) -> str:
            table_flattened = [
                [kernel_name[:name_limit], *kernel_vals]
                for kernel_name, kernel_vals in table_rows.items()
            ]
            return tabulate_2d(table_flattened, headers=table_headers)

        if other is not None:
            self._compute_stats()
            other._compute_stats()

            self_tables = self._create_tables(self._devices)
            other_tables = self._create_tables(other._devices)

            self_name = (
                self.benchmark_name if self.benchmark_name is not None else "Table 1"
            )
            other_name = (
                other.benchmark_name if other.benchmark_name is not None else "Table 2"
            )

            ret = []
            if self._devices.keys() != other._devices.keys():
                raise AssertionError(
                    f"Device key mismatch: {self._devices.keys()} != {other._devices.keys()}"
                )
            for device_idx, t1, t2 in zip_dicts(
                self_tables, other_tables, d1_default=None, d2_default=None
            ):
                if t1 is None:
                    raise AssertionError("t1 must not be None for device comparison")
                if t2 is None:
                    raise AssertionError("t2 must not be None for device comparison")
                table_headers, table_rows = self._combine_tables(
                    t1, self_name, t2, other_name
                )
                tab_string = create_ret(table_headers, table_rows)
                # pyrefly: ignore [bad-argument-type]
                ret.append(f"{self._devices[device_idx]}:\n{tab_string}")
            return "\n".join(ret)
        self._compute_stats()

        self_tables = self._create_tables(self._devices)

        ret = []
        for idx, table in self_tables.items():
            table_headers, table_rows = table
            tab_string = create_ret(table_headers, table_rows)
            # pyrefly: ignore [bad-argument-type]
            ret.append(f"{self._devices[idx]}:\n{tab_string}")
        return "\n".join(ret)

    def dump(self, out: str) -> None:
        with open(out, "w") as f:
            json.dump(self.data, f)

    def combine_with(self, other: "JsonProfile") -> "JsonProfile":
        """
        Combine this profile with another profile by merging their trace events.
        Returns a new JsonProfile object with combined data.
        """
        # Create a new combined data structure
        combined_data = {
            "traceEvents": self.data["traceEvents"] + other.data["traceEvents"],
            "deviceProperties": self.data.get("deviceProperties", []),
        }

        # Merge device properties, avoiding duplicates
        other_device_props = other.data.get("deviceProperties", [])
        existing_device_ids = OrderedSet(
            [dev["id"] for dev in combined_data["deviceProperties"]]
        )

        for device_prop in other_device_props:
            if device_prop["id"] not in existing_device_ids:
                combined_data["deviceProperties"].append(device_prop)

        # Copy any other top-level properties from the first profile
        for key, value in self.data.items():
            if key not in combined_data:
                combined_data[key] = value

        import os

        # Create a temporary file to write the combined data
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            json.dump(combined_data, tmp_file)
            tmp_path = tmp_file.name

        try:
            # Create new JsonProfile from the combined data
            combined_profile = JsonProfile(
                tmp_path,
                benchmark_name=f"{self.benchmark_name or 'Profile1'}_+_{other.benchmark_name or 'Profile2'}",
                dtype=self.dtype or other.dtype,
            )
            return combined_profile
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)


class ParseException(RuntimeError):
    pass


def main() -> None:
    """
    Main function for the profile analysis script.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diff",
        nargs=5,
        metavar=(
            "input_file1",
            "name1",
            "input_file2",
            "name2",
            "dtype",
        ),
        help="Two json traces to compare with, specified as <file1> <name1> <file2> <name2> <dtype>",
    )
    parser.add_argument(
        "--name_limit",
        type=int,
        help="the maximum name size in the final report",
    )
    parser.add_argument(
        "--augment_trace",
        "-a",
        nargs=3,
        metavar=("input_file", "output_file", "dtype"),
        help="Augment a trace with inductor meta information. Provide input and output file paths.",
    )
    parser.add_argument(
        "--analysis",
        nargs=2,
        metavar=("input_file", "dtype"),
        help="Run analysis on a single trace, specified as <file> <dtype>",
    )
    parser.add_argument(
        "--combine",
        nargs="+",
        metavar=("input_files", "output_file"),
        help="Combine multiple profiles into a single profile by merging trace events. Specify as <input_file1> \
<input_file2> [input_file3 ...] <output_file>. The last argument is the output file, all preceding arguments are \
input files to combine.",
    )
    args = parser.parse_args()

    if args.diff:
        p1 = JsonProfile(args.diff[0], args.diff[1], dtype=args.diff[4])
        p1.augment_trace()
        p2 = JsonProfile(args.diff[2], args.diff[3], dtype=args.diff[4])
        p2.augment_trace()
        if args.name_limit:
            print(p1.report(p2, name_limit=args.name_limit))
        else:
            print(p1.report(p2))
    if args.analysis:
        p1 = JsonProfile(
            args.analysis[0],
            dtype=args.analysis[1],
        )
        p1.augment_trace()
        if args.name_limit:
            print(p1.report(name_limit=args.name_limit))
        else:
            print(p1.report())
    if args.augment_trace:
        p = JsonProfile(args.augment_trace[0], dtype=args.augment_trace[2])
        p.augment_trace()
        p.dump(args.augment_trace[1])
    if args.combine:
        input_files = args.combine[:-1]  # All arguments except the last one
        output_file = args.combine[-1]  # Last argument is the output file

        if len(input_files) < 2:
            print("Error: At least 2 input files are required for combining")
            return

        # Load the first profile
        combined = JsonProfile(input_files[0], dtype=None)

        # Iteratively combine with all other profiles
        for input_file in input_files[1:]:
            profile = JsonProfile(input_file, dtype=None)
            combined = combined.combine_with(profile)

        combined.dump(output_file)
        print(f"Successfully combined {', '.join(input_files)} into {output_file}")


if __name__ == "__main__":
    main()
<|endoftext|>import logging
from dataclasses import dataclass

import torch


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    """
    Theoretical numbers from data sheet.  When a data sheet reports both
    Tensor/Matrix-Core and non-Tensor-Core numbers, the higher (Tensor Core)
    number is used.

    NVIDIA data sheets since Hopper (H100) only publish Tensor-Core TFLOPS
    with 2:4 structured sparsity (marked ``*With sparsity``).  For devices
    whose ``tops`` include sparsity, set ``tops_sparsity_factor`` to 2 so
    that callers can recover the dense (non-sparse) throughput used by
    cuBLAS / cuDNN in normal (non-sparse) workloads.

    Bandwidth numbers are tricky, because there are platform differences
    that may not show up in the profiler trace.
    """

    tops: dict[torch.dtype | str, float]
    dram_bw_gbs: float
    dram_gb: float
    tops_sparsity_factor: int = 1


# Indexing is based on `torch.cuda.get_device_name()`, normalized to upper-case.
# TODO investigate profiler support for tf32 and allow device to report correct number when it's turned on.
_device_mapping: dict[str, DeviceInfo] = {
    # Source:
    # @lint-ignore https://www.nvidia.com/en-us/data-center/h100/
    # Tensor Core values are *with sparsity* per the datasheet.
    "NVIDIA H100": DeviceInfo(
        tops={
            torch.float64: 67.0,
            torch.float32: 67.0,
            "torch.tf32": 989.0,
            torch.bfloat16: 1979.0,
            torch.float16: 1979.0,
            torch.float8_e8m0fnu: 3958.0,
            torch.float8_e8m0fnu: 3958.0,
            torch.float8_e4m3fnuz: 3958.0,
            torch.float8_e5m2: 3958.0,
            torch.float8_e5m2fnuz: 3958.0,
            torch.float8_e8m0fnu: 3958.0,
            torch.int8: 3958.0,
        },
        dram_bw_gbs=3350,
        dram_gb=80,
        tops_sparsity_factor=2,
    ),
    # Source:
    # @lint-ignore https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/
    # nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
    "NVIDIA A100": DeviceInfo(
        tops={
            torch.float64: 19.5,
            torch.float32: 19.5,
            torch.bfloat16: 312.5,
            torch.float16: 312.5,
            # Not in datasheet: float8
            torch.int8: 624.0,
            "torch.tf32": 156.0,
        },
        dram_bw_gbs=2039.0,
        dram_gb=80.0,
    ),
    # Source:
    # @lint-ignore https://resources.nvidia.com/en-us-gpu-resources/l4-tensor-datasheet
    "NVIDIA L4": DeviceInfo(
        tops={
            # This is a guess, not in datasheet
            torch.float64: 15.1,
            torch.float32: 30.3,
            "torch.tf32": 120.0,
            torch.bfloat16: 242.0,
            torch.float16: 242.0,
            torch.float8_e8m0fnu: 485.0,
            torch.float8_e8m0fnu: 485.0,
            torch.float8_e4m3fnuz: 485.0,
            torch.float8_e5m2: 485.0,
            torch.float8_e5m2fnuz: 485.0,
            torch.float8_e8m0fnu: 485.0,
            torch.int8: 485.0,
        },
        dram_bw_gbs=3350,
        dram_gb=24,
    ),
    # Source:
    # @lint-ignore https://www.amd.com/content/dam/amd/en/documents\
    # /instinct-tech-docs/product-briefs/amd-instinct-mi350x-gpu-brochure.pdf
    "AMD MI350X": DeviceInfo(
        tops={
            torch.float64: 72.1,
            torch.float32: 144.2,
            # not specified, fall back to float32 numbers
            "torch.tf32": 144.2,
            torch.bfloat16: 2309.6,
            torch.float16: 2309.6,
            torch.float8_e8m0fnu: 4614.0,
            torch.float8_e8m0fnu: 4614.0,
            torch.float8_e4m3fnuz: 4614.0,
            torch.float8_e5m2: 4614.0,
            torch.float8_e5m2fnuz: 4614.0,
            torch.float8_e8m0fnu: 4614.0,
            torch.int8: 4614.0,
        },
        dram_bw_gbs=8000.0,
        dram_gb=288.0,
    ),
    # Source:
    # @lint-ignore https://www.amd.com/content/dam/amd/en/documents\
    # /instinct-tech-docs/data-sheets/amd-instinct-mi300a-data-sheet.pdf
    "AMD MI300A": DeviceInfo(
        tops={
            torch.float64: 122.6,
            torch.float32: 122.6,
            "torch.tf32": 490.3,
            torch.bfloat16: 980.6,
            torch.float16: 980.6,
            torch.float8_e8m0fnu: 1961.2,
            torch.float8_e8m0fnu: 1961.2,
            torch.float8_e4m3fnuz: 1961.2,
            torch.float8_e5m2: 1961.2,
            torch.float8_e5m2fnuz: 1961.2,
            torch.float8_e8m0fnu: 1961.2,
            torch.int8: 1961.2,
        },
        dram_bw_gbs=5300.0,
        dram_gb=128.0,
    ),
    # Source:
    # @lint-ignore https://www.amd.com/content/dam/amd/en/documents/\
    # instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf
    "AMD MI300X": DeviceInfo(
        tops={
            torch.float64: 163.4,
            torch.float32: 163.4,
            "torch.tf32": 653.7,
            torch.bfloat16: 1307.4,
            torch.float16: 1307.4,
            torch.float8_e8m0fnu: 2614.9,
            torch.float8_e8m0fnu: 2614.9,
            torch.float8_e4m3fnuz: 2614.9,
            torch.float8_e5m2: 2614.9,
            torch.float8_e5m2fnuz: 2614.9,
            torch.float8_e8m0fnu: 2614.9,
            torch.int8: 2614.9,
        },
        dram_bw_gbs=5300.0,
        dram_gb=192.0,
    ),
    # Source:
    # @lint-ignore https://www.amd.com/content/dam/amd/\
    # en/documents/instinct-business-docs/product-briefs/instinct-mi210-brochure.pdf
    "AMD MI210X": DeviceInfo(
        tops={
            torch.float64: 45.3,
            torch.float32: 45.3,
            # not specified, fall back to float32 numbers
            "torch.tf32": 45.3,
            torch.bfloat16: 181.0,
            torch.float16: 181.0,
            # not specified, fall back to float16 numbers
            torch.float8_e8m0fnu: 181.0,
            torch.float8_e8m0fnu: 181.0,
            torch.float8_e4m3fnuz: 181.0,
            torch.float8_e5m2: 181.0,
            torch.float8_e5m2fnuz: 181.0,
            torch.float8_e8m0fnu: 181.0,
            torch.int8: 181.0,
        },
        # pcie4.0x16
        dram_bw_gbs=1600.0,
        dram_gb=64.0,
    ),
    # Source:
    # @lint-ignore https://www.intel.com/content/www/us/en/products/sku/241598/
    # intel-arc-b580-graphics/specifications.html
    "INTEL B580": DeviceInfo(
        tops={
            # Estimated from published single-precision throughput.
            torch.float64: 6.83,
            torch.float32: 13.67,
            "torch.tf32": 116.5,
            torch.bfloat16: 116.5,
            torch.float16: 116.5,
            # not specified, fall back to fp16 matrix throughput
            torch.float8_e8m0fnu: 116.5,
            torch.float8_e4m3fnuz: 116.5,
            torch.float8_e5m2: 116.5,
            torch.float8_e5m2fnuz: 116.5,
            torch.int8: 233,
        },
        dram_bw_gbs=456.0,
        dram_gb=12.0,
    ),
    # Source:
    # @lint-ignore https://www.intel.com/content/www/us/en/products/sku/245797/
    # intel-arc-pro-b70-graphics/specifications.html
    "INTEL B70": DeviceInfo(
        tops={
            torch.float64: 11.47,
            torch.float32: 22.94,
            "torch.tf32": 183.5,
            torch.bfloat16: 183.5,
            torch.float16: 183.5,
            torch.float8_e8m0fnu: 183.5,
            torch.float8_e4m3fnuz: 183.5,
            torch.float8_e5m2: 183.5,
            torch.float8_e5m2fnuz: 183.5,
            torch.int8: 367,
        },
        dram_bw_gbs=608.0,
        dram_gb=32.0,
    ),
    # Source:
    # @lint-ignore https://www.intel.com/content/www/us/en/products/sku/232876/\
    # intel-data-center-gpu-max-1100/specifications.html
    "Intel(R) Data Center GPU Max 1100": DeviceInfo(
        # XeCore count: 56,
        # vector/matrix engines per XeCore: 8
        # freq: 1.55GHz
        tops={
            # Vector engine
            torch.float64: 11.1,
            torch.float32: 22.2,
            # not specified, fall back to float32 numbers
            "torch.tf32": 22.2,
            # Matrix engine
            torch.float16: 355.5,
            torch.bfloat16: 355.5,
            # not supported, fall back to float32 numbers
            torch.float8_e8m0fnu: 22.2,
            torch.float8_e4m3fnuz: 22.2,
            torch.float8_e5m2: 22.2,
            torch.float8_e5m2fnuz: 22.2,
            torch.int8: 711.1,
        },
        dram_bw_gbs=1228.8,
        dram_gb=48,
    ),
}
_device_mapping["AMD INSTINCT MI350X"] = _device_mapping["AMD MI350X"]
_device_mapping["AMD INSTINCT MI300X"] = _device_mapping["AMD MI300X"]
_device_mapping["AMD INSTINCT MI210X"] = _device_mapping["AMD MI210X"]
_device_mapping["Intel(R) Arc(TM) B580 Graphics"] = _device_mapping["INTEL B580"]
_device_mapping["Intel(R) Arc(TM) Pro B70 Graphics"] = _device_mapping["INTEL B70"]

# Enforce the upper-case-key invariant so entries cannot silently miss
# `lookup_device_info` (which upper-cases the query before lookup).
_device_mapping = {k.upper(): v for k, v in _device_mapping.items()}


def lookup_device_info(name: str) -> DeviceInfo | None:
    """
    Problem: when diffing profiles between amd and nvidia, we don't have access to the device information
    of the other one. Also, since the analysis is static, we should be able to do it on another device unrelated
    to the recorded device. Therefore, _device_mapping statically contains the information for lots of devices.
    If one is missing, please run DeviceInfo.get_device_info() and add it to _device_mapping.
      name (str): name of the device to lookup. Should map onto torch.cuda.get_device_name().
      Will be upper-cased before lookup.
    """
    return _device_mapping.get(name.upper())


def datasheet_tops(
    dtype: torch.dtype, is_tf32: bool = False, device_name: str | None = None
) -> float | None:
    """
    Get the theoretical *dense* TFLOPS of the device for a given dtype.

    If the datasheet values include 2:4 structured sparsity
    (``tops_sparsity_factor > 1``), they are divided by that factor so
    callers always receive the throughput achievable by cuBLAS/cuDNN on
    non-sparse data.

    If ``device_name`` is given, the datasheet is looked up for that named
    device instead of querying the current device. This lets callers pin a
    device spec for deterministic, hardware-independent estimates.
    """
    if device_name is not None:
        name: str | None = device_name
    elif torch.cuda.is_available():
        name = torch.cuda.get_device_name()
    elif torch.xpu.is_available():
        name = torch.xpu.get_device_name()
    else:
        log.info("No supported device available, skipping datasheet lookup")
        return None

    if name is None:
        log.info("No device found, returning None")
        return None
    device_info = lookup_device_info(name)
    if device_info is None:
        log_str = f"Device {name} not in datasheet, returning None"
        log.info(log_str)
        return None
    if dtype not in device_info.tops:
        log.info(
            "Device %s does not have a datasheet entry for %s, returning None",
            name,
            dtype,
        )
        return None

    tops = device_info.tops[
        "torch.tf32" if dtype == torch.float32 and is_tf32 else dtype
    ]
    return tops / device_info.tops_sparsity_factor


def datasheet_dram_bw_gbs(device_name: str | None = None) -> float | None:
    """
    Get the theoretical DRAM bandwidth (GB/s) of a device from the datasheet,
    or None if the device is not present.

    If ``device_name`` is given, that named device is looked up; otherwise the
    current CUDA/XPU device is queried.
    """
    if device_name is None:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name()
        elif torch.xpu.is_available():
            device_name = torch.xpu.get_device_name()
        else:
            log.info("No supported device available, skipping datasheet lookup")
            return None

    if device_name is None:
        log.info("No device found, returning None")
        return None

    device_info = lookup_device_info(device_name)
    if device_info is None:
        log.info("Device %s not in datasheet, returning None", device_name)
        return None
    return device_info.dram_bw_gbs
<|endoftext|>import atexit
import concurrent
import dataclasses
import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from time import time
from typing import Any, TypeVar
from typing_extensions import ParamSpec

# _thread_safe_fork is needed because the subprocesses in the pool can read
# justknobs, e.g., in the Triton compiler. For internal, the import installs
# functionality to destroy singletons before forking and re-enable them after.
import torch._thread_safe_fork  # noqa: F401


_P = ParamSpec("_P")
_R = TypeVar("_R")


log = logging.getLogger(__name__)


@dataclass
class _QueueStats:
    # Mapping from id(future) -> start time
    pending: dict[int, float] = dataclasses.field(default_factory=dict)
    timing: list[float] = dataclasses.field(default_factory=list)
    enqueue_count: int = 0
    dequeue_count: int = 0
    max_queue_depth: int = 0
    pool_count: int = 0


# The queue statistics tracked by TrackedProcessPoolExecutor. Always grab
# _queue_stats_lock before touching.
_queue_stats = _QueueStats()
_queue_stats_lock = threading.Lock()


class TrackedProcessPoolExecutor(ProcessPoolExecutor):
    def __init__(
        self,
        max_workers: int | None = None,
        mp_context: BaseContext | None = None,
        initializer: Callable[[], object] | None = None,
    ) -> None:
        with _queue_stats_lock:
            _queue_stats.pool_count += 1
        super().__init__(max_workers, mp_context, initializer)

    def _record_dequeue(self, f: Future[Any]) -> None:
        now = time()
        with _queue_stats_lock:
            stats = _queue_stats
            if (start_time := stats.pending.pop(id(f), None)) is None:
                return
            stats.dequeue_count += 1
            duration = now - start_time
            stats.timing.append(duration)

    def _record_enqueue(self, f: Future[Any]) -> None:
        # Monkeypatch the set_running_or_notify_cancel so we can track when the Future moves out of PENDING.
        saved_running_or_notify_cancel = f.set_running_or_notify_cancel

        def set_running_or_notify_cancel() -> Any:
            self._record_dequeue(f)
            return saved_running_or_notify_cancel()

        now = time()
        with _queue_stats_lock:
            stats = _queue_stats
            stats.pending[id(f)] = now
            stats.enqueue_count += 1
            stats.max_queue_depth = max(stats.max_queue_depth, len(stats.pending))
            f.set_running_or_notify_cancel = set_running_or_notify_cancel  # type: ignore[method-assign]

        if f._state != concurrent.futures._base.PENDING:
            self._record_dequeue(f)

    def submit(
        self, fn: Callable[_P, _R], /, *args: _P.args, **kwargs: _P.kwargs
    ) -> Future[_R]:
        # pyrefly: ignore [bad-argument-type]
        f = super().submit(fn, *args, **kwargs)
        self._record_enqueue(f)
        return f


@atexit.register
def _queue_stats_report() -> None:
    stats = _queue_stats
    if stats.pool_count == 0:
        return

    timing = stats.timing
    timing.sort()

    log.info("AsyncCompile Metrics:")
    log.info("  Pools %s", stats.pool_count)
    log.info(
        "  Items %d enqueued / %d dequeued", stats.enqueue_count, stats.dequeue_count
    )
    log.info("  Max Queue Depth: %d", stats.max_queue_depth)
    n = len(timing)
    if n > 0:
        log.info("  Longest queue time: %0.2fs", timing[-1])
        log.info("  P50: %0.2fs", timing[n // 2])
        if n >= 20:
            log.info("  P95: %0.2fs", timing[n * 95 // 100])
<|endoftext|>from collections.abc import Callable
from threading import Lock, Thread
from time import monotonic, sleep


class Timer:
    """
    This measures how long we have gone since last receiving an event and if it is greater than a set interval, calls a function.
    """

    def __init__(
        self,
        duration: int | float,  # Duration in seconds
        call: Callable[[], None],  # Function to call when we expire
    ) -> None:
        # We don't start the background thread until we actually get an event.
        self.background_thread: Thread | None = None
        self.last_called: float | None = None
        self.duration = duration
        self.sleep_time = duration / 2
        self.call = call
        self.exit = False

        self.lock = Lock()

    def record_call(self) -> None:
        with self.lock:
            if self.background_thread is None:
                self.background_thread = Thread(
                    target=self.check, daemon=True, name="subproc_worker_timer"
                )
                self.background_thread.start()
            self.last_called = monotonic()

    def quit(self) -> None:
        with self.lock:
            self.exit = True

    def check(self) -> None:
        while True:
            # We have to be sensitive on checking here, to avoid too much impact on cpu
            sleep(self.sleep_time)
            with self.lock:
                if self.exit:
                    return
                if self.last_called is None:
                    raise AssertionError("last_called must be set before timer check")
                if self.last_called + self.duration >= monotonic():
                    continue
                self.last_called = None
                self.background_thread = None

            # Releasing lock in case self.call() takes a very long time or is reentrant
            self.call()
            return
<|endoftext|>import os
import signal
from threading import Thread
from time import sleep

from torch._inductor.compile_worker import watchdog


_IN_TOPLEVEL_PROCESS = True


def in_toplevel_process() -> bool:
    global _IN_TOPLEVEL_PROCESS
    return _IN_TOPLEVEL_PROCESS


# If this process dies abnormally (e.g. segfault)
# it will not shut down the workers. Instead,
# the workers will have their parent reassigned to the
# init process. This launches a separate thread to
# watch for the worker getting reassigned,
# and cleans it up in this case.
#
# This function cannot be an inner function since otherwise mp_context="spawn" would
# not work for ProcessPoolExecutor since inner functions cannot be pickled.
def _async_compile_initializer(orig_ppid: int, close_fds: tuple[int, ...] = ()) -> None:
    # In fork mode a worker inherits the sidecar's entire fd table, including the
    # sidecar<->parent pipes. The worker must not keep those open: holding the
    # result pipe's write end would stop the parent from ever seeing EOF when the
    # sidecar dies. The caller only passes these fds under fork; under spawn the
    # worker doesn't inherit them and the same integers would name unrelated fds
    # the fresh interpreter reused, so closing them there would be silent
    # corruption (the OSError guard wouldn't catch it) rather than a no-op.
    for fd in close_fds:
        try:
            os.close(fd)
        except OSError:
            pass

    import torch._C

    def run() -> None:
        while True:
            sleep(60)
            if orig_ppid != os.getppid():
                os.kill(os.getpid(), signal.SIGKILL)

    global _watchdog_thread, _original_parent
    _original_parent = orig_ppid
    _watchdog_thread = Thread(target=run, daemon=True)
    _watchdog_thread.start()
    # Ignore Ctrl-C (i.e. SIGINT) sent to pool workers to avoid meaningless log spam.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Install a crash handler to print out the stacktrace for SEGV
    torch._C._initCrashHandler()

    # Set a bit to distinguish async_compile subprocesses from the toplevel process.
    global _IN_TOPLEVEL_PROCESS
    _IN_TOPLEVEL_PROCESS = False

    # Claim a heartbeat slot so the sidecar watchdog can report this worker's
    # compile phase (no-op unless the sidecar allocated the shared buffer).
    watchdog.init_worker_slot()


_watchdog_thread: Thread | None = None
_original_parent: int | None = None


def has_parent_changed() -> bool:
    return _original_parent != os.getppid()
<|endoftext|>"""Shared-memory heartbeat so the compile-worker sidecar's watchdog can report
which phase a slow/stuck worker is in (see SubprocMain._watchdog_loop).

A single multiprocessing.Array is allocated in the sidecar (create()) before it
forks its worker pool; every forked worker inherits the same buffer and writes
only its own slot -- single writer per slot, so no lock is needed there and a
rare torn read is harmless for a coarse once-a-minute diagnostic. The sidecar's
watchdog thread reads all slots.

Slots are recycled off the pool's lifecycle rather than by polling worker
liveness: the sidecar calls reset() every time it (re)builds the worker pool,
and workers then claim slots 0..nprocs-1 with a per-generation counter. A worker
death tears down and rebuilds the whole pool (ProcessPoolExecutor's behavior), so
reset() runs before the replacement generation forks and there is never more than
one live worker per slot. (If the pool were ever configured to replace workers
individually -- e.g. max_tasks_per_child -- a replacement would simply run out of
counter and report nothing, rather than collide with a live worker's slot.)

Everything is a no-op unless the sidecar allocated the buffer and the worker
claimed a slot, so the report_phase() calls sprinkled through the compile path
are free in the main process and in non-forking pools (spawn workers do not
inherit the buffer and simply degrade to duration-only reporting).
"""

from __future__ import annotations

import multiprocessing
import os
import time
from enum import IntEnum
from typing import Any


class Phase(IntEnum):
    # Persisted as ints in the shared buffer; append new phases, don't renumber.
    RUNNING = 0
    QUERYING_CACHE = 1
    COMPILING = 2


# Per-slot layout: _FIELDS longs.
_FIELDS = 4
_F_JOB_ID = 0
_F_PHASE = 1
_F_PHASE_START_NS = 2
_F_PID = 3
_EMPTY = -1  # job_id sentinel for an idle slot

# Allocated in the sidecar (create) before forking; inherited by fork workers.
# _counter's lock also serializes slot claims and reset().
_heartbeat: Any | None = None
_counter: Any | None = None
_nprocs: int = 0
# Claimed post-fork by each worker (init_worker_slot).
_slot: int | None = None


def create(nprocs: int) -> None:
    """Sidecar: allocate the shared buffer before forking the worker pool."""
    global _heartbeat, _counter, _nprocs
    ctx = multiprocessing.get_context("fork")
    _nprocs = nprocs
    _counter = ctx.Value("i", 0)
    buf = ctx.Array("q", nprocs * _FIELDS, lock=False)
    for s in range(nprocs):
        buf[s * _FIELDS + _F_JOB_ID] = _EMPTY
    _heartbeat = buf


def reset() -> None:
    """Sidecar: recycle all slots for a fresh worker generation. Call before
    (re)building the worker pool so the new workers claim clean slots 0..n-1."""
    if _heartbeat is None or _counter is None:
        return
    with _counter.get_lock():
        _counter.value = 0
        for s in range(_nprocs):
            base = s * _FIELDS
            _heartbeat[base + _F_JOB_ID] = _EMPTY
            _heartbeat[base + _F_PID] = 0


def init_worker_slot() -> None:
    """Worker: claim this generation's next slot. No-op if the sidecar didn't
    allocate a buffer (main process / spawn pools)."""
    global _slot
    if _heartbeat is None or _counter is None or _nprocs <= 0:
        return
    with _counter.get_lock():
        slot = _counter.value
        _counter.value = slot + 1
    # Only the first nprocs workers of a generation get a slot; any beyond that
    # (which shouldn't happen without per-worker replacement) report nothing.
    if slot < _nprocs:
        _slot = slot
        _heartbeat[slot * _FIELDS + _F_PID] = os.getpid()


def set_current_job(job_id: int) -> None:
    """Worker: mark the start of a job in this worker's slot."""
    if _heartbeat is None or _slot is None:
        return
    base = _slot * _FIELDS
    _heartbeat[base + _F_PHASE] = int(Phase.RUNNING)
    _heartbeat[base + _F_PHASE_START_NS] = time.monotonic_ns()
    # Written last so a concurrent reader sees a fully-populated slot or none.
    _heartbeat[base + _F_JOB_ID] = job_id


def report_phase(phase: Phase) -> None:
    """Worker: record the phase this worker just entered."""
    if _heartbeat is None or _slot is None:
        return
    base = _slot * _FIELDS
    _heartbeat[base + _F_PHASE_START_NS] = time.monotonic_ns()
    _heartbeat[base + _F_PHASE] = int(phase)


def clear_current_job() -> None:
    """Worker: mark this slot idle when its job finishes."""
    if _heartbeat is None or _slot is None:
        return
    _heartbeat[_slot * _FIELDS + _F_JOB_ID] = _EMPTY


def enabled() -> bool:
    """Sidecar: whether phase heartbeats are active (fork pools only)."""
    return _heartbeat is not None


def read_heartbeats() -> dict[int, tuple[int, int, int]]:
    """Sidecar: job_id -> (phase, phase_start_ns, pid) for currently-busy slots.

    phase_start_ns is worker monotonic_ns; on Linux CLOCK_MONOTONIC is
    system-wide so the sidecar can diff it against its own monotonic_ns.
    """
    if _heartbeat is None:
        return {}
    out: dict[int, tuple[int, int, int]] = {}
    for s in range(_nprocs):
        base = s * _FIELDS
        job_id = _heartbeat[base + _F_JOB_ID]
        if job_id != _EMPTY:
            out[job_id] = (
                _heartbeat[base + _F_PHASE],
                _heartbeat[base + _F_PHASE_START_NS],
                _heartbeat[base + _F_PID],
            )
    return out
<|endoftext|># mypy: allow-untyped-defs
import argparse
import base64
import functools
import importlib
import logging
import os
import sys
from typing import TypeVar

from torch._inductor.async_compile import pre_fork_setup
from torch._inductor.codecache import torch_key
from torch._inductor.compile_worker.subproc_pool import (
    SubprocKind,
    SubprocMain,
    SubprocPickler,
)
from torch._inductor.compile_worker.utils import _async_compile_initializer
from torch._inductor.runtime.compile_tasks import _set_triton_ptxas_path


_T = TypeVar("_T")


log = logging.getLogger(__name__)

_set_triton_ptxas_path()

try:
    import triton

    if triton is None:
        raise AssertionError("triton failed to preload in parent")
except ImportError:
    pass


def _lookup_and_create_type(base: type[_T], qname: str) -> _T:
    """
    Given a base type and qualified name: import & lookup that name, check
    that it's of the given type and then instantiate it.
    """
    pkg, name = qname.rsplit(".", 1)
    mod = importlib.import_module(pkg)
    ty = getattr(mod, name)
    if not issubclass(ty, base):
        raise TypeError(f"Type {ty} is not a subtype of {base}")
    return ty()


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--pickler", type=functools.partial(_lookup_and_create_type, SubprocPickler)
        )
        parser.add_argument("--kind", type=SubprocKind)
        parser.add_argument("--workers", type=int)
        parser.add_argument("--parent", type=int)
        parser.add_argument("--read-fd", type=int)
        parser.add_argument("--write-fd", type=int)
        parser.add_argument("--torch-key", type=str)
        args = parser.parse_args()
        if os.getppid() != args.parent:
            sys.exit(0)
        read_fd = os.fdopen(args.read_fd, "rb")
        write_fd = os.fdopen(args.write_fd, "wb")

        pre_fork_setup()

        torch_key.set(base64.b64decode(args.torch_key.encode("utf-8")))  # type: ignore[attr-defined]

        _async_compile_initializer(args.parent)

        SubprocMain(args.pickler, args.kind, args.workers, read_fd, write_fd).main()
    except Exception:
        log.exception("Uncaught exception in compile_worker subprocess")


if __name__ == "__main__":
    main()
<|endoftext|>import base64
import functools
import itertools
import json
import logging
import multiprocessing
import os
import pickle
import struct
import subprocess
import sys
import threading
import time
import traceback
import typing
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, IO, TypeVar
from typing_extensions import Never, ParamSpec

# _thread_safe_fork is needed because the subprocesses in the pool can read
# justknobs, e.g., in the Triton compiler. For internal, the import installs
# functionality to destroy singletons before forking and re-enable them after.
import torch._thread_safe_fork  # noqa: F401
from torch._inductor import config
from torch._inductor.codecache import torch_key
from torch._inductor.compile_worker import watchdog
from torch._inductor.compile_worker.timer import Timer
from torch._inductor.compile_worker.tracked_process_pool import (
    TrackedProcessPoolExecutor,
)
from torch._inductor.compile_worker.utils import _async_compile_initializer
from torch._inductor.utils import get_ld_library_path, python_subprocess_env
from torch._logging import trace_structured
from torch._utils_internal import find_compile_subproc_binary
from torch.monitor import _WaitCounter, _WaitCounterTracker


log = logging.getLogger(__name__)

_P = ParamSpec("_P")
_T = TypeVar("_T")


class MsgHeader(IntEnum):
    ERROR = 0
    SHUTDOWN = 1
    QUIESCE = 2
    WAKEUP = 3
    JOB = 4
    # Sidecar -> parent watchdog report about a still-running job (out-of-band,
    # low volume). Payload is a pickled status dict; does not affect the future.
    STATUS = 5


def _current_compile_id() -> Any:
    # Snapshotted at submit so a later watchdog STATUS report (handled on the
    # read thread, which has no ambient compile context) can be attributed to the
    # right compile in tlparse.
    try:
        from torch._guards import CompileContext

        return CompileContext.current_compile_id()
    except Exception:
        return None


def _pack_msg(msg_header: MsgHeader, job_id: int, length: int) -> bytes:
    return struct.pack("nnn", int(msg_header), job_id, length)


def _unpack_msg(data: bytes) -> tuple[MsgHeader, int, int]:
    if not data:
        return MsgHeader.ERROR, -1, -1
    msg_header, job_id, length = struct.unpack("nnn", data)
    return MsgHeader(msg_header), job_id, length


msg_bytes = len(_pack_msg(MsgHeader.JOB, 0, 0))

# How often the parent polls the sidecar process for liveness. The sidecar
# dying is the only signal this catches, so a coarse interval is fine.
_SIDECAR_HEALTH_POLL_SECONDS = 2.0

# How long, in total, to let compile workers exit on SIGTERM during pool
# teardown before escalating to SIGKILL. A worker wedged holding a lock or in a
# C/GIL section won't honor SIGTERM, and an unbounded wait for it would stall the
# whole shutdown.
_WORKER_TERMINATE_GRACE_SECONDS = 10.0


def _send_msg(
    write_pipe: IO[bytes], msg_header: MsgHeader, job_id: int = -1, data: bytes = b""
) -> None:
    length = len(data)
    write_pipe.write(_pack_msg(msg_header, job_id, length))
    if length > 0:
        write_pipe.write(data)
    write_pipe.flush()


def _recv_msg(read_pipe: IO[bytes]) -> tuple[MsgHeader, int, bytes]:
    msg_header, job_id, length = _unpack_msg(read_pipe.read(msg_bytes))
    data = read_pipe.read(length) if length > 0 else b""
    return msg_header, job_id, data


class _SubprocExceptionInfo:
    """
    Carries exception info from subprocesses across the wire. traceback
    objects are not pickleable, so we store the trace as a string and
    use it for the message in the exception thrown in the main process.
    """

    def __init__(self, details: str) -> None:
        self.details = details


class SubprocException(Exception):
    """
    Thrown when a job in a subprocess raises an Exception.
    """

    def __init__(self, details: str, name: str = "<unknown>") -> None:
        self.details = details
        super().__init__(
            f"An exception occurred in a subprocess:\n\nName={name}\n{details}"
        )

    def with_name(self, name: str) -> "SubprocException":
        return SubprocException(self.details, name)


class SubprocPickler:
    """
    Allows a caller to provide a custom pickler for passing data with the
    subprocess.
    """

    def dumps(self, obj: object) -> bytes:
        return pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)

    def loads(self, data: bytes) -> object:
        return pickle.loads(data)


class SubprocKind(Enum):
    FORK = "fork"
    SPAWN = "spawn"


class SubprocPool:
    """
    Mimic a concurrent.futures.ProcessPoolExecutor, but wrap it in
    a subprocess.Popen() to try to avoid issues with forking/spawning
    """

    def __init__(
        self,
        nprocs: int,
        pickler: SubprocPickler | None = None,
        kind: SubprocKind = SubprocKind.FORK,
        quiesce: bool = False,
    ) -> None:
        entry = os.path.join(os.path.dirname(__file__), "__main__.py")
        self.pickler = pickler or SubprocPickler()
        self.kind = kind

        subproc_read_fd, write_fd = os.pipe()
        read_fd, subproc_write_fd = os.pipe()
        self.write_pipe = os.fdopen(write_fd, "wb")
        self.read_pipe = os.fdopen(read_fd, "rb")
        torch_key_str = base64.b64encode(torch_key()).decode("utf-8")

        cmd = [
            sys.executable,
            entry,
        ]
        if (binary := find_compile_subproc_binary()) is not None:
            cmd = [binary]

        args = [
            f"--pickler={self.pickler.__class__.__module__}.{self.pickler.__class__.__name__}",
            f"--kind={self.kind.value}",
            f"--workers={nprocs}",
            f"--parent={os.getpid()}",
            f"--read-fd={str(subproc_read_fd)}",
            f"--write-fd={str(subproc_write_fd)}",
            f"--torch-key={torch_key_str}",
        ]
        cmd.extend(args)
        log_path = None
        self.log_file = None

        if config.worker_suppress_logging:
            log_path = os.devnull
            log.info("Suppressing compile worker output due to config")
        else:
            log_path = config.torchinductor_worker_logpath
            if not log_path:
                log_path = config.get_worker_log_path()

        if log_path:
            # pyrefly: ignore [bad-assignment]
            self.log_file = open(log_path, "w")  # noqa:SIM115
        # Kept so the liveness watchdog can tail worker output if the sidecar dies.
        self.log_path = log_path

        self.process = subprocess.Popen(
            cmd,
            env={
                **python_subprocess_env(),
                # Safeguard against creating a SubprocPool in the subprocess.
                "TORCH_WARM_POOL": "0",
                # Some internal usages need a modified LD_LIBRARY_PATH.
                "LD_LIBRARY_PATH": get_ld_library_path(),
            },
            pass_fds=(subproc_read_fd, subproc_write_fd),
            stdout=self.log_file,
            stderr=self.log_file,
        )
        # The sidecar inherited these via pass_fds; the parent uses its own ends
        # (write_pipe/read_pipe) and must drop its copies. Otherwise the parent
        # itself keeps the result pipe's write end open, so read_pipe would never
        # EOF even after the sidecar and all workers are gone.
        os.close(subproc_read_fd)
        os.close(subproc_write_fd)

        self.write_lock = threading.Lock()
        self.read_thread = threading.Thread(
            target=self._read_thread, name="InductorSubproc", daemon=True
        )
        # Backstop for the sidecar dying. Closing the inherited pipe fds (in the
        # parent above and in each worker's initializer) means a dead sidecar
        # normally yields a clean EOF that _read_thread handles. This watchdog
        # covers the cases where EOF does not arrive -- e.g. a stray fd copy still
        # holding the write end open -- by detecting the dead process directly.
        self.health_thread = threading.Thread(
            target=self._health_monitor, name="InductorSubprocHealth", daemon=True
        )

        self.futures_lock = threading.Lock()
        self.pending_futures: dict[int, Future[Any]] = {}
        # Compile id captured at submit so watchdog STATUS reports (handled on the
        # read thread) attribute to the right compile in tlparse. Keyed by job_id.
        self._job_compile_id: dict[int, Any] = {}
        # The pending waitcounter, is used to indicate the time when we have any specific job running.
        self.pending_waitcounters: dict[int, Any] = {}
        self.job_id_count = itertools.count()

        # The running waitcounter indicates the time when the SubProcPool object exists.
        self.running = True
        self.running_waitcounter = _WaitCounter(
            "pytorch.wait_counter.subproc_pool.running"
        ).guard()
        self.running_waitcounter.__enter__()

        # The quiesce waitcounter indicates when the job is in a quiesced state.
        self.quiesce_waitcounter: _WaitCounterTracker | None = None

        # Firstjob is used to capture the time from when the firstjob is queued, to when the first job is done.
        self.firstjob = True
        self.firstjob_id: int | None = None
        self.firstjob_waitcounter = _WaitCounter(
            "pytorch.wait_counter.subproc_pool.first_job"
        ).guard()

        if quiesce:
            self.timer: Timer | None = Timer(
                config.quiesce_async_compile_time, self.quiesce
            )
        else:
            self.timer = None

        # Start threads last to ensure all member variables are initialized
        # before any access.
        self.read_thread.start()
        self.health_thread.start()

    def submit(
        self, job_fn: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs
    ) -> Future[_T]:
        if args or kwargs:
            # pyrefly: ignore [bad-assignment]
            job_fn = functools.partial(job_fn, *args, **kwargs)
        job_data = self.pickler.dumps(job_fn)
        future: Future[_T]
        with self.futures_lock:
            job_id = next(self.job_id_count)
            self.pending_futures[job_id] = future = Future()
            self._job_compile_id[job_id] = _current_compile_id()
            self.pending_waitcounters[job_id] = _WaitCounter(
                "pytorch.wait_counter.subproc_pool.job"
            ).guard()
            self.pending_waitcounters[job_id].__enter__()
            if self.quiesce_waitcounter:
                self.firstjob = True
                self.quiesce_waitcounter.__exit__()
                self.quiesce_waitcounter = None
            # This can be entered from either quiesce wakeup, or from startup.
            if self.firstjob:
                self.firstjob_id = job_id
                self.firstjob_waitcounter.__enter__()
                self.firstjob = False
        future.set_running_or_notify_cancel()
        self._send(MsgHeader.JOB, job_id, job_data)
        return future

    def _send(self, msg_header: MsgHeader, job_id: int = -1, data: bytes = b"") -> None:
        with self.write_lock:
            if not self.running:
                raise RuntimeError("Attempting to use a closed pool")
            _send_msg(self.write_pipe, msg_header, job_id, data)

    def _read_thread(self) -> None:
        while True:
            data = b""
            job_id = -1
            try:
                msg_header, job_id, data = _recv_msg(self.read_pipe)
            except Exception:
                # Something went wrong during the read. There's no way we have a
                # valid msg.
                log.exception("failure in subproc_pool._recv_msg")
                msg_header = MsgHeader.ERROR

            if msg_header == MsgHeader.STATUS:
                # Out-of-band watchdog report; does not touch futures.
                self._handle_worker_status(job_id, data)
                continue

            if msg_header != MsgHeader.JOB:
                # read_pipe returned None or got exception
                if self.running:
                    log.warning("SubprocPool unclean exit")
                    self.running = False
                    self.running_waitcounter.__exit__()
                self.read_pipe.close()
                # Cancel all the pending futures.
                self.shutdown()
                return

            try:
                result = self.pickler.loads(data)
            except Exception as e:
                # Something went wrong unpickling. We have a job_id so just
                # notify that particular future and continue on.
                log.exception("unpickle failure in SubprocPool._read_thread")
                result = e

            with self.futures_lock:
                if not self.running:
                    return
                if self.timer:
                    self.timer.record_call()
                if isinstance(result, _SubprocExceptionInfo):
                    # An exception occurred in the submitted job
                    self.pending_futures[job_id].set_exception(
                        SubprocException(result.details)
                    )
                elif isinstance(result, Exception):
                    # An exception occurred in some of our subprocess machinery.
                    self.pending_futures[job_id].set_exception(result)
                else:
                    self.pending_futures[job_id].set_result(result)

                self.pending_waitcounters[job_id].__exit__()
                del self.pending_waitcounters[job_id]
                if self.firstjob_id == job_id:
                    self.firstjob_waitcounter.__exit__()

                del self.pending_futures[job_id]
                self._job_compile_id.pop(job_id, None)

    def _health_monitor(self) -> None:
        # Poll the sidecar for liveness. If it dies while we still think we are
        # running, fail the pending compiles rather than let waiters block
        # forever (see health_thread comment in __init__).
        while self.running:
            returncode = self.process.poll()
            if returncode is not None:
                self._on_sidecar_death(returncode)
                return
            time.sleep(_SIDECAR_HEALTH_POLL_SECONDS)

    def _on_sidecar_death(self, returncode: int) -> None:
        with self.futures_lock:
            if not self.running:
                # Expected exit (shutdown) or already handled.
                return
            self.running = False
        # `running` is set under different locks across shutdown()/_read_thread/
        # here, so this exit can race another for the same transition. That's
        # safe: _WaitCounterTracker.__exit__ is idempotent (optional::reset()).
        self.running_waitcounter.__exit__()

        pid = self.process.pid
        log.error(
            "Inductor compile worker sidecar (pid %s) exited unexpectedly with "
            "code %s during compilation; failing pending compile jobs. Re-run "
            "with TORCHINDUCTOR_COMPILE_THREADS=1 to compile in the main process.",
            pid,
            returncode,
        )
        self._log_sidecar_death_diagnostics(returncode)

        exc = RuntimeError(
            f"Inductor compile worker sidecar (pid {pid}) exited unexpectedly "
            f"with code {returncode} during compilation. Re-run with "
            "TORCHINDUCTOR_COMPILE_THREADS=1 to compile in the main process."
        )
        with self.futures_lock:
            for job_id, future in self.pending_futures.items():
                if not future.cancel():
                    future.set_exception(exc)
                waitcounter = self.pending_waitcounters.pop(job_id, None)
                if waitcounter is not None:
                    waitcounter.__exit__()
            self.pending_futures.clear()
            self._job_compile_id.clear()
        # We intentionally do not close read_pipe here. _read_thread owns it and
        # may be mid-read; closing a buffered pipe out from under a concurrent
        # read can deadlock on the buffer lock. It is a daemon thread, so leaving
        # it is harmless now that the futures are resolved -- and if the sidecar's
        # death already produced an EOF, _read_thread will close the pipe itself.

    def _log_sidecar_death_diagnostics(self, returncode: int) -> None:
        tail = ""
        if self.log_path and self.log_path != os.devnull:
            try:
                with open(self.log_path, errors="replace") as f:
                    tail = "".join(f.readlines()[-50:])
            except OSError:
                pass
        if tail:
            log.error(
                "Last output from compile worker sidecar (pid %s):\n%s",
                self.process.pid,
                tail,
            )
        # Surface the same info to structured tracing so production jobs (which
        # only have tlparse access) can diagnose the crash.
        try:
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "compile_worker_sidecar_death",
                    "encoding": "string",
                },
                payload_fn=lambda: (
                    f"sidecar pid={self.process.pid} returncode={returncode}\n\n{tail}"
                ),
            )
        except Exception:
            log.warning("Failed to emit sidecar-death trace artifact", exc_info=True)

    def _handle_worker_status(self, job_id: int, data: bytes) -> None:
        # Best-effort, and runs on the read thread: a bad status report (corrupt
        # payload or a logging failure) must not escape and kill result
        # processing, so guard the whole thing.
        try:
            status = self.pickler.loads(data)
            if not isinstance(status, dict):
                return
            compile_id = self._job_compile_id.get(job_id)
            record = {**status, "compile_id": str(compile_id) if compile_id else None}
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "compile_worker_status",
                    "encoding": "json",
                },
                payload_fn=lambda: json.dumps(record),
                compile_id=compile_id,
                expect_trace_id=False,
                suppress_context=True,
                record_logging_overhead=False,
            )
        except Exception:
            log.warning("failed to report compile worker status", exc_info=True)

    def quiesce(self) -> None:
        self._send(MsgHeader.QUIESCE)
        if self.quiesce_waitcounter is None:
            self.quiesce_waitcounter = _WaitCounter(
                "pytorch.wait_counter.subproc_pool.quiesced"
            ).guard()
            self.quiesce_waitcounter.__enter__()

    def wakeup(self) -> None:
        self._send(MsgHeader.WAKEUP)

    def shutdown(self) -> None:
        try:
            with self.write_lock:
                if not self.running:
                    return
                if self.timer:
                    self.timer.quit()
                self.running = False
                self.running_waitcounter.__exit__()
                _send_msg(self.write_pipe, MsgHeader.SHUTDOWN)
                self.write_pipe.close()
            try:
                self.process.wait(300)
            except subprocess.TimeoutExpired:
                # The sidecar did not exit in time (e.g. wedged tearing down its
                # own worker pool). Don't stall the whole process; kill it and
                # move on. TimeoutExpired is not an OSError, so it would
                # otherwise propagate uncaught out of shutdown().
                log.warning(
                    "Compile worker sidecar (pid %s) did not exit within 300s; "
                    "killing it.",
                    self.process.pid,
                )
                self.process.kill()
                self.process.wait()
            if self.log_file:
                self.log_file.close()
        except OSError:
            log.warning("Ignored OSError in pool shutdown", exc_info=True)
        finally:
            with self.futures_lock:
                for future in self.pending_futures.values():
                    if not future.cancel():
                        future.set_exception(RuntimeError("SubprocPool closed"))
                self.pending_futures.clear()
                self._job_compile_id.clear()


class SubprocMain:
    """Communicates with a SubprocPool in the parent process, called by __main__.py"""

    def __init__(
        self,
        pickler: SubprocPickler,
        kind: SubprocKind,
        nprocs: int,
        read_pipe: IO[bytes],
        write_pipe: IO[bytes],
    ) -> None:
        self.pickler = pickler
        self.kind = kind
        self.read_pipe = read_pipe
        self.write_pipe = write_pipe
        self.write_lock = threading.Lock()
        self.nprocs = nprocs
        self.pool: ProcessPoolExecutor | None = None
        self.pool_finalizer: Any | None = None
        self.running = True
        # job_id -> monotonic submit time; scanned by the watchdog thread.
        self._inflight: dict[int, float] = {}
        self._inflight_lock = threading.Lock()
        self._watchdog_stop = threading.Event()

    def main(self) -> None:
        # Phase heartbeats rely on fork inheritance of the shared buffer; spawn
        # workers get no buffer and degrade to duration-only reporting.
        if self.kind == SubprocKind.FORK:
            watchdog.create(self.nprocs)
        self._start_watchdog()
        while True:
            msg_header, job_id, data = _recv_msg(self.read_pipe)
            if msg_header == MsgHeader.JOB:
                self.submit(job_id, data)
            elif msg_header == MsgHeader.WAKEUP:
                self._start_pool()
            elif msg_header == MsgHeader.QUIESCE:
                self._quiesce()
            else:
                return self._shutdown()

    def _quiesce(self) -> None:
        self._shutdown_pool(terminate_workers=False)

    def _shutdown(self) -> None:
        self._watchdog_stop.set()
        with self.write_lock:
            self.running = False
            try:
                _send_msg(self.write_pipe, MsgHeader.SHUTDOWN)
                self.write_pipe.close()
            except BrokenPipeError:
                pass  # parent process already shutdown
            self.read_pipe.close()
        self._shutdown_pool(terminate_workers=True)

    def _shutdown_pool(self, *, terminate_workers: bool) -> None:
        if self.pool is None:
            return

        pool = self.pool
        self.pool = None

        if self.pool_finalizer is not None:
            if terminate_workers:
                self.pool_finalizer.cancel()
            self.pool_finalizer = None

        if terminate_workers:
            # The sidecar is exiting, so do not let ProcessPoolExecutor's
            # interpreter finalization wait for running compiler workers.
            _terminate_process_pool(pool)
        else:
            pool.shutdown(wait=False)

    def submit(self, job_id: int, data: bytes) -> None:
        # Clock starts before _start_pool/_warm_process_pool, so the first job's
        # reported elapsed intentionally includes cold pool creation and the fork
        # of the workers -- that wait is real and worth surfacing.
        with self._inflight_lock:
            self._inflight[job_id] = time.monotonic()
        while self.running:
