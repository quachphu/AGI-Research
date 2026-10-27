           if reshape would happen anyway, i.e.  if the weight tensor is constant, is not contiguous,
           or is using AMX VNNI layout.
        2. Packing the weight tensor into a VNNI-friendly shape. For constant input,
           this is done at the same time as the weight blocking.

        At compile time, the constant weight tensors are blocked and packed. For non-constant tensors (e.g. BMM)
        which will be blocked (non-contiguous or VNNI-layout tensors), the weight tensor is blocked and packed at runtime.

        CppBmmTemplate overrides the methods get_padded_size, and block_weight in order to accommodate
        an additional dimension for the batch size and to determine if the weight tensor should be blocked.
        """
        W = inputs[1]
        new_inputs = list(inputs)
        if cls.is_woq_int4():
            if not (
                len(W.get_size()) == 2
                if isinstance(W, ir.IRNode)
                else len(W.shape) == 2
            ):
                raise AssertionError("expected WOQ int4 weight to be 2D")
            n, k = W.get_size() if isinstance(W, ir.IRNode) else W.shape
        else:
            k, n = W.get_size()[-2:] if isinstance(W, ir.IRNode) else W.shape[-2:]
        _, block_n, _ = micro_gemm.register_blocking
        new_size, padded_n = cls.get_padded_size(n, block_n, k, should_block_weight)
        padding = padded_n - n

        if should_block_weight and not cls.is_woq_int4():
            blocked_w = cls.block_weight(W, new_size, padding)
            new_inputs[1] = cls.pack_vnni_weight(blocked_w, micro_gemm, new_size)
        elif should_block_weight:
            if not cls.is_woq_int4():
                raise AssertionError("expected WOQ int4 in this branch")
            new_inputs[1] = cls.block_weight(W, new_size, padding)
        elif isinstance(W, ir.IRNode):
            # Require W layout to be fixed & contiguous, happens inplace.
            ir.ExternKernel.require_contiguous(W)
            new_inputs[1] = cls._maybe_remove_storage_offset(W)

        if not skip_int8_compensation and _is_int8_gemm(new_inputs):
            BCompensate = None
            x_w_scale = None

            def _get_compensation_node(W, use_int8_fast_compensation_path):
                BCompensate = V.graph.add_tensor_constant(
                    V.graph.constants[W.get_name() + "_BMatrixCompens"],
                    W.get_name() + "_BMatrixCompens",
                )
                x_w_scale = None
                if use_int8_fast_compensation_path:
                    x_w_scale = V.graph.add_tensor_constant(
                        V.graph.constants[W.get_name() + "_x_w_compens"],
                        W.get_name() + "_x_w_compens",
                    )
                return BCompensate, x_w_scale

            if use_int8_fast_compensation_path:
                # new_inputs has been reordered: [x, w, optional[bias], x_scale, x_zp, w_scale, w_zp]
                x_scale = new_inputs[-4]
                x_zp = new_inputs[-3]
                w_scale = new_inputs[-2]
                if isinstance(W, ir.IRNode):
                    BCompensate, x_w_scale = _get_compensation_node(
                        W, use_int8_fast_compensation_path
                    )
                else:
                    # Use the original W, not the blocked_w in new_inputs[1] to calculate BCompensate
                    BCompensate = torch.sum(W.to_dense().to(torch.float), dim=0)  # type: ignore[assignment]
                    if not all(
                        isinstance(item, torch.Tensor)
                        for item in (x_scale, x_zp, w_scale)
                    ):
                        raise AssertionError(
                            "expected x_scale, x_zp, w_scale to be tensors"
                        )
                    BCompensate = BCompensate * x_scale * w_scale * x_zp
                    x_w_scale = x_scale * w_scale
                new_inputs.append(BCompensate)
                new_inputs.append(x_w_scale)
            else:
                if isinstance(W, ir.IRNode):
                    BCompensate, _ = _get_compensation_node(
                        W, use_int8_fast_compensation_path
                    )
                else:
                    # Use the original W, not the blocked_w in new_inputs[1] to calculate BCompensate
                    BCompensate = torch.sum(W.to_dense().to(torch.float), dim=0)  # type: ignore[assignment]
                new_inputs.append(BCompensate)
        return new_inputs, layout

    @staticmethod
    def check_if_block_weight(W, micro_gemm):
        return True

    @classmethod
    def block_weight(cls, W, new_size, padding):
        # These are separated into two methods to allow subclasses to override them separately
        if isinstance(W, ir.IRNode):
            if W.get_name() in V.graph.constants:
                # Create a new buffer, representing the constant blocked tensor
                blocked_w = ir.Buffer(
                    name=W.get_name(),  # Borrow the registered buffer name
                    layout=ir.FixedLayout(
                        W.get_device_or_error(),
                        W.get_dtype(),
                        new_size,
                        ir.FlexibleLayout.contiguous_strides(new_size),
                        0,
                    ),
                )
            else:
                if not isinstance(W, ir.TensorBox):
                    W = ir.TensorBox(W)
                permute_dims = list(range(len(new_size)))
                permute_dims[-2], permute_dims[-3] = permute_dims[-3], permute_dims[-2]
                permute_size = list(new_size)
                permute_size[-2], permute_size[-3] = permute_size[-3], permute_size[-2]
                blocked_w = L.constant_pad_nd(W, (0, padding))
                blocked_w = L.permute(
                    L.view(blocked_w, permute_size),
                    permute_dims,
                )
        else:
            if not isinstance(W, torch.Tensor):
                raise AssertionError(f"expected torch.Tensor, got {type(W)}")
            # Pad the weight tensor and reshape it into a 3D blocked shape
            blocked_size = list(new_size)
            blocked_size[-2], blocked_size[-3] = blocked_size[-3], blocked_size[-2]
            blocked_w = (
                torch.nn.functional.pad(W, (0, padding))  # type: ignore[assignment]
                .reshape(*blocked_size)
                .transpose(-3, -2)
                .contiguous()
            )
        return blocked_w

    @classmethod
    def pack_vnni_weight(cls, W, micro_gemm, new_size):
        # WOQ INT4 weights are reordered in microkernel so do not pack them here
        should_pack = (
            micro_gemm.get_b_layout() != LayoutType.NORMAL
            and not micro_gemm.is_woq_int4()
        )

        # These are separated into two methods to allow subclasses to override them separately
        if isinstance(W, ir.IRNode):
            if isinstance(W, ir.Buffer) and W.get_name() in V.graph.constants:
                return W
            k = new_size[-2]
            if not isinstance(W, ir.TensorBox):
                W = ir.TensorBox(W)
            if should_pack:
                permute_dims = list(range(len(new_size) + 1))
                permute_dims[-1], permute_dims[-2] = permute_dims[-2], permute_dims[-1]
                vnni_size = 4 if micro_gemm.get_b_layout() == LayoutType.VNNI4 else 2
                vnni_view_size = list(new_size)
                vnni_view_size[-2] = k // vnni_size
                vnni_view_size.insert(-1, vnni_size)
                W = L.view(
                    L.permute(L.view(W, vnni_view_size), permute_dims),
                    new_size,
                )
            W = ir.ExternKernel.realize_input(W)
            W = ir.ExternKernel.require_contiguous(W)
            return W
        else:
            k = new_size[-2]
            # Apply VNNI packing to the weight tensor
            if should_pack:
                # TODO: Move VNNI weight packing for non-constant tensors into the template,
                # to improve cache locality and avoid full-tensor copy.
                layout_str = (
                    "VNNI4"
                    if micro_gemm.get_b_layout() == LayoutType.VNNI4
                    else "VNNI2"
                )
                if micro_gemm.get_b_layout() not in [
                    LayoutType.VNNI2,
                    LayoutType.VNNI4,
                ]:
                    raise AssertionError(f"We only support {layout_str} for now")
                vnni_size = 4 if micro_gemm.get_b_layout() == LayoutType.VNNI4 else 2
                if k % vnni_size != 0:
                    raise AssertionError(
                        f"k should be divisible by vnni_size for {layout_str} layout"
                    )
                vnni_view_size = list(new_size)
                vnni_view_size[-2] = k // vnni_size
                vnni_view_size.insert(-1, vnni_size)
                W = W.view(vnni_view_size).transpose(-1, -2).reshape(new_size)
            # normalize stride to be "contiguous_strides" per size
            # this avoids the problems in L.view during template codegen
            new_stride = [1]
            for sz in reversed(W.shape[1:]):
                new_stride.insert(0, new_stride[0] * sz)
            W = W.as_strided(W.shape, new_stride)
            return W

    def get_default_reindexers(self, epilogues):
        return [None] * len(epilogues)

    def get_options(
        self,
        kernel: CppTemplateKernel,
        template_buffer_node: ir.CppTemplateBuffer | None = None,
        flag_template_buffer_has_other_users: bool | None = None,
        epilogue_nodes: list[ir.IRNode] | None = None,
    ) -> dict[str, Any]:
        if len(self.input_nodes) < 2:
            raise AssertionError(
                f"expected >= 2 input nodes, got {len(self.input_nodes)}"
            )

        int8_gemm = self.input_nodes[0].get_dtype() in [torch.uint8, torch.int8]
        x_scale = None
        x_zp = None
        w_scale = None
        w_zp = None
        inp = None
        q_group_size_node = None
        qscale_and_zeros = None
        if int8_gemm:
            X, W = self.input_nodes[0], self.input_nodes[1]
            bias_idx = 2 if self.has_bias else 1
            inp = self.input_nodes[bias_idx] if self.has_bias else None
            x_scale = self.input_nodes[bias_idx + 1]
            x_zp = self.input_nodes[bias_idx + 2]
            w_scale = self.input_nodes[bias_idx + 3]
            w_zp = self.input_nodes[bias_idx + 4]
            Y = self.output_node
        elif self.is_woq_int4():
            X, W = self.input_nodes[0], self.input_nodes[1]
            Y = self.output_node
            q_group_size_node = self.input_nodes[2]
            qscale_and_zeros = self.input_nodes[3]
        else:
            X, W = self.input_nodes[0], self.input_nodes[1]
            Y = self.output_node
            inp = self.input_nodes[2] if self.has_bias else None

        template_buffer_has_other_users = None

        if template_buffer_node is not None:
            # Use the updated prepacked weight buffer
            W = template_buffer_node.inputs[1]
            Y = template_buffer_node

            if flag_template_buffer_has_other_users is None:
                raise AssertionError(
                    "expected flag_template_buffer_has_other_users to be set"
                )
            template_buffer_has_other_users = flag_template_buffer_has_other_users

        template_buffer = Y
        gemm_output_buffer = template_buffer

        epilogues: list[ir.IRNode] = []
        reindexers: list[Callable[[list[Any]], list[Any]] | None] = []
        epilogue_creators: list[Callable[[ir.Buffer], ir.Pointwise]] = []
        fake_buffers: list[ir.Buffer] = []
        Y_aliases: OrderedSet[str] = OrderedSet()

        use_local_acc = (
            self.layout.dtype != torch.float
            or template_buffer_has_other_users
            or int8_gemm
            or self.padded_n != self.n
            or self.maybe_k_slicing()
            or (epilogue_nodes and epilogue_nodes[-1].get_dtype() != self.layout.dtype)
        )

        # TODO(jgong5): for int8 gemm, bias-add is handled outside of gemm template,
        # but we'd better move it here to align with fp.
        if inp is not None and self.beta != 0 and not int8_gemm:
            # add an epilogue for bias add
            def _bias_add_epilogue(buf):
                return create_epilogue_with_attr(
                    buf, "bias_add", other=inp, beta=self.beta, dtype=self.layout.dtype
                )

            epilogue_creators.append(_bias_add_epilogue)

        if self.epilogue_creator is not None:
            epilogue_creators.append(self.epilogue_creator)

        # When the GEMM output buffer is localized but it has users other than the epilogue nodes,
        # we need to copy the value in the GEMM output local buffer to a global buffer.
        def need_copy_from_local_to_global_buffer_epilogue(
            use_local_acc, template_buffer_has_other_users, epilogue_creators
        ):
            # The GEMM output buffer is a global buffer, thus copy is not needed.
            if not use_local_acc:
                return False

            # The possible value of template_buffer_has_other_users is (None, False, True)
            # It is None when generating the gemm template during autotune and it will have value during scheduler codegen.
            # extra copy_from_local_to_global_buffer_epilogue is not needed in either of the below two cases:
            #   1. template_buffer_has_other_users is None (i.e. when doing the codegen during autotune)
            #   2. template_buffer_has_other_users is False, which means it's safe to keep the value in the
            #       GEMM output buffer in local buffer only (no users outside of the epilogues will use its value).
            if not template_buffer_has_other_users:
                return False

            # When bias is not None or self.epilogue_creator is not None,
            # there will be epilogue_creators after the GEMM.
            # The GEMM output buffer is localized while
            # the output buffer of the epilogue_creators is a global buffer.
            if epilogue_creators:
                return False

            return True

        if need_copy_from_local_to_global_buffer_epilogue(
            use_local_acc, template_buffer_has_other_users, epilogue_creators
        ):

            def copy_from_local_to_global_buffer_epilogue(input_buffer: ir.Buffer):
                dtype = self.layout.dtype
                input_loader = input_buffer.make_loader()

                def copy_inner(index):
                    input = input_loader(index)
                    result = ops.to_dtype(input, dtype)
                    return result

                return ir.Pointwise(
                    device=input_buffer.get_device_or_error(),
                    dtype=self.layout.dtype,
                    inner_fn=copy_inner,
                    ranges=input_buffer.get_size(),
                )

            epilogue_creators.append(copy_from_local_to_global_buffer_epilogue)

        # NOTE [How CPP GEMM template epilogues are organized]
        #   gemm_output_buffer
        #     --> zero or more in-template epilogues (created by `epilogue_creators`) -->
        #   template_buffer
        #     --> zero or more out-of-template epilogues (`epilogue_nodes`) -->
        #   Y
        if epilogue_creators:
            if not isinstance(template_buffer, ir.IRNode):
                raise AssertionError(f"expected ir.IRNode, got {type(template_buffer)}")
            gemm_output_name = f"{template_buffer.get_name()}_GemmOut"
            gemm_output_buffer = ir.Buffer(
                name=gemm_output_name,
                # pyrefly: ignore [missing-attribute]
                layout=template_buffer.layout,
            )
            current_input_buffer = gemm_output_buffer
            # In-template epilogues are built over the GEMM output buffer's ranges,
            # whose rank can exceed that of the 2D tiles the template stores into
            # (the BMM template keeps a leading batch dim). Let the template supply
            # the reindexers mapping the 2D tile indices onto those ranges.
            creator_reindexers = self.get_default_reindexers(epilogue_creators)
            for i, creator in enumerate(epilogue_creators):
                if i == len(epilogue_creators) - 1:
                    buffer_name = template_buffer.get_name()
                else:
                    buffer_name = f"{gemm_output_name}_epilogue_{i}"
                epilogues.append(
                    ir.ComputedBuffer(
                        name=buffer_name,
                        # pyrefly: ignore [missing-attribute]
                        layout=template_buffer.layout,
                        data=creator(current_input_buffer),
                    )
                )
                fake_buffers.append(current_input_buffer)
                Y_aliases.add(current_input_buffer.get_name())
                reindexers.append(creator_reindexers[i])
                if i < len(epilogue_creators) - 1:
                    current_input_buffer = ir.Buffer(
                        name=buffer_name,
                        # pyrefly: ignore [missing-attribute]
                        layout=template_buffer.layout,
                    )

        if not isinstance(Y, (ir.Buffer, ir.ReinterpretView)):
            raise AssertionError(
                f"expected ir.Buffer or ir.ReinterpretView, got {type(Y)}"
            )
        Y_2d: ir.Buffer | ir.ReinterpretView = Y

        if epilogue_nodes:
            if not template_buffer_has_other_users:
                if not isinstance(template_buffer, ir.IRNode):
                    raise AssertionError(
                        f"expected ir.IRNode, got {type(template_buffer)}"
                    )
                Y_aliases.add(template_buffer.get_name())
            epilogues.extend(epilogue_nodes)
            if Y.get_numel() != epilogues[-1].get_numel():
                raise AssertionError(
                    f"expected Y numel == epilogue numel, got {Y.get_numel()} "
                    f"and {epilogues[-1].get_numel()}"
                )
            Y = cast(ir.Buffer, epilogues[-1])
            if not isinstance(template_buffer, ir.Buffer):
                raise AssertionError(f"expected ir.Buffer, got {type(template_buffer)}")
            Y_2d, reindexers = gen_2d_view_of_epilogue_buf(
                Y,
                template_buffer,
                epilogue_nodes,
                reindexers,
                default_reindexers=self.get_default_reindexers(epilogue_nodes),
            )

        output_dtype, compute_dtype = get_gemm_template_output_and_compute_dtype(
            X.get_dtype()
        )
        micro_gemm = create_micro_gemm(
            f"{kernel.kernel_name}_micro_gemm",
            self.m,
            self.n,
            self.k,
            input_dtype=X.get_dtype(),
            input2_dtype=W.get_dtype(),
            output_dtype=output_dtype,
            compute_dtype=compute_dtype,
            alpha=self.alpha,
            num_threads=self.num_threads,
            use_ref=not self.is_woq_int4(),
            q_group_size=self.q_group_size(),
        )
        if micro_gemm is None:
            raise AssertionError("expected micro_gemm to be created, got None")
        micro_gemm.use_local_vnni_blocking(not self.should_block_weights)
        if self.register_blocking != micro_gemm.register_blocking:
            raise AssertionError(
                f"expected register_blocking == {micro_gemm.register_blocking}, "
                f"got {self.register_blocking}"
            )
        self.log_blockings()
        if isinstance(micro_gemm, CppMicroGemmAMX):
            counters["inductor"]["cpp_micro_gemm_amx_counter"] += 1
        if isinstance(micro_gemm, CppMicroBrgemm):
            counters["inductor"]["cpp_micro_brgemm_counter"] += 1

        L1_cache_size = torch.cpu.get_capabilities().get(
            "l1d_cache_size", 0
        )  # per core cache size in Bytes
        if L1_cache_size <= 0:
            raise AssertionError(f"Expect L1_cache_size > 0 but got {L1_cache_size}")

        L2_cache_size = torch.cpu.get_capabilities().get(
            "l2_cache_size", 0
        )  # per core cache size in Bytes
        if L2_cache_size <= 0:
            raise AssertionError(f"Expect L2_cache_size > 0 but got {L2_cache_size}")

        options = dict(
            X=X,
            W=W,
            inp=inp,
            Y=Y,
            N=self.n,
            K=self.k,
            PADDED_N=self.padded_n,
            GemmOut=gemm_output_buffer,
            aliases={alias: Y.get_name() for alias in Y_aliases},
            beta=self.beta,
            alpha=self.alpha,
            num_threads=self.num_threads,
            micro_gemm=micro_gemm,
            is_dynamic_M=self.is_dynamic_M,
            template=self,
            kernel=kernel,
            export_declaration=get_export_declaration(),
            epilogue_nodes=epilogues,
            reindexers=reindexers,
            Y_2d=Y_2d,
            use_local_acc=use_local_acc,
            maybe_k_slicing=self.maybe_k_slicing(),
            x_scale=x_scale,
            x_zp=x_zp,
            w_scale=w_scale,
            w_zp=w_zp,
            acc_buf_dtype=torch.int32 if int8_gemm else torch.float,
            DTYPE_TO_CPP=DTYPE_TO_CPP,
            L1_cache_size=L1_cache_size,
            L2_cache_size=L2_cache_size,
            config=config,
            fake_buffers=fake_buffers,
            is_woq_int4=self.is_woq_int4(),
            q_group_size=q_group_size_node,
            qscale_and_zeros=qscale_and_zeros,
            cpu_count=os.cpu_count(),
        )
        return options

    def is_int8_woq_gemm_small_m_dim(
        self,
        X: ir.ReinterpretView,
        W: ir.ReinterpretView,
        N,
        K,
        micro_gemm,
    ):
        """Use SMALL_M_GEMM_TEMPLATE"""
        return (
            isinstance(micro_gemm, CppMicroGemmFP32Vec)
            and is_int8_woq_gemm_small_m_dim_corner_case(
                micro_gemm, X.get_size()[0], N, K
            )
            and X.get_dtype() is torch.bfloat16
            and W.get_dtype() is torch.int8
        )

    def get_kernel_prefix_name(self) -> str:
        return f"gemm_m{self.m}n{self.n}k{self.k}"

    def render(  # type: ignore[override, return]
        self,
        kernel: CppTemplateKernel,
        template_buffer_node: ir.CppTemplateBuffer | None = None,
        flag_template_buffer_has_other_users: bool | None = None,
        epilogue_nodes: list[ir.IRNode] | None = None,
        **kwargs,
    ) -> str:
        options = self.get_options(
            kernel=kernel,
            template_buffer_node=template_buffer_node,
            flag_template_buffer_has_other_users=flag_template_buffer_has_other_users,
            epilogue_nodes=epilogue_nodes,
        )
        self.render_options = options

        with contextlib.ExitStack() as stack:
            for buf in options["fake_buffers"]:
                stack.enter_context(
                    patch.object(V.graph, "get_dtype", self._fake_get_dtype(buf))
                )
            if not options["is_dynamic_M"] and self.is_int8_woq_gemm_small_m_dim(
                options["X"],
                options["W"],
                options["N"],
                options["K"],
                options["micro_gemm"],
            ):
                template_str = SMALL_M_GEMM_TEMPLATE
            else:
                template_str = GEMM_TEMPLATE
            return self._template_from_string(template_str).render(**options)

    def codegen_blocks(
        self,
        num_threads,
        N,
        K,
        micro_gemm,
        is_dynamic_M,
        kernel,
        GemmOut,
        config,
        L1_cache_size,
        L2_cache_size,
        X,
        W,
    ):
        options = dict(
            num_threads=num_threads,
            N=N,
            K=K,
            micro_gemm=micro_gemm,
            is_dynamic_M=is_dynamic_M,
            kernel=kernel,
            GemmOut=GemmOut,
            config=config,
            L1_cache_size=L1_cache_size,
            L2_cache_size=L2_cache_size,
            template=self,
            X=X,
            W=W,
            is_woq_int4=self.is_woq_int4(),
        )
        template_str = GEMM_TEMPLATE_INIT_BLOCKING_BASIC_BLOCK
        if not (
            not is_dynamic_M
            and self.is_int8_woq_gemm_small_m_dim(X, W, N, K, micro_gemm)
        ):
            template_str += GEMM_TEMPLATE_INIT_BLOCKING_EXTENDED
        return self._template_from_string(template_str).render(options)

    def codegen_microkernel_def(self):
        return self._template_from_string(GEMM_TEMPLATE_MICROKERNEL_DEF).render(
            self.render_options
        )

    def codegen_gemm_stub_def(self):
        microkernel = self.codegen_microkernel_def()
        return microkernel + self._template_from_string(GEMM_TEMPLATE_STUB_DEF).render(
            self.render_options
        )

    def codegen_multi_threads_params(self):
        return self._template_from_string(GEMM_TEMPLATE_MULTI_THREADS_PARAMS).render()

    def codegen_single_thread_params(self, is_dynamic_M):
        options = dict(
            is_dynamic_M=is_dynamic_M,
        )
        return self._template_from_string(GEMM_TEMPLATE_SINGLE_THREAD_PARAMS).render(
            options
        )

    def codegen_m_loop_params(self):
        return self._template_from_string(GEMM_TEMPLATE_M_LOOP_PARAMS).render()

    def codegen_n_loop_params(self):
        return self._template_from_string(GEMM_TEMPLATE_N_LOOP_PARAMS).render()

    @classmethod
    def is_woq_int4(cls):
        return False

    @classmethod
    def q_group_size(cls):
        return None


class CppWoqInt4GemmTemplateMeta(type):
    def __getitem__(cls, q_group_size):
        class CppWoqInt4GemmTemplateInstance(CppGemmTemplate):
            def __init__(
                self,
                *args,
                **kwargs,
            ) -> None:
                super().__init__(
                    *args,
                    **kwargs,
                )

            @classmethod
            def is_woq_int4(cls):
                return True

            @classmethod
            def q_group_size(cls):
                return q_group_size

            @staticmethod
            def check_if_block_weight(W, micro_gemm):
                # For WOQ INT4, weight is already packed
                # However, for AMX microkernel, we want to change the blocking of weight
                from .cpp_micro_gemm import CppMicroGemmWoQInt4Amx

                return isinstance(micro_gemm, CppMicroGemmWoQInt4Amx)

            @classmethod
            def block_weight(cls, W, new_size, padding):
                # This method is called only if AMX microkernels are used.
                # In this case, we unpack and repack weight so that block_n=32
                # the format of packed weight is described here:
                # https://github.com/pytorch/pytorch/blob/32eee8ed225d9f10fbbcb38c24b8b44c24c0c97c/aten/src/ATen/native/cpu/int4mm_kernel.cpp#L583
                if isinstance(W, ir.IRNode):
                    # in this case, we do nothing
                    ir.ExternKernel.require_contiguous(W)
                    blocked_w = W
                else:
                    # in this case, we unpack and repack weight
                    if not isinstance(W, torch.Tensor):
                        raise AssertionError(f"expected torch.Tensor, got {type(W)}")
                    if W.dim() != 2:
                        raise AssertionError(f"expected 2D weight, got {W.dim()}D")
                    N = W.size(0)
                    K = W.size(-1) * 2
                    G = cls.q_group_size()
                    # x and qscales_and_zeros are in bfloat16 instead of float to use the optimized kernel
                    # so that the unpacking process is faster
                    x = torch.eye(K).bfloat16()
                    # Here we use scale=1 and qzero=8 because we want to unpack weight
                    # without dequantizing it. The qzero here is 8 instead of 0 because
                    # int4 values are converted to [-7, 8] in the _weight_int4pack_mm_for_cpu kernel:
                    # https://github.com/pytorch/pytorch/blob/32eee8ed225d9f10fbbcb38c24b8b44c24c0c97c/aten/src/ATen/native/cpu/int4mm_kernel.cpp#L95
                    qscales_and_zeros = (
                        torch.tensor([1.0, 8.0])
                        .bfloat16()
                        .expand(K // G, N, 2)
                        .contiguous()
                    )
                    # shape: [K, N]
                    unpacked_w = torch.ops.aten._weight_int4pack_mm_for_cpu(
                        x,
                        W,
                        G,
                        qscales_and_zeros,
                    ).to(torch.uint8)
                    block_n = 32
                    # shape: [N // block_n, K, block_n]
                    w_blocked = (
                        unpacked_w.view(K, N // block_n, block_n)
                        .permute(1, 0, 2)
                        .contiguous()
                    )
                    # pack 2 int4 -> 1 int8
                    # block_n: [a0, a1, ..., a15, b0, b1, ..., b15]
                    # -> [(a0 & 0xf) | (b0 << 4), (a1 & 0xf) | (b1 << 4), ...]
                    # shape: [N // block_n, K, 2, block_n // 2]
                    w_blocked = w_blocked.view(N // block_n, K, 2, block_n // 2)
                    # shape: [N // block_n, K, block_n // 2]
                    w_blocked_packed = (w_blocked[:, :, 0, :] & 0xF) | (
                        w_blocked[:, :, 1, :] << 4
                    )
                    # shape: [N, K // 2]
                    blocked_w = w_blocked_packed.view(N, K // 2)

                return blocked_w

        return CppWoqInt4GemmTemplateInstance


class CppWoqInt4GemmTemplate(metaclass=CppWoqInt4GemmTemplateMeta):
    pass
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import dataclasses
import os
import re
import sys
from itertools import count, zip_longest
from typing import Any, cast
from typing_extensions import Self

import sympy

import torch
from torch import dtype as torch_dtype
from torch._inductor.codecache import get_cpp_wrapper_cubin_path_name
from torch._inductor.runtime.runtime_utils import dynamo_timed
from torch.utils._ordered_set import OrderedSet

from .. import config
from ..codecache import CudaKernelParamCache
from ..ir import (
    GraphPartitionSignature,
    TensorBox,
    TMADescriptorExperimental,
    TMADescriptorStable,
)
from ..runtime.hints import (
    InductorMeta,
    TRITON_DEFAULT_BLOCK_SIZES,
    TRITON_DEFAULT_RSPLIT,
    TRITON_DEFAULT_RSPLIT_SIZE,
    TritonMeta,
)
from ..stream_utils import (
    AOTI_SUPPORTED_STREAM_OP_NAMES,
    AOTI_UNSUPPORTED_STREAM_OP_REASONS,
    get_stream_name,
)
from ..utils import (
    cache_on_self,
    DeferredLineBase,
    get_gpu_type,
    GPU_ALIGN_BYTES,
    IndentedBuffer,
    make_codegen_buffer,
    XPU_KERNEL_FORMAT,
)
from ..virtualized import V
from .aoti_hipify_utils import maybe_hipify_code_wrapper
from .common import get_device_op_overrides, TritonScratchWorkspace
from .cpp_utils import cexpr
from .cpp_wrapper_cpu import CppWrapperCpu
from .multi_kernel import MultiKernelCall
from .triton_utils import should_unwrap_unspec_arg
from .wrapper import PythonWrapperCodegen, SymbolicCallArg


_cpp_string_literal_escapes = {
    "\\": "\\\\",
    '"': '\\"',
    "\n": "\\n",
    "\t": "\\t",
    "\r": "\\r",
}
_cpp_string_literal_pattern = re.compile(r'["\\\n\t\r]')


def cpp_string_literal(s: str) -> str:
    escaped = _cpp_string_literal_pattern.sub(
        lambda match: _cpp_string_literal_escapes[match.group(0)], s
    )
    return f'"{escaped}"'


def generate_aoti_kernel_config_header(kernel_names: list[str]) -> str:
    """Generate a C header defining macros for each lazy-compiled kernel.

    Called after the JIT first-pass runs and populates CudaKernelParamCache.
    The AOTI compilation includes this header so that LazyKernelCompileResult
    structs get compile-time-initialized with the autotuned values.
    """

    def braced(values: list[int]) -> str:
        return "{" + ", ".join(str(v) for v in values) + "}"

    buf = IndentedBuffer()
    buf.splice("""
        #pragma once
        // Auto-generated kernel configurations for AOTInductor lazy compile.
    """)

    for kernel_name in kernel_names:
        params = CudaKernelParamCache.get(kernel_name)
        if params is None:
            raise RuntimeError(
                "When autotune_at_compile_time is False, AOTInductor generates"
                " both JIT code and AOT code. They are expected to have exactly"
                f" the same kernels. However, AOT code contains kernels, {kernel_names},"
                " that is not in the JIT code."
            )

        macro_prefix = kernel_name.upper()
        cubin_path = cpp_string_literal(params[get_cpp_wrapper_cubin_path_name()])
        mangled_name = cpp_string_literal(params["mangled_name"])
        num_warps = params["num_warps"]
        shared_mem = params["shared_mem"]

        # params["config"] is already a dict (from config_to_dict in CachingAutotuner)
        config_dict = params.get("config") or {}

        # For combo/foreach kernels, merge default_config
        inductor_meta = params.get("inductor_meta") or {}
        combo_grid_meta = inductor_meta.get("combo_grid_meta")
        default_config = (
            combo_grid_meta.get("default_config") if combo_grid_meta else None
        )
        if default_config:
            config_dict = {**default_config, **config_dict}

        config_index: int | None = None
        grid_type = inductor_meta.get("grid_type")
        if grid_type == "PrecomputedGrid":
            precomputed_grids = inductor_meta.get("precomputed_grids", [])
            for idx, entry in enumerate(precomputed_grids):
                entry_config = entry.get("config", {})
                if all(config_dict.get(k) == v for k, v in entry_config.items()):
                    config_index = idx
                    break

        # Per-subkernel block sizes for combo kernels, or single-element lists.
        num_kernels = combo_grid_meta.get("num_kernels", 1) if combo_grid_meta else 1
        if num_kernels > 1 and "XBLOCK_0" in config_dict:
            xblocks = [
                config_dict.get(f"XBLOCK_{i}", TRITON_DEFAULT_BLOCK_SIZES["XBLOCK"])
                for i in range(num_kernels)
            ]
            yblocks = [
                config_dict.get(f"YBLOCK_{i}", TRITON_DEFAULT_BLOCK_SIZES["YBLOCK"])
                for i in range(num_kernels)
            ]
            zblocks = [
                config_dict.get(f"ZBLOCK_{i}", TRITON_DEFAULT_BLOCK_SIZES["ZBLOCK"])
                for i in range(num_kernels)
            ]
            r0blocks = [
                config_dict.get(f"R0_BLOCK_{i}", TRITON_DEFAULT_BLOCK_SIZES["R0_BLOCK"])
                for i in range(num_kernels)
            ]
        else:
            xblocks = [config_dict.get("XBLOCK", TRITON_DEFAULT_BLOCK_SIZES["XBLOCK"])]
            yblocks = [config_dict.get("YBLOCK", TRITON_DEFAULT_BLOCK_SIZES["YBLOCK"])]
            zblocks = [config_dict.get("ZBLOCK", TRITON_DEFAULT_BLOCK_SIZES["ZBLOCK"])]
            r0blocks = [
                config_dict.get("R0_BLOCK", TRITON_DEFAULT_BLOCK_SIZES["R0_BLOCK"])
            ]
        rsplit = config_dict.get("RSPLIT", TRITON_DEFAULT_RSPLIT)
        rsplit_size = config_dict.get("RSPLIT_SIZE", TRITON_DEFAULT_RSPLIT_SIZE)
        ci = config_index if config_index is not None else -1
        gs = params.get("global_scratch", -1) or -1
        ps = params.get("profile_scratch", -1) or -1

        buf.writeline("")
        buf.splice(f"""
            // Kernel: {kernel_name}
            #define {macro_prefix}_CUBIN_PATH {cubin_path}
            #define {macro_prefix}_MANGLED_NAME {mangled_name}
            #define {macro_prefix}_NUM_WARPS {num_warps}
            #define {macro_prefix}_SHARED_MEM {shared_mem}
            #define {macro_prefix}_XBLOCKS {braced(xblocks)}
            #define {macro_prefix}_YBLOCKS {braced(yblocks)}
            #define {macro_prefix}_ZBLOCKS {braced(zblocks)}
            #define {macro_prefix}_R0BLOCKS {braced(r0blocks)}
            #define {macro_prefix}_RSPLIT {rsplit}
            #define {macro_prefix}_RSPLIT_SIZE {rsplit_size}
            #define {macro_prefix}_CONFIG_INDEX {ci}
            #define {macro_prefix}_GLOBAL_SCRATCH {gs}
            #define {macro_prefix}_PROFILE_SCRATCH {ps}
        """)

    return buf.getvalue()


TRITON_SIGNATURE_TO_CPP = {
    "i32": "int32_t",
    "i64": "int64_t",
    "fp32": "float",
    "fp64": "double",
}


def signature_is_tma_desc(sig: str | None) -> bool:
    """Check if a Triton signature represents a TMA descriptor."""
    if not sig:
        return False
    if sig == "nvTmaDesc":
        return True
    if sig.startswith("tensordesc<"):
        return True
    return False


def _unpack_tma_descriptor_args(var_name: str, sig_type: str) -> list[str]:
    """Unpack a StableTMADescriptor into kernel launch args.

    Given a variable name holding a StableTMADescriptor and its tensordesc<...>
    signature, returns the list of pointer args: &var.m, &var.block_shape[i]...,
    &var.strides[i]...
    """
    match = re.match(r"tensordesc<[^[]*\[([^\]]*)\]", sig_type)
    if match is None:
        raise AssertionError(f"Cannot parse tensordesc signature: {sig_type}")
    ndim = match.group(1).count(",") + 1
    result = [f"&{var_name}.m"]
    for i in range(ndim):
        result.append(f"&{var_name}.block_shape[{i}]")
    for i in range(ndim):
        result.append(f"&{var_name}.strides[{i}]")
    return result


class _LazyTritonCompileKickoffLine(DeferredLineBase):
    def __init__(self, lazy_kernel_names: list[str], line: str):
        super().__init__(line)
        self.lazy_kernel_names = lazy_kernel_names

    def __call__(self) -> str | None:
        return self.line if self.lazy_kernel_names else None

    def _new_line(self, line: str) -> _LazyTritonCompileKickoffLine:
        return _LazyTritonCompileKickoffLine(self.lazy_kernel_names, line)


@dataclasses.dataclass
class DeferredTritonCallWrapper:
    """
    When using cpp wrapper, GPU kernel load and launch needs to wait for Triton kernels
    to be tuned and stored as cubin files, so use a deferred generating the final wrapper around
    the triton kernel until right before the prefix is written.
    """

    wrapper_name: str
    kernel_name: str
    kernel_name_to_body: dict[str, str]
    arg_types: list[Any]
    triton_meta: TritonMeta | None = None
    inductor_meta: dict[str, Any] | None = None
    tma_tensor_args: dict[str, str] = dataclasses.field(default_factory=dict)

    @cache_on_self
    def _get_tma_args(self) -> dict[str, str]:
        """Get mapping of TMA descriptor arg names to their signature types."""
        triton_meta = self.triton_meta or {}
        signature = triton_meta.get("signature", {})
        for name, sig_type in signature.items():
            if sig_type == "nvTmaDesc":
                raise RuntimeError(
                    f"nvTmaDesc (experimental TMA API) is not supported in lazy compile "
                    f"for arg '{name}'. Use the stable tensordesc API instead."
                )
        return {
            name: sig_type
            for name, sig_type in signature.items()
            if isinstance(sig_type, str) and sig_type.startswith("tensordesc<")
        }

    def _get_cpp_param_type(
        self, name: str, arg_type: Any, signature: dict[str, str] | None = None
    ) -> str:
        """Get the C++ parameter declaration for a given arg type."""
        if isinstance(arg_type, (torch_dtype, UnwrapUnspecArg)):
            # TMA descriptors need non-const references since their fields
            # are passed as void* pointers to kernel launch args
            if signature and signature_is_tma_desc(signature.get(name)):
                return f"{name}_type_& {name}"
            return f"const {name}_type_& {name}"
        elif issubclass(arg_type, (SymbolicCallArg, sympy.Expr, int)):
            return f"int64_t {name}"
        elif arg_type is float:
            return f"float {name}"
        elif arg_type is bool:
            return f"bool {name}"
        else:
            raise ValueError(f"Unexpected arg type {arg_type}")

    def _write_wrapper_signature(
        self,
        prefix: IndentedBuffer,
        wrapper: CppWrapperGpu,
        arg_names: list[str],
        arg_types: list[Any] | None = None,
        signature: dict[str, str] | None = None,
    ) -> None:
        """Write the wrapper function signature including template and parameters."""
        if arg_types is None:
            arg_types = self.arg_types

        template_types = [
            f"typename {name}_type_"
            for name, arg_type in zip(arg_names, arg_types)
            if isinstance(arg_type, (torch_dtype, UnwrapUnspecArg))
        ]
        if template_types:
            prefix.writeline_jit(f"template <{', '.join(template_types)}>")
        prefix.writeline_aot(
            f"template <{', '.join([*template_types, 'typename kernels_type_'])}>"
        )

        cubin_dir_param = "const std::optional<std::string>& cubin_dir_ = std::nullopt"
        kernels_param = "kernels_type_& kernels_"

        shared_params = [
            self._get_cpp_param_type(name, arg_type, signature)
            for name, arg_type in zip(arg_names, arg_types)
        ]
        shared_params.append("int32_t device_idx_")
        shared_params.append(
            maybe_hipify_code_wrapper(
                f"{wrapper.device_codegen.cpp_stream_type()} stream_"
            )
        )

        def emit(writer, params):
            for p in params[:-1]:
                writer(f"{p},")
            writer(params[-1])

        prefix.writeline(f"static __attribute__((noinline)) void {self.wrapper_name}(")
        with prefix.indent():
            emit(prefix.writeline_jit, [*shared_params, cubin_dir_param])
            emit(prefix.writeline_aot, [*shared_params, kernels_param, cubin_dir_param])
        prefix.writeline("){")

    def generate(self, wrapper: CppWrapperGpu):
        """
        Generate the GPU kernel definition, as well as load and launch code.
        """
        prefix = wrapper.prefix
        if self.kernel_name.startswith("multi_kernel_"):
            # MultiKernel will select one kernel after running the autotune block
            self.kernel_name = MultiKernelCall.lookup_choice(self.kernel_name)

        # Defer compilation to runtime if autotune_at_compile_time is False (JIT only).
        # AOTI lazy-compile emission is wired up later in the stack.
        if config.triton.autotune_at_compile_time is False:
            return self.generate_lazy(wrapper)

        params = CudaKernelParamCache.get(self.kernel_name)
        if not params:
            raise AssertionError(
                f"CudaKernelParamCache not populated for {self.kernel_name}"
            )
        def_args = params["def_args"]
        arg_types = self.arg_types
        inductor_meta = params["inductor_meta"]

        if "extra_launcher_args" in inductor_meta and len(def_args) > len(arg_types):
            # extra_launcher_args should already be in def_args
            if len(def_args) != len(arg_types) - len(
                inductor_meta["extra_launcher_args"]
            ):
                raise AssertionError(
                    "expected len(def_args) == len(arg_types) - "
                    f"len(extra_launcher_args), got {len(def_args)}"
                )
            arg_types = arg_types + [SymbolicCallArg] * len(
                inductor_meta["extra_launcher_args"]
            )

        if not V.graph.aot_mode:
            prefix.writeline(
                maybe_hipify_code_wrapper(
                    f"static {wrapper.device_codegen.cpp_kernel_type()} {self.kernel_name} = nullptr;"
                )
            )
            kernel_var_name = self.kernel_name
        else:
            kernel_var_name = f"kernels_.{self.kernel_name}"

        # Write wrapper function signature
        self._write_wrapper_signature(prefix, wrapper, def_args, arg_types)

        with prefix.indent():
            if V.graph.aot_mode:
                # Emit the original Triton kernel for debugging purposes
                prefix.writeline("/*")
                prefix.splice(self.kernel_name_to_body[self.kernel_name])
                prefix.writeline("*/")
            self.generate_grid(prefix, inductor_meta, params)
            self.generate_load_kernel(prefix, kernel_var_name, params)
            self.generate_launch_kernel(prefix, wrapper, kernel_var_name, params)
        prefix.writeline("}")

        if not config.aot_inductor.embed_kernel_binary:
            # Ensure the cubin file is included in the package
            V.graph.wrapper_code.additional_files.append(
                params[get_cpp_wrapper_cubin_path_name()]
            )

    def _resolve_lazy_arg_names(self) -> tuple[list[str], list[str]]:
        """Compute wrapper and kernel arg names from triton_meta signature.

        Returns (wrapper_arg_names, kernel_arg_names) where:
        - wrapper_arg_names: params accepted by the C++ wrapper function
        - kernel_arg_names: params passed to the GPU kernel launch (non-constexpr only)
        """
        if self.triton_meta is None:
            raise AssertionError(
                f"triton_meta is required for lazy compile of {self.kernel_name}"
            )
        signature = self.triton_meta.get("signature", {})
        inductor_meta = self.inductor_meta or {}
        extra_launcher_args_count = len(inductor_meta.get("extra_launcher_args", []))
        tma_tensor_args = self.tma_tensor_args
        num_tma_tensor_args = len(tma_tensor_args)

        # Matches internal config params like XBLOCK, RSPLIT_SIZE, and their
        # per-subkernel variants like XBLOCK_0, YBLOCK_1.
        internal_config_re = re.compile(r"(?:BLOCK|RSPLIT_SIZE|RSPLIT)(?:_\d+)?$")
        # Declared constexpr params (tl.constexpr in kernel signature) are excluded
        # from arg_types for user-defined kernels, while value-based constexpr params
        # (e.g. numel=1, arg=None) are still in arg_types.
        declared_constexpr_names = OrderedSet(
            inductor_meta.get("declared_constexpr_names", [])
        )
        wrapper_arg_names = []
        kernel_arg_names = []
        for name, sig_type in signature.items():
            if internal_config_re.search(name):
                continue
            if sig_type != "constexpr":
                kernel_arg_names.append(name)
            if name not in declared_constexpr_names:
                wrapper_arg_names.append(name)

        num_wrapper_args = (
            len(self.arg_types) - extra_launcher_args_count - num_tma_tensor_args
        )
        if num_wrapper_args != len(wrapper_arg_names):
            raise AssertionError(
                f"Mismatch between ({num_wrapper_args}) arg_types and "
                f"{len(wrapper_arg_names)} wrapper_arg_names for {self.kernel_name}."
            )

        # Append grid args: passed to wrapper. Kernel args will handle grids separately.
        for i in range(extra_launcher_args_count):
            wrapper_arg_names.append(f"_grid_{i}")

        # Add TMA tensor args after grid args
        if tma_tensor_args:
            sig_tma_keys = list(self._get_tma_args().keys())
            if list(tma_tensor_args.keys()) != sig_tma_keys:
                raise AssertionError(
                    f"TMA tensor args order mismatch for {self.kernel_name}: "
                    f"{list(tma_tensor_args.keys())} vs signature order {sig_tma_keys}"
                )
        for desc_name in tma_tensor_args:
            wrapper_arg_names.append(f"_tma_tensor_{desc_name}")

        return wrapper_arg_names, kernel_arg_names

    def _generate_lazy_grid(self, prefix: IndentedBuffer) -> None:
        """Generate grid computation code for lazy-compiled kernels."""
        kernel_name = self.kernel_name
        grid_type = self.inductor_meta.get("grid_type") if self.inductor_meta else None

        # For PrecomputedGrid, generate switch statement on config_index
        if grid_type == "PrecomputedGrid":
            if self.inductor_meta is None:
                raise AssertionError("inductor_meta is required for PrecomputedGrid")
            precomputed_grids = self.inductor_meta.get("precomputed_grids", [])
            extra_launcher_args = self.inductor_meta.get("extra_launcher_args", [])

            switch_cases = []
            for idx, entry in enumerate(precomputed_grids):
                cpp_grids = list(entry.get("cpp", ["1L", "1L", "1L"]))
                # Replace internal arg names with C++ parameter names
                # e.g., _launcher_s0 -> _grid_0
                for i, arg_name in enumerate(extra_launcher_args):
                    cpp_grids = [g.replace(arg_name, f"_grid_{i}") for g in cpp_grids]
                g0 = cpp_grids[0]
                g1 = cpp_grids[1] if len(cpp_grids) > 1 else "1"
                g2 = cpp_grids[2] if len(cpp_grids) > 2 else "1"
                switch_cases.append(
                    f"case {idx}: grid_0 = {g0}; grid_1 = {g1}; grid_2 = {g2}; break;"
                )
            switch_cases.append("default: grid_0 = 1; grid_1 = 1; grid_2 = 1; break;")
            switch_body = "\n                        ".join(switch_cases)

            prefix.splice(
                f"""\
                uint32_t grid_0, grid_1, grid_2;
                switch ({kernel_name}_result.config_index) {{
                    {switch_body}
                }}
                if (grid_0 == 0) return;
                """
            )
        else:
            from ..runtime.triton_heuristics import GridExpr

            grid = GridExpr.from_meta_lazy(
                cast("InductorMeta | None", self.inductor_meta), kernel_name
            )
            for line in grid.prefix:
                prefix.writeline(line)

            prefix.splice(
                f"""\
                uint32_t grid_0 = {grid.x_grid};
                uint32_t grid_1 = {grid.y_grid};
                uint32_t grid_2 = {grid.z_grid};
                if (grid_0 == 0) return;
                """
            )

    def _generate_lazy_tma_args(
        self,
        prefix: IndentedBuffer,
        call_args_str: str,
        kernel_arg_names: list[str],
        tma_arg_names: OrderedSet[str],
        signature: dict[str, str],
    ) -> str:
        """Unpack TMA descriptor args into kernel launch args."""
        for arg_name in kernel_arg_names:
            if arg_name in tma_arg_names:
                tma_parts = _unpack_tma_descriptor_args(arg_name, signature[arg_name])
                tma_str = ", ".join(tma_parts)
                call_args_str = (
                    f"{call_args_str}, {tma_str}" if call_args_str else tma_str
                )
        return call_args_str

    def _generate_lazy_scratch(
        self,
        prefix: IndentedBuffer,
        wrapper: CppWrapperGpu,
        call_args_str: str,
    ) -> str:
        """Generate scratch space allocations with runtime-known sizes."""
        kernel_name = self.kernel_name
        dtype_str = wrapper.codegen_dtype(torch.uint8)
        device_type, _ = wrapper.codegen_device(torch.device(get_gpu_type())).split(
            ", "
        )
        device_ptr_type = wrapper.device_codegen.cpp_device_ptr()
        # Triton reports per-CTA scratch via kernel.metadata.global_scratch_size;
        # the kernel writes its slot at offset (pid * scratch_size). Scale by the
        # full launch grid so concurrent CTAs don't collide.
        grid_extent = "static_cast<int64_t>(grid_0) * grid_1 * grid_2"
        for scratch_name in ("global_scratch", "profile_scratch"):
            size_expr = f"{kernel_name}_result.{scratch_name}"
            var = f"{scratch_name}_ptr"
            prefix.splice(
                maybe_hipify_code_wrapper(
                    f"""\
                int64_t {var}_numel = {size_expr} * {grid_extent};
                RAIIAtenTensorHandle {var}_tensor;
                {device_ptr_type} {var} = allocate_scratch_tensor<{device_ptr_type}>(
                    {var}_numel, {dtype_str}, {device_type}, device_idx_, {var}_tensor);
            """
                )
            )
            call_args_str += f", &{var}"
        return call_args_str

    def _generate_lazy_launch(
        self,
        prefix: IndentedBuffer,
        wrapper: CppWrapperGpu,
        wrapper_arg_names: list[str],
        kernel_arg_names: list[str],
    ) -> None:
        """Generate kernel launch code for lazy-compiled kernels."""
        kernel_name = self.kernel_name
        signature = (self.triton_meta or {}).get("signature", {})
        tma_tensor_args = self.tma_tensor_args
        num_tma_tensor_args = len(tma_tensor_args)

        # wrapper_arg_names may include grid and TMA tensor args at the end;
        # only the leading portion maps 1:1 to kernel signature params.
        num_signature_args = len(wrapper_arg_names) - num_tma_tensor_args
        inductor_meta = self.inductor_meta or {}
        num_signature_args -= len(inductor_meta.get("extra_launcher_args", []))

        arg_type_lookup = dict(
            zip(wrapper_arg_names, self.arg_types[:num_signature_args])
        )

        # Identify TMA args — they are already passed as StableTMADescriptor params,
        # so we just unpack them directly (no need to reconstruct from tensors).
        tma_arg_names = OrderedSet(self._get_tma_args().keys())

        # Non-TMA args go through generate_args_decl
        non_tma_arg_names = [n for n in kernel_arg_names if n not in tma_arg_names]
        non_tma_arg_types = [
            arg_type_lookup[n] for n in non_tma_arg_names if n in arg_type_lookup
        ]
        non_tma_arg_sigs = [signature.get(n) for n in non_tma_arg_names]

        call_args_str = wrapper.generate_args_decl(
            prefix,
            non_tma_arg_names,
            non_tma_arg_types,
            non_tma_arg_sigs,
        )

        call_args_str = self._generate_lazy_tma_args(
            prefix, call_args_str, kernel_arg_names, tma_arg_names, signature
        )
        call_args_str = self._generate_lazy_scratch(prefix, wrapper, call_args_str)

        common_launch_args = (
            f"grid_0, grid_1, grid_2,"
            f" {kernel_name}_result.num_warps,"
            f" {kernel_name}_result.shared_mem,"
            f" kernel_args_, stream_"
        )
        # stream_ comes from the generated wrapper signature on both JIT and
        # AOTI sides.
        launch_kernel_args = [
            "grid_0",
            "grid_1",
            "grid_2",
            f"{kernel_name}_result.num_warps",
            f"{kernel_name}_result.shared_mem",
        ]

        # kernel_args_ is consumed by both JIT and AOT launchKernel calls.
        prefix.writeline(f"void* kernel_args_[] = {{{call_args_str}}};")
        enable_kernel_profile = config.cpp.enable_kernel_profile and sys.platform in [
            "linux",
            "win32",
        ]
        prefix.writeline_jit(f"launchKernel({kernel_name}, {common_launch_args});")
        if enable_kernel_profile:
            profile_arg_types = [arg_type_lookup.get(n) for n in kernel_arg_names]
            profile_arg_sigs = [signature.get(n) for n in kernel_arg_names]
            aot_profile = IndentedBuffer(initial_indent=prefix._indent)
            self.generate_profiled_launch_kernel(
                aot_profile,
                f"kernels_.{kernel_name}",
                kernel_arg_names,
                profile_arg_types,
                profile_arg_sigs,
                [
                    f"kernels_.{kernel_name}",
                    *launch_kernel_args,
                    "kernel_args_",
                    "stream_",
                ],
                num_warps=f"{kernel_name}_result.num_warps",
                shared_mem=f"{kernel_name}_result.shared_mem",
            )
            prefix.splice_aot(aot_profile)
        else:
            prefix.writeline_aot(
                f"launchKernel(kernels_.{kernel_name}, {common_launch_args});"
            )

    def generate_lazy(self, wrapper: CppWrapperGpu):
        """
        Generate dual-wrapper-mode C++ code for lazy Triton kernel compilation.

        DualIndentedBuffer routes lines into separate JIT and AOTI sources:
        - JIT side: embeds Triton source, compiles at runtime, autotunes
          with real inputs.
        - AOTI side: uses a compile-time-initialized LazyKernelCompileResult
          from a config header generated after the JIT first-pass.

        Grid computation and kernel launch are shared between both sides
        via the LazyKernelCompileResult struct.
        """
        prefix = wrapper.prefix
        kernel_name = self.kernel_name
        macro_prefix = kernel_name.upper()

        # Track kernel names for parallel initialization (JIT only)
        wrapper._lazy_kernel_names.append(kernel_name)

        # Include TMA helpers if any args use TMA descriptors
        tma_signature_types = self._get_tma_args()
        if tma_signature_types:
            wrapper.write_tma_descriptor_helpers_once()

        kernel_type = maybe_hipify_code_wrapper(
            wrapper.device_codegen.cpp_kernel_type()
        )

        # JIT-only: static CUfunction and embedded Triton source
        prefix.writeline_jit(f"static {kernel_type} {kernel_name} = nullptr;")
        kernel_source_str = self.kernel_name_to_body.get(kernel_name, "")
        kernel_body = f'R"TRITON(\n{kernel_source_str}\n)TRITON"'
        prefix.writeline_jit(
            f"static const char* {kernel_name}_source = {kernel_body};"
        )

        # LazyKernelCompileResult: JIT fills at runtime; AOTI uses compile-time
        # init from the config header generated after the first pass.
        prefix.writeline_jit(f"static LazyKernelCompileResult {kernel_name}_result;")
        prefix.splice_aot(
            f"""\
            static LazyKernelCompileResult {kernel_name}_result = {{
                {macro_prefix}_CUBIN_PATH,
                {macro_prefix}_MANGLED_NAME,
                {macro_prefix}_NUM_WARPS,
                {macro_prefix}_SHARED_MEM,
                {macro_prefix}_XBLOCKS,
                {macro_prefix}_YBLOCKS,
                {macro_prefix}_ZBLOCKS,
                {macro_prefix}_R0BLOCKS,
                {macro_prefix}_RSPLIT,
                {macro_prefix}_RSPLIT_SIZE,
                {macro_prefix}_CONFIG_INDEX,
                {macro_prefix}_GLOBAL_SCRATCH,
                {macro_prefix}_PROFILE_SCRATCH,
            }};
            """
        )

        wrapper_arg_names, kernel_arg_names = self._resolve_lazy_arg_names()
        signature = (self.triton_meta or {}).get("signature", {})

        # kernels_type_/kernels_ are routed to the AOTI buffer; if prefix is a
        # plain IndentedBuffer (pure JIT lazy compile) those writes are dropped.
        self._write_wrapper_signature(
            prefix,
            wrapper,
            wrapper_arg_names,
            self.arg_types,
            signature,
        )

        # Build autotune args - for TMA, pass tensors instead of descriptors.
        # Only iterate over signature params and grid args, not the trailing
        # TMA tensor params (those are only in the C++ wrapper signature).
        tma_tensor_args = self.tma_tensor_args
        num_autotune_args = len(wrapper_arg_names) - len(tma_tensor_args)
        autotune_arg_list = []
        # Track which args need scalar extraction for the autotune call.
        # UnwrapUnspecArg args are 0-dim tensors in C++ that Triton expects
        # as Python scalars; we use codegen_tensor_item to extract them.
        scalar_extractions: list[tuple[str, str, torch_dtype]] = []
        for idx, name in enumerate(wrapper_arg_names[:num_autotune_args]):
            if name in tma_signature_types:
                autotune_arg_list.append(f"_tma_tensor_{name}")
            elif isinstance(self.arg_types[idx], UnwrapUnspecArg):
                scalar_var = f"_autotune_scalar_{name}"
                scalar_extractions.append((name, scalar_var, self.arg_types[idx].dtype))
                autotune_arg_list.append(scalar_var)
            else:
                autotune_arg_list.append(name)
        autotune_args = ", ".join(autotune_arg_list)

        with prefix.indent():
            # First-call initialization: JIT lazy compiles, AOTI loads cubin

            # JIT: lazy compile with autotuning on first invocation.
            # Build into temp buffer to avoid DualIndentedBuffer dispatch.
            jit_init = IndentedBuffer(initial_indent=prefix._indent)
            jit_init.writeline(f"if ({kernel_name} == nullptr) {{")
            with jit_init.indent():
                for tensor_name, scalar_var, dtype in scalar_extractions:
                    wrapper.codegen_tensor_item(
                        dtype, tensor_name, scalar_var, indented_buffer=jit_init
                    )
                jit_init.splice(
                    f"""\
                    {kernel_name}_result = runTritonKernelWithAutotune(
                        _module_pending_kernels, "{kernel_name}", stream_, {autotune_args});

                    {kernel_name} = loadKernel(
                        {kernel_name}_result.cubin_path,
                        {kernel_name}_result.mangled_name,
                        {kernel_name}_result.shared_mem);

                    // First invocation already ran the kernel, so return early
                    return;
                    """
                )
            jit_init.writeline("}")
            prefix.splice_jit(jit_init)

            # AOTI: load precompiled cubin from compile-time-initialized result.
            loaded_modules_arg = (
                ",\n                        &kernels_.loaded_modules_"
                if V.graph.device_type != "xpu"
                else ""
            )
            aoti_init = IndentedBuffer(initial_indent=prefix._indent)
            aoti_init.writeline(f"if (kernels_.{kernel_name} == nullptr) {{")
            with aoti_init.indent():
                aoti_init.splice(
                    f"""\
                    kernels_.{kernel_name} = loadKernel(
                        {kernel_name}_result.cubin_path,
                        {kernel_name}_result.mangled_name,
                        {kernel_name}_result.shared_mem,
                        cubin_dir_{loaded_modules_arg});
                    """
                )
            aoti_init.writeline("}")
            prefix.splice_aot(aoti_init)

            # Shared: grid computation and launch using result struct
            self._generate_lazy_grid(prefix)
            self._generate_lazy_launch(
                prefix,
                wrapper,
                wrapper_arg_names,
                kernel_arg_names,
            )
        prefix.writeline("}")

    def generate_grid(
        self,
        prefix: IndentedBuffer,
        inductor_meta: dict[str, Any],
        params: dict[str, Any],
    ):
        from ..runtime.triton_heuristics import GridExpr

        grid = GridExpr.from_meta(
            cast("InductorMeta", inductor_meta), params["config"], mode="cpp"
        )
        for line in grid.prefix:
            prefix.writeline(line)
        prefix.splice(
            f"""\
            uint32_t grid_0 = {grid.x_grid};
            uint32_t grid_1 = {grid.y_grid};
            uint32_t grid_2 = {grid.z_grid};
            """
        )
        prefix.writeline("if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;")

    def generate_load_kernel(self, prefix, kernel_var_name, params):
        prefix.writeline(f"if ({kernel_var_name} == nullptr) {{")
        with prefix.indent():
            embed_kernel_args = [f"__{params['inductor_meta']['kernel_name']}_start"]
            if torch.xpu.is_available():
                # XPU needs the end address of the kernel to calculate the size of the kernel binary.
                embed_kernel_args.append(
                    f"__{params['inductor_meta']['kernel_name']}_end"
                )

            if V.graph.aot_mode and config.aot_inductor.embed_kernel_binary:
                load_kernel_args = [
                    *embed_kernel_args,
                    cpp_string_literal(params["mangled_name"]),
                    str(params["shared_mem"]),
                ]
                if torch.xpu.is_available():
                    is_spv = "true" if XPU_KERNEL_FORMAT == "spv" else "false"
                    if config.aot_inductor.emit_multi_arch_kernel:
                        is_spv = "true"
                    load_kernel_args.append(is_spv)
            else:
                load_kernel_args = [
                    cpp_string_literal(params[get_cpp_wrapper_cubin_path_name()]),
                    cpp_string_literal(params["mangled_name"]),
                    str(params["shared_mem"]),
                    "cubin_dir_",
                ]

            # In AOTI mode on CUDA/HIP, pass the loaded_modules_ vector so
            # CUmodule handles are tracked and can be unloaded on destruction,
            # preventing GPU code object leaks. XPU is excluded because its
            # loadKernel returns std::unique_ptr<sycl::kernel> and manages
            # cleanup via RAII.
            if V.graph.aot_mode and V.graph.device_type != "xpu":
                load_kernel_args = load_kernel_args + ["&kernels_.loaded_modules_"]

            prefix.writeline(
                f"{kernel_var_name} = loadKernel({', '.join(load_kernel_args)}); "
            )
        prefix.writeline("}")

    def generate_launch_kernel(self, prefix, wrapper, kernel_var_name, params):
        """
        Generate the GPU kernel launching code.
        This is where all the call args are sorted out and generated.
        If enable_kernel_profile is enabled, all args related information would be packed in this function.
        """
        triton_meta = params["triton_meta"]
        if len(self.arg_types) != len(params["def_args"]):
            raise AssertionError((self.arg_types, params["def_args"]))
        arg_type_lookup = dict(zip(params["def_args"], self.arg_types))
        # difference between Python and C++ wrapper: C++ wrapper strips out equal_to_1 constants
        call_args = [
            name for name in params["call_args"] if name not in triton_meta["constants"]
        ]
        arg_types = [arg_type_lookup[name] for name in call_args]
        arg_signatures = [triton_meta["signature"][name] for name in call_args]
        num_ctas = params.get("config", {}).get("num_ctas", 1)
        scratch_spaces = {
            name: params[name] * num_ctas
            for name in ["global_scratch", "profile_scratch"]
            if params.get(name, None) is not None
        }
        call_args_str = wrapper.generate_args_decl(
            prefix,
            call_args,
            arg_types,
            arg_signatures,
            scratch_spaces=scratch_spaces,
        )
        prefix.writeline(f"void* kernel_args_[] = {{{call_args_str}}};")
        num_warps = str(params["num_warps"])
        shared_mem = str(params["shared_mem"])
        launch_kernel_args = [
            kernel_var_name,
            "grid_0",
            "grid_1",
            "grid_2",
            num_warps,
            shared_mem,
            "kernel_args_",
            "stream_",
        ]

        enable_kernel_profile = config.cpp.enable_kernel_profile and sys.platform in [
            "linux",
            "win32",
        ]
        if enable_kernel_profile:
            self.generate_profiled_launch_kernel(
                prefix,
                kernel_var_name,
                call_args,
                arg_types,
                arg_signatures,
                launch_kernel_args,
                num_warps=num_warps,
                shared_mem=shared_mem,
            )
        else:
            prefix.writeline(f"launchKernel({', '.join(launch_kernel_args)});")

    def generate_profiled_launch_kernel(
        self,
        prefix: IndentedBuffer,
        kernel_var_name: str,
        call_args: list[str],
        arg_types: list[Any],
        arg_signatures: list[str | None],
        launch_kernel_args: list[str],
        num_warps: str,
        shared_mem: str,
    ) -> None:
        """Wrap a kernel launch in an AOTI record_function profiling scope."""
        normalized_kernel_name = re.sub(r"[^a-zA-Z0-9_]", "_", kernel_var_name)
        prefix.writeline("{")
        with prefix.indent():
            prefix.writelines(
                [
                    f"std::unordered_map<std::string, C10IValueHandle> kwargs_{normalized_kernel_name};",
                    "",
                ]
            )
            # Add launch args info
            record_launch_kernel_args = [
                ("grid_0", "grid_0"),
                ("grid_1", "grid_1"),
                ("grid_2", "grid_2"),
                ("num_warps", num_warps),
                ("shared_mem", shared_mem),
            ]
            for k, v in record_launch_kernel_args:
                arg_name = f"{normalized_kernel_name}_{k}"
                prefix.writelines(
                    [
                        f"// Create c10::IValue for {k}",
                        f"C10IValueHandle tmp_{arg_name};",
                        f"aoti_torch_int64_to_ivalue({v}, &tmp_{arg_name});",
                        f"RAIIC10IValueHandle RAII_{arg_name}(tmp_{arg_name});",
                        f'kwargs_{normalized_kernel_name}.emplace("{k}", RAII_{arg_name});',
                    ]
                )

            # Add input info (This copies the logic from args_decl)
            curr_arg_id = -1
            total_args = []
            ordered_argsname = []

            def write_dummy_scalar_ivalue(arg_name):
                # We only care about the shape, therefore we create a dummy scalar here.
                prefix.writelines(
                    [
                        f"// Create c10::IValue for arg_{curr_arg_id}",
                        f"C10IValueHandle tmp_{arg_name};",
                        f"aoti_torch_int64_to_ivalue(0, &tmp_{arg_name});",
                        f"RAIIC10IValueHandle RAII_{arg_name}(tmp_{arg_name});",
                    ]
                )
                # pyrefly: ignore [bad-argument-type]
                total_args.append(f"tmp_{arg_name}")

            def process_args_for_input_shape(arg, arg_type, arg_signature=None):
                nonlocal curr_arg_id
                curr_arg_id += 1
                arg_name = f"{normalized_kernel_name}_arg_{curr_arg_id}"
                # ignore tma descriptors, as host-side TMA descriptors need
                # to be passed to the compiled Triton kernel by value
                if isinstance(arg_type, UnwrapUnspecArg) and not signature_is_tma_desc(
                    arg_signature
                ):
                    write_dummy_scalar_ivalue(arg_name)
                elif isinstance(arg_type, torch_dtype) and not signature_is_tma_desc(
                    arg_signature
                ):
                    # This is an at::Tensor.
                    prefix.writelines(
                        [
                            f"// Create c10::IValue for arg_{curr_arg_id}",
                            f"C10IValueHandle tmp_{arg_name};",
                            f"aoti_torch_tensor_to_ivalue({arg}, &tmp_{arg_name});",
                            f"RAIIC10IValueHandle RAII_{arg_name}(tmp_{arg_name});",
                        ]
                    )
                    # pyrefly: ignore [bad-argument-type]
                    total_args.append(f"tmp_{arg_name}")
                elif (
                    isinstance(arg_type, type(SymbolicCallArg))
                    and arg_signature is not None
                    and arg_signature in TRITON_SIGNATURE_TO_CPP
                ) or arg_type in (sympy.Integer, int, sympy.Float, float):
                    write_dummy_scalar_ivalue(arg_name)
                elif arg_signature and arg_signature.startswith("tensordesc<"):
                    # Skip tma related args
                    pass
                else:
                    write_dummy_scalar_ivalue(arg_name)

            # Add input name and shape information
            for arg, arg_type, arg_signature in zip_longest(
                call_args, arg_types, arg_signatures
            ):
                # pyrefly: ignore [bad-argument-type]
                ordered_argsname.append(f'"{arg}"')
                process_args_for_input_shape(arg, arg_type, arg_signature)

            # Add input name into kwargs
            name_var = f"{normalized_kernel_name}_input_names"
            prefix.writelines(
                [
                    "// Create c10::IValue for input names",
                    f"C10IValueHandle tmp_{name_var};",
                    f"std::vector<const char*> {name_var}({{{', '.join(ordered_argsname)}}});",
                    f"aoti_torch_strlist_to_ivalue({name_var}.data(), {len(ordered_argsname)}, &tmp_{name_var});",
                    f"RAIIC10IValueHandle RAII_{name_var}(tmp_{name_var});",
                    f'kwargs_{normalized_kernel_name}.emplace("Input Args", RAII_{name_var});',
                ]
            )

            inputs_info_ = f"{normalized_kernel_name}_inputs_info_"
            # We pass in the non-RAII handles, since C10 doesn't automatically free them.
            # The RAII will make sure they get freed when they are out of scope.
            tmp_args = ",".join(total_args)
            prefix.writelines(
                [
                    "// Aggregate all c10::IValue for inputs",
                    f"std::vector<C10IValueHandle> {inputs_info_}({{{tmp_args}}});",
                ]
            )

            # Start recording Function
            prefix.writelines(
                [
                    "",
                    (
                        "torch::aot_inductor::RAIIAtenRecordFunctionHandle "
                        f"record_{normalized_kernel_name}_"
                        f'("{kernel_var_name}", '
                        f"reinterpret_cast<IValueMapHandle>(&kwargs_{normalized_kernel_name}), "
                        f"{inputs_info_});"
                    ),
                    "",
                    f"launchKernel({', '.join(launch_kernel_args)});",
                ]
            )
        prefix.writeline("}")


class CppWrapperGpu(CppWrapperCpu):
    """
    Generates cpp wrapper for running on GPU and calls CUDA kernels
    """

    def __init__(self) -> None:
        self.device = get_gpu_type()
        self.device_codegen = get_device_op_overrides(self.device)
        super().__init__()
        self.grid_id = count()
        self._kernel_name_to_body: dict[str, str] = {}
        self._triton_call_wrappers: dict[str, DeferredTritonCallWrapper] = {}
        self.autotune_input_prefix = "_REAL_AUTOTUNE_INPUT"
        self._lazy_kernel_names: list[str] = []
        self._declared_aux_stream_slots: OrderedSet[int] = OrderedSet()
        self._aoti_current_stream_guard_declared = False
        self._aoti_stream_helpers_emitted = False

    def generate_debug_sync(self, buffer):
        if self.device == "cuda":
            # The fbcode JIT cpp_wrapper CUDA build links only the CUDA driver
            # (libcuda), not libcudart, so the runtime cudaDeviceSynchronize symbol
            # is undefined at dlopen -> use the driver-API cuCtxSynchronize there.
            # On ROCm the driver-context sync hipCtxSynchronize returns
            # hipErrorNotSupported at runtime, so keep the runtime cudaDeviceSynchronize
            # (which hipifies to hipDeviceSynchronize and IS linked in the ROCm build).
            if torch.version.hip is not None:
                buffer.writeline(
                    maybe_hipify_code_wrapper(
                        "AOTI_RUNTIME_CUDA_CHECK(cudaDeviceSynchronize());"
                    )
                )
            else:
                buffer.writeline("CUDA_DRIVER_CHECK(cuCtxSynchronize());")
            return

        raise NotImplementedError(
            f"triton debug sync is not supported with {self.device} cpp_wrapper"
        )

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: str | None,
        parent_wrapper: PythonWrapperCodegen | None,
        partition_signatures: GraphPartitionSignature | None = None,
    ):
        # TODO - support subgraph codegen by lifting functions. Check the
        # comment at CppWrapperCpu `codegen_subgraph` function.
        return CppWrapperGpu()

    def write_header(self):
        if V.graph.is_const_graph and not V.graph.is_dual_wrapper_mode:
            # We do not write header for constant graph, it will be written by main module.
            return

        super().write_header()
        kernel_driver = maybe_hipify_code_wrapper(self.device_codegen.kernel_driver())
        if V.graph.is_const_graph and V.graph.is_dual_wrapper_mode:
            # For a dual-wrapper-mode const graph, only the standalone JIT
            # output needs this header content. The AOTI const body is spliced
            # into the main AOTI source, which has its own kernel driver.
            # super().write_header() early-returns for const graphs before it
            # can call add_device_include, so emit the JIT device include here;
            # otherwise the kernel driver's CUfunction/CUmodule/uint32_t types
            # have no declaring header and fail to compile under -nostdinc.
            for device in V.graph.device_types:
                if device != "meta":
                    self.header.splice_jit(self.get_device_include_path_jit(device))
            self.header.splice_jit(kernel_driver)
        else:
            self.header.splice(kernel_driver)

    def _generate(self, is_inference):
        # Per-Run()-function state, reset each generation. Do NOT reset
        # _aoti_stream_helpers_emitted here: the helper structs are spliced into
        # the file-level header once per instance, and _generate can run more
        # than once against the same header (resetting it re-splices and yields
        # a C++ redefinition).
        self._declared_aux_stream_slots.clear()
        self._aoti_current_stream_guard_declared = False
        return super()._generate(is_inference)

    @cache_on_self
    def write_tma_descriptor_helpers_once(self):
        self.header.splice(self.device_codegen.tma_descriptor_helpers())

    def write_get_raw_stream(self, device_idx: int, graph_name: str) -> str:
        # Pure AOTI receives the stream as a function parameter. JIT and
        # dual-wrapper-mode code use an explicit stream variable so the shared kernel
        # call arguments are valid for the JIT entry point.
        if V.graph.aot_mode and not V.graph.is_dual_wrapper_mode:
            return "stream"

        name = f"stream{device_idx}"
        # In dual-wrapper mode, the JIT stream is declared at the entry function
        # prologue (see _codegen_entry_impl_prologue) so it stays in scope
        # across all kernel call sites.
        if V.graph.is_dual_wrapper_mode:
            return name

        self.writeline(
            maybe_hipify_code_wrapper(
                f"{self.device_codegen.cpp_stream_type()} {name};"
            )
        )
        self.writeline(
            f"AOTI_TORCH_ERROR_CODE_CHECK({self.device_codegen.aoti_get_stream()}({device_idx}, (void**)&{name}));"
        )
        return name

    def _ensure_aoti_stream_helpers_emitted(self) -> None:
        if self._aoti_stream_helpers_emitted:
            return
        # The stream/event helpers in streams.h are CUDA-specific (cudaEvent_t,
        # cudaStream_t, cudaEventRecord, ...). Guarding here on the device type
        # prevents the CUDA-only symbols from being emitted into XPU generated
        # code, where SYCL in-order queues handle event ordering implicitly.
        if self.device == "xpu":
            return
        self._aoti_stream_helpers_emitted = True
        with open(
            os.path.join(os.path.dirname(__file__), "aoti_runtime", "streams.h")
        ) as f:
            self.header.splice(maybe_hipify_code_wrapper(f.read()))
        self.header.splice(
            """
            namespace {

            static thread_local torch::aot_inductor::AOTIPerThreadEventCache
                _aoti_event_cache;
            static thread_local torch::aot_inductor::AOTIPerThreadStreamCache
                _aoti_aux_stream_cache;

            }  // namespace
            """
        )

    def codegen_stream_info_prologue(
        self,
        code: IndentedBuffer,
        num_streams: int,
        stream_idx_to_user_obj_idx: dict[int, int],
    ) -> None:
        if num_streams <= 1:
            return
        if not V.graph.aot_mode:
            raise NotImplementedError(
                "Multi-stream cpp_wrapper codegen is only supported for AOTI."
            )
        self._ensure_aoti_stream_helpers_emitted()
        code.writeline(
            f"_aoti_aux_stream_cache.ensure({num_streams}, this->device_idx_, stream);"
        )
        if not self._aoti_current_stream_guard_declared:
            code.writeline(
                f"std::unique_ptr<{V.graph.device_ops.cpp_aoti_stream_guard()}> "
                "_aoti_current_stream_guard;"
            )
            self._aoti_current_stream_guard_declared = True

        stream_type = self.device_codegen.cpp_stream_type()
        for i in range(1, num_streams):
            if i in self._declared_aux_stream_slots:
                continue
            code.writeline(
                maybe_hipify_code_wrapper(
                    f"{stream_type} {get_stream_name(i)} = "
                    f"_aoti_aux_stream_cache.get({i}, this->device_idx_, stream);"
                )
            )
            self._declared_aux_stream_slots.add(i)

    def codegen_enter_cuda_stream_context(
        self, code: IndentedBuffer, stream_idx: int
    ) -> None:
        if stream_idx == 0:
            return
        code.writeline(
            "_aoti_current_stream_guard = "
            f"std::make_unique<{V.graph.device_ops.cpp_aoti_stream_guard()}>("
            f"{self._stream_expr_for_idx(stream_idx)}, this->device_idx_);"
        )

    def codegen_exit_cuda_stream_context(self, code: IndentedBuffer) -> None:
        code.writeline("_aoti_current_stream_guard.reset();")

    def _stream_expr_for_idx(self, stream_idx: int) -> str:
        if stream_idx == 0:
            return "stream"
        return f"_aoti_aux_stream_cache.get({stream_idx}, this->device_idx_, stream)"

    def _emit_stream_op_inline(self, kernel_name: str | None, args: list[str]) -> bool:
        if kernel_name is None or not V.graph.aot_mode:
            return False
        if self.device == "xpu":
            return False
        if kernel_name in AOTI_UNSUPPORTED_STREAM_OP_REASONS:
            raise NotImplementedError(
                f"{kernel_name} is not supported in AOTI cpp_wrapper. "
                f"{AOTI_UNSUPPORTED_STREAM_OP_REASONS[kernel_name]}"
            )
        op = AOTI_SUPPORTED_STREAM_OP_NAMES.get(kernel_name)
        if op is None:
            return False

        def _parse_idx(arg: str) -> int:
            return int(arg.rstrip("Ll"))

        self._ensure_aoti_stream_helpers_emitted()
        event_idx = _parse_idx(args[0])
        if op == "record_event":
            stream_idx = _parse_idx(args[1])
            self.writeline(
                maybe_hipify_code_wrapper(
                    "AOTI_RUNTIME_CUDA_CHECK(cudaEventRecord("
                    f"_aoti_event_cache.get({event_idx}, this->device_idx_), "
                    f"{self._stream_expr_for_idx(stream_idx)}));"
                )
            )
            return True
        if op == "wait_event":
            stream_idx = _parse_idx(args[1])
            self.writeline(
                maybe_hipify_code_wrapper(
                    "AOTI_RUNTIME_CUDA_CHECK(cudaStreamWaitEvent("
                    f"{self._stream_expr_for_idx(stream_idx)}, "
                    f"_aoti_event_cache.get({event_idx}, this->device_idx_), 0));"
                )
            )
            return True
        if op == "synchronize_event":
            self.writeline(
                maybe_hipify_code_wrapper(
                    "AOTI_RUNTIME_CUDA_CHECK(cudaEventSynchronize("
                    f"_aoti_event_cache.get({event_idx}, this->device_idx_)));"
                )
            )
            return True
        return False

    def _generate_extern_kernel_alloc_helper(self, extern_kernel, args):
        kernel_name = getattr(extern_kernel, "python_kernel_name", None)
        if self._emit_stream_op_inline(kernel_name, args):
            if V.extern_kernel_nodes:
                V.extern_kernel_nodes.pop()
            return
        super()._generate_extern_kernel_alloc_helper(extern_kernel, args)

    def generate_fallback_kernel_with_runtime_lookup(
        self,
        buf_name,
        python_kernel_name,
        get_args,
        op_overload,
        raw_args,
        outputs,
    ):
        if self._emit_stream_op_inline(python_kernel_name, list(get_args())):
            if V.extern_kernel_nodes:
                V.extern_kernel_nodes.pop()
            return
        super().generate_fallback_kernel_with_runtime_lookup(
            buf_name,
            python_kernel_name,
            get_args,
            op_overload,
            raw_args,
            outputs,
        )

    def get_autotuning_input_name(self, idx):
        return f"{self.autotune_input_prefix}_{idx}"

    def codegen_inputs(self):
        # See Note: [Input Alignment handling in Inductor]
        #
        # JIT Inductor does not guard on input alignment. It relies on copy_misaligned_inputs to
        # copy misaligned inputs to aligned buffers. For AOTInductor, we need to do the same in cpp.

        if config.is_fbcode():
            # TODO: This is added because FC. Remove this once the newly added shim symbols,
            # e.g. aoti_torch_clone_preserve_strides, have landed
            return super().codegen_inputs()

        if V.graph.aot_mode and V.graph.inputs_to_check:
            for idx in V.graph.inputs_to_check:
                input_name = V.graph.graph_input_names[idx]
                if input_name not in V.graph.graph_inputs:
                    raise AssertionError(f"{input_name} not found in graph inputs")
                value = V.graph.graph_inputs[input_name]
                if not isinstance(value, TensorBox):
                    raise AssertionError(
                        f"{input_name} is expected to be tensor but found as {type(value)}"
                    )
                warn_msg = (
                    f"Input {idx} was compiled as {GPU_ALIGN_BYTES}-bytes aligned, "
                    "but it is not aligned at run time. Copying to an aligned tensor "
                    "to guarantee correctness, but expect a performance hit."
                )
                alignment_check = f"""
                    if ((reinterpret_cast<std::uintptr_t>({input_name}.data_ptr()) & ({GPU_ALIGN_BYTES} -1)) != 0) {{
                        AOTI_TORCH_WARN("{warn_msg}");
                        AtenTensorHandle {input_name}_aligned;
                        AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_clone_preserve_strides({input_name}, &{input_name}_aligned));
                        {input_name} = std::move(RAIIAtenTensorHandle({input_name}_aligned));
                    }}
                    """
                self.prefix.splice_aot(alignment_check)

        super().codegen_inputs()

    def _define_kernel_helper(
        self,
        kernel_name: str,
        kernel_body: str,
        metadata: str | None = None,
        gpu: bool = True,
        cpp_definition: str | None = None,
    ):
        if gpu:
            self._kernel_name_to_body[kernel_name] = kernel_body
            if config.triton.autotune_at_compile_time:
                # Call PythonWrapperCodegen to create the autotune code block
                PythonWrapperCodegen._define_kernel_helper(
                    self, kernel_name, kernel_body, metadata, gpu, cpp_definition
                )
        else:
            return CppWrapperCpu._define_kernel_helper(
                self, kernel_name, kernel_body, metadata, gpu, cpp_definition
            )

    def generate(self, is_inference):
        with dynamo_timed("CppWrapperGpu.generate", log_pt2_compile_event=True):
            return super().generate(is_inference)

    def _codegen_entry_impl_prologue(self):
        super()._codegen_entry_impl_prologue()
        # ensure_triton_kernel_compiles_started() is JIT-only; AOTI has no
        # Python-dependent lazy compile flow.
        self.prefix.writeline_jit(
            _LazyTritonCompileKickoffLine(
                self._lazy_kernel_names, "ensure_triton_kernel_compiles_started();"
            )
        )
        # In dual-wrapper mode, hoist the JIT-side stream declaration to the entry
        # function prologue. Kernel calls run inside KernelContextGuard
        # scopes, so a per-call declaration would be scoped to the guard
        # and unavailable to other kernel calls in the same function.
        if V.graph.is_dual_wrapper_mode:
            stream_type = maybe_hipify_code_wrapper(
                self.device_codegen.cpp_stream_type()
            )
            get_stream = self.device_codegen.aoti_get_stream()
            for device_idx in sorted(V.graph.device_idxs):
                name = f"stream{device_idx}"
                self.prefix.writeline_jit(f"{stream_type} {name};")
                self.prefix.writeline_jit(
                    f"AOTI_TORCH_ERROR_CODE_CHECK("
                    f"{get_stream}({device_idx}, (void**)&{name}));"
                )

    def finalize_prefix(self):
        """Define the triton kernels now that autotuning is finished"""
        old_prefix = self.prefix  # new content should go at start of prefix

        # Generating triton kernel callers can modify the prefix (cached dtypes),
        # so do this before running finalize_prefix(), but put the generated code
        # after the finalize_prefix() code.
        triton_prefix = make_codegen_buffer()
        with self._target_buf("prefix", triton_prefix):
            for kernel in self._triton_call_wrappers.values():
                self.prefix.writeline("\n")
                kernel.generate(self)

            # Generate parallel kernel compilation initialization function.
            # JIT-only since AOTI has no Python dependency.
            if self._lazy_kernel_names:
                start_compile_body = (
                    "loadLazyCompileFuncs();\n"
                    "    _module_pending_kernels = PyDict_New();\n"
                    '    AOTI_TORCH_CHECK(_module_pending_kernels, "Failed to create pending kernels dict");\n'
                    "    "
                    + "\n    ".join(
                        f'startKernelCompile(_module_pending_kernels, "{name}", {name}_source);'
                        for name in self._lazy_kernel_names
                    )
                )
                self.include_extra_header("mutex")
                self.prefix.writeline_jit("")
                self.prefix.splice_jit(
                    f"""\
// Start parallel compilation of all Triton kernels.
static inline void start_all_triton_kernel_compiles() {{
    {start_compile_body}
}}

// inductor_entry_impl calls this on every forward;
// std::call_once makes the first call do the work.
static std::once_flag _triton_kernel_compile_init_flag;
static inline void ensure_triton_kernel_compiles_started() {{
    std::call_once(_triton_kernel_compile_init_flag, [] {{
        start_all_triton_kernel_compiles();
    }});
}}
"""
                )

        self.prefix = make_codegen_buffer()
        super().finalize_prefix()

        self.prefix.splice(triton_prefix)
        self.prefix.writeline("\n")
        self.prefix.splice(old_prefix)

    def generate_tma_descriptor(self, desc):
        self.write_tma_descriptor_helpers_once()

        if isinstance(desc, TMADescriptorExperimental):
            self._generate_experimental_tma_descriptor(desc)
        else:
            if not isinstance(desc, TMADescriptorStable):
                raise AssertionError(f"expected TMADescriptorStable, got {type(desc)}")
            self._generate_stable_tma_descriptor(desc)

    def _generate_experimental_tma_descriptor(self, desc):
        # generate data pointer for the source tensor
        source = self.generate_args_decl(
            code=self,
            call_args=[self.val_to_arg_str(desc.tensor)],
            arg_types=[desc.tensor.get_dtype()],
            arg_signatures=[None],
            # these args are passed to initNDTMADescriptor, which is NOT a triton kernel
            is_triton_kernel=False,
        )

        desc_name = desc.name
        self.writeline(f"alignas(64) CUtensorMap {desc_name};")

        # `source` is in the form of `&var_x`, where `var_x` is the data pointer
        # (CUdeviceptr); we dereference `source` and cast to `void*` to pass to
        # the data pointer of the source tensor to the helper function
        # `init{1,2}DTMADescriptor`
        ptr = f"reinterpret_cast<void*>(*({source}))"
        dims = ", ".join(self.val_to_arg_str(dim) for dim in desc.dims)
        block_dims = ", ".join(self.val_to_arg_str(dim) for dim in desc.block_dims)
        element_size = self.val_to_arg_str(desc.element_size)
        fn = f"init{desc.rank}DTMADescriptor"
        args = f"&{desc_name}, {ptr}, {dims}, {block_dims}, {element_size}"
        self.writeline(f"{fn}({args});")

    def _generate_stable_tma_descriptor(self, desc):
        source = self.generate_args_decl(
            code=self,
            call_args=[self.val_to_arg_str(desc.tensor)],
            arg_types=[desc.tensor.get_dtype()],
            arg_signatures=[None],
            # these args are passed to initNDTMADescriptor, which is NOT a triton kernel
            is_triton_kernel=False,
        )

        desc_name = desc.name
        # Pack the relevant information into a StableTMADescriptor struct.
        # See [Note: AOTI TMA Stable handling] for more details.
        self.writeline(f"alignas(64) StableTMADescriptor {desc_name};")

        def fill_array(name, values):
            for i, val in enumerate(values):
                self.writeline(f"{name}[{i}] = {val};")

        ptr = f"reinterpret_cast<void*>(*({source}))"
        rank = len(desc.tensor.get_size())

        fill_array(f"{desc_name}.block_shape", desc.block_shape)
        fill_array(f"{desc_name}.global_shape", desc.tensor.get_size())
        fill_array(f"{desc_name}.strides", desc.tensor.get_stride())

        element_size = self.val_to_arg_str(desc.tensor.get_dtype().itemsize)
        fn = "initTMADescriptor"
        args = ", ".join(
            str(x)
            for x in [
                f"&{desc_name}.m",
                ptr,
                element_size,
                rank,
                f"{desc_name}.block_shape",
                f"{desc_name}.global_shape",
                f"{desc_name}.strides",
            ]
        )
        self.writeline(f"{fn}({args});")

    def generate_args_decl(
        self,
        code: IndentedBuffer | Self,
        call_args,
        arg_types,
        arg_signatures,
        is_triton_kernel=True,
        scratch_spaces: dict[str, int] | None = None,
    ):
        """
        Generates any declarations of args to pass into a kernel call, and then returns the arg names.

        In more detail:
        * declarations: e.g. this function has a side effect of generating lines like `auto var_0 = ...;`
        * returns: a string with the list of args, e.g. "var_0, var_1"

        call_args: list of call arguments
        arg_types: list of argument types
        arg_signatures: list with signatures of all the args
        is_triton_kernel: whether these are passed into a triton kernel or not. In particular,
                          calls to triton kernels will have an additional global scratch space
                          arg injected at the front of the arg list.
        """
        new_args: list[str] = []

        def process_tma_stable_arg(arg, arg_type, arg_signature, var_name):
            # [Note: AOTI TMA Stable handling]
            # For most args, a single arg passed to the python triton interface
            # maps to a single arg in the cubin interface. However, for host-side
            # TMA descriptors, a single python arg turns into 1 + 2 * N args in the
            # cubin interface (where N is the rank).
            #
            # To do this: at TMA codegen time (for aoti), we generate a struct
            # (StableTMADescriptor) containing the necessary information; and then
            # when we call the function (i.e. here), we unpack the struct members.
            code.writeline(f"auto {var_name} = {cexpr(arg)};")
            return _unpack_tma_descriptor_args(var_name, arg_signature)

        def process_args(arg, arg_type, arg_signature=None):
            var_name = f"var_{next(self.arg_var_id)}"
            # ignore tma descriptors, as host-side TMA descriptors need
            # to be passed to the compiled Triton kernel by value
            if isinstance(arg_type, UnwrapUnspecArg) and not signature_is_tma_desc(
                arg_signature
            ):
                self.codegen_tensor_item(
                    arg_type.dtype,
                    arg,
                    var_name,
                    indented_buffer=code,
                )
                new_args.append(f"&{var_name}")
            elif isinstance(arg_type, torch_dtype) and not signature_is_tma_desc(
                arg_signature
            ):
                device_ptr_type = self.device_codegen.cpp_device_ptr()
                code.writeline(
                    maybe_hipify_code_wrapper(
                        f"{device_ptr_type} {var_name} = reinterpret_cast<{device_ptr_type}>({arg}.data_ptr());"
                    )
                )
                new_args.append(f"&{var_name}")
            # For symbolic call arguments, examine the arg signatures from triton meta
            # to explicitly cast to the right type
            # Reason: `auto` can infer unexpected type against kernel input signature.
            elif (
                isinstance(arg_type, type(SymbolicCallArg))
                and arg_signature is not None
                and arg_signature in TRITON_SIGNATURE_TO_CPP
            ):
                code.writeline(
                    f"{TRITON_SIGNATURE_TO_CPP[arg_signature]} {var_name} = {cexpr(arg)};"
                )
                new_args.append(f"&{var_name}")
            elif arg_type in (sympy.Integer, int):
                code.writeline(f"int {var_name} = {cexpr(arg)};")
                new_args.append(f"&{var_name}")
            elif arg_type in (sympy.Float, float):
                # Use signature type if available, otherwise default to float
                cpp_type = TRITON_SIGNATURE_TO_CPP.get(  # pyrefly: ignore[no-matching-overload]
                    arg_signature, "float"
                )
                code.writeline(f"{cpp_type} {var_name} = {cexpr(arg)};")
                new_args.append(f"&{var_name}")
            elif arg_signature and arg_signature.startswith("tensordesc<"):
                new_args.extend(
                    process_tma_stable_arg(arg, arg_type, arg_signature, var_name)
                )
            else:
                code.writeline(f"auto {var_name} = {cexpr(arg)};")
                new_args.append(f"&{var_name}")

        for arg, arg_type, arg_signature in zip_longest(
            call_args, arg_types, arg_signatures
        ):
            process_args(arg, arg_type, arg_signature)

        for scratch_name, workspace_size in (scratch_spaces or {}).items():
            if (
                is_triton_kernel
                and (
                    scratch := self.device_codegen.cpp_scratch(
                        next(self.arg_var_id),
                        workspace=TritonScratchWorkspace(
                            size=workspace_size,
                            generate_dtype_str=(
                                lambda: self.codegen_dtype(torch.uint8)
                            ),
                        ),
                        prefix=scratch_name,
                    )
                )
                is not None
            ):
                scratch_def, scratch_var = scratch
                code.writelines([maybe_hipify_code_wrapper(x) for x in scratch_def])
                new_args.append(f"&{scratch_var}")

        return ", ".join(new_args)

    def _generate_kernel_call_helper(
        self,
        kernel_name: str,
        call_args,
        *,
        device=None,
        triton=True,
        arg_types=None,
        raw_keys=None,
        raw_args=None,
        triton_meta: TritonMeta | None = None,
        inductor_meta=None,
        graph_name="",
        original_fxnode_name=None,
        current_stream_idx=None,
    ):
        """
        Override the default value of argument 'gpu' to True here.
        generate_kernel_call can still be called with gpu=False because of
        a mix of cpu kernels and gpu kernels.
        """
        device = device or V.graph.get_current_device_or_throw()
        if device.type == "cpu":
            # Even in CppWrapperGpu, we may see cpp kernels
            return CppWrapperCpu._generate_kernel_call_helper(
                self,
                kernel_name,
                call_args,
                device=device,
                triton=triton,
                arg_types=arg_types,
                raw_keys=raw_keys,
                raw_args=raw_args,
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
            )

        if (
            triton
            and config.triton.autotune_at_compile_time
            and kernel_name not in self.kernel_autotune_names
        ):
            # Call PythonWrapperCodegen to create the autotune code block
            PythonWrapperCodegen._generate_kernel_call_helper(
                self,
                kernel_name,
                call_args,
                device=device,
                triton=triton,
                arg_types=arg_types,
                raw_keys=raw_keys,
                raw_args=raw_args,
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
                original_fxnode_name=original_fxnode_name,
            )

        if (
            V.graph.aot_mode
            and current_stream_idx is not None
            and current_stream_idx != 0
        ):
            stream = get_stream_name(current_stream_idx)
        else:
            stream = self.write_get_raw_stream(device.index, graph_name)

        if triton:
            call_args, arg_types = self.prepare_triton_wrapper_args(
                call_args,
                # pyrefly: ignore [bad-argument-type]
                arg_types,
            )

            # For lazy compile mode with TMA, extract underlying tensor names
            tma_tensor_args: dict[str, str] = {}
            is_lazy_compile = config.triton.autotune_at_compile_time is False
            if is_lazy_compile and raw_args and triton_meta:
                signature = triton_meta.get("signature", {})
                raw_keys_list = raw_keys or []
                for key, raw_arg in zip(raw_keys_list, raw_args):
                    sig_type = signature.get(key, "")
                    if isinstance(sig_type, str) and signature_is_tma_desc(sig_type):
                        if isinstance(raw_arg, TMADescriptorStable):
                            # Get the underlying tensor name
                            tensor_name = raw_arg.get_tensor().get_name()
                            tma_tensor_args[key] = tensor_name
                        else:
                            raise AssertionError("Unsupported TMA descriptor type")

            wrapper_name = f"call_{kernel_name}"
            if wrapper_name not in self._triton_call_wrappers:
                self._triton_call_wrappers[wrapper_name] = DeferredTritonCallWrapper(
                    wrapper_name,
                    kernel_name,
                    self._kernel_name_to_body,
                    arg_types,
                    triton_meta=triton_meta,
                    inductor_meta=inductor_meta,
                    tma_tensor_args=tma_tensor_args,
                )

            # For TMA in lazy compile mode, add tensor args to the call
            if is_lazy_compile and tma_tensor_args:
                for tensor_name in tma_tensor_args.values():
                    call_args.append(tensor_name)
                    arg_types.append(
                        torch.float32
                    )  # dtype doesn't matter, just need tensor type

            # AOTI side uses this->device_idx_ so the model honors runtime device
            # assignment; JIT side has no this->device_idx_ in scope, so use the
            # concrete graph device index. Similarly, AOTI side always uses the
            # `stream` function parameter of run_impl, while JIT side uses the
            # locally-declared stream{idx} (see _codegen_entry_impl_prologue).
            aoti_device_idx = (
                "this->device_idx_" if V.graph.aot_mode else str(device.index)
            )
            jit_device_idx = str(device.index)
            jit_call_args = [*call_args, jit_device_idx, stream]
            aot_call_args = [*call_args, aoti_device_idx, "stream"]
            debug_printer_manager = V.graph.wrapper_code.debug_printer
            debug_printer_manager.set_printer_args(
                jit_call_args[: len(arg_types)], kernel_name, arg_types, None
            )
            # DebugPrinterManager is AOTI-only: route its writes through
            # writeline_aot so they're dropped on the JIT side (no-op for the
            # pure-JIT IndentedBuffer; AOT-only for DualIndentedBuffer). Without
            # this, the JIT buffer would receive before/after-launch prints
            # after the JIT launch, reporting post-launch state as pre-launch.
            with self.set_writeline(self.wrapper_call, self.wrapper_call.writeline_aot):
                with debug_printer_manager:
                    self.wrapper_call.writeline_jit(
                        f"{wrapper_name}({', '.join(jit_call_args)});"
                    )
                    self.wrapper_call.writeline_aot(
                        f"{wrapper_name}({', '.join(aot_call_args)}, "
                        f"kernels, this->cubin_dir_);"
                    )
        else:
            casted = []
            # pyrefly: ignore [bad-argument-type, no-matching-overload]
            for arg_type, arg in zip(arg_types, call_args):
                new_arg = arg
                if arg_type.endswith("*") and arg != "nullptr":
                    new_arg = f"{arg}.data_ptr()"
                # pyrefly: ignore [bad-argument-type]
                casted.append(f"({arg_type}){cexpr(new_arg)}")
            call_args_str = ", ".join(casted)
            # AOT: dispatch through AOTInductorModelKernels member.
            # JIT: call the extern "C" symbol directly (resolved at link time
            # via extra_flags pointing at the compiled .so).
            kernel_prefix = "kernels." if V.graph.aot_mode else ""
            self.writeline(f"{kernel_prefix}{kernel_name}({call_args_str}, {stream});")

    def prepare_triton_wrapper_args(
        self, call_args: list[Any], arg_types: list[Any]
    ) -> tuple[list[Any], list[Any]]:
        if len(call_args) != len(arg_types):
            raise AssertionError((call_args, arg_types))
        new_args = []
        new_args_types = []
        for arg, arg_type in zip(call_args, arg_types):
            if isinstance(arg, str):
                if isinstance(arg_type, torch_dtype) and should_unwrap_unspec_arg(arg):
                    # dynamo wraps unspec variable as 0d CPU tensor, need convert to scalar
                    arg_type = UnwrapUnspecArg(dtype=arg_type)
                new_args.append(arg)
            elif isinstance(arg, bool):
                new_args.append(str(arg).lower())
            elif isinstance(arg, (int, float, SymbolicCallArg)):
                if isinstance(arg, float):
                    new_args.append(self.generate_float_value(arg))
                else:
                    new_args.append(str(arg))
            else:
                new_args.append(cexpr(V.graph.sizevars.simplify(arg)))
            new_args_types.append(arg_type)
        return new_args, new_args_types

    def make_zero_buffer(self, name):
        return f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_zero_({name}.get()));"


@dataclasses.dataclass
class UnwrapUnspecArg:
    """Marker that we need to call .item() on the tensor"""

    dtype: torch_dtype
<|endoftext|>from typing_extensions import override

from torch._inductor import ir

from .wrapper import PythonWrapperCodegen


class PythonWrapperMtia(PythonWrapperCodegen):
    """
    A thin wrapper of PythonWrapperCodegen with MTIA specific logic
    """

    @override
    def write_header(self) -> None:
        super().write_header()

        # MITA specific imports
        self.imports.splice("import mtia.host_runtime.torch_mtia.dynamic_library")

    @override
    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: str | None,
        parent_wrapper: PythonWrapperCodegen | None,
        partition_signatures: ir.GraphPartitionSignature | None = None,
    ) -> PythonWrapperCodegen:
        if is_subgraph:
            # Delegate to the parent class to handle the case of subgraph
            return PythonWrapperCodegen.create(
                is_subgraph, subgraph_name, parent_wrapper, partition_signatures
            )
        return PythonWrapperMtia()
<|endoftext|># mypy: allow-untyped-defs
import contextlib
import itertools
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import sympy

from .. import ir
from ..select_algorithm import PartialRender
from ..virtualized import V
from .common import ArgName
from .cpp_gemm_template import CppGemmTemplate, GEMM_TEMPLATE
from .cpp_micro_gemm import LayoutType
from .cpp_template_kernel import CppTemplateKernel
from .cpp_utils import DTYPE_TO_CPP, GemmBlocking


# We pass all sizevars present in BY to the GEMM templates so variables are not renamed in the BMM definition
GEMM_SINGLE_THREAD_MM_STUB = r"""
{{kernel.def_kernel(
    inputs={"X": X, "W": W},
    outputs={"Y": Y_2d},
    aliases=aliases,
    function_name=kernel_name+"_single_thread_mm",
    extra_sizevars=BY_sizevars + [b_index],
    placeholder="<SINGLE_THREAD_MM_DEF_FOR_BMM>")}}"""

GEMM_THREADED_MM_STUB = r"""
{{kernel.def_kernel(
    inputs={"X": X, "W": W},
    outputs={"Y": Y_2d},
    aliases=aliases,
    function_name=kernel_name+"_threaded_mm",
    extra_sizevars=BY_sizevars + [b_index],
    placeholder="<THREADED_MM_DEF_FOR_BMM>")}}"""

BMM_TEMPLATE = r"""
{{ template.codegen_microkernel_def() }}
{{ template.codegen_single_thread_gemm() }}
{{ template.codegen_multi_thread_gemm() }}

extern "C"
{{kernel.def_kernel(inputs={"X": BX, "W": BW}, outputs={"Y": BY}, aliases=aliases)}}
{
    const int64_t B = {{kernel.size(BY_2d, 0)}};
    {%- if num_threads > 1 %}
    constexpr int64_t num_threads = {{num_threads}};
    int64_t B_single_thread_block = (B / num_threads) * num_threads;

    {%- set use_dynamic_threads = ((config.cpp.threads < 1) and (num_threads == cpu_count)) or config.cpp.dynamic_threads %}
    {%- if use_dynamic_threads %}
    #pragma omp parallel for
    {%- else %}
    #pragma omp parallel for num_threads({{num_threads}})
    {%- endif %}
    {%- else %}
    int64_t B_single_thread_block = B;
    {%- endif %}
    for (int64_t b_start = 0; b_start < B_single_thread_block; ++b_start) {
        {{template.get_gemm_function_call(
            kernel,
            kernel_name+"_single_thread_mm",
            "<SINGLE_THREAD_CALL_FOR_BMM>",
            b_index="b_start",
        )}}
    }
    for (int64_t b_start = B_single_thread_block; b_start < B; ++b_start) {
        {{template.get_gemm_function_call(
            kernel,
            kernel_name+"_threaded_mm",
            "<THREADED_MM_CALL_FOR_BMM>",
            b_index="b_start",
        )}}
    }
}
"""


class CppBmmTemplate(CppGemmTemplate):
    def __init__(
        self,
        input_nodes,
        layout: ir.Layout,
        num_threads: int,
        register_blocking: GemmBlocking,
        beta=1,
        alpha=1,
        has_bias=False,
        epilogue_creator: Callable[[ir.Buffer], ir.Pointwise] | None = None,
        should_block_weights: bool = False,
        name="bmm",
    ):
        """
        In order to simplify the implementation and increase code reuse, the BMM template implements
        two versions of the GEMM kernel: a single-threaded version and a multi-threaded version.
        GEMM kernels are called in a loop over the batch dimension, with single-threaded GEMM calls
        for all but the last (B % num_threads), which are handled by the multi-threaded GEMM kernel.

        We use an extra sizevar `b_index` to index the batch dimension, which we pass into the GEMM
        template as a sympy.Symbol. This allows us to slice the 3D batch tensors in the GEMM template
        without any changes to the GEMM template itself.
        """
        super().__init__(
            input_nodes,
            layout,
            num_threads,
            register_blocking,
            beta=beta,
            alpha=alpha,
            has_bias=has_bias,
            epilogue_creator=epilogue_creator,
            should_block_weights=should_block_weights,
            name=name,
        )
        self.b_index = sympy.Symbol("s_b_index", integer=True, nonnegative=True)

    @staticmethod
    def get_padded_size(n, block_n, k, should_block_weight):
        if should_block_weight:
            # Tensor is constant or not contiguous, so we will pad and block
            new_size, padded_n = CppGemmTemplate.get_padded_size(
                n, block_n, k, should_block_weight
            )
            # Add the new batch dimension
            new_size.insert(0, -1)
            return new_size, padded_n
        else:
            new_size = [-1, k, n]
            return new_size, n

    @staticmethod
    def check_if_block_weight(W, micro_gemm):
        if not isinstance(W, ir.IRNode):
            raise AssertionError(f"expected W to be an ir.IRNode, got {type(W)}")
        _, n = W.get_size()[-2:]
        result = (
            not W.get_layout().is_contiguous()
            or W.get_name() in V.graph.constants
            or (
                n % micro_gemm.register_blocking.block_n != 0
                and micro_gemm.get_b_layout != LayoutType.NORMAL
            )
        )
        return result

    def get_gemm_function_call(
        self,
        kernel: CppTemplateKernel,
        function_name: str,
        placeholder: str,
        b_index: str,
    ) -> str:
        """
        Similar to 'def_kernel' in cpp_template_kernel, but instead of generating a function definition,
        generate a function call for the GEMM kernel.
        Args:
            placeholder: The string to replace the function call with
            b_index: The index for slicing the 3D batch tensors
        """

        def hook():
            arg_defs, call_args, _, _ = kernel.args.python_argdefs()
            for i, buf in enumerate(call_args):
                if buf == self.b_index:
                    arg_defs[i] = ArgName(b_index)
            call = f"{function_name}({', '.join(x.full_name() for x in arg_defs)});"
            return call

        if placeholder in kernel.render_hooks:
            raise AssertionError(f"render hook already registered for {placeholder}")
        kernel.render_hooks[placeholder] = hook
        return placeholder

    def get_default_reindexers(self, epilogues):
        def reindexer(args):
            # if epilogue nodes exist, they have 3D ranges but args are 2D, so add 0 index
            return [self.b_index] + args

        return [reindexer] * len(epilogues)

    def get_options(
        self,
        kernel: CppTemplateKernel,
        template_buffer_node: ir.CppTemplateBuffer | None = None,
        flag_template_buffer_has_other_users: bool | None = None,
        epilogue_nodes: list[ir.IRNode] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        options = super().get_options(
            kernel=kernel,
            template_buffer_node=template_buffer_node,
            flag_template_buffer_has_other_users=flag_template_buffer_has_other_users,
            epilogue_nodes=epilogue_nodes,
            **kwargs,
        )

        BX, BW, BY = options["X"], options["W"], options["Y"]
        options["BX"], options["BW"], options["BY"] = BX, BW, BY
        options["BY_2d"] = options["Y_2d"]
        for kword in ["X", "W", "GemmOut", "Y_2d"]:
            options[kword] = kernel.select(options[kword], 0, self.b_index)
        for kword in ["X", "W", "Y_2d"]:
            options[kword + "_dtype"] = DTYPE_TO_CPP[options[kword].dtype]
        options["b_index"] = self.b_index
        options["BY_sizevars"] = [
            s
            for sym in itertools.chain(BY.get_size(), BY.get_stride())
            if isinstance(sym, sympy.Expr)
            for s in sym.free_symbols
        ]
        options["kernel_name"] = kernel.kernel_name

        return options

    def render(  # type: ignore[override, return]
        self,
        kernel: CppTemplateKernel,
        template_buffer_node: ir.CppTemplateBuffer | None = None,
        flag_template_buffer_has_other_users: bool | None = None,
        epilogue_nodes: list[ir.IRNode] | None = None,
        **kwargs,
    ) -> str:
        options = self.get_options(
            kernel=kernel,
            template_buffer_node=template_buffer_node,
            flag_template_buffer_has_other_users=flag_template_buffer_has_other_users,
            epilogue_nodes=epilogue_nodes,
            **kwargs,
        )
        self.render_options = options

        with contextlib.ExitStack() as stack:
            for buf in options["fake_buffers"]:
                stack.enter_context(
                    patch.object(V.graph, "get_dtype", self._fake_get_dtype(buf))
                )
            result = self._template_from_string(BMM_TEMPLATE).render(**options)

            # Finalize the function definitions for the gemm routines
            sub_mm_hooks = {
                name: hook
                for name, hook in kernel.render_hooks.items()
                if "FOR_BMM" in name
            }
            result = PartialRender(result, sub_mm_hooks).finalize_all()
            for name in sub_mm_hooks:
                del kernel.render_hooks[name]
            del kernel.args.sizevars[options["b_index"]]
            return result

    def codegen_single_thread_gemm(self):
        stub = self._template_from_string(GEMM_SINGLE_THREAD_MM_STUB).render(
            self.render_options
        )
        return stub + self._template_from_string(GEMM_TEMPLATE).render(
            {**self.render_options, "num_threads": 1}
        )

    def codegen_multi_thread_gemm(self):
        stub = self._template_from_string(GEMM_THREADED_MM_STUB).render(
            self.render_options
        )
        return stub + self._template_from_string(GEMM_TEMPLATE).render(
            self.render_options
        )

    def codegen_gemm_stub_def(self):
        return ""
<|endoftext|># mypy: allow-untyped-defs
import itertools
from collections.abc import Callable, Iterable
from typing import Any
from unittest.mock import patch

import sympy
from sympy.parsing.sympy_parser import parse_expr

import torch
from torch._inductor.utils import do_bench_using_profiling
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.symbol import SymT

from .. import config, cpp_builder, ir, lowering as L
from ..autotune_process import CppBenchmarkRequest
from ..loop_body import LoopBody
from ..select_algorithm import PartialRender
from ..utils import sympy_index_symbol, sympy_index_symbol_with_prefix
from ..virtualized import V
from .common import REMOVED
from .cpp import CppKernel, CppKernelProxy, KernelGroup, ParallelDepth
from .cpp_utils import cexpr_index, DTYPE_TO_CPP, LocalBufferContext


def parse_expr_with_index_symbols(expr):
    if isinstance(expr, sympy.Expr):
        return expr
    elif isinstance(expr, (list, tuple)):
        return [parse_expr_with_index_symbols(e) for e in expr]
    else:
        expr = parse_expr(str(expr))
        int_symbols = {sym: sympy_index_symbol(sym.name) for sym in expr.free_symbols}
        return expr.subs(int_symbols)


def wrap_with_tensorbox(node) -> ir.TensorBox:
    return (
        ir.TensorBox.create(node) if isinstance(node, ir.Buffer) else ir.TensorBox(node)
    )


class CppTemplateKernel(CppKernel):
    def __init__(self, kernel_name, num_threads):
        super().__init__(None, num_threads)
        self.kernel_name = kernel_name
        self.render_hooks = {}
        self.local_buffers = {}

    def render(self, template, **kwargs):
        return PartialRender(
            template.render(kernel=self, **kwargs), self.render_hooks
        ).finalize_all()

    def def_kernel(
        self,
        inputs: dict[str, ir.Buffer],
        outputs: dict[str, ir.Buffer],
        aliases: dict[str, str] | None = None,
        function_name: str = "",
        extra_sizevars: list[sympy.Expr] | None = None,
        placeholder: str = "<DEF_KERNEL>",
    ) -> str:
        if len(function_name) == 0:
            function_name = str(self.kernel_name)
        for name, inp in inputs.items():
            if inp is not None:
                self.args.input_buffers[inp.get_name()] = name
        for name, out in outputs.items():
            self.args.output_buffers[out.get_name()] = name
        if aliases is not None:
            for alias, orig in aliases.items():
                if orig in self.args.input_buffers:
                    self.args.input_buffers[alias] = self.args.input_buffers[orig]
                if orig in self.args.output_buffers:
                    self.args.output_buffers[alias] = self.args.output_buffers[orig]

        unique_sizevars = OrderedSet(
            s
            for input in inputs.values()
            if input is not None
            for sym in itertools.chain(input.get_size(), input.get_stride())
            if isinstance(sym, sympy.Expr)
            for s in sym.free_symbols
        )
        unique_sizevars.update(
            s
            for sym in extra_sizevars or []
            if isinstance(sym, sympy.Expr)
            for s in sym.free_symbols
        )
        unique_sizevars.update(
            s
            for output in outputs.values()
            for sym in itertools.chain(output.get_size(), output.get_stride())
            if isinstance(sym, sympy.Expr)
            for s in sym.free_symbols
        )
        sizevars = sorted(unique_sizevars, key=str)
        for sizevar in sizevars:
            self.args.sizevars[sizevar] = f"k{sizevar}"

        def hook():
            # remove all aliases before generate function definition
            if aliases is not None:
                for alias in aliases:
                    if alias in self.args.input_buffers:
                        raise AssertionError(
                            f"input_buffers cannot be removed: {alias}"
                        )
                    if alias in self.args.output_buffers:
                        self.args.output_buffers[alias] = REMOVED
            cpp_argdefs, _, _ = self.args.cpp_argdefs()
            return f"void {function_name}({', '.join(cpp_argdefs)})"

        if placeholder in self.render_hooks:
            raise AssertionError(f"placeholder already registered: {placeholder}")
        self.render_hooks[placeholder] = hook
        return placeholder

    def call_kernel(self, name: str, node: ir.CppTemplateBuffer):
        wrapper = V.graph.wrapper_code
        _, call_args, arg_types = self.args.cpp_argdefs()
        wrapper.generate_kernel_call(name, call_args, triton=False, arg_types=arg_types)

    def dtype(self, node: ir.Buffer) -> str:
        return DTYPE_TO_CPP[node.get_dtype()]

    def acc_dtype(self, node: ir.Buffer) -> str:
        if node.get_dtype() in [torch.float32, torch.bfloat16, torch.half]:
            return "float"
        else:
            raise NotImplementedError(f"Unsupported dtype: {node.get_dtype()}")

    def size(self, node: ir.Buffer, dim: int) -> str:
        return cexpr_index(self.rename_indexing(node.get_size()[dim]))

    def stride(self, node: ir.Buffer, dim: int) -> str:
        return cexpr_index(self.rename_indexing(node.get_stride()[dim]))

    def index(self, node: ir.Buffer, indices: list[Any]) -> str:
        indexer = node.get_layout().as_fixed().make_indexer()
        index = indexer(parse_expr_with_index_symbols(indices))
        index = self.rename_indexing(index)
        outer_name = node.get_name()
        inner_name = (
            outer_name
            if outer_name in self.local_buffers
            else self.args.input(node.get_name())
        )
        return f"{inner_name}[{cexpr_index(index)}]"

    def slice_nd(self, node, ranges: list[tuple[Any, Any]]) -> ir.ReinterpretView:
        """
        Slice the given node with a list of ranges (start and end) corresponding to its dims.
        The dim is not sliced if the corresponding range is empty.
        """
        if len(ranges) != len(node.get_size()):
            raise AssertionError(f"{ranges=}, {node=}")
        sliced = wrap_with_tensorbox(node)
        for dim, _range in enumerate(ranges):
            if len(_range) == 0:
                continue
            if len(_range) != 2:
                raise AssertionError(f"expected range of length 2, got {len(_range)}")
            start, end = parse_expr_with_index_symbols(_range)
            sliced = L.slice_(sliced, dim, start, end, clamp=False)
        if not isinstance(sliced, ir.TensorBox):
            raise AssertionError(f"expected ir.TensorBox, got {type(sliced)}")
        if not isinstance(sliced.data, ir.ReinterpretView):
            raise AssertionError(sliced.data)
        return sliced.data

    def select(self, node, dim: int, idx: int) -> ir.ReinterpretView:
        # We avoid using L.select here because we need clamp=False so the dim after slicing
        # is 1 instead of a sympy expression of symbol - dim_size.
        node = wrap_with_tensorbox(node)
        idx = ir.View.handle_negative_index(idx, node.get_size()[dim])
        sliced = L.squeeze(L.slice_(node, dim, idx, idx + 1, clamp=False), dim)
        if not isinstance(sliced.data, ir.ReinterpretView):
            raise AssertionError(sliced.data)
        return sliced.data

    def view(self, node, sizes: list[Any]) -> ir.IRNode:
        node = wrap_with_tensorbox(node)
        sizes = parse_expr_with_index_symbols(sizes)
        return L.view(node, sizes).data

    def permute(self, node, dims):
        node = wrap_with_tensorbox(node)
        permuted = L.permute(node, dims).data
        if not isinstance(permuted, ir.ReinterpretView):
            raise AssertionError(f"expected ir.ReinterpretView, got {type(permuted)}")
        return permuted

    def maybe_codegen_profile(self, prefix_kernel_name: str | None = None) -> str:
        if config.cpp.enable_kernel_profile:
            graph_id = V.graph.graph_id
            prefix = "graph_" + str(graph_id) + "_" if graph_id is not None else ""
            if prefix and prefix_kernel_name:
                prefix += prefix_kernel_name + "_"
            handle_str = (
                "torch::aot_inductor::RAIIAtenRecordFunctionHandle "
                f'record_{prefix}{self.kernel_name}_("{prefix}{self.kernel_name}", nullptr);'
            )
            return handle_str
        else:
            return ""

    def unroll_pragma(self, unroll):
        if cpp_builder.is_gcc():
            return f"#pragma GCC unroll {unroll}"
        else:
            return f"#pragma unroll {unroll}"

    def define_buffer(self, name, sizes: list[Any], dtype=torch.float) -> str:
        """Define kernel local buffer"""
        sizes = parse_expr_with_index_symbols(sizes)
        buf = ir.Buffer(
            name=name, layout=ir.FixedLayout(torch.device("cpu"), dtype, sizes)
        )
        self.local_buffers[name] = buf
        ctype = f"{DTYPE_TO_CPP[dtype]}"
        numel = f"{cexpr_index(buf.get_numel())}"
        return f"auto _{name} = std::make_unique<{ctype}[]>({numel}); auto {name} = _{name}.get();"

    def define_stack_allocated_buffer(
        self, name, sizes: list[Any], dtype=torch.float
    ) -> str:
        """Define stack-allocated buffer"""
        sizes = parse_expr_with_index_symbols(sizes)
        buf = ir.Buffer(
            name=name, layout=ir.FixedLayout(torch.device("cpu"), dtype, sizes)
        )
        self.local_buffers[name] = buf
        ctype = f"{DTYPE_TO_CPP[dtype]}"
        numel = f"{cexpr_index(buf.get_numel())}"
        return f"alignas(64) {ctype} _{name}[{numel}]; {ctype}* {name} = _{name};"

    def reinit_buffer_if_null(self, name):
        """Reinit the previously defined local buffer if it is null"""
        if name not in self.local_buffers:
            raise AssertionError(f"unknown local buffer: {name}")
        buf = self.local_buffers[name]
        ctype = f"{DTYPE_TO_CPP[buf.layout.dtype]}"
        numel = f"{cexpr_index(buf.get_numel())}"
        return f"if (_{name} == nullptr) {{ _{name} = std::make_unique<{ctype}[]>({numel}); {name} = _{name}.get(); }}"

    def release_buffer(self, name):
        """Codegen the code to release the ownership of a local buffer to others"""
        if name not in self.local_buffers:
            raise AssertionError(f"unknown local buffer: {name}")
        return f"_{name}.release()"

    def store_pointwise_nodes(
        self,
        dst: ir.Buffer,
        nodes: list[ir.IRNode],
        offsets: list[sympy.Expr] | None = None,
        reindexers: list[Callable[[list[Any]], list[Any]] | None] | None = None,
    ) -> str:
        var_sizes = (tuple(dst.get_size()), ())
        var_ranges = {
            sympy_index_symbol_with_prefix(SymT.INDEX, i): sz
            for i, sz in enumerate(var_sizes[0])
        }
        if not offsets:
            offsets = [sympy.S.Zero] * len(var_sizes[0])
        if not reindexers:
            reindexers = [None] * len(nodes)
        if len(offsets) != len(var_sizes[0]):
            raise AssertionError(
                f"expected {len(var_sizes[0])} offsets, got {len(offsets)}"
            )
        output_index = dst.get_layout().make_indexer()([*var_ranges.keys()])
        kernel_group = KernelGroup()
        kernel_group.args = self.args
        cpp_kernel_proxy = CppKernelProxy(kernel_group)
        bodies = []
        var_sizes_list = []
        for i, node in enumerate(nodes):
            output_name = node.get_name() if i < len(nodes) - 1 else dst.get_name()
            node = node.data if isinstance(node, ir.ComputedBuffer) else node
            if not isinstance(node, ir.Pointwise):
                raise AssertionError(node)

            def fn(*args):
                if len(args) != 2:
                    raise AssertionError(f"expected 2 args, got {len(args)}")
                if len(args[0]) != len(var_sizes[0]):
                    raise AssertionError(
                        f"expected {len(var_sizes[0])} indices, got {len(args[0])}"
                    )
                if len(args[1]) != 0:
                    raise AssertionError(
                        f"expected no reduction vars, got {len(args[1])}"
                    )
                new_args = [arg + offset for arg, offset in zip(args[0], offsets)]  # type: ignore[arg-type]
                if reindexers[i] is not None:
                    new_args = reindexers[i](new_args)  # type: ignore[misc]
                V.ops.store(
                    output_name,
                    output_index,
                    node.make_loader()(new_args).value,
                )

            body = LoopBody(
                fn,
                (list(var_ranges.keys()), ()),
                var_ranges,
                list(var_ranges.keys()),
                tuple(),
            )
            bodies.append(body)
            var_sizes_list.append(var_sizes)

        cpp_kernel_proxy.codegen_loop_bodies(bodies, var_sizes_list)

        def max_parallel_depth():
            return ParallelDepth(parallel_depth=0, start_depth=0)

        # This loop is not parallelized since it is not the outermost loop.
        with patch.object(
            cpp_kernel_proxy.loop_nest, "max_parallel_depth", max_parallel_depth
        ):
            kernel_group.finalize_kernel(cpp_kernel_proxy, [])
        return kernel_group.loops_code.getvalue()

    def store_grouped_gemm_pointwise_nodes(
        self,
        dst: tuple[ir.Buffer],
        nodes: list[ir.IRNode],
        offsets: list[sympy.Expr],
        reindexers: list[Callable[[list[Any]], list[Any]] | None],
        output_names: list[str],
    ) -> str:
        ref_dst = dst[0]
        var_sizes = (tuple(ref_dst.get_size()), ())
        var_ranges = {
            sympy_index_symbol_with_prefix(SymT.INDEX, i): sz
            for i, sz in enumerate(var_sizes[0])
        }
        if not offsets:
            raise AssertionError("offsets should be set outside")
        if not all(len(offset) == len(var_sizes[0]) for offset in offsets):
            raise AssertionError(f"expected all offsets of length {len(var_sizes[0])}")
        output_index = ref_dst.get_layout().make_indexer()([*var_ranges.keys()])
        kernel_group = KernelGroup()
        kernel_group.args = self.args
        cpp_kernel_proxy = CppKernelProxy(kernel_group)
        bodies = []
        var_sizes_list = []
        for i, node in enumerate(nodes):
            output_name = output_names[i]
            node = node.data if isinstance(node, ir.ComputedBuffer) else node
            if not isinstance(node, ir.Pointwise):
                raise AssertionError(node)

            def fn(*args):
                if len(args) != 2:
                    raise AssertionError(f"expected 2 args, got {len(args)}")
                if len(args[0]) != len(var_sizes[0]):
                    raise AssertionError(
                        f"expected {len(var_sizes[0])} indices, got {len(args[0])}"
                    )
                if len(args[1]) != 0:
                    raise AssertionError(
                        f"expected no reduction vars, got {len(args[1])}"
                    )
                new_args = [arg + offset for arg, offset in zip(args[0], offsets[i])]  # type: ignore[arg-type]
                if reindexers[i] is not None:
                    new_args = reindexers[i](new_args)  # type: ignore[misc]
                V.ops.store(
                    output_name,
                    output_index,
                    node.make_loader()(new_args).value,
                )

            body = LoopBody(
                fn,
                (list(var_ranges.keys()), ()),
                var_ranges,
                list(var_ranges.keys()),
                tuple(),
            )
            bodies.append(body)
            var_sizes_list.append(var_sizes)

        cpp_kernel_proxy.codegen_loop_bodies(bodies, var_sizes_list)

        def max_parallel_depth():
            return ParallelDepth(parallel_depth=0, start_depth=0)

        # This loop is not parallelized since it is not the outermost loop.
        with patch.object(
            cpp_kernel_proxy.loop_nest, "max_parallel_depth", max_parallel_depth
        ):
            kernel_group.finalize_kernel(cpp_kernel_proxy, [])
        return kernel_group.loops_code.getvalue()

    def store_output(
        self,
        dst: ir.Buffer,
        src: ir.Buffer,
        orig_src: ir.Buffer | None = None,
        epilogue_nodes: list[ir.IRNode] | None = None,
        offsets: list[Any] | None = None,
        reindexers: list[Callable[[list[Any]], list[Any]] | None] | None = None,
    ):
        """
        Store the `src` buffer to the `dst` buffer. The size of `src` and `dst` should match.
        If `epilogue_nodes` is provided, the `src` buffer is firstly computed with the epilogues
        before stored to `dst`. The `epilogues_nodes` are all pointwise.

        Notes:
        1. `src` and `dst` buffer could be the same buffer in which case we are doing in-place compute
           and stores. In case `epilogue_nodes` are not provided, we do nothing.
        2. The `epilogue_nodes`, if exist, have computations on `src` before storing to `dst` but since
           they come from the original Inductor IR, they might need to be adjusted before working with
           `src` and `dst` as outlined below:
           a) `src` or `dst` buffer could be a sub-slice of the ranges the `epilogue_nodes`work on.
              In this case, the `offsets` could be provided to adjust the indices passed to
              `epilogue_nodes` during codegen and the data ranges are also configured according to
              the sizes of `src` and `dst`.
           b) `dst` might be indexed in a different way as the `epilogue_nodes`, hence a `reindexer` is
              needed on the indices to `epilogue_nodes` to match the indexing of `dst`.
           c) If `src` is local, we need to add a local buffer for it and localize the `orig_src` buffer
              in `epilogue_nodes` with `src`.
        """
        if not isinstance(dst, (ir.Buffer, ir.ReinterpretView)):
            raise AssertionError(
                f"expected ir.Buffer or ir.ReinterpretView, got {type(dst)}"
            )
        if dst.get_size() != src.get_size():
            raise AssertionError(f"{dst=}, {src=}")
        if offsets:
            offsets = parse_expr_with_index_symbols(offsets)
        if epilogue_nodes:
            with LocalBufferContext(self.args) as scope:
                if orig_src is None:
                    raise AssertionError("orig_src must not be None")
                if orig_src.get_name() != src.get_name():
                    scope.add_local_buffer(
                        src,
                        [
                            orig_src,
                        ],
                    )
                    epilogue_nodes = scope.localize_nodes(epilogue_nodes)
                return self.store_pointwise_nodes(
                    dst,
                    epilogue_nodes,  # type: ignore[arg-type]
                    offsets,
                    reindexers,
                )
        else:
            if dst.get_name() != src.get_name():
                # src is local
                copy = L.copy(dst, src).data.data
                with LocalBufferContext(self.args) as scope:
                    scope.add_local_buffer(src)

                    return self.store_pointwise_nodes(dst, [copy])
            else:
                if dst.layout != src.layout:
                    raise AssertionError(f"{dst=}, {src=}")
                return ""

    def store_outputs(
        self,
        dst: tuple[ir.Buffer],
        src: tuple[ir.IRNode],
        orig_src: tuple[ir.IRNode] | None = None,
        epilogue_nodes: list[ir.IRNode] | None = None,
        offsets: list[Any] | None = None,
        reindexers: list[Callable[[list[Any]], list[Any]] | None] | None = None,
        multi_output_buffers: tuple[ir.MultiOutput, ...] | None = None,
    ):
        if not isinstance(dst, Iterable):
            raise AssertionError(f"expected Iterable, got {type(dst)}")
        if not all(_dst.get_size() == _src.get_size() for _src, _dst in zip(src, dst)):
            raise AssertionError("src and dst sizes must match")
        if offsets:
            offsets = parse_expr_with_index_symbols(offsets)
        gemm_num = len(src)
        final_offsets = []
        output_names = []
        if epilogue_nodes:
            if not reindexers:
                reindexers = [None] * len(epilogue_nodes)
            with LocalBufferContext(self.args) as scope:
                if orig_src is None:
                    raise AssertionError("orig_src must not be None")
                localize_epilogue_nodes = []
                all_read_names = []
                for epilogue in epilogue_nodes:
                    all_read_names.extend(list(epilogue.get_read_names()))
                localize_epilogue_nodes.extend(scope.localize_nodes(epilogue_nodes))
                final_offsets.extend([offsets] * len(localize_epilogue_nodes))
                output_names.extend(
                    [node.get_name() for node in localize_epilogue_nodes]
                )
                for gemm_idx in range(gemm_num):
                    if orig_src[gemm_idx].get_name() != src[gemm_idx].get_name():
                        if orig_src[gemm_idx].get_name() in all_read_names or (
                            multi_output_buffers
                            and multi_output_buffers[gemm_idx].get_name()
                            in all_read_names
                        ):
                            # If any of the Epilogue nodes use this GEMM output, let's localize the GEMM output
                            global_buffers = [orig_src[gemm_idx]]
                            if (
                                multi_output_buffers
                                and multi_output_buffers[gemm_idx].get_name()
                                in all_read_names
                                and orig_src[gemm_idx].get_name() not in all_read_names
                            ):
                                # Epilogue might directly read the MultiOutput, Localize MultiOutput to the local Buffer
                                # if this MultiOutput has not been stored by in-template epilogue
                                # otherwise, use the cse store cache if it will be stored before used
                                global_buffers.append(multi_output_buffers[gemm_idx])
                            scope.add_local_buffer(
                                src[gemm_idx],
                                global_buffers,
                            )
                        else:
                            scope.add_local_buffer(src[gemm_idx])
                            localize_epilogue_nodes.extend(
                                [L.copy(dst[gemm_idx], src[gemm_idx]).data.data]
                            )
                            reindexers.append(None)
                            output_names.append(dst[gemm_idx].get_name())
                            final_offsets.append(
                                [sympy.S.Zero] * len(dst[gemm_idx].get_size())
                            )
                res = self.store_grouped_gemm_pointwise_nodes(
                    dst,
                    localize_epilogue_nodes,
                    final_offsets,
                    reindexers,
                    output_names=output_names,
                )
                for gemm_idx in range(gemm_num):
                    if (
                        multi_output_buffers
                        and multi_output_buffers[gemm_idx].get_name() in all_read_names
                    ):
                        # If the MultiOutput is used in the Epilogue, let's remove it from args
                        multi_output_name = multi_output_buffers[gemm_idx].get_name()
                        if (
                            multi_output_name in self.args.output_buffers
                            and self.args.output_buffers[multi_output_name]
                            is not REMOVED
                        ):
                            self.remove_buffer(multi_output_name)
                return res
        else:
            if dst[0].get_name() != src[0].get_name():
                copy_list = []
                with LocalBufferContext(self.args) as scope:
                    for _src, _dst in zip(src, dst):
                        copy_list.extend([L.copy(_dst, _src).data.data])
                        scope.add_local_buffer(_src)
                        output_names.append(_dst.get_name())
                        final_offsets.append([sympy.S.Zero] * len(_dst.get_size()))
                    reindexers = [None] * len(copy_list)
                    return self.store_grouped_gemm_pointwise_nodes(
                        dst,
                        nodes=copy_list,
                        offsets=final_offsets,
                        reindexers=reindexers,
                        output_names=output_names,
                    )
            else:
                if not all(
                    _src.get_name() == _dst.get_name() for _src, _dst in zip(src, dst)
                ):
                    raise AssertionError("src and dst names must match")
                if not all(
                    _src.get_layout() == _dst.get_layout()
                    for _src, _dst in zip(src, dst)
                ):
                    raise AssertionError("src and dst layouts must match")
                return ""

    def check_bounds(self, expr, size, lower, upper):
        # CppTemplateKernel does not need codegen related operations
        return


class CppTemplateCaller(ir.ChoiceCaller):
    """
    CppTemplateCaller

    This class represents a caller for CPP template kernels. It is a subclass of ir.ChoiceCaller.
    Attributes:
        name (str): The name of the caller.
        category (str): The category of the caller.
        bmreq (CppBenchmarkRequest): The benchmark request for the caller.
        template_buffer (ir.CppTemplateBuffer): The template buffer for the caller.
    """

    def __init__(
        self,
        name: str,
        category: str,
        input_nodes: list[ir.Buffer],
        layout: ir.Layout,
        make_kernel_render: Callable[
            [
                ir.CppTemplateBuffer,
                bool,
                list[ir.IRNode] | None,
            ],
            str,
        ],
        bmreq: CppBenchmarkRequest,
        template: "CppTemplate",  # type: ignore[name-defined]  # noqa: F821
        info_kwargs: dict[str, ir.PrimitiveInfoType | list[ir.PrimitiveInfoType]]
        | None = None,
    ):
        super().__init__(name, input_nodes, layout, description="")
        self.category = category
        self.make_kernel_render = make_kernel_render
        self.bmreq = bmreq
        self.template = template
        self.info_kwargs = info_kwargs

    def precompile(self) -> None:
        if self.bmreq is None:
            raise AssertionError("bmreq must not be None")
        self.bmreq.precompile()

    def benchmark(self, *args, out) -> float:
        if self.bmreq is None:
            raise AssertionError("bmreq must not be None")
        if config.profile_bandwidth_with_do_bench_using_profiling:
            algo = self.bmreq.make_run_fn(*args, out=out)
            return do_bench_using_profiling(algo)
        return self.bmreq.benchmark(*args, out=out)

    def hash_key(self) -> str:
        return "-".join(
            [
                self.category,
                self.bmreq.hash_key,
            ]
        )

    def info_dict(
        self,
    ) -> dict[str, ir.PrimitiveInfoType | list[ir.PrimitiveInfoType]]:
        return {"backend": "CPP", "op_type": "unknown"}

    def output_node(self) -> ir.TensorBox:
        buffer = ir.CppTemplateBuffer(
            layout=self.layout,
            inputs=self.input_nodes,
            make_kernel_render=self.make_kernel_render,
            template=self.template,
            choice=self,
        )
        # Pass KTC annotation to the buffer for encoding
        if "ktc" in self.annotations:
            buffer.annotations["ktc"] = self.annotations["ktc"]
        return ir.TensorBox.create(buffer)
<|endoftext|>import collections
import functools
import textwrap

import sympy
from sympy import Expr, Symbol

from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv, ModularIndexing

from ..utils import sympy_dot, sympy_subs
from ..virtualized import V


class BlockPatternMatcher:
    """
    Matches block indexing expressions.
    """

    _indexing_wild_signed_int = functools.partial(
        sympy.Wild, properties=[lambda x: x.is_integer]
    )
    _indexing_wild_unsigned_int = functools.partial(
        sympy.Wild, properties=[lambda x: x.is_integer and x.is_nonnegative]
    )

    @classmethod
    def get_subexpr_involving_symbol(cls, expr: Expr, symbol: Symbol) -> Expr:
        """
        Given a sympy expression, return the subexpression comprised only of terms
        involving the specified symbol.

        For example, if `expr` is `x * 5 + x ** 2 + y * 2 + 5`, and `symbol` is `x`,
        this returns `x * 5 + x ** 2`.
        """
        expr = cls._preprocess(expr)
        return sympy.S.Zero + sum(
            term for term in sympy.Add.make_args(expr) if symbol in term.free_symbols
        )

    @staticmethod
    def factor_index_expr(expr: sympy.Expr, index_var: Symbol) -> sympy.Expr:
        """
        Given an index expression, factor the expression around
        - FloorDiv(index_var, ...)
        - ModularIndexing(index_var, ...)
        - xindex

        e.g. FloorDiv(index_var, d0)*s0 + FloorDiv(index_var, d0)*s1 ->
        FloorDiv(index_var, d0) * (s0 + s1)
        """
        centres = OrderedSet()
        for sub in sympy.preorder_traversal(expr):
            if isinstance(sub, FloorDiv) and sub.args[0] == index_var:
                centres.add(sub)
            elif isinstance(sub, ModularIndexing) and sub.args[0] == index_var:
                centres.add(sub)
        centres.add(index_var)

        expr_out = expr
        for c in centres:
            expr_out = sympy.collect(expr_out, c)
        return expr_out

    @staticmethod
    def get_slice_numels(dims: list[Expr]) -> list[Expr]:
        """
        Compute the cumulative size of each dimension's slice.
        This proceeds from the last dim up to the second.
        """
        numels = collections.deque([sympy.S.One])
        for dim in dims[:0:-1]:
            numel = dim * numels[0]
            numels.appendleft(numel)
        return [*numels]

    @staticmethod
    def _preprocess(expr: Expr) -> Expr:
        # Remove any Identity nodes, e.g. expand x + (5 * y) to x + 5 * y.
        # Disable mul and multinomial as those expansions affect op trees:
        # e.g. sympy expects to match Mul(a, b), but expansion has simplified
        # to Add(...). Even though they may be algebraically equivalent,
        # sympy `match` performs structural pattern matching
        return expr.expand(mul=False, multinomial=False, identity=True)

    @classmethod
    def match_mod_div_block_expr(
        cls,
        index: Expr,
        index_var: Symbol,
        numel: Expr,
        num_dims: int,
    ) -> tuple[list[Expr], list[Expr], list[Expr]] | None:
        """
        Matches modular indexing expressions, converting them to implied block dimensions and strides.
        See triton.py for more information.

        Warning: this function requires that `index`, `numel` and any other sympy
        expression does not have precomputed replacements since otherwise block
        pattern matching may fail.
        See [Note: Precomputed replacements with BlockPatternMatch]
        """
        index = cls._preprocess(index)

        # Pattern match to find the strides and offset.
        wild_unsigned_int = functools.partial(
            cls._indexing_wild_unsigned_int, exclude=[index_var]
        )
        wild_signed_int = functools.partial(
            cls._indexing_wild_signed_int, exclude=[index_var]
        )
        dims: list[Expr] = [
            wild_unsigned_int(f"dim_mod{idx}") for idx in range(num_dims)
        ]
        strides: list[Expr] = [
            wild_signed_int(f"stride_mod{idx}") for idx in range(num_dims)
        ]

        # The first dimension's index is computed by division.
        # The remaining are computed by modulo.
        slice_numels = cls.get_slice_numels(dims[:num_dims])
        block_index_exprs = [FloorDiv(index_var, slice_numels[0])] + [
            ModularIndexing(index_var, numel, dim)
            for dim, numel in zip(dims[1:], slice_numels[1:])
        ]

        # Calculate a linear index from block indices.
        match_expr = sympy_dot(strides, block_index_exprs)

        # Heuristic: if the number of dimensions is high, check that the minimum requirements
        # are met before attempting an expensive full match. see triton.py:match_mod_div_block
        # for more details. In short, here we check that each subexpression in sympy.Add contains
        # only FloorDiv or ModularIndexing expressions.
        if num_dims >= 5:
            stride = sympy.symbols("stride", cls=wild_signed_int)
            denom, other = sympy.symbols("denominator other", cls=wild_unsigned_int)
            mod_div_pattern = stride * ModularIndexing(index_var, denom, other)
            floor_div_pattern = stride * FloorDiv(index_var, denom)
            first_dim_floor_div_matched = False
            match_failed = False
            for arg in sympy.Add.make_args(index):
                if arg.match(floor_div_pattern):
                    # There should only be a single FloorDiv(index, denom) expression
                    # corresponding to the first dimension
                    if first_dim_floor_div_matched:
                        match_failed = True
                        break
                    first_dim_floor_div_matched = True
                elif arg.match(mod_div_pattern):
                    continue
                else:
                    match_failed = True
                    break

            if match_failed:
                return None

        # Pattern match.
        match = index.match(match_expr)
        if match is None:
            return None

        # Provide default values for unmatched dims and strides.
        for dim in dims[1:]:
            if dim not in match:
                match[dim] = sympy.S.One
        for stride in strides[1:]:
            if stride not in match:
                match[stride] = sympy.S.Zero

        # Replace wildcards with matched expressions.
        dims = [dims[0]] + [match[dim] for dim in dims[1:]]
        strides = [match[stride] for stride in strides]
        slice_numels = cls.get_slice_numels(dims)
        block_index_exprs = [sympy_subs(expr, match) for expr in block_index_exprs]

        sizevars = V.graph.sizevars

        # The leading dimension is not directly matched in our expression.
        # We solve for it by dividing the range tree numel by the product of
        # all other dimensions. We quit if they are not known to be divisible.
        if dims[0] in match:
            raise AssertionError("Expected not to match the leading dimension!")
        if not sizevars.statically_known_multiple_of(numel, slice_numels[0]):
            return None
        dims[0] = numel / slice_numels[0]

        # Sanity check that we can recover the index from the matched subexpressions.
        matched_index = sympy_dot(strides, block_index_exprs)
        if not sizevars.statically_known_equals(
            matched_index,
            index,
        ):
            raise AssertionError(
                textwrap.dedent(
                    f"""
                    Invalid match!
                    Index: {index}
                    Matched expression: {matched_index}
                    """
                )
            )

        return dims, strides, block_index_exprs

    @classmethod
    def match_affine_block_expr(
        cls,
        index: Expr,
        index_var: Symbol,
    ) -> Expr | None:
        """
        Matches simple expressions of the form stride * index, returning the
        stride.
        """
        index = cls._preprocess(index)
        stride = cls._indexing_wild_signed_int(name="stride", exclude=[index_var])
        m = index.match(index_var * stride)
        if m is None:
            return None

        return m[stride]
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from ..ir import MultiTemplateBuffer
from ..scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
    FusedSchedulerNode,
    Scheduler,
    SchedulerNode,
)
from .cutedsl.cutedsl_scheduling import CuteDSLScheduling
from .cutlass.scheduling import CUTLASSScheduling
from .nv_universal_gemm.nv_universal_gemm_scheduling import NVUniversalGemmScheduling
from .rocm.rocm_cpp_scheduling import ROCmCPPScheduling
from .triton import TritonScheduling


log = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TypeAlias

    from sympy import Expr

    import torch
    from torch.utils._ordered_set import OrderedSet

    from .common import BackendFeature

    _IntLike: TypeAlias = int | Expr


class CUDACombinedScheduling(BaseScheduling):
    """
    Scheduler for CUDA Kernels, which delegates calls as appropriate
    to the CUDA-C++ and Triton Schedulers, which both work for CUDA devices
    and use a unified-wrapper for codegen.

    If Scheduling code needs to be specialized for the case of mixed Triton / CUDA C++ code,
    this would also be the place to do it.
    """

    def __init__(self, scheduler: Scheduler | None) -> None:
        super().__init__(scheduler)
        self._triton_scheduling = TritonScheduling(scheduler)
        self._cutlass_scheduling = CUTLASSScheduling(scheduler)
        self._rocm_cpp_scheduling = ROCmCPPScheduling(scheduler)
        self._cutedsl_scheduling = CuteDSLScheduling(scheduler)
        self._nv_universal_gemm_scheduling = NVUniversalGemmScheduling(scheduler)

    def get_backend_features(self, device: torch.device) -> OrderedSet[BackendFeature]:
        return self._triton_scheduling.get_backend_features(device)

    def choose_node_backend(self, node: BaseSchedulerNode) -> BaseScheduling:
        if self._cutlass_scheduling.is_cutlass_template(node):
            return self._cutlass_scheduling
        if self._rocm_cpp_scheduling.is_rocm_cpp_template(node):
            return self._rocm_cpp_scheduling
        if self._cutedsl_scheduling.is_cutedsl_template(node):
            return self._cutedsl_scheduling
        if self._nv_universal_gemm_scheduling.is_nv_universal_gemm_template(node):
            return self._nv_universal_gemm_scheduling
        if self._nv_universal_gemm_scheduling.is_nv_universal_gemm_fused_template(node):
            return self._nv_universal_gemm_scheduling
        return self._triton_scheduling

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        if self._cutlass_scheduling.can_fuse_vertical(node1, node2):
            return True
        elif self._cutlass_scheduling.is_cutlass_template(
            node1
        ) or self._cutlass_scheduling.is_cutlass_template(node2):
            return False
        # CuteDSL doesn't support vertical fusion currently
        elif self._cutedsl_scheduling.is_cutedsl_template(
            node1
        ) or self._cutedsl_scheduling.is_cutedsl_template(node2):
            return False
        # Only intercept when node1 is the NVGEMM template (epilogue direction).
        # Prologue direction (node1=pointwise, node2=template) must fall through to
        # Triton, or NVGEMM-winning MTBs silently lose Triton prologue fusion.
        # For MTBs, if NVGEMM can't fuse, fall through to Triton scheduling so
        # Triton choices in the same MTB can still attempt epilogue fusion.
        elif self._nv_universal_gemm_scheduling.is_nv_universal_gemm_template(node1):
            if self._nv_universal_gemm_scheduling.can_fuse_vertical(node1, node2):
                return True
            if isinstance(node1, SchedulerNode) and isinstance(
                node1.node, MultiTemplateBuffer
            ):
                return self._triton_scheduling.can_fuse_vertical(node1, node2)
            return False
        elif self._nv_universal_gemm_scheduling.is_nv_universal_gemm_fused_template(
            node1
        ):
            if self._nv_universal_gemm_scheduling.is_nv_universal_gemm_fused_template(
                node2
            ):
                return False
            return self._nv_universal_gemm_scheduling.can_fuse_vertical(node1, node2)
        return self._triton_scheduling.can_fuse_vertical(node1, node2)

    def can_fuse_horizontal(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        for node in (node1, node2):
            if self._cutlass_scheduling.is_cutlass_template(node):
                return self._cutlass_scheduling.can_fuse_horizontal(
                    node1, node2
                )  # always False at the moment
            if self._cutedsl_scheduling.is_cutedsl_template(node):
                return self._cutedsl_scheduling.can_fuse_horizontal(
                    node1, node2
                )  # always False at the moment
            if self._nv_universal_gemm_scheduling.is_nv_universal_gemm_template(
                node
            ) or self._nv_universal_gemm_scheduling.is_nv_universal_gemm_fused_template(
                node
            ):
                return self._nv_universal_gemm_scheduling.can_fuse_horizontal(
                    node1, node2
                )  # always False at the moment
        return self._triton_scheduling.can_fuse_horizontal(node1, node2)

    def can_fuse_reduction_epilogue(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        if self._nv_universal_gemm_scheduling.is_nv_universal_gemm_template(
            node1
        ) or self._nv_universal_gemm_scheduling.is_nv_universal_gemm_fused_template(
            node1
        ):
            return self._nv_universal_gemm_scheduling.can_fuse_reduction_epilogue(
                node1, node2
            )
        return False

    def group_fn(
        self, sizes: Sequence[Sequence[_IntLike]]
    ) -> tuple[tuple[_IntLike, ...], ...]:
        return self._triton_scheduling.group_fn(sizes)

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
    ) -> str | None:
        if self._cutlass_scheduling.is_cutlass_template(template_node):
            if prologue_nodes:
                raise AssertionError("cutlass template does not support prologue nodes")
            return self._cutlass_scheduling.codegen_template(
                template_node, epilogue_nodes, prologue_nodes
            )
        elif self._rocm_cpp_scheduling.is_rocm_cpp_template(template_node):
            if epilogue_nodes:
                raise AssertionError(
                    "rocm cpp template does not support epilogue nodes"
                )
            if prologue_nodes:
                raise AssertionError(
                    "rocm cpp template does not support prologue nodes"
                )
            return self._rocm_cpp_scheduling.codegen_template(
                template_node, epilogue_nodes, prologue_nodes
            )
        elif self._cutedsl_scheduling.is_cutedsl_template(template_node):
            # TODO remove this when we add epilogue support
            if epilogue_nodes:
                raise AssertionError("cutedsl template does not support epilogue nodes")
            if prologue_nodes:
                raise AssertionError("cutedsl template does not support prologue nodes")
            return self._cutedsl_scheduling.codegen_template(
                template_node, epilogue_nodes, prologue_nodes
            )
        elif self._nv_universal_gemm_scheduling.is_nv_universal_gemm_template(
            template_node
        ):
            if prologue_nodes:
                raise AssertionError(
                    "NVIDIA Universal GEMM doesn't support prologue fusion yet"
                )
            return self._nv_universal_gemm_scheduling.codegen_template(
                template_node, epilogue_nodes, prologue_nodes
            )
        else:
            return self._triton_scheduling.codegen_template(
                template_node, epilogue_nodes, prologue_nodes
            )

    def codegen_mix_order_reduction(self, node):
        return self._triton_scheduling.codegen_mix_order_reduction(node)

    def codegen_nested_reduction(self, node):
        return self._triton_scheduling.codegen_nested_reduction(node)

    def codegen_node(self, node: FusedSchedulerNode | SchedulerNode) -> None:
        return self._triton_scheduling.codegen_node(node)

    def codegen_sync(self) -> None:
        return self._triton_scheduling.codegen_sync()

    def flush(self) -> None:
        return self._triton_scheduling.flush()

    def codegen_combo_kernel(self, *args: Any, **kwargs: Any) -> None:
        return self._triton_scheduling.codegen_combo_kernel(*args, **kwargs)

    def benchmark_fused_nodes(
        self, nodes: Sequence[BaseSchedulerNode]
    ) -> tuple[float, str]:
        return self._triton_scheduling.benchmark_fused_nodes(nodes)

    def benchmark_codegened_module(self, module):
        if getattr(module, "is_nvgemm", False):
            return self._benchmark_nvgemm_module(module)
        return self._triton_scheduling.benchmark_codegened_module(module)

    def _benchmark_nvgemm_module(self, module) -> tuple[float, str]:
        from torch._dynamo.utils import preserve_rng_state
        from torch._inductor.runtime.benchmarking import benchmarker
        from torch._inductor.utils import get_interface_for_device
        from torch._inductor.virtualized import V

        device_interface = get_interface_for_device(V.graph.device_type)

        with (
            preserve_rng_state(),
            device_interface.device(V.graph.get_current_device_or_throw()),
        ):
            args = module.get_args()
            call = module.call

            try:
                call(args)
            except Exception as e:
                log.debug(
                    "Exception (%s) in compiling NVGEMM fused kernel",
                    e,
                )
                return float("inf"), module.__file__ or ""

            device = V.graph.get_current_device_or_throw()
            try:
                ms = benchmarker.benchmark(
                    lambda: call(args),
                    device=device,
                )
            except Exception as e:
                log.debug(
                    "Exception (%s) while benchmarking NVGEMM fused kernel",
                    e,
                )
                return float("inf"), module.__file__ or ""

            return ms, module.__file__ or ""

    def _is_nvgemm_node(self, node) -> bool:
        """Check if a template node is currently configured for NVGEMM codegen."""
        from torch._inductor.ir import MultiTemplateBuffer, NVUniversalGemmBuffer

        template_node = node.get_template_node()
        if isinstance(template_node, NVUniversalGemmBuffer):
            return True
        if isinstance(template_node, MultiTemplateBuffer):
            return template_node._render_kind == "nvgemm"
        return False

    def generate_kernel_code_from_nodes(
        self,
        nodes: Sequence[Any],
        benchmark_kernel: bool = False,
        hint_override: int | None = None,
    ) -> str:
        for node in nodes:
            if not (hasattr(node, "is_template") and node.is_template()):
                continue
            if self._is_nvgemm_node(node):
                return (
                    self._nv_universal_gemm_scheduling.generate_kernel_code_from_nodes(
                        nodes, benchmark_kernel, hint_override=hint_override
                    )
                )
            break
        return self._triton_scheduling.generate_kernel_code_from_nodes(
            nodes, benchmark_kernel, hint_override=hint_override
        )

    def benchmark_combo_kernel(
        self, node_list: Sequence[BaseSchedulerNode], node_benchmark_results
    ) -> tuple[float, float, list[str | None]]:
        return self._triton_scheduling.benchmark_combo_kernel(
            node_list, node_benchmark_results
        )
<|endoftext|># mypy: allow-untyped-defs
from collections.abc import Callable, Sequence
from typing import Any

import sympy

import torch
import torch._inductor.async_compile
import torch._ops
from torch.utils._ordered_set import OrderedSet

from .. import config, ir
from ..utils import IndentedBuffer, sympy_product
from ..virtualized import V
from .cpp_utils import DTYPE_TO_CPP
from .cpp_wrapper_cpu import CppWrapperCpu
from .wrapper import (
    BufferLike,
    EnterSubgraphLine,
    ExitSubgraphLine,
    MemoryPlanningLine,
    MemoryPlanningState,
    PythonWrapperCodegen,
)


BufferName = str

# Default thread stack sizes vary by platform:
# - Linux: 8 MB
# - macOS: 512 KB
# - Windows: 1 MB
# Just pick something comfortably smaller than the smallest for now.
MAX_STACK_ALLOCATION_SIZE = 1024 * 100


class CppWrapperCpuArrayRef(CppWrapperCpu):
    """
    Generates cpp wrapper for running on CPU and calls cpp kernels

    This class is forked from CppWrapperCpu, with a difference that tensors may be
    represented as ArrayRef, see torch/csrc/inductor/aoti_runtime/arrayref_tensor.h
    """

    def __init__(self):
        super().__init__()
        if self.device != "cpu":
            raise AssertionError("ArrayRefTensor only supported on CPU!")
        self.allow_stack_allocation = config.aot_inductor.allow_stack_allocation
        self.stack_allocated_buffers: dict[BufferName, BufferLike] = {}
        self.v2_raw_wrapper_body = IndentedBuffer()
        self.v2_raw_output_refs: list[str] | None = None

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: str | None,
        parent_wrapper: PythonWrapperCodegen | None,
        partition_signatures: ir.GraphPartitionSignature | None = None,
    ):
        # TODO - support subgraph codegen by lifting functions. Check the
        # comment at CppWrapperCpu `codegen_subgraph` function.
        return CppWrapperCpuArrayRef()

    @staticmethod
    def get_input_cpp_type(input):
        if not config.aot_inductor.use_minimal_arrayref_interface:
            raise AssertionError(
                "expected config.aot_inductor.use_minimal_arrayref_interface to be set"
            )

        if isinstance(input, sympy.Expr):
            from ..graph import may_get_constant_buffer_dtype

            dtype = may_get_constant_buffer_dtype(input)
            if dtype is None:
                raise AssertionError(f"Failed to get the dtype of sympy.Expr: {input}")
            return DTYPE_TO_CPP[dtype]
        return f"ArrayRefTensor<{DTYPE_TO_CPP[input.get_dtype()]}>"

    @staticmethod
    def get_input_element_cpp_type(input):
        if isinstance(input, sympy.Expr):
            from ..graph import may_get_constant_buffer_dtype

            dtype = may_get_constant_buffer_dtype(input)
            if dtype is None:
                raise AssertionError(f"Failed to get the dtype of sympy.Expr: {input}")
            return DTYPE_TO_CPP[dtype]
        return DTYPE_TO_CPP[input.get_dtype()]

    @staticmethod
    def get_device_include_path_jit(device: str) -> str:
        if device != "cpu":
            raise AssertionError("ArrayRef only supported on CPU!")
        return "#include <torch/csrc/inductor/cpp_wrapper/array_ref.h>"

    @staticmethod
    def get_device_include_path_aot(device: str) -> str:
        if device != "cpu":
            raise AssertionError("ArrayRef only supported on CPU!")
        return "#include <torch/csrc/inductor/aoti_include/array_ref.h>"

    def codegen_input_numel_asserts(self, indented_buffer=None):
        writer = indented_buffer or self.prefix
        for name, buf in V.graph.graph_inputs.items():
            if isinstance(buf, sympy.Expr):
                continue

            # comparing strides for 0 size tensor is tricky. Ignore them for now.
            if sympy_product(buf.get_size()) == 0:
                continue
            numel = buf.get_numel()
            writer.writeline(f"assert_numel({name}, {numel});")

    def _codegen_assert_size_stride(
        self,
        code: IndentedBuffer,
        name: str,
        size: str,
        stride: str,
        op_name: str,
        dtype: torch.dtype | None = None,
    ) -> None:
        # Inputs/outputs are ArrayRefTensor, not AtenTensorHandle, so
        # assert_size_stride would fail to compile.
        return

    def _codegen_v2_raw_input_bindings(self, code: IndentedBuffer):
        for idx, (input_key, input_value) in enumerate(V.graph.graph_inputs.items()):
            input_cpp_type = CppWrapperCpuArrayRef.get_input_element_cpp_type(
                input_value
            )
            if isinstance(input_value, sympy.Expr):
                # cond / symint wrappers can surface symbolic scalar inputs here.
                from ..graph import may_get_constant_buffer_dtype

                dtype = may_get_constant_buffer_dtype(input_value)
                if dtype is None:
                    raise AssertionError("Fails to get the dtype of the sympy.Expr")
                input_tensor = f"{input_key}_arrayref_tensor"
                code.writeline(
                    f"auto {input_tensor} = torch::aot_inductor::c_to_arrayref_tensor<{input_cpp_type}>(c_inputs[{idx}]);"
                )
                self.codegen_tensor_item(dtype, input_tensor, input_key, code)
            else:
                code.writeline(
                    f"auto {input_key} = torch::aot_inductor::c_to_arrayref_tensor<{input_cpp_type}>(c_inputs[{idx}]);"
                )

    def _codegen_v2_raw_input_symbols(self, code: IndentedBuffer) -> None:
        bound_vars = OrderedSet[sympy.Symbol]()
        graph_inputs = self.get_graph_inputs()
        inputs = [
            (k, v) for k, v in graph_inputs.items() if isinstance(v, sympy.Symbol)
        ] + [(k, v) for k, v in graph_inputs.items() if not isinstance(v, sympy.Symbol)]

        # Redirect self.prefix so the base class codegen_input_symbol_assignment
        # writes into our buffer.
        deferred_symbol_assignments = []
        with self._target_buf("prefix", code):
            for name, value in inputs:
                self.codegen_input_symbol_assignment(
                    name, value, bound_vars, deferred_symbol_assignments
                )
            self._retry_deferred_symbol_assignments(deferred_symbol_assignments)

        for _, value in inputs:
            if not isinstance(value, ir.TensorBox):
                continue
            for expr in [*value.get_size(), *value.get_stride()]:
                if not isinstance(expr, sympy.Expr) or isinstance(expr, sympy.Symbol):
                    continue
                undefined_symbols = [
                    sym for sym in expr.free_symbols if sym not in bound_vars
                ]
                if len(undefined_symbols) > 0:
                    raise AssertionError(
                        f"For {expr}, expected {undefined_symbols} to have been codegen-ed."
                    )

    def _codegen_v2_raw_prelude(self, code: IndentedBuffer):
        self._codegen_v2_raw_input_bindings(code)

        if not all(
            isinstance(v, torch.Tensor) for v in list(V.graph.constants.values())
        ):
            raise AssertionError("Expect all constants to be Tensor")
        for idx, constants_key in enumerate(V.graph.constants.keys()):
            code.writeline(f"""auto {constants_key} = constants_->at({idx});""")

        self._codegen_v2_raw_input_symbols(code)

        self.codegen_input_numel_asserts(code)
        code.writeline(
            "[[maybe_unused]] auto& kernels = static_cast<AOTInductorModelKernels&>(*this->kernels_.get());"
        )

    def _codegen_v2_raw_outputs(
        self, code: IndentedBuffer, output_refs: list[str]
    ) -> None:
        cst_names = V.graph.constants.keys()

        def write_output_to_c_array(idx: int, output: str) -> None:
            output_arrayref_name = f"output_arrayref_{idx}"
            code.splice(
                f"""
                std::tuple_element_t<{idx}, AOTInductorModelOutputs> {output_arrayref_name};
                convert_handle_to_arrayref_tensor({output}, {output_arrayref_name});
                torch::aot_inductor::arrayref_tensor_to_c({output_arrayref_name}, c_outputs[{idx}]);
                """
            )

        for idx, output in enumerate(output_refs):
            if output == "nullptr":
                continue

            is_constant_buffer = output in cst_names
            output_buffer = V.graph.graph_outputs[idx]
            if isinstance(output_buffer, ir.BaseView):
                output_storage = output_buffer.unwrap_view()
                if not isinstance(output_storage, (ir.BaseView, ir.MutableBox)):
                    raise AssertionError(
                        f"expected output_storage to be BaseView or MutableBox, got "
                        f"{type(output_storage).__name__}"
                    )
                if isinstance(output_storage.data, ir.ConstantBuffer):
                    is_constant_buffer = True

            if isinstance(output_buffer, ir.ShapeAsConstantBuffer):
                output_tensor = f"scalar_to_tensor_{next(self.scalar_to_tensor_id)}"
                code.writeline(
                    f"RAIIAtenTensorHandle {output_tensor} = scalar_to_tensor_handle({output});"
                )
                write_output_to_c_array(idx, output_tensor)
                continue

            output_is_tensor_handle_expr = (
                f"std::is_same_v<std::decay_t<decltype({output})>,"
                "RAIIAtenTensorHandle> || "
                f"std::is_same_v<std::decay_t<decltype({output})>,"
                "AtenTensorHandle> || "
                f"std::is_same_v<std::decay_t<decltype({output})>,"
                "ConstantHandle>"
            )
            code.writeline(f"if constexpr ({output_is_tensor_handle_expr}) {{")
            with code.indent():
                cached_output_name = f"cached_output_{next(self.cached_output_id)}"
                code.writeline(
                    f"thread_local RAIIAtenTensorHandle {cached_output_name};"
                )
                if is_constant_buffer:
                    code.splice(
                        f"""
                        AtenTensorHandle {cached_output_name}_tmp;
                        AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_clone({output}, &{cached_output_name}_tmp));
                        {cached_output_name} = {cached_output_name}_tmp;
                        """
                    )
                else:
                    code.writeline(f"{cached_output_name} = {output}.release();")
                write_output_to_c_array(idx, cached_output_name)
            code.writeline("} else {")
            with code.indent():
                cached_output_name = f"cached_output_{next(self.cached_output_id)}"
                output_arrayref_type = f"output_arrayref_{idx}_type"
                output_element_type = f"output_arrayref_{idx}_element_type"
                output_arrayref_name = f"output_arrayref_{idx}"
                code.splice(
                    f"""
                    thread_local ThreadLocalCachedOutputArray<std::decay_t<decltype({output})>>
                        {cached_output_name}{{{output}}};
                    {cached_output_name}.copy_data_from({output});
                    using {output_arrayref_type} = std::tuple_element_t<{idx}, AOTInductorModelOutputs>;
                    using {output_element_type} = typename {output_arrayref_type}::value_type;
                    auto {output_arrayref_name} = {cached_output_name}.arrayref_tensor<{output_element_type}>();
                    torch::aot_inductor::arrayref_tensor_to_c({output_arrayref_name}, c_outputs[{idx}]);
                    """
                )
            code.writeline("}")

    def generate_extern_kernel_alloc(self, *args, **kwargs):
        # Disable stack allocation for extern kernels.
        self.allow_stack_allocation = False
        super().generate_extern_kernel_alloc(*args, **kwargs)

    def generate_extern_kernel_out(self, *args, **kwargs):
        # Disable stack allocation for extern kernels.
        self.allow_stack_allocation = False
        super().generate_extern_kernel_out(*args, **kwargs)

    def generate_fallback_kernel(self, node: ir.FallbackKernel) -> None:
        # Disable stack allocation for extern kernels.
        self.allow_stack_allocation = False
        super().generate_fallback_kernel(node)

    def _generate_kernel_call_helper(
        self,
        kernel_name: str,
        call_args,
        *,
        device=None,
        triton=True,
        arg_types=None,
        raw_keys=None,
        raw_args=None,
        triton_meta=None,
        inductor_meta=None,
        graph_name="",
        original_fxnode_name=None,
        current_stream_idx=None,
    ):
        """
        Generates kernel call code.

        triton: Defines whether the GPU backend uses Triton for codegen.
                Otherwise it uses the CUDA language for codegen.
                Only valid when cuda == True.
        """
        if triton:
            raise AssertionError(
                "CppWrapperCpuArrayRef.generate_kernel_call does not support GPU"
            )
        if not (arg_types is not None and len(call_args) == len(arg_types)):
            raise AssertionError(
                "Mismatch call_args and arg_types in generate_kernel_call"
            )
        new_args = []
        for idx, arg in enumerate(call_args):
            if "*" in arg_types[idx]:
                var_name = f"var_{next(self.arg_var_id)}"
                self.writeline(f"auto* {var_name} = get_data_ptr_wrapper({arg});")
                new_args.append(f"({arg_types[idx]})({var_name})")
            else:
                # arg is a scalar
                new_args.append(arg)
        # debug printer related logic for cpp kernel type.
        debug_printer_manager = V.graph.wrapper_code.debug_printer
        debug_printer_manager.set_printer_args(
            call_args,
            kernel_name,
            None,
            None,
            "cpp",
        )
        with debug_printer_manager:
            self.writeline(self.wrap_kernel_call(kernel_name, new_args))

    def write_wrapper_decl(self):
        """Declare the generated AOTI wrapper entry points."""
        inputs_len = len(V.graph.graph_inputs.keys())
        if V.graph.aot_mode:
            if (
                config.aot_inductor.use_minimal_arrayref_interface
                and not V.graph.is_const_graph
            ):
                input_cpp_types = ", ".join(
                    f"{CppWrapperCpuArrayRef.get_input_cpp_type(x)}"
                    for x in V.graph.graph_inputs.values()
                )
                output_arrayref_types = ", ".join(
                    f"ArrayRefTensor<{DTYPE_TO_CPP[x.get_dtype()]}>"
                    for x in V.graph.graph_outputs
                )

                self.prefix.splice(
                    f"""
                    using AOTInductorModelInputs = std::tuple<{input_cpp_types}>;
                    using AOTInductorModelOutputs = std::tuple<{output_arrayref_types}>;
                    """
                )

            if V.graph.const_module:
                self.header.splice(V.graph.const_module.wrapper_code.header)

                if V.graph.const_wrapper_code is None:
                    raise AssertionError(
                        "expected V.graph.const_wrapper_code to be set"
                    )
                self.prefix.splice(V.graph.const_wrapper_code)

                if V.graph.const_kernel_code is None:
                    raise AssertionError("expected V.graph.const_kernel_code to be set")
                self.kernel_declarations.splice(V.graph.const_kernel_code)

            if V.graph.is_const_graph:
                self.prefix.splice(
                    """
                    void AOTInductorModel::_const_run_impl(
                        std::vector<AtenTensorHandle>& output_handles,
                        DeviceStreamType stream,
                        AOTIProxyExecutorHandle proxy_executor
                    ) {
                    """
                )
            else:
                if not config.aot_inductor.use_runtime_constant_folding:
                    # If we do not split the constant graph, we'll just create
                    # an empty implementation when wrapping the main module.
                    self.prefix.splice(
                        """
                        void AOTInductorModel::_const_run_impl(
                            std::vector<AtenTensorHandle>& output_handles,
                            DeviceStreamType stream,
                            AOTIProxyExecutorHandle proxy_executor
                        ) {}

                        """
                    )

                run_impl_proto = """
                    void AOTInductorModel::run_impl(
                        AtenTensorHandle*
                            input_handles, // array of input AtenTensorHandle; handles
                                            // are stolen; the array itself is borrowed
                        AtenTensorHandle*
                            output_handles, // array for writing output AtenTensorHandle; handles
                                            // will be stolen by the caller; the array itself is
                                            // borrowed
                        DeviceStreamType stream,
                        AOTIProxyExecutorHandle proxy_executor
                    ) {
                    """

                self.generate_input_output_runtime_checks()
                run_impl_proto += """
                    __check_inputs_outputs(input_handles, output_handles);
                """

                if config.aot_inductor.use_minimal_arrayref_interface:
                    self.prefix.splice(
                        """
                        template <>
                        AOTInductorModelOutputs AOTInductorModel::run_impl_minimal_arrayref_interface<
                          AOTInductorModelInputs, AOTInductorModelOutputs>(
                            const AOTInductorModelInputs& inputs,
                            DeviceStreamType stream,
                            AOTIProxyExecutorHandle proxy_executor
                        ) {
                        """
                    )
                    self.suffix.splice(run_impl_proto)
                    self.suffix.splice(
                        """
                            AOTInductorModelInputs inputs;
                            convert_handles_to_inputs(input_handles, inputs);
                            auto outputs = run_impl_minimal_arrayref_interface<AOTInductorModelInputs, AOTInductorModelOutputs>(
                                inputs, stream, proxy_executor);
                            // NOTE: outputs is full of ArrayRef to thread_local storage. If in the future we need this
                            // interface to perform well for a DSO using the minimal arrayref interface, all we need
                            // to do is provide ThreadLocalCachedTensor for each one!
                            convert_outputs_to_handles(outputs, output_handles);
                        }
                    """
                    )

                    self.suffix.splice(
                        """
                        extern "C" AOTIRuntimeError AOTInductorModelRunMinimalArrayrefInterface(
                            AOTInductorModelHandle model_handle,
                            const AOTInductorModelInputs& inputs,
                            AOTInductorModelOutputs& outputs) {
                          auto model = reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
                          CONVERT_EXCEPTION_TO_ERROR_CODE({
                              outputs = model->run_impl_minimal_arrayref_interface<AOTInductorModelInputs, AOTInductorModelOutputs>(
                                  inputs,
                                  (torch::aot_inductor::DeviceStreamType)nullptr,
                                  nullptr);
                          })
                        }
                    """
                    )

                    self.suffix.splice(
                        f"""
                        // C-ABI-safe variant: uses flat AOTInductorArrayRefTensor arrays
                        // instead of std::tuple across the DSO boundary, and
                        // runs directly on the descriptor arrays to avoid
                        // DSO-side tuple marshaling.
                        extern "C" AOTIRuntimeError AOTInductorModelRunMinimalArrayrefInterfaceV2(
                            AOTInductorModelHandle model_handle,
                            int32_t num_inputs,
                            const AOTInductorArrayRefTensor* c_inputs,
                            int32_t num_outputs,
                            AOTInductorArrayRefTensor* c_outputs) {{
                          constexpr int32_t expected_num_inputs = {len(V.graph.graph_inputs)};
                          constexpr int32_t expected_num_outputs = {len(V.graph.graph_outputs)};
                          auto model = reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
                          CONVERT_EXCEPTION_TO_ERROR_CODE({{
                              if (num_inputs != expected_num_inputs) {{
                                throw std::runtime_error(
                                    std::string("AOTInductorModelRunMinimalArrayrefInterfaceV2 expected ")
                                    + std::to_string(expected_num_inputs)
                                    + " inputs but got "
                                    + std::to_string(num_inputs));
                              }}
                              if (num_outputs != expected_num_outputs) {{
                                throw std::runtime_error(
                                    std::string("AOTInductorModelRunMinimalArrayrefInterfaceV2 expected ")
                                    + std::to_string(expected_num_outputs)
                                    + " outputs but got "
                                    + std::to_string(num_outputs));
                              }}
                              if (num_inputs > 0 && c_inputs == nullptr) {{
                                throw std::runtime_error(
                                    "AOTInductorModelRunMinimalArrayrefInterfaceV2 received null input descriptors");
                              }}
                              if (num_outputs > 0 && c_outputs == nullptr) {{
                                throw std::runtime_error(
                                    "AOTInductorModelRunMinimalArrayrefInterfaceV2 received null output descriptors");
                              }}
                              model->run_impl_minimal_arrayref_interface_v2_raw(
                                  c_inputs,
                                  c_outputs,
                                  (torch::aot_inductor::DeviceStreamType)nullptr,
                                  nullptr);
                          }})
                        }}
                    """
                    )
                else:
                    self.prefix.splice(run_impl_proto)
        else:
            # cpp entry function for JIT with cpp wrapper
            self.prefix.splice(
                """
                void inductor_entry_impl(
                    AtenTensorHandle*
                        input_handles, // array of input AtenTensorHandle; handles
                                        // are stolen; the array itself is borrowed
                    AtenTensorHandle*
                        output_handles  // array for writing output AtenTensorHandle; handles
                                        // will be stolen by the caller; the array itself is
                                        // borrowed)
                ) {
                """
            )
        with self.prefix.indent():
            # assign inputs and outputs in both cases so the later codegen can be simplified
            if not config.aot_inductor.use_minimal_arrayref_interface:
                if not V.graph.is_const_graph:
                    if V.graph.aot_mode:
                        num_args = len(V.graph.graph_inputs)
                    else:
                        # Weights are promoted in the JIT mode
                        num_args = len(V.graph.graph_inputs) + len(V.graph.constants)
                        # release GIL to support multiple instances inference (in different threads of the same process)
                        self.prefix.splice("py::gil_scoped_release_simple release;")

                    self.prefix.splice(
                        f"""
                            auto inputs = steal_from_raw_handles_to_raii_handles(input_handles, {num_args});
                        """
                    )

            if inputs_len != 0:
                for idx, input_key in enumerate(V.graph.graph_inputs.keys()):
                    if config.aot_inductor.use_minimal_arrayref_interface:
                        self.prefix.writeline(
                            f"auto {input_key} = std::get<{idx}>(inputs);"
                        )
                        continue
                    # unwrap input tensor back to scalar
                    if isinstance(V.graph.graph_inputs[input_key], sympy.Expr):
                        from ..graph import may_get_constant_buffer_dtype

                        dtype = may_get_constant_buffer_dtype(
                            V.graph.graph_inputs[input_key]  # type: ignore[arg-type]
                        )
                        if dtype is None:
                            raise AssertionError(
                                "Fails to get the dtype of the sympy.Expr"
                            )
                        self.codegen_tensor_item(
                            dtype, f"inputs[{idx}]", input_key, self.prefix
                        )
                    else:
                        self.prefix.writeline(
                            f"auto {input_key} = std::move(inputs[{idx}]);"
                        )

            if not all(
                isinstance(v, torch.Tensor) for v in list(V.graph.constants.values())
            ):
                raise AssertionError("Expect all constants to be Tensor")
            for idx, constants_key in enumerate(V.graph.constants.keys()):
                if V.graph.aot_mode:
                    # Weights are stored in constants_ and owned by RAIIAtenTensorHandle there.
                    # Don't call std::move here because it will cause constants_ to lose the ownership.
                    self.prefix.writeline(
                        f"""auto {constants_key} = constants_->at({idx});"""
                    )
                else:
                    # Append constants as inputs to the graph
                    constants_idx = inputs_len + idx
                    self.prefix.writeline(
                        f"auto {constants_key} = std::move(inputs[{constants_idx}]);"
                    )

            self.codegen_inputs()

            if V.graph.aot_mode:
                if not V.graph.is_const_graph:
                    if config.aot_inductor.use_minimal_arrayref_interface:
                        # TODO: input shape checking for regular tensor interface as well?
                        self.codegen_input_numel_asserts()
                    else:
                        self.prefix.writeline("inputs.clear();")
                self.prefix.writeline(
                    "[[maybe_unused]] auto& kernels = static_cast<AOTInductorModelKernels&>(*this->kernels_.get());"
                )

    def generate_return(self, output_refs: list[str]):
        cst_names = V.graph.constants.keys()
        arr_iface = (
            not V.graph.is_const_graph
            and config.aot_inductor.use_minimal_arrayref_interface
        )  # For brevity.

        if arr_iface and V.graph.aot_mode:
            self.v2_raw_wrapper_body.clear()
            self.v2_raw_wrapper_body.splice(self.wrapper_call)
            self.v2_raw_output_refs = list(output_refs)

        def use_thread_local_cached_output_tensor(idx, output):
            cached_output_name = f"cached_output_{next(self.cached_output_id)}"
            cache_type = "Array" if arr_iface else "Tensor"
            self.wrapper_call.writeline(
                f"thread_local ThreadLocalCachedOutput{cache_type}<std::decay_t<decltype({output})>> "
                f"{cached_output_name}{{{output}}};"
            )
            if arr_iface:
                self.wrapper_call.writeline(
                    f"{cached_output_name}.copy_data_from({output});"
                )
                output_entry = f"std::get<{idx}>(output_arrayref_tensors)"
                element_type = f"std::decay_t<decltype({output_entry}.data()[0])>"
                self.wrapper_call.writeline(
                    f"{output_entry} = {cached_output_name}.arrayref_tensor<{element_type}>();"
                )
            else:
                self.wrapper_call.writeline(
                    f"{cached_output_name}.copy_data_from({output});"
                )
                self.wrapper_call.writeline(
                    f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_new_uninitialized_tensor(&output_handles[{idx}]));"
                )
                self.wrapper_call.writeline(
                    f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_assign_tensors({cached_output_name}.tensor(), "
                    f"output_handles[{idx}]));"
                )

        if arr_iface:
            self.wrapper_call.writeline(
                "AOTInductorModelOutputs output_arrayref_tensors;"
            )

        output2idx: dict[str, int] = {}
        for idx, output in enumerate(output_refs):
            if output == "nullptr":
                continue

            is_constant_buffer = output in cst_names
            output_buffer = V.graph.graph_outputs[idx]
