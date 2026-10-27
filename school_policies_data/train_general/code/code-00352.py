    def _alignment_match(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _set_bias_layout_and_alignment(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _define_gemm_instance(
        self,
        op: GemmOperation,
        evt_name: str | None = None,
    ) -> tuple[str, str]:
        raise NotImplementedError

    @abstractmethod
    def _get_extra_inputs_and_names(
        self,
        op: "cutlass_gemm_op.GemmOperation" = None,  # type: ignore[name-defined]  # noqa: F821
    ) -> tuple[Buffer | None, list[Buffer | None], list[str]]:
        raise NotImplementedError

    @abstractmethod
    def _update_arg_names_for_test_call_statement(
        self,
        arg_names: list[str],
        input_nodes: list[Buffer],
    ) -> list[str]:
        raise NotImplementedError

    def _add_cutlass_gemm_choices(
        self,
        choices: list[ChoiceCaller],
        layout: ir.Layout,
        input_nodes: list[Buffer],
        alpha: float | int = 1,
        beta: float | int = 0,
        input_reorder: list[int] | None = None,
        **extra_kwargs,
    ) -> None:
        """
        Adds Cutlass GEMM configurations choices to the auto-tuning list.

        This function mutates the passed list of choices by appending the choices for Cutlass GEMM configs to it.

        Args:
            choices (list): The list to which choices are appended.
            layout (ir.Layout): The layout configuration.
            input_nodes (list): The list of input nodes.
            alpha (float,int): Scaling factor, defaults to 1.
            beta (float,int): Offset, defaults to 0.
            input_reorder (list, optional): Order of the inputs, defaults to None.
            **extra_kwargs: Additional keyword arguments.

        """

        ops = self.gen_ops()

        # pre-computation
        layout_repr: str = str(layout)
        input_tensor_meta: TensorMeta | list[TensorMeta] = TensorMeta.from_irnodes(
            self.input_nodes
        )
        # When input_reorder is set (e.g. [2, 0, 1] for addmm), the kernel
        # function signature is reordered (e.g. from [X, W, Bias] to
        # [Bias, X, W]).  input_tensor_meta must follow the same order
        # because subprocess benchmarking creates tensors from this metadata
        # and passes them positionally to the compiled kernel.  Without this
        # reorder the kernel receives mismatched pointers/strides, causing
        # out-of-bounds GPU memory access for large shapes.
        if self.input_reorder is not None and isinstance(input_tensor_meta, list):
            input_tensor_meta = [input_tensor_meta[idx] for idx in self.input_reorder]
        output_tensor_meta: TensorMeta | list[TensorMeta] = TensorMeta.from_irnodes(
            self.output_node
        )

        with dynamo_timed("CUTLASSGemmTemplate.maybe_append_choice"):
            device_config = self._device_cutlass_config
            for name, op in ops:
                for swizzle in device_config.cutlass_max_profiling_swizzle_options:
                    description = f"{name} swizzle={swizzle}"
                    self.maybe_append_choice(
                        choices,
                        op=op,
                        name=name,
                        description=description,
                        input_key=self.cache_key,
                        layout_repr=layout_repr,
                        input_tensor_meta=input_tensor_meta,
                        output_tensor_meta=output_tensor_meta,
                        swizzle=swizzle,
                    )

        if len(ops) == 0:
            log.info(
                "No suitable Cutlass GEMM configs found, fallbacks used "
                "( len(ops)=%d, output_layout=%s, input_layouts=%s, input_strides=%s )",
                len(ops),
                layout,
                [node.get_layout() for node in input_nodes],
                [node.get_stride() for node in input_nodes],
            )
        log.debug(
            "Added %d Cutlass gemm configs.",
            len(ops),
        )

    def header(self) -> IndentedBuffer:
        """
        Returns a buffer containing CUDA C++ code for the header section of the CUTLASS GEMM template.
        This section primarily includes the necessary header files.

        Returns:
            IndentedBuffer: An instance of IndentedBuffer that contains the generated CUDA C++ header code.
        """
        res = super().header()
        res.splice(
            """
                #include "cutlass/gemm/gemm.h"
                #include "cutlass/gemm/device/gemm_universal.h"
                #include "cutlass/gemm/device/gemm_universal_adapter.h"
                #include "cutlass/gemm/kernel/gemm_universal.hpp"
                #include "cutlass/gemm/collective/collective_builder.hpp"
                #include "cutlass/epilogue/collective/collective_builder.hpp"
                #include "cutlass/epilogue/collective/default_epilogue.hpp"
                #include "cutlass/epilogue/thread/linear_combination.h"
                #include "cutlass/epilogue/thread/activation.h"
                #include "cutlass/gemm/dispatch_policy.hpp"
                #include "cutlass/gemm/kernel/tile_scheduler.hpp"
                #include "cutlass/tensor_ref.h"
                #include "cutlass/util/distribution.h"
                #include "cutlass/util/packed_stride.hpp"
                #include "cutlass/util/tensor_view_io.h"
            """
        )
        if self.device_type != "xpu":
            # XPU SYCL-TLA does not support sparse gemm yet
            res.splice(
                """
                #include "cutlass/gemm/device/gemm_sparse.h"
                """
            )
        if inductor_cutlass_config.generate_test_runner and not is_dynamic(
            *self.input_nodes, self.output_node
        ):
            res.splice(GEMM_STANDALONE_RUNNER_ADDITIONAL_INCLUDES)
        return res

    @staticmethod
    def cutlass_layout(torch_layout: ir.Layout) -> "cutlass_lib.LayoutType | None":  # type: ignore[name-defined]  # noqa: F821
        """
        Converts an ir.Layout instance into the corresponding cutlass_library.LayoutType enum value
        (RowMajor, ColumnMajor, or None if no matching value is found ).

        Args:
            torch_layout (ir.Layout): The layout that needs to be looked up.

        Returns:
            cutlass_lib.LayoutType: The converted layout corresponding to the `torch_layout` or None if no matching
            value is found.
        """
        if not cutlass_utils.try_import_cutlass():
            raise AssertionError("could not import cutlass")
        import cutlass_library.library as cutlass_lib

        if V.graph.sizevars.statically_known_equals(torch_layout.stride[-1], 1):
            return cutlass_lib.LayoutType.RowMajor
        elif V.graph.sizevars.statically_known_equals(torch_layout.stride[-2], 1):
            return cutlass_lib.LayoutType.ColumnMajor
        else:
            return None

    @staticmethod
    def flip_cutlass_layout(
        cutlass_layout: "cutlass_lib.LayoutType",  # type: ignore[name-defined]  # noqa: F821
    ) -> "cutlass_lib.LayoutType":  # type: ignore[name-defined]  # noqa: F821
        """Helper method: Flips a given cutlass layout (cutlass_lib.LayoutType) from RowMajor
        to ColumnMajor or vice versa"""
        if not cutlass_utils.try_import_cutlass():
            raise AssertionError("could not import cutlass")
        import cutlass_library.library as cutlass_lib

        if cutlass_layout == cutlass_lib.LayoutType.RowMajor:
            return cutlass_lib.LayoutType.ColumnMajor
        else:
            return cutlass_lib.LayoutType.RowMajor

    @staticmethod
    @functools.lru_cache(32)
    def layout_match(
        torch_layout: ir.Layout,
        cutlass_layout: "cutlass_lib.LayoutType",  # type: ignore[name-defined] # noqa: F821
    ) -> bool:
        """Helper Method: Determines whether a given torch layout matches a given Cutlass layout"""
        return CUTLASSGemmTemplate.cutlass_layout(torch_layout) == cutlass_layout

    @staticmethod
    def set_layout(tensor_desc: "TensorDescription", torch_layout: ir.Layout) -> None:  # type: ignore[name-defined]  # noqa: F821
        """
        Helper method: Sets the layout of a given tensor description to match the given torch layout
        """
        if CUTLASSGemmTemplate.layout_match(torch_layout, tensor_desc.layout):
            return
        tensor_desc.layout = CUTLASSGemmTemplate.cutlass_layout(torch_layout)

    @staticmethod
    def set_alignment(torch_layout, op_element) -> bool:
        """
        Helper method to update the alignment of a given CUTLASS GEMM op operand's element.

        This method modifies the alignment of the given Cutlass GEMM op operand's element to match the
        layout of the corresponding ir.Buffer node.

        Args:
            torch_layout: The layout of the corresponding ir.Buffer node.
            op_element: The Cutlass GEMM op operand's element whose alignment is to be updated.

        Returns:
            bool: True if the alignment was successfully updated, False otherwise.
        """
        alignment = cutlass_utils.get_max_alignment(torch_layout)
        if torch.cuda.is_available():
            cuda_arch = cuda_env.get_cuda_arch()
            cuda_arch = cutlass_utils._normalize_cuda_arch(cuda_arch)
            if cuda_arch and int(cuda_arch) >= 90 and alignment < op_element.alignment:
                return False

        op_element.alignment = alignment
        return True

    @staticmethod
    def should_swap_XW(
        bias: IRNode,
    ) -> bool:
        """
        Helper method to determine whether we should do an explicit transpose by switching the order of the
        matmul operands. This might be necessary when we can't otherwise arrive at the right memory
        layout for the given Bias operand.

        Note: This method is a workaround for CUDA Errors that seemingly non-deterministically
        occurred in practice in some CUTLASS GEMM Kernels with Linear epilogues that have a bias term.
        it might make sense to check on newer Cutlass releases whether it makes sense to keep
        returning True in certain cases or whether it becomes unnecessary.
        """
        # If bias is row major, swap all M and N dimensions
        if (
            bias is not None
            and len(bias.get_stride()) >= 2
            and bias.get_stride()[-1] in (0, 1)
        ):
            log.debug("GEMM Layout swapped X and W -> explicit transpose")
            return True
        return False

    @staticmethod
    def swap_XW(
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> "cutlass_library.gemm_op.GemmOperation":  # type: ignore[name-defined]  # noqa: F821
        """
        Swap operands X and W (aka operands A and B) of the GEMM operation. This
        requires transposing the operands, which is done by swapping the strides.
        Note that we don't change the apparent external layout, just the operand layout.
        this is intentional.
        """
        new_op = copy.deepcopy(op)
        new_op.A.layout = CUTLASSGemmTemplate.flip_cutlass_layout(new_op.A.layout)
        new_op.B.layout = CUTLASSGemmTemplate.flip_cutlass_layout(new_op.B.layout)
        new_op.A, new_op.B = new_op.B, new_op.A
        new_op.C.layout = CUTLASSGemmTemplate.flip_cutlass_layout(new_op.C.layout)
        new_op.D.layout = CUTLASSGemmTemplate.flip_cutlass_layout(new_op.D.layout)
        return new_op

    def fix_op_layout(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined] # noqa: F821
        X: Buffer,
        W: Buffer,
        Bias: Buffer | None,
        Y: Buffer | ReinterpretView,
    ) -> "cutlass_library.gemm_op.GemmOperation":  # type: ignore[name-defined]  # noqa: F821
        # This is a workaround to deal with cases where the input layouts have changed
        # between autotuning and rendering. This happens if the inputs layout
        # are FlexibleLayout instances. In this case, we need to update the
        # op's input layouts. It is a hack, because now the op
        # we benchmarked is not the same as the op we render,
        # but there is no simple way to fix this in the autotuner, since that would
        # potentially disable other optimizations.
        a_layout = X.get_layout()
        b_layout = W.get_layout()
        c_layout = Bias.get_layout() if Bias is not None else None

        d_layout = copy.deepcopy(Y.get_layout())
        match_list = [
            CUTLASSGemmTemplate.layout_match(buf.get_layout(), op_layout)
            for buf, op_layout in zip(
                (X, W, Bias, Y),
                (op.A.layout, op.B.layout, op.C.layout, op.D.layout),
            )
            if buf is not None
        ]
        all_match = all(match_list)
        if all_match:
            return op
        log.warning(
            f"Cutlass GEMM Layout change: Input and/or output layouts have changed between autotuning/retuning and call to render on {self}. Applying workaround. This can lead to suboptimal performance. Match List: {match_list}"  # noqa: G004
        )
        new_op = copy.deepcopy(op)

        if a_layout is not None:
            new_op.A.layout = CUTLASSGemmTemplate.cutlass_layout(a_layout)
        if b_layout is not None:
            new_op.B.layout = CUTLASSGemmTemplate.cutlass_layout(b_layout)
        if c_layout is not None:
            new_op.C.layout = CUTLASSGemmTemplate.cutlass_layout(c_layout)
            new_op.C.element = cutlass_utils.torch_dtype_to_cutlass_type(c_layout.dtype)
        if d_layout is not None:
            new_op.D.layout = CUTLASSGemmTemplate.cutlass_layout(d_layout)
        return new_op

    def _dtype_match(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> bool:
        """
        Checking dtypes of A, B, acc, D here.

        Empirically speaking, CUTLASS2x ops have same dtype for C and D.
        """
        X = self.input_nodes[0]
        W = self.input_nodes[1]

        accumulator_torch_dtype = cutlass_utils.get_accumulator_dtype(
            [X.get_dtype(), W.get_dtype()],
        )
        if not (
            cutlass_utils.dtype_match(X.get_dtype(), op.A.element)
            and cutlass_utils.dtype_match(W.get_dtype(), op.B.element)
            and cutlass_utils.dtype_match(
                self.output_node.get_layout().dtype, op.D.element
            )
            and cutlass_utils.dtype_match(
                accumulator_torch_dtype, op.accumulator_type()
            )
        ):
            return False

        return True

    @classmethod
    def global_filter_ops(
        cls,
        ops: list["cutlass_library.gemm_op.GemmOperation"],  # type: ignore[name-defined]  # noqa: F821
    ) -> list["cutlass_library.gemm_op.GemmOperation"]:  # type: ignore[name-defined]  # noqa: F821
        """
        Filter ops without using information about the torch op, input nodes and output node.
        """
        if not cutlass_utils.try_import_cutlass():
            raise AssertionError("could not import cutlass")
        import cutlass_library.library as cutlass_lib  # type: ignore[import]

        # Skip simt kernels
        ops = [
            op
            for op in ops
            if op.tile_description.math_instruction.opcode_class
            != cutlass_lib.OpcodeClass.Simt
        ]

        # only keep the set of row x column ops
        # for other layout, we modify in place in filter_op, after deepcopy
        ops = [
            op
            for op in ops
            if op.A.layout.name == "RowMajor" and op.B.layout.name == "ColumnMajor"
        ]

        # filter by supported accumulator types
        ops = [
            op
            for op in ops
            if any(
                dtype_match(torch_dtype, op.accumulator_type())
                for torch_dtype in ACCUMULATOR_DTYPES
            )
        ]

        # check if dtypes of A and B are supported
        ops = [
            op
            for op in ops
            if any(dtype_match(torch_dtype, op.A.element) for torch_dtype in XW_DTYPES)
            and any(dtype_match(torch_dtype, op.B.element) for torch_dtype in XW_DTYPES)
        ]

        return ops

    def filter_op(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> "cutlass_library.gemm_op.GemmOperation":  # type: ignore[name-defined]  # noqa: F821
        """
        Helper method:

        Determines whether a given Cutlass GEMM op definition is suitable for the current
        input / output of the operation that this template is supposed to implement.

        Takes memory layout, dtype and support for EVT operations into account,
        and filters potentially problematic ops.

        Returns None if the op is not suitable, otherwise returns the op to be used, which might
        have been mutated.
        """

        if op.gemm_kind not in self._get_supported_ops():
            return None

        X = self.input_nodes[0]
        W = self.input_nodes[1]

        # Filter ops according to the shape match.
        if not self._shape_match(op):
            return None

        # Filter ops by dtypes.
        if not self._dtype_match(op):
            return None

        # Filter ops by alignment.
        if not self._alignment_match(op):
            log.debug(
                "Skipping due to alignment mismatch. op: %s", op.configuration_name()
            )
            return None

        # only use stream k for static shape
        if op.tile_scheduler.name == "StreamK":
            static_shape = PythonWrapperCodegen.statically_known_list_of_ints_or_none(
                tuple(X.get_size()) + tuple(W.get_size())
            )
            if not static_shape:
                return None

        # Update op.
        op = copy.deepcopy(op)

        # set layouts for X and W
        self.set_layout(op.A, X.get_layout())
        self.set_layout(op.B, W.get_layout())

        # Set output layout.
        op.D.layout = CUTLASSGemmTemplate.cutlass_layout(self.output_node.get_layout())

        # Filter ops by alignments and set alignments.
        status = (
            self.set_alignment(X.get_layout(), op.A)
            and self.set_alignment(W.get_layout(), op.B)
            and self.set_alignment(self.output_node.get_layout(), op.D)
        )
        if not status:
            log.debug(
                "Skipping due to alignment setting failure. op: %s",
                op.configuration_name(),
            )
            return None

        if inductor_cutlass_config.cutlass_tma_only and not self._has_tma_epilogue(op):
            return None

        # Set epilogue.
        # TODO: update epilogue functor according to epilogues.
        op.element_epilogue = op.accumulator_type()

        if (
            self.use_fast_accum is not None
            and int(cutlass_utils._normalize_cuda_arch(cuda_env.get_cuda_arch())) == 90
        ):
            is_op_fast_accum = "fastaccum" in op.configuration_name()
            if self.use_fast_accum ^ is_op_fast_accum:
                return None

        # Set bias layout and alignment.
        status = self._set_bias_layout_and_alignment(op)
        if not status:
            log.debug(
                "Skipping due to bias layout and alignment setting failure. op: %s",
                op.configuration_name(),
            )
            return None

        # Apply regex filters at the end when configuration name doesn't change anymore
        if inductor_cutlass_config.cutlass_op_allowlist_regex:
            if not re.search(
                inductor_cutlass_config.cutlass_op_allowlist_regex,
                op.configuration_name(),
            ):
                return None
        if inductor_cutlass_config.cutlass_op_denylist_regex is not None:
            if re.search(
                inductor_cutlass_config.cutlass_op_denylist_regex,
                op.configuration_name(),
            ):
                return None

        # `_procedural_name` is decorated with @functools.cached_property in cutlass, and its value is
        # cached based on the key `self`. After we modify some attributes of
        # `self` (e.g., layout or alignment), the `self` itself doesn’t change, so the
        # cached value remains stale. We therefore need to clear the cached value so that
        # `_procedural_name` can be recomputed with the updated attributes.
        del op._procedural_name
        return op

    def gen_ops(self) -> "list[tuple[str, cutlass_gemm_op.GemmOperation]]":  # type: ignore[name-defined]  # noqa: F821
        """
        Creates a list of Cutlass GemmOperation instances that match the operation this template is designed to represent.
        The matching is carried out with respect to the input and output specifications of the operation.

        No function arguments.

        Returns:
            List[tuple[str, cutlass_gemm_op.GemmOperation]]: A list of (cutlass_name, GemmOperation)
            tuples that are compatible with the operation requirements of this template.
        """
        if not cutlass_utils.try_import_cutlass():
            raise AssertionError("could not import cutlass")
        import cutlass_library.gemm_operation as cutlass_gemm_op

        if self.cache_key in self.filtered_ops_cache:
            log.debug("Using cached ops for %s", self.cache_key)
            return self.filtered_ops_cache[self.cache_key]

        counters["inductor"]["cutlass_filtered_ops_cache_miss"] += 1

        with dynamo_timed("CUTLASSGemmTemplate.maybe_fetch_ops"):
            maybe_ops = maybe_fetch_ops(self.device_type)
        if maybe_ops is None:
            log.debug("Cannot fetch ops from cache, generating ops from scratch")
            full_ops = cutlass_utils.gen_ops(self.device_type)
            ops = pytree.tree_flatten(full_ops)[0]
        else:
            log.debug("Using cached ops from cache")
            ops = maybe_ops

        ops = self.global_filter_ops(ops)

        res: dict[str, cutlass_gemm_op.GemmOperation] = {}
        start_time = time.time()
        for op in ops:
            # if changed, need to also change CUTLASS_OPERATION_KIND
            if not isinstance(op, cutlass_gemm_op.GemmOperation):
                raise AssertionError(f"expected GemmOperation, got {type(op).__name__}")
            filter_res = self.filter_op(op)
            if (
                filter_res is not None
                and res.get(filter_res.configuration_name()) is None
            ):
                res[filter_res.configuration_name()] = filter_res
        log.info(
            "Got cutlass configs: total number of ops: %d. Filtering took %.2f seconds",
            len(res),
            time.time() - start_time,
        )
        sorted_res = sorted(res.items())
        device_config = self._device_cutlass_config
        ret_res = sorted_res[: device_config.cutlass_max_profiling_configs]
        if len(self.filtered_ops_cache) < 50:
            self.filtered_ops_cache[self.cache_key] = ret_res
        else:
            log.debug("Not caching ops since filtered_ops_cache has reached size 50.")
        return ret_res

    def gemm_mode(self) -> str:
        """
        Returns a Cutlass GEMM mode string for the current operation, dependent on whether this op implements
        a batched GEMM or a simple GEMM without batch dimension.

        Returns:
        str: A string indicating the Cutlass GEMM mode. If the output node has more than two dimensions,
            "cutlass::gemm::GemmUniversalMode::kBatched" is returned, otherwise
            "cutlass::gemm::GemmUniversalMode::kGemm" is returned.
        """
        sizes = self.output_node.get_size()
        if len(sizes) > 2:
            return "cutlass::gemm::GemmUniversalMode::kBatched"
        else:
            return "cutlass::gemm::GemmUniversalMode::kGemm"

    def _dynamic_cluster_block(self, op: "cutlass_gemm_op.GemmOperation") -> str:  # type: ignore[name-defined]  # noqa: F821
        """
        Temporary workaround for CUTLASS GEMMs that encode cluster shape as runtime values.

        For dynamic-cluster kernels, CUTLASS expects `KernelHardwareInfo.cluster_shape` and
        `KernelHardwareInfo.cluster_shape_fallback` to be populated with valid runtime values.
        Today we provide a single global preferred/fallback pair (configurable via env),
        which is sufficient for correctness but is not performance-optimal.

        Note: This is intentionally minimal because this code path is transitional. Cluster-shape
        selection should ultimately be handled in the CuTe/DSL implementation rather than
        investing heavily in the legacy CUTLASS template pipeline.
        """
        shape = getattr(getattr(op, "tile_description", None), "cluster_shape", None)
        if not shape or len(shape) < 2 or (shape[0] > 0 and shape[1] > 0):
            return ""

        preferred = inductor_cutlass_config.cutlass_dynamic_cluster_shape
        fallback = inductor_cutlass_config.cutlass_dynamic_cluster_fallback

        cluster_k = shape[2] if len(shape) > 2 and shape[2] > 0 else preferred[2]
        preferred = (preferred[0], preferred[1], cluster_k)
        fallback_k = fallback[2] if len(fallback) > 2 and fallback[2] > 0 else cluster_k
        fallback = (fallback[0], fallback[1], fallback_k)

        return (
            f"  hw_info.cluster_shape = {{{preferred[0]}, {preferred[1]}, {preferred[2]}}};\n"
            f"  hw_info.cluster_shape_fallback = {{{fallback[0]}, {fallback[1]}, {fallback[2]}}};"
        )

    def render(  # type: ignore[override]
        self,
        kernel: CUTLASSTemplateKernel,
        op: "cutlass_gemm_op.GemmOperation" = None,  # type: ignore[name-defined]  # noqa: F821
        template_buffer_node: CUTLASSTemplateBuffer | None = None,
        epilogue_nodes: list[BaseSchedulerNode] | None = None,
        **kwargs,
    ) -> str:
        """
        The primary entry point for the code rendering process used in this template.
        Renders the Cutlass based CUDA/XPU C++ code for the GEMM Kernel that this template is designed to implement,
        including potentially fused epilogues.

        Args:
            kernel (CUTLASSTemplateKernel): The kernel to be rendered.
            op (cutlass_gemm_op.GemmOperation, optional): A GEMM operation that is required to be compatible with the
                input and output definitions as well as a possible epilogue. Defaults to None.
            **kwargs: Additional keyword arguments. Currently unused.

        Returns:
            str: Cutlass based CUDA/XPU C++ code fragment as a string, to be used by the current
            CUTLASSTemplateKernel or autotuning code.

        Note:
            All inputs and their corresponding buffer addresses and names take precedence over previously
            passed inputs to the template at construction time. However, they should be layout compatible.
        """
        if not cutlass_utils.try_import_cutlass():
            raise AssertionError("could not import cutlass")
        import cutlass_library.gemm_operation as cutlass_gemm_op
        import cutlass_library.library as cutlass_lib

        if not isinstance(op, cutlass_gemm_op.GemmOperation):
            raise AssertionError(
                "op argument is required and has to be an instance of GemmOperation"
            )

        if epilogue_nodes:
            if self.device_type == "cuda" and not self._has_tma_epilogue(op):
                raise NotImplementedError(
                    "Non-TMA epilogue visitor tree is not supported in NV-Cutlass."
                )

        if not (len(self.input_nodes) >= 2 and self.output_node is not None):
            raise AssertionError(
                "expected at least 2 input nodes and a non-None output node"
            )
        X, W = self.input_nodes[0], self.input_nodes[1]
        for input_node in self.input_nodes:
            if not isinstance(X.layout, FixedLayout):
                input_node.freeze_layout()

        Y = self.output_node
        if template_buffer_node is not None:
            Y = template_buffer_node

        Bias, extra_inputs, extra_names = self._get_extra_inputs_and_names(op)

        # Define Kernel call signature
        # Important: This step also populates Kernel name to node mapping data structures,
        # which are required further below ( for example by the template renderer )
        inputs = [X, W, Bias, *extra_inputs]
        names = ["X", "W", "Bias", *extra_names] + ["Y"]
        names_str = ",".join(names)
        if self.input_reorder is not None:
            input_reorder = self.input_reorder
        else:
            input_reorder = None

        # The layouts might have changed between autotuning and this call if they were FlexibleLayout
        # we need to adapt, which might lead to suboptimal performance.
        op = self.fix_op_layout(op, X, W, Bias, Y)

        # to make op mutable without affecting others
        op = copy.deepcopy(op)
        is_scaled_mm = len(self.input_nodes) in (4, 5)
        if Bias is not None and not is_scaled_mm:
            if Bias.get_dtype() != X.get_dtype():
                raise AssertionError(
                    f"expected Bias dtype {X.get_dtype()}, got {Bias.get_dtype()}"
                )
            # This might have been set to void during filtering, when the assumption was still that there's no C
            # operand
            op.C.element = op.A.element

            if op.C.element != op.D.element:
                raise AssertionError(
                    f"Expect C and D to have the same dtype, found {op.C.element} and {op.D.element}"
                )

        argument_template, epilogue_template = self._get_template_args(op)
        should_swap_xw: bool = False
        if Bias is not None and self._has_tma_epilogue(op):
            if (
                op.epilogue_schedule
                != cutlass_lib.EpilogueScheduleType.EpilogueTransposed
                and self.should_swap_XW(Bias)
            ):
                # TMA epilogue requires bias vector in column major to get best perf.
                op = self.swap_XW(op)
                should_swap_xw = True

        name_to_buffer = {node.get_name(): node for node in self.input_nodes}
        # handle the fake output buffer during lowering
        name_to_buffer[Y.get_name()] = Y  # type: ignore[assignment]

        if epilogue_nodes or is_scaled_mm:
            if epilogue_nodes:
                (
                    input_names,
                    output_names,
                    var_name_to_buffer_name,
                    evt_py_code,
                ) = CutlassEVTCodegen.ir_to_evt_python_code(
                    Y.get_name(), epilogue_nodes, V.kernel.removed_buffers
                )

                # TODO: mlazos remove this by returning buffer metadata from
                # ir_to_evt_python code
                for name, buf in (
                    V.graph.name_to_buffer | V.graph.graph_inputs
                ).items():
                    if name not in name_to_buffer:
                        name_to_buffer[name] = buf  # type: ignore[assignment]

                D_output_name = var_name_to_buffer_name["D"]
                D_output_buffer = name_to_buffer[D_output_name]
                Y = D_output_buffer  # type: ignore[assignment]
                # Interestingly, I don't think the rest of the layout matters here since we
                # use the properties of the Y buffer to fill in D's properties in the epilogue
                # args. This is needed though because it defines types expected in the epilogue args.
                op.D.element = cutlass_utils.torch_dtype_to_cutlass_type(
                    D_output_buffer.get_dtype()
                )

                # When the D output buffer has a compatible reshape of the template
                # output (e.g., [16, 512, 3072] vs [8192, 3072]), use the template's
                # layout for EVT since CUTLASS operates on the 2D GEMM output shape.
                if (
                    template_buffer_node is not None
                    and D_output_buffer.get_size() != template_buffer_node.get_size()
                ):
                    name_to_buffer[D_output_name] = template_buffer_node

                # Apply the same normalization to external read inputs. EVT
                # propagates shapes against the 2D (M, N) accumulator, so a
                # higher-rank read with the same number of elements (e.g. a bias
                # whose shape is a compatible reshape of the GEMM output) must be
                # viewed as the 2D template shape. This is only valid for
                # contiguous reads, where the row-major flatten is
                # memory-equivalent and thus numerically identical.
                if template_buffer_node is not None:
                    template_size = list(template_buffer_node.get_size())
                    for name in input_names:
                        buf = name_to_buffer[name]
                        buf_layout = buf.get_layout()
                        if (
                            list(buf.get_size()) != template_size
                            and buf_layout.is_contiguous()
                            and V.graph.sizevars.statically_known_equals(
                                sympy_product(buf.get_size()),
                                sympy_product(template_size),
                            )
                        ):
                            new_layout = FixedLayout(
                                buf_layout.device,
                                buf_layout.dtype,
                                template_size,
                                FlexibleLayout.contiguous_strides(template_size),
                            )
                            name_to_buffer[name] = ReinterpretView(  # type: ignore[assignment] # pyrefly: ignore[unsupported-operation]
                                data=buf, layout=new_layout
                            )

                if not output_names:
                    raise AssertionError("There should be at least one write")

                epilogue_inputs = [name_to_buffer[name] for name in input_names]
                # Use D_output_buffer directly for the D output entry in outputs,
                # not name_to_buffer[D_output_name] which may have been replaced
                # with template_buffer_node for EVT layout purposes.
                outputs = [
                    D_output_buffer if name == D_output_name else name_to_buffer[name]
                    for name in output_names
                ]
            else:  # Scaled MM, we read the two scale matrices (and optional bias) and write a single output
                bias = None if len(self.input_nodes) < 5 else self.input_nodes[4]
                bias_name = bias.get_name() if bias else None

                (
                    evt_read_names,
                    var_name_to_buffer_name,
                    evt_py_code,
                ) = scaled_mm_evt(
                    self.input_nodes[2].get_name(),  # scale_A
                    self.input_nodes[3].get_name(),  # scale_B
                    bias_name,
                    Y.get_name(),
                )

                input_names = list(evt_read_names)
                output_names = []  # We only need Y
                epilogue_inputs = [self.input_nodes[2], self.input_nodes[3]]
                if bias:
                    epilogue_inputs.append(bias)
                outputs = []

            acc_dtype = cutlass_utils.get_accumulator_dtype(
                [X.get_dtype(), W.get_dtype()]
            )
            if not acc_dtype:
                raise AssertionError("Could not determine accumulator dtype")

            evt_name, evt_args, evt_code, evt_arg_renames = self._render_evt(
                op,
                evt_py_code,
                var_name_to_buffer_name,
                name_to_buffer,
                Y.get_dtype(),
                acc_dtype,
            )

            inputs = [
                X,
                W,
                Bias,
                *epilogue_inputs,  # type: ignore[list-item]
                Y,
                *extra_inputs,
            ]
            # Extend input_reorder to cover epilogue inputs, Y, and extra_inputs
            # at their natural positions (identity mapping for new entries)
            if input_reorder is not None:
                base_len = len(input_reorder)
                if not (base_len <= len(inputs)):
                    raise AssertionError(
                        "input_reorder must be a prefix mapping for the current input list"
                    )
                num_new = len(inputs) - base_len
                if num_new > 0:
                    input_reorder = input_reorder + list(
                        range(base_len, base_len + num_new)
                    )
            input_names = [evt_arg_renames.get(name) for name in input_names]
            output_names = [evt_arg_renames.get(name) for name in output_names]

            names_str = ",".join(
                ["X", "W", "Bias", *input_names, "Y", *output_names, *extra_names]
            )
        else:
            evt_name = None
            outputs = [Y]
            evt_args = f"{{ElementComputeEpilogue({self.alpha}), ElementComputeEpilogue({self.beta})}}"
            evt_code = ""

        kernel_call_signature = kernel.def_kernel(
            inputs=inputs,  # type: ignore[arg-type]
            outputs=outputs,  # type: ignore[arg-type]
            names_str=names_str,
            input_reorder=input_reorder,
        )

        test_call_statement = self.test_call_statement(kernel, inputs, names_str)

        instance_definition, instance_type = self._define_gemm_instance(op, evt_name)
        dynamic_cluster = self._dynamic_cluster_block(op)

        options = {
            "alpha": self.alpha,
            "beta": self.beta,
            "X": X,
            "W": W,
            "Y": Y,
            "kernel_call_signature": kernel_call_signature,
            "Bias": Bias,
            "epilogue_template": epilogue_template,
            "argument_template": argument_template,
            "should_swap_xw": should_swap_xw,
            "template": self,
            "kernel": kernel,
            "instance_definition": instance_definition,
            "instance_type": instance_type,
            "input_reorder": self.input_reorder,
            "epilogue_args": evt_args,
            "test_call_statement": test_call_statement,
            "op_conf_name": op.configuration_name(),
            "epilogue_visitor_tree": evt_code,
            "dynamic_cluster": dynamic_cluster,
        }
        options.update(dict(zip(extra_names, extra_inputs)))
        res = self._template_from_string(self._get_template()).render(**options)
        if inductor_cutlass_config.generate_test_runner and not is_dynamic(
            X, W, Y, Bias
        ):
            test_runner_code = self._template_from_string(
                GEMM_STANDALONE_RUNNER_TEMPLATE
            ).render(**options)
            res += "\n\n" + test_runner_code

        # splice to remove trailing spaces in each line
        buf = IndentedBuffer()
        buf.splice(res)
        return buf.getvalue()

    def test_call_statement(
        self,
        kernel,
        input_nodes,
        names_str: str = "",
    ) -> str:
        """
        Helper method to render the Cutlass CUDA/XPU C++ code required for calling the GEMM operation in the standalone
        test runner that might also be generated along with the rest of the code, if the corresponding config is
        enabled.

        Returns a C++ statement that calls the GEMM operation with the correct arguments.
        """
        _, __, arg_types = kernel.args.cpp_argdefs(cutlass_utils.DTYPE_TO_CUTLASS_TYPE)
        arg_names = [name.strip() for name in names_str.strip().split(",")]
        arg_names = self._update_arg_names_for_test_call_statement(
            arg_names, input_nodes
        )
        arguments = [
            f"(({arg_type}){arg_name}_data.get())"
            for arg_type, arg_name in zip(arg_types, arg_names)
        ]
        return f"{kernel.kernel_name}({', '.join(arguments)}, M, N, K, B, lda, ldb, ldc, ldd, 0, 0, 0, swizzle, workspace_size_ptr, (uint8_t*)workspace_data.get(), 0);"

    def _render_evt(
        self,
        op: GemmOperation,
        evt_py_code: str,
        buffer_renames: dict[str, str],
        name_to_buffer: dict[str, Buffer],
        output_dtype: torch.dtype,
        accumulator_dtype: torch.dtype,
    ) -> tuple[str, str, str, EVTArgRenames]:  # type: ignore[name-defined]
        raise NotImplementedError("_render_evt in CUTLASSGemmTemplate not implemented")


class CUTLASS3xGemmTemplate(CUTLASSGemmTemplate):
    """
    CUTLASS 3x GEMM Template, which is used to generate CUTLASS GEMM kernels
    including those which allow flexible fusions with epilogues.
    """

    @staticmethod
    def add_cutlass_gemm_choices(
        choices: list[ChoiceCaller],
        layout: ir.Layout,
        input_nodes: list[Buffer],
        alpha: float | int = 1,
        beta: float | int = 0,
        input_reorder: list[int] | None = None,
        use_fast_accum: bool | None = None,
        **extra_kwargs,
    ) -> None:
        template = CUTLASS3xGemmTemplate(
            input_nodes,
            layout,
            alpha,
            beta,
            input_reorder,
            use_fast_accum,
        )
        template._add_cutlass_gemm_choices(
            choices, layout, input_nodes, alpha, beta, input_reorder, **extra_kwargs
        )

    @staticmethod
    @functools.lru_cache(1)
    def _get_supported_ops() -> "list[cutlass_library.gemm_operation.GemmOperation]":  # type: ignore[name-defined]  # noqa: F821
        import cutlass_library.library as cutlass_lib

        return [cutlass_lib.GemmKind.Universal3x]

    def _get_template(self) -> str:
        return GEMM_TEMPLATE_CUTLASS_3X

    def _get_template_args(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> tuple[str, str | None]:
        return (GEMM_ARGS_CUTLASS_3X, GEMM_ARGS_CUTLASS_3X_EPILOGUE)

    @staticmethod
    def _has_tma_epilogue(  # type: ignore[arg-type,name-defined]
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined,arg-type] # noqa: F821
    ) -> bool:  # type: ignore[name-defined]
        """Helper method: Determine whether a given Cutlass GEMM op has a TMA Epilogue"""
        if not cutlass_utils.try_import_cutlass():
            raise AssertionError("could not import cutlass")
        import cutlass_library.library as cutlass_lib

        result = False
        if op.gemm_kind == cutlass_lib.GemmKind.Universal3x:
            epilogue_schedule_str = str(op.epilogue_schedule).split(".")[-1]
            result = epilogue_schedule_str.lower().startswith("tma")
        return result

    @staticmethod
    def supports_epilogue_fusion(op: GemmOperation, device_type: str) -> bool:
        if device_type == "xpu":
            return True
        return CUTLASS3xGemmTemplate._has_tma_epilogue(op)

    def _are_inputs_layout_compatible(self, layouts: list[Layout]) -> bool:
        """
        Evaluates whether input layouts are compatible for General Matrix Multiply (GEMM).

        This function checks compatibility of A, B, and possibly C operand layouts for
        a General Matrix Multiply (GEMM) operation, expressed as 'alpha * matmul(A, B) + beta * C'.
        It verifies requirements such as matching data types, minimum rank, and suitability
        for broadcasting, as defined by PyTorch operations like `torch.matmul`, `torch.aten.mm`,
        `addmm`, `bmm`, `baddbmm`, etc.

        Args:
            layouts (List[Layout]): List containing 2 or 3 Layout objects representing
                                    the input matrices A, B, and possibly C.

        Returns:
            bool: True if layouts are GEMM compatible, otherwise False.
        """
        if not (2 <= len(layouts) <= 5):
            raise AssertionError(f"expected 2 to 5 layouts, got {len(layouts)}")
        # Check if A and B are compatible
        A_layout, B_layout = layouts[:2]
        if len(A_layout.size) < 1:
            return False
        if len(B_layout.size) < 1:
            return False
        A_size = list(V.graph.sizevars.guarding_hints_or_throw(A_layout.size))
        B_size = list(V.graph.sizevars.guarding_hints_or_throw(B_layout.size))
        if len(A_size) < 2:
            A_size.insert(0, 1)
        if len(B_size) < 2:
            A_size.insert(1, 1)
        # Are batch dims broadcastable?
        while len(A_size) < len(B_size):
            A_size.insert(0, 1)
        while len(B_size) < len(A_size):
            B_size.insert(0, 1)
        K = max(A_size[-1], B_size[-2])
        M = A_size[-2]
        N = B_size[-1]
        if K != A_size[-1] and A_size[-1] != 1:
            return False
        if K != B_size[-2] and B_size[-1] != 1:
            return False
        # check batch dim broadcastable
        for i in range(len(A_size) - 2):
            if A_size[i] != B_size[i] and A_size[i] != 1 and B_size[i] != 1:
                return False
        if len(layouts) == 3:
            C_layout = layouts[2]
            C_size = list(V.graph.sizevars.guarding_hints_or_throw(C_layout.size))
            while len(C_size) < len(A_size):
                C_size.insert(0, 1)
            # check batch dims
            for i in range(len(A_size) - 2):
                bd = max(A_size[i], B_size[i])
                if bd != C_size[i] and C_size[i] != 1:
                    return False
            if len(C_size) > len(A_size):
                # This may happen if the last elements of C are contiguous and
                # their multiplied size equals the last dim size of B
                if M != C_size[len(A_size) - 2] and C_size[len(A_size) - 2] != 1:
                    return False
                remaining_size = 1
                for i in range(len(A_size) - 1, len(C_size)):
                    remaining_size *= C_size[i]
                if N != remaining_size and remaining_size != 1:
                    return False
                return True
            if len(C_size) != len(A_size):
                raise AssertionError(
                    f"expected C and A to have same rank, got {len(C_size)} and {len(A_size)}"
                )
            if M != C_size[-2] and C_size[-2] != 1:
                return False
            if N != C_size[-1] and C_size[-1] != 1:
                return False
        return True

    def _render_evt(
        self,
        op: GemmOperation,
        evt_py_code: str,
        var_name_to_buffer_name: dict[str, str],
        name_to_buffer: dict[str, Buffer],
        output_dtype: torch.dtype,
        accumulator_dtype: torch.dtype,
    ) -> tuple[str, str, str, EVTArgRenames]:
        from .lib_extensions.evt_extensions import create_example_tensors, trace

        acc_dtype = torch_dtype_to_cutlass_type(accumulator_dtype)
        output_dtype = torch_dtype_to_cutlass_type(output_dtype)

        # TODO: size_hint_fn is passed to both create_example_tensors (just for
        # tracing examples) and trace -> _render_argument_type -> _get_arg_from_node
        # where stride values are baked as C++ literals into the generated CUTLASS
        # argument struct. For the latter, accessing hint is wrong: even for
        # backed symbols we access the hint without installing a guard, so the
        # generated strides could be incorrect for different dynamic shape inputs.
        # guarding_hint_or_throw would be wrong here — it does not install
        # guards either. This should use guard_int to properly guard on the values.
        examples = create_example_tensors(
            var_name_to_buffer_name,
            name_to_buffer,  # type: ignore[arg-type]
            V.graph.sizevars.optimization_hint,
        )
        evt_name, evt_args, evt_code, arg_renames = trace(
            evt_py_code,
            examples,
            acc_dtype,
            output_dtype,
            op.tile_description,  # type: ignore[attr-defined]
            op.epilogue_schedule,  # type: ignore[attr-defined]
            {k: name_to_buffer[v] for k, v in var_name_to_buffer_name.items()},  # type: ignore[arg-type,misc]
            V.graph.sizevars.guarding_hint_or_throw,
            kernel_schedule=op.kernel_schedule,
            device_type=self.device_type,
        )

        return (
            evt_name,
            evt_args,
            evt_code,
            arg_renames,
        )

    def _shape_match(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> bool:
        return True

    def _alignment_match(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> bool:
        return True

    def _set_bias_layout_and_alignment(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> bool:
        import cutlass_library.library as cutlass_lib

        has_bias = len(self.input_nodes) == 3 and self.input_nodes[2] is not None
        if has_bias:
            Bias = self.input_nodes[2]
            # bias dtype
            op.C.element = cutlass_utils.torch_dtype_to_cutlass_type(
                Bias.get_layout().dtype
            )

            # Bias layout
            bias_layout = CUTLASSGemmTemplate.cutlass_layout(Bias.get_layout())
            op.C.layout = bias_layout

            # Bias alignment
            status = self.set_alignment(Bias.get_layout(), op.C)
            if not status:
                return False
        else:
            op.C.element = cutlass_lib.DataType.void
        return True

    def _define_gemm_instance(
        self,
        op: GemmOperation,
        evt_name: str | None = None,
    ) -> tuple[str, str]:
        """Defines and renders the Cutlass / CUDA/XPU C++ code for a given GEMM operation instance.

        This function uses the Cutlass library to generate key parts of the codegen process. General Matrix Multiply
        forms a core part of a number of scientific applications, so this efficient and adaptable implementation is
        crucial.

        Args:
            op (cutlass_library.gemm_op.GemmOperation): This is the core GEMM operation that we are defining and rendering.

        Returns:
            tuple[str, str]: A tuple where the first part is a string that constitutes the defined GEMM operation in C++
                             code (render) and the second part is the string that specifies the operation type.
        """
        if not cutlass_utils.try_import_cutlass():
            raise AssertionError("could not import cutlass")
        import cutlass_library.library as cutlass_lib

        from .lib_extensions import gemm_operation_extensions as gemm_extensions

        emitter = gemm_extensions.EmitGemmUniversal3xInstanceWithEVT(
            evt_name=evt_name, device_type=self.device_type
        )  # type: ignore[call-arg]

        if not hasattr(op, "epilogue_functor") or not isinstance(
            op.epilogue_functor, enum.Enum
        ):
            op = copy.deepcopy(op)
            op.epilogue_functor = cutlass_lib.EpilogueFunctor.LinearCombination

        op_def = emitter.emit(op)
        pattern = re.compile(r"\s*struct\s(.*?)\s:")
        decl = [line for line in op_def.split("\n") if "struct " in line][-1]

        match = pattern.match(decl)
        if match is None:
            raise RuntimeError("Invalid Gemm config: \n" + op_def)
        op_type = match.groups()[0]
        if op.gemm_kind == cutlass_lib.GemmKind.Universal3x:
            # Append KERNEL_NAME to the struct name to ensure uniqueness across
            # compiled .so files. When multiple kernels share the same GEMM core
            # but have different epilogues (e.g., LinearCombination vs EVT), they
            # produce identical struct names. On SYCL/XPU this causes the runtime
            # to dispatch the wrong binary; on CUDA it avoids potential symbol
            # collisions. KERNEL_NAME gets replaced with a unique per-.so hash
            # in scheduling.py.
            unique_op_type = f"{op_type}_{Placeholder.KERNEL_NAME}"
            op_def = op_def.replace(f"struct {op_type} :", f"struct {unique_op_type} :")
            op_def += f"\n  using {unique_op_type}_device_type = cutlass::gemm::device::GemmUniversalAdapter<{unique_op_type}>;\n"
            op_type = f"{unique_op_type}_device_type"

        return op_def, op_type

    def _get_extra_inputs_and_names(
        self,
        op: "cutlass_gemm_op.GemmOperation" = None,  # type: ignore[name-defined]  # noqa: F821
    ) -> tuple[Buffer | None, list[Buffer | None], list[str]]:
        Bias = self.input_nodes[2] if len(self.input_nodes) == 3 else None
        inputs: list[Buffer | None] = []
        names: list[str] = []
        return (Bias, inputs, names)

    def _update_arg_names_for_test_call_statement(
        self,
        arg_names: list[str],
        input_nodes: list[Buffer],
    ) -> list[str]:
        if input_nodes[2] is None:
            del arg_names[2]
        else:
            # Reorder them as Bias, A, B
            if self.input_reorder is not None:
                arg_names[0 : len(self.input_reorder)] = [
                    arg_names[i] for i in self.input_reorder
                ]
        return arg_names

    def render_gemm_arguments(
        self,
        argument_template: str,
        epilogue_template: str,
        should_swap_xw: bool,
        X: IRNode,
        W: IRNode,
        Bias: IRNode,
        Y: IRNode,
        alpha: float,
        beta: float,
        kernel: CUTLASSTemplateKernel,
        epilogue_args,
    ) -> str:
        """
        Render the Cutlass CUDA/XPU C++ code required for passing arguments to the GEMM operation.

        Args:
            argument_template (str): Template for the GEMM operation arguments.
            epilogue_template (str): Template for the epilogue arguments.
            should_swap_xw (bool): Determines whether X, W operands should be swapped. If True, applies an explicit
            transpose operation to X and W.
            X (IRNode): The X input tensor.
            W (IRNode): The W input tensor.
            Bias (IRNode): The bias tensor.
            Y (IRNode): The output tensor.
            alpha (float): Scaling factor for the product of the inputs.
            beta (float): Scaling factor for the output tensor.
            kernel (CUTLASSTemplateKernel): CUDA/XPU Template kernel for the operation.
            epilogue_args (any): Additional arguments for the epilogue state.

        Returns:
            str: A block of CUDA/XPU C++ code as a string, ready to be used as arguments for the GEMM operation.

        Note: If `should_swap_xw` is True, a transpose operation will be applied to the X, W, Bias, and Y
        tensors. This operation also implies the M and N dimensions of Bias and GEMM output to be swapped
        before the function call.
        """
        options = {
            "alpha": alpha,
            "beta": beta,
            "X": X,
            "W": W,
            "Y": Y,
            "Bias": Bias,
            "template": self,
            "kernel": kernel,
            "M": "M",
            "N": "N",
            "epilogue_args": epilogue_args,
        }
        if epilogue_template is None:
            raise AssertionError("expected epilogue_template to be non-None")

        if should_swap_xw:
            # Swap
            def clone_with_transposed_stride(node: IRNode) -> IRNode:
                old_layout = node.get_layout()
                new_stride = list(old_layout.stride)  # type: ignore[union-attr]
                new_stride[-2], new_stride[-1] = new_stride[-1], new_stride[-2]
                if old_layout.device is None:
                    raise AssertionError("expected old_layout.device to be non-None")
                new_layout = FixedLayout(
                    old_layout.device,
                    old_layout.dtype,
                    list(old_layout.size),  # type: ignore[union-attr]
                    new_stride,
                    old_layout.offset,  # type: ignore[union-attr]
                )
                return Buffer(name=node.get_name(), layout=new_layout)

            new_X = clone_with_transposed_stride(X)
            new_W = clone_with_transposed_stride(W)
            new_Bias = clone_with_transposed_stride(Bias)
            new_Y = clone_with_transposed_stride(Y)
            options["X"], options["W"], options["Bias"], options["Y"] = (
                new_W,
                new_X,
                new_Bias,
                new_Y,
            )
            options["M"], options["N"] = "N", "M"

        epilogue_arguments = self._template_from_string(epilogue_template).render(
            **options
        )
        arguments = self._template_from_string(argument_template).render(
            epilogue_arguments=epilogue_arguments, **options
        )

        return arguments


class CUTLASS2xGemmTemplate(CUTLASSGemmTemplate):
    """CUTLASS 2x GEMM Template, which is used to generate CUTLASS 2x GEMM kernels"""

    def __init__(
        self,
        input_nodes: list[Buffer],
        layout: Layout,
        alpha: float,
        beta: float,
        input_reorder: list[int] | None = None,
    ):
        super().__init__(input_nodes, layout, alpha, beta, input_reorder)

    @staticmethod
    def add_cutlass_gemm_choices(
        choices: list[ChoiceCaller],
        layout: ir.Layout,
        input_nodes: list[Buffer],
        alpha: float | int = 1,
        beta: float | int = 0,
        input_reorder: list[int] | None = None,
        use_fast_accum: bool | None = False,
        **extra_kwargs,
    ) -> None:
        template = CUTLASS2xGemmTemplate(
            input_nodes, layout, alpha, beta, input_reorder
        )
        template._add_cutlass_gemm_choices(
            choices, layout, input_nodes, alpha, beta, input_reorder, **extra_kwargs
        )

    @staticmethod
    def _get_supported_ops() -> "list[cutlass_library.gemm_operation.GemmOperation]":  # type: ignore[name-defined]  # noqa: F821
        import cutlass_library.library as cutlass_lib

        return [cutlass_lib.GemmKind.Universal, cutlass_lib.GemmKind.Sparse]

    @staticmethod
    def _has_tma_epilogue(self) -> bool:
        return False

    def _get_template(self) -> str:
        return GEMM_TEMPLATE_CUTLASS_2X

    def _get_template_args(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> tuple[str, str | None]:
        import cutlass_library.library as cutlass_lib

        if op.gemm_kind == cutlass_lib.GemmKind.Sparse:
            return (GEMM_ARGS_SPARSE_CUTLASS_2X, None)

        return (GEMM_ARGS_CUTLASS_2X, None)

    def _are_inputs_layout_compatible(self, layouts: list[Layout]) -> bool:
        """
        Evaluates whether input layouts are compatible for set of operations supported by this class.

        Args:
            layouts (List[Layout]): List containing Layout objects representing
                                    the input matrices.

        Returns:
            bool: True if layouts are GEMM compatible, otherwise False.
        """
        if not (len(layouts) == 2 or len(layouts) == 3):
            raise AssertionError(f"expected 2 or 3 layouts, got {len(layouts)}")
        # Check if A and B are compatible
        A_layout, B_layout = layouts[:2]
        if len(A_layout.size) != 2:
            return False
        if len(B_layout.size) != 2:
            return False
        A_size = [int(i) for i in A_layout.size]
        B_size = [int(i) for i in B_layout.size]
        K = max(A_size[1], B_size[0])
        return (K == A_size[1] or K == 2 * A_size[1]) and K == B_size[0]

    def _shape_match(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> bool:
        import cutlass_library.library as cutlass_lib

        X, W = self.input_nodes[0], self.input_nodes[1]

        if op.gemm_kind == cutlass_lib.GemmKind.Sparse:
            return X.get_size()[1] * 2 == W.get_size()[0]

        return X.get_size()[1] == W.get_size()[0]

    def _alignment_match(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> bool:
        import cutlass_library.library as cutlass_lib

        if op.gemm_kind != cutlass_lib.GemmKind.Sparse:
            return True

        # SparseGemm in CUTLASS has specific alignment check that for
        # small k could make some of the choices throw kMisalignedOperand
        # CUTLASS error when run, see:
        # https://github.com/NVIDIA/cutlass/blob/e01b9b5029b7caca5a43c29f7d2714d7cf1dcae8/include/cutlass/gemm/kernel/sparse_gemm.h#L198-L200
        # So, let's skip these choices if that would be the case.
        X = self.input_nodes[0]
        return (X.get_size()[1] * 2) % op.tile_description.tile_shape[2] == 0

    def _set_bias_layout_and_alignment(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> bool:
        import cutlass_library.library as cutlass_lib

        if op.gemm_kind == cutlass_lib.GemmKind.Sparse:
            op.C.layout = op.D.layout
            return True

        if len(self.input_nodes) >= 3 and self.input_nodes[2] is not None:
            Bias = self.input_nodes[2]
            bias_layout = CUTLASSGemmTemplate.cutlass_layout(Bias.get_layout())
            if bias_layout != op.D.layout:
                # For cutlass2, bias and output layout must match
                return False
            if not self.set_alignment(Bias.get_layout(), op.C):
                return False
        else:
            op.C.layout = op.D.layout
        return True

    def _define_gemm_instance(
        self,
        op: GemmOperation,
        evt_name: str | None = None,
    ) -> tuple[str, str]:
        """Defines and renders the Cutlass / CUDA C++ code for a given GEMM operation instance.

        This function uses the Cutlass library to generate key parts of the codegen process. General Matrix Multiply
        forms a core part of a number of scientific applications, so this efficient and adaptable implementation is
        crucial.

        Args:
            op (cutlass_library.gemm_op.GemmOperation): This is the core GEMM operation that we are defining and rendering.

        Returns:
            tuple[str, str]: A tuple where the first part is a string that constitutes the defined GEMM operation in C++
                             code (render) and the second part is the string that specifies the operation type.
        """
        if not cutlass_utils.try_import_cutlass():
            raise AssertionError("could not import cutlass")
        import cutlass_library.gemm_operation as cutlass_gemm_op
        import cutlass_library.library as cutlass_lib

        if op.gemm_kind == cutlass_lib.GemmKind.Sparse:
            emitter = cutlass_gemm_op.EmitSparseGemmInstance()
        else:
            emitter = cutlass_gemm_op.EmitGemmInstance()
        op_def = emitter.emit(op)
        op_def = op_def.replace(
            "cutlass::gemm::device::Gemm", "cutlass::gemm::device::GemmUniversal"
        )
        if op.gemm_kind != cutlass_lib.GemmKind.Sparse:
            op_def = op_def.replace("false,", "")
        pattern = re.compile(r"\s*using\s(.*?)\s=")
        decl = op_def.split("\n")[2]

        match = pattern.match(decl)
        if match is None:
            raise RuntimeError("Invalid Gemm config: \n" + op_def)
        op_type = match.groups()[0]
        return op_def, op_type

    def _get_extra_inputs_and_names(
        self,
        op: "cutlass_gemm_op.GemmOperation" = None,  # type: ignore[name-defined]  # noqa: F821
    ) -> tuple[Buffer | None, list[Buffer | None], list[str]]:
        import cutlass_library.library as cutlass_lib

        if op.gemm_kind == cutlass_lib.GemmKind.Sparse:
            Bias = None
            Meta = self.input_nodes[2]
        else:
            Bias = None if len(self.input_nodes) == 2 else self.input_nodes[2]
            Meta = None
        inputs = [Meta]
        names = ["Meta"]
        return (Bias, inputs, names)

    def _update_arg_names_for_test_call_statement(
        self,
        arg_names: list[str],
        input_nodes: list[Buffer],
    ) -> list[str]:
        if input_nodes[3] is None:
            del arg_names[3]
        if input_nodes[2] is None:
            del arg_names[2]
        return arg_names

    def render_gemm_arguments(
        self,
        instance_type: str,
        argument_template: str,
        epilogue_template: str,
        should_swap_xw: bool,
        X: IRNode,
        W: IRNode,
        Bias: IRNode,
        Meta: IRNode,
        Y: IRNode,
        alpha: float,
        beta: float,
        kernel: CUTLASSTemplateKernel,
        epilogue_args,
    ) -> str:
        """
        Render the Cutlass CUDA C++ code required for passing arguments to the GEMM operation.

        Args:
            instance_type (str): GEMM instance type.
            argument_template (str): Template for the GEMM operation arguments.
            epilogue_template (str): Template for the epilogue arguments.
            should_swap_xw (bool): Determines whether X, W operands should be swapped. If True, applies an explicit
            transpose operation to X and W.
            X (IRNode): The X input tensor.
            W (IRNode): The W input tensor.
            Bias (IRNode): The bias tensor.
            Meta (IRNode): The meta tensor.
            Y (IRNode): The output tensor.
            alpha (float): Scaling factor for the product of the inputs.
            beta (float): Scaling factor for the output tensor.
            kernel (CUTLASSTemplateKernel): CUDA Template kernel for the operation.
            epilogue_args (any): Additional arguments for the epilogue state.

        Returns:
            str: A block of CUDA C++ code as a string, ready to be used as arguments for the GEMM operation.

        Note: If `should_swap_xw` is True, a transpose operation will be applied to the X, W, Bias, and Y
        tensors. This operation also implies the M and N dimensions of Bias and GEMM output to be swapped
        before the function call.
        """
        options = {
            "instance_type": instance_type,
            "alpha": alpha,
            "beta": beta,
            "X": X,
            "W": W,
            "Y": Y,
            "Bias": Bias,
            "Meta": Meta,
            "template": self,
            "kernel": kernel,
            "M": "M",
            "N": "N",
            "epilogue_args": epilogue_args,
        }

        if epilogue_template is None:
            arguments = self._template_from_string(argument_template).render(
                split_k=1, **options
            )
            return arguments

        epilogue_arguments = self._template_from_string(epilogue_template).render(
            **options
        )
        arguments = self._template_from_string(argument_template).render(
            epilogue_arguments=epilogue_arguments, **options
        )

        return arguments
<|endoftext|># mypy: allow-untyped-defs
import functools
import hashlib
import inspect
import json
import logging
import os
import time
from typing import Any

import torch._inductor.config as config
from torch._inductor.codecache import cutlass_key
from torch._inductor.codegen.cutlass import serialization, utils
from torch._inductor.codegen.cutlass.serialization import (
    get_cutlass_operation_serializer,
)
from torch._inductor.runtime.cache_dir_utils import cache_dir
from torch._inductor.utils import clear_on_fresh_cache


log = logging.getLogger(__name__)


CONFIG_PREFIX: str = "configs"


def get_config_request_key(
    arch: str,
    toolkit_version: str,
    instantiation_level: str,
) -> str:
    """
    Return a key for the full ops, based on cutlass key, arch, toolkit version, instantiation level, and serialization.py file hash.
    """

    # Get hash of serialization.py and cutlass_utils.py files using their module file paths
    def get_file_hash(file_module):
        file_path = inspect.getfile(file_module)
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    serialization_hash = get_file_hash(serialization)
    cutlass_utils_hash = get_file_hash(utils)

    hash_target = "-".join(
        [
            cutlass_key().hex(),
            arch,
            toolkit_version,
            instantiation_level,
            serialization_hash,
            cutlass_utils_hash,
        ]
    )
    return hashlib.sha256(hash_target.encode("utf-8")).hexdigest()[0:8]


def _generate_config_filename(request_key: str) -> str:
    """
    Generate a filename for the full ops.
    """
    return f"{CONFIG_PREFIX}_{request_key}.json"


@clear_on_fresh_cache
@functools.cache
def maybe_fetch_ops(device_type: str) -> list[Any] | None:
    """
    Fetch ops from databases.
    """
    if config.force_disable_caches:
        return None

    # setup
    arch: str = utils.cutlass_arch(device_type)
    version: str = utils.toolkit_version(device_type)
    if device_type == "cuda":
        # get_cuda_version might return "12.4.0" or "12.4"
        # but we want to use "12.4"
        version = ".".join(version.split(".")[:2])
    instantiation_level: str = config.cutlass.cutlass_instantiation_level

    # filename and filepath
    request_key: str = get_config_request_key(arch, version, instantiation_level)
    filename: str = _generate_config_filename(request_key)
    filepath: str = os.path.join(cache_dir(), filename)

    # try fetch
    serialized_ops: list[str] | None = None
    start_time = time.time()
    if os.path.isfile(filepath):
        # locally
        try:
            with open(filepath) as f:
                serialized_ops = json.load(f)

            if not isinstance(serialized_ops, list):
                raise AssertionError(
                    f"Expected serialized ops is a list, got {type(serialized_ops)}"
                )
        except Exception:
            log.warning(
                "Failed to load CUTLASS config %s from local cache",
                filename,
                exc_info=True,
            )
            serialized_ops = None
    elif config.is_fbcode():
        from torch._inductor.fb.cutlass_remote_cache import (
            maybe_fetch_cutlass_configs_from_remote,
        )

        # from remote
        serialized_ops = maybe_fetch_cutlass_configs_from_remote(filepath)

    if serialized_ops is None:
        return None

    # deserialize
    serializer = get_cutlass_operation_serializer()
    full_ops = [serializer.deserialize(x) for x in serialized_ops]  # type: ignore[union-attr]
    log.info("Loaded ops from %s cache in %.3fs", filename, time.time() - start_time)
    return full_ops
<|endoftext|>import ast
import itertools
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from os import linesep
from typing import Any

import sympy

import torch
import torch._inductor.virtualized as virtualized
from torch._inductor.ir import ComputedBuffer, Pointwise
from torch._inductor.ops_handler import DefaultHandler, WrapperHandler
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.utils import DelayReplaceLine, IndentedBuffer, OrderedSet
from torch._inductor.virtualized import OpsValue

from ...virtualized import V


_ACCUMULATOR_ARG_NAME = "accum"

# 1/sqrt(2) (M_SQRT1_2), the constant Inductor's gelu decomposition multiplies
# the input by before applying erf.
_GELU_ERF_INV_SQRT2 = 0.7071067811865476


def _is_close_constant(node: ast.expr, value: float, tol: float = 1e-6) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
        and abs(float(node.value) - value) < tol
    )


def _mult_factors(node: ast.expr) -> list[ast.expr]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _mult_factors(node.left) + _mult_factors(node.right)
    return [node]


def _match_call(node: ast.expr, name: str) -> "ast.Call | None":
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
        and len(node.args) == 1
    ):
        return node
    return None


def _same(a: ast.expr, b: ast.expr) -> bool:
    return ast.dump(a) == ast.dump(b)


def _match_gelu_erf(node: ast.expr) -> "ast.expr | None":
    """Match gelu's erf decomposition ``x * 0.5 * (1 + erf(x / sqrt(2)))`` and
    return the inner argument ``x``. Matching is commutativity-aware for the
    multiplications and the addition.
    """
    factors = _mult_factors(node)
    if len(factors) != 3:
        return None
    half = next((f for f in factors if _is_close_constant(f, 0.5)), None)
    if half is None:
        return None
    rest = [f for f in factors if f is not half]

    def match_one_plus_erf(f: ast.expr) -> "ast.expr | None":
        # Match ``1 + erf(x / sqrt(2))`` and return ``x``.
        if not (isinstance(f, ast.BinOp) and isinstance(f.op, ast.Add)):
            return None
        operands = [f.left, f.right]
        if not any(_is_close_constant(o, 1.0) for o in operands):
            return None
        erf_call = next(
            (c for c in (_match_call(o, "erf") for o in operands) if c is not None),
            None,
        )
        if erf_call is None:
            return None
        erf_factors = _mult_factors(erf_call.args[0])
        if len(erf_factors) != 2 or not any(
            _is_close_constant(g, _GELU_ERF_INV_SQRT2) for g in erf_factors
        ):
            return None
        return next(
            g for g in erf_factors if not _is_close_constant(g, _GELU_ERF_INV_SQRT2)
        )

    # ``x`` may itself be an addition (e.g. gelu(accum + bias)), so identify the
    # ``1 + erf(...)`` factor by structure rather than assuming it is the only Add.
    for i, f in enumerate(rest):
        erf_x = match_one_plus_erf(f)
        if erf_x is not None and _same(erf_x, rest[1 - i]):
            return rest[1 - i]
    return None


def _match_silu(node: ast.expr) -> "ast.expr | None":
    """Match SiLU decomposition ``x / (1 + exp(0.0 - x))`` and return ``x``.

    Inductor decomposes silu(x) as x / (1 + exp(-x)). With our neg() op
    emitting (0.0 - x), the generated code is: x / (1.0 + exp((0.0 - x))).
    """
    # Must be a division: x / denom
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)):
        return None
    x = node.left
    denom = node.right

    # denom must be: 1.0 + exp(neg_x) or exp(neg_x) + 1.0
    if not (isinstance(denom, ast.BinOp) and isinstance(denom.op, ast.Add)):
        return None

    operands = [denom.left, denom.right]
    one = next((o for o in operands if _is_close_constant(o, 1.0)), None)
    if one is None:
        return None
    exp_part = next((o for o in operands if o is not one), None)
    if exp_part is None:
        return None

    # exp_part must be exp(neg_x)
    exp_call = _match_call(exp_part, "exp")
    if exp_call is None:
        return None
    neg_x = exp_call.args[0]

    # neg_x must be (0.0 - x) where x matches the numerator
    if (
        isinstance(neg_x, ast.BinOp)
        and isinstance(neg_x.op, ast.Sub)
        and _is_close_constant(neg_x.left, 0.0)
        and _same(neg_x.right, x)
    ):
        return x

    # Also handle unary minus: -x
    if isinstance(neg_x, ast.UnaryOp) and isinstance(neg_x.op, ast.USub):
        if _same(neg_x.operand, x):
            return x

    return None


@dataclass(frozen=True)
class _ActivationPattern:
    # ``name`` must be a functor the CUTLASS EVT frontend binds natively (see
    # ``ast_op_to_bindings`` in the cutlass python_ast frontend).
    name: str
    # Cheap substring guard so we only parse epilogues that may contain the
    # decomposition (a primitive that the activation expands into).
    trigger: str
    match: Callable[[ast.expr], "ast.expr | None"]


# Activations that Inductor decomposes but CUTLASS can emit as a single functor.
# Add new entries here to re-fuse additional decomposed activations.
_ACTIVATION_PATTERNS: tuple[_ActivationPattern, ...] = (
    _ActivationPattern("gelu", "erf", _match_gelu_erf),
    _ActivationPattern("silu", "0.0 -", _match_silu),
)


def _fuse_activations(code: str) -> str:
    """Re-compose decomposed activations back into single CUTLASS functor calls.

    Inductor lowers activations such as ``aten.gelu`` into
    primitive ops (``erf``, ``sigmoid``, ``mul``, ...) before reaching CUTLASS
    codegen. The CUTLASS EVT frontend natively supports a number of activation
    functors, so we pattern-match each decomposition (``_ACTIVATION_PATTERNS``)
    and fold it back into a single activation node. This enables fusion for
    activations whose primitives are unsupported (e.g. the ``erf`` in gelu) and
    reduces the compute-node count for the rest.
    """
    active = [p for p in _ACTIVATION_PATTERNS if p.trigger in code]
    if not active:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
        return code
    func = tree.body[0]

    assigns: dict[str, ast.expr] = {}
    for stmt in func.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            assigns[stmt.targets[0].id] = stmt.value

    _EMPTY_SET: OrderedSet[str] = OrderedSet()

    def inline(node: ast.expr, seen: OrderedSet[str] | None = None) -> ast.expr:
        if seen is None:
            seen = _EMPTY_SET
        # Fully inline temporary assignments so the expression tree only refers
        # to function parameters (accum / read buffers) and literal constants.
        if isinstance(node, ast.Name) and node.id in assigns:
            if node.id in seen:
                return node
            return inline(assigns[node.id], seen | OrderedSet([node.id]))
        if isinstance(node, ast.BinOp):
            return ast.BinOp(
                left=inline(node.left, seen),
                op=node.op,
                right=inline(node.right, seen),
            )
        if isinstance(node, ast.UnaryOp):
            return ast.UnaryOp(op=node.op, operand=inline(node.operand, seen))
        if isinstance(node, ast.Call):
            return ast.Call(
                func=node.func,
                args=[inline(a, seen) for a in node.args],
                keywords=[],
            )
        return node

    changed = False
    for stmt in func.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            inlined = inline(stmt.value)
            for pattern in active:
                x = pattern.match(inlined)
                if x is not None:
                    stmt.value = ast.Call(
                        func=ast.Name(id=pattern.name, ctx=ast.Load()),
                        args=[x],
                        keywords=[],
                    )
                    changed = True
                    break

    if not changed:
        return code

    # The folded assignment no longer references the decomposition temporaries,
    # so drop the now-dead intermediates (the CUTLASS frontend builds a compute
    # node for every assignment it sees). The frontend emits ``return`` dedented
    # to module level (a sibling of the function def in Python 3.12), so seed
    # liveness from every Return in the tree, not only those inside ``func.body``.
    live: OrderedSet[str] = OrderedSet()
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and node.value is not None:
            live.update(n.id for n in ast.walk(node.value) if isinstance(n, ast.Name))

    kept: list[ast.stmt] = []
    for stmt in reversed(func.body):
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            if stmt.targets[0].id not in live:
                continue
            live.update(n.id for n in ast.walk(stmt.value) if isinstance(n, ast.Name))
        kept.append(stmt)
    func.body = list(reversed(kept))

    folded = ast.unparse(ast.fix_missing_locations(tree))

    # After dead-code elimination, any activation primitive that served as a
    # trigger (e.g. erf) should no longer appear in the live code -- it was
    # consumed by the pattern match and eliminated.  If one still appears, the
    # fold was incomplete (e.g. a standalone erf not part of a known pattern);
    # bail out and return the original code so the CUTLASS frontend rejects the
    # epilogue cleanly rather than emitting an unsupported functor.
    for p in active:
        if p.trigger in folded:
            return code

    return folded


def scaled_mm_evt(
    scale_A_name: str, scale_B_name: str, bias_name: str | None, output_name: str
) -> tuple[list[str], dict[str, Any], str]:
    evt_read_names = [scale_A_name, scale_B_name]
    var_name_to_buffer_name = {n: n for n in [scale_A_name, scale_B_name]}
    var_name_to_buffer_name["D"] = output_name
    var_name_to_buffer_name[_ACCUMULATOR_ARG_NAME] = output_name
    expr = f"accum * {scale_A_name} * {scale_B_name}{linesep}"
    if bias_name:
        expr = f"({expr}) + {bias_name}"
        evt_read_names.append(bias_name)
        var_name_to_buffer_name[bias_name] = bias_name

    evt_py_code = f"def fn(accum, {','.join(evt_read_names)}):{linesep}\
    D = {expr}{linesep}\
    return D{linesep}"

    return evt_read_names, var_name_to_buffer_name, evt_py_code


class CutlassEVTOpsMixIn:
    @staticmethod
    def _infix_bin_op(op: str, a: str, b: str) -> str:
        return f"{a} {op} {b}"

    @staticmethod
    def _prefix_bin_op(op: str, a: str, b: str) -> str:
        return f"{op}({a}, {b})"

    @staticmethod
    def _prefix_un_op(op: str, a: str) -> str:
        return f"{op}({a})"

    @staticmethod
    def to_dtype(
        x: str,
        dtype: Any,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = False,
    ) -> str:
        return x

    @staticmethod
    def constant(value: Any, dtype: Any) -> str:
        # Return the constant as a Python float literal so the EVT python code
        # string can embed it directly (e.g. for the 0.5 and 1/sqrt(2) factors
        # in gelu).  The CUTLASS EVT frontend will render it as a C++ scalar.
        # Only numeric constants appear here (Inductor does not lower non-numeric
        # constants into pointwise epilogues).
        return str(float(value))

    @staticmethod
    def neg(x0: str) -> str:
        # Use subtraction from zero instead of unary minus because the
        # CUTLASS PythonASTFrontend has visit_BinOp but no visit_UnaryOp.
        return f"(0.0 - {x0})"

    @staticmethod
    def mul(x0: str, x1: str) -> str:
        return CutlassEVTOpsMixIn._infix_bin_op("*", x0, x1)

    @staticmethod
    def truediv(x0: str, x1: str) -> str:
        return CutlassEVTOpsMixIn._infix_bin_op("/", x0, x1)

    @staticmethod
    def ge(x0: str, x1: str) -> str:
        raise NotImplementedError

    @staticmethod
    def add(x0: str, x1: str) -> str:
        return CutlassEVTOpsMixIn._infix_bin_op("+", x0, x1)

    @staticmethod
    def relu(x0: str) -> str:
        return CutlassEVTOpsMixIn._prefix_un_op("relu", x0)

    @staticmethod
    def sigmoid(x0: str) -> str:
        return CutlassEVTOpsMixIn._prefix_un_op("sigmoid", x0)

    @staticmethod
    def sub(x0: str, x1: str) -> str:
        return CutlassEVTOpsMixIn._infix_bin_op("-", x0, x1)

    @staticmethod
    def tanh(x0: str) -> str:
        return CutlassEVTOpsMixIn._prefix_un_op("tanh", x0)

    @staticmethod
    def exp(x0: str) -> str:
        return CutlassEVTOpsMixIn._prefix_un_op("exp", x0)

    @staticmethod
    def erf(x0: str) -> str:
        return CutlassEVTOpsMixIn._prefix_un_op("erf", x0)

    @staticmethod
    def silu(x0: str) -> str:
        return CutlassEVTOpsMixIn._prefix_un_op("silu", x0)


class MockCutlassHandler(CutlassEVTOpsMixIn, WrapperHandler):
    """Passthrough handler for cutlass ops, used for running epilogue nodes for memory planning"""


class _AssignmentFormatter(DefaultHandler):
    def __init__(self, parent_handler: "CutlassEVTCodegen"):
        self.parent_handler = parent_handler

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        # Handle op dispatch here
        if hasattr(self.parent_handler, name):
            fn = getattr(self.parent_handler, name)
            line = fn(*args, **kwargs)
            if name in ("load", "store"):
                return OpsValue(line)
            else:
                var = self.parent_handler._tmp_var()
                line = DelayReplaceLine(
                    var,
                    lambda: "D"
                    if var == self.parent_handler.last_stored_var_name
                    else var,
                    f"{var} = {line}",
                )
                self.parent_handler.body.writeline(line)
                return OpsValue(var)
        else:
            raise NotImplementedError(name)


class CutlassEVTCodegen(CutlassEVTOpsMixIn):
    """
    Notes:
        * Used by CUTLASSGemmTemplate.
        * This class should not be instantiated by users, it is intended to be used
            by calling CutlassEVTCodegen.ir_to_evt_python_code(...)
            which instantiates this class as an ops handler for virtualized.V.ops.[op-name]
        * Extend this with more _op_<whatever> nodes to add support for new pointwise operations.
    """

    def __init__(self, accumulator_node_name: str, removed_buffers: OrderedSet[str]):
        """

        Initializes a CutlassEVTEpilogueArgumentFormatter object. Do not instantiate directly.
        Use the CutlassEVTCodegen.ir_to_evt_python_code static method.

        Args:
            accumulator_node_name: The name of the accumulator node which should contain
                                          the Matmul result before fusion according to the IR graph.
            epilogue_nodes: The list of scheduler nodes to be fused into the epilogue
        """
        self.accumulator_node_name: str = accumulator_node_name  #
        self.body: IndentedBuffer = IndentedBuffer(1)  # The body buffer for codegen
        self.var_counter: Iterator[int] = itertools.count()
        self.store_name_to_value: dict[
            str, OpsValue
        ] = {}  # Aliases for subexpression functors
        self.reads: OrderedSet[str] = OrderedSet([])
        # Used for creating example tensors
        self.var_name_to_buffer_name: dict[str, str] = {
            _ACCUMULATOR_ARG_NAME: accumulator_node_name
        }
        self.removed_buffers: OrderedSet[str] = removed_buffers
        self.cur_node: ComputedBuffer | None = None
        self.name_to_buffer = V.graph.name_to_buffer | V.graph.graph_inputs
        for name in V.graph.constants:
            # pyrefly: ignore [unsupported-operation]
            self.name_to_buffer[name] = V.graph.add_tensor_constant(
                V.graph.constants[name], name
            )
        self.is_D_assigned = False
        self.D_var_name = None

        if accumulator_node_name not in removed_buffers:
            # cannot return accumulator directly, so alias it
            var = self._tmp_var()
            self.body.writeline(f"{var} = {_ACCUMULATOR_ARG_NAME}")
            self.store(accumulator_node_name, value=OpsValue(var))

    @staticmethod
    def ir_to_evt_python_code(
        cutlass_template_node_name: str,
        epilogue_nodes: list[BaseSchedulerNode],
        removed_buffers: OrderedSet[str],
        fn_name: str = "fn",
        as_standalone_function: bool = False,
    ) -> tuple[list[str], list[str], dict[str, Any], str]:
        codegen = CutlassEVTCodegen(cutlass_template_node_name, removed_buffers)
        handler = _AssignmentFormatter(codegen)

        with virtualized.V.set_ops_handler(handler):
            for s_node in epilogue_nodes:
                node = s_node.node
                if not isinstance(node, ComputedBuffer):
                    raise AssertionError(
                        f"expected node to be a ComputedBuffer, got {type(node)}"
                    )
                with codegen.set_cur_node(node):
                    index_vars = CutlassEVTCodegen.get_index_vars(node)
                    node.get_store_function()(index_vars)

        codegen.finalize()

        return (
            codegen.get_reads(),
            codegen.get_writes(),
            codegen.get_renames(),
            codegen.get_value(
                fn_name=fn_name,
                as_standalone_function=as_standalone_function,
            ),
        )

    def get_value(
        self,
        fn_name: str = "fn",
        as_standalone_function: bool = False,
    ) -> str:
        ret = self._render_return_statement()
        if as_standalone_function:
            ret = "    " + ret
        return _fuse_activations(
            linesep.join(
                [
                    self._render_input_signature(fn_name),
                    self.body.getvalue(),
                    ret,
                ]
            )
        )

    def finalize(self) -> None:
        # Rename the last store to D
        # no other code references this store
        # to workaround https://github.com/NVIDIA/cutlass/issues/2288
        # Note: the delayed line will automatically rewrite the last assignment to
        # be to D
        buffer_name = self.var_name_to_buffer_name[self.last_stored_var_name]
        self.var_name_to_buffer_name.pop(self.last_stored_var_name)
        self.var_name_to_buffer_name["D"] = buffer_name
        self.store_name_to_value[buffer_name] = OpsValue("D")

    @contextmanager
    def set_cur_node(self, node: ComputedBuffer) -> Generator[None, Any, Any]:
        prev_node = self.cur_node
        try:
            self.cur_node = node
            yield
        finally:
            self.cur_node = prev_node

    def get_renames(self) -> dict[str, str]:
        return dict(self.var_name_to_buffer_name)

    def get_reads(self) -> list[str]:
        return list(self.reads.difference(self.store_name_to_value.keys()))

    def get_writes(self) -> list[str]:
        return list(self.store_name_to_value.keys())

    def load(self, name: str, index: Any) -> str:
        self._check_indexing(name, index)
        if name in self.store_name_to_value:
            return self.store_name_to_value[name].value
        elif name == self.accumulator_node_name:
            return _ACCUMULATOR_ARG_NAME
        else:
            self.reads.add(name)
            self.var_name_to_buffer_name[name] = name
            return name

    def store(
        self, name: Any, index: Any = None, value: Any = None, mode: Any = None
    ) -> None:
        if name not in self.removed_buffers:
            if index:
                self._check_indexing(name, index)
            if value.value == _ACCUMULATOR_ARG_NAME:
                raise AssertionError("Cannot store accumulator arg name")
            self.var_name_to_buffer_name[value.value] = name
            self.store_name_to_value[name] = value
            self.last_stored_var_name = value.value
        return None

    def _get_cur_node(self) -> ComputedBuffer:
        if not self.cur_node:
            raise AssertionError("expected cur_node to be set")
        return self.cur_node

    @staticmethod
    def get_index_vars(node: ComputedBuffer) -> Sequence[sympy.Expr]:
        data = node.data
        # TODO mlazos: relax this, cutlass supports reductions and other ops
        if not isinstance(data, Pointwise):
            raise AssertionError(f"expected data to be Pointwise, got {type(data)}")
        return data._index(data.ranges)

    def _get_current_index_vars(self) -> Sequence[sympy.Expr]:
        return self.get_index_vars(self._get_cur_node())

    def _check_indexing(self, name: str, index: sympy.Expr) -> None:
        # We only support indexing that matches the layout today because
        # CUTLASS doesn't support arbitrary indexing
        buffer_name = (
            self.accumulator_node_name if name == _ACCUMULATOR_ARG_NAME else name
        )
        buffer = self.name_to_buffer[buffer_name]
        index_strides = V.graph.sizevars.stride_vars(
            index, self._get_current_index_vars()
        )
        stride = buffer.get_layout().stride
        if not self._stride_compatible(stride, index_strides):
            raise NotImplementedError(
                f"Unsupported indexing for {name} with index {index}, index strides {index_strides}, and layout stride {stride}"
            )

    def _stride_compatible(
        self, left: Iterable[sympy.Expr], right: Iterable[sympy.Expr]
    ) -> bool:
        def _provably_equal_or_zero(a: sympy.Expr, b: sympy.Expr) -> bool:
            # sympy.Eq can return an unevaluated Equality object; only accept
            # cases sympy can prove true.
            if a == b:
                return True
            if V.graph.sizevars.statically_known_equals(a, b):
                return True
            if V.graph.sizevars.statically_known_equals(a, 0):
                return True
            if V.graph.sizevars.statically_known_equals(b, 0):
                return True
            return (
                sympy.Eq(a, b) is sympy.true
                or sympy.Eq(a, 0) is sympy.true
                or sympy.Eq(b, 0) is sympy.true
            )

        left_list = list(left)
        right_list = list(right)
        # Same length: direct comparison
        if len(left_list) == len(right_list):
            return all(
                _provably_equal_or_zero(lv, rv) for lv, rv in zip(left_list, right_list)
            )
        # Different lengths: allow compatible reshapes where trailing strides match.
        # This handles view/reshape between template output and consumer, e.g.,
        # buffer strides [3072, 1] vs index strides [1572864, 3072, 1]
        shorter, longer = (
            (left_list, right_list)
            if len(left_list) <= len(right_list)
            else (right_list, left_list)
        )
        n = len(shorter)
        # Check that the trailing strides match
        return all(
            _provably_equal_or_zero(shorter[-(i + 1)], longer[-(i + 1)])
            for i in range(n)
        )

    def _render_input_signature(self, fn_name: str = "fn") -> str:
        arguments = ", ".join(
            [_ACCUMULATOR_ARG_NAME]
            + [name for name in self.reads if name != self.accumulator_node_name]
        )
        return f"def {fn_name}({arguments}):"

    def _render_return_statement(self) -> str:
        return_vars = OrderedSet(
            op_v.value for op_v in self.store_name_to_value.values()
        )
        if "D" not in return_vars:
            raise AssertionError(f"expected 'D' in return_vars, got {return_vars}")
        return f"return {', '.join(return_vars)}"

    def _tmp_var(self) -> str:
        return f"tmp_{next(self.var_counter)}"
<|endoftext|># mypy: allow-untyped-defs
import functools
import itertools
import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TYPE_CHECKING

from sympy import Expr, symbols

import torch._inductor.config as config
from torch import dtype as torch_dtype
from torch._inductor.codegen.common import get_device_op_overrides
from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.utils import do_bench_using_profiling, OrderedSet, Placeholder
from torch.utils._sympy.value_ranges import ValueRanges

from .utils import DTYPE_TO_CUTLASS_TYPE


if TYPE_CHECKING:
    from .template import ArgInfo

from ...autotune_process import CUTLASSBenchmarkRequest
from ...ir import (
    Buffer,
    ChoiceCaller,
    CUTLASSTemplateBuffer,
    IRNode,
    Layout,
    PrimitiveInfoType,
    TensorBox,
)
from ...utils import sympy_product
from ...virtualized import V
from ..common import (
    CSEVariable,
    IndentedBuffer,
    Kernel,
    OpOverrides,
    WorkspaceArg,
    WorkspaceZeroMode,
)
from ..cpp_utils import CppPrinter, DTYPE_TO_CPP


if TYPE_CHECKING:
    from torch._inductor.codegen.cutlass.template import CUTLASSTemplate

log = logging.getLogger(__name__)

cexpr = CppPrinter().doprint


def _normalize_idx(index: int, total_length: int) -> int:
    return index if index >= 0 else index + total_length


ValidLayoutSymbols = Literal["M", "N", "K", "B", "lda", "ldb", "ldc", "ldd"]
ValidLayoutAttrs = Literal["size", "stride"]


@dataclass(frozen=True)
class LayoutArg:
    node: IRNode
    symbol: ValidLayoutSymbols
    attr: ValidLayoutAttrs
    dim: int

    def matches(self, node, attr, dim) -> bool:
        return self.node == node and self.attr == attr and self.dim == dim


class CUTLASSKernel(Kernel):
    """
    Baseclass for Cutlass based Kernels
    """

    overrides = OpOverrides  # type: ignore[assignment]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layout_args: dict[str, list[LayoutArg]] = defaultdict(list)
        self.size_args: list[Expr | int] = []
        # Mapping from arg name to IRNode.
        self.named_nodes: dict[str, IRNode] = {}

    def find_symbol(self, node: IRNode, attr: ValidLayoutAttrs, dim: int) -> str | None:
        arg = self.find_layout_arg(node, attr, dim)
        return arg.symbol if arg else None

    def find_layout_arg(
        self, node: IRNode, attr: ValidLayoutAttrs, dim: int
    ) -> LayoutArg | None:
        matches = [
            arg
            for arg in itertools.chain.from_iterable(self.layout_args.values())
            if arg.matches(node, attr, dim)
        ]
        if len(matches) >= 1:
            # Verify all matches have the same node, attribute, and dimension
            # And if they come from the same node, whichever symbol we use is fine.
            # if in runtime the logic changes, this would trigger guard
            first_match = matches[0]
            if not all(
                match.node == first_match.node
                and match.attr == first_match.attr
                and match.dim == first_match.dim
                for match in matches
            ):
                raise AssertionError("All matching layout args should be identical")
            return first_match
        attr_values = node.get_size() if attr == "size" else node.get_stride()
        if dim >= len(attr_values):
            return None
        expr = attr_values[dim]
        fallback_matches = []
        for arg in itertools.chain.from_iterable(self.layout_args.values()):
            if arg.attr != attr:
                continue
            if arg.node.get_name() != node.get_name():
                continue
            arg_values = (
                arg.node.get_size() if arg.attr == "size" else arg.node.get_stride()
            )
            if arg.dim >= len(arg_values):
                continue
            if arg_values[arg.dim] == expr:
                fallback_matches.append(arg)
        if fallback_matches:
            return fallback_matches[0]
        return None

    def add_layout_arg(
        self, symbol: ValidLayoutSymbols, node: IRNode, attr: ValidLayoutAttrs, dim: int
    ):
        arg = LayoutArg(node, symbol, attr, dim)
        self.layout_args[symbol].append(arg)

    def init_layout_args(self) -> None:
        X = self.named_nodes["X"]
        W = self.named_nodes["W"]
        Y = self.named_nodes["Y"]
        Bias = self.named_nodes.get("Bias", None)
        x_mdim = _normalize_idx(-2, len(X.get_size()))
        x_kdim = _normalize_idx(-1, len(X.get_size()))
        w_kdim = _normalize_idx(-2, len(W.get_size()))
        w_ndim = _normalize_idx(-1, len(W.get_size()))
        y_mdim = _normalize_idx(-2, len(Y.get_size()))
        y_ndim = _normalize_idx(-1, len(Y.get_size()))
        self.add_layout_arg("M", X, "size", x_mdim)
        self.add_layout_arg("K", X, "size", x_kdim)
        self.add_layout_arg("K", W, "size", w_kdim)
        self.add_layout_arg("N", W, "size", w_ndim)
        self.add_layout_arg("M", Y, "size", y_mdim)
        self.add_layout_arg("N", Y, "size", y_ndim)
        if len(X.get_size()) > 2:
            self.add_layout_arg("B", X, "size", 0)

        lda_dim = self.find_ld_idx(X)
        ldb_dim = self.find_ld_idx(W)
        ldc_dim = self.find_ld_idx(Bias) if Bias else None
        ldd_dim = self.find_ld_idx(Y)
        self.add_layout_arg("lda", X, "stride", lda_dim)
        self.add_layout_arg("ldb", W, "stride", ldb_dim)
        if Bias is not None and ldc_dim is not None:
            self.add_layout_arg("ldc", Bias, "stride", ldc_dim)
        self.add_layout_arg("ldd", Y, "stride", ldd_dim)

    def get_layout_args(self) -> tuple[Expr | int, ...]:
        X = self.named_nodes["X"]
        W = self.named_nodes["W"]
        Y = self.named_nodes["Y"]
        Bias = self.named_nodes.get("Bias", None)
        mdim = _normalize_idx(-2, len(X.get_size()))
        ndim = _normalize_idx(-1, len(W.get_size()))
        kdim = _normalize_idx(-1, len(X.get_size()))

        def get_ld(node) -> Expr | int:
            dim = self.find_ld_idx(node)
            return node.get_stride()[dim]

        M = X.get_size()[mdim]
        N = W.get_size()[ndim]
        K = X.get_size()[kdim]
        B = X.get_size()[0] if len(X.get_size()) > 2 else 1
        LDA = get_ld(X)
        LDB = get_ld(W)
        LDC = get_ld(Bias) if Bias else 0
        LDD = get_ld(Y)
        return (M, N, K, B, LDA, LDB, LDC, LDD)

    def get_dynamic_shape_args(self) -> list[Expr | int]:
        return [*self.get_layout_args(), *self.size_args]

    def get_offset_args(self) -> list[Expr]:
        return [node.get_layout().offset for node in self.named_nodes.values()]

    @staticmethod
    def find_ld_idx(node: IRNode) -> int:
        strides = node.get_stride()
        # Handle 1D tensor case
        if V.graph.sizevars.statically_known_equals(strides[-1], 1):
            return _normalize_idx(-2, len(strides))

        if not V.graph.sizevars.statically_known_equals(strides[-2], 1):
            raise AssertionError(strides[-2])
        return _normalize_idx(-1, len(strides))


class CUTLASSTemplateKernel(CUTLASSKernel):
    """
    Template kernels defined by Cutlass in C++.
    """

    def __init__(
        self,
        kernel_name: str,
        runtime_arg_info: list["ArgInfo"],
        runtime_arg_values: list[Any],
        device_type: str = "cuda",  # type: ignore[assignment]
    ) -> None:
        """
        Initializes a new instance of the CUTLASSTemplateKernel class.

        Args:
            kernel_name (str): The name of the kernel.
        """
        super().__init__()
        self.kernel_name = kernel_name
        self.runtime_arg_info = runtime_arg_info
        self.runtime_arg_values = runtime_arg_values
        self.device_type = device_type
        self.device_codegen = get_device_op_overrides(self.device_type)
        self._EXTRA_CPP_ARGS = f"size_t* workspace_size, uint8_t* workspace, {self.device_codegen.cpp_stream_type()} stream"

    def check_not_null(self, node: IRNode) -> str:
        """
        Generates code to check that a node is not null.
        """
        if node is None:
            return ""

        size_str = self.size(node, 0, -1)
        name_str = self.arg_name(node)
        if name_str is None:
            return ""

        res = IndentedBuffer(initial_indent=2)
        res.tabwidth = 1
        res.splice(
            f"""
            {{
              if (!{name_str}) {{
                int64_t {name_str}_size = {size_str};
                if ({name_str}_size > 0) {{
                  throw std::runtime_error("input {name_str} is null but size is not 0!");
                }}
              }}
            }}
            """
        )
        return res.getvalue()

    def get_signature(self) -> str:
        return self.signature

    def _collect_unbound_layout_free_symbols(self, node: IRNode) -> OrderedSet[Expr]:
        free_symbols: OrderedSet[Expr] = OrderedSet()
        for attr_name, values in (
            ("size", node.get_size()),
            ("stride", node.get_stride()),
        ):
            attr = attr_name  # help mypy narrow the Literal argument below
            for dim, expr in enumerate(values):
                if not isinstance(expr, Expr):
                    continue
                if self.find_layout_arg(node, attr, dim) is not None:
                    continue
                for symbol in expr.free_symbols:
                    free_symbols.add(symbol)  # type: ignore[arg-type]
        return free_symbols

    def def_kernel(
        self,
        inputs: list[IRNode],
        outputs: list[IRNode],
        names_str: str = "",
        input_reorder: list[int] | None = None,
    ) -> str:
        """
        Hook called from template code to generate function definition and
        needed args.

        Args:
            inputs: List of input IRNodes
            outputs: List of output IRNodes
            names_str: Comma separated list of input + output argument names.
            input_reorder: The actual order of input nodes.
                           e.g. The template might have input argument defined as [X, W, Bias],
                           and the actual input passed into this template could be [Bias, X, W].
                           In this case, the `input_reorder` would be [2, 0, 1].
            additional_size_args: Additional size arguments for epilogue inputs
        """
        # NB: name order matters here, it's used to match up offsets
        names = [x.strip() for x in names_str.strip().split(",")]
        if len(inputs) + len(outputs) != len(names):
            raise RuntimeError(
                f"{len(inputs) + len(outputs)=} != {len(names)=}, {inputs=}, {outputs=}, {names=}"
            )

        if input_reorder is not None:
            if len(inputs) != len(input_reorder):
                raise AssertionError(
                    f"expected len(inputs) == len(input_reorder), got "
                    f"{len(inputs)} != {len(input_reorder)}"
                )
        else:
            input_reorder = list(range(len(inputs)))

        for idx in input_reorder:
            name = names[idx]
            node = inputs[idx]
            if node is not None:
                self.named_nodes[name] = node
                self.args.input_buffers[node.get_name()] = name

        for name, node in zip(names[len(inputs) : len(inputs) + len(outputs)], outputs):
            if node is not None:
                # NB: named nodes must be populated in the order of names
                self.named_nodes[name] = node
                self.args.output_buffers[node.get_name()] = name

        arg_defs, *_ = self.args.cpp_argdefs(DTYPE_TO_CUTLASS_TYPE)

        self.init_layout_args()
        free_symbols: OrderedSet[Expr] = OrderedSet()
        for node in self.named_nodes.values():
            free_symbols |= self._collect_unbound_layout_free_symbols(node)
        size_vars = ["M", "N", "K", "B", "lda", "ldb", "ldc", "ldd"]
        size_vars.extend(str(s) for s in free_symbols)
        self.size_args.extend(free_symbols)
        size_args = [f"const int {s}" for s in size_vars]
        offset_args = [f"const int {name}_offset" for name in self.named_nodes]
        runtime_arg_decls = ",".join(
            [f"{arg.ty} {arg.name}" for arg in self.runtime_arg_info]
        )
        if runtime_arg_decls:
            runtime_arg_decls += ", "

        signature = (
            f"int {self.kernel_name}({', '.join(arg_defs + size_args + offset_args)},\
 {runtime_arg_decls}{self._EXTRA_CPP_ARGS})"
        )
        self.signature = signature
        return signature

    def call_kernel(
        self,
        name: str,
        node: "CUTLASSTemplateBuffer",  # type: ignore[name-defined]
    ) -> None:
        """
        Generates code to call the kernel through V.graph.wrapper_code.
        used from within torch._inductor.wrapper.PythonWrapperCodegen

        name: Name of kernel function.
        node: The CUTLASSTemplateBuffer node which contains information about the kernel, it's fused epilogue nodes
        as well as all required inputs and outputs.
        """
        wrapper = V.graph.wrapper_code

        arg_types: list[Any]
        if V.graph.cpp_wrapper:
            # Make sure we initialize these kernels since they're exported as
            # C-style symbol names.
            if not isinstance(wrapper, CppWrapperCpu):
                raise AssertionError(
                    f"expected wrapper to be CppWrapperCpu, got {type(wrapper)}"
                )
            wrapper.initialized_kernels[name] = self
            # We always originally initialize name with "KERNEL_NAME". So, we
            # we replace with the real kernel name passed as an arg to this function.
            self.signature = self.signature.replace(str(Placeholder.KERNEL_NAME), name)
            _, call_args, arg_types = self.args.cpp_argdefs(DTYPE_TO_CUTLASS_TYPE)
        else:
            _, call_args, _, arg_types = self.args.python_argdefs()

        dynamic_shape_args = self.get_dynamic_shape_args()
        offset_args = self.get_offset_args()
        call_args.extend(dynamic_shape_args)  # type: ignore[arg-type]
        call_args.extend(offset_args)  # type: ignore[arg-type]
        for arg in self.runtime_arg_values:
            call_args.append(str(arg))
        arg_types.extend("const int" for _ in dynamic_shape_args)
        arg_types.extend("const int" for _ in offset_args)
        for arg in self.runtime_arg_info:
            arg_types.append(arg.ty)
        # dynamo wraps unspec variable as 0d CPU tensor, need convert to scalar
        for i in range(len(call_args)):
            if V.graph.is_unspec_arg(call_args[i]):
                call_args[i] = call_args[i] + ".item()"
            elif isinstance(arg_types[i], torch_dtype):
                call_args[i] = (
                    call_args[i]
                    if V.graph.cpp_wrapper
                    else f"c_void_p({call_args[i]}.data_ptr())"
                )

        # workspace_size ptr is NULL to mark this call is not intended for retrieving workspace_size.
        # workspace_size should have already been retrieved prior to this call.
        # workspace_size is here.
        call_args.append("nullptr" if V.graph.cpp_wrapper else "None")
        if V.graph.cpp_wrapper:
            arg_types.append("size_t*")

        if node.get_workspace_size() > 0:
            ws = WorkspaceArg(
                count=node.get_workspace_size(),
                device=V.graph.get_current_device_or_throw(),
                zero_mode=WorkspaceZeroMode.UNINITIALIZED,
                outer_name=WorkspaceArg.unique_name(),
            )
            wrapper.generate_workspace_allocation(ws)
            workspace = str(ws.outer_name)
            call_args.append(
                workspace
                if V.graph.cpp_wrapper
                else f"c_void_p({workspace}.data_ptr())"
            )
        else:
            ws = None
            call_args.append("nullptr" if V.graph.cpp_wrapper else "None")
        if V.graph.cpp_wrapper:
            arg_types.append("uint8_t*")

        wrapper.generate_kernel_call(
            name,
            call_args,
            triton=False,
            arg_types=arg_types,
        )
        if ws:
            wrapper.generate_workspace_deallocation(ws)

    def dtype(self, node: IRNode) -> str | None:
        """
        Generates code which represents dtype of a given node.
        """

        if node is None:
            return "void"
        return DTYPE_TO_CPP.get(node.get_layout().dtype)

    def cutlass_dtype(self, node: IRNode, default_dtype="void") -> str | None:
        # Helper method, called into from CUTLASSGemmTemplate
        if node is None:
            return default_dtype
        from torch._inductor.codegen.cutlass.template import CUTLASSTemplate

        return CUTLASSTemplate._DTYPE_TO_CUTLASS[node.get_layout().dtype]

    def max_valid_index(self, node: IRNode, default=-1):
        # Helper method, called into from CUTLASSGemmTemplate
        if node is None:
            return default
        max_valid_offset = 0
        for i in range(len(node.get_size())):
            max_valid_offset += (node.get_size()[i] - 1) * node.get_stride()[i]
        return max_valid_offset

    def ptr(self, node: IRNode) -> str:
        """
        Generates code which represents pointer of a given node.
        """

        if node is None:
            return "nullptr"
        arg_name = self.arg_name(node)
        if arg_name is None:
            return "nullptr"
        return f"{arg_name} + {arg_name}_offset"

    def size(
        self,
        node: IRNode,
        start_index: int,
        end_index: int | None = None,
        default_value: int = 0,
    ) -> str:
        """
        Hook called from template code to get the size of an arg.
        Generates code which represents size of a given node in [start_index, end_index).
        If node is None, returns default_value.

        TODO: Will add needed args to pass it in if it is dynamic.
        """

        if node is None:
            return str(default_value)

        start_index = _normalize_idx(start_index, len(node.get_size()))
        if end_index is None:
            end_index = start_index
        end_index = _normalize_idx(end_index, len(node.get_size()))
        sizes = [
            self.find_symbol(node, "size", dim=i) or node.get_size()[i]
            for i in range(start_index, end_index + 1)
        ]
        if len(sizes) == 0:
            return str(default_value)

        sizes = [symbols(v) if isinstance(v, str) else v for v in sizes]
        val = sympy_product(sizes)
        return val

    def stride(self, node: IRNode, index: int, default_value: int = 0) -> str:
        """
        Hook called from template code to get the stride of an arg.
        Generates code which represents stride of a given node at index.
        If node is None, returns default_value.

        TODO: Will add needed args to pass it in if it is dynamic.
        """

        if node is None:
            return str(default_value)

        index = _normalize_idx(index, len(node.get_size()))
        if index < 0:
            return str(default_value)

        stride = node.get_stride()[index]
        if V.graph.sizevars.statically_known_leq(stride, 1):
            return str(stride)
        return self.find_symbol(node, "stride", dim=index) or str(stride)

    def batch_stride(self, node: IRNode, default_value: int = 0) -> str:
        """
        Hook called from template code to get the batch stride of an arg.
        Returns 0 if batch dim is not present.

        This method assumes that batch stride is the largest stride.
        """

        if node is None:
            return str(default_value)

        if len(node.get_size()) < 3:
            return str(default_value)

        batch_stride = node.get_stride()[0]
        if V.graph.sizevars.statically_known_leq(batch_stride, 1):
            return str(batch_stride)

        return "{}*{}".format(
            self.find_symbol(node, "size", dim=1) or node.get_size()[1],
            self.find_symbol(node, "size", dim=2) or node.get_size()[2],
        )

    def row_or_column_stride(self, node: IRNode, default_value: int = 0) -> str:
        """
        Hook called from template code to get the row or column stride of an arg.
        This is required by some CUTLASS 2.X APIs.
        If the node is in row_major, it returns stride[-2].
        If the node is in column_major, it returns stride[-1].

        TODO: Will add needed args to pass it in if it is dynamic.
        """

        if node is None or len(node.get_stride()) < 2:
            return str(default_value)

        stride0 = node.get_stride()[-1]
        stride1 = node.get_stride()[-2]
        if stride0 == 1:
            return cexpr(self.rename_indexing(stride1))
        elif stride1 == 1:
            return cexpr(self.rename_indexing(stride0))
        else:
            raise RuntimeError(
                f"At least 1 stride should be 1. Strides: {node.get_stride()=}"
            )

    def load(self, name: str, index: Expr, mode: Any = None) -> CSEVariable:
        """
        Mock load function for memory planning to optimize allocations properly.
        """
        return self.create_cse_var(name, bounds=ValueRanges.unknown())

    def store(self, name: str, index: Expr, value: Any, mode: Any = None) -> None:
        """
        Mock store function for memory planning to optimize allocations properly.
        """
        self.store_buffer_names.add(name)


class CUTLASSTemplateCaller(ChoiceCaller):
    """
    CUTLASSTemplateCaller

    This class represents a caller for CUTLASS template kernels. It is a subclass of ChoiceCaller.
    Attributes:
        name (str): The name of the caller.
        category (str): The category of the caller.
        bmreq (CUTLASSBenchmarkRequest): The benchmark request for the caller.
        template_buffer (CUTLASSTemplateBuffer): The template buffer for the caller.
    """

    def __init__(
        self,
        name: str,
        category: str,
        input_nodes: list[Buffer],
        layout: Layout,
        make_kernel_render: Callable[
            [CUTLASSTemplateBuffer, list[BaseSchedulerNode] | None],
            tuple[CUTLASSTemplateKernel, functools.partial[str]],
        ],
        bmreq: CUTLASSBenchmarkRequest,
        supports_epilogue_fusion: bool,
        template: "CUTLASSTemplate",  # type: ignore[name-defined]
        info_kwargs: dict[str, PrimitiveInfoType | list[PrimitiveInfoType]] | None,  # type: ignore[type-arg]
        description: str,
    ) -> None:
        super().__init__(name, input_nodes, layout, description)
        self.category = category
        self.make_kernel_render = make_kernel_render
        self.bmreq = bmreq
        self.supports_epilogue_fusion = supports_epilogue_fusion
        self.template = template
        self.info_kwargs = info_kwargs

    def precompile(self) -> None:
        if self.bmreq is None:
            raise AssertionError("expected self.bmreq to not be None")
        self.bmreq.precompile()

    def benchmark(self, *args, out) -> float:
        if self.bmreq is None:
            raise AssertionError("expected self.bmreq to not be None")
        if config.profile_bandwidth_with_do_bench_using_profiling:
            algo = self.bmreq.make_run_fn(*args, out=out)
            return do_bench_using_profiling(algo)
        return self.bmreq.benchmark(*args, out=out)

    def __str__(self) -> str:
        return f"CUTLASSTemplateCaller(source_file={self.bmreq.source_file})"

    def call_name(self) -> str:
        return f"cutlass_template_kernels.{self.name}"

    def kernel_hash_key(self) -> str:
        """
        Return kernel hash key that does not depend on swizzle.
        """
        return "-".join(
            [
                self.category,
                self.bmreq.hash_key,
            ]
        )

    def hash_key(self) -> str:
        """
        Return kernel hash key that does not depend on swizzle.
        """
        swizzle_str: str = (
            str(self.info_kwargs.get("swizzle"))
            if isinstance(self.info_kwargs, dict)
            else "None"
        )
        return "-".join(
            [
                self.category,
                self.bmreq.hash_key,
                swizzle_str,
            ]
        )

    def info_dict(self) -> dict[str, PrimitiveInfoType | list[PrimitiveInfoType]]:
        """
        Information returned here is logged to the autotune log file when that is enabled.

        In general, we should avoid calling this function as it is expensive to compute,
        and can add up very fast.
        """
        if self.info_kwargs is not None and "op" in self.info_kwargs:
            op: Any = self.info_kwargs["op"]
            return {
                "backend": "CUDA",
                "op_type": type(op).__name__,
                "op_conf_name": str(op.configuration_name()),
                "op_arch": str(op.arch),
                "tile_shape": str(op.tile_description.tile_shape),
                "epilogue_schedule": str(op.epilogue_schedule),
                "kernel_schedule": str(op.kernel_schedule),
                "element_accumulator": str(op.accumulator_type()),
                "op_name": str(op.procedural_name()),
                "instruction_shape": str(
                    op.tile_description.math_instruction.instruction_shape
                ),
                "swizzle": str(self.info_kwargs["swizzle"]),
            }
        else:
            return {"backend": "CUDA", "op_type": "unknown"}

    def output_node(self) -> TensorBox:
        self.bmreq.update_workspace_size()
        buffer = CUTLASSTemplateBuffer(
            layout=self.layout,
            inputs=self.input_nodes,
            make_kernel_render=self.make_kernel_render,
            workspace_size=self.bmreq.workspace_size,
            supports_epilogue_fusion=self.supports_epilogue_fusion,
            template=self.template,
        )
        # Pass KTC annotation to the buffer for encoding
        if "ktc" in self.annotations:
            buffer.annotations["ktc"] = self.annotations["ktc"]
        return TensorBox.create(buffer)
<|endoftext|># mypy: allow-untyped-defs
import functools
import json
from enum import Enum
from typing import Any, Optional

from torch._inductor.codegen.cutlass.utils import try_import_cutlass


class CUTLASSOperationSerializer:
    """Serializes and deserializes CUTLASS GEMM operations to/from JSON.

    Handles GemmOperation objects and their nested components (TileDescription, TensorDescription).
    """

    # not used, but keeping in case we want to generalize the serializer
    _SUPPORTED_CLASSES: list[str] = [
        "GemmOperation",
        "GemmKind",
        "TileDescription",
        "TensorDescription",
        "DataType",
        "EpilogueFunctor",
        "EpilogueFunctor3x",
        "SwizzlingFunctor",
        "KernelScheduleType",
        "EpilogueScheduleType",
        "TileSchedulerType",
    ]

    @classmethod
    def serialize(cls, operation: "GemmOperation") -> str:  # type: ignore[name-defined]  # noqa: F821
        """Serialize a GEMM operation to JSON string.

        Args:
            operation: GemmOperation object

        Returns:
            str: JSON string representation of the operation
        """
        if operation.__class__.__qualname__ != "GemmOperation":
            raise AssertionError(
                "Only GemmOperation objects are supported via the main API"
            )
        return json.dumps(cls._gemm_operation_to_json(operation))

    @classmethod
    def deserialize(cls, json_str: str) -> "GemmOperation":  # type: ignore[name-defined]  # noqa: F821
        """Deserialize JSON string to a GEMM operation.

        Args:
            json_str: JSON string of a GEMM operation

        Returns:
            GemmOperation: Reconstructed operation
        """
        json_dict = json.loads(json_str)
        return cls._json_to_gemm_operation(json_dict)

    @classmethod
    def _gemm_operation_to_json(cls, operation: "GemmOperation") -> dict[str, Any]:  # type: ignore[name-defined]  # noqa: F821
        """Convert GemmOperation to JSON-serializable dict.

        Args:
            operation: GemmOperation object

        Returns:
            dict: Dictionary representation
        """
        from cutlass_library.library import TensorDescription

        # Create the main dictionary with required and optional parameters
        result = {
            # Required parameters
            "gemm_kind": cls._enum_to_json(operation.gemm_kind),
            "arch": operation.arch,
            "tile_description": cls._tile_description_to_json(
                operation.tile_description
            ),
            "A": cls._tensor_description_to_json(operation.A),
            "B": cls._tensor_description_to_json(operation.B),
            "C": cls._tensor_description_to_json(operation.C),
            "element_epilogue": cls._enum_to_json(operation.element_epilogue),
            # Optional parameters
            "epilogue_functor": cls._enum_to_json(operation.epilogue_functor),
            "swizzling_functor": cls._enum_to_json(operation.swizzling_functor),
            "D": cls._tensor_description_to_json(operation.D) if operation.D else None,
            "kernel_schedule": cls._enum_to_json(operation.kernel_schedule),
            "epilogue_schedule": cls._enum_to_json(operation.epilogue_schedule),
            "tile_scheduler": cls._enum_to_json(operation.tile_scheduler),
        }

        # Process optional attributes
        optional_attrs = [
            "mixed_input_mode",
            "mixed_input_shuffle",
            "ScaleFactorA",
            "ScaleFactorB",
            "ScaleFactorD",
            "ScaleFactorMVecSize",
            "ScaleFactorNVecSize",
            "ScaleFactorKVecSize",
            "ScaleFactorVectorSize",
            "is_3x",
        ]

        for attr in optional_attrs:
            if not hasattr(operation, attr):
                continue

            value = getattr(operation, attr)

            if isinstance(value, TensorDescription):
                result[attr] = cls._tensor_description_to_json(value)
            elif isinstance(value, Enum):
                result[attr] = cls._enum_to_json(value)
            else:
                result[attr] = value

        return result

    @classmethod
    def _json_to_gemm_operation(cls, json_dict: dict[str, Any]) -> "GemmOperation":  # type: ignore[name-defined]  # noqa: F821
        """Convert JSON dict to GemmOperation object.

        Args:
            json_dict: Dictionary representation

        Returns:
            GemmOperation: Reconstructed object
        """
        from cutlass_library import DataType
        from cutlass_library.gemm_operation import GemmKind, GemmOperation
        from cutlass_library.library import (
            EpilogueFunctor,
            EpilogueFunctor3x,
            EpilogueScheduleType,
            KernelScheduleType,
            MixedInputMode,
            SwizzlingFunctor,
            TileSchedulerType,
        )

        # Extract constructor parameters from the JSON dictionary
        gemm_kind = cls._json_to_enum(json_dict["gemm_kind"], GemmKind)
        arch = json_dict["arch"]
        tile_description = cls._json_to_tile_description(json_dict["tile_description"])
        A = cls._json_to_tensor_description(json_dict.get("A"), "A")
        B = cls._json_to_tensor_description(json_dict.get("B"), "B")
        C = cls._json_to_tensor_description(json_dict.get("C"), "C")
        element_epilogue = cls._json_to_enum(json_dict["element_epilogue"], DataType)

        # Get optional parameters with defaults
        epilogue_functor = cls._json_to_enum(
            json_dict.get("epilogue_functor"),
            EpilogueFunctor3x if json_dict.get("is_3x") else EpilogueFunctor,
        )
        swizzling_functor = cls._json_to_enum(
            json_dict.get("swizzling_functor"), SwizzlingFunctor
        )
        D = cls._json_to_tensor_description(json_dict.get("D"), "D")
        kernel_schedule = cls._json_to_enum(
            json_dict.get("kernel_schedule"), KernelScheduleType
        )
        epilogue_schedule = cls._json_to_enum(
            json_dict.get("epilogue_schedule"), EpilogueScheduleType
        )
        tile_scheduler = cls._json_to_enum(
            json_dict.get("tile_scheduler"), TileSchedulerType
        )

        mixed_input_mode = cls._json_to_enum(
            json_dict.get("mixed_input_mode"), MixedInputMode
        )
        mixed_input_shuffle = json_dict.get("mixed_input_shuffle", False)

        # Scale factors
        ScaleFactorA = cls._json_to_enum(json_dict.get("ScaleFactorA"), DataType)
        ScaleFactorB = cls._json_to_enum(json_dict.get("ScaleFactorB"), DataType)

        ScaleFactorD = None
        if "ScaleFactorD" in json_dict and "ScaleFactorVectorSize" in json_dict:
            ScaleFactorD = {
                "tensor": cls._json_to_tensor_description(
                    json_dict.get("ScaleFactorD"), "ScaleFactorD"
                ),
                "vector_size": json_dict.get("ScaleFactorVectorSize"),
            }

        ScaleFactorMVecSize = json_dict.get("ScaleFactorMVecSize")
        ScaleFactorNVecSize = json_dict.get("ScaleFactorNVecSize")
        ScaleFactorKVecSize = json_dict.get("ScaleFactorKVecSize")

        # Create the GemmOperation with the extracted parameters
        operation = GemmOperation(
            gemm_kind=gemm_kind,
            arch=arch,
            tile_description=tile_description,
            A=A,
            B=B,
            C=C,
            element_epilogue=element_epilogue,
            epilogue_functor=epilogue_functor,
            swizzling_functor=swizzling_functor,
            D=D,
            kernel_schedule=kernel_schedule,
            epilogue_schedule=epilogue_schedule,
            tile_scheduler=tile_scheduler,
            mixed_input_mode=mixed_input_mode,
            mixed_input_shuffle=mixed_input_shuffle,
            ScaleFactorA=ScaleFactorA,
            ScaleFactorB=ScaleFactorB,
            ScaleFactorD=ScaleFactorD,
            ScaleFactorMVecSize=ScaleFactorMVecSize,
            ScaleFactorNVecSize=ScaleFactorNVecSize,
            ScaleFactorKVecSize=ScaleFactorKVecSize,
        )

        return operation

    @classmethod
    @functools.lru_cache(None)
    def _tile_description_to_json(cls, tile_desc: "TileDescription") -> str:  # type: ignore[name-defined]  # noqa: F821
        """
        Convert TileDescription to JSON string.

        Args:
            tile_desc: TileDescription object

        Returns:
            str: JSON string representation
        """

        # Create the main dictionary with field names matching TileDescription constructor parameters
        result = {
            "threadblock_shape": tile_desc.threadblock_shape,
            "stages": tile_desc.stages,
            "warp_count": tile_desc.warp_count,
            "math_instruction": cls._math_instruction_to_json(
                tile_desc.math_instruction
            ),
            "min_compute": tile_desc.minimum_compute_capability,  # Store as min_compute for constructor
            "max_compute": tile_desc.maximum_compute_capability,  # Store as max_compute for constructor
            "cluster_shape": tile_desc.cluster_shape,
            "explicit_vector_sizes": tile_desc.explicit_vector_sizes,
        }

        # Add tile_shape if it exists and differs from threadblock_shape
        if (
            hasattr(tile_desc, "tile_shape")
            and tile_desc.tile_shape != tile_desc.threadblock_shape
        ):
            result["tile_shape"] = tile_desc.tile_shape

        return json.dumps(result)

    @classmethod
    @functools.lru_cache(None)
    def _json_to_tile_description(
        cls, json_dict: str | None
    ) -> Optional["TileDescription"]:  # type: ignore[name-defined]  # noqa: F821
        """
        Convert JSON dict to TileDescription object.

        Args:
            json_dict: Dictionary representation

        Returns:
            TileDescription: Reconstructed object
        """
        if json_dict is None:
            return None

        tile_dict = json.loads(json_dict)

        from cutlass_library.library import TileDescription

        math_instruction = cls._json_to_math_instruction(tile_dict["math_instruction"])

        # Get compute capability values, checking both naming conventions
        min_compute = tile_dict.get(
            "min_compute", tile_dict.get("minimum_compute_capability")
        )
        max_compute = tile_dict.get(
            "max_compute", tile_dict.get("maximum_compute_capability")
        )

        # Get cluster shape with default value
        cluster_shape = tile_dict.get("cluster_shape", [1, 1, 1])

        # Create the TileDescription object
        tile_desc = TileDescription(
            threadblock_shape=tile_dict["threadblock_shape"],
            stages=tile_dict["stages"],
            warp_count=tile_dict["warp_count"],
            math_instruction=math_instruction,
            min_compute=min_compute,
            max_compute=max_compute,
            cluster_shape=cluster_shape,
            explicit_vector_sizes=tile_dict.get("explicit_vector_sizes"),
        )

        # Set tile_shape if it exists and differs from threadblock_shape
        if (
            "tile_shape" in tile_dict
            and tile_dict["tile_shape"] != tile_dict["threadblock_shape"]
        ):
            tile_desc.tile_shape = tile_dict["tile_shape"]

        return tile_desc

    @classmethod
    @functools.lru_cache(None)
    def _math_instruction_to_json(
        cls,
        math_instruction: Optional["MathInstruction"],  # type: ignore[name-defined]  # noqa: F821
    ) -> str | None:
        """Convert MathInstruction to JSON string.

        Args:
            math_instruction: MathInstruction object

        Returns:
            Optional[str]: JSON string representation or None
        """
        if math_instruction is None:
            return None

        result = {
            "instruction_shape": math_instruction.instruction_shape,
            "element_a": cls._enum_to_json(math_instruction.element_a),
            "element_b": cls._enum_to_json(math_instruction.element_b),
            "element_accumulator": cls._enum_to_json(
                math_instruction.element_accumulator
            ),
            "opcode_class": cls._enum_to_json(math_instruction.opcode_class),
            "math_operation": cls._enum_to_json(math_instruction.math_operation),
            "element_scale_factor": cls._enum_to_json(
                math_instruction.element_scale_factor
            ),
        }

        return json.dumps(result)

    @classmethod
    @functools.lru_cache(None)
    def _json_to_math_instruction(
        cls, json_dict: str | None
    ) -> Optional["MathInstruction"]:  # type: ignore[name-defined]  # noqa: F821
        """Convert JSON string to MathInstruction object.

        Args:
            json_dict: JSON string representation

        Returns:
            Optional[MathInstruction]: Reconstructed object or None
        """
        if json_dict is None:
            return None

        from cutlass_library import DataType
        from cutlass_library.library import MathInstruction, MathOperation, OpcodeClass

        mi_dict = json.loads(json_dict)

        # Convert string enum names back to enum values
        element_a = cls._json_to_enum(mi_dict["element_a"], DataType)
        element_b = cls._json_to_enum(mi_dict["element_b"], DataType)
        element_acc = cls._json_to_enum(mi_dict["element_accumulator"], DataType)

        # Get the opcode_class enum
        opcode_class = cls._json_to_enum(mi_dict["opcode_class"], OpcodeClass)

        # Get the math_operation enum
        math_op = cls._json_to_enum(mi_dict["math_operation"], MathOperation)

        # Create the MathInstruction object
        math_instruction_obj = MathInstruction(
            instruction_shape=mi_dict["instruction_shape"],
            element_a=element_a,
            element_b=element_b,
            element_accumulator=element_acc,
            opcode_class=opcode_class,
            math_operation=math_op,
        )

        # Add element_scale_factor if it exists
        if (
            "element_scale_factor" in mi_dict
            and mi_dict["element_scale_factor"] is not None
        ):
            math_instruction_obj.element_scale_factor = cls._json_to_enum(
                mi_dict["element_scale_factor"], DataType
            )

        return math_instruction_obj

    @classmethod
    @functools.lru_cache(None)
    def _tensor_description_to_json(
        cls,
        tensor_desc: Optional["TensorDescription"],  # type: ignore[name-defined]  # noqa: F821
    ) -> str | None:
        """Convert TensorDescription to JSON string.

        Args:
            tensor_desc: TensorDescription object

        Returns:
            Optional[str]: JSON string representation or None
        """
        if tensor_desc is None:
            return None

        result = {
            "element": cls._enum_to_json(tensor_desc.element),
            "layout": cls._enum_to_json(tensor_desc.layout),
            "alignment": tensor_desc.alignment,
            "complex_transform": cls._enum_to_json(tensor_desc.complex_transform),
        }

        return json.dumps(result)

    @classmethod
    @functools.lru_cache(None)
    def _json_to_tensor_description(
        cls,
        json_dict: str | None,
        tensor_name: str | None = None,
    ) -> Optional["TensorDescription"]:  # type: ignore[name-defined]  # noqa: F821
        """Convert JSON string to TensorDescription object.

        Args:
            json_dict: JSON string representation
            tensor_name: Name of the tensor to avoid cache in the same op

        Returns:
            Optional[TensorDescription]: Reconstructed object or None
        """
        if json_dict is None:
            return None

        tensor_dict = json.loads(json_dict)

        from cutlass_library import DataType
        from cutlass_library.library import (
            ComplexTransform,
            LayoutType,
            TensorDescription,
        )

        element = cls._json_to_enum(tensor_dict["element"], DataType)
        layout = cls._json_to_enum(tensor_dict["layout"], LayoutType)
        alignment = tensor_dict["alignment"]
        complex_transform = cls._json_to_enum(
            tensor_dict["complex_transform"], ComplexTransform
        )

        return TensorDescription(element, layout, alignment, complex_transform)

    @classmethod
    @functools.lru_cache(None)
    def _enum_to_json(cls, enum_value: Enum | None) -> str | None:
        """Convert enum value to JSON string.

        Args:
            enum_value: Enum value

        Returns:
            Optional[str]: JSON string representation or None
        """
        if enum_value is None:
            return None

        result = {
            "type": enum_value.__class__.__name__,
            "name": enum_value.name,
        }

        return json.dumps(result)

    @classmethod
    @functools.lru_cache(None)
    def _json_to_enum(cls, json_dict: str | None, enum_class: Any) -> Enum | None:
        """Convert JSON string to enum value.

        Format: {name: "EnumName", value: 1}

        Args:
            json_dict: JSON string representation
            enum_class: Target enum class

        Returns:
            Optional[Enum]: Reconstructed enum value or None
        """
        if json_dict is None:
            return None

        enum_dict = json.loads(json_dict)

        return enum_class[enum_dict["name"]]


@functools.lru_cache(1)
def get_cutlass_operation_serializer() -> CUTLASSOperationSerializer | None:
    if not try_import_cutlass():
        return None
    return CUTLASSOperationSerializer()
<|endoftext|>from collections.abc import Callable
from typing import Any

from sympy import Expr

from torch._inductor.ir import (
    ComputedBuffer,
    InputBuffer,
    is_contiguous_strides_for_shape,
)
from torch.utils._ordered_set import OrderedSet

from ..utils import cutlass_arch, torch_dtype_to_cutlass_type, try_import_cutlass


EpilogueFunctor = Any  # EpilogueFunctor local class defined in _trace
Buffer = ComputedBuffer | InputBuffer
CutlassTupleType = Any  # cutlass.backend.c_types.tuple_factory_.<locals>.TupleType
CutlassVisitorType = Any  # cutlass.backend.c_types.visitor_factory.<locals>.VisitorType
CutlassArgType = (
    Any  # Can be a CutlassTupleType, CutlassVisitorType, EmptyByte, or ctype.c_void_p
)


if try_import_cutlass():
    import ast
    import ctypes
    import textwrap

    from cutlass_cppgen.backend.c_types import (  # type: ignore[import-not-found]
        EmptyByte,
    )
    from cutlass_cppgen.backend.epilogue import (  # type: ignore[import-not-found]
        dtype2ctype,
    )
    from cutlass_cppgen.backend.evt import (  # type: ignore[import-not-found]
        backend as evt_backend,
        EpilogueFunctorVisitor,
    )
    from cutlass_cppgen.backend.evt.backend.emitter_base import (  # type: ignore[import-not-found]
        FusionCallbacks,
    )
    from cutlass_cppgen.backend.evt.backend.sm100_emitter import (  # type: ignore[import-not-found]
        Sm100CollectiveEpilogue,
    )
    from cutlass_cppgen.backend.evt.frontend import (  # type: ignore[import-not-found]
        PythonASTFrontend,
    )
    from cutlass_cppgen.backend.evt.ir.tensor import (  # type: ignore[import-not-found]
        Tensor as CutlassTensor,
    )
    from cutlass_library import (
        DataType,
        EpilogueScheduleType,
        LayoutType,
        TileDescription,
    )

    from torch._inductor.utils import IndentedBuffer

    _CUTLASS_C_DTYPES = OrderedSet(dtype2ctype.values())  # type: ignore[var-annotated]

    class EVTArgRenames:
        """Handles mapping buffer names to variable names in the cpp kernel signature and body"""

        def __init__(self) -> None:
            self.buf_renames: dict[str, str] = {}

        def new_name(self, name: str) -> str:
            if name in self.buf_renames:
                return self.buf_renames[name]
            else:
                new_name = f"ptr_{len(self.buf_renames)}"
                self.buf_renames[name] = new_name
                return new_name

        def get(self, name: str) -> str:
            return self.buf_renames.get(name, name)

    def create_example_tensors(
        var_name_to_buffer_name: dict[str, str],
        name_to_buffer: dict[str, Buffer],
        size_hint_fn: Callable[[Expr | int], int],
    ) -> dict[str, CutlassTensor]:
        def cutlass_tensor_from_buffer(
            buffer: Buffer,
        ) -> CutlassTensor:
            shape = buffer.get_layout().size
            stride = buffer.get_layout().stride
            shape = tuple(size_hint_fn(x) for x in shape)
            stride = tuple(size_hint_fn(x) for x in stride)

            is_row_major = is_contiguous_strides_for_shape(stride, shape)
            is_column_major = is_contiguous_strides_for_shape(stride[::-1], shape[::-1])

            if not is_row_major and not is_column_major:
                raise RuntimeError(
                    f"Cannot create example tensor for {buffer.get_name()} with \
non-contiguous layout, received stride: {stride} and shape: {shape}"
                )

            return CutlassTensor(
                shape=shape,
                layout_tag=(
                    LayoutType.RowMajor if is_row_major else LayoutType.ColumnMajor
                ),
                element=torch_dtype_to_cutlass_type(buffer.get_layout().dtype),
            )

        return {
            key: cutlass_tensor_from_buffer(name_to_buffer[name])
            for key, name in var_name_to_buffer_name.items()
        }

    def trace(
        fn_src: str,
        example_tensors: dict[str, CutlassTensor],
        accum_type: DataType,
        output_type: DataType,
        tile_description: TileDescription,
        epilogue_schedule: EpilogueScheduleType,
        name_to_buffer: dict[str, Buffer],
        size_hint_fn: Callable[[Expr | int], int],
        kernel_schedule: Any | None = None,
        device_type: str = "cuda",
        **kwargs: dict[str, Any],
    ) -> tuple[str, str, str, EVTArgRenames]:
        arch = int(cutlass_arch(device_type))
        if device_type == "cuda" and arch < 90:
            raise AssertionError("For CUDA, only SM90+ is supported for EVT")
        epilogue_functor = _trace(fn_src, example_tensors, arch, **kwargs)
        visitor = EpilogueFunctorVisitor(arch, epilogue_functor)
        fusion_callbacks = FusionCallbacks(visitor.graph, arch, emit_CD=False)
        arch_prefix = "xe" if device_type == "xpu" else "sm"

        if device_type == "xpu" or arch < 100:
            try:
                evt_emitter = getattr(evt_backend, f"{arch_prefix}{arch}_emitter")
                CollectiveEpilogue = evt_emitter.CollectiveEpilogue
            except AttributeError as e:
                raise NotImplementedError(
                    f"EVT backend is not supported on Arch {arch_prefix}{arch}."
                ) from e

            collective_epilogue = CollectiveEpilogue(
                tile_description,
                epilogue_schedule,
                accum_type,
                output_type,
                fusion_callbacks,
            )
        else:
            collective_epilogue = Sm100CollectiveEpilogue(
                tile_description=tile_description,
                kernel_schedule=kernel_schedule,
                epilogue_schedule=epilogue_schedule,
                element_accumulator=accum_type,
                element_d=output_type,
                fusion_callbacks=fusion_callbacks,
            )
        evt_name, evt_code = collective_epilogue.emit()
        evt_args, arg_renames = _render_argument_type(
            epilogue_functor, name_to_buffer, size_hint_fn
        )
        return evt_name, evt_args, evt_code, arg_renames

    # Based off of
    # https://github.com/NVIDIA/cutlass/blob/df18f5e4f5de76bed8be1de8e4c245f2f5ec3020/python/cutlass/epilogue/epilogue.py#L117
    # This is modified to enable directly passing the source code of the epilogue vs getting it from a bona-fide python function
    # The reason for this is that inspect.getsource does not work with functions defined at runtime via exec/eval
    def _trace(
        fn_src: str,
        example_tensors: dict[str, CutlassTensor],
        cc: int,
        **kwargs: Any,
    ) -> EpilogueFunctor:
        class EpilogueFunctor(PythonASTFrontend):
            def __init__(self, cc: int, **kwargs: Any):
                self.source = textwrap.dedent(fn_src)
                super().__init__(cc, **kwargs)

            def parse(
                self,
                example_inputs: dict[str, CutlassTensor],
            ) -> None:
                self.example_inputs = example_inputs
                self.ast = ast.parse(self.source)
                # pyrefly: ignore [missing-attribute]
                self.visit(self.ast)

        epilogue_functor = EpilogueFunctor(cc=cc, **kwargs)
        epilogue_functor.trace(example_tensors)
        return epilogue_functor

    def _render_argument_type(
        epilogue_functor: EpilogueFunctor,
        name_to_buffer: dict[str, Buffer],
        size_hint_fn: Callable[[Expr | int], int],
    ) -> tuple[str, EVTArgRenames]:
        epilogue_thread_type = epilogue_functor.epilogue_thread_type
        arg_renames = EVTArgRenames()

        # Fragile, but this is the only way to guarantee t is expected type because t is a local class
        def is_nested_visitor_type(t: type) -> bool:
            return (
                ".".join([t.__module__, t.__qualname__])
                == "cutlass_cppgen.backend.c_types.visitor_factory.<locals>.VisitorType"
            )

        buffer = IndentedBuffer()
        with buffer.set_tabwidth(2):

            def get_constant_value(name: str) -> float | int | None:
                dag_ir = getattr(epilogue_functor, "dag_ir", None)
                if dag_ir is not None:
                    try:
                        node_meta = dag_ir.get_node_meta(name)
                    except Exception:
                        node_meta = None
                    if node_meta is not None:
                        tensor_meta = getattr(node_meta, "tensor", None)
                        if isinstance(tensor_meta, dict) and tensor_meta.get(
                            "is_constant"
                        ):
                            return tensor_meta.get("tensor")

                # Immediate constants are encoded as "imm_<value>_k<idx>" where
                # the first underscore in <value> stands in for the decimal point
                # (e.g. "imm_0_044715_k1" -> 0.044715). dag_ir meta is the primary
                # source; this name parsing is a best-effort fallback.
                if not name.startswith("imm_"):
                    return None

                value_part, _, _ = name[len("imm_") :].rpartition("_k")
                if not value_part:
                    return None
                try:
                    return float(value_part.replace("_", ".", 1))
                except ValueError:
                    return None

            def render_argument_type(
                name: str,
                t: CutlassArgType,
                constant_value: float | int | None = None,
            ) -> None:
                if constant_value is None and name not in name_to_buffer:
                    # An imm_* node is an immediate constant whose value we
                    # failed to recover (dag_ir meta missing AND name parsing
                    # failed, e.g. an unsupported encoding). Emitting an empty
                    # struct here would silently drop the constant and produce a
                    # numerically wrong kernel, so fail loudly instead.
                    if name.startswith("imm_"):
                        raise RuntimeError(
                            f"Could not recover the value of immediate constant "
                            f"node {name!r} for the CUTLASS EVT epilogue. "
                            f"Refusing to emit an empty argument that would "
                            f"silently drop the constant."
                        )
                    # Other non-buffer nodes (e.g. autogen_identity_*/compute_*
                    # placeholders) legitimately render as an empty struct,
                    # matching CUTLASS reference EVT kernels.
                    buffer.writeline(f"{{}}, /* {name} */")
                    return
                if issubclass(t, ctypes.c_byte):
                    buffer.writeline(f"{{}}, /* {name} */")
                else:
                    fields = [
                        (
                            fname,
                            _get_arg_from_node(
                                ty,
                                name_to_buffer.get(name),
                                size_hint_fn,
                                arg_renames,
                                constant_value,
                            ),
                        )
                        for fname, ty in t._fields_
                    ]
                    field_strs = [
                        f"/* {fname} */ {str(field)}" for fname, field in fields
                    ]
                    buffer.writeline(f"{{{', '.join(field_strs)}}}, /* {name} */")

            def render_thread_type(name: str, t: CutlassArgType) -> None:
                if is_nested_visitor_type(t):
                    buffer.writeline(f"{{ /* {name} */")
                    with buffer.indent():
                        for name, inner_t in t._fields_:
                            render_thread_type(name, inner_t)
                    buffer.writeline("},")
                else:
                    render_argument_type(name, t, get_constant_value(name))

            # unroll the recursion once to address special case formatting
            # namely, no ending comma and no indentation for the outermost thread type
            buffer.writeline("{ /* thread */")
            with buffer.indent(3):
                if is_nested_visitor_type(epilogue_thread_type):
                    with buffer.indent():
                        for name, inner_t in epilogue_thread_type._fields_:
                            render_thread_type(name, inner_t)
                else:
                    render_argument_type("thread", epilogue_thread_type)
                buffer.writeline("}")

        return buffer.getvalue(), arg_renames

    def _get_arg_from_node(
        arg_ty: type,
        node: Buffer | None,
        size_hint_fn: Callable[[Expr | int], int],
        arg_renames: EVTArgRenames,
        constant_value: float | int | None = None,
    ) -> str:
        from ..template import CUTLASSTemplate

        # Today, arguments are either a pointer to the
        # node's memory, a stride tuple, the datatype
        # Once again, need to check for local class type for stride tuple
        if (
            str(arg_ty)
            == "<class 'cutlass_cppgen.backend.c_types.tuple_factory_.<locals>.TupleType'>"
        ):
            if constant_value is not None:
                raise NotImplementedError(
                    "Immediate constants do not use tuple argument types"
                )
            DEFAULT_STRIDE_LEN = 3
            if node is None:
                raise AssertionError("node must not be None for tuple argument types")
            if len(node.get_layout().stride) > DEFAULT_STRIDE_LEN:
                raise AssertionError(
                    f"expected stride length <= {DEFAULT_STRIDE_LEN}, "
                    f"got {len(node.get_layout().stride)}"
                )
            stride = [size_hint_fn(x) for x in node.get_layout().stride]
            for _ in range(DEFAULT_STRIDE_LEN - len(stride)):
                stride.append(0)

            def render_stride(x: int) -> str:
                # Handle EBO for 0 and 1
                if x == 0:
                    return "_0{}"
                elif x == 1:
                    return "_1{}"
                else:
                    return str(x)

            return f"{{{', '.join([render_stride(x) for x in stride])}}}"

        elif issubclass(arg_ty, ctypes.c_void_p):
            if constant_value is not None:
                return "nullptr"
            if node is None:
                raise AssertionError("node must not be None for pointer argument")
            name = arg_renames.new_name(node.get_name())
            return f"({CUTLASSTemplate._DTYPE_TO_CUTLASS[node.get_layout().dtype]}*) ({name} + {name}_offset)"
        elif (
            arg_ty in _CUTLASS_C_DTYPES
        ):  # Assumption: this is the element dtype, this holds for all cutlass ir nodes currently
            if constant_value is not None:
                return str(constant_value)
            if node is None:
                raise AssertionError("node must not be None for dtype argument")
            return f"{CUTLASSTemplate._DTYPE_TO_CUTLASS[node.get_layout().dtype]}(0)"
        elif issubclass(arg_ty, EmptyByte):
            return "{}"

        raise NotImplementedError(f"Unsupported arg type: {arg_ty}")
<|endoftext|># mypy: ignore-errors
from ..utils import try_import_cutlass


# copied / modified from original at
# https://github.com/NVIDIA/cutlass/blob/8783c41851cd3582490e04e69e0cd756a8c1db7f/tools/library/scripts/gemm_operation.py#L658

if try_import_cutlass():
    import enum

    from cutlass_library.gemm_operation import *  # noqa: F403
    from cutlass_library.library import *  # noqa: F403

    _LOGGER = logging.getLogger(__name__)

    class EmitGemmUniversal3xInstanceWithEVT:
        """Responsible for emitting a CUTLASS 3.x template definition"""

        def __init__(self, operation_suffix="", evt_name=None, device_type="cuda"):
            self.operation_suffix = operation_suffix
            self.includes = [
                "cutlass/cutlass.h",
                "cutlass/gemm/gemm.h",
                "cutlass/numeric_types.h",
                "cutlass/gemm/kernel/gemm_universal.hpp",
                "cutlass/gemm/collective/collective_builder.hpp",
                "cutlass/epilogue/collective/collective_builder.hpp",
            ]
            self.builtin_epilogue_functor_template = """${epilogue_functor}<
            ${element_d},
            ${element_epilogue},
            ${element_c},
            ${element_epilogue}
            >"""
            self.evt_name = evt_name
            self.device_type = device_type
            self.gemm_template = """
using ${operation_name}_epilogue =
typename cutlass::epilogue::collective::CollectiveBuilder<
    ${arch}, ${opcode_class_epi},
    cute::Shape<cute::_${tile_shape_m}, cute::_${tile_shape_n}, cute::_${tile_shape_k}>,
    cute::Shape<${cluster_shape_m}, ${cluster_shape_n}, ${cluster_shape_k}>,
    ${epi_tile_mn},
    ${element_accumulator}, ${element_epilogue},
    ${element_c}, ${layout_c}, ${align_c},
    ${element_d}, ${layout_d}, ${align_d},
    ${epilogue_schedule},
    ${epilogue_functor}
>::CollectiveOp;

${mixed_dtype_prepare_code}

using ${operation_name}_mainloop =
typename cutlass::gemm::collective::CollectiveBuilder<
    ${arch}, ${opcode_class_main},
    ${element_a}, ${layout_a}, ${align_a},
    ${element_b}, ${layout_b}, ${align_b},
    ${element_accumulator},
    cute::Shape<cute::_${tile_shape_m}, cute::_${tile_shape_n}, cute::_${tile_shape_k}>,
    cute::Shape<${cluster_shape_m}, ${cluster_shape_n}, ${cluster_shape_k}>,
    ${stages},
    ${kernel_schedule}
>::CollectiveOp;

// Gemm operator ${operation_name}
using ${operation_name}_base = cutlass::gemm::kernel::GemmUniversal<
    ${problem_shape},
    ${operation_name}_mainloop,
    ${operation_name}_epilogue,
    ${tile_scheduler}>;

// Define named type
struct ${operation_name} :
public ${operation_name}_base { };

        """

        #
        def instance_template(self):
            return """
${compile_guard_start}
{
    using GemmKernel = cutlass::gemm::device::GemmUniversalAdapter<${operation_name}>;
    manifest.append(
    new ${gemm_kind}<GemmKernel>("${operation_name}"));
}
${compile_guard_end}
        """

        def emit_block_scale_epilogue_functor(self, operation):
            block_scaled_template = """
            ${epilogue_functor}<
                ${epi_vs},
                ${element_d},
                ${element_accumulator},
                ${element_sfd},
                ${layout_sfd},
                ${element_c},
                ${element_scalar}
            >
            """
            block_scaled_values = {
                "epi_vs": str(operation.ScaleFactorVectorSize),
                "element_d": str(DataTypeTag[operation.D.element]),
                "element_sfd": str(DataTypeTag[operation.ScaleFactorD.element]),
                "layout_sfd": LayoutTag[operation.ScaleFactorD.layout],
                "epilogue_functor": EpilogueFunctor3xTag[
                    EpilogueFunctor3x.LinearCombinationBlockScaleFactor
                ],
                "element_accumulator": str(DataTypeTag[operation.accumulator_type()]),
                "element_scalar": str(DataTypeTag[operation.accumulator_type()]),
                "element_c": str(DataTypeTag[operation.C.element]),
            }
            return SubstituteTemplate(block_scaled_template, block_scaled_values)

        @staticmethod
        def pointerize_if_grouped(operation, layout):
            return layout if not is_grouped(operation.gemm_kind) else layout + "* "

        @staticmethod
        def problem_shape(operation):
            gemm_shape_type = "cute::Shape<int,int,int,int>"
            grouped_gemm_shape_type = "cute::Shape<int,int,int>"
            grouped_gemm_shape_type = (
                "cutlass::gemm::GroupProblemShape<" + grouped_gemm_shape_type + ">"
            )

            return (
                gemm_shape_type
                if not is_grouped(operation.gemm_kind)
                else grouped_gemm_shape_type
            )

        def emit(self, operation):
            """Given a gem operation, emits a template definition of the operation"""

            opcode_class_main = operation.tile_description.math_instruction.opcode_class
            opcode_class_epi = opcode_class_main

            tile_shape = operation.tile_description.tile_shape
            instruction_shape = (
                operation.tile_description.math_instruction.instruction_shape
            )
            cluster_m = operation.tile_description.cluster_shape[0]
            cluster_n = operation.tile_description.cluster_shape[1]

            tile_shape_m, tile_shape_n, tile_shape_k = tile_shape

            # account for static/dynamic cluster shapes
            cta_m = tile_shape[0] // cluster_m if cluster_m > 0 else tile_shape[0]
            cta_n = tile_shape[1] // cluster_n if cluster_n > 0 else tile_shape[1]

            # Shape passed to epilogue builder
            is_sm100_kernel = operation.arch == 100
            if is_sm100_kernel:
                cta_m_per_mma_instruction = (
                    2 if "2sm" in operation.procedural_name() else 1
                )
                if cluster_m <= 0:
                    cta_m = cta_m // cta_m_per_mma_instruction

                if opcode_class_main in [
                    OpcodeClass.TensorOp,
                    OpcodeClass.BlockScaledTensorOp,
                ]:
                    tile_shape_m = instruction_shape[0]
                    tile_shape_n = instruction_shape[1]

            # stage count set to zero indicates builder automatic stage selection
            if operation.tile_description.stages > 0:
                stage_count_string = f"cutlass::gemm::collective::StageCount<\
{str(operation.tile_description.stages)}>"
            else:
                stage_count_string = (
                    f"cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(\
sizeof(typename {str(operation.procedural_name())}_epilogue::SharedStorage))>"
                )
            if self.device_type == "xpu":
                stage_count_string = "cutlass::gemm::collective::StageCountAuto"

            epi_tile_mn = "cutlass::epilogue::collective::EpilogueTileAuto"

            (
                instance_layout_A,
                instance_layout_B,
                instance_layout_C,
                instance_layout_D,
            ) = (
                operation.A.layout,
                operation.B.layout,
                operation.C.layout,
                operation.D.layout,
            )

            # 3.0 profiler integration only supports trivial epilogues for now
            epilogue_vector_length = 1

            # Support built-in epilogue functors or user-defined functions
            if isinstance(operation.epilogue_functor, enum.Enum):
                values = {
                    "element_epilogue": str(DataTypeTag[operation.element_epilogue]),
                    "epilogue_functor": EpilogueFunctor3xTag[
                        operation.epilogue_functor
                    ],
                }
                epilogue_functor = SubstituteTemplate(
                    self.builtin_epilogue_functor_template, values
                )

                if (
                    is_block_scaled(operation.gemm_kind)
                    and operation.ScaleFactorD.element != DataType.void
                ):
                    epilogue_functor = self.emit_block_scale_epilogue_functor(operation)
            else:
                epilogue_functor = self.epilogue_functor.emit_declaration()

            if (
                is_block_scaled(operation.gemm_kind)
                and operation.ScaleFactorD.element != DataType.void
            ):
                epilogue_functor = self.emit_block_scale_epilogue_functor(operation)

            #
            # Cutlass3x complex kernels' ElementA(B) is a tuple in collective mainloop builder,
            # e.g. cute::tuple<Element, Transform>, Transform : cute::identity / cute::conjugate.
            element_a = (
                DataTypeTag[operation.A.element]
                if not operation.is_complex()
                else f"cute::tuple<{str(DataTypeTag[operation.A.element])},\
{str(ComplexTransformTag3x[operation.A.complex_transform])}>"
            )
            element_b = (
                DataTypeTag[operation.B.element]
                if not operation.is_complex()
                else f"cute::tuple<{str(DataTypeTag[operation.B.element])},\
{str(ComplexTransformTag3x[operation.B.complex_transform])}>"
            )
            epilogue_schedule_type = EpilogueScheduleTag[operation.epilogue_schedule]

            if opcode_class_main == OpcodeClass.BlockScaledTensorOp:
                is_no_smem_epilogue = operation.epilogue_schedule in [
                    EpilogueScheduleType.NoSmemWarpSpecialized1Sm,
                    EpilogueScheduleType.NoSmemWarpSpecialized2Sm,
                ]
                grouped = is_grouped(operation.gemm_kind)
                if cta_n == 256 and operation.kernel_schedule == to_grouped_schedule(
                    KernelScheduleType.Nvf4TmaWarpSpecialized1SmSm100, grouped
                ):
                    epi_tile_mn = "cute::Shape<cute::_128,cute::_64>"
                    if not is_no_smem_epilogue:
                        epilogue_schedule_type = EpilogueScheduleTag[
                            to_grouped_schedule(
                                EpilogueScheduleType.TmaWarpSpecialized1Sm, grouped
                            )
                        ]
                if cta_n == 256 and operation.kernel_schedule == to_grouped_schedule(
                    KernelScheduleType.Nvf4TmaWarpSpecialized2SmSm100, grouped
                ):
                    epi_tile_mn = "cute::Shape<cute::_128,cute::_64>"
                    if not is_no_smem_epilogue:
                        epilogue_schedule_type = EpilogueScheduleTag[
                            to_grouped_schedule(
                                EpilogueScheduleType.TmaWarpSpecialized2Sm, grouped
                            )
                        ]
                element_a = f"cute::tuple<{str(element_a)},{str(DataTypeTag[operation.ScaleFactorA])}>"
                element_b = f"cute::tuple<{str(element_b)},{str(DataTypeTag[operation.ScaleFactorB])}>"

            operation_name_str = operation.procedural_name()
            layout_a_str = LayoutTag[instance_layout_A]
            layout_b_str = LayoutTag[instance_layout_B]
            mixed_dtype_prepare_code = ""
            if operation.mixed_input_mode is not None:
                A_dtype = operation.A.element
                B_dtype = operation.B.element
                A_dtype_bits = DataTypeSize[A_dtype]
                B_dtype_bits = DataTypeSize[B_dtype]
                is_A_dtype_narrow = A_dtype_bits < B_dtype_bits
                if is_A_dtype_narrow:
                    narrow_dtype, wide_dtype = (A_dtype, B_dtype)
                    narrow_dtype_bits, wide_dtype_bits = (A_dtype_bits, B_dtype_bits)
                else:
                    narrow_dtype, wide_dtype = (B_dtype, A_dtype)
                    narrow_dtype_bits, wide_dtype_bits = (B_dtype_bits, A_dtype_bits)

                narrow_tag = DataTypeTag[narrow_dtype]
                wide_tag = DataTypeTag[wide_dtype]
                scale_tag = DataTypeTag[wide_dtype]
                zero_tag = DataTypeTag[wide_dtype]

                do_shuffle = False
                value_shuffle_str = ""
                if narrow_dtype_bits == 4 and wide_dtype_bits == 16:
                    value_shuffle_str = "cute::Layout<cute::Shape<cute::_2,cute::_4>, \
cute::Stride<cute::_4,cute::_1>>"
                    do_shuffle = True
                if narrow_dtype_bits == 8 and wide_dtype_bits == 16:
                    value_shuffle_str = "cute::Layout<cute::Shape<cute::_2,cute::_2>, \
cute::Stride<cute::_2,cute::_1>>"
                    do_shuffle = True
                do_shuffle = operation.mixed_input_shuffle and do_shuffle

                if do_shuffle:
                    if is_A_dtype_narrow:
                        stride_narrow_str = (
                            f"cutlass::detail::TagToStrideA_t<{layout_a_str}>"
                        )
                        layout_a_str = f"{operation_name_str}_LayoutNarrowReordered"
                    else:
                        stride_narrow_str = (
                            f"cutlass::detail::TagToStrideB_t<{layout_b_str}>"
                        )
                        layout_b_str = f"{operation_name_str}_LayoutNarrowReordered"
                    # The {operation_name_str}_ prefixes in mixed_dtype_prepare_code and
                    # layout_{a, b}_str are to prevent errors in Windows platform unity build
                    mixed_dtype_prepare_code = f"""
            using {operation_name_str}_StrideNarrow = {stride_narrow_str};
            using {operation_name_str}_ValueShuffle = {value_shuffle_str};
            static constexpr int {operation_name_str}_NumShuffleAtoms = 1;
            using {operation_name_str}_MmaAtomShape = \
cute::Layout<cute::Shape<cute::_1, cute::Int<{operation_name_str}_NumShuffleAtoms>>>;
            using {operation_name_str}_LayoutAtomQuant = \
decltype(cutlass::compute_memory_reordering_atom<{wide_tag}, {operation_name_str}_MmaAtomShape, \
{operation_name_str}_ValueShuffle>());
            using {operation_name_str}_LayoutNarrowReordered = \
decltype(cute::tile_to_shape({operation_name_str}_LayoutAtomQuant{{}}, \
cute::Layout<cute::Shape<int,int,int>, {operation_name_str}_StrideNarrow>{{}}));
                    """

                mixed_input_modes_to_element = {
                    MixedInputMode.ConvertOnly: narrow_tag,
                    MixedInputMode.ScaleOnly: f"cute::tuple<{narrow_tag}, {scale_tag}>",
                    MixedInputMode.ScaleWithZeroPoint: f"cute::tuple<{narrow_tag}, {scale_tag}, {zero_tag}>",
                }
                narrow_element = mixed_input_modes_to_element.get(
                    operation.mixed_input_mode, narrow_tag
                )

                if narrow_dtype == DataType.s4 and (
                    wide_dtype == DataType.e4m3 or wide_dtype == DataType.e5m2
                ):
                    narrow_element = (
                        f"cute::tuple<{narrow_tag}, cutlass::Array<{scale_tag}, 8>>"
                    )

                if is_A_dtype_narrow:
                    element_a = narrow_element
                else:
                    element_b = narrow_element

            if self.evt_name:
                epilogue_functor = self.evt_name

            if self.device_type == "xpu":
                arch = f"cutlass::arch::Xe{operation.arch}"
            else:
                arch = f"cutlass::arch::Sm{operation.arch}"

            values = {
                "operation_name": operation_name_str,
                "operation_suffix": self.operation_suffix,
                "problem_shape": self.problem_shape(operation),
                "element_a": element_a,
                "layout_a": self.pointerize_if_grouped(operation, layout_a_str),
                "element_b": element_b,
                "layout_b": self.pointerize_if_grouped(operation, layout_b_str),
                "element_c": DataTypeTag[operation.C.element],
                "layout_c": self.pointerize_if_grouped(
                    operation, LayoutTag[instance_layout_C]
                ),
                "element_d": DataTypeTag[operation.D.element],
                "layout_d": self.pointerize_if_grouped(
                    operation, LayoutTag[instance_layout_D]
                ),
                "element_accumulator": DataTypeTag[operation.accumulator_type()],
                "opcode_class_main": OpcodeClassTag[opcode_class_main],
                "opcode_class_epi": OpcodeClassTag[opcode_class_epi],
                "arch": arch,
                "tile_shape_m": str(tile_shape_m),
                "tile_shape_n": str(tile_shape_n),
                "tile_shape_k": str(tile_shape_k),
                "cluster_shape_m": "cute::_"
                + str(operation.tile_description.cluster_shape[0])
                if operation.tile_description.cluster_shape[0] > 0
                else "int",
                "cluster_shape_n": "cute::_"
                + str(operation.tile_description.cluster_shape[1])
                if operation.tile_description.cluster_shape[1] > 0
                else "int",
                "cluster_shape_k": "cute::_"
                + str(operation.tile_description.cluster_shape[2])
                if operation.tile_description.cluster_shape[2] > 0
                else "int",
                "instruction_shape_m": str(instruction_shape[0]),
                "instruction_shape_n": str(instruction_shape[1]),
                "instruction_shape_k": str(instruction_shape[2]),
                "kernel_schedule": str(KernelScheduleTag[operation.kernel_schedule]),
                "epilogue_schedule": str(epilogue_schedule_type),
                "epi_tile_mn": epi_tile_mn,
                "epilogue_functor": epilogue_functor,
                "stages": stage_count_string,
                "align_a": str(operation.A.alignment),
                "align_b": str(operation.B.alignment),
                "align_c": str(operation.C.alignment),
                "align_d": str(operation.D.alignment),
                "transform_a": ComplexTransformTag[operation.A.complex_transform],
                "transform_b": ComplexTransformTag[operation.B.complex_transform],
                "math_operation": MathOperationTag[
                    operation.tile_description.math_instruction.math_operation
                ],
                "epilogue_vector_length": str(epilogue_vector_length),
                "element_epilogue": str(DataTypeTag[operation.element_epilogue]),
                "tile_scheduler": str(TileSchedulerTag[operation.tile_scheduler]),
                "mixed_dtype_prepare_code": mixed_dtype_prepare_code,
            }

            return SubstituteTemplate(self.gemm_template, values)
<|endoftext|># mypy: disable-error-code="no-untyped-def"
import torch.cuda


class cudaError_t:
    cudaSuccess = True


def cudaFree(n):
    return (cudaError_t.cudaSuccess,)


def cudaGetDeviceProperties(d):
    class DummyError:
        value = False

    return (DummyError(), torch.cuda.get_device_properties(d))
<|endoftext|># mypy: disable-error-code="no-untyped-def"
# flake8: noqa
import torch


class CUdeviceptr:
    pass


class CUstream:
    def __init__(self, v):
        pass


class CUresult:
    CUDA_SUCCESS = True


class nvrtc:
    pass


def cuDeviceGetCount():
    return (CUresult.CUDA_SUCCESS, torch.cuda.device_count())
<|endoftext|># mypy: allow-untyped-defs
import functools
import logging
import os
import shutil
import subprocess

from torch._inductor import config
from torch._inductor.codegen.xpu.xpu_env import get_xpu_arch
from torch._inductor.utils import is_linux

from ..cuda.compile_utils import _cutlass_include_paths


log = logging.getLogger(__name__)


@functools.cache
def icpx_exist(icpx_path: str | None = "icpx") -> bool:
    return icpx_path is not None and shutil.which(icpx_path) is not None


def _sycl_compiler() -> str:
    # Search order:
    # 0) which icpx
    # 1) config.xpu.oneapi_root
    # 2) ONEAPI_ROOT environment variable
    # 3) default system search PATH.
    if icpx_exist("icpx"):
        return "icpx"

    if os.path.exists(config.xpu.oneapi_root or ""):
        oneapi_root = config.xpu.oneapi_root
    elif os.path.exists(os.getenv("ONEAPI_ROOT") or ""):
        oneapi_root = os.getenv("ONEAPI_ROOT")
    else:
        oneapi_root = None

    if oneapi_root:
        oneapi_inclue = os.path.join(oneapi_root, "include")
        if "CPLUS_INCLUDE_PATH" in os.environ:
            os.environ["CPLUS_INCLUDE_PATH"] += ":" + oneapi_inclue
        else:
            os.environ["CPLUS_INCLUDE_PATH"] = oneapi_inclue
        return os.path.realpath(os.path.join(oneapi_root, "bin/icpx"))

    # Mirrors _cuda_compiler(): return the bare name and let the compile
    # subprocess produce the error, so callers can probe with icpx_exist().
    return "icpx"


def _sycl_lib_options() -> list[str]:
    """
    Util function for CUTLASS backend to find the correct XPU libraries.
    """
    # _set_gpu_runtime_env()  # cpp_extension consults the env
    from torch.utils import cpp_extension

    lpaths = cpp_extension.library_paths(device_type="xpu")
    extra_ldflags: list[str] = []
    if is_linux():
        for path in lpaths:
            if "torch/lib" in path:
                # don't want to depend on pytorch
                continue
            # -rpath ensures the DLL can find its dependencies when loaded, even
            # if the library path is non-standard.
            extra_ldflags.extend([f"-L{path}", "-Xlinker", f"-rpath={path}"])

        extra_ldflags.append("-lsycl")
    else:
        raise NotImplementedError(
            "Unsupported env, failed to find xpu libs! Currently only Linux is supported."
        )
    return extra_ldflags


def _sycl_arch_as_compile_option() -> str:
    arc_option_map = {"Xe12": "pvc", "Xe20": "bmg-g21,bmg-g31"}
    arch = get_xpu_arch()
    return arc_option_map.get(arch, "pvc")


def _sycl_compiler_options() -> list[str]:
    options = [
        "-DCUTLASS_ENABLE_SYCL",
        "-DSYCL_INTEL_TARGET",
        "-DCUTLASS_VERSIONS_GENERATED",
        "-O3",
        "-DNDEBUG",
        "-std=c++20",
        "-fPIC",
        "-fsycl",
        "-fsycl-targets=spir64_gen",
        "-Xspirv-translator",
        "-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate",
        "-fno-sycl-instrument-device-code",
        "-DMKL_ILP64",
        "-MD",
        "-Xs",
        (
            f"-device {_sycl_arch_as_compile_option()} "
            "-options \"-igc_opts 'VISAOptions=-perfmodel,VectorAliasBBThreshold=100000000000,"
            "ExtraOCLOptions=-cl-intel-256-GRF-per-thread'\" "
            "-options -ze-opt-large-register-file"
        ),
    ]
    if config.cutlass.enable_debug_info:
        options.extend(["-lineinfo", "-g", "-DCUTLASS_DEBUG_TRACE_LEVEL=1"])
    return options


def xpu_compile_command(
    src_files: list[str],
    dst_file: str,
    dst_file_ext: str,
    extra_args: list[str] | None = None,
) -> str:
    if extra_args is None:
        extra_args = []
    include_paths = _cutlass_include_paths()
    sycl_lib_options = _sycl_lib_options()
    sycl_compiler_options = _sycl_compiler_options()

    # Build command as a list to preserve arguments with spaces
    cmd_parts = (
        [_sycl_compiler()]
        + extra_args
        + ["-I" + path for path in include_paths]
        + ["-isystem", "/include"]
        + sycl_compiler_options
        + sycl_lib_options
    )
    if dst_file_ext == "o":
        cmd_parts.extend(["-c", "-o", dst_file] + src_files)
    elif dst_file_ext == "so":
        cmd_parts.extend(["-shared", "-o", dst_file] + src_files)
    elif dst_file_ext == "exe":
        cmd_parts.extend(["-o", dst_file] + src_files)
    else:
        raise NotImplementedError(f"Unsupported output file suffix {dst_file_ext}!")

    res = subprocess.list2cmdline(cmd_parts)
    log.debug("XPU command: %s", res)
    return res
<|endoftext|>from __future__ import annotations

from ..common import (
    DeviceIdx,
    DeviceOpOverrides,
    register_device_op_overrides,
    TritonScratchWorkspace,
)


class XPUDeviceOpOverrides(DeviceOpOverrides):
    def import_get_raw_stream_as(self, name: str) -> str:
        return f"from torch._C import _xpu_getCurrentRawStream as {name}"

    def set_device(self, device_idx: DeviceIdx) -> str:
        return f"torch.xpu.set_device({device_idx})"

    def synchronize(self) -> str:
        return "torch.xpu.synchronize()"

    def device_guard(self, device_idx: DeviceIdx) -> str:
        return f"torch.xpu._DeviceGuard({device_idx})"

    def current_device_idx_expr(self) -> str:
        return "torch.xpu.current_device()"

    def current_stream(self) -> str:
        return "torch.xpu.current_stream()"

    def cpp_device_guard(self) -> str:
        return "at::DeviceGuard"

    def cpp_aoti_device_guard(self) -> str:
        return "AOTIXpuGuard"

    def cpp_stream_guard(self) -> str:
        return "at::xpu::XPUStreamGuard"

    def cpp_aoti_stream_guard(self) -> str:
        return "AOTIXpuStreamGuard"

    def cpp_getStreamFromExternal(self) -> str:
        return "at::xpu::getStreamFromExternal"

    def kernel_header(self) -> str:
        source_codes = """
        #include <torch/csrc/inductor/aoti_runtime/sycl_runtime_wrappers.h>
        """
        return source_codes

    def kernel_driver(self) -> str:
        return ""

    def cpp_stream_type(self) -> str:
        return "sycl::queue*"

    def aoti_get_stream(self) -> str:
        return "aoti_torch_get_current_xpu_stream"

    def cpp_kernel_type(self) -> str:
        return "std::unique_ptr<sycl::kernel>"

    def cpp_device_ptr(self) -> str:
        return "void *"

    def cpp_scratch(
        self, idx: int, workspace: TritonScratchWorkspace, prefix: str | None = None
    ) -> tuple[list[str], str] | None:
        return [f"void *global_scratch_{idx} = 0;"], f"global_scratch_{idx}"


register_device_op_overrides("xpu", XPUDeviceOpOverrides())
<|endoftext|>import functools
import logging

import torch
from torch._inductor.utils import clear_on_fresh_cache


log = logging.getLogger(__name__)


@clear_on_fresh_cache
@functools.lru_cache(1)
def get_xpu_arch() -> str | None:
    from torch.testing._internal.common_xpu import get_xpu_codename, XPUCodename

    name2arch = {
        XPUCodename.PVC: "Xe12",
        XPUCodename.BMG: "Xe20",
    }

    codename = get_xpu_codename()
    if not codename or codename not in name2arch:
        log.warning("Unknown XPU codename, cannot determine architecture")
        return None

    return name2arch[codename]


@clear_on_fresh_cache
@functools.lru_cache(1)
def get_xpu_version() -> str | None:
    # string of version, like 20250101
    try:
        xpu_version = torch.version.xpu or ""
        return xpu_version
    except Exception:
        log.exception("Error getting xpu version")
        return None
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from torch._inductor.scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
    FusedSchedulerNode,
    Scheduler,
    SchedulerNode,
)

from ..cutlass.scheduling import CUTLASSScheduling
from ..triton import TritonScheduling


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TypeAlias

    from sympy import Expr

    import torch
    from torch.utils._ordered_set import OrderedSet

    from ..common import BackendFeature

    _IntLike: TypeAlias = int | Expr


class XPUCombinedScheduling(BaseScheduling):
    """
    Scheduler for XPU Kernels, which delegates calls as appropriate
    to the SYCL-C++ and Triton Schedulers, which both work for XPU devices
    and use a unified-wrapper for codegen.

    If Scheduling code needs to be specialized for the case of mixed Triton / SYCL C++ code,
    this would also be the place to do it.
    """

    def __init__(self, scheduler: Scheduler | None) -> None:
        super().__init__(scheduler)
        self._triton_scheduling = TritonScheduling(scheduler)
        self._cutlass_scheduling = CUTLASSScheduling(scheduler)

    def get_backend_features(self, device: torch.device) -> OrderedSet[BackendFeature]:
        return self._triton_scheduling.get_backend_features(device)

    def choose_node_backend(self, node: BaseSchedulerNode) -> BaseScheduling:
        if self._cutlass_scheduling.is_cutlass_template(node):
            return self._cutlass_scheduling
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
        return self._triton_scheduling.can_fuse_vertical(node1, node2)

    def can_fuse_horizontal(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        for node in (node1, node2):
            if self._cutlass_scheduling.is_cutlass_template(node):
                return self._cutlass_scheduling.can_fuse_horizontal(
                    node1, node2
                )  # always False at the moment
        return self._triton_scheduling.can_fuse_horizontal(node1, node2)

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
                raise AssertionError("cutlass template does not support prologue_nodes")
            return self._cutlass_scheduling.codegen_template(
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
        return self._triton_scheduling.benchmark_codegened_module(module)

    def generate_kernel_code_from_nodes(
        self,
        nodes: Sequence[Any],
        benchmark_kernel: bool = False,
        hint_override: int | None = None,
    ) -> str:
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
import logging
from collections.abc import Sequence
from typing import cast, TypeGuard

from ... import config
from ...codecache import code_hash, get_path
from ...scheduler import BaseSchedulerNode, BaseScheduling, SchedulerNode
from ...utils import get_fused_kernel_name, get_kernel_metadata, sympy_product
from ...virtualized import V
from ..common import IndentedBuffer
from .rocm_template_buffer import ROCmTemplateBuffer


log = logging.getLogger(__name__)


class ROCmCPPScheduling(BaseScheduling):
    """
    Partial Scheduling implementation for ROCm C++ Kernels.
    This class is intended to be used in combination with TritonScheduling,
    and delegated to by CUDACombinedScheduling.

    It handles fusion decisions and ROCm C++ specific template code generation.
    """

    def group_fn(self, sizes):
        return tuple(V.graph.sizevars.simplify(sympy_product(s)) for s in sizes)

    @staticmethod
    def is_rocm_cpp_template(node: BaseSchedulerNode) -> TypeGuard[SchedulerNode]:
        return isinstance(node, SchedulerNode) and isinstance(
            node.node, ROCmTemplateBuffer
        )

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        return False

    def define_kernel(self, src_code: str, node_schedule) -> str:
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            fused_name = (
                get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
                if config.triton.descriptive_names
                else ""
            )
            kernel_name = "_".join(["rocm", fused_name, wrapper.next_kernel_suffix()])
            # use the original src_code as the key
            wrapper.src_to_kernel[src_code] = kernel_name
            src_code = src_code.replace("KERNEL_NAME", kernel_name)

            _, _, kernel_path = get_path(code_hash(src_code), "py")

            compile_wrapper = IndentedBuffer()
            compile_wrapper.writeline("async_compile.rocm(r'''")
            compile_wrapper.splice(src_code, strip=True)
            compile_wrapper.writeline(
                f"''', 'so', aot_compile={str(V.graph.aot_mode)})"
            )

            metadata_comment = f"# kernel path: {kernel_path}"
            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            metadata_comment += "\n" + origins + "\n" + detailed_origins
            wrapper.define_kernel(
                kernel_name, compile_wrapper.getvalue(), metadata_comment
            )

            # For JIT cpp_wrapper, the kernel call site emits a bare
            # `extern "C"` symbol reference; compile the .so now so the
            # wrapper compile can link against it via extra_flags.
            # Autotune compiled a .so for the unfused kernel name
            # (e.g. rocm_<hash>), but at codegen time the kernel name
            # gains a descriptive prefix (rocm_fused_mm_<hash>), so the
            # exported symbol differs and we need a fresh compile.
            if V.graph.cpp_wrapper and not V.graph.aot_mode:
                from ...codecache import ROCmCodeCache

                so_path, _, _ = ROCmCodeCache.compile(src_code, "so")
                wrapper.external_kernel_libs.add(so_path)
        return kernel_name

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
    ):
        """
        Codegen a ROCm template, possibly with fused epilogues
        """
        if not self.is_rocm_cpp_template(template_node):
            raise AssertionError(
                "Template node passed to ROCmScheduler.codegen_template must be a SchedulerNode that wraps a ROCmTemplateBuffer"
            )
        _, (_numel, rnumel) = template_node.group
        if rnumel != 1:
            raise AssertionError(f"expected rnumel == 1, got {rnumel}")
        ctb: ROCmTemplateBuffer = cast(ROCmTemplateBuffer, template_node.node)
        kernel, render = ctb.make_kernel_render(ctb)  # type: ignore[misc]
        with kernel:
            template_node.mark_run()
            src_code = render()

        with V.set_kernel_handler(kernel):
            node_schedule = [template_node]
            kernel_name = self.define_kernel(src_code, node_schedule)
        self.codegen_comment(node_schedule, kernel_name)
        kernel.call_kernel(kernel_name, ctb)
        V.graph.removed_buffers |= kernel.removed_buffers
        self.free_buffers_in_scheduler()
<|endoftext|># mypy: allow-untyped-defs
import logging
from collections.abc import Callable, Sequence
from typing import Any, TYPE_CHECKING

import torch._inductor.config as config
from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
from torch._inductor.utils import do_bench_using_profiling

from ...ir import Buffer, ChoiceCaller, IRNode, Layout, PrimitiveInfoType, TensorBox
from ...virtualized import V
from ..common import Kernel, OpOverrides, WorkspaceArg, WorkspaceZeroMode
from ..cpp_utils import CppPrinter
from .rocm_benchmark_request import ROCmBenchmarkRequest
from .rocm_template_buffer import ROCmTemplateBuffer
from .rocm_utils import DTYPE_TO_ROCM_TYPE


if TYPE_CHECKING:
    from torch._inductor.codegen.rocm.rocm_template import ArgInfo, ROCmTemplate

log = logging.getLogger(__name__)

cexpr = CppPrinter().doprint


def _normalize_idx(index: int, total_length: int) -> int:
    return index if index >= 0 else index + total_length


class ROCmKernel(Kernel):
    """
    Baseclass for ROCm based Kernels
    """

    overrides = OpOverrides  # type: ignore[assignment]


class ROCmTemplateKernel(ROCmKernel):
    """
    Template kernels defined by ROCm in C++.
    """

    _EXTRA_CPP_ARGS = "size_t* workspace_size, uint8_t* workspace, hipStream_t stream"

    def __init__(
        self,
        kernel_name: str,
        runtime_arg_info: list["ArgInfo"],
        runtime_arg_values: list[Any],
    ) -> None:
        """
        Initializes a new instance of the ROCmTemplateKernel class.

        Args:
            kernel_name (str): The name of the kernel.
        """
        super().__init__()
        self.kernel_name = kernel_name
        # Mapping from arg name to IRNode.
        self.named_nodes: dict[str, IRNode] = {}
        self.runtime_arg_info = runtime_arg_info
        self.runtime_arg_values = runtime_arg_values

    def get_signature(self):
        return self.signature

    def def_kernel(
        self,
        inputs: list[IRNode],
        outputs: list[IRNode],
        size_args: list[str],
        names_str: str = "",
        input_reorder: list[int] | None = None,
    ) -> str:
        """
        Hook called from template code to generate function definition and
        needed args.

        Args:
            inputs: List of input IRNodes
            outputs: List of output IRNodes
            names_str: Comma separated list of input + output argument names.
            input_reorder: The actual order of input nodes.
                           e.g. The template might have input argument defined as [X, W, Bias],
                           and the actual input passed into this template could be [Bias, X, W].
                           In this case, the `input_reorder` would be [2, 0, 1].
        """
        names = [x.strip() for x in names_str.strip().split(",")]
        if len(inputs) + len(outputs) != len(names):
            raise RuntimeError(
                f"{len(inputs) + len(outputs)=} != {len(names)=}, {inputs=}, {outputs=}, {names=}"
            )

        if input_reorder == [2, 0, 1]:
            input_reorder = [4, 0, 1, 2, 3]

        if input_reorder is not None:
            if len(inputs) != len(input_reorder):
                raise AssertionError(
                    f"expected len(inputs) == len(input_reorder), "
                    f"got {len(inputs)} and {len(input_reorder)}"
                )
        else:
            input_reorder = list(range(len(inputs)))

        for idx in input_reorder:
            name = names[idx]
            node = inputs[idx]
            if node is not None:
                self.named_nodes[name] = node
                self.args.input_buffers[node.get_name()] = name

        for name, node in zip(names[len(inputs) : len(inputs) + len(outputs)], outputs):
            if node is not None:
                self.named_nodes[name] = node
                self.args.output_buffers[node.get_name()] = name

        arg_defs, *_ = self.args.cpp_argdefs(DTYPE_TO_ROCM_TYPE)

        runtime_arg_defs = [f"{arg.ty} {arg.name}" for arg in self.runtime_arg_info]

        signature = f"int {self.kernel_name}({', '.join(arg_defs + size_args + runtime_arg_defs)},{self._EXTRA_CPP_ARGS})"
        self.signature = signature
        return signature

    def call_kernel(
        self,
        name: str,
        node: "ROCmTemplateBuffer",  # type: ignore[name-defined]
    ) -> None:
        """
        Generates code to call the kernel through V.graph.wrapper_code.
        used from within torch._inductor.wrapper.PythonWrapperCodegen

        name: Name of kernel function.
        node: The ROCmTemplateBuffer node which contains information about the kernel, it's fused epilogue nodes
        as well as all required inputs and outputs.
        """
        wrapper = V.graph.wrapper_code

        arg_types: list[Any]
        if V.graph.cpp_wrapper:
            # Make sure we initialize these kernels since they're exported as
            # C-style symbol names.
            if not isinstance(wrapper, CppWrapperCpu):
                raise AssertionError(
                    f"expected wrapper to be CppWrapperCpu, got {type(wrapper)}"
                )
            wrapper.initialized_kernels[name] = self
            # Kinda hacky because we always originally initialize name with "KERNEL_NAME"
            # So, we replace with the real kernel name passed as an arg to this function.
            self.signature = self.signature.replace("KERNEL_NAME", name)
            _, call_args, arg_types = self.args.cpp_argdefs(DTYPE_TO_ROCM_TYPE)
        else:
            _, call_args, _, arg_types = self.args.python_argdefs()

        kernel_args = []
        for arg in call_args:
            # dynamo wraps unspec variable as 0d CPU tensor, need convert to scalar
            if V.graph.is_unspec_arg(arg):
                arg = arg + ".item()"
            else:
                if not V.graph.cpp_wrapper:
                    arg = f"c_void_p({arg}.data_ptr())"
            kernel_args.append(arg)

        # add size args
        size_args = [
            f"{V.graph.sizevars.simplify(sarg)}" for sarg in node.template.size_args()
        ]

        if V.graph.cpp_wrapper:
            kernel_args.extend(size_args)
        else:
            kernel_args.extend(f"c_int({sarg})" for sarg in size_args)

        if V.graph.cpp_wrapper:
            arg_types.extend(["int"] * len(node.template.size_args()))

        # the runtime args come right after the size args
        kernel_args.extend(self.runtime_arg_values)
        for arg in self.runtime_arg_info:
            arg_types.append(arg.ty)

        # workspace_size ptr is NULL to mark this call is not intended for retrieving workspace_size.
