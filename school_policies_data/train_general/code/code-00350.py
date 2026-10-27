                _, mod = e.args
                maybe_add_divisor(mod, "modulo")
            elif isinstance(e, ModularIndexing):
                _, div, mod = e.args
                maybe_add_divisor(div, "modular indexing division")
                maybe_add_divisor(mod, "modular indexing modulo")

            # Recurse into arguments
            if hasattr(e, "args"):
                for arg in e.args:
                    if isinstance(arg, sympy.Expr):
                        walk(arg)

        walk(expr)
        return divisors

    def codegen_cpp_sizevar(self, x: sympy.Expr, *, simplify: bool = True) -> str:
        maybe_simplified_x = V.graph.sizevars.simplify(x) if simplify else x
        # In AOT mode, emit runtime checks for potential division/modulo by zero
        # to prevent SIGFPE crashes when symbolic tensor shapes can be 0.
        if V.graph.aot_mode:
            for divisor, op_name in self._extract_divisors_from_expr(
                maybe_simplified_x
            ):
                self.write_assert_div_by_zero(cexpr(divisor), op_name)
        return cexpr(maybe_simplified_x)

    def _codegen_assert_div_by_zero(
        self, code: IndentedBuffer, divisor_str: str, op_name: str
    ) -> None:
        """Emit one div-by-zero AOTI check to `code` (replay-phase target).

        Uses writeline_aot so the check is routed to the AOTI side only in
        dual-wrapper mode; on plain buffers writeline_aot has the expected
        single-stream behavior (AotOnlyBuffer delegates to writeline, plain
        IndentedBuffer is a no-op).
        """
        code.writeline_aot(
            f'AOTI_TORCH_CHECK({divisor_str} != 0, "Integer {op_name} by zero");'
        )

    def codegen_sizevar(self, x: sympy.Expr) -> str:
        return self.codegen_cpp_sizevar(x)

    def codegen_tuple_access(self, basename: str, name: str, index: str) -> str:
        # in the abi_compatible mode, outputs are returned via arguments
        return name

    def codegen_shape_tuple(self, shape: Sequence[sympy.Expr]) -> str:
        parts = [*map(self.codegen_sizevar, shape)]
        if len(parts) == 0:
            return "{}"
        if len(parts) == 1:
            return f"{{{parts[0]}, }}"
        return f"{{{', '.join(parts)}}}"

    def ensure_size_computed(self, sym: sympy.Symbol):
        if isinstance(sym, sympy.Symbol) and symbol_is_type(sym, SymT.PRECOMPUTED_SIZE):
            if sym in self.computed_sizes:
                return
            self.computed_sizes.add(sym)
            expr = V.graph.sizevars.inv_precomputed_replacements[sym]
            self.writeline(f"int64_t {sym} = {cexpr(expr)};")

    def _generate_symbolic_call_arg_helper(
        self, arg: SymbolicCallArg, graph: GraphLowering
    ) -> None:
        enable_kernel_profile = config.cpp.enable_kernel_profile and sys.platform in [
            "linux",
            "win32",
        ]
        if enable_kernel_profile or (arg.inner, graph) not in self.kernel_numel_expr:
            # When enable_kernel_profile is on, each kernel call is wrapped in
            # its own {} scope block, so we must redeclare the variable each
            # time since prior declarations are no longer visible.
            self.kernel_numel_expr.add((arg.inner, graph))
            self.writeline(f"int64_t {arg.inner} = {cexpr(arg.inner_expr)};")
        else:
            self.writeline(f"{arg.inner} = {cexpr(arg.inner_expr)};")

    def _codegen_dynamic_scalar(self, node):
        (data,) = (t.codegen_reference() for t in node.inputs)
        self.codegen_tensor_item(node.inputs[0].get_dtype(), data, f"{node.sym}_raw")

        if len(node.keypath) == 0:
            self.writeline(f"auto {node.sym} = {node.sym}_raw;")
        elif len(node.keypath) == 1 and isinstance(node.keypath[0], ConvertIntKey):
            self.writeline(f"int64_t {node.sym} = {node.sym}_raw ? 1 : 0;")
        elif len(node.keypath) == 1 and isinstance(node.keypath[0], DivideByKey):
            # TODO: assert divisibility here
            self.writeline(
                f"int64_t {node.sym} = {node.sym}_raw / {node.keypath[0].divisor};"
            )
        else:
            raise AssertionError(f"unrecognized keypath {node.keypath}")

        # record in unbacked_symbol_decls so we won't generate a declaration of the symbol again
        self.unbacked_symbol_decls.add(str(node.sym))

    def codegen_dynamic_select_index(self, node, clamp):
        index_cpp_str = self.val_to_arg_str_for_prim_type(node.index, int)
        size_cpp_str = self.val_to_arg_str_for_prim_type(node.size, int)

        # codegen index
        sym = node.unbacked_offset_symbol
        index_str = (
            f"{index_cpp_str} < 0 ? {index_cpp_str} + "
            f"{self.val_to_arg_str_for_prim_type(node.size, int)}: {index_cpp_str}"
        )
        self.writeline(f"auto {sym}_index = {index_str};")
        index_str_clamped = (
            f"{sym}_index < 0 ? 0 : ({sym}_index > {size_cpp_str} ? {size_cpp_str} : {sym}_index)"
            if clamp
            else f"{sym}_index"
        )
        self.writeline(f"auto {sym}_index_clamped = {index_str_clamped};")
        self.writeline(
            f"auto {sym} = {self.val_to_arg_str_for_prim_type(node.base_offset, int)} + "
            f"{self.val_to_arg_str_for_prim_type(node.base_dim_stride, int)} * {sym}_index_clamped;"
        )
        # record in unbacked_symbol_decls so we won't generate a declaration of the symbol again
        self.unbacked_symbol_decls.add(str(sym))

    def codegen_dynamic_slice_size(self, node):
        start_cpp_str = self.val_to_arg_str_for_prim_type(node.start, int)
        end_cpp_str = self.val_to_arg_str_for_prim_type(node.end, int)
        size_cpp_str = self.val_to_arg_str_for_prim_type(node.size, int)
        step_cpp_str = self.val_to_arg_str_for_prim_type(node.step, int)
        sym = node.unbacked_size_symbol

        def codegen_clamp(index_str, start=True):
            suf = "st" if start else "en"
            index_ = f"{sym}_{suf}_index"
            self.writeline(
                f"int64_t {index_} = {index_str} < 0 ? {index_str} + {size_cpp_str} : {index_str};"
            )
            self.writeline(
                f"int64_t {sym}_{suf}_cl = {index_} < 0 ? 0 : ({index_} > {size_cpp_str} ? {size_cpp_str} : {index_});"
            )

        codegen_clamp(start_cpp_str, start=True)
        codegen_clamp(end_cpp_str, start=False)
        if node.step == 1:
            step_str = f"{sym}_en_cl - {sym}_st_cl"
        else:
            step_str = (
                f"({sym}_en_cl - {sym}_st_cl + {step_cpp_str} - 1) / {step_cpp_str}"
            )
        self.writeline(f"int64_t {sym}_with_step = {step_str};")
        self.writeline(f"int64_t {sym} = {sym}_with_step < 0 ? 0 : {sym}_with_step;")
        self.unbacked_symbol_decls.add(str(sym))

    def make_buffer_free(self, buffer):
        return (
            ""
            if isinstance(buffer.get_output_spec(), ir.MultiOutputLayout)
            or isinstance(buffer, ir.TMADescriptor)
            else f"{buffer.get_name()}.reset();"
        )

    def make_free_by_names(self, names_to_del: list[str]):
        return " ".join(f"{name}.reset();" for name in names_to_del)

    def codegen_exact_buffer_reuse(self, old_name: str, new_name: str, del_line: str):
        return f"auto {new_name} = std::move({old_name});  // reuse"

    def generate_debug_sync(self, buffer):
        pass

    def generate_profiler_mark_wrapper_call(self, stack):
        self.wrapper_call.writeline(
            'RAIIAtenRecordFunctionHandle record_inductor_wrapper_call_("inductor_wrapper_call", nullptr);'
        )

    def generate_start_graph(self):
        pass

    def generate_end_graph(self):
        pass

    def generate_inf_and_nan_checker(self, nodes):
        for buf in nodes.get_names():
            # TODO: Add buf name directly into check_inf_and_nan.
            self.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_check_inf_and_nan({buf}));"
            )

    def codegen_input_nan_asserts(self) -> None:
        self.wrapper_call.writeline("// make sure graph inputs are not nan/inf")
        for name, buf in self.get_graph_inputs().items():
            if isinstance(buf, (sympy.Expr, ir.TorchBindObject)):
                continue
            self.wrapper_call.writeline(
                f'AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_check_inf_and_nan("{name}", {name}));'
            )

    def _codegen_assert_size_stride(
        self,
        code: IndentedBuffer,
        name: str,
        size: str,
        stride: str,
        op_name: str,
        dtype: torch.dtype | None = None,
    ) -> None:
        if V.graph.aot_mode and V.graph.is_const_graph:
            return
        dtype_args = (
            f', {self.codegen_dtype(dtype)}, "{dtype}"' if dtype is not None else ""
        )
        stmt = f'assert_size_stride({name}, {size}, {stride}, "{op_name}"{dtype_args});'
        if V.graph.aot_mode:
            guarded = f"if (_check_aoti_runtime_check_inputs_env()) {{ {stmt} }}"
            if V.graph.is_dual_wrapper_mode:
                # _check_aoti_runtime_check_inputs_env() is AOTI-only; the JIT
                # side gets the plain assert.
                code.writeline_jit(stmt)
                code.writeline_aot(guarded)
            else:
                code.writeline(guarded)
        else:
            code.writeline(stmt)

    def _codegen_assert_size_stride_grouped(
        self, code: IndentedBuffer, asserts: list[tuple[str, str, str]], op_name: str
    ) -> None:
        for name, size, stride in asserts:
            self._codegen_assert_size_stride(code, name, size, stride, op_name)

    def codegen_device(self, device):
        if device.type not in DEVICE_TO_ATEN:
            raise AssertionError(device.type + " not found in DEVICE_TO_ATEN")
        device_str = DEVICE_TO_ATEN[device.type][5:].lower()  # remove "at::k"
        self.used_cached_devices.add(device_str)
        return f"cached_torch_device_type_{device_str}, {device.index or 0}"

    def codegen_device_idx(self, device, device_id):
        device_id = device_id.strip()
        if not V.graph.aot_mode:
            return device_id

        if device.type == V.graph.device_type and device.index is not None:
            # The primary compile-time device is relocatable via device_index at load.
            # Other explicit device indices must stay intact for cross-device copies.
            primary_device_idx = next(iter(V.graph.device_idxs), None)
            if primary_device_idx is not None and device.index != primary_device_idx:
                return device_id

        return "this->device_idx_"

    def codegen_dtype(self, dtype):
        dtype_str = str(dtype).split(".")[-1]
        self.used_cached_dtypes.add(dtype_str)
        return f"cached_torch_dtype_{dtype_str}"

    def codegen_layout(self, layout):
        layout_str = str(layout).split(".")[-1]
        self.used_cached_layouts.add(layout_str)
        return f"cached_torch_layout_{layout_str}"

    def codegen_memory_format(self, memory_format):
        memory_format_str = str(memory_format).split(".")[-1]
        self.used_cached_memory_formats.add(memory_format_str)
        return f"cached_torch_memory_format_{memory_format_str}"

    def codegen_int_array_var(
        self,
        int_array: str,
        writeline: Callable[..., None],
        known_statically=False,
        graph=None,  # for per-graph caching
    ) -> str:
        # Use id(graph) for caching to avoid circular references
        # Bound methods have transient IDs, so we use the ID of the bound object instead.
        writeline_id = (
            id(writeline.__self__) if hasattr(writeline, "__self__") else id(writeline)
        )
        cache_key = (
            int_array,
            writeline_id,
            known_statically,
            id(graph) if graph else None,
        )
        if cache_key not in self.codegen_int_array_var_cache:
            self.codegen_int_array_var_cache[cache_key] = (
                self._codegen_int_array_var_impl(int_array, writeline, known_statically)
            )

        return self.codegen_int_array_var_cache[cache_key]

    def _codegen_int_array_var_impl(
        self,
        int_array: str,
        writeline: Callable[..., None],
        known_statically: bool,
    ) -> str:
        # Used for size/stride declaration
        #
        # Because the memory planning is done in two passes (see the implementation
        # of self.generate), the writeline behavior is different in the two passes.
        # As a result, the emitted int array declarations may appear in a later
        # position of the generated code, so the second pass codegen should not
        # reuse int array declarations generated in the first pass.
        # This is why writeline needs to explicitly passed in as a parameter.
        var = f"int_array_{next(self.int_array_id)}"
        ctype = "int64_t"
        if int_array == "{}":
            #  An array of unknown bound cannot be initialized with {}.
            if known_statically:
                if config.cpp.use_constexpr_for_int_array:
                    writeline(f"static constexpr {ctype} *{var}=nullptr;")
                else:
                    writeline(f"static const {ctype} *{var}=nullptr;")
            else:
                writeline(f"const {ctype} *{var}=nullptr;")
        else:
            if var not in self.declared_int_array_vars:
                self.declared_int_array_vars.add(var)
                if known_statically:
                    if config.cpp.use_constexpr_for_int_array:
                        writeline(f"static constexpr {ctype} {var}[] = {int_array};")
                    else:
                        writeline(f"static const {ctype} {var}[] = {int_array};")
                else:
                    writeline(f"const {ctype} {var}[] = {int_array};")
        return var

    def make_buffer_allocation(self, buffer):
        return self.make_allocation(
            buffer.get_name(),
            buffer.get_device(),
            buffer.get_dtype(),
            buffer.get_size(),
            buffer.get_stride(),
            V.graph.get_allocation_size(buffer),
            buffer.get_is_pinned(),
        )

    def make_allocation(
        self, name, device, dtype, shape, stride, allocation_shape=None, is_pinned=False
    ):
        """Emit C++ to allocate a tensor handle via aoti_torch_empty_strided.

        Returns the RAIIAtenTensorHandle binding string; if the allocation
        shape differs from the logical shape, a follow-up aoti_torch_as_strided
        is emitted to reinterpret the handle.
        """
        if allocation_shape is None:
            allocation_shape = shape

        orig_stride = stride
        device_str = self.codegen_device(device)
        dtype_code = self.codegen_dtype(dtype)
        size = self.codegen_shape_tuple(shape)
        allocation_size = self.codegen_shape_tuple(allocation_shape)
        stride = self.codegen_shape_tuple(orig_stride)

        size_array_var = self.codegen_int_array_var(
            size,
            self.wrapper_call.writeline,
            known_statically=self.is_statically_known_list_of_ints(shape),
            graph=self.get_codegened_graph(),
        )

        if allocation_size != size:
            allocation_size_array_var = self.codegen_int_array_var(
                allocation_size,
                self.wrapper_call.writeline,
                known_statically=self.is_statically_known_list_of_ints(
                    allocation_shape
                ),
                graph=self.get_codegened_graph(),
            )
        else:
            allocation_size_array_var = size_array_var

        stride_array_var = self.codegen_int_array_var(
            stride,
            self.wrapper_call.writeline,
            known_statically=self.is_statically_known_list_of_ints(orig_stride),
            graph=self.get_codegened_graph(),
        )
        device_type, device_id = device_str.split(",")
        handle_name = f"{name}_handle"
        self.wrapper_call.writeline(f"AtenTensorHandle {handle_name};")
        pinned_str = "_pinned" if is_pinned else ""

        def _alloc_line(device_idx: str) -> str:
            args = ", ".join(
                [
                    str(len(shape)),
                    allocation_size_array_var,
                    stride_array_var,
                    dtype_code,
                    device_type,
                    device_idx,
                    f"&{handle_name}",
                ]
            )
            return f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided{pinned_str}({args}));"

        # AOTI side: codegen_device_idx returns this->device_idx_ for the
        # primary device so the model honors runtime device assignment, but
        # keeps concrete indices for non-primary (cross-device) allocations.
        # JIT side (pure or dual mode's JIT) has no this->device_idx_ in scope,
        # so use the concrete graph device id.
        self.wrapper_call.writeline_aot(
            _alloc_line(self.codegen_device_idx(device, device_id))
        )
        self.wrapper_call.writeline_jit(_alloc_line(device_id))

        if allocation_size != size:
            old_handle_name, handle_name = handle_name, f"{name}_handle_restrided"
            self.wrapper_call.writeline(f"AtenTensorHandle {handle_name};")
            args = [
                old_handle_name,
                size_array_var,
                stride_array_var,
                f"&{handle_name}",
            ]
            self.wrapper_call.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_as_strided({', '.join(args)}));"
            )
            self.wrapper_call.writeline(
                f"wrap_with_raii_handle_if_needed({old_handle_name});"
            )

        return f"RAIIAtenTensorHandle {name}({handle_name});"

    def codegen_alloc_from_pool(
        self, name, offset, dtype, shape, stride
    ) -> tuple[str, list[str]]:
        size = self.codegen_shape_tuple(shape)
        stride = self.codegen_shape_tuple(stride)
        tmp_name = f"tmp_tensor_handle_{next(self.tmp_tensor_id)}"
        args = [
            name,
            cexpr(offset),  # bytes not numel
            self.codegen_dtype(dtype),
            str(len(shape)),
            self.codegen_int_array_var(
                size, self.wrapper_call.writeline, graph=self.get_codegened_graph()
            ),
            self.codegen_int_array_var(
                stride, self.wrapper_call.writeline, graph=self.get_codegened_graph()
            ),
            f"&{tmp_name}",
        ]
        # We return the lines instead of writing here because writing here is bug prone.
        # If you write aoti_torch__alloc_from_pool lines, you must write the RAIIAtenTensorHandle
        # as well, otherwise you get memory leaks
        allocations_to_write = [
            f"AtenTensorHandle {tmp_name};",
            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch__alloc_from_pool({', '.join(args)}));",
        ]
        return f"RAIIAtenTensorHandle({tmp_name})", allocations_to_write

    def codegen_reinterpret_view(
        self,
        data,
        size,
        stride,
        offset,
        writeline: Callable[..., None],
        dtype=None,
    ) -> str:
        """Returns a newly-created, temporary RAII tensor handle containing the
        reinterpreted tensor data.  Callers of this function are responsible for saving
        the handle if persistent access is needed."""

        d_size, d_stride, d_offset, d_dtype, collapsible = (
            codegen_reinterpret_view_helper(data)
        )

        dim = str(len(size))
        original_offset = offset
        offset = self.codegen_sizevar(offset)
        call_strs = []
        final_tensor_str = None

        def create_reinterpret_call() -> str:
            args = [
                f"{data.get_name()}",
                dim,
                self.codegen_int_array_var(
                    self.codegen_shape_tuple(size),
                    writeline,
                    known_statically=self.is_statically_known_list_of_ints(size),
                    graph=self.get_codegened_graph(),
                ),
                self.codegen_int_array_var(
                    self.codegen_shape_tuple(stride),
                    writeline,
                    known_statically=self.is_statically_known_list_of_ints(stride),
                    graph=self.get_codegened_graph(),
                ),
                offset,
            ]
            return f"wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper({', '.join(args)}))"

        def create_dtypeview_call(reinterpret_call: str) -> tuple[str, list[str]]:
            tmp_AtenTensorHandle = f"tmp_{data.get_name()}_{next(self.tmp_tensor_id)}"
            tmp_call_strs = [f"AtenTensorHandle {tmp_AtenTensorHandle};"]
            device_name = data.layout.device.type
            dtypeview_function = f"aoti_torch_{device_name}_view_dtype"
            tmp_call_strs.append(
                f"AOTI_TORCH_ERROR_CODE_CHECK({dtypeview_function}"
                f"({reinterpret_call}, {self.codegen_dtype(dtype)}, &{tmp_AtenTensorHandle}));"
            )
            return f"RAIIAtenTensorHandle({tmp_AtenTensorHandle})", tmp_call_strs

        def create_new_tensor_handle() -> tuple[str, list[str]]:
            tmp_AtenTensorHandle = f"tmp_{data.get_name()}_{next(self.tmp_tensor_id)}"
            tmp_call_strs = [
                f"AtenTensorHandle {tmp_AtenTensorHandle};",
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_new_tensor_handle({data.get_name()}, &{tmp_AtenTensorHandle}));",
            ]
            return f"RAIIAtenTensorHandle({tmp_AtenTensorHandle})", tmp_call_strs

        collapsed = collapsible and original_offset == d_offset
        if collapsed:
            same_layout = size == d_size and stride == d_stride
            base_dtype = d_dtype
        else:
            same_layout = (
                size == data.layout.size
                and stride == data.layout.stride
                and original_offset == data.layout.offset
            )
            base_dtype = data.dtype

        if same_layout:
            # pure dtypeview
            if dtype is not None and dtype != base_dtype:
                final_tensor_str, tmp_call_strs = create_dtypeview_call(data.get_name())
            else:
                final_tensor_str, tmp_call_strs = create_new_tensor_handle()
            call_strs.extend(tmp_call_strs)
        else:
            # firstly create reinterpretview
            final_tensor_str = create_reinterpret_call()
            if dtype is not None and dtype != base_dtype:
                # wrap it with dtypeview
                final_tensor_str, tmp_call_strs = create_dtypeview_call(
                    final_tensor_str
                )
                call_strs.extend(tmp_call_strs)

        for line in call_strs:
            writeline(line)

        # NB, the return handle here represents a temporary tensor, which will be automatically
        # released.
        # Here's a sample usage in the cpp wrapper code:
        # ```
        # aoti_torch_addmm_out(
        #     buf1,
        #     arg1_1,
        #     RAIIAtenTensorHandle(tmp_tensor_handle_0),
        #     buf0,
        #     1L,
        #     1L));
        # ```
        # RAIIAtenTensorHandle(tmp_tensor_handle_0) will be released after the call to addmm_out.
        # This could be problematic when it's used in a different pattern, for example:
        # ````
        # AtenTensorHandle tensor_args[] = {RAIIAtenTensorHandle(tmp_tensor_handle_2), buf5, buf6};
        # aoti_torch_proxy_executor_call_function(..., tensor_args);
        # ````
        # RAIIAtenTensorHandle(tmp_tensor_handle_2) will be invalid when it's used in the latter
        # kernel call.
        #
        # This is solved by updating the proxy_executor invocation to
        # ```
        # aoti_torch_proxy_executor_call_function(...,
        #     std::array<AtenTensorHandle, 3>{
        #         RAIIAtenTensorHandle(tmp_tensor_handle_2), buf5, buf6
        #     }.cbegin()
        # );
        # ```
        return final_tensor_str

    def codegen_device_copy(self, src, dst, non_blocking: bool | str):
        """This function is overridden by cpp_wrapper_cpu_array_ref, so we don't need to
        handle cases where dst is not an AtenTensorHandle."""
        self.writeline(
            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_copy_({dst}, {src}, {non_blocking}));"
        )

    def sync_d2h_copy(self, buffer_name: str) -> None:
        # TODO: add AOTI C API for event-based D2H copy synchronization
        pass

    def codegen_multi_output(self, node: ir.MultiOutput):
        # in the abi_compatible mode, outputs are retrieved by passing
        # output pointers, so we skip its codegen here.
        pass

    def codegen_subgraph_prefix(self, subgraph, outer_inputs, outer_outputs):
        if len(subgraph.graph.graph_inputs) != len(outer_inputs):
            raise AssertionError(
                f"expected len(subgraph.graph.graph_inputs) == len(outer_inputs), "
                f"got {len(subgraph.graph.graph_inputs)} and {len(outer_inputs)}"
            )

        for (inner_input, inner_input_val), outer_input in zip(
            subgraph.graph.graph_inputs.items(), outer_inputs
        ):
            if not isinstance(inner_input_val, ir.TensorBox):
                continue

            # in ABI-compatible mode, we copy the underlying at::Tensor of the conditional
            # input (outer_input) into another at::Tensor to be used as a subgraph input
            # (inner_input) in the nested scope. we can't std::move here, as the codegened
            # outer input may be an expression / rvalue (e.g., reinterpret_view(x)), so we
            # can't necessarily std::move it back to the origin (x).
            self.writeline(f"AtenTensorHandle {inner_input}_handle;")
            self.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_assign_tensors_out({outer_input}, &{inner_input}_handle));"
            )
            self.writeline(f"RAIIAtenTensorHandle {inner_input}({inner_input}_handle);")

    def codegen_subgraph_suffix(self, subgraph, outer_inputs, outer_outputs):
        for inner_output, outer_output in zip(
            subgraph.graph.graph_outputs, outer_outputs
        ):
            src = inner_output.codegen_reference()
            if not isinstance(inner_output, ir.ShapeAsConstantBuffer):
                # in ABI-compatible mode, we need to std::move subgraph output (inner_output)
                # to the conditional output (outer_output), as RAIIAtenTensorHandle's copy
                # constructor is deleted.
                src = f"std::move({src})"
                # in case the outer_output carried a value
                # before (e.g., in the while_loop codegen)
                self.writeline(f"{outer_output}.reset();")
            self.writeline(f"{outer_output} = {src};")

    @staticmethod
    def _get_while_loop_carried_output_names(while_loop):
        mutation_output_names = {
            mutation_name: mutation_output.get_name()
            for mutation_output in while_loop.mutation_outputs
            for mutation_name in mutation_output.get_mutation_names()
        }

        def resolve_mutation_output_name(name):
            seen = OrderedSet()
            while name in mutation_output_names:
                if name in seen:
                    raise AssertionError("while_loop mutation outputs contain a cycle")
                seen.add(name)
                name = mutation_output_names[name]
            return name

        output_names = []
        output_idx = 0
        for carried_input in while_loop.carried_inputs:
            carried_input_name = carried_input.get_name()
            mutation_output_name = resolve_mutation_output_name(carried_input_name)
            if mutation_output_name != carried_input_name:
                output_names.append(mutation_output_name)
                continue

            if output_idx >= len(while_loop.outputs):
                raise AssertionError(
                    "while_loop has fewer non-mutated outputs than carried inputs"
                )
            output_names.append(while_loop.outputs[output_idx].get_name())
            output_idx += 1

        if output_idx != len(while_loop.outputs):
            raise AssertionError("while_loop has unused non-mutated outputs")

        return output_names

    def codegen_invoke_subgraph(self, invoke_subgraph):
        raise NotImplementedError(
            "codegen invoke_subgraph is not implemented for cpp wrapper"
        )

    def codegen_switch(self, node) -> None:
        """Emit ABI-compatible C++ for a higher-order cond/switch."""
        outer_inputs = [f"{buf.codegen_reference()}" for buf in node.operands]
        outer_outputs = []
        for out in node.outputs:
            # in ABI-compatible mode, ir.MultiOutput is not codegened,
            # hence pre-declare output variables directly and separately
            self.writeline(f"RAIIAtenTensorHandle {out.get_name()};")
            outer_outputs.append(out.get_name())

        if not isinstance(node.selector, ir.ShapeAsConstantBuffer):
            # use the ABI shim to extract the scalar selector value from the Tensor;
            # always as int64 so the branch loop can use uniform index comparisons
            # (cond: True==1, False==0)
            selector_var = f"{node.selector.get_name()}_scalar"
            if selector_var not in self.used_switch_selector:
                self.codegen_tensor_item(
                    torch.int64,
                    node.selector.codegen_reference(),
                    selector_var,
                )
                self.used_switch_selector.add(selector_var)
        else:
            # ShapeAsConstantBuffer yields a C++ bool/int expression; Cast to int64_t.
            selector_var = f"static_cast<int64_t>({node.selector.codegen_reference()})"

        def _emit_branch(header: str, branch) -> None:
            self.writeline(header)
            with self._preserve_device_guard_state():
                self.writeline(EnterSubgraphLine(self, branch.graph))
                self.codegen_subgraph(branch, outer_inputs, outer_outputs)
                self.writeline(ExitSubgraphLine(self))

        def emit_switch_body() -> None:
            num_branches = len(node.branches)
            for b_idx, branch in enumerate(node.branches):
                if b_idx == 0:
                    header = f"if ({selector_var} == 0) {{"
                elif b_idx < num_branches - 1:
                    header = f"}} else if ({selector_var} == {b_idx}) {{"
                else:
                    header = "} else {"
                _emit_branch(header, branch)
            self.writeline("}")

        if not V.graph.is_dual_wrapper_mode:
            return emit_switch_body()

        graphs = tuple(branch.graph for branch in node.branches)
        jit_code = IndentedBuffer(initial_indent=self.wrapper_call._indent)
        with (
            self._preserve_codegen_state(graphs),
            self._target_buf("wrapper_call", jit_code),
            self.set_writeline(jit_code, jit_code.writeline_jit),
        ):
            # JIT pass: visit each branch once for lazy kernel autotune
            for branch in node.branches:
                self.writeline("{")
                with self._preserve_device_guard_state():
                    self.writeline(EnterSubgraphLine(self, branch.graph))
                    self.codegen_subgraph(branch, outer_inputs, outer_outputs)
                    self.writeline(ExitSubgraphLine(self))
                self.writeline("}")

        aot_code = AotOnlyBuffer(initial_indent=self.wrapper_call._indent)
        with (
            self._target_buf("wrapper_call", aot_code),
            self.set_writeline(aot_code, aot_code.writeline_aot),
        ):
            emit_switch_body()

        self.writeline(DualWrapperCodeLine(jit_code, aot_code))

    def codegen_subgraph(self, subgraph, outer_inputs, outer_outputs):
        # TODO (desertfire) - This function is the old way of supporting
        # subgraph codegen by inlining subgraphs in the output code. For python
        # wrapper, we have moved to lifting subgraphs as functions, supported by
        # PythonWrapperCode `codegen_subgraph` function. We should perhaps
        # support lifting of subgraphs as functions for cpp wrapper as well.
        try:
            self.push_codegened_graph(subgraph.graph)
            self.writeline(f"// subgraph: {subgraph.name}")
            self.codegen_subgraph_prefix(subgraph, outer_inputs, outer_outputs)
            parent_graph = V.graph
            with V.set_graph_handler(subgraph.graph):
                subgraph.graph.codegen_subgraph(
                    parent_graph=parent_graph,
                )
            self.codegen_subgraph_suffix(subgraph, outer_inputs, outer_outputs)
        finally:
            self.pop_codegened_graph()

    def codegen_while_loop(self, while_loop, stack_output=False):
        """Emit ABI-compatible C++ for a higher-order while_loop.

        In dual-wrapper mode, keep the real loop on the AOTI side while making
        the JIT first pass visit the condition and body once for lazy autotune.
        """
        if stack_output:
            raise NotImplementedError("NYI cpp wrapper for while_loop_stack_output")
        is_bool_pred = isinstance(
            while_loop.cond_subgraph.graph.graph_outputs[0], ir.ShapeAsConstantBuffer
        )
        name = while_loop.get_name()
        outer_carried_inputs = [
            buf.codegen_reference() for buf in while_loop.carried_inputs
        ]
        outer_additional_inputs = [
            buf.codegen_reference() for buf in while_loop.additional_inputs
        ]
        carried_output_names = self._get_while_loop_carried_output_names(while_loop)
        cond_result_name = f"{name}_cond_result"
        if is_bool_pred:
            self.writeline(f"bool {cond_result_name};")
        else:
            self.writeline(f"RAIIAtenTensorHandle {cond_result_name};")

        cond_outer_inputs = []
        for inp, out_name in zip(outer_carried_inputs, carried_output_names):
            # in ABI-compatible mode, the carried inputs are codegened
            # as buffers outside the while loop and set to the initial
            # values. at the end of each while_loop iteration, they
            # will be assigned the carried values.
            self.writeline(f"AtenTensorHandle {out_name}_handle;")
            self.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_assign_tensors_out({inp}, &{out_name}_handle));"
            )
            self.writeline(f"RAIIAtenTensorHandle {out_name}({out_name}_handle);")
            cond_outer_inputs.append(out_name)

        # additional inputs will be assigned within the while_loop
        # iteration directly from the corresponding outer graph buffers
        cond_outer_inputs.extend(outer_additional_inputs)

        cond_outer_outputs = [cond_result_name]
        body_outer_inputs = list(cond_outer_inputs)
        body_outer_outputs = body_outer_inputs[: len(outer_carried_inputs)]

        def codegen_original_while_loop() -> None:
            self.writeline("while (1) {")
            with self._preserve_device_guard_state():
                self.writeline(EnterSubgraphLine(self, while_loop.cond_subgraph.graph))
                self.codegen_subgraph(
                    while_loop.cond_subgraph, cond_outer_inputs, cond_outer_outputs
                )

                if is_bool_pred:
                    cond_result = f"{cond_result_name}"
                else:
                    cond_result = f"{cond_result_name}_scalar"
                    self.codegen_tensor_item(torch.bool, cond_result_name, cond_result)
                self.writeline(f"if (!{cond_result}) break;")

                self.writeline(ExitSubgraphLine(self))
                self.writeline(EnterSubgraphLine(self, while_loop.body_subgraph.graph))
                self.codegen_subgraph(
                    while_loop.body_subgraph, body_outer_inputs, body_outer_outputs
                )
                self.writeline(ExitSubgraphLine(self))
                self.writeline("}")

        if not V.graph.is_dual_wrapper_mode:
            return codegen_original_while_loop()

        graphs = (
            while_loop.cond_subgraph.graph,
            while_loop.body_subgraph.graph,
        )
        jit_code = IndentedBuffer(initial_indent=self.wrapper_call._indent)
        with (
            self._preserve_codegen_state(graphs),
            self._target_buf("wrapper_call", jit_code),
            self.set_writeline(jit_code, jit_code.writeline_jit),
        ):
            self.writeline("{")
            with self._preserve_device_guard_state():
                self.writeline(EnterSubgraphLine(self, while_loop.cond_subgraph.graph))
                self.codegen_subgraph(
                    while_loop.cond_subgraph, cond_outer_inputs, cond_outer_outputs
                )
                self.writeline(ExitSubgraphLine(self))
                self.writeline(EnterSubgraphLine(self, while_loop.body_subgraph.graph))
                self.codegen_subgraph(
                    while_loop.body_subgraph, body_outer_inputs, body_outer_outputs
                )
                self.writeline(ExitSubgraphLine(self))
            self.writeline("}")

        aot_code = AotOnlyBuffer(initial_indent=self.wrapper_call._indent)
        with (
            self._target_buf("wrapper_call", aot_code),
            self.set_writeline(aot_code, aot_code.writeline_aot),
        ):
            codegen_original_while_loop()

        self.writeline(DualWrapperCodeLine(jit_code, aot_code))

    def generate_extern_kernel_args_decl_if_needed(
        self,
        op_overload: torch._ops.OpOverload | torch._ops.HigherOrderOperator,
        raw_args: Sequence[Any],
        output_args: _OUTPUT_ARGS_TYPE,
        raw_outputs: Sequence[ir.Buffer],
    ):
        """
        Generates declarations for external kernel arguments if needed, based on the provided
        operator and its arguments. It processes both input and output arguments, categorizing
        them into tensor and integer arguments for further code generation.
        """
        schema = None
        if isinstance(op_overload, torch._higher_order_ops.torchbind.CallTorchBind):
            obj = raw_args[0]
            method = raw_args[1]
            schema = op_overload.schema(obj, method)
        else:
            if not isinstance(op_overload, torch._ops.OpOverload):
                raise AssertionError(type(op_overload))
            schema = op_overload._schema
        if schema is None:
            raise AssertionError("expected schema to be set")
        arg_types = [x.real_type for x in schema.arguments]
        return_types = [x.type for x in schema.returns]

        new_tensor_args = []
        new_int_args = []

        def fill_args(arg, arg_type):
            static_arg_types = (
                torch.FloatType,
                torch.BoolType,
                torch.StringType,
                torch.Type,
                torch.DeviceObjType,
            )
            inductor_tensor_buffers = (
                ir.Buffer,
                ir.ReinterpretView,
            )

            if isinstance(arg_type, torch.TensorType):
                if not isinstance(arg, inductor_tensor_buffers):
                    raise AssertionError(f"got {type(arg)}")
                new_tensor_args.append(f"{arg.codegen_reference()}")
            elif isinstance(arg_type, torch.IntType):
                # int
                new_int_args.append(str(arg))
            elif isinstance(arg_type, torch.SymIntType):
                # SymInt
                expr = arg.node.expr if isinstance(arg, torch.SymInt) else arg
                new_int_args.append(cexpr(expr))
            elif isinstance(arg_type, torch.NumberType):
                # Scalar of type int
                if not isinstance(arg, (int, float, bool)):
                    raise AssertionError(
                        f"expected arg to be int, float, or bool, got {type(arg)}"
                    )
                # Only treat int Scalar as dynamic
                if isinstance(arg, int):
                    new_int_args.append(str(arg))
            elif isinstance(arg, ir.TorchBindObject):
                # torchbind objects are loaded in proxy executor
                pass
            elif isinstance(arg_type, torch.ListType):
                if not isinstance(arg, (list, tuple)):
                    raise AssertionError(
                        f"expected arg to be list or tuple, got {type(arg)}"
                    )

                # List[Tensor]
                if isinstance(arg_type.getElementType(), torch.TensorType):
                    new_tensor_args.extend([f"{a.codegen_reference()}" for a in arg])
                # List[Optional[Tensor]]
                elif isinstance(
                    arg_type.getElementType(), torch.OptionalType
                ) and isinstance(
                    arg_type.getElementType().getElementType(), torch.TensorType
                ):
                    new_tensor_args.extend(
                        [f"{a.codegen_reference()}" for a in arg if a is not None]
                    )
                # List[int]
                elif isinstance(arg_type.getElementType(), torch.IntType):
                    new_int_args.extend([str(a) for a in arg])
                # List[SymInt]
                elif isinstance(arg_type.getElementType(), torch.SymIntType):
                    expressions = [
                        a.node.expr if isinstance(a, torch.SymInt) else a for a in arg
                    ]
                    new_int_args.extend([cexpr(expr) for expr in expressions])
                # List[Scalar]
                elif isinstance(arg_type.getElementType(), torch.NumberType):
                    # Only treat int Scalar as dynamic
                    is_int_type = [isinstance(a, int) for a in arg]
                    if any(is_int_type):
                        if not all(is_int_type):
                            raise AssertionError(
                                "AOTInductor only supports int scalars of the same type"
                            )
                        new_int_args.extend([str(a) for a in arg])
                else:
                    if not isinstance(
                        arg_type.getElementType(),
                        static_arg_types,  # type: ignore[arg-type]
                    ):
                        raise AssertionError(
                            f"Fall through arguments must be one of static_arg_types, got {type(arg_type)}"
                        )
            else:
                if not isinstance(
                    arg_type,
                    static_arg_types,  # type: ignore[arg-type]
                ):
                    raise AssertionError(
                        f"Fall through arguments must be one of static_arg_types, got {type(arg_type)}"
                    )

        for arg, arg_type in zip(raw_args, arg_types):
            if arg is not None:
                if isinstance(arg_type, torch.OptionalType):
                    fill_args(arg, arg_type.getElementType())
                else:
                    fill_args(arg, arg_type)

        def fill_output_arg(
            arg: str, return_type: torch.JitType, is_mutated_output: bool
        ) -> None:
            if isinstance(return_type, torch.TensorType):
                if not is_mutated_output:
                    self.writeline(f"AtenTensorHandle {arg}_handle;  // output buffer")
                    self.writeline(
                        f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_new_uninitialized_tensor(&{arg}_handle));"
                    )
                    self.writeline(f"RAIIAtenTensorHandle {arg}({arg}_handle);")
                new_tensor_args.append(f"{arg}")
            elif isinstance(return_type, torch.SymIntType):
                raise NotImplementedError("NYI support for return type: SymInt")
            elif isinstance(return_type, torch.ListType) and isinstance(
                return_type.getElementType(), torch.SymIntType
            ):
                raise NotImplementedError("NYI support for return type: List[SymInt]")
            else:
                raise AssertionError(f"Unsupported return type found: {return_type}")

        # TODO: Only support None and tensor(s) returns for now, SymInt is not implemented yet
        for return_type in return_types:
            if isinstance(
                return_type, (torch.TensorType, torch.NoneType, torch.IntType)
            ):
                pass
            elif isinstance(return_type, torch.OptionalType):
                if not isinstance(return_type.getElementType(), torch.TensorType):
                    raise AssertionError(
                        f"expected Optional element type to be TensorType, "
                        f"got {return_type.getElementType()}"
                    )
            elif isinstance(return_type, torch.ListType):
                if not isinstance(return_type.getElementType(), torch.TensorType):
                    raise AssertionError(
                        f"expected List element type to be TensorType, "
                        f"got {return_type.getElementType()}"
                    )
            else:
                raise NotImplementedError(
                    f"return type {return_type} is not yet supported."
                )

        for output_arg, raw_output_arg in zip(output_args, raw_outputs):  # type: ignore[arg-type]
            # None output is supported, but Optional return types are not yet supported
            if output_arg is None:
                continue
            elif isinstance(raw_output_arg, int):
                new_int_args.append(str(raw_output_arg))
            elif isinstance(output_arg, list):
                for out in output_arg:
                    if out is None:
                        raise AssertionError(out)
                    fill_output_arg(
                        out,
                        torch.TensorType.get(),
                        isinstance(raw_output_arg, ir.MutationOutput),
                    )
            else:
                fill_output_arg(
                    output_arg,
                    torch.TensorType.get(),
                    isinstance(raw_output_arg, ir.MutationOutput),
                )

        return new_tensor_args, new_int_args

    @staticmethod
    def _compatible_with_stableivalue(op: torch._ops.OpOverload) -> bool:
        """Returns true if op_overload._schema only utilizes types supported by the AOT
        C-shim *internal* function to_ivalue.  to_ivalue is an implementation detail, so
        these types are not guaranteed to be supported long-term.  When generating code
        for cpp_wrapper mode, we don't have to be forward-compatible, so changing this
        function's implementation in future is fine."""
        supported_types = (
            torch.BoolType,
            torch.DeviceObjType,
            torch.FloatType,
            # ScalarTypeType, LayoutType, and MemoryFormatType are seen as IntType
            # when queried via torch.JitType.type.
            torch.IntType,
            torch.TensorType,
        )

        def type_supported(t: torch.JitType) -> bool:
            if isinstance(t, torch.OptionalType):
                return type_supported(t.getElementType())
            return isinstance(t, supported_types)

        def uses_symint(t: torch.JitType) -> bool:
            # SymInt/SymBool/SymFloat are reported as Int/Bool/Float by
            # JitType.type, so they pass type_supported above.  But the
            # StableIValue codegen below emits no symbolic-int-aware conversion,
            # so route such ops to the boxed dispatch path, which handles
            # c10::SymInt correctly.  real_type preserves the symbolic types.
            if isinstance(t, (torch.OptionalType, torch.ListType)):
                return uses_symint(t.getElementType())
            return (
                isinstance(t, (torch.SymIntType, torch.SymBoolType))
                or repr(t) == "SymFloat"
            )

        return all(
            type_supported(a.type) and not uses_symint(a.real_type)
            for a in chain(op._schema.arguments, op._schema.returns)
        )

    @staticmethod
    def _compatible_with_cpp_boxed_dispatch(op: torch._ops.OpOverload) -> bool:
        """Returns true if cpp_wrapper JIT can directly codegen c10::IValue boxing
        for this schema and can unpack the returned values into tensor handles.

        Unlike the StableIValue path, this is only used for JIT cpp_wrapper, where
        we can include ATen/c10 headers and call the dispatcher directly.
        """

        def arg_supported(t: torch.JitType) -> bool:
            if isinstance(t, torch.OptionalType):
                return arg_supported(t.getElementType())
            if isinstance(t, torch.ListType):
                return arg_supported(t.getElementType())
            if isinstance(t, torch.TupleType):
                return all(arg_supported(e) for e in t.elements())
            return isinstance(
                t,
                (
                    torch.BoolType,
                    torch.DeviceObjType,
                    torch.FloatType,
                    torch.IntType,
                    torch.NumberType,
                    torch.StringType,
                    torch.SymBoolType,
                    torch.SymIntType,
                    torch.TensorType,
                ),
            ) or repr(t) in ("Layout", "MemoryFormat", "ScalarType", "SymFloat")

        def return_supported(t: torch.JitType) -> bool:
            if isinstance(t, torch.OptionalType):
                return isinstance(t.getElementType(), torch.TensorType)
            return isinstance(t, (torch.NoneType, torch.TensorType))

        return all(arg_supported(a.real_type) for a in op._schema.arguments) and all(
            return_supported(r.real_type) for r in op._schema.returns
        )

    def generate_fallback_kernel_with_runtime_lookup(
        self,
        buf_name: str,
        python_kernel_name: str,
        get_args: Callable[[], Sequence[str]],
        op_overload: torch._ops.OpOverload | torch._ops.HigherOrderOperator,
        raw_args: Sequence[Any],
        outputs: Sequence[ir.Buffer],
    ) -> None:
        """Generate a call to a kernel not contained in the C-shim.  This results in
        different code paths for AOT Inductor vs cpp_wrapper Inductor mode."""

        def extract_output_name(
            out: ir.Buffer | Sequence[ir.Buffer] | None,
        ) -> str | None | _OUTPUT_ARGS_TYPE:
            if out is None:
                return None
            if isinstance(
                out, (ir.MultiOutput, ir._CollectiveKernel, ir.FallbackKernel)
            ):
                return out.get_name()
            if isinstance(out, ir.MutationOutput):
                mutated_buf_names = out.get_mutation_names()
                if not (
                    isinstance(mutated_buf_names, list) and len(mutated_buf_names) == 1
                ):
                    raise AssertionError(
                        "Expect only one mutated buffer in MutationOutput"
                    )
                return mutated_buf_names[0]
            if isinstance(out, (list, tuple)):
                return [extract_output_name(o) for o in out]  # type: ignore[misc]
            if isinstance(out, int):
                return str(out)
            raise AssertionError(f"Unexpected output: {type(out)}")

        if isinstance(op_overload, torch._ops.HigherOrderOperator):
            if not isinstance(
                op_overload, torch._higher_order_ops.torchbind.CallTorchBind
            ):
                raise AssertionError(type(op_overload))
            if len(raw_args) <= 1:
                raise AssertionError(f"expected len(raw_args) > 1, got {len(raw_args)}")
            obj = raw_args[0]
            method = raw_args[1]
            return_schema = op_overload.schema(obj, method).returns
        else:
            return_schema = op_overload._schema.returns

        # output_args has the same pytree structure as outputs
        if not return_schema:
            # kernel does not return a value
            output_args: _OUTPUT_ARGS_TYPE = []
        elif isinstance(output_name := extract_output_name(outputs), str):
            output_args = [output_name]
        else:
            # If the schema indicates a return value, we should have a non-None value by
            # this point.
            if not isinstance(output_name, list):
                raise AssertionError(type(output_name))
            output_args = output_name

        # In AOT mode, we use a ProxyExecutor to run fallback kernels.
        if V.graph.aot_mode and V.graph.is_dual_wrapper_mode:
            jit_code = IndentedBuffer(initial_indent=self.wrapper_call._indent)
            with (
                self._target_buf("wrapper_call", jit_code),
                self.set_writeline(jit_code, jit_code.writeline_jit),
            ):
                self._generate_fallback_kernel_jit(
                    buf_name,
                    python_kernel_name,
                    get_args,
                    op_overload,
                    raw_args,
                    output_args,
                    outputs,
                )

            aot_code = AotOnlyBuffer(initial_indent=self.wrapper_call._indent)
            with (
                self._target_buf("wrapper_call", aot_code),
                self.set_writeline(aot_code, aot_code.writeline_aot),
            ):
                self.generate_fallback_kernel_with_runtime_lookup_aot(
                    op_overload,
                    raw_args,
                    output_args,
                    outputs,
                )

            self.writeline(DualWrapperCodeLine(jit_code, aot_code))
            return

        if V.graph.aot_mode:
            self.generate_fallback_kernel_with_runtime_lookup_aot(
                op_overload,
                raw_args,
                output_args,
                outputs,
            )

        if not V.graph.aot_mode:
            self._generate_fallback_kernel_jit(
                buf_name,
                python_kernel_name,
                get_args,
                op_overload,
                raw_args,
                output_args,
                outputs,
            )

    def _generate_fallback_kernel_jit(
        self,
        buf_name,
        python_kernel_name,
        get_args,
        op_overload,
        raw_args,
        output_args,
        outputs,
    ):
        """Emit JIT-mode fallback kernel code (no-Python dispatcher or Python callback)."""
        if not isinstance(op_overload, torch._ops.OpOverload):
            raise AssertionError(type(op_overload))
        for output in output_args:
            if not (output is None or isinstance(output, str)):
                raise AssertionError(
                    "fallback kernels with runtime lookup currently only support tensor "
                    "returns, not more complicated types (such as list-of-list-of-tensor)"
                )

        # In non-AOT mode, we use aoti_torch_call_dispatcher if all the inputs and
        # outputs of the op can be represented with StableIValue.  This avoids the
        # overhead of calling back into Python, and covers most remaining fallback ops.
        if self._compatible_with_stableivalue(op_overload):
            self.generate_fallback_kernel_with_runtime_lookup_nopython(
                get_args,
                op_overload,
                output_args,  # type: ignore[arg-type]
                outputs,
            )
            return

        # For cpp_wrapper JIT, fall back to direct boxed dispatcher calls for
        # schemas the StableIValue bridge cannot represent, such as Tensor[].
        if self._compatible_with_cpp_boxed_dispatch(op_overload):
            self.generate_fallback_kernel_with_runtime_lookup_boxed(
                op_overload,
                raw_args,
                output_args,  # type: ignore[arg-type]
                outputs,
            )
            return

        # Otherwise, we call back into Python, which has some extra runtime overhead,
        # but handles situations like list[Tensor] (currently unrepresentable via
        # StableIValue).
        self.generate_fallback_kernel_with_runtime_lookup_python(
            buf_name,
            python_kernel_name,
            op_overload,
            raw_args,
            output_args,  # type: ignore[arg-type]
            outputs,
        )

    def generate_scoped_gil_acquire(self, declarations_before_scope, lines_in_scope):
        scoped_lines = IndentedBuffer()
        for declaration in declarations_before_scope:
            scoped_lines.writeline(declaration)

        scoped_lines.writeline("{")
        with scoped_lines.indent():
            scoped_lines.writeline("py::gil_scoped_acquire_simple acquire;")
            scoped_lines.writelines(lines_in_scope.split("\n"))
        scoped_lines.writelines("}")
        return scoped_lines._lines

    def load_custom_op_wrapper(self):
        # TODO: need to support control flow
        if self.custom_op_wrapper_loaded:
            return

        lines = """
RAIIPyObject codecache_module(PyImport_ImportModule("torch._inductor.codecache"));
if (!codecache_module) {
    throw std::runtime_error("Failed to load torch._inductor.codecache");
}
custom_op_wrapper = PyObject_GetAttrString(codecache_module, "custom_op_wrapper");
if (!custom_op_wrapper) {
    throw std::runtime_error("Failed to load torch._inductor.codecache.custom_op_wrapper");
}"""

        declarations_before_scope = ["RAIIPyObject custom_op_wrapper;"]
        scope_gil_acquire = self.generate_scoped_gil_acquire(
            declarations_before_scope, lines
        )
        self.writelines(scope_gil_acquire)

        self.custom_op_wrapper_loaded = True

    def generate_float_value(self, val):
        if not isinstance(val, float):
            raise AssertionError(f"expected val to be float, got {type(val)}")
        if val == float("inf"):
            return "std::numeric_limits<double>::infinity()"
        elif val == float("-inf"):
            return "-std::numeric_limits<double>::infinity()"
        elif math.isnan(val):
            return "std::numeric_limits<double>::quiet_NaN()"
        else:
            return f"{val}"

    def generate_py_arg(self, py_args_var, idx, raw_arg, arg_type):
        """Generate C++ code that converts a single operator argument into a Python
        object and inserts it into a PyTuple at the given index."""

        def generate_py_arg_inner(lines, raw_arg, arg_type):
            def handle_scalar(scalar):
                if isinstance(scalar, bool):
                    return f"PyBool_FromLong({1 if scalar else 0})"
                if isinstance(scalar, int):
                    return f"PyLong_FromLongLong({scalar})"
                if isinstance(scalar, float):
                    return f"PyFloat_FromDouble({self.generate_float_value(scalar)})"
                if isinstance(scalar, complex):
                    real = self.generate_float_value(scalar.real)
                    imag = self.generate_float_value(scalar.imag)
                    return f"PyComplex_FromDoubles({real}, {imag})"
                if isinstance(scalar, SymTypes):
                    scalar_var = cexpr(scalar.node.expr)
                    if isinstance(scalar, torch.SymBool):
                        return f"PyBool_FromLong({scalar_var})"
                    if isinstance(scalar, torch.SymFloat):
                        return f"PyFloat_FromDouble({scalar_var})"
                    return f"PyLong_FromLongLong({scalar_var})"
                raise NotImplementedError(
                    f"scalar {scalar}, {type(scalar)} cannot be handled by handle_scalar"
                )

            is_any = isinstance(arg_type, torch.AnyType)

            def handle_any(types: type | tuple[type, ...]):
                return is_any and isinstance(raw_arg, types)

            if raw_arg is None:
                # Py_None is a singleton, so we have to explicitly incref it here
                lines.append("Py_INCREF(Py_None);\n")
                return "Py_None"
            elif isinstance(arg_type, torch.TensorType) or handle_any(ir.IRNode):
                # In some cases, scalar arguments may be passed in place of tensors.
                if not hasattr(raw_arg, "codegen_reference"):
                    return handle_scalar(raw_arg)

                # Store AtenTensorHandle as void*.  All Python args are constructed in a
                # nested scope, so this handle will self-destruct after the function
                # call.
                base_handle = self.create_tmp_raii_handle_var_if_needed(
                    raw_arg.codegen_reference(), lines
                )
                return f"PyCapsule_New(reinterpret_cast<void*>({base_handle}.get()), NULL, NULL)"
            elif isinstance(arg_type, torch.OptionalType):
                return generate_py_arg_inner(lines, raw_arg, arg_type.getElementType())
            elif isinstance(arg_type, torch.BoolType) or handle_any(bool):
                return f"PyBool_FromLong({1 if raw_arg else 0})"
            elif isinstance(arg_type, torch.IntType) or handle_any(int):
                # int
                return f"PyLong_FromLongLong({raw_arg})"
            elif isinstance(arg_type, torch.SymIntType) or handle_any(torch.SymInt):
                # SymInt
                expr = (
                    raw_arg.node.expr if isinstance(raw_arg, torch.SymInt) else raw_arg
                )
                return f"PyLong_FromLongLong({cexpr(expr)})"
            elif isinstance(arg_type, torch.FloatType) or handle_any(float):
                return f"PyFloat_FromDouble({self.generate_float_value(raw_arg)})"
            elif isinstance(arg_type, torch.StringType) or handle_any(str):
                return f'PyUnicode_FromString("{raw_arg}")'
            elif isinstance(arg_type, torch.NumberType) or handle_any(
                (*SymTypes, torch.types.Number, complex)
            ):
                # Union[bool, int, float, complex]
                # torch/_prims_common/__init__.py
                return handle_scalar(raw_arg)
            elif isinstance(raw_arg, torch.device):
                device_str, device_index = self.codegen_device(raw_arg).split(", ")
                return f"torch_python_thp_device_new({device_str}, {device_index})"
            elif isinstance(raw_arg, torch.dtype):
                return f"torch_python_get_thp_dtype({self.codegen_dtype(raw_arg)})"
            elif isinstance(raw_arg, torch.layout):
                return f"torch_python_get_thp_layout({self.codegen_layout(raw_arg)})"
            elif isinstance(raw_arg, torch.memory_format):
                return (
                    "torch_python_get_thp_memory_format("
                    f"{self.codegen_memory_format(raw_arg)})"
                )
            else:
                raise NotImplementedError(
                    f"arg type {arg_type} is not yet supported by custom_op_wrapper"
                )

        def handle_sequence_arg(raw_arg_, arg_type_, lines_):
            if not isinstance(raw_arg_, (list, tuple)):
                raise AssertionError(str(raw_arg) + " is not a list")
            lines_.append(
                f"PyObject* {py_args_var}_{idx} = PyList_New({len(raw_arg)});\n"
            )
            for i, elem in enumerate(raw_arg_):
                lines_.append(
                    f"PyList_SetItem({py_args_var}_{idx}, {i}, {generate_py_arg_inner(lines, elem, arg_type_.getElementType())});\n"
                )
            lines_.append(
                f"PyTuple_SetItem({py_args_var}, {idx}, {py_args_var}_{idx});\n"
            )

        lines = []
        if isinstance(arg_type, torch.ListType):
            handle_sequence_arg(raw_arg, arg_type, lines)
        elif isinstance(arg_type, torch.OptionalType) and isinstance(
            arg_type.getElementType(), torch.ListType
        ):
            if raw_arg is None:
                lines.append(
                    f"PyTuple_SetItem({py_args_var}, {idx}, {generate_py_arg_inner(lines, raw_arg, arg_type)});\n"
                )
            else:
                handle_sequence_arg(raw_arg, arg_type.getElementType(), lines)
        else:
            lines.append(
                f"PyTuple_SetItem({py_args_var}, {idx}, {generate_py_arg_inner(lines, raw_arg, arg_type)});\n"
            )
        return "".join(lines)

    def generate_fallback_kernel_with_runtime_lookup_nopython(
        self,
        get_args: Callable[[], Sequence[str]],
        op_overload: torch._ops.OpOverload,
        output_args: Sequence[str | None],
        raw_outputs: Sequence[ir.Buffer],
    ) -> None:
        """Generate fallback kernel calls with runtime (non-AOT) dispatch.  This can
        only be called in cpp_wrapper mode, and assumes that the input is a non-None
        OpOverload.

        In the future, we may switch over to directly calling c10::Dispatcher if we need
        to support more datatypes."""
        if raw_outputs:
            declarations_before_scope = [
                f"RAIIAtenTensorHandle {output_arg};"
                for output_arg, raw_output_arg in zip(output_args, raw_outputs)  # type: ignore[arg-type]
                if output_arg is not None
                and not isinstance(raw_output_arg, ir.MutationOutput)
            ]
        else:
            declarations_before_scope = [
                f"RAIIAtenTensorHandle {output_arg};"
                for output_arg in output_args  # type: ignore[arg-type]
                if output_arg is not None
            ]

        dispatch_lines = IndentedBuffer()
        dispatch_lines.writelines(declarations_before_scope)
        dispatch_lines.writeline("{")

        with dispatch_lines.indent():
            tmp_var_number = count()

            def parse_arg(arg_type: torch.JitType, codegen_arg: str) -> str:
                # Strip off any temporary references; we're in an indented context, so
                # any saved-off variables will be auto-destroyed.
                new_codegen_arg = codegen_arg.removeprefix("&temporary_reference(")
                if new_codegen_arg != codegen_arg:
                    # If we removed temporary_reference, there's a good chance the
                    # variable ends with get() (which would retrieve an ATenTensorHandle
                    # from a temporary RAII handle).  Strip that off too, since we're
                    # going to save this in a temporary RAII handle.
                    if codegen_arg.endswith(".get())"):
                        codegen_arg = new_codegen_arg.removesuffix(".get())")
                    else:
                        codegen_arg = new_codegen_arg.removesuffix(")")

                if isinstance(arg_type, torch.OptionalType):
                    # If we have a pointer to a variable, strip it off and let
                    # from<std::optional> handle any internal pointers.
                    codegen_arg = codegen_arg.removeprefix("&")

                    if codegen_arg == "nullptr":
                        return "torch::stable::detail::from(std::nullopt)"

                    var_name = f"tmp_var_{next(tmp_var_number)}"
                    dispatch_lines.writeline(
                        f"std::optional {var_name}{{{parse_arg(arg_type.getElementType(), codegen_arg)}}};"
                    )
                    return f"torch::stable::detail::from({var_name})"

                raii_var = self.create_tmp_raii_handle_var_if_needed(
                    codegen_arg, dispatch_lines
                )
                temp_handle = raii_var != codegen_arg

                if isinstance(arg_type, torch.TensorType):
                    if not temp_handle:
                        # If the RAII tensor being referenced _isn't_ a temporary,
                        # scoped to this fallback call, then create a new handle
                        # referencing it which from<AtenTensorHandle> can steal.
                        var_name = f"tmp_var_{next(tmp_var_number)}"
                        dispatch_lines.writeline(f"AtenTensorHandle {var_name};")
                        dispatch_lines.writeline(
                            "AOTI_TORCH_ERROR_CODE_CHECK("
                            f"aoti_torch_new_tensor_handle({raii_var}, &{var_name}));"
                        )
                        return f"torch::stable::detail::from({var_name})"
                    # If the RAII tensor _is_ a temporary scoped to this fallback call,
                    # simply release and steal the handle.
                    return f"torch::stable::detail::from({raii_var}.release())"
                return f"torch::stable::detail::from({codegen_arg})"

            codegen_args = get_args()
            ivalue_args = (
                parse_arg(a.type, c)
                for a, c in zip(op_overload._schema.arguments, codegen_args)
            )
            array_len = max(len(codegen_args), len(output_args))
            dispatch_lines.writeline(
                f"std::array<StableIValue, {array_len}> dispatch_vars{{{', '.join(ivalue_args)}}};"
            )
            dispatch_lines.writeline("AOTI_TORCH_ERROR_CODE_CHECK(")
            with dispatch_lines.indent():
                dispatch_lines.writeline(
                    f'aoti_torch_call_dispatcher("{op_overload._schema.name}", "{op_overload._schema.overload_name}", dispatch_vars.data())'
                )
            dispatch_lines.writeline(");")

            # assign result(s), ignoring None
            for idx, output_arg in enumerate(output_args):
                if output_arg is None:
                    continue
                dispatch_lines.writeline(
                    f"{output_arg} = torch::stable::detail::to<AtenTensorHandle>(dispatch_vars[{idx}]);"
                )

        dispatch_lines.writeline("}")
        self.writelines(dispatch_lines.getvalue().splitlines())

    def generate_fallback_kernel_with_runtime_lookup_boxed(
        self,
        op_overload: torch._ops.OpOverload,
        raw_args: Sequence[Any],
        output_args: Sequence[str | None],
        raw_outputs: Sequence[ir.Buffer],
    ) -> None:
        """Generate a JIT-only fallback call through c10::Dispatcher.

        This covers custom C++ ops whose schemas are outside StableIValue's
        supported subset without routing through Python.
        """
        self.include_extra_header("ATen/core/dispatch/Dispatcher.h")
        self.include_extra_header("ATen/core/ivalue.h")
        self.include_extra_header("ATen/core/jit_type.h")
        self.include_extra_header("torch/csrc/inductor/aoti_torch/utils.h")

        if raw_outputs:
            declarations_before_scope = [
                f"RAIIAtenTensorHandle {output_arg};"
                for output_arg, raw_output_arg in zip(output_args, raw_outputs)
                if output_arg is not None
                and not isinstance(raw_output_arg, ir.MutationOutput)
            ]
        else:
            declarations_before_scope = [
                f"RAIIAtenTensorHandle {output_arg};"
                for output_arg in output_args
                if output_arg is not None
            ]

        dispatch_lines = IndentedBuffer()
        dispatch_lines.writelines(declarations_before_scope)
        dispatch_lines.writeline("{")

        with dispatch_lines.indent():
            tmp_var_number = count()

            def next_tmp(prefix: str) -> str:
                return f"{prefix}_{next(tmp_var_number)}"

            def type_expr(t: torch.JitType) -> str:
                if isinstance(t, torch.OptionalType):
                    return f"c10::OptionalType::create({type_expr(t.getElementType())})"
                if isinstance(t, torch.ListType):
                    return f"c10::ListType::create({type_expr(t.getElementType())})"
                if isinstance(t, torch.TupleType):
                    elems = ", ".join(type_expr(e) for e in t.elements())
                    var_name = next_tmp("tmp_type")
                    dispatch_lines.writeline(
                        f"std::vector<c10::TypePtr> {var_name}{{{elems}}};"
                    )
                    return f"c10::TupleType::create(std::move({var_name}))"
                if isinstance(t, torch.BoolType):
                    return "c10::BoolType::get()"
                if isinstance(t, torch.DeviceObjType):
                    return "c10::DeviceObjType::get()"
                if isinstance(t, torch.FloatType):
                    return "c10::FloatType::get()"
                if isinstance(t, torch.IntType):
                    return "c10::IntType::get()"
                if isinstance(t, torch.NumberType):
                    return "c10::NumberType::get()"
                if isinstance(t, torch.StringType):
                    return "c10::StringType::get()"
                if isinstance(t, torch.SymBoolType):
                    return "c10::SymBoolType::get()"
                if isinstance(t, torch.SymIntType):
                    return "c10::SymIntType::get()"
                if isinstance(t, torch.TensorType):
                    return "c10::TensorType::get()"
                type_repr = repr(t)
                if type_repr == "Layout":
                    return "c10::LayoutType::get()"
                if type_repr == "MemoryFormat":
                    return "c10::MemoryFormatType::get()"
                if type_repr == "ScalarType":
                    return "c10::ScalarTypeType::get()"
                if type_repr == "SymFloat":
                    return "c10::SymFloatType::get()"
                raise AssertionError(f"Unsupported list element type: {t}")

            def codegen_memory_format(memory_format: torch.memory_format) -> str:
                return {
                    torch.contiguous_format: "c10::MemoryFormat::Contiguous",
                    torch.channels_last: "c10::MemoryFormat::ChannelsLast",
                    torch.channels_last_3d: "c10::MemoryFormat::ChannelsLast3d",
                    torch.preserve_format: "c10::MemoryFormat::Preserve",
                }[memory_format]

            def codegen_tensor_ivalue(raw_arg: Any, arg_type: torch.JitType) -> str:
                if hasattr(raw_arg, "codegen_reference"):
                    codegen_arg = raw_arg.codegen_reference()
                else:
                    codegen_arg = self.val_to_arg_str(raw_arg, arg_type)

                raii_var = self.create_tmp_raii_handle_var_if_needed(
                    codegen_arg, dispatch_lines
                )
                return (
                    "c10::IValue(*torch::aot_inductor::"
                    f"tensor_handle_to_tensor_pointer({raii_var}))"
                )

            def codegen_ivalue(raw_arg: Any, arg_type: torch.JitType) -> str:
                if raw_arg is None:
                    # A None at a non-optional Tensor arg means an absent tensor.
                    # Box it as an undefined at::Tensor (ATen's absent-tensor
                    # sentinel, matching eager value_or(Tensor())); boxing None
                    # would fail to unbox as `const Tensor&`.  Optionals stay None.
                    if isinstance(arg_type, torch.TensorType):
                        return "c10::IValue(at::Tensor())"
                    return "c10::IValue()"

                if isinstance(arg_type, torch.OptionalType):
                    return codegen_ivalue(raw_arg, arg_type.getElementType())

                if isinstance(raw_arg, torch.device):
                    if raw_arg.type not in DEVICE_TO_ATEN:
                        raise AssertionError(
                            raw_arg.type + " not found in DEVICE_TO_ATEN"
                        )
                    return (
                        "c10::IValue(c10::Device("
                        f"{DEVICE_TO_ATEN[raw_arg.type]}, "
                        f"{raw_arg.index if raw_arg.index is not None else 0}))"
                    )
                if isinstance(raw_arg, torch.dtype):
                    return f"c10::IValue({DTYPE_TO_ATEN[raw_arg]})"
                if isinstance(raw_arg, torch.layout):
                    return f"c10::IValue({LAYOUT_TO_ATEN[raw_arg]})"
                if isinstance(raw_arg, torch.memory_format):
                    return f"c10::IValue({codegen_memory_format(raw_arg)})"

                if isinstance(arg_type, torch.TensorType):
                    return codegen_tensor_ivalue(raw_arg, arg_type)

                if isinstance(arg_type, torch.ListType):
                    if not isinstance(raw_arg, (list, tuple)):
                        raise AssertionError(type(raw_arg))
                    list_var = next_tmp("tmp_list")
                    elem_type = arg_type.getElementType()
                    dispatch_lines.writeline(
                        f"c10::impl::GenericList {list_var}({type_expr(elem_type)});"
                    )
                    dispatch_lines.writeline(f"{list_var}.reserve({len(raw_arg)});")
                    for elem in raw_arg:
                        dispatch_lines.writeline(
                            f"{list_var}.emplace_back({codegen_ivalue(elem, elem_type)});"
                        )
                    return f"c10::IValue(std::move({list_var}))"

                if isinstance(arg_type, torch.TupleType):
                    if not isinstance(raw_arg, (list, tuple)):
                        raise AssertionError(type(raw_arg))
                    tuple_var = next_tmp("tmp_tuple")
                    dispatch_lines.writeline(f"std::vector<c10::IValue> {tuple_var};")
                    dispatch_lines.writeline(f"{tuple_var}.reserve({len(raw_arg)});")
                    for elem, elem_type in zip(raw_arg, arg_type.elements()):
                        dispatch_lines.writeline(
                            f"{tuple_var}.emplace_back({codegen_ivalue(elem, elem_type)});"
                        )
                    return (
                        "c10::IValue(c10::ivalue::Tuple::create("
                        f"std::move({tuple_var})))"
                    )

                if isinstance(arg_type, torch.BoolType):
                    return (
                        "c10::IValue(static_cast<bool>("
                        f"{self.val_to_arg_str_for_prim_type(raw_arg, bool)}))"
                    )
                if isinstance(arg_type, torch.IntType):
                    return (
                        "c10::IValue(static_cast<int64_t>("
                        f"{self.val_to_arg_str_for_prim_type(raw_arg, int)}))"
                    )
                if isinstance(arg_type, torch.FloatType):
                    return (
                        "c10::IValue(static_cast<double>("
                        f"{self.val_to_arg_str_for_prim_type(raw_arg, float)}))"
                    )
                if isinstance(arg_type, torch.StringType):
                    return (
                        "c10::IValue(std::string("
                        f"{self.val_to_arg_str_for_prim_type(raw_arg, str)}))"
                    )
                if isinstance(arg_type, torch.SymBoolType):
                    return (
                        "c10::IValue(c10::SymBool(static_cast<bool>("
                        f"{self.val_to_arg_str_for_prim_type(raw_arg, bool)})))"
                    )
                if isinstance(arg_type, torch.SymIntType):
                    return (
                        "c10::IValue(c10::SymInt(static_cast<int64_t>("
                        f"{self.val_to_arg_str_for_prim_type(raw_arg, int)})))"
                    )
                if isinstance(arg_type, torch.NumberType):
                    return f"c10::IValue({self.val_to_arg_str_for_prim_type(raw_arg, None)})"

                arg_type_repr = repr(arg_type)
                if arg_type_repr == "Layout":
                    if not isinstance(raw_arg, torch.layout):
                        raise AssertionError(type(raw_arg))
                    return f"c10::IValue({LAYOUT_TO_ATEN[raw_arg]})"
                if arg_type_repr == "MemoryFormat":
                    if not isinstance(raw_arg, torch.memory_format):
                        raise AssertionError(type(raw_arg))
                    return f"c10::IValue({codegen_memory_format(raw_arg)})"
                if arg_type_repr == "ScalarType":
                    if not isinstance(raw_arg, torch.dtype):
                        raise AssertionError(type(raw_arg))
                    return f"c10::IValue({DTYPE_TO_ATEN[raw_arg]})"
                if arg_type_repr == "SymFloat":
                    return (
                        "c10::IValue(c10::SymFloat(static_cast<double>("
                        f"{self.val_to_arg_str_for_prim_type(raw_arg, float)})))"
                    )

                raise AssertionError(f"Unsupported boxed dispatcher arg: {arg_type}")

            stack_var = next_tmp("dispatch_stack")
            dispatch_lines.writeline(f"std::vector<c10::IValue> {stack_var};")
            dispatch_lines.writeline(
                f"{stack_var}.reserve({len(op_overload._schema.arguments)});"
            )
            for raw_arg, schema_arg in zip(raw_args, op_overload._schema.arguments):
                dispatch_lines.writeline(
                    f"{stack_var}.emplace_back({codegen_ivalue(raw_arg, schema_arg.real_type)});"
                )

            dispatch_lines.writeline(
                "static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("
                f'"{op_overload._schema.name}", "{op_overload._schema.overload_name}");'
            )
            dispatch_lines.writeline(f"op.callBoxed(&{stack_var});")

            stack_idx = 0
            for output_arg, raw_output_arg in zip(output_args, raw_outputs):
                if isinstance(raw_output_arg, ir.MutationOutput):
                    continue
                if output_arg is not None:
                    dispatch_lines.writeline(
                        f"{output_arg} = torch::aot_inductor::new_tensor_handle("
                        f"std::move({stack_var}[{stack_idx}]).toTensor());"
                    )
                stack_idx += 1

        dispatch_lines.writeline("}")
        self.writelines(dispatch_lines.getvalue().splitlines())

    def generate_fallback_kernel_with_runtime_lookup_python(
        self,
        buf_name: str,
        python_kernel_name: str,
        op_overload: torch._ops.OpOverload,
        raw_args: Sequence[Any],
        output_args: Sequence[str | None],
        raw_outputs: Sequence[ir.Buffer],
    ) -> None:
        """Generate fallback kernel calls with runtime (non-AOT) dispatch.  This can
        only be called in cpp_wrapper mode, and assumes that the input is a non-None
        OpOverload.

        This function calls into Python to dispatch, which allows it to handle datatypes
        that cannot be contained in StableIValue, at the cost of some performance."""
        self.load_custom_op_wrapper()

        num_args = len(raw_args)
        py_args_var = f"py_args_{next(self.arg_var_id)}"
        # First arg is always the python op name
        lines = textwrap.dedent(f"""
            RAIIPyObject {py_args_var}(PyTuple_New({num_args + 1}));
            if (!{py_args_var}) {{
                throw std::runtime_error("PyTuple_New {py_args_var} failed");
            }}
            PyTuple_SetItem({py_args_var}, 0, PyUnicode_FromString("{python_kernel_name}"));
            """)

        for idx, (raw_arg, schema_arg) in enumerate(
            zip(raw_args, op_overload._schema.arguments)
        ):
            lines += self.generate_py_arg(
                py_args_var, idx + 1, raw_arg, schema_arg.real_type
            )

        lines += textwrap.dedent(f"""
            // Call the custom op in Python
            RAIIPyObject py_{buf_name}(PyObject_CallObject(custom_op_wrapper, {py_args_var}));
            if (!py_{buf_name}) {{
                if (PyErr_Occurred()) {{
                    return;
                }}
                throw std::runtime_error("PyObject_CallObject {python_kernel_name} failed");
            }}
            """)

        if raw_outputs:
            num_returned_outputs = 0
            returned_output_slots = []
            for output_arg, raw_output_arg in zip(output_args, raw_outputs):
                if isinstance(raw_output_arg, ir.MutationOutput):
                    continue
                if output_arg is not None:
                    returned_output_slots.append((num_returned_outputs, output_arg))
                num_returned_outputs += 1
        else:
            num_returned_outputs = len(output_args)
            returned_output_slots = [
                (idx, output_arg)
                for idx, output_arg in enumerate(output_args)
                if output_arg is not None
            ]

        # custom_op_wrapper returns a bare capsule only for a single, non-list
        # Tensor return. A Tensor[] return (even of length 1) or multiple returns
        # come back as a Python list that must be indexed with PyList_GET_ITEM.
        schema_returns = op_overload._schema.returns
        returns_python_list = len(schema_returns) != 1 or isinstance(
            schema_returns[0].real_type, torch.ListType
        )
        if (
            num_returned_outputs == 1
            and returned_output_slots
            and not returns_python_list
        ):
            # result is a single tensor
            _, output = returned_output_slots[0]
            lines += f"{output} = reinterpret_cast<AtenTensorHandle>(PyCapsule_GetPointer(py_{buf_name}.get(), NULL));\n"
        else:
            # result is a list of tensors (Tensor[] or multiple returns)
            for idx, output_arg in returned_output_slots:
                lines += f"{output_arg} = reinterpret_cast<AtenTensorHandle>(PyCapsule_GetPointer(PyList_GET_ITEM(py_{buf_name}.get(), {idx}), NULL));\n"

        if raw_outputs:
            declarations_before_scope = [
                f"RAIIAtenTensorHandle {output_arg};"
                for output_arg, raw_output_arg in zip(output_args, raw_outputs)  # type: ignore[arg-type]
                if output_arg is not None
                and not isinstance(raw_output_arg, ir.MutationOutput)
            ]
        else:
            declarations_before_scope = [
                f"RAIIAtenTensorHandle {output_arg};"
                for output_arg in output_args  # type: ignore[arg-type]
                if output_arg is not None
            ]
        scope_gil_acquire = self.generate_scoped_gil_acquire(
            declarations_before_scope, lines
        )
        self.writelines(scope_gil_acquire)

    def generate_fallback_kernel_with_runtime_lookup_aot(
        self,
        op_overload: torch._ops.OpOverload | torch._ops.HigherOrderOperator,
        raw_args: Sequence[Any],
        output_args: _OUTPUT_ARGS_TYPE,
        raw_outputs: Sequence[ir.Buffer],
    ) -> None:
        (
            tensor_call_args,
            int_call_args,
        ) = self.generate_extern_kernel_args_decl_if_needed(
            op_overload,
            raw_args,
            output_args,
            raw_outputs,
        )
        # force both temporary arrays to generate mutable data pointers, since the proxy
        # executor signature requires that datatype
        int_call_str = self._generate_temporary_array_pointer(
            "int64_t", int_call_args, force_mutable=True
        )
        tensor_call_args = self.codegen_runtime_lookup_tensor_call_args(
            tensor_call_args
        )
        tensor_call_str = self._generate_temporary_array_pointer(
            "AtenTensorHandle", tensor_call_args, force_mutable=True
        )

        extern_kernel_node_index = len(V.extern_kernel_nodes) - 1
        self.writeline(
            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_proxy_executor_call_function(proxy_executor, "
            f"{extern_kernel_node_index}, "
            f"{len(int_call_args)}, "
            f"{int_call_str}, "
            f"{len(tensor_call_args)}, "
            f"{tensor_call_str}));"
        )

    def codegen_runtime_lookup_tensor_call_args(
        self, tensor_call_args: Sequence[str]
    ) -> Sequence[str]:
        return tensor_call_args

    def generate_reset_kernel_saved_flags(self):
        pass

    def generate_save_uncompiled_kernels(self):
        pass

    def c_type_for_prim_type(self, val, type_) -> str:
        if isinstance(type_, torch.OptionalType):
            return f"{self.c_type_for_prim_type(val, type_.getElementType())}*"
        elif isinstance(type_, torch.TensorType):
            return "AtenTensorHandle"
        elif isinstance(type_, (torch.IntType, torch.SymIntType)):
            return "int64_t"
        elif isinstance(
            type_, (torch.BoolType, torch.SymBoolType, torch.EnumType)
        ) or repr(type_) in ("Layout", "MemoryFormat", "ScalarType"):
            return "int32_t"
        elif isinstance(type_, torch.FloatType):
            return "double"
        elif isinstance(type_, torch.NumberType):
            if isinstance(val, bool):
                return "int32_t"
            elif isinstance(val, (int, float)):
                return "double"
            elif val is None:
                # This could happen when val is an optional value
                return "double"
            else:
                raise AssertionError(
                    f"Unexpected type in c_type_for_prim_type: {type_=}"
                )
        elif isinstance(type_, torch.StringType):
            return "const char*"
        else:
            raise AssertionError(f"Unexpected type in c_type_for_prim_type: {type_=}")

    def val_to_arg_str_for_prim_type(self, val, type_) -> str:
        # TODO: not using type_ as the first step of refactoring. Will update this later.
        if isinstance(val, bool):
            return "1" if val else "0"
        elif isinstance(val, int):
            # uint64_t is long on Linux, but long long on MacOS and Windows
            return f"{val}LL" if sys.platform in ["darwin", "win32"] else f"{val}L"
        elif isinstance(val, complex):
            return f"c10::complex<double>{{ {self.generate_float_value(val.real)}, {self.generate_float_value(val.imag)} }}"
        elif isinstance(val, str):
            return f'"{val}"'
        elif isinstance(
            val, (ir.Buffer, ir.ReinterpretView, ir.StorageBox, ir.TensorBox)
        ):
            return val.codegen_reference()
        elif isinstance(val, torch.device):
            return self.codegen_device(val)
        elif isinstance(val, torch.dtype):
            return self.codegen_dtype(val)
        elif isinstance(val, torch.layout):
            return self.codegen_layout(val)
        elif isinstance(val, torch.memory_format):
            return self.codegen_memory_format(val)
        elif isinstance(val, float):
            return self.generate_float_value(val)
        elif isinstance(val, (list, tuple)):
            # FIXME: This happens because type_ is not always properly set to torch.ListType
            return f"{{{', '.join(self.val_to_arg_str(x, None) for x in val)}}}"
        elif isinstance(val, SymTypes):
            return cexpr(val.node.expr)
        elif isinstance(val, sympy.Expr):
            return cexpr(val)
        else:
            return repr(val)

    def val_to_arg_str(self, val, type_=None) -> str:
        if val is None:
            # None needs special care. It either represent nullopt or an empty tensor
            if type_ is None or isinstance(type_, torch.OptionalType):
                if type_ is not None and isinstance(
                    type_.getElementType(),
                    (
                        torch.DeviceObjType,
                        torch.ListType,
                        torch.TupleType,
                    ),
                ):
                    return "nullptr, 0"
                return "nullptr"

            if isinstance(type_, torch.TensorType):
                # create an empty tensor, the equivalent of at::Tensor()
                var_name = f"var_{next(self.arg_var_id)}"
                self.writeline(f"AtenTensorHandle {var_name}_handle;")
                self.writeline(
                    f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_new_uninitialized_tensor(&{var_name}_handle));"
                )
                self.writeline(f"RAIIAtenTensorHandle {var_name}({var_name}_handle);")
                return var_name

            raise AssertionError("Can not map None to a known data type")

        if isinstance(type_, torch.OptionalType):
            element_type = type_.getElementType()
            arg_str = self.val_to_arg_str(val, element_type)
            # Handle optional iterables as a special case.  Utilize the
            # temporary_reference function to avoid saving them off and increasing
            # memory usage.
            if isinstance(element_type, (torch.ListType, torch.TupleType)):
                main_value, aux = arg_str.rsplit(", ", maxsplit=1)
                return f"&temporary_reference({main_value}), {aux}"

            # Handle optional tensors as a special case, as above.
            if isinstance(element_type, torch.TensorType):
                base_handle = self.val_to_arg_str(val, element_type)
                return f"&temporary_reference({base_handle}.get())"

            var_name = f"var_{next(self.arg_var_id)}"
            if isinstance(element_type, torch.DeviceObjType):
                main_value, aux = arg_str.rsplit(", ", maxsplit=1)
                self.writeline(f"auto {var_name} = {main_value};")
                return f"&{var_name}, {aux}"

            self.writeline(
                f"{self.c_type_for_prim_type(val, element_type)} {var_name} = {arg_str};"
            )
            return f"&{var_name}"

        if isinstance(type_, (torch.ListType, torch.TupleType)):
            if not isinstance(val, (list, tuple)):
                raise AssertionError(f"{val} does not match with arg type {type_}")
            element_type = type_.getElementType()

            if len(val) == 0:
                # Zero-size array is not supported in the C or C++ standard, so return a
                # nullptr.
                return "nullptr, 0"

            result = [self.val_to_arg_str(x, element_type) for x in val]
            if isinstance(element_type, torch.TensorType):
                result = [f"{t}.get()" for t in result]

            c_type = self.c_type_for_prim_type(val[0], element_type)
            # see the comment in self._generate_temporary_array_pointer for an
            # explanation of why this c_type gets modified
            if isinstance(element_type, torch.OptionalType) and not c_type.startswith(
                "const"
            ):
                c_type = f"const {c_type}"

            # need to pass the array length, because we can't use the std::array member
            # function
            return (
                f"{self._generate_temporary_array_pointer(c_type, result)}, {len(val)}"
            )

        val_is_scalar = isinstance(val, (bool, complex, float, int, *SymTypes))
        if isinstance(type_, torch.TensorType) and val_is_scalar:
            val_str = self.val_to_arg_str_for_prim_type(val, None)
            return self.codegen_scalar_to_tensor(val_str)

        return self.val_to_arg_str_for_prim_type(val, type_)

    def create_tmp_raii_handle_var_if_needed(
        self, handle: str, writer: HasWriteLine | list[str] | None = None
    ) -> str:
        """If the input handle is an rvalue RAII tensor, creates an lvalue variable for
        it in writer.  Returns a variable name that can be used to access handle."""
        if not handle.startswith(
            (
                "borrow_arrayref_tensor_as_tensor(",
                "copy_arrayref_tensor_to_tensor(",
                "wrap_with_raii_handle_if_needed(",
                "RAIIAtenTensorHandle(",
            )
        ):
            return handle

        tmp_var_name = f"var_{next(self.arg_var_id)}"
        call_str = f"auto {tmp_var_name} = {handle};"

        if writer is None:
            self.writeline(call_str)
        elif isinstance(writer, list):
            writer.append(call_str)
        else:
            writer.writeline(call_str)

        return tmp_var_name

    def write_kernel_context_guard_begin(
        self,
    ):
        # Beginning of a kernel context guarded block.
        # The block looks like this:
        # {
        # KernelContextGuard _ctx("{kernel_name}", {stack_trace_str});
        # ... operations...
        # }
        self.writeline("{")

    def write_kernel_context_guard_end(
        self,
    ):
        # End of a kernel context guarded block.
        self.writeline("}")

    def write_kernel_context_guard(
        self,
        kernel_name: str,
        node_schedule: Sequence[BaseSchedulerNode] | ExternKernel,
    ):
        def aggregate_stack_traces(
            node_schedule: Sequence[BaseSchedulerNode] | ExternKernel,
        ) -> OrderedSet[str]:
            if isinstance(node_schedule, list):
                return functools.reduce(
                    lambda a, b: a | b,
                    [
                        # pyrefly: ignore [missing-attribute]
                        node.node.get_stack_traces()
                        for node in node_schedule
                        if hasattr(node, "node") and node.node
                    ],
                    OrderedSet(),
                )
            elif isinstance(node_schedule, ExternKernel):
                return node_schedule.get_stack_traces()
            else:
                return OrderedSet()

        stack_trace_str = 'R"('
        stack_traces = aggregate_stack_traces(node_schedule)

        for stack_trace in stack_traces:
            for line in stack_trace.split("\n"):
                stack_trace_str += f"\n{line}"
            stack_trace_str += "\n"
        stack_trace_str += ')"'
        self.writeline(f'KernelContextGuard _ctx("{kernel_name}", {stack_trace_str});')
<|endoftext|>import contextlib
import logging
import os
from collections.abc import Callable
from typing import Any, cast, TypeVar
from unittest.mock import patch

import torch
import torch.utils
from torch.utils._ordered_set import OrderedSet

from ..._dynamo.utils import counters
from .. import config, ir
from ..kernel.mm_common import mm_args
from ..select_algorithm import ChoiceCaller, DataProcessorTemplateWrapper
from ..utils import parallel_num_threads
from ..virtualized import V
from .cpp import get_export_declaration
from .cpp_gemm_template import (
    CppGemmTemplate,
    expand_bias,
    gen_2d_view_of_epilogue_buf,
    prune_tensors,
    transpose_w,
)
from .cpp_micro_gemm import CppMicroGemmAMX, create_micro_gemm
from .cpp_template_kernel import CppTemplateKernel
from .cpp_utils import (
    create_epilogue_with_attr,
    GemmBlocking,
    get_gemm_template_output_and_compute_dtype,
)


log = logging.getLogger(__name__)

GEMM_TEMPLATE = r"""
{{template.header().getvalue()}}
{{micro_gemm.codegen_define(kernel)}}

extern "C" {{export_declaration}}
{{kernel.def_kernel(inputs=kernel_args, outputs=Y_list, aliases=aliases)}}
{
    {{kernel.maybe_codegen_profile(template.get_kernel_prefix_name())}}
    {{ template.codegen_blocks(
        num_threads, N, K, micro_gemm, is_dynamic_M, kernel, GemmOuts[0], config, L1_cache_size, L2_cache_size, X_list[0], W_list[0]
    ) }}
{%- if num_threads > 1 %}
    {%- set use_dynamic_threads = ((config.cpp.threads < 1) and (num_threads == cpu_count)) or config.cpp.dynamic_threads %}
    {%- if use_dynamic_threads %}
    #pragma omp parallel
    {%- else %}
    #pragma omp parallel num_threads({{num_threads}})
    {%- endif %}
    {
        {{ micro_gemm.codegen_init(kernel) }}
        #pragma omp for schedule(static, 1)
        for (int64_t tid = 0; tid < {{num_threads}}; tid++) {
            {{ template.codegen_multi_threads_params()|indent(12, false) }}
{%- else %}
    {
        {{ template.codegen_single_thread_params(is_dynamic_M)|indent(8, false) }}
        {{ micro_gemm.codegen_init(kernel) }}
{%- endif %}
{%- set acc_buf_name_list=[] %}
{%- set acc_buf_name_prefix = "local_acc_buf_" %}
{%- for gemm_idx in range(0, gemm_grouped_num, 1) %}
    {%- set acc_buf_name = acc_buf_name_prefix + gemm_idx|string %}
    {{ kernel.define_buffer(acc_buf_name, ["Mc_blocks*Mr", "Nc_blocks*Nr"], acc_buf_dtype) }}
    {%- set acc_buf_name_list=acc_buf_name_list.append(acc_buf_name) %}
{%- endfor %}
        for (int64_t mc_block_id = 0; mc_block_id < num_Mc_blocks_per_thread; mc_block_id++) {
            {{ template.codegen_m_loop_params()|indent(12, false) }}
            for (int64_t nc = n_block_start; nc < n_block_end; nc += Nc_blocks) {
                {{ template.codegen_n_loop_params()|indent(16, false) }}
{%- set acc_list=[] %}
{%- for gemm_idx in range(0, gemm_grouped_num, 1) %}
    {%- set acc_list = acc_list.append( kernel.local_buffers[acc_buf_name_list[gemm_idx]] ) %}
    {{ kernel.reinit_buffer_if_null(acc_buf_name_list[gemm_idx]) }}
{%- endfor %}
                for (int64_t kc = k_block_start; kc < k_block_end; kc += Kc_blocks) {
                    int64_t k_start = kc * Kr;
                    int64_t k_end = std::min(std::min(kc + Kc_blocks, k_block_end) * Kr, K);
{%- set tile_X_list=[] %}
{%- for gemm_idx in range(0, gemm_grouped_num, 1) %}
    {%- set tile_X_list = tile_X_list.append( kernel.slice_nd(X_list[gemm_idx], [("m_start", "m_end"), ("k_start", "k_end")]) ) %}
{%- endfor %}
                    for (int64_t nci = nc; nci < nc_block_end; nci++) {
{%- set tile_W_3d_list=[] %}
{%- set tile_W_list=[] %}
{%- set acc_slice_list=[] %}
{%- for gemm_idx in range(0, gemm_grouped_num, 1) %}
    {%- set acc_slice_list = acc_slice_list.append(
        kernel.slice_nd(acc_list[gemm_idx], [("0", "m_end - m_start"), ("(nci - nc)*Nr", "(nci - nc + 1)*Nr")])
    ) %}
    {%- set tile_W_3d_list = tile_W_3d_list.append(
        kernel.slice_nd(W_list[gemm_idx], [("nci", "nci + 1"), ("k_start", "k_end"), ()])
    ) %}
{%- endfor %}
{%- for gemm_idx in range(0, gemm_grouped_num, 1) %}
    {%- set tile_W_list = tile_W_list.append(
        kernel.view(tile_W_3d_list[gemm_idx], ["k_end - k_start", micro_gemm.register_blocking.block_n])
    ) %}
{%- endfor %}
                        if (kc == k_block_start) {
                            {%- for gemm_idx in range(0, gemm_grouped_num, 1) %}
                                {{ micro_gemm.codegen_call(
                                    kernel, tile_X_list[gemm_idx], tile_W_list[gemm_idx], acc_slice_list[gemm_idx], accum=False
                                )|indent(28, false) }}
                            {%- endfor %}
                        } else {
                            {%- for gemm_idx in range(0, gemm_grouped_num, 1) %}
                                {{ micro_gemm.codegen_call(
                                    kernel, tile_X_list[gemm_idx], tile_W_list[gemm_idx], acc_slice_list[gemm_idx], accum=True
                                )|indent(28, false) }}
                            {%- endfor %}
                        }
                    }
                }
                {
{%- set tile_acc_list = [] %}
{%- set tile_Y_list = [] %}
{%- for gemm_idx in range(0, gemm_grouped_num, 1) %}
    {%- set tile_acc_list = tile_acc_list.append(
        kernel.slice_nd(acc_list[gemm_idx], [("0", "m_end - m_start"), ("0", "n_end - n_start")])
    ) %}
    {%- set tile_Y_list = tile_Y_list.append(
        kernel.slice_nd(Y_2d_list[gemm_idx], [("m_start", "m_end"), ("n_start", "n_end")])
    ) %}
{%- endfor %}
                    {{ kernel.store_outputs(
                        tile_Y_list,
                        tile_acc_list,
                        GemmOuts,
                        epilogue_nodes,
                        offsets=("m_start", "n_start"),
                        reindexers=reindexers,
                        multi_output_buffers=multi_output_buffers
                    )|indent(20, false)
                    }}
                }
            }
        }
{%- if num_threads > 1 %}
        }
        {{ micro_gemm.codegen_finalize(kernel) }}
    }
{%- else %}
        {{ micro_gemm.codegen_finalize(kernel) }}
    }
{%- endif %}
}
"""


def get_deduplicated_act(act_mapping: dict[int, ir.IRNode]) -> list[ir.IRNode]:
    act_deduplicated = []
    act_deduplicated_name: OrderedSet[str] = OrderedSet()
    for act_idx in range(len(act_mapping.values())):
        act = act_mapping[act_idx]
        if act.get_name() not in act_deduplicated_name:
            act_deduplicated.append(act)
            act_deduplicated_name.add(act.get_name())
    return act_deduplicated


class CppGroupedGemmTemplate(CppGemmTemplate):
    def __init__(
        self,
        input_nodes: list[ir.IRNode],
        layout: ir.Layout,
        num_threads: int,
        register_blocking: GemmBlocking,
        beta: int = 1,
        alpha: int = 1,
        has_bias: bool = False,
        epilogue_creator: Callable[[ir.Buffer], ir.Pointwise] | None = None,
        act_mapping: dict[int, ir.IRNode] | None = None,
        gemm_grouped_num: int = 1,
    ) -> None:
        """
        Template for Group of GEMMs:
        * Each GEMM has the same dimensions (m, n, k) and the same leading dimensions (lda, ldb, ldc)
          for their A, B, and C matrices.
        * Each GEMM has distinct or shared activations, has distinct weight, has unique bias or no bias, has distinct epilogues.
        * In the current implementation, the outputs of all GEMMs are accumulated using pointwise epilogues.
          This behavior can be extended in the future if needed.
        """
        super().__init__(
            input_nodes,
            layout,
            num_threads,
            register_blocking,
            beta,
            alpha,
            has_bias,
            epilogue_creator,
        )
        self.act_mapping = act_mapping
        self.gemm_grouped_num = gemm_grouped_num
        # pyrefly: ignore [bad-override]
        self.output_node: list[ir.Buffer] = [
            ir.Buffer(name="buf_out" + str(idx), layout=layout)
            for idx in range(gemm_grouped_num)
        ]

    @classmethod
    # pyrefly: ignore [bad-override]
    def add_choices(
        cls,
        choices: list[ChoiceCaller],
        layout: ir.Layout,
        input_nodes: list[ir.IRNode],
        beta: int = 1,
        alpha: int = 1,
        has_bias: tuple[bool, ...] = (False, False),
        trans_w: bool = False,
        input_indices: list[int] | None = None,
        epilogue_creator: Callable[[ir.Buffer], ir.Pointwise] | None = None,
        act_mapping: dict[int, ir.IRNode] | None = None,  # gemm idx to its act buf
    ) -> DataProcessorTemplateWrapper:
        # Input nodes order: x, optional[x1], ... w0, w1, ... optional[b0], optional[b1], ...
        gemm_grouped_num = len(has_bias)
        if not act_mapping:
            raise AssertionError("expected non-empty act_mapping")
        act_deduplicated = get_deduplicated_act(act_mapping)
        wgt_start_idx = len(act_deduplicated)
        bias_start_idx = wgt_start_idx + gemm_grouped_num
        input_indices = list(range(len(input_nodes)))

        _T = TypeVar("_T", ir.IRNode, torch.Tensor)
        _U = TypeVar("_U", ir.Layout, torch.Tensor)

        def reorder_and_filter(
            inputs: list[_T],
            layout_or_out: _U,
        ) -> tuple[list[_T], _U]:
            if input_indices is None:
                raise AssertionError("input_indices must be set")
            return [inputs[idx] for idx in input_indices], layout_or_out

        new_inputs, new_layout = reorder_and_filter(input_nodes, layout)

        def maybe_to_dense(
            inputs: list[_T],
            layout_or_out: _U,
        ) -> tuple[list[_T], _U]:
            new_inputs = list(inputs)
            for idx in range(wgt_start_idx, wgt_start_idx + gemm_grouped_num):
                if isinstance(inputs[idx], torch.Tensor):
                    W = inputs[idx]
                    if not isinstance(W, torch.Tensor):
                        raise AssertionError("W must be a torch.Tensor")
                    # pyrefly: ignore [unsupported-operation]
                    new_inputs[idx] = W.to_dense() if W.is_mkldnn else W
            return new_inputs, layout_or_out

        def normalize_shapes(
            inputs: list[_T],
            layout_or_out: _U,
        ) -> tuple[list[_T], _U]:
            new_inputs: list[_T] = list(inputs)
            if not trans_w:
                return new_inputs, layout_or_out
            X = new_inputs[0]
            for wgt_idx in range(wgt_start_idx, wgt_start_idx + gemm_grouped_num):
                new_input = new_inputs[wgt_idx]
                new_inputs[wgt_idx] = transpose_w(new_input, trans_w)
            for bias_idx in range(bias_start_idx, len(new_inputs)):
                # pyrefly: ignore [bad-argument-type]
                new_bias = expand_bias(new_inputs[bias_idx], X)
                if new_bias is None:
                    raise AssertionError("expected new_bias to be not None")
                # pyrefly: ignore [unsupported-operation]
                new_inputs[bias_idx] = new_bias
            return new_inputs, layout_or_out

        num_threads = parallel_num_threads()
        new_inputs, _ = normalize_shapes(*maybe_to_dense(new_inputs, new_layout))
        m, n, k, *_ = mm_args(new_inputs[0], new_inputs[wgt_start_idx])
        output_dtype, compute_dtype = get_gemm_template_output_and_compute_dtype(
            new_inputs[0].get_dtype()
        )
        micro_gemm = create_micro_gemm(
            "micro_gemm",
            m,
            n,
            k,
            input_dtype=new_inputs[0].get_dtype(),
            input2_dtype=new_inputs[wgt_start_idx].get_dtype(),
            output_dtype=output_dtype,
            compute_dtype=compute_dtype,
            alpha=alpha,
            num_threads=num_threads,
        )
        if micro_gemm is None:
            raise AssertionError("expected micro_gemm to be not None")
        _, block_n, _ = micro_gemm.register_blocking
        new_size, padded_n = cls.get_padded_size(
            n, block_n, k, should_block_weight=True
        )
        padding = padded_n - n

        def pack_weight(
            inputs: list[_T],
            layout_or_out: _U,
        ) -> tuple[list[_T], _U]:
            new_W_list = []
            new_inputs = list(inputs)
            W_list = new_inputs[wgt_start_idx : wgt_start_idx + gemm_grouped_num]
            for W in W_list:
                blocked_w = cls.block_weight(W, new_size, padding)
                new_W_list.append(cls.pack_vnni_weight(blocked_w, micro_gemm, new_size))
            new_inputs[wgt_start_idx : wgt_start_idx + gemm_grouped_num] = new_W_list
            return new_inputs, layout_or_out

        def preprocessor(
            inputs: list[_T],
            layout: _U,
        ) -> tuple[list[_T], _U]:
            return pack_weight(
                *normalize_shapes(*maybe_to_dense(*reorder_and_filter(inputs, layout)))
            )

        def postprocessor(output: _T) -> _T:
            if isinstance(output, ir.TensorBox):
                template_buffer = ir.InputsKernel.unwrap_storage_for_input(output)
                if not isinstance(template_buffer, ir.CppTemplateBuffer):
                    raise AssertionError(
                        "expected template_buffer to be ir.CppTemplateBuffer"
                    )
                new_input_nodes, _ = reorder_and_filter(input_nodes, layout)
                W_nodes = new_input_nodes[
                    wgt_start_idx : wgt_start_idx + gemm_grouped_num
                ]
                W_tensor = []
                for W_node in W_nodes:
                    if W_node.get_name() not in V.graph.constants:
                        raise AssertionError(
                            f"expected {W_node.get_name()} in V.graph.constants"
                        )
                    # pyrefly: ignore [bad-argument-type]
                    W_tensor.append(V.graph.constants[W_node.get_name()])
                # pyrefly: ignore [unsupported-operation]
                new_input_nodes[wgt_start_idx : wgt_start_idx + gemm_grouped_num] = (
                    W_tensor  # type: ignore[assignment]
                )
                new_input_nodes, _ = pack_weight(
                    *normalize_shapes(*maybe_to_dense(new_input_nodes, layout))
                )
                # Prune unused tensors
                prune_tensors(input_nodes, new_input_nodes)
                for idx in range(wgt_start_idx, wgt_start_idx + gemm_grouped_num):
                    W_packed = new_input_nodes[idx]
                    if not isinstance(W_packed, torch.Tensor):
                        raise AssertionError("expected W_packed to be a torch.Tensor")
                    W_packed_constant = V.graph.add_tensor_constant(W_packed)
                    template_buffer.inputs[idx] = (
                        ir.InputsKernel.unwrap_storage_for_input(W_packed_constant)
                    )

            return output

        template = DataProcessorTemplateWrapper(
            CppGroupedGemmTemplate,
            preprocessor,
            postprocessor,
            input_nodes=input_nodes,
            layout=layout,
            num_threads=num_threads,
            register_blocking=micro_gemm.register_blocking,
            beta=beta,
            alpha=alpha,
            has_bias=has_bias,
            epilogue_creator=epilogue_creator,
            act_mapping=act_mapping,
            gemm_grouped_num=gemm_grouped_num,
        )
        template.maybe_append_choice(choices)
        return template

    def get_kernel_prefix_name(self) -> str:
        return f"grouped_gemm_g{self.gemm_grouped_num}m{self.m}n{self.n}k{self.k}"

    def render(  # type: ignore[override,return,no-untyped-def]
        self,
        kernel: CppTemplateKernel,
        template_buffer_node: ir.CppTemplateBuffer | None = None,
        flag_template_buffer_has_other_users: bool | None = None,
        epilogue_nodes: list[ir.IRNode] | None = None,
        **kwargs,
    ) -> str:
        if not self.act_mapping:
            raise AssertionError("expected non-empty self.act_mapping")
        act_deduplicated = get_deduplicated_act(self.act_mapping)
        wgt_start_idx = len(act_deduplicated)
        bias_start_idx = wgt_start_idx + self.gemm_grouped_num
        X_list = list(self.act_mapping.values())
        W_list = self.input_nodes[wgt_start_idx : wgt_start_idx + self.gemm_grouped_num]
        inp_list = []
        cur_idx = bias_start_idx
        for inp_idx in range(self.gemm_grouped_num):
            inp = None
            # pyrefly: ignore [bad-index, index-error]
            if self.has_bias[inp_idx]:
                inp = self.input_nodes[cur_idx]
                cur_idx += 1
            inp_list.append(inp)

        Y_list = self.output_node
        multi_output_buffers = None
        if template_buffer_node is not None:
            W_list = template_buffer_node.inputs[
                wgt_start_idx : wgt_start_idx + self.gemm_grouped_num
            ]
            if not isinstance(template_buffer_node.outputs, list):
                raise AssertionError(
                    "expected template_buffer_node.outputs to be a list"
                )
            Y_list = template_buffer_node.outputs
            counters["inductor"]["cpp_grouped_gemm_template"] += 1
            multi_output_buffers = template_buffer_node.outputs

        template_buffer = Y_list[0]
        fake_buffers: list[ir.Buffer] = []
        Y_2d_list = Y_list
        output_dtype, compute_dtype = get_gemm_template_output_and_compute_dtype(
            X_list[0].get_dtype()
        )
        micro_gemm = create_micro_gemm(
            f"{kernel.kernel_name}_micro_gemm",
            self.m,
            self.n,
            self.k,
            input_dtype=X_list[0].get_dtype(),
            input2_dtype=W_list[0].get_dtype(),
            output_dtype=output_dtype,
            compute_dtype=compute_dtype,
            alpha=self.alpha,
            num_threads=self.num_threads,
        )
        if micro_gemm is None:
            raise AssertionError("expected micro_gemm to be not None")
        if self.register_blocking != micro_gemm.register_blocking:
            raise AssertionError(
                "expected self.register_blocking == micro_gemm.register_blocking"
            )
        self.log_blockings()
        if isinstance(micro_gemm, CppMicroGemmAMX):
            counters["inductor"]["cpp_micro_gemm_amx_counter"] += 1

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

        epilogues: list[ir.IRNode] = []
        reindexers: list[Callable[[list[Any]], list[Any]] | None] = []
        gemm_output_buffers: list[ir.Buffer] = []
        for out_buf_idx in range(self.gemm_grouped_num):
            gemm_output_name = f"{template_buffer.get_name()}_GemmOut" + str(
                out_buf_idx
            )
            gemm_output_buffers.append(
                ir.Buffer(name=gemm_output_name, layout=template_buffer.layout)
            )

        if self.epilogue_creator:
            raise AssertionError(
                "epilogue_creator is not supported yet in Grouped GEMM Template"
            )

        kernel_args: dict[str, ir.IRNode | None] = {}
        for x_idx in range(wgt_start_idx):
            kernel_args["X" + str(x_idx)] = act_deduplicated[x_idx]
        for w_idx in range(self.gemm_grouped_num):
            kernel_args["W" + str(w_idx)] = W_list[w_idx]
        for inp_idx in range(self.gemm_grouped_num):
            kernel_args["inp" + str(inp_idx)] = inp_list[inp_idx]

        def _bias_add_epilogue(buf: ir.IRNode, inp: ir.IRNode) -> ir.Pointwise:
            return create_epilogue_with_attr(
                buf, "bias_add", other=inp, beta=self.beta, dtype=self.layout.dtype
            )

        for gemm_idx, inp in enumerate(inp_list):
            if inp:
                buffer_name = Y_list[gemm_idx].get_name()
                epilogues.append(
                    ir.ComputedBuffer(
                        name=buffer_name,
                        layout=template_buffer.layout,
                        data=_bias_add_epilogue(gemm_output_buffers[gemm_idx], inp),
                    )
                )
                reindexers.append(None)

        if epilogue_nodes:
            epilogues.extend(epilogue_nodes)
            for epilogue_node in epilogue_nodes:
                Y = cast(ir.Buffer, epilogue_node)
                _, reindexers = gen_2d_view_of_epilogue_buf(
                    Y,
                    template_buffer,
                    [
                        epilogue_node,
                    ],
                    reindexers,
                    default_reindexers=[
                        None,
                    ],
                )

        options = dict(
            N=self.n,
            K=self.k,
            PADDED_N=self.padded_n,
            aliases={},
            beta=self.beta,
            alpha=self.alpha,
            num_threads=self.num_threads,
            micro_gemm=micro_gemm,
            is_dynamic_M=self.is_dynamic_M,
            template=self,
            kernel=kernel,
            export_declaration=get_export_declaration(),
            acc_buf_dtype=torch.float,
            L1_cache_size=L1_cache_size,
            L2_cache_size=L2_cache_size,
            config=config,
            epilogue_nodes=epilogues,
            GemmOuts=gemm_output_buffers,
            reindexers=reindexers,
            kernel_args=kernel_args,
            X_list=X_list,
            W_list=W_list,
            gemm_grouped_num=self.gemm_grouped_num,
            Y_list={"Y" + str(idx): Y for idx, Y in enumerate(Y_list)},
            Y_2d_list=Y_2d_list,
            multi_output_buffers=multi_output_buffers,
            cpu_count=os.cpu_count(),
        )
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch.object(V.graph, "get_dtype", self._fake_get_dtype(fake_buffers))
            )
            return self._template_from_string(GEMM_TEMPLATE).render(**options)
<|endoftext|># mypy: allow-untyped-defs
import ctypes
import functools
import itertools
import logging
import sys
from collections.abc import Callable, Iterable
from unittest.mock import patch

import sympy

from .. import config, ir
from ..autotune_process import CppBenchmarkRequest, TensorMeta
from ..utils import IndentedBuffer, Placeholder, unique
from ..virtualized import V
from .common import KernelTemplate
from .cpp_template_kernel import CppTemplateCaller, CppTemplateKernel


log = logging.getLogger(__name__)


class CppTemplate(KernelTemplate):
    index_counter = itertools.count()

    def __init__(
        self,
        name: str,
        input_nodes,
        layout: ir.Layout,
        num_threads: int,
        epilogue_creator: Callable[[ir.Buffer], ir.Pointwise] | None = None,
    ) -> None:
        super().__init__(name)
        self.input_nodes = input_nodes
        self.index = next(self.index_counter)
        self.output_node: ir.Buffer | list[ir.Buffer] = ir.Buffer(
            name=f"buf_out{self.index}", layout=layout
        )
        self.layout = layout
        self.num_threads = num_threads
        self.epilogue_creator = epilogue_creator

    def generate(self, **kwargs):
        kernel_name = f"cpp_{self.name}"
        with (
            patch.object(V.graph, "get_dtype", self._fake_get_dtype(self.output_node)),
            patch.object(ir.FlexibleLayout, "allow_indexing", True),
            V.graph.set_current_device(self.layout.device),
            CppTemplateKernel(
                kernel_name=kernel_name, num_threads=self.num_threads
            ) as kernel,
        ):
            code = kernel.render(self, **kwargs)
            _, call_args, _, _ = kernel.args.python_argdefs()
            log.debug("Generated Code:\n%s", code)
            log.debug(
                "Args: cpp_argdefs: %s, python_argdefs: %s",
                kernel.args.cpp_argdefs(),
                kernel.args.python_argdefs(),
            )

        expected_args = list(
            unique(input_node.get_name() for input_node in self.input_nodes)
        )
        if isinstance(self.output_node, Iterable):
            expected_args.extend([node.get_name() for node in self.output_node])
        else:
            expected_args.extend([self.output_node.get_name()])
        if list(call_args)[: len(expected_args)] != expected_args:
            raise AssertionError(
                (
                    call_args,
                    expected_args,
                )
            )
        # extra_args are only used for benchmarking, not compiled kernel correctness
        extra_args = V.graph.sizevars.optimization_hints(
            map(sympy.expand, call_args[len(expected_args) :])
        )
        # Cast the size hint from int to ctypes.c_ulonglong explicitly
        # since in cpp kernel, we bind it to C long
        extra_args = tuple(ctypes.c_ulonglong(x) for x in extra_args)

        kernel_hash_name = f"cpp_{self.name}_{self.index}"

        # Create the BenchmarkRequest for CPP
        bmreq = CppBenchmarkRequest(
            kernel_name=kernel_name,
            input_tensor_meta=TensorMeta.from_irnodes(self.input_nodes),
            output_tensor_meta=TensorMeta.from_irnodes(self.output_node),
            extra_args=extra_args,
            source_code=code,
        )

        def make_kernel_render(
            template_node: ir.CppTemplateBuffer,
            flag_template_buffer_has_other_users: bool,
            epilogue_nodes: list[ir.IRNode] | None = None,
        ):
            kernel = CppTemplateKernel(
                kernel_name=str(Placeholder.KERNEL_NAME), num_threads=self.num_threads
            )
            render = functools.partial(
                kernel.render,
                self,
                template_buffer_node=template_node,
                flag_template_buffer_has_other_users=flag_template_buffer_has_other_users,
                epilogue_nodes=epilogue_nodes,
                **kwargs,
            )
            return kernel, render

        return CppTemplateCaller(
            kernel_hash_name,
            self.name,
            self.input_nodes,
            # pyrefly: ignore [bad-index, index-error]
            self.output_node[0].get_layout()
            if isinstance(self.output_node, Iterable)
            else self.output_node.get_layout(),
            make_kernel_render,
            bmreq,
            self,
        )

    def header(self) -> IndentedBuffer:
        res = IndentedBuffer()
        res.writeline("#include <torch/csrc/inductor/cpp_prefix.h>")
        # TODO: add c10::ForcedUnroll test to test_aoti_abi_check
        res.splice("""#include <c10/util/Unroll.h>""")
        res.splice("""#include <torch/csrc/inductor/aoti_torch/c/shim.h>""")
        enable_kernel_profile = config.cpp.enable_kernel_profile and sys.platform in [
            "linux",
            "win32",
        ]
        if enable_kernel_profile:
            res.writelines(["#include <torch/csrc/inductor/aoti_runtime/utils.h>"])
        return res

    def render(self, **kwargs) -> str:
        raise NotImplementedError
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import functools
import logging
import os
from enum import Enum
from typing import TYPE_CHECKING

import torch
from torch import dtype as torch_dtype

from .. import config
from ..virtualized import V
from .multi_kernel import MultiKernel


if TYPE_CHECKING:
    from collections.abc import Callable


log = logging.getLogger(__name__)


def _print_debugging_tensor_value_info(msg, arg):
    # helper for printing debugging stats for intermediate tensor values
    # at jit inductor level codegen
    max_numel_to_print = 64
    print(msg)
    if not isinstance(arg, torch.Tensor):
        print("Value: ", arg)
        return
    numel = arg.numel()
    # print the debug printing stats
    if numel <= max_numel_to_print:
        print(arg)
    print("Number of elements: ", numel)
    print("Size: ", arg.size())
    print("Dtype: ", arg.dtype)
    arg_f = arg.float()
    std, mean = torch.std_mean(arg_f)
    amin, amax = torch.aminmax(arg_f)
    print("Mean: ", mean.item())
    print("Min: ", amin.item())
    print("Max: ", amax.item())
    print("Std: ", std.item())


# AOTI debug printing related configs
class IntermediateValueDebuggingLevel(Enum):
    # OFF: No intermediate tensor value debug info will be printed or saved.
    OFF = "0"
    # LEVEL 1: Save all intermediate tensor values to individual `.pt` files. No debug printing will be displayed.
    SAVE_ONLY = "1"
    # LEVEL 2: Print all intermediate tensor values by default to the console. No debug saving will be performed.
    PRINT_ONLY = "2"
    # LEVEL 3: Print all kernel names to the console only. No debug saving/printing for input tensor value info will be performed.
    # This mode can be helpful in cases when you just want to pinpoint what kernel is running into a CUDA IMA issue, etc.
    PRINT_KERNEL_NAMES_ONLY = "3"


class DebugPrinterManager:
    def __init__(
        self,
        debug_printer_level,
        use_array_ref: bool,
        writeline: Callable[..., None] | None = None,
        args_to_print_or_save: list[str] | None = None,
        kernel_name: str = "",
        kernel=None,
        arg_signatures: list[type] | None = None,
        kernel_type=None,
    ):
        self.debug_printer_level = IntermediateValueDebuggingLevel(debug_printer_level)
        self.use_array_ref = use_array_ref
        if args_to_print_or_save is None:
            args_to_print_or_save = []
        self.args_to_print_or_save = args_to_print_or_save
        self.kernel_name = kernel_name
        self.arg_signatures: list[type] | None = None
        self.kernel = kernel
        self.filtered_kernel_names_to_print = self._get_debug_filtered_kernel_names()
        self.kernel_type = None

    def __enter__(self):
        self._perform_debug_print_or_save_helper(
            self.args_to_print_or_save,
            self.kernel_name,
            before_launch=True,
            arg_signatures=self.arg_signatures,
        )

    def __exit__(self, args_to_print_or_save, kernel_name, arg_signatures):
        self._perform_debug_print_or_save_helper(
            args_to_print_or_save,
            kernel_name,
            before_launch=False,
            arg_signatures=arg_signatures,
        )

    def _perform_debug_print_or_save_helper(
        self,
        args_to_print_or_save,
        kernel_name,
        before_launch,
        arg_signatures: list[type] | None = None,
    ):
        if self.debug_printer_level == IntermediateValueDebuggingLevel.OFF:
            return
        if self.debug_printer_level == IntermediateValueDebuggingLevel.SAVE_ONLY:
            # by default save all the tensor values before launch
            self.codegen_intermediate_tensor_value_save(
                self.args_to_print_or_save,
                self.kernel_name,
                before_launch,
                arg_signatures=self.arg_signatures,
            )
        if self.debug_printer_level == IntermediateValueDebuggingLevel.PRINT_ONLY:
            # by default print all the tensor values before launch
            self.codegen_intermediate_tensor_value_print(
                self.args_to_print_or_save,
                self.kernel_name,
                before_launch,
                arg_signatures=self.arg_signatures,
            )
        if (
            self.debug_printer_level
            == IntermediateValueDebuggingLevel.PRINT_KERNEL_NAMES_ONLY
        ):
            # Print all kernel names to the console only
            self.codegen_intermediate_tensor_value_print(
                [],
                self.kernel_name,
                before_launch,
            )

    @functools.lru_cache  # noqa: B019
    def _get_debug_filtered_kernel_names(self) -> list[str]:
        if config.aot_inductor.filtered_kernel_names is None:
            return []
        return [
            x.strip()
            for x in config.aot_inductor.filtered_kernel_names.lower().split(",")
        ]

    def set_printer_args(
        self,
        args_to_print_or_save: list[str],
        kernel_name: str,
        arg_signatures: list[type] | None,
        kernel,
        kernel_type=None,
    ):
        # Note: MultiKernel debug printing is not supported for now
        if isinstance(kernel, MultiKernel):
            log.info(
                "MultiKernel type is not supported in AOTI debug printer tool yet."
            )
            self.debug_printer_level = IntermediateValueDebuggingLevel.OFF

        self.kernel_type = kernel_type
        # Note: if the kernel type is an extern kernel (or cpp kernel), we do a special handling to
        # get the list of args_to_print_or_save
        # TODO: Find a more reliable way to detect kernel args types to print for extern kernel calls
        if kernel_type == "extern":
            args_to_print_or_save_extern = [
                arg
                for arg in args_to_print_or_save
                if isinstance(arg, str) and arg.startswith(("buf", "arg"))
            ]
            self.args_to_print_or_save = args_to_print_or_save_extern
        elif kernel_type == "cpp":
            self.args_to_print_or_save = [
                (
                    f"copy_arrayref_tensor_to_tensor({arg})"
                    if self.use_array_ref
                    else arg
                )
                for arg in args_to_print_or_save
                if isinstance(arg, str) and arg.startswith(("buf", "arg"))
            ]
        else:
            self.args_to_print_or_save = args_to_print_or_save
        self.kernel_name = kernel_name
        self.arg_signatures = arg_signatures
        self.kernel = kernel

    def codegen_model_inputs_value_print(self, input_args_to_print: list[str]) -> None:
        if self.debug_printer_level != IntermediateValueDebuggingLevel.PRINT_ONLY:
            return
        for arg in input_args_to_print:
            if V.graph.cpp_wrapper:
                V.graph.wrapper_code.prefix.writeline(
                    f'aoti_torch_print_tensor_handle({arg}, "aoti_model_inputs - {arg}");'
                )

    def codegen_intermediate_tensor_value_save(
        self,
        args_to_save,
        kernel_name,
        before_launch=True,
        arg_signatures: list[type] | None = None,
    ) -> None:
        for i, arg in enumerate(args_to_save):
            if arg_signatures is not None and not isinstance(
                arg_signatures[i], torch_dtype
            ):
                # infer from the arg data type (has torch.dtype) to see if it is a tensor type
                continue
            launch_prefix = "before_launch" if before_launch else "after_launch"
            if V.graph.cpp_wrapper:
                V.graph.wrapper_code.writeline(
                    f'aoti_torch_save_tensor_handle({arg}, "{arg}", "{launch_prefix}", "{kernel_name}");'
                )
            else:
                cwd = os.getcwd()
                saved_dir = cwd + "/tmp/jit_inductor/"
                if not os.path.exists(saved_dir):
                    log.info(
                        "Creating directory to save inductor intermediate tensor values."
                    )
                    os.makedirs(saved_dir)
                # Save the model to the directory
                saved_path = saved_dir + f"{launch_prefix}_{kernel_name}_{arg}.pt"
                log.info(
                    "Saved intermediate tensor %s for %s to %s",
                    arg,
                    kernel_name,
                    saved_path,
                )
                line = f"torch.save({arg}, '{saved_path}')"
                V.graph.wrapper_code.writeline(line)

    def codegen_intermediate_tensor_value_print(
        self,
        args_to_print,
        kernel_name,
        before_launch=True,
        arg_signatures: list[type] | None = None,
    ) -> None:
        launch_prefix = "before_launch" if before_launch else "after_launch"

        # if the debug printing level is PRINT_KERNEL_NAMES_ONLY
        # we only print the kernel name to the console
        if (
            self.debug_printer_level
            == IntermediateValueDebuggingLevel.PRINT_KERNEL_NAMES_ONLY
        ):
            if V.graph.cpp_wrapper:
                V.graph.wrapper_code.writeline(
                    f'printf("[ {launch_prefix}: {kernel_name} ]\\n");'
                )
            return

        if self.debug_printer_level != IntermediateValueDebuggingLevel.PRINT_ONLY:
            return
        for i, arg in enumerate(args_to_print):
            # when debug printing is enabled i.e. IntermediateValueDebuggingLevel.PRINT_ONLY,
            # check if filtered kernel name list is provided
            if (
                len(self.filtered_kernel_names_to_print) > 0
                and kernel_name.lower() not in self.filtered_kernel_names_to_print
            ):
                continue
            if V.graph.cpp_wrapper:
                if arg_signatures is not None and isinstance(
                    arg_signatures[i], torch_dtype
                ):
                    # infer from the arg data type (has torch.dtype) to see if it is a tensor type
                    V.graph.wrapper_code.writeline(
                        f'aoti_torch_print_tensor_handle({arg}, "{launch_prefix} - {kernel_name} - {arg}");'
                    )
                elif arg_signatures is not None and isinstance(
                    arg_signatures[i],
                    (
                        type(torch._inductor.codegen.wrapper.SymbolicCallArg),
                        type(int),
                        type(float),
                        type(bool),
                    ),
                ):
                    V.graph.wrapper_code.writeline(
                        f'printf("[  {launch_prefix} - {kernel_name} - {arg}: %ld  ]", {arg}); printf("\\\\n");'
                    )
                else:
                    if arg_signatures is None and self.kernel_type in ("cpp", "extern"):
                        V.graph.wrapper_code.writeline(
                            f'aoti_torch_print_tensor_handle({arg}, "{launch_prefix} - {kernel_name} - {arg}");'
                        )
            else:
                V.graph.wrapper_code.writeline(
                    f'_print_debugging_tensor_value_info("inductor: {launch_prefix} - {kernel_name} - {arg}", {arg})'
                )
<|endoftext|>import dataclasses
import functools
import logging
import operator
import textwrap
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import sympy

import torch
from torch._export.passes._node_metadata_hook import (
    _node_metadata_hook,
    _set_node_metadata_hook,
)
from torch._higher_order_ops.triton_kernel_wrap import (
    TraceableTritonKernelWrapper,
    tracing_triton_hopifier_singleton,
    triton_kernel_wrapper_mutation,
)
from torch._inductor.codecache import LambdaFuture, PyCodeCache
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from torch._inductor.select_algorithm import extern_kernels  # noqa: F401
from torch._inductor.utils import convert_to_symint
from torch._inductor.virtualized import V
from torch._library.triton import wrap_triton
from torch.fx import GraphModule
from torch.fx.experimental.symbolic_shapes import (
    CallMethodKey,
    ConvertIntKey,
    DivideByKey,
)
from torch.utils import _pytree as pytree
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.interp import _run_sympy_handler, sympy_interp
from torch.utils._sympy.reference import OptimizedPythonReferenceAnalysis
from torch.utils._sympy.solve import try_solve

from .. import config, ir
from ..runtime.triton_compat import Config
from ..utils import cache_property_on_self, LineContext, ValueWithLineMap
from .common import (
    CodegenSymbol,
    FileBackedGraphModule,
    WorkspaceArg,
    WorkspaceZeroMode,
)
from .wrapper import (
    AllocateLine,
    BufferLike,
    CommentLine,
    DynamicScalarLine,
    EnterDeviceContextManagerLine,
    EnterSubgraphLine,
    ExitDeviceContextManagerLine,
    ExitSubgraphLine,
    ExternKernelAllocLine,
    ExternKernelOutLine,
    FreeIfNotReusedLine,
    FreeLine,
    IndexPutFallbackLine,
    KernelCallLine,
    KernelDefinitionLine,
    Line,
    MultiOutputLine,
    NullLine,
    PythonWrapperCodegen,
    ReinterpretLine,
    ReuseLine,
    ScatterFallbackLine,
    SubgraphPythonWrapperCodegen,
    SwitchLine,
    SymbolicCallArg,
    SymbolicCallArgLine,
    UnbackedSymbolDefsLine,
    WrapperLine,
)


aten = torch.ops.aten
log = logging.getLogger(__name__)


@dataclasses.dataclass
class SymbolBuffer(CodegenSymbol):
    """
    Represents a symbolic graph input. Expressions more complex than a single
    sympy.Symbol require a name.
    """

    expr: sympy.Expr
    name: str | None = None

    def get_name(self) -> str:
        if self.name is not None:
            return self.name
        if not isinstance(self.expr, sympy.Symbol):
            raise AssertionError(f"expression requires a name: {self.expr}")
        return str(self.expr)

    def get_example(self) -> torch.Tensor | torch.SymInt:
        sym_int = convert_to_symint(self.expr)
        if not isinstance(sym_int, torch.SymInt):
            raise AssertionError(f"expected torch.SymInt, got {type(sym_int)}")
        return sym_int


CodegenBuffer = BufferLike | SymbolBuffer


@dataclasses.dataclass
class TritonKernel:
    """
    Stores metadata about Triton kernels for use in FX.
    """

    tuner: CachingAutotuner
    wrapped: TraceableTritonKernelWrapper


def replace_floor_div(expr: sympy.Expr) -> sympy.Expr:
    """
    Replace sympy.floor with FloorDiv.
    """

    def replace(expr: sympy.Expr) -> sympy.Expr:
        expr = sympy.together(expr)

        # Division is represented as a Mul with a Rational factor or a Pow with negative
        # exponent. We convert floor(Mul(...)) to FloorDiv(numerator, denominator) by
        # partitioning factors into the numerator and denominator.
        (numerator, denominator) = (sympy.S.One,) * 2
        for arg in sympy.Mul.make_args(expr):
            if isinstance(arg, sympy.Rational):
                numerator *= arg.numerator
                denominator *= arg.denominator
            elif isinstance(arg, sympy.Pow) and arg.exp.is_negative:
                denominator *= arg.base**-arg.exp
            else:
                numerator *= arg

        return FloorDiv(numerator, denominator)

    return expr.replace(sympy.floor, replace)


class WrapperFxCodegen(PythonWrapperCodegen):
    """
    Backend to generate wrapper code as an FX IR graph.
    """

    supports_caching = False

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.subgms: dict[str, torch.fx.GraphModule] = {}

    def codegen_inputs(self) -> None:
        """
        This would generate code for symbolic input shapes, strides, etc.
        Since the FX converter handles this, do nothing here.
        """

    def codegen_switch(self, node: ir.Switch) -> None:
        """
        Switch/cond codegen normally emits a number of different wrapper lines.
        Instead, FX conversion uses a dedicated line for the whole node.
        """
        self.writeline(SwitchLine(self, node))
        for subgraph in node.branches:  # pyrefly: ignore [not-iterable]
            self.codegen_subgraph_common(subgraph)

    def define_subgraph_launcher_fn(
        self, name: str, subgraph_code: ValueWithLineMap | FileBackedGraphModule
    ) -> None:
        """
        Record subgms as they're generated.
        """
        if not isinstance(subgraph_code, FileBackedGraphModule):
            raise AssertionError(
                f"expected FileBackedGraphModule, got {type(subgraph_code)}"
            )
        self.subgms[name] = subgraph_code.gm

    @property
    @cache_property_on_self
    def is_subgraph(self) -> bool:
        return isinstance(self, SubgraphPythonWrapperCodegen)

    def get_fx_graph_inputs(
        self,
    ) -> dict[str, ir.TensorBox | ir.TorchBindObject | sympy.Expr | None]:
        """
        Get the input nodes corresponding to FX graph placeholders.
        """

        if V.aot_compilation and not self.is_subgraph:
            # AOT graphs must match the signature of the input module.
            return {
                node.name: V.graph.graph_inputs.get(node.name)
                for node in V.graph.module.graph.find_nodes(op="placeholder")  # type: ignore[operator, union-attr]
            }

        return self.get_graph_inputs()

    def _generate(self, is_inference: bool) -> tuple[FileBackedGraphModule, None]:
        self.run_wrapper_ir_passes(is_inference)

        prologue = "\n".join(
            [
                self.imports.getvalue(),
                self.header.getvalue(),
            ]
        )
        gm = FxConverter(
            lines=self.lines,
            prologue=prologue,
            graph_inputs=self.get_fx_graph_inputs(),
            graph_outputs=self.get_graph_outputs(),
            subgms=self.subgms,
            is_subgraph=self.is_subgraph,
        ).generate()

        compiled_fn = self.compile_graph(gm)

        return FileBackedGraphModule(gm, compiled_fn), None

    def compile_graph(self, gm: GraphModule) -> Callable[..., Any]:
        """
        Converts the graph module into a runnable function. The default implementation
        is simply an interpreter calling kernels in eager mode. Derived backends can
        override this to do further compilation.
        """
        return gm.forward

    def write_header(self) -> None:
        """
        Python subgraphs normally lack headers.
        Override this behavior to generate prologues for FX subgraphs.
        """
        PythonWrapperCodegen.write_header(self)

    def register_alignment_check_inputs(self) -> None:
        """FXIR does not emit deferred alignment copies.
        Alignment is handled by the runtime wrapper."""

    def codegen_deferred_alignment_copies(
        self, input_names: Iterable[str], stream: int = 0
    ) -> None:
        """FXIR does not emit deferred alignment copies."""

    @classmethod
    def create(
        cls: type["WrapperFxCodegen"],
        is_subgraph: bool,
        subgraph_name: str | None,
        parent_wrapper: PythonWrapperCodegen | None,
        partition_signatures: ir.GraphPartitionSignature | None = None,
    ) -> "WrapperFxCodegen":
        if is_subgraph:
            if subgraph_name is None:
                raise AssertionError("subgraph_name must not be None for subgraphs")
            if parent_wrapper is None:
                raise AssertionError("parent_wrapper must not be None for subgraphs")

            # Subgraphs override some methods of PythonWrapperCodegen.
            # Apply these overrides to the user-provided class, with priority given to
            # user-provided methods.
            class SubgraphFxWrapperCodegen(cls, SubgraphPythonWrapperCodegen):  # type: ignore[misc,valid-type]
                def compile_graph(self, gm: GraphModule) -> Callable[..., Any]:
                    """
                    Skip graph compilation for subgraphs.
                    """

                    def crash_if_run(*args: Any) -> None:
                        raise NotImplementedError("Cannot run a subgraph in isolation!")

                    return crash_if_run

            return SubgraphFxWrapperCodegen(
                subgraph_name, parent_wrapper, partition_signatures
            )

        return cls()


@dataclasses.dataclass
class FxConverter:
    """
    Generates FX IR from Wrapper IR. As each instance is only meant to be used once, the
    input and output code are stored as attributes.
    """

    lines: list[Line]
    prologue: str
    graph_inputs: dict[str, ir.TensorBox | ir.TorchBindObject | sympy.Expr | None]
    graph_outputs: list[ir.IRNode]
    subgms: dict[str, torch.fx.GraphModule]
    is_subgraph: bool

    def __post_init__(self) -> None:
        graph = torch.fx.Graph()
        self.gm = GraphModule({}, graph)  # Wrapper FX IR.
        self.buffer_to_node: dict[
            str | None, torch.fx.Node
        ] = {}  # Symbol table for codegen.
        self.kernels: dict[str, TritonKernel] = {}  # Table to store Triton kernels.
        self._unique_symbol_ids: Counter[str] = Counter()
        self.tracer = torch.fx.proxy.GraphAppendingTracer(graph)
        self.expr_to_proxy: dict[sympy.Expr, torch.fx.Proxy] = {}

    def _import_kernel(self, code: str, kernel_name: str) -> CachingAutotuner:
        """
        Imports a kernel from source, possibly autotuning block parameters.
        """
        module_code = "\n".join([self.prologue, code])
        mod = PyCodeCache.load(module_code)
        kernel = getattr(mod, kernel_name)

        if isinstance(kernel, LambdaFuture):
            kernel = kernel.result()

        if not isinstance(kernel, CachingAutotuner):
            raise NotImplementedError(
                textwrap.dedent(f"""
                Unsupported type for kernel {kernel_name}: {type(kernel)}.
                FX conversion only supports Triton kernels.
            """)
            )

        # Parallel compile workers strip the Python function from the pickled
        # JITFunction. FXIR stores the JITFunction in the Triton HOP side table,
        # so reload it in the parent before runtime execution can need it.
        kernel._ensure_kernel_loaded()

        return kernel

    def _create_as_strided(
        self,
        input_node: torch.fx.Node,
        size: tuple[Any, ...],
        stride: tuple[Any, ...],
        offset: int | sympy.Expr,
    ) -> torch.fx.Node:
        if isinstance(offset, sympy.Expr):
            offset = replace_floor_div(offset)
        return self.gm.graph.call_function(
            torch.as_strided,
            args=(
                input_node,
                self._generate_sym_nodes(size),
                self._generate_sym_nodes(stride),
                self._generate_sym_node(offset),
            ),
        )

    def _record_allocation(self, buffer: CodegenBuffer, node: torch.fx.Node) -> None:
        """
        Updates the symbol table to record that an Inductor buffer maps to the result of
        an FX node.
        """
        if node in self.buffer_to_node:
            raise AssertionError(f"node already recorded in buffer_to_node: {node}")
        self.buffer_to_node[buffer.get_name()] = node

    def _free(self, buffer: CodegenBuffer | ir.TorchBindObject) -> None:
        """
        Removes the buffer from the symbol table.
        """
        name = buffer.get_name()
        del self.buffer_to_node[name]

    def _lookup_args(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        """
        Maps call args back to FX nodes.
        """
        return tuple(
            self.buffer_to_node[arg]
            if isinstance(arg, str)
            else arg.inner_expr
            if isinstance(arg, SymbolicCallArg)
            else arg
            for arg in args
        )

    def _get_buffer(self, node: ir.IRNode) -> CodegenBuffer:
        """
        Extract buffer data from an IR node.
        """
        if isinstance(node, (ir.Buffer, WorkspaceArg)):
            return node
        elif isinstance(node, (ir.BaseView, ir.MutableBox)):
            return self._get_buffer(node.data)
        elif isinstance(node, sympy.Symbol):
            return SymbolBuffer(node)
        else:
            raise NotImplementedError(f"Unable to extract buffer from node: {node}")

    def _generate_size_proxy(
        self, node: torch.fx.Node, expr: sympy.Expr
    ) -> torch.fx.Proxy:
        proxy = torch.fx.Proxy(node, tracer=self.tracer)
        self.expr_to_proxy[expr] = proxy
        return proxy

    def _generate_graph_inputs(self) -> None:
        """
        Converts graph inputs to FX placeholders.
        """

        for name, ir_node in self.graph_inputs.items():
            if ir_node is None:
                # Create dummy input nodes to match the input signature
                self.gm.graph.placeholder(name)
                continue

            # Introduce a new symbol for constant inputs.
            is_constant = isinstance(ir_node, (int, float, sympy.Integer, sympy.Float))

            # Special handling for dynamic shapes which are not simple symbols.
            is_expr = isinstance(ir_node, sympy.Expr) and not isinstance(
                ir_node, (sympy.Symbol, sympy.Integer, sympy.Float)
            )
            if is_expr and ir_node.is_integer is not True:
                raise NotImplementedError(
                    f"Unsupported non-integer symbolic graph input: {ir_node}"
                )

            if is_constant:
                buffer = SymbolBuffer(sympy.Symbol(name, is_integer=True))
            elif is_expr:
                buffer = SymbolBuffer(ir_node, name=name)
            else:
                buffer = self._get_buffer(ir_node)
            placeholder_node = self.gm.graph.placeholder(buffer.get_name())
            placeholder_node.meta["val"] = (
                ir_node if is_constant else buffer.get_example()
            )
            self._record_allocation(buffer, placeholder_node)

            # Record symbol definitions for dynamic shapes.
            if isinstance(ir_node, sympy.Expr) and not is_constant:
                self._generate_size_proxy(placeholder_node, ir_node)

    def _generate_graph_input_shapes(self) -> None:
        """
        Generate nodes creating symints that are part of graph input
        shape/strides.
        """

        def _codegen_symbol(
            sym_or_exp: sympy.Symbol | sympy.Expr,
            base_node: torch.fx.Node,
            target: torch._ops.OpOverload,
            dim: int,
        ) -> None:
            def codegen_proxy() -> torch.fx.Proxy:
                size_node = self.gm.graph.call_function(target, (base_node, dim))
                size_proxy = self._generate_size_proxy(size_node, sym_or_exp)
                return size_proxy

            if isinstance(sym_or_exp, sympy.Symbol):
                if sym_or_exp in self.expr_to_proxy:
                    return
                codegen_proxy()

            elif isinstance(sym_or_exp, sympy.Integer):
                return

            elif isinstance(sym_or_exp, sympy.Expr):
                # Check if we need to solve for an undefined symbol.
                undefined_symbols = [
                    sym
                    for sym in sym_or_exp.free_symbols
                    if sym not in self.expr_to_proxy
                ]
                if len(undefined_symbols) == 0:
                    self._sympy_interp(sym_or_exp)
                    return
                elif len(undefined_symbols) > 1:
                    raise NotImplementedError(
                        f"Underdetermined input expression: {sym_or_exp}"
                    )

                # Define a new symbol for the input size.
                size_proxy = codegen_proxy()
                size_symbol = sympy.Symbol(
                    size_proxy.node.name, integer=True, nonnegative=True
                )
                self.expr_to_proxy[size_symbol] = size_proxy

                self._define_symbol_by_solving(
                    sym_or_exp, size_symbol, undefined_symbols[0]
                )

        for ir_node in self.graph_inputs.values():
            if isinstance(ir_node, ir.TensorBox):
                buffer = self._get_buffer(ir_node)
                placeholder_node = self.buffer_to_node[buffer.get_name()]

                for dim, size in enumerate(ir_node.get_size()):
                    _codegen_symbol(
                        size, placeholder_node, torch.ops.aten.sym_size.int, dim
                    )
                for dim, stride in enumerate(ir_node.get_stride()):
                    _codegen_symbol(
                        stride, placeholder_node, torch.ops.aten.sym_stride.int, dim
                    )

        # A compound input binds its whole expression to one placeholder, so a
        # symbol appearing only inside it has no node of its own. Recover it
        # from that placeholder, after the loop above has defined every symbol
        # the tensor shapes determine.
        for ir_node in self.graph_inputs.values():
            if not isinstance(ir_node, sympy.Expr) or isinstance(ir_node, sympy.Symbol):
                continue
            proxy = self.expr_to_proxy.get(ir_node)
            if proxy is None:
                continue
            undefined = [
                sym for sym in ir_node.free_symbols if sym not in self.expr_to_proxy
            ]
            if len(undefined) == 0:
                continue
            elif len(undefined) > 1:
                # One bound value cannot determine several symbols. This is a
                # branch closing over an expression whose symbols come from
                # tensors it does not take, such as y.shape[0] + z.shape[0].
                raise NotImplementedError(
                    f"Compound input {ir_node} leaves these symbols undefined: "
                    f"{sorted(undefined, key=str)}"
                )

            # A tensor-derived FX node does not reserve its symbol's name, so a
            # Dummy is what keeps the anchor from colliding with one.
            anchor = sympy.Dummy(proxy.node.name, integer=True)
            self.expr_to_proxy[anchor] = proxy
            self._define_symbol_by_solving(ir_node, anchor, undefined[0])

    def _generate_graph_constants(self) -> None:
        for name, value in V.graph.constants.items():
            node = self.gm.graph.get_attr(name)
            node.meta["val"] = value
            setattr(self.gm, name, value)
            self.buffer_to_node[name] = node

    def _generate_buffer(self, node: ir.IRNode) -> torch.fx.Node | None:
        """
        Generates FX IR for transformations on a buffer, such as ReinterpretView.
        Does nothing if no such transformations are present.
        """

        if isinstance(node, ir.ShapeAsConstantBuffer):
            # Generate FX nodes to compute the shape expression.
            return self._sympy_interp(node.expr).node

        def generate_to_buffer(node: ir.IRNode) -> BufferLike | None:
            if isinstance(node, (ir.Buffer, WorkspaceArg)):
                return node
            elif isinstance(node, ir.NoneAsConstantBuffer):
                return None
            elif isinstance(node, ir.MutableBox):
                return generate_to_buffer(node.data)
            elif isinstance(node, ir.ReinterpretView):
                # We need to introduce a new symbol if the output is a ReinterpretView.
                # Use a WorkspaceArg for this.
                buffer = self._get_buffer(node.data)
                if not isinstance(buffer, (ir.Buffer, WorkspaceArg)):
                    raise AssertionError(
                        f"expected ir.Buffer or WorkspaceArg, got {type(buffer)}"
                    )
                unique_name = self.gm.graph._graph_namespace.create_name(
                    f"{buffer.get_name()}_view", None
                )
                device = buffer.get_device()
                if not device:
                    raise AssertionError(f"buffer has no device: {buffer}")
                reused_as = WorkspaceArg(
                    count=buffer.get_size(),
                    zero_mode=WorkspaceZeroMode.UNINITIALIZED,
                    device=device,
                    outer_name=unique_name,
                    dtype=buffer.get_dtype(),
                )

                # Generate FX IR for the view.
                self._generate_reinterpret_helper(buffer, reused_as, node.layout)

                return reused_as
            else:
                raise NotImplementedError(f"Unrecognized buffer/view node: {node}")

        buffer = generate_to_buffer(node)
        return self.buffer_to_node[buffer.get_name()] if buffer is not None else None

    def _generate_outputs(
        self,
    ) -> torch.fx.Node | None | list[torch.fx.Node | None]:
        """
        Generate FX IR for graph outputs.
        """
        output_nodes = [
            self._generate_buffer(node) for idx, node in enumerate(self.graph_outputs)
        ]

        # Parent graphs with single return elements don't use a tuple.
        output_value = (
            output_nodes[0]
            if len(output_nodes) == 1 and not self.is_subgraph
            else output_nodes
        )

        return output_value

    def _generate_subgm_getattrs(self) -> None:
        """
        Generate getattr nodes for subgms.
        """

        def generate_getattr(name: str, subgm: torch.fx.GraphModule) -> torch.fx.Node:
            self.gm.add_submodule(name, subgm)
            node = self.gm.graph.get_attr(name)
            node.meta["val"] = subgm
            return node

        self.subgm_getattrs = {
            name: generate_getattr(name, subgm) for name, subgm in self.subgms.items()
        }

    def _get_subgm_attr(self, subgraph: ir.Subgraph) -> torch.fx.Node:
        """
        Look up the getattr node for a subgraph.
        """
        graph = subgraph.graph
        if graph is None:
            raise AssertionError("subgraph.graph must not be None")
        return self.subgm_getattrs[graph.name]

    def generate(self) -> torch.fx.GraphModule:
        """
        Main entrypoint for FX codegen.
        """
        self._generate_graph_inputs()
        self._generate_graph_constants()
        self._generate_subgm_getattrs()

        with _set_node_metadata_hook(
            self.gm,
            functools.partial(_node_metadata_hook, fake_mode=V.fake_mode),
        ):
            self._generate_graph_input_shapes()

            # Generate FX IR from Wrapper IR lines.
            for line in self.lines:
                if isinstance(line, WrapperLine):
                    line.codegen_fx(self)(line)
                elif isinstance(line, LineContext):
                    # Ignore line context in FX IR.
                    pass
                else:
                    raise NotImplementedError(
                        textwrap.dedent(
                            f"""
                        Found line of unrecognized type '{type(line)}':
                            '{line}'

                        FX conversion only supports Wrapper IR lines.
                        """
                        )
                    )

            output = self._generate_outputs()

        self.gm.graph.output(output)
        self.gm.recompile()
        return self.gm

    def _sympy_interp(self, expr: sympy.Expr) -> torch.fx.Proxy:
        # hash cons
        if expr in self.expr_to_proxy:
            return self.expr_to_proxy[expr]
        # base cases, don't cache
        if isinstance(
            expr,
            (
                sympy.Integer,
                sympy.Number,
                sympy.Symbol,
                sympy.logic.boolalg.BooleanAtom,
            ),
        ):
            return sympy_interp(
                OptimizedPythonReferenceAnalysis, self.expr_to_proxy, expr
            )

        # hash cons on arguments, run expr handler
        self.expr_to_proxy[expr] = _run_sympy_handler(
            OptimizedPythonReferenceAnalysis,
            [self._sympy_interp(arg) for arg in expr.args],
            expr,
        )
        return self.expr_to_proxy[expr]

    def _define_symbol_by_solving(
        self, expr: sympy.Expr, anchor: sympy.Symbol, symbol: sympy.Symbol
    ) -> None:
        """
        Define symbol by solving expr == anchor, which already maps to a proxy.
        """
        # A returned solution directly defines the symbol in terms of the anchor.
        solution = try_solve(sympy.Eq(expr, anchor), symbol)
        if solution is None:
            raise NotImplementedError(
                f"Cannot solve input expression {expr} for {symbol}"
            )

        symbol_expr = solution[1]
        # If the symbol is an integer, division becomes FloorDiv.
        if symbol.is_integer:
            symbol_expr = replace_floor_div(sympy.floor(symbol_expr))

        # Generate FX for the symbol.
        self._sympy_interp(symbol_expr)
        self.expr_to_proxy[symbol] = self.expr_to_proxy[symbol_expr]

    def _generate_sym_node(self, s: int | sympy.Expr) -> int | torch.fx.Node:
        if isinstance(s, (int, sympy.Integer)):
            return int(s)
        elif isinstance(s, sympy.Symbol):
            if s not in self.expr_to_proxy:
                raise AssertionError(
                    f"Could not find a node corresponding to the symbol {s}"
                )
            return self.expr_to_proxy[s].node
        elif isinstance(s, sympy.Expr):
            return self._sympy_interp(s).node

        elif isinstance(s, torch.fx.Node):
            return s

        else:
            raise ValueError(f"{s} of type {type(s)} is not a valid input")

    def _generate_sym_nodes(
        self, shape: Sequence[sympy.Expr]
    ) -> list[int | torch.fx.Node]:
        return [self._generate_sym_node(s) for s in shape]

    def _generate_allocate(self, line: WrapperLine) -> None:
        if not isinstance(line, AllocateLine):
            raise AssertionError(f"expected AllocateLine, got {type(line)}")
        buffer = line.node
        name = buffer.get_name()
        if name in V.graph.removed_buffers:
            raise AssertionError(f"buffer {name} is in removed_buffers")

        device = buffer.get_device()
        if not device:
            raise AssertionError(f"buffer has no device: {buffer}")
        dtype = buffer.get_dtype()
        shape = self._generate_sym_nodes(buffer.get_size())
        stride = self._generate_sym_nodes(buffer.get_stride())

        node = self.gm.graph.call_function(
            torch.empty_strided,
            args=(shape, stride),
            kwargs={"dtype": dtype, "device": device.type},
        )
        if not name:
            raise AssertionError(f"buffer has an empty name: {buffer}")
        node.name = name
        self._record_allocation(buffer, node)

    def _generate_switch(self, line: WrapperLine) -> None:
        if not isinstance(line, SwitchLine):
            raise AssertionError(f"expected SwitchLine, got {type(line)}")

        ir_node = line.node
        if ir_node.branches is None:
            raise AssertionError("ir_node.branches must not be None")
        if ir_node.operands is None:
            raise AssertionError("ir_node.operands must not be None")

        def generate_buffer(node: ir.IRNode | None) -> torch.fx.Node | None:
            if node is None:
                raise AssertionError("node must not be None")
            return self._generate_buffer(node)

        selector = generate_buffer(ir_node.selector)
        operands = tuple(generate_buffer(arg) for arg in ir_node.operands)
        branch_subgms = [self._get_subgm_attr(branch) for branch in ir_node.branches]

        if ir_node.is_cond:
            # cond expects (selector, true_fn, false_fn, operands) -- branches unpacked positionally.
            if len(branch_subgms) != 2:
                raise AssertionError(
                    f"cond requires exactly 2 branches, got {len(branch_subgms)}"
                )
            # branches are stored as [false_fn, true_fn] in ir.Switch
            false_subgm, true_subgm = branch_subgms
            fx_node = self.gm.graph.call_function(
                torch.ops.higher_order.cond,
                args=(selector, true_subgm, false_subgm, operands),
            )
        else:
            # switch expects (selector, [branch_fn, ...], operands) -- branches passed as a list.
            fx_node = self.gm.graph.call_function(
                torch.ops.higher_order.switch,
                args=(selector, branch_subgms, operands),
            )
        self._record_allocation(ir_node, fx_node)

    def _generate_assert_size_stride(self, line: WrapperLine) -> None:
        pass

    def _generate_comment(self, line: WrapperLine) -> None:
        if not isinstance(line, CommentLine):
            raise AssertionError(f"expected CommentLine, got {type(line)}")
        # We ignore comments in FX IR.

    def _generate_dynamic_scalar(self, line: WrapperLine) -> None:
        if not isinstance(line, DynamicScalarLine):
            raise AssertionError(f"expected DynamicScalarLine, got {type(line)}")

        ir_node = line.node
        (input_ir_node,) = ir_node.inputs
        if not isinstance(input_ir_node, ir.IRNode):
            raise AssertionError(f"expected ir.IRNode, got {type(input_ir_node)}")
        input_fx_node = self._generate_buffer(input_ir_node)
        keypath = ir_node.keypath
        graph = self.gm.graph

        def generate_item(x: torch.fx.Node | None) -> torch.fx.Node:
            if x is None:
                raise AssertionError("x must not be None")
            return graph.call_function(
                aten.item.default,
                args=(x,),
            )

        if len(keypath) == 0:
            result_fx_node = generate_item(input_fx_node)
        elif len(keypath) == 1 and isinstance(keypath[0], ConvertIntKey):
            where_fx_node = graph.call_function(
                aten.where.Scalar,
                args=(input_fx_node, 1, 0),
            )
            result_fx_node = generate_item(where_fx_node)
        else:
            raise NotImplementedError(f"Unsupported keypath: {keypath}")

        result_symbol = ir_node.sym
        result_buffer = SymbolBuffer(result_symbol)
        self._record_allocation(result_buffer, result_fx_node)
        self._generate_size_proxy(result_fx_node, result_symbol)

    def _generate_enter_device_context_manager(self, line: WrapperLine) -> None:
        if not isinstance(line, EnterDeviceContextManagerLine):
            raise AssertionError(
                f"expected EnterDeviceContextManagerLine, got {type(line)}"
            )
        # We ignore the device context in FX IR.

    def _generate_exit_device_context_manager(self, line: WrapperLine) -> None:
        if not isinstance(line, ExitDeviceContextManagerLine):
            raise AssertionError(
                f"expected ExitDeviceContextManagerLine, got {type(line)}"
            )
        # We ignore the device context in FX IR.

    def _generate_enter_subgraph(self, line: WrapperLine) -> None:
        if not isinstance(line, EnterSubgraphLine):
            raise AssertionError(f"expected EnterSubgraphLine, got {type(line)}")
        # We ignore memory planning lines in FX IR.

    def _generate_exit_subgraph(self, line: WrapperLine) -> None:
        if not isinstance(line, ExitSubgraphLine):
            raise AssertionError(f"expected ExitSubgraphLine, got {type(line)}")
        # We ignore memory planning lines in FX IR.

    def _generate_free(self, line: WrapperLine) -> None:
        if not isinstance(line, FreeLine):
            raise AssertionError(f"expected FreeLine, got {type(line)}")

        buf = line.node

        # No need to free placeholders.
        if self.buffer_to_node[buf.get_name()].op == "placeholder":
            return

        self._free(buf)

    def _generate_free_if_not_reused(self, line: WrapperLine) -> None:
        if not isinstance(line, FreeIfNotReusedLine):
            raise AssertionError(f"expected FreeIfNotReusedLine, got {type(line)}")
        buf = line.node
        if buf.get_name() in V.graph.removed_buffers:
            raise AssertionError(f"buffer {buf.get_name()} is in removed_buffers")
        if not line.is_reused:
            self._free(buf)

    def _generate_line_context(self, line: WrapperLine) -> None:
        if not isinstance(line, LineContext):
            raise AssertionError(f"expected LineContext, got {type(line)}")
        # We ignore line context in FX IR.

    def _generate_reinterpret(self, line: WrapperLine) -> None:
        if not isinstance(line, ReinterpretLine):
            raise AssertionError(f"expected ReinterpretLine, got {type(line)}")
        self._generate_reinterpret_helper(line.node, line.reused_as, line.layout)

    def _generate_reinterpret_helper(
        self, input_buffer: BufferLike, result_buffer: BufferLike, layout: ir.Layout
    ) -> None:
        input_node = self.buffer_to_node[input_buffer.get_name()]

        # Look up output metadata.
        name = result_buffer.get_name()
        if not name:
            raise AssertionError(f"result_buffer has an empty name: {result_buffer}")
        size = tuple(layout.size)
        stride = tuple(layout.stride)
        if isinstance(layout, ir.NonOwningLayout):
            # Look up the view's layout.
            view = layout.view
            if not isinstance(view, ir.ReinterpretView):
                raise AssertionError(f"unexpected type: {type(view)}")
            layout = view.layout
        offset = input_buffer.get_offset() + layout.offset

        # Map ReinterpretView to as_strided.
        result_node = self._create_as_strided(input_node, size, stride, offset)
        result_node.name = name
        self._record_allocation(result_buffer, result_node)

    def _generate_reuse(self, line: WrapperLine) -> None:
        if not isinstance(line, ReuseLine):
            raise AssertionError(f"expected ReuseLine, got {type(line)}")
        old = line.node
        new = line.reused_as
        if any(buf.get_name() in V.graph.removed_buffers for buf in (old, new)):
            raise AssertionError("old or new buffer is in removed_buffers")
        if old.get_dtype() != new.get_dtype():
            raise AssertionError(
                f"dtype mismatch: {old.get_dtype()} != {new.get_dtype()}"
            )

        old_node = self.buffer_to_node[old.get_name()]
        result_node = old_node

        # Change shape and stride.
        size = tuple(new.get_size())
        stride = tuple(new.get_stride())
        offset = new.get_offset()
        if (
            tuple(old.get_size()) != size
            or tuple(old.get_stride()) != stride
            or old.get_offset() != offset
        ):
            result_node = self._create_as_strided(old_node, size, stride, offset)

        self._record_allocation(new, result_node)

        # Free the old buffer, if we allocated a new tensor.
        if (
            old.get_name() not in V.graph.get_output_names()
            and line.delete_old
            and result_node is not old_node
        ):
            self._free(old)

    def _generate_multi_output(self, line: WrapperLine) -> None:
        if not isinstance(line, MultiOutputLine):
            raise AssertionError(f"expected MultiOutputLine, got {type(line)}")

        arg_node = self.buffer_to_node[line.arg_name]

        # For non-tuple / non-list outputs, map the
        # output to the same node as the input.
        if len(line.indices) == 0:
            self.buffer_to_node[line.result_name] = arg_node
            return

        # Extract the index for tuple access.
        inds = line.indices[0][1:]
        if len(inds) != 1:
            raise AssertionError(f"Cannot convert {inds} to an index.")
        idx = inds[0]

        node = self.gm.graph.call_function(operator.getitem, args=(arg_node, idx))
        node.name = line.result_name
        self.buffer_to_node[line.result_name] = node

    def _generate_fallback_call(
        self,
        ir_node: ir.ExternKernel,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        fx_node = self.gm.graph.call_function(
            ir_node.op_overload,  # type: ignore[arg-type]
            args=args,
            kwargs=kwargs,
        )
        result_buffer = ir_node.codegen_reference()
        self.buffer_to_node[result_buffer] = fx_node
        # For in-place mutation ops (e.g., scatter_reduce_, index_put_),
        # update the buffer mapping for mutated inputs so downstream
        # references to the mutated buffer see the post-mutation node.
        for mutated_name in ir_node.get_mutation_names():
            self.buffer_to_node[mutated_name] = fx_node

    def _generate_index_put_fallback(self, line: WrapperLine) -> None:
        if not isinstance(line, IndexPutFallbackLine):
            raise AssertionError(f"expected IndexPutFallbackLine, got {type(line)}")
        ir_node = line.node

        def generate_buffer_or_none(
            x: ir.IRNode | Sequence[ir.IRNode] | None,
        ) -> torch.fx.Node | None:
            """
            Handles None before calling _generate_buffer.
            """
            if x is None:
                return None

            if not isinstance(x, ir.IRNode):
                raise AssertionError(f"expected ir.IRNode, got {type(x)}")
            return self._generate_buffer(x)

        (x, values) = [generate_buffer_or_none(t) for t in ir_node.inputs[:2]]
        indices = tuple(generate_buffer_or_none(t) for t in line.indices)
        accumulate = ir_node.constant_args[0]
        args = (x, indices, values, accumulate)
        self._generate_fallback_call(ir_node, args)

    def _generate_scatter_fallback(self, line: WrapperLine) -> None:
        if not isinstance(line, ScatterFallbackLine):
            raise AssertionError(f"expected ScatterFallbackLine, got {type(line)}")
        ir_node = line.node
        if not ir.is_node_sequence(ir_node.inputs):
            raise AssertionError("ir_node.inputs is not a node sequence")
        (x, index, src) = [self._generate_buffer(t) for t in ir_node.inputs] + (
            [] if ir_node.src_is_tensor else [ir_node.constant_args[1]]
        )
        args = (x, ir_node.constant_args[0], index, src)
        kwargs = {}
        if reduce := ir_node.kwargs.get("reduce"):
            kwargs["reduce"] = reduce
        # Only pass kwargs that the op's schema actually accepts, since
        # ScatterFallback stores both reduce and include_self for all
        # scatter variants, but not all ops support them (e.g.,
        # scatter_.value has no kwargs, scatter_reduce_.two has both).
        if not isinstance(ir_node.op_overload, torch._ops.OpOverload):
            raise AssertionError(
                f"expected torch._ops.OpOverload, got {type(ir_node.op_overload)}"
            )
        schema_arg_names = OrderedSet(
            [a.name for a in ir_node.op_overload._schema.arguments]
        )
        kwargs = {k: v for k, v in ir_node.kwargs.items() if k in schema_arg_names}

        self._generate_fallback_call(ir_node, args, kwargs)

    def _generate_null(self, line: WrapperLine) -> None:
        if not isinstance(line, NullLine):
            raise AssertionError(f"expected NullLine, got {type(line)}")
        # Does nothing.

    def _generate_comm_buffer_allocate(self, line: WrapperLine) -> None:
        if not (isinstance(line, AllocateLine) and line.comm_buffer):
            raise AssertionError("expected AllocateLine with comm_buffer set")
        raise NotImplementedError("Comm buffer allocation is not yet supported")

    def _generate_comm_buffer_free(self, line: WrapperLine) -> None:
        if not (isinstance(line, FreeIfNotReusedLine) and line.comm_buffer):
            raise AssertionError("expected FreeIfNotReusedLine with comm_buffer set")
        self._free(line.node)

    def _generate_triton_call(self, line: WrapperLine) -> None:
        if not isinstance(line, KernelCallLine):
            raise AssertionError(f"expected KernelCallLine, got {type(line)}")

        # Collect all kwargs, including autotuned block sizes.
        call_args = self._lookup_args(line.call_args)
        kernel = self.kernels[line.kernel_name]
        tuner = kernel.tuner

        def tune_kernel(tuner: CachingAutotuner, call_args: Sequence[Any]) -> None:
            from triton.runtime import driver

            log.info("Autotuning Triton kernel %s at compile time.", kernel_name)

            device = driver.active.get_current_device()

            stream = driver.active.get_current_stream(device)

            def node_to_tuning_arg(arg: Any) -> Any:
                """
                Create real tensors for autotuning arguments, substituting size hints
                for dynamic shapes.
                """

                def to_size_hint_sympy_int(arg: sympy.Expr | int) -> int:
                    return V.graph.sizevars.optimization_hint(arg)

                def to_size_hint_list(arg: list[torch.SymInt | int]) -> list[int]:
                    args_sympy = [
                        x.node.expr if isinstance(x, torch.SymInt) else x for x in arg
                    ]
                    return pytree.tree_map(to_size_hint_sympy_int, args_sympy)

                if not isinstance(arg, torch.fx.Node):
                    return to_size_hint_sympy_int(arg)

                fake = arg.meta["val"]
                return torch.empty_strided(
                    to_size_hint_list(fake.shape),
                    to_size_hint_list(fake.stride()),
                    dtype=fake.dtype,
                    device=device,
                ).zero_()

            # call args can be fx nodes or sympy expressions or integers!
            arg_values = [node_to_tuning_arg(arg) for arg in call_args]
            tuner.run(*arg_values, stream=stream)

        # Optionally autotune the kernels.
        # The FX backend currently only supports compile-time tuning.
        kernel_name = tuner.fn.__name__
        if config.triton.autotune_at_compile_time:
            # Skip compile-time autotuning if any unbacked symbol lacks a user-provided
            # optimization hint — autotuning with the generic fallback would
            # produce meaningless results.
            hinted = V.graph.sizevars.all_unbacked_explicitly_hinted
            can_tune = True
            for arg in call_args:
                if isinstance(arg, torch.fx.Node):
                    fake = arg.meta["val"]
                    if not hinted(list(fake.shape) + list(fake.stride())):
                        can_tune = False
                        break
                elif not hinted(arg):
                    can_tune = False
                    break
            if can_tune:
                tune_kernel(tuner, call_args)
            else:
                log.info(
                    "Detected unhinted unbacked symints. Skipping compile-time autotuning for kernel %s.",
                    kernel_name,
                )
        else:
            log.info(
                "Skipping autotuning for kernel %s. Set config.triton.autotune_at_compile_time = True to enable.",
                kernel_name,
            )

        triton_meta = tuner.triton_meta
        signature = triton_meta["signature"]

        def add_constants_to_call_args(
            call_args: Sequence[Any], cfg: Config
        ) -> tuple[Any, ...]:
            """
            Add constant kwargs to the arg list.
            """
            # Add args from the proper Triton signature.
            # Exclude constants and config kwargs, as those are tracked separately.
            new_call_args = []
            constants = triton_meta["constants"]
            call_kwargs = {
                key: val
                for key, val in zip(signature, call_args)
                # pyrefly: ignore [missing-attribute]
                if key not in constants and key not in cfg.kwargs
            }

            # Add constants stored as Triton metadata, in signature order.
            call_kwargs |= constants
            new_call_args = [
                call_kwargs[key]
                for key in signature
                # pyrefly: ignore [missing-attribute]
                if key not in cfg.kwargs
            ]

            # Add Inductor's extra launcher args to the end.
            if extra_launcher_args := tuner.inductor_meta.get("extra_launcher_args"):
                new_call_args.extend(
                    call_args[len(call_args) - len(extra_launcher_args) :]
                )

            return tuple(new_call_args)

        kernel_config = tuner.compile_results[0].config
        extra_options = getattr(kernel_config, "extra_options", None)
        call_args = add_constants_to_call_args(call_args, kernel_config)
        call_args, grid = tuner._interpret_args_grid(call_args, kernel_config)
        call_kwargs = dict(zip(signature, call_args))
        # pyrefly: ignore [missing-attribute]
        if any(kwarg in kernel_config.kwargs for kwarg in call_kwargs):
            raise AssertionError(f"kwargs overlap config: {call_kwargs}")
        # pyrefly: ignore [missing-attribute]
        call_kwargs.update(kernel_config.kwargs)

        # Replace sympy.floor with FloorDiv, to make the expression traceable.
        grid = [replace_floor_div(x) if isinstance(x, sympy.Expr) else x for x in grid]
        wrapper_grid = [tuple(self._generate_sym_nodes(grid))]
        call_kwargs = {
            name: self._generate_sym_node(val) for name, val in call_kwargs.items()
        }
        backend_options = triton_meta.get("backend_options", {})
        if backend_options:
            # FXIR executes Triton kernels through the HOP, not the already
            # materialized CachingAutotuner launcher. Preserve backend option
            # values in the HOP payload so a direct FXIR run or later Inductor
            # re-lowering re-enters Triton JIT with the same compiler options
            # that were used during FXIR precompile/autotune.
            for name, value in backend_options.items():
                # If the backend option is also a kernel parameter, call_kwargs
                # already contains the launch value reconstructed from the Triton
                # signature/config. Keep that value and only add backend options
                # that are not otherwise represented in the HOP payload.
                call_kwargs.setdefault(name, value)

        # Store non-graphable kwargs in the side table.
        (
            call_kwargs,
            constant_args_idx,
        ) = tracing_triton_hopifier_singleton.store_non_graphable_args(call_kwargs)

        hop_kwargs: dict[str, Any] = {
            "kernel_idx": kernel.wrapped.kernel_idx,
            "constant_args_idx": constant_args_idx,
            "grid": wrapper_grid,
            "tma_descriptor_metadata": {},
            "kwargs": call_kwargs,
        }
        if backend_options:
            # Keep the FXIR HOP call shape unchanged for normal Inductor
            # kernels. Some downstream HOP py_impls have fixed keyword-only
            # signatures and only need launch_kwargs when there are real
            # Triton backend options to preserve.
            hop_kwargs["launch_kwargs"] = tuple(backend_options)

        triton_node = self.gm.graph.call_function(
            triton_kernel_wrapper_mutation,
            kwargs=hop_kwargs,
        )
        if extra_options:
            triton_node.meta["extra_options"] = extra_options

    def _generate_extern_kernel_alloc(self, line: WrapperLine) -> None:
        if not isinstance(line, ExternKernelAllocLine):
            raise AssertionError(f"expected ExternKernelAllocLine, got {type(line)}")
        node = line.node
        self._generate_extern_kernel_common(node, node)

    def _generate_extern_kernel_out(
        self,
        line: WrapperLine,
    ) -> None:
        if not isinstance(line, ExternKernelOutLine):
            raise AssertionError(f"expected ExternKernelOutLine, got {type(line)}")
        node = line.node
        out_node = node.output_view if node.output_view else node
        self._generate_extern_kernel_common(node, out_node)

    def _generate_extern_kernel_common(
        self, kernel: ir.ExternKernel, out_ir_node: ir.IRNode
    ) -> None:
        """
        Generates FX IR from either ExternKernelAlloc or ExternKernelOut.
        """

        # Get FX nodes corresponding to the call args.
        if not ir.is_node_sequence(kernel.inputs):
            raise AssertionError("kernel.inputs is not a node sequence")
        tensor_nodes = tuple(self._generate_buffer(arg) for arg in kernel.inputs)
        if hasattr(kernel, "unflatten_args"):
            args, _ = kernel.unflatten_args(tensor_nodes, kernel.constant_args)
        else:
            args = tensor_nodes + tuple(kernel.constant_args)

        # Get the result buffer.
        # Some kernels write to a pre-existing output tensor via the "out" kwarg.
        # Materialize any IR nodes in kwargs to FX nodes (e.g., TensorBox -> Tensor).
        kwargs = {
            k: self._generate_buffer(v) if isinstance(v, ir.IRNode) else v
            for k, v in kernel.kwargs.items()
        }

        result_buffer: str | None = None
        if isinstance(kernel, ir.ExternKernelOut):
            kwargs["out"] = self.buffer_to_node[out_ir_node.codegen_reference()]
        elif isinstance(kernel.layout, (ir.Layout, ir.MultiOutputLayout)):
            result_buffer = kernel.get_name()
        elif isinstance(kernel.layout, ir.NoneLayout):
            pass
        else:
            raise NotImplementedError(f"Unrecognized output layout: {kernel.layout}")

        fx_node = self.gm.graph.call_function(
            kernel.op_overload,  # type: ignore[arg-type]
            args=args,
            kwargs=kwargs,
        )

        # Assign the result to the given name.
        if result_buffer:
            if "out" in kwargs:
                raise AssertionError(
                    f"Extern kernel '{kernel}' has both result and out kwarg. Expected only one."
                )
            fx_node.name = result_buffer
            self.buffer_to_node[result_buffer] = fx_node

    def _generate_kernel_call(self, line: WrapperLine) -> None:
        if not isinstance(line, KernelCallLine):
            raise AssertionError(f"expected KernelCallLine, got {type(line)}")
        if not line.triton:
            raise NotImplementedError("FX conversion only supports Triton kernels.")

        self._generate_triton_call(line)

    def _generate_kernel_definition(self, line: WrapperLine) -> None:
        if not isinstance(line, KernelDefinitionLine):
            raise AssertionError(f"expected KernelDefinitionLine, got {type(line)}")

        # Generate code for the kernel.
        kernel_code = PythonWrapperCodegen._format_kernel_definition(
            line.kernel_name, line.kernel_body, metadata=line.metadata
        )

        # Import the module and store the JIT kernel.
        tuner = self._import_kernel(kernel_code, line.kernel_name)
        wrapped = wrap_triton(tuner.fn)
        self.kernels[line.kernel_name] = TritonKernel(tuner, wrapped)

    def _generate_symbolic_call_arg(self, line: WrapperLine) -> None:
        if not isinstance(line, SymbolicCallArgLine):
            raise AssertionError(f"expected SymbolicCallArgLine, got {type(line)}")
        # Store the arg: expr mapping for later use.
        arg = line.arg

        inner_expr_proxy = self._sympy_interp(arg.inner_expr)
        self.expr_to_proxy[arg.inner] = inner_expr_proxy

    def _generate_unbacked_symbol_defs(self, line: WrapperLine) -> None:
        if not isinstance(line, UnbackedSymbolDefsLine):
            raise AssertionError(f"expected UnbackedSymbolDefsLine, got {type(line)}")
        graph = self.gm.graph

        def convert_key(node: torch.fx.Node, path: pytree.KeyPath) -> torch.fx.Node:
            """
            Generate FX IR for each key entry.
            """
            # Base case.
            if len(path) == 0:
                return node

            # Process the first entry and recurse.
            entry = path[0]
            if isinstance(entry, CallMethodKey):
                target = {
                    "size": aten.sym_size.int,
                    "stride": aten.sym_stride.int,
                    "storage_offset": aten.sym_storage_offset,
                }[entry.name]
                if not callable(target):
                    raise AssertionError(f"target is not callable: {target}")
                node = graph.call_function(
                    target,
                    args=(
                        (node, path[1].idx)
                        if len(path) > 1 and isinstance(path[1], pytree.SequenceKey)
                        else (node,)
                    ),
                )
                return convert_key(node, path[1 + len(node.args) :])
            elif isinstance(entry, pytree.SequenceKey):
                node = graph.call_function(operator.getitem, args=(node, entry.idx))
                return convert_key(node, path[1:])
            elif isinstance(entry, DivideByKey):
                node = graph.call_function(
                    operator.floordiv, args=(node, entry.divisor)
                )
                return convert_key(node, path[1:])
            else:
                raise NotImplementedError(f"Unrecognized entry type: {type(entry)}")

        unbacked_bindings = line.unbacked_bindings
        if unbacked_bindings is None:
            raise AssertionError("line.unbacked_bindings must not be None")
        # Kernels with no unbacked symbols aren't recorded in buffer_to_node, so
        # return before the output-buffer lookup to avoid a KeyError. Mirrors
        # the non-FX wrapper's early return.
        if not unbacked_bindings:
            return
        root_node = self.buffer_to_node[line.output_name]
        for s, keypath in unbacked_bindings.items():
            # Check if we already generated this symbol.
            if s.name in self.buffer_to_node:
                continue

            node = convert_key(root_node, keypath)
            out_buffer = SymbolBuffer(s)
            self._record_allocation(out_buffer, node)
            self._generate_size_proxy(node, s)
<|endoftext|># mypy: allow-untyped-defs
import functools
import logging
import math
import os
import pathlib
from typing import Any

from torch._inductor.ir import MultiTemplateBuffer
from torch._inductor.metrics import get_metric_table, is_metric_table_enabled
from torch.utils._ordered_set import OrderedSet

from .. import config
from ..codecache import code_hash, CodeCacheFuture, get_path, write_atomic
from ..runtime.benchmarking import benchmarker
from ..utils import cache_on_self, IndentedBuffer
from ..virtualized import V
from .common import TensorArg, WorkspaceArg


log = logging.getLogger(__name__)


class MultiKernelState:
    """
    Maintain state of multi-kernel compilation so we don't define duplicated
    multi-kernel for the same set of sub-kernels.

    V.graph.wrapper_code has a reference to MultiKernelState instance.
    """

    def __init__(self):
        self.subkernel_to_kernel_name = {}
        self.kernel_defs = IndentedBuffer()

    def define_kernel(
        self,
        kernels: list[Any],
        kernel_shape_keys: list[None | tuple[tuple[int, ...], ...]] | None = None,
    ) -> str:
        """
        Previously we name the multi kernel as "multi_kernel_{kernel_names[0]}".
        This has some minor issue.

        E.g. for persistent reduction https://gist.github.com/shunting314/39e7c00ff8bb2055942ed5a3255d61ca ,
        there are 2 flavors of non-persistent reduction:
          https://gist.github.com/shunting314/056d43d35907e87efb883970b35c17d4
        and
          https://gist.github.com/shunting314/02ee753b65c513c54e695626afe682bd

        The only different is cache eviction policy.

        We should name the multi-kernel differently in these 2 cases.

        kernels:
            A list of kernels
        kernel_shape_keys:
            Specified for size-hint multi-kernels.
            Each list element is a shape key, corresponding to the concrete input & output size hints each kernel was tuned for.
        """
        # Prevent circular import
        from ..select_algorithm import TritonTemplateKernel

        kernel_names = tuple(k.kernel_name for k in kernels)
        if kernel_names in self.subkernel_to_kernel_name:
            return self.subkernel_to_kernel_name[kernel_names]

        # name the multi kernel based on the first kernel
        multi_kernel_name = f"multi_kernel_{len(self.subkernel_to_kernel_name)}"
        self.subkernel_to_kernel_name[kernel_names] = multi_kernel_name

        if V.graph.cpp_wrapper and not config.triton.autotune_at_compile_time:
            # we should not generate any python code for multi-kernel during
            # the second pass of cpp-wrapper.
            return multi_kernel_name

        arg_index: dict[int, list[slice]] = {}
        _, call_args, _, arg_types = kernels[0].args.python_argdefs()
        if isinstance(kernels[0], TritonTemplateKernel) and isinstance(
            kernels[0].output_node, MultiTemplateBuffer
        ):
            for i, kernel in enumerate(kernels):
                additional_call_args, _ = kernel.additional_call_args_and_types()
                if i not in arg_index:
                    arg_index[i] = []
                arg_index[i].append(slice(0, len(call_args)))
                arg_index[i].append(
                    slice(
                        len(call_args) + i * len(additional_call_args),
                        len(call_args) + (i + 1) * len(additional_call_args),
                    )
                )
        else:
            kernels[0].add_numel_to_call_args(multi_kernel_name, call_args, arg_types)
            for i in range(len(kernels)):
                arg_index[i] = [slice(0, len(call_args))]

        keyed_by_sizes = kernel_shape_keys is not None
        buf = self.kernel_defs
        buf.writeline("")
        buf.writeline("arg_index = {")
        for key, slice_list in arg_index.items():
            slice_reprs = ", ".join(repr(s) for s in slice_list)
            buf.writeline(f"    {key}: [{slice_reprs}],")
        buf.writeline("}")

        if not keyed_by_sizes:  # no size hint keys, just call with list of kernels
            buf.writeline(
                f"{multi_kernel_name} = async_compile.multi_kernel({multi_kernel_name!r}, ["
            )
            with buf.indent():
                for name in kernel_names:
                    buf.writeline(f"{name},")
            buf.writeline("], arg_index=arg_index)")
        else:  # call with dict[size hint key, kernel]
            if not isinstance(kernels[0], TritonTemplateKernel):
                raise AssertionError("expected kernels[0] to be a TritonTemplateKernel")
            if not isinstance(kernel_shape_keys, list):
                raise AssertionError("expected kernel_shape_keys to be a list")
            if len(kernels) != len(kernel_shape_keys):
                raise AssertionError(
                    f"expected len(kernels) == len(kernel_shape_keys), "
                    f"got {len(kernels)} != {len(kernel_shape_keys)}"
                )
            buf.writeline(
                f"{multi_kernel_name} = async_compile.size_hint_multi_kernel({multi_kernel_name!r}, {{"
            )
            with buf.indent():
                for shape_key, name in zip(kernel_shape_keys, kernel_names):
                    buf.writeline(f"{shape_key}: {name},")
            buf.writeline("}, arg_index=arg_index)")

        if config.triton.autotune_at_compile_time:
            V.graph.wrapper_code.src_to_kernel["\n".join(kernel_names)] = (
                multi_kernel_name
            )

        return multi_kernel_name


class MultiKernel:
    """
    This class maintains the compile time state for multi kernels.

    Assume we do codegen for a MultiKernel encapsulating kernel1 and kernel2.
    The generated definition for the multi-kernel will looks like:
    ```
    multi_kernel_kernel1 = MultiKernelCall(
        [kernel1, kernel2], multi_kernel_definition_code
    )
    ```

    Here is a concrete example: https://gist.github.com/shunting314/d9f3fb6bc6cee3dbae005825ca196d39
    """

    def __init__(self, kernels):
        if len(kernels) < 2:
            raise AssertionError(f"expected at least 2 kernels, got {len(kernels)}")

        self.kernels = kernels
