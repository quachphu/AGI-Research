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
                # Need to wrap scalar into tensor as the main function returns a vector of tensors
                output_tensor = self.codegen_scalar_to_tensor(output)
                self.wrapper_call.writeline(
                    f"output_handles[{idx}] = {output_tensor}.release();"
                )
                continue

            output_is_tensor_handle_expr = (
                f"std::is_same_v<std::decay_t<decltype({output})>,"
                "RAIIAtenTensorHandle> || "
                f"std::is_same_v<std::decay_t<decltype({output})>,"
                "AtenTensorHandle> || "
                f"std::is_same_v<std::decay_t<decltype({output})>,"
                "ConstantHandle>"
            )
            self.wrapper_call.writeline(
                f"if constexpr ({output_is_tensor_handle_expr}) {{"
            )
            with self.wrapper_call.indent():
                if arr_iface:
                    cached_output_name = f"cached_output_{next(self.cached_output_id)}"
                    self.wrapper_call.writeline(
                        f"thread_local RAIIAtenTensorHandle {cached_output_name};"
                    )
                    if is_constant_buffer:
                        # NOTE(return_constant): In some rare cases where we return
                        # a constant, we have to return a copy of this constant,
                        # because (1) constants are not owned by the Model instance
                        # (2) constants remain the same cross inference runs,
                        # assuming they are not updated at runtime Basically, we
                        # cannot release or transfer the ownership of any original
                        # constant to the user.
                        self.wrapper_call.writeline(
                            f"AtenTensorHandle {cached_output_name}_tmp;"
                        )
                        self.wrapper_call.writeline(
                            "AOTI_TORCH_ERROR_CODE_CHECK("
                            f"aoti_torch_clone({output}, &{cached_output_name}_tmp));"
                        )
                        self.wrapper_call.writeline(
                            f"{cached_output_name} = {cached_output_name}_tmp;"
                        )
                    else:
                        self.wrapper_call.writeline(
                            f"{cached_output_name} = {output}.release();"
                        )
                    self.wrapper_call.writeline(
                        f"convert_handle_to_arrayref_tensor({cached_output_name}, "
                        f"std::get<{idx}>(output_arrayref_tensors));"
                    )
                else:
                    if is_constant_buffer:
                        # See NOTE(return_constant) above.
                        self.wrapper_call.writeline(
                            "AOTI_TORCH_ERROR_CODE_CHECK("
                            f"aoti_torch_clone({output}, &output_handles[{idx}]));"
                        )
                    else:
                        if output in output2idx:
                            src_idx = output2idx[output]
                            self.wrapper_call.writeline(
                                f"output_handles[{idx}] = output_handles[{src_idx}];"
                            )
                        else:
                            self.wrapper_call.writeline(
                                f"output_handles[{idx}] = {output}.release();"
                            )
            self.wrapper_call.writeline("} else {")
            with self.wrapper_call.indent():
                use_thread_local_cached_output_tensor(idx, output)
            self.wrapper_call.writeline("}")

            if output not in output2idx:
                output2idx[output] = idx
        if arr_iface:
            self.wrapper_call.writeline("return output_arrayref_tensors;")

    def generate_before_suffix(self, result):
        super().generate_before_suffix(result)
        if self.v2_raw_output_refs is None:
            return

        raw_impl = IndentedBuffer()
        raw_impl.splice(
            """
            void AOTInductorModel::run_impl_minimal_arrayref_interface_v2_raw(
                const AOTInductorArrayRefTensor* c_inputs,
                AOTInductorArrayRefTensor* c_outputs,
                DeviceStreamType stream,
                AOTIProxyExecutorHandle proxy_executor
            ) {
            """
        )
        with raw_impl.indent():
            self._codegen_v2_raw_prelude(raw_impl)
            raw_impl.splice(self.v2_raw_wrapper_body)
            self._codegen_v2_raw_outputs(raw_impl, self.v2_raw_output_refs)
        raw_impl.writeline(
            "} // AOTInductorModel::run_impl_minimal_arrayref_interface_v2_raw"
        )
        result.splice(raw_impl)

    def memory_plan(self):
        from .memory_planning import MemoryPlanner

        self.lines = MemoryPlanner(self).plan(self.lines)
        # TODO: integrate memory planning & stack allocation?
        self.allow_stack_allocation = False

    def memory_plan_reuse(self):
        out_names = V.graph.get_output_names()

        while (
            self.lines
            and isinstance(self.lines[-1], MemoryPlanningLine)
            # TODO: this seems legit, NullLine has no node
            and self.lines[-1].node.name not in out_names  # type: ignore[attr-defined]
        ):
            # these lines will be pointless
            self.lines.pop()

        # codegen allocations in two passes
        planning_states = [MemoryPlanningState()]
        past_planning_states = []
        for i in range(len(self.lines)):
            line = self.lines[i]
            if isinstance(line, MemoryPlanningLine):
                self.lines[i] = line.plan(planning_states[-1])
            elif isinstance(line, EnterSubgraphLine):
                planning_states.append(MemoryPlanningState())
            elif isinstance(line, ExitSubgraphLine):
                past_planning_states.append(planning_states.pop())
        past_planning_states.append(planning_states.pop())
        if len(planning_states) != 0:
            raise AssertionError(
                f"expected planning_states to be empty, got {len(planning_states)}"
            )

        # conservatively use the sum of all allocated buffer sizes
        # in potentially nested scopes as the total allocated size
        total_allocated_buffer_size = sum(
            s.total_allocated_buffer_size for s in past_planning_states
        )

        self.allow_stack_allocation = (
            self.allow_stack_allocation is not False
            and config.aot_inductor.allow_stack_allocation
            and total_allocated_buffer_size <= MAX_STACK_ALLOCATION_SIZE
        )

    def can_stack_allocate_buffer(self, buffer):
        return (
            self.allow_stack_allocation
            and buffer.get_device().type == "cpu"
            and self.can_prove_buffer_has_static_shape(buffer)
            and ir.is_contiguous_strides_for_shape(
                buffer.get_stride(), buffer.get_size()
            )
        )

    def make_buffer_free(self, buffer):
        return (
            ""
            if isinstance(buffer.get_output_spec(), ir.MultiOutputLayout)
            or (V.graph.aot_mode and buffer.get_name() in self.stack_allocated_buffers)
            or (
                config.aot_inductor.use_minimal_arrayref_interface
                and V.graph.aot_mode
                and buffer.get_name() in V.graph.graph_inputs
            )
            else f"{buffer.get_name()}.reset();"
        )

    def make_buffer_allocation(self, buffer):
        return self.make_allocation(
            buffer.get_name(),
            buffer.get_device(),
            buffer.get_dtype(),
            buffer.get_size(),
            buffer.get_stride(),
            buffer if self.can_stack_allocate_buffer(buffer) else None,
            buffer.get_is_pinned(),
        )

    def make_allocation(
        self,
        name,
        device,
        dtype,
        shape,
        stride,
        buffer_if_can_stack_allocate=None,
        is_pinned=False,
    ):
        orig_stride = stride
        device_str = self.codegen_device(device)
        dtype_code = self.codegen_dtype(dtype)
        size = self.codegen_shape_tuple(shape)
        stride = self.codegen_shape_tuple(orig_stride)
        size_array_var = self.codegen_int_array_var(
            size,
            self.wrapper_call.writeline,
            known_statically=self.is_statically_known_list_of_ints(shape),
            graph=self.get_codegened_graph(),
        )
        stride_array_var = self.codegen_int_array_var(
            stride,
            self.wrapper_call.writeline,
            known_statically=self.is_statically_known_list_of_ints(orig_stride),
            graph=self.get_codegened_graph(),
        )
        device_type, device_id = device_str.split(",")
        device_idx = self.codegen_device_idx(device, device_id)
        if buffer_if_can_stack_allocate is not None:
            self.stack_allocated_buffers[name] = buffer_if_can_stack_allocate
            cpp_type = DTYPE_TO_CPP[dtype]
            numel = buffer_if_can_stack_allocate.get_numel()
            # Note: we don't zero storage because empty_strided doesn't zero either.
            self.wrapper_call.writeline(f"{cpp_type} {name}_storage[{numel}];")
            args = [
                f"{name}_storage",
                size_array_var,
                stride_array_var,
                device_type,
                device_idx,
            ]
            return f"ArrayRefTensor<{cpp_type}> {name}({', '.join(args)});"

        args = [
            str(len(shape)),
            size_array_var,
            stride_array_var,
            dtype_code,
            device_type,
            device_idx,
            f"&{name}_handle",
        ]

        self.wrapper_call.writeline(f"AtenTensorHandle {name}_handle;")
        pinned_str = "_pinned" if is_pinned else ""
        self.wrapper_call.writeline(
            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided{pinned_str}({', '.join(args)}));"
        )

        return f"RAIIAtenTensorHandle {name}({name}_handle);"

    def make_buffer_reuse(self, old: BufferLike, new: BufferLike, delete_old: bool):
        if old.get_dtype() != new.get_dtype():
            raise AssertionError(
                f"expected matching dtypes, got {old.get_dtype()} and {new.get_dtype()}"
            )
        old_name = old.get_name()
        new_name = new.get_name()
        del_line = ";"
        if old_name not in V.graph.get_output_names() and delete_old:
            del_line = f"; {self.make_buffer_free(old)}"

        if old.get_size() == new.get_size() and old.get_stride() == new.get_stride():
            if old_name in self.stack_allocated_buffers:
                self.stack_allocated_buffers[new_name] = new
            return self.codegen_exact_buffer_reuse(old_name, new_name, del_line)

        reinterpret_view = self.codegen_reinterpret_view(
            old, new.get_size(), new.get_stride(), 0, self.wrapper_call.writeline
        )
        if reinterpret_view in self.stack_allocated_buffers:
            self.stack_allocated_buffers[new_name] = new
            # The only way to get into this case is via an exact buffer reuse, since all
            # other options result in a new tensor handle.
            return self.codegen_exact_buffer_reuse(old_name, new_name, del_line)
        return f"{self.declare}{new_name} = {reinterpret_view}{del_line}  // reuse"

    def _assert_safe_to_use_borrow_arrayref_tensor_as_tensor(self):
        # Borrowing arguments to shim functions is only safe because we know
        # that the arguments can't be stack-allocated. Otherwise, to be sure
        # we can't return a dangling pointer, we need to either 1) be
        # certain that the shim function cannot return an alias of a
        # borrowed argument, or 2) be certain that the returned Tensor from
        # the shim function cannot escape.
        if not self.is_safe_to_use_borrow_arrayref_tensor_as_tensor():
            raise AssertionError(
                "borrowing arguments to shim functions is unsafe with "
                "stack allocation on! (see comment above this assertion)"
            )

    def is_safe_to_use_borrow_arrayref_tensor_as_tensor(self):
        return not self.allow_stack_allocation and not self.stack_allocated_buffers

    def codegen_subgraph_prefix(self, subgraph, outer_inputs, outer_outputs):
        if len(subgraph.graph.graph_inputs) != len(outer_inputs):
            raise AssertionError(
                f"expected {len(outer_inputs)} subgraph inputs, got "
                f"{len(subgraph.graph.graph_inputs)}"
            )

        for (inner_input, inner_input_val), outer_input in zip(
            subgraph.graph.graph_inputs.items(), outer_inputs
        ):
            if not isinstance(inner_input_val, ir.TensorBox):
                continue

            # Wrap with a generic lambda so if constexpr can discard the
            # ill-formed branch (if constexpr only discards in dependent
            # contexts, i.e. templates / generic lambdas).
            self.writeline(f"AtenTensorHandle {inner_input}_handle;")
            self.writeline(
                f"[&](auto&& _t) {{ "
                f"if constexpr (::torch::aot_inductor::is_arrayref_tensor_type_v"
                f"<std::decay_t<decltype(_t)>>) {{ "
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_assign_tensors_out("
                f"borrow_arrayref_tensor_as_tensor(_t), &{inner_input}_handle)); "
                f"}} else {{ "
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_assign_tensors_out("
                f"_t, &{inner_input}_handle)); "
                f"}} }}({outer_input});"
            )
            self.writeline(f"RAIIAtenTensorHandle {inner_input}({inner_input}_handle);")

    def codegen_while_loop(self, while_loop, stack_output=False):
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
            self.writeline(f"AtenTensorHandle {out_name}_handle;")
            self.writeline(
                "AOTI_TORCH_ERROR_CODE_CHECK("
                f"aoti_torch_assign_tensors_out(borrow_arrayref_tensor_as_tensor({inp}), "
                f"&{out_name}_handle));"
            )
            self.writeline(f"RAIIAtenTensorHandle {out_name}({out_name}_handle);")
            cond_outer_inputs.append(out_name)

        cond_outer_inputs.extend(outer_additional_inputs)

        cond_outer_outputs = [cond_result_name]
        body_outer_inputs = list(cond_outer_inputs)
        body_outer_outputs = body_outer_inputs[: len(outer_carried_inputs)]

        self.writeline("while (1) {")
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

    def generate_c_shim_extern_kernel_call(
        self, kernel: str, args: list[str], device: str, **_
    ) -> None:
        # In the abi_compatible mode, we call fallback aten ops through a C shim layer
        # Setting self.allow_stack_allocation to False because the exchange between
        # ArrayRefTensor and at::Tensor is still fragile.
        self.allow_stack_allocation = False

        wrapped_args = []
        for arg in args:
            # We only really *need* borrow_arrayref_tensor_as_tensor for
            # ArrayRefTensors. The code flowing into here uses `0` for nullptr, which
            # borrow_arrayref_tensor_as_tensor would blindly coerce to int, so just
            # avoid wrapping integers.  Name matching is to find tensor is hacky, but
            # fixing all the ArrayRefTensor issues is not a priority for now.
            if isinstance(arg, str) and arg.startswith(
                ("buf", "arg", "wrap_with_raii_handle_if_needed")
            ):
                self._assert_safe_to_use_borrow_arrayref_tensor_as_tensor()
                arg = f"borrow_arrayref_tensor_as_tensor({arg})"
            wrapped_args.append(arg)

        super().generate_c_shim_extern_kernel_call(
            kernel, wrapped_args, device, debug_args=args
        )

    def generate_scatter_fallback(self, node: ir.ScatterFallback):
        # No stack allocation when there is a fallback op
        self.allow_stack_allocation = False
        super().generate_scatter_fallback(node)

    def _generate_scatter_fallback_args(self, inputs: Sequence[Any]) -> list[str]:
        self._assert_safe_to_use_borrow_arrayref_tensor_as_tensor()
        return [
            f"borrow_arrayref_tensor_as_tensor({x})" if isinstance(x, str) else str(x)
            for x in inputs
        ]

    def generate_index_put_fallback(self, node: ir.IndexPutFallback) -> None:
        # No stack allocation when there is a fallback op
        self.allow_stack_allocation = False
        super().generate_index_put_fallback(node)

    def _generate_index_put_fallback(self, kernel, x, indices, values, accumulate):
        self._assert_safe_to_use_borrow_arrayref_tensor_as_tensor()
        # TODO: update aoti_torch_index_put_out in ir.py to use autogen out version
        # See the comment in codegen_reinterpret_view about why having something like
        # RAIIAtenTensorHandle(tmp_tensor_handle_2) in a tmp array can cause the corresponding
        # tensor prematurely deallocated, thus the temporary array trick here.
        indices_str = self._generate_temporary_array_pointer(
            "AtenTensorHandle",
            [f"borrow_arrayref_tensor_as_tensor({i})" for i in indices],
        )
        args = [
            f"borrow_arrayref_tensor_as_tensor({x})",
            indices_str,
            str(len(indices)),
            f"borrow_arrayref_tensor_as_tensor({values})",
            accumulate,
        ]
        args.insert(
            0, f"borrow_arrayref_tensor_as_tensor({x})"
        )  # set x as the output tensor, this fallback mutates x.
        # Wrap in AOTI_TORCH_ERROR_CODE_CHECK so a shim failure surfaces instead
        # of being silently swallowed.
        self.writeline(f"AOTI_TORCH_ERROR_CODE_CHECK({kernel}({', '.join(args)}));")

    def generate_fallback_kernel_with_runtime_lookup(
        self,
        buf_name: str,
        python_kernel_name: str,
        get_args: Callable[[], Sequence[str]],
        op_overload: torch._ops.OpOverload | torch._ops.HigherOrderOperator,
        raw_args: Sequence[Any],
        outputs: Sequence[ir.Buffer],
    ) -> None:
        # No stack allocation when there is a fallback op
        self.allow_stack_allocation = False
        super().generate_fallback_kernel_with_runtime_lookup(
            buf_name, python_kernel_name, get_args, op_overload, raw_args, outputs
        )

    def codegen_runtime_lookup_tensor_call_args(
        self, tensor_call_args: Sequence[str]
    ) -> Sequence[str]:
        self._assert_safe_to_use_borrow_arrayref_tensor_as_tensor()
        return [f"borrow_arrayref_tensor_as_tensor({arg})" for arg in tensor_call_args]

    def codegen_device_copy(self, src, dst, non_blocking: bool | str):
        # aoti_torch_tensor_copy_ takes AtenTensorHandle as input,
        # while stack-allocation results in ArrayRefTensor
        # so disable stack allocation here
        self.allow_stack_allocation = False
        self.writeline(
            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_copy_(expensive_copy_to_tensor_if_needed({dst}), {src}, {non_blocking}));"
        )

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
        dim = str(len(size))

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

        def create_new_tensor_handle() -> tuple[str, list[str]]:
            # Calling reset() on ArrayRefTensor does nothing, since the array is
            # const-allocated on the stack.  Thus, it's safe to return a reference to
            # the original array.
            if (name := data.get_name()) in self.stack_allocated_buffers:
                return name, []

            tmp_AtenTensorHandle = f"tmp_{name}_{next(self.tmp_tensor_id)}"
            tmp_call_strs = [
                f"AtenTensorHandle {tmp_AtenTensorHandle};",
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_new_tensor_handle({data.get_name()}, &{tmp_AtenTensorHandle}));",
            ]
            return f"RAIIAtenTensorHandle({tmp_AtenTensorHandle})", tmp_call_strs

        if (
            size == data.layout.size
            and stride == data.layout.stride
            and offset == data.layout.offset
            and (dtype is None or dtype == data.dtype)
        ):
            final_tensor_str, call_strs = create_new_tensor_handle()
            for line in call_strs:
                writeline(line)
            return final_tensor_str

        return super().codegen_reinterpret_view(
            data, size, stride, offset, writeline, dtype
        )

    def val_to_arg_str(self, val, type_=None) -> str:
        if (
            val is not None
            and isinstance(type_, torch.OptionalType)
            and isinstance(type_.getElementType(), torch.TensorType)
        ):
            # Handle optional tensors as a special case, as in the parent class.
            base_handle = self.val_to_arg_str(val, torch.TensorType)
            if config.aot_inductor.use_minimal_arrayref_interface:
                if self.is_safe_to_use_borrow_arrayref_tensor_as_tensor():
                    base_handle = f"borrow_arrayref_tensor_as_tensor({base_handle})"
                else:
                    base_handle = f"copy_arrayref_tensor_to_tensor({base_handle})"
            return f"&temporary_reference({base_handle}.get())"

        return super().val_to_arg_str(val, type_)

    def codegen_tensor_item(
        self, dtype: torch.dtype, tensor: str, scalar: str, indented_buffer=None
    ):
        dtype_str = str(dtype).split(".")[-1]
        writer = indented_buffer or self

        if dtype == torch.float16 or dtype == torch.bfloat16:
            scalar_tmp = f"{scalar}_tmp"
            writer.writeline(f"{DTYPE_TO_CPP[dtype]} {scalar_tmp};")

            # We know that item_ doesn't alias the input, so borrowing should be safe.
            tensor = f"borrow_arrayref_tensor_as_tensor({tensor})"

            writer.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_item_{dtype_str}({tensor}, &{scalar_tmp}));"
            )
            writer.writeline(f"float {scalar} = float({scalar_tmp});")
        else:
            writer.writeline(f"{DTYPE_TO_CPP[dtype]} {scalar};")

            # We know that item_ doesn't alias the input, so borrowing should be safe.
            tensor = f"borrow_arrayref_tensor_as_tensor({tensor})"

            writer.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_item_{dtype_str}({tensor}, &{scalar}));"
            )
<|endoftext|># mypy: allow-untyped-defs
"""Metal shader template for flex attention on MPS"""

import itertools
import operator
from collections import namedtuple
from collections.abc import Sequence
from typing import Any

import torch
from torch.utils._ordered_set import OrderedSet

from .. import ir


METAL_DTYPE_MAP = {
    torch.float32: "float",
    torch.float16: "half",
    torch.bfloat16: "bfloat",
}

# Captured score_mod/mask_mod buffers may hold more types than q/k/v; elements
# are always loaded and cast to float for use in the (float-typed) op table.
# Integer captures (e.g. document ids) are therefore exact only below 2**24,
# which covers realistic id/index ranges.
CAPTURE_DTYPE_MAP = {
    torch.float32: "float",
    torch.float16: "half",
    torch.bfloat16: "bfloat",
    torch.int64: "long",
    torch.int32: "int",
    torch.int16: "short",
    torch.int8: "char",
    torch.uint8: "uchar",
    torch.bool: "bool",
}

# log2(e) for converting flex attention stats back
RCP_LN2 = "1.44269504088896340736f"

# A captured buffer bound as a Metal kernel argument, with its static shape.
_CapturedBuf = namedtuple("_CapturedBuf", ["name", "sizes", "strides", "dtype"])
# A captured SymInt (dynamic-shape closure) packed into the scalar _params
# buffer; `name` is the unpacked Metal variable holding its runtime value.
_CapturedScalar = namedtuple("_CapturedScalar", ["name"])

# Python operator builtins appear in score_mod/mask_mod when a closure does
# arithmetic/comparison on captured SymInts (e.g. q_idx < (s0 // 32) * 32).
# Arithmetic preserves integer typing when both operands are integral, so a
# downstream floordiv/index stays exact; otherwise it falls back to float.
_INT_PRESERVING_OPS = {operator.add: "+", operator.sub: "-", operator.mul: "*"}
_CMP_OPS = {
    operator.lt: "<",
    operator.le: "<=",
    operator.gt: ">",
    operator.ge: ">=",
    operator.eq: "==",
    operator.ne: "!=",
}

# Fixed score/mask placeholders bound to integer index variables in the kernel;
# the score placeholder (score_val) is the only float-typed fixed input.
_INT_INDEX_VARS = OrderedSet(["b_idx", "h_idx", "m_idx", "n_idx"])


# FX aten target -> (output C type, Metal format string with positional {} placeholders).
aten = torch.ops.aten
prims = torch.ops.prims
_OP_TABLE: dict[Any, tuple[str, str]] = {
    # Arithmetic
    aten.add.Tensor: ("float", "{} + {}"),
    aten.add.Scalar: ("float", "{} + {}"),
    aten.sub.Tensor: ("float", "{} - {}"),
    aten.sub.Scalar: ("float", "{} - {}"),
    aten.mul.Tensor: ("float", "{} * {}"),
    aten.mul.Scalar: ("float", "{} * {}"),
    # Cast to float so Python truediv semantics hold even when both operands are ints
    aten.div.Tensor: ("float", "static_cast<float>({}) / static_cast<float>({})"),
    aten.div.Scalar: ("float", "static_cast<float>({}) / static_cast<float>({})"),
    aten.neg.default: ("float", "-{}"),
    aten.abs.default: ("float", "metal::abs({})"),
    # Comparisons
    aten.gt.Tensor: ("bool", "{} > {}"),
    aten.gt.Scalar: ("bool", "{} > {}"),
    aten.ge.Tensor: ("bool", "{} >= {}"),
    aten.ge.Scalar: ("bool", "{} >= {}"),
    aten.lt.Tensor: ("bool", "{} < {}"),
    aten.lt.Scalar: ("bool", "{} < {}"),
    aten.le.Tensor: ("bool", "{} <= {}"),
    aten.le.Scalar: ("bool", "{} <= {}"),
    aten.eq.Tensor: ("bool", "{} == {}"),
    aten.eq.Scalar: ("bool", "{} == {}"),
    aten.ne.Tensor: ("bool", "{} != {}"),
    aten.ne.Scalar: ("bool", "{} != {}"),
    aten.logical_and.default: ("bool", "{} && {}"),
    aten.logical_or.default: ("bool", "{} || {}"),
    aten.logical_not.default: ("bool", "!{}"),
    # Math
    aten.exp.default: ("float", "metal::precise::exp({})"),
    aten.log.default: ("float", "metal::precise::log({})"),
    aten.sqrt.default: ("float", "metal::precise::sqrt({})"),
    aten.rsqrt.default: ("float", "metal::precise::rsqrt({})"),
    aten.tanh.default: ("float", "metal::precise::tanh({})"),
    aten.sin.default: ("float", "metal::precise::sin({})"),
    aten.cos.default: ("float", "metal::precise::cos({})"),
    aten.exp2.default: ("float", "metal::precise::exp2({})"),
    # remainder
    aten.remainder.Scalar: ("float", "c10::metal::remainder({}, (float){})"),
    aten.remainder.Tensor: ("float", "c10::metal::remainder({}, {})"),
    aten.maximum.default: ("float", "metal::max({}, {})"),
    aten.max.other: ("float", "metal::max({}, {})"),
    aten.minimum.default: ("float", "metal::min({}, {})"),
    aten.min.other: ("float", "metal::min({}, {})"),
    # Bitwise - int output
    aten.bitwise_and.Tensor: ("int", "(int){} & (int){}"),
    aten.bitwise_and.Scalar: ("int", "(int){} & (int){}"),
    aten.bitwise_or.Tensor: ("int", "(int){} | (int){}"),
    aten.bitwise_or.Scalar: ("int", "(int){} | (int){}"),
    aten.where.self: ("float", "{} ? {} : {}"),
    aten._to_copy.default: ("float", "static_cast<float>({})"),
    aten.to.dtype: ("float", "static_cast<float>({})"),
    prims.convert_element_type.default: ("float", "static_cast<float>({})"),
    aten.scalar_tensor.default: ("float", "{}"),
}
del aten, prims


def _fx_graph_to_metal(
    graph_module: torch.fx.GraphModule,
    fixed_inputs: dict[str, str],
    captured: dict[str, Any],
    output_var: str,
    var_prefix: str,
    int_ctype: str = "int",
) -> str:
    """Compile an FX GraphModule to inline Metal code.

    `fixed_inputs` maps scalar placeholder names (score/b/h/q/kv) to Metal
    variables. `captured` maps captured placeholder names to a `_CapturedBuf`
    (a tensor, indexed by `aten.index.Tensor` down to a scalar load) or a
    `_CapturedScalar` (a SymInt bound directly to a Metal variable). The output
    is assigned to `output_var`; `var_prefix` namespaces fresh temps.
    """
    var_map: dict[str, str] = {}
    views: dict[str, tuple] = {}
    # Names of temps/placeholders known to hold an integer value, so operator
    # arithmetic on captured SymInts stays in integer math (exact, proper floor).
    int_vars: OrderedSet[str] = OrderedSet(
        ph for ph, mv in fixed_inputs.items() if mv in _INT_INDEX_VARS
    )
    code_lines: list[str] = []
    tmp_counter = itertools.count()

    def _is_int(node) -> bool:
        if isinstance(node, bool):
            return False
        if isinstance(node, int):
            return True
        return isinstance(node, torch.fx.Node) and node.name in int_vars

    # Only emit nodes the output depends on; this drops dead constant captures
    output_node = next(n for n in graph_module.graph.nodes if n.op == "output")
    needed: OrderedSet[str] = OrderedSet()
    stack = list(output_node.all_input_nodes)
    while stack:
        nd = stack.pop()
        if nd.name not in needed:
            needed.add(nd.name)
            stack.extend(nd.all_input_nodes)

    def _tmp() -> str:
        return f"{var_prefix}{next(tmp_counter)}"

    def _val(node) -> str:
        if isinstance(node, torch.fx.Node):
            if node.name in views:
                raise NotImplementedError(
                    "flex_attention on MPS: captured buffer used without being "
                    "fully indexed to a scalar"
                )
            return var_map[node.name]
        if isinstance(node, bool):
            return "true" if node else "false"
        if isinstance(node, float):
            if node == float("inf"):
                return "INFINITY"
            if node == float("-inf"):
                return "(-INFINITY)"
            if node != node:  # NaN
                return "NAN"
            return f"{node}f"
        if isinstance(node, int):
            return str(node)
        return str(node)

    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            if node.name in fixed_inputs:
                var_map[node.name] = fixed_inputs[node.name]
            elif node.name in captured:
                cap = captured[node.name]
                if isinstance(cap, _CapturedScalar):
                    var_map[node.name] = cap.name
                    int_vars.add(node.name)
                elif len(cap.sizes) == 0:
                    var_map[node.name] = f"(float){cap.name}[0]"
                else:
                    views[node.name] = (
                        cap.name,
                        "0",
                        list(cap.sizes),
                        list(cap.strides),
                    )
            else:
                var_map[node.name] = f"/* unknown placeholder {node.name} */"
            continue

        if node.op == "output":
            out_node = node.args[0]
            if isinstance(out_node, (tuple, list)):
                out_node = out_node[0]
            code_lines.append(f"{output_var} = {_val(out_node)};")
            continue

        if node.name not in needed:
            continue

        if node.op == "get_attr":
            raise NotImplementedError(
                "flex_attention on MPS does not support constant tensor captures "
                "in score_mod/mask_mod yet"
            )

        if node.op != "call_function":
            continue

        target = node.target
        args = node.args

        # Index a captured buffer: accumulate a flat offset over the leading
        # dims, peeling them off; emit a scalar load when fully indexed.
        if target == torch.ops.aten.index.Tensor:
            base_node, idx_list = args[0], args[1]
            if not (isinstance(base_node, torch.fx.Node) and base_node.name in views):
                raise NotImplementedError(
                    "flex_attention on MPS only supports indexing captured buffers"
                )
            base, offset, sizes, strides = views[base_node.name]
            if any(ix is None for ix in idx_list) or len(idx_list) > len(sizes):
                raise NotImplementedError(
                    "flex_attention on MPS does not support this captured-buffer "
                    "indexing pattern"
                )
            for j, ix in enumerate(idx_list):
                term = f"(long)({_val(ix)}) * {strides[j]}"
                offset = term if offset == "0" else f"{offset} + {term}"
            k = len(idx_list)
            rem_sizes, rem_strides = sizes[k:], strides[k:]
            if rem_sizes:
                views[node.name] = (base, offset, rem_sizes, rem_strides)
            else:
                t = _tmp()
                code_lines.append(f"float {t} = (float){base}[{offset}];")
                var_map[node.name] = t
            continue

        t = _tmp()
        # Python operator builtins on captured SymInts (dynamic-shape closures).
        if target in _INT_PRESERVING_OPS:
            both_int = _is_int(args[0]) and _is_int(args[1])
            op = _INT_PRESERVING_OPS[target]
            ctype = int_ctype if both_int else "float"
            code_lines.append(f"{ctype} {t} = {_val(args[0])} {op} {_val(args[1])};")
            if both_int:
                int_vars.add(node.name)
        elif target in _CMP_OPS:
            code_lines.append(
                f"bool {t} = {_val(args[0])} {_CMP_OPS[target]} {_val(args[1])};"
            )
        elif target in (operator.floordiv, operator.mod):
            fn = "floor_divide" if target is operator.floordiv else "remainder"
            both_int = _is_int(args[0]) and _is_int(args[1])
            if both_int:
                a, b, ctype = _val(args[0]), _val(args[1]), int_ctype
            else:
                a = f"static_cast<float>({_val(args[0])})"
                b = f"static_cast<float>({_val(args[1])})"
                ctype = "float"
            code_lines.append(f"{ctype} {t} = c10::metal::{fn}({a}, {b});")
            if both_int:
                int_vars.add(node.name)
        elif target is operator.neg:
            is_int = _is_int(args[0])
            ctype = int_ctype if is_int else "float"
            code_lines.append(f"{ctype} {t} = -{_val(args[0])};")
            if is_int:
                int_vars.add(node.name)
        elif target in (operator.and_, operator.or_):
            a, b = _val(args[0]), _val(args[1])
            if _is_int(args[0]) and _is_int(args[1]):
                op = "&" if target is operator.and_ else "|"
                code_lines.append(f"{int_ctype} {t} = {a} {op} {b};")
                int_vars.add(node.name)
            else:
                op = "&&" if target is operator.and_ else "||"
                code_lines.append(f"bool {t} = ({a}) {op} ({b});")
        elif target in _OP_TABLE:
            out_type, fmt = _OP_TABLE[target]
            code_lines.append(
                f"{out_type} {t} = {fmt.format(*(_val(a) for a in args))};"
            )
        elif target == torch.ops.aten.clamp.default:
            v = _val(args[0])
            lo = _val(args[1]) if len(args) > 1 and args[1] is not None else None
            hi = _val(args[2]) if len(args) > 2 and args[2] is not None else None
            if lo is not None and hi is not None:
                code_lines.append(f"float {t} = metal::clamp({v}, {lo}, {hi});")
            elif lo is not None:
                code_lines.append(f"float {t} = metal::max({v}, {lo});")
            elif hi is not None:
                code_lines.append(f"float {t} = metal::min({v}, {hi});")
            else:
                code_lines.append(f"float {t} = {v};")
        elif target in (
            torch.ops.aten.div.Tensor_mode,
            torch.ops.aten.div.Scalar_mode,
        ):
            # Floor/trunc division (e.g. paged attention's kv_idx // page_size).
            mode = args[2] if len(args) > 2 else node.kwargs.get("rounding_mode")
            quot = f"static_cast<float>({_val(args[0])}) / static_cast<float>({_val(args[1])})"
            if mode == "floor":
                code_lines.append(f"float {t} = metal::floor({quot});")
            elif mode == "trunc":
                code_lines.append(f"float {t} = metal::trunc({quot});")
            else:
                code_lines.append(f"float {t} = {quot};")
        elif target in (
            torch.ops.aten.full_like.default,
            torch.ops.aten.full.default,
        ):
            fill = args[1] if len(args) > 1 else node.kwargs.get("fill_value", 0)
            code_lines.append(f"float {t} = {_val(fill)};")
        else:
            raise NotImplementedError(
                f"flex_attention on MPS does not support op {target} in "
                f"score_mod/mask_mod yet"
            )

        var_map[node.name] = t

    return "\n".join(code_lines)


def _compile_subgraph_to_metal(
    graph_module: torch.fx.GraphModule,
    placeholder_metal_names: list[str],
    captured_meta: list[Any],
    output_var: str,
    var_prefix: str,
    int_ctype: str = "int",
) -> str:
    """
    Bind placeholders by position to Metal variable names and emit parts of Metal .

    score_mod placeholders are [score, b, h, q_idx, kv_idx, *captured];
    mask_mod's are [b, h, q_idx, kv_idx, *captured]. The captured placeholders
    map positionally to `captured_meta` (each a `_CapturedBuf` or
    `_CapturedScalar`).
    """
    fixed: dict[str, str] = {}
    captured: dict[str, Any] = {}
    n_fixed = len(placeholder_metal_names)
    for i, node in enumerate(
        n for n in graph_module.graph.nodes if n.op == "placeholder"
    ):
        if i < n_fixed:
            fixed[node.name] = placeholder_metal_names[i]
        else:
            captured[node.name] = captured_meta[i - n_fixed]
    return _fx_graph_to_metal(
        graph_module, fixed, captured, output_var, var_prefix, int_ctype
    )


def _generate_mma_shader(
    metal_dtype,
    d_qk,
    d_v,
    kv_dim,
    block_m,
    block_n,
    full_kv_params,
    captured_params,
    scalar_params_str,
    unpack_code,
    score_code,
    mask_code,
    has_full_blocks,
    scale,
    lse_param="",
    lse_write_code="",
    max_param="",
    max_write_code="",
):
    """Generate Metal shader using simdgroup_matrix for Q@K^T."""

    def _block_loop_body(is_partial):
        if is_partial:
            mask_section = f"""
                    bool mask_result = true;
{mask_code}
"""
        else:
            mask_section = """
                    bool mask_result = true;
"""
        return f"""\
        for (int tile_start = kv_start; tile_start < kv_end; tile_start += BLOCK_N) {{
            int tile_end = min(tile_start + BLOCK_N, kv_end);
            int tile_size = tile_end - tile_start;

            // Load K into KV_tg, zero-padding rows past tile_size for MMA alignment.
            for (int i = (int)tid; i < BLOCK_N * D_QK; i += BLOCK_M) {{
                int n_local = i / D_QK;
                int d = i % D_QK;
                if (n_local < tile_size) {{
                    int n_global = tile_start + n_local;
                    long k_off = b_kv * stride_kz + hkv_idx * stride_kh + (long)n_global * stride_kn + (long)d * stride_kk;
                    KV_tg[n_local * D_QK + d] = K[k_off];
                }} else {{
                    KV_tg[n_local * D_QK + d] = ({metal_dtype})0;
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // S = Q_tg @ KV_tg^T via simdgroup MMA.
            simdgroup_matrix<float, 8, 8> S_frag[BLOCK_M / 8][BLOCK_N / 8];
            for (int ii = 0; ii < BLOCK_M / 8; ii++)
                for (int jj = 0; jj < BLOCK_N / 8; jj++) {{
                    S_frag[ii][jj].thread_elements()[0] = 0.0f;
                    S_frag[ii][jj].thread_elements()[1] = 0.0f;
                }}
            for (int k = 0; k < D_QK / 8; k++) {{
                simdgroup_matrix<{metal_dtype}, 8, 8> K_frags[BLOCK_N / 8];
                for (int j = 0; j < BLOCK_N / 8; j++)
                    simdgroup_load(K_frags[j], &KV_tg[j * 8 * D_QK + k * 8], (ulong)D_QK, ulong2(0), true);
                for (int i = 0; i < BLOCK_M / 8; i++) {{
                    simdgroup_matrix<{metal_dtype}, 8, 8> Q_frag;
                    simdgroup_load(Q_frag, &Q_tg[i * 8 * D_QK + k * 8], (ulong)D_QK);
                    for (int j = 0; j < BLOCK_N / 8; j++)
                        simdgroup_multiply_accumulate(S_frag[i][j], Q_frag, K_frags[j], S_frag[i][j]);
                }}
            }}
            for (int i = 0; i < BLOCK_M / 8; i++)
                for (int j = 0; j < BLOCK_N / 8; j++)
                    simdgroup_store(S_frag[i][j], &S_tg[i * 8 * BLOCK_N + j * 8], (ulong)BLOCK_N);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Load V into KV_tg (reuses the K slot; safe past the MMA barrier).
            for (int i = (int)tid; i < tile_size * D_V; i += BLOCK_M) {{
                int n_local = i / D_V;
                int d = i % D_V;
                int n_global = tile_start + n_local;
                long v_off = b_kv * stride_vz + hkv_idx * stride_vh + (long)n_global * stride_vn + (long)d * stride_vk;
                KV_tg[n_local * D_V + d] = V[v_off];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Fused per-thread softmax + V accumulation.
            if (active) {{
                for (int n_local = 0; n_local < tile_size; n_local++) {{
                    int n_idx = tile_start + n_local;
                    float s = S_tg[(int)tid * BLOCK_N + n_local] * SCALE_VAL;
                    float score_val = s;
{score_code}
                    s = score_val;
{mask_section}\
                    // s > -inf guards exp(-inf - -inf) = NaN when an entire row is -inf so far.
                    if (mask_result && s > -INFINITY) {{
                        float new_max = metal::max(row_max, s);
                        float old_scale_v = metal::precise::exp(row_max - new_max);
                        float p = metal::precise::exp(s - new_max);
                        row_sum = row_sum * old_scale_v + p;
                        for (int d = 0; d < D_V; d++) o_acc[d] *= old_scale_v;
                        for (int d = 0; d < D_V; d++)
                            o_acc[d] += p * float(KV_tg[n_local * D_V + d]);
                        row_max = new_max;
                    }}
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}"""

    partial_loop = _block_loop_body(is_partial=True)
    full_loop = _block_loop_body(is_partial=False)

    shader = f"""\
#include <metal_stdlib>
#include <c10/metal/utils.h>
using namespace metal;

constant int D_QK = {d_qk};
constant int D_V = {d_v};
constant int BLOCK_M = {block_m};
constant int BLOCK_N = {block_n};
constant int KV_DIM = {kv_dim};
constant float SCALE_VAL = {scale!r}f;

kernel void flex_attn_fwd(
    device {metal_dtype}* out [[buffer(0)]],
    constant {metal_dtype}* Q [[buffer(1)]],
    constant {metal_dtype}* K [[buffer(2)]],
    constant {metal_dtype}* V [[buffer(3)]],
    constant int* kv_num_blocks [[buffer(4)]],
    constant int* kv_indices [[buffer(5)]],
{full_kv_params}{captured_params}{lse_param}{max_param}{scalar_params_str},
    uint3 tgpos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {{
{unpack_code}

    int q_block = tgpos.x;
    int h_idx = tgpos.y;
    int b_idx = tgpos.z;
    int b_kv = b_idx % (int)Bkv;
    int m_base = q_block * BLOCK_M;
    int m_idx = m_base + (int)tid;
    bool active = (m_idx < N_Q);

    int hkv_idx = h_idx / (int)gqa_shared_heads;

    threadgroup {metal_dtype} Q_tg[BLOCK_M * D_QK];
    threadgroup {metal_dtype} KV_tg[BLOCK_N * KV_DIM];
    threadgroup float S_tg[BLOCK_M * BLOCK_N];

    // Q stays resident across all KV tiles.
    {{
        long q_base_global = b_idx * stride_qz + h_idx * stride_qh;
        for (int i = (int)tid; i < BLOCK_M * D_QK; i += BLOCK_M) {{
            int row = i / D_QK;
            int d = i % D_QK;
            int global_row = m_base + row;
            Q_tg[i] = (global_row < N_Q)
                ? Q[q_base_global + (long)global_row * stride_qm + (long)d * stride_qk]
                : ({metal_dtype})0;
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float o_acc[D_V];
    for (int d = 0; d < D_V; d++) o_acc[d] = 0.0f;

    int sparse_q_idx = m_base / (int)SPARSE_Q_BLOCK_SIZE;
    long sparse_idx_z = b_idx % SPARSE_Z;
    long sparse_idx_hq = h_idx % SPARSE_HQ;

    // Partial blocks: score_mod + mask_mod.
    int num_kv_blks = kv_num_blocks[sparse_idx_z * kv_nb_stride_z + sparse_idx_hq * kv_nb_stride_h + sparse_q_idx * kv_nb_stride_q];
    long kv_idx_base = sparse_idx_z * kv_idx_stride_z + sparse_idx_hq * kv_idx_stride_h + (long)sparse_q_idx * kv_idx_stride_q;

    for (int blk = 0; blk < num_kv_blks; blk++) {{
        int kv_block_idx = kv_indices[kv_idx_base + (long)blk * kv_idx_stride_b];
        int kv_start = kv_block_idx * (int)SPARSE_KV_BLOCK_SIZE;
        int kv_end = min(kv_start + (int)SPARSE_KV_BLOCK_SIZE, (int)N_KV);

{partial_loop}
    }}
"""

    if has_full_blocks:
        shader += f"""
    // Full blocks: score_mod only.
    int full_num_kv_blks = full_kv_num_blocks[sparse_idx_z * full_kv_nb_stride_z + sparse_idx_hq * full_kv_nb_stride_h + sparse_q_idx * full_kv_nb_stride_q];
    long full_kv_idx_base = sparse_idx_z * full_kv_idx_stride_z + sparse_idx_hq * full_kv_idx_stride_h + (long)sparse_q_idx * full_kv_idx_stride_q;

    for (int blk = 0; blk < full_num_kv_blks; blk++) {{
        int kv_block_idx = full_kv_indices[full_kv_idx_base + (long)blk * full_kv_idx_stride_b];
        int kv_start = kv_block_idx * (int)SPARSE_KV_BLOCK_SIZE;
        int kv_end = min(kv_start + (int)SPARSE_KV_BLOCK_SIZE, (int)N_KV);

{full_loop}
    }}
"""

    shader += f"""
    if (active) {{
{lse_write_code}{max_write_code}        if (row_sum == 0.0f) row_sum = 1.0f;
        long out_base = b_idx * stride_oz + h_idx * stride_oh + (long)m_idx * stride_om;
        for (int d = 0; d < D_V; d++)
            out[out_base + (long)d * stride_ok] = {metal_dtype}(o_acc[d] / row_sum);
    }}
}}
"""
    return shader


def _generate_metal_shader(
    dtype: torch.dtype,
    d_qk: int,
    d_v: int,
    score_mod_graph: torch.fx.GraphModule,
    mask_mod_graph: torch.fx.GraphModule,
    has_full_blocks: bool,
    block_m: int,
    scale: float,
    score_captured: Sequence[tuple] = (),
    mask_captured: Sequence[tuple] = (),
    write_lse: bool = False,
    write_max: bool = False,
    captures_fit_int32: bool = True,
) -> str:
    """Generate the complete Metal shader source for flex attention.

    `score_captured` / `mask_captured` describe the captures of score_mod /
    mask_mod in placeholder order. Each entry is a tagged tuple: a tensor
    capture ("tensor", sizes, strides, dtype) becomes an extra Metal buffer,
    while a SymInt capture ("scalar",) is read from the packed scalar buffer.
    """
    metal_dtype = METAL_DTYPE_MAP[dtype]

    bytes_per_elem = 4 if dtype == torch.float32 else 2
    kv_dim = max(d_qk, d_v)

    # simdgroup MMA needs each tile dim a multiple of 8.
    use_mma = (d_qk % 8 == 0) and (d_v % 8 == 0) and (block_m % 8 == 0)
    block_n = 0

    if use_mma:
        # Q_tg + KV_tg + S_tg must fit in the 32KB threadgroup memory budget.
        q_tg_bytes = block_m * d_qk * bytes_per_elem
        budget = 32768 - q_tg_bytes
        per_n = kv_dim * bytes_per_elem + block_m * 4
        max_block_n = budget // per_n
        block_n = min(32, max_block_n)
        block_n = (block_n // 8) * 8
        if block_n < 8:
            use_mma = False

    if not use_mma:
        block_n = min(32, 32768 // ((d_qk + d_v) * bytes_per_elem))
        block_n = max(1, block_n)

    buf_idx = 6  # 0=out, 1=Q, 2=K, 3=V, 4=kv_num_blocks, 5=kv_indices
    if has_full_blocks:
        full_kv_params = (
            f"    constant int* full_kv_num_blocks [[buffer({buf_idx})]],\n"
        )
        buf_idx += 1
        full_kv_params += f"    constant int* full_kv_indices [[buffer({buf_idx})]],\n"
        buf_idx += 1
    else:
        full_kv_params = ""

    # Captures follow the (full) kv buffers, in the order built in lower_mps:
    # score captures first, then mask captures. Tensor captures become extra
    # Metal buffers; SymInt captures are read from the packed scalar buffer
    captured_decls: list[str] = []
    scalar_capture_names: list[str] = []
    score_meta: list[Any] = []
    mask_meta: list[Any] = []
    for meta_list, caps in ((score_meta, score_captured), (mask_meta, mask_captured)):
        for entry in caps:
            if entry[0] == "scalar":
                sname = f"capsym{len(scalar_capture_names)}"
                scalar_capture_names.append(sname)
                meta_list.append(_CapturedScalar(sname))
                continue
            _, sizes, strides, cap_dtype = entry
            mtype = CAPTURE_DTYPE_MAP.get(cap_dtype)
            if mtype is None:
                raise NotImplementedError(
                    f"flex_attention on MPS does not support captured buffer "
                    f"dtype {cap_dtype}"
                )
            name = f"capbuf{len(captured_decls)}"
            captured_decls.append(
                f"    constant {mtype}* {name} [[buffer({buf_idx})]],"
            )
            meta_list.append(_CapturedBuf(name, list(sizes), list(strides), cap_dtype))
            buf_idx += 1
    captured_params = "".join(d + "\n" for d in captured_decls)

    # Pack scalars into one buffer to stay under Metal's 31-buffer limit.
    scalar_names = [
        "B",
        "Bkv",
        "Hq",
        "Hkv",
        "N_Q",
        "N_KV",
        "stride_qz",
        "stride_qh",
        "stride_qm",
        "stride_qk",
        "stride_kz",
        "stride_kh",
        "stride_kn",
        "stride_kk",
        "stride_vz",
        "stride_vh",
        "stride_vn",
        "stride_vk",
        "stride_oz",
        "stride_oh",
        "stride_om",
        "stride_ok",
        "SPARSE_KV_BLOCK_SIZE",
        "gqa_shared_heads",
        "SPARSE_Z",
        "SPARSE_HQ",
        "kv_nb_stride_z",
        "kv_nb_stride_h",
        "kv_nb_stride_q",
        "kv_idx_stride_z",
        "kv_idx_stride_h",
        "kv_idx_stride_q",
        "kv_idx_stride_b",
        "SPARSE_Q_BLOCK_SIZE",
    ]
    if has_full_blocks:
        scalar_names += [
            "full_kv_nb_stride_z",
            "full_kv_nb_stride_h",
            "full_kv_nb_stride_q",
            "full_kv_idx_stride_z",
            "full_kv_idx_stride_h",
            "full_kv_idx_stride_q",
            "full_kv_idx_stride_b",
        ]
    scalar_names += scalar_capture_names
    lse_param = ""
    lse_write_code = ""
    if write_lse:
        lse_param = f"    device float* lse [[buffer({buf_idx})]],\n"
        buf_idx += 1
        lse_write_code = (
            "        lse[(long)b_idx * (Hq * N_Q)"
            " + (long)h_idx * N_Q + (long)m_idx]"
            " = (row_sum > 0.0f)"
            f" ? (row_max + metal::precise::log(row_sum)) * {RCP_LN2}"
            " : -INFINITY;\n"
        )

    max_param = ""
    max_write_code = ""
    if write_max:
        max_param = f"    device float* max_scores [[buffer({buf_idx})]],\n"
        buf_idx += 1
        max_write_code = (
            "        max_scores[(long)b_idx * (Hq * N_Q)"
            " + (long)h_idx * N_Q + (long)m_idx]"
            f" = row_max * {RCP_LN2};\n"
        )

    scalar_params_str = f"    constant long* _params [[buffer({buf_idx})]]"

    int_ctype = "int" if captures_fit_int32 else "long"
    score_code = _compile_subgraph_to_metal(
        graph_module=score_mod_graph,
        placeholder_metal_names=["score_val", "b_idx", "h_idx", "m_idx", "n_idx"],
        captured_meta=score_meta,
        output_var="score_val",
        var_prefix="_sm",
        int_ctype=int_ctype,
    )
    mask_code = _compile_subgraph_to_metal(
        graph_module=mask_mod_graph,
        placeholder_metal_names=["b_idx", "h_idx", "m_idx", "n_idx"],
        captured_meta=mask_meta,
        output_var="mask_result",
        var_prefix="_mm",
        int_ctype=int_ctype,
    )

    capsym_set = OrderedSet(scalar_capture_names)
    unpack_code = "\n".join(
        f"    {int_ctype if name in capsym_set else 'long'} {name} = _params[{i}];"
        for i, name in enumerate(scalar_names)
    )

    if use_mma:
        return _generate_mma_shader(
            metal_dtype,
            d_qk,
            d_v,
            kv_dim,
            block_m,
            block_n,
            full_kv_params,
            captured_params,
            scalar_params_str,
            unpack_code,
            score_code,
            mask_code,
            has_full_blocks,
            scale,
            lse_param=lse_param,
            lse_write_code=lse_write_code,
            max_param=max_param,
            max_write_code=max_write_code,
        )

    # Fallback: per-thread dot product with cooperative K/V tiling.
    def _fallback_block_loop_body(is_partial):
        if is_partial:
            mask_section = f"""
                bool mask_result = true;
                {mask_code}
            """
        else:
            mask_section = """
                bool mask_result = true;
            """
        return f"""\
        for (int tile_start = kv_start; tile_start < kv_end; tile_start += BLOCK_N) {{
            int tile_end = min(tile_start + BLOCK_N, kv_end);
            int tile_size = tile_end - tile_start;

            for (int i = (int)tid; i < tile_size * D_QK; i += BLOCK_M) {{
                int n_local = i / D_QK;
                int d = i % D_QK;
                int n_global = tile_start + n_local;
                long k_off = b_kv * stride_kz + hkv_idx * stride_kh + (long)n_global * stride_kn + (long)d * stride_kk;
                K_tile[n_local * D_QK + d] = K[k_off];
            }}

            for (int i = (int)tid; i < tile_size * D_V; i += BLOCK_M) {{
                int n_local = i / D_V;
                int d = i % D_V;
                int n_global = tile_start + n_local;
                long v_off = b_kv * stride_vz + hkv_idx * stride_vh + (long)n_global * stride_vn + (long)d * stride_vk;
                V_tile[n_local * D_V + d] = V[v_off];
            }}

            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (active) {{
                for (int n_local = 0; n_local < tile_size; n_local++) {{
                    int n_idx = tile_start + n_local;

                    float s = 0.0f;
                    for (int d = 0; d < D_QK; d++) {{
                        s += q_row[d] * float(K_tile[n_local * D_QK + d]);
                    }}
                    s *= SCALE_VAL;

                    float score_val = s;
{score_code}
                    s = score_val;
{mask_section}\
                    // s > -inf guards exp(-inf - -inf) = NaN when an entire row is -inf so far.
                    if (mask_result && s > -INFINITY) {{
                        float new_max = metal::max(row_max, s);
                        float old_scale = metal::precise::exp(row_max - new_max);
                        float p = metal::precise::exp(s - new_max);
                        row_sum = row_sum * old_scale + p;
                        for (int d = 0; d < D_V; d++) o_acc[d] *= old_scale;
                        for (int d = 0; d < D_V; d++) {{
                            o_acc[d] += p * float(V_tile[n_local * D_V + d]);
                        }}
                        row_max = new_max;
                    }}
                }}
            }}

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}"""

    partial_loop = _fallback_block_loop_body(is_partial=True)
    full_loop = _fallback_block_loop_body(is_partial=False)

    shader = f"""\
#include <metal_stdlib>
#include <c10/metal/utils.h>
using namespace metal;

constant int D_QK = {d_qk};
constant int D_V = {d_v};
constant int BLOCK_M = {block_m};
constant int BLOCK_N = {block_n};
constant float SCALE_VAL = {scale!r}f;

kernel void flex_attn_fwd(
    device {metal_dtype}* out [[buffer(0)]],
    constant {metal_dtype}* Q [[buffer(1)]],
    constant {metal_dtype}* K [[buffer(2)]],
    constant {metal_dtype}* V [[buffer(3)]],
    constant int* kv_num_blocks [[buffer(4)]],
    constant int* kv_indices [[buffer(5)]],
{full_kv_params}{captured_params}{lse_param}{max_param}{scalar_params_str},
    uint3 tgpos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {{
{unpack_code}

    int q_block = tgpos.x;
    int h_idx = tgpos.y;
    int b_idx = tgpos.z;
    int b_kv = b_idx % (int)Bkv;
    int m_idx = q_block * BLOCK_M + (int)tid;
    bool active = (m_idx < N_Q);

    int hkv_idx = h_idx / (int)gqa_shared_heads;

    float q_row[D_QK];
    if (active) {{
        long q_base = b_idx * stride_qz + h_idx * stride_qh + (long)m_idx * stride_qm;
        for (int d = 0; d < D_QK; d++) {{
            q_row[d] = float(Q[q_base + (long)d * stride_qk]);
        }}
    }}

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float o_acc[D_V];
    for (int d = 0; d < D_V; d++) o_acc[d] = 0.0f;

    threadgroup {metal_dtype} K_tile[BLOCK_N * D_QK];
    threadgroup {metal_dtype} V_tile[BLOCK_N * D_V];

    int m_base = q_block * BLOCK_M;
    int sparse_q_idx = m_base / (int)SPARSE_Q_BLOCK_SIZE;
    long sparse_idx_z = b_idx % SPARSE_Z;
    long sparse_idx_hq = h_idx % SPARSE_HQ;

    int num_kv_blks = kv_num_blocks[sparse_idx_z * kv_nb_stride_z + sparse_idx_hq * kv_nb_stride_h + sparse_q_idx * kv_nb_stride_q];
    long kv_idx_base = sparse_idx_z * kv_idx_stride_z + sparse_idx_hq * kv_idx_stride_h + (long)sparse_q_idx * kv_idx_stride_q;

    for (int blk = 0; blk < num_kv_blks; blk++) {{
        int kv_block_idx = kv_indices[kv_idx_base + (long)blk * kv_idx_stride_b];
        int kv_start = kv_block_idx * (int)SPARSE_KV_BLOCK_SIZE;
        int kv_end = min(kv_start + (int)SPARSE_KV_BLOCK_SIZE, (int)N_KV);

{partial_loop}
    }}
"""

    if has_full_blocks:
        shader += f"""
    int full_num_kv_blks = full_kv_num_blocks[sparse_idx_z * full_kv_nb_stride_z + sparse_idx_hq * full_kv_nb_stride_h + sparse_q_idx * full_kv_nb_stride_q];
    long full_kv_idx_base = sparse_idx_z * full_kv_idx_stride_z + sparse_idx_hq * full_kv_idx_stride_h + (long)sparse_q_idx * full_kv_idx_stride_q;

    for (int blk = 0; blk < full_num_kv_blks; blk++) {{
        int kv_block_idx = full_kv_indices[full_kv_idx_base + (long)blk * full_kv_idx_stride_b];
        int kv_start = kv_block_idx * (int)SPARSE_KV_BLOCK_SIZE;
        int kv_end = min(kv_start + (int)SPARSE_KV_BLOCK_SIZE, (int)N_KV);

{full_loop}
    }}
"""

    shader += f"""
    if (active) {{
{lse_write_code}{max_write_code}        if (row_sum == 0.0f) row_sum = 1.0f;
        long out_base = b_idx * stride_oz + h_idx * stride_oh + (long)m_idx * stride_om;
        for (int d = 0; d < D_V; d++) {{
            out[out_base + (long)d * stride_ok] = {metal_dtype}(o_acc[d] / row_sum);
        }}
    }}
}}
"""
    return shader


class MetalFlexAttentionNode(ir.ExternKernelAlloc):
    """IR node for MPS flex attention; emits a compiled Metal shader at codegen."""

    def __init__(
        self,
        layout: ir.Layout,
        inputs: list[ir.IRNode],
        shader_source: str,
        scalar_args: list[Any],
        grid: tuple[Any, ...],
        block_m: int,
        num_mutated_outputs: int = 0,
    ):
        super().__init__(
            layout=layout,
            inputs=inputs,
            python_kernel_name="metal_flex_attention",
        )
        self.shader_source = shader_source
        self.scalar_args = scalar_args
        self.grid = grid
        self.block_m = block_m
        # The last num_mutated_outputs inputs (logsumexp, max_scores) are written
        # in place by the kernel.
        self.mutation_outputs = [
            # pyrefly: ignore [bad-argument-type]
            ir.MutationOutput(ir.NoneLayout(device=layout.device), buf, self)
            for buf in self.inputs[len(self.inputs) - num_mutated_outputs :]
        ]

    def codegen(self, wrapper) -> None:
        wrapper.add_import_once(
            "from torch._inductor.runtime.runtime_utils import compile_mps_shader"
        )
        wrapper.add_import_once("import torch")

        src_code = self.shader_source
        if src_code in wrapper.src_to_kernel:
            lib_name = wrapper.src_to_kernel[src_code]
        else:
            lib_name = f"_mps_flex_lib_{wrapper.next_kernel_suffix()}"
            wrapper.src_to_kernel[src_code] = lib_name
            wrapper.header.splice(
                f"{lib_name} = compile_mps_shader('''\n{src_code}\n''')"
            )

        name = self.get_name()
        layout: ir.Layout = self.layout  # type: ignore[assignment]
        sizes = ", ".join(wrapper.codegen_sizevar(s) for s in layout.size)
        strides = ", ".join(wrapper.codegen_sizevar(s) for s in layout.stride)
        device_str = f"{layout.device.type!r}"
        wrapper.writeline(
            f"{name} = torch.empty_strided("
            f"({sizes},), ({strides},), "
            f"dtype={layout.dtype!r}, device={device_str})"
        )

        # One packed int64 buffer keeps us under Metal's 31-buffer limit.
        scalar_strs = [wrapper.codegen_sizevar(s) for s in self.scalar_args]
        params_name = f"_flex_params_{name}"
        wrapper.writeline(
            f"{params_name} = torch.tensor("
            f"[{', '.join(scalar_strs)}], dtype=torch.int64, device={device_str})"
        )

        arg_parts = [name]
        for inp in self.inputs:
            # pyrefly: ignore [missing-attribute]
            arg_parts.append(inp.codegen_reference())
        arg_parts.append(params_name)
        args_str = ", ".join(arg_parts)

        grid_x = wrapper.codegen_sizevar(self.grid[0])
        grid_y = wrapper.codegen_sizevar(self.grid[1])
        grid_z = wrapper.codegen_sizevar(self.grid[2])

        wrapper.writeline(
            f"{lib_name}.flex_attn_fwd("
            f"{args_str}, "
            f"threads=[{grid_x} * {self.block_m}, {grid_y}, {grid_z}], "
            f"group_size=[{self.block_m}, 1, 1])"
        )
<|endoftext|># mypy: allow-untyped-defs
from typing import Any

import sympy

import torch
from torch.utils._sympy.symbol import symbol_is_type, SymT

from .. import config
from ..runtime.hints import AttrsDescriptorWrapper, DeviceProperties
from ..utils import (
    _type_of,
    device_supports_fp64,
    expr_fits_within_32bit,
    triton_version_uses_attrs_dict,
)
from ..virtualized import V
from .common import (
    ArgName,
    ConstexprArg,
    KernelArgType,
    SizeArg,
    TensorArg,
    TMADescriptorArg,
    WorkspaceArg,
)


def should_unwrap_unspec_arg(name: str):
    if V.graph.is_unspec_arg(name):
        # Unwrap on all devices except CPU
        if V.graph.get_current_device_or_throw().type != "cpu":
            return True
        # Only unwrap on CPU if the input is not used as an output
        if name not in V.graph.mutated_buffers:
            return True
    return False


def use_uint8_triton_storage_for_cuda_float8_e4m3fn(
    dtype: torch.dtype,
    arg_name: str | None = None,
    *,
    device: torch.device | None = None,
) -> bool:
    # Triton rejects fp8e4nv pointer types before sm89, but eager CUDA can
    # still dequantize float8_e4m3fn values by treating storage as raw bytes.
    if dtype != torch.float8_e4m3fn or torch.version.hip is not None:
        return False
    if arg_name is not None and not arg_name.startswith("in_ptr"):
        return False

    if device is None:
        try:
            device = V.graph.get_current_device_or_throw()
        except AttributeError:
            return False

    if device.type != "cuda":
        return False

    return DeviceProperties.create(device).cc < 89


def signature_of(arg: KernelArgType, *, size_dtype: str | None) -> str:
    if isinstance(arg, TensorArg):
        typ = _type_of(arg.dtype)
        if should_unwrap_unspec_arg(arg.buffer):
            # had unwrapped 0d tensor as scalar
            new_typ = typ.lstrip("*")
            if new_typ in ["fp16", "bf16"]:
                return "fp32"
            else:
                return new_typ
        elif use_uint8_triton_storage_for_cuda_float8_e4m3fn(arg.dtype, arg.name):
            return "*u8"
        else:
            return typ
    if isinstance(arg, SizeArg):
        if arg.expr is None:
            if triton_version_uses_attrs_dict():
                # In newer versions of Triton, the signature includes "None" args
                # and their type is marked as "constexpr"
                return "constexpr"
            else:
                # In older versions of Triton...
                # From triton/runtime/jit.py
                # `None` is nullptr.  Implicitly convert to *i8.
                return "*i8"
        elif _arg_equals_1(arg) and triton_version_uses_attrs_dict():
            # In new versions of Triton, if we have an equal-to-1 arg that's marked as a constant,
            # it should be marked as "constexpr" in the signature.
            return "constexpr"
        elif isinstance(arg.expr, (float, sympy.Float)):
            # Python floats are natively fp64, so use fp64 to preserve precision
            if config._use_fp64_for_unbacked_floats and device_supports_fp64(
                V.graph.current_device
            ):
                return "fp64"
            return "fp32"
        elif isinstance(arg.expr, sympy.Symbol) and symbol_is_type(
            arg.expr, (SymT.UNBACKED_FLOAT)
        ):
            # Unbacked floats from .item() should preserve fp64 precision
            if config._use_fp64_for_unbacked_floats and device_supports_fp64(
                V.graph.current_device
            ):
                return "fp64"
            return "fp32"
        elif isinstance(arg.expr, bool):
            return "i1"

        # if this is an integer
        if size_dtype == "tl.int32":
            return "i32"
        elif size_dtype == "tl.int64":
            return "i64"
        elif size_dtype is None:
            # no hint: we'll see if we know that this is a 32-bit int, and guard if possible.
            int_max = torch.iinfo(torch.int32).max
            if expr_fits_within_32bit(arg.expr):
                V.graph.sizevars.check_leq(arg.expr, int_max)
                return "i32"
            else:
                return "i64"
        else:
            raise NotImplementedError(f"unhandled size_dtype {size_dtype}")
    if isinstance(arg, WorkspaceArg):
        return _type_of(arg.dtype)
    if isinstance(arg, TMADescriptorArg):
        if arg.api_type == "experimental":
            return "nvTmaDesc"
        else:
            # https://github.com/triton-lang/triton/blob/9695baed9b46cf957e08b157bb4133f4a4b331c5/python/triton/runtime/jit.py#L360-L363
            if arg.api_type != "stable":
                raise AssertionError(
                    f"expected api_type == 'stable', got {arg.api_type}"
                )
            if arg.block_shape is None:
                raise AssertionError("expected block_shape to not be None")
            if arg.dtype is None:
                raise AssertionError("expected dtype to not be None")
            inner = _type_of(arg.dtype)[1:]  # strip the `*`: *fp32 -> fp32
            return f"tensordesc<{inner}{list(arg.block_shape)}>"
    if isinstance(arg, ConstexprArg):
        return "constexpr"
    raise NotImplementedError(f"unhandled {type(arg)}: {arg}")


def non_constexpr_signature(signature):
    new_signature = []
    for arg in signature:
        if not isinstance(arg, ConstexprArg):
            new_signature.append(arg)

    return new_signature


def select_tile_hint(size_hints, signature):
    """Pick TileHint.SQUARE vs TileHint.DEFAULT for 2D pointwise; None for 1D.

    SQUARE applies when the kernel has exactly 2 size hints and 4 non-constexpr
    signature args (input, output, and 2 numel args -> a square 2D tile).
    """
    from torch._inductor.runtime.hints import TileHint

    if len(size_hints) != 2:
        return None
    if len(non_constexpr_signature(signature)) == 4:
        return TileHint.SQUARE
    return TileHint.DEFAULT


def signature_to_meta(
    signature: list[KernelArgType],
    *,
    size_dtype: str | None,
    argdefs: list[ArgName],
    indices: list[int] | None = None,
    is_template: bool = False,
) -> dict[str, str]:
    if indices is None:
        indices = list(range(len(signature)))

    def _decide_tl_dtype(arg):
        # Even if the ks0 symbol itself is within tl.int32 range, it's
        # risky to use tl.int32 dtype since we may have ks0*ks1 later
        # for kernels like torch.mean when dynamic shape is enabled.
        #
        # Check config.triton.use_block_ptr, since Triton block pointer
        # does not support 64bit indexing:
        # https://gist.github.com/shunting314/6a41c776171720ce4561f202dcde0ad6
        #
        # If the triton metadata is for a template, don't use tl.int64 index.
        # Templates like flex attention/decoding uses block pointers which
        # does not support 64 bit indexing.
        if (
            not config.triton.use_block_ptr
            and not is_template
            and isinstance(arg, SizeArg)
            and arg.name.startswith("ks")
        ):
            return "tl.int64"
        return size_dtype

    return {
        argdefs[i].name: signature_of(arg, size_dtype=_decide_tl_dtype(arg))
        for i, arg in zip(indices, signature)
    }


def _get_buffer_layout(buf_name: str) -> "torch._inductor.ir.Layout":
    """Get the layout for a buffer, handling both scheduler buffers and graph inputs."""
    if V.graph.scheduler:
        layout = V.graph.scheduler.get_buffer_layout(buf_name)
    else:
        buffer = V.graph.try_get_buffer(buf_name)
        # output arg
        if not buffer:
            if buf_name != V.kernel.output_node.name:
                raise AssertionError(
                    f"expected buf_name == output_node.name, got {buf_name}"
                )
            layout = V.kernel.output_node.layout
        else:
            layout = buffer.get_layout()
    return layout


def is_unaligned_buffer_name(buf_name: str) -> bool:
    if buf_name in V.graph.unaligned_buffers:
        return True

    if buf_name in V.graph.graph_inputs:
        # See Note: [Input Alignment handling in Inductor]
        # For graph inputs that is not recorded in V.graph.unaligned_buffers,
        # we know for sure the tensor is aligned.
        return False

    if buf_name in V.graph.constants:
        # all constants are assumed to be aligned
        return False

    layout = _get_buffer_layout(buf_name)
    if isinstance(layout, torch._inductor.ir.NonOwningLayout):
        return not layout.maybe_guard_aligned()
    else:
        return False


def is_unaligned_buffer(arg: TensorArg):
    return is_unaligned_buffer_name(arg.buffer)


def _arg_equals_1(arg: KernelArgType) -> bool:
    return (
        isinstance(arg, SizeArg)
        and isinstance(arg.expr, (int, sympy.Integer))
        and V.graph.sizevars.statically_known_equals(arg.expr, 1)  # type: ignore[arg-type]
    )


def equal_1_arg_indices(
    args: list[KernelArgType],
    *,
    indices: list[int] | None = None,
) -> tuple[int, ...]:
    if indices is None:
        indices = list(range(len(args)))

    equal_to_1 = tuple(i for i, arg in zip(indices, args) if _arg_equals_1(arg))

    return equal_to_1


def _is_tensor_within_2gb(arg: TensorArg) -> bool:
    """Check if a tensor argument's storage is provably within 2GB.

    Mirrors HIPBackend.is_within_2gb() but uses compile-time symbolic analysis
    instead of runtime tensor inspection. This enables canonicalize_pointers to
    decompose pointer arithmetic into (splat(base), offset) form for buffer ops.
    """
    MAX_BYTES = 2**31 - 1
    try:
        # Graph inputs aren't tracked by the scheduler; get their layout
        # from the graph_inputs dict to avoid KeyError in get_buffer_layout.
        if arg.buffer in V.graph.graph_inputs:
            inp = V.graph.graph_inputs[arg.buffer]
            if hasattr(inp, "get_layout"):
                layout = inp.get_layout()
            else:
                return False
        else:
            layout = _get_buffer_layout(arg.buffer)
        storage_bytes = layout.storage_size() * arg.dtype.itemsize
        return V.graph.sizevars.statically_known_true(storage_bytes <= MAX_BYTES)
    except Exception:
        return False


def config_of(
    args: list[KernelArgType],
    *,
    indices: list[int] | None = None,
    pointer_range_override: tuple[int, ...] | None = None,
    skip_cpp_wrapper_input_tensor_alignment: bool = False,
) -> Any:
    if indices is None:
        indices = list(range(len(args)))

    def is_aligned(x: KernelArgType, alignment: int, include_tensor: bool) -> bool:
        """
        Roughly follow triton code here:
        https://github.com/triton-lang/triton/blob/5282ed890d453e10b9ee30076ef89115dd197761/python/triton/runtime/jit.py#L208-L222
        """
        if isinstance(x, TensorArg):
            if include_tensor:
                offset_aligned = V.graph.sizevars.statically_known_multiple_of(
                    x.offset * x.dtype.itemsize,
                    alignment,  # type: ignore[arg-type]
                )
                return offset_aligned and not is_unaligned_buffer(x)
            else:
                return False
        if isinstance(x, SizeArg):
            # TODO(voz): These are kinda redundant, if we can solve out statically_known_multiple_of with
            # _maybe_evaluate_static...
            if x.name.startswith("load_seed_offset"):
                return False
            if x.expr is None:
                return False
            if isinstance(x.expr, (float, bool)):
                return False
            return V.graph.sizevars.statically_known_multiple_of(x.expr, alignment)  # type: ignore[arg-type]
        if isinstance(x, WorkspaceArg):
            # We allocate the workspace ourselves, so it is always aligned
            return True
        if isinstance(x, (TMADescriptorArg, ConstexprArg)):
            return False
        raise NotImplementedError(f"unhandled {type(x)}: {x}")

    def include_tensor_alignment(arg: KernelArgType) -> bool:
        if (
            not skip_cpp_wrapper_input_tensor_alignment
            or not V.graph.cpp_wrapper
            or V.graph.aot_mode
            or not isinstance(arg, TensorArg)
            or arg.buffer not in V.graph.graph_inputs
        ):
            return True

        try:
            input_idx = V.graph.graph_input_names.index(arg.buffer)
        except ValueError:
            return True
        return input_idx not in (V.graph.inputs_to_check or ())

    if config.triton.divisible_by_16:
        divisible_by_16 = tuple(
            i
            for i, arg in zip(indices, args)
            if is_aligned(
                arg,
                alignment=16,
                include_tensor=include_tensor_alignment(arg),
            )
        )
    else:
        divisible_by_16 = ()

    equal_to_1 = equal_1_arg_indices(args, indices=indices)

    # On AMD/HIP, tag tensor args whose storage fits in 2GB so Triton
    # can use 32-bit pointer offsets and emit buffer load/store ops.
    if pointer_range_override is not None:
        pointer_range_32 = pointer_range_override
    elif torch.version.hip is not None:
        pointer_range_32 = tuple(
            i
            for i, arg in zip(indices, args)
            if isinstance(arg, TensorArg) and _is_tensor_within_2gb(arg)
        )
    else:
        pointer_range_32 = ()

    # pyrefly: ignore [bad-argument-count, bad-argument-type]
    return AttrsDescriptorWrapper(divisible_by_16, equal_to_1, pointer_range_32)
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import dataclasses
import functools
import itertools
import logging
import re
import sys
from collections import defaultdict
from typing import Any, cast, TYPE_CHECKING

import sympy

import torch
import torch._logging

from ..._prims_common import is_integer_dtype
from ...utils._ordered_set import OrderedSet
from ...utils._sympy.functions import FloorDiv, Max, Min, ModularIndexing
from ...utils._sympy.symbol import symbol_is_type, SymT
from ...utils._sympy.value_ranges import ValueRanges
from .. import config, ir
from ..codecache import HalideCodeCache
from ..ir import get_reduction_combine_fn
from ..metrics import is_metric_table_enabled, log_kernel_metadata
from ..ops_handler import AddParenHandler
from ..runtime.hints import HalideInputSpec, HalideMeta
from ..utils import (
    get_bounds_index_expr,
    get_kernel_metadata,
    parallel_num_threads,
    sympy_index_symbol,
    sympy_subs,
)
from ..virtualized import _ops as ops, V
from .common import (
    BackendFeature,
    CSEVariable,
    DeferredLine,
    IndentedBuffer,
    KernelArgType,
    OpOverrides,
    PythonPrinter,
    SizeArg,
    TensorArg,
)
from .cpp import DTYPE_TO_CPP
from .cpp_utils import cexpr
from .simd import constant_repr, SIMDKernel, SIMDScheduling


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ..ops_handler import ReductionType, StoreMode
    from ..shape_propagation import BlockShapeType

log = logging.getLogger(__name__)


def halide_constant(val):
    if isinstance(val, int) and not (-2147483648 <= val <= 2147483647):
        info = torch.iinfo(torch.int64)
        if val == info.min:
            return "hl.Int(64).min()"
        if val == info.max:
            return "hl.Int(64).max()"
        return f"hl.i64({val!r})"
    if isinstance(val, float):
        return f"hl.f64({constant_repr(val)})"
    return repr(val)


class Unsupported(RuntimeError):
    def __init__(self, thing) -> None:
        super().__init__(f"halide backend does not support: {thing}")


class HalidePrinter(PythonPrinter):  # noqa: docstring_linter
    @staticmethod
    def cast_index(expr):
        return f"hl.cast({V.kernel.index_dtype}, {expr})"

    @staticmethod
    def cast_float(expr):
        return f"hl.cast(hl.Float(32), {expr})"

    def _print_Float(self, expr):
        return f"hl.f32({expr})"

    def _print_ToFloat(self, expr):
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"hl.f32({self._print(expr.args[0])})"

    def _print_floor(self, expr):
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return self.cast_index(f"hl.floor({self._print(expr.args[0])})")

    _print_FloorToInt = _print_floor

    def _print_Trunc(self, expr):
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return self.cast_index(f"hl.trunc({self._print(expr.args[0])})")

    _print_TruncToInt = _print_Trunc

    def _print_ceiling(self, expr):
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return self.cast_index(f"hl.ceil({self._print(expr.args[0])})")

    def _helper_sqrt(self, expr):
        # pyrefly: ignore [missing-attribute]
        return f"hl.sqrt({self.cast_float(self._print(expr))})"

    def _print_Where(self, expr):
        c = self.doprint(expr.args[0])
        p = self.doprint(expr.args[1])
        q = self.doprint(expr.args[2])
        return f"hl.select({c}, {p}, {q})"

    def _print_Min(self, expr):
        if len(expr.args) == 1:
            # pyrefly: ignore [missing-attribute]
            return self._print(expr.args[0])

        mid = len(expr.args) // 2
        # pyrefly: ignore [missing-attribute]
        a = self._print(Min(*expr.args[:mid]))
        # pyrefly: ignore [missing-attribute]
        b = self._print(Min(*expr.args[mid:]))
        return f"hl.min({a}, {b})"

    def _print_Max(self, expr):
        if len(expr.args) == 1:
            # pyrefly: ignore [missing-attribute]
            return self._print(expr.args[0])

        mid = len(expr.args) // 2
        # pyrefly: ignore [missing-attribute]
        a = self._print(Max(*expr.args[:mid]))
        # pyrefly: ignore [missing-attribute]
        b = self._print(Max(*expr.args[mid:]))

        return f"hl.max({a}, {b})"

    def _print_Abs(self, expr):
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return self.cast_index(f"hl.abs({self._print(expr.args[0])})")

    def _print_OpaqueUnaryFn_cos(self, expr):
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"hl.cos({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_cosh(self, expr):
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"hl.cosh({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_acos(self, expr):
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"hl.acos({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_sin(self, expr):
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"hl.sin({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_sinh(self, expr):
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"hl.sinh({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_asin(self, expr):
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"hl.asin({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_tan(self, expr):
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"hl.tan({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_tanh(self, expr):
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"hl.tanh({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_atan(self, expr):
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"hl.atan({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_log2(self, expr):
        raise NotImplementedError("log2")

    def _print_FloorDiv(self, expr):
        if expr.is_integer:
            return super()._print_FloorDiv(expr)

        x, div = expr.args
        x = self.cast_float(self.doprint(x))
        div = self.cast_float(self.doprint(div))
        return self.cast_index(f"hl.floor({x} / {div})")

    def _print_Round(self, expr):
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return self.cast_index(f"hl.round({self._print(expr.args[0])})")

    _print_RoundToInt = _print_Round

    def _print_IntTrueDiv(self, expr):
        a, b = expr.args
        # force a cast to float
        return f"({a}) / ({b}+hl.f32(0))"

    def _print_RoundDecimal(self, expr):
        val, n = expr.args
        # pyrefly: ignore [missing-attribute]
        val = self._print(val)
        n = int(n)
        return f"hl.f32({10.0 ** (-n)!r})*hl.round(({val})*hl.f32({10.0**n!r}))"


texpr = HalidePrinter().doprint
pexpr = PythonPrinter().doprint


_halide_type = {
    torch.bool: "hl.Bool()",
    torch.bfloat16: "hl.BFloat(16)",
    torch.float16: "hl.Float(16)",
    torch.float32: "hl.Float(32)",
    torch.float64: "hl.Float(64)",
    torch.int8: "hl.Int(8)",
    torch.int16: "hl.Int(16)",
    torch.int32: "hl.Int(32)",
    torch.int64: "hl.Int(64)",
    torch.uint8: "hl.UInt(8)",
    torch.uint16: "hl.UInt(16)",
    torch.uint32: "hl.UInt(32)",
    torch.uint64: "hl.UInt(64)",
}


def halide_type(dtype):
    return _halide_type[dtype]


def halide_acc_type(dtype):
    if is_integer_dtype(dtype) and dtype.is_signed and dtype != torch.int64:
        dtype = torch.int32
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    return halide_type(dtype)


class HalideOverrides(OpOverrides):
    @staticmethod
    def to_dtype(
        x,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types=True,
    ):
        if dtype == torch.bool:
            return f"({x} != 0)"
        return f"hl.cast({halide_type(dtype)}, {x})"

    @staticmethod
    def to_dtype_bitcast(x, dtype: torch.dtype, src_dtype: torch.dtype):
        if src_dtype in (torch.float16, torch.bfloat16):
            x = f"hl.cast({halide_type(src_dtype)}, {x})"  # body compute is upcast to fp32
        line = f"hl.reinterpret({halide_type(dtype)}, {x})"
        if dtype in (torch.float16, torch.bfloat16):
            line = f"hl.cast(hl.Float(32), {line})"
        return line

    @classmethod
    def constant(cls, value, dtype):
        return cls.to_dtype(halide_constant(value), dtype)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def abs(x):
        return f"hl.abs({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def exp(x):
        if not hasattr(x, "name"):
            return f"hl.exp({x})"
        return f"hl.fast_exp(hl.cast(hl.Float(32), {x})) if {x.name}.type().bits() <= 32 else hl.exp({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def sqrt(x):
        return f"hl.sqrt({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def minimum(a, b):
        # return f"hl.min({a}, {b})"  <== handles nan wrong
        if not hasattr(a, "name"):
            return f"hl.min({a}, {b})"
        b = f"hl.cast({a.name}.type(), {b})"
        return f"hl.select(({a}<{b})|hl.is_nan({a}), {a}, {b}) if {a.name}.type().is_float() else hl.min({a}, {b})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def maximum(a, b):
        # return f"hl.max({a}, {b})"  <== handles nan wrong
        if not hasattr(a, "name"):
            return f"hl.max({a}, {b})"
        b = f"hl.cast({a.name}.type(), {b})"
        return f"hl.select(({a}>{b})|hl.is_nan({a}), {a}, {b}) if {a.name}.type().is_float() else hl.max({a}, {b})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def fmaximum(a, b):
        return f"hl.max({a}, {b})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def where(a, b, c):
        if hasattr(b, "name"):
            c = f"hl.cast({b.name}.type(), {c})"
        return f"hl.select({a}, {b}, {c})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def cos(x):
        return f"hl.cos({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def sin(x):
        return f"hl.sin({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def lgamma(x):
        raise Unsupported("lgamma")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def erf(x):
        return f"hl.erf({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def cosh(x):
        return f"hl.cosh({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def sinh(x):
        return f"hl.sinh({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def acos(x):
        return f"hl.acos({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def acosh(x):
        return f"hl.acosh({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def asin(x):
        return f"hl.asin({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def asinh(x):
        return f"hl.asinh({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def atan2(x, y):
        return f"hl.atan2({x}, {y})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def atan(x):
        return f"hl.atan({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def atanh(x):
        return f"hl.atanh({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def copysign(x, y):
        raise Unsupported("copysign")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def erfinv(x):
        raise Unsupported("erfinv")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def hypot(x, y):
        return f"hl.hypot({x}, {y})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def nextafter(x, y):
        raise Unsupported("nextafter")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def logical_and(a, b):
        return f"{a} & {b}"

    @staticmethod
    def logical_not(a):
        return f"{a} == 0"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def logical_or(a, b):
        return f"{a} | {b}"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def logical_xor(a, b):
        return f"({a} ^ {b})"

    @staticmethod
    def bitwise_and(a, b):
        return f"{a} & {b}"

    @staticmethod
    def bitwise_not(a):
        return f"~{a}"

    @staticmethod
    def bitwise_or(a, b):
        return f"{a} | {b}"

    @staticmethod
    def bitwise_xor(a, b):
        return f"{a} ^ {b}"

    @staticmethod
    def bitwise_left_shift(a, b):
        return f"{a} << {b}"

    @staticmethod
    def bitwise_right_shift(a, b):
        return f"{a} >> {b}"

    @staticmethod
    def rand(seed, offset):
        return f"halide_helpers.rand({seed}, {offset})"

    @staticmethod
    def randn(seed, offset):
        return f"halide_helpers.randn({seed}, {offset})"

    @staticmethod
    def rand_eager(seed, base_offset, threads_per_round, tid, vec):
        return f"halide_helpers.rand_eager_kernel({seed}, {base_offset}, {threads_per_round}, {tid}, {vec})"

    @staticmethod
    def randint64(seed, offset, low, high):
        return f"halide_helpers.randint64({seed}, {offset}, {low}, {high})"

    @staticmethod
    def load_seed(name, offset):
        return f"{ops.load(name, 0)} + {V.kernel.args.seed_offset('load_seed_offset', offset)}"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def rsqrt(x):
        # return f"hl.fast_inverse_sqrt({x})"  <== accuracy issues
        return f"1./hl.sqrt({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def tan(x):
        return f"hl.tan({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def tanh(x):
        return f"hl.tanh({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def signbit(x):
        return f"(hl.reinterpret(hl.UInt(32), hl.cast(hl.Float(32), {x})) >> 31) != 0"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def fmod(a, b):
        # TODO(jansel): find a better way to do this, builtin % has wrong sign
        return f"{a} - hl.trunc({a}/{b})*{b}"

    @staticmethod
    def pow(a, b):
        return f"hl.pow({a}, {b})"  # hl.fast_pow fails accuracy

    @staticmethod
    # pyrefly: ignore [bad-override]
    def ldexp(x, n):
        raise Unsupported("ldexp")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def log(x):
        return f"hl.log({x})"  # hl.fast_log fails accuracy

    @staticmethod
    def log2(x):
        raise NotImplementedError("log2")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def isinf(x):
        # workaround https://github.com/halide/Halide/issues/8309
        return f"hl.is_inf(hl.cast(hl.Float(32), {x}))"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def isnan(x):
        # workaround https://github.com/halide/Halide/issues/8309
        return f"hl.is_nan(hl.cast(hl.Float(32), {x}))"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def round(x):
        return f"hl.round({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def floor(x):
        return f"hl.floor({x})"

    @staticmethod
    def int_truediv(a, b):
        return f"({a}) / ({b} + hl.f32(0))"

    @staticmethod
    def floordiv(a, b):
        # TODO(jansel): find a better ways to do this, the select-based trick from triton.py didn't work
        return (
            f"hl.floor(hl.cast(hl.Float(max(32, {a.name}.type().bits())), {a}) / {b})"
        )

    @classmethod
    def sign(cls, x):
        left = ops.to_dtype(ops.lt("0", x), torch.int8)
        right = ops.to_dtype(ops.lt(x, "0"), torch.int8)
        sub = ops.sub(left, right)
        return f"hl.cast({x.name}.type(), {sub})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def trunc(x):
        return f"hl.trunc({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def truncdiv(a, b):
        # this causes crashes with floating point exception, see test_div_zero_dim_cpu
        # return f"hl.div_round_to_zero({a}, {b})"
        return (
            f"hl.trunc(hl.cast(hl.Float(max(32, {a.name}.type().bits())), {a}) / {b})"
        )

    @staticmethod
    # pyrefly: ignore [bad-override]
    def ceil(x):
        return f"hl.ceil({x})"

    @staticmethod
    def relu(x):
        return f"hl.max({x}, 0)"

    @classmethod
    def index_expr(cls, expr, dtype):
        index = V.kernel.prepare_indexing(expr)
        var = V.kernel.genfunc(
            V.kernel.index_to_str(index),
            V.kernel.used_dims_from_index(index),
            bounds=get_bounds_index_expr(expr),
        )
        if dtype not in (torch.int32, torch.int64):
            return ops.to_dtype(var, dtype)
        return var

    @classmethod
    def value_expr(cls, expr, dtype):
        index = V.kernel.prepare_indexing(expr)
        var = V.kernel.genfunc(
            V.kernel.index_to_str(index),
            V.kernel.used_dims_from_index(index),
            bounds=get_bounds_index_expr(expr),
        )
        return ops.to_dtype(var, dtype)

    @classmethod
    def indirect_indexing(cls, index_var, size, check=True, wrap_neg=True):
        # TODO(jansel): Halide only supports 32-bit indexing, we should error on overflow
        index_var = ops.to_dtype(index_var, torch.int32)
        index_var = ops.halide_clamp(index_var, size, check)
        index_var.indirect_indexing_size = size
        return sympy_index_symbol(str(index_var))

    @classmethod
    def halide_clamp(cls, value, size, check):
        end = V.kernel.kexpr(V.kernel.rename_indexing(size) - 1)
        if not isinstance(size, (int, sympy.Integer)):
            end = f"hl.cast({value.name}.type(), {end})"
        # Skip unsafe_promise_clamped to workaround: https://github.com/halide/Halide/issues/8261#issuecomment-2148835692
        # return f"hl.unsafe_promise_clamped({value}, 0, {end})"
        return f"hl.clamp({value}, 0, {end})"

    @staticmethod
    def masked(mask, body, other):
        with V.kernel.mask_loads(mask, other) as new_mask:
            result = body()

        if result.bounds.is_bool:
            other = bool(other)

        # Take dtype from result to prevent accidental promotion
        other = V.kernel.genfunc(
            f"hl.cast({result.name}.type(), {halide_constant(other)})",
            [],
            bounds=ValueRanges.wrap(other),
            shape=result.shape,
        )
        # TODO(jansel): look into removing the where in the same places triton does
        return ops.where(new_mask, result, other)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def frexp(x):
        raise NotImplementedError("frexp")

    @staticmethod
    def device_assert_async(cond, msg):
        raise NotImplementedError("device_assert_async")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def partial_accumulate(
        name: str,
        reduction_type: str,
        value: CSEVariable,
        extra_meta: dict[str, Any],
    ) -> None:
        raise NotImplementedError


HalideOverrides._initialize_pointwise_overrides("halide")


class HalideCSEVariable(CSEVariable):
    undefined_re = re.compile(r"\b(tmp\d+)\[\?\]")

    def __init__(
        self,
        name,
        bounds: ValueRanges[Any],
        dtype: torch.dtype | None = None,
        shape: BlockShapeType = None,
    ) -> None:
        super().__init__(name, bounds, dtype, shape=shape)
        self.used_dims: list[sympy.Symbol] | None = None

    def update_on_args(self, name, args, kwargs):
        used = OrderedSet(self.used_dims or ())
        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, HalideCSEVariable):
                if arg.used_dims is None:
                    raise AssertionError((name, arg, args))
                used.update(arg.used_dims)
        self.used_dims = V.kernel.sort_used_dims(used)

    def index_str(self, dims):
        if len(dims) == 0:
            return f"{self.name}[()]"
        # Reversed since Halide is column major
        return f"{self.name}[{', '.join(map(str, dims))}]"

    def __str__(self) -> str:
        if self.used_dims is None:
            # This will get recomputed and replaced in codegen_kernel()
            return f"{self.name}[?]"
        return self.index_str(self.used_dims)

    def subs_str(self, replacements):
        if not (
            self.used_dims is not None
            and all(isinstance(x, sympy.Expr) for x in self.used_dims)
        ):
            raise AssertionError("used_dims must be set sympy expressions")
        return self.index_str([replacements.get(n, n) for n in self.used_dims])


@dataclasses.dataclass
class DimensionInfo:
    expr: sympy.Expr | None
    size: sympy.Expr
    stride: sympy.Expr

    def __init__(self, expr, size, stride) -> None:
        super().__init__()
        if V.graph.sizevars.statically_known_lt(stride, 0):
            stride = -stride
            expr = -expr
        self.expr = expr
        self.size = size
        self.stride = stride

    def index_str(self, replacements=None, zero_vars=False):
        if self.expr is None:
            raise AssertionError("expr must not be None")
        expr = self.expr
        if zero_vars and expr == 0:
            return "hl.Var()"
        if replacements:
            replacements = {**replacements}
            # pyrefly: ignore [missing-attribute]
            for sym in expr.free_symbols:
                if symbol_is_type(sym, SymT.TMP):
                    if not isinstance(sym, sympy.Symbol):
                        raise AssertionError(f"expected sympy.Symbol, got {type(sym)}")
                    var = V.kernel.lookup_cse_var(sym.name)
                    if not isinstance(var, HalideCSEVariable):
                        raise AssertionError(
                            f"expected HalideCSEVariable, got {type(var)}"
                        )
                    replacements[sym] = sympy_index_symbol(var.subs_str(replacements))
            expr = sympy_subs(expr, replacements)
        return V.kernel.index_to_str(expr)


def eq(left, right):
    return V.graph.sizevars.guard_or_false(sympy.Eq(left, right))


def lt(left, right):
    """Compare sizes: only use on inputs known to be >= 0."""
    if V.graph.sizevars.guard_or_false(sympy.Lt(left, right)):
        return True

    # GCD fallback: if gcd(left, right) == left then left divides right,
    # so left <= right.  Combined with left != right this gives left < right.
    #
    # TODO: This is NOT always sound for unbacked symints.
    # e.g. lt(u0, 10*u0): gcd=u0, gcd==left, u0 != 10*u0 → returns True,
    # but if u0=0 then 0 < 0 is False.  The >= 0 checks mitigate the
    # negative case but the zero case remains.
    # TODO shall we add a runtime assertion at least.
    gcd = sympy.gcd(left, right)
    if gcd == left:
        return left != right

    return False


class HalideKernel(SIMDKernel):
    overrides = HalideOverrides  # type: ignore[assignment]
    kexpr: Callable[[sympy.Expr], str] = texpr

    def __init__(
        self,
        tiling: dict[str, sympy.Expr],
        **kwargs,
    ) -> None:
        super().__init__(tiling, **kwargs)
        # For halide, we just write directly to the body
        self.compute = self.body
        self.loads = self.body
        self.stores = self.body
        self.indexing_code_dom = IndentedBuffer()
        self.needs_dom_indexing = self.inside_reduction
        self.has_reduction = self.inside_reduction
        self.buffer_dimensions: dict[str, list[DimensionInfo]] = {}
        self.buffer_offsets: dict[str, sympy.Expr] = {}
        # {h0: size1, h1: size2, ...}
        self.halide_vars: dict[sympy.Symbol, sympy.Expr] = {}
        # {x0: h0, x1: h1+10*h2, ...}
        self.index_replacements: dict[sympy.Expr, sympy.Expr] = {}
        # {h1: hr1, ...}
        self.reduction_renames: dict[sympy.Symbol, sympy.Symbol] = {}
        # {"i": {h0: hi0}, "o": ...}
        self.dom_renames: dict[str, dict[sympy.Symbol, sympy.Symbol]] = {}
        # {"in_ptr0": ["in_ptr0_view0"], ...}
        self.buffer_aliases: dict[str, list[str]] = defaultdict(list)
        self.has_indirect_indexing = False

    def dtype_to_str(self, dtype: torch.dtype) -> str:
        return halide_type(dtype)

    # pyrefly: ignore [bad-override]
    def create_cse_var(self, name, bounds=None, dtype=None, shape=None):
        self.body.writeline(f"{name} = hl.Func({name!r})")
        # pyrefly: ignore [bad-argument-type]
        return HalideCSEVariable(name, bounds, dtype, shape)

    def finalize_indexing(self, indices: Sequence[sympy.Expr]):
        """
        Hook called right before codegen with every index that will be
        used in the fused kernel.

        This populates self.halide_vars/index_replacements/reduction_renames which is an alternate indexing
        scheme that avoids using divide and modulus.  Instead of xindex/yindex/rindex
        we base indexing on a larger number of vars whose product combines to those.

        This function populates self.halide_vars, self.index_replacements, and self.reduction_renames
        """
        if self.index_replacements or self.halide_vars or self.reduction_renames:
            raise AssertionError(
                "indexing state must be empty before finalize_indexing"
            )
        size_hint = functools.partial(
            V.graph.sizevars.optimization_hint, fallback=sys.maxsize
        )
        # pyrefly: ignore [bad-assignment]
        indices = dict.fromkeys(map(super().prepare_indexing, indices))
        all_used_symbols = OrderedSet[Any]()
        sym_to_node = {
            n.symbol(): n
            for n in itertools.chain.from_iterable(
                [tree.nodes.values() for tree in self.range_trees]
            )
        }

        def simplify(expr):
            return sympy.simplify(
                V.graph.sizevars.remove_precomputed_replacements(expr)
            )

        def visit_modular_indexing(base, divisor, modulus):
            if base in sym_to_node:
                node = sym_to_node[base]
                all_used_symbols.add(
                    node.root.lookup(
                        node.divisor * divisor,
                        V.graph.sizevars.evaluate_min(
                            modulus,
                            FloorDiv(node.length, divisor),
                        ),
                    ).symbol()
                )

        def visit_floor_div(base, divisor):
            if base in sym_to_node:
                node = sym_to_node[base]
                all_used_symbols.add(
                    node.root.lookup(
                        node.divisor * divisor,
                        FloorDiv(node.length, divisor),
                    ).symbol()
                )

        # first figure out all_used_symbols to do dead symbol elimination
        for index in indices:
            if index.has(ModularIndexing):
                index.replace(
                    ModularIndexing(
                        sympy.Wild("base"),
                        sympy.Wild("divisor"),
                        sympy.Wild("modulus"),
                    ),
                    visit_modular_indexing,
                )
            if index.has(FloorDiv):
                index.replace(
                    FloorDiv(
                        sympy.Wild("base"),
                        sympy.Wild("divisor"),
                    ),
                    visit_floor_div,
                )
            all_used_symbols.update(super().prepare_indexing(index).free_symbols)

        self.has_indirect_indexing = any(
            symbol_is_type(sym, SymT.INDIRECT) for sym in all_used_symbols
        )

        had_fallback = False
        for tree in reversed(self.range_trees):
            nodes = [n for n in tree.nodes.values() if n.symbol() in all_used_symbols]
            nodes.sort(key=lambda n: size_hint(n.divisor))
            if not nodes:
                nodes.append(tree.lookup(1, tree.numel))
            handled_count = 0
            divisor = sympy.S.One
            added_sym_size = []
            # decide on a minimal set of symbols and put them in self.halide_vars
            while handled_count < len(nodes) and not eq(tree.numel, divisor):
                sizes_to_add = [
                    simplify(n.length) for n in nodes if eq(n.divisor, divisor)
                ]
                handled_count += len(sizes_to_add)
                if not sizes_to_add:
                    raise AssertionError(nodes)
                end = divisor * functools.reduce(
                    lambda a, b: V.graph.sizevars.evaluate_max(a, b),
                    sizes_to_add,
                )
                sizes_to_add.extend(
                    [
                        simplify(n.divisor / divisor)
                        for n in nodes
                        if lt(divisor, n.divisor) and lt(n.divisor, end)
                    ]
                )
                # pyrefly: ignore [bad-assignment]
                while sizes_to_add:
                    next_size = functools.reduce(sympy.gcd, sizes_to_add)
                    if eq(next_size, 1):
                        # sizes share no common factors, e.g [2, 21, 42, 441, 889056]
                        # TODO(jansel): we should just prevent fusion in cases that hit this
                        next_size = simplify(tree.numel / divisor)
                        if eq(next_size, 1):
                            raise AssertionError("fallback next_size must not be 1")
                        sizes_to_add = []
                        handled_count = len(nodes)
                        had_fallback = True
                    sym = sympy_index_symbol(f"h{len(self.halide_vars)}")
                    # pyrefly: ignore [missing-argument]
                    if tree.is_reduction:
                        self.reduction_renames[sym] = sympy_index_symbol(
                            f"hr{len(self.halide_vars)}"
                        )
                    self.halide_vars[sym] = next_size
                    added_sym_size.append((sym, next_size))
                    divisor *= next_size
                    new_sizes = [n.length for n in nodes if eq(n.divisor, divisor)]
                    handled_count += len(new_sizes)
                    prior_len = len(sizes_to_add)
                    sizes_to_add = [
                        sympy.simplify(s / next_size)
                        for s in sizes_to_add
                        if not eq(s, next_size)
                    ]
                    if not (len(sizes_to_add) < prior_len or prior_len == 0):
                        raise AssertionError("sizes_to_add must shrink")
                    sizes_to_add.extend(new_sizes)

            # create a mapping to the new set of symbols in self.index_replacements
            for node in nodes:
                try:
                    idx = 0
                    divisor = 1
                    while not eq(node.divisor, divisor):
                        sym, size = added_sym_size[idx]
                        idx += 1
                        divisor *= size
                    length = 1
                    expr = sympy.S.Zero
                    while not eq(node.length, length):
                        sym, size = added_sym_size[idx]
                        idx += 1
                        expr += length * sym
                        length *= size
                    self.index_replacements[node.symbol()] = expr
                except IndexError:
                    if not had_fallback:
                        raise AssertionError("IndexError without fallback") from None
                    full_index = sympy.S.Zero
                    stride = sympy.S.One
                    for sym, size in added_sym_size:
                        full_index += stride * sym
                        stride *= size
                    self.index_replacements[node.symbol()] = (
                        V.graph.sizevars.simplify_with_ranges(
                            ModularIndexing(full_index, node.divisor, node.length),
                            self.halide_vars,  # type: ignore[arg-type]
                        )
                    )

        # codegen the variable definitions
        for sym in self.halide_vars:
            self.indexing_code.writeline(f"{sym} = hl.Var({sym.name!r})")
        if self.reduction_renames:
            self.codegen_rdom(
                "rdom",
                {rv: self.halide_vars[v] for v, rv in self.reduction_renames.items()},
            )

    def setup_dom_indexing(self):
        """RDom based indexing uses explicit iteration ranges for Func updates"""
        prefix = "i" if self.inside_reduction else "o"
        if prefix in self.dom_renames:
            return self.dom_renames[prefix]

        renames = {}
        for var in self.halide_vars:
            if not self.inside_reduction and var in self.reduction_renames:
                continue
            m = re.match(r"^h(\d+)$", var.name)
            if not m:
                raise AssertionError(f"unexpected halide var name {var.name}")
            renames[var] = sympy_index_symbol(f"h{prefix}{m.group(1)}")

        self.codegen_rdom(
            f"{prefix}dom", {rv: self.halide_vars[v] for v, rv in renames.items()}
        )

        self.dom_renames[prefix] = renames
        return renames

    def codegen_rdom(self, name, vars):
        rsizes = [
            f"hl.Range(0, {self.kexpr(self.rename_indexing(size))})"
            for size in vars.values()
        ]
        self.indexing_code.writeline(f"{name} = hl.RDom([{', '.join(rsizes)}])")
        for i, rsym in enumerate(vars.keys()):
            self.indexing_code.writeline(f"{rsym} = {name}[{i}]")

    def prepare_indexing(
        self,
        index: sympy.Expr,
    ):
        index = super().prepare_indexing(index)
        index = sympy_subs(index, self.index_replacements)
        return V.graph.sizevars.simplify_with_ranges(index, self.halide_vars)  # type: ignore[arg-type]

    def sym_size(self, sym):
        """The size of an index symbol"""
        if symbol_is_type(sym, SymT.TMP):
            return self.lookup_cse_var(sym.name).indirect_indexing_size
        return self.halide_vars[sym]

    def indexing_to_dimensions(self, var: str, index: sympy.Expr, is_store: bool):
        """Convert address-based indexing into dimensions using self.halide_vars"""
        symbols = []
        for sym in sorted(index.free_symbols, key=lambda x: x.name):  # type: ignore[attr-defined]
            if symbol_is_type(sym, (SymT.HALIDE, SymT.TMP)):
                symbols.append(sym)
            else:
                if not symbol_is_type(
                    sym,
                    (
                        SymT.UNBACKED_INT,
                        SymT.SIZE,
                        SymT.PRECOMPUTED_SIZE,
                    ),
                ):
                    raise AssertionError(sym)

        # group the expression by variables used
        offset = sympy.S.Zero
        split_expr = dict.fromkeys(symbols, sympy.S.Zero)
        split_failed: list[tuple[list[sympy.Symbol], sympy.Expr]] = []
        index = sympy.expand(self.rename_indexing(index))
        for part in index.args if isinstance(index, sympy.Add) else [index]:
            part_vars = [v for v in part.free_symbols if v in split_expr]
            if len(part_vars) == 0:
                offset += part
            elif len(part_vars) == 1:
                split_expr[part_vars[0]] += part
            else:
                new_split_failed = []
                for i in range(len(split_failed)):
                    if split_failed[i] is None:
                        raise AssertionError("split_failed entry must not be None")
                    other_vars, other_part = split_failed[i]
                    if OrderedSet(other_vars) & OrderedSet(part_vars):
                        part_vars.extend([v for v in other_vars if v not in part_vars])
                        part += other_part
                    else:
                        new_split_failed.append((other_vars, other_part))
                split_failed = [*new_split_failed, (part_vars, part)]

        def expr_to_dimension(expr, syms):
            expr = sympy.factor(expr)
            if len(syms) == 1:
                stride_wild = sympy.Wild("wild", exclude=symbols)
                m = expr.match(stride_wild * syms[0])
                if m:
                    return DimensionInfo(
                        syms[0], self.sym_size(syms[0]), m[stride_wild]
                    )
            if is_store:
                raise AssertionError(expr)
            length = sympy.simplify(
                sympy_subs(expr, {sym: self.sym_size(sym) - 1 for sym in syms}) + 1
            )
            stride = sympy.S.One
            if isinstance(expr, sympy.Mul):
                for term in expr.args:
                    if isinstance(term, sympy.Integer):
                        stride *= term
                        expr = sympy.simplify(expr / term)
                        length = sympy.simplify(sympy.ceiling(length / term))
            return DimensionInfo(expr, length, stride)

        # try to turn each group into a strided access
        dims = []
        for syms, expr in split_failed:
            for v in syms:
                expr += split_expr.pop(v)
            dims.append(expr_to_dimension(expr, syms))
        for sym, expr in split_expr.items():
            dims.append(expr_to_dimension(expr, [sym]))
        dims.sort(
            key=lambda d: V.graph.sizevars.optimization_hint(
                d.stride, fallback=sys.maxsize
            )
        )  # type: ignore[arg-type]

        if not dims:  # scalar load/store
            if self.has_indirect_indexing:
                # workaround https://github.com/halide/Halide/issues/8338
                dims.append(DimensionInfo(sympy.S.Zero, 1, 1))
        elif not V.graph.sizevars.statically_known_equals(dims[0].stride, 1):
            # Halide assumes dimension 0 is stride == 1, so add a dummy dimension
            dims.insert(
                0, DimensionInfo(sympy.S.Zero, 1 if is_store else dims[0].stride, 1)
            )

        if dims and not is_store:
            if var in self.buffer_offsets and V.graph.sizevars.statically_known_geq(
                offset, self.buffer_offsets[var]
            ):
                # reuse the existing offset to avoid needing an input alias
                self.apply_offset_to_dimension(dims, offset - self.buffer_offsets[var])
                offset = self.buffer_offsets[var]
            elif V.graph.sizevars.statically_known_gt(
                offset, 0
            ):  # TODO(jansel): negative offsets
                # roll the offset into the dimensions for cleaner indexing
                self.apply_offset_to_dimension(dims, offset)
                offset = 0

        orig_var = var
        for i in itertools.count():
            if self.install_dims(var, dims, offset, is_store):
                return var, dims
            if is_store:
                raise AssertionError("install_dims must succeed for stores")
            var = f"{orig_var}_view{i}"
            if var not in self.buffer_aliases[orig_var]:
                self.buffer_aliases[orig_var].append(var)

    def install_dims(self, var, dims, offset, is_store):
        """Try to set self.buffer_dimensions[var], return True on success"""
        if var not in self.buffer_dimensions:
            self.buffer_dimensions[var] = dims
            self.buffer_offsets[var] = offset
            return True
        if self.buffer_offsets[var] != offset or len(
            self.buffer_dimensions[var]
        ) != len(dims):
            return False
        if is_store:
            return self.buffer_dimensions[var] == dims
        for old, new in zip(self.buffer_dimensions[var], dims):
            if old.stride != new.stride:
                return False
            if old.size != new.size or old.expr != new.expr:
                old.size = V.graph.sizevars.evaluate_max(old.size, new.size)
                old.expr = None
        return True

    def apply_offset_to_dimension(self, dims, offset):
        if offset == 0:
            return
        for i in reversed(range(len(dims))):
            if dims[i].stride == 1 or V.graph.sizevars.statically_known_geq(
                offset, dims[i].stride
            ):
                part = FloorDiv(offset, dims[i].stride)
                offset -= part * dims[i].stride
                dims[i].expr += part
        if offset != 0:
            raise AssertionError(f"offset must be fully applied, got {offset}")

    def used_dims_from_index(self, index: sympy.Expr):
        """Detect which range trees are used to populate HalideCSEVariable.used_dims"""
        used_dims = OrderedSet[sympy.Symbol]()
        for sym in index.free_symbols:
            if not isinstance(sym, sympy.Symbol):
                raise AssertionError(f"expected sympy.Symbol, got {type(sym)}")
            if symbol_is_type(sym, SymT.TMP):
                # indirect indexing
                cse_var = self.lookup_cse_var(sym.name)
                if not (
                    isinstance(cse_var, HalideCSEVariable)
                    and cse_var.used_dims is not None
                ):
                    raise AssertionError(
                        "cse_var must be HalideCSEVariable with used_dims"
                    )
                used_dims.update(cse_var.used_dims)
            elif symbol_is_type(sym, SymT.HALIDE):
                used_dims.add(sym)
            elif symbol_is_type(
                sym, (SymT.UNBACKED_INT, SymT.SIZE, SymT.PRECOMPUTED_SIZE, SymT.INDEX)
            ):
                pass
            else:
                raise NotImplementedError(f"unhandled symbol {sym}")
        return self.sort_used_dims(used_dims)

    def sort_used_dims(self, used_dims):
        if not all(isinstance(x, sympy.Expr) for x in used_dims):
            raise AssertionError("used_dims must all be sympy expressions")
        ordered = [
            sym
            for sym in itertools.chain(
                self.halide_vars, self.reduction_renames.values()
            )
            if sym in used_dims
        ]
        if len(ordered) != len(used_dims):
            raise AssertionError(
                f"expected {len(used_dims)} ordered dims, got {len(ordered)}"
            )
        return ordered

    def make_index_str(self, dims, replacements=None, zero_vars=False):
        index_str = ", ".join(d.index_str(replacements, zero_vars) for d in dims)
        if len(dims) == 0:
            index_str = "()"
        elif len(dims) == 1:
            # workaround for https://github.com/halide/Halide/issues/8299
            index_str = f"{index_str},"
        return index_str

    def load(self, name: str, index: sympy.Expr):
        """Codegen a load from an InputBuffer"""
        var = self.args.input(name)
        index = self.prepare_indexing(index)
        var, dims = self.indexing_to_dimensions(var, index, False)
        line = f"{var}[{self.make_index_str(dims)}]"
        dtype = V.graph.get_dtype(name)
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
            line = f"hl.cast(hl.Float(32), {line})"

        if self._load_mask:
            if not (
                isinstance(self._load_mask, HalideCSEVariable)
                and self._load_mask.used_dims is not None
            ):
                raise AssertionError(
                    "_load_mask must be HalideCSEVariable with used_dims"
                )
            used_dims = OrderedSet(
                (*self.used_dims_from_index(index), *self._load_mask.used_dims)
            )
            result = self.newfunc(self.sort_used_dims(used_dims))
            if result.used_dims:
                self.body.writeline(f"{result.name}_mask = hl.RDom([hl.Range(0, 1)])")
                self.body.writeline(f"{result.name}_mask.where({self._load_mask})")
                other = self.kexpr(self._load_other or 0)  # type: ignore[arg-type]
                self.body.writeline(
                    f"{result} = hl.cast({halide_type(dtype)}, {other})"
                )
                self.body.writeline(
                    f"{result} = {line} + hl.cast({halide_type(dtype)}, {result.name}_mask)"
                )
            else:
                # scalar case
                self.body.writeline(
                    f"{result} = hl.select({self._load_mask}, {line}, hl.cast({halide_type(dtype)}, 0))"
                )
            return result
        else:
            return self.genfunc(line, self.used_dims_from_index(index))

    def lookup_cse_var(self, name: str):
        return self.cse.varname_map[re.sub(r"\[.*", "", name)]

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> None:
        """Codegen a store to an OutputBuffer"""
        if not isinstance(value, HalideCSEVariable):
            raise AssertionError(f"expected HalideCSEVariable, got {type(value)}")
        var = self.args.output(name)
        index = self.prepare_indexing(index)
        var, dims = self.indexing_to_dimensions(var, index, True)
        if self.is_indirect_indexing(index) or mode is not None:
            replacements = self.setup_dom_indexing()
            index_str = self.make_index_str(dims, replacements)
            value_str = value.subs_str(replacements)
            undef_dims = (", ".join(["hl.Var()"] * len(dims))) or "()"
            self.body.writeline(
                DeferredLine(name, f"{var}[{undef_dims}] = hl.undef({var}.type())")
            )
        else:
            index_str = self.make_index_str(dims, zero_vars=True)
            value_str = str(value)

        dtype = V.graph.get_dtype(name)
        if mode is None:
            line = f"{var}[{index_str}] = hl.cast({halide_type(dtype)}, {value_str})"
        elif mode == "atomic_add":
            line = f"{var}[{index_str}] += hl.cast({halide_type(dtype)}, {value_str})"
        else:
            raise NotImplementedError(f"store mode={mode}")
        self.body.writeline(DeferredLine(name, line))

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: CSEVariable | tuple[CSEVariable, ...],
    ) -> CSEVariable | tuple[CSEVariable, ...]:
        """Codegen a reduction operation"""
        if not self.inside_reduction:
            raise AssertionError("reduction called outside reduction")
        if self._load_mask:
            raise AssertionError("reduction with active load mask")
        cache_key = (src_dtype, reduction_type, value)
        if cache_key in self.cse.reduction_cache:
            return self.cse.reduction_cache[cache_key]

        if isinstance(value, tuple):
            if reduction_type != "welford_combine":
                raise AssertionError(f"expected welford_combine, got {reduction_type}")
            self.cse.reduction_cache[cache_key] = result_tuple = (
                self.welford_combine_impl(*value)
            )
            return result_tuple

        if not (isinstance(value, HalideCSEVariable) and value.used_dims is not None):
            raise AssertionError("value must be HalideCSEVariable with used_dims")
        reduction_vars = OrderedSet(self.reduction_renames)
        result_var = self.newfunc(
            [v for v in value.used_dims if v not in reduction_vars],
        )
        if reduction_vars - OrderedSet(value.used_dims):
            value = self.genfunc(
                f"{value}",
                self.sort_used_dims(OrderedSet((*value.used_dims, *reduction_vars))),
                shape=value.shape,
            )
        value_str = value.subs_str(self.reduction_renames)
        default = ir.Reduction.default_accumulator(reduction_type, src_dtype)
        acc_type = halide_acc_type(dtype)

        if reduction_type in ("argmax", "argmin"):
            index = f"{result_var.name}_{reduction_type}"
            self.body.writeline(f"{index} = hl.{reduction_type}(rdom, {value_str})")
            # turn the N-D argmax index into a 1-D one
            parts = []
            stride = 1
            for i, sym in enumerate(self.reduction_renames):
                # pyrefly: ignore [bad-argument-type]
                parts.append(f"{index}[{i}]")
                if stride != 1:
                    # pyrefly: ignore [unsupported-operation]
                    parts[-1] += f"*{stride}"
                stride *= self.halide_vars[sym]
            self.body.writeline(f"{result_var} = {' + '.join(parts)}")
        elif reduction_type == "welford_reduce":
            # TODO(jansel): implement welford_reduce without fallback
            result_var = self.welford_reduce_fallback(dtype, value)
        else:
            combine_fn = get_reduction_combine_fn(reduction_type, acc_type)
            with V.set_ops_handler(AddParenHandler(HalideOverrides())):
                combine_str = combine_fn(result_var, value_str)  # type: ignore[arg-type]
            default_str = f"hl.cast({acc_type}, {halide_constant(default)})"
            self.body.writeline(f"{result_var} = {default_str}")
            self.body.writeline(f"{result_var} = {combine_str}")

        self.cse.reduction_cache[cache_key] = result_var
        return result_var

    def welford_combine_impl(self, mean, m2, weight):
        if not (isinstance(mean, HalideCSEVariable) and mean.used_dims is not None):
            raise AssertionError("mean must be HalideCSEVariable with used_dims")
        if not (isinstance(m2, HalideCSEVariable) and m2.used_dims is not None):
            raise AssertionError("m2 must be HalideCSEVariable with used_dims")
        if not (isinstance(weight, HalideCSEVariable) and weight.used_dims is not None):
            raise AssertionError("weight must be HalideCSEVariable with used_dims")
        used_dims = OrderedSet(
            (*mean.used_dims, *m2.used_dims, *weight.used_dims) or self.halide_vars
        )
        used_dims -= OrderedSet(self.reduction_renames)
        result_var = self.newfunc(self.sort_used_dims(used_dims))
        default = [f"hl.cast({x.name}.type(), 0)" for x in (mean, m2, weight)]
        pfx = result_var.name
        self.body.writeline(f"{result_var} = hl.Tuple([{', '.join(default)}])")
        self.body.writeline(f"{pfx}_mean_1 = {result_var}[0]")
        self.body.writeline(f"{pfx}_m2_1 = {result_var}[1]")
        self.body.writeline(f"{pfx}_weight_1 = {result_var}[2]")
        self.body.writeline(f"{pfx}_mean_2 = {mean.subs_str(self.reduction_renames)}")
        self.body.writeline(f"{pfx}_m2_2 = {m2.subs_str(self.reduction_renames)}")
        self.body.writeline(
            f"{pfx}_weight_2 = {weight.subs_str(self.reduction_renames)}"
        )
        self.body.writeline(f"{pfx}_delta = {pfx}_mean_2 - {pfx}_mean_1")
        self.body.writeline(f"{pfx}_new_weight = {pfx}_weight_1 + {pfx}_weight_2")
        self.body.writeline(
            f"{pfx}_w2_over_w = hl.select({pfx}_new_weight == 0.0, 0.0, {pfx}_weight_2 / {pfx}_new_weight)"
        )
        update = [
            f"{pfx}_mean_1 + {pfx}_delta * {pfx}_w2_over_w",
            f"{pfx}_m2_1 + {pfx}_m2_2 + {pfx}_delta * {pfx}_delta * {pfx}_weight_1 * {pfx}_w2_over_w",
            f"{pfx}_new_weight",
        ]
        self.body.writeline(f"{result_var} = hl.Tuple([{', '.join(update)}])")

        unpacked = []
        for i in range(3):
            unpacked.append(self.newfunc(result_var.used_dims))
            self.body.writeline(f"{unpacked[-1]} = {result_var}[{i}]")
        return tuple(unpacked)

    def scan(
        self,
        dtypes: tuple[torch.dtype, ...],
        combine_fn: Callable[
            [tuple[CSEVariable, ...], tuple[CSEVariable, ...]], tuple[CSEVariable, ...]
        ],
        values_orig: tuple[CSEVariable, ...],
    ) -> tuple[CSEVariable, ...]:
        if not self.inside_reduction:
            raise AssertionError("scan called outside reduction")
        if len(dtypes) != len(values_orig):
            raise AssertionError(
                f"expected {len(values_orig)} dtypes, got {len(dtypes)}"
            )
        values: list[HalideCSEVariable] = []
        all_used_dims = OrderedSet[sympy.Symbol]()

        for value in values_orig:
            if not (
                isinstance(value, HalideCSEVariable) and value.used_dims is not None
            ):
                raise AssertionError("value must be HalideCSEVariable with used_dims")
            if OrderedSet(value.used_dims) & OrderedSet(self.reduction_renames):
                values.append(value)
            else:
                values.append(
                    self.genfunc(
                        f"{value}",
                        [*value.used_dims, [*self.reduction_renames][:1]],
                        shape=value.shape,
                    )
                )
            all_used_dims.update(value.used_dims)
        result_var = self.newfunc(self.sort_used_dims(all_used_dims))
        if not (
            result_var.used_dims
            and OrderedSet(result_var.used_dims) & OrderedSet(self.reduction_renames)
        ):
            raise AssertionError("result_var must use reduction dims")
        initial = [
            f"hl.cast({halide_acc_type(dtype)}, {value})"
            for dtype, value in zip(dtypes, values)
        ]

        length = self.kexpr(self.rename_indexing(self.range_trees[-1].numel))
        scan_dom = f"{result_var.name}_rdom"
        scan = f"{scan_dom}.x"
        self.body.writeline(f"{scan_dom} = hl.RDom([hl.Range(1, {length})])")

        if len(self.reduction_renames) != 1:
            raise AssertionError("multi-dimensional scan not implemented")
        (scan_var,) = [*self.reduction_renames]  # type: ignore[misc]
        scan_renames_cur = {scan_var: sympy_index_symbol(scan)}
        scan_renames_pri = {scan_var: sympy_index_symbol(scan) - 1}

        if len(values) == 1:

            def maybe_tuple(x):
                return x[0]

            read_left = [result_var.subs_str(scan_renames_pri)]
            read_right = [result_var.subs_str(scan_renames_cur)]
        else:

            def maybe_tuple(x):
                return f"hl.Tuple([{', '.join(x)}])"

            read_left = [
                result_var.subs_str(scan_renames_pri) + f"[{i}]"
                for i in range(len(values))
            ]
            read_right = [
                result_var.subs_str(scan_renames_cur) + f"[{i}]"
                for i in range(len(values))
            ]

        self.body.writeline(f"{result_var} = {maybe_tuple(initial)}")

        # Disable CSE for update fn
        with V.set_ops_handler(AddParenHandler(HalideOverrides())):
            combine_str = combine_fn(read_left, read_right)  # type: ignore[arg-type]
        self.body.writeline(
            f"{result_var.subs_str(scan_renames_cur)} = {maybe_tuple(combine_str)}"
        )

        if len(values) == 1:
            return (result_var,)

        unpack_vars = [self.newfunc(self.sort_used_dims(all_used_dims)) for _ in values]
        for i, v in enumerate(unpack_vars):
            self.body.writeline(f"{v} = {result_var}[{i}]")
        return tuple(unpack_vars)

    def genfunc(
        self,
        line,
        used_dims,
        *,
        bounds=ValueRanges.unknown(),
        shape: BlockShapeType = None,
    ) -> HalideCSEVariable:
        var = self.cse.generate(self.body, line, bounds=bounds, shape=shape)
        if not isinstance(var, HalideCSEVariable):
            raise AssertionError(f"expected HalideCSEVariable, got {type(var)}")
        var.used_dims = used_dims
        return var

    def newfunc(self, used_dims, *, shape: BlockShapeType = None) -> HalideCSEVariable:
        var = self.cse.newvar(shape=shape)
        if not isinstance(var, HalideCSEVariable):
            raise AssertionError(f"expected HalideCSEVariable, got {type(var)}")
        var.used_dims = used_dims
        return var

    def halide_buffer_numel(self, name: str):
        """
        We map all tensors to 1D buffers in Halide since Halide has trouble representing some strides that PyTorch
        supports.  If there are gaps in the underlying layout the numel we pass to Halide includes the gaps while
        PyTorch's numel excludes them.
        """
        return V.graph.get_buffer(name).get_layout().storage_size()

    def halide_argdefs(self):
        """
        Halide requires scalar inputs before outputs, so need to reorder args.
        """

        def arg_order(arg_tuple):
            _call_str, arg = arg_tuple
            if isinstance(arg, SizeArg):
                return 1  # this would normally be at the end, move it to middle
            elif "out_ptr" in arg.name:
                return 2
            else:
                if "in_ptr" not in arg.name:
                    raise AssertionError(f"expected in_ptr in arg name, got {arg.name}")
                return 0

        result: list[tuple[str | None, KernelArgType]] = []
        _, a, b, _ = self.args.python_argdefs()
        for call_str, arg in sorted(zip(a, b), key=arg_order):
            result.append((call_str, arg))
            if isinstance(arg, TensorArg):
                if not (arg.offset == 0 and arg.alias_of is None):
                    raise AssertionError("TensorArg must have zero offset and no alias")
                result.extend(
                    (
                        None,
                        TensorArg(
                            alias,
                            arg.buffer,
                            arg.dtype,
                            arg.offset,
                            alias_of=arg.name,
                        ),
                    )
                    for alias in self.buffer_aliases.get(arg.name, ())
                )
        return result

    def halide_kernel_meta(self) -> HalideMeta:
        """Compute metadata required by codecache.py"""
        argtypes = []
        for _, arg in self.halide_argdefs():
            if isinstance(arg, SizeArg):
                shape = None
                stride = None
                offset = None
                dtype = "long"
            else:
                shape = [
                    cexpr(self.rename_indexing(x.size))
                    for x in self.buffer_dimensions[arg.name]
                ]
                stride = [
                    cexpr(self.rename_indexing(x.stride))
                    for x in self.buffer_dimensions[arg.name]
                ]
                if len(shape) != len(stride):
                    raise AssertionError(
                        f"shape/stride length mismatch: {len(shape)} vs {len(stride)}"
                    )
                offset = cexpr(self.buffer_offsets[arg.name])
                dtype = f"{DTYPE_TO_CPP[arg.dtype]}*"
            argtypes.append(
                HalideInputSpec(
                    dtype,
                    arg.name,
                    shape=shape,
                    stride=stride,
                    offset=offset,
                    alias_of=arg.alias_of,
                )
            )

        current_device = V.graph.get_current_device_or_throw()
        if current_device.type == "cpu":
            target = [config.halide.cpu_target]
            scheduler = config.halide.scheduler_cpu
            scheduler_flags = {
                "parallelism": parallel_num_threads(),
            }
            cuda_device = None
        else:
            if current_device.type != "cuda":
                raise AssertionError("only cpu/cuda supported")
            if current_device.index > 0:
                raise AssertionError("only default device supported")
            target = [config.halide.gpu_target]
            scheduler = config.halide.scheduler_cuda
            capability = torch.cuda.get_device_properties(current_device)
            if "cuda_capability" not in target[0]:
                for major, minor in [(8, 6), (8, 0), (7, 5), (7, 0), (6, 1)]:
                    if capability.major >= major and capability.minor >= minor:
                        target.append(f"cuda_capability_{major}{minor}")
                        break
            target.append("user_context")
            scheduler_flags = {
                "parallelism": capability.multi_processor_count,
                # TODO(jansel): explore other flags, see:
                # grep parser.parse ~/Halide/src/autoschedulers/anderson2021/AutoSchedule.cpp
            }
            cuda_device = max(0, current_device.index)

        # strict_float is required for correctness
        target.append("strict_float")

        # without this we will initialize cuda once per kernel and hit errors
        target.append("no_runtime")

        if not config.halide.asserts:
            target.append("no_asserts")

        if config.halide.debug:
            target.append("debug")

        if "64" in self.index_dtype:
            # TODO(jansel): it is unclear if this does anything, since input sizes are still int32
            target.append("large_buffers")

        return HalideMeta(
            argtypes,
            target="-".join(target),
            scheduler=scheduler,
            scheduler_flags=scheduler_flags,  # type: ignore[arg-type]
            cuda_device=cuda_device,
        )

    def codegen_kernel(self, name=None):
        """Called at the end to generate a final kernel string"""
        if self.args.inplace_buffers:
            raise Unsupported("inplace_buffers")
        meta = self.halide_kernel_meta()  # ensure needed args are added early
        code = IndentedBuffer()
        code.splice(
            """
            import halide as hl
            from torch._inductor.runtime import halide_helpers
            from math import inf, nan

            @hl.generator(name="kernel")
            class Kernel:
        """,
            strip=True,
        )
        code.do_indent()
        for _, arg in self.halide_argdefs():
            if isinstance(arg, SizeArg):
                code.writeline(f"{arg.name} = hl.InputScalar({self.index_dtype})")
            else:
                if not arg.buffer:
                    raise AssertionError(arg)
                argcls = "hl.OutputBuffer" if "out" in arg.name else "hl.InputBuffer"
                argtype = halide_type(arg.dtype)
                ndim = len(self.buffer_dimensions[arg.name])
                code.writeline(f"{arg.name} = {argcls}({argtype}, {ndim})")
        code.splice(
            """
            def generate(g):
        """
        )
        code.do_indent()
        for _, arg in self.halide_argdefs():
            code.writeline(f"{arg.name} = g.{arg.name}")
        for old, new in self.args.aliases():
            code.writeline(f"{old} = {new}")
        code.splice(self.indexing_code)

        def update_index(m):
            var = cast(HalideCSEVariable, self.cse.varname_map[m.group(1)])
            if var.used_dims is None:
                raise AssertionError(var)
            return str(var)

        for line in self.body._lines:
            if isinstance(line, str):
                # fill in missing indices
                line = HalideCSEVariable.undefined_re.sub(update_index, line)
            code.writeline(line)
        code.writeline("")
        code.writeline("assert g.using_autoscheduler()")

        for _, arg in self.halide_argdefs():
            # fallback=1 below because halide requires buffers to be at least as large as the estimates
            # This causes crashes if our estimate is greater than the vector length
            # https://github.com/halide/Halide/issues/3103
            if isinstance(arg, SizeArg):
                hint = V.graph.sizevars.optimization_hint(arg.expr, fallback=1)
                code.writeline(f"{arg.name}.set_estimate({hint})")
            else:
                dims = self.buffer_dimensions[arg.name]
                range_hints = []
                for i, dim in enumerate(dims):
                    hint = self._autoscheduler_workarounds(
                        V.graph.sizevars.optimization_hint(dim.size, fallback=1), dims
                    )
                    # pyrefly: ignore [bad-argument-type]
                    range_hints.append(f"hl.Range(0, {hint})")
                    if "out" not in arg.name:
                        code.writeline(f"{arg.name}.dim({i}).set_min(0)")
                        try:
                            code.writeline(
                                f"{arg.name}.dim({i}).set_stride({int(dim.stride)})"
                            )
                        except TypeError:
                            pass  # not integer
                        try:
                            code.writeline(
                                f"{arg.name}.dim({i}).set_extent({int(dim.size)})"
                            )
                        except TypeError:
                            pass  # not integer
                code.writeline(f"{arg.name}.set_estimates([{', '.join(range_hints)}])")

        code.do_unindent(2)
        code.splice(
            """
            if __name__ == "__main__":
                hl.main()
            """.rstrip(),
        )
        if meta.scheduler:
            code.splice(
                f"""
                else:
                    hl.load_plugin({HalideCodeCache.find_libautoschedule(meta.scheduler)!r})
                    target = hl.Target({meta.target!r})
                    autoscheduler = hl.AutoschedulerParams({meta.scheduler!r}, {meta.scheduler_flags!r})
                    with hl.GeneratorContext(target, autoscheduler):
                        gen = Kernel()
                        pipeline = gen._build_pipeline()
                        # gen.compile_to_callable() does not run the autoscheduler
                        pipeline.apply_autoscheduler(target, autoscheduler)
                        kernel = pipeline.compile_to_callable([
                                gen._get_input_parameter(a.name)._to_argument()
                                for a in gen._get_arginfos()
                                if a.dir == hl.ArgInfoDirection.Input
                            ], target)
                """,
                strip=True,
            )
        else:
            code.splice(
                f"""
                  else:
                      with hl.GeneratorContext(hl.Target({meta.target!r})):
                          kernel = Kernel().compile_to_callable()
                  """,
                strip=True,
            )
        return code.getvalue()

    @staticmethod
    def _autoscheduler_workarounds(n, dims):
        if (
            len(dims) == 1
            and config.halide.scheduler_cuda == "Anderson2021"
            and V.graph.get_current_device_or_throw().type == "cuda"
        ):
            # workaround https://github.com/halide/Halide/issues/8246
            n = max(2, n)
        return n

    def call_kernel(self, name: str, node=None, deallocate_ws: bool = True):
        """Codegen a call to this kernel"""
        wrapper = V.graph.wrapper_code
        call_args = [f"{n}" for n, arg in self.halide_argdefs() if arg.alias_of is None]
        current_device = V.graph.get_current_device_or_throw()
        if current_device.type == "cuda":
            stream_name = wrapper.write_get_raw_stream(
                current_device.index, V.graph.name
            )
            call_args.append(stream_name)
        wrapper.generate_kernel_call(
            name,
            call_args,
            device=current_device,
            triton=False,
        )

    def generate_assert(self, check):
        return False  # TODO(jansel): support asserts

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ):
        pass  # TODO(jansel): support asserts


class HalideScheduling(SIMDScheduling):
    kernel_type = HalideKernel  # type: ignore[arg-type,assignment]

    @classmethod
    def get_backend_features(cls, device: torch.device) -> OrderedSet[BackendFeature]:
        result = OrderedSet(
            [
                BackendFeature.TUPLE_REDUCTION,
                BackendFeature.PREFER_STORE_LOOP_ORDER,
                BackendFeature.REDUCE_TO_SINGLE_ELEMENT,
            ]
        )
        if config.halide.scan_kernels:
            result.add(BackendFeature.SCAN)
        return result

    def define_kernel(self, src_code, node_schedule, kernel):
        """Codegen kernel definition to go in output wrapper code"""
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            kernel_name = f"halide_kernel_{wrapper.next_kernel_suffix()}"
            wrapper.src_to_kernel[src_code] = kernel_name
            wrapper.add_import_once(
                "from torch._inductor.runtime.hints import HalideMeta, HalideInputSpec"
            )

            compile_wrapper = IndentedBuffer()
            compile_wrapper.writeline(
                f"async_compile.halide({kernel.halide_kernel_meta()!r}, '''"
            )
            compile_wrapper.splice(src_code, strip=True)
            compile_wrapper.writeline("''')")

            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            metadata_comment = f"{origins}\n{detailed_origins}"
            wrapper.define_kernel(
                kernel_name, compile_wrapper.getvalue(), metadata_comment
            )
            if is_metric_table_enabled("kernel_metadata"):
                log_kernel_metadata(kernel_name, "", src_code)

        return kernel_name
<|endoftext|>import contextlib
import functools
import itertools
import logging
from collections.abc import Callable
from typing import Any

import sympy

import torch
import torch._inductor.config as config
from torch._dynamo.utils import counters
from torch._inductor import ir
from torch._inductor.autotune_process import (
    SubgraphCPUBenchmarkRequest,
    SubgraphGPUBenchmarkRequest,
    TensorMeta,
)
from torch._inductor.codegen.common import KernelTemplate
from torch._inductor.ir import (
    Buffer,
    FixedLayout,
    get_free_symbols,
    get_symbolic_inputs,
    gm_original_output_strides,
    ir_node_to_tensor,
    Layout,
)
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.utils import do_bench_using_profiling
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)


def inline_subgraph_to_ir_nodes(
    gm: torch.fx.GraphModule, inputs: list[Any], name: str
) -> Any:
    """Inline a subgraph by converting its FX operations to individual IR nodes.

    This converts a subgraph to multiple ComputedBuffer nodes (fusable),
    enabling epilogue fusion with subsequent operations.

    Returns:
        TensorBox containing the final operation result as individual IR nodes
    """
    from torch._inductor.lowering import process_subgraph_nodes

    # Temporarily switch V.graph.module to subgraph during processing; restore to prevent IR nodes added to wrong graph
    original_module = V.graph.module
    try:
        V.graph.module = gm
        return process_subgraph_nodes(gm, inputs)
    finally:
        V.graph.module = original_module


class SubgraphChoiceCaller(ir.ChoiceCaller):
    """
    Represents a Subgraph Autotuning choice, and the subgraph can be any arbitrary
    GraphModule. Compiles the Subgraph down to a module for benchmarking.
    """

    def __init__(
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        description: str,
        make_fx_graph: Callable[..., Any],
        input_gen_fns: dict[int, Callable[[Any], torch.Tensor]] | None = None,
    ) -> None:
        super().__init__(name, input_nodes, layout, description)

        # Create inputs for tracing and benchmarking:
        # - trace_inputs: Always use symbolic inputs for tracing so the graph works
        #   for any size at runtime. This correctly captures shape-dependent operations
        #   like x * x.shape[0].
        # - benchmark_inputs: Use input_gen_fns if provided (for range-specific benchmarks)
        #   or concrete inputs from ir_node_to_tensor with guard_shape=False.
        trace_inputs = []
        self.benchmark_inputs = []
        with V.fake_mode:
            for i, inp in enumerate(self.input_nodes):
                # Here there will be no unbacked symbols, as SubgraphBuffer does not support them
                if len(get_free_symbols(inp.get_size(), unbacked_only=True)) != 0:
                    raise AssertionError("expected no unbacked symbols in input size")
                if len(get_free_symbols(inp.get_stride(), unbacked_only=True)) != 0:
                    raise AssertionError("expected no unbacked symbols in input stride")

                inp.data.freeze_layout()  # type: ignore[attr-defined]

                # Always use symbolic inputs for tracing
                trace_inputs.append(ir_node_to_tensor(inp))

                # Use input_gen_fn for benchmarking if provided, otherwise concrete sizes
                if input_gen_fns is not None and i in input_gen_fns:
                    self.benchmark_inputs.append(input_gen_fns[i](inp))
                else:
                    self.benchmark_inputs.append(
                        ir_node_to_tensor(inp, replace_symbols_with_hints=True)
                    )

        # Trace with symbolic inputs for guard detection
        self.gm = make_fx_graph(*trace_inputs)
        gm_original_output_strides(self.gm)
        # Store symbolic inputs for sym_input computation
        self.example_inputs = trace_inputs

        self.sym_inputs = get_symbolic_inputs(self.input_nodes)
        self.sym_input_values = self._compute_sym_input_values()

        # Cached decomposition info for range-based dispatch (set via cache_decomposition)
        self.decomposition: Callable[..., Any] | None = None
        self.decomposition_kwargs: dict[str, Any] = {}
        # Config patches to apply during kernel codegen (e.g., coordinate_descent_tuning)
        self.config_patches: dict[str, Any] = {}
        # Cache compiled module to avoid recompiling on every benchmark call
        self._compiled_module: Any = None
        # Cache benchmark request for async autotuning
        self._bmreq: (
            SubgraphGPUBenchmarkRequest | SubgraphCPUBenchmarkRequest | None
        ) = None

        # Pre-compile only if using async pipelined autotuning
        # Must happen in __init__ because compilation requires virtualized context (V.graph, V.debug)
        if config.pipeline_max_autotune_gemm:
            with V.fake_mode:
                self._compiled_module = self._compile_for_benchmarking()
                self._bmreq = self._create_benchmark_request()

    def _compute_sym_input_values(self) -> list[int]:
        """Extract concrete dimension values for sym_inputs from benchmark_inputs.

        The compiled module expects symbolic dimension values as runtime arguments.
        This maps each symbolic variable to its concrete value from the benchmark tensors.
        Used for range based autotuning.
        """
        sym_input_names = OrderedSet(
            [s.name for s in self.sym_inputs if hasattr(s, "name")]
        )

        # Build mapping: symbolic dimension name -> concrete value
        sym_name_to_value: dict[str, int] = {}
        for inp_node, benchmark_inp in zip(self.input_nodes, self.benchmark_inputs):
            if isinstance(benchmark_inp, torch.Tensor):
                for sym_dim, actual_dim in zip(
                    inp_node.get_size(), benchmark_inp.shape
                ):
                    if isinstance(sym_dim, sympy.Symbol):
                        sym_name_to_value[sym_dim.name] = int(actual_dim)
                    elif str(sym_dim) in sym_input_names:
                        sym_name_to_value[str(sym_dim)] = int(actual_dim)

        result = []
        for sym_var in self.sym_inputs:
            if isinstance(sym_var, sympy.Symbol) and sym_var.name in sym_name_to_value:
                result.append(sym_name_to_value[sym_var.name])
            else:
                hint = V.graph.sizevars.shape_env.optimization_hint(sym_var, fallback=1)
                result.append(int(hint))
        return result

    def cache_decomposition(
        self, decomposition: Callable[..., Any], kwargs: dict[str, Any]
    ) -> None:
        """Cache decomposition function and kwargs for range-based dispatch lookup."""
        self.decomposition = decomposition
        self.decomposition_kwargs = kwargs

    def __str__(self) -> str:
        return f"SubgraphCaller({self.name})"

    def _compile_for_benchmarking(self) -> Any:
        """Compile the subgraph for benchmarking, returns the compiled module."""
        from torch._inductor.graph import GraphLowering

        safe_name = self.name.replace("::", "_").replace(".", "_")

        # Symbolic inputs produce code with symbolic sizes resolved at runtime
        # via sym_input_values. Without symbols (static shapes), compile with
        # benchmark_inputs whose sizes match the actual benchmark tensors.
        compile_inputs = (
            self.example_inputs if self.sym_inputs else self.benchmark_inputs
        )
        log.debug("Benchmark compile %s: sym_inputs=%s", self.name, self.sym_inputs)

        if self.gm is None:
            raise AssertionError("expected self.gm to be set")
        bm_graph_lowering = GraphLowering(
            gm=self.gm,
            example_inputs=compile_inputs,
            shape_env=V.graph._shape_env,
            cpp_wrapper=V.graph.cpp_wrapper,
            aot_mode=V.graph.aot_mode,
            extern_node_serializer=V.graph.extern_node_serializer,
            is_inference=V.graph.is_inference,
            is_backward=V.graph.is_backward,
            name=f"benchmark_{safe_name}",
        )

        for sym_inp in self.sym_inputs:
            bm_graph_lowering.graph_inputs[sym_inp.name] = sym_inp
            bm_graph_lowering.graph_input_names.append(sym_inp.name)

        with V.set_graph_handler(bm_graph_lowering):
            # Apply config_patches during benchmarking (e.g., coordinate_descent_tuning)
            # Also disable max_autotune to avoid nested autotuning
            benchmark_config: dict[str, Any] = {
                "max_autotune": False,
                "max_autotune_gemm": False,
                "max_autotune_gemm_backends": "ATEN",
                "benchmark_fusion": False,
                "pipeline_max_autotune_gemm": False,
                **self.config_patches,
            }
            with config.patch(benchmark_config):
                bm_graph_lowering.run(*compile_inputs)
                return bm_graph_lowering.compile_to_module()

    def _create_benchmark_request(
        self,
    ) -> SubgraphGPUBenchmarkRequest | SubgraphCPUBenchmarkRequest:
        """Create a benchmark request for async autotuning."""
        if self._compiled_module is None:
            raise AssertionError(
                "Module must be compiled before creating benchmark request"
            )
        input_tensor_meta = TensorMeta.from_irnodes(self.input_nodes)
        output_tensor_meta = TensorMeta.from_irnodes(self.layout)

        if self.layout.device.type == "cpu":
            bmreq_cls = SubgraphCPUBenchmarkRequest
        else:
            bmreq_cls = SubgraphGPUBenchmarkRequest

        return bmreq_cls(
            kernel_name=self.name,
            input_tensor_meta=input_tensor_meta,
            output_tensor_meta=output_tensor_meta,
            extra_args=tuple(),
            module_path=self._compiled_module.__file__,
            module_cache_key=self._compiled_module.key,
            sym_input_values=self.sym_input_values,
        )

    @property
    def bmreq(
        self,
    ) -> SubgraphGPUBenchmarkRequest | SubgraphCPUBenchmarkRequest:
        """Benchmark request for async autotuning. Pre-compiled when pipeline_max_autotune_gemm is enabled."""
        if self._bmreq is None:
            raise AssertionError(
                "bmreq accessed but pipeline_max_autotune_gemm was not enabled during __init__"
            )
        return self._bmreq

    def _ensure_compiled(self) -> None:
        """Ensure the module is compiled. Used for lazy compilation in non-async path."""
        if self._compiled_module is None:
            self._compiled_module = self._compile_for_benchmarking()

    def benchmark(self, *args: list[Any], out: torch.Tensor) -> float:
        """Regular benchmarking: compile if needed, then use benchmarker."""
        self._ensure_compiled()
        bm_func = self._compiled_module.call
        sym_inputs = self.sym_input_values

        def fn() -> Any:
            return bm_func([*sym_inputs, *args])

        if self._benchmark_with_cudagraphs:
            return benchmarker.benchmark_gpu_with_cuda_graph(fn)

        if config.profile_bandwidth_with_do_bench_using_profiling:
            return do_bench_using_profiling(fn)
        return benchmarker.benchmark(
            fn,
            device=benchmarker.infer_device(*sym_inputs, *args),
        )

    def benchmark_collective(self, *args: list[Any], out: torch.Tensor) -> None:
        """Run once for collective benchmarking (barrier sync handled by caller)."""
        self._ensure_compiled()
        self._compiled_module.call([*self.sym_input_values, *args])

    def hash_key(self) -> str:
        if self.gm is None:
            raise AssertionError("expected self.gm to be set")
        return "-".join(
            [
                self.name.rsplit("_", 1)[0],
                *[str(inp.get_size()) for inp in self.input_nodes],
                *[str(inp.get_stride()) for inp in self.input_nodes],
                str(self.gm.graph),
            ]
        )

    def output_node(self) -> ir.TensorBox:
        if self.gm is None:
            raise AssertionError("expected self.gm to be set")
        return ir.TensorBox.create(
            ir.SubgraphBuffer(
                layout=self.layout,
                input_nodes=self.input_nodes,
                gm=self.gm,
                example_inputs=self.example_inputs,
                subgraph_name=self.name,
                config_patches=self.config_patches if self.config_patches else None,
            )
        )

    def info_dict(self) -> dict[str, Any]:
        """Information returned here is logged to the autotune log file when that is enabled."""
        return {
            "backend": "subgraph",
            "kernel_name": self.name,
        }

    def autoheuristic_id(self) -> str:
        return f"subgraph_{self.name}"


class SubgraphTemplate(KernelTemplate):
    """
    A template for subgraph evaluation to be used in autotuning.

    This class allows creating customized subgraphs that can be appended
    as choices during the autotuning process, enabling the selection of
    optimal implementations for complex operations.
    """

    index_counter = itertools.count()

    def __init__(
        self,
        name: str,
    ):
        """
        Initialize a subgraph template.

        Args:
            name: The name of this template
            graph: The FX graph
        """
        super().__init__(name=name)

    def generate(  # type: ignore[override]
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        make_fx_graph: Callable[..., Any],
        description: str = "",
        input_gen_fns: dict[int, Callable[[Any], torch.Tensor]] | None = None,
        **kwargs: Any,
    ) -> SubgraphChoiceCaller:
        """
        Generate a SubgraphChoiceCaller instance for autotuning.

        Args:
            name: The name for this subgraph choice
            input_nodes: List of input nodes to the subgraph
            layout: Memory layout information for the output
            make_fx_graph: Callable that creates the FX graph for this subgraph
            description: Optional description of this choice
            input_gen_fns: Optional dict mapping input indices to tensor generators
            **kwargs: Additional keyword arguments

        Returns:
            SubgraphChoiceCaller: A callable object that can be used for autotuning
        """

        return SubgraphChoiceCaller(
            name=f"{name}_{next(SubgraphTemplate.index_counter)}",
            input_nodes=input_nodes,
            layout=layout,
            description=description,
            make_fx_graph=make_fx_graph,
            input_gen_fns=input_gen_fns,
        )

    def generate_custom_op_choices(
        self,
        name: str,
        decompositions: list[Callable[..., Any]],
        input_nodes: list[Buffer],
        non_tensor_args: list[dict[str, Any]],
        default_impl: Callable[..., Any] | None = None,
        input_gen_fns: dict[int, Callable[[Any], torch.Tensor]] | None = None,
        config_patches_list: list[dict[str, Any]] | None = None,
    ) -> list[SubgraphChoiceCaller]:
        """
        Generate multiple SubgraphChoiceCaller instances for custom op autotuning.

        This method extends SubgraphTemplate to support custom op decompositions,
        allowing multiple implementations to compete in autotuning.

        Args:
            name: Base name for the choices
            decompositions: List of decomposition functions to compete in autotuning
            input_nodes: List of tensor inputs. All tensor arguments must be passed here.
            non_tensor_args: List of non-tensor kwargs only, one dict per corresponding decomposition.
            default_impl: Default implementation for layout inference
            input_gen_fns: Optional dict mapping input indices to tensor generators
            config_patches_list: Optional list of config patches per decomposition

        Returns:
            List of SubgraphChoiceCaller instances for autotuning
        """
        if not decompositions:
            return []

        if len(decompositions) != len(non_tensor_args):
            raise AssertionError(
                f"decompositions and non_tensor_args must have same length, "
                f"got {len(decompositions)} decompositions and {len(non_tensor_args)} kwargs"
            )

        # Default to empty config_patches if not provided
        if config_patches_list is None:
            config_patches_list = [{} for _ in decompositions]

        # Infer layouts and ensure layout consistency for fair autotuning comparison
        layouts = [
            self._infer_custom_op_layout(
                input_nodes, decomp, kwargs, default_impl, input_gen_fns
            )
            for decomp, kwargs in zip(decompositions, non_tensor_args)
        ]

        # Validate all decompositions produce equivalent layouts for fair comparison
        self._validate_layout_equivalence(name, decompositions, layouts)
        layout = layouts[0]  # All layouts are now validated to be equivalent

        choices: list[SubgraphChoiceCaller] = []
        for decomp, decomp_kwargs, config_patches in zip(
            decompositions, non_tensor_args, config_patches_list
        ):
            # Create make_fx_graph function for this decomposition
            # Uses error_on_new_guards to detect impls that add guards
            from torch.fx.experimental.symbolic_shapes import _ShapeEnvGuardError

            def make_fx_graph(
                *args: Any,
                decomp: Callable[..., Any] = decomp,
                decomp_kwargs: dict[str, Any] = decomp_kwargs,
            ) -> Any:
                # decomp_kwargs contains all merged parameters: CustomOpConfig params + runtime kwargs

                from torch.fx.experimental.proxy_tensor import make_fx

                from ..decomposition import select_decomp_table

                decomposition_table = select_decomp_table()
                shape_env = V.fake_mode.shape_env

                # Use error_on_new_guards to detect impls that add guards during tracing
                guard_ctx = (
                    shape_env.error_on_new_guards()
                    if shape_env is not None
                    else contextlib.nullcontext()
                )
                with guard_ctx:
                    return make_fx(
                        functools.partial(decomp, **decomp_kwargs),
                        decomposition_table=decomposition_table,
                        tracing_mode="symbolic",
                    )(*args)

            # Generate descriptive name for this variant
            variant_name = self._generate_variant_name(decomp, decomp_kwargs)

            # Try to create choice; skip if it adds guards
            try:
                choice = self.generate(
                    name=f"{name}_{variant_name}",
                    input_nodes=input_nodes,
                    layout=layout,
                    make_fx_graph=make_fx_graph,
                    description=f"CustomOp {decomp.__name__}",
                    input_gen_fns=input_gen_fns,
                )
            except _ShapeEnvGuardError:
                log.info(
                    "Skipping decomposition %s: adds guards during tracing",
                    decomp.__name__,
                )
                counters["inductor"]["custom_op_decomp_guard_skips"] += 1
                continue

            # Cache decomposition info for range-based dispatch
            choice.cache_decomposition(decomp, decomp_kwargs)
            # Store config_patches for this choice
            choice.config_patches = config_patches
            choices.append(choice)

        return choices

    def _generate_variant_name(
        self, decomp: Callable[..., Any], kwargs: dict[str, Any]
    ) -> str:
        """Generate a descriptive name for a decomposition variant with its parameters."""
        import re

        base_name = decomp.__name__
        if not kwargs:
            return base_name

        def sanitize_value(v: Any) -> str:
            """Convert a value to a valid Python identifier component."""
            s = str(v)
            # Replace invalid characters with underscores
            s = re.sub(r"[^a-zA-Z0-9_]", "_", s)
            # Ensure it doesn't start with a digit
            if s and s[0].isdigit():
                s = "_" + s
            return s

        param_suffix = "_".join(
            f"{k}_{sanitize_value(v)}" for k, v in sorted(kwargs.items())
        )
        return f"{base_name}_{param_suffix}"

    def _validate_non_tensor_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Validate that kwargs contains only non-tensor arguments."""
        for key, value in kwargs.items():
            if isinstance(value, (torch.Tensor, Buffer)):
                raise AssertionError(
                    f"kwargs['{key}'] contains tensor {type(value)}. "
                    f"Tensor arguments should be in input_nodes, not kwargs. "
                    f"Only scalar/non-tensor parameters should be in kwargs."
                )

    def _validate_layout_equivalence(
        self,
        op_name: str,
        decompositions: list[Callable[..., Any]],
        layouts: list[Layout],
    ) -> None:
        """Ensure all layouts have consistent stride, device, dtype, and sizes for fair autotuning."""
        if not layouts:
            return

        reference = layouts[0]
        for i, layout in enumerate(layouts[1:], start=1):
            if (layout.device, layout.dtype, layout.size, layout.stride) != (
                reference.device,
                reference.dtype,
                reference.size,
                reference.stride,
            ):
                raise AssertionError(
                    f"Layout mismatch in custom op '{op_name}': "
                    f"decomposition '{decompositions[i].__name__}' produces "
                    f"({layout.device}, {layout.dtype}, {layout.size}, {layout.stride}) "
                    f"but '{decompositions[0].__name__}' produces "
                    f"({reference.device}, {reference.dtype}, {reference.size}, {reference.stride})"
                )

    def _infer_custom_op_layout(
        self,
        input_nodes: list[Buffer],
        function_decomposition: Callable[..., Any],
        kwargs: dict[str, Any],
        default_impl: Callable[..., Any] | None = None,
        input_gen_fns: dict[int, Callable[[Any], torch.Tensor]] | None = None,
    ) -> Layout:
        """Infer output layout for custom ops using the default implementation when available.
        Note that the Subgraph assumes custom ops return exactly one tensor output.
        TODO: Add support for multiple output custom ops.
        """
        import functools

        from torch._inductor.virtualized import V

        # Assert kwargs contain only non-tensor arguments
        self._validate_non_tensor_kwargs(kwargs)

        with V.fake_mode:
            example_inputs = []
            for i, inp in enumerate(input_nodes):
                if input_gen_fns and i in input_gen_fns:
                    fake_tensor = input_gen_fns[i](inp)
                else:
                    raw_shape = inp.get_size()
                    concrete_shape = V.graph.sizevars.optimization_hints(raw_shape)
                    raw_stride = inp.get_stride()
                    concrete_stride = V.graph.sizevars.optimization_hints(raw_stride)
                    fake_tensor = torch.empty_strided(
                        concrete_shape,
                        concrete_stride,
                        dtype=inp.get_dtype(),
                        device=inp.get_device(),
                    )
                example_inputs.append(fake_tensor)

            fn = functools.partial(function_decomposition, **kwargs)
            output = fn(*example_inputs)

            # Assert single output
            if not isinstance(output, torch.Tensor):
                raise AssertionError(
                    f"Expected single tensor output, got {type(output)}. "
                    f"Multi-output custom ops not yet supported in autotuning."
                )

            return FixedLayout(
                device=output.device,
                dtype=output.dtype,
                size=output.shape,
                stride=output.stride(),
            )
<|endoftext|># mypy: allow-untyped-defs
import functools

import sympy

from torch._inductor import config
from torch._inductor.codegen.simd import IterationRangesRoot, prefix_is_reduction
from torch._inductor.codegen.triton import (
    triton_compute_type,
    TritonCSEVariable,
    TritonKernel,
)
from torch._inductor.runtime.triton_heuristics import SplitScanGrid
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import CeilDiv

from ..utils import sympy_product, upcast_compute_type


class TritonSplitScanKernel(TritonKernel):
    """Generates a triton kernel that supports ops.scan calls while also splitting
    the reduction dimension over multiple triton programs.

    For this kernel, loop numels will always take the form ``(xdim, rdim)``
    and the grid has the shape ``(CeilDiv(rdim, RBLOCK), xdim)``. Communication
    between blocks occurs within a global memory workspace buffer, which
    must be zero-filled before launching the kernel.

    Note that generation for ``ops.reduction`` is not supported.

    For details of the communication strategy, see
    https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back

    """

    def __init__(
        self,
        tiling: dict[str, sympy.Expr],
        pid_cache=None,
        fixed_config=None,
        **kwargs,
    ) -> None:
        if pid_cache is not None:
            raise AssertionError("not supported")
        if fixed_config is not None:
            raise AssertionError("not supported")
        super().__init__(
            tiling,
            **kwargs,
        )
        self.no_x_dim = True

    def should_use_persistent_reduction(self) -> bool:
        return False

    def should_use_cooperative_reduction(self) -> bool:
        return False

    def initialize_range_tree(self, pid_cache):
        prefixes = ["y", "x", "r0_"]
        if len(self.numels) > len(prefixes):
            raise AssertionError("z dimension not supported for split scan")
        active_prefixes = prefixes[len(prefixes) - len(self.numels) :]

        grid_dims = {"r0_": 0, "x": 1, "y": 2}
        for prefix in active_prefixes:
            numel = self.numels[prefix]
            tensor_dim = 0 if prefix_is_reduction(prefix) else None
            grid_dim = grid_dims[prefix]
            self.range_trees.append(
                IterationRangesRoot(
                    f"{prefix}index",
                    numel,
                    prefix,
                    grid_dim,
                    self,  # type: ignore[arg-type]
                    pid_cache=pid_cache,
                    is_loop=False,
                    tensor_dim=tensor_dim,
                    grid_dim=grid_dim,
                    has_zdim=False,
                )
            )

    def reduction(self, dtype, src_dtype, reduction_type, value):
        raise NotImplementedError("NYI TritonSplitDimKernel reductions")

    def scan(self, dtypes, combine_fn, values):
        """
        Perform an associative scan on 'values'.
        """
        import triton.language as tl

        (dtype,) = dtypes
        (value,) = values

        dtype = upcast_compute_type(dtype)
        compute_type = triton_compute_type(dtype)
        compute_type_triton = getattr(tl, compute_type[3:])

        element_nbits = compute_type_triton.primitive_bitwidth

        scratch_type = "tl.uint32" if element_nbits <= 16 else "tl.uint64"
        scratch_type_triton = getattr(tl, scratch_type[3:])
        scratch_elems_per_block = 3 if element_nbits == 64 else 1
        scratch_nbytes_per_block = scratch_elems_per_block * (
            scratch_type_triton.primitive_bitwidth // 8
        )

        cse_compute = functools.partial(self.cse.generate, self.compute)

        if len(self.numels) != 2:
            raise AssertionError("Unexpected tiling")
        min_rblock = config.triton.min_split_scan_rblock
        reduction_numel = sympy_product(
            numel
            for prefix, numel in self.numels.items()
            if prefix_is_reduction(prefix)
        )
        pointwise_numel = sympy_product(
            numel
            for prefix, numel in self.numels.items()
            if not prefix_is_reduction(prefix)
        )
        max_blocks = pointwise_numel * CeilDiv(reduction_numel, min_rblock)
        nbytes = scratch_nbytes_per_block * max_blocks
        scratch_base: str | TritonCSEVariable
        scratch_base, _, offset = self.args.workspace(nelem=nbytes, zero_fill=True)
        if offset != 0:
            scratch_base = f"({scratch_base} + {self.index_to_str(offset)})"
        runtime_rblocks = f"tl.num_programs({self.range_trees[-1].index})"
        scratch_base = (
            f"{scratch_base}.to(tl.pointer_type({scratch_type})) + xoffset * "
            f"{scratch_elems_per_block} * {runtime_rblocks}"
        )

        masks = OrderedSet(f"{tree.prefix}mask" for tree in self.range_trees)
        self.filter_masks(masks)
        if self._load_mask:
            raise AssertionError("ops.scan not supported inside ops.masked")

        value = cse_compute(
            f"{value}.to({compute_type})",
            dtype=dtype,
            shape=value.shape,
        )
        value = cse_compute(
            f"tl.broadcast_to({value}, {self.dense_size_str()})",
            dtype=dtype,
            shape=self.dense_size_list(),
        )

        combine_helper_fn = self._lift_helper(combine_fn, (value,), (dtype,))
        dim = self.triton_tensor_ndim() - 1
        if dim != 0:
            raise AssertionError(f"expected scan dim == 0, got {dim}")
        scan_shape = value.shape
        if scan_shape is None:
            raise AssertionError("expected value.shape to be set")
        reduced_shape = list(scan_shape)
        del reduced_shape[dim]

        block_sum = cse_compute(
            f"tl.reduce({value}, {dim}, {combine_helper_fn})",
            dtype=dtype,
            shape=reduced_shape,
        )
        exclusive_prefix = self.cse.newvar(
            dtype=dtype,
            shape=reduced_shape,
        )
        if element_nbits == 64:
            self.compute.splice(
                f"""
                {exclusive_prefix} = triton_helpers.exclusive_scan_decoupled_lookback_64(
                    {scratch_base},
                    {block_sum},
                    {self.iteration_ranges_get_pid(self.range_trees[-1])},
                    {combine_helper_fn},
                )
                """,
                strip=True,
            )

        else:
            if element_nbits > 32:
                raise AssertionError(
                    f"expected element_nbits <= 32, got {element_nbits}"
                )
            value_as_uint_dtype = f"tl.uint{element_nbits}"

            self.compute.splice(
                f"""
                {exclusive_prefix} = triton_helpers.exclusive_scan_decoupled_lookback(
                    {scratch_base},
                    {block_sum},
                    {self.iteration_ranges_get_pid(self.range_trees[-1])},
                    {combine_helper_fn},
                    DTYPE_VALUE_AS_UINT={value_as_uint_dtype},
                    DTYPE_PACK={scratch_type},
                )
                """,
                strip=True,
            )
        # Compute final cumsum
        block_scan = cse_compute(
            f"tl.associative_scan({value}, {dim}, {combine_helper_fn})",
            dtype=dtype,
            shape=scan_shape,
        )
        combined_result = cse_compute(
            f"{combine_helper_fn}({exclusive_prefix}, {block_scan})",
            dtype=dtype,
            shape=scan_shape,
        )
        return (
            cse_compute(
                f"tl.where(roffset == 0, {block_scan}, {combined_result})",
                dtype=dtype,
                shape=scan_shape,
            ),
        )

    def _get_heuristic(self):
        return "split_scan"

    def _get_grid_type(self) -> type[SplitScanGrid]:
        return SplitScanGrid
<|endoftext|>from __future__ import annotations

import collections
import dataclasses
import functools
import itertools
import typing
from typing import Any

import sympy

import torch

from ...utils._ordered_set import OrderedSet
from ...utils._sympy.functions import FloorDiv, Min, ModularIndexing
from ...utils._sympy.symbol import make_symbol, SymT
from .. import ir
from ..dependencies import Dep, extract_loop_body_with_args, MemoryDep
from ..runtime.hints import ReductionHint
from ..runtime.runtime_utils import next_power_of_2
from ..scheduler import SchedulerNode
from ..utils import cache_on_self
from ..virtualized import V


if typing.TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from torch._inductor.tiling_utils import CoalesceVarAnalysis


_INNER_REDUCTION_RATIO = 32
_SMALL_INNER_REDUCTION_RATIO = 16
_SMALL_INNER_REDUCTION_MAX_RBLOCK = 512


def tiling_scores_suggest_inner_reduction(
    tiling_scores: dict[str, sympy.Expr], reduction_numel: sympy.Expr
) -> bool:
    """Return whether tiling scores justify treating a reduction as inner."""
    sizevars = V.graph.sizevars
    non_reduction_score = sum(
        sizevars.optimization_hint(tiling_scores.get(dim, 0), fallback=0)
        for dim in ("x", "y", "z")
    )
    non_reduction_score = max(non_reduction_score, 1)
    r_score = sizevars.optimization_hint(tiling_scores["r0_"], fallback=0)
    if r_score >= _INNER_REDUCTION_RATIO * non_reduction_score:
        return True
    if r_score < _SMALL_INNER_REDUCTION_RATIO * non_reduction_score:
        return False

    # Moderate score ratios are useful while persistent configs can keep XBLOCK=8.
    rblock_hint = next_power_of_2(
        max(sizevars.optimization_hint(reduction_numel, fallback=1), 1)
    )
    return rblock_hint <= _SMALL_INNER_REDUCTION_MAX_RBLOCK


class NodeScheduleMarker:
    @staticmethod
    def only_nodes(it: Iterable[NodeScheduleEntry]) -> Iterable[SchedulerNode]:
        for item in it:
            if not (item is DisableReduction or item is EnableReduction):
                yield item  # type: ignore[misc]

    @staticmethod
    def is_reduction() -> bool:
        return False


NodeScheduleEntry = SchedulerNode | type[NodeScheduleMarker]


class DisableReduction(NodeScheduleMarker):
    """
    Marker to invoke `kernel.disable_reduction()`.  This closes a
    reduction loop and allows for pointwise ops to occur on the output
    of a reduction.
    """


class EnableReduction(NodeScheduleMarker):
    """
    Marker to end a DisableReduction block.
    """

    @staticmethod
    def filter(node_schedule: list[NodeScheduleEntry]) -> Iterable[SchedulerNode]:
        """
        Get the nodes from node_schedule skipping those in a
        DisableReduction block.
        """
        disabled = False
        for node in node_schedule:
            if node in (EnableReduction, DisableReduction):
                # Don't tile stuff outside the main reduction loop
                disabled = node is DisableReduction
            elif disabled:
                pass
            else:
                yield node  # type: ignore[misc]


class SIMDKernelFeatures:
    """
    An ordered schedule of nodes that will become a single kernel.
    """

    def __init__(
        self,
        node_schedule: list[NodeScheduleEntry],
        numel: sympy.Expr,
        reduction_numel: sympy.Expr = sympy.S.One,
        coalesce_analysis: CoalesceVarAnalysis | None = None,
        tiling_scores: dict[str, sympy.Expr] | None = None,
    ):
        self.node_schedule = node_schedule
        # numel excludes reduction_numel
        self.numel: sympy.Expr = V.graph.sizevars.simplify(numel)
        self.reduction_numel: sympy.Expr = V.graph.sizevars.simplify(reduction_numel)
        self._stats_cache: dict[tuple[sympy.Expr, ...], MemoryStats] = {}
        self.coalesce_analysis = coalesce_analysis
        self.tiling_scores = tiling_scores

    def with_tiling_scores(
        self, tiling_scores: dict[str, sympy.Expr] | None
    ) -> SIMDKernelFeatures:
        return SIMDKernelFeatures(
            self.node_schedule,
            self.numel,
            self.reduction_numel,
            self.coalesce_analysis,
            tiling_scores,
        )

    @cache_on_self
    def is_reduction(self) -> bool:
        return self.reduction_numel != 1

    @cache_on_self
    def scheduler_nodes(self) -> Iterable[SchedulerNode]:
        return tuple(NodeScheduleMarker.only_nodes(self.node_schedule))

    def reduction_nodes(self) -> list[SchedulerNode]:
        return [n for n in self.scheduler_nodes() if n.is_reduction()]

    @cache_on_self
    def strict_sum_reductions(self) -> tuple[ir.Reduction, ...]:
        return tuple(
            node.node.data
            for node in self.reduction_nodes()
            if isinstance(node.node, ir.ComputedBuffer)
            and isinstance(node.node.data, ir.Reduction)
            and node.node.data.strict_sum_rblock is not None
        )

    def has_strict_sum_multirow_reduction(self) -> bool:
        return any(r.strict_sum_multirow for r in self.strict_sum_reductions())

    @cache_on_self
    def strict_sum_rblock(self) -> int | None:
        rblocks = OrderedSet(
            reduction.strict_sum_rblock for reduction in self.strict_sum_reductions()
        )
        if not rblocks:
            return None
        if len(rblocks) != 1:
            raise AssertionError(
                f"strict sums require one reduction block size, got {rblocks}"
            )
        return next(iter(rblocks))

    @cache_on_self
    def buf_accesses(self) -> dict[str, list[Dep]]:
        """only needed for config.benchmark_kernel"""
        buf_accesses = collections.defaultdict(list)
        for node in self.scheduler_nodes():
            for access in node.read_writes.reads | node.read_writes.writes:
                buf_accesses[access.name].append(access)
        return buf_accesses

    @cache_on_self
    def op_counts(self) -> collections.Counter[str]:
        counts: collections.Counter[str] = collections.Counter()
        for node in self.scheduler_nodes():
            counts.update(node._body.op_counts)
        return counts

    def contains_op(self, op_name: str) -> bool:
        """True if V.ops.{op_name} is used in node_schedule"""
        return bool(self.op_counts().get(op_name))

    def get_mutations(self) -> OrderedSet[str]:
        mutations: OrderedSet[str] = OrderedSet()
        for node in self.scheduler_nodes():
            for buf in node.get_outputs():
                mutations.update(buf.get_mutations())
        return mutations

    @cache_on_self
    def select_index_dtype(self) -> torch.dtype:
        # Gather all used buffer names
        buffer_names: OrderedSet[str] = OrderedSet()
        for node in self.scheduler_nodes():
            buffer_names.update(node.get_buffer_names())
            buffer_names.update(node.used_buffer_names())
        buffers = [V.graph.get_buffer(name) for name in buffer_names]

        # In theory we can separately check xnumel and rnumel are <= int_max
        # but some indexers do use the full linear index so we need to be
        # conservative here.
        total_numel = self.numel * self.reduction_numel

        from .simd import SIMDScheduling

        if SIMDScheduling.can_use_32bit_indexing(total_numel, buffers):
            # Fused address expressions can overflow int32 even when
            # numel/storage_size are within range, so check them directly.
            if self.any_index_expr_overflows_int32():
                return torch.int64
            return torch.int32
        return torch.int64

    @cache_on_self
    def any_index_expr_overflows_int32(self) -> bool:
        """Return True if any MemoryDep addressing expression can overflow int32.

        Two complementary checks are applied to each index:

        - Constant offset: the index evaluated at the origin (every loop
          variable set to 0) must lie within [-2**31, 2**31 - 1]. This is cheap
          and independent of loop-var ranges, so it also covers deps whose sizes
          are symbolic.
        - Variable-scaled bound: the index bounded over the actual iteration
          range of every loop variable must not exceed 2**31 - 1. This catches
          overflow such as the `V * x0` synthesized when unrolled chunked-loop
          slices are stitched into one pointwise kernel (`V` a per-chunk stride,
          `x0` spanning the fused numel); unlike a `numel * max_stride` proxy it
          is exact, since a stride on a size-1 dim multiplies an empty range and
          drops out. Deps with a symbolic loop-var size, or whose bound is
          unknown/indirect (infinite), are skipped here and left to the
          constant-offset and per-buffer storage-size checks.
        """
        from torch.utils._sympy.numbers import int_oo
        from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges

        int32_max = sympy.Integer(2**31 - 1)
        int32_min = sympy.Integer(-(2**31))
        for node in self.scheduler_nodes():
            for dep in itertools.chain(node.read_writes.reads, node.read_writes.writes):
                if not isinstance(dep, MemoryDep):
                    continue
                index = dep.index
                if not isinstance(index, sympy.Expr):
                    continue

                # Constant offset (index at the origin).
                try:
                    const_part = index.subs(
                        {s: sympy.Integer(0) for s in index.free_symbols}
                    )
                    if isinstance(const_part, sympy.Expr) and (
                        const_part > int32_max or const_part < int32_min
                    ):
                        return True
                except (ZeroDivisionError, TypeError, ValueError):
                    pass

                # Variable-scaled bound over concrete loop-var ranges. A
                # ValueRanges bound must be concrete, so skip a dep with any
                # symbolic loop-var size.
                sizes = dep.size
                if any(getattr(s, "free_symbols", None) for s in sizes):
                    continue
                try:
                    var_ranges = {
                        var: ValueRanges(0, max(int(size) - 1, 0))
                        for var, size in zip(dep.var_names, sizes)
                    }
                    upper = bound_sympy(index, var_ranges).upper
                    if (
                        isinstance(upper, sympy.Expr)
                        and not upper.has(int_oo, sympy.oo)
                        and upper > int32_max
                    ):
                        return True
                except (ZeroDivisionError, TypeError, ValueError):
                    continue
        return False

    def get_reduction_hint(
        self, tiling_scores: dict[str, sympy.Expr] | None = None
    ) -> ReductionHint:
        if tiling_scores is None:
            tiling_scores = self.tiling_scores
        reductions = self.reduction_nodes()
        if len(reductions) > 0:
            hints = [self.reduction_hint(n) for n in reductions]
            if hints.count(hints[0]) == len(hints):
                reduction_hint_val = hints[0]
            else:
                reduction_hint_val = ReductionHint.DEFAULT

            if (
                reduction_hint_val == ReductionHint.INNER
                and self.has_non_contiguous_pw_in_reduction_kernel()
            ):
                reduction_hint_val = ReductionHint.DEFAULT

            # Upgrade DEFAULT to INNER for inner reductions based on tiling scores
            if (
                reduction_hint_val == ReductionHint.DEFAULT
                and tiling_scores is not None
                and "x" in tiling_scores
                and "r0_" in tiling_scores
            ):
                if tiling_scores_suggest_inner_reduction(
                    tiling_scores, self.reduction_numel
                ):
                    reduction_hint_val = ReductionHint.INNER
        else:
            reduction_hint_val = ReductionHint.DEFAULT
        return reduction_hint_val

    @cache_on_self
    def buffer_read_counts(self) -> dict[str, int]:
        """Counts how many times each buffer is read within the kernel"""
        read_counts: dict[str, int] = collections.defaultdict(int)

        for node in self.scheduler_nodes():
            # node.read_writes.reads contains MemoryDep objects for each read
            for read_dep in node.read_writes.reads:
                read_counts[read_dep.name] += 1

        return dict(read_counts)  # Convert defaultdict to regular dict

    def has_non_contiguous_pw_in_reduction_kernel(self) -> bool:
        pointwise_nodes = [
            n
            for n in self.scheduler_nodes()
            if not n.is_reduction()
            and n.group[1][0] == self.numel * self.reduction_numel
        ]
        for node in pointwise_nodes:
            # An index can be an integer when loading a random seed.
            if not all(
                not isinstance(dep, MemoryDep)
                or dep.is_contiguous()
                or isinstance(dep.index, (sympy.Integer, int))
                or dep.stride1_for_last_dim()
                for dep in itertools.chain(
                    node.read_writes.reads, node.read_writes.writes
                )
            ):
                return True
        return False

    @staticmethod
    def reduction_hint(node: Any) -> ReductionHint:
        if not node.is_reduction():
            raise AssertionError("expected node to be a reduction")
        if node.node.data.reduction_hint != ReductionHint.INNER and all(
            dep.is_contiguous()
            for dep in itertools.chain(node.read_writes.reads, node.read_writes.writes)
        ):
            return ReductionHint.INNER
        else:
            return node.node.data.reduction_hint

    def memory_stats(
        self, groups_dict: dict[str, sympy.Expr] | None = None
    ) -> MemoryStats:
        """Analysis to generate features that can be used in heuristics"""
        if groups_dict is None:
            groups = (self.numel, self.reduction_numel)
        elif groups_dict.keys() == OrderedSet(["x", "r0_"]):
            groups = (groups_dict["x"], groups_dict["r0_"])
        else:
            raise NotImplementedError(f"groups_dict={groups_dict!r}")
        result = self._stats_cache.get(groups)
        if result is None:
            self._stats_cache[groups] = result = MemoryStats.compute(
                MemoryEstimator(self, groups)
            )
        return result


class MemoryEstimator:
    """
    Estimate various properties of the kernel for use in heuristics.
    We simulate the memory effects of CSE/buffer elimination in codegen.
    """

    kernel_sizes: tuple[sympy.Expr, ...]
    outside_loop: MemoryEstimate
    loops: list[MemoryEstimate]
    persistent: MemoryEstimate
    symbols: list[sympy.Symbol]

    def __init__(self, features: SIMDKernelFeatures, groups: Sequence[sympy.Expr]):
        self.features = features
        self.inside_reduction = features.is_reduction()
        self.store_buffer_names: OrderedSet[str] = OrderedSet()
        self.must_keep_buffers: OrderedSet[str] = OrderedSet()
        self.num_reductions_dims = 1
        self.groups = groups
        self.symbols = [make_symbol(SymT.INDEX, i) for i in range(len(groups))]
        # We are doing two estimates simultaneously:
        # 1) the first is a for a non-persistent (aka looped) reduction, using self.outside_loop/self.loops
        # we add an item to loops each corresponding to each reduction loop in the kernel
        # outside_loop is only used for broadcasting or point-wise ops that don't use the reduction dimension
        # 2) the second is for a persistent kernel, using self.persistent
        # persistent kernels don't have loops, so we only have one MemoryEstimate()
