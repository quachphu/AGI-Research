        swizzle_type_b: Any | None = None,
        epilogue_fn_code: str | None = None,
        epilogue_reads: list[str] | None = None,
        epilogue_writes: list[str] | None = None,
        epilogue_var_renames: dict[str, Any] | None = None,
        local_reduce: GemmReductionPlan | None = None,
        swap_ab: bool = False,
        bias_node: Buffer | None = None,
    ) -> None:
        super().__init__()
        self.kernel_name = kernel_name
        self.input_nodes = input_nodes
        self.output_node = output_node
        self.kernel_metadata = kernel_metadata
        self.accumulator_type = accumulator_type
        self.workspace_size = workspace_size
        self.variant = variant
        self.scale_type_a = scale_type_a
        self.scale_type_b = scale_type_b
        self.swizzle_type_a = swizzle_type_a
        self.swizzle_type_b = swizzle_type_b
        self.epilogue_fn_code = epilogue_fn_code
        self.epilogue_reads = epilogue_reads or []
        self.epilogue_writes = epilogue_writes or []
        self.epilogue_var_renames = epilogue_var_renames or {}
        self.local_reduce = local_reduce
        self.swap_ab = swap_ab

        # An addmm bias baked into the choice becomes a bias-add epilogue. With
        # no scheduler-fused epilogue it's a standalone bias-add; when the
        # scheduler also fuses pointwise ops, compose the bias-add into them so
        # the fused epilogue sees A@B + bias (matches Triton's epilogue fusion).
        if bias_node is not None:
            if not self.epilogue_fn_code:
                (
                    self.epilogue_fn_code,
                    self.epilogue_reads,
                    self.epilogue_writes,
                    self.epilogue_var_renames,
                ) = _build_bias_epilogue(bias_node.get_name(), output_node.get_name())
            else:
                (
                    self.epilogue_fn_code,
                    self.epilogue_reads,
                    self.epilogue_var_renames,
                ) = _compose_bias_into_epilogue(
                    self.epilogue_fn_code,
                    self.epilogue_reads,
                    self.epilogue_var_renames,
                    bias_node.get_name(),
                )

        self._template_input_args: list[tuple[str, Buffer]] = []

        for i, input_node in enumerate(input_nodes):
            param_name = f"in_ptr{i}"
            self._template_input_args.append((param_name, input_node))

    def render(self) -> str:
        """Render the Python source for the NVGEMM kernel wrapper."""
        kernel_name_str = self.kernel_metadata["kernel_name"]
        acc_dtype_str = CuteDSLOpOverrides.TORCH_TO_CUTE_DTYPE.get(
            self.accumulator_type, "cutlass.Float32"
        )

        input_tensor_names = [f"in_ptr{i}" for i, _ in enumerate(self.input_nodes)]
        output_buffers = self._ordered_output_buffers()
        input_params = list(input_tensor_names)
        input_params.extend(f"out_ptr{i}" for i in range(len(output_buffers)))
        input_params.extend(self.epilogue_reads)
        if self.workspace_size > 0:
            input_params.append("workspace")
        input_params.append("stream=None")
        params_str = ", ".join(input_params)

        if len(input_tensor_names) == 1:
            input_tensors_expr = f"({input_tensor_names[0]},)"
        else:
            input_tensors_expr = f"({', '.join(input_tensor_names)})"

        workspace_arg = "workspace" if self.workspace_size > 0 else "None"
        has_epilogue = bool(self.epilogue_fn_code) or self.local_reduce is not None

        # Build variant_kwargs dict expression for SCALED_GEMM
        variant_kwargs_expr = "None"
        variant_extra_imports = ""
        if self.variant.name == "SCALED_GEMM":
            sma, swza, smb, swzb = _get_scaled_gemm_modes(
                self.scale_type_a,
                self.swizzle_type_a,
                self.scale_type_b,
                self.swizzle_type_b,
            )
            variant_extra_imports = (
                "from cutlass.operators import ScaleMode, ScaleSwizzleMode"
            )
            variant_kwargs_expr = (
                "{"
                f'"scale_mode_a": ScaleMode.{sma.name}, '
                f'"swizzle_mode_a": ScaleSwizzleMode.{swza.name}, '
                f'"scale_mode_b": ScaleMode.{smb.name}, '
                f'"swizzle_mode_b": ScaleSwizzleMode.{swzb.name}'
                "}"
            )

        code = IndentedBuffer()

        # -- Module-level imports and constants --
        code.writeline("import cutlass")
        code.writeline(
            "from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_kernel import ("
        )
        with code.indent():
            code.writeline("_make_disk_config_key,")
            code.writeline("_nvgemm_run,")
            code.writeline("_nvgemm_precompile,")
        code.writeline(")")
        code.writeline("from torch._inductor.utils import _ensure_fp4_dtype_registered")
        code.writeline("_ensure_fp4_dtype_registered()")
        if variant_extra_imports:
            code.writeline(variant_extra_imports)
        if has_epilogue:
            code.writeline("from cutlass.operators.arguments import EpilogueArguments")
        code.writeline("")

        # -- Epilogue function definition (must be module-level for cutlass.operators) --
        epilogue_fn_code = self.epilogue_fn_code
        if has_epilogue and epilogue_fn_code is not None:
            epilogue_source_hash = hashlib.sha256(epilogue_fn_code.encode()).hexdigest()
            code.writeline(f'_EPILOGUE_FN_SOURCE = "{epilogue_source_hash}"')
            # Pass the epilogue to cutlass.operators as a SOURCE STRING, not a
            # callable: EpilogueArguments traces a string directly, whereas a
            # callable triggers inspect.getsource() at every kernel launch --
            # which fails ("could not get source code") for generated modules
            # loaded from the subprocess PyCodeCache.
            code.writeline(f"_EPILOGUE_FN_SRC = {epilogue_fn_code!r}")
            code.writeline("")

        # -- Module-level state --
        code.writeline(f"_KERNEL_NAME = {kernel_name_str!r}")
        disk_key_args = ", ".join(
            repr(str(v))
            for v in (
                kernel_name_str,
                self.variant.name,
                self.accumulator_type,
                self.scale_type_a,
                self.scale_type_b,
                self.swizzle_type_a,
                self.swizzle_type_b,
            )
        )
        code.writeline(
            f"_DISK_CACHE_CONFIG_KEY = _make_disk_config_key({disk_key_args})"
        )
        code.writeline("_compiled_cache = {}")
        code.writeline("_disk_fn_cache = {}")
        code.writeline(f"_VARIANT_KWARGS = {variant_kwargs_expr}")
        code.writeline(f"_ACC_TYPE = {acc_dtype_str}")
        input_param_names_repr = repr(input_tensor_names)
        code.writeline(f"_INPUT_PARAM_NAMES = {input_param_names_repr}")
        code.writeline("")

        # -- Main function --
        code.writeline(f"def {self.kernel_name}_main({params_str}):")
        with code.indent():
            # Build epilogue args if needed (user-specific variable names)
            epi_args_expr = "None"
            epi_source_expr = '""'
            aux_tensors: list[str] = []
            if self.epilogue_fn_code:
                epilogue_kwargs = self._render_epilogue_kwargs()
                epi_kwargs_str = "epilogue_fn=_EPILOGUE_FN_SRC"
                if epilogue_kwargs:
                    epi_kwargs_str += f", {epilogue_kwargs}"
                code.writeline(f"epi_args = EpilogueArguments({epi_kwargs_str})")
                epi_args_expr = "epi_args"
                epi_source_expr = "_EPILOGUE_FN_SOURCE"
                aux_tensors.extend(self.epilogue_reads)

            run_variant_kwargs = "_VARIANT_KWARGS"
            if self.local_reduce is not None:
                reduction = self.local_reduce
                feed_main = reduction.feeds_main
                reduce_ptr = (
                    "None"
                    if reduction.reduction_output is None
                    else f"out_ptr{output_buffers.index(reduction.reduction_output)}"
                )
                run_variant_kwargs = (
                    "_VARIANT_KWARGS | {"
                    f"'local_reduce_out': {reduce_ptr}, "
                    f"'local_reduce_group': {reduction.group}, "
                    f"'local_reduce_axis': {reduction.axis}, "
                    f"'local_reduce_type': {reduction.reduction_type!r}, "
                    f"'local_reduce_source': {reduction.source_type!r}"
                    f", 'local_reduce_feeds_main': {feed_main!r}"
                    "}"
                )
                if not feed_main:
                    aux_tensors.append(reduce_ptr)

            aux_tensors_expr = f"({', '.join(aux_tensors)},)" if aux_tensors else "()"

            code.writeline("_nvgemm_run(")
            with code.indent():
                code.writeline(f'"{self.variant.name}", _KERNEL_NAME,')
                code.writeline(f"{input_tensors_expr}, out_ptr0, _ACC_TYPE,")
                code.writeline(
                    "_compiled_cache, _disk_fn_cache, __file__, _DISK_CACHE_CONFIG_KEY,"
                )
                code.writeline(f"stream=stream, workspace={workspace_arg},")
                code.writeline(f"variant_kwargs={run_variant_kwargs},")
                code.writeline(f"epilogue_args={epi_args_expr},")
                code.writeline(f"epilogue_source={epi_source_expr},")
                code.writeline(f"has_epilogue={has_epilogue},")
                code.writeline(f"aux_tensors={aux_tensors_expr},")
                if self.swap_ab:
                    code.writeline("swap_ab=True,")
            code.writeline(")")

        # -- Precompile hook --
        code.writeline("")
        code.writeline(
            f"def {self.kernel_name}_precompile("
            "precompile_shapes, precompile_strides, precompile_dtypes, "
            "device_index=0, device_capability=None, **kwargs):"
        )
        with code.indent():
            code.writeline("_nvgemm_precompile(")
            with code.indent():
                code.writeline(
                    "precompile_shapes, precompile_strides, precompile_dtypes,"
                )
                code.writeline("device_index, device_capability,")
                code.writeline(f'variant_name="{self.variant.name}",')
                code.writeline("kernel_name=_KERNEL_NAME,")
                code.writeline("accumulator_type=_ACC_TYPE,")
                code.writeline(
                    "compiled_cache=_compiled_cache, disk_fn_cache=_disk_fn_cache,"
                )
                code.writeline(
                    "module_path=__file__, disk_config_key=_DISK_CACHE_CONFIG_KEY,"
                )
                code.writeline("input_param_names=_INPUT_PARAM_NAMES,")
                code.writeline("variant_kwargs=_VARIANT_KWARGS,")
                code.writeline("max_active_clusters=kwargs.get('max_active_clusters'),")
            code.writeline(")")

        return code.getvalue()

    def _ordered_output_buffers(self) -> list[str]:
        """Graph-output buffer names in out_ptr order.

        out_ptr0 is the primary GEMM output (the epilogue's `D` store, passed as
        the kernel's `out`). Additional stores -- a multi-store epilogue where the
        GEMM output feeds more than one graph output -- follow in epilogue_writes
        order as out_ptr1, out_ptr2, ...
        """
        if not self.epilogue_writes:
            primary_output = (
                self.local_reduce.primary_output
                if self.local_reduce is not None
                else self.output_node.get_name()
            )
            ordered = [primary_output]
            if self.local_reduce is not None:
                ordered.extend(self.local_reduce.auxiliary_outputs)
            return ordered
        d_buf = (self.epilogue_var_renames or {}).get("D")
        ordered: list[str] = []
        if d_buf is not None:
            ordered.append(d_buf)
        for w in self.epilogue_writes:
            if w != d_buf and w not in ordered:
                ordered.append(w)
        if self.local_reduce is not None:
            ordered.extend(
                output
                for output in self.local_reduce.auxiliary_outputs
                if output not in ordered
            )
        return ordered

    def _render_epilogue_kwargs(self) -> str:
        """Render kwargs for the EpilogueArguments constructor.

        Each epilogue variable maps to either an output pointer -- the `D` store
        and any additional multi-store outputs become out_ptr0, out_ptr1, ... in
        _ordered_output_buffers order -- or, for a read, the aux input buffer.
        `accum` is the kernel-supplied accumulator and is not a kwarg.
        """
        out_ptr_of = {
            buf: f"out_ptr{i}" for i, buf in enumerate(self._ordered_output_buffers())
        }
        kwargs_parts = []
        for var_name, buffer_name in self.epilogue_var_renames.items():
            if var_name == _ACCUMULATOR_ARG_NAME:
                continue
            if buffer_name in out_ptr_of:
                kwargs_parts.append(f"{var_name}={out_ptr_of[buffer_name]}")
            else:
                kwargs_parts.append(f"{var_name}={buffer_name}")
        return ", ".join(kwargs_parts)

    def _get_reinterpret_view(self, node) -> ReinterpretView | None:
        """Extract or convert to ReinterpretView from a node, handling all views."""
        while isinstance(node, MutableBox):
            node = node.data
        if isinstance(node, BaseView):
            return ExternKernel.convert_to_reinterpret_view(node)
        return None

    def call_kernel(self, name: str, node=None):
        """
        Generate the kernel call in the wrapper code.

        Similar to CuteDSLTemplateKernel.call_kernel but simplified for NVIDIA Universal GEMM.
        Includes epilogue input tensors when epilogue fusion is enabled.
        """
        wrapper = V.graph.wrapper_code

        if self.local_reduce is not None:
            primary_name = self.local_reduce.primary_output
            V.graph.removed_buffers.discard(primary_name)
            primary_output = V.graph.get_buffer(primary_name)
            wrapper.codegen_allocation(
                primary_output
                if isinstance(primary_output, Buffer)
                else self.output_node
            )

        call_args: list[str] = []
        arg_types: list[Any] = []
        raw_args: list[Buffer | ReinterpretView | None] = []
        raw_keys: list[str | None] = []

        for param_name, input_node in self._template_input_args:
            reinterpret_view = self._get_reinterpret_view(input_node)
            if reinterpret_view is not None:
                call_args.append(reinterpret_view.codegen_reference())
                # Pass the ReinterpretView as raw_arg so autotune_at_compile_time
                # can use it to generate example tensors
                raw_args.append(reinterpret_view)
            else:
                call_args.append(input_node.get_name())
                raw_args.append(input_node)
            arg_types.append(V.graph.get_dtype(input_node.get_name()))
            raw_keys.append(param_name)

        # The kernel writes the epilogue's output store(s), not the GEMM buffer
        # (which is removed via removed_buffers aliasing). out_ptr0 is the primary
        # (`D`) output; a multi-store epilogue adds out_ptr1, ... in order.
        for i, output_name in enumerate(self._ordered_output_buffers()):
            call_args.append(output_name)
            arg_types.append(V.graph.get_dtype(output_name))
            raw_args.append(None)  # Output buffer is findable by name
            raw_keys.append(f"out_ptr{i}")

        for read_name in self.epilogue_reads:
            call_args.append(read_name)
            arg_types.append(V.graph.get_dtype(read_name))
            buf = V.graph.get_buffer(read_name)
            if buf is None:
                # Epilogue may read model parameters/inputs (e.g. bias) that are
                # graph inputs rather than computed buffers.
                buf = V.graph.graph_inputs.get(read_name)
            # pyrefly: ignore [bad-argument-type]
            raw_args.append(buf)
            raw_keys.append(read_name)

        # Allocate workspace if needed
        ws: WorkspaceArg | None = None
        if self.workspace_size > 0:
            ws = WorkspaceArg(
                count=self.workspace_size,
                device=V.graph.get_current_device_or_throw(),
                zero_mode=WorkspaceZeroMode.UNINITIALIZED,
                outer_name=WorkspaceArg.unique_name(),
            )
            wrapper.generate_workspace_allocation(ws)
            call_args.append(ws.outer_name)
            arg_types.append(ws.dtype)
            raw_args.append(None)
            raw_keys.append(None)

        wrapper.generate_kernel_call(
            name,
            call_args,
            triton=True,
            arg_types=arg_types,
            raw_args=raw_args,
            raw_keys=raw_keys,
        )

        # Deallocate workspace after kernel call
        if ws is not None:
            wrapper.generate_workspace_deallocation(ws)
<|endoftext|>import operator

import torch
from torch._dynamo.utils import counters

from ..pattern_matcher import CallFunction, KeywordArg, Match, register_graph_pattern
from .pre_grad import apply_gumbel_max_trick_pass


@register_graph_pattern(
    CallFunction(
        torch.argmax,
        CallFunction(
            operator.truediv,
            # we don't rely on PatternMatcher to match softmax
            # and exponential_ due to the mutation op
            KeywordArg("softmax"),
            KeywordArg("rand_exp"),
        ),
        dim=-1,
        keepdim=True,
    ),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=apply_gumbel_max_trick_pass,
)
def apply_gumbel_max_trick(match: Match, softmax, rand_exp):
    if not torch._inductor.config.apply_gumbel_max_trick:
        return

    if (
        rand_exp.op != "call_method"
        or rand_exp.target != "exponential_"
        or len(rand_exp.users) != 1
    ):
        return

    empty_node, rate = rand_exp.args
    if rate != 1.0:
        return

    if len(empty_node.users) != 1:
        return

    if (
        softmax.op != "call_function"
        or softmax.target != torch.nn.functional.softmax
        or len(softmax.users) != 1
    ):
        return
    logits = softmax.args[0]

    truediv, argmax = match.nodes
    nodes_to_erase = [truediv, softmax]

    graph = match.graph

    with graph.inserting_before(argmax):
        log = graph.call_function(torch.log, (rand_exp,))
        gumbel_noise = graph.call_function(operator.neg, (log,))
        argmax_input = graph.call_function(operator.add, (logits, gumbel_noise))
        # pyrefly: ignore [missing-attribute]
        argmax.args[0].replace_all_uses_with(argmax_input)

    # erase nodes
    for n in nodes_to_erase:
        match.graph.erase_node(n)

    counters["inductor"]["apply_gumbel_max_trick"] += 1
<|endoftext|># mypy: allow-untyped-defs
import functools
import inspect
import logging

import torch
from torch.utils._ordered_set import OrderedSet

from ..._dynamo.utils import counters
from ..pattern_matcher import (
    filter_nodes,
    fwd_only,
    gen_register_replacement,
    joint_fwd_bwd,
)


log = logging.getLogger(__name__)
perf_hint_log = torch._logging.getArtifactLogger(__name__, "perf_hints")
aten = torch.ops.aten

_scaled_dot_product_attention = aten.scaled_dot_product_attention


_INFERENCE_ONLY_SFDP_PATTERNS = frozenset(
    OrderedSet(
        [
            "_sfdp_pattern_13",
            "_sfdp_pattern_15",
            "_sfdp_pattern_17",
            "_sfdp_pattern_18",
            "_sfdp_pattern_19",
            "_sfdp_pattern_20",
            "_sfdp_pattern_21",
            "_sfdp_pattern_22",
            "_sfdp_pattern_23",
            "_sfdp_pattern_24",
        ]
    )
)


def _sfdp_pattern_1(query, key, value, inv_scale):
    return (
        torch.matmul(query, key.transpose(-2, -1))
        .div(inv_scale)
        .softmax(dim=-1)
        .matmul(value)
    )


def _sfdp_replacement_1(query, key, value, inv_scale):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=1.0 / inv_scale,
    )


def _sfdp_pattern_2(query, key, value, scale_factor):
    return (
        torch.matmul(query, key.transpose(-2, -1))
        .mul(scale_factor)
        .softmax(dim=-1)
        .matmul(value)
    )


def _sfdp_replacement_2(query, key, value, scale_factor):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=scale_factor,
    )


def _sfdp_pattern_3(query, key, value, inv_scale_factor, dropout_p):
    return torch.nn.functional.dropout(
        torch.matmul(query, key.transpose(-2, -1))
        .div(inv_scale_factor)
        .softmax(dim=-1),
        p=dropout_p,
    ).matmul(value)


def _sfdp_replacement_3(query, key, value, inv_scale_factor, dropout_p):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=False,
        scale=1.0 / inv_scale_factor,
    )


def _sfdp_pattern_4(query, key, value, scale_factor, dropout_p):
    return torch.nn.functional.dropout(
        torch.matmul(query, key.transpose(-2, -1)).mul(scale_factor).softmax(dim=-1),
        p=dropout_p,
    ).matmul(value)


def _sfdp_replacement_4(query, key, value, scale_factor, dropout_p):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=False,
        scale=scale_factor,
    )


def _sfdp_pattern_5(query, key, value, attn_mask, inv_scale):
    attn_weight = torch.softmax(
        (query @ key.transpose(-2, -1) / (inv_scale)) + attn_mask, dim=-1
    )
    # attn_weight = torch.dropout(attn_weight, dropout_p)
    return attn_weight @ value


def _sfdp_replacement_5(query, key, value, attn_mask, inv_scale):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask.to(dtype=query.dtype),
        dropout_p=0.0,
        scale=1.0 / inv_scale,
        is_causal=False,
    )


def _sfdp_pattern_6(query, key, value, attn_mask, inv_scale, dropout_p):
    attn_weight = torch.softmax(
        (query @ key.transpose(-2, -1) / inv_scale) + attn_mask, dim=-1
    )
    attn_weight = torch.dropout(attn_weight, dropout_p, True)
    return attn_weight @ value


def _sfdp_replacement_6(query, key, value, attn_mask, inv_scale, dropout_p):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask.to(dtype=query.dtype),
        dropout_p=dropout_p,
        scale=1.0 / inv_scale,
        is_causal=False,
    )


def _sfdp_pattern_7(query, key, value, inv_scale, dropout_p):
    # in real workloads inputs to matmul are permuted
    # causing matmul to expand to a series of expand and clone calls
    # we want the same to happen during pattern tracing
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    div = q @ k.transpose(-2, -1) / inv_scale
    div = div.to(torch.float32)
    attn_weight = torch.softmax(div, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, True)
    attn_weight = attn_weight.to(torch.float16)
    v = v.to(attn_weight.dtype)
    return attn_weight @ v


def _sfdp_replacement_7(query, key, value, inv_scale, dropout_p):
    # sdpa prefers inputs in permuted format
    # it makes a copy to put them in this format
    # if they aren't already
    # to make replacement efficient ensure that inputs to sdpa
    # are in required order
    counters["inductor"]["fuse_attention"] += 1
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    return _scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,  # attn_mask,
        dropout_p=dropout_p,
        scale=1.0 / inv_scale,
        is_causal=False,
    )


def _sfdp_pattern_8(query, key, value, inv_scale):
    # no dropout version of pattern 7
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    div = q @ k.transpose(-2, -1) / inv_scale
    div = div.to(torch.float32)
    attn_weight = torch.softmax(div, dim=-1)
    attn_weight = attn_weight.to(torch.float16)
    v = v.to(attn_weight.dtype)
    return attn_weight @ v


def _sfdp_replacement_8(query, key, value, inv_scale):
    counters["inductor"]["fuse_attention"] += 1
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    return _scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,  # attn_mask,
        dropout_p=0.0,
        scale=1.0 / inv_scale,
        is_causal=False,
    )


def _sfdp_pattern_9(query, key, value, inv_scale, dropout_p):
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    q = q / inv_scale
    div = q @ k.transpose(-2, -1)
    div = div.to(torch.float32)
    attn_weight = torch.softmax(div, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, True)
    attn_weight = attn_weight.to(torch.float16)
    v = v.to(attn_weight.dtype)
    return attn_weight @ v


def _sfdp_replacement_9(query, key, value, inv_scale, dropout_p):
    counters["inductor"]["fuse_attention"] += 1
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    return _scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,  # attn_mask,
        dropout_p=dropout_p,
        scale=1.0 / inv_scale,
        is_causal=False,
    )


def _sfdp_pattern_10(query, key, value, inv_scale):
    # no dropout version of 9
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    q = q / inv_scale
    div = q @ k.transpose(-2, -1)
    div = div.to(torch.float32)
    attn_weight = torch.softmax(div, dim=-1)
    attn_weight = attn_weight.to(torch.float16)
    v = v.to(attn_weight.dtype)
    return attn_weight @ v


def _sfdp_replacement_10(query, key, value, inv_scale):
    counters["inductor"]["fuse_attention"] += 1
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    return _scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,  # attn_mask,
        dropout_p=0.0,
        scale=1.0 / inv_scale,
        is_causal=False,
    )


def _sfdp_pattern_11(query, key, value, inv_scale):
    # Mainly for huggingface models
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    return torch.matmul(q, k.transpose(-2, -1)).div(inv_scale).softmax(dim=-1).matmul(v)


def _sfdp_replacement_11(query, key, value, inv_scale):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=1.0 / inv_scale,
    )


def _sfdp_pattern_12(query, key, value, inv_scale_factor, dropout_p):
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    return torch.nn.functional.dropout(
        torch.matmul(q, k.transpose(-2, -1)).div(inv_scale_factor).softmax(dim=-1),
        p=dropout_p,
    ).matmul(v)


def _sfdp_replacement_12(query, key, value, inv_scale_factor, dropout_p):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=False,
        scale=1.0 / inv_scale_factor,
    )


def _sfdp_pattern_13(query, key, value, dropout_p):
    attn_weight = torch.bmm(query, key.transpose(1, 2)).softmax(dim=-1)
    attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p)
    return torch.bmm(attn_weight, value)


def _sfdp_replacement_13(query, key, value, dropout_p):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(
        query.unsqueeze(0),
        key.unsqueeze(0),
        value.unsqueeze(0),
        dropout_p=dropout_p,
        scale=1.0,
    ).squeeze(0)


def _sfdp_pattern_14(query, key, value, attn_mask, inv_scale):
    # for BertLarge
    # Permutations are needed to create clones in graph.
    q = query.permute([0, 2, 1, 3])
    k = key.permute([0, 2, 1, 3])
    v = value.permute([0, 2, 1, 3])
    return (
        (torch.matmul(q, k.transpose(-2, -1)).div(inv_scale) + attn_mask)
        .softmax(dim=-1)
        .matmul(v)
    )


def _sfdp_replacement_14(query, key, value, attn_mask, inv_scale):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=attn_mask.to(dtype=query.dtype),
        dropout_p=0.0,
        is_causal=False,
        scale=1.0 / inv_scale,
    )


def _sfdp_pattern_15(query, key, value, attn_mask, inv_scale):
    # for DistilBert
    # Permutations are needed to create clones in graph.
    # Ref: https://github.com/pytorch/pytorch/issues/119911
    q = query.permute([0, 2, 1, 3])
    k = key.permute([0, 2, 1, 3])
    v = value.permute([0, 2, 1, 3])
    bs = q.size(0)
    k_len = k.size(-2)
    scores = q @ k.transpose(-2, -1)
    scores = scores.div(inv_scale)
    fill_value = torch.full((), -float("inf"), dtype=query.dtype, device=query.device)
    attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)
    return torch.softmax(scores.masked_fill(attn_mask, fill_value), dim=-1) @ v


def _sfdp_replacement_15(query, key, value, attn_mask, inv_scale):
    counters["inductor"]["fuse_attention"] += 1
    bs = query.size(0)
    n_head = query.size(2)
    q_len = query.size(1)
    k_len = key.size(1)
    # do attn_mask->logical_not() in _scaled_dot_product_attention
    attn_mask = (
        (attn_mask == 1).view((bs, 1, 1, k_len)).expand((bs, n_head, q_len, k_len))
    )
    return _scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=attn_mask.to(dtype=torch.bool),
        dropout_p=0.0,
        is_causal=False,
        scale=1.0 / inv_scale,
    )


def _sfdp_pattern_16(query, key, value, attn_mask, inv_scale, dropout_p):
    # for BertLarge with dropout
    q = query.permute([0, 2, 1, 3])
    k = key.permute([0, 2, 1, 3])
    v = value.permute([0, 2, 1, 3])
    return (
        torch.nn.functional.dropout(
            (torch.matmul(q, k.transpose(-2, -1)).div(inv_scale) + attn_mask).softmax(
                dim=-1
            ),
            dropout_p,
        )
        .to(dtype=query.dtype)
        .matmul(v)
    )


def _sfdp_replacement_16(query, key, value, attn_mask, inv_scale, dropout_p):
    counters["inductor"]["fuse_attention"] += 1
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    if query.device.type == "cuda" and torch.version.hip is None:
        # Keep the Bert CUDA pattern on its original math path; generic SDPA can
        # pick fused backends whose scaling/dropout numerics fail tight checks.
        attn_weight = torch.matmul(query, key.transpose(-2, -1))
        attn_weight = attn_weight.div(inv_scale) + attn_mask
        attn_weight = attn_weight.softmax(dim=-1)
        if dropout_p != 0.0:
            attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p)
        return attn_weight.to(dtype=query.dtype).matmul(value)
    attn_mask = attn_mask.to(dtype=query.dtype)
    return _scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=False,
        scale=1.0 / inv_scale,
    )


def _sfdp_pattern_17(query, key, value, attn_mask, inv_scale, dropout_p):
    # for DistilBert with dropout
    q = query.permute([0, 2, 1, 3])
    k = key.permute([0, 2, 1, 3])
    v = value.permute([0, 2, 1, 3])
    bs = q.size(0)
    k_len = k.size(-2)
    scores = q @ k.transpose(-2, -1)
    scores = scores.div(inv_scale)
    fill_value = torch.full((), -float("inf"), dtype=query.dtype, device=query.device)
    attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)
    return (
        torch.nn.functional.dropout(
            torch.softmax(scores.masked_fill(attn_mask, fill_value), dim=-1), dropout_p
        )
        @ v
    )


def _sfdp_replacement_17(query, key, value, attn_mask, inv_scale, dropout_p):
    counters["inductor"]["fuse_attention"] += 1
    bs = query.size(0)
    n_head = query.size(2)
    q_len = query.size(1)
    k_len = key.size(1)
    # do attn_mask->logical_not() in _scaled_dot_product_attention
    attn_mask = (
        (attn_mask == 1).view((bs, 1, 1, k_len)).expand((bs, n_head, q_len, k_len))
    )
    return _scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=attn_mask.to(dtype=torch.bool),
        dropout_p=dropout_p,
        is_causal=False,
        scale=1.0 / inv_scale,
    )


def _sfdp_pattern_18(query, key, value, causal_mask, inv_scale, dropout_p):
    # for hf_GPT2 with dropout (introduces clone node) for inference
    # it also returns permuted key & value
    query = query.permute([0, 2, 1, 3])
    key = key.permute([0, 2, 1, 3])
    value = value.permute([0, 2, 1, 3])
    attn_weights = torch.matmul(query, key.permute(0, 1, 3, 2))
    inv_scale = torch.full(
        [],
        inv_scale,
        dtype=attn_weights.dtype,
        device=attn_weights.device,
    )
    attn_weights = attn_weights.div(inv_scale)
    causal_mask_value = torch.full(
        (), torch.finfo(query.dtype).min, dtype=query.dtype, device=query.device
    )
    attn_weights = torch.where(causal_mask, attn_weights, causal_mask_value)
    return (
        (
            torch.nn.functional.dropout(attn_weights.softmax(dim=-1), dropout_p).matmul(
                value
            )
        ),
        key,
        value,
    )


def _sfdp_replacement_18(query, key, value, causal_mask, inv_scale, dropout_p):
    counters["inductor"]["fuse_attention"] += 1
    permuted_key = key.transpose(1, 2)
    permuted_value = value.transpose(1, 2)
    return (
        _scaled_dot_product_attention(
            query.transpose(1, 2),
            permuted_key,
            permuted_value,
            attn_mask=causal_mask,
            dropout_p=dropout_p,
            is_causal=False,
            scale=1.0 / inv_scale,
        ),
        permuted_key,
        permuted_value,
    )


def _sfdp_pattern_19(query, key, value, causal_mask, attn_mask, inv_scale, dropout_p):
    # for token-classification+gpt2 / text-generation+gpt2
    attn_weights = torch.matmul(query, key.permute(0, 1, 3, 2))
    inv_scale = torch.full(
        [],
        inv_scale,
        dtype=attn_weights.dtype,
        device=attn_weights.device,
    )
    attn_weights = attn_weights.div(inv_scale)
    causal_mask_value = torch.full(
        (), torch.finfo(query.dtype).min, dtype=query.dtype, device=query.device
    )
    attn_weights = torch.where(causal_mask, attn_weights, causal_mask_value)
    attn_weights = attn_weights + attn_mask
    attn_weights = attn_weights.softmax(dim=-1).type(value.dtype)
    return torch.nn.functional.dropout(attn_weights, dropout_p).matmul(value)


def _sfdp_replacement_19(
    query, key, value, causal_mask, attn_mask, inv_scale, dropout_p
):
    counters["inductor"]["fuse_attention"] += 1
    fill_value = torch.full((), -float("inf"), dtype=query.dtype, device=query.device)
    attn_mask = torch.where(causal_mask, attn_mask, fill_value)
    return _scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=False,
        scale=1.0 / inv_scale,
    )


def _sfdp_pattern_20(query, key, value, attn_mask, inv_scale, dropout_p):
    # for DistilBert with dropout transformers==4.44.2
    q = query.permute([0, 2, 1, 3])
    k = key.permute([0, 2, 1, 3])
    v = value.permute([0, 2, 1, 3])
    bs = q.size(0)
    k_len = k.size(-2)
    q = q.div(inv_scale)
    scores = q @ k.transpose(-2, -1)
    fill_value = torch.full((), -float("inf"), dtype=query.dtype, device=query.device)
    attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)
    return (
        torch.nn.functional.dropout(
            torch.softmax(scores.masked_fill(attn_mask, fill_value), dim=-1), dropout_p
        )
        @ v
    )


def _sfdp_replacement_20(query, key, value, attn_mask, inv_scale, dropout_p):
    counters["inductor"]["fuse_attention"] += 1
    bs = query.size(0)
    n_head = query.size(2)
    q_len = query.size(1)
    k_len = key.size(1)
    # do attn_mask->logical_not() in _scaled_dot_product_attention
    attn_mask = (
        (attn_mask == 1).view((bs, 1, 1, k_len)).expand((bs, n_head, q_len, k_len))
    )
    return _scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=attn_mask.to(dtype=torch.bool),
        dropout_p=dropout_p,
        is_causal=False,
        scale=1.0 / inv_scale,
    )


def _sfdp_pattern_21(query, key, value, attn_mask):
    # for T5 with inplace add
    query = query.permute([0, 2, 1, 3])
    key = key.permute([0, 2, 1, 3])
    value = value.permute([0, 2, 1, 3])
    score = torch.matmul(query, key.permute(0, 1, 3, 2))
    masked_score = score + attn_mask
    score = masked_score.type_as(query)
    return score.float().softmax(dim=-1).type_as(query).matmul(value)


def _sfdp_replacement_21(query, key, value, attn_mask):
    counters["inductor"]["fuse_attention"] += 1
    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)
    return _scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask.to(dtype=query.dtype),
        is_causal=False,
        scale=1.0,
    )


def _sfdp_pattern_22(query, key, value, attn_mask):
    # for T5 with inplace add and return key and value
    query = query.permute([0, 2, 1, 3])
    key = key.permute([0, 2, 1, 3])
    value = value.permute([0, 2, 1, 3])
    score = torch.matmul(query, key.permute(0, 1, 3, 2))
    masked_score = score + attn_mask
    score = masked_score.type_as(query)
    return score.float().softmax(dim=-1).type_as(query).matmul(value), key, value


def _sfdp_replacement_22(query, key, value, attn_mask):
    counters["inductor"]["fuse_attention"] += 1
    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)
    return (
        _scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask.to(dtype=query.dtype),
            is_causal=False,
            scale=1.0,
        ),
        key,
        value,
    )


def _sfdp_pattern_23(query, key, value):
    # for T5 with inplace add and
    # return key and value and
    # attn_mask is generated by atem.full(..., 0)
    query = query.permute([0, 2, 1, 3])
    key = key.permute([0, 2, 1, 3])
    value = value.permute([0, 2, 1, 3])
    score = torch.matmul(query, key.permute(0, 1, 3, 2))
    fp32_score = score.float()
    score = fp32_score.type_as(query)
    return score.float().softmax(dim=-1).type_as(query).matmul(value), key, value


def _sfdp_replacement_23(query, key, value):
    counters["inductor"]["fuse_attention"] += 1
    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)
    return (
        _scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            is_causal=False,
            scale=1.0,
        ),
        key,
        value,
    )


def _sfdp_pattern_24(query, key, value, attention_mask):
    """
    this pattern is for MBartForCausalLM/PLBartForCausalLM.
    attn_mask has a different dtype with QKV.
    there is no scale in sdpa.
    """
    bs = query.size(0)
    n_head = query.size(1)
    seq_len = query.size(2)
    head_size = query.size(3)
    q = query.view(bs * n_head, -1, head_size)
    k = key.reshape(bs * n_head, -1, head_size)
    v = value.reshape(bs * n_head, -1, head_size)
    attn_weights = torch.bmm(q, k.transpose(1, 2))
    attn_weights = attn_weights.view(bs, n_head, seq_len, -1) + attention_mask
    attn_weights = attn_weights.view(bs * n_head, seq_len, -1)
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
    if query.dtype == torch.half:
        attn_weights = attn_weights.to(torch.half)
    attn_output = torch.bmm(attn_weights, v)
    attn_output = attn_output.view(bs, n_head, seq_len, head_size)
    return attn_output


def _sfdp_replacement_24(query, key, value, attention_mask):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask.to(dtype=query.dtype),
        is_causal=False,
        scale=1,
    )


def _sfdp_pattern_25(query, key, value, attn_mask, dropout_p):
    # for T5 with inplace add
    query = query.permute([0, 2, 1, 3])
    key = key.permute([0, 2, 1, 3])
    value = value.permute([0, 2, 1, 3])
    score = torch.matmul(query, key.permute(0, 1, 3, 2))
    masked_score = score + attn_mask
    return torch.nn.functional.dropout(
        masked_score.float().softmax(dim=-1).type_as(query), dropout_p
    ).matmul(value)


def _sfdp_replacement_25(query, key, value, attn_mask, dropout_p):
    counters["inductor"]["fuse_attention"] += 1
    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)
    if attn_mask.device.type == "xpu":
        attn_mask = attn_mask.contiguous()
    return _scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=False,
        scale=1.0,
    )


def _sfdp_pattern_26(query, key, value, attn_mask, dropout_p):
    # for T5 with inplace add and return key and value
    query = query.permute([0, 2, 1, 3])
    key = key.permute([0, 2, 1, 3])
    value = value.permute([0, 2, 1, 3])
    score = torch.matmul(query, key.permute(0, 1, 3, 2))
    masked_score = score + attn_mask
    return (
        torch.nn.functional.dropout(
            masked_score.float().softmax(dim=-1).type_as(query), dropout_p
        ).matmul(value),
        key,
        value,
    )


def _sfdp_replacement_26(query, key, value, attn_mask, dropout_p):
    counters["inductor"]["fuse_attention"] += 1
    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)
    if attn_mask.device.type == "xpu":
        attn_mask = attn_mask.contiguous()
    return (
        _scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
            scale=1.0,
        ),
        key,
        value,
    )


def _sfdp_pattern_27(query, key, value, dropout_p):
    # for T5 with inplace add and
    # return key and value and
    # attn_mask is generated by atem.full(..., 0)
    query = query.permute([0, 2, 1, 3])
    key = key.permute([0, 2, 1, 3])
    value = value.permute([0, 2, 1, 3])
    score = torch.matmul(query, key.permute(0, 1, 3, 2))
    return (
        torch.nn.functional.dropout(
            score.float().softmax(dim=-1).type_as(query), dropout_p
        ).matmul(value),
        key,
        value,
    )


def _sfdp_replacement_27(query, key, value, dropout_p):
    counters["inductor"]["fuse_attention"] += 1
    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)
    return (
        _scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=False,
            scale=1.0,
        ),
        key,
        value,
    )


def _sfdp_pattern_28(query, key, value, scale_factor, dropout_p):
    # Visformer pattern
    # same as pattern 4 but non-contiguous q/k/v
    return _sfdp_pattern_4(query, key, value, scale_factor, dropout_p)


def _sfdp_replacement_28(query, key, value, scale_factor, dropout_p):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=False,
        scale=scale_factor,
    )


def _sfdp_pattern_29(query, key, value, attn_mask, inv_scale):
    # Matches _scaled_dot_product_attention_math decomposition for
    # BERT-like models calling F.scaled_dot_product_attention with attn_mask.
    # Scale is split as sqrt across Q and K (mul.Scalar in fp16), bmm stays
    # in fp16, and _safe_softmax upcasts to fp32 internally.
    # Permutations are needed to create clones in graph since real workloads
    # have non-contiguous inputs from .transpose(1, 2).
    q = query.permute([0, 2, 1, 3])
    k = key.permute([0, 2, 1, 3])
    v = value.permute([0, 2, 1, 3])
    q = torch.ops.aten.mul.Scalar(q, inv_scale)
    k = torch.ops.aten.mul.Scalar(k.transpose(-2, -1), inv_scale)
    attn_weight = (q @ k) + attn_mask
    attn_weight = torch.ops.aten._safe_softmax(attn_weight, -1)
    return attn_weight @ v


def _sfdp_replacement_29(query, key, value, attn_mask, inv_scale):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=attn_mask.to(dtype=query.dtype),
        dropout_p=0.0,
        is_causal=False,
        scale=inv_scale * inv_scale,
    )


def _sfdp_pattern_30(query, key, value, inv_scale):
    # Same as pattern 29 but without attention mask.
    # Permutations are needed to create clones in graph.
    q = query.permute([0, 2, 1, 3])
    k = key.permute([0, 2, 1, 3])
    v = value.permute([0, 2, 1, 3])
    q = torch.ops.aten.mul.Scalar(q, inv_scale)
    k = torch.ops.aten.mul.Scalar(k.transpose(-2, -1), inv_scale)
    attn_weight = q @ k
    attn_weight = torch.ops.aten._safe_softmax(attn_weight, -1)
    return attn_weight @ v


def _sfdp_replacement_30(query, key, value, inv_scale):
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=inv_scale * inv_scale,
    )


@functools.lru_cache(None)
def _warn_tf32_disabled() -> None:
    if (
        torch.cuda.is_available()
        and torch.backends.cuda.matmul.fp32_precision != "tf32"
        and torch.cuda.get_device_capability() >= (8, 0)
    ):
        perf_hint_log.info(
            "TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. "
            "Skipping pattern matching to fused flash-attention. "
            "Consider setting `torch.set_float32_matmul_precision('high')` for better performance."
        )


def _sfdp_params_check(match):
    if not all(k in match.kwargs for k in ("query", "key", "value")):
        raise AssertionError("expected query, key, value in match.kwargs")
    query = match.kwargs["query"].meta["val"]
    key = match.kwargs["key"].meta["val"]
    value = match.kwargs["value"].meta["val"]
    if not (query.dtype == key.dtype == value.dtype) or not (
        query.device == key.device == value.device
    ):
        return False
    # fused kernels use tf32
    if (
        query.device.type == "cuda"
        and query.dtype == torch.float32
        and torch.backends.cuda.matmul.fp32_precision != "tf32"
    ):
        _warn_tf32_disabled()
        return False

    add_mask_node = filter_nodes(match.nodes, aten.add.Tensor)
    # Has attn_mask add.
    if len(add_mask_node) > 0:
        attn_mask_node = add_mask_node[0].args[1]
        # attn_mask_node may be a float/int number.
        if not hasattr(attn_mask_node, "meta"):
            return False
        attn_mask = attn_mask_node.meta["val"]  # type: ignore[union-attr]
        # Make sure attn_mask.dtype == query.dtype or attn_mask.dtype == torch.bool
        # attn_mask.dtype == torch.float for models like albert.
        if (
            not isinstance(attn_mask, torch.Tensor)
            or not (
                attn_mask.dtype == query.dtype
                or attn_mask.dtype == torch.bool
                or attn_mask.dtype == torch.float
            )
            or query.device != attn_mask.device
            # When we tensorify floats we end up turning floats
            # into 0d scalar tensors. It doesn't make any sense
            # to have a 0d scalar tensor attention mask so
            # conveniently we can insert this check to get
            # tests that erroneously passing in a float
            # attention mask to fail as expected.
            or attn_mask.dim() == 0
        ):
            return False
    return True


def _sfdp_pattern_13_check(match):
    if not _sfdp_params_check(match):
        return False
    permutes = filter_nodes(match.nodes, aten.permute.default)
    if len(permutes) != 1:
        return False
    # The serialized pattern wildcard-matches the permute dimensions.
    permute_dims = permutes[0].args[1]
    return isinstance(permute_dims, (list, tuple)) and tuple(permute_dims) == (0, 2, 1)


def _sfdp_extra_check(scale_factor_op=None, disable_cuda=False):
    def fn(match):
        if (
            disable_cuda
            and "query" in match.kwargs
            and "cuda" in str(match.kwargs["query"].meta["val"].device)
        ):
            return False
        if scale_factor_op is not None:
            scale_factor_node = filter_nodes(match.nodes, scale_factor_op)[0]
            # Note: args[1] of the scale_factor_node is always the scale_factor for the current patterns.
            scale_factor = scale_factor_node.args[1]
            # make sure the scale_factor a float/int. SymInt?
            if not isinstance(scale_factor, (float, int)):
                return False
        return _sfdp_params_check(match)

    return fn


def partialize_and_update_signature(func, **kwargs):
    """
    Equivalent to functools.partial but also updates the signature on returned function
    """
    original_sig = inspect.signature(func)
    parameters = original_sig.parameters

    new_parameters = {
        key: value for key, value in parameters.items() if key not in kwargs
    }
    new_sig = inspect.Signature(parameters=list(new_parameters.values()))

    partial_func = functools.partial(func, **kwargs)

    def wrapper(*args, **kwargs):
        return partial_func(*args, **kwargs)

    wrapper.__signature__ = new_sig  # type: ignore[attr-defined]
    wrapper.__name__ = func.__name__

    return wrapper


def _get_sfdp_patterns(input_device: torch.device | None = None):
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
    g_inp = functools.partial(
        torch.empty, (2, 4, 8, 16), device=device, requires_grad=True
    )
    # non-contiguous input to cover more patterns.
    gn_inp = functools.partial(
        torch.empty_strided,
        (2, 6, 16, 8),
        (2304, 128, 1, 16),
        device=device,
        requires_grad=True,
    )
    # attn_mask
    b_inp = functools.partial(torch.empty, (1, 1, 8, 8), device=device)
    m_inp = functools.partial(torch.empty, (2, 1, 1, 4), device=device)
    # need 2d attn_mask to generate patterns with view op
    m_inp_2d = functools.partial(torch.empty, (2, 4), device=device)
    # inv_scale
    c_inp = functools.partial(torch.tensor, 2.0, device=device)
    # workaround https://github.com/pytorch/pytorch/issues/97894
    # 0.113377 is a "magic" value that lets us recover the lost input arg relationship
    d = {"dropout_p": 0.113377}
    s = {"inv_scale": 0.66666}
    sd = {"inv_scale": 0.66666, "dropout_p": 0.113377}

    # we could also generate all these patterns in 3d.. TODO
    g_3d_inp = functools.partial(
        torch.empty, (1024, 128, 128), device=device, requires_grad=True
    )

    # reshape in matmul decomposition generates a clone when batch_size>1 due to the memory layout change.
    # however when batch_size=1, reshape does not change the memory layout, so clone would not be generated.
    # here we need to trace with input of batch_size=1 to generate a pattern graph without clone.
    g_bs1_inp = functools.partial(
        torch.empty, (1, 4, 8, 16), device=device, requires_grad=True
    )
    m_bs1_inp = functools.partial(torch.empty, (1, 1, 1, 4), device=device)

    # For patterns with permute([0,2,1,3]) inside (e.g. BERT-like models),
    # input is [B, S, H, D] and after permute becomes [B, H, S, D].
    # S must match the mask dimension (8) so Q@K^T shape [B, H, S, S] matches b().
    # gp = "generic with permute"; gp_bs1 is the batch_size=1 variant (no clone).
    gp_inp = functools.partial(
        torch.empty, (2, 8, 4, 16), device=device, requires_grad=True
    )
    gp_bs1_inp = functools.partial(
        torch.empty, (1, 8, 4, 16), device=device, requires_grad=True
    )

    # softmax will generate a dtype conversion on inputs if they are in half,
    # but will not in float, so we generate a pattern for both
    for dtype in [torch.float, torch.half]:
        g = functools.partial(g_inp, dtype=dtype)
        gn = functools.partial(gn_inp, dtype=dtype)
        b = functools.partial(b_inp, dtype=dtype)
        b_float = functools.partial(b_inp, dtype=torch.float)
        b_bool = functools.partial(b_inp, dtype=torch.bool)
        m = functools.partial(m_inp, dtype=dtype)
        m_float = functools.partial(m_inp, dtype=torch.float)
        m_bool = functools.partial(m_inp, dtype=torch.bool)
        m_2d = functools.partial(m_inp_2d, dtype=dtype)
        c = functools.partial(c_inp, dtype=dtype)
        g_3d = functools.partial(g_3d_inp, dtype=dtype)
        g_bs1 = functools.partial(g_bs1_inp, dtype=dtype)
        gp = functools.partial(gp_inp, dtype=dtype)
        gp_bs1 = functools.partial(gp_bs1_inp, dtype=dtype)
        m_bs1 = functools.partial(m_bs1_inp, dtype=dtype)
        m_bs1_float = functools.partial(m_bs1_inp, dtype=torch.float)
        m_bs1_bool = functools.partial(m_bs1_inp, dtype=torch.bool)

        candidates = [
            (
                _sfdp_pattern_1,
                _sfdp_replacement_1,
                [g(), g(), g(), c()],
                {},
                _sfdp_extra_check(aten.div.Tensor),
            ),
            (
                _sfdp_pattern_2,
                _sfdp_replacement_2,
                [g(), g(), g(), c()],
                {},
                _sfdp_extra_check(aten.mul.Tensor),
            ),
            (
                _sfdp_pattern_3,
                _sfdp_replacement_3,
                [g(), g(), g(), c()],
                d,
                _sfdp_extra_check(aten.div.Tensor),
            ),
            (
                _sfdp_pattern_4,
                _sfdp_replacement_4,
                [g(), g(), g(), c()],
                d,
                _sfdp_extra_check(aten.mul.Tensor),
            ),
            (
                _sfdp_pattern_5,
                _sfdp_replacement_5,
                [g(), g(), g(), b()],
                s,
                _sfdp_params_check,
            ),
            (
                _sfdp_pattern_6,
                _sfdp_replacement_6,
                [g(), g(), g(), b()],
                sd,
                _sfdp_params_check,
            ),
            (
                _sfdp_pattern_7,
                _sfdp_replacement_7,
                [g(), g(), g()],
                sd,
                _sfdp_params_check,
            ),
            (
                _sfdp_pattern_8,
                _sfdp_replacement_8,
                [g(), g(), g()],
                s,
                _sfdp_params_check,
            ),
            (
                _sfdp_pattern_9,
                _sfdp_replacement_9,
                [g(), g(), g()],
                sd,
                _sfdp_params_check,
            ),
            (
                _sfdp_pattern_10,
                _sfdp_replacement_10,
                [g(), g(), g()],
                s,
                _sfdp_params_check,
            ),
            (
                _sfdp_pattern_11,
                _sfdp_replacement_11,
                [g(), g(), g(), c()],
                {},
                _sfdp_extra_check(aten.div.Tensor),
            ),
            (
                _sfdp_pattern_12,
                _sfdp_replacement_12,
                [g(), g(), g(), c()],
                d,
                _sfdp_extra_check(aten.div.Tensor),
            ),
            (
                _sfdp_pattern_13,
                _sfdp_replacement_13,
                [g_3d(), g_3d(), g_3d()],
                d,
                _sfdp_pattern_13_check,
            ),
            (
                _sfdp_pattern_14,
                _sfdp_replacement_14,
                [g(), g(), g(), m(), c()],
                {},
                _sfdp_extra_check(aten.div.Tensor),
            ),
            (
                _sfdp_pattern_15,
                _sfdp_replacement_15,
                [g(), g(), g(), m_2d(), c()],
                {},
                _sfdp_extra_check(aten.div.Tensor),
            ),
            (
                _sfdp_pattern_16,
                _sfdp_replacement_16,
                [g(), g(), g(), m(), c()],
                d,
                _sfdp_extra_check(aten.div.Tensor),
            ),
            (
                _sfdp_pattern_16,
                _sfdp_replacement_16,
                [g_bs1(), g_bs1(), g_bs1(), m_bs1(), c()],
                d,
                _sfdp_extra_check(aten.div.Tensor),
            ),
            (
                _sfdp_pattern_17,
                _sfdp_replacement_17,
                [g(), g(), g(), m_2d(), c()],
                d,
                _sfdp_extra_check(aten.div.Tensor),
            ),
            (
                _sfdp_pattern_18,
                _sfdp_replacement_18,
                [g(), g(), g(), m_bool()],
                sd,
                _sfdp_params_check,
            ),
            (
                _sfdp_pattern_18,
                _sfdp_replacement_18,
                [g_bs1(), g_bs1(), g_bs1(), m_bs1_bool()],
                sd,
                _sfdp_params_check,
            ),
            (
                _sfdp_pattern_19,
                _sfdp_replacement_19,
                [g(), g(), g(), b_bool(), b_float()],
                sd,
                _sfdp_params_check,
            ),
            (
                _sfdp_pattern_20,
                _sfdp_replacement_20,
                [g(), g(), g(), m_2d()],
                sd,
                _sfdp_params_check,
            ),
            (
                _sfdp_pattern_21,
                _sfdp_replacement_21,
                [g(), g(), g(), m_float()],
                {},
                _sfdp_params_check,
            ),
            (
                _sfdp_pattern_21,
                _sfdp_replacement_21,
                [g_bs1(), g_bs1(), g_bs1(), m_bs1_float()],
                {},
                _sfdp_params_check,
            ),
            (
                _sfdp_pattern_22,
                _sfdp_replacement_22,
                [g(), g(), g(), m_float()],
                {},
                _sfdp_params_check,
            ),
            (
                _sfdp_pattern_22,
                _sfdp_replacement_22,
                [g_bs1(), g_bs1(), g_bs1(), m_bs1_float()],
                {},
                _sfdp_params_check,
            ),
            (
                _sfdp_pattern_23,
                _sfdp_replacement_23,
                [g(), g(), g()],
                {},
                _sfdp_params_check,
            ),
            (
                _sfdp_pattern_23,
                _sfdp_replacement_23,
                [g_bs1(), g_bs1(), g_bs1()],
                {},
                _sfdp_params_check,
            ),
            (
                _sfdp_pattern_24,
                _sfdp_replacement_24,
                [g(), g(), g(), b_float()],
                {},
                _sfdp_extra_check,
            ),
            (
                _sfdp_pattern_25,
                _sfdp_replacement_25,
                [g(), g(), g(), m()],
                d,
                _sfdp_extra_check(disable_cuda=True),
            ),
            (
                _sfdp_pattern_25,
                _sfdp_replacement_25,
                [g_bs1(), g_bs1(), g_bs1(), m_bs1()],
                d,
                _sfdp_extra_check(disable_cuda=True),
            ),
            (
                _sfdp_pattern_26,
                _sfdp_replacement_26,
                [g(), g(), g(), m()],
                d,
                _sfdp_extra_check(disable_cuda=True),
            ),
            (
                _sfdp_pattern_26,
                _sfdp_replacement_26,
                [g_bs1(), g_bs1(), g_bs1(), m_bs1()],
                d,
                _sfdp_extra_check(disable_cuda=True),
            ),
            (
                _sfdp_pattern_27,
                _sfdp_replacement_27,
                [g(), g(), g()],
                d,
                _sfdp_extra_check(disable_cuda=True),
            ),
            (
                _sfdp_pattern_27,
                _sfdp_replacement_27,
                [g_bs1(), g_bs1(), g_bs1()],
                d,
                _sfdp_extra_check(disable_cuda=True),
            ),
            (
                _sfdp_pattern_28,
                _sfdp_replacement_28,
                [gn(), gn(), gn(), c()],
                d,
                _sfdp_extra_check(aten.mul.Tensor),
            ),
            (
                _sfdp_pattern_29,
                _sfdp_replacement_29,
                [gp(), gp(), gp(), b()],
                s,
                _sfdp_extra_check(aten.mul.Scalar),
            ),
            (
                _sfdp_pattern_29,
                _sfdp_replacement_29,
                [gp_bs1(), gp_bs1(), gp_bs1(), b()],
                s,
                _sfdp_extra_check(aten.mul.Scalar),
            ),
            (
                _sfdp_pattern_30,
                _sfdp_replacement_30,
                [gp(), gp(), gp()],
                s,
                _sfdp_extra_check(aten.mul.Scalar),
            ),
            (
                _sfdp_pattern_30,
                _sfdp_replacement_30,
                [gp_bs1(), gp_bs1(), gp_bs1()],
                s,
                _sfdp_extra_check(aten.mul.Scalar),
            ),
        ]
        mask_fp32_patterns = ["pattern_16"]
        if dtype == torch.half:
            # Add inputs of bf16 q/k/v and fp32 mask, for models like albert.
            candidates.append(
                (
                    _sfdp_pattern_16,
                    _sfdp_replacement_16,
                    [g(), g(), g(), m_float(), c()],
                    d,
                    _sfdp_extra_check(
                        aten.div.Tensor, disable_cuda=torch.version.hip is None
                    ),
                )
            )
            candidates.append(
                (
                    _sfdp_pattern_16,
                    _sfdp_replacement_16,
                    [g_bs1(), g_bs1(), g_bs1(), m_bs1_float(), c()],
                    d,
                    _sfdp_extra_check(
                        aten.div.Tensor, disable_cuda=torch.version.hip is None
                    ),
                )
            )

        for pattern, replacement, args, workaround, extra_check in candidates:
            # XXX: when adding a new pattern, re-run `gen_attention_patterns` so the pattern
            # gets serialized to a python file and does not require tracing at runtime.
            if not isinstance(workaround, dict):
                raise AssertionError(
                    f"expected workaround to be a dict, got {type(workaround)}"
                )
            name = pattern.__name__
            pattern_name = name

            if dtype != torch.float:
                name += "_half"
                if (
                    any(p in name for p in mask_fp32_patterns)
                    and args[3].dtype == torch.float32
                ):
                    name += "_mask_fp32"
            if args[0].size(0) == 1:
                name += "_bs1"

            if pattern_name not in _INFERENCE_ONLY_SFDP_PATTERNS:
                training_name = name + "_training"
                yield (
                    training_name,
                    {
                        "search_fn": pattern,
                        "replace_fn": replacement,
                        "example_inputs": args,
                        "trace_fn": joint_fwd_bwd,
                        "pass_dicts": patterns,
                        "extra_check": extra_check,
                        "scalar_workaround": workaround,
                        "skip_duplicates": True,
                    },
                )
            inference_workaround = {}
            if workaround:
                if len(workaround) > 2:
                    raise AssertionError(
                        f"expected len(workaround) <= 2, got {len(workaround)}"
                    )
                if "inv_scale" in workaround:
                    inference_workaround["inv_scale"] = workaround["inv_scale"]
                if "dropout_p" in workaround:
                    # functools.partial insufficient because we look at signature downstream
                    pattern = partialize_and_update_signature(pattern, dropout_p=0.0)
                    replacement = partialize_and_update_signature(
                        replacement, dropout_p=0.0
                    )

            inference_name = name + "_inference"
            yield (
                inference_name,
                {
                    "search_fn": pattern,
                    "replace_fn": replacement,
                    "example_inputs": args,
                    "trace_fn": fwd_only,
                    "pass_dicts": patterns,
                    "extra_check": extra_check,
                    "scalar_workaround": inference_workaround,
                    # with dropout turned into clone, we end up with a number of
                    # semantically identical graphs
                    "skip_duplicates": True,
                },
            )


@functools.cache
def _sfdp_init(input_device: torch.device | None = None):
    for key, register_replacement_kwargs in _get_sfdp_patterns(input_device):
        gen_register_replacement(key, **register_replacement_kwargs)
<|endoftext|># mypy: allow-untyped-defs
import logging

import torch
from torch import Tensor
from torch._dynamo.utils import counters, is_node_meta_valid
from torch.fx.experimental.symbolic_shapes import (
    statically_known_false,
    statically_known_true,
)

from .. import config
from ..pattern_matcher import Arg, CallFunction, Match, register_graph_pattern
from .split_cat import construct_pattern_matcher_pass


aten = torch.ops.aten
log = logging.getLogger(__name__)

# TODO: need a better strategy for decomposing mm
# The following two constants are for CUDA device only
MIN_FIRST_DIMENSION_DECOMPOSITION = 10240
MAX_OTHER_DIMENSION_DECOMPOSITION = 32
# The following two constants are for CPU device only
CPU_MAX_FIRST_DIMENSION_DECOMPOSITION = 1
CPU_MAX_OTHER_DIMENSION_DECOMPOSITION = 2048

min_first_dimension_decomposition = MIN_FIRST_DIMENSION_DECOMPOSITION
max_other_dimension_decomposition = MAX_OTHER_DIMENSION_DECOMPOSITION
cpu_max_first_dimension_decomposition = CPU_MAX_FIRST_DIMENSION_DECOMPOSITION
cpu_max_other_dimension_decomposition = CPU_MAX_OTHER_DIMENSION_DECOMPOSITION
if "decompose_mm_pass" in config.post_grad_fusion_options:
    min_first_dimension_decomposition = config.post_grad_fusion_options[
        "decompose_mm_pass"
    ].get("min_first_dimension_decomposition", MIN_FIRST_DIMENSION_DECOMPOSITION)
    max_other_dimension_decomposition = config.post_grad_fusion_options[
        "decompose_mm_pass"
    ].get("max_other_dimension_decomposition", MAX_OTHER_DIMENSION_DECOMPOSITION)
    cpu_max_first_dimension_decomposition = config.post_grad_fusion_options[
        "decompose_mm_pass"
    ].get(
        "cpu_max_first_dimension_decomposition", CPU_MAX_FIRST_DIMENSION_DECOMPOSITION
    )
    cpu_max_other_dimension_decomposition = config.post_grad_fusion_options[
        "decompose_mm_pass"
    ].get(
        "cpu_max_other_dimension_decomposition", CPU_MAX_OTHER_DIMENSION_DECOMPOSITION
    )


def check_device(a: Tensor, b: Tensor, device="cuda") -> bool:
    return (a.device.type == b.device.type) and (b.device.type == device)


def realize_inputs(inputs: list[torch.fx.Node]):
    for inp in inputs:
        if isinstance(inp, torch.fx.node.Node):
            inp.meta["inductor_realize_to_strides"] = True


def should_decompose_bmm(mat1, mat2) -> bool:
    if is_node_meta_valid(mat1) and is_node_meta_valid(mat2):
        mat1 = mat1.meta["val"]
        mat2 = mat2.meta["val"]
    else:
        return False
    if len(mat1.shape) != 3 or len(mat2.shape) != 3:
        return False
    if check_device(mat1, mat2, device="cuda") or check_device(
        mat1, mat2, device="xpu"
    ):
        if mat1.shape[0] < min_first_dimension_decomposition:
            return False
        # 2 of m, n, k must be <= MAX_OTHER_DIMENSION_DECOMPOSITION
        # use bool() to deal with BooleanAtom type
        if (
            bool(mat1.shape[1] < max_other_dimension_decomposition)
            + bool(mat1.shape[2] < max_other_dimension_decomposition)
            + bool(mat2.shape[2] < max_other_dimension_decomposition)
            < 2
        ):
            return False
        return True
    elif check_device(mat1, mat2, device="cpu"):
        if (
            mat1.shape[0] <= cpu_max_first_dimension_decomposition
            and mat2.shape[0] <= cpu_max_first_dimension_decomposition
        ):
            return True
    return False


def should_decompose_mm(mat1, mat2) -> bool:
    """
    Determines whether matrix multiplication (mm) should be decomposed into pointwise operations
    based on the input matrices' metadata, shapes, device placement, and configuration options.
    Args:
        mat1: The first matrix operand. Expected to be an object with a `.meta` attribute containing
              a "val" key, or a tensor-like object with a `.shape` attribute.
        mat2: The second matrix operand. Same requirements as `mat1`.
    Returns:
        bool: True if the matrix multiplication should be decomposed according to the following logic:
            - Both inputs must have valid node metadata.
            - Both matrices must be 2-dimensional.
            - If the configuration option `skip_dynamic_shape_dim_check` is False:
                - Decomposition is only considered for statically-shaped matrices.
                - For CUDA devices: `mat1.shape[0]` must be at least `min_first_dimension_decomposition`,
                  and both dimensions of `mat2` must be less than `max_other_dimension_decomposition`.
                - For CPU devices: All relevant dimensions must be less than or equal to their respective
                  CPU decomposition thresholds.
            - If `skip_dynamic_shape_dim_check` is True:
                - Decomposition is considered for dynamic shapes as well, using a combination of
                  `statically_known_true` and `statically_known_false` checks to handle uncertainty.
                - The same dimension and device checks apply, but allow for dynamic/static uncertainty.
            - Returns False if any of the above conditions are not met.
    Notes:
        - Relies on helper functions such as `is_node_meta_valid`, `check_device`, `statically_known_true`,
          and `statically_known_false`, as well as configuration values like
          `min_first_dimension_decomposition`, `max_other_dimension_decomposition`, etc.
        - Designed for use in graph optimization or fusion passes where decomposing large or dynamic
          matrix multiplications can improve performance or memory usage.
    """
    if is_node_meta_valid(mat1) and is_node_meta_valid(mat2):
        mat1 = mat1.meta["val"]
        mat2 = mat2.meta["val"]
    else:
        return False
    if len(mat1.shape) != 2 or len(mat2.shape) != 2:
        return False
    # case 1: we skip decompose mm if the input is dynamic shape
    if not config.post_grad_fusion_options["decompose_mm_pass"].get(
        "skip_dynamic_shape_dim_check", False
    ):
        return (
            (
                check_device(mat1, mat2, device="cuda")
                or check_device(mat1, mat2, device="xpu")
            )
            and statically_known_true(
                mat1.shape[0] >= min_first_dimension_decomposition
            )
            and statically_known_true(mat2.shape[0] < max_other_dimension_decomposition)
            and statically_known_true(mat2.shape[1] < max_other_dimension_decomposition)
        ) or (
            check_device(mat1, mat2, device="cpu")
            and statically_known_true(
                mat1.shape[0] <= cpu_max_first_dimension_decomposition
            )
            and statically_known_true(
                mat2.shape[0] <= cpu_max_other_dimension_decomposition
            )
            and statically_known_true(
                mat2.shape[1] <= cpu_max_other_dimension_decomposition
            )
        )
    # case 2: we decompose mm if the input is dynamic shape
    else:
        return (
            (
                check_device(mat1, mat2, device="cuda")
                or check_device(mat1, mat2, device="xpu")
            )
            and (
                statically_known_true(
                    mat1.shape[0] >= min_first_dimension_decomposition
                )
                or not statically_known_false(
                    mat1.shape[0] >= min_first_dimension_decomposition
                )
            )
            and (
                statically_known_true(mat2.shape[0] < max_other_dimension_decomposition)
                or not statically_known_false(
                    mat2.shape[0] < max_other_dimension_decomposition
                )
            )
            and (
                statically_known_true(mat2.shape[1] < max_other_dimension_decomposition)
                or not statically_known_false(
                    mat2.shape[1] < max_other_dimension_decomposition
                )
            )
        ) or (
            check_device(mat1, mat2, device="cpu")
            and (
                statically_known_true(
                    mat1.shape[0] <= cpu_max_first_dimension_decomposition
                )
                or not statically_known_false(
                    mat1.shape[0] <= cpu_max_first_dimension_decomposition
                )
            )
            and (
                statically_known_true(
                    mat2.shape[0] <= cpu_max_other_dimension_decomposition
                )
                or not statically_known_false(
                    mat2.shape[0] <= cpu_max_other_dimension_decomposition
                )
            )
            and (
                statically_known_true(
                    mat2.shape[1] <= cpu_max_other_dimension_decomposition
                )
                or not statically_known_false(
                    mat2.shape[1] <= cpu_max_other_dimension_decomposition
                )
            )
        )


def print_decompose_pattern(match: Match, inputs: list[torch.fx.Node]):
    node = match.nodes[-1]
    log.debug(
        "Decompose %s with input shape: %s",
        node.target,
        ", ".join(
            str(input.meta["val"].shape) if "val" in input.meta else "None"
            for input in inputs
        ),
    )


@register_graph_pattern(
    CallFunction(aten.bmm, Arg(), Arg()),
    pass_dict=construct_pattern_matcher_pass("decompose_mm_pass"),
)
def decompose_bmm(match: Match, mat1: torch.fx.Node, mat2: torch.fx.Node):
    def repl(mat1, mat2):
        return torch.sum(mat1[:, :, :, None] * mat2[:, None, :, :], dim=-2).to(
            mat1.dtype
        )

    if should_decompose_bmm(mat1, mat2):
        counters["inductor"]["decompose_bmm"] += 1
        # pyrefly: ignore [bad-argument-type]
        match.replace_by_example(repl, [mat1, mat2])
        print_decompose_pattern(match, [mat1, mat2])
        realize_inputs([mat1, mat2])
    return


@register_graph_pattern(
    CallFunction(aten.addmm, Arg(), Arg(), Arg()),
    pass_dict=construct_pattern_matcher_pass("decompose_mm_pass"),
)
def decompose_addmm(
    match: Match,
    mat1: torch.fx.Node,
    mat2: torch.fx.Node,
    mat3: torch.fx.Node,
):
    def repl(mat1, mat2, mat3):
        return (
            torch.sum(mat2[:, :, None] * mat3[None, :, :], dim=-2).to(mat2.dtype) + mat1
        )

    if should_decompose_mm(mat2, mat3):
        counters["inductor"]["decompose_addmm"] += 1
        # pyrefly: ignore [bad-argument-type]
        match.replace_by_example(repl, [mat1, mat2, mat3])
        print_decompose_pattern(match, [mat1, mat2, mat3])
        realize_inputs([mat1, mat2, mat3])
    return


@register_graph_pattern(
    CallFunction(aten.mm, Arg(), Arg()),
    pass_dict=construct_pattern_matcher_pass("decompose_mm_pass"),
)
def decompose_mm(
    match: Match,
    mat1: torch.fx.Node,
    mat2: torch.fx.Node,
):
    def repl(mat1, mat2):
        return torch.sum(mat1[:, :, None] * mat2[None, :, :], dim=-2).to(mat1.dtype)

    if should_decompose_mm(mat1, mat2):
        counters["inductor"]["decompose_mm"] += 1
        # pyrefly: ignore [bad-argument-type]
        match.replace_by_example(repl, [mat1, mat2])
        print_decompose_pattern(match, [mat1, mat2])
        realize_inputs([mat1, mat2])
    return
<|endoftext|>import itertools
import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.fx as fx
from torch.fx.experimental.symbolic_shapes import optimization_hint
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_map_only


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StorageKey:
    storage: torch.UntypedStorage
    device: torch.device

    def __hash__(self) -> int:
        return self.storage._cdata

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StorageKey):
            return False
        return (
            self.storage._cdata == other.storage._cdata and self.device == other.device
        )


class GraphAliasTracker:
    """
    Tracks storage allocation and usage relationships in an FX graph.

    Differentiates between:
    - Fresh allocations: nodes that allocate new storage (not views/aliases)
    - Uses: nodes that use a storage as input
    """

    def __init__(self, nodes: list[fx.Node]):
        # Map from node to the fresh storages it allocates (not views/aliases)
        self.node_to_fresh_allocations: dict[fx.Node, OrderedSet[StorageKey]] = {}

        # Map from storage to the node that originally allocated it
        self.storage_to_allocator: dict[StorageKey, fx.Node] = {}

        # Map from node to all storages it uses as inputs
        self.node_to_storage_uses: dict[fx.Node, OrderedSet[StorageKey]] = {}

        # Map from storage to all nodes that use it
        self.storage_to_uses: dict[StorageKey, OrderedSet[fx.Node]] = defaultdict(
            OrderedSet
        )

        # Map from storage to the last node that uses it
        self.storage_to_last_user: dict[StorageKey, fx.Node] = {}

        # Map from node to storages that have their last use at that node
        self.node_to_storages_last_used: dict[fx.Node, OrderedSet[StorageKey]] = (
            defaultdict(OrderedSet)
        )

        # Track all output storages for each node (for building usage graph)
        self.node_to_output_storages: dict[fx.Node, OrderedSet[StorageKey]] = {}

        # First pass: build storage allocations and track uses
        for node in nodes:
            # Get output storages
            output_storages = self._get_output_storages(node)
            self.node_to_output_storages[node] = output_storages

            # Track fresh allocations
            fresh_allocations: OrderedSet[StorageKey] = OrderedSet()
            for storage_key in output_storages:
                if storage_key not in self.storage_to_allocator:
                    self.storage_to_allocator[storage_key] = node
                    fresh_allocations.add(storage_key)
            self.node_to_fresh_allocations[node] = fresh_allocations

            # Track input storage uses (safe because inputs were already processed)
            input_storages = self._get_input_storages(node)
            self.node_to_storage_uses[node] = input_storages
            for storage_key in input_storages:
                self.storage_to_uses[storage_key].add(node)

        # Second pass: find last users (iterate in reverse)
        for node in reversed(nodes):
            input_storages = self.node_to_storage_uses[node]
            for storage_key in input_storages:
                if storage_key not in self.storage_to_last_user:
                    self.storage_to_last_user[storage_key] = node
                    self.node_to_storages_last_used[node].add(storage_key)

    @staticmethod
    def _get_output_storages(node: fx.Node) -> OrderedSet[StorageKey]:
        """
        Get all storages from a node's outputs.

        Uses pytree to handle arbitrary nested structures.
        """
        val = node.meta.get("val")
        if val is None:
            return OrderedSet()

        storages: OrderedSet[StorageKey] = OrderedSet()

        def collect_storage(tensor: torch._subclasses.FakeTensor) -> None:
            storages.add(StorageKey(tensor.untyped_storage(), tensor.device))

        # Use tree_map_only to handle FakeTensors in nested structures
        tree_map_only(torch._subclasses.FakeTensor, collect_storage, val)

        return storages

    def _get_input_storages(self, node: fx.Node) -> OrderedSet[StorageKey]:
        """
        Get all storages from a node's inputs.
        """
        input_storages: OrderedSet[StorageKey] = OrderedSet()

        for input_node in node.all_input_nodes:
            input_storages.update(self.node_to_output_storages[input_node])

        return input_storages

    def get_fresh_allocations(self, node: fx.Node) -> OrderedSet[StorageKey]:
        """Get all fresh storage allocations by this node (not views/aliases)."""
        return self.node_to_fresh_allocations[node]

    def get_storage_uses(self, node: fx.Node) -> OrderedSet[StorageKey]:
        """Get all storages that this node uses as inputs."""
        return self.node_to_storage_uses[node]

    def get_storages_last_used(
        self,
        node: fx.Node,
    ) -> OrderedSet[StorageKey]:
        """
        Get storages whose last use is at this node.
        """
        return self.node_to_storages_last_used[node]


def _size_of_default(num_bytes: int | torch.SymInt) -> int:
    return optimization_hint(num_bytes)


def device_filter(device: torch.device) -> bool:
    return device.type != "cpu"


def build_memory_profile(
    graph: fx.Graph,
    is_releasable: Callable[[fx.Node], bool],
    size_of: Callable[[int | torch.SymInt], int] | None = None,
) -> list[int]:
    """
    Function to estimate the memory profile of an input FX graph.

    Args:
    - graph (fx.Graph): The input FX graph for which the memory profile
      is to be estimated.
    - is_releasable (Callable[[fx.Node], bool]): A function that
      determines if a node's memory can be released (e.g. primal nodes
      cannot be released).
    - size_of (Callable[[int | torch.SymInt], int]): A function that converts
      byte counts (possibly symbolic) to concrete integers.

    Returns:
    - List[int]: A list representing the memory profile over the execution
      of the graph, where each entry corresponds to the memory usage at
      a particular point in the execution.
    """

    size_of = size_of or _size_of_default
    nodes = list(graph.nodes)
    alias_info = GraphAliasTracker(nodes)

    # Build memory profile
    current_memory = 0

    for node in itertools.chain(
        graph.find_nodes(op="placeholder"), graph.find_nodes(op="get_attr")
    ):
        for storage_key in alias_info.get_fresh_allocations(node):
            if device_filter(storage_key.device):
                current_memory += size_of(storage_key.storage.nbytes())

    memory_profile = [current_memory]

    for node in nodes:
        if node.op in ("placeholder", "get_attr", "output"):
            continue

        # Process allocations
        for storage_key in alias_info.get_fresh_allocations(node):
            if device_filter(storage_key.device):
                current_memory += size_of(storage_key.storage.nbytes())

        memory_profile.append(current_memory)

        # Process deallocations
        # pyrefly: ignore [bad-assignment]
        for storage_key in alias_info.get_storages_last_used(node):
            allocator = alias_info.storage_to_allocator[storage_key]
            if is_releasable(allocator):
                if device_filter(storage_key.device):
                    current_memory -= size_of(storage_key.storage.nbytes())

        memory_profile.append(current_memory)

    return memory_profile


def get_fwd_bwd_interactions(
    fwd_graph: fx.Graph,
    bwd_graph: fx.Graph,
    size_of: Callable[[int | torch.SymInt], int] | None = None,
) -> tuple[int, OrderedSet[str]]:
    """
    Analyze the interactions between the forward (fwd) and backward (bwd) graphs
    to determine memory usage characteristics.

    Args:
    - fwd_graph (fx.Graph): The forward graph representing the forward pass.
    - bwd_graph (fx.Graph): The backward graph representing the backward pass.
    - size_of (Callable[[int | torch.SymInt], int]): A function that converts
      byte counts (possibly symbolic) to concrete integers.

    Returns:
    - tuple[int, OrderedSet[str]]: A tuple containing:
        1. The baseline memory usage during the backward pass, accounting for
           storages that persist from the forward pass (i.e., in fwd output but
           not in bwd input).
        2. A set of node names whose storage cannot be released during the bwd pass.
           These include nodes that use storage from primals or are in bwd input
           but not in fwd output.
    """

    size_of = size_of or _size_of_default

    # Build alias info for forward graph
    fwd_nodes = list(fwd_graph.nodes)
    fwd_alias_info = GraphAliasTracker(fwd_nodes)

    # Identify storages allocated by primal placeholder nodes
    primal_storages: OrderedSet[StorageKey] = OrderedSet()
    for node in fwd_graph.find_nodes(op="placeholder"):
        if node.name.startswith("primals"):
            primal_storages.update(fwd_alias_info.get_fresh_allocations(node))

    # Get storages in forward output
    fwd_output_node = next(iter(reversed(fwd_graph.nodes)))[-1]
    if fwd_output_node.op != "output":
        raise AssertionError(f"expected output node, got op {fwd_output_node.op}")
    fwd_output_storages = fwd_alias_info.get_storage_uses(fwd_output_node)

    # Node names that should not be deleted during memory profile estimation of bwd_graph
    do_not_delete: OrderedSet[str] = OrderedSet()

    # Collect all storages in backward inputs and identify nodes to not delete
    bwd_input_storages: OrderedSet[StorageKey] = OrderedSet()
    for node in bwd_graph.find_nodes(op="placeholder"):
        node_storages = GraphAliasTracker._get_output_storages(node)
        bwd_input_storages.update(node_storages)

        # Check if this node uses primal storage
        if node_storages & primal_storages:
            do_not_delete.add(node.name)

        # Check if this node's storages are not in forward outputs
        # (meaning it's an external input to backward pass)
        if not (node_storages & fwd_output_storages):
            do_not_delete.add(node.name)

    # Calculate baseline memory: storages in fwd output but not in bwd input
    # These storages persist throughout the backward pass
    baseline_storages = fwd_output_storages - bwd_input_storages
    bwd_baseline_memory = 0
    for storage_key in baseline_storages:
        if storage_key.device.type != "cpu":
            bwd_baseline_memory += size_of(storage_key.storage.nbytes())

    return bwd_baseline_memory, do_not_delete


def _is_releasable(n: fx.Node) -> bool:
    # Storages of primals cannot be released during fwd or bwd pass.
    return not n.name.startswith("primals")


def get_peak_memory(
    fwd_graph: fx.Graph,
    bwd_graph: fx.Graph,
) -> int:
    fwd_peak_memory = max(build_memory_profile(fwd_graph, _is_releasable))

    bwd_baseline_memory, bwd_do_not_delete = get_fwd_bwd_interactions(
        fwd_graph,
        bwd_graph,
    )

    def _is_bwd_releasable(n: fx.Node) -> bool:
        # Storages of nodes in bwd_do_not_delete cannot be released
        # during the bwd pass.
        return _is_releasable(n) and n.name not in bwd_do_not_delete

    bwd_peak_memory = bwd_baseline_memory + max(
        build_memory_profile(bwd_graph, _is_bwd_releasable)
    )
    return max(
        fwd_peak_memory,
        bwd_peak_memory,
    )


class MemoryTracker:
    """
    Tracks memory usage for alternative scheduling orders of an FX graph.

    This class enables tracking memory usage as nodes are scheduled in a different
    order than the original graph.
    """

    def __init__(
        self,
        graph: fx.Graph,
        is_releasable: Callable[[fx.Node], bool] | None = None,
        device_filter: Callable[[torch.device], bool] | None = None,
    ):
        """
        Initialize memory tracker for alternative scheduling of the given graph.

        Args:
            graph: FX graph to track memory for under alternative scheduling
            is_releasable: do we consider this input to the graph to release memory
            upon final use, or is allocated for the duration of the graph ?
            by default, we assume all nodes but those that start with "primals" to be releasable
            device_filter: Function to determine which devices to track (default: non-CPU)
        """

        self.graph = graph
        self.nodes = list(graph.nodes)
        self.device_filter = device_filter or (lambda device: device.type != "cpu")
        self.scheduled: OrderedSet[fx.Node] = OrderedSet()

        # Memory tracking using GraphAliasTracker
        self.alias_tracker = GraphAliasTracker(self.nodes)
        self.current_live_storages: OrderedSet[StorageKey] = OrderedSet()
        self.current_memory_bytes = 0
        self.is_releasable = _is_releasable if is_releasable is None else is_releasable

        # Initialize live storages with placeholders and get_attr nodes
        for node in self.nodes:
            if node.op in ("placeholder", "get_attr"):
                fresh_allocations = self.alias_tracker.get_fresh_allocations(node)
                for storage_key in fresh_allocations:
                    if self.device_filter(storage_key.device):
                        self.current_live_storages.add(storage_key)
                        self.current_memory_bytes += self._get_storage_size(storage_key)

        self.peak_memory = self.current_memory_bytes

        log.debug(
            "Memory tracker initialized with initial memory: %d MB",
            self.current_memory_bytes // (1024 * 1024),
        )

    def schedule_node(self, node: fx.Node) -> None:
        """
        Schedule a node and update memory tracking for the new scheduling order.

        Args:
            node: The node being scheduled (potentially out of original order)
        """
        if node in self.scheduled:
            raise AssertionError("should not schedule node twice")
        self.scheduled.add(node)
        self._update_memory_for_node(node)

    def get_current_memory_bytes(self) -> int:
        """Get current live memory in bytes under the current scheduling."""
        return self.current_memory_bytes

    def _get_storage_size(self, storage_key: StorageKey) -> int:
        """Get the size of a storage in bytes, handling symbolic shapes."""
        size_bytes = storage_key.storage.nbytes()
        return optimization_hint(size_bytes)

    def _get_storages_freed_by_node(self, node: fx.Node) -> OrderedSet[StorageKey]:
        """Get storages that would be freed if we schedule this node."""
        freed_storages: OrderedSet[StorageKey] = OrderedSet()

        input_storages = self.alias_tracker.get_storage_uses(node)
        for storage_key in input_storages:
            if not self.device_filter(storage_key.device):
                continue

            # Invariant: if a node uses a storage, it must be live
            if storage_key not in self.current_live_storages:
                raise AssertionError("all input storages should be currently allocated")

            if not self.is_releasable(
                self.alias_tracker.storage_to_allocator[storage_key]
            ):
                continue

            all_uses = self.alias_tracker.storage_to_uses[storage_key]

            # If no more unscheduled uses remain, the storage can be freed
            if all(u in self.scheduled for u in all_uses):
                freed_storages.add(storage_key)

        return freed_storages

    def _update_memory_for_node(self, node: fx.Node) -> None:
        """Update memory tracking when a node is scheduled."""
        if node.op in ("placeholder", "get_attr", "output"):
            return

        # Add fresh allocations
        fresh_allocations = self.alias_tracker.get_fresh_allocations(node)
        alloc_bytes = 0
        for storage_key in fresh_allocations:
            if (
                self.device_filter(storage_key.device)
                and storage_key not in self.current_live_storages
            ):
                size = self._get_storage_size(storage_key)
                self.current_live_storages.add(storage_key)
                self.current_memory_bytes += size
                alloc_bytes += size

        self.peak_memory = max(self.current_memory_bytes, self.peak_memory)

        # Remove storages that are no longer used
        storages_to_free = self._get_storages_freed_by_node(node)
        freed_bytes = 0
        for storage_key in storages_to_free:
            if storage_key in self.current_live_storages:
                size = self._get_storage_size(storage_key)
                self.current_live_storages.remove(storage_key)
                self.current_memory_bytes -= size
                freed_bytes += size

        log.debug(
            "Scheduled %s: memory change %d allocs, %d frees, current memory: %d MB",
            node.name,
            len(fresh_allocations),
            len(storages_to_free),
            self.current_memory_bytes // (1024 * 1024),
        )
<|endoftext|>import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal, Optional

import torch
import torch.fx as fx
from torch._dynamo.utils import counters
from torch._inductor.augmented_graph_helper import AugmentedGraphHelper
from torch._inductor.fx_passes.bucketing import (
    _default_bucket_mode,
    _get_collective_node_from_wait,
    _schedulable_wait_node,
    BucketMode,
    get_full_bucket_key,
    has_mergeable_all_gather_convert_dtype,
    is_all_gather_into_tensor as is_all_gather,
    is_reduce_scatter_tensor as is_reduce_scatter,
)
from torch._inductor.fx_passes.fsdp import is_fsdp_all_gather
from torch._inductor.fx_passes.overlap_scheduling import (
    CollBucket,
    CollectiveInfo,
    get_group_name,
    is_compute_node,
    log as overlap_scheduling_log,
)
from torch._inductor.fx_passes.utils import BitsetAncestors
from torch._logging import trace_structured
from torch.utils._ordered_set import OrderedSet


bucket_log = logging.getLogger(__name__)


@dataclass
class WhyNoBucket:
    name1: str
    name2: str
    reason: str
    args: tuple[Any, ...]

    def __init__(self, node1: fx.Node, node2: fx.Node) -> None:
        self.name1 = node1.name
        self.name2 = node2.name
        self.reason = ""
        self.args = ()

    def __call__(self, reason: str, *args: Any) -> None:
        if bucket_log.isEnabledFor(logging.DEBUG):
            bucket_log.debug(
                "cannot bucket %s with %s: " + reason,
                self.name1,
                self.name2,
                *args,
            )


def is_collective_or_wait(n: fx.Node) -> bool:
    """Check if node is a collective start or wait."""
    if _schedulable_wait_node(n):
        return True
    # Collective starts have exactly one use: the wait_tensor
    if len(n.users) == 1:
        user = next(iter(n.users.keys()))
        if _schedulable_wait_node(user):
            return True
    return False


@dataclass
class PGEvent:
    """
    Represents an important event in a process group timeline. Either
    a collective start, wait, or hiding compute. Each node is linked
    to its prev and next and these dependencies are reflected
    in the augmented graph.

    We want to enforce a sequential ordering of collective starts and waits
    because NCCL collectives on the same process group execute on the same CUDA
    stream, creating implicit dependencies between all operations on that PG.

    A wait of a particular collective will implicitly force realization of all collectives
    enqueued prior to that collective.
    """

    node: fx.Node
    event_type: Literal["compute", "starts", "waits"]
    position: int
    prev: Optional["PGEvent"] = None
    next: Optional["PGEvent"] = None

    @property
    def is_start(self) -> bool:
        return self.event_type == "starts"

    @property
    def is_wait(self) -> bool:
        return self.event_type == "waits"

    @property
    def is_compute(self) -> bool:
        return self.event_type == "compute"

    def unlink(self) -> tuple[Optional["PGEvent"], Optional["PGEvent"]]:
        """Remove this event from the linked list, return (prev, next)."""
        prev_event, next_event = self.prev, self.next
        if self.prev:
            self.prev.next = self.next
        if self.next:
            self.next.prev = self.prev
        self.prev = None
        self.next = None
        return prev_event, next_event

    def insert_between(
        self, prev_event: Optional["PGEvent"], next_event: Optional["PGEvent"]
    ) -> None:
        """Insert this event between prev_event and next_event in the linked list."""
        if prev_event:
            prev_event.next = self
        self.prev = prev_event

        if next_event:
            next_event.prev = self
        self.next = next_event


class OverlapPreservingBucketer:
    """
    Buckets collective operations while preserving compute-collective overlap relationships.
    Uses an augmented graph to track dependencies between compute and collective operations.
    """

    def __init__(
        self,
        graph: fx.Graph,
        collective_info: dict[fx.Node, CollectiveInfo],
        scheduled: OrderedSet[fx.Node],
        max_bucket_memory_gb: float = 1.0,
        max_coll_distance: int = 1000,
        insert_overlap_deps: bool = False,
        collective_bucketing: bool = True,
        bucket_mode: BucketMode | None = None,
        bucket_exposed_first: bool | None = None,
        bucket_only_internode_comms: bool = False,
    ):
        self.graph = graph
        self.collective_info = collective_info
        self.scheduled = scheduled
        self.max_bucket_memory_gb = max_bucket_memory_gb
        self.node_idx = {n: i for i, n in enumerate(scheduled)}
        self.max_coll_distance = max_coll_distance
        self.insert_overlap_deps = insert_overlap_deps
        self.bucket_exposed_first = bucket_exposed_first
        self.bucket_only_internode_comms = bucket_only_internode_comms
        self.bucket_mode = bucket_mode or _default_bucket_mode()
        self.collective_bucketing = collective_bucketing
        self.node_to_event: dict[fx.Node, PGEvent] = {}
        self.all_hiding_nodes: OrderedSet[fx.Node] = OrderedSet()

        # Compute ancestors including original graph edges and hiding interval dependencies
        self.node_ancestors = self._compute_node_ancestors()
        self.aug_graph = AugmentedGraphHelper(self.graph, self.node_ancestors)

        # Build timelines and add constraints to aug_graph
        self.pg_to_timeline_head: dict[str, PGEvent | None] = self.build_timelines()
        self._add_hiding_interval_constraints()

    def _compute_node_ancestors(self) -> BitsetAncestors:
        """Compute ancestor sets including hiding interval deps using bitsets."""
        extra_inputs: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        for start, info in self.collective_info.items():
            if info.is_exposed:
                continue
            for hiding_node in info.hiding_nodes:
                extra_inputs[hiding_node].add(start)
                extra_inputs[info.wait_node].add(hiding_node)

        return BitsetAncestors(list(self.scheduled), extra_inputs=extra_inputs)

    def build_timelines(self) -> dict[str, PGEvent | None]:
        "Construct each process groups ordered series of event"
        all_pgs: OrderedSet[str] = OrderedSet()
        for start in self.collective_info:
            pg = get_group_name(start)
            all_pgs.add(pg)

        pg_timeline: dict[str, PGEvent | None] = {}
        for pg in all_pgs:
            pg_timeline[pg] = self.build_timeline(pg)

        return pg_timeline

    def build_timeline(self, pg: str) -> PGEvent | None:
        """
        Build a timeline of important events (starts, waits, hiding compute) for this process group
        and constrain this ordering in the augmented graph.

        Sequential dependencies are added between all events because NCCL collectives on the same
        process group execute on the same CUDA stream, enforcing LIFO semantics where later-issued
        collectives must complete before earlier ones can finish.
        """

        head = None
        prev_event = None
        position = 0
        hiding_nodes = OrderedSet()

        for node in self.scheduled:
            node_type = None

            # Determine if this node is relevant for this PG
            if node in self.collective_info and get_group_name(node) == pg:
                node_type = "starts"
                hiding_nodes |= self.collective_info[node].hiding_nodes
            elif _schedulable_wait_node(node):
                wait_coll = _get_collective_node_from_wait(node)
                if isinstance(wait_coll, fx.Node) and get_group_name(wait_coll) == pg:
                    node_type = "waits"
                # Wait for a different PG but hiding a collective on this PG
                elif node in hiding_nodes:
                    node_type = "compute"
            elif is_compute_node(node) or node in hiding_nodes:
                node_type = "compute"

            if node_type is None:
                continue

            event = PGEvent(node=node, event_type=node_type, position=position)  # type: ignore[arg-type]

            event.insert_between(prev_event, None)

            # Add sequential dependency to augmented graph
            if prev_event:
                self.aug_graph.add_extra_dep(n=event.node, dep=prev_event.node)
            else:
                head = event

            prev_event = event
            position += 1

        return head

    def _populate_node_to_event(self, pg: str) -> None:
        """Populate node_to_event mapping for a specific PG's timeline."""
        self.node_to_event.clear()
        head = self.pg_to_timeline_head[pg]
        curr = head
        while curr is not None:
            self.node_to_event[curr.node] = curr
            curr = curr.next

    def _add_hiding_interval_constraints(self) -> None:
        """
        Add hiding interval constraints: start -> compute -> wait.
        """
        for start, info in self.collective_info.items():
            if info.is_exposed:
                continue
            for hn in info.hiding_nodes:
                # Enforce: start -> compute -> wait
                self.aug_graph.add_extra_dep(n=hn, dep=start)
                self.aug_graph.add_extra_dep(n=info.wait_node, dep=hn)

            self.all_hiding_nodes |= info.hiding_nodes

    def identify_internode_group_names(self) -> OrderedSet[str]:
        # Identify internode comm groups.
        # Temporary uses FSDP pattern as heuristic for outer groups.
        checked_pgs = OrderedSet()
        internode_pgs = OrderedSet()
        for start in self.collective_info:
            pg = get_group_name(start)
            if is_all_gather(start) and pg not in checked_pgs:
                if is_fsdp_all_gather(start):
                    internode_pgs.add(pg)
                checked_pgs.add(pg)
        return internode_pgs

    def _bucket_collectives_impl(self) -> list[CollBucket]:
        """Find and apply bucket transformations for collectives."""
        pg_collectives: dict[str, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        internode_pgs = self.identify_internode_group_names()

        for start in self.collective_info:
            pg = get_group_name(start)
            pg_collectives[pg].add(start)

        all_buckets: list[CollBucket] = []
        for pg, collectives in pg_collectives.items():
            if self.bucket_only_internode_comms and pg not in internode_pgs:
                continue

            # Populate node_to_event for this PG's timeline
            self._populate_node_to_event(pg)

            grouped_collectives: dict[object, OrderedSet[fx.Node]] = defaultdict(
                OrderedSet
            )
            for start in collectives:
                key = get_full_bucket_key(start, self.bucket_mode)
                if key[1] is not None:
                    grouped_collectives[key].add(start)

            for key, collective_group in grouped_collectives.items():
                bucket_log.debug(
                    "bucketing collective group with key %s: %s",
                    key,
                    [n.name for n in collective_group],
                )
                buckets = self._find_buckets(collective_group, internode_pgs)
                all_buckets.extend(buckets)

        for coll_bucket in all_buckets:
            if len(coll_bucket.collectives) <= 1:
                continue

            counters["inductor"]["collective_buckets"] += 1
            self._apply_bucket(coll_bucket)
        return all_buckets

    def _apply_deps_and_effect_tokens(self) -> None:
        """Apply topological sort and effect tokens to preserve overlap."""
        from torch._dynamo.graph_deduplication import _stable_topological_sort

        # Clean up any remaining erased node references and cycles
        self.aug_graph.remove_erased_extra_deps()
        autofix = torch._inductor.config.aten_distributed_optimizations.overlap_scheduling_autofix_cycles
        self.aug_graph.check_and_maybe_autofix_cyclic_extra_deps(autofix=autofix)
        additional_deps = self.aug_graph.get_all_extra_deps()

        for n, deps in additional_deps.items():
            torch._check(
                not n._erased, lambda: f"Erased node deps not transferred: {n}"
            )
            for d in deps:
                torch._check(
                    not d._erased, lambda: f"Erased node deps not transferred: {d}"
                )

        _stable_topological_sort(self.graph, additional_deps)

        if self.insert_overlap_deps:
            # Filter out collective-to-collective deps (handled by NCCL stream ordering)
            filtered_deps: dict[fx.Node, OrderedSet[fx.Node]] = {}
            for node, deps in additional_deps.items():
                filtered_node_deps: OrderedSet[fx.Node] = OrderedSet()
                for dep in deps:
                    if not (is_collective_or_wait(node) and is_collective_or_wait(dep)):
                        filtered_node_deps.add(dep)
                if filtered_node_deps:
                    filtered_deps[node] = filtered_node_deps

            if filtered_deps:
                from torch._inductor.fx_passes.control_dependencies import (
                    preserve_node_ordering,
                )

                preserve_node_ordering(self.graph, filtered_deps)

    def bucket_collectives(self) -> None:
        """Run the full bucketing and dep application flow."""
        # Step 1: Bucket collectives
        all_buckets: list[CollBucket] | None = None
        if self.collective_bucketing:
            all_buckets = self._bucket_collectives_impl()

        # Step 2: Add control deps
        self._apply_deps_and_effect_tokens()
        self.graph.lint()

        if (
            overlap_scheduling_log.isEnabledFor(logging.DEBUG)
            and all_buckets is not None
        ):
            log_strs: list[str] = []
            stats_num_buckets_per_key = defaultdict(int)
            stats_num_bucketed_collectives_per_key = defaultdict(int)
            stats_num_total_collectives_per_key = defaultdict(int)

            def _bucket_key(node):
                return get_full_bucket_key(node, self.bucket_mode)

            for start, info in self.collective_info.items():
                stats_num_total_collectives_per_key[_bucket_key(start)] += 1

            for i, bucket in enumerate(all_buckets):
                bucket_n = len(bucket.collectives)
                if bucket_n == 0:
                    continue
                node = bucket.collectives[0]
                key = _bucket_key(node)
                stats_num_buckets_per_key[key] += 1
                stats_num_bucketed_collectives_per_key[key] += bucket_n
                log_strs.append(f"bucket[{i}] key:{key} len:{bucket_n}:{bucket}")
                for coll in bucket.collectives:
                    info = self.collective_info[coll]
                    hns = info.hiding_nodes
                    log_strs.append(f"coll:{coll} hiding_nodes:{hns}")

            bucket_log_strs: list[str] = []
            for key, num_buckets in stats_num_buckets_per_key.items():
                num_colls = stats_num_bucketed_collectives_per_key[key]
                bucket_log_strs.append(
                    f"bucket key stats {key}: {num_colls} in {num_buckets}"
                    f" buckets of total:{stats_num_total_collectives_per_key[key]}"
                )
            bucket_log_strs.append("")
            # Add stats to the beginning
            log_strs[:0] = bucket_log_strs
            bucket_logs = "\n".join(log_strs)
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "inductor_fx_passes_overlap_bucketing",
                    "encoding": "string",
                },
                payload_fn=lambda: bucket_logs,
            )

    def _compute_overlap_ratio(self, node: fx.Node) -> float:
        """
        Compute what fraction of the collective's time is hidden by compute.
        Returns 0.0 for fully exposed, 1.0 for fully hidden.
        """
        info = self.collective_info[node]
        if info.estimated_time_ms <= 0:
            return 0.0
        hidden_time = info.estimated_time_ms - info.exposed_time_ms
        return hidden_time / info.estimated_time_ms

    def _should_bucket_exposed_first(
        self,
        collective_group: OrderedSet[fx.Node],
        current_pg: str,
        internode_pgs: OrderedSet[str],
    ) -> bool:
        """Determine whether to bucket exposed collectives first."""
        if self.bucket_exposed_first is None:
            return current_pg in internode_pgs
        return self.bucket_exposed_first

    def _find_buckets(
        self,
        collective_group: OrderedSet[fx.Node],
        internode_pgs: OrderedSet[str],
    ) -> list[CollBucket]:
        """Find valid buckets within a group of similar collectives."""
        max_bucket_bytes = int(self.max_bucket_memory_gb * 1024 * 1024 * 1024)
        buckets = []
        processed: OrderedSet[fx.Node] = OrderedSet()
        if len(collective_group) == 0:
            return []

        current_pg = get_group_name(next(iter(collective_group)))

        bucket_exposed_first = self._should_bucket_exposed_first(
            collective_group, current_pg, internode_pgs
        )
        if bucket_exposed_first:
            # Sort by overlap ratio (ascending) to bucket least hidden collectives first.
            # Exposed collectives benefit most from bucketing since their latency is on the
            # critical path. Prioritizing them also preserves hiding relationships for
            # already-hidden collectives, which have less to gain from bucketing.
            sorted_collectives = sorted(
                collective_group,
                key=lambda n: (self._compute_overlap_ratio(n), self.node_idx[n]),
            )
        else:
            sorted_collectives = sorted(
                collective_group,
                key=lambda n: self.node_idx[n],
            )

        for i, start_node in enumerate(sorted_collectives):
            if start_node in processed:
                continue

            if (
                start_node in self.all_hiding_nodes
                or self.collective_info[start_node].wait_node in self.all_hiding_nodes
            ):
                continue

            # Initialize bucket with first collective
            bucket_info = CollBucket(
                collectives=[start_node],
                total_bytes=self.collective_info[start_node].size_bytes,
            )
            processed.add(start_node)

            # Greedy optimization: stop after consecutive failures
            consecutive_failures = 0
            max_consecutive_failures = 20
            start_node_idx = self.node_idx[start_node]

            # Check candidates in sorted order, break when beyond max distance
            for candidate in sorted_collectives[i + 1 : i + 1 + self.max_coll_distance]:
                candidate_bytes = self.collective_info[candidate].size_bytes
                # proxy on memory use, if we see a too large bucket,
                # don't look for another, later bucket
                if bucket_info.total_bytes + candidate_bytes > max_bucket_bytes:
                    break

                if candidate in processed:
                    continue

                candidate_node_idx = self.node_idx[candidate]
                if (
                    bucket_exposed_first
                    and abs(candidate_node_idx - start_node_idx)
                    > max_consecutive_failures
                ):
                    # Since collectives are sorted by overlap ratio rather than graph
                    # position, skip candidates too far apart in the graph to avoid
                    # creating buckets that block future bucketing opportunities.
                    continue

                if self._can_add_to_bucket(bucket_info, candidate):
                    bucket_info.collectives.append(candidate)
                    bucket_info.total_bytes += candidate_bytes
                    processed.add(candidate)
                    consecutive_failures = 0  # Reset on success
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        break

            if len(bucket_info.collectives) > 1:
                buckets.append(bucket_info)

        return buckets

    def _ancestor_dep(self, n1: fx.Node, n2: fx.Node) -> bool:
        """Check if there's an ancestor relationship between two nodes."""
        return self.node_ancestors.has_dep(n1, n2)

    def _get_intervals(
        self, event: PGEvent
    ) -> tuple[tuple[int, int] | None, list[tuple[int, int]]]:
        """Get (execution_interval, hiding_intervals) for a collective event.

        Returns:
            (execution_interval, hiding_intervals) where:
            - execution_interval is (start_pos, wait_pos) or None
            - hiding_intervals is a list of (start_pos, compute_pos) tuples, one for each hiding node

        Works for both start and wait events by looking up the collective info.
        """
        # For start events, directly use the node
        if event.is_start:
            coll = event.node
        # For wait events, look up the start node from the event's args
        elif event.is_wait:
            coll = _get_collective_node_from_wait(event.node)
            if coll is None:
                return None, []
        else:
            return None, []

        if coll not in self.collective_info:
            return None, []

        info = self.collective_info[coll]
        start_event = self.node_to_event[coll]
        wait_event = self.node_to_event[info.wait_node]

        execution_interval = (start_event.position, wait_event.position)

        hiding_intervals = []
        if info.hiding_nodes:
            for hiding_node in info.hiding_nodes:
                hiding_intervals.append(
                    (
                        start_event.position,
                        self.node_to_event[hiding_node].position,
                    )
                )

        return execution_interval, hiding_intervals

    def _preserves_hiding_intervals(
        self,
        bucket_info: CollBucket,
        candidate: fx.Node,
        start_pos: fx.Node,
        wait_pos: fx.Node,
        why: WhyNoBucket,
    ) -> bool:
        """
        Check that (start_pos, wait_pos) doesn't violate any hiding intervals or collectives.

        Collects all execution and hiding intervals in the affected timeline regions,
        then checks:
        1. All bucket hiding compute stays between new start/wait
        2. No other collective's compute interval is enclosed by bucket execution interval
        3. No other collective's execution interval encloses bucket compute intervals
        """
        # Collect all collectives being bucketed
        all_bucketed_colls = [candidate] + list(bucket_info.collectives)
        all_bucketed_waits = [
            self.collective_info[coll].wait_node for coll in all_bucketed_colls
        ]

        # Collect hiding compute positions for the bucket
        bucket_hiding_compute_positions = []
        for coll in all_bucketed_colls:
            for coll_hiding_node in self.collective_info[coll].hiding_nodes:
                bucket_hiding_compute_positions.append(
                    self.node_to_event[coll_hiding_node].position
                )

        # Get new positions
        new_start_event = self.node_to_event[start_pos]
        new_wait_event = self.node_to_event[wait_pos]

        # Check 1: All bucket hiding compute must be between new start and wait
        for compute_pos in bucket_hiding_compute_positions:
            if not (new_start_event.position < compute_pos < new_wait_event.position):
                why(
                    "hiding compute at pos %d not between start %d and wait %d",
                    compute_pos,
                    new_start_event.position,
                    new_wait_event.position,
                )
                return False

        def get_wait(n: fx.Node) -> fx.Node:
            return self.collective_info[n].wait_node

        def get_pos(n: fx.Node) -> int:
            return self.node_to_event[n].position

        latest_start_pos = max(get_pos(candidate), get_pos(bucket_info.collectives[0]))
        earliest_wait_pos = min(
            get_pos(get_wait(candidate)), get_pos(get_wait(bucket_info.collectives[0]))
        )

        # Bucket execution interval
        bucket_execution_interval = (new_start_event.position, new_wait_event.position)

        # Because collectives on the same PG operate under LIFO semantics,
        # it's only possible for us to force an early realization of an unrelated collective
        # by delaying a start or raising a wait.
        # We search in the interval from old_start -> new_start, to see if would be
        # forcing another collective to be realized prior to its hiding nodes.
        # Similarly, we search from old_wait -> new_wait, in the reverse direction,
        # to check the same thing.

        execution_intervals = [bucket_execution_interval]
        hiding_intervals = [
            (bucket_execution_interval[0], pos)
            for pos in bucket_hiding_compute_positions
        ]

        curr_event = new_start_event.next
        while curr_event is not None and curr_event.position < latest_start_pos:
            if (
                curr_event.node not in all_bucketed_colls
                and curr_event.node not in all_bucketed_waits
            ):
                exec_interval, hiding_interval_list = self._get_intervals(curr_event)
                if exec_interval:
                    execution_intervals.append(exec_interval)
                hiding_intervals.extend(hiding_interval_list)
            curr_event = curr_event.next

        curr_event = new_wait_event.prev
        while curr_event is not None and curr_event.position > earliest_wait_pos:
            if (
                curr_event.node not in all_bucketed_colls
                and curr_event.node not in all_bucketed_waits
            ):
                exec_interval, hiding_interval_list = self._get_intervals(curr_event)
                if exec_interval:
                    execution_intervals.append(exec_interval)
                hiding_intervals.extend(hiding_interval_list)
            curr_event = curr_event.prev

        # Check: no hiding interval should be enclosed by any execution interval
        def enclosed_interval(inner: tuple[int, int], outer: tuple[int, int]) -> bool:
            return outer[0] < inner[0] and inner[1] < outer[1]

        for hiding_interval in hiding_intervals:
            for execution_interval in execution_intervals:
                if enclosed_interval(hiding_interval, execution_interval):
                    why(
                        "hiding interval %s enclosed by execution interval %s",
                        hiding_interval,
                        execution_interval,
                    )
                    return False

        return True

    def remove_from_event(self, node: fx.Node) -> tuple[PGEvent | None, PGEvent | None]:
        """Remove node from timeline and return (prev_event, next_event)."""
        event = self.node_to_event[node]
        if event.is_compute:
            raise AssertionError("Cannot remove compute events from timeline")

        prev_event, next_event = event.unlink()

        # Remove augmented graph dependency
        if prev_event:
            self.aug_graph.remove_extra_dep(n=node, dep=prev_event.node)
        if next_event:
            self.aug_graph.remove_extra_dep(n=next_event.node, dep=node)

        # Add bypass dependency
        if prev_event and next_event:
            self.aug_graph.add_extra_dep(n=next_event.node, dep=prev_event.node)

        return prev_event, next_event

    def restore_to_event(
        self,
        node: fx.Node,
        prev_event: PGEvent | None,
        next_event: PGEvent | None,
    ) -> None:
        """Restore node to timeline after failed merge attempt."""
        event = self.node_to_event[node]

        # Reinsert into linked list
        event.insert_between(prev_event, next_event)
        if prev_event:
            self.aug_graph.add_extra_dep(n=node, dep=prev_event.node)
        if next_event and not prev_event:
            self.aug_graph.add_extra_dep(n=next_event.node, dep=node)

        # Remove bypass dependency
        if prev_event and next_event:
            self.aug_graph.remove_extra_dep(n=next_event.node, dep=prev_event.node)

    def _try_timeline_position(
        self,
        bucket_info: CollBucket,
        candidate: fx.Node,
        start_pos: fx.Node,
        wait_pos: fx.Node,
        why: WhyNoBucket,
    ) -> bool:
        """
        Try a specific timeline position for the candidate.
        Returns True if valid and merges are successful.
        """
        candidate_info = self.collective_info[candidate]
        candidate_wait = candidate_info.wait_node

        # Quick check: does this violate hiding intervals?
        if not self._preserves_hiding_intervals(
            bucket_info, candidate, start_pos, wait_pos, why
        ):
            return False

        # Determine which start needs to move
        existing_coll = bucket_info.collectives[0]
        if start_pos == existing_coll:
            start_to_move = candidate
        else:
            if start_pos != candidate:
                raise AssertionError(
                    f"expected start_pos == candidate, got {start_pos}"
                )
            start_to_move = existing_coll

        # Remove start from timeline
        start_prev, start_next = self.remove_from_event(start_to_move)

        # Check if starts can be merged
        if self.aug_graph.has_path(existing_coll, candidate) or self.aug_graph.has_path(
            candidate, existing_coll
        ):
            # Restore start constraints
            self.restore_to_event(start_to_move, start_prev, start_next)
            why("path exists between starts")
            return False

        # Merge starts
        self.aug_graph.merge_to_set(existing_coll, candidate)

        # Determine which wait needs to move
        existing_wait = self.collective_info[existing_coll].wait_node
        candidate_wait = self.collective_info[candidate].wait_node

        if wait_pos == existing_wait:
            wait_to_move = candidate_wait
        else:
            wait_to_move = existing_wait

        # Remove wait from timeline
        wait_prev, wait_next = self.remove_from_event(wait_to_move)

        # Check if waits can be merged
        if self.aug_graph.has_path(
            existing_wait, candidate_wait
        ) or self.aug_graph.has_path(candidate_wait, existing_wait):
            # Restore wait constraints
            self.restore_to_event(wait_to_move, wait_prev, wait_next)
            # Unmerge the start we just merged
            self.aug_graph.unmerge_node(candidate)
            # Restore start constraints
            self.restore_to_event(start_to_move, start_prev, start_next)
            why("path exists between waits")
            return False

        # Merge waits - success!
        self.aug_graph.merge_to_set(existing_wait, candidate_wait)

        # Update node_to_event for moved nodes
        target_start_event = self.node_to_event[start_pos]
        target_wait_event = self.node_to_event[wait_pos]

        self.node_to_event[candidate] = target_start_event
        self.node_to_event[candidate_wait] = target_wait_event
        self.node_to_event[existing_coll] = target_start_event
        self.node_to_event[existing_wait] = target_wait_event

        return True

    def _has_ancestor_conflicts(
        self, bucket_info: CollBucket, candidate: fx.Node
    ) -> bool:
        """
        Check if candidate has ancestor conflicts with bucket collectives.
        Returns True if there are conflicts.
        """
        candidate_info = self.collective_info[candidate]
        candidate_wait = candidate_info.wait_node

        for coll in bucket_info.collectives:
            if self.node_ancestors.has_dep(coll, candidate):
                return True

            # Check if waits are ancestors of each other
            coll_wait = self.collective_info[coll].wait_node
            if self.node_ancestors.has_dep(coll_wait, candidate_wait):
                return True

            # Check if existing hiding node conflicts with candidate wait
            for old_hiding_node in self.collective_info[coll].hiding_nodes:
                if self.node_ancestors.is_ancestor(candidate_wait, old_hiding_node):
                    return True

            # Check if candidate hiding node conflicts with existing wait
            for new_hiding_node in candidate_info.hiding_nodes:
                if self.node_ancestors.is_ancestor(coll_wait, new_hiding_node):
                    return True

        return False

    def _can_add_to_bucket(
        self,
        bucket_info: CollBucket,
        candidate: fx.Node,
    ) -> bool:
        """
        Check if candidate can be added to bucket without breaking comm/compute overlap.

        Strategy: Try all timeline positions - combinations of [existing_start, candidate_start]
        x [existing_wait, candidate_wait]. For each position, verify:
        1. Hiding intervals preserved - for any (start, hiding_compute, wait) interval, no other
           collective's (start, wait) pair falls between start and hiding_compute, which would
           force realization and break overlap due to LIFO semantics
        2. Topologically valid (no dependency cycles)

        Return True if any timeline position satisfies both constraints.
        """
        existing_coll = bucket_info.collectives[0]
        why = WhyNoBucket(existing_coll, candidate)

        candidate_info = self.collective_info[candidate]

        if (
            candidate in self.all_hiding_nodes
            or candidate_info.wait_node in self.all_hiding_nodes
        ):
            why("nyi: bucketing collective used for overlap")
            return False

        # Step 1: Quick check using precomputed ancestors
        # These ancestors are computed prior to adding augmented dependencies and not updated,
        # so if any of these checks fail then the merge will not be topologically valid
        # even ignoring comm/compute overlap
        if self._has_ancestor_conflicts(bucket_info, candidate):
            why("has ancestor conflicts")
            return False

        # Step 2: Try different rail positions
        existing_wait = self.collective_info[existing_coll].wait_node

        candidate_start = candidate
        candidate_wait = candidate_info.wait_node

        # Try combinations in order of likelihood to succeed
        # (early start, later wait is most likely to work)
        combinations = [
            (
                existing_coll,
                candidate_wait,
            ),  # Move candidate start early, keep wait late
            (
                existing_coll,
                existing_wait,
            ),  # Move candidate start early, move wait early
            (candidate_start, candidate_wait),  # Keep both in place
            (candidate_start, existing_wait),  # Keep start in place, move wait early
        ]

        for i, (start_pos, wait_pos) in enumerate(combinations):
            if self._try_timeline_position(
                bucket_info, candidate, start_pos, wait_pos, why
            ):
                bucket_log.debug(
                    "bucketed %s with %s using timeline position %d: (start=%s, wait=%s)",
                    candidate.name,
                    existing_coll.name,
                    i + 1,
                    start_pos.name,
                    wait_pos.name,
                )
                return True

        why("all timeline positions failed")
        return False

    def _apply_bucket(self, bucket_info: CollBucket) -> None:
        """
        Apply bucketing transformation.

        Dependencies are added to aug_graph.extra_deps and transferred from old nodes.
        """

        from torch._inductor.fx_passes.bucketing import (
            is_all_reduce_tensor,
            merge_all_gather_bucket,
            merge_all_reduce_bucket,
            merge_reduce_scatter_bucket,
        )

        bucket = bucket_info.collectives

        # Collect old nodes BEFORE they're erased
        old_starts = list(bucket)
        old_waits = [self.collective_info[n].wait_node for n in bucket]

        fused_convert_dtypes = []
        for n in old_starts:
            if has_mergeable_all_gather_convert_dtype(n):
                fused_convert_dtypes.append(n.args[0])

        # Find where to place the bucketed operations
        next_node = bucket[0]
        while next_node in bucket:
            next_node = next_node.next

        # Don't use wait_insertion_point - let merge functions place waits naturally
        # The wait_insertion_point feature tries to move waits to a specific location,
        # but this can cause issues when that location is one of the nodes being erased
        # Create bucketed collective (this will erase old nodes)
        if is_all_gather(bucket[0]):
            new_nodes, replacements = merge_all_gather_bucket(
                self.graph,
                bucket,
                insert_before=next_node,
                mode=self.bucket_mode,
            )
        elif is_all_reduce_tensor(bucket[0]):
            new_nodes, replacements = merge_all_reduce_bucket(
                self.graph,
                bucket,
                mode=self.bucket_mode,
                insert_before=next_node,
            )
        else:
            if not is_reduce_scatter(bucket[0]):
                raise AssertionError(f"expected reduce_scatter, got {bucket[0]}")
            new_nodes, replacements = merge_reduce_scatter_bucket(
                self.graph,
                bucket,
                insert_before=next_node,
                mode=self.bucket_mode,
            )

        # Identify the new wait(s) and their collective start in a single pass
        wait_to_start = {
            n: start
            for n in new_nodes
            if (start := _get_collective_node_from_wait(n)) is not None
        }
        new_waits = list(wait_to_start)

        # Create mapping of all erased nodes to their replacements
        erased_to_new: dict[fx.Node, fx.Node | None] = {}
        new_start = wait_to_start[new_waits[0]]
        if len(new_waits) == 1:
            # Standard bucketing: single start + single wait
            new_wait = new_waits[0]
            for old_start in old_starts:
                erased_to_new[old_start] = new_start
            for old_wait in old_waits:
                erased_to_new[old_wait] = new_wait
        else:
            # Coalesced bucketing: single start + N waits (one per original tensor)
            if len(new_waits) != len(old_waits):
                raise AssertionError(
                    f"expected len(new_waits) == len(old_waits), "
                    f"got {len(new_waits)} != {len(old_waits)}"
                )
            for old_start in old_starts:
                erased_to_new[old_start] = new_start
            erased_to_new.update(dict(zip(old_waits, new_waits)))

        # Handle convert_element_type nodes that were fused and erased
        # The bucketed operation may have a _pre_bucket op that handles dtype conversion
        if fused_convert_dtypes:
            # In custom_ops mode, the _pre_bucket_all_gather node handles dtype conversion
            # In default mode, convert nodes are just erased — map them to new_start
            new_convert_dtypes_node = new_start.kwargs.get("out")
            if (
                isinstance(new_convert_dtypes_node, fx.Node)
                and new_convert_dtypes_node.target
                == torch.ops.bucketing._pre_bucket_all_gather.default
            ):
                replacement = new_convert_dtypes_node
            else:
                replacement = new_start

            for n in fused_convert_dtypes:
                erased_to_new[n] = replacement

        # Transfer all dependencies from old nodes to new nodes
        self.aug_graph.transfer_erased_node_deps(erased_to_new)


def finalize_overlap_scheduling(
    gm: fx.GraphModule,
    collective_info: dict[fx.Node, CollectiveInfo],
    scheduled: OrderedSet[fx.Node],
    *,
    collective_bucketing: bool = False,
    insert_overlap_deps: bool = False,
    max_bucket_memory_gb: float = 2.0,
    max_coll_distance: int = 1000,
    bucket_exposed_first: bool | None = None,
    bucket_only_internode_comms: bool = False,
    bucket_mode: BucketMode | None = None,
) -> None:
    """
    Finalize overlap scheduling by applying deps and optionally bucketing.

    This is the main entry point for post-scheduling graph transformations:
    1. Bucket collectives (if collective_bucketing=True)
    2. Apply topological sort and effect tokens

    Args:
        gm: The graph module to modify
        collective_info: Dict mapping collective start nodes to their CollectiveInfo
        scheduled: Ordered set of scheduled nodes
        collective_bucketing: Whether to bucket collectives
        insert_overlap_deps: Whether to insert effect tokens for overlap deps
        max_bucket_memory_gb: Maximum memory for a bucket in GB
        max_coll_distance: Maximum distance for bucketing candidates
    """
    bucketer = OverlapPreservingBucketer(
        graph=gm.graph,
        collective_info=collective_info,
        scheduled=scheduled,
        max_bucket_memory_gb=max_bucket_memory_gb,
        max_coll_distance=max_coll_distance,
        insert_overlap_deps=insert_overlap_deps,
        collective_bucketing=collective_bucketing,
        bucket_exposed_first=bucket_exposed_first,
        bucket_only_internode_comms=bucket_only_internode_comms,
        bucket_mode=bucket_mode,
    )
    bucketer.bucket_collectives()
<|endoftext|># mypy: allow-untyped-defs
import functools
import itertools
import logging
import operator
import typing
from collections import Counter
from collections.abc import Sequence
from typing import Any

import torch
import torch._guards
import torch.utils._pytree as pytree
from torch._dynamo.utils import counters
from torch._higher_order_ops.flex_gemm import _PRESERVE_FLEX_GEMM_GEMM_OP
from torch._inductor.constant_folding import ConstantFolder
from torch._inductor.fx_passes.dedupe_symint_uses import _SymHashingDict
from torch._inductor.utils import get_gpu_type
from torch.fx.experimental.symbolic_shapes import (
    guard_or_false,
    guard_or_true,
    statically_known_true,
)
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._ordered_set import OrderedSet

from .. import config
from ..custom_graph_pass import get_custom_graph_passes
from ..pattern_matcher import (
    Arg,
    CallFunction,
    init_once_fakemode,
    KeywordArg,
    Match,
    MULTIPLE,
    PatternMatcherPass as PatternMatcherPassBase,
    register_graph_pattern,
    stable_topological_sort,
)
from .decompose_mem_bound_mm import check_device
from .replace_random import replace_random_passes


PatternMatcherPass = functools.partial(
    PatternMatcherPassBase, subsystem="joint_graph_passes"
)

log = logging.getLogger(__name__)
early_patterns = PatternMatcherPass()
patterns = PatternMatcherPass()
aten = torch.ops.aten
prims = torch.ops.prims

pass_patterns = [
    patterns,
    PatternMatcherPass(),
]


def _is_lossless_fp_widening_cast(
    src_dtype: torch.dtype, dst_dtype: torch.dtype
) -> bool:
    if src_dtype == dst_dtype:
        return True

    if not (src_dtype.is_floating_point and dst_dtype.is_floating_point):
        return False

    src_info = torch.finfo(src_dtype)
    dst_info = torch.finfo(dst_dtype)

    # A floating-point cast is only pointless if the first conversion cannot
    # discard precision or range from the source values.
    return (
        dst_info.eps <= src_info.eps
        and dst_info.max >= src_info.max
        and dst_info.tiny <= src_info.tiny
    )


@init_once_fakemode
def lazy_init(input_device: torch.device | None = None):
    from .fuse_attention import _sfdp_init
    from .misc_patterns import _misc_patterns_init
    from .pad_mm import _pad_mm_init

    _pad_mm_init(input_device)
    _sfdp_init(input_device)
    _misc_patterns_init(input_device)


def remove_no_ops(
    gm: torch.fx.GraphModule,
    zeros: OrderedSet[torch.fx.Node],
    ones: OrderedSet[torch.fx.Node],
):
    """
    Removes operations that are essentially no-ops e.g. (+ 0, - 0, * 1, / 1)
    """
    with torch.utils._python_dispatch._disable_current_modes():
        graph = gm.graph

        def fake_tensors_eq(t1, t2, fields=("shape", "dtype", "device")):
            if any(not isinstance(t, torch.Tensor) for t in (t1, t2)):
                return False
            for field in fields:
                if getattr(t1, field) != getattr(t2, field):
                    return False
            return True

        def is_mutated(n):
            """Check if a node is mutated by any in-place operation."""
            for user in n.users:
                if user.op != "call_function" or not hasattr(user.target, "_schema"):
                    continue
                for i, arg in enumerate(user.args):
                    if arg is n:
                        schema_arg = user.target._schema.arguments[i]
                        if schema_arg.alias_info and schema_arg.alias_info.is_write:
                            return True
            return False

        def isScalarValue(arg):
            return isinstance(arg, (int, float))

        def replace_no_op(node, replace_input_index):
            replacement = node.args[replace_input_index]

            # https://github.com/pytorch/pytorch/issues/86128 causes
            # non-Tensor inputs even for ops with only Tensor inputs.
            # TODO - decompose/type promote to avoid this
            if not all(isinstance(arg, torch.fx.Node) for arg in node.args):
                if all(isScalarValue(arg) for arg in node.args) or not isinstance(
                    replacement, torch.fx.Node
                ):
                    return

            # https://github.com/pytorch/pytorch/issues/174187
            # Don't replace if the replacement value is mutated in-place.
            # The original node acts as an implicit copy; removing it would
            # cause users to observe the post-mutation value instead.
            if is_mutated(replacement):
                return

            if not fake_tensors_eq(node.meta["val"], replacement.meta["val"]):
                if fake_tensors_eq(
                    node.meta["val"],
                    replacement.meta["val"],
                    ("shape", "device"),
                ):
                    with graph.inserting_after(node):
                        replacement = graph.call_function(
                            torch.ops.prims.convert_element_type.default,
                            args=(replacement, node.meta["val"].dtype),
                        )
                else:
                    return

            node.replace_all_uses_with(replacement)
            replacement.meta.update(node.meta)
            graph.erase_node(node)

        for node in graph.find_nodes(op="call_function", target=aten.add.Tensor):
            if len(node.args) == 2:
                if (
                    not any(
                        e in zeros or (isScalarValue(e) and e == 0) for e in node.args
                    )
                    or node.kwargs.get("alpha", 1) != 1
                ):
                    continue

                replace_index = (
                    1
                    if node.args[0] in zeros
                    or (isScalarValue(node.args[0]) and node.args[0] == 0)
                    else 0
                )
                replacement = node.args[replace_index]
                if isinstance(replacement, torch.fx.Node):
                    val = replacement.meta.get("val")
                    if isinstance(val, torch.Tensor) and val.is_conj():
                        continue
                replace_no_op(node, replace_index)

        for node in graph.find_nodes(op="call_function", target=aten.sub.Tensor):
            if len(node.args) == 2:
                if (
                    not (
                        node.args[1] in zeros
                        or (isScalarValue(node.args[1]) and node.args[1] == 0)
                    )
                    or node.kwargs.get("alpha", 1) != 1
                ):
                    continue

                replace_no_op(node, 0)

        for node in graph.find_nodes(op="call_function", target=aten.mul.Tensor):
            if len(node.args) == 2:
                if not any(
                    e in ones or (isScalarValue(e) and e == 1) for e in node.args
                ):
                    continue

                replace_input_index = (
                    1
                    if node.args[0] in ones
                    or (isScalarValue(node.args[0]) and node.args[0] == 1)
                    else 0
                )
                replace_no_op(node, replace_input_index)

        for node in graph.find_nodes(op="call_function", target=aten.div.Tensor):
            if len(node.args) == 2 and (
                node.args[1] in ones
                or (isScalarValue(node.args[1]) and node.args[1] == 1)
            ):
                replace_no_op(node, 0)

        # meta tensors returned from the graph have no data and can be replaced with empty_strided
        for output_node in graph.find_nodes(op="output"):
            had_meta_return = False

            def visit(n):
                nonlocal had_meta_return
                val = n.meta.get("val")
                if isinstance(val, torch.Tensor) and val.device.type == "meta":
                    with graph.inserting_before(output_node):
                        # size/stride may be symbolic under dynamic shapes;
                        # materialize them so we never pass raw SymInts as args.
                        # Use materialize_symints (roots backed sizes on input
                        # placeholders) rather than create_size_node(n, d), which
                        # would query `n` and pin it alive, blocking the
                        # eliminate_dead_code() that removes this meta tensor.
                        size = graph.materialize_symints(val.size())
                        stride = graph.materialize_symints(val.stride())
                        n.replace_all_uses_with(
                            graph.call_function(
                                torch.ops.aten.empty_strided.default,
                                args=(size, stride),
                                kwargs={"dtype": val.dtype, "device": val.device},
                            )
                        )
                    had_meta_return = True

            torch.fx.map_arg(output_node.args, visit)
            if had_meta_return:
                graph.eliminate_dead_code()


def remove_redundant_views(gm: torch.fx.GraphModule):
    """
    Removes redundant views by reusing existing ones.
    """
    with torch.utils._python_dispatch._disable_current_modes():
        # A dictionary mapping a tensor to all aliased views.
        views: dict[torch.fx.Node, dict[torch.dtype, torch.fx.Node]] = {}
        graph = gm.graph

        for node in graph.find_nodes(
            op="call_function", target=torch.ops.aten.view.dtype
        ):
            src = node.args[0]
            to_type = node.args[1]
            existing_views = views.get(src)
            is_needed = True

            if existing_views:
                # Replace the view with an existing view if available.
                alias = existing_views.get(to_type)
                if alias:
                    is_needed = False
                    node.replace_all_uses_with(alias)
                    alias.meta.update(node.meta)
                    graph.erase_node(node)
            else:
                from_type = src.meta["val"].dtype
                existing_views = {from_type: src}
                views[src] = existing_views

            if is_needed:
                # Save the new alias but do not replace existing one.
                existing_views.setdefault(to_type, node)
                views[node] = existing_views

        # Clean up unused views.
        while True:
            unused_views = [alias for alias in views if not alias.users]
            if len(unused_views) == 0:
                break
            for unused in unused_views:
                views.pop(unused)
                graph.erase_node(unused)


class UniformValueConstantFolder(ConstantFolder):
    """
    Runs constant folding and replaces tensors that have a uniform value
    with a tensor constructor call: aten.full([shape], value, ...)
    """

    def __init__(self, gm, skip_constructors=False) -> None:
        super().__init__(gm, skip_constructors)
        self.node_storages_ptrs: dict[torch.fx.Node, int] = {}
        self.constant_data_ptrs: dict[torch.fx.Node, StorageWeakRef] = {}
        # we may constant fold a tensor which in the graph has a sym size
        # see: [constant folding refining of symints]
        self.node_replacements_shapes: dict[torch.fx.Node, list[int]] = {}

        # initialize symint -> node mapping so that we can
        # use symint nodes in full constructors
        self.symint_nodes = _SymHashingDict()
        for n in self.module.graph.nodes:  # type: ignore[union-attr]
            if "val" in n.meta and isinstance(n.meta["val"], torch.SymInt):
                if n.meta["val"] not in self.symint_nodes:
                    self.symint_nodes[n.meta["val"]] = n

        # reference from torch/_funtorch/partitioners.py:get_default_op_list
        self.view_op_packets = [
            aten.squeeze,
            aten.unsqueeze,
            aten.alias,
            aten.view,
            aten.slice,
            aten.t,
            prims.broadcast_in_dim,
            aten.expand,
            aten.as_strided,
            aten.permute,
        ]

        self.indexing_op_packets = OrderedSet(
            [
                aten.slice,
            ]
        )

        self._add_peephole_patterns()

    def _add_peephole_patterns(self) -> None:
        """
        Add peephole patterns for nodes where we can infer constant value even if some inputs
        of the node are unknown.
        """
        for op in itertools.chain(
            self.module.graph.find_nodes(  # type: ignore[operator, union-attr]
                op="call_function", target=torch.ops.aten.mul.Tensor
            ),
            self.module.graph.find_nodes(  # type: ignore[operator, union-attr]
                op="call_function", target=torch.ops.aten.mul.Scalar
            ),
        ):
            tensor_val = op.meta.get("val", None)
            if not isinstance(tensor_val, torch.Tensor):
                continue

            def is_zero_int(arg: Any) -> bool:
                return isinstance(arg, int) and arg == 0

            if not any(is_zero_int(a) for a in op.args):
                continue

            # x * 0 is only uniformly 0 for integer/bool dtypes. For floating
            # point (and complex) dtypes nan * 0 == nan and (+/-inf) * 0 == nan,
            # so folding x * 0 -> 0 would incorrectly drop NaN/Inf when x is not
            # known to be finite.
            if tensor_val.dtype.is_floating_point or tensor_val.dtype.is_complex:
                continue

            t = torch.full(
                [1],  # shape
                0,  # value
                dtype=tensor_val.dtype,
                device=tensor_val.device,
                pin_memory=False,
            )
            self.add_node_replacement(op, t)

    def _support_dynamic_shape(self):
        return True

    def insertable_tensor_check(self, t: torch.Tensor) -> bool:
        return True

    def add_node_replacement(self, node: torch.fx.Node, tensor: torch.Tensor) -> None:
        self.node_replacements[node] = tensor.flatten()[0].item()
        self.node_replacements_shapes[node] = node.meta["val"].shape
        self.constant_data_ptrs[node] = StorageWeakRef(tensor.untyped_storage())

    def insert_placerholder_values(self, env: dict[torch.fx.Node, Any]) -> None:
        for n in self.module.graph.find_nodes(op="placeholder"):  # type: ignore[operator, union-attr]
            if "val" in n.meta and isinstance(n.meta["val"], torch.SymInt):
                env[n] = n.meta["val"]
            else:
                env[n] = self.unknown_value

    def _deduce_value(self, node: torch.fx.Node):
        # deduce value for full-like nodes
        # 1. for constructors, substitute value is a tensor of size [1]
        # 2. for view ops/indexing, substitute value is the same as the input
        # 3. for pointwise ops, run node to get the substitute value
        # 4. deal with some special ops
        # otherwise, stop deduce value and return unknown value

        # TODO: cat, more indexing
        # TODO - do on cpu to avoid syncs

        # single-elem attrs
        if node.op == "get_attr" or (
            node.op == "call_function"
            and node.target is torch.ops.aten.lift_fresh_copy.default
        ):
            out = super(ConstantFolder, self).run_node(node)
            if isinstance(out, torch.Tensor) and out.numel() == 1:
                return out

        # handle device_put op
        if node.target == prims.device_put.default:
            return super(ConstantFolder, self).run_node(node)

        # constructors ops
        if (
            node.op == "call_function"
            and node.target is aten.full.default
            and len(node.args) == 2
        ):
            args, kwargs = self.fetch_args_kwargs_from_env(node)
            value = args[1]
            # Don't specialize symbolic value.
            if not isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
                new_args = [[1], value]
                return aten.full.default(*new_args, **node.kwargs)

        # handle before view ops because this changes value
        if node.target is aten.view.dtype:
            (input_tensor, output_dtype), kwargs = self.fetch_args_kwargs_from_env(node)
            # view.dtype with different element sizes changes element count
            # (e.g., complex64 [1+0j] viewed as float32 becomes [1.0, 0.0]),
            # making uniform values non-uniform. Also crashes on 0-d tensors.
            if input_tensor.element_size() != output_dtype.itemsize:
                return self.unknown_value
            return super(ConstantFolder, self).run_node(node)

        # view ops, return input tensor, the first argument
        if hasattr(node.target, "overloadpacket") and (
            node.target.overloadpacket in self.view_op_packets
            or node.target.overloadpacket in self.indexing_op_packets
        ):
            if not isinstance(node.args[0], torch.fx.Node):
                raise AssertionError(f"expected fx.Node, got {type(node.args[0])}")
            return self.env[node.args[0]]

        # we don't want to return unknown value for symints so that we can
        # still constant fold through their use in constructors or views
        # if we see them in a pointwise node (e.g., tensor * symint)
        # we will bail
        if "val" in node.meta and isinstance(node.meta["val"], torch.SymInt):
            return node.meta["val"]

        # pointwise ops
        if isinstance(node.target, torch._ops.OpOverload) and (
            torch.Tag.pointwise in node.target.tags
            or node.target is torch.ops.aten.scalar_tensor.default
        ):
            args, kwargs = self.fetch_args_kwargs_from_env(node)
            flattened_inputs = pytree.arg_tree_leaves(*args, **kwargs)

            if any(isinstance(inp, torch.SymInt) for inp in flattened_inputs):
                return self.unknown_value

            # we run the ops with dim 1, so remove memory_format to avoid error
            kwargs = dict(kwargs)
            kwargs.pop("memory_format", None)

            return node.target(*args, **kwargs)

        return self.unknown_value


def _has_self_referential_shape(
    shapes: list[int | torch.fx.Node], node: torch.fx.Node
) -> bool:
    """
    Check if any shape in `shapes` depends on `node`.

    This is used to detect cycles when constant_fold_uniform_value creates a
    replacement full() node whose shape includes a sym_size computed from the
    original tensor being replaced.

    Checks direct args only - shape nodes typically come from sym_size(tensor, dim)
    where tensor is a direct arg.
    """
    for shape_node in shapes:
        if isinstance(shape_node, torch.fx.Node):
            if node in shape_node.args:
                return True
    return False


def constant_fold_uniform_value(gm: torch.fx.GraphModule):
    """Runs constant folding and replaces constants which can be constructed with a single `full` call. Calls into remove_no_ops."""
    with torch.utils._python_dispatch._disable_current_modes():
        aten = torch.ops.aten

        # Constant folding can leak memory, especially with repeated compilation, so we are only going to
        # remove constants which can be replaced with a constructor.
        cf = UniformValueConstantFolder(gm)
        cf.run()

        node_replacements = cf.node_replacements

        # note: [constant folding refining of symints]
        # constant folding will partially evaluate a graph such that values which have dependencies which
        # are entirely known at compile time may also become compile time constants. in some cases,
        # this will include symints which we had not yet previously deduced are guaranteed a
        # constant value and is then deduced in constant folding. an example is:
        # unbacked_symint_eq_11 = torch.full((), 11).item()
        # torch.full((unbacked_symint_eq_11,), 0)
        node_replacements_shapes = cf.node_replacements_shapes

        graph = gm.graph

        zeros = OrderedSet[Any]()
        ones = OrderedSet[Any]()

        # Got failures in `test_is_set_to_cuda` if we change aliasing on constants,
        # so just constant-ify if a Tensor is unaliased
        constant_data_ptr_count: typing.Counter[StorageWeakRef] = Counter()

        for node in cf.node_replacements:
            constant_data_ptr_count[cf.constant_data_ptrs[node]] += 1

        for node, value in node_replacements.items():
            # we don't have a functional way right now of instantiating a non-contiguous tensor with full/zeros/ones right now
            # hasn't shown up to be important yet
            if "val" not in node.meta:
                # This can only happen in AOTI
                continue

            fake_tensor = node.meta["val"]
            if not fake_tensor.is_contiguous(memory_format=torch.contiguous_format):
                continue

            # TODO - not sure about lossy uint->python value->uint conversions
            if fake_tensor.dtype in (
                torch.uint8,
                torch.uint16,
                torch.uint32,
                torch.uint64,
            ):
                continue

            if constant_data_ptr_count[cf.constant_data_ptrs[node]] > 1:
                continue

            with graph.inserting_after(node):
                # the conversion from tensor and back to value can be lossy, just use the original full ctor value
                if (
                    node.op == "call_function"
                    and node.target is aten.full.default
                    and len(node.args) == 2
                ):
                    value = node.args[1]

                # refines symints, see [constant folding refining of symints] above
                for runtime_size, compile_time_size in zip(
                    node_replacements_shapes[node], fake_tensor.shape
                ):
                    torch._check(runtime_size == compile_time_size)

                # replace SymInt as Node before creating a new full node
                # e.g. (1, s0) -> (1, arg0_1)
                node_shape = node_replacements_shapes[node]
                if not all(
                    not isinstance(s, torch.SymInt) or s in cf.symint_nodes
                    for s in node_shape
                ):
                    continue

                shapes = [
                    cf.symint_nodes[s] if isinstance(s, torch.SymInt) else s
                    for s in node_replacements_shapes[node]
                ]

                # Check if any shape depends on a symint that was computed from
                # the node being replaced - this would create a cycle
                if _has_self_referential_shape(shapes, node):
                    continue

                # zeros and ones just get traced into full, so we insert those
                new_node = graph.call_function(
                    aten.full.default,
                    args=(shapes, value),
                    kwargs={
                        "dtype": fake_tensor.dtype,
                        "layout": torch.strided,
                        "device": fake_tensor.device,
                        "pin_memory": node.kwargs.get("pin_memory", False),
                    },
                )

                new_node.meta.update(node.meta)
                node.replace_all_uses_with(new_node)
                graph.erase_node(node)

                if value == 0:
                    zeros.add(new_node)
                elif value == 1:
                    ones.add(new_node)

        remove_no_ops(gm, zeros, ones)
        remove_redundant_views(gm)


def canonicalize_quant_mapping(gm: torch.fx.GraphModule):
    """


    torch.ops.higher_order.invoke_quant_packed(repeated_subgraph0, 'quant_invoke_0_0', (arg0_1, arg1_1));
    ->
    torch.ops.higher_order.invoke_quant(repeated_subgraph0, arg0_1, arg1_1, scheme = 'nf4');
    """
    graph = gm.graph
    invoke_quant_invocations = graph.find_nodes(
        op="call_function", target=torch.ops.higher_order.invoke_quant_packed
    )
    for invoke_quant in invoke_quant_invocations:
        kwargs = dict(invoke_quant.kwargs)

        quant_options_node = kwargs.pop("quant_options", None)
        if quant_options_node is not None:
            if not isinstance(quant_options_node, torch.fx.Node):
                raise AssertionError(
                    f"expected fx.Node, got {type(quant_options_node)}"
                )
            quant_options = torch._higher_order_ops.InvokeQuant(
                *invoke_quant.kwargs["quant_options"].args,
                **invoke_quant.kwargs["quant_options"].kwargs,
            )
        else:
            quant_options = torch._higher_order_ops.InvokeQuant()

        subgraph, *args = invoke_quant.args
        with gm.graph.inserting_before(invoke_quant):
            invoke_quant_replacement = graph.call_function(
                torch._higher_order_ops.invoke_quant,
                (subgraph, *args),
                # pyrefly: ignore [bad-argument-type]
                kwargs,
            )
            invoke_quant_replacement.meta.update(subgraph.meta)
            invoke_quant_replacement.meta["quant_options"] = quant_options

            invoke_quant.replace_all_uses_with(invoke_quant_replacement)
            graph.erase_node(invoke_quant)

            if quant_options_node and len(quant_options_node.users) == 0:
                graph.erase_node(quant_options_node)

            first_user = next(iter(invoke_quant_replacement.users))

            if (
                len(invoke_quant_replacement.users) == 1
                and len(subgraph.users) == 1
                and first_user.target is operator.getitem
                and first_user.args[1] == 0
            ):
                subgraph_graph = getattr(gm, subgraph.target)
                output_node = torch._inductor.utils.output_node(subgraph_graph)
                if not (
                    isinstance(output_node.args[0], (list, tuple))
                    and len(output_node.args[0]) == 1
                ):
                    raise AssertionError(
                        "expected subgraph output to be a single-element list or tuple"
                    )

                unpacked_output = output_node.args[0][0]
                # pyrefly: ignore [bad-argument-type]
                output_node.args = (unpacked_output,)
                if "val" in output_node.meta:
                    output_node.meta["val"] = output_node.meta["val"][0]
                subgraph_graph.recompile()

                invoke_quant_replacement.meta.update(first_user.meta)
                first_user.replace_all_uses_with(invoke_quant_replacement)
                graph.erase_node(first_user)


def canonicalize_aten_ir_passes(gm: torch.fx.GraphModule):
    """
    Canonicalization passes that will run immediately after aot autograd
    tracing. Thsis must be run before all other graph passes.
    """
    canonicalize_quant_mapping(gm)


def joint_graph_passes(
    graph: torch.fx.GraphModule,
    input_device: torch.device | None = None,
):
    """
    Run FX transformations on the joint forwards+backwards graph.
    """
    GraphTransformObserver = functools.partial(
        torch.fx.passes.graph_transform_observer.GraphTransformObserver,
        subsystem="joint_graph_passes",
    )

    lazy_init(input_device)
    count = 0

    # must occur before other passes
    canonicalize_aten_ir_passes(graph)

    for joint_custom_pre_pass in get_custom_graph_passes(config.joint_custom_pre_pass):
        GraphTransformObserver(graph, "joint_custom_pre_pass").apply_graph_pass(
            joint_custom_pre_pass
        )
        count += 1

    from .post_grad import remove_noop_ops

    GraphTransformObserver(graph, "remove_noop_ops").apply_graph_pass(remove_noop_ops)

    if config.joint_graph_constant_folding:
        GraphTransformObserver(graph, "constant_fold_uniform_value").apply_gm_pass(
            constant_fold_uniform_value
        )

    if config.pattern_matcher:
        count += early_patterns.apply(graph.graph)

    # Make sure AutoChunker happens before pad_mm so we don't need
    # to handle padding when searching for chunking patterns.
    if config.auto_chunker.enable:
        from .auto_chunker import CantChunk, chunk

        try:
            graph = chunk(graph)
        except CantChunk:
            auto_chunker_log = torch._logging.getArtifactLogger(
                __name__, "auto_chunker"
            )
            auto_chunker_log.debug("AutoChunker fail.", exc_info=True)

    if config.pattern_matcher:
        for i, patterns in enumerate(pass_patterns):
            maybe_count = GraphTransformObserver(
                graph, f"pass_pattern_{i}"
            ).apply_graph_pass(patterns.apply)
            count += maybe_count if maybe_count is not None else 0

    if not config.fallback_random:
        # not trying into the bisector because decomps may have already affected rng reproducibility
        # we'll instead explicitly turn off the config
        count += replace_random_passes(graph)

    for joint_custom_post_pass in get_custom_graph_passes(
        config.joint_custom_post_pass
    ):
        GraphTransformObserver(graph, "joint_custom_post_pass").apply_graph_pass(
            joint_custom_post_pass
        )
        count += 1

    if count:
        stable_topological_sort(graph.graph)
        graph.graph.lint()
        graph.recompile()
    return graph


@register_graph_pattern(
    CallFunction(
        torch.ops.prims.iota.default,
        KeywordArg("length"),
        start=KeywordArg("start"),
        step=KeywordArg("step"),
        dtype=KeywordArg("dtype"),
        device=KeywordArg("device"),
        requires_grad=KeywordArg("requires_grad"),
    ),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=patterns,
)
def fix_iota_device(match: Match, length, start, step, dtype, device, requires_grad):
    """
    Eager supports:

        aten.index(cuda_tensor, torch.arange(..., device="cpu"))

    But this results in an implicit host-device-copy and breaks cudagraphs.
    Rewrite the arange to use CUDA.
    """
    (node,) = match.nodes
    user_devices = OrderedSet[torch.device]()
    for user in node.users:
        if (
            user.op == "call_function"
            and user.target in (aten.index.Tensor, aten.index_put.default)
            and hasattr(user.meta.get("val"), "device")
        ):
            user_devices.add(user.meta["val"].device)  # type: ignore[union-attr]
        else:
            return  # bail out

    if len(user_devices) == 1 and "val" in node.meta:
        (user_device,) = user_devices
        if device.type != user_device.type:
            repl = match.graph.call_function(
                torch.ops.prims.iota.default,
                (length,),
                {
                    "start": start,
                    "step": step,
                    "dtype": dtype,
                    "device": user_device,
                    "requires_grad": requires_grad,
                },
            )
            repl.meta.update(node.meta)
            repl.meta["val"] = repl.meta["val"].to(user_device)
            node.replace_all_uses_with(repl)
            match.erase_nodes()


@register_graph_pattern(
    CallFunction(
        torch.ops.prims.convert_element_type.default,
        CallFunction(
            torch.ops.prims.convert_element_type.default,
            KeywordArg("arg"),
            KeywordArg("dtype1"),
        ),
        KeywordArg("dtype2"),
    ),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=patterns,
)
def pointless_convert(match: Match, arg, dtype1: torch.dtype, dtype2: torch.dtype):
    """Remove chain of dtype conversions often created by AMP"""
    graph = match.graph
    node = match.output_node()
    allowed = torch.float16, torch.bfloat16, torch.float32, torch.float64
    arg_val = arg.meta.get("val", None)
    if not isinstance(arg_val, torch.Tensor):
        return

    if arg_val.dtype in allowed and dtype1 in allowed and dtype2 in allowed:
        # A narrower intermediate can be worth materializing before upcasting.
        if dtype1.itemsize < dtype2.itemsize:
            return
        if config.emulate_precision_casts and not _is_lossless_fp_widening_cast(
            arg_val.dtype, dtype1
        ):
            return
        repl = graph.call_function(
            torch.ops.prims.convert_element_type.default, (arg, dtype2)
        )
        repl.meta.update(node.meta)
        node.replace_all_uses_with(repl)
        match.erase_nodes()


def definitely_equal(
    old_sizes: Sequence[torch.SymInt | int],
    new_sizes: Sequence[torch.SymInt | torch.fx.Node | int],
) -> bool:
    """
    Leverage guard_or_true/false to compare if two lists of int/symint are equal.
    Useful to compare sizes, strides etc.

    Can handle -1 in new_sizes which happens in the size arguments of a
    view op. old_sizes is supposed to be the tensor shape and should not
    contain -1.

    new_sizes can contains fx.Node when dynamic shape is enabled. In that
    case new_sizes[i].meta['val'] contains the real torch.SymInt.
    """

    num_neg1 = 0

    if len(old_sizes) != len(new_sizes):
        return False

    for lhs_item, rhs_item in zip(old_sizes, new_sizes):
        if isinstance(rhs_item, torch.fx.Node):
            rhs_item = rhs_item.meta["val"]

        if not isinstance(lhs_item, (int, torch.SymInt)):
            raise AssertionError(type(lhs_item))
        if not isinstance(rhs_item, (int, torch.SymInt)):
            raise AssertionError(type(rhs_item))

        # It still makes sense to call guard_or_true/false since lhs_item
        # rhs_item are torch.SymInt rather than sympy expressions when
        # dynamic shape is enabled.
        if guard_or_false(lhs_item == rhs_item):
            continue

        if guard_or_true(rhs_item != -1):
            return False

        num_neg1 += 1

        if num_neg1 > 1:
            return False
    return True


@register_graph_pattern(
    CallFunction(torch.ops.aten.view.default, KeywordArg("arg"), KeywordArg("size")),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=early_patterns,
)
def pointless_view(match: Match, arg, size):
    """Remove no-op view"""
    node = match.output_node()
    arg_size = list(node.args[0].meta["val"].shape)  # type: ignore[union-attr]
    if definitely_equal(arg_size, size):
        node.replace_all_uses_with(node.args[0])  # type: ignore[arg-type]
        match.erase_nodes()


@register_graph_pattern(
    CallFunction(
        aten.view.default,
        CallFunction(aten.view.default, KeywordArg("arg"), KeywordArg("size1")),
        KeywordArg("size2"),
    ),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=early_patterns,
)
def pointless_view_pair(match: Match, arg, size1, size2):
    """
    Remove a pair of views that are pointless.
    """
    node = match.output_node()
    arg_size = list(arg.meta["val"].shape)
    if definitely_equal(arg_size, size2):
        node.replace_all_uses_with(arg)
        match.erase_nodes()
        counters["inductor"]["removed_pointless_view_pair"] += 1


@register_graph_pattern(
    CallFunction(
        aten.permute.default,
        CallFunction(aten.permute.default, KeywordArg("arg"), KeywordArg("perm1")),
        KeywordArg("perm2"),
    ),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=early_patterns,
)
def pointless_permute_pair(match: Match, arg, perm1, perm2):
    rank = len(perm1)
    if len(perm2) != rank:
        raise AssertionError(f"expected len(perm2) == {rank}, got {len(perm2)}")

    for i in range(rank):
        if perm1[perm2[i]] != i:
            return  # bail out
    node = match.output_node()
    node.replace_all_uses_with(arg)
    match.erase_nodes()


@register_graph_pattern(
    CallFunction(
        aten.bmm,
        Arg(),
        Arg(),
    ),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=patterns,
)
def bmm_to_mm(match: Match, mat1: torch.fx.Node, mat2: torch.fx.Node):
    """Convert bmm to mm when batch size is 1"""
    # See Note [Preserving FlexGEMM body GEMMs].
    if match.output_node().meta.get(_PRESERVE_FLEX_GEMM_GEMM_OP):
        return

    def repl(a, b):
        return torch.mm(a.squeeze(0), b.squeeze(0)).unsqueeze(0)

    if (
        check_device(mat1.meta["val"], mat2.meta["val"], get_gpu_type())
        and statically_known_true(mat1.meta["val"].shape[0] == 1)
        and statically_known_true(mat2.meta["val"].shape[0] == 1)
    ):
        # pyrefly: ignore [bad-argument-type]
        match.replace_by_example(repl, [mat1, mat2])


# When softmax is used with temperature or other scaling, we get the pattern
#
#   scale(x) - scale(x).amax(dim, keepdim=True)
#
# which is expected to be at most zero, but we may end up with numerical
# discrepancies # between the recomputed values of scale(x) inside and out
# of the reduction, # depending on compiler optimizations, e.g. use of fma
# instructions.
#
# Here we replace it with the mathematically equivalent,
#
#   scale(x - x.amax(dim, keepdim=True))
#
# which is more stable as we only compute the scaling once.
#
# NOTE: This pattern must come after fused attention matching!


def _partial_softmax_pattern(linear_func, reverse=False, to_dtype=False):
    # Allow matching inp * other and other * input
    if reverse:
        scaled = CallFunction(
            linear_func, KeywordArg("other"), KeywordArg("inp"), _users=MULTIPLE
        )
    else:
        scaled = CallFunction(
            linear_func, KeywordArg("inp"), KeywordArg("other"), _users=MULTIPLE
        )
    if to_dtype:
        scaled = CallFunction(
            prims.convert_element_type, scaled, KeywordArg("dtype"), _users=MULTIPLE
        )
    amax = CallFunction(
        aten.amax.default, scaled, KeywordArg("dim"), KeywordArg("keepdim")
    )
    return CallFunction(aten.sub.Tensor, scaled, amax)


def _preserve_scaled_softmax_nonfinite_semantics(scaled, stable, dim, keepdim):
    # This pattern also matches the raw scaled - amax(scaled) expression.  Only
    # use the stable form when it preserves that expression's nonfinite behavior.
    original = scaled - torch.amax(scaled, dim=dim, keepdim=keepdim)
    finite_scaled = torch.all(torch.isfinite(scaled), dim=dim, keepdim=True)
    return torch.where(finite_scaled, stable, original)


def _other_is_broadcasted_in_dim(match):
    # Check that the scaling factor is constant across the reduction dim,
    # so scaling doesn't change which index corresponds to the maximum value
    other = match.kwargs["other"]
    if isinstance(other, (int, float)):
        return True

    inp = match.kwargs["inp"]
    if not all(isinstance(x, torch.fx.Node) for x in (inp, other)):
        return False

    inp_example = inp.meta["val"]
    other_example = other.meta["val"]
    if isinstance(other_example, (torch.SymInt, torch.SymFloat)):
        return True

    if not all(isinstance(x, torch.Tensor) for x in (inp_example, other_example)):
        return False

    inp_ndim = inp_example.ndim
    other_shape = other_example.shape
    if inp_ndim < len(other_shape):
        return False

    # Pad other_shape to the same ndim as inp
    other_shape = [1] * (inp_ndim - len(other_shape)) + list(other_shape)

    dim = match.kwargs["dim"]
    if isinstance(dim, int):
        dim = (dim,)

    if any(d >= len(other_shape) for d in dim):
        return False

    return all(statically_known_true(other_shape[d] == 1) for d in dim)


def mul_softmax_pattern(match: Match, *, inp, other, dim, keepdim, dtype=None):
    def repl(inp, other):
        scaled = inp * other
        if dtype is not None:
            scaled = scaled.to(dtype)
            inp = inp.to(dtype)

        sign: int | float | torch.Tensor
        if isinstance(other, (int, float, torch.SymInt, torch.SymFloat)):
            sign = 1 if other >= 0 else -1
        else:
            one = torch.scalar_tensor(1, dtype=inp.dtype, device=inp.device)
            sign = torch.where(other >= 0, one, -one)

        inp = inp * sign
        max_ = torch.amax(inp, dim=dim, keepdim=keepdim)

        stable = (inp - max_) * (sign * other)
        return _preserve_scaled_softmax_nonfinite_semantics(
            scaled, stable, dim, keepdim
        )

    # pyrefly: ignore [bad-argument-type]
    match.replace_by_example(repl, [inp, other])


for reverse, to_dtype in itertools.product((False, True), repeat=2):
    register_graph_pattern(
        _partial_softmax_pattern(aten.mul.Tensor, reverse=reverse, to_dtype=to_dtype),
        # pyrefly: ignore [bad-argument-type]
        pass_dict=pass_patterns[1],
        extra_check=_other_is_broadcasted_in_dim,
    )(mul_softmax_pattern)


def div_softmax_pattern(match: Match, *, inp, other, dim, keepdim, dtype=None):
    def repl(inp, other):
        scaled = inp / other
        if dtype is not None:
            scaled = scaled.to(dtype)
            inp = inp.to(dtype)

        sign: int | float | torch.Tensor
        if isinstance(other, (int, float, torch.SymInt, torch.SymFloat)):
            sign = 1 if other >= 0 else -1
        else:
            one = torch.scalar_tensor(1, dtype=inp.dtype, device=inp.device)
            sign = torch.where(other >= 0, one, -one)

        inp = inp * sign
        max_ = torch.amax(inp, dim=dim, keepdim=keepdim)

        stable = (inp - max_) / (sign * other)
        return _preserve_scaled_softmax_nonfinite_semantics(
            scaled, stable, dim, keepdim
        )

    # pyrefly: ignore [bad-argument-type]
    match.replace_by_example(repl, [inp, other])


for to_dtype in (False, True):
    register_graph_pattern(
        _partial_softmax_pattern(aten.div.Tensor, to_dtype=to_dtype),
        # pyrefly: ignore [bad-argument-type]
        pass_dict=pass_patterns[1],
        extra_check=_other_is_broadcasted_in_dim,
    )(div_softmax_pattern)


def scatter_upon_const_tensor_extra_check(m):
    if not config.optimize_scatter_upon_const_tensor:
        return False
    full_shape = m.kwargs["shape"]
    selector = m.kwargs["selector"]
    dim = m.kwargs["dim"]
    if dim < 0:
        dim += len(full_shape)

    selector_ft = selector.meta["val"]
    if selector_ft.dim() != len(full_shape):
        raise AssertionError(
            f"expected selector_ft.dim() == {len(full_shape)}, got {selector_ft.dim()}"
        )

    for idx, select_sz, full_sz in zip(
        itertools.count(), selector_ft.shape, full_shape
    ):
        if idx == dim:
            continue

        # TODO: the pattern can be updated to support the case that index tensor
        # is shorter. But that will need a more complex condition expression
        # especially for multi-dimensional tensors.
        # Skip it for now.
        if isinstance(full_sz, torch.fx.Node):
            full_sz = full_sz.meta["val"]
        if select_sz < full_sz:
            return False

    # Actually we can support small size larger than 1. It would be a bit
    # tedious. E.g., we load all the index values (not many) and compare
    # them with the position in tensor to decide what value to return.
    return selector_ft.size(dim) == 1


@register_graph_pattern(
    CallFunction(
        aten.scatter.value,
        CallFunction(
            aten.full,
            KeywordArg("shape"),
            KeywordArg("background_val"),
            dtype=KeywordArg("dtype"),
        ),
        KeywordArg("dim"),
        KeywordArg("selector"),
        KeywordArg("val"),  # scalar value
    ),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=patterns,
    extra_check=scatter_upon_const_tensor_extra_check,
)
def scatter_upon_const_tensor(
    match: Match, shape, background_val, dtype, dim, selector, val
):
    """
    Match the pattern of full+scatter into a pointwise operation in joint graph.

    TODO: Right now the scatter value must be a scalar. But we could support it
    when it is a tensor as well.
    """
    from torch._inductor import metrics

    # pyrefly: ignore  # bad-assignment
    metrics.num_matches_for_scatter_upon_const_tensor += 1

    # Create a replacement that uses torch.where for the pointwise operation
    def repl_fn(shape, background_val, dim, selector, val):
        # Create a tensor of indices for the scatter dimension
        length = shape[dim]
        indices = torch.arange(length, device=selector.device, dtype=torch.int64)

        # Reshape indices to have size 'length' at dim, then broadcast
        view_shape = [1] * len(shape)
        view_shape[dim] = length
        indices_view = indices.view(*view_shape)

        # Broadcast selector to match full tensor shape
        selector_expanded = selector.expand(shape)

        # Create a mask for where to scatter
        mask = selector_expanded == indices_view

        # Use torch.where to implement the scatter pointwise operation.
        # When val and background_val are Python scalars, torch.where promotes
        # the result to the default floating dtype (float32), which loses the
        # const tensor's dtype. Cast back to dtype so the rewrite preserves
        # aten.scatter.value semantics (the result has self's dtype, with the
        # scalar value cast to it).
        return torch.where(mask, val, background_val).to(dtype)

    # replace the scatter operation with pointwise equivalent
    # pyrefly: ignore [bad-argument-type]
    match.replace_by_example(repl_fn, [shape, background_val, dim, selector, val])
<|endoftext|># mypy: allow-untyped-defs
import functools

import torch
from torch._dynamo.utils import counters
from torch._ops import OpOverload, OpOverloadPacket
from torch.utils._ordered_set import OrderedSet

from ..pattern_matcher import fwd_only, register_replacement
from ..utils import is_nvidia_sm100_or_later


aten = torch.ops.aten


@functools.cache
def _misc_patterns_init(input_device: torch.device | None = None):
    from .joint_graph import patterns as joint_graph_patterns
    from .post_grad import pass_patterns as post_grad_patterns_all

    post_grad_patterns = post_grad_patterns_all[1]  # medium priority

    if input_device:
        device = str(input_device)
    else:
        if torch.cuda.is_available():
            # workaround https://github.com/pytorch/pytorch/issues/97894
            device = "cuda"
        else:
            device = "cpu"

    # These patterns do 2 things
    # 1. Since we know that index is completely unique, we can codegen it using
    # stores instead of atomic adds, which is quite a bit faster.
    # 2. Also, since we are guaranteed that they are completely within bounds,
    # we can use unsafe indexing and skip debug asserts
    def randperm_index_add_pattern(x, y):
        index = torch.randperm(x.shape[0], device=x.device)[: y.shape[0]]
        return torch.index_add(x, dim=0, source=y, index=index), index

    def randperm_index_add_replacement(x, y):
        index = torch.randperm(x.shape[0], device=x.device)[: y.shape[0]]
        return (
            torch.ops.aten._unsafe_index_put(
                x, (index,), aten._unsafe_index(x, (index,)) + y, accumulate=False
            ),
            index,
        )

    register_replacement(
        # pyrefly: ignore [bad-argument-type]
        randperm_index_add_pattern,
        # pyrefly: ignore [bad-argument-type]
        randperm_index_add_replacement,
        [torch.empty(4, 8, device=device), torch.empty(2, 8, device=device)],
        # pyrefly: ignore [bad-argument-type]
        fwd_only,
        # pyrefly: ignore [bad-argument-type]
        [post_grad_patterns, joint_graph_patterns],
        skip_duplicates=True,
    )

    def randperm_index_full_pattern(x):
        index = torch.randperm(x.shape[0], device=x.device)
        return torch.ops.aten.index(x, (index,)), index

    def randperm_index_full_replacement(x):
        index = torch.randperm(x.shape[0], device=x.device)
        return torch.ops.aten._unsafe_index(x, (index,)), index

    register_replacement(
        # pyrefly: ignore [bad-argument-type]
        randperm_index_full_pattern,
        # pyrefly: ignore [bad-argument-type]
        randperm_index_full_replacement,
        [torch.empty(4, 8, device=device)],
        # pyrefly: ignore [bad-argument-type]
        fwd_only,
        # pyrefly: ignore [bad-argument-type]
        [post_grad_patterns, joint_graph_patterns],
        skip_duplicates=True,
    )

    def randperm_index_pattern(x, slice_shape):
        index = torch.randperm(x.shape[0], device=x.device)[:slice_shape]
        return torch.ops.aten.index(x, (index,)), index

    def randperm_index_replacement(x, slice_shape):
        index = torch.randperm(x.shape[0], device=x.device)[:slice_shape]
        return torch.ops.aten._unsafe_index(x, (index,)), index

    register_replacement(
        # pyrefly: ignore [bad-argument-type]
        randperm_index_pattern,
        # pyrefly: ignore [bad-argument-type]
        randperm_index_replacement,
        [torch.empty(4, 8, device=device)],
        # pyrefly: ignore [bad-argument-type]
        fwd_only,
        # pyrefly: ignore [bad-argument-type]
        [post_grad_patterns, joint_graph_patterns],
        # Keep this smaller than the example input dim so tracing preserves the slice.
        scalar_workaround={"slice_shape": 2},
        skip_duplicates=True,
    )

    # Pattern: e8m0 extraction with ceiling rounding (for MX format scaling)
    if device.startswith("cuda"):

        def e8m0_extra_check(match):
            inp = match.kwargs.get("inp")
            if inp is None:
                return False
            inp_val = inp.meta.get("val")
            return (
                inp_val is not None
                and inp_val.device.type == "cuda"
                and inp_val.dtype == torch.float32
            )

        is_sm100_plus = is_nvidia_sm100_or_later()

        if is_sm100_plus:
            from .. import inductor_prims

            # Pattern 1: Bit manipulation approach (NVIDIA SM100+ only - uses PTX instruction)
            def e8m0_rceil_pattern(inp):
                inp_bits = inp.view(torch.int32)
                biased_exp = (inp_bits >> 23) & 0xFF
                mantissa = inp_bits & 0x7FFFFF
                needs_round_up = mantissa != 0
                e8m0_biased = biased_exp + needs_round_up.to(torch.int32)
                e8m0_biased = torch.clamp(e8m0_biased, 0, 255)
                return e8m0_biased.to(torch.uint8)

            def e8m0_rceil_replacement(inp):
                return inductor_prims.cvt_e8m0_rceil(inp)

            register_replacement(
                # pyrefly: ignore [bad-argument-type]
                e8m0_rceil_pattern,
                # pyrefly: ignore [bad-argument-type]
                e8m0_rceil_replacement,
                [torch.randn(32, device="cuda", dtype=torch.float32)],
                # pyrefly: ignore [bad-argument-type]
                fwd_only,
                # pyrefly: ignore [bad-argument-type]
                [post_grad_patterns],
                extra_check=e8m0_extra_check,
                skip_duplicates=True,
            )

        # Pattern 2: log2 + ceil approach (used by torchao MX formats)
        # Matches: (clamp(ceil(log2(x)), -127, 127) + 127).to(uint8)
        #
        # Registered on ALL CUDA hardware. On NVIDIA SM100+ uses the PTX instruction;
        # on earlier hardware uses exact IEEE 754 bit-manipulation.
        #
        # The bit-manipulation replacement is preferred over the software
        # log2+ceil because Triton's fused kernels can produce inputs that
        # differ by ~1 ULP from eager mode (e.g. due to FMA or libdevice
        # polynomial differences in erf-based GELU). When such an input falls
        # exactly on a power-of-2 boundary, software log2 may round the result
        # to the integer below, making ceil return the wrong value and the
        # uint8 output differ by 1.  The bit-manipulation approach reads the
        # IEEE 754 biased exponent directly, which is always correct for any
        # representable positive normal float32 value.
        E8M0_BIAS = 127

        def e8m0_rceil_log2_pattern(inp):
            log2_val = torch.log2(inp)
            ceil_val = torch.ceil(log2_val)
            clamped = torch.clamp(ceil_val, min=-E8M0_BIAS, max=E8M0_BIAS)
            biased = clamped + E8M0_BIAS
            return biased.to(torch.uint8)

        if is_sm100_plus:

            def e8m0_rceil_log2_replacement(inp):
                return inductor_prims.cvt_e8m0_rceil(inp)

        else:
            # Bit-manipulation fallback: extract IEEE 754 biased exponent with
            # ceiling rounding.  Equivalent to clamp(ceil(log2(inp)), -127, 127)
            # + 127 for all positive normal float32 values, but avoids software
            # log2 imprecision near exact powers of 2.
            # Clamp to [0, 254] to match the satfinite semantics of the original
            # pattern (clamp(ceil_val, -127, 127) + 127 gives at most 254).
            def e8m0_rceil_log2_replacement(inp):
                inp_bits = inp.view(torch.int32)
                biased_exp = (inp_bits >> 23) & 0xFF
                mantissa = inp_bits & 0x7FFFFF
                needs_round_up = mantissa != 0
                e8m0_biased = biased_exp + needs_round_up.to(torch.int32)
                e8m0_biased = torch.clamp(e8m0_biased, 0, 254)
                return e8m0_biased.to(torch.uint8)

        register_replacement(
            # pyrefly: ignore [bad-argument-type]
            e8m0_rceil_log2_pattern,
            # pyrefly: ignore [bad-argument-type]
            e8m0_rceil_log2_replacement,
            [torch.randn(32, device="cuda", dtype=torch.float32).abs() + 1e-10],
            # pyrefly: ignore [bad-argument-type]
            fwd_only,
            # pyrefly: ignore [bad-argument-type]
            [post_grad_patterns],
            extra_check=e8m0_extra_check,
            skip_duplicates=True,
        )

    # TODO: Add pattern for cvt.rn.bf16x2.ue8m0x2 (e8m0 -> bf16 conversion)
    # This is the inverse operation for MX format dequantization


class NumpyCompatNormalization:
    numpy_compat: dict[str, tuple[str, ...]] = {
        "dim": ("axis",),
        "keepdim": ("keepdims",),
        "input": ("x", "a", "x1"),
        "other": ("x2",),
    }
    inverse_mapping: dict[str, str]
    cache: dict["torch.fx.graph.Target", OrderedSet[str]]

    def __init__(self) -> None:
        self.cache = {}  # callable -> tuple of replaceable args e.g. ["axis"]
        self.inverse_mapping = {}
        for actual_kwarg, numpy_kwargs in self.numpy_compat.items():
            for numpy_kwarg in numpy_kwargs:
                if numpy_kwarg in self.inverse_mapping:
                    raise AssertionError(
                        f"duplicate numpy kwarg mapping for {numpy_kwarg}"
                    )
                self.inverse_mapping[numpy_kwarg] = actual_kwarg

    def __call__(self, graph: torch.fx.Graph):
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            if isinstance(node.target, (OpOverload, OpOverloadPacket)):
                # only applies to torch ops; e.g. torch.stack(axis=1) works, torch.ops.aten.stack(axis=1) doesn't.
                continue
            kwargs = node.kwargs

            if node.target in self.cache:
                replaceable_kwargs = self.cache[node.target]
            else:
                signatures = torch.fx.operator_schemas.get_signature_for_torch_op(
                    node.target
                )
                signatures = () if signatures is None else signatures
                replaceable_kwargs = OrderedSet()
                for sig in signatures:
                    for param_name in sig.parameters:
                        if param_name in self.numpy_compat:
                            replaceable_kwargs.update(self.numpy_compat[param_name])

                self.cache[node.target] = replaceable_kwargs

            if not replaceable_kwargs:
                continue

            new_kwargs = {}
            kwargs_changed = False
            for k, v in kwargs.items():
                if k in replaceable_kwargs:
                    kwargs_changed = True
                    new_kwargs[self.inverse_mapping[k]] = v
                else:
                    new_kwargs[k] = v

            if kwargs_changed:
                node.kwargs = torch.fx.immutable_collections.immutable_dict(new_kwargs)
                counters["inductor"]["numpy_compat_normalization"] += 1


numpy_compat_normalization = NumpyCompatNormalization()
<|endoftext|># mypy: allow-untyped-defs
import functools
import itertools

import torch

from ..._dynamo.utils import counters
from .. import config
from ..pattern_matcher import Arg, CallFunction, KeywordArg
from .freezing_patterns import register_binary_folding_pattern


aten = torch.ops.aten
prims = torch.ops.prims


def mark_mixed_dtype(computation_node):
    computation_node_dtype = computation_node.meta["val"].dtype
    if computation_node_dtype not in (torch.float16, torch.bfloat16):
        return

    if len(computation_node.users) != 1:
        return

    computation_node_user = next(iter(computation_node.users.keys()))
    if not isinstance(computation_node_user.meta["val"], torch.Tensor):
        return

    if computation_node_user.meta["val"].dtype != torch.float32:
        return

    while computation_node_user.target in _binary_ops:
        if len(computation_node_user.users) != 1:
            return

        computation_node_user = next(iter(computation_node_user.users.keys()))

    if computation_node_user.target != prims.convert_element_type.default:
        return

    computation_node.meta["_allow_mixed_dtype_folding"] = computation_node_dtype


def mark_mixed_dtype_allowed_computation_ops(gm):
    """
    Mark convolutions/linear which we will binary fold even with mixed precision constants. We constant fold in the higher precision
    for better accuracy and then recover the original precision after.
    """
    for target in [aten.convolution.default, aten.addmm.default, aten.mm.default]:
        for node in gm.graph.find_nodes(op="call_function", target=target):
            mark_mixed_dtype(node)


def recover_original_precision_folded_computation_ops(gm):
    """
    After binary folding conv/linear weights and biases to a higher dtype, recover the original precision they were in.
    """
    graph = gm.graph
    for target, idx in (
        (aten.convolution.default, (1, 2)),
        (aten.addmm.default, (0, 2)),
        (aten.mm.default, (1,)),
    ):
        for node in graph.find_nodes(op="call_function", target=target):
            orig_dtype = node.meta.get("_allow_mixed_dtype_folding", None)
            if orig_dtype is None:
                continue

            with graph.inserting_before(node):
                for i in idx:
                    old_input = node.args[i]
                    if old_input is None:
                        continue

                    new_input = graph.create_node(
                        "call_function",
                        prims.convert_element_type.default,
                        (old_input, orig_dtype),
                    )
                    node.replace_input_with(old_input, new_input)


_binary_ops = [aten.add.Tensor, aten.sub.Tensor, aten.mul.Tensor, aten.div.Tensor]


@functools.cache
def binary_folding_init():
    _conv_args = [Arg() for _ in range(9)]
    _addmm_args = [Arg() for _ in range(3)]
    _mm_args = [Arg() for _ in range(2)]
    _computation_ops = [aten.convolution.default, aten.addmm.default, aten.mm.default]
    _computation_calls = [
        CallFunction(aten.convolution.default, *_conv_args, _users=1),
        CallFunction(aten.addmm.default, *_addmm_args, _users=1),
        CallFunction(
            aten.reshape.default,
            CallFunction(aten.addmm.default, *_addmm_args, _users=1),
            Arg(),
            _users=1,
        ),
        CallFunction(aten.mm.default, *_mm_args, _users=1),
        CallFunction(
            aten.reshape.default,
