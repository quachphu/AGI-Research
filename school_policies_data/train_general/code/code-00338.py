        consumers = []
        if local_reduce.feeds_main:
            consumers.append("main")
        if store is not None:
            consumers.append("returned")
        normalized = analysis.local_reduce.graph.normalized_nodes.get(
            local_reduce.match.value_node
        )
        dataflow = (
            str(local_reduce.match.value_node.target)
            if normalized is None
            else _format_normalized_dataflow(local_reduce.match.value_node, normalized)
        )
        lines.extend(
            (
                "local_reduction:",
                f"  value: {local_reduce.match.value_node.name}",
                f"  dataflow: {dataflow}",
                f"  geometry: {_format_geometry(local_reduce.match.geometry)}",
                f"  consumers: {' + '.join(consumers)}",
            )
        )
        if store is not None:
            lines.extend(
                (
                    f"  returned_as: {store.node.name}",
                    "  output_layout: dense",
                )
            )

    lines.append("")
    _append_items(
        lines,
        "config_constraints",
        map(_format_geometry, analysis.required_geometries),
    )
    return "\n".join(lines)


def format_flex_gemm_analysis_details(
    analysis: "FlexGemmEpilogueAnalysis",
) -> str:
    """Render normalized nodes and recognizer records for deep debugging."""
    lines: list[str] = []
    for label, values in (
        ("normalized_nodes", analysis.local_reduce.graph.normalized_nodes),
        ("grouped_layouts", analysis.local_reduce.grouped_tensors),
        ("local_reduce_matches", analysis.local_reduce.matches),
        (
            "output_contraction_select_indices",
            analysis.output_contraction_select_indices,
        ),
    ):
        _append_items(
            lines,
            label,
            (f"{node.name}: {value!r}" for node, value in values.items()),
        )
        lines.append("")
    return "\n".join(lines).rstrip()


def _format_tensor_meta(meta: torch.Tensor) -> str:
    """Format output metadata used for allocation and ABI planning."""
    return (
        f"shape={tuple(meta.shape)}, stride={tuple(meta.stride())}, dtype={meta.dtype}"
    )


def format_flex_gemm_lowering_plan(
    logical_output_size: Sequence[Any],
    physical_output_size: Sequence[Any],
    output_dtype: torch.dtype,
    capture_kinds: Sequence[tuple[str, str]],
    aux_metas: Sequence[torch.Tensor],
    local_reduce_metas: Sequence[torch.Tensor],
) -> str:
    """Render buffer allocation and runtime-ABI decisions."""
    lines = [
        "output_storage:",
        f"  logical: shape={tuple(logical_output_size)}, dtype={output_dtype}",
        f"  physical: shape={tuple(physical_output_size)}",
        "",
    ]
    _append_items(
        lines,
        "captures",
        (f"{name} -> {kind}" for name, kind in capture_kinds),
    )
    lines.append("")
    _append_items(lines, "auxiliary_storage", map(_format_tensor_meta, aux_metas))
    lines.append("")
    if local_reduce_metas:
        _append_items(
            lines,
            "local_reduction_storage",
            (
                f"output {index}: {_format_tensor_meta(meta)}"
                for index, meta in enumerate(local_reduce_metas)
            ),
        )
        lines.append("  layout: dense")
    else:
        lines.append("local_reduction_storage: (none)")
    return "\n".join(lines)


def format_flex_gemm_config_key(config_key: "GemmConfigKey") -> str:
    """Render every config field so new GemmConfig fields remain visible."""
    return "\n".join(
        f"{name}: {'auto' if value is None else repr(value)}"
        for name, value in config_key
    )


def format_flex_gemm_config_candidates(
    config_keys: Sequence["GemmConfigKey"],
) -> str:
    """Render every lowering-approved config for verbose diagnostics."""
    lines: list[str] = []
    for index, config_key in enumerate(config_keys):
        _append_items(
            lines,
            f"candidate {index}",
            format_flex_gemm_config_key(config_key).splitlines(),
        )
        lines.append("")
    return "\n".join(lines).rstrip() if lines else "(none)"


def format_flex_gemm_selection(
    choice: "ir.ChoiceCaller | None",
    config_key: "GemmConfigKey | None",
    *,
    candidate_count: int,
    tuned: bool,
) -> str:
    """Render the search summary and selected FlexGEMM template."""
    lines = [
        "search:",
        f"  mode: {'autotuned' if tuned else 'fixed'}",
        f"  lowering_approved_candidates: {candidate_count}",
        "",
        "selected:",
    ]
    if choice is None:
        lines.append("  deferred to a multi-template buffer")
    else:
        lines.append(f"  template: {choice.name}")
        lines.append("  config:")
        lines.extend(
            "    " + line
            for line in (
                ("(unavailable)",)
                if config_key is None
                else format_flex_gemm_config_key(config_key).splitlines()
            )
        )
    lines.extend(
        (
            "",
            "more_detail_commands:",
            '  analysis/codegen: TORCH_LOGS="+flex_gemm"',
            '  autotune timings: TORCH_LOGS="flex_gemm,autotuning"',
            '  generated kernel: TORCH_LOGS="flex_gemm,kernel_code"',
            '  final wrapper: TORCH_LOGS="flex_gemm,output_code"',
            '  candidate failures: TORCH_LOGS="+inductor,flex_gemm"',
        )
    )
    return "\n".join(lines)
<|endoftext|># mypy: allow-untyped-defs
"""Lower FlexGEMM HOP bodies and connect epilogue analysis to backend templates.

``flex_gemm_lowering`` is the main entry point. Non-QUACK
requests execute the captured body through ordinary Inductor lowering. Although this
is stale and will fix up later on. See ``lower_quack_flex_gemm`` for the flow.
"""

from __future__ import annotations

from functools import partial
from typing import Any, TYPE_CHECKING

import torch
import torch.utils._pytree as pytree
from torch._higher_order_ops.flex_gemm import (
    _SUPPORTED_FLEX_GEMM_OP_NAMES,
    flex_gemm_hop,
    FLEX_GEMM_OP_SPECS,
)
from torch._inductor import config
from torch.utils._ordered_set import OrderedSet

from ... import ir
from ...ir import IRNode, TensorBox
from ...lowering import empty_strided, process_subgraph_nodes, register_lowering
from ...utils import ceildiv, has_free_symbols
from .constraints import (
    FLEX_GEMM_CHUNKED_CONTIGUOUS_B_ERROR,
    flex_gemm_local_reduce_config_error,
    FLEX_GEMM_OUTPUT_CONTRACTION_CAPTURE_ERROR,
    FLEX_GEMM_OUTPUT_CONTRACTION_COMPOSITION_ERROR,
    FlexGemmOutputContraction,
    is_flex_gemm_partial_reduction_shape,
    LOCAL_REDUCE_AUX_OUTPUT_CONTRACT_ERROR,
    LOCAL_REDUCE_DENSE_MM_SCOPE_ERROR,
    LOCAL_REDUCE_PARTIAL_OUTPUT_CONTRACT_ERROR,
    output_contraction_capture_supported,
    output_contraction_config_supported,
    statically_known_shape_equal,
    validate_flex_gemm_local_reduce_config,
)
from .debug import (
    format_flex_gemm_analysis,
    format_flex_gemm_analysis_details,
    format_flex_gemm_config_candidates,
    format_flex_gemm_lowering_plan,
    format_flex_gemm_problem,
    format_flex_gemm_selection,
    log_flex_gemm_artifact,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def decompose_nvgemm_additive_gemm(graph_module: torch.fx.GraphModule) -> None:
    graph = graph_module.graph
    changed = False
    for node in list(graph.nodes):
        if node.target not in (
            torch.ops.aten.addmm.default,
            torch.ops.aten.baddbmm.default,
        ):
            continue
        bias, mat1, mat2 = node.args[:3]
        alpha = node.kwargs.get("alpha", 1.0)
        beta = node.kwargs.get("beta", 1.0)
        gemm_target = (
            torch.ops.aten.mm.default
            if node.target is torch.ops.aten.addmm.default
            else torch.ops.aten.bmm.default
        )
        with graph.inserting_before(node):
            result = graph.call_function(gemm_target, (mat1, mat2))
            if alpha != 1:
                result = graph.call_function(torch.ops.aten.mul.Tensor, (result, alpha))
            if beta != 0:
                if beta != 1:
                    bias = graph.call_function(torch.ops.aten.mul.Tensor, (bias, beta))
                result = graph.call_function(torch.ops.aten.add.Tensor, (result, bias))
        result.meta = node.meta
        node.replace_all_uses_with(result)
        graph.erase_node(node)
        changed = True
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()


def flex_gemm_tensor_placeholders(
    graph_module: torch.fx.GraphModule,
) -> list[torch.fx.Node]:
    """Return placeholders QuACK can bind as tensor epilogue arguments.

    FlexGEMM identifies the GEMM A/B inputs from the mm node, then treats the
    remaining tensor-valued placeholders as closed-over epilogue tensors. Scalar
    SymInt placeholders are shape values, not tensor arguments; the current QuACK
    FlexGEMM entrypoint has no scalar epilogue-argument slots for them.
    """
    return [
        node
        for node in graph_module.graph.nodes
        if node.op == "placeholder" and isinstance(node.meta.get("val"), torch.Tensor)
    ]


def flex_gemm_epilogue_arg_placeholders(
    graph_module: torch.fx.GraphModule, gemm_fx_node: torch.fx.Node
) -> tuple[torch.fx.Node, ...]:
    """Find tensor inputs captured by epilogue loads, excluding GEMM operands."""
    gemm_placeholders = OrderedSet(
        arg
        for arg in pytree.tree_leaves((gemm_fx_node.args, gemm_fx_node.kwargs))
        if isinstance(arg, torch.fx.Node)
    )
    return tuple(
        node
        for node in flex_gemm_tensor_placeholders(graph_module)
        if node not in gemm_placeholders
    )


def infer_flex_gemm_epilogue_arg_kinds(
    gemm_op: torch._ops.OpOverload,
    epilogue_args: list[IRNode],
    output_size: list[Any],
) -> tuple[str, ...]:
    """Classify realized captured epilogue tensors for static wrapper kwargs."""
    if not epilogue_args:
        return ()
    if gemm_op is not torch.ops.aten.mm.default:
        raise NotImplementedError(
            "FlexGEMM generated epilogues with captured tensor reads currently support only aten.mm"
        )
    m, n = output_size[-2], output_size[-1]
    epilogue_arg_kinds = []
    for epilogue_arg in epilogue_args:
        epilogue_arg_size = epilogue_arg.get_size()
        if statically_known_shape_equal(epilogue_arg_size, [1, 1]):
            epilogue_arg_kinds.append("scalar")
        elif statically_known_shape_equal(epilogue_arg_size, output_size):
            epilogue_arg_kinds.append("tile")
        elif statically_known_shape_equal(epilogue_arg_size, [1, n]):
            epilogue_arg_kinds.append("row")
        elif statically_known_shape_equal(epilogue_arg_size, [m, 1]):
            epilogue_arg_kinds.append("col")
        else:
            raise NotImplementedError(
                "FlexGEMM captured tensor epilogue args currently must match "
                "the GEMM output shape or broadcast as [1, N] / [M, 1] / [1, 1]"
            )
    return tuple(epilogue_arg_kinds)


def validate_flex_gemm_aux_outputs(
    gemm_op: torch._ops.OpOverload,
    aux_outputs: tuple[torch.fx.Node, ...],
    output_size: list[Any],
) -> tuple[Any, ...]:
    """Validate QUACK aux-output support and return fake tensor metadata."""
    if not aux_outputs:
        return ()
    if gemm_op is not torch.ops.aten.mm.default:
        raise NotImplementedError(
            "FlexGEMM generic aux tuple epilogues currently support only aten.mm"
        )
    aux_metas = []
    for aux_output in aux_outputs:
        aux_meta = aux_output.meta.get("val")
        if aux_meta is None:
            raise NotImplementedError(
                "FlexGEMM generic aux tuple epilogues require aux output metadata"
            )
        aux_size = ir.convert_shape_to_inductor(aux_meta.shape)
        if not statically_known_shape_equal(aux_size, output_size):
            if is_flex_gemm_partial_reduction_shape(aux_size, output_size):
                raise NotImplementedError(LOCAL_REDUCE_PARTIAL_OUTPUT_CONTRACT_ERROR)
            raise NotImplementedError(LOCAL_REDUCE_AUX_OUTPUT_CONTRACT_ERROR)
        aux_metas.append(aux_meta)
    return tuple(aux_metas)


def allocate_flex_gemm_aux_outs(
    aux_metas: tuple[Any, ...], mat1: TensorBox
) -> tuple[TensorBox, ...]:
    """Allocate same-shape aux output buffers beside the main GEMM output."""
    return tuple(
        empty_strided(
            ir.convert_shape_to_inductor(aux_meta.shape),
            ir.convert_shape_to_inductor(aux_meta.stride()),
            dtype=aux_meta.dtype,
            device=mat1.get_device_or_error(),
        )
        for aux_meta in aux_metas
    )


def flex_gemm_ordered_outputs(result, aux_outs, local_reduce_outs, local_reduce_index):
    """Return generated outputs in the user-visible tuple order."""
    match local_reduce_outs, local_reduce_index:
        case (), _:
            return (result, *aux_outs)
        case (local_reduce_out,), None:
            return (result, *aux_outs, local_reduce_out)
        case (local_reduce_out,), index:
            return (
                result,
                *aux_outs[:index],
                local_reduce_out,
                *aux_outs[index:],
            )
        case _:
            raise AssertionError("FlexGEMM expects at most one local-reduce output")


def flex_gemm_local_reduce_metas(local_reduce) -> tuple[Any, ...]:
    """Return metadata for the optional compressed local-reduce output."""
    if local_reduce is None or local_reduce.store is None:
        return ()
    return (local_reduce.store.node.meta["val"],)


def flex_gemm_autotune_view_input(node: ir.ReinterpretView) -> torch.Tensor:
    """Rebuild a logical view for Python-backed template benchmarks."""
    from torch._inductor.select_algorithm import (
        AlgorithmSelectorCache,
        get_strides_with_layout_constraints,
    )
    from torch._inductor.virtualized import V

    value = AlgorithmSelectorCache.benchmark_example_value(node)
    base = value if value._base is None else value._base
    sizevars = V.graph.sizevars
    sizes = sizevars.optimization_hints(node.get_size())
    strides = sizevars.optimization_hints(get_strides_with_layout_constraints(node))
    offset = sizevars.optimization_hint(node.get_layout().offset)
    return torch.as_strided(base, sizes, strides, offset)


def filter_gemm_configs(
    configs: Sequence[Any],
    predicate: Callable[[Any], bool],
    *,
    explicit_config: dict[str, Any] | None,
    explicit_error: str,
    inferred_error: str,
) -> tuple[Any, ...]:
    """Filter configs while preserving explicit and inferred failure diagnostics."""
    supported = tuple(config for config in configs if predicate(config))
    if supported:
        return supported
    raise NotImplementedError(
        explicit_error if explicit_config is not None else inferred_error
    )


def flex_gemm_config_keys(
    device,
    m: int,
    n: int,
    local_reduce_geometries: tuple[Any, ...],
    tuned: bool,
    output_contraction: FlexGemmOutputContraction | None = None,
    explicit_config: dict[str, Any] | None = None,
) -> tuple[tuple[Any, ...], ...]:
    """Select QuACK config keys after applying grouped-layout config constraints.

    Every grouped geometry constrains the config the same way, whether it backs
    a runtime local-reduce plan or a plan-less grouped TensorSSA fragment
    reshape in the generated epilogue: swap_ab reorients the accumulator
    fragment and non-divisible tiles split groups across fragments. Output
    contractions additionally require tile_n not to oversubscribe physical N and
    a single CTA along N until QuACK predicates those layouts correctly. Explicit
    config fields constrain the candidate set; untuned selection fills omitted
    fields from the normal default, while tuned selection benchmarks the matches.
    """
    if output_contraction is not None:
        output_contraction.validate_quack(torch.cuda.get_device_capability(device)[0])
        if has_free_symbols((n,)):
            raise NotImplementedError(
                "FlexGEMM output contractions require statically known physical N"
            )

    from torch._inductor.heuristics.template.flex_gemm import (
        candidate_gemm_configs_for_device,
        default_gemm_config_key,
        explicit_gemm_configs_for_device,
        gemm_config_from_key,
        gemm_config_key,
    )

    candidate_configs = (
        candidate_gemm_configs_for_device(device)
        if explicit_config is None
        else explicit_gemm_configs_for_device(explicit_config, device)
    )
    if not tuned:
        default_key = default_gemm_config_key(device, m, n, candidate_configs)
        default_config = gemm_config_from_key(default_key)
        if all(
            validate_flex_gemm_local_reduce_config(
                default_config, geometry.group, geometry.axis
            )
            for geometry in local_reduce_geometries
        ) and (
            output_contraction is None
            or output_contraction_config_supported(default_config, n)
        ):
            return (default_key,)

    configs = candidate_configs
    if output_contraction is not None:
        configs = filter_gemm_configs(
            configs,
            lambda config: output_contraction_config_supported(config, n),
            explicit_config=explicit_config,
            explicit_error=(
                "FlexGEMM explicit QUACK config constraints are incompatible with "
                "output contraction"
            ),
            inferred_error=(
                "FlexGEMM output contraction physical N is smaller than every "
                "validated QuACK tile_n"
            ),
        )
    for geometry in local_reduce_geometries:
        configs = filter_gemm_configs(
            configs,
            lambda config: validate_flex_gemm_local_reduce_config(
                config, geometry.group, geometry.axis
            ),
            explicit_config=explicit_config,
            explicit_error=(
                "FlexGEMM explicit QUACK config constraints are incompatible with "
                f"local reduction group={geometry.group}, axis={geometry.axis}"
            ),
            inferred_error=flex_gemm_local_reduce_config_error(
                candidate_configs,
                geometry.group,
                geometry.axis,
            ),
        )
    if tuned:
        return tuple(gemm_config_key(config) for config in configs)
    return (gemm_config_key(configs[0]),)


def lower_quack_flex_gemm(gemm_op, subgraph, args, gemm_kwargs, kernel_options):
    """Lower FlexGEMM through the generated QUACK CuTeDSL template.

    The current pipeline is:

    ::

        FlexGEMM HOP body (FX GraphModule)
                         |
                         v
          find GEMM operands and captured epilogue args
                         |
                         v
             analyze_flex_gemm_epilogue()
                         |
                         +--> output plan + buffer ABI
                         |        `--> derive layout + allocate aux
                         |
                         +--> grouped/reduction geometry
                         |        `--> filter QuACK configurations
                         |
                         `--> materialize_flex_gemm_epilogue()
                                  `--> CuTeDSL epilogue + callbacks
                         |
                         v
             combine into template choices -> autotune
                         |
                         v
              restore captured output order
    """
    if gemm_op not in FLEX_GEMM_OP_SPECS:
        raise NotImplementedError(
            f"FlexGEMM QUACK backend currently supports only aten.{_SUPPORTED_FLEX_GEMM_OP_NAMES}"
        )
    tuned = kernel_options.get("tuned", False)
    fast_math = kernel_options.get("fast_math", False)
    explicit_config = kernel_options.get("config")
    unsupported_options = OrderedSet(kernel_options) - OrderedSet(
        ["backend", "tuned", "fast_math", "config"]
    )
    if unsupported_options:
        raise NotImplementedError(
            f"unsupported FlexGEMM kernel options: {sorted(unsupported_options)}"
        )
    if not isinstance(fast_math, bool):
        raise NotImplementedError("FlexGEMM fast_math kernel option must be bool")
    if "config" in kernel_options and not isinstance(explicit_config, dict):
        raise NotImplementedError("FlexGEMM config kernel option must be a dict")

    from torch._inductor.kernel.flex_gemm.fx_cutedsl_codegen import (
        analyze_flex_gemm_epilogue,
        gemm_node as flex_gemm_node,
        materialize_flex_gemm_epilogue,
    )
    from torch._inductor.kernel.flex_gemm.template import (
        flex_gemm_epilogue_template,
        FlexGemmEpilogueConfig,
        FlexGemmEpilogueLocalReduceConfig,
        FlexGemmEpilogueOutputConfig,
    )
    from torch._inductor.select_algorithm import autotune_select_algorithm

    op_spec = FLEX_GEMM_OP_SPECS[gemm_op]
    mat1_index, mat2_index = op_spec.mat1_index, op_spec.mat2_index
    unsupported_gemm_kwargs = OrderedSet(gemm_kwargs) - OrderedSet(["alpha", "beta"])
    if unsupported_gemm_kwargs:
        raise NotImplementedError(
            f"unsupported FlexGEMM GEMM kwargs: {sorted(unsupported_gemm_kwargs)}"
        )
    gemm_fx_node = flex_gemm_node(subgraph.graph_module, gemm_op)
    placeholders = [
        node for node in subgraph.graph_module.graph.nodes if node.op == "placeholder"
    ]
    placeholder_args = dict(zip(placeholders, args, strict=True))
    gemm_args: list[TensorBox] = []
    for arg in gemm_fx_node.args:
        gemm_arg = placeholder_args[arg] if isinstance(arg, torch.fx.Node) else arg
        if not isinstance(gemm_arg, TensorBox):
            raise NotImplementedError("FlexGEMM lowering expects tensor GEMM operands")
        gemm_args.append(gemm_arg)
    epilogue_arg_placeholders = flex_gemm_epilogue_arg_placeholders(
        subgraph.graph_module, gemm_fx_node
    )
    epilogue_args: list[TensorBox] = []
    for arg in epilogue_arg_placeholders:
        epilogue_arg = placeholder_args[arg]
        if not isinstance(epilogue_arg, TensorBox):
            raise NotImplementedError(
                "FlexGEMM lowering expects tensor epilogue operands"
            )
        epilogue_args.append(epilogue_arg)
    alpha = gemm_fx_node.kwargs.get("alpha", gemm_kwargs.get("alpha", 1.0))
    beta = gemm_fx_node.kwargs.get("beta", gemm_kwargs.get("beta", 1.0))
    if not isinstance(alpha, (int, float)) or not isinstance(beta, (int, float)):
        raise NotImplementedError("FlexGEMM alpha/beta must be static scalars")
    gemm_input_names = tuple(
        arg.name if isinstance(arg, torch.fx.Node) else f"gemm_arg{index}"
        for index, arg in enumerate(gemm_fx_node.args)
    )
    problem_report = partial(
        format_flex_gemm_problem,
        subgraph.graph_module,
        gemm_op,
        tuple(zip(gemm_input_names, gemm_args, strict=True)),
        tuple(
            zip(
                (node.name for node in epilogue_arg_placeholders),
                epilogue_args,
                strict=True,
            )
        ),
        alpha=float(alpha),
        beta=float(beta),
        tuned=tuned,
        fast_math=fast_math,
        explicit_config=explicit_config,
    )
    log_flex_gemm_artifact(
        "problem",
        problem_report,
        lowering_name=subgraph.name,
    )
    # This is where we figure out what the fx-graph body is doing
    epilogue_analysis = analyze_flex_gemm_epilogue(subgraph.graph_module, gemm_fx_node)
    log_flex_gemm_artifact(
        "analysis",
        lambda: format_flex_gemm_analysis(epilogue_analysis),
        lowering_name=subgraph.name,
    )
    log_flex_gemm_artifact(
        "analysis_details",
        lambda: format_flex_gemm_analysis_details(epilogue_analysis),
        lowering_name=subgraph.name,
        verbose=True,
    )
    outputs = epilogue_analysis.outputs
    if (
        epilogue_analysis.required_geometries
        and gemm_op is not torch.ops.aten.mm.default
    ):
        error = (
            FLEX_GEMM_OUTPUT_CONTRACTION_COMPOSITION_ERROR
            if outputs.output_contraction is not None
            else LOCAL_REDUCE_DENSE_MM_SCOPE_ERROR
        )
        raise NotImplementedError(error)
    local_reduce_store = (
        None if outputs.local_reduce is None else outputs.local_reduce.store
    )
    output_meta = outputs.output.meta.get("val")
    if output_meta is None:
        raise NotImplementedError(
            "FlexGEMM generated epilogues require output metadata"
        )
    output_storage = outputs.output_storage
    output_storage_dtype = (
        output_meta.dtype
        if output_storage is None
        else output_storage.meta["val"].dtype
    )
    logical_output_size = ir.convert_shape_to_inductor(output_meta.shape)
    physical_output_size = [
        *gemm_args[mat1_index].get_size()[:-1],
        gemm_args[mat2_index].get_size()[-1],
    ]
    output_contraction = outputs.output_contraction
    if (
        output_contraction is not None
        and output_contraction.chunked
        and gemm_args[mat2_index].get_stride()[-1] == 1
    ):
        raise NotImplementedError(FLEX_GEMM_CHUNKED_CONTIGUOUS_B_ERROR)
    aux_metas = validate_flex_gemm_aux_outputs(
        gemm_op, outputs.aux_outputs, physical_output_size
    )
    local_reduce_metas = flex_gemm_local_reduce_metas(outputs.local_reduce)
    output_stride = ir.convert_shape_to_inductor(output_meta.stride())
    if output_contraction is not None:
        # Output contractions use TMA stores with 16-byte-aligned outer strides.
        output_alignment = max(16 // output_storage_dtype.itemsize, 1)
        output_stride[-2] = (
            ceildiv(logical_output_size[-1], output_alignment) * output_alignment
        )
    layout = ir.FixedLayout(
        gemm_args[mat1_index].get_device_or_error(),
        output_storage_dtype,
        logical_output_size,
        output_stride,
    )
    gemm_input_nodes = [
        ir.TemplateBuffer.realize_template_input(arg) for arg in gemm_args
    ]
    epilogue_input_nodes = [
        ir.TemplateBuffer.realize_template_input(arg) for arg in epilogue_args
    ]
    aux_outs = allocate_flex_gemm_aux_outs(aux_metas, gemm_args[mat1_index])
    local_reduce_outs = allocate_flex_gemm_aux_outs(
        local_reduce_metas, gemm_args[mat1_index]
    )
    aux_input_nodes = [
        ir.TemplateBuffer.realize_template_input(aux_out) for aux_out in aux_outs
    ]
    local_reduce_input_nodes = [
        ir.TemplateBuffer.realize_template_input(local_reduce_out)
        for local_reduce_out in local_reduce_outs
    ]
    input_nodes = [
        *gemm_input_nodes,
        *epilogue_input_nodes,
        *aux_input_nodes,
        *local_reduce_input_nodes,
    ]
    mutated_input_nodes = aux_input_nodes + local_reduce_input_nodes
    aux_out_start = len(gemm_input_nodes) + len(epilogue_input_nodes)
    aux_out_indices = tuple(range(aux_out_start, aux_out_start + len(aux_input_nodes)))
    local_reduce_out_index = (
        aux_out_start + len(aux_input_nodes) if local_reduce_input_nodes else None
    )
    epilogue_arg_kinds = infer_flex_gemm_epilogue_arg_kinds(
        gemm_op, epilogue_input_nodes, physical_output_size
    )
    if output_contraction is not None and any(
        not output_contraction_capture_supported(kind, arg.get_dtype() is torch.bool)
        for arg, kind in zip(epilogue_input_nodes, epilogue_arg_kinds, strict=True)
    ):
        raise NotImplementedError(FLEX_GEMM_OUTPUT_CONTRACTION_CAPTURE_ERROR)
    log_flex_gemm_artifact(
        "lowering_plan",
        lambda: format_flex_gemm_lowering_plan(
            logical_output_size,
            physical_output_size,
            output_meta.dtype,
            tuple(
                zip(
                    (node.name for node in epilogue_arg_placeholders),
                    epilogue_arg_kinds,
                    strict=True,
                )
            ),
            aux_metas,
            local_reduce_metas,
        ),
        lowering_name=subgraph.name,
    )
    template_local_reduce = FlexGemmEpilogueLocalReduceConfig.from_output_plan(
        outputs.local_reduce, local_reduce_out_index
    )
    epilogue_name, epilogue_source = materialize_flex_gemm_epilogue(
        subgraph.graph_module,
        epilogue_analysis,
        epilogue_arg_placeholders,
        fast_math=fast_math,
    )
    log_flex_gemm_artifact(
        "generated_epilogue",
        lambda: epilogue_source.strip(),
        lowering_name=subgraph.name,
        verbose=True,
    )
    quack_config_keys = flex_gemm_config_keys(
        layout.device,
        gemm_args[mat1_index].get_size()[-2],
        gemm_args[mat2_index].get_size()[-1],
        epilogue_analysis.required_geometries,
        tuned,
        output_contraction,
        explicit_config,
    )
    log_flex_gemm_artifact(
        "config_candidates",
        lambda: format_flex_gemm_config_candidates(quack_config_keys),
        lowering_name=subgraph.name,
        verbose=True,
    )
    epilogue_arg_indices = tuple(
        range(
            len(gemm_input_nodes),
            len(gemm_input_nodes) + len(epilogue_input_nodes),
        )
    )
    choices: list[Any] = []
    choice_config_keys = {}
    for quack_config_key in quack_config_keys:
        choice_start = len(choices)
        error = flex_gemm_epilogue_template.maybe_append_choice(
            choices,
            input_nodes=input_nodes,
            layout=layout,
            mutated_inputs=mutated_input_nodes or None,
            config=FlexGemmEpilogueConfig(
                epilogue_name=epilogue_name,
                epilogue_source=epilogue_source,
                gemm_op=op_spec,
                alpha=float(alpha),
                beta=float(beta),
                quack_config_key=quack_config_key,
                epilogue_arg_indices=epilogue_arg_indices,
                epilogue_arg_kinds=epilogue_arg_kinds,
                outputs=FlexGemmEpilogueOutputConfig(
                    aux_out_indices=aux_out_indices,
                    local_reduce=template_local_reduce,
                    output_contraction=output_contraction,
                ),
            ),
        )
        if error is not None:
            raise error
        choice_config_keys.update(
            (choice.name, quack_config_key) for choice in choices[choice_start:]
        )
    input_gen_fns = {
        index: flex_gemm_autotune_view_input
        for index, input_node in enumerate(input_nodes)
        if isinstance(input_node, ir.ReinterpretView)
    }
    result, selected_choice = autotune_select_algorithm(
        "flex_gemm_epilogue",
        choices,
        input_nodes,
        layout,
        input_gen_fns=input_gen_fns or None,
        **({"return_multi_template": False} if mutated_input_nodes else {}),
    )
    if output_storage is not None and output_storage_dtype is not output_meta.dtype:
        result = TensorBox(ir.DtypeView.create(result, output_meta.dtype))
    selected_config_key = (
        None
        if selected_choice is None
        else choice_config_keys.get(selected_choice.name)
    )
    log_flex_gemm_artifact(
        "selection",
        lambda: format_flex_gemm_selection(
            selected_choice,
            selected_config_key,
            candidate_count=len(quack_config_keys),
            tuned=tuned,
        ),
        lowering_name=subgraph.name,
    )
    return flex_gemm_ordered_outputs(
        result,
        aux_outs,
        local_reduce_outs,
        None if local_reduce_store is None else local_reduce_store.aux_index,
    )


@register_lowering(flex_gemm_hop, type_promotion_kind=None)
def flex_gemm_lowering(gemm_op, subgraph, args, gemm_kwargs, kernel_options):
    """Dispatch FlexGEMM to ordinary Inductor lowering or the QUACK template."""
    backend = kernel_options.get("backend", "TRITON")
    if backend == "NVGEMM":
        decompose_nvgemm_additive_gemm(subgraph.graph_module)
        with config.patch(
            max_autotune=True,
            max_autotune_gemm_backends="NVGEMM",
        ):
            return process_subgraph_nodes(subgraph.graph_module, list(args))
    if backend == "QUACK":
        return lower_quack_flex_gemm(
            gemm_op, subgraph, args, gemm_kwargs, kernel_options
        )
    return process_subgraph_nodes(subgraph.graph_module, list(args))
<|endoftext|># mypy: allow-untyped-defs
import dataclasses
import logging
from typing_extensions import override

import torch
from torch._dynamo.utils import counters
from torch._higher_order_ops.flex_gemm import FlexGemmOpSpec
from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.codegen.cutedsl.cutedsl_kernel import CuteDSLTemplateKernel
from torch._inductor.codegen.cutedsl.cutedsl_template import (
    CuteDSLTemplate,
    CuteDSLTemplateCaller,
)
from torch._inductor.heuristics.template.flex_gemm import GemmConfigKey
from torch._inductor.kernel.flex_gemm.constraints import (
    FlexGemmLocalReduceGeometry,
    FlexGemmOutputContraction,
    LOCAL_REDUCE_COMBINE_FN_SUFFIX,
    LOCAL_REDUCE_FINALIZE_FN_SUFFIX,
)
from torch._inductor.kernel.flex_gemm.runtime import inductor_quack_cache_dir
from torch._inductor.kernel.gemm_epilogue_analysis import GemmOutputLocalReducePlan
from torch._inductor.select_algorithm import PartialRender
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class FlexGemmEpilogueLocalReduceConfig:
    """Template-time local-reduce metadata for output and/or feed-main consumers."""

    geometry: FlexGemmLocalReduceGeometry
    out_index: int | None = None
    feeds_main: bool = False

    @classmethod
    def from_output_plan(
        cls,
        local_reduce: GemmOutputLocalReducePlan | None,
        out_index: int | None,
    ) -> "FlexGemmEpilogueLocalReduceConfig | None":
        """Bind analyzed local-reduction consumers to FlexGEMM's runtime ABI."""
        if local_reduce is None:
            return None
        return FlexGemmEpilogueLocalReduceConfig(
            FlexGemmLocalReduceGeometry(
                local_reduce.match.geometry.group,
                local_reduce.match.geometry.axis,
            ),
            out_index,
            local_reduce.feeds_main,
        )

    @property
    def group(self) -> int:
        return self.geometry.group

    @property
    def axis(self) -> int:
        return self.geometry.axis

    @property
    def needs_physical_callbacks(self) -> bool:
        return self.geometry.needs_physical_callbacks


@dataclasses.dataclass(frozen=True)
class FlexGemmEpilogueOutputConfig:
    """Template input indices and plans for all user-visible outputs."""

    aux_out_indices: tuple[int, ...] = ()
    local_reduce: FlexGemmEpilogueLocalReduceConfig | None = None
    output_contraction: FlexGemmOutputContraction | None = None


@dataclasses.dataclass(frozen=True)
class FlexGemmEpilogueConfig:
    """Metadata needed to render one Inductor-owned QuACK GEMM epilogue choice.

    Attributes:
        epilogue_name: Name of the generated CuTeDSL epilogue callable.
        epilogue_source: Python source that defines ``epilogue_name``.
        gemm_op: Original aten GEMM op spec used to map inputs into QuACK.
        alpha: Static alpha multiplier for addmm/baddbmm inputs.
        beta: Static beta multiplier for addmm/baddbmm bias inputs.
        quack_config_key: Lossless key for the selected QuACK GEMM config.
        epilogue_arg_indices: Template input indices for read-only epilogue captures.
        epilogue_arg_kinds: Broadcast kind for each captured epilogue tensor.
        outputs: Structural plans for auxiliary, reduced, or transformed outputs.
    """

    epilogue_name: str
    epilogue_source: str
    gemm_op: FlexGemmOpSpec
    alpha: float
    beta: float
    quack_config_key: GemmConfigKey
    epilogue_arg_indices: tuple[int, ...] = ()
    epilogue_arg_kinds: tuple[str, ...] = ()
    outputs: FlexGemmEpilogueOutputConfig = dataclasses.field(
        default_factory=FlexGemmEpilogueOutputConfig
    )


class FlexGemmEpilogueKernel(CuteDSLTemplateKernel):
    """Render generated FlexGEMM epilogue modules with a compile-only hook."""

    @override
    def render(self, template, **kwargs):
        config = kwargs.pop("config")
        if kwargs:
            raise RuntimeError(f"unexpected FlexGEMM epilogue options: {kwargs}")

        self._template_input_args = []
        self._seen_input_args = OrderedSet()
        for index, input_node in enumerate(self.input_nodes):
            buf_name = input_node.get_name()
            self.args.input(buf_name)
            arg_name = f"arg{index}"
            self.args.input_buffers[buf_name] = arg_name
            self._template_input_args.append((arg_name, input_node))
            self._seen_input_args.add(arg_name)

        self.args.output(self.output_node.get_name())
        arg_defs, _, _, _ = self.args.python_argdefs()
        params = [arg_name for arg_name, _ in self._template_input_args]
        for arg_def in arg_defs:
            if arg_def.full_name() not in self._seen_input_args:
                params.append(arg_def.full_name())
        params.append("stream")
        params.append("device_capacity_override=None")
        quack_cache_dir_param = f"quack_cache_dir={inductor_quack_cache_dir()!r}"
        params.append(quack_cache_dir_param)

        template_input_arg_names = [
            arg_name for arg_name, _ in self._template_input_args
        ]
        # Template inputs include GEMM operands plus closed-over epilogue tensors for reads and aux writes.
        call_args, call_kwargs = self._gemm_call_args(template_input_arg_names, config)
        call_kwargs += self._epilogue_kwargs(template_input_arg_names, config)
        call_kwargs += (
            f", out={self.get_output()}, "
            f"expected_ndim={config.gemm_op.input_ndim}, "
            "stream=stream, "
            "device_capacity_override=device_capacity_override, "
            "quack_cache_dir=quack_cache_dir"
        )
        call_kwargs += (
            f", config_key={config.quack_config_key!r}, "
            "config_is_lowering_validated=True"
        )

        output_name = self.get_output()

        code = IndentedBuffer()
        code.splice(
            """
            import torch
            from torch._inductor.kernel.flex_gemm.constraints import (
                FlexGemmOutputContraction,
                FlexGemmLocalReduceCallbacks,
                FlexGemmLocalReduceGeometry,
            )
            from torch._inductor.kernel.flex_gemm.runtime import (
                FlexGemmRuntimeLocalReducePlan,
                FlexGemmRuntimeOutputPlan,
                gemm_epilogue as flex_gemm_epilogue,
            )
            """
        )
        code.splice(config.epilogue_source)
        code.splice(
            f"""
            def {self.kernel_name}_main({", ".join(params)}):
                flex_gemm_epilogue(
                    {", ".join(call_args)},
                    {config.epilogue_name},
                    {config.epilogue_name!r}{call_kwargs},
                )

            def {self.kernel_name}_precompile(
                precompile_shapes,
                precompile_strides,
                precompile_dtypes,
                device_index=0,
                device_capability=None,
                hw_info=None,
                quack_cache_dir={inductor_quack_cache_dir()!r},
            ):
                from torch._vendor.quack.cache import compile_only_mode

                device = f"cuda:{{device_index}}"
                with compile_only_mode():
            """
        )
        with code.indent():
            with code.indent():
                for arg_name, _ in self._template_input_args:
                    code.writeline(
                        f"{arg_name} = torch.empty_strided(tuple(precompile_shapes[{arg_name!r}]), "
                        f"tuple(precompile_strides[{arg_name!r}]), device=device, "
                        f"dtype=getattr(torch, precompile_dtypes[{arg_name!r}]))"
                    )
                code.writeline(
                    f"{output_name} = torch.empty_strided(tuple(precompile_shapes['output']), "
                    "tuple(precompile_strides['output']), device=device, "
                    "dtype=getattr(torch, precompile_dtypes['output']))"
                )
                code.writeline(f"{self.kernel_name}_main(")
                with code.indent():
                    for arg_name in template_input_arg_names:
                        code.writeline(f"{arg_name},")
                    code.writeline(f"{output_name}={output_name},")
                    code.writeline("stream=None,")
                    code.writeline("device_capacity_override=device_capability,")
                    code.writeline("quack_cache_dir=quack_cache_dir,")
                code.writeline(")")
        return PartialRender(code.getvalue(), self.render_hooks)

    def _gemm_call_args(
        self, input_args: list[str], config: FlexGemmEpilogueConfig
    ) -> tuple[list[str], str]:
        """Return positional GEMM operands and scalar/bias kwargs for runtime dispatch."""
        op = config.gemm_op
        call_args = [input_args[op.mat1_index], input_args[op.mat2_index]]
        if op.bias_index is None:
            return call_args, ""
        return call_args, (
            f", C={input_args[op.bias_index]}, alpha={config.alpha!r}, beta={config.beta!r}"
        )

    def _local_reduce_callbacks(self, epilogue_name: str) -> str:
        """Render generated physical reducer callbacks for runtime registration."""
        combine_name = f"{epilogue_name}{LOCAL_REDUCE_COMBINE_FN_SUFFIX}"
        finalize_name = f"{epilogue_name}{LOCAL_REDUCE_FINALIZE_FN_SUFFIX}"
        return (
            "FlexGemmLocalReduceCallbacks("
            f"combine_fn={combine_name}, finalize_fn={finalize_name})"
        )

    def _local_reduce_geometry(
        self, local_reduce: FlexGemmEpilogueLocalReduceConfig
    ) -> str:
        """Render the shared grouped M/N local-reduce geometry."""
        return (
            "FlexGemmLocalReduceGeometry("
            f"group={local_reduce.group!r}, axis={local_reduce.axis!r})"
        )

    def _local_reduce_expr(
        self,
        input_args: list[str],
        local_reduce: FlexGemmEpilogueLocalReduceConfig | None,
        epilogue_name: str,
    ) -> str:
        """Render one structural local-reduce plan for runtime dispatch."""
        if local_reduce is None:
            return "None"
        geometry = self._local_reduce_geometry(local_reduce)
        plan = f"FlexGemmRuntimeLocalReducePlan({geometry}"
        if local_reduce.out_index is not None:
            plan += f", out={input_args[local_reduce.out_index]}"
        if local_reduce.feeds_main:
            plan += ", feeds_main=True"
        if local_reduce.feeds_main or local_reduce.needs_physical_callbacks:
            plan += f", callbacks={self._local_reduce_callbacks(epilogue_name)}"
        return f"{plan})"

    def _output_plan_expr(
        self, input_args: list[str], config: FlexGemmEpilogueConfig
    ) -> str:
        """Render all output consumers into one runtime ABI value."""
        outputs = config.outputs
        aux_outs = tuple(input_args[index] for index in outputs.aux_out_indices)
        aux_expr = f"({', '.join(aux_outs)},)" if aux_outs else "()"
        local_reduce_expr = self._local_reduce_expr(
            input_args, outputs.local_reduce, config.epilogue_name
        )
        contraction = outputs.output_contraction
        contraction_expr = (
            "None"
            if contraction is None
            else "FlexGemmOutputContraction("
            f"group={contraction.group!r}, chunked={contraction.chunked!r})"
        )
        return (
            "FlexGemmRuntimeOutputPlan("
            f"aux_outs={aux_expr}, local_reduce={local_reduce_expr}, "
            f"output_contraction={contraction_expr})"
        )

    def _epilogue_kwargs(
        self, input_args: list[str], config: FlexGemmEpilogueConfig
    ) -> str:
        """Render only values that differ from the runtime ABI defaults."""
        epilogue_args = [input_args[index] for index in config.epilogue_arg_indices]
        kwargs: list[str] = []
        if epilogue_args:
            kwargs.append(
                f", epilogue_args=({', '.join(epilogue_args)},), "
                f"epilogue_arg_kinds={config.epilogue_arg_kinds!r}"
            )
        outputs = config.outputs
        if (
            outputs.aux_out_indices
            or outputs.local_reduce is not None
            or outputs.output_contraction is not None
        ):
            kwargs.append(f", output_plan={self._output_plan_expr(input_args, config)}")
        return "".join(kwargs)


class FlexGemmEpilogueCaller(CuteDSLTemplateCaller):
    def precompile(self) -> None:
        """Warm the generated FlexGEMM epilogue module's QuACK object cache."""
        metadata = self.precompile_metadata()
        if metadata is None:
            return
        from torch._inductor.async_compile import AsyncCompile

        if not AsyncCompile.use_process_pool():
            AsyncCompile.wait_pool_ready()
        if not AsyncCompile.use_process_pool():
            return
        AsyncCompile().cutedsl(
            self.bmreq.kernel_name,
            self.bmreq.source_code,
            precompile_metadata=metadata,
        ).result()

    def precompile_metadata(self) -> dict[str, object] | None:
        """Build the generated FlexGEMM precompile hook's tensor metadata."""
        precompile_shapes = {}
        precompile_strides = {}
        precompile_dtypes = {}
        tensor_metas = [
            *(
                (f"arg{index}", tensor_meta)
                for index, tensor_meta in enumerate(self.bmreq.input_tensor_meta)
            ),
            ("output", self.bmreq.output_tensor_meta),
        ]
        try:
            for name, tensor_meta in tensor_metas:
                precompile_shapes[name] = [int(size) for size in tensor_meta.sizes]
                precompile_strides[name] = [
                    int(stride) for stride in tensor_meta.strides
                ]
                precompile_dtypes[name] = str(tensor_meta.dtype).removeprefix("torch.")
        except (TypeError, RuntimeError, ValueError):
            counters["inductor"]["flex_gemm_precompile_skipped_dynamic"] += 1
            log.debug("Skipping FlexGEMM precompile for symbolic tensor metadata")
            return None
        device_index = self.bmreq.output_tensor_meta.device.index or 0
        device_capability = None
        if torch.cuda.is_available():
            device_capability = torch.cuda.get_device_capability(device_index)
        return {
            "precompile_shapes": precompile_shapes,
            "precompile_strides": precompile_strides,
            "precompile_dtypes": precompile_dtypes,
            "device_index": device_index,
            "device_capability": device_capability,
            "quack_cache_dir": inductor_quack_cache_dir(),
        }


class FlexGemmEpilogueTemplate(CuteDSLTemplate):
    kernel_type = FlexGemmEpilogueKernel
    caller_type = FlexGemmEpilogueCaller

    def __init__(self) -> None:
        super().__init__("flex_gemm_epilogue", source="")


flex_gemm_epilogue_template = FlexGemmEpilogueTemplate()
<|endoftext|># mypy: allow-untyped-defs
"""Analyze FlexGEMM epilogue FX graphs and materialize CuTeDSL source.

``analyze_flex_gemm_epilogue`` indexes FX dependencies, identifies nodes that
carry grouped TensorSSA layouts, matches supported local reductions, and plans
the main, auxiliary, and local-reduction consumers.

``materialize_flex_gemm_epilogue`` uses that analysis to generate the CuTeDSL
epilogue and physical reduction callbacks.
"""

import dataclasses
import hashlib
from typing import Any

from sympy import Max, Min

import torch
from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
    CuteDSLCSEVariable,
    upcast_compute_type,
    use_cutedsl_fast_math,
)
from torch._inductor.kernel import gemm_epilogue_analysis as _epilogue_analysis
from torch._inductor.kernel.flex_gemm.constraints import (
    FLEX_GEMM_CHUNKED_OUTPUT_CONTRACTION_REDUCE_ERROR,
    FLEX_GEMM_MAIN_OUTPUT_SHAPE_ERROR,
    FLEX_GEMM_OUTPUT_CONTRACTION_COMPOSITION_ERROR,
    FLEX_GEMM_OUTPUT_CONTRACTION_SHAPE_ERROR,
    FLEX_GEMM_OUTPUT_TENSOR_ERROR,
    FlexGemmLocalReduceGeometry,
    FlexGemmOutputContraction,
    LOCAL_REDUCE_AUX_TENSORSSA_ERROR,
    LOCAL_REDUCE_COMBINE_FN_SUFFIX,
    LOCAL_REDUCE_FEED_MAIN_ARG_NAME,
    LOCAL_REDUCE_FINALIZE_FN_SUFFIX,
    LOCAL_REDUCE_FINALIZE_SCALAR_ONLY_ERROR,
    LOCAL_REDUCE_MIXED_MATCH_ERROR,
    LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR,
    LOCAL_REDUCE_POST_POINTWISE_FINALIZE_ERROR,
    LOCAL_REDUCE_SINGLE_PHYSICAL_FINALIZE_ERROR,
    local_reduce_unsupported_tensorssa_error,
)
from torch._inductor.kernel.flex_gemm.quack_reductions import (
    _cute_arg,
    _cute_call,
    _cute_op_name,
    _keepdim_and_broadcast,
    _local_reduce_store_arg,
    FlexGemmPhysicalReduction,
    GroupedTensorSSALayout,
    is_shape_preserving_pointwise_node,
    lower_full_scalar,
    lower_getitem,
    lower_grouped_n_select,
    lower_grouped_n_split,
    lower_prepare_softmax_online,
    lower_squeeze,
    lower_tensorssa_reduce,
    lower_view_or_reshape,
    tensor_meta_shape,
)
from torch._inductor.kernel.gemm_epilogue import (
    GemmReductionPlan,
    iter_fx_node_inputs,
    NormalizedDtypeView,
    NormalizedGetItem,
    NormalizedPrepareSoftmax,
    NormalizedReduction,
    NormalizedSelect,
    NormalizedSplit,
    NormalizedSqueeze,
    NormalizedUnsupportedReduction,
    NormalizedView,
)
from torch._inductor.kernel.gemm_epilogue_codegen import (
    GemmEpilogueCuteDSLKernel,
    GemmEpilogueCuteDSLOpOverrides,
)
from torch._inductor.kernel.gemm_epilogue_utils import (
    guarded_int,
    statically_known_shape_equal,
)
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.value_ranges import ValueRanges


FlexGemmEpilogueGraph = _epilogue_analysis.GemmEpilogueGraph
FlexGemmLocalReduceAnalysis = _epilogue_analysis.GemmLocalReduceAnalysis
FlexGemmLocalReduceMatch = _epilogue_analysis.GemmLocalReduceMatch
FlexGemmLocalReduceStore = _epilogue_analysis.GemmLocalReduceStore
FlexGemmOutputLocalReducePlan = _epilogue_analysis.GemmOutputLocalReducePlan


@dataclasses.dataclass(frozen=True)
class FlexGemmOutputPlan(_epilogue_analysis.GemmOutputPlan):
    """Extend output planning with contraction and terminal storage metadata."""

    output_contraction: FlexGemmOutputContraction | None = None
    output_storage: torch.fx.Node | None = None


FlexGemmCuteDSLKernel = GemmEpilogueCuteDSLKernel


class FlexGemmCuteDSLOpOverrides(GemmEpilogueCuteDSLOpOverrides):
    """Add FlexGEMM-specific NaN-propagating clamp semantics."""

    @staticmethod
    def clamp(x: Any, min: Any = None, max: Any = None) -> Any:
        result = x
        if min is not None:
            result = FlexGemmCuteDSLOpOverrides.clamp_min(result, min)
        if max is not None:
            result = FlexGemmCuteDSLOpOverrides.clamp_max(result, max)
        return result

    @staticmethod
    def clamp_min(x: Any, min: Any) -> Any:
        return GemmEpilogueCuteDSLOpOverrides._apply_binary_op(
            x,
            min,
            "cutlass_math.max({a}, {b}, propagate_nan=True)",
            Max,
        )

    @staticmethod
    def clamp_max(x: Any, max: Any) -> Any:
        return GemmEpilogueCuteDSLOpOverrides._apply_binary_op(
            x,
            max,
            "cutlass_math.min({a}, {b}, propagate_nan=True)",
            Min,
        )


def tuple_output_plan(
    output: Any,
    aux_outputs: tuple[Any, ...],
    analysis: FlexGemmLocalReduceAnalysis,
) -> FlexGemmOutputPlan:
    """Classify multi-output epilogues after checking local-reduce consumers."""
    if not isinstance(output, torch.fx.Node) or not all(
        isinstance(aux_output, torch.fx.Node) for aux_output in aux_outputs
    ):
        raise NotImplementedError(FLEX_GEMM_OUTPUT_TENSOR_ERROR)
    feed_match = analysis.common_feed_main_match((output, *aux_outputs))
    compressed_aux_plans = tuple(
        (index, match, plan)
        for index, aux_output in enumerate(aux_outputs)
        if (match := analysis.matches.get(aux_output)) is not None
        if (plan := analysis.compressed_aux_plan(output, aux_output, index)) is not None
    )
    if len(compressed_aux_plans) > 1:
        raise NotImplementedError(LOCAL_REDUCE_MIXED_MATCH_ERROR)
    if compressed_aux_plans:
        local_reduce_index, compressed_match, compressed_aux_plan = (
            compressed_aux_plans[0]
        )
        if feed_match is not None:
            if feed_match.value_node is not compressed_match.value_node:
                raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
            compressed_aux_plan = feed_match.to_plan(
                store=FlexGemmLocalReduceStore(
                    aux_outputs[local_reduce_index], local_reduce_index
                ),
                feeds_main=True,
            )
        return FlexGemmOutputPlan(
            output,
            tuple(
                aux_output
                for index, aux_output in enumerate(aux_outputs)
                if index != local_reduce_index
            ),
            local_reduce=compressed_aux_plan,
        )
    feed_main_plan = analysis.feed_main_output_plan(output, aux_outputs)
    if feed_main_plan is not None:
        return FlexGemmOutputPlan(
            feed_main_plan.output,
            feed_main_plan.aux_outputs,
            feed_main_plan.local_reduce,
        )
    return FlexGemmOutputPlan(output, aux_outputs)


def output_plan(
    graph_module: torch.fx.GraphModule,
    local_reduce: FlexGemmLocalReduceAnalysis,
) -> FlexGemmOutputPlan:
    """Classify output consumers from one shared local-reduce analysis."""
    output_nodes = [node for node in graph_module.graph.nodes if node.op == "output"]
    if len(output_nodes) != 1:
        raise NotImplementedError("FlexGEMM expects one output node")
    output_value = output_nodes[0].args[0]
    if isinstance(output_value, (tuple, list)):
        if len(output_value) == 1:
            output_value = output_value[0]
        else:
            output, *aux_outputs = output_value
            return tuple_output_plan(output, tuple(aux_outputs), local_reduce)
    if not isinstance(output_value, torch.fx.Node):
        raise NotImplementedError("FlexGEMM expects one tensor output")
    feed_main_plan = local_reduce.feed_main_output_plan(output_value)
    if feed_main_plan is None:
        return FlexGemmOutputPlan(output_value)
    return FlexGemmOutputPlan(
        feed_main_plan.output,
        feed_main_plan.aux_outputs,
        feed_main_plan.local_reduce,
    )


def bind_terminal_output_storage(
    graph: FlexGemmEpilogueGraph, outputs: FlexGemmOutputPlan
) -> FlexGemmOutputPlan:
    """Record the physical value beneath a terminal same-width dtype view."""
    normalized = graph.normalized_nodes.get(outputs.output)
    if not isinstance(normalized, NormalizedDtypeView):
        return outputs
    source_meta = normalized.source.meta.get("val")
    output_meta = outputs.output.meta.get("val")
    if (
        not isinstance(source_meta, torch.Tensor)
        or not isinstance(output_meta, torch.Tensor)
        or normalized.dtype is not output_meta.dtype
        or source_meta.dtype.itemsize != output_meta.dtype.itemsize
        or not statically_known_shape_equal(source_meta.shape, output_meta.shape)
        or not statically_known_shape_equal(source_meta.stride(), output_meta.stride())
    ):
        raise NotImplementedError(
            "FlexGEMM terminal dtype views must preserve shape, stride, and element size"
        )
    return dataclasses.replace(outputs, output_storage=normalized.source)


@dataclasses.dataclass(frozen=True)
class OutputContractionUse:
    """Describe one supported use of grouped GEMM values in the main output."""

    source: torch.fx.Node
    group: int
    chunked: bool
    group_indices: tuple[int, ...]
    layout_node: torch.fx.Node | None = None


@dataclasses.dataclass(frozen=True)
class OutputContractionPlan:
    """Stage one complete output contraction until analysis commits it."""

    contraction: FlexGemmOutputContraction
    select_indices: dict[torch.fx.Node, int]
    grouped_tensors: dict[torch.fx.Node, FlexGemmLocalReduceGeometry]

    def commit(
        self,
        outputs: FlexGemmOutputPlan,
        local_reduce: FlexGemmLocalReduceAnalysis,
    ) -> FlexGemmOutputPlan:
        """Validate composition and install this output contraction."""
        if outputs.aux_outputs or outputs.local_reduce is not None:
            raise NotImplementedError(FLEX_GEMM_OUTPUT_CONTRACTION_COMPOSITION_ERROR)
        if self.contraction.chunked and any(
            local_reduce.graph.depends_on(outputs.output, reduced)
            for reduced in local_reduce.matches
        ):
            raise NotImplementedError(FLEX_GEMM_CHUNKED_OUTPUT_CONTRACTION_REDUCE_ERROR)
        local_reduce.grouped_tensors.update(self.grouped_tensors)
        return dataclasses.replace(outputs, output_contraction=self.contraction)


def canonical_output_contraction_source(
    node: torch.fx.Node,
    gemm: torch.fx.Node,
    local_reduce: FlexGemmLocalReduceAnalysis,
) -> torch.fx.Node:
    """Find one contraction use's GEMM provenance through pointwise wrappers."""
    while node is not gemm and is_shape_preserving_pointwise_node(node):
        inputs = [
            arg
            for arg in iter_fx_node_inputs((node.args, node.kwargs))
            if local_reduce.graph.depends_on(arg, gemm)
        ]
        if len(inputs) != 1:
            break
        node = inputs[0]
    return node


def match_output_contraction_use(
    node: torch.fx.Node,
    gemm: torch.fx.Node,
    gemm_shape: tuple[Any, ...] | None,
    local_reduce: FlexGemmLocalReduceAnalysis,
) -> OutputContractionUse | None:
    """Match one supported use of grouped GEMM values in the main output."""
    if gemm_shape is None:
        return None
    normalized = local_reduce.graph.normalized_nodes.get(node)

    if isinstance(normalized, NormalizedGetItem):
        split = normalized.source
        split_normalized = local_reduce.graph.normalized_nodes.get(split)
        if not isinstance(split_normalized, NormalizedSplit):
            return None
        shape = tensor_meta_shape(split_normalized.source)
        if (
            shape is None
            or len(shape) != 2
            or not statically_known_shape_equal(shape, gemm_shape)
            or not local_reduce.graph.depends_on(split_normalized.source, gemm)
            or not isinstance(shape[-1], int)
            or split_normalized.dim not in (-1, 1)
        ):
            return None
        split_size = guarded_int(split_normalized.split_size)
        if split_size is None or split_size <= 0 or shape[-1] % split_size != 0:
            return None
        group = shape[-1] // split_size
        if group <= 1:
            return None
        return OutputContractionUse(
            source=split_normalized.source,
            group=group,
            chunked=True,
            group_indices=(normalized.index,),
            layout_node=split,
        )

    if not isinstance(normalized, NormalizedSelect):
        return None
    view = normalized.source
    view_normalized = local_reduce.graph.normalized_nodes.get(view)
    shape = tensor_meta_shape(view)
    if (
        not isinstance(view_normalized, NormalizedView)
        or not local_reduce.graph.depends_on(view_normalized.source, gemm)
        or shape is None
        or len(shape) != 3
        or not statically_known_shape_equal((shape[0], shape[1] * shape[2]), gemm_shape)
        or not -len(shape) <= normalized.dim < len(shape)
    ):
        return None
    index = guarded_int(normalized.index)
    if index is None:
        return None
    dim = normalized.dim % len(shape)
    if dim == 1:
        group = guarded_int(shape[1])
        if group is None or group <= 1:
            return None
        chunked = True
        layout_node = view
    elif dim == len(shape) - 1:
        grouped_layout = local_reduce.grouped_tensors.get(view)
        if grouped_layout is None or grouped_layout.axis != 1:
            return None
        group = grouped_layout.group
        chunked = False
        layout_node = None
    else:
        return None
    return OutputContractionUse(
        source=view_normalized.source,
        group=group,
        chunked=chunked,
        group_indices=(index,),
        layout_node=layout_node,
    )


def collect_output_contraction_uses(
    output: torch.fx.Node,
    gemm: torch.fx.Node,
    local_reduce: FlexGemmLocalReduceAnalysis,
) -> list[tuple[torch.fx.Node, OutputContractionUse]] | None:
    """Collect contraction uses without mutating analysis state."""
    gemm_shape = tensor_meta_shape(gemm)
    uses: list[tuple[torch.fx.Node, OutputContractionUse]] = []
    seen: OrderedSet[torch.fx.Node] = OrderedSet()
    stack: list[Any] = [output]
    while stack:
        node = stack.pop()
        if not isinstance(node, torch.fx.Node) or node in seen:
            continue
        seen.add(node)
        match = match_output_contraction_use(node, gemm, gemm_shape, local_reduce)
        if match is not None:
            uses.append((node, match))
            continue
        if node is gemm or (
            node in local_reduce.grouped_tensors
            and local_reduce.graph.depends_on(node, gemm)
        ):
            return None
        stack.extend(reversed(tuple(iter_fx_node_inputs((node.args, node.kwargs)))))
    return uses or None


def build_output_contraction_plan(
    output: torch.fx.Node,
    gemm: torch.fx.Node,
    local_reduce: FlexGemmLocalReduceAnalysis,
) -> OutputContractionPlan | None:
    """Build a plan when all grouped GEMM values form one output contraction."""
    collected = collect_output_contraction_uses(output, gemm, local_reduce)
    if collected is None:
        return None
    first = collected[0][1]
    source = canonical_output_contraction_source(first.source, gemm, local_reduce)
    group_indices: OrderedSet[int] = OrderedSet()
    select_indices: dict[torch.fx.Node, int] = {}
    grouped_tensors: dict[torch.fx.Node, FlexGemmLocalReduceGeometry] = {}
    for node, match in collected:
        if (
            canonical_output_contraction_source(match.source, gemm, local_reduce)
            is not source
            or match.group != first.group
            or match.chunked != first.chunked
        ):
            return None
        for index in match.group_indices:
            if not -first.group <= index < first.group:
                return None
            group_indices.add(index % first.group)
        if isinstance(local_reduce.graph.normalized_nodes.get(node), NormalizedSelect):
            select_indices[node] = match.group_indices[0] % first.group
        if match.layout_node is not None:
            grouped_tensors[match.layout_node] = FlexGemmLocalReduceGeometry(
                group=match.group, axis=1
            )
    if group_indices != OrderedSet(range(first.group)):
        return None
    gemm_meta = gemm.meta.get("val")
    output_meta = output.meta.get("val")
    if gemm_meta is None or output_meta is None or len(gemm_meta.shape) != 2:
        return None
    expected_shape = (gemm_meta.shape[0], gemm_meta.shape[1] // first.group)
    if not statically_known_shape_equal(output_meta.shape, expected_shape):
        raise NotImplementedError(FLEX_GEMM_OUTPUT_CONTRACTION_SHAPE_ERROR)
    return OutputContractionPlan(
        FlexGemmOutputContraction(group=first.group, chunked=first.chunked),
        select_indices,
        grouped_tensors,
    )


def validate_shape_preserving_main_output(
    outputs: FlexGemmOutputPlan,
    gemm: torch.fx.Node,
    local_reduce: FlexGemmLocalReduceAnalysis,
) -> None:
    """Reject incomplete grouped selects and other shape-changing main outputs."""
    # TODO: Consider DCE before analysis if dead grouped selects become common.
    if any(
        isinstance(normalized, NormalizedSelect)
        and normalized.source in local_reduce.grouped_tensors
        and local_reduce.grouped_tensors[normalized.source].axis == 1
        for normalized in local_reduce.graph.normalized_nodes.values()
    ):
        raise NotImplementedError(
            "FlexGEMM grouped selects must form a complete output contraction"
        )
    output_shape = tensor_meta_shape(outputs.output)
    gemm_shape = tensor_meta_shape(gemm)
    if (
        output_shape is None
        or gemm_shape is None
        or not statically_known_shape_equal(output_shape, gemm_shape)
    ):
        raise NotImplementedError(FLEX_GEMM_MAIN_OUTPUT_SHAPE_ERROR)


@dataclasses.dataclass(frozen=True)
class FlexGemmEpilogueAnalysis:
    """Bundle the immutable analysis consumed by FlexGEMM lowering and emission.

    Attributes:
        gemm: The validated GEMM node shared by lowering and emission.
        outputs: Classification of main, auxiliary, and local-reduction outputs.
        local_reduce: Grouped layouts and local-reduction matches from the FX graph.
        output_contraction_select_indices: Select indices committed by the contraction plan.
    """

    gemm: torch.fx.Node
    outputs: FlexGemmOutputPlan
    local_reduce: FlexGemmLocalReduceAnalysis
    output_contraction_select_indices: dict[torch.fx.Node, int] = dataclasses.field(
        default_factory=dict
    )

    @classmethod
    def from_graph_module(
        cls, graph_module: torch.fx.GraphModule, gemm: torch.fx.Node
    ) -> "FlexGemmEpilogueAnalysis":
        """Analyze grouped values and classify logical output consumers."""
        local_reduce = FlexGemmLocalReduceAnalysis.from_graph_module(graph_module)
        outputs = bind_terminal_output_storage(
            local_reduce.graph, output_plan(graph_module, local_reduce)
        )
        contraction_plan = build_output_contraction_plan(
            outputs.output, gemm, local_reduce
        )
        if contraction_plan is None:
            validate_shape_preserving_main_output(outputs, gemm, local_reduce)
            output_contraction_select_indices = {}
        else:
            outputs = contraction_plan.commit(outputs, local_reduce)
            output_contraction_select_indices = contraction_plan.select_indices
        return cls(gemm, outputs, local_reduce, output_contraction_select_indices)

    @property
    def required_geometries(self) -> tuple[FlexGemmLocalReduceGeometry, ...]:
        """Return every grouped geometry that constrains kernel configuration."""
        geometries = OrderedSet(
            match.geometry for match in self.local_reduce.matches.values()
        )
        if self.outputs.local_reduce is not None:
            geometries.add(self.outputs.local_reduce.match.geometry)
        if self.outputs.output_contraction is not None:
            geometries.add(
                FlexGemmLocalReduceGeometry(
                    group=self.outputs.output_contraction.group,
                    axis=1,
                )
            )
        return tuple(geometries)

    @property
    def reduction_plan(self) -> GemmReductionPlan | None:
        """Return the backend-neutral reduction plan produced by FX analysis."""
        return self.outputs.reduction_plan


def analyze_flex_gemm_epilogue(
    graph_module: torch.fx.GraphModule,
    gemm: torch.fx.Node,
) -> FlexGemmEpilogueAnalysis:
    """Analyze FlexGEMM body for output planning and epilogue code generation.

    This is the analysis entry point called by FlexGEMM lowering. It builds a
    dependency index, performs topological local-reduction analysis, and
    returns the shared immutable plan consumed by config selection and
    ``materialize_flex_gemm_epilogue``.

    Args:
        graph_module: FlexGEMM body graph containing GEMM and epilogue nodes.

    Returns:
        Output and local-reduction analysis shared by later lowering phases.
    """
    return FlexGemmEpilogueAnalysis.from_graph_module(graph_module, gemm)


def gemm_node(
    graph_module: torch.fx.GraphModule, gemm_op: torch._ops.OpOverload
) -> torch.fx.Node:
    gemm_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target == gemm_op
    ]
    if len(gemm_nodes) != 1:
        raise NotImplementedError("FlexGEMM expects one GEMM body")
    return gemm_nodes[0]


class FlexGemmEpilogueEmitter:
    """Visit an analyzed FlexGEMM FX graph and emit its CuTeDSL epilogue.

    The analysis dataclasses flow into each other as follows:

    ::

        FlexGemmEpilogueGraph
          `--> FlexGemmLocalReduceAnalysis
                 +--> grouped_tensors
                 `--> matches
                        `--> FlexGemmLocalReduceMatch
                               `--> FlexGemmOutputLocalReducePlan
                                      `--> optional FlexGemmLocalReduceStore

        FlexGemmLocalReduceAnalysis
          `--> output_plan()
                 `--> FlexGemmOutputPlan

        FlexGemmLocalReduceAnalysis + FlexGemmOutputPlan
          `--> FlexGemmEpilogueAnalysis
                 `--> FlexGemmEpilogueEmitter

    At emitter construction, ``analysis.outputs`` becomes ``self.outputs``;
    its local-reduce match and optional store initialize ``self.feed_main`` and
    ``self.aux``. ``analysis.local_reduce.grouped_tensors`` is copied into
    mutable emission state, while ``analysis.required_geometries`` determines
    the active grouped layouts.

    The emitter owns all mutable code-generation state: FX values lowered so far,
    grouped TensorSSA layouts, compressed-store expressions, and physical
    reduction callbacks. ``lower_graph`` performs a topological traversal and
    delegates each ``call_function`` node to ordered handlers; ``render`` turns
    the resulting state into the generated epilogue and callback source.
    """

    def __init__(
        self,
        graph_module: torch.fx.GraphModule,
        analysis: FlexGemmEpilogueAnalysis,
        epilogue_arg_placeholders: tuple[torch.fx.Node, ...] = (),
        *,
        fast_math: bool = False,
    ) -> None:
        self.graph_module = graph_module
        self.epilogue_arg_placeholders = epilogue_arg_placeholders
        self.fast_math = fast_math
        self.gemm = analysis.gemm
        self.graph = analysis.local_reduce.graph
        self.outputs = analysis.outputs
        self.output_contraction_select_indices = (
            analysis.output_contraction_select_indices
        )
        self.kernel = FlexGemmCuteDSLKernel()
        self.env: dict[torch.fx.Node, Any] = {
            self.gemm: CuteDSLCSEVariable(
                "acc", ValueRanges.unknown(), dtype=torch.float32, shape=(1,)
            )
        }
        self.grouped_tensors = {
            node: GroupedTensorSSALayout(group=layout.group, axis=layout.axis)
            for node, layout in analysis.local_reduce.grouped_tensors.items()
        }
        self.active_grouped_layouts = OrderedSet(
            GroupedTensorSSALayout(group=geometry.group, axis=geometry.axis)
            for geometry in analysis.required_geometries
        )
        self.store_sources: dict[torch.fx.Node, Any] = {}
        self.physical_reductions: dict[torch.fx.Node, FlexGemmPhysicalReduction] = {}
        self.local_reduce = self.outputs.local_reduce
        self.feed_main: torch.fx.Node | None = None
        self.aux: torch.fx.Node | None = None
        self.feed_main_input: torch.fx.Node | None = None
        match self.local_reduce:
            case FlexGemmOutputLocalReducePlan(
                match=local_reduce_match, store=store, feeds_main=True
            ):
                self.feed_main = local_reduce_match.value_node
                reduction = self.graph.normalized_nodes.get(
                    local_reduce_match.value_node
                )
                if not isinstance(reduction, NormalizedReduction):
                    raise AssertionError("feed-main plans require a matched reduction")
                self.feed_main_input = reduction.source
                self.aux = None if store is None else store.node
            case FlexGemmOutputLocalReducePlan(
                store=FlexGemmLocalReduceStore(node=store_node)
            ):
                self.aux = store_node
            case None:
                pass

    def bind_epilogue_args(self) -> None:
        """Bind captured tensor placeholders to generated CuTeDSL parameters."""
        for index, node in enumerate(self.epilogue_arg_placeholders):
            epilogue_arg_meta = node.meta["val"]
            physical_dtype = (
                torch.uint8
                if epilogue_arg_meta.dtype is torch.bool
                else epilogue_arg_meta.dtype
            )
            logical_dtype = upcast_compute_type(epilogue_arg_meta.dtype)
            self.env[node] = CuteDSLCSEVariable(
                f"aux{index}",
                ValueRanges.unknown(),
                dtype=physical_dtype,
                shape=(1,),
                is_scalar_expr=self.outputs.output_contraction is not None,
            )
            if logical_dtype != physical_dtype:
                self.env[node] = FlexGemmCuteDSLOpOverrides.to_dtype(
                    self.env[node], logical_dtype, use_compute_types=False
                )

    def bind_reduction(self, node: torch.fx.Node, lowered_reduce: Any) -> None:
        """Bind a generated reduction or replace it with the feed-main parameter."""
        if self.feed_main is not None and node is self.feed_main:
            self.env[node] = CuteDSLCSEVariable(
                LOCAL_REDUCE_FEED_MAIN_ARG_NAME,
                ValueRanges.unknown(),
                dtype=lowered_reduce.dtype,
                shape=lowered_reduce.shape,
            )
            if self.feed_main_input in self.grouped_tensors:
                self.grouped_tensors[node] = self.grouped_tensors[self.feed_main_input]
            return
        self.env[node] = lowered_reduce

    def lower_compact_inline_asm(self, node: torch.fx.Node) -> bool:
        """Apply inline asm before broadcasting a compact reduction value."""
        if (
            self.feed_main is not None
            or _cute_op_name(node.target) != "inline_asm_elementwise"
            or len(node.args) != 1
        ):
            return False
        tensor_inputs = [
            input_node
            for input_node in node.all_input_nodes
            if tensor_meta_shape(input_node) is not None
        ]
        if len(tensor_inputs) != 1:
            return False
        source = tensor_inputs[0]
        normalized = self.graph.normalized_nodes.get(source)
        compact_source = _cute_arg(source, self.env)
        if not isinstance(normalized, NormalizedReduction):
            if not is_shape_preserving_pointwise_node(source):
                return False
            source_inputs = [
                input_node
                for input_node in source.all_input_nodes
                if tensor_meta_shape(input_node) is not None
            ]
            if len(source_inputs) != 1:
                return False
            normalized = self.graph.normalized_nodes.get(source_inputs[0])
            if not isinstance(normalized, NormalizedReduction):
                return False
            compact_source = _cute_call(
                source,
                tuple(_cute_arg(arg, self.env) for arg in source.args),
                {
                    key: _cute_arg(value, self.env)
                    for key, value in source.kwargs.items()
                },
            )
        reduction_input = normalized.source
        layout = self.grouped_tensors.get(reduction_input)
        if layout is None:
            return False
        self.env[node] = _cute_call(
            node,
            (compact_source,),
            {key: _cute_arg(value, self.env) for key, value in node.kwargs.items()},
        )
        _, self.store_sources[node] = _keepdim_and_broadcast(
            self.kernel,
            self.env[node],
            layout,
            _cute_arg(reduction_input, self.env),
        )
        return True

    def lower_pointwise_store(self, node: torch.fx.Node) -> bool:
        """Lower pointwise expressions that consume a compressed store value."""
        if (
            self.feed_main is not None
            or not is_shape_preserving_pointwise_node(node)
            or not any(
                arg in self.store_sources
                for arg in iter_fx_node_inputs((node.args, tuple(node.kwargs.values())))
            )
        ):
            return False
        store_args = tuple(
            _local_reduce_store_arg(arg, self.env, self.store_sources)
            for arg in node.args
        )
        store_kwargs = {
            key: _local_reduce_store_arg(value, self.env, self.store_sources)
            for key, value in node.kwargs.items()
        }
        self.env[node] = _cute_call(node, store_args, store_kwargs)
        self.store_sources[node] = self.env[node]
        return True

    def propagate_physical_reduction(self, node: torch.fx.Node, source: Any) -> None:
        """Preserve physical callback provenance through shape-only wrappers."""
        if isinstance(source, torch.fx.Node) and source in self.physical_reductions:
            self.physical_reductions[node] = self.physical_reductions[source]

    def physical_finalize_arg(self, value: Any) -> Any:
        """Replace physical reduction inputs with their generated value expression."""
        if isinstance(value, torch.fx.Node) and value in self.physical_reductions:
            return self.physical_reductions[value].finalize_expr
        if isinstance(value, (tuple, list)):
            return type(value)(self.physical_finalize_arg(item) for item in value)
        return _cute_arg(value, self.env)

    def compose_physical_finalize(self, node: torch.fx.Node) -> Any | None:
        """Fold a pointwise consumer into one generated physical finalizer."""
        physical_inputs = list(
            OrderedSet(
                arg
                for arg in iter_fx_node_inputs((node.args, node.kwargs))
                if arg in self.physical_reductions
            )
        )
        if not physical_inputs:
            return None
        if len(physical_inputs) > 1:
            raise NotImplementedError(LOCAL_REDUCE_SINGLE_PHYSICAL_FINALIZE_ERROR)
        base = physical_inputs[0]
        args = tuple(self.physical_finalize_arg(arg) for arg in node.args)
        kwargs = {
            key: self.physical_finalize_arg(value) for key, value in node.kwargs.items()
        }
        finalize_expr = _cute_call(node, args, kwargs)
        if not isinstance(finalize_expr, str):
            raise NotImplementedError(LOCAL_REDUCE_FINALIZE_SCALAR_ONLY_ERROR)
        self.store_sources[node] = self.store_sources[base]
        self.physical_reductions[node] = dataclasses.replace(
            self.physical_reductions[base], finalize_expr=finalize_expr
        )
        return finalize_expr

    def lower_call_function(self, node: torch.fx.Node) -> None:
        """Lower one call_function node using the ordered FlexGEMM handlers."""
        lowered = lower_full_scalar(node)
        if lowered is not None:
            self.env[node] = lowered
            return
        normalized = self.graph.normalized_nodes.get(node)
        match normalized:
            case NormalizedDtypeView():
                if (
                    node is not self.outputs.output
                    or normalized.source is not self.outputs.output_storage
                ):
                    raise NotImplementedError(
                        "FlexGEMM supports dtype views only as terminal same-width main outputs"
                    )
                self.env[node] = _cute_arg(normalized.source, self.env)
                return
            case NormalizedSqueeze():
                lowered = lower_squeeze(node, normalized, self.env, self.store_sources)
                if lowered is not None:
                    self.env[node] = lowered
                    self.propagate_physical_reduction(node, normalized.source)
                    return
            case NormalizedSplit():
                lowered = lower_grouped_n_split(
                    node, normalized, self.env, self.kernel, self.grouped_tensors
                )
                if lowered is not None:
                    self.env[node] = lowered
                    return
            case NormalizedSelect():
                index = self.output_contraction_select_indices.get(node)
                if index is not None:
                    self.env[node] = lower_grouped_n_select(
                        normalized, index, self.env, self.kernel
                    )
                    return
            case NormalizedGetItem():
                lowered = lower_getitem(node, normalized, self.env, self.store_sources)
                if lowered is not None:
                    self.env[node] = lowered
                    self.propagate_physical_reduction(node, normalized.source)
                    return
            case NormalizedPrepareSoftmax():
                self.env[node] = lower_prepare_softmax_online(
                    node,
                    normalized,
                    self.env,
                    self.kernel,
                    self.grouped_tensors,
                    self.store_sources,
                )
                return
            case NormalizedView():
                lowered = lower_view_or_reshape(
                    node,
                    normalized,
                    self.env,
                    self.kernel,
                    self.grouped_tensors,
                    self.active_grouped_layouts,
                    self.store_sources,
                    node is self.feed_main_input,
                )
                if lowered is not None:
                    self.env[node] = lowered
                    self.propagate_physical_reduction(node, normalized.source)
                    return
            case NormalizedReduction():
                self.bind_reduction(
                    node,
                    lower_tensorssa_reduce(
                        node,
                        normalized,
                        self.env,
                        self.kernel,
                        self.grouped_tensors,
                        self.store_sources,
                        self.physical_reductions,
                    ),
                )
                return
            case NormalizedUnsupportedReduction():
                raise local_reduce_unsupported_tensorssa_error(
                    normalized.target, value_only=True
                )
        is_shape_preserving = is_shape_preserving_pointwise_node(node)
        if is_shape_preserving and self.feed_main is None:
            if self.aux is None and any(
                arg in self.physical_reductions
                for arg in iter_fx_node_inputs((node.args, node.kwargs))
            ):
                raise NotImplementedError(LOCAL_REDUCE_POST_POINTWISE_FINALIZE_ERROR)
            physical_finalize = self.compose_physical_finalize(node)
            if physical_finalize is not None:
                self.env[node] = physical_finalize
                return
        if self.lower_compact_inline_asm(node) or self.lower_pointwise_store(node):
            return
        node_args = tuple(_cute_arg(arg, self.env) for arg in node.args)
        node_kwargs = {
            key: _cute_arg(value, self.env) for key, value in node.kwargs.items()
        }
        self.env[node] = _cute_call(node, node_args, node_kwargs)

    def lower_graph(self) -> None:
        """Lower body nodes in FX topological order."""
        for node in self.graph_module.graph.nodes:
            if node is self.gemm or node.op in ("placeholder", "output"):
                continue
            if isinstance(node.meta.get("val"), (int, torch.SymInt)):
                continue
            with V.set_current_node(node):
                if node.op != "call_function":
                    raise NotImplementedError(
                        f"unsupported FlexGEMM epilogue node: {node.format_node()}"
                    )
                self.lower_call_function(node)

    @staticmethod
    def aux_result(
        aux: torch.fx.Node | None, store_sources: dict[torch.fx.Node, Any]
    ) -> Any | None:
        """Return the compressed-aux expression or reject missing TensorSSA."""
        if aux is None:
            return None
        result = store_sources.get(aux)
        if result is None:
            raise NotImplementedError(LOCAL_REDUCE_AUX_TENSORSSA_ERROR)
        return result

    def render(self) -> tuple[str, str]:
        """Render the generated epilogue and physical callback source."""
        from torch._inductor.codegen.cutedsl._inline_asm import inline_asm_cache_key

        body = "\n".join(f"    {line}" for line in self.kernel.body.lines)
        if body:
            body += "\n"
        aux_args = [
            f"aux{index}" for index in range(len(self.epilogue_arg_placeholders))
        ]
        feed_main_args = (
            [LOCAL_REDUCE_FEED_MAIN_ARG_NAME] if self.feed_main is not None else []
        )
        epilogue_params = ", ".join(["acc", *aux_args, *feed_main_args])
        result = _cute_arg(self.outputs.output, self.env)
        aux_result = self.aux_result(self.aux, self.store_sources)
        if self.outputs.aux_outputs or aux_result is not None:
            tuple_items = [result]
            tuple_items.extend(
                _cute_arg(aux_output, self.env)
                for aux_output in self.outputs.aux_outputs
            )
            if aux_result is not None:
                tuple_items.append(aux_result)
            result = f"({', '.join(str(item) for item in tuple_items)})"
        physical_reduction = (
            None
            if self.local_reduce is None
            else self.physical_reductions.get(self.local_reduce.match.value_node)
        )
        physical_reduction_payload = (
            ""
            if physical_reduction is None
            else (
                f"\ncombine {physical_reduction.combine_expr}"
                f"\nfinalize {physical_reduction.finalize_expr}"
            )
        )
        key_payload = (
            f"inline_asm={inline_asm_cache_key()}\n"
            f"fast_math={self.fast_math}\n{self.graph_module.code}\n"
            f"{body}\nreturn {result}{physical_reduction_payload}"
        )
        key = hashlib.sha256(key_payload.encode()).hexdigest()[:16]
        name = f"flex_gemm_epilogue_{key}"
        local_reduce_source = ""
        if physical_reduction is not None:
            combine_name = f"{name}{LOCAL_REDUCE_COMBINE_FN_SUFFIX}"
            finalize_name = f"{name}{LOCAL_REDUCE_FINALIZE_FN_SUFFIX}"
            local_reduce_source = (
                f"@cute.jit\ndef {combine_name}(lhs, rhs):\n"
                f"    return {physical_reduction.combine_expr}\n"
                f"{combine_name}.__cache_key__ = lambda: {combine_name!r}\n\n"
                f"@cute.jit\ndef {finalize_name}(value):\n"
                f"    return {physical_reduction.finalize_expr}\n"
                f"{finalize_name}.__cache_key__ = lambda: {finalize_name!r}\n\n"
            )
        return (
            name,
            "import cutlass\n"
            "import cutlass.cute as cute\n"
            "import operator\n"
            "from cutlass._mlir.dialects import math as mlir_math\n"
            "from cutlass._mlir_helpers import math as cutlass_math\n"
            "from torch._inductor.codegen.cutedsl._inline_asm import (\n"
            "    inline_asm_elementwise_intrinsic,\n"
            ")\n\n"
            f"{local_reduce_source}"
            f"@cute.jit\ndef {name}({epilogue_params}):\n"
            f"{body}    return {result}\n",
        )

    def materialize(self) -> tuple[str, str]:
        """Lower and render this epilogue under the CuTeDSL virtualized handlers."""
        with (
            V.set_kernel_handler(self.kernel),
            V.set_ops_handler(FlexGemmCuteDSLOpOverrides()),
            use_cutedsl_fast_math(self.fast_math),
        ):
            self.bind_epilogue_args()
            self.lower_graph()
        return self.render()


def materialize_flex_gemm_epilogue(
    graph_module: torch.fx.GraphModule,
    analysis: FlexGemmEpilogueAnalysis,
    epilogue_arg_placeholders: tuple[torch.fx.Node, ...] = (),
    *,
    fast_math: bool = False,
) -> tuple[str, str]:
    """Materialize an analyzed FlexGEMM body as generated CuTeDSL source.

    This is the code-generation entry point called by FlexGEMM lowering after
    ``analyze_flex_gemm_epilogue`` has classified outputs and local-reduction
    matches. The emitter visits the FX graph once in topological order while
    owning the environment and reduction state needed across nodes.

    Args:
        graph_module: FlexGEMM body graph containing the GEMM and epilogue nodes.
        gemm_op: GEMM overload expected to occur exactly once in the body.
        analysis: Shared output and local-reduction analysis for the graph.
        epilogue_arg_placeholders: Captured tensor placeholders exposed as
            generated epilogue parameters.
        fast_math: Whether supported CuTeDSL math operations may use approximate
            fast-math lowering.

    Returns:
        The generated epilogue function name and complete CuTeDSL source.
    """
    return FlexGemmEpilogueEmitter(
        graph_module,
        analysis,
        epilogue_arg_placeholders,
        fast_math=fast_math,
    ).materialize()
<|endoftext|># mypy: allow-untyped-defs
"""Lower FlexGEMM grouped local reductions into QuACK/CuTeDSL epilogues.

FlexGEMM recognizes a narrow local-reduction contract inside the GEMM output
tile: an epilogue reshapes the accumulator to expose contiguous groups along M
or N, then reduces only that grouped dimension. N-axis groups up to one 32-lane
fragment lower as ordinary in-fragment TensorSSA reductions; larger N groups
produce TensorSSA partials that QuACK combines physically.
M-axis groups currently always use QuACK's physical row-lane/warp combine path,
even when the group is small enough to fit in one fragment. Inductor owns FX
normalization and output contracts; these helpers describe the supported
TensorSSA shapes and generated combine/finalize expressions QuACK needs.

The main caller is ``materialize_flex_gemm_epilogue`` in ``fx_cutedsl_codegen.py``.
Shared GEMM epilogue analysis recognizes grouped layouts and reductions;
materialization wraps accepted geometries in ``GroupedTensorSSALayout`` and
routes FX nodes through ``lower_view_or_reshape``,
``lower_prepare_softmax_online``, and ``lower_tensorssa_reduce``.
"""

import dataclasses
import math
from typing import Any, cast

import torch
from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
    CuteDSLOpOverrides,
    tensorssa_reduction,
)
from torch._inductor.kernel.flex_gemm.constraints import (
    LOCAL_REDUCE_EXPLICIT_DTYPE_ERROR,
    LOCAL_REDUCE_INNERMOST_GROUPED_DIM_ERROR,
    LOCAL_REDUCE_PARTIAL_OUTPUT_CONTRACT_ERROR,
)
from torch._inductor.kernel.gemm_epilogue import (
    GemmReductionGeometry,
    iter_fx_node_inputs,
    NormalizedGetItem,
    NormalizedPrepareSoftmax,
    NormalizedReduction,
    NormalizedSelect,
    NormalizedSplit,
    NormalizedSqueeze,
    NormalizedView,
)
from torch._inductor.kernel.gemm_epilogue_utils import normalize_shape
from torch._inductor.ops_handler import ReductionType
from torch._inductor.shape_propagation import get_broadcasted_shape
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet


@dataclasses.dataclass(frozen=True)
class GroupedTensorSSALayout(GemmReductionGeometry):
    """Describe a grouped M/N TensorSSA view inside the generated epilogue."""

    def fragment_group_size_expr(self, source: Any) -> str:
        """Return the local group size available in this epilogue fragment."""
        return (
            f"cutlass.const_expr(min({self.group_size}, "
            f"cute.size({source}.shape, mode=[0])))"
        )

    def fragment_repeat_expr(self, source: Any) -> str:
        """Return the repeat count needed to cover the current epilogue fragment."""
        return (
            f"cutlass.const_expr(cute.size({source}.shape, mode=[0]) "
            f"// min({self.group_size}, cute.size({source}.shape, mode=[0])))"
        )

    def tensorssa_shape(self, source: Any) -> str:
        fragment_group_size = self.fragment_group_size_expr(source)
        repeats = self.fragment_repeat_expr(source)
        if self.axis == 1:
            return f"((1, {fragment_group_size}, {repeats}), 1, 1)"
        return f"(({fragment_group_size}, 1, {repeats}), 1, 1)"

    def keepdim_shape(self, source: Any) -> str:
        return f"((1, 1, {self.fragment_repeat_expr(source)}), 1, 1)"

    @property
    def needs_physical_combine(self) -> bool:
        return self.needs_physical_callbacks

    @property
    def reduction_profile(self) -> str:
        if self.axis == 1:
            return "((None, 1, None), 1, 1)"
        return "((1, None, None), 1, 1)"


def _cute_op_name(target: Any) -> str | None:
    if isinstance(target, torch._ops.OpOverload):
        op_name = target.overloadpacket.__name__
    elif isinstance(target, str):
        op_name = target
    else:
        op_name = target.__name__ if callable(target) else None
    return "truediv" if op_name == "div" else op_name


@dataclasses.dataclass(frozen=True)
class FlexGemmPhysicalReduction:
    """Describe QuACK's physical combine/finalize callback for local reductions."""

    combine_expr: str
    finalize_expr: str = "value"


FLEX_GEMM_POINTWISE_OP_NAMES = frozenset(
    (
        "_to_copy",
        "clamp",
        "clamp_max",
        "clamp_min",
        "convert_element_type",
        "inline_asm_elementwise",
    )
)


def _cute_arg(value: Any, env: dict[torch.fx.Node, Any]) -> Any:
    """Translate FX node references and constants into CuTeDSL epilogue values."""
    if isinstance(value, torch.fx.Node):
        if value in env:
            return env[value]
        raise NotImplementedError(
            f"unsupported FlexGEMM epilogue dependency: {value.format_node()}"
        )
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return 'float("nan")'
        return 'float("inf")' if value > 0 else 'float("-inf")'
    if isinstance(
        value,
        (
            int,
            float,
            bool,
            str,
            torch.dtype,
            torch.device,
            torch.layout,
            torch.memory_format,
        ),
    ):
        return value
    if isinstance(value, (tuple, list)):
        return type(value)(_cute_arg(item, env) for item in value)
    raise NotImplementedError(f"unsupported FlexGEMM epilogue constant: {value!r}")


def _generate_like(
    kernel: Any, expr: str, ref: Any, shape_ref: Any | None = None
) -> Any:
    """Emit CuTeDSL while preserving dtype and shape metadata from references."""
    if shape_ref is None:
        shape_ref = ref
    return kernel.cse.generate(
        kernel.body,
        expr,
        dtype=getattr(ref, "dtype", None),
        shape=getattr(shape_ref, "shape", None),
    )


def _keepdim_and_broadcast(
    kernel: Any, reduced: Any, layout: GroupedTensorSSALayout, source: Any
) -> tuple[Any, Any]:
    """Materialize keepdim and store-shaped forms of a grouped reduction."""
    keepdim_source = _generate_like(
        kernel, f"{reduced}.reshape({layout.keepdim_shape(source)})", reduced
    )
    return keepdim_source, _generate_like(
        kernel, f"{keepdim_source}.broadcast_to({source}.shape)", keepdim_source, source
    )


def _cute_call(
    node: torch.fx.Node,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Lower one FX call through the active CuTeDSL operations handler."""
    target = node.target
    op_name = _cute_op_name(target)
    if op_name is None:
        raise NotImplementedError(f"unsupported FlexGEMM epilogue op: {target}")
    if op_name == "inline_asm_elementwise":
        # The HOP spells the asm text `asm_str`; the ops handler spells it `asm`.
        kwargs = dict(kwargs)
        kwargs["asm"] = kwargs.pop("asm_str")
        input_values = []
        for input_node in node.args[: len(args)]:
            value = (
                input_node.meta.get("val")
                if isinstance(input_node, torch.fx.Node)
                else None
            )
            if not isinstance(value, torch.Tensor):
                raise NotImplementedError(
                    "FlexGEMM inline asm inputs require tensor metadata"
                )
            input_values.append(value)
        kwargs["input_dtypes"] = tuple(value.dtype for value in input_values)
        kwargs["scalar_sources"] = tuple(
            all(isinstance(dim, int) and dim == 1 for dim in value.shape)
            for value in input_values
        )
    try:
        op = getattr(V.get_ops_handler(), op_name)
    except AttributeError:
        raise NotImplementedError(
            f"unsupported FlexGEMM epilogue op: {target}"
        ) from None
    return op(*args, **kwargs)


def _local_reduce_store_arg(
    value: Any, env: dict[torch.fx.Node, Any], sources: dict[torch.fx.Node, Any]
) -> Any:
    if isinstance(value, torch.fx.Node) and value in sources:
        return sources[value]
    if isinstance(value, (tuple, list)):
        return type(value)(
            _local_reduce_store_arg(item, env, sources) for item in value
        )
    return _cute_arg(value, env)


def tensor_meta_shape(node: torch.fx.Node) -> tuple[Any, ...] | None:
    """Return fake-tensor shape metadata when the FX value is tensor-like."""
    meta = node.meta.get("val")
    if isinstance(meta, torch.Tensor):
        return tuple(meta.shape)
    return None


def node_preserves_tensor_shapes(node: torch.fx.Node) -> bool:
    """Reject pointwise broadcasts that cannot preserve a grouped TensorSSA input."""
    output_shape = tensor_meta_shape(node)
    if output_shape is None:
        return False
    output_shape_key = tuple(str(dim) for dim in output_shape)
    has_same_shape_input = False
    for input_node in iter_fx_node_inputs((node.args, node.kwargs)):
        input_shape = tensor_meta_shape(input_node)
        if input_shape is None:
            continue
        input_shape_key = tuple(str(dim) for dim in input_shape)
        if input_shape_key == output_shape_key:
            has_same_shape_input = True
            continue
        try:
            broadcast_shape = get_broadcasted_shape(input_shape_key, output_shape_key)
        except AssertionError:
            return False
        if broadcast_shape is None or tuple(broadcast_shape) != output_shape_key:
            return False
    return has_same_shape_input


def is_pointwise_node(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    return (
        isinstance(node.target, torch._ops.OpOverload)
        and torch.Tag.pointwise in node.target.tags
    ) or _cute_op_name(node.target) in FLEX_GEMM_POINTWISE_OP_NAMES


def is_shape_preserving_pointwise_node(node: torch.fx.Node) -> bool:
    return is_pointwise_node(node) and node_preserves_tensor_shapes(node)


def lower_view_or_reshape(
    node: torch.fx.Node,
    normalized: NormalizedView,
    env: dict[torch.fx.Node, Any],
    kernel: Any,
    grouped_tensors: dict[torch.fx.Node, GroupedTensorSSALayout],
    active_grouped_layouts: OrderedSet[GroupedTensorSSALayout],
    local_reduce_store_sources: dict[torch.fx.Node, Any],
    preserve_value_layout: bool = False,
) -> Any | None:
    """Emit an analyzed view using grouped provenance."""
    source_node = normalized.source
    if source_node in local_reduce_store_sources:
        local_reduce_store_sources[node] = local_reduce_store_sources[source_node]
        return _cute_arg(source_node, env)
    source = _cute_arg(source_node, env)
    grouped_layout = grouped_tensors.get(node)
    if grouped_layout is not None:
        if preserve_value_layout or grouped_layout not in active_grouped_layouts:
            return source
        return _generate_like(
            kernel,
            f"{source}.reshape({grouped_layout.tensorssa_shape(source)})",
            source,
        )
    if source_node in grouped_tensors:
        return source
    return None


def lower_grouped_n_split(
    node: torch.fx.Node,
    normalized: NormalizedSplit,
    env: dict[torch.fx.Node, Any],
    kernel: Any,
    grouped_tensors: dict[torch.fx.Node, GroupedTensorSSALayout],
) -> tuple[Any, ...] | None:
    """Split an analyzed output contraction into grouped TensorSSA values."""
    layout = grouped_tensors.get(node)
    if layout is None or layout.axis != 1:
        return None
    source = _cute_arg(normalized.source, env)
    grouped = _generate_like(
        kernel, f"{source}.reshape({layout.tensorssa_shape(source)})", source
    )
    return tuple(
        _generate_like(
            kernel,
            f"{grouped}[((0, {index}, None), None, None)]",
            grouped,
        )
        for index in range(layout.group)
    )


def lower_grouped_n_select(
    normalized: NormalizedSelect,
    index: int,
    env: dict[torch.fx.Node, Any],
    kernel: Any,
) -> Any:
    """Select one analysis-validated output-contraction group value."""
    source = _cute_arg(normalized.source, env)
    return _generate_like(
        kernel,
        f"{source}[((0, {index}, None), None, None)]",
        source,
    )


def lower_full_scalar(node: torch.fx.Node) -> Any | None:
    if node.op != "call_function" or node.target is not torch.ops.aten.full.default:
        return None
    shape = normalize_shape(node.args[0])
    if shape != ():
        return None
    value = node.args[1]
    return value if isinstance(value, (bool, int, float)) else None


def lower_squeeze(
    node: torch.fx.Node,
    normalized: NormalizedSqueeze,
    env: dict[torch.fx.Node, Any],
    local_reduce_store_sources: dict[torch.fx.Node, Any],
) -> Any | None:
    """Forward an analyzed squeeze alias."""
    source_node = normalized.source
    if source_node not in env:
        return None
    if source_node in local_reduce_store_sources:
        local_reduce_store_sources[node] = local_reduce_store_sources[source_node]
    return _cute_arg(source_node, env)


def lower_getitem(
    node: torch.fx.Node,
    normalized: NormalizedGetItem,
    env: dict[torch.fx.Node, Any],
    local_reduce_store_sources: dict[torch.fx.Node, Any],
) -> Any | None:
    """Index an analyzed aggregate value."""
    source_node, index = normalized.source, normalized.index
    source = _cute_arg(source_node, env)
    if not isinstance(source, (tuple, list)) or not -len(source) <= index < len(source):
        return None
    if source_node in local_reduce_store_sources:
        local_reduce_store_sources[node] = local_reduce_store_sources[source_node][
            index
        ]
    return source[index]


def lower_prepare_softmax_online(
    node: torch.fx.Node,
    normalized: NormalizedPrepareSoftmax,
    env: dict[torch.fx.Node, Any],
    kernel: Any,
    grouped_tensors: dict[torch.fx.Node, GroupedTensorSSALayout],
    local_reduce_store_sources: dict[torch.fx.Node, Any],
) -> Any:
    """Lower analyzed online-softmax preparation."""
    input_node, dim = normalized.source, normalized.dim
    if input_node not in grouped_tensors:
        raise NotImplementedError(LOCAL_REDUCE_PARTIAL_OUTPUT_CONTRACT_ERROR)
    layout = grouped_tensors[input_node]
    if layout.needs_physical_callbacks:
        raise NotImplementedError(
            "unsupported FlexGEMM physical local reduction: prepare_softmax_online "
            "needs a multi-value generated physical reducer"
        )
    if not layout.matches_reduction_dim(dim):
        raise NotImplementedError(
            "unsupported FlexGEMM epilogue local reduction: prepare_softmax_online "
            "currently supports only the grouped dimension"
        )
    source = _cute_arg(input_node, env)
    max_reduced = _generate_like(
        kernel,
        f'{source}.reduce(cute.ReductionOp.MAX, init_val=float("-inf"), reduction_profile={layout.reduction_profile})',
        source,
    )
    _, max_store = _keepdim_and_broadcast(kernel, max_reduced, layout, source)
    centered = _generate_like(kernel, f"({source} - {max_store})", source)
    exp_centered = CuteDSLOpOverrides.exp(centered)
    sum_reduced = _generate_like(
        kernel,
        f"{exp_centered}.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile={layout.reduction_profile})",
        exp_centered,
    )
    _, sum_store = _keepdim_and_broadcast(kernel, sum_reduced, layout, source)
    local_reduce_store_sources[node] = (max_store, sum_store)
    return max_store, sum_store


def lower_tensorssa_reduce(
    node: torch.fx.Node,
    normalized: NormalizedReduction,
    env: dict[torch.fx.Node, Any],
    kernel: Any,
    grouped_tensors: dict[torch.fx.Node, GroupedTensorSSALayout],
    local_reduce_store_sources: dict[torch.fx.Node, Any],
    local_reduce_physical_reductions: dict[torch.fx.Node, FlexGemmPhysicalReduction],
) -> Any:
    """Lower an analyzed reduction while deferring physical finalization."""
    input_node = normalized.source
    dim = normalized.dim
    keepdim = normalized.keepdim
    dtype = normalized.dtype
    reduction_type = normalized.reduction_type
    if dtype is not None:
        raise NotImplementedError(LOCAL_REDUCE_EXPLICIT_DTYPE_ERROR)
    if input_node not in grouped_tensors:
        raise NotImplementedError(LOCAL_REDUCE_PARTIAL_OUTPUT_CONTRACT_ERROR)
    layout = grouped_tensors[input_node]
    if not layout.matches_reduction_dim(dim):
        raise NotImplementedError(LOCAL_REDUCE_INNERMOST_GROUPED_DIM_ERROR)
    reduction_name = cast(
        ReductionType, "sum" if reduction_type == "mean" else reduction_type
    )
    desc = tensorssa_reduction(reduction_name)
    finalize_expr = f"value / {layout.group}.0" if reduction_type == "mean" else "value"
    source = _cute_arg(input_node, env)
    needs_physical_callbacks = layout.needs_physical_callbacks
    if needs_physical_callbacks:
        local_reduce_physical_reductions[node] = FlexGemmPhysicalReduction(
            desc.combine_expr, finalize_expr
        )
        if layout.axis == 0:
            local_reduce_store_sources[node] = source
            return source
    reduced = _generate_like(
        kernel,
        f"{source}.reduce({desc.cute_op}, init_val={desc.init_val}, reduction_profile={layout.reduction_profile})",
        source,
    )
    if reduction_type == "mean" and not needs_physical_callbacks:
        reduced = _generate_like(
            kernel, f"{reduced} / {float(layout.group)!r}", reduced
        )
    keepdim_source, local_reduce_store_sources[node] = _keepdim_and_broadcast(
        kernel, reduced, layout, source
    )
    if keepdim:
        return keepdim_source
    return reduced
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import contextlib
import dataclasses
import os
from typing import Any, TYPE_CHECKING

import torch
from torch._inductor.kernel.flex_gemm.constraints import (
    FLEX_GEMM_CHUNKED_CONTIGUOUS_B_ERROR,
    FLEX_GEMM_OUTPUT_CONTRACTION_CAPTURE_ERROR,
    FLEX_GEMM_OUTPUT_CONTRACTION_COMPOSITION_ERROR,
    FLEX_GEMM_OUTPUT_CONTRACTION_SHAPE_ERROR,
    FlexGemmLocalReduceCallbacks,
    FlexGemmLocalReduceGeometry,
    FlexGemmOutputContraction,
    LOCAL_REDUCE_CALLBACKS_REQUIRED_ERROR,
    LOCAL_REDUCE_COMBINE_KEY_SUFFIX,
    local_reduce_compressed_shape,
    LOCAL_REDUCE_FINALIZE_KEY_SUFFIX,
    LOCAL_REDUCE_RUNTIME_OUT_ERROR,
    LOCAL_REDUCE_SWAP_AB_ERROR,
    output_contraction_capture_supported,
    output_contraction_config_supported,
    validate_local_reduce_feed_main_capability,
    validate_local_reduce_no_c_alpha_beta,
    validate_local_reduce_out_shape,
    validate_local_reduce_runtime_dense_mm,
    validate_local_reduce_selected_dim_divisible,
)
from torch._inductor.runtime.cache_dir_utils import cache_dir
from torch._prims_common import is_expandable_to


if TYPE_CHECKING:
    from torch._inductor.heuristics.template.flex_gemm import GemmConfigKey


# swap_ab transposes the dispatched GEMM, so a row broadcast becomes a col
# broadcast (and vice versa) while tile broadcasts only transpose their data and
# scalars are orientation-invariant.
_SWAPPED_ARG_KIND = {"row": "col", "col": "row", "tile": "tile", "scalar": "scalar"}


def inductor_quack_cache_dir() -> str:
    """Return the Inductor-owned QuACK cache root for generated FlexGEMM."""
    return os.path.join(cache_dir(), "quack")


def check_matrix(
    name: str, tensor: torch.Tensor, expected_ndim: int | None = None
) -> None:
    """Require a CUDA matrix operand with optional generated-op rank checking."""
    if expected_ndim is not None and tensor.ndim != expected_ndim:
        raise RuntimeError(
            f"FlexGEMM expected {expected_ndim}-D {name}, got {tensor.ndim}-D"
        )
    if tensor.ndim not in (2, 3):
        raise NotImplementedError(f"FlexGEMM currently supports only 2-D or 3-D {name}")
    if not tensor.is_cuda:
        raise RuntimeError(f"FlexGEMM requires CUDA {name}")


def check_same_device(a: torch.Tensor, b: torch.Tensor, *rest: torch.Tensor) -> None:
    """Require all runtime tensors to live on the same CUDA device."""
    device = a.device
    if b.device != device or any(tensor.device != device for tensor in rest):
        raise RuntimeError("FlexGEMM inputs must be on the same device")


def check_broadcast_shape(
    name: str, shape: torch.Size, expected_shape: tuple[int, ...]
) -> None:
    """Require a tensor shape to broadcast exactly to the GEMM output shape."""
    if not is_expandable_to(tuple(shape), expected_shape):
        raise RuntimeError(
            f"{name} shape must broadcast to {expected_shape}, got {tuple(shape)}"
        )


def check_matrix_major_layout(name: str, tensor: torch.Tensor) -> None:
    """Require row-major or column-major dense matrix strides."""
    if tensor.stride(-1) != 1 and tensor.stride(-2) != 1:
        raise NotImplementedError(
            f"FlexGEMM requires {name} to be row- or column-major"
        )


def check_matrix_row_major_layout(name: str, tensor: torch.Tensor) -> None:
    """Require last-dim-contiguous matrix strides for QuACK-written aux storage."""
    if tensor.stride(-1) != 1:
        raise NotImplementedError(f"FlexGEMM requires {name} to be row-major")


def check_epilogue_arg_kinds(epilogue_arg_kinds: tuple[str, ...]) -> None:
    """Require each epilogue arg kind to be row, col, tile, or scalar."""
    for kind in epilogue_arg_kinds:
        if kind not in ("tile", "row", "col", "scalar"):
            raise NotImplementedError(
                f"FlexGEMM supports only tile/row/col/scalar args, got {epilogue_arg_kinds}"
            )


def infer_epilogue_arg_kind(a: torch.Tensor, b: torch.Tensor, arg: torch.Tensor) -> str:
    """Infer a captured epilogue tensor's broadcast kind from its shape."""
    m, n = a.shape[-2], b.shape[-1]
    if tuple(arg.shape) == (1, 1):
        return "scalar"
    if tuple(arg.shape) == (m, n):
        return "tile"
    if tuple(arg.shape) == (1, n):
        return "row"
    if tuple(arg.shape) == (m, 1):
        return "col"
    raise NotImplementedError(
        "FlexGEMM captured tensor args must match the GEMM output "
        "shape or broadcast as [1, N] / [M, 1] / [1, 1]"
    )


def validate_epilogue_arg_shape(
    a: torch.Tensor,
    b: torch.Tensor,
    arg: torch.Tensor,
    kind: str,
) -> None:
    """Require a captured epilogue tensor shape to match its declared kind."""
    m, n = a.shape[-2], b.shape[-1]
    expected_shapes = {
        "tile": (m, n),
        "row": (1, n),
        "col": (m, 1),
        "scalar": (1, 1),
    }
    if tuple(arg.shape) != expected_shapes[kind]:
        raise RuntimeError(
            f"{kind} epilogue arg shape must be {expected_shapes[kind]}, "
            f"got {tuple(arg.shape)}"
        )


def resolve_epilogue_arg_kinds(
    a: torch.Tensor,
    b: torch.Tensor,
    epilogue_args: tuple[torch.Tensor, ...],
    epilogue_arg_kinds: tuple[str, ...],
) -> tuple[str, ...]:
    """Validate declared epilogue arg kinds or infer them from tensor shapes."""
    if epilogue_arg_kinds and len(epilogue_arg_kinds) != len(epilogue_args):
        raise RuntimeError("epilogue_arg_kinds must match epilogue_args length")
    check_epilogue_arg_kinds(epilogue_arg_kinds)
    if not epilogue_arg_kinds:
        return tuple(infer_epilogue_arg_kind(a, b, arg) for arg in epilogue_args)
    for arg, kind in zip(epilogue_args, epilogue_arg_kinds):
        validate_epilogue_arg_shape(a, b, arg, kind)
    return epilogue_arg_kinds


# NOTE [Boolean epilogue tensor storage]
# PyTorch bool tensors are byte-addressed, but CuTeDSL models cutlass.Boolean as
# a 1-bit logical type. Keep the manual uint8 storage view
# see https://github.com/NVIDIA/cutlass/issues/3348 for details
def quack_epilogue_arg(arg: torch.Tensor) -> torch.Tensor:
    """Adapt logical epilogue tensors to QuACK's physical tensor ABI."""
    return arg.view(torch.uint8) if arg.dtype is torch.bool else arg


def split_epilogue_args(
    epilogue_args: tuple[torch.Tensor, ...],
    epilogue_arg_kinds: tuple[str, ...],
) -> tuple[
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, ...],
]:
    """Group epilogue tensors into QuACK row, col, tile, and scalar argument lists."""
    row_args = []
    col_args = []
    tile_args = []
    scalar_args = []
    for arg, kind in zip(epilogue_args, epilogue_arg_kinds):
        arg = quack_epilogue_arg(arg)
        match kind:
            case "row":
                row_args.append(arg)
            case "col":
                col_args.append(arg.squeeze(-1).unsqueeze(0))
            case "tile":
                tile_args.append(arg.unsqueeze(0))
            case "scalar":
                scalar_args.append(arg.reshape(1))
    return tuple(row_args), tuple(col_args), tuple(tile_args), tuple(scalar_args)


def normalize_c(
    C: torch.Tensor | None, expected_shape: tuple[int, ...], beta: float
) -> torch.Tensor | None:
    """Return the effective C tensor that QuACK should read for alpha/beta GEMMs."""
    if C is None:
        return None
    check_broadcast_shape("C", C.shape, expected_shape)
    if beta == 0:
        return None
    broadcast_C = torch.broadcast_to(C, expected_shape)
    check_matrix("C", broadcast_C)
    check_matrix_major_layout("C", broadcast_C)
    return broadcast_C


@dataclasses.dataclass(frozen=True)
class FlexGemmRuntimeLocalReducePlan:
    """Runtime plan for one local reduction and its output/feed-main consumers."""

    geometry: FlexGemmLocalReduceGeometry
    out: torch.Tensor | None = None
    callbacks: FlexGemmLocalReduceCallbacks | None = None
    feeds_main: bool = False

    def __post_init__(self) -> None:
        """Reject plans without the output/callback state required by their consumer."""
        if self.out is None and not self.feeds_main:
            raise RuntimeError(LOCAL_REDUCE_RUNTIME_OUT_ERROR)
        if self.feeds_main:
            if self.callbacks is None:
                raise RuntimeError(LOCAL_REDUCE_CALLBACKS_REQUIRED_ERROR)
            validate_local_reduce_feed_main_capability(self.axis, self.group)
        elif self.geometry.needs_physical_callbacks and self.callbacks is None:
            raise RuntimeError(LOCAL_REDUCE_CALLBACKS_REQUIRED_ERROR)

    @property
    def group(self) -> int:
        return self.geometry.group

    @property
    def axis(self) -> int:
        return self.geometry.axis


@dataclasses.dataclass(frozen=True)
class FlexGemmRuntimeOutputPlan:
    """Bundle all user-visible output consumers into one generated ABI value."""

    aux_outs: tuple[torch.Tensor, ...] = ()
    local_reduce: FlexGemmRuntimeLocalReducePlan | None = None
    output_contraction: FlexGemmOutputContraction | None = None

    def __post_init__(self) -> None:
        if self.output_contraction is not None and (
            self.aux_outs or self.local_reduce is not None
        ):
            raise NotImplementedError(FLEX_GEMM_OUTPUT_CONTRACTION_COMPOSITION_ERROR)

    def output_shape(self, physical_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Return the logical shape after applying the output contraction."""
        if self.output_contraction is None:
            return physical_shape
        if physical_shape[-1] % self.output_contraction.group != 0:
            raise RuntimeError(FLEX_GEMM_OUTPUT_CONTRACTION_SHAPE_ERROR)
        return (
            *physical_shape[:-1],
            physical_shape[-1] // self.output_contraction.group,
        )


def validate_runtime_local_reduce(
    plan: FlexGemmRuntimeLocalReducePlan | None,
    a: torch.Tensor,
    expected_shape: tuple[int, ...],
    effective_C: torch.Tensor | None,
    alpha: float,
    beta: float,
) -> None:
    """Validate local-reduce runtime tensor shapes and unsupported consumers."""
    if plan is None:
        return
    validate_local_reduce_runtime_dense_mm(a.ndim)
    validate_local_reduce_selected_dim_divisible(expected_shape, plan.group, plan.axis)
    validate_local_reduce_no_c_alpha_beta(effective_C, alpha, beta)
    local_reduce_out = plan.out
    if local_reduce_out is None:
        return
    check_matrix("local_reduce_out", local_reduce_out)
    check_matrix_row_major_layout("local_reduce_out", local_reduce_out)
    expected_local_reduce_shape = local_reduce_compressed_shape(
        expected_shape, plan.group, plan.axis
    )
    validate_local_reduce_out_shape(local_reduce_out.shape, expected_local_reduce_shape)


def local_reduce_callback_key(callback: Any, fallback_key: str) -> str:
    """Use a generated callback cache key when present, otherwise its caller key."""
    cache_key = getattr(callback, "__cache_key__", None)
    if cache_key is None:
        return fallback_key
    key = cache_key() if callable(cache_key) else cache_key
    if not isinstance(key, str):
        raise RuntimeError("local-reduce callback __cache_key__ must return a string")
    return key


def register_runtime_local_reduce_callbacks(
    local_reduce: FlexGemmRuntimeLocalReducePlan | None,
    epilogue_key: str,
) -> tuple[str | None, str | None]:
    """Register generated physical callbacks and return QuACK registry keys."""
    if local_reduce is None or local_reduce.callbacks is None:
        return None, None
    callbacks = local_reduce.callbacks
    local_reduce_combine_key = local_reduce_callback_key(
        callbacks.combine_fn, f"{epilogue_key}{LOCAL_REDUCE_COMBINE_KEY_SUFFIX}"
    )
    local_reduce_finalize_key = local_reduce_callback_key(
        callbacks.finalize_fn, f"{epilogue_key}{LOCAL_REDUCE_FINALIZE_KEY_SUFFIX}"
    )
    from torch._vendor.quack.gemm_act import register_local_reduce_fns

    register_local_reduce_fns(
        local_reduce_combine_key,
        callbacks.combine_fn,
        local_reduce_finalize_key,
        callbacks.finalize_fn,
    )
    return local_reduce_combine_key, local_reduce_finalize_key


def local_reduce_gemm_act_kwargs(
    local_reduce: FlexGemmRuntimeLocalReducePlan | None,
    local_reduce_out: torch.Tensor | None,
    callback_keys: tuple[str | None, str | None],
) -> dict[str, Any]:
    """Map a concrete runtime plan onto QuACK's public local-reduce kwargs."""
    if local_reduce is None:
        return {}
    local_reduce_combine_key, local_reduce_finalize_key = callback_keys
    return {
        "tensor_epilogue_returns_local_reduce": local_reduce_out is not None,
        "local_reduce_feeds_main": local_reduce.feeds_main,
        "local_reduce_out": local_reduce_out,
        "local_reduce_group": local_reduce.group,
        "local_reduce_axis": local_reduce.axis,
        "local_reduce_combine_key": local_reduce_combine_key,
        "local_reduce_finalize_key": local_reduce_finalize_key,
    }


def dispatch_gemm_act(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor | None,
    out: torch.Tensor,
    output_plan: FlexGemmRuntimeOutputPlan,
    local_reduce_callback_keys: tuple[str | None, str | None],
    epilogue_key: str,
    epilogue_arg_kinds: tuple[str, ...],
    row_args: tuple[torch.Tensor, ...],
    col_args: tuple[torch.Tensor, ...],
    tile_args: tuple[torch.Tensor, ...],
    scalar_args: tuple[torch.Tensor, ...],
    alpha: float,
    beta: float,
    config,
    device_capacity_override: tuple[int, int] | None = None,
) -> None:
    """Dispatch one dense FlexGEMM call to the vendored QuACK GEMM kernel.

    ``config.swap_ab`` dispatches the transposed problem (only the tile schedule
    changes, not numerics): it swaps the A/B operands, writes through transposed
    ``out``/``C``/``aux_outs`` views, and swaps the row/col broadcast roles of
    captured epilogue tensors so each still aligns with the transposed accumulator.
    Tuple epilogues route the main result through QuACK ``D`` and aux outputs through
    ``PostAct``/``mAuxOut``. Output contractions instead leave ``D`` unused and use
    ``PostAct``/``mAuxOut`` as the contracted logical main store.
    """
    from torch._vendor.quack.gemm_act import gemm_act as gemm_act_dispatch

    # QuACK consumes A as (l, m, k) and B as (l, n, k); b is (k, n) so b.mT is (n, k).
    quack_a, quack_b = a, b.mT
    output_contraction = output_plan.output_contraction
    aux_outs = output_plan.aux_outs
    local_reduce = output_plan.local_reduce
    quack_out, quack_aux_outs, quack_local_reduce_out, quack_c = (
        out,
        aux_outs,
        None if local_reduce is None else local_reduce.out,
        C,
    )
    if config.swap_ab:
        if output_contraction is not None:
            raise NotImplementedError(
                "FlexGEMM output contractions do not support swap_ab configs yet"
            )
        quack_a, quack_b = quack_b, quack_a
        quack_out = out.mT
        if local_reduce is not None:
            raise NotImplementedError(LOCAL_REDUCE_SWAP_AB_ERROR)
        quack_aux_outs = tuple(aux_out.mT for aux_out in aux_outs)
        quack_c = None if C is None else C.mT
        row_args, col_args = col_args, row_args
        tile_args = tuple(tile.mT for tile in tile_args)
        epilogue_arg_kinds = tuple(
            _SWAPPED_ARG_KIND[kind] for kind in epilogue_arg_kinds
        )

    # QuACK expects a leading batch dim; 2-D (non-batched) operands get one here.
    quack_a = quack_a.unsqueeze(0) if quack_a.ndim == 2 else quack_a
    quack_b = quack_b.unsqueeze(0) if quack_b.ndim == 2 else quack_b
    quack_out = quack_out.unsqueeze(0) if quack_out.ndim == 2 else quack_out
    quack_aux_outs = tuple(quack_epilogue_arg(aux_out) for aux_out in quack_aux_outs)
    quack_aux_outs = tuple(
        aux_out.unsqueeze(0) if aux_out.ndim == 2 else aux_out
        for aux_out in quack_aux_outs
    )
    if quack_local_reduce_out is not None and quack_local_reduce_out.ndim == 2:
        quack_local_reduce_out = quack_local_reduce_out.unsqueeze(0)
    if quack_c is not None and quack_c.ndim == 2:
        quack_c = quack_c.unsqueeze(0)

    returns_aux = bool(quack_aux_outs)
    quack_d = quack_out if returns_aux else None
    quack_postact = quack_aux_outs if returns_aux else quack_out
    local_reduce_kwargs = local_reduce_gemm_act_kwargs(
        local_reduce, quack_local_reduce_out, local_reduce_callback_keys
    )

    gemm_act_dispatch(
        quack_a,
        quack_b,
        quack_d,
        quack_c,
        quack_postact,
        None,  # tile_count_semaphore
        None,  # activation
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        tile_K=config.tile_k,
        pingpong=config.pingpong,
        persistent=True,
        is_dynamic_persistent=config.is_dynamic_persistent,
        tensor_epilogue_key=epilogue_key,
        tensor_epilogue_returns_aux=returns_aux,
        tensor_epilogue_arg_kinds=epilogue_arg_kinds,
        **local_reduce_kwargs,
        tensor_epilogue_rowvec_biases=row_args,
        tensor_epilogue_colvec_biases=col_args,
        tensor_epilogue_tile_biases=tile_args,
        tensor_epilogue_scalar_biases=scalar_args,
        main_output_transform_group=(
            None if output_contraction is None else output_contraction.group
        ),
        concat_layout=(
            () if output_contraction is None else output_contraction.concat_layout
        ),
        alpha=alpha,
        beta=beta,
        use_tma_gather=config.use_tma_gather,
        device_capacity_override=device_capacity_override,
    )


def gemm_epilogue(
    a: torch.Tensor,
    b: torch.Tensor,
    epilogue_fn,
    epilogue_key: str,
    *,
    C: torch.Tensor | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    out_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
    output_plan: FlexGemmRuntimeOutputPlan = FlexGemmRuntimeOutputPlan(),
    epilogue_args: tuple[torch.Tensor, ...] = (),
    epilogue_arg_kinds: tuple[str, ...] = (),
    config_key: GemmConfigKey | None = None,
    config_is_lowering_validated: bool = False,
    expected_ndim: int | None = None,
    device_capacity_override: tuple[int, int] | None = None,
    stream: int | None = None,
    quack_cache_dir: str | None = None,
) -> torch.Tensor:
    """Run a dense GEMM through QuACK with a CuTeDSL epilogue.

    Args:
        a: Left operand with shape ``[M, K]`` or ``[B, M, K]``.
        b: Right operand with shape ``[K, N]`` or ``[B, K, N]``.
        epilogue_fn: CuTeDSL epilogue callable applied to the accumulator tile.
        epilogue_key: Stable cache key component for the epilogue.
        C: Optional bias/addend broadcastable to the output shape.
        alpha: Scale applied to the GEMM accumulator.
        beta: Scale applied to ``C`` when ``C`` is present.
        out_dtype: Optional output dtype. Defaults to ``a.dtype``.
        out: Optional preallocated output tensor with shape ``[M, N]`` or ``[B, M, N]``.
        output_plan: Structural plan for auxiliary and contracted outputs.
        epilogue_args: Optional tensor args captured by the epilogue.
        epilogue_arg_kinds: Explicit ``tile``, ``row``, ``col``, or ``scalar`` kind per arg.
        config_key: Optional explicit QuACK config key selected by Inductor autotune.
        config_is_lowering_validated: Whether generated lowering already checked the
            config against the output-contraction contract.
        expected_ndim: Optional generated-op rank contract for A and B operands.
        device_capacity_override: Parent-computed capability for compile-only workers.
        stream: Optional raw CUDA stream pointer supplied by the generated wrapper.
        quack_cache_dir: Optional scoped cache root for Inductor-generated QuACK work.

    Returns:
        Tensor with shape ``[M, N]`` or ``[B, M, N]``.
    """
    check_matrix("a", a, expected_ndim)
    check_matrix("b", b, expected_ndim)
    check_matrix_major_layout("a", a)
    check_matrix_major_layout("b", b)
    if a.ndim != b.ndim:
        raise RuntimeError("FlexGEMM inputs must both be 2-D or both be 3-D")
    if a.ndim == 3 and a.shape[0] != b.shape[0]:
        raise RuntimeError("FlexGEMM batched inputs must have the same batch size")
    if a.shape[-1] != b.shape[-2]:
        raise RuntimeError(
            f"mat1 and mat2 shapes cannot be multiplied ({a.shape} and {b.shape})"
        )
    physical_output_shape = (*a.shape[:-2], a.shape[-2], b.shape[-1])
    aux_outs = output_plan.aux_outs
    local_reduce = output_plan.local_reduce
    output_contraction = output_plan.output_contraction
    logical_output_shape = output_plan.output_shape(physical_output_shape)
    if output_contraction is not None:
        device_capacity = (
            torch.cuda.get_device_capability(a.device)
            if device_capacity_override is None
            else device_capacity_override
        )
        output_contraction.validate_quack(device_capacity[0])
        if output_contraction.chunked and b.stride(-1) == 1:
            raise NotImplementedError(FLEX_GEMM_CHUNKED_CONTIGUOUS_B_ERROR)
        if a.ndim != 2 or C is not None or alpha != 1.0 or beta != 1.0:
            raise NotImplementedError(FLEX_GEMM_OUTPUT_CONTRACTION_COMPOSITION_ERROR)
    expected_dtype = out_dtype
    if expected_dtype is None:
        expected_dtype = out.dtype if out is not None else a.dtype
    effective_C = normalize_c(C, physical_output_shape, beta)
    if out is not None:
        check_matrix("out", out)
        check_matrix_major_layout("out", out)
        if tuple(out.shape) != logical_output_shape:
            raise RuntimeError(
                f"out shape must be {logical_output_shape}, got {tuple(out.shape)}"
            )
        if out.dtype != expected_dtype:
            raise RuntimeError(f"out dtype must be {expected_dtype}, got {out.dtype}")
    if aux_outs:
        if a.ndim != 2:
            raise NotImplementedError(
                "FlexGEMM generic aux tuple epilogues currently support only 2-D aten.mm"
            )
        if effective_C is not None or alpha != 1.0 or beta != 1.0:
            raise NotImplementedError(
                "FlexGEMM generic aux tuple epilogues cannot be combined with C/alpha/beta yet"
            )
        for index, aux_out in enumerate(aux_outs):
            check_matrix(f"aux_outs[{index}]", aux_out)
            check_matrix_major_layout(f"aux_outs[{index}]", aux_out)
            if tuple(aux_out.shape) != physical_output_shape:
                raise RuntimeError(
                    f"aux_outs[{index}] shape must be {physical_output_shape}, "
                    f"got {tuple(aux_out.shape)}"
                )
    validate_runtime_local_reduce(
        local_reduce,
        a,
        physical_output_shape,
        effective_C,
        alpha,
        beta,
    )
    if a.ndim == 3 and epilogue_args:
        raise NotImplementedError("FlexGEMM batched args are not supported yet")
    if epilogue_args and effective_C is not None:
        # TODO: Route this through the flex frontend so validated A/B/C metadata
        # can be reused here.
        raise NotImplementedError("FlexGEMM args cannot be combined with C yet")
    if epilogue_args and (alpha != 1.0 or beta != 1.0):
        raise NotImplementedError(
            "FlexGEMM args cannot be combined with non-default alpha/beta yet"
        )
    tensors = (
        C,
        out,
        *aux_outs,
        None if local_reduce is None else local_reduce.out,
        *epilogue_args,
    )
    check_same_device(a, b, *(tensor for tensor in tensors if tensor is not None))
    inferred_arg_kinds = resolve_epilogue_arg_kinds(
        a, b, epilogue_args, epilogue_arg_kinds
    )
    if output_contraction is not None and any(
        not output_contraction_capture_supported(kind, arg.dtype is torch.bool)
        for arg, kind in zip(epilogue_args, inferred_arg_kinds, strict=True)
    ):
        raise NotImplementedError(FLEX_GEMM_OUTPUT_CONTRACTION_CAPTURE_ERROR)
    for index, arg in enumerate(epilogue_args):
        check_matrix_major_layout(f"epilogue_args[{index}]", arg)
    row_args, col_args, tile_args, scalar_args = split_epilogue_args(
        epilogue_args, inferred_arg_kinds
    )

    from torch._vendor.quack.gemm_act import register_tensor_epilogue_fn

    register_tensor_epilogue_fn(epilogue_key, epilogue_fn)
    local_reduce_callback_keys = register_runtime_local_reduce_callbacks(
        local_reduce,
        epilogue_key,
    )
    out = (
        torch.empty(logical_output_shape, device=a.device, dtype=expected_dtype)
        if out is None
        else out
    )
    from torch._inductor.heuristics.template.flex_gemm import (
        candidate_gemm_configs_for_device,
        gemm_config_from_key,
    )
    from torch._vendor.quack.cache import cache_dir_override

    config = gemm_config_from_key(config_key) if config_key is not None else None
    if config is None or (
        output_contraction is not None and not config_is_lowering_validated
    ):
        candidates = candidate_gemm_configs_for_device(a.device)
        if config is not None and config not in candidates:
            raise NotImplementedError(
                "FlexGEMM explicit QUACK config is not supported on this device"
            )
        if output_contraction is None:
            config = candidates[0]
        elif config is None:
            config = next(
                (
                    candidate
                    for candidate in candidates
                    if output_contraction_config_supported(
                        candidate, physical_output_shape[-1]
                    )
                ),
                None,
            )
        elif not output_contraction_config_supported(config, physical_output_shape[-1]):
            config = None
        if config is None:
            raise NotImplementedError(
                "FlexGEMM QUACK config is incompatible with output contraction"
            )

    stream_context = (
        torch.cuda.stream(torch.cuda.ExternalStream(stream, device=a.device))
        if stream is not None
        else contextlib.nullcontext()
    )
    with cache_dir_override(quack_cache_dir), stream_context:
        dispatch_gemm_act(
            a,
            b,
            effective_C,
            out,
            output_plan,
            local_reduce_callback_keys,
            epilogue_key,
            inferred_arg_kinds,
            row_args,
            col_args,
            tile_args,
            scalar_args,
            alpha,
            beta,
            config=config,
            device_capacity_override=device_capacity_override,
        )
    return out
<|endoftext|># mypy: allow-untyped-defs
"""Shared FlexGEMM local-reduce geometry, constants, and validation helpers."""

import dataclasses
from collections.abc import Sequence
from typing import Any, Final

from torch._inductor.kernel.gemm_epilogue import GemmReductionGeometry
from torch._inductor.kernel.gemm_epilogue_utils import (
    statically_known,
    statically_known_shape_equal,
)


LOCAL_REDUCE_FEED_MAIN_ARG_NAME: Final = "local_reduce0"
LOCAL_REDUCE_COMBINE_FN_SUFFIX: Final = "_local_reduce_combine_fn"
LOCAL_REDUCE_FINALIZE_FN_SUFFIX: Final = "_local_reduce_finalize_fn"
LOCAL_REDUCE_COMBINE_KEY_SUFFIX: Final = ":local_reduce_combine"
LOCAL_REDUCE_FINALIZE_KEY_SUFFIX: Final = ":local_reduce_finalize"


# The physical feed-main path currently reduces only within one lane-layout M
# group; cross-warp M stitching needs the two-phase/replay path used by
# compressed aux reductions. Axis-1 feeds whose groups fit in one TensorSSA
# fragment lower as plain generated TensorSSA without a feed plan.
LOCAL_REDUCE_FRAGMENT_WIDTH = 32
LOCAL_REDUCE_FEED_MAIN_AXIS_ERROR = (
    "FlexGEMM local-reduce feed-main currently supports only axis 0"
)
LOCAL_REDUCE_FEED_MAIN_SAME_WARP_ERROR = (
    "FlexGEMM local-reduce feed-main currently supports only same-warp axis-0 "
    f"groups <= {LOCAL_REDUCE_FRAGMENT_WIDTH}"
)
LOCAL_REDUCE_FEED_MAIN_AXIS1_FRAGMENT_ERROR = (
    "FlexGEMM local-reduce feed-main for axis-1 groups larger than one "
    "TensorSSA fragment is not supported yet"
)
LOCAL_REDUCE_DIVISIBLE_SHAPE_ERROR = (
    "local_reduce_group must divide the selected FlexGEMM output dimension"
)
LOCAL_REDUCE_GROUP_POSITIVE_ERROR = "local_reduce_group must be positive"
LOCAL_REDUCE_AXIS_ERROR = "local_reduce_axis must be 0 or 1"
LOCAL_REDUCE_TENSORSSA_GROUP_SIZE_ERROR = (
    "FlexGEMM local reductions require group size greater than 1"
)
LOCAL_REDUCE_TENSORSSA_FRAGMENT_MULTIPLE_ERROR = (
    "FlexGEMM local reductions larger than TensorSSA fragment width 32 "
    "require group size to be a multiple of 32"
)
LOCAL_REDUCE_TENSORSSA_FRAGMENT_DIVISIBLE_ERROR = (
    "FlexGEMM local reductions require group size to divide TensorSSA fragment width 32"
)
LOCAL_REDUCE_PARTIAL_OUTPUT_CONTRACT_ERROR = (
    "unsupported FlexGEMM epilogue partial-output contract: FlexGEMM does not "
    "support this local-reduce output contract yet. Please file an issue with "
    "the FlexGEMM epilogue expression."
)
LOCAL_REDUCE_MIXED_GROUPED_LAYOUT_ERROR = (
    "FlexGEMM local reductions do not support mixing grouped TensorSSA "
    "values with different grouped layouts"
)
LOCAL_REDUCE_DENSE_MM_SCOPE_ERROR = (
    "FlexGEMM local reductions currently support only aten.mm"
)
LOCAL_REDUCE_C_ALPHA_BETA_ERROR = (
    "FlexGEMM local reductions cannot be combined with C/alpha/beta yet"
)
LOCAL_REDUCE_SWAP_AB_ERROR = (
    "FlexGEMM local reductions do not support swap_ab configs yet"
)
LOCAL_REDUCE_AUX_TENSORSSA_ERROR = (
    "FlexGEMM local-reduce aux output must be produced by a grouped TensorSSA reduction"
)
LOCAL_REDUCE_AUX_OUTPUT_CONTRACT_ERROR = (
    "FlexGEMM does not support this aux output shape yet. Please file an issue "
    "with the FlexGEMM epilogue expression."
)
LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR = (
    "FlexGEMM local-reduce broadcast values support one generated physical reduction"
)
LOCAL_REDUCE_SINGLE_PHYSICAL_FINALIZE_ERROR = (
    "FlexGEMM physical finalize expressions support a single physical local reduction"
)
LOCAL_REDUCE_POST_POINTWISE_FINALIZE_ERROR = (
    "FlexGEMM physical local reductions do not support post-reduction "
    "pointwise transforms yet. Please file an issue with the FlexGEMM epilogue "
    "expression."
)
LOCAL_REDUCE_FINALIZE_SCALAR_ONLY_ERROR = (
    "FlexGEMM physical local reductions require finalize expressions to depend "
    "only on the reduced value and scalar constants"
)
LOCAL_REDUCE_SOURCE_EXPRESSION_ERROR = (
    "FlexGEMM physical local-reduce feed-main source expressions require "
    "two-phase local-reduce source lowering"
)
LOCAL_REDUCE_CONFIG_ERROR = (
    "FlexGEMM local-reduce aux outputs require a non-swap_ab config whose CTA "
    "tile axis is divisible by group"
)
LOCAL_REDUCE_EXPLICIT_DTYPE_ERROR = (
    "FlexGEMM does not support explicit reduction dtype yet"
)
LOCAL_REDUCE_INNERMOST_GROUPED_DIM_ERROR = (
    "FlexGEMM local reductions currently support only reductions over the "
    "innermost grouped dimension"
)
LOCAL_REDUCE_GROUPED_RESHAPE_ERROR = (
    "FlexGEMM local-reduce grouped reshape must split exactly one GEMM output dimension"
)
LOCAL_REDUCE_MIXED_MATCH_ERROR = (
    "FlexGEMM local reductions do not support mixing different grouped layouts"
)
LOCAL_REDUCE_FEED_MAIN_MIXED_MATCH_ERROR = (
    "FlexGEMM local-reduce broadcast values must share one grouped layout"
)
FLEX_GEMM_OUTPUT_PLAN_NODE_ERROR = "FlexGEMM output plans require tensor output nodes"
FLEX_GEMM_OUTPUT_TENSOR_ERROR = "FlexGEMM expects tensor outputs"
LOCAL_REDUCE_MATCH_NODE_ERROR = "local-reduce matches require tensor nodes"
LOCAL_REDUCE_OUTPUT_PLAN_NODE_ERROR = "local-reduce output plans require tensor nodes"
LOCAL_REDUCE_RUNTIME_OUT_ERROR = "compressed local reductions require local_reduce_out"
LOCAL_REDUCE_RUNTIME_DENSE_MM_ERROR = (
    "FlexGEMM local reductions currently support only 2-D aten.mm"
)
LOCAL_REDUCE_OUT_SHAPE_ERROR = "local_reduce_out shape must be {expected}, got {actual}"
LOCAL_REDUCE_CALLBACKS_REQUIRED_ERROR = (
    "physical local reductions require generated local-reduce callbacks"
)
FLEX_GEMM_OUTPUT_CONTRACTION_COMPOSITION_ERROR = (
    "FlexGEMM output contractions do not compose with aux outputs, local "
    "reductions, C, alpha/beta, or batched GEMMs yet"
)
FLEX_GEMM_OUTPUT_CONTRACTION_CAPTURE_ERROR = (
    "FlexGEMM output contractions currently support only numeric [1, 1] and "
    "[M, 1] captured tensors"
)
FLEX_GEMM_CHUNKED_OUTPUT_CONTRACTION_REDUCE_ERROR = (
    "FlexGEMM concat-layout output contractions do not compose with grouped "
    "reductions because concat layout permutes accumulator columns"
)
FLEX_GEMM_CHUNKED_CONTIGUOUS_B_ERROR = (
    "FlexGEMM concat-layout output contractions require B's output dimension to "
    "be non-contiguous, as in linear weight.t()"
)
FLEX_GEMM_OUTPUT_CONTRACTION_SHAPE_ERROR = (
    "unsupported FlexGEMM epilogue: contracted output shape must equal the "
    "physical GEMM output shape with N divided by the contraction group"
)
FLEX_GEMM_MAIN_OUTPUT_SHAPE_ERROR = (
    "unsupported FlexGEMM epilogue: main output shape must equal the physical "
    "GEMM output shape"
)


def statically_known_multiple(value: Any, divisor: int) -> bool:
    """Return whether a symbolic shape value is known divisible without guards."""
    return statically_known(value % divisor == 0)


def is_flex_gemm_partial_reduction_shape(
    aux_size: Sequence[Any], output_size: Sequence[Any]
) -> bool:
    """Recognize aux shapes that imply a final PyTorch reduction, not local reduce.

    FlexGEMM's generic aux-output path supports one same-shape aux tensor beside
    the main output. Reduced shapes such as ``[]``, ``[M]``, ``[N]``, ``[M, 1]``,
    ``[1, N]``, or exact 2-D divisors of ``[M, N]`` mean the epilogue tried to
    return a final PyTorch reduction/block reduction. Those are different from
    QuACK local-reduce aux outputs, which are only accepted after the epilogue
    exposes an explicit grouped view such as ``acc.view(M, -1, group).sum(-1)``.
    """
    if len(output_size) != 2:
        return False
    aux_shape = tuple(aux_size)
    m, n = output_size
    if any(
        statically_known_shape_equal(aux_shape, candidate)
        for candidate in ((), (m,), (n,), (1, 1), (m, 1), (1, n))
    ):
        return True
    if len(aux_shape) != 2:
        return False
    aux_m, aux_n = aux_shape
    return (
        statically_known(aux_m > 0)
        and statically_known(aux_n > 0)
        and statically_known(aux_m <= m)
        and statically_known(aux_n <= n)
        and (statically_known(aux_m < m) or statically_known(aux_n < n))
        and statically_known_multiple(m, aux_m)
        and statically_known_multiple(n, aux_n)
    )


def local_reduce_unsupported_tensorssa_error(
    reduction: Any, *, value_only: bool = False
) -> NotImplementedError:
    """Explain why a grouped reduction is outside the current TensorSSA subset."""
    suffix = " value-only reduction" if value_only else ""
    return NotImplementedError(
        "FlexGEMM does not support this grouped local reduction yet: "
        f"{reduction} does not map to a CuTe TensorSSA{suffix}. Please file "
        "an issue with the FlexGEMM epilogue expression."
    )


def validate_local_reduce_group_axis(group: int, axis: int) -> None:
    """Keep local-reduce specs inside the GEMM tile's M/N grouping model."""
    if group <= 0:
        raise RuntimeError(LOCAL_REDUCE_GROUP_POSITIVE_ERROR)
    if axis not in (0, 1):
        raise RuntimeError(LOCAL_REDUCE_AXIS_ERROR)


def validate_local_reduce_selected_dim_divisible(
    shape: Sequence[Any], group: int, axis: int
) -> None:
    """Reject selected M/N dimensions known not to have an integral compressed shape."""
    validate_local_reduce_group_axis(group, axis)
    selected_dim = shape[axis - 2]
    if statically_known_multiple(selected_dim, group):
        return
    if statically_known(selected_dim % group != 0):
        raise RuntimeError(LOCAL_REDUCE_DIVISIBLE_SHAPE_ERROR)


def validate_local_reduce_tensorssa_group_size(axis: int, group: int) -> None:
    """Mirror the TensorSSA fragment tiling constraints used by QuACK.

    Groups within one fragment must divide the 32-lane TensorSSA width. Larger
    groups are handled as 32-lane TensorSSA partials plus physical combine, so
    they must be exact multiples of that fragment width.
    """
    if group <= 1:
        raise NotImplementedError(LOCAL_REDUCE_TENSORSSA_GROUP_SIZE_ERROR)
    validate_local_reduce_group_axis(group, axis)
    if group > LOCAL_REDUCE_FRAGMENT_WIDTH and group % LOCAL_REDUCE_FRAGMENT_WIDTH != 0:
        raise NotImplementedError(LOCAL_REDUCE_TENSORSSA_FRAGMENT_MULTIPLE_ERROR)
    if (
        group <= LOCAL_REDUCE_FRAGMENT_WIDTH
        and LOCAL_REDUCE_FRAGMENT_WIDTH % group != 0
    ):
        raise NotImplementedError(LOCAL_REDUCE_TENSORSSA_FRAGMENT_DIVISIBLE_ERROR)


def local_reduce_needs_physical_callbacks(axis: int, group: int) -> bool:
    """Return whether QuACK must merge TensorSSA partials outside the fragment path."""
    return axis == 0 or group > LOCAL_REDUCE_FRAGMENT_WIDTH


def validate_local_reduce_runtime_dense_mm(ndim: int) -> None:
    """Keep runtime wrappers on the only layout QuACK currently supports.

    Local-reduce group/axis semantics are defined relative to dense ``mm`` output
    dimensions. Batched or vectorized matmul layouts would need separate shape
    compression and epilogue argument mapping rules before the same ABI is valid.
    """
    if ndim != 2:
        raise NotImplementedError(LOCAL_REDUCE_RUNTIME_DENSE_MM_ERROR)


def validate_local_reduce_out_shape(
    actual_shape: Sequence[Any], expected_shape: Sequence[Any]
) -> None:
    """Ensure caller-provided aux storage matches the structural reduce plan.

    Runtime cannot reinterpret an arbitrary aux tensor as the compressed local-
    reduce domain: QuACK writes exactly the shape produced by dividing the chosen
    GEMM output dimension by the group size, so mismatches would corrupt memory
    or silently expose the wrong logical tensor.
    """
    actual = tuple(actual_shape)
    expected = tuple(expected_shape)
    if not statically_known_shape_equal(actual, expected):
        raise RuntimeError(
            LOCAL_REDUCE_OUT_SHAPE_ERROR.format(expected=expected, actual=actual)
        )


def validate_local_reduce_feed_main_capability(axis: int, group: int) -> None:
    """Limit feed-main reducers to the physical path QuACK can re-inject today.

    Feeding a reduction back into the main epilogue needs the physical row-lane
    combine result to be available as a scalar value for each output element.
    That is currently implemented only for same-warp M-axis groups.
    """
    if axis != 0:
        raise NotImplementedError(LOCAL_REDUCE_FEED_MAIN_AXIS_ERROR)
    if group > LOCAL_REDUCE_FRAGMENT_WIDTH:
        raise NotImplementedError(LOCAL_REDUCE_FEED_MAIN_SAME_WARP_ERROR)


def local_reduce_compressed_shape(
    shape: Sequence[Any], group: int, axis: int
) -> tuple[Any, ...]:
    """Compute the explicit aux shape that mirrors QuACK's grouped store."""
    validate_local_reduce_selected_dim_divisible(shape, group, axis)
    result = list(shape)
    result[axis - 2] //= group
    return tuple(result)


def validate_local_reduce_no_c_alpha_beta(
    effective_C: Any | None, alpha: float, beta: float
) -> None:
    """Reject C/alpha/beta composition until local-reduce ordering is explicit."""
    if effective_C is not None or alpha != 1.0 or beta != 1.0:
        raise NotImplementedError(LOCAL_REDUCE_C_ALPHA_BETA_ERROR)


def validate_flex_gemm_local_reduce_config(config: Any, group: int, axis: int) -> bool:
    """Return whether a QuACK config has a validated grouped-reduction layout.

    This matches ``GemmConfig`` fields against layout families covered by forced
    kernel tests; tile divisibility alone is not sufficient. Axis-1 groups within
    one 32-value epilogue fragment need no cross-fragment combine. Some SM100
    two-CTA layouts expose only 16 contiguous N values, reducing that local limit.

    Axis-0 groups and larger axis-1 groups use ``GroupedLocalReduce``'s physical
    callback path, which combines epilogue fragments inside one CTA and directly
    stores one value per ``(row, group)``. For two-CTA ``tile_m=128`` kernels, each
    CTA has a 64-row epilogue tile whose warps are split 2x2 across M and N. An
    axis-1 group spanning the full N tile therefore crosses N-warp ownership that
    the temporal fragment combine does not stitch; strict subgroups remain valid.

    Two-CTA ``tile_m=256`` kernels instead have a 128-row epilogue tile with a 4x1
    warp layout, so one N-warp partition owns each row's full N tile. Their axis-1
    temporal fragment combine supports a full-tile group without cross-CTA state.
    Axis-0 full groups still exceed the per-CTA M tile and remain unsupported.
    """
    match axis:
        case 0:
            tile = config.tile_m
        case 1:
            tile = config.tile_n
        case _:
            return False
    if group <= 0 or config.swap_ab:
        return False
    if config.tile_n % LOCAL_REDUCE_FRAGMENT_WIDTH != 0 or tile % group != 0:
        return False

    fragment_width = LOCAL_REDUCE_FRAGMENT_WIDTH
    has_half_n_fragment = (
        axis == 1
        and config.tile_m == 128
        and config.tile_n in (128, 160, 224)
        and config.cluster_m > 1
    )
    if has_half_n_fragment:
        fragment_width //= 2
    if group <= LOCAL_REDUCE_FRAGMENT_WIDTH:
        return fragment_width % group == 0 and group < tile

    if group % LOCAL_REDUCE_FRAGMENT_WIDTH != 0 or config.cluster_n != 1:
        return False

    is_single_cta_layout = config.tile_m == 128 and config.cluster_m == 1
    is_wide_m_two_cta_layout = config.tile_m == 256 and config.cluster_m == 2
    is_split_n_warp_two_cta_layout = (
        axis == 1
        and config.tile_m == 128
        and config.tile_n == 256
        and config.cluster_m == 2
    )
    if is_single_cta_layout:
        return True
    if is_wide_m_two_cta_layout:
        return axis == 1 or group < tile
    return is_split_n_warp_two_cta_layout and group < tile


def flex_gemm_local_reduce_candidate_groups(config: Any, axis: int) -> tuple[int, ...]:
    """Enumerate group sizes worth checking against the config capability gate."""
    match axis:
        case 0:
            tile = config.tile_m
        case 1:
            tile = config.tile_n
        case _:
            return ()
    return (2, 4, 8, 16, 32, *range(64, tile + 1, 32))


def max_flex_gemm_local_reduce_group_for_configs(
    configs: Sequence[Any], axis: int
) -> int | None:
    """Return the largest group accepted by the current local-reduce config gate."""
    candidates = [
        group
        for config in configs
        for group in flex_gemm_local_reduce_candidate_groups(config, axis)
        if validate_flex_gemm_local_reduce_config(config, group, axis)
    ]
    return max(candidates) if candidates else None


def flex_gemm_local_reduce_config_error(
    configs: Sequence[Any], group: int, axis: int
) -> str:
    """Explain the current config-filter frontier for local-reduce groups."""
    max_group = max_flex_gemm_local_reduce_group_for_configs(configs, axis)
    if max_group is None:
        return LOCAL_REDUCE_CONFIG_ERROR
    return (
        f"{LOCAL_REDUCE_CONFIG_ERROR}; requested group={group}, "
        f"max supported group={max_group} for axis={axis}"
    )


# NOTE [Non-shape-preserving FlexGEMM outputs]
# FlexGEMM normally returns one value per physical GEMM accumulator. Output
# contraction is the current exception: it exposes grouped physical N values to
# the ordinary FX epilogue, requires the main expression to consume the complete
# group, and stores one logical value per group. Interleaved groups use
# ``view(M, logical_N, group)``; chunked groups use
# ``view(M, group, logical_N)`` or ``split(logical_N, dim=-1)``. This covers
# SwiGLU-like pointwise combinations without claiming to support arbitrary slices,
# permutations, expansions, or M-axis contraction.
#
# Numeric ``[1, 1]`` and ``[M, 1]`` captures are N-invariant. QuACK loads their
# one value per epilogue thread and lets generated pointwise code broadcast it in
# either the physical or contracted layout. N-varying ``[1, N]`` and ``[M, N]``
# captures need the same concat-to-interleave mapping as B for chunked outputs.
@dataclasses.dataclass(frozen=True)
class FlexGemmOutputContraction:
    """Describe the contraction in NOTE [Non-shape-preserving FlexGEMM outputs].

    Attributes:
        group: Number of physical N values contracted into each logical output.
        chunked: Whether group values are contiguous N chunks rather than interleaved.
    """

    group: int
    chunked: bool = False

    def __post_init__(self) -> None:
        if self.group <= 0:
            raise ValueError("output-contraction group must be positive")

    @property
    def concat_layout(self) -> tuple[str, ...]:
        """Return the QuACK ABI tag that interleaves chunked B columns."""
        return ("B",) if self.chunked else ()

    def validate_quack(self, device_capacity: int) -> None:
        if device_capacity == 12:
            raise NotImplementedError(
                "FlexGEMM output contractions are not yet supported on SM120"
            )
        if device_capacity not in (10, 11):
            raise NotImplementedError(
                "FlexGEMM output contractions are currently validated only on "
                "SM100 and SM110"
            )
        if self.group == 2 or (
            self.group == 4 and not self.chunked and device_capacity == 10
        ):
            return
        raise NotImplementedError(
            "FlexGEMM output-contraction stores support group 2 on SM100 and "
            "SM110, plus interleaved group 4 on SM100"
        )


def output_contraction_capture_supported(kind: str, is_boolean: bool) -> bool:
    """Return whether output-contraction codegen can broadcast a captured tensor."""
    return kind in ("scalar", "col") and not is_boolean


def output_contraction_config_supported(config: Any, n: Any) -> bool:
    """Return whether a config has validated output-contraction store ownership.

    Keep the physical M/N orientation, one CTA per cluster along N, and require
    the physical N tile not to exceed the problem. Admit only M-cluster families
    whose row ownership has been validated: a single M CTA or the wide-M two-CTA
    layout. Multiple N tiles and partial final tiles are supported by the ordinary
    tile scheduler and store predicates.
    """
    supported_m_cluster = config.cluster_m == 1 or (
        config.tile_m == 256 and config.cluster_m == 2
    )
    return (
        not config.swap_ab
        and supported_m_cluster
        and config.cluster_n == 1
        and statically_known(config.tile_n <= n)
    )


FlexGemmLocalReduceGeometry = GemmReductionGeometry


@dataclasses.dataclass(frozen=True)
class FlexGemmLocalReduceCallbacks:
    """Carry generated physical combine/finalize functions."""

    combine_fn: Any
    finalize_fn: Any

    def __post_init__(self) -> None:
        """Keep physical reducers from existing without their generated code."""
        if self.combine_fn is None or self.finalize_fn is None:
            raise RuntimeError(LOCAL_REDUCE_CALLBACKS_REQUIRED_ERROR)
<|endoftext|># mypy: allow-untyped-defs
"""CPU-specific implementations for flex attention"""

import copy
import os
import sys
from typing import Any

import sympy

import torch
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.value_ranges import ValueRanges

from ...codegen.cpp_flex_attention_template import CppFlexAttentionTemplate
from ...ir import Buffer, ExternKernel, FixedLayout, TensorBox
from ...select_algorithm import autotune_select_algorithm
from .common import (
    build_subgraph_buffer,
    build_subgraph_module_buffer,
    contiguous_last_dim,
    create_placeholder,
    get_fwd_subgraph_outputs,
    infer_dense_strides,
    maybe_realize,
)


def check_cpu_supported():
    requires_avx2_on_cpu = (
        torch.cpu._is_avx2_supported() and os.getenv("ATEN_CPU_CAPABILITY") != "default"
    )
    supported = (
        requires_avx2_on_cpu
        and not torch.xpu.is_available()
        and sys.platform != "darwin"
    )
    return supported


def _realize_duplicate_buffer_inputs(inputs):
    names = [inp.get_name() for inp in inputs]
    duplicate_names = OrderedSet(name for name in names if names.count(name) > 1)
    if not duplicate_names:
        return inputs

    # The C++ template names pointer arguments by storage buffer name.
    # ReinterpretView inputs that share storage (for example, q/k from
    # qk.chunk()) need distinct realized buffers so their view offsets are
    # represented correctly at the template boundary.
    return [
        ExternKernel.copy_input(inp) if inp.get_name() in duplicate_names else inp
        for inp in inputs
    ]


def lower_cpu(
    query,
    key,
    value,
    subgraph,
    block_mask,
    scale,
    kernel_options,
    score_mod_other_buffers,
    mask_mod_other_buffers,
):
    """CPP based template for flex attention for x86 CPUs"""
    (
        _,  # q_length
        _,  # kv_length
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        q_num_blocks,
        q_indices,
        full_q_num_blocks,
        full_q_indices,
        _,  # dq_write_order
        _,  # dq_write_order_full
        _,  # dq_kv_order
        _,  # dq_kv_order_spt
        SPARSE_Q_BLOCK_SIZE,
        SPARSE_KV_BLOCK_SIZE,
        mask_graph,
    ) = block_mask

    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError(
            f"Mixed query, key, and value dtype is not supported on this platform, "
            f"got query.dtype: {query.dtype}, key.dtype: {key.dtype}, "
            f"and value.dtype: {value.dtype}."
        )

    if kernel_options["OUTPUT_LOGSUMEXP"]:
        raise NotImplementedError(
            "torch.compile on CPU only supports inference and `return_lse` is not supported yet."
        )
    if not check_cpu_supported():
        raise NotImplementedError(
            "torch.compile on current platform is not supported for CPU."
        )

    fake_buffers: list[Buffer] = []

    # [Note] Handle the case where the split sizes are not statically known.
    # The value of cur_qSplitSize and cur_kvSplitSize are decided during runtime.
    # We use symbols to represent them during the compilation here.
    # They'll be replaced by the string "cur_qSplitSize" and "cur_kvSplitSize" in
    # the modification function of the CppFlexAttentionTemplate class.
    cur_qSplitSize = V.graph.sizevars.shape_env.create_unbacked_symint().node.expr
    cur_kvSplitSize = V.graph.sizevars.shape_env.create_unbacked_symint().node.expr
    shape_env = V.graph.sizevars.shape_env

    # We don't know the concrete value of cur_qSplitSize and cur_kvSplitSize during the compilation.
    # Mark symbols > 1 to ensure broadcasting is always applied.
    # This avoids treating them as equal when `eq(var, 1)` is evaluated in `broadcast_symbolic_shapes`.
    shape_env.var_to_range[cur_qSplitSize] = ValueRanges(2, int_oo)
    shape_env.var_to_range[cur_kvSplitSize] = ValueRanges(2, int_oo)

    score_dtype = torch.float
    placeholder_inps = [
        create_placeholder(name, dtype, query.get_device(), size)
        for name, dtype, size in [
            ("score", score_dtype, [cur_qSplitSize, cur_kvSplitSize]),
            ("b", torch.int64, []),
            ("h", torch.int64, []),
            ("q_idx", torch.int64, [cur_qSplitSize, 1]),
            ("kv_idx", torch.int64, [1, cur_kvSplitSize]),
        ]
    ]
    subgraph_buffer = build_subgraph_buffer(
        placeholder_inps + list(score_mod_other_buffers), subgraph
    )
    if subgraph_buffer is not None:
        if isinstance(subgraph_buffer, list):
            for _buf in subgraph_buffer:
                if _buf is not None:
                    _buf.freeze_layout()
        else:
            subgraph_buffer.freeze_layout()
    mask_graph_placeholder_inps = [
        create_placeholder(name, dtype, query.get_device(), size)
        for name, dtype, size in [
            ("score", score_dtype, [cur_qSplitSize, cur_kvSplitSize]),
            ("b", torch.int64, []),
            ("h", torch.int64, []),
            ("q_idx", torch.int64, [cur_qSplitSize, 1]),
            ("kv_idx", torch.int64, [1, cur_kvSplitSize]),
        ]
    ]

    # The original mask_graph works on a scalar and only includes
    # the logic of calculating the mask value.
    # We need to add the logic of applying the mask to the qk_data tensor
    # into the graph for the later codegen of this part.
    # Example:
    #   mask_graph:
    #   def mask_fn(b, h, q_idx, kv_idx):
    #       mask = q_idx >= kv_idx
    #       return mask
    #   The converted_mask_graph should be:
    #   def converted_mask_fn(qk_data, b, h, q_idx, kv_idx):
    #       mask = q_idx >= kv_idx
    #       qk_data = torch.where(mask, qk_data, torch.full_like(qk_data, -float("inf")))
    #       return qk_data
    def convert_mask_graph_module(mask_graph):
        gm = copy.deepcopy(mask_graph.graph_module)
        graph = gm.graph
        # Add qk_data as the first input
        with graph.inserting_before(next(iter(graph.nodes))):
            qk_data_node = graph.placeholder("qk_data")

        # Find the node that returns the mask
        output_node = None
        for node in graph.nodes:
            if node.op == "output":
                output_node = node
                break

        # Get the mask node
        if output_node is None:
            raise AssertionError("output_node must not be None")
        mask_node = output_node.args[0]

        size_node = [cur_qSplitSize, cur_kvSplitSize]
        # Create a new node for torch.full
        with graph.inserting_after(mask_node):
            full_node = graph.call_function(
                torch.full,
                args=(size_node, -float("inf")),
                kwargs={"dtype": score_dtype},
            )

        # Create a new node for torch.where
        with graph.inserting_after(full_node):
            where_node = graph.call_function(
                torch.ops.aten.where, args=(mask_node, qk_data_node, full_node)
            )

        # Update the output node to return the result of torch.where
        output_node.args = (where_node,)

        graph.lint()
        converted = torch.fx.GraphModule(gm, graph)
        return converted

    converted_mask_graph_module = convert_mask_graph_module(mask_graph)

    mask_graph_buffer = build_subgraph_module_buffer(
        mask_graph_placeholder_inps + list(mask_mod_other_buffers),
        converted_mask_graph_module,
    )

    # Clear the pending fresh unbacked symbols that are created for cur_qSplitSize and cur_kvSplitSize in the current kernel.
    pending = V.graph.sizevars.shape_env.pending_fresh_unbacked_symbols
    V.graph.sizevars.shape_env.pending_fresh_unbacked_symbols = [
        x for x in pending if x not in (cur_qSplitSize, cur_kvSplitSize)
    ]

    buffer_list = (
        placeholder_inps
        + list(score_mod_other_buffers)
        + mask_graph_placeholder_inps
        + list(mask_mod_other_buffers)
    )
    for item in buffer_list:
        if isinstance(item, TensorBox):
            fake_buffers.append(item.data.data)  # type: ignore[attr-defined]

    # CPU kernel requires last dim to be contiguous
    query, key, value = map(contiguous_last_dim, [query, key, value])

    (
        query,
        key,
        value,
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        q_num_blocks,
        q_indices,
        full_q_num_blocks,
        full_q_indices,
    ) = maybe_realize(
        [
            query,
            key,
            value,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            q_num_blocks,
            q_indices,
            full_q_num_blocks,
            full_q_indices,
        ]
    )

    query, key, value = _realize_duplicate_buffer_inputs([query, key, value])
    if query.get_dtype() not in [torch.float, torch.bfloat16, torch.float16]:
        raise NotImplementedError(
            "`torch.float` , `torch.float16` and `torch.bfloat16` are supported in FlexAttention for CPU device. "
            f"Found input tensors are `{query.get_dtype()}`."
        )
    score_mod_other_buffers = maybe_realize(score_mod_other_buffers)
    mask_mod_other_buffers = maybe_realize(mask_mod_other_buffers)
    Bq, Hq, seq_len_q, qk_head_dim = query.get_size()
    Bkv, Hkv, seq_len_kv, v_head_dim = value.get_size()
    B = Bq

    # Construct output layout with strides matching the query.
    out_size = [B, Hq, seq_len_q, v_head_dim]
    out_strides = infer_dense_strides(out_size, query.get_stride())

    layout = FixedLayout(
        query.get_device(),
        query.get_dtype(),
        [B, Hq, seq_len_q, v_head_dim],
        stride=[sympy.sympify(s) for s in out_strides],
    )
    _choices: list[Any] = []
    input_nodes = [query, key, value, kv_num_blocks, kv_indices]
    if not full_kv_num_blocks:
        no_full_kv_block = True
    else:
        no_full_kv_block = False
        input_nodes += [full_kv_num_blocks]
        input_nodes += [full_kv_indices]
    has_other_buffer = False
    kernel_input_name_to_buffer = {}
    if score_mod_other_buffers or mask_mod_other_buffers:
        has_other_buffer = True

        for prefix, buffers in [
            ("score_others", score_mod_other_buffers),
            ("mask_others", mask_mod_other_buffers),
        ]:
            kernel_input_name_to_buffer.update(
                {f"{prefix}_{i}": buf for i, buf in enumerate(buffers)}
            )
        input_nodes += [
            value
            for value in kernel_input_name_to_buffer.values()
            if not isinstance(value, sympy.Expr)
        ]

    skip_mask_score = kernel_options.get("SKIP_MASK_SCORE", False)
    # Mark SPARSE_KV_BLOCK_SIZE & SPARSE_Q_BLOCK_SIZE as static shapes and add guards.
    SPARSE_KV_BLOCK_SIZE = V.graph.sizevars.guard_int(SPARSE_KV_BLOCK_SIZE)
    SPARSE_Q_BLOCK_SIZE = V.graph.sizevars.guard_int(SPARSE_Q_BLOCK_SIZE)
    # In flash decoding, the partition size of doing the parallelism on KV length dim
    PARTITION_SIZE = kernel_options.get("PARTITION_SIZE", 128)
    if not V.graph.sizevars.evaluate_expr(
        sympy.Le(seq_len_q, sympy.Mul(kv_indices.get_size()[-2], SPARSE_Q_BLOCK_SIZE))
    ):
        raise AssertionError(
            "Q seqlen must be smaller than the block_mask size in the Q dimension, considering pass a larger block_mask."
        )
    if not V.graph.sizevars.evaluate_expr(
        sympy.Le(seq_len_kv, sympy.Mul(kv_indices.get_size()[-1], SPARSE_KV_BLOCK_SIZE))
    ):
        raise AssertionError(
            "KV seqlen must be smaller than the block_mask size in the KV dimension, considering pass a larger block_mask."
        )
    CppFlexAttentionTemplate.add_choices(
        choices=_choices,
        input_nodes=input_nodes,
        layout=layout,
        scale=scale,
        score_mod=None if skip_mask_score else subgraph_buffer,
        mask_mod=None if skip_mask_score else mask_graph_buffer,
        kv_block_size=SPARSE_KV_BLOCK_SIZE,
        q_block_size=SPARSE_Q_BLOCK_SIZE,
        partition_size=PARTITION_SIZE,
        has_other_buffer=has_other_buffer,
        no_full_kv_block=no_full_kv_block,
        fake_buffers=fake_buffers,
        len_score_other=len(score_mod_other_buffers),
        len_mask_other=len(mask_mod_other_buffers),
        kernel_input_name_to_buffer=kernel_input_name_to_buffer,
        block_vars=(cur_qSplitSize, cur_kvSplitSize),
    )
    inputs_for_autotuning = [
        query,
        key,
        value,
    ]
    res, _ = autotune_select_algorithm(
        "flex_attention",
        _choices,
        inputs_for_autotuning,
        layout,
    )

    # need subgraph inputs and outputs to analyze all symints used in flex attention
    res.data.data.subgraph_inps = list(score_mod_other_buffers) + list(
        mask_mod_other_buffers
    )
    res.data.data.subgraph_outs = get_fwd_subgraph_outputs(
        subgraph_buffer, mask_graph_buffer
    )

    return (res,)
<|endoftext|># mypy: allow-untyped-defs
"""Common utilities and functions for flex attention kernels"""

import math
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any, TYPE_CHECKING

import sympy

import torch
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_map, tree_map_only
from torch.utils._sympy.functions import Mod


if TYPE_CHECKING:
    from torch._inductor.codegen.cuda_combined_scheduling import _IntLike
else:
    _IntLike = int | sympy.Expr


from ...ir import (
    ComputedBuffer,
    ExternKernel,
    FixedLayout,
    FlexibleLayout,
    get_fill_order,
    InputBuffer,
    IRNode,
    MutationLayoutSHOULDREMOVE,
    Scatter,
    StorageBox,
    Subgraph,
    TensorBox,
)
from ...lowering import (
    _full,
    check_and_broadcast_indices,
    expand,
    index_output_size_and_inner_fn,
    to_dtype,
)
from ...select_algorithm import realize_inputs
from ...utils import load_template


SubgraphResults = list[ComputedBuffer | None] | ComputedBuffer | None


def can_skip_boundary_checks(seq_len, sparse_block_size) -> bool:
    """True when per-tile bounds masking can be skipped along this dim.

    This is decided before a config is chosen, so divisibility is checked
    against 128, the max (and LCM) of all candidate pow2 tile sizes for
    inner block_m/n.
    """
    return V.graph.sizevars.statically_known_true(
        sympy.And(
            sympy.Eq(Mod(seq_len, 128), 0),
            sympy.Or(
                sympy.Eq(Mod(seq_len, sparse_block_size), 0),
                sympy.Ge(sparse_block_size, seq_len),
            ),
        )
    )


def is_tensor_ir_node(node: object) -> bool:
    return isinstance(node, IRNode) and node.has_tensor_output()


def _flex_kernel_options_example(kind: str) -> str:
    match kind:
        case "backward":
            return (
                "kernel_options={'bwd_BLOCK_M1': 32, 'bwd_BLOCK_N1': 32, "
                "'bwd_BLOCK_M2': 32, 'bwd_BLOCK_N2': 32, "
                "'bwd_num_stages': 1, 'bwd_num_warps': 4}"
            )
        case _:
            return (
                "kernel_options={'fwd_BLOCK_M': 32, 'fwd_BLOCK_N': 64, "
                "'fwd_num_stages': 1, 'fwd_num_warps': 4}"
            )


def _flex_kernel_tuning_options(kind: str) -> str:
    match kind:
        case "backward":
            return (
                "BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2, num_warps, and "
                "num_stages; use the bwd_ prefix to set backward-only options"
            )
        case "decode":
            return (
                "BLOCK_M, BLOCK_N, num_warps, and num_stages; use the fwd_ "
                "prefix to set decode-only options"
            )
        case _:
            return (
                "BLOCK_M, BLOCK_N, num_warps, and num_stages; use the fwd_ "
                "prefix to set forward-only options"
            )


def zeros_and_scatter_lowering(shape: list[int], indices, values):
    """To support backwards on captured buffers we register a specific lowering for our specific custom up"""
    # Always accumulate into fp32 then cast
    grad = _full(0, values.get_device(), torch.float32, shape)
    if not isinstance(grad, TensorBox):
        raise AssertionError(f"Expected TensorBox, got {type(grad)}")
    grad.realize()
    x_size = grad.get_size()
    values = to_dtype(values, grad.get_dtype())
    device = grad.get_device()
    if device is None:
        raise AssertionError("device must not be None")
    if not indices:
        if shape:
            raise AssertionError(
                "zeros_and_scatter with no indices only supports scalar outputs"
            )
        expected_vals_size = values.get_size()

        def inner_fn(index):
            return []

    else:
        indices_loaders = [i.make_loader() if i is not None else None for i in indices]
        indices, tensor_indices = check_and_broadcast_indices(
            indices, grad.get_device()
        )
        # We can use the first one since they are all required to be the same size
        tensor_size = list(indices[tensor_indices[0]].get_size())
        indexed_size = [x_size[i] for i in range(len(indices))]

        expected_vals_size, inner_fn = index_output_size_and_inner_fn(
            x_size,
            indices,
            tensor_indices,
            tensor_size,
            indices_loaders,
            indexed_size,
            None,
            check=True,
        )
        values = expand(values, expected_vals_size)
    scatter = Scatter(
        device=device,
        dtype=grad.get_dtype(),
        inner_fn=values.make_loader(),
        ranges=expected_vals_size,  # iter_ranges,
        output_indexer=inner_fn,
        scatter_mode="atomic_add",
    )

    buffer = ComputedBuffer(
        name=grad.data.data.name,  # type: ignore[attr-defined]
        layout=MutationLayoutSHOULDREMOVE(grad),
        data=scatter,
    )
    return buffer


def get_fwd_subgraph_outputs(
    subgraph_buffer: SubgraphResults, mask_graph_buffer: SubgraphResults
) -> list[ComputedBuffer | None]:
    subgraph_buffer = (
        subgraph_buffer if isinstance(subgraph_buffer, Sequence) else [subgraph_buffer]
    )
    mask_graph_buffer = (
        mask_graph_buffer
        if isinstance(mask_graph_buffer, Sequence)
        else [mask_graph_buffer]
    )

    return [*subgraph_buffer, *mask_graph_buffer]


def build_subgraph_module_buffer(
    args: list[TensorBox],
    graph_module: torch.fx.GraphModule,
) -> SubgraphResults:
    """This function's goal is to take in the required args and produce the subgraph buffer
    The subgraph buffer is a ComputedBuffer that will be inlined into the triton template

    Args:
        args: The args that are passed into the subgraph. Contains both fixed and lifted inputs.
        subgraph: The Subgraph ir for which to produce the output node
    """
    # This one we gotta keep lazy
    from ...subgraph_lowering import PointwiseSubgraphLowering

    pw_subgraph = PointwiseSubgraphLowering(
        graph_module,
        root_graph_lowering=V.graph,
        allowed_mutations=OrderedSet([torch.ops.flex_lib.zeros_and_scatter.default]),
        additional_lowerings={
            torch.ops.flex_lib.zeros_and_scatter.default: zeros_and_scatter_lowering
        },
    )
    with V.set_graph_handler(pw_subgraph):  # type: ignore[arg-type]
        pw_subgraph.run(*args)

    def convert_output_node_to_buffer(output_buffer) -> ComputedBuffer | None:
        if output_buffer is None:
            return None
        if isinstance(output_buffer, ComputedBuffer):
            # These nodes are coming from the output of zeros_and_scatter
            return output_buffer
        if not isinstance(output_buffer, TensorBox):
            raise AssertionError(
                f"The output node for flex attention's subgraph must be a TensorBox, but got: {type(output_buffer)}"
            )
        if not isinstance(output_buffer.data, StorageBox):
            raise AssertionError(
                f"The output node for the flex attention subgraph must be a StorageBox, but got: {type(output_buffer.data)}"
            )
        device = output_buffer.data.get_device()
        if device is None:
            raise AssertionError("device must not be None for output buffer")
        subgraph_buffer = ComputedBuffer(
            name=None,
            layout=FlexibleLayout(
                device=device,
                dtype=output_buffer.data.get_dtype(),
                size=output_buffer.data.get_size(),
            ),
            data=output_buffer.data.data,  # type: ignore[arg-type]
        )
        return subgraph_buffer

    return tree_map(convert_output_node_to_buffer, pw_subgraph.graph_outputs)


def build_subgraph_buffer(args: list[TensorBox], subgraph: Subgraph) -> SubgraphResults:
    return build_subgraph_module_buffer(args, subgraph.graph_module)


def maybe_realize(args: list[IRNode | None]):
    """Accepts a list of optional IRNodes and returns a list of realized IRNodes"""
    return tree_map(
        lambda x: (
            realize_inputs(x) if x is not None and not isinstance(x, sympy.Expr) else x
        ),
        args,
    )


def realize_captures_for_cutedsl(buffers):
    """Realize captured buffers for CuteDSL, preserving views and plain inputs.

    Unlike maybe_realize (used by the Triton path), CuteDSL needs physical
    tensors passed at runtime. Pointwise/computed captures must be materialized
    into a fresh buffer whose layout matches the logical shape the subgraph
    indexes. ReinterpretView captures use unique synthetic inputs so aliased
    views keep distinct call-site size/stride/offset metadata while the kernel
    indexes each captured view argument from offset zero.

    Realized captures are registered on V.graph so the CuteDSL template can
    resolve view nodes without explicit plumbing from callers.
    """
    from ...ir import ExternKernel, FixedLayout, InputBuffer, ReinterpretView

    view_captures: dict[str, ReinterpretView] = {}

    def _add_alignment_check_for_input(input_buffer: InputBuffer) -> None:
        # Preserved captures can be used by vectorized CuteDSL loads, so make
        # sure the wrapper still emits copy_if_misaligned for graph inputs.
        # Normal template inputs are already in inputs_to_check; captures need
        # to be added explicitly because they were not direct template inputs
        # when alignment-check inputs were selected.
        name = input_buffer.get_name()
        graph_input_names = getattr(V.graph, "graph_input_names", [])
        if name in graph_input_names:
            idx = graph_input_names.index(name)
            inputs_to_check = list(V.graph.inputs_to_check or ())
            if idx not in inputs_to_check:
                V.graph.inputs_to_check = [*inputs_to_check, idx]

    def _realize(x):
        if x is None or isinstance(x, sympy.Expr):
            return x
        realized = ExternKernel.realize_input(x)
        if isinstance(realized, StorageBox) and realized.is_input_buffer():
            # Plain graph inputs are commonly represented as
            # TensorBox(StorageBox(InputBuffer(...))).  Preserve the underlying
            # input/view instead of materializing it with ExternKernel.copy_input.
            realized = realized.data
        if isinstance(realized, ReinterpretView):
            layout = realized.get_layout()
            capture_index = len(V.graph._cutedsl_capture_nodes) + len(view_captures)
            name = V.graph.qualify_name(f"cutedsl_capture{capture_index}")
            view_captures[name] = realized
            # Give each captured view a logical input name so aliasing views do
            # not collapse to their shared base buffer.
            return InputBuffer(
                name=name,
                layout=FixedLayout(
                    layout.device,
                    layout.dtype,
                    layout.size,
                    layout.stride,
                    is_pinned=layout.is_pinned,
                ),
            )
        if isinstance(realized, InputBuffer):
            _add_alignment_check_for_input(realized)
            return realized
        return ExternKernel.copy_input(realized)

    buffers = tree_map(_realize, buffers)
    freeze_irnodes(buffers)

    for buf in tree_map_only(IRNode, lambda x: x, buffers) if buffers else []:
        if isinstance(buf, IRNode) and (name := buf.maybe_get_name()):
            V.graph._cutedsl_capture_nodes[name] = buf
    # Keep the original view nodes for call-site reinterpret_tensor emission.
    V.graph._cutedsl_capture_nodes.update(view_captures)

    return buffers


def freeze_irnodes(tree: Any) -> Any:
    """Freeze layouts for every IRNode contained in a pytree."""

    if tree is None:
        return None

    def _freeze(node: IRNode) -> IRNode:
        try:
            node.freeze_layout()
        except NotImplementedError:
            pass
        return node

    return tree_map_only(IRNode, _freeze, tree)


def create_placeholder(
    name: str,
    dtype: torch.dtype,
    device: torch.device,
    size: list[int] | None = None,
) -> TensorBox:
    """Creates a placeholder input buffers for producing subgraph_output."""
    input_buffer = InputBuffer(
        name=name,
        layout=FixedLayout(
            device,
            dtype,
            size if size else [],
            FlexibleLayout.contiguous_strides(size) if size else [],
        ),
    )
    return TensorBox.create(input_buffer)


def construct_strides(
    sizes: Sequence[_IntLike],
    fill_order: Sequence[int],
) -> Sequence[_IntLike]:
    """From a list of sizes and a fill order, construct the strides of the permuted tensor."""
    # Initialize strides
    if len(sizes) != len(fill_order):
        raise AssertionError("Length of sizes must match the length of the fill order")
    strides: list[_IntLike] = [0] * len(sizes)

    # Start with stride 1 for the innermost dimension
    current_stride: _IntLike = 1

    # Iterate through the fill order populating strides
    for dim in fill_order:
        strides[dim] = current_stride
        current_stride *= sizes[dim]

    return strides


def infer_dense_strides(
    size: Sequence[_IntLike],
    orig_strides: Sequence[_IntLike],
):
    """This is a mirror of the same function in aten/src/ATen/ExpandUtils.cpp

    Args:
        size: The size of the output tensor
        orig_strides: The strides of the input tensor
    Returns:
        List[int]: Dense non-overlapping strides that preserve the input tensor's layout permutation.
        The returned strides follow the same stride propagation rules as TensorIterator. This matches
        The behavior of empty_like()
    """
    fill_order = get_fill_order(orig_strides, V.graph.sizevars.shape_env)
    strides = construct_strides(size, fill_order)

    # Attention kernels require stride[-1]=1 for efficient memory access.
    # Ensure this by moving last dim to front of fill_order if needed.
    if strides[-1] != 1:
        last_dim = len(size) - 1
        fill_order = list(fill_order)
        fill_order.remove(last_dim)
        fill_order = [last_dim] + fill_order
        strides = construct_strides(size, fill_order)

    return strides


def create_indices_fake(x) -> torch.Tensor:
    """Create a fake indices that is used for autotuning."""
    size = V.graph.sizevars.optimization_hints(x.get_size())
    indices = torch.arange(0, size[-1], dtype=x.get_dtype(), device=x.get_device())
    indices = indices.expand(size).contiguous()
    return indices


def create_num_blocks_fake_generator(sparse_indices):
    """Create a fake num_blocks that is used for autotuning.

    The idea here is that we need to create a real tensor with real data
    that's representative for benchmarking.
    For example, returning all zeros for the `kv_num_blocks` input would mean
    that we are computing 0 blocks for each row, which would provide bogus
    autotuning results.

    In this case, we choose to use min(16, max_block) blocks, because I
    (Horace) think it'll probably result in pretty representative performance.
    If it's too short then prefetching won't help. If it's too long then
    autotuning will take longer for no good reason.
    """

    def create_num_blocks_fake(x) -> torch.Tensor:
        num_blocks_for_autotuning = V.graph.sizevars.optimization_hint(
            sparse_indices.shape[-1]
        )
        size = V.graph.sizevars.optimization_hints(x.get_size())
        return torch.full(
            size,
            num_blocks_for_autotuning,
            dtype=x.get_dtype(),
            device=x.get_device(),
        )

    return create_num_blocks_fake


def contiguous_last_dim(x):
    """Ensure that realized IR node has a contiguous stride in the last dimension."""
    strides = x.maybe_get_stride()
    if strides and strides[-1] != 1:
        contiguous_stride_order = list(reversed(range(len(x.get_size()))))
        return ExternKernel.require_stride_order(x, contiguous_stride_order)
    return x


def set_head_dim_values(
    kernel_options: dict[str, Any], qk_head_dim, v_head_dim, graph_sizevars
):
    """
    Mutates kernel options, adding head dimension calculations.

    Args:
        kernel_options: Dictionary to populate with options
        qk_head_dim: Query/Key head dimension
        v_head_dim: Value head dimension
        graph_sizevars: Graph size variables object with guard_int method

    """
    # QK dimensions
    qk_head_dim_static = graph_sizevars.guard_int(qk_head_dim)
    kernel_options.setdefault("QK_HEAD_DIM", qk_head_dim_static)
    kernel_options.setdefault(
        "QK_HEAD_DIM_ROUNDED", next_power_of_two(qk_head_dim_static)
    )

    # V dimensions
    v_head_dim_static = graph_sizevars.guard_int(v_head_dim)
    kernel_options.setdefault("V_HEAD_DIM", v_head_dim_static)
    kernel_options.setdefault(
        "V_HEAD_DIM_ROUNDED", next_power_of_two(v_head_dim_static)
    )

    # Safety flag
    kernel_options.setdefault(
        "SAFE_HEAD_DIM",
        is_power_of_2(qk_head_dim_static) and is_power_of_2(v_head_dim_static),
    )


def is_power_of_2(n):
    return n != 0 and ((n & (n - 1)) == 0)


def next_power_of_two(n):
    if n <= 0:
        return 1
    return 2 ** math.ceil(math.log2(n))


_FLEX_TEMPLATE_DIR = Path(__file__).parent / "templates"
load_flex_template = partial(load_template, template_dir=_FLEX_TEMPLATE_DIR)


# Template strings have been moved to templates/common.py.jinja
<|endoftext|># mypy: allow-untyped-defs
"""Triton Implementation of the flex_attention Kernel for short query length (FlexDecoding)"""

import logging
from typing import Any

import sympy

import torch
from torch._inductor.virtualized import V
from torch.utils._sympy.functions import FloorDiv, Mod

from ... import ir
from ...ir import FixedLayout, FlexibleLayout
from ...lowering import empty, empty_strided, lowerings
from ...runtime.runtime_utils import ceildiv, is_power_of_2, next_power_of_2
from ...select_algorithm import (
    autotune_select_algorithm,
    SymbolicGridFn,
    TritonTemplate,
)
from ...utils import can_use_tma
from .common import (
    _flex_kernel_options_example,
    _flex_kernel_tuning_options,
    can_skip_boundary_checks,
    create_indices_fake,
    create_num_blocks_fake_generator,
    freeze_irnodes,
    get_fwd_subgraph_outputs,
    is_tensor_ir_node,
    load_flex_template,
    maybe_realize,
    set_head_dim_values,
)


aten = torch.ops.aten
prims = torch.ops.prims

log = logging.getLogger(__name__)


def raise_flex_decoding_kernel_options_error(
    kernel_options: dict[str, Any],
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
) -> None:
    formated_kernel_options = ", ".join(
        f"{name}={kernel_options[name]}" for name in ("BLOCK_M", "BLOCK_N")
    )
    raise ValueError(
        "Invalid FlexAttention decode kernel options: Q and KV block sizes must "
        "be divisible by the selected tile sizes. Got "
        f"SPARSE_Q_BLOCK_SIZE={sparse_q_block_size}, "
        f"SPARSE_KV_BLOCK_SIZE={sparse_kv_block_size}, and "
        f"{formated_kernel_options}. "
        "Pass compatible values with kernel_options. Available decode tuning "
        f"options are {_flex_kernel_tuning_options('decode')}. For example: "
        f"{_flex_kernel_options_example('decode')}. If you did not pin "
        "these options, compiling with mode='max-autotune-no-cudagraphs' "
        "can also fix this by trying more FlexAttention configs."
    )


def _use_flex_decoding(query, kv_indices, value, kernel_options, enable_gqa) -> bool:
    """Decide which kernel to use, return true if use flex decoding kernel.
    Note:
       Since the number of splits is calculated based on the number of batch and head dims
       we need to ensure that the batch and head dims are statically known. Otherwise we just
       use the main flex_attention kernel.
    """
    force_flex = kernel_options.get("FORCE_USE_FLEX_ATTENTION", False)

    # Decode eligibility is an optimization choice, not a user-visible contract,
    # so every predicate below uses guard_or_false (case 2 of Note [guard_or_]:
    # the program behaves equivalently whether we pick decode or the general
    # kernel). guard_or_false returns the real result when ShapeEnv can prove or
    # guard it, and conservatively returns False for an unprovable unbacked size.
    # We deliberately do NOT use torch._check here: failing to prove eligibility
    # must fall back to the general FlexAttention kernel, never impose a runtime
    # constraint on the user's shapes. We also do NOT use guard_or_true: an
    # unknown predicate should disable decode, not silently enable it.
    short_query_length = V.graph.sizevars.guard_or_false(
        sympy.Lt(query.get_size()[-2], 128)
    )
    non_zero_length = V.graph.sizevars.guard_or_false(sympy.Gt(query.get_size()[-2], 0))
    static_batch = isinstance(query.get_size()[0], (int, sympy.Integer))
    static_num_heads = isinstance(query.get_size()[1], (int, sympy.Integer))
    if enable_gqa:
        # in the current flex decoding triton kernel, grouped query heads for the
        # same kv head are handled by the same block. So it's hard to support different
        # kv num blocks for grouped query heads. We just fall back to main flex_attention
        # kernel where each query head is handled by a separate block.
        valid_block_mask_num_heads = V.graph.sizevars.guard_or_false(
            sympy.Eq(kv_indices.get_size()[1], 1)
        )
    else:
        valid_block_mask_num_heads = V.graph.sizevars.guard_or_false(
            sympy.Or(
                sympy.Eq(kv_indices.get_size()[1], 1),
                sympy.Eq(kv_indices.get_size()[1], query.get_size()[1]),
            )
        )

    Hq = query.get_size()[1]
    Hkv = value.get_size()[1]
    ratio = FloorDiv(Hq, Hkv)

    pw_of_two = V.graph.sizevars.guard_or_false(
        sympy.And(sympy.Gt(ratio, 0), sympy.Eq(ratio & (ratio - 1), 0))
    )

    out = (
        not force_flex
        and not kernel_options.get("OUTPUT_MAX", False)
        and short_query_length
        and static_batch
        and static_num_heads
        and non_zero_length
        and valid_block_mask_num_heads
        and pw_of_two
    )
    log.debug(
        "Use flex decoding %s, force_flex_attention=%s, short_query_length=%s, static_batch=%s, static_num_heads=%s",
        out,
        force_flex,
        short_query_length,
        static_batch,
        static_num_heads,
    )
    return out


@SymbolicGridFn
def flex_decoding_grid(batch_size, kv_heads, gqa_group_size, seq_len_q, d_model, meta):
    """How is this kernel parallelized?
    We create a grid of (batch_size * kv_heads, SPLIT_KV, 1)
    Each block is responsible for iterating over blocks of keys and values calculating
    the local output for their tile of keys and values over all full length of query.
    groups of SPLIT_KV blocks then combine their output to produce the final result.
    """

    BLOCK_M = meta["BLOCK_M"]
    num_block_m = ceildiv(seq_len_q * gqa_group_size, BLOCK_M)
    return (num_block_m, batch_size * kv_heads, meta["SPLIT_KV"])


flex_decoding_template = TritonTemplate(
    name="flex_decoding",
    grid=flex_decoding_grid,
    source=load_flex_template("flex_decode")
    + load_flex_template("utilities")
    + load_flex_template("common"),
    always_freeze_layout=True,
)


def get_split_k(B: int, H: int, Mk: int) -> int:
    if torch.xpu.is_available():
        num_SM = torch.xpu.get_device_properties("xpu").gpu_subslice_count
    else:
        num_SM = torch.cuda.get_device_properties("cuda").multi_processor_count
    bh = max(B * H, 1)  # NOTE: Handle B*h=0 case
    if not isinstance(bh, (int, sympy.Integer)):
        raise AssertionError("B and H must be concrete integers")
    split_k = num_SM // bh * 2  # Each SM should at least get one block.
    # TODO: workload evening at runtime for splits fully masked out.
    # Before we have runtime workload evening, assign 2 splits per SM.
    split_k = max(split_k, 1)

    return split_k


def create_flex_decoding_kernel(*args, **kwargs):
    """Flex decode lowering that is optimized for small Q_LEN and GQA packing"""
    (
        query,
        key,
        value,
        block_mask,
        scale,
        kernel_options,
        score_mod_subgraph,
        mask_mod_subgraph,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    ) = args
    (
        _,  # q_length
        _,  # kv_length
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,  # full_kv_num_blocks,
        full_kv_indices,  # full_kv_indices,
        _,  # q_num_blocks
        _,  # q_indices
        _,  # full_q_num_blocks,
        _,  # full_q_indices,
        _,  # dq_write_order
        _,  # dq_write_order_full
        _,  # dq_kv_order
        _,  # dq_kv_order_spt
        SPARSE_Q_BLOCK_SIZE,
        SPARSE_KV_BLOCK_SIZE,
        _,
    ) = block_mask

    Bq, Hq, seq_len_q, qk_head_dim = query.get_size()
    Bkv, Hkv, seq_len_kv, v_head_dim = value.get_size()

    if not V.graph.sizevars.evaluate_expr(sympy.Eq(Bq, Bkv) | sympy.Eq(Bkv, 1)):
        raise AssertionError(
            f"Bq and Bkv must broadcastable. Got Bq={Bq} and Bkv={Bkv}"
        )

    B = Bq
    kernel_options = dict(kernel_options)
    # Mark symbols in custom kernel options as static shapes and add guards.
    kernel_options = {
        k: V.graph.sizevars.guard_int(v) if isinstance(v, sympy.Symbol) else v
        for k, v in kernel_options.items()
    }

    seq_q_divisible = can_skip_boundary_checks(seq_len_q, SPARSE_Q_BLOCK_SIZE)
    seq_kv_divisible = can_skip_boundary_checks(seq_len_kv, SPARSE_KV_BLOCK_SIZE)
    if seq_q_divisible and seq_kv_divisible:
        kernel_options.setdefault("IS_DIVISIBLE", True)
    else:
        kernel_options.setdefault("IS_DIVISIBLE", False)

    # Calculate GQA head sharing
    gqa_shared_heads = FloorDiv(Hq, Hkv)
    if not is_power_of_2(gqa_shared_heads):
        raise ValueError(
            "Number of shared query heads sharing the same KV head must be power of 2. "
        )
    kernel_options.setdefault("GQA_SHARED_HEADS", gqa_shared_heads)

    # Determine if there are "full" blocks where we only need to apply score_mod, and can skip mask_mod
    has_full_blocks = full_kv_num_blocks is not None
    kernel_options.setdefault("HAS_FULL_BLOCKS", has_full_blocks)
    if not has_full_blocks:
        # Create a placeholder full block list in case it is empty
        full_kv_num_blocks, full_kv_indices = (
            empty(0, device=query.get_device()) for _ in range(2)
        )

    (
        query,
        key,
        value,
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
    ) = maybe_realize(
        [
            query,
            key,
            value,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
        ]
    )
    score_mod_other_buffers = maybe_realize(score_mod_other_buffers)
    mask_mod_other_buffers = maybe_realize(mask_mod_other_buffers)

    freeze_irnodes(score_mod_other_buffers)
    freeze_irnodes(mask_mod_other_buffers)

    choices: list[Any] = []
    dtype = key.get_dtype()
    head_dim = V.graph.sizevars.guard_int(key.get_size()[-1])
    configs = V.choices.get_flex_decode_configs(
        head_dim, dtype, query.get_device().type
    )

    # TODO: fix autotuning.

    kernel_options.setdefault("SM_SCALE", scale)
    kernel_options.setdefault("SPLIT_KV", get_split_k(B, Hkv, seq_len_kv))
    MAX_SPLIT_KV = kernel_options["SPLIT_KV"]

    # create config dependent intermediate buffers
    buf_ACC_shape = [B, MAX_SPLIT_KV, Hq, seq_len_q, v_head_dim]
    buf_ML_shape = buf_ACC_shape[:-1]
    buf_M = empty_strided(
        buf_ML_shape,
        None,
        dtype=torch.float32,  # The rowmax is always stored in fp32 regardless of the input dtype
        device=query.get_device(),
    )
    buf_L = empty_strided(
        buf_ML_shape,
        None,
        dtype=torch.float32,  # The intermediate sumexp is always stored in fp32 regardless of the input dtype
        device=query.get_device(),
    )

    layout_acc = FixedLayout(
        query.get_device(),
        torch.float32,
        buf_ACC_shape,
        FlexibleLayout.contiguous_strides(buf_ACC_shape),
    )

    set_head_dim_values(kernel_options, qk_head_dim, v_head_dim, V.graph.sizevars)

    kernel_options.setdefault(
        "BLOCK_M",
        (
            # m
            # if V.graph.sizevars.evaluate_expr(sympy.Lt(query.get_size()[-2], 0))
            # else  # Always use a BLOCK_M > 16 before Triton fix https://github.com/triton-lang/triton/pull/4061 is in pin
            max(
                next_power_of_2(
                    V.graph.sizevars.optimization_hint(seq_len_q) * gqa_shared_heads
                ),
                1 if torch.xpu.is_available() else 16,
            )
        ),
    )

    query = ir.ExternKernel.realize_input(query)
    stride_b, stride_hq, stride_seq_len_q, stride_qk_head_dim = query.get_stride()

    # Reshape query for GQA: [B, Hq, Mq, D] -> [B, Hkv, G, Mq, D]
    gqa_query_shape = (B, Hkv, gqa_shared_heads, seq_len_q, qk_head_dim)
    gqa_query_stride = (
        stride_b,
        stride_hq * gqa_shared_heads,
        stride_hq,
        stride_seq_len_q,
        stride_qk_head_dim,
    )
    query = lowerings[aten.as_strided](query, gqa_query_shape, gqa_query_stride)

    kernel_options.setdefault(
        "SAFE_M_BOUNDARY",
        Mod(seq_len_q * gqa_shared_heads, kernel_options["BLOCK_M"]) == 0,
    )
    # TODO: This feels sketchy
    kernel_options.setdefault("SAFE_N_BOUNDARY", True)
    # Mark SPARSE_KV_BLOCK_SIZE as static shapes and add guards.
    SPARSE_Q_BLOCK_SIZE = V.graph.sizevars.guard_int(SPARSE_Q_BLOCK_SIZE)
    SPARSE_KV_BLOCK_SIZE = V.graph.sizevars.guard_int(SPARSE_KV_BLOCK_SIZE)

    original_kernel_options = kernel_options.copy()
    # Note, we don't need to pass in the captured buffers explicitly
    # because they're implicitly added by the score_mod function
    # We do need to explicitly pass it in for autotuning though.

    # Default config for warp specialization
    num_consumer_groups, num_buffers_warp_spec = 0, 0
    invalid_block_options: dict[str, Any] | None = None

    for conf in configs:
        cur_kernel_options = original_kernel_options.copy()
        # Remove prefix for forward kernels options and delete backward kernel options.
        for k in list(cur_kernel_options.keys()):
            if k.startswith("fwd_"):
                v = cur_kernel_options.pop(k)
                cur_kernel_options[k[4:]] = v
            if k.startswith("bwd_"):
                cur_kernel_options.pop(k)
        # Performance tuning
