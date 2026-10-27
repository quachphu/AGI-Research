        cur_kernel_options.setdefault(
            "BLOCK_N", min(conf.block_n, SPARSE_KV_BLOCK_SIZE)
        )
        cur_kernel_options.setdefault("SPARSE_Q_BLOCK_SIZE", SPARSE_Q_BLOCK_SIZE)
        cur_kernel_options.setdefault("SPARSE_KV_BLOCK_SIZE", SPARSE_KV_BLOCK_SIZE)
        cur_kernel_options.setdefault("num_warps", conf.num_warps)
        cur_kernel_options.setdefault("num_stages", conf.num_stages)

        if (
            cur_kernel_options["SPARSE_Q_BLOCK_SIZE"] % cur_kernel_options["BLOCK_M"]
            != 0
            or cur_kernel_options["SPARSE_KV_BLOCK_SIZE"]
            % cur_kernel_options["BLOCK_N"]
            != 0
        ):
            invalid_block_options = cur_kernel_options
            if len(configs) == 1:
                raise_flex_decoding_kernel_options_error(
                    cur_kernel_options,
                    SPARSE_Q_BLOCK_SIZE,
                    SPARSE_KV_BLOCK_SIZE,
                )
            continue

        if cur_kernel_options.get("num_consumer_groups", False):
            cur_kernel_options.setdefault("num_consumer_groups", num_consumer_groups)
            cur_kernel_options.setdefault(
                "num_buffers_warp_spec", num_buffers_warp_spec
            )

        # Intel GPU enables TMA by default
        cur_kernel_options.setdefault("USE_TMA", bool(torch.xpu.is_available()))

        if cur_kernel_options["USE_TMA"] and not can_use_tma(query, key, value):
            cur_kernel_options["USE_TMA"] = False

        # Add ROCm-specific parameters if they exist in the config
        for attrib in ["kpack", "matrix_instr_nonkdim", "waves_per_eu"]:
            if hasattr(conf, attrib):
                cur_kernel_options[attrib] = getattr(conf, attrib)

        flex_decoding_template.maybe_append_choice(
            choices=choices,
            input_nodes=[
                query,
                key,
                value,
                buf_M,
                buf_L,
                kv_num_blocks,
                kv_indices,
                full_kv_num_blocks,
                full_kv_indices,
            ],
            layout=layout_acc,
            subgraphs=[
                score_mod_subgraph,
                mask_mod_subgraph,
            ],
            mutated_inputs=[buf_M, buf_L],
            call_sizes=query.get_size(),
            **cur_kernel_options,
        )

    if not choices and invalid_block_options is not None:
        raise_flex_decoding_kernel_options_error(
            invalid_block_options,
            SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE,
        )

    inputs_for_flex_decoding = (
        [
            query,
            key,
            value,
            buf_M,
            buf_L,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
        ]
        + list(score_mod_other_buffers)
        + list(mask_mod_other_buffers)
    )

    input_gen_fns = {
        5: create_num_blocks_fake_generator(kv_indices),
        6: create_indices_fake,
        7: create_num_blocks_fake_generator(full_kv_indices),
        8: create_indices_fake,
    }

    buf_ACC, _ = autotune_select_algorithm(
        "flex_decoding",
        choices,
        # Autotuning materializes benchmark tensors. Scalar shape captures stay
        # in subgraph_inps below for dependency tracking and codegen.
        [x for x in inputs_for_flex_decoding if is_tensor_ir_node(x)],
        layout_acc,
        input_gen_fns=input_gen_fns,
    )

    # need subgraph inputs and outputs to analyze all symints used in flex attention
    buf_ACC.data.data.subgraph_inps = list(score_mod_other_buffers) + list(
        mask_mod_other_buffers
    )
    buf_ACC.data.data.subgraph_outs = get_fwd_subgraph_outputs(
        score_mod_subgraph, mask_mod_subgraph
    )

    # Reduction

    g_M = lowerings[aten.max](buf_M, dim=1, keepdim=True)[0]
    # See [Note] Handle fully masked out rows:
    # g_M Is the global max among split kv blocks.
    masked_rows = lowerings[aten.eq](g_M, -float("inf"))
    adj_M = lowerings[aten.sub](buf_M, g_M)
    adj_M = lowerings[aten.where](masked_rows, 0, adj_M)
    alpha = lowerings[aten.exp2](adj_M)

    buf_L = lowerings[aten.mul](buf_L, alpha)
    g_L = lowerings[aten.sum](buf_L, axis=1)
    masked_rows_squeezed = lowerings[aten.squeeze](masked_rows, dim=1)
    g_L = lowerings[aten.where](masked_rows_squeezed, 1.0, g_L)
    logsumexp = lowerings[aten.log2](g_L)
    logsumexp = lowerings[aten.add](logsumexp, lowerings[aten.squeeze](g_M, dim=1))

    alpha_unseq = lowerings[aten.unsqueeze](alpha, 4)
    buf_ACC = lowerings[aten.mul](buf_ACC, alpha_unseq)
    output = lowerings[aten.sum](buf_ACC, axis=1)
    L_unseq = lowerings[aten.unsqueeze](g_L, 3)
    output = lowerings[aten.div](output, L_unseq)
    output = lowerings[prims.convert_element_type](output, query.get_dtype())

    return (
        output,
        logsumexp,
    )
<|endoftext|># mypy: allow-untyped-defs
"""Triton Implementation of the flex_attention Kernel"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast, TYPE_CHECKING

import sympy

import torch
from torch._inductor.virtualized import V
from torch.nn.attention.flex_attention import _Backend
from torch.utils._sympy.functions import FloorDiv

from ...ir import ComputedBuffer, ExternKernel, FixedLayout, TensorBox
from ...lowering import empty, empty_strided, lowerings, register_lowering, to_dtype
from ...runtime.runtime_utils import is_power_of_2
from ...select_algorithm import (
    autotune_select_algorithm,
    SymbolicGridFn,
    TritonTemplate,
)
from ...utils import can_use_tma
from .common import (
    _flex_kernel_options_example,
    _flex_kernel_tuning_options,
    build_subgraph_buffer,
    can_skip_boundary_checks,
    create_indices_fake,
    create_num_blocks_fake_generator,
    create_placeholder,
    freeze_irnodes,
    get_fwd_subgraph_outputs,
    infer_dense_strides,
    is_tensor_ir_node,
    load_flex_template,
    maybe_realize,
    realize_captures_for_cutedsl,
    set_head_dim_values,
    SubgraphResults,
)
from .flex_cpu import lower_cpu
from .flex_decoding import _use_flex_decoding, create_flex_decoding_kernel
from .flex_flash_attention import (
    _use_flex_flash_attention,
    _use_flex_flash_attention_backward,
    create_flex_flash_attention_backward_kernel,
    create_flex_flash_attention_kernel,
    has_unsupported_cpu_scalar_tensor_captures,
    is_trivial_mask_graph,
    is_trivial_score_graph,
)


if TYPE_CHECKING:
    from ...heuristics.template.triton import FlexBwDConfig, FlexConfig


log = logging.getLogger(__name__)
aten = torch.ops.aten
prims = torch.ops.prims
Expr = sympy.Expr


def _sanitize_kernel_options_for_triton(
    kernel_options: dict[str, Any],
) -> tuple[dict[str, Any], _Backend]:
    """We always strip quotes around str values, we only need this in lowering, so we pop it here
    to avoid passing to triton constexpr dict
    """
    sanitized = dict(kernel_options)
    backend = cast(_Backend, sanitized.pop("BACKEND", "AUTO"))
    return sanitized, backend


def raise_flex_kernel_options_error(
    kernel_name: str,
    kernel_options: dict[str, Any],
    option_names: Sequence[str],
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
) -> None:
    option_values = ", ".join(f"{name}={kernel_options[name]}" for name in option_names)
    raise ValueError(
        f"Invalid FlexAttention {kernel_name} kernel options: Q and KV block sizes "
        f"must be divisible by the selected tile sizes. Got "
        f"SPARSE_Q_BLOCK_SIZE={sparse_q_block_size}, "
        f"SPARSE_KV_BLOCK_SIZE={sparse_kv_block_size}, and {option_values}. "
        f"Pass compatible values with kernel_options. Available {kernel_name} "
        f"tuning options are {_flex_kernel_tuning_options(kernel_name)}. For example: "
        f"{_flex_kernel_options_example(kernel_name)}. If you did not pin "
        f"these options, and the default choice errors, compiling with "
        f"mode='max-autotune-no-cudagraphs' can also fix this by trying more "
        f"FlexAttention configs."
    )


def _check_flash_supported_scalar_captures(
    score_mod_other_buffers,
    mask_mod_other_buffers,
    *,
    backward: bool = False,
) -> None:
    if has_unsupported_cpu_scalar_tensor_captures(
        score_mod_other_buffers, mask_mod_other_buffers
    ):
        direction = " backward" if backward else ""
        raise RuntimeError(
            f"BACKEND='FLASH' but flash attention{direction} cannot be used: "
            "NYI: score_mod or mask_mod captures a 0-dim CPU tensor scalar. "
            "Workarounds: use BACKEND='TRITON' or pass the value as a tensor "
            "on device instead of capturing a CPU scalar tensor."
        )


@SymbolicGridFn
def flex_attention_grid(batch_size, q_heads, num_queries, d_model, meta, *, cdiv):
    """How is this kernel parallelized?
    We create a grid of (ceil_div(n_queries, query_block_size), batch_size, num_heads)
    Each block is responsible for iterating over blocks of keys and values calculating
    the final attention output.
    """
    return (cdiv(num_queries, meta["BLOCK_M"]), batch_size, q_heads)


def get_float32_precision():
    if (
        (
            torch.backends.cuda.matmul.fp32_precision == "ieee"
            if torch.backends.cuda.matmul.fp32_precision != "none"
            else torch.get_float32_matmul_precision() == "highest"
        )
        or torch.version.hip
        or torch.mtia.is_available()
    ):
        return "'ieee'"
    else:
        return "'tf32'"


flex_attention_template = TritonTemplate(
    name="flex_attention",
    grid=flex_attention_grid,
    source=load_flex_template("flex_attention")
    + load_flex_template("utilities")
    + load_flex_template("common"),
    always_freeze_layout=True,
)


@register_lowering(torch.ops.higher_order.flex_attention, type_promotion_kind=None)
def flex_attention(
    query,
    key,
    value,
    subgraph,
    block_mask,
    scale,
    kernel_options: dict[str, Any],
    score_mod_other_buffers,
    mask_mod_other_buffers,
):
    """The main lowering for the flex_attention hop
    This can currently lower to one of 4 templates:
    1. Base Triton Template
    2. Flex Decode Triton Template
    3. Cpu specific CPP template
    4. MPS specific Metal template
    """
    if query.get_device().type == "cpu":
        return lower_cpu(
            query,
            key,
            value,
            subgraph,
            block_mask,
            scale,
            kernel_options,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        )
    if query.get_device().type == "mps":
        from .flex_mps import lower_mps

        return lower_mps(
            query,
            key,
            value,
            subgraph,
            block_mask,
            scale,
            kernel_options,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        )
    # below is cuda path if device is not cpu or mps
    # tl.dot does not support embedding size less than 16
    small_dqk = V.graph.sizevars.evaluate_expr(sympy.Lt(query.get_size()[-1], 16))
    small_dv = V.graph.sizevars.evaluate_expr(sympy.Lt(value.get_size()[-1], 16))
    if small_dqk or small_dv:
        raise NotImplementedError(
            f"NYI: embedding dimension of the query, key, and value must be "
            f"at least 16 but got E={query.get_size()[-1]} and Ev={value.get_size()[-1]}"
        )

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
        _,  # dq_write_order (backward-only)
        _,  # dq_write_order_full (backward-only)
        _,  # dq_kv_order (backward-only)
        _,  # dq_kv_order_spt (backward-only)
        SPARSE_Q_BLOCK_SIZE,
        SPARSE_KV_BLOCK_SIZE,
        mask_graph,
    ) = block_mask

    kernel_options, backend = _sanitize_kernel_options_for_triton(kernel_options)

    # Early check for FLASH backend: reject scalar captures that cannot be
    # represented by flash-attn aux_scalars before building subgraph buffers.
    if backend == "FLASH":
        _check_flash_supported_scalar_captures(
            score_mod_other_buffers, mask_mod_other_buffers
        )

    if backend == "FLASH":
        score_mod_other_buffers = realize_captures_for_cutedsl(score_mod_other_buffers)
        mask_mod_other_buffers = realize_captures_for_cutedsl(mask_mod_other_buffers)

    placeholder_inps = [
        create_placeholder(name, dtype, query.get_device())
        for name, dtype in [
            ("score", query.get_dtype()),
            ("b", torch.int32),
            ("h", torch.int32),
            ("m", torch.int32),
            ("n", torch.int32),
        ]
    ]
    subgraph_buffer = build_subgraph_buffer(
        placeholder_inps + list(score_mod_other_buffers), subgraph
    )
    freeze_irnodes(subgraph_buffer)

    mask_graph_placeholder_inps = [
        create_placeholder(name, dtype, query.get_device())
        for name, dtype in [
            ("b", torch.int32),
            ("h", torch.int32),
            ("m", torch.int32),
            ("n", torch.int32),
        ]
    ]
    mask_graph_buffer = build_subgraph_buffer(
        mask_graph_placeholder_inps + list(mask_mod_other_buffers), mask_graph
    )
    freeze_irnodes(mask_graph_buffer)
    # Mark symbols in custom kernel options as static shapes and add guards.
    kernel_options = {
        k: V.graph.sizevars.guard_int(v) if isinstance(v, sympy.Symbol) else v
        for k, v in kernel_options.items()
    }
    kernel_options.setdefault("FLOAT32_PRECISION", get_float32_precision())
    enable_gqa = V.graph.sizevars.evaluate_expr(
        sympy.Ne(query.get_size()[1], key.get_size()[1]),
    )

    can_use_decode = _use_flex_decoding(
        query, kv_indices, value, kernel_options, enable_gqa
    )
    use_decode = (backend == "TRITON_DECODE") or (backend == "AUTO" and can_use_decode)

    if backend == "TRITON_DECODE" and not can_use_decode:
        raise RuntimeError(
            "BACKEND='TRITON_DECODE' was specified but flex_decoding cannot be used for this input. "
            "flex_decoding is only available for short sequence lengths with specific configurations."
        )

    if use_decode:
        return create_flex_decoding_kernel(
            query,
            key,
            value,
            block_mask,
            scale,
            kernel_options,
            subgraph_buffer,
            mask_graph_buffer,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        )

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

    if _use_flex_flash_attention(
        subgraph,
        mask_graph,
        kernel_options,
        num_score_mod_placeholders=len(placeholder_inps),
        backend=backend,
    ):
        return create_flex_flash_attention_kernel(
            query,
            key,
            value,
            block_mask,
            scale,
            kernel_options,
            subgraph_buffer,
            mask_graph_buffer,
            score_mod_other_buffers,
            mask_mod_other_buffers,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE,
            mask_graph=mask_graph,
            subgraph=subgraph,
        )

    score_mod_other_buffers = maybe_realize(score_mod_other_buffers)
    mask_mod_other_buffers = maybe_realize(mask_mod_other_buffers)

    freeze_irnodes(score_mod_other_buffers)
    freeze_irnodes(mask_mod_other_buffers)

    Bq, Hq, seq_len_q, qk_head_dim = query.get_size()
    Bkv, Hkv, seq_len_kv, v_head_dim = value.get_size()
    if not V.graph.sizevars.evaluate_expr(sympy.Eq(Bq, Bkv) | sympy.Eq(Bkv, 1)):
        raise AssertionError(
            f"Bq and Bkv must broadcastable. Got Bq={Bq} and Bkv={Bkv}"
        )
    if not V.graph.sizevars.evaluate_expr(sympy.Gt(seq_len_q, 0)):
        raise AssertionError("Query length must be greater than 0")
    if not V.graph.sizevars.evaluate_expr(sympy.Gt(seq_len_kv, 0)):
        raise AssertionError("Key length must be greater than 0")

    B = Bq

    seq_q_divisible = can_skip_boundary_checks(seq_len_q, SPARSE_Q_BLOCK_SIZE)
    seq_kv_divisible = can_skip_boundary_checks(seq_len_kv, SPARSE_KV_BLOCK_SIZE)
    if seq_q_divisible and seq_kv_divisible:
        kernel_options.setdefault("IS_DIVISIBLE", True)
    else:
        kernel_options.setdefault("IS_DIVISIBLE", False)

    # NB it is okay that the v_head_dim is different
    # We are using these to match fill order of the output.
    q_strides = query.get_stride()
    # Construct output layout with strides matching the query.
    out_size = [B, Hq, seq_len_q, v_head_dim]
    out_strides = infer_dense_strides(out_size, q_strides)

    layout = FixedLayout(
        query.get_device(),
        query.get_dtype(),
        [B, Hq, seq_len_q, v_head_dim],
        stride=[sympy.sympify(s) for s in out_strides],
    )
    # see NOTE:[TritonTemplates with multiple outputs]
    logsumexp_shape = [B, Hq, seq_len_q]
    logsumexp = empty_strided(
        logsumexp_shape,
        None,
        dtype=torch.float32,  # The logsumexp is always stored in fp32 regardless of the input dtype
        device=query.get_device(),
    )
    max_scores = empty_strided(
        logsumexp_shape,  # Same shape as logsumexp
        None,
        dtype=torch.float32,  # The max scores are always stored in fp32 regardless of the input dtype
        device=query.get_device(),
    )
    kernel_options.setdefault("SM_SCALE", scale)

    # Determine GQA broadcast factor.
    gqa_shared_heads = FloorDiv(Hq, Hkv)
    kernel_options.setdefault("GQA_SHARED_HEADS", gqa_shared_heads)

    # Inside of Triton kernel, only apply partial masking if partial blocks are computed.
    # full_kv_num_blocks is None if partial blocks are not computed
    has_full_blocks = full_kv_num_blocks is not None
    kernel_options.setdefault("HAS_FULL_BLOCKS", has_full_blocks)
    if not has_full_blocks:
        full_kv_num_blocks, full_kv_indices = (
            empty(0, device=query.get_device()) for _ in range(2)
        )

    set_head_dim_values(kernel_options, qk_head_dim, v_head_dim, V.graph.sizevars)

    choices: list[Any] = []

    dtype = query.get_dtype()
    head_dim = V.graph.sizevars.guard_int(query.get_size()[-1])
    configs: list[FlexConfig] = V.choices.get_flex_attention_fwd_configs(
        head_dim, seq_len_q, dtype, query.get_device().type
    )

    # Mark SPARSE_KV_BLOCK_SIZE & SPARSE_Q_BLOCK_SIZE as static shapes and add guards.
    SPARSE_KV_BLOCK_SIZE = V.graph.sizevars.guard_int(SPARSE_KV_BLOCK_SIZE)
    SPARSE_Q_BLOCK_SIZE = V.graph.sizevars.guard_int(SPARSE_Q_BLOCK_SIZE)

    # Note, we don't need to pass in the captured buffers explicitly
    # because they're implicitly added by the score_mod function
    # We do need to explicitly pass it in for autotuning though.
    original_kernel_options = kernel_options.copy()
    # Default config for warp specialization
    num_consumer_groups, num_buffers_warp_spec = 0, 0
    invalid_block_options: dict[str, Any] | None = None

    for conf in configs:
        cur_kernel_options = original_kernel_options.copy()
        # Performance tuning
        # Triton parameters
        # Remove prefix for forward kernels options and delete backward kernel options.
        for k in list(cur_kernel_options.keys()):
            if k.startswith("fwd_"):
                v = cur_kernel_options.pop(k)
                cur_kernel_options[k[4:]] = v
            if k.startswith("bwd_"):
                cur_kernel_options.pop(k)
        cur_kernel_options.setdefault("num_stages", conf.num_stages)
        cur_kernel_options.setdefault("num_warps", conf.num_warps)
        if cur_kernel_options.get("num_consumer_groups", False):
            cur_kernel_options.setdefault("num_consumer_groups", num_consumer_groups)
            cur_kernel_options.setdefault(
                "num_buffers_warp_spec", num_buffers_warp_spec
            )

        # Intel GPU enables TMA by default
        cur_kernel_options.setdefault("USE_TMA", bool(torch.xpu.is_available()))

        if cur_kernel_options["USE_TMA"] and not can_use_tma(query, key, value):
            cur_kernel_options["USE_TMA"] = False

        # Shrink default tiles to fit smaller pow2 sparse block sizes;
        # user-pinned tiles and non-pow2 sparse sizes still error out below.
        block_m, block_n = conf.block_m, conf.block_n
        if len(configs) == 1 and all(
            is_power_of_2(s) and s >= 16
            for s in (SPARSE_Q_BLOCK_SIZE, SPARSE_KV_BLOCK_SIZE)
        ):
            block_m = min(block_m, SPARSE_Q_BLOCK_SIZE)
            block_n = min(block_n, SPARSE_KV_BLOCK_SIZE)
        cur_kernel_options.setdefault("BLOCK_M", block_m)
        cur_kernel_options.setdefault("BLOCK_N", block_n)
        # Blocksparse options
        cur_kernel_options.setdefault("SPARSE_Q_BLOCK_SIZE", SPARSE_Q_BLOCK_SIZE)
        cur_kernel_options.setdefault("SPARSE_KV_BLOCK_SIZE", SPARSE_KV_BLOCK_SIZE)

        if (
            cur_kernel_options["SPARSE_KV_BLOCK_SIZE"] % cur_kernel_options["BLOCK_N"]
            != 0
            or cur_kernel_options["SPARSE_Q_BLOCK_SIZE"] % cur_kernel_options["BLOCK_M"]
            != 0
        ):
            invalid_block_options = cur_kernel_options
            if len(configs) == 1:
                raise_flex_kernel_options_error(
                    "forward",
                    cur_kernel_options,
                    ("BLOCK_M", "BLOCK_N"),
                    SPARSE_Q_BLOCK_SIZE,
                    SPARSE_KV_BLOCK_SIZE,
                )
            continue

        # ROCm specific kernargs
        for attrib in ["kpack", "matrix_instr_nonkdim", "waves_per_eu"]:
            if hasattr(conf, attrib):
                cur_kernel_options[attrib] = getattr(conf, attrib)

        error = flex_attention_template.maybe_append_choice(
            choices=choices,
            input_nodes=[
                query,
                key,
                value,
                logsumexp,
                max_scores,
                kv_num_blocks,
                kv_indices,
                full_kv_num_blocks,
                full_kv_indices,
            ],
            layout=layout,
            subgraphs=[
                subgraph_buffer,
                mask_graph_buffer,
            ],
            mutated_inputs=[
                logsumexp,
                max_scores,
            ],
            call_sizes=query.get_size(),
            **cur_kernel_options,
        )
        if error is not None and len(configs) == 1:
            raise error

    # Let the active choices handler append any backend-specific flex-attention
    # template choices (e.g. TLX on Blackwell in fbcode). No-op by default.
    choices = V.choices.append_flex_attention_choices(
        choices,
        configs,
        [
            query,
            key,
            value,
            logsumexp,
            max_scores,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
        ],
        [subgraph_buffer, mask_graph_buffer],
        layout,
        original_kernel_options,
        SPARSE_Q_BLOCK_SIZE,
        SPARSE_KV_BLOCK_SIZE,
    )

    if not choices and invalid_block_options is not None:
        raise_flex_kernel_options_error(
            "forward",
            invalid_block_options,
            ("BLOCK_M", "BLOCK_N"),
            SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE,
        )

    inputs_for_autotuning = (
        [
            query,
            key,
            value,
            logsumexp,
            max_scores,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
        ]
        + list(score_mod_other_buffers)
        + list(mask_mod_other_buffers)
    )
    input_gen_fns = {
        5: create_num_blocks_fake_generator(kv_indices),
        6: create_indices_fake,
        7: create_num_blocks_fake_generator(full_kv_indices),
        8: create_indices_fake,
    }

    out, _ = autotune_select_algorithm(
        "flex_attention",
        choices,
        # Autotuning materializes benchmark tensors. Scalar shape captures stay
        # in subgraph_inps below for dependency tracking and codegen.
        [x for x in inputs_for_autotuning if is_tensor_ir_node(x)],
        layout,
        input_gen_fns=input_gen_fns,
    )

    # need subgraph inputs and outputs to analyze all symints used in flex attention
    out.data.data.subgraph_inps = list(score_mod_other_buffers) + list(
        mask_mod_other_buffers
    )
    out.data.data.subgraph_outs = get_fwd_subgraph_outputs(
        subgraph_buffer, mask_graph_buffer
    )

    return (out, logsumexp, max_scores)


# ---------------------------- Backward HOP Implementation ----------------------------


@SymbolicGridFn
def flex_attention_backward_grid(
    batch_size, q_heads, num_queries, d_model, kv_heads, num_key_value, meta, *, cdiv
):
    """How is this kernel parallelized?
    We create a grid of (ceil_div(n_queries, query_block_size) * heads_ratio + ceil_div(n_kv, kv_block_size), batch_size, kv_heads)
    Currently this is only parallelizing over batch* kv_heads, but we can, and want to
    parallelize over ceil_div(q_heads//kv_heads * num_key_value, key_value_block_size).
    To do this will either require atomic updates to some grad values or to have a two pass kernel design.
    """
    return (
        cdiv(num_queries, meta["BLOCK_M2"]) * (q_heads // kv_heads)
        + cdiv(num_key_value, meta["BLOCK_N1"]),
        batch_size,
        kv_heads,
    )


flex_attention_backward_template = TritonTemplate(
    name="flex_attention_backward",
    grid=flex_attention_backward_grid,
    source=load_flex_template("flex_backwards") + load_flex_template("utilities"),
    always_freeze_layout=True,
)


def validate_joint_graph(joint_graph: torch.fx.Graph):
    """We do some pre lowering graph checks in order to raise nicer error messages"""
    for node in joint_graph.nodes:
        if (
            node.op == "call_function"
            and node.target is torch.ops.flex_lib.zeros_and_scatter.default
        ):
            for user in node.users:
                if user.op != "output":
                    raise NotImplementedError(
                        "Using multiple indexing operations on the same tensor that requires gradients "
                        "in a score_mod function is not currently supported. "
                        "This typically happens when indexing the same tensor multiple times, like:\n\n"
                        "    def score_mod(score, b, h, q_idx, kv_idx):\n"
                        "        return score + bias[q_idx] + bias[kv_idx]  # bias used twice!\n\n"
                        "A valid workaround is to clone() the tensors that will be indexed multiple times. For example:\n\n"
                        "    bias1 = bias.clone()\n"
                        "    def score_mod(score, b, h, q_idx, kv_idx):\n"
                        "        return score + bias[q_idx] + bias1[kv_idx]\n\n"
                        "Note that this solution will use additional memory."
                    )
    return


@dataclass(frozen=True)
class JointOutputResult:
    """Results from processing joint outputs."""

    grad_input: ComputedBuffer
    captured_grads_compute: list[ComputedBuffer]
    captured_grads: list[TensorBox | None]
    mutated_grads: list[TensorBox]


def process_joint_outputs(
    all_joint_outputs: SubgraphResults, num_placeholders: int
) -> JointOutputResult:
    """Process joint outputs and extract various buffers needed for lowering

    Args:
        all_joint_outputs: List of all the outputs from build_subgraphs
        num_placeholders: The number of placeholder inputs, used to skip over unused backward compute buffers

    Returns:
        JointOutputResult containing processed buffers and gradients
    """
    if not isinstance(all_joint_outputs, list):
        raise AssertionError(f"Expected list, got {type(all_joint_outputs)}")
    if all_joint_outputs[0] is None:
        raise AssertionError("joint_subgraph_buffer is None - this is a bug!")

    joint_buffer = all_joint_outputs[0]
    other_grads = all_joint_outputs[num_placeholders - 1 :]

    # outer_grads has the structure: Len(other_buffer_grads) if buffer doesn't require grad than it will be None
    # We only grab the buffers that require grad for inlining into kernel
    grads_compute = [buf for buf in other_grads if buf is not None]

    def get_out(buf):
        if buf is None:
            return None
        if not isinstance(buf, ComputedBuffer):
            raise AssertionError(f"Expected ComputedBuffer, got {type(buf)}")
        if buf.name is None:
            raise AssertionError("ComputedBuffer name must not be None")
        return TensorBox.create(V.graph.get_buffer(buf.name))

    grads_out = [get_out(x) for x in other_grads]
    mutated_grads = [buf for buf in grads_out if buf is not None]

    return JointOutputResult(
        grad_input=joint_buffer,
        captured_grads_compute=grads_compute,
        captured_grads=grads_out,
        mutated_grads=mutated_grads,
    )


# TODO: We probably also need a layout constraint?
@register_lowering(
    torch.ops.higher_order.flex_attention_backward, type_promotion_kind=None
)
def flex_attention_backward(*args, **kwargs):
    """Lowering for the flex_attention_backward op in triton"""
    (
        query,
        key,
        value,
        out,
        logsumexp,
        grad_out,
        grad_logsumexp,
        fw_graph,
        joint_graph,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    ) = args
    if query.get_device().type == "mps":
        raise NotImplementedError(
            "flex_attention backward is not yet supported on MPS. "
            "Use torch.no_grad() for inference."
        )
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
        dq_write_order,
        dq_write_order_full,
        dq_kv_order,
        dq_kv_order_spt,
        SPARSE_Q_BLOCK_SIZE,
        SPARSE_KV_BLOCK_SIZE,
        mask_graph,
    ) = block_mask

    (
        query,
        key,
        value,
        logsumexp,
        grad_out,
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        q_num_blocks,
        q_indices,
        full_q_num_blocks,
        full_q_indices,
        dq_write_order,
        dq_write_order_full,
        dq_kv_order,
    ) = maybe_realize(
        [
            query,
            key,
            value,
            logsumexp,
            grad_out,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            q_num_blocks,
            q_indices,
            full_q_num_blocks,
            full_q_indices,
            dq_write_order,
            dq_write_order_full,
            dq_kv_order,
        ]
    )

    device = query.get_device()
    dtype = query.get_dtype()
    Bq, Hq, seq_len_q, qk_head_dim = query.get_size()
    Bkv, Hkv, seq_len_kv, v_head_dim = value.get_size()

    if not V.graph.sizevars.evaluate_expr(sympy.Eq(Bq, Bkv) | sympy.Eq(Bkv, 1)):
        raise AssertionError(
            f"Bq and Bkv must broadcastable. Got Bq={Bq} and Bkv={Bkv}"
        )

    kernel_options, backend = _sanitize_kernel_options_for_triton(kernel_options)
    if backend == "FLASH":
        _check_flash_supported_scalar_captures(
            score_mod_other_buffers, mask_mod_other_buffers, backward=True
        )
        score_mod_other_buffers = realize_captures_for_cutedsl(score_mod_other_buffers)
        mask_mod_other_buffers = realize_captures_for_cutedsl(mask_mod_other_buffers)

    # Add check for mixed dtypes
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError(
            f"Backward pass with mixed query, key, and value dtype is not supported, "
            f"got query.dtype={query.dtype}, key.dtype={key.dtype}, "
            f"and value.dtype={value.dtype}"
        )
    # Mark symbols in custom kernel options as static shapes and add guards.
    kernel_options = {
        k: V.graph.sizevars.guard_int(v) if isinstance(v, sympy.Symbol) else v
        for k, v in kernel_options.items()
    }
    kernel_options.setdefault("FLOAT32_PRECISION", get_float32_precision())
    kernel_options.setdefault("PRESCALE_QK", False)
    kernel_options.setdefault("ROWS_GUARANTEED_SAFE", False)
    kernel_options.setdefault("BLOCKS_ARE_CONTIGUOUS", False)
    kernel_options.setdefault("WRITE_DQ", True)
    seq_q_divisible = can_skip_boundary_checks(seq_len_q, SPARSE_Q_BLOCK_SIZE)
    seq_kv_divisible = can_skip_boundary_checks(seq_len_kv, SPARSE_KV_BLOCK_SIZE)
    if seq_q_divisible and seq_kv_divisible:
        kernel_options.setdefault("IS_DIVISIBLE", True)
    else:
        kernel_options.setdefault("IS_DIVISIBLE", False)

    fwd_placeholder_inps = [
        create_placeholder(name, dtype, device)
        for name, dtype in [
            ("score", dtype),
            ("b", torch.int32),
            ("h", torch.int32),
            ("m", torch.int32),
            ("n", torch.int32),
        ]
    ]
    fw_subgraph_buffer = build_subgraph_buffer(
        fwd_placeholder_inps + list(score_mod_other_buffers), fw_graph
    )
    freeze_irnodes(fw_subgraph_buffer)

    joint_placeholder_inps = fwd_placeholder_inps + [
        create_placeholder("grad_score_mod", dtype, device)
    ]
    # Sometimes we have weird unused nodes here
    joint_graph.graph_module.graph.eliminate_dead_code()

    # It is hard to raise nice errors for some joint graphs during subgraph lowering
    # This lets us do some checks before attempting to lower
    validate_joint_graph(joint_graph.graph_module.graph)

    all_joint_outputs = build_subgraph_buffer(
        joint_placeholder_inps + list(score_mod_other_buffers),
        joint_graph,
    )

    freeze_irnodes(all_joint_outputs)

    joint_outputs = process_joint_outputs(
        all_joint_outputs, len(joint_placeholder_inps)
    )

    mask_graph_placeholder_inps = [
        create_placeholder(name, dtype, query.get_device())
        for name, dtype in [
            ("b", torch.int32),
            ("h", torch.int32),
            ("m", torch.int32),
            ("n", torch.int32),
        ]
    ]
    mask_graph_buffer = build_subgraph_buffer(
        mask_graph_placeholder_inps + list(mask_mod_other_buffers), mask_graph
    )
    freeze_irnodes(mask_graph_buffer)

    if _use_flex_flash_attention_backward(
        fw_graph,
        mask_graph,
        backend=backend,
        joint_outputs=joint_outputs,
        score_mod_other_buffers=score_mod_other_buffers,
    ):
        needs_block_mask = not is_trivial_mask_graph(mask_graph.graph_module)
        if grad_logsumexp is not None:
            (grad_logsumexp,) = maybe_realize([grad_logsumexp])

        score_is_trivial = is_trivial_score_graph(fw_graph.graph_module)
        return create_flex_flash_attention_backward_kernel(
            query,
            key,
            value,
            out,
            logsumexp,
            grad_out,
            grad_logsumexp,
            scale,
            kernel_options,
            SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE,
            fw_subgraph_buffer=None if score_is_trivial else fw_subgraph_buffer,
            joint_subgraph_buffer=None
            if score_is_trivial
            else joint_outputs.grad_input,
            score_mod_other_buffers=list(score_mod_other_buffers),
            mask_graph_buffer=mask_graph_buffer if needs_block_mask else None,
            mask_mod_other_buffers=list(mask_mod_other_buffers),
            q_num_blocks=q_num_blocks if needs_block_mask else None,
            q_indices=q_indices if needs_block_mask else None,
            full_q_num_blocks=full_q_num_blocks if needs_block_mask else None,
            full_q_indices=full_q_indices if needs_block_mask else None,
            dq_write_order=dq_write_order if needs_block_mask else None,
            dq_write_order_full=dq_write_order_full if needs_block_mask else None,
            dq_kv_order=dq_kv_order if needs_block_mask else None,
            dq_kv_order_spt=dq_kv_order_spt,
        )

    # Construct layout with stride order matching K
    key_size = [Bq, Hkv, seq_len_kv, qk_head_dim]
    key_strides = infer_dense_strides(key_size, key.get_stride())

    layout_broadcasted_k = FixedLayout(
        key.get_device(),
        key.get_dtype(),
        key_size,
        stride=[sympy.sympify(s) for s in key_strides],
    )

    # Create delta which will is needed for the bwd's kernel
    mul_delta = lowerings[aten.mul](out, grad_out)
    delta = lowerings[aten.sum](mul_delta, axis=-1)
    delta = lowerings[prims.convert_element_type](delta, torch.float32)
    if grad_logsumexp is not None:
        grad_lse_exp2 = lowerings[aten.mul](grad_logsumexp, 1 / math.log(2))
        grad_lse_exp2 = ExternKernel.require_contiguous(grad_lse_exp2)
        delta = lowerings[aten.sub](delta, grad_lse_exp2)
        delta = ExternKernel.require_contiguous(delta)
        delta, grad_lse_exp2 = maybe_realize([delta, grad_lse_exp2])
    else:
        delta = ExternKernel.require_contiguous(delta)
        (delta,) = maybe_realize([delta])

    # # see NOTE:[TritonTemplates with multiple outputs]
    query_size = [Bq, Hq, seq_len_q, qk_head_dim]
    grad_query_strides = infer_dense_strides(query_size, query.get_stride())
    grad_query = empty_strided(
        query_size,
        stride=[sympy.sympify(s) for s in grad_query_strides],
        dtype=query.get_dtype(),
        device=query.get_device(),
    )

    # Construct output layout with stride order matching value
    value_size = [Bq, Hkv, seq_len_kv, v_head_dim]
    value_strides = infer_dense_strides(value_size, value.get_stride())

    broadcasted_grad_value = empty_strided(
        value_size,
        stride=[sympy.sympify(s) for s in value_strides],
        dtype=value.get_dtype(),
        device=value.get_device(),
    )

    kernel_options.setdefault("SM_SCALE", scale)

    # Determine GQA factor
    gqa_shared_heads = FloorDiv(Hq, Hkv)
    kernel_options.setdefault("GQA_SHARED_HEADS", gqa_shared_heads)

    # Inside of Triton kernel, only apply partial masking if partial blocks are computed.
    # full_kv_num_blocks is torch.zeros([1, 1, 1]) if partial blocks are not computed.
    has_full_blocks = full_kv_num_blocks is not None
    kernel_options.setdefault("HAS_FULL_BLOCKS", has_full_blocks)
    if not has_full_blocks:
        full_kv_num_blocks, full_kv_indices, full_q_num_blocks, full_q_indices = (
            empty(0, device=query.get_device()) for _ in range(4)
        )

    set_head_dim_values(kernel_options, qk_head_dim, v_head_dim, V.graph.sizevars)

    SPARSE_Q_BLOCK_SIZE = V.graph.sizevars.guard_int(SPARSE_Q_BLOCK_SIZE)
    SPARSE_KV_BLOCK_SIZE = V.graph.sizevars.guard_int(SPARSE_KV_BLOCK_SIZE)

    choices: list[Any] = []

    dtype = query.get_dtype()
    head_dim = V.graph.sizevars.guard_int(query.get_size()[-1])
    configs: list[FlexBwDConfig] = V.choices.get_flex_attention_bwd_configs(
        head_dim, dtype, query.get_device().type
    )

    # Default config for warp specialization
    num_consumer_groups, num_buffers_warp_spec = 0, 0
    invalid_block_options: dict[str, Any] | None = None

    original_kernel_options = kernel_options.copy()

    for conf in configs:
        # Performance tuning
        # Triton heuristics
        cur_kernel_options = original_kernel_options.copy()
        # Remove prefix for backward kernels options and delete forward kernel options.
        for k in list(cur_kernel_options.keys()):
            if k.startswith("bwd_"):
                v = cur_kernel_options.pop(k)
                cur_kernel_options[k[4:]] = v
            if k.startswith("fwd_"):
                cur_kernel_options.pop(k)
        cur_kernel_options.setdefault("num_warps", conf.num_warps)
        cur_kernel_options.setdefault("num_stages", conf.num_stages)

        if cur_kernel_options.get("num_consumer_groups", False):
            cur_kernel_options.setdefault("num_consumer_groups", num_consumer_groups)
            cur_kernel_options.setdefault(
                "num_buffers_warp_spec", num_buffers_warp_spec
            )

        # Intel GPU enables TMA by default
        cur_kernel_options.setdefault("USE_TMA", bool(torch.xpu.is_available()))

        if cur_kernel_options["USE_TMA"] and not can_use_tma(query, key, value):
            cur_kernel_options["USE_TMA"] = False

        # Shrink default tiles to fit smaller pow2 sparse block sizes;
        # user-pinned tiles and non-pow2 sparse sizes still error out below.
        block_m1, block_n1 = conf.block_m1, conf.block_n1
        block_m2, block_n2 = conf.block_m2, conf.block_n2
        if len(configs) == 1 and all(
            is_power_of_2(s) and s >= 16
            for s in (SPARSE_Q_BLOCK_SIZE, SPARSE_KV_BLOCK_SIZE)
        ):
            block_m1 = min(block_m1, SPARSE_Q_BLOCK_SIZE)
            block_n1 = min(block_n1, SPARSE_KV_BLOCK_SIZE)
            block_m2 = min(block_m2, SPARSE_Q_BLOCK_SIZE)
            block_n2 = min(block_n2, SPARSE_KV_BLOCK_SIZE)
            # Kernel static asserts: BLOCK_N1 % BLOCK_M1 == BLOCK_M2 % BLOCK_N2 == 0
            block_m1 = min(block_m1, block_n1)
            block_n2 = min(block_n2, block_m2)
        cur_kernel_options.setdefault("BLOCK_M1", block_m1)
        cur_kernel_options.setdefault("BLOCK_N1", block_n1)
        cur_kernel_options.setdefault("BLOCK_M2", block_m2)
        cur_kernel_options.setdefault("BLOCK_N2", block_n2)

        # Blocksparse options
        cur_kernel_options.setdefault("SPARSE_Q_BLOCK_SIZE", SPARSE_Q_BLOCK_SIZE)
        cur_kernel_options.setdefault("SPARSE_KV_BLOCK_SIZE", SPARSE_KV_BLOCK_SIZE)

        if (
            cur_kernel_options["SPARSE_KV_BLOCK_SIZE"] % cur_kernel_options["BLOCK_N1"]
            != 0
            or cur_kernel_options["SPARSE_Q_BLOCK_SIZE"]
            % cur_kernel_options["BLOCK_M1"]
            != 0
            or cur_kernel_options["SPARSE_KV_BLOCK_SIZE"]
            % cur_kernel_options["BLOCK_N2"]
            != 0
            or cur_kernel_options["SPARSE_Q_BLOCK_SIZE"]
            % cur_kernel_options["BLOCK_M2"]
            != 0
            # Kernel static asserts; validate here for a friendly error.
            or cur_kernel_options["BLOCK_N1"] % cur_kernel_options["BLOCK_M1"] != 0
            or cur_kernel_options["BLOCK_M2"] % cur_kernel_options["BLOCK_N2"] != 0
        ):
            invalid_block_options = cur_kernel_options
            if len(configs) == 1:
                raise_flex_kernel_options_error(
                    "backward",
                    cur_kernel_options,
                    ("BLOCK_M1", "BLOCK_N1", "BLOCK_M2", "BLOCK_N2"),
                    SPARSE_Q_BLOCK_SIZE,
                    SPARSE_KV_BLOCK_SIZE,
                )
            continue

        # ROCm specific kernargs
        for attrib in ["kpack", "matrix_instr_nonkdim", "waves_per_eu"]:
            if hasattr(conf, attrib):
                cur_kernel_options[attrib] = getattr(conf, attrib)

        flex_attention_backward_template.maybe_append_choice(
            choices=choices,
            input_nodes=[
                query,
                key,
                value,
                logsumexp,
                delta,
                grad_out,
                grad_query,
                broadcasted_grad_value,
                kv_num_blocks,
                kv_indices,
                q_num_blocks,
                q_indices,
                full_kv_num_blocks,
                full_kv_indices,
                full_q_num_blocks,
                full_q_indices,
            ],
            layout=layout_broadcasted_k,  # We use store_output only for grad_key
            subgraphs=[
                fw_subgraph_buffer,
                joint_outputs.grad_input,
                mask_graph_buffer,
                joint_outputs.captured_grads_compute,
            ],
            mutated_inputs=[
                grad_query,
                broadcasted_grad_value,
                *joint_outputs.mutated_grads,
            ],
            call_sizes=query.get_size() + key.get_size()[1:3],
            **cur_kernel_options,
        )

    if not choices and invalid_block_options is not None:
        raise_flex_kernel_options_error(
            "backward",
            invalid_block_options,
            ("BLOCK_M1", "BLOCK_N1", "BLOCK_M2", "BLOCK_N2"),
            SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE,
        )

    score_mod_other_buffers = maybe_realize(score_mod_other_buffers)
    mask_mod_other_buffers = maybe_realize(mask_mod_other_buffers)

    inputs_for_autotuning = (
        [
            query,
            key,
            value,
            logsumexp,
            delta,
            grad_out,
            grad_query,
            broadcasted_grad_value,
            kv_num_blocks,
            kv_indices,
            q_num_blocks,
            q_indices,
            full_kv_num_blocks,
            full_kv_indices,
            full_q_num_blocks,
            full_q_indices,
        ]
        + list(score_mod_other_buffers)
        + list(mask_mod_other_buffers)
        + joint_outputs.mutated_grads
    )
    input_gen_fns = {
        8: create_num_blocks_fake_generator(kv_indices),  # kv_num_blocks
        9: create_indices_fake,
        10: create_num_blocks_fake_generator(q_indices),  # q_num_blocks
        11: create_indices_fake,
        12: create_num_blocks_fake_generator(full_kv_indices),  # full_kv_num_blocks
        13: create_indices_fake,
        14: create_num_blocks_fake_generator(full_q_indices),  # full_q_num_blocks
        15: create_indices_fake,
    }

    broadcasted_grad_key, _ = autotune_select_algorithm(
        "flex_attention_backward",
        choices,
        # Autotuning materializes benchmark tensors. Scalar shape captures stay
        # in subgraph_inps below for dependency tracking and codegen.
        [x for x in inputs_for_autotuning if is_tensor_ir_node(x)],
        layout_broadcasted_k,
        input_gen_fns=input_gen_fns,
    )  # [Bq, Hkv, seq_len_kv, k_head_dim]

    # need subgraph inputs and outputs to analyze all symints used in flex attention
    broadcasted_grad_key.data.data.subgraph_inps = list(score_mod_other_buffers) + list(
        mask_mod_other_buffers
    )
    broadcasted_grad_key.data.data.subgraph_outs = get_bwd_subgraph_outputs(
        fw_subgraph_buffer, mask_graph_buffer, joint_outputs
    )

    if V.graph.sizevars.evaluate_expr(sympy.Eq(Bq, Bkv)):
        grad_key = broadcasted_grad_key
        grad_value = broadcasted_grad_value
    else:
        if not V.graph.sizevars.evaluate_expr(sympy.Gt(Bq, 1) & sympy.Eq(Bkv, 1)):
            raise AssertionError(
                f"Bq and Bkv must broadcastable. "
                f"Got Bq={V.graph.sizevars.evaluate_expr(Bq)} "
                f"and Bkv={V.graph.sizevars.evaluate_expr(Bkv)}"
            )
        grad_key = lowerings[aten.sum](broadcasted_grad_key, axis=0, keepdims=True)
        grad_value = lowerings[aten.sum](broadcasted_grad_value, axis=0, keepdims=True)

    # Cast captured grads to match original buffer dtypes. Gradients are accumulated
    # in fp32 for precision, then cast to the original dtype (e.g., bf16) here.
    captured_grads = tuple(
        to_dtype(g, orig.get_dtype())
        if g is not None and g.get_dtype() != orig.get_dtype()
        else g
        for g, orig in zip(joint_outputs.captured_grads, score_mod_other_buffers)
    )

    return (grad_query, grad_key, grad_value, captured_grads)


def get_bwd_subgraph_outputs(
    subgraph_buffer: SubgraphResults,
    mask_graph_buffer: SubgraphResults,
    joint_outputs: JointOutputResult,
) -> list[ComputedBuffer | TensorBox | None]:
    subgraph_buffer = (
        subgraph_buffer if isinstance(subgraph_buffer, Sequence) else [subgraph_buffer]
    )
    mask_graph_buffer = (
        mask_graph_buffer
        if isinstance(mask_graph_buffer, Sequence)
        else [mask_graph_buffer]
    )
    joint_output_buffers = [
        joint_outputs.grad_input,
        *joint_outputs.captured_grads_compute,
        *joint_outputs.captured_grads,
        *joint_outputs.mutated_grads,
    ]

    return [*subgraph_buffer, *mask_graph_buffer, *joint_output_buffers]
<|endoftext|># mypy: allow-untyped-defs
"""Captured aux tensor vectorization analysis for flex attention."""

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from enum import auto, Enum

import sympy

import torch
from torch.fx import GraphModule
from torch.utils._sympy.functions import FloorDiv, Max, Min, ModularIndexing

from ...codegen.cutedsl.lane_analysis import classify_lane_expr
from ...ir import TensorBox
from ...virtualized import V


# Match one vectorized mask_mod call to one 32-bit R2P keep mask; on Blackwell,
# vec32 aux loads are close to vec16.
DEFAULT_MASK_MOD_VEC_SIZE = 32


class LoadKind(Enum):
    """How a captured aux tensor load behaves across vectorized KV lanes.

    GATHER: Cannot use a direct vector load.
    LANE_UNIFORM: The same value is used for every KV lane.
    CONTIGUOUS: Consecutive KV lanes map to contiguous storage.
    """

    GATHER = auto()
    LANE_UNIFORM = auto()
    CONTIGUOUS = auto()


@dataclasses.dataclass(frozen=True)
class AuxLoadVecInfo:
    """Vectorization classification for one captured aux tensor load.

    kind: Load classification.
    vec_size: Direct vector width for contiguous loads; None otherwise.
    """

    kind: LoadKind
    vec_size: int | None = None

    def __post_init__(self) -> None:
        """Validate that only contiguous loads carry a vector width."""
        if self.kind is LoadKind.CONTIGUOUS:
            if self.vec_size is None:
                raise AssertionError("CONTIGUOUS load requires a vec_size")
        else:
            if self.vec_size is not None:
                raise AssertionError(
                    f"non-CONTIGUOUS load must not carry vec_size, got {self.vec_size}"
                )

    @classmethod
    def gather(cls) -> "AuxLoadVecInfo":
        """Create a gather-load classification."""
        return cls(LoadKind.GATHER)

    @classmethod
    def lane_uniform(cls) -> "AuxLoadVecInfo":
        """Create a lane-uniform-load classification."""
        return cls(LoadKind.LANE_UNIFORM)

    @classmethod
    def contiguous(cls, vec_size: int) -> "AuxLoadVecInfo":
        """Create a contiguous direct-vector-load classification."""
        return cls(LoadKind.CONTIGUOUS, vec_size)


@dataclasses.dataclass(frozen=True)
class AuxVecPolicy:
    """ABI-specific rules for selecting an aux tensor vector width.

    q_idx_placeholder: Placeholder index for the query lane.
    kv_idx_placeholder: Placeholder index for the vectorized KV lane.
    max_vec_size: Largest direct captured-tensor load width to consider.
    min_index_rank_for_contiguous_load: Minimum indexed rank for direct vector loads.
        mask_mod uses rank 2 so 1-D KV masks remain scalar per-lane gathers;
        otherwise a short 1-D mask can shrink the packed-mask vector width.
    non_lane_placeholder_start: First non-lane placeholder participating in indices.
    """

    q_idx_placeholder: int
    kv_idx_placeholder: int
    max_vec_size: int
    min_index_rank_for_contiguous_load: int = 1
    non_lane_placeholder_start: int = 0


MASK_MOD_AUX_VEC_POLICY = AuxVecPolicy(
    q_idx_placeholder=2,
    kv_idx_placeholder=3,
    max_vec_size=32,
    min_index_rank_for_contiguous_load=2,
)
SCORE_MOD_AUX_VEC_POLICY = AuxVecPolicy(
    q_idx_placeholder=3,
    kv_idx_placeholder=4,
    max_vec_size=8,
    non_lane_placeholder_start=1,
)


@dataclasses.dataclass(frozen=True)
class AuxIndexedTensor:
    """Captured tensor plus any safe partial indices already accumulated."""

    buffer: TensorBox
    indices: tuple[object, ...]


def make_fx_index_symbols(
    q_idx_node: torch.fx.Node,
    kv_idx_node: torch.fx.Node,
    non_lane_index_nodes: Sequence[torch.fx.Node] = (),
    *,
    kv_expr: sympy.Expr | None = None,
) -> tuple[sympy.Symbol, sympy.Symbol, dict[torch.fx.Node, sympy.Expr]]:
    """Build symbolic replacements for lane and non-lane FX index nodes."""
    q_idx = sympy.Symbol("q_idx", integer=True, nonnegative=True)
    kv_idx = sympy.Symbol("kv_idx", integer=True, nonnegative=True)
    index_symbols = {
        node: sympy.Symbol(node.name, integer=True, nonnegative=True)
        for node in non_lane_index_nodes
    }
    index_symbols[q_idx_node] = q_idx
    index_symbols[kv_idx_node] = kv_idx if kv_expr is None else kv_expr
    return q_idx, kv_idx, index_symbols


def select_mask_mod_vec_size(
    *,
    has_mask_mod: bool,
    has_mask_aux_tensors: bool,
    supports_mask_mod_vec: bool,
    graph_module: GraphModule | None,
    other_buffers: Sequence[TensorBox],
) -> int | None:
    """Use mask_mod's (b, h, q, kv) ABI; vec32 matches the packed R2P mask width."""
    if not has_mask_mod or not supports_mask_mod_vec:
        return None
    if not has_mask_aux_tensors:
        return DEFAULT_MASK_MOD_VEC_SIZE

    vec_size = select_aux_mod_vec_size(
        graph_module,
        other_buffers,
        MASK_MOD_AUX_VEC_POLICY,
    )
    return vec_size if vec_size > 1 else None


def select_score_mod_vec_size(
    *,
    has_score_mod: bool,
    has_aux_tensors: bool,
    is_sm100_or_later: bool,
    graph_module: GraphModule | None,
    other_buffers: Sequence[TensorBox],
) -> int | None:
    """Select score_mod vector width for captured aux loads.

    A return value of None means there are no captured tensor loads constraining
    score_mod.__vec_size__. For captured tensors, direct vector loads are capped
    to the SM100 CuTe score_mod aux-load path's supported width of 8.
    """
    if not has_score_mod or not has_aux_tensors:
        return None
    if not is_sm100_or_later:
        return 1
    return select_aux_mod_vec_size(
        graph_module,
        other_buffers,
        SCORE_MOD_AUX_VEC_POLICY,
    )


def select_aux_mod_vec_size(
    graph_module: GraphModule | None,
    other_buffers: Sequence[TensorBox],
    policy: AuxVecPolicy,
) -> int:
    """Choose a safe vector width for captured tensor loads.

    The analysis follows aten.index chains from captured tensor placeholders,
    accumulates partial indices that do not depend on the vectorized KV lane,
    classifies complete loads, and picks the narrowest required contiguous
    width. Gather-like loads stay scalar per lane and can coexist with any
    vectorizable load that selects a non-scalar width.
    """
    if graph_module is None:
        return 1

    placeholders = [
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    ]
    num_fixed_placeholders = policy.kv_idx_placeholder + 1
    if len(placeholders) < num_fixed_placeholders:
        return 1

    aux_indexed_tensors = {
        placeholder: AuxIndexedTensor(buffer, ())
        for placeholder, buffer in zip(
            placeholders[num_fixed_placeholders:], other_buffers
        )
    }
    non_lane_index_nodes = placeholders[
        policy.non_lane_placeholder_start : policy.q_idx_placeholder
    ]
    selected_vec_size = policy.max_vec_size
    found_vectorizable_load = False
    for node in graph_module.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.index.Tensor:
            continue
        buffer_node, indices = node.args
        if buffer_node not in aux_indexed_tensors:
            continue
        indexed_tensor = aux_indexed_tensors[buffer_node]
        if not isinstance(indices, (list, tuple)) or not indices:
            continue
        full_indices = indexed_tensor.indices + tuple(indices)
        rank = len(indexed_tensor.buffer.get_size())
        if len(full_indices) < rank:
            # Chained indexing is safe to track only until the KV lane is used.
            if is_safe_partial_aux_index(
                full_indices,
                placeholders[policy.q_idx_placeholder],
                placeholders[policy.kv_idx_placeholder],
                non_lane_index_nodes,
            ):
                aux_indexed_tensors[node] = AuxIndexedTensor(
                    indexed_tensor.buffer, full_indices
                )
            continue
        if len(full_indices) > rank:
            continue
        aux_load_vec_info = direct_aux_load_vec_size_and_kind(
            full_indices,
            indexed_tensor.buffer,
            placeholders[policy.q_idx_placeholder],
            placeholders[policy.kv_idx_placeholder],
            non_lane_index_nodes=non_lane_index_nodes,
            max_vec_size=policy.max_vec_size,
            min_index_rank_for_contiguous_load=policy.min_index_rank_for_contiguous_load,
        )
        match aux_load_vec_info.kind:
            case LoadKind.LANE_UNIFORM:
                found_vectorizable_load = True
            case LoadKind.GATHER:
                pass
            case LoadKind.CONTIGUOUS:
                contiguous_vec_size = aux_load_vec_info.vec_size
                if contiguous_vec_size is None:
                    raise AssertionError("CONTIGUOUS load must have a vec_size")
                selected_vec_size = min(selected_vec_size, contiguous_vec_size)
                found_vectorizable_load = True

    return selected_vec_size if found_vectorizable_load else 1


def is_safe_partial_aux_index(
    indices: tuple[object, ...],
    q_idx_node: torch.fx.Node,
    kv_idx_node: torch.fx.Node,
    non_lane_index_nodes: Sequence[torch.fx.Node],
) -> bool:
    """Return whether partial indices can be chained before the KV-lane index."""
    _, kv_idx, index_symbols = make_fx_index_symbols(
        q_idx_node, kv_idx_node, non_lane_index_nodes
    )
    for index in indices:
        expr = fx_aux_index_to_sympy(index, index_symbols)
        if expr is None or kv_idx in expr.free_symbols:
            return False
    return True


def direct_aux_load_vec_size_and_kind(
    indices: object,
    buffer: TensorBox,
    q_idx_node: torch.fx.Node,
    kv_idx_node: torch.fx.Node,
    non_lane_index_nodes: Sequence[torch.fx.Node] = (),
    max_vec_size: int = 8,
    min_index_rank_for_contiguous_load: int = 1,
) -> AuxLoadVecInfo:
    """Return vector-load information for a captured aux load.

    The load is lane-uniform if none of its indices depend on KV. Otherwise,
    only the final indexed dimension may depend on KV, the storage for that
    dimension must be contiguous and vector-aligned, and the lane expression
    must enumerate consecutive values for the chosen vector width.
    """
    if not isinstance(indices, (list, tuple)) or not indices:
        return AuxLoadVecInfo.gather()
    if not (max_vec_size >= 2 and max_vec_size.bit_count() == 1):
        raise AssertionError(
            f"max_vec_size must be a power of two >= 2, got {max_vec_size}"
        )

    _, kv_idx, index_symbols = make_fx_index_symbols(
        q_idx_node, kv_idx_node, non_lane_index_nodes
    )
    index_exprs = [fx_aux_index_to_sympy(index, index_symbols) for index in indices]
    if any(expr is None for expr in index_exprs):
        return AuxLoadVecInfo.gather()
    if all(kv_idx not in expr.free_symbols for expr in index_exprs):
        return AuxLoadVecInfo.lane_uniform()

    last_expr = index_exprs[-1]
    if kv_idx not in last_expr.free_symbols:
        return AuxLoadVecInfo.gather()
    if len(indices) < min_index_rank_for_contiguous_load:
        return AuxLoadVecInfo.gather()

    prefix_exprs = index_exprs[:-1]
    if any(kv_idx in expr.free_symbols for expr in prefix_exprs):
        return AuxLoadVecInfo.gather()

    sizes = buffer.get_size()
    strides = buffer.get_stride()
    if not V.graph.sizevars.statically_known_equals(strides[-1], 1):
        return AuxLoadVecInfo.gather()

    # FlashAttention groups vectorized mod calls across consecutive KV lanes.
    offset = buffer.get_layout().offset
    vec_size = max_vec_size
    while vec_size >= 2:
        lane_info = classify_lane_expr(last_expr, kv_idx, max_width=vec_size)
        if (
            V.graph.sizevars.statically_known_multiple_of(sizes[-1], vec_size)
            and V.graph.sizevars.statically_known_multiple_of(offset, vec_size)
            and all(
                V.graph.sizevars.statically_known_multiple_of(stride, vec_size)
                for stride in strides[:-1]
            )
            and lane_info.is_contiguous
        ):
            return AuxLoadVecInfo.contiguous(vec_size)
        vec_size //= 2
    return AuxLoadVecInfo.gather()


def fx_aux_index_to_sympy(
    index: object,
    index_symbols: Mapping[torch.fx.Node, sympy.Expr],
    node_to_sympy: Callable[[torch.fx.Node], sympy.Expr | None] | None = None,
) -> sympy.Expr | None:
    """Translate the supported FX integer index operations to sympy."""
    if isinstance(index, int | sympy.Integer):
        return sympy.Integer(index)
    if not isinstance(index, torch.fx.Node):
        return None
    if index in index_symbols:
        return index_symbols[index]
    if node_to_sympy is not None:
        expr = node_to_sympy(index)
        if expr is not None:
            return expr
    if index.op != "call_function":
        return None

    args = index.args
    target = index.target
    match target:
        case torch.ops.aten.abs.default:
            operand = fx_aux_index_to_sympy(args[0], index_symbols, node_to_sympy)
            return None if operand is None else sympy.Abs(operand)
        case torch.ops.aten.neg.default:
            operand = fx_aux_index_to_sympy(args[0], index_symbols, node_to_sympy)
            return None if operand is None else -operand
        case torch.ops.aten.clamp.default:
            operand = fx_aux_index_to_sympy(args[0], index_symbols, node_to_sympy)
            if operand is None:
                return None
            lo = args[1] if len(args) > 1 else index.kwargs.get("min")
            hi = args[2] if len(args) > 2 else index.kwargs.get("max")
            for bound, combine in ((lo, Max), (hi, Min)):
                if bound is not None:
                    bound_expr = fx_aux_index_to_sympy(
                        bound, index_symbols, node_to_sympy
                    )
                    if bound_expr is None:
                        return None
                    operand = combine(operand, bound_expr)
            return operand

    if len(args) < 2:
        return None
    lhs = fx_aux_index_to_sympy(args[0], index_symbols, node_to_sympy)
    rhs = fx_aux_index_to_sympy(args[1], index_symbols, node_to_sympy)
    if lhs is None or rhs is None:
        return None
    match target:
        case torch.ops.aten.add.Tensor | torch.ops.aten.add.Scalar:
            return V.graph.sizevars.simplify(lhs + rhs)
        case torch.ops.aten.sub.Tensor | torch.ops.aten.sub.Scalar:
            return V.graph.sizevars.simplify(lhs - rhs)
        case torch.ops.aten.mul.Tensor | torch.ops.aten.mul.Scalar:
            return V.graph.sizevars.simplify(lhs * rhs)
        case torch.ops.aten.minimum.default:
            return Min(lhs, rhs)
        case torch.ops.aten.maximum.default:
            return Max(lhs, rhs)
        case torch.ops.aten.remainder.Tensor | torch.ops.aten.remainder.Scalar:
            return ModularIndexing(lhs, 1, rhs)
        case torch.ops.aten.div.Tensor_mode if (
            index.kwargs.get("rounding_mode") == "floor"
        ):
            return FloorDiv(lhs, rhs)
        case _:
            return None
<|endoftext|># mypy: allow-untyped-defs
"""Call into flash-attention 4 for flexattention"""

import dataclasses
import functools
import importlib
import inspect
import warnings
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import Any, cast, Literal

import sympy
from sympy import Expr, Integer

import torch
from torch.fx import GraphModule

from ...codegen.cutedsl.aux_scalars import CuteDSLAuxScalarBindings
from ...ir import FixedLayout, ShapeAsConstantBuffer, Subgraph, TensorBox
from ...lowering import empty_strided
from ...select_algorithm import autotune_select_algorithm
from ...virtualized import V
from .aux_vectorization import (
    DEFAULT_MASK_MOD_VEC_SIZE,
    select_mask_mod_vec_size,
    select_score_mod_vec_size,
)
from .common import (
    create_indices_fake,
    create_num_blocks_fake_generator,
    infer_dense_strides,
    load_flex_template,
    SubgraphResults,
)
from .interval_mask_packing import PackedMaskInterval, select_packed_mask_intervals


@dataclasses.dataclass
class FlexFlashConfig:
    """Autotuning configuration for CuteDSL flex flash attention kernels.

    score_mod_vec_size: Number of elements processed per thread in the score_mod
        application loop. Maps to score_mod.__vec_size__ in CuTe flash attention.
        None uses the kernel default. Only effective for forward; backward does
        not currently support vectorized score_mod.
    mask_mod_vec_size: Number of consecutive KV lanes evaluated per mask_mod
        call. Maps to mask_mod.__vec_size__ in CuTe flash attention and to
        the direct captured-tensor vector-load width for mask_mod.
    mask_mod_packed_intervals: Precomputed 32-lane packed mask intervals.
    """

    score_mod_vec_size: int | None = None
    mask_mod_vec_size: int | None = None
    mask_mod_packed_intervals: tuple[PackedMaskInterval, ...] | None = None


def collect_aux_scalar_symbols(
    *buffer_groups: Sequence[Any],
) -> tuple[sympy.Symbol, ...]:
    symbols: dict[sympy.Symbol, None] = {}
    for buffers in buffer_groups:
        for buffer in buffers:
            if isinstance(buffer, sympy.Expr):
                for symbol in sorted(buffer.free_symbols, key=lambda s: s.name):
                    symbols.setdefault(symbol, None)
    return tuple(symbols)


def get_flex_flash_fwd_configs(
    has_score_mod: bool,
    has_aux_tensors: bool,
    device: torch.device | None = None,
    score_mod_graph_module: GraphModule | None = None,
    score_mod_other_buffers: Sequence[TensorBox] = (),
    has_mask_mod: bool = False,
    has_mask_aux_tensors: bool = False,
    mask_mod_graph_module: GraphModule | None = None,
    mask_mod_other_buffers: Sequence[TensorBox] = (),
    aux_scalar_symbols: Sequence[sympy.Symbol] = (),
) -> list[FlexFlashConfig]:
    cuda_major = None
    if torch.cuda.is_available() and (
        has_mask_mod or (has_score_mod and has_aux_tensors)
    ):
        device_index = None if device is None else device.index
        cuda_major = torch.cuda.get_device_capability(device_index)[0]
    mask_mod_vec_size = select_mask_mod_vec_size(
        has_mask_mod=has_mask_mod,
        has_mask_aux_tensors=has_mask_aux_tensors,
        supports_mask_mod_vec=cuda_major in (10, 11),
        graph_module=mask_mod_graph_module,
        other_buffers=mask_mod_other_buffers,
    )
    score_mod_vec_size = select_score_mod_vec_size(
        has_score_mod=has_score_mod,
        has_aux_tensors=has_aux_tensors,
        is_sm100_or_later=cuda_major is not None and cuda_major >= 10,
        graph_module=score_mod_graph_module,
        other_buffers=score_mod_other_buffers,
    )
    mask_mod_packed_intervals = None
    if has_mask_mod and cuda_major in (10, 11) and mask_mod_graph_module is not None:
        mask_mod_packed_intervals = select_packed_mask_intervals(
            mask_mod_graph_module,
            mask_mod_other_buffers,
            CuteDSLAuxScalarBindings(tuple(aux_scalar_symbols)).symbol_codes(),
        )
    if mask_mod_packed_intervals is not None:
        mask_mod_vec_size = DEFAULT_MASK_MOD_VEC_SIZE

    if (
        has_score_mod
        and score_mod_vec_size is None
        and torch._inductor.config.max_autotune
    ):
        # None means no captured tensor load constrained the score_mod vector width.
        # Scalar captures are lane-uniform aux_scalars, so any kernel-supported
        # power-of-two vector width is legal.
        score_mod_vec_sizes = (1, 2, 4, 8, 16, 32, 64, 128)
    else:
        score_mod_vec_sizes = (score_mod_vec_size,)
    configs = [
        FlexFlashConfig(
            score_mod_vec_size=v,
            mask_mod_vec_size=mask_mod_vec_size,
            mask_mod_packed_intervals=mask_mod_packed_intervals,
        )
        for v in score_mod_vec_sizes
    ]
    max_configs = torch._inductor.config.test_configs.max_flex_configs
    if max_configs is not None and len(configs) > max_configs:
        configs = configs[:max_configs]
    return configs


def _get_flex_flash_bwd_configs() -> list[FlexFlashConfig]:
    return [FlexFlashConfig()]


aten = torch.ops.aten
prims = torch.ops.prims

FLASH_ATTENTION_INSTALL_MESSAGE = (
    "Install a compatible Flash Attention package, for example "
    '`pip install --pre flash-attn-4` (`pip install --pre "flash-attn-4[cu13]"` '
    "for CUDA 13), and see https://pypi.org/project/flash-attn-4/ "
    "for PyPI packaging details."
)


def _flash_attention_unavailable_message() -> str:
    return (
        "CUTE flash attention library is not available. "
        f"{FLASH_ATTENTION_INSTALL_MESSAGE}"
    )


@functools.lru_cache(maxsize=1)
def ensure_flash_available() -> bool:
    """Check if flash-attn is importable; cache the result for reuse.

    Call ensure_flash_available.cache_clear() after installing flash-attn
    in the same interpreter to retry the import.
    """
    try:
        return importlib.util.find_spec("flash_attn.cute") is not None  # type: ignore[attr-defined]
    except ImportError:
        return False


@functools.lru_cache(maxsize=1)
def flash_supports_aux_scalars() -> bool:
    """Check whether the installed FA4 package supports scalar captures."""
    try:
        interface = importlib.import_module("flash_attn.cute.interface")
    except ImportError:
        return False
    return (
        "aux_scalars" in inspect.signature(interface._flash_attn_fwd).parameters
        and "aux_scalars" in inspect.signature(interface._flash_attn_bwd).parameters
    )


from ...codegen.cutedsl.cutedsl_template import CuteDSLTemplate


flash_attention_cutedsl_template = CuteDSLTemplate(
    name="flash_attention_cutedsl", source=load_flex_template("flash_attention")
)
flash_attention_backward_cutedsl_template = CuteDSLTemplate(
    name="flash_attention_backward_cutedsl",
    source=load_flex_template("flash_attention_backward"),
)


class HierarchicalIndex(sympy.Function):
    """
    Inert wrapper to carry an N-D index tuple through Inductor's SymPy-based IR.

    Inductor generally represents a tensor index as a single `sympy.Expr` (often a
    flattened linear offset in memory). CuteDSL, however, wants structured coordinates so it
    can emit `tensor[i, j, ...]` and handle strides internally. We therefore wrap
    the per-dimension indices in a `sympy.Function` node: this keeps the value a
    `sympy.Expr` for existing substitution/CSE machinery, while letting CuteDSL
    codegen pattern-match and unpack the coordinates via `index.args`.

    `eval()` returns None to keep the node inert (no simplification/flattening).

    These nodes are intended to be short-lived wrappers and are only interpreted by
    CuteDSL codegen (see `ModificationWrapperCuteDSL.load` in
    `torch/_inductor/codegen/cutedsl/cutedsl_kernel.py`).
    """

    @classmethod
    def eval(cls, *args):
        return None


def _hierarchical_indexer_cute(
    size: Sequence[int],
    stride: Sequence[int] | None = None,
    offset: Expr = Integer(0),
) -> Callable[[Sequence[Expr]], Expr]:
    """Return an indexer that preserves multi-dimensional indices for CuteDSL."""

    def indexer(indices: Sequence[Expr]) -> Expr:
        if offset != Integer(0):
            raise AssertionError("Offset not supported for hierarchical indexing")
        if len(indices) != len(size):
            raise AssertionError(
                f"Rank mismatch: got {len(indices)} indices for tensor of rank {len(size)}"
            )
        if not indices:
            return Integer(0)
        if len(indices) == 1:
            return indices[0]
        return HierarchicalIndex(*indices)

    return indexer


@contextmanager
def patch_fixed_layout_indexer_for_cutedsl():
    """
    Temporarily swap FixedLayout.make_indexer so CuteDSL sees hierarchical indexing.

    Note [CuteDSL indexer patch]:
    Flex flash attention only supports a limited set of IR ops (pointwise, reads, no stores),
    so temporarily changing the indexing behavior is safe for the kernels we emit today.
    TODO(dynamic shapes): Reconfirm once flex flash attention supports dynamic shapes.
    """
    original_make_indexer = FixedLayout.make_indexer

    def cutedsl_make_indexer(self):
        return _hierarchical_indexer_cute(self.size, self.stride, self.offset)

    FixedLayout.make_indexer = cutedsl_make_indexer  # type: ignore[assignment]
    try:
        yield
    finally:
        FixedLayout.make_indexer = original_make_indexer  # type: ignore[assignment]


def wrap_choice_render_with_cutedsl_indexer(choice: Any) -> None:
    """
    Wrap a template choice's kernel render to apply CuteDSL indexer patching.

    See Note [CuteDSL indexer patch]:
    CuteDSL handles tensor strides internally, so template rendering must use
    hierarchical indexing.
    """
    original_make_kernel_render = choice.make_kernel_render

    def make_kernel_render_with_patch(*args, **kwargs):
        render_kernel, render = original_make_kernel_render(*args, **kwargs)

        def render_with_patch():
            with patch_fixed_layout_indexer_for_cutedsl():
                return render()

        return render_kernel, render_with_patch

    choice.make_kernel_render = make_kernel_render_with_patch


def input_buffers_require_grads(graph_module, num_score_mod_placeholders: int):
    """Check if any of the input buffers (beyond the score mod placeholders) require gradients."""
    inputs = []
    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            inputs.append(node)
    if len(inputs) <= num_score_mod_placeholders:
        return False

    def requires_grad(n):
        tensor_meta = n.meta.get("tensor_meta")
        return tensor_meta.requires_grad if tensor_meta is not None else False

    return any(requires_grad(n) for n in inputs[num_score_mod_placeholders:])


def is_trivial_score_graph(graph_module: GraphModule) -> bool:
    """Backwards currently doesn't support score_mods, match against identity"""
    graph = graph_module.graph
    nodes = list(graph.nodes)
    placeholders = [n for n in nodes if n.op == "placeholder"]
    output = [n for n in nodes if n.op == "output"]
    if len(output) != 1:
        raise AssertionError("Got graph w/ multiple outputs")
    output_val = output[0].args[0]
    # The identity graph just sends the score straight through
    return output_val == placeholders[0]


def is_trivial_mask_graph(graph_module: GraphModule) -> bool:
    """Mask graph is trivial when it only gates via the default full op."""
    graph = graph_module.graph
    nodes = list(graph.nodes)
    placeholders = [n for n in nodes if n.op == "placeholder"]
    output = [n for n in nodes if n.op == "output"]
    if len(output) != 1:
        raise AssertionError("Got graph w/ multiple outputs")
    output_val = output[0].args[0]

    # mask mod graph is empty if we have 4 inputs and full_default output
    return len(placeholders) == 4 and output_val.target is torch.ops.aten.full.default


def has_unsupported_cpu_scalar_tensor_captures(
    score_mod_other_buffers: Sequence[Any],
    mask_mod_other_buffers: Sequence[Any],
) -> bool:
    """Return True for CPU 0-d tensor captures that need scalarization first."""
    for buf in list(score_mod_other_buffers) + list(mask_mod_other_buffers):
        if isinstance(buf, TensorBox):
            device = buf.get_device()
            size = buf.get_size()
            if device is not None and device.type == "cpu" and len(size) == 0:
                return True
    return False


def _can_use_flex_flash_attention(
    subgraph: Subgraph,
    mask_graph: Subgraph,
    num_score_mod_placeholders: int,
) -> tuple[bool, str]:
    """Check if flex flash attention can be used for the given inputs.

    Returns:
        tuple: (can_use, reason) where reason explains why it can't be used if can_use is False
    """
    if not ensure_flash_available():
        return False, _flash_attention_unavailable_message()

    if input_buffers_require_grads(subgraph.graph_module, num_score_mod_placeholders):
        return (
            False,
            "Input buffers require gradients (not supported by flash attention)",
        )

    return True, ""


def _use_flex_flash_attention(
    subgraph: Subgraph,
    mask_graph: Subgraph,
    kernel_options: dict[str, Any],
    num_score_mod_placeholders: int,
    backend: Literal["AUTO", "TRITON", "FLASH", "TRITON_DECODE"],
) -> bool:
    """Determine if we should use flex flash attention for the given inputs.

    Args:
        subgraph: The score modification subgraph
        mask_graph: The mask modification subgraph
        kernel_options: Kernel configuration options
        num_score_mod_placeholders: Number of placeholders in score_mod
        backend: Implementation selector (AUTO, TRITON, FLASH, TRITON_DECODE)

    Returns:
        True if flash attention should be used, False otherwise
    """
    # Flash is experimental and must be explicitly requested
    if backend != "FLASH":
        return False

    can_use, reason = _can_use_flex_flash_attention(
        subgraph,
        mask_graph,
        num_score_mod_placeholders,
    )

    if not can_use:
        raise RuntimeError(
            f"BACKEND='FLASH' but flash attention cannot be used: {reason}"
        )

    return True


def create_flex_flash_attention_kernel(
    query: TensorBox,
    key: TensorBox,
    value: TensorBox,
    block_mask: tuple[Any, ...],
    scale: float,
    kernel_options: dict[str, Any],
    subgraph_buffer: SubgraphResults,
    mask_graph_buffer: SubgraphResults,
    score_mod_other_buffers: list[TensorBox],
    mask_mod_other_buffers: list[TensorBox],
    kv_num_blocks: TensorBox | None,
    kv_indices: TensorBox | None,
    full_kv_num_blocks: TensorBox | None,
    full_kv_indices: TensorBox | None,
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
    mask_graph: Subgraph,
    subgraph: Subgraph | None = None,
) -> tuple[TensorBox, TensorBox]:
    """Create a flex flash attention kernel using CuteDSL template."""
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError(
            f"Mixed query, key, and value dtype is not supported on this platform, "
            f"got query.dtype: {query.dtype}, key.dtype: {key.dtype}, "
            f"and value.dtype: {value.dtype}."
        )
    if not ensure_flash_available():
        raise RuntimeError(_flash_attention_unavailable_message())

    # Get dimensions
    batch_size, num_heads, seq_len_q, head_dim = query.get_size()
    v_head_dim = value.get_size()[-1]
    device = query.get_device()
    dtype = query.get_dtype()
    if device is None:
        raise AssertionError("Device must be specified")

    # Match stride pattern from query tensor
    q_strides = query.get_stride()
    out_size = [batch_size, num_heads, seq_len_q, v_head_dim]
    out_strides = infer_dense_strides(out_size, q_strides)

    output = empty_strided(
        size=out_size,
        stride=out_strides,
        dtype=dtype,
        device=device,
    )

    lse = empty_strided(
        size=[batch_size, num_heads, seq_len_q],
        stride=None,  # LSE can be contiguous
        dtype=torch.float32,  # LSE is always fp32
        device=device,
    )

    # Create layout for primary output
    output_layout = FixedLayout(
        device=device,
        dtype=dtype,
        size=[batch_size, num_heads, seq_len_q, v_head_dim],
        stride=[sympy.sympify(s) for s in output.get_stride()],
    )

    mask_graph_is_trivial = is_trivial_mask_graph(mask_graph.graph_module)
    score_graph_is_trivial = subgraph is None or is_trivial_score_graph(
        subgraph.graph_module
    )

    if kv_num_blocks is None or kv_indices is None:
        raise AssertionError("kv block metadata is required for flex FLASH")

    has_score_mod = not score_graph_is_trivial
    has_mask_mod = not mask_graph_is_trivial
    has_full_blocks = full_kv_num_blocks is not None
    if has_full_blocks and full_kv_indices is None:
        raise AssertionError("full_kv_indices must be provided with full_kv_num_blocks")
    is_noop_block_mask = not has_full_blocks and V.graph.sizevars.statically_known_true(
        sympy.And(
            sympy.Eq(sparse_q_block_size, 1 << 30),
            sympy.Eq(sparse_kv_block_size, 1 << 30),
        )
    )
    needs_block_mask = not (mask_graph_is_trivial and is_noop_block_mask)
    sparse_q_block_size = V.graph.sizevars.guard_int(sparse_q_block_size)
    sparse_kv_block_size = V.graph.sizevars.guard_int(sparse_kv_block_size)

    choices: list[Any] = []
    if flash_attention_cutedsl_template is None:
        raise AssertionError("flash_attention_cutedsl_template must not be None")

    input_nodes = [query, key, value, lse]
    if needs_block_mask:
        input_nodes.extend([kv_num_blocks, kv_indices])
        if has_full_blocks:
            input_nodes.extend([full_kv_num_blocks, full_kv_indices])

    subgraphs = []
    if has_score_mod:
        subgraphs.append(subgraph_buffer)
    if has_mask_mod:
        subgraphs.append(mask_graph_buffer)

    aux_scalar_symbols = collect_aux_scalar_symbols(
        score_mod_other_buffers, mask_mod_other_buffers
    )
    if aux_scalar_symbols and not flash_supports_aux_scalars():
        raise RuntimeError(
            "CUTE flash attention scalar captures require flash-attn-4>=4.0.0b17. "
            f"{FLASH_ATTENTION_INSTALL_MESSAGE}"
        )
    has_score_aux_tensors = any(
        not isinstance(buffer, sympy.Expr) for buffer in score_mod_other_buffers
    )
    has_mask_aux_tensors = any(
        not isinstance(buffer, sympy.Expr) for buffer in mask_mod_other_buffers
    )

    configs = get_flex_flash_fwd_configs(
        has_score_mod=has_score_mod,
        has_aux_tensors=has_score_aux_tensors,
        device=device,
        score_mod_graph_module=(
            subgraph.graph_module if has_score_mod and subgraph is not None else None
        ),
        score_mod_other_buffers=score_mod_other_buffers,
        has_mask_mod=has_mask_mod,
        has_mask_aux_tensors=has_mask_aux_tensors,
        mask_mod_graph_module=mask_graph.graph_module,
        mask_mod_other_buffers=mask_mod_other_buffers,
        aux_scalar_symbols=aux_scalar_symbols,
    )
    error: NotImplementedError | None = None
    for conf in configs:
        with patch_fixed_layout_indexer_for_cutedsl():
            error = flash_attention_cutedsl_template.maybe_append_choice(
                choices,
                input_nodes=input_nodes,
                layout=output_layout,
                mutated_inputs=[lse],
                subgraphs=subgraphs,
                SM_SCALE=scale,
                HAS_SCORE_MOD=has_score_mod,
                SCORE_MOD_VEC_SIZE=conf.score_mod_vec_size,
                MASK_MOD_VEC_SIZE=conf.mask_mod_vec_size,
                MASK_MOD_PACKED_INTERVALS=conf.mask_mod_packed_intervals,
                MASK_MOD_OTHER_BUFFERS=mask_mod_other_buffers,
                AUX_SCALAR_SYMBOLS=aux_scalar_symbols,
                NEEDS_BLOCK_MASK=needs_block_mask,
                HAS_MASK_MOD=has_mask_mod,
                HAS_FULL_BLOCKS=has_full_blocks,
                SPARSE_Q_BLOCK_SIZE=sparse_q_block_size,
                SPARSE_KV_BLOCK_SIZE=sparse_kv_block_size,
            )
        if error is not None and len(configs) == 1:
            raise RuntimeError(f"CuteDSL template failed: {error}")

    for choice in choices:
        wrap_choice_render_with_cutedsl_indexer(choice)

    if not choices:
        raise RuntimeError(f"CuteDSL template failed: {error}")

    input_gen_fns: dict[int, Callable] | None = None
    if needs_block_mask:
        input_gen_fns = {
            4: create_num_blocks_fake_generator(kv_indices),
            5: create_indices_fake,
        }
        if has_full_blocks:
            input_gen_fns.update(
                {
                    6: create_num_blocks_fake_generator(full_kv_indices),
                    7: create_indices_fake,
                }
            )

    template_output, _ = autotune_select_algorithm(
        "flex_flash_attention",
        choices,
        input_nodes,
        output_layout,
        input_gen_fns=input_gen_fns,
        return_multi_template=False,
    )

    return (template_output, lse)


def _can_use_flex_flash_attention_backward(
    fw_subgraph: Subgraph,
    mask_graph: Subgraph,
    joint_outputs: Any | None = None,
    score_mod_other_buffers: Sequence[TensorBox] | None = None,
    num_score_mod_placeholders: int = 5,
) -> tuple[bool, str]:
    if not ensure_flash_available():
        return False, _flash_attention_unavailable_message()

    if input_buffers_require_grads(
        fw_subgraph.graph_module, num_score_mod_placeholders
    ):
        return (
            False,
            "Input buffers require gradients (not supported by flash attention backward)",
        )

    if joint_outputs is not None:
        if joint_outputs.captured_grads_compute:
            return (
                False,
                "NYI: Flex Flash Attention bwd doesn't support captured grads yet.",
            )
        if joint_outputs.mutated_grads:
            return (
                False,
                "NYI: Flex Flash Attention bwd doesn't support mutated grads yet.",
            )

    return True, ""


def _use_flex_flash_attention_backward(
    fw_subgraph: Subgraph,
    mask_graph: Subgraph,
    backend: Literal["AUTO", "TRITON", "FLASH", "TRITON_DECODE"],
    joint_outputs: Any | None = None,
    score_mod_other_buffers: Sequence[TensorBox] | None = None,
) -> bool:
    """Determine if we should use flex flash attention for the given inputs.

    Args:
        fw_subgraph: The forward score modification subgraph
        mask_graph: The mask modification subgraph
        backend: Implementation selector (AUTO, TRITON, FLASH, TRITON_DECODE)
        joint_outputs: Processed joint outputs (for PR1 constraint checking)
        score_mod_other_buffers: Additional buffers used by score_mod

    Returns:
        True if flash attention should be used, False otherwise
    """
    # Flash is experimental and must be explicitly requested
    if backend != "FLASH":
        return False

    can_use, reason = _can_use_flex_flash_attention_backward(
        fw_subgraph,
        mask_graph,
        joint_outputs,
        score_mod_other_buffers,
    )

    if not can_use:
        raise RuntimeError(
            f"BACKEND='FLASH' but flash attention cannot be used: {reason}"
        )

    return True


def create_flex_flash_attention_backward_kernel(
    query: TensorBox,
    key: TensorBox,
    value: TensorBox,
    out: TensorBox,
    logsumexp: TensorBox,
    grad_out: TensorBox,
    grad_logsumexp: TensorBox | None,
    scale: float,
    kernel_options: dict[str, Any],
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
    fw_subgraph_buffer: SubgraphResults | None = None,
    joint_subgraph_buffer: Any | None = None,
    score_mod_other_buffers: list[TensorBox] | None = None,
    mask_graph_buffer: SubgraphResults | None = None,
    mask_mod_other_buffers: list[TensorBox] | None = None,
    q_num_blocks: TensorBox | None = None,
    q_indices: TensorBox | None = None,
    full_q_num_blocks: TensorBox | None = None,
    full_q_indices: TensorBox | None = None,
    dq_write_order: TensorBox | None = None,
    dq_write_order_full: TensorBox | None = None,
    dq_kv_order: TensorBox | None = None,
    dq_kv_order_spt: bool | None = None,
) -> tuple[TensorBox | ShapeAsConstantBuffer, TensorBox, TensorBox, tuple]:
    """Create a CuteDSL flash attention backward kernel for the default mod path."""
    if not ensure_flash_available():
        raise RuntimeError(_flash_attention_unavailable_message())

    batch_size, num_heads, seq_len_q, head_dim = query.get_size()
    _, num_heads_kv, seq_len_kv, v_head_dim = value.get_size()
    device = query.get_device()
    dtype = query.get_dtype()
    if device is None:
        raise AssertionError("Device must not be None")

    grad_query_strides = infer_dense_strides(
        [batch_size, num_heads, seq_len_q, head_dim], query.get_stride()
    )
    grad_query = empty_strided(
        size=[batch_size, num_heads, seq_len_q, head_dim],
        stride=grad_query_strides,
        dtype=dtype,
        device=device,
    )

    grad_key_strides = infer_dense_strides(
        [batch_size, num_heads_kv, seq_len_kv, head_dim], key.get_stride()
    )
    grad_key = empty_strided(
        size=[batch_size, num_heads_kv, seq_len_kv, head_dim],
        stride=grad_key_strides,
        dtype=dtype,
        device=device,
    )

    grad_value_strides = infer_dense_strides(
        [batch_size, num_heads_kv, seq_len_kv, v_head_dim], value.get_stride()
    )
    grad_value = empty_strided(
        size=[batch_size, num_heads_kv, seq_len_kv, v_head_dim],
        stride=grad_value_strides,
        dtype=dtype,
        device=device,
    )

    # we use dq as the output layout
    output_layout = FixedLayout(
        device=device,
        dtype=dtype,
        size=[batch_size, num_heads, seq_len_q, head_dim],
        stride=[sympy.sympify(s) for s in grad_query.get_stride()],
    )

    sparse_q_block_size = V.graph.sizevars.guard_int(sparse_q_block_size)
    sparse_kv_block_size = V.graph.sizevars.guard_int(sparse_kv_block_size)

    has_dlse = grad_logsumexp is not None
    if (
        has_dlse
        and V.graph.sizevars.statically_known_equals(head_dim, 256)
        and V.graph.sizevars.statically_known_equals(v_head_dim, 256)
        and torch.cuda.get_device_capability(device)[0] in (10, 11)
    ):
        raise NotImplementedError(
            "FLASH backend dLSE is not supported by the SM100/SM110 dedicated "
            "head_dim=256 FA4 backward kernel. Use BACKEND='TRITON' for this "
            "configuration."
        )

    choices: list[Any] = []

    input_nodes: list[TensorBox] = [
        query,
        key,
        value,
        out,
        grad_out,
        logsumexp,
        grad_key,
        grad_value,
    ]

    has_block_mask = mask_graph_buffer is not None
    if has_block_mask:
        if q_indices is None:
            raise AssertionError("q_indices required when block mask is present")
        if full_q_num_blocks is None:
            raise AssertionError(
                "full_q_num_blocks required when block mask is present"
            )
        if full_q_indices is None:
            raise AssertionError("full_q_indices required when block mask is present")
        input_nodes.extend(
            [
                cast(TensorBox, q_num_blocks),
                q_indices,
                full_q_num_blocks,
                full_q_indices,
            ]
        )

    has_dq_write_order = dq_write_order is not None
    if has_dq_write_order:
        input_nodes.append(dq_write_order)
        if dq_write_order_full is not None:
            input_nodes.append(dq_write_order_full)
    has_dq_kv_order = dq_kv_order is not None and has_dq_write_order
    dq_kv_order_spt_for_flash = dq_kv_order_spt if has_dq_write_order else None
    if has_dq_kv_order:
        input_nodes.append(dq_kv_order)
    if has_dlse:
        input_nodes.append(cast(TensorBox, grad_logsumexp))

    supports_dq_kv_order = False
    supports_spt = False
    if has_block_mask:
        # pyrefly: ignore[missing-import]
        from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

        block_sparse_fields = getattr(BlockSparseTensorsTorch, "_fields", ())
        supports_dq_kv_order = "dq_kv_order" in block_sparse_fields
        supports_spt = "spt" in block_sparse_fields
        if has_dq_kv_order and not supports_dq_kv_order:
            raise NotImplementedError(
                "Explicit tensor dq_kv_order requires flash-attn-4 with dq_kv_order support"
            )
        if dq_kv_order_spt_for_flash is not None and not (
            supports_dq_kv_order or supports_spt
        ):
            raise NotImplementedError(
                "Boolean dq_kv_order requires flash-attn-4 with dq_kv_order or spt support"
            )

    deterministic_requested = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    deterministic_backward_enabled = deterministic_requested
    if deterministic_requested and has_block_mask:
        major, _ = torch.cuda.get_device_capability(device)
        missing_dq_write_order = dq_write_order is None or (
            full_q_num_blocks is not None and dq_write_order_full is None
        )
        missing_dq_kv_order = not (
            has_dq_kv_order or dq_kv_order_spt_for_flash is not None
        )
        if major < 10:
            if warn_only:
                deterministic_backward_enabled = False
            else:
                raise NotImplementedError(
                    "Deterministic backward for flex_attention with block_mask and BACKEND='FLASH' "
                    "requires SM100+ (compute capability >= 10.0). "
                    "Use BACKEND='TRITON' for deterministic backward on older architectures."
                )
        elif missing_dq_write_order:
            if warn_only:
                deterministic_backward_enabled = False
            else:
                raise ValueError(
                    "Deterministic backward for flex_attention with block_mask and BACKEND='FLASH' "
                    "requires dQ write-order metadata. Create the block mask with "
                    "create_block_mask(..., compute_dq_write_order=True)."
                )
        elif missing_dq_kv_order:
            if warn_only:
                deterministic_backward_enabled = False
            else:
                raise ValueError(
                    "Deterministic backward for flex_attention with block_mask and BACKEND='FLASH' "
                    "requires dQ KV scheduler-order metadata. Create the block mask with "
                    "create_block_mask(..., compute_dq_write_order=True)."
                )
    if deterministic_requested and not deterministic_backward_enabled:
        warnings.warn(
            "flex_attention backward with block_mask and BACKEND='FLASH' does not have "
            "a deterministic implementation for this configuration, but you set "
            "'torch.use_deterministic_algorithms(True, warn_only=True)'. "
            "Running non-deterministic backward.",
        )

    has_score_mod = fw_subgraph_buffer is not None and joint_subgraph_buffer is not None
    subgraphs = []
    if has_score_mod:
        subgraphs.append(fw_subgraph_buffer)
        subgraphs.append(joint_subgraph_buffer)
    if has_block_mask:
        subgraphs.append(mask_graph_buffer)

    aux_scalar_symbols = collect_aux_scalar_symbols(
        score_mod_other_buffers or (), mask_mod_other_buffers or ()
    )
    if aux_scalar_symbols and not flash_supports_aux_scalars():
        raise RuntimeError(
            "CUTE flash attention scalar captures require flash-attn-4>=4.0.0b17. "
            f"{FLASH_ATTENTION_INSTALL_MESSAGE}"
        )
    configs = _get_flex_flash_bwd_configs()

    error: NotImplementedError | None = None
    for conf in configs:
        with patch_fixed_layout_indexer_for_cutedsl():
            error = flash_attention_backward_cutedsl_template.maybe_append_choice(
                choices,
                input_nodes=input_nodes,
                layout=output_layout,
                mutated_inputs=[grad_key, grad_value],
                subgraphs=subgraphs or None,
                SM_SCALE=scale,
                HAS_SCORE_MOD=has_score_mod,
                SCORE_MOD_VEC_SIZE=conf.score_mod_vec_size,
                HAS_BLOCK_MASK=has_block_mask,
                HAS_DQ_WRITE_ORDER=has_dq_write_order,
                HAS_DQ_WRITE_ORDER_FULL=dq_write_order_full is not None,
                HAS_DQ_KV_ORDER=has_dq_kv_order,
                DQ_KV_ORDER_SPT=dq_kv_order_spt_for_flash,
                HAS_DLSE=has_dlse,
                AUX_SCALAR_SYMBOLS=aux_scalar_symbols,
                SUPPORTS_DQ_KV_ORDER=supports_dq_kv_order,
                SUPPORTS_SPT=supports_spt,
                DETERMINISTIC_BACKWARD_ENABLED=deterministic_backward_enabled,
                SPARSE_Q_BLOCK_SIZE=sparse_q_block_size,
                SPARSE_KV_BLOCK_SIZE=sparse_kv_block_size,
            )
        if error is not None and len(configs) == 1:
            raise RuntimeError(f"CuteDSL template failed: {error}")

    for choice in choices:
        wrap_choice_render_with_cutedsl_indexer(choice)

    if not choices:
        raise RuntimeError(f"CuteDSL template failed: {error}")

    input_gen_fns: dict[int, Callable] | None = None
    if has_block_mask:
        input_gen_fns = {
            8: create_num_blocks_fake_generator(q_indices),
            9: create_indices_fake,
            10: create_num_blocks_fake_generator(full_q_indices),
            11: create_indices_fake,
        }

    template_output, _ = autotune_select_algorithm(
        "flex_flash_attention_backward",
        choices,
        input_nodes,
        output_layout,
        input_gen_fns=input_gen_fns,
        return_multi_template=False,
    )

    return (template_output, grad_key, grad_value, tuple())
<|endoftext|># mypy: allow-untyped-defs
"""MPS-specific lowering for flex attention."""

import sympy

import torch
from torch._inductor.virtualized import V

from ...ir import FixedLayout, ShapeAsConstantBuffer, TensorBox
from ...lowering import empty_strided
from ...select_algorithm import realize_inputs
from ...utils import expr_fits_within_32bit
from .common import infer_dense_strides, maybe_realize


BLOCK_M = 32


def lower_mps(
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
    """Lower flex_attention to a Metal shader for MPS."""
    from ...codegen.metal_flex_attention_template import (
        _generate_metal_shader,
        MetalFlexAttentionNode,
    )

    (
        _,  # q_length
        _,  # kv_length
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        _,  # q_num_blocks
        _,  # q_indices
        _,  # full_q_num_blocks
        _,  # full_q_indices
        _,  # dq_write_order (backward-only)
        _,  # dq_write_order_full (backward-only)
        _,  # dq_kv_order (backward-only)
        _,  # dq_kv_order_spt (backward-only)
        SPARSE_Q_BLOCK_SIZE,
        SPARSE_KV_BLOCK_SIZE,
        mask_graph,
    ) = block_mask

    if query.get_dtype() not in (torch.float32, torch.float16, torch.bfloat16):
        raise NotImplementedError(
            f"flex_attention on MPS supports float32/float16/bfloat16, got {query.get_dtype()}"
        )
    if query.get_dtype() != key.get_dtype() or query.get_dtype() != value.get_dtype():
        raise NotImplementedError(
            "Mixed dtypes for query, key, value not supported on MPS"
        )

    write_lse = bool(kernel_options.get("OUTPUT_LOGSUMEXP", False))
    write_max = bool(kernel_options.get("OUTPUT_MAX", False))

    dtype = query.get_dtype()

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

    B, Hq, seq_len_q, qk_head_dim = query.get_size()
    Bkv, Hkv, seq_len_kv, v_head_dim = value.get_size()
    _, _, _, k_head_dim = key.get_size()

    has_full_blocks = full_kv_num_blocks is not None

    d_qk = V.graph.sizevars.guard_int(qk_head_dim)
    d_v = V.graph.sizevars.guard_int(v_head_dim)
    d_k = V.graph.sizevars.guard_int(k_head_dim)
    if d_k != d_qk:
        raise NotImplementedError(
            f"flex_attention on MPS requires query and key to share head dim, "
            f"got d_qk={d_qk} d_k={d_k}"
        )

    sizevars = V.graph.sizevars
    if not sizevars.evaluate_expr(sympy.Eq(B, Bkv) | sympy.Eq(Bkv, 1)):
        raise AssertionError(f"Bq and Bkv must broadcastable. Got Bq={B} and Bkv={Bkv}")

    SPARSE_KV_BLOCK_SIZE_val = sizevars.guard_int(SPARSE_KV_BLOCK_SIZE)
    SPARSE_Q_BLOCK_SIZE_val = sizevars.guard_int(SPARSE_Q_BLOCK_SIZE)
    if SPARSE_Q_BLOCK_SIZE_val < BLOCK_M or SPARSE_Q_BLOCK_SIZE_val % BLOCK_M != 0:
        # Each threadgroup tiles BLOCK_M query rows and looks up sparse-mask info
        # at sparse_q_idx = m_base / SPARSE_Q_BLOCK_SIZE; a smaller/non-multiple
        # SPARSE_Q_BLOCK_SIZE makes one threadgroup span multiple sparse blocks.
        raise NotImplementedError(
            f"flex_attention on MPS requires SPARSE_Q_BLOCK_SIZE to be a positive "
            f"multiple of {BLOCK_M}, got {SPARSE_Q_BLOCK_SIZE_val}"
        )

    if not sizevars.statically_known_multiple_of(Hq, Hkv):
        raise NotImplementedError(
            f"flex_attention on MPS requires Hq to be a positive multiple of "
            f"Hkv, got Hq={Hq} Hkv={Hkv}"
        )
    gqa_shared_heads = Hq // Hkv

    scale_val = float(scale)

    # Tensor captures become extra Metal buffers; their sizes/strides are baked
    # into the index offset math. SymInt captures (from dynamic-shape closures)
    # become extra packed scalars, evaluated at the call site. maybe_realize
    # passes SymInts through untouched. `metas` keeps the placeholder order so
    # the subgraph codegen can bind each capture (tensor or scalar) by position.
    def _capture_meta(items):
        metas, tensors, scalars = [], [], []
        for cap in maybe_realize(list(items)):
            if isinstance(cap, ShapeAsConstantBuffer):
                cap = cap.expr
            if isinstance(cap, (int, sympy.Expr)):
                metas.append(("scalar",))
                scalars.append(cap)
                continue
            metas.append(
                (
                    "tensor",
                    [sizevars.guard_int(s) for s in cap.get_size()],
                    [sizevars.guard_int(s) for s in cap.get_stride()],
                    cap.get_dtype(),
                )
            )
            tensors.append(cap)
        return metas, tensors, scalars

    score_meta, score_tensors, score_scalars = _capture_meta(score_mod_other_buffers)
    mask_meta, mask_tensors, mask_scalars = _capture_meta(mask_mod_other_buffers)

    int32_max = torch.iinfo(torch.int32).max
    all_scalars = [sympy.sympify(cap) for cap in (*score_scalars, *mask_scalars)]
    captures_fit_int32 = all(expr_fits_within_32bit(cap) for cap in all_scalars)
    if captures_fit_int32:
        for cap in all_scalars:
            sizevars.check_leq(cap, int32_max)

    shader_source = _generate_metal_shader(
        dtype=dtype,
        d_qk=d_qk,
        d_v=d_v,
        score_mod_graph=subgraph.graph_module,
        mask_mod_graph=mask_graph.graph_module,
        has_full_blocks=has_full_blocks,
        block_m=BLOCK_M,
        scale=scale_val,
        score_captured=score_meta,
        mask_captured=mask_meta,
        write_lse=write_lse,
        write_max=write_max,
        captures_fit_int32=captures_fit_int32,
    )

    out_size = [B, Hq, seq_len_q, v_head_dim]
    out_strides = infer_dense_strides(out_size, query.get_stride())
    layout = FixedLayout(
        query.get_device(),
        dtype,
        out_size,
        stride=[sympy.sympify(s) for s in out_strides],
    )

    kv_nb_strides = _get_strides(kv_num_blocks)
    kv_idx_strides = _get_strides(kv_indices)
    full_kv_nb_strides = _get_strides(full_kv_num_blocks) if has_full_blocks else []
    full_kv_idx_strides = _get_strides(full_kv_indices) if has_full_blocks else []

    # Buffer order: 0=Out, 1=Q, 2=K, 3=V, 4=kv_num_blocks, 5=kv_indices,
    # 6=(full_kv_num_blocks), 7=(full_kv_indices), then score/mask captured
    # buffers, then the packed scalar buffer.
    input_nodes = [query, key, value, kv_num_blocks, kv_indices]
    if has_full_blocks:
        input_nodes += [full_kv_num_blocks, full_kv_indices]
    input_nodes += [*score_tensors, *mask_tensors]

    realized_inputs = realize_inputs(*input_nodes)

    # Left-pad to (Z, H, Q_blocks); a sliced kv_num_blocks may be 1D or 2D.
    kv_nb_sizes = list(kv_num_blocks.get_size())
    while len(kv_nb_sizes) < 3:
        kv_nb_sizes.insert(0, sympy.Integer(1))
    sparse_z = kv_nb_sizes[0]
    sparse_hq = kv_nb_sizes[1]

    # Order must match the unpack in metal_flex_attention_template's `scalar_names`.
    scalar_args = [
        B,
        Bkv,
        Hq,
        Hkv,
        seq_len_q,
        seq_len_kv,
        *query.get_stride(),
        *key.get_stride(),
        *value.get_stride(),
        *[sympy.sympify(s) for s in out_strides],
        sympy.Integer(SPARSE_KV_BLOCK_SIZE_val),
        gqa_shared_heads,
        sparse_z,
        sparse_hq,
        *_pad_strides(kv_nb_strides, 3),
        *_pad_strides(kv_idx_strides, 4),
        sympy.Integer(SPARSE_Q_BLOCK_SIZE_val),
    ]
    if has_full_blocks:
        scalar_args += [
            *_pad_strides(full_kv_nb_strides, 3),
            *_pad_strides(full_kv_idx_strides, 4),
        ]

    # SymInt captures trail the fixed scalars; order (score then mask) matches
    # the scalar_capture_names appended in _generate_metal_shader.
    scalar_args += [*score_scalars, *mask_scalars]

    # (num_q_blocks, Hq, B); each threadgroup owns BLOCK_M query rows.
    grid = (
        sympy.ceiling(seq_len_q / BLOCK_M),
        Hq,
        B,
    )

    lse_shape = [B, Hq, seq_len_q]
    logsumexp = empty_strided(
        lse_shape, None, dtype=torch.float32, device=query.get_device()
    )
    max_scores = empty_strided(
        lse_shape, None, dtype=torch.float32, device=query.get_device()
    )
    node_inputs = [*realized_inputs]
    if write_lse:
        logsumexp.realize()
        node_inputs.append(logsumexp)
    if write_max:
        max_scores.realize()
        node_inputs.append(max_scores)

    node = MetalFlexAttentionNode(
        layout=layout,
        inputs=node_inputs,
        shader_source=shader_source,
        scalar_args=scalar_args,
        grid=grid,
        block_m=BLOCK_M,
        num_mutated_outputs=int(write_lse) + int(write_max),
    )

    return (TensorBox.create(node), logsumexp, max_scores)


def _get_strides(node):
    if node is None:
        return []
    return list(node.get_stride())


def _pad_strides(strides, target_len):
    """Left-pad strides to target_len with 0 (broadcast over missing leading dims)."""
    strides = list(strides)
    if len(strides) > target_len:
        strides = strides[-target_len:]
    while len(strides) < target_len:
        strides.insert(0, sympy.Integer(0))
    return strides
<|endoftext|># mypy: allow-untyped-defs
"""Packed interval mask analysis for flex flash attention.

Flex attention can evaluate arbitrary ``mask_mod`` functions lane by lane, but
SM100 packed masks can use a cheaper path when the kept KV lanes in each
32-lane chunk are window-like. This pass recognizes mask graphs where, after
substituting ``kv_idx + mask_lane`` for the per-lane KV index, the predicate can
be represented as a small union of half-open intervals over ``mask_lane``.

This works for causal, sliding-window, block, and document-bound masks whose
bounds depend only on lane-uniform values such as ``q_idx``, ``b_idx``, or aux
loads indexed by them. It intentionally rejects masks that need arbitrary
per-lane inclusion, such as ``doc_ids[b, kv_idx] == doc_ids[b, q_idx]`` or
``kv_idx % 2 == 0``; those would degenerate into many singleton intervals and
are better handled by the generic mask path.
"""

import dataclasses
from collections.abc import Mapping, Sequence
from itertools import product

import sympy

import torch
from torch.fx import GraphModule
from torch.utils._sympy.functions import FloorDiv, Max, Min

from ...codegen.cutedsl.lane_analysis import decompose_affine_lane_expr
from ...virtualized import NullHandler, V
from .aux_vectorization import fx_aux_index_to_sympy


perf_hint_log = torch._logging.getArtifactLogger(__name__, "perf_hints")


# Fall back to scalar mask lowering before interval expansion makes generated CuTe too large.
MAX_PACKED_MASK_INTERVALS_FOR_CODE_SIZE = 8
LANE_UNIFORM_BINARY_OPS: Mapping[object, str] = {
    torch.ops.aten.add.Tensor: "+",
    torch.ops.aten.add.Scalar: "+",
    torch.ops.aten.sub.Tensor: "-",
    torch.ops.aten.sub.Scalar: "-",
    torch.ops.aten.mul.Tensor: "*",
    torch.ops.aten.mul.Scalar: "*",
    torch.ops.aten.remainder.Tensor: "%",
    torch.ops.aten.remainder.Scalar: "%",
}


@dataclasses.dataclass(frozen=True)
class PackedMaskInterval:
    """Half-open keep interval [lower_lane, upper_lane_exclusive) for a packed mask.

    Packed interval lowering replaces per-lane boolean mask evaluation with one
    32-bit keep mask when the mask predicate can be expressed as lane bounds.
    For example, ``q_idx >= kv_idx`` over a 32-lane KV window starting at
    ``kv_idx`` keeps lanes ``[0, q_idx - kv_idx + 1)``.
    """

    lower_lane: sympy.Expr
    upper_lane_exclusive: sympy.Expr
    # CuTe code for temporary symbols that stand in for lane-uniform aux loads.
    symbol_codes: Mapping[sympy.Symbol, str] = dataclasses.field(
        default_factory=dict, compare=False, repr=False
    )

    @classmethod
    def full(cls) -> "PackedMaskInterval":
        return cls(sympy.Integer(0), sympy.Integer(32))

    def is_full(self) -> bool:
        return self.lower_lane == 0 and self.upper_lane_exclusive == 32

    def with_symbol_codes(
        self, symbol_codes: Mapping[sympy.Symbol, str]
    ) -> "PackedMaskInterval":
        return dataclasses.replace(self, symbol_codes=dict(symbol_codes))

    def render_lower(self) -> str:
        return self._render(self.lower_lane)

    def render_upper(self) -> str:
        return self._render(self.upper_lane_exclusive)

    def keep_mask_expr(self) -> str:
        """Render the 32-bit CuTe keep-mask expression for this interval."""
        lower = self.render_lower()
        upper = self.render_upper()
        return (
            "(utils.shr_u32(cutlass.Uint32(0xFFFFFFFF), "
            "cutlass.Uint32(min(max(cutlass.Int32(32) - "
            f"{upper}, cutlass.Int32(0)), cutlass.Int32(32)))) & "
            "utils.shl_u32(cutlass.Uint32(0xFFFFFFFF), "
            "cutlass.Uint32(min(max("
            f"{lower}, cutlass.Int32(0)), cutlass.Int32(32)))))"
        )

    def _render(self, expr: sympy.Expr) -> str:
        rendered = sympy_to_cute_index(expr, self.symbol_codes)
        if rendered is None:
            raise AssertionError(f"failed to render expr to cute index: {expr}")
        return rendered


IntervalSet = tuple[PackedMaskInterval, ...]
MaybeIntervalSet = IntervalSet | None


def is_bool_full_node(node: torch.fx.Node, value: bool) -> bool:
    return (
        node.op == "call_function"
        and node.target is torch.ops.aten.full.default
        and len(node.args) >= 2
        and node.args[0] == []
        and node.args[1] is value
    )


def _simplify_expr(expr: sympy.Expr) -> sympy.Expr:
    """Simplify with Inductor sizevars when available, otherwise use SymPy."""
    if isinstance(V.graph, NullHandler):
        return sympy.simplify(expr)
    return V.graph.sizevars.simplify(expr)


def render_sympy_args(
    args: Sequence[sympy.Expr],
    symbol_codes: Mapping[sympy.Symbol, str] | None,
) -> list[str] | None:
    """Render every child expression, returning ``None`` if any is unsupported."""
    rendered = []
    for arg in args:
        cute_arg = sympy_to_cute_index(arg, symbol_codes)
        if cute_arg is None:
            return None
        rendered.append(cute_arg)
    return rendered


def sympy_to_cute_index(
    expr: sympy.Expr, symbol_codes: Mapping[sympy.Symbol, str] | None = None
) -> str | None:
    """Render a supported SymPy integer expression as CuTe index code."""
    expr = _simplify_expr(expr)
    match expr:
        case sympy.Integer():
            return f"cutlass.Int32({int(expr)})"
        case sympy.Symbol():
            if symbol_codes is not None and expr in symbol_codes:
                return symbol_codes[expr]
            if expr.name == "q_idx":
                return "q_idx[0]"
            if expr.name == "kv_idx":
                return "kv_idx[0]"
        case FloorDiv():
            args = render_sympy_args(expr.args, symbol_codes)
            if args is not None:
                return f"({args[0]} // {args[1]})"
        case sympy.Add() | sympy.Mul():
            args = render_sympy_args(expr.args, symbol_codes)
            if args is None:
                return None
            separator = " + " if isinstance(expr, sympy.Add) else " * "
            return "(" + separator.join(args) + ")"
        case Min() | Max():
            args = render_sympy_args(expr.args, symbol_codes)
            if args is None:
                return None
            op = "min" if isinstance(expr, Min) else "max"
            return f"{op}(" + ", ".join(args) + ")"
    if isinstance(expr, (sympy.Min, sympy.Max)):
        args = render_sympy_args(expr.args, symbol_codes)
        if args is None:
            return None
        op = "min" if isinstance(expr, sympy.Min) else "max"
        return f"{op}(" + ", ".join(args) + ")"
    return None


def is_aten_index_node(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target is torch.ops.aten.index.Tensor


def fx_node_dtype(expr: torch.fx.Node) -> torch.dtype | None:
    """Read an FX node dtype from metadata, following aten.index bases if needed."""
    tensor_meta = expr.meta.get("tensor_meta")
    if tensor_meta is not None:
        return tensor_meta.dtype
    val = expr.meta.get("val")
    if isinstance(val, torch.Tensor):
        return val.dtype
    if is_aten_index_node(expr):
        base = expr.args[0]
        if isinstance(base, torch.fx.Node):
            return fx_node_dtype(base)
    return None


def fx_node_shape(expr: torch.fx.Node) -> tuple[int, ...] | None:
    """Read an FX node shape from tensor metadata or fake tensor value."""
    tensor_meta = expr.meta.get("tensor_meta")
    if tensor_meta is not None:
        return tuple(tensor_meta.shape)
    val = expr.meta.get("val")
    if isinstance(val, torch.Tensor):
        return tuple(val.shape)
    return None


@dataclasses.dataclass
class PackedMaskAnalyzer:
    """Lower one Flex mask_mod FX graph to packed 32-lane mask intervals.

    The analyzer treats each packed mask as a 32-lane window starting at kv_idx,
    rewrites supported mask predicates into bounds over mask_lane, and records
    renderable code for lane-uniform aux loads used by those bounds. For example,
    a document mask like ``doc_offsets[b_idx] <= kv_idx and kv_idx < q_idx + 128``
    is analyzed over the packed lane expression ``kv_idx + mask_lane`` and
    becomes the interval ``[doc_offsets[b_idx] - kv_idx, q_idx + 128 - kv_idx)``.

    nonnegative_indices: FX index placeholders that do not need wraparound handling.
    q_idx: FX placeholder for the scalar query index.
    kv_idx: FX placeholder for the first KV index in the packed 32-lane window.
    q_symbol: SymPy symbol used for q_idx during interval analysis.
    kv_symbol: SymPy symbol used for the packed window base KV index.
    lane_symbol: SymPy symbol for the lane offset within the packed window.
    symbol_codes: CuTe renderings for temporary symbols introduced for aux loads.
    placeholder_codes: CuTe renderings for lane-uniform ABI and captured placeholders.
    placeholder_exprs: SymPy expressions for captured scalar placeholders.
    aux_load_symbols: Stable symbols assigned to supported lane-uniform aux loads.
    next_symbol_id: Counter for naming temporary aux-load symbols.
    """

    nonnegative_indices: tuple[torch.fx.Node, ...]
    q_idx: torch.fx.Node
    kv_idx: torch.fx.Node
    q_symbol: sympy.Symbol = dataclasses.field(
        default_factory=lambda: sympy.Symbol("q_idx", integer=True, nonnegative=True)
    )
    kv_symbol: sympy.Symbol = dataclasses.field(
        default_factory=lambda: sympy.Symbol("kv_idx", integer=True, nonnegative=True)
    )
    lane_symbol: sympy.Symbol = dataclasses.field(
        default_factory=lambda: sympy.Symbol(
            "mask_lane", integer=True, nonnegative=True
        )
    )
    symbol_codes: dict[sympy.Symbol, str] = dataclasses.field(default_factory=dict)
    placeholder_codes: Mapping[torch.fx.Node, str] = dataclasses.field(
        default_factory=dict
    )
    placeholder_exprs: Mapping[torch.fx.Node, sympy.Expr] = dataclasses.field(
        default_factory=dict
    )
    aux_load_symbols: dict[torch.fx.Node, sympy.Symbol] = dataclasses.field(
        default_factory=dict
    )
    next_symbol_id: int = 0

    def node_to_intervals(self, node: torch.fx.Node) -> MaybeIntervalSet:
        """Lower one supported boolean FX node into packed mask intervals."""
        if is_bool_full_node(node, True):
            return (PackedMaskInterval.full(),)
        if is_bool_full_node(node, False):
            return ()
        if node.op != "call_function":
            return None
        match node.target:
            case torch.ops.aten.bitwise_and.Tensor | torch.ops.aten.logical_and.default:
                return self.combine_binary_intervals(node, intersect=True)
            case torch.ops.aten.bitwise_or.Tensor | torch.ops.aten.logical_or.default:
                return self.combine_binary_intervals(node, intersect=False)
            case torch.ops.aten.le.Tensor | torch.ops.aten.le.Scalar:
                return self.comparison_to_intervals(
                    node.args[0], node.args[1], strict=False
                )
            case torch.ops.aten.lt.Tensor | torch.ops.aten.lt.Scalar:
                return self.comparison_to_intervals(
                    node.args[0], node.args[1], strict=True
                )
            case torch.ops.aten.ge.Tensor | torch.ops.aten.ge.Scalar:
                return self.comparison_to_intervals(
                    node.args[1], node.args[0], strict=False
                )
            case torch.ops.aten.gt.Tensor | torch.ops.aten.gt.Scalar:
                return self.comparison_to_intervals(
                    node.args[1], node.args[0], strict=True
                )
            case torch.ops.aten.eq.Tensor | torch.ops.aten.eq.Scalar:
                return self.equality_to_intervals(node.args[0], node.args[1])
            case torch.ops.aten.ne.Tensor | torch.ops.aten.ne.Scalar:
                return self.complement_intervals(
                    self.equality_to_intervals(node.args[0], node.args[1])
                )
            case (
                torch.ops.aten.logical_not.default | torch.ops.aten.bitwise_not.default
            ):
                child = node.args[0]
                if not isinstance(child, torch.fx.Node):
                    return None
                return self.complement_intervals(self.node_to_intervals(child))
            case _:
                return None

    def combine_binary_intervals(
        self, node: torch.fx.Node, *, intersect: bool
    ) -> MaybeIntervalSet:
        lhs, rhs = node.args
        if not isinstance(lhs, torch.fx.Node) or not isinstance(rhs, torch.fx.Node):
            return None
        lhs_intervals = self.node_to_intervals(lhs)
        rhs_intervals = self.node_to_intervals(rhs)
        if intersect:
            return self.intersect_interval_sets(lhs_intervals, rhs_intervals)
        return self.union_interval_sets(lhs_intervals, rhs_intervals)

    def comparison_to_intervals(
        self,
        lhs: object,
        rhs: object,
        *,
        strict: bool,
    ) -> MaybeIntervalSet:
        exprs = self.mask_operands_to_sympy(lhs, rhs)
        if exprs is None:
            return None
        return self.lane_comparison_to_intervals(*exprs, strict=strict)

    def equality_to_intervals(self, lhs: object, rhs: object) -> MaybeIntervalSet:
        exprs = self.mask_operands_to_sympy(lhs, rhs)
        if exprs is None:
            return None
        return self.lane_equality_to_intervals(*exprs)

    def mask_operands_to_sympy(
        self, lhs: object, rhs: object
    ) -> tuple[sympy.Expr, sympy.Expr] | None:
        lhs_expr = self.fx_mask_expr_to_sympy(lhs)
        rhs_expr = self.fx_mask_expr_to_sympy(rhs)
        if lhs_expr is None or rhs_expr is None:
            return None
        return lhs_expr, rhs_expr

    def lane_comparison_to_intervals(
        self,
        lhs_expr: sympy.Expr,
        rhs_expr: sympy.Expr,
        *,
        strict: bool,
    ) -> MaybeIntervalSet:
        """Convert ``lhs_expr < rhs_expr`` or ``lhs_expr <= rhs_expr`` to an interval.

        The caller normalizes ``>`` and ``>=`` by swapping operands, so this method
        only has to lower ``lhs_expr - rhs_expr < 0`` when ``strict`` is true and
        ``lhs_expr - rhs_expr <= 0`` otherwise. The difference decomposes as
        ``lane_coeff * mask_lane + rest``, and each unit coefficient maps to a mask
        shape:

        - ``1``: kv on the small side, e.g. ``kv_idx < q_idx`` (causal-style) ->
          prefix interval ``[0, upper)``.
        - ``-1``: kv on the large side, e.g. ``start[b] <= kv_idx`` (window or
          document starts) -> suffix interval ``[lower, 32)``.
        - ``0``: no kv dependence, e.g. ``q_idx < seq_lens[b]`` (padding or other
          row/batch/head bounds) -> the whole window is kept or dropped. Truth is
          encoded as the upper bound ``32 * keep`` with ``keep`` positive iff the
          predicate holds; keep-mask rendering clamps bounds to ``[0, 32]``, so any
          positive multiple of 32 keeps all lanes and any non-positive value keeps
          none.

        Other coefficients (e.g. ``2 * kv_idx < q_idx``, dilated/strided patterns)
        describe non-interval lane sets and fall back to per-lane evaluation.
        """
        diff = V.graph.sizevars.simplify(lhs_expr - rhs_expr)
        affine = decompose_affine_lane_expr(diff, self.lane_symbol)
        if affine is None:
            return None
        lane_coeff, rest = affine
        match lane_coeff:
            case 0:
                keep = -rest if strict else 1 - rest
                upper = V.graph.sizevars.simplify(sympy.Integer(32) * keep)
                return self.interval_if_renderable(sympy.Integer(0), upper)
            case 1:
                upper = V.graph.sizevars.simplify(-rest if strict else -rest + 1)
                return self.interval_if_renderable(sympy.Integer(0), upper)
            case -1:
                lower = V.graph.sizevars.simplify(rest + 1 if strict else rest)
                return self.interval_if_renderable(lower, sympy.Integer(32))
            case _:
                return None

    def lane_equality_to_intervals(
        self, lhs_expr: sympy.Expr, rhs_expr: sympy.Expr
    ) -> MaybeIntervalSet:
        """Convert ``lhs_expr == rhs_expr`` to packed keep intervals.

        Matching ``FloorDiv`` expressions can describe a contiguous block, e.g.
        ``q_idx // B == kv_idx // B``, and are handled before the generic
        lane-affine path. Otherwise the difference decomposes as
        ``lane_coeff * mask_lane + rest``:

        - ``1`` or ``-1``: exactly one lane can satisfy the equality, e.g.
          ``kv_idx == q_idx + c`` -> singleton interval ``[v, v + 1)``.
        - ``0``: no kv dependence, e.g. ``head_type[h] == 0`` -> the whole window
          is kept or dropped. ``Min(32 * (1 - rest), 32 * (1 + rest))`` is >= 32
          iff ``rest == 0`` and non-positive otherwise, reusing the clamped
          all-or-none upper-bound encoding from the comparison case.

        Other coefficients are rejected rather than emitting strided or
        divisibility-dependent masks.
        """
        if isinstance(lhs_expr, FloorDiv) and isinstance(rhs_expr, FloorDiv):
            intervals = self._floor_div_equality_to_intervals(lhs_expr, rhs_expr)
            if intervals is not None:
                return intervals
        diff = V.graph.sizevars.simplify(lhs_expr - rhs_expr)
        affine = decompose_affine_lane_expr(diff, self.lane_symbol)
        if affine is None:
            return None
        lane_coeff, rest = affine
        match lane_coeff:
            case 0:
                scale = sympy.Integer(32)
                upper = V.graph.sizevars.simplify(
                    Min(scale * (1 - rest), scale * (1 + rest))
                )
                return self.interval_if_renderable(sympy.Integer(0), upper)
            case 1 | -1:
                lane_value = -rest if lane_coeff == 1 else rest
                lower = V.graph.sizevars.simplify(lane_value)
                upper = V.graph.sizevars.simplify(lane_value + 1)
                return self.interval_if_renderable(lower, upper)
            case _:
                return None

    def _floor_div_equality_to_intervals(
        self, lhs_expr: FloorDiv, rhs_expr: FloorDiv
    ) -> MaybeIntervalSet:
        """Convert matching block-id equality into a contiguous lane interval."""
        lhs_base, lhs_divisor = lhs_expr.args
        rhs_base, rhs_divisor = rhs_expr.args
        if lhs_divisor != rhs_divisor:
            return None
        if lhs_base == self.q_symbol and rhs_base == self.kv_symbol + self.lane_symbol:
            q_block = lhs_expr
        elif (
            rhs_base == self.q_symbol and lhs_base == self.kv_symbol + self.lane_symbol
        ):
            q_block = rhs_expr
        else:
            return None
        block_start = V.graph.sizevars.simplify(q_block * lhs_divisor - self.kv_symbol)
        block_end = V.graph.sizevars.simplify(block_start + lhs_divisor)
        return self.interval_if_renderable(block_start, block_end)

    def interval_if_renderable(
        self, lower: sympy.Expr, upper: sympy.Expr
    ) -> MaybeIntervalSet:
        if (
            sympy_to_cute_index(lower, self.symbol_codes) is not None
            and sympy_to_cute_index(upper, self.symbol_codes) is not None
        ):
            return (PackedMaskInterval(lower, upper),)
        return None

    def complement_intervals(self, intervals: MaybeIntervalSet) -> MaybeIntervalSet:
        """Complement a keep-set within the 32-lane window via De Morgan.

        The complement of a union of intervals is the intersection of each
        interval's two complement pieces, which stays interval-representable.
        Interval bounds may be symbolic, so this composes existing renderable
        bounds instead of sorting; intersect_interval_sets caps the blowup.
        """
        if intervals is None:
            return None
        result: MaybeIntervalSet = (PackedMaskInterval.full(),)
        for interval in intervals:
            result = self.intersect_interval_sets(
                result,
                (
                    PackedMaskInterval(sympy.Integer(0), interval.lower_lane),
                    PackedMaskInterval(
                        interval.upper_lane_exclusive, sympy.Integer(32)
                    ),
                ),
            )
        return result

    def union_interval_sets(
        self, intervals: MaybeIntervalSet, new_intervals: MaybeIntervalSet
    ) -> MaybeIntervalSet:
        """Union supported interval sets while enforcing the code-size cap."""
        if intervals is None or new_intervals is None:
            return None
        if (
            len(intervals) + len(new_intervals)
            > MAX_PACKED_MASK_INTERVALS_FOR_CODE_SIZE
        ):
            return None
        return intervals + new_intervals

    def intersect_interval_sets(
        self,
        intervals: MaybeIntervalSet,
        new_intervals: MaybeIntervalSet,
    ) -> MaybeIntervalSet:
        """Intersect supported interval sets while enforcing the code-size cap."""
        if intervals is None or new_intervals is None:
            return None
        if (
            len(intervals) * len(new_intervals)
            > MAX_PACKED_MASK_INTERVALS_FOR_CODE_SIZE
        ):
            return None
        return tuple(
            PackedMaskInterval(
                V.graph.sizevars.simplify(Max(lhs.lower_lane, rhs.lower_lane)),
                V.graph.sizevars.simplify(
                    Min(lhs.upper_lane_exclusive, rhs.upper_lane_exclusive)
                ),
            )
            for lhs, rhs in product(intervals, new_intervals)
        )

    def fx_mask_expr_to_sympy(self, expr: object) -> sympy.Expr | None:
        """Translate a mask expression with kv_idx expanded by mask_lane."""
        index_symbols = {
            self.q_idx: self.q_symbol,
            self.kv_idx: self.kv_symbol + self.lane_symbol,
        }
        index_symbols.update(self.placeholder_exprs)
        return fx_aux_index_to_sympy(expr, index_symbols, self.mask_aux_load_to_symbol)

    def mask_aux_load_to_symbol(self, node: torch.fx.Node) -> sympy.Expr | None:
        """Assign a temporary SymPy symbol to a supported aux tensor load."""
        if not is_aten_index_node(node):
            return None
        if node in self.aux_load_symbols:
            return self.aux_load_symbols[node]
        lane_uniform_code = self.render_lane_uniform_scalar_expr(node)
        if lane_uniform_code is None:
            return None
        symbol = sympy.Symbol(f"mask_bound_{self.next_symbol_id}", integer=True)
        self.next_symbol_id += 1
        self.symbol_codes[symbol] = lane_uniform_code
        self.aux_load_symbols[node] = symbol
        return symbol

    def render_lane_uniform_scalar_expr(
        self,
        expr: object,
        *,
        for_index: bool = False,
        index_dim_size: int | sympy.Expr | None = None,
    ) -> str | None:
        """Render an expression that is invariant across the packed KV lanes."""
        if isinstance(expr, int | sympy.Integer):
            index = int(expr)
            if for_index and index < 0:
                if index_dim_size is None:
                    return None
                index = V.graph.sizevars.guard_int(index + index_dim_size)
            return f"cutlass.Int32({index})"
        if not isinstance(expr, torch.fx.Node):
            return None

        if expr is self.kv_idx:
            return None
        if expr in self.placeholder_codes:
            return self.placeholder_codes[expr]

        if is_aten_index_node(expr):
            return self._render_lane_uniform_index_expr(expr, for_index=for_index)
        if expr.op != "call_function":
            return None
        match expr.target:
            case torch.ops.aten.div.Tensor_mode:
                if expr.kwargs.get("rounding_mode") != "floor":
                    return None
                op = "//"
            case _:
                op = LANE_UNIFORM_BINARY_OPS.get(expr.target)
                if op is None:
                    return None
        args = expr.args
        if len(args) < 2:
            return None
        lhs = self.render_lane_uniform_scalar_expr(args[0])
        rhs = self.render_lane_uniform_scalar_expr(args[1])
        if lhs is None or rhs is None:
            return None
        return f"({lhs} {op} {rhs})"

    def _render_lane_uniform_index_expr(
        self,
        expr: torch.fx.Node,
        *,
        for_index: bool,
    ) -> str | None:
        """Render a scalar integer aten.index load that is uniform across lanes."""
        if fx_node_dtype(expr) not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            return None
        result_shape = fx_node_shape(expr)
        if result_shape is not None and len(result_shape) != 0:
            return None
        base, indices = expr.args
        base_code = self.render_lane_uniform_scalar_expr(base)
        base_shape = fx_node_shape(base) if isinstance(base, torch.fx.Node) else None
        if base_code is None or not isinstance(indices, (list, tuple)):
            return None
        if base_shape is None or len(indices) != len(base_shape):
            return None
        index_codes = []
        for dim, index in enumerate(indices):
            dim_size = base_shape[dim]
            index_code = self.render_lane_uniform_scalar_expr(
                index, for_index=True, index_dim_size=dim_size
            )
            if index_code is None:
                return None
            if (
                dim_size is not None
                and isinstance(index, torch.fx.Node)
                and index not in self.nonnegative_indices
            ):
                if not isinstance(dim_size, int | sympy.Integer):
                    return None
                dtype = fx_node_dtype(index)
                integer_type = "Int64" if dtype == torch.int64 else "Int32"
                size_code = f"cutlass.{integer_type}({dim_size})"
                zero_code = f"cutlass.{integer_type}(0)"
                index_code = (
                    f"({index_code} + {size_code} "
                    f"if {index_code} < {zero_code} else {index_code})"
                )
            index_codes.append(index_code)
        load = f"{base_code}[{', '.join(index_codes)}]"
        if for_index and fx_node_dtype(expr) == torch.int64:
            return load
        return f"cutlass.Int32({load})"


@dataclasses.dataclass(frozen=True)
class PackedMaskAuxPlaceholderMap:
    """Maps mask_mod aux placeholders to runtime scalar expressions or tensors."""

    codes: Mapping[torch.fx.Node, str]
    exprs: Mapping[torch.fx.Node, sympy.Expr]

    @classmethod
    def from_placeholders(
        cls,
        placeholders: Sequence[torch.fx.Node],
        other_buffers: Sequence[object],
        symbol_codes: Mapping[sympy.Symbol, str],
    ) -> "PackedMaskAuxPlaceholderMap | None":
        """Render lane-uniform mask_mod captures for packed interval analysis."""
        codes: dict[torch.fx.Node, str] = {}
        exprs: dict[torch.fx.Node, sympy.Expr] = {}
        tensor_idx = 0
        for placeholder, buffer in zip(placeholders, other_buffers):
            if isinstance(buffer, sympy.Expr):
                rendered = sympy_to_cute_index(buffer, symbol_codes)
                if rendered is None:
                    return None
                codes[placeholder] = rendered
                exprs[placeholder] = buffer
            else:
                codes[placeholder] = f"aux_tensors[{tensor_idx}]"
                tensor_idx += 1
        for placeholder in placeholders[len(other_buffers) :]:
            codes[placeholder] = f"aux_tensors[{tensor_idx}]"
            tensor_idx += 1
        return cls(codes, exprs)


def select_packed_mask_intervals(
    graph_module: GraphModule,
    other_buffers: Sequence[object] = (),
    aux_scalar_symbol_codes: Mapping[sympy.Symbol, str] | None = None,
) -> MaybeIntervalSet:
    """Entry point for selecting packed mask intervals from a mask_mod graph.

    Returns ``None`` when the graph is unsupported, when interval packing would
    increase code size too much, or when analysis reduces the whole mask to the
    unconditional ``[0, 32)`` interval.
    """
    graph = graph_module.graph
    nodes = list(graph.nodes)
    placeholders = [node for node in nodes if node.op == "placeholder"]
    output = [node for node in nodes if node.op == "output"]
    if len(placeholders) < 4 or len(output) != 1:
        return None

    b_idx, h_idx, q_idx, kv_idx, *aux_placeholders = placeholders
    output_val = output[0].args[0]
    if not isinstance(output_val, torch.fx.Node):
        return None

    symbol_codes = dict(aux_scalar_symbol_codes or {})
    placeholder_map = PackedMaskAuxPlaceholderMap.from_placeholders(
        aux_placeholders, other_buffers, symbol_codes
    )
    if placeholder_map is None:
        return None

    analyzer = PackedMaskAnalyzer(
        nonnegative_indices=(b_idx, h_idx, q_idx),
        q_idx=q_idx,
        kv_idx=kv_idx,
        symbol_codes=symbol_codes,
        placeholder_codes={
            b_idx: "b_idx[0]",
            h_idx: "h_idx[0]",
            q_idx: "q_idx[0]",
            **placeholder_map.codes,
        },
        placeholder_exprs=placeholder_map.exprs,
    )
    intervals = analyzer.node_to_intervals(output_val)
    if intervals is None:
        perf_hint_log.info(
            "flex FLASH: mask_mod could not be lowered to packed 32-lane intervals; "
            "partial blocks will evaluate the mask per lane, which can be much "
            "slower. Prefer comparisons of q_idx/kv_idx against lane-uniform bounds "
            "(e.g. seq_lens[b]) over per-token gathers. mask graph:\n%s",
            graph,
        )
        return None
    if len(intervals) == 1 and intervals[0].is_full():
        return None
    return tuple(
        interval.with_symbol_codes(analyzer.symbol_codes) for interval in intervals
    )
<|endoftext|>"""CuTeDSL geometry helpers shared by NVGEMM reduction kernels."""

__all__ = ["get_lane_warp_layouts", "partition_for_epilogue"]


def get_lane_warp_layouts(tiled_copy, reference_src=True):
    """Derive two-dimensional lane and warp layouts for an epilogue copy."""
    import cutlass.cute as cute

    from torch._vendor.quack import layout_utils

    layout_tv = (
        tiled_copy.layout_src_tv_tiled
        if reference_src
        else tiled_copy.layout_dst_tv_tiled
    )
    ref_layout = cute.right_inverse(layout_tv)
    tile_m = cute.size(tiled_copy.tiler_mn[0])
    tile_n = cute.size(tiled_copy.tiler_mn[1])
    ref_layout_mn = cute.composition(ref_layout, cute.make_layout((tile_m, tile_n)))

    num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
    tv_to_lane = cute.make_layout((cute.arch.WARP_SIZE, num_warps, 1), stride=(1, 0, 0))
    ref_to_lane = cute.composition(tv_to_lane, ref_layout_mn)
    lane_m = cute.filter(cute.select(ref_to_lane, [0]))
    lane_n = cute.filter(cute.select(ref_to_lane, [1]))
    lane_layout_mn = layout_utils.concat_layout(lane_m, lane_n)

    tv_to_warp = cute.make_layout((cute.arch.WARP_SIZE, num_warps, 1), stride=(0, 1, 0))
    ref_to_warp = cute.composition(tv_to_warp, ref_layout_mn)
    warp_m = cute.filter(cute.select(ref_to_warp, [0]))
    warp_n = cute.filter(cute.select(ref_to_warp, [1]))
    warp_layout_mn = layout_utils.concat_layout(warp_m, warp_n)
    return lane_layout_mn, warp_layout_mn


def partition_for_epilogue(*args, **kwargs):
    from torch._vendor.quack.sm90_utils import partition_for_epilogue as partition

    return partition(*args, **kwargs)
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

import types
import typing

import cuda.bindings.driver as cuda  # pyrefly: ignore [missing-import]

import cutlass
import cutlass.cute as cute
import cutlass.operators.providers.cutedsl.evt.common_efc as common_efc
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.operators.providers.cutedsl.evt.common_efc import log

from torch._inductor.kernel.gemm_epilogue import GemmReductionDescriptor
from torch._inductor.kernel.vendored_templates.cutedsl.reduction_utils import (
    get_lane_warp_layouts,
    partition_for_epilogue,
)
from torch.utils._ordered_set import OrderedSet


"""
Common base infrastructure for high-performance persistent batched dense GEMM with custom epilogue fusion
for the NVIDIA Blackwell SM100 architecture using CUTE DSL and Epilogue Fusion Configuration (EFC).

This module provides the PersistentDenseGemmEFCKernel base class that implements the core GEMM
functionality with support for custom epilogue operations. Subclasses define specific epilogue
configurations by providing an epilogue function that operates on the accumulator and supplemental
tensors.

Key Features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations
    - Utilizes Blackwell's tcgen05.mma for matrix multiply-accumulate (MMA) operations (including 2CTA MMA instructions)
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Supports persistent tile scheduling to better overlap memory load/store with MMA across tiles
    - Supports warp specialization to avoid explicit pipelining between mainloop load and MMA
    - Uses Epilogue Fusion Configuration (EFC) to define custom epilogue operations

GEMM Execution Flow:
1. DMA warp: Load A and B matrices from global memory (GMEM) to shared memory (SMEM) using TMA operations.
2. MMA warp: Perform matrix multiply-accumulate (MMA) operations using tcgen05.mma instruction.
3. EPILOGUE warp (customizable via EFC):
    - Load completed accumulator from tensor memory (TMEM) to registers (RMEM) using tcgen05.ld.
    - Load supplemental input tensors from GMEM to SMEM using TMA, then to RMEM.
    - Execute custom epilogue function (defined by subclass) that:
      * Accesses the accumulator via efc_config.accum()
      * Reads from supplemental input tensors via tensor.load()
      * Writes to supplemental output tensors via tensor.store()
      * Can apply arbitrary element-wise operations, scaling, and fusion
    - Store result tensors from RMEM to SMEM to GMEM with TMA operations.

SM100 tcgen05.mma instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Write accumulator to TMEM
The accumulator in TMEM must then be loaded to registers before writing back to GMEM.

Base Tensor Dimensions:
- Matrix A is MxKxL, L is batch dimension, A can be row-major("K") or column-major("M")
- Matrix B is NxKxL, L is batch dimension, B can be row-major("N") or column-major("K")
- Supplemental tensors are MxNxL with layout matching the epilogue configuration

Common Constraints:
* Supported input data types: fp16, bf16, tf32, int8, uint8, fp8 (e4m3fn, e5m2)
* A/B tensors must have the same data type
* MMA tiler M must be 64/128 (use_2cta_instrs=False) or 128/256 (use_2cta_instrs=True)
* MMA tiler N must be 32-256, step 32
* Cluster shape M/N must be positive and power of 2, total cluster size <= 16
* Cluster shape M must be multiple of 2 if use_2cta_instrs=True
* The contiguous dimension of all tensors must be at least 16 bytes aligned,
  i.e., number of elements is a multiple of 4, 8, and 16 for TFloat32,
  Float16/BFloat16, and Int8/Uint8/Float8, respectively.
* OOB tiles are not allowed when TMA store is disabled

Subclass Examples:
- custom_epilogue_dense_gemm.py: Custom fused epilogue with multiple read/write tensors
- broadcast_custom_epilogue_dense_gemm.py: Broadcasting and mode remapping with transposed tensors
- activation_custom_epilogue_dense_gemm.py: Activation functions with Ada FP8 GEMM epilogue pattern
- synthetic_custom_epilogue_dense_gemm.py: Synthetic epilogue for testing with configurable tensor counts

Writing an Epilogue Function (EFC User Guide)
==============================================

The Epilogue Fusion Configuration (EFC) lets you define custom epilogue
operations as a plain Python function. The framework analyzes the
function signature, JIT-compiles it for the GPU kernel, and also
executes it on the CPU with PyTorch for automatic verification.

Function Signature
------------------

The epilogue function must follow this contract:

    def epilogue(efc_config, <parameter_1>, <parameter_2>, ...):
        ...

- The first parameter **must** be named ``efc_config``. It is an
  ``EFC.Configuration`` instance that provides access to the GEMM
  accumulator, activation functions, and mode remapping.
- The remaining parameters are user-defined. Their names must match
  the supplemental arguments passed to the GEMM ``compile()`` call.
- **Tensor** parameters correspond to ``cutlass.cute.Tensor`` values
  and support ``.load()`` / ``.store()`` operations.
- **Scalar** parameters (``float``, ``int``, etc.) are used directly
  in arithmetic expressions without ``.load()``.
- The function does **not** return a value. Results are written via
  ``.store()`` on tensor parameters.  At least one ``.store()`` call
  is required for a useful computation.

Accessing the Accumulator
~~~~~~~~~~~~~~~~~~~~~~~~~

``efc_config.accum()`` returns the GEMM accumulator (the result of
A * B). It can be called multiple times within the same epilogue::

    def epilogue(efc_config, D):
        D.store(efc_config.accum())

Loading and Storing Tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor parameters expose two operations:

- ``tensor.load()`` — reads the tensor value (marks it as an input).
- ``tensor.store(value)`` — writes a computed value (marks it as an output).

A tensor can be both loaded and stored (read-modify-write). When
using multiple tensors, all ``.load()`` calls should be issued before
any ``.store()`` calls::

    def epilogue(efc_config, C, D, alpha, beta):
        D.store(efc_config.accum() * alpha + C.load() * beta)

Activation Functions
~~~~~~~~~~~~~~~~~~~~

The ``efc_config`` object provides built-in activation functions:

- ``efc_config.identity(x)`` — f(x) = x
- ``efc_config.relu(x)`` — f(x) = max(0, x)
- ``efc_config.leaky_relu(x, negative_slope=0.01)``
- ``efc_config.tanh(x)``
- ``efc_config.sigmoid(x)`` — f(x) = 1 / (1 + exp(-x))
- ``efc_config.silu(x)`` — f(x) = x * sigmoid(x) (Swish)
- ``efc_config.hardswish(x)`` — f(x) = x * relu6(x + 3) / 6
- ``efc_config.gelu(x)``

Example::

    def epilogue(efc_config, C, D, alpha, beta):
        D.store(efc_config.relu(efc_config.accum() * alpha + C.load() * beta))

Element-wise Utilities
~~~~~~~~~~~~~~~~~~~~~~

- ``efc_config.maximum(x, y)`` — element-wise maximum
- ``efc_config.minimum(x, y)`` — element-wise minimum

Additionally, any method called on ``efc_config`` that is not
explicitly defined is dispatched to ``cutlass.cute.<name>()`` on GPU
and ``torch.<name>()`` during PyTorch verification. This gives access
to functions like ``efc_config.abs(x)``, ``efc_config.exp(x)``,
``efc_config.sqrt(x)``, ``efc_config.clamp(x, min, max)``,
``efc_config.full_like(x, value)``, etc.

Broadcasting and Mode Remapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When supplemental tensors have fewer dimensions or a different mode
ordering than the output tensor, use ``tensor.remap_modes[...]`` to
broadcast or transpose them::

    tensor.remap_modes[<subscript>]

The target shape is implicitly the output shape ``(m, n, l)`` derived
from the GEMM input matrices ``A(m, k, l)`` and ``B(n, k, l)``.

The subscript uses Python's ``__getitem__`` syntax where each element
is either:

- An **integer** — the index of the source mode to place at this
  position.
- ``:`` — broadcast the source along this dimension (stride 0).

The returned remapped tensor proxy supports ``.load()`` and
``.store()`` just like regular tensor parameters.

Examples (output is (M, N, L)):

- **Broadcast along M** — ``C`` is (N, L)::

    C.remap_modes[:, 0, 1].load()

- **Broadcast along N** — ``X`` is (M, L)::

    X.remap_modes[0, :, 1].load()

- **Scalar broadcast** — ``x_factor`` is a 0-d tensor::

    x_factor.remap_modes[:, :, :].load()

- **Transpose** — ``Y`` is (N, M, L)::

    Y.remap_modes[1, 0, 2].store(value)

A remapped tensor can also be both loaded and stored for
read-modify-write on a transposed layout::

    remapped_Y = Y.remap_modes[1, 0, 2]
    remapped_Y.store(efc_config.accum() + remapped_Y.load())

Complete Examples
~~~~~~~~~~~~~~~~~

**Alpha-beta scaling** (simplest useful epilogue)::

    def epilogue(efc_config, C, D, alpha, beta):
        D.store(efc_config.accum() * alpha + C.load() * beta)

**Multi-tensor with activation**::

    def epilogue(efc_config, C, D, alpha, beta, X, x_factor, Y):
        Y.store(efc_config.accum())
        result = (
            efc_config.relu(efc_config.accum() * alpha + C.load() * beta)
            + X.load() * x_factor
        )
        D.store(result)

**Ada FP8 GEMM pattern with dynamic activation**::

    def epilogue(efc_config, C, Aux, alpha, beta, bias,
                 scale_a, scale_b, scale_c, D, leaky_relu_alpha):
        aux_val = (
            (alpha * scale_a * scale_b) * efc_config.accum()
            + (beta * scale_c) * C.load()
            + bias
        )
        Aux.store(aux_val)
        D.store(efc_config.relu(aux_val))

**Broadcasting with mode remapping**::

    def epilogue(efc_config, C, D, alpha, beta, X, x_factor, Y):
        # Y is (N, M, L), output is (M, N, L): transpose via [1, 0, 2]
        Y.remap_modes[1, 0, 2].store(efc_config.accum())
        # C is (N, L): broadcast along M via [:, 0, 1]
        # X is (M, L): broadcast along N via [0, :, 1]
        result = (
            efc_config.relu(
                efc_config.accum() * alpha
                + C.remap_modes[:, 0, 1].load() * beta
            )
            + X.remap_modes[0, :, 1].load()
            * x_factor.remap_modes[:, :, :].load()
        )
        D.store(result)
"""


class PersistentDenseGemmEFCKernel:
    """Base class for batched GEMM with custom epilogue fusion using EFC.

    This class provides the core infrastructure for persistent batched GEMM operations
    with customizable epilogue fusion. Subclasses define specific epilogue behaviors
    by providing an epilogue configuration function that describes operations on the
    accumulator and supplemental tensors.

    The class handles:
    - GEMM mainloop (A * B computation)
    - TMA-based memory operations
    - Warp specialization
    - Persistent tile scheduling
    - EFC (Epilogue Fusion Configuration) integration
    - Tensor creation and validation

    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param epi_dtype: Data type for epilogue operation
    :type epi_dtype: type[cutlass.Numeric]
    :param use_2cta_instrs: Whether to use CTA group 2 for advanced thread cooperation
    :type use_2cta_instrs: bool
    :param mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tile (M,N)
    :type mma_tiler_mn: tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: tuple[int, int]
    :param epilogue_function_configuration: Function defining the epilogue behavior via EFC
    :type epilogue_function_configuration: Callable

    :note: Supported A/B data types:
        - TFloat32
        - Float16/BFloat16
        - Int8/Uint8
        - Float8E4M3FN/Float8E5M2
        (A and B must have the same data type)

    :note: Supported accumulator data types:
        - Float32 (for all floating point A/B data types)
        - Float16 (only for fp16 and fp8 A/B data types)
        - Int32 (only for uint8/int8 A/B data types)

    :note: Supported supplemental tensor data types (epilogue-dependent):
        - Float32 (for float32 and int32 accumulator data types)
        - Int32 (for float32 and int32 accumulator data types)
        - Float16/BFloat16 (for fp16 and fp8 accumulator data types)
        - Int8/Uint8 (for uint8/int8 accumulator data types)
        - Float8E4M3FN/Float8E5M2 (for float32 accumulator data types)

    :note: Constraints:
        - MMA tiler M must be 64/128 (use_2cta_instrs=False) or 128/256 (use_2cta_instrs=True)
        - MMA tiler N must be 32-256, step 32
        - Cluster shape M must be multiple of 2 if use_2cta_instrs=True
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16
    """

    def __init__(
        self,
        acc_dtype: type[cutlass.Numeric],
        epi_dtype: type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
        epilogue_function_configuration: typing.Callable,
    ):
        """Initializes the configuration for a Blackwell dense GEMM kernel with EFC.

        This configuration includes several key aspects:

        1.  MMA Instruction Settings (tcgen05):
            - acc_dtype: Data type for MMA accumulator.
            - mma_tiler_mn: The (M, N) shape of the MMA instruction tiler.
            - use_2cta_instrs: Boolean indicating if the tcgen05 MMA variant
              with cta_group=2 should be used.

        2.  Cluster Shape:
            - cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster.

        3.  Epilogue Configuration:
            - epilogue_function_configuration: Defines custom epilogue behavior
              that operates on accumulator and supplemental tensors.

        :param acc_dtype: Data type of the accumulator.
        :type acc_dtype: type[cutlass.Numeric]
        :param epi_dtype: Data type of the epilogue.
        :type epi_dtype: type[cutlass.Numeric]
        :param use_2cta_instrs: Boolean, True to use cta_group=2 MMA variant.
        :type use_2cta_instrs: bool
        :param mma_tiler_mn: tuple (M, N) shape of the MMA instruction.
        :type mma_tiler_mn: tuple[int, int]
        :param cluster_shape_mn: tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mn: tuple[int, int]
        :param epilogue_function_configuration: Function defining epilogue behavior via EFC.
        :type epilogue_function_configuration: Callable
        """
        self.acc_dtype: type[cutlass.Numeric] = acc_dtype
        self.epi_dtype: type[cutlass.Numeric] = epi_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        # K dimension is deferred in _setup_attributes
        self.mma_tiler_mn = mma_tiler_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.arch = "sm_100"

        self.c_dtype = self.epi_dtype

        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.occupancy = 1
        # Set specialized warp ids:

        # The warps responsible for computing the epilogue function and storing
        # the results.
        self.epilogue_warp_id = (0, 1, 2, 3)
        # The warp responsible for computing the matrix multiplication.
        self.mma_warp_id = 4
        # The warp responsible for loading the tensors A & B to feed the MMA.
        self.tma_warp_id = 5
        # The warp responsible for loading the auxiliary tensors used in the epilogue.
        self.epilogue_load_warp_id = 6
        self.threads_per_cta = 32 * len(
            (
                self.mma_warp_id,
                self.tma_warp_id,
                *self.epilogue_warp_id,
                self.epilogue_load_warp_id,
            )
        )
        # Barrier ids for cta sync, epilogue sync and tmem ptr sync.
        self.epilogue_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3
        # Amount of available shared memory.
        self.smem_capacity = utils.get_smem_capacity_in_bytes(self.arch)

        # Setup the EFC from the given function representing the epilogue
        # configuration.
        self.efc = common_efc.EFC(self, epilogue_function_configuration)

    def _create_tiled_mma(self):
        """Make a tiled MMA atom with given data type, leading dimension, CTA
        group and MMA tile shape. Use SMEM operand source for A.
        """
        return utils.sm100.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/C/D stage counts in shared memory
        - Computing A/B/C/D shared memory layout
        - Computing tensor memory allocation columns
        """
        # Get the right tiled MMA.
        self._tiled_mma = self._create_tiled_mma()
        log(f"{self._tiled_mma = !s}")

        # Compute mma/cluster/tile shapes
        mma_inst_shape_k = cute.size(self._tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        # Extend mma_tiler with k-dimension (MMA_M, MMA_N, MMA_K)
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        log(f"{self.mma_tiler = !s}")
        # CTA tiler with the 2CTA instruction correction.
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(self._tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        log(f"{self.cta_tile_shape_mnk = !s}")
        # Compute cluster layout, V for the 2CTA instructions.
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (self._tiled_mma.thr_id.shape,),
        )
        log(f"{cute.make_layout((*self.cluster_shape_mn, 1)) = !s}")
        log(f"{self.cluster_layout_vmnk = !s}")
        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        log(f"{self.num_mcast_ctas_a = }, {self.num_mcast_ctas_b = }")
        log(f"{self.is_a_mcast = }, {self.is_b_mcast = }")

        # Compute epilogue (EPI_TILE_M, EPI_TILE_N) subtile of cta_tile_shape_mnk
        # according to some heuristics.
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            layout_d=self.d_layout,
            elem_ty_d=self.d_dtype,
            layout_c=self.c_layout,
            elem_ty_c=self.c_dtype,
        )
        self.epi_tile_m = cute.size(self.epi_tile[0])
        self.epi_tile_n = cute.size(self.epi_tile[1])
        self.has_cross_warp_local_reduce = cutlass.const_expr(
            (self.has_local_reduce_tensor or self.local_reduce_feeds_main)
            and (
                (
                    self.local_reduce_axis == 0
                    and (self.local_reduce_group > 4 or self.local_reduce_feeds_main)
                )
                or (
                    self.local_reduce_axis == 1
                    and self.local_reduce_group > self.epi_tile_n
                )
            )
        )
        self.local_reduce_smem_rows = (
            self.cta_tile_shape_mnk[1]
            if self.local_reduce_axis == 0
            else self.cta_tile_shape_mnk[0]
        )
        self.local_reduce_smem_cols = (
            3 + self.cta_tile_shape_mnk[0] // self.local_reduce_group
            if self.local_reduce_feeds_main and self.local_reduce_axis == 0
            else max(
                3,
                self.local_reduce_group // self.epi_tile_n
                if self.local_reduce_axis == 1
                else 3,
            )
        )
        self.local_reduce_smem_bytes = (
            self.local_reduce_smem_rows * self.local_reduce_smem_cols * 4
            if self.has_cross_warp_local_reduce
            else 0
        )
        log(f"{self.epi_tile = !s}")

        # Setup A/B/C/D pipeline stage count in shared memory and ACC stage
        # count in tensor memory.
        self.compute_stages()
        log(f"{self.num_acc_stage = }, {self.num_ab_stage = }, {self.num_c_stage = }")
        # Compute A/B shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            self._tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        log(f"{self.a_smem_layout_staged = !s}")
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            self._tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        log(f"{self.b_smem_layout_staged = !s}")
        # Get the smem_layout for the tensors used in the EFC.
        self.efc.jit.smem_layout()

        # Compute the number of tensor memory allocation columns
        self.compute_num_tmem_alloc_cols()

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        local_reduce_tensor: cute.Tensor,
        local_reduce_feed_tensor: cute.Tensor,
        local_reduce_secondary_feed_tensor: cute.Tensor,
        local_reduce_group: cutlass.Constexpr,
        local_reduce_axis: cutlass.Constexpr,
        local_reduce_type: cutlass.Constexpr,
        local_reduce_source: cutlass.Constexpr,
        local_reduce_feeds_main: cutlass.Constexpr,
        local_reduce_secondary_feed_type: cutlass.Constexpr,
        local_reduce_op: cutlass.Constexpr,
        local_reduce_init: cutlass.Constexpr,
        local_reduce_combine: cutlass.Constexpr,
        local_reduce_source_op: cutlass.Constexpr,
        local_reduce_finalize: cutlass.Constexpr,
        *supplemental_parameters,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes before smem/grid/tma computation
        - Setup TMA load/store atoms and tensors
        - Compute grid size with regard to hardware constraints
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param a: Input tensor A in (L, M, K) or (M, K) layout.
        :type a: cute.Tensor
        :param b: Input tensor B in (L, K, N) or (K, N) layout.
        :type b: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr
        :param stream: CUDA stream for asynchronous execution.
        :type stream: cuda.CUstream
        :param supplemental_parameters: Variadic epilogue parameters (tensors and scalars).
        :raises TypeError: If input data types are incompatible with the MMA instruction.
        :raises AssertionError: If OOB (Out-Of-Bounds) tiles are present when TMA store is disabled.

        """

        def add_batch_mode(tensor: cute.Tensor) -> cute.Tensor:
            return cute.make_tensor(
                tensor.iterator,
                cute.prepend(tensor.layout, cute.make_layout(1), up_to_rank=3),
            )

        a = add_batch_mode(a)
        b = add_batch_mode(b)
        if cutlass.const_expr(local_reduce_tensor is not None):
            local_reduce_tensor = add_batch_mode(local_reduce_tensor)
        if cutlass.const_expr(local_reduce_feed_tensor is not None):
            local_reduce_feed_tensor = add_batch_mode(local_reduce_feed_tensor)
            local_reduce_feed_tensor = cute.make_tensor(
                local_reduce_feed_tensor.iterator,
                cute.select(local_reduce_feed_tensor.layout, [1, 2, 0]),
            )
        if cutlass.const_expr(local_reduce_secondary_feed_tensor is not None):
            local_reduce_secondary_feed_tensor = add_batch_mode(
                local_reduce_secondary_feed_tensor
            )
            local_reduce_secondary_feed_tensor = cute.make_tensor(
                local_reduce_secondary_feed_tensor.iterator,
                cute.select(local_reduce_secondary_feed_tensor.layout, [1, 2, 0]),
            )

        # Permute tensor modes from torch to cute convention
        # A: (L, M, K) -> (M, K, L)
        a = cute.make_tensor(a.iterator, cute.select(a.layout, [1, 2, 0]))
        # B: (L, K, N) -> (N, K, L)
        b = cute.make_tensor(b.iterator, cute.select(b.layout, [2, 1, 0]))
        self.local_reduce_group = local_reduce_group
        self.local_reduce_axis = local_reduce_axis
        self.local_reduce_input_coefficient = 1.0
        self.local_reduce_result_coefficient = 0.0
        self.local_reduce_bias = 0.0
        self.local_reduce_affine_scale = 1.0
        self.local_reduce_affine_bias = 0.0
        self.local_reduce_denominator_scale = 1.0
        self.local_reduce_denominator_bias = 0.0
        self.local_reduce_secondary_scale = 1.0
        self.local_reduce_secondary_bias = 0.0
        self.local_reduce_variance_scale = 1.0
        self.local_reduce_variance_bias = 0.0
        self.local_reduce_direct_bool = False
        self.local_reduce_secondary_reuses_primary = False
        secondary_expression = (
            GemmReductionDescriptor.parse(local_reduce_secondary_feed_type)
            if local_reduce_secondary_feed_type is not None
            else None
        )
        secondary_parameters = (
            () if secondary_expression is None else secondary_expression.parameters
        )
        if cutlass.const_expr(
            secondary_expression is not None
            and secondary_expression.kind == "sum_mul_affine"
        ):
            secondary_scale, secondary_bias = secondary_parameters
            self.local_reduce_secondary_scale = float(secondary_scale)
            self.local_reduce_secondary_bias = float(secondary_bias)
        elif cutlass.const_expr(
            secondary_expression is not None
            and secondary_expression.kind == "direct_bool_gt_zero"
        ):
            self.local_reduce_direct_bool = True
        elif cutlass.const_expr(local_reduce_secondary_feed_type is not None):
            self.local_reduce_secondary_reuses_primary = True
        reduction_expression = GemmReductionDescriptor.parse(local_reduce_type)
        local_reduce_type = reduction_expression.kind
        if cutlass.const_expr(local_reduce_type == "mean_linear"):
            input_coefficient, result_coefficient, bias = (
                reduction_expression.parameters
            )
            self.local_reduce_input_coefficient = float(input_coefficient)
            self.local_reduce_result_coefficient = float(result_coefficient)
            self.local_reduce_bias = float(bias)
        elif cutlass.const_expr(local_reduce_type == "variance_affine"):
            variance_scale, variance_bias = reduction_expression.parameters
            local_reduce_type = "variance"
            self.local_reduce_variance_scale = float(variance_scale)
            self.local_reduce_variance_bias = float(variance_bias)
        elif cutlass.const_expr(local_reduce_type == "normalize_sum_affine"):
            (
                affine_scale,
                affine_bias,
                denominator_scale,
                denominator_bias,
            ) = reduction_expression.parameters
            local_reduce_type = "normalize_sum"
            self.local_reduce_affine_scale = float(affine_scale)
            self.local_reduce_affine_bias = float(affine_bias)
            self.local_reduce_denominator_scale = float(denominator_scale)
            self.local_reduce_denominator_bias = float(denominator_bias)
        elif cutlass.const_expr(local_reduce_type == "normalize_sum_reverse_affine"):
            (
                affine_scale,
                affine_bias,
                denominator_scale,
                denominator_bias,
            ) = reduction_expression.parameters
            local_reduce_type = "normalize_sum_reverse"
            self.local_reduce_affine_scale = float(affine_scale)
            self.local_reduce_affine_bias = float(affine_bias)
            self.local_reduce_denominator_scale = float(denominator_scale)
            self.local_reduce_denominator_bias = float(denominator_bias)
        self.local_reduce_type = local_reduce_type
        self.local_reduce_source = local_reduce_source
        self.local_reduce_feeds_main = local_reduce_feeds_main
        self.local_reduce_op = local_reduce_op
        self.local_reduce_init = local_reduce_init
        self.local_reduce_combine = local_reduce_combine
        self.local_reduce_source_op = local_reduce_source_op
        self.local_reduce_finalize = local_reduce_finalize
        self.has_local_reduce_tensor = local_reduce_tensor is not None
        # Identify parameters that are remap_modes sources (broadcast tensors).
        # These skip add_batch_mode + permute entirely — they are passed as-is
        # to the EFC and remap_modes inside the kernel handles all the layout
        # work (broadcast expansion, dimension reordering).
        remap_sources = OrderedSet(
            [
                attrs.mapped_source
                for attrs in self.efc.parameter_attributes.values()
                if attrs.mapped_source
            ]
        )

        # Add batch mode and permute epilogue parameters, except for
        # broadcast remap sources which are passed untouched.
        param_names = self.efc.epilogue_parameter_names
        supplemental_parameters = (
            # Broadcast source: pass through untouched.
            t
            if name in remap_sources
            # Regular tensor: (L, M, N) -> rank 3 -> (M, N, L).
            else cute.make_tensor(
                add_batch_mode(t).iterator,
                cute.select(add_batch_mode(t).layout, [1, 2, 0]),
            )
            if isinstance(t, cute.Tensor)
            # Scalar: pass through.
            else t
            for name, t in zip(param_names, supplemental_parameters)
        )

        # Process the variadic parameters.
        self.efc.jit.unpack_parameters(supplemental_parameters)

        # - Matrix A is MxKxL, L is batch dimension, A can be row-major("K") or column-major("M")
        # - Matrix B is NxKxL, L is batch dimension, B can be row-major("N") or column-major("K")
        # - Supplemental tensors are MxNxL with layout matching the epilogue configuration

        # The output shape (m, n, l) is derived from A(m, k, l) and B(n, k, l).
        self.efc.output_shape = (a.shape[0], b.shape[0], a.shape[2])
        self.efc.jit.handle_remapping()

        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: type[cutlass.Numeric] = a.element_type
        self.b_dtype: type[cutlass.Numeric] = b.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()

        # Gather all the auxiliary tensor element data types.
        self.efc.jit.record_tensor_dtypes()

        # There is no D tensor to be used as a returned tensor. In the
        # following, D is used more like a "store" concept. So use the
        # written tensor with the biggest element_type to set up all the tiling
        # heuristics and epilogue store pipeline.
        self.d_name_bigger = self.efc.jit.written_tensor_name_with_bigger_element_type()
        d = self.efc.jit.parameter[self.d_name_bigger]
        self.d_dtype: type[cutlass.Numeric] = d.element_type
        self.d_layout = utils.LayoutEnum.from_tensor(d)
        log(f"d{self.d_name_bigger} = {d!s}")

        # C is the read tensor with the biggest element_type, if any, used by
        # some heuristics for tiling.
        self.c_dtype = typing.cast(type[cutlass.Numeric], None)
        self.c_layout = None
        self.c_name_bigger = self.efc.jit.read_tensor_name_with_bigger_element_type()
        if cutlass.const_expr(self.c_name_bigger):
            # The tensor with the biggest data type might be a broadcast vector.
            c = self.efc.jit.get_remapped_tensor_or_itself(self.c_name_bigger)
            log(f"{self.c_name_bigger = } -> {c = !s}")
            self.c_dtype = c.element_type
            # If only broadcast tensors are read, there is no leading dimension
            # defined, so use the same as the output.
            try:
                self.c_layout = utils.LayoutEnum.from_tensor(c)
            except ValueError:
                self.c_layout = self.d_layout

        # Check if input data types are compatible with MMA instruction
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Types must match: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes that depend on gemm inputs
        self._setup_attributes()

        atom_thr_size = cute.size(self._tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = utils.sm100.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, self._tiled_mma.thr_id
        )
        log(f"{a_op = !s}")
        # Get rid of the pipeline dimension.
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        log(f"{a_smem_layout = !s}")
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            self._tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(
                cutlass.TFloat32 if a.element_type is cutlass.Float32 else None
            ),
        )
        log(f"{tma_atom_a = !s}")
        log(f"{tma_tensor_a = !s}")
        # Setup TMA load for B
        b_op = utils.sm100.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, self._tiled_mma.thr_id
        )
        log(f"{b_op = !s}")
        # Get rid of the pipeline dimension.
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        log(f"{b_smem_layout = !s}")
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            self._tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(
                cutlass.TFloat32 if b.element_type is cutlass.Float32 else None
            ),
        )
        log(f"{tma_atom_b = !s}")
        log(f"{tma_tensor_b = !s}")
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size
        log(f"{self.num_tma_load_bytes = }")

        # Set the TMA related arguments for the tensors used in the EFC.
        self.efc.jit.create_tma_arguments()

        # Compute grid size
        self.tile_sched_params, grid = self._compute_grid(
            d, self.cta_tile_shape_mnk, self.cluster_shape_mn, max_active_clusters
        )

        self.efc.jit.create_supplemental_arguments_for_kernel()

        # Launch the kernel synchronously
        self.kernel(
            self._tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            local_reduce_tensor,
            local_reduce_feed_tensor,
            local_reduce_secondary_feed_tensor,
            self.efc.kernel.pack_arguments(),
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        local_reduce_tensor: cute.Tensor,
        local_reduce_feed_tensor: cute.Tensor,
        local_reduce_secondary_feed_tensor: cute.Tensor,
        supplemental_parameters: tuple,
    ):
        """GPU device kernel performing the Persistent batched GEMM computation."""
        # Process the variadic parameters.
        self.efc.kernel.unpack_parameters(supplemental_parameters)

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch TMA descriptors
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            # Prefetch the TMA descriptors for all the supplemental tensors.
            self.efc.kernel.prefetch_tma_descriptors()

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
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        self.buffer_align_bytes = 1024

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            # Barriers used by the supplemental load tensor pipeline.
            c_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_c_stage]
            c_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_c_stage]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sLocalReduce: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32,
                    self.local_reduce_smem_rows * self.local_reduce_smem_cols
                    if self.has_cross_warp_local_reduce
                    else 1,
                ],
                16,
            ]

        self.shared_storage = SharedStorage

        self.smem = utils.SmemAllocator()
        storage = self.smem.allocate(self.shared_storage)
        sLocalReduce = storage.sLocalReduce.get_tensor(
            cute.make_layout((self.local_reduce_smem_rows, self.local_reduce_smem_cols))
            if self.has_cross_warp_local_reduce
            else cute.make_layout(1)
        )

        # Allocate the shared memory for all the supplemental tensors.
        self.efc.kernel.allocate_smem()

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilogue_warp_id) * (
            2 if self.use_2cta_instrs else 1
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

        # Load pipeline, used to load all the supplemental tensors of the
        # epilogue.
        c_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        c_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len(self.epilogue_warp_id),
        )
        c_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.c_full_mbar_ptr.data_ptr(),
            num_stages=self.num_c_stage,
            producer_group=c_producer_group,
            consumer_group=c_consumer_group,
            # Unlock the barrier when all the tensor bytes have been loaded.
            tx_count=self.efc.jit.total_tma_load_bytes,
            defer_sync=True,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        # Cluster arrive after barrier init
        pipeline.pipeline_init_arrive(
            cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True
        )

        #
        # Setup smem tensor A/B
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )

        #
        # Compute multicast mask for A/B buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(
            self.is_a_mcast or self.is_b_mcast or self.use_2cta_instrs
        ):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, loopM, loopK, loopL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, loopN, loopK, loopL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )

        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        #
        # Partition global tensor for TiledMMA_A/B/D
        #
        self.thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, loopM, loopK, loopL)
        tCgA = self.thr_mma.partition_A(gA_mkl)
        log(f"{tCgA = !s}")
        # (MMA, MMA_N, MMA_K, loopN, loopK, loopL)
        tCgB = self.thr_mma.partition_B(gB_nkl)
        log(f"{tCgB = !s}")
        # Create the local_tile gX_mnl for all the EFC supplemental tensors.
        self.efc.kernel.partition_global_tensors_for_tiled_mma()

        #
        # Partition global/shared tensor for TMA load A/B
        #
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), tiles_m, tiles_k, tiles_l)
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
        # ((atom_v, rest_v), tiles_n, tiles_k, tiles_l)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C/D
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        log(f"{tCrA = !s}")
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        log(f"{tCrB = !s}")
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        log(f"{acc_shape = !s}")
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )
        log(f"{tCtAcc_fake = !s}")

        # Named barriers
        #
        epilogue_sync_barrier = pipeline.NamedBarrier(
            self.epilogue_sync_bar_id, 32 * len(self.epilogue_warp_id)
        )

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline.pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

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
                # ((atom_v, rest_v), loopK)
                tAgA_slice = tAgA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                # ((atom_v, rest_v), loopK)
                tBgB_slice = tBgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                ab_producer.reset()
                peek_ab_empty_status = ab_producer.try_acquire()

                #
                # Tma load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Conditionally wait for AB buffer empty
                    handle = ab_producer.acquire_and_advance(peek_ab_empty_status)

                    # TMA load A/B
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, handle.count)],
                        tAsA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, handle.count)],
                        tBsB[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=b_full_mcast_mask,
                    )

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek_ab_empty_status = ab_producer.try_acquire()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Wait A/B buffer empty
            #
            ab_producer.tail()

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Bar sync to retrieve tensor memory ptr from shared mem
            #
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
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

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

                # Set tensor memory buffer for current tile
                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer.reset()
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    peek_ab_full_status = ab_consumer.try_wait()

                #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

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
                        handle = ab_consumer.wait_and_advance(peek_ab_full_status)

                        # tCtAcc += tCrA * tCrB
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                        tile_crd = (None, None, None, handle.index)
                        cute.gemm(
                            tiled_mma, tCtAcc, tCrA[tile_crd], tCrB[tile_crd], tCtAcc
                        )

                        # Async arrive AB buffer empty
                        handle.release()

                        # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                        peek_ab_full_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_full_status = ab_consumer.try_wait()

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
            # Bar sync to retrieve tensor memory ptr from shared memory
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            log(f"tmem_ptr = {tmem_ptr!s}")
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            log(f"tCtAcc_base = {tCtAcc_base!s}")
            #
            # Partition for epilogue
            #
            epi_tidx = tidx
            tCgD = self.efc.kernel.tCgD_written[self.d_name_bigger]
            log(f"tCgD (aka tCgD_written[{self.d_name_bigger}])= {tCgD!s}")

            (
                tiled_copy_t2r,  # (EPI_TILE_M, EPI_TILE_N)
                tTR_tAcc_base,  # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
                tTR_rAcc,  # (T2R, T2R_M, T2R_N)
            ) = self.epilogue_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, tCgD, epi_tile
            )
            log(f"{tiled_copy_t2r = !s}")
            log(f"{tTR_tAcc_base = !s}")
            log(f"{tTR_rAcc = !s}")
            # Copy and partition for the supplemental EFC tensors.
            self.efc.kernel.copy_and_partition_supplemental_rmem_tensors(
                tiled_copy_t2r, tTR_rAcc, epi_tidx, epi_tile
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

            # Store D pipeline used for all the written tensors in the epilogue.
            d_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilogue_warp_id),
            )
            d_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_d_stage,
                producer_group=d_producer_group,
            )

            c_pipeline_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_c_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # Slice the supplemental written tensors per MMA tile index.
                self.efc.kernel.slice_written_tensors_per_mma_tile_index(
                    mma_tile_coord_mnl
                )

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_consumer_state.index)
                ]
                log(f"tTR_tAcc = {tTR_tAcc!s}")
                #
                # Wait for accumulator buffer full
                #
