            CallFunction(aten.mm.default, *_mm_args, _users=1),
            Arg(),
            _users=1,
        ),
    ]

    """
    In order to fuse add/sub/mul/div with conv/linear, the dimensions of its
    constant tensor must satisfy the following:
    - with resizing, broadcast to w/ weight/bias tensor shape
    - broadcast to the conv/linear output shape
    It needs to have a shape that can resize to weight/bias
    tensor shape because we need to run the op with the conv/linear
    weights/bias without changing their sizes.
    It needs to broadcast to the conv/linear output shape so that we do
    accidentally change the shape of op output by pre-fusing it
    compared to eager.
    The only dimension value shared by weight, bias, and conv/linear output
    is they all contain a dim with value = channels-out. In the
    conv/linear output tensor, this is in the second dimension,
    so the pointwise op tensor may have a second dimension of
    value == channels-out, but all the other dimensions have to be 1
    """

    def _op_not_broadcasting_with_conv(weight_tensor, other_tensor):
        # According to opDoesNotBroadCastWithConv of frozen_conv_folding.cpp
        weight_shape = weight_tensor.shape
        other_shape = other_tensor.shape
        if len(weight_shape) < len(other_shape):
            return False
        if len(weight_shape) == len(other_shape) + 1:
            # weight shape is [o, i, *], other_shape is [o, 1...].
            for i in reversed(range(len(other_shape))):
                if i == 0 and weight_shape[0] == other_shape[i]:
                    continue
                if other_shape[i] != 1:
                    return False
        else:
            # weight shape is [o, i, *], other_shape is [1, i, *]
            for i in reversed(range(len(other_shape))):
                if i == 1 and weight_shape[0] == other_shape[i]:
                    continue
                if other_shape[i] != 1:
                    return False
        return True

    def _op_not_broadcasting_with_linear(weight_tensor, other_tensor, has_reshape):
        weight_shape = weight_tensor.shape
        other_shape = other_tensor.shape
        other_shapes = [
            torch.Size(
                [
                    weight_shape[1],
                ]
            ),
            torch.Size([1, weight_shape[1]]),
            torch.Size(
                [
                    1,
                ]
            ),
            torch.Size([1, 1]),
        ]
        if has_reshape:
            other_shapes.extend(
                [
                    torch.Size([1, 1, weight_shape[1]]),
                    torch.Size([1, 1, 1]),
                ]
            )
        return other_shape in other_shapes

    def _check_conv_and_broadcast_op(conv_node, other):
        # According to checkConvAndBroadcastingOpPreConditions of frozen_conv_folding.cpp.
        # conv.weight
        if conv_node.args[1].op != "get_attr":
            return False
        # conv.bias
        bias_node = conv_node.args[2]
        if bias_node is not None and bias_node.op != "get_attr":
            return False
        if (
            not isinstance(other, int)
            and not isinstance(other, float)
            and other.op != "get_attr"
        ):
            return False

        if len(conv_node.args[1].users) != 1:
            return False

        weight_meta_value = conv_node.args[1].meta.get("val")
        if weight_meta_value is None:
            return False
        # Avoid fusing op that causes type promotion
        # restricting to float avoids int/float difficulties with scalar overload
        if not weight_meta_value.is_floating_point():
            return False
        if isinstance(other, torch.fx.Node) and other.op == "get_attr":
            other_meta_value = other.meta.get("val")
            if not other_meta_value.is_floating_point():  # type: ignore[union-attr]
                return False
            if (
                torch.promote_types(other_meta_value.dtype, weight_meta_value.dtype)  # type: ignore[union-attr]
                != weight_meta_value.dtype
            ):
                if not conv_node.meta.get("_allow_mixed_dtype_folding", False):
                    return False

                if (
                    other_meta_value.dtype != torch.float  # type: ignore[union-attr]
                    and weight_meta_value.dtype not in (torch.float16, torch.bfloat16)
                ):
                    return False

            if not _op_not_broadcasting_with_conv(weight_meta_value, other_meta_value):
                return False
        elif not isinstance(other, float):
            return False

        return True

    def _check_linear_and_broadcast_op(linear_node, other, has_reshape):
        weight_node = (
            linear_node.args[2]
            if linear_node.target is aten.addmm.default
            else linear_node.args[1]
        )
        bias_node = (
            linear_node.args[0] if linear_node.target is aten.addmm.default else None
        )
        if weight_node.op != "get_attr":
            return False
        if bias_node is not None and bias_node.op != "get_attr":
            return False
        if (
            not isinstance(other, int)
            and not isinstance(other, float)
            and other.op != "get_attr"
        ):
            return False

        if len(weight_node.users) != 1:
            return False

        weight_meta_value = weight_node.meta.get("val")
        if weight_meta_value is None:
            return False
        # Avoid fusing op that causes type promotion
        # restricting to float avoids int/float difficulties with scalar overload
        if not weight_meta_value.is_floating_point():
            return False
        if isinstance(other, torch.fx.Node) and other.op == "get_attr":
            other_meta_value = other.meta.get("val")
            if not other_meta_value.is_floating_point():  # type: ignore[union-attr]
                return False
            if (
                torch.promote_types(other_meta_value.dtype, weight_meta_value.dtype)  # type: ignore[union-attr]
                != weight_meta_value.dtype
            ):
                if not linear_node.meta.get("_allow_mixed_dtype_folding", False):
                    return False

                if (
                    other_meta_value.dtype != torch.float  # type: ignore[union-attr]
                    and weight_meta_value.dtype not in (torch.float16, torch.bfloat16)
                ):
                    return False

            if not _op_not_broadcasting_with_linear(
                weight_meta_value, other_meta_value, has_reshape
            ):
                return False
        elif not isinstance(other, float):
            return False

        return True

    def _is_foldable_pattern(match):
        binary_node = match.output_node()
        has_reshape = False
        if binary_node.args[0].target in _computation_ops:
            computation_node = binary_node.args[0]
            other = binary_node.args[1]
        elif binary_node.args[0].target is aten.reshape.default:
            computation_node = binary_node.args[0].args[0]
            other = binary_node.args[1]
            has_reshape = True
        elif binary_node.args[1].target in _computation_ops:
            computation_node = binary_node.args[1]
            other = binary_node.args[0]
        else:
            computation_node = binary_node.args[1].args[0]
            other = binary_node.args[0]
            has_reshape = False
        if computation_node.target is aten.convolution.default:
            return _check_conv_and_broadcast_op(computation_node, other)
        elif computation_node.target in [aten.addmm.default, aten.mm.default]:
            return (
                config.enable_linear_binary_folding
                and _check_linear_and_broadcast_op(computation_node, other, has_reshape)
            )

        return False

    def resize_scalar_or_tensor_to_shape(graph, other, shape, weight):
        if isinstance(other, float):
            with torch.utils._python_dispatch._disable_current_modes():
                other_tensor = torch.tensor(
                    other, dtype=weight.dtype, device=weight.device
                )
            graph.owning_module.register_buffer("other_tensor", other_tensor)
            res = graph.create_node("get_attr", "other_tensor")
            res = graph.create_node(
                "call_function",
                aten.reshape.default,
                (res, (1,)),
            )
            res = graph.create_node(
                "call_function",
                aten.expand.default,
                (res, shape),
            )
        elif other.meta.get("val").numel() == 1:
            # expand errors if the shape input has less # dims than the tensor input
            res = graph.create_node(
                "call_function",
                aten.reshape.default,
                (other, (1,)),
            )
            res = graph.create_node(
                "call_function",
                aten.expand.default,
                (res, shape),
            )
        else:
            res = graph.create_node(
                "call_function",
                aten.reshape.default,
                (other, shape),
            )
        return res

    def _create_new_conv_node(graph, conv_node, binary_node, other):
        if conv_node.target is not aten.convolution.default:
            raise AssertionError(
                f"expected aten.convolution.default, got {conv_node.target}"
            )
        conv_args = list(conv_node.args)
        weight_meta_value = conv_node.args[1].meta.get("val")
        bias = conv_args[2]
        if binary_node.target in [aten.add.Tensor, aten.sub.Tensor]:
            other_reshape = resize_scalar_or_tensor_to_shape(
                graph,
                other,
                (weight_meta_value.size(0),),
                weight_meta_value,
            )
            new_bias = graph.create_node(
                "call_function",
                binary_node.target,
                (0 if bias is None else bias, other_reshape),
            )
            conv_args[2] = new_bias
        else:
            if binary_node.target not in [aten.mul.Tensor, aten.div.Tensor]:
                raise AssertionError(
                    f"expected mul or div binary node, got {binary_node.target}"
                )
            weight_broadcast_shape = [1 for _ in range(len(weight_meta_value.shape))]
            weight_broadcast_shape[0] = weight_meta_value.size(0)
            other_reshape1 = resize_scalar_or_tensor_to_shape(
                graph,
                other,
                tuple(weight_broadcast_shape),
                weight_meta_value,
            )
            new_weight = graph.create_node(
                "call_function", binary_node.target, (conv_args[1], other_reshape1)
            )
            new_weight.meta.update(conv_args[1].meta)
            conv_args[1] = new_weight
            if bias is not None:
                other_reshape = resize_scalar_or_tensor_to_shape(
                    graph,
                    other,
                    (weight_meta_value.size(0),),
                    weight_meta_value,
                )
                new_bias = graph.create_node(
                    "call_function", binary_node.target, (bias, other_reshape)
                )
                new_bias.meta.update(bias.meta)
                conv_args[2] = new_bias
        return graph.create_node("call_function", conv_node.target, tuple(conv_args))

    def _create_new_linear_node(graph, linear_node, binary_node, other):
        if linear_node.target not in [aten.addmm.default, aten.mm.default]:
            raise AssertionError(f"expected addmm or mm, got {linear_node.target}")
        input_node = (
            linear_node.args[1]
            if linear_node.target is aten.addmm.default
            else linear_node.args[0]
        )
        weight_node = (
            linear_node.args[2]
            if linear_node.target is aten.addmm.default
            else linear_node.args[1]
        )
        bias_node = (
            linear_node.args[0] if linear_node.target is aten.addmm.default else None
        )
        weight_meta_value = weight_node.meta.get("val")
        if binary_node.target in [aten.add.Tensor, aten.sub.Tensor]:
            other_reshape = resize_scalar_or_tensor_to_shape(
                graph,
                other,
                (weight_meta_value.size(1),),
                weight_meta_value,
            )
            new_bias_node = graph.create_node(
                "call_function",
                binary_node.target,
                (0 if bias_node is None else bias_node, other_reshape),
            )
            return graph.create_node(
                "call_function",
                aten.addmm.default,
                (new_bias_node, input_node, weight_node),
            )
        else:
            if binary_node.target not in [aten.mul.Tensor, aten.div.Tensor]:
                raise AssertionError(
                    f"expected mul or div binary node, got {binary_node.target}"
                )
            weight_broadcast_shape = [1, weight_meta_value.size(1)]
            other_reshape1 = resize_scalar_or_tensor_to_shape(
                graph,
                other,
                tuple(weight_broadcast_shape),
                weight_meta_value,
            )
            new_weight_node = graph.create_node(
                "call_function", binary_node.target, (weight_node, other_reshape1)
            )
            new_weight_node.meta.update(weight_node.meta)
            if bias_node is not None:
                other_reshape = resize_scalar_or_tensor_to_shape(
                    graph,
                    other,
                    (weight_meta_value.size(1),),
                    weight_meta_value,
                )
                new_bias_node = graph.create_node(
                    "call_function", binary_node.target, (bias_node, other_reshape)
                )
                new_bias_node.meta.update(bias_node.meta)
                return graph.create_node(
                    "call_function",
                    linear_node.target,
                    (new_bias_node, input_node, new_weight_node),
                )
            else:
                return graph.create_node(
                    "call_function", linear_node.target, (input_node, new_weight_node)
                )

    for _computation_call, binary_op in itertools.product(
        _computation_calls, _binary_ops
    ):

        @register_binary_folding_pattern(
            CallFunction(binary_op, _computation_call, KeywordArg("other")),
            extra_check=_is_foldable_pattern,
        )
        def folded_op(match, *args, **kwargs):
            counters["inductor"]["binary_folding"] += 1
            other = kwargs.get("other")
            binary_node = match.output_node()
            reshape_node = None
            if binary_node.args[0].target in _computation_ops:
                computation_node = binary_node.args[0]
            elif binary_node.args[0].target is aten.reshape.default:
                computation_node = binary_node.args[0].args[0]
                reshape_node = binary_node.args[0]
            elif binary_node.args[1].target in _computation_ops:
                computation_node = binary_node.args[1]
            else:
                computation_node = binary_node.args[1].args[0]
                reshape_node = binary_node.args[1]
            graph = match.graph
            with graph.inserting_before(reshape_node if reshape_node else binary_node):
                if computation_node.target not in _computation_ops:
                    raise AssertionError(
                        f"expected a computation op, got {computation_node.target}"
                    )
                if computation_node.target is aten.convolution.default:
                    counters["inductor"]["binary_folding_conv"] += 1
                    new_computation_node = _create_new_conv_node(
                        graph, computation_node, binary_node, other
                    )
                else:
                    new_computation_node = _create_new_linear_node(
                        graph, computation_node, binary_node, other
                    )
                new_computation_node.meta.update(computation_node.meta)
                if reshape_node:
                    if reshape_node.target is not aten.reshape.default:
                        raise AssertionError(
                            f"expected aten.reshape.default, got {reshape_node.target}"
                        )
                    computation_node.replace_all_uses_with(new_computation_node)
                    binary_node.replace_all_uses_with(reshape_node)
                else:
                    binary_node.replace_all_uses_with(new_computation_node)
                graph.erase_node(binary_node)
                graph.erase_node(computation_node)
<|endoftext|># mypy: allow-untyped-defs
import functools
from collections import deque

import torch
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_map

from ..._dynamo.utils import counters
from ..ir import (
    ComputedBuffer,
    FixedLayout,
    FlexibleLayout,
    InputBuffer,
    StorageBox,
    Subgraph,
    TensorBox,
)
from ..lowering import lowerings
from ..pattern_matcher import (
    Arg,
    CallFunction,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)
from ..select_algorithm import (
    autotune_select_algorithm,
    SymbolicGridFn,
    TritonTemplate,
    TritonTemplateCaller,
)
from ..utils import ceildiv


B2B_GEMM_PASS = PatternMatcherPass(
    pass_name="b2b_gemm_pass",
)


@SymbolicGridFn
def b2b_gemm_grid(M, P, meta, *, cdiv):
    return (cdiv(M, meta["BLOCK_SIZE_M"]) * cdiv(P, meta["BLOCK_SIZE_P"]), 1, 1)


b2b_gemm_left_template = TritonTemplate(
    name="b2b_gemm_left",
    grid=b2b_gemm_grid,
    debug=False,
    source=r"""
{{def_kernel("A", "B", "C")}}


    # B2B_GEMM_LEFT_TRITON_ENTRANCE

    # dynamic shapes
    M = {{size("A", 0)}}
    N = {{size("A", 1)}}
    O = {{size("C", 0)}}
    P = {{size("C", 1)}}

    # dynamic strides
    stride_am = {{stride("A", 0)}}
    stride_an = {{stride("A", 1)}}
    stride_bn = {{stride("B", 0)}}
    stride_bo = {{stride("B", 1)}}
    stride_co = {{stride("C", 0)}}
    stride_cp = {{stride("C", 1)}}

    # output block counts
    num_m_block = tl.cdiv(M, BLOCK_SIZE_M)
    num_p_block = tl.cdiv(P, BLOCK_SIZE_P)

    # internal block counts
    num_n_block = tl.cdiv(N, BLOCK_SIZE_N)
    num_o_block = tl.cdiv(O, BLOCK_SIZE_O)

    # output block ids
    pid = tl.program_id(axis=0)
    m_block_id = pid // num_p_block
    p_block_id = pid % num_p_block

    # accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_P), dtype=tl.float32)

    # main loop
    offs_m = (m_block_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_p = (p_block_id * BLOCK_SIZE_P + tl.arange(0, BLOCK_SIZE_P))
    # (subgraph(A @ B) @ C)
    offs_o = tl.arange(0, BLOCK_SIZE_O)
    for _ in range(num_o_block):
        c_mask = (offs_o[:, None] < O) & (offs_p[None, :] < P)
        c_ptrs = C + (offs_o[:, None] * stride_co + offs_p[None, :] * stride_cp)
        c = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)  # BLOCK_SIZE_O * BLOCK_SIZE_P
        acc_ab = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_O), dtype=tl.float32)
        offs_n = tl.arange(0, BLOCK_SIZE_N)
        for __ in range(num_n_block):
            a_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
            a_ptrs = A + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # BLOCK_SIZE_M * BLOCK_SIZE_N
            b_mask = (offs_n[:, None] < N) & (offs_o[None, :] < O)
            b_ptrs = B + (offs_n[:, None] * stride_bn + offs_o[None, :] * stride_bo)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # BLOCK_SIZE_N * BLOCK_SIZE_O
            acc_ab += tl.dot(a, b, out_dtype=tl.float32)
            offs_n += BLOCK_SIZE_N
        # apply the subgraph
        {{ modification(
            subgraph_number=0,
            output_name="post_subgraph_acc_ab",
            inner_mm="acc_ab"
        ) | indent_except_first(2) }}
        acc += tl.dot(post_subgraph_acc_ab, c, out_dtype=tl.float32)
        offs_o += BLOCK_SIZE_O

    # type conversion
    acc = acc.to(tl.float16)

    # store preparation
    idx_m = offs_m[:, None]
    idx_p = offs_p[None, :]
    out_mask = (idx_m < M) & (idx_p < P)

    {{store_output(("idx_m", "idx_p"), "acc", "out_mask", val_shape=("BLOCK_SIZE_M", "BLOCK_SIZE_P"))}}
""",
)


b2b_gemm_right_template = TritonTemplate(
    name="b2b_gemm_right",
    grid=b2b_gemm_grid,
    debug=False,
    source=r"""
{{def_kernel("A", "B", "C")}}


    # B2B_GEMM_RIGHT_TRITON_ENTRANCE

    # dynamic shapes
    M = {{size("A", 0)}}
    N = {{size("A", 1)}}
    O = {{size("C", 0)}}
    P = {{size("C", 1)}}

    # dynamic strides
    stride_am = {{stride("A", 0)}}
    stride_an = {{stride("A", 1)}}
    stride_bn = {{stride("B", 0)}}
    stride_bo = {{stride("B", 1)}}
    stride_co = {{stride("C", 0)}}
    stride_cp = {{stride("C", 1)}}

    # output block counts
    num_m_block = tl.cdiv(M, BLOCK_SIZE_M)
    num_p_block = tl.cdiv(P, BLOCK_SIZE_P)

    # internal block counts
    num_n_block = tl.cdiv(N, BLOCK_SIZE_N)
    num_o_block = tl.cdiv(O, BLOCK_SIZE_O)

    # output block ids
    pid = tl.program_id(axis=0)
    m_block_id = pid // num_p_block
    p_block_id = pid % num_p_block

    # accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_P), dtype=tl.float32)

    # main loop (two cases)
    offs_m = (m_block_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_p = (p_block_id * BLOCK_SIZE_P + tl.arange(0, BLOCK_SIZE_P))
    # (A @ subgraph(B @ C))
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    for _ in range(num_n_block):
        a_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        a_ptrs = A + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # BLOCK_SIZE_M * BLOCK_SIZE_N
        acc_bc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_P), dtype=tl.float32)
        offs_o = tl.arange(0, BLOCK_SIZE_O)
        for __ in range(num_o_block):
            b_mask = (offs_n[:, None] < N) & (offs_o[None, :] < O)
            b_ptrs = B + (offs_n[:, None] * stride_bn + offs_o[None, :] * stride_bo)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # BLOCK_SIZE_N * BLOCK_SIZE_O
            c_mask = (offs_o[:, None] < O) & (offs_p[None, :] < P)
            c_ptrs = C + (offs_o[:, None] * stride_co + offs_p[None, :] * stride_cp)
            c = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)  # BLOCK_SIZE_O * BLOCK_SIZE_P
            acc_bc += tl.dot(b, c, out_dtype=tl.float32)
            offs_o += BLOCK_SIZE_O
        # apply the subgraph
        {{ modification(
            subgraph_number=0,
            output_name="post_subgraph_acc_bc",
            inner_mm="acc_bc"
        ) | indent_except_first(2) }}
        acc += tl.dot(a, post_subgraph_acc_bc, out_dtype=tl.float32)
        offs_n += BLOCK_SIZE_N

    # type conversion
    acc = acc.to(tl.float16)

    # store preparation
    idx_m = offs_m[:, None]
    idx_p = offs_p[None, :]
    out_mask = (idx_m < M) & (idx_p < P)

    {{store_output(("idx_m", "idx_p"), "acc", "out_mask", val_shape=("BLOCK_SIZE_M", "BLOCK_SIZE_P"))}}
""",
)


# Note: load_ratio_left and load_ratio_right are only calculating numbers
# in the trivial subgraph case; i.e. (A @ (B @ C)) or ((A @ B) @ C)


def load_ratio_left(
    M: int, N: int, O: int, P: int, m: int, n: int, o: int, p: int
) -> float:
    """
    compute the ratio of estimated numbers of loads in baseline and b2bgemm
    M, N, O, P are matrix sizes
    m, n, o, p are block sizes
    |       | baseline (lower bound)        | b2bgemm
    | load  | M * N + N * O + M * O + O * P | M / m * P / p * O / o * (o * p + N / n * (m * n + n * o))
    | store | M * O + M * P                 | M * P
    b2bgemm is always better on stores, but for loads we need to find out beneficial cases using this function
    """
    base = M * N + N * O + M * O + O * P
    gemm = (
        ceildiv(M, m)
        * ceildiv(P, p)
        * ceildiv(O, o)
        * (o * p + ceildiv(N, n) * (m * n + n * o))
    )
    return base / gemm


def load_ratio_right(
    M: int, N: int, O: int, P: int, m: int, n: int, o: int, p: int
) -> float:
    """
    compute the ratio of estimated numbers of loads in baseline and b2bgemm
    M, N, O, P are matrix sizes
    m, n, o, p are block sizes
    |       | baseline (lower bound)        | b2bgemm
    | load  | N * O + O * P + M * N + N * P | M / m * P / p * N / n * (m * n + O / o * (n * o + o * p))
    | store | N * P + M * P                 | M * P
    b2bgemm is always better on stores, but for loads we need to find out beneficial cases using this function
    """
    base = N * O + O * P + M * N + N * P
    gemm = (
        ceildiv(M, m)
        * ceildiv(P, p)
        * ceildiv(N, n)
        * (m * n + ceildiv(O, o) * (n * o + o * p))
    )
    return base / gemm


# the block sizes are limited by hardware (the shared memory)
# intuitively, the optimization works when the intermediate matrix is large
# and we assign large block sizes to large dimensions
b2b_gemm_configs = [
    {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 16,
        "BLOCK_SIZE_O": 16,
        "BLOCK_SIZE_P": 16,
        "num_stages": 4,
        "num_warps": 8,
    },
    {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_O": 32,
        "BLOCK_SIZE_P": 32,
        "num_stages": 2,
        "num_warps": 4,
    },
    {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_O": 64,
        "BLOCK_SIZE_P": 64,
        "num_stages": 2,
        "num_warps": 4,
    },
    {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 16,
        "BLOCK_SIZE_O": 128,
        "BLOCK_SIZE_P": 16,
        "num_stages": 4,
        "num_warps": 8,
    },
    {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_O": 128,
        "BLOCK_SIZE_P": 32,
        "num_stages": 2,
        "num_warps": 4,
    },
    {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_O": 128,
        "BLOCK_SIZE_P": 64,
        "num_stages": 2,
        "num_warps": 4,
    },
    {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 16,
        "BLOCK_SIZE_O": 16,
        "BLOCK_SIZE_P": 128,
        "num_stages": 4,
        "num_warps": 8,
    },
    {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_O": 32,
        "BLOCK_SIZE_P": 128,
        "num_stages": 2,
        "num_warps": 4,
    },
    {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_O": 64,
        "BLOCK_SIZE_P": 128,
        "num_stages": 2,
        "num_warps": 4,
    },
    {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_O": 16,
        "BLOCK_SIZE_P": 128,
        "num_stages": 4,
        "num_warps": 8,
    },
    {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_O": 32,
        "BLOCK_SIZE_P": 128,
        "num_stages": 2,
        "num_warps": 4,
    },
    {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_O": 64,
        "BLOCK_SIZE_P": 128,
        "num_stages": 2,
        "num_warps": 4,
    },
]


def is_b2b_gemm_good_on(
    is_left_assoc: bool,
    A_node: torch.fx.Node,
    B_node: torch.fx.Node,
    C_node: torch.fx.Node,
) -> bool:
    """
    checks whether the sizes are good for b2b_gemm
    """
    # basic checks
    if not all(["val" in A_node.meta, "val" in B_node.meta, "val" in C_node.meta]):
        return False
    fake_tensors = (
        A_node.meta["val"],
        B_node.meta["val"],
        C_node.meta["val"],
    )  # torch._subclasses.fake_tensor.FakeTensor

    A, B, C = fake_tensors

    def check_all_attr_true(objects, attr):
        return all(hasattr(obj, attr) and getattr(obj, attr) for obj in objects)

    if not check_all_attr_true(fake_tensors, "is_cuda") and not check_all_attr_true(
        fake_tensors, "is_xpu"
    ):
        return False
    if not all([len(A.shape) == 2, len(B.shape) == 2, len(C.shape) == 2]):
        return False
    if not ((A.shape[1] == B.shape[0]) and (B.shape[1] == C.shape[0])):
        return False
    # size checks: we only dispatch to B2B-GEMM when the average load ratio is > 1
    M, N = A.shape
    O, P = C.shape
    ratios = []
    if is_left_assoc:
        for config in b2b_gemm_configs:
            ratio = load_ratio_left(
                M,
                N,
                O,
                P,
                config["BLOCK_SIZE_M"],
                config["BLOCK_SIZE_N"],
                config["BLOCK_SIZE_O"],
                config["BLOCK_SIZE_P"],
            )
            ratios.append(ratio)
    else:
        for config in b2b_gemm_configs:
            ratio = load_ratio_right(
                M,
                N,
                O,
                P,
                config["BLOCK_SIZE_M"],
                config["BLOCK_SIZE_N"],
                config["BLOCK_SIZE_O"],
                config["BLOCK_SIZE_P"],
            )
            ratios.append(ratio)
    ratios.sort(reverse=True)
    average_ratio = 1.0
    for r in ratios[:3]:  # top 3 choices
        average_ratio *= r
    average_ratio = average_ratio ** (1 / 3)
    return (
        average_ratio > 1
    )  # even if average_ratio is close to 1, the number of stores is always better


def _make_unoptimized_b2b_gemm_choice(
    is_left_assoc: bool,
    subgraph: Subgraph,
    input_nodes: list[TensorBox],
    layout: FixedLayout,
):  # -> SubgraphChoiceCaller (local import)
    from ..codegen.subgraph import SubgraphChoiceCaller

    tag = "left" if is_left_assoc else "right"
    epilogue = subgraph.graph_module

    def make_fx_graph(*args: torch.Tensor) -> torch.fx.GraphModule:
        from torch.fx.experimental.proxy_tensor import make_fx

        from ..decomposition import select_decomp_table

        A, B, C = args

        def computation(A, B, C):
            if is_left_assoc:
                return torch.mm(epilogue(torch.mm(A, B)), C)
            else:
                return torch.mm(A, epilogue(torch.mm(B, C)))

        return make_fx(
            computation,
            decomposition_table=select_decomp_table(),
            tracing_mode="symbolic",
        )(A, B, C)

    return SubgraphChoiceCaller(
        name=f"unoptimized_b2b_gemm_{tag}",
        # pyrefly: ignore [bad-argument-type]
        input_nodes=input_nodes,
        layout=layout,
        description=f"unoptimized b2b_gemm ({tag}-associative)",
        make_fx_graph=make_fx_graph,
    )


def build_subgraph_buffer(
    args: list[TensorBox],
    subgraph: Subgraph,
):
    """
    This function is adapted from ../kernel/flex_attention.py.
    The goal is to take in the required args and produce the subgraph buffer
    The subgraph buffer is a ComputedBuffer that will be inlined into the triton template

    Args:
        args: The args that are passed into the subgraph
        subgraph: The Subgraph ir for which to produce the output node
    """
    cnt = 0
    env = {}
    for node in subgraph.graph_module.graph.nodes:
        if node.op == "placeholder":
            env[node] = args[cnt]
            cnt += 1
        elif node.op == "call_function":
            # For call_function we use the default lowerings and pass in the
            # already created TensorBoxes as args
            args, kwargs = tree_map(lambda x: env.get(x, x), (node.args, node.kwargs))
            env[node] = lowerings[node.target](*args, **kwargs)
        elif node.op == "output":

            def convert_output_node_to_buffer(output):
                if output is None:
                    return None
                output_node = output
                output_buffer = env[output_node]
                if not isinstance(output_buffer, TensorBox):
                    raise AssertionError(
                        (
                            "The output node for B2B-GEMM's subgraph must be a TensorBox, but got: ",
                            type(output_buffer),
                        )
                    )
                if not isinstance(output_buffer.data, StorageBox):
                    raise AssertionError(
                        (
                            "The output node for B2B-GEMM's subgraph must be a StorageBox, but got: ",
                            type(output_buffer),
                        )
                    )
                device = output_buffer.data.get_device()
                if device is None:
                    raise AssertionError("expected output buffer to have a device")
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

            # node.args[0] should be a single element representing the output of the subgraph
            return tree_map(convert_output_node_to_buffer, node.args[0])

    raise ValueError("B2B-GEMM was passed a subgraph with no output node!")


def create_placeholder(
    name: str, dtype: torch.dtype, device: torch.device
) -> TensorBox:
    """
    Creates a placeholder input buffers for producing subgraph_output
    """
    input_buffer = InputBuffer(name=name, layout=FixedLayout(device, dtype, [], []))
    return TensorBox.create(input_buffer)


def tuned_b2b_gemm(
    is_left_assoc: bool,
    subgraph: Subgraph,
    A: torch._inductor.ir.TensorBox,
    B: torch._inductor.ir.TensorBox,
    C: torch._inductor.ir.TensorBox,
    *,
    layout=None,
) -> torch._inductor.ir.TensorBox:
    # call .realize() to get rid of Pointwise
    A.realize()
    B.realize()
    C.realize()
    layout = FixedLayout(
        A.get_device_or_error(),
        A.get_dtype(),
        [A.shape[0], C.shape[1]],  # type: ignore[index]
    )
    placeholders = [
        create_placeholder("inner_mm", A.get_dtype(), A.get_device_or_error())
    ]
    subgraph_buffer = build_subgraph_buffer(
        placeholders,  # type: ignore[arg-type, list-item]
        subgraph,
    )
    choices: list[TritonTemplateCaller] = []
    for config in b2b_gemm_configs:
        if is_left_assoc:
            b2b_gemm_left_template.maybe_append_choice(
                choices,
                input_nodes=(A, B, C),
                layout=layout,
                subgraphs=[subgraph_buffer],
                **config,
            )
        else:
            b2b_gemm_right_template.maybe_append_choice(
                choices,
                input_nodes=(A, B, C),
                layout=layout,
                subgraphs=[subgraph_buffer],
                **config,
            )
    # add the unoptimized choice to mitigate performance degradation
    choices.append(
        _make_unoptimized_b2b_gemm_choice(is_left_assoc, subgraph, [A, B, C], layout)
    )
    # autotune
    node, _ = autotune_select_algorithm("b2b_gemm", choices, [A, B, C], layout)
    return node


# match the inner mm of a potential b2b_gemm
@register_graph_pattern(
    CallFunction(torch.ops.aten.mm, Arg(), Arg()),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=B2B_GEMM_PASS,
)
def b2b_gemm_handler(match: Match, mat1: torch.fx.Node, mat2: torch.fx.Node) -> None:
    # match.args: list[torch.fx.Node]

    def is_pointwise_node(node: torch.fx.Node) -> bool:
        return (
            node.op == "call_function"
            and isinstance(node.target, torch._ops.OpOverload)
            and (torch.Tag.pointwise in node.target.tags)
        )

    def is_mm(node: torch.fx.Node) -> bool:
        return node.target is torch.ops.aten.mm.default

    # the inner MM
    inner_mm = match.nodes[-1]

    # find the (candidate) outer MM, which will be re-checked below to ensure every path reaches it
    # In a real (A @ f(B @ C)), every path starting from (B @ C) must reach (A @ _).
    outer_mm = None
    node = inner_mm
    while len(node.users) > 0:
        node = next(iter(node.users))
        if is_mm(node):
            outer_mm = node
            break
        elif is_pointwise_node(node):
            continue
        else:
            break
    if not outer_mm:
        return

    # find the unique input node for outer_mm representing f(B @ C) in (A @ f(B @ C))
    # we call it the "f_node"
    # when the pattern is simply (A @ (B @ C)), f_node is just inner_mm
    f_node = inner_mm
    while next(iter(f_node.users)) is not outer_mm:
        f_node = next(iter(f_node.users))

    def all_reach_via_pointwise_with_no_other_inputs(
        src: torch.fx.Node,
        dst: torch.fx.Node,
    ) -> tuple[bool, OrderedSet[torch.fx.Node]]:
        """
        check whether every user path from src reaches dst via pointwise nodes,
        with no other input nodes for the intermediates and dst;
        return
        (1) the Boolean value
        (2) the subgraph node set including src and dst (which only makes sense when the Boolean value is True)
        """
        visited = OrderedSet[torch.fx.Node]()
        input_counter: dict[torch.fx.Node, int] = {}

        all_reachable = True
        queue = deque([src])
        while queue:
            node = queue.popleft()
            if node not in visited:
                if node is dst:
                    visited.add(node)
                elif (node is src) or is_pointwise_node(node):
                    for user in node.users:
                        # for nodes other than dst, bookkeep their users' input counts
                        if user not in input_counter:
                            input_counter[user] = len(user.all_input_nodes)
                        input_counter[user] -= 1
                        # continue BFS
                        queue.append(user)
                    visited.add(node)
                else:
                    all_reachable = False
                    break

        return (
            all_reachable and all(count == 0 for count in input_counter.values()),
            visited,
        )

    # check inner_mm reaches f_node on every user path via pointwise nodes with no outside input_nodes
    ok, subgraph_node_set = all_reach_via_pointwise_with_no_other_inputs(
        inner_mm, f_node
    )
    if not ok:
        return

    # check inner_mm's inputs and f_node's outputs
    if not (len(inner_mm.all_input_nodes) == 2 and len(f_node.users) == 1):
        return

    # at this point, the nodes between inner_mm and f_node (both included)
    # are all used internally inside (A @ subgraph(B @ C))
    # i.e. they neither have other users nor have other inputs

    # original graph and module
    graph = inner_mm.graph
    module = graph.owning_module
    if module is None:
        raise AssertionError("graph.owning_module must not be None")

    # construct the new (sub)graph
    subgraph_node_list: list[
        torch.fx.Node
    ] = []  # ordered list of nodes used for node removal later
    new_graph: torch.fx.Graph = torch.fx.Graph()
    node_remapping: dict[torch.fx.Node, torch.fx.Node] = {}
    new_input_anchor: torch.fx.Node  # inner_mm, to be changed to an input node
    new_output_anchor: torch.fx.Node  # f_node, to be used to construct an output node
    new_input_node: torch.fx.Node
    new_output_node: torch.fx.Node
    for node in graph.nodes:  # preserve the order of nodes
        if node in subgraph_node_set:
            subgraph_node_list.append(node)
            new_node = new_graph.node_copy(node, lambda x: node_remapping.get(x, x))
            node_remapping[node] = new_node
            if node is inner_mm:
                new_input_anchor = new_node
            if node is f_node:
                new_output_anchor = new_node
    # pyrefly: ignore [unbound-name]
    if new_input_anchor is not new_output_anchor:  # subgraph is non-trivial
        # update the input node
        # pyrefly: ignore [unbound-name]
        with new_graph.inserting_before(new_input_anchor):
            new_input_node = new_graph.placeholder(name="subgraph_input")
            # pyrefly: ignore [unbound-name]
            new_input_node.meta.update(new_input_anchor.meta)
            # pyrefly: ignore [unbound-name]
            new_input_anchor.replace_all_uses_with(new_input_node)
        # pyrefly: ignore [unbound-name]
        new_graph.erase_node(new_input_anchor)
        # add the output node
        # pyrefly: ignore [unbound-name]
        new_output_node = new_graph.output(new_output_anchor)
        # pyrefly: ignore [unbound-name]
        new_output_node.meta.update(new_output_anchor.meta)
    else:  # subgraph is trivial, e.g. (A @ (B @ C))
        # update the input node
        # pyrefly: ignore [unbound-name]
        with new_graph.inserting_before(new_input_anchor):
            new_input_node = new_graph.placeholder(name="subgraph_input")
            # pyrefly: ignore [unbound-name]
            new_input_node.meta.update(new_input_anchor.meta)
            # pyrefly: ignore [unbound-name]
            new_input_anchor.replace_all_uses_with(new_input_node)
        # pyrefly: ignore [unbound-name]
        new_graph.erase_node(new_input_anchor)
        # update the output node (don't use new_output_anchor since it has been erased)
        new_output_node = new_graph.output(new_input_node)
        new_output_node.meta.update(new_input_node.meta)
    new_graph.lint()

    # construct the subgraph
    subgraph = Subgraph(
        name="subgraph", graph_module=torch.fx.GraphModule(module, new_graph)
    )

    # two cases
    # (1) (subgraph(A @ B) @ C), called "left_assoc"
    # (2) (A @ subgraph(B @ C)), called "right_assoc"
    is_left_assoc = outer_mm.args[0] is f_node

    # find the nodes A, B, C and check the sizes
    A: torch.fx.Node
    B: torch.fx.Node
    C: torch.fx.Node
    if is_left_assoc:
        A = inner_mm.args[0]  # type: ignore[assignment]
        B = inner_mm.args[1]  # type: ignore[assignment]
        C = outer_mm.args[1]  # type: ignore[assignment]
    else:
        A = outer_mm.args[0]  # type: ignore[assignment]
        B = inner_mm.args[0]  # type: ignore[assignment]
        C = inner_mm.args[1]  # type: ignore[assignment]
    if not is_b2b_gemm_good_on(is_left_assoc, A, B, C):
        return

    # finally update the original graph
    counters["inductor"]["b2b_gemm"] += 1
    graph = match.graph
    with graph.inserting_before(outer_mm):
        function = functools.partial(tuned_b2b_gemm, is_left_assoc, subgraph)
        function.__name__ = tuned_b2b_gemm.__name__  # type: ignore[attr-defined]
        function._inductor_lowering_function = True  # type: ignore[attr-defined]
        function._inductor_lowering_output_metadata_ignores_input_storage = True  # type: ignore[attr-defined]
        replacement: torch.fx.Node = graph.call_function(
            function,
            (A, B, C),
            match.kwargs,
        )
        replacement.meta.update(outer_mm.meta)
        outer_mm.replace_all_uses_with(replacement)
    # erase unnecessary nodes
    graph.erase_node(outer_mm)
    for node in reversed(subgraph_node_list):
        graph.erase_node(node)
    graph.lint()
<|endoftext|>from __future__ import annotations

import itertools
import re
from typing import Any, TYPE_CHECKING

import torch.fx as fx  # noqa: TC001
from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from collections.abc import Callable


def _get_module_stack(node: fx.Node) -> list[tuple[str, type[Any]]]:
    nn_stack = node.meta.get("nn_module_stack", "")
    if nn_stack:
        return list(nn_stack.values())

    fwd_nn_stack = node.meta.get("fwd_nn_module_stack", "")
    if fwd_nn_stack:
        return list(fwd_nn_stack.values())

    return []


def _addindent(s_: str, num_spaces: int) -> str:
    s: list[str] = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first: str = s.pop(0)
    s: list[str] = [(num_spaces * " ") + line for line in s]
    joint_s: str = "\n".join(s)
    joint_s = first + "\n" + joint_s
    return joint_s


class GraphView:
    """
    A hierarchical class for organizing and managing torch.fx nodes by their module stack.

    This class provides a tree-like structure where each node in the hierarchy corresponds
    to a module or submodule in a traced FX graph. Each `GraphView` instance can hold a list
    of FX nodes (`self.data`) belonging to that module scope, maintain a unique set of nodes
    (`self.unique_nodes`), and manage its child containers (`self.children`).

    Attributes:
        name (str): The name of the module or container scope.
        klass (type[Any]): The class type associated with this module/container.
        data (list[fx.Node]): A list of FX graph nodes belonging to this module.
        unique_nodes (OrderedSet[fx.Node]): A deduplicated set of nodes to ensure no duplicates.
        children (dict[str, GraphView]): A mapping of child module names to their corresponding GraphView instances.
    """

    def __init__(self, name: str, klass: type[Any]) -> None:
        self.name: str = name
        self.klass: type[Any] = klass
        self.data: list[fx.Node] = []
        self.unique_nodes: OrderedSet[fx.Node] = OrderedSet()
        self.children: dict[str, GraphView] = {}

    def add(self, data: fx.Node) -> None:
        if data not in self.unique_nodes:
            self.data.append(data)
            self.unique_nodes.add(data)

    def get_child(self, module_stack: str, klass: type[Any] | None = None) -> GraphView:
        if module_stack not in self.children:
            new_stack = GraphView(module_stack, klass or self.klass)
            self.children[module_stack] = new_stack
        return self.children[module_stack]

    def __getitem__(self, name: str) -> GraphView:
        return self.children[name]

    def __getattr__(self, name: str) -> GraphView:
        return self.children[name]

    def __repr__(self) -> str:
        child_lines: list[str] = []
        for name, child in self.children.items():
            mod_str = repr(child)
            mod_str = _addindent(mod_str, 2)
            child_lines.append(f"({name}): {mod_str}")
        main_str = f"{self.klass.__name__}("
        if child_lines:
            main_str += "\n  " + "\n  ".join(child_lines) + "\n"
        main_str += ")"
        return main_str


def _clean_stack_name(stack_name: str) -> str:
    """
    Clean up FX node's nn_module_stack metadata string to match the module name hierarchies

    Example:
        Input: "L['self']._modules['layers']['0']._modules['attention']"
        Output: "layers.0.attention"
    """
    cleaned = re.sub(r"^L\['self'\]\.?", "", stack_name)
    parts = re.findall(r"\['([^']+)'\]", cleaned)
    return ".".join(parts) if parts else cleaned


def _is_root(stack: str) -> bool:
    return stack == ""


def make_graph_view(
    graph: fx.Graph,
    module_stack_fn: Callable[[fx.Node], list[tuple[str, type[Any]]]] | None = None,
) -> GraphView | None:
    """
    Code from: https://github.com/meta-pytorch/autoparallel/pull/158

    Make a graph view from the fx.Graph. This is a tree structure that
    represents the module hierarchy of the graph, and enables us to
    easily find the nodes that belong to each module, and gives a slightly
    easier way of visualize different parts of the graph by extracting
    subgraphs that belong to a particular module FQN.

    For example, if we have the following model with module hierarchy:

    Transformer(
        (tok_embeddings): Embedding(128256, 4096)
        (layers): ModuleDict(
            (0): TransformerBlock(
            (attention): Attention(
                (wq): Linear(in_features=4096, out_features=4096, bias=False)
                (wk): Linear(in_features=4096, out_features=1024, bias=False)
                (wv): Linear(in_features=4096, out_features=1024, bias=False)
                (wo): Linear(in_features=4096, out_features=4096, bias=False)
                (sdpa): ScaledDotProductAttention()
            )
            (feed_forward): FeedForward(
                (w1): Linear(in_features=4096, out_features=14336, bias=False)
                (w2): Linear(in_features=14336, out_features=4096, bias=False)
                (w3): Linear(in_features=4096, out_features=14336, bias=False)
            )
            (attention_norm): RMSNorm((4096,), eps=1e-05, elementwise_affine=True)
            (ffn_norm): RMSNorm((4096,), eps=1e-05, elementwise_affine=True)
            )
        )
        (norm): RMSNorm((4096,), eps=1e-05, elementwise_affine=True)
        (output): Linear(in_features=4096, out_features=128256, bias=False)
    )

    Then we can get a GraphView for the fx.Graph that enables us to do

    graph_view = make_graph_view(graph)
    subgraph = get_subgraph_by_path(graph_view, "layers.0")

    where subgraph contains all the nodes that belong to this region

    module_stack_fn: Optional callable for extracting module hierarchy information from nodes.

        Signature: Callable[[fx.Node], list[tuple[str, type[Any]]]]

        Takes an FX node and returns a list of (module_path, module_class) tuples representing
        the nested module hierarchy for that node, ordered from outermost to innermost scope.

        - module_path (str): Dot-separated path identifying the module in the hierarchy
          (e.g., "layers.0.attention.wq")
        - module_class (type): The Python class type of the module

        This enables custom logic for determining module membership, useful for:
        - Graphs without standard nn_module_stack metadata
        - Filtering or grouping nodes by custom criteria

        Example of getting the module stack from annotation:

        def module_stack_fn(node):
            module_stack = node.meta.get("custom", {}).get("module_path", "")
            return [(module_stack, torch.nn.Module)]

        If None, defaults to extracting from node.meta["nn_module_stack"] or
        node.meta["fwd_nn_module_stack"].
    """

    def nn_module_stack_meta(node: fx.Node) -> list[tuple[str, type[Any]]]:
        result = []
        for module_stack, module_class in _get_module_stack(node):
            module_stack = _clean_stack_name(module_stack)
            result.append((module_stack, module_class))
        return result

    if module_stack_fn is None:
        module_stack_fn = nn_module_stack_meta
    nodes: list[fx.Node] = list(graph.nodes)
    nodes_by_module_stack_root: GraphView | None = None
    for node in nodes:
        for module_stack, module_class in module_stack_fn(node):
            nodes_by_module_stack: GraphView | None = nodes_by_module_stack_root
            for name in module_stack.split("."):
                if nodes_by_module_stack is None:
                    nodes_by_module_stack = GraphView(name, module_class)
                    nodes_by_module_stack_root = nodes_by_module_stack
                if _is_root(module_stack):
                    new_stack: GraphView = nodes_by_module_stack
                else:
                    new_stack = nodes_by_module_stack.get_child(name, module_class)
                nodes_by_module_stack = new_stack
                nodes_by_module_stack.add(node)

    return nodes_by_module_stack_root


def get_subgraph_by_path(
    graph_view: GraphView, paths: str | list[str]
) -> list[fx.Node]:
    """
    Get subgraph by path(s).
    Args:
        graph_view (object): Root graph view object.
        paths (str or list of str): Path(s) to subgraph.
    Returns:
        list[fx.Node]: fx nodes belong to the subgraph
    """

    def get_node_by_path(node: GraphView, path: str) -> GraphView:
        for p in path.split("."):
            if p in node.children:
                node = node.children[p]
            else:
                return GraphView("", object)
        return node

    if isinstance(paths, list):
        nodes = list(
            itertools.chain.from_iterable(
                get_node_by_path(graph_view, p).data for p in paths
            )
        )
        return nodes
    else:
        node = get_node_by_path(graph_view, paths)
        return node.data
<|endoftext|># mypy: allow-untyped-defs
import itertools
import logging
import operator
from collections import defaultdict
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.fx.node
from torch._C._dynamo.guards import compute_overlapping_tensors
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import ReinplaceCounters, ReInplaceTrigger
from torch._guards import detect_fake_mode
from torch._higher_order_ops.triton_kernel_wrap import (
    kernel_side_table,
    triton_kernel_wrapper_functional,
)
from torch._inductor import config, inductor_prims
from torch._inductor.fx_utils import get_node_storage, is_node_realized
from torch._inductor.lowering import (
    inplaceable_foreach_ops as inplaceable_foreach_ops_lowerings,
)
from torch._inductor.virtualized import V
from torch.fx.experimental.symbolic_shapes import (
    compute_unbacked_bindings,
    GuardOnDataDependentSymNode,
    statically_known_true,
    sym_eq,
)
from torch.fx.immutable_collections import immutable_dict, immutable_list
from torch.fx.passes.reinplace import _is_view_op
from torch.utils import _pytree as pytree
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)
aten = torch.ops.aten


@dataclass(frozen=True)
class InplaceableOp:
    inplace_op: Callable[..., Any]
    mutated_arg: int | tuple[int, ...]  # Single index or tuple of indices
    extra_check: Callable[[torch.fx.Node], bool] = lambda node: True

    @property
    def mutated_args(self) -> tuple[int, ...]:
        """Return mutated_arg as a tuple for uniform handling."""
        if isinstance(self.mutated_arg, int):
            return (self.mutated_arg,)
        return self.mutated_arg


_SCATTER_OP_TO_VIEW = {
    torch.ops.aten.diagonal_scatter.default: torch.ops.aten.diagonal.default,
    torch.ops.aten.select_scatter.default: torch.ops.aten.select.int,
    torch.ops.aten.slice_scatter.default: torch.ops.aten.slice.Tensor,
    torch.ops.aten.as_strided_scatter.default: torch.ops.aten.as_strided.default,
}
_VIEW_OP_TO_SCATTER = {v: k for k, v in _SCATTER_OP_TO_VIEW.items()}


def graph_call_function(graph: torch.fx.Graph, fn, *args, **kwargs):
    fake_args, fake_kwargs = pytree.tree_map(
        lambda node: node.meta["val"] if isinstance(node, torch.fx.Node) else node,
        (args, kwargs),
    )
    with V.fake_mode:
        fake_result = fn(*fake_args, **fake_kwargs)

    node = graph.call_function(fn, args, kwargs)

    node.meta["val"] = fake_result
    if (V.fake_mode.shape_env) and (
        symbol_to_path := compute_unbacked_bindings(V.fake_mode.shape_env, fake_result)
    ):
        node.meta["unbacked_bindings"] = symbol_to_path

    return node


@dataclass
class ViewOp:
    target: torch._ops.OpOverload
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


def _inplace_generalized_scatter(
    inp: torch.Tensor, src: torch.Tensor, view_ops: list[ViewOp]
) -> torch.Tensor:
    tmp = inp
    for view in view_ops:
        fake_args, fake_kwargs = pytree.tree_map(
            lambda node: node.meta["val"] if isinstance(node, torch.fx.Node) else node,
            (view.args, view.kwargs),
        )
        # slice and select can allocate new unbacked symints, but those won't be reflected
        # in the output of this function, hence shall be ignored.
        fake_mode = detect_fake_mode(fake_args)
        with (
            fake_mode.shape_env.ignore_fresh_unbacked_symbols()
            if fake_mode and fake_mode.shape_env
            else nullcontext()
        ):
            tmp = view.target(tmp, *fake_args, **fake_kwargs)
    try:
        tmp.copy_(src)
    except RuntimeError as e:
        raise RuntimeError(
            f"shape error in scatter op, can not broadcast {src.shape} to {tmp.shape}"
        ) from e
    return inp


def _generalized_scatter(
    inp: torch.Tensor, src: torch.Tensor, view_ops: list[ViewOp]
) -> torch.Tensor:
    out = inp.clone()
    return _inplace_generalized_scatter(out, src, view_ops)


def _decompose_scatter_functional_helper(
    graph: torch.fx.Graph,
    inp: torch.Tensor,
    src: torch.Tensor,
    view_ops: list[ViewOp],
) -> torch.fx.Node:
    view_op, view_ops_tail = view_ops[0], view_ops[1:]

    if view_ops_tail:
        view = graph_call_function(
            graph, view_op.target, inp, *view_op.args, **view_op.kwargs
        )
        src = _decompose_scatter_functional_helper(graph, view, src, view_ops[1:])  # type: ignore[assignment]

    return graph_call_function(
        graph,
        _VIEW_OP_TO_SCATTER[view_op.target],
        inp,
        src,
        *view_op.args,
        **view_op.kwargs,
    )


def _decompose_scatter_functional(
    graph: torch.fx.Graph, node: torch.fx.Node
) -> torch.fx.Node:
    """Decompose _generalized_scatter to a sequence of view_scatter operations

    e.g. _generalized_scatter(inp, src, [(aten.slice, 0, 0, 10), (aten.slice, 1, 10, -10)])

    will become

    view = aten.slice(inp, 0, 0, 10)
    view_updated = aten.slice_scatter(view, src, 1, 10, -10)
    inp_updated = aten.slice_scatter(inp, view_updated, 0, 0, 10)
    """
    if node.target is not _generalized_scatter:
        raise AssertionError(
            f"expected node.target to be _generalized_scatter, got {node.target}"
        )
    return _decompose_scatter_functional_helper(graph, *node.args)  # type: ignore[arg-type]


def _decompose_scatter_mutating(
    graph: torch.fx.Graph, node: torch.fx.Node
) -> torch.fx.Node:
    """Decompose _generalized_scatter using mutations

    e.g. _generalized_scatter(inp, src, [(aten.slice, 0, 0, 10), (aten.slice, 1, 10, -10)])

    will become

    inp_updated = aten.clone(inp)
    slice1 = aten.slice(inp_updated, 0, 0, 10)
    slice2 = aten.slice(slice1, 1, 10, -10)
    slice2.copy_(src)

    """
    if node.target not in (_generalized_scatter, _inplace_generalized_scatter):
        raise AssertionError(f"unexpected node.target: {node.target}")
    inp, src, view_ops = node.args
    if node.kwargs:
        raise AssertionError(f"expected no kwargs, got {node.kwargs}")

    if node.target is _generalized_scatter:
        inp = graph_call_function(graph, aten.clone, inp)

    tmp = inp
    for view in view_ops:  # type: ignore[union-attr]
        tmp = graph_call_function(graph, view.target, tmp, *view.args, **view.kwargs)  # type: ignore[union-attr]

    graph_call_function(graph, aten.copy_.default, tmp, src)
    return inp  # type: ignore[return-value]


# View ops whose view_scatter op is lowered into mutations anyway,
# so is never a pessimisation to decompose.
_ALWAYS_MUTATING_SCATTER_OPS = OrderedSet(
    [
        aten.as_strided.default,
        aten.diagonal.default,
    ]
)


def scatter_always_uses_mutation(node: torch.fx.Node) -> bool:
    _, _, view_ops = node.args
    view_ops = cast(Sequence[torch.fx.node.Argument], view_ops)
    return any(
        target in _ALWAYS_MUTATING_SCATTER_OPS
        for view in view_ops
        if isinstance(target := getattr(view, "target", None), torch._ops.OpOverload)
    )


def should_reinplace_scatter(node: torch.fx.Node) -> bool:
    """Choose between mutating and functional scatter decompositions

    Reinplacing view scatter ops can be pessimising as it blocks fusion with the
    input or output tensor computations. However, it is still profitable if the
    input and output would have been realized anyway.

    """
    inp, _src, _view_ops = node.args

    # Mutating scatter ops unconditionally realize input and output
    if scatter_always_uses_mutation(node):
        return True

    if is_node_realized(inp) and is_node_realized(node):  # type: ignore[arg-type]
        return True

    # If the output is copied back into the input, this forces both to be
    # realized as the output is a user of the input
    if inp.op in ("placeholder", "get_attr") and any(  # type: ignore[union-attr]
        user.target is aten.copy_.default and user.args[0] is inp for user in node.users
    ):
        return True

    # Otherwise, assume fusions will make functional variants profitable
    return False


def decompose_generalized_scatter(graph: torch.fx.Graph) -> None:
    """Replace _generalized_scatter with normal aten ops"""
    for node in itertools.chain(
        graph.find_nodes(op="call_function", target=_generalized_scatter),
        graph.find_nodes(op="call_function", target=_inplace_generalized_scatter),
    ):
        use_mutation = (
            node.target is _inplace_generalized_scatter
            or scatter_always_uses_mutation(node)
        )

        with graph.inserting_before(node):
            if use_mutation:
                new_node = _decompose_scatter_mutating(graph, node)
            else:
                new_node = _decompose_scatter_functional(graph, node)

        node.replace_all_uses_with(new_node)
        graph.erase_node(node)


def canonicalize_view_scatter_ops(graph: torch.fx.Graph) -> None:
    """
    This canonicalizes view scatter ops into a generalized form, defined as:
      def scatter(inp, src, views):
        tmp = inp.clone()
        for view in views:
          tmp = view(tmp)
        tmp.copy_(src)

    We also fuse consecutive view scatter ops of the form
        a = scatter(view2(self), src, [view1])
        b = scatter(self, a, [view2])
    which can be rewritten as
        b = scatter(self, src, [view2, view1])
        a = view2(b)

    This is both more efficient as we only do a single scatter, and also
    easier to reinplace since there is only one use of `self`
    """

    node_to_view_base: dict[torch.fx.Node, torch.fx.Node] = {}
    node_to_view_op: dict[torch.fx.Node, list[ViewOp]] = defaultdict(list)

    def handle_views(node: torch.fx.Node):
        inp = node.args[0]
        node_to_view_base[node] = node_to_view_base.get(inp, inp)  # type: ignore[arg-type, assignment]
        node_to_view_op[node] = [
            *node_to_view_op[inp],  # type: ignore[index]
            ViewOp(
                node.target,  # type: ignore[arg-type]
                args=node.args[1:],
                kwargs=node.kwargs,
            ),
        ]

    def handle_view_scatter(node: torch.fx.Node):
        if len(node.args) < 2:
            raise AssertionError(f"expected at least 2 args, got {len(node.args)}")
        inp, src = node.args[:2]

        if not isinstance(node.target, torch._ops.OpOverload):
            raise AssertionError(
                f"expected node.target to be an OpOverload, got {type(node.target)}"
            )
        scatter_view_op = ViewOp(
            _SCATTER_OP_TO_VIEW[node.target],
            args=node.args[2:],
            kwargs=node.kwargs,
        )

        def can_fuse():
            if src.target is not _generalized_scatter:  # type: ignore[union-attr]
                return False
            src_inp, _src_src, _src_scatter_view_op = src.args  # type: ignore[union-attr]

            inp_base = node_to_view_base.get(inp, inp)  # type: ignore[arg-type]
            src_base = node_to_view_base.get(src_inp, src_inp)  # type: ignore[arg-type]
            return inp_base is src_base and node_to_view_op[src_inp] == [  # type: ignore[index]
                *node_to_view_op[inp],  # type: ignore[index]
                scatter_view_op,
            ]

        if not can_fuse():
            with graph.inserting_before(node):
                new_node = graph_call_function(
                    graph,
                    _generalized_scatter,
                    inp,
                    src,
                    [scatter_view_op],
                )
            node.replace_all_uses_with(new_node)
            graph.erase_node(node)
            return

        _src_inp, src_src, src_scatter_view_op = src.args  # type: ignore[union-attr]
        with graph.inserting_before(src):  # type: ignore[arg-type]
            new_node = graph_call_function(
                graph,
                _generalized_scatter,
                inp,
                src_src,
                [scatter_view_op, *src_scatter_view_op],  # type: ignore[misc]
            )
            node.replace_all_uses_with(new_node)
            graph.erase_node(node)

            if src.users:  # type: ignore[union-attr]
                new_src = graph_call_function(
                    graph,
                    _SCATTER_OP_TO_VIEW[node.target],
                    new_node,
                    *node.args[2:],
                    **node.kwargs,
                )

                handle_views(new_src)
                src.replace_all_uses_with(new_src)  # type: ignore[union-attr]

            graph.erase_node(src)  # type: ignore[arg-type]

    for node in graph.nodes:
        if _is_view_op(node.target):
            handle_views(node)
        elif node.target in _SCATTER_OP_TO_VIEW:
            handle_view_scatter(node)


inplaceable_ops: dict[Callable[..., Any], InplaceableOp] = {
    aten.index_put.default: InplaceableOp(aten.index_put_.default, 0),
    aten._unsafe_index_put.default: InplaceableOp(inductor_prims._unsafe_index_put_, 0),
    _generalized_scatter: InplaceableOp(
        _inplace_generalized_scatter,
        0,
        extra_check=should_reinplace_scatter,
    ),
    # Stateless Philox RNG: reinplace the functionalized clone onto the dead
    # output buffer, so out-of-place uniform()/normal() don't pay an extra copy.
    aten._philox_uniform.default: InplaceableOp(aten._philox_uniform_.default, 0),
    aten._philox_normal.default: InplaceableOp(aten._philox_normal_.default, 0),
}

try:
    c10d_functional = torch.ops._c10d_functional
    inplaceable_collective_ops: dict[Callable[..., Any], InplaceableOp] = {
        c10d_functional.all_reduce.default: InplaceableOp(
            c10d_functional.all_reduce_.default, 0
        ),
        c10d_functional.all_reduce_coalesced.default: InplaceableOp(
            c10d_functional.all_reduce_coalesced_.default, 0
        ),
    }
    inplaceable_ops.update(inplaceable_collective_ops)
except AttributeError:
    # _c10d_functional ops are only available when torch
    # is built with USE_DISTRIBUTED=1.
    pass

inplaceable_foreach_ops: dict[torch._ops.OpOverload, InplaceableOp] = {}
for outplace_op, inplace_op in inplaceable_foreach_ops_lowerings.items():
    inplaceable_foreach_ops[outplace_op] = InplaceableOp(inplace_op, 0)


inplaceable_triton_ops = OrderedSet([triton_kernel_wrapper_functional])


# Operators that don't depend on the tensor data
META_ONLY_OPS = OrderedSet(
    [
        aten.sym_size.int,
        aten.sym_stride.int,
        aten.sym_numel.default,
        aten.sym_storage_offset.default,
    ]
)


def _get_view_base(node: torch.fx.Node) -> torch.fx.Node:
    while _is_view_op(node.target) and node.args:
        base = node.args[0]
        if not isinstance(base, torch.fx.Node):
            break
        node = base
    return node


def _same_tensor_metadata(lhs: torch.fx.Node, rhs: torch.fx.Node) -> bool:
    lhs_val = lhs.meta.get("val")
    rhs_val = rhs.meta.get("val")
    if not isinstance(lhs_val, torch.Tensor) or not isinstance(rhs_val, torch.Tensor):
        return False

    def same_value(lhs_value, rhs_value) -> bool:
        return statically_known_true(sym_eq(lhs_value, rhs_value))

    def same_sequence(lhs_values, rhs_values) -> bool:
        return len(lhs_values) == len(rhs_values) and all(
            same_value(lhs_value, rhs_value)
            for lhs_value, rhs_value in zip(lhs_values, rhs_values)
        )

    return (
        same_sequence(lhs_val.size(), rhs_val.size())
        and same_sequence(lhs_val.stride(), rhs_val.stride())
        and same_value(lhs_val.storage_offset(), rhs_val.storage_offset())
    )


def _is_layout_preserving_view_copy_back(
    dst: torch.fx.Node,
    src: torch.fx.Node,
    mutated_arg: torch.fx.Node,
    src_base: torch.fx.Node,
) -> bool:
    return _same_tensor_metadata(src_base, mutated_arg) and _same_tensor_metadata(
        src, dst
    )


def _is_control_deps_ordering_only_use(
    user: torch.fx.Node, view: torch.fx.Node
) -> bool:
    """Check if view appears only in control_deps' ordering-only additional_deps tuple.

    control_deps schema (control_dependencies.py): control_deps(additional_deps, subgraph, *args)
    kwargs is always empty -- preserve_node_ordering threads all nodes positionally.
    """
    if not (
        user.op == "call_function"
        and isinstance(user.target, torch._ops.HigherOrderOperator)
        and user.target._name == "control_deps"
    ):
        return False
    additional_deps = cast(tuple[torch.fx.Node, ...], user.args[0])
    pass_through = cast(tuple[torch.fx.Node, ...], user.args[2:])
    return view in additional_deps and view not in pass_through


def reinplace_inplaceable_ops_core(graph: torch.fx.Graph) -> None:
    """
    Reinplaces in-placeable operations.
    If there are no uses of a view of the mutated arg after the current node,
    it is possible to inplace the op.
    This above algorithm could be justified by observing side effects. While
    we traverse the graph in forwards direction, only latter nodes could view
    side effects of the current node. If the current node is not used later as
    well as no view of this node is used later in the graph, then it is safe to
    inplace as there would be no way to observe the side effects.
    This condition is slightly different for graph inputs where they can only
    be inplaced if the above condition is true and there's a copy_ in the
    epilogue that signals that the caller wants to observe the mutation.

    Unlike JIT Inductor, AOTInductor currently unlifts weights and buffers from
    input args, so instead of checking mutation on placeholder, AOTInductor
    checks mutation on get_attr. This is subject to change in future.
    """

    copy_args_to_copy_nodes = {}
    # maps argument to the first copy_ node that mutates it.
    copy_nodes = {}
    # maps (view_arg, inplaceable_op_node) to copy_ nodes that copy a view of
    # the op result back into the graph input that view_arg aliases.
    copy_args_to_copy_nodes_via_views = {}
    mutated_inputs = OrderedSet[Any]()
    storage_to_nodes = defaultdict(list)
    node_order: dict[Any, int] = {}
    for i, node in enumerate(reversed(graph.nodes)):
        node_order[node] = len(graph.nodes) - i - 1
        storage_to_nodes[get_node_storage(node)].append(node)
        if node.target is aten.copy_.default and node.args[0].op in (
            "placeholder",
            "get_attr",
        ):
            dst = node.args[0]
            src = node.args[1]
            # If the target is a getitem and it indexes a possible clone,
            # then skip over it
            if src.target is operator.getitem and (
                (
                    src.args[0].target == triton_kernel_wrapper_functional
                    and src.args[0].kwargs["kwargs"][src.args[1]] == node.args[0]
                )
                or (src.args[0].target in inplaceable_foreach_ops)
                or (src.args[0].target is torch.ops.higher_order.auto_functionalized)
            ):
                src = src.args[0]

            copy_args_to_copy_nodes[(dst, src)] = node
            copy_nodes[dst] = node

            mutated_inputs.add(node.args[0])

            src_base = _get_view_base(src)
            if src_base is not src and isinstance(src_base.target, Callable):
                inplaceable_op = inplaceable_ops.get(src_base.target)
                if inplaceable_op is not None:
                    for mutated_arg_idx in inplaceable_op.mutated_args:
                        mutated_arg = src_base.args[mutated_arg_idx]
                        if (
                            isinstance(mutated_arg, torch.fx.Node)
                            and _get_view_base(mutated_arg) is dst
                            and _is_layout_preserving_view_copy_back(
                                dst, src, mutated_arg, src_base
                            )
                        ):
                            copy_args_to_copy_nodes_via_views[
                                (mutated_arg, src_base)
                            ] = node

    def any_use_of_views_after_node(node, shared_view_nodes, *, copy_node, mutated_arg):
        node_loc = node_order[node]
        copy_node_loc = node_order[copy_node] if copy_node is not None else None

        def is_meta_only_user(node):
            if _is_view_op(node.target):
                return all(is_meta_only_user(u) for u in node.users)
            return node.target in META_ONLY_OPS

        for view in shared_view_nodes:
            for user in view.users:
                user_loc = node_order[user]
                # Skip all users before node
                if user_loc <= node_loc:
                    continue
                # Ignore uses after the copy_ epilogue node, where the input
                # has already been mutated anyway
                if copy_node_loc is not None and copy_node_loc <= user_loc:
                    continue
                # Reinplacing does not change shape metadata
                if is_meta_only_user(user):
                    continue
                # If our graph looks like:
                # foo(mutated_arg)
                # mutated_arg.copy_(other)
                # then it's safe for us to reinplace foo because mutated_arg
                # will get overwritten anyways.
                if (
                    user.target is torch.ops.aten.copy_.default
                    and mutated_arg is user.args[0]
                ):
                    continue
                # control_deps args[0] is an ordering-only deps tuple;
                # appearing there does not constitute a real data use.
                if _is_control_deps_ordering_only_use(user, view):
                    continue
                return True
        return False

    def can_inplace(node, mutated_arg):
        # ls should be a list of tensors that all shares the same storage.
        def _overlap(ls) -> bool:
            try:
                return len(compute_overlapping_tensors(ls)) != 0
            except GuardOnDataDependentSymNode:
                # If we fail with data dependent error we assume they all overlap.
                return True

        if isinstance(mutated_arg, (list, tuple)):
            # TODO Using _overlap here causes a several issues.
            unique_storages = OrderedSet(get_node_storage(arg) for arg in mutated_arg)
            if len(unique_storages) != len(mutated_arg):
                # At least two Tensors in mutated_arg alias each other, so we can't reinplace it.
                # We can probably do better (that is, reinplace one of them and clone the other)
                # but that requires more work and mutable List[Tensor] are not that common.
                return False
            return all(can_inplace(node, arg) for arg in mutated_arg)

        if get_node_storage(mutated_arg) is None:
            return False

        if torch._debug_has_internal_overlap(mutated_arg.meta["val"]) == 1:
            return False

        shared_view_nodes = storage_to_nodes[get_node_storage(mutated_arg)]

        # Only keep tensor that might overlap with mutated_arg.
        shared_view_nodes = [
            v
            for v in shared_view_nodes
            if _overlap([mutated_arg.meta["val"], v.meta["val"]])
        ]

        if mutated_arg.op in ("placeholder", "get_attr"):
            # Get the first copy_ node that mutates the mutated_arg.
            copy_node = copy_nodes.get(mutated_arg)
            if copy_node is None:
                # There is no copy_ back to the candidate mutated_arg (which is a graph input).
                # Therefore the semantics of the program are that it does not mutate
                # mutated_arg, so we cannot re-inplace it.
                return False
            if any_use_of_views_after_node(
                node, shared_view_nodes, copy_node=copy_node, mutated_arg=mutated_arg
            ):
                return False

            return True
        elif any(view.op in ("placeholder", "get_attr") for view in shared_view_nodes):
            # If mutated_arg is a view of a graph input, we can only reinplace
            # when a later copy_ writes this op's result back to that input.
            copy_node = copy_args_to_copy_nodes_via_views.get((mutated_arg, node))
            if copy_node is None:
                return False

            mutated_arg_base = copy_node.args[0]
            copy_src = copy_node.args[1]
            if not (
                isinstance(mutated_arg_base, torch.fx.Node)
                and isinstance(copy_src, torch.fx.Node)
                and _is_layout_preserving_view_copy_back(
                    mutated_arg_base, copy_src, mutated_arg, node
                )
            ):
                return False

            if any_use_of_views_after_node(
                node,
                shared_view_nodes,
                copy_node=copy_node,
                mutated_arg=mutated_arg_base,
            ):
                return False
            return True
        else:
            return not any_use_of_views_after_node(
                node, shared_view_nodes, copy_node=None, mutated_arg=mutated_arg
            )

    def copy_node_for_reinplaced_arg(node, mutated_arg):
        copy_node = copy_args_to_copy_nodes.get((mutated_arg, node))
        if copy_node is None:
            copy_node = copy_args_to_copy_nodes_via_views.get((mutated_arg, node))
            if copy_node is not None:
                copy_dst = copy_node.args[0]
                copy_src = copy_node.args[1]
                if not (
                    isinstance(copy_dst, torch.fx.Node)
                    and isinstance(copy_src, torch.fx.Node)
                    and _is_layout_preserving_view_copy_back(
                        copy_dst, copy_src, mutated_arg, node
                    )
                ):
                    return None
        return copy_node

    def all_can_inplace(node, mutated_args):
        return all(can_inplace(node, arg) for arg in mutated_args)

    def log_inplace_results(
        node_name,
        old_tensors_to_clone,
        tensors_to_clone,
        missed_args,
        missed_nodes,
        trigger,
    ):
        # Total size of possibly_missed_reinplacing_opportunities for tensors with static shapes.
        missed_bytes = 0

        def bytes(node):
            t = node.meta.get("val", None)
            if (
                t is not None
                and isinstance(t.element_size(), int)
                and isinstance(t.numel(), int)
            ):
                return t.element_size() * t.numel()
            else:
                return 0

        for node in missed_nodes:
            if isinstance(node, (list, tuple)):
                for n in node:
                    missed_bytes += bytes(n)
            else:
                missed_bytes += bytes(node)

        log.info(
            "For node %s, attempted to reinplace %s. We were unable to reinplace %s; "
            "%s (if non-empty) are possible missed reinplacing opportunities that may be bad for "
            "memory usage and performance. Total size of missed opportunities with static shapes is"
            " : %s bytes.",
            node_name,
            old_tensors_to_clone,
            tensors_to_clone,
            missed_args,
            missed_bytes,
        )

        ReinplaceCounters.add_missed_opportunities(trigger, len(missed_args))
        ReinplaceCounters.add_missed_bytes(trigger, missed_bytes)

    replace_dict: dict[torch.fx.Node, torch.fx.Node] = {}

    def reinplace_and_refine_tensors_to_clone(
        old_tensors_to_clone, kwargs, node_name, trigger
    ):
        tensors_to_clone: list[str] = []
        storage_of_reinplaced_args = OrderedSet[int | None]()

        # Those used to count possibly_missed_reinplacing_opportunities
        missed_nodes = []
        missed_args = []

        # TODO this logic can be made more precise using _overlap
        def tensor_with_same_storage_already_reinplaced(arg):
            if isinstance(arg, (list, tuple)):
                return any(
                    get_node_storage(a) in storage_of_reinplaced_args for a in arg
                )
            return get_node_storage(mutated_arg) in storage_of_reinplaced_args

        for arg in old_tensors_to_clone:
            if arg not in kwargs:
                raise AssertionError(f"expected {arg} to be in kwargs")

            mutated_arg = kwargs[arg]

            # Let's say we have:
            # - op(x, y) that mutates both x and y
            # - new_x, new_y = functional_op(x, y) is the functional variant
            # If we are presented with functional_op(x, x), we must not reinplace
            # this into op(x, x), because then it would be writing to the same Tensor.
            # Instead, it's OK to reinplace one of them and to clone the other:
            # >>> y = x.clone()
            # >>> op(x, y)
            # This also applies if we have views: functional_op(x, x[0])
            # should not reinplace into op(x, x[0]).
            should_attempt_reinplace = not tensor_with_same_storage_already_reinplaced(
                mutated_arg
            )
            if should_attempt_reinplace and can_inplace(node, mutated_arg):
                # In general, we probably do not need those optimizations.
                copy_node = copy_node_for_reinplaced_arg(node, mutated_arg)
                if copy_node is not None:
                    replace_dict[copy_node] = copy_node.args[0]
                if trigger != ReInplaceTrigger.AUTO_FUNC_V2:
                    for user in node.users:
                        # For auto_functionalize_v2, arg is the index of the base, where base at index i corresponds to
                        # output at index size(out)+i.
                        # This used to compare string with integers before for auto_functionalize_v2. Not sure
                        # if it was needed for inplaceable_triton_ops?
                        if user.target is operator.getitem and user.args[1] == arg:
                            replace_dict[user] = mutated_arg

                if isinstance(mutated_arg, (list, tuple)):
                    for a in mutated_arg:
                        storage_of_reinplaced_args.add(get_node_storage(a))
                else:
                    storage_of_reinplaced_args.add(get_node_storage(mutated_arg))
            else:
                if should_attempt_reinplace:
                    missed_args.append(arg)
                    missed_nodes.append(mutated_arg)

                tensors_to_clone.append(arg)

        log_inplace_results(
            node_name,
            old_tensors_to_clone,
            tensors_to_clone,
            missed_args,
            missed_nodes,
            trigger,
        )
        return tensors_to_clone

    for node in graph.nodes:
        if (inplaceable_op := inplaceable_ops.get(node.target)) is not None:
            # Check if ALL mutated args can be inplaced
            # Only convert if we don't need to clone any tensor
            mutated_args = [node.args[idx] for idx in inplaceable_op.mutated_args]
            if all_can_inplace(node, mutated_args) and inplaceable_op.extra_check(node):
                for mutated_arg in mutated_args:
                    copy_node = copy_node_for_reinplaced_arg(node, mutated_arg)
                    if copy_node is not None:
                        replace_dict[copy_node] = copy_node.args[0]
                node.target = inplaceable_op.inplace_op
        elif node.target is torch.ops.higher_order.auto_functionalized_v2:
            _mutable_op = node.args[0]
            kwargs = node.kwargs

            all_bases = kwargs["_all_bases"]
            bases_to_clone = range(len(all_bases))
            base_tensors_dct = dict(enumerate(all_bases))
            new_bases_to_clone: list[int] = reinplace_and_refine_tensors_to_clone(
                bases_to_clone,
                base_tensors_dct,
                _mutable_op._name,
                ReInplaceTrigger.AUTO_FUNC_V2,
            )
            # Stash the metadata. There is a pass later on where we decompose
            # auto_functionalized into clones + a mutable op; this metadata
            # tells the decomp to only clone the following inputs
            node.meta["only_clone_these_tensors"] = new_bases_to_clone
        elif node.target is torch.ops.higher_order.auto_functionalized:
            _mutable_op = node.args[0]
            from torch._higher_order_ops.auto_functionalize import get_mutable_args

            tensors_to_clone, _ = get_mutable_args(_mutable_op)
            # Don't try to reinplace Tensor | None args that are None.
            tensors_to_clone = [
                t for t in tensors_to_clone if node.kwargs[t] is not None
            ]
            tensors_to_clone = reinplace_and_refine_tensors_to_clone(
                tensors_to_clone,
                node.kwargs,
                _mutable_op._name,
                ReInplaceTrigger.AUTO_FUNC_V1,
            )

            # Stash the metadata. There is a pass later on where we decompose
            # auto_functionalized into clones + a mutable op; this metadata
            # tells the decomp to only clone the following inputs
            node.meta["only_clone_these_tensors"] = tensors_to_clone
        elif node.target is torch.ops.higher_order.with_effects:
            # Handle effectful ops wrapped with with_effects
            # args[0] is the token, args[1] is the inner op, args[2:] are the op's args
            inner_op = node.args[1]
            log.debug(
                "reinplace: checking with_effects node with inner_op=%s", inner_op
            )
            if (inplaceable_op := inplaceable_ops.get(inner_op)) is not None:
                log.debug("reinplace: found inplaceable_op for %s", inner_op)
                # Get the mutated arg indices, offset by 2 (token + op)
                mutated_arg_indices = inplaceable_op.mutated_args

                # Build flat list of tensors for can_inplace check
                # and a mapping of output index -> replacement tensor(s)
                mutated_tensors_flat = []
                output_idx_to_replacement: dict[int, Any] = {}

                for position, idx in enumerate(mutated_arg_indices):
                    actual_idx = idx + 2  # offset for token and op
                    if actual_idx >= len(node.args):
                        raise AssertionError(
                            f"mutated arg idx {actual_idx} out of range {len(node.args)}"
                        )
                    arg = node.args[actual_idx]

                    # Output index is position + 1 (index 0 is the token)
                    output_idx = position + 1
                    output_idx_to_replacement[output_idx] = arg

                    # Flatten for can_inplace check
                    if isinstance(arg, (list, tuple)):
                        mutated_tensors_flat.extend(arg)
                    else:
                        mutated_tensors_flat.append(arg)

                # Check if all mutated args can be inplaced
                can_inplace_all = all_can_inplace(node, mutated_tensors_flat)

                log.debug(
                    "reinplace with_effects: mutated_tensors=%s, can_inplace_all=%s",
                    [str(a) for a in mutated_tensors_flat],
                    can_inplace_all,
                )

                if can_inplace_all and inplaceable_op.extra_check(node):
                    log.debug(
                        "reinplace with_effects: converting %s -> %s",
                        inner_op,
                        inplaceable_op.inplace_op,
                    )
                    # Update the inner op to inplace version
                    node.update_arg(1, inplaceable_op.inplace_op)

                    # The output structure changes: functional returns (token, tensors),
                    # inplace returns (token, None). We need to redirect tensor uses
                    # to the input tensors.

                    def get_index_from_node(n):
                        """Extract the index from a getitem node."""
                        if n.target is operator.getitem:
                            return n.args[1]
                        return None

                    def is_getitem_node(n, parent):
                        """Check if node n is a getitem indexing into parent."""
                        return n.target is operator.getitem and n.args[0] is parent

                    def replace_and_collect(current_node, replacement_tensors):
                        """
                        Collect replacements for getitem nodes into replace_dict.
                        Nodes are added in child-first order so children are erased before parents.
                        """
                        # Find all users that are getitem nodes indexing into current_node
                        getitem_users = [
                            u
                            for u in current_node.users
                            if is_getitem_node(u, current_node)
                        ]

                        if not getitem_users:
                            # Leaf node - add to replace_dict with actual replacement
                            if isinstance(replacement_tensors, (list, tuple)):
                                if len(replacement_tensors) == 1:
                                    replace_dict[current_node] = replacement_tensors[0]
                                    return True
                                else:
                                    # Multiple tensors but no indexing - can't replace
                                    return False
                            else:
                                replace_dict[current_node] = replacement_tensors
                                return True

                        # Process children first (so they're added to replace_dict before parent)
                        all_children_replaced = True
                        first_replacement = None
                        for getitem_user in getitem_users:
                            idx = get_index_from_node(getitem_user)
                            if idx is None or not isinstance(idx, int):
                                all_children_replaced = False
                                continue

                            if not isinstance(replacement_tensors, (list, tuple)):
                                all_children_replaced = False
                                continue

                            if idx >= len(replacement_tensors):
                                all_children_replaced = False
                                continue

                            if first_replacement is None:
                                first_replacement = replacement_tensors[idx]

                            if not replace_and_collect(
                                getitem_user, replacement_tensors[idx]
                            ):
                                all_children_replaced = False

                        # Add this node to replace_dict after children (even if it has non-getitem users)
                        # Non-getitem users will have their input replaced via replace_all_uses_with
                        if all_children_replaced and first_replacement is not None:
                            replace_dict[current_node] = first_replacement

                        return all_children_replaced

                    # Find getitem nodes that extract tensor results
                    # Use the output_idx_to_replacement mapping built above
                    for user in list(node.users):
                        if not is_getitem_node(user, node):
                            continue
                        idx = get_index_from_node(user)
                        if idx is None or idx not in output_idx_to_replacement:
                            continue
                        replacement = output_idx_to_replacement[idx]
                        replace_and_collect(user, replacement)
        elif node.target in inplaceable_triton_ops:
            kernel_idx = node.kwargs["kernel_idx"]
            kernel = kernel_side_table.get_kernel(kernel_idx)
            from triton.runtime.autotuner import Autotuner
            from triton.runtime.jit import JITFunction

            if isinstance(kernel, JITFunction):
                kernel_name = kernel.fn.__name__
            elif isinstance(kernel, Autotuner):
                if config.is_fbcode():
                    # Autotuner has different implementations for AMD and NV
                    if torch.version.hip is None:
                        kernel_name = kernel.base_fn.__name__
                    else:
                        kernel_name = kernel.fn.__name__
                else:
                    kernel_name = kernel.base_fn.__name__
            else:
                raise AssertionError("Unknown triton kernel type")

            # inplaceable_triton_ops take an additional argument called
            # tensors_to_clone which contain a list of tensors to clone
            # This pass iterates over them and sees which ones are safe
            # to eliminate (i.e. no longer need the clones)
            tensors_to_clone = reinplace_and_refine_tensors_to_clone(
                node.kwargs["tensors_to_clone"],
                node.kwargs["kwargs"],
                kernel_name,
                ReInplaceTrigger.TRITON_OPS,
            )

            kwargs = dict(node.kwargs)
            kwargs["tensors_to_clone"] = tensors_to_clone
            node.kwargs = immutable_dict(kwargs)
            if "eager_input_vals" in node.meta:
                # We changed the kwargs, so we need to update eager_input_vals
                # to something sane.
                args, kwargs = node.meta["eager_input_vals"]
                new_kwargs = {**kwargs}
                new_kwargs["tensors_to_clone"] = immutable_list(tensors_to_clone)
                new_kwargs = immutable_dict(new_kwargs)
                node.meta["eager_input_vals"] = (args, new_kwargs)
        elif (inplaceable_op := inplaceable_foreach_ops.get(node.target)) is not None:
            mutated_args = node.args[inplaceable_op.mutated_arg]

            if not all((arg, node) in copy_args_to_copy_nodes for arg in mutated_args):
                continue

            if can_inplace(node, mutated_args):
                for arg in mutated_args:
                    copy_node = copy_args_to_copy_nodes[(arg, node)]
                    replace_dict[copy_node] = copy_node.args[0]

                node.target = inplaceable_op.inplace_op
    for node, replacement in replace_dict.items():
        while replacement in replace_dict:
            replacement = replace_dict[replacement]
        replace_dict[node] = replacement

        node.replace_all_uses_with(replacement)
        graph.erase_node(node)


def reinplace_inplaceable_ops(
    fake_tensor_updater: torch._inductor.fx_utils.FakeTensorUpdater,
    graph: torch.fx.Graph,
) -> None:
    with enable_python_dispatcher():
        canonicalize_view_scatter_ops(graph)
        # canonicalize_view_scatter_ops adds new operations to the graph.
        # We run fake_tensor_updater to update the alias information.
        # Correct alias information is required for `reinplace_inplaceable_ops_core`.
        fake_tensor_updater.incremental_update()
        reinplace_inplaceable_ops_core(graph)
        decompose_generalized_scatter(graph)
<|endoftext|># mypy: allow-untyped-defs
import inspect

import torch
import torch.nn as nn
from torch._dynamo.utils import counters
from torch._inductor import config as inductor_config
from torch.func import functional_call

from ..pattern_matcher import (
    CallFunctionVarArgs,
    CallModuleVarArgs,
    Match,
    register_graph_pattern,
)
from .pre_grad import efficient_conv_bn_eval_pass


# Cache the signature of F.batch_norm at module load time to avoid repeated
# introspection during graph transformation (fixes performance regression).
_BATCH_NORM_SIGNATURE = inspect.signature(torch.nn.functional.batch_norm)


def efficient_conv_bn_eval(
    bn: nn.modules.batchnorm._BatchNorm, conv: nn.modules.conv._ConvNd, x: torch.Tensor
):
    """
    Implementation based on https://arxiv.org/abs/2305.11624
    "Efficient ConvBN Blocks for Transfer Learning and Beyond"
    It leverages the associative law between convolution and affine transform,
    i.e., normalize (weight conv feature) = (normalize weight) conv feature.
    It works for Eval mode of ConvBN blocks during validation, and can be used
    for **training** as well, but only if one sets `bn.training=False`. It
     reduces memory footprint and computation cost, at the cost of slightly
     reduced numerical stability.
    Args:
        bn (nn.modules.batchnorm._BatchNorm): a BatchNorm module.
        conv (nn.modules.conv._ConvNd): a conv module
        x (torch.Tensor): Input feature map.
    """

    if bn.running_var is None:
        raise AssertionError("expected bn.running_var to not be None")
    if bn.running_mean is None:
        raise AssertionError("expected bn.running_mean to not be None")

    # These lines of code are designed to deal with various cases
    # like bn without affine transform, and conv without bias
    weight_on_the_fly = conv.weight
    if conv.bias is not None:
        bias_on_the_fly = conv.bias
    else:
        bias_on_the_fly = torch.zeros_like(bn.running_var)

    if bn.weight is not None:
        bn_weight = bn.weight
    else:
        bn_weight = torch.ones_like(bn.running_var)

    if bn.bias is not None:
        bn_bias = bn.bias
    else:
        bn_bias = torch.zeros_like(bn.running_var)

    # shape of [C_out, 1, 1, 1] in Conv2d
    target_shape = [-1] + [1] * (conv.weight.ndim - 1)
    if isinstance(conv, nn.modules.conv._ConvTransposeNd):
        # for transposed conv, the C_out dimension should be at index 1.
        target_shape[:2] = [target_shape[1], target_shape[0]]
    weight_coeff = torch.rsqrt(bn.running_var + bn.eps).reshape(target_shape)
    # shape of [C_out, 1, 1, 1] in Conv2d
    coefff_on_the_fly = bn_weight.view_as(weight_coeff) * weight_coeff

    # shape of [C_out, C_in, k, k] in Conv2d
    weight_on_the_fly = weight_on_the_fly * coefff_on_the_fly
    # shape of [C_out] in Conv2d
    bias_on_the_fly = bn_bias + coefff_on_the_fly.flatten() * (
        bias_on_the_fly - bn.running_mean
    )

    input = x
    params = {"weight": weight_on_the_fly, "bias": bias_on_the_fly}
    output = functional_call(conv, params, input)
    return output


def efficient_conv_bn_eval_decomposed(
    bn_weight,
    bn_bias,
    bn_running_mean,
    bn_running_var,
    bn_eps,
    conv: torch._ops.OpOverload,
    conv_weight,
    conv_bias,
    x,
    conv_remaining_args,
):
    """
    Implementation based on https://arxiv.org/abs/2305.11624
    "Efficient ConvBN Blocks for Transfer Learning and Beyond"
    It leverages the associative law between convolution and affine transform,
    i.e., normalize (weight conv feature) = (normalize weight) conv feature.
    It works for Eval mode of ConvBN blocks during validation, and can be used
    for **training** as well, but only if one sets `bn.training=False`. It
     reduces memory footprint and computation cost, at the cost of slightly
     reduced numerical stability.
    Args:
    """
    if bn_running_var is None:
        raise AssertionError("expected bn_running_var to not be None")

    # These lines of code are designed to deal with various cases
    # like bn without affine transform, and conv without bias
    weight_on_the_fly = conv_weight
    if conv_bias is not None:
        bias_on_the_fly = conv_bias
    else:
        bias_on_the_fly = torch.zeros_like(bn_running_var)

    if bn_weight is None:
        bn_weight = torch.ones_like(bn_running_var)

    if bn_bias is None:
        bn_bias = torch.zeros_like(bn_running_var)

    # shape of [C_out, 1, 1, 1] in Conv2d
    target_shape = [-1] + [1] * (conv_weight.ndim - 1)
    if "conv_transpose" in conv.__str__():
        # for transposed conv, the C_out dimension should be at index 1.
        target_shape[:2] = [target_shape[1], target_shape[0]]
    weight_coeff = torch.rsqrt(bn_running_var + bn_eps).reshape(target_shape)
    # shape of [C_out, 1, 1, 1] in Conv2d
    coefff_on_the_fly = bn_weight.view_as(weight_coeff) * weight_coeff

    # shape of [C_out, C_in, k, k] in Conv2d
    weight_on_the_fly = weight_on_the_fly * coefff_on_the_fly
    # shape of [C_out] in Conv2d
    bias_on_the_fly = bn_bias + coefff_on_the_fly.flatten() * (
        bias_on_the_fly - bn_running_mean
    )

    input = x
    return conv(*((input, weight_on_the_fly, bias_on_the_fly) + conv_remaining_args))


@register_graph_pattern(
    CallFunctionVarArgs(
        [
            torch.nn.functional.batch_norm,
        ]
    ),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=efficient_conv_bn_eval_pass,
    extra_check=lambda match: not inductor_config.freezing
    and inductor_config.efficient_conv_bn_eval_fx_passes,
)
def efficient_conv_bn_eval_graph_transform_inlined(match: Match, *args, **kwargs):
    """
    Graph transformation pass for fusing F.batch_norm with preceding conv operations.

    This pass handles F.batch_norm calls with default arguments by normalizing
    the args tuple using inspect.signature. It fuses batch normalization with
    the preceding convolution for more efficient evaluation.
    """
    bn_node = match.nodes[0]
    graph = match.graph

    # Normalize arguments by binding to cached signature and applying defaults.
    # This handles cases where F.batch_norm is called with fewer than 8 args.
    bound_args = _BATCH_NORM_SIGNATURE.bind(*bn_node.args, **bn_node.kwargs)
    bound_args.apply_defaults()
    # Use bound_args.args instead of mutating bn_node.args
    normalized_args = bound_args.args

    # We can only use efficient conv-bn for eval mode with track_running_stats
    # normalized_args[5] is the "training" argument
    training_arg = normalized_args[5]

    # Safety check: if 'training' is a symbolic Node (from tracing/export),
    # we cannot optimize since we don't know the value at compile time.
    if isinstance(training_arg, torch.fx.Node):
        return

    if training_arg:
        return

    # Check if the input is Conv
    input_node = normalized_args[0]

    if input_node.op != "call_function":  # type: ignore[union-attr]
        return

    input_fn = input_node.target  # type: ignore[arg-type, union-attr]
    supported_convs = [
        torch._C._nn.linear,
        torch.conv1d,
        torch.conv2d,
        torch.conv3d,
        torch.conv_transpose1d,
        torch.conv_transpose2d,
        torch.conv_transpose3d,
    ]

    if not any(input_fn is cls for cls in supported_convs):
        return

    conv_node = input_node
    # Output of conv is used by other nodes, cannot optimize
    if len(conv_node.users) > 1:  # type: ignore[union-attr]
        return

    counters["inductor"]["efficient_conv_bn_eval"] += 1

    with graph.inserting_before(bn_node):
        # prepare args for the fused function
        bn_running_mean = normalized_args[1]
        bn_running_var = normalized_args[2]
        bn_weight = normalized_args[3]
        bn_bias = normalized_args[4]
        bn_eps = normalized_args[7]
        if len(conv_node.args) < 2:  # type: ignore[union-attr]
            raise AssertionError(
                f"expected at least 2 conv_node args, got {len(conv_node.args)}"  # type: ignore[union-attr]
            )
        conv_input = conv_node.args[0]  # type: ignore[union-attr]
        conv_weight = conv_node.args[1]  # type: ignore[union-attr]
        conv_bias = conv_node.args[2] if len(conv_node.args) >= 3 else None  # type: ignore[union-attr]
        conv_remaining_args = conv_node.args[3:]  # type: ignore[union-attr]
        args = (
            bn_weight,
            bn_bias,
            bn_running_mean,
            bn_running_var,
            bn_eps,
            conv_node.target,  # type: ignore[union-attr]
            conv_weight,
            conv_bias,
            conv_input,
            conv_remaining_args,
        )

        # create a new node
        new_node = graph.create_node(
            op="call_function",
            target=efficient_conv_bn_eval_decomposed,
            args=args,  # type: ignore[arg-type]
            name="efficient_conv_bn_eval",
        )

    # this node replaces the original conv + bn, and therefore
    # should replace the uses of bn_node
    bn_node.replace_all_uses_with(new_node)
    # take care of the deletion order:
    # delete bn_node first, and then conv_node
    graph.erase_node(bn_node)
    graph.erase_node(conv_node)  # type: ignore[arg-type]

    return


@register_graph_pattern(
    CallFunctionVarArgs(
        [
            torch.ops.aten.batch_norm.default,
        ]
    ),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=efficient_conv_bn_eval_pass,
    extra_check=lambda match: not inductor_config.freezing
    and inductor_config.efficient_conv_bn_eval_fx_passes,
)
def efficient_conv_bn_eval_graph_transform_decomposed(match: Match, *args, **kwargs):
    bn_node = match.nodes[0]
    graph = match.graph
    if len(bn_node.args) != 9:
        raise AssertionError(f"expected 9 bn_node args, got {len(bn_node.args)}")

    # We can only use efficient conv-bn for eval mode with track_running_stats
    # bn_node.args is `training`
    if bn_node.args[-4]:
        return

    # Check if the input is Conv
    input_node = bn_node.args[0]

    if input_node.op != "call_function":  # type: ignore[union-attr]
        return

    input_fn = input_node.target  # type: ignore[arg-type, union-attr]
    supported_convs = [
        torch.ops.aten.linear.default,
        torch.ops.aten.conv1d.default,
        torch.ops.aten.conv2d.default,
        torch.ops.aten.conv3d.default,
        torch.ops.aten.conv_transpose1d.default,
        torch.ops.aten.conv_transpose2d.input,
        torch.ops.aten.conv_transpose3d.input,
    ]

    if not any(input_fn is cls for cls in supported_convs):
        return

    conv_node = input_node
    # Output of conv is used by other nodes, cannot optimize
    if len(conv_node.users) > 1:  # type: ignore[union-attr]
        return

    counters["inductor"]["efficient_conv_bn_eval"] += 1

    with graph.inserting_before(bn_node):
        # prepare args for the fused function
        bn_weight = bn_node.args[1]
        bn_bias = bn_node.args[2]
        bn_running_mean = bn_node.args[3]
        bn_running_var = bn_node.args[4]
        bn_eps = bn_node.args[7]
        if len(conv_node.args) < 2:  # type: ignore[union-attr]
            raise AssertionError(
                f"expected at least 2 conv_node args, got {len(conv_node.args)}"  # type: ignore[union-attr]
            )
        conv_input = conv_node.args[0]  # type: ignore[union-attr]
        conv_weight = conv_node.args[1]  # type: ignore[union-attr]
        conv_bias = conv_node.args[2] if len(conv_node.args) >= 3 else None  # type: ignore[union-attr]
        conv_remaining_args = conv_node.args[3:]  # type: ignore[union-attr]
        args = (
            bn_weight,
            bn_bias,
            bn_running_mean,
            bn_running_var,
            bn_eps,
            conv_node.target,  # type: ignore[union-attr]
            conv_weight,
            conv_bias,
            conv_input,
            conv_remaining_args,
        )

        # create a new node
        new_node = graph.create_node(
            op="call_function",
            target=efficient_conv_bn_eval_decomposed,
            args=args,  # type: ignore[arg-type]
            name="efficient_conv_bn_eval",
        )

    # this node replaces the original conv + bn, and therefore
    # should replace the uses of bn_node
    bn_node.replace_all_uses_with(new_node)
    # take care of the deletion order:
    # delete bn_node first, and then conv_node
    graph.erase_node(bn_node)
    graph.erase_node(conv_node)  # type: ignore[arg-type]

    return


@register_graph_pattern(
    CallModuleVarArgs(
        [
            nn.modules.batchnorm._BatchNorm,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
        ],
    ),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=efficient_conv_bn_eval_pass,
    extra_check=lambda match: not inductor_config.freezing
    and inductor_config.efficient_conv_bn_eval_fx_passes,
)
def efficient_conv_bn_eval_graph_transform(match: Match, *args, **kwargs):
    # We matched a BN node
    bn_node = match.nodes[0]
    graph = match.graph
    gm = graph.owning_module
    bn_mod = getattr(gm, bn_node.target)  # type: ignore[arg-type]

    # We can only use efficient conv-bn for eval mode with track_running_stats
    if not bn_mod.track_running_stats or bn_mod.training:
        return

    # Check if the input is Conv
    if bn_node.args:
        input_node = bn_node.args[0]
    else:
        input_node = bn_node.kwargs["input"]
    if input_node.op != "call_module":  # type: ignore[union-attr]
        return
    if not hasattr(gm, input_node.target):  # type: ignore[arg-type, union-attr]
        return
    input_mod = getattr(gm, input_node.target)  # type: ignore[arg-type, union-attr]
    supported_convs = [
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
    ]
    if not any(isinstance(input_mod, cls) for cls in supported_convs):
        return
    conv_node = input_node
    # Output of conv is used by other nodes, cannot optimize
    if len(conv_node.users) > 1:  # type: ignore[union-attr]
        return

    # Find a pair of conv and bn computation nodes to optimize.
    counters["inductor"]["efficient_conv_bn_eval"] += 1

    with graph.inserting_before(conv_node):  # type: ignore[arg-type]
        # create `get_attr` node to access modules
        # note that we directly call `create_node` to fill the `name`
        # argument. `graph.get_attr` and
        # `graph.call_function` does not allow the `name` argument.
        conv_get_node = graph.create_node(
            op="get_attr",
            target=conv_node.target,  # type: ignore[union-attr]
            name="get_conv",
        )
        bn_get_node = graph.create_node(
            op="get_attr", target=bn_node.target, name="get_bn"
        )
        if conv_node.args:  # type: ignore[union-attr]
            conv_input = conv_node.args[0]  # type: ignore[union-attr]
        else:
            conv_input = conv_node.kwargs["input"]  # type: ignore[union-attr]
        # prepare args for the fused function
        args = (bn_get_node, conv_get_node, conv_input)
        # create a new node
        new_node = graph.create_node(
            op="call_function",
            target=efficient_conv_bn_eval,
            args=args,
            name="efficient_conv_bn_eval",
        )
    # this node replaces the original conv + bn, and therefore
    # should replace the uses of bn_node
    bn_node.replace_all_uses_with(new_node)
    # take care of the deletion order:
    # delete bn_node first, and then conv_node
    graph.erase_node(bn_node)
    graph.erase_node(conv_node)  # type: ignore[arg-type]
<|endoftext|># mypy: allow-untyped-defs
from dataclasses import dataclass
from typing import Any

import torch
from torch import SymBool, SymFloat, SymInt
from torch.types import py_sym_types
from torch.utils._ordered_set import OrderedSet


@dataclass
class _SymExprHash:
    """
    Hash for a py_sym_types that will use the underlying sympy expression
    """

    sym_obj: SymInt | SymFloat | SymBool

    def __hash__(self) -> int:
        return hash((type(self.sym_obj), self.sym_obj.node.expr))

    def __eq__(self, value) -> bool:
        if not isinstance(value, _SymExprHash):
            return False
        return self.sym_obj.node.expr == value.sym_obj.node.expr


class _SymHashingDict:
    """
    Wrapper around a dictionary that will convert sym types to hash with _SymExprHash and reuse
    existing sym proxies.

    SymPy hash is not always reliable so optimistically hash sympy expression, and if those fail,
    fallback to symnodes.
    """

    def __init__(self):
        self.sym_hash_dict = {}

    def __setitem__(self, key, value):
        self.sym_hash_dict.__setitem__(self._wrap_to_sym_expr_hash(key), value)

    def __getitem__(self, key):
        return self.sym_hash_dict[self._wrap_to_sym_expr_hash(key)]

    def __contains__(self, key):
        return self._wrap_to_sym_expr_hash(key) in self.sym_hash_dict

    def get(self, key, default=None):
        return self.sym_hash_dict.get(self._wrap_to_sym_expr_hash(key), default)

    def _wrap_to_sym_expr_hash(self, key):
        return _SymExprHash(key) if isinstance(key, py_sym_types) else key


def dedupe_symints(graph: torch.fx.Graph):
    """
    Dedupes sym ints in the graph to nodes are resolvable to symint graph inputs.

    We only dedupe from graph inputs to avoid adding a potential dependency in the forward
    from the backward.

    """

    sym_dict = _SymHashingDict()
    resolvable_from_input_symints = OrderedSet[Any]()

    for node in graph.nodes:
        val = node.meta.get("val", None)
        if val is None or not isinstance(val, py_sym_types):
            continue

        if node.op == "placeholder":
            resolvable_from_input_symints.add(node)
            sym_dict[val] = node
        elif existing_node := sym_dict.get(val):
            node.replace_all_uses_with(existing_node)
            graph.erase_node(node)
        elif all(n in resolvable_from_input_symints for n in node.all_input_nodes):
            sym_dict[val] = node
            resolvable_from_input_symints.add(node)
<|endoftext|>import collections
import contextlib
import logging
import operator
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Literal, TYPE_CHECKING, TypeAlias

import sympy

import torch
import torch.distributed as dist
import torch.utils._pytree as pytree
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import detect_fake_mode
from torch._inductor.comm_analysis import (
    get_collective_type_from_kernel_name,
    NCCL_COLL,
)
from torch._inductor.fx_passes.utils import BitsetAncestors
from torch._inductor.runtime.runtime_utils import dynamo_timed
from torch._logging import trace_structured
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.traceback import NodeSource, NodeSourceAction
from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import GroupName


logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

overlap_log = torch._logging.getArtifactLogger(__name__, "overlap")


def _resolve_group_name(group_name: Any) -> "GroupName":
    """Resolve group_name to a GroupName string.

    In compile-on-one-rank graphs, collective ops receive their
    group_name argument as an FX Node reference (pointing to a
    mesh_get_process_group call) rather than a string literal. For
    bucketing key purposes we resolve via the ProcessGroup stored in
    node.meta["val"].
    """
    if isinstance(group_name, str):
        return group_name  # pyrefly: ignore [bad-return]
    pg = group_name.meta["val"]
    return pg.group_name


BucketMode: TypeAlias = Literal[
    "default", "custom_ops", "custom_ops_multidtype", "coalesced"
]


def _default_bucket_mode() -> BucketMode:
    from torch._inductor import config

    return config.aten_distributed_optimizations.bucket_mode or "default"


# Helper functions moved to top for better organization
def _ag_group_key(node: torch.fx.Node) -> tuple[str, torch.dtype]:  # type: ignore[name-defined]
    _, group_size, group_name = node.args
    dtype = node.meta["val"].dtype
    return (_resolve_group_name(group_name), dtype)


def _ag_group_key_multidtype(node: torch.fx.Node) -> tuple[str]:
    _, group_size, group_name = node.args
    return (_resolve_group_name(group_name),)


def _rs_group_key(node: torch.fx.Node) -> tuple[str, str, torch.dtype]:  # type: ignore[name-defined]
    _, reduce_op, group_size, group_name = node.args
    dtype = node.meta["val"].dtype
    if not isinstance(reduce_op, str):
        raise AssertionError(f"expected reduce_op to be str, got {type(reduce_op)}")
    return (_resolve_group_name(group_name), reduce_op, dtype)


def _ar_group_key(node: torch.fx.Node) -> tuple[str, str, torch.dtype]:
    _, reduce_op, group_name = node.args
    dtype = node.meta["val"].dtype
    if not isinstance(reduce_op, str):
        raise AssertionError(f"expected reduce_op to be str, got {type(reduce_op)}")
    return (_resolve_group_name(group_name), reduce_op, dtype)


def _compute_foreach_groups(
    ag_ins: list[torch.Tensor],
    out_dtypes: list[torch.dtype],  # type: ignore[name-defined]
) -> list[int] | None:
    """
    Compute groups of indices that have the same src/dst dtype and shape.

    Groups tensors by (src_dtype, dst_dtype, shape) to avoid falling back to the foreach slow path.

    Returns a flat list with -1 as group delimiter, or None if only one group exists.
    For example, groups [[0, 2], [1]] would be encoded as [0, 2, -1, 1].
    """
    groups: defaultdict[tuple[torch.dtype, torch.dtype, tuple[int, ...]], list[int]] = (
        defaultdict(list)
    )
    for i, (ag_in, out_dtype) in enumerate(zip(ag_ins, out_dtypes)):
        shape = tuple(
            _hint_int_or_raise(s, context="all-gather foreach grouping")
            for s in ag_in.shape
        )
        key = (ag_in.dtype, out_dtype, shape)
        groups[key].append(i)

    if len(groups) <= 1:
        return None

    # Encode as flat list with -1 as delimiter
    result: list[int] = []
    for i, group_indices in enumerate(groups.values()):
        result.extend(group_indices)
        if i < len(groups) - 1:
            result.append(-1)

    return result


# Bucketing has two separate shape contracts:
#
# 1. Semantic tensor shapes stay symbolic.  The traced bucket merge graph must
#    preserve runtime tensor expressions such as numel(), split sizes,
#    torch.empty extents, narrow offsets/lengths, and reshape shapes.  Those
#    values describe actual tensor semantics and must not be specialized from
#    optimization hints.
#
# 2. Optimization-policy choices may use hints.  Bucket byte accounting and
#    foreach grouping need Python integers to decide how to group collectives.
#    For those policy-only decisions, concrete ints, backed SymInts, hinted
#    unbacked SymInts, and derived expressions from hinted symbols are valid.
#    Unhinted symbolic values fail fast instead of forcing guards or guessing.
#
# In short: hints may decide which bucket/group we choose, but they must never
# replace symbolic sizes in the graph we trace for the bucketed collective.
def _hint_int_or_raise(value: object, *, context: str) -> int:
    if type(value) is int:
        return value

    if not isinstance(value, torch.SymInt):
        raise AssertionError(f"Expected int or SymInt for {context}, got {type(value)}")

    node = value.node
    if node._hint is not None:
        return int(node._hint)
    shape_env = node.shape_env
    if shape_env is None:
        raise AssertionError(f"ShapeEnv is required to hint {context}: {value}")

    expr = sympy.sympify(node.expr).xreplace(shape_env.replacements)
    expr = expr.xreplace(shape_env.backed_var_to_val)
    expr = expr.xreplace(shape_env.var_to_hint_override)
    if isinstance(expr, sympy.Expr):
        expr = expr.expand(identity=True)
    if getattr(expr, "free_symbols", None):
        raise RuntimeError(
            f"Could not extract optimization hint for {context}: {value}. "
            "Collective bucketing requires hinted symbolic sizes for policy "
            "decisions."
        )
    return int(expr)


def _numel_hint_or_raise(tensor: torch.Tensor, *, context: str) -> int:
    return _hint_int_or_raise(tensor.numel(), context=context)


def _size_bytes_hint_or_raise(
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    context: str,
) -> int:
    element_size = tensor.element_size() if dtype is None else dtype.itemsize
    return _numel_hint_or_raise(tensor, context=context) * element_size


def _get_collective_node_from_wait(node: torch.fx.Node) -> torch.fx.Node | None:
    """Given a wait node, return the collective it waits on.

    Handles both standard (wait -> collective) and coalesced
    (wait -> getitem -> coalesced_collective) patterns.
    Returns None if the node is not a wait on a recognized NCCL collective.
    """
    if not is_wait_tensor(node):
        return None
    arg = node.args[0]
    if not isinstance(arg, torch.fx.Node):
        raise AssertionError(f"expected arg to be a Node, got {type(arg)}")
    if arg.op != "call_function":
        return None
    if arg.target is operator.getitem:
        if not isinstance(arg.args[0], torch.fx.Node):
            raise AssertionError(
                f"expected arg.args[0] to be a Node, got {type(arg.args[0])}"
            )
        arg = arg.args[0]
        if arg.op != "call_function":
            return None
    if not isinstance(arg.target, Callable):
        return None
    # pyrefly: ignore [missing-attribute]
    coll: NCCL_COLL = get_collective_type_from_kernel_name(arg.target.name())
    if coll == NCCL_COLL.UNSUPPORTED:
        return None
    return arg


def _schedulable_wait_node(node: torch.fx.Node) -> bool:
    """Check if this wait node is schedulable (waits on a recognized NCCL collective)."""
    return _get_collective_node_from_wait(node) is not None


def _populate_node_meta(
    bucket_nodes: list[torch.fx.Node], new_nodes: list[torch.fx.Node]
):
    if bucket_nodes:
        for n in new_nodes:
            # For the following keys, we only store the information of the first node so
            # gm.print_readable shows some information
            # Full information is stored in "bucketing_{key}_sources"
            for key, default in [
                ("nn_module_stack", ""),
                ("fwd_nn_module_stack", ""),
                ("stack_trace", ""),
                ("custom", {}),
            ]:
                n.meta[key] = bucket_nodes[0].meta.get(key, default)

                # Collect sources from all bucket nodes for this metadata key, for debugging purposes only
                bucketing_sources_key = f"bucketing_{key}_sources"
                # Use set to remove duplicates
                if key == "stack_trace":
                    sources = OrderedSet(
                        [
                            node.meta.get(key, default)
                            for node in bucket_nodes
                            if node.meta.get(key, default)
                        ]
                    )
                else:
                    # type might not be hashable
                    sources = [
                        node.meta.get(key, default)
                        for node in bucket_nodes
                        if node.meta.get(key, default)
                    ]
                n.meta[bucketing_sources_key] = sources

            # used by inductor provenance tracking
            n.meta["from_node"] = [
                NodeSource(
                    original_node,
                    "bucketing_pass",
                    [NodeSourceAction.CREATE, NodeSourceAction.REPLACE],
                )
                for original_node in bucket_nodes
            ]


def _meta_arg(arg: object) -> object:
    if isinstance(arg, torch.fx.Node):
        return arg.meta.get("val", arg)
    return arg


def _same_tensor_metadata(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    def same_dim(lhs_dim: object, rhs_dim: object) -> bool:
        from torch.fx.experimental.symbolic_shapes import statically_known_true

        try:
            return statically_known_true(lhs_dim == rhs_dim)
        except Exception:
            return lhs_dim == rhs_dim

    def same_dims(lhs_dims: tuple[object, ...], rhs_dims: tuple[object, ...]) -> bool:
        return len(lhs_dims) == len(rhs_dims) and all(
            same_dim(lhs_dim, rhs_dim) for lhs_dim, rhs_dim in zip(lhs_dims, rhs_dims)
        )

    return (
        lhs.dtype == rhs.dtype
        and lhs.device == rhs.device
        and lhs.layout == rhs.layout
        and same_dims(tuple(lhs.shape), tuple(rhs.shape))
        and same_dims(tuple(lhs.stride()), tuple(rhs.stride()))
        and same_dim(lhs.storage_offset(), rhs.storage_offset())
    )


def _same_metadata(lhs: object, rhs: object) -> bool:
    lhs_leaves, lhs_spec = pytree.tree_flatten(lhs)
    rhs_leaves, rhs_spec = pytree.tree_flatten(rhs)
    if lhs_spec != rhs_spec or len(lhs_leaves) != len(rhs_leaves):
        return False
    for lhs_leaf, rhs_leaf in zip(lhs_leaves, rhs_leaves):
        if isinstance(lhs_leaf, torch.Tensor) or isinstance(rhs_leaf, torch.Tensor):
            if not isinstance(lhs_leaf, torch.Tensor) or not isinstance(
                rhs_leaf, torch.Tensor
            ):
                return False
            if not _same_tensor_metadata(lhs_leaf, rhs_leaf):
                return False
    return True


def _recompute_changed_user_metadata(start_users: list[torch.fx.Node]) -> None:
    """
    Repair fake metadata after bucketing replaces a value with a layout-different
    equivalent. The replacement is semantically valid, but downstream stride
    metadata is no longer valid until consumers are re-run from metadata.
    """
    worklist = collections.deque(start_users)
    queued: OrderedSet[torch.fx.Node] = OrderedSet(start_users)
    while worklist:
        node = worklist.popleft()
        queued.discard(node)
        if node.op != "call_function" or not callable(node.target):
            continue
        if "val" not in node.meta:
            continue

        args = pytree.tree_map(_meta_arg, node.args)
        kwargs = pytree.tree_map(_meta_arg, node.kwargs)
        if any(
            isinstance(leaf, torch.fx.Node)
            for leaf in pytree.tree_leaves((args, kwargs))
        ):
            continue

        fake_mode = detect_fake_mode((node.meta.get("val"), args, kwargs))
        try:
            with fake_mode if fake_mode is not None else contextlib.nullcontext():
                new_val = node.target(*args, **kwargs)
        except Exception:
            logger.debug(
                "Skipping metadata repair for bucketing user %s",
                node.name,
                exc_info=True,
            )
            continue

        if _same_metadata(node.meta["val"], new_val):
            continue

        node.meta["val"] = new_val
        for user in node.users:
            if user not in queued:
                queued.add(user)
                worklist.append(user)


def bucket_key(node: torch.fx.Node, mode: BucketMode | None = None) -> object | None:
    if is_all_gather_into_tensor(node):
        group_key_fn = (
            _ag_group_key_multidtype if mode and "multidtype" in mode else _ag_group_key
        )
        return group_key_fn(node)
    elif is_reduce_scatter_tensor(node):
        return _rs_group_key(node)
    elif is_all_reduce_tensor(node):
        return _ar_group_key(node)
    else:
        return None


def pick_bucket_dtype(dtypes: list[torch.dtype]) -> torch.dtype:  # type: ignore[name-defined]
    if len(dtypes) == 0:
        raise AssertionError("expected at least one dtype, got empty list")
    return min(dtypes, key=operator.attrgetter("itemsize"))


def bucket_cap_mb_by_bucket_idx_default(bucket_id: int) -> float:
    """
    Determine the size of a bucket based on its ID.

    Args:
    bucket_id (int): The ID of the bucket.

    Returns:
    float: The size of the bucket.
    """
    return 2000.0


def bucket_all_gather(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float] | None = None,
    mode: BucketMode | None = None,
) -> None:
    mode = mode or _default_bucket_mode()
    if bucket_cap_mb_by_bucket_idx is None:
        from torch._inductor.fx_passes.bucketing import (
            bucket_cap_mb_by_bucket_idx_default,  # pyrefly: ignore [missing-module-attribute]
        )

        bucket_cap_mb_by_bucket_idx = bucket_cap_mb_by_bucket_idx_default
    ag_buckets = bucket_all_gather_by_mb(gm, bucket_cap_mb_by_bucket_idx, None, mode)
    if len(ag_buckets) == 0:
        return
    merge_all_gather(gm, ag_buckets, mode)


def bucket_reduce_scatter(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float] | None = None,
    mode: BucketMode | None = None,
) -> None:
    mode = mode or _default_bucket_mode()
    if bucket_cap_mb_by_bucket_idx is None:
        from torch._inductor.fx_passes.bucketing import (
            bucket_cap_mb_by_bucket_idx_default,  # pyrefly: ignore [missing-module-attribute]
        )

        bucket_cap_mb_by_bucket_idx = bucket_cap_mb_by_bucket_idx_default
    rs_buckets = bucket_reduce_scatter_by_mb(
        gm, bucket_cap_mb_by_bucket_idx, None, mode
    )
    if len(rs_buckets) == 0:
        return
    merge_reduce_scatter(gm, rs_buckets, mode)


def is_all_gather_into_tensor(node: torch.fx.Node) -> bool:  # type: ignore[arg-type]
    return node.op == "call_function" and (
        node.target == torch.ops._c10d_functional.all_gather_into_tensor.default
        or node.target == torch.ops._c10d_functional.all_gather_into_tensor_out.default
    )


def is_reduce_scatter_tensor(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is torch.ops._c10d_functional.reduce_scatter_tensor.default
    )


def is_wait_tensor(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is torch.ops._c10d_functional.wait_tensor.default
    )


def is_all_reduce_tensor(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is torch.ops._c10d_functional.all_reduce.default
    )


def is_all_to_all_tensor(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is torch.ops._c10d_functional.all_to_all_single.default
    )


def get_collective_type(node: torch.fx.Node) -> str:
    """Get the collective type name for a node."""
    if is_all_gather_into_tensor(node):
        return "all_gather"
    elif is_reduce_scatter_tensor(node):
        return "reduce_scatter"
    elif is_all_reduce_tensor(node):
        return "all_reduce"
    return ""


def get_full_bucket_key(
    node: torch.fx.Node, bucket_mode: BucketMode | None
) -> tuple[str, Any]:
    """Get the full bucket key including collective type and bucket key."""
    return (get_collective_type(node), bucket_key(node, mode=bucket_mode))


def is_wait_tensor_from_all_gather_into_tensor(node: torch.fx.Node) -> bool:
    return is_wait_tensor(node) and is_all_gather_into_tensor(node.args[0])  # type: ignore[arg-type]


def is_fsdp_all_gather(
    node: torch.fx.Node,
    all_node_ancestors: BitsetAncestors | None = None,
) -> bool:
    """Check if an all_gather derives from exactly one placeholder (parameter).

    When all_node_ancestors is provided, uses it for O(|ancestors|) lookup.
    Otherwise delegates to the BFS implementation in fsdp.py.
    """
    if not is_all_gather_into_tensor(node):
        return False
    if all_node_ancestors is not None:
        phs = (
            a for a in all_node_ancestors.iter_ancestors(node) if a.op == "placeholder"
        )
        return next(phs, None) is not None and next(phs, None) is None
    from torch._inductor.fx_passes.fsdp import is_fsdp_all_gather as _is_fsdp_all_gather

    return _is_fsdp_all_gather(node)


def is_fsdp_reduce_scatter(node: torch.fx.Node) -> bool:
    """
    Check if a reduce_scatter node is FSDP-related by verifying its output flows
    directly to graph outputs through only unary ops (e.g., to_copy, wait).
    """
    if not is_reduce_scatter_tensor(node):
        return False

    visited: OrderedSet[torch.fx.Node] = OrderedSet()
    stack = [node]

    while stack:
        curr = stack.pop()
        if curr in visited:
            continue
        visited.add(curr)

        for user in curr.users:
            if user.op == "output":
                continue
            # Non-unary op means computation with external data
            if len(user.all_input_nodes) != 1:
                return False
            stack.append(user)

    return True


def collect_node_descendants(
    graph: torch.fx.Graph,
) -> dict[torch.fx.Node, OrderedSet[torch.fx.Node]]:
    """
    Collects the descendants of each node in the graph.
    Args:
        graph (torch.fx.Graph): The graph to collect descendants from.
    Returns:
        dict[torch.fx.Node, OrderedSet[torch.fx.Node]]: A dictionary mapping each node to its descendants.
    """
    node_descendants: dict[torch.fx.Node, OrderedSet[torch.fx.Node]] = (
        collections.defaultdict(OrderedSet)
    )
    outdegree = collections.defaultdict(int)
    queue = []

    for node in graph.nodes:
        n_outdegree = len(node.users)
        if n_outdegree == 0:
            queue.append(node)
        else:
            outdegree[node] = len(node.users)

    while queue:
        node = queue.pop()
        for input_node in node.all_input_nodes:
            node_descendants[input_node] |= node_descendants[node]
            node_descendants[input_node].add(node)
            outdegree[input_node] -= 1

            if outdegree[input_node] == 0:
                queue.append(input_node)

    return node_descendants


def greedy_bucket_collective_by_mb(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float],
    filter_node: Callable[[torch.fx.Node], bool],
    node_group_key: Callable[[torch.fx.Node], Any],
    filter_wait_node: Callable[[torch.fx.Node], bool] | None = None,
) -> list[list[torch.fx.Node]]:
    """
    Bucketing adjacent collectives with equal node_group_key.
    We can not bucket non adjacent collectives,
    as this will effectively change the order of collectives.
    Reordering can lead to different order on different ranks.
    """
    g = gm.graph
    found_candidates = False
    for node in g.nodes:
        if filter_node(node):
            found_candidates = True
            break
    if not found_candidates:
        return []

    # Build forward adjacency list for incremental descendant tracking
    children: dict[torch.fx.Node, list[torch.fx.Node]] = collections.defaultdict(list)
    for node in g.nodes:
        for inp in node._input_nodes:
            children[inp].append(node)

    nodes_groups: list[list[torch.fx.Node]] = []
    cur_group: list[torch.fx.Node] = []
    cur_group_key = None

    for node in g.nodes:
        if is_wait_tensor(node) and filter_node(node.args[0]):
            if (filter_wait_node is None) or filter_wait_node(node):
                coll_node = node.args[0]
                group_key = node_group_key(coll_node)
                if group_key == cur_group_key:
                    cur_group.append(coll_node)
                else:
                    if len(cur_group) > 1:
                        nodes_groups.append(cur_group)
                    cur_group = [coll_node]
                    cur_group_key = group_key

    if len(cur_group) > 1:
        nodes_groups.append(cur_group)

    def _add_descendants(node: torch.fx.Node, desc: OrderedSet[torch.fx.Node]) -> None:
        """Forward BFS from node, adding all reachable nodes to desc."""
        stack = [node]
        while stack:
            n = stack.pop()
            for child in children[n]:
                if child not in desc:
                    desc.add(child)
                    stack.append(child)

    buckets: list[list[torch.fx.Node]] = []
    for nodes in nodes_groups:
        cur_bucket: list[torch.fx.Node] = []
        cur_bucket_descendents: OrderedSet[torch.fx.Node] = OrderedSet()
        cur_bucket_size_bytes: int = 0
        cur_bucket_id: int = 0
        bucket_size_bytes = int(
            bucket_cap_mb_by_bucket_idx(cur_bucket_id) * 1024 * 1024
        )
        for node in nodes:
            if node in cur_bucket_descendents:
                # if there is a path from node to the current bucket, we cannot horizontally fuse (bucket)
                continue
            if "val" not in node.meta:
                raise AssertionError(f"expected 'val' in node.meta for {node}")
            n_val = node.meta["val"]
            out_size_bytes = _size_bytes_hint_or_raise(
                n_val, context="collective bucket output size"
            )
            n_input_val = node.all_input_nodes[0].meta["val"]
            in_size_bytes = _size_bytes_hint_or_raise(
                n_input_val, context="collective bucket input size"
            )
            size_bytes = max(out_size_bytes, in_size_bytes)
            if cur_bucket_size_bytes + size_bytes > bucket_size_bytes and cur_bucket:
                # Current bucket is full, create new bucket
                if len(cur_bucket) > 1:
                    buckets.append(cur_bucket)
                cur_bucket = []
                cur_bucket_size_bytes = 0
                cur_bucket_id += 1
                cur_bucket_descendents = OrderedSet()
            cur_bucket_size_bytes += size_bytes
            cur_bucket.append(node)
            _add_descendants(node, cur_bucket_descendents)
        if len(cur_bucket) > 1:
            buckets.append(cur_bucket)
    return buckets


def bucket_all_gather_by_mb(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float],
    filter_wait_node: Callable[[torch.fx.Node], bool] | None = None,
    mode: BucketMode | None = None,
) -> list[list[torch.fx.Node]]:
    """
    Identifies all all_gather nodes and groups them into buckets,
    based on size limit `bucket_cap_mb_by_bucket_idx`.

    Args:
        gm (torch.fx.GraphModule): GraphModule where to bucket all_gathers.
        bucket_cap_mb_by_bucket_idx (Callable[[int], float]): Callable to specify cap of the bucket
            in megabytes by bucket idx.  The idea of `bucket_cap_mb_by_bucket_idx` is to allow
            to specify different sizes of the buckets at the start,
            as first all_gather is usually exposed.  Interface of bucket_cap_mb_by_bucket_idx
            is `bucket_cap_mb_by_bucket_idx_default` function that is default value for `bucket_cap_mb_by_bucket_idx`.
        filter_wait_node (Callable[[torch.fx.Node], bool] | None): If specified,
            only all_gather nodes with wait_node that satisfy `filter_wait_node` will be bucketed.

    Returns:
        list[list[torch.fx.Node]]: List of buckets, where each bucket is a list of all_gather nodes.
    """
    mode = mode or _default_bucket_mode()

    group_key_fn = (
        _ag_group_key_multidtype if mode and "multidtype" in mode else _ag_group_key
    )

    return greedy_bucket_collective_by_mb(
        gm,
        bucket_cap_mb_by_bucket_idx,
        is_all_gather_into_tensor,
        group_key_fn,
        filter_wait_node,
    )


def bucket_reduce_scatter_by_mb(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float],
    filter_wait_node: Callable[[torch.fx.Node], bool] | None = None,
    mode: BucketMode | None = None,
) -> list[list[torch.fx.Node]]:
    """
    Identifies all reduce_scatter nodes and groups them into buckets,
        based on size limit `bucket_cap_mb_by_bucket_idx`.

    Args:
        gm (torch.fx.GraphModule): GraphModule where to bucket reduce_scatters.
        bucket_cap_mb_by_bucket_idx (Callable[[int], float]): Callable to specify cap of the bucket
            in megabytes by bucket idx.  The idea of `bucket_cap_mb_by_bucket_idx` is to allow
            to specify different sizes of the buckets.
        filter_wait_node (Callable[[torch.fx.Node], bool] | None): If specified,
            only reduce_scatter nodes with wait_node that satisfy `filter_wait_node` will be bucketed.

    Returns:
        list[list[torch.fx.Node]]: List of buckets, where each bucket is a list of reduce_scatter nodes.
    """
    mode = mode or _default_bucket_mode()

    if mode is not None and "multidtype" in mode:
        raise AssertionError("reduce scatter bucketing does not support multidtype")

    return greedy_bucket_collective_by_mb(
        gm,
        bucket_cap_mb_by_bucket_idx,
        is_reduce_scatter_tensor,
        _rs_group_key,
        filter_wait_node,
    )


def bucket_all_reduce_by_mb(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float],
    filter_wait_node: Callable[[torch.fx.Node], bool] | None = None,
) -> list[list[torch.fx.Node]]:
    return greedy_bucket_collective_by_mb(
        gm,
        bucket_cap_mb_by_bucket_idx,
        is_all_reduce_tensor,
        _ar_group_key,
        filter_wait_node,
    )


def bucket_all_reduce(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float] | None = None,
    mode: str | None = None,
) -> None:
    if bucket_cap_mb_by_bucket_idx is None:
        from torch._inductor.fx_passes.bucketing import (
            bucket_cap_mb_by_bucket_idx_default,  # pyrefly: ignore [missing-module-attribute]
        )

        bucket_cap_mb_by_bucket_idx = bucket_cap_mb_by_bucket_idx_default
    ar_buckets = bucket_all_reduce_by_mb(gm, bucket_cap_mb_by_bucket_idx)
    if len(ar_buckets) == 0:
        return
    for bucket in ar_buckets:
        merge_all_reduce_bucket(gm.graph, bucket, mode)


@torch.library.custom_op("bucketing::_pre_bucket_reduce_scatter", mutates_args={})
def _pre_bucket_reduce_scatter(
    rs_ins: list[torch.Tensor],
    group_size: int,
) -> torch.Tensor:
    rs_ins_flattened = [x.reshape(group_size, -1) for x in rs_ins]
    new_rs_in = torch.cat(rs_ins_flattened, dim=1).flatten()
    return new_rs_in


def _pre_bucket_reduce_scatter_fake(
    rs_ins: list[torch.Tensor],
    group_size: int,
) -> torch.Tensor:
    out_numel = sum(rs_in.numel() for rs_in in rs_ins)
    return torch.empty((out_numel,), device=rs_ins[0].device, dtype=rs_ins[0].dtype)


_pre_bucket_reduce_scatter.register_fake(_pre_bucket_reduce_scatter_fake)


def reduce_scatter_merge_fn_to_trace_custom_ops(
    rs_ins: list[torch.Tensor],
    group_name: Any,
    group_size: int,
    reduce_op: str,
    reduce_dtype: torch.dtype,  # type: ignore[name-defined]
    device: torch.device,  # type: ignore[name-defined]
) -> list[torch.Tensor]:  # type: ignore[no-untyped-def]
    new_out_sizes = [(x.shape[0] // group_size,) + x.shape[1:] for x in rs_ins]
    new_out_numels = [x.numel() // group_size for x in rs_ins]

    new_rs_in = torch.ops.bucketing._pre_bucket_reduce_scatter(rs_ins, group_size)

    # TODO - either use torch.cat or make sure inductor foreach codegen
    # fires more reliably
    new_rs_out = torch.ops.c10d_functional.wait_tensor(
        torch.ops._c10d_functional.reduce_scatter_tensor.default(
            new_rs_in, reduce_op, group_size, group_name
        )
    )
    new_out_flat = new_rs_out.split(new_out_numels, 0)
    new_outs = [x.reshape(s) for x, s in zip(new_out_flat, new_out_sizes)]
    return new_outs


def reduce_scatter_merge_fn_to_trace(
    rs_ins: list[torch.Tensor],
    group_name: Any,
    group_size: int,
    reduce_op: str,
    reduce_dtype: torch.dtype,  # type: ignore[name-defined]
    device: torch.device,  # type: ignore[name-defined]
) -> list[torch.Tensor]:  # type: ignore[no-untyped-def]
    rs_ins_flattened = [x.reshape(group_size, -1) for x in rs_ins]

    new_out_sizes = [(x.shape[0] // group_size,) + x.shape[1:] for x in rs_ins]
    new_out_numels = [x.numel() // group_size for x in rs_ins]

    new_rs_in = torch.cat(rs_ins_flattened, dim=1).flatten()

    new_rs_out = torch.ops.c10d_functional.wait_tensor(
        torch.ops._c10d_functional.reduce_scatter_tensor.default(
            new_rs_in, reduce_op, group_size, group_name
        )
    )
    new_out_flat = new_rs_out.split(new_out_numels, 0)
    new_outs = [x.reshape(s) for x, s in zip(new_out_flat, new_out_sizes)]
    return new_outs


def reduce_scatter_merge_fn_coalesced(
    rs_ins: list[torch.Tensor],
    group_name: Any,
    group_size: int,
    reduce_op: str,
    reduce_dtype: torch.dtype,
    device: torch.device,
) -> list[torch.Tensor]:
    """Bucketed RS via NCCL's coalesced API (ncclGroupStart/End).

    Avoids cat-ing inputs into one buffer; instead passes the tensor list
    directly to reduce_scatter_tensor_coalesced for zero-copy batching.
    """
    rs_ins_flat = [x.reshape(-1) for x in rs_ins]
    new_out_sizes = [(x.shape[0] // group_size,) + x.shape[1:] for x in rs_ins]

    rs_outs = torch.ops._c10d_functional.reduce_scatter_tensor_coalesced(
        rs_ins_flat, reduce_op, group_size, group_name
    )
    rs_outs = [torch.ops.c10d_functional.wait_tensor(o) for o in rs_outs]
    return [o.reshape(s) for o, s in zip(rs_outs, new_out_sizes)]


def all_reduce_merge_fn_to_trace(
    ar_ins: list[torch.Tensor],
    group_name: Any,
    reduce_op: str,
    reduce_dtype: torch.dtype,  # type: ignore[name-defined]
    device: torch.device,  # type: ignore[name-defined]
) -> list[torch.Tensor]:  # type: ignore[no-untyped-def]
    ar_ins_flattened = [x.reshape(-1) for x in ar_ins]
    new_ar_in = torch.cat(ar_ins_flattened)
    new_ar_out = torch.ops.c10d_functional.wait_tensor(
        torch.ops._c10d_functional.all_reduce.default(new_ar_in, reduce_op, group_name)
    )
    split_sizes = [x.numel() for x in ar_ins]
    new_outs_flat = new_ar_out.split(split_sizes)
    new_outs = [x.reshape(ar_in.shape) for x, ar_in in zip(new_outs_flat, ar_ins)]
    return new_outs


# List of all torch dtypes for serialization through custom ops
# TODO: custom ops support list[dtype] input
_ALL_DTYPES = tuple(
    [
        getattr(torch, attr)
        for attr in dir(torch)
        if isinstance(getattr(torch, attr), torch.dtype)
    ]
)


@torch.library.custom_op("bucketing::_pre_bucket_all_gather", mutates_args={})
def _pre_bucket_all_gather(
    ag_ins: list[torch.Tensor],
    group_size: int,
    dtype: torch.dtype,  # type: ignore[name-defined]
    out_dtype_ints: list[
        int
    ],  # dtype enum values, that inputs are converted to before all_gather
    rank: int,
    foreach_group_indices: list[int] | None = None,
) -> torch.Tensor:
    """
    Pre-bucket all gather operation.

    Args:
        ag_ins: Input tensors to gather
        group_size: Size of the process group
        dtype: Target dtype for the bucket
        out_dtype_ints: Dtype enum values for each input
        rank: Current rank
        foreach_group_indices: Optional flat list of grouped indices with -1 as delimiter.
            E.g., [0, 2, -1, 1] means groups [[0, 2], [1]].
    """
    # Convert int indices back to torch.dtype
    out_dtypes = [_ALL_DTYPES[d] for d in out_dtype_ints]
    ins_split_sizes_bytes = [
        ag_in.numel() * out_dtype.itemsize
        for ag_in, out_dtype in zip(ag_ins, out_dtypes, strict=True)
    ]
    bucket_dtype_size_bytes = dtype.itemsize
    ins_split_sizes = [
        _bytes // bucket_dtype_size_bytes for _bytes in ins_split_sizes_bytes
    ]
    ag_input_numel = sum(ins_split_sizes)
    device = ag_ins[0].device
    new_ag_out = torch.empty(ag_input_numel * group_size, dtype=dtype, device=device)
    new_ag_in = new_ag_out.narrow(0, ag_input_numel * rank, ag_input_numel)
    foreach_copy_dsts = torch.split(new_ag_in, ins_split_sizes)
    # View each destination slice as its output dtype, then copy
    # The copy operation handles dtype conversion from input dtype to output dtype
    foreach_copy_dsts_typed = [
        dst.view(out_dtype)
        for dst, out_dtype in zip(foreach_copy_dsts, out_dtypes, strict=True)
    ]
    ag_ins_flattened = [ag_in.reshape(-1) for ag_in in ag_ins]

    # Parse pre-computed groups from flat list with -1 delimiters
    if foreach_group_indices is not None:
        groups_list: list[list[int]] = []
        current_group: list[int] = []
        for idx in foreach_group_indices:
            if idx == -1:
                if current_group:
                    groups_list.append(current_group)
                    current_group = []
            else:
                current_group.append(idx)
        # Add last group if not empty
        if current_group:
            groups_list.append(current_group)

        # Call foreach_copy_ per group
        for group_indices in groups_list:
            group_dsts = [foreach_copy_dsts_typed[idx] for idx in group_indices]
            group_srcs = [ag_ins_flattened[idx] for idx in group_indices]
            torch._foreach_copy_(group_dsts, group_srcs)
    else:
        # No grouping provided - single foreach_copy_ call
        torch._foreach_copy_(foreach_copy_dsts_typed, ag_ins_flattened)
    return new_ag_out


def _pre_bucket_all_gather_fake(
    ag_ins: list[torch.Tensor],
    group_size: int,
    dtype: torch.dtype,  # type: ignore[name-defined]
    out_dtype_ints: list[int],
    rank: int,
    foreach_group_indices: list[int] | None = None,
) -> torch.Tensor:
    out_dtypes = [_ALL_DTYPES[d] for d in out_dtype_ints]
    ins_split_sizes_bytes = [
        ag_in.numel() * out_dtype.itemsize
        for ag_in, out_dtype in zip(ag_ins, out_dtypes, strict=True)
    ]
    bucket_dtype_size_bytes = dtype.itemsize
    ins_split_sizes = [
        _bytes // bucket_dtype_size_bytes for _bytes in ins_split_sizes_bytes
    ]
    ag_input_numel = sum(ins_split_sizes)
    device = ag_ins[0].device
    new_ag_out = torch.empty(ag_input_numel * group_size, dtype=dtype, device=device)
    return new_ag_out


_pre_bucket_all_gather.register_fake(_pre_bucket_all_gather_fake)


def all_gather_merge_fn_to_trace_custom_ops(
    _ag_ins: list[torch.Tensor],
    group_name: Any,
    group_size: int,
    dtype: torch.dtype,  # type: ignore[name-defined]
    out_dtypes: list[torch.dtype],  # type: ignore[name-defined]
    rank: int,
) -> list[torch.Tensor]:
    # Don't create convert_element_type ops - _pre_bucket_all_gather handles conversion
    # by viewing destination slices as output dtypes and letting copy do the conversion
    ag_ins = _ag_ins
    ins_sizes = [ag_in.shape for ag_in in ag_ins]
    ins_split_sizes_bytes = [
        ag_in.numel() * out_dtype.itemsize
        for ag_in, out_dtype in zip(ag_ins, out_dtypes)
    ]
    bucket_dtype_size_bytes = dtype.itemsize
    ins_split_sizes = [
        _bytes // bucket_dtype_size_bytes for _bytes in ins_split_sizes_bytes
    ]
    ag_input_numel = sum(ins_split_sizes)

    # Convert out_dtypes to indices for custom_op
    # TODO: custom ops support list[dtype] input
    out_dtype_ints = [_ALL_DTYPES.index(dt) for dt in out_dtypes]

    # Pre-compute foreach groups for better foreach_copy_ performance
    foreach_group_indices = _compute_foreach_groups(ag_ins, out_dtypes)

    new_ag_out = torch.ops.bucketing._pre_bucket_all_gather(
        ag_ins,
        group_size,
        dtype,
        out_dtype_ints,
        rank,
        foreach_group_indices,
    )
    new_ag_in = new_ag_out.narrow(0, ag_input_numel * rank, ag_input_numel)
    wait_tensor = torch.ops.c10d_functional.wait_tensor(
        torch.ops._c10d_functional.all_gather_into_tensor_out.default(
            new_ag_in, group_size, group_name, out=new_ag_out
        )
    )
    new_ag_out_reshaped = wait_tensor.reshape(group_size, -1)
    outs_bucket_dtype = torch.split_with_sizes(
        new_ag_out_reshaped,
        ins_split_sizes,
        dim=1,
    )
    outs_reshaped = [
        o.view(out_dtype).reshape((shape[0] * group_size,) + shape[1:])
        for o, shape, out_dtype in zip(outs_bucket_dtype, ins_sizes, out_dtypes)
    ]
    return outs_reshaped


def all_gather_merge_fn_to_trace(
    ag_ins: list[torch.Tensor],
    group_name: Any,
    group_size: int,
    dtype: torch.dtype,  # type: ignore[name-defined]
    out_dtypes: list[torch.dtype],  # type: ignore[name-defined]
    rank: int,
) -> list[torch.Tensor]:
    ins_sizes = [ag_in.shape for ag_in in ag_ins]
    ins_split_sizes = [ag_in.numel() for ag_in in ag_ins]
    ag_input_numel = sum(ins_split_sizes)
    device = ag_ins[0].device
    new_ag_out = torch.empty(ag_input_numel * group_size, dtype=dtype, device=device)
    new_ag_in = new_ag_out.narrow(0, ag_input_numel * rank, ag_input_numel)
    ag_ins_flattened = [ag_in.reshape(-1) for ag_in in ag_ins]
    # Inductor fuses copy_(cat(...)) into 1 Triton kernel with no allocation for cat.
    # _foreach_copy_(..., ag_ins_flattened) emits separate kernel per item,
    # resulting in large number of small triton kernels to launch.
    new_ag_in.copy_(torch.cat(ag_ins_flattened))
    wait_tensor = torch.ops.c10d_functional.wait_tensor(
        torch.ops._c10d_functional.all_gather_into_tensor_out.default(
            new_ag_in, group_size, group_name, out=new_ag_out
        )
    )
    new_ag_out_reshaped = wait_tensor.reshape(group_size, -1)
    outs = torch.split_with_sizes(
        new_ag_out_reshaped,
        ins_split_sizes,
        dim=1,
    )
    outs_reshaped = [
        o.reshape((shape[0] * group_size,) + shape[1:])
        for o, shape in zip(outs, ins_sizes)
    ]
    return outs_reshaped


def all_gather_merge_fn_to_trace_functional(
    ag_ins: list[torch.Tensor],
    group_name: Any,
    group_size: int,
    dtype: torch.dtype,  # type: ignore[name-defined]
    out_dtypes: list[torch.dtype],  # type: ignore[name-defined]
    rank: int,
    use_fsdp_ag_copy_in: bool = False,
) -> list[torch.Tensor]:
    # Implementation that is functional in graph,
    # but uses custom op torch.ops.fsdp.all_gather_copy_in.
    ins_sizes = [ag_in.shape for ag_in in ag_ins]
    ins_split_sizes = [ag_in.numel() for ag_in in ag_ins]
    ag_input_numel = sum(ins_split_sizes)
    device = ag_ins[0].device
    new_ag_out = torch.empty(ag_input_numel * group_size, dtype=dtype, device=device)
    ag_ins_flattened = [ag_in.reshape(-1) for ag_in in ag_ins]
    if use_fsdp_ag_copy_in:
        new_ag_in, new_ag_out = torch.ops.fsdp.all_gather_copy_in(
            ag_ins_flattened, new_ag_out, ins_split_sizes, ag_input_numel, rank
        )
    else:
        new_ag_in = torch.cat(ag_ins_flattened, dim=0)
    wait_tensor = torch.ops.c10d_functional.wait_tensor(
        torch.ops._c10d_functional.all_gather_into_tensor_out.default(
            new_ag_in, group_size, group_name, out=new_ag_out
        )
    )
    new_ag_out_reshaped = wait_tensor.reshape(group_size, -1)
    outs = torch.split_with_sizes(
        new_ag_out_reshaped,
        ins_split_sizes,
        dim=1,
    )
    outs_reshaped = [
        o.reshape((shape[0] * group_size,) + shape[1:])
        for o, shape in zip(outs, ins_sizes)
    ]
    return outs_reshaped


def _trace(fn, inps) -> torch.fx.GraphModule:  # type: ignore[no-untyped-def]
    with dynamo_timed("fx.bucketing._trace", log_pt2_compile_event=True):
        fake_mode = detect_fake_mode(inps)
        if fake_mode is None:
            raise AssertionError("expected a fake mode to be detected, got None")
        shape_env = fake_mode.shape_env
        pending_unbacked = None
        ignorable_unbacked = None
        if shape_env is not None:
            pending_unbacked = list(shape_env.pending_fresh_unbacked_symbols)
            ignorable_unbacked = list(shape_env.ignorable_fresh_unbacked_symbols)
            shape_env.pending_fresh_unbacked_symbols.clear()
            shape_env.ignorable_fresh_unbacked_symbols.clear()
        try:
            with fake_mode, enable_python_dispatcher():
                out = make_fx(fn)(*inps)
        finally:
            if shape_env is not None:
                if pending_unbacked is None:
                    raise AssertionError("expected pending_unbacked to be set")
                if ignorable_unbacked is None:
                    raise AssertionError("expected ignorable_unbacked to be set")
                shape_env.pending_fresh_unbacked_symbols[:] = pending_unbacked
                shape_env.ignorable_fresh_unbacked_symbols[:] = ignorable_unbacked
        for node in out.graph.find_nodes(
            op="call_function", target=torch.ops.aten.detach.default
        ):
            node.replace_all_uses_with(node.args[0])
            out.graph.erase_node(node)
        return out


def _insert_fn_trace_before_node(  # type: ignore[no-untyped-def]
    g: torch.fx.Graph,
    fn_to_trace,
    inps,
    insert_before_node: torch.fx.Node,
    g_fn_inps: list[torch.fx.Node],
    g_fn_outs: list[torch.fx.Node],
) -> tuple[dict[torch.fx.Node, torch.fx.Node], list[torch.fx.Node]]:  # type: ignore[no-untyped-def]
    """
    Helper function that traces :attr:`fn_to_trace` with inputs
    :attr:`inps`.
    The result function graph will be inserted before :attr:`insert_before_node`,
    using :attr:`g_fn_inps` nodes of original graph as inputs of function graph,
    function graph outputs will replace :attr:`g_fn_outs` in original graph.

    Returns:
        (replacements, new_nodes): Dictionary mapping old to new nodes, and list of all newly inserted nodes
    """
    with dynamo_timed(
        "fx.bucketing._insert_fn_trace_before_node", log_pt2_compile_event=True
    ):
        fn_gm = _trace(
            fn_to_trace,
            inps,
        )
        fn_g = fn_gm.graph
        fn_g_ins = fn_g.find_nodes(op="placeholder")
        env = {fn_g_ins[idx]: g_fn_inps[idx] for idx in range(len(g_fn_inps))}
        g_fn_new_outs: list[torch.fx.Node] = []
        new_nodes: list[torch.fx.Node] = []  # Track all newly inserted nodes

        with g.inserting_before(insert_before_node):
            for _n in fn_g.nodes:
                if _n.op == "placeholder":
                    continue
                _new_n = g.node_copy(_n, lambda x: env[x])
                env[_n] = _new_n
                if _n.op == "output":
                    g_fn_new_outs = _new_n.args[0]  # type: ignore[assignment]
                    g.erase_node(_new_n)
                else:
                    new_nodes.append(_new_n)  # Track non-output nodes

        replacements = {  # noqa: C416
            orig_out: new_out for orig_out, new_out in zip(g_fn_outs, g_fn_new_outs)
        }
        replaced_users = []
        for orig_out, new_out in zip(g_fn_outs, g_fn_new_outs):
            replaced_users.extend(orig_out.users)
            orig_out.replace_all_uses_with(new_out)
        _recompute_changed_user_metadata(replaced_users)

        return replacements, new_nodes


def has_mergeable_all_gather_convert_dtype(n: torch.fx.Node) -> bool:
    node_in = n.args[0]
    return (
        is_all_gather_into_tensor(n)
        and isinstance(node_in, torch.fx.Node)
        and node_in.op == "call_function"
        and (
            node_in.target is torch.ops.prims.convert_element_type.default
            or node_in.target is torch.ops.aten._to_copy.default
        )
        and len(node_in.users) == 1
    )


def _sort_bucket_region(
    g: torch.fx.Graph,
    new_nodes: list[torch.fx.Node],
    new_nodes_inputs: list[torch.fx.Node],
) -> None:
    """Topologically sort the smallest region spanning *new_nodes* and their inputs.

    After bucketing inserts *new_nodes*, some *new_nodes_inputs* may sit after
    the new collective in the linked list.  This sorts just the affected
    region so every input precedes its consumer.

    Complexity: O(D) where D = distance between the farthest input and
    new_nodes in the linked list.  No full-graph enumeration.
    """
    if not new_nodes or not new_nodes_inputs:
        return

    new_set: OrderedSet[torch.fx.Node] = OrderedSet(new_nodes)
    external: OrderedSet[torch.fx.Node] = OrderedSet(
        [inp for inp in new_nodes_inputs if inp not in new_set]
    )
    if not external:
        return

    # Walk backward/forward from the new_nodes span to find all external
    # inputs.  ``remaining`` counts how many we still need to locate;
    # each walk stops as soon as its share is found.
    first = new_nodes[0]
    last = new_nodes[-1]
    remaining = len(external)

    cursor: torch.fx.Node | None = first.prev
    while cursor is not None and cursor.op != "placeholder" and remaining > 0:
        if cursor in external:
            first = cursor
            remaining -= 1
        cursor = cursor.prev

    cursor = last.next
    while cursor is not None and cursor.op != "output" and remaining > 0:
        if cursor in external:
            last = cursor
            remaining -= 1
        cursor = cursor.next

    if first is last:
        return

    region: OrderedSet[torch.fx.Node] = OrderedSet()
    cursor = first
    while cursor is not None:
        region.add(cursor)
        if cursor is last:
            break
        cursor = cursor.next

    from torch._dynamo.graph_deduplication import _stable_topological_sort_region

    _stable_topological_sort_region(g, region)


def process_collective_bucket(
    g: torch.fx.Graph,
    bucket_nodes: list[torch.fx.Node],
    fn_to_trace: Callable[..., list[torch.Tensor]],
    trace_args_fn: Callable[[list[torch.fx.Node]], tuple[Any, ...]],
    insert_before: torch.fx.Node | None = None,
    wait_insertion_point: torch.fx.Node | None = None,
    extra_graph_inps: list[torch.fx.Node] | None = None,
) -> tuple[list[torch.fx.Node], dict[torch.fx.Node, torch.fx.Node]]:
    """
    Process a single bucket of collective operation nodes with flexible insertion control.

    Args:
        g: The graph to modify
        bucket_nodes: Nodes in the current bucket to process
        fn_to_trace: Function to trace and insert
        trace_args_fn: Function to create trace arguments from inputs
        insert_before: Where to insert the traced function (default: after last bucket node)
        wait_insertion_point: If provided, move all nodes from wait() onwards to before this node
        extra_graph_inps: Additional non-tensor graph nodes to wire as traced
            inputs (appended after tensor inputs). Used for compile-on-one-rank
            graphs where group_name is a Node reference that make_fx proxies
            as an opaque input.

    Returns:
        new_nodes: List of all newly inserted nodes
        replacements: Dictionary mapping old wait nodes to new output nodes
    """
    # Collect inputs and waits from current bucket
    bucket_ins: list[torch.fx.Node] = []
    bucket_waits: list[torch.fx.Node] = []
    ag_node_to_pre_nodes: dict[torch.fx.Node, list[torch.fx.Node]] = defaultdict(list)

    for n in bucket_nodes:
        if len(n.users) != 1:
            raise AssertionError(f"Expected single user for {n}, got {n.users}")
        wait_n = next(iter(n.users))

        # Handle convert_element_type operations (for all_gather)
        node_in = n.args[0]
        if has_mergeable_all_gather_convert_dtype(n):
            # pyrefly: ignore [bad-argument-type]
            ag_node_to_pre_nodes[n].append(node_in)
            # pyrefly: ignore [missing-attribute]
            node_in = node_in.args[0]

        if not isinstance(node_in, torch.fx.Node):  # Ensure node_in is a Node
            raise AssertionError(f"expected node_in to be a Node, got {type(node_in)}")
        bucket_ins.append(node_in)
        bucket_waits.append(wait_n)

    # Create trace arguments
    trace_args = trace_args_fn(bucket_ins)

    # Determine insertion point
    if insert_before is None:
        insert_before = bucket_nodes[-1].next

    g_fn_inps = bucket_ins + (extra_graph_inps or [])

    # Insert traced function and get replacements + new nodes
    replacements, new_nodes = _insert_fn_trace_before_node(
        g,
        fn_to_trace,
        trace_args,
        insert_before,
        g_fn_inps,
        bucket_waits,
    )

    # If requested, move wait nodes and everything after to specified location
    if wait_insertion_point is not None:
        # Find the first wait node in new_nodes
        wait_start_idx = None
        for i, node in enumerate(new_nodes):
            if is_wait_tensor(node):
                wait_start_idx = i
                break

        # Move all nodes from wait onwards (including the wait)
        if wait_start_idx is not None:
            nodes_to_move = new_nodes[wait_start_idx:]
            for node in nodes_to_move:
                wait_insertion_point.prepend(node)

    # Preserve metadata from original collective nodes to new bucketed nodes
    if bucket_nodes:
        overlap_log.debug(
            "Bucketing nodes: %s, New nodes: %s",
            ",".join([n.name for n in bucket_nodes]),
            ",".join([n.name for n in new_nodes]),
        )
    _populate_node_meta(bucket_nodes, new_nodes)

    # Erase old nodes
    for node, wait_n in zip(bucket_nodes, bucket_waits):
        g.erase_node(wait_n)
        g.erase_node(node)
        # Erase any convert_element_type nodes we tracked
        for pre_node in reversed(ag_node_to_pre_nodes[node]):
            g.erase_node(pre_node)

    _sort_bucket_region(g, new_nodes, g_fn_inps)

    return new_nodes, replacements


def merge_reduce_scatter_bucket(
    g: torch.fx.Graph,
    rs_nodes: list[torch.fx.Node],
    mode: BucketMode | None = None,
    insert_before: torch.fx.Node | None = None,
    wait_insertion_point: torch.fx.Node | None = None,
) -> tuple[list[torch.fx.Node], dict[torch.fx.Node, torch.fx.Node]]:
    mode = mode or _default_bucket_mode()
    # Validate bucket consistency
    rs0 = rs_nodes[0]
    rs0_val = rs0.meta["val"]
    _, reduce_op, group_size, group_name = rs0.args
    group_name_str = _resolve_group_name(group_name)
    reduce_dtype = rs0_val.dtype
    device = rs0_val.device

    for n in rs_nodes:
        rs_val = n.meta["val"]
        if not (
            n.args[1] == reduce_op
            and n.args[2] == group_size
            and _resolve_group_name(n.args[3]) == group_name_str
            and rs_val.device == device
            and rs_val.dtype == reduce_dtype
        ):
            raise AssertionError(
                f"reduce_scatter node {n} does not match bucket parameters"
            )

    # Choose merge function based on mode
    rs_merge_fn = reduce_scatter_merge_fn_to_trace
    if mode == "coalesced":
        rs_merge_fn = reduce_scatter_merge_fn_coalesced
    elif mode and "custom_ops" in mode:
        rs_merge_fn = reduce_scatter_merge_fn_to_trace_custom_ops

    group_name_val = (
        group_name.meta["val"] if isinstance(group_name, torch.fx.Node) else group_name
    )

    def create_trace_args(bucket_ins: list[torch.fx.Node]) -> tuple[Any, ...]:
        return (
            pytree.tree_map(lambda node: node.meta["val"], bucket_ins),
            group_name_val,
            group_size,
            reduce_op,
            reduce_dtype,
            device,
        )

    return process_collective_bucket(
        g,
        rs_nodes,
        rs_merge_fn,
        create_trace_args,
        insert_before=insert_before,
        wait_insertion_point=wait_insertion_point,
        extra_graph_inps=(
            [group_name] if isinstance(group_name, torch.fx.Node) else None
        ),
    )


def merge_all_reduce_bucket(
    g: torch.fx.Graph,
    ar_nodes: list[torch.fx.Node],
    mode: str | None = None,
    insert_before: torch.fx.Node | None = None,
    wait_insertion_point: torch.fx.Node | None = None,
) -> tuple[list[torch.fx.Node], dict[torch.fx.Node, torch.fx.Node]]:
    ar0 = ar_nodes[0]
    ar0_val = ar0.meta["val"]
    _, reduce_op, group_name = ar0.args
    group_name_str = _resolve_group_name(group_name)
    reduce_dtype = ar0_val.dtype
    device = ar0_val.device

    for n in ar_nodes:
        ar_val = n.meta["val"]
        if not (
            n.args[1] == reduce_op
            and _resolve_group_name(n.args[2]) == group_name_str
            and ar_val.device == device
            and ar_val.dtype == reduce_dtype
        ):
            raise AssertionError(
                f"all_reduce node {n} does not match bucket parameters"
            )

    ar_merge_fn = all_reduce_merge_fn_to_trace

    group_name_val = (
        group_name.meta["val"] if isinstance(group_name, torch.fx.Node) else group_name
    )

    def create_trace_args(bucket_ins: list[torch.fx.Node]) -> tuple[Any, ...]:
        return (
            pytree.tree_map(lambda node: node.meta["val"], bucket_ins),
            group_name_val,
            reduce_op,
            reduce_dtype,
            device,
        )

    return process_collective_bucket(
        g,
        ar_nodes,
        ar_merge_fn,
        create_trace_args,
        insert_before=insert_before,
        wait_insertion_point=wait_insertion_point,
        extra_graph_inps=(
            [group_name] if isinstance(group_name, torch.fx.Node) else None
        ),
    )


def merge_all_gather_bucket(
    g: torch.fx.Graph,
    ag_nodes: list[torch.fx.Node],
    mode: BucketMode | None = None,
    insert_before: torch.fx.Node | None = None,
    wait_insertion_point: torch.fx.Node | None = None,
) -> tuple[list[torch.fx.Node], dict[torch.fx.Node, torch.fx.Node]]:
    mode = mode or _default_bucket_mode()
    from torch.distributed.distributed_c10d import _resolve_process_group

    ag0 = ag_nodes[0]
    _, group_size, group_name = ag0.args
    group_name_str = _resolve_group_name(group_name)
    _ag_dtypes: list[torch.dtype] = []  # type: ignore[name-defined]

    for n in ag_nodes:
        if not (
            n.args[1] == group_size and _resolve_group_name(n.args[2]) == group_name_str
        ):
            raise AssertionError(
                f"all_gather node {n} does not match bucket parameters"
            )
        _ag_dtypes.append(n.meta["val"].dtype)

    bucket_dtype = pick_bucket_dtype(_ag_dtypes)

    # Choose merge function based on mode
    ag_merge_fn = all_gather_merge_fn_to_trace
    if mode == "coalesced":
        logger.info("coalesced bucket_mode not supported for all_gather, using default")
    elif mode and "custom_ops" in mode:
        ag_merge_fn = all_gather_merge_fn_to_trace_custom_ops  # type: ignore[assignment]

    # pyrefly: ignore [bad-argument-type]
    rank: int = dist.get_rank(_resolve_process_group(group_name_str))

    group_name_val = (
        group_name.meta["val"] if isinstance(group_name, torch.fx.Node) else group_name
    )

    def create_trace_args(bucket_ins: list[torch.fx.Node]) -> tuple[Any, ...]:
        return (
            pytree.tree_map(lambda node: node.meta["val"], bucket_ins),
            group_name_val,
            group_size,
            bucket_dtype,
            _ag_dtypes,
            rank,
        )

    return process_collective_bucket(
        g,
        ag_nodes,
        ag_merge_fn,
        create_trace_args,
        insert_before=insert_before,
        wait_insertion_point=wait_insertion_point,
        extra_graph_inps=(
            [group_name] if isinstance(group_name, torch.fx.Node) else None
        ),
    )


def merge_reduce_scatter(
    gm: torch.fx.GraphModule,
    rs_buckets: list[list[torch.fx.Node]],
    mode: BucketMode | None = None,
) -> None:
    """
    Merges specified buckets of reduce_scatter to joint reduce_scatter.
    """
    mode = mode or _default_bucket_mode()
    with dynamo_timed("fx.bucketing.merge_reduce_scatter", log_pt2_compile_event=True):
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "fx_bucketing_passes_reduce_scatter_buckets",
                "encoding": "string",
            },
            payload_fn=lambda: str(rs_buckets),
        )

        g = gm.graph

        for rs_nodes in rs_buckets:
            merge_reduce_scatter_bucket(g, rs_nodes, mode)


def merge_all_gather(
    gm: torch.fx.GraphModule,
    ag_buckets: list[list[torch.fx.Node]],
    mode: BucketMode | None = None,
) -> None:
    """
    Merges specified buckets of all_gather to joint all_gather.
    """
    mode = mode or _default_bucket_mode()
    with dynamo_timed("fx.bucketing.merge_all_gather", log_pt2_compile_event=True):
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "fx_bucketing_passes_all_gather_buckets",
                "encoding": "string",
            },
            payload_fn=lambda: str(ag_buckets),
        )

        g = gm.graph

        for ag_nodes in ag_buckets:
            merge_all_gather_bucket(g, ag_nodes, mode)
<|endoftext|># mypy: allow-untyped-defs
import logging
import operator
from collections import defaultdict
from dataclasses import dataclass, field
from math import prod
from typing import Any, cast

import torch
from torch.utils._ordered_set import OrderedSet

from .. import config, inductor_prims
from ..pattern_matcher import (
    CallFunction,
    Ignored,
    KeywordArg,
    ListOf,
    Match,
    MULTIPLE,
    PatternExpr,
    PatternMatcherPass,
)


log = logging.getLogger(__name__)
aten = torch.ops.aten
patterns = PatternMatcherPass()


def _is_last_dim(t: torch.Tensor, dim: int) -> bool:
    return dim == t.ndim - 1 or dim == -1


def _is_backward(graph: torch.fx.Graph) -> bool:
    placeholders = []
    for node in graph.nodes:
        if node.op != "placeholder":
            break
        placeholders.append(node)
    return not all(node.name.startswith("primal") for node in placeholders)


def _compute_mm_arithmetic_intensity(M: int, N: int, K: int) -> float:
    return M * N * K / (M * K + N * K + M * N)


def _filter_nodes_by_target(nodes: list[torch.fx.Node], target) -> list[torch.fx.Node]:
    return [x for x in nodes if x.target == target]


def _find_ancestors(node: torch.fx.Node) -> OrderedSet[torch.fx.Node]:
    ancestors = OrderedSet[torch.fx.Node]()
    ancestors.add(node)
    cur_nodes = [node]
    while len(cur_nodes) > 0:
        new_nodes = []
        for node in cur_nodes:
            for inp in node.all_input_nodes:
                if inp not in ancestors:
                    ancestors.add(inp)
                    new_nodes.append(inp)
        cur_nodes = new_nodes
    return OrderedSet(node for node in ancestors if node.op != "placeholder")


def _get_tensor(node: torch.fx.Node) -> torch.Tensor:
    val = node.meta["val"]
    if not isinstance(val, torch.Tensor):
        raise AssertionError(f"expected node val to be a Tensor, got {type(val)}")
    return val


@dataclass
class _AllGatherMatch:
    match: Match | list[torch.fx.Node]
    shard_node: torch.fx.Node
    ag_node: torch.fx.Node
    res_node: torch.fx.Node
    gather_dim: int
    group_name: "torch.distributed.distributed_c10d.GroupName"

    def replace_with(self, new_node: torch.fx.Node) -> None:
        self.res_node.replace_all_uses_with(new_node)

    def erase(self) -> None:
        nodes = self.match.nodes if isinstance(self.match, Match) else self.match
        for node in reversed(nodes):
            if len(node.users) == 0:
                node.graph.erase_node(node)


def find_all_gather_patterns(graph: torch.fx.Graph):
    """Find all-gather outputs that can fuse with downstream matmul users.

    This recognizes both the direct all-gather form and slice/cat reassembly
    forms that preserve the same tensor-parallel communication contract.
    """
    c10d = torch.ops._c10d_functional
    split_targets = [aten.split.Tensor, aten.split.sizes, aten.split_with_sizes.default]

    def make_zero_dim_all_gather_pattern(shard):
        return CallFunction(
            c10d.wait_tensor.default,
            CallFunction(
                c10d.all_gather_into_tensor.default,
                shard,
                Ignored(),
                KeywordArg("group_name"),
            ),
        )

    # Matches funcol.all_gather_single with gather_dim == 0
    zero_dim_all_gather_pattern = make_zero_dim_all_gather_pattern(KeywordArg("shard"))

    def make_all_gather_split_pattern(shard):
        return CallFunction(
            operator.getitem,
            CallFunction(
                split_targets,
                make_zero_dim_all_gather_pattern(shard),
                Ignored(),
                _users=MULTIPLE,
            ),
            Ignored(),
        )

    def make_cat_pattern(splits):
        return CallFunction(
            aten.cat.default,
            ListOf(splits),
            KeywordArg("gather_dim"),
        )

    # Matches funcol.all_gather_single with gather_dim > 0
    non_zero_dim_all_gather_pattern = make_cat_pattern(
        make_all_gather_split_pattern(KeywordArg("shard")),
    )

    # Match a zero-dim all-gather in which the data is transferred as uint8 and
    # viewed back as the original dtype.
    zero_dim_type_erased_all_gather_pattern = CallFunction(
        aten.view.dtype,
        make_zero_dim_all_gather_pattern(
            KeywordArg("shard"),
        ),
        Ignored(),
    )

    # Match a non-zero dim all-gather in which the data is transferred as uint8
    # and viewed back as the original dtype.
    non_zero_dim_type_erased_all_gather_pattern = CallFunction(
        aten.view.dtype,
        make_cat_pattern(
            CallFunction(
                aten.view.dtype,
                make_all_gather_split_pattern(
                    KeywordArg("shard"),
                ),
                Ignored(),
            ),
        ),
        Ignored(),
    )

    # If two patterns with the same res_node_target have the same suffix, the
    # longer pattern should appear first in the list.
    # e.g. supposed we have (1) A -> B -> C -> D and (2) B -> C -> D, (1)
    # should appear before (2) in the list.
    res_node_target_to_patterns = {
        aten.cat.default: [
            (non_zero_dim_all_gather_pattern, 0),
        ],
        aten.view.dtype: [
            (non_zero_dim_type_erased_all_gather_pattern, 0),
            (zero_dim_type_erased_all_gather_pattern, 0),
        ],
        c10d.wait_tensor.default: [
            (zero_dim_all_gather_pattern, 0),
        ],
    }

    # Match in reverse to ensure longer patterns is prioritized
    all_gathers = []
    visited_ag_nodes = OrderedSet[torch.fx.Node]()

    def _match_slice_cat_all_gather(
        node: torch.fx.Node,
    ) -> _AllGatherMatch | None:
        res_node = node
        trim_node = None
        if node.target is aten.slice.Tensor:
            if len(node.args) < 4 or not isinstance(node.args[0], torch.fx.Node):
                return None
            trim_node = node
            node = node.args[0]

        if node.target is not aten.cat.default:
            return None
        cat_inputs = node.args[0]
        if not isinstance(cat_inputs, (list, tuple)) or len(cat_inputs) == 0:
            return None
        gather_dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        if not isinstance(gather_dim, int):
            return None
        if trim_node is not None and (
            len(trim_node.args) < 4
            or trim_node.args[1] != gather_dim
            or trim_node.args[2] != 0
        ):
            return None

        slice_nodes: list[torch.fx.Node] = []
        wait_node = None
        for inp in cat_inputs:
            if not (
                isinstance(inp, torch.fx.Node)
                and inp.target is aten.slice.Tensor
                and len(inp.args) >= 2
                and inp.args[1] == 0
            ):
                return None
            cur_wait = inp.args[0]
            if not (
                isinstance(cur_wait, torch.fx.Node)
                and cur_wait.target is c10d.wait_tensor.default
            ):
                return None
            if wait_node is None:
                wait_node = cur_wait
            elif wait_node is not cur_wait:
                return None
            slice_nodes.append(inp)

        if wait_node is None:
            return None
        ag_node = wait_node.args[0]
        if not (
            isinstance(ag_node, torch.fx.Node)
            and ag_node.target is c10d.all_gather_into_tensor.default
        ):
            return None
        group_name = ag_node.kwargs.get("group_name")
        if group_name is None and len(ag_node.args) >= 3:
            group_name = ag_node.args[2]
        if group_name is None:
            return None

        return _AllGatherMatch(
            match=[
                ag_node,
                wait_node,
                *slice_nodes,
                node,
                *((trim_node,) if trim_node is not None else ()),
            ],
            shard_node=cast("torch.fx.Node", ag_node.args[0]),
            ag_node=ag_node,
            res_node=res_node,
            gather_dim=gather_dim,
            group_name=cast("torch.distributed.distributed_c10d.GroupName", group_name),
        )

    for node in reversed(graph.nodes):
        if ag_match := _match_slice_cat_all_gather(node):
            if ag_match.ag_node not in visited_ag_nodes:
                visited_ag_nodes.add(ag_match.ag_node)
                all_gathers.append(ag_match)
            continue
        for target, patterns in res_node_target_to_patterns.items():
            if node.target != target:
                continue
            for pattern, ag_node_idx in patterns:
                match = pattern.match(node)
                if not match:
                    continue

                if not isinstance(match, Match):
                    raise AssertionError(f"expected a Match, got {type(match)}")
                ag_node = match.nodes[ag_node_idx]
                if ag_node.target != c10d.all_gather_into_tensor.default:
                    raise AssertionError(
                        "expected ag_node target to be "
                        f"all_gather_into_tensor, got {ag_node.target}"
                    )

                if ag_node in visited_ag_nodes:
                    continue
                visited_ag_nodes.add(ag_node)

                ag_match = _AllGatherMatch(
                    match=match,
                    shard_node=match.kwargs["shard"],
                    ag_node=ag_node,
                    res_node=node,
                    gather_dim=match.kwargs.get("gather_dim", 0),
                    group_name=match.kwargs["group_name"],
                )
                all_gathers.append(ag_match)

    return list(reversed(all_gathers))


@dataclass
class _ReduceScatterMatch:
    match: Match | list[torch.fx.Node]
    input_node: torch.fx.Node
    reduce_scatter_node: torch.fx.Node
    wait_tensor_node: torch.fx.Node
    reduce_op: str
    scatter_dim: int
    group_name: "torch.distributed.distributed_c10d.GroupName"

    def replace_with(self, new_node: torch.fx.Node) -> None:
        # Replace all uses of the result node (wait_tensor) with the fused node.
        self.wait_tensor_node.replace_all_uses_with(new_node)

        # If the reduce-scatter result is saved for backward, save the fused node for backward instead.
        self._update_save_for_backward(new_node)

    def _update_save_for_backward(self, new_node: torch.fx.Node) -> None:
        """
        If the output node is a user of the reduce_scatter node (indicating the reduce_scatter
        result is saved for backward), this method will update the output node to use the fused node instead.
        """
        output_node = None
        for user in self.reduce_scatter_node.users:
            if user.target == "output":
                output_node = user
                break
        if output_node is not None:
            output_node.replace_input_with(self.reduce_scatter_node, new_node)

            # Assert that now the reduce scatter node has only one user (the wait_tensor) and it's not
            # saved for backward anymore.
            if len(self.reduce_scatter_node.users) != 1:
                raise AssertionError(
                    "Reduce scatter node has multiple users, this is not expected"
                )

    def erase(self) -> None:
        nodes = self.match.nodes if isinstance(self.match, Match) else self.match
        for node in reversed(nodes):
            if len(node.users) == 0:
                node.graph.erase_node(node)


def find_reduce_scatter_patterns(graph: torch.fx.Graph):
    """Find matmul outputs that can fuse with downstream reduce-scatter users.

    This recognizes both the direct reduce-scatter form and slice/cat forms
    that reassemble the same tensor-parallel communication payload.
    """
    c10d = torch.ops._c10d_functional
    split_targets = [aten.split.Tensor, aten.split.sizes, aten.split_with_sizes.default]

    def reduce_scatter_template(inp: PatternExpr, users: int):
        return CallFunction(
            c10d.wait_tensor.default,
            CallFunction(
                c10d.reduce_scatter_tensor.default,
                inp,
                KeywordArg("reduce_op"),
                Ignored(),
                KeywordArg("group_name"),
                _users=users,
            ),
        )

    # Matches funcol.reduce_scatter_single with scatter_dim == 0
    zero_dim_reduce_scatter_pattern_single_user = reduce_scatter_template(
        KeywordArg("input"), users=1
    )

    # Two users will occur when the reduce-scatter result is saved for backward
    zero_dim_reduce_scatter_pattern_multi_user = reduce_scatter_template(
        KeywordArg("input"), users=2
    )

    # Matches funcol.reduce_scatter_single with scatter_dim > 0
    non_zero_dim_reduce_scatter_pattern_single_user = reduce_scatter_template(
        CallFunction(
            aten.cat.default,
            ListOf(
                CallFunction(
                    operator.getitem,
                    CallFunction(
                        split_targets,
                        KeywordArg("input"),
                        Ignored(),
                        KeywordArg("scatter_dim"),
                        _users=MULTIPLE,
                    ),
                    Ignored(),
                )
            ),
        ),
        users=1,
    )

    # Two users will occur when the reduce-scatter result is saved for backward
    non_zero_dim_reduce_scatter_pattern_multi_user = reduce_scatter_template(
        CallFunction(
            aten.cat.default,
            ListOf(
                CallFunction(
                    operator.getitem,
                    CallFunction(
                        split_targets,
                        KeywordArg("input"),
                        Ignored(),
                        KeywordArg("scatter_dim"),
                        _users=MULTIPLE,
                    ),
                    Ignored(),
                )
            ),
        ),
        users=2,
    )

    reduce_scatters = []

    def _match_slice_cat_reduce_scatter(
        wait_node: torch.fx.Node,
    ) -> _ReduceScatterMatch | None:
        if wait_node.target is not c10d.wait_tensor.default:
            return None
        reduce_scatter_node = wait_node.args[0]
        if not (
            isinstance(reduce_scatter_node, torch.fx.Node)
            and reduce_scatter_node.target is c10d.reduce_scatter_tensor.default
        ):
            return None
        cat_node = reduce_scatter_node.args[0]
        if not (
            isinstance(cat_node, torch.fx.Node) and cat_node.target is aten.cat.default
        ):
            return None
        cat_dim = (
            cat_node.args[1]
            if len(cat_node.args) > 1
            else cat_node.kwargs.get("dim", 0)
        )
        if cat_dim != 0:
            return None
        cat_inputs = cat_node.args[0]
        if not isinstance(cat_inputs, (list, tuple)) or len(cat_inputs) == 0:
            return None

        slice_nodes: list[torch.fx.Node] = []
        input_node = None
        scatter_dim = None
        for inp in cat_inputs:
            if not (
                isinstance(inp, torch.fx.Node)
                and inp.target is aten.slice.Tensor
                and len(inp.args) >= 2
            ):
                return None
            cur_input = inp.args[0]
            cur_dim = inp.args[1]
            if not isinstance(cur_input, torch.fx.Node) or not isinstance(cur_dim, int):
                return None
            if input_node is None:
                input_node = cur_input
                scatter_dim = cur_dim
            elif input_node is not cur_input or scatter_dim != cur_dim:
                return None
            slice_nodes.append(inp)

        if input_node is None or scatter_dim is None:
            return None
        group_name = reduce_scatter_node.kwargs.get("group_name")
        if group_name is None and len(reduce_scatter_node.args) >= 4:
            group_name = reduce_scatter_node.args[3]
        reduce_op = reduce_scatter_node.kwargs.get("reduce_op")
        if reduce_op is None and len(reduce_scatter_node.args) >= 2:
            reduce_op = reduce_scatter_node.args[1]
        if group_name is None or reduce_op is None:
            return None

        return _ReduceScatterMatch(
            match=[*slice_nodes, cat_node, reduce_scatter_node, wait_node],
            input_node=input_node,
            reduce_scatter_node=reduce_scatter_node,
            wait_tensor_node=wait_node,
            reduce_op=cast(str, reduce_op),
            scatter_dim=scatter_dim,
            group_name=cast("torch.distributed.distributed_c10d.GroupName", group_name),
        )

    for node in reversed(graph.nodes):
        if node.target == c10d.wait_tensor.default:
            if rs_match := _match_slice_cat_reduce_scatter(node):
                reduce_scatters.append(rs_match)
            elif match := non_zero_dim_reduce_scatter_pattern_single_user.match(node):
                if not isinstance(match, Match):
                    raise AssertionError(f"expected a Match, got {type(match)}")
                reduce_scatters.append(
                    _ReduceScatterMatch(
                        match=match,
                        input_node=match.kwargs["input"],
                        reduce_scatter_node=match.nodes[-2],
                        wait_tensor_node=node,
                        reduce_op=match.kwargs["reduce_op"],
                        scatter_dim=match.kwargs["scatter_dim"],
                        group_name=match.kwargs["group_name"],
                    )
                )
            elif match := zero_dim_reduce_scatter_pattern_single_user.match(node):
                if not isinstance(match, Match):
                    raise AssertionError(f"expected a Match, got {type(match)}")
                reduce_scatters.append(
                    _ReduceScatterMatch(
                        match=match,
                        input_node=match.kwargs["input"],
                        reduce_scatter_node=match.nodes[0],
                        wait_tensor_node=node,
                        reduce_op=match.kwargs["reduce_op"],
                        scatter_dim=0,
                        group_name=match.kwargs["group_name"],
                    )
                )
            elif match := non_zero_dim_reduce_scatter_pattern_multi_user.match(node):
                if not isinstance(match, Match):
                    raise AssertionError(f"expected a Match, got {type(match)}")
                reduce_scatters.append(
                    _ReduceScatterMatch(
                        match=match,
                        input_node=match.kwargs["input"],
                        reduce_scatter_node=match.nodes[-2],
                        wait_tensor_node=node,
                        reduce_op=match.kwargs["reduce_op"],
                        scatter_dim=match.kwargs["scatter_dim"],
                        group_name=match.kwargs["group_name"],
                    )
                )
            elif match := zero_dim_reduce_scatter_pattern_multi_user.match(node):
                if not isinstance(match, Match):
                    raise AssertionError(f"expected a Match, got {type(match)}")
                reduce_scatters.append(
                    _ReduceScatterMatch(
                        match=match,
                        input_node=match.kwargs["input"],
                        reduce_scatter_node=match.nodes[0],
                        wait_tensor_node=node,
                        reduce_op=match.kwargs["reduce_op"],
                        scatter_dim=0,
                        group_name=match.kwargs["group_name"],
                    )
                )
    return list(reversed(reduce_scatters))


@dataclass
class _Matmul:
    nodes: list[torch.fx.Node]
    arg_ancestor_nodes: OrderedSet[torch.fx.Node] = field(init=False)
    A_node: torch.fx.Node
    B_node: torch.fx.Node
    pre_mm_reshape: torch.fx.Node | None
    post_mm_reshape: torch.fx.Node | None

    def __post_init__(self):
        if len(self.nodes) not in (1, 3):
            raise AssertionError(f"expected 1 or 3 nodes, got {len(self.nodes)}")
        if len(self.nodes) == 1:
            if self.nodes[0].target not in (aten.mm.default, aten._scaled_mm.default):
                raise AssertionError(
                    f"expected mm or _scaled_mm, got {self.nodes[0].target}"
                )
        else:
            if self.nodes[0].target is not aten.reshape.default:
                raise AssertionError(f"expected reshape, got {self.nodes[0].target}")
            if self.nodes[1].target not in (aten.mm.default, aten._scaled_mm.default):
                raise AssertionError(
                    f"expected mm or _scaled_mm, got {self.nodes[1].target}"
                )
            if self.nodes[2].target is not aten.reshape.default:
                raise AssertionError(f"expected reshape, got {self.nodes[2].target}")
        self.arg_ancestor_nodes = _find_ancestors(self.B_node)

    def replace_with(self, new_node: torch.fx.Node) -> None:
        """
        Replace the matmul with the new node.
        """
        graph = new_node.graph

        # For 2D-matmuls, we simply replace the mm node with `new_node`.
        if len(self.nodes) == 1:
            mm_node = self.nodes[0]
            if mm_node.target not in (aten.mm.default, aten._scaled_mm.default):
                raise AssertionError(f"expected mm or _scaled_mm, got {mm_node.target}")
            mm_node.replace_all_uses_with(new_node)
            return

        # An ND-matmul is reshape -> mm -> reshape sequence. We first replace
        # the second reshape node with `new_node`. Then, we ensure that the
        # original mm node in the sequence ends up with zero users by replacing
        # it with a reverse reshape of `new_node`.
        graph = new_node.graph
        if len(self.nodes) != 3:
            raise AssertionError(f"expected 3 nodes, got {len(self.nodes)}")
        mm_node = self.nodes[1]
        output_reshape_node = self.nodes[2]

        if mm_node.target not in (aten.mm.default, aten._scaled_mm.default):
            raise AssertionError(f"expected mm or _scaled_mm, got {mm_node.target}")
        if output_reshape_node.target is not aten.reshape.default:
            raise AssertionError(f"expected reshape, got {output_reshape_node.target}")

        output_reshape_node.replace_all_uses_with(new_node)
        if len(mm_node.users) > 1:
            with graph.inserting_after(new_node):
                new_mm_node = graph.call_function(
                    aten.reshape.default,
                    args=(new_node, list(_get_tensor(mm_node).shape)),
                )
            mm_node.replace_all_uses_with(new_mm_node)

    def erase(self) -> None:
        for node in reversed(self.nodes):
            if len(node.users) == 0:
                node.graph.erase_node(node)

    @classmethod
    def from_match(cls, match: list[torch.fx.Node]) -> "_Matmul":
        if len(match) not in (1, 3):
            raise AssertionError(f"expected 1 or 3 nodes, got {len(match)}")
        if match[0].target not in (
            aten.mm.default,
            aten.reshape.default,
        ):
            raise AssertionError(f"expected mm or reshape, got {match[0].target}")
        mm_node = match[0] if len(match) == 1 else match[1]
        return _Matmul(
            nodes=match,
            A_node=cast("torch.fx.Node", match[0].args[0]),
            B_node=cast("torch.fx.Node", mm_node.args[1]),
            # _Matmul handles reshapes via custom graph manipulation logic, see `replace_with()` method.
            # TODO: explore unifying the _Matmul and _ScaledMatmul approaches to handling reshapes.
            pre_mm_reshape=None,
            post_mm_reshape=None,
        )


@dataclass
class _ScaledMatmul(_Matmul):
    A_scale_node: torch.fx.Node
    B_scale_node: torch.fx.Node
    bias_node: torch.fx.Node | None
    result_scale_node: torch.fx.Node | None
    out_dtype: torch.dtype | None
    use_fast_accum: bool
    pre_mm_reshape: torch.fx.Node | None
    post_mm_reshape: torch.fx.Node | None

    def __post_init__(self):
        super().__post_init__()
        self.arg_ancestor_nodes |= _find_ancestors(self.A_scale_node)
        self.arg_ancestor_nodes |= _find_ancestors(self.B_scale_node)

    @classmethod
    def from_match(cls, match: list[torch.fx.Node]) -> "_ScaledMatmul":
        if len(match) not in (1, 3):
            raise AssertionError(f"expected 1 or 3 nodes, got {len(match)}")
        if match[0].target not in (
            aten._scaled_mm.default,
            aten.reshape.default,
        ):
            raise AssertionError(
                f"expected _scaled_mm or reshape, got {match[0].target}"
            )

        def get_arg(node: torch.fx.Node, idx: int, default: Any) -> Any:
            if idx >= len(node.args):
                return default
            return node.args[idx]

        # Use mm_node with 2D args for both A and B, even if this is a "reshape -> mm -> reshape" pattern.
        # We will store the reshapes in pre_mm_reshape and post_mm_reshape, to be referenced later to
        # produce the correct output shapes, reduce-scatter along the correct dimensions, etc.
        is_reshape_mm_reshape_pattern = match[0].target is aten.reshape.default
        mm_node = match[1] if is_reshape_mm_reshape_pattern else match[0]
        pre_mm_reshape = match[0] if is_reshape_mm_reshape_pattern else None
        post_mm_reshape = match[-1] if is_reshape_mm_reshape_pattern else None
        A_node = cast("torch.fx.Node", mm_node.args[0])
        B_node = cast("torch.fx.Node", mm_node.args[1])
        A_scale_node = cast("torch.fx.Node", mm_node.args[2])
        B_scale_node = cast("torch.fx.Node", mm_node.args[3])

