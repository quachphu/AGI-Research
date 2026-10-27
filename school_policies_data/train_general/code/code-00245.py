    if symbolic_helper._is_packed_list(index):
        indices = symbolic_helper._unpack_list(index)
    else:
        indices = [index]

    # Handle single mask index.
    if len(indices) == 1:
        index = indices[0]
        if not symbolic_helper._is_none(index) and (
            symbolic_helper._is_bool(index)
            or _type_utils.JitScalarType.from_value(index)
            == _type_utils.JitScalarType.UINT8
        ):
            index = opset9.nonzero(g, index)
            return g.op("GatherND", self, index)
    return opset9.index(g, self, index)


@_onnx_symbolic("aten::index_fill")
def index_fill(g: jit_utils.GraphContext, self, dim, index, value):
    expanded_index_shape, expanded_index = symbolic_helper._index_fill_reshape_helper(
        g, self, dim, index
    )
    value = symbolic_helper._maybe_get_scalar(value)
    value = symbolic_helper._if_scalar_type_as(value, self)
    expanded_value = opset9.expand(g, value, expanded_index_shape, None)
    return scatter(g, self, dim, expanded_index, expanded_value)


@_onnx_symbolic("aten::index_copy")
def index_copy(g: jit_utils.GraphContext, self, dim, index, source):
    _expanded_index_shape, expanded_index = symbolic_helper._index_fill_reshape_helper(
        g, self, dim, index
    )
    return scatter(g, self, dim, expanded_index, source)


@_onnx_symbolic("aten::bitwise_right_shift")
@_onnx_symbolic("aten::__rshift_")
def __rshift_(g: jit_utils.GraphContext, self, other):
    # make sure to cast other to self's type
    # (when self is long, make sure that other is not float)
    if _type_utils.JitScalarType.from_value(
        other, _type_utils.JitScalarType.UNDEFINED
    ) != _type_utils.JitScalarType.from_value(self):
        other = g.op(
            "Cast",
            other,
            to_i=_type_utils.JitScalarType.from_value(self).onnx_type(),
        )

    if (
        _type_utils.JitScalarType.from_value(self, _type_utils.JitScalarType.UNDEFINED)
        == _type_utils.JitScalarType.UINT8
    ):
        # shifting an unsigned integer right is a logical shift
        return g.op("BitShift", self, other, direction_s="RIGHT")

    return opset9.__rshift_(g, self, other)


@_onnx_symbolic("aten::bitwise_left_shift")
@_onnx_symbolic("aten::__lshift_")
def __lshift_(g: jit_utils.GraphContext, self, other):
    # make sure to cast other to self's type
    # (when self is long, make sure that other is not float)
    if _type_utils.JitScalarType.from_value(
        other, _type_utils.JitScalarType.UNDEFINED
    ) != _type_utils.JitScalarType.from_value(self):
        other = g.op(
            "Cast",
            other,
            to_i=_type_utils.JitScalarType.from_value(self).onnx_type(),
        )

    if (
        _type_utils.JitScalarType.from_value(self, _type_utils.JitScalarType.UNDEFINED)
        == _type_utils.JitScalarType.UINT8
    ):
        return g.op("BitShift", self, other, direction_s="LEFT")

    two = g.op("Constant", value_t=torch.tensor(2, dtype=torch.float32))
    # exponent (same type as self) has to be float or double in onnx::Pow
    if not symbolic_helper._is_fp(self):
        other = g.op("Cast", other, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    two_pow = g.op("Pow", two, other)
    two_pow = g.op(
        "Cast",
        two_pow,
        to_i=_type_utils.JitScalarType.from_value(self).onnx_type(),
    )
    lshift = g.op("Mul", self, two_pow)
    return lshift


def _get_im2col_indices_along_dim(
    g: jit_utils.GraphContext, input_d, kernel_size_d, dilation_d, padding_d, stride_d
):
    # Input is always 4-D (N, C, H, W)
    # Calculate indices of sliding blocks along spatial dimension
    # Slide kernel over input each dim d:
    # each dimension d ranges from 0 to input[d]+2xpadding[d]-dilation[d]x(kernel_size[d]-1)
    # with steps = stride

    blocks_d = g.op(
        "Add", input_d, g.op("Constant", value_t=torch.tensor(padding_d * 2))
    )
    blocks_d = g.op(
        "Sub",
        blocks_d,
        g.op("Constant", value_t=torch.tensor(dilation_d * (kernel_size_d - 1))),
    )

    # Stride kernel over input and find starting indices along dim d
    blocks_d_indices = g.op(
        "Range",
        g.op("Constant", value_t=torch.tensor(0)),
        blocks_d,
        g.op("Constant", value_t=torch.tensor(stride_d)),
    )

    # Apply dilation on kernel and find its indices along dim d
    kernel_grid = torch.arange(0, kernel_size_d * dilation_d, dilation_d)
    kernel_grid = g.op("Constant", value_t=kernel_grid.unsqueeze(0))

    # Broadcast and add kernel staring positions (indices) with
    # kernel_grid along dim d, to get block indices along dim d
    blocks_d_indices = symbolic_helper._unsqueeze_helper(
        g, blocks_d_indices, [0]
    )  # Reshape to [1, -1]
    kernel_mask = symbolic_helper._reshape_helper(
        g, kernel_grid, g.op("Constant", value_t=torch.tensor([-1, 1]))
    )
    block_mask = g.op("Add", blocks_d_indices, kernel_mask)

    return block_mask


def _get_im2col_padded_input(g: jit_utils.GraphContext, input, padding_h, padding_w):
    # Input is always 4-D tensor (N, C, H, W)
    # Padding tensor has the following format: (padding_h, padding_w)
    # Reshape the padding to follow ONNX format: (dim1_begin, dim2_begin,...,dim1_end, dim2_end,...)
    pad = g.op("Constant", value_t=torch.LongTensor([0, 0, padding_h, padding_w] * 2))
    return g.op("Pad", input, pad)


def _get_im2col_output_shape(g: jit_utils.GraphContext, input, kernel_h, kernel_w):
    batch_dim = size(g, input, g.op("Constant", value_t=torch.tensor(0)))
    channel_dim = size(g, input, g.op("Constant", value_t=torch.tensor(1)))
    channel_unfolded = g.op(
        "Mul", channel_dim, g.op("Constant", value_t=torch.tensor(kernel_h * kernel_w))
    )

    return g.op(
        "Concat",
        symbolic_helper._unsqueeze_helper(g, batch_dim, [0]),
        symbolic_helper._unsqueeze_helper(g, channel_unfolded, [0]),
        g.op("Constant", value_t=torch.tensor([-1])),
        axis_i=0,
    )


@_onnx_symbolic("aten::im2col")
@symbolic_helper.parse_args("v", "is", "is", "is", "is")
def im2col(g: jit_utils.GraphContext, input, kernel_size, dilation, padding, stride):
    # Input is always 4-D tensor (N, C, H, W)
    # All other args are int[2]

    input_h = size(g, input, g.op("Constant", value_t=torch.tensor(2)))
    input_w = size(g, input, g.op("Constant", value_t=torch.tensor(3)))

    stride_h, stride_w = stride[0], stride[1]
    padding_h, padding_w = padding[0], padding[1]
    dilation_h, dilation_w = dilation[0], dilation[1]
    kernel_h, kernel_w = kernel_size[0], kernel_size[1]

    blocks_row_indices = _get_im2col_indices_along_dim(
        g, input_h, kernel_h, dilation_h, padding_h, stride_h
    )
    blocks_col_indices = _get_im2col_indices_along_dim(
        g, input_w, kernel_w, dilation_w, padding_w, stride_w
    )

    output_shape = _get_im2col_output_shape(g, input, kernel_h, kernel_w)
    padded_input = _get_im2col_padded_input(g, input, padding_h, padding_w)

    # For a 4D matrix of size (1, 1, 3, 3) as below with kernel_size=2, stride=1, and dilation=1
    # [[[[1., 2., 3.,],
    #    [4., 5., 6.,],
    #    [7., 8., 9.,]]]]
    # First gather indices along rows (dim=2) with blocks_row_indices = [[0,1], [1,2]] to get:
    # [[[[[1., 2., 3.],
    #     [4., 5., 6.]],
    #    [[4., 5., 6.],
    #     [7., 8., 9.]]]]]
    # And then gather along cols (dim=4) with blocks_row_indices = [[0,1], [1,2]] to get:
    # [[[[[[1., 2.],
    #      [4., 5.]],
    #     [[2., 3.],
    #      [5., 6]]],
    #    [[[4., 5.],
    #      [7., 8.]],
    #     [[5., 6.],
    #      [8., 9.]]]]]]
    # Transpose dims 3 (depth) and 4 (rows), and then reshape to output shape (1, 1, 4, 4) to get:
    #  [[[1., 2., 4., 5.],
    #    [2., 3., 5., 6.],
    #    [4., 5., 7., 8.],
    #    [5., 6., 8., 9.]]]
    output = g.op("Gather", padded_input, blocks_row_indices, axis_i=2)
    output = g.op("Gather", output, blocks_col_indices, axis_i=4)
    output = g.op("Transpose", output, perm_i=[0, 1, 2, 4, 3, 5])
    return symbolic_helper._reshape_helper(g, output, output_shape)


@_onnx_symbolic("aten::narrow")
def narrow(g: jit_utils.GraphContext, input, dim, start, length):
    end = g.op("Add", start, length)
    return symbolic_helper._slice_helper(g, input, axes=dim, starts=start, ends=end)


@_onnx_symbolic("aten::flatten")
@symbolic_helper.quantized_args(True, False, False)
@symbolic_helper.parse_args("v", "i", "i")
def flatten(g: jit_utils.GraphContext, input, start_dim, end_dim):
    dim = symbolic_helper._get_tensor_rank(input)
    if dim == 1:
        return input
    # use ONNX's Flatten operator for cases where the output shape is 2D
    if start_dim == 1:
        if end_dim == -1 or (dim is not None and end_dim == dim - 1):
            return g.op("Flatten", input, axis_i=start_dim)
    elif start_dim == 0:
        if end_dim == -2 or (dim is not None and end_dim == dim - 2):
            return g.op("Flatten", input, axis_i=end_dim + 1)
    if dim is None:
        return symbolic_helper._unimplemented(
            "dim",
            "ONNX and PyTorch use different strategies to split the input. "
            "Input rank must be known at export time.",
        )
    # if end_dim is negative add dim
    if end_dim < 0:
        end_dim = dim + end_dim

    return symbolic_helper._flatten_helper(g, input, start_dim, end_dim, dim)


@_onnx_symbolic("aten::linalg_vector_norm")
@symbolic_helper.parse_args("v", "f", "is", "b", "v")
def linalg_vector_norm(
    g: jit_utils.GraphContext,
    self,
    ord,
    dim: Sequence[int] | None,
    keepdim: bool,
    dtype,
):
    return symbolic_helper._linalg_vector_norm_helper(g, self, ord, dim, keepdim, dtype)


@_onnx_symbolic("aten::embedding_bag")
@symbolic_helper.parse_args("v", "v", "v", "i", "i", "i", "v", "i", "i")
def embedding_bag(
    g: jit_utils.GraphContext,
    embedding_matrix,
    indices,
    offsets,
    scale_grad_by_freq,
    mode,
    sparse,
    per_sample_weights,
    include_last_offset,
    padding_idx,
):
    return symbolic_helper._embedding_bag_helper(
        g,
        embedding_matrix,
        indices,
        offsets,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights,
        include_last_offset,
        padding_idx,
    )


@_onnx_symbolic("aten::embedding_renorm")
@symbolic_helper.parse_args("v", "v", "f", "f")
def embedding_renorm(g: jit_utils.GraphContext, weight, indices, max_norm, norm_type):
    unique_indices = g.op("Unique", indices)
    partial_weight = g.op("Gather", weight, unique_indices)
    norm_i = int(norm_type)
    if norm_i == 1:
        norm_type = "ReduceL1"
    elif norm_i == 2:
        norm_type = "ReduceL2"
    else:
        raise errors.SymbolicValueError(
            f"Unsupported: ONNX export of embedding_renorm with norm: {norm_i}. "
            "Only 1. and 2. are supported.",
            weight,
        )
    partial_weight_norm = g.op(norm_type, partial_weight, axes_i=[1], keepdims_i=1)
    # https://github.com/pytorch/pytorch/blob/0a07488ed2c47765e337e290bd138c0e6e459cbd/aten/src/ATen/native/Embedding.cpp#L177
    # Add 1e-7 to prevent division by zero.
    partial_weight_norm_ = g.op(
        "Add", partial_weight_norm, g.op("Constant", value_t=torch.tensor(1e-7))
    )
    max_norm = torch.tensor(max_norm)
    scales = g.op("Div", max_norm, partial_weight_norm_)
    partial_weight_renorm = g.op("Mul", partial_weight, scales)
    partial_weight_renorm = g.op(
        "Where",
        g.op("Greater", partial_weight_norm, max_norm),
        partial_weight_renorm,
        partial_weight,
    )
    return g.op(
        "ScatterND",
        weight,
        symbolic_helper._unsqueeze_helper(g, unique_indices, [1]),
        partial_weight_renorm,
    )


@_onnx_symbolic("aten::chunk")
def chunk(g: jit_utils.GraphContext, self, chunks, dim):
    # Calculate chunk size for dynamic chunk
    dim_size = g.op("Gather", g.op("Shape", self), dim, axis_i=0)
    chunk_size_s = g.op(
        "Sub", chunks, g.op("Constant", value_t=torch.tensor([1], dtype=torch.long))
    )
    chunk_size = g.op("Div", g.op("Add", dim_size, chunk_size_s), chunks)
    # Create splits vector
    chunk_vec = [
        opset9.expand(g, chunk_size, chunk_size_s, None),
        g.op("Sub", dim_size, g.op("Mul", chunk_size, chunk_size_s)),
    ]
    chunk_vec = g.op("Concat", *chunk_vec, axis_i=0)
    return split(g, self, chunk_vec, dim)


@_onnx_symbolic("aten::normal")
def normal(
    g: jit_utils.GraphContext,
    mean,
    std,
    sizes=None,
    generator=None,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
):
    # If you can sample from a given distribution with mean 0 and variance 1, then you can easily sample from a
    # scale-location transformation of that distribution, which has mean mu and variance sigma's square. If x is a sample
    # from a mean 0 and variance 1 distribution then
    #       sigma x+mu
    # is a sample with mean mu and variance sigma's square.
    if sizes is not None and not symbolic_helper._is_none(sizes):
        mean = opset9.expand(g, mean, sizes, None)
    result = opset9.mul(g, std, g.op("RandomNormalLike", mean))
    return add(g, result, mean)


@_onnx_symbolic("aten::atleast_1d")
def atleast_1d(g: jit_utils.GraphContext, self: torch._C.Value):
    # NOTE: If it's 0D, reshape to 1D

    # NOTE: self could be a packed list or a tensor
    if symbolic_helper._is_value(self) and symbolic_helper._is_packed_list(self):
        tensor_list = symbolic_helper._unpack_list(self)
        new_tensor_list = []
        for tensor in tensor_list:
            new_tensor = tensor
            tensor_rank = symbolic_helper._get_tensor_rank(tensor)
            if tensor_rank == 0:
                new_tensor = symbolic_helper._reshape_helper(
                    g, new_tensor, g.op("Constant", value_t=torch.tensor([1]))
                )
            new_tensor_list.append(new_tensor)
        return g.op("SequenceConstruct", *new_tensor_list)

    tensor_rank = symbolic_helper._get_tensor_rank(self)
    if tensor_rank == 0:
        self = symbolic_helper._reshape_helper(
            g, self, g.op("Constant", value_t=torch.tensor([1]))
        )
    return self


@_onnx_symbolic("aten::atleast_2d")
def atleast_2d(g: jit_utils.GraphContext, self: torch._C.Value):
    # NOTE: If it's 0D, reshape to 2D
    #       If it's 1D, unsqueeze to 2D

    # NOTE: self could be a packed list or a tensor
    if symbolic_helper._is_value(self) and symbolic_helper._is_packed_list(self):
        tensor_list = symbolic_helper._unpack_list(self)
        new_tensor_list = []
        for tensor in tensor_list:
            new_tensor = tensor
            tensor_rank = symbolic_helper._get_tensor_rank(tensor)
            if tensor_rank == 0:
                new_tensor = symbolic_helper._reshape_helper(
                    g, new_tensor, g.op("Constant", value_t=torch.tensor([1, 1]))
                )
            elif tensor_rank == 1:
                new_tensor = symbolic_helper._unsqueeze_helper(
                    g, new_tensor, axes_i=[0]
                )
            new_tensor_list.append(new_tensor)
        return g.op("SequenceConstruct", *new_tensor_list)

    tensor_rank = symbolic_helper._get_tensor_rank(self)
    if tensor_rank == 0:
        self = symbolic_helper._reshape_helper(
            g, self, g.op("Constant", value_t=torch.tensor([1, 1]))
        )
    elif tensor_rank == 1:
        self = symbolic_helper._unsqueeze_helper(g, self, axes_i=[0])
    return self


@_onnx_symbolic("aten::atleast_3d")
def atleast_3d(g: jit_utils.GraphContext, self: torch._C.Value):
    # NOTE: If it's 0D, reshape to 3D
    #       If it's 1D, unsqueeze to 3D
    #       If it's 2D, unsqueeze to 3D

    # NOTE: self could be a packed list or a tensor
    if symbolic_helper._is_value(self) and symbolic_helper._is_packed_list(self):
        tensor_list = symbolic_helper._unpack_list(self)
        new_tensor_list = []
        for tensor in tensor_list:
            new_tensor = tensor
            tensor_rank = symbolic_helper._get_tensor_rank(tensor)
            if tensor_rank == 0:
                new_tensor = symbolic_helper._reshape_helper(
                    g, new_tensor, g.op("Constant", value_t=torch.tensor([1, 1, 1]))
                )
            elif tensor_rank == 1:
                new_tensor = symbolic_helper._unsqueeze_helper(
                    g, new_tensor, axes_i=[0]
                )
                new_tensor = symbolic_helper._unsqueeze_helper(
                    g, new_tensor, axes_i=[-1]
                )
            elif tensor_rank == 2:
                new_tensor = symbolic_helper._unsqueeze_helper(
                    g, new_tensor, axes_i=[-1]
                )
            new_tensor_list.append(new_tensor)
        return g.op("SequenceConstruct", *new_tensor_list)

    tensor_rank = symbolic_helper._get_tensor_rank(self)
    if tensor_rank == 0:
        self = symbolic_helper._reshape_helper(
            g, self, g.op("Constant", value_t=torch.tensor([1, 1, 1]))
        )
    elif tensor_rank == 1:
        self = symbolic_helper._unsqueeze_helper(g, self, axes_i=[0])
        self = symbolic_helper._unsqueeze_helper(g, self, axes_i=[-1])
    elif tensor_rank == 2:
        self = symbolic_helper._unsqueeze_helper(g, self, axes_i=[-1])
    return self


@_onnx_symbolic("prim::ConstantChunk")
def prim_constant_chunk(g: jit_utils.GraphContext, self, chunks, dim):
    input_shape = g.op("Shape", self)
    axis = g.op("Constant", value_t=torch.tensor([dim], dtype=torch.long))
    input_shape_dim = g.op("Gather", input_shape, axis, axis_i=0)
    start = g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))
    chunk_size = g.op("Constant", value_t=torch.tensor([chunks], dtype=torch.long))
    chunk_size_minus_1 = g.op(
        "Constant", value_t=torch.tensor([chunks - 1], dtype=torch.long)
    )
    input_shape_dim_shift = g.op("Add", input_shape_dim, chunk_size_minus_1)
    chunk_dim = g.op("Div", input_shape_dim_shift, chunk_size)
    res = []
    for i in range(chunks):
        index = g.op("Constant", value_t=torch.tensor([i + 1], dtype=torch.long))
        end = g.op("Mul", chunk_dim, index)
        res.append(g.op("Slice", self, start, end, axis))
        start = end
    return res


@_onnx_symbolic("aten::hstack")
def hstack(g: jit_utils.GraphContext, tensor_list: _C.Value):
    tensor_list = atleast_1d(g, tensor_list)
    first_tensor = g.op(
        "SequenceAt",
        tensor_list,
        g.op("Constant", value_t=torch.tensor(0, dtype=torch.long)),
    )
    first_tensor_shape = g.op("Shape", first_tensor)
    first_tensor_dim = g.op("Size", first_tensor_shape)

    const_one = g.op("Constant", value_t=torch.tensor(1, dtype=torch.long))
    equal_to_one = g.op("Equal", first_tensor_dim, const_one)

    (
        if_op_greater,
        (if_context_equal, else_context_equal),
        _,
    ) = jit_utils.add_op_with_blocks(g, "If", equal_to_one, n_blocks=2, outputs=1)
    result_if = if_context_equal.op(
        "ConcatFromSequence", tensor_list, axis_i=0, new_axis_i=0
    )
    utils._add_output_to_block(if_context_equal.block, result_if)
    result_else = else_context_equal.op(
        "ConcatFromSequence", tensor_list, axis_i=1, new_axis_i=0
    )
    utils._add_output_to_block(else_context_equal.block, result_else)
    result = if_op_greater.node().output()

    return result


@_onnx_symbolic("aten::vstack")
def vstack(g: jit_utils.GraphContext, tensor_list: _C.Value):
    tensor_list = atleast_2d(g, tensor_list)
    return g.op("ConcatFromSequence", tensor_list, axis_i=0, new_axis_i=0)
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations


__all__ = [
    "_apply_params",
    "_arange_cast_helper",
    "_arange_helper",
    "_argmin_argmax_helper",
    "_as_list_type",
    "_avgpool_helper",
    "_batchnorm_helper",
    "_block_list_in_opset",
    "_embedding_bag_helper",
    "_flatten_helper",
    "_generate_wrapped_number",
    "_get_const",
    "_get_dim_for_cross",
    "_get_interpolate_attributes",
    "_get_tensor_dim_size",
    "_get_tensor_rank",
    "_get_tensor_sizes",
    "_handle_reduce_dim_none",
    "_if_scalar_type_as",
    "_index_fill_reshape_helper",
    "_interpolate_get_scales_and_mode",
    "_interpolate_get_scales_if_available",
    "_interpolate_get_scales",
    "_interpolate_helper",
    "_interpolate_size_to_scales",
    "_interpolate_warning",
    "_is_bool",
    "_is_constant",
    "_is_fp",
    "_is_list",
    "_is_none",
    "_is_onnx_constant",
    "_is_packed_list",
    "_is_scalar_list",
    "_is_split_static",
    "_is_tensor_list",
    "_is_tensor",
    "_is_tuple_construct",
    "_is_value",
    "_linalg_vector_norm_helper",
    "_lt_helper",
    "_max_helper",
    "_maybe_cast_reduce_op_input",
    "_maybe_cast_to_type",
    "_maybe_get_const",
    "_maybe_get_scalar",
    "_min_helper",
    "_node_get",
    "_numel_helper",
    "_onnx_opset_unsupported_detailed",
    "_onnx_opset_unsupported",
    "_onnx_unsupported",
    "_op_with_optional_float_cast",
    "_optional_input_placeholder_tensor",
    "_overload_by_arg_count",
    "_parse_arg",
    "_reduce_op_symbolic_helper",
    "_reduce_with_dtype_helper",
    "_reducesum_helper",
    "_repeat_interleave_single_value_repeat_helper",
    "_repeat_interleave_split_helper",
    "_reshape_helper",
    "_scalar",
    "_scatter_helper",
    "_select_helper",
    "_size_helper",
    "_slice_helper",
    "_sort_helper",
    "_squeeze_helper",
    "_topk_helper",
    "_try_get_scalar_type",
    "_type_promote_from_values",
    "_unbind_helper",
    "_unimplemented",
    "_unpack_list",
    "_unpack_quantized_tensor",
    "_unpack_tuple",
    "_unsqueeze_helper",
    "_var_mean_helper",
    "args_have_same_dtype",
    "cast_pytorch_to_onnx",
    "check_training_mode",
    "dequantize_helper",
    "is_complex_value",
    "parse_args",
    "pytorch_name_to_type",
    "quantize_helper",
    "quantized_args",
    "requantize_bias_helper",
    "scalar_name_to_pytorch",
    "scalar_type_to_onnx",
    "scalar_type_to_pytorch_type",
]

import functools
import inspect
import math
import sys
import typing
import warnings
from typing import (
    Any,
    Concatenate as _Concatenate,
    Literal,
    NoReturn,
    TypeVar as _TypeVar,
)
from typing_extensions import ParamSpec as _ParamSpec, TypeIs as _TypeIs

import torch
import torch._C._onnx as _C_onnx
from torch import _C
from torch.onnx import _constants, errors
from torch.onnx._internal.torchscript_exporter import _type_utils, jit_utils, utils
from torch.onnx._internal.torchscript_exporter._globals import GLOBALS


if typing.TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch.types import Number

_T = _TypeVar("_T")
_U = _TypeVar("_U")
_P = _ParamSpec("_P")

# ---------------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------------

_ValueDescriptor = Literal[
    "v",
    "i",
    "is",
    "f",
    "fs",
    "b",
    "s",
    "t",
    "none",
]


def _parse_arg(
    value,
    desc: _ValueDescriptor,
    arg_name: str | None = None,
    node_name: str | None = None,
):
    if desc == "none":
        return value
    if desc == "v" or not _is_value(value):
        return value

    node = value.node()
    if node.mustBeNone():
        return None
    if node.kind() == "onnx::Constant":
        node_val = _node_get(node, "value")
        if desc == "i":
            return int(node_val)
        elif desc == "f":
            return float(node_val)
        elif desc == "b":
            return bool(node_val)
        elif desc == "s":
            return str(node_val)
        elif desc == "t":
            return node_val
        elif desc == "is":
            return [int(v) for v in node_val]
        elif desc == "fs":
            return [float(v) for v in node_val]
        else:
            raise errors.SymbolicValueError(
                f"ONNX symbolic does not understand the Constant node '{node}' "
                f"specified with descriptor '{desc}'.",
                value,
            )
    elif node.kind() == "prim::ListConstruct":
        if desc == "is":
            for v in node.inputs():
                element_node = v.node()
                if element_node.kind() != "onnx::Constant":
                    raise errors.SymbolicValueError(
                        f"Failed to export a node '{element_node}' "
                        f"(in list node {node}) "
                        f"because it is not constant. "
                        f"Please try to make things (e.g. kernel sizes) static if possible.",
                        value,
                    )
            return [int(_node_get(v.node(), "value")) for v in value.node().inputs()]
        else:
            raise errors.SymbolicValueError(
                f"ONNX symbolic does not know how to unpack the ListConstruct node that "
                f"is not a list of integers: '{node}'",
                value,
            )

    if arg_name is None or node_name is None:
        raise errors.SymbolicValueError(
            f"Expected node type 'onnx::Constant', got '{node.kind()}'.",
            value,
        )

    raise errors.SymbolicValueError(
        "Expected node type 'onnx::Constant' "
        f"for argument '{arg_name}' of node '{node_name}', got '{node.kind()}'.",
        value,
    )


def _node_get(node: _C.Node, key: str):
    """Gets attributes of a node which is polymorphic over return type."""
    if not isinstance(node, _C.Node):
        raise AssertionError(f"Expected _C.Node, got {type(node)}")
    sel = node.kindOf(key)
    return getattr(node, sel)(key)


def _is_onnx_constant(value: _C.Value):
    """Whether a Value is an ONNX constant."""
    return value.node().kind() == "onnx::Constant"


def _maybe_get_const(
    value: _C.Value | torch.Tensor | Number | Sequence | None,
    descriptor: _ValueDescriptor,
):
    # NOTE: prim::Constant at this stage usually means something not compatible in ONNX,
    # otherwise it'd be converted to onnx::Constant
    # TODO(justinchuby): Replace insinstance with _is_value once we figure out mypy
    if isinstance(value, _C.Value) and _is_onnx_constant(value):
        return _parse_arg(value, descriptor)
    return value


def _maybe_get_scalar(value):
    value_t = _maybe_get_const(value, "t")
    if isinstance(value_t, torch.Tensor) and value_t.shape == ():
        return value_t
    return value


def _get_const(value, desc, arg_name):
    if not _is_constant(value):
        raise errors.SymbolicValueError(
            f"ONNX symbolic expected a constant value of the '{arg_name}' argument, "
            f"got '{value}'",
            value,
        )
    return _parse_arg(value, desc)


def _unpack_list(list_value: _C.Value) -> list[_C.Value]:
    list_node = list_value.node()
    if list_node.kind() != "prim::ListConstruct":
        raise errors.SymbolicValueError(
            f"ONNX symbolic expected node type prim::ListConstruct, got '{list_node}'.",
            list_value,
        )
    return list(list_node.inputs())


def _unpack_tuple(tuple_value: _C.Value) -> tuple[_C.Value, ...]:
    tuple_node = tuple_value.node()
    if not _is_tuple_construct(tuple_value):
        raise errors.SymbolicValueError(
            f"ONNX symbolic expected node type 'prim::TupleConstruct', "
            f"got '{tuple_node.kind()}'.",
            tuple_value,
        )
    return tuple(tuple_node.inputs())


def _unpack_quantized_tensor(tuple_value: _C.Value) -> tuple[_C.Value, ...]:
    """Unpacks a quantized tensor into a tuple of tensor and scale/zero_point.
    Args:
        tuple_value: A tuple of tensor, scale, zero_point, and optionally axis.
    Returns:
        A tuple of tensor, scale, zero_point, and optionally axis.
    """
    tuple_node = tuple_value.node()
    # A quantized tensor is represented as tuple of the form (tensor, scale, zero_point, <axis>)
    if not _is_tuple_construct(tuple_value):
        raise errors.SymbolicValueError(
            f"ONNX symbolic expected the output of `{tuple_node}` to be a quantized "
            f"tensor. Is this likely due to missing support for quantized "
            f"`{tuple_node.kind()}`. Please create an issue on {_constants.PYTORCH_GITHUB_ISSUES_URL}",
            tuple_value,
        )
    unpacked = tuple(tuple_node.inputs())
    if not (len(unpacked) == 3 or len(unpacked) == 4):
        raise AssertionError(f"Expected 3 or 4 elements, got {len(unpacked)}")
    return unpacked


# Check if list_value is output from prim::ListConstruct
# This is usually called before _unpack_list to ensure the list can be unpacked.
def _is_packed_list(list_value: Any) -> bool:
    return _is_value(list_value) and list_value.node().kind() == "prim::ListConstruct"


def parse_args(
    *arg_descriptors: _ValueDescriptor,
) -> Callable[[Callable[_Concatenate[_U, _P], _T]], Callable[_Concatenate[_U, _P], _T]]:
    """A decorator which converts args from torch._C.Value to built-in types.

    For example:

    ```
    @parse_args('v', 'i', 'fs')
    foo(g, a, b, c):
        # a is torch._C.Value
        # b is int
        # c is list of floats
    ```

    Args:
        arg_descriptors: list of str, where each element is
            a string that specifies the type to convert to. Valid descriptors:
            "v": no conversion, keep torch._C.Value.
            "i": int
            "is": list of int
            "f": float
            "fs": list of float
            "b": bool
            "s": str
            "t": torch.Tensor
            "none": the variable is unused
    """

    def decorator(
        fn: Callable[_Concatenate[_U, _P], _T],
    ) -> Callable[_Concatenate[_U, _P], _T]:
        fn._arg_descriptors = arg_descriptors  # type: ignore[attr-defined]

        @functools.wraps(fn)
        def wrapper(g: _U, *args: _P.args, **kwargs: _P.kwargs) -> _T:
            # some args may be optional, so the length may be smaller
            FILE_BUG_MSG = (
                "If you believe this is not due to custom symbolic implementation within your code or "
                "an external library, please file an issue at "
                "https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml to report this bug."
            )
            if len(arg_descriptors) < len(args):
                raise AssertionError(
                    f"A mismatch between the number of arguments ({len(args)}) and "
                    f"their descriptors ({len(arg_descriptors)}) was found at "
                    f"symbolic function '{fn.__name__}'. {FILE_BUG_MSG}"
                )

            try:
                sig = inspect.signature(fn)
                arg_names = list(sig.parameters.keys())[1:]
                fn_name = fn.__name__
            except Exception:
                # FIXME(justinchuby): Avoid catching Exception.
                # Catch a more specific exception instead.
                arg_names = [None] * len(args)  # type: ignore[list-item]
                fn_name = None
            # pyrefly: ignore [bad-assignment]
            args = [
                _parse_arg(arg, arg_desc, arg_name, fn_name)  # type: ignore[method-assign]
                for arg, arg_desc, arg_name in zip(args, arg_descriptors, arg_names)
            ]
            # only support _outputs in kwargs
            if len(kwargs) > 1:
                raise AssertionError(
                    f"Symbolic function {fn.__name__}'s '**kwargs' can contain a single "
                    f"key/value entry. "
                    f"{FILE_BUG_MSG}"
                )

            if len(kwargs) == 1:
                if "_outputs" not in kwargs:
                    raise AssertionError(
                        f"Symbolic function {fn.__name__}'s '**kwargs' can only contain "
                        f"'_outputs' key at '**kwargs'. "
                        f"{FILE_BUG_MSG}"
                    )
            return fn(g, *args, **kwargs)

        return wrapper

    return decorator


def quantized_args(
    *arg_q_descriptors: bool,
    scale: float | None = None,
    zero_point: int | None = None,
    quantize_output: bool = True,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """A decorator which extends support for quantized version of the base operator.

    Quantization is detected by examining the arguments that are annotated by
    `arg_q_descriptors`.

    If quantization is detected, the base operator symbolic function will be wrapped with
    argument de-quantization and output quantization.

    Otherwise, only the base symbolic function will be invoked.

    For example:

    ```
    @quantized_args(True, False)
    def foo(g, x, y):
        return x + y
    ```

    is equivalent to

    ```
    def q_foo(g, x, y):
        if is_quantized_tensor(x):
            x = dequantize(x)
            out = foo(g, x, y)
            return quantize(out)
        else:
            return foo(g, x, y)
    ```

    Args:
        arg_q_descriptors: A sequence of bool, where each element represents if the
          argument is QTensor for quantized version of this operator. It defaults
          to False for unspecified (variable length) arguments.
        scale: Quantized output scale. If None, derive from
          the first quantized input scale.
        zero_point: Quantized output zero point. If None,
          derive from the first quantized input zero point.
        quantize_output: If True, quantize the output of the base operator. Default is True
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(g, *args, **kwargs):
            nonlocal scale
            nonlocal zero_point
            if scale is not None:
                _scale = g.op("Constant", value_t=torch.tensor(scale))
            else:
                _scale = None
            if zero_point is not None:
                _zero_point = g.op("Constant", value_t=torch.tensor(zero_point))
            else:
                _zero_point = None

            # Support variable length arguments by marking unspecified ones as non-quantized
            arg_q_descriptors_extended = arg_q_descriptors + (False,) * (
                len(args) - len(arg_q_descriptors)
            )
            descriptor_args = tuple(zip(arg_q_descriptors_extended, args))

            def _is_arg_quantized(descriptor, arg):
                return descriptor and _is_value(arg) and _is_tuple_construct(arg)

            # Run regular symbolic function if none of the argument is QTensor.
            is_quantized: list[bool] = []
            for descriptor, arg in descriptor_args:
                # ListConstruct
                if _is_packed_list(arg):
                    is_quantized.extend(
                        _is_arg_quantized(descriptor, arg_input)
                        for arg_input in arg.node().inputs()
                    )
                else:
                    is_quantized.append(_is_arg_quantized(descriptor, arg))

            if not any(is_quantized):
                return fn(g, *args, **kwargs)

            # Dequantize arguments that are quantized
            non_quantized_args = []
            for descriptor, arg in descriptor_args:
                if _is_arg_quantized(descriptor, arg):
                    # Quantized arg is a tuple of (value, scale, zero_point)
                    dequantized_arg, arg_scale, arg_zero_point, _ = dequantize_helper(
                        g, arg
                    )
                    non_quantized_args.append(dequantized_arg)
                    # Set scale and zero_point to the first quantized input if not already set
                    if _scale is None:
                        _scale = arg_scale
                    if _zero_point is None:
                        _zero_point = arg_zero_point
                # ListConstruct
                elif _is_packed_list(arg):
                    for arg_input in arg.node().inputs():
                        if _is_arg_quantized(descriptor, arg_input):
                            # Quantized arg is a tuple of (value, scale, zero_point)
                            (
                                dequantized_arg,
                                arg_scale,
                                arg_zero_point,
                                _,
                            ) = dequantize_helper(g, arg_input)
                            # Set scale and zero_point to the first quantized input if not already set
                            if _scale is None:
                                _scale = arg_scale
                            if _zero_point is None:
                                _zero_point = arg_zero_point
                            arg_input.replaceAllUsesWith(dequantized_arg)
                    non_quantized_args.append(arg)
                else:
                    # Non-quantized arg
                    non_quantized_args.append(arg)
            # TODO(justinchuby): Only single output is supported for now. We may want to
            # support multiple outputs in the future.
            output = fn(g, *non_quantized_args, **kwargs)

            if _scale is None:
                raise AssertionError("Bug: Scale must be set for quantized operator")
            if _zero_point is None:
                raise AssertionError(
                    "Bug: Zero point must be set for quantized operator"
                )

            if quantize_output:
                return quantize_helper(g, output, _scale, _zero_point)
            return output

        return wrapper

    return decorator


def _scalar(x: Any) -> Number | None:
    """Convert a scalar tensor into a Python value."""
    if isinstance(x, torch.Tensor) and x.shape == ():
        return x.item()
    return None


def _if_scalar_type_as(self, tensor):
    """
    Convert self into the same type of tensor, as necessary.
    We only support implicit casting for scalars, so we never
    actually need to insert an ONNX cast operator here; just
    fix up the scalar.
    """
    if isinstance(self, _C.Value):
        return self

    scalar_type = _type_utils.JitScalarType.from_value(
        tensor, _type_utils.JitScalarType.UNDEFINED
    )
    if scalar_type != _type_utils.JitScalarType.UNDEFINED:
        ty = scalar_type.scalar_name().lower()
        return getattr(self, ty)()
    return self


def _is_none(x: Any) -> bool:
    return x is None or (x.node().mustBeNone() if isinstance(x, _C.Value) else False)


def _is_value(x: Any) -> _TypeIs[_C.Value]:
    return isinstance(x, _C.Value)


def _is_constant(value: Any) -> bool:
    return not _is_value(value) or value.node().kind() in {
        "onnx::Constant",
        "prim::Constant",
    }


def _is_tensor(x: _C.Value) -> bool:
    return x.type().isSubtypeOf(_C.TensorType.get())


# Note: _C.JitType is not exposed to Python and cannot be checked in runtime.
def _as_list_type(jit_type: _C.JitType) -> _C.ListType | None:
    if isinstance(jit_type, _C.ListType):
        return jit_type
    return None


def _is_list(x: _C.Value) -> bool:
    return _as_list_type(x.type()) is not None


def _is_tensor_list(x: _C.Value) -> bool:
    x_type = _as_list_type(x.type())
    if x_type is None:
        return False
    return isinstance(x_type.getElementType(), _C.TensorType)


def _is_scalar_list(x: _C.Value) -> bool:
    """Checks if x is a scalar list, for example: List[float], List[int].

    Besides checking the type is ListType, we also check if the data type is
    a valid ONNX data type.
    """
    x_type = _as_list_type(x.type())
    if x_type is None:
        return False
    scalar_type = _type_utils.JitScalarType.from_value(x)
    return scalar_type.onnx_compatible()


def _is_tuple_construct(x: _C.Value) -> bool:
    return x.node().kind() == "prim::TupleConstruct"


def is_complex_value(x: _C.Value) -> bool:
    if not _is_value(x):
        raise AssertionError("Expected a Value")
    return _type_utils.JitScalarType.from_value(
        x, _type_utils.JitScalarType.UNDEFINED
    ) in {
        _type_utils.JitScalarType.COMPLEX32,
        _type_utils.JitScalarType.COMPLEX64,
        _type_utils.JitScalarType.COMPLEX128,
    }


def _get_tensor_rank(x: _C.Value) -> int | None:
    if not _is_tensor(x) or x.type() is None:
        return None
    x_type = x.type()
    x_type = typing.cast(_C.TensorType, x_type)
    return x_type.dim()


def _get_tensor_sizes(x: _C.Value, allow_nonstatic: bool = True):
    if not _is_tensor(x) or x.type() is None:
        return None
    x_type = x.type()
    x_type = typing.cast(_C.TensorType, x_type)
    if allow_nonstatic:
        # Each individual symbol is returned as None.
        # e.g. [1, "a", "b"] -> [1, None, None]
        return x_type.varyingSizes()
    # returns None, if exists any symbol in sizes.
    # e.g. [1, "a", "b"] -> None
    return x_type.sizes()


def _get_tensor_dim_size(x: _C.Value, dim: int) -> int | None:
    sizes = _get_tensor_sizes(x)
    return sizes[dim] if sizes else None


def _get_dim_for_cross(x: _C.Value, dim: int | None):
    if dim == -1:
        tensor_rank = _get_tensor_rank(x)
        if tensor_rank is None:
            raise AssertionError("Expected tensor_rank to be non-None")
        return dim + tensor_rank
    # If dim is not given, it defaults to the first dimension found with the size 3
    if dim is None:
        sizes = _get_tensor_sizes(x)
        if sizes is None:
            raise AssertionError("Expected sizes to be non-None")
        for index, size in enumerate(sizes):
            if size is not None and size == 3:
                return index
    return dim


def _unimplemented(op: str, msg: str, value: _C.Value | None = None) -> None:
    # For BC reasons, the behavior for Caffe2 does not raise exception for unimplemented operators
    if GLOBALS.operator_export_type == _C_onnx.OperatorExportTypes.ONNX:
        _onnx_unsupported(f"{op}, {msg}", value)


def _onnx_unsupported(op_name: str, value: _C.Value | None = None) -> NoReturn:
    message = (
        f"Unsupported: ONNX export of operator {op_name}. "
        f"Please feel free to request support or submit a pull request "
        f"on PyTorch GitHub: {_constants.PYTORCH_GITHUB_ISSUES_URL}"
    )
    if isinstance(value, _C.Value):
        raise errors.SymbolicValueError(
            message,
            value,
        )
    raise errors.OnnxExporterError(message)


def _onnx_opset_unsupported(
    op_name: str,
    current_opset: int,
    supported_opset: int,
    value: _C.Value | None = None,
) -> NoReturn:
    message = (
        f"Unsupported: ONNX export of {op_name} in opset {current_opset}. "
        f"Please try opset version {supported_opset}."
    )
    if isinstance(value, _C.Value):
        raise errors.SymbolicValueError(
            message,
            value,
        )
    raise errors.OnnxExporterError(message)


def _onnx_opset_unsupported_detailed(
    op_name: str,
    current_opset: int,
    supported_opset: int,
    reason: str,
    value: _C.Value | None = None,
) -> NoReturn:
    message = (
        f"Unsupported: ONNX export of {op_name} in "
        f"opset {current_opset}. {reason}. Please try opset version {supported_opset}."
    )
    if isinstance(value, _C.Value):
        raise errors.SymbolicValueError(
            message,
            value,
        )
    raise errors.OnnxExporterError(message)


def _block_list_in_opset(name: str):
    def symbolic_fn(*args, **kwargs):
        raise errors.OnnxExporterError(
            f"ONNX export failed on {name}, which is not implemented for opset "
            f"{GLOBALS.export_onnx_opset_version}. "
            "Try exporting with other opset versions."
        )

    return symbolic_fn


def _try_get_scalar_type(*args) -> _type_utils.JitScalarType | None:
    for arg in args:
        scalar_type = _type_utils.JitScalarType.from_value(
            arg, _type_utils.JitScalarType.UNDEFINED
        )
        if scalar_type != _type_utils.JitScalarType.UNDEFINED:
            return scalar_type
    return None


def _type_promote_from_values(*args) -> _type_utils.JitScalarType:
    undef = _type_utils.JitScalarType.UNDEFINED
    jit_types = [_try_get_scalar_type(arg) for arg in args]
    if len(jit_types) == 0:
        return undef
    if len(jit_types) == 1:
        return jit_types[0]  # type: ignore[return-value]
    new_dtype = jit_types[0].dtype()  # type: ignore[union-attr]
    for t in jit_types:
        new_dtype = torch.promote_types(new_dtype, t.dtype())  # type: ignore[union-attr]
    return _type_utils.JitScalarType.from_dtype(new_dtype)


def _maybe_cast_to_type(
    g: jit_utils.GraphContext, value, jit_type: _type_utils.JitScalarType
):
    if (
        _type_utils.JitScalarType.from_value(value, _type_utils.JitScalarType.UNDEFINED)
        != jit_type
    ):
        return g.op(
            "Cast",
            value,
            to_i=jit_type.onnx_type(),
        )
    return value


def _select_helper(g: jit_utils.GraphContext, self, dim, index, apply_reshape=True):
    index_const = _maybe_get_scalar(index)
    index_dim = _get_tensor_rank(index)
    if not _is_value(index_const):
        # Index is a constant scalar. Make it a size 1 constant tensor.
        index = g.op("Constant", value_t=torch.LongTensor([index_const]))
    elif index_dim is not None and apply_reshape:
        if index_dim == 0:
            # Index is a scalar. Reshape it to a size 1 tensor.
            index = _reshape_helper(
                g, index, g.op("Constant", value_t=torch.LongTensor([1]))
            )

    index_scalar_type = _type_utils.JitScalarType.from_value(
        index, _type_utils.JitScalarType.UNDEFINED
    )
    if index_scalar_type not in {
        _type_utils.JitScalarType.INT64,
        _type_utils.JitScalarType.INT,
    }:
        index = g.op("Cast", index, to_i=_C_onnx.TensorProtoDataType.INT64)
    return g.op("Gather", self, index, axis_i=dim)


def _slice_helper(
    g: jit_utils.GraphContext,
    input,
    axes,
    starts,
    ends,
    steps=None,
):
    if g.opset <= 9:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import (
            _slice as _slice9,
        )

        return _slice9(g, input, axes, starts, ends)
    else:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset10 import (
            _slice as _slice10,
        )

        return _slice10(g, input, axes, starts, ends, steps)


def _is_fp(value) -> bool:
    return _type_utils.JitScalarType.from_value(
        value, _type_utils.JitScalarType.UNDEFINED
    ) in {
        _type_utils.JitScalarType.FLOAT,
        _type_utils.JitScalarType.DOUBLE,
        _type_utils.JitScalarType.HALF,
        _type_utils.JitScalarType.BFLOAT16,
    }


def _is_bool(value) -> bool:
    return (
        _type_utils.JitScalarType.from_value(value, _type_utils.JitScalarType.UNDEFINED)
        == _type_utils.JitScalarType.BOOL
    )


def _generate_wrapped_number(g: jit_utils.GraphContext, scalar):
    """Creates a wrapped number based on https://github.com/pytorch/pytorch/issues/9515.

    A Tensor is a considered a "wrapped number" if it is
    auto-wrapped from a C++ or Python number type. Integer types are
    wrapped as 0-dim int64 tensors and floating-point types are
    wrapped as 0-dim double tensors.

    The input to this function is constant value. If the data type
    is a floating point type, it is converted to a 0-dim double
    tensor, else it is converted to a 0-dim tensor of its original type
    """
    if isinstance(scalar, torch.Tensor):
        raise AssertionError("Expected scalar, not torch.Tensor")
    if isinstance(scalar, float):
        return g.op("Constant", value_t=torch.tensor(scalar, dtype=torch.double))
    return g.op("Constant", value_t=torch.tensor(scalar))


def _sort_helper(g: jit_utils.GraphContext, input, dim, descending=True, out=None):
    if out is not None:
        _unimplemented("Sort", "Out parameter is not supported")
    shape_ = g.op("Shape", input)
    dim_size_ = g.op(
        "Gather",
        shape_,
        g.op("Constant", value_t=torch.tensor([dim], dtype=torch.int64)),
    )
    if g.opset <= 10:
        if not descending:
            _unimplemented("Sort", "Ascending is not supported")
        return g.op("TopK", input, dim_size_, axis_i=dim, outputs=2)
    else:
        return g.op(
            "TopK", input, dim_size_, axis_i=dim, largest_i=descending, outputs=2
        )


def _topk_helper(
    g: jit_utils.GraphContext, input, k, dim, largest=True, sorted=False, out=None
):
    if out is not None:
        _unimplemented("TopK", "Out parameter is not supported")
    if not _is_value(k):
        k = g.op("Constant", value_t=torch.tensor([k], dtype=torch.int64))
    else:
        k = _reshape_helper(g, k, g.op("Constant", value_t=torch.tensor([1])))
        if _try_get_scalar_type(k) != _type_utils.JitScalarType.INT64:
            k = g.op("Cast", k, to_i=_C_onnx.TensorProtoDataType.INT64)
    if g.opset <= 10:
        if not largest:
            _unimplemented("TopK", "Ascending is not supported")
        return g.op("TopK", input, k, axis_i=dim, outputs=2)
    else:
        return g.op(
            "TopK", input, k, axis_i=dim, largest_i=largest, sorted_i=sorted, outputs=2
        )


def _lt_helper(g: jit_utils.GraphContext, input, other):
    if g.opset <= 8:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset8 import lt as _lt8

        return _lt8(g, input, other)
    else:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import lt as _lt9

        return _lt9(g, input, other)


def _interpolate_warning(interpolate_mode):
    onnx_op = (
        "onnx:Resize" if GLOBALS.export_onnx_opset_version >= 10 else "onnx:Upsample"
    )
    warnings.warn(
        "You are trying to export the model with "
        + onnx_op
        + " for ONNX opset version "
        "" + str(GLOBALS.export_onnx_opset_version) + ". "
        "This operator might cause results to not match the expected results by PyTorch.\n"
        "ONNX's Upsample/Resize operator did not match Pytorch's Interpolation until opset 11. "
        "Attributes to determine how to transform the input were added in onnx:Resize in opset 11 "
        "to support Pytorch's behavior (like coordinate_transformation_mode and nearest_mode).\n"
        "We recommend using opset 11 and above for models using this operator.",
        stacklevel=2,
    )


def _unsqueeze_helper(g: jit_utils.GraphContext, input, axes_i):
    if len(axes_i) == 0:
        # unnecessary unsqueeze if axes length==0
        return input
    elif _is_constant(axes_i[0]):
        if g.opset >= 13:
            axes = g.op("Constant", value_t=torch.tensor(axes_i, dtype=torch.long))
            return g.op("Unsqueeze", input, axes)
        return g.op("Unsqueeze", input, axes_i=axes_i)
    # Tensor type
    if g.opset < 13:
        raise errors.SymbolicValueError(
            "Opset version must be >= 13 for Unsqueeze with dynamic axes.", input
        )
    return g.op("Unsqueeze", input, axes_i[0])


def _squeeze_helper(g: jit_utils.GraphContext, input, axes_i):
    if _is_constant(axes_i[0]):
        if g.opset >= 13:
            axes = g.op("Constant", value_t=torch.tensor(axes_i, dtype=torch.long))
            return g.op("Squeeze", input, axes)
        return g.op("Squeeze", input, axes_i=axes_i)
    # Tensor type
    if g.opset < 13:
        raise errors.SymbolicValueError(
            "Opset version must be >= 13 for Squeeze with dynamic axes.", input
        )
    axes_t = axes_i[0]
    axes_rank = _get_tensor_rank(axes_t)
    if axes_rank is None:
        raise AssertionError("Expected axes_rank to be non-None")
    if axes_rank > 1:
        raise errors.SymbolicValueError(
            "For Squeeze axses as input, the axes rank must be one in ONNX spec.", input
        )
    elif axes_rank == 0:
        # The axes is a scalar. Unsqueeze it to a rank 1 tensor.
        axes_t = _unsqueeze_helper(g, axes_t, [0])
        return g.op("Squeeze", input, axes_t)
    return g.op("Squeeze", input, axes_t)


def _reducesum_helper(
    g: jit_utils.GraphContext,
    input,
    axes_i=None,
    keepdims_i=1,
    noop_with_empty_axes_i=0,
):
    keepdims_i = _maybe_get_const(keepdims_i, "i")
    if g.opset >= 13:
        if axes_i:
            if not _is_value(axes_i):
                axes_i = g.op(
                    "Constant", value_t=torch.tensor(axes_i, dtype=torch.long)
                )
            return g.op(
                "ReduceSum",
                input,
                axes_i,
                keepdims_i=keepdims_i,
                noop_with_empty_axes_i=noop_with_empty_axes_i,
            )
        return g.op(
            "ReduceSum",
            input,
            keepdims_i=keepdims_i,
            noop_with_empty_axes_i=noop_with_empty_axes_i,
        )
    else:
        return g.op("ReduceSum", input, axes_i=axes_i, keepdims_i=keepdims_i)


def _interpolate_size_to_scales(g: jit_utils.GraphContext, input, output_size, dim):
    output_size = _maybe_get_const(output_size, "is")
    if _is_value(output_size):
        offset = 2
        offsets = g.op("Constant", value_t=torch.ones(offset, dtype=torch.float32))
        dividend = g.op("Cast", output_size, to_i=_C_onnx.TensorProtoDataType.FLOAT)
        divisor = _slice_helper(
            g, g.op("Shape", input), axes=[0], ends=[sys.maxsize], starts=[offset]
        )
        divisor = g.op("Cast", divisor, to_i=_C_onnx.TensorProtoDataType.FLOAT)
        scale_dims = g.op("Div", dividend, divisor)
        scales = g.op("Concat", offsets, scale_dims, axis_i=0)
    else:
        scales_constant = [
            1.0
            if i < 2
            else float(output_size[-(dim - i)])
            / float(input.type().sizes()[-(dim - i)])
            for i in range(dim)
        ]
        scales = g.op(
            "Constant", value_t=torch.tensor(scales_constant, dtype=torch.float32)
        )
    return scales


def _interpolate_get_scales_if_available(g: jit_utils.GraphContext, scales):
    available_scales = _maybe_get_const(scales[0], "fs") != -1 and not _is_none(
        scales[0]
    )

    if not available_scales:
        return None

    offsets = g.op("Constant", value_t=torch.ones(2, dtype=torch.float32))
    scales_list = g.op(
        "Constant", value_t=torch.tensor(_maybe_get_const(scales[0], "fs"))
    )
    scales = g.op("Concat", offsets, scales_list, axis_i=0)
    return scales


def _get_interpolate_attributes(g: jit_utils.GraphContext, mode, args):
    if mode == "nearest":
        align_corners = None
        scales = args[0:]
    else:
        align_corners = args[0]
        scales = args[1:]
    scales = _interpolate_get_scales_if_available(g, scales)
    return scales, align_corners


def _interpolate_get_scales(g: jit_utils.GraphContext, scale_factor, dim):
    offsets = g.op("Constant", value_t=torch.ones(2, dtype=torch.float32))
    scale_factor_rank = _get_tensor_rank(scale_factor)
    if isinstance(scale_factor.type(), _C.ListType) or (
        scale_factor_rank is not None and scale_factor_rank > 0
    ):
        return g.op("Concat", offsets, scale_factor, axis_i=0)
    else:
        scale_factor = _unsqueeze_helper(g, scale_factor, [0])
        scale_factor = g.op(
            "Cast", scale_factor, to_i=_C_onnx.TensorProtoDataType.FLOAT
        )
        scales = [scale_factor for i in range(dim - 2)]
    scale_factor = g.op("Concat", offsets, *scales, axis_i=0)
    return scale_factor


def _interpolate_get_scales_and_mode(
    g: jit_utils.GraphContext, input, size, scale_factor, mode, align_corners
):
    mode = _maybe_get_const(mode, "s")
    if "linear" in mode:
        mode = "linear"
    if "cubic" in mode:
        mode = "cubic"
    _interpolate_warning(mode)

    align_corners = _maybe_get_const(align_corners, "b")
    if isinstance(align_corners, bool) and align_corners:
        return _unimplemented("interpolate", "align_corners == True")

    if not input.type().dim():
        return _unimplemented("interpolate", "missing input shape")
    dim = input.type().dim()

    if not _is_none(scale_factor):
        scale_factor = _interpolate_get_scales(g, scale_factor, dim)
    elif not _is_none(size):
        if not _is_packed_list(size):
            is_scalar = _maybe_get_const(size, "t").dim() == 0
            if is_scalar:
                size = _unsqueeze_helper(g, size, [0])
                size = [size for i in range(dim - 2)]
                size = g.op("Concat", *size, axis_i=0)
        scale_factor = _interpolate_size_to_scales(g, input, size, dim)
    else:
        return _unimplemented(
            "interpolate", "Both size and scales are None in __interpolate"
        )
    return scale_factor, mode


def _argmin_argmax_helper(
    g: jit_utils.GraphContext,
    input: torch._C.Value,
    dim: torch._C.Value,
    keepdim: bool,
    op_name: str,
):
    def op_wrapper(input, axis_i, keepdims_i):
        if g.opset >= 12:
            return g.op(
                op_name,
                input,
                axis_i=axis_i,
                keepdims_i=keepdims_i,
                select_last_index_i=False,
            )
        return g.op(op_name, input, axis_i=axis_i, keepdims_i=keepdims_i)

    if _is_none(dim):
        flattened = _reshape_helper(
            g, input, g.op("Constant", value_t=torch.tensor([-1]))
        )
        output = op_wrapper(flattened, axis_i=0, keepdims_i=False)
        if keepdim:
            input_shape = g.op("Shape", input)
            input_shape_shape = g.op("Shape", input_shape)
            new_shape = g.op(
                "ConstantOfShape",
                input_shape_shape,
                value_t=torch.tensor([1], dtype=torch.int64),
            )
            output = g.op("Reshape", output, new_shape)
        return output

    dim = _parse_arg(dim, "i")
    return op_wrapper(input, axis_i=dim, keepdims_i=keepdim)


def _interpolate_helper(name, dim, interpolate_mode):
    @quantized_args(True, False, False)
    def symbolic_fn(g, input, output_size, *args):
        scales, align_corners = _get_interpolate_attributes(g, interpolate_mode, args)
        align_corners = _maybe_get_scalar(align_corners)
        coordinate_transformation_mode = (
            "asymmetric"
            if interpolate_mode == "nearest"
            else "align_corners"
            if align_corners
            else "half_pixel"
        )

        if scales is None:
            input_size = g.op("Shape", input)
            input_size_beg = _slice_helper(
                g, input_size, axes=[0], ends=[2], starts=[0]
            )
            output_size = g.op(
                "Cast", output_size, to_i=_C_onnx.TensorProtoDataType.INT64
            )
            output_size = g.op("Concat", input_size_beg, output_size, axis_i=0)

            if g.opset >= 13:
                empty_roi = _optional_input_placeholder_tensor(g)
                empty_scales = _optional_input_placeholder_tensor(g)
            else:
                empty_roi = g.op(
                    "Constant", value_t=torch.tensor([], dtype=torch.float32)
                )
                empty_scales = g.op(
                    "Constant", value_t=torch.tensor([], dtype=torch.float32)
                )

            return g.op(
                "Resize",
                input,
                empty_roi,
                empty_scales,
                output_size,
                coordinate_transformation_mode_s=coordinate_transformation_mode,
                cubic_coeff_a_f=-0.75,  # only valid when mode="cubic"
                mode_s=interpolate_mode,  # nearest, linear, or cubic
                nearest_mode_s="floor",
            )  # only valid when mode="nearest"
        else:
            if g.opset >= 13:
                empty_roi = _optional_input_placeholder_tensor(g)
            else:
                empty_roi = g.op(
                    "Constant", value_t=torch.tensor([], dtype=torch.float32)
                )

            return g.op(
                "Resize",
                input,
                empty_roi,
                scales,
                coordinate_transformation_mode_s=coordinate_transformation_mode,
                cubic_coeff_a_f=-0.75,  # only valid when mode="cubic"
                mode_s=interpolate_mode,  # nearest, linear, or cubic
                nearest_mode_s="floor",
            )  # only valid when mode="nearest"

    return symbolic_fn


def __interpolate_helper(
    g: jit_utils.GraphContext,
    input,
    size,
    scale_factor,
    mode,
    align_corners,
    recompute_scale_factor,
):
    mode = _maybe_get_const(mode, "s")
    if "linear" in mode:
        mode = "linear"
    if "cubic" in mode:
        mode = "cubic"
    align_corners = _maybe_get_const(align_corners, "b")
    align_corners = False if not isinstance(align_corners, bool) else align_corners
    coordinate_transformation_mode = (
        "asymmetric"
        if mode == "nearest"
        else "align_corners"
        if align_corners
        else "half_pixel"
    )

    if not _is_none(size):
        input_size = g.op("Shape", input)
        input_size = _slice_helper(g, input_size, axes=[0], ends=[2], starts=[0])
        # in some cases size is not a packed list but size is a scalar
        # We need to also verify that (_maybe_get_const(size, "t").dim() == 0)
        # but this information is not always available. Try to get the dim,
        # and if not assume that it is not a scalar.
        try:
            is_scalar = not _is_packed_list(size) and (
                _maybe_get_const(size, "t").dim() == 0
            )
        except AttributeError:
            is_scalar = not _is_packed_list(size)
            if not is_scalar:
                warnings.warn(
                    "Cannot verify if the output_size is a scalar "
                    "while exporting interpolate. Assuming that it is not a scalar.",
                    stacklevel=2,
                )

        if is_scalar:
            rank = _get_tensor_rank(input)
            if rank is None:
                return _unimplemented(
                    "interpolate (with a scalar output_size)",
                    "missing input shape (try giving an array of output_size values)",
                )
            size = _unsqueeze_helper(g, size, [0])
            size = [size for i in range(rank - 2)]
            size = g.op("Concat", *size, axis_i=0)
        size = g.op("Cast", size, to_i=_C_onnx.TensorProtoDataType.INT64)
        size = g.op("Concat", input_size, size, axis_i=0)

        if g.opset >= 13:
            empty_roi = _optional_input_placeholder_tensor(g)
            empty_scales = _optional_input_placeholder_tensor(g)
        else:
            empty_roi = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))
            empty_scales = g.op(
                "Constant", value_t=torch.tensor([], dtype=torch.float32)
            )

        return g.op(
            "Resize",
            input,
            empty_roi,
            empty_scales,
            size,
            coordinate_transformation_mode_s=coordinate_transformation_mode,
            cubic_coeff_a_f=-0.75,  # only valid when mode="cubic"
            mode_s=mode,  # nearest, linear, or cubic
            nearest_mode_s="floor",
        )
    else:  # if not _is_none(scales)
        rank = _get_tensor_rank(input)
        if rank is None:
            return _unimplemented("interpolate (with scales)", "missing input shape")

        if g.opset >= 13:
            empty_roi = _optional_input_placeholder_tensor(g)
        else:
            empty_roi = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))

        scales = _interpolate_get_scales(g, scale_factor, rank)
        return g.op(
            "Resize",
            input,
            empty_roi,
            scales,
            coordinate_transformation_mode_s=coordinate_transformation_mode,
            cubic_coeff_a_f=-0.75,  # only valid when mode="cubic"
            mode_s=mode,  # nearest, linear, or cubic
            nearest_mode_s="floor",
        )  # only valid when mode="nearest"


def _unbind_helper(g: jit_utils.GraphContext, self, dim, _outputs):
    if g.opset < 11:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import unbind
    elif g.opset <= 12:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset11 import (
            unbind,  # type: ignore[no-redef]
        )
    else:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset13 import (
            unbind,  # type: ignore[no-redef]
        )
    return unbind(g, self, dim, _outputs)


def _scatter_helper(g: jit_utils.GraphContext, self, dim, index, src):
    if g.opset <= 10:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import scatter
    else:
        # for mypy, scatter was imported two lines above
        from torch.onnx._internal.torchscript_exporter.symbolic_opset11 import (
            scatter,  # type: ignore[no-redef]
        )
    return scatter(g, self, dim, index, src)


def _repeat_interleave_split_helper(g: jit_utils.GraphContext, self, reps, dim):
    if g.opset <= 12:
        split_out = g.op("Split", self, split_i=[1] * reps, axis_i=dim, outputs=reps)
    else:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset13 import split

        repeats = g.op("Constant", value_t=torch.tensor([1] * reps))
        split_out = split(g, self, repeats, dim, _outputs=reps)
    return split_out if reps > 1 else [split_out]


def _repeat_interleave_single_value_repeat_helper(
    g: jit_utils.GraphContext, self, repeats, dim
):
    from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import (
        flatten,
        unsqueeze,
    )

    if not _is_tensor(repeats):
        repeats = g.op("Constant", value_t=torch.LongTensor(repeats))

    const_repeats: bool = _is_constant(repeats)
    reps = _maybe_get_const(repeats, "t")

    # Convert 'repeats' to 1-d if it is 0-d.
    if _get_tensor_rank(repeats) == 0:
        repeats = g.op("Reshape", repeats, g.op("Constant", value_t=torch.tensor([1])))

    # Create a new dim of size 1, then expand it to be 'repeats' long, and finally collapse it.
    unsqueezed = unsqueeze(g, self, dim + 1)

    # repeats_per_dim is 1 for all dims except for the new unsqueezed dim, where it has value 'repeats'.
    if const_repeats:
        # 'Repeats' is a constant, 'repeats_per_dim' can be a constant.
        onehot = torch.ones(_get_tensor_rank(unsqueezed), dtype=torch.int64)  # type: ignore[arg-type]
        onehot[dim + 1] = reps
        repeats_per_dim = g.op("Constant", value_t=onehot)
    else:
        # 'Repeats' is a variable, 'repeats_per_dim' cannot be a constant.
        onehot = g.op(
            "OneHot",
            unsqueeze(g, dim + 1, 0),  # indices, must be >= 1-dimensional
            g.op(
                "Constant", value_t=torch.tensor(_get_tensor_rank(unsqueezed))
            ),  # depth
            g.op(
                "Concat", g.op("Constant", value_t=torch.tensor([1])), repeats, axis_i=0
            ),  # on/off values
        )
        repeats_per_dim = flatten(g, onehot, 0, 1)

    tiled = g.op("Tile", unsqueezed, repeats_per_dim)
    return flatten(g, tiled, dim, dim + 1)


def _arange_cast_helper(
    g: jit_utils.GraphContext, end, start=None, step=None, dtype=None
) -> tuple[
    _type_utils.JitScalarType,
    _C.Value | None,
    _C.Value | None,
    _C.Value | None,
]:
    def _is_all_integral(scalars):
        for scalar in scalars:
            scalar_type = _type_utils.JitScalarType.from_value(
                scalar, _type_utils.JitScalarType.UNDEFINED
            )
            if (
                scalar_type != _type_utils.JitScalarType.INT64
                and scalar_type != _type_utils.JitScalarType.UNDEFINED
            ):
                return False
        return True

    # This logic is based on torch.arange docs. If "dtype" is provided,
    # infer input types from dtype. If not, then check if any of start, stop,
    # or step are floating point, and infer the type from get_default.
    # Otherwise, the dtype is inferred to be torch.int64.
    if dtype is None or (_is_value(dtype) and _is_none(dtype)):
        if _is_all_integral([start, end, step]):
            scalar_type = _type_utils.JitScalarType.INT64
        else:
            scalar_type = _type_utils.JitScalarType.from_dtype(
                torch.get_default_dtype()
            )
    else:
        if not isinstance(dtype, int):
            raise AssertionError(f"Expected dtype to be int, got {type(dtype)}")
        # TODO(justinchuby): Check if dtype is indeed an int.
        scalar_type = _type_utils.JitScalarType(dtype)

    start = g.op("Cast", start, to_i=scalar_type.onnx_type()) if start else None
    end = g.op("Cast", end, to_i=scalar_type.onnx_type()) if end else None
    step = g.op("Cast", step, to_i=scalar_type.onnx_type()) if step else None
    return scalar_type, end, start, step


def _arange_helper(g: jit_utils.GraphContext, *args):
    if g.opset <= 10:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import arange
    else:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset11 import (
            arange,  # type: ignore[no-redef]
        )
    return arange(g, *args)


def _size_helper(g: jit_utils.GraphContext, self, dim):
    full_shape = g.op("Shape", self)
    from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import select

    return select(g, full_shape, g.op("Constant", value_t=torch.tensor([0])), dim)


def _index_fill_reshape_helper(g: jit_utils.GraphContext, self, dim, index):
    # 1. reshape index => [1, ..., 1, dim, 1, ..., 1]
    # 2. expand index => [..., dim, ...], same shape as self except for dim.
    # 3. expand value as well.
    # 4. apply onnx::scatter.

    from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import expand

    if g.opset <= 10:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import scatter
    else:
        # for mypy, scatter was imported two lines above
        from torch.onnx._internal.torchscript_exporter.symbolic_opset11 import (
            scatter,  # type: ignore[no-redef]
        )

    if self.type().dim() is None:
        return _unimplemented("index_fill", "input rank not accessible")
    self_dim = self.type().dim()
    dim_value = _parse_arg(dim, "i")
    if dim_value < 0:
        dim_value += self_dim
    unsqueezed_index = _unsqueeze_helper(
        g, index, [i for i in range(self_dim) if i != dim_value]
    )
    expanded_index_shape = scatter(
        g, g.op("Shape", self), 0, _unsqueeze_helper(g, dim, [0]), g.op("Shape", index)
    )
    expanded_index = expand(g, unsqueezed_index, expanded_index_shape, None)
    return expanded_index_shape, expanded_index


# By default, when any value in the 'shape' input is equal to zero
# the corresponding dimension value is copied from the input tensor dynamically.
# allowzero=1 indicates that if any value in the 'shape' input is set to zero,
# the zero value is honored, similar to NumPy.
# allowzero=1 is only supported for opset version >= 14.
def _reshape_helper(g: jit_utils.GraphContext, input, shape, allowzero=0):
    shape = _maybe_get_const(shape, "is")
    if not _is_value(shape):
        shape = g.op("Constant", value_t=torch.LongTensor(shape))
    if g.opset <= 13:
        if allowzero == 1:
            _onnx_opset_unsupported(
                "Reshape with allowzero=1", GLOBALS.export_onnx_opset_version, 14, input
            )
        return g.op("Reshape", input, shape)
    else:
        return g.op("Reshape", input, shape, allowzero_i=allowzero)


def _batchnorm_helper(
    g: jit_utils.GraphContext, input, weight, bias, running_mean, running_var
):
    from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import _var_mean

    batch_size = _get_tensor_dim_size(input, 0)
    channel_size = _get_tensor_dim_size(input, 1)

    if weight is None or _is_none(weight):
        if channel_size is None:
            raise errors.SymbolicValueError(
                "Unsupported: ONNX export of batch_norm for unknown channel size.",
                input,
            )
        weight_value = torch.tensor(
            [1.0] * channel_size,
            dtype=_type_utils.JitScalarType.from_value(input).dtype(),
        )
        weight = g.op("Constant", value_t=weight_value)
    if bias is None or _is_none(bias):
        if channel_size is None:
            raise errors.SymbolicValueError(
                "Unsupported: ONNX export of batch_norm for unknown channel size.",
                input,
            )
        bias_value = torch.tensor(
            [0.0] * channel_size,
            dtype=_type_utils.JitScalarType.from_value(input).dtype(),
        )
        bias = g.op("Constant", value_t=bias_value)
    # If track_running_stats is set to False batch statistics are instead used during evaluation time
    if (
        running_mean is None
        or _is_none(running_mean)
        or running_var is None
        or _is_none(running_var)
    ):
        if batch_size is None or channel_size is None:
            raise AssertionError("batch_size and channel_size must be non-None")
        reshape_in = _reshape_helper(
            g,
            input,
            g.op(
                "Constant",
                value_t=torch.tensor([batch_size, channel_size, -1], dtype=torch.int64),
            ),
        )
        trans_in = g.op("Transpose", reshape_in, perm_i=[0, 2, 1])
        running_var, running_mean = _var_mean(
            g,
            trans_in,
            g.op("Constant", value_t=torch.tensor([0, 1], dtype=torch.int64)),
            False,
            False,
        )
    return weight, bias, running_mean, running_var


def _avgpool_helper(
    tuple_fn: Callable[[Any], Sequence[int]],
    padding: int | Sequence[int],
    kernel_size,
    stride,
    divisor_override,
    name,
) -> tuple[int, ...]:
    if divisor_override and divisor_override.node().kind() != "prim::Constant":
        _unimplemented(name, "divisor_override")
    return tuple(tuple_fn(padding))


def check_training_mode(op_train_mode: int, op_name: str) -> None:
    """Warns the user if the model's training mode and the export mode do not agree."""
    if GLOBALS.training_mode == _C_onnx.TrainingMode.PRESERVE:
        return

    if op_train_mode:
        op_mode_enum = _C_onnx.TrainingMode.TRAINING
    else:
        op_mode_enum = _C_onnx.TrainingMode.EVAL
    if op_mode_enum == GLOBALS.training_mode:
        # The modes agree. Do nothing
        return

    op_mode_text = f"train={bool(op_train_mode)}"
    # Setting the model mode could result in op_mode != GLOBALS.training_mode
    # if the model is a FuncModule. In this case we warn the user of
    # the state and export depending on op_mode
    # This is to support use-cases of fixing certain layer weights
    # in training.
    warnings.warn(
        f"ONNX export mode is set to {GLOBALS.training_mode}, but operator '{op_name}' "
        f"is set to {op_mode_text}. Exporting with {op_mode_text}.",
        stacklevel=2,
    )


def _flatten_helper(g: jit_utils.GraphContext, input, start_dim, end_dim, dim):
    input_size = g.op("Shape", input)
    slice1 = _slice_helper(g, input_size, axes=[0], starts=[0], ends=[start_dim])
    slices = [slice1, g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long))]
    if end_dim < dim - 1:
        slice3 = _slice_helper(
            g, input_size, axes=[0], starts=[end_dim + 1], ends=[dim]
        )
        slices = [
            slice1,
            g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long)),
            slice3,
        ]

    final_shape = g.op("Concat", *slices, axis_i=0)
    from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import (
        _reshape_from_tensor,
    )

    return _reshape_from_tensor(g, input, final_shape)


def _is_split_static(split_size_or_sizes, _outputs):
    if _outputs is None:
        return False
    if (
        _is_value(split_size_or_sizes)
        and split_size_or_sizes.node().kind() != "onnx::Constant"
    ):
        return False
    return True


def _optional_input_placeholder_tensor(g):
    n = g.op("prim::Constant")
    n.setType(_C.OptionalType.ofTensor())
    return n


def _handle_reduce_dim_none(g: jit_utils.GraphContext, self, op_name):
    rank = _get_tensor_rank(self)
    if rank is not None and any(
        _get_tensor_dim_size(self, i) == 0 for i in range(rank)
    ):
        # If input tensor is empty, according to ONNX ReduceSum definition,
        # set keepdims=1 so that the resulted tensor has the same rank as the input.
        return g.op(op_name, self, keepdims_i=1)
    return g.op(op_name, self, keepdims_i=0)


def dequantize_helper(
    g: jit_utils.GraphContext,
    qtensor: _C.Value,
    qdtype: _C_onnx.TensorProtoDataType | None = None,
) -> tuple[_C.Value, _C.Value, _C.Value, _C.Value | None]:
    """Appends to graph `g` ONNX nodes that dequantizes `qtensor` into `tensor`.

    Args:
        g: Graph, the ONNX IR graph that is under construction.
        qtensor: torch._C.Value, either a tuple of (quantized_tensor, scale, zero_point)
            for per tensor quantization, or
            (quantized_tensor, scale, zero_point, axis) for per channel quantization,
            representing the quantized tensor.
        qdtype: torch.onnx.TensorProtoDataType default None, if not None, represents the
            data type of quantized tensor. It must be either
            torch.onnx.TensorProtoDataType.UINT8 or torch.onnx.TensorProtoDataType.INT8.
    """
    unpacked_qtensors = _unpack_quantized_tensor(qtensor)
    tensor, scale, zero_point = unpacked_qtensors[:3]
    axis = unpacked_qtensors[3] if len(unpacked_qtensors) >= 4 else None
    axis_i = _get_const(axis, "i", "axis")
    input_qdtype = _type_utils.JitScalarType.from_value(tensor)
    if qdtype is None:
        if input_qdtype is not None:
            qdtype = input_qdtype.onnx_type()
        else:
            qdtype = _C_onnx.TensorProtoDataType.UINT8
    value = g.op("Cast", tensor, to_i=qdtype)
    scale = g.op("Cast", scale, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    zero_point = g.op("Cast", zero_point, to_i=qdtype)

    if axis_i is not None and GLOBALS.export_onnx_opset_version < 13:
        _onnx_opset_unsupported_detailed(
            "DequantizeLinear",
            GLOBALS.export_onnx_opset_version,
            13,
            "Attribute axis is not supported.",
            qtensor,
        )

    return (
        g.op("DequantizeLinear", value, scale, zero_point, axis_i=axis_i),
        scale,
        zero_point,
        axis,
    )


def quantize_helper(
    g: jit_utils.GraphContext,
    tensor: _C.Value,
    scale: _C.Value,
    zero_point: _C.Value,
    axis: _C.Value | None = None,
) -> _C.Value:
    """Appends to graph `g` ONNX nodes that quantizes `tensor` based on `scale`, `zero_point` and `axis`.

    Args:
        g: Graph, the ONNX IR graph that is under construction.
        tensor: torch._C.Value, representing the tensor to be quantized.
        scale: torch._C.Value, quantized scale.
        zero_point: torch._C.Value, quantized zero point.
        axis: Optional[torch._C.Value] default None, if None, represents per tensor quantization.
            Otherwise, represents per channel quantization, along given axis.

    Returns:
        A TupleConstruct storing information of the quantized tensor.
    """
    if (
        axis is not None
        and not _is_none(axis)
        and GLOBALS.export_onnx_opset_version < 13
    ):
        _onnx_opset_unsupported_detailed(
            "QuantizeLinear",
            GLOBALS.export_onnx_opset_version,
            13,
            "Attribute axis is not supported.",
            tensor,
        )

    if scale is None:
        raise AssertionError("scale must be non-None")
    if (
        _type_utils.JitScalarType.from_value(scale, _type_utils.JitScalarType.UNDEFINED)
        != _type_utils.JitScalarType.FLOAT
    ):
        scale = g.op("Cast", scale, to_i=_C_onnx.TensorProtoDataType.FLOAT)

    if zero_point is None:
        raise AssertionError("zero_point must be non-None")
    if _type_utils.JitScalarType.from_value(
        zero_point, _type_utils.JitScalarType.UNDEFINED
    ) not in {
        _type_utils.JitScalarType.UINT8,
        _type_utils.JitScalarType.INT8,
    }:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.UINT8)
    output = g.op(
        "QuantizeLinear",
        tensor,
        scale,
        zero_point,
        axis_i=_get_const(axis, "i", "axis"),
    )
    args = [output, scale, zero_point]
    if axis is not None and not _is_none(axis):
        args.append(axis)
    return g.op("prim::TupleConstruct", *args)


def requantize_bias_helper(
    g: jit_utils.GraphContext, bias, input_scale, weight_scale, axis=None
):
    """In PyTorch, bias is float and is quantized to int32 implicitly inside the quantized ATen op kernel.
    In ONNX we need to make the quantization explicit because operators expect all of their inputs to be quantized.
    Since int32 is not a supported output type by ONNX operator `QuantizeLinear`, quantization is exported using
    regular operators.
    """
    bias_scale = g.op("Mul", weight_scale, input_scale)
    bias_scale_shape = g.op("Shape", bias_scale)
    bias_zero_point = g.op(
        "ConstantOfShape", bias_scale_shape, value_t=torch.tensor([0], dtype=torch.int)
    )
    q_bias = g.op(
        "Cast", g.op("Div", bias, bias_scale), to_i=_C_onnx.TensorProtoDataType.INT32
    )
    axis_args = []
    if axis is not None and not _is_none(axis):
        axis_args.append(axis)
    return g.op("prim::TupleConstruct", q_bias, bias_scale, bias_zero_point, *axis_args)


def args_have_same_dtype(args):
    if not args:
        raise AssertionError("args must be non-empty")
    base_dtype = _type_utils.JitScalarType.from_value(args[0])
    has_same_dtype = all(
        _type_utils.JitScalarType.from_value(elem) == base_dtype for elem in args
    )
    return has_same_dtype


def _op_with_optional_float_cast(g: jit_utils.GraphContext, op_name, *args, **kwargs):
    """Some PyTorch operators (e.g., Clip/Min/ReLU/Pad) are super set of ONNX in terms of data types.
    This function maximizes the exportability of PyTorch-ONNX by allowing ONNX-unsupported PyTorch
    operator data type. For example, `Cast<int>(Clip<float>(Cast<float>(INPUT)))` can be used to mimic
    `Clip<int>(INPUT)` (opset version < 12).

    Args:
        g (torch._C.Graph): graph to write the ONNX representation into.
        op_name (str): operator name in ONNX.
        *args (tuple): operands to the operator.
        **kwargs (dict): attributes to the operator along with "opset_before" (optional, None by default)
            indicating the smallest opset version to trigger such casting behavior and "target_float_t"
            (optional, torch.onnx.JitScalarType.FLOAT by default) indicating the data type of internal operator.

    Returns:
        Optional[torch._C.Value, Tuple[torch._C.Value, ...]]: output(s) of the operator.
    """
    opset_before = kwargs.pop("opset_before", None)
    target_float_t = kwargs.pop("target_float_t", _type_utils.JitScalarType.FLOAT)

    inputs = list(args)
    dtype_0 = _type_utils.JitScalarType.from_value(inputs[0])

    require_cast = not _is_fp(inputs[0]) and (
        opset_before is None or GLOBALS.export_onnx_opset_version < opset_before
    )

    if require_cast:
        for input in inputs:
            if input.isCompleteTensor():
                input_scalar_type = _type_utils.JitScalarType.from_value(input)
                if input_scalar_type != dtype_0:
                    raise errors.SymbolicValueError(
                        f"Inputs of {op_name} must have same dtype."
                        f"Got {dtype_0.scalar_name()} and {input_scalar_type.scalar_name()}",
                        input,
                    )
        for i, input in enumerate(inputs):
            if input.isCompleteTensor() and not _is_fp(input):
                inputs[i] = g.op(
                    "Cast",
                    input,
                    to_i=target_float_t.onnx_type(),
                )

    self = g.op(op_name, *inputs, **kwargs)

    if require_cast:
        self = g.op("Cast", self, to_i=dtype_0.onnx_type())

    return self


def _maybe_cast_reduce_op_input(g: jit_utils.GraphContext, self):
    scalar_type = _type_utils.JitScalarType.from_value(
        self, _type_utils.JitScalarType.UNDEFINED
    )
    if scalar_type != _type_utils.JitScalarType.UNDEFINED:
        # This check only covers traced modules where dtype is present
        # pytorch reduce-ops cast all other integral types to int64
        if not _is_fp(self) and scalar_type != _type_utils.JitScalarType.INT64:
            self = g.op("Cast", self, to_i=_C_onnx.TensorProtoDataType.INT64)
    return self


def _apply_params(*args, **kwargs):
    """Returns a decorator that calls the decorated (higher-order) function with the given parameters."""

    def _apply(fn):
        return fn(*args, **kwargs)

    return _apply


def _reduce_op_symbolic_helper(onnx_op_name, allow_multi_dim_support=True):
    def symbolic(g, self, dim=None, keepdim=None):
        self = _maybe_cast_reduce_op_input(g, self)
        if dim is None or dim == ():
            # Dim can be 0, which will cause (not dim) == True. So we don't want to do
            # (not dim)
            # all-reduce path
            return _handle_reduce_dim_none(g, self, onnx_op_name)
        else:
            # dim-reduce path
            keepdim = _get_const(keepdim, "i", "keepdim")
            if g.opset < 18:
                desc = "is" if allow_multi_dim_support else "i"
                dim = _get_const(dim, desc, "dim")
                dim_list = dim if allow_multi_dim_support else [dim]
                return g.op(onnx_op_name, self, axes_i=dim_list, keepdims_i=keepdim)
            else:
                if _is_value(dim):
                    axes = dim
                else:
                    if allow_multi_dim_support:
                        axes = g.op(
                            "Constant", value_t=torch.tensor(dim, dtype=torch.long)
                        )
                    else:
                        axes = g.op(
                            "Constant", value_t=torch.tensor([dim], dtype=torch.long)
                        )
                return g.op(onnx_op_name, self, axes, keepdims_i=keepdim)

    return symbolic


def _overload_by_arg_count(fn):
    @functools.wraps(fn)
    def wrapper(g, *args):
        overloads = fn(g, *args)
        for overload in overloads:
            arg_descriptors = overload._arg_descriptors
            if len(arg_descriptors) == len(args):
                return overload(g, *args)
        return _unimplemented(f"aten::{fn.__name__}", f"with {len(args)} arguments")

    return wrapper


def _reduce_with_dtype_helper(
    onnx_op: str, name: str, allow_multi_dim_support: bool = True
):
    symbolic = _reduce_op_symbolic_helper(
        onnx_op, allow_multi_dim_support=allow_multi_dim_support
    )

    @_overload_by_arg_count
    def reduce(g, *args, **kwargs):
        @quantized_args(True)
        @parse_args("v", "none")
        def reduce_nodim(g, self, dtype):
            dtype_onnx = None
            if dtype.node().kind() == "onnx::Constant":
                dtype = _get_const(dtype, "i", "dtype")
                dtype_onnx = _type_utils.JitScalarType(dtype).onnx_type()
                self = g.op("Cast", self, to_i=dtype_onnx)
            elif dtype.node().kind() != "prim::Constant":
                return _unimplemented(name, "dtype", dtype)
            result = symbolic(g, self)
            if dtype_onnx is not None:
                result_dtype_onnx = _type_utils.JitScalarType.from_value(
                    result
                ).onnx_type()
                if result_dtype_onnx != dtype_onnx:
                    result = g.op("Cast", result, to_i=dtype_onnx)
            return result

        dim_desc = "is" if allow_multi_dim_support else "i"

        @quantized_args(True)
        @parse_args("v", dim_desc, "i", "none")  # type: ignore[arg-type]
        def reduce_dim(g, self, dim, keepdim, dtype):
            dtype_onnx = None
            if dtype.node().kind() == "onnx::Constant":
                dtype = _get_const(dtype, "i", "dtype")
                dtype_onnx = _type_utils.JitScalarType(dtype).onnx_type()
                self = g.op("Cast", self, to_i=dtype_onnx)
            elif dtype.node().kind() != "prim::Constant":
                return _unimplemented(name, "dtype", dtype)
            result = symbolic(g, self, dim, keepdim)
            if dtype_onnx is not None:
                result_dtype_onnx = _type_utils.JitScalarType.from_value(
                    result
                ).onnx_type()
                if result_dtype_onnx != dtype_onnx:
                    result = g.op("Cast", result, to_i=dtype_onnx)
            return result

        return reduce_nodim, reduce_dim

    return reduce


def _max_helper(g: jit_utils.GraphContext, self, dim_or_y=None, keepdim=None):
    # torch.max(input)
    if dim_or_y is None and keepdim is None:
        return g.op("ReduceMax", self, keepdims_i=0)
    # torch.max(input, other)
    if keepdim is None:
        return _op_with_optional_float_cast(g, "Max", self, dim_or_y, opset_before=12)
    # torch.max(input, dim, keepdim)
    else:
        keepdim = _get_const(keepdim, "i", "keepdim")
        dim = _get_const(dim_or_y, "i", "dim")
        if g.opset < 18:
            max = g.op("ReduceMax", self, axes_i=[dim], keepdims_i=keepdim)
        else:
            axes = g.op("Constant", value_t=torch.tensor([dim], dtype=torch.long))
            max = g.op("ReduceMax", self, axes, keepdims_i=keepdim)
        indices = g.op("ArgMax", self, axis_i=dim, keepdims_i=keepdim)
        return max, indices


def _min_helper(g: jit_utils.GraphContext, self, dim_or_y=None, keepdim=None):
    # torch.min(input)
    if dim_or_y is None and keepdim is None:
        return g.op("ReduceMin", self, keepdims_i=0)
    # torch.min(input, other)
    if keepdim is None:
        return _op_with_optional_float_cast(g, "Min", self, dim_or_y, opset_before=12)
    # torch.min(input, dim, keepdim)
    else:
        keepdim = _get_const(keepdim, "i", "keepdim")
        dim = _get_const(dim_or_y, "i", "dim")
        if g.opset < 18:
            min = g.op("ReduceMin", self, axes_i=[dim], keepdims_i=keepdim)
        else:
            axes = g.op("Constant", value_t=torch.tensor([dim], dtype=torch.long))
            min = g.op("ReduceMin", self, axes, keepdims_i=keepdim)
        indices = g.op("ArgMin", self, axis_i=dim, keepdims_i=keepdim)
        return min, indices


def _numel_helper(g: jit_utils.GraphContext, self):
    shape = g.op("Shape", self)
    return g.op("ReduceProd", shape, keepdims_i=0)


@parse_args("v", "is", "i", "i")
def _var_mean_helper(g: jit_utils.GraphContext, input, dim, correction, keepdim):
    if g.opset < 18:
        if dim is None:
            mean = g.op("ReduceMean", input, keepdims_i=0)
            t_mean = mean
            num_elements = _numel_helper(g, input)
        else:
            mean = g.op("ReduceMean", input, axes_i=dim, keepdims_i=keepdim)
            t_mean = g.op("ReduceMean", input, axes_i=dim, keepdims_i=1)
            redudced_dims = g.op("Shape", input)
            # dim could contain one or multiple dimensions
            redudced_dims = g.op(
                "Gather",
                redudced_dims,
                g.op("Constant", value_t=torch.tensor(dim)),
                axis_i=0,
            )
            num_elements = g.op("ReduceProd", redudced_dims, keepdims_i=0)
        sub_v = g.op("Sub", input, t_mean)
        sqr_sub = g.op("Mul", sub_v, sub_v)
        keepdim_mean = 0 if dim is None else keepdim
        var = g.op("ReduceMean", sqr_sub, axes_i=dim, keepdims_i=keepdim_mean)
        # Correct bias in calculating variance, by dividing it over (N - correction) instead on N
        if correction is None:
            correction = 1
        if correction != 0:
            num_elements = g.op(
                "Cast", num_elements, to_i=_C_onnx.TensorProtoDataType.FLOAT
            )
            one = g.op("Constant", value_t=torch.tensor(correction, dtype=torch.float))
            mul = g.op("Mul", var, num_elements)
            var = g.op("Div", mul, g.op("Sub", num_elements, one))
        return var, mean
    else:
        axes = None
        if dim is None:
            mean = g.op("ReduceMean", input, keepdims_i=0)
            t_mean = mean
            num_elements = _numel_helper(g, input)
        else:
            axes = g.op("Constant", value_t=torch.tensor(dim, dtype=torch.long))
            mean = g.op("ReduceMean", input, axes, keepdims_i=keepdim)
            t_mean = g.op("ReduceMean", input, axes, keepdims_i=1)
            redudced_dims = g.op("Shape", input)
            # dim could contain one or multiple dimensions
            redudced_dims = g.op(
                "Gather",
                redudced_dims,
                g.op("Constant", value_t=torch.tensor(dim)),
                axis_i=0,
            )
            num_elements = g.op("ReduceProd", redudced_dims, keepdims_i=0)
        sub_v = g.op("Sub", input, t_mean)
        sqr_sub = g.op("Mul", sub_v, sub_v)
        keepdim_mean = 0 if dim is None else keepdim
        if axes is None:
            var = g.op("ReduceMean", sqr_sub, keepdims_i=keepdim_mean)
        else:
            var = g.op("ReduceMean", sqr_sub, axes, keepdims_i=keepdim_mean)
        # Correct bias in calculating variance, by dividing it over (N - correction) instead on N
        if correction is None:
            correction = 1
        if correction != 0:
            num_elements = g.op(
                "Cast", num_elements, to_i=_C_onnx.TensorProtoDataType.FLOAT
            )
            one = g.op("Constant", value_t=torch.tensor(correction, dtype=torch.float))
            mul = g.op("Mul", var, num_elements)
            var = g.op("Div", mul, g.op("Sub", num_elements, one))
        return var, mean


def _embedding_bag_helper(
    g: jit_utils.GraphContext,
    embedding_matrix,
    indices,
    offsets,
    scale_grad_by_freq,
    mode,
    sparse,
    per_sample_weights,
    include_last_offset,
    padding_idx,
):
    if scale_grad_by_freq and GLOBALS.export_training:
        return _onnx_unsupported(
            "embedding_bag with scale_grad_by_freq for training mode"
        )
    if padding_idx is not None and padding_idx >= 0:
        raise RuntimeError("embedding_bag with padding_idx")

    loop_condition = g.op("Constant", value_t=torch.tensor(1))
    loop_condition = g.op("Cast", loop_condition, to_i=_C_onnx.TensorProtoDataType.BOOL)
    zero = g.op("Constant", value_t=torch.tensor([0]))

    indices_len = _unsqueeze_helper(
        g,
        _size_helper(g, indices, g.op("Constant", value_t=torch.tensor(0))),
        [0],
    )
    if not include_last_offset:
        offsets = [offsets, indices_len]
        offsets = g.op("Concat", *offsets, axis_i=0)

    # Offsets holds the starting index position of each bag. So we create a list of the indices slices (determined by
    # offsets) and gather those indices in indices_row. Then we use this subset of indices to gather from embeddings.
    # The embeddings output is a loop scan output, so we can avoid creating a sequence and inserting elements in.
    offsets_starts = _slice_helper(
        g, offsets, axes=[0], starts=[0], ends=[sys.maxsize], steps=[1]
    )
    offsets_ends = _slice_helper(
        g, offsets, axes=[0], starts=[1], ends=[sys.maxsize], steps=[1]
    )

    loop_len = _size_helper(g, offsets_ends, g.op("Constant", value_t=torch.tensor(0)))

    loop, (loop_context,), _ = jit_utils.add_op_with_blocks(
        g, "Loop", loop_len, loop_condition, n_blocks=1
    )
    loop_block = loop_context.block

    # FIXME(justinchuby): We need to handle what happens when we call b.op on a node return
    block_input_iter = utils._add_input_to_block(loop_block)
    utils._add_input_to_block(loop_block)

    indices_start = loop_context.op(
        "Gather", offsets_starts, block_input_iter, axis_i=0
    )
    indices_end = loop_context.op("Gather", offsets_ends, block_input_iter, axis_i=0)
    indices_start = _unsqueeze_helper(loop_context, indices_start, [0])
    indices_end = _unsqueeze_helper(loop_context, indices_end, [0])

    indices_row = loop_context.op("Slice", indices, indices_start, indices_end, zero)
    embeddings = loop_context.op("Gather", embedding_matrix, indices_row, axis_i=0)
    if not _is_none(per_sample_weights):
        per_sample_weights_row = loop_context.op(
            "Slice", per_sample_weights, indices_start, indices_end, zero
        )
        per_sample_weights_row = _unsqueeze_helper(
            loop_context, per_sample_weights_row, [1]
        )
        embeddings = loop_context.op("Mul", embeddings, per_sample_weights_row)
    if mode == 0:
        embeddings = _reducesum_helper(
            loop_context, embeddings, axes_i=[0], keepdims_i=0
        )
    elif mode == 1:
        if loop_context.opset < 18:
            embeddings = loop_context.op(
                "ReduceMean", embeddings, axes_i=[0], keepdims_i=0
            )
        else:
            axes = loop_context.op(
                "Constant", value_t=torch.tensor([0], dtype=torch.long)
            )
            embeddings = loop_context.op("ReduceMean", embeddings, axes, keepdims_i=0)
    else:
        if loop_context.opset < 18:
            embeddings = loop_context.op(
                "ReduceMax", embeddings, axes_i=[0], keepdims_i=0
            )
        else:
            axes = loop_context.op(
                "Constant", value_t=torch.tensor([0], dtype=torch.long)
            )
            embeddings = loop_context.op("ReduceMax", embeddings, axes, keepdims_i=0)

    cond_out = loop_context.op(
        "Cast", loop_condition, to_i=_C_onnx.TensorProtoDataType.BOOL
    )
    utils._add_output_to_block(loop_block, cond_out)
    utils._add_output_to_block(loop_block, embeddings)

    # aten::embedding_bag returns a tuple of 4 elements: output, offset2bag, bag_size, max_indices.
    # But the last three outputs are not used in torch.nn.EmbeddingBag or torch.nn.functional.embedding_bag.
    return loop.node().output(), None, None, None


def _linalg_vector_norm_helper(
    g: jit_utils.GraphContext,
    self: torch._C.Value,
    ord: float,
    dim: Sequence[int] | None,
    keepdim: bool,
    dtype: torch._C.Value,
):
    axes = None
    # Conditions based on https://pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html
    if _is_none(dim):
        self = _reshape_helper(g, self, [-1])
        keepdim = False
    elif g.opset >= 18:
        axes = g.op("Constant", value_t=torch.tensor(dim, dtype=torch.long))

    if ord == math.inf:
        if g.opset < 18:
            result = g.op(
                "ReduceMax", g.op("Abs", self), axes_i=dim, keepdims_i=keepdim
            )
        else:
            if axes is None:
                result = g.op("ReduceMax", g.op("Abs", self), keepdims_i=keepdim)
            else:
                result = g.op("ReduceMax", g.op("Abs", self), axes, keepdims_i=keepdim)
    elif ord == -math.inf:
        if g.opset < 18:
            result = g.op(
                "ReduceMin", g.op("Abs", self), axes_i=dim, keepdims_i=keepdim
            )
        else:
            if axes is None:
                result = g.op("ReduceMin", g.op("Abs", self), keepdims_i=keepdim)
            else:
                result = g.op("ReduceMin", g.op("Abs", self), axes, keepdims_i=keepdim)
    elif ord == 0:
        if g.opset < 11:
            return _onnx_opset_unsupported_detailed(
                "linalg_vector_norm", 9, 11, "ord=0 not supported", self
            )
        else:
            if dim is None:
                self = _reshape_helper(
                    g,
                    self,
                    g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64)),
                )
                keepdim = False

            cond_op = g.op(
                "Not",
                g.op("Equal", self, g.op("Constant", value_t=torch.LongTensor([0]))),
            )
            cond_op = g.op(
                "Cast",
                cond_op,
                to_i=_type_utils.JitScalarType.from_value(self).onnx_type(),
            )
            return _reducesum_helper(g, cond_op, axes_i=dim, keepdims_i=keepdim)
    elif ord == 1:
        if g.opset < 18:
            result = _reduce_op_symbolic_helper("ReduceL1")(
                g, self, dim=dim, keepdim=keepdim
            )
        else:
            if axes is None:
                result = _reduce_op_symbolic_helper("ReduceL1")(
                    g, self, keepdim=keepdim
                )
            else:
                result = _reduce_op_symbolic_helper("ReduceL1")(
                    g, self, axes, keepdim=keepdim
                )
    elif ord == 2:
        if g.opset < 18:
            result = _reduce_op_symbolic_helper("ReduceL2")(
                g, self, dim=dim, keepdim=keepdim
            )
        else:
            if axes is None:
                result = _reduce_op_symbolic_helper("ReduceL2")(
                    g, self, keepdim=keepdim
                )
            else:
                result = _reduce_op_symbolic_helper("ReduceL2")(
                    g, self, axes, keepdim=keepdim
                )
    else:
        ord_op = g.op("Constant", value_t=torch.tensor(ord, dtype=torch.float32))
        result = _reducesum_helper(
            g, g.op("Pow", g.op("Abs", self), ord_op), axes_i=dim, keepdims_i=keepdim
        )
        result = g.op(
            "Pow",
            result,
            g.op(
                "Div",
                g.op("Constant", value_t=torch.tensor(1, dtype=torch.float32)),
                ord_op,
            ),
        )

    if not _is_none(dtype):
        dtype = _get_const(dtype, "i", "dtype")
        result = g.op("Cast", result, to_i=_type_utils.JitScalarType(dtype).onnx_type())  # type: ignore[arg-type]
    return result


# Deprecated. Internally use _type_utils.ScalarType
# TODO: remove these once we support Type's in the JIT IR and we can once again
# use the unified toType operator
cast_pytorch_to_onnx = {
    "Byte": _C_onnx.TensorProtoDataType.UINT8,
    "Char": _C_onnx.TensorProtoDataType.INT8,
    "Double": _C_onnx.TensorProtoDataType.DOUBLE,
    "Float": _C_onnx.TensorProtoDataType.FLOAT,
    "Half": _C_onnx.TensorProtoDataType.FLOAT16,
    "Int": _C_onnx.TensorProtoDataType.INT32,
    "Long": _C_onnx.TensorProtoDataType.INT64,
    "Short": _C_onnx.TensorProtoDataType.INT16,
    "Bool": _C_onnx.TensorProtoDataType.BOOL,
    "ComplexFloat": _C_onnx.TensorProtoDataType.COMPLEX64,
    "ComplexDouble": _C_onnx.TensorProtoDataType.COMPLEX128,
    "BFloat16": _C_onnx.TensorProtoDataType.BFLOAT16,
    "Undefined": _C_onnx.TensorProtoDataType.UNDEFINED,
}

# Deprecated. Internally use _type_utils.ScalarType
scalar_name_to_pytorch = {
    "uint8_t": "Byte",
    "int8_t": "Char",
    "double": "Double",
    "float": "Float",
    "half": "Half",
    "int": "Int",
    "int64_t": "Long",
    "int16_t": "Short",
    "bool": "Bool",
    "complex64": "ComplexFloat",
    "complex128": "ComplexDouble",
    "qint8": "QInt8",
    "quint8": "QUInt8",
    "qint32": "QInt32",
    "bfloat16": "BFloat16",
}


# Deprecated. Internally use _type_utils.ScalarType
# This indicates each scalar type's corresponding
# torch type. Related source:
# https://github.com/pytorch/pytorch/blob/344defc9733a45fee8d0c4d3f5530f631e823196/c10/core/ScalarType.h
scalar_type_to_pytorch_type = [
    torch.uint8,  # 0
    torch.int8,  # 1
    torch.short,  # 2
    torch.int,  # 3
    torch.int64,  # 4
    torch.half,  # 5
    torch.float,  # 6
    torch.double,  # 7
    torch.complex32,  # 8
    torch.complex64,  # 9
    torch.complex128,  # 10
    torch.bool,  # 11
    torch.qint8,  # 12
    torch.quint8,  # 13
    torch.qint32,  # 14
    torch.bfloat16,  # 15
]

# Deprecated. Internally use _type_utils.ScalarType
# source of truth is
# https://github.com/pytorch/pytorch/blob/master/torch/csrc/utils/tensor_dtypes.cpp
pytorch_name_to_type = {
    "Byte": torch.uint8,
    "Char": torch.int8,
    "Double": torch.double,
    "Float": torch.float,
    "Half": torch.half,
    "Int": torch.int,
    "Long": torch.int64,
    "Short": torch.short,
    "Bool": torch.bool,
    "ComplexFloat": torch.complex64,
    "ComplexDouble": torch.complex128,
    "QInt8": torch.qint8,
    "QUInt8": torch.quint8,
    "QInt32": torch.qint32,
    "BFloat16": torch.bfloat16,
}


# Deprecated. Internally use _type_utils.ScalarType
scalar_type_to_onnx = [
    cast_pytorch_to_onnx["Byte"],  # 0
    cast_pytorch_to_onnx["Char"],  # 1
    cast_pytorch_to_onnx["Short"],  # 2
    cast_pytorch_to_onnx["Int"],  # 3
    cast_pytorch_to_onnx["Long"],  # 4
    cast_pytorch_to_onnx["Half"],  # 5
    cast_pytorch_to_onnx["Float"],  # 6
    cast_pytorch_to_onnx["Double"],  # 7
    cast_pytorch_to_onnx["Undefined"],  # 8
    cast_pytorch_to_onnx["ComplexFloat"],  # 9
    cast_pytorch_to_onnx["ComplexDouble"],  # 10
    cast_pytorch_to_onnx["Bool"],  # 11
    cast_pytorch_to_onnx["Char"],  # 12
    cast_pytorch_to_onnx["Byte"],  # 13
    cast_pytorch_to_onnx["Int"],  # 14
    cast_pytorch_to_onnx["BFloat16"],  # 15
]

# Global set to store the list of quantized operators in the network.
# This is currently only used in the conversion of quantized ops from PT -> C2 via ONNX.
_quantized_ops: set[int] = set()
<|endoftext|># mypy: allow-untyped-defs
# mypy: disable-error-code=arg-type
"""This file exports ONNX ops for opset 14.

Note [ONNX operators that are added/updated in opset 14]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
New operators:
    HardSwish, Trilu

Updated operators:
    Reshape
    Add, Sub, Mul, Div
    GRU, LSTM, RNN
    BatchNorm, Cumsum, Relu
"""

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in README.md
from __future__ import annotations

import functools

import torch
from torch.onnx import _constants
from torch.onnx._internal.torchscript_exporter import (
    _type_utils,
    jit_utils,
    registration,
    symbolic_helper,
)
from torch.onnx._internal.torchscript_exporter._globals import GLOBALS


__all__ = [
    "hardswish",
    "tril",
    "triu",
    "reshape",
    "batch_norm",
    "quantized_hardswish",
    "scaled_dot_product_attention",
]

_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=14)


@_onnx_symbolic("aten::hardswish")
@symbolic_helper.parse_args("v")
def hardswish(g: jit_utils.GraphContext, self):
    return g.op("HardSwish", self)


@_onnx_symbolic("aten::tril")
def tril(g: jit_utils.GraphContext, self, diagonal, out=None):
    return g.op("Trilu", self, diagonal, upper_i=0)


@_onnx_symbolic("aten::triu")
def triu(g: jit_utils.GraphContext, self, diagonal, out=None):
    return g.op("Trilu", self, diagonal, upper_i=1)


@_onnx_symbolic("aten::reshape")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "v")
def reshape(g: jit_utils.GraphContext, self, shape):
    # NOTE: Due to bug in ORT https://github.com/microsoft/onnxruntime/issues/10664
    #       Reshape export cannot utilize the new allowzero attribute introduced in opset 14.
    return symbolic_helper._reshape_helper(g, self, shape, allowzero=0)


@_onnx_symbolic("aten::batch_norm")
@symbolic_helper.parse_args("v", "v", "v", "v", "v", "i", "f", "f", "i")
def batch_norm(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    running_mean,
    running_var,
    training,
    momentum,
    eps,
    cudnn_enabled,
):
    if (
        torch.is_autocast_enabled()
        and not symbolic_helper.args_have_same_dtype(
            [input, weight, bias, running_mean, running_var]
        )
        and GLOBALS.export_onnx_opset_version < 15
    ):
        return symbolic_helper._onnx_opset_unsupported_detailed(
            "BatchNormalization",
            14,
            15,
            "All input tensors must have the same `dtype`."
            " Turn off Autocast or export using opset version 15.",
            input,
        )

    symbolic_helper.check_training_mode(training, "batch_norm")
    weight, bias, running_mean, running_var = symbolic_helper._batchnorm_helper(
        g, input, weight, bias, running_mean, running_var
    )
    out = g.op(
        "BatchNormalization",
        input,
        weight,
        bias,
        running_mean,
        running_var,
        epsilon_f=eps,
        momentum_f=1 - momentum,
        training_mode_i=0 if not training else 1,
        outputs=1 if not training else 3,
    )
    if not training:
        return out
    else:
        res, new_running_mean, new_running_var = out
        new_running_mean.setType(running_mean.type())
        new_running_var.setType(running_var.type())
        return res


@_onnx_symbolic("quantized::hardswish")
def quantized_hardswish(g: jit_utils.GraphContext, x, op_scale, op_zero_point):
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)

    output = hardswish(g, x)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


# Ported from
# https://github.com/microsoft/onnxscript/blob/6b1b81700b4523f31d8c6d3321e5d8ef5d42b764/onnxscript/function_libs/torch_aten/ops/nn.py#L1504
# aten_scaled_dot_product_attention
# NOTE: Need op.Trilu
@_onnx_symbolic("aten::scaled_dot_product_attention")
@symbolic_helper.parse_args("v", "v", "v", "v", "f", "b", "v", "b")
def scaled_dot_product_attention(
    g: jit_utils.GraphContext,
    query: torch._C.Value,
    key: torch._C.Value,
    value: torch._C.Value,
    attn_mask: torch._C.Value | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: torch._C.Value | None = None,
    enable_gqa: bool = False,
):
    if is_causal and not symbolic_helper._is_none(attn_mask):
        raise AssertionError("is_causal and attn_mask cannot be set at the same time")
    if enable_gqa:
        raise AssertionError(
            "conversion of scaled_dot_product_attention not implemented if enable_gqa is True"
        )

    if symbolic_helper._is_none(scale):
        scale = _attention_scale(g, query)

    if is_causal:
        attn_mask = _causal_attention_mask(g, query, key)

    # Swap the last two axes of key
    # NOTE: onnx-script has different logic here, because the attribute perms in
    # transpose needs list of ints
    key_shape_builtin = symbolic_helper._get_tensor_rank(key)
    # pyrefly: ignore [bad-argument-type, no-matching-overload]
    key_transposed_axes = list(range(key_shape_builtin))
    key_transposed_axes[-1], key_transposed_axes[-2] = (
        key_transposed_axes[-2],
        key_transposed_axes[-1],
    )
    key_transposed = g.op("Transpose", key, perm_i=key_transposed_axes)

    # https://github.com/pytorch/pytorch/blob/12da0c70378b5be9135c6fda62a9863bce4a4818/aten/src/ATen/native/transformers/attention.cpp#L653
    # Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96 for math
    # pyrefly: ignore [bad-argument-type]
    query_scaled = g.op("Mul", query, g.op("Sqrt", scale))
    # pyrefly: ignore [bad-argument-type]
    key_transposed_scaled = g.op("Mul", key_transposed, g.op("Sqrt", scale))
    mul_qk = g.op("MatMul", query_scaled, key_transposed_scaled)

    if symbolic_helper._is_none(attn_mask):
        mul_qk_add = mul_qk
        attn_weight = g.op("Softmax", mul_qk_add, axis_i=-1)
    elif (
        _type_utils.JitScalarType.from_value(attn_mask)
        == _type_utils.JitScalarType.BOOL
    ):
        # Turn the Boolean mask to float: attn_mask.masked_fill(not attn_mask, -float('inf'))
        const_zero = g.op("Constant", value_t=torch.tensor([0.0]))
        const_neg_inf = g.op("Constant", value_t=torch.tensor([-float("inf")]))
        # pyrefly: ignore [bad-argument-type]
        attn_mask = g.op("Where", attn_mask, const_zero, const_neg_inf)
        mul_qk_add = g.op("Add", mul_qk, attn_mask)
        attn_weight = g.op("Softmax", mul_qk_add, axis_i=-1)
        # When using scaled dot product attention with a boolean mask, the softmax operation might return NaN values
        # due to the presence of -inf in an entire row (padding tokens), resulting in 0/0 (NaN) in the softmax output.
        # This is because there's no safe softmax imp in ONNX, so we need to handle NaN values explicitly to match
        # the behavior of PyTorch with boolean masks.
        attn_weight = g.op("Where", g.op("IsNaN", attn_weight), const_zero, attn_weight)
    elif _type_utils.JitScalarType.from_value(attn_mask) in (
        _type_utils.JitScalarType.FLOAT,
        _type_utils.JitScalarType.HALF,
        _type_utils.JitScalarType.BFLOAT16,
    ):
        # pyrefly: ignore [bad-argument-type]
        mul_qk_add = g.op("Add", mul_qk, attn_mask)
        attn_weight = g.op("Softmax", mul_qk_add, axis_i=-1)
    else:
        raise ValueError(
            f"Unsupported type for attn_mask: {_type_utils.JitScalarType.from_value(attn_mask)}"
        )

    if dropout_p != 0:
        attn_weight = g.op(
            "Dropout",
            attn_weight,
            g.op("Constant", value_t=torch.tensor(dropout_p, dtype=torch.float)),
        )

    return g.op("MatMul", attn_weight, value)


def _attention_scale(
    g: jit_utils.GraphContext, query: torch._C.Value
) -> torch._C.Value:
    """Calculate the scale factor for the attention result.

    Args:
        query: Tensor of shape [..., L, E]

    Returns:
        Scalar scale factor := 1 / math.sqrt(query.size(-1))
    """
    query_shape = g.op("Shape", query)
    query_shape_last = g.op(
        "Slice",
        query_shape,
        g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64)),
        g.op(
            "Constant", value_t=torch.tensor([_constants.INT64_MAX], dtype=torch.int64)
        ),
    )
    embedding_size = g.op(
        "Cast",
        query_shape_last,
        to_i=_type_utils.JitScalarType.from_value(query).onnx_type(),
    )
    const_one = g.op("Constant", value_t=torch.tensor([1.0], dtype=torch.float))
    scale = g.op("Div", const_one, g.op("Sqrt", embedding_size))
    # Add a Cast to convert the scale back to original type
    scale = g.op(
        "Cast",
        scale,
        to_i=_type_utils.JitScalarType.from_value(query).onnx_type(),
    )
    return scale


def _causal_attention_mask(
    g: jit_utils.GraphContext, query: torch._C.Value, key: torch._C.Value
) -> torch._C.Value:
    """Create a causal mask for the given query and key tensors.

    Equivalent to::
        mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_mask = torch.zeros(L, S, dtype=torch.float)
        attn_mask = attn_mask.masked_fill(not mask, -float("inf"))

    Args:
        query: Tensor of shape [..., L, E]
        key: Tensor of shape [..., S, E]

    Returns:
        Tensor of shape [L, S]
    """

    query_shape = g.op("Shape", query)
    key_shape = g.op("Shape", key)

    last_idx = g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
    second_last_idx = g.op("Constant", value_t=torch.tensor([-2], dtype=torch.int64))
    target_length = g.op("Slice", query_shape, second_last_idx, last_idx)
    source_length = g.op("Slice", key_shape, second_last_idx, last_idx)
    # attn_mask = torch.ones(L, S) := {
    size = g.op("Concat", target_length, source_length, axis_i=0)
    const_one = g.op("Constant", value_t=torch.tensor([1.0]))
    attn_mask = g.op("Expand", const_one, size)
    # }
    attn_mask = g.op("Trilu", attn_mask, upper_i=0)
    # The causal mask has 0s in the lower triangle and -inf in the upper triangle.
    const_zero = g.op("Constant", value_t=torch.tensor([0.0]))
    const_neg_inf = g.op("Constant", value_t=torch.tensor([-float("inf")]))
    attn_mask = g.op(
        "Where", g.op("Equal", attn_mask, const_zero), const_neg_inf, const_zero
    )
    return attn_mask
<|endoftext|>"""Globals used internally by the ONNX exporter.

Do not use this module outside of `torch.onnx` and its tests.

Be very judicious when adding any new global variables. Do not create new global
variables unless they are absolutely necessary.
"""

import torch._C._onnx as _C_onnx

# This module should only depend on _constants and nothing else in torch.onnx to keep
# dependency direction clean.
from torch.onnx import _constants


class _InternalGlobals:
    """Globals used internally by ONNX exporter.

    NOTE: Be very judicious when adding any new variables. Do not create new
    global variables unless they are absolutely necessary.
    """

    def __init__(self) -> None:
        self._export_onnx_opset_version = _constants.ONNX_DEFAULT_OPSET
        self._training_mode: _C_onnx.TrainingMode = _C_onnx.TrainingMode.EVAL
        self._in_onnx_export: bool = False
        # Whether the user's model is training during export
        self.export_training: bool = False
        self.operator_export_type: _C_onnx.OperatorExportTypes = (
            _C_onnx.OperatorExportTypes.ONNX
        )
        self.onnx_shape_inference: bool = True
        self._autograd_inlining: bool = True

    @property
    def training_mode(self) -> _C_onnx.TrainingMode:
        """The training mode for the exporter."""
        return self._training_mode

    @training_mode.setter
    def training_mode(self, training_mode: _C_onnx.TrainingMode) -> None:
        if not isinstance(training_mode, _C_onnx.TrainingMode):
            raise TypeError(
                "training_mode must be of type 'torch.onnx.TrainingMode'. This is "
                "likely a bug in torch.onnx."
            )
        self._training_mode = training_mode

    @property
    def export_onnx_opset_version(self) -> int:
        """Opset version used during export."""
        return self._export_onnx_opset_version

    @export_onnx_opset_version.setter
    def export_onnx_opset_version(self, value: int) -> None:
        self._export_onnx_opset_version = value

    @property
    def in_onnx_export(self) -> bool:
        """Whether it is in the middle of ONNX export."""
        return self._in_onnx_export

    @in_onnx_export.setter
    def in_onnx_export(self, value: bool) -> None:
        if type(value) is not bool:
            raise TypeError("in_onnx_export must be a boolean")
        self._in_onnx_export = value

    @property
    def autograd_inlining(self) -> bool:
        """Whether Autograd must be inlined."""
        return self._autograd_inlining

    @autograd_inlining.setter
    def autograd_inlining(self, value: bool) -> None:
        if type(value) is not bool:
            raise TypeError("autograd_inlining must be a boolean")
        self._autograd_inlining = value


GLOBALS = _InternalGlobals()
<|endoftext|># mypy: allow-untyped-defs
"""
Note [ONNX operators that are added/updated from opset 7 to opset 8]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
New operators:
  Expand

Updated operators:
  Min, Max, Sum, Mean: supports multidirectional broadcasting.
  MaxPool: added optional indices output.
  Scan
"""

import functools
import warnings

from torch.onnx._internal.torchscript_exporter import (
    jit_utils,
    registration,
    symbolic_helper,
    symbolic_opset9 as opset9,
)


_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=7)

block_listed_operators = (
    "scan",
    "expand",
    "expand_as",
    "meshgrid",
    "adaptive_max_pool1d",
    "adaptive_max_pool2d",
    "adaptive_max_pool3d",
    "max_pool1d_with_indices",
    "max_pool2d_with_indices",
    "max_pool3d_with_indices",
)


# NOTE: max, min, sum, mean: broadcasting is not supported in opset 7.
# torch.max (same for torch.min) actually has two interfaces smashed together:
# torch.max(x, dim, keepdim) and torch.max(x, y)
@_onnx_symbolic("aten::max")
def max(g: jit_utils.GraphContext, self, dim_or_y=None, keepdim=None):
    # torch.max(input, other)
    if keepdim is None and dim_or_y is not None:
        warnings.warn(
            "Multidirectional broadcasting is not supported in opset 7. "
            "This might cause the onnx model to be incorrect, if inputs to max operators "
            "have different shapes",
            stacklevel=2,
        )
    return opset9.max(g, self, dim_or_y, keepdim)


@_onnx_symbolic("aten::min")
def min(g: jit_utils.GraphContext, self, dim_or_y=None, keepdim=None):
    # torch.min(input, other)
    if keepdim is None and dim_or_y is not None:
        warnings.warn(
            "Multidirectional broadcasting is not supported in opset 7. "
            "This might cause the onnx model to be incorrect, if inputs to min operators "
            "have different shapes",
            stacklevel=2,
        )
    return opset9.min(g, self, dim_or_y, keepdim)


for block_listed_op in block_listed_operators:
    _onnx_symbolic(f"aten::{block_listed_op}")(
        symbolic_helper._block_list_in_opset(block_listed_op)
    )
<|endoftext|># mypy: allow-untyped-defs
# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in README.md

# This file exports ONNX ops for opset 13
import functools

import torch
import torch._C._onnx as _C_onnx
from torch.onnx import _constants, errors
from torch.onnx._internal.torchscript_exporter import (
    _type_utils,
    jit_utils,
    registration,
    symbolic_helper,
    symbolic_opset11 as opset11,
    symbolic_opset9 as opset9,
    utils,
)


_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=13)


@_onnx_symbolic("aten::softmax")
@symbolic_helper.parse_args("v", "i", "none")
def softmax(g: jit_utils.GraphContext, input, dim, dtype=None):
    softmax = g.op("Softmax", input, axis_i=dim)
    if dtype and dtype.node().kind() != "prim::Constant":
        parsed_dtype = symbolic_helper._get_const(dtype, "i", "dtype")
        softmax = g.op(
            "Cast", softmax, to_i=_type_utils.JitScalarType(parsed_dtype).onnx_type()
        )

    return softmax


@_onnx_symbolic("aten::log_softmax")
@symbolic_helper.parse_args("v", "i", "none")
def log_softmax(g: jit_utils.GraphContext, input, dim, dtype=None):
    return_op = g.op("LogSoftmax", input, axis_i=dim)
    if dtype and dtype.node().kind() != "prim::Constant":
        parsed_dtype = symbolic_helper._get_const(dtype, "i", "dtype")
        return_op = g.op(
            "Cast", return_op, to_i=_type_utils.JitScalarType(parsed_dtype).onnx_type()
        )
    return return_op


@_onnx_symbolic("aten::frobenius_norm")
@symbolic_helper.parse_args("v", "v", "i")
def frobenius_norm(g: jit_utils.GraphContext, self, dim=None, keepdim=False):
    dim_val = symbolic_helper._maybe_get_const(dim, "is")
    if not symbolic_helper._is_value(dim_val) and len(dim_val) == 0:
        return g.op("ReduceL2", self, keepdims_i=0)
    sqr = g.op("Mul", self, self)
    sumsqr = symbolic_helper._reducesum_helper(g, sqr, dim, keepdims_i=keepdim)
    return g.op("Sqrt", sumsqr)


@_onnx_symbolic("aten::split")
@symbolic_helper.parse_args("v", "v", "i", "i")
def split(g: jit_utils.GraphContext, self, split_size_or_sizes, dim, _outputs=None):
    if not symbolic_helper._is_split_static(split_size_or_sizes, _outputs):
        split_out = g.op("SplitToSequence", self, split_size_or_sizes, axis_i=dim)
        if _outputs is None:
            return split_out
        # Convert to multiple slice nodes iff number of splits and number of outputs are statically known.
        if (
            symbolic_helper._is_packed_list(split_size_or_sizes)
            and len(symbolic_helper._unpack_list(split_size_or_sizes)) == _outputs
        ):
            split_sizes = [
                symbolic_helper._unsqueeze_helper(g, v, [0])
                for v in symbolic_helper._unpack_list(split_size_or_sizes)
            ]

            start = g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))
            axis = g.op("Constant", value_t=torch.tensor([dim], dtype=torch.long))
            res = []
            for i in range(_outputs):
                end = g.op(
                    "Add", start, split_sizes[i]
                )  # split_sizes is a list of same length as _outputs
                res.append(g.op("Slice", self, start, end, axis))
                start = end
            return res
        return [
            g.op(
                "SequenceAt",
                split_out,
                g.op("Constant", value_t=torch.tensor([i], dtype=torch.long)),
            )
            for i in range(_outputs)
        ]

    split_val = symbolic_helper._node_get(split_size_or_sizes.node(), "value")
    if split_val.dim() > 0:
        # pyrefly: ignore [bad-argument-type]
        return g.op("Split", self, split_size_or_sizes, axis_i=dim, outputs=_outputs)
    split_size = symbolic_helper._get_const(split_size_or_sizes, "i", "split_size")

    size = symbolic_helper._get_tensor_dim_size(self, dim)
    if size is None:
        if _outputs is not None:
            size = split_size * _outputs
        else:
            raise errors.SymbolicValueError(
                "Unknown dimension size not supported", self
            )
    splits = [split_size] * (size // split_size)
    leftover = size % split_size
    if leftover:
        splits.append(leftover)
    splits = g.op("Constant", value_t=torch.tensor(splits))
    # pyrefly: ignore [bad-argument-type]
    return g.op("Split", self, splits, axis_i=dim, outputs=_outputs)


@_onnx_symbolic("aten::split_with_sizes")
def split_with_sizes(g: jit_utils.GraphContext, self, split_sizes, dim, _outputs=None):
    return split(g, self, split_sizes, dim, _outputs)


@_onnx_symbolic("aten::unsafe_split")
def unsafe_split(
    g: jit_utils.GraphContext, self, split_size_or_sizes, dim, _outputs=None
):
    return split(g, self, split_size_or_sizes, dim, _outputs)


@_onnx_symbolic("aten::unsafe_split_with_sizes")
def unsafe_split_with_sizes(
    g: jit_utils.GraphContext, self, split_sizes, dim, _outputs=None
):
    return split_with_sizes(g, self, split_sizes, dim, _outputs)


@_onnx_symbolic("aten::tensor_split")
@symbolic_helper.parse_args("v", "v", "i", "i")
def tensor_split(
    g: jit_utils.GraphContext, self, indices_or_sections, dim, _outputs=None
):
    axis = g.op("Constant", value_t=torch.tensor(dim, dtype=torch.long))
    axis = opset11.unsqueeze(g, axis, 0)
    const_1 = g.op("Constant", value_t=torch.tensor(1, dtype=torch.long))

    if symbolic_helper._is_split_static(indices_or_sections, _outputs):
        split_val = symbolic_helper._node_get(indices_or_sections.node(), "value")

        if split_val.dim() > 0:
            start = g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))
            res = []
            if _outputs is None:
                raise AssertionError("_outputs must be non-None")
            for i in range(_outputs - 1):
                end = g.op(
                    "Gather",
                    indices_or_sections,
                    g.op("Constant", value_t=torch.tensor([i], dtype=torch.long)),
                    axis_i=0,
                )
                res.append(g.op("Slice", self, start, end, axis))
                start = end

            end = symbolic_helper._size_helper(g, self, axis)
            res.append(g.op("Slice", self, start, end, axis))
            return res

        split_size = symbolic_helper._get_const(
            indices_or_sections, "i", "indices_or_sections"
        )

        size = symbolic_helper._get_tensor_dim_size(self, dim)
        if size is None:
            if _outputs is not None:
                size = split_size * _outputs
            else:
                raise errors.SymbolicValueError(
                    "Unknown dimension size not supported", self
                )

        min_split_size = size // split_size
        num_splits_one_extra = size % split_size

        splits = num_splits_one_extra * [min_split_size + 1]
        leftover = (split_size - num_splits_one_extra) * [min_split_size]

        splits = g.op(
            "Constant", value_t=torch.tensor(splits + leftover, dtype=torch.long)
        )
        # pyrefly: ignore [bad-argument-type]
        return g.op("Split", self, splits, axis_i=dim, outputs=_outputs)

    if (
        symbolic_helper._is_tensor(indices_or_sections)
        and symbolic_helper._get_tensor_rank(indices_or_sections) == 1
    ):
        loop_len = symbolic_helper._size_helper(
            g, indices_or_sections, g.op("Constant", value_t=torch.tensor(0))
        )
        loop_len = opset11.unsqueeze(g, loop_len, 0)
        loop_condition = g.op("Cast", const_1, to_i=_C_onnx.TensorProtoDataType.BOOL)

        # To make the first slice in the below loop work,
        # we pad a zero to the first position so that it will be the initial start of slice.
        padding_0 = g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))
        indices_or_sections = g.op("Concat", padding_0, indices_or_sections, axis_i=0)

        final_splits = g.op("SequenceEmpty")
        # Loop inputs
        loop, (loop_context,), _ = jit_utils.add_op_with_blocks(
            g, "Loop", loop_len, loop_condition, final_splits, outputs=1, n_blocks=1
        )

        loop_block = loop_context.block
        block_input_iter = utils._add_input_to_block(loop_block)
        cond = utils._add_input_to_block(loop_block)  # noqa: F841
        final_splits = utils._add_input_to_block(loop_block)

        start = loop_context.op(
            "Gather", indices_or_sections, block_input_iter, axis_i=0
        )
        end = loop_context.op(
            "Gather",
            indices_or_sections,
            loop_context.op("Add", block_input_iter, const_1),
            axis_i=0,
        )

        slice = loop_context.op("Slice", self, start, end, axis)
        final_splits = loop_context.op("SequenceInsert", final_splits, slice)

        # Loop outputs
        cond_out = loop_context.op("Identity", loop_condition)
        utils._add_output_to_block(loop_block, cond_out)
        utils._add_output_to_block(loop_block, final_splits)

        loop_out = loop.node().output()
        start = g.op(
            "Gather",
            indices_or_sections,
            g.op("Constant", value_t=torch.tensor(-1, dtype=torch.long)),
            axis_i=0,
        )
        start = opset11.unsqueeze(g, start, 0)
        end = symbolic_helper._size_helper(g, self, axis)

        last_slice = g.op("Slice", self, start, end, axis)

        return g.op("SequenceInsert", loop_out, last_slice)

    else:  # scalar tensor
        dim_size = symbolic_helper._size_helper(g, self, axis)
        min_split_size = g.op("Div", dim_size, indices_or_sections)
        min_split_size_plus_1 = g.op(
            "Add",
            min_split_size,
            const_1,
        )
        num_splits_one_extra = g.op("Mod", dim_size, indices_or_sections)
        splits = g.op("Tile", min_split_size_plus_1, num_splits_one_extra)
        leftover = g.op(
            "Tile",
            min_split_size,
            g.op(
                "Sub",
                opset11.unsqueeze(g, indices_or_sections, 0),
                num_splits_one_extra,
            ),
        )

        splits = g.op("Concat", splits, leftover, axis_i=0)
        if _outputs is None:
            return g.op("SplitToSequence", self, splits, axis_i=dim)
        return g.op("Split", self, splits, axis_i=dim, outputs=_outputs)


@_onnx_symbolic("aten::unbind")
@symbolic_helper.parse_args("v", "i", "i")
def unbind(g: jit_utils.GraphContext, self, dim=0, _outputs=None):
    if _outputs is None:
        return g.op(
            "SplitToSequence",
            self,
            g.op("Constant", value_t=torch.tensor(1, dtype=torch.long)),
            axis_i=dim,
            keepdims_i=0,
        )

    splits = g.op("Constant", value_t=torch.tensor([1] * _outputs))
    outputs = g.op("Split", self, splits, axis_i=dim, outputs=_outputs)
    outputs = [outputs] if _outputs == 1 else outputs
    squeezed_outputs = [
        g.op("Squeeze", out, g.op("Constant", value_t=torch.tensor([dim])))
        for out in outputs
    ]
    return squeezed_outputs


@_onnx_symbolic("aten::nonzero_numpy")
# Emitted from `torch.nonzero(x, as_tuple=True)`
def nonzero_numpy(g: jit_utils.GraphContext, input, _outputs=None):
    return unbind(g, opset9.nonzero(g, input), 1, _outputs=_outputs)


@_onnx_symbolic("aten::where")
@symbolic_helper.parse_args("v", "v", "v", "i")
def where(g: jit_utils.GraphContext, condition, self=None, other=None, _outputs=None):
    # Assumes that torch.where's first argument takes only Bool and Byte tensors.
    if not symbolic_helper._is_bool(condition):
        condition = g.op("Cast", condition, to_i=_C_onnx.TensorProtoDataType.BOOL)
    if self is None:
        condition = opset9.nonzero(g, condition)
        return symbolic_helper._unbind_helper(
            g, condition, g.op("Constant", value_t=torch.tensor(1)), _outputs
        )
    # pyrefly: ignore [bad-argument-type]
    return g.op("Where", condition, self, other)


@_onnx_symbolic("aten::fake_quantize_per_channel_affine")
@symbolic_helper.parse_args("v", "v", "v", "i", "i", "i")
def fake_quantize_per_channel_affine(
    g: jit_utils.GraphContext,
    inputs,
    scale,
    zero_point,
    axis,
    quant_min=-128,
    quant_max=127,
):
    # NOTE: (0, 127) is allowed as special case. PyTorch restricts activations to be in the range (0, 127).
    #   https://github.com/pytorch/pytorch/blob/b34b192d6b97325c9f78e5995c48c8498ede34bd/torch/ao/quantization/observer.py#L1422
    if (quant_min, quant_max) not in [(0, 255), (-128, 127), (0, 127)]:
        raise errors.SymbolicValueError(
            "For (quant_min, quant_max), ONNX allows only (0, 127), (0, 255) and (-128, 127). "
            f"Got ({quant_min}, {quant_max})",
            inputs,
        )
    # ONNX defines zero_point to be int8 or uint8
    if quant_min == 0:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.UINT8)
    else:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.INT8)
    quantized = g.op("QuantizeLinear", inputs, scale, zero_point, axis_i=axis)
    if (quant_min, quant_max) == (0, 127):
        quantized = g.op(
            "Clip",
            quantized,
            opset9.unused(g),
            g.op("Constant", value_t=torch.tensor(127, dtype=torch.uint8)),
        )
    return g.op("DequantizeLinear", quantized, scale, zero_point, axis_i=axis)


@_onnx_symbolic("aten::fake_quantize_per_tensor_affine")
@symbolic_helper.parse_args("v", "v", "v", "i", "i")
def fake_quantize_per_tensor_affine(
    g: jit_utils.GraphContext,
    inputs,
    scale,
    zero_point,
    quant_min=-128,
    quant_max=127,
):
    # NOTE: (0, 127) is allowed as special case. PyTorch restricts activations to be in the range (0, 127).
    #   https://github.com/pytorch/pytorch/blob/b34b192d6b97325c9f78e5995c48c8498ede34bd/torch/ao/quantization/observer.py#L1422
    if (quant_min, quant_max) not in [(0, 255), (-128, 127), (0, 127)]:
        raise errors.SymbolicValueError(
            "For (quant_min, quant_max), ONNX allows only (0, 127), (0, 255) and (-128, 127). "
            f"Got ({quant_min}, {quant_max})",
            inputs,
        )
    if quant_min == 0:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.UINT8)
    else:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.INT8)
    if (
        _type_utils.JitScalarType.from_value(scale, _type_utils.JitScalarType.UNDEFINED)
        != _type_utils.JitScalarType.FLOAT
    ):
        scale = g.op("Cast", scale, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    quantized = g.op("QuantizeLinear", inputs, scale, zero_point)
    if (quant_min, quant_max) == (0, 127):
        quantized = g.op(
            "Clip",
            quantized,
            opset9.unused(g),
            g.op("Constant", value_t=torch.tensor(127, dtype=torch.uint8)),
        )
    return g.op("DequantizeLinear", quantized, scale, zero_point)


def _reduce_op_symbolic(onnx_op_name):
    def symbolic(g, self, dim=None, keepdim=None):
        self = symbolic_helper._maybe_cast_reduce_op_input(g, self)
        if dim is None:
            # all-reduce path
            return symbolic_helper._handle_reduce_dim_none(g, self, onnx_op_name)
        else:
            keepdim = symbolic_helper._get_const(keepdim, "i", "keepdim")
            return g.op(onnx_op_name, self, dim, keepdims_i=keepdim)

    return symbolic


@_onnx_symbolic(
    "aten::sum",
    decorate=[symbolic_helper._apply_params("ReduceSum", "sum")],
)
def _reduce_with_dtype(onnx_op, name):
    symbolic = _reduce_op_symbolic(onnx_op)

    @symbolic_helper._overload_by_arg_count
    def reduce(g, *args, **kwargs):
        @symbolic_helper.parse_args("v", "none")
        def reduce_nodim(g, self, dtype):
            dtype_onnx = None
            if dtype.node().kind() == "onnx::Constant":
                dtype = symbolic_helper._get_const(dtype, "i", "dtype")
                dtype_onnx = _type_utils.JitScalarType(dtype).onnx_type()
                self = g.op("Cast", self, to_i=dtype_onnx)
            elif dtype.node().kind() != "prim::Constant":
                return symbolic_helper._unimplemented(name, "dtype", dtype)
            result = symbolic(g, self)
            if dtype_onnx is not None:
                result_dtype_onnx = _type_utils.JitScalarType.from_value(
                    result
                ).onnx_type()
                if result_dtype_onnx != dtype_onnx:
                    result = g.op("Cast", result, to_i=dtype_onnx)
            return result

        @symbolic_helper.parse_args("v", "v", "i", "none")
        def reduce_dim(g, self, dim, keepdim, dtype):
            dtype_onnx = None
            if dtype.node().kind() == "onnx::Constant":
                dtype = symbolic_helper._get_const(dtype, "i", "dtype")
                dtype_onnx = _type_utils.JitScalarType(dtype).onnx_type()
                self = g.op("Cast", self, to_i=dtype_onnx)
            elif dtype.node().kind() != "prim::Constant":
                return symbolic_helper._unimplemented(name, "dtype", dtype)
            result = symbolic(g, self, dim, keepdim)
            if dtype_onnx is not None:
                result_dtype_onnx = _type_utils.JitScalarType.from_value(
                    result
                ).onnx_type()
                if result_dtype_onnx != dtype_onnx:
                    result = g.op("Cast", result, to_i=dtype_onnx)
            return result

        return reduce_nodim, reduce_dim

    return reduce


# Ported from
# https://github.com/microsoft/onnxscript/blob/6b1b81700b4523f31d8c6d3321e5d8ef5d42b764/onnxscript/function_libs/torch_aten/ops/core.py#L6097
# NOTE: Supporting aten::unflatten before opset13 needs helper function to adjust ONNX op changes in Concat, Slice, ...
@_onnx_symbolic("aten::unflatten")
def unflatten(g: jit_utils.GraphContext, input, dim, unflattened_size):
    input_dim = symbolic_helper._get_tensor_rank(input)
    if input_dim is None:
        return symbolic_helper._unimplemented(
            "dim",
            "ONNX and PyTorch use different strategies to split the input. "
            "Input rank must be known at export time.",
        )

    # dim could be negative
    input_dim = g.op("Constant", value_t=torch.tensor([input_dim], dtype=torch.int64))
    dim = g.op("Add", input_dim, dim)
    dim = g.op("Mod", dim, input_dim)

    input_size = g.op("Shape", input)

    head_start_idx = g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64))
    head_end_idx = g.op(
        "Reshape", dim, g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64))
    )
    head_part_rank = g.op("Slice", input_size, head_start_idx, head_end_idx)

    dim_plus_one = g.op(
        "Add", dim, g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64))
    )
    tail_start_idx = g.op(
        "Reshape",
        dim_plus_one,
        g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64)),
    )
    tail_end_idx = g.op(
        "Constant", value_t=torch.tensor([_constants.INT64_MAX], dtype=torch.int64)
    )
    tail_part_rank = g.op("Slice", input_size, tail_start_idx, tail_end_idx)

    final_shape = g.op(
        "Concat", head_part_rank, unflattened_size, tail_part_rank, axis_i=0
    )

    return symbolic_helper._reshape_helper(g, input, final_shape)


@_onnx_symbolic("aten::unsafe_chunk")
@symbolic_helper.parse_args("v", "i", "i", "i")
def unsafe_chunk(g: jit_utils.GraphContext, self, chunks, dim, _outputs=None):
    if _outputs is None:
        return g.op(
            "SplitToSequence",
            self,
            g.op("Constant", value_t=torch.tensor(1, dtype=torch.long)),
            axis_i=dim,
            keepdims_i=0,
        )

    size = symbolic_helper._get_tensor_dim_size(self, dim)
    if size is None:
        return symbolic_helper._unimplemented("unsafe_chunk", "unknown dimension size")
    split_size = (size + chunks - 1) // chunks
    splits = [split_size] * (size // split_size)
    leftover = size % split_size
    if leftover:
        splits.append(leftover)

    # TODO: So far we don"t have a module using this method. We"ll keep
    # this as a constant unless we see a request of dynamics in any
    # user's modules.
    splits = g.op("Constant", value_t=torch.tensor(splits, dtype=torch.long))
    return g.op("Split", self, splits, axis_i=dim, outputs=_outputs)


@_onnx_symbolic("aten::tile")
def tile(g: jit_utils.GraphContext, self, dims):
    self_shape = g.op("Shape", self)
    self_rank = g.op("Size", self_shape)
    dims_rank = g.op("Size", dims)
    diff = g.op("Sub", self_rank, dims_rank)
    const_zero = g.op("Constant", value_t=torch.tensor([0]))

    # 1. If dims is shorter than self.shape pad dims with 1
    dims_shorter_than_self_shape = g.op("Greater", diff, const_zero)
    (
        if_op_greater,
        (if_context_greater, else_context_greater),
        _,
    ) = jit_utils.add_op_with_blocks(
        g, "If", dims_shorter_than_self_shape, n_blocks=2, outputs=1
    )
    const_one = if_context_greater.op("Constant", value_t=torch.LongTensor([1]))
    diff_1d_greater = if_context_greater.op("Reshape", diff, const_one)
    exapnd_ones_greater = if_context_greater.op("Expand", const_one, diff_1d_greater)
    dims_ = if_context_greater.op("Concat", exapnd_ones_greater, dims, axis_i=0)
    utils._add_output_to_block(if_context_greater.block, dims_)
    identity_dim = else_context_greater.op("Identity", dims)
    utils._add_output_to_block(else_context_greater.block, identity_dim)
    dims_final = if_op_greater.node().output()

    # 2. If dims is longer than self.shape pad self.shape with 1
    dims_longer_than_self_shape = g.op("Less", diff, const_zero)
    (
        if_op_less,
        (if_context_less, else_context_less),
        _,
    ) = jit_utils.add_op_with_blocks(
        g, "If", dims_longer_than_self_shape, n_blocks=2, outputs=1
    )
    const_one = if_context_less.op("Constant", value_t=torch.LongTensor([1]))
    diff_1d_less = if_context_less.op(
        "Reshape",
        if_context_less.op("Abs", diff),
        const_one,
    )
    exapnd_ones_less = if_context_less.op("Expand", const_one, diff_1d_less)
    self_final_shape = if_context_less.op(
        "Concat", exapnd_ones_less, self_shape, axis_i=0
    )
    self_ = if_context_less.op("Reshape", self, self_final_shape)
    utils._add_output_to_block(if_context_less.block, self_)
    identity_self = else_context_less.op("Identity", self)
    utils._add_output_to_block(else_context_less.block, identity_self)
    self_final = if_op_less.node().output()

    dims_final = g.op("Cast", dims_final, to_i=_C_onnx.TensorProtoDataType.INT64)
    return g.op("Tile", self_final, dims_final)


@_onnx_symbolic("aten::repeat_interleave")
def repeat_interleave(
    g: jit_utils.GraphContext, self, repeats, dim=None, output_size=None
):
    repeats_dim = symbolic_helper._get_tensor_rank(repeats)
    repeats_sizes = symbolic_helper._get_tensor_sizes(repeats)
    input_sizes = symbolic_helper._get_tensor_sizes(self)
    if repeats_dim is None:
        raise errors.SymbolicValueError(
            "Unsupported: ONNX export of repeat_interleave for unknown repeats rank.",
            self,
        )
    if repeats_sizes is None:
        raise errors.SymbolicValueError(
            "Unsupported: ONNX export of repeat_interleave for unknown repeats size.",
            self,
        )
    if input_sizes is None:
        raise errors.SymbolicValueError(
            "Unsupported: ONNX export of repeat_interleave for unknown input size.",
            self,
        )

    final_dim = dim
    # if dim is None flatten
    # By default, use the flattened input array, and return a flat output array
    if symbolic_helper._is_none(dim):
        self = symbolic_helper._reshape_helper(
            g, self, g.op("Constant", value_t=torch.tensor([-1]))
        )
        dim = torch.tensor(0, dtype=torch.int64)
    else:
        dim = symbolic_helper._maybe_get_scalar(dim)

    # Handle cases where dim is negative
    if dim < 0:
        dim += len(input_sizes)

    output_sizes = input_sizes.copy()
    for idx, input_size in enumerate(input_sizes):
        if input_size is None:
            output_sizes[idx], input_sizes[idx] = 0, -1

    # Check if all indices should be repeated the same number of times.
    if repeats_dim == 0 or (repeats_dim == 1 and repeats_sizes[0] == 1):
        return symbolic_helper._repeat_interleave_single_value_repeat_helper(
            g, self, repeats, dim
        )

    cond_dynamic_repeats = repeats_dim == 1 and repeats_sizes[0] is None
    # If input size is dynamic or repeats vector is dynamic
    if output_sizes[dim] == 0 or cond_dynamic_repeats:
        reps = symbolic_helper._size_helper(g, self, dim)
        reps = opset11.unsqueeze(g, reps, 0)

        # Check if repeats is dynamic
        # As repeats is dynamic, we use a where node as a substitute for the if statement
        # If repeats_dim = 1, expand repeats otherwise use original tensor
        if cond_dynamic_repeats:
            repeat_dim = symbolic_helper._size_helper(
                g, repeats, g.op("Constant", value_t=torch.LongTensor([0]))
            )
            repeat_cond = g.op(
                "Equal", repeat_dim, g.op("Constant", value_t=torch.LongTensor([1]))
            )
            repeats = where(g, repeat_cond, g.op("Expand", repeats, reps), repeats)
    # There are cases when the repeats are 1-d tensor with multiple repeats, but dim
    # provided along one of the dynamic axes provided. A simple example would be
    # input.shape -> [1, 1, *] where * represents the dynamic axes, and dim = 2
    # Now, repeat interleaving can be performed in pytorch when the value of * matches
    # with the number of elements in repeat, for example if * -> 2, number of repeats
    # should be 2 as well.
    else:
        return opset9.repeat_interleave(g, self, repeats, final_dim)

    reps_like = g.op(
        "ConstantOfShape",
        g.op("Shape", repeats),
        value_t=torch.tensor([1], dtype=torch.long),
    )
    r_splits = split(g, repeats, reps_like, 0)
    i_splits = split(g, self, reps_like, dim)

    output_sizes[dim], input_sizes[dim] = -1, 1

    # Create a loop to iterate over each value along the dimension
    # and perform individual interleaving using the repeats tensor
    # Loop is of the following pattern
    # input (trip_count, cond)
    #   int trip_count = ...;
    #   bool cond = ...;
    #   for (int i=0; i < trip_count && cond; ++i) {
    #     cond = ...;
    #   }

    # Loop conditions
    loop_condition = g.op("Constant", value_t=torch.tensor(1))
    loop_condition = g.op("Cast", loop_condition, to_i=_C_onnx.TensorProtoDataType.BOOL)
    loop_len = reps

    # Create an empty sequence to store final expansions
    final_splits = g.op("SequenceEmpty")

    # Loop inputs
    loop, (loop_context,), _ = jit_utils.add_op_with_blocks(
        g, "Loop", loop_len, loop_condition, final_splits, n_blocks=1
    )

    loop_block = loop_context.block
    block_input_iter = utils._add_input_to_block(loop_block)
    cond = utils._add_input_to_block(loop_block)  # noqa: F841
    final_splits = utils._add_input_to_block(loop_block)

    r_split = loop_context.op("SequenceAt", r_splits, block_input_iter)
    i_split = loop_context.op("SequenceAt", i_splits, block_input_iter)

    i_split = opset11.unsqueeze(loop_context, i_split, dim + 1)
    r_concat = [
        loop_context.op("Constant", value_t=torch.LongTensor(input_sizes[: dim + 1])),
        r_split,
        loop_context.op("Constant", value_t=torch.LongTensor(input_sizes[dim + 1 :])),
    ]
    r_concat = loop_context.op("Concat", *r_concat, axis_i=0)
    i_split = opset9.expand(loop_context, i_split, r_concat, None)
    i_split = symbolic_helper._reshape_helper(
        loop_context, i_split, g.op("Constant", value_t=torch.LongTensor(output_sizes))
    )
    final_splits = loop_context.op("SequenceInsert", final_splits, i_split)

    # Loop outputs
    cond_out = loop_context.op(
        "Cast", loop_condition, to_i=_C_onnx.TensorProtoDataType.BOOL
    )
    utils._add_output_to_block(loop_block, cond_out)
    utils._add_output_to_block(loop_block, final_splits)

    loop_out = loop.node().output()
    loop_out = g.op("ConcatFromSequence", loop_out, axis_i=dim)
    return loop_out


@_onnx_symbolic("aten::diagonal")
@symbolic_helper.parse_args("v", "i", "i", "i")
def diagonal(g: jit_utils.GraphContext, self, offset, dim1, dim2):
    rank = symbolic_helper._get_tensor_rank(self)
    # Replace negative indexing when rank is known
    if rank is not None:
        dim1 = dim1 if dim1 >= 0 else dim1 + rank
        dim2 = dim2 if dim2 >= 0 else dim2 + rank

    dim1_size = opset9.size(
        g, self, dim=g.op("Constant", value_t=torch.LongTensor([dim1]))
    )
    dim2_size = opset9.size(
        g, self, dim=g.op("Constant", value_t=torch.LongTensor([dim2]))
    )
    # Create appropriate mask
    mask_shape = g.op("Concat", dim1_size, dim2_size, axis_i=0)
    mask = opset9.zeros(g, mask_shape, None, None, None)
    mask = g.op("EyeLike", mask, k_i=offset)
    # dim1 and dim2 appended as a dimension at the end of the shape

    if rank is not None:
        axes = list(range(rank))
        axes.remove(dim1)
        axes.remove(dim2)
        self = g.op("Transpose", self, perm_i=axes + [dim1, dim2])
    else:
        return symbolic_helper._unimplemented("diagonal", "unknown input rank")

    # Multiply input and mask to calculate values along diagonal
    # The mask consists of one values where diagonal values are to be calculated
    # For example:
    # [[1.1, 1.2, 1.3],   *    [[1, 0, 0]   =   [[1.1, 0, 0],
    #  [2.1, 2.2, 2.3],         [0, 1, 0]        [0, 2.2, 0],
    #  [3.1, 3.2, 3.3]]         [0, 0, 1]]       [0, 0, 3.3]]
    result = g.op("Mul", self, mask)
    result = symbolic_helper._reducesum_helper(g, result, axes_i=[-1], keepdims_i=0)

    # Calculate gather indices based on offset and dims
    # If offset is greater than zero, set offset to zero as this aids in
    # calculation of selection window
    offset_op = g.op("Constant", value_t=torch.LongTensor([offset]))
    if offset >= 0:
        diag_size = g.op(
            "Max",
            g.op("Min", dim1_size, g.op("Sub", dim2_size, offset_op)),
            g.op("Constant", value_t=torch.LongTensor([0])),
        )
        offset = 0
    else:
        diag_size = g.op(
            "Max",
            g.op("Min", g.op("Add", dim1_size, offset_op), dim2_size),
            g.op("Constant", value_t=torch.LongTensor([0])),
        )
    diag_size = g.op("Concat", diag_size, axis_i=0)

    # Calculate which diagonal values to select
    # For example, in cases with offsets:
    # [[0, 1.1, 0]
    #  [0, 0, 2.2]]
    # we need to select the last two columns, so we create a tensor
    # with all columns that are to be selected
    # So in this example, it is [1, 2]
    select_window_ones_fill = opset9.ones(g, diag_size, 4, None, None)
    select_window = g.op(
        "CumSum",
        select_window_ones_fill,
        g.op("Constant", value_t=torch.LongTensor([0])),
    )
    select_window = g.op(
        "Add",
        select_window,
        g.op("Constant", value_t=torch.LongTensor([abs(offset) - 1])),
    )

    gather_shape = [
        opset9.size(g, result, dim=g.op("Constant", value_t=torch.LongTensor([axis])))
        for axis in list(range(rank))[:-2]
    ]
    gather_shape.append(diag_size)
    gather_shape = g.op("Concat", *gather_shape, axis_i=0)
    gather_indices = opset9.zeros(g, gather_shape, 4, None, None)

    # There might be cases where offset value is greater than number of rows/columns
    # and might cause the diagonal to overrun and as a result of this, diag_size would be zero.
    # For example, if
    #       offset = 9, dim1_size = 2 (columns), dim2_size = 4 (rows)
    #       diag_size = max(min(2, (4-9)), 0) = 0, based on calculation above
    # Cases with diagonal overrun always result in diag_size = max(0, -ve value) = 0
    # In cases without diagonal overrun, we select the appropriate rows/columns along which we
    # are calculating diagonal values. In cases with diagonal overrun, we return a tensor which has
    # the dimension of the row/column where overrun occurred as 0-dim, as we are essentially
    # returning an empty tensor
    overrun_cond = g.op(
        "Not",
        g.op(
            "Equal",
            diag_size,
            g.op("Constant", value_t=torch.tensor(0, dtype=torch.int64)),
        ),
    )

    if_op, (if_context, else_context), _ = jit_utils.add_op_with_blocks(
        g, "If", overrun_cond, n_blocks=2
    )

    gather_indices_if_block = if_context.op("Add", gather_indices, select_window)
    gather_indices_if_block = symbolic_helper._unsqueeze_helper(
        if_context, gather_indices_if_block, [rank - 1]
    )
    final_non_overrun = if_context.op(
        "GatherND", result, gather_indices_if_block, batch_dims_i=rank - 2
    )
    final_overrun = opset9.zeros(else_context, gather_shape, 6, None, None)
    utils._add_output_to_block(if_context.block, final_non_overrun)
    utils._add_output_to_block(else_context.block, final_overrun)
    return if_op


# Quantized ops


@_onnx_symbolic("quantized::linear")
def quantized_linear(
    g: jit_utils.GraphContext, q_input, q_weight, bias, op_scale, op_zero_point
):
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, axis = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(
        g, bias, input_scale, weight_scale, axis
    )
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.linear(g, input, weight, bias)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::linear_relu")
def quantized_linear_relu(
    g: jit_utils.GraphContext, q_input, q_weight, bias, op_scale, op_zero_point
):
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, axis = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(
        g, bias, input_scale, weight_scale, axis
    )
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.linear(g, input, weight, bias)
    output = opset9.relu(g, output)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::conv1d_relu")
def quantized_conv1d_relu(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
):
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, axis = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(
        g, bias, input_scale, weight_scale, axis
    )
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.conv1d(g, input, weight, bias, stride, padding, dilation, groups)
    output = opset9.relu(g, output)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::conv2d_relu")
def quantized_conv2d_relu(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
):
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, axis = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(
        g, bias, input_scale, weight_scale, axis
    )
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.conv2d(g, input, weight, bias, stride, padding, dilation, groups)
    output = opset9.relu(g, output)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::conv3d_relu")
def quantized_conv3d_relu(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
):
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, axis = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(
        g, bias, input_scale, weight_scale, axis
    )
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.conv3d(g, input, weight, bias, stride, padding, dilation, groups)
    output = opset9.relu(g, output)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::conv1d")
def quantized_conv1d(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
):
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, axis = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(
        g, bias, input_scale, weight_scale, axis
    )
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.conv1d(g, input, weight, bias, stride, padding, dilation, groups)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::conv2d")
def quantized_conv2d(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
):
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, axis = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(
        g, bias, input_scale, weight_scale, axis
    )
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.conv2d(g, input, weight, bias, stride, padding, dilation, groups)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::conv3d")
def quantized_conv3d(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
):
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, axis = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(
        g, bias, input_scale, weight_scale, axis
    )
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.conv3d(g, input, weight, bias, stride, padding, dilation, groups)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::conv_transpose1d")
def quantized_conv_transpose1d(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    output_padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
):
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, axis = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(
        g, bias, input_scale, weight_scale, axis
    )
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.conv_transpose2d(
        g, input, weight, bias, stride, padding, output_padding, groups, dilation
    )

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::conv_transpose2d")
def quantized_conv_transpose2d(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    output_padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
):
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, axis = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(
        g, bias, input_scale, weight_scale, axis
    )
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.conv_transpose2d(
        g, input, weight, bias, stride, padding, output_padding, groups, dilation
    )

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::conv_transpose3d")
def quantized_conv_transpose3d(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    output_padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
):
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, axis = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(
        g, bias, input_scale, weight_scale, axis
    )
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.conv_transpose3d(
        g, input, weight, bias, stride, padding, output_padding, groups, dilation
    )

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)
<|endoftext|># mypy: allow-untyped-defs
"""This file exports ONNX ops for opset 15.

Note [ONNX operators that are added/updated in opset 15]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/master/docs/Changelog.md#version-15-of-the-default-onnx-operator-set
New operators:
    Bernoulli
    CastLike
    Optional
    OptionalGetElement
    OptionalHasElement

Updated operators:
    BatchNormalization https://github.com/onnx/onnx/pull/3545
                        Backwards compatible
                        TODO: test coverage for mixed types inputs.
    Pow                https://github.com/onnx/onnx/pull/3412
                        Backwards compatible
                        TODO: bfloat16 support.
    Shape              https://github.com/onnx/onnx/pull/3580
                        Backwards compatible
                        TODO: optional start/end attribute.
"""

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in README.md

import functools

import torch
from torch import _C
from torch.onnx._internal.torchscript_exporter import (
    jit_utils,
    registration,
    symbolic_helper,
    symbolic_opset9 as opset9,
)


_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=15)


@_onnx_symbolic("aten::__is_")
def aten__is_(g: jit_utils.GraphContext, self, other):
    if symbolic_helper._is_none(other):
        if isinstance(self.type(), _C.OptionalType):
            none = g.op("OptionalHasElement", self)
            return g.op("Not", none)
        else:
            return g.op("Constant", value_t=torch.BoolTensor([0]))
    return opset9.eq(g, self, other)


@_onnx_symbolic("aten::__isnot_")
@opset9.wrap_logical_op_with_negation  # type: ignore[has-type]
def aten__isnot_(g: jit_utils.GraphContext, self, other):
    return aten__is_(g, self, other)


@_onnx_symbolic("aten::bernoulli")
def bernoulli(g: jit_utils.GraphContext, input, p=None, generator=None, out=None):
    if out is not None and not symbolic_helper._is_none(out):
        symbolic_helper._unimplemented(
            "Bernoulli", "out parameter is not supported for bernoulli", input
        )
    if generator is not None and not symbolic_helper._is_none(generator):
        symbolic_helper._unimplemented(
            "Bernoulli", "generator is not supported for bernoulli", input
        )
    if p is None or symbolic_helper._is_none(p):
        return g.op("Bernoulli", input)
    return opset9.bernoulli(g, input, p, generator, out)


@_onnx_symbolic("prim::unchecked_cast")
def prim_unchecked_cast(g: jit_utils.GraphContext, self):
    # exists to refine the type of the Value
    # if x is Optional[Tensor], unchecked_cast will cast
    # x to Tensor, so the rest of the graph knows that x is a Tensor.
    if isinstance(self.type(), _C.OptionalType):
        return g.op("OptionalGetElement", self)

    return self
<|endoftext|># mypy: allow-untyped-defs
# mypy: disable-error-code=arg-type
from __future__ import annotations

import functools
import sys

import torch
from torch._C import _onnx as _C_onnx
from torch.onnx import errors
from torch.onnx._internal.torchscript_exporter import (
    _type_utils,
    jit_utils,
    registration,
    symbolic_helper,
    symbolic_opset9 as opset9,
    utils,
)


# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in README.md

# This file exports ONNX ops for opset 12

__all__ = [
    "argmax",
    "argmin",
    "binary_cross_entropy_with_logits",
    "celu",
    "cross_entropy_loss",
    "dropout",
    "einsum",
    "ge",
    "le",
    "native_dropout",
    "nll_loss",
    "nll_loss2d",
    "nll_loss_nd",
    "outer",
    "pow",
    "tensordot",
    "unfold",
]

_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=12)


def _einsum_helper(g: jit_utils.GraphContext, equation, tensors):
    if not tensors:
        raise RuntimeError("Einsum inputs are empty.")
    # ONNX does not support bool for Einsum inputs.
    if symbolic_helper._is_bool(tensors[0]):
        tensors = [
            g.op("Cast", tensor, to_i=_C_onnx.TensorProtoDataType.INT64)
            for tensor in tensors
        ]
        return g.op(
            "Cast",
            g.op("Einsum", *tensors, equation_s=equation),
            to_i=_C_onnx.TensorProtoDataType.BOOL,
        )
    else:
        return g.op("Einsum", *tensors, equation_s=equation)


@_onnx_symbolic("aten::einsum")
@symbolic_helper.parse_args("s", "v", "is")
def einsum(g: jit_utils.GraphContext, equation, tensor_list, path=None):
    tensors = symbolic_helper._unpack_list(tensor_list)
    return _einsum_helper(g, equation, tensors)


@_onnx_symbolic("aten::outer")
@symbolic_helper.parse_args("v", "v")
def outer(g: jit_utils.GraphContext, input, other):
    # make sure to cast other to self's type
    if _type_utils.JitScalarType.from_value(
        other, _type_utils.JitScalarType.UNDEFINED
    ) != _type_utils.JitScalarType.from_value(input):
        other = g.op(
            "Cast",
            other,
            to_i=_type_utils.JitScalarType.from_value(input).onnx_type(),
        )
    return _einsum_helper(g, "i,j->ij", [input, other])


def _dropout_returns_masked_input_and_mask(
    g: jit_utils.GraphContext, input: torch._C.Value, p: float, train: bool
) -> tuple[torch._C.Value, torch._C.Value | None]:
    symbolic_helper.check_training_mode(train, "dropout")
    # In eval mode, dropout is non-op. That is, if the node's
    # train param is set to False, dropout just returns its inputs.
    if not train:
        return input, None
    p = g.op("Constant", value_t=torch.tensor(p))
    t = g.op("Constant", value_t=torch.tensor(train, dtype=torch.bool))
    r, mask = g.op("Dropout", input, p, t, outputs=2)
    return r, mask


@_onnx_symbolic("aten::dropout")
@symbolic_helper.parse_args("v", "f", "b")
def dropout(g: jit_utils.GraphContext, input, p, train):
    masked, _ = _dropout_returns_masked_input_and_mask(g, input, p, train)
    return masked


@_onnx_symbolic("aten::native_dropout")
@symbolic_helper.parse_args("v", "f", "b")
def native_dropout(g: jit_utils.GraphContext, input, p, train):
    return _dropout_returns_masked_input_and_mask(g, input, p, train)


@_onnx_symbolic("aten::nll_loss")
def nll_loss(g: jit_utils.GraphContext, self, target, weight, reduction, ignore_index):
    # none reduction : onnx::Constant[value={0}]
    # mean reduction : onnx::Constant[value={1}]
    # sum reduction : onnx::Constant[value={2}]
    reduction = symbolic_helper._maybe_get_const(reduction, "i")
    reduction_vals = ["none", "mean", "sum"]
    reduction = reduction_vals[reduction]

    # in onnx NegativeLogLikelihoodLoss specification, ignore_index is optional without default value.
    # therefore we need to set ignore_index attribute even if it is not specified (e.g. ignore_index=-100).
    ignore_index = symbolic_helper._maybe_get_const(ignore_index, "i")
    if weight.node().mustBeNone():
        nllloss = g.op(
            "NegativeLogLikelihoodLoss",
            self,
            target,
            reduction_s=reduction,
            ignore_index_i=ignore_index,
        )
    else:
        nllloss = g.op(
            "NegativeLogLikelihoodLoss",
            self,
            target,
            weight,
            reduction_s=reduction,
            ignore_index_i=ignore_index,
        )

    return nllloss


@_onnx_symbolic("aten::nll_loss2d")
def nll_loss2d(
    g: jit_utils.GraphContext, self, target, weight, reduction, ignore_index
):
    return nll_loss(g, self, target, weight, reduction, ignore_index)


@_onnx_symbolic("aten::nll_loss_nd")
def nll_loss_nd(
    g: jit_utils.GraphContext, self, target, weight, reduction, ignore_index
):
    return nll_loss(g, self, target, weight, reduction, ignore_index)


@_onnx_symbolic("aten::cross_entropy_loss")
def cross_entropy_loss(
    g: jit_utils.GraphContext,
    self,
    target,
    weight,
    reduction,
    ignore_index,
    label_smoothing,
):
    # none reduction : onnx::Constant[value={0}]
    # mean reduction : onnx::Constant[value={1}]
    # sum reduction : onnx::Constant[value={2}]
    reduction = symbolic_helper._maybe_get_const(reduction, "i")
    reduction_vals = ["none", "mean", "sum"]
    reduction = reduction_vals[reduction]

    label_smoothing = symbolic_helper._maybe_get_const(label_smoothing, "f")
    if label_smoothing is not None and label_smoothing > 0.0:
        raise errors.SymbolicValueError(
            "Unsupported: ONNX does not support label_smoothing", self
        )

    # in onnx SoftmaxCrossEntropyLoss specification, ignore_index is optional without default value.
    # therefore we need to set ignore_index attribute even if it is not specified (e.g. ignore_index=-100).
    ignore_index = symbolic_helper._maybe_get_const(ignore_index, "i")
    if weight.node().mustBeNone():
        celoss = g.op(
            "SoftmaxCrossEntropyLoss",
            self,
            target,
            reduction_s=reduction,
            ignore_index_i=ignore_index,
        )
    else:
        celoss = g.op(
            "SoftmaxCrossEntropyLoss",
            self,
            target,
            weight,
            reduction_s=reduction,
            ignore_index_i=ignore_index,
        )

    return celoss


@_onnx_symbolic("aten::binary_cross_entropy_with_logits")
@symbolic_helper.parse_args("v", "v", "v", "v", "i")
def binary_cross_entropy_with_logits(
    g: jit_utils.GraphContext, input, target, weight, pos_weight, reduction
):
    p = g.op("Constant", value_t=torch.tensor([1]))
    sig_x = opset9.sigmoid(g, input)
    log_sig_x = opset9.log(g, sig_x)
    sub_1_x = opset9.sub(g, p, sig_x)
    sub_1_y = opset9.sub(g, p, target)
    log_1_x = opset9.log(g, sub_1_x)
    if pos_weight is None or symbolic_helper._is_none(pos_weight):
        output = opset9.neg(
            g,
            opset9.add(
                g, opset9.mul(g, target, log_sig_x), opset9.mul(g, sub_1_y, log_1_x)
            ),
        )
    else:
        output = opset9.neg(
            g,
            opset9.add(
                g,
                opset9.mul(g, opset9.mul(g, target, log_sig_x), pos_weight),
                opset9.mul(g, sub_1_y, log_1_x),
            ),
        )

    if weight is not None and not symbolic_helper._is_none(weight):
        output = opset9.mul(g, weight, output)

    reduction = symbolic_helper._maybe_get_const(reduction, "i")
    if reduction == 0:
        return output
    elif reduction == 1:
        return g.op("ReduceMean", output, keepdims_i=0)
    elif reduction == 2:
        return g.op("ReduceSum", output, keepdims_i=0)
    else:
        return symbolic_helper._onnx_unsupported(
            "binary_cross_entropy_with_logits with reduction other than none, mean, or sum",
            input,
        )


@_onnx_symbolic("aten::celu")
def celu(g: jit_utils.GraphContext, self, alpha):
    alpha = symbolic_helper._maybe_get_const(alpha, "f")
    # if the input is of type double cast it to float
    if (
        _type_utils.JitScalarType.from_value(self, _type_utils.JitScalarType.UNDEFINED)
        == _type_utils.JitScalarType.DOUBLE
    ):
        self = g.op("Cast", self, to_i=_C_onnx.TensorProtoDataType.FLOAT)
        out = g.op("Celu", self, alpha_f=alpha)
        return g.op("Cast", out, to_i=_C_onnx.TensorProtoDataType.DOUBLE)

    return g.op("Celu", self, alpha_f=alpha)


@_onnx_symbolic("aten::argmax")
@symbolic_helper.parse_args("v", "v", "b")
def argmax(
    g: jit_utils.GraphContext,
    input: torch._C.Value,
    dim: torch._C.Value,
    keepdim: bool,
):
    return symbolic_helper._argmin_argmax_helper(g, input, dim, keepdim, "ArgMax")


@_onnx_symbolic("aten::argmin")
@symbolic_helper.parse_args("v", "v", "b")
def argmin(
    g: jit_utils.GraphContext,
    input: torch._C.Value,
    dim: torch._C.Value,
    keepdim: bool,
):
    return symbolic_helper._argmin_argmax_helper(g, input, dim, keepdim, "ArgMin")


@_onnx_symbolic("aten::pow")
def pow(g: jit_utils.GraphContext, self, exponent):
    return g.op("Pow", self, exponent)


@_onnx_symbolic("aten::ge")
def ge(g: jit_utils.GraphContext, input, other):
    return g.op("GreaterOrEqual", input, other)


@_onnx_symbolic("aten::le")
def le(g: jit_utils.GraphContext, input, other):
    return g.op("LessOrEqual", input, other)


@_onnx_symbolic("aten::unfold")
@symbolic_helper.parse_args("v", "i", "v", "v")
def unfold(g: jit_utils.GraphContext, input, dimension, size, step):
    const_size = symbolic_helper._maybe_get_const(size, "i")
    const_step = symbolic_helper._maybe_get_const(step, "i")
    if not symbolic_helper._is_value(const_size) and not symbolic_helper._is_value(
        const_step
    ):
        return opset9.unfold(g, input, dimension, const_size, const_step)

    sizedim = symbolic_helper._get_tensor_dim_size(input, dimension)
    if sizedim is not None:
        low_start = g.op("Constant", value_t=torch.tensor(0))
        low_end = g.op("Constant", value_t=torch.tensor(sizedim))
        hi_end = g.op("Constant", value_t=torch.tensor(sizedim + 1))
        low_indices = g.op("Range", low_start, low_end, step)
        hi_indices = g.op("Range", size, hi_end, step)

        low_size = symbolic_helper._size_helper(
            g, low_indices, g.op("Constant", value_t=torch.tensor(0))
        )
        hi_size = symbolic_helper._size_helper(
            g, hi_indices, g.op("Constant", value_t=torch.tensor(0))
        )

        ndim = symbolic_helper._get_tensor_rank(input)
        if ndim is None:
            raise AssertionError("ndim must be non-None")
        perm = list(range(ndim))
        perm.append(perm.pop(dimension))

        unsqueeze_list = []
        loop_condition = g.op("Constant", value_t=torch.tensor(1))
        loop_condition = g.op(
            "Cast", loop_condition, to_i=_C_onnx.TensorProtoDataType.BOOL
        )
        loop_len = g.op("Min", low_size, hi_size)

        loop, (loop_context,), _ = jit_utils.add_op_with_blocks(
            g, "Loop", loop_len, loop_condition, n_blocks=1
        )

        loop_block = loop_context.block
        block_input_iter = utils._add_input_to_block(loop_block)
        cond = utils._add_input_to_block(loop_block)  # noqa: F841

        starts = loop_context.op("Gather", low_indices, block_input_iter)
        ends = loop_context.op("Gather", hi_indices, block_input_iter)
        axes = loop_context.op("Constant", value_t=torch.tensor([2]))
        starts = symbolic_helper._unsqueeze_helper(loop_context, starts, [0])
        ends = symbolic_helper._unsqueeze_helper(loop_context, ends, [0])
        stack = loop_context.op("Slice", input, starts, ends, axes)

        unsqueeze = symbolic_helper._unsqueeze_helper(
            loop_context, loop_context.op("Transpose", stack, perm_i=perm), [dimension]
        )
        unsqueeze_list.append(unsqueeze)
        concat = loop_context.op("Concat", *unsqueeze_list, axis_i=0)

        cond_out = loop_context.op(
            "Cast",
            loop_condition,
            # pyrefly: ignore [bad-argument-type]
            _C_onnx.TensorProtoDataType.BOOL,
        )
        utils._add_output_to_block(loop_block, cond_out)
        utils._add_output_to_block(loop_block, concat)

        loop_output = loop.node().output()
        perm = [0, 1, 2, 3, 4]
        perm[0], perm[dimension + 1] = perm[dimension + 1], perm[0]
        transpose = g.op("Transpose", loop_output, perm_i=perm)
        squeeze = symbolic_helper._squeeze_helper(g, transpose, [0])

        return squeeze

    return symbolic_helper._unimplemented("Unfold", "input size not accessible")


@_onnx_symbolic("aten::tensordot")
@symbolic_helper.parse_args("v", "v", "is", "is", "v")
def tensordot(g: jit_utils.GraphContext, input_a, input_b, dims_a, dims_b, out=None):
    if out is not None:
        symbolic_helper._unimplemented(
            "Tensordot", "Out parameter is not supported for tensordot."
        )

    dim_count_a = symbolic_helper._get_tensor_rank(input_a)
    if dim_count_a is None:
        raise errors.SymbolicValueError(
            "Unsupported: ONNX export of tensordot for tensor(input_a) of unknown rank.",
            input_a,
        )

    dim_count_b = symbolic_helper._get_tensor_rank(input_b)
    if dim_count_b is None:
        raise errors.SymbolicValueError(
            "Unsupported: ONNX export of tensordot for tensor(input_b) of unknown rank.",
            input_b,
        )

    dims_a = [
        (dims_a[i] + dim_count_a) if (dims_a[i] < 0) else dims_a[i]
        for i in range(len(dims_a))
    ]
    dims_b = [
        (dims_b[i] + dim_count_b) if (dims_b[i] < 0) else dims_b[i]
        for i in range(len(dims_b))
    ]

    left_dims_a = [i for i in range(dim_count_a) if (i not in dims_a)]
    left_dims_b = [i for i in range(dim_count_b) if (i not in dims_b)]

    new_input_a = opset9.permute(g, input_a, left_dims_a + dims_a)
    new_input_b = opset9.permute(g, input_b, dims_b + left_dims_b)

    input_shape = g.op("Shape", new_input_a)
    left_sizes_a = symbolic_helper._slice_helper(
        g, input_shape, axes=[0], starts=[0], ends=[len(left_dims_a)]
    )
    shape_sizes = [
        left_sizes_a,
        g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long)),
    ]
    output_a = opset9._reshape_from_tensor(g, new_input_a, shape_sizes)

    input_shape = g.op("Shape", output_a)
    slices = symbolic_helper._slice_helper(
        g, input_shape, axes=[0], starts=[-1], ends=[sys.maxsize]
    )
    shape_sizes = [
        g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long)),
        slices,
    ]
    output_a = opset9._reshape_from_tensor(g, new_input_a, shape_sizes)

    input_shape = g.op("Shape", new_input_b)
    left_sizes_b = symbolic_helper._slice_helper(
        g, input_shape, axes=[0], starts=[len(dims_b)], ends=[sys.maxsize]
    )
    slices = symbolic_helper._slice_helper(
        g, input_shape, axes=[0], starts=[0], ends=[len(dims_b)]
    )
    shape_sizes = [
        slices,
        g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long)),
    ]
    output_b = opset9._reshape_from_tensor(g, new_input_b, shape_sizes)

    input_shape = g.op("Shape", output_b)
    slices = symbolic_helper._slice_helper(
        g, input_shape, axes=[0], starts=[-1], ends=[sys.maxsize]
    )
    shape_sizes = [
        g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long)),
        slices,
    ]
    output_b = opset9._reshape_from_tensor(g, new_input_b, shape_sizes)

    output = einsum(g, "ij,jk->ik", g.op("prim::ListConstruct", *[output_a, output_b]))

    shape_sizes = [left_sizes_a, left_sizes_b]
    return opset9._reshape_from_tensor(g, output, shape_sizes)
<|endoftext|># mypy: allow-untyped-defs
"""This file exports ONNX ops for opset 16.

Note [ONNX Operators that are added/updated in opset 16]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-16-of-the-default-onnx-operator-set
New operators:
    GridSample https://github.com/onnx/onnx/pull/3557

Updated operators:
    Identity
    If
    LeakyRelu
    Loop
    PRelu
    RoiAlign
    Scan
    ScatterElements
    ScatterND
    Where
    GreaterOrEqual
    LessOrEqual
"""

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in README.md

import functools

import torch
from torch.nn.functional import (
    GRID_SAMPLE_INTERPOLATION_MODES,
    GRID_SAMPLE_PADDING_MODES,
)
from torch.onnx import errors
from torch.onnx._internal.torchscript_exporter import (
    _type_utils,
    jit_utils,
    registration,
    symbolic_helper,
    utils,
)


_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=16)


# note (mkozuki): Why `grid_sampler` instead of `grid_sample`?
# Because `torch.nn.functional.grid_sample` calls `torch.grid_sampler`.
@_onnx_symbolic("aten::grid_sampler")
@symbolic_helper.parse_args("v", "v", "i", "i", "b")
def grid_sampler(
    g: jit_utils.GraphContext,
    input,
    grid,
    mode_enum,
    padding_mode_enum,
    align_corners,
):
    # Check the input and grid tensor rank beforehand.
    if symbolic_helper._get_tensor_rank(input) == 5:
        return symbolic_helper._onnx_unsupported("GridSample with 5D volumetric input")
    mode_s = {v: k for k, v in GRID_SAMPLE_INTERPOLATION_MODES.items()}[mode_enum]  # type: ignore[call-arg]
    padding_mode_s = {v: k for k, v in GRID_SAMPLE_PADDING_MODES.items()}[  # type: ignore[call-arg]
        padding_mode_enum
    ]
    return g.op(
        "GridSample",
        input,
        grid,
        align_corners_i=int(align_corners),
        mode_s=mode_s,
        padding_mode_s=padding_mode_s,
    )


@_onnx_symbolic("aten::scatter_add")
@symbolic_helper.parse_args("v", "i", "v", "v")
def scatter_add(g: jit_utils.GraphContext, self, dim, index, src):
    src_type = _type_utils.JitScalarType.from_value(
        src, _type_utils.JitScalarType.UNDEFINED
    )
    src_sizes = symbolic_helper._get_tensor_sizes(src)
    index_sizes = symbolic_helper._get_tensor_sizes(index)

    if len(src_sizes) != len(index_sizes):
        return symbolic_helper._unimplemented(
            "scatter_add",
            f"`index` ({index_sizes}) should have the same dimensionality as `src` ({src_sizes})",
        )

    # PyTorch only allows index shape <= src shape, so we can only consider
    # taking index as subset size to src, like PyTorch does. When sizes for src
    # and index are not matched or there are dynamic axes, we take index shape to
    # slice src to accommodate.
    if src_sizes != index_sizes or None in index_sizes:
        adjusted_shape = g.op("Shape", index)
        starts = g.op("Constant", value_t=torch.tensor([0] * len(index_sizes)))
        src = g.op("Slice", src, starts, adjusted_shape)

    src = symbolic_helper._maybe_get_scalar(src)
    if symbolic_helper._is_value(src):
        return g.op("ScatterElements", self, index, src, axis_i=dim, reduction_s="add")
    else:
        # Check if scalar "src" has same type as self (PyTorch allows different
        # type for scalar src (but not when src is tensor)). If not, insert Cast node.
        if _type_utils.JitScalarType.from_value(self) != src_type:
            src = g.op(
                "Cast",
                src,
                to_i=_type_utils.JitScalarType.from_value(self).onnx_type(),
            )

        return g.op(
            "ScatterElements",
            self,
            index,
            src,
            axis_i=dim,
            reduction_s="add",
        )


@_onnx_symbolic("aten::scatter_reduce")
@symbolic_helper.parse_args("v", "i", "v", "v", "s", "b")
def scatter_reduce(
    g: jit_utils.GraphContext,
    self: torch._C.Value,
    dim: int,
    index: torch._C.Value,
    src: torch._C.Value,
    reduce: str,
    include_self: bool,
):
    if reduce == "mean":
        raise errors.OnnxExporterError(
            "ONNX does not support mean reduction for scatter_reduce"
        )
    if not include_self:
        raise errors.OnnxExporterError(
            "ONNX does not support include_self=False for scatter_reduce"
        )

    reduce_mode = {  # convert torch string name to onnx string name
        "mean": "none",  # 'mean' doesn't support in ONNX 1.14 definition
        "sum": "add",
        "prod": "mul",
        "amin": "min",
        "amax": "max",
    }
    onnx_reduce = reduce_mode[reduce]

    self_rank = g.op("Size", g.op("Shape", self))

    # if self_rank == 0:  # assert (index_rank == 0 and rank_src == 0)
    self_rank_is_zero = g.op(
        "Equal", self_rank, g.op("Constant", value_t=torch.tensor(0, dtype=torch.int64))
    )
    if_op, (if_context, else_context), _ = jit_utils.add_op_with_blocks(
        g, "If", self_rank_is_zero, n_blocks=2, outputs=3
    )
    neg_1 = if_context.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))

    self_reshape = if_context.op("Reshape", self, neg_1)
    utils._add_output_to_block(if_context.block, self_reshape)
    index_reshape = if_context.op("Reshape", index, neg_1)
    utils._add_output_to_block(if_context.block, index_reshape)
    src_reshape = if_context.op("Reshape", src, neg_1)
    utils._add_output_to_block(if_context.block, src_reshape)

    self_identity = else_context.op("Identity", self)
    utils._add_output_to_block(else_context.block, self_identity)
    index_identitye = else_context.op("Identity", index)
    utils._add_output_to_block(else_context.block, index_identitye)
    src_identity = else_context.op("Identity", src)
    utils._add_output_to_block(else_context.block, src_identity)

    result = g.op("ScatterElements", *if_op, axis_i=dim, reduction_s=onnx_reduce)

    # if self_rank == 0:
    if_op, (if_context, else_context), _ = jit_utils.add_op_with_blocks(
        g, "If", self_rank_is_zero, n_blocks=2, outputs=1
    )
    result_squeezed = if_context.op("Squeeze", result)
    utils._add_output_to_block(if_context.block, result_squeezed)
    result_identity = else_context.op("Identity", result)
    utils._add_output_to_block(else_context.block, result_identity)
    result_final = if_op.node().output()

    return result_final
<|endoftext|># mypy: allow-untyped-defs
"""Utilities for manipulating the onnx and onnx-script dependencies and ONNX proto."""

from __future__ import annotations

import glob
import os
import shutil
from typing import Any, TYPE_CHECKING

import torch
import torch.serialization
from torch.onnx import errors
from torch.onnx._internal.torchscript_exporter import jit_utils, registration


if TYPE_CHECKING:
    import io
    from collections.abc import Mapping


def export_as_test_case(
    model_bytes: bytes, inputs_data, outputs_data, name: str, dir: str
) -> str:
    """Export an ONNX model as a self contained ONNX test case.

    The test case contains the model and the inputs/outputs data. The directory structure
    is as follows:

    dir
    \u251c\u2500\u2500 test_<name>
    \u2502   \u251c\u2500\u2500 model.onnx
    \u2502   \u2514\u2500\u2500 test_data_set_0
    \u2502       \u251c\u2500\u2500 input_0.pb
    \u2502       \u251c\u2500\u2500 input_1.pb
    \u2502       \u251c\u2500\u2500 output_0.pb
    \u2502       \u2514\u2500\u2500 output_1.pb

    Args:
        model_bytes: The ONNX model in bytes.
        inputs_data: The inputs data, nested data structure of numpy.ndarray.
        outputs_data: The outputs data, nested data structure of numpy.ndarray.

    Returns:
        The path to the test case directory.
    """
    try:
        import onnx
    except ImportError as exc:
        raise ImportError(
            "Export test case to ONNX format failed: Please install ONNX."
        ) from exc

    test_case_dir = os.path.join(dir, "test_" + name)
    os.makedirs(test_case_dir, exist_ok=True)
    _export_file(
        model_bytes,
        os.path.join(test_case_dir, "model.onnx"),
        {},
    )
    data_set_dir = os.path.join(test_case_dir, "test_data_set_0")
    if os.path.exists(data_set_dir):
        shutil.rmtree(data_set_dir)
    os.makedirs(data_set_dir)

    proto = onnx.load_model_from_string(model_bytes)  # type: ignore[attr-defined]

    for i, (input_proto, input) in enumerate(zip(proto.graph.input, inputs_data)):
        export_data(input, input_proto, os.path.join(data_set_dir, f"input_{i}.pb"))
    for i, (output_proto, output) in enumerate(zip(proto.graph.output, outputs_data)):
        export_data(output, output_proto, os.path.join(data_set_dir, f"output_{i}.pb"))

    return test_case_dir


def load_test_case(dir: str) -> tuple[bytes, Any, Any]:
    """Load a self contained ONNX test case from a directory.

    The test case must contain the model and the inputs/outputs data. The directory structure
    should be as follows:

    dir
    \u251c\u2500\u2500 test_<name>
    \u2502   \u251c\u2500\u2500 model.onnx
    \u2502   \u2514\u2500\u2500 test_data_set_0
    \u2502       \u251c\u2500\u2500 input_0.pb
    \u2502       \u251c\u2500\u2500 input_1.pb
    \u2502       \u251c\u2500\u2500 output_0.pb
    \u2502       \u2514\u2500\u2500 output_1.pb

    Args:
        dir: The directory containing the test case.

    Returns:
        model_bytes: The ONNX model in bytes.
        inputs: the inputs data, mapping from input name to numpy.ndarray.
        outputs: the outputs data, mapping from output name to numpy.ndarray.
    """
    try:
        import onnx
        from onnx import numpy_helper  # type: ignore[attr-defined]
    except ImportError as exc:
        raise ImportError(
            "Load test case from ONNX format failed: Please install ONNX."
        ) from exc

    with open(os.path.join(dir, "model.onnx"), "rb") as f:
        model_bytes = f.read()

    test_data_dir = os.path.join(dir, "test_data_set_0")

    inputs = {}
    input_files = glob.glob(os.path.join(test_data_dir, "input_*.pb"))
    for input_file in input_files:
        tensor = onnx.load_tensor(input_file)  # type: ignore[attr-defined]
        inputs[tensor.name] = numpy_helper.to_array(tensor)
    outputs = {}
    output_files = glob.glob(os.path.join(test_data_dir, "output_*.pb"))
    for output_file in output_files:
        tensor = onnx.load_tensor(output_file)  # type: ignore[attr-defined]
        outputs[tensor.name] = numpy_helper.to_array(tensor)

    return model_bytes, inputs, outputs


def export_data(data, value_info_proto, f: str) -> None:
    """Export data to ONNX protobuf format.

    Args:
        data: The data to export, nested data structure of numpy.ndarray.
        value_info_proto: The ValueInfoProto of the data. The type of the ValueInfoProto
            determines how the data is stored.
        f: The file to write the data to.
    """
    try:
        from onnx import numpy_helper  # type: ignore[attr-defined]
    except ImportError as exc:
        raise ImportError(
            "Export data to ONNX format failed: Please install ONNX."
        ) from exc

    with open(f, "wb") as opened_file:
        if value_info_proto.type.HasField("map_type"):
            opened_file.write(
                numpy_helper.from_dict(data, value_info_proto.name).SerializeToString()
            )
        elif value_info_proto.type.HasField("sequence_type"):
            opened_file.write(
                numpy_helper.from_list(data, value_info_proto.name).SerializeToString()
            )
        elif value_info_proto.type.HasField("optional_type"):
            opened_file.write(
                numpy_helper.from_optional(
                    data, value_info_proto.name
                ).SerializeToString()
            )
        else:
            if not value_info_proto.type.HasField("tensor_type"):
                raise AssertionError("Expected tensor_type field to be set")
            opened_file.write(
                numpy_helper.from_array(data, value_info_proto.name).SerializeToString()
            )


def _export_file(
    model_bytes: bytes,
    f: io.BytesIO | str,
    export_map: Mapping[str, bytes],
) -> None:
    """export/write model bytes into directory/protobuf/zip"""
    if len(export_map) != 0:
        raise AssertionError(f"export_map must be empty, got {len(export_map)} items")
    with torch.serialization._open_file_like(f, "wb") as opened_file:
        opened_file.write(model_bytes)


def _add_onnxscript_fn(
    model_bytes: bytes,
    custom_opsets: Mapping[str, int],
) -> bytes:
    """Insert model-included custom onnx-script function into ModelProto"""
    try:
        import onnx
    except ImportError as e:
        raise errors.OnnxExporterError("Module onnx is not installed!") from e

    # For > 2GB model, onnx.load_fromstring would fail. However, because
    # in _export_onnx, the tensors should be saved separately if the proto
    # size > 2GB, and if it for some reason did not, the model would fail on
    # serialization anyway in terms of the protobuf limitation. So we don't
    # need to worry about > 2GB model getting here.
    model_proto = onnx.load_model_from_string(model_bytes)  # type: ignore[attr-defined]

    # Iterate graph nodes to insert only the included custom
    # function_proto into model_proto
    onnx_function_list = []  # type: ignore[var-annotated]
    included_node_func: set[str] = set()
    # onnx_function_list and included_node_func are expanded in-place
    _find_onnxscript_op(
        model_proto.graph, included_node_func, custom_opsets, onnx_function_list
    )

    if onnx_function_list:
        model_proto.functions.extend(onnx_function_list)
        model_bytes = model_proto.SerializeToString()
    return model_bytes


def _find_onnxscript_op(
    graph_proto,
    included_node_func: set[str],
    custom_opsets: Mapping[str, int],
    onnx_function_list: list,
):
    """Recursively iterate ModelProto to find ONNXFunction op as it may contain control flow Op."""
    for node in graph_proto.node:
        node_kind = node.domain + "::" + node.op_type
        # Recursive needed for control flow nodes: IF/Loop which has inner graph_proto
        for attr in node.attribute:
            if attr.g is not None:
                _find_onnxscript_op(
                    attr.g, included_node_func, custom_opsets, onnx_function_list
                )
        # Only custom Op with ONNX function and aten with symbolic_fn should be found in registry
        onnx_function_group = registration.registry.get_function_group(node_kind)
        # Ruled out corner cases: onnx/prim in registry
        if (
            node.domain
            and not jit_utils.is_aten(node.domain)
            and not jit_utils.is_prim(node.domain)
            and not jit_utils.is_onnx(node.domain)
            and onnx_function_group is not None
            and node_kind not in included_node_func
        ):
            specified_version = custom_opsets.get(node.domain, 1)
            onnx_fn = onnx_function_group.get(specified_version)
            if onnx_fn is not None:
                if hasattr(onnx_fn, "to_function_proto"):
                    onnx_function_proto = onnx_fn.to_function_proto()  # type: ignore[attr-defined]
                    onnx_function_list.append(onnx_function_proto)
                    included_node_func.add(node_kind)
                continue

            raise errors.UnsupportedOperatorError(
                node_kind,
                specified_version,
                onnx_function_group.get_min_supported()
                if onnx_function_group
                else None,
            )
    return onnx_function_list, included_node_func
<|endoftext|># mypy: allow-untyped-defs
"""Module for handling symbolic function registration."""

import warnings
from collections.abc import Callable, Collection, Sequence
from typing import Generic, TypeVar
from typing_extensions import ParamSpec

from torch.onnx import _constants, errors


OpsetVersion = int


def _dispatch_opset_version(
    target: OpsetVersion, registered_opsets: Collection[OpsetVersion]
) -> OpsetVersion | None:
    """Finds the registered opset given a target opset version and the available opsets.

    Args:
        target: The target opset version.
        registered_opsets: The available opsets.

    Returns:
        The registered opset version.
    """
    if not registered_opsets:
        return None

    descending_registered_versions = sorted(registered_opsets, reverse=True)
    # Linear search for the opset version, which is fine since the number of opset
    # versions is small.

    if target >= _constants.ONNX_BASE_OPSET:
        # Always look down toward opset 1 when the target is >= ONNX_BASE_OPSET (opset 9).
        # When a custom op is register at opset 1, we want to be able to discover it as a
        # fallback for all opsets >= ONNX_BASE_OPSET.
        for version in descending_registered_versions:
            if version <= target:
                return version
        return None

    # target < opset 9. This is the legacy behavior to support opset 7 and opset 8.
    # for caffe2 support. We search up toward opset 9.
    for version in reversed(descending_registered_versions):
        # Count back up until _constants.ONNX_BASE_OPSET
        if target <= version <= _constants.ONNX_BASE_OPSET:
            return version

    return None


_K = TypeVar("_K")
_V = TypeVar("_V")
_R = TypeVar("_R")
_P = ParamSpec("_P")


class OverrideDict(Collection[_K], Generic[_K, _V]):
    """A dictionary that merges built-in and custom symbolic functions.

    It supports overriding and un-overriding built-in symbolic functions with custom
    ones.
    """

    def __init__(self) -> None:
        self._base: dict[_K, _V] = {}
