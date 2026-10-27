_ONNX_TO_SCALAR_TYPE = {v: k for k, v in _SCALAR_TYPE_TO_ONNX.items()}

# source of truth is
# https://github.com/pytorch/pytorch/blob/master/torch/csrc/utils/tensor_dtypes.cpp
_SCALAR_TYPE_TO_DTYPE = {
    JitScalarType.BOOL: torch.bool,
    JitScalarType.UINT8: torch.uint8,
    JitScalarType.INT8: torch.int8,
    JitScalarType.INT16: torch.short,
    JitScalarType.INT: torch.int,
    JitScalarType.INT64: torch.int64,
    JitScalarType.HALF: torch.half,
    JitScalarType.FLOAT: torch.float,
    JitScalarType.DOUBLE: torch.double,
    JitScalarType.COMPLEX32: torch.complex32,
    JitScalarType.COMPLEX64: torch.complex64,
    JitScalarType.COMPLEX128: torch.complex128,
    JitScalarType.QINT8: torch.qint8,
    JitScalarType.QUINT8: torch.quint8,
    JitScalarType.QINT32: torch.qint32,
    JitScalarType.BFLOAT16: torch.bfloat16,
    JitScalarType.FLOAT8E5M2: torch.float8_e5m2,
    JitScalarType.FLOAT8E4M3FN: torch.float8_e4m3fn,
    JitScalarType.FLOAT8E5M2FNUZ: torch.float8_e5m2fnuz,
    JitScalarType.FLOAT8E4M3FNUZ: torch.float8_e4m3fnuz,
}

_DTYPE_TO_SCALAR_TYPE = {v: k for k, v in _SCALAR_TYPE_TO_DTYPE.items()}
<|endoftext|># mypy: allow-untyped-defs
"""The ONNX verification module provides a set of tools to verify the correctness of ONNX models."""

from __future__ import annotations


__all__ = [
    "OnnxBackend",
    "VerificationOptions",
    "verify",
]

import contextlib
import copy
import dataclasses
import enum
import io
import os
import tempfile
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

import torch
import torch._C._onnx as _C_onnx
from torch.onnx._internal.torchscript_exporter import utils
from torch.types import Number


# Everything below are deprecated ##############################################

_ORT_PROVIDERS = ("CPUExecutionProvider",)

_NumericType = Number | torch.Tensor | np.ndarray
_ModelType = torch.nn.Module | torch.jit.ScriptModule
_InputArgsType = torch.Tensor | tuple[Any, ...]
_InputKwargsType = Mapping[str, Any]
_OutputsType = Sequence[_NumericType] | Sequence


class OnnxBackend(enum.Enum):
    """Enum class for ONNX backend used for export verification.

    .. deprecated:: 2.7
        Consider using ``torch.onnx.export(..., dynamo=True)`` and use the returned
        ``ONNXProgram`` to test the ONNX model.
    """

    REFERENCE = "ONNXReferenceEvaluator"
    ONNX_RUNTIME_CPU = "CPUExecutionProvider"
    ONNX_RUNTIME_CUDA = "CUDAExecutionProvider"


@dataclasses.dataclass
class VerificationOptions:
    """Options for ONNX export verification.

    .. deprecated:: 2.7
        Consider using ``torch.onnx.export(..., dynamo=True)`` and use the returned
        ``ONNXProgram`` to test the ONNX model.

    Attributes:
        flatten: If True, unpack nested list/tuple/dict inputs into a flattened list of
            Tensors for ONNX. Set this to False if nested structures are to be preserved
            for ONNX, which is usually the case with exporting ScriptModules. Default True.
        ignore_none: Whether to ignore None type in torch output, which is usually the
            case with tracing. Set this to False, if torch output should keep None type,
            which is usually the case with exporting ScriptModules. Default to True.
        check_shape: Whether to check the shapes between PyTorch and ONNX Runtime outputs
            are exactly the same. Set this to False to allow output shape broadcasting.
            Default to True.
        check_dtype: Whether to check the dtypes between PyTorch and ONNX Runtime outputs
            are consistent. Default to True.
        backend: ONNX backend for verification. Default to OnnxBackend.ONNX_RUNTIME_CPU.
        rtol: relative tolerance in comparison between ONNX and PyTorch outputs.
        atol: absolute tolerance in comparison between ONNX and PyTorch outputs.
        remained_onnx_input_idx: If provided, only the specified inputs will be passed
            to the ONNX model. Supply a list when there are unused inputs in the model.
            Since unused inputs will be removed in the exported ONNX model, supplying
            all inputs will cause an error on unexpected inputs. This parameter tells
            the verifier which inputs to pass into the ONNX model.
        acceptable_error_percentage: acceptable percentage of element mismatches in comparison.
            It should be a float of value between 0.0 and 1.0.
    """

    flatten: bool = True
    ignore_none: bool = True
    check_shape: bool = True
    check_dtype: bool = True
    backend: OnnxBackend = OnnxBackend.ONNX_RUNTIME_CPU
    rtol: float = 1e-3
    atol: float = 1e-7
    remained_onnx_input_idx: Sequence[int] | None = None
    acceptable_error_percentage: float | None = None


def _flatten_tuples(elem):
    flattened = []
    for t in elem:
        if isinstance(t, tuple):
            flattened.extend(_flatten_tuples(t))
        else:
            flattened.append(t)
    return flattened


def _to_numpy(elem) -> list | npt.NDArray:
    if isinstance(elem, torch.Tensor):
        if elem.requires_grad:
            return elem.detach().cpu().numpy()
        else:
            return elem.cpu().numpy()
    elif isinstance(elem, (list, tuple)):
        return [_to_numpy(inp) for inp in elem]
    elif isinstance(elem, (bool, int, float)):
        return np.array(elem)
    elif isinstance(elem, dict):
        flattened = []
        for k in elem:
            flattened.extend([_to_numpy(k), _to_numpy(elem[k])])
        return flattened
    return elem


def _inline_flatten_list(inputs, res_list) -> list:
    for i in inputs:
        res_list.append(i) if not isinstance(
            i, (list, tuple)
        ) else _inline_flatten_list(i, res_list)
    return res_list


def _unpack_to_numpy(values, cast_onnx_accepted=True) -> list:
    value_unpacked = []
    for value in values:
        value_unpacked.extend(
            utils.unpack_quantized_tensor(value, cast_onnx_accepted=cast_onnx_accepted)
        )
    return [_to_numpy(v) for v in value_unpacked]


def _run_onnx(onnx_session, inputs) -> _OutputsType:
    kw_inputs = {}
    if inputs and isinstance(inputs[-1], dict):
        kw_inputs = inputs[-1]
        inputs = inputs[:-1]
    inputs = _unpack_to_numpy(_flatten_tuples(inputs))
    ort_inputs = {}
    for input_name, input in kw_inputs.items():
        ort_inputs[input_name] = _to_numpy(input)
    inputs = _to_numpy(inputs)
    if hasattr(onnx_session, "get_inputs"):
        # onnxruntime.InferenceSession
        input_names = [i.name for i in onnx_session.get_inputs()]
    elif hasattr(onnx_session, "input_names"):
        # onnx.reference.ReferenceEvaluator
        input_names = onnx_session.input_names
    else:
        raise ValueError(f"Unknown ONNX backend type: {type(onnx_session)}.")

    for i, input in enumerate(inputs):
        if i == len(input_names) or input_names[i] in ort_inputs:
            raise ValueError(
                f"got too many positional inputs. inputs: {inputs}. kw_inputs: {kw_inputs}. "
                f"input names: {input_names}."
            )
        ort_inputs[input_names[i]] = input
    onnx_outs = onnx_session.run(None, ort_inputs)
    return onnx_outs


def _ort_session(
    model: str | io.BytesIO, ort_providers: Sequence[str] = _ORT_PROVIDERS
):
    try:
        import onnxruntime  # type: ignore[import]
    except ImportError as e:
        raise ImportError("onnxruntime is required for export verification.") from e

    if ort_providers is None:
        ort_providers = _ORT_PROVIDERS

    session_options = onnxruntime.SessionOptions()
    # suppress ort warnings.
    # 0:Verbose, 1:Info, 2:Warning. 3:Error, 4:Fatal. Default is 2.
    session_options.log_severity_level = 3
    ort_session = onnxruntime.InferenceSession(
        model if isinstance(model, str) else model.getvalue(),
        session_options,
        providers=ort_providers,
    )
    return ort_session


def _onnx_backend_session(model: str | io.BytesIO, backend: OnnxBackend):
    if backend == OnnxBackend.REFERENCE:
        raise NotImplementedError
    elif backend in {OnnxBackend.ONNX_RUNTIME_CPU, OnnxBackend.ONNX_RUNTIME_CUDA}:
        onnx_session = _ort_session(model, (backend.value,))
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    return onnx_session


def _compare_onnx_pytorch_outputs_in_np(
    onnx_outs: _OutputsType,
    pt_outs: _OutputsType,
    options: VerificationOptions,
) -> None:
    if len(onnx_outs) != len(pt_outs):
        raise AssertionError(
            f"Number of outputs differ ONNX runtime: ({len(onnx_outs)}) PyTorch: ({len(pt_outs)})"
        )
    acceptable_error_percentage = options.acceptable_error_percentage
    if acceptable_error_percentage and (
        acceptable_error_percentage > 1.0 or acceptable_error_percentage < 0.0
    ):
        raise ValueError(
            "If set, acceptable_error_percentage should be between 0.0 and 1.0"
        )

    for ort_out, pt_out in zip(onnx_outs, pt_outs):
        try:
            # TODO: Remove `check_shape` option once every shape inconsistent issue is addressed.
            if not options.check_shape:
                # Allow different but broadcastable output shapes.
                ort_out, pt_out = np.broadcast_arrays(ort_out, pt_out)
            torch.testing.assert_close(
                ort_out,
                pt_out,
                rtol=options.rtol,
                atol=options.atol,
                check_dtype=options.check_dtype,
                equal_nan=True,
            )
        except AssertionError as e:
            if acceptable_error_percentage:
                error_percentage = 1 - np.sum(
                    np.isclose(ort_out, pt_out, rtol=options.rtol, atol=options.atol)
                ) / np.prod(ort_out.shape)  # pyrefly: ignore [missing-attribute]
                if error_percentage <= acceptable_error_percentage:
                    warnings.warn(
                        f"Suppressed AssertionError:\n{e}.\n"
                        f"Error percentage {error_percentage} "
                        f"within acceptable range {acceptable_error_percentage}.",
                        stacklevel=2,
                    )
                    continue
            # pyrefly: ignore [missing-attribute]
            if ort_out.dtype == np.uint8 or ort_out.dtype == np.int8:
                warnings.warn("ONNX output is quantized", stacklevel=2)
            # pyrefly: ignore [missing-attribute]
            if pt_out.dtype == np.uint8 or pt_out.dtype == np.int8:
                warnings.warn("PyTorch output is quantized", stacklevel=2)
            raise


def _compare_onnx_pytorch_outputs(
    onnx_outs: _OutputsType,
    pt_outs: Any,
    options: VerificationOptions,
) -> None:
    """
    Compare ONNX and PyTorch outputs.

    Args:
        onnx_outs: outputs from ONNX backend.
        pt_outs: outputs from PyTorch.
        options: options for verification.

    Raises:
        AssertionError: if outputs from ONNX model and PyTorch model are not
            equal up to specified precision.
        ValueError: if arguments provided are invalid.
    """
    if options.ignore_none:
        # torch.jit._flatten filters None type
        pt_outs, _ = torch.jit._flatten(pt_outs)
    else:
        pt_outs = _inline_flatten_list([pt_outs], [])
    pt_outs_np = _unpack_to_numpy(pt_outs, cast_onnx_accepted=False)
    onnx_outs = _inline_flatten_list(onnx_outs, [])
    _compare_onnx_pytorch_outputs_in_np(onnx_outs, pt_outs_np, options)


def _prepare_input_for_pytorch(args, kwargs):
    """Prepare input for PyTorch model execution.

    Any future changes/formatting to the input before dispatching to the PyTorch
    model should be made in this function.

    Args:
        args: positional arguments for PyTorch model forward method.
        kwargs: keyword arguments for PyTorch model forward method.

    Returns:
        args: positional arguments for PyTorch model forward method.
        kwargs: keyword arguments for PyTorch model forward method.
    """
    if isinstance(args, (torch.Tensor, dict)):
        args = (args,)
    # In-place operators will update input tensor data as well.
    # Thus inputs are replicated before every forward call.
    args = copy.deepcopy(args)
    if kwargs:
        kwargs = copy.deepcopy(kwargs)
    else:
        kwargs = {}
    return args, kwargs


def _prepare_input_for_export(args, kwargs):
    """Prepare input for ONNX model export.

    Any future changes/formatting to the input before dispatching to the
    :func:`torch.onnx.export` api should be made in this function.

    Args:
        args: positional arguments for PyTorch model forward method.
        kwargs: keyword arguments for PyTorch model forward method.

    Returns:
        onnx_inputs: positional arguments for ONNX model export, as `args` in
            :func:`torch.onnx.export`.
    """
    args, kwargs = _prepare_input_for_pytorch(args, kwargs)
    if not kwargs and len(args) > 0 and isinstance(args[-1], dict):
        onnx_inputs = args + ({},)
    elif kwargs:
        onnx_inputs = args + (kwargs,)
    else:
        onnx_inputs = args
    return onnx_inputs


def _prepare_input_for_onnx(
    args, kwargs, remained_onnx_input_idx: Sequence[int] | None, flatten: bool
):
    """Prepare input for ONNX model execution in ONNX backend.

    Any future changes/formatting to the input before dispatching to the ONNX backend
    run should be made in this function.

    Args:
        args: positional arguments for PyTorch model forward method.
        kwargs: keyword arguments for PyTorch model forward method.
        remained_onnx_input_idx: indices of inputs to be used for ONNX model execution.
        flatten: whether to flatten the input before dispatching to the ONNX model execution.

    Returns:
        onnx_inputs: positional arguments for ONNX model execution in ONNX backend.
    """
    onnx_inputs = _prepare_input_for_export(args, kwargs)
    if flatten:
        onnx_inputs, _ = torch.jit._flatten(onnx_inputs)
    elif onnx_inputs and onnx_inputs[-1] == {}:
        # Handle empty kwargs (normally removed by flatten).
        onnx_inputs = onnx_inputs[:-1]
    if remained_onnx_input_idx is not None:
        return [onnx_inputs[i] for i in remained_onnx_input_idx]
    else:
        return onnx_inputs


def _try_clone_model(model):
    """Used for preserving original model in case forward mutates model states."""
    try:
        return copy.deepcopy(model)
    except Exception:
        warnings.warn(
            "Failed to clone model. Model state might be mutated during verification.",
            stacklevel=2,
        )
        return model


def _compare_onnx_pytorch_model(
    pt_model: _ModelType,
    onnx_model_f: str | io.BytesIO,
    input_args: _InputArgsType,
    input_kwargs: _InputKwargsType | None,
    additional_test_inputs: Sequence[_InputArgsType] | None,
    options: VerificationOptions,
) -> None:
    """Compare outputs from ONNX model runs with outputs from PyTorch model runs.

    Args:
        pt_model: PyTorch model.
        onnx_model_f: ONNX model file path or file-like object.
        input_args: positional arguments for PyTorch model forward method.
        input_kwargs: keyword arguments for PyTorch model forward method.
        additional_test_inputs: additional positional arguments for PyTorch model
            forward method.
        options: options for verification.

    Raises:
        AssertionError: if outputs from ONNX model and PyTorch model are not
            equal up to specified precision.
    """
    onnx_session = _onnx_backend_session(onnx_model_f, options.backend)

    def compare_onnx_pytorch_model_with_input(input_args, input_kwargs) -> None:
        pt_args, pt_kwargs = _prepare_input_for_pytorch(input_args, input_kwargs)
        # TODO: remove this and treat mutating model separately. See #77679
        pt_model_copy = _try_clone_model(pt_model)
        pt_outs = pt_model_copy(*pt_args, **pt_kwargs)

        onnx_inputs = _prepare_input_for_onnx(
            input_args, input_kwargs, options.remained_onnx_input_idx, options.flatten
        )

        onnx_outs = _run_onnx(onnx_session, onnx_inputs)

        _compare_onnx_pytorch_outputs(
            onnx_outs=onnx_outs,
            pt_outs=pt_outs,
            options=options,
        )

    compare_onnx_pytorch_model_with_input(input_args, input_kwargs)

    if additional_test_inputs:
        for test_input_args in additional_test_inputs:
            compare_onnx_pytorch_model_with_input(test_input_args, {})


def verify(
    model: _ModelType,
    input_args: _InputArgsType,
    input_kwargs: _InputKwargsType | None = None,
    do_constant_folding: bool = True,
    dynamic_axes: Mapping[str, Mapping[int, str] | Mapping[str, Sequence[int]]]
    | None = None,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    training: _C_onnx.TrainingMode = _C_onnx.TrainingMode.EVAL,
    opset_version: int | None = None,
    keep_initializers_as_inputs: bool = True,
    verbose: bool = False,
    fixed_batch_size: bool = False,
    use_external_data: bool = False,
    additional_test_inputs: Sequence[_InputArgsType] | None = None,
    options: VerificationOptions | None = None,
) -> None:
    """Verify model export to ONNX against original PyTorch model.

    .. deprecated:: 2.7
        Consider using ``torch.onnx.export(..., dynamo=True)`` and use the returned
        ``ONNXProgram`` to test the ONNX model.

    Args:
        model: See :func:`torch.onnx.export`.
        input_args: See :func:`torch.onnx.export`.
        input_kwargs: See :func:`torch.onnx.export`.
        do_constant_folding: See :func:`torch.onnx.export`.
        dynamic_axes: See :func:`torch.onnx.export`.
        input_names: See :func:`torch.onnx.export`.
        output_names: See :func:`torch.onnx.export`.
        training: See :func:`torch.onnx.export`.
        opset_version: See :func:`torch.onnx.export`.
        keep_initializers_as_inputs: See :func:`torch.onnx.export`.
        verbose: See :func:`torch.onnx.export`.
        fixed_batch_size: Legacy argument, used only by rnn test cases.
        use_external_data: Explicitly specify whether to export the model with external data.
        additional_test_inputs: List of tuples. Each tuple is a group of
            input arguments to test. Currently only ``*args`` are supported.
        options: A VerificationOptions object that controls the verification behavior.

    Raises:
        AssertionError: if outputs from ONNX model and PyTorch model are not
            equal up to specified precision.
        ValueError: if arguments provided are invalid.
    """
    if options is None:
        options = VerificationOptions()

    if training == torch.onnx.TrainingMode.TRAINING:
        model.train()
    elif training == torch.onnx.TrainingMode.EVAL:
        model.eval()
    with torch.no_grad(), contextlib.ExitStack() as stack:
        model_f: str | io.BytesIO = io.BytesIO()
        if use_external_data:
            tmpdir_path = stack.enter_context(tempfile.TemporaryDirectory())
            model_f = os.path.join(tmpdir_path, "model.onnx")

        inputs_for_export = _prepare_input_for_export(input_args, input_kwargs)

        # TODO(#77679): remove this and treat mutating model separately.
        model_copy = _try_clone_model(model)
        utils._export(
            model,
            inputs_for_export,
            model_f,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
            dynamic_axes=dynamic_axes,
            input_names=input_names,
            output_names=output_names,
            fixed_batch_size=fixed_batch_size,
            training=training,
            verbose=verbose,
        )

        _compare_onnx_pytorch_model(
            pt_model=model_copy,
            onnx_model_f=model_f,
            input_args=input_args,
            input_kwargs=input_kwargs,
            additional_test_inputs=additional_test_inputs,
            options=options,
        )
<|endoftext|># mypy: allow-untyped-defs
"""
Note [ONNX operators that are added/updated from opset 8 to opset 9]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
New operators:
    Compress
    ConstantOfShape
    EyeLike
    MaxUnpool
    OneHot
    Sinh
    Cosh
    Asinh
    Acosh
    Atanh
    Shrink
    IsNaN
    Sign
    Erf
    Scatter
    Where
    NonZero
    TfIdfVectorizer
    MeanVarianceNormalization

Updated operators:
    BatchNormalization: removed spatial attribute.
    Greater, Less, Constant, MatMul, PRelu, Gemm, Flatten: more data types{integers} supported.
    Cast: more data types{string} supported.
    Upsample: moved scales from attribute to input.
    Scan
"""

import functools
import warnings

import torch
from torch._C import _onnx as _C_onnx
from torch.onnx import errors
from torch.onnx._internal.torchscript_exporter import (
    _type_utils,
    jit_utils,
    registration,
    symbolic_helper,
    symbolic_opset9 as opset9,
)


_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=8)

block_listed_operators = (
    "nonzero",
    "where",
    "scatter",
    "scatter_add",
    "erf",
    "sign",
    "isnan",
    "gather",
    "arange",
    "masked_fill",
    "index_fill",
    "index_copy",
    "repeat_interleave",
    "any",
    "all",
)

for block_listed_op in block_listed_operators:
    _onnx_symbolic(f"aten::{block_listed_op}")(
        symbolic_helper._block_list_in_opset(block_listed_op)
    )


@_onnx_symbolic(
    "aten::upsample_nearest1d",
    decorate=[symbolic_helper._apply_params("upsample_nearest1d", 3, "nearest")],
)
@_onnx_symbolic(
    "aten::upsample_nearest2d",
    decorate=[symbolic_helper._apply_params("upsample_nearest2d", 4, "nearest")],
)
@_onnx_symbolic(
    "aten::upsample_nearest3d",
    decorate=[symbolic_helper._apply_params("upsample_nearest3d", 5, "nearest")],
)
@_onnx_symbolic(
    "aten::upsample_linear1d",
    decorate=[symbolic_helper._apply_params("upsample_linear1d", 3, "linear")],
)
@_onnx_symbolic(
    "aten::upsample_bilinear2d",
    decorate=[symbolic_helper._apply_params("upsample_bilinear2d", 4, "linear")],
)
@_onnx_symbolic(
    "aten::upsample_trilinear3d",
    decorate=[symbolic_helper._apply_params("upsample_trilinear3d", 5, "linear")],
)
def _interpolate(name, dim, interpolate_mode):
    def symbolic_fn(g, input, output_size, *args):
        scales, align_corners = symbolic_helper._get_interpolate_attributes(
            g, interpolate_mode, args
        )
        symbolic_helper._interpolate_warning(interpolate_mode)
        align_corners = symbolic_helper._maybe_get_scalar(align_corners)
        if align_corners:
            return symbolic_helper._unimplemented(name, "align_corners == True", input)
        output_size = symbolic_helper._maybe_get_const(output_size, "is")
        if symbolic_helper._is_value(output_size):
            return symbolic_helper._unimplemented(
                name, "torch._C.Value (output_size) indexing"
            )
        if scales is None:
            scales = [
                1.0
                if i < 2
                else float(output_size[-(dim - i)])
                / float(input.type().sizes()[-(dim - i)])
                for i in range(dim)
            ]
        return g.op("Upsample", input, mode_s=interpolate_mode, scales_f=scales)

    return symbolic_fn


@_onnx_symbolic("aten::__interpolate")
def __interpolate(
    g: jit_utils.GraphContext,
    input,
    size,
    scale_factor,
    mode,
    align_corners,
    recompute_scale_factor,
    antialias,
):
    align_corners = symbolic_helper._maybe_get_const(align_corners, "b")
    if not symbolic_helper._is_none(align_corners) and align_corners:
        return symbolic_helper._unimplemented("interpolate", "align_corners == True")

    if not symbolic_helper._is_none(scale_factor) and symbolic_helper._is_value(
        scale_factor
    ):
        return symbolic_helper._unimplemented(
            "interpolate", "dynamic scales in opset 8"
        )

    if not symbolic_helper._is_none(size) and symbolic_helper._is_value(size):
        return symbolic_helper._unimplemented("interpolate", "dynamic size in opset 8")

    scales, mode = symbolic_helper._interpolate_get_scales_and_mode(
        g, input, size, scale_factor, mode, align_corners
    )
    return g.op("Upsample", input, mode_s=mode, scales_f=scales)


# NOTE: We should create a wrapper for this kind of operation, after resolving the shape/type propagation
#       issue for "cast" operators. Some symbolic functions depend on shape information of input tensor, which
#       is lost after casting.
def _try_cast_integer_to_float(g: jit_utils.GraphContext, *args):
    floating_scalar_types = {
        _type_utils.JitScalarType.HALF,
        _type_utils.JitScalarType.FLOAT,
        _type_utils.JitScalarType.DOUBLE,
    }
    old_type = None
    # Cast the input tensor to Float if its scalarType is known and is not floating number.
    # If casting is performed, return the old scalarType, otherwise return None.
    arg0_type = _type_utils.JitScalarType.from_value(
        args[0], _type_utils.JitScalarType.UNDEFINED
    )
    if arg0_type != _type_utils.JitScalarType.UNDEFINED:
        old_type = arg0_type
        if old_type not in floating_scalar_types:
            old_type = old_type.scalar_name()  # type: ignore[assignment]
            args = tuple(
                g.op("Cast", arg, to_i=_C_onnx.TensorProtoDataType.FLOAT)
                for arg in args
            )
        else:
            return (None,) + args
    else:
        warnings.warn(
            "Only floating datatype is supported for these operators: "
            "{Greater, Less, MatMul, PRelu, Gemm, Flatten}. This might cause "
            "the onnx model to be incorrect, if inputs have integer datatypes.",
            stacklevel=2,
        )
    return (old_type,) + args


def _cast_to_type(g: jit_utils.GraphContext, input, to_type):
    if to_type is None:
        return input
    return g.op("Cast", input, to_i=symbolic_helper.cast_pytorch_to_onnx[to_type])


def _comparison_operator(g: jit_utils.GraphContext, input, other, op_name):
    other = symbolic_helper._maybe_get_scalar(other)
    other = symbolic_helper._if_scalar_type_as(other, input)
    _, input, other = _try_cast_integer_to_float(g, input, other)
    return g.op(op_name, input, other)


# NOTE: For symbolics {gt, lt, bmm, matmul, prelu, mm, addmm, view, flatten},
#       integer input type not supported in opset8. Cast to float if possible.
@_onnx_symbolic("aten::gt")
def gt(g: jit_utils.GraphContext, input, other):
    return _comparison_operator(g, input, other, "Greater")


@_onnx_symbolic("aten::lt")
def lt(g: jit_utils.GraphContext, input, other):
    return _comparison_operator(g, input, other, "Less")


@_onnx_symbolic("aten::bmm")
def bmm(g: jit_utils.GraphContext, self, other):
    if symbolic_helper._try_get_scalar_type(self):
        old_type, self, other = _try_cast_integer_to_float(g, self, other)
        return _cast_to_type(g, g.op("MatMul", self, other), old_type)
    else:
        return g.op("MatMul", self, other)


@_onnx_symbolic("aten::matmul")
def matmul(g: jit_utils.GraphContext, self, other):
    return bmm(g, self, other)


@_onnx_symbolic("aten::prelu")
def prelu(g: jit_utils.GraphContext, self, weight):
    self_rank = symbolic_helper._get_tensor_rank(self)
    weight_sizes = symbolic_helper._get_tensor_sizes(weight)
    if self_rank is not None and self_rank > 2:
        weight = g.op("Unsqueeze", weight, axes_i=list(range(1, self_rank - 1)))
    elif self_rank == 0 and weight_sizes == [1]:
        # self and weight are both scalar but weight has rank == 1, squeeze weight.
        weight = symbolic_helper._squeeze_helper(g, weight, [0])
    if symbolic_helper._try_get_scalar_type(self):
        old_type, self, weight = _try_cast_integer_to_float(g, self, weight)
        return _cast_to_type(g, g.op("PRelu", self, weight), old_type)
    else:
        return g.op("PRelu", self, weight)


@_onnx_symbolic("aten::mm")
def mm(g: jit_utils.GraphContext, self, other):
    # Create a dummy C tensor. Only needed for API purposes, the value is
    # since beta = 0
    scalar_type = symbolic_helper._try_get_scalar_type(self, other)
    if scalar_type is None:
        raise errors.SymbolicValueError(
            "mm can only operate on tensors with known types", self
        )
    zero_constant = g.op(
        "Constant",
        value_t=torch.tensor([0], dtype=scalar_type.dtype()),
    )

    if symbolic_helper._try_get_scalar_type(self):
        old_type, self, other, zero_constant = _try_cast_integer_to_float(
            g, self, other, zero_constant
        )
        return _cast_to_type(
            g,
            g.op("Gemm", self, other, zero_constant, beta_f=0.0, alpha_f=1.0),
            old_type,
        )
    return g.op("Gemm", self, other, zero_constant, beta_f=0.0, alpha_f=1.0)


@_onnx_symbolic("aten::addmm")
@symbolic_helper.parse_args("v", "v", "v", "t", "t")
def addmm(g: jit_utils.GraphContext, self, mat1, mat2, beta, alpha):
    if symbolic_helper._try_get_scalar_type(self):
        old_type, self, mat1, mat2 = _try_cast_integer_to_float(g, self, mat1, mat2)
        return _cast_to_type(
            g,
            g.op(
                "Gemm",
                mat1,
                mat2,
                self,
                beta_f=symbolic_helper._scalar(beta),
                alpha_f=symbolic_helper._scalar(alpha),
            ),
            old_type,
        )
    else:
        return g.op(
            "Gemm",
            mat1,
            mat2,
            self,
            beta_f=symbolic_helper._scalar(beta),
            alpha_f=symbolic_helper._scalar(alpha),
        )


@_onnx_symbolic("aten::flatten")
def flatten(g: jit_utils.GraphContext, input, start_dim, end_dim):
    start_dim_i = symbolic_helper._get_const(start_dim, "i", "start_dim")
    end_dim_i = symbolic_helper._get_const(end_dim, "i", "end_dim")

    dim = input.type().dim()
    if end_dim_i < 0:
        end_dim_i = dim + end_dim_i
    # use ONNX's Flatten operator for cases where the output shape is 2D
    if start_dim_i == 1 and end_dim_i == dim - 1:
        if symbolic_helper._try_get_scalar_type(input):
            old_type, input = _try_cast_integer_to_float(g, input)
            return _cast_to_type(
                g, g.op("Flatten", input, axis_i=start_dim_i), old_type
            )
        else:
            return g.op("Flatten", input, axis_i=start_dim_i)
    if start_dim_i == 0 and end_dim_i == dim - 2:
        if symbolic_helper._try_get_scalar_type(input):
            old_type, input = _try_cast_integer_to_float(g, input)
            return _cast_to_type(
                g, g.op("Flatten", input, axis_i=end_dim_i + 1), old_type
            )
        else:
            return g.op("Flatten", input, axis_i=end_dim_i + 1)

    return opset9.flatten(g, input, start_dim, end_dim)


def _constant_fill(g: jit_utils.GraphContext, sizes, dtype: int, const_value):
    if dtype is None:
        scalar_type = _type_utils.JitScalarType.FLOAT
    else:
        scalar_type = _type_utils.JitScalarType(dtype)
    if not scalar_type.dtype().is_floating_point:
        result = g.op(
            "ConstantFill",
            sizes,
            dtype_i=_type_utils.JitScalarType.FLOAT.onnx_type(),
            input_as_shape_i=1,
            value_f=const_value,
        )
        return g.op("Cast", result, to_i=scalar_type.onnx_type())
    else:
        return g.op(
            "ConstantFill",
            sizes,
            dtype_i=scalar_type.onnx_type(),
            input_as_shape_i=1,
            value_f=const_value,
        )


@_onnx_symbolic("aten::empty")
@symbolic_helper.parse_args("v", "i", "v", "v", "v", "v")
def empty(
    g: jit_utils.GraphContext,
    sizes,
    dtype,
    layout,
    device,
    pin_memory=False,
    memory_format=None,
):
    return zeros(g, sizes, dtype, layout, device, pin_memory)


@_onnx_symbolic("aten::empty_like")
@symbolic_helper.parse_args("v", "i", "v", "v", "v", "v")
def empty_like(
    g: jit_utils.GraphContext,
    input,
    dtype,
    layout,
    device,
    pin_memory=False,
    memory_format=None,
):
    return zeros_like(g, input, dtype, layout, device, pin_memory)


@_onnx_symbolic("aten::zeros")
@symbolic_helper.parse_args("v", "i", "v", "v", "v")
def zeros(g: jit_utils.GraphContext, sizes, dtype, layout, device, pin_memory=False):
    # NOTE: no way to set device and layout in ONNX, so we ignore it
    return _constant_fill(g, sizes, dtype, 0)


@_onnx_symbolic("aten::zeros_like")
@symbolic_helper.parse_args("v", "i", "v", "v", "v", "v")
def zeros_like(
    g: jit_utils.GraphContext,
    input,
    dtype,
    layout,
    device,
    pin_memory=False,
    memory_format=None,
):
    shape = g.op("Shape", input)
    return _constant_fill(g, shape, dtype, 0)


@_onnx_symbolic("aten::ones")
@symbolic_helper.parse_args("v", "i", "v", "v", "v")
def ones(g: jit_utils.GraphContext, sizes, dtype, layout, device, pin_memory=False):
    return _constant_fill(g, sizes, dtype, 1)


@_onnx_symbolic("aten::ones_like")
@symbolic_helper.parse_args("v", "i", "v", "v", "v", "v")
def ones_like(
    g: jit_utils.GraphContext,
    input,
    dtype,
    layout,
    device,
    pin_memory=False,
    memory_format=None,
):
    shape = g.op("Shape", input)
    return _constant_fill(g, shape, dtype, 1)


@_onnx_symbolic("aten::full")
def full(
    g: jit_utils.GraphContext, sizes, value, dtype, layout, device, pin_memory=False
):
    const_value = symbolic_helper._maybe_get_const(value, "t")
    if symbolic_helper._is_value(const_value):
        tmp = zeros(g, sizes, dtype, layout, device)
        return opset9.add(g, tmp, value, g.op("Constant", value_t=torch.tensor(1)))
    else:
        dtype = symbolic_helper._get_const(dtype, "i", "dtype")
        return _constant_fill(g, sizes, dtype, const_value)


@_onnx_symbolic("aten::full_like")
@symbolic_helper.parse_args("v", "f", "i", "v", "v", "v", "v")
def full_like(
    g: jit_utils.GraphContext,
    input,
    fill_value,
    dtype,
    layout,
    device,
    pin_memory=False,
    memory_format=None,
):
    shape = g.op("Shape", input)
    return _constant_fill(g, shape, dtype, fill_value)


@_onnx_symbolic("aten::repeat")
def repeat(g: jit_utils.GraphContext, self, repeats):
    if not symbolic_helper._is_value(repeats):
        repeats = g.op("Constant", value_t=torch.LongTensor(repeats))
    if symbolic_helper._is_packed_list(repeats):
        repeat_size_len = len(symbolic_helper._unpack_list(repeats))
    else:
        const_repeats = symbolic_helper._maybe_get_const(repeats, "is")
        repeat_size_len = len(const_repeats)
    if self.isCompleteTensor():
        sizes = self.type().sizes()
        diff_dims = repeat_size_len - len(sizes)
        if diff_dims > 0:
            self = opset9.view(
                g, self, g.op("Constant", value_t=torch.tensor([1] * diff_dims + sizes))
            )
    return g.op("Tile", self, repeats)
<|endoftext|># mypy: allow-untyped-defs
# mypy: disable-error-code=arg-type
from __future__ import annotations

import functools
import sys
import warnings
from typing import TYPE_CHECKING

import torch
import torch._C._onnx as _C_onnx
from torch import _C
from torch.onnx import _constants, errors
from torch.onnx._internal.torchscript_exporter import (
    _type_utils,
    jit_utils,
    registration,
    symbolic_helper,
    symbolic_opset9 as opset9,
)
from torch.onnx._internal.torchscript_exporter._globals import GLOBALS


if TYPE_CHECKING:
    from collections.abc import Sequence


# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in README.md

# This file exports ONNX ops for opset 10
# Opset 10 is supported by ONNX release 1.5.0
# release on 04/24/19


__all__ = [
    "dequantize",
    "div",
    "embedding_bag",
    "fake_quantize_per_tensor_affine",
    "flip",
    "fmod",
    "isfinite",
    "isinf",
    "nan_to_num",
    "quantize_per_tensor",
    "quantized_add_relu",
    "quantized_add",
    "quantized_cat",
    "quantized_conv1d_relu",
    "quantized_conv2d_relu",
    "quantized_conv3d_relu",
    "quantized_conv1d",
    "quantized_conv2d",
    "quantized_conv3d",
    "quantized_conv_transpose1d",
    "quantized_conv_transpose2d",
    "quantized_conv_transpose3d",
    "quantized_group_norm",
    "quantized_hardswish",
    "quantized_instance_norm",
    "quantized_layer_norm",
    "quantized_leaky_relu",
    "quantized_linear",
    "quantized_linear_relu",
    "quantized_mul",
    "quantized_sigmoid",
    "slice",
    "sort",
    "topk",
]


_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=10)


@_onnx_symbolic("aten::div")
def div(g: jit_utils.GraphContext, self, other, *args):
    if len(args) == 0:
        return opset9.true_divide(g, self, other)
    else:
        return _div_rounding_mode(g, self, other, *args)


@symbolic_helper.parse_args("v", "v", "s")
def _div_rounding_mode(g: jit_utils.GraphContext, self, other, rounding_mode):
    if rounding_mode == "floor":
        return _floor_divide(g, self, other)
    else:
        return opset9._div_rounding_mode(g, self, other, rounding_mode)


@_onnx_symbolic("aten::_floor_divide")
def _floor_divide(g: jit_utils.GraphContext, self, other):
    if symbolic_helper._is_fp(self) or symbolic_helper._is_fp(other):
        out = opset9.true_divide(g, self, other)
        return g.op("Floor", out)
    else:
        # Integer division does truncation rounding
        div = g.op("Div", self, other)
        # Division is negative if: self < 0 != other < 0
        zero = g.op("Constant", value_t=torch.tensor(0, dtype=torch.int64))
        negative = g.op("Xor", g.op("Less", self, zero), g.op("Less", other, zero))

        # For negative numbers with self % other != 0, subtract 1 to round down instead of up
        mod = g.op("Mod", self, other, fmod_i=0)
        fixup_mask = g.op("And", negative, g.op("Not", g.op("Equal", mod, zero)))

        one = g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64))
        fixup = g.op("Sub", div, one)
        return g.op("Where", fixup_mask, fixup, div)


@_onnx_symbolic("aten::sort")
@symbolic_helper.parse_args("v", "i", "i", "none")
def sort(g: jit_utils.GraphContext, self, dim, descending, out=None):
    return symbolic_helper._sort_helper(g, self, dim, descending=descending, out=out)


@_onnx_symbolic("aten::topk")
@symbolic_helper.parse_args("v", "v", "i", "i", "i", "none")
def topk(g: jit_utils.GraphContext, self, k, dim, largest, sorted, out=None):
    return symbolic_helper._topk_helper(
        g, self, k, dim, largest=largest, sorted=sorted, out=out
    )


def _aten_max_pool_onnx(
    g: jit_utils.GraphContext,
    self: _C.Value,
    kernel_shape: Sequence[int],
    strides: Sequence[int],
    pads: Sequence[int],
    dilations: Sequence[int],
    ceil_mode: bool,
    unbatched_rank: int,
) -> _C.Value:
    self_rank = g.op("Size", g.op("Shape", self))
    if self_rank == unbatched_rank:  # C,H,W -> N,C,H,W and N=1
        self = g.op(
            "Unsqueeze",
            self,
            g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)),
        )

    pool_result, _ = g.op(
        "MaxPool",
        self,
        outputs=2,
        ceil_mode_i=ceil_mode,
        dilations_i=dilations,
        kernel_shape_i=kernel_shape,
        pads_i=pads,
        strides_i=strides,
    )

    if self_rank == unbatched_rank:
        pool_result = g.op(
            "Squeeze",
            pool_result,
            g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)),
        )

    return pool_result


# For MaxPool
def _adjust_attributes_of_max_pool(
    expand_size: int,
    kernel_size: Sequence[int] | int,
    stride: Sequence[int] | int,
    padding: Sequence[int] | int,
    dilation: Sequence[int] | int,
) -> tuple[Sequence[int], Sequence[int], Sequence[int], Sequence[int]]:
    """Adjust attributes of avg_pool to match ONNX specification."""

    if isinstance(dilation, int):
        dilation = [dilation] * expand_size

    if isinstance(kernel_size, int):
        kernel_shape = [kernel_size] * expand_size
    else:
        kernel_shape = kernel_size  # type: ignore[assignment]

    if isinstance(padding, int):
        pads = [padding] * expand_size * 2  # type: ignore[operator, assignment]
    elif len(padding) == 1:
        pads = padding * expand_size * 2  # type: ignore[operator, assignment]
    elif len(padding) == 2:
        # 2D padding
        pads = padding * 2  # type: ignore[operator, assignment]
    elif len(padding) == 3:
        # 3D padding
        pads = padding * 2  # type: ignore[operator, assignment]
    else:
        # When padding is already done for all dimensions,
        # we don't need to double it
        # eg: (1, 1, 1, 1, 1, 1)
        pads = padding  # type: ignore[assignment]

    if isinstance(stride, int):
        strides = [stride] * expand_size
    elif not stride:
        strides = kernel_shape
    else:
        strides = stride  # type: ignore[assignment]

    # pyrefly: ignore [bad-return]
    return (kernel_shape, strides, pads, dilation)


def _aten_max_pool_with_indices_onnx(
    g: jit_utils.GraphContext,
    self: _C.Value,
    kernel_shape: Sequence[int],
    strides: Sequence[int],
    pads: Sequence[int],
    dilations: Sequence[int],
    ceil_mode: bool,
    unbatched_rank: int,
    n_dims_one: Sequence[int],
    n_dims_zero: Sequence[int],
    n_dims_axes: Sequence[int],
) -> tuple[_C.Value, Sequence[int]]:
    self_rank = g.op("Size", g.op("Shape", self))
    if self_rank == unbatched_rank:  # C,H,W -> N,C,H,W and N=1
        self = g.op(
            "Unsqueeze",
            self,
            g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)),
        )

    pool_result, indices = g.op(
        "MaxPool",
        self,
        outputs=2,
        ceil_mode_i=ceil_mode,
        dilations_i=dilations,
        kernel_shape_i=kernel_shape,
        pads_i=pads,
        strides_i=strides,
    )
    _, flatten_indices = g.op(
        "MaxPool",
        self,
        outputs=2,
        dilations_i=dilations,
        kernel_shape_i=n_dims_one,
        strides_i=n_dims_one,
    )

    ends = g.op("Constant", value_t=torch.tensor(n_dims_one))
    starts = g.op("Constant", value_t=torch.tensor(n_dims_zero))
    axes = g.op("Constant", value_t=torch.tensor(n_dims_axes))

    delta = g.op("Slice", flatten_indices, starts, ends, axes)
    indices = g.op("Sub", indices, delta)

    if self_rank == unbatched_rank:
        pool_result = g.op(
            "Squeeze", pool_result, value_t=torch.tensor([0], dtype=torch.int64)
        )
        indices = g.op("Squeeze", indices, value_t=torch.tensor([0], dtype=torch.int64))

    return (pool_result, indices)


@_onnx_symbolic(
    "aten::max_pool1d",
    decorate=[symbolic_helper._apply_params("max_pool1d", 1, return_indices=False)],
)
@_onnx_symbolic(
    "aten::max_pool2d",
    decorate=[symbolic_helper._apply_params("max_pool2d", 2, return_indices=False)],
)
@_onnx_symbolic(
    "aten::max_pool3d",
    decorate=[symbolic_helper._apply_params("max_pool3d", 3, return_indices=False)],
)
@_onnx_symbolic(
    "aten::max_pool1d_with_indices",
    decorate=[
        symbolic_helper._apply_params(
            "max_pool1d_with_indices",
            1,
            return_indices=True,
        )
    ],
)
@_onnx_symbolic(
    "aten::max_pool2d_with_indices",
    decorate=[
        symbolic_helper._apply_params(
            "max_pool2d_with_indices",
            2,
            return_indices=True,
        )
    ],
)
@_onnx_symbolic(
    "aten::max_pool3d_with_indices",
    decorate=[
        symbolic_helper._apply_params(
            "max_pool3d_with_indices",
            3,
            return_indices=True,
        )
    ],
)
def _max_pool(name: str, expand_size: int, return_indices: bool):
    @symbolic_helper.quantized_args(True, False, False, False, False, False)
    @symbolic_helper.parse_args("v", "is", "is", "is", "is", "i")
    def symbolic_fn(
        g: jit_utils.GraphContext,
        input: _C.Value,
        kernel_size: Sequence[int],
        stride: Sequence[int],
        padding: int | Sequence[int],
        dilation: Sequence[int],
        ceil_mode: bool,
    ):
        kernel_shape, strides, pads, dilations = _adjust_attributes_of_max_pool(
            expand_size, kernel_size, stride, padding, dilation
        )

        if return_indices:
            return _aten_max_pool_with_indices_onnx(
                g,
                input,
                kernel_shape,
                strides,
                pads,
                dilations,
                ceil_mode,
                expand_size + 1,
                ([1] * expand_size),
                ([0] * expand_size),
                ([2 + i for i in range(expand_size)]),
            )
        else:
            return _aten_max_pool_onnx(
                g,
                input,
                kernel_shape,
                strides,
                pads,
                dilations,
                ceil_mode,
                expand_size + 1,
            )

    return symbolic_fn


# For AvgPool
def _adjust_attributes_of_avg_pool(
    expand_size: int,
    kernel_size: Sequence[int] | int,
    stride: Sequence[int] | int,
    padding: Sequence[int] | int,
) -> tuple[Sequence[int], Sequence[int], Sequence[int]]:
    """Adjust attributes of avg_pool to match ONNX specification."""

    if isinstance(kernel_size, int):
        kernel_shape = [kernel_size] * expand_size
    else:
        kernel_shape = kernel_size  # type: ignore[assignment]

    if isinstance(padding, int):
        pads = [padding] * expand_size * 2
    elif len(padding) == 1:
        pads = padding * expand_size * 2  # type: ignore[operator, assignment]
    elif len(padding) == 2:
        pads = padding * expand_size  # type: ignore[operator, assignment]
    else:
        pads = padding * 2  # type: ignore[operator, assignment]

    if isinstance(stride, int):
        strides = [stride] * expand_size
    elif not stride:
        strides = kernel_shape
    else:
        strides = stride  # type: ignore[assignment]

    # pyrefly: ignore [bad-return]
    return (kernel_shape, strides, pads)


@_onnx_symbolic(
    "aten::avg_pool1d",
    decorate=[symbolic_helper._apply_params("avg_pool1d", 1)],
)
@_onnx_symbolic(
    "aten::avg_pool2d",
    decorate=[symbolic_helper._apply_params("avg_pool2d", 2)],
)
@_onnx_symbolic(
    "aten::avg_pool3d",
    decorate=[symbolic_helper._apply_params("avg_pool3d", 3)],
)
def _avg_pool(name, expand_size):
    @symbolic_helper.quantized_args(True, False, False, False, False, False, False)
    @symbolic_helper.parse_args("v", "is", "is", "is", "i", "i", "none")
    def symbolic_fn(
        g,
        input: _C.Value,
        kernel_size: Sequence[int],
        stride: Sequence[int],
        padding: int | Sequence[int],
        ceil_mode: int,
        count_include_pad: int,
        divisor_override=None,
    ):
        kernel_shape, strides, pads = _adjust_attributes_of_avg_pool(
            expand_size, kernel_size, stride, padding
        )

        result = g.op(
            "AveragePool",
            input,
            ceil_mode_i=ceil_mode,
            count_include_pad_i=count_include_pad,
            kernel_shape_i=kernel_shape,
            pads_i=pads,
            strides_i=strides,
        )

        return result

    return symbolic_fn


@_onnx_symbolic(
    "aten::upsample_nearest1d",
    decorate=[symbolic_helper._apply_params("upsample_nearest1d", 3, "nearest")],
)
@_onnx_symbolic(
    "aten::upsample_nearest2d",
    decorate=[symbolic_helper._apply_params("upsample_nearest2d", 4, "nearest")],
)
@_onnx_symbolic(
    "aten::upsample_nearest3d",
    decorate=[symbolic_helper._apply_params("upsample_nearest3d", 5, "nearest")],
)
@_onnx_symbolic(
    "aten::upsample_linear1d",
    decorate=[symbolic_helper._apply_params("upsample_linear1d", 3, "linear")],
)
@_onnx_symbolic(
    "aten::upsample_bilinear2d",
    decorate=[symbolic_helper._apply_params("upsample_bilinear2d", 4, "linear")],
)
@_onnx_symbolic(
    "aten::upsample_trilinear3d",
    decorate=[symbolic_helper._apply_params("upsample_trilinear3d", 5, "linear")],
)
def _interpolate(name, dim, interpolate_mode):
    @symbolic_helper.quantized_args(True, False, False)
    def symbolic_fn(g, input, output_size, *args):
        scales, align_corners = symbolic_helper._get_interpolate_attributes(
            g, interpolate_mode, args
        )
        symbolic_helper._interpolate_warning(interpolate_mode)
        align_corners = symbolic_helper._maybe_get_scalar(align_corners)
        if align_corners:
            return symbolic_helper._unimplemented(name, "align_corners == True", input)
        if scales is None:
            scales = symbolic_helper._interpolate_size_to_scales(
                g, input, output_size, dim
            )
        return g.op("Resize", input, scales, mode_s=interpolate_mode)

    return symbolic_fn


@_onnx_symbolic("aten::__interpolate")
def __interpolate(
    g: jit_utils.GraphContext,
    input,
    size,
    scale_factor,
    mode,
    align_corners,
    recompute_scale_factor,
    antialias,
):
    scales, mode = symbolic_helper._interpolate_get_scales_and_mode(
        g, input, size, scale_factor, mode, align_corners
    )
    return g.op("Resize", input, scales, mode_s=mode)


def _slice(
    g: jit_utils.GraphContext,
    input: torch._C.Value,
    axes: list | torch.Tensor | torch._C.Value,
    starts: list | torch.Tensor | torch._C.Value,
    ends: list | torch.Tensor | torch._C.Value,
    steps: list | torch.Tensor | torch._C.Value | None = None,
):
    def is_none_value(value):
        if value is None:
            return True
        return (
            isinstance(value, torch._C.Value)
            and value.node().kind() == "prim::Constant"
            and isinstance(value.type(), _C.NoneType)
        )

    def to_slice_input(list_or_value, default_value=None):
        # Convert input param into a 1D torch.Value.
        if is_none_value(list_or_value) and default_value is not None:
            list_or_value = [default_value]

        if isinstance(list_or_value, torch.Tensor):
            return g.op("Constant", value_t=list_or_value.clone().detach())
        elif isinstance(list_or_value, list):
            return g.op("Constant", value_t=torch.tensor(list_or_value))

        rank = symbolic_helper._get_tensor_rank(list_or_value)
        if rank == 0:
            return symbolic_helper._unsqueeze_helper(g, list_or_value, [0])
        if rank == 1:
            return list_or_value
        raise errors.SymbolicValueError(
            f"Rank must be 0 or 1, not {rank}", list_or_value
        )

    def get_const_value(list_or_value):
        if isinstance(list_or_value, (list, torch.Tensor)):
            if len(list_or_value) == 1:
                return list_or_value[0]
            return None
        return symbolic_helper._maybe_get_const(list_or_value, "i")

    # Check if slice is a no-op
    if (
        get_const_value(starts) == 0
        and get_const_value(ends) == _constants.INT64_MAX
        and (steps is None or get_const_value(steps) == 1)
    ):
        return input

    axes = to_slice_input(axes)
    starts = to_slice_input(starts, default_value=0)
    ends = to_slice_input(ends, default_value=_constants.INT64_MAX)
    if steps is None:
        return g.op("Slice", input, starts, ends, axes)
    steps = to_slice_input(steps, default_value=1)
    return g.op("Slice", input, starts, ends, axes, steps)


@_onnx_symbolic("aten::slice")
def slice(g: jit_utils.GraphContext, self, *args):
    if len(args) == 4:
        # aten::slice(Tensor self, int dim, int? start=None, int? end=None, int step=1) -> Tensor
        dims, start, end, step = args
    elif len(args) == 3:
        # aten::slice(t[] l, int? start=None, int? end=None, int step=1) -> t[]
        start, end, step = args
        dims = [0]
    else:
        raise errors.SymbolicValueError("Unknown aten::slice signature", self)

    return symbolic_helper._slice_helper(
        g,
        self,
        axes=dims,
        starts=start,
        ends=end,
        steps=step,
    )


@_onnx_symbolic("aten::flip")
@symbolic_helper.parse_args("v", "is")
def flip(g: jit_utils.GraphContext, input, dims):
    return symbolic_helper._slice_helper(
        g,
        input,
        axes=dims,
        starts=[-1] * len(dims),
        ends=[-_constants.INT64_MAX] * len(dims),
        steps=[-1] * len(dims),
    )


@_onnx_symbolic("aten::fmod")
def fmod(g: jit_utils.GraphContext, input, other):
    return g.op("Mod", input, other, fmod_i=1)


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
    if scale_grad_by_freq and GLOBALS.export_training:
        return symbolic_helper._onnx_unsupported(
            "embedding_bag with scale_grad_by_freq for training mode"
        )
    if padding_idx is not None and padding_idx >= 0:
        raise RuntimeError("embedding_bag with padding_idx")

    warnings.warn(
        "Export of embedding_bag with dynamic input/offsets shape is not supported in opset 10. "
        "Please use opset 11 or higher to export model for dynamic input shape.'",
        stacklevel=2,
    )
    offsets_dim_0 = symbolic_helper._get_tensor_dim_size(offsets, 0)
    if offsets_dim_0 is not None:
        if include_last_offset:
            offset_len = offsets_dim_0 - 1
            offsets_extended = offsets
        else:
            offset_len = offsets_dim_0
            offsets_extended = [
                offsets,
                g.op("Constant", value_t=torch.tensor([sys.maxsize])),
            ]
            offsets_extended = g.op("Concat", *offsets_extended, axis_i=0)
        list_ = []
        for i in range(offset_len):
            start_ = symbolic_helper._unsqueeze_helper(
                g,
                opset9.select(g, offsets_extended, torch.tensor(0), torch.tensor(i)),
                [0],
            )
            end_ = symbolic_helper._unsqueeze_helper(
                g,
                opset9.select(
                    g, offsets_extended, torch.tensor(0), torch.tensor(i + 1)
                ),
                [0],
            )
            axes_ = g.op("Constant", value_t=torch.tensor([0]))
            indices_row = g.op("Slice", indices, start_, end_, axes_)

            embeddings = g.op("Gather", embedding_matrix, indices_row)
            if not symbolic_helper._is_none(per_sample_weights):
                per_sample_weights_row = g.op(
                    "Slice", per_sample_weights, start_, end_, axes_
                )
                per_sample_weights_row = symbolic_helper._unsqueeze_helper(
                    g, per_sample_weights_row, [1]
                )
                embeddings = g.op("Mul", embeddings, per_sample_weights_row)
            if mode == 0:
                embeddings = symbolic_helper._reducesum_helper(
                    g, embeddings, axes_i=[0], keepdims_i=0
                )
            elif mode == 1:
                embeddings = g.op("ReduceMean", embeddings, axes_i=[0], keepdims_i=0)
            else:
                embeddings = g.op("ReduceMax", embeddings, axes_i=[0], keepdims_i=0)

            embeddings = symbolic_helper._unsqueeze_helper(g, embeddings, [0])
            list_.append(embeddings)

        output = g.op("Concat", *list_, axis_i=0)
        # aten::embedding_bag returns a tuple of 4 elements: output, offset2bag, bag_size, max_indices.
        # But the last three outputs are not used in torch.nn.EmbeddingBag or torch.nn.functional.embedding_bag.
        return output, None, None, None
    else:
        return symbolic_helper._onnx_unsupported(
            "embedding_bag with unknown shape of offsets for opset 10 is not supported. "
            "please use opset 11 or higher."
        )


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
    # NOTE: (0, 127) is a special case. PyTorch restricts activations to be in the range (0, 127).
    #   https://github.com/pytorch/pytorch/blob/b34b192d6b97325c9f78e5995c48c8498ede34bd/torch/ao/quantization/observer.py#L1422
    if (quant_min, quant_max) == (0, 127):
        symbolic_helper._onnx_opset_unsupported_detailed(
            "fake_quantize_per_tensor_affine",
            10,
            13,
            "Quantize range (0, 127) not supported, requires opset 13 Clip",
            inputs,
        )
    if (quant_min, quant_max) not in [(0, 255), (-128, 127)]:
        raise errors.SymbolicValueError(
            f"For (quant_min, quant_max), ONNX allows only (0, 255) and (-128, 127). "
            f"Got ({quant_min}, {quant_max})",
            inputs,
        )
    scale = symbolic_helper._maybe_get_scalar(scale)
    if scale is None:
        symbolic_helper._onnx_opset_unsupported_detailed(
            "fake_quantize_per_tensor_affine",
            10,
            13,
            "Non-constant scale not supported",
            inputs,
        )

    scale = scale.float().data  # Avoid exporter generating double type
    if quant_min == 0:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.UINT8)
    else:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.INT8)
    return g.op(
        "DequantizeLinear",
        g.op("QuantizeLinear", inputs, scale, zero_point),
        scale,
        zero_point,
    )


@_onnx_symbolic("aten::isinf")
def isinf(g: jit_utils.GraphContext, input):
    return g.op("IsInf", g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.DOUBLE))


@_onnx_symbolic("aten::isfinite")
def isfinite(g: jit_utils.GraphContext, input):
    inf_node = isinf(g, input)
    nan_node = opset9.isnan(g, input)
    return opset9.__not_(g, opset9.__or_(g, inf_node, nan_node))


@_onnx_symbolic("aten::quantize_per_tensor")
def quantize_per_tensor(g: jit_utils.GraphContext, input, scale, zero_point, dtype):
    dtype = symbolic_helper._get_const(dtype, "i", "dtype")
    # TODO(justinchuby): Extract all the cast ops into a helper function.
    zero_point = g.op(
        "Cast", zero_point, to_i=_type_utils.JitScalarType(dtype).onnx_type()
    )
    scale = g.op("Cast", scale, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    return symbolic_helper.quantize_helper(g, input, scale, zero_point)


@_onnx_symbolic("aten::dequantize")
def dequantize(g: jit_utils.GraphContext, input):
    return symbolic_helper.dequantize_helper(g, input)[0]


@_onnx_symbolic("aten::nan_to_num")
@symbolic_helper.parse_args("v", "f", "f", "f")
def nan_to_num(g: jit_utils.GraphContext, input, nan, posinf, neginf):
    # Cannot create an int type tensor with inf/nan values, so we simply
    # return the original tensor
    if not symbolic_helper._is_fp(input):
        return input
    input_dtype = _type_utils.JitScalarType.from_value(input).dtype()
    if nan is None:
        nan = 0.0
    nan_cond = opset9.isnan(g, input)
    nan_result = g.op(
        "Where",
        nan_cond,
        g.op("Constant", value_t=torch.tensor([nan], dtype=input_dtype)),
        input,
    )

    # For None values of posinf, neginf we use the greatest/lowest finite
    # value representable by input's dtype.
    finfo = torch.finfo(input_dtype)
    if posinf is None:
        posinf = finfo.max
    posinf_cond = opset9.logical_and(
        g,
        isinf(g, nan_result),
        opset9.gt(g, nan_result, g.op("Constant", value_t=torch.LongTensor([0]))),
    )
    nan_posinf_result = g.op(
        "Where",
        posinf_cond,
        g.op("Constant", value_t=torch.tensor([posinf], dtype=input_dtype)),
        nan_result,
    )

    if neginf is None:
        neginf = finfo.min
    neginf_cond = opset9.logical_and(
        g,
        isinf(g, nan_posinf_result),
        opset9.lt(
            g, nan_posinf_result, g.op("Constant", value_t=torch.LongTensor([0]))
        ),
    )
    return g.op(
        "Where",
        neginf_cond,
        g.op("Constant", value_t=torch.tensor([neginf], dtype=input_dtype)),
        nan_posinf_result,
    )


# Quantized symbolics ---------------------------------------------------------
# https://github.com/pytorch/pytorch/wiki/PyTorch-ONNX-exporter#quantized-model-export
# Support starts from opset 10 because `DequantizeLinear` and `QuantizeLinear` were
# introduced in opset version 10.
@_onnx_symbolic("quantized::linear")
def quantized_linear(
    g: jit_utils.GraphContext, q_input, q_weight, bias, op_scale, op_zero_point
):
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.linear(g, input, weight, bias)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::linear_relu")
def quantized_linear_relu(
    g: jit_utils.GraphContext, q_input, q_weight, bias, op_scale, op_zero_point
):
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.linear(g, input, weight, bias)
    output = opset9.relu(g, output)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::add")
def quantized_add(g: jit_utils.GraphContext, x, y, op_scale, op_zero_point):
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)
    y, _, _, _ = symbolic_helper.dequantize_helper(g, y)

    output = opset9.add(g, x, y)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::add_relu")
def quantized_add_relu(g: jit_utils.GraphContext, x, y, op_scale, op_zero_point):
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)
    y, _, _, _ = symbolic_helper.dequantize_helper(g, y)

    output = opset9.add(g, x, y)
    output = opset9.relu(g, output)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::mul")
def quantized_mul(g: jit_utils.GraphContext, x, y, op_scale, op_zero_point):
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)
    y, _, _, _ = symbolic_helper.dequantize_helper(g, y)

    output = opset9.mul(g, x, y)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::hardswish")
def quantized_hardswish(g: jit_utils.GraphContext, x, op_scale, op_zero_point):
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)

    output = opset9.hardswish(g, x)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::sigmoid")
def quantized_sigmoid(g: jit_utils.GraphContext, x, op_scale, op_zero_point):
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)

    output = opset9.sigmoid(g, x)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::leaky_relu")
def quantized_leaky_relu(
    g: jit_utils.GraphContext, x, negative_slope, inplace, op_scale, op_zero_point
):
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)

    output = opset9.leaky_relu(g, x, negative_slope, inplace)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::layer_norm")
def quantized_layer_norm(
    g: jit_utils.GraphContext,
    x,
    normalized_shape,
    weight,
    bias,
    eps,
    op_scale,
    op_zero_point,
):
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)

    output = opset9.layer_norm(g, x, normalized_shape, weight, bias, eps, False)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::group_norm")
def quantized_group_norm(
    g: jit_utils.GraphContext,
    x,
    num_groups,
    weight,
    bias,
    eps,
    op_scale,
    op_zero_point,
):
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)

    output = opset9.group_norm(g, x, num_groups, weight, bias, eps, False)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::instance_norm")
@symbolic_helper.parse_args("v", "v", "v", "f", "v", "v")
def quantized_instance_norm(
    g: jit_utils.GraphContext,
    q_input,
    weight,
    bias,
    eps,
    op_scale,
    op_zero_point,
):
    input, _, _, _ = symbolic_helper.dequantize_helper(g, q_input)

    output = opset9.instance_norm(
        g, input, weight, bias, None, None, False, 0.0, eps, False
    )

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
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
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
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
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
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
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
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
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
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
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
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
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
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
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
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
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
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.conv_transpose3d(
        g, input, weight, bias, stride, padding, output_padding, groups, dilation
    )

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::cat")
@symbolic_helper.parse_args("v", "i", "v", "v")
def quantized_cat(
    g: jit_utils.GraphContext,
    q_inputs: _C.Value,
    dim: int,
    op_scale: _C.Value,
    op_zero_point: _C.Value,
) -> _C.Value:
    unpacked_inputs = symbolic_helper._unpack_list(q_inputs)
    dequantized = [
        symbolic_helper.dequantize_helper(g, input)[0] for input in unpacked_inputs
    ]
    concatenated = g.op("Concat", *dequantized, axis_i=dim)
    return symbolic_helper.quantize_helper(g, concatenated, op_scale, op_zero_point)
<|endoftext|># mypy: allow-untyped-defs
# mypy: disable-error-code=arg-type
"""This file exports ONNX ops for opset 17.

Note [ONNX Operators that are added/updated in opset 17]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-17-of-the-default-onnx-operator-set
New operators:
    BlackmanWindow
    DFT
    HammingWindow
    HannWindow
    LayerNormalization
    MelWeightMatrix
    STFT
    SequenceMap
"""

import functools
from collections.abc import Sequence

import torch
from torch import _C
from torch.onnx import errors
from torch.onnx._internal.torchscript_exporter import (
    _type_utils,
    jit_utils,
    registration,
    symbolic_helper,
)


# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in README.md

__all__ = ["layer_norm", "stft", "quantized_layer_norm"]

_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=17)


@_onnx_symbolic("aten::layer_norm")
@symbolic_helper.parse_args("v", "is", "v", "v", "f", "none")
def layer_norm(
    g: jit_utils.GraphContext,
    input: _C.Value,
    normalized_shape: Sequence[int],
    weight: _C.Value,
    bias: _C.Value,
    eps: float,
    cudnn_enable: bool,
):
    # normalized_shape: input shape from an expected input of size
    # axis: The first normalization dimension.
    # layer_norm normalizes on the last D dimensions,
    # where D is the size of normalized_shape
    axis = -len(normalized_shape)
    scalar_type = _type_utils.JitScalarType.from_value(
        input, _type_utils.JitScalarType.FLOAT
    )
    dtype = scalar_type.dtype()
    if symbolic_helper._is_none(weight):
        weight_value = torch.ones(normalized_shape, dtype=dtype)
        weight = g.op("Constant", value_t=weight_value)
    if symbolic_helper._is_none(bias):
        bias_value = torch.zeros(normalized_shape, dtype=dtype)
        bias = g.op("Constant", value_t=bias_value)
    return g.op(
        "LayerNormalization",
        input,
        weight,
        bias,
        epsilon_f=eps,
        axis_i=axis,
    )


@_onnx_symbolic("quantized::layer_norm")
def quantized_layer_norm(
    g: jit_utils.GraphContext,
    x,
    normalized_shape,
    weight,
    bias,
    eps,
    op_scale,
    op_zero_point,
):
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)

    output = layer_norm(g, x, normalized_shape, weight, bias, eps, False)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


def _compute_edge_sizes(n_fft, window_size):
    """Helper function to compute the sizes of the edges (left and right)
    of a given window centered within an FFT size."""
    left = (n_fft - window_size) // 2
    right = n_fft - left - window_size
    return left, right


@_onnx_symbolic("aten::stft")
@symbolic_helper.parse_args("v", "i", "i", "i", "v", "b", "b", "b", "b")
def stft(
    g: jit_utils.GraphContext,
    input: _C.Value,
    n_fft: int,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: _C.Value | None = None,
    normalized: bool = False,
    onesided: bool | None = True,
    return_complex: bool | None = False,
    align_to_window: bool | None = None,
) -> _C.Value:
    """Associates `torch.stft` with the `STFT` ONNX operator.
    Note that torch.stft calls _VF.stft, without centering or padding options.
    Hence, this function does not contain these two arguments.
    See torch.stft source code for more info.

    Args:
        g: Graph to write the ONNX representation into
        input: Input tensor for the transformation
        n_fft: FFT size
        hop_length: Size of the hop. Defaults to `floor(n_fft // 4)`
        win_length: Size of the analysis window. Defaults to `n_fft`
        window: Analysis window. Defaults to a window of all ones
        normalized: Whether to return a normalized STFT
        onesided: Whether to return only half (+1) of the results, given the
            symmetry of the STFT
        return_complex: Whether to return the complex value (Note: Must be
            `False` or `None`)

    Returns:
        op: Operator for torch.stft associated with STFT (ONNX)
    """
    # Checks
    if return_complex:
        raise errors.SymbolicValueError(
            msg="STFT does not currently support complex types", value=input
        )

    if align_to_window is not None:
        raise errors.SymbolicValueError(
            msg="STFT does not currently support the align_to_window option",
            value=input,
        )  # TODO(#145944): add compatibility with align_to_window option.

    # Get STFT sizes
    frame_step_value = hop_length if hop_length is not None else n_fft // 4
    frame_step_const = g.op(
        "Constant", value_t=torch.tensor(frame_step_value, dtype=torch.int64)
    )
    frame_length_const = g.op(
        "Constant", value_t=torch.tensor(n_fft, dtype=torch.int64)
    )

    # Pre-process input if needed
    signal = input
    signal_rank = symbolic_helper._get_tensor_rank(signal)
    if signal_rank == 1:
        # Add batch dimension
        signal = g.op(
            "Unsqueeze",
            signal,
            g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)),
        )
    elif signal_rank is None or signal_rank > 2:
        raise errors.SymbolicValueError(
            msg="STFT can only take inputs of 1 [signal] or 2 [batch, signal] dimensions. "
            f"Current rank of signal is {signal_rank}, please reduce it.",
            value=input,
        )

    # Get window and make sure it's the same size as `win_length` or `n_fft`
    # pyrefly: ignore [bad-argument-type]
    n_win = symbolic_helper._get_tensor_dim_size(window, dim=0)
    if n_win is not None:
        win_length_default = win_length if win_length else n_fft
        if n_win != win_length_default:
            raise AssertionError(
                "Analysis window size must equal `win_length` or `n_fft`. "
                f"Please, set `win_length` or `n_fft` to match `window` size ({n_win})"
            )

        # Center window around zeros if needed (required by ONNX's STFT)
        if n_win < n_fft:
            left, right = _compute_edge_sizes(n_fft, n_win)
            left_win = g.op("Constant", value_t=torch.zeros(left))
            right_win = g.op("Constant", value_t=torch.zeros(right))
            # pyrefly: ignore [bad-argument-type]
            window = g.op("Concat", left_win, window, right_win, axis_i=0)

    # Create window, if needed
    if symbolic_helper._is_none(window):
        if win_length:
            if win_length > n_fft:
                raise errors.SymbolicValueError(
                    msg="The analysis window can't be longer than the size of the FFT. "
                    f"Please set `win_length` ({win_length}) to `n_fft` ({n_fft}) or less.",
                    value=input,
                )

            # Center window, if needed
            left, right = _compute_edge_sizes(n_fft, win_length)
            torch_window = torch.hstack(
                (torch.zeros(left), torch.ones(win_length), torch.zeros(right))
            )
        else:
            # Rectangle window
            torch_window = torch.ones(n_fft)
        if torch_window.shape[0] != n_fft:
            raise AssertionError(
                f"torch_window.shape[0]={torch_window.shape[0]} != n_fft={n_fft}"
            )
        window = g.op("Constant", value_t=torch_window)
    window = g.op(
        "Cast",
        # pyrefly: ignore [bad-argument-type]
        window,
        to_i=_type_utils.JitScalarType.from_value(signal).onnx_type(),
    )

    # Run STFT
    result = g.op(
        "STFT",
        signal,
        frame_step_const,
        window,
        frame_length_const,
        onesided_i=1 if onesided is None or onesided else 0,
    )

    # Transpose to mimic torch.stft's behavior
    result = g.op("Transpose", result, perm_i=[0, 2, 1, 3])

    # Remove batch dimension, if needed
    if signal_rank == 1:
        result = g.op(
            "Squeeze",
            result,
            g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)),
        )

    # Normalize, if needed
    if normalized:
        sqrt_nfft = torch.sqrt(torch.tensor(n_fft, dtype=signal.type().dtype()))
        result = g.op("Div", result, g.op("Constant", value_t=sqrt_nfft))

    return result
<|endoftext|># mypy: allow-untyped-defs
"""This file exports ONNX ops for opset 18.

Note [ONNX Operators that are added/updated in opset 18]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-18-of-the-default-onnx-operator-set
New operators:
    BitwiseAnd
    CenterCropPad
    Col2Im
    Mish
    OptionalGetElement
    OptionalHasElement
    Pad
    Resize
    ScatterElements
    ScatterND
    Split
"""

import functools
from collections.abc import Sequence

import torch
from torch import _C
from torch.onnx._internal.torchscript_exporter import (
    _type_utils,
    jit_utils,
    registration,
    symbolic_helper,
    symbolic_opset9 as opset9,
)


# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

__all__ = [
    "col2im",
]

_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=18)


@_onnx_symbolic("aten::__and_")
@_onnx_symbolic("aten::bitwise_and")
def __and_(g: jit_utils.GraphContext, self, other):
    # do type promotion (scalars don't seem to apply)
    args = [self, other]
    # type promotion doesn't happen with torch.bitwise_and(tensor, scalar)
    prom_args = [arg for arg in args if symbolic_helper._get_tensor_rank(arg)]
    if len(prom_args) == 0:
        prom_args = args
    promotion_jit_type = symbolic_helper._type_promote_from_values(*prom_args)
    self = symbolic_helper._maybe_cast_to_type(g, self, promotion_jit_type)
    other = symbolic_helper._maybe_cast_to_type(g, other, promotion_jit_type)
    if promotion_jit_type == _type_utils.JitScalarType.BOOL:
        return g.op("And", self, other)
    return g.op("BitwiseAnd", self, other)


@_onnx_symbolic("aten::col2im")
@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is")
def col2im(
    g,
    input: _C.Value,
    output_size: _C.Value,
    kernel_size: _C.Value,
    dilation: Sequence[int],
    padding: Sequence[int],
    stride: Sequence[int],
):
    # convert [i0, i1, ..., in] into [i0, i0, i1, i1, ..., in, in]
    adjusted_padding: list[int] = []
    for pad in padding:
        adjusted_padding.extend(pad for _ in range(2))

    num_dimensional_axis = symbolic_helper._get_tensor_sizes(output_size)[0]
    if not adjusted_padding:
        adjusted_padding = [0, 0] * num_dimensional_axis

    if not dilation:
        dilation = [1] * num_dimensional_axis

    if not stride:
        stride = [1] * num_dimensional_axis

    return g.op(
        "Col2Im",
        input,
        output_size,
        kernel_size,
        dilations_i=dilation,
        pads_i=adjusted_padding,
        strides_i=stride,
    )


@_onnx_symbolic(
    "aten::mean", decorate=[symbolic_helper._apply_params("ReduceMean", "mean")]
)
@_onnx_symbolic(
    "aten::prod",
    decorate=[
        symbolic_helper._apply_params(
            "ReduceProd", "prod", allow_multi_dim_support=False
        )
    ],
)
def _reduce_with_dtype(onnx_op: str, name: str, allow_multi_dim_support: bool = True):
    return symbolic_helper._reduce_with_dtype_helper(
        onnx_op, name, allow_multi_dim_support
    )


@_onnx_symbolic("aten::native_layer_norm")
@symbolic_helper.quantized_args(True, False, False, False)
@symbolic_helper.parse_args("v", "is", "v", "v", "f")
def _native_layer_norm(
    g: jit_utils.GraphContext,
    input: _C.Value,
    normalized_shape: Sequence[int],
    weight: _C.Value,
    bias: _C.Value,
    eps: float,
) -> tuple[_C.Value, _C.Value, _C.Value]:
    return opset9.native_layer_norm(g, input, normalized_shape, weight, bias, eps)


@_onnx_symbolic("aten::glu")
@symbolic_helper.parse_args("v", "i")
def _glu(g: jit_utils.GraphContext, input, dim):
    dim_size = symbolic_helper._get_tensor_dim_size(input, dim)
    if dim_size is not None:
        if dim_size % 2 != 0:
            raise AssertionError(f"dim_size must be even, got {dim_size}")

    first, second = g.op("Split", input, axis_i=dim, num_outputs_i=2, outputs=2)
    return g.op("Mul", first, g.op("Sigmoid", second))


@_onnx_symbolic("aten::max")
# torch.max (same for torch.min) actually has two interfaces smashed together:
# torch.max(x, dim, keepdim) and torch.max(x, y)
# TODO(justinchuby): Support multiple quantized args in output
def max(g: jit_utils.GraphContext, self, dim_or_y=None, keepdim=None):
    return symbolic_helper._max_helper(g, self, dim_or_y, keepdim)


@_onnx_symbolic("aten::maximum")
@symbolic_helper.quantized_args(True, True)
def maximum(g: jit_utils.GraphContext, input, other):
    # pyrefly: ignore [no-matching-overload]
    return max(g, input, dim_or_y=other)


@_onnx_symbolic("aten::min")
# TODO(justinchuby): Support multiple quantized args in output
def min(g: jit_utils.GraphContext, self, dim_or_y=None, keepdim=None):
    return symbolic_helper._min_helper(g, self, dim_or_y, keepdim)


@_onnx_symbolic("aten::minimum")
@symbolic_helper.quantized_args(True, True)
def minimum(g: jit_utils.GraphContext, input, other):
    # pyrefly: ignore [no-matching-overload]
    return min(g, input, dim_or_y=other)


@_onnx_symbolic("aten::amax")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "is", "i")
def amax(g: jit_utils.GraphContext, self, dim, keepdim):
    axes = g.op("Constant", value_t=torch.tensor(dim, dtype=torch.long))
    return g.op("ReduceMax", self, axes, keepdims_i=keepdim)


@_onnx_symbolic("aten::amin")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "is", "i")
def amin(g: jit_utils.GraphContext, self, dim, keepdim):
    axes = g.op("Constant", value_t=torch.tensor(dim, dtype=torch.long))
    return g.op("ReduceMin", self, axes, keepdims_i=keepdim)


@_onnx_symbolic("aten::aminmax")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "v", "i")
def aminmax(g: jit_utils.GraphContext, self, dim, keepdim):
    if not symbolic_helper._is_none(dim):
        dim = symbolic_helper._get_const(dim, "i", "dim")
        axes = g.op("Constant", value_t=torch.tensor([dim], dtype=torch.long))
        return g.op("ReduceMin", self, axes, keepdims_i=keepdim), g.op(
            "ReduceMax", self, axes, keepdims_i=keepdim
        )
    else:
        return g.op("ReduceMin", self, keepdims_i=keepdim), g.op(
            "ReduceMax", self, keepdims_i=keepdim
        )


@_onnx_symbolic("aten::var_mean")
def _var_mean(g: jit_utils.GraphContext, input, *args):
    if len(args) == 1:
        return symbolic_helper._var_mean_helper(g, input, None, args[0], None)
    else:
        return symbolic_helper._var_mean_helper(g, input, *args)


@_onnx_symbolic("aten::logsumexp")
@symbolic_helper.parse_args("v", "is", "i")
def _logsumexp(g: jit_utils.GraphContext, input, dim, keepdim):
    if dim is None:
        return g.op("ReduceLogSumExp", input, keepdims_i=0)
    else:
        axes = g.op("Constant", value_t=torch.tensor(dim, dtype=torch.long))
        return g.op("ReduceLogSumExp", input, axes, keepdims_i=keepdim)


@_onnx_symbolic("aten::linalg_matrix_norm")
@symbolic_helper.parse_args("v", "v", "is", "b", "v")
def _linalg_matrix_norm(
    g: jit_utils.GraphContext,
    self: torch._C.Value,
    ord: torch._C.Value,
    dim: list[int],
    keepdim: bool,
    dtype: torch._C.Value,
):
    return opset9.linalg_matrix_norm(g, self, ord, dim, keepdim, dtype)


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


@_onnx_symbolic("aten::linalg_vector_norm")
@symbolic_helper.parse_args("v", "f", "is", "b", "v")
def linalg_vector_norm(
    g: jit_utils.GraphContext,
    self: torch._C.Value,
    ord: float,
    dim: Sequence[int] | None,
    keepdim: bool,
    dtype: torch._C.Value,
):
    return symbolic_helper._linalg_vector_norm_helper(g, self, ord, dim, keepdim, dtype)
<|endoftext|># mypy: allow-untyped-defs
"""Functions to export models into the ONNX IR format.

These models can be loaded with the ONNX library and then
converted to models which run on other deep learning frameworks.
"""

from __future__ import annotations


__all__ = [
    "select_model_mode_for_export",
    "disable_apex_o2_state_dict_hook",
    "setup_onnx_logging",
    "exporter_context",
    "export",
    "model_signature",
    "warn_on_static_input_change",
    "unpack_quantized_tensor",
    "unconvertible_ops",
    "register_custom_op_symbolic",
    "unregister_custom_op_symbolic",
    "_add_block",
    "_add_input_to_block",
    "_add_output_to_block",
    "_apply_friendly_debug_names",
    "_check_flatten_did_not_remove",
    "_create_jit_graph",
    "_decide_add_node_names",
    "_decide_constant_folding",
    "_decide_input_format",
    "_decide_keep_init_as_input",
    "_export",
    "_get_aten_op_overload_name",
    "_get_example_outputs",
    "_get_module_attributes",
    "_get_named_param_dict",
    "_get_param_count_list",
    "_is_constant_tensor_list",
    "_model_to_graph",
    "_optimize_graph",
    "_pre_trace_quant_model",
    "_reset_trace_module_map",
    "_resolve_args_by_export_type",
    "_run_symbolic_function",
    "_run_symbolic_method",
    "_set_input_and_output_names",
    "_setup_trace_module_map",
    "_should_aten_fallback",
    "_signature",
    "_split_tensor_list_constants",
    "_trace_and_get_graph_from_model",
    "_trace",
    "_trigger_symbolic_function_registration",
    "_validate_dynamic_axes",
    "_verify_custom_op_name",
]

import contextlib
import copy
import inspect
import re
import typing
import warnings
from typing import Any, cast
from typing_extensions import deprecated

import torch
import torch._C._onnx as _C_onnx
import torch.jit._trace
from torch import _C
from torch.onnx import _constants, errors
from torch.onnx._internal.torchscript_exporter import (
    jit_utils,
    onnx_proto_utils,
    registration,
    symbolic_helper,
)
from torch.onnx._internal.torchscript_exporter._globals import GLOBALS


if typing.TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence


# TODO(justinchuby): Remove dependency to this global variable from constant_fold.cpp
# Skip check due to cannot import IValue from torch._C
_params_dict = {}  # type: ignore[var-annotated]


@deprecated("Please set training mode before exporting the model", category=None)
@contextlib.contextmanager
def select_model_mode_for_export(model, mode: _C_onnx.TrainingMode):
    """A context manager to temporarily set the training mode of ``model``
    to ``mode``, resetting it when we exit the with-block.

    .. deprecated:: 2.7
        Please set training mode before exporting the model.

    Args:
        model: Same type and meaning as ``model`` arg to :func:`export`.
        mode: Same type and meaning as ``training`` arg to :func:`export`.
    """
    if not isinstance(mode, _C_onnx.TrainingMode):
        raise TypeError(
            f"'mode' should be a torch.onnx.TrainingMode enum, but got '{type(mode)}'."
        )
    originally_training: bool = False

    if hasattr(model, "training"):
        originally_training = model.training

        # ONNX opset 12 has better support for training amenable models, with updated
        # versions of the dropout and batch_norm operators
        if mode == _C_onnx.TrainingMode.TRAINING or (
            mode == _C_onnx.TrainingMode.PRESERVE and originally_training
        ):
            GLOBALS.export_training = True
            if GLOBALS.export_onnx_opset_version < 12:
                warnings.warn(
                    "You are exporting the model in training mode with onnx opset "
                    f"version {GLOBALS.export_onnx_opset_version}. "
                    "Opset versions lower than opset 12 will not be able to export "
                    "nodes such as Dropout and BatchNorm correctly.",
                    stacklevel=2,
                )
        else:
            GLOBALS.export_training = False

        GLOBALS.training_mode = mode
        if mode == _C_onnx.TrainingMode.TRAINING:
            model.train(True)
        elif mode == _C_onnx.TrainingMode.EVAL:
            model.train(False)
        # else mode == _C_onnx.TrainingMode.PRESERVE, do nothing

    try:
        yield
    finally:
        if hasattr(model, "training") and mode != _C_onnx.TrainingMode.PRESERVE:
            model.train(originally_training)


@deprecated(
    "Please remove usage of this function. Copy its logic if it is required in user code",
    category=None,
)
@contextlib.contextmanager
def disable_apex_o2_state_dict_hook(model: torch.nn.Module | torch.jit.ScriptFunction):
    """A context manager to temporarily disable the Apex O2 hook that returns.

    .. deprecated:: 2.7
        Please remove usage of this function.
    """
    # Apex O2 hook state_dict to return fp16 weights as fp32.
    # Exporter cannot identify them as same tensors.
    # Since this hook is only used by optimizer, it is safe to
    # remove this hook while exporting.
    if not isinstance(model, torch.jit.ScriptFunction):
        model_hooks = {}  # type: ignore[var-annotated]
        for module in model.modules():
            for key, hook in module._state_dict_hooks.items():
                if type(hook).__name__ == "O2StateDictHook":
                    if module not in model_hooks:
                        model_hooks[module] = {}
                    model_hooks[module][key] = hook
            if module in model_hooks:
                for key in model_hooks[module]:
                    module._state_dict_hooks.pop(key)
        try:
            yield
        finally:
            # Add the hooks back
            for module, m_map in model_hooks.items():
                for key, hook in m_map.items():
                    module._state_dict_hooks[key] = hook
    else:
        try:
            yield
        finally:
            pass


@deprecated("The feature will be removed. Please remove usage of this function")
@contextlib.contextmanager
def setup_onnx_logging(verbose: bool):
    """A context manager to temporarily set the ONNX logging verbosity.

    .. deprecated:: 2.7
        Please remove usage of this function.
    """
    is_originally_enabled = _C._jit_is_onnx_log_enabled
    if is_originally_enabled or verbose:  # type: ignore[truthy-function]
        _C._jit_set_onnx_log_enabled(True)
    try:
        yield
    finally:
        if not is_originally_enabled:  # type: ignore[truthy-function]
            _C._jit_set_onnx_log_enabled(False)


@deprecated(
    "The feature will be removed. Please remove usage of this function "
    "and implement equivalent logic if needed",
    category=None,
)
@contextlib.contextmanager
def exporter_context(model, mode: _C_onnx.TrainingMode, verbose: bool):
    """A context manager to temporarily set the training mode of ``model``
    to ``mode``, disable the Apex O2 hook, and set the ONNX logging verbosity.

    .. deprecated:: 2.7
        Please set training mode before exporting the model.
    """
    with (
        select_model_mode_for_export(model, mode) as mode_ctx,
        disable_apex_o2_state_dict_hook(model) as apex_ctx,
        setup_onnx_logging(verbose) as log_ctx,
    ):
        yield (mode_ctx, apex_ctx, log_ctx)


def export(
    model: torch.nn.Module | torch.jit.ScriptModule | torch.jit.ScriptFunction,
    args: tuple[Any, ...] | torch.Tensor,
    f: str,
    *,
    kwargs: dict[str, Any] | None = None,
    export_params: bool = True,
    verbose: bool = False,
    training: _C_onnx.TrainingMode = _C_onnx.TrainingMode.EVAL,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    operator_export_type: _C_onnx.OperatorExportTypes = _C_onnx.OperatorExportTypes.ONNX,
    opset_version: int | None = None,
    do_constant_folding: bool = True,
    dynamic_axes: Mapping[str, Mapping[int, str]]
    | Mapping[str, Sequence[int]]
    | None = None,
    keep_initializers_as_inputs: bool | None = None,
    custom_opsets: Mapping[str, int] | None = None,
    export_modules_as_functions: bool | Collection[type[torch.nn.Module]] = False,
    autograd_inlining: bool = True,
) -> None:
    r"""Exports a model into ONNX format.

    If ``model`` is not a :class:`torch.jit.ScriptModule` nor a
    :class:`torch.jit.ScriptFunction`, this runs
    ``model`` once in order to convert it to a TorchScript graph to be exported
    (the equivalent of :func:`torch.jit.trace`). Thus this has the same limited support
    for dynamic control flow as :func:`torch.jit.trace`.

    Args:
        model: The model to be exported.
        args:

            args can be structured either as:

            1. ONLY A TUPLE OF ARGUMENTS::

                args = (x, y, z)

            The tuple should contain model inputs such that ``model(*args)`` is a valid
            invocation of the model. Any non-Tensor arguments will be hard-coded into the
            exported model; any Tensor arguments will become inputs of the exported model,
            in the order they occur in the tuple.

            2. A TENSOR::

                args = torch.Tensor([1])

            This is equivalent to a 1-ary tuple of that Tensor.

            3. A TUPLE OF ARGUMENTS ENDING WITH A DICTIONARY OF NAMED ARGUMENTS::

                args = (x, {"y": input_y, "z": input_z})

            All but the last element of the tuple will be passed as non-keyword arguments,
            and named arguments will be set from the last element. If a named argument is
            not present in the dictionary, it is assigned the default value, or None if a
            default value is not provided.

            .. warning::
                This behavior will be deprecated in a future release. Please use the
                kwargs argument instead.

            .. note::
                If a dictionary is the last element of the args tuple, it will be
                interpreted as containing named arguments. In order to pass a dict as the
                last non-keyword arg, provide an empty dict as the last element of the args
                tuple. For example, instead of::

                    torch.onnx.export(
                        model,
                        (
                            x,
                            # WRONG: will be interpreted as named arguments
                            {y: z},
                        ),
                        "test.onnx.pb",
                    )

                Write::

                    torch.onnx.export(model, (x, {y: z}, {}), "test.onnx.pb")

        f: Path to the output ONNX model file. E.g. "model.onnx".
        kwargs: Named arguments to the model.
        export_params: If True, all parameters will
            be exported. Set this to False if you want to export an untrained model.
            In this case, the exported model will first take all of its parameters
            as arguments, with the ordering as specified by ``model.state_dict().values()``
        verbose: if True, prints a description of the
            model being exported to stdout. In addition, the final ONNX graph will include the
            field ``doc_string`` from the exported model which mentions the source code locations
            for ``model``. If True, ONNX exporter logging will be turned on.
        training:
            * ``TrainingMode.EVAL``: export the model in inference mode.
            * ``TrainingMode.PRESERVE``: export the model in inference mode if model.training is
                False and in training mode if model.training is True.
            * ``TrainingMode.TRAINING``: export the model in training mode. Disables optimizations
                which might interfere with training.
        input_names (list of str, default empty list): names to assign to the
            input nodes of the graph, in order.
        output_names (list of str, default empty list): names to assign to the
            output nodes of the graph, in order. These are labels only and do not
            affect the order of outputs. If the model returns a dictionary, outputs
            are flattened in the dictionary's iteration order regardless of the
            names specified here.
        operator_export_type (enum, default OperatorExportTypes.ONNX):

            .. warning::
                This option will be deprecated in a future release. Future exported
                graphs will always use the default opset domain.

            * ``OperatorExportTypes.ONNX``: Export all ops as regular ONNX ops
                (in the default opset domain).
            * ``OperatorExportTypes.ONNX_FALLTHROUGH``: Try to convert all ops
                to standard ONNX ops in the default opset domain. If unable to do so
                (e.g. because support has not been added to convert a particular torch op to ONNX),
                fall back to exporting the op into a custom opset domain without conversion. Applies
                to `custom ops <https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html>`_
                as well as ATen ops. For the exported model to be usable, the runtime must support
                these non-standard ops.
            * ``OperatorExportTypes.ONNX_ATEN``: All ATen ops (in the TorchScript namespace "aten")
                are exported as ATen ops (in opset domain "org.pytorch.aten").
                `ATen <https://pytorch.org/cppdocs/#aten>`_ is PyTorch's built-in tensor library, so
                this instructs the runtime to use PyTorch's implementation of these ops.

                Note: Caffe2 support has been removed as of PyTorch 2.9. This export type
                    is primarily intended for debugging ATen operator execution.

            * ``OperatorExportTypes.ONNX_ATEN_FALLBACK``: Try to export each ATen op
                (in the TorchScript namespace "aten") as a regular ONNX op. If we are unable to do so
                (e.g. because support has not been added to convert a particular torch op to ONNX),
                fall back to exporting an ATen op. See documentation on OperatorExportTypes.ONNX_ATEN for
                context.
                For example::

                    graph(%0 : Float):
                    %3 : int = prim::Constant[value=0]()
                    # conversion unsupported
                    %4 : Float = aten::triu(%0, %3)
                    # conversion supported
                    %5 : Float = aten::mul(%4, %0)
                    return (%5)

                Assuming ``aten::triu`` is not supported in ONNX, this will be exported as::

                    graph(%0 : Float):
                    %1 : Long() = onnx::Constant[value={0}]()
                    # not converted
                    %2 : Float = aten::ATen[operator="triu"](%0, %1)
                    # converted
                    %3 : Float = onnx::Mul(%2, %0)
                    return (%3)

                .. warning::

                    Note: Caffe2 support has been removed as of PyTorch 2.9.

        opset_version (int, default 18): The version of the
            `default (ai.onnx) opset <https://github.com/onnx/onnx/blob/master/docs/Operators.md>`_
            to target. Must be >= 7.
        do_constant_folding: Apply the constant-folding optimization.
            Constant-folding will replace some of the ops that have all constant inputs
            with pre-computed constant nodes.
        dynamic_axes:

            By default the exported model will have the shapes of all input and output tensors
            set to exactly match those given in ``args``. To specify axes of tensors as
            dynamic (i.e. known only at run-time), set ``dynamic_axes`` to a dict with schema:

            * KEY (str): an input or output name. Each name must also be provided in ``input_names`` or
                ``output_names``.
            * VALUE (dict or list): If a dict, keys are axis indices and values are axis names. If a
                list, each element is an axis index.

            For example::

                class SumModule(torch.nn.Module):
                    def forward(self, x):
                        return torch.sum(x, dim=1)


                torch.onnx.export(
                    SumModule(),
                    (torch.ones(2, 2),),
                    "onnx.pb",
                    input_names=["x"],
                    output_names=["sum"],
                )

            Produces::

                input {
                  name: "x"
                  ...
                      shape {
                        dim {
                          dim_value: 2  # axis 0
                        }
                        dim {
                          dim_value: 2  # axis 1
                ...
                output {
                  name: "sum"
                  ...
                      shape {
                        dim {
                          dim_value: 2  # axis 0
                ...

            While::

                torch.onnx.export(
                    SumModule(),
                    (torch.ones(2, 2),),
                    "onnx.pb",
                    input_names=["x"],
                    output_names=["sum"],
                    dynamic_axes={
                        # dict value: manually named axes
                        "x": {0: "my_custom_axis_name"},
                        # list value: automatic names
                        "sum": [0],
                    },
                )

            Produces::

                input {
                  name: "x"
                  ...
                      shape {
                        dim {
                          dim_param: "my_custom_axis_name"  # axis 0
                        }
                        dim {
                          dim_value: 2  # axis 1
                ...
                output {
                  name: "sum"
                  ...
                      shape {
                        dim {
                          dim_param: "sum_dynamic_axes_1"  # axis 0
                ...

        keep_initializers_as_inputs: If True, all the
            initializers (typically corresponding to parameters) in the
            exported graph will also be added as inputs to the graph. If False,
            then initializers are not added as inputs to the graph, and only
            the non-parameter inputs are added as inputs.
            This may allow for better optimizations (e.g. constant folding) by
            backends/runtimes.

            If True, `deduplicate_initializers` pass will not be executed. This means
            initializers with duplicated values will not be deduplicated and
            will be treated as distinct inputs to the graph. This allows different
            input initializers to be supplied at the runtime following export.

            If ``opset_version < 9``, initializers MUST be part of graph
            inputs and this argument will be ignored and the behavior will be
            equivalent to setting this argument to True.

        custom_opsets (dict[str, int], default empty dict): A dict with schema:

            * KEY (str): opset domain name
            * VALUE (int): opset version

            If a custom opset is referenced by ``model`` but not mentioned in this dictionary,
            the opset version is set to 1. Only custom opset domain name and version should be
            indicated through this argument.

        export_modules_as_functions: Flag to enable
            exporting all ``nn.Module`` forward calls as local functions in ONNX. Or a set to indicate the
            particular types of modules to export as local functions in ONNX.
            This feature requires ``opset_version`` >= 15, otherwise the export will fail. This is because
            ``opset_version`` < 15 implies IR version < 8, which means no local function support.
            Module variables will be exported as function attributes. There are two categories of function
            attributes.

            1. Annotated attributes: class variables that have type annotations via
            `PEP 526-style <https://www.python.org/dev/peps/pep-0526/#class-and-instance-variable-annotations>`_
            will be exported as attributes.
            Annotated attributes are not used inside the subgraph of ONNX local function because
            they are not created by PyTorch JIT tracing, but they may be used by consumers
            to determine whether or not to replace the function with a particular fused kernel.

            2. Inferred attributes: variables that are used by operators inside the module. Attribute names
            will have prefix "inferred::". This is to differentiate from predefined attributes retrieved from
            python module annotations. Inferred attributes are used inside the subgraph of ONNX local function.

            * ``False`` (default): export ``nn.Module`` forward calls as fine grained nodes.
            * ``True``: export all ``nn.Module`` forward calls as local function nodes.
            * Set of type of nn.Module: export ``nn.Module`` forward calls as local function nodes,
                only if the type of the ``nn.Module`` is found in the set.

        autograd_inlining: Flag used to control whether to inline autograd functions.
            Refer to https://github.com/pytorch/pytorch/pull/74765 for more details.

    Raises:
        :class:`torch.onnx.errors.CheckerError`: If the ONNX checker detects an invalid ONNX graph.
        :class:`torch.onnx.errors.UnsupportedOperatorError`: If the ONNX graph cannot be exported because it
            uses an operator that is not supported by the exporter.
        :class:`torch.onnx.errors.OnnxExporterError`: Other errors that can occur during export.
            All errors are subclasses of :class:`errors.OnnxExporterError`.
    """
    if operator_export_type != _C_onnx.OperatorExportTypes.ONNX:
        warnings.warn(
            "Setting `operator_export_type` to something other than default is deprecated. "
            "The option will be removed in a future release.",
            stacklevel=2,
            category=DeprecationWarning,
        )
    if training == _C_onnx.TrainingMode.TRAINING:
        warnings.warn(
            "Setting `training` to something other than default is deprecated. "
            "The option will be removed in a future release. Please set the training mode "
            "before exporting the model.",
            stacklevel=2,
            category=DeprecationWarning,
        )

    args = (args,) if isinstance(args, torch.Tensor) else args
    if kwargs is not None:
        args = args + (kwargs,)

    _export(
        model,
        args,
        f,
        export_params,
        verbose,
        training,
        input_names,
        output_names,
        operator_export_type=operator_export_type,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        dynamic_axes=dynamic_axes,
        keep_initializers_as_inputs=keep_initializers_as_inputs,
        custom_opsets=custom_opsets,
        export_modules_as_functions=export_modules_as_functions,
        autograd_inlining=autograd_inlining,
    )

    return None


def _is_constant_tensor_list(node) -> bool | None:
    if node.kind() != "prim::Constant":
        return False
    output_type = node.output().type()
    if output_type.isSubtypeOf(_C.ListType.ofTensors()):
        return True
    if output_type.isSubtypeOf(_C.ListType(_C.OptionalType.ofTensor())):
        return True


# ONNX can't handle constants that are lists of tensors, which can
# get generated in constant prop. So we split them back into prim::ListConstructs


def _split_tensor_list_constants(g, block) -> None:
    for node in block.nodes():
        for subblock in node.blocks():
            _split_tensor_list_constants(g, subblock)
        if _is_constant_tensor_list(node):
            inputs = []
            for val in node.output().toIValue():
                input = g.insertConstant(val)
                input.node().moveBefore(node)
                input.node().copyMetadata(node)
                inputs.append(input)

            lc = (
                g.create("prim::ListConstruct", inputs)
                .insertBefore(node)
                .output()
                .setType(_C.ListType.ofTensors())
            )
            lc.node().copyMetadata(node)
            node.output().replaceAllUsesWith(lc)


def _optimize_graph(
    graph: _C.Graph,
    operator_export_type: _C_onnx.OperatorExportTypes,
    _disable_torch_constant_prop: bool = False,
    fixed_batch_size: bool = False,
    params_dict=None,
    dynamic_axes=None,
    input_names=None,
    module=None,
):
    if params_dict is None:
        params_dict = {}

    # Inline everything
    _C._jit_pass_inline(graph)

    # Remove fork/wait nodes
    _C._jit_pass_inline_fork_wait(graph)
    _C._jit_pass_lint(graph)
    if GLOBALS.autograd_inlining:
        _C._jit_pass_onnx_autograd_function_process(graph)
    _C._jit_pass_lower_all_tuples(graph)

    # we now record some ops like ones/zeros
    # into a trace where we previously recorded constants.
    # use constant prop to maintain our current level of onnx support
    # without implementing symbolics for all of them
    if _disable_torch_constant_prop is False:
        _C._jit_pass_constant_propagation(graph)

    _split_tensor_list_constants(graph, graph)
    # run dce to eliminate dead parts of the graph that might have been
    # left behind by things like symbolic_override
    _C._jit_pass_dce(graph)
    _C._jit_pass_lint(graph)

    # CSE should improve perf when Autocast is used with disabled cache
    # Autocast is disabled due to a limitation on tracer as described at https://github.com/pytorch/pytorch/issues/84092
    # Must run before _C._jit_pass_erase_number_types to prevent type substitution
    if _C._jit_pass_cse(graph):
        _C._jit_pass_onnx_lint(graph)

    _C._jit_pass_canonicalize_graph_fuser_ops(graph)
    _C._jit_pass_lint(graph)
    _C._jit_pass_peephole(graph, True)
    _C._jit_pass_fuse_addmm(graph)
    _C._jit_pass_lint(graph)

    _C._jit_pass_peephole(graph, True)
    _C._jit_pass_lower_all_tuples(graph)
    # in _jit_pass_onnx, symbolic functions are called for each node for conversion.
    # However, there are nodes that cannot be converted without additional context.
    # For example, the number of outputs from split (and whether it is static or dynamic) is unknown
    # until the point where it is unpacked by listUnpack node.
    # This pass does a preprocess, and prepares the nodes such that enough context can be received
    # by the symbolic function.
    _C._jit_pass_onnx_remove_inplace_ops_for_onnx(graph, module)
    _C._jit_pass_onnx_preprocess(graph)

    # onnx does not support tuples, so try to remove them
    _C._jit_pass_lint(graph)

    # onnx only supports tensors, but 1 / 2 = 0.5 and tensor(1) / tensor(2) = 0
    _C._jit_pass_prepare_division_for_onnx(graph)

    _C._jit_pass_onnx_remove_print(graph)
    _C._jit_pass_onnx_preprocess_caffe2(graph)

    symbolic_helper._quantized_ops.clear()
    # Unpack quantized weights for conv and linear ops and insert into graph.
    _C._jit_pass_onnx_unpack_quantized_weights(graph, params_dict)
    # onnx only supports tensors, so we turn all out number types into tensors
    _C._jit_pass_erase_number_types(graph)
    if GLOBALS.onnx_shape_inference:
        input_names = [] if input_names is None else input_names
        dynamic_axes = {} if dynamic_axes is None else dynamic_axes
        _C._jit_pass_onnx_set_dynamic_input_shape(graph, dynamic_axes, input_names)
    _C._jit_pass_onnx_lint(graph)

    graph = _C._jit_pass_onnx(graph, operator_export_type)
    _C._jit_pass_onnx_lint(graph)
    _C._jit_pass_lint(graph)

    _C._jit_pass_onnx_scalar_type_analysis(
        graph, True, GLOBALS.export_onnx_opset_version
    )
    _C._jit_pass_lint(graph)

    _C._jit_pass_onnx_peephole(
        graph, GLOBALS.export_onnx_opset_version, fixed_batch_size
    )
    _C._jit_pass_lint(graph)

    # graph is not a valid jit graph anymore because types have been replaced
    # (e.g. int with Tensor), so it now contains operators that don't actually
    # exist. We can't run normal dead code elimination because it'd fail trying
    # to look up if an operator has side effects, but we can run a dead code
    # elimination variant that doesn't need to look up if an op has side effects.
    _C._jit_pass_dce_allow_deleting_nodes_with_side_effects(graph)
    _C._jit_pass_lint(graph)
    graph = _C._jit_pass_canonicalize(graph)
    _C._jit_pass_lint(graph)
    if GLOBALS.onnx_shape_inference:
        try:
            _C._jit_pass_onnx_graph_shape_type_inference(
                graph, params_dict, GLOBALS.export_onnx_opset_version
            )
        except RuntimeError:
            # NOTE: shape type inference error should not stop the export process
            # https://github.com/pytorch/pytorch/issues/132205
            pass

    return graph


def warn_on_static_input_change(input_states) -> None:
    """Warns that changes to input dictionaries and strings won't take effect in the traced ONNX graph.

    We accept dictionaries and strings as ONNX inputs, but they should be only for
    configuration use. we detect here if these inputs are modified, and if so we warn
    the user that the changes won't take effect in the traced ONNX graph.
    """
    for input, traced_input in zip(input_states[0], input_states[1]):
        if isinstance(input, dict):
            if list(input.keys()) != list(traced_input.keys()):
                warning = (
                    "We detected that you are modifying a dictionary that is an input to your "
                    "model. "
                    "Note that dictionaries are allowed as inputs in ONNX but they should be "
                    "handled with care. "
                    "Usages of dictionaries is not recommended, and should not be used except "
                    "for configuration use. "
                    "Also note that the order and values of the keys must remain the same. "
                )
                warnings.warn(warning, stacklevel=2)
        elif isinstance(input, str):
            if input != traced_input:
                warning = (
                    "The model seems to have string inputs/outputs. "
                    "Note that strings will not appear as inputs/outputs of the ONNX graph. "
                )
                warnings.warn(warning, stacklevel=2)


def _resolve_args_by_export_type(arg_name, arg_value, operator_export_type):
    """Resolves the arguments that are ignored when export_type != operator_export_type.ONNX."""
    return arg_value


def _decide_keep_init_as_input(
    keep_initializers_as_inputs: bool | None,
    operator_export_type: _C_onnx.OperatorExportTypes,
    opset_version: int,
):
    """Decides whether the initializers in the graph should be listed as ONNX graph inputs.

    This method encapsulates the logic to decide whether the initializers in the graph
    should be listed as ONNX graph inputs (i.e., whether to choose ONNX IR v3 or v4).
    If keep_initializers_as_inputs is not specified (None), then we decide whether to keep
    initializers as graph inputs (val_keep_init_as_ip) based on export type. If export type
    is ONNX, then do not keep initializers as input (val_keep_init_as_ip=False). For all other
    export types keep initializers as input (val_keep_init_as_ip=True).
    If keep_initializers_as_inputs is specified, then respect it. Unless opset version <= 8,
    in which case it must be ignored because for opset version <= 8, all initializers MUST be
    part of graph input (only ONNX IR v3 is allowed), i.e. val_keep_init_as_ip=True.

    Special handling is needed for opset version 8 or lower, because irrespective
    of user input for keep_initializers_as_inputs, the graph must follow ONNX IR v3
    semantics, i.e. all initializers must be listed as ONNX graph input.
    """

    if opset_version < 9:
        if keep_initializers_as_inputs is False:
            warnings.warn(
                "Setting 'keep_initializers_as_inputs=False' for opset version"
                "8 or lower would lead to an invalid ONNX graph. Therefore, "
                "'keep_initializers_as_inputs=False' is ignored during export."
                "Exported model will have initializers as graph inputs (compliant "
                " to ONNX IR v3).",
                stacklevel=2,
            )
        return True  # i.e. True == initializers are part of graph input (ONNX IR v3)
    val_keep_init_as_ip = (
        True if keep_initializers_as_inputs is None else keep_initializers_as_inputs
    )
    if (
        keep_initializers_as_inputs is None
        and operator_export_type is _C_onnx.OperatorExportTypes.ONNX
    ):
        val_keep_init_as_ip = False
    return val_keep_init_as_ip


def _decide_add_node_names(add_node_names, operator_export_type):
    return _resolve_args_by_export_type(
        "add_node_names", add_node_names, operator_export_type
    )


def _decide_constant_folding(do_constant_folding, operator_export_type, training):
    do_constant_folding = _resolve_args_by_export_type(
        "do_constant_folding", do_constant_folding, operator_export_type
    )
    if do_constant_folding and (
        training is not None and training is not _C_onnx.TrainingMode.EVAL
    ):
        warnings.warn(
            "It is recommended that constant folding be turned off ('do_constant_folding=False') "
            "when exporting the model in training-amenable mode, i.e. with 'training=TrainingMode.TRAIN' "
            "or 'training=TrainingMode.PRESERVE' (when model is in training mode). Otherwise, some "
            "learnable model parameters may not translate correctly in the exported ONNX model "
            "because constant folding mutates model parameters. Please consider "
            "turning off constant folding or setting the training=TrainingMode.EVAL.",
            stacklevel=2,
        )
    return do_constant_folding


def _signature(model) -> inspect.Signature:
    should_be_callable = getattr(model, "forward", model)
    if callable(should_be_callable):
        return inspect.signature(should_be_callable)
    raise ValueError("model has no forward method and is not callable")


def _decide_input_format(model, args):
    try:
        sig = _signature(model)
    except ValueError as e:
        warnings.warn(f"{e}, skipping _decide_input_format", stacklevel=2)
        return args
    try:
        ordered_list_keys = list(sig.parameters.keys())
        if ordered_list_keys[0] == "self":
            ordered_list_keys = ordered_list_keys[1:]
        args_dict: dict = {}
        if isinstance(args, list):
            args_list = args
        elif isinstance(args, tuple):
            args_list = list(args)
        else:
            args_list = [args]
        if isinstance(args_list[-1], dict):
            args_dict = args_list[-1]
            args_list = args_list[:-1]
        n_nonkeyword = len(args_list)
        for optional_arg in ordered_list_keys[n_nonkeyword:]:
            if optional_arg in args_dict:
                args_list.append(args_dict[optional_arg])
            # Check if this arg has a default value
            else:
                param = sig.parameters[optional_arg]
                if param.default != param.empty:
                    args_list.append(param.default)
        args = args_list if isinstance(args, list) else tuple(args_list)
    # Cases of models with no input args
    except IndexError:
        warnings.warn("No input args, skipping _decide_input_format", stacklevel=2)
    except Exception as e:
        warnings.warn(f"Skipping _decide_input_format\n {e.args[0]}", stacklevel=2)
    return args


def _trace(func, args, operator_export_type, return_outs=False):
    # Special case for common case of passing a single Tensor
    if isinstance(args, torch.Tensor):
        args = (args,)

    trace_graph, torch_out, inputs_states = torch.jit._get_trace_graph(
        func,
        args,
        strict=False,
        _force_outplace=False,
        _return_inputs_states=True,
    )
    warn_on_static_input_change(inputs_states)

    trace_graph = _optimize_graph(trace_graph, operator_export_type, params_dict={})
    if return_outs:
        return trace_graph, torch_out
    return trace_graph


def _trace_and_get_graph_from_model(model, args):
    # A basic sanity check: make sure the state_dict keys are the same
    # before and after running the model.  Fail fast!
    orig_state_dict_keys = torch.jit._unique_state_dict(model).keys()

    # Disable Autocast cache because it replaces kernel's weight and bias
    # by (undesired) constants.
    # No perf impact for when there are reused weights since https://github.com/pytorch/pytorch/pull/85665
    prev_autocast_cache_enabled = torch.is_autocast_cache_enabled()
    torch.set_autocast_cache_enabled(False)
    trace_graph, torch_out, inputs_states = torch.jit._get_trace_graph(
        model,
        args,
        strict=False,
        _force_outplace=False,
        _return_inputs_states=True,
    )
    torch.set_autocast_cache_enabled(prev_autocast_cache_enabled)

    warn_on_static_input_change(inputs_states)

    if orig_state_dict_keys != torch.jit._unique_state_dict(model).keys():
        raise RuntimeError(
            "state_dict changed after running the tracer; "
            "something weird is happening in your model!"
        )

    return trace_graph, torch_out


def _get_param_count_list(method_graph, args_params):
    param_count_list = []
    for input_, arg_params_ in zip(method_graph.inputs(), args_params):
        if "PackedParams" in str(input_.type()):
            in_vars, _ = torch.jit._flatten(arg_params_)
            param_count_list.append(len(in_vars))
        else:
            param_count_list.append(arg_params_ is not None)

    return param_count_list


def _check_flatten_did_not_remove(original, jit_flattened) -> None:
    """torch.jit._flatten removes None. Check if it did so in this case."""

    def flatten(x):
        if isinstance(x, (list, tuple)):
            for inner in x:
                yield from flatten(inner)
        elif isinstance(x, dict):
            for inner in x.values():
                yield from flatten(inner)
        else:
            yield x

    flattened_with_none = list(flatten(original))
    num_none = len(flattened_with_none) - len(jit_flattened)
    if num_none < 0:
        raise AssertionError(f"num_none must be >= 0, got {num_none}")
    if num_none:
        raise ValueError(
            f"args contained {num_none} None's after flattening. "
            "When exporting a ScriptModule or ScriptFunction, no args may "
            "be None because that breaks type propagation."
        )


def _create_jit_graph(
    model: torch.nn.Module | torch.jit.ScriptFunction, args: Sequence[Any]
) -> tuple[_C.Graph, list[_C.IValue], Any | None, _C.ScriptModule | None]:
    if isinstance(model, (torch.jit.ScriptFunction, torch.jit.ScriptModule)):
        flattened_args = tuple(torch.jit._flatten(tuple(args))[0])
        _check_flatten_did_not_remove(args, flattened_args)
        torch_out = None

        if isinstance(model, torch.jit.ScriptModule):
            try:
                graph = model.forward.graph  # type: ignore[attr-defined]
            except AttributeError as e:
                raise RuntimeError("'forward' method must be a script method") from e
            _C._jit_pass_onnx_function_substitution(graph)
            freezed_module = _C._freeze_module(
                cast(_C.ScriptModule, model._c), preserveParameters=True
            )
            module, params = _C._jit_onnx_list_model_parameters(freezed_module)
            method_graph = module._get_method("forward").graph
            args_params = tuple(args) + tuple(params)
            param_count_list = _get_param_count_list(method_graph, args_params)
            in_vars, _ = torch.jit._flatten(args_params)
            graph = _C._propagate_and_assign_input_shapes(
                method_graph, tuple(in_vars), param_count_list, False, False
            )
            return graph, params, torch_out, module

        # torch.jit.ScriptFunction
        params = []
        graph = model.graph
        _C._jit_pass_onnx_function_substitution(graph)
        param_count_list = _get_param_count_list(graph, args)
        graph = _C._propagate_and_assign_input_shapes(
            graph, flattened_args, param_count_list, False, False
        )
        return graph, params, torch_out, None

    graph, torch_out = _trace_and_get_graph_from_model(model, args)
    _C._jit_pass_onnx_lint(graph)
    state_dict = torch.jit._unique_state_dict(model)
    params = list(state_dict.values())
    graph_inputs = list(graph.inputs())
    user_input_num = len(graph_inputs) - len(state_dict)
    param_names = list(state_dict.keys())
    for i, inp in enumerate(graph_inputs):
        if i >= user_input_num:
            inp.setDebugName(param_names[i - user_input_num])
    _C._jit_pass_onnx_function_substitution(graph)
    return graph, params, torch_out, None


def _get_named_param_dict(graph, params):
    input_and_param_names = [val.debugName() for val in graph.inputs()]
    param_names = input_and_param_names[len(input_and_param_names) - len(params) :]
    _params_dict = dict(zip(param_names, params))
    return _params_dict


def _get_example_outputs(model, args):
    input_args = copy.deepcopy(args)
    input_kwargs = {}
    if input_args and isinstance(input_args[-1], dict):
        input_kwargs = input_args[-1]
        input_args = input_args[:-1]

    example_outputs = model(*input_args, **input_kwargs)
    if isinstance(example_outputs, list):
        example_outputs = [example_outputs]
    elif not isinstance(example_outputs, tuple):
        example_outputs = (example_outputs,)

    return example_outputs


_qtype_vtype_map = {
    torch.quint8: torch.uint8,
    torch.qint8: torch.int8,
    torch.qint32: torch.int32,
    torch.quint4x2: torch.int8,
}


def unpack_quantized_tensor(value, cast_onnx_accepted=True):
    if isinstance(value, torch.Tensor) and value.dtype in _qtype_vtype_map:
        q_value_dequantize = value.dequantize()
        q_scale = (
            torch.tensor(value.q_scale(), dtype=torch.double)
            if cast_onnx_accepted
            else torch.tensor(value.q_scale(), dtype=torch.float32)
        )
        q_zero_point = (
            torch.tensor(value.q_zero_point(), dtype=torch.int64)
            if cast_onnx_accepted
            else torch.tensor(value.q_zero_point(), dtype=_qtype_vtype_map[value.dtype])
        )
        q_value = q_value_dequantize / q_scale + q_zero_point
        q_value = q_value.to(dtype=_qtype_vtype_map[value.dtype])
        return q_value, q_scale, q_zero_point
    else:
        return (value,)


def _pre_trace_quant_model(model, args):
    r"""Returns `torch.jit.trace(model, args)` if model is quantized. Otherwise do nothing and return
    original model.

    This is due to https://github.com/pytorch/pytorch/issues/75761.
    """
    if any(
        hasattr(m, "_packed_params") for m in getattr(model, "modules", list)()
    ) or any(getattr(arg, "is_quantized", False) for arg in args):
        return torch.jit.trace(model, args)
    return model


def _model_to_graph(
    model,
    args,
    verbose=False,
    input_names=None,
    output_names=None,
    operator_export_type=_C_onnx.OperatorExportTypes.ONNX,
    do_constant_folding=True,
    _disable_torch_constant_prop=False,
    fixed_batch_size=False,
    training=_C_onnx.TrainingMode.EVAL,
    dynamic_axes=None,
) -> tuple[
    _C.Graph,
    dict[str, torch.Tensor],
    torch.Tensor
    | tuple[torch.Tensor, ...]
    | list[torch.Tensor]
    | dict[str, torch.Tensor]
    | Any
    | None,
]:
    """Converts model into an ONNX graph.

    Returns:
        graph: A TorchScript IR Graph with ONNX nodes.
        params_dict: Dict from input param name to param value.
        torch_out: The output tensors resulting from the trace of ``model``.
            If ``model`` is a :class:`torch.jit.ScriptModule` or :class:`torch.jit.ScriptFunction`,
            this will be None, since we are not doing any tracing.
    """
    # TODO: can we simplify this to always return a tuple of Tensor or None?

    # Special case for common case of passing a single Tensor
    if isinstance(args, (torch.Tensor, int, float, bool)):
        args = (args,)

    model = _pre_trace_quant_model(model, args)
    graph, params, torch_out, module = _create_jit_graph(model, args)
    params_dict = _get_named_param_dict(graph, params)

    try:
        graph = _optimize_graph(
            graph,
            operator_export_type,
            _disable_torch_constant_prop=_disable_torch_constant_prop,
            fixed_batch_size=fixed_batch_size,
            params_dict=params_dict,
            dynamic_axes=dynamic_axes,
            input_names=input_names,
            module=module,
        )
    except Exception:
        _C._jit_onnx_log("Torch IR graph at exception: ", graph)
        raise

    is_script = isinstance(model, (torch.jit.ScriptFunction, torch.jit.ScriptModule))
    if is_script:
        example_outputs = _get_example_outputs(model, args)
        example_outputs_final = ()
        for example_output in example_outputs:
            example_outputs_final += unpack_quantized_tensor(example_output)
        out_vars, desc = torch.jit._flatten(example_outputs_final)
        _C._jit_pass_onnx_assign_output_shape(
            graph,
            out_vars,
            desc,
            GLOBALS.onnx_shape_inference,
            is_script,
            GLOBALS.export_onnx_opset_version,
        )

    # NB: ONNX requires complete information about output types, which might be
    # erased by some optimizations, so we need to set it explicitly again.
    else:
        if not isinstance(torch_out, (list, tuple)):
            output_wrapped = [torch_out]
        else:
            output_wrapped = torch_out  # type: ignore[assignment]

        output_tensors, out_desc = torch.jit._flatten(tuple(output_wrapped))
        # assign_output_shape pass is not compatible with quantized outputs.
        # Quantized outputs are flattened to 3 values in ONNX, while packed as
        # single value in PyTorch.
        if not any(getattr(out, "is_quantized", False) for out in output_tensors):
            _C._jit_pass_onnx_assign_output_shape(
                graph,
                output_tensors,
                out_desc,
                GLOBALS.onnx_shape_inference,
                is_script,
                GLOBALS.export_onnx_opset_version,
            )

    _set_input_and_output_names(graph, input_names, output_names)
    params_dict = _get_named_param_dict(graph, params)

    if (
        do_constant_folding
        and GLOBALS.export_onnx_opset_version
        >= _constants.ONNX_CONSTANT_FOLDING_MIN_OPSET
    ):
        if training is None or training == _C_onnx.TrainingMode.EVAL:
            params_dict = _C._jit_pass_onnx_eval_peephole(graph, params_dict)

        params_dict = _C._jit_pass_onnx_constant_fold(
            graph, params_dict, GLOBALS.export_onnx_opset_version
        )
        _C._jit_pass_dce_allow_deleting_nodes_with_side_effects(graph)

    if GLOBALS.onnx_shape_inference:
        try:
            _C._jit_pass_onnx_graph_shape_type_inference(
                graph, params_dict, GLOBALS.export_onnx_opset_version
            )
        except RuntimeError:
            # NOTE: shape type inference error should not stop the export process
            # https://github.com/pytorch/pytorch/issues/132205
            pass

    params_dict = _C._jit_pass_onnx_eliminate_unused_items(graph, params_dict)

    # For ONNX opset < 9, constants only have three data types: float16, float, double.
    # In this pass transform constants of other data types to float/double + cast operator.
    if GLOBALS.export_onnx_opset_version < 9:
        _C._jit_pass_onnx_cast_all_constant_to_floating(graph)

    params_dict = _C._jit_pass_filter_non_tensor_arguments(params_dict)
    _C._jit_decay_packed_param_input_types(graph)

    # If output names lack a proper name and are identified only by their unique
    # give them a legible name for debugging purposes
    _apply_friendly_debug_names(graph, params_dict)

    return graph, params_dict, torch_out


@deprecated(
    "Unconvertible ops are not definitive. Please remove usage of this function"
)
def unconvertible_ops(
    model,
    args,
    training: _C_onnx.TrainingMode = _C_onnx.TrainingMode.EVAL,
    opset_version: int | None = None,
) -> tuple[_C.Graph, list[str]]:
    """Returns an approximated list of all ops that are not yet supported by :mod:`torch.onnx`.

    .. deprecated:: 2.5
        Unconvertible ops are not definitive. Please remove usage of this function.

    The list is approximated because some ops may be removed during the conversion
    process and don't need to be converted. Some other ops may have partial support
    that will fail conversion with particular inputs. Please open a Github Issue
    for op support requests.

    Args:
        model: Same as the `model` parameter in :func:`torch.onnx.export`.
        args: Same as the `args` parameter in :func:`torch.onnx.export`.
        training: Same as the `training` parameter in :func:`torch.onnx.export`.
        opset_version: Same as the `opset_version` parameter in :func:`torch.onnx.export`.

    Returns:
        The JIT graph and a list of unconvertible ops in the format of "domain::op".
    """

    opset_version = opset_version or _constants.ONNX_DEFAULT_OPSET
    GLOBALS.export_onnx_opset_version = opset_version

    try:
        with exporter_context(model, training, verbose=False):
            # Create a mostly clean JIT graph that contains the plain aten and
            # other ops we can check with the symbolic registry.
            # NOTE: We don't want to actually convert any ops to ONNX or run any
            # symbolic functions because there is a higher chance that a pass
            # fails or an unconvertible op messes up the graph during ONNX conversion.
            # This way we can always generate a list just by looking at the names
            # of the ops in the graph.
            args = _decide_input_format(model, args)
            model = _pre_trace_quant_model(model, args)
            graph, _, _, module = _create_jit_graph(model, args)
            _C._jit_pass_inline(graph)
            _C._jit_pass_onnx_remove_inplace_ops_for_onnx(graph, module)
            _C._jit_pass_erase_number_types(graph)
            _C._jit_pass_dce_allow_deleting_nodes_with_side_effects(graph)
    except Exception as e:
        raise errors.OnnxExporterError(
            "Failed to discover unconvertible ops because of errors during the JIT graph "
            "generation process."
        ) from e

    unsupported_ops = []
    for node in graph.nodes():
        domain_op = node.kind()
        if domain_op.startswith(("onnx::", "prim::")):
            # We consider onnx and prim ops as supported ops, even though some "prim"
            # ops are not implemented as symbolic functions, because they may be
            # eliminated in the conversion passes. Users may still see errors caused
            # by prim ops even though they don't show up in the list.
            continue
        if not registration.registry.is_registered_op(
            domain_op.rstrip("_"), opset_version
        ):
            # We consider all registered ops supported, even though some of them are
            # only partially supported, because there is not yet a good way to check
            # if an op is fully supported.
            # TODO(justinchuby): Create a way to check if an op is fully supported.
            unsupported_ops.append(domain_op)
    return graph, unsupported_ops


def _setup_trace_module_map(
    model: torch.nn.Module | torch.jit.ScriptModule,
    export_modules_as_functions: bool | Collection[type[torch.nn.Module]],
) -> set[str]:
    def __register_attribute_hook() -> None:
        attr_name = "_onnx_attrs"

        def _track_module_attributes_forward_pre_hook(module, input) -> None:
            setattr(module, attr_name, _get_module_attributes(module))

        def _track_module_attributes_forward_hook(module, input, output) -> None:
            tracing_state = _C._get_tracing_state()
            if not tracing_state:
                return

            graph = tracing_state.graph()
            onnx_attrs = {}
            if hasattr(module, attr_name):
                onnx_attrs = getattr(module, attr_name)
                delattr(module, attr_name)

            _C._jit_pass_onnx_track_scope_attributes(graph, onnx_attrs)

        for m in model.modules():
            m.register_forward_hook(_track_module_attributes_forward_hook)
            m.register_forward_pre_hook(_track_module_attributes_forward_pre_hook)

    def _unqualified_variable_name(qualified_name: str) -> str:
        """
        Parse qualified variable name and return the unqualified version.

        Pure numeric atoms are considered inadequate, so this function will look past them,
        and start from the first non-numeric atom.

        Example:
            >>> _unqualified_variable_name("__main__.Foo.bar")
            'bar'
            >>> _unqualified_variable_name("__main__.Foo.bar.0")
            'bar.0'
        """
        name_atoms = qualified_name.split(".")
        for i, atom in reversed(list(enumerate(name_atoms))):
            if not atom.isnumeric():
                return ".".join(name_atoms[i:])
        return qualified_name

    trace_module_map = {
        _m: torch._C._jit_onnx_create_full_scope_name(
            torch.typename(type(_m)), _unqualified_variable_name(_n)
        )
        for _n, _m in model.named_modules()
    }
    torch.jit._trace._trace_module_map = trace_module_map
    if isinstance(export_modules_as_functions, bool) and export_modules_as_functions:
        module_typenames = {torch.typename(type(module)) for module in trace_module_map}
    elif isinstance(export_modules_as_functions, set) and export_modules_as_functions:

        def _find_typename(v):
            if isinstance(v, type):
                return torch.typename(v)
            else:
                raise RuntimeError(
                    "Only type of the `nn.Module` should be "
                    "passed in the set for argument `export_modules_as_functions`. "
                    f"Got `{type(v).__name__}`."
                )

        module_typenames = {_find_typename(v) for v in export_modules_as_functions}
    else:
        module_typenames = set()

    if module_typenames:
        __register_attribute_hook()

    return module_typenames


def _reset_trace_module_map() -> None:
    torch.jit._trace._trace_module_map = None
    _C._jit_pass_onnx_clear_scope_records()


def _get_module_attributes(module):
    annotations = typing.get_type_hints(type(module))
    base_m_annotations = typing.get_type_hints(torch.nn.Module)
    [annotations.pop(k, None) for k in base_m_annotations]
    # Check whether module attributes can be accessed. Some classes
    # define attributes but don't provide access to them in their
    # constructor.
    #
    # For example, torch.nn.Embedding has the `freeze` variable and its
    # type specified in the class but the attribute is not created in the
    # constructor. In other words, there is no `self.freeze = <True | False>`
    # in the constructor.
    #
    # Reference: https://github.com/pytorch/pytorch/blob/92de1d322223fb5584e384971b32c46b93bc2f4b/torch/nn/modules/sparse.py#L120
    attrs = {}
    for k in annotations:
        try:
            attrs[k] = getattr(module, k)
        except AttributeError:
            _C._jit_onnx_log(f"Skipping module attribute '{k}'")
            continue
    return attrs


def _trigger_symbolic_function_registration() -> None:
    """Trigger the registration of symbolic functions for all supported opsets."""

    from torch.onnx._internal.torchscript_exporter import (  # noqa: F401
        symbolic_opset10,
        symbolic_opset11,
        symbolic_opset12,
        symbolic_opset13,
        symbolic_opset14,
        symbolic_opset15,
        symbolic_opset16,
        symbolic_opset17,
        symbolic_opset18,
        symbolic_opset19,
        symbolic_opset20,
        symbolic_opset7,
        symbolic_opset8,
        symbolic_opset9,
    )


def _export(
    model,
    args,
    f,
    export_params=True,
    verbose=False,
    training=_C_onnx.TrainingMode.EVAL,
    input_names=None,
    output_names=None,
    operator_export_type=_C_onnx.OperatorExportTypes.ONNX,
    export_type=None,
    opset_version=None,
    do_constant_folding=True,
    dynamic_axes=None,
    keep_initializers_as_inputs=None,
    fixed_batch_size=False,
    custom_opsets=None,
    add_node_names=True,
    onnx_shape_inference=True,
    export_modules_as_functions: Any = False,
    autograd_inlining=True,
):
    if GLOBALS.in_onnx_export is not False:
        raise AssertionError("GLOBALS.in_onnx_export must be False")

    _trigger_symbolic_function_registration()

    if isinstance(model, torch.nn.DataParallel):
        raise ValueError(
            "torch.nn.DataParallel is not supported by ONNX "
            "exporter, please use 'attribute' module to "
            "unwrap model from torch.nn.DataParallel. Try "
            "torch.onnx.export(model.module, ...)"
        )

    GLOBALS.onnx_shape_inference = onnx_shape_inference

    if opset_version is None:
        opset_version = _constants.ONNX_DEFAULT_OPSET

    if opset_version > _constants.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET:
        warnings.warn(
            f"Exporting to ONNX opset version {opset_version} is not supported. "
            f"by 'torch.onnx.export()'. "
            f"The highest opset version supported is {_constants.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET}. "
            f"To use a newer opset version, consider 'torch.onnx.export(..., dynamo=True)'. ",
            stacklevel=2,
            category=errors.OnnxExporterWarning,
        )

    if export_modules_as_functions and opset_version < 15:
        raise ValueError(
            "`export_modules_as_functions` is not supported for `opset_version` < 15."
            "This is because `opset_version` < 15 implies IR version < 8, which means "
            "no local function support. "
        )
    if not operator_export_type:
        operator_export_type = _C_onnx.OperatorExportTypes.ONNX

    # By default, training=TrainingMode.EVAL,
    # which is good because running a model in training mode could result in
    # internal buffers getting updated, dropout getting applied, etc.
    # If you really know what you're doing, you can turn
    # training=TrainingMode.TRAINING or training=TrainingMode.PRESERVE,
    # (to preserve whatever the original training mode was.)
    GLOBALS.export_onnx_opset_version = opset_version
    GLOBALS.operator_export_type = operator_export_type

    try:
        GLOBALS.in_onnx_export = True
        _autograd_inlining_previous = GLOBALS.autograd_inlining
        GLOBALS.autograd_inlining = autograd_inlining

        module_typenames_to_export_as_functions: set[str] = set()
        if isinstance(model, (torch.nn.Module, torch.jit.ScriptModule)):
            module_typenames_to_export_as_functions = _setup_trace_module_map(
                model, export_modules_as_functions
            )

        with exporter_context(model, training, verbose):
            val_keep_init_as_ip = _decide_keep_init_as_input(
                keep_initializers_as_inputs,
                operator_export_type,
                opset_version,
            )
            val_add_node_names = _decide_add_node_names(
                add_node_names, operator_export_type
            )
            val_do_constant_folding = _decide_constant_folding(
                do_constant_folding, operator_export_type, training
            )
            # Normally f can be a file-like object, but for large models, the external data format requires a
            # valid `model_file_location`. Code in export.cpp will enforce this.
            if isinstance(f, str):
                model_file_location = f
            else:
                model_file_location = ""
            args = _decide_input_format(model, args)
            if dynamic_axes is None:
                dynamic_axes = {}
            _validate_dynamic_axes(dynamic_axes, model, input_names, output_names)

            graph, params_dict, torch_out = _model_to_graph(
                model,
                args,
                verbose,
                input_names,
                output_names,
                operator_export_type,
                val_do_constant_folding,
                fixed_batch_size=fixed_batch_size,
                training=training,
                dynamic_axes=dynamic_axes,
            )

            if custom_opsets is None:
                custom_opsets = {}

            _C._jit_pass_dce_allow_deleting_nodes_with_side_effects(graph)
            node_attr_to_name = {}  # type: ignore[var-annotated]
            if module_typenames_to_export_as_functions:
                # NOTE: cannot call DCE after this pass. DCE will remove function definition nodes.
                node_attr_to_name = _C._jit_pass_onnx_function_extraction(
                    graph,
                    module_typenames_to_export_as_functions,
                    list(params_dict.keys()),
                )

            if keep_initializers_as_inputs is not True:
                params_dict = _C._jit_pass_onnx_deduplicate_initializers(  # type: ignore[assignment]
                    graph,
                    params_dict,  # type: ignore[arg-type]
                    getattr(model, "training", False),  # type: ignore[arg-type]
                )
            _C._jit_pass_onnx_assign_scoped_names_for_node_and_value(graph)
            defer_weight_export = False
            if export_params:
                (
                    proto,
                    export_map,
                    _val_use_external_data_format,
                    _node_names,
                ) = graph._export_onnx(  # type: ignore[attr-defined]
                    params_dict,
                    opset_version,
                    dynamic_axes,
                    defer_weight_export,
                    operator_export_type,
                    not verbose,
                    val_keep_init_as_ip,
                    custom_opsets,
                    val_add_node_names,
                    model_file_location,
                    node_attr_to_name,
                )
            else:
                (
                    proto,
                    export_map,
                    _,
                    _,
                ) = graph._export_onnx(  # type: ignore[attr-defined]
                    {},
                    opset_version,
                    dynamic_axes,
                    defer_weight_export,
                    operator_export_type,
                    not verbose,
                    val_keep_init_as_ip,
                    custom_opsets,
                    val_add_node_names,
                    model_file_location,
                    node_attr_to_name,
                )
            # insert function_proto into model_proto.
            proto = onnx_proto_utils._add_onnxscript_fn(
                proto,
                custom_opsets,
            )
            if verbose:
                _C._jit_onnx_log("Exported graph: ", graph)
            onnx_proto_utils._export_file(proto, f, export_map)
    finally:
        if not GLOBALS.in_onnx_export:
            raise AssertionError("GLOBALS.in_onnx_export must be True")
        GLOBALS.in_onnx_export = False
        GLOBALS.autograd_inlining = _autograd_inlining_previous
        _reset_trace_module_map()

    return torch_out


def _apply_friendly_debug_names(graph, params) -> None:
    for n in graph.nodes():
        for v in n.inputs():
            old_name = v.debugName()
            if old_name != str(v.unique()):
                continue
            new_name = f"{n.kind()}_{v.unique()}"
            v.setDebugName(new_name)
            if old_name in params:
                params[new_name] = params.pop(old_name)


def _set_input_and_output_names(graph, input_names, output_names) -> None:
    def set_names(node_list, name_list, descriptor) -> None:
        if name_list is None:
            return
        if len(name_list) > len(node_list):
            raise RuntimeError(
                f"number of {descriptor} names provided ({len(name_list)}) "
                f"exceeded number of {descriptor}s ({len(node_list)})"
            )

        # Mark if the output node DebugName is set before.
        output_node_set = set()
        for i, (name, node) in enumerate(zip(name_list, node_list)):
            # Duplicated output node, insert onnx::Identity to avoid setting the same DebugName after setDebugName().
            if descriptor == "output":
                if node in output_node_set:
                    identity_node = graph.create("onnx::Identity")
                    identity_node.insertAfter(node.node())
                    identity_node.addInput(node)
                    identity_node.output().setType(node.type())
                    graph.return_node().replaceInput(i, identity_node.output())
                    node = identity_node.output()
                output_node_set.add(node)

            if node.debugName() != name:
                node.setDebugName(name)

    set_names(list(graph.inputs()), input_names, "input")
    set_names(list(graph.outputs()), output_names, "output")


def _run_symbolic_method(g, op_name, symbolic_fn, args):
    r"""
    This trampoline function gets invoked for every symbolic method
    call from C++.
    """
    try:
        graph_context = jit_utils.GraphContext(
            graph=g,
            block=g.block(),
            opset=GLOBALS.export_onnx_opset_version,
            original_node=None,  # type: ignore[arg-type]
            params_dict=_params_dict,
            env={},
            values_in_env=set(),
            new_nodes=[],
        )
        return symbolic_fn(graph_context, *args)
    except TypeError as e:
        # Handle the specific case where we didn't successfully dispatch
        # to symbolic_fn.  Otherwise, the backtrace will have the clues
        # you need.
        e.args = (f"{e.args[0]} (occurred when translating {op_name})",)
        raise


def _add_block(node: _C.Node) -> _C.Block:
    return node.addBlock()


def _add_input_to_block(block: _C.Block):
    return block.addInputToBlock()  # type: ignore[attr-defined]


def _add_output_to_block(block: _C.Block, value: _C.Value) -> int:
    return block.registerOutput(value)


def _should_aten_fallback(
    name: str, opset_version: int, operator_export_type: _C_onnx.OperatorExportTypes
) -> bool:
    # For all builds, if domain=="aten" and operator_export_type==ONNX_ATEN,
    #   an aten::ATen operator is created regardless of symbolics existence

    is_exportable_aten_op = registration.registry.is_registered_op(name, opset_version)
    is_onnx_aten_export = operator_export_type == _C_onnx.OperatorExportTypes.ONNX_ATEN
    is_aten_fallback_export = (
        operator_export_type == _C_onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    )

    if not name.startswith("aten::"):
        return False

    if is_onnx_aten_export or (is_aten_fallback_export and not is_exportable_aten_op):
        return True

    return False


def _get_aten_op_overload_name(n: _C.Node) -> str:
    # Returns `overload_name` attribute to ATen ops
    schema = n.schema()
    if not schema.startswith("aten::"):
        return ""
    return _C.parse_schema(schema).overload_name


def _run_symbolic_function(
    graph: _C.Graph,
    block: _C.Block,
    node: _C.Node,
    inputs: Any,
    env: dict[_C.Value, _C.Value],
    values_in_env: set[_C.Value],
    new_nodes: list[_C.Node],
    operator_export_type=_C_onnx.OperatorExportTypes.ONNX,
) -> _C.Value | Sequence[_C.Value | None] | None:
    """Runs a symbolic function.

    The function is used in C++ to export the node to ONNX.

    Returns:
        A single or a tuple of Values.
        None when the node gets cloned as is into the new graph.
    """

    opset_version = GLOBALS.export_onnx_opset_version

    # See Note [Export inplace]
    node_kind = node.kind()
    if node_kind.endswith("_"):
        # Treat relu_ -> relu; add_ -> add etc.
        ns_op_name = node_kind[:-1]
    else:
        ns_op_name = node_kind

    namespace, op_name = jit_utils.parse_node_kind(ns_op_name)

    graph_context = jit_utils.GraphContext(
        graph=graph,
        block=block,
        opset=opset_version,
        original_node=node,
        params_dict=_params_dict,
        env=env,
        values_in_env=values_in_env,
        new_nodes=new_nodes,
    )

    # Direct ATen export requested
    if _should_aten_fallback(ns_op_name, opset_version, operator_export_type):
        attrs = {
            k + "_" + node.kindOf(k)[0]: symbolic_helper._node_get(node, k)
            for k in node.attributeNames()
        }
        outputs = node.outputsSize()
        attrs["outputs"] = outputs
        return graph_context.aten_op(
            op_name,
            *inputs,
            overload_name=_get_aten_op_overload_name(node),
            **attrs,
        )

    try:
        domain = namespace
        symbolic_function_name = f"{domain}::{op_name}"

        symbolic_function_group = registration.registry.get_function_group(
            symbolic_function_name
        )
        if symbolic_function_group is not None:
            symbolic_fn = symbolic_function_group.get(opset_version)
            if symbolic_fn is not None:
                # TODO Wrap almost identical attrs assignment or comment the difference.
                attrs = {
                    k: symbolic_helper._node_get(node, k) for k in node.attributeNames()
                }
                return symbolic_fn(graph_context, *inputs, **attrs)

        attrs = {
            k + "_" + node.kindOf(k)[0]: symbolic_helper._node_get(node, k)
            for k in node.attributeNames()
        }
        if namespace == "onnx":
            # Clone node to trigger ONNX shape inference
            return graph_context.op(
                op_name, *inputs, **attrs, outputs=node.outputsSize()
            )  # type: ignore[attr-defined]

        raise errors.UnsupportedOperatorError(
            symbolic_function_name,
            opset_version,
            symbolic_function_group.get_min_supported()
            if symbolic_function_group
            else None,
        )

    except RuntimeError:
        if operator_export_type == _C_onnx.OperatorExportTypes.ONNX_FALLTHROUGH:
            return None
        elif operator_export_type == _C_onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
            # Emit ATen op when `operator_export_type==ONNX_ATEN_FALLBACK`
            attrs = {
                k + "_" + node.kindOf(k)[0]: symbolic_helper._node_get(node, k)
                for k in node.attributeNames()
            }
            return graph_context.aten_op(
                op_name,
                *inputs,
                overload_name=_get_aten_op_overload_name(node),
                **attrs,
            )
        raise
    except TypeError as e:
        # Handle the specific case where we didn't successfully dispatch.
        # Otherwise, the backtrace will have the clues you need.
        e.args = (f"{e.args[0]} \n(Occurred when translating {op_name}).",)
        raise


def _verify_custom_op_name(symbolic_name: str) -> None:
    if not re.match(r"^[a-zA-Z0-9-_]+::[a-zA-Z-_]+[a-zA-Z0-9-_]*$", symbolic_name):
        raise errors.OnnxExporterError(
            f"Failed to register operator {symbolic_name}. "
            "The symbolic name must match the format domain::name, "
            "and should start with a letter and contain only "
            "alphanumeric characters"
        )

    ns, _ = jit_utils.parse_node_kind(symbolic_name)
    if ns == "onnx":
        raise ValueError(
            f"Failed to register operator {symbolic_name}. {ns} domain cannot be modified."
        )


def register_custom_op_symbolic(
    symbolic_name: str,
    symbolic_fn: Callable,
    opset_version: int,
) -> None:
    """Registers a symbolic function for a custom operator.

    When the user registers symbolic for custom/contrib ops,
    it is highly recommended to add shape inference for that operator via setType API,
    otherwise the exported graph may have incorrect shape inference in some extreme cases.
    An example of setType is `test_aten_embedding_2` in `test_operators.py`.

    See "Custom Operators" in the module documentation for an example usage.

    Args:
        symbolic_name (str): The name of the custom operator in "<domain>::<op>"
            format.
        symbolic_fn (Callable): A function that takes in the ONNX graph and
            the input arguments to the current operator, and returns new
            operator nodes to add to the graph.
        opset_version (int): The ONNX opset version in which to register.
    """
    if symbolic_name.startswith("::"):
        symbolic_name = f"aten{symbolic_name}"

    _verify_custom_op_name(symbolic_name)

    registration.custom_onnx_symbolic(symbolic_name, opset_version)(symbolic_fn)


def unregister_custom_op_symbolic(symbolic_name: str, opset_version: int) -> None:
    """Unregisters ``symbolic_name``.

    See "Custom Operators" in the module documentation for an example usage.

    Args:
        symbolic_name (str): The name of the custom operator in "<domain>::<op>"
            format.
        opset_version (int): The ONNX opset version in which to unregister.
    """
    if symbolic_name.startswith("::"):
        symbolic_name = f"aten{symbolic_name}"

    _verify_custom_op_name(symbolic_name)

    registration.registry.unregister(symbolic_name, opset_version)


def _validate_dynamic_axes(dynamic_axes, model, input_names, output_names) -> None:
    """Ensures dynamic axes argument follows the expected format."""
    if len(dynamic_axes) == 0:
        return

    if hasattr(model, "graph"):
        # Extracting set of valid input/output names that shall be used for dynamic_axes
        if (input_names is None) or len(input_names) == 0:
            input_names = [x.debugName() for x in model.graph.inputs()]
        if (output_names is None) or len(output_names) == 0:
            output_names = [y.debugName() for y in model.graph.outputs()]

    valid_names = set((input_names or []) + (output_names or []))

    # If dynamic axes are provided as a list rather than dictionary, they should
    # first get converted to a dictionary in expected format. If desired axes names
    # are not provided for dynamic axes, automatic names shall be generated for
    # provided dynamic axes of specified input/output
    for key, value in dynamic_axes.items():
        if key not in valid_names:
            warnings.warn(
                f"Provided key {key} for dynamic axes is not a valid input/output name",
                stacklevel=2,
            )
        if isinstance(value, list):
            warnings.warn(
                "No names were found for specified dynamic axes of provided input."
                f"Automatically generated names will be applied to each dynamic axes of input {key}",
                stacklevel=2,
            )

            value_dict = {}
            for i, x in enumerate(value):
                if not isinstance(x, int):
                    raise ValueError(
                        "The type of axis index is expected to be an integer"
                    )
                if x in value_dict:
                    warnings.warn(
                        f"Duplicate dynamic axis index {x} was provided for input {key}.",
                        stacklevel=2,
                    )
                else:
                    value_dict[x] = str(key) + "_dynamic_axes_" + str(i + 1)
            dynamic_axes[key] = value_dict


def model_signature(model: torch.nn.Module | Callable) -> inspect.Signature:
    return inspect.signature(
        model.forward if isinstance(model, torch.nn.Module) else model
    )
<|endoftext|># mypy: allow-untyped-defs
# mypy: disable-error-code=arg-type
"""This file exports ONNX ops for opset 11."""

from __future__ import annotations

import functools
import sys
import warnings
from typing import TYPE_CHECKING

import torch
from torch import _C
from torch._C import _onnx as _C_onnx
from torch.onnx import errors
from torch.onnx._internal.torchscript_exporter import (
    _type_utils,
    jit_utils,
    registration,
    symbolic_helper,
    symbolic_opset10 as opset10,
    symbolic_opset9 as opset9,
    utils,
)


if TYPE_CHECKING:
    from collections.abc import Sequence


# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in README.md

__all__ = [
    "add",
    "append",
    "arange",
    "argsort",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "cat",
    "chunk",
    "clamp_max",
    "clamp_min",
    "clamp",
    "constant_pad_nd",
    "cumsum",
    "Delete",
    "embedding_bag",
    "embedding_renorm",
    "flatten",
    "gather",
    "hardtanh",
    "hstack",
    "im2col",
    "index_fill",
    "index",
    "index_copy",
    "index_put",
    "insert",
    "linalg_det",
    "linalg_vector_norm",
    "logdet",
    "masked_scatter",
    "masked_select",
    "mm",
    "narrow",
    "normal",
    "pad",
    "pixel_shuffle",
    "pop",
    "prim_constant_chunk",
    "reflection_pad",
    "relu6",
    "remainder",
    "replication_pad",
    "round",
    "scatter",
    "select",
    "size",
    "sort",
    "split_with_sizes",
    "split",
    "squeeze",
    "stack",
    "topk",
    "unbind",
    "unique_dim",
    "unsqueeze",
    "vstack",
]

_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=11)


@_onnx_symbolic("aten::hardtanh")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "f", "f")
def hardtanh(g: jit_utils.GraphContext, self: _C.Value, min_val: float, max_val: float):
    scalar_type = _type_utils.JitScalarType.from_value(
        self, _type_utils.JitScalarType.FLOAT
    )
    min_val = g.op(
        "Constant",
        value_t=torch.tensor(min_val, dtype=scalar_type.dtype()),
    )
    max_val = g.op(
        "Constant",
        value_t=torch.tensor(max_val, dtype=scalar_type.dtype()),
    )
    return symbolic_helper._op_with_optional_float_cast(
        g, "Clip", self, min_val, max_val, opset_before=12
    )


@_onnx_symbolic("aten::clamp")
def clamp(g: jit_utils.GraphContext, self, min, max):
    def _cast_if_not_none(tensor, dtype):
        if tensor is not None and not symbolic_helper._is_none(tensor):
            return g.op(
                "Cast",
                tensor,
                to_i=dtype.onnx_type(),
            )
        else:
            return tensor

    scalar_type = _type_utils.JitScalarType.from_value(
        self, _type_utils.JitScalarType.UNDEFINED
    )
    if scalar_type != _type_utils.JitScalarType.UNDEFINED:
        min = _cast_if_not_none(min, scalar_type)
        max = _cast_if_not_none(max, scalar_type)

    if symbolic_helper._is_none(min):
        return clamp_max(g, self, max)
    elif symbolic_helper._is_none(max):
        return clamp_min(g, self, min)
    else:
        if (
            symbolic_helper._get_tensor_rank(min) == 0
            and symbolic_helper._get_tensor_rank(max) == 0
        ):
            return symbolic_helper._op_with_optional_float_cast(
                g, "Clip", self, min, max, opset_before=12
            )
        else:
            return clamp_max(g, clamp_min(g, self, min), max)


@_onnx_symbolic("aten::clamp_min")
@symbolic_helper.parse_args("v", "v")
def clamp_min(g: jit_utils.GraphContext, self, min):
    min = g.op("Cast", min, to_i=_type_utils.JitScalarType.from_value(self).onnx_type())
    if symbolic_helper._get_tensor_rank(min) == 0:
        max = opset9.unused(g)
        return symbolic_helper._op_with_optional_float_cast(
            g, "Clip", self, min, max, opset_before=12
        )
    else:
        return symbolic_helper._op_with_optional_float_cast(
            g, "Max", self, min, opset_before=12
        )


@_onnx_symbolic("aten::clamp_max")
@symbolic_helper.parse_args("v", "v")
def clamp_max(g: jit_utils.GraphContext, self, max):
    max = g.op("Cast", max, to_i=_type_utils.JitScalarType.from_value(self).onnx_type())
    if symbolic_helper._get_tensor_rank(max) == 0:
        min = opset9.unused(g)
        return symbolic_helper._op_with_optional_float_cast(
            g, "Clip", self, min, max, opset_before=12
        )
    else:
        return symbolic_helper._op_with_optional_float_cast(
            g, "Min", self, max, opset_before=12
        )


@_onnx_symbolic("aten::relu6")
def relu6(g: jit_utils.GraphContext, input):
    scalar_type = _type_utils.JitScalarType.from_value(
        input, _type_utils.JitScalarType.FLOAT
    )
    min_val = g.op(
        "Constant",
        value_t=torch.tensor(0, dtype=scalar_type.dtype()),
    )
    max_val = g.op(
        "Constant",
        value_t=torch.tensor(6, dtype=scalar_type.dtype()),
    )
    return clamp(g, input, min_val, max_val)


@_onnx_symbolic("aten::select")
# Opset 11 gather accepts negative indices
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "i", "v")
def select(g: jit_utils.GraphContext, self, dim, index):
    return g.op("Gather", self, index, axis_i=dim)


@_onnx_symbolic("aten::index_put")
def index_put(
    g: jit_utils.GraphContext, self, indices_list_value, values, accumulate=False
):
    if symbolic_helper._is_packed_list(indices_list_value):
        indices_list = symbolic_helper._unpack_list(indices_list_value)
    else:
        indices_list = [indices_list_value]
    accumulate = symbolic_helper._parse_arg(accumulate, "b")

    if len(indices_list) == 0:
        return values

    if len(indices_list) > 1:
        for idx_ in range(len(indices_list)):
            if symbolic_helper._is_bool(indices_list[idx_]):
                indices_list[idx_] = g.op("NonZero", indices_list[idx_])
        index = indices_list[0]

        for ind in indices_list[1:]:
            index = opset9.add(g, index, ind)
        broadcast_index_shape = g.op("Shape", index)
        indices_list = [
            symbolic_helper._unsqueeze_helper(
                g, opset9.expand(g, ind, broadcast_index_shape, None), [-1]
            )
            for ind in indices_list
        ]
        index = g.op("Concat", *indices_list, axis_i=-1)
    else:
        # Replace index_put node with masked_scatter or masked_fill
        # when inputs to the index_put node contains a single boolean input.
        #
        # index_put -> masked_fill
        #   * input index contains single tensor of Bool type (e.g.: %24 <- %23).
        #   * input value contains single element (e.g.: %18).
        #
        # Torch IR
        #   %mask : Float(2, 2, 2, strides=[4, 2, 1], requires_grad=0, device=cpu) = aten::clone(%0, %6)
        #   %16 : Bool(2, 2, 2, strides=[4, 2, 1], requires_grad=0, device=cpu) =
        #               aten::to(%8, %26, %27, %11, %12, %28, %29, %15)
        #   %18 : Float(requires_grad=0, device=cpu) = prim::Constant[value={1}]()
        #   %23 : Bool(8, strides=[1], device=cpu) = aten::view(%16, %22)
        #   %24 : Tensor?[] = prim::ListConstruct(%23)
        #   %25 : Float(2, 2, 2, strides=[4, 2, 1], requires_grad=0, device=cpu) =
        #                aten::index_put(%mask, %24, %18, %30)
        #   return (%25)
        #
        #
        # index_put -> masked_scatter
        #   * input index contains single tensor of Bool type (e.g.: %32 <- %31).
        #   * input value contains multiple elements (e.g.: %28).
        #
        # Torch IR
        #   %mask : Float(2, 2, 2, strides=[4, 2, 1], requires_grad=0, device=cpu) = aten::clone(%0, %6)
        #   %28 : Float(8, strides=[1], requires_grad=0, device=cpu)
        #                = prim::Constant[value= 1  1  1  1  1  1  1  1 [ CPUFloatType{8} ]]()
        #   %15 : Bool(2, 2, 2, strides=[4, 2, 1], requires_grad=0, device=cpu)
        #                = aten::ne(%mask, %some_const)
        #   %23 : Bool(2, 2, 2, strides=[4, 2, 1], requires_grad=0, device=cpu)
        #                = aten::to(%15, %34, %35, %18, %19, %36, %37, %22)
        #   %38 : Long(requires_grad=0, device=cpu) = prim::Constant[value={0}]()
        #   %30 : int[] = prim::Constant[value=[-1]]()
        #   %31 : Bool(8, strides=[1], device=cpu) = aten::view(%23, %30)
        #   %32 : Tensor?[] = prim::ListConstruct(%31)
        #   %33 : Float(2, 2, 2, strides=[4, 2, 1], requires_grad=0, device=cpu)
        #               = aten::index_put(%mask, %32, %28, %38)
        #   return (%33)
        index = indices_list[0]
        bool_inp = index
        if symbolic_helper._is_bool(bool_inp):
            rank = symbolic_helper._get_tensor_rank(values)
            if rank is not None and rank == 0:
                return opset9.masked_fill(g, self, bool_inp, values)
            mask_rank = symbolic_helper._get_tensor_rank(bool_inp)
            self_rank = symbolic_helper._get_tensor_rank(self)
            if (
                mask_rank is not None
                and self_rank is not None
                and self_rank > mask_rank
            ):
                # Unsqueeze 'bool_inp' to be broadcastable to shape of 'self'.
                bool_inp = symbolic_helper._unsqueeze_helper(
                    g, bool_inp, list(range(mask_rank, self_rank))
                )
            return masked_scatter(g, self, bool_inp, values)
        broadcast_index_shape = g.op("Shape", index)
        index = symbolic_helper._unsqueeze_helper(g, index, [-1])
    sub_data_shape = symbolic_helper._slice_helper(
        g, g.op("Shape", self), axes=[0], starts=[len(indices_list)], ends=[sys.maxsize]
    )
    values_shape = g.op("Concat", broadcast_index_shape, sub_data_shape, axis_i=0)
    # Check if values is a singular value and expand accordingly
    rank = symbolic_helper._get_tensor_rank(values)
    if rank is not None and rank == 0:
        values = opset9.expand(g, values, values_shape, None)
    values = symbolic_helper._reshape_helper(g, values, values_shape)

    self_scalar_type = _type_utils.JitScalarType.from_value(
        self, _type_utils.JitScalarType.UNDEFINED
    )
    if self_scalar_type != _type_utils.JitScalarType.UNDEFINED:
        values_scalar_type = _type_utils.JitScalarType.from_value(
            values, _type_utils.JitScalarType.UNDEFINED
        )
        if self_scalar_type != values_scalar_type:
            values = g.op("Cast", values, to_i=self_scalar_type.onnx_type())
    elif accumulate:
        raise errors.SymbolicValueError("self does not have a valid scalar type.", self)

    if accumulate:
        zeros = g.op(
            "ConstantOfShape",
            g.op("Shape", self),
            value_t=torch.tensor([0], dtype=self_scalar_type.dtype()),
        )
        result = g.op("ScatterND", zeros, index, values)
        result = add(g, self, result)
    else:
        result = g.op("ScatterND", self, index, values)

    return result


@_onnx_symbolic("aten::pixel_shuffle")
@symbolic_helper.parse_args("v", "i")
def pixel_shuffle(g: jit_utils.GraphContext, self, upscale_factor):
    rank = symbolic_helper._get_tensor_rank(self)
    if rank is not None and rank != 4:
        return symbolic_helper._unimplemented("pixel_shuffle", "only support 4d input")
    return g.op("DepthToSpace", self, blocksize_i=upscale_factor, mode_s="CRD")


@_onnx_symbolic(
    "aten::upsample_nearest1d",
    decorate=[symbolic_helper._apply_params("upsample_nearest1d", 3, "nearest")],
)
@_onnx_symbolic(
    "aten::upsample_nearest2d",
    decorate=[symbolic_helper._apply_params("upsample_nearest2d", 4, "nearest")],
)
@_onnx_symbolic(
    "aten::upsample_nearest3d",
    decorate=[symbolic_helper._apply_params("upsample_nearest3d", 5, "nearest")],
)
@_onnx_symbolic(
    "aten::upsample_linear1d",
    decorate=[symbolic_helper._apply_params("upsample_linear1d", 3, "linear")],
)
@_onnx_symbolic(
    "aten::upsample_bilinear2d",
    decorate=[symbolic_helper._apply_params("upsample_bilinear2d", 4, "linear")],
)
@_onnx_symbolic(
    "aten::upsample_trilinear3d",
    decorate=[symbolic_helper._apply_params("upsample_trilinear3d", 5, "linear")],
)
@_onnx_symbolic(
    "aten::upsample_bicubic2d",
    decorate=[symbolic_helper._apply_params("upsample_bicubic2d", 4, "cubic")],
)
def _interpolate(name: str, dim: int, interpolate_mode: str):
    return symbolic_helper._interpolate_helper(name, dim, interpolate_mode)


@_onnx_symbolic("aten::__interpolate")
@symbolic_helper.quantized_args(True, False, False, False, False, False, False)
def __interpolate(
    g: jit_utils.GraphContext,
    input,
    size,
    scale_factor,
    mode,
    align_corners,
    recompute_scale_factor,
    antialias,
):
    return symbolic_helper.__interpolate_helper(
        g, input, size, scale_factor, mode, align_corners, recompute_scale_factor
    )


@_onnx_symbolic("aten::gather")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "i", "v", "v")
def gather(g: jit_utils.GraphContext, self, dim, index, sparse_grad=False):
    if symbolic_helper._maybe_get_const(sparse_grad, "i"):
        return symbolic_helper._unimplemented("gather", "sparse_grad == True")
    return g.op("GatherElements", self, index, axis_i=dim)


@_onnx_symbolic("aten::scatter")
@symbolic_helper.parse_args("v", "i", "v", "v")
def scatter(g: jit_utils.GraphContext, self, dim, index, src):
    src_type = _type_utils.JitScalarType.from_value(src)
    src = symbolic_helper._maybe_get_scalar(src)
    if symbolic_helper._is_value(src):
        return g.op("ScatterElements", self, index, src, axis_i=dim)
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
            "ScatterElements", self, index, opset9.expand_as(g, src, index), axis_i=dim
        )


@_onnx_symbolic("aten::cumsum")
@symbolic_helper.parse_args("v", "i", "none")
def cumsum(g: jit_utils.GraphContext, self, dim, dtype=None):
    dim_tensor = g.op("Constant", value_t=torch.tensor(dim, dtype=torch.int))
    if dtype and dtype.node().kind() != "prim::Constant":
        parsed_dtype = symbolic_helper._get_const(dtype, "i", "dtype")
        cast = g.op(
            "Cast", self, to_i=_type_utils.JitScalarType(parsed_dtype).onnx_type()
        )
    else:
        cast = self
    csum = g.op("CumSum", cast, dim_tensor)
    return csum


@_onnx_symbolic("aten::masked_select")
def masked_select(g: jit_utils.GraphContext, self, mask):
    index = opset9.nonzero(g, opset9.expand_as(g, mask, self))
    return g.op("GatherND", self, index)


@_onnx_symbolic("aten::masked_scatter")
def masked_scatter(g: jit_utils.GraphContext, self, mask, source):
    index = opset9.nonzero(g, opset9.expand_as(g, mask, self))
    # NOTE: source can have more elements than needed.
    # It could also have arbitrary shape.
    # This is not supported by ONNX::ScatterND, so we need to flatten and slice source tensor.
    source = symbolic_helper._reshape_helper(g, source, torch.LongTensor([-1]))
    source = symbolic_helper._slice_helper(
        g,
        source,
        axes=torch.LongTensor([0]),
        starts=torch.LongTensor([0]),
        ends=opset9.size(g, index, torch.LongTensor([0])),
    )
    return g.op("ScatterND", self, index, source)


@_onnx_symbolic("aten::len")
def _len(g: jit_utils.GraphContext, self):
    if (
        symbolic_helper._is_tensor_list(self)
        or self.node().kind() == "onnx::SplitToSequence"
    ):
        return g.op("SequenceLength", self)
    sz_0 = size(g, self, g.op("Constant", value_t=torch.LongTensor([0])))
    return symbolic_helper._squeeze_helper(g, sz_0, [0])


@_onnx_symbolic("aten::__getitem_")
def __getitem_(g: jit_utils.GraphContext, self, i):
    if symbolic_helper._is_tensor_list(self):
        # SequenceAt requires that the input be a List of Tensors
        return g.op("SequenceAt", self, i)
    else:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import (
            __getitem_ as getitem,
        )

        return getitem(g, self, i)


@_onnx_symbolic("aten::_set_item")
def _set_item(g: jit_utils.GraphContext, tensor_list, i, v):
    tensor_list = g.op("SequenceErase", tensor_list, i)
    return g.op("SequenceInsert", tensor_list, v, i)


@_onnx_symbolic("aten::append")
def append(g: jit_utils.GraphContext, self, tensor):
    return g.op("SequenceInsert", self, tensor)


@_onnx_symbolic("aten::add")
def add(g: jit_utils.GraphContext, self, other, alpha=None):
    if symbolic_helper._is_value(self) and symbolic_helper._is_tensor_list(self):
        tensor_list_node = other.node()
        if tensor_list_node.kind() != "prim::ListConstruct":
            return symbolic_helper._unimplemented(
                "add", "does not support adding dynamic tensor list to another"
            )
        tensors = symbolic_helper._unpack_list(other)
        l = self
        for t in tensors:
            l = g.op("SequenceInsert", l, t)
        return l

    return opset9.add(g, self, other, alpha)


@_onnx_symbolic("aten::insert")
def insert(g: jit_utils.GraphContext, self, pos, tensor):
    return g.op("SequenceInsert", self, tensor, pos)


@_onnx_symbolic("aten::pop")
def pop(g: jit_utils.GraphContext, tensor_list, dim):
    return g.op("SequenceErase", tensor_list, dim)


@_onnx_symbolic("aten::Delete")
def Delete(g: jit_utils.GraphContext, tensor_list, dim):
    return g.op("SequenceErase", tensor_list, dim)


@_onnx_symbolic("aten::cat")
@symbolic_helper.quantized_args(True)
def cat(g: jit_utils.GraphContext, tensor_list, dim):
    if symbolic_helper._is_packed_list(tensor_list):
        return opset9.cat(g, tensor_list, dim)
    else:
        dim = symbolic_helper._get_const(dim, "i", "dim")
        return g.op("ConcatFromSequence", tensor_list, axis_i=dim)


@_onnx_symbolic("aten::stack")
def stack(g: jit_utils.GraphContext, tensor_list, dim):
    if symbolic_helper._is_packed_list(tensor_list):
        return opset9.stack(g, tensor_list, dim)
    else:
        dim = symbolic_helper._get_const(dim, "i", "dim")
        return g.op("ConcatFromSequence", tensor_list, axis_i=dim, new_axis_i=1)


@_onnx_symbolic("aten::_unique2")
@symbolic_helper.parse_args("v", "i", "i", "i")
def _unique2(g: jit_utils.GraphContext, self, sorted, return_inverse, return_counts):
    u, _indices, inverse_indices, counts = g.op(
        "Unique", self, sorted_i=sorted, outputs=4
    )
    return u, inverse_indices, counts


@_onnx_symbolic("aten::unique_dim")
@symbolic_helper.parse_args("v", "i", "i", "i", "i")
def unique_dim(
    g: jit_utils.GraphContext, self, dim, sorted, return_inverse, return_counts
):
    u, _indices, inverse_indices, counts = g.op(
        "Unique", self, axis_i=dim, sorted_i=sorted, outputs=4
    )
    return u, inverse_indices, counts


@_onnx_symbolic("aten::topk")
@symbolic_helper.parse_args("v", "v", "i", "i", "i", "none")
def topk(g: jit_utils.GraphContext, self, k, dim, largest, sorted, out=None):
    return symbolic_helper._topk_helper(
        g, self, k, dim, largest=largest, sorted=sorted, out=out
    )


@_onnx_symbolic("aten::sort")
@symbolic_helper.parse_args("v", "i", "i", "none")
def sort(g: jit_utils.GraphContext, self, dim, descending, out=None):
    return symbolic_helper._sort_helper(g, self, dim, descending=descending, out=out)


@_onnx_symbolic("aten::argsort")
@symbolic_helper.parse_args("v", "i", "i", "none")
def argsort(g: jit_utils.GraphContext, self, dim, descending, out=None):
    _, indices = symbolic_helper._sort_helper(
        g, self, dim, descending=descending, out=out
    )
    return indices


@_onnx_symbolic("aten::round")
@symbolic_helper.parse_args("v", "i")
def round(g: jit_utils.GraphContext, self, decimals=0):
    if not symbolic_helper._is_fp(self):
        return self
    if decimals == 0:
        return g.op("Round", self)
    mul = g.op("Mul", self, g.op("Constant", value_t=torch.tensor(pow(10, decimals))))
    round = g.op("Round", mul)
    return g.op(
        "Mul", round, g.op("Constant", value_t=torch.tensor(pow(10, -1 * decimals)))
    )


@_onnx_symbolic("aten::remainder")
def remainder(g: jit_utils.GraphContext, input, other):
    if symbolic_helper._is_fp(input) or symbolic_helper._is_fp(other):
        return opset9.remainder(g, input, other)
    return g.op("Mod", input, other, fmod_i=0)


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
    else:
        return opset9.split(g, self, split_size_or_sizes, dim, _outputs)


@_onnx_symbolic("aten::split_with_sizes")
@symbolic_helper.parse_args("v", "v", "i", "i")
def split_with_sizes(g: jit_utils.GraphContext, self, split_sizes, dim, _outputs=None):
    return split(g, self, split_sizes, dim, _outputs)


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
    else:
        return opset9.unbind(g, self, dim, _outputs)


def _prepare_onnx_paddings(g: jit_utils.GraphContext, input, pad):
    """Generate paddings in ONNX order based on pad in pytorch.

    Args:
        input: the input tensor.
        pad: the paddings in pytorch.
            The order is dim_n_begin, dim_n_end, dim_n-1_begin, dim_n-1_end, ..., dim_m_begin, dim_m_end,
            where m is in range [0, n].
    """
    if (
        not symbolic_helper._is_packed_list(pad)
        and symbolic_helper._is_list(pad)
        and symbolic_helper._is_scalar_list(pad)
    ):
        pad = g.op("ConcatFromSequence", pad, axis_i=0, new_axis_i=1)
    # The desired order of paddings is
    # dim_0_begin, dim_1_begin, ... , dim_0_end, ..., dim_n_end.
    # n is the dimension of input.
    # Assume zero-dimensions in the beginning, pad the "pad" sequence with zeros in the beginning
    pad_len = opset9.size(g, pad, g.op("Constant", value_t=torch.tensor([0])))
    # Set extension = [0] * (dim * 2 - len(pad))
    rank = symbolic_helper._get_tensor_rank(input)
    if rank is None:
        rank = g.op("Size", g.op("Shape", input))
    else:
        rank = g.op("Constant", value_t=torch.tensor(rank, dtype=torch.int64))
    extension = g.op(
        "Sub",
        g.op("Mul", rank, g.op("Constant", value_t=torch.tensor(2, dtype=torch.int64))),
        pad_len,
    )
    # Concat pad with extension: paddings = [dim_n_begin, dim_n_end, dim_n-1_begin, dim_n-1_end, 0, 0, ... ]
    # Currently ONNX only supports int64 type for Pad
    pad = g.op("Cast", pad, to_i=_C_onnx.TensorProtoDataType.INT64)
    paddings = g.op(
        "Concat",
        pad,
        g.op(
            "ConstantOfShape", extension, value_t=torch.tensor([0], dtype=torch.int64)
        ),
        axis_i=0,
    )
    # Reshape and reverse order and collate first beginnings and then ends
    # paddings = [[..., 0, dim_n-1_begin, dim_n_begin],
    #               [..., 0, dim_n-1_end, dim_n_end]]
    # Reshape back to 1-D paddings = [..., 0, dim_n - 1_begin, dim_n_begin, ..., 0, dim_n - 1_end, dim_n_end]
    paddings = symbolic_helper._reshape_helper(
        g, paddings, g.op("Constant", value_t=torch.tensor([-1, 2]))
    )
    paddings = g.op("Transpose", opset10.flip(g, paddings, [0]), perm_i=[1, 0])
    paddings = symbolic_helper._reshape_helper(
        g, paddings, g.op("Constant", value_t=torch.tensor([-1]))
    )
    padding_c = g.op("Cast", paddings, to_i=_C_onnx.TensorProtoDataType.INT64)
    return padding_c


@_onnx_symbolic("aten::constant_pad_nd")
def constant_pad_nd(g: jit_utils.GraphContext, input, padding, value=None):
    mode = "constant"
    value = symbolic_helper._maybe_get_scalar(value)
    value = symbolic_helper._if_scalar_type_as(value, input)
    pad = _prepare_onnx_paddings(g, input, padding)
    return g.op("Pad", input, pad, value, mode_s=mode)


@_onnx_symbolic("aten::reflection_pad1d")
@_onnx_symbolic("aten::reflection_pad2d")
@_onnx_symbolic("aten::reflection_pad3d")
def reflection_pad(g: jit_utils.GraphContext, input, padding):
    mode = "reflect"
    paddings = _prepare_onnx_paddings(g, input, padding)
    return g.op("Pad", input, paddings, mode_s=mode)


@_onnx_symbolic("aten::replication_pad1d")
@_onnx_symbolic("aten::replication_pad2d")
@_onnx_symbolic("aten::replication_pad3d")
def replication_pad(g: jit_utils.GraphContext, input, padding):
    mode = "edge"
    paddings = _prepare_onnx_paddings(g, input, padding)
    return g.op("Pad", input, paddings, mode_s=mode)


@_onnx_symbolic("aten::pad")
def pad(
    g: jit_utils.GraphContext,
    input: _C.Value,
    pad: _C.Value,
    mode: _C.Value,
    value: _C.Value,
):
    mode = symbolic_helper._parse_arg(mode, "s")
    if mode == "replicate":
        return replication_pad(g, input, pad)
    elif mode == "reflect":
        return reflection_pad(g, input, pad)
    elif mode == "constant":
        return constant_pad_nd(g, input, pad, value)
    elif mode == "circular":
        return opset9._pad_circular(g, input, pad)
    else:
        raise errors.SymbolicValueError(f"Unrecognized padding mode {mode}", input)


@_onnx_symbolic("aten::linalg_det")
def linalg_det(g: jit_utils.GraphContext, self):
    return g.op("Det", self)


@_onnx_symbolic("aten::logdet")
def logdet(g: jit_utils.GraphContext, input):
    return opset9.log(g, linalg_det(g, input))


@_onnx_symbolic("aten::arange")
def arange(g: jit_utils.GraphContext, *args):
    def _get_arange_dtype(dtype):
        dtype = symbolic_helper._maybe_get_const(dtype, "i")
        return dtype

    if len(args) == 2 and all(isinstance(val, int) for val in args):
        # aten::arange(Scalar start, Scalar end)
        dtype = torch.int64
        # Start index.
        start = g.op(
            "Constant",
            value_t=torch.tensor(args[0], dtype=dtype),
        )
        # End (exclusive) index.
        end = g.op(
            "Constant",
            value_t=torch.tensor(args[1], dtype=dtype),
        )
        # Step size from start to end indexes.
        delta_default = g.op(
            "Constant",
            value_t=torch.tensor(1, dtype=dtype),
        )
        return g.op("Range", start, end, delta_default)
    elif len(args) == 2 or len(args) == 5:
        if len(args) == 2:
            # aten::arange(Scalar end, Tensor out)
            dtype = None
        else:
            # aten::arange(Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
            dtype = _get_arange_dtype(args[1])
        type_, end, start, step = symbolic_helper._arange_cast_helper(
            g, end=args[0], dtype=dtype
        )
        start_default = g.op(
            "Constant",
            value_t=torch.tensor(0, dtype=type_.dtype()),
        )
        delta_default = g.op(
            "Constant",
            value_t=torch.tensor(1, dtype=type_.dtype()),
        )
        # pyrefly: ignore [bad-argument-type]
        return g.op("Range", start_default, end, delta_default)
    elif len(args) == 4 or len(args) == 7:
        if len(args) == 4:
            # aten::arange(Scalar start, Scalar end, Scalar step, Tensor out)
            dtype = None
        else:
            # aten::arange(Scalar start, Scalar end, Scalar step, ScalarType dtype, Layout, Device, bool pin_memory)
            dtype = _get_arange_dtype(args[3])
        _, end, start, step = symbolic_helper._arange_cast_helper(
            g, start=args[0], end=args[1], step=args[2], dtype=dtype
        )
        # pyrefly: ignore [bad-argument-type]
        return g.op("Range", start, end, step)
    elif len(args) == 6:
        # aten::arange(Scalar start, Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
        dtype = _get_arange_dtype(args[2])
        type_, end, start, step = symbolic_helper._arange_cast_helper(
            g, start=args[0], end=args[1], dtype=dtype
        )
        delta_default = g.op(
            "Constant",
            value_t=torch.tensor(1, dtype=type_.dtype()),
        )
        # pyrefly: ignore [bad-argument-type]
        return g.op("Range", start, end, delta_default)
    else:
        return symbolic_helper._unimplemented(
            "aten::arange", f"with {len(args)} arguments"
        )


@_onnx_symbolic("aten::_dim_arange")
@symbolic_helper.parse_args("v", "i")
def _dim_arange(g: jit_utils.GraphContext, like, dim):
    like_shape = g.op("Shape", like)
    stop = g.op(
        "Gather", like_shape, g.op("Constant", value_t=torch.tensor(dim)), axis_i=0
    )
    return arange(g, stop, 4, None, None, None)


@_onnx_symbolic("aten::size")
@symbolic_helper.quantized_args(True, quantize_output=False)
def size(g: jit_utils.GraphContext, self, dim=None):
    if dim is None:
        return g.op("Shape", self)
    return symbolic_helper._size_helper(g, self, dim)


@_onnx_symbolic("aten::squeeze")
def squeeze(g: jit_utils.GraphContext, self, dim=None):
    if dim is None:
        return g.op("Squeeze", self)

    # dim as a tensor
    if not symbolic_helper._is_constant(dim):
        return symbolic_helper._squeeze_helper(g, self, [dim])

    dim = symbolic_helper._get_const(dim, "i", "dim")

    input_rank = symbolic_helper._get_tensor_rank(self)
    adjusted_dim = dim
    if input_rank is not None and dim < 0:
        adjusted_dim += input_rank
    dim_size = symbolic_helper._get_tensor_dim_size(self, adjusted_dim)
    if (dim < 0 and input_rank is None) or dim_size is None:
        # If onnx shape inference is not on, export always as dynamic.
        # Because we cannot tell if observed static shape is also static at runtime.
        # create "cond" node (condition is shape[i]==1)
        dim_constant = g.op("Constant", value_t=torch.tensor([dim]))
        size = symbolic_helper._size_helper(g, self, dim_constant)
        const_one = g.op("Constant", value_t=torch.ones(1, dtype=torch.int64))
        cond = g.op("Equal", size, const_one)
        # create the "If" node and add the "then" and "else" blocks to it.
        if_op, (if_context, else_context), _ = jit_utils.add_op_with_blocks(
            g, "If", cond, n_blocks=2
        )
        squeeze_ = symbolic_helper._squeeze_helper(if_context, self, [dim])
        utils._add_output_to_block(if_context.block, squeeze_)
        identity_ = else_context.op("Identity", self)
        utils._add_output_to_block(else_context.block, identity_)
        return if_op

    # For static input shape
    dim = adjusted_dim
    if dim_size > 1:
        warnings.warn(
            "This model contains a squeeze operation on dimension "
            + str(dim)
            + ". The size of "
            + "this dimension in the given input is "
            + str(dim_size)
            + ". The model will "
            + "be exported without the squeeze node. If the model is intended to be used with dynamic "
            + "input shapes, please export with dynamic_axes argument.",
            stacklevel=2,
        )
        return self
    return symbolic_helper._squeeze_helper(g, self, [dim])


@_onnx_symbolic("aten::unsqueeze")
def unsqueeze(g: jit_utils.GraphContext, self, dim):
    if symbolic_helper._is_constant(dim):
        dim = symbolic_helper._get_const(dim, "i", "dim")

    return symbolic_helper._unsqueeze_helper(g, self, [dim])


@_onnx_symbolic("aten::mm")
def mm(g: jit_utils.GraphContext, self, other):
    return g.op("Gemm", self, other, beta_f=0.0, alpha_f=1.0)


@_onnx_symbolic("aten::index")
def index(g: jit_utils.GraphContext, self, index):
