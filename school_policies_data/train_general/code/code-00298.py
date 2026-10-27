        input,
        scale.item(),
        zero_point.item(),  # type: ignore[arg-type]
        quant_min,
        quant_max,
        dtype,
        out_dtype=out_dtype,
    )


@impl(quantized_decomposed_lib, "dequantize_per_tensor.tensor", "Meta")
def dequantize_per_tensor_tensor_meta(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if out_dtype is None:
        out_dtype = torch.float32
    if zero_point.numel() != 1:
        raise AssertionError(
            f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
        )
    if scale.numel() != 1:
        raise AssertionError(
            f"Expecting scale tensor to be one element, but received : {scale.numel()}"
        )
    if input.dtype != dtype:
        raise AssertionError(
            f"Expecting input to have dtype: {dtype}, but got {input.dtype}"
        )
    if dtype in _DTYPE_TO_QVALUE_BOUNDS:
        return torch.empty_like(input, dtype=out_dtype)
    else:
        raise ValueError(f"Unsupported dtype in dequantize_per_tensor: {dtype}")


# TODO: remove other variants and keep this one
quantized_decomposed_lib.define(
    "dequantize_per_tensor.tensor2(Tensor input, Tensor scale, Tensor zero_point, "
    "Tensor quant_min, Tensor quant_max, ScalarType dtype, *, ScalarType? out_dtype=None) -> Tensor"
)


@impl(
    quantized_decomposed_lib,
    "dequantize_per_tensor.tensor2",
    "CompositeExplicitAutograd",
)
def dequantize_per_tensor_tensor2(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: torch.Tensor,
    quant_max: torch.Tensor,
    dtype: torch.dtype,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Affine dequantization for the Tensor using the same quantization parameters to map
    from quantized values to floating point values
    Same as `dequantize_per_tensor` but scale and zero_point are Scalar Tensor instead of
    scalar values
    """
    if zero_point.numel() != 1:
        raise AssertionError(
            f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
        )
    if scale.numel() != 1:
        raise AssertionError(
            f"Expecting scale tensor to be one element, but received : {scale.numel()}"
        )
    return dequantize_per_tensor(
        input,
        scale.item(),
        zero_point.item(),  # type: ignore[arg-type]
        quant_min.item(),  # type: ignore[arg-type]
        quant_max.item(),  # type: ignore[arg-type]
        dtype,
        out_dtype=out_dtype,
    )


@impl(quantized_decomposed_lib, "dequantize_per_tensor.tensor2", "Meta")
def dequantize_per_tensor_tensor2_meta(
    input,
    scale,
    zero_point,
    quant_min,
    quant_max,
    dtype,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return dequantize_per_tensor_tensor_meta(
        input, scale, zero_point, quant_min, quant_max, dtype, out_dtype=out_dtype
    )


quantized_decomposed_lib.define(
    "choose_qparams.tensor(Tensor input, int quant_min, int quant_max, "
    "float eps, ScalarType dtype) -> (Tensor, Tensor)"
)


@impl(quantized_decomposed_lib, "choose_qparams.tensor", "CompositeExplicitAutograd")
def choose_qparams_tensor(
    input: torch.Tensor, qmin: int, qmax: int, eps: float, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Given an input Tensor, derive the per tensor affine quantization parameter
    (scale and zero_point) for target quantized Tensor from the Tensor

    Args:
       input (torch.Tensor): floating point input Tensor
       quant_min (int): minimum quantized value for target quantized Tensor
       quant_max (int): maximum quantized value for target quantized Tensor
       dtype (torch.dtype): dtype for target quantized Tensor

    Returns:
       scale (float): quantization parameter for the target quantized Tensor
       zero_point (int): quantization parameter for the target quantized Tensor
    """
    if input.dtype not in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ]:
        raise AssertionError(
            f"Expecting input to have dtype torch.float32/16/b16, but got dtype: {input.dtype}"
        )
    if dtype not in _DTYPE_TO_QVALUE_BOUNDS:
        raise AssertionError(
            f"Expecting target dtype to be one of {_DTYPE_TO_QVALUE_BOUNDS.keys()}, but got: {dtype}"
        )
    validate_qmin_qmax(qmin, qmax)

    min_val, max_val = torch.aminmax(input)

    return determine_qparams(
        min_val,
        max_val,
        qmin,
        qmax,
        dtype,
        torch.Tensor([eps]),
        has_customized_qrange=False,
    )


quantized_decomposed_lib.define(
    "choose_qparams_symmetric.tensor(Tensor input, int quant_min, int quant_max, "
    "float eps, ScalarType dtype) -> (Tensor, Tensor)"
)


@impl(
    quantized_decomposed_lib,
    "choose_qparams_symmetric.tensor",
    "CompositeExplicitAutograd",
)
def choose_qparams_symmetric_tensor(
    input: torch.Tensor, qmin: int, qmax: int, eps: float, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Given an input Tensor, derive the per tensor affine quantization parameter
    (scale and zero_point) for target quantized Tensor from the Tensor

    Args:
       input (torch.Tensor): floating point input Tensor
       quant_min (int): minimum quantized value for target quantized Tensor
       quant_max (int): maximum quantized value for target quantized Tensor
       dtype (torch.dtype): dtype for target quantized Tensor

    Returns:
       scale (float): quantization parameter for the target quantized Tensor
       zero_point (int): quantization parameter for the target quantized Tensor
    """
    if input.dtype not in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ]:
        raise AssertionError(
            f"Expecting input to have dtype torch.float32/16/b16, but got dtype: {input.dtype}"
        )
    if dtype not in _DTYPE_TO_QVALUE_BOUNDS:
        raise AssertionError(
            f"Expecting target dtype to be one of {_DTYPE_TO_QVALUE_BOUNDS.keys()}, but got: {dtype}"
        )
    validate_qmin_qmax(qmin, qmax)

    min_val, max_val = torch.aminmax(input)
    return determine_qparams(
        min_val,
        max_val,
        qmin,
        qmax,
        dtype,
        torch.Tensor([eps]),
        has_customized_qrange=False,
        qscheme=torch.per_tensor_symmetric,
    )


@impl(quantized_decomposed_lib, "choose_qparams.tensor", "Meta")
def choose_qparams_tensor_meta(
    input: torch.Tensor, quant_min: int, quant_max: int, eps: float, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    if input.dtype not in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ]:
        raise AssertionError(
            f"Expecting input to have dtype torch.float32/16/b16, but got dtype: {input.dtype}"
        )
    if quant_min >= quant_max:
        raise AssertionError(
            f"Expecting quant_min to be smaller than quant_max but received min: {quant_min} max: {quant_max}"
        )
    return torch.empty(1, dtype=torch.double, device=input.device), torch.empty(
        1, dtype=torch.int64, device=input.device
    )


@impl(quantized_decomposed_lib, "choose_qparams_symmetric.tensor", "Meta")
def choose_qparams_symmetric_tensor_meta(
    input: torch.Tensor, quant_min: int, quant_max: int, eps: float, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty(1, dtype=torch.double, device=input.device), torch.empty(
        1, dtype=torch.int64, device=input.device
    )


# Helper function used to implement per-channel quantization against any axis
def _permute_to_axis_zero(x, axis):
    new_axis_list = list(range(x.dim()))
    new_axis_list[axis] = 0
    new_axis_list[0] = axis
    y = x.permute(tuple(new_axis_list))
    return y, new_axis_list


quantized_decomposed_lib.define(
    "quantize_per_channel(Tensor input, Tensor scales, Tensor zero_points, int axis, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor"
)


@impl(quantized_decomposed_lib, "quantize_per_channel", "CompositeExplicitAutograd")
def quantize_per_channel(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    axis: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Affine per channel quantization for the Tensor using the same quantization
    parameters for each channel/axis to map from floating point to quantized values

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scales (torch.Tensor): a list of scale quantization parameter for
       affine quantization, one per channel
       zero_point (torch.Tensor): a list of zero_point quantization parameter for
       affine quantization, one per channel
       quant_min (int): minimum quantized value for output Tensor
       quant_max (int): maximum quantized value for output Tensor
       dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. torch.uint8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """
    if input.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(torch.float32)
    if input.dtype != torch.float32:
        raise AssertionError(
            f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
        )
    if axis >= input.dim():
        raise AssertionError(f"Expecting axis to be < {input.dim()}")
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    input, permute_axis_list = _permute_to_axis_zero(input, axis)

    new_shape = [1] * input.dim()
    new_shape[0] = scales.shape[0]
    scales = scales.view(new_shape)
    zero_points = zero_points.view(new_shape)

    res = torch.clamp(
        torch.round(input * (1.0 / scales)) + zero_points, quant_min, quant_max
    )
    out = res.permute(tuple(permute_axis_list))
    return out.to(dtype)


@impl(quantized_decomposed_lib, "quantize_per_channel", "Meta")
def quantize_per_channel_meta(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    axis: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if input.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(torch.float32)
    if input.dtype != torch.float32:
        raise AssertionError(
            f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
        )
    if axis >= input.dim():
        raise AssertionError(f"Expecting axis to be < {input.dim()}")
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    return torch.empty_like(input, dtype=dtype)


# Note: quant_min/quant_max/dtype are not used in the operator, but for now it's kept in
# the signature as metadata for the input Tensor, this might be useful for pattern
# matching in the future
# We will revisit this later if we found there are no use cases for it
quantized_decomposed_lib.define(
    "dequantize_per_channel(Tensor input, Tensor scales, Tensor? zero_points, int axis, "
    "int quant_min, int quant_max, ScalarType dtype, *, ScalarType? out_dtype=None) -> Tensor"
)


@impl(quantized_decomposed_lib, "dequantize_per_channel", "CompositeExplicitAutograd")
def dequantize_per_channel(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor | None,
    axis: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Affine per channel dequantization for the Tensor using the same quantization
    parameters for each channel/axis to map from quantized values to floating point values

    Args:
       input (torch.Tensor): Tensor with dtype matching `dtype` argument,
       e.g. (`torch.uint8`), it is a per channel quantized Tensor if combined with
       quantization parameter in the argument of this function (scales/zero_points/axis)

       scales (torch.Tensor): a list of scale quantization parameter for
       affine quantization, one per channel

       zero_points (torch.Tensor): a list of zero_point quantization parameter for
       affine quantization, one per channel

       quant_min (int): minimum quantized value for output Tensor (not used in computation,
       reserved for pattern matching)

       quant_max (int): maximum quantized value for output Tensor (not used in computation,
       reserved for pattern matching)

       dtype (torch.dtype): requested dtype for output Tensor (not used in computation,
       reserved for pattern matching)

       out_dtype (torch.dtype?): optional dtype for output Tensor

    Returns:
       dequantized float32 Tensor
    """
    if input.dtype != dtype:
        raise AssertionError(
            f"Expecting input to have dtype: {dtype}, but got dtype: {input.dtype}"
        )
    if out_dtype is None:
        out_dtype = torch.float32
    if axis >= input.dim():
        raise AssertionError(f"Expecting axis to be < {input.dim()}")
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    input, permute_axis_list = _permute_to_axis_zero(input, axis)

    new_shape = [1] * input.dim()
    new_shape[0] = scales.shape[0]
    scales = scales.view(new_shape)
    if zero_points is not None:
        res = (input - zero_points.view(new_shape)) * scales
    else:
        res = input * scales

    res = res.to(out_dtype)

    out = res.permute(tuple(permute_axis_list))
    return out


@impl(quantized_decomposed_lib, "dequantize_per_channel", "Meta")
def dequantize_per_channel_meta(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor | None,
    axis: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if input.dtype != dtype:
        raise AssertionError(
            f"Expecting input to have dtype {dtype}, but got dtype: {input.dtype}"
        )
    if out_dtype is None:
        out_dtype = torch.float32
    if axis >= input.dim():
        raise AssertionError(f"Expecting axis to be < {input.dim()}")
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    return torch.empty_like(input, dtype=out_dtype)


quantized_decomposed_lib.define(
    "choose_qparams_per_token(Tensor input, ScalarType dtype) -> (Tensor, Tensor)"
)


@impl(
    quantized_decomposed_lib,
    "choose_qparams_per_token",
    "CompositeExplicitAutograd",
)
def choose_qparams_per_token(
    input: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose quantization parameters for per token quantization. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (torch.Tensor): original float32/float16 Tensor
       dtype (torch.dtype): dtype (e.g. torch.uint8) for input Tensor

    Returns:
        scales and zero_points, both float32 Tensors
    """

    scales = input.abs().amax(dim=-1, keepdim=True)
    if scales.dtype == torch.float16:
        scales = (
            scales.float()
        )  # want float scales to avoid overflows for fp16, (bf16 has wide enough range)
    if dtype == torch.int8:
        n_bits = 8
        quant_max = 2 ** (n_bits - 1) - 1
    else:
        raise Exception(  # noqa: TRY002
            f"unsupported dtype in choose_qparams_per_token: {dtype}"
        )

    scales = scales.clamp(min=1e-5).div(quant_max)
    zero_points = torch.zeros_like(scales)
    return scales, zero_points


@impl(
    quantized_decomposed_lib,
    "choose_qparams_per_token",
    "Meta",
)
def choose_qparams_per_token_meta(
    input: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    size = list(input.shape[:-1]) + [1]
    return torch.empty(size, dtype=torch.double, device=input.device), torch.empty(
        size, dtype=torch.int64, device=input.device
    )


quantized_decomposed_lib.define(
    "_choose_qparams_per_token_asymmetric_impl(Tensor input, ScalarType dtype) -> (Tensor, Tensor)"
)


@impl(
    quantized_decomposed_lib,
    "_choose_qparams_per_token_asymmetric_impl",
    "CompositeImplicitAutograd",
)
def _choose_qparams_per_token_asymmetric_impl(
    input: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose quantization parameters for per token quantization. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (torch.Tensor): original float32/float16 Tensor
       dtype (torch.dtype): dtype (e.g. torch.uint8) for input Tensor

    Returns:
        scales and zero_points, both float32 Tensors
    """
    # Based on https://github.com/google/XNNPACK/blob/df156f0cf3db5a4576cc711123eeb54915f82ffc/src/xnnpack/quantization.h#L18
    qmin, qmax = -128, 127
    min_val = torch.amin(input, dim=-1, keepdim=True)
    max_val = torch.amax(input, dim=-1, keepdim=True)
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    eps = torch.finfo(torch.float32).eps  # use xnnpack eps?

    # scale
    scale = (max_val_pos - min_val_neg) / float(qmax - qmin)
    scale = scale.clamp(min=eps)

    # zero point
    descaled_min = min_val_neg / scale
    descaled_max = max_val_pos / scale
    zero_point_from_min_error = qmin + descaled_min
    zero_point_from_max_error = qmax + descaled_max
    zero_point = torch.where(
        zero_point_from_min_error + zero_point_from_max_error > 0,
        qmin - descaled_min,
        qmax - descaled_max,
    )
    zero_point = torch.clamp(zero_point, qmin, qmax).round()

    return scale.to(torch.float64), zero_point.to(torch.int64)


quantized_decomposed_lib.define(
    "choose_qparams_per_token_asymmetric(Tensor input, ScalarType dtype) -> (Tensor, Tensor)"
)


@impl(
    quantized_decomposed_lib,
    "choose_qparams_per_token_asymmetric",
    "CompositeExplicitAutograd",
)
def choose_qparams_per_token_asymmetric(
    input: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _choose_qparams_per_token_asymmetric_impl(input, dtype)


@impl(
    quantized_decomposed_lib,
    "choose_qparams_per_token_asymmetric",
    "Meta",
)
def choose_qparams_per_token_asymmetric_meta(
    input: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    size = list(input.shape[:-1]) + [1]
    return torch.empty(size, dtype=torch.double, device=input.device), torch.empty(
        size, dtype=torch.int64, device=input.device
    )


def _per_token_quant_qparam_dim_check(input, scales, zero_points):
    num_tokens = math.prod(list(input.size())[:-1])
    if num_tokens != scales.numel():
        raise AssertionError(f"num_tokens: {num_tokens} scales: {scales.size()}")
    if num_tokens != zero_points.numel():
        raise AssertionError(
            f"num_tokens: {num_tokens} zero_points: {zero_points.size()}"
        )


quantized_decomposed_lib.define(
    "quantize_per_token(Tensor input, Tensor scales, Tensor zero_points, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor"
)


@impl(quantized_decomposed_lib, "quantize_per_token", "CompositeExplicitAutograd")
def quantize_per_token(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
):
    """Per token quantization for the Tensor using the quantization parameters to map
    from floating point to quantized values. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scales (float32 torch.Tensor): quantization parameter for per token affine quantization
       zero_points (int32 torch.Tensor): quantization parameter for per token affine quantization
       quant_min (int): minimum quantized value for output Tensor
       quant_max (int): maximum quantized value for output Tensor
       dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. torch.uint8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    _per_token_quant_qparam_dim_check(input, scales, zero_points)
    input = (
        input.mul(1.0 / scales)
        .add(zero_points)
        .round()
        .clamp(quant_min, quant_max)
        .to(dtype)
    )
    return input


@impl(quantized_decomposed_lib, "quantize_per_token", "Meta")
def quantize_per_token_meta(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
):
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    return torch.empty_like(input, dtype=dtype)


quantized_decomposed_lib.define(
    "dequantize_per_token(Tensor input, Tensor scales, Tensor zero_points, "
    "int quant_min, int quant_max, ScalarType dtype, ScalarType output_dtype) -> Tensor"
)


@impl(quantized_decomposed_lib, "dequantize_per_token", "CompositeExplicitAutograd")
def dequantize_per_token(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    output_dtype: torch.dtype = torch.float32,
):
    """Per token dequantization for the Tensor using the quantization parameters to map
    from floating point to quantized values. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (torch.Tensor): quantized Tensor (uint8, int8 etc.)
       scales (float64 torch.Tensor): quantization parameter for per token affine quantization
       zero_points (int64 torch.Tensor): quantization parameter for per token affine quantization
       quant_min (int): minimum quantized value for input Tensor
       quant_max (int): maximum quantized value for input Tensor
       dtype (torch.dtype): dtype (e.g. torch.uint8) for input Tensor
       output_dtype (torch.dtype): dtype (e.g. torch.float32) for output Tensor

    Returns:
       dequantized Tensor with dtype `output_dtype`
    """
    input = input - zero_points
    input = input * scales
    # Since scales are of float64 type, we need to cast it to output dtype requested
    return input.to(output_dtype)


@impl(quantized_decomposed_lib, "dequantize_per_token", "Meta")
def dequantize_per_token_meta(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    output_dtype: torch.dtype = torch.float32,
):
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    # TODO: support fp16
    return torch.empty_like(input, dtype=output_dtype)


quantized_decomposed_lib.define(
    "quantize_per_channel_group(Tensor input, Tensor scales, Tensor zero_points, int quant_min, "
    "int quant_max, ScalarType dtype, int group_size) -> Tensor"
)


# TODO: dtype is ignored for now
@impl(
    quantized_decomposed_lib, "quantize_per_channel_group", "CompositeExplicitAutograd"
)
def quantize_per_channel_group(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    group_size=128,
):
    if group_size <= 1:
        raise AssertionError("group_size must be > 1")
    # needed for GPTQ single column quantize
    if group_size > input.shape[-1] and scales.shape[-1] == 1:
        group_size = input.shape[-1]

    if input.shape[-1] % group_size != 0:
        raise AssertionError("input.shape[-1] must be divisible by group_size")
    if input.dim() != 2:
        raise AssertionError("input must be 2-dimensional")

    # TODO: check for dtype, currently we can't express torch.int4 so it's omitted
    to_quant = input.reshape(-1, group_size)
    if torch.isnan(to_quant).sum() != 0:
        raise AssertionError("to_quant must not contain NaNs")

    scales = scales.reshape(-1, 1)
    zero_points = zero_points.reshape(-1, 1)

    input_int8 = (
        to_quant.mul(1.0 / scales)
        .add(zero_points)
        .round()
        .clamp_(quant_min, quant_max)
        .to(dtype)
        .reshape_as(input)
    )

    return input_int8


@impl(quantized_decomposed_lib, "quantize_per_channel_group", "Meta")
def quantize_per_channel_group_meta(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    group_size=128,
):
    """Groupwise quantization within each channel for a 2-d Tensor using the quantization parameters
    to map from floating point to quantized values. This means for each row of a 2-d Tensor
    (M, N), we calculate scales/zero_points for each `group_size` elements
    and quantize every `group_size` elements with the same quantization parameter.
    The dimension for scales/zero_points will be (M * ceil(N, group_size),)

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scales (float32 torch.Tensor): quantization parameter for per channel group affine quantization
       zero_points (int32 torch.Tensor): quantization parameter for per channel group affine quantization
       quant_min (int): minimum quantized value for output Tensor
       quant_max (int): maximum quantized value for output Tensor
       dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. torch.uint8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """
    if group_size <= 1:
        raise AssertionError("group_size must be > 1")
    # needed for GPTQ single column quantize
    if group_size > input.shape[-1] and scales.shape[-1] == 1:
        group_size = input.shape[-1]

    if input.shape[-1] % group_size != 0:
        raise AssertionError("input.shape[-1] must be divisible by group_size")
    if input.dim() != 2:
        raise AssertionError("input must be 2-dimensional")
    return torch.empty_like(input, dtype=dtype)


quantized_decomposed_lib.define(
    "dequantize_per_channel_group(Tensor input, Tensor scales, Tensor? zero_points, int quant_min, "
    "int quant_max, ScalarType dtype, int group_size, ScalarType output_dtype) -> Tensor"
)


@impl(
    quantized_decomposed_lib,
    "dequantize_per_channel_group",
    "CompositeExplicitAutograd",
)
def dequantize_per_channel_group(
    w_int8: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor | None,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    group_size: int = 128,
    output_dtype: torch.dtype = torch.float32,
):
    """Groupwise dequantization within each channel for a 2-d Tensor using the quantization parameters
    to map from floating point to quantized values. This means for each row of a 2-d Tensor
    (M, N), we calculate scales/zero_points for each `group_size` elements
    and quantize every `group_size` elements with the same quantization parameter.
    The dimension for scales/zero_points will be (M * ceil(N, group_size),)

    Args:
       input (torch.Tensor): quantized Tensor (uint8/int8 etc.)
       scales (float32 torch.Tensor): quantization parameter for per channel group affine quantization
       zero_points (int32 torch.Tensor): quantization parameter for per channel group affine quantization
       quant_min (int): minimum quantized value for input Tensor
       quant_max (int): maximum quantized value for input Tensor
       dtype (torch.dtype): dtype (e.g. torch.uint8) for input Tensor
       output_dtype (torch.dtype): dtype (e.g. torch.float32) for output Tensor

    Returns:
       dequantized Tensor with dtype `output_dtype`
    """

    if group_size <= 1:
        raise AssertionError("group_size must be > 1")
    # needed for GPTQ single column dequantize
    if group_size > w_int8.shape[-1] and scales.shape[-1] == 1:
        group_size = w_int8.shape[-1]
    if w_int8.shape[-1] % group_size != 0:
        raise AssertionError("w_int8.shape[-1] must be divisible by group_size")
    if w_int8.dim() != 2:
        raise AssertionError("w_int8 must be 2-dimensional")

    w_int8_grouped = w_int8.reshape(-1, group_size)
    scales = scales.reshape(-1, 1)
    if zero_points is not None:
        zp = zero_points.reshape(-1, 1)
    else:
        zp = torch.zeros([], dtype=torch.int32, device=scales.device)
    w_dq = w_int8_grouped.sub(zp).mul(scales).reshape_as(w_int8).to(output_dtype)
    return w_dq


quantized_decomposed_lib.define(
    "fake_quant_per_channel(Tensor input, Tensor scales, Tensor zero_points, int axis, "
    "int quant_min, int quant_max) -> Tensor"
)


class FakeQuantPerChannel(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, input, scales, zero_points, axis, quant_min, quant_max):
        if scales.dtype != torch.float32:
            scales = scales.to(torch.float32)
        if zero_points.dtype != torch.int32:
            zero_points = zero_points.to(torch.int32)
        if input.dtype != torch.float32:
            raise AssertionError(
                f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
            )
        if axis >= input.dim():
            raise AssertionError(f"Expecting axis to be < {input.dim()}")
        broadcast_dims = list(range(axis)) + list(range(axis + 1, input.ndim))
        unsqueeze_scales = _unsqueeze_multiple(scales, broadcast_dims)
        unsqueeze_zero_points = _unsqueeze_multiple(zero_points, broadcast_dims)
        temp = torch.round(input * (1.0 / unsqueeze_scales)) + unsqueeze_zero_points
        out = (
            torch.clamp(temp, quant_min, quant_max) - unsqueeze_zero_points
        ) * unsqueeze_scales
        mask = torch.logical_and((temp >= quant_min), (temp <= quant_max))

        ctx.save_for_backward(mask)
        return out

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, gy):
        (mask,) = ctx.saved_tensors
        return gy * mask, None, None, None, None, None


@impl(quantized_decomposed_lib, "fake_quant_per_channel", "Autograd")
def fake_quant_per_channel(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    axis: int,
    quant_min: int,
    quant_max: int,
) -> torch.Tensor:
    return FakeQuantPerChannel.apply(
        input, scales, zero_points, axis, quant_min, quant_max
    )


@impl(quantized_decomposed_lib, "fake_quant_per_channel", "Meta")
def fake_quant_per_channel_meta(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    axis: int,
    quant_min: int,
    quant_max: int,
) -> torch.Tensor:
    return torch.empty_like(input)


quantized_decomposed_lib.define(
    "convert_element_type.no_fuse(Tensor input, ScalarType dtype) -> Tensor"
)


@impl(
    quantized_decomposed_lib,
    "convert_element_type.no_fuse",
    "CompositeExplicitAutograd",
)
def convert_element_type(input: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return torch.ops.prims.convert_element_type.default(input, dtype)


@impl(quantized_decomposed_lib, "convert_element_type.no_fuse", "Meta")
def convert_element_type_meta(input: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty_like(input, dtype=dtype)
<|endoftext|># mypy: allow-untyped-defs
import sys
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.utils import MatchAllNode, Pattern
from torch.fx.graph import Graph, Node
from torch.nn.utils.parametrize import type_before_parametrizations

from .graph_module import _is_observed_standalone_module
from .quantize_handler import QuantizeHandler


__all__: list[str] = []

# TODO(future PR): the 1st argument is typed as `List[Node]`, but a better type
# would be a recursive `List[Union[Node, Tuple[Union[Node, ...]]]]`
_MatchResult = tuple[Node, list[Node], Pattern | None, QuantizeHandler]

_MatchResultWithQConfig = tuple[
    Node, list[Node], Pattern | None, QuantizeHandler, QConfigAny
]


# Note: The order of patterns is important! match function will take whatever is matched first, so we'll
# need to put the fusion patterns before single patterns. For example, add_relu should be registered come before relu.
# decorators are applied in the reverse order we see. Also when we match the nodes in the graph with these patterns,
# we'll start from the last node of the graph and traverse back.
def _is_match(modules, node, pattern, max_uses=sys.maxsize):
    """Matches a node in fx against a pattern"""
    if isinstance(pattern, tuple):
        self_match, *arg_matches = pattern
        if self_match is getattr:
            if len(pattern) != 2:
                raise AssertionError("Expecting getattr pattern to have two elements")
            arg_matches = []
    else:
        self_match = pattern
        arg_matches = []

    if isinstance(self_match, type) and issubclass(self_match, MatchAllNode):
        return True

    if node == pattern:
        return True

    if not isinstance(node, Node) or len(node.users) > max_uses:
        return False

    if isinstance(self_match, type) and issubclass(self_match, torch.nn.Module):
        if node.op != "call_module":
            return False
        if type_before_parametrizations(modules[node.target]) != self_match:
            return False
    elif callable(self_match):
        if node.op != "call_function" or node.target is not self_match:
            return False
        elif node.target is getattr:
            if node.args[1] != pattern[1]:
                return False
    elif isinstance(self_match, str):
        if node.op != "call_method" or node.target != self_match:
            return False
    elif node.target != self_match:
        return False

    if not arg_matches:
        return True

    if len(arg_matches) != len(node.args):
        return False

    return all(
        _is_match(modules, node, arg_match, max_uses=1)
        for node, arg_match in zip(node.args, arg_matches)
    )


def _find_matches(
    graph: Graph,
    modules: dict[str, torch.nn.Module],
    patterns: dict[Pattern, QuantizeHandler],
    root_node_getter_mapping: dict[Pattern, Callable],
    standalone_module_names: list[str] | None = None,
    standalone_module_classes: list[type] | None = None,
    custom_module_classes: list[Any] | None = None,
) -> dict[str, _MatchResult]:
    """
    Matches the nodes in the input graph to quantization patterns, and
    outputs the information needed to quantize them in future steps.

    Inputs:
      - graph: an fx.Graph object
      - modules: a mapping of fully qualified module name to instance,
          for example, {'foo': ModuleFoo, ...}
      - patterns: a mapping from a tuple of nodes in reverse order to
          uninitialized QuantizeHandler subclass.

    Outputs a map of
      node_name ->
        (node, matched_values, matched_pattern, QuantizeHandler instance,
         qconfig)

    For example, {
      'relu_1': (relu_1, [relu_1], torch.nn.functional.relu,
                 <CopyNodeQuantizeHandler instance>, QConfig(...)),
      ...
    }
    """
    if custom_module_classes is None:
        custom_module_classes = []

    if standalone_module_classes is None:
        standalone_module_classes = []

    if standalone_module_names is None:
        standalone_module_names = []

    match_map: dict[str, _MatchResult] = {}
    all_matched: set[str] = set()

    def _recursive_record_node_in_match_map(
        last_node, match_map, node_pattern, matched_node_pattern, pattern, match_value
    ):
        if isinstance(node_pattern, Node):
            match_map[node_pattern.name] = (
                last_node,
                matched_node_pattern,
                pattern,
                match_value,
            )
        elif not isinstance(node_pattern, Iterable):
            return
        else:
            for n in node_pattern:
                _recursive_record_node_in_match_map(
                    last_node, match_map, n, matched_node_pattern, pattern, match_value
                )

    # TODO: 1. merge with fuse matcher 2. document the code
    def record_match(pattern, node, last_node, matched_node_pattern, match_map):
        if isinstance(pattern, tuple):
            s, *args = pattern
            is_single_arg = len(args) == 1
            current_node_pattern: list[Node] = []
            record_match(s, node, last_node, matched_node_pattern, match_map)
            if pattern[0] is not getattr:
                for subpattern, arg in zip(args, node.args):
                    record_match(subpattern, arg, node, current_node_pattern, match_map)
            if len(current_node_pattern) > 1:
                # current_node_pattern is  the node pattern we get from matching
                # the subpattern with arguments of the node
                # we use is_single_arg to recover the original structure of the pattern
                # if the original pattern has a single argument, we will have
                # (original_op, (original_arg, ...))
                # otherwise, we'll have a list of arguments
                # (original_op, arg0, arg1, arg2, ...)
                if is_single_arg:
                    matched_node_pattern.append(tuple(current_node_pattern))
                else:
                    matched_node_pattern.extend(list(current_node_pattern))
            else:
                matched_node_pattern.append(current_node_pattern[0])
        else:
            matched_node_pattern.append(node)

    for node in reversed(graph.nodes):
        if node.name not in match_map and node.name not in all_matched:
            for pattern, quantize_handler_cls in patterns.items():
                root_node_getter = root_node_getter_mapping.get(pattern)
                if _is_match(modules, node, pattern) and node.name not in match_map:
                    matched_node_pattern: list[Node] = []
                    record_match(pattern, node, node, matched_node_pattern, match_map)
                    quantize_handler = quantize_handler_cls(  # type: ignore[operator]
                        matched_node_pattern, modules, root_node_getter
                    )
                    last_node = node
                    # record the match for all nodes in the pattern
                    _recursive_record_node_in_match_map(
                        last_node,
                        match_map,
                        # we need to record all nodes in the matched pattern in the match_map
                        matched_node_pattern,
                        # this is a part of the value corresponding to the node
                        matched_node_pattern,
                        pattern,
                        quantize_handler,
                    )
                    break

    # add custom module instances to the match result
    if modules is None:
        raise AssertionError("modules must not be None")
    for node in graph.nodes:
        if (
            node.op == "call_module"
            and type(modules[node.target]) in custom_module_classes
        ):
            match_map[node.name] = (
                node,
                node,
                None,
                QuantizeHandler(node, modules, is_custom_module=True),
            )

    def is_standalone_module(node_target: str, modules: dict[str, torch.nn.Module]):
        if modules is None:
            raise AssertionError("modules must not be None")
        return (
            node_target in standalone_module_names
            or type(modules[node_target])  # type: ignore[operator]
            in standalone_module_classes  # type: ignore[operator]
        )

    # add standalone modules to the match
    for node in graph.nodes:
        if node.op == "call_module" and (
            is_standalone_module(node.target, modules)
            or _is_observed_standalone_module(modules[node.target])
        ):
            # add node to matched nodes
            match_map[node.name] = (
                node,
                node,
                None,
                QuantizeHandler(node, modules, is_standalone_module=True),
            )

    return match_map
<|endoftext|># mypy: allow-untyped-defs
import operator
from collections.abc import Callable
from typing import Any

import torch
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.quantized as nniq
import torch.ao.nn.intrinsic.quantized.dynamic as nniqd
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
import torch.ao.nn.quantized.reference as nnqr
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.nn.quantized.modules.utils import WeightedQuantizedModule
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.quantization_mappings import get_quantized_operator
from torch.ao.quantization.utils import _parent_name
from torch.fx import GraphModule, map_arg, Node
from torch.fx.graph import Graph

from .utils import (
    collect_producer_nodes,
    create_node_from_old_node_preserve_meta,
    get_linear_prepack_op_for_dtype,
    get_new_attr_name_with_prefix,
    get_qconv_prepack_op,
    graph_module_from_producer_nodes,
)


QOP_TO_ARG_NAMES_TO_SKIP: dict[Callable[..., Any], list[str]] = {
    torch._ops.ops.quantized.hardswish: ["inplace"],
    torch._ops.ops.quantized.elu: ["inplace"],
    torch._ops.ops.quantized.dropout: ["inplace"],
    torch._ops.ops.quantized.instance_norm: [
        "running_mean",
        "running_var",
        "use_input_stats",
        "momentum",
    ],
}


def _is_node_in_list(node, modules, func_list, method_list, module_type_list):
    is_call_function = node.op == "call_function" and node.target in func_list
    is_call_method = node.op == "call_method" and node.target in method_list
    is_call_module = (
        node.op == "call_module" and type(modules[str(node.target)]) in module_type_list
    )
    return is_call_function, is_call_method, is_call_module


def is_fixed_qparams_node(node, modules):
    func_list = [
        torch.nn.functional.hardsigmoid,
        torch.nn.functional.sigmoid,
        torch.sigmoid,
        torch.tanh,
    ]
    method_list = [
        "hardsigmoid",
        "hardsigmoid_",
        "sigmoid",
        "sigmoid_",
        "tanh",
        "tanh_",
    ]
    module_type_list = [
        torch.nn.Hardsigmoid,
        torch.nn.Sigmoid,
        torch.nn.Tanh,
        torch.nn.Softmax,
    ]
    return _is_node_in_list(node, modules, func_list, method_list, module_type_list)


def is_default_node(node, modules):
    func_list = [
        torch.nn.functional.elu,
        torch.nn.functional.hardswish,
        torch.nn.functional.instance_norm,
        torch.nn.functional.layer_norm,
        torch.nn.functional.leaky_relu,
        torch.nn.functional.dropout,
    ]
    method_list: list[Any] = []
    module_type_list = [
        nnqr.ConvTranspose1d,
        nnqr.ConvTranspose2d,
        nnqr.ConvTranspose3d,
        torch.nn.ELU,
        torch.nn.LeakyReLU,
        torch.nn.Hardswish,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.Dropout,
        torch.nn.PReLU,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.ao.nn.intrinsic.BNReLU2d,
        torch.ao.nn.intrinsic.BNReLU3d,
    ]
    return _is_node_in_list(node, modules, func_list, method_list, module_type_list)


def is_copy_node(node, modules):
    func_list = [
        torch.adaptive_avg_pool1d,
        torch.nn.functional.adaptive_avg_pool2d,
        torch.nn.functional.adaptive_avg_pool3d,
        torch.nn.functional.hardtanh,
        torch.nn.functional.hardtanh_,
        torch.nn.functional.interpolate,
        torch.nn.functional.max_pool1d,
        torch.nn.functional.max_pool2d,
        torch.nn.functional.max_pool3d,
        torch.nn.functional.relu,
        torch.nn.functional.relu6,
        torch.avg_pool1d,
        torch._C._nn.avg_pool2d,
        torch._C._nn.avg_pool3d,
        torch.clamp,
        torch.flatten,
        torch.mean,
        operator.floordiv,
        # F.channel_shuffle and torch.channel_shuffle are essentially the same thing
        # so we only need to put one of them here
        torch.channel_shuffle,
    ]
    method_list = [
        "clamp",
        "mean",
        "relu",
        "relu_",
    ]
    module_type_list = [
        torch.nn.AdaptiveAvgPool1d,
        torch.nn.AdaptiveAvgPool2d,
        torch.nn.AdaptiveAvgPool3d,
        torch.nn.AvgPool1d,
        torch.nn.AvgPool2d,
        torch.nn.AvgPool3d,
        torch.nn.Hardtanh,
        torch.nn.MaxPool1d,
        torch.nn.MaxPool2d,
        torch.nn.MaxPool3d,
        torch.nn.ReLU,
        torch.nn.ReLU6,
        torch.nn.ChannelShuffle,
    ]
    return _is_node_in_list(node, modules, func_list, method_list, module_type_list)


def is_general_tensor_shape_node(node, modules):
    func_list = [
        torch.narrow,
        torch.transpose,
        torch.repeat_interleave,
        torch.squeeze,
        torch.stack,
        torch.unsqueeze,
        torch.nn.functional.pixel_shuffle,
        torch.nn.functional.pixel_unshuffle,
    ]
    method_list = [
        "contiguous",
        "detach",
        "detach_",
        "permute",
        "repeat",
        "repeat_interleave",
        "reshape",
        "resize_",
        "shape",
        "size",
        "squeeze",
        "squeeze_",
        "transpose",
        "unsqueeze",
        "unsqueeze_",
        "view",
    ]
    module_type_list = [
        torch.nn.Identity,
        torch.nn.PixelShuffle,
        torch.nn.PixelUnshuffle,
    ]
    return _is_node_in_list(node, modules, func_list, method_list, module_type_list)


def is_other_node(node, modules):
    func_list = [
        torch.cat,
    ]
    method_list: list[Any] = []
    module_type_list: list[Any] = []
    return _is_node_in_list(node, modules, func_list, method_list, module_type_list)


def is_special_pattern_node(node, modules):
    res_function, res_method, res_module = False, False, False
    for checker in [
        is_fixed_qparams_node,
        is_default_node,
        is_copy_node,
        is_general_tensor_shape_node,
        is_other_node,
    ]:
        is_call_function, is_call_method, is_call_module = checker(node, modules)
        res_function = res_function or is_call_function
        res_method = res_method or is_call_method
        res_module = res_module or is_call_module
    return res_function, res_method, res_module


def is_dequantize_node(node):
    return (
        isinstance(node, Node)
        and node.op == "call_method"
        and node.target == "dequantize"
    )


def is_getattr_tensor_metadata_node(node):
    return (
        node.op == "call_function"
        and node.target is getattr
        and node.args[1] == "shape"
    )


def is_get_tensor_info_node(node):
    return node.op == "call_method" and node.target in ["shape", "size"]


def should_skip_lowering(op: torch.fx.node.Node, qconfig_map: dict[str, QConfigAny]):
    """
    Return True if the op is configured with a None qconfig, False otherwise.
    Note: maybe need to generalize this to also check for the dtype, and we
    only lower when dtype matches, but right now fbgemm/qnnpack only support
    a single dtype, so it is OK for now.
    """
    return op.name in qconfig_map and qconfig_map[op.name] is None


# Mapping from reference module class to the replacement static quantized module class for lowering
STATIC_LOWER_MODULE_MAP: dict[type[nn.Module], type[WeightedQuantizedModule]] = {
    nnqr.Linear: nnq.Linear,
    nnqr.Conv1d: nnq.Conv1d,
    nnqr.Conv2d: nnq.Conv2d,
    nnqr.Conv3d: nnq.Conv3d,
}

# Mapping from reference module class to the replacement dynamic quantized module class for lowering
DYNAMIC_LOWER_MODULE_MAP: dict[type[nn.Module], type[nn.Module]] = {
    nnqr.Linear: nnqd.Linear,
    nnqr.GRUCell: nnqd.GRUCell,
    nnqr.LSTMCell: nnqd.LSTMCell,
    nnqr.RNNCell: nnqd.RNNCell,
    nnqr.LSTM: nnqd.LSTM,
    nnqr.GRU: nnqd.GRU,
}

# Mapping from reference module class to the replacement weight only quantized module class for lowering
# TODO: correct the namespace for these modules
WEIGHT_ONLY_LOWER_MODULE_MAP: dict[type[nn.Module], type[nn.Module]] = {
    nnqr.Embedding: nnq.Embedding,
    nnqr.EmbeddingBag: nnq.EmbeddingBag,
}

# TODO: merge with STATIC_LOWER_MODULE_MAP after we merge
# _lower_static_weighted_ref_module and special_pattern_replacement
SPECIAL_PATTERN_LOWER_MODULE_MAP = {
    nn.BatchNorm2d: nnq.BatchNorm2d,
    nn.BatchNorm3d: nnq.BatchNorm3d,
    nnqr.ConvTranspose1d: nnq.ConvTranspose1d,
    nnqr.ConvTranspose2d: nnq.ConvTranspose2d,
    nnqr.ConvTranspose3d: nnq.ConvTranspose3d,
    nn.ELU: nnq.ELU,
    nn.LeakyReLU: nnq.LeakyReLU,
    nn.Hardswish: nnq.Hardswish,
    nn.InstanceNorm1d: nnq.InstanceNorm1d,
    nn.InstanceNorm2d: nnq.InstanceNorm2d,
    nn.InstanceNorm3d: nnq.InstanceNorm3d,
    nn.LayerNorm: nnq.LayerNorm,
    nn.Dropout: nnq.Dropout,
    nn.Softmax: nnq.Softmax,
    nn.PReLU: nnq.PReLU,
    nni.BNReLU2d: nniq.BNReLU2d,
    nni.BNReLU3d: nniq.BNReLU3d,
}

# Mapping from fused module class to a 2-tuple of:
#   1) The inner reference module class
#   2) The replacement static quantized module class for lowering
STATIC_LOWER_FUSED_MODULE_MAP: dict[
    type[nn.Module], tuple[type[nn.Module], type[WeightedQuantizedModule]]
] = {
    nni.LinearReLU: (nnqr.Linear, nniq.LinearReLU),
    # TODO: LinearLeakyReLU is registered as global but it is only fused and
    # lowered when ondnn's backend config is used. Maybe need to separate
    # registration and lowering functions for different backends in the future.
    nni.LinearLeakyReLU: (nnqr.Linear, nniq.LinearLeakyReLU),
    nni.LinearTanh: (nnqr.Linear, nniq.LinearTanh),
    nni.ConvReLU1d: (nnqr.Conv1d, nniq.ConvReLU1d),
    nni.ConvReLU2d: (nnqr.Conv2d, nniq.ConvReLU2d),
    nni.ConvReLU3d: (nnqr.Conv3d, nniq.ConvReLU3d),
}

# The difference between STATIC_LOWER_FUSED_MODULE_TWO_INPUTS_MAP and STATIC_LOWER_FUSED_MODULE_MAP:
# The refer node inside STATIC_LOWER_FUSED_MODULE_TWO_INPUTS_MAP has 2 inputs.
# Mapping from fused module class to a 2-tuple of:
#   1) The inner reference module class
#   2) The replacement static quantized module class for lowering
STATIC_LOWER_FUSED_MODULE_TWO_INPUTS_MAP: dict[
    type[nn.Module], tuple[type[nn.Module], type[WeightedQuantizedModule]]
] = {
    nni.ConvAdd2d: (nnqr.Conv2d, nniq.ConvAdd2d),
    nni.ConvAddReLU2d: (nnqr.Conv2d, nniq.ConvAddReLU2d),
}

# Mapping from fused module class to a 2-tuple of:
#   1) The inner reference module class
#   2) The replacement dynamic quantized module class for lowering
DYNAMIC_LOWER_FUSED_MODULE_MAP: dict[
    type[nn.Module], tuple[type[nn.Module], type[nn.Module]]
] = {
    nni.LinearReLU: (nnqr.Linear, nniqd.LinearReLU),
}

# Mapping from a functional to lower to a 2-tuple of
#   1) The quantized version of the op
#   2) The quantized version of the op fused with relu, if it exists, else None
STATIC_LOWER_FUNCTIONAL_MAP: dict[Callable, tuple[Callable, Callable | None]] = {
    F.linear: (torch.ops.quantized.linear, torch.ops.quantized.linear_relu),
    F.conv1d: (torch.ops.quantized.conv1d, torch.ops.quantized.conv1d_relu),
    F.conv2d: (torch.ops.quantized.conv2d, torch.ops.quantized.conv2d_relu),
    F.conv3d: (torch.ops.quantized.conv3d, torch.ops.quantized.conv3d_relu),
    F.conv_transpose1d: (torch.ops.quantized.conv_transpose1d, None),
    F.conv_transpose2d: (torch.ops.quantized.conv_transpose2d, None),
    F.conv_transpose3d: (torch.ops.quantized.conv_transpose3d, None),
}

WEIGHT_PREPACK_OPS: set[Callable] = {
    torch._ops.ops.quantized.linear_prepack,
    torch._ops.ops.quantized.linear_prepack_fp16,
    torch._ops.ops.quantized.conv1d_prepack,
    torch._ops.ops.quantized.conv2d_prepack,
    torch._ops.ops.quantized.conv3d_prepack,
    torch.ops.quantized.conv_transpose1d_prepack,
    torch.ops.quantized.conv_transpose2d_prepack,
    torch.ops.quantized.conv_transpose3d_prepack,
}

# Mapping from a functional to a dictionary, where the key is a 2-tuple of
# (input_activation_dtype, weight_dtype) and the value is a 2-tuple of
#   1) The dynamically quantized version of the op
#   2) The dynamically quantized version of the op fused with relu, if it exists, else None
DYNAMIC_LOWER_FUNCTIONAL_MAP: dict[
    Callable, dict[tuple[torch.dtype, torch.dtype], tuple[Callable, Callable | None]]
] = {
    F.linear: {
        (torch.quint8, torch.qint8): (
            torch.ops.quantized.linear_dynamic,
            torch.ops.quantized.linear_relu_dynamic,
        ),
        (torch.float16, torch.float16): (
            torch.ops.quantized.linear_dynamic_fp16,
            torch.ops.quantized.linear_relu_dynamic_fp16,
        ),
    },
    # dynamic conv + relu is not available yet
    F.conv1d: {
        (torch.quint8, torch.qint8): (torch.ops.quantized.conv1d_dynamic, None),
    },
    F.conv2d: {
        (torch.quint8, torch.qint8): (torch.ops.quantized.conv2d_dynamic, None),
    },
    F.conv3d: {
        (torch.quint8, torch.qint8): (torch.ops.quantized.conv3d_dynamic, None),
    },
}

CONV_FUNCTIONAL_OPS: set[Callable] = {
    F.conv1d,
    F.conv2d,
    F.conv3d,
}

CONV_TRANSPOSE_FUNCTIONAL_OPS: set[Callable] = {
    F.conv_transpose1d,
    F.conv_transpose2d,
    F.conv_transpose3d,
}

# TODO: add tests for lowering these ops
QBIN_OP_MAPPING: dict[Callable | str, Callable] = {
    operator.add: torch.ops.quantized.add,
    torch.add: torch.ops.quantized.add,
    operator.mul: torch.ops.quantized.mul,
    operator.matmul: torch.ops.quantized.matmul,
    torch.mul: torch.ops.quantized.mul,
    torch.matmul: torch.ops.quantized.matmul,
}
QBIN_RELU_OP_MAPPING: dict[Callable | str, Callable] = {
    operator.add: torch.ops.quantized.add_relu,
    torch.add: torch.ops.quantized.add_relu,
    operator.mul: torch.ops.quantized.mul_relu,
    torch.mul: torch.ops.quantized.mul_relu,
}

ORIGINAL_WEIGHTS_LOOKUP = "original_weights_lookup"


def _save_packed_weight(self, destination, prefix, keep_vars):
    for attr_name in dir(self):
        if "_packed_weight" in attr_name and isinstance(
            getattr(self, attr_name), torch._C.ScriptObject
        ):  # type: ignore[attr-defined]
            packed_weight = getattr(self, attr_name)
            destination[prefix + attr_name] = packed_weight


def _load_packed_weight(
    self,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    attrs_to_pop = []
    for attr_name in state_dict:
        if attr_name.startswith("_packed_weight") and isinstance(
            state_dict[attr_name], torch._C.ScriptObject
        ):  # type: ignore[attr-defined]
            setattr(self, attr_name, state_dict[attr_name])
            attrs_to_pop.append(attr_name)

    # pop the packed param attributes
    for attr_name in attrs_to_pop:
        state_dict.pop(attr_name)


def fold_weight(
    quantized_model: GraphModule,
    node_name_to_scope: dict[str, tuple[str, type]],
    keep_original_weights: bool = False,
) -> GraphModule:
    """
    Trace back from the weight node util we hit getattr, reconstruct the
    graph module with the traced nodes and run the graph module to pack the
    weight. then replace the original chain of ops with the packed weight.
    """
    packed_weights = {}
    # map from folded node name to the prepacked weight name
    folded_nodes = {}
    original_weights_lookup: dict[str, list] = {}
    lookup_counter = 0
    # get packed weights
    for node in quantized_model.graph.nodes:
        if node.op == "call_function" and node.target in WEIGHT_PREPACK_OPS:
            nodes_to_fold = collect_producer_nodes(node)
            if nodes_to_fold is not None:
                for node_to_fold in nodes_to_fold:
                    folded_nodes[node_to_fold.name] = node

                prepacking_module = graph_module_from_producer_nodes(
                    quantized_model, nodes_to_fold
                )
                packed_weight = prepacking_module()
                packed_weights[node.name] = packed_weight
                if keep_original_weights:
                    original_weights = list(prepacking_module.state_dict().values())
                    original_weights_lookup[str(lookup_counter)] = sorted(
                        original_weights, key=lambda x: x.numel(), reverse=True
                    )
                    if len(original_weights_lookup[str(lookup_counter)]) == 1:
                        # bias is None
                        original_weights_lookup[str(lookup_counter)].append(None)
                    lookup_counter += 1
    lookup_counter = 0

    # remove folded nodes and replace the prepacking node with getattr
    folded_graph = Graph()
    env: dict[Any, Any] = {}

    def load_arg(a):
        return map_arg(a, lambda node: env[node.name])

    for node in quantized_model.graph.nodes:
        prepack_node = folded_nodes.get(node.name)
        if prepack_node is node:
            packed_weight = packed_weights[node.name]
            # add a prepacked attribute to root
            op_node = next(iter(prepack_node.users))
            module_path, _ = node_name_to_scope[op_node.name]
            get_new_packed_weight_name = get_new_attr_name_with_prefix(
                module_path + "_packed_weight_"
            )
            packed_weight_name = get_new_packed_weight_name(quantized_model)
            setattr(quantized_model, packed_weight_name, packed_weight)
            # replace prepack node with a getattr node
            env[node.name] = folded_graph.create_node(
                "get_attr", packed_weight_name, (), {}
            )
            if keep_original_weights:
                key_name = (
                    packed_weight_name.replace(":", "_")
                    .replace("/", "_")
                    .replace("|", "_")
                    .replace(" ", "")
                    .lower()
                )
                original_weights_lookup[key_name] = original_weights_lookup[
                    str(lookup_counter)
                ]
                del original_weights_lookup[str(lookup_counter)]
                lookup_counter += 1
        elif prepack_node is not None:
            # remove the fold node
            continue
        else:
            # copy other nodes
            env[node.name] = folded_graph.node_copy(node, load_arg)

    quantized_model = GraphModule(quantized_model, folded_graph)
    quantized_model._register_state_dict_hook(_save_packed_weight)
    quantized_model.register_load_state_dict_pre_hook(_load_packed_weight)

    if keep_original_weights:
        setattr(quantized_model, ORIGINAL_WEIGHTS_LOOKUP, original_weights_lookup)

    return quantized_model


def _get_module(node: Node, modules: dict[str, nn.Module]) -> nn.Module | None:
    """
    Return the `torch.nn.Module` that corresponds to the specified node's target.
    If no such node exists, return None.
    """
    if node.op == "call_module" and str(node.target) in modules:
        return modules[str(node.target)]
    else:
        return None


def _match_static_pattern(
    node: Node,
    modules: dict[str, nn.Module],
    qconfig_map: dict[str, QConfigAny],
    matching_modules_or_ops: list[Callable],
    dequantize_node_arg_indices: list[int],
) -> tuple[Node, Node, Node] | tuple[None, None, None]:
    """
    Match the pattern (dequantize - ref node - quantize) against the node provided.

    If there is a match, return a 3-tuple of:
      1) q_node: the quantize node,
      2) relu_node: a relu node wrapping the ref_node, and
      3) ref_node: a reference module or functional node to replace with its quantized counterpart
    Otherwise, if there is no match, return a 3-tuple of (None, None, None).

    Parameters:
      node: The `torch.fx.Node` to match against.
      modules: A mapping from node names to modules in the model graph, used for module lookup.
      qconfig_map: A mapping from node names to the qconfigs associated with the nodes.
          If the corresponding qconfig for the reference node is None, then return no match.
      matching_modules_or_ops: Either a list of functions or a list of `torch.nn.Module`s.
          If the reference node is not in this list, then return no match.
      dequantize_node_arg_indices: A list of indices in the reference node args where dequantize
          nodes may be present. An empty list means skipping the check for dequantize nodes.
    """
    SKIP_LOWERING_VALUE = (None, None, None)

    # Match quantize node
    if node.op != "call_function" or node.target != torch.quantize_per_tensor:
        return SKIP_LOWERING_VALUE
    q_node = node
    ref_node = q_node.args[0]
    if not isinstance(ref_node, Node):
        raise AssertionError("Expected the reference node to be a torch.fx Node")

    # Handle cases where the node is wrapped in a ReLU
    if (ref_node.op == "call_function" and ref_node.target in (F.relu, torch.relu)) or (
        ref_node.op == "call_module" and type(_get_module(ref_node, modules)) is nn.ReLU
    ):
        relu_node = ref_node
        ref_node = relu_node.args[0]
        if not isinstance(ref_node, Node):
            raise AssertionError(
                "Expected the reference node after ReLU to be a torch.fx Node"
            )
    else:
        relu_node = None
    if should_skip_lowering(ref_node, qconfig_map):
        return SKIP_LOWERING_VALUE

    # Match reference module or functional
    if isinstance(matching_modules_or_ops[0], type) and issubclass(
        matching_modules_or_ops[0], nn.Module
    ):
        expected_op = "call_module"
        match_key = type(_get_module(ref_node, modules))
    else:
        expected_op = "call_function"
        match_key = ref_node.target  # type: ignore[assignment]
    if ref_node.op != expected_op or match_key not in matching_modules_or_ops:
        return SKIP_LOWERING_VALUE

    # Match dequantize node(s). Both of the following conditions must pass:
    # (1) All `torch.fx.Node`s at the matching indices must be a dequantize node
    # (2) There must be at least one dequantize node
    matched_dequantize = False
    for i in dequantize_node_arg_indices:
        if i >= len(ref_node.args):
            raise AssertionError(
                f"Dequantize index {i} exceeded reference node's arg length {len(ref_node.args)}"
            )
        arg = ref_node.args[i]
        if is_dequantize_node(arg):
            matched_dequantize = True
        elif isinstance(arg, Node):
            return SKIP_LOWERING_VALUE
    if not matched_dequantize:
        return SKIP_LOWERING_VALUE

    return (q_node, relu_node, ref_node)  # type: ignore[return-value]


def _match_static_pattern_with_two_inputs(
    node: Node,
    modules: dict[str, nn.Module],
    qconfig_map: dict[str, QConfigAny],
    matching_modules_or_ops: list[Callable],
) -> tuple[Node, Node] | tuple[None, None]:
    """
                      (dequantize \
    Match the pattern (dequantize - ref node - quantize) against the node provided.

    If there is a match, return a 2-tuple of:
      1) q_node: the quantize node,
      2) ref_node: a reference module or functional node to replace with its quantized counterpart
    Otherwise, if there is no match, return a 2-tuple of (None, None).

    Parameters:
      node: The `torch.fx.Node` to match against.
      modules: A mapping from node names to modules in the model graph, used for module lookup.
      qconfig_map: A mapping from node names to the qconfigs associated with the nodes.
          If the corresponding qconfig for the reference node is None, then return no match.
      matching_modules_or_ops: Either a list of functions or a list of `torch.nn.Module`s.
          If the reference node is not in this list, then return no match.
    """
    SKIP_LOWERING_VALUE = (None, None)

    # Match quantize node
    if node.op != "call_function" or node.target != torch.quantize_per_tensor:
        return SKIP_LOWERING_VALUE
    q_node = node
    ref_node = q_node.args[0]
    if not isinstance(ref_node, Node):
        raise AssertionError("Expected the reference node to be a torch.fx Node")

    if should_skip_lowering(ref_node, qconfig_map):
        return SKIP_LOWERING_VALUE

    # Match reference module or functional
    if isinstance(matching_modules_or_ops[0], type) and issubclass(
        matching_modules_or_ops[0], nn.Module
    ):
        expected_op = "call_module"
        match_key = type(_get_module(ref_node, modules))
    else:
        # This pass only support op of "call_module"
        return SKIP_LOWERING_VALUE

    if ref_node.op != expected_op or match_key not in matching_modules_or_ops:
        return SKIP_LOWERING_VALUE

    # Check ref_node has 2 input nodes, both are dq node.
    if len(ref_node.args) != 2:
        return SKIP_LOWERING_VALUE
    for i in range(len(ref_node.args)):
        arg = ref_node.args[i]
        if not is_dequantize_node(arg):
            return SKIP_LOWERING_VALUE

    return (q_node, ref_node)


def _lower_static_weighted_ref_module(
    model: GraphModule, qconfig_map: dict[str, QConfigAny]
):
    """
    Traverse the graph and find dequantize - ref module - quantize patterns
    and replace them with the quantized version of the ref module.
    """
    modules = dict(model.named_modules(remove_duplicate=False))
    for n in model.graph.nodes:
        # Step 0: Find nodes that match this pattern (dequantize - ref module - quantize)
        matching_modules = list(STATIC_LOWER_MODULE_MAP.keys()) + list(
            STATIC_LOWER_FUSED_MODULE_MAP.keys()
        )
        q_node, _relu_node, ref_node = _match_static_pattern(
            n,
            modules,
            qconfig_map,
            matching_modules,  # type: ignore[arg-type]
            dequantize_node_arg_indices=[0],
        )
        if q_node is None:
            continue
        if ref_node is None:
            raise AssertionError(
                "Expected a reference node when matching static pattern"
            )
        (_, scale_node, zero_point_node, _) = q_node.args
        ref_module = _get_module(ref_node, modules)
        ref_class = type(ref_module)
        if not isinstance(scale_node, Node):
            raise AssertionError("Expected scale_node to be a Node")
        if not isinstance(zero_point_node, Node):
            raise AssertionError("Expected zero_point_node to be a Node")
        if not issubclass(ref_class, nn.Module):
            raise AssertionError(
                "Expected reference module class to be a subclass of nn.Module"
            )

        # Step 1: Change this pattern to use the corresponding quantized module
        # For fused modules, we also check whether the inner module is a reference module
        # If so, we replace the entire fused module with the corresponding quantized module
        if ref_class in STATIC_LOWER_FUSED_MODULE_MAP:
            inner_ref_class, q_class = STATIC_LOWER_FUSED_MODULE_MAP[ref_class]
            if type(ref_module[0]) is not inner_ref_class:  # type: ignore[index]
                continue
        else:
            q_class = STATIC_LOWER_MODULE_MAP[ref_class]
        output_scale = getattr(model, scale_node.target)  # type: ignore[arg-type]
        output_zero_point = getattr(model, zero_point_node.target)  # type: ignore[arg-type]
        q_module = q_class.from_reference(ref_module, output_scale, output_zero_point)
        # replace reference module with quantized module
        parent_name, module_name = _parent_name(ref_node.target)
        setattr(modules[parent_name], module_name, q_module)

        # Step 2: Reroute around dq_node, and remove q_node and its args
        if len(ref_node.args) != 1:
            raise AssertionError("Expected reference node to have exactly 1 arg")
        dq_node = ref_node.args[0]
        if not isinstance(dq_node, Node):
            raise AssertionError("Expected dq_node to be a Node")
        ref_node.replace_input_with(dq_node, dq_node.args[0])  # type: ignore[arg-type]
        q_node.replace_all_uses_with(ref_node)
        model.graph.erase_node(q_node)
        model.graph.erase_node(scale_node)
        model.graph.erase_node(zero_point_node)


def _lower_static_weighted_ref_module_with_two_inputs(
    model: GraphModule, qconfig_map: dict[str, QConfigAny]
):
    """
    Traverse the graph and find patterns
    dequantize   dequantize
       \\         //
        ref module
            \\
          quantize
    and replace them with the quantized version of the ref module.
    """
    modules = dict(model.named_modules(remove_duplicate=False))
    for n in model.graph.nodes:
        #                                            (dequantize \
        # Step 0: Find nodes that match this pattern (dequantize - ref module - quantize)
        matching_modules = list(STATIC_LOWER_FUSED_MODULE_TWO_INPUTS_MAP.keys())
        (q_node, ref_node) = _match_static_pattern_with_two_inputs(
            n,
            modules,
            qconfig_map,
            matching_modules,  # type: ignore[arg-type]
        )
        if q_node is None:
            continue
        if ref_node is None:
            raise AssertionError(
                "Expected a reference node when matching static pattern with two inputs"
            )
        (_, scale_node, zero_point_node, _) = q_node.args
        ref_module = _get_module(ref_node, modules)
        ref_class = type(ref_module)
        if not isinstance(scale_node, Node):
            raise AssertionError("Expected scale_node to be a Node")
        if not isinstance(zero_point_node, Node):
            raise AssertionError("Expected zero_point_node to be a Node")
        if not issubclass(ref_class, nn.Module):
            raise AssertionError(
                "Expected reference module class to be a subclass of nn.Module"
            )

        # Step 1: Change this pattern to use the corresponding quantized module
        # For fused modules, we also check whether the inner module is a reference module
        # If so, we replace the entire fused module with the corresponding quantized module
        if ref_class in STATIC_LOWER_FUSED_MODULE_TWO_INPUTS_MAP:
            inner_ref_class, q_class = STATIC_LOWER_FUSED_MODULE_TWO_INPUTS_MAP[
                ref_class
            ]
            if type(ref_module[0]) is not inner_ref_class:  # type: ignore[index]
                continue
        else:
            continue
        output_scale = getattr(model, scale_node.target)  # type: ignore[arg-type]
        output_zero_point = getattr(model, zero_point_node.target)  # type: ignore[arg-type]
        q_module = q_class.from_reference(ref_module, output_scale, output_zero_point)
        # replace reference module with quantized module
        parent_name, module_name = _parent_name(ref_node.target)
        setattr(modules[parent_name], module_name, q_module)

        # Step 2: Reroute around dq_node, and remove q_node and its args
        if len(ref_node.args) != 2:
            raise AssertionError("Expected reference node to have exactly 2 args")
        for arg in ref_node.args:
            if not is_dequantize_node(arg):
                continue
            dq_node = arg
            if not isinstance(dq_node, Node):
                raise AssertionError("Expected dq_node to be a Node")
            ref_node.replace_input_with(dq_node, dq_node.args[0])  # type: ignore[arg-type]

        q_node.replace_all_uses_with(ref_node)
        model.graph.erase_node(q_node)
        model.graph.erase_node(scale_node)
        model.graph.erase_node(zero_point_node)


def _lower_dynamic_weighted_ref_module(model: GraphModule):
    """
    Traverse the graph and find quantize_per_tensor_dynamic - dequantize - ref_module patterns
    and replace them with the dynamically quantized version of the ref module.
    """
    named_modules = dict(model.named_modules(remove_duplicate=False))
    for n in model.graph.nodes:
        if n.op != "call_module" or type(named_modules[str(n.target)]) not in set(
            DYNAMIC_LOWER_MODULE_MAP.keys()
        ).union(set(DYNAMIC_LOWER_FUSED_MODULE_MAP.keys())):
            continue
        ref_node = n
        dq_node = ref_node.args[0]
        if dq_node.op != "call_method" or dq_node.target != "dequantize":
            continue

        input_dynamic_q_node = dq_node.args[0]

        if (
            input_dynamic_q_node.op != "call_function"
            or input_dynamic_q_node.target != torch.quantize_per_tensor_dynamic
        ):
            continue

        activation_dtype = input_dynamic_q_node.args[1]
        is_fp16 = activation_dtype == torch.float16
        is_int8 = activation_dtype in [torch.quint8, torch.qint8]
        if not is_int8 and not is_fp16:
            continue

        ref_module = named_modules[str(ref_node.target)]
        ref_class = type(ref_module)
        if ref_class in DYNAMIC_LOWER_FUSED_MODULE_MAP:
            inner_ref_class, q_class = DYNAMIC_LOWER_FUSED_MODULE_MAP[ref_class]
            if type(ref_module[0]) is not inner_ref_class:
                continue
        else:
            q_class = DYNAMIC_LOWER_MODULE_MAP.get(ref_class)  # type: ignore[assignment]
        # TODO: maybe define a WeightedDynamicallyQuantizedModule
        q_module = q_class.from_reference(ref_module)  # type: ignore[attr-defined]

        # replace reference module with dynamically quantized module
        parent_name, module_name = _parent_name(ref_node.target)
        setattr(named_modules[parent_name], module_name, q_module)
        ref_node.replace_input_with(dq_node, input_dynamic_q_node.args[0])


def _lower_weight_only_weighted_ref_module(model: GraphModule):
    """
    Traverse the graph and find ref_module patterns
    and replace them with the weight only quantized version of the ref module.
    """
    named_modules = dict(model.named_modules(remove_duplicate=False))
    for n in model.graph.nodes:
        if n.op != "call_module" or type(named_modules[str(n.target)]) not in set(
            WEIGHT_ONLY_LOWER_MODULE_MAP.keys()
        ):
            continue
        ref_node = n
        ref_module = named_modules[str(ref_node.target)]
        ref_class = type(ref_module)
        q_class = WEIGHT_ONLY_LOWER_MODULE_MAP.get(ref_class)
        # TODO: WeightedQuantizedModule is currently assuming static quant apis
        # with output_scale, output_zero_point in from_reference, we may want to
        # relax that, or rename this
        # TODO: maybe define a WeightedWeightOnlyQuantizedModule
        q_module = q_class.from_reference(ref_module)  # type: ignore[union-attr]

        # replace reference module with dynamically quantized module
        parent_name, module_name = _parent_name(ref_node.target)
        setattr(named_modules[parent_name], module_name, q_module)


def _lower_static_weighted_ref_functional(
    model: GraphModule, qconfig_map: dict[str, QConfigAny]
):
    """
    Traverse the graph and replace functional reference patterns with their quantized versions.
    """
    modules = dict(model.named_modules(remove_duplicate=False))
    for n in model.graph.nodes:
        # Step 0: Find nodes that match this pattern (dequantize - functional op - quantize)
        matching_ops = list(STATIC_LOWER_FUNCTIONAL_MAP.keys())
        (q_node, relu_node, func_node) = _match_static_pattern(
            n, modules, qconfig_map, matching_ops, dequantize_node_arg_indices=[0, 1]
        )
        if q_node is None:
            continue
        if func_node is None:
            raise AssertionError(
                "Expected a function node when matching static functional pattern"
            )
        (_, output_scale_node, output_zp_node, _) = q_node.args
        (input_dq_node, weight_dq_node, *remaining_func_args) = func_node.args
        if not isinstance(output_zp_node, Node):
            raise AssertionError("Expected output_zp_node to be a Node")
        if not isinstance(input_dq_node, Node):
            raise AssertionError("Expected input_dq_node to be a Node")
        if not isinstance(weight_dq_node, Node):
            raise AssertionError("Expected weight_dq_node to be a Node")
        quantized_weight = weight_dq_node.args[0]
        if not isinstance(quantized_weight, Node):
            raise AssertionError("Expected quantized_weight to be a Node")
        if quantized_weight.op != "call_function" or quantized_weight.target not in (
            torch.quantize_per_tensor,
            torch.quantize_per_channel,
        ):
            continue

        # Step 1: Replace quantized weights with packed weights, which will be folded later
        # Use the right prepack op and prepare the corresponding args
        # Linear prepack args: (quantized weights[, bias])
        # Conv prepack args: (quantized weights[, bias, stride, padding, dilation, groups])
        prepack_args: list[Any] = [quantized_weight] + remaining_func_args
        if func_node.target is F.linear:
            weight_dtype = quantized_weight.args[-1]
            prepack_op = get_linear_prepack_op_for_dtype(weight_dtype)
        elif func_node.target in CONV_FUNCTIONAL_OPS:
            prepack_op = get_qconv_prepack_op(func_node.target)  # type: ignore[arg-type]
            # For conv1d, the stride, padding, and dilation args may be ints,
            # in which case we need to convert them to tuples
            if func_node.target is F.conv1d:
                for i in [2, 3, 4]:
                    if len(prepack_args) > i and isinstance(prepack_args[i], int):
                        prepack_args[i] = (prepack_args[i],)
        elif func_node.target in CONV_TRANSPOSE_FUNCTIONAL_OPS:
            prepack_op = get_qconv_prepack_op(func_node.target)  # type: ignore[arg-type]
            # For conv_transpose1d, the stride, padding, and dilation args may be ints,
            # in which case we need to convert them to tuples
            if func_node.target is F.conv_transpose1d:
                # Note prepack_args[5] is groups.
                for i in [2, 3, 4, 6]:
                    if len(prepack_args) > i and isinstance(prepack_args[i], int):
                        prepack_args[i] = (prepack_args[i],)
            # swap dilation and groups
            # prepack op has arguments: {w, b, stride, padding, output_padding, dilation, groups}
            # transposed conv op has arguments: {x, w, b, stride, padding, output_padding, groups, dilation}
            if len(prepack_args) > 6:
                prepack_args[5], prepack_args[6] = prepack_args[6], prepack_args[5]
        else:
            raise ValueError(f"Lowering is not supported for op '{func_node.target}'")
        with model.graph.inserting_before(output_scale_node):  # type: ignore[arg-type]
            # kwargs of the func node are needed for prepack op (i.e., quantized::linear_prepack)
            # They are not needed for compute op (i.e., quantized::linear)
            kwargs = func_node.kwargs
            # F.linear uses 'bias' key for bias while qlinear_prepack uses 'B' for bias
            if func_node.target is F.linear and "bias" in kwargs:
                kwargs = kwargs.copy()
                kwargs["B"] = kwargs["bias"]
                del kwargs["bias"]
            packed_weight = model.graph.create_node(
                "call_function", prepack_op, tuple(prepack_args), kwargs
            )

        # Step 2: Replace reference pattern with the corresponding quantized op
        (q_func, q_relu_func) = STATIC_LOWER_FUNCTIONAL_MAP[func_node.target]  # type: ignore[index]
        # conv_transpose does not support fusion with relu yet. q_relu_func is None in such cases
        if q_relu_func is not None:
            func_node.target = q_relu_func if relu_node is not None else q_func
        else:
            func_node.target = q_func
        func_node.args = (
            input_dq_node.args[0],
            packed_weight,
            output_scale_node,
            output_zp_node,
        )
        # kwargs for func_node has been moved to kwargs for prepack op
        func_node.kwargs = {}
        q_node.replace_all_uses_with(func_node)
        # Move func_node after output_zp_node in the graph
        output_zp_node.append(func_node)

        # Clean up: Remove quantize node, and the relu node if it exists
        model.graph.erase_node(q_node)
        if relu_node is not None and q_relu_func is not None:
            model.graph.erase_node(relu_node)


def _lower_dynamic_weighted_ref_functional(
    model: GraphModule, qconfig_map: dict[str, QConfigAny]
):
    """
    Traverse the graph and replace functional reference patterns with their dynamically
    quantized versions.
    Examples:
    quantize_per_tensor_dynamic - dequantize - functional linear --> linear_dynamic
    to(torch.float16) - dequantize - functional linear --> linear_dynamic_fp16
    """
    modules = dict(model.named_modules(remove_duplicate=False))
    # we want to search in reserved order so that we can match the larger patterns first
    # e.g. we want to match linear - relu before linear.
    for n in reversed(model.graph.nodes):
        # Step 0: Find nodes that match this pattern
        # (quantize_per_tensor_dynamic - dequantize - dynamically quantized op)
        # We search for the pattern backwards, starting with the quantize node
        # Quantize node args: (func, scale, zp, dtype)
        func_node = n
        # Handle cases where the functional op is wrapped in a ReLU
        if (
            func_node.op == "call_function"
            and func_node.target is F.relu
            or func_node.op == "call_module"
            and type(modules[str(func_node.target)]) is torch.nn.ReLU
        ):
            relu_node = func_node
            func_node = relu_node.args[0]
        else:
            relu_node = None
        if should_skip_lowering(func_node, qconfig_map):
            continue
        # Linear args: (dequantized inputs, dequantized weights[, bias])
        # Conv args: (dequantized inputs, dequantized weights[, bias, stride, padding, dilation, groups])
        if (
            func_node.op != "call_function"
            or func_node.target not in DYNAMIC_LOWER_FUNCTIONAL_MAP
        ):
            continue
        (input_dq_node, weight_dq_node, *remaining_func_args) = func_node.args
        if (
            input_dq_node.op != "call_method"
            or input_dq_node.target != "dequantize"
            or weight_dq_node.op != "call_method"
            or weight_dq_node.target != "dequantize"
        ):
            continue

        input_dynamic_q_node = input_dq_node.args[0]

        if (
            input_dynamic_q_node.op != "call_function"
            or input_dynamic_q_node.target != torch.quantize_per_tensor_dynamic
        ):
            continue

        reduce_range_node = None
        (pattern_input, activation_dtype, reduce_range_node) = input_dynamic_q_node.args
        is_fp16 = activation_dtype == torch.float16
        is_int8 = activation_dtype in [torch.quint8, torch.qint8]
        if not is_int8 and not is_fp16:
            continue

        quantized_weight = weight_dq_node.args[0]
        weight_dtype = quantized_weight.args[-1]

        # Step 1: Try to select reference pattern with the corresponding quantized op
        dynamic_quant_dtype_key = (activation_dtype, weight_dtype)
        if (
            dynamic_quant_dtype_key
            not in DYNAMIC_LOWER_FUNCTIONAL_MAP[func_node.target]
        ):
            print(
                f"Didn't find dtype combination {dynamic_quant_dtype_key} during "
                f"dynamic quantized op lowering for {func_node.target}"
            )
            continue
        (q_func, q_relu_func) = DYNAMIC_LOWER_FUNCTIONAL_MAP[func_node.target][
            dynamic_quant_dtype_key
        ]

        if q_func is None or q_relu_func is None:
            print(
                "Didn't find corresponding quantized function or quantized relu function "
                f"for {func_node.target}, {dynamic_quant_dtype_key}"
            )
            continue

        # Step 2: Replace quantized weights with packed weights, which will be folded later
        # Use the right prepack op and prepare the corresponding args
        # Linear prepack args: (quantized weights[, bias])
        # Conv prepack args: (quantized weights[, bias, stride, padding, dilation, groups])
        prepack_args: list[Any] = [quantized_weight] + remaining_func_args
        prepack_kwargs = {}
        if func_node.target is F.linear:
            prepack_op = get_linear_prepack_op_for_dtype(weight_dtype)
            kwargs = func_node.kwargs.copy()
            if "bias" in kwargs:
                prepack_kwargs["B"] = kwargs["bias"]
                del kwargs["bias"]
                func_node.kwargs = kwargs
        elif func_node.target in CONV_FUNCTIONAL_OPS:
            prepack_op = get_qconv_prepack_op(func_node.target)
            # For conv1d, the stride, padding, and dilation args may be ints,
            # in which case we need to convert them to tuples
            if func_node.target is F.conv1d:
                for i in [2, 3, 4]:
                    if len(prepack_args) > i and isinstance(prepack_args[i], int):
                        prepack_args[i] = (prepack_args[i],)
        else:
            raise ValueError(f"Lowering is not supported for op '{func_node.target}'")
        with model.graph.inserting_before(func_node):
            packed_weight = model.graph.create_node(
                "call_function", prepack_op, tuple(prepack_args), prepack_kwargs
            )

        # Step 3: Replace reference pattern with the corresponding quantized op
        func_node.target = q_relu_func if relu_node is not None else q_func
        if is_int8:
            func_node.args = (pattern_input, packed_weight, reduce_range_node)
        else:
            func_node.args = (pattern_input, packed_weight)

        if relu_node is not None:
            relu_node.replace_all_uses_with(func_node)

        # Step 4: Remove the relu node if it exists
        if relu_node is not None:
            model.graph.erase_node(relu_node)


def _lower_quantized_binary_op(model: GraphModule, qconfig_map: dict[str, QConfigAny]):
    binary_ops_to_lower: list[Callable] = [
        operator.add,
        torch.add,
        operator.mul,
        torch.mul,
        torch.matmul,
    ]
    modules = dict(model.named_modules(remove_duplicate=False))
    for n in model.graph.nodes:
        # Step 0: Find nodes that match this pattern (dequantize - ref module - quantize)
        (q_node, relu_node, bop_node) = _match_static_pattern(
            n,
            modules,
            qconfig_map,
            binary_ops_to_lower,
            dequantize_node_arg_indices=[0, 1],
        )
        if q_node is None:
            continue
        if bop_node is None:
            raise AssertionError(
                "Expected a binary op node when matching quantized binary op pattern"
            )
        (_, scale_node, zero_point_node, _) = q_node.args

        # Step 1: Remove dequant nodes
        num_dq_nodes = 0
        for arg in bop_node.args:
            if not is_dequantize_node(arg):
                continue
            dq_node = arg
            if not isinstance(dq_node, Node):
                raise AssertionError("Expected dq_node to be a Node")
            dn_input = dq_node.args[0]
            bop_node.replace_input_with(dq_node, dn_input)  # type: ignore[arg-type]
            num_dq_nodes += 1
        if num_dq_nodes <= 0:
            raise AssertionError(
                "Expected at least one dequantize node in binary op args"
            )

        # Step 2: Swap binary op to quantized binary op
        if bop_node.target not in QBIN_OP_MAPPING:
            raise AssertionError(
                f"Unsupported binary op {bop_node.target} for lowering"
            )
        binop_to_qbinop = QBIN_OP_MAPPING if relu_node is None else QBIN_RELU_OP_MAPPING
        qbin_op = binop_to_qbinop[bop_node.target]
        # prepare the args for quantized binary op
        # (x, y)
        qop_node_args = list(bop_node.args)
        # (x, y, scale, zero_point)
        # add scale and zero_point arguments for Tensor - Tensor operation
        if num_dq_nodes == 2:
            qop_node_args.extend([scale_node, zero_point_node])
        # insert a call to quantized binary op and remove the original binary op
        with model.graph.inserting_after(q_node):
            qop_node = create_node_from_old_node_preserve_meta(
                model.graph,
                ("call_function", qbin_op, tuple(qop_node_args), {}),
                bop_node,
            )
            q_node.replace_all_uses_with(qop_node)

        # Step 3: Remove quantize node, binary op node, and relu node if any
        model.graph.erase_node(q_node)
        if relu_node is not None:
            model.graph.erase_node(relu_node)
        model.graph.erase_node(bop_node)


def special_pattern_replacement(model: GraphModule):
    modules = dict(model.named_modules(remove_duplicate=False))
    for n in model.graph.nodes:
        q_node = n
        is_quantize = q_node.target is torch.quantize_per_tensor
        is_to_fp16 = (
            q_node.op == "call_method"
            and q_node.target == "to"
            and len(q_node.args) == 2
            and q_node.args[1] == torch.float16
        )
        # Only continue when neither quantize nor to_fp16
        if not is_quantize and not is_to_fp16:
            continue
        ref_node = q_node.args[0]
        # get output scale/zero_point/dtype from the quantize node
        # ref_node, scale_node, zero_point_node, dtype = q_node.args
        # TODO: add safety checks that users for the ref_node and dq_node needs to be one
        is_call_function, is_call_method, is_call_module = is_fixed_qparams_node(
            ref_node, modules
        )
        if is_to_fp16 and (is_call_function or is_call_method or is_call_module):
            # TODO: add a warning or error out here? (bc-breaking if error out)
            # warnings.warn(
            #     "Only reference patterns are currently supported for {dtype} dtype with {op} op"
            #     "".format(dtype=dtypes, op=ref_node))
            continue

        is_call_function, is_call_method, is_call_module = is_default_node(
            ref_node, modules
        )
        if is_to_fp16 and (is_call_function or is_call_method or is_call_module):
            # TODO: add a warning or error out here? (bc-breaking if error out)
            continue

        # This check includes all supported ops
        is_call_function, is_call_method, is_call_module = is_special_pattern_node(
            ref_node, modules
        )
        if not (is_call_module or is_call_function or is_call_method):
            continue
        if len(ref_node.args) <= 0 and len(ref_node.kwargs) <= 0:
            raise AssertionError("Expected ref_node to have args or kwargs")
        dq_node_or_nodes = (
            ref_node.args[0]
            if len(ref_node.args) > 0
            else next(iter(ref_node.kwargs.values()))
        )
        if not isinstance(dq_node_or_nodes, (Node, tuple, list)):
            raise AssertionError(
                "Expected dq_node_or_nodes to be a Node, tuple, or list"
            )
        is_dequantize = False
        if isinstance(dq_node_or_nodes, Node):
            is_dequantize = (
                dq_node_or_nodes.op == "call_method"
                and dq_node_or_nodes.target == "dequantize"
            )
        elif isinstance(dq_node_or_nodes, (tuple, list)):
            is_dequantize = all(
                x.op == "call_method" and x.target == "dequantize"
                for x in dq_node_or_nodes
            )

        if not is_dequantize:
            continue

        # TODO: enable we have patterns that needs to swap the modules
        if is_call_module:
            ref_module = modules[ref_node.target]
            if type(ref_module) in SPECIAL_PATTERN_LOWER_MODULE_MAP and is_quantize:
                qmodule_cls = SPECIAL_PATTERN_LOWER_MODULE_MAP.get(type(ref_module))
                scale_node = q_node.args[1]
                zero_point_node = q_node.args[2]
                output_scale = getattr(model, scale_node.target)
                output_zero_point = getattr(model, zero_point_node.target)

                qmodule = qmodule_cls.from_reference(  # type:ignore[union-attr]
                    ref_module, output_scale, output_zero_point
                )
                # replace reference module with quantized module
                parent_name, module_name = _parent_name(ref_node.target)
                setattr(modules[parent_name], module_name, qmodule)

        # reroute around dq node:
        dq_nodes: list[Node] = []
        if isinstance(dq_node_or_nodes, Node):
            dq_nodes = [dq_node_or_nodes]
        elif isinstance(dq_node_or_nodes, (tuple, list)):
            dq_nodes = list(dq_node_or_nodes)

        for dq_node in dq_nodes:
            dn_input = dq_node.args[0]
            ref_node.replace_input_with(dq_node, dn_input)

        # store q node args
        qnode_qparams = list(q_node.args)[1:]
        # replace uses of q node with input and remove q node
        q_node_input = q_node.args[0]
        q_node.replace_all_uses_with(q_node_input)
        model.graph.erase_node(q_node)

        is_call_function, is_call_method, is_call_module = is_default_node(
            ref_node, modules
        )
        if is_call_function:
            # pass scale/zer_point arguments from quantize_per_tensor to the default node operator
            # insert an op after the zero_point node so that the scale/zero_point
            # nodes are available
            qop = get_quantized_operator(ref_node.target)
            args = list(ref_node.args)
            kwargs = dict(ref_node.kwargs)
            if qop in QOP_TO_ARG_NAMES_TO_SKIP:
                args_to_skip = QOP_TO_ARG_NAMES_TO_SKIP[qop]
                for arg in args_to_skip:
                    if arg in kwargs:
                        kwargs.pop(arg)
            kwargs["output_scale"] = qnode_qparams[0]
            kwargs["output_zero_point"] = qnode_qparams[1]
            with model.graph.inserting_after(qnode_qparams[1]):
                qop_node = create_node_from_old_node_preserve_meta(
                    model.graph, ("call_function", qop, tuple(args), kwargs), ref_node
                )
                ref_node.replace_all_uses_with(qop_node)
                model.graph.erase_node(ref_node)
        else:
            # remove scale/zero_point node for quantize node
            for n in qnode_qparams:
                if isinstance(n, Node):
                    model.graph.erase_node(n)

    return model


def _lower_getattr_tensor_metadta_op(model: GraphModule):
    """Modified the graph of the model inplace, to skip extra dequantize op before
    the general tensor shape ops when possible
    """
    for n in model.graph.nodes:
        if is_getattr_tensor_metadata_node(n):
            maybe_dq = n.args[0]
            if maybe_dq.op != "call_method" or maybe_dq.target != "dequantize":
                continue
            # skip the dequantize node
            args = list(n.args)
            args[0] = n.args[0].args[0]
            n.args = tuple(args)


def _lower_get_tensor_info_op(model: GraphModule):
    """Modified the graph of the model inplace, to skip extra dequantize op before
    the general tensor shape ops when possible
    """
    for n in model.graph.nodes:
        if not is_get_tensor_info_node(n):
            continue
        maybe_dq = n.args[0]
        if maybe_dq.op != "call_method" or maybe_dq.target != "dequantize":
            continue
        # skip the dequantize node
        args = list(n.args)
        args[0] = n.args[0].args[0]
        n.args = tuple(args)


def _lower_to_native_backend(
    model: GraphModule,
    qconfig_map: dict[str, QConfigAny],
    node_name_to_scope: dict[str, tuple[str, type]],
    keep_original_weights: bool = False,
) -> GraphModule:
    """Lower a quantized reference model (with reference quantized operator patterns)
    to the native backend in PyTorch (fbgemm/qnnpack), both backends shares the same
    operator signature so they can be lowered with the same function
    """
    _lower_static_weighted_ref_module(model, qconfig_map)
    _lower_static_weighted_ref_module_with_two_inputs(model, qconfig_map)
    _lower_dynamic_weighted_ref_module(model)
    _lower_weight_only_weighted_ref_module(model)
    _lower_static_weighted_ref_functional(model, qconfig_map)
    _lower_dynamic_weighted_ref_functional(model, qconfig_map)
    _lower_quantized_binary_op(model, qconfig_map)
    _lower_getattr_tensor_metadta_op(model)
    _lower_get_tensor_info_op(model)
    special_pattern_replacement(model)
    model.graph.eliminate_dead_code()
    model = fold_weight(model, node_name_to_scope, keep_original_weights)
    model.graph.eliminate_dead_code()
    model.recompile()
    model.graph.lint()
    return model
<|endoftext|>from collections.abc import Callable

import torch
from torch.ao.nn.intrinsic import _FusedModule
from torch.fx._symbolic_trace import Tracer
from torch.fx.proxy import Scope


__all__ = [
    "QuantizationTracer",
]


class ScopeContextManager(torch.fx.proxy.ScopeContextManager):
    def __init__(
        self, scope: Scope, current_module: torch.nn.Module, current_module_path: str
    ):
        super().__init__(scope, Scope(current_module_path, type(current_module)))


class QuantizationTracer(Tracer):
    def __init__(
        self, skipped_module_names: list[str], skipped_module_classes: list[Callable]
    ):
        super().__init__()
        self.skipped_module_names = skipped_module_names
        self.skipped_module_classes = skipped_module_classes
        # NB: initialized the module_type of top level module to None
        # we are assuming people won't configure the model with the type of top level
        # module here, since people can use "" for global config
        # We can change this if there is a use case that configures
        # qconfig using top level module type
        self.scope = Scope("", None)
        self.record_stack_traces = not torch.fx.config.do_not_emit_stack_traces

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        return (
            (
                (
                    m.__module__.startswith("torch.nn")
                    or m.__module__.startswith("torch.ao.nn")
                )
                and not isinstance(m, torch.nn.Sequential)
            )
            or module_qualified_name in self.skipped_module_names
            or type(m) in self.skipped_module_classes
            or isinstance(m, _FusedModule)
        )
<|endoftext|># mypy: allow-untyped-defs
from abc import ABC
from collections.abc import Callable

import torch
from torch.ao.quantization.backend_config import (
    BackendConfig,
    DTypeConfig,
    ObservationType,
)
from torch.ao.quantization.utils import NodePattern, Pattern, QuantizerCls
from torch.fx.graph import Node

from .utils import all_node_args_have_no_tensors


__all__ = [
    "QuantizeHandler",
    "BinaryOpQuantizeHandler",
    "CatQuantizeHandler",
    "ConvReluQuantizeHandler",
    "LinearReLUQuantizeHandler",
    "BatchNormQuantizeHandler",
    "EmbeddingQuantizeHandler",
    "RNNDynamicQuantizeHandler",
    "DefaultNodeQuantizeHandler",
    "FixedQParamsOpQuantizeHandler",
    "CopyNodeQuantizeHandler",
    "GeneralTensorShapeOpQuantizeHandler",
    "CustomModuleQuantizeHandler",
    "StandaloneModuleQuantizeHandler",
]


def _default_root_node_getter(node_pattern):
    if node_pattern is None:
        return node_pattern
    while not isinstance(node_pattern, Node):
        node_pattern = node_pattern[-1]
    return node_pattern


# Base Pattern Handler
class QuantizeHandler(ABC):  # noqa: B024
    """Base handler class for the quantizer patterns"""

    def __init__(
        self,
        node_pattern: NodePattern,
        modules: dict[str, torch.nn.Module],
        root_node_getter: Callable | None = None,
        is_custom_module=False,
        is_standalone_module=False,
    ):
        """Records pattern information in __init__, which will be used
        in convert
        """
        self.node_pattern = node_pattern
        self.modules = modules
        if root_node_getter is None:
            root_node_getter = _default_root_node_getter
        self.root_node = root_node_getter(node_pattern)
        self.is_custom_module_ = is_custom_module
        self.is_standalone_module_ = is_standalone_module
        self.num_tensor_args = 0
        # determine how many of the first two args are Tensors (versus scalars)
        # this distinguishes things like "x + y" from "x + 2" or "2 + x"
        if isinstance(self.root_node, Node):
            cache_for_no_tensor_check: dict[Node, bool] = {}
            for arg_idx in range(len(self.root_node.args)):
                arg = self.root_node.args[arg_idx]
                if isinstance(arg, Node) and (
                    not all_node_args_have_no_tensors(
                        arg, self.modules, cache_for_no_tensor_check
                    )
                ):
                    self.num_tensor_args += 1

    def is_general_tensor_value_op(self) -> bool:
        """
        Returns True if the operator works for both floating point and
        quantized input, and does some computation based on the input Tensor,
        or the ops that only re-arranges the Tensor values or query some metadata
        about the Tensor
        so we need to insert observer/fake_quant for the output of the
        operator (same observer instance as input)
        since the distribution of values is different for input and output
        Tensors (for HistogramObserver) while they share the same quantization
        parameters
        Example operator: avgpool2d, reshape, transpose, maxpool2d
        Example observed operator:
        observer_0 - avgpool2d - observer_0 (same observer instance as input)
        """
        return False

    def is_custom_module(self):
        return self.is_custom_module_

    def is_standalone_module(self):
        return self.is_standalone_module_


def _get_quantize_handler_cls(
    observation_type: ObservationType,
    dtype_configs: list[DTypeConfig],
    num_tensor_args_to_observation_type: dict[int, ObservationType],
) -> type[QuantizeHandler]:
    """
    Return a configurable QuantizeHandler that matches the given specifications from the backend.
    """

    class ConfigurableQuantizeHandler(QuantizeHandler):
        def __init__(
            self,
            node_pattern: NodePattern,
            modules: dict[str, torch.nn.Module],
            root_node_getter: Callable | None = None,
        ):
            super().__init__(node_pattern, modules, root_node_getter)
            if num_tensor_args_to_observation_type:
                if self.num_tensor_args not in num_tensor_args_to_observation_type:
                    raise AssertionError(
                        f"Must provide observation_type config for tensor number {self.num_tensor_args}"
                        f" in num_tensor_args_to_observation_type for {node_pattern}"
                    )
                self.observation_type = num_tensor_args_to_observation_type[
                    self.num_tensor_args
                ]
            else:
                self.observation_type = observation_type
            self.dtype_configs = dtype_configs

        def is_general_tensor_value_op(self) -> bool:
            return (
                self.observation_type
                == ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT
            )

    return ConfigurableQuantizeHandler


def _get_pattern_to_quantize_handlers(
    backend_config: BackendConfig,
) -> dict[Pattern, QuantizerCls]:
    """
    Note: Quantize handler is just a holder for some check methods like
    (should_insert_observer_for_output), maybe this can be a enum as well,
    we can refactor this after we convert the path for fbgemm/qnnpack fully to the
    new path, this is not exposed to backend developers
    """
    pattern_to_quantize_handlers = {}
    for pattern, config in backend_config._pattern_complex_format_to_config.items():
        observation_type = config.observation_type
        dtype_configs = config.dtype_configs
        num_tensor_args_to_observation_type = (
            config._num_tensor_args_to_observation_type
        )
        pattern_to_quantize_handlers[pattern] = _get_quantize_handler_cls(
            observation_type, dtype_configs, num_tensor_args_to_observation_type
        )
    return pattern_to_quantize_handlers


# TODO: remove this class, this is still exposed in torch.ao.quantization
# but we should be able to break bc
class BinaryOpQuantizeHandler(QuantizeHandler):
    pass


class CatQuantizeHandler(QuantizeHandler):
    pass


# TODO: remove this class
class ConvReluQuantizeHandler(QuantizeHandler):
    pass


# TODO: remove this class
class LinearReLUQuantizeHandler(QuantizeHandler):
    pass


# TODO: remove this class
class BatchNormQuantizeHandler(QuantizeHandler):
    pass


# TODO: remove this class
class EmbeddingQuantizeHandler(QuantizeHandler):
    pass


# TODO: remove this class
class RNNDynamicQuantizeHandler(QuantizeHandler):
    pass


# TODO: remove this class
class DefaultNodeQuantizeHandler(QuantizeHandler):
    """Common quantized op, first input and first output will be quantized"""


# TODO: remove this class
class FixedQParamsOpQuantizeHandler(QuantizeHandler):
    pass


# TODO: remove
class CopyNodeQuantizeHandler(QuantizeHandler):
    pass


# TODO: remove
class GeneralTensorShapeOpQuantizeHandler(QuantizeHandler):
    pass


# TODO: not used, can be removed after torch.ao.quantization namespace is deprecated
class CustomModuleQuantizeHandler(QuantizeHandler):
    pass


# TODO: not used, can be removed after torch.ao.quantization namespace is deprecated
class StandaloneModuleQuantizeHandler(QuantizeHandler):
    pass
<|endoftext|># mypy: allow-untyped-defs
import copy
from typing import Any

import torch
from torch.fx import GraphModule
from torch.fx.graph import Graph


__all__ = [
    "FusedGraphModule",
    "ObservedGraphModule",
    "ObservedStandaloneGraphModule",
    "QuantizedGraphModule",
]


class FusedGraphModule(GraphModule):
    def __init__(
        self,
        root: torch.nn.Module | dict[str, Any],
        graph: Graph,
        preserved_attr_names: set[str],
    ):
        self.preserved_attr_names = preserved_attr_names
        preserved_attrs = {
            attr: getattr(root, attr)
            for attr in self.preserved_attr_names
            if hasattr(root, attr)
        }
        super().__init__(root, graph)
        for attr in preserved_attrs:
            setattr(self, attr, preserved_attrs[attr])

    # GraphModule does not copy attributes which are not in the __dict__
    # of vanilla nn.Module.  So, we override __deepcopy__ in order
    # to copy the quantization specific attributes correctly.
    def __deepcopy__(self, memo):
        fake_mod = torch.nn.Module()
        fake_mod.__dict__ = copy.deepcopy(self.__dict__)
        return FusedGraphModule(
            fake_mod,
            copy.deepcopy(self.graph),
            copy.deepcopy(self.preserved_attr_names),
        )


class ObservedGraphModule(GraphModule):
    def __init__(
        self,
        root: torch.nn.Module | dict[str, Any],
        graph: Graph,
        preserved_attr_names: set[str],
    ):
        self.preserved_attr_names = {
            "_activation_post_process_map",
            "_activation_post_process_indexes",
            "_patterns",
            "_node_name_to_qconfig",
            "_prepare_custom_config",
            "_equalization_node_name_to_qconfig",
            "_node_name_to_scope",
            "_qconfig_mapping",
            "_is_qat",
            "_observed_node_names",
        }.union(preserved_attr_names)
        preserved_attrs = {
            attr: getattr(root, attr)
            for attr in self.preserved_attr_names
            if hasattr(root, attr)
        }
        super().__init__(root, graph)
        for attr in preserved_attrs:
            setattr(self, attr, preserved_attrs[attr])

    # GraphModule does not copy attributes which are not in the __dict__
    # of vanilla nn.Module.  So, we override __deepcopy__ in order
    # to copy the quantization specific attributes correctly.
    def __deepcopy__(self, memo):
        fake_mod = torch.nn.Module()
        fake_mod.__dict__ = copy.deepcopy(self.__dict__)
        return ObservedGraphModule(
            fake_mod,
            copy.deepcopy(self.graph),
            copy.deepcopy(self.preserved_attr_names),
        )


def _is_observed_module(module: Any) -> bool:
    return hasattr(module, "meta") and "_observed_graph_module_attrs" in module.meta


def _get_observed_graph_module_attr(
    model: torch.nn.Module | GraphModule, attr_name: str
) -> Any:
    if hasattr(model, "meta") and "_observed_graph_module_attrs" in model.meta:  # type: ignore[operator, index]
        return getattr(model.meta["_observed_graph_module_attrs"], attr_name)  # type: ignore[index]
    return None


class ObservedStandaloneGraphModule(ObservedGraphModule):
    def __init__(
        self,
        root: torch.nn.Module | dict[str, Any],
        graph: Graph,
        preserved_attr_names: set[str],
    ):
        preserved_attr_names = preserved_attr_names.union(
            {
                "_standalone_module_input_quantized_idxs",
                "_standalone_module_output_quantized_idxs",
            }
        )
        super().__init__(root, graph, preserved_attr_names)

    def __deepcopy__(self, memo):
        fake_mod = torch.nn.Module()
        fake_mod.__dict__ = copy.deepcopy(self.__dict__)
        return ObservedStandaloneGraphModule(
            fake_mod,
            copy.deepcopy(self.graph),
            copy.deepcopy(self.preserved_attr_names),
        )


def _is_observed_standalone_module(module: Any) -> bool:
    return (
        _is_observed_module(module)
        and module.meta["_observed_graph_module_attrs"].is_observed_standalone_module
    )


def _save_packed_weight(self, destination, prefix, keep_vars):
    for attr_name in dir(self):
        if "_packed_weight" in attr_name and isinstance(
            getattr(self, attr_name), torch._C.ScriptObject
        ):  # type: ignore[attr-defined]
            packed_weight = getattr(self, attr_name)
            destination[prefix + attr_name] = packed_weight


class QuantizedGraphModule(GraphModule):
    """This class is created to make sure PackedParams
    (e.g. LinearPackedParams, Conv2dPackedParams) to appear in state_dict
    so that we can serialize and deserialize quantized graph module with
    torch.save(m.state_dict()) and m.load_state_dict(state_dict)
    """

    def __init__(
        self,
        root: torch.nn.Module | dict[str, Any],
        graph: Graph,
        preserved_attr_names: set[str],
    ):
        self.preserved_attr_names = preserved_attr_names
        preserved_attrs = {
            attr: getattr(root, attr)
            for attr in self.preserved_attr_names
            if hasattr(root, attr)
        }
        super().__init__(root, graph)
        for attr in preserved_attrs:
            setattr(self, attr, preserved_attrs[attr])
        self._register_state_dict_hook(_save_packed_weight)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        attrs_to_pop = []
        for attr_name in state_dict:
            if attr_name.startswith("_packed_weight") and isinstance(
                state_dict[attr_name], torch._C.ScriptObject
            ):  # type: ignore[attr-defined]
                setattr(self, attr_name, state_dict[attr_name])
                attrs_to_pop.append(attr_name)

        # pop the packed param attributes
        for attr_name in attrs_to_pop:
            state_dict.pop(attr_name)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def __deepcopy__(self, memo):
        fake_mod = torch.nn.Module()
        fake_mod.__dict__ = copy.deepcopy(self.__dict__)
        return QuantizedGraphModule(
            fake_mod,
            copy.deepcopy(self.graph),
            copy.deepcopy(self.preserved_attr_names),
        )
<|endoftext|>import copy
import operator
from typing import Any, TYPE_CHECKING

import torch
from torch.ao.quantization import (
    default_weight_fake_quant,
    default_weight_observer,
    FakeQuantizeBase,
    QConfig,
    QConfigMapping,
)
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.observer import _PartialWrapper
from torch.ao.quantization.quantize_fx import convert_to_reference_fx, prepare_fx


if TYPE_CHECKING:
    from collections.abc import Callable


# TODO: move all LSTM util functions from fx/utils.py to this file
def _get_lstm_with_individually_observed_parts(
    float_lstm: torch.nn.LSTM,
    example_inputs: tuple[Any, ...],
    backend_config: BackendConfig | None = None,
    linear_output_obs_ctr: _PartialWrapper | None = None,
    sigmoid_obs_ctr: _PartialWrapper | None = None,
    tanh_obs_ctr: _PartialWrapper | None = None,
    cell_state_obs_ctr: _PartialWrapper | None = None,
    hidden_state_obs_ctr: _PartialWrapper | None = None,
    split_gates: bool = False,
) -> torch.ao.nn.quantizable.LSTM:
    """
    Return an observed `torch.ao.nn.quantizable.LSTM` created from a `torch.nn.LSTM`
    with specific observers or fake quantizes assigned to the inner ops or submodules.

    In both eager and FX graph mode quantization, `torch.ao.nn.quantizable.LSTM` is
    used as an observed custom module, which is responsible for inserting its own
    observers. By default, all inner ops inherit the parent custom module's QConfig.
    Users who wish to override this behavior may extend `torch.ao.nn.quantizable.LSTM`
    and use this helper function to customize the observer insertion logic.

    This is meant to be used to convert a float module to an observed module in the
    custom module flow.

    Args:
        `float_lstm`: The float LSTM module
        `example_inputs`: example inputs for the forward function of the LSTM module
        `backend_config`: BackendConfig to use to observe the LSTM module
        `linear_output_obs_ctr`: observer or fake quantize for linear outputs Wx + b,
            where W is the weight matrix, b is the bias, and x is either the inputs
            or the hidden state from the previous layer (if any)
        `sigmoid_obs_ctr`: observer or fake quantize for sigmoid activations
        `tanh_obs_ctr`: observer or fake quantize for tanh activations
        `cell_state_obs_ctr`: observer or fake quantize for the cell state
        `hidden_state_obs_ctr`: observer or fake quantize for the hidden state and
            the output

    Return:
        A `torch.ao.nn.quantizable.LSTM` with the specified observers or fake quantizes
        assigned to the inner ops.
    """

    def make_qconfig(obs_ctr: _PartialWrapper) -> QConfig:
        """
        Make a QConfig with fixed qparams observers or fake quantizes.
        """
        if isinstance(obs_ctr(), FakeQuantizeBase):
            weight = default_weight_fake_quant
        else:
            weight = default_weight_observer
        return QConfig(activation=obs_ctr, weight=weight)

    quantizable_lstm = torch.ao.nn.quantizable.LSTM(
        float_lstm.input_size,
        float_lstm.hidden_size,
        float_lstm.num_layers,
        float_lstm.bias,
        float_lstm.batch_first,
        float_lstm.dropout,
        float_lstm.bidirectional,
        split_gates=split_gates,
    )
    quantizable_lstm.qconfig = float_lstm.qconfig

    for idx in range(float_lstm.num_layers):
        quantizable_lstm.layers[idx] = (
            torch.ao.nn.quantizable.modules.rnn._LSTMLayer.from_float(
                float_lstm,
                idx,
                float_lstm.qconfig,
                batch_first=False,
                split_gates=split_gates,
            )
        )

    # Build QConfigMapping for the LSTM cell
    # Note: FloatFunctional qconfigs will be configured separately below
    cell_qm = QConfigMapping().set_global(float_lstm.qconfig)  # type: ignore[arg-type]
    if sigmoid_obs_ctr is not None:
        cell_qm.set_module_name("input_gate", make_qconfig(sigmoid_obs_ctr))
        cell_qm.set_module_name("forget_gate", make_qconfig(sigmoid_obs_ctr))
        cell_qm.set_module_name("output_gate", make_qconfig(sigmoid_obs_ctr))
    if tanh_obs_ctr is not None:
        cell_qm.set_module_name("cell_gate", make_qconfig(tanh_obs_ctr))

    # Insert observers into each LSTM cell
    # TODO: maybe make this work for layer_bw as well
    for layer in quantizable_lstm.layers:
        cell = layer.layer_fw.cell  # type: ignore[union-attr]
        if not isinstance(cell, torch.nn.Module):
            raise AssertionError("cell should be a nn.Module")
        cell = prepare_fx(cell, cell_qm, example_inputs, backend_config=backend_config)
        # HACK: Manually replace the activation_post_process following these ops.
        # This is needed for FloatFunctional ops because there is currently no way
        # to configure these ops in FX graph mode quantization today. This is because
        # the FloatFunctional modules simply disappear from the graph after tracing.
        # In the future, we should rewrite quantizable LSTM without FloatFunctionals.
        if not split_gates:
            op_index_to_activation_post_process_ctr = {
                (torch.add, 0): linear_output_obs_ctr,  # gates.add
                (torch.mul, 0): cell_state_obs_ctr,  # fgate_cx.mul
                (torch.mul, 1): cell_state_obs_ctr,  # igate_cgate.mul
                (torch.add, 1): cell_state_obs_ctr,  # fgate_cx_igate_cgate.add
                (torch.mul, 2): hidden_state_obs_ctr,  # ogate_cy.mul
            }
        else:
            op_index_to_activation_post_process_ctr = {
                (torch.add, 0): linear_output_obs_ctr,  # gates.add (input)
                (torch.add, 1): linear_output_obs_ctr,  # gates.add (forget)
                (torch.add, 2): linear_output_obs_ctr,  # gates.add (cell)
                (torch.add, 3): linear_output_obs_ctr,  # gates.add (output)
                (torch.mul, 0): cell_state_obs_ctr,  # fgate_cx.mul
                (torch.mul, 1): cell_state_obs_ctr,  # igate_cgate.mul
                (torch.add, 4): cell_state_obs_ctr,  # fgate_cx_igate_cgate.add
                (torch.mul, 2): hidden_state_obs_ctr,  # ogate_cy.mul
            }
        add_count = 0
        mul_count = 0
        for node in cell.graph.nodes:
            op_index: tuple[Callable, int] | None = None  # e.g. (torch.add, 1)
            if node.target is torch.add:
                op_index = (torch.add, add_count)
                add_count += 1
            elif node.target is torch.mul:
                op_index = (torch.mul, mul_count)
                mul_count += 1
            else:
                # Neither torch.add nor torch.mul
                continue
            if op_index not in op_index_to_activation_post_process_ctr:
                continue
            if len(node.users) != 1:
                raise AssertionError("expected exactly one user for the node")
            activation_post_process_name = next(iter(node.users.keys())).name
            activation_post_process_ctr = op_index_to_activation_post_process_ctr[
                op_index
            ]
            if activation_post_process_ctr is not None:
                setattr(
                    cell, activation_post_process_name, activation_post_process_ctr()
                )
        layer.layer_fw.cell = cell  # type: ignore[union-attr]
    return quantizable_lstm


def _get_reference_quantized_lstm_module(
    observed_lstm: torch.ao.nn.quantizable.LSTM,
    backend_config: BackendConfig | None = None,
) -> torch.ao.nn.quantized.LSTM:
    """
    Return a `torch.ao.nn.quantized.LSTM` created from a `torch.ao.nn.quantizable.LSTM`
    with observers or fake quantizes inserted through `prepare_fx`, e.g. from
    `_get_lstm_with_individually_observed_parts`.

    This is meant to be used to convert an observed module to a quantized module in the
    custom module flow.

    Args:
        `observed_lstm`: a `torch.ao.nn.quantizable.LSTM` observed through `prepare_fx`
        `backend_config`: BackendConfig to use to produce the reference quantized model

    Return:
        A reference `torch.ao.nn.quantized.LSTM` module.
    """
    quantized_lstm = torch.ao.nn.quantized.LSTM(
        observed_lstm.input_size,
        observed_lstm.hidden_size,
        observed_lstm.num_layers,
        observed_lstm.bias,
        observed_lstm.batch_first,
        observed_lstm.dropout,
        observed_lstm.bidirectional,
    )

    for i, layer in enumerate(quantized_lstm.layers):
        cell = copy.deepcopy(observed_lstm.layers.get_submodule(str(i)).layer_fw.cell)  # type: ignore[union-attr]
        cell = convert_to_reference_fx(cell, backend_config=backend_config)  # type: ignore[arg-type]
        if not isinstance(cell, torch.fx.GraphModule):
            raise AssertionError("cell must be converted to a torch.fx.GraphModule")
        # HACK: Manually remove input quantize nodes and output dequantize nodes,
        # since custom modules expect quint8 inputs and outputs for now. Note that
        # this functionality is supposedly handled through PrepareCustomConfig's
        # `set_input_quantized_indexes` and `set_output_quantized_indexes`, but that
        # API doesn't currently handle tuple inputs and outputs, so we have to do
        # this manually for now. In the future we should (1) relax the restriction
        # on custom module input/output dtypes, and (2) expand support for complex
        # input/output structures.
        for node in cell.graph.nodes:
            if node.target is torch.quantize_per_tensor:
                arg = node.args[0]
                # Remove quantize(x), quantize(hidden[0]), and quantize(hidden[1])
                if arg.target == "x" or (
                    arg.target is operator.getitem and arg.args[0].target == "hidden"
                ):
                    with cell.graph.inserting_before(node):
                        node.replace_all_uses_with(arg)
                        cell.graph.erase_node(node)
            if node.target == "output":
                # Remove all dequantize nodes in the output tuple
                for arg in node.args[0]:
                    with cell.graph.inserting_before(node):
                        node.replace_input_with(arg, arg.args[0])
        cell.graph.eliminate_dead_code()
        cell.recompile()
        layer.layer_fw.cell = cell  # type: ignore[union-attr]
    return quantized_lstm
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.quant_type import (
    _get_quant_type_to_str,
    _quant_type_from_str,
    QuantType,
)


__all__ = [
    "ConvertCustomConfig",
    "FuseCustomConfig",
    "PrepareCustomConfig",
    "StandaloneModuleConfigEntry",
]


# TODO: replace all usages with these constants
STANDALONE_MODULE_NAME_DICT_KEY = "standalone_module_name"
STANDALONE_MODULE_CLASS_DICT_KEY = "standalone_module_class"
FLOAT_TO_OBSERVED_DICT_KEY = "float_to_observed_custom_module_class"
OBSERVED_TO_QUANTIZED_DICT_KEY = "observed_to_quantized_custom_module_class"
NON_TRACEABLE_MODULE_NAME_DICT_KEY = "non_traceable_module_name"
NON_TRACEABLE_MODULE_CLASS_DICT_KEY = "non_traceable_module_class"
INPUT_QUANTIZED_INDEXES_DICT_KEY = "input_quantized_idxs"
OUTPUT_QUANTIZED_INDEXES_DICT_KEY = "output_quantized_idxs"
PRESERVED_ATTRIBUTES_DICT_KEY = "preserved_attributes"


@dataclass
class StandaloneModuleConfigEntry:
    # qconfig_mapping for the prepare function called in the submodule,
    # None means use qconfig from parent qconfig_mapping
    qconfig_mapping: QConfigMapping | None
    example_inputs: tuple[Any, ...]
    prepare_custom_config: PrepareCustomConfig | None
    backend_config: BackendConfig | None


class PrepareCustomConfig:
    """
    Custom configuration for :func:`~torch.ao.quantization.quantize_fx.prepare_fx` and
    :func:`~torch.ao.quantization.quantize_fx.prepare_qat_fx`.

    Example usage::

        prepare_custom_config = PrepareCustomConfig() \
            .set_standalone_module_name("module1", qconfig_mapping, example_inputs, \
                child_prepare_custom_config, backend_config) \
            .set_standalone_module_class(MyStandaloneModule, qconfig_mapping, example_inputs, \
                child_prepare_custom_config, backend_config) \
            .set_float_to_observed_mapping(FloatCustomModule, ObservedCustomModule) \
            .set_non_traceable_module_names(["module2", "module3"]) \
            .set_non_traceable_module_classes([NonTraceableModule1, NonTraceableModule2]) \
            .set_input_quantized_indexes([0]) \
            .set_output_quantized_indexes([0]) \
            .set_preserved_attributes(["attr1", "attr2"])
    """

    def __init__(self) -> None:
        self.standalone_module_names: dict[str, StandaloneModuleConfigEntry] = {}
        self.standalone_module_classes: dict[type, StandaloneModuleConfigEntry] = {}
        self.float_to_observed_mapping: dict[QuantType, dict[type, type]] = {}
        self.non_traceable_module_names: list[str] = []
        self.non_traceable_module_classes: list[type] = []
        self.input_quantized_indexes: list[int] = []
        self.output_quantized_indexes: list[int] = []
        self.preserved_attributes: list[str] = []

    def __repr__(self):
        dict_nonempty = {k: v for k, v in self.__dict__.items() if len(v) > 0}
        return f"PrepareCustomConfig({dict_nonempty})"

    def set_standalone_module_name(
        self,
        module_name: str,
        qconfig_mapping: QConfigMapping | None,
        example_inputs: tuple[Any, ...],
        prepare_custom_config: PrepareCustomConfig | None,
        backend_config: BackendConfig | None,
    ) -> PrepareCustomConfig:
        """
        Set the configuration for running a standalone module identified by ``module_name``.

        If ``qconfig_mapping`` is None, the parent ``qconfig_mapping`` will be used instead.
        If ``prepare_custom_config`` is None, an empty ``PrepareCustomConfig`` will be used.
        If ``backend_config`` is None, the parent ``backend_config`` will be used instead.
        """
        self.standalone_module_names[module_name] = StandaloneModuleConfigEntry(
            qconfig_mapping, example_inputs, prepare_custom_config, backend_config
        )
        return self

    def set_standalone_module_class(
        self,
        module_class: type,
        qconfig_mapping: QConfigMapping | None,
        example_inputs: tuple[Any, ...],
        prepare_custom_config: PrepareCustomConfig | None,
        backend_config: BackendConfig | None,
    ) -> PrepareCustomConfig:
        """
        Set the configuration for running a standalone module identified by ``module_class``.

        If ``qconfig_mapping`` is None, the parent ``qconfig_mapping`` will be used instead.
        If ``prepare_custom_config`` is None, an empty ``PrepareCustomConfig`` will be used.
        If ``backend_config`` is None, the parent ``backend_config`` will be used instead.
        """
        self.standalone_module_classes[module_class] = StandaloneModuleConfigEntry(
            qconfig_mapping, example_inputs, prepare_custom_config, backend_config
        )
        return self

    def set_float_to_observed_mapping(
        self,
        float_class: type,
        observed_class: type,
        quant_type: QuantType = QuantType.STATIC,
    ) -> PrepareCustomConfig:
        """
        Set the mapping from a custom float module class to a custom observed module class.

        The observed module class must have a ``from_float`` class method that converts the float module class
        to the observed module class. This is currently only supported for static quantization.
        """
        if quant_type != QuantType.STATIC:
            raise ValueError(
                "set_float_to_observed_mapping is currently only supported for static quantization"
            )
        if quant_type not in self.float_to_observed_mapping:
            self.float_to_observed_mapping[quant_type] = {}
        self.float_to_observed_mapping[quant_type][float_class] = observed_class
        return self

    def set_non_traceable_module_names(
        self, module_names: list[str]
    ) -> PrepareCustomConfig:
        """
        Set the modules that are not symbolically traceable, identified by name.
        """
        self.non_traceable_module_names = module_names
        return self

    def set_non_traceable_module_classes(
        self, module_classes: list[type]
    ) -> PrepareCustomConfig:
        """
        Set the modules that are not symbolically traceable, identified by class.
        """
        self.non_traceable_module_classes = module_classes
        return self

    def set_input_quantized_indexes(self, indexes: list[int]) -> PrepareCustomConfig:
        """
        Set the indexes of the inputs of the graph that should be quantized.
        Inputs are otherwise assumed to be in fp32 by default instead.
        """
        self.input_quantized_indexes = indexes
        return self

    def set_output_quantized_indexes(self, indexes: list[int]) -> PrepareCustomConfig:
        """
        Set the indexes of the outputs of the graph that should be quantized.
        Outputs are otherwise assumed to be in fp32 by default instead.
        """
        self.output_quantized_indexes = indexes
        return self

    def set_preserved_attributes(self, attributes: list[str]) -> PrepareCustomConfig:
        """
        Set the names of the attributes that will persist in the graph module even if they are not used in
        the model's ``forward`` method.
        """
        self.preserved_attributes = attributes
        return self

    # TODO: remove this
    @classmethod
    def from_dict(
        cls, prepare_custom_config_dict: dict[str, Any]
    ) -> PrepareCustomConfig:
        """
        Create a ``PrepareCustomConfig`` from a dictionary with the following items:

            "standalone_module_name": a list of (module_name, qconfig_mapping, example_inputs,
            child_prepare_custom_config, backend_config) tuples

            "standalone_module_class" a list of (module_class, qconfig_mapping, example_inputs,
            child_prepare_custom_config, backend_config) tuples

            "float_to_observed_custom_module_class": a nested dictionary mapping from quantization
            mode to an inner mapping from float module classes to observed module classes, e.g.
            {"static": {FloatCustomModule: ObservedCustomModule}}

            "non_traceable_module_name": a list of modules names that are not symbolically traceable
            "non_traceable_module_class": a list of module classes that are not symbolically traceable
            "input_quantized_idxs": a list of indexes of graph inputs that should be quantized
            "output_quantized_idxs": a list of indexes of graph outputs that should be quantized
            "preserved_attributes": a list of attributes that persist even if they are not used in ``forward``

        This function is primarily for backward compatibility and may be removed in the future.
        """

        def _get_qconfig_mapping(obj: Any, dict_key: str) -> QConfigMapping | None:
            """
            Convert the given object into a QConfigMapping if possible, else throw an exception.
            """
            if isinstance(obj, QConfigMapping) or obj is None:
                return obj
            if isinstance(obj, dict):
                return QConfigMapping.from_dict(obj)
            raise ValueError(
                f"Expected QConfigMapping in prepare_custom_config_dict[\"{dict_key}\"], got '{type(obj)}'"
            )

        def _get_prepare_custom_config(
            obj: Any, dict_key: str
        ) -> PrepareCustomConfig | None:
            """
            Convert the given object into a PrepareCustomConfig if possible, else throw an exception.
            """
            if isinstance(obj, PrepareCustomConfig) or obj is None:
                return obj
            if isinstance(obj, dict):
                return PrepareCustomConfig.from_dict(obj)
            raise ValueError(
                f"Expected PrepareCustomConfig in prepare_custom_config_dict[\"{dict_key}\"], got '{type(obj)}'"
            )

        def _get_backend_config(obj: Any, dict_key: str) -> BackendConfig | None:
            """
            Convert the given object into a BackendConfig if possible, else throw an exception.
            """
            if isinstance(obj, BackendConfig) or obj is None:
                return obj
            if isinstance(obj, dict):
                return BackendConfig.from_dict(obj)
            raise ValueError(
                f"Expected BackendConfig in prepare_custom_config_dict[\"{dict_key}\"], got '{type(obj)}'"
            )

        conf = cls()
        for (
            module_name,
            qconfig_dict,
            example_inputs,
            _prepare_custom_config_dict,
            backend_config_dict,
        ) in prepare_custom_config_dict.get(STANDALONE_MODULE_NAME_DICT_KEY, []):
            qconfig_mapping = _get_qconfig_mapping(
                qconfig_dict, STANDALONE_MODULE_NAME_DICT_KEY
            )
            prepare_custom_config = _get_prepare_custom_config(
                _prepare_custom_config_dict, STANDALONE_MODULE_NAME_DICT_KEY
            )
            backend_config = _get_backend_config(
                backend_config_dict, STANDALONE_MODULE_NAME_DICT_KEY
            )
            conf.set_standalone_module_name(
                module_name,
                qconfig_mapping,
                example_inputs,
                prepare_custom_config,
                backend_config,
            )
        for (
            module_class,
            qconfig_dict,
            example_inputs,
            _prepare_custom_config_dict,
            backend_config_dict,
        ) in prepare_custom_config_dict.get(STANDALONE_MODULE_CLASS_DICT_KEY, []):
            qconfig_mapping = _get_qconfig_mapping(
                qconfig_dict, STANDALONE_MODULE_CLASS_DICT_KEY
            )
            prepare_custom_config = _get_prepare_custom_config(
                _prepare_custom_config_dict, STANDALONE_MODULE_CLASS_DICT_KEY
            )
            backend_config = _get_backend_config(
                backend_config_dict, STANDALONE_MODULE_CLASS_DICT_KEY
            )
            conf.set_standalone_module_class(
                module_class,
                qconfig_mapping,
                example_inputs,
                prepare_custom_config,
                backend_config,
            )
        for quant_type_name, custom_module_mapping in prepare_custom_config_dict.get(
            FLOAT_TO_OBSERVED_DICT_KEY, {}
        ).items():
            quant_type = _quant_type_from_str(quant_type_name)
            for float_class, observed_class in custom_module_mapping.items():
                conf.set_float_to_observed_mapping(
                    float_class, observed_class, quant_type
                )
        conf.set_non_traceable_module_names(
            prepare_custom_config_dict.get(NON_TRACEABLE_MODULE_NAME_DICT_KEY, [])
        )
        conf.set_non_traceable_module_classes(
            prepare_custom_config_dict.get(NON_TRACEABLE_MODULE_CLASS_DICT_KEY, [])
        )
        conf.set_input_quantized_indexes(
            prepare_custom_config_dict.get(INPUT_QUANTIZED_INDEXES_DICT_KEY, [])
        )
        conf.set_output_quantized_indexes(
            prepare_custom_config_dict.get(OUTPUT_QUANTIZED_INDEXES_DICT_KEY, [])
        )
        conf.set_preserved_attributes(
            prepare_custom_config_dict.get(PRESERVED_ATTRIBUTES_DICT_KEY, [])
        )
        return conf

    def to_dict(self) -> dict[str, Any]:
        """
        Convert this ``PrepareCustomConfig`` to a dictionary with the items described in
        :func:`~torch.ao.quantization.fx.custom_config.PrepareCustomConfig.from_dict`.
        """

        def _make_tuple(key: Any, e: StandaloneModuleConfigEntry):
            qconfig_dict = e.qconfig_mapping.to_dict() if e.qconfig_mapping else None
            prepare_custom_config_dict = (
                e.prepare_custom_config.to_dict() if e.prepare_custom_config else None
            )
            return (
                key,
                qconfig_dict,
                e.example_inputs,
                prepare_custom_config_dict,
                e.backend_config,
            )

        d: dict[str, Any] = {}
        for module_name, sm_config_entry in self.standalone_module_names.items():
            if STANDALONE_MODULE_NAME_DICT_KEY not in d:
                d[STANDALONE_MODULE_NAME_DICT_KEY] = []
            d[STANDALONE_MODULE_NAME_DICT_KEY].append(
                _make_tuple(module_name, sm_config_entry)
            )
        for module_class, sm_config_entry in self.standalone_module_classes.items():
            if STANDALONE_MODULE_CLASS_DICT_KEY not in d:
                d[STANDALONE_MODULE_CLASS_DICT_KEY] = []
            d[STANDALONE_MODULE_CLASS_DICT_KEY].append(
                _make_tuple(module_class, sm_config_entry)
            )
        for (
            quant_type,
            float_to_observed_mapping,
        ) in self.float_to_observed_mapping.items():
            if FLOAT_TO_OBSERVED_DICT_KEY not in d:
                d[FLOAT_TO_OBSERVED_DICT_KEY] = {}
            d[FLOAT_TO_OBSERVED_DICT_KEY][_get_quant_type_to_str(quant_type)] = (
                float_to_observed_mapping
            )
        if len(self.non_traceable_module_names) > 0:
            d[NON_TRACEABLE_MODULE_NAME_DICT_KEY] = self.non_traceable_module_names
        if len(self.non_traceable_module_classes) > 0:
            d[NON_TRACEABLE_MODULE_CLASS_DICT_KEY] = self.non_traceable_module_classes
        if len(self.input_quantized_indexes) > 0:
            d[INPUT_QUANTIZED_INDEXES_DICT_KEY] = self.input_quantized_indexes
        if len(self.output_quantized_indexes) > 0:
            d[OUTPUT_QUANTIZED_INDEXES_DICT_KEY] = self.output_quantized_indexes
        if len(self.preserved_attributes) > 0:
            d[PRESERVED_ATTRIBUTES_DICT_KEY] = self.preserved_attributes
        return d


class ConvertCustomConfig:
    """
    Custom configuration for :func:`~torch.ao.quantization.quantize_fx.convert_fx`.

    Example usage::

        convert_custom_config = ConvertCustomConfig() \
            .set_observed_to_quantized_mapping(ObservedCustomModule, QuantizedCustomModule) \
            .set_preserved_attributes(["attr1", "attr2"])
    """

    def __init__(self) -> None:
        self.observed_to_quantized_mapping: dict[QuantType, dict[type, type]] = {}
        self.preserved_attributes: list[str] = []

    def __repr__(self):
        dict_nonempty = {k: v for k, v in self.__dict__.items() if len(v) > 0}
        return f"ConvertCustomConfig({dict_nonempty})"

    def set_observed_to_quantized_mapping(
        self,
        observed_class: type,
        quantized_class: type,
        quant_type: QuantType = QuantType.STATIC,
    ) -> ConvertCustomConfig:
        """
        Set the mapping from a custom observed module class to a custom quantized module class.

        The quantized module class must have a ``from_observed`` class method that converts the observed module class
        to the quantized module class.
        """
        if quant_type not in self.observed_to_quantized_mapping:
            self.observed_to_quantized_mapping[quant_type] = {}
        self.observed_to_quantized_mapping[quant_type][observed_class] = quantized_class
        return self

    def set_preserved_attributes(self, attributes: list[str]) -> ConvertCustomConfig:
        """
        Set the names of the attributes that will persist in the graph module even if they are not used in
        the model's ``forward`` method.
        """
        self.preserved_attributes = attributes
        return self

    # TODO: remove this
    @classmethod
    def from_dict(
        cls, convert_custom_config_dict: dict[str, Any]
    ) -> ConvertCustomConfig:
        """
        Create a ``ConvertCustomConfig`` from a dictionary with the following items:

            "observed_to_quantized_custom_module_class": a nested dictionary mapping from quantization
            mode to an inner mapping from observed module classes to quantized module classes, e.g.::
            {
            "static": {FloatCustomModule: ObservedCustomModule},
            "dynamic": {FloatCustomModule: ObservedCustomModule},
            "weight_only": {FloatCustomModule: ObservedCustomModule}
            }
            "preserved_attributes": a list of attributes that persist even if they are not used in ``forward``

        This function is primarily for backward compatibility and may be removed in the future.
        """
        conf = cls()
        for quant_type_name, custom_module_mapping in convert_custom_config_dict.get(
            OBSERVED_TO_QUANTIZED_DICT_KEY, {}
        ).items():
            quant_type = _quant_type_from_str(quant_type_name)
            for observed_class, quantized_class in custom_module_mapping.items():
                conf.set_observed_to_quantized_mapping(
                    observed_class, quantized_class, quant_type
                )
        conf.set_preserved_attributes(
            convert_custom_config_dict.get(PRESERVED_ATTRIBUTES_DICT_KEY, [])
        )
        return conf

    def to_dict(self) -> dict[str, Any]:
        """
        Convert this ``ConvertCustomConfig`` to a dictionary with the items described in
        :func:`~torch.ao.quantization.fx.custom_config.ConvertCustomConfig.from_dict`.
        """
        d: dict[str, Any] = {}
        for (
            quant_type,
            observed_to_quantized_mapping,
        ) in self.observed_to_quantized_mapping.items():
            if OBSERVED_TO_QUANTIZED_DICT_KEY not in d:
                d[OBSERVED_TO_QUANTIZED_DICT_KEY] = {}
            d[OBSERVED_TO_QUANTIZED_DICT_KEY][_get_quant_type_to_str(quant_type)] = (
                observed_to_quantized_mapping
            )
        if len(self.preserved_attributes) > 0:
            d[PRESERVED_ATTRIBUTES_DICT_KEY] = self.preserved_attributes
        return d


class FuseCustomConfig:
    """
    Custom configuration for :func:`~torch.ao.quantization.quantize_fx.fuse_fx`.

    Example usage::

        fuse_custom_config = FuseCustomConfig().set_preserved_attributes(
            ["attr1", "attr2"]
        )
    """

    def __init__(self) -> None:
        self.preserved_attributes: list[str] = []

    def __repr__(self):
        dict_nonempty = {k: v for k, v in self.__dict__.items() if len(v) > 0}
        return f"FuseCustomConfig({dict_nonempty})"

    def set_preserved_attributes(self, attributes: list[str]) -> FuseCustomConfig:
        """
        Set the names of the attributes that will persist in the graph module even if they are not used in
        the model's ``forward`` method.
        """
        self.preserved_attributes = attributes
        return self

    # TODO: remove this
    @classmethod
    def from_dict(cls, fuse_custom_config_dict: dict[str, Any]) -> FuseCustomConfig:
        """
        Create a ``ConvertCustomConfig`` from a dictionary with the following items:

            "preserved_attributes": a list of attributes that persist even if they are not used in ``forward``

        This function is primarily for backward compatibility and may be removed in the future.
        """
        conf = cls()
        conf.set_preserved_attributes(
            fuse_custom_config_dict.get(PRESERVED_ATTRIBUTES_DICT_KEY, [])
        )
        return conf

    def to_dict(self) -> dict[str, Any]:
        """
        Convert this ``FuseCustomConfig`` to a dictionary with the items described in
        :func:`~torch.ao.quantization.fx.custom_config.ConvertCustomConfig.from_dict`.
        """
        d: dict[str, Any] = {}
        if len(self.preserved_attributes) > 0:
            d[PRESERVED_ATTRIBUTES_DICT_KEY] = self.preserved_attributes
        return d
<|endoftext|># mypy: allow-untyped-defs
from collections import OrderedDict, OrderedDict as OrdDict
from typing import Any

import torch


# try to import tablate
got_tabulate = True
try:
    from tabulate import tabulate
except ImportError:
    got_tabulate = False


# var to see if we could import matplotlib
got_matplotlib = True
try:
    import matplotlib.pyplot as plt
except ImportError:
    got_matplotlib = False


class ModelReportVisualizer:
    r"""
    The ModelReportVisualizer class aims to provide users a way to visualize some of the statistics
    that were generated by the ModelReport API. However, at a higher level, the class aims to provide
    some level of visualization of statistics to PyTorch in order to make it easier to parse data and
    diagnose any potential issues with data or a specific model. With respect to the visualizations,
    the ModelReportVisualizer class currently supports several methods of visualizing data.

    Supported Visualization Methods Include:
    - Table format
    - Plot format (line graph)
    - Histogram format

    For all of the existing visualization methods, there is the option to filter data based on:
    - A module fqn prefix
    - Feature [required for the plot and histogram]

    * :attr:`generated_reports` The reports generated by the ModelReport class in the structure below
        Ensure that features that are the same across different report contain the same name
        Ensure that objects representing the same features are the same type / dimension (where applicable)

    Note:
        Currently, the ModelReportVisualizer class supports visualization of data generated by the
        ModelReport class. However, this structure is extensible and should allow the visualization of
        other information as long as the information is structured in the following general format:

        Report Structure
        -- module_fqn [module with attached detectors]
            |
            -- feature keys [not every detector extracts same information]
                                    [same collected info has same keys, unless can be specific to detector]


    The goal behind the class is that the generated visualizations can be used in conjunction with the generated
    report for people to get a better understanding of issues and what the fix might be. It is also just to provide
    a good visualization platform, since it might be hard to parse through the ModelReport returned dictionary as
    that grows in size.

    General Use Flow Expected
    1.) Initialize ModelReport object with reports of interest by passing in initialized detector objects
    2.) Prepare your model with prepare_fx
    3.) Call model_report.prepare_detailed_calibration on your model to add relevant observers
    4.) Calibrate your model with data
    5.) Call model_report.generate_report on your model to generate report and optionally remove added observers
    6.) Use output of model_report.generate_report to initialize ModelReportVisualizer instance
    7.) Use instance to view different views of data as desired, applying filters as needed
        8.) Either see the super detailed information or just the actual printed or shown table / plot / histogram

    """

    # keys for table dict
    TABLE_TENSOR_KEY = "tensor_level_info"
    TABLE_CHANNEL_KEY = "channel_level_info"

    # Constants for header vals
    NUM_NON_FEATURE_TENSOR_HEADERS = 2
    NUM_NON_FEATURE_CHANNEL_HEADERS = 3

    # Constants for row index in header
    CHANNEL_NUM_INDEX = 2

    def __init__(self, generated_reports: OrderedDict[str, Any]):
        r"""
        Initializes the ModelReportVisualizer instance with the necessary reports.

        Args:
            generated_reports (Dict[str, Any]): The reports generated by the ModelReport class
                can also be a dictionary generated in another manner, as long as format is same
        """
        self.generated_reports = generated_reports

    def get_all_unique_module_fqns(self) -> set[str]:
        r"""
        The purpose of this method is to provide a user the set of all module_fqns so that if
        they wish to use some of the filtering capabilities of the ModelReportVisualizer class,
        they don't need to manually parse the generated_reports dictionary to get this information.

        Returns all the unique module fqns present in the reports the ModelReportVisualizer
        instance was initialized with.
        """
        # returns the keys of the ordered dict
        return set(self.generated_reports.keys())

    def get_all_unique_feature_names(
        self, plottable_features_only: bool = True
    ) -> set[str]:
        r"""
        The purpose of this method is to provide a user the set of all feature names so that if
        they wish to use the filtering capabilities of the generate_table_view(), or use either of
        the generate_plot_view() or generate_histogram_view(), they don't need to manually parse
        the generated_reports dictionary to get this information.

        Args:
            plottable_features_only (bool): True if the user is only looking for plottable features,
                False otherwise
                plottable features are those that are tensor values
                Default: True (only return those feature names that are plottable)

        Returns all the unique module fqns present in the reports the ModelReportVisualizer
        instance was initialized with.
        """
        unique_feature_names = set()
        for module_fqn in self.generated_reports:
            # get dict of the features
            feature_dict: dict[str, Any] = self.generated_reports[module_fqn]

            # loop through features
            for feature_name in feature_dict:
                # if we need plottable, ensure type of val is tensor
                if (
                    not plottable_features_only
                    or type(feature_dict[feature_name]) is torch.Tensor
                ):
                    unique_feature_names.add(feature_name)

        # return our compiled set of unique feature names
        return unique_feature_names

    def _get_filtered_data(
        self, feature_filter: str, module_fqn_filter: str
    ) -> OrderedDict[str, Any]:
        r"""
        Filters the data and returns it in the same ordered dictionary format so the relevant views can be displayed.

        Args:
            feature_filter (str): The feature filter, if we want to filter the set of data to only include
                a certain set of features that include feature_filter
                If feature = "", then we do not filter based on any features
            module_fqn_filter (str): The filter on prefix for the module fqn. All modules that have fqn with
                this prefix will be included
                If module_fqn_filter = "" we do not filter based on module fqn, and include all modules

        First, the data is filtered based on module_fqn, and then filtered based on feature
        Returns an OrderedDict (sorted in order of model) mapping:
            module_fqns -> feature_names -> values
        """
        # create return dict
        filtered_dict: OrderedDict[str, Any] = OrdDict()

        for module_fqn in self.generated_reports:
            # first filter based on module
            if module_fqn_filter == "" or module_fqn_filter in module_fqn:
                # create entry for module and loop through features
                filtered_dict[module_fqn] = {}
                module_reports = self.generated_reports[module_fqn]
                for feature_name in module_reports:
                    # check if filtering on features and do so if desired
                    if feature_filter == "" or feature_filter in feature_name:
                        filtered_dict[module_fqn][feature_name] = module_reports[
                            feature_name
                        ]

        # we have populated the filtered dict, and must return it

        return filtered_dict

    def _generate_tensor_table(
        self,
        filtered_data: OrderedDict[str, dict[str, Any]],
        tensor_features: list[str],
    ) -> tuple[list, list]:
        r"""
        Takes in the filtered data and features list and generates the tensor headers and table

        Currently meant to generate the headers and table for both the tensor information.

        Args:
            filtered_data (OrderedDict[str, Dict[str, Any]]): An OrderedDict (sorted in order of model) mapping:
                module_fqns -> feature_names -> values
            tensor_features (List[str]): A list of the tensor level features

        Returns a tuple with:
            A list of the headers of the tensor table
            A list of lists containing the table information row by row
            The 0th index row will contain the headers of the columns
            The rest of the rows will contain data
        """
        # now we compose the tensor information table
        tensor_table: list[list[Any]] = []
        tensor_headers: list[str] = []

        # append the table row to the table only if we have features
        if len(tensor_features) > 0:
            # now we add all the data
            for index, module_fqn in enumerate(filtered_data):
                # we make a new row for the tensor table
                tensor_table_row = [index, module_fqn]
                for feature in tensor_features:
                    # we iterate in same order of added features

                    if feature in filtered_data[module_fqn]:
                        # add value if applicable to module
                        feature_val = filtered_data[module_fqn][feature]
                    else:
                        # add that it is not applicable
                        feature_val = "Not Applicable"

                    # if it's a tensor we want to extract val
                    if isinstance(feature_val, torch.Tensor):
                        feature_val = feature_val.item()

                    # we add to our list of values
                    # pyrefly: ignore [bad-argument-type]
                    tensor_table_row.append(feature_val)

                tensor_table.append(tensor_table_row)

        # add row of headers of we actually have something, otherwise just empty
        if len(tensor_table) != 0:
            tensor_headers = ["idx", "layer_fqn"] + tensor_features

        return (tensor_headers, tensor_table)

    def _generate_channels_table(
        self,
        filtered_data: OrderedDict[str, Any],
        channel_features: list[str],
        num_channels: int,
    ) -> tuple[list, list]:
        r"""
        Takes in the filtered data and features list and generates the channels headers and table

        Currently meant to generate the headers and table for both the channels information.

        Args:
            filtered_data (OrderedDict[str, Any]): An OrderedDict (sorted in order of model) mapping:
                module_fqns -> feature_names -> values
            channel_features (List[str]): A list of the channel level features
            num_channels (int): Number of channels in the channel data

        Returns a tuple with:
            A list of the headers of the channel table
            A list of lists containing the table information row by row
            The 0th index row will contain the headers of the columns
            The rest of the rows will contain data
        """
        # now we compose the table for the channel information table
        channel_table: list[list[Any]] = []
        channel_headers: list[str] = []

        # counter to keep track of number of entries in
        channel_table_entry_counter: int = 0

        if len(channel_features) > 0:
            # now we add all channel data
            for module_fqn in filtered_data:
                # we iterate over all channels
                for channel in range(num_channels):
                    # we make a new row for the channel
                    new_channel_row = [channel_table_entry_counter, module_fqn, channel]
                    for feature in channel_features:
                        if feature in filtered_data[module_fqn]:
                            # add value if applicable to module
                            feature_val = filtered_data[module_fqn][feature][channel]
                        else:
                            # add that it is not applicable
                            feature_val = "Not Applicable"

                        # if it's a tensor we want to extract val
                        if type(feature_val) is torch.Tensor:
                            feature_val = feature_val.item()

                        # add value to channel specific row
                        # pyrefly: ignore [bad-argument-type]
                        new_channel_row.append(feature_val)

                    # add to table and increment row index counter
                    channel_table.append(new_channel_row)
                    channel_table_entry_counter += 1

        # add row of headers of we actually have something, otherwise just empty
        if len(channel_table) != 0:
            channel_headers = ["idx", "layer_fqn", "channel"] + channel_features

        return (channel_headers, channel_table)

    def generate_filtered_tables(
        self, feature_filter: str = "", module_fqn_filter: str = ""
    ) -> dict[str, tuple[list, list]]:
        r"""
        Takes in optional filter values and generates two tables with desired information.

        The generated tables are presented in both a list-of-lists format

        The reason for the two tables are that they handle different things:
        1.) the first table handles all tensor level information
        2.) the second table handles and displays all channel based information

        The reasoning for this is that having all the info in one table can make it ambiguous which collected
            statistics are global, and which are actually per-channel, so it's better to split it up into two
            tables. This also makes the information much easier to digest given the plethora of statistics collected

        Tensor table columns:
            idx  layer_fqn  feature_1   feature_2   feature_3   .... feature_n
            ----  ---------  ---------   ---------   ---------        ---------

        Per-Channel table columns:
            idx  layer_fqn  channel  feature_1   feature_2   feature_3   .... feature_n
            ----  ---------  -------  ---------   ---------   ---------        ---------

        Args:
            feature_filter (str, optional): Filters the features presented to only those that
                contain this filter substring
                Default = "", results in all the features being printed
            module_fqn_filter (str, optional): Only includes modules that contains this string
                Default = "", results in all the modules in the reports to be visible in the table

        Returns a dictionary with two keys:
            (Dict[str, Tuple[List, List]]) A dict containing two keys:
            "tensor_level_info", "channel_level_info"
                Each key maps to a tuple with:
                    A list of the headers of each table
                    A list of lists containing the table information row by row
                    The 0th index row will contain the headers of the columns
                    The rest of the rows will contain data

        Example Use:
            >>> # xdoctest: +SKIP("undefined variables")
            >>> mod_report_visualizer.generate_filtered_tables(
            ...     feature_filter="per_channel_min", module_fqn_filter="block1"
            ... )  # generates table with per_channel_min info for all modules in block 1 of the model
        """
        # first get the filtered data
        filtered_data: OrderedDict[str, Any] = self._get_filtered_data(
            feature_filter, module_fqn_filter
        )

        # now we split into tensor and per-channel data
        tensor_features: set[str] = set()
        channel_features: set[str] = set()

        # keep track of the number of channels we have
        num_channels: int = 0

        for module_fqn in filtered_data:
            for feature_name in filtered_data[module_fqn]:
                # get the data for that specific feature
                feature_data = filtered_data[module_fqn][feature_name]

                # check if not zero dim tensor
                is_tensor: bool = isinstance(feature_data, torch.Tensor)
                is_not_zero_dim: bool = is_tensor and len(feature_data.shape) != 0

                if is_not_zero_dim or isinstance(feature_data, list):
                    # works means per channel
                    channel_features.add(feature_name)
                    num_channels = len(feature_data)
                else:
                    # means is per-tensor
                    tensor_features.add(feature_name)

        # we make them lists for iteration purposes
        tensor_features_list: list[str] = sorted(tensor_features)
        channel_features_list: list[str] = sorted(channel_features)

        # get the tensor info
        tensor_headers, tensor_table = self._generate_tensor_table(
            filtered_data, tensor_features_list
        )

        # get the channel info
        channel_headers, channel_table = self._generate_channels_table(
            filtered_data, channel_features_list, num_channels
        )

        # let's now create the dictionary to return
        table_dict = {
            self.TABLE_TENSOR_KEY: (tensor_headers, tensor_table),
            self.TABLE_CHANNEL_KEY: (channel_headers, channel_table),
        }

        # return the two tables
        return table_dict

    def generate_table_visualization(
        self, feature_filter: str = "", module_fqn_filter: str = ""
    ):
        r"""
        Takes in optional filter values and prints out formatted tables of the information.

        The reason for the two tables printed out instead of one large one are that they handle different things:
        1.) the first table handles all tensor level information
        2.) the second table handles and displays all channel based information

        The reasoning for this is that having all the info in one table can make it ambiguous which collected
            statistics are global, and which are actually per-channel, so it's better to split it up into two
            tables. This also makes the information much easier to digest given the plethora of statistics collected

        Tensor table columns:
         idx  layer_fqn  feature_1   feature_2   feature_3   .... feature_n
        ----  ---------  ---------   ---------   ---------        ---------

        Per-Channel table columns:

         idx  layer_fqn  channel  feature_1   feature_2   feature_3   .... feature_n
        ----  ---------  -------  ---------   ---------   ---------        ---------

        Args:
            feature_filter (str, optional): Filters the features presented to only those that
                contain this filter substring
                Default = "", results in all the features being printed
            module_fqn_filter (str, optional): Only includes modules that contains this string
                Default = "", results in all the modules in the reports to be visible in the table

        Example Use:
            >>> # xdoctest: +SKIP("undefined variables")
            >>> mod_report_visualizer.generate_table_visualization(
            ...     feature_filter="per_channel_min", module_fqn_filter="block1"
            ... )
            >>> # prints out neatly formatted table with per_channel_min info
            >>> # for all modules in block 1 of the model
        """
        # see if we got tabulate
        if not got_tabulate:
            print("Make sure to install tabulate and try again.")
            return None

        # get the table dict and the specific tables of interest
        table_dict = self.generate_filtered_tables(feature_filter, module_fqn_filter)
        tensor_headers, tensor_table = table_dict[self.TABLE_TENSOR_KEY]
        channel_headers, channel_table = table_dict[self.TABLE_CHANNEL_KEY]

        # get the table string and print it out
        # now we have populated the tables for each one
        # let's create the strings to be returned
        table_str = ""
        # the tables will have some headers columns that are non-feature
        # ex. table index, module name, channel index, etc.
        # we want to look at header columns for features, that come after those headers
        if len(tensor_headers) > self.NUM_NON_FEATURE_TENSOR_HEADERS:
            # if we have at least one tensor level feature to be added we add tensor table
            table_str += "Tensor Level Information \n"
            table_str += tabulate(tensor_table, headers=tensor_headers)
        if len(channel_headers) > self.NUM_NON_FEATURE_CHANNEL_HEADERS:
            # if we have at least one channel level feature to be added we add tensor table
            table_str += "\n\n Channel Level Information \n"
            table_str += tabulate(channel_table, headers=channel_headers)

        # if no features at all, let user know
        if table_str == "":
            table_str = "No data points to generate table with."

        print(table_str)

    def _get_plottable_data(
        self, feature_filter: str, module_fqn_filter: str
    ) -> tuple[list, list[list], bool]:
        r"""
        Takes in the feature filters and module filters and outputs the x and y data for plotting

        Args:
            feature_filter (str): Filters the features presented to only those that
                contain this filter substring
            module_fqn_filter (str): Only includes modules that contains this string

        Returns a tuple of three elements
            The first is a list containing relevant x-axis data
            The second is a list containing the corresponding y-axis data
            If the data is per channel
        """
        # get the table dict and the specific tables of interest
        table_dict = self.generate_filtered_tables(feature_filter, module_fqn_filter)
        tensor_headers, tensor_table = table_dict[self.TABLE_TENSOR_KEY]
        channel_headers, channel_table = table_dict[self.TABLE_CHANNEL_KEY]

        # make sure it is only 1 feature that is being plotted
        # get the number of features in each of these
        tensor_info_features_count = (
            len(tensor_headers) - ModelReportVisualizer.NUM_NON_FEATURE_TENSOR_HEADERS
        )
        channel_info_features_count = (
            len(channel_headers) - ModelReportVisualizer.NUM_NON_FEATURE_CHANNEL_HEADERS
        )

        # see if valid tensor or channel plot
        is_valid_per_tensor_plot: bool = tensor_info_features_count == 1
        is_valid_per_channel_plot: bool = channel_info_features_count == 1

        # offset should either be one of tensor or channel table or neither
        feature_column_offset = ModelReportVisualizer.NUM_NON_FEATURE_TENSOR_HEADERS
        table = tensor_table

        # if a per_channel plot, we have different offset and table
        if is_valid_per_channel_plot:
            feature_column_offset = (
                ModelReportVisualizer.NUM_NON_FEATURE_CHANNEL_HEADERS
            )
            table = channel_table

        x_data: list = []
        y_data: list[list] = []
        # the feature will either be a tensor feature or channel feature
        if is_valid_per_tensor_plot:
            for table_row_num, row in enumerate(table):
                # get x_value to append
                x_val_to_append = table_row_num
                # the index of the feature will the 0 + num non feature columns
                tensor_feature_index = feature_column_offset
                row_value = row[tensor_feature_index]
                if type(row_value) is not str:
                    x_data.append(x_val_to_append)
                    y_data.append(row_value)
        elif is_valid_per_channel_plot:
            # gather the x_data and multiple y_data
            # calculate the number of channels
            num_channels: int = max(row[self.CHANNEL_NUM_INDEX] for row in table) + 1

            # separate data list per channel
            y_data.extend([] for _ in range(num_channels))

            for table_row_num, row in enumerate(table):
                # get x_value to append
                x_val_to_append = table_row_num
                current_channel = row[
                    self.CHANNEL_NUM_INDEX
                ]  # initially chose current channel
                new_module_index: int = table_row_num // num_channels
                x_val_to_append = new_module_index

                # the index of the feature will the 0 + num non feature columns
                tensor_feature_index = feature_column_offset
                row_value = row[tensor_feature_index]
                if type(row_value) is not str:
                    # only append if new index we are appending
                    if len(x_data) == 0 or x_data[-1] != x_val_to_append:
                        x_data.append(x_val_to_append)

                    # append value for that channel
                    y_data[current_channel].append(row_value)
        else:
            # more than one feature was chosen
            error_str = "Make sure to pick only a single feature with your filter to plot a graph."
            error_str += " We recommend calling get_all_unique_feature_names() to find unique feature names."
            error_str += " Pick one of those features to plot."
            raise ValueError(error_str)

        # return x, y values, and if data is per-channel
        return (x_data, y_data, is_valid_per_channel_plot)

    def generate_plot_visualization(
        self, feature_filter: str, module_fqn_filter: str = ""
    ):
        r"""
        Takes in a feature and optional module_filter and plots of the desired data.

        For per channel features, it averages the value across the channels and plots a point
        per module. The reason for this is that for models with hundreds of channels, it can
        be hard to differentiate one channel line from another, and so the point of generating
        a single average point per module is to give a sense of general trends that encourage
        further deep dives.

        Note:
            Only features in the report that have tensor value data are plottable by this class
            When the tensor information is plotted, it will plot:
                idx as the x val, feature value as the y_val
            When the channel information is plotted, it will plot:
                the first idx of each module as the x val, feature value as the y_val [for each channel]
                The reason for this is that we want to be able to compare values across the
                channels for same layer, and it will be hard if values are staggered by idx
                This means each module is represented by only 1 x value
        Args:
            feature_filter (str): Filters the features presented to only those that
                contain this filter substring
            module_fqn_filter (str, optional): Only includes modules that contains this string
                Default = "", results in all the modules in the reports to be visible in the table

        Example Use:
            >>> # xdoctest: +SKIP("undefined variables")
            >>> mod_report_visualizer.generate_plot_visualization(
            ...     feature_filter="per_channel_min", module_fqn_filter="block1"
            ... )
            >>> # outputs line plot of per_channel_min information for all
            >>> # modules in block1 of model each channel gets its own line,
            >>> # and it's plotted across the in-order modules on the x-axis
        """
        # checks if we have matplotlib and let's user know to install it if don't
        if not got_matplotlib:
            print("make sure to install matplotlib and try again.")
            return None

        # get the x and y data and if per channel
        x_data, y_data, data_per_channel = self._get_plottable_data(
            feature_filter, module_fqn_filter
        )

        # plot based on whether data is per channel or not
        ax = plt.subplot()
        ax.set_ylabel(feature_filter)
        ax.set_title(feature_filter + " Plot")
        plt.xticks(x_data)  # only show ticks for actual points

        if data_per_channel:
            ax.set_xlabel("First idx of module")
            # set the legend as well
            # plot a single line that is average of the channel values
            num_modules = len(
                y_data[0]
            )  # all y_data have same length, so get num modules
            num_channels = len(
                y_data
            )  # we want num channels to be able to calculate average later

            avg_vals = [
                sum(y_data[:][index]) / num_channels for index in range(num_modules)
            ]

            # plot the three things we measured
            ax.plot(
                x_data, avg_vals, label=f"Average Value Across {num_channels} Channels"
            )
            ax.legend(loc="upper right")
        else:
            ax.set_xlabel("idx")
            ax.plot(x_data, y_data)

        # actually show the plot
        plt.show()

    def generate_histogram_visualization(
        self, feature_filter: str, module_fqn_filter: str = "", num_bins: int = 10
    ):
        r"""
        Takes in a feature and optional module_filter and plots the histogram of desired data.

        Note:
            Only features in the report that have tensor value data can be viewed as a histogram
            If you want to plot a histogram from all the channel values of a specific feature for
                a specific model, make sure to specify both the model and the feature properly
                in the filters and you should be able to see a distribution of the channel data

        Args:
            feature_filter (str, optional): Filters the features presented to only those that
                contain this filter substring
                Default = "", results in all the features being printed
            module_fqn_filter (str, optional): Only includes modules that contains this string
                Default = "", results in all the modules in the reports to be visible in the table
            num_bins (int, optional): The number of bins to create the histogram with
                Default = 10, the values will be split into 10 equal sized bins

        Example Use:
            >>> # xdoctest: +SKIP
            >>> mod_report_visualizer.generategenerate_histogram_visualization_plot_visualization(
            ...     feature_filter="per_channel_min", module_fqn_filter="block1"
            ... )
            # outputs histogram of per_channel_min information for all modules in block1 of model
                information is gathered across all channels for all modules in block 1 for the
                per_channel_min and is displayed in a histogram of equally sized bins
        """
        # checks if we have matplotlib and let's user know to install it if don't
        if not got_matplotlib:
            print("make sure to install matplotlib and try again.")
            return None

        # get the x and y data and if per channel
        _x_data, y_data, data_per_channel = self._get_plottable_data(
            feature_filter, module_fqn_filter
        )

        # for histogram, we just care about plotting the y data
        # plot based on whether data is per channel or not
        ax = plt.subplot()
        ax.set_xlabel(feature_filter)
        ax.set_ylabel("Frequency")
        ax.set_title(feature_filter + " Histogram")

        if data_per_channel:
            # set the legend as well
            # combine all the data
            all_data = []
            for channel_info in y_data:
                all_data.extend(channel_info)

            _val, bins, _ = plt.hist(
                all_data,
                bins=num_bins,
                stacked=True,
                rwidth=0.8,
            )
            plt.xticks(bins)
        else:
            _val, bins, _ = plt.hist(
                y_data,
                bins=num_bins,
                stacked=False,
                rwidth=0.8,
            )
            plt.xticks(bins)

        plt.show()
<|endoftext|># mypy: allow-untyped-defs
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import torch
from torch.ao.quantization.fx._equalize import EqualizationQConfig
from torch.ao.quantization.fx._model_report.detector import (
    DETECTOR_IS_POST_OBS_KEY,
    DETECTOR_OBS_ARGS_KEY,
    DETECTOR_OBS_TO_INSERT_KEY,
    DETECTOR_TARGET_NODE_KEY,
    DetectorBase,
    DetectorQConfigInfo,
)
from torch.ao.quantization.fx._model_report.model_report_visualizer import (
    ModelReportVisualizer,
)
from torch.ao.quantization.fx.graph_module import GraphModule
from torch.ao.quantization.observer import ObserverBase
from torch.ao.quantization.qconfig_mapping import QConfig, QConfigMapping


class ModelReport:
    r"""
    The ModelReport class aims to provide users an easy way to diagnose issues that they run into
    with their models. The class works with all traceable GraphModules to help diagnose issues,
    though the requirements on the type of model more-so depends on the specific report the user
    is trying to generate. With respect to the reports, the ModelReport class is initialized with
    a set of Detector classes, each of which generate reports on quantization configuration
    issues a use might have.

    Currently supports generating reports on:
    - Suggestions for per-channel vs. per-tensor quantization (nn.Module)
    - Suggestions for dynamic vs static quantization for linear layers (Graph Modules)
    - Suggestions for input-weight equalization for linear and conv layers (Graph Modules)
    - Suggestions for outlier detection for all layers (Graph Modules)

    The ModelReport class has the primary functionality of inserting observers (primarily the ModelReportObserver)
    where needed for each detector to gather the information it needs, and then after calibration, the ModelReport
    class compiles the report generated by each Detector class into a single report to return to the user. It also
    has the capability to remove all the observers it inserted as well.

    * :attr:`_model` The model we wish to generate the report for. Must be a traceable GraphModule

    * :attr:`_desired_report_detectors` The set of Detectors representing desired reports from the ModelReport class
        Make sure that these are all unique types of detectors [do not have more than 1 of the same class]

    * :attr:`_desired_detector_names` The set of detector names of the _desired_report_detectors.
        This set is generated by calling the get_detector_name() of each detector

    * :attr:`_detector_name_to_observer_fqns` The mapping from each detector to fqns of observers of interest
        The purpose of this is to keep track of what observers were inserted for each detector, so that they
        can be removed at the end if desired

    * :attr:`_prepared_flag` A boolean flag that keeps track of whether we have prepared the model or not
        This is to ensure we only insert observers once with the ModelReport instance

    * :attr:`_removed_observers` A boolean to track if we have removed observers already
        The purpose is to ensure we don't attempt to remove observers twice with the same ModelReport
        instance. This also allows the functionality where we can generate the report multiple times
        as long as we haven't removed the observers yet.

    Note:
        This class was initially designed to work with the Fx Graph Mode workflow in mind. However,
        full functionality is available as long as there is a traceable GraphModule that is being used.
        One method to get a traceable GraphModule without going through the Fx workflow is to use
        the QuantizationTracer class.

    General Flow for Fx workflow:
    1.) Initialize ModelReport object with reports of interest by passing in initialized detector objects and model
    2.) Prepare your model with prepare_fx
    3.) Call model_report.prepare_detailed_calibration to add relevant observers
    4.) Calibrate your model with data
    5.) Call model_report.generate_report on your model to generate report and optionally remove added observers
    Optional
        6.) Call model_report.generate_visualizer to get a ModelReportVisualizer instance
        7.) To help in parsing report information and debugging, view report info as a:
            - Table
            - Histogram
            - Line plot
    8.) Call model_report.generate_qconfigs to generate the qconfigs based on the report suggestions

    Example (with QuantizationTracer):
        >>> # xdoctest: +SKIP
        >>> # get the necessary qconfig
        >>> config = PrepareCustomConfig()
        >>> skipped_module_names, skipped_module_classes = (
        ...     get_skipped_module_name_and_classes(config, False)
        ... )

        >>> # initialize our model and get GraphModule
        >>> model = SomeModel()
        >>> tracer = QuantizationTracer(skipped_module_names, skipped_module_classes)
        >>> graph_module = GraphModule(model, tracer.trace(model))

        >>> # get our set of detectors and ModelReport instance
        >>> detector_set = set(
        ...     [
        ...         DynamicStaticDetector(tolerance=0.5),
        ...         InputWeightEqualizationDetector(ratio_threshold=0.7),
        ...     ]
        ... )
        >>> tracer_reporter = ModelReport(graph_module, tracer_detector_set)

        >>> # now we insert the observers and calibrate the model
        >>> tracer_model_with_observers = tracer_reporter.prepare_detailed_calibration()
        >>> for i in range(num_callibration_batches):
        >>>     example_input = get_callibration_input()
        >>>     tracer_model_with_observers(example_input)

        >>> # finally we generate the reports and optionally remove the observers we inserted
        >>> reports = tracer_reporter.generate_model_report(
        ...     remove_inserted_observers=True
        ... )

        >>> # Optional: we can generate the qconfig mapping based on the suggestions
        >>> qconfigs = model_report.generate_qconfig_mapping()

        >>> # Optional: we can generate the equalization mapping based on the suggestions
        >>> qconfigs = model_report.generate_equalization_mapping()

        >>> # Optional: we get a ModelReportVisualizer instance to do any visualizations desired
        >>> model_report_visualizer = tracer_reporter.generate_visualizer()

    """

    def __init__(self, model: GraphModule, desired_report_detectors: set[DetectorBase]):
        if len(desired_report_detectors) == 0:
            raise ValueError("Should include at least 1 desired report")

        # keep track of the model we wish to generate report for
        self._model: GraphModule = model

        # keep the reports private so they can't be modified
        self._desired_report_detectors = desired_report_detectors
        self._desired_detector_names = {
            detector.get_detector_name() for detector in desired_report_detectors
        }

        # keep a mapping of desired reports to observers of interest
        # this is to get the readings, and to remove them, can create a large set
        # this set can then be used to traverse the graph and remove added observers
        self._detector_name_to_observer_fqns: dict[str, set[str]] = {}

        # initialize each report to have empty set of observers of interest
        for desired_report in self._desired_detector_names:
            self._detector_name_to_observer_fqns[desired_report] = set()

        # flags to ensure that we can only prepare and remove observers once
        self._prepared_flag = False
        self._removed_observers = False

        # store the reports that we generated for visualization purposes
        # initially empty since no reports generated
        self._generated_reports: dict[str, dict] = {}

    def get_desired_reports_names(self) -> set[str]:
        """Returns a copy of the desired reports for viewing"""
        return self._desired_detector_names.copy()

    def get_observers_of_interest(self) -> dict[str, set[str]]:
        """Returns a copy of the observers of interest for viewing"""
        return self._detector_name_to_observer_fqns.copy()

    def prepare_detailed_calibration(self) -> GraphModule:
        r"""
        Takes in a graph model and inserts the following observers:
        - ModelReportObserver

        Each observer is inserted based on the desired_reports into the relevant locations

        Right now, each report in self._desired_detector_names has independent insertions
            However, if a module already has a Observer of the same type, the insertion will not occur
            This is because all of the same type of Observer collect same information, so redundant

        Returns the same GraphModule with the observers inserted
        """

        # if already prepared once, cannot prepare again
        if self._prepared_flag:
            raise ValueError(
                "Already ran preparing detailed calibration. Run the report generation next after calibration."
            )

        # loop through each detector, find where placements should be, and keep track
        insert_observers_fqns: dict[str, Any] = {}

        for detector in self._desired_report_detectors:
            # determine observer points for each detector
            obs_fqn_to_info = detector.determine_observer_insert_points(self._model)
            # map each insert point to the observer to use
            insert_observers_fqns.update(obs_fqn_to_info)
            # update the set of observers this report cares about
            self._detector_name_to_observer_fqns[detector.get_detector_name()] = set(
                obs_fqn_to_info.keys()
            )

        # now insert all the observers at their desired locations
        for observer_fqn in insert_observers_fqns:
            target_node = insert_observers_fqns[observer_fqn][DETECTOR_TARGET_NODE_KEY]
            insert_obs = insert_observers_fqns[observer_fqn][DETECTOR_OBS_TO_INSERT_KEY]
            insert_post = insert_observers_fqns[observer_fqn][DETECTOR_IS_POST_OBS_KEY]
            observer_args = insert_observers_fqns[observer_fqn][DETECTOR_OBS_ARGS_KEY]
            self._insert_observer_around_module(
                observer_fqn, target_node, insert_obs, observer_args, insert_post
            )

        self._prepared_flag = True

        return self._model

    def _insert_observer_around_module(
        self,
        obs_fqn: str,
        target_node: torch.fx.node.Node,
        obs_to_insert: ObserverBase,
        observer_args: tuple,
        insert_post: bool,
    ):
        r"""
        Helper function that inserts the observer into both the graph structure and the module of the model

        Args
            node_fqn (str): The fully qualified name of the observer we want to insert
            target_node (torch.fx.node.Node): The node in model we are inserting observers around
            obs_to_insert (ObserverBase): The observer we are inserting around target_node
            observer_args (Tuple): The arguments we want to pass into the observer
            insert_post (bool): whether this is meant to be a post observer for this node
        """
        # if we are inserting post, then our target node is the next node
        if insert_post:
            target_node = target_node.next

        with self._model.graph.inserting_before(target_node):
            self._model.add_submodule(obs_fqn, obs_to_insert)
            self._model.graph.create_node(
                op="call_module", target=obs_fqn, args=observer_args
            )

        # recompile model after inserts are made
        self._model.recompile()

    def _get_node_from_fqn(self, node_fqn: str) -> torch.fx.node.Node:
        r"""
        Takes in a node fqn and returns the node based on the fqn

        Args
            node_fqn (str): The fully qualified name of the node we want to find in model

        Returns the Node object of the given node_fqn otherwise returns None
        """
        node_to_return = None
        for node in self._model.graph.nodes:
            # if the target matches the fqn, it's the node we are looking for
            if node.target == node_fqn:
                node_to_return = node
                break

        if node_to_return is None:
            raise ValueError("The node_fqn is was not found within the module.")

        # assert for MyPy
        if not isinstance(node_to_return, torch.fx.node.Node):
            raise AssertionError("node_to_return must be a torch.fx.node.Node")

        return node_to_return

    def generate_model_report(
        self, remove_inserted_observers: bool
    ) -> dict[str, tuple[str, dict]]:
        r"""
        Generates all the requested reports.

        Note:
            You should have calibrated the model with relevant data before calling this

        The reports generated are specified by the desired_reports specified in desired_reports

        Can optionally remove all the observers inserted by the ModelReport instance

        Args:
            remove_inserted_observers (bool): True to remove the observers inserted by this ModelReport instance

        Returns a mapping of each desired report name to a tuple with:
            The textual summary of that report information
            A dictionary containing relevant statistics or information for that report

        Note:
            Throws exception if we try to generate report on model we already removed observers from
            Throws exception if we try to generate report without preparing for calibration
        """
        # if we haven't prepped model for calibration, then we shouldn't generate report yet
        if not self._prepared_flag:
            raise Exception(  # noqa: TRY002
                "Cannot generate report without preparing model for calibration"
            )

        # if we already removed the observers, we cannot generate report
        if self._removed_observers:
            raise Exception(  # noqa: TRY002
                "Cannot generate report on model you already removed observers from"
            )

        # keep track of all the reports of interest and their outputs
        reports_of_interest = {}

        for detector in self._desired_report_detectors:
            # generate the individual report for the detector
            report_output = detector.generate_detector_report(self._model)
            reports_of_interest[detector.get_detector_name()] = report_output

        # if user wishes to remove inserted observers, go ahead and remove
        if remove_inserted_observers:
            self._removed_observers = True
            # get the set of all Observers inserted by this instance of ModelReport
            all_observers_of_interest: set[str] = set()
            for desired_report in self._detector_name_to_observer_fqns:
                observers_of_interest = self._detector_name_to_observer_fqns[
                    desired_report
                ]
                all_observers_of_interest.update(observers_of_interest)

            # go through all_observers_of_interest and remove them from the graph and model
            for observer_fqn in all_observers_of_interest:
                # remove the observer from the model
                self._model.delete_submodule(observer_fqn)

                # remove the observer from the graph structure
                node_obj = self._get_node_from_fqn(observer_fqn)

                if node_obj:
                    self._model.graph.erase_node(node_obj)
                else:
                    raise ValueError("Node no longer exists in GraphModule structure")

            # remember to recompile the model
            self._model.recompile()

        # save the generated reports for visualization purposes
        saved_reports: dict[str, dict] = {
            report_name: report_tuple[1]
            for report_name, report_tuple in reports_of_interest.items()
        }

        self._generated_reports = saved_reports

        # return the reports of interest
        return reports_of_interest

    def _is_same_info_for_same_key(self, info_dict_a: dict, info_dict_b: dict) -> bool:
        r"""
        Takes in two dictionaries and ensures that any common keys between the two have the same
        values.

        Args:
            info_dict_a (Dict): First dictionary we wish to compare
            info_dict_b (Dict): Second dictionary we wish to compare

        Returns True if all shared keys have same values, false otherwise
        """
        # get the set of keys for both
        dict_a_keys: set = set(info_dict_a.keys())
        dict_b_keys: set = set(info_dict_b.keys())

        # get the intersection keys and check if same value for both dicts
        intersecting_keys: set = dict_a_keys.intersection(dict_b_keys)

        for key in intersecting_keys:
            dict_a_val = info_dict_a[key]
            dict_b_val = info_dict_b[key]

            # if it's a tensor we have to handle separately
            if type(dict_a_val) is torch.Tensor:
                # if dict_b_val not tensor, automatically false
                if (
                    type(dict_b_val) is not torch.Tensor
                    or sum(dict_a_val != dict_b_val) != 0
                ):
                    return False
            else:
                # for non-tensor vals
                if dict_a_val != dict_b_val:
                    return False

        # if no non matching shared keys found, return true
        return True

    def _reformat_reports_for_visualizer(self) -> OrderedDict:
        r"""
        Takes the generated reports and reformats them into the format that is desired by the
        ModelReportVisualizer

        Returns an OrderedDict mapping module_fqns to their features
        """
        # we want to reorder and reformat the information so it is ordered in terms of order
        # found in the model

        # first create new dict with all modules as keys and features under respective module
        module_fqns_to_features: dict[str, dict] = {}

        for report_name in self._generated_reports:
            # get mod -> feature dict and go through
            module_info = self._generated_reports[report_name]

            for module_fqn in module_info:
                # check if already in our accumulation dict
                if module_fqn in module_fqns_to_features:
                    # we merge all the features together
                    new_info: dict = module_info[module_fqn]
                    present_info: dict = module_fqns_to_features[module_fqn]

                    # merge them together into the new unioned dict
                    # same features keys -> same info, so okay if override

                    # do safety check to make sure shared keys have same info
                    if self._is_same_info_for_same_key(new_info, present_info):
                        module_fqns_to_features[module_fqn] = {
                            **new_info,
                            **present_info,
                        }
                    else:
                        error_str = "You have the same key with different values across detectors. "
                        error_str += "Someone incorrectly implemented a detector with conflicting keys to existing detectors."
                        raise ValueError(error_str)
                else:
                    # we just set it
                    module_fqns_to_features[module_fqn] = module_info[module_fqn]

        # our ordered dict so that modules can be ordered in order of how they appear in model
        features_by_module: OrderedDict[str, dict] = OrderedDict()

        # we loop through modules in graph in order
        for fqn, _module in self._model.named_modules():
            # find that fqn in fqns_to_features
            if fqn in module_fqns_to_features:
                # add it to our ordered dict
                features_by_module[fqn] = module_fqns_to_features[fqn]

        # return the ordered dict of info we created
        return features_by_module

    def generate_visualizer(self) -> ModelReportVisualizer:
        r"""
        Generates a ModelReportVisualizer instance using the reports generated
        by the generate_model_report() method.

        Returns the generated ModelReportVisualizer instance initialized

        Note:
