        ret_dims = utils.canonicalize_dims(input_dim, dim, wrap_scalar=False)

        # Check dims are unique
        torch._check(
            len(set(ret_dims)) == len(ret_dims), lambda: "FFT dims must be unique"
        )

    if shape is not None:
        if not isinstance(shape, Sequence):
            shape = (shape,)

        # Has shape, might have dim
        torch._check(
            dim is None or len(dim) == len(shape),
            lambda: "When given, dim and shape arguments must have the same length",
        )
        transform_ndim = len(shape)

        torch._check(
            transform_ndim <= input_dim,
            lambda: f"Got shape with {transform_ndim} values but input tensor "
            f"only has {input_dim} dimensions.",
        )

        # If shape is given, dims defaults to the last len(shape) dimensions
        if dim is None:
            ret_dims = tuple(range(input_dim - transform_ndim, input_dim))

        # Translate any -1 values in shape to the default length
        ret_shape = tuple(
            s if s != -1 else input_sizes[d]
            for (s, d) in zip(shape, ret_dims)  # type: ignore[possibly-undefined]
        )
    elif dim is None:
        # No shape, no dim
        ret_dims = tuple(range(input_dim))
        ret_shape = tuple(input_sizes)
    else:
        # No shape, has dim
        ret_shape = tuple(input_sizes[d] for d in ret_dims)  # type: ignore[possibly-undefined]

    for n in ret_shape:
        torch._check(n > 0, lambda: f"Invalid number of data points ({n}) specified")

    return _ShapeAndDims(shape=ret_shape, dims=ret_dims)  # type: ignore[possibly-undefined]


def _prod(xs: Iterable[int]) -> int:
    """Compute product of a list"""
    prod = 1
    for x in xs:
        prod *= x
    return prod


def _fftn_c2c(
    function_name: str,
    input: TensorLikeType,
    shape: tuple[int, ...],
    dim: tuple[int, ...],
    norm: NormType,
    forward: bool,
) -> TensorLikeType:
    """Common code for n-dimensional complex to complex FFTs (fftn or ifftn)"""
    torch._check(
        input.dtype.is_complex,
        lambda: f"{function_name} expects a complex input tensor, "
        f"but got {input.dtype}",
    )
    x = _resize_fft_input(input, dim, shape)
    output = prims.fft_c2c(x, dim=dim, forward=forward)
    return _apply_norm(output, norm=norm, signal_numel=_prod(shape), forward=forward)


@register_decomposition(aten.fft_fftn)
@out_wrapper()
def fftn(
    input: TensorLikeType,
    s: ShapeType | None = None,
    dim: DimsType | None = None,
    norm: NormType = None,
) -> TensorLikeType:
    (shape, dim) = _canonicalize_fft_shape_and_dim_args(input, s, dim)
    x = _maybe_promote_tensor_fft(input, require_complex=True)
    return _fftn_c2c("fftn", x, shape, dim, norm, forward=True)


@register_decomposition(aten.fft_ifftn)
@out_wrapper()
def ifftn(
    input: TensorLikeType,
    s: ShapeType | None = None,
    dim: DimsType | None = None,
    norm: NormType = None,
) -> TensorLikeType:
    (shape, dim) = _canonicalize_fft_shape_and_dim_args(input, s, dim)
    x = _maybe_promote_tensor_fft(input, require_complex=True)
    return _fftn_c2c("ifftn", x, shape, dim, norm, forward=False)


@register_decomposition(aten.fft_rfftn)
@out_wrapper()
def rfftn(
    input: TensorLikeType,
    s: ShapeType | None = None,
    dim: DimsType | None = None,
    norm: NormType = None,
) -> TensorLikeType:
    torch._check_type(
        not input.dtype.is_complex,
        lambda: f"rfftn expects a real-valued input tensor, but got {input.dtype}",
    )
    shape, dim = _canonicalize_fft_shape_and_dim_args(input, s, dim)
    input = _maybe_promote_tensor_fft(input, require_complex=False)
    input = _resize_fft_input(input, dim, shape)
    out = prims.fft_r2c(input, dim=dim, onesided=True)
    return _apply_norm(out, norm=norm, signal_numel=_prod(shape), forward=True)


@register_decomposition(aten.fft_ihfftn)
@out_wrapper()
def ihfftn(
    input: TensorLikeType,
    s: ShapeType | None = None,
    dim: DimsType | None = None,
    norm: NormType = None,
) -> TensorLikeType:
    torch._check_type(
        not input.dtype.is_complex,
        lambda: f"ihfftn expects a real-valued input tensor, but got {input.dtype}",
    )
    shape, dim = _canonicalize_fft_shape_and_dim_args(input, s, dim)
    torch._check(len(shape) > 0, lambda: "ihfftn must transform at least one axis")
    input = _maybe_promote_tensor_fft(input, require_complex=False)
    input = _resize_fft_input(input, dim, shape)

    tmp = prims.fft_r2c(input, dim=dim[-1:], onesided=True)

    if len(dim) == 1:
        tmp = _apply_norm(tmp, norm=norm, signal_numel=shape[0], forward=False)
        return prims.conj(tmp)

    tmp = prims.conj_physical(tmp)
    tmp = prims.fft_c2c(tmp, dim=dim[:-1], forward=False)
    return _apply_norm(tmp, norm=norm, signal_numel=_prod(shape), forward=False)


class _CanonicalizeC2rReturn(NamedTuple):
    shape: tuple[int, ...]
    dim: tuple[int, ...]
    last_dim_size: int


def _canonicalize_fft_c2r_shape_and_dim_args(
    fname: str,
    input: TensorLikeType,
    s: ShapeType | None,
    dim: DimsType | None,
) -> _CanonicalizeC2rReturn:
    """Canonicalize shape and dim arguments for n-dimensional c2r transforms,
    as well as calculating the last_dim_size which is shape[dim[-1]] for the output"""
    (shape, dim) = _canonicalize_fft_shape_and_dim_args(input, s, dim)
    torch._check(len(shape) > 0, lambda: f"{fname} must transform at least one axis")

    if s is None or s[-1] == -1:
        last_dim_size = 2 * (input.shape[dim[-1]] - 1)
    else:
        last_dim_size = shape[-1]

    torch._check(
        last_dim_size >= 1,
        lambda: f"Invalid number of data points ({last_dim_size}) specified",
    )

    shape_list = list(shape)
    shape_list[-1] = last_dim_size // 2 + 1
    return _CanonicalizeC2rReturn(
        shape=tuple(shape_list), dim=dim, last_dim_size=last_dim_size
    )


@register_decomposition(aten.fft_irfftn)
@out_wrapper()
def irfftn(
    input: TensorLikeType,
    s: ShapeType | None = None,
    dim: DimsType | None = None,
    norm: NormType = None,
) -> TensorLikeType:
    shape, dim, last_dim_size = _canonicalize_fft_c2r_shape_and_dim_args(
        "irfftn", input, s, dim
    )
    input = _maybe_promote_tensor_fft(input, require_complex=True)
    input = _resize_fft_input(input, dim, shape)
    out = prims.fft_c2r(input, dim=dim, last_dim_size=last_dim_size)
    return _apply_norm(out, norm, _prod(out.shape[d] for d in dim), forward=False)


@register_decomposition(aten.fft_hfftn)
@out_wrapper()
def hfftn(
    input: TensorLikeType,
    s: ShapeType | None = None,
    dim: DimsType | None = None,
    norm: NormType = None,
) -> TensorLikeType:
    shape, dim, last_dim_size = _canonicalize_fft_c2r_shape_and_dim_args(
        "hfftn", input, s, dim
    )
    input = _maybe_promote_tensor_fft(input, require_complex=True)
    input = _resize_fft_input(input, dim, shape)

    tmp = prims.fft_c2c(input, dim=dim[:-1], forward=True) if len(dim) > 1 else input
    tmp = _apply_norm(tmp, norm, _prod(shape[:-1]), forward=True)
    tmp = prims.conj_physical(tmp)
    out = prims.fft_c2r(tmp, dim=dim[-1:], last_dim_size=last_dim_size)
    return _apply_norm(out, norm, last_dim_size, forward=True)


@register_decomposition(aten.fft_fft2)
@out_wrapper()
def fft2(
    input: TensorLikeType,
    s: ShapeType | None = None,
    dim: DimsType | None = (-2, -1),
    norm: NormType = None,
) -> TensorLikeType:
    return torch.fft.fftn(input, s=s, dim=dim, norm=norm)


@register_decomposition(aten.fft_ifft2)
@out_wrapper()
def ifft2(
    input: TensorLikeType,
    s: ShapeType | None = None,
    dim: DimsType | None = (-2, -1),
    norm: NormType = None,
) -> TensorLikeType:
    return torch.fft.ifftn(input, s=s, dim=dim, norm=norm)


@register_decomposition(aten.fft_rfft2)
@out_wrapper()
def rfft2(
    input: TensorLikeType,
    s: ShapeType | None = None,
    dim: DimsType | None = (-2, -1),
    norm: NormType = None,
) -> TensorLikeType:
    return torch.fft.rfftn(input, s=s, dim=dim, norm=norm)


@register_decomposition(aten.fft_irfft2)
@out_wrapper()
def irfft2(
    input: TensorLikeType,
    s: ShapeType | None = None,
    dim: DimsType | None = (-2, -1),
    norm: NormType = None,
) -> TensorLikeType:
    return torch.fft.irfftn(input, s=s, dim=dim, norm=norm)


@register_decomposition(aten.fft_hfft2)
@out_wrapper()
def hfft2(
    input: TensorLikeType,
    s: ShapeType | None = None,
    dim: DimsType | None = (-2, -1),
    norm: NormType = None,
) -> TensorLikeType:
    return torch.fft.hfftn(input, s=s, dim=dim, norm=norm)


@register_decomposition(aten.fft_ihfft2)
@out_wrapper()
def ihfft2(
    input: TensorLikeType,
    s: ShapeType | None = None,
    dim: DimsType | None = (-2, -1),
    norm: NormType = None,
) -> TensorLikeType:
    return torch.fft.ihfftn(input, s=s, dim=dim, norm=norm)


def _default_alldims(dim: DimsType | None, x: TensorLikeType) -> list[int]:
    """Convert Optional[DimsType] to a simple list, defaulting to all dimensions"""
    if dim is None:
        return list(range(x.ndim))
    elif not isinstance(dim, Sequence):
        return [dim]
    else:
        return list(dim)


@register_decomposition(aten.fft_fftshift)
def fftshift(input: TensorLikeType, dim: DimsType | None = None) -> TensorLikeType:
    dims = _default_alldims(dim, input)
    shift = [input.shape[d] // 2 for d in dims]
    return torch.roll(input, shift, dims)


@register_decomposition(aten.fft_ifftshift)
def ifftshift(input: TensorLikeType, dim: DimsType | None = None) -> TensorLikeType:
    dims = _default_alldims(dim, input)
    shift = [(input.shape[d] + 1) // 2 for d in dims]
    return torch.roll(input, shift, dims)
<|endoftext|># mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import math
from collections.abc import Callable
from functools import wraps
from typing import Concatenate, Optional, TypeVar, Union
from typing_extensions import ParamSpec

import torch
import torch._prims as prims
import torch._prims_common as utils
import torch._refs as refs
from torch._decomp import register_decomposition
from torch._prims_common import (
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    NumberType,
    ShapeType,
    TensorLike,
    TensorLikeType,
)
from torch._prims_common.wrappers import (
    elementwise_type_promotion_wrapper,
    elementwise_unary_scalar_wrapper,
    out_wrapper,
)
from torch._refs import _make_inplace


__all__ = [
    "alpha_dropout",
    "celu",
    "celu_",
    "channel_shuffle",
    "dropout",
    "elu",
    "elu_",
    "gelu",
    "glu",
    "group_norm",
    "hardshrink",
    "hardtanh",
    "hinge_embedding_loss",
    "huber_loss",
    "l1_loss",
    "layer_norm",
    "leaky_relu",
    "log_softmax",
    "margin_ranking_loss",
    "mish",
    "mish_",
    "mse_loss",
    "nll_loss",
    "pairwise_distance",
    "pdist",
    "poisson_nll_loss",
    "prelu",
    "relu",
    "relu6",
    "selu",
    "selu_",
    "smooth_l1_loss",
    "softmax",
    "softmin",
    "softplus",
    "softshrink",
    "tanhshrink",
    "threshold",
    "threshold_",
    "triplet_margin_loss",
]

_T = TypeVar("_T")
_P = ParamSpec("_P")

Tensor = torch.Tensor
aten = torch._ops.ops.aten
DispatchKey = torch._C.DispatchKey  # type: ignore[attr-defined]


def _dropout_helper(
    self: TensorLikeType,
    val: float,
) -> TensorLikeType:
    """
    Helper function for all dropout-type operators. During training,
    some of the elements of the input tensor are randomly masked.

    Returns the masked tensor of the boolean values.

    """

    return (
        refs._uniform_helper(
            self.shape,
            low=0.0,
            high=1.0,
            dtype=torch.float32,
            device=self.device,
            stride=self.stride(),
        )
        < val
    )


@register_decomposition(aten.alpha_dropout)
def alpha_dropout(
    self: TensorLikeType, p: float = 0.5, training: bool = False, inplace: bool = False
) -> TensorLikeType:
    if inplace:
        raise NotImplementedError

    if not training:
        return self

    torch._check(
        p <= 1 and p >= 0,
        lambda: f"dropout probability has to be between 0 and 1, but got, {p}",
    )

    if p == 1:
        return torch.zeros_like(self)

    if p == 0:
        return self

    dropout_mask = _dropout_helper(self, 1 - p)

    # From paper: Self-Normalizing Neural Networks (https://arxiv.org/pdf/1706.02515.pdf)
    # alpha = - SELU.alpha * SELU.scale, here
    # SELU.alpha = 1.6732632423543772848170429916717 and
    # SELU.scale = 1.0507009873554804934193349852946
    alpha = -1.7580993408473766

    a = 1.0 / math.sqrt((alpha * alpha * p + 1) * (1 - p))
    b = torch.logical_not(dropout_mask)
    b = b * (alpha * a) + alpha * a * p
    dropout_mask = a * dropout_mask

    return self * dropout_mask + b


def _inplace_wrapper(fn: Callable[_P, _T]) -> Callable[_P, _T]:
    """
    Given a nn.functional non-linearity, implements its `inplace: bool` argument
    """

    # nb. We use the name of the first argument used in the unary references
    @wraps(fn)
    def _fn(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        a = args[0]
        if "inplace" not in kwargs:
            kwargs["inplace"] = False

        if kwargs["inplace"]:
            torch._check(
                "out" not in kwargs,
                lambda: "Cannot set inplace=True and pass out= at the same time",
            )
            kwargs["inplace"] = False
            kwargs["out"] = a
            return fn(*args, **kwargs)
        else:
            return fn(*args, **kwargs)

    return _fn


# celu is implemented specially because it has an alpha argument
# celu is very similar to elu
@register_decomposition(aten.celu)
@_inplace_wrapper
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def celu(
    a: TensorLikeType, alpha: NumberType | None = None, inplace: bool = False
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.celu
    """

    if inplace:
        raise NotImplementedError

    rhs: TensorLikeType
    if alpha is not None:
        torch._check(
            alpha != 0,
            lambda: "ZeroDivisionError: alpha cannot be 0 for CELU",
        )
        python_type = utils.dtype_to_type(a.dtype)
        if not utils.is_weakly_lesser_type(type(alpha), python_type):
            msg = f"alpha argument of type {type(alpha)} cannot be safely cast to type {python_type}!"
            raise ValueError(msg)
        torch._check(
            alpha != 0,
            lambda: "ZeroDivisionError: alpha cannot be 0 for CELU",
        )
        rhs = alpha * torch.expm1(torch.true_divide(a, alpha))  # type: ignore[arg-type]
    else:
        rhs = torch.expm1(a)

    return torch.where(a > 0, a, rhs)


@_inplace_wrapper
@out_wrapper()
def dropout(
    a: TensorLikeType, p: float = 0.5, training: bool = True, inplace: bool = False
) -> TensorLikeType:
    if inplace:
        raise NotImplementedError

    if not training:
        return a

    torch._check(
        p <= 1 and p >= 0,
        lambda: f"dropout probability has to be between 0 and 1, but got, {p}",
    )

    if p == 1:
        return torch.zeros_like(a)

    if p == 0:
        return a

    scale = 1 / (1 - p)
    dropout_mask = _dropout_helper(a, 1 - p)

    return a * dropout_mask * scale


@register_decomposition(aten.elu)
@_inplace_wrapper
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def elu(
    a: TensorLikeType,
    alpha: NumberType = 1.0,
    scale: NumberType = 1.0,
    input_scale: NumberType = 1.0,
    inplace: bool = False,
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.elu
    """
    if inplace:
        raise NotImplementedError

    # nb. This should be factored out into a can_cast aux function
    python_type = utils.dtype_to_type(a.dtype)
    torch._check(
        utils.is_weakly_lesser_type(type(input_scale), python_type),
        lambda: f"input_scale argument of type {type(input_scale)} cannot be safely cast to type {python_type}!",
    )
    torch._check(
        utils.is_weakly_lesser_type(type(scale), python_type),
        lambda: f"scale argument of type {type(scale)} cannot be safely cast to type {python_type}!",
    )
    torch._check(
        utils.is_weakly_lesser_type(type(alpha), python_type),
        lambda: f"alpha argument of type {type(alpha)} cannot be safely cast to type {python_type}!",
    )

    return torch.where(a > 0, scale * a, (alpha * scale) * torch.expm1(a * input_scale))


@register_decomposition(aten.relu)
@_inplace_wrapper
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def relu(a: TensorLikeType, inplace: bool = False) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.relu
    """

    if inplace:
        raise NotImplementedError

    return torch.where(torch.le(a, 0), 0, a)


@_inplace_wrapper
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def hardsigmoid(a: TensorLikeType, inplace: bool = False) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.hardsigmoid
    """
    return torch.clamp(a / 6 + 0.5, 0, 1)


@register_decomposition(aten.channel_shuffle)
@out_wrapper()
def channel_shuffle(input: TensorLikeType, groups: int) -> TensorLikeType:
    """
    Reference implementation of :func:`torch.nn.functional.channel_shuffle`.
    """
    from torch._meta_registrations import device_hint

    torch._check(
        input.dim() > 2,
        lambda: f"channel_shuffle expects input with > 2 dims, but got input with sizes {list(input.size())}",
    )
    c = input.shape[1]
    torch._check(
        groups > 0,
        lambda: f"Number of groups to divide channels in must be positive. Value of groups:{groups}",
    )
    torch._check(
        (c % groups) == 0,
        lambda: f"Number of channels must be divisible by groups. Got {c} channels and {groups} groups.",
    )
    n = input.shape[0]
    cg = c // groups
    dhw = input.shape[2:]

    if input.numel() == 0 or (
        device_hint(input) == "cuda" and (groups == 1 or groups == c)
    ):
        return input.view(input.shape)

    return (
        input.reshape(n, groups, cg, *dhw)
        .transpose(1, 2)
        .reshape(input.shape)
        .contiguous()
    )


def group_norm(
    input: Tensor,
    num_groups: int,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
    """
    Reference implementation of :func:`torch.nn.functional.group_norm`.
    """
    torch._check(
        input.ndim >= 2,
        lambda: f"Expected at least 2 dimensions for input tensor but received {input.ndim}",
    )

    batch_size = input.shape[0]
    num_channels = input.shape[1]
    torch._check(
        num_channels % num_groups == 0,
        lambda: "Expected number of channels in input to be divisible by num_groups, "
        + f"but got input of shape {input.shape} and num_groups = {num_groups}",
    )

    # input shape is (N, C, *), so we flatten all inner dimensions except (N, C)
    flattened_inner_size = 1
    for dim_length in input.shape[2:]:
        flattened_inner_size *= dim_length

    return torch.native_group_norm(
        input,
        weight,
        bias,
        batch_size,
        num_channels,
        flattened_inner_size,
        num_groups,
        eps,
    )[0]


def layer_norm(
    input: Tensor,
    normalized_shape: ShapeType,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
    """
    Reference implementation of :func:`torch.nn.functional.layer_norm`.
    """
    return torch.native_layer_norm(input, normalized_shape, weight, bias, eps)[0]


@register_decomposition(aten.leaky_relu)
@_inplace_wrapper
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def leaky_relu(
    a: TensorLikeType, negative_slope: float = 0.01, inplace: bool = False
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.leaky_relu
    """

    if inplace:
        raise NotImplementedError

    python_type = utils.dtype_to_type(a.dtype)
    if not utils.is_weakly_lesser_type(type(negative_slope), python_type):
        msg = f"negative_slope argument of type {type(negative_slope)} cannot be safely cast to type {python_type}!"
        raise ValueError(msg)
    return torch.where(torch.gt(a, 0), a, torch.mul(a, negative_slope))


@register_decomposition(aten.mish)
@_inplace_wrapper
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def mish(a: TensorLikeType, inplace: bool = False) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.mish
    """

    if inplace:
        raise NotImplementedError
    return a * torch.tanh(torch.nn.functional.softplus(a))


@register_decomposition(aten.selu)
@_inplace_wrapper
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def selu(a: TensorLikeType, inplace: bool = False) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.selu
    """
    if inplace:
        raise NotImplementedError

    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    rhs = alpha * torch.expm1(a)

    return scale * torch.where(a > 0, a, rhs)


# Forwarding alias: the functional variant doesn't support the out kwarg
# CompositeImplicitAutograd - don't register decomp
def softmax(
    a: TensorLikeType,
    dim: int | None = None,
    _stacklevel: int = 3,  # for compat when using TorchRefsMode(strict=True)
    dtype: torch.dtype | None = None,
) -> TensorLikeType:
    # The error is for compat with regular PyTorch, which has this behavior
    # deprecated.  For PrimTorch, it's fine to drop support for deprecated
    # behavior because it requires explicit opt in.  This error is to inform
    # users how to update their calls.
    torch._check(dim is not None, lambda: "implicit dim not supported, use dim=X")
    return torch.softmax(a=a, dim=dim, dtype=dtype)  # type: ignore[call-overload]


# CompositeImplicitAutograd - don't register decomp
def softmin(
    a: TensorLikeType,
    dim: int | None = None,
    _stacklevel: int = 3,  # for compat when using TorchRefsMode(strict=True)
    dtype: torch.dtype | None = None,
) -> TensorLikeType:
    # The error is for compat with regular PyTorch, which has this behavior
    # deprecated.  For PrimTorch, it's fine to drop support for deprecated
    # behavior because it requires explicit opt in.  This error is to inform
    # users how to update their calls.
    torch._check(dim is not None, lambda: "implicit dim not supported, use dim=X")
    return torch.softmax(a=-a, dim=dim, dtype=dtype)  # type: ignore[call-overload]


# softplus is implemented specially because it has beta and threshold arguments
@register_decomposition(aten.softplus)
@_inplace_wrapper
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def softplus(
    a: TensorLikeType,
    beta: NumberType | None = None,
    threshold: NumberType = 20,
    inplace: bool = False,
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.softplus
    """

    if inplace:
        raise NotImplementedError

    rhs: TensorLikeType
    if beta is not None:
        python_type = utils.dtype_to_type(a.dtype)
        if not utils.is_weakly_lesser_type(type(beta), python_type):
            msg = f"beta argument of type {type(beta)} cannot be safely cast to type {python_type}!"
            raise ValueError(msg)
        scaled_input = a * beta
        rhs = torch.true_divide(torch.log1p(torch.exp(scaled_input)), beta)  # type: ignore[arg-type]

    else:
        scaled_input = a
        rhs = torch.log1p(torch.exp(scaled_input))

    return torch.where(scaled_input > threshold, a, rhs)


@aten.hardshrink.default.py_impl(DispatchKey.Autograd)
@register_decomposition(aten.hardshrink)
@out_wrapper()
def hardshrink(a: TensorLikeType, lambd: float = 0.5):
    # Formula for reference,
    # hardshrink(x) = x if x > lambd
    #               = x if x < -lambd
    #               = 0 otherwise
    return torch.where(torch.abs(a) <= lambd, 0, a)


@aten.softshrink.default.py_impl(DispatchKey.Autograd)
@register_decomposition(aten.softshrink)
@out_wrapper()
def softshrink(a: TensorLikeType, lambd: float = 0.5):
    # Formula for reference,
    # softshrink(x) = x - lambd if x > lambd
    #               = x + lambd if x < -lambd
    #               = 0 otherwise
    torch._check(
        0 <= lambd <= torch.finfo(a.dtype).max,
        lambda: f"lambda must be in range [0, {torch.finfo(a.dtype).max}] for input dtype {a.dtype}, but found {lambd}",
    )
    if a.dtype in (torch.bfloat16, torch.float16):
        lambd = torch.scalar_tensor(lambd, dtype=a.dtype, device=a.device)  # type: ignore[arg-type]
    # We implement this in one torch.where to generate better code in the backward
    # see https://github.com/pytorch/pytorch/pull/107052#discussion_r1293748211
    # We multiply by 0 for dealing with nans
    return torch.where(torch.abs(a) > lambd, a - torch.sign(a) * lambd, a * 0)


# Losses
def _reduction_int_to_str(reduction: int) -> str:
    from torch._decomp.decompositions import Reduction

    if reduction == Reduction.NONE.value:
        return "none"
    elif reduction == Reduction.MEAN.value:
        return "mean"
    elif reduction == Reduction.SUM.value:
        return "sum"
    else:
        raise ValueError(f"{reduction} is not a valid value for reduction")


def _apply_loss_reduction(loss: TensorLikeType, reduction: str) -> TensorLikeType:
    if reduction == "sum":
        return torch.sum(loss)
    elif reduction == "mean":
        return torch.mean(loss)
    else:  # reduction == "none"
        return loss


def _check_reduction_value(reduction: str):
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"{reduction} is not a valid value for reduction")


# This helper function maps deprecated arguments, "size_average" and "reduce"
# to their corresponding "reduction" string argument
def _get_string_reduction_arg(*, size_average: bool | None, reduce: bool | None) -> str:
    if size_average is None:
        size_average = True
    if reduce is None:
        reduce = True
    if size_average and reduce:
        ret = "mean"
    elif reduce:
        ret = "sum"
    else:
        ret = "none"
    return ret


# CompositeImplicitAutograd - don't register decomp
@elementwise_type_promotion_wrapper(
    type_promoting_args=("input", "target"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
)
def l1_loss(
    input: TensorLikeType,
    target: TensorLikeType,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.l1_loss
    """
    if size_average is not None or reduce is not None:
        # TODO: Raise exception instead of converting value.  This is only for
        # primTorch since it can drop support for deprecated arguments.
        # msg = "size_average and reduce args are deprecated, please use reduction argument."
        reduction = _get_string_reduction_arg(size_average=size_average, reduce=reduce)
    _check_reduction_value(reduction)
    loss = torch.abs(input - target)
    return _apply_loss_reduction(loss, reduction)


@elementwise_type_promotion_wrapper(
    type_promoting_args=("input", "target"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
)
def smooth_l1_loss(
    input: TensorLikeType,
    target: TensorLikeType,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
    beta: float = 1.0,
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.smooth_l1_loss
    """
    if size_average is not None or reduce is not None:
        # TODO: Raise exception instead of converting value.  This is only for
        # primTorch since it can drop support for deprecated arguments.
        # msg = "size_average and reduce args are deprecated, please use reduction argument."
        reduction = _get_string_reduction_arg(size_average=size_average, reduce=reduce)
    _check_reduction_value(reduction)

    if beta == 0.0:
        return torch.nn.functional.l1_loss(
            input, target, size_average=size_average, reduce=reduce, reduction=reduction
        )
    else:
        loss = torch.abs(input - target)

        loss = torch.where(loss < beta, 0.5 * loss**2 / beta, loss - 0.5 * beta)
        return _apply_loss_reduction(loss, reduction)


# Forwarding alias: the functional variant doesn't support the out kwarg
# CompositeImplicitAutograd - don't register decomp
def log_softmax(
    a: TensorLikeType,
    dim: int | None = None,
    _stacklevel: int = 3,  # for compat when using TorchRefsMode(strict=True)
    dtype: torch.dtype | None = None,
) -> TensorLikeType:
    # The error is for compat with regular PyTorch, which has this behavior
    # deprecated.  For PrimTorch, it's fine to drop support for deprecated
    # behavior because it requires explicit opt in.  This error is to inform
    # users how to update their calls.
    torch._check(dim is not None, lambda: "implicit dim not supported, use dim=X")
    return torch.log_softmax(a=a, dim=dim, dtype=dtype)  # type: ignore[call-overload]


@register_decomposition(aten.margin_ranking_loss)
def margin_ranking_loss(
    input1: TensorLikeType,
    input2: TensorLikeType,
    target: TensorLikeType,
    margin: float = 0.0,
    reduction: str = "mean",
) -> TensorLikeType:
    # loss_without_reduction = max(0, -target * (input1 - input2) + margin)
    if input1.ndim != input2.ndim or input1.ndim != target.ndim:
        raise RuntimeError(
            "margin_ranking_loss : All input tensors should have same dimension but got sizes: "
            f"input1: {input1.shape}, input2: {input2.shape}, target: {target.shape} "
        )
    _check_reduction_value(reduction)
    loss = torch.clamp_min(-target * (input1 - input2) + margin, 0)
    return _apply_loss_reduction(loss, reduction)


@elementwise_type_promotion_wrapper(
    type_promoting_args=("input", "target"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
)
def mse_loss(
    input: TensorLikeType,
    target: TensorLikeType,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
) -> TensorLikeType:
    if size_average is not None or reduce is not None:
        # TODO: Raise exception instead of converting value.  This is only for
        # primTorch since it can drop support for deprecated arguments.
        # msg = "size_average and reduce args are deprecated, please use reduction argument."
        reduction = _get_string_reduction_arg(size_average=size_average, reduce=reduce)
    _check_reduction_value(reduction)
    loss = torch.pow(input - target, 2)
    return _apply_loss_reduction(loss, reduction)


@register_decomposition(aten.hinge_embedding_loss)
def hinge_embedding_loss(
    input: TensorLikeType,
    target: TensorLikeType,
    margin: float = 1.0,
    reduction: str = "mean",
) -> TensorLikeType:
    # loss_without_reduction = input if y == 1
    #                        = max(0, margin - input) if y == -1
    _check_reduction_value(reduction)
    margin_clamp = torch.clamp_min(margin - input, 0)
    output_margin = torch.where(target != 1, margin_clamp, 0)
    output_self = torch.where(target != -1, input, 0)
    loss = output_margin + output_self
    return _apply_loss_reduction(loss, reduction)


def _nll_loss_nd(
    input: TensorLikeType,
    target: TensorLikeType,
    weight: TensorLikeType | None,
    reduction: str,
    ignore_index: int,
) -> TensorLikeType:
    torch._check(
        input.ndim > 0 and input.ndim <= 3,
        lambda: f"Expected input dimension to be either [1, 2, 3] but received {input.ndim}.",
    )

    torch._check(
        (input.ndim == 1) or (input.shape[0] == target.shape[0]),
        lambda: f"Expected input batch size {input.shape[0]} to match target batch size {target.shape[0]}.",
    )

    _check_reduction_value(reduction)

    flat_target = torch.flatten(target)
    ignore_classes_mask = torch.eq(flat_target, ignore_index)

    # TODO: Enable data-dependent checks with debug mode
    # TODO: This check does not work with FakeTensor inputs; See Issue #85834
    # Explicit cast for class_check to bool; See Issue #78071
    """
    from torch._subclasses.fake_tensor import FakeTensor, is_fake_tensor
    num_classes = input.shape[1] if input.ndim > 1 else input.shape[0]
    valid_classes_mask = torch.logical_and(
        (flat_target >= 0), (flat_target < num_classes)
    )
    class_check = torch.all(torch.logical_or(ignore_classes_mask, valid_classes_mask))
    torch._check(
        is_fake_tensor(target) or bool(class_check.item()),
        lambda: "A target class is out-of-bounds and not the ignore index.",
    )
    """

    ignore_class_weight = torch.scalar_tensor(0, dtype=input.dtype, device=input.device)
    class_weight = (
        torch.scalar_tensor(1, dtype=input.dtype, device=input.device)
        if weight is None
        else weight[flat_target]
    )
    current_weight = torch.where(
        ignore_classes_mask,
        ignore_class_weight,
        class_weight,
    )

    if input.ndim == 1:
        # implicit batch size = 1
        # input (1 batch size, C classes)
        loss = -input[target] * current_weight
    elif input.ndim == 2:
        # input (N batch size, C classes)
        batch_size = input.shape[0]
        loss = -input[torch.arange(batch_size), target] * current_weight
    else:
        # 3D case (N batch size, C classes, K dimensions)
        # input (N batch size, C classes, K)
        batch_size = input.shape[0]
        extent = input.shape[2]
        numel = batch_size * extent
        indices = torch.arange(numel)
        bdx = indices // extent
        kdx = indices % extent
        loss = -input[bdx, flat_target, kdx] * current_weight
    loss = torch.reshape(loss, target.shape)

    if reduction == "none":
        return loss
    elif reduction == "sum":
        return torch.sum(loss)
    else:
        # calculate weighted mean of the loss function
        return torch.sum(loss) / torch.sum(current_weight)


@register_decomposition(aten.nll_loss)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("input",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def nll_loss(
    input: TensorLikeType,
    target: TensorLikeType,
    weight: TensorLikeType | None = None,
    size_average: bool | None = None,
    ignore_index: int = -100,
    reduce: bool | None = None,
    reduction: str = "mean",
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.nll_loss
    """
    torch._check(
        input.ndim > 0,
        lambda: f"Expected input tensor to have 1 or more dimensions (got {input.ndim})",
    )

    # TODO: raise exception instead of converting value
    # msg = "size_average and reduce args are deprecated, please use reduction argument."
    # Convert these options for consistency with the eager mode
    if size_average is not None or reduce is not None:
        reduction = _get_string_reduction_arg(size_average=size_average, reduce=reduce)

    # The expected behavior when the target and input have zero elements:
    #   reduction = 'none' --- tensor([])
    #   reduction = 'sum'  --- tensor(0.)
    #   reduction = 'mean' --- tensor(nan)
    # Mean reduction on empty tensors produces NaN. See the discussion in
    # https://github.com/pytorch/pytorch/pull/64572#issuecomment-926504162
    if input.numel() == 0 and target.numel() == 0:
        if reduction == "none":
            return torch.zeros_like(target)
        elif reduction == "sum":
            return torch.empty_like(target)
        else:
            return torch.full_like(target, float("nan"))

    # The _nll_loss_nd helper function handles the most common cases.
    # ndim == 1 (Single Example)
    #   => Batch Size: 1, Input: (C), Target: ()
    # ndim == 2 (k = 1)
    #   => Batch Size: N, Input: (N, C), Target: (N)
    # ndim == 3 (k > 1)
    #   => Batch Size: N, Input: (N, C, K), Target: (N, K)
    if input.ndim <= 3:
        return _nll_loss_nd(input, target, weight, reduction, ignore_index)

    # For ndim > 3, we reshape the input and target to 3-D case.
    # Input (N batch-size, C classes, k-dimensions)
    # Target (N batch-size, k-dimensions)
    torch._check(
        input.ndim > 0 and target.ndim > 0 and target.shape[1:] == input.shape[2:],
        lambda: (
            "Expected input and target to both have ndim > 0 and "
            "target.shape[1:] == input.shape[2:], but got "
            f"target.shape {target.shape} and input.shape {input.shape}"
        ),
    )

    batch_size = input.shape[0]
    num_classes = input.shape[1]
    out_size = [batch_size] + list(target.shape[1:])

    input = torch.reshape(input, [batch_size, num_classes, -1])
    target = torch.reshape(target, [batch_size, -1])
    if reduction != "none":
        return _nll_loss_nd(input, target, weight, reduction, ignore_index)
    else:
        result = _nll_loss_nd(input, target, weight, reduction, ignore_index)
        # reshape flattened inner-dim to original k-dimensions
        return torch.reshape(result, out_size)


# TODO: This ref supports int reduction and out kwarg to be compatible with ATen:
# https://github.com/pytorch/pytorch/issues/83931
# TODO: Could be rewritten to support complex:
# https://github.com/pytorch/pytorch/pull/85041
@register_decomposition(aten.huber_loss)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("input", "target"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def huber_loss(
    input: TensorLikeType,
    target: TensorLikeType,
    reduction: str | int = "mean",
    delta: float = 1.0,
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.huber_loss
    """
    if type(reduction) is int:
        reduction = _reduction_int_to_str(reduction)
    _check_reduction_value(reduction)  # type: ignore[arg-type]
    torch._check(
        delta > 0,
        lambda: "huber_loss does not support non-positive values for delta.",
    )
    z = (input - target).abs()
    loss = torch.where(z < delta, 0.5 * z * z, delta * (z - 0.5 * delta))
    return _apply_loss_reduction(loss, reduction)  # type: ignore[arg-type]


# tanhshrink does not use _make_elementwise_unary_reference because it does not support out
@elementwise_unary_scalar_wrapper
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def tanhshrink(a: TensorLikeType) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.tanhshrink
    """
    if not isinstance(a, TensorLike):
        raise RuntimeError(
            "Expected a tensor input for an elementwise unary operation!"
        )
    return a - torch.tanh(a)


@register_decomposition(aten.threshold)
@_inplace_wrapper
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def threshold(
    a: TensorLikeType,
    threshold: NumberType,
    value: bool | int | float,
    inplace: bool = False,
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.threshold
    """

    if inplace:
        raise NotImplementedError

    return torch.where(a <= threshold, value, a)


# CompositeImplicitAutograd - don't register decomp
# No elementwise type promotion - core op doesn't explicitly type promote
def triplet_margin_loss(
    anchor: TensorLikeType,
    positive: TensorLikeType,
    negative: TensorLikeType,
    margin: float = 1.0,
    p: float = 2,
    eps: float = 1e-6,
    swap: bool = False,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
) -> TensorLikeType:
    if size_average is not None or reduce is not None:
        # TODO: Raise exception instead of converting value.  This is only for
        # primTorch since it can drop support for deprecated arguments.
        # msg = "size_average and reduce args are deprecated, please use reduction argument."
        reduction = _get_string_reduction_arg(size_average=size_average, reduce=reduce)

    if margin <= 0:
        raise ValueError(f"margin must be greater than 0, got {margin}")

    # torch.nn.functional.triplet_margin_with_distance_loss has no ref defined
    # since it's a pure Python implementation.  Use this helper instead.
    return _triplet_margin_with_distance_loss(
        anchor=anchor,
        positive=positive,
        negative=negative,
        distance_function=lambda x, y: torch.pairwise_distance(x, y, p, eps),
        margin=margin,
        swap=swap,
        reduction=reduction,
    )


# Pure Python impl - don't register decomp and don't add a ref.  Defined as a
# helper here since triplet_margin_loss can be nicely implemented with it.
def _triplet_margin_with_distance_loss(
    anchor: TensorLikeType,
    positive: TensorLikeType,
    negative: TensorLikeType,
    *,
    distance_function: Callable[[TensorLikeType, TensorLikeType], TensorLikeType]
    | None = None,
    margin: float = 1.0,
    swap: bool = False,
    reduction: str = "mean",
) -> TensorLikeType:
    _check_reduction_value(reduction)

    a_dim = anchor.ndim
    p_dim = positive.ndim
    n_dim = negative.ndim
    torch._check(
        a_dim == p_dim and p_dim == n_dim,
        lambda: (
            f"The anchor, positive, and negative tensors are expected to have "
            f"the same number of dimensions, but got: anchor {a_dim}D, "
            f"positive {p_dim}D, and negative {n_dim}D inputs"
        ),
    )

    if distance_function is None:
        distance_function = torch.pairwise_distance

    dist_pos = distance_function(anchor, positive)
    dist_neg = distance_function(anchor, negative)
    # The distance swap is described in the paper "Learning shallow
    # convolutional feature descriptors with triplet losses" by V. Balntas, E.
    # Riba et al.  If True, and if the positive example is closer to the
    # negative example than the anchor is, swaps the positive example and the
    # anchor in the loss computation.
    if swap:
        dist_swap = distance_function(positive, negative)
        dist_neg = torch.minimum(dist_neg, dist_swap)
    loss = torch.clamp_min(margin + dist_pos - dist_neg, 0)
    return _apply_loss_reduction(loss, reduction)


def _hardtanh(
    a: TensorLikeType,
    min_val: NumberType = -1,
    max_val: NumberType = 1,
    inplace: bool = False,
    *,
    check_bounds: bool,
) -> TensorLikeType:
    if inplace:
        raise NotImplementedError
    if utils.is_boolean_dtype(a.dtype):
        raise NotImplementedError("Bool inputs not supported for hardtanh")

    # preserve legacy behavior of boundaries not causing type promotion
    if utils.is_integer_dtype(a.dtype):
        min_val = int(min_val)  # type: ignore[arg-type]
        max_val = int(max_val)  # type: ignore[arg-type]
        if not (a.dtype != torch.uint8 or (min_val >= 0 and max_val >= 0)):
            raise RuntimeError(
                "Cannot do hardtanh on an unsigned type with negative limits"
            )

    if check_bounds and min_val > max_val:  # type: ignore[operator]
        raise ValueError("min_val cannot be greater than max_val")

    return torch.clamp(a, min_val, max_val)  # type: ignore[arg-type]


@register_decomposition([aten.hardtanh.default, aten.hardtanh.out])
@_inplace_wrapper
@out_wrapper()
@elementwise_unary_scalar_wrapper
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def _aten_hardtanh(
    a: TensorLikeType,
    min_val: NumberType = -1,
    max_val: NumberType = 1,
    inplace: bool = False,
) -> TensorLikeType:
    return _hardtanh(
        a, min_val=min_val, max_val=max_val, inplace=inplace, check_bounds=False
    )


@_inplace_wrapper
@out_wrapper()
@elementwise_unary_scalar_wrapper
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def hardtanh(
    a: TensorLikeType,
    min_val: NumberType = -1,
    max_val: NumberType = 1,
    inplace: bool = False,
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.hardtanh
    """
    return _hardtanh(
        a, min_val=min_val, max_val=max_val, inplace=inplace, check_bounds=True
    )


@register_decomposition(aten.gelu)
@out_wrapper()
@elementwise_unary_scalar_wrapper
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def gelu(a: TensorLikeType, approximate: str = "none") -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.gelu
    """
    if not isinstance(a, TensorLike):
        raise RuntimeError(
            "Expected a tensor input for an elementwise unary operation!"
        )
    M_SQRT2 = 1.41421356237309504880
    M_SQRT1_2 = 0.70710678118654752440
    M_2_SQRTPI = 1.12837916709551257390
    if approximate == "tanh":
        kBeta = M_SQRT2 * M_2_SQRTPI * 0.5
        kKappa = 0.044715
        a_cube = a * a * a
        inner = kBeta * (a + kKappa * a_cube)
        return 0.5 * a * (1 + torch.tanh(inner))
    elif approximate == "none":
        kAlpha = M_SQRT1_2
        return a * 0.5 * (1 + torch.erf(a * kAlpha))
    else:
        raise RuntimeError("approximate argument must be either none or tanh.")


# CompositeImplicitAutograd - don't register decomp
@elementwise_type_promotion_wrapper(
    type_promoting_args=("input", "target"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def poisson_nll_loss(
    input: TensorLikeType,
    target: TensorLikeType,
    log_input: bool = True,
    full: bool = False,
    size_average: bool | None = None,
    eps: float = 1e-8,
    reduce: bool | None = None,
    reduction: str = "mean",
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.poisson_nll_loss
    """
    if size_average is not None or reduce is not None:
        # TODO: Raise exception instead of converting value.  This is only for
        # primTorch since it can drop support for deprecated arguments.
        # msg = "size_average and reduce args are deprecated, please use reduction argument."
        reduction = _get_string_reduction_arg(size_average=size_average, reduce=reduce)
    _check_reduction_value(reduction)
    if log_input:
        loss = torch.exp(input) - target * input
    else:
        loss = input - target * torch.log(input + eps)

    if full:
        stirling_term = (
            target * torch.log(target) - target + 0.5 * torch.log(2 * torch.pi * target)
        )
        # avoid inplace add
        loss = loss + stirling_term.masked_fill(target <= 1, 0)
    return _apply_loss_reduction(loss, reduction)


@register_decomposition(aten.prelu)
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "weight"),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def prelu(a: TensorLikeType, weight: TensorLikeType) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.prelu
    """
    torch._check(
        isinstance(a, TensorLike),
        lambda: f"prelu: Expected `a` to be tensor, but got: {type(a)}",
    )
    torch._check(
        isinstance(weight, TensorLike),
        lambda: f"prelu: Expected `weight` to be tensor, but got: {type(weight)}",
    )

    if weight.numel() != 1:
        torch._check(a.ndim > 0, lambda: "Not allow zero-dim input tensor.")
        channel_size = a.shape[1] if a.ndim >= 2 else 1
        torch._check(
            weight.numel() == channel_size,
            lambda: f"Mismatch of parameter numbers and input channel size. Found parameter numbers ="
            f" {weight.numel()} and channel size = {channel_size}.",
        )

    torch._check(
        weight.ndim == 0 or weight.ndim == 1,
        lambda: f"prelu: Expected `weight` to be a scalar or 1D tensor, but got: "
        f"ndim = {weight.ndim}",
    )
    if a.ndim == 0:
        weight = weight[0] if weight.ndim == 1 else weight
    else:
        weight = prims.broadcast_in_dim(
            weight, a.shape, () if weight.ndim == 0 else (0 if a.ndim == 1 else 1,)
        )

    return torch.where(a > 0, a, a * weight)


@register_decomposition(aten.relu6)
@_inplace_wrapper
@out_wrapper()
def relu6(a: TensorLikeType, inplace: bool = False) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.relu6
    """
    if inplace:
        raise NotImplementedError

    # See https://github.com/pytorch/pytorch/pull/81142#discussion_r918220126
    # It may be better to use clamp here, but we use hardtanh to replicate
    # the behavior of the existing implementation
    return torch.nn.functional.hardtanh(a, 0, 6)


@register_decomposition(aten.glu)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def glu(a: TensorLikeType, dim: int = -1) -> TensorLikeType:
    dim = utils.canonicalize_dims(a.ndim, dim)
    torch._check(
        a.shape[dim] % 2 == 0,
        lambda: f"Halving dimension must be even, but dimension {dim} is size {a.shape[dim]}",
    )
    b, c = torch.tensor_split(a, 2, dim)

    return b * torch.sigmoid(c)


@register_decomposition(aten.pairwise_distance)
@out_wrapper()
def pairwise_distance(
    x1: TensorLikeType,
    x2: TensorLikeType,
    p: NumberType = 2.0,
    eps: NumberType = 1e-6,
    keepdim=False,
) -> TensorLikeType:
    return torch.linalg.vector_norm(x1 - x2 + eps, ord=p, dim=-1, keepdim=keepdim)


@register_decomposition(aten.pdist)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def pdist(a: TensorLikeType, p: float = 2) -> TensorLikeType:
    torch._check(a.ndim == 2, lambda: f"pdist only supports 2D tensors, got: {a.ndim}D")
    torch._check(p >= 0, lambda: "pdist only supports non-negative p values")
    # For p == 2 we can use an efficient implementation, but other values of p
    # require creating a much bigger tensor for an intermediate step
    if p == 2:
        aTa = torch.mm(a, a.T)
        aTa_diag = torch.diag(aTa)
        t = torch.sqrt(torch.clamp(aTa_diag + aTa_diag.unsqueeze(-1) - 2 * aTa, min=0))
    else:
        t = torch.linalg.vector_norm(a.unsqueeze(1) - a, ord=p, dim=2)
    i = torch.triu_indices(t.shape[0], t.shape[1], offset=1, device=a.device)
    return t.flatten().index_select(0, i[0] * t.shape[0] + i[1])


@register_decomposition(aten.pixel_shuffle)
@out_wrapper()
def pixel_shuffle(self: Tensor, upscale_factor: int):
    torch._check(
        self.dim() >= 3,
        lambda: f"pixel_shuffle expects input to have at least 3 dimensions, but got input with {self.dim} dimension(s)",
    )
    batch = self.shape[:-3]
    C_out = self.shape[-3] // upscale_factor**2
    HW_out = (self.shape[-2] * upscale_factor, self.shape[-1] * upscale_factor)
    n = len(batch)
    B_dims = range(n)
    C_dim, r1_dim, r2_dim, H_dim, W_dim = range(n, n + 5)
    return (
        self.view(
            *batch,
            C_out,
            upscale_factor,
            upscale_factor,
            self.shape[-2],
            self.shape[-1],
        )
        .permute(*B_dims, C_dim, H_dim, r1_dim, W_dim, r2_dim)
        .reshape(*batch, C_out, *HW_out)
        .clone(memory_format=utils.suggest_memory_format(self))
    )


@register_decomposition(aten.pixel_unshuffle)
@out_wrapper()
def pixel_unshuffle(self: Tensor, downscale_factor: int):
    torch._check(
        self.dim() >= 3,
        lambda: f"pixel_unshuffle expects input to have at least 3 dimensions, but got input with {self.dim} dimension(s)",
    )
    batch = self.shape[:-3]
    C_out = self.shape[-3] * downscale_factor**2
    HW_out = (self.shape[-2] // downscale_factor, self.shape[-1] // downscale_factor)
    n = len(batch)
    B_dims = range(n)
    C_dim, H_dim, r1_dim, W_dim, r2_dim = range(n, n + 5)
    return (
        self.view(
            *batch,
            self.shape[-3],
            HW_out[0],
            downscale_factor,
            HW_out[1],
            downscale_factor,
        )
        .permute(*B_dims, C_dim, r1_dim, r2_dim, H_dim, W_dim)
        .reshape(*batch, C_out, *HW_out)
        .clone(memory_format=utils.suggest_memory_format(self))
    )


# Needed as aten.{celu_,elu_...} exist (even if they don't have the in-place kwarg)
celu_ = _make_inplace(celu)
elu_ = _make_inplace(elu)
mish_ = _make_inplace(mish)
selu_ = _make_inplace(selu)
threshold_ = _make_inplace(threshold)
<|endoftext|># mypy: allow-untyped-defs
import math
from typing import Optional, Union

import torch
import torch._prims as prims
import torch._prims_common as utils
import torch._refs as refs
from torch import Tensor
from torch._decomp import register_decomposition
from torch._prims_common import (
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    Number,
    NumberType,
    TensorLike,
    TensorLikeType,
)
from torch._prims_common.wrappers import elementwise_type_promotion_wrapper, out_wrapper
from torch._refs import (
    _make_alias,
    _make_elementwise_binary_reference,
    _make_elementwise_unary_reference,
)


__all__ = [
    "bessel_j0",
    "bessel_j1",
    "entr",
    "erfcx",
    "expit",
    "i0e",
    "i1",
    "i1e",
    "log_ndtr",
    "logit",
    "log_softmax",
    "multigammaln",
    "ndtr",
    "ndtri",
    "softmax",
    "spherical_bessel_j0",
    "xlog1py",
    "zeta",
]
aten = torch._ops.ops.aten


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def bessel_j0(a: TensorLikeType) -> TensorLikeType:
    return prims.bessel_j0(a)


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def bessel_j1(a: TensorLikeType) -> TensorLikeType:
    return prims.bessel_j1(a)


@register_decomposition(aten.special_entr)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def entr(a: TensorLikeType) -> TensorLikeType:
    return torch.where(
        torch.isnan(a),
        a,
        torch.where(a > 0, -a * torch.log(a), torch.where(a == 0, 0, -torch.inf)),
    )


@register_decomposition(aten.special_erfcx)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def erfcx(a: TensorLikeType) -> TensorLikeType:
    return prims.erfcx(a)


# alias for sigmoid
expit = _make_alias(torch.sigmoid, "expit")


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def i0e(a: TensorLikeType) -> TensorLikeType:
    return prims.bessel_i0e(a)


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def i1(a: TensorLikeType) -> TensorLikeType:
    return prims.bessel_i1(a)


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def i1e(a: TensorLikeType) -> TensorLikeType:
    return prims.bessel_i1e(a)


@register_decomposition(aten.special_log_ndtr)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def log_ndtr(a: TensorLikeType) -> TensorLikeType:
    # Note: M_SQRT1_2 is the value of 1 / sqrt(2)
    M_SQRT1_2 = 0.707106781186547524400844362104849039
    t = a * M_SQRT1_2
    return torch.where(
        a < 1.0,
        torch.log(torch.special.erfcx(-t) / 2) - t * t,
        torch.log1p(-torch.erfc(t) / 2),
    )


@register_decomposition(aten.logit)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def logit(self: TensorLikeType, eps: float | None = None) -> TensorLikeType:
    if eps is None:
        eps = -1.0
    lo = eps
    hi = 1 - eps
    self = torch.where(self < lo, lo, torch.where(self > hi, hi, self))
    return torch.log(torch.true_divide(self, torch.sub(1, self)))


@register_decomposition(aten.special_xlog1py)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "b"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def xlog1py(a: TensorLikeType | NumberType, b: TensorLikeType | NumberType):
    torch._check(
        isinstance(a, TensorLike) or isinstance(b, TensorLike),
        lambda: 'Expected either argument a or b to be a Tensor"',
    )

    # Operations like eq and log do not handle scalar values, so we convert them to scalar_tensors.
    if isinstance(a, TensorLike) and isinstance(b, Number):
        # pyrefly: ignore [bad-argument-type]
        b = refs.scalar_tensor(b, dtype=a.dtype, device=a.device)
    elif isinstance(b, TensorLike) and isinstance(a, Number):
        # pyrefly: ignore [bad-argument-type]
        a = refs.scalar_tensor(a, dtype=b.dtype, device=b.device)

    # mypy: expected "Tensor"
    if not isinstance(a, TensorLike):
        raise AssertionError(f"a must be TensorLike, got {type(a)}")
    if not isinstance(b, TensorLike):
        raise AssertionError(f"b must be TensorLike, got {type(b)}")
    rhs = torch.where(torch.eq(a, 0), 0, torch.mul(a, torch.log1p(b)))
    return torch.where(torch.isnan(b), float("nan"), rhs)


@register_decomposition(aten.mvlgamma)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def multigammaln(a: TensorLikeType, p: int) -> TensorLikeType:
    c = 0.25 * p * (p - 1) * math.log(math.pi)
    b = 0.5 * torch.arange(start=(1 - p), end=1, step=1, dtype=a.dtype, device=a.device)
    return torch.sum(torch.lgamma(a.unsqueeze(-1) + b), dim=-1) + c


@register_decomposition(aten.special_ndtr)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def ndtr(a: TensorLikeType) -> TensorLikeType:
    # Note: M_SQRT1_2 is the value of 1 / sqrt(2)
    M_SQRT1_2 = 0.707106781186547524400844362104849039
    a_sqrt_2 = a * M_SQRT1_2
    return (1 + torch.erf(a_sqrt_2)) * 0.5


@register_decomposition(aten.special_ndtri)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def ndtri(a: TensorLikeType) -> TensorLikeType:
    return prims.ndtri(a)


# Forwarding alias: the special variant doesn't support the out kwarg
# CompositeImplicitAutograd - don't register decomp
def log_softmax(
    a: TensorLikeType,
    dim: int,
    dtype: torch.dtype | None = None,
) -> TensorLikeType:
    return torch.log_softmax(a=a, dim=dim, dtype=dtype)  # type: ignore[call-overload]


# Forwarding alias: the special variant doesn't support the out kwarg
# CompositeImplicitAutograd - don't register decomp
def softmax(
    a: TensorLikeType,
    dim: int,
    dtype: torch.dtype | None = None,
) -> TensorLikeType:
    return torch.softmax(a=a, dim=dim, dtype=dtype)  # type: ignore[call-overload]


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def spherical_bessel_j0(a: TensorLikeType) -> TensorLikeType:
    return prims.spherical_bessel_j0(a)


# TODO: add docstring
@_make_elementwise_binary_reference(
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def zeta(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.zeta(a, b)
<|endoftext|># mypy: allow-untyped-defs
import math
from functools import partial
from typing import Optional, Union

import torch
import torch._prims as prims
import torch._prims_common as utils
import torch._refs as refs
import torch._refs.linalg as linalg
from torch import Tensor
from torch._prims_common import (
    check_fp_or_complex,
    check_is_matrix,
    Dim,
    DimsType,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    IntLike,
    TensorLikeType,
)
from torch._prims_common.wrappers import (
    _maybe_convert_to_dtype,
    elementwise_type_promotion_wrapper,
    out_wrapper,
)


__all__ = [
    "diagonal",
    "matrix_norm",
    "norm",
    "svd",
    "svdvals",
    "vector_norm",
    "vecdot",
    "cross",
]


def _check_norm_dtype(dtype: torch.dtype | None, x_dtype: torch.dtype, fn_name: str):
    """
    Checks related to the dtype kwarg in `linalg.*norm` functions
    """
    if dtype is not None:
        torch._check(
            utils.is_float_dtype(dtype) or utils.is_complex_dtype(dtype),
            lambda: f"{fn_name}: dtype should be floating point or complex. Got {dtype}",
        )
        torch._check(
            utils.is_complex_dtype(dtype) == utils.is_complex_dtype(x_dtype),
            lambda: "{fn_name}: dtype should be {d} for {d} inputs. Got {dtype}".format(
                fn_name=fn_name,
                d="complex" if utils.is_complex_dtype(x_dtype) else "real",
                dtype=dtype,
            ),
        )
        torch._check(
            utils.get_higher_dtype(dtype, x_dtype) == dtype,
            lambda: f"{fn_name}: the dtype of the input ({x_dtype}) should be convertible "
            f"without narrowing to the specified dtype ({dtype})",
        )


import operator

# Utilities should come BEFORE this import
from torch._decomp import register_decomposition
from torch._decomp.decompositions import pw_cast_for_opmath


@register_decomposition(torch._ops.ops.aten.linalg_cross)
@out_wrapper()
@pw_cast_for_opmath
def cross(a: Tensor, b: Tensor, dim: int = -1):
    torch._check(
        a.ndim == b.ndim,
        lambda: "linalg.cross: inputs must have the same number of dimensions.",
    )
    torch._check(
        a.size(dim) == 3 and b.size(dim) == 3,
        lambda: f"linalg.cross: inputs dim {dim} must have length 3, got {a.size(dim)} and {b.size(dim)}",
    )
    a, b = torch.broadcast_tensors(a, b)
    dim = utils.canonicalize_dim(a.ndim, dim)
    idx = torch.arange(3, device=a.device)
    return a.index_select(dim, (idx + 1) % 3) * b.index_select(
        dim, (idx + 2) % 3
    ) - a.index_select(dim, (idx + 2) % 3) * b.index_select(dim, (idx + 1) % 3)


def diagonal(
    input: TensorLikeType,
    *,
    offset: int = 0,
    dim1: int = -2,
    dim2: int = -1,
) -> TensorLikeType:
    return torch.diagonal(input, offset=offset, dim1=dim1, dim2=dim2)


def _check_vector_norm_args(
    x: TensorLikeType, ord: float | int = 2, dim: DimsType | None = None
):
    from torch.fx.experimental.symbolic_shapes import sym_or

    if not (ord < 0.0 or ord == float("inf")):
        return

    torch._check(
        sym_or(
            x.numel() != 0,
            not isinstance(dim, IntLike) and dim is not None and len(dim) != 0,
        ),
        lambda: f"linalg.vector_norm cannot compute the {ord} norm on an empty tensor "
        "because the operation does not have an identity",
    )

    shape = x.shape
    if dim is not None and not isinstance(dim, IntLike):
        for d in dim:
            torch._check(
                sym_or(x.numel() != 0, d < len(shape) and d >= 0 and shape[d] != 0),
                lambda: f"linalg.vector_norm cannot compute the {ord} norm on the "
                f"dimension {d} because this dimension is empty and the "
                "operation does not have an identity",
            )


@register_decomposition(torch._ops.ops.aten.linalg_vector_norm)
@out_wrapper(exact_dtype=True)
def vector_norm(
    x: TensorLikeType,
    ord: float | int = 2,
    dim: DimsType | None = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
) -> Tensor:
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    check_fp_or_complex(x.dtype, "linalg.vector_norm")

    if isinstance(dim, Dim):
        dim = [dim]  # type: ignore[assignment]

    _check_vector_norm_args(x, ord, dim)

    _check_norm_dtype(dtype, x.dtype, "linalg.vector_norm")

    computation_dtype, result_dtype = utils.reduction_dtypes(
        x, utils.REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT, dtype
    )

    to_result_dtype = partial(_maybe_convert_to_dtype, dtype=result_dtype)

    # Implementation
    if ord == 0.0:
        return torch.sum(torch.ne(x, 0.0), dim=dim, keepdim=keepdim, dtype=result_dtype)
    elif ord == float("inf"):
        return to_result_dtype(torch.amax(torch.abs(x), dim=dim, keepdim=keepdim))  # type: ignore[return-value,arg-type]
    elif ord == float("-inf"):
        return to_result_dtype(torch.amin(torch.abs(x), dim=dim, keepdim=keepdim))  # type: ignore[return-value,arg-type]
    else:
        # From here on the computation dtype is important as the reduction is non-trivial
        x = _maybe_convert_to_dtype(x, computation_dtype)  # type: ignore[assignment]
        is_ord_even = ord % 2 == 0 if isinstance(ord, IntLike) else ord % 2.0 == 0.0
        if dim is not None and not isinstance(dim, IntLike) and len(dim) == 0:
            dim = None
        elif dim is not None:
            dim = utils.canonicalize_dims(x.ndim, dim)

        reduce_sum = partial(torch.sum, dim=dim, keepdim=keepdim)

        if (dim is None and guard_or_false(x.numel() == 1)) or (
            dim is not None
            and (x.ndim > 0 and all(guard_or_false(x.shape[d] == 1) for d in dim))
        ):
            if x.ndim > 64:
                raise RuntimeError(
                    f"Received a tensor with {x.ndim} dimensions, but only tensors with up to 64 dims are supported!"
                )
            x = torch.abs(x)
            if keepdim or x.ndim == 0:
                return to_result_dtype(x).contiguous()
            elif dim is None:
                return to_result_dtype(x).flatten()[0]
            else:
                new_shape = [s for d, s in enumerate(x.shape) if d not in dim]
                return to_result_dtype(x.view(new_shape)).contiguous()

        if not (is_ord_even and utils.is_float_dtype(x.dtype)):
            x = torch.abs(x)
        return to_result_dtype(torch.pow(reduce_sum(torch.pow(x, ord)), 1.0 / ord))  # type: ignore[return-value]


def _backshift_permutation(dim0, dim1, ndim):
    # Auxiliary function for matrix_norm
    # Computes the permutation that moves the two given dimensions to the back
    ret = [i for i in range(ndim) if i != dim0 and i != dim1]
    ret.extend((dim0, dim1))
    return ret


def _inverse_permutation(perm):
    # Given a permutation, returns its inverse. It's equivalent to argsort on an array
    return [i for i, j in sorted(enumerate(perm), key=operator.itemgetter(1))]


# CompositeImplicitAutograd
@out_wrapper(exact_dtype=True)
def matrix_norm(
    A: TensorLikeType,
    ord: float | str = "fro",
    dim: DimsType = (-2, -1),
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
) -> TensorLikeType:
    # shape
    check_is_matrix(A, "linalg.matrix_norm")
    # dim

    dim = utils.canonicalize_dims(A.ndim, dim)
    if isinstance(dim, Dim):
        dim = (dim,)  # type: ignore[assignment]
    torch._check(
        len(dim) == 2, lambda: f"linalg.matrix_norm: dim must be a 2-tuple. Got {dim}"
    )
    torch._check(
        # pyrefly: ignore [bad-index]
        dim[0] != dim[1],
        # pyrefly: ignore [bad-index, index-error]
        # pyrefly: ignore [bad-index, index-error]
        lambda: f"linalg.matrix_norm: dims must be different. Got ({dim[0]}, {dim[1]})",
    )
    # dtype arg
    _check_norm_dtype(dtype, A.dtype, "linalg.matrix_norm")

    if isinstance(ord, str):
        # ord
        torch._check(
            ord in ("fro", "nuc"),
            lambda: f"linalg.matrix_norm: Order {ord} not supported.",
        )
        # dtype
        check_fp_or_complex(
            A.dtype, "linalg.matrix_norm", allow_low_precision_dtypes=ord != "nuc"
        )

        if ord == "fro":
            return vector_norm(A, 2, dim, keepdim, dtype=dtype)
        else:  # ord == "nuc"
            if dtype is not None:
                A = _maybe_convert_to_dtype(A, dtype)  # type: ignore[assignment]
            # pyrefly: ignore [bad-index, index-error]
            perm = _backshift_permutation(dim[0], dim[1], A.ndim)
            result = torch.sum(svdvals(prims.transpose(A, perm)), -1, keepdim)
            if keepdim:
                inv_perm = _inverse_permutation(perm)
                result = prims.transpose(torch.unsqueeze(result, -1), inv_perm)
            return result
    else:
        # ord
        abs_ord = abs(ord)
        torch._check(
            abs_ord in (2, 1, float("inf")),
            lambda: f"linalg.matrix_norm: Order {ord} not supported.",
        )
        # dtype
        check_fp_or_complex(
            A.dtype, "linalg.matrix_norm", allow_low_precision_dtypes=ord != 2
        )

        max_min = partial(torch.amax if ord > 0.0 else torch.amin, keepdim=keepdim)

        def _max_min_wrapper(A, dim):
            # pyrefly: ignore [unsupported-operation]
            if A.size(dim) == 0 and ord > 0.0:
                new_size = list(A.size())
                if keepdim:
                    new_size[dim] = 1
                else:
                    del new_size[dim]
                return torch.zeros(new_size, dtype=A.dtype, device=A.device)
            else:
                return max_min(A, dim)

        if abs_ord == 2.0:
            if dtype is not None:
                A = _maybe_convert_to_dtype(A, dtype)  # type: ignore[assignment]
            # pyrefly: ignore [bad-index, index-error]
            perm = _backshift_permutation(dim[0], dim[1], A.ndim)
            result = _max_min_wrapper(svdvals(prims.transpose(A, perm)), dim=-1)
            if keepdim:
                inv_perm = _inverse_permutation(perm)
                result = prims.transpose(torch.unsqueeze(result, -1), inv_perm)
            return result
        else:  # 1, -1, inf, -inf
            # pyrefly: ignore [bad-unpacking]
            dim0, dim1 = dim
            if abs_ord == float("inf"):
                dim0, dim1 = dim1, dim0
            if not keepdim and (dim0 < dim1):
                dim1 -= 1
            return _max_min_wrapper(
                vector_norm(A, 1.0, dim=dim0, keepdim=keepdim, dtype=dtype), dim1
            )


# CompositeImplicitAutograd
@out_wrapper(exact_dtype=True)
def norm(
    A: TensorLikeType,
    ord: float | str | None = None,
    dim: DimsType | None = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
) -> TensorLikeType:
    if dim is not None:
        if isinstance(dim, Dim):
            dim = (dim,)  # type: ignore[assignment]
        torch._check(
            len(dim) in (1, 2),
            lambda: f"linalg.norm: If dim is specified, it must be of length 1 or 2. Got {dim}",
        )
    elif ord is not None:
        torch._check(
            A.ndim in (1, 2),
            lambda: f"linalg.norm: If dim is not specified but ord is, the input must be 1D or 2D. Got {A.ndim}D",
        )

    if ord is not None and (
        (dim is not None and len(dim) == 2) or (dim is None and A.ndim == 2)
    ):
        if dim is None:
            dim = (0, 1)
        return matrix_norm(A, ord, dim, keepdim, dtype=dtype)
    else:
        if ord is None:
            ord = 2.0
        return vector_norm(A, ord, dim, keepdim, dtype=dtype)  # type: ignore[arg-type]


# CompositeImplicitAutograd
@out_wrapper("U", "S", "Vh", exact_dtype=True)
def svd(A: TensorLikeType, full_matrices: bool = True) -> tuple[Tensor, Tensor, Tensor]:
    return prims.svd(A, full_matrices=full_matrices)


# CompositeImplicitAutograd
@out_wrapper(exact_dtype=True)
def svdvals(A: TensorLikeType) -> Tensor:
    return svd(A, full_matrices=False)[1]


# CompositeImplicitAutograd
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("x", "y"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def vecdot(x: Tensor, y: Tensor, dim: int = -1) -> Tensor:
    check_fp_or_complex(x.dtype, "linalg.vecdot")
    return (x.conj() * y).sum(dim=dim)
<|endoftext|># torch.ao is a package with a lot of interdependencies.
# We will use lazy import to avoid cyclic dependencies here.

from typing import TYPE_CHECKING as _TYPE_CHECKING


if _TYPE_CHECKING:
    from types import ModuleType

    from torch.ao import (
        nn as nn,
        ns as ns,
        pruning as pruning,
        quantization as quantization,
    )


__all__ = [
    "nn",
    "ns",
    "pruning",
    "quantization",
]


def __getattr__(name: str) -> "ModuleType":
    if name in __all__:
        import importlib

        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
<|endoftext|># mypy: allow-untyped-defs

import torch
from torch.nn.parameter import Parameter


__all__: list[str] = []


class _LearnableFakeQuantize(torch.ao.quantization.FakeQuantizeBase):
    r"""Generalized extension of the FakeQuantize module in fake_quantize.py.

    This is an extension of the FakeQuantize module in fake_quantize.py, which
    supports more generalized lower-bit quantization and supports learning of the scale
    and zero point parameters through backpropagation.

    In addition to the attributes in the original FakeQuantize module, the _LearnableFakeQuantize
    module also includes the following attributes to support quantization parameter learning.

    * :attr:`channel_len` defines the length of the channel when initializing scale and zero point
      for the per channel case.

    * :attr:`use_grad_scaling` defines the flag for whether the gradients for scale and zero point are
      normalized by the constant, which is proportional to the square root of the number of
      elements in the tensor. The related literature justifying the use of this particular constant
      can be found here: https://openreview.net/pdf?id=rkgO66VKDS.

    * :attr:`fake_quant_enabled` defines the flag for enabling fake quantization on the output.

    * :attr:`static_enabled` defines the flag for using observer's static estimation for
      scale and zero point.

    * :attr:`learning_enabled` defines the flag for enabling backpropagation for scale and zero point.
    """

    def __init__(
        self,
        observer,
        quant_min=0,
        quant_max=255,
        scale=1.0,
        zero_point=0.0,
        channel_len=-1,
        use_grad_scaling=False,
        **observer_kwargs,
    ):
        super().__init__()
        if quant_min >= quant_max:
            raise AssertionError("quant_min must be strictly less than quant_max.")
        self.quant_min = quant_min
        self.quant_max = quant_max
        # also pass quant_min and quant_max to observer
        observer_kwargs["quant_min"] = quant_min
        observer_kwargs["quant_max"] = quant_max
        self.use_grad_scaling = use_grad_scaling
        if channel_len == -1:
            self.scale = Parameter(torch.tensor([scale]))
            self.zero_point = Parameter(torch.tensor([zero_point]))
        else:
            if not (isinstance(channel_len, int) and channel_len > 0):
                raise AssertionError("Channel size must be a positive integer.")
            self.scale = Parameter(torch.tensor([scale] * channel_len))
            self.zero_point = Parameter(torch.tensor([zero_point] * channel_len))

        self.activation_post_process = observer(**observer_kwargs)
        if torch.iinfo(self.activation_post_process.dtype).min > quant_min:
            raise AssertionError("quant_min out of bound")
        if quant_max > torch.iinfo(self.activation_post_process.dtype).max:
            raise AssertionError("quant_max out of bound")
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        self.ch_axis = (
            self.activation_post_process.ch_axis
            if hasattr(self.activation_post_process, "ch_axis")
            else -1
        )
        self.register_buffer("fake_quant_enabled", torch.tensor([1], dtype=torch.uint8))
        self.register_buffer("static_enabled", torch.tensor([1], dtype=torch.uint8))
        self.register_buffer("learning_enabled", torch.tensor([0], dtype=torch.uint8))

        bitrange = torch.tensor(quant_max - quant_min + 1).double()
        self.bitwidth = int(torch.log2(bitrange).item())
        self.register_buffer("eps", torch.tensor([torch.finfo(torch.float32).eps]))

    @torch.jit.export
    def enable_param_learning(self):
        r"""Enable parameter learning over static observer estimates.

        Enables learning of quantization parameters and
        disables static observer estimates. Forward path returns fake quantized X.
        """
        self.toggle_qparam_learning(enabled=True).toggle_fake_quant(
            enabled=True
        ).toggle_observer_update(enabled=False)
        return self

    @torch.jit.export
    def enable_static_estimate(self):
        """Enable static estimates of quantization parameters.

        Enables static observer estimates and disables learning of
        quantization parameters. Forward path returns fake quantized X.
        """
        self.toggle_qparam_learning(enabled=False).toggle_fake_quant(
            enabled=True
        ).toggle_observer_update(enabled=True)

    @torch.jit.export
    def enable_static_observation(self):
        """Enable accumulation of data without updating quantization parameters.

        Enables static observer accumulating data from input but doesn't
        update the quantization parameters. Forward path returns the original X.
        """
        self.toggle_qparam_learning(enabled=False).toggle_fake_quant(
            enabled=False
        ).toggle_observer_update(enabled=True)

    @torch.jit.export
    def toggle_observer_update(self, enabled=True):
        self.static_enabled[0] = int(enabled)  # type: ignore[operator]
        return self

    @torch.jit.export
    def enable_observer(self, enabled=True):
        self.toggle_observer_update(enabled)

    @torch.jit.export
    def toggle_qparam_learning(self, enabled=True):
        self.learning_enabled[0] = int(enabled)  # type: ignore[operator]
        self.scale.requires_grad = enabled
        self.zero_point.requires_grad = enabled
        return self

    @torch.jit.export
    def toggle_fake_quant(self, enabled=True):
        self.fake_quant_enabled[0] = int(enabled)
        return self

    @torch.jit.export
    def observe_quant_params(self):
        print(f"_LearnableFakeQuantize Scale: {self.scale.detach()}")
        print(f"_LearnableFakeQuantize Zero Point: {self.zero_point.detach()}")

    @torch.jit.export
    def calculate_qparams(self):  # type: ignore[override]
        self.scale.data.clamp_(min=self.eps.item())  # type: ignore[operator]
        scale = self.scale.detach()
        zero_point = (
            self.zero_point.detach()
            .round()
            .clamp(self.quant_min, self.quant_max)
            .long()
        )
        return scale, zero_point

    def forward(self, X):
        if self.static_enabled[0] == 1:  # type: ignore[index]
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.activation_post_process.calculate_qparams()
            _scale = _scale.to(self.scale.device)
            _zero_point = _zero_point.to(self.zero_point.device)
            self.scale.data.copy_(_scale)
            self.zero_point.data.copy_(_zero_point)
        else:
            self.scale.data.clamp_(min=self.eps.item())  # type: ignore[operator]

        if self.fake_quant_enabled[0] == 1:
            if self.qscheme in (
                torch.per_channel_symmetric,
                torch.per_tensor_symmetric,
            ):
                self.zero_point.data.zero_()

            if self.use_grad_scaling:
                grad_factor = 1.0 / (X.numel() * self.quant_max) ** 0.5
            else:
                grad_factor = 1.0
            if self.qscheme in (torch.per_channel_symmetric, torch.per_channel_affine):
                X = torch._fake_quantize_learnable_per_channel_affine(
                    X,
                    self.scale,
                    self.zero_point,
                    self.ch_axis,
                    self.quant_min,
                    self.quant_max,
                    grad_factor,
                )
            else:
                X = torch._fake_quantize_learnable_per_tensor_affine(
                    X,
                    self.scale,
                    self.zero_point,
                    self.quant_min,
                    self.quant_max,
                    grad_factor,
                )

        return X
<|endoftext|># mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
"""Implements modules  used to perform fake quantization."""

import re
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.ao.quantization.observer import (
    _with_args,
    default_fixed_qparams_range_0to1_observer,
    default_fixed_qparams_range_neg1to1_observer,
    FixedQParamsObserver,
    HistogramObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)
from torch.nn import Module


__all__ = [
    "FakeQuantizeBase",
    "FakeQuantize",
    "FixedQParamsFakeQuantize",
    "FusedMovingAvgObsFakeQuantize",
    "disable_fake_quant",
    "disable_observer",
    "enable_fake_quant",
    "enable_observer",
    "default_fake_quant",
    "default_weight_fake_quant",
    "default_dynamic_fake_quant",
    "default_fixed_qparams_range_neg1to1_fake_quant",
    "default_fixed_qparams_range_0to1_fake_quant",
    "default_symmetric_fixed_qparams_fake_quant",
    "default_affine_fixed_qparams_fake_quant",
    "default_per_channel_weight_fake_quant",
    "default_embedding_fake_quant",
    "default_embedding_fake_quant_4bit",
    "default_histogram_fake_quant",
    "default_fused_act_fake_quant",
    "default_fused_wt_fake_quant",
    "default_fused_per_channel_wt_fake_quant",
    "fused_wt_fake_quant_range_neg_127_to_127",
    "fused_per_channel_wt_fake_quant_range_neg_127_to_127",
]


def _is_per_channel(qscheme: "torch.qscheme") -> bool:
    return qscheme in [
        torch.per_channel_symmetric,
        torch.per_channel_affine,
        torch.per_channel_affine_float_qparams,
    ]


def _is_per_tensor(qscheme: "torch.qscheme") -> bool:
    return qscheme in [torch.per_tensor_symmetric, torch.per_tensor_affine]


def _is_symmetric_quant(qscheme: "torch.qscheme") -> bool:
    return qscheme in [torch.per_tensor_symmetric, torch.per_channel_symmetric]


def _is_float_qparams(qscheme: "torch.qscheme") -> bool:
    return qscheme == torch.per_channel_affine_float_qparams


class FakeQuantizeBase(ABC, Module):
    r"""Base fake quantize module.

    Base fake quantize module
    Any fake quantize implementation should derive from this class.

    Concrete fake quantize module should follow the same API. In forward, they will update
    the statistics of the observed Tensor and fake quantize the input. They should also provide a
    `calculate_qparams` function that computes the quantization parameters given
    the collected statistics.

    """

    fake_quant_enabled: torch.Tensor
    observer_enabled: torch.Tensor

    def __init__(self) -> None:
        """Set fake_quant_enabled and observer_enabled."""
        super().__init__()
        # fake_quant_enabled and observer_enabled are buffers to support their
        # replication in DDP. Data type is uint8 because NCCL does not support
        # bool tensors.
        self.register_buffer("fake_quant_enabled", torch.tensor([1], dtype=torch.uint8))
        self.register_buffer("observer_enabled", torch.tensor([1], dtype=torch.uint8))

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def calculate_qparams(self, **kwargs):
        pass

    @torch.jit.export
    def enable_fake_quant(self, enabled: bool = True) -> None:
        self.fake_quant_enabled[0] = 1 if enabled else 0

    @torch.jit.export
    def disable_fake_quant(self):
        self.enable_fake_quant(False)

    @torch.jit.export
    def enable_observer(self, enabled: bool = True) -> None:
        self.observer_enabled[0] = 1 if enabled else 0

    @torch.jit.export
    def disable_observer(self):
        self.enable_observer(False)

    @classmethod
    def with_args(cls, **kwargs):
        fake_quant_constructor = _with_args(cls, **kwargs)
        # need to assign the correct module to fake_quantize
        # constructors to satisfy public v private requirements
        fake_quant_constructor.__module__ = "torch.ao.quantization.fake_quantize"
        return fake_quant_constructor


class FakeQuantize(FakeQuantizeBase):
    r"""Simulate the quantize and dequantize operations in training time.

    The output of this module is given by::

        x_out = (
            clamp(round(x / scale + zero_point), quant_min, quant_max) - zero_point
        ) * scale

    * :attr:`is_dynamic` indicates whether the fake quantize is a placeholder for dynamic quantization
      operators (choose_qparams -> q -> dq) or static quantization operators (q -> dq)

    * :attr:`scale` defines the scale factor used for quantization.

    * :attr:`zero_point` specifies the quantized value to which 0 in floating point maps to

    * :attr:`fake_quant_enabled` controls the application of fake quantization on tensors, note that
      statistics can still be updated.

    * :attr:`observer_enabled` controls statistics collection on tensors

    * :attr:`dtype` specifies the quantized dtype that is being emulated with fake-quantization,
        allowable values are torch.qint8 and torch.quint8.

    Args:

        observer (module): Module for observing statistics on input tensors and calculating scale
          and zero-point.
        observer_kwargs (optional): Arguments for the observer module

    Attributes:
        activation_post_process (Module): User provided module that collects statistics on the input tensor and
          provides a method to calculate scale and zero-point.

    """

    scale: torch.Tensor
    zero_point: torch.Tensor

    def __init__(
        self,
        observer=MovingAverageMinMaxObserver,
        quant_min=None,
        quant_max=None,
        is_dynamic=False,
        **observer_kwargs,
    ):
        super().__init__()
        # Populate quant_min/quant_max to observer_kwargs if valid
        if quant_min is not None and quant_max is not None:
            if quant_min > quant_max:
                raise AssertionError(
                    "quant_min must be less than or equal to quant_max"
                )
            dtype = observer_kwargs.get("dtype", torch.quint8)
            if hasattr(observer, "p"):
                # In case observer is _PartialWrapper, dtype can be stored in
                # observer.p.keywords["dtype"]
                dtype = getattr(getattr(observer, "p", {}), "keywords", {}).get(
                    "dtype", dtype
                )

            if torch.iinfo(dtype).min > quant_min:
                raise AssertionError("quant_min out of bound")

            if quant_max > torch.iinfo(dtype).max:
                raise AssertionError("quant_max out of bound")
            observer_kwargs.update({"quant_min": quant_min, "quant_max": quant_max})
        observer_kwargs["is_dynamic"] = is_dynamic
        self.activation_post_process = observer(**observer_kwargs)
        # TODO: keeping self.quant_min/max for BC; remove after a couple releases
        # Users should use self.activation_post_process.quant_min
        self.quant_min = self.activation_post_process.quant_min
        self.quant_max = self.activation_post_process.quant_max
        self.is_dynamic = self.activation_post_process.is_dynamic
        if _is_float_qparams(self.activation_post_process.qscheme):
            zero_point_dtype = torch.float
        else:
            zero_point_dtype = torch.int
        self.register_buffer("scale", torch.tensor([1.0], dtype=torch.float))
        self.register_buffer("zero_point", torch.tensor([0], dtype=zero_point_dtype))
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        self.ch_axis = (
            self.activation_post_process.ch_axis
            if hasattr(self.activation_post_process, "ch_axis")
            else -1
        )
        if not (_is_per_channel(self.qscheme) or _is_per_tensor(self.qscheme)):
            raise AssertionError(
                "Only per channel and per tensor quantization are supported in fake quantize"
                + " got qscheme: "
                + str(self.qscheme)
            )
        self.is_per_channel = _is_per_channel(self.qscheme)

    @torch.jit.export
    def calculate_qparams(self):  # type: ignore[override]
        return self.activation_post_process.calculate_qparams()

    def forward(self, X):
        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.calculate_qparams()
            _scale, _zero_point = (
                _scale.to(self.scale.device),
                _zero_point.to(self.zero_point.device),
            )
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled[0] == 1:
            if self.is_per_channel:
                X = torch.fake_quantize_per_channel_affine(
                    X,
                    self.scale,
                    self.zero_point,
                    self.ch_axis,
                    self.activation_post_process.quant_min,
                    self.activation_post_process.quant_max,
                )
            else:
                X = torch.fake_quantize_per_tensor_affine(
                    X,
                    self.scale,
                    self.zero_point,
                    self.activation_post_process.quant_min,
                    self.activation_post_process.quant_max,
                )
        return X

    @torch.jit.export
    def extra_repr(self):
        return (
            f"fake_quant_enabled={self.fake_quant_enabled}, observer_enabled={self.observer_enabled}, "
            f"quant_min={self.activation_post_process.quant_min}, quant_max={self.activation_post_process.quant_max}, "
            f"dtype={self.dtype}, qscheme={self.qscheme}, ch_axis={self.ch_axis}, "
            f"scale={self.scale}, zero_point={self.zero_point}"
        )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # We cannot currently register scalar values as buffers, so need to manually
        # specify serialization here.
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "scale"] = self.scale
        destination[prefix + "zero_point"] = self.zero_point

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
        # Removing this function throws an error that the size of the loaded tensor does not match the original size
        # i.e., These buffers start out with numel 0 and become numel 1 once they have their first forward pass.
        local_state = ["scale", "zero_point"]
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                # Custom handling to allow loading scale and zero_point
                # of size N into uninitialized buffers of size 0. The
                # buffers are resized here, and the values are copied in
                # the default state_dict loading code of the parent.
                if name == "scale":
                    self.scale.resize_(val.shape)
                else:
                    if name != "zero_point":
                        raise AssertionError(
                            "Expected 'zero_point' but got different state key"
                        )
                    self.zero_point.resize_(val.shape)
                # For torchscript module we need to update the attributes here since we do not
                # call the `_load_from_state_dict` function defined module.py
                if torch.jit.is_scripting():
                    if name == "scale":
                        self.scale.copy_(val)
                    else:
                        if name != "zero_point":
                            raise AssertionError(
                                "Expected 'zero_point' but got different state key"
                            )
                        self.zero_point.copy_(val)
            elif strict:
                missing_keys.append(key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class FixedQParamsFakeQuantize(FakeQuantize):
    """Simulate quantize and dequantize in training time.

    Simulate quantize and dequantize with fixed quantization
    parameters in training time. Only per tensor quantization
    is supported.
    """

    # TODO: rename observer to observer_ctr
    def __init__(self, observer):
        super().__init__(observer=observer)
        if type(self.activation_post_process) is not FixedQParamsObserver:
            raise AssertionError(
                f"{self.__class__.__name__}'s observer must be a {FixedQParamsObserver.__name__}"
            )
        self._observer_ctr = observer
        self.scale = self.activation_post_process.scale
        self.zero_point = self.activation_post_process.zero_point
        if not _is_per_tensor(self.qscheme):
            raise AssertionError(
                "Only per tensor quantization is supported"
                + " FixedQParamsFakeQuantize module, got qscheme:"
                + str(self.qscheme)
            )

    @torch.jit.export
    def calculate_qparams(self):  # type: ignore[override]
        return self.scale, self.zero_point

    @torch.jit.export
    def extra_repr(self):
        """Define a string representation of the object's attributes."""
        return (
            f"fake_quant_enabled={self.fake_quant_enabled}, observer_enabled={self.observer_enabled}, "
            f"scale={self.scale}, zero_point={self.zero_point}, "
            f"dtype={self.dtype}, quant_min={self.activation_post_process.quant_min}, "
            f"quant_max={self.activation_post_process.quant_max}, qscheme={self.qscheme}"
        )


class FusedMovingAvgObsFakeQuantize(FakeQuantize):
    r"""Define a fused module to observe the tensor.

    Fused module that is used to observe the input tensor (compute min/max), compute
    scale/zero_point and fake_quantize the tensor.
    This module uses calculation similar MovingAverageMinMaxObserver for the inputs,
    to compute the min/max values in order to compute the scale/zero_point.
    The qscheme input in the observer is used to differentiate between symmetric/affine
    quantization scheme.

    The output of this module is given by
    x_out = (clamp(round(x/scale + zero_point), quant_min, quant_max)-zero_point)*scale

    Similar to :class:`~torch.ao.quantization.FakeQuantize`, and accepts the same attributes as the
    base class.

    """

    def __init__(
        self,
        observer: Any = MovingAverageMinMaxObserver,
        quant_min: int = 0,
        quant_max: int = 255,
        **observer_kwargs: Any,
    ) -> None:
        super().__init__(observer, quant_min, quant_max, **observer_kwargs)
        if not isinstance(
            self.activation_post_process,
            (MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver),
        ):
            raise AssertionError(
                "Fused observer+fake_quant module only works with MovingAverageMinMaxObserver"
            )
        self.register_buffer("fake_quant_enabled", torch.tensor([1], dtype=torch.long))
        self.register_buffer("observer_enabled", torch.tensor([1], dtype=torch.long))
        self.is_symmetric_quant = _is_symmetric_quant(
            self.activation_post_process.qscheme
        )

    @torch.jit.export
    def calculate_qparams(self) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        return self.activation_post_process.calculate_qparams()

    @torch.jit.export
    def extra_repr(self) -> str:
        return (
            f"fake_quant_enabled={self.fake_quant_enabled}, observer_enabled={self.observer_enabled}, "
            f"scale={self.scale}, zero_point={self.zero_point}, dtype={self.dtype}, "
            f"quant_min={self.activation_post_process.quant_min}, quant_max={self.activation_post_process.quant_max}, "
            f"qscheme={self.qscheme}, reduce_range={self.activation_post_process.reduce_range}"
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return torch.fused_moving_avg_obs_fake_quant(
            X,
            self.observer_enabled,
            self.fake_quant_enabled,
            self.activation_post_process.min_val,
            self.activation_post_process.max_val,
            self.scale,
            self.zero_point,
            self.activation_post_process.averaging_constant,
            self.activation_post_process.quant_min,
            self.activation_post_process.quant_max,
            self.ch_axis,
            self.is_per_channel,
            self.is_symmetric_quant,
        )


default_fake_quant = FakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=0,
    quant_max=255,
    dtype=torch.quint8,
    qscheme=torch.per_tensor_affine,
    reduce_range=True,
)
"""
Default fake_quant for activations.
"""

default_weight_fake_quant = FakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=-128,
    quant_max=127,
    dtype=torch.qint8,
    qscheme=torch.per_tensor_symmetric,
    reduce_range=False,
)
"""
Default fake_quant for weights.
Observer is memoryless since averaging_constant is 1.
"""

default_dynamic_fake_quant = FakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=0,
    quant_max=255,
    is_dynamic=True,
    dtype=torch.quint8,
    averaging_constant=1,
)
"""
Default dynamic fake_quant for activations.
"""

default_fixed_qparams_range_neg1to1_fake_quant = FixedQParamsFakeQuantize.with_args(
    observer=default_fixed_qparams_range_neg1to1_observer
)
default_fixed_qparams_range_0to1_fake_quant = FixedQParamsFakeQuantize.with_args(
    observer=default_fixed_qparams_range_0to1_observer
)
# TODO: the following 2 variables are kept for backwards compatibility; remove after a few releases
default_symmetric_fixed_qparams_fake_quant = (
    default_fixed_qparams_range_neg1to1_fake_quant
)
default_affine_fixed_qparams_fake_quant = default_fixed_qparams_range_0to1_fake_quant

default_per_channel_weight_fake_quant = FakeQuantize.with_args(
    observer=MovingAveragePerChannelMinMaxObserver,
    quant_min=-128,
    quant_max=127,
    dtype=torch.qint8,
    qscheme=torch.per_channel_symmetric,
    reduce_range=False,
    ch_axis=0,
)
"""
Default fake_quant for per-channel weights.
Observer is memoryless since averaging_constant is 1.
"""
default_embedding_fake_quant = FakeQuantize.with_args(
    observer=MovingAveragePerChannelMinMaxObserver,
    qscheme=torch.per_channel_affine_float_qparams,
    dtype=torch.quint8,
    quant_min=0,
    quant_max=255,
    ch_axis=0,
    averaging_constant=1,
)
"""
Default fake_quant for embeddings.
Observer is memoryless since averaging_constant is 1.
"""

default_embedding_fake_quant_4bit = FakeQuantize.with_args(
    observer=MovingAveragePerChannelMinMaxObserver,
    qscheme=torch.per_channel_affine_float_qparams,
    ch_axis=0,
    dtype=torch.quint4x2,
    averaging_constant=1,
)

default_histogram_fake_quant = FakeQuantize.with_args(
    observer=HistogramObserver,
    quant_min=0,
    quant_max=255,
    dtype=torch.quint8,
    qscheme=torch.per_tensor_affine,
    reduce_range=True,
)
"""
Fake_quant for activations using a histogram..
"""


default_fused_act_fake_quant = FusedMovingAvgObsFakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=0,
    quant_max=255,
    dtype=torch.quint8,
)

"""
Fused version of `default_fake_quant`, with improved performance.
"""


default_fused_wt_fake_quant = FusedMovingAvgObsFakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=-128,
    quant_max=127,
    dtype=torch.qint8,
    qscheme=torch.per_tensor_symmetric,
)
"""
Fused version of `default_weight_fake_quant`, with improved performance.
"""

default_fused_per_channel_wt_fake_quant = FusedMovingAvgObsFakeQuantize.with_args(
    observer=MovingAveragePerChannelMinMaxObserver,
    quant_min=-128,
    quant_max=127,
    dtype=torch.qint8,
    qscheme=torch.per_channel_symmetric,
)
"""
Fused version of `default_per_channel_weight_fake_quant`, with improved performance.
"""

fused_wt_fake_quant_range_neg_127_to_127 = FusedMovingAvgObsFakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=-127,
    quant_max=127,
    dtype=torch.qint8,
    qscheme=torch.per_tensor_symmetric,
    eps=2**-12,
)
"""
Fused version of `default_weight_fake_quant`, with the 8-bit values restricted to [-127, +127], excluding -128.
"""

fused_per_channel_wt_fake_quant_range_neg_127_to_127 = (
    FusedMovingAvgObsFakeQuantize.with_args(
        observer=MovingAveragePerChannelMinMaxObserver,
        quant_min=-127,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        eps=2**-12,
    )
)

"""
Fused version of `default_per_channel_weight_fake_quant`, with the 8-bit values restricted to [-127, +127], excluding -128.
"""


def _is_fake_quant_script_module(mod):
    """Return true if given mod is an instance of FakeQuantize script module."""
    if isinstance(mod, torch.jit.RecursiveScriptModule):
        # qualified name looks like '__torch__.torch.ao.quantization.fake_quantize.___torch_mangle_2.FakeQuantize'
        suffix = mod._c.qualified_name.split(".", 1)[1]
        name = re.sub(r"\.___torch_mangle_\d+", "", suffix)
        return (
            name == "torch.ao.quantization.fake_quantize.FakeQuantize"
            or name
            == "torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize"
        )
    return False


def disable_fake_quant(mod):
    """Disable fake quantization for the module.

    Disable fake quantization for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.disable_fake_quant)

    """
    if isinstance(mod, FakeQuantizeBase) or _is_fake_quant_script_module(mod):
        mod.disable_fake_quant()


def enable_fake_quant(mod):
    """Enable fake quantization for the module.

    Enable fake quantization for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.enable_fake_quant)

    """
    if isinstance(mod, FakeQuantizeBase) or _is_fake_quant_script_module(mod):
        mod.enable_fake_quant()


def disable_observer(mod):
    """Disable observation for this module.

    Disable observation for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.disable_observer)

    """
    if isinstance(mod, FakeQuantizeBase) or _is_fake_quant_script_module(mod):
        mod.disable_observer()


def enable_observer(mod):
    """Enable observation for this module.

    Enable observation for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.enable_observer)

    """
    if isinstance(mod, FakeQuantizeBase) or _is_fake_quant_script_module(mod):
        mod.enable_observer()
<|endoftext|>import copy
import typing_extensions
import warnings
from typing import Any

import torch
from torch.fx import GraphModule
from torch.fx.graph_module import _USER_PRESERVED_ATTRIBUTES_KEY

from .backend_config import BackendConfig, get_tensorrt_backend_config  # noqa: F401
from .fx.convert import convert
from .fx.custom_config import ConvertCustomConfig, FuseCustomConfig, PrepareCustomConfig
from .fx.fuse import fuse
from .fx.graph_module import ObservedGraphModule  # noqa: F401
from .fx.prepare import prepare
from .fx.tracer import QuantizationTracer, Scope, ScopeContextManager  # noqa: F401
from .fx.utils import (  # noqa: F401
    get_custom_module_class_keys,
    get_skipped_module_name_and_classes,
)
from .qconfig_mapping import QConfigMapping
from .utils import DEPRECATION_WARNING


def attach_preserved_attrs_to_model(
    model: GraphModule | torch.nn.Module,
    preserved_attrs: dict[str, Any],
) -> None:
    """Store preserved attributes to the model.meta so that it can be preserved during deepcopy"""
    model.meta[_USER_PRESERVED_ATTRIBUTES_KEY] = copy.copy(preserved_attrs)  # type: ignore[operator, index, assignment]
    # set the preserved attributes in the model so that user can call
    # model.attr as they do before calling fx graph mode quantization
    for attr_name, attr in model.meta[_USER_PRESERVED_ATTRIBUTES_KEY].items():  # type: ignore[index, union-attr]
        setattr(model, attr_name, attr)


def _check_is_graph_module(model: torch.nn.Module) -> None:
    if not isinstance(model, GraphModule):
        raise ValueError(
            "input model must be a GraphModule, "
            + "Got type:"
            + str(type(model))
            + " Please make "
            + "sure to follow the tutorials."
        )


def _attach_meta_to_node_if_not_exist(model: GraphModule) -> None:
    """Attach meta field to all nodes of the graph if it does not exist,
    meta field is a field stores some meta information about the node, such
    as dtype and shape information for output of the node, this only exists
    if the program is captured by make_fx (used in quantize_pt2e flow), if
    the program is captured by torch.fx symbolic tracing, this field may not exist,
    so we add it here to avoid checking this all over the places
    """
    for node in model.graph.nodes:
        if not hasattr(node, "meta"):
            node.meta = {}


def _swap_ff_with_fxff(model: torch.nn.Module) -> None:
    r"""Swap FloatFunctional with FXFloatFunctional"""
    modules_to_swap = []
    for name, module in model.named_children():
        if isinstance(module, torch.ao.nn.quantized.FloatFunctional):
            modules_to_swap.append(name)
        else:
            _swap_ff_with_fxff(module)

    for name in modules_to_swap:
        del model._modules[name]
        model._modules[name] = torch.ao.nn.quantized.FXFloatFunctional()


def _fuse_fx(
    model: GraphModule,
    is_qat: bool,
    fuse_custom_config: FuseCustomConfig | dict[str, Any] | None = None,
    backend_config: BackendConfig | dict[str, Any] | None = None,
) -> GraphModule:
    r"""Internal helper function to fuse modules in preparation for quantization

    Args:
        model: GraphModule object from symbolic tracing (torch.fx.symbolic_trace)
    """
    _check_is_graph_module(model)
    return fuse(model, is_qat, fuse_custom_config, backend_config)  # type: ignore[operator]


def _prepare_fx(
    model: torch.nn.Module,
    qconfig_mapping: QConfigMapping | dict[str, Any],
    is_qat: bool,
    example_inputs: tuple[Any, ...],
    prepare_custom_config: PrepareCustomConfig | dict[str, Any] | None = None,
    _equalization_config: QConfigMapping | dict[str, Any] | None = None,
    backend_config: BackendConfig | dict[str, Any] | None = None,
    is_standalone_module: bool = False,
) -> GraphModule:
    r"""Internal helper function for prepare_fx
        Args:
          `model`, `qconfig_mapping`, `prepare_custom_config`, `_equalization_config`:
          see docs for :func:`~torch.ao.quantization.prepare_fx`
          `is_standalone_module`: a boolean flag indicates whether we are
          quantizing a standalone module or not, a standalone module
          is a submodule of the parent module that is not inlined in the
    forward graph of the parent module,
          the way we quantize standalone module is described in:
          :func:`~torch.ao.quantization._prepare_standalone_module_fx`
    """
    if prepare_custom_config is None:
        prepare_custom_config = PrepareCustomConfig()
    if _equalization_config is None:
        _equalization_config = QConfigMapping()

    if isinstance(prepare_custom_config, dict):
        warnings.warn(
            "Passing a prepare_custom_config_dict to prepare is deprecated and will not be supported "
            "in a future version. Please pass in a PrepareCustomConfig instead.",
            FutureWarning,
            stacklevel=3,
        )
        prepare_custom_config = PrepareCustomConfig.from_dict(prepare_custom_config)

    # swap FloatFunctional with FXFloatFunctional
    _swap_ff_with_fxff(model)

    skipped_module_names, skipped_module_classes = get_skipped_module_name_and_classes(
        prepare_custom_config, is_standalone_module
    )
    preserved_attr_names = prepare_custom_config.preserved_attributes
    preserved_attrs = {
        attr: getattr(model, attr)
        for attr in preserved_attr_names
        if hasattr(model, attr)
    }
    # symbolically trace the model
    tracer = QuantizationTracer(skipped_module_names, skipped_module_classes)  # type: ignore[arg-type]
    graph_module = GraphModule(model, tracer.trace(model))
    _attach_meta_to_node_if_not_exist(graph_module)

    fuse_custom_config = FuseCustomConfig().set_preserved_attributes(
        prepare_custom_config.preserved_attributes
    )
    graph_module = _fuse_fx(graph_module, is_qat, fuse_custom_config, backend_config)
    prepared = prepare(
        graph_module,
        qconfig_mapping,
        is_qat,
        tracer.node_name_to_scope,
        example_inputs=example_inputs,
        prepare_custom_config=prepare_custom_config,
        _equalization_config=_equalization_config,
        backend_config=backend_config,
        is_standalone_module=is_standalone_module,
    )  # type: ignore[operator]

    attach_preserved_attrs_to_model(prepared, preserved_attrs)
    return prepared


def _prepare_standalone_module_fx(
    model: torch.nn.Module,
    qconfig_mapping: QConfigMapping | dict[str, Any],
    is_qat: bool,
    example_inputs: tuple[Any, ...],
    prepare_custom_config: PrepareCustomConfig | dict[str, Any] | None = None,
    backend_config: BackendConfig | dict[str, Any] | None = None,
) -> GraphModule:
    r"""[Internal use only] Prepare a standalone module, so that it can be used when quantizing the
    parent module.
    standalone_module means it a submodule that is not inlined in parent module,
    and will be quantized separately as one unit.

    How the standalone module is observed is specified by `input_quantized_idxs` and
    `output_quantized_idxs` in the prepare_custom_config for the standalone module

    Returns:

        * model(GraphModule): prepared standalone module. It has these attributes in
          model.meta:

            * `standalone_module_input_quantized_idxs(List[Int])`: a list of
              indexes for the graph input that is expected to be quantized,
              same as input_quantized_idxs configuration provided
              for the standalone module
            * `standalone_module_output_quantized_idxs(List[Int])`: a list of
              indices for the graph output that is quantized
              same as input_quantized_idxs configuration provided
              for the standalone module

    """
    return _prepare_fx(
        model,
        qconfig_mapping,
        is_qat,
        example_inputs,
        prepare_custom_config,
        backend_config=backend_config,
        is_standalone_module=True,
    )


def fuse_fx(
    model: torch.nn.Module,
    fuse_custom_config: FuseCustomConfig | dict[str, Any] | None = None,
    backend_config: BackendConfig | dict[str, Any] | None = None,
) -> GraphModule:
    r"""Fuse modules like conv+bn, conv+bn+relu etc, model must be in eval mode.
    Fusion rules are defined in torch.ao.quantization.fx.fusion_pattern.py

    Args:

        * `model` (torch.nn.Module): a torch.nn.Module model
        * `fuse_custom_config` (FuseCustomConfig): custom configurations for fuse_fx.
            See :class:`~torch.ao.quantization.fx.custom_config.FuseCustomConfig` for more details
    Example::

        from torch.ao.quantization import fuse_fx

        m = Model().eval()
        m = fuse_fx(m)

    """
    if fuse_custom_config is None:
        fuse_custom_config = FuseCustomConfig()

    if isinstance(fuse_custom_config, dict):
        warnings.warn(
            "Passing a fuse_custom_config_dict to fuse is deprecated and will not be supported "
            "in a future version. Please pass in a FuseCustomConfig instead.",
            FutureWarning,
            stacklevel=2,
        )
        fuse_custom_config = FuseCustomConfig.from_dict(fuse_custom_config)

    torch._C._log_api_usage_once("quantization_api.quantize_fx.fuse_fx")
    preserved_attr_names = fuse_custom_config.preserved_attributes
    preserved_attrs = {
        attr: getattr(model, attr)
        for attr in preserved_attr_names
        if hasattr(model, attr)
    }

    graph_module = torch.fx.symbolic_trace(model)
    _attach_meta_to_node_if_not_exist(graph_module)
    graph_module = _fuse_fx(graph_module, False, fuse_custom_config, backend_config)

    attach_preserved_attrs_to_model(graph_module, preserved_attrs)
    return graph_module


@typing_extensions.deprecated(DEPRECATION_WARNING)
def prepare_fx(
    model: torch.nn.Module,
    qconfig_mapping: QConfigMapping | dict[str, Any],
    example_inputs: tuple[Any, ...],
    prepare_custom_config: PrepareCustomConfig | dict[str, Any] | None = None,
    _equalization_config: QConfigMapping | dict[str, Any] | None = None,
    backend_config: BackendConfig | dict[str, Any] | None = None,
) -> GraphModule:
    r""" Prepare a model for post training quantization

    Args:
      * `model` (torch.nn.Module): torch.nn.Module model

      * `qconfig_mapping` (QConfigMapping): QConfigMapping object to configure how a model is
         quantized, see :class:`~torch.ao.quantization.qconfig_mapping.QConfigMapping`
         for more details

      * `example_inputs` (Tuple[Any, ...]): Example inputs for forward function of the model,
         Tuple of positional args (keyword args can be passed as positional args as well)

      * `prepare_custom_config` (PrepareCustomConfig): customization configuration for quantization tool.
          See :class:`~torch.ao.quantization.fx.custom_config.PrepareCustomConfig` for more details

      * `_equalization_config`: config for specifying how to perform equalization on the model

      * `backend_config` (BackendConfig): config that specifies how operators are quantized
         in a backend, this includes how the operators are observed,
         supported fusion patterns, how quantize/dequantize ops are
         inserted, supported dtypes etc. See :class:`~torch.ao.quantization.backend_config.BackendConfig` for more details

    Return:
      A GraphModule with observer (configured by qconfig_mapping), ready for calibration

    Example::

        import torch
        from torch.ao.quantization import get_default_qconfig_mapping
        from torch.ao.quantization.quantize_fx import prepare_fx

        class Submodule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)
            def forward(self, x):
                x = self.linear(x)
                return x

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)
                self.sub = Submodule()

            def forward(self, x):
                x = self.linear(x)
                x = self.sub(x) + x
                return x

        # initialize a floating point model
        float_model = M().eval()

        # define calibration function
        def calibrate(model, data_loader):
            model.eval()
            with torch.no_grad():
                for image, target in data_loader:
                    model(image)

        # qconfig is the configuration for how we insert observers for a particular
        # operator
        # qconfig = get_default_qconfig("fbgemm")
        # Example of customizing qconfig:
        # qconfig = torch.ao.quantization.QConfig(
        #    activation=MinMaxObserver.with_args(dtype=torch.qint8),
        #    weight=MinMaxObserver.with_args(dtype=torch.qint8))
        # `activation` and `weight` are constructors of observer module

        # qconfig_mapping is a collection of quantization configurations, user can
        # set the qconfig for each operator (torch op calls, functional calls, module calls)
        # in the model through qconfig_mapping
        # the following call will get the qconfig_mapping that works best for models
        # that target "fbgemm" backend
        qconfig_mapping = get_default_qconfig_mapping("fbgemm")

        # We can customize qconfig_mapping in different ways.
        # e.g. set the global qconfig, which means we will use the same qconfig for
        # all operators in the model, this can be overwritten by other settings
        # qconfig_mapping = QConfigMapping().set_global(qconfig)
        # e.g. quantize the linear submodule with a specific qconfig
        # qconfig_mapping = QConfigMapping().set_module_name("linear", qconfig)
        # e.g. quantize all nn.Linear modules with a specific qconfig
        # qconfig_mapping = QConfigMapping().set_object_type(torch.nn.Linear, qconfig)
        # for a more complete list, please see the docstring for :class:`torch.ao.quantization.QConfigMapping`
        # argument

        # example_inputs is a tuple of inputs, that is used to infer the type of the
        # outputs in the model
        # currently it's not used, but please make sure model(*example_inputs) runs
        example_inputs = (torch.randn(1, 3, 224, 224),)

        # TODO: add backend_config after we split the backend_config for fbgemm and qnnpack
        # e.g. backend_config = get_default_backend_config("fbgemm")
        # `prepare_fx` inserts observers in the model based on qconfig_mapping and
        # backend_config. If the configuration for an operator in qconfig_mapping
        # is supported in the backend_config (meaning it's supported by the target
        # hardware), we'll insert observer modules according to the qconfig_mapping
        # otherwise the configuration in qconfig_mapping will be ignored
        #
        # Example:
        # in qconfig_mapping, user sets linear module to be quantized with quint8 for
        # activation and qint8 for weight:
        # qconfig = torch.ao.quantization.QConfig(
        #     observer=MinMaxObserver.with_args(dtype=torch.quint8),
        #     weight=MinMaxObserver.with-args(dtype=torch.qint8))
        # Note: current qconfig api does not support setting output observer, but
        # we may extend this to support these more fine grained control in the
        # future
        #
        # qconfig_mapping = QConfigMapping().set_object_type(torch.nn.Linear, qconfig)
        # in backend config, linear module also supports in this configuration:
        # weighted_int8_dtype_config = DTypeConfig(
        #   input_dtype=torch.quint8,
        #   output_dtype=torch.quint8,
        #   weight_dtype=torch.qint8,
        #   bias_type=torch.float)

        # linear_pattern_config = BackendPatternConfig(torch.nn.Linear) \
        #    .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
        #    .add_dtype_config(weighted_int8_dtype_config) \
        #    ...

        # backend_config = BackendConfig().set_backend_pattern_config(linear_pattern_config)
        # `prepare_fx` will check that the setting requested by user in qconfig_mapping
        # is supported by the backend_config and insert observers and fake quant modules
        # in the model
        prepared_model = prepare_fx(float_model, qconfig_mapping, example_inputs)
        # Run calibration
        calibrate(prepared_model, sample_inference_data)
    """
    torch._C._log_api_usage_once("quantization_api.quantize_fx.prepare_fx")
    return _prepare_fx(
        model,
        qconfig_mapping,
        False,  # is_qat
        example_inputs,
        prepare_custom_config,
        _equalization_config,
        backend_config,
    )


@typing_extensions.deprecated(DEPRECATION_WARNING)
def prepare_qat_fx(
    model: torch.nn.Module,
    qconfig_mapping: QConfigMapping | dict[str, Any],
    example_inputs: tuple[Any, ...],
    prepare_custom_config: PrepareCustomConfig | dict[str, Any] | None = None,
    backend_config: BackendConfig | dict[str, Any] | None = None,
) -> GraphModule:
    r"""Prepare a model for quantization aware training

    Args:
      * `model` (torch.nn.Module): torch.nn.Module model
      * `qconfig_mapping` (QConfigMapping): see :func:`~torch.ao.quantization.prepare_fx`
      * `example_inputs` (Tuple[Any, ...]): see :func:`~torch.ao.quantization.prepare_fx`
      * `prepare_custom_config` (PrepareCustomConfig): see :func:`~torch.ao.quantization.prepare_fx`
      * `backend_config` (BackendConfig): see :func:`~torch.ao.quantization.prepare_fx`

    Return:
      A GraphModule with fake quant modules (configured by qconfig_mapping and backend_config), ready for
      quantization aware training

    Example::

        import torch
        from torch.ao.quantization import get_default_qat_qconfig_mapping
        from torch.ao.quantization.quantize_fx import prepare_qat_fx


        class Submodule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                x = self.linear(x)
                return x


        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)
                self.sub = Submodule()

            def forward(self, x):
                x = self.linear(x)
                x = self.sub(x) + x
                return x


        # initialize a floating point model
        float_model = M().train()
        # (optional, but preferred) load the weights from pretrained model
        # float_model.load_weights(...)


        # define the training loop for quantization aware training
        def train_loop(model, train_data):
            model.train()
            for image, target in data_loader:
                ...


        # qconfig is the configuration for how we insert observers for a particular
        # operator
        # qconfig = get_default_qconfig("fbgemm")
        # Example of customizing qconfig:
        # qconfig = torch.ao.quantization.QConfig(
        #    activation=FakeQuantize.with_args(observer=MinMaxObserver.with_args(dtype=torch.qint8)),
        #    weight=FakeQuantize.with_args(observer=MinMaxObserver.with_args(dtype=torch.qint8)))
        # `activation` and `weight` are constructors of observer module

        # qconfig_mapping is a collection of quantization configurations, user can
        # set the qconfig for each operator (torch op calls, functional calls, module calls)
        # in the model through qconfig_mapping
        # the following call will get the qconfig_mapping that works best for models
        # that target "fbgemm" backend
        qconfig_mapping = get_default_qat_qconfig_mapping("fbgemm")

        # We can customize qconfig_mapping in different ways, please take a look at
        # the docstring for :func:`~torch.ao.quantization.prepare_fx` for different ways
        # to configure this

        # example_inputs is a tuple of inputs, that is used to infer the type of the
        # outputs in the model
        # currently it's not used, but please make sure model(*example_inputs) runs
        example_inputs = (torch.randn(1, 3, 224, 224),)

        # TODO: add backend_config after we split the backend_config for fbgemm and qnnpack
        # e.g. backend_config = get_default_backend_config("fbgemm")
        # `prepare_qat_fx` inserts observers in the model based on qconfig_mapping and
        # backend_config, if the configuration for an operator in qconfig_mapping
        # is supported in the backend_config (meaning it's supported by the target
        # hardware), we'll insert fake_quantize modules according to the qconfig_mapping
        # otherwise the configuration in qconfig_mapping will be ignored
        # see :func:`~torch.ao.quantization.prepare_fx` for a detailed explanation of
        # how qconfig_mapping interacts with backend_config
        prepared_model = prepare_qat_fx(float_model, qconfig_mapping, example_inputs)
        # Run training
        train_loop(prepared_model, train_loop)

    """
    torch._C._log_api_usage_once("quantization_api.quantize_fx.prepare_qat_fx")
    return _prepare_fx(
        model,
        qconfig_mapping,
        True,  # is_qat
        example_inputs,
        prepare_custom_config,
        backend_config=backend_config,
    )


def _convert_fx(
    graph_module: GraphModule,
    is_reference: bool,
    convert_custom_config: ConvertCustomConfig | dict[str, Any] | None = None,
    is_standalone_module: bool = False,
    _remove_qconfig: bool = True,
    qconfig_mapping: QConfigMapping | dict[str, Any] | None = None,
    backend_config: BackendConfig | dict[str, Any] | None = None,
    is_decomposed: bool = False,
    keep_original_weights: bool = False,
) -> GraphModule:
    """`is_standalone_module`: see docs in :func:`~torch.ao.quantization.prepare_standalone_module_fx`"""
    if convert_custom_config is None:
        convert_custom_config = ConvertCustomConfig()

    if isinstance(convert_custom_config, dict):
        warnings.warn(
            "Passing a convert_custom_config_dict to convert is deprecated and will not be supported "
            "in a future version. Please pass in a ConvertCustomConfig instead.",
            FutureWarning,
            stacklevel=3,
        )
        convert_custom_config = ConvertCustomConfig.from_dict(convert_custom_config)

    _check_is_graph_module(graph_module)
    preserved_attr_names = convert_custom_config.preserved_attributes
    preserved_attrs = {
        attr: getattr(graph_module, attr)
        for attr in preserved_attr_names
        if hasattr(graph_module, attr)
    }

    quantized = convert(
        graph_module,
        is_reference,
        convert_custom_config,
        is_standalone_module,
        _remove_qconfig_flag=_remove_qconfig,
        qconfig_mapping=qconfig_mapping,
        backend_config=backend_config,
        is_decomposed=is_decomposed,
        keep_original_weights=keep_original_weights,
    )

    attach_preserved_attrs_to_model(quantized, preserved_attrs)
    return quantized


@typing_extensions.deprecated(DEPRECATION_WARNING)
def convert_fx(
    graph_module: GraphModule,
    convert_custom_config: ConvertCustomConfig | dict[str, Any] | None = None,
    _remove_qconfig: bool = True,
    qconfig_mapping: QConfigMapping | dict[str, Any] | None = None,
    backend_config: BackendConfig | dict[str, Any] | None = None,
    keep_original_weights: bool = False,
) -> GraphModule:
    r"""Convert a calibrated or trained model to a quantized model

    Args:
        * `graph_module` (torch.fx.GraphModule): A prepared and calibrated/trained model (GraphModule)

        * `convert_custom_config` (ConvertCustomConfig): custom configurations for convert function.
            See :class:`~torch.ao.quantization.fx.custom_config.ConvertCustomConfig` for more details

        * `_remove_qconfig` (bool): Option to remove the qconfig attributes in the model after convert.

        * `qconfig_mapping` (QConfigMapping): config for specifying how to convert a model for quantization.

           The keys must include the ones in the qconfig_mapping passed to `prepare_fx` or `prepare_qat_fx`,
           with the same values or `None`. Additional keys can be specified with values set to `None`.

          For each entry whose value is set to None, we skip quantizing that entry in the model::

            qconfig_mapping = QConfigMapping
                .set_global(qconfig_from_prepare)
                .set_object_type(torch.nn.functional.add, None)  # skip quantizing torch.nn.functional.add
                .set_object_type(torch.nn.functional.linear, qconfig_from_prepare)
                .set_module_name("foo.bar", None)  # skip quantizing module "foo.bar"

         * `backend_config` (BackendConfig): A configuration for the backend which describes how
            operators should be quantized in the backend, this includes quantization
            mode support (static/dynamic/weight_only), dtype support (quint8/qint8 etc.),
            observer placement for each operators and fused operators.
            See :class:`~torch.ao.quantization.backend_config.BackendConfig` for more details

    Return:
        A quantized model (torch.nn.Module)

    Example::

        # prepared_model: the model after prepare_fx/prepare_qat_fx and calibration/training
        # convert_fx converts a calibrated/trained model to a quantized model for the
        # target hardware, this includes converting the model first to a reference
        # quantized model, and then lower the reference quantized model to a backend
        # Currently, the supported backends are fbgemm (onednn), qnnpack (xnnpack) and
        # they share the same set of quantized operators, so we are using the same
        # lowering procedure
        #
        # backend_config defines the corresponding reference quantized module for
        # the weighted modules in the model, e.g. nn.Linear
        # TODO: add backend_config after we split the backend_config for fbgemm and qnnpack
        # e.g. backend_config = get_default_backend_config("fbgemm")
        quantized_model = convert_fx(prepared_model)

    """
    torch._C._log_api_usage_once("quantization_api.quantize_fx.convert_fx")
    return _convert_fx(
        graph_module,
        is_reference=False,
        convert_custom_config=convert_custom_config,
        _remove_qconfig=_remove_qconfig,
        qconfig_mapping=qconfig_mapping,
        backend_config=backend_config,
        keep_original_weights=keep_original_weights,
    )


def convert_to_reference_fx(
    graph_module: GraphModule,
    convert_custom_config: ConvertCustomConfig | dict[str, Any] | None = None,
    _remove_qconfig: bool = True,
    qconfig_mapping: QConfigMapping | dict[str, Any] | None = None,
    backend_config: BackendConfig | dict[str, Any] | None = None,
) -> GraphModule:
    r"""Convert a calibrated or trained model to a reference quantized model,
    see https://github.com/pytorch/rfcs/blob/master/RFC-0019-Extending-PyTorch-Quantization-to-Custom-Backends.md for more details,
    reference quantized model is a standard representation of a quantized model provided
    by FX Graph Mode Quantization, it can be further lowered to run on the target
    hardware, like accelerators

    Args:
        * `graph_module` (GraphModule): A prepared and calibrated/trained model (GraphModule)

        * `convert_custom_config` (ConvertCustomConfig): custom configurations for convert function.
            See :func:`~torch.ao.quantization.quantize_fx.convert_fx` for more details.

        * `_remove_qconfig` (bool): Option to remove the qconfig attributes in the model after convert.

        * `qconfig_mapping` (QConfigMapping): config for specifying how to convert a model for quantization.
            See :func:`~torch.ao.quantization.quantize_fx.convert_fx` for more details.

         * `backend_config` (BackendConfig): A configuration for the backend which describes how
            operators should be quantized in the backend. See
            :func:`~torch.ao.quantization.quantize_fx.convert_fx` for more details.

    Return:
        A reference quantized model (GraphModule)

    Example::

        # prepared_model: the model after prepare_fx/prepare_qat_fx and calibration/training
        # TODO: add backend_config after we split the backend_config for fbgemm and qnnpack
        # e.g. backend_config = get_default_backend_config("fbgemm")
        reference_quantized_model = convert_to_reference_fx(prepared_model)

    """
    torch._C._log_api_usage_once("quantization_api.quantize_fx.convert_to_reference_fx")
    return _convert_fx(
        graph_module,
        is_reference=True,
        convert_custom_config=convert_custom_config,
        _remove_qconfig=_remove_qconfig,
        qconfig_mapping=qconfig_mapping,
        backend_config=backend_config,
    )


def _convert_to_reference_decomposed_fx(
    graph_module: GraphModule,
    convert_custom_config: ConvertCustomConfig | dict[str, Any] | None = None,
    qconfig_mapping: QConfigMapping | dict[str, Any] | None = None,
    backend_config: BackendConfig | dict[str, Any] | None = None,
) -> GraphModule:
    r"""Convert a calibrated or trained model to a reference quantized model, with
    decomposed representation for quantized Tensor
    see https://github.com/pytorch/rfcs/blob/master/RFC-0019-Extending-PyTorch-Quantization-to-Custom-Backends.md for more details,
    reference quantized model is a standard representation of a quantized model provided
    by FX Graph Mode Quantization, it can be further lowered to run on the target
    hardware, like accelerators

    Note: this is not public API

    Args:
        * `graph_module` (GraphModule): A prepared and calibrated/trained model (GraphModule)

        * `convert_custom_config` (ConvertCustomConfig): custom configurations for convert function.
            See :func:`~torch.ao.quantization.quantize_fx.convert_fx` for more details.

        * `_remove_qconfig` (bool): Option to remove the qconfig attributes in the model after convert.

        * `qconfig_mapping` (QConfigMapping): config for specifying how to convert a model for quantization.
            See :func:`~torch.ao.quantization.quantize_fx.convert_fx` for more details.

         * `backend_config` (BackendConfig): A configuration for the backend which describes how
            operators should be quantized in the backend. See
            :func:`~torch.ao.quantization.quantize_fx.convert_fx` for more details.

    Return:
        A reference quantized model (GraphModule) with operators working with decomposed quantized Tensor

    Example::

        # prepared_model: the model after prepare_fx/prepare_qat_fx and calibration/training
        # TODO: add backend_config after we split the backend_config for fbgemm and qnnpack
        # e.g. backend_config = get_default_backend_config("fbgemm")
        reference_quantized_model = _convert_to_reference_decomposed_fx(prepared_model)

    """
    torch._C._log_api_usage_once(
        "quantization_api.quantize_fx._convert_to_reference_decomposed_fx"
    )
    return _convert_fx(
        graph_module,
        is_reference=True,
        convert_custom_config=convert_custom_config,
        _remove_qconfig=False,
        qconfig_mapping=qconfig_mapping,
        backend_config=backend_config,
        is_decomposed=True,
    )


def _convert_standalone_module_fx(
    graph_module: GraphModule,
    is_reference: bool = False,
    convert_custom_config: ConvertCustomConfig | dict[str, Any] | None = None,
) -> GraphModule:
    r"""[Internal use only] Convert a model produced by :func:`~torch.ao.quantization.prepare_standalone_module_fx`
    and convert it to a quantized model

    Returns a quantized standalone module, whether input/output is quantized is
    specified by prepare_custom_config, with
    input_quantized_idxs, output_quantized_idxs, please
    see docs for prepare_fx for details
    """
    return _convert_fx(
        graph_module,
        is_reference,
        convert_custom_config,
        is_standalone_module=True,
    )
<|endoftext|># mypy: allow-untyped-defs
import copy
import inspect
import itertools
import typing_extensions
import warnings

import torch
import torch.ao.nn.quantized as nnq
import torch.nn as nn
from torch.ao.nn.intrinsic import _FusedModule
from torch.ao.quantization.observer import _is_activation_post_process
from torch.ao.quantization.qconfig import (
    _activation_is_memoryless,
    _add_module_to_qconfig_obs_ctr,
    default_dynamic_qconfig,
    float16_dynamic_qconfig,
    float_qparams_weight_only_qconfig,
    float_qparams_weight_only_qconfig_4bit,
)
from torch.ao.quantization.quantization_mappings import (
    _get_special_act_post_process,
    _has_special_act_post_process,
    get_default_dynamic_quant_module_mappings,
    get_default_qat_module_mappings,
    get_default_qconfig_propagation_list,
    get_default_static_quant_module_mappings,
    get_default_static_quant_reference_module_mappings,
    no_observer_set,
)
from torch.ao.quantization.stubs import DeQuantStub, QuantWrapper
from torch.nn.utils.parametrize import type_before_parametrizations

from .utils import (
    DEPRECATION_WARNING,
    get_qparam_dict,
    has_no_children_ignoring_parametrizations,
)


__all__ = [
    "get_default_custom_config_dict",
    "propagate_qconfig_",
    "add_quant_dequant",
    "prepare",
    "quantize",
    "quantize_dynamic",
    "prepare_qat",
    "quantize_qat",
    "convert",
    "swap_module",
]


# TODO remove this once BC is no longer required to avoid a SEV
is_activation_post_process = _is_activation_post_process


_DEFAULT_CUSTOM_CONFIG_DICT = {
    "float_to_observed_custom_module_class": {
        nn.LSTM: nn.quantizable.LSTM,
        nn.MultiheadAttention: nn.quantizable.MultiheadAttention,
    },
    "observed_to_quantized_custom_module_class": {
        nn.quantizable.LSTM: nn.quantized.LSTM,
        nn.quantizable.MultiheadAttention: nn.quantized.MultiheadAttention,
    },
}


def get_default_custom_config_dict():
    r"""Defines the default custom config dict."""
    return _DEFAULT_CUSTOM_CONFIG_DICT


def _propagate_qconfig_helper(
    module,
    qconfig_dict,
    qconfig_parent=None,
    prefix="",
    prepare_custom_config_dict=None,
):
    r"""This is a helper function for `propagate_qconfig_`

    Args:
        module: input module
        qconfig_dict: dictionary that maps from name of submodule to quantization
                     configuration
        qconfig_parent: quantization config of parent module, we will fallback to
                       this config when there is no specified config for current
                       module
        prefix: corresponding prefix of the current module, used as key in
                qconfig_dict
        prepare_custom_config_dict: dictionary for custom handling of modules
                                    see docs for :func:`~torch.ao.quantization.prepare_fx`

    Return:
        None, module is modified inplace with qconfig attached
    """

    module_qconfig = qconfig_dict.get(
        type_before_parametrizations(module), qconfig_parent
    )
    module_qconfig = qconfig_dict.get(prefix, module_qconfig)
    module_qconfig = getattr(module, "qconfig", module_qconfig)

    torch.ao.quantization.qconfig._assert_valid_qconfig(module_qconfig, module)

    qconfig_with_device_check = _add_module_to_qconfig_obs_ctr(module_qconfig, module)
    module.qconfig = qconfig_with_device_check

    for name, child in module.named_children():
        module_prefix = prefix + "." + name if prefix else name
        #  do not propagate qconfig to child if child is non traceable
        if prepare_custom_config_dict is None or not (
            name in prepare_custom_config_dict.get("non_traceable_module_name", [])
            or type(child)
            in prepare_custom_config_dict.get("non_traceable_module_class", [])
        ):
            _propagate_qconfig_helper(
                child, qconfig_dict, qconfig_with_device_check, module_prefix
            )


def propagate_qconfig_(module, qconfig_dict=None, prepare_custom_config_dict=None):
    r"""Propagate qconfig through the module hierarchy and assign `qconfig`
    attribute on each leaf module

    Args:
        module: input module
        qconfig_dict: dictionary that maps from name or type of submodule to
            quantization configuration, qconfig applies to all submodules of a
            given module unless qconfig for the submodules are specified (when
            the submodule already has qconfig attribute)
        prepare_custom_config_dict: dictionary for custom handling of modules
            see docs for :func:`~torch.ao.quantization.prepare_fx`

    Return:
        None, module is modified inplace with qconfig attached
    """
    if qconfig_dict is None:
        qconfig_dict = {}
    if prepare_custom_config_dict is None:
        prepare_custom_config_dict = {}
    _propagate_qconfig_helper(
        module, qconfig_dict, prepare_custom_config_dict=prepare_custom_config_dict
    )


def _observer_forward_hook(self, input, output):
    r"""Forward hook that calls observer on the output"""
    return self.activation_post_process(output)


def _observer_forward_pre_hook(self, input):
    r"""Forward pre hook that calls observer on the output"""
    return self.activation_post_process(input[0])


def _register_activation_post_process_hook(module, pre_hook=False):
    if not hasattr(module, "activation_post_process"):
        raise AssertionError(
            "Expect activation_post_process attribute already attached to the module"
        )
    if pre_hook:
        module.register_forward_pre_hook(_observer_forward_pre_hook, prepend=True)
    else:
        module.register_forward_hook(_observer_forward_hook, prepend=True)


def _add_observer_(
    module,
    qconfig_propagation_list=None,
    non_leaf_module_list=None,
    device=None,
    custom_module_class_mapping=None,
):
    r"""Add observer for the leaf child of the module.

    This function insert observer module to all leaf child module that
    has a valid qconfig attribute.

    Args:
        module: input module with qconfig attributes for all the leaf modules that we want to quantize
        qconfig_propagation_list: a list of quantizable modules that will have observers added to them
            if they are leaf nodes
        device: parent device, if any
        non_leaf_module_list: list of non-leaf modules we want to add observer

    Return:
        None, module is modified inplace with added observer modules and forward_hooks
    """
    if qconfig_propagation_list is None:
        qconfig_propagation_list = get_default_qconfig_propagation_list()

    if custom_module_class_mapping is None:
        custom_module_class_mapping = {}

    # respect device affinity when adding observers
    if device is None:
        devices = _get_unique_devices_(module)
        if len(devices) > 1:
            raise AssertionError(
                f"_add_observer_ only works with cpu or single-device CUDA modules, but got devices {devices}"
            )
        device = next(iter(devices)) if len(devices) > 0 else None

    def get_activation_post_process(qconfig, device, special_act_post_process=None):
        activation = (
            qconfig.activation()
            if special_act_post_process is None
            else special_act_post_process()
        )
        if device is not None:
            activation.to(device)
        return activation

    def needs_observation(m):
        return hasattr(m, "qconfig") and m.qconfig is not None

    def insert_activation_post_process(m, special_act_post_process=None):
        """Adds an activation post process module and register
        a pre or post hook that calls the module
        """
        # We don't insert observer/fake_quantize for DeQuantStub
        if needs_observation(m) and not isinstance(m, DeQuantStub):
            # observer and hook will be gone after we swap the module
            m.add_module(
                "activation_post_process",
                get_activation_post_process(
                    m.qconfig, device, special_act_post_process
                ),
            )
            # Register observer as the first entry in the hook list
            # All post forward hooks are preserved and will be executed after the observer before convert
            _register_activation_post_process_hook(
                m, pre_hook=_activation_is_memoryless(m.qconfig)
            )

    for name, child in module.named_children():
        # TODO remove Dropout special after codebase stable
        if type_before_parametrizations(child) is nn.Dropout:
            continue
        elif issubclass(
            type_before_parametrizations(child), (nnq.FloatFunctional, nnq.QFunctional)
        ):
            if needs_observation(child):
                if not hasattr(child, "activation_post_process"):
                    raise AssertionError(
                        f"functional class {type_before_parametrizations(child)} has no pre-defined `activation_post_process`"
                    )
                child.activation_post_process = get_activation_post_process(
                    child.qconfig, device
                )
        elif isinstance(child, _FusedModule):
            # activation_post_process are now added directly to nn.Sequential/_FusedModule
            if needs_observation(child):
                insert_activation_post_process(child)
        elif (
            non_leaf_module_list is not None
            and type_before_parametrizations(child) in non_leaf_module_list
        ):
            if needs_observation(child):
                insert_activation_post_process(child)
        elif _has_special_act_post_process(child):
            special_act_post_process = _get_special_act_post_process(child)
            insert_activation_post_process(child, special_act_post_process)
        elif (
            needs_observation(child)
            and type_before_parametrizations(child) in custom_module_class_mapping
        ):
            observed_class = custom_module_class_mapping[
                type_before_parametrizations(child)
            ]
            observed_child = observed_class.from_float(child)
            setattr(module, name, observed_child)
            # TODO: These are the modules that cannot be observed
            #       Once there are more, we should move them to a separate list
            if not issubclass(observed_class, tuple(no_observer_set())):
                insert_activation_post_process(observed_child)
        else:
            _add_observer_(
                child,
                qconfig_propagation_list,
                non_leaf_module_list,
                device,
                custom_module_class_mapping,
            )

    # Insert observers only for leaf nodes, note that this observer is for
    # the output of the module, for input QuantStub will observe them
    if (
        has_no_children_ignoring_parametrizations(module)
        and not isinstance(module, torch.nn.Sequential)
        and type_before_parametrizations(module) in qconfig_propagation_list
    ):
        insert_activation_post_process(module)
    # This is a special case for AdaRound eager mode
    # AdaRound contains weight_fake_quant to be propagated from API to convert
    # leaf node check with a number of children looks naive assumption that blocks
    # Adding an exception case for AdaRound
    if (
        hasattr(module, "weight_fake_quant")
        and not isinstance(module, torch.nn.Sequential)
        and type_before_parametrizations(module) in qconfig_propagation_list
    ):
        insert_activation_post_process(module)


def _get_unique_devices_(module):
    return {p.device for p in module.parameters() if p.device.type != "meta"} | {
        p.device for p in module.buffers() if p.device.type != "meta"
    }


def add_quant_dequant(module):
    r"""Wrap the leaf child module in QuantWrapper if it has a valid qconfig
    Note that this function will modify the children of module inplace and it
    can return a new module which wraps the input module as well.

    Args:
        module: input module with qconfig attributes for all the leaf modules
        that we want to quantize

    Return:
        Either the inplace modified module with submodules wrapped in
        `QuantWrapper` based on qconfig or a new `QuantWrapper` module which
        wraps the input module, the latter case only happens when the input
        module is a leaf module and we want to quantize it.
    """
    if (
        has_no_children_ignoring_parametrizations(module)
        and hasattr(module, "qconfig")
        and module.qconfig
    ):
        return QuantWrapper(module)

    for name, child in module.named_children():
        module._modules[name] = add_quant_dequant(child)
    return module


@typing_extensions.deprecated(DEPRECATION_WARNING)
def prepare(
    model,
    inplace=False,
    allow_list=None,
    observer_non_leaf_module_list=None,
    prepare_custom_config_dict=None,
):
    r"""Prepares a copy of the model for quantization calibration or quantization-aware training.

    Quantization configuration should be assigned preemptively
    to individual submodules in `.qconfig` attribute.

    The model will be attached with observer or fake quant modules, and qconfig
    will be propagated.

    Args:
        `model`: input model to be modified in-place
        `inplace`: carry out model transformations in-place, the original module is mutated
        `allow_list`: list of quantizable modules
        `observer_non_leaf_module_list`: list of non-leaf modules we want to add observer
        `prepare_custom_config_dict`: customization configuration dictionary for prepare function

    .. code-block:: python

       # Example of prepare_custom_config_dict:
       prepare_custom_config_dict = {
           # user will manually define the corresponding observed
           # module class which has a from_float class method that converts
           # float custom module to observed custom module
           "float_to_observed_custom_module_class": {CustomModule: ObservedCustomModule}
       }

    """
    torch._C._log_api_usage_once("quantization_api.quantize.prepare")
    if prepare_custom_config_dict is None:
        prepare_custom_config_dict = get_default_custom_config_dict()
    custom_module_class_mapping = prepare_custom_config_dict.get(
        "float_to_observed_custom_module_class", {}
    )

    if not inplace:
        model = copy.deepcopy(model)

    # TODO: remove allow_list
    qconfig_propagation_list = allow_list
    if allow_list is None:
        qconfig_propagation_list = get_default_qconfig_propagation_list()
    propagate_qconfig_(model, qconfig_dict=None)

    # sanity check common API misusage
    if not any(hasattr(m, "qconfig") and m.qconfig for m in model.modules()):
        warnings.warn(
            "None of the submodule got qconfig applied. Make sure you "
            "passed correct configuration through `qconfig_dict` or "
            "by assigning the `.qconfig` attribute directly on submodules",
            stacklevel=2,
        )

    _add_observer_(
        model,
        qconfig_propagation_list,
        observer_non_leaf_module_list,
        custom_module_class_mapping=custom_module_class_mapping,
    )
    return model


def _remove_activation_post_process(module):
    # TODO: maybe we should change activation_post_process to _activation_post_process
    # to prevent it from being used by user
    if hasattr(module, "activation_post_process") and _is_activation_post_process(
        module.activation_post_process
    ):
        delattr(module, "activation_post_process")

    # remove activation_post_process pre and post hooks
    def remove_hooks(pre_hook=False):
        hook_map = module._forward_pre_hooks if pre_hook else module._forward_hooks
        observer_hook = (
            _observer_forward_pre_hook if pre_hook else _observer_forward_hook
        )
        handle_ids_to_remove = set()
        for handle_id, hook_fn in hook_map.items():
            if hook_fn is observer_hook:
                handle_ids_to_remove.add(handle_id)
        for handle_id in handle_ids_to_remove:
            hook_map.pop(handle_id)

    remove_hooks(pre_hook=True)
    remove_hooks(pre_hook=False)


# TODO: rename to something more general
def _remove_qconfig(module):
    r"""Clean up the qconfig left in the module so that new qconfig can be
    propagated.

    Args:
        module: module to be cleaned up
    """
    for child in module.children():
        _remove_qconfig(child)

    if hasattr(module, "qconfig"):
        del module.qconfig

    _remove_activation_post_process(module)


@typing_extensions.deprecated(DEPRECATION_WARNING)
def quantize(model, run_fn, run_args, mapping=None, inplace=False):
    r"""Quantize the input float model with post training static quantization.

    First it will prepare the model for calibration, then it calls
    `run_fn` which will run the calibration step, after that we will
    convert the model to a quantized model.

    Args:
        model: input float model
        run_fn: a calibration function for calibrating the prepared model
        run_args: positional arguments for `run_fn`
        inplace: carry out model transformations in-place, the original module is mutated
        mapping: correspondence between original module types and quantized counterparts

    Return:
        Quantized model.
    """
    torch._C._log_api_usage_once("quantization_api.quantize.quantize")
    if mapping is None:
        mapping = get_default_static_quant_module_mappings()
    if not inplace:
        model = copy.deepcopy(model)
    model.eval()
    prepare(model, inplace=True)
    run_fn(model, *run_args)
    convert(model, mapping, inplace=True)
    return model


@typing_extensions.deprecated(DEPRECATION_WARNING)
def quantize_dynamic(
    model, qconfig_spec=None, dtype=torch.qint8, mapping=None, inplace=False
):
    r"""Converts a float model to dynamic (i.e. weights-only) quantized model.

    Replaces specified modules with dynamic weight-only quantized versions and output the quantized model.

    For simplest usage provide `dtype` argument that can be float16 or qint8. Weight-only quantization
    by default is performed for layers with large weights size - i.e. Linear and RNN variants.

    Fine grained control is possible with `qconfig` and `mapping` that act similarly to `quantize()`.
    If `qconfig` is provided, the `dtype` argument is ignored.

    Args:
        model: input model
        qconfig_spec: Either:

            - A dictionary that maps from name or type of submodule to quantization
              configuration, qconfig applies to all submodules of a given
              module unless qconfig for the submodules are specified (when the
              submodule already has qconfig attribute). Entries in the dictionary
              need to be QConfig instances.

            - A set of types and/or submodule names to apply dynamic quantization to,
              in which case the `dtype` argument is used to specify the bit-width

        inplace: carry out model transformations in-place, the original module is mutated
        mapping: maps type of a submodule to a type of corresponding dynamically quantized version
            with which the submodule needs to be replaced

    """
    torch._C._log_api_usage_once("quantization_api.quantize.quantize_dynamic")
    if qconfig_spec is None:
        if dtype == torch.qint8:
            qconfig_spec = {
                nn.Linear: default_dynamic_qconfig,
                nn.LSTM: default_dynamic_qconfig,
                nn.GRU: default_dynamic_qconfig,
                nn.LSTMCell: default_dynamic_qconfig,
                nn.RNNCell: default_dynamic_qconfig,
                nn.GRUCell: default_dynamic_qconfig,
            }
        elif dtype == torch.float16:
            qconfig_spec = {
                nn.Linear: float16_dynamic_qconfig,
                nn.LSTM: float16_dynamic_qconfig,
                nn.GRU: float16_dynamic_qconfig,
                nn.LSTMCell: float16_dynamic_qconfig,
                nn.RNNCell: float16_dynamic_qconfig,
                nn.GRUCell: float16_dynamic_qconfig,
            }
        elif dtype == torch.quint8:
            qconfig_spec = {
                nn.EmbeddingBag: float_qparams_weight_only_qconfig,
                nn.Embedding: float_qparams_weight_only_qconfig,
            }
        elif dtype == torch.quint4x2:
            qconfig_spec = {
                nn.EmbeddingBag: float_qparams_weight_only_qconfig_4bit,
            }
        else:
            raise ValueError(
                f"Don't know how to quantize with default settings for {dtype}. Provide full qconfig please"
            )
    elif isinstance(qconfig_spec, set):
        if dtype is torch.qint8:
            default_qconfig = default_dynamic_qconfig
        elif dtype is torch.float16:
            default_qconfig = float16_dynamic_qconfig
        elif dtype is torch.quint8:
            default_qconfig = float_qparams_weight_only_qconfig
        elif dtype is torch.quint4x2:
            default_qconfig = float_qparams_weight_only_qconfig_4bit
        else:
            raise RuntimeError(
                "Unknown dtype specified for quantize_dynamic: ", str(dtype)
            )
        qconfig_spec = dict(zip(qconfig_spec, itertools.repeat(default_qconfig)))

    if mapping is None:
        mapping = get_default_dynamic_quant_module_mappings()

    if not inplace:
        model = copy.deepcopy(model)
    model.eval()
    propagate_qconfig_(model, qconfig_spec)
    convert(model, mapping, inplace=True)
    return model


@typing_extensions.deprecated(DEPRECATION_WARNING)
def prepare_qat(model, mapping=None, inplace=False):
    r"""
    Prepares a copy of the model for quantization calibration or
    quantization-aware training and converts it to quantized version.

    Quantization configuration should be assigned preemptively
    to individual submodules in `.qconfig` attribute.

    Args:
        model: input model to be modified in-place
        mapping: dictionary that maps float modules to quantized modules to be
                 replaced.
        inplace: carry out model transformations in-place, the original module
                 is mutated
    """
    torch._C._log_api_usage_once("quantization_api.quantize.prepare_qat")
    if not model.training:
        raise AssertionError("prepare_qat only works on models in training mode")
    if mapping is None:
        mapping = get_default_qat_module_mappings()

    if not inplace:
        model = copy.deepcopy(model)

    propagate_qconfig_(model, qconfig_dict=None)
    convert(model, mapping=mapping, inplace=True, remove_qconfig=False)
    prepare(model, observer_non_leaf_module_list=set(mapping.values()), inplace=True)
    return model


@typing_extensions.deprecated(DEPRECATION_WARNING)
def quantize_qat(model, run_fn, run_args, inplace=False):
    r"""Do quantization aware training and output a quantized model

    Args:
        model: input model
        run_fn: a function for evaluating the prepared model, can be a
                function that simply runs the prepared model or a training
                loop
        run_args: positional arguments for `run_fn`

    Return:
        Quantized model.
    """
    torch._C._log_api_usage_once("quantization_api.quantize.quantize_qat")
    if not inplace:
        model = copy.deepcopy(model)
    model.train()
    prepare_qat(model, inplace=True)
    run_fn(model, *run_args)
    convert(model, inplace=True)
    return model


@typing_extensions.deprecated(DEPRECATION_WARNING)
def convert(
    module,
    mapping=None,
    inplace=False,
    remove_qconfig=True,
    is_reference=False,
    convert_custom_config_dict=None,
    use_precomputed_fake_quant=False,
):
    r"""Converts submodules in input module to a different module according to `mapping`
    by calling `from_float` method on the target module class. And remove qconfig at the
    end if remove_qconfig is set to True.

    Args:
        `module`: prepared and calibrated module
        `mapping`: a dictionary that maps from source module type to target
                   module type, can be overwritten to allow swapping user defined
                   Modules
        `inplace`: carry out model transformations in-place, the original module
                   is mutated
        `convert_custom_config_dict`: custom configuration dictionary for convert function
        `use_precomputed_fake_quant`: a flag to enable use of precomputed fake quant

    .. code-block:: python

       # Example of convert_custom_config_dict:
       convert_custom_config_dict = {
           # user will manually define the corresponding quantized
           # module class which has a from_observed class method that converts
           # observed custom module to quantized custom module
           "observed_to_quantized_custom_module_class": {
               ObservedCustomModule: QuantizedCustomModule
           }
       }

    """
    torch._C._log_api_usage_once("quantization_api.quantize.convert")
    if not inplace:
        module = copy.deepcopy(module)
    _convert(
        module,
        mapping,
        inplace=True,
        is_reference=is_reference,
        convert_custom_config_dict=convert_custom_config_dict,
        use_precomputed_fake_quant=use_precomputed_fake_quant,
    )
    if remove_qconfig:
        _remove_qconfig(module)
    return module


def _convert(
    module,
    mapping=None,
    inplace=False,
    is_reference=False,
    convert_custom_config_dict=None,
    use_precomputed_fake_quant=False,
):
    r"""Converts submodules in input module to a different module according to `mapping`
    by calling `from_float` method on the target module class

    Args:
        module: input module
        mapping: a dictionary that maps from source module type to target
                 module type, can be overwritten to allow swapping user defined
                 Modules
        inplace: carry out model transformations in-place, the original module
                 is mutated
        is_reference: a flag to enable quantized reference module
        use_precomputed_fake_quant: a flag to enable use of precomputed fake quant

    """
    if mapping is None:
        mapping = (
            get_default_static_quant_reference_module_mappings()
            if is_reference
            else get_default_static_quant_module_mappings()
        )
    if convert_custom_config_dict is None:
        convert_custom_config_dict = get_default_custom_config_dict()
    custom_module_class_mapping = convert_custom_config_dict.get(
        "observed_to_quantized_custom_module_class", {}
    )

    if not inplace:
        module = copy.deepcopy(module)
    reassign = {}
    for name, mod in module.named_children():
        # both fused modules and observed custom modules are
        # swapped as one unit
        if (
            not isinstance(mod, _FusedModule)
            and type_before_parametrizations(mod) not in custom_module_class_mapping
        ):
            _convert(
                mod,
                mapping,
                True,  # inplace
                is_reference,
                convert_custom_config_dict,
                use_precomputed_fake_quant=use_precomputed_fake_quant,
            )
        reassign[name] = swap_module(
            mod, mapping, custom_module_class_mapping, use_precomputed_fake_quant
        )

    for key, value in reassign.items():
        module._modules[key] = value

    return module


def swap_module(
    mod, mapping, custom_module_class_mapping, use_precomputed_fake_quant=False
):
    r"""Swaps the module if it has a quantized counterpart and it has an
    `observer` attached.

    Args:
        mod: input module
        mapping: a dictionary that maps from nn module to nnq module

    Return:
        The corresponding quantized module of `mod`
    """
    new_mod = mod
    if hasattr(mod, "qconfig") and mod.qconfig is not None:
        swapped = False
        if type_before_parametrizations(mod) in custom_module_class_mapping:
            new_mod = custom_module_class_mapping[
                type_before_parametrizations(mod)
            ].from_observed(mod)
            swapped = True
        elif type_before_parametrizations(mod) in mapping:
            qmod = mapping[type_before_parametrizations(mod)]
            if hasattr(qmod, "_IS_REFERENCE") and qmod._IS_REFERENCE:
                if mod.qconfig is None:
                    raise AssertionError(
                        "module qconfig must not be None when swapping to reference module"
                    )
                weight_post_process = mod.qconfig.weight()
                weight_post_process(mod.weight)
                weight_qparams = get_qparam_dict(weight_post_process)
                new_mod = qmod.from_float(mod, weight_qparams)
            else:
                sig = inspect.signature(qmod.from_float)
                if "use_precomputed_fake_quant" in sig.parameters:
                    new_mod = qmod.from_float(
                        mod, use_precomputed_fake_quant=use_precomputed_fake_quant
                    )
                else:
                    new_mod = qmod.from_float(mod)
            swapped = True

        if swapped:
            # Preserve module's pre forward hooks. They'll be called on quantized input
            for pre_hook_fn in mod._forward_pre_hooks.values():
                new_mod.register_forward_pre_hook(pre_hook_fn)
            # Preserve module's post forward hooks except _observer_forward_hook
            # After convert they'll work with quantized output
            for hook_fn in mod._forward_hooks.values():
                if hook_fn is not _observer_forward_hook:
                    new_mod.register_forward_hook(hook_fn)

            # respect device affinity when swapping modules
            devices = _get_unique_devices_(mod)
            if not (
                len(devices) <= 1
                or (len(devices) == 2 and torch.device("meta") in devices)
            ):
                raise AssertionError(
                    f"swap_module only works with cpu or single-device CUDA modules, but got devices {devices}"
                )
            device = next(iter(devices)) if len(devices) > 0 else None
            if device:
                new_mod.to(device)
    return new_mod


def _get_observer_dict(mod, target_dict, prefix=""):
    r"""Traverse the modules and save all observers into dict.
    This is mainly used for quantization accuracy debug
    Args:
        mod: the top module we want to save all observers
        prefix: the prefix for the current module
        target_dict: the dictionary used to save all the observers
    """

    def get_prefix(prefix):
        return prefix if prefix == "" else prefix + "."

    if hasattr(mod, "activation_post_process"):
        target_dict[get_prefix(prefix) + "activation_post_process"] = (
            mod.activation_post_process
        )
    for name, child in mod.named_children():
        module_prefix = get_prefix(prefix) + name if prefix else name
        _get_observer_dict(child, target_dict, module_prefix)
<|endoftext|>import copy
from collections.abc import Callable
from typing import Any

import torch
import torch.ao.nn as ao_nn
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.qat as nniqat
import torch.ao.nn.intrinsic.quantized as nniq
import torch.ao.nn.intrinsic.quantized.dynamic as nniqd
import torch.ao.nn.qat as nnqat
import torch.ao.nn.qat.dynamic as nnqatd
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
import torch.ao.nn.quantized.reference as nnqr

# Because `torch.ao.nn` uses lazy imports, we need to make
# sure we import the contents explicitly here.
import torch.ao.nn.sparse
import torch.nn.functional as F
from torch import nn
from torch.ao.quantization.fake_quantize import (
    default_fixed_qparams_range_0to1_fake_quant,
    default_fixed_qparams_range_neg1to1_fake_quant,
)
from torch.ao.quantization.stubs import DeQuantStub, QuantStub
from torch.ao.quantization.utils import get_combined_dict
from torch.nn.utils.parametrize import type_before_parametrizations


__all__ = [
    "DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS",
    "DEFAULT_STATIC_QUANT_MODULE_MAPPINGS",
    "DEFAULT_QAT_MODULE_MAPPINGS",
    "DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS",
    "DEFAULT_FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS",
    "DEFAULT_MODULE_TO_ACT_POST_PROCESS",
    "DEFAULT_STATIC_SPARSE_QUANT_MODULE_MAPPINGS",
    "DEFAULT_DYNAMIC_SPARSE_QUANT_MODULE_MAPPINGS",
    "no_observer_set",
    "get_default_static_quant_module_mappings",
    "get_default_static_quant_reference_module_mappings",
    "get_embedding_static_quant_module_mappings",
    "get_default_static_sparse_quant_module_mappings",
    "get_static_quant_module_class",
    "get_dynamic_quant_module_class",
    "get_default_qat_module_mappings",
    "get_embedding_qat_module_mappings",
    "get_default_dynamic_quant_module_mappings",
    "get_default_dynamic_sparse_quant_module_mappings",
    "get_default_qconfig_propagation_list",
    "get_default_compare_output_module_list",
    "get_default_float_to_quantized_operator_mappings",
    "get_quantized_operator",
]

# Default map for swapping float module to reference quantized modules
DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS: dict[Callable, Any] = {
    QuantStub: nnq.Quantize,
    DeQuantStub: nnq.DeQuantize,
    nn.Linear: nnqr.Linear,
    nn.Conv1d: nnqr.Conv1d,
    nn.Conv2d: nnqr.Conv2d,
    nn.Conv3d: nnqr.Conv3d,
    nn.ConvTranspose1d: nnqr.ConvTranspose1d,
    nn.ConvTranspose2d: nnqr.ConvTranspose2d,
    nn.ConvTranspose3d: nnqr.ConvTranspose3d,
    nn.Embedding: nnqr.Embedding,
    nn.EmbeddingBag: nnqr.EmbeddingBag,
    nn.GRUCell: nnqr.GRUCell,
    nn.LSTMCell: nnqr.LSTMCell,
    nn.RNNCell: nnqr.RNNCell,
    nn.LSTM: nnqr.LSTM,
}

# Default map for swapping float module to quantized ones
DEFAULT_STATIC_QUANT_MODULE_MAPPINGS: dict[Callable, Any] = {
    QuantStub: nnq.Quantize,
    DeQuantStub: nnq.DeQuantize,
    nn.BatchNorm2d: nnq.BatchNorm2d,
    nn.BatchNorm3d: nnq.BatchNorm3d,
    nn.Dropout: nnq.Dropout,
    nn.Conv1d: nnq.Conv1d,
    nn.Conv2d: nnq.Conv2d,
    nn.Conv3d: nnq.Conv3d,
    nn.ConvTranspose1d: nnq.ConvTranspose1d,
    nn.ConvTranspose2d: nnq.ConvTranspose2d,
    nn.ConvTranspose3d: nnq.ConvTranspose3d,
    nn.ELU: nnq.ELU,
    nn.Embedding: nnq.Embedding,
    nn.EmbeddingBag: nnq.EmbeddingBag,
    nn.GroupNorm: nnq.GroupNorm,
    nn.Hardswish: nnq.Hardswish,
    nn.InstanceNorm1d: nnq.InstanceNorm1d,
    nn.InstanceNorm2d: nnq.InstanceNorm2d,
    nn.InstanceNorm3d: nnq.InstanceNorm3d,
    nn.LayerNorm: nnq.LayerNorm,
    nn.LeakyReLU: nnq.LeakyReLU,
    nn.modules.linear.NonDynamicallyQuantizableLinear: nnq.Linear,
    nn.Linear: nnq.Linear,
    nn.ReLU6: nnq.ReLU6,
    nn.PReLU: nnq.PReLU,
    # Wrapper Modules:
    nnq.FloatFunctional: nnq.QFunctional,
    # Intrinsic modules:
    nni.BNReLU2d: nniq.BNReLU2d,
    nni.BNReLU3d: nniq.BNReLU3d,
    nni.ConvReLU1d: nniq.ConvReLU1d,
    nni.ConvReLU2d: nniq.ConvReLU2d,
    nni.ConvReLU3d: nniq.ConvReLU3d,
    nni.ConvAdd2d: nniq.ConvAdd2d,
    nni.ConvAddReLU2d: nniq.ConvAddReLU2d,
    nni.LinearReLU: nniq.LinearReLU,
    nni.LinearLeakyReLU: nniq.LinearLeakyReLU,
    nni.LinearTanh: nniq.LinearTanh,
    nniqat.ConvBn1d: nnq.Conv1d,
    nniqat.ConvBn2d: nnq.Conv2d,
    nniqat.ConvBn3d: nnq.Conv3d,
    nniqat.ConvBnReLU1d: nniq.ConvReLU1d,
    nniqat.ConvBnReLU2d: nniq.ConvReLU2d,
    nniqat.ConvBnReLU3d: nniq.ConvReLU3d,
    nniqat.ConvReLU2d: nniq.ConvReLU2d,
    nniqat.ConvReLU3d: nniq.ConvReLU3d,
    nniqat.LinearReLU: nniq.LinearReLU,
    nniqat.LinearBn1d: nnq.Linear,
    # QAT modules:
    nnqat.Linear: nnq.Linear,
    nnqat.Conv2d: nnq.Conv2d,
    nnqat.Conv3d: nnq.Conv3d,
}

# Default map for swapping float module to qat modules
DEFAULT_QAT_MODULE_MAPPINGS: dict[Callable, Any] = {
    nn.Conv2d: nnqat.Conv2d,
    nn.Conv3d: nnqat.Conv3d,
    nn.Linear: nnqat.Linear,
    nn.modules.linear.NonDynamicallyQuantizableLinear: nnqat.Linear,
    # Intrinsic modules:
    nni.ConvBn1d: nniqat.ConvBn1d,
    nni.ConvBn2d: nniqat.ConvBn2d,
    nni.ConvBn3d: nniqat.ConvBn3d,
    nni.ConvBnReLU1d: nniqat.ConvBnReLU1d,
    nni.ConvBnReLU2d: nniqat.ConvBnReLU2d,
    nni.ConvBnReLU3d: nniqat.ConvBnReLU3d,
    nni.ConvReLU2d: nniqat.ConvReLU2d,
    nni.ConvReLU3d: nniqat.ConvReLU3d,
    nni.LinearReLU: nniqat.LinearReLU,
    nni.LinearBn1d: nniqat.LinearBn1d,
}

# Default map for swapping dynamic modules
DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS: dict[Callable, Any] = {
    nn.GRUCell: nnqd.GRUCell,
    nn.Linear: nnqd.Linear,
    nnqatd.Linear: nnqd.Linear,
    nn.modules.linear.NonDynamicallyQuantizableLinear: nnqd.Linear,
    nn.LSTM: nnqd.LSTM,
    nn.GRU: nnqd.GRU,
    nn.LSTMCell: nnqd.LSTMCell,
    nn.RNNCell: nnqd.RNNCell,
    nni.LinearReLU: nniqd.LinearReLU,
    nn.EmbeddingBag: nnq.EmbeddingBag,
    nn.Embedding: nnq.Embedding,
    # Don't want to enable these by default because the numerical
    # accuracy is poor compared to other dynamic ops
    # nn.Conv1d: nnqd.Conv1d,
    # nn.Conv2d: nnqd.Conv2d,
    # nn.Conv3d: nnqd.Conv3d,
    # nn.ConvTranspose1d: nnqd.ConvTranspose1d,
    # nn.ConvTranspose2d: nnqd.ConvTranspose2d,
    # nn.ConvTranspose3d: nnqd.ConvTranspose3d,
}

# Allowlist for propagating the qconfig
_INCLUDE_QCONFIG_PROPAGATE_LIST: set[Callable] = {
    nn.Sequential,
}

# Default mapping from floating point function or torch ops to quantized ops
# TODO: merge with default static mapping
DEFAULT_FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS: dict[Callable | str, Callable] = {
    F.elu: torch.ops.quantized.elu,
    F.hardswish: torch.ops.quantized.hardswish,
    F.instance_norm: torch.ops.quantized.instance_norm,
    F.layer_norm: torch.ops.quantized.layer_norm,
    F.leaky_relu: torch.ops.quantized.leaky_relu,
    F.dropout: torch.ops.quantized.dropout,
}

# mapping from module to output activation post process class
DEFAULT_MODULE_TO_ACT_POST_PROCESS: dict[Callable, Callable] = {
    nn.Hardsigmoid: default_fixed_qparams_range_0to1_fake_quant,
    nn.Sigmoid: default_fixed_qparams_range_0to1_fake_quant,
    nn.Softmax: default_fixed_qparams_range_0to1_fake_quant,
    nn.Tanh: default_fixed_qparams_range_neg1to1_fake_quant,
}

# Default map for swapping float module to static sparse quantized ones
DEFAULT_STATIC_SPARSE_QUANT_MODULE_MAPPINGS: dict[Callable, Any] = {
    nn.Linear: ao_nn.sparse.quantized.Linear
}

# Default map for swapping float module to dynamic sparse quantized ones
DEFAULT_DYNAMIC_SPARSE_QUANT_MODULE_MAPPINGS: dict[Callable, Any] = {
    nn.Linear: ao_nn.sparse.quantized.dynamic.Linear
}


def no_observer_set() -> set[Any]:
    r"""These modules cannot have observers inserted by default."""
    no_observers = {nn.quantizable.LSTM, nn.quantizable.MultiheadAttention}
    return no_observers


def get_default_static_quant_module_mappings() -> dict[Callable, Any]:
    """Get module mapping for post training static quantization"""
    return copy.deepcopy(DEFAULT_STATIC_QUANT_MODULE_MAPPINGS)


def get_default_static_quant_reference_module_mappings() -> dict[Callable, Any]:
    """Get reference module mapping for post training static quantization"""
    return copy.deepcopy(DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS)


def get_embedding_static_quant_module_mappings() -> dict[Callable, Any]:
    """Get module mapping, including mapping for embedding QAT"""
    mapping = copy.deepcopy(DEFAULT_STATIC_QUANT_MODULE_MAPPINGS)
    mapping[nnqat.EmbeddingBag] = nnq.EmbeddingBag
    mapping[nnqat.Embedding] = nnq.Embedding
    return mapping


def get_default_static_sparse_quant_module_mappings() -> dict[Callable, Any]:
    """Get module mapping for post training static sparse quantization"""
    return copy.deepcopy(DEFAULT_STATIC_SPARSE_QUANT_MODULE_MAPPINGS)


def get_static_quant_module_class(
    float_module_class: Callable,
    additional_static_quant_mapping: dict[Callable, Any] | None = None,
    is_reference: bool = False,
) -> Any:
    r"""Get the statically quantized module class corresponding to
    the floating point module class
    """
    if additional_static_quant_mapping is None:
        additional_static_quant_mapping = {}
    all_mappings = get_combined_dict(
        DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS
        if is_reference
        else DEFAULT_STATIC_QUANT_MODULE_MAPPINGS,
        additional_static_quant_mapping,
    )
    static_quant_module_class = all_mappings.get(float_module_class, None)
    if static_quant_module_class is None:
        raise AssertionError(
            f"Floating point module class {str(float_module_class)}"
            + " does not have a corresponding quantized module class"
        )
    return copy.deepcopy(static_quant_module_class)


def get_dynamic_quant_module_class(
    float_module_class: Callable,
    additional_dynamic_quant_mapping: dict[Callable, Any] | None = None,
) -> Any:
    r"""Get the dynamically quantized module class corresponding to
    the floating point module class
    """
    if additional_dynamic_quant_mapping is None:
        additional_dynamic_quant_mapping = {}
    all_mappings = get_combined_dict(
        DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS, additional_dynamic_quant_mapping
    )
    dynamic_quant_module_class = all_mappings.get(float_module_class, None)
    if dynamic_quant_module_class is None:
        raise AssertionError(
            f"Floating point module class {str(float_module_class)}"
            + " does not have a corresponding quantized module class"
        )
    return copy.deepcopy(dynamic_quant_module_class)


def get_default_qat_module_mappings() -> dict[Callable, Any]:
    """Get default module mapping for quantization aware training"""
    return copy.deepcopy(DEFAULT_QAT_MODULE_MAPPINGS)


def get_embedding_qat_module_mappings() -> dict[Callable, Any]:
    """Get module mapping for quantization aware training
    This is includes default values in addition to
    enabling qat for embeddings.
    """
    mapping = copy.deepcopy(DEFAULT_QAT_MODULE_MAPPINGS)
    mapping[nn.EmbeddingBag] = nnqat.EmbeddingBag
    mapping[nn.Embedding] = nnqat.Embedding
    return mapping


def get_default_dynamic_quant_module_mappings() -> dict[Callable, Any]:
    """Get module mapping for post training dynamic quantization"""
    return DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS


def get_default_dynamic_sparse_quant_module_mappings() -> dict[Callable, Any]:
    """Get module mapping for post training dynamic sparse quantization"""
    return DEFAULT_DYNAMIC_SPARSE_QUANT_MODULE_MAPPINGS


def get_default_qconfig_propagation_list() -> set[Callable]:
    """Get the default list of module types that we'll attach qconfig
    attribute to in prepare
    """
    QCONFIG_PROPAGATE_MODULE_CLASS_LIST = (
        set(DEFAULT_STATIC_QUANT_MODULE_MAPPINGS.keys())
        | set(DEFAULT_QAT_MODULE_MAPPINGS.keys())
        | set(DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS.keys())
        | _INCLUDE_QCONFIG_PROPAGATE_LIST
    )
    return copy.deepcopy(QCONFIG_PROPAGATE_MODULE_CLASS_LIST)


def get_default_compare_output_module_list() -> set[Callable]:
    """Get list of module class types that we will record output
    in numeric suite
    """
    NUMERIC_SUITE_COMPARE_MODEL_OUTPUT_MODULE_LIST = (
        set(DEFAULT_STATIC_QUANT_MODULE_MAPPINGS.values())
        | set(DEFAULT_QAT_MODULE_MAPPINGS.values())
        | set(DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS.values())
        | set(DEFAULT_STATIC_QUANT_MODULE_MAPPINGS.keys())
        | set(DEFAULT_QAT_MODULE_MAPPINGS.keys())
        | set(DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS.keys())
        | _INCLUDE_QCONFIG_PROPAGATE_LIST
    )
    return copy.deepcopy(NUMERIC_SUITE_COMPARE_MODEL_OUTPUT_MODULE_LIST)


def get_default_float_to_quantized_operator_mappings() -> dict[
    Callable | str, Callable
]:
    return copy.deepcopy(DEFAULT_FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS)


# TODO: merge with get_static_quant_module_class
def get_quantized_operator(float_op: Callable | str) -> Callable:
    """Get the quantized operator corresponding to the float operator"""
    quantized_op = DEFAULT_FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS.get(float_op)
    if quantized_op is None:
        raise AssertionError(
            f"Operator {str(float_op)} does not have corresponding quantized op"
        )
    return quantized_op


def _get_special_act_post_process(module: torch.nn.Module) -> Callable | None:
    r"""Get the special activation post process for `module`, this has
    higher priority than the activation post process in `qconfig`
    e.g.
    input: torch.nn.Sigmoid
    output: default_affine_fixed_qparam_fake_quant
    """
    return DEFAULT_MODULE_TO_ACT_POST_PROCESS.get(type_before_parametrizations(module))


def _has_special_act_post_process(module: torch.nn.Module) -> bool:
    return module.training and type(module) in DEFAULT_MODULE_TO_ACT_POST_PROCESS
<|endoftext|>from typing import Any

import torch
from torch import nn
from torch.ao.quantization import QConfig


__all__ = ["QuantStub", "DeQuantStub", "QuantWrapper"]


class QuantStub(nn.Module):
    r"""Quantize stub module, before calibration, this is same as an observer,
    it will be swapped as `nnq.Quantize` in `convert`.

    Args:
        qconfig: quantization configuration for the tensor,
            if qconfig is not provided, we will get qconfig from parent modules
    """

    def __init__(self, qconfig: QConfig | None = None):
        super().__init__()
        if qconfig:
            self.qconfig = qconfig

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DeQuantStub(nn.Module):
    r"""Dequantize stub module, before calibration, this is same as identity,
    this will be swapped as `nnq.DeQuantize` in `convert`.

    Args:
        qconfig: quantization configuration for the tensor,
            if qconfig is not provided, we will get qconfig from parent modules
    """

    def __init__(self, qconfig: Any | None = None):
        super().__init__()
        if qconfig:
            self.qconfig = qconfig

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class QuantWrapper(nn.Module):
    r"""A wrapper class that wraps the input module, adds QuantStub and
    DeQuantStub and surround the call to module with call to quant and dequant
    modules.

    This is used by the `quantization` utility functions to add the quant and
    dequant modules, before `convert` function `QuantStub` will just be observer,
    it observes the input tensor, after `convert`, `QuantStub`
    will be swapped to `nnq.Quantize` which does actual quantization. Similarly
    for `DeQuantStub`.
    """

    quant: QuantStub
    dequant: DeQuantStub
    module: nn.Module

    def __init__(self, module: nn.Module):
        super().__init__()
        qconfig = getattr(module, "qconfig", None)
        self.add_module("quant", QuantStub(qconfig))
        self.add_module("dequant", DeQuantStub(qconfig))
        self.add_module("module", module)
        self.train(module.training)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.quant(X)
        X = self.module(X)
        return self.dequant(X)
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

from collections import OrderedDict
from typing import Any, TYPE_CHECKING

import torch

from .fake_quantize import default_weight_fake_quant, FixedQParamsFakeQuantize
from .observer import (
    _PartialWrapper,
    default_fixed_qparams_range_0to1_observer,
    default_fixed_qparams_range_neg1to1_observer,
    default_placeholder_observer,
    default_weight_observer,
)
from .qconfig import (
    default_quint8_weight_qconfig,
    default_reuse_input_qconfig,
    default_symmetric_qnnpack_qat_qconfig,
    default_symmetric_qnnpack_qconfig,
    get_default_qat_qconfig,
    get_default_qconfig,
    QConfig,
    QConfigAny,
)


if TYPE_CHECKING:
    from collections.abc import Callable


__all__ = [
    "get_default_qconfig_mapping",
    "get_default_qat_qconfig_mapping",
    "QConfigMapping",
]


# TODO: replace all usages with these constants
_GLOBAL_DICT_KEY = ""
_OBJECT_TYPE_DICT_KEY = "object_type"
_MODULE_NAME_REGEX_DICT_KEY = "module_name_regex"
_MODULE_NAME_DICT_KEY = "module_name"
_MODULE_NAME_OBJECT_TYPE_ORDER_DICT_KEY = "module_name_object_type_order"

# TODO: derive this map from the BackendConfig
_FIXED_QPARAMS_OP_TO_OBSERVER: dict[Callable | str, _PartialWrapper] = {
    torch.nn.Hardsigmoid: default_fixed_qparams_range_0to1_observer,
    torch.nn.functional.hardsigmoid: default_fixed_qparams_range_0to1_observer,
    "hardsigmoid": default_fixed_qparams_range_0to1_observer,
    "hardsigmoid_": default_fixed_qparams_range_0to1_observer,
    torch.nn.Sigmoid: default_fixed_qparams_range_0to1_observer,
    torch.sigmoid: default_fixed_qparams_range_0to1_observer,
    "sigmoid": default_fixed_qparams_range_0to1_observer,
    "sigmoid_": default_fixed_qparams_range_0to1_observer,
    torch.nn.Softmax: default_fixed_qparams_range_0to1_observer,
    torch.nn.Tanh: default_fixed_qparams_range_neg1to1_observer,
    torch.tanh: default_fixed_qparams_range_neg1to1_observer,
    "tanh": default_fixed_qparams_range_neg1to1_observer,
    "tanh_": default_fixed_qparams_range_neg1to1_observer,
}


def _get_default_qconfig_mapping(
    is_qat: bool, backend: str, version: int
) -> QConfigMapping:
    """
    Return the default QConfigMapping for the given quantization type and backend.
    """
    if is_qat:
        qconfig = get_default_qat_qconfig(backend, version)
    else:
        qconfig = get_default_qconfig(backend, version)
    default_weight = default_weight_fake_quant if is_qat else default_weight_observer

    # default_per_channel_weight_observer is not currently compatible with fbgemm backend
    # so we have to modify the weight observer to default_weight_observer or another
    # per tensor supported observer.
    # see https://github.com/pytorch/pytorch/issues/47535
    if backend in ("fbgemm", "x86"):
        qconfig_transpose = QConfig(
            activation=qconfig.activation, weight=default_weight
        )
    else:
        qconfig_transpose = qconfig

    # currently layernorm only supports float weights
    # we have to add this because otherwise there will be an extra quantize-dequantize pair
    qconfig_layernorm = QConfig(
        activation=qconfig.activation, weight=default_placeholder_observer
    )

    qconfig_mapping = (
        QConfigMapping()
        .set_global(qconfig)
        .set_object_type("reshape", default_reuse_input_qconfig)
        .set_object_type(torch.nn.ConvTranspose1d, qconfig_transpose)
        .set_object_type(torch.nn.ConvTranspose2d, qconfig_transpose)
        .set_object_type(torch.nn.ConvTranspose3d, qconfig_transpose)
        .set_object_type(torch.nn.functional.conv_transpose1d, qconfig_transpose)
        .set_object_type(torch.nn.functional.conv_transpose2d, qconfig_transpose)
        .set_object_type(torch.nn.functional.conv_transpose3d, qconfig_transpose)
        .set_object_type(torch.nn.functional.layer_norm, qconfig_layernorm)
        .set_object_type(torch.nn.LayerNorm, qconfig_layernorm)
        .set_object_type(torch.nn.PReLU, default_quint8_weight_qconfig)
    )
    # Use special observers for ops with fixed qparams
    fixed_qparams_observer_to_qconfig: dict[Any, QConfigAny] = {}
    for fixed_qparams_op, observer in _FIXED_QPARAMS_OP_TO_OBSERVER.items():
        if observer in fixed_qparams_observer_to_qconfig:
            fixed_qparams_qconfig = fixed_qparams_observer_to_qconfig[observer]
        else:
            if is_qat:
                activation = FixedQParamsFakeQuantize.with_args(observer=observer)
            else:
                activation = observer
            fixed_qparams_qconfig = QConfig(
                activation=activation, weight=default_weight
            )
            fixed_qparams_observer_to_qconfig[observer] = fixed_qparams_qconfig
        qconfig_mapping.set_object_type(fixed_qparams_op, fixed_qparams_qconfig)

    # TODO Currently it's required that separate ops in a fused op/module have the same qconfig.
    #      Need to be able to support fusion of ops with different qconfigs

    return qconfig_mapping


def get_default_qconfig_mapping(backend="x86", version=0) -> QConfigMapping:
    """
    Return the default QConfigMapping for post training quantization.

    Args:
      * ``backend`` (str) : the quantization backend for the default qconfig mapping, should be
         one of ["x86" (default), "fbgemm", "qnnpack", "onednn"]
      * ``version`` (int) : the version for the default qconfig mapping
    """
    # TODO: add assert for backend choices
    return _get_default_qconfig_mapping(False, backend, version)


def get_default_qat_qconfig_mapping(backend="x86", version=1) -> QConfigMapping:
    """
    Return the default QConfigMapping for quantization aware training.

    Args:
      * ``backend`` (str) : the quantization backend for the default qconfig mapping, should be
         one of ["x86" (default), "fbgemm", "qnnpack", "onednn"]
      * ``version`` (int) : the version for the default qconfig mapping
    """
    return _get_default_qconfig_mapping(True, backend, version)


def _get_symmetric_qnnpack_qconfig_mapping() -> QConfigMapping:
    """
    Return a QConfigMapping that uses `torch.ao.quantization.default_symmetric_qnnpack_qconfig`
    as the default QConfig.
    """
    default_qconfig = default_symmetric_qnnpack_qconfig
    return _get_default_qconfig_mapping_with_default_qconfig(
        False, "qnnpack", default_qconfig
    )


def _get_symmetric_qnnpack_qat_qconfig_mapping() -> QConfigMapping:
    """
    Return a QConfigMapping that uses `torch.ao.quantization.default_symmetric_qnnpack_qat_qconfig`
    as the default QConfig.
    """
    default_qconfig = default_symmetric_qnnpack_qat_qconfig
    return _get_default_qconfig_mapping_with_default_qconfig(
        True, "qnnpack", default_qconfig
    )


def _get_default_qconfig_mapping_with_default_qconfig(
    is_qat: bool,
    backend: str,
    default_qconfig: QConfig,
) -> QConfigMapping:
    """
    Return a QConfigMapping that uses the provided qconfig as the default QConfig.
    """
    if is_qat:
        qconfig_mapping = get_default_qat_qconfig_mapping(backend)
    else:
        qconfig_mapping = get_default_qconfig_mapping(backend)
    qconfig_mapping.set_global(default_qconfig)
    for pattern in qconfig_mapping.object_type_qconfigs:
        if pattern not in _FIXED_QPARAMS_OP_TO_OBSERVER:
            qconfig_mapping.set_object_type(pattern, default_qconfig)
    return qconfig_mapping


_QCONFIG_STYLE_ORDER: list[str] = [
    "global_qconfig",
    "object_type_qconfigs",
    "module_name_regex_qconfigs",
    "module_name_qconfigs",
    "module_name_object_type_order_qconfigs",
]


class QConfigMapping:
    """
    Mapping from model ops to :class:`torch.ao.quantization.QConfig` s.

    The user can specify QConfigs using the following methods (in increasing match priority):

        ``set_global`` : sets the global (default) QConfig

        ``set_object_type`` : sets the QConfig for a given module type, function, or method name

        ``set_module_name_regex`` : sets the QConfig for modules matching the given regex string

        ``set_module_name`` : sets the QConfig for modules matching the given module name

        ``set_module_name_object_type_order`` : sets the QConfig for modules matching a combination
        of the given module name, object type, and the index at which the module appears

    Example usage::

        qconfig_mapping = QConfigMapping()
            .set_global(global_qconfig)
            .set_object_type(torch.nn.Linear, qconfig1)
            .set_object_type(torch.nn.ReLU, qconfig1)
            .set_module_name_regex("foo.*bar.*conv[0-9]+", qconfig1)
            .set_module_name_regex("foo.*", qconfig2)
            .set_module_name("module1", qconfig1)
            .set_module_name("module2", qconfig2)
            .set_module_name_object_type_order("foo.bar", torch.nn.functional.linear, 0, qconfig3)

    """

    def __init__(self) -> None:
        # In increasing match priority:
        self.global_qconfig: QConfigAny = None
        self.object_type_qconfigs: OrderedDict[Callable | str, QConfigAny] = (
            OrderedDict()
        )
        self.module_name_regex_qconfigs: OrderedDict[str, QConfigAny] = OrderedDict()
        self.module_name_qconfigs: OrderedDict[str, QConfigAny] = OrderedDict()
        self.module_name_object_type_order_qconfigs: OrderedDict[
            tuple[str, Callable, int], QConfigAny
        ] = OrderedDict()

    def set_global(self, global_qconfig: QConfigAny) -> QConfigMapping:
        """
        Set the global (default) QConfig.
        """
        self.global_qconfig = global_qconfig
        return self

    def set_object_type(
        self, object_type: Callable | str, qconfig: QConfigAny
    ) -> QConfigMapping:
        """
        Set the QConfig for a given module type, function, or method name.
        If the QConfig for an existing object type was already set, the new QConfig will override the old one.
        """
        self.object_type_qconfigs[object_type] = qconfig
        return self

    def set_module_name_regex(
        self, module_name_regex: str, qconfig: QConfigAny
    ) -> QConfigMapping:
        """
        Set the QConfig for modules matching the given regex string.

        Regexes will be matched in the order in which they are registered through this method.
        Thus, the caller should register more specific patterns first, e.g.::

            qconfig_mapping = QConfigMapping()
                .set_module_name_regex("foo.*bar.*conv[0-9]+", qconfig1)
                .set_module_name_regex("foo.*bar.*", qconfig2)
                .set_module_name_regex("foo.*", qconfig3)

        In this example, "foo.bar.conv0" would match qconfig1, "foo.bar.linear" would match qconfig2,
        and "foo.baz.relu" would match qconfig3.

        If the QConfig for an existing module name regex was already set, the new QConfig will override the
        old one while preserving the order in which the regexes were originally registered.
        """
        self.module_name_regex_qconfigs[module_name_regex] = qconfig
        return self

    def set_module_name(self, module_name: str, qconfig: QConfigAny) -> QConfigMapping:
        """
        Set the QConfig for modules matching the given module name.
        If the QConfig for an existing module name was already set, the new QConfig will override the old one.
        """
        self.module_name_qconfigs[module_name] = qconfig
        return self

    def set_module_name_object_type_order(
        self, module_name: str, object_type: Callable, index: int, qconfig: QConfigAny
    ) -> QConfigMapping:
        """
        Set the QConfig for modules matching a combination of the given module name, object type,
        and the index at which the module appears.

        If the QConfig for an existing (module name, object type, index)  was already set, the new QConfig
        will override the old one.
        """
        self.module_name_object_type_order_qconfigs[
            (module_name, object_type, index)
        ] = qconfig
        return self

    def __repr__(self) -> str:
        output = self.__class__.__name__ + " ("
        for style_name in _QCONFIG_STYLE_ORDER:
            output += f"\n {style_name}"
            qconfigs = getattr(self, style_name)
            if isinstance(qconfigs, OrderedDict) and len(qconfigs) > 0:
                for key, qconfig in qconfigs.items():
                    output += f"\n  {key}: {qconfig}"
            else:
                output += f"\n  {qconfigs}"
        return output + "\n)"

    # TODO: remove this
    def to_dict(self) -> dict[str, Any]:
        """
        Convert this ``QConfigMapping`` to a dictionary with the following keys:

            "" (for global QConfig)

            "object_type"

            "module_name_regex"

            "module_name"

            "module_name_object_type_order"

        The values of this dictionary are lists of tuples.
        """
        return {
            _GLOBAL_DICT_KEY: self.global_qconfig,
            _OBJECT_TYPE_DICT_KEY: list(self.object_type_qconfigs.items()),
            _MODULE_NAME_REGEX_DICT_KEY: list(self.module_name_regex_qconfigs.items()),
            _MODULE_NAME_DICT_KEY: list(self.module_name_qconfigs.items()),
            _MODULE_NAME_OBJECT_TYPE_ORDER_DICT_KEY: [
                (*k, v) for k, v in self.module_name_object_type_order_qconfigs.items()
            ],
        }

    # TODO: remove this
    @classmethod
    def from_dict(cls, qconfig_dict: dict[str, Any]) -> QConfigMapping:
        """
        Create a ``QConfigMapping`` from a dictionary with the following keys (all optional):

            "" (for global QConfig)

            "object_type"

            "module_name_regex"

            "module_name"

            "module_name_object_type_order"

        The values of this dictionary are expected to be lists of tuples.
        """
        conf = cls()
        if _GLOBAL_DICT_KEY in qconfig_dict:
            conf.set_global(qconfig_dict[_GLOBAL_DICT_KEY])
        for object_type, qconfig in qconfig_dict.get(_OBJECT_TYPE_DICT_KEY, []):
            conf.set_object_type(object_type, qconfig)
        for module_name_regex, qconfig in qconfig_dict.get(
            _MODULE_NAME_REGEX_DICT_KEY, []
        ):
