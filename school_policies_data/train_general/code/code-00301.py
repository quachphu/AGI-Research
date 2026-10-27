        >>> q_add = QFunctional()
        >>> # xdoctest: +SKIP
        >>> a = torch.quantize_per_tensor(torch.tensor(3.0), 1.0, 0, torch.qint32)
        >>> b = torch.quantize_per_tensor(torch.tensor(4.0), 1.0, 0, torch.qint32)
        >>> q_add.add(a, b)  # Equivalent to ``torch.ops.quantized.add(a, b, 1.0, 0)``

    Valid operation names:
        - add
        - cat
        - mul
        - add_relu
        - add_scalar
        - mul_scalar
    """

    def __init__(self) -> None:
        super().__init__()
        self.scale = 1.0
        self.zero_point = 0
        self.activation_post_process = torch.nn.Identity()

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "scale"] = torch.tensor(self.scale)
        destination[prefix + "zero_point"] = torch.tensor(self.zero_point)

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
        self.scale = float(state_dict.pop(prefix + "scale"))
        self.zero_point = int(state_dict.pop(prefix + "zero_point"))
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            False,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _get_name(self):
        return "QFunctional"

    def extra_repr(self):
        return f"scale={self.scale}, zero_point={self.zero_point}"

    def forward(self, x):
        raise RuntimeError(
            "Functional is not intended to use the "
            + "'forward'. Please use the underlying operation"
        )

    r"""Operation equivalent to ``torch.ops.quantized.add``"""

    def add(self, x: Tensor, y: Tensor) -> Tensor:
        r = ops.quantized.add(x, y, scale=self.scale, zero_point=self.zero_point)
        r = self.activation_post_process(r)
        return r

    r"""Operation equivalent to ``torch.ops.quantized.add(Tensor, float)``"""

    def add_scalar(self, x: Tensor, y: float) -> Tensor:
        r = ops.quantized.add_scalar(x, y)
        # Note: this operation is not observed because the observation is not
        # needed for the quantized op.
        return r

    r"""Operation equivalent to ``torch.ops.quantized.mul(Tensor, Tensor)``"""

    def mul(self, x: Tensor, y: Tensor) -> Tensor:
        r = ops.quantized.mul(x, y, scale=self.scale, zero_point=self.zero_point)
        r = self.activation_post_process(r)
        return r

    r"""Operation equivalent to ``torch.ops.quantized.mul(Tensor, float)``"""

    def mul_scalar(self, x: Tensor, y: float) -> Tensor:
        r = ops.quantized.mul_scalar(x, y)
        # Note: this operation is not observed because the observation is not
        # needed for the quantized op.
        return r

    r"""Operation equivalent to ``torch.ops.quantized.cat``"""

    def cat(self, x: list[Tensor], dim: int = 0) -> Tensor:
        r = ops.quantized.cat(x, scale=self.scale, zero_point=self.zero_point, dim=dim)
        r = self.activation_post_process(r)
        return r

    r"""Operation equivalent to ``torch.ops.quantized.add_relu``"""

    def add_relu(self, x: Tensor, y: Tensor) -> Tensor:
        r = ops.quantized.add_relu(x, y, scale=self.scale, zero_point=self.zero_point)
        r = self.activation_post_process(r)
        return r

    r"""Operation equivalent to ``torch.ops.quantized.matmul(Tensor, Tensor)``"""

    def matmul(self, x: Tensor, y: Tensor) -> Tensor:
        r = ops.quantized.matmul(x, y, scale=self.scale, zero_point=self.zero_point)
        # Note: this operation is not observed because the observation is not
        # needed for the quantized op.
        return r

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        if type(mod) is not FloatFunctional:
            raise AssertionError(
                f"QFunctional.from_float expects an instance of FloatFunctional, "
                f"got {type(mod).__name__}"
            )
        scale, zero_point = mod.activation_post_process.calculate_qparams()  # type: ignore[operator]
        new_mod = QFunctional()
        new_mod.scale = float(scale)
        new_mod.zero_point = int(zero_point)
        return new_mod
<|endoftext|># mypy: allow-untyped-defs
import torch


__all__ = ["Dropout"]


class Dropout(torch.nn.Dropout):
    r"""This is the quantized equivalent of :class:`~torch.nn.Dropout`.
        And this is a placeholder to enable models where fp32 tensors
        had dropout to work with quantized tensors in train and eval mode.

    Args:
        p: probability of an element to be zeroed
        inplace: can optionally do the operation in-place. Default: ``False``
    """

    def forward(self, input):
        return input

    def _get_name(self):
        return "QuantizedDropout"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return cls(mod.p, mod.inplace)

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        return cls(mod.p, mod.inplace)
<|endoftext|># mypy: allow-untyped-defs
import abc
import collections
import itertools

import torch
from torch.nn.modules.module import _addindent


__all__ = [
    "WeightedQuantizedModule",
]


class WeightedQuantizedModule(torch.nn.Module, metaclass=abc.ABCMeta):
    """Wrapper for quantized modules that can be lowered from reference modules."""

    @classmethod
    @abc.abstractmethod
    def from_reference(cls, ref_module, output_scale, output_zero_point):
        raise NotImplementedError


def _get_weight_observer(observer):
    # FakeQuantize observer
    if hasattr(observer, "activation_post_process"):
        observer = observer.activation_post_process
    # UniformQuantizationObserverBase observer
    return observer


def _needs_weight_clamping(observer, dtype):
    observer = _get_weight_observer(observer)
    if dtype in [torch.qint8, torch.quint8, torch.qint32]:
        info = torch.iinfo(dtype)
        return observer.quant_min > info.min or observer.quant_max < info.max
    return False


def _clamp_weights(qweight, observer, scale, zp):
    if not _needs_weight_clamping(observer, qweight.dtype):
        return qweight

    observer = _get_weight_observer(observer)
    min_, max_ = observer.quant_min, observer.quant_max

    # Doing this because can't use torch.ops.quantized.clamp() with per_channel qscheme yet.
    qw_int_max = torch.clone(qweight.int_repr()).fill_(max_)
    qw_int_min = torch.clone(qweight.int_repr()).fill_(min_)
    qw_int = torch.minimum(torch.maximum(qweight.int_repr(), qw_int_min), qw_int_max)

    if observer.qscheme in [torch.per_tensor_symmetric, torch.per_tensor_affine]:
        qweight = torch._make_per_tensor_quantized_tensor(
            qw_int, scale.item(), zp.item()
        )
    elif observer.qscheme in [
        torch.per_channel_symmetric,
        torch.per_channel_affine,
        torch.per_channel_affine_float_qparams,
    ]:
        qweight = torch._make_per_channel_quantized_tensor(
            qw_int, scale, zp, axis=observer.ch_axis
        )
    else:
        raise ValueError("Unexpected qscheme " + observer.qscheme)
    return qweight


def _quantize_weight(float_wt, observer):
    wt_scale, wt_zp = observer.calculate_qparams()
    if observer.qscheme in [torch.per_tensor_symmetric, torch.per_tensor_affine]:
        qweight = torch.quantize_per_tensor(
            float_wt, float(wt_scale), int(wt_zp), torch.qint8
        )
        qweight = _clamp_weights(qweight, observer, wt_scale, wt_zp)
    elif observer.qscheme in [torch.per_channel_symmetric, torch.per_channel_affine]:
        wt_axis = observer.ch_axis
        qweight = torch.quantize_per_channel(
            float_wt,
            wt_scale.to(torch.double),
            wt_zp.to(torch.int64),
            wt_axis,
            torch.qint8,
        )
        qweight = _clamp_weights(qweight, observer, wt_scale, wt_zp)
    elif observer.qscheme == torch.per_channel_affine_float_qparams:
        qweight = torch.quantize_per_channel(
            float_wt,
            wt_scale.to(torch.float),
            wt_zp.to(torch.float),
            observer.ch_axis,
            observer.dtype,
        )
        qweight = _clamp_weights(qweight, observer, wt_scale, wt_zp)
    else:
        raise ValueError("Unexpected qscheme " + observer.qscheme)
    return qweight


def _ntuple_from_first(n):
    """Converts the argument to a tuple of size n
    with the first element repeated."""

    def parse(x):
        while isinstance(x, collections.abc.Sequence):
            if len(x) == n:
                break
            x = x[0]
        return tuple(itertools.repeat(x, n))

    return parse


def _hide_packed_params_repr(self, params):
    # We don't want to show `PackedParams` children, hence custom
    # `__repr__`. This is the same as nn.Module.__repr__, except the check
    # for the `params module`.
    extra_lines = []
    extra_repr = self.extra_repr()
    # empty string will be split into list ['']
    if extra_repr:
        extra_lines = extra_repr.split("\n")
    child_lines = []
    for key, module in self._modules.items():
        if isinstance(module, params):
            continue
        mod_str = repr(module)
        mod_str = _addindent(mod_str, 2)
        child_lines.append("(" + key + "): " + mod_str)
    lines = extra_lines + child_lines

    main_str = self._get_name() + "("
    if lines:
        # simple one-liner info, which most builtin Modules will use
        if len(extra_lines) == 1 and not child_lines:
            main_str += extra_lines[0]
        else:
            main_str += "\n  " + "\n  ".join(lines) + "\n"

    main_str += ")"
    return main_str


_pair_from_first = _ntuple_from_first(2)
<|endoftext|># mypy: allow-untyped-defs
r"""Quantized convolution modules."""

from typing import ClassVar, Literal

import torch
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.qat as nniqat
import torch.nn as nn
import torch.nn.functional as F
from torch._ops import ops
from torch.nn.common_types import _size_1_t
from torch.nn.modules.utils import _pair, _single, _triple
from torch.nn.utils import fuse_conv_bn_weights

from .utils import _quantize_weight, WeightedQuantizedModule


__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
]

_SUPPORTED_PADDING = {"zeros", "reflect"}


def _reverse_repeat_padding(padding: list[int]) -> list[int]:
    _reversed_padding_repeated_twice: list[int] = []
    N = len(padding)
    for idx in range(N):
        _reversed_padding_repeated_twice.extend(padding[N - idx - 1] for _ in range(2))
    return _reversed_padding_repeated_twice


class _ConvNd(WeightedQuantizedModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        # All subclasses have this signature - See PR #49702s
        raise NotImplementedError

    def _init(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        bias,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if out_channels <= 0:
            raise ValueError(f"out_channels must be greater than 0, got {out_channels}")
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if padding_mode not in _SUPPORTED_PADDING:
            raise ValueError(
                f"'padding_mode' {padding_mode} is not supported by quantized convolution"
            )
        self.padding_mode = padding_mode
        # Initialize as NCHW. set_weight will internally transpose to NHWC.
        if self.transposed:
            weight_shape = [in_channels, out_channels // self.groups]
        else:
            weight_shape = [out_channels, in_channels // self.groups]
        qweight = torch._empty_affine_quantized(
            weight_shape + list(kernel_size),
            scale=1,
            zero_point=0,
            dtype=torch.qint8,
            **{k: v for k, v in factory_kwargs.items() if k != "dtype"},
        )
        bias_float = (
            torch.zeros(
                out_channels,
                dtype=torch.float,
                **{k: v for k, v in factory_kwargs.items() if k != "dtype"},
            )
            if bias
            else None
        )

        self.set_weight_bias(qweight, bias_float)
        self.scale = 1.0
        self.zero_point = 0

    def set_weight_bias(self, qweight, bias_float):
        raise NotImplementedError

    def bias(self):
        raise NotImplementedError

    def _weight_bias(self):
        raise NotImplementedError

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}, scale={scale}, zero_point={zero_point}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias() is None:
            s += ", bias=False"
        return s.format(**self.__dict__)

    # ===== Serialization methods =====
    # The special consideration here is that we have to unpack the weights into
    # their regular QTensor form for serialization. Packed weights should not
    # live outside the process in which they were created, rather they should be
    # derived from the QTensor weight.
    #   self
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #
    # TODO: maybe change to this when https://github.com/pytorch/pytorch/pull/32958 is landed
    #   self
    #   |--- _packed_params : Conv2dPackedParamsBase or Conv3dPackedParamsBase
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        (w, b) = self._weight_bias()
        destination[prefix + "weight"] = w
        destination[prefix + "bias"] = b
        destination[prefix + "scale"] = torch.tensor(self.scale)
        destination[prefix + "zero_point"] = torch.tensor(self.zero_point)

    @torch.jit.export
    def __getstate__(self):
        (w, b) = self._weight_bias()
        return (
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.transposed,
            self.output_padding,
            self.groups,
            self.padding_mode,
            w,
            b,
            self.scale,
            self.zero_point,
            self.training,
        )

    # ===== Deserialization methods =====
    # Counterpart to the serialization methods, we must pack the serialized
    # QTensor weight into its packed format for use by the FBGEMM ops.
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
        self.set_weight_bias(state_dict[prefix + "weight"], state_dict[prefix + "bias"])
        state_dict.pop(prefix + "weight")
        state_dict.pop(prefix + "bias")
        self.scale = float(state_dict[prefix + "scale"])
        state_dict.pop(prefix + "scale")
        self.zero_point = int(state_dict[prefix + "zero_point"])
        state_dict.pop(prefix + "zero_point")
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            False,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @torch.jit.export
    def __setstate__(self, state):
        self.in_channels = state[0]
        self.out_channels = state[1]
        self.kernel_size = state[2]
        self.stride = state[3]
        self.padding = state[4]
        self.dilation = state[5]
        self.transposed = state[6]
        self.output_padding = state[7]
        self.groups = state[8]
        self.padding_mode = state[9]
        self.set_weight_bias(state[10], state[11])
        self.scale = state[12]
        self.zero_point = state[13]
        self.training = state[14]

    def __deepcopy__(self, memo):
        new_instance = type(self).__new__(type(self))
        torch.nn.Module.__init__(new_instance)
        state = self.__getstate__()
        new_instance.__setstate__(state)
        return new_instance

    def __copy__(self):
        return self.__deepcopy__({})

    @classmethod
    def get_qconv(cls, mod, activation_post_process, weight_post_process=None):
        r"""Creates a qconv object and returns it."""
        if weight_post_process is None:
            weight_post_process = mod.qconfig.weight()
        weight_post_process(mod.weight)
        if weight_post_process.dtype != torch.qint8:
            raise AssertionError(
                f"Weight observer must have a dtype of qint8, got {weight_post_process.dtype}"
            )
        qweight = _quantize_weight(mod.weight.float(), weight_post_process)
        # the __init__ call used is the one from derived classes and not the one from _ConvNd
        qconv = cls(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,
            mod.stride,
            mod.padding,
            mod.dilation,
            mod.groups,
            mod.bias is not None,
            mod.padding_mode,
        )
        qconv.set_weight_bias(qweight, mod.bias)
        if (
            activation_post_process is None
            or activation_post_process.dtype == torch.float
        ):
            return qconv  # dynamic quantization doesn't need scale/zero_point
        else:
            act_scale, act_zp = activation_post_process.calculate_qparams()
            qconv.scale = float(act_scale)
            qconv.zero_point = int(act_zp)
            return qconv

    @staticmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        if hasattr(mod, "weight_fake_quant"):
            # assert type(mod) is cls.__QAT_MODULE, " nnq." + cls.__name__ + \
            # ".from_float only works for " + cls.__QAT_MODULE.__name__
            if type(mod) is cls._NNIQAT_CONV_BN_MODULE:
                mod.weight, mod.bias = fuse_conv_bn_weights(
                    mod.weight,
                    mod.bias,
                    mod.bn.running_mean,
                    mod.bn.running_var,
                    mod.bn.eps,
                    mod.bn.weight,
                    mod.bn.bias,
                )
            if not hasattr(mod, "activation_post_process"):
                raise AssertionError("Input QAT module must have observer attached")
            weight_post_process = mod.weight_fake_quant
            activation_post_process = mod.activation_post_process
        else:
            if type(mod) is not cls._FLOAT_MODULE:
                raise AssertionError(
                    f"nnq.{cls.__name__}.from_float only works for "
                    f"{cls._FLOAT_MODULE.__name__} but got: {type(mod).__name__}"
                )
            if not hasattr(mod, "qconfig"):
                raise AssertionError("Input float module must have qconfig defined.")
            activation_post_process = (
                None
                if not hasattr(mod, "activation_post_process")
                else mod.activation_post_process
            )
            if type(mod) in [
                cls._NNI_CONV_RELU_MODULE,
                cls._NNI_CONV_ADD_MODULE,
                cls._NNI_CONV_ADD_RELU_MODULE,
            ]:
                mod = mod[0]
            weight_post_process = mod.qconfig.weight()
        return cls.get_qconv(mod, activation_post_process, weight_post_process)

    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point):
        r"""Create a (fbgemm/qnnpack) quantized module from a reference quantized module
        Args:
            ref_qconv (Module): a reference quantized  module, either produced by torch.ao.quantization
                                utilities or provided by the user
            output_scale (float): scale for output Tensor
            output_zero_point (int): zero point for output Tensor
        """
        qconv = cls(
            ref_qconv.in_channels,
            ref_qconv.out_channels,
            ref_qconv.kernel_size,  # type: ignore[arg-type]
            ref_qconv.stride,  # type: ignore[arg-type]
            ref_qconv.padding,  # type: ignore[arg-type]
            ref_qconv.dilation,  # type: ignore[arg-type]
            ref_qconv.groups,
            ref_qconv.bias is not None,  # type: ignore[arg-type]
            ref_qconv.padding_mode,
            device=ref_qconv.weight.device,
            dtype=ref_qconv.weight.dtype,
        )
        qweight = ref_qconv.get_quantized_weight()
        qconv.set_weight_bias(qweight, ref_qconv.bias)
        qconv.scale = float(output_scale)
        qconv.zero_point = int(output_zero_point)
        return qconv


class Conv1d(_ConvNd):
    r"""Applies a 1D convolution over a quantized input signal composed of
    several quantized input planes.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv1d`.

    .. note::
        Only `zeros` is supported for the :attr:`padding_mode` argument.

    .. note::
        Only `torch.quint8` is supported for the input data type.


    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv1d` for other attributes.

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> m = nn.quantized.Conv1d(16, 33, 3, stride=2)
        >>> input = torch.randn(20, 16, 100)
        >>> # quantize input to quint8
        >>> # xdoctest: +SKIP
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0,
        ...                                     dtype=torch.quint8)
        >>> output = m(q_input)

    """

    _FLOAT_MODULE: ClassVar[type[nn.Conv1d]] = nn.Conv1d
    _NNIQAT_CONV_BN_MODULE: ClassVar[type[nn.Module] | None] = nniqat.ConvBn1d
    _NNI_CONV_RELU_MODULE: ClassVar[type[nn.Module] | None] = nni.ConvReLU1d
    _NNI_CONV_ADD_MODULE: ClassVar[type[nn.Module] | None] = None
    _NNI_CONV_ADD_RELU_MODULE: ClassVar[type[nn.Module] | None] = None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _single(kernel_size)
        stride = _single(stride)

        padding = padding if isinstance(padding, str) else _single(padding)
        dilation = _single(dilation)

        # Subclasses of _ConvNd needs to call _init rather than __init__. See
        # discussion on PR #49702
        super()._init(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _single(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _get_name(self):
        return "QuantizedConv1d"

    def set_weight_bias(self, w: torch.Tensor, b: torch.Tensor | None) -> None:
        if self.padding_mode == "zeros":
            self._packed_params = torch.ops.quantized.conv1d_prepack(
                w, b, self.stride, self.padding, self.dilation, self.groups
            )
        else:
            self._packed_params = torch.ops.quantized.conv1d_prepack(
                w, b, self.stride, _pair(0), self.dilation, self.groups
            )

    def _weight_bias(self):
        w, b = torch.ops.quantized.conv1d_unpack(self._packed_params)
        return w, b

    def weight(self):
        return self._weight_bias()[0]

    def bias(self):
        return self._weight_bias()[1]

    def forward(self, input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 3:
            raise ValueError("Input shape must be `(N, C, L)`!")
        if self.padding_mode != "zeros":
            # Padding in Conv1d is stored as (p, p), need to get (p,)
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding[:1])
            input = F.pad(
                input, _reversed_padding_repeated_twice, mode=self.padding_mode
            )
        return ops.quantized.conv1d(
            input, self._packed_params, self.scale, self.zero_point
        )

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        r"""Creates a quantized module from a float module or qparams_dict.

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
              utilities or provided by the user
        """
        return _ConvNd.from_float(
            cls, mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )


class Conv2d(_ConvNd):
    r"""Applies a 2D convolution over a quantized input signal composed of
    several quantized input planes.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv2d`.

    .. note::
        Only `zeros` is supported for the :attr:`padding_mode` argument.

    .. note::
        Only `torch.quint8` is supported for the input data type.


    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv2d` for other attributes.

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> # With square kernels and equal stride
        >>> m = nn.quantized.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.quantized.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.quantized.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> # quantize input to quint8
        >>> # xdoctest: +SKIP
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)

    """

    _FLOAT_MODULE: ClassVar[type[nn.Conv2d]] = nn.Conv2d
    _NNIQAT_CONV_BN_MODULE: ClassVar[type[nn.Module] | None] = nniqat.ConvBn2d
    _NNI_CONV_RELU_MODULE: ClassVar[type[nn.Module] | None] = nni.ConvReLU2d
    _NNI_CONV_ADD_MODULE: ClassVar[type[nni.ConvAdd2d]] = nni.ConvAdd2d
    _NNI_CONV_ADD_RELU_MODULE: ClassVar[type[nni.ConvAddReLU2d]] = nni.ConvAddReLU2d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        # Subclasses of _ConvNd need to call _init rather than __init__. See
        # discussion on PR #49702
        super()._init(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _get_name(self):
        return "QuantizedConv2d"

    def set_weight_bias(self, w: torch.Tensor, b: torch.Tensor | None) -> None:
        if self.padding_mode == "zeros":
            self._packed_params = torch.ops.quantized.conv2d_prepack(
                w, b, self.stride, self.padding, self.dilation, self.groups
            )
        else:
            self._packed_params = torch.ops.quantized.conv2d_prepack(
                w, b, self.stride, _pair(0), self.dilation, self.groups
            )

    def _weight_bias(self):
        return self._packed_params.unpack()

    def weight(self):
        return self._weight_bias()[0]

    def bias(self):
        return self._weight_bias()[1]

    def forward(self, input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        if self.padding_mode != "zeros":
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
            input = F.pad(
                input, _reversed_padding_repeated_twice, mode=self.padding_mode
            )
        return ops.quantized.conv2d(
            input, self._packed_params, self.scale, self.zero_point
        )

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        r"""Creates a quantized module from a float module or qparams_dict.

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
              utilities or provided by the user
        """
        return _ConvNd.from_float(
            cls, mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )


class Conv3d(_ConvNd):
    r"""Applies a 3D convolution over a quantized input signal composed of
    several quantized input planes.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv3d`.

    .. note::
        Only `zeros` is supported for the :attr:`padding_mode` argument.

    .. note::
        Only `torch.quint8` is supported for the input data type.


    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv3d` for other attributes.

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> # With square kernels and equal stride
        >>> m = nn.quantized.Conv3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.quantized.Conv3d(16, 33, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.quantized.Conv3d(16, 33, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), dilation=(1, 2, 2))
        >>> input = torch.randn(20, 16, 56, 56, 56)
        >>> # quantize input to quint8
        >>> # xdoctest: +SKIP
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)

    """

    _FLOAT_MODULE: ClassVar[type[nn.Conv3d]] = nn.Conv3d
    _NNIQAT_CONV_BN_MODULE: ClassVar[type[nn.Module] | None] = nniqat.ConvBn3d
    _NNI_CONV_RELU_MODULE: ClassVar[type[nn.Module] | None] = nni.ConvReLU3d
    _NNI_CONV_ADD_MODULE: ClassVar[type[nn.Module] | None] = None
    _NNI_CONV_ADD_RELU_MODULE: ClassVar[type[nn.Module] | None] = None

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        if padding_mode == "reflect":
            raise AssertionError("Conv3d does not support reflection padding")
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        # Subclasses of _ConvNd need to call _init rather than __init__. See
        # discussion on PR #49702
        super()._init(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _triple(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _get_name(self):
        return "QuantizedConv3d"

    def set_weight_bias(self, w: torch.Tensor, b: torch.Tensor | None) -> None:
        if self.padding_mode == "zeros":
            self._packed_params = torch.ops.quantized.conv3d_prepack(
                w, b, self.stride, self.padding, self.dilation, self.groups
            )
        else:
            self._packed_params = torch.ops.quantized.conv3d_prepack(
                w, b, self.stride, _triple(0), self.dilation, self.groups
            )

    def _weight_bias(self):
        return self._packed_params.unpack()

    def weight(self):
        return self._weight_bias()[0]

    def bias(self):
        return self._weight_bias()[1]

    def forward(self, input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 5:
            raise ValueError("Input shape must be `(N, C, D, H, W)`!")
        if self.padding_mode != "zeros":
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
            input = F.pad(
                input, _reversed_padding_repeated_twice, mode=self.padding_mode
            )
        return ops.quantized.conv3d(
            input, self._packed_params, self.scale, self.zero_point
        )

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        r"""Creates a quantized module from a float module or qparams_dict.

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
              utilities or provided by the user
        """
        return _ConvNd.from_float(
            cls, mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )


# === Transposed Convolutions ===


class _ConvTransposeNd(_ConvNd):
    _FLOAT_MODULE: ClassVar[type[nn.modules.conv._ConvNd]]

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        bias,
        padding_mode,
        device=None,
        dtype=None,
    ):
        if padding_mode != "zeros":
            raise ValueError(
                f'Only "zeros" padding mode is supported for {self.__class__.__name__}'
            )
        factory_kwargs = {"device": device, "dtype": dtype}
        # Subclasses of _ConvNd need to call _init rather than __init__. See
        # discussion on PR #49702
        super()._init(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _input_padding(
        self, kernel_size: list[int], dilation: list[int], padding: list[int]
    ) -> list[int]:
        res = torch.jit.annotate(list[int], [])
        for kdx in range(len(kernel_size)):
            pad = dilation[kdx] * (kernel_size[kdx] - 1) - padding[kdx]
            res.append(pad)
        return res

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        r"""Creates a quantized module from a float module or qparams_dict.
        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
              utilities or provided by the user
        """
        # derived classes override cls._FLOAT_MODULE attribute
        msg = (
            " nnq."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__  # type: ignore[attr-defined]
        )
        if type(mod) is not cls._FLOAT_MODULE:
            raise AssertionError(msg)
        if not hasattr(mod, "qconfig"):
            raise AssertionError("Input float module must have qconfig defined.")
        weight_post_process = mod.qconfig.weight()  # type: ignore[operator, union-attr]
        weight_post_process(mod.weight)
        if weight_post_process.dtype != torch.qint8:
            raise AssertionError(
                f"Weight observer must have a dtype of qint8, got {weight_post_process.dtype}"
            )
        qweight = _quantize_weight(mod.weight.float(), weight_post_process)
        # the __init__ call used is the one from derived classes and not the one from _ConvTransposeNd
        # pyrefly: ignore [missing-argument]
        qconv = cls(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,  # type: ignore[call-arg]
            mod.stride,
            mod.padding,
            mod.output_padding,
            mod.groups,
            mod.bias is not None,
            mod.dilation,
            mod.padding_mode,
        )
        qconv.set_weight_bias(qweight, mod.bias)
        if (
            not hasattr(mod, "activation_post_process")
            or mod.activation_post_process.dtype == torch.float
        ):
            return qconv  # dynamic quantization doesn't need scale/zero_point
        else:
            act_scale, act_zp = mod.activation_post_process.calculate_qparams()  # type: ignore[operator, union-attr]
            qconv.scale = float(act_scale)
            qconv.zero_point = int(act_zp)
            return qconv

    @staticmethod
    def from_reference(cls, ref_qconvt, output_scale, output_zero_point):  # type: ignore[override]
        r"""Create a (fbgemm/qnnpack) quantized module from a reference quantized module
        Args:
            ref_qconvt (Module): a reference quantized  module, either produced by torch.ao.quantization
                                 utilities or provided by the user
            output_scale (float): scale for output Tensor
            output_zero_point (int): zero point for output Tensor
        """
        qconv = cls(
            ref_qconvt.in_channels,
            ref_qconvt.out_channels,
            ref_qconvt.kernel_size,  # type: ignore[arg-type]
            ref_qconvt.stride,  # type: ignore[arg-type]
            ref_qconvt.padding,  # type: ignore[arg-type]
            ref_qconvt.output_padding,  # type: ignore[arg-type]
            ref_qconvt.groups,
            ref_qconvt.bias is not None,  # type: ignore[arg-type]
            ref_qconvt.dilation,  # type: ignore[arg-type]
            ref_qconvt.padding_mode,
            device=ref_qconvt.weight.device,
            dtype=ref_qconvt.weight.dtype,
        )
        qweight = ref_qconvt.get_quantized_weight()
        qconv.set_weight_bias(qweight, ref_qconvt.bias)
        qconv.scale = float(output_scale)
        qconv.zero_point = int(output_zero_point)
        return qconv


class ConvTranspose1d(_ConvTransposeNd):
    r"""Applies a 1D transposed convolution operator over an input image
    composed of several input planes.
    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.ConvTranspose1d`.

    .. note:: Currently only the QNNPACK engine is implemented.
        Please, set the `torch.backends.quantized.engine = 'qnnpack'`

    For special notes, please, see :class:`~torch.ao.nn.quantized.Conv1d`

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point
    See :class:`~torch.nn.ConvTranspose2d` for other attributes.

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> torch.backends.quantized.engine = 'qnnpack'
        >>> from torch.ao.nn import quantized as nnq
        >>> # With square kernels and equal stride
        >>> m = nnq.ConvTranspose1d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nnq.ConvTranspose1d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> input = torch.randn(20, 16, 50)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)
        >>> # exact output size can be also specified as an argument
        >>> input = torch.randn(1, 16, 12)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> downsample = nnq.Conv1d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nnq.ConvTranspose1d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(q_input)
        >>> h.size()
        torch.Size([1, 16, 6])
        >>> # xdoctest: +SKIP("FIXME: output_size is not a parameter)
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12])
    """

    _FLOAT_MODULE: ClassVar[type[nn.ConvTranspose1d]] = nn.ConvTranspose1d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        output_padding = _single(output_padding)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            True,
            output_padding,
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _get_name(self):
        return "QuantizedConvTranspose1d"

    def set_weight_bias(self, w: torch.Tensor, b: torch.Tensor | None) -> None:
        self._packed_params = torch.ops.quantized.conv_transpose1d_prepack(
            w,
            b,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups,
        )

    def _weight_bias(self):
        w, b = torch.ops.quantized.conv_transpose1d_unpack(self._packed_params)
        return w, b

    def weight(self):
        (w, _) = self._weight_bias()
        return w

    def bias(self):
        (_, b) = self._weight_bias()
        return b

    def forward(self, input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 3:
            raise ValueError("Input shape must be `(N, C, L)`!")
        return torch.ops.quantized.conv_transpose1d(
            input, self._packed_params, self.scale, self.zero_point
        )

    @classmethod
    def from_reference(cls, ref_qconvt, output_scale, output_zero_point):  # type: ignore[override]
        return _ConvTransposeNd.from_reference(
            cls, ref_qconvt, output_scale, output_zero_point
        )


class ConvTranspose2d(_ConvTransposeNd):
    r"""Applies a 2D transposed convolution operator over an input image
    composed of several input planes.
    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.ConvTranspose2d`.

    For special notes, please, see :class:`~torch.ao.nn.quantized.Conv2d`

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point
    See :class:`~torch.nn.ConvTranspose2d` for other attributes.

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> # QNNPACK or FBGEMM as backend
        >>> torch.backends.quantized.engine = 'qnnpack'
        >>> # With square kernels and equal stride
        >>> import torch.ao.nn.quantized as nnq
        >>> m = nnq.ConvTranspose2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nnq.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)
        >>> # exact output size can be also specified as an argument
        >>> input = torch.randn(1, 16, 12, 12)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> downsample = nnq.Conv2d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nnq.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(q_input)
        >>> h.size()
        torch.Size([1, 16, 6, 6])
        >>> # xdoctest: +SKIP("FIXME: output_size is not a parameter)
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12])
    """

    _FLOAT_MODULE: ClassVar[type[nn.ConvTranspose2d]] = nn.ConvTranspose2d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            True,
            output_padding,
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _get_name(self):
        return "QuantizedConvTranspose2d"

    def set_weight_bias(self, w: torch.Tensor, b: torch.Tensor | None) -> None:
        self._packed_params = torch.ops.quantized.conv_transpose2d_prepack(
            w,
            b,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups,
        )

    def _weight_bias(self):
        w, b = torch.ops.quantized.conv2d_unpack(self._packed_params)
        return w, b

    def weight(self):
        (w, _) = self._weight_bias()
        return w

    def bias(self):
        (_, b) = self._weight_bias()
        return b

    def forward(self, input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        return ops.quantized.conv_transpose2d(
            input, self._packed_params, self.scale, self.zero_point
        )

    @classmethod
    def from_reference(cls, ref_qconvt, output_scale, output_zero_point):  # type: ignore[override]
        return _ConvTransposeNd.from_reference(
            cls, ref_qconvt, output_scale, output_zero_point
        )


class ConvTranspose3d(_ConvTransposeNd):
    r"""Applies a 3D transposed convolution operator over an input image
    composed of several input planes.
    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.ConvTranspose3d`.

    .. note:: Currently only the FBGEMM engine is implemented.
        Please, set the `torch.backends.quantized.engine = 'fbgemm'`

    For special notes, please, see :class:`~torch.ao.nn.quantized.Conv3d`

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point
    See :class:`~torch.nn.ConvTranspose3d` for other attributes.

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> torch.backends.quantized.engine = 'fbgemm'
        >>> from torch.ao.nn import quantized as nnq
        >>> # With cubic kernels and equal stride
        >>> m = nnq.ConvTranspose3d(16, 33, 3, stride=2)
        >>> # non-cubic kernels and unequal stride and with padding
        >>> m = nnq.ConvTranspose3d(16, 33, (3, 3, 5), stride=(2, 1, 1), padding=(4, 2, 2))
        >>> input = torch.randn(20, 16, 50, 100, 100)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)
        >>> # exact output size can be also specified as an argument
        >>> input = torch.randn(1, 16, 12, 12, 12)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> downsample = nnq.Conv3d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nnq.ConvTranspose3d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(q_input)
        >>> h.size()
        torch.Size([1, 16, 6, 6, 6])
        >>> # xdoctest: +SKIP("FIXME: output_size is not a parameter)
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12, 12])
    """

    _FLOAT_MODULE: ClassVar[type[nn.ConvTranspose3d]] = nn.ConvTranspose3d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        output_padding = _triple(output_padding)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            True,
            output_padding,
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _get_name(self):
        return "QuantizedConvTranspose3d"

    def set_weight_bias(self, w: torch.Tensor, b: torch.Tensor | None) -> None:
        self._packed_params = torch.ops.quantized.conv_transpose3d_prepack(
            w,
            b,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups,
        )

    def _weight_bias(self):
        w, b = torch.ops.quantized.conv3d_unpack(self._packed_params)
        return w, b

    def weight(self):
        (w, _) = self._weight_bias()
        return w

    def bias(self):
        (_, b) = self._weight_bias()
        return b

    def forward(self, input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 5:
            raise ValueError("Input shape must be `(N, C, T, H, W)`!")
        return ops.quantized.conv_transpose3d(
            input, self._packed_params, self.scale, self.zero_point
        )

    @classmethod
    def from_reference(cls, ref_qconvt, output_scale, output_zero_point):  # type: ignore[override]
        return _ConvTransposeNd.from_reference(
            cls, ref_qconvt, output_scale, output_zero_point
        )
<|endoftext|># mypy: allow-untyped-defs
from warnings import warn

import torch


__all__ = [
    "ReLU6",
    "Hardswish",
    "ELU",
    "LeakyReLU",
    "Sigmoid",
    "Softmax",
    "MultiheadAttention",
    "PReLU",
]


class ReLU6(torch.nn.ReLU):
    r"""Applies the element-wise function:

    :math:`\text{ReLU6}(x) = \min(\max(x_0, x), q(6))`, where :math:`x_0` is the
    zero_point, and :math:`q(6)` is the quantized representation of number 6.

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: ../scripts/activation_images/ReLU6.png

    Examples::

        >>> m = nn.quantized.ReLU6()
        >>> input = torch.randn(2)
        >>> # xdoctest: +SKIP
        >>> input = torch.quantize_per_tensor(input, 1.0, 0, dtype=torch.qint32)
        >>> output = m(input)
    """

    def __init__(self, inplace=False):
        super().__init__(inplace)
        self.inplace = inplace

    def forward(self, input):
        return torch.ops.quantized.relu6(input, self.inplace)

    def _get_name(self):
        return "QuantizedReLU6"

    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=False):
        return ReLU6(mod.inplace)


class Hardswish(torch.nn.Hardswish):
    r"""This is the quantized version of :class:`~torch.nn.Hardswish`.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
    """

    def __init__(self, scale, zero_point, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale, **factory_kwargs))
        self.register_buffer("zero_point", torch.tensor(zero_point, **factory_kwargs))

    def forward(self, input):
        return torch.ops.quantized.hardswish(input, self.scale, self.zero_point)

    def _get_name(self):
        return "QuantizedHardswish"

    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=False):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        return Hardswish(float(scale), int(zero_point))

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        return cls(float(scale), int(zero_point))


class ELU(torch.nn.ELU):
    r"""This is the quantized equivalent of :class:`~torch.nn.ELU`.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
        alpha: the alpha constant
    """

    def __init__(self, scale, zero_point, alpha=1.0):
        super().__init__(alpha)
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, input):
        return torch.ao.nn.quantized.functional.elu(
            input, self.scale, self.zero_point, self.alpha
        )

    def _get_name(self):
        return "QuantizedELU"

    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=False):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        return ELU(float(scale), int(zero_point), mod.alpha)

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        return cls(float(scale), int(zero_point), mod.alpha)


class LeakyReLU(torch.nn.LeakyReLU):
    r"""This is the quantized equivalent of :class:`~torch.nn.LeakyReLU`.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
        negative_slope: Controls the angle of the negative slope. Default: 1e-2
    """

    def __init__(
        self,
        scale: float,
        zero_point: int,
        negative_slope: float = 1e-2,
        inplace: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(negative_slope, inplace)
        self.register_buffer("scale", torch.tensor(scale, **factory_kwargs))
        self.register_buffer("zero_point", torch.tensor(zero_point, **factory_kwargs))

    def forward(self, input):
        return torch.ops.quantized.leaky_relu(
            input, self.negative_slope, self.inplace, self.scale, self.zero_point
        )

    def _get_name(self):
        return "QuantizedLeakyReLU"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        return cls(float(scale), int(zero_point), mod.negative_slope, mod.inplace)

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        return cls(float(scale), int(zero_point), mod.negative_slope, mod.inplace)


class Sigmoid(torch.nn.Sigmoid):
    r"""This is the quantized equivalent of :class:`~torch.nn.Sigmoid`.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
    """

    def __init__(self, output_scale: float, output_zero_point: int):
        super().__init__()
        self.output_scale = output_scale
        self.output_zero_point = output_zero_point

    def forward(self, input):
        return torch.ops.quantized.sigmoid(
            input, self.output_scale, self.output_zero_point
        )

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        (
            output_scale,
            output_zero_point,
        ) = mod.activation_post_process.calculate_qparams()
        return cls(float(output_scale), int(output_zero_point))


class Softmax(torch.nn.Softmax):
    r"""This is the quantized version of :class:`~torch.nn.Softmax`.

    Args:
        dim: A dimension along which Softmax will be computed (so every slice along dim will sum to 1).
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
    """

    def __init__(self, dim=None, scale=1.0, zero_point=0):
        super().__init__()
        self.dim = dim
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, input):
        dim = self.dim
        if dim is None:
            stacklevel = 3
            # Note: adding the mypy ignore on _get_softmax_dim seems less bad
            # than making `_get_softmax_dim` an official API.
            dim = torch.nn.functional._get_softmax_dim(  # type: ignore[attr-defined]
                "softmax", input.dim(), stacklevel
            )
        return torch.ops.quantized.softmax(input, dim, self.scale, self.zero_point)

    def _get_name(self):
        return "QuantizedSoftmax"

    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=False):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        return Softmax(mod.dim, float(scale), int(zero_point))

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        return cls(mod.dim, float(scale), int(zero_point))


class MultiheadAttention(torch.ao.nn.quantizable.MultiheadAttention):
    _FLOAT_MODULE = torch.ao.nn.quantizable.MultiheadAttention

    def _get_name(self):
        return "QuantizedMultiheadAttention"

    @classmethod
    def from_float(cls, other):
        # The whole flow is float -> observed -> quantized
        # This class does observed -> quantized only
        raise NotImplementedError(
            "It looks like you are trying to convert a "
            "non-observed MHA module. Please, see "
            "the examples on quantizable MHAs."
        )

    @classmethod
    def from_observed(cls, other):
        converted = torch.ao.quantization.convert(
            other,
            mapping=None,
            inplace=False,
            remove_qconfig=True,
            convert_custom_config_dict=None,
        )
        converted.__class__ = cls
        # Remove the parameters for the bias_k and bias_v to quantize them
        # TODO: This is a potential source of accuracy drop.
        #       quantized cat takes the scale and zp of the first
        #       element, which might lose the precision in the bias_k
        #       and the bias_v (which are cat'ed with k/v being first).
        if converted.bias_k is not None:
            bias_k = converted._parameters.pop("bias_k")
            sc, zp = torch._choose_qparams_per_tensor(bias_k, reduce_range=False)
            bias_k = torch.quantize_per_tensor(bias_k, sc, zp, torch.quint8)
            setattr(converted, "bias_k", bias_k)  # noqa: B010

        if converted.bias_v is not None:
            bias_v = converted._parameters.pop("bias_v")
            sc, zp = torch._choose_qparams_per_tensor(
                bias_k,  # type: ignore[possibly-undefined]
                reduce_range=False,
            )
            bias_v = torch.quantize_per_tensor(bias_v, sc, zp, torch.quint8)
            setattr(converted, "bias_v", bias_v)  # noqa: B010

        del converted.in_proj_weight
        del converted.in_proj_bias

        return converted


class PReLU(torch.nn.Module):
    r"""This is the quantized equivalent of :class:`~torch.nn.PReLU`.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
        num_parameters: number of parameters: 1, or the number of channels at input. Default: 1
    """

    def __init__(
        self, output_scale: float, output_zero_point: int, num_parameters: int = 1
    ) -> None:
        super().__init__()
        self.num_parameters = num_parameters
        self.scale = output_scale
        self.zero_point = output_zero_point
        w = torch.randn(num_parameters, dtype=torch.float)
        qw = torch.quantize_per_tensor(w, scale=1.0, zero_point=0, dtype=torch.quint8)
        self.set_weight(qw)

    def set_weight(self, w: torch.Tensor) -> None:
        self.weight = w

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.ops.quantized.prelu(
            input, self.weight, self.scale, self.zero_point
        )

    def _get_name(self):
        return "QuantizedPReLU"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        qprelu = cls(float(scale), int(zero_point), mod.num_parameters)
        float_wt = mod.weight.float()
        observer = mod.qconfig.weight()
        observer(float_wt)
        if observer.dtype != torch.quint8:
            warn(
                f"PReLU's weight observer should have dtype quint8 but got {observer.dtype}",
                stacklevel=2,
            )
        wt_scale, wt_zp = observer.calculate_qparams()
        qweight = torch.quantize_per_tensor(
            float_wt, float(wt_scale), int(wt_zp), torch.quint8
        )
        qprelu.set_weight(qweight)
        return qprelu

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        qprelu = cls(float(scale), int(zero_point), mod.num_parameters)
        float_wt = mod.weight.float()
        observer = mod.qconfig.weight()
        observer(float_wt)
        if observer.dtype != torch.quint8:
            warn(
                f"PReLU's weight observer should have dtype quint8 but got {observer.dtype}",
                stacklevel=2,
            )
        wt_scale, wt_zp = observer.calculate_qparams()
        qweight = torch.quantize_per_tensor(
            float_wt, float(wt_scale), int(wt_zp), torch.quint8
        )
        qprelu.set_weight(qweight)
        return qprelu
<|endoftext|># mypy: allow-untyped-defs
import torch

# The quantized modules use `torch.nn` and `torch.ao.nn.quantizable`
# packages. However, the `quantizable` package uses "lazy imports"
# to avoid circular dependency.
# Hence we need to include it here to make sure it is resolved before
# they are used in the modules.
import torch.ao.nn.quantizable
from torch.nn.modules.pooling import MaxPool2d

from .activation import (
    ELU,
    Hardswish,
    LeakyReLU,
    MultiheadAttention,
    PReLU,
    ReLU6,
    Sigmoid,
    Softmax,
)
from .batchnorm import BatchNorm2d, BatchNorm3d
from .conv import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)
from .dropout import Dropout
from .embedding_ops import Embedding, EmbeddingBag
from .functional_modules import FloatFunctional, FXFloatFunctional, QFunctional
from .linear import Linear
from .normalization import (
    GroupNorm,
    InstanceNorm1d,
    InstanceNorm2d,
    InstanceNorm3d,
    LayerNorm,
)
from .rnn import LSTM


__all__ = [
    "BatchNorm2d",
    "BatchNorm3d",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "DeQuantize",
    "ELU",
    "Embedding",
    "EmbeddingBag",
    "GroupNorm",
    "Hardswish",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "LayerNorm",
    "LeakyReLU",
    "Linear",
    "LSTM",
    "MultiheadAttention",
    "Quantize",
    "ReLU6",
    "Sigmoid",
    "Softmax",
    "Dropout",
    "PReLU",
    # Wrapper modules
    "FloatFunctional",
    "FXFloatFunctional",
    "QFunctional",
]


class Quantize(torch.nn.Module):
    r"""Quantizes an incoming tensor

    Args:
     `scale`: scale of the output Quantized Tensor
     `zero_point`: zero_point of output Quantized Tensor
     `dtype`: data type of output Quantized Tensor
     `factory_kwargs`: Dictionary of kwargs used for configuring initialization
         of internal buffers. Currently, `device` and `dtype` are supported.
         Example: `factory_kwargs={'device': 'cuda', 'dtype': torch.float64}`
         will initialize internal buffers as type `torch.float64` on the current CUDA device.
         Note that `dtype` only applies to floating-point buffers.

    Examples::
        >>> t = torch.tensor([[1., -1.], [1., -1.]])
        >>> scale, zero_point, dtype = 1.0, 2, torch.qint8
        >>> qm = Quantize(scale, zero_point, dtype)
        >>> # xdoctest: +SKIP
        >>> qt = qm(t)
        >>> print(qt)
        tensor([[ 1., -1.],
                [ 1., -1.]], size=(2, 2), dtype=torch.qint8, scale=1.0, zero_point=2)
    """

    scale: torch.Tensor
    zero_point: torch.Tensor

    def __init__(self, scale, zero_point, dtype, factory_kwargs=None):
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        super().__init__()
        self.register_buffer("scale", torch.tensor([scale], **factory_kwargs))
        self.register_buffer(
            "zero_point",
            torch.tensor(
                [zero_point],
                dtype=torch.long,
                **{k: v for k, v in factory_kwargs.items() if k != "dtype"},
            ),
        )
        self.dtype = dtype

    def forward(self, X):
        return torch.quantize_per_tensor(
            X, float(self.scale), int(self.zero_point), self.dtype
        )

    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=False):
        if not hasattr(mod, "activation_post_process"):
            raise AssertionError(
                f"Module {type(mod).__name__} must have activation_post_process attribute"
            )
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        return Quantize(
            scale.float().item(),
            zero_point.long().item(),
            mod.activation_post_process.dtype,
        )

    def extra_repr(self):
        return f"scale={self.scale}, zero_point={self.zero_point}, dtype={self.dtype}"


class DeQuantize(torch.nn.Module):
    r"""Dequantizes an incoming tensor

    Examples::
        >>> input = torch.tensor([[1., -1.], [1., -1.]])
        >>> scale, zero_point, dtype = 1.0, 2, torch.qint8
        >>> qm = Quantize(scale, zero_point, dtype)
        >>> # xdoctest: +SKIP
        >>> quantized_input = qm(input)
        >>> dqm = DeQuantize()
        >>> dequantized = dqm(quantized_input)
        >>> print(dequantized)
        tensor([[ 1., -1.],
                [ 1., -1.]], dtype=torch.float32)
    """

    def forward(self, Xq):
        return Xq.dequantize()

    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=False):
        return DeQuantize()
<|endoftext|># mypy: allow-untyped-defs
import torch
import torch.ao.nn.intrinsic as nni


__all__ = ["BatchNorm2d", "BatchNorm3d"]


class _BatchNorm(torch.nn.modules.batchnorm._BatchNorm):
    def __init__(
        self, num_features, eps=1e-5, momentum=0.1, device=None, dtype=None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(num_features, eps, momentum, True, True, **factory_kwargs)
        self.register_buffer("scale", torch.tensor(1.0, **factory_kwargs))
        self.register_buffer("zero_point", torch.tensor(0, **factory_kwargs))

    @staticmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        activation_post_process = mod.activation_post_process
        if type(mod) is cls._NNI_BN_RELU_MODULE:
            mod = mod[0]
        scale, zero_point = activation_post_process.calculate_qparams()
        new_mod = cls(mod.num_features, mod.eps)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        new_mod.running_mean = mod.running_mean
        new_mod.running_var = mod.running_var
        new_mod.scale = scale
        new_mod.zero_point = zero_point
        return new_mod

    @classmethod
    def from_reference(cls, bn, output_scale, output_zero_point):
        qbn = cls(
            bn.num_features,
            bn.eps,
            bn.momentum,
            device=bn.weight.device,
            dtype=bn.weight.dtype,
        )
        qbn.weight = bn.weight
        qbn.bias = bn.bias
        qbn.running_mean = bn.running_mean
        qbn.running_var = bn.running_var
        qbn.scale = output_scale
        qbn.zero_point = output_zero_point
        return qbn


class BatchNorm2d(_BatchNorm):
    r"""This is the quantized version of :class:`~torch.nn.BatchNorm2d`."""

    _NNI_BN_RELU_MODULE = nni.BNReLU2d

    def __init__(
        self, num_features, eps=1e-5, momentum=0.1, device=None, dtype=None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(num_features, eps, momentum, **factory_kwargs)

    def _get_name(self):
        return "QuantizedBatchNorm2d"

    def _check_input_dim(self, input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # disabling this since this is not symbolically traceable
        # self._check_input_dim(input)
        return torch.ops.quantized.batch_norm2d(
            input,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.eps,
            self.scale,
            self.zero_point,
        )

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        return _BatchNorm.from_float(
            cls, mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )


class BatchNorm3d(_BatchNorm):
    r"""This is the quantized version of :class:`~torch.nn.BatchNorm3d`."""

    _NNI_BN_RELU_MODULE = nni.BNReLU3d

    def __init__(self, num_features, eps=1e-5, momentum=0.1, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(num_features, eps, momentum, **factory_kwargs)

    def _get_name(self):
        return "QuantizedBatchNorm3d"

    def _check_input_dim(self, input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 5:
            raise ValueError("Input shape must be `(N, C, H, W)`!")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # disabling this since this is not symbolically traceable
        # self._check_input_dim(input)
        return torch.ops.quantized.batch_norm3d(
            input,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.eps,
            self.scale,
            self.zero_point,
        )

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        return _BatchNorm.from_float(
            cls, mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )
<|endoftext|># mypy: allow-untyped-defs
import torch
import torch.nn as nn
from torch import Tensor
from torch._jit_internal import List, Optional  # noqa: F401

from .utils import _hide_packed_params_repr, _quantize_weight


__all__ = ["EmbeddingPackedParams", "Embedding", "EmbeddingBag"]


class EmbeddingPackedParams(torch.nn.Module):
    _version = 1

    def __init__(self, num_embeddings, embedding_dim, dtype=torch.quint8):
        super().__init__()
        self.dtype = dtype
        if self.dtype in [torch.quint8, torch.quint4x2]:
            scales = torch.ones(num_embeddings, dtype=torch.float)
            zero_points = torch.zeros(num_embeddings, dtype=torch.float)
            wq = torch._empty_per_channel_affine_quantized(
                [num_embeddings, embedding_dim],
                scales=scales,
                zero_points=zero_points,
                axis=0,
                dtype=self.dtype,
            )
            self.set_weight(wq)
        else:
            raise NotImplementedError(
                f"Unsupported dtype on quantized embedding! Supports quint8 and quint4x2. Got dtype: {dtype}"
            )

    @torch.jit.export
    def set_weight(self, weight: torch.Tensor) -> None:
        if self.dtype in [torch.quint8, torch.quint4x2]:
            self._packed_weight = torch.ops.quantized.embedding_bag_prepack(weight)
        else:
            raise NotImplementedError(
                "Unsupported dtype for quantized embedding prepack! Supports quint8 and quint4x2."
            )

    @torch.jit.export
    def _weight(self):
        if self.dtype in [torch.quint8, torch.quint4x2]:
            return torch.ops.quantized.embedding_bag_unpack(self._packed_weight)
        else:
            raise NotImplementedError(
                "Unsupported dtype for quantized embedding unpack! Supports quint8 and quint4x2."
            )

    def forward(self, x):
        return x

    # Version 1
    #   self
    #   |--- _packed_weight : Tensor representing weight of EmbeddingPackedParamsBase
    #   |--- dtype : torch.dtype

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "dtype"] = self.dtype
        destination[prefix + "_packed_weight"] = self._weight()

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
        self.dtype = state_dict[prefix + "dtype"]
        state_dict.pop(prefix + "dtype")

        weight = state_dict[prefix + "_packed_weight"]
        state_dict.pop(prefix + "_packed_weight")
        self.set_weight(weight)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            False,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def __repr__(self):
        return self._weight().__repr__()


class Embedding(torch.nn.Module):
    r"""
    A quantized Embedding module with quantized packed weights as inputs.
    We adopt the same interface as `torch.nn.Embedding`, please see
    https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html for documentation.

    Similar to :class:`~torch.nn.Embedding`, attributes will be randomly
    initialized at module creation time and will be overwritten later

    Attributes:
        weight (Tensor): the non-learnable quantized weights of the module of
                         shape :math:`(\text{num\_embeddings}, \text{embedding\_dim})`.

    Examples::
        >>> warnings.filterwarnings("ignore", message=".*quantized tensor creation functions")  # docs: hide
        >>> m = nn.quantized.Embedding(num_embeddings=10, embedding_dim=12)
        >>> indices = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8])
        >>> output = m(indices)
        >>> print(output.size())
        torch.Size([9, 12])

    """

    _version = 1

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Optional[Tensor] = None,
        dtype=torch.quint8,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.dtype = dtype

        if _weight is None:
            scales = torch.ones(num_embeddings, dtype=torch.float)
            zero_points = torch.zeros(num_embeddings, dtype=torch.float)
            qweight = torch._empty_per_channel_affine_quantized(
                [num_embeddings, embedding_dim],
                scales=scales,
                zero_points=zero_points,
                axis=0,
                dtype=torch.quint8,
            )
        else:
            expected_shape = [num_embeddings, embedding_dim]
            if list(_weight.shape) != expected_shape:
                raise AssertionError(
                    f"Shape of weight does not match num_embeddings and embedding_dim: "
                    f"expected {expected_shape}, got {list(_weight.shape)}"
                )
            qweight = _weight

        self._packed_params = EmbeddingPackedParams(
            num_embeddings, embedding_dim, dtype
        )
        self._packed_params.set_weight(qweight)

    def forward(self, indices: Tensor) -> Tensor:
        if self.dtype == torch.quint4x2:
            return torch.ops.quantized.embedding_4bit(
                self._packed_params._packed_weight, indices
            )
        else:
            return torch.ops.quantized.embedding_byte(
                self._packed_params._packed_weight, indices
            )

    def _get_name(self):
        return "QuantizedEmbedding"

    def __repr__(self):
        return _hide_packed_params_repr(self, EmbeddingPackedParams)

    def extra_repr(self):
        extra_repr_str = (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, "
            f"dtype={self._packed_params.dtype}, qscheme={self.weight().qscheme()}"
        )

        return extra_repr_str

    def set_weight(self, w: torch.Tensor) -> None:
        self._packed_params.set_weight(w)

    def weight(self):
        return self._packed_params._weight()

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Create a quantized embedding module from a float module

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
                          utilities or provided by user
        """
        if hasattr(mod, "weight_fake_quant"):
            if type(mod) is not torch.ao.nn.qat.Embedding:
                raise AssertionError(
                    "nnq."
                    + cls.__name__
                    + ".from_float "
                    + "with fake quant only works for "
                    + torch.ao.nn.qat.Embedding.__name__
                )
            weight_observer = mod.weight_fake_quant
        else:
            if type(mod) is not nn.Embedding:
                raise AssertionError(
                    "nnq."
                    + cls.__name__
                    + ".from_float only works for "
                    + nn.Embedding.__name__
                )
            if not hasattr(mod, "qconfig"):
                raise AssertionError(
                    "Embedding input float module must have qconfig defined"
                )
            from torch.ao.quantization import float_qparams_weight_only_qconfig

            if mod.qconfig is not None and mod.qconfig.weight is not None:  # type: ignore[union-attr]
                weight_observer = mod.qconfig.weight()  # type: ignore[union-attr, operator]
            else:
                weight_observer = float_qparams_weight_only_qconfig.weight()

        dtype = weight_observer.dtype
        is_float_qparams_qconfig = (
            weight_observer.qscheme == torch.per_channel_affine_float_qparams
        )
        if not is_float_qparams_qconfig:
            raise AssertionError(
                "Embedding quantization is only supported with float_qparams_weight_only_qconfig."
            )

        if dtype != torch.quint8 and dtype != torch.quint4x2:
            raise AssertionError(
                f"The only supported dtype for nnq.Embedding is torch.quint8 and torch.quint4x2, got {dtype}"
            )

        # Run the observer to calculate qparams.
        weight_observer(mod.weight)
        qweight = _quantize_weight(mod.weight.float(), weight_observer)

        # Create quantized Embedding module and pass in the quantized weight
        qembedding = Embedding(mod.num_embeddings, mod.embedding_dim)
        qembedding.set_weight(qweight)
        return qembedding

    @classmethod
    def from_reference(cls, ref_embedding):
        qembedding = cls(
            ref_embedding.num_embeddings,
            ref_embedding.embedding_dim,
            ref_embedding.padding_idx,
            ref_embedding.max_norm,
            ref_embedding.norm_type,
            ref_embedding.scale_grad_by_freq,
            ref_embedding.sparse,
            ref_embedding.get_quantized_weight(),
            ref_embedding.weight_dtype,
        )
        return qembedding


class EmbeddingBag(Embedding):
    r"""
    A quantized EmbeddingBag module with quantized packed weights as inputs.
    We adopt the same interface as `torch.nn.EmbeddingBag`, please see
    https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html for documentation.

    Similar to :class:`~torch.nn.EmbeddingBag`, attributes will be randomly
    initialized at module creation time and will be overwritten later

    Attributes:
        weight (Tensor): the non-learnable quantized weights of the module of
                         shape :math:`(\text{num\_embeddings}, \text{embedding\_dim})`.

    Examples::
        >>> m = nn.quantized.EmbeddingBag(num_embeddings=10, embedding_dim=12, include_last_offset=True, mode='sum')
        >>> indices = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8, 6, 6, 9, 1, 6, 8, 8, 3, 2, 3, 6, 3, 6, 5, 7, 0, 8, 4, 6, 5, 8, 2, 3])
        >>> offsets = torch.tensor([0, 19, 20, 28, 28, 32])
        >>> output = m(indices, offsets)
        >>> print(output.size())
        torch.Size([5, 12])

    """

    _version = 1

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        mode: str = "sum",
        sparse: bool = False,
        _weight: Optional[Tensor] = None,
        include_last_offset: bool = False,
        dtype=torch.quint8,
    ) -> None:
        super().__init__(num_embeddings, embedding_dim, _weight=_weight, dtype=dtype)

        self.mode = mode
        self.pruned_weights = False
        self.include_last_offset = include_last_offset
        self.dtype = dtype

    def forward(
        self,
        indices: Tensor,
        offsets: Optional[Tensor] = None,
        per_sample_weights: Optional[Tensor] = None,
        compressed_indices_mapping: Optional[Tensor] = None,
    ) -> Tensor:
        if self.dtype == torch.quint4x2:
            return torch.ops.quantized.embedding_bag_4bit(
                self._packed_params._packed_weight,
                indices,
                offsets,
                False,
                0,
                self.pruned_weights,
                per_sample_weights,
                compressed_indices_mapping,
                self.include_last_offset,
            )
        else:
            return torch.ops.quantized.embedding_bag_byte(
                self._packed_params._packed_weight,
                indices,
                offsets,
                False,
                0,
                self.pruned_weights,
                per_sample_weights,
                compressed_indices_mapping,
                self.include_last_offset,
            )

    def _get_name(self):
        return "QuantizedEmbeddingBag"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Create a quantized embedding_bag module from a float module

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
                          utilities or provided by user
        """
        if hasattr(mod, "weight_fake_quant"):
            weight_observer = mod.weight_fake_quant
        else:
            if type(mod) is not nn.EmbeddingBag:
                raise AssertionError(
                    "nnq."
                    + cls.__name__
                    + ".from_float only works for "
                    + nn.EmbeddingBag.__name__
                )
            if not hasattr(mod, "qconfig"):
                raise AssertionError(
                    "EmbeddingBag input float module must have qconfig defined"
                )
            from torch.ao.quantization.qconfig import float_qparams_weight_only_qconfig

            if mod.qconfig is not None and mod.qconfig.weight is not None:  # type: ignore[union-attr]
                weight_observer = mod.qconfig.weight()  # type: ignore[union-attr, operator]
            else:
                weight_observer = float_qparams_weight_only_qconfig.weight()

        dtype = weight_observer.dtype
        is_float_qparams_qconfig = (
            weight_observer.qscheme == torch.per_channel_affine_float_qparams
        )
        if not is_float_qparams_qconfig:
            raise AssertionError(
                "EmbeddingBag quantization is only supported with float_qparams_weight_only_qconfig."
            )

        if dtype != torch.quint8 and dtype != torch.quint4x2:
            raise AssertionError(
                f"The only supported dtype for nnq.EmbeddingBag is torch.quint8 and torch.quint4x2, got {dtype}"
            )

        # Run the observer to calculate qparams.
        weight_observer(mod.weight)
        qweight = _quantize_weight(mod.weight.float(), weight_observer)

        # Create quantized EmbeddingBag module and pass in the quantized weight
        qembedding_bag = EmbeddingBag(
            mod.num_embeddings,
            mod.embedding_dim,
            max_norm=mod.max_norm,
            norm_type=mod.norm_type,
            scale_grad_by_freq=mod.scale_grad_by_freq,
            mode=mod.mode,
            sparse=mod.sparse,
            include_last_offset=mod.include_last_offset,
            dtype=dtype,
        )
        qembedding_bag.set_weight(qweight)
        return qembedding_bag

    @classmethod
    def from_reference(cls, ref_embedding_bag):
        qembedding_bag = cls(
            ref_embedding_bag.num_embeddings,
            ref_embedding_bag.embedding_dim,
            ref_embedding_bag.max_norm,
            ref_embedding_bag.norm_type,
            ref_embedding_bag.scale_grad_by_freq,
            ref_embedding_bag.mode,
            ref_embedding_bag.sparse,
            ref_embedding_bag.get_quantized_weight(),
            ref_embedding_bag.include_last_offset,
            ref_embedding_bag.weight_dtype,
        )
        return qembedding_bag
<|endoftext|>from typing import Any

import torch


__all__ = [
    "LSTM",
]


class LSTM(torch.ao.nn.quantizable.LSTM):
    r"""A quantized long short-term memory (LSTM).

    For the description and the argument types, please, refer to :class:`~torch.nn.LSTM`

    Attributes:
        layers : instances of the `_LSTMLayer`

    .. note::
        To access the weights and biases, you need to access them per layer.
        See examples in :class:`~torch.ao.nn.quantizable.LSTM`

    Examples::
        >>> # xdoctest: +SKIP
        >>> custom_module_config = {
        ...     'float_to_observed_custom_module_class': {
        ...         nn.LSTM: nn.quantizable.LSTM,
        ...     },
        ...     'observed_to_quantized_custom_module_class': {
        ...         nn.quantizable.LSTM: nn.quantized.LSTM,
        ...     }
        ... }
        >>> tq.prepare(model, prepare_custom_module_class=custom_module_config)
        >>> tq.convert(model, convert_custom_module_class=custom_module_config)
    """

    _FLOAT_MODULE = torch.ao.nn.quantizable.LSTM  # type: ignore[assignment]

    def _get_name(self) -> str:
        return "QuantizedLSTM"

    @classmethod
    def from_float(cls, *args: Any, **kwargs: Any) -> None:
        # The whole flow is float -> observed -> quantized
        # This class does observed -> quantized only
        raise NotImplementedError(
            "It looks like you are trying to convert a "
            "non-observed LSTM module. Please, see "
            "the examples on quantizable LSTMs."
        )

    @classmethod
    def from_observed(cls: type["LSTM"], other: torch.ao.nn.quantizable.LSTM) -> "LSTM":
        if not isinstance(other, cls._FLOAT_MODULE):  # type: ignore[has-type]
            raise AssertionError(
                f"Expected module type {cls._FLOAT_MODULE}, got {type(other)}"
            )
        converted = torch.ao.quantization.convert(
            other, inplace=False, remove_qconfig=True
        )
        converted.__class__ = cls
        return converted
<|endoftext|># mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
from collections.abc import Iterable

import torch
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.qat as nniqat
import torch.nn as nn
from torch.nn.utils.fusion import fuse_linear_bn_weights
from torch.nn.utils.parametrize import type_before_parametrizations

from .utils import _hide_packed_params_repr, _quantize_weight, WeightedQuantizedModule


__all__ = ["LinearPackedParams", "Linear"]


class LinearPackedParams(torch.nn.Module):
    _version = 3

    def __init__(self, dtype=torch.qint8):
        super().__init__()
        self.dtype = dtype
        if self.dtype == torch.qint8:
            wq = torch._empty_affine_quantized(
                [1, 1], scale=1.0, zero_point=0, dtype=torch.qint8
            )
        elif self.dtype == torch.float16:
            wq = torch.zeros([1, 1], dtype=torch.float)
        self.set_weight_bias(wq, None)  # type: ignore[possibly-undefined]

    @torch.jit.export
    def set_weight_bias(self, weight: torch.Tensor, bias: torch.Tensor | None) -> None:
        if self.dtype == torch.qint8:
            self._packed_params = torch.ops.quantized.linear_prepack(weight, bias)
        elif self.dtype == torch.float16:
            self._packed_params = torch.ops.quantized.linear_prepack_fp16(weight, bias)
        else:
            raise RuntimeError("Unsupported dtype on dynamic quantized linear!")

    @torch.jit.export
    def _weight_bias(self):
        if self.dtype == torch.qint8:
            return torch.ops.quantized.linear_unpack(self._packed_params)
        elif self.dtype == torch.float16:
            return torch.ops.quantized.linear_unpack_fp16(self._packed_params)
        else:
            raise RuntimeError("Unsupported dtype on dynamic quantized linear!")

    def forward(self, x):
        return x

    # Version 1
    #   self
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #
    # Version 2
    #   self
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #   |--- dtype : torch.dtype
    #
    # Version 3
    #   self
    #   |--- _packed_params : (Tensor, Tensor) representing (weight, bias)
    #                         of LinearPackedParams
    #   |--- dtype : torch.dtype
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "dtype"] = self.dtype
        destination[prefix + "_packed_params"] = self._weight_bias()

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
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            self.dtype = torch.qint8
        else:
            self.dtype = state_dict[prefix + "dtype"]
            state_dict.pop(prefix + "dtype")

        if version is None or version < 3:
            self.set_weight_bias(
                state_dict[prefix + "weight"], state_dict[prefix + "bias"]
            )
            state_dict.pop(prefix + "weight")
            state_dict.pop(prefix + "bias")

        if version == 3:
            weight, bias = state_dict[prefix + "_packed_params"]
            state_dict.pop(prefix + "_packed_params")
            self.set_weight_bias(weight, bias)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            False,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def __repr__(self):
        return self._weight_bias().__repr__()


class Linear(WeightedQuantizedModule):
    r"""
    A quantized linear module with quantized tensor as inputs and outputs.
    We adopt the same interface as `torch.nn.Linear`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear for documentation.

    Similar to :class:`~torch.nn.Linear`, attributes will be randomly
    initialized at module creation time and will be overwritten later

    Attributes:
        weight (Tensor): the non-learnable quantized weights of the module of
                         shape :math:`(\text{out\_features}, \text{in\_features})`.
        bias (Tensor): the non-learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized to zero.
        scale: `scale` parameter of output Quantized Tensor, type: double
        zero_point: `zero_point` parameter for output Quantized Tensor, type: long

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> m = nn.quantized.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> # xdoctest: +SKIP
        >>> input = torch.quantize_per_tensor(input, 1.0, 0, torch.quint8)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    _version = 3
    _FLOAT_MODULE = (nn.Linear, nn.modules.linear.NonDynamicallyQuantizableLinear)

    def __init__(self, in_features, out_features, bias_=True, dtype=torch.qint8):
        super().__init__()
        # We don't muck around with buffers or attributes or anything here
        # to keep the module simple. *everything* is simply a Python attribute.
        # Serialization logic is explicitly handled in the below serialization and
        # deserialization modules
        self.in_features = in_features
        self.out_features = out_features
        bias = None
        if bias_:
            bias = torch.zeros(out_features, dtype=torch.float)

        if dtype == torch.qint8:
            qweight = torch._empty_affine_quantized(
                [out_features, in_features], scale=1, zero_point=0, dtype=torch.qint8
            )
        elif dtype == torch.float16:
            qweight = torch.zeros([out_features, in_features], dtype=torch.float)
        else:
            raise RuntimeError("Unsupported dtype specified for quantized Linear!")

        self._packed_params = LinearPackedParams(dtype)
        self._packed_params.set_weight_bias(qweight, bias)
        self.scale = 1.0
        self.zero_point = 0

    def _get_name(self):
        return "QuantizedLinear"

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, scale={self.scale}, "
            f"zero_point={self.zero_point}, qscheme={self.weight().qscheme()}"
        )

    def __repr__(self):
        return _hide_packed_params_repr(self, LinearPackedParams)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.quantized.linear(
            x, self._packed_params._packed_params, self.scale, self.zero_point
        )

    # ===== Serialization methods =====
    # The special consideration here is that we have to unpack the weights into their
    # regular QTensor form for serialization. Packed weights should not live
    # outside the process in which they were created, rather they should be derived
    # from the QTensor weight.
    #
    # Version 1
    #   self
    #   |--- scale : float
    #   |--- zero_point : int
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #
    # Version 2
    #   self
    #   |--- scale : float
    #   |--- zero_point : int
    #   |--- _packed_params : Module
    #        |--- weight : Tensor
    #        |--- bias : Tensor
    #
    # Version 3
    #   self
    #   |--- scale : float
    #   |--- zero_point : int
    #   |--- _packed_params : Module
    #        |--- _packed_params : (Tensor, Tensor) representing weight, bias
    #                              of LinearPackedParams C++ struct
    #
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "scale"] = torch.tensor(self.scale)
        destination[prefix + "zero_point"] = torch.tensor(self.zero_point)

    # ===== Deserialization methods =====
    # Counterpart to the serialization methods, we must pack the serialized QTensor
    # weight into its packed format for use by the FBGEMM ops.
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
        self.scale = float(state_dict[prefix + "scale"])
        state_dict.pop(prefix + "scale")

        self.zero_point = int(state_dict[prefix + "zero_point"])
        state_dict.pop(prefix + "zero_point")

        version = local_metadata.get("version", None)

        if version is None or version == 1:
            # We moved the parameters into a LinearPackedParameters submodule
            weight = state_dict.pop(prefix + "weight")
            bias = state_dict.pop(prefix + "bias")
            state_dict.update(
                {
                    prefix + "_packed_params.weight": weight,
                    prefix + "_packed_params.bias": bias,
                }
            )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            False,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    # Function rather than property to make sure that JIT serialization doesn't
    # register this as an attribute
    def _weight_bias(self):
        return self._packed_params._weight_bias()

    def weight(self):
        return self._weight_bias()[0]

    def bias(self):
        return self._weight_bias()[1]

    def set_weight_bias(self, w: torch.Tensor, b: torch.Tensor | None) -> None:
        self._packed_params.set_weight_bias(w, b)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Create a quantized module from an observed float module

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
                          utilities or provided by the user
            use_precomputed_fake_quant (bool): if True, the module will reuse min/max
                          values from the precomputed fake quant module.
        """
        if hasattr(mod, "weight_fake_quant"):
            if type_before_parametrizations(mod) == nniqat.LinearBn1d:
                mod.weight, mod.bias = fuse_linear_bn_weights(
                    mod.weight,
                    mod.bias,
                    mod.bn.running_mean,
                    mod.bn.running_var,
                    mod.bn.eps,
                    mod.bn.weight,
                    mod.bn.bias,
                )
            weight_post_process = mod.weight_fake_quant
            activation_post_process = mod.activation_post_process
        else:
            # This function does not participate in JIT, so it is OK to ignore
            # the type mismatch in assignment. Also, mypy has an issue with
            # iterables not being implemented, so we are ignoring those too.
            if not isinstance(cls._FLOAT_MODULE, Iterable):
                # pyrefly: ignore [bad-assignment]
                cls._FLOAT_MODULE = [cls._FLOAT_MODULE]
            supported_modules = ", ".join(
                [float_mod.__name__ for float_mod in cls._FLOAT_MODULE]
            )
            error_msg = f"nnq.{cls.__name__}.from_float only works for {supported_modules}, but got: {type(mod)}"
            if type_before_parametrizations(mod) not in cls._FLOAT_MODULE:
                raise AssertionError(error_msg)
            if not hasattr(mod, "qconfig"):
                raise AssertionError("Input float module must have qconfig defined")
            activation_post_process = mod.activation_post_process
            if type_before_parametrizations(mod) == nni.LinearReLU:
                mod = mod[0]
            weight_post_process = (
                mod.qconfig.weight()
                if not hasattr(mod, "weight_fake_quant")
                else mod.weight_fake_quant
            )

        if not use_precomputed_fake_quant:
            # Observer may not have been called yet
            # Observer might have been called in the previous stage via PTQ algorithm e.g. AdaRound
            weight_post_process(mod.weight)
        dtype = weight_post_process.dtype
        act_scale, act_zp = activation_post_process.calculate_qparams()
        if dtype != torch.qint8:
            raise AssertionError(
                f"Weight observer must have dtype torch.qint8, got {dtype}"
            )
        qweight = _quantize_weight(mod.weight.float(), weight_post_process)
        qlinear = cls(mod.in_features, mod.out_features, dtype=dtype)
        qlinear.set_weight_bias(qweight, mod.bias)
        qlinear.scale = float(act_scale)
        qlinear.zero_point = int(act_zp)
        return qlinear

    @classmethod
    def from_reference(cls, ref_qlinear, output_scale, output_zero_point):
        r"""Create a (fbgemm/qnnpack) quantized module from a reference quantized module

        Args:
            ref_qlinear (Module): a reference quantized linear module, either produced by torch.ao.quantization
                          utilities or provided by the user
            output_scale (float): scale for output Tensor
            output_zero_point (int): zero point for output Tensor
        """
        qlinear = cls(ref_qlinear.in_features, ref_qlinear.out_features)
        qweight = ref_qlinear.get_quantized_weight()
        qlinear.set_weight_bias(qweight, ref_qlinear.bias)

        qlinear.scale = float(output_scale)
        qlinear.zero_point = int(output_zero_point)
        return qlinear
<|endoftext|># mypy: allow-untyped-defs
import typing

import torch


__all__ = [
    "ReferenceQuantizedModule",
]


class ReferenceQuantizedModule(torch.nn.Module):
    def _init_weight_qparams(self, weight_qparams, device):
        if weight_qparams is None:
            weight_qparams = {
                "qscheme": torch.per_tensor_affine,
                "dtype": torch.quint8,
                "scale": 1.0,
                "zero_point": 0,
            }

        self.weight_qscheme: torch.qscheme = weight_qparams["qscheme"]
        self.weight_dtype = weight_qparams["dtype"]
        if self.weight_qscheme not in [
            None,
            torch.per_tensor_affine,
            torch.per_channel_affine,
            torch.per_channel_affine_float_qparams,
        ]:
            raise AssertionError(
                f"qscheme: {self.weight_qscheme} is not supported in reference quantized {self._get_name()}"
            )
        if self.weight_dtype in [
            torch.quint8,
            torch.qint8,
            torch.quint4x2,
            torch.qint32,
        ]:
            zero_point_dtype = (
                weight_qparams["zero_point"].dtype
                if isinstance(weight_qparams["zero_point"], torch.Tensor)
                else torch.int
            )
            w_scale = weight_qparams["scale"]
            w_scale_tensor = (
                w_scale.detach().clone()
                if isinstance(w_scale, torch.Tensor)
                else torch.tensor(w_scale, dtype=torch.float, device=device)
            )
            self.register_buffer("weight_scale", w_scale_tensor)
            w_zp = weight_qparams["zero_point"]
            w_zp_tensor = (
                w_zp.detach().clone()
                if isinstance(w_zp, torch.Tensor)
                else torch.tensor(w_zp, dtype=zero_point_dtype, device=device)
            )
            self.register_buffer("weight_zero_point", w_zp_tensor)
            if self.weight_qscheme in [
                torch.per_channel_affine,
                torch.per_channel_affine_float_qparams,
            ]:
                w_axis = weight_qparams["axis"]
                w_axis_tensor = (
                    w_axis.detach().clone()
                    if isinstance(w_axis, torch.Tensor)
                    else torch.tensor(w_axis, dtype=torch.int, device=device)
                )
                self.register_buffer("weight_axis", w_axis_tensor)
            else:
                # added for TorchScriptability, not used
                self.register_buffer(
                    "weight_axis", torch.tensor(0, dtype=torch.int, device=device)
                )
        else:
            # added for TorchScriptability, and for torch.float
            self.register_buffer(
                "weight_scale", torch.tensor(1.0, dtype=torch.float, device=device)
            )
            self.register_buffer(
                "weight_zero_point", torch.tensor(0, dtype=torch.int, device=device)
            )
            self.register_buffer(
                "weight_axis", torch.tensor(0, dtype=torch.int, device=device)
            )

        self.is_decomposed: bool = weight_qparams.get("is_decomposed", False)
        # store weight_axis as weight_axis_int due to some constraints of torchdynamo.export
        # for capturing `.item` operations
        self.weight_axis_int: int = self.weight_axis.item()  # type: ignore[operator, assignment]

        self.weight_quant_min: int | None = weight_qparams.get("quant_min")

        self.weight_quant_max: int | None = weight_qparams.get("quant_max")

    def get_weight(self):
        """
        Fake quantize (quantize and dequantize) the weight with
        the quantization parameters for weight, this is used to
        simulate the numerics for the quantized weight in a quantized
        model
        """
        # suppress mypy warning
        if not isinstance(self.weight_scale, torch.Tensor):
            raise AssertionError("weight_scale must be a Tensor")
        if not isinstance(self.weight_zero_point, torch.Tensor):
            raise AssertionError("weight_zero_point must be a Tensor")
        if self.is_decomposed:
            return _quantize_and_dequantize_weight_decomposed(
                self.weight,  # type: ignore[arg-type]
                self.weight_qscheme,
                self.weight_dtype,
                self.weight_scale,
                self.weight_zero_point,
                self.weight_axis_int,
                self.weight_quant_min,
                self.weight_quant_max,
            )
        else:
            return _quantize_and_dequantize_weight(
                self.weight,  # type: ignore[arg-type]
                self.weight_qscheme,
                self.weight_dtype,
                self.weight_scale,
                self.weight_zero_point,
                self.weight_axis_int,
            )

    def get_quantized_weight(self):
        # suppress mypy warning
        if not isinstance(self.weight_scale, torch.Tensor):
            raise AssertionError("weight_scale must be a Tensor")
        if not isinstance(self.weight_zero_point, torch.Tensor):
            raise AssertionError("weight_zero_point must be a Tensor")
        # assert isinstance(self.weight_axis, torch.Tensor)
        if self.is_decomposed:
            return _quantize_weight_decomposed(
                self.weight,  # type: ignore[arg-type]
                self.weight_qscheme,
                self.weight_dtype,
                self.weight_scale,
                self.weight_zero_point,
                self.weight_axis_int,
                self.weight_quant_min,
                self.weight_quant_max,
            )
        else:
            return _quantize_weight(
                self.weight,  # type: ignore[arg-type]
                self.weight_qscheme,
                self.weight_dtype,
                self.weight_scale,
                self.weight_zero_point,
                self.weight_axis_int,
            )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        _save_weight_qparams(
            destination,
            prefix,
            self.weight_qscheme,
            self.weight_dtype,
            self.weight_scale,
            self.weight_zero_point,
            self.weight_axis,
        )

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
        for key in _get_weight_qparam_keys(state_dict, prefix):
            setattr(self, key, state_dict[prefix + key])
            state_dict.pop(prefix + key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            False,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def _quantize_weight_decomposed(
    weight: torch.Tensor,
    weight_qscheme: torch.qscheme,
    weight_dtype: torch.dtype,
    weight_scale: torch.Tensor,
    weight_zero_point: torch.Tensor,
    weight_axis: int,
    weight_quant_min: int | None,
    weight_quant_max: int | None,
) -> torch.Tensor:
    _DTYPE_TO_QVALUE_BOUNDS: dict[torch.dtype, tuple[int, int]] = {
        torch.uint8: (0, 255),
        torch.int8: (-128, 127),
        torch.int32: (-2147483648, 2147483647),  # torch.jit interprets 2**31 as a float
    }

    # TODO: add an util function for converting qdtype to dtype
    _QDTYPE_TO_UNDERLYING_INT_REPR_DTYPE = {
        torch.quint8: torch.uint8,
        torch.qint8: torch.int8,
        torch.qint32: torch.int32,
    }
    if weight_qscheme == torch.per_tensor_affine:
        if weight_dtype in [torch.quint8, torch.qint8, torch.qint32]:
            weight_dtype_ = _QDTYPE_TO_UNDERLYING_INT_REPR_DTYPE[weight_dtype]
            if weight_quant_min is None or weight_quant_max is None:
                weight_quant_min, weight_quant_max = _DTYPE_TO_QVALUE_BOUNDS[
                    weight_dtype_
                ]
            weight = torch.ops.quantized_decomposed.quantize_per_tensor(
                weight,
                weight_scale,
                weight_zero_point,
                weight_quant_min,
                weight_quant_max,
                weight_dtype_,
            )
            return weight
    elif weight_qscheme in [
        torch.per_channel_affine,
        torch.per_channel_affine_float_qparams,
    ]:
        # TODO: torch.quint4x2 is not supported
        if weight_dtype in [torch.quint8, torch.qint8, torch.qint32]:
            weight_dtype_ = _QDTYPE_TO_UNDERLYING_INT_REPR_DTYPE[weight_dtype]
            if weight_quant_min is None or weight_quant_max is None:
                weight_quant_min, weight_quant_max = _DTYPE_TO_QVALUE_BOUNDS[
                    weight_dtype_
                ]
            weight = torch.ops.quantized_decomposed.quantize_per_channel(
                weight,
                weight_scale,
                weight_zero_point,
                weight_axis,
                weight_quant_min,
                weight_quant_max,
                weight_dtype_,
            )  # type: ignore[arg-type]
            return weight
    raise ValueError(f"Unsupported dtype and qscheme: {weight_dtype}, {weight_qscheme}")


def _dequantize_weight_decomposed(
    weight: torch.Tensor,
    weight_qscheme: torch.qscheme,
    weight_dtype: torch.dtype,
    weight_scale: torch.Tensor,
    weight_zero_point: torch.Tensor,
    weight_axis: int,
    weight_quant_min: int | None,
    weight_quant_max: int | None,
) -> torch.Tensor:
    # TODO: get the quant_min and quant_max from activation_post_process
    _DTYPE_TO_QVALUE_BOUNDS: dict[torch.dtype, tuple[int, int]] = {
        torch.uint8: (0, 255),
        torch.int8: (-128, 127),
        torch.int32: (-2147483648, 2147483647),  # torch.jit interprets 2**31 as a float
    }
    # TODO: add an util function for converting qdtype to dtype
    _QDTYPE_TO_UNDERLYING_INT_REPR_DTYPE = {
        torch.quint8: torch.uint8,
        torch.qint8: torch.int8,
        torch.qint32: torch.int32,
    }
    weight_dtype_ = _QDTYPE_TO_UNDERLYING_INT_REPR_DTYPE[weight_dtype]
    if weight_quant_min is None or weight_quant_max is None:
        weight_quant_min, weight_quant_max = _DTYPE_TO_QVALUE_BOUNDS[weight_dtype_]
    if weight_qscheme == torch.per_tensor_affine:
        if weight_dtype in [torch.quint8, torch.qint8, torch.qint32]:
            weight = torch.ops.quantized_decomposed.dequantize_per_tensor(
                weight,
                weight_scale,
                weight_zero_point,
                weight_quant_min,
                weight_quant_max,
                weight_dtype_,
            )
            return weight
    elif weight_qscheme in [
        torch.per_channel_affine,
        torch.per_channel_affine_float_qparams,
    ]:
        # TODO: torch.quint4x2 is not supported
        if weight_dtype in [torch.quint8, torch.qint8, torch.qint32]:
            weight = torch.ops.quantized_decomposed.dequantize_per_channel(
                weight,
                weight_scale,
                weight_zero_point,
                weight_axis,
                weight_quant_min,
                weight_quant_max,
                weight_dtype_,
            )  # type: ignore[arg-type]
            return weight
    raise ValueError(f"Unsupported dtype and qscheme: {weight_dtype}, {weight_qscheme}")


def _quantize_weight(
    weight: torch.Tensor,
    weight_qscheme: torch.qscheme,
    weight_dtype: torch.dtype,
    weight_scale: torch.Tensor,
    weight_zero_point: torch.Tensor,
    weight_axis_int: int,
) -> torch.Tensor:
    if weight_dtype == torch.float16:
        weight = weight.to(weight_dtype)
        return weight

    if weight_qscheme == torch.per_tensor_affine:
        if weight_dtype in [torch.quint8, torch.qint8, torch.qint32]:
            weight = torch.quantize_per_tensor(
                weight, weight_scale, weight_zero_point, weight_dtype
            )
            return weight
    elif weight_qscheme in [
        torch.per_channel_affine,
        torch.per_channel_affine_float_qparams,
    ]:
        if weight_dtype in [torch.quint8, torch.qint8, torch.quint4x2, torch.qint32]:
            weight = torch.quantize_per_channel(
                weight, weight_scale, weight_zero_point, weight_axis_int, weight_dtype
            )  # type: ignore[arg-type]
            return weight
    raise ValueError(f"Unsupported dtype and qscheme: {weight_dtype}, {weight_qscheme}")


def _quantize_and_dequantize_weight_decomposed(
    weight: torch.Tensor,
    weight_qscheme: torch.qscheme,
    weight_dtype: torch.dtype,
    weight_scale: torch.Tensor,
    weight_zero_point: torch.Tensor,
    weight_axis_int: int,
    weight_quant_min: int | None,
    weight_quant_max: int | None,
) -> torch.Tensor:
    """Quantize and then dequantize the weight based on
    the quantization parameters
    """
    if weight_qscheme in [
        torch.per_tensor_affine,
        torch.per_channel_affine,
        torch.per_channel_affine_float_qparams,
    ]:
        weight_quant = _quantize_weight_decomposed(
            weight,
            weight_qscheme,
            weight_dtype,
            weight_scale,
            weight_zero_point,
            weight_axis_int,
            weight_quant_min,
            weight_quant_max,
        )
        weight_dequant = _dequantize_weight_decomposed(
            weight_quant,
            weight_qscheme,
            weight_dtype,
            weight_scale,
            weight_zero_point,
            weight_axis_int,
            weight_quant_min,
            weight_quant_max,
        )
    else:
        weight_dequant = weight
    return weight_dequant


def _quantize_and_dequantize_weight(
    weight: torch.Tensor,
    weight_qscheme: torch.qscheme,
    weight_dtype: torch.dtype,
    weight_scale: torch.Tensor,
    weight_zero_point: torch.Tensor,
    weight_axis_int: int,
) -> torch.Tensor:
    """Quantize and then dequantize the weight based on
    the quantization parameters
    """
    if weight_qscheme in [
        torch.per_tensor_affine,
        torch.per_channel_affine,
        torch.per_channel_affine_float_qparams,
    ]:
        weight_quant = _quantize_weight(
            weight,
            weight_qscheme,
            weight_dtype,
            weight_scale,
            weight_zero_point,
            weight_axis_int,
        )
        weight_dequant = weight_quant.dequantize()
    else:
        weight_dequant = weight
    return weight_dequant


def _save_weight_qparams(
    destination,
    prefix,
    weight_qscheme,
    weight_dtype,
    weight_scale,
    weight_zero_point,
    weight_axis,
):
    destination[prefix + "weight_qscheme"] = weight_qscheme
    destination[prefix + "weight_dtype"] = weight_dtype
    if weight_qscheme is not None:
        destination[prefix + "weight_scale"] = weight_scale
        destination[prefix + "weight_zero_point"] = weight_zero_point
        if weight_qscheme == torch.per_channel_affine:
            destination[prefix + "weight_axis"] = weight_axis


def _get_weight_qparam_keys(state_dict: dict[str, typing.Any], prefix: str):
    keys = ["weight_qscheme", "weight_dtype"]
    weight_qscheme = state_dict[prefix + "weight_qscheme"]
    if weight_qscheme is not None:
        keys.append("weight_scale")
        keys.append("weight_zero_point")
        if weight_qscheme == torch.quantize_per_channel:
            keys.append("weight_axis")
    return keys
<|endoftext|># mypy: allow-untyped-defs
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t

from .utils import ReferenceQuantizedModule


__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
]


class _ConvNd(torch.nn.modules.conv._ConvNd, ReferenceQuantizedModule):
    """A reference version of nn.quantized.Conv2d
    we will not pack the parameters in this module, since weight packing is an
    optimization for quantized backends supported in PyTorch (fbgemm/qnnpack),
    this is useful when user want to use this module in other backends like Glow.
    """

    __annotations__ = {"bias": torch.Tensor | None}
    _IS_REFERENCE = True

    @staticmethod
    def from_float(cls, float_conv, weight_qparams):
        qref_conv = cls(
            float_conv.in_channels,
            float_conv.out_channels,
            float_conv.kernel_size,  # type: ignore[arg-type]
            float_conv.stride,  # type: ignore[arg-type]
            float_conv.padding,  # type: ignore[arg-type]
            float_conv.dilation,  # type: ignore[arg-type]
            float_conv.groups,
            float_conv.bias is not None,  # type: ignore[arg-type]
            float_conv.padding_mode,
            device=float_conv.weight.device,
            dtype=float_conv.weight.dtype,
            weight_qparams=weight_qparams,
        )
        qref_conv.weight = torch.nn.Parameter(float_conv.weight.detach())
        if float_conv.bias is not None:
            qref_conv.bias = torch.nn.Parameter(float_conv.bias.detach())
        return qref_conv


class Conv1d(_ConvNd, nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        device=None,
        dtype=None,
        weight_qparams: dict[str, Any] | None = None,
    ):
        nn.Conv1d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        self._init_weight_qparams(weight_qparams, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        we have:
        w(float) -- quant - dequant \
        x(float) ------------- F.conv1d ---

        In the full model, we will see
        w(float) -- quant - *dequant \
        x -- quant --- *dequant --  *F.conv1d --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized conv1d
        """
        weight_quant_dequant = self.get_weight()

        result = F.conv1d(
            x,
            weight_quant_dequant,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return result

    def _get_name(self):
        return "QuantizedConv1d(Reference)"

    @classmethod
    def from_float(cls, float_conv, weight_qparams):  # type: ignore[override]
        return _ConvNd.from_float(cls, float_conv, weight_qparams)


class Conv2d(_ConvNd, nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        weight_qparams: dict[str, Any] | None = None,
    ):
        nn.Conv2d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            # pyrefly: ignore [bad-argument-type]
            padding_mode,
            device,
            dtype,
        )
        self._init_weight_qparams(weight_qparams, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        we have:
        w(float) -- quant - dequant \
        x(float) ------------- F.conv2d ---

        In the full model, we will see
        w(float) -- quant - *dequant \
        x -- quant --- *dequant --  *F.conv2d --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized conv2d
        """
        weight_quant_dequant = self.get_weight()

        result = F.conv2d(
            x,
            weight_quant_dequant,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return result

    def _get_name(self):
        return "QuantizedConv2d(Reference)"

    @classmethod
    def from_float(cls, float_conv, weight_qparams):  # type: ignore[override]
        return _ConvNd.from_float(cls, float_conv, weight_qparams)


class Conv3d(_ConvNd, nn.Conv3d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        weight_qparams: dict[str, Any] | None = None,
    ):
        nn.Conv3d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            # pyrefly: ignore [bad-argument-type]
            padding_mode,
            device,
            dtype,
        )
        self._init_weight_qparams(weight_qparams, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        we have:
        w(float) -- quant - dequant \
        x(float) ------------- F.conv3d ---

        In the full model, we will see
        w(float) -- quant - *dequant \
        x -- quant --- *dequant --  *F.conv3d --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized conv3d
        """
        weight_quant_dequant = self.get_weight()

        result = F.conv3d(
            x,
            weight_quant_dequant,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return result

    def _get_name(self):
        return "QuantizedConv3d(Reference)"

    @classmethod
    def from_float(cls, float_conv, weight_qparams):  # type: ignore[override]
        return _ConvNd.from_float(cls, float_conv, weight_qparams)


class _ConvTransposeNd(_ConvNd, torch.nn.modules.conv._ConvTransposeNd):
    """A reference version of nn.quantized.ConvTranspose2d
    we will not pack the parameters in this module, since weight packing is an
    optimization for quantized backends supported in PyTorch (fbgemm/qnnpack),
    this is useful when user want to use this module in other backends like Glow.
    """

    @staticmethod
    def from_float(cls, float_conv, weight_qparams):
        qref_conv = cls(
            float_conv.in_channels,
            float_conv.out_channels,
            float_conv.kernel_size,  # type: ignore[arg-type]
            float_conv.stride,  # type: ignore[arg-type]
            float_conv.padding,  # type: ignore[arg-type]
            float_conv.output_padding,  # type: ignore[arg-type]
            float_conv.groups,
            float_conv.bias is not None,  # type: ignore[arg-type]
            float_conv.dilation,  # type: ignore[arg-type]
            float_conv.padding_mode,
            device=float_conv.weight.device,
            dtype=float_conv.weight.dtype,
            weight_qparams=weight_qparams,
        )
        qref_conv.weight = torch.nn.Parameter(float_conv.weight.detach())
        if float_conv.bias is not None:
            qref_conv.bias = torch.nn.Parameter(float_conv.bias.detach())
        return qref_conv


class ConvTranspose1d(_ConvTransposeNd, nn.ConvTranspose1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        output_padding: _size_1_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_1_t = 1,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        device=None,
        dtype=None,
        weight_qparams: dict[str, Any] | None = None,
    ):
        nn.ConvTranspose1d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
            device,
            dtype,
        )
        self._init_weight_qparams(weight_qparams, device)

    def forward(
        self, x: torch.Tensor, output_size: list[int] | None = None
    ) -> torch.Tensor:
        """
        we have:
        w(float) -- quant - dequant \
        x(float) ------------- F.convTranspose1d ---
        In the full model, we will see
        w(float) -- quant - *dequant \
        x -- quant --- *dequant --  *F.convTranspose1d --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized conv1d
        """

        if not isinstance(self.padding, tuple):
            raise AssertionError(
                f"Expected self.padding to be a tuple, got {type(self.padding)}"
            )
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        output_padding = self._output_padding(
            input,  # type: ignore[arg-type]
            output_size,
            self.stride,  # type: ignore[arg-type]
            self.padding,  # type: ignore[arg-type]
            self.kernel_size,  # type: ignore[arg-type]
            self.dilation,  # type: ignore[arg-type]
        )

        weight_quant_dequant = self.get_weight()
        result = F.conv_transpose1d(
            x,
            weight_quant_dequant,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )
        return result

    def _get_name(self):
        return "QuantizedConvTranspose1d(Reference)"

    @classmethod
    def from_float(cls, float_conv, weight_qparams):  # type: ignore[override]
        return _ConvTransposeNd.from_float(cls, float_conv, weight_qparams)


class ConvTranspose2d(_ConvTransposeNd, nn.ConvTranspose2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        device=None,
        dtype=None,
        weight_qparams: dict[str, Any] | None = None,
    ):
        nn.ConvTranspose2d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            # pyrefly: ignore [bad-argument-type]
            padding_mode,
            device,
            dtype,
        )
        self._init_weight_qparams(weight_qparams, device)

    def forward(
        self, x: torch.Tensor, output_size: list[int] | None = None
    ) -> torch.Tensor:
        """
        we have:
        w(float) -- quant - dequant \
        x(float) ------------- F.convTranspose2d ---
        In the full model, we will see
        w(float) -- quant - *dequant \
        x -- quant --- *dequant --  *F.convTranspose2d --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized conv2d
        """
        if not isinstance(self.padding, tuple):
            raise AssertionError(
                f"Expected self.padding to be a tuple, got {type(self.padding)}"
            )
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.

        output_padding = self._output_padding(
            input,  # type: ignore[arg-type]
            output_size,
            self.stride,  # type: ignore[arg-type]
            self.padding,  # type: ignore[arg-type]
            self.kernel_size,  # type: ignore[arg-type]
            self.dilation,  # type: ignore[arg-type]
        )

        weight_quant_dequant = self.get_weight()
        result = F.conv_transpose2d(
            x,
            weight_quant_dequant,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )

        return result

    def _get_name(self):
        return "QuantizedConvTranspose2d(Reference)"

    @classmethod
    def from_float(cls, float_conv, weight_qparams):  # type: ignore[override]
        return _ConvTransposeNd.from_float(cls, float_conv, weight_qparams)


class ConvTranspose3d(_ConvTransposeNd, nn.ConvTranspose3d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        device=None,
        dtype=None,
        weight_qparams: dict[str, Any] | None = None,
    ):
        nn.ConvTranspose3d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            # pyrefly: ignore [bad-argument-type]
            padding_mode,
            device,
            dtype,
        )
        self._init_weight_qparams(weight_qparams, device)

    def forward(
        self, x: torch.Tensor, output_size: list[int] | None = None
    ) -> torch.Tensor:
        """
        we have:
        w(float) -- quant - dequant \
        x(float) ------------- F.convTranspose3d ---
        In the full model, we will see
        w(float) -- quant - *dequant \
        x -- quant --- *dequant --  *F.convTranspose3d --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized conv3d
        """

        if not isinstance(self.padding, tuple):
            raise AssertionError(
                f"Expected self.padding to be a tuple, got {type(self.padding)}"
            )
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        output_padding = self._output_padding(
            input,  # type: ignore[arg-type]
            output_size,
            self.stride,  # type: ignore[arg-type]
            self.padding,  # type: ignore[arg-type]
            self.kernel_size,  # type: ignore[arg-type]
            self.dilation,  # type: ignore[arg-type]
        )

        weight_quant_dequant = self.get_weight()
        result = F.conv_transpose3d(
            x,
            weight_quant_dequant,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )
        return result

    def _get_name(self):
        return "QuantizedConvTranspose3d(Reference)"

    @classmethod
    def from_float(cls, float_conv, weight_qparams):  # type: ignore[override]
        return _ConvTransposeNd.from_float(cls, float_conv, weight_qparams)
<|endoftext|>from .conv import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)
from .linear import Linear
from .rnn import GRU, GRUCell, LSTM, LSTMCell, RNNCell
from .sparse import Embedding, EmbeddingBag


__all__ = [
    "Linear",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "RNNCell",
    "LSTMCell",
    "GRUCell",
    "LSTM",
    "GRU",
    "Embedding",
    "EmbeddingBag",
]
<|endoftext|># mypy: allow-untyped-defs
from typing import Any

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils import ReferenceQuantizedModule


__all__ = ["Embedding", "EmbeddingBag"]


class Embedding(nn.Embedding, ReferenceQuantizedModule):
    """A reference quantized Embedding module that fits into the
    FX Graph Mode Quantization workflow, activation will be floating point Tensor,
    we will store floating point weight as well in the module, but in forward we'll
    quantize and dequantize the weight before running the floating point functional
    embedding operator.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Tensor | None = None,
        device=None,
        dtype=None,
        weight_qparams: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
            # pyrefly: ignore [bad-argument-type]
            device,
            dtype,
        )
        self._init_weight_qparams(weight_qparams, device)

    def _get_name(self):
        return "QuantizedEmbedding(Reference)"

    def forward(self, input: Tensor) -> Tensor:
        weight_quant_dequant = self.get_weight()
        return F.embedding(
            input,
            weight_quant_dequant,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    @classmethod
    def from_float(cls, mod, weight_qparams):
        return cls(
            mod.num_embeddings,
            mod.embedding_dim,
            mod.padding_idx,
            mod.max_norm,
            mod.norm_type,
            mod.scale_grad_by_freq,
            mod.sparse,
            mod.weight,
            mod.weight.device,
            mod.weight.dtype,
            weight_qparams,
        )


class EmbeddingBag(nn.EmbeddingBag, ReferenceQuantizedModule):
    """A reference quantized EmbeddingBag module that fits into the
    FX Graph Mode Quantization workflow, activation will be floating point Tensor,
    we will store floating point weight as well in the module, but in forward we'll
    quantize and dequantize the weight before running the floating point functional
    embedding operator.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        mode: str = "mean",
        sparse: bool = False,
        _weight: Tensor | None = None,
        include_last_offset: bool = False,
        padding_idx: int | None = None,
        device=None,
        dtype=None,
        weight_qparams: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            mode,
            sparse,
            _weight,
            include_last_offset,
            padding_idx,
            device,
            dtype,
        )
        self._init_weight_qparams(weight_qparams, device)

    def _get_name(self):
        return "QuantizedEmbedding(Reference)"

    def forward(
        self,
        input: Tensor,
        offsets: Tensor | None = None,
        per_sample_weights: Tensor | None = None,
    ) -> Tensor:
        weight_quant_dequant = self.get_weight()
        return F.embedding_bag(
            input,
            weight_quant_dequant,
            offsets,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.mode,
            self.sparse,
            per_sample_weights,
            self.include_last_offset,
            self.padding_idx,
        )

    @classmethod
    def from_float(cls, mod, weight_qparams, use_precomputed_fake_quant=False):
        return cls(
            mod.num_embeddings,
            mod.embedding_dim,
            mod.max_norm,
            mod.norm_type,
            mod.scale_grad_by_freq,
            mod.mode,
            mod.sparse,
            mod.weight,
            mod.include_last_offset,
            mod.padding_idx,
            mod.weight.device,
            mod.weight.dtype,
            weight_qparams,
        )
<|endoftext|># mypy: allow-untyped-defs
from typing import Any

import torch
import torch.nn as nn
from torch import _VF, Tensor
from torch.nn.utils.rnn import PackedSequence

from .utils import _quantize_and_dequantize_weight, _quantize_weight


__all__ = [
    "RNNCellBase",
    "RNNCell",
    "LSTMCell",
    "GRUCell",
    "RNNBase",
    "LSTM",
    "GRU",
    "get_quantized_weight",
]


def _apply_permutation(tensor: Tensor, permutation: Tensor, dim: int = 1) -> Tensor:
    return tensor.index_select(dim, permutation)


def _get_weight_and_quantization_params(module, wn):
    weight = getattr(module, wn)
    params = [weight]
    for param_name in [
        wn + n for n in ["_qscheme", "_dtype", "_scale", "_zero_point", "_axis_int"]
    ]:
        if hasattr(module, param_name):
            param = getattr(module, param_name)
        else:
            param = None
        params.append(param)
    return params


def get_quantized_weight(module, wn):
    if not hasattr(module, wn):
        return None
    params = _get_weight_and_quantization_params(module, wn)
    weight = _quantize_weight(*params)
    return weight


def _get_quantize_and_dequantized_weight(module, wn):
    if not hasattr(module, wn):
        return None
    params = _get_weight_and_quantization_params(module, wn)
    weight = _quantize_and_dequantize_weight(*params)
    return weight


class RNNCellBase(nn.RNNCellBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool,
        num_chunks: int,
        device=None,
        dtype=None,
        weight_qparams_dict=None,
    ) -> None:
        super().__init__(
            input_size, hidden_size, bias, num_chunks, device=device, dtype=dtype
        )
        # TODO(jerryzh168): maybe make this arg a required arg
        if weight_qparams_dict is None:
            weight_qparams = {
                "qscheme": torch.per_tensor_affine,
                "dtype": torch.quint8,
                "scale": 1.0,
                "zero_point": 0,
            }
            weight_qparams_dict = {
                "weight_ih": weight_qparams,
                "weight_hh": weight_qparams,
                "is_decomposed": False,
            }
        if len(weight_qparams_dict) != 3:
            raise AssertionError(
                f"Expected length for weight_qparams_dict to be 3 for QuantizedRNNCellBase(Reference), "
                f"got {len(weight_qparams_dict)}"
            )
        self._init_weight_qparams_dict(weight_qparams_dict, device)

    def _init_weight_qparams_dict(self, weight_qparams_dict, device):
        if weight_qparams_dict is None:
            raise AssertionError("weight_qparams_dict must not be None")
        self.is_decomposed = weight_qparams_dict["is_decomposed"]
        for key, weight_qparams in weight_qparams_dict.items():
            if key == "is_decomposed":
                continue
            # TODO: refactor the duplicated code to utils.py
            weight_qscheme = weight_qparams["qscheme"]
            weight_dtype = weight_qparams["dtype"]
            setattr(self, key + "_qscheme", weight_qscheme)
            setattr(self, key + "_dtype", weight_dtype)
            if weight_qscheme not in [
                None,
                torch.per_tensor_affine,
                torch.per_channel_affine,
            ]:
                raise AssertionError(
                    f"qscheme: {weight_qscheme} is not supported in {self._get_name()}"
                )
            if weight_qscheme is not None:
                scale = weight_qparams["scale"]
                scale_tensor = (
                    scale.detach().clone()
                    if isinstance(scale, torch.Tensor)
                    else torch.tensor(scale, dtype=torch.float, device=device)
                )
                self.register_buffer(key + "_scale", scale_tensor)
                zp = weight_qparams["zero_point"]
                zp_tensor = (
                    zp.detach().clone()
                    if isinstance(zp, torch.Tensor)
                    else torch.tensor(zp, dtype=torch.int, device=device)
                )
                self.register_buffer(key + "_zero_point", zp_tensor)
                if weight_qscheme == torch.per_channel_affine:
                    axis = weight_qparams["axis"]
                    axis_tensor = (
                        axis.detach().clone()
                        if isinstance(axis, torch.Tensor)
                        else torch.tensor(axis, dtype=torch.int, device=device)
                    )
                    self.register_buffer(key + "_axis", axis_tensor)
                else:
                    # added for TorchScriptability, not used
                    self.register_buffer(
                        key + "_axis", torch.tensor(0, dtype=torch.int, device=device)
                    )
                setattr(self, key + "_axis_int", getattr(self, key + "_axis").item())

    def _get_name(self):
        return "QuantizedRNNCellBase(Reference)"

    def get_quantized_weight_ih(self):
        return get_quantized_weight(self, "weight_ih")

    def get_quantized_weight_hh(self):
        return get_quantized_weight(self, "weight_hh")

    def get_weight_ih(self):
        return _get_quantize_and_dequantized_weight(self, "weight_ih")

    def get_weight_hh(self):
        return _get_quantize_and_dequantized_weight(self, "weight_hh")


class RNNCell(RNNCellBase):
    """
    We'll store weight_qparams for all the weights (weight_ih and weight_hh),
    we need to pass in a `weight_qparams_dict` that maps from weight name,
    e.g. weight_ih, to the weight_qparams for that weight
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: str = "tanh",
        device=None,
        dtype=None,
        weight_qparams_dict: dict[str, Any] | None = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
            "weight_qparams_dict": weight_qparams_dict,
        }
        super().__init__(input_size, hidden_size, bias, num_chunks=1, **factory_kwargs)
        self.nonlinearity = nonlinearity

    def _get_name(self):
        return "QuantizedRNNCell(Reference)"

    # TODO: refactor nn.RNNCell to have a _forward that takes weight_ih and weight_hh as input
    # and remove duplicated code, same for the other two Cell modules
    def forward(self, input: Tensor, hx: Tensor | None = None) -> Tensor:
        if input.dim() not in (1, 2):
            raise AssertionError(
                f"RNNCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"
            )
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            hx = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
        else:
            hx = hx.unsqueeze(0) if not is_batched else hx

        if self.nonlinearity == "tanh":
            ret = _VF.rnn_tanh_cell(
                input,
                hx,
                self.get_weight_ih(),
                self.get_weight_hh(),
                self.bias_ih,
                self.bias_hh,
            )
        elif self.nonlinearity == "relu":
            ret = _VF.rnn_relu_cell(
                input,
                hx,
                self.get_weight_ih(),
                self.get_weight_hh(),
                self.bias_ih,
                self.bias_hh,
            )
        else:
            ret = input  # TODO: remove when jit supports exception flow
            raise RuntimeError(f"Unknown nonlinearity: {self.nonlinearity}")

        if not is_batched:
            ret = ret.squeeze(0)

        return ret

    @classmethod
    def from_float(cls, mod, weight_qparams_dict):
        ref_mod = cls(
            mod.input_size,
            mod.hidden_size,
            mod.bias,
            mod.nonlinearity,
            mod.weight_ih.device,
            mod.weight_ih.dtype,
            weight_qparams_dict,
        )
        ref_mod.weight_ih = mod.weight_ih
        ref_mod.weight_hh = mod.weight_hh
        ref_mod.bias_ih = mod.bias_ih
        ref_mod.bias_hh = mod.bias_hh
        return ref_mod


class LSTMCell(RNNCellBase):
    """
    We'll store weight_qparams for all the weights (weight_ih and weight_hh),
    we need to pass in a `weight_qparams_dict` that maps from weight name,
    e.g. weight_ih, to the weight_qparams for that weight
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device=None,
        dtype=None,
        weight_qparams_dict: dict[str, Any] | None = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
            "weight_qparams_dict": weight_qparams_dict,
        }
        super().__init__(input_size, hidden_size, bias, num_chunks=4, **factory_kwargs)

    def _get_name(self):
        return "QuantizedLSTMCell(Reference)"

    def forward(
        self, input: Tensor, hx: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, Tensor]:
        if input.dim() not in (1, 2):
            raise AssertionError(
                f"LSTMCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"
            )
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            zeros = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
            hx = (zeros, zeros)
        else:
            hx = (hx[0].unsqueeze(0), hx[1].unsqueeze(0)) if not is_batched else hx

        ret = _VF.lstm_cell(
            input,
            hx,
            self.get_weight_ih(),
            self.get_weight_hh(),
            self.bias_ih,
            self.bias_hh,
        )

        if not is_batched:
            ret = (ret[0].squeeze(0), ret[1].squeeze(0))
        return ret

    @classmethod
    def from_float(cls, mod, weight_qparams_dict, use_precomputed_fake_quant=False):
        ref_mod = cls(
            mod.input_size,
            mod.hidden_size,
            mod.bias,
            mod.weight_ih.device,
            mod.weight_ih.dtype,
            weight_qparams_dict,
        )
        ref_mod.weight_ih = mod.weight_ih
        ref_mod.weight_hh = mod.weight_hh
        ref_mod.bias_ih = mod.bias_ih
        ref_mod.bias_hh = mod.bias_hh
        return ref_mod


class GRUCell(RNNCellBase):
    """
    We'll store weight_qparams for all the weights (weight_ih and weight_hh),
    we need to pass in a `weight_qparams_dict` that maps from weight name,
    e.g. weight_ih, to the weight_qparams for that weight
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device=None,
        dtype=None,
        weight_qparams_dict: dict[str, Any] | None = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
            "weight_qparams_dict": weight_qparams_dict,
        }
        super().__init__(input_size, hidden_size, bias, num_chunks=3, **factory_kwargs)

    def _get_name(self):
        return "QuantizedGRUCell(Reference)"

    def forward(self, input: Tensor, hx: Tensor | None = None) -> Tensor:
        if input.dim() not in (1, 2):
            raise AssertionError(
                f"GRUCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"
            )
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            hx = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
        else:
            hx = hx.unsqueeze(0) if not is_batched else hx

        ret = _VF.gru_cell(
            input,
            hx,
            self.get_weight_ih(),
            self.get_weight_hh(),
            self.bias_ih,
            self.bias_hh,
        )

        if not is_batched:
            ret = ret.squeeze(0)

        return ret

    @classmethod
    def from_float(cls, mod, weight_qparams_dict):
        ref_mod = cls(
            mod.input_size,
            mod.hidden_size,
            mod.bias,
            mod.weight_ih.device,
            mod.weight_ih.dtype,
            weight_qparams_dict,
        )
        ref_mod.weight_ih = mod.weight_ih
        ref_mod.weight_hh = mod.weight_hh
        ref_mod.bias_ih = mod.bias_ih
        ref_mod.bias_hh = mod.bias_hh
        return ref_mod


class RNNBase(nn.RNNBase):
    def __init__(
        self,
        mode: str,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
        device=None,
        dtype=None,
        weight_qparams_dict: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            mode,
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            proj_size,
            device,
            dtype,
        )
        # TODO(jerryzh168): maybe make this arg a required arg
        if weight_qparams_dict is None:
            weight_qparams = {
                "qscheme": torch.per_tensor_affine,
                "dtype": torch.quint8,
                "scale": 1.0,
                "zero_point": 0,
            }
            weight_qparams_dict = {"is_decomposed": False}  # type: ignore[dict-item]
            for wn in self._flat_weights_names:
                if wn.startswith("weight"):
                    weight_qparams_dict[wn] = weight_qparams
        self._init_weight_qparams_dict(weight_qparams_dict, device)

    def _init_weight_qparams_dict(self, weight_qparams_dict, device):
        self.is_decomposed = weight_qparams_dict["is_decomposed"]
        for key, weight_qparams in weight_qparams_dict.items():
            if key == "is_decomposed":
                continue
            weight_qscheme = weight_qparams["qscheme"]
            weight_dtype = weight_qparams["dtype"]
            setattr(self, key + "_qscheme", weight_qscheme)
            setattr(self, key + "_dtype", weight_dtype)
            if weight_qscheme not in [
                None,
                torch.per_tensor_affine,
                torch.per_channel_affine,
            ]:
                raise AssertionError(
                    f"qscheme: {weight_qscheme} is not supported in {self._get_name()}"
                )
            if weight_qscheme is not None:
                self.register_buffer(
                    key + "_scale",
                    torch.tensor(
                        weight_qparams["scale"], dtype=torch.float, device=device
                    ),
                )
                self.register_buffer(
                    key + "_zero_point",
                    torch.tensor(
                        weight_qparams["zero_point"], dtype=torch.int, device=device
                    ),
                )
                if weight_qscheme == torch.per_channel_affine:
                    self.register_buffer(
                        key + "_axis",
                        torch.tensor(
                            weight_qparams["axis"], dtype=torch.int, device=device
                        ),
                    )
                else:
                    # added for TorchScriptability, not used
                    self.register_buffer(
                        key + "_axis", torch.tensor(0, dtype=torch.int, device=device)
                    )
                setattr(self, key + "_axis_int", getattr(self, key + "_axis").item())


class LSTM(RNNBase):
    """Reference Quantized LSTM Module
    We'll store weight_qparams for all the weights in _flat_weights, we need to pass in
    a `weight_qparams_dict` that maps from weight name, e.g. weight_ih_l0,
    to the weight_qparams for that weight
    """

    def __init__(self, *args, **kwargs):
        super().__init__("LSTM", *args, **kwargs)

    # Same as above, see torch/nn/modules/module.py::_forward_unimplemented
    def permute_hidden(  # type: ignore[override]
        self,
        hx: tuple[Tensor, Tensor],
        permutation: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if permutation is None:
            return hx
        return _apply_permutation(hx[0], permutation), _apply_permutation(
            hx[1], permutation
        )

    def get_expected_cell_size(
        self, input: Tensor, batch_sizes: Tensor | None
    ) -> tuple[int, int, int]:
        if batch_sizes is not None:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (
            self.num_layers * num_directions,
            mini_batch,
            self.hidden_size,
        )
        return expected_hidden_size

    # In the future, we should prevent mypy from applying contravariance rules here.
    # See torch/nn/modules/module.py::_forward_unimplemented
    def check_forward_args(  # type: ignore[override]
        self,
        input: Tensor,
        hidden: tuple[Tensor, Tensor],
        batch_sizes: Tensor | None,
    ):
        self.check_input(input, batch_sizes)
        self.check_hidden_size(
            hidden[0],
            self.get_expected_hidden_size(input, batch_sizes),
            "Expected hidden[0] size {}, got {}",
        )
        self.check_hidden_size(
            hidden[1],
            self.get_expected_cell_size(input, batch_sizes),
            "Expected hidden[1] size {}, got {}",
        )

    def get_quantized_weight_bias_dict(self):
        """dictionary from flat_weight_name to quantized weight or (unquantized) bias
        e.g.
        {
          "weight_ih_l0": quantized_weight,
          "bias_ih_l0": unquantized_bias,
          ...
        }
        """
        quantized_weight_bias_dict = {}
        for wn in self._flat_weights_names:
            if hasattr(self, wn):
                if wn.startswith("weight"):
                    weight_or_bias = get_quantized_weight(self, wn)
                else:
                    weight_or_bias = getattr(self, wn)
            else:
                weight_or_bias = None
            quantized_weight_bias_dict[wn] = weight_or_bias
        return quantized_weight_bias_dict

    def get_flat_weights(self):
        flat_weights = []
        for wn in self._flat_weights_names:
            if hasattr(self, wn):
                weight = getattr(self, wn)
                if wn.startswith("weight"):
                    params = _get_weight_and_quantization_params(self, wn)
                    weight = _quantize_and_dequantize_weight(*params)
            else:
                weight = None
            flat_weights.append(weight)
        return flat_weights

    def forward(self, input, hx=None):
        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        batch_sizes = None
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            is_batched = input.dim() == 3
            batch_dim = 0 if self.batch_first else 1
            if not is_batched:
                input = input.unsqueeze(batch_dim)
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            real_hidden_size = (
                self.proj_size if self.proj_size > 0 else self.hidden_size
            )
            h_zeros = torch.zeros(
                self.num_layers * num_directions,
                max_batch_size,
                real_hidden_size,
                dtype=input.dtype,
                device=input.device,
            )
            c_zeros = torch.zeros(
                self.num_layers * num_directions,
                max_batch_size,
                self.hidden_size,
                dtype=input.dtype,
                device=input.device,
            )
            hx = (h_zeros, c_zeros)
        else:
            if batch_sizes is None:  # If not PackedSequence input.
                if is_batched:  # type: ignore[possibly-undefined]
                    if hx[0].dim() != 3 or hx[1].dim() != 3:
                        msg = (
                            "For batched 3-D input, hx and cx should "
                            f"also be 3-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors"
                        )
                        raise RuntimeError(msg)
                else:
                    if hx[0].dim() != 2 or hx[1].dim() != 2:
                        msg = (
                            "For unbatched 2-D input, hx and cx should "
                            f"also be 2-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors"
                        )
                        raise RuntimeError(msg)
                    hx = (hx[0].unsqueeze(1), hx[1].unsqueeze(1))

            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        if batch_sizes is None:
            result = _VF.lstm(
                input,
                hx,
                self.get_flat_weights(),
                self.bias,
                self.num_layers,
                self.dropout,
                self.training,
                self.bidirectional,
                self.batch_first,
            )
        else:
            result = _VF.lstm(
                input,
                batch_sizes,
                hx,
                self.get_flat_weights(),
                self.bias,
                self.num_layers,
                self.dropout,
                self.training,
                self.bidirectional,
            )
        output = result[0]
        hidden = result[1:]
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(
                output,
                batch_sizes,
                sorted_indices,
                unsorted_indices,
            )
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            if not is_batched:  # type: ignore[possibly-undefined]
                output = output.squeeze(batch_dim)  # type: ignore[possibly-undefined]
                hidden = (hidden[0].squeeze(1), hidden[1].squeeze(1))
            return output, self.permute_hidden(hidden, unsorted_indices)

    def _get_name(self):
        return "QuantizedLSTM(Reference)"

    @classmethod
    def from_float(cls, mod, weight_qparams_dict):
        ref_mod = cls(
            mod.input_size,
            mod.hidden_size,
            mod.num_layers,
            mod.bias,
            mod.batch_first,
            mod.dropout,
            mod.bidirectional,
            weight_qparams_dict=weight_qparams_dict,
        )
        for wn in mod._flat_weights_names:
            setattr(ref_mod, wn, getattr(mod, wn))
        return ref_mod


class GRU(RNNBase):
    """Reference Quantized GRU Module
    We'll store weight_qparams for all the weights in _flat_weights, we need to pass in
    a `weight_qparams_dict` that maps from weight name, e.g. weight_ih_l0,
    to the weight_qparams for that weight
    """

    def __init__(self, *args, **kwargs):
        if "proj_size" in kwargs:
            raise ValueError(
                "proj_size argument is only supported for LSTM, not RNN or GRU"
            )
        super().__init__("GRU", *args, **kwargs)

    def get_quantized_weight_bias_dict(self):
        """dictionary from flat_weight_name to quantized weight or (unquantized) bias
        e.g.
        {
          "weight_ih_l0": quantized_weight,
          "bias_ih_l0": unquantized_bias,
          ...
        }
        """
        quantized_weight_bias_dict = {}
        for wn in self._flat_weights_names:
            if hasattr(self, wn):
                if wn.startswith("weight"):
                    weight_or_bias = get_quantized_weight(self, wn)
                else:
                    weight_or_bias = getattr(self, wn)
            else:
                weight_or_bias = None
            quantized_weight_bias_dict[wn] = weight_or_bias
        return quantized_weight_bias_dict

    def get_flat_weights(self):
        flat_weights = []
        for wn in self._flat_weights_names:
            if hasattr(self, wn):
                weight = getattr(self, wn)
                if wn.startswith("weight"):
                    params = _get_weight_and_quantization_params(self, wn)
                    weight = _quantize_and_dequantize_weight(*params)
            else:
                weight = None
            flat_weights.append(weight)
        return flat_weights

    def forward(self, input, hx=None):
        # Note: this is copied from the forward of GRU in https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py
        # only changed self._flat_weights to self.get_flat_weights()
        # TODO: maybe we can try inheriting from that class and define get_flat_weights
        # as a @property? this might interfere with TorchScript, if we remove that
        # requirement in the future we should be able to do this
        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            if input.dim() not in (2, 3):
                raise AssertionError(
                    f"GRU: Expected input to be 2-D or 3-D but received {input.dim()}-D tensor"
                )
            is_batched = input.dim() == 3
            batch_dim = 0 if self.batch_first else 1
            if not is_batched:
                input = input.unsqueeze(batch_dim)
                if hx is not None:
                    if hx.dim() != 2:
                        raise RuntimeError(
                            f"For unbatched 2-D input, hx should also be 2-D but got {hx.dim()}-D tensor"
                        )
                    hx = hx.unsqueeze(1)
            else:
                if hx is not None and hx.dim() != 3:
                    raise RuntimeError(
                        f"For batched 3-D input, hx should also be 3-D but got {hx.dim()}-D tensor"
                    )
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.zeros(
                self.num_layers * num_directions,
                max_batch_size,
                self.hidden_size,
                dtype=input.dtype,
                device=input.device,
            )
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        if batch_sizes is None:
            result = _VF.gru(
                input,
                hx,
                self.get_flat_weights(),
                self.bias,
                self.num_layers,
                self.dropout,
                self.training,
                self.bidirectional,
                self.batch_first,
            )
        else:
            result = _VF.gru(
                input,
                batch_sizes,
                hx,
                self.get_flat_weights(),
                self.bias,
                self.num_layers,
                self.dropout,
                self.training,
                self.bidirectional,
            )
        output = result[0]
        hidden = result[1]

        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(
                output,
                batch_sizes,
                sorted_indices,
                unsorted_indices,
            )
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            if not is_batched:  # type: ignore[possibly-undefined]
                output = output.squeeze(batch_dim)  # type: ignore[possibly-undefined]
                hidden = hidden.squeeze(1)

            return output, self.permute_hidden(hidden, unsorted_indices)

    def _get_name(self):
        return "QuantizedGRU(Reference)"

    @classmethod
    def from_float(cls, mod, weight_qparams_dict):
        ref_mod = cls(
            mod.input_size,
            mod.hidden_size,
            mod.num_layers,
            mod.bias,
            mod.batch_first,
            mod.dropout,
            mod.bidirectional,
            weight_qparams_dict=weight_qparams_dict,
        )
        for wn in mod._flat_weights_names:
            setattr(ref_mod, wn, getattr(mod, wn))
        return ref_mod
<|endoftext|>from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import ReferenceQuantizedModule


__all__ = ["Linear"]


class Linear(nn.Linear, ReferenceQuantizedModule):
    """A reference quantized linear module that fits into the FX
    Graph Mode Quantization workflow
    activation will be floating point Tensor, we will store floating
    point weight as well in the module, but in forward we'll quantize
    and dequantize the weight before running the floating point functional
    linear operator.
    """

    _IS_REFERENCE = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias_: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        weight_qparams: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(in_features, out_features, bias_, device, dtype)
        self._init_weight_qparams(weight_qparams, device)

    def _get_name(self) -> str:
        return "QuantizedLinear(Reference)"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        we have:
        w(float) -- quant - dequant \
        x(float) ------------- F.linear ---

        In the full model, we will see
        w(float) -- quant - *dequant \
        x -- quant --- *dequant --  *F.linear --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized linear
        """
        weight_quant_dequant = self.get_weight()
        result = F.linear(x, weight_quant_dequant, self.bias)
        return result

    @classmethod
    def from_float(
        cls, float_linear: nn.Linear, weight_qparams: dict[str, Any]
    ) -> "Linear":
        qref_linear = Linear(
            float_linear.in_features,
            float_linear.out_features,
            float_linear.bias is not None,
            device=float_linear.weight.device,
            dtype=float_linear.weight.dtype,
            weight_qparams=weight_qparams,
        )
        qref_linear.weight = torch.nn.Parameter(float_linear.weight.detach())
        if float_linear.bias is not None:
            qref_linear.bias = torch.nn.Parameter(float_linear.bias.detach())
        return qref_linear
<|endoftext|># mypy: allow-untyped-defs
r"""Dynamically quantized convolution modules."""

import warnings
from typing import ClassVar, Literal

import torch
import torch.ao.nn.quantized as nnq
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch._ops import ops
from torch.ao.nn.quantized.modules.conv import _reverse_repeat_padding
from torch.nn.common_types import _size_1_t
from torch.nn.modules.utils import _pair, _single, _triple


__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
]


class Conv1d(nnq.Conv1d):
    r"""A dynamically quantized conv module with floating point tensors as inputs and outputs.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv1d` and :class:`~torch.ao.nn.quantized.dynamic.Conv1d` and

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv1d` for other attributes.

    Examples::

        >>> # xdoctest: +SKIP
        >>> m = nn.quantized.dynamic.Conv1d(16, 33, 3, stride=2)
        >>> input = torch.randn(20, 16, 100)
        >>> output = m(input)

    """

    _FLOAT_MODULE: ClassVar[type[nn.Conv1d]] = nn.Conv1d
    _NNIQAT_CONV_BN_MODULE: ClassVar[type[nn.Module] | None] = None
    _NNI_CONV_RELU_MODULE: ClassVar[type[nn.Module] | None] = None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        device=None,
        dtype=None,
        reduce_range=True,
    ):
        warnings.warn(
            f"The current implementation of the {self._get_name()} module has poor numerical accuracy and its use is not recommended",
            stacklevel=2,
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _single(kernel_size)
        stride = _single(stride)

        padding = padding if isinstance(padding, str) else _single(padding)
        dilation = _single(dilation)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _get_name(self):
        return "DynamicQuantizedConv1d"

    def forward(self, input: Tensor, reduce_range: bool = True) -> Tensor:
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 3:
            raise ValueError("Input shape must be `(N, C, L)`!")
        if self.padding_mode != "zeros":
            # Padding in Conv1d is stored as (p, p), need to get (p,)
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding[:1])
            input = F.pad(
                input, _reversed_padding_repeated_twice, mode=self.padding_mode
            )
        return ops.quantized.conv1d_dynamic(input, self._packed_params, reduce_range)


class Conv2d(nnq.Conv2d):
    r"""A dynamically quantized conv module with floating point tensors as inputs and outputs.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv2d` and :class:`~torch.ao.nn.quantized.dynamic.Conv2d` and

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv2d` for other attributes.

    Examples::

        >>> # xdoctest: +SKIP
        >>> # With square kernels and equal stride
        >>> m = nn.quantized.dynamic.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.quantized.dynamic.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.quantized.dynamic.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)

    """

    _FLOAT_MODULE: ClassVar[type[nn.Conv2d]] = nn.Conv2d
    _NNIQAT_CONV_BN_MODULE: ClassVar[type[nn.Module] | None] = None
    _NNI_CONV_RELU_MODULE: ClassVar[type[nn.Module] | None] = None

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        warnings.warn(
            f"The current implementation of the {self._get_name()} module "
            "has poor numerical accuracy and its use is not recommended",
            stacklevel=2,
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _get_name(self):
        return "DynamicQuantizedConv2d"

    def forward(self, input: Tensor, reduce_range: bool = True) -> Tensor:
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        if self.padding_mode != "zeros":
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
            input = F.pad(
                input, _reversed_padding_repeated_twice, mode=self.padding_mode
            )
        return ops.quantized.conv2d_dynamic(input, self._packed_params, reduce_range)


class Conv3d(nnq.Conv3d):
    r"""A dynamically quantized conv module with floating point tensors as inputs and outputs.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv3d` and :class:`~torch.ao.nn.quantized.dynamic.Conv3d` and

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv3d` for other attributes.

    Examples::

        >>> # xdoctest: +SKIP
        >>> # With square kernels and equal stride
        >>> m = nn.quantized.dynamic.Conv3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.quantized.dynamic.Conv3d(16, 33, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.quantized.dynamic.Conv3d(16, 33, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), dilation=(1, 2, 2))
        >>> input = torch.randn(20, 16, 56, 56, 56)
        >>> output = m(input)

    """

    _FLOAT_MODULE: ClassVar[type[nn.Conv3d]] = nn.Conv3d
    _NNIQAT_CONV_BN_MODULE: ClassVar[type[nn.Module] | None] = None
    _NNI_CONV_RELU_MODULE: ClassVar[type[nn.Module] | None] = None

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        warnings.warn(
            f"The current implementation of the {self._get_name()} module has poor numerical accuracy and its use is not recommended",
            stacklevel=2,
        )
        if padding_mode == "reflect":
            raise AssertionError("Conv3d does not support reflection padding")
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        super()._init(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _triple(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _get_name(self):
        return "DynamicQuantizedConv3d"

    def forward(self, input: Tensor, reduce_range: bool = True) -> Tensor:
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 5:
            raise ValueError("Input shape must be `(N, C, D, H, W)`!")
        if self.padding_mode != "zeros":
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
            input = F.pad(
                input, _reversed_padding_repeated_twice, mode=self.padding_mode
            )
        return ops.quantized.conv3d_dynamic(input, self._packed_params, reduce_range)


class ConvTranspose1d(nnq.ConvTranspose1d):
    r"""A dynamically quantized transposed convolution module with floating point tensors as inputs and outputs.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.ConvTranspose1d`.

    For special notes, please, see :class:`~torch.ao.nn.quantized.dynamic.Conv1d`

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point
    See :class:`~torch.nn.ConvTranspose1d` for other attributes.

    Examples::

        >>> # xdoctest: +SKIP
        >>> # With square kernels and equal stride
        >>> m = nndq.ConvTranspose1d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nndq.ConvTranspose1d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> output = m(input)
        >>> # exact output size can be also specified as an argument
        >>> downsample = nndq.Conv1d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nndq.ConvTranspose1d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(input)
        >>> h.size()
        torch.Size([1, 16, 6])
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12])
    """

    _FLOAT_MODULE: ClassVar[type[nn.ConvTranspose1d]] = nn.ConvTranspose1d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        warnings.warn(
            f"The current implementation of the {self._get_name()} module has poor numerical accuracy and its use is not recommended",
            stacklevel=2,
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
            **factory_kwargs,
        )

    def _get_name(self):
        return "DynamicQuantizedConvTranspose1d"

    def forward(self, input: Tensor, reduce_range: bool = True) -> Tensor:
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 3:
            raise ValueError("Input shape must be `(N, C, L)`!")
        return torch.ops.quantized.conv_transpose1d_dynamic(
            input, self._packed_params, reduce_range
        )


class ConvTranspose2d(nnq.ConvTranspose2d):
    r"""A dynamically quantized transposed convolution module with floating point tensors as inputs and outputs.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.ConvTranspose2d`.

    For special notes, please, see :class:`~torch.ao.nn.quantized.dynamic.Conv2d`

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point
    See :class:`~torch.nn.ConvTranspose2d` for other attributes.

    Examples::

        >>> # xdoctest: +SKIP
        >>> # With square kernels and equal stride
        >>> m = nnq.ConvTranspose2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nnq.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> output = m(input)
        >>> # exact output size can be also specified as an argument
        >>> downsample = nnq.Conv2d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nnq.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(input)
        >>> h.size()
        torch.Size([1, 16, 6, 6])
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12])
    """

    _FLOAT_MODULE: ClassVar[type[nn.ConvTranspose2d]] = nn.ConvTranspose2d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        warnings.warn(
            f"The current implementation of the {self._get_name()} module has poor numerical accuracy and its use is not recommended",
            stacklevel=2,
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
            **factory_kwargs,
        )

    def _get_name(self):
        return "DynamicQuantizedConvTranspose2d"

    def forward(self, input: Tensor, reduce_range: bool = True) -> Tensor:
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        return ops.quantized.conv_transpose2d_dynamic(
            input, self._packed_params, reduce_range
        )


class ConvTranspose3d(nnq.ConvTranspose3d):
    r"""A dynamically quantized transposed convolution module with floating point tensors as inputs and outputs.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.ConvTranspose3d`.

    For special notes, please, see :class:`~torch.ao.nn.quantized.dynamic.Conv3d`

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point
    See :class:`~torch.nn.ConvTranspose3d` for other attributes.

    Examples::

        >>> # xdoctest: +SKIP
        >>> # With cubic kernels and equal stride
        >>> m = nnq.ConvTranspose3d(16, 33, 3, stride=2)
        >>> # non-cubic kernels and unequal stride and with padding
        >>> m = nnq.ConvTranspose3d(16, 33, (3, 3, 5), stride=(2, 1, 1), padding=(4, 2, 2))
        >>> output = m(input)
        >>> # exact output size can be also specified as an argument
        >>> downsample = nnq.Conv3d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nnq.ConvTranspose3d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(input)
        >>> h.size()
        torch.Size([1, 16, 6, 6, 6])
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12, 12])
    """

    _FLOAT_MODULE: ClassVar[type[nn.ConvTranspose3d]] = nn.ConvTranspose3d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        warnings.warn(
            f"The current implementation of the {self._get_name()} module has poor numerical accuracy and its use is not recommended",
            stacklevel=2,
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
            **factory_kwargs,
        )

    def _get_name(self):
        return "DynamicQuantizedConvTranspose3d"

    def forward(self, input: Tensor, reduce_range: bool = True) -> Tensor:
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 5:
            raise ValueError("Input shape must be `(N, C, T, H, W)`!")
        return ops.quantized.conv_transpose3d_dynamic(
            input, self._packed_params, reduce_range
        )
<|endoftext|>from .conv import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)
from .linear import Linear
from .rnn import GRU, GRUCell, LSTM, LSTMCell, RNNCell


__all__ = [
    "Linear",
    "LSTM",
    "GRU",
    "LSTMCell",
    "RNNCell",
    "GRUCell",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
]
<|endoftext|># mypy: allow-untyped-defs
import numbers
import warnings
from typing_extensions import deprecated

import torch
import torch.nn as nn
from torch import Tensor
from torch._jit_internal import Dict, List, Optional, Tuple, Union  # noqa: F401
from torch.ao.nn.quantized.modules.utils import _quantize_weight
from torch.nn.utils.rnn import PackedSequence


__all__ = [
    "pack_weight_bias",
    "PackedParameter",
    "RNNBase",
    "LSTM",
    "GRU",
    "RNNCellBase",
    "RNNCell",
    "LSTMCell",
    "GRUCell",
    "apply_permutation",
]


def _apply_permutation(tensor: Tensor, permutation: Tensor, dim: int = 1) -> Tensor:
    return tensor.index_select(dim, permutation)


@deprecated(
    "`apply_permutation` is deprecated, please use `tensor.index_select(dim, permutation)` instead",
    category=FutureWarning,
)
def apply_permutation(tensor: Tensor, permutation: Tensor, dim: int = 1) -> Tensor:
    return _apply_permutation(tensor, permutation, dim)


def pack_weight_bias(qweight, bias, dtype):
    if dtype == torch.qint8:
        # for each layer, for each direction we need to quantize and pack
        # weights and pack parameters in this order:
        #
        #   w_ih, w_hh
        packed_weight = torch.ops.quantized.linear_prepack(qweight, bias)

        return packed_weight
    else:
        # for each layer, for each direction we need to quantize and pack
        # weights and pack parameters in this order:
        #
        #   packed_ih, packed_hh, b_ih, b_hh
        packed_weight = torch.ops.quantized.linear_prepack_fp16(qweight, bias)

        return packed_weight


class PackedParameter(torch.nn.Module):
    def __init__(self, param):
        super().__init__()
        self.param = param

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "param"] = self.param

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
        self.param = state_dict[prefix + "param"]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            False,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class RNNBase(torch.nn.Module):
    _FLOAT_MODULE = nn.RNNBase

    _version = 2

    def __init__(
        self,
        mode,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0.0,
        bidirectional=False,
        dtype=torch.qint8,
    ):
        super().__init__()

        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.dtype = dtype
        self.version = 2
        self.training = False
        num_directions = 2 if bidirectional else 1

        # "type: ignore" is required since ints and Numbers are not fully comparable
        # https://github.com/python/mypy/issues/8566
        if (
            not isinstance(dropout, numbers.Number)
            or not 0 <= dropout <= 1  # type: ignore[operator]
            or isinstance(dropout, bool)
        ):
            raise ValueError(
                "dropout should be a number in range [0, 1] "
                "representing the probability of an element being "
                "zeroed"
            )
        if dropout > 0 and num_layers == 1:  # type: ignore[operator]
            warnings.warn(
                "dropout option adds dropout after all but last "
                "recurrent layer, so non-zero dropout expects "
                f"num_layers greater than 1, but got dropout={dropout} and "
                f"num_layers={num_layers}",
                stacklevel=2,
            )

        if mode == "LSTM":
            gate_size = 4 * hidden_size
        elif mode == "GRU":
            gate_size = 3 * hidden_size
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)

        _all_weight_values = []
        for layer in range(num_layers):
            for _ in range(num_directions):
                layer_input_size = (
                    input_size if layer == 0 else hidden_size * num_directions
                )

                w_ih = torch.randn(gate_size, layer_input_size).to(torch.float)
                w_hh = torch.randn(gate_size, hidden_size).to(torch.float)
                b_ih = torch.randn(gate_size).to(torch.float)
                b_hh = torch.randn(gate_size).to(torch.float)
                if dtype == torch.qint8:
                    w_ih = torch.quantize_per_tensor(
                        w_ih, scale=0.1, zero_point=0, dtype=torch.qint8
                    )
                    w_hh = torch.quantize_per_tensor(
                        w_hh, scale=0.1, zero_point=0, dtype=torch.qint8
                    )
                    packed_ih = torch.ops.quantized.linear_prepack(w_ih, b_ih)
                    packed_hh = torch.ops.quantized.linear_prepack(w_hh, b_hh)
                    # pyrefly: ignore [unnecessary-comparison]
                    if self.version is None or self.version < 2:
                        cell_params = (
                            torch.ops.quantized.make_quantized_cell_params_dynamic(
                                packed_ih, packed_hh, b_ih, b_hh
                            )
                        )
                    else:
                        cell_params = (
                            torch.ops.quantized.make_quantized_cell_params_dynamic(
                                packed_ih, packed_hh, b_ih, b_hh, True
                            )
                        )
                else:
                    packed_ih = torch.ops.quantized.linear_prepack_fp16(w_ih, b_ih)
                    packed_hh = torch.ops.quantized.linear_prepack_fp16(w_hh, b_hh)
                    cell_params = torch.ops.quantized.make_quantized_cell_params_fp16(
                        packed_ih, packed_hh
                    )

                _all_weight_values.append(PackedParameter(cell_params))
        self._all_weight_values = torch.nn.ModuleList(_all_weight_values)

    def _get_name(self):
