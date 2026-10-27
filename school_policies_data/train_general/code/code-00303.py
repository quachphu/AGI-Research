            cx_list.append(torch.jit._unwrap_optional(c))
        hx_tensor = torch.stack(hx_list)
        cx_tensor = torch.stack(cx_list)

        # We are creating another dimension for bidirectional case
        # need to collapse it
        hx_tensor = hx_tensor.reshape(-1, hx_tensor.shape[-2], hx_tensor.shape[-1])
        cx_tensor = cx_tensor.reshape(-1, cx_tensor.shape[-2], cx_tensor.shape[-1])

        if self.batch_first:
            x = x.transpose(0, 1)

        return x, (hx_tensor, cx_tensor)

    def _get_name(self):
        return "QuantizableLSTM"

    @classmethod
    def from_float(cls, other, qconfig=None, split_gates=False):
        if not isinstance(other, cls._FLOAT_MODULE):
            raise AssertionError(
                f"Expected module type {cls._FLOAT_MODULE}, got {type(other)}"
            )
        if not hasattr(other, "qconfig") and not qconfig:
            raise AssertionError("other must have qconfig or qconfig must be provided")
        observed = cls(
            other.input_size,
            other.hidden_size,
            other.num_layers,
            other.bias,
            other.batch_first,
            other.dropout,
            other.bidirectional,
            split_gates=split_gates,
        )
        # pyrefly: ignore [bad-argument-type]
        observed.qconfig = getattr(other, "qconfig", qconfig)
        for idx in range(other.num_layers):
            observed.layers[idx] = _LSTMLayer.from_float(
                other, idx, qconfig, batch_first=False, split_gates=split_gates
            )

        # Prepare the model
        if other.training:
            observed.train()
            observed = torch.ao.quantization.prepare_qat(observed, inplace=True)
        else:
            observed.eval()
            observed = torch.ao.quantization.prepare(observed, inplace=True)
        return observed

    @classmethod
    def from_observed(cls, other):
        # The whole flow is float -> observed -> quantized
        # This class does float -> observed only
        raise NotImplementedError(
            "It looks like you are trying to convert a "
            "non-quantizable LSTM module. Please, see "
            "the examples on quantizable LSTMs."
        )
<|endoftext|># mypy: allow-untyped-defs
from typing import ClassVar, Literal

import torch
import torch.nn as nn
from torch.ao.nn.intrinsic import _FusedModule
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _pair, _single, _triple


__all__ = ["Conv1d", "Conv2d", "Conv3d"]


class _ConvNd(nn.modules.conv._ConvNd):
    _FLOAT_MODULE: ClassVar[type[nn.modules.conv._ConvNd]]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        padding: str | tuple[int, ...],
        dilation: tuple[int, ...],
        transposed: bool,
        output_padding: tuple[int, ...],
        groups: int,
        bias: bool,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"],
        qconfig=None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        nn.modules.conv._ConvNd.__init__(
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
            **factory_kwargs,
        )
        if not qconfig:
            raise AssertionError("qconfig must be provided for QAT module")
        self.qconfig = qconfig
        self.weight_fake_quant = qconfig.weight(factory_kwargs=factory_kwargs)

    def forward(self, input):
        return self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias)

    @staticmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Create a qat module from a float module

        Args:
           `mod`: a float module, either produced by torch.ao.quantization utilities
           or directly from user
        """
        if type(mod) is not cls._FLOAT_MODULE:
            raise AssertionError(
                f"qat.{cls.__name__}.from_float only works for "
                f"{cls._FLOAT_MODULE.__name__}, got {type(mod).__name__}"
            )
        if not hasattr(mod, "qconfig"):
            raise AssertionError("Input float module must have qconfig defined")
        if not mod.qconfig:
            raise AssertionError("Input float module must have a valid qconfig")
        if issubclass(type(mod), _FusedModule):
            mod = mod[0]
        qconfig = mod.qconfig
        qat_conv = cls(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
            groups=mod.groups,
            bias=mod.bias is not None,
            padding_mode=mod.padding_mode,
            qconfig=qconfig,
        )
        qat_conv.weight = mod.weight
        qat_conv.bias = mod.bias
        return qat_conv

    def to_float(self):
        """This works for both single qat conv, and the qat conv - relu modules
        to convert the qat module to a floating point module
        """
        cls = type(self)
        conv = cls._FLOAT_CONV_MODULE(  # type: ignore[attr-defined]
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.bias is not None,
            self.padding_mode,
        )
        conv.weight = torch.nn.Parameter(self.weight.detach())
        if self.bias is not None:
            conv.bias = torch.nn.Parameter(self.bias.detach())
        # conv relu
        if issubclass(cls, _FusedModule):
            modules = [conv]
            if not hasattr(cls, "_FLOAT_RELU_MODULE"):
                raise AssertionError(
                    f"{cls.__name__} must have _FLOAT_RELU_MODULE attribute"
                )
            relu = cls._FLOAT_RELU_MODULE()
            modules.append(relu)

            # pyrefly: ignore [missing-attribute]
            fused = cls._FLOAT_MODULE(*modules)
            fused.train(self.training)
            return fused
        else:
            return conv


class Conv1d(_ConvNd, nn.Conv1d):
    r"""
    A Conv1d module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as :class:`~torch.nn.Conv1d`

    Similar to :class:`~torch.nn.Conv2d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """

    _FLOAT_MODULE: ClassVar[type[nn.Conv1d]] = nn.Conv1d
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv1d]] = nn.Conv1d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: str | _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        qconfig=None,
        device=None,
        dtype=None,
    ) -> None:
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = padding if isinstance(padding, str) else _single(padding)
        dilation_ = _single(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride=stride_,
            padding=padding_,
            dilation=dilation_,
            transposed=False,
            output_padding=_single(0),
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        return super().from_float(
            cls, mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )


class Conv2d(_ConvNd, nn.Conv2d):
    r"""
    A Conv2d module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.Conv2d`, please see
    https://pytorch.org/docs/stable/nn.html?highlight=conv2d#torch.nn.Conv2d
    for documentation.

    Similar to `torch.nn.Conv2d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """

    _FLOAT_MODULE: ClassVar[type[nn.Conv2d]] = nn.Conv2d
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv2d]] = nn.Conv2d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: str | _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        qconfig=None,
        device=None,
        dtype=None,
    ) -> None:
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride=stride_,
            padding=padding_,
            dilation=dilation_,
            transposed=False,
            output_padding=_pair(0),
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
            device=device,
            dtype=dtype,
        )

    def forward(self, input):
        return self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        return super().from_float(
            cls, mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )


class Conv3d(_ConvNd, nn.Conv3d):
    r"""
    A Conv3d module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.Conv3d`, please see
    https://pytorch.org/docs/stable/nn.html?highlight=conv3d#torch.nn.Conv3d
    for documentation.

    Similar to `torch.nn.Conv3d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """

    _FLOAT_MODULE: ClassVar[type[nn.Conv3d]] = nn.Conv3d
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv3d]] = nn.Conv3d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: str | _size_3_t = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        qconfig=None,
        device=None,
        dtype=None,
    ) -> None:
        kernel_size_ = _triple(kernel_size)
        stride_ = _triple(stride)
        padding_ = padding if isinstance(padding, str) else _triple(padding)
        dilation_ = _triple(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride=stride_,
            padding=padding_,
            dilation=dilation_,
            transposed=False,
            output_padding=_triple(0),
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
            device=device,
            dtype=dtype,
        )

    def forward(self, input):
        return self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        return super().from_float(
            cls, mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )
<|endoftext|>from .conv import Conv1d, Conv2d, Conv3d
from .embedding_ops import Embedding, EmbeddingBag
from .linear import Linear


__all__ = [
    "Linear",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "Embedding",
    "EmbeddingBag",
]
<|endoftext|># mypy: allow-untyped-defs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


__all__ = ["Embedding", "EmbeddingBag"]


class Embedding(nn.Embedding):
    r"""
    An embedding bag module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.Embedding`, please see
    https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding
    for documentation.

    Similar to `torch.nn.Embedding`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight: fake quant module for weight
    """

    _FLOAT_MODULE = nn.Embedding

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        device=None,
        dtype=None,
        qconfig=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
            **factory_kwargs,
        )
        if not qconfig:
            raise AssertionError("qconfig must be provided for QAT module")
        weight_qscheme = qconfig.weight().qscheme
        if weight_qscheme != torch.per_channel_affine_float_qparams:
            raise AssertionError(
                "Embedding weights requires a qscheme of torch.per_channel_affine_float_qparams Got "
                + str(weight_qscheme)
            )
        self.qconfig = qconfig
        self.weight_fake_quant = qconfig.weight(factory_kwargs=factory_kwargs)

    def forward(self, input) -> Tensor:
        return F.embedding(
            input,
            self.weight_fake_quant(self.weight),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Create a qat module from a float module

        Args: `mod` a float module, either produced by torch.ao.quantization utilities
        or directly from user
        """
        if type(mod) is not cls._FLOAT_MODULE:
            raise AssertionError(
                " qat."
                + cls.__name__
                + ".from_float only works for "
                + cls._FLOAT_MODULE.__name__
            )
        if not hasattr(mod, "qconfig"):
            raise AssertionError("Input float module must have qconfig defined")
        if not mod.qconfig:
            raise AssertionError("Input float module must have a valid qconfig")
        weight_qscheme = mod.qconfig.weight().qscheme  # type: ignore[union-attr, operator]
        if weight_qscheme != torch.per_channel_affine_float_qparams:
            raise AssertionError(
                "Embedding weights requires a qscheme of torch.per_channel_affine_float_qparams Got "
                + str(weight_qscheme)
            )

        qconfig = mod.qconfig
        qat_embedding_bag = cls(
            mod.num_embeddings,
            mod.embedding_dim,
            mod.padding_idx,
            mod.max_norm,
            mod.norm_type,
            mod.scale_grad_by_freq,
            mod.sparse,
            mod.weight,
            qconfig=qconfig,
        )

        return qat_embedding_bag

    def to_float(self):
        embedding_bag = torch.nn.Embedding(
            self.num_embeddings,
            self.embedding_dim,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
            None,
        )
        embedding_bag.weight = torch.nn.Parameter(self.weight.detach())
        embedding_bag.train(self.training)
        return embedding_bag


class EmbeddingBag(nn.EmbeddingBag):
    r"""
    An embedding bag module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.EmbeddingBag`, please see
    https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag
    for documentation.

    Similar to `torch.nn.EmbeddingBag`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight: fake quant module for weight
    """

    _FLOAT_MODULE = nn.EmbeddingBag

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        mode="mean",
        sparse=False,
        _weight=None,
        include_last_offset=False,
        padding_idx=None,
        qconfig=None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
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
            **factory_kwargs,
        )
        if not qconfig:
            raise AssertionError("qconfig must be provided for QAT module")
        weight_qscheme = qconfig.weight().qscheme
        if weight_qscheme != torch.per_channel_affine_float_qparams:
            raise AssertionError(
                "Embedding Bag weights requires a qscheme of torch.per_channel_affine_float_qparams Got "
                + str(weight_qscheme)
            )
        self.qconfig = qconfig
        self.weight_fake_quant = qconfig.weight(factory_kwargs=factory_kwargs)

    def forward(self, input, offsets=None, per_sample_weights=None) -> Tensor:
        return F.embedding_bag(
            input,
            self.weight_fake_quant(self.weight),
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
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Create a qat module from a float module

        Args: `mod` a float module, either produced by torch.ao.quantization utilities
        or directly from user
        """
        if type(mod) is not cls._FLOAT_MODULE:
            raise AssertionError(
                " qat."
                + cls.__name__
                + ".from_float only works for "
                + cls._FLOAT_MODULE.__name__
            )
        if not hasattr(mod, "qconfig"):
            raise AssertionError("Input float module must have qconfig defined")
        if not mod.qconfig:
            raise AssertionError("Input float module must have a valid qconfig")
        weight_qscheme = mod.qconfig.weight().qscheme  # type: ignore[union-attr, operator]
        if weight_qscheme != torch.per_channel_affine_float_qparams:
            raise AssertionError(
                "Embedding Bag weights requires a qscheme of torch.per_channel_affine_float_qparams Got "
                + str(weight_qscheme)
            )

        qconfig = mod.qconfig
        qat_embedding_bag = cls(
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
            qconfig=qconfig,
        )

        return qat_embedding_bag

    def to_float(self):
        embedding_bag = torch.nn.EmbeddingBag(
            self.num_embeddings,
            self.embedding_dim,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.mode,
            self.sparse,
            None,
            self.include_last_offset,
            self.padding_idx,
        )
        embedding_bag.weight = torch.nn.Parameter(self.weight.detach())
        embedding_bag.train(self.training)
        return embedding_bag
<|endoftext|># mypy: allow-untyped-defs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.nn.intrinsic import LinearReLU
from torch.nn.utils.parametrize import (
    is_parametrized,
    transfer_parametrizations_and_params,
    type_before_parametrizations,
)


__all__ = ["Linear"]


class Linear(nn.Linear):
    r"""
    A linear module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.Linear`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
    for documentation.

    Similar to `torch.nn.Linear`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight: fake quant module for weight
    """

    _FLOAT_MODULE = nn.Linear

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        qconfig=None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(in_features, out_features, bias, **factory_kwargs)
        if not qconfig:
            raise AssertionError("qconfig must be provided for QAT module")
        self.qconfig = qconfig
        self.weight_fake_quant = qconfig.weight(factory_kwargs=factory_kwargs)

    def forward(self, input):
        return F.linear(input, self.weight_fake_quant(self.weight), self.bias)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Create a qat module from a float module or qparams_dict
        Args: `mod` a float module, either produced by torch.ao.quantization utilities
        or directly from user
        """
        if type_before_parametrizations(mod) != cls._FLOAT_MODULE:
            raise AssertionError(
                f"qat.{cls.__name__}.from_float only works for "
                f"{cls._FLOAT_MODULE.__name__}, got {type_before_parametrizations(mod).__name__}"
            )
        if not hasattr(mod, "qconfig"):
            raise AssertionError("Input float module must have qconfig defined")
        if not mod.qconfig:
            raise AssertionError("Input float module must have a valid qconfig")
        if type_before_parametrizations(mod) == LinearReLU:
            mod = mod[0]

        qconfig = mod.qconfig
        qat_linear = cls(
            mod.in_features,
            mod.out_features,
            bias=mod.bias is not None,
            qconfig=qconfig,
        )

        if is_parametrized(mod, "weight"):
            transfer_parametrizations_and_params(mod, qat_linear, "weight")
        else:
            qat_linear.weight = mod.weight

        if is_parametrized(mod, "bias"):
            transfer_parametrizations_and_params(mod, qat_linear, "bias")
        else:
            qat_linear.bias = mod.bias

        return qat_linear

    def to_float(self):
        linear = torch.nn.Linear(
            self.in_features, self.out_features, self.bias is not None
        )
        linear.weight = torch.nn.Parameter(self.weight.detach())
        if self.bias is not None:
            linear.bias = torch.nn.Parameter(self.bias.detach())
        linear.train(self.training)
        return linear
<|endoftext|>from typing import Optional, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from torch.ao.quantization.qconfig import QConfig


__all__ = ["Linear"]


class Linear(torch.ao.nn.qat.Linear):
    r"""
    A linear module attached with FakeQuantize modules for weight,
    used for dynamic quantization aware training.

    We adopt the same interface as `torch.nn.Linear`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
    for documentation.

    Similar to `torch.nn.Linear`, with FakeQuantize modules initialized to
    default.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        qconfig: Optional["QConfig"] = None,
        device: int | str | torch.device | None = None,
        dtype: str | None = None,
    ) -> None:
        super().__init__(in_features, out_features, bias, qconfig, device, dtype)
        if not torch.ao.quantization.qconfig._activation_is_memoryless(qconfig):  # type: ignore[arg-type]
            raise ValueError(
                "Dynamic QAT requires a memoryless observer."
                + "This means a MovingAverage observer with averaging constant equal to 1"
            )
<|endoftext|># Variables
from ._mappings import (
    get_dynamic_sparse_quantized_mapping,
    get_static_sparse_quantized_mapping,
)

# Scheduler
from .scheduler.base_scheduler import BaseScheduler
from .scheduler.cubic_scheduler import CubicSL
from .scheduler.lambda_scheduler import LambdaSL

# Sparsifier
from .sparsifier.base_sparsifier import BaseSparsifier
from .sparsifier.nearly_diagonal_sparsifier import NearlyDiagonalSparsifier

# Parametrizations
from .sparsifier.utils import (
    FakeSparsity,
    fqn_to_module,
    get_arg_info_from_tensor_fqn,
    module_to_fqn,
)
from .sparsifier.weight_norm_sparsifier import WeightNormSparsifier
<|endoftext|># mypy: allow-untyped-defs
__all__ = [
    "get_static_sparse_quantized_mapping",
    "get_dynamic_sparse_quantized_mapping",
]


def get_static_sparse_quantized_mapping():
    import torch.ao.nn.sparse

    _static_sparse_quantized_mapping = {
        torch.nn.Linear: torch.ao.nn.sparse.quantized.Linear,
    }
    return _static_sparse_quantized_mapping


def get_dynamic_sparse_quantized_mapping():
    import torch.ao.nn.sparse

    _dynamic_sparse_quantized_mapping = {
        torch.nn.Linear: torch.ao.nn.sparse.quantized.dynamic.Linear,
    }
    return _dynamic_sparse_quantized_mapping
<|endoftext|># mypy: allow-untyped-defs
import torch

from . import base_sparsifier


class NearlyDiagonalSparsifier(base_sparsifier.BaseSparsifier):
    r"""Nearly Diagonal Sparsifier

    This sparsifier creates a nearly diagonal mask to be applied to the weight matrix.
    Nearly Diagonal Matrix is a matrix that contains non-zero elements near the diagonal and the rest are zero.
    An example of a nearly diagonal matrix with degree (or nearliness) 3 and 5 are follows respectively.
    1 1 0 0       1 1 1 0
    1 1 1 0       1 1 1 1
    0 1 1 1       1 1 1 1
    0 0 1 1       0 1 1 1
    Note that a nearly diagonal matrix with degree 1 is just a matrix with main diagonal populated

    This sparsifier is controlled by one variable:
    1. `nearliness` defines the number of non-zero diagonal lines that are closest to the main diagonal.
        Currently - supports only odd number

    Note:
        This can be accelerated (vectorized) once the Spdiagonal feature (PR: #78439) is landed or the banded matrix
        feature is landed: https://stackoverflow.com/questions/52463972/generating-banded-matrices-using-numpy

    Args:
        nearliness: The degree of nearliness (default = 1)

    """

    def __init__(self, nearliness: int = 1):
        defaults = {"nearliness": nearliness}
        super().__init__(defaults=defaults)

    def update_mask(  # type:ignore[override]
        self, module, tensor_name, nearliness, **kwargs
    ):
        mask = getattr(module.parametrizations, tensor_name)[0].mask
        mask.data = torch.zeros_like(mask)
        if nearliness <= 0:
            return

        tensor = getattr(module, tensor_name)
        height, width = tensor.shape

        if nearliness % 2 == 0:
            raise ValueError("nearliness can only be an odd number")
        dist_to_diagonal = nearliness // 2
        # check
        if dist_to_diagonal >= min(height, width):
            raise ValueError(
                "nearliness cannot be larger than the dimensions of tensor."
            )

        for row in range(height):
            # Bounds of entries that needs to be set to 1
            low = max(0, row - dist_to_diagonal)
            high = min(width, row + dist_to_diagonal + 1)
            mask[row, low:high].fill_(1)
<|endoftext|># mypy: allow-untyped-defs
from itertools import chain
from typing import Any

from torch import nn
from torch.nn.utils.parametrize import is_parametrized, type_before_parametrizations


__all__ = [
    "module_contains_param",
    "swap_module",
    "module_to_fqn",
    "fqn_to_module",
    "get_arg_info_from_tensor_fqn",
    "FakeSparsity",
]


def module_contains_param(module: nn.Module, parametrization: type[nn.Module]) -> bool:
    if is_parametrized(module):
        # see if any of the module tensors have a parametrization attached that matches the one passed in
        return any(
            any(isinstance(param, parametrization) for param in param_list)
            for param_list in module.parametrizations.values()  # type: ignore[union-attr,operator]
        )
    return False


def swap_module(
    mod: nn.Module, mapping: dict[type[nn.Module], type[nn.Module]]
) -> nn.Module:
    r"""Swaps the module using from_dense according to the mapping passed in.
    Args:
        mod: input module
        mapping: a dictionary that maps from nn module to sparse nn module
    Return:
        The corresponding sparse module of `mod` according to mapping, created using from_dense
    """
    if type_before_parametrizations(mod) in mapping:
        sparse_mod = mapping[type_before_parametrizations(mod)]

        # TODO Fix this typing, as Type[Module] has no attribute "from_dense"
        new_mod = sparse_mod.from_dense(mod)  # type: ignore[attr-defined]

        # Preserve module's pre forward hooks. They'll be called on quantized input
        for pre_hook_fn in mod._forward_pre_hooks.values():
            new_mod.register_forward_pre_hook(pre_hook_fn)
        # Preserve module's post forward hooks except _observer_forward_hook
        # After convert they'll work with quantized output
        for hook_fn in mod._forward_hooks.values():
            new_mod.register_forward_hook(hook_fn)

        # respect device affinity when swapping modules
        # pyrefly: ignore [bad-argument-type]
        devices = {p.device for p in chain(mod.parameters(), mod.buffers())}
        if len(devices) > 1:
            raise AssertionError(
                f"swap_module only works with cpu or single-device CUDA modules, but got devices {devices}"
            )
        device = next(iter(devices)) if len(devices) > 0 else None
        if device:
            new_mod.to(device)

        return new_mod

    else:
        return mod


def module_to_fqn(model: nn.Module, module: nn.Module, prefix: str = "") -> str | None:
    """
    Returns the fqn for a module or None if module not a descendant of model.
    """
    if module is model:
        return ""
    for name, child in model.named_children():
        fqn = module_to_fqn(child, module, ".")
        if isinstance(fqn, str):
            return prefix + name + fqn
    return None


def fqn_to_module(model: nn.Module | None, path: str) -> nn.Module | None:
    """
    Given an fqn, returns the corresponding module or tensor or None if the fqn given by `path`
    doesn't correspond to anything. Similar to model.get_submodule(path) but works for tensors.
    """
    if path != "":
        for name in path.split("."):
            model = getattr(model, name, None)
    return model


def get_arg_info_from_tensor_fqn(model: nn.Module, tensor_fqn: str) -> dict[str, Any]:
    """
    Uses tensor_fqn to obtain a dict containing module_fqn, module and tensor_name
    """
    # string manip to split tensor_fqn into module_fqn and tensor_name
    # if tensor_fqn is 'weight' then module_fqn and tensor_name are '' and 'weight'
    # if tensor_fqn is 'linear.weight' then module_fqn and tensor_name are 'linear' and 'weight'
    tensor_name = tensor_fqn.rsplit(".", maxsplit=1)[-1]
    module_fqn = tensor_fqn[: -len(tensor_name) - ("." in tensor_fqn)]

    module = fqn_to_module(model, module_fqn)

    return {
        "module_fqn": module_fqn,
        "module": module,
        "tensor_name": tensor_name,
        "tensor_fqn": tensor_fqn,
    }


# Parametrizations
class FakeSparsity(nn.Module):
    r"""Parametrization for the weights. Should be attached to the 'weight' or
    any other parameter that requires a mask applied to it.

    Note::

        Once the mask is passed, the variable should not change the id. The
        contents of the mask can change, but the mask reference itself should
        not.
    """

    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, x):
        if self.mask.shape != x.shape:
            raise AssertionError(
                f"mask shape ({self.mask.shape}) must match x shape ({x.shape})"
            )
        return self.mask * x

    def state_dict(self, *args, **kwargs):
        # We don't want to let the parametrizations to save the mask.
        # That way we make sure that the linear module doesn't store the masks
        # alongside their parametrizations.
        return {}
<|endoftext|># mypy: allow-untyped-defs
import operator
from collections.abc import Callable
from functools import reduce

import torch
import torch.nn.functional as F

from .base_sparsifier import BaseSparsifier


__all__ = ["WeightNormSparsifier"]


def _flat_idx_to_2d(idx, shape):
    rows = idx // shape[1]
    cols = idx % shape[1]
    return rows, cols


class WeightNormSparsifier(BaseSparsifier):
    r"""Weight-Norm Sparsifier

    This sparsifier computes the norm of every sparse block and "zeroes-out" the
    ones with the lowest norm. The level of sparsity defines how many of the
    blocks is removed.

    This sparsifier is controlled by three variables:
    1. `sparsity_level` defines the number of *sparse blocks* that are zeroed-out
    2. `sparse_block_shape` defines the shape of the sparse blocks. Note that
        the sparse blocks originate at the zero-index of the tensor.
    3. `zeros_per_block` is the number of zeros that we are expecting in each
        sparse block. By default we assume that all elements within a block are
        zeroed-out. However, setting this variable sets the target number of
        zeros per block. The zeros within each block are chosen as the *smallest
        absolute values*.

    Args:

        sparsity_level: The target level of sparsity
        sparse_block_shape: The shape of a sparse block (see note below)
        zeros_per_block: Number of zeros in a sparse block
        norm: Norm to use. Could be either `int` or a callable.
            If `int`, only L1 and L2 are implemented.

    Note::
        The `sparse_block_shape` is tuple representing (block_ROWS, block_COLS),
        irrespective of what the rows / cols mean in the data tensor. That means,
        if you were to sparsify a weight tensor in the nn.Linear, which has a
        weight shape `(Cout, Cin)`, the `block_ROWS` would refer to the output
        channels, while the `block_COLS` would refer to the input channels.

    Note::
        All arguments to the WeightNormSparsifier constructor are "default"
        arguments and could be overridden by the configuration provided in the
        `prepare` step.
    """

    def __init__(
        self,
        sparsity_level: float = 0.5,
        sparse_block_shape: tuple[int, int] = (1, 4),
        zeros_per_block: int | None = None,
        norm: Callable | int | None = None,
    ):
        if zeros_per_block is None:
            zeros_per_block = reduce(operator.mul, sparse_block_shape)
        defaults = {
            "sparsity_level": sparsity_level,
            "sparse_block_shape": sparse_block_shape,
            "zeros_per_block": zeros_per_block,
        }
        if norm is None:
            norm = 2
        if callable(norm):
            self.norm_fn = norm
        elif norm == 1:
            self.norm_fn = lambda T: T.abs()
        elif norm == 2:
            self.norm_fn = lambda T: T * T
        else:
            raise NotImplementedError(f"L-{norm} is not yet implemented.")
        super().__init__(defaults=defaults)

    def _scatter_fold_block_mask(
        self,
        output_shape,
        dim,
        indices,
        block_shape,
        mask=None,
        input_shape=None,
        device=None,
    ):
        r"""Creates patches of size `block_shape` after scattering the indices."""
        if mask is None:
            if input_shape is None:
                raise AssertionError("input_shape must be provided when mask is None")
            mask = torch.ones(input_shape, device=device)
        mask.scatter_(dim=dim, index=indices, value=0)
        mask.data = F.fold(
            mask, output_size=output_shape, kernel_size=block_shape, stride=block_shape
        )
        return mask

    def _make_tensor_mask(
        self, data, input_shape, sparsity_level, sparse_block_shape, mask=None
    ):
        r"""Creates a tensor-level mask.

        Tensor-level mask is described as a mask, where the granularity of sparsification of the
        smallest patch is the sparse_block_shape. That means, that for a given mask and a
        sparse_block_shape, the smallest "patch" of zeros/ones could be the sparse_block_shape.

        In this context, `sparsity_level` describes the fraction of sparse patches.
        """
        h, w = data.shape[-2:]
        block_h, block_w = sparse_block_shape
        dh = (block_h - h % block_h) % block_h
        dw = (block_w - w % block_w) % block_w

        if mask is None:
            mask = torch.ones(h + dh, w + dw, device=data.device)

        if sparsity_level >= 1.0:
            mask.data = torch.zeros_like(mask)
            return mask
        elif sparsity_level <= 0.0:
            mask.data = torch.ones_like(mask)
            return mask

        values_per_block = reduce(operator.mul, sparse_block_shape)
        if values_per_block > 1:
            # Reduce the data
            data = F.avg_pool2d(
                data[None, None, :],
                kernel_size=sparse_block_shape,
                stride=sparse_block_shape,
                ceil_mode=True,
            )
        data = data.flatten()
        num_blocks = len(data)

        data = data.repeat(1, values_per_block, 1)

        threshold_idx = round(sparsity_level * num_blocks)
        threshold_idx = max(0, min(num_blocks - 1, threshold_idx))  # Sanity check
        _, sorted_idx = torch.topk(data, k=threshold_idx, dim=2, largest=False)

        # Temp reshape for mask
        mask_reshape = mask.reshape(data.shape)  # data might be reshaped
        self._scatter_fold_block_mask(
            dim=2,
            output_shape=(h + dh, w + dw),
            indices=sorted_idx,
            block_shape=sparse_block_shape,
            mask=mask_reshape,
        )
        mask.data = mask_reshape.squeeze().reshape(mask.shape)[:h, :w].contiguous()
        return mask

    def _make_block_mask(self, data, sparse_block_shape, zeros_per_block, mask=None):
        r"""Creates a block-level mask.

        Block-level mask is described as a mask, where the granularity of sparsification of the
        largest patch is the sparse_block_shape. That means that for a given mask and a
        sparse_block_shape, the sparsity is computed only within a patch of a size sparse_block_shape.

        In this context the `zeros_per_block` describes the number of zeroed-out elements within a patch.
        """
        h, w = data.shape[-2:]
        block_h, block_w = sparse_block_shape
        dh = (block_h - h % block_h) % block_h
        dw = (block_w - w % block_w) % block_w
        values_per_block = reduce(operator.mul, sparse_block_shape)

        if mask is None:
            mask = torch.ones((h + dh, w + dw), device=data.device)

        if values_per_block == zeros_per_block:
            # Everything should be sparsified
            mask.data = torch.zeros_like(mask)
            return mask

        # create a new padded tensor like data (to match the block_shape)
        padded_data = torch.ones(h + dh, w + dw, dtype=data.dtype, device=data.device)
        padded_data.fill_(torch.nan)
        padded_data[:h, :w] = data
        unfolded_data = F.unfold(
            padded_data[None, None, :],
            kernel_size=sparse_block_shape,
            stride=sparse_block_shape,
        )

        # Temp reshape for mask
        mask_reshape = mask.reshape(unfolded_data.shape)
        _, sorted_idx = torch.topk(
            unfolded_data, k=zeros_per_block, dim=1, largest=False
        )

        self._scatter_fold_block_mask(
            dim=1,
            indices=sorted_idx,
            output_shape=padded_data.shape,
            block_shape=sparse_block_shape,
            mask=mask_reshape,
        )

        mask.data = mask_reshape.squeeze().reshape(mask.shape).contiguous()
        return mask

    def update_mask(  # type: ignore[call-override, override]
        self,
        module,
        tensor_name,
        sparsity_level,
        sparse_block_shape,
        zeros_per_block,
        **kwargs,
    ):
        values_per_block = reduce(operator.mul, sparse_block_shape)
        if zeros_per_block > values_per_block:
            raise ValueError(
                "Number of zeros per block cannot be more than the total number of elements in that block."
            )
        if zeros_per_block < 0:
            raise ValueError("Number of zeros per block should be positive.")

        mask = getattr(module.parametrizations, tensor_name)[0].mask
        if sparsity_level <= 0 or zeros_per_block == 0:
            mask.data = torch.ones_like(mask)
        elif sparsity_level >= 1.0 and (zeros_per_block == values_per_block):
            mask.data = torch.zeros_like(mask)
        else:
            ww = self.norm_fn(getattr(module, tensor_name))
            tensor_mask = self._make_tensor_mask(
                data=ww,
                input_shape=ww.shape,
                sparsity_level=sparsity_level,
                sparse_block_shape=sparse_block_shape,
            )
            if values_per_block != zeros_per_block:
                block_mask = self._make_block_mask(
                    data=ww,
                    sparse_block_shape=sparse_block_shape,
                    zeros_per_block=zeros_per_block,
                )
                tensor_mask = torch.logical_or(tensor_mask, block_mask)
            mask.data = tensor_mask
<|endoftext|># mypy: allow-untyped-defs
import abc
import copy
from collections import defaultdict
from typing import Any

import torch
from torch import nn
from torch.nn.utils import parametrize
from torch.nn.utils.parametrize import type_before_parametrizations

from .utils import (
    FakeSparsity,
    get_arg_info_from_tensor_fqn,
    module_contains_param,
    module_to_fqn,
    swap_module,
)


__all__ = ["BaseSparsifier"]

SUPPORTED_MODULES = {nn.Linear}

KEYS_NOT_IN_STATE_DICT = ["module", "module_fqn", "tensor_name"]


# TODO update desc with new config args
class BaseSparsifier(abc.ABC):
    r"""Base class for all sparsifiers.

    Abstract methods that need to be implemented:

    - update_mask: Function to compute a new mask for all keys in the
        `groups`.

    Args:
        - model [nn.Module]: model to configure. The model itself is not saved
            but used for the state_dict saving / loading.
        - config [list]: configuration elements should be a dict map that includes
            `tensor_fqn` of tensors to sparsify
        - defaults [dict]: default configurations will be attached to the
            configuration. Only the keys that don't exist in the `config` will
            be updated.

    Example::

        >>> # xdoctest: +SKIP("Can't instantiate abstract class BaseSparsifier with abstract method update_mask")
        >>> config = [{'tensor_fqn': 'layer1.weight', 'tensor_fqn': 'linear2.weight2', 'sparsity_level': 0.5}]
        >>> defaults = {'sparsity_level': 0.7}
        >>> # model.layer1.weight will have `sparsity_level` = 0.7 (getting default)
        >>> sparsifier = BaseSparsifier(config, defaults)
    """

    def __init__(self, defaults: dict[str, Any] | None = None):
        super().__init__()
        self.defaults: dict[str, Any] = defaults or {}

        self.state: dict[str, dict] = defaultdict(dict)
        self.groups: list[dict[str, Any]] = []
        self.enable_mask_update = True

    def __getstate__(self) -> dict[str, Any]:
        return {
            "defaults": self.defaults,
            "state": self.state,
            "groups": self.groups,
        }

    def __setstate__(self, state: dict[str, dict[str, Any]]) -> None:
        self.__dict__.update(state)

    def __repr__(self):
        format_string = self.__class__.__name__ + " ("
        for i, sparse_args in enumerate(self.groups):
            module = sparse_args["module"]
            format_string += "\n"
            format_string += f"\tGroup {i}\n"
            format_string += f"\t    module: {module}\n"
            for key in sorted(sparse_args.keys()):
                if key == "module":
                    continue
                format_string += f"\t    {key}: {sparse_args[key]}\n"
        format_string += ")"
        return format_string

    def state_dict(self) -> dict[str, Any]:
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains:
        * state - current state of the sparsification.
        * groups - a list containing all sparsity configuration groups
            with the key 'tensor_fqn' specifying the path to the sparsified tensor within a model

        TODO: Need a clean way of loading the state of the "prepared" module
        """

        groups: list[dict[str, Any]] = [
            dict(
                filter(
                    lambda key_value: key_value[0] not in KEYS_NOT_IN_STATE_DICT,
                    mg.items(),
                )
            )
            for mg in self.groups
        ]

        return {
            "state": self.state,
            "groups": groups,
        }

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True):
        groups = copy.deepcopy(state_dict["groups"])
        states = state_dict["state"]
        for tensor_fqn, s in states.items():
            arg_info = get_arg_info_from_tensor_fqn(self.model, tensor_fqn)
            module = arg_info["module"]
            tensor_name = arg_info["tensor_name"]
            if strict and module is None:
                raise RuntimeError(f"Error loading {tensor_fqn} into the model")

            found = False
            for p in module.parametrizations[tensor_name]:
                if isinstance(p, FakeSparsity):
                    found = True
                    break
            if not found:
                p = FakeSparsity(torch.ones(getattr(module, tensor_name).shape))
                parametrize.register_parametrization(module, tensor_name, p)
            if s.get("mask", None) is not None:
                mask = s.pop("mask")
                p.mask = mask

            for mg in groups:
                if mg["tensor_fqn"] == tensor_fqn:
                    mg.update(arg_info)
        self.__setstate__({"state": states, "groups": groups})

    def make_config_from_model(
        self,
        model: nn.Module,
        SUPPORTED_MODULES: set[type[nn.Linear]] = SUPPORTED_MODULES,
    ) -> None:
        self.config = []
        stack = [model]
        while stack:
            module = stack.pop()
            for _name, child in module.named_children():
                if type(child) in SUPPORTED_MODULES:
                    module_fqn = module_to_fqn(model, child)
                    if not isinstance(module_fqn, str):
                        raise AssertionError("module_fqn must be a string")
                    self.config.append({"tensor_fqn": module_fqn + ".weight"})
                else:
                    stack.append(child)

    def prepare(self, model, config):
        r"""Prepares a model, by adding the parametrizations.

        Note::

            The model is modified inplace. If you need to preserve the original
            model, use copy.deepcopy.
        """
        self.model = model  # TODO: Need to figure out how to load without this.
        self.config = config

        # If no config -- try getting all the supported layers
        if self.config is None:
            self.make_config_from_model(model)

        # TODO: Remove the configuration by reference ('module')

        for module_config in self.config:
            if not isinstance(module_config, dict):
                raise AssertionError(
                    "config elements should be dicts not modules i.e.:"
                    "[{`tensor_fqn`: `foo.bar.weight`}, {`tensor_fqn`: ... }, ...]"
                )

            if not isinstance(self.defaults, dict):
                raise AssertionError("defaults must be a dict")
            local_args = copy.deepcopy(self.defaults)
            local_args.update(module_config)

            tensor_fqn = local_args.get("tensor_fqn", None)
            if tensor_fqn is None:
                raise AssertionError(
                    "tensor_fqn is a required argument in the sparsity config which "
                    "replaces previous `module` and [module]`fqn` arguments"
                )

            # populate all information from tensor_fqn
            info_from_tensor_fqn = get_arg_info_from_tensor_fqn(model, tensor_fqn)

            # check that whatever was put into local_args agrees with what was obtained
            # from tensor_fqn
            for key in info_from_tensor_fqn:
                if key in local_args:
                    if not (
                        info_from_tensor_fqn[key] == local_args[key]
                        or (
                            key == "tensor_fqn"
                            and "." + info_from_tensor_fqn[key] == local_args[key]
                        )
                        # info_from_tensor_fqn will chop leading '.' from tensor_fqn so ignore that
                    ):
                        raise AssertionError(
                            f"Given both `{key}` and `tensor_fqn` in the config, it is expected them to agree!"
                        )
            local_args.update(info_from_tensor_fqn)
            self.groups.append(local_args)
        self._prepare()

    def _prepare(self, *args, **kwargs):
        r"""Adds mask parametrization to the layer weight"""
        for config in self.groups:
            module = config["module"]
            tensor_name = config["tensor_name"]
            parametrization = config.get("parametrization", FakeSparsity)
            mask = config.get("mask", torch.ones_like(getattr(module, tensor_name)))
            self.state[config["tensor_fqn"]]["mask"] = mask
            parametrize.register_parametrization(
                module, tensor_name, parametrization(mask)
            )

    def squash_mask(
        self,
        params_to_keep: tuple[str, ...] | None = None,
        params_to_keep_per_layer: dict[str, tuple[str, ...]] | None = None,
        *args,
        **kwargs,
    ):
        r"""Squashes the sparse masks into the appropriate tensors.

        If either the `params_to_keep` or `params_to_keep_per_layer` is set,
        the module will have a `sparse_params` dict attached to it.

        Args:
            params_to_keep: List of keys to save in the module or a dict
                            representing the modules and keys that will have
                            sparsity parameters saved
            params_to_keep_per_layer: Dict to specify the params that should be
                            saved for specific layers. The keys in the dict
                            should be the module fqn, while the values should
                            be a list of strings with the names of the variables
                            to save in the `sparse_params`

        Examples:
            >>> # xdoctest: +SKIP("locals are undefined")
            >>> # Don't save any sparse params
            >>> sparsifier.squash_mask()
            >>> hasattr(model.submodule1, "sparse_params")
            False

            >>> # Keep sparse params per layer
            >>> sparsifier.squash_mask(
            ...     params_to_keep_per_layer={
            ...         "submodule1.linear1": ("foo", "bar"),
            ...         "submodule2.linear42": ("baz",),
            ...     }
            ... )
            >>> print(model.submodule1.linear1.sparse_params)
            {'foo': 42, 'bar': 24}
            >>> print(model.submodule2.linear42.sparse_params)
            {'baz': 0.1}

            >>> # Keep sparse params for all layers
            >>> sparsifier.squash_mask(params_to_keep=("foo", "bar"))
            >>> print(model.submodule1.linear1.sparse_params)
            {'foo': 42, 'bar': 24}
            >>> print(model.submodule2.linear42.sparse_params)
            {'foo': 42, 'bar': 24}

            >>> # Keep some sparse params for all layers, and specific ones for
            >>> # some other layers
            >>> sparsifier.squash_mask(
            ...     params_to_keep=("foo", "bar"),
            ...     params_to_keep_per_layer={"submodule2.linear42": ("baz",)},
            ... )
            >>> print(model.submodule1.linear1.sparse_params)
            {'foo': 42, 'bar': 24}
            >>> print(model.submodule2.linear42.sparse_params)
            {'foo': 42, 'bar': 24, 'baz': 0.1}
        """
        for config in self.groups:
            module = config["module"]
            tensor_name = config["tensor_name"]
            parametrize.remove_parametrizations(
                module, tensor_name, leave_parametrized=True
            )
            sparse_params = {}
            if params_to_keep is not None:
                global_params = {k: config[k] for k in params_to_keep}
                sparse_params.update(global_params)
            if params_to_keep_per_layer is not None:
                params = params_to_keep_per_layer.get(config["module_fqn"], None)
                if params is not None:
                    per_layer_params = {k: config[k] for k in params}
                    sparse_params.update(per_layer_params)
            if sparse_params:
                # TODO handle multiple tensor being quantized on a single module, where to store sparse_params?
                module.sparse_params = sparse_params

    def convert(
        self,
        module: nn.Module,
        mapping: dict[type[nn.Module], type[nn.Module]] | None = None,
        inplace: bool = False,
        parameterization: type[nn.Module] = FakeSparsity,
    ):
        r"""Converts submodules in input module to a different module according to `mapping`
        by calling `from_dense` method on the target module class
        Args:
            module: input module
            mapping: a dictionary that maps from source module type to target
                module type, can be overwritten to allow swapping user defined
                Modules
            inplace: carry out model transformations in-place, the original module
                is mutated
        """
        if mapping is None:
            raise NotImplementedError("Need to auto generate mapping ")
        if not inplace:
            module = copy.deepcopy(module)

        reassign = {}
        for name, mod in module.named_children():
            # leaf node
            if (
                module_contains_param(mod, parameterization)
                and type_before_parametrizations(mod) in mapping
            ):
                reassign[name] = swap_module(mod, mapping)
            else:
                # recurse
                reassign[name] = self.convert(
                    mod,
                    mapping=mapping,
                    inplace=True,
                    parameterization=parameterization,
                )

        for key, value in reassign.items():
            module._modules[key] = value

        return module

    def step(self, use_path: bool = True) -> None:
        if not self.enable_mask_update:
            return
        with torch.no_grad():
            for config in self.groups:
                self.update_mask(**config)

    @abc.abstractmethod
    def update_mask(self, module: nn.Module, tensor_name: str, **kwargs):
        pass
<|endoftext|># mypy: allow-untyped-defs

import torch
import torch.nn as nn
from torch.ao.pruning.sparsifier.utils import fqn_to_module, module_to_fqn


SUPPORTED_MODULES = {nn.Embedding, nn.EmbeddingBag}


def _fetch_all_embeddings(model):
    """Fetches Embedding and EmbeddingBag modules from the model"""
    embedding_modules = []
    stack = [model]
    while stack:
        module = stack.pop()
        for _, child in module.named_children():
            fqn_name = module_to_fqn(model, child)
            if type(child) in SUPPORTED_MODULES:
                embedding_modules.append((fqn_name, child))
            else:
                stack.append(child)
    return embedding_modules


def post_training_sparse_quantize(
    model,
    data_sparsifier_class,
    sparsify_first=True,
    select_embeddings: list[nn.Module] | None = None,
    **sparse_config,
):
    """Takes in a model and applies sparsification and quantization to only embeddings & embeddingbags.
    The quantization step can happen before or after sparsification depending on the `sparsify_first` argument.

    Args:
        - model (nn.Module)
            model whose embeddings need to be sparsified
        - data_sparsifier_class (type of data sparsifier)
            Type of sparsification that needs to be applied to model
        - sparsify_first (bool)
            if true, sparsifies first and then quantizes
            otherwise, quantizes first and then sparsifies.
        - select_embeddings (List of Embedding modules)
            List of embedding modules in the model to be sparsified & quantized.
            If None, all embedding modules will be sparsified
        - sparse_config (Dict)
            config that will be passed to the constructor of data sparsifier object.

    Note:
        1. When `sparsify_first=False`, quantization occurs first followed by sparsification.
            - before sparsifying, the embedding layers are dequantized.
            - scales and zero-points are saved
            - embedding layers are sparsified and `squash_mask` is applied
            - embedding weights are requantized using the saved scales and zero-points
        2. When `sparsify_first=True`, sparsification occurs first followed by quantization.
            - embeddings are sparsified first
            - quantization is applied on the sparsified embeddings
    """
    data_sparsifier = data_sparsifier_class(**sparse_config)

    # if select_embeddings is None, perform it on all embeddings
    if select_embeddings is None:
        embedding_modules = _fetch_all_embeddings(model)

    else:
        embedding_modules = []
        if not isinstance(select_embeddings, list):
            raise AssertionError(
                "the embedding_modules must be a list of embedding modules"
            )
        for emb in select_embeddings:
            if type(emb) not in SUPPORTED_MODULES:
                raise AssertionError(
                    "the embedding_modules list must be an embedding or embedding bags"
                )
            fqn_name = module_to_fqn(model, emb)
            if fqn_name is None:
                raise AssertionError(
                    "the embedding modules must be part of input model"
                )
            embedding_modules.append((fqn_name, emb))

    if sparsify_first:
        # sparsify
        for name, emb_module in embedding_modules:
            valid_name = name.replace(".", "_")
            data_sparsifier.add_data(name=valid_name, data=emb_module)

        data_sparsifier.step()
        data_sparsifier.squash_mask()

        # quantize
        for _, emb_module in embedding_modules:
            # pyrefly: ignore [bad-argument-type]
            emb_module.qconfig = torch.ao.quantization.float_qparams_weight_only_qconfig

        torch.ao.quantization.prepare(model, inplace=True)
        torch.ao.quantization.convert(model, inplace=True)

    else:
        # quantize
        for _, emb_module in embedding_modules:
            # pyrefly: ignore [bad-argument-type]
            emb_module.qconfig = torch.ao.quantization.float_qparams_weight_only_qconfig

        torch.ao.quantization.prepare(model, inplace=True)
        torch.ao.quantization.convert(model, inplace=True)

        # retrieve scale & zero_points
        quantize_params: dict[str, dict] = {
            "scales": {},
            "zero_points": {},
            "dequant_weights": {},
            "axis": {},
            "dtype": {},
        }

        for name, _ in embedding_modules:
            quantized_emb = fqn_to_module(model, name)
            if quantized_emb is None:
                raise AssertionError(f"quantized embedding {name} not found in model")

            quantized_weight = quantized_emb.weight()  # type: ignore[operator]
            quantize_params["scales"][name] = quantized_weight.q_per_channel_scales()
            quantize_params["zero_points"][name] = (
                quantized_weight.q_per_channel_zero_points()
            )
            quantize_params["dequant_weights"][name] = torch.dequantize(
                quantized_weight
            )
            quantize_params["axis"][name] = quantized_weight.q_per_channel_axis()
            quantize_params["dtype"][name] = quantized_weight.dtype

            # attach data to sparsifier
            data_sparsifier.add_data(
                name=name.replace(".", "_"),
                data=quantize_params["dequant_weights"][name],
            )

        data_sparsifier.step()
        data_sparsifier.squash_mask()

        for name, _ in embedding_modules:
            quantized_emb = fqn_to_module(model, name)
            if quantized_emb is None:
                raise AssertionError(f"quantized embedding {name} not found in model")
            requantized_vector = torch.quantize_per_channel(
                quantize_params["dequant_weights"][name],
                scales=quantize_params["scales"][name],
                zero_points=quantize_params["zero_points"][name],
                dtype=quantize_params["dtype"][name],
                axis=quantize_params["axis"][name],
            )

            quantized_emb.set_weight(requantized_vector)  # type: ignore[operator]
<|endoftext|># mypy: allow-untyped-defs
import operator
from functools import reduce
from typing import Any

import torch
from torch.nn import functional as F

from .base_data_sparsifier import BaseDataSparsifier


__all__ = ["DataNormSparsifier"]


class DataNormSparsifier(BaseDataSparsifier):
    r"""L1-Norm Sparsifier
    This sparsifier computes the *L1-norm* of every sparse block and "zeroes-out" the
    ones with the lowest norm. The level of sparsity defines how many of the
    blocks are removed.
    This sparsifier is controlled by three variables:
    1. `sparsity_level` defines the number of *sparse blocks* that are zeroed-out
    2. `sparse_block_shape` defines the shape of the sparse blocks. Note that
        the sparse blocks originate at the zero-index of the tensor.
    3. `zeros_per_block` is the number of zeros that we are expecting in each
        sparse block. By default we assume that all elements within a block are
        zeroed-out. However, setting this variable sets the target number of
        zeros per block. The zeros within each block are chosen as the *smallest
        absolute values*.
    Args:
        sparsity_level: The target level of sparsity
        sparse_block_shape: The shape of a sparse block
        zeros_per_block: Number of zeros in a sparse block
    Note::
        All arguments to the DataNormSparsifier constructor are "default"
        arguments and could be overridden by the configuration provided in the
        `add_data` step.
    """

    def __init__(
        self,
        data_list: list[tuple[str, Any]] | None = None,
        sparsity_level: float = 0.5,
        sparse_block_shape: tuple[int, int] = (1, 4),
        zeros_per_block: int | None = None,
        norm: str = "L1",
    ):
        if zeros_per_block is None:
            zeros_per_block = reduce(operator.mul, sparse_block_shape)

        if norm not in ["L1", "L2"]:
            raise AssertionError("only L1 and L2 norm supported at the moment")

        defaults = {
            "sparsity_level": sparsity_level,
            "sparse_block_shape": sparse_block_shape,
            "zeros_per_block": zeros_per_block,
        }
        self.norm = norm
        super().__init__(data_list=data_list, **defaults)

    def __get_scatter_folded_mask(
        self, data, dim, indices, output_size, sparse_block_shape
    ):
        mask = torch.ones_like(data)
        mask.scatter_(dim=dim, index=indices, value=0)  # zeroing out
        mask = F.fold(
            mask,
            output_size=output_size,
            kernel_size=sparse_block_shape,
            stride=sparse_block_shape,
        )
        mask = mask.to(torch.int8)
        return mask

    def __get_block_level_mask(self, data, sparse_block_shape, zeros_per_block):
        # Assume data is a squeezed tensor
        height, width = data.shape[-2], data.shape[-1]
        block_height, block_width = sparse_block_shape
        values_per_block = block_height * block_width

        # just return zeros if zeroing all elements in block
        if values_per_block == zeros_per_block:
            return torch.zeros_like(data, dtype=torch.int8)

        # creating additional height and width to support padding
        dh = (block_height - height % block_height) % block_height
        dw = (block_width - width % block_width) % block_width

        # create a new padded tensor like data (to match the block_shape)
        padded_data = torch.ones(
            height + dh, width + dw, dtype=data.dtype, device=data.device
        )
        padded_data = (
            padded_data * torch.nan
        )  # can also be replaced with 0 to stop the removal of edge data
        padded_data[0:height, 0:width] = data
        unfolded_data = F.unfold(
            padded_data[None, None, :],
            kernel_size=sparse_block_shape,
            stride=sparse_block_shape,
        )

        _, sorted_idx = torch.sort(unfolded_data, dim=1)
        sorted_idx = sorted_idx[
            :, :zeros_per_block, :
        ]  # zero out zeros_per_block number of elements

        mask = self.__get_scatter_folded_mask(
            data=unfolded_data,
            dim=1,
            indices=sorted_idx,
            output_size=padded_data.shape,
            sparse_block_shape=sparse_block_shape,
        )

        mask = (
            mask.squeeze(0).squeeze(0)[:height, :width].contiguous()
        )  # remove padding and make contiguous
        return mask

    def __get_data_level_mask(self, data, sparsity_level, sparse_block_shape):
        height, width = data.shape[-2], data.shape[-1]
        block_height, block_width = sparse_block_shape
        dh = (block_height - height % block_height) % block_height
        dw = (block_width - width % block_width) % block_width

        data_norm = F.avg_pool2d(
            data[None, None, :],
            kernel_size=sparse_block_shape,
            stride=sparse_block_shape,
            ceil_mode=True,
        )

        values_per_block = reduce(operator.mul, sparse_block_shape)

        data_norm = data_norm.flatten()
        num_blocks = len(data_norm)

        data_norm = data_norm.repeat(
            1, values_per_block, 1
        )  # get similar shape after unfold
        _, sorted_idx = torch.sort(data_norm, dim=2)

        threshold_idx = round(sparsity_level * num_blocks)  # number of blocks to remove
        sorted_idx = sorted_idx[:, :, :threshold_idx]

        mask = self.__get_scatter_folded_mask(
            data=data_norm,
            dim=2,
            indices=sorted_idx,
            output_size=(height + dh, width + dw),
            sparse_block_shape=sparse_block_shape,
        )

        mask = mask.squeeze(0).squeeze(0)[
            :height, :width
        ]  # squeeze only the first 2 dimension
        return mask

    def update_mask(  # type: ignore[override]
        self, name, data, sparsity_level, sparse_block_shape, zeros_per_block, **kwargs
    ):
        values_per_block = reduce(operator.mul, sparse_block_shape)
        if zeros_per_block > values_per_block:
            raise ValueError(
                "Number of zeros per block cannot be more than "
                "the total number of elements in that block."
            )
        if zeros_per_block < 0:
            raise ValueError("Number of zeros per block should be positive.")

        if self.norm == "L1":
            data_norm = torch.abs(data).squeeze()  # absolute value based (L1)
        else:
            data_norm = (data * data).squeeze()  # square every element for L2

        if len(data_norm.shape) > 2:  # only supports 2 dimensional data at the moment
            raise ValueError("only supports 2-D at the moment")

        elif len(data_norm.shape) == 1:  # in case the data is bias (or 1D)
            data_norm = data_norm[None, :]

        mask = self.get_mask(name)
        if sparsity_level <= 0 or zeros_per_block == 0:
            mask.data = torch.ones_like(mask)
        elif sparsity_level >= 1.0 and (zeros_per_block == values_per_block):
            mask.data = torch.zeros_like(mask)

        # Fetch the high level mask that zeros out entire blocks
        data_lvl_mask = self.__get_data_level_mask(
            data=data_norm,
            sparsity_level=sparsity_level,
            sparse_block_shape=sparse_block_shape,
        )

        # Fetch block level mask that zeros out 'zeros_per_block' number of elements in every block
        block_lvl_mask = self.__get_block_level_mask(
            data=data_norm,
            sparse_block_shape=sparse_block_shape,
            zeros_per_block=zeros_per_block,
        )

        # zero out the entries inside those blocks whose block is sparsified
        mask.data = torch.where(data_lvl_mask == 1, data_lvl_mask, block_lvl_mask)
<|endoftext|># mypy: allow-untyped-defs
import abc
import copy
import sys
import warnings
from collections import defaultdict
from typing import Any

import torch
from torch import nn
from torch.ao.pruning.sparsifier import base_sparsifier, utils
from torch.nn.utils import parametrize


if not sys.warnoptions:
    # to suppress repeated warnings when being used in a training loop.
    warnings.simplefilter("once")

__all__ = ["BaseDataSparsifier"]

EMBEDDING_TYPES = {
    nn.Embedding,
    nn.EmbeddingBag,
}

SUPPORTED_TYPES = {
    torch.Tensor,
    nn.Parameter,
    *EMBEDDING_TYPES,
}


class _Container(nn.Module):
    pass


class BaseDataSparsifier(base_sparsifier.BaseSparsifier):
    r"""
    Base Data Sparsifier class for all Data sparsifiers.
    The abstract class accepts raw torch tensors / embedding / embedding bags (refer to SUPPORTED_TYPES above)
    to prepare for sparsification.
    In this case, mask (and parametrizations) is owned by the class and not by the user.
    Specifically, the container object inside the class maintains the mask and parametrizations of the input data

    Args:
        data_list (list of tuples)
            list of (name, data) tuples to sparsify. Lookup SUPPORTED_TYPES
            for type of data. Internally, a container module handles the data sparsification.

        defaults (dict)
            default configurations will be attached to the
            configuration. Only the keys that don't exist in the `config` will
            be updated.
    Example::
        >>> # xdoctest: +SKIP
        >>> data_list = [('tensor_1', torch.randn(3,3)), ('tensor_2', torch.randn(4,4))]
        >>> defaults = {'sparsity_level': 0.7}
        >>> sparsifier = DerivedDataSparsifier(data_list = data_list, **defaults) # Some sparsifier that inherits BaseDataSparsifier
        >>> new_tensor_to_add = {'name': 'tensor_3', 'data': torch.randn(5,5), 'sparsity_level': 0.3}
        >>> sparsifier.add_data(**new_tensor_to_add)
        >>> # tensor_1 and tensor_2 will have sparsity_level of 0.7 but tensor_3 will have sparsity_level=0.3
    """

    def __init__(self, data_list: list[tuple[str, Any]] | None = None, **defaults):
        super().__init__(defaults=defaults)

        self._container = _Container()

        self.data_groups: dict[str, dict] = defaultdict(dict)  # name -> {**config}
        if data_list is not None:
            # add data with default config here
            [self.add_data(name, data, **self.defaults) for name, data in data_list]

    def prepare(self, model, config):
        raise NotImplementedError("this function is undefined for this class")

    def _extract_weight(self, data):
        # extract the weight parameter instead of underlying data
        if type(data) in [torch.Tensor, nn.Parameter]:
            return data
        elif type(data) in EMBEDDING_TYPES:
            return data.weight

    def add_data(self, name: str, data, reuse_mask=True, **config):
        r"""Configures and parametrizes the internal container model with name and data.

        **Note**:
            1. If the data with name already exists, it replaces the data.
            2. While replacing, the old mask is reused when `reuse_mask=True`
            3. If `reuse_mask=True`, then the replacing data needs to have the same shape as that of old data.
            4. By default, the config of the replaced data is used as config for the replacing data, unless something
               is specified in the config dictionary.
        """
        if type(data) not in SUPPORTED_TYPES:
            raise AssertionError(
                f"specified data type:{type(data)} not supported at the moment"
            )
        local_args = copy.deepcopy(self.defaults)
        local_args.update(config)
        weight = self._extract_weight(data)

        # Bookkeeping in the container class
        mask = local_args.get("mask", torch.ones_like(weight))
        param_class = local_args.get("parametrization", utils.FakeSparsity)

        if name in self.state:
            # If the named data already exists - replace
            warnings.warn(
                "Replacing existing data of the same name. - Did you mean a different name?",
                stacklevel=2,
            )

            # reuse old config
            old_args = self.data_groups[name]
            local_args = copy.deepcopy(old_args)
            local_args.update(config)

            if reuse_mask:
                current_data = self.get_data(name=name)
                if weight.shape != current_data.shape:
                    raise AssertionError(
                        "to retain the old mask, the shape of the new data must be the same as the previous one"
                    )
                mask = self.get_mask(
                    name=name
                )  # reuse mask instead of creating a new one

            self._delete_data(name=name)

        # parameter creates a deepcopy of the weight inside, so create a buffer
        self._container.register_buffer(name=name, tensor=weight)
        parametrize.register_parametrization(self._container, name, param_class(mask))
        self.state[name]["mask"] = mask
        self.data_groups[name] = local_args
        return getattr(self._container, name)

    def get_data(self, name: str, return_original: bool = True):
        r"""Returns weight tensor (or data)
        Args:
            - name: name of the data to be returned
            - return_original returns weight tensor without applying parametrization if True
                else - returns the sparsified version (parametrized)
        """
        if name not in self.data_groups:
            raise ValueError("data with specified name does not exist")

        if return_original:
            if not parametrize.is_parametrized(self._container, name):
                raise ValueError("mask squashed - original mask value does not exist")
            data = getattr(self._container.parametrizations, name).original
            return data
        else:
            return getattr(self._container, name)

    def _convert_mask(self, states, sparse_coo=True):
        r"""Converts the mask to sparse coo or dense tensors depending on the `sparse_coo` argument."""
        states = copy.deepcopy(states)
        for state in states.values():
            if sparse_coo:
                state["mask"] = state["mask"].to_sparse_coo()
            else:
                state["mask"] = state["mask"].to_dense()

        return states

    def state_dict(self):
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains:
        * state - contains name -> mask mapping.
        * data_groups - a list containing all sparsity configuration groups
            with the key name specifying the name of the data
        * container_state_dict - the state dictionary of the internal
            container model used for sparsification
        """
        state = self._convert_mask(self.state)
        return {
            "state": state,
            "data_groups": self.data_groups,
            "_container": self._container.state_dict(),
        }

    def _load_container_from_state(self, states, data_groups, container_state_dict):
        r"""This restores the state of the container specifically based on the data present in state and data_groups
        If the data was parametrized, then the data would be added to the container and then parametrized,
        else it would just add the attribute to the container.
        """
        for name, state in states.items():
            config_name = data_groups.get(name, None)
            if config_name is None:
                raise RuntimeError(f"Error loading {name}")

            # check if the data with such a name was parametrized, if so parametrize
            # otherwise just set the attribute and continue
            parametrized_name = f"parametrizations.{name}.original"
            parametrized = False
            data = container_state_dict.get(name, None)
            if name in container_state_dict:
                # the parametrization was probably removed for this
                data = container_state_dict.get(name)

            elif parametrized_name in container_state_dict:
                # so the weight was parametrized
                data = container_state_dict.get(parametrized_name)
                parametrized = True

            else:
                raise RuntimeError(f"Error loading {name}")

            self._container.register_buffer(name=name, tensor=data)

            if parametrized:
                # register parameter if parametrized
                mask = state.get("mask", torch.ones_like(data))
                param_class = data_groups.get(
                    "parametrization", utils.FakeSparsity
                )  # change once public_api for utils is fixed!
                parametrize.register_parametrization(
                    self._container, name, param_class(mask)
                )

    def load_state_dict(self, state_dict, strict=True):
        r"""The load_state_dict() restores the state of the sparsifier based on the state_dict

        Args:
        * state_dict - the dictionary to which the current sparsifier needs to be restored to
        * strict - If True - the sparsifier is reset and is restored exactly to the state in state_dict.
            If False - the current sparsifier is not reset before loading the state_dict i.e. data added
            before loading the state_dict is not erased.
        """
        states = copy.deepcopy(state_dict["state"])
        data_groups = copy.deepcopy(state_dict["data_groups"])
        container_state_dict = copy.deepcopy(state_dict["_container"])

        states = self._convert_mask(
            states, sparse_coo=False
        )  # convert sparse coo mask to dense
        if strict:
            # if strict load -> then reset container
            self._container = _Container()

        self._load_container_from_state(states, data_groups, container_state_dict)

        if not strict:
            states.update(self.state)
            data_groups.update(self.data_groups)

        self.__setstate__({"state": states, "data_groups": data_groups})

    def __setstate__(self, state):
        if "_container" in state:  # If container object is in state then load model
            container_dict = state.pop("_container")
            self._container = _Container()
            state["state"] = self._convert_mask(
                state["state"], sparse_coo=False
            )  # convert sparse coo mask to dense
            self._load_container_from_state(
                state["state"], state["data_groups"], container_dict
            )

        self.__dict__.update(state)

    def __getstate__(self):
        state = self._convert_mask(self.state)
        return {
            "defaults": self.defaults,
            "state": state,
            "data_groups": self.data_groups,
            "_container": self._container.state_dict(),
        }

    def __repr__(self):  # type:ignore[override]
        format_string = self.__class__.__name__ + " ("
        for name, sparse_args in self.data_groups.items():
            format_string += "\n"
            format_string += "\tData Group\n"
            format_string += f"\t    name: {name}\n"
            for key in sorted(sparse_args.keys()):
                if key == "data":
                    continue
                format_string += f"\t    {key}: {sparse_args[key]}\n"
        format_string += ")"
        return format_string

    def get_mask(self, name: str):
        if name not in self.state:
            raise ValueError("data with specified name does not exist")
        return self.state[name]["mask"]

    def squash_mask(self, *args, leave_parametrized=True, names=None, **kwargs):
        r"""Squashes the sparse masks into the appropriate tensors. Also, accepts list of strings
        to squash mask for. If none, squashes mask for all the keys
        kwargs:
            * names: list of strings to squash mask for
            * sparsified: if true - applies the mask before squashing
                          if false - does not apply the mask before squashing
        """
        if names is None:
            names = list(self.data_groups.keys())
        for name in names:
            parametrize.remove_parametrizations(
                self._container, name, leave_parametrized=leave_parametrized
            )

    def step(self):  # type:ignore[override]
        if not self.enable_mask_update:
            return
        with torch.no_grad():
            for name, config in self.data_groups.items():
                # get non-sparsified data
                data = self.get_data(name)
                # need name for the mask otherwise can directly pass mask?
                self.update_mask(name, data, **config)

    @abc.abstractmethod
    def update_mask(self, name, data, **kwargs):  # type: ignore[override]
        pass

    def _delete_data(self, name):
        """Detaches some data from the sparsifier.

        Args:
            name (str)
                Name of the data to be removed from the sparsifier

        Note:
            Currently private. Kind of used as a helper function when replacing data of the same name
        """
        self.squash_mask(
            names=[name], leave_parametrized=False
        )  # do not apply the mask while deleting
        delattr(self._container, name)
        self.state.pop(name)
        self.data_groups.pop(name)
<|endoftext|># mypy: allow-untyped-defs
from collections import defaultdict
from copy import deepcopy
from typing import Any, TYPE_CHECKING

import pytorch_lightning as pl  # type: ignore[import]

from ._data_sparstity_utils import (
    _attach_model_to_data_sparsifier,
    _get_valid_name,
    _log_sparsified_level,
)


if TYPE_CHECKING:
    import torch


class PostTrainingDataSparsity(pl.callbacks.Callback):
    """Lightning callback that enables post-training sparsity.

    This callback aims to sparsify the model inside lightning module after training.
    **Note that the model is copied and then sparsified, so the existing model is not modified**

    The sparsified model can be used for comparison and can be accessed using
        <callback_obj>.sparsified

    Args:
        data_sparsifier_class (some implemented class of BaseDataSparsifier)
            The data sparsifier object of this class is created when the
            training starts.
            Note: Objects should not be passed in here as they are created
            once the training completes.

        data_sparsifier_args (Dict)
            Dictionary of args to be passed to the data sparsifier.
            Note: data_list arg should be ignored

    Hooks implemented:
        on_fit_end()
            1. copies the model and attaches it to the sparsifier
            2. sparsifier step() is called
            3. squashes the mask()
    """

    def __init__(self, data_sparsifier_class, data_sparsifier_args):
        super().__init__()
        self.data_sparsifier_class = data_sparsifier_class
        self.data_sparsifier_args = data_sparsifier_args
        self.data_sparsifier: Any = None
        self.sparsified: torch.nn.Module | None = None

    def on_fit_end(self, trainer, pl_module) -> None:
        self.sparsified = deepcopy(pl_module.model).eval()
        self.data_sparsifier = self.data_sparsifier_class(**self.data_sparsifier_args)

        _attach_model_to_data_sparsifier(self.sparsified, self.data_sparsifier)

        self.data_sparsifier.step()

        self.data_sparsifier.squash_mask()  # currently squashes params for all mask

        _log_sparsified_level(self.sparsified, self.data_sparsifier)


class TrainingAwareDataSparsity(pl.callbacks.Callback):
    """Lightning callback that enables in-training sparsity.

    This callback aims to sparsify the model inside lightning module during training.
    **Note that the model is copied and then sparsified, so the existing model is not modified**

    The sparsified model can be used for comparison and can be accessed using
        <callback_obj>.sparsified

    Args:
        data_sparsifier_class (some implemented class of BaseDataSparsifier)
            The data sparsifier object of this class is created when the
            training starts.
            Note: Objects should not be passed in here as they are created
            when the training starts.

        data_sparsifier_args (Dict)
            Dictionary of args to be passed to the data sparsifier.
            Note: data_list arg should be ignored

        data_scheduler_class (some implemented class of BaseDataScheduler)
            The data scheduler of this class is created when the training starts
            Note: Objects should not be passed in here as they are created
            when the training starts.

        data_scheduler_args(Dict)
            Dictionary of args to be passed to the data scheduler.
            **Note: data_sparsifier arg should be ignored as the recipe
            creates and pass sparsifier object into the class**

    Hooks implemented:
        on_train_start()
            Data sparsifier and scheduler objects are created.
            Pytorch model attached to the sparsifier

        on_train_epoch_start()
            Loads the state_dict of the data sparsifier

        on_train_epoch_end()
            1. Copies the model and attaches it to the sparsifier
            2. sparsifier step() and scheduler step()
            3. Dump state_dict of the current sparsifier

        on_train_end()
            squash mask
    """

    def __init__(
        self,
        data_sparsifier_class,
        data_sparsifier_args,
        data_scheduler_class,
        data_scheduler_args,
    ):
        super().__init__()
        # data sparsifier objects
        self.data_sparsifier_class = data_sparsifier_class
        self.data_sparsifier_args = data_sparsifier_args

        # scheduler objects
        self.data_scheduler_class = data_scheduler_class
        self.data_scheduler_args = data_scheduler_args

        # fields
        self.data_sparsifier: Any = None
        self.data_scheduler: Any = None
        self.sparsified: torch.nn.Module | None = None

        self.data_sparsifier_state_dict: Any = None

    def on_train_start(self, trainer, pl_module) -> None:
        # create sparsifier
        self.data_sparsifier = self.data_sparsifier_class(**self.data_sparsifier_args)
        self.sparsified = deepcopy(pl_module.model)

        _attach_model_to_data_sparsifier(
            self.sparsified, self.data_sparsifier
        )  # just to populate the base_sl in the scheduler

        # create scheduler
        args = deepcopy(self.data_scheduler_args)
        args["data_sparsifier"] = self.data_sparsifier
        self.data_scheduler = self.data_scheduler_class(**args)

    def on_train_epoch_start(self, trainer, pl_module):
        if self.data_sparsifier_state_dict is None:
            return  # probably first epoch

        # load the existing config for each data
        self.data_sparsifier.load_state_dict(self.data_sparsifier_state_dict)

    def __create_config_based_on_state(self, pl_module):
        config: dict = defaultdict()
        if self.data_sparsifier_state_dict is None:
            return config
        for name, _ in pl_module.model.named_parameters():
            valid_name = _get_valid_name(name)
            config[valid_name] = self.data_sparsifier.data_groups[valid_name]

        return config

    def on_train_epoch_end(self, trainer, pl_module):
        self.sparsified = deepcopy(pl_module.model)
        config = self.__create_config_based_on_state(pl_module)

        # attach model to the data sparsifier
        _attach_model_to_data_sparsifier(
            self.sparsified, self.data_sparsifier, config=config
        )
        self.data_sparsifier.step()
        self.data_scheduler.step()

        self.data_sparsifier_state_dict = self.data_sparsifier.state_dict()

    def on_train_end(self, trainer, pl_module):
        self.data_sparsifier.squash_mask()
<|endoftext|># mypy: allow-untyped-defs
import logging

from torch.ao.pruning._experimental.data_sparsifier.base_data_sparsifier import (
    SUPPORTED_TYPES,
)


logger: logging.Logger = logging.getLogger(__name__)


def _attach_model_to_data_sparsifier(module, data_sparsifier, config=None):
    """Attaches a data sparsifier to all the layers of the module.
    Essentially, loop over all the weight parameters in the module and
    attach it to the data sparsifier.
    Note::
        The '.' in the layer names are replaced with '_' (refer to _get_valid_name() below)
        before attaching to the sparsifier. This is because, the data
        sparsifier uses a dummy model inside to store the weight parameters.
    """
    if config is None:
        config = {}
    for name, parameter in module.named_parameters():
        if type(parameter) in SUPPORTED_TYPES:
            valid_name = _get_valid_name(name)
            # will be defaulted to default configs
            data_sparsifier.add_data(
                name=valid_name, data=parameter, **config.get(valid_name, {})
            )


def _get_valid_name(name):
    return name.replace(".", "_")  # . is not allowed as a name


def _log_sparsified_level(model, data_sparsifier) -> None:
    # Show the level of sparsity AFTER step:
    for name, parameter in model.named_parameters():
        if type(parameter) not in SUPPORTED_TYPES:
            continue
        valid_name = _get_valid_name(name)
        mask = data_sparsifier.get_mask(name=valid_name)
        sparsity_level = 1.0 - mask.float().mean()
        logger.info("Sparsity in layer %s = % .2%", name, sparsity_level)
<|endoftext|># mypy: allow-untyped-defs
from .base_structured_sparsifier import BaseStructuredSparsifier


class SaliencyPruner(BaseStructuredSparsifier):
    """
    Prune rows based on the saliency (L1 norm) of each row.

    This pruner works on N-Dimensional weight tensors.
    For each row, we will calculate the saliency, which is the sum of the L1 norm of all weights in that row.
    We expect that the resulting saliency vector has the same shape as our mask.
    We then pick elements to remove until we reach the target sparsity_level.
    """

    def update_mask(self, module, tensor_name, **kwargs):
        # tensor_name will give you the FQN, all other entries in sparse config is present in kwargs
        weights = getattr(module, tensor_name)
        mask = getattr(module.parametrizations, tensor_name)[0].mask

        # use negative weights so we can use topk (we prune out the smallest)
        if weights.dim() <= 1:
            raise Exception(  # noqa: TRY002
                "Structured pruning can only be applied to a 2+dim weight tensor!"
            )
        saliency = -weights.norm(dim=tuple(range(1, weights.dim())), p=1)
        if saliency.shape != mask.shape:
            raise AssertionError(
                f"saliency shape ({saliency.shape}) must match mask shape ({mask.shape})"
            )

        num_to_pick = int(len(mask) * kwargs["sparsity_level"])
        prune = saliency.topk(num_to_pick).indices

        # Set the mask to be false for the rows we want to prune
        mask.data[prune] = False
<|endoftext|># mypy: allow-untyped-defs
from collections.abc import Callable
from itertools import chain
from operator import getitem

import torch
import torch.nn.functional as F
from torch import nn
from torch.ao.pruning.sparsifier.base_sparsifier import BaseSparsifier
from torch.fx import symbolic_trace
from torch.nn.utils import parametrize

from .match_utils import apply_match, MatchAllNode
from .parametrization import BiasHook, FakeStructuredSparsity, module_contains_param
from .prune_functions import (
    prune_conv2d,
    prune_conv2d_activation_conv2d,
    prune_conv2d_activation_pool_conv2d,
    prune_conv2d_conv2d,
    prune_conv2d_pool_activation_conv2d,
    prune_conv2d_pool_flatten_linear,
    prune_linear,
    prune_linear_activation_linear,
    prune_linear_linear,
    prune_lstm_output_layernorm_linear,
    prune_lstm_output_linear,
)


def _get_supported_structured_pruning_modules():
    SUPPORTED_STRUCTURED_PRUNING_MODULES = {  # added to config if None given
        nn.Linear,
        nn.Conv2d,
        nn.LSTM,
    }
    return SUPPORTED_STRUCTURED_PRUNING_MODULES


def _get_supported_activation_functions():
    SUPPORTED_ACTIVATION_FUNCTIONS = {
        F.relu,
        F.rrelu,
        F.hardtanh,
        F.relu6,
        F.sigmoid,
        F.hardsigmoid,
        F.tanh,
        F.silu,
        F.mish,
        F.hardswish,
        F.elu,
        F.celu,
        F.selu,
        F.hardshrink,
        F.leaky_relu,
        F.logsigmoid,
        F.softplus,
        F.prelu,
        F.softsign,
        F.tanhshrink,
        F.gelu,
    }
    return SUPPORTED_ACTIVATION_FUNCTIONS


def _get_supported_activation_modules():
    SUPPORTED_ACTIVATION_MODULES = {
        nn.ReLU,
        nn.RReLU,
        nn.Hardtanh,
        nn.ReLU6,
        nn.Sigmoid,
        nn.Hardsigmoid,
        nn.Tanh,
        nn.SiLU,
        nn.Mish,
        nn.Hardswish,
        nn.ELU,
        nn.CELU,
        nn.SELU,
        nn.Hardshrink,
        nn.LeakyReLU,
        nn.LogSigmoid,
        nn.Softplus,
        nn.PReLU,
        nn.Softsign,
        nn.Tanhshrink,
        nn.GELU,
    }
    return SUPPORTED_ACTIVATION_MODULES


def _get_default_structured_pruning_patterns() -> dict[
    tuple[type[nn.Module] | Callable | MatchAllNode | str, ...],
    Callable[..., None],
]:
    """
    Returns the patterns for conv2d / linear conversion for each element in the activation functions/modules defined above.
    """
    patterns: dict[
        tuple[type[nn.Module] | Callable | MatchAllNode | str, ...],
        Callable[..., None],
    ] = {
        # linear -> linear
        (nn.Linear, "output"): prune_linear,
        (nn.Linear, nn.Linear): prune_linear_linear,
        # conv2d -> conv2d
        (nn.Conv2d, "output"): prune_conv2d,
        (nn.Conv2d, nn.Conv2d): prune_conv2d_conv2d,
        # TODO LSTM Structured pruning does not support returned state currently.
        # Should find a way to explicitly match getitem(0) instead of getitem.
        # This will also require changing the pruning function.
        # lstm -> getitem(0) -> linear
        (nn.LSTM, getitem, nn.Linear): prune_lstm_output_linear,
        # lstm -> getitem(0) -> layernorm -> linear
        (nn.LSTM, getitem, nn.LayerNorm, nn.Linear): prune_lstm_output_layernorm_linear,
    }

    for activation in chain(
        _get_supported_activation_functions(), _get_supported_activation_modules()
    ):
        patterns.update(
            {
                # linear -> activation -> linear
                (nn.Linear, activation, nn.Linear): prune_linear_activation_linear,
                # conv2d -> activation -> conv2d
                (nn.Conv2d, activation, nn.Conv2d): prune_conv2d_activation_conv2d,
                # conv2d -> activation -> pool -> conv2d
                (
                    nn.Conv2d,
                    activation,
                    nn.AvgPool2d,
                    nn.Conv2d,
                ): prune_conv2d_activation_pool_conv2d,
                (
                    nn.Conv2d,
                    activation,
                    F.avg_pool2d,
                    nn.Conv2d,
                ): prune_conv2d_activation_pool_conv2d,
                (
                    nn.Conv2d,
                    activation,
                    nn.MaxPool2d,
                    nn.Conv2d,
                ): prune_conv2d_activation_pool_conv2d,
                (
                    nn.Conv2d,
                    activation,
                    F.max_pool2d,
                    nn.Conv2d,
                ): prune_conv2d_activation_pool_conv2d,
                # conv2d -> pool -> activation -> conv2d
                (
                    nn.Conv2d,
                    nn.AvgPool2d,
                    activation,
                    nn.Conv2d,
                ): prune_conv2d_pool_activation_conv2d,
                (
                    nn.Conv2d,
                    F.avg_pool2d,
                    activation,
                    nn.Conv2d,
                ): prune_conv2d_pool_activation_conv2d,
                (
                    nn.Conv2d,
                    nn.MaxPool2d,
                    activation,
                    nn.Conv2d,
                ): prune_conv2d_pool_activation_conv2d,
                (
                    nn.Conv2d,
                    F.max_pool2d,
                    activation,
                    nn.Conv2d,
                ): prune_conv2d_pool_activation_conv2d,
                # conv2d -> adaptive pool -> flatten -> linear
                (
                    nn.Conv2d,
                    nn.AdaptiveAvgPool2d,
                    nn.Flatten,
                    nn.Linear,
                ): prune_conv2d_pool_flatten_linear,
                (
                    nn.Conv2d,
                    nn.AdaptiveAvgPool2d,
                    torch.flatten,
                    nn.Linear,
                ): prune_conv2d_pool_flatten_linear,
                (
                    nn.Conv2d,
                    nn.AdaptiveMaxPool2d,
                    nn.Flatten,
                    nn.Linear,
                ): prune_conv2d_pool_flatten_linear,
                (
                    nn.Conv2d,
                    nn.AdaptiveMaxPool2d,
                    torch.flatten,
                    nn.Linear,
                ): prune_conv2d_pool_flatten_linear,
            }
        )
    return patterns


class BaseStructuredSparsifier(BaseSparsifier):
    r"""Base class for structured pruning.

    Abstract methods that need to be implemented:
        - update_mask: Function to compute a new mask for all keys in the
            `groups` attribute.

    Args:
        - defaults [dict]: default configurations will be attached to the
            configuration. Only the keys that don't exist in the `config` will
            be updated.
    """

    def __init__(self, defaults, patterns=None):
        super().__init__(defaults)
        if patterns is None:
            patterns = _get_default_structured_pruning_patterns()
        self.patterns = patterns

    def make_config_from_model(
        self,
        model: nn.Module,
        SUPPORTED_MODULES: set[type] | None = None,
    ) -> None:
        if SUPPORTED_MODULES is None:
            SUPPORTED_MODULES = _get_supported_structured_pruning_modules()
        super().make_config_from_model(model, SUPPORTED_MODULES=SUPPORTED_MODULES)

    def _prepare(self, *args, **kwargs) -> None:
        r"""This function will attach the FakeStructuredSparsity parameterizations
        and BiasHooks at the appropriate points in the model.
        """
        for config in self.groups:
            module = config["module"]
            tensor_name = config["tensor_name"]
            parametrization = config.get("parametrization", FakeStructuredSparsity)
            tensor = getattr(module, tensor_name)

            mask = config.get(
                "mask",
                torch.ones(tensor.shape[0], dtype=torch.bool, device=tensor.device),
            )
            self.state[config["tensor_fqn"]]["mask"] = mask
            parametrize.register_parametrization(
                module, tensor_name, parametrization(mask)
            )

            # if linear / conv, we add in bias hooks
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                prune_bias = config.get("prune_bias", True)
                if module.bias is not None:
                    module.register_parameter(
                        "_bias", nn.Parameter(module.bias.detach())
                    )
                    # pyrefly: ignore [bad-assignment]
                    module.bias = None
                    module.prune_bias = prune_bias

                module.register_forward_hook(
                    BiasHook(module.parametrizations.weight[0], prune_bias)  # type: ignore[union-attr, index]
                )

    def prune(self) -> None:
        r"""
        This function will FX symbolically trace the model and then find instances of the patterns
        defined in self.patterns (by default SUPPORTED_STRUCTURED_PRUNING_PATTERNS ).

        For each pattern, it will apply to corresponding conversion function, which will modify the output
        and input size expected by the modules within the pattern
        """

        self.traced = symbolic_trace(self.model)
        modules = dict(self.traced.named_modules())

        # Right now we check for matches simply by iterating across all the patterns
        # if this is slow we can store patterns in a trie-structure and modify this code for faster lookup
        for node in self.traced.graph.nodes:
            for pattern, convert_fn in self.patterns.items():
                matched = apply_match(modules, pattern, node, [])
                if matched is None:
                    continue

                # pyrefly: ignore [no-matching-overload]
                first_module = modules.get(node.target)
                # check if first module exists and has appropriate parameterization, otherwise skip
                if (
                    first_module is not None
                    and parametrize.is_parametrized(first_module)
                    and module_contains_param(first_module, FakeStructuredSparsity)
                ):
                    convert_block = []
                    for node in matched:
                        if node.op == "call_module":
                            convert_block.append(modules.get(node.target))
                        elif node.op == "call_function":
                            convert_block.append(node.target)
                    convert_fn(*convert_block)

        for module in self.traced.modules():
            if module_contains_param(module, FakeStructuredSparsity):
                raise Exception(  # noqa: TRY002
                    f"Error: {module} still contains FakeStructuredSparsity parametrizations!"
                )

        self.traced.graph.lint()
        self.traced.recompile()
        return self.traced  # type: ignore[return-value]
<|endoftext|># mypy: allow-untyped-defs
"""
Collection of conversion functions for linear / conv2d structured pruning
Also contains utilities for bias propagation
"""

from collections.abc import Callable
from typing import cast

import torch
from torch import nn, Tensor
from torch.nn.utils import parametrize
from torch.nn.utils.parametrize import ParametrizationList

from .parametrization import BiasHook, FakeStructuredSparsity


# BIAS PROPAGATION
def _remove_bias_handles(module: nn.Module) -> None:
    if hasattr(module, "_forward_hooks"):
        bias_hooks: list[int] = []
        for key, hook in module._forward_hooks.items():
            if isinstance(hook, BiasHook):
                bias_hooks.append(key)

        for key in bias_hooks:
            del module._forward_hooks[key]


def _get_adjusted_next_layer_bias(
    next_layer: nn.Module, pruned_biases: Tensor, mask: Tensor
) -> nn.Parameter:
    r"""Returns new adjusted bias for the second supported module"""
    if parametrize.is_parametrized(next_layer):
        # need to access original weight
        parametrization_dict = cast(nn.ModuleDict, next_layer.parametrizations)
        weight_parameterizations = cast(
            ParametrizationList, parametrization_dict.weight
        )
        next_weight = weight_parameterizations.original
    else:
        next_weight = cast(Tensor, next_layer.weight)

    scaling_weight = next_weight[:, ~mask]
    if isinstance(next_layer, nn.Conv2d):  # checking for Conv2d
        # Propagating first layer pruned biases and calculating the new second layer bias
        # involves more steps since the Conv2d scaling weight has extra dimensions,
        # so adding bias involves broadcasting, logically:
        # for each channel k in range(oC):
        #     scaled_biases = sum(first_bias[pruned_idx] @ next_weight[k, pruned_idx, :, :].T)
        #     new_next_bias[k] = old_next_bias[k] + scaled_biases
        scaling_product = torch.matmul(
            pruned_biases.reshape(1, -1), torch.transpose(scaling_weight, 1, 2)
        )
        sum_range = list(range(len(scaling_product.shape)))[
            1:
        ]  # all but the first dimension
        scaled_biases = torch.sum(scaling_product, sum_range)
    elif isinstance(next_layer, nn.Linear):  # Linear
        scaled_biases = torch.matmul(
            pruned_biases, torch.transpose(scaling_weight, 0, 1)
        )  # recall b2_new = b1 @ w2.T + b2
    else:
        raise NotImplementedError(f"Type {type(next_layer)} not supported yet.")

    if (
        parametrize.is_parametrized(next_layer)
        and getattr(next_layer, "_bias", None) is not None
    ):  # next_layer is parametrized & has original bias ._bias
        adjusted_bias = nn.Parameter(scaled_biases + next_layer._bias)  # type: ignore[operator]
    elif (
        not parametrize.is_parametrized(next_layer) and next_layer.bias is not None
    ):  # next_layer not parametrized & has .bias
        adjusted_bias = nn.Parameter(scaled_biases + next_layer.bias)  # type: ignore[operator]
    else:  # next_layer has no bias
        adjusted_bias = nn.Parameter(scaled_biases)
    return adjusted_bias


def _prune_module_bias(module: nn.Module, mask: Tensor) -> None:
    r"""Applies mask to given modules bias"""
    # prune bias along with weights, discard pruned indices of bias
    original_bias = cast(Tensor, getattr(module, "_bias", module.bias))
    if original_bias is not None:
        module.bias = nn.Parameter(original_bias[mask])

    #  remove _bias parameter
    if hasattr(module, "_bias"):
        delattr(module, "_bias")


def _propagate_module_bias(module: nn.Module, mask: Tensor) -> Tensor | None:
    r"""
    In the case that we need to propagate biases, this function will return the biases we need
    """
    # set current module bias
    if module.bias is not None:
        module.bias = nn.Parameter(cast(Tensor, module.bias)[mask])
    elif getattr(module, "_bias", None) is not None:
        # pyrefly: ignore [bad-assignment]
        module.bias = nn.Parameter(cast(Tensor, module._bias)[mask])

    # get pruned biases to propagate to subsequent layer
    if getattr(module, "_bias", None) is not None:
        pruned_biases = cast(Tensor, module._bias)[~mask]
    else:
        pruned_biases = None

    if hasattr(module, "_bias"):
        delattr(module, "_bias")

    return pruned_biases


# LINEAR
def _prune_linear_helper(linear: nn.Linear) -> Tensor:
    # expects linear to be a parameterized linear module
    parametrization_dict = cast(nn.ModuleDict, linear.parametrizations)
    weight_parameterizations = cast(ParametrizationList, parametrization_dict.weight)
    for p in weight_parameterizations:
        if isinstance(p, FakeStructuredSparsity):
            mask = cast(Tensor, p.mask)

    with torch.no_grad():
        parametrize.remove_parametrizations(linear, "weight", leave_parametrized=True)
        linear.weight = nn.Parameter(linear.weight[mask])  # type: ignore[possibly-undefined]
    linear.out_features = linear.weight.shape[0]
    _remove_bias_handles(linear)

    # pyrefly: ignore [unbound-name]
    return mask


def prune_linear(linear: nn.Linear) -> None:
    mask = _prune_linear_helper(linear)
    if getattr(linear, "prune_bias", False):
        _prune_module_bias(linear, mask)


def prune_linear_linear(linear1: nn.Linear, linear2: nn.Linear) -> None:
    prune_linear_activation_linear(linear1, None, linear2)


def prune_linear_activation_linear(
    linear1: nn.Linear,
    activation: Callable[[Tensor], Tensor] | None,
    linear2: nn.Linear,
):
    mask = _prune_linear_helper(linear1)
    if getattr(linear1, "prune_bias", False):
        _prune_module_bias(linear1, mask)
    else:
        pruned_biases = _propagate_module_bias(linear1, mask)
        if pruned_biases is not None:
            if activation:
                pruned_biases = activation(pruned_biases)
            linear2.bias = _get_adjusted_next_layer_bias(linear2, pruned_biases, mask)

    with torch.no_grad():
        if parametrize.is_parametrized(linear2):
            parametrization_dict = cast(nn.ModuleDict, linear2.parametrizations)
            weight_parameterizations = cast(
                ParametrizationList, parametrization_dict.weight
            )

            weight_parameterizations.original = nn.Parameter(
                weight_parameterizations.original[:, mask]
            )
            linear2.in_features = weight_parameterizations.original.shape[1]
        else:
            linear2.weight = nn.Parameter(linear2.weight[:, mask])
            linear2.in_features = linear2.weight.shape[1]


# CONV2D
def _prune_conv2d_helper(conv2d: nn.Conv2d) -> Tensor:
    parametrization_dict = cast(nn.ModuleDict, conv2d.parametrizations)
    weight_parameterizations = cast(ParametrizationList, parametrization_dict.weight)
    for p in weight_parameterizations:
        if isinstance(p, FakeStructuredSparsity):
            mask = cast(Tensor, p.mask)

    with torch.no_grad():
        parametrize.remove_parametrizations(conv2d, "weight", leave_parametrized=True)
        conv2d.weight = nn.Parameter(conv2d.weight[mask])  # type: ignore[possibly-undefined]
    conv2d.out_channels = conv2d.weight.shape[0]

    _remove_bias_handles(conv2d)
    # pyrefly: ignore [unbound-name]
    return mask


def prune_conv2d_padded(conv2d_1: nn.Conv2d) -> None:
    parametrization_dict = cast(nn.ModuleDict, conv2d_1.parametrizations)
    weight_parameterizations = cast(ParametrizationList, parametrization_dict.weight)
    for p in weight_parameterizations:
        if isinstance(p, FakeStructuredSparsity):
            mask = cast(Tensor, p.mask)

    with torch.no_grad():
        parametrize.remove_parametrizations(conv2d_1, "weight", leave_parametrized=True)

    if getattr(conv2d_1, "_bias", None) is not None:
        if (
            conv2d_1.bias is not None
        ):  # conv2d_1 has original bias and bias propagated from previous layer
            new_bias = torch.zeros(conv2d_1.bias.shape)
            new_bias[mask] = conv2d_1.bias[mask]  # type: ignore[possibly-undefined]
            # adjusted bias to keep in conv2d_1
            # pyrefly: ignore [unbound-name]
            new_bias[~mask] = cast(Tensor, conv2d_1._bias)[~mask]
            # pruned biases that are kept instead of propagated
            conv2d_1.bias = nn.Parameter(new_bias)
        else:  # conv2d_1 has only original bias
            conv2d_1.bias = nn.Parameter(cast(Tensor, conv2d_1._bias))
    else:
        # no original bias, only propagated bias
        if (
            conv2d_1.bias is not None
        ):  # conv2d_1 has bias propagated from previous layer
            conv2d_1.bias.data[~mask] = 0  # type: ignore[possibly-undefined]

    if hasattr(conv2d_1, "_bias"):
        delattr(conv2d_1, "_bias")


def prune_conv2d(conv2d: nn.Conv2d) -> None:
    mask = _prune_conv2d_helper(conv2d)
    if getattr(conv2d, "prune_bias", False):
        _prune_module_bias(conv2d, mask)


def prune_conv2d_conv2d(conv2d_1: nn.Conv2d, conv2d_2: nn.Conv2d) -> None:
    prune_conv2d_activation_conv2d(conv2d_1, None, conv2d_2)


def prune_conv2d_activation_conv2d(
    conv2d_1: nn.Conv2d,
    activation: Callable[[Tensor], Tensor] | None,
    conv2d_2: nn.Conv2d,
):
    r"""
    Fusion Pattern for conv2d -> some activation module / function -> conv2d layers
    """
    parametrization_dict = cast(nn.ModuleDict, conv2d_1.parametrizations)
    weight_parameterizations = cast(ParametrizationList, parametrization_dict.weight)
    for p in weight_parameterizations:
        if isinstance(p, FakeStructuredSparsity):
            mask = cast(Tensor, p.mask)

    prune_bias = getattr(conv2d_1, "prune_bias", False)
    if (
        hasattr(conv2d_2, "padding")
        and cast(tuple[int], conv2d_2.padding) > (0, 0)
        and (conv2d_1.bias is not None or getattr(conv2d_1, "_bias", None) is not None)
    ):
        prune_conv2d_padded(conv2d_1)
    else:
        mask = _prune_conv2d_helper(conv2d_1)
        if prune_bias:
            _prune_module_bias(conv2d_1, mask)
        else:
            pruned_biases = _propagate_module_bias(conv2d_1, mask)
            if pruned_biases is not None:
                if activation:
                    pruned_biases = activation(pruned_biases)
                conv2d_2.bias = _get_adjusted_next_layer_bias(
                    conv2d_2, pruned_biases, mask
                )

        if (
            not (
                hasattr(conv2d_2, "padding")
                and cast(tuple[int], conv2d_2.padding) > (0, 0)
            )
            or conv2d_1.bias is None
        ):
            with torch.no_grad():
                if parametrize.is_parametrized(conv2d_2):
                    parametrization_dict = cast(
                        nn.ModuleDict, conv2d_2.parametrizations
                    )
                    weight_parameterizations = cast(
                        ParametrizationList, parametrization_dict.weight
                    )
                    weight_parameterizations.original = nn.Parameter(
                        weight_parameterizations.original[:, mask]
                    )
                    conv2d_2.in_channels = weight_parameterizations.original.shape[1]
                else:
                    conv2d_2.weight = nn.Parameter(conv2d_2.weight[:, mask])
                    conv2d_2.in_channels = conv2d_2.weight.shape[1]


def prune_conv2d_pool_activation_conv2d(
    c1: nn.Conv2d,
    pool: nn.Module,
    activation: Callable[[Tensor], Tensor] | None,
    c2: nn.Conv2d,
) -> None:
    prune_conv2d_activation_conv2d(c1, activation, c2)


def prune_conv2d_activation_pool_conv2d(
    c1: nn.Conv2d,
    activation: Callable[[Tensor], Tensor] | None,
    pool: nn.Module,
    c2: nn.Conv2d,
) -> None:
    prune_conv2d_activation_conv2d(c1, activation, c2)


def prune_conv2d_pool_flatten_linear(
    conv2d: nn.Conv2d,
    pool: nn.Module,
    flatten: Callable[[Tensor], Tensor] | None,
    linear: nn.Linear,
) -> None:
    mask = _prune_conv2d_helper(conv2d)

    # We map the pruned indices of the Conv2d output to the flattened indices of the Linear following the Flatten layer.
    # we determine the flattening scale (h * w), and readjust `first_pruned_indices`
    # (each idx maps to range idx * h * w to (idx+1) * h * w), `first_valid_indices`,
    # and `pruned_biases` (repeat each bias by h * w).
    if parametrize.is_parametrized(linear):
        parametrization_dict = cast(nn.ModuleDict, linear.parametrizations)
        weight_parameterizations = cast(
            ParametrizationList, parametrization_dict.weight
        )
        linear_ic = weight_parameterizations.original.shape[1]
    else:
        linear_ic = linear.weight.shape[1]

    conv2d_oc = len(mask)
    if linear_ic % conv2d_oc != 0:
        raise AssertionError(
            f"Flattening from dimensions {conv2d_oc} to {linear_ic} not supported"
        )

    flatten_scale = linear_ic // conv2d_oc
    flattened_mask = torch.tensor(
        [[val] * flatten_scale for val in mask], dtype=torch.bool, device=mask.device
    ).flatten()

    if getattr(conv2d, "prune_bias", False):
        _prune_module_bias(conv2d, mask)
    else:
        pruned_biases = cast(Tensor, _propagate_module_bias(conv2d, mask))
        flattened_pruned_biases = torch.tensor(
            [[bias] * flatten_scale for bias in pruned_biases], device=mask.device
        ).flatten()
        linear.bias = _get_adjusted_next_layer_bias(
            linear, flattened_pruned_biases, flattened_mask
        )

    with torch.no_grad():
        if parametrize.is_parametrized(linear):
            parametrization_dict = cast(nn.ModuleDict, linear.parametrizations)
            weight_parameterizations = cast(
                ParametrizationList, parametrization_dict.weight
            )
            weight_parameterizations.original = nn.Parameter(
                weight_parameterizations.original[:, flattened_mask]
            )
            linear.in_features = weight_parameterizations.original.shape[1]
        else:
            linear.weight = nn.Parameter(linear.weight[:, flattened_mask])
            linear.in_features = linear.weight.shape[1]


def prune_lstm_output_linear(
    lstm: nn.LSTM, getitem: Callable, linear: nn.Linear
) -> None:
    prune_lstm_output_layernorm_linear(lstm, getitem, None, linear)


def prune_lstm_output_layernorm_linear(
    lstm: nn.LSTM,
    getitem: Callable,
    layernorm: nn.LayerNorm | None,
    linear: nn.Linear,
) -> None:
    for i in range(lstm.num_layers):
        if parametrize.is_parametrized(lstm, f"weight_ih_l{i}"):
            parametrization_dict = cast(nn.ModuleDict, lstm.parametrizations)
            weight_parameterizations = cast(
                ParametrizationList, parametrization_dict[f"weight_ih_l{i}"]
            )
            mask = weight_parameterizations[0].mask

            with torch.no_grad():
                parametrize.remove_parametrizations(
                    lstm, f"weight_ih_l{i}", leave_parametrized=True
                )
                setattr(
                    lstm,
                    f"weight_ih_l{i}",
                    nn.Parameter(getattr(lstm, f"weight_ih_l{i}")[mask]),
                )
                setattr(
                    lstm,
                    f"bias_ih_l{i}",
                    nn.Parameter(getattr(lstm, f"bias_ih_l{i}")[mask]),
                )

        if parametrize.is_parametrized(lstm, f"weight_hh_l{i}"):
            parametrization_dict = cast(nn.ModuleDict, lstm.parametrizations)
            weight_parameterizations = cast(
                ParametrizationList, parametrization_dict[f"weight_hh_l{i}"]
            )
            mask = weight_parameterizations[0].mask

            with torch.no_grad():
                parametrize.remove_parametrizations(
                    lstm, f"weight_hh_l{i}", leave_parametrized=True
                )
                # splitting out hidden-hidden masks
                W_hi, W_hf, W_hg, W_ho = torch.split(
                    getattr(lstm, f"weight_hh_l{i}"), lstm.hidden_size
                )
                M_hi, M_hf, M_hg, M_ho = torch.split(mask, lstm.hidden_size)  # type: ignore[arg-type]

                # resize each individual weight separately
                W_hi = W_hi[M_hi][:, M_hi]
                W_hf = W_hf[M_hf][:, M_hf]
                W_hg = W_hg[M_hg][:, M_hg]
                W_ho = W_ho[M_ho][:, M_ho]

                # concat, use this as new weight
                new_weight = torch.cat((W_hi, W_hf, W_hg, W_ho))
                setattr(lstm, f"weight_hh_l{i}", nn.Parameter(new_weight))
                setattr(
                    lstm,
                    f"bias_hh_l{i}",
                    nn.Parameter(getattr(lstm, f"bias_hh_l{i}")[mask]),
                )

            # If this is the final layer, then we need to prune linear layer columns
            if i + 1 == lstm.num_layers:
                lstm.hidden_size = int(M_hi.sum())
                with torch.no_grad():
                    if parametrize.is_parametrized(linear):
                        parametrization_dict = cast(
                            nn.ModuleDict, linear.parametrizations
                        )
                        weight_parameterizations = cast(
                            ParametrizationList, parametrization_dict.weight
                        )

                        weight_parameterizations.original = nn.Parameter(
                            weight_parameterizations.original[:, M_ho]
                        )
                        linear.in_features = weight_parameterizations.original.shape[1]
                    else:
                        linear.weight = nn.Parameter(linear.weight[:, M_ho])
                        linear.in_features = linear.weight.shape[1]

                    # if layernorm module, prune weight and bias
                    if layernorm is not None:
                        layernorm.normalized_shape = (linear.in_features,)
                        layernorm.weight = nn.Parameter(layernorm.weight[M_ho])
                        layernorm.bias = nn.Parameter(layernorm.bias[M_ho])

            # otherwise need to prune the columns of the input of the next LSTM layer
            else:
                with torch.no_grad():
                    if parametrize.is_parametrized(lstm, f"weight_ih_l{i + 1}"):
                        parametrization_dict = cast(
                            nn.ModuleDict, lstm.parametrizations
                        )
                        weight_parameterizations = cast(
                            ParametrizationList,
                            getattr(parametrization_dict, f"weight_ih_l{i + 1}"),
                        )

                        weight_parameterizations.original = nn.Parameter(
                            weight_parameterizations.original[:, M_ho]
                        )
                    else:
                        next_layer_weight = getattr(lstm, f"weight_ih_l{i + 1}")
                        setattr(
                            lstm,
                            f"weight_ih_l{i + 1}",
                            nn.Parameter(next_layer_weight[:, M_ho]),
                        )
<|endoftext|># mypy: allow-untyped-defs
from collections.abc import Callable

import torch

from .base_structured_sparsifier import BaseStructuredSparsifier


__all__ = ["FPGMPruner"]


class FPGMPruner(BaseStructuredSparsifier):
    r"""Filter Pruning via Geometric Median (FPGM) Structured Pruner
    This sparsifier prune filter (row) in a tensor according to distances among filters according to
    `Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration <https://arxiv.org/abs/1811.00250>`_.

    This sparsifier is controlled by three variables:
    1. `sparsity_level` defines the number of filters (rows) that are zeroed-out.
    2. `dist` defines the distance measurement type. Default: 3 (L2 distance).
    Available options are: [1, 2, (custom callable distance function)].

    Note::
        Inputs should be a 4D convolutional tensor of shape (N, C, H, W).
            - N: output channels size
            - C: input channels size
            - H: height of kernel
            - W: width of kernel
    """

    def __init__(self, sparsity_level: float = 0.5, dist: Callable | int | None = None):
        defaults = {
            "sparsity_level": sparsity_level,
        }

        if dist is None:
            dist = 2

        if callable(dist):
            self.dist_fn = dist
        elif dist == 1:
            self.dist_fn = lambda x: torch.cdist(x, x, p=1)
        elif dist == 2:
            self.dist_fn = lambda x: torch.cdist(x, x, p=2)
        else:
            raise NotImplementedError("Distance function is not yet implemented.")
        super().__init__(defaults=defaults)

    def _compute_distance(self, t):
        r"""Compute distance across all entries in tensor `t` along all dimension
        except for the one identified by dim.
        Args:
            t (torch.Tensor): tensor representing the parameter to prune
        Returns:
            distance (torch.Tensor): distance computed across filters
        """
        dim = 0  # prune filter (row)

        size = t.size(dim)
        slc = [slice(None)] * t.dim()

        # flatten the tensor along the dimension
        t_flatten = [
            t[tuple(slc[:dim] + [slice(i, i + 1)] + slc[dim + 1 :])].reshape(-1)
            for i in range(size)
        ]
        t_flatten = torch.stack(t_flatten)

        # distance measurement
        dist_matrix = self.dist_fn(t_flatten)

        # more similar with other filter indicates large in the sum of row
        distance = torch.sum(torch.abs(dist_matrix), 1)

        return distance

    def update_mask(  # type: ignore[override]
        self, module, tensor_name, sparsity_level, **kwargs
    ):
        tensor_weight = getattr(module, tensor_name)
        mask = getattr(module.parametrizations, tensor_name)[0].mask

        if sparsity_level <= 0:
            mask.data = torch.ones_like(mask).bool()
        elif sparsity_level >= 1.0:
            mask.data = torch.zeros_like(mask).bool()
        else:
            distance = self._compute_distance(tensor_weight)

            tensor_size = tensor_weight.shape[0]  # prune filter (row)
            nparams_toprune = round(sparsity_level * tensor_size)
            nparams_toprune = min(
                max(nparams_toprune, 0), tensor_size
            )  # clamp to [0, tensor_size]
            topk = torch.topk(distance, k=nparams_toprune, largest=False)
            mask[topk.indices] = False
<|endoftext|># mypy: allow-untyped-defs
import torch
from torch import nn
from torch.nn.utils.parametrize import is_parametrized


def module_contains_param(module, parametrization):
    if is_parametrized(module):
        # see if any of the module tensors have a parametrization attached that matches the one passed in
        return any(
            any(isinstance(param, parametrization) for param in param_list)
            for param_list in module.parametrizations.values()
        )
    return False


# Structured Pruning Parameterizations
class FakeStructuredSparsity(nn.Module):
    r"""
    Parametrization for Structured Pruning. Like FakeSparsity, this should be attached to
    the  'weight' or any other parameter that requires a mask.

    Instead of an element-wise bool mask, this parameterization uses a row-wise bool mask.
    """

    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, x):
        if not isinstance(self.mask, torch.Tensor):
            raise AssertionError("mask must be a torch.Tensor")
        if self.mask.shape[0] != x.shape[0]:
            raise AssertionError(
                f"mask shape[0] ({self.mask.shape[0]}) must match x shape[0] ({x.shape[0]})"
            )
        shape = [1] * len(x.shape)
        shape[0] = -1
        return self.mask.reshape(shape) * x

    def state_dict(self, *args, **kwargs):
        # avoid double saving masks
        return {}


class BiasHook:
    def __init__(self, parametrization, prune_bias):
        self.param = parametrization
        self.prune_bias = prune_bias

    def __call__(self, module, input, output):
        if getattr(module, "_bias", None) is not None:
            bias = module._bias.data
            if self.prune_bias:
                bias[~self.param.mask] = 0

            # reshape bias to broadcast over output dimensions
            idx = [1] * len(output.shape)
            idx[1] = -1
            bias = bias.reshape(idx)

            output += bias
        return output
<|endoftext|>from .base_structured_sparsifier import BaseStructuredSparsifier
from .FPGM_pruner import FPGMPruner
from .lstm_saliency_pruner import LSTMSaliencyPruner
from .parametrization import BiasHook, FakeStructuredSparsity
from .saliency_pruner import SaliencyPruner
<|endoftext|>from typing import Any, cast

import torch
from torch import nn

from .base_structured_sparsifier import BaseStructuredSparsifier
from .parametrization import FakeStructuredSparsity


class LSTMSaliencyPruner(BaseStructuredSparsifier):
    """
    Prune packed LSTM weights based on saliency.
    For each layer {k} inside a LSTM, we have two packed weight matrices
    - weight_ih_l{k}
    - weight_hh_l{k}

    These tensors pack the weights for the 4 linear layers together for efficiency.

    [W_ii | W_if | W_ig | W_io]

    Pruning this tensor directly will lead to weights being misassigned when unpacked.
    To ensure that each packed linear layer is pruned the same amount:
        1. We split the packed weight into the 4 constituent linear parts
        2. Update the mask for each individual piece using saliency individually

    This applies to both weight_ih_l{k} and weight_hh_l{k}.
    """

    def update_mask(self, module: nn.Module, tensor_name: str, **kwargs: Any) -> None:
        weights = getattr(module, tensor_name)

        for p in getattr(module.parametrizations, tensor_name):
            if isinstance(p, FakeStructuredSparsity):
                mask = cast(torch.Tensor, p.mask)

                # select weights based on magnitude
                if weights.dim() <= 1:
                    raise Exception(  # noqa: TRY002
                        "Structured pruning can only be applied to a 2+dim weight tensor!"
                    )
                # take norm over all but first dim
                dims = tuple(range(1, weights.dim()))
                saliency = weights.norm(dim=dims, p=1)

                # handle weights in 4 groups
                split_size = len(mask) // 4
                masks = torch.split(mask, split_size)
                saliencies = torch.split(saliency, split_size)

                for keep_mask, sal in zip(masks, saliencies):
                    # mask smallest k values to be removed
                    k = int(len(keep_mask) * kwargs["sparsity_level"])
                    prune = sal.topk(k, largest=False, sorted=False).indices
                    keep_mask.data[prune] = False  # modifies underlying p.mask directly
<|endoftext|>"""
Contains utility functions to check if a pattern is in the graph and return the matching nodes
"""

from typing import Any

import torch
from torch import nn
from torch.ao.quantization.utils import MatchAllNode
from torch.fx import Node
from torch.nn.utils import parametrize


def _match(
    modules: dict[str, nn.ModuleDict],
    node: Node,
    current: nn.Module | Any,
) -> bool:
    r"""
    checks to see if a single node of a pattern matches
    """
    if isinstance(current, type) and issubclass(current, MatchAllNode):
        return True
    if not isinstance(node, Node):
        return False
    if isinstance(current, type) and issubclass(current, torch.nn.Module):
        return (
            node.op == "call_module"
            and parametrize.type_before_parametrizations(modules[node.target])  # type: ignore[index]
            == current
        )
    elif callable(current):
        return node.op == "call_function" and node.target is current
    elif isinstance(current, str):
        return node.target == current
    return False


def apply_match(
    modules: dict[str, nn.ModuleDict],
    pattern: tuple[Any] | Any,
    node: Node,
    matched_node_pattern: list[Node],
) -> list[Node] | None:
    r"""
    This function will return the matched nodes if the pattern matches the node given
    If there is no match, it will return None
    """
    if isinstance(pattern, tuple):
        if len(pattern) == 1:
            if _match(modules, node, pattern[0]):
                return matched_node_pattern + [node]

        first, *rest = pattern
        if _match(modules, node, first):
            if rest is None:
                return matched_node_pattern + [node]

            for user in node.users:
                return apply_match(
                    modules, tuple(rest), user, matched_node_pattern + [node]
                )
    elif _match(modules, node, pattern):
        return [node]
    return None
<|endoftext|># mypy: allow-untyped-defs
import abc
import warnings
import weakref
from functools import wraps

from torch.ao.pruning._experimental.data_sparsifier import BaseDataSparsifier


__all__ = ["BaseDataScheduler"]


class BaseDataScheduler:
    r"""
    The BaseDataScheduler is the abstract scheduler class specifically for the
    BaseDataSparsifier class. This class controls a specific hyperparameter of
    the sparsifier class and varies it across the training process (or across time).

    Args:
        data_sparsifier (instance of BaseDataSparsifier)
            Implemented class data sparsifier class wherein the update_mask is implemented
        schedule_param (str)
            A specific hyperparameter of the passed sparsifier that needs to be scheduled/varied
        last_epoch (int, default=-1)
            This is specifically passed when training needs to be resumed from a particular
            point.
        verbose (bool, default=False)
            Verbosity of the BaseDataScheduler

    The *get_hyperparam()* function needs to be implemented by the user.
    """

    def __init__(
        self, data_sparsifier, schedule_param: str, last_epoch=-1, verbose=False
    ):
        # Attach sparsifier
        if not isinstance(data_sparsifier, BaseDataSparsifier):
            raise TypeError(
                f"{type(data_sparsifier).__name__} is not an instance of torch.ao.pruning.BaseDataSparsifier"
            )
        self.data_sparsifier = data_sparsifier
        self.schedule_param = schedule_param

        # Initialize epoch and base hyper-params
        self.base_param = {
            name: config.get(schedule_param, None)
            for name, config in self.data_sparsifier.data_groups.items()
        }

        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `scheduler.step()` is called after
        # `sparsifier.step()`
        def with_counter(method):
            if getattr(method, "_with_counter", False):
                # `sparsifier.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the sparsifier instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1  # type: ignore[union-attr]
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True  # type: ignore[attr-defined]
            return wrapper

        self.data_sparsifier.step = with_counter(self.data_sparsifier.step)  # type: ignore[assignment]
        self.data_sparsifier._step_count = 0  # type: ignore[attr-defined]
        self._step_count: int = 0
        self.verbose = verbose

        # Housekeeping
        self._get_sp_called_within_step: bool = False  # sp -> schedule parameter
        self.step()

    @abc.abstractmethod
    def get_schedule_param(self):
        r"""
        Abstract method that needs to be implemented by the child class.
        The expected return type should be a dictionary of name to schedule_param value
        The returned values will be updated in sparsifier when the scheduler step() function
        is called.

        Example:
            >>> def get_schedule_param(self):
            ...     new_param = {}
            ...     for name in self.sparsifier.data_groups.keys():
            ...         new_param[name] = (
            ...             self.sparsifier.data_groups[name][self.schedule_param] * 0.5
            ...         )
            ...     return new_param

        When the step() function is called, the value in self.sparsifier.data_groups[name][self.schedule_param]
        would be halved
        """
        raise NotImplementedError

    def __repr__(self):
        format_string = self.__class__.__name__ + " ("
        format_string += "\n"
        format_string += f"Data Sparsifier {self.data_sparsifier}\n"
        format_string += f"    {self.schedule_param}: {self.base_param}\n"
        format_string += ")"
        return format_string

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the sparsifier.

        Note:
            The scheduler class does not track the state of the data_sparsifier.
            Make sure to store the state of the sparsifier before storing the
            state of the scheduler
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != "data_sparsifier"
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Note:
            Remember to restore the state of the data_sparsifier before the scheduler.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_param(self):
        return self._last_param

    def step(self):
        # Raise warning if trying to call scheduler step before the sparsifier.
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.data_sparsifier.step, "_with_counter"):
                warnings.warn(
                    "Seems like `data_sparsifier.step()` has been overridden after sparsity scheduler "
                    "initialization. Please, make sure to call `data_sparsifier.step()` before "
                    "`scheduler.step()`.",
                    UserWarning,
                    stacklevel=2,
                )

            # Just check if there were two first scheduler.step() calls before sparsifier.step()
            elif self.data_sparsifier._step_count < 1:  # type: ignore[attr-defined]
                warnings.warn(
                    "Detected call of `scheduler.step()` before `data_sparsifier.step()`. "
                    "You have to make sure you run the data_sparsifier.step() BEFORE any "
                    "calls to the scheduler.step().",
                    UserWarning,
                    stacklevel=2,
                )
        self._step_count += 1

        class _enable_get_sp_call:
            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_sp_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_sp_called_within_step = False

        with _enable_get_sp_call(self):
            self.last_epoch += 1
            updated_scheduler_params = self.get_schedule_param()

        for name, param in updated_scheduler_params.items():
            self.data_sparsifier.data_groups[name][self.schedule_param] = param
            if self.verbose:
                print(f"Adjusting {self.schedule_param} for group {name} to {param}")

        self._last_param = {
            name: config.get(self.schedule_param, None)
            for name, config in self.data_sparsifier.data_groups.items()
        }
        self.data_sparsifier.enable_mask_update = True
<|endoftext|># mypy: allow-untyped-defs
import copy
import warnings
from collections import defaultdict
from typing import Any

import torch
from torch import nn
from torch.ao.pruning.sparsifier.utils import fqn_to_module, module_to_fqn


__all__ = ["ActivationSparsifier"]


class ActivationSparsifier:
    r"""
    The Activation sparsifier class aims to sparsify/prune activations in a neural
    network. The idea is to attach the sparsifier to a layer (or layers) and it
    zeroes out the activations based on the mask_fn (or sparsification function)
    input by the user.
    The mask_fn is applied once all the inputs are aggregated and reduced i.e.
    mask = mask_fn(reduce_fn(aggregate_fn(activations)))

    Note::
        The sparsification mask is computed on the input **before it goes through the attached layer**.

    Args:
        model (nn.Module):
            The model whose layers will be sparsified. The layers that needs to be
            sparsified should be added separately using the register_layer() function
        aggregate_fn (Optional, Callable):
            default aggregate_fn that is used if not specified while registering the layer.
            specifies how inputs should be aggregated over time.
            The aggregate_fn should usually take 2 torch tensors and return the aggregated tensor.
            Example
                def add_agg_fn(tensor1, tensor2):  return tensor1 + tensor2
                reduce_fn (Optional, Callable):
                    default reduce_fn that is used if not specified while registering the layer.
                    reduce_fn will be called on the aggregated tensor i.e. the tensor obtained after
                    calling agg_fn() on all inputs.
                    Example
                def mean_reduce_fn(agg_tensor):    return agg_tensor.mean(dim=0)
                mask_fn (Optional, Callable):
                    default mask_fn that is used to create the sparsification mask using the tensor obtained after
                    calling the reduce_fn(). This is used by default if a custom one is passed in the
                    register_layer().
                    Note that the mask_fn() definition should contain the sparse arguments that is passed in sparse_config
                    arguments.
                features (Optional, list):
                    default selected features to sparsify.
                    If this is non-empty, then the mask_fn will be applied for each feature of the input.
                    For example,
                mask = [mask_fn(reduce_fn(aggregated_fn(input[feature])) for feature in features]
                feature_dim (Optional, int):
                    default dimension of input features. Again, features along this dim will be chosen
                    for sparsification.
                sparse_config (Dict):
                    Default configuration for the mask_fn. This config will be passed
                    with the mask_fn()

    Example:
        >>> # xdoctest: +SKIP
        >>> model = SomeModel()
        >>> act_sparsifier = ActivationSparsifier(...)  # init activation sparsifier
        >>> # Initialize aggregate_fn
        >>> def agg_fn(x, y):
        >>>     return x + y
        >>>
        >>> # Initialize reduce_fn
        >>> def reduce_fn(x):
        >>>     return torch.mean(x, dim=0)
        >>>
        >>> # Initialize mask_fn
        >>> def mask_fn(data):
        >>>     return torch.eye(data.shape).to(data.device)
        >>>
        >>>
        >>> act_sparsifier.register_layer(
        ...     model.some_layer,
        ...     aggregate_fn=agg_fn,
        ...     reduce_fn=reduce_fn,
        ...     mask_fn=mask_fn,
        ... )
        >>>
        >>> # start training process
        >>> for _ in [...]:
        >>> # epoch starts
        >>> # model.forward(), compute_loss() and model.backwards()
        >>> # epoch ends
        >>>     act_sparsifier.step()
        >>> # end training process
        >>> sparsifier.squash_mask()
    """

    def __init__(
        self,
        model: nn.Module,
        aggregate_fn=None,
        reduce_fn=None,
        mask_fn=None,
        features=None,
        feature_dim=None,
        **sparse_config,
    ):
        self.model = model
        self.defaults: dict[str, Any] = defaultdict()
        self.defaults["sparse_config"] = sparse_config

        # functions
        self.defaults["aggregate_fn"] = aggregate_fn
        self.defaults["reduce_fn"] = reduce_fn
        self.defaults["mask_fn"] = mask_fn

        # default feature and feature_dim
        self.defaults["features"] = features
        self.defaults["feature_dim"] = feature_dim

        self.data_groups: dict[str, dict] = defaultdict(
            dict
        )  # contains all relevant info w.r.t each registered layer

        self.state: dict[str, Any] = defaultdict(dict)  # layer name -> mask

    @staticmethod
    def _safe_rail_checks(args):
        """Makes sure that some of the functions and attributes are not passed incorrectly"""

        # if features are not None, then feature_dim must not be None
        features, feature_dim = args["features"], args["feature_dim"]
        if features is not None:
            if feature_dim is None:
                raise AssertionError("need feature dim to select features")

        # all the *_fns should be callable
        fn_keys = ["aggregate_fn", "reduce_fn", "mask_fn"]
        for key in fn_keys:
            fn = args[key]
            if not callable(fn):
                raise AssertionError(f"{fn} must be callable")

    def _aggregate_hook(self, name):
        """Returns hook that computes aggregate of activations passing through."""

        # gather some data
        feature_dim = self.data_groups[name]["feature_dim"]
        features = self.data_groups[name]["features"]
        agg_fn = self.data_groups[name]["aggregate_fn"]

        def hook(module, input) -> None:
            input_data = input[0]

            data = self.data_groups[name].get("data")  # aggregated data
            if features is None:
                # no features associated, data should not be a list
                if data is None:
                    data = torch.zeros_like(input_data)
                    self.state[name]["mask"] = torch.ones_like(input_data)
                out_data = agg_fn(data, input_data)
            else:
                # data should be a list [aggregated over each feature only]
                if data is None:
                    out_data = [
                        0 for _ in range(len(features))
                    ]  # create one in case of 1st forward
                    self.state[name]["mask"] = [0 for _ in range(len(features))]
                else:
                    out_data = data  # a list

                # compute aggregate over each feature
                for feature_idx in range(len(features)):
                    # each feature is either a list or scalar, convert it to torch tensor
                    feature_tensor = (
                        torch.Tensor([features[feature_idx]])
                        .long()
                        .to(input_data.device)
                    )
                    data_feature = torch.index_select(
                        input_data, feature_dim, feature_tensor
                    )
                    if data is None:
                        curr_data = torch.zeros_like(data_feature)
                        self.state[name]["mask"][feature_idx] = torch.ones_like(
                            data_feature
                        )
                    else:
                        curr_data = data[feature_idx]
                    out_data[feature_idx] = agg_fn(curr_data, data_feature)
            self.data_groups[name]["data"] = out_data

        return hook

    def register_layer(
        self,
        layer: nn.Module,
        aggregate_fn=None,
        reduce_fn=None,
        mask_fn=None,
        features=None,
        feature_dim=None,
        **sparse_config,
    ):
        r"""
        Registers a layer for sparsification. The layer should be part of self.model.
        Specifically, registers a pre-forward hook to the layer. The hook will apply the aggregate_fn
        and store the aggregated activations that is input over each step.

        Note::
            - There is no need to pass in the name of the layer as it is automatically computed as per
              the fqn convention.

            - All the functions (fn) passed as argument will be called at a dim, feature level.
        """
        name = module_to_fqn(self.model, layer)
        if name is None:
            raise AssertionError("layer not found in the model")

        if name in self.data_groups:  # unregister layer if already present
            warnings.warn(
                "layer already attached to the sparsifier, deregistering the layer and registering with new config",
                stacklevel=2,
            )
            self.unregister_layer(name=name)

        local_args = copy.deepcopy(self.defaults)
        update_dict = {
            "aggregate_fn": aggregate_fn,
            "reduce_fn": reduce_fn,
            "mask_fn": mask_fn,
            "features": features,
            "feature_dim": feature_dim,
            "layer": layer,
        }
        local_args.update(
            (arg, val) for arg, val in update_dict.items() if val is not None
        )
        local_args["sparse_config"].update(sparse_config)

        self._safe_rail_checks(local_args)

        self.data_groups[name] = local_args
        agg_hook = layer.register_forward_pre_hook(self._aggregate_hook(name=name))

        self.state[name]["mask"] = (
            None  # mask will be created when model forward is called.
        )

        # attach agg hook
        self.data_groups[name]["hook"] = agg_hook

        # for serialization purposes, we know whether aggregate_hook is attached
        # or sparsify_hook()
        self.data_groups[name]["hook_state"] = "aggregate"  # aggregate hook is attached

    def get_mask(self, name: str | None = None, layer: nn.Module | None = None):
        """
        Returns mask associated to the layer.

        The mask is
            - a torch tensor is features for that layer is None.
            - a list of torch tensors for each feature, otherwise

        Note::
            The shape of the mask is unknown until model.forward() is applied.
            Hence, if get_mask() is called before model.forward(), an
            error will be raised.
        """
        if name is None and layer is None:
            raise AssertionError("Need at least name or layer obj to retrieve mask")

        if name is None:
            if layer is None:
                raise AssertionError("layer must be provided when name is None")
            name = module_to_fqn(self.model, layer)
            if name is None:
                raise AssertionError("layer not found in the specified model")

        if name not in self.state:
            raise ValueError("Error: layer with the given name not found")

        mask = self.state[name].get("mask", None)

        if mask is None:
            raise ValueError(
                "Error: shape unknown, call layer() routine at least once to infer mask"
            )
        return mask

    def unregister_layer(self, name):
        """Detaches the sparsifier from the layer"""

        # detach any hooks attached
        self.data_groups[name]["hook"].remove()

        # pop from the state dict
        self.state.pop(name)

        # pop from the data groups
        self.data_groups.pop(name)

    def step(self):
        """Internally calls the update_mask() function for each layer"""
        with torch.no_grad():
            for name, configs in self.data_groups.items():
                data = configs["data"]
                self.update_mask(name, data, configs)

                self.data_groups[name].pop("data")  # reset the accumulated data

    def update_mask(self, name, data, configs):
        """
        Called for each registered layer and does the following-
            1. apply reduce_fn on the aggregated activations
            2. use mask_fn to compute the sparsification mask

        Note:
            the reduce_fn and mask_fn is called for each feature, dim over the data
        """
        mask = self.get_mask(name)
        sparse_config = configs["sparse_config"]
        features = configs["features"]
        reduce_fn = configs["reduce_fn"]
        mask_fn = configs["mask_fn"]
        if features is None:
            data = reduce_fn(data)
            mask.data = mask_fn(data, **sparse_config)
        else:
            for feature_idx in range(len(features)):
                data_feature = reduce_fn(data[feature_idx])
                mask[feature_idx].data = mask_fn(data_feature, **sparse_config)

    def _sparsify_hook(self, name):
        """Returns hook that applies sparsification mask to input entering the attached layer"""
        mask = self.get_mask(name)
        features = self.data_groups[name]["features"]
        feature_dim = self.data_groups[name]["feature_dim"]

        def hook(module, input):
            input_data = input[0]
            if features is None:
                # apply to all the features
                return input_data * mask
            else:
                # apply per feature, feature_dim
                for feature_idx in range(len(features)):
                    feature = (
                        torch.Tensor([features[feature_idx]])
                        .long()
                        .to(input_data.device)
                    )
                    sparsified = (
                        torch.index_select(input_data, feature_dim, feature)
                        * mask[feature_idx]
                    )
                    input_data.index_copy_(feature_dim, feature, sparsified)
                return input_data

        return hook

    def squash_mask(self, attach_sparsify_hook=True, **kwargs):
        """
        Unregisters aggregate hook that was applied earlier and registers sparsification hooks if
        attach_sparsify_hook = True.
        """
        for name, configs in self.data_groups.items():
            # unhook agg hook
            configs["hook"].remove()
            configs.pop("hook")
            self.data_groups[name]["hook_state"] = "None"
            if attach_sparsify_hook:
                configs["hook"] = configs["layer"].register_forward_pre_hook(
                    self._sparsify_hook(name)
                )
            configs["hook_state"] = (
                "sparsify"  # signals that sparsify hook is now attached
            )

    def _get_serializable_data_groups(self):
        """Exclude hook and layer from the config keys before serializing

        TODO: Might have to treat functions (reduce_fn, mask_fn etc) in a different manner while serializing.
              For time-being, functions are treated the same way as other attributes
        """
        data_groups: dict[str, Any] = defaultdict()
        for name, config in self.data_groups.items():
            new_config = {
                key: value
                for key, value in config.items()
                if key not in ["hook", "layer"]
            }
            data_groups[name] = new_config
        return data_groups

    def _convert_mask(self, states_dict, sparse_coo=True):
        r"""Converts the mask to sparse coo or dense depending on the `sparse_coo` argument.
        If `sparse_coo=True`, then the mask is stored as sparse coo else dense tensor
        """
        states = copy.deepcopy(states_dict)
        for state in states.values():
            if state["mask"] is not None:
                if isinstance(state["mask"], list):
                    for idx in range(len(state["mask"])):
                        if sparse_coo:
                            state["mask"][idx] = state["mask"][idx].to_sparse_coo()
                        else:
                            state["mask"][idx] = state["mask"][idx].to_dense()
                else:
                    if sparse_coo:
                        state["mask"] = state["mask"].to_sparse_coo()
                    else:
                        state["mask"] = state["mask"].to_dense()
        return states

    def state_dict(self) -> dict[str, Any]:
        r"""Returns the state of the sparsifier as a :class:`dict`.

        It contains:
        * state - contains name -> mask mapping.
        * data_groups - a dictionary containing all config information for each
            layer
        * defaults - the default config while creating the constructor
        """
        data_groups = self._get_serializable_data_groups()
        state = self._convert_mask(self.state)
        return {"state": state, "data_groups": data_groups, "defaults": self.defaults}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        r"""The load_state_dict() restores the state of the sparsifier based on the state_dict

        Args:
        * state_dict - the dictionary to which the current sparsifier needs to be restored to
        """
        state = state_dict["state"]
        data_groups, defaults = state_dict["data_groups"], state_dict["defaults"]

        self.__set_state__(
            {"state": state, "data_groups": data_groups, "defaults": defaults}
        )

    def __get_state__(self) -> dict[str, Any]:
        data_groups = self._get_serializable_data_groups()
        state = self._convert_mask(self.state)
        return {
            "defaults": self.defaults,
            "state": state,
            "data_groups": data_groups,
        }

    def __set_state__(self, state: dict[str, Any]) -> None:
        state["state"] = self._convert_mask(
            state["state"], sparse_coo=False
        )  # convert mask to dense tensor
        self.__dict__.update(state)

        # need to attach layer and hook info into the data_groups
        for name, config in self.data_groups.items():
            # fetch layer
            layer = fqn_to_module(self.model, name)
            if layer is None:
                raise AssertionError(f"layer {name} not found in the model")

            # if agg_mode is True, then layer in aggregate mode
            if "hook_state" in config and config["hook_state"] == "aggregate":
                hook = layer.register_forward_pre_hook(self._aggregate_hook(name))

            elif "hook_state" in config and config["hook_state"] == "sparsify":
                hook = layer.register_forward_pre_hook(self._sparsify_hook(name))

            config["layer"] = layer
            config["hook"] = hook  # type: ignore[possibly-undefined]

    def __repr__(self):
        format_string = self.__class__.__name__ + " ("
        for name, config in self.data_groups.items():
            format_string += "\n"
            format_string += "\tData Group\n"
            format_string += f"\t    name: {name}\n"
            for key in sorted(config.keys()):
                if key in ["data", "hook", "reduce_fn", "mask_fn", "aggregate_fn"]:
                    continue
                format_string += f"\t    {key}: {config[key]}\n"
        format_string += ")"
        return format_string
<|endoftext|>import warnings
from collections.abc import Callable

from torch.ao.pruning.sparsifier.base_sparsifier import BaseSparsifier

from .base_scheduler import BaseScheduler


__all__ = ["LambdaSL"]


class LambdaSL(BaseScheduler):
    """Sets the sparsity level of each parameter group to the final sl
    times a given function. When last_epoch=-1, sets initial sl as zero.
    Args:
        sparsifier (BaseSparsifier): Wrapped sparsifier.
        sl_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in sparsifier.param_groups.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    Example:
        >>> # Assuming sparsifier has two groups.
        >>> lambda1 = lambda epoch: epoch // 30
        >>> lambda2 = lambda epoch: 0.95**epoch
        >>> # xdoctest: +SKIP
        >>> scheduler = LambdaSL(sparsifier, sl_lambda=[lambda1, lambda2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
        self,
        sparsifier: BaseSparsifier,
        sl_lambda: Callable[[int], float] | list[Callable[[int], float]],
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        self.sparsifier = sparsifier

        if not isinstance(sl_lambda, list) and not isinstance(sl_lambda, tuple):
            self.sl_lambdas = [sl_lambda] * len(sparsifier.groups)
        else:
            if len(sl_lambda) != len(sparsifier.groups):
                raise ValueError(
                    f"Expected {len(sparsifier.groups)} lr_lambdas, but got {len(sl_lambda)}"
                )
            self.sl_lambdas = list(sl_lambda)
        super().__init__(sparsifier, last_epoch, verbose)  # type: ignore[no-untyped-call]

    def get_sl(self) -> list[float]:
        if not self._get_sl_called_within_step:
            warnings.warn(
                "To get the last sparsity level computed by the scheduler, "
                "please use `get_last_sl()`.",
                stacklevel=2,
            )
        return [
            base_sl * lmbda(self.last_epoch)
            for lmbda, base_sl in zip(self.sl_lambdas, self.base_sl)
        ]
<|endoftext|># mypy: allow-untyped-defs

import warnings
import weakref
from functools import wraps

from torch.ao.pruning.sparsifier.base_sparsifier import BaseSparsifier


__all__ = ["BaseScheduler"]


class BaseScheduler:
    def __init__(self, sparsifier, last_epoch=-1, verbose=False):
        # Attach sparsifier
        if not isinstance(sparsifier, BaseSparsifier):
            raise TypeError(
                f"{type(sparsifier).__name__} is not an instance of torch.ao.pruning.BaseSparsifier"
            )
        self.sparsifier = sparsifier

        # Initialize epoch and base sparsity levels

        self.base_sl = [group["sparsity_level"] for group in sparsifier.groups]
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `scheduler.step()` is called after
        # `sparsifier.step()`
        def with_counter(method):
            if getattr(method, "_with_counter", False):
                # `sparsifier.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the sparsifier instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1  # type: ignore[union-attr]
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True  # type: ignore[attr-defined]
            return wrapper

        self.sparsifier.step = with_counter(self.sparsifier.step)  # type: ignore[assignment]
        self.sparsifier._step_count = 0  # type: ignore[attr-defined]
        self._step_count: int = 0
        self.verbose = verbose

        # Housekeeping
        self._get_sl_called_within_step: bool = False

        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the sparsifier.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "sparsifier"
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_sl(self):
        """Return last computed sparsity level by current scheduler."""
        return self._last_sl

    def get_sl(self):
        # Compute sparsity level using chainable form of the scheduler
        # Note: This method is not intended to be called directly, and is only
        #       used by the ".step" method. Use .get_last_sl() instead.
        if not self._get_sl_called_within_step:
            warnings.warn(
                "To get the last sparsity level computed by the scheduler, "
                "please use `get_last_sl()`.",
                stacklevel=2,
            )
        raise NotImplementedError

    def print_sl(self, is_verbose, group, sl, epoch=None):
        """Display the current sparsity level."""
        if is_verbose:
            if epoch is None:
                print(f"Adjusting sparsity level of group {group} to {sl:.4e}.")
            else:
                print(
                    f"Epoch {epoch:5d}: adjusting sparsity level of group {group} to {sl:.4e}."
                )

    def __repr__(self):
        format_string = self.__class__.__name__ + " ("
        format_string += "\n"
        format_string += f"Sparsifier {self.sparsifier}\n"
        format_string += f"    base_sl: {self.base_sl}\n"
        format_string += ")"
        return format_string

    def step(self, epoch=None):
        # Raise warning if trying to call scheduler step before the sparsifier.
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.sparsifier.step, "_with_counter"):
                warnings.warn(
                    "Seems like `sparsifier.step()` has been overridden after sparsity scheduler "
                    "initialization. Please, make sure to call `sparsifier.step()` before "
                    "`scheduler.step()`.",
                    UserWarning,
                    stacklevel=2,
                )

            # Just check if there were two first scheduler.step() calls before sparsifier.step()
            elif self.sparsifier._step_count < 1:  # type: ignore[attr-defined]
                warnings.warn(
                    "Detected call of `scheduler.step()` before `sparsifier.step()`. "
                    "You have to make sure you run the sparsifier.step() BEFORE any "
                    "calls to the scheduler.step().",
                    UserWarning,
                    stacklevel=2,
                )
        self._step_count += 1

        class _enable_get_sl_call:
            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_sl_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_sl_called_within_step = False

        with _enable_get_sl_call(self):
            self.last_epoch += 1
            values = self.get_sl()

        for i, data in enumerate(zip(self.sparsifier.groups, values)):
            param_group, sl = data
            param_group["sparsity_level"] = sl
            self.print_sl(self.verbose, i, sl, epoch)

        self._last_sl = [group["sparsity_level"] for group in self.sparsifier.groups]
        self.sparsifier.enable_mask_update = True

    def _make_sure_a_list(self, var):
        r"""Utility that extends it to the same length as the .groups, ensuring it is a list"""
        n = len(self.sparsifier.groups)
        if not isinstance(var, (list, tuple)):
            return [var] * n
        else:
            if len(var) != n:
                raise ValueError(f"Expected variable of length {n}, but got {len(var)}")
            return list(var)  # We want the result to be in a list, not tuple
<|endoftext|># mypy: allow-untyped-defs
import warnings

from .base_scheduler import BaseScheduler


__all__ = ["CubicSL"]


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


class CubicSL(BaseScheduler):
    r"""Sets the sparsity level of each parameter group to the final sl
    plus a given exponential function.

    .. math::

        s_i = s_f + (s_0 - s_f) \cdot \left( 1 - \frac{t - t_0}{n\Delta t} \right)^3

    where :math:`s_i` is the sparsity at epoch :math:`t`, :math;`s_f` is the final
    sparsity level, :math:`f(i)` is the function to be applied to the current epoch
    :math:`t`, initial epoch :math:`t_0`, and final epoch :math:`t_f`.
    :math:`\Delta t` is used to control how often the update of the sparsity level
    happens. By default,

    Args:
        sparsifier (BaseSparsifier): Wrapped sparsifier.
        init_sl (int, list): Initial level of sparsity
        init_t (int, list): Initial step, when pruning starts
        delta_t (int, list): Pruning frequency
        total_t (int, list): Total number of pruning steps
        initially_zero (bool, list): If True, sets the level of sparsity to 0
            before init_t (:math:`t_0`). Otherwise, the sparsity level before
            init_t (:math:`t_0`) is set to init_sl(:math:`s_0`)
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(
        self,
        sparsifier,
        init_sl=0.0,
        init_t=0,
        delta_t=10,
        total_t=100,
        initially_zero=False,
        last_epoch=-1,
        verbose=False,
    ):
        self.sparsifier = sparsifier

        self.init_sl = self._make_sure_a_list(init_sl)
        self.init_t = self._make_sure_a_list(init_t)
        self.delta_t = self._make_sure_a_list(delta_t)
        self.total_t = self._make_sure_a_list(total_t)

        self.initially_zero = self._make_sure_a_list(initially_zero)

        super().__init__(sparsifier, last_epoch, verbose)

    @staticmethod
    def sparsity_compute_fn(s_0, s_f, t, t_0, dt, n, initially_zero=False):
        r""" "Computes the current level of sparsity.

        Based on https://arxiv.org/pdf/1710.01878.pdf

        Args:
            s_0: Initial level of sparsity, :math:`s_i`
            s_f: Target level of sparsity, :math:`s_f`
            t: Current step, :math:`t`
            t_0: Initial step, :math:`t_0`
            dt: Pruning frequency, :math:`\Delta T`
            n: Pruning steps, :math:`n`
            initially_zero: Sets the level of sparsity to 0 before t_0.
                If False, sets to s_0

        Returns:
            The sparsity level :math:`s_t` at the current step :math:`t`
        """
        if initially_zero and t < t_0:
            return 0
        s_t = s_f + (s_0 - s_f) * (1.0 - (t - t_0) / (dt * n)) ** 3
        s_t = _clamp(s_t, s_0, s_f)
        return s_t

    def get_sl(self):
        if not self._get_sl_called_within_step:
            warnings.warn(
                "To get the last sparsity level computed by the scheduler, "
                "please use `get_last_sl()`.",
                stacklevel=2,
            )
        return [
            self.sparsity_compute_fn(
                s_0=initial_sparsity,
                s_f=final_sparsity,
                t=self.last_epoch,
                t_0=initial_epoch,
                dt=delta_epoch,
                n=interval_epochs,
                initially_zero=initially_zero,
            )
            for initial_sparsity, final_sparsity, initial_epoch, delta_epoch, interval_epochs, initially_zero in zip(
                self.init_sl,
                self.base_sl,
                self.init_t,
                self.delta_t,
                self.total_t,
                self.initially_zero,
            )
        ]
<|endoftext|># mypy: allow-untyped-defs
import os.path
from glob import glob
from typing import cast

import torch
from torch.types import Storage


__serialization_id_record_name__ = ".data/serialization_id"


# because get_storage_from_record returns a tensor!?
class _HasStorage:
    def __init__(self, storage):
        self._storage = storage

    def storage(self):
        return self._storage


class DirectoryReader:
    """
    Class to allow PackageImporter to operate on unzipped packages. Methods
    copy the behavior of the internal PyTorchFileReader class (which is used for
    accessing packages in all other cases).

    N.B.: ScriptObjects are not depickleable or accessible via this DirectoryReader
    class due to ScriptObjects requiring an actual PyTorchFileReader instance.
    """

    def __init__(self, directory):
        self.directory = directory

    def get_record(self, name):
        filename = f"{self.directory}/{name}"
        with open(filename, "rb") as f:
            return f.read()

    def get_storage_from_record(self, name, numel, dtype):
        filename = f"{self.directory}/{name}"
        nbytes = torch._utils._element_size(dtype) * numel
        storage = cast(Storage, torch.UntypedStorage)
        return _HasStorage(storage.from_file(filename=filename, nbytes=nbytes))

    def has_record(self, path):
        full_path = os.path.join(self.directory, path)
        return os.path.isfile(full_path)

    def get_all_records(
        self,
    ):
        files = [
            filename[len(self.directory) + 1 :]
            for filename in glob(f"{self.directory}/**", recursive=True)
            if not os.path.isdir(filename)
        ]
        return files

    def serialization_id(
        self,
    ):
        if self.has_record(__serialization_id_record_name__):
            return self.get_record(__serialization_id_record_name__)
        else:
            return ""
<|endoftext|># mypy: allow-untyped-defs
import builtins
import importlib
import importlib.machinery
import inspect
import io
import linecache
import os
import sys
import types
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from typing import Any, cast, TYPE_CHECKING
from weakref import WeakValueDictionary

import torch
from torch.serialization import _get_restore_location, _maybe_decode_ascii
from torch.types import FileLike

from ._directory_reader import DirectoryReader
from ._importlib import (
    _calc___package__,
    _normalize_line_endings,
    _normalize_path,
    _resolve_name,
    _sanity_check,
)
from ._mangling import demangle, PackageMangler
from ._package_unpickler import PackageUnpickler
from .file_structure_representation import _create_directory_from_file_list, Directory
from .importer import Importer


if TYPE_CHECKING:
    from .glob_group import GlobPattern

__all__ = ["PackageImporter"]


# This is a list of imports that are implicitly allowed even if they haven't
# been marked as extern. This is to work around the fact that Torch implicitly
# depends on numpy and package can't track it.
# https://github.com/pytorch/multipy/issues/46  # codespell:ignore multipy
IMPLICIT_IMPORT_ALLOWLIST: Iterable[str] = [
    "numpy",
    "numpy.core",
    "numpy.core._multiarray_umath",
    # FX GraphModule might depend on builtins module and users usually
    # don't extern builtins. Here we import it here by default.
    "builtins",
]


# Compatibility name mapping to facilitate upgrade of external modules.
# The primary motivation is to enable Numpy upgrade that many modules
# depend on. The latest release of Numpy removed `numpy.str` and
# `numpy.bool` breaking unpickling for many modules.
EXTERN_IMPORT_COMPAT_NAME_MAPPING: dict[str, dict[str, Any]] = {
    "numpy": {
        "str": str,
        "bool": bool,
    },
}


class PackageImporter(Importer):
    """Importers allow you to load code written to packages by :class:`PackageExporter`.
    Code is loaded in a hermetic way, using files from the package
    rather than the normal python import system. This allows
    for the packaging of PyTorch model code and data so that it can be run
    on a server or used in the future for transfer learning.

    The importer for packages ensures that code in the module can only be loaded from
    within the package, except for modules explicitly listed as external during export.
    The file ``extern_modules`` in the zip archive lists all the modules that a package externally depends on.
    This prevents "implicit" dependencies where the package runs locally because it is importing
    a locally-installed package, but then fails when the package is copied to another machine.
    """

    """The dictionary of already loaded modules from this package, equivalent to ``sys.modules`` but
    local to this importer.
    """

    modules: dict[str, types.ModuleType]

    def __init__(
        self,
        file_or_buffer: FileLike | torch._C.PyTorchFileReader,
        module_allowed: Callable[[str], bool] = lambda module_name: True,
    ):
        """Open ``file_or_buffer`` for importing. This checks that the imported package only requires modules
        allowed by ``module_allowed``

        Args:
            file_or_buffer: a file-like object (has to implement :meth:`read`, :meth:`readline`, :meth:`tell`, and :meth:`seek`),
                a string, or an ``os.PathLike`` object containing a filename.
            module_allowed (Callable[[str], bool], optional): A method to determine if a externally provided module
                should be allowed. Can be used to ensure packages loaded do not depend on modules that the server
                does not support. Defaults to allowing anything.

        Raises:
            ImportError: If the package will use a disallowed module.
        """
        torch._C._log_api_usage_once("torch.package.PackageImporter")

        self.zip_reader: Any
        if isinstance(file_or_buffer, torch._C.PyTorchFileReader):
            self.filename = "<pytorch_file_reader>"
            self.zip_reader = file_or_buffer
        elif isinstance(file_or_buffer, (os.PathLike, str)):
            self.filename = os.fspath(file_or_buffer)
            if not os.path.isdir(self.filename):
                self.zip_reader = torch._C.PyTorchFileReader(self.filename)
            else:
                self.zip_reader = DirectoryReader(self.filename)
        else:
            self.filename = "<binary>"
            self.zip_reader = torch._C.PyTorchFileReader(file_or_buffer)

        torch._C._log_api_usage_metadata(
            "torch.package.PackageImporter.metadata",
            {
                "serialization_id": self.zip_reader.serialization_id(),
                "file_name": self.filename,
            },
        )

        self.root = _PackageNode(None)
        self.modules = {}
        self.extern_modules = self._read_extern()

        for extern_module in self.extern_modules:
            if not module_allowed(extern_module):
                raise ImportError(
                    f"package '{file_or_buffer}' needs the external module '{extern_module}' "
                    f"but that module has been disallowed"
                )
            self._add_extern(extern_module)

        for fname in self.zip_reader.get_all_records():
            self._add_file(fname)

        self.patched_builtins = builtins.__dict__.copy()
        self.patched_builtins["__import__"] = self.__import__
        # Allow packaged modules to reference their PackageImporter
        self.modules["torch_package_importer"] = self  # type: ignore[assignment]

        self._mangler = PackageMangler()

        # used for reduce deserialization
        self.storage_context: Any = None
        self.last_map_location = None

        # used for torch.serialization._load
        self.Unpickler = lambda *args, **kwargs: PackageUnpickler(self, *args, **kwargs)

    # pyrefly: ignore [bad-override]
    def import_module(self, name: str, package=None):
        """Load a module from the package if it hasn't already been loaded, and then return
        the module. Modules are loaded locally
        to the importer and will appear in ``self.modules`` rather than ``sys.modules``.

        Args:
            name (str): Fully qualified name of the module to load.
            package ([type], optional): Unused, but present to match the signature of importlib.import_module. Defaults to ``None``.

        Returns:
            types.ModuleType: The (possibly already) loaded module.
        """
        # We should always be able to support importing modules from this package.
        # This is to support something like:
        #   obj = importer.load_pickle(...)
        #   importer.import_module(obj.__module__)  <- this string will be mangled
        #
        # Note that _mangler.demangle will not demangle any module names
        # produced by a different PackageImporter instance.
        name = self._mangler.demangle(name)

        return self._gcd_import(name)

    def load_binary(self, package: str, resource: str) -> bytes:
        """Load raw bytes.

        Args:
            package (str): The name of module package (e.g. ``"my_package.my_subpackage"``).
            resource (str): The unique name for the resource.

        Returns:
            bytes: The loaded data.
        """

        path = self._zipfile_path(package, resource)
        return self.zip_reader.get_record(path)

    def load_text(
        self,
        package: str,
        resource: str,
        encoding: str = "utf-8",
        errors: str = "strict",
    ) -> str:
        """Load a string.

        Args:
            package (str): The name of module package (e.g. ``"my_package.my_subpackage"``).
            resource (str): The unique name for the resource.
            encoding (str, optional): Passed to ``decode``. Defaults to ``'utf-8'``.
            errors (str, optional): Passed to ``decode``. Defaults to ``'strict'``.

        Returns:
            str: The loaded text.
        """
        data = self.load_binary(package, resource)
        return data.decode(encoding, errors)

    def load_pickle(self, package: str, resource: str, map_location=None) -> Any:
        """Unpickles the resource from the package, loading any modules that are needed to construct the objects
        using :meth:`import_module`.

        Args:
            package (str): The name of module package (e.g. ``"my_package.my_subpackage"``).
            resource (str): The unique name for the resource.
            map_location: Passed to `torch.load` to determine how tensors are mapped to devices. Defaults to ``None``.

        Returns:
            Any: The unpickled object.
        """
        pickle_file = self._zipfile_path(package, resource)
        restore_location = _get_restore_location(map_location)
        loaded_storages = {}
        loaded_reduces = {}
        storage_context = torch._C.DeserializationStorageContext()

        def load_tensor(dtype, size, key, location, restore_location):
            name = f"{key}.storage"

            if storage_context.has_storage(name):
                storage = storage_context.get_storage(name, dtype)._typed_storage()
            else:
                tensor = self.zip_reader.get_storage_from_record(
                    ".data/" + name, size, dtype
                )
                if isinstance(self.zip_reader, torch._C.PyTorchFileReader):
                    storage_context.add_storage(name, tensor)
                storage = tensor._typed_storage()
            loaded_storages[key] = restore_location(storage, location)

        def persistent_load(saved_id):
            if not isinstance(saved_id, tuple):
                raise AssertionError(
                    f"saved_id must be a tuple, got {type(saved_id).__name__}"
                )
            typename = _maybe_decode_ascii(saved_id[0])
            data = saved_id[1:]

            if typename == "storage":
                storage_type, key, location, size = data
                if storage_type is torch.UntypedStorage:
                    dtype = torch.uint8
                else:
                    dtype = storage_type.dtype

                if key not in loaded_storages:
                    load_tensor(
                        dtype,
                        size,
                        key,
                        _maybe_decode_ascii(location),
                        restore_location,
                    )
                storage = loaded_storages[key]
                # TODO: Once we decide to break serialization FC, we can
                # stop wrapping with TypedStorage
                return torch.storage.TypedStorage(
                    wrap_storage=storage._untyped_storage, dtype=dtype, _internal=True
                )
            elif typename == "reduce_package":
                # to fix BC breaking change, objects on this load path
                # will be loaded multiple times erroneously
                if len(data) == 2:
                    func, args = data
                    return func(self, *args)
                reduce_id, func, args = data
                if reduce_id not in loaded_reduces:
                    loaded_reduces[reduce_id] = func(self, *args)
                return loaded_reduces[reduce_id]
            else:
                f"Unknown typename for persistent_load, expected 'storage' or 'reduce_package' but got '{typename}'"

        # Load the data (which may in turn use `persistent_load` to load tensors)
        data_file = io.BytesIO(self.zip_reader.get_record(pickle_file))
        unpickler = self.Unpickler(data_file)
        unpickler.persistent_load = persistent_load  # type: ignore[assignment]

        @contextmanager
        def set_deserialization_context():
            # to let reduce_package access deserialization context
            self.storage_context = storage_context
            self.last_map_location = map_location
            try:
                yield
            finally:
                self.storage_context = None
                self.last_map_location = None

        with set_deserialization_context():
            result = unpickler.load()

        # TODO from zdevito:
        #   This stateful weird function will need to be removed in our efforts
        #   to unify the format. It has a race condition if multiple python
        #   threads try to read independent files
        torch._utils._validate_loaded_sparse_tensors()

        return result

    def id(self):
        """
        Returns internal identifier that torch.package uses to distinguish :class:`PackageImporter` instances.
        Looks like::

            <torch_package_0>
        """
        return self._mangler.parent_name()

    def file_structure(
        self, *, include: "GlobPattern" = "**", exclude: "GlobPattern" = ()
    ) -> Directory:
        """Returns a file structure representation of package's zipfile.

        Args:
            include (list[str] | str): An optional string e.g. ``"my_package.my_subpackage"``, or optional list of strings
                for the names of the files to be included in the zipfile representation. This can also be
                a glob-style pattern, as described in :meth:`PackageExporter.mock`

            exclude (list[str] | str): An optional pattern that excludes files whose name match the pattern.

        Returns:
            :class:`Directory`
        """
        return _create_directory_from_file_list(
            self.filename, self.zip_reader.get_all_records(), include, exclude
        )

    def python_version(self):
        """Returns the version of python that was used to create this package.

        Note: this function is experimental and not Forward Compatible. The plan is to move this into a lock
        file later on.

        Returns:
            :class:`str | None` a python version e.g. 3.8.9 or None if no version was stored with this package
        """
        python_version_path = ".data/python_version"
        return (
            self.zip_reader.get_record(python_version_path).decode("utf-8").strip()
            if self.zip_reader.has_record(python_version_path)
            else None
        )

    def _read_extern(self):
        return (
            self.zip_reader.get_record(".data/extern_modules")
            .decode("utf-8")
            .splitlines(keepends=False)
        )

    def _make_module(
        self, name: str, filename: str | None, is_package: bool, parent: str
    ):
        mangled_filename = self._mangler.mangle(filename) if filename else None
        spec = importlib.machinery.ModuleSpec(
            name,
            self,  # type: ignore[arg-type]
            origin="<package_importer>",
            is_package=is_package,
        )
        module = importlib.util.module_from_spec(spec)
        self.modules[name] = module
        module.__name__ = self._mangler.mangle(name)
        ns = module.__dict__
        ns["__spec__"] = spec
        ns["__loader__"] = self
        ns["__file__"] = mangled_filename
        ns["__cached__"] = None
        ns["__builtins__"] = self.patched_builtins
        ns["__torch_package__"] = True

        # Add this module to our private global registry. It should be unique due to mangling.
        if module.__name__ in _package_imported_modules:
            raise AssertionError(
                f"module {module.__name__} already exists in _package_imported_modules"
            )
        _package_imported_modules[module.__name__] = module

        # preemptively install on the parent to prevent IMPORT_FROM from trying to
        # access sys.modules
        self._install_on_parent(parent, name, module)

        if filename is not None:
            if mangled_filename is None:
                raise AssertionError(
                    "mangled_filename must not be None when filename is set"
                )
            # preemptively install the source in `linecache` so that stack traces,
            # `inspect`, etc. work.
            if filename in linecache.cache:  # type: ignore[attr-defined]
                raise AssertionError(f"filename {filename} already in linecache.cache")
            linecache.lazycache(mangled_filename, ns)

            code = self._compile_source(filename, mangled_filename)
            exec(code, ns)

        return module

    def _load_module(self, name: str, parent: str):
        cur: _PathNode = self.root
        for atom in name.split("."):
            if not isinstance(cur, _PackageNode) or atom not in cur.children:
                if name in IMPLICIT_IMPORT_ALLOWLIST:
                    module = self.modules[name] = importlib.import_module(name)
                    return module
                raise ModuleNotFoundError(
                    f'No module named "{name}" in self-contained archive "{self.filename}"'
                    f" and the module is also not in the list of allowed external modules: {self.extern_modules}",
                    name=name,
                )
            cur = cur.children[atom]
            if isinstance(cur, _ExternNode):
                module = self.modules[name] = importlib.import_module(name)

                if compat_mapping := EXTERN_IMPORT_COMPAT_NAME_MAPPING.get(name):
                    for old_name, new_name in compat_mapping.items():
                        module.__dict__.setdefault(old_name, new_name)

                return module
        return self._make_module(
            name,
            cur.source_file,  # type: ignore[attr-defined]
            isinstance(cur, _PackageNode),
            parent,
        )

    def _compile_source(self, fullpath: str, mangled_filename: str):
        source = self.zip_reader.get_record(fullpath)
        source = _normalize_line_endings(source)
        return compile(source, mangled_filename, "exec", dont_inherit=True)

    # note: named `get_source` so that linecache can find the source
    # when this is the __loader__ of a module.
    def get_source(self, module_name) -> str:
        # linecache calls `get_source` with the `module.__name__` as the argument, so we must demangle it here.
        module = self.import_module(demangle(module_name))
        return self.zip_reader.get_record(demangle(module.__file__)).decode("utf-8")

    # note: named `get_resource_reader` so that importlib.resources can find it.
    # This is otherwise considered an internal method.
    def get_resource_reader(self, fullname):
        try:
            package = self._get_package(fullname)
        except ImportError:
            return None
        if package.__loader__ is not self:
            return None
        return _PackageResourceReader(self, fullname)

    def _install_on_parent(self, parent: str, name: str, module: types.ModuleType):
        if not parent:
            return
        # Set the module as an attribute on its parent.
        parent_module = self.modules[parent]
        if parent_module.__loader__ is self:
            setattr(parent_module, name.rpartition(".")[2], module)

    # note: copied from cpython's import code, with call to create module replaced with _make_module
    def _do_find_and_load(self, name):
        parent = name.rpartition(".")[0]
        module_name_no_parent = name.rpartition(".")[-1]
        if parent:
            if parent not in self.modules:
                self._gcd_import(parent)
            # Crazy side-effects!
            if name in self.modules:
                return self.modules[name]
            parent_module = self.modules[parent]

            if not hasattr(parent_module, "__path__"):
                # when we attempt to import a package only containing pybinded files,
                # the parent directory isn't always a package as defined by python,
                # so we search if the package is actually there or not before calling the error.
                if isinstance(
                    parent_module.__loader__,
                    importlib.machinery.ExtensionFileLoader,
                ):
                    if name not in self.extern_modules:
                        msg = (
                            _ERR_MSG
                            + "; {!r} is a c extension module which was not externed. C extension modules \
                            need to be externed by the PackageExporter in order to be used as we do not support interning them.}."
                        ).format(name, name)
                        raise ModuleNotFoundError(msg, name=name) from None
                    if not isinstance(
                        parent_module.__dict__.get(module_name_no_parent),
                        types.ModuleType,
                    ):
                        msg = (
                            _ERR_MSG
                            + "; {!r} is a c extension package which does not contain {!r}."
                        ).format(name, parent, name)
                        raise ModuleNotFoundError(msg, name=name) from None
                else:
                    msg = (_ERR_MSG + "; {!r} is not a package").format(name, parent)
                    raise ModuleNotFoundError(msg, name=name) from None

        module = self._load_module(name, parent)

        self._install_on_parent(parent, name, module)

        return module

    # note: copied from cpython's import code
    def _find_and_load(self, name):
        module = self.modules.get(name, _NEEDS_LOADING)
        if module is _NEEDS_LOADING:
            return self._do_find_and_load(name)

        if module is None:
            message = f"import of {name} halted; None in sys.modules"
            raise ModuleNotFoundError(message, name=name)

        # To handle https://github.com/pytorch/pytorch/issues/57490, where std's
        # creation of fake submodules via the hacking of sys.modules is not import
        # friendly
        if name == "os":
            self.modules["os.path"] = cast(Any, module).path
        elif name == "typing":
            if sys.version_info < (3, 13):
                self.modules["typing.io"] = cast(Any, module).io
                self.modules["typing.re"] = cast(Any, module).re

        return module

    def _gcd_import(self, name, package=None, level=0):
        """Import and return the module based on its name, the package the call is
        being made from, and the level adjustment.

        This function represents the greatest common denominator of functionality
        between import_module and __import__. This includes setting __package__ if
        the loader did not.

        """
        _sanity_check(name, package, level)
        if level > 0:
            name = _resolve_name(name, package, level)

        return self._find_and_load(name)

    # note: copied from cpython's import code
    def _handle_fromlist(self, module, fromlist, *, recursive=False):
        """Figure out what __import__ should return.

        The import_ parameter is a callable which takes the name of module to
