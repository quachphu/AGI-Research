            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in ``__init__``.
        """
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


# This class exists solely to avoid triggering an obscure error when scripting
# an improperly quantized attention layer. See this issue for details:
# https://github.com/pytorch/pytorch/issues/58969
# TODO: fail fast on quantization API usage error, then remove this class
# and replace uses of it with plain Linear
class NonDynamicallyQuantizableLinear(Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )


class Bilinear(Module):
    r"""Applies a bilinear transformation to the incoming data: :math:`y = x_1^T A x_2 + b`.

    Args:
        in1_features: size of each first input sample, must be > 0
        in2_features: size of each second input sample, must be > 0
        out_features: size of each output sample, must be > 0
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input1: :math:`(*, H_\text{in1})` where :math:`H_\text{in1}=\text{in1\_features}` and
          :math:`*` means any number of additional dimensions including none. All but the last dimension
          of the inputs should be the same.
        - Input2: :math:`(*, H_\text{in2})` where :math:`H_\text{in2}=\text{in2\_features}`.
        - Output: :math:`(*, H_\text{out})` where :math:`H_\text{out}=\text{out\_features}`
          and all but the last dimension are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in1\_features}, \text{in2\_features})`.
            The values are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in1\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
                :math:`k = \frac{1}{\text{in1\_features}}`

    Examples::

        >>> m = nn.Bilinear(20, 30, 40)
        >>> input1 = torch.randn(128, 20)
        >>> input2 = torch.randn(128, 30)
        >>> output = m(input1, input2)
        >>> print(output.size())
        torch.Size([128, 40])
    """

    __constants__ = ["in1_features", "in2_features", "out_features"]
    in1_features: int
    in2_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in1_features, in2_features), **factory_kwargs)
        )

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in ``__init__``.
        """
        if self.in1_features <= 0:
            raise ValueError(
                f"in1_features must be > 0, but got (in1_features={self.in1_features})"
            )
        bound = 1 / math.sqrt(self.weight.size(1))
        init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return F.bilinear(input1, input2, self.weight, self.bias)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return (
            f"in1_features={self.in1_features}, in2_features={self.in2_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None}"
        )


class LazyLinear(LazyModuleMixin, Linear):
    r"""A :class:`torch.nn.Linear` module where `in_features` is inferred.

    In this module, the `weight` and `bias` are of :class:`torch.nn.UninitializedParameter`
    class. They will be initialized after the first call to ``forward`` is done and the
    module will become a regular :class:`torch.nn.Linear` module. The ``in_features`` argument
    of the :class:`Linear` is inferred from the ``input.shape[-1]``.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`


    """

    cls_to_become = Linear  # type: ignore[assignment]
    # pyrefly: ignore [bad-override]
    weight: UninitializedParameter
    bias: UninitializedParameter  # type: ignore[assignment]

    def __init__(
        self, out_features: int, bias: bool = True, device=None, dtype=None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        # bias is hardcoded to False to avoid creating tensor
        # that will soon be overwritten.
        # pyrefly: ignore [bad-argument-type]
        super().__init__(0, 0, False)
        # pyrefly: ignore [unexpected-keyword]
        self.weight = UninitializedParameter(**factory_kwargs)
        self.out_features = out_features
        if bias:
            # pyrefly: ignore [unexpected-keyword]
            self.bias = UninitializedParameter(**factory_kwargs)

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in ``__init__``.
        """
        # pyrefly: ignore [bad-argument-type]
        if not self.has_uninitialized_params() and self.in_features != 0:
            super().reset_parameters()

    def initialize_parameters(self, input) -> None:  # type: ignore[override]
        """
        Infers ``in_features`` based on ``input`` and initializes parameters.
        """
        # pyrefly: ignore [bad-argument-type]
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.in_features = input.shape[-1]
                self.weight.materialize((self.out_features, self.in_features))
                if self.bias is not None:
                    self.bias.materialize((self.out_features,))
                self.reset_parameters()
        if self.in_features == 0:
            if input.shape[-1] != self.weight.shape[-1]:
                raise AssertionError(
                    f"The in_features inferred from input: {input.shape[-1]} "
                    f"is not equal to in_features from self.weight: "
                    f"{self.weight.shape[-1]}"
                )
            self.in_features = input.shape[-1]


# TODO: PartialLinear - maybe in sparse?
<|endoftext|>from torch.ao.nn.intrinsic import (
    BNReLU2d,
    BNReLU3d,
    ConvBn1d,
    ConvBn2d,
    ConvBn3d,
    ConvBnReLU1d,
    ConvBnReLU2d,
    ConvBnReLU3d,
    ConvReLU1d,
    ConvReLU2d,
    ConvReLU3d,
    LinearBn1d,
    LinearReLU,
)
from torch.ao.nn.intrinsic.modules.fused import _FusedModule

# Include the subpackages in case user imports from it directly
from torch.nn.intrinsic import modules, qat, quantized


__all__ = [
    "ConvBn1d",
    "ConvBn2d",
    "ConvBn3d",
    "ConvBnReLU1d",
    "ConvBnReLU2d",
    "ConvBnReLU3d",
    "ConvReLU1d",
    "ConvReLU2d",
    "ConvReLU3d",
    "LinearReLU",
    "BNReLU2d",
    "BNReLU3d",
    "LinearBn1d",
]
<|endoftext|>from torch.nn.intrinsic.quantized.modules.bn_relu import BNReLU2d, BNReLU3d
from torch.nn.intrinsic.quantized.modules.conv_relu import (
    ConvReLU1d,
    ConvReLU2d,
    ConvReLU3d,
)
from torch.nn.intrinsic.quantized.modules.linear_relu import LinearReLU


__all__ = [
    "LinearReLU",
    "ConvReLU1d",
    "ConvReLU2d",
    "ConvReLU3d",
    "BNReLU2d",
    "BNReLU3d",
]
<|endoftext|>from torch.nn.intrinsic.qat.modules.conv_fused import (
    ConvBn1d,
    ConvBn2d,
    ConvBn3d,
    ConvBnReLU1d,
    ConvBnReLU2d,
    ConvBnReLU3d,
    ConvReLU1d,
    ConvReLU2d,
    ConvReLU3d,
    freeze_bn_stats,
    update_bn_stats,
)
from torch.nn.intrinsic.qat.modules.linear_fused import LinearBn1d
from torch.nn.intrinsic.qat.modules.linear_relu import LinearReLU


__all__ = [
    "LinearReLU",
    "LinearBn1d",
    "ConvReLU1d",
    "ConvReLU2d",
    "ConvReLU3d",
    "ConvBn1d",
    "ConvBn2d",
    "ConvBn3d",
    "ConvBnReLU1d",
    "ConvBnReLU2d",
    "ConvBnReLU3d",
    "update_bn_stats",
    "freeze_bn_stats",
]
<|endoftext|># mypy: allow-untyped-defs
"""Defines bias subclasses that work with scaled_dot_product_attention"""

from enum import auto, IntEnum
from warnings import warn

import torch
import torch.nn.functional as F
from torch.backends.cuda import (
    can_use_efficient_attention,
    can_use_flash_attention,
    is_flash_attention_available,
    SDPAParams,
)
from torch.nn.attention import _raise_kernel_warnings
from torch.nn.attention._utils import (
    _calculate_scale,
    _input_requires_grad,
    _postprocess_flash_output,
    _validate_sdpa_input,
)


__all__ = ["causal_upper_left", "causal_lower_right", "CausalVariant", "CausalBias"]


torch._dynamo.allow_in_graph(is_flash_attention_available)
torch._dynamo.allow_in_graph(can_use_flash_attention)
torch._dynamo.allow_in_graph(can_use_efficient_attention)
torch._dynamo.allow_in_graph(SDPAParams)


class CausalVariant(IntEnum):
    r"""
    Enum for causal variants used in attention mechanisms.

    Defines two types of causal biases:

    ``UPPER_LEFT``: Represents upper-left triangular bias for standard causal attention.
    The equivalent pytorch code for constructing this bias is:

    .. code-block:: python

        torch.tril(torch.ones(size, dtype=torch.bool))

    For instance, with ``shape=(3,4)``, the materialized bias tensor will be:

    .. code-block:: text

        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0]]


    ``LOWER_RIGHT``: Represents lower-right triangular bias, the include values are aligned to the lower
    right corner of the matrix.

    The equivalent pytorch code for constructing this bias is:

    .. code-block:: python

        diagonal_offset = size[1] - size[0]
        torch.tril(
            torch.ones(size, dtype=torch.bool),
            diagonal=diagonal_offset,
        )

    For instance, with ``shape=(3,4)``, the materialized bias tensor will be:

    .. code-block:: text

        [[1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]

    Note that these variants are equivalent to each other when the sequence lengths of the query and key/value
    tensors are equal since the triangular matrix is square.

    .. warning:: This enum is a prototype and subject to change.
    """

    UPPER_LEFT = auto()
    LOWER_RIGHT = auto()


class CausalBias(torch.Tensor):
    """
    A bias representing causal attention patterns. For an overview of the bias structure, see the :class:`CausalVariant` enum.

    This class is used for defining causal (triangular) attention biases. For constructing the bias, there exist
    two factory functions: :func:`causal_upper_left` and :func:`causal_lower_right`.

    Example:

    .. code-block:: python

        from torch.nn.attention.bias import causal_lower_right

        bsz, num_heads, seqlen_q, seqlen_kv, head_dim = 32, 8, 4, 12, 8

        # Create a lower-right causal bias
        attn_bias = causal_lower_right(seqlen_q, seqlen_kv)

        q = torch.randn(
            bsz, num_heads, seqlen_q, head_dim, device="cuda", dtype=torch.float16
        )
        k = torch.randn(
            bsz, num_heads, seqlen_kv, head_dim, device="cuda", dtype=torch.float16
        )
        v = torch.randn(
            bsz, num_heads, seqlen_kv, head_dim, device="cuda", dtype=torch.float16
        )

        out = F.scaled_dot_product_attention(q, k, v, attn_bias)

    .. warning:: This class is a prototype and subject to change.
    """

    def __init__(self, variant: CausalVariant, seq_len_q: int, seq_len_kv: int) -> None:
        """
        Initializes the CausalBias instance with a specified variant and sequence lengths.

        Args:
            variant (CausalVariant): The type of causal bias to use (either UPPER_LEFT or LOWER_RIGHT).
            seq_len_q (int): The sequence length of the query tensor.
            seq_len_kv (int): The sequence length of the key/value tensor.

        Raises a warning if the LOWER_RIGHT variant is used with seq_len_q > seq_len_kv, as it may produce NaNs.
        """
        if not isinstance(variant, CausalVariant):
            raise AssertionError(
                f"variant must be a CausalVariant, got {type(variant).__name__}"
            )
        super().__init__()
        self.variant = variant
        self.seq_len_q = seq_len_q
        self.seq_len_kv = seq_len_kv
        if seq_len_q > seq_len_kv and variant == CausalVariant.LOWER_RIGHT:
            warn(
                "Lower right causal bias will produce NaNs in the output when seq_len_q > seq_len_kv!",
                stacklevel=2,
            )

    def _upper_left(self, device: torch.device) -> torch.Tensor:
        """Upper left causal bias"""
        return torch.tril(
            torch.ones(self.seq_len_q, self.seq_len_kv, device=device, dtype=torch.bool)
        )

    def _lower_right(self, device: torch.device) -> torch.Tensor:
        """Lower right causal bias"""
        diagonal_offset = self.seq_len_kv - self.seq_len_q
        return torch.tril(
            torch.ones(
                self.seq_len_q, self.seq_len_kv, device=device, dtype=torch.bool
            ),
            diagonal=diagonal_offset,
        )

    # pyrefly: ignore [bad-return]
    def _materialize(self, device: torch.device | None = None) -> torch.Tensor:
        """
        Materializes the causal bias into a tensor form.

        Depending on the variant, this method generates either an upper-left or lower-right
        triangular matrix to represent the causal bias.

        Args:
            device (Optional[torch.device]): The device on which to create the tensor. Defaults to CPU.

        Returns:
            torch.Tensor: The materialized bias tensor.
        """
        if device is None:
            device = torch.device("cpu")
        if self.variant == CausalVariant.UPPER_LEFT:
            return self._upper_left(device)
        elif self.variant == CausalVariant.LOWER_RIGHT:
            return self._lower_right(device)

    @staticmethod
    def _dispatch(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: "CausalBias",
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        r"""
        Handles the logic for computing attention with the specified causal bias.

        Args:
            query (Tensor): Query tensor; shape :math:`(N, ..., L, E)`.
            key (Tensor): Key tensor; shape :math:`(N, ..., S, E)`.
            value (Tensor): Value tensor; shape :math:`(N, ..., S, Ev)`.
            attn_mask (CausalBias): The type of causal attention to apply.
                A boolean mask where a value of True indicates that the element *should* take part in attention.
                A float mask of the same type as query, key, value that is added to the attention score.
            dropout_p (float): Dropout probability; if greater than 0.0, dropout is applied
            is_causal (bool): If true, assumes upper left causal attention masking and errors if both attn_mask and is_causal
                are set.
            scale (optional float): Scaling factor applied prior to softmax. If None, the default value is set
                to :math:`\frac{1}{\sqrt{E}}`.
            enable_gqa (optional bool): If set to True, Grouped Query Attention (GQA) is enabled, by default it is set to False.

        Returns:
            output (Tensor): Attention output; shape :math:`(N, ..., L, Ev)`.

        Raises:
            ValueError: If the causal bias variant is not a CausalVariant type.

        """
        if is_causal:
            raise ValueError("CausalBias should not be used with causal=True")

        if (
            attn_mask.seq_len_q == attn_mask.seq_len_kv
            or attn_mask.variant == CausalVariant.UPPER_LEFT
        ):
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True,
                scale=scale,
                enable_gqa=enable_gqa,
            )
        elif attn_mask.variant == CausalVariant.LOWER_RIGHT:
            _validate_sdpa_input(query, key, value, None, dropout_p, is_causal, scale)
            sdpa_params = SDPAParams(
                query, key, value, None, dropout_p, is_causal, enable_gqa
            )
            if can_use_flash_attention(sdpa_params):
                alignment = 1 if query.device.type == "xpu" else 8
                og_head_size = query.size(-1)
                og_scale = _calculate_scale(og_head_size, scale)
                needs_padding = og_head_size % alignment != 0
                if needs_padding:
                    pad_len = alignment - (og_head_size % alignment)
                    query = torch.nn.functional.pad(query, (0, pad_len))
                    key = torch.nn.functional.pad(key, (0, pad_len))
                    value = torch.nn.functional.pad(value, (0, pad_len))
                out = torch.ops.aten._scaled_dot_product_flash_attention(
                    query,
                    key,
                    value,
                    dropout_p,
                    is_causal=True,  # TODO: Flash accepts causal = True and for this particular op it means lower right
                    return_debug_mask=False,
                    scale=og_scale,
                )[0]
                return _postprocess_flash_output(out, og_head_size)
            if can_use_efficient_attention(sdpa_params):
                compute_log_sumexp = False
                if _input_requires_grad(query, key, value):
                    compute_log_sumexp = True
                return torch.ops.aten._efficient_attention_forward(
                    query.transpose(1, 2),
                    key.transpose(1, 2),
                    value.transpose(1, 2),
                    bias=None,
                    cu_seqlens_q=None,
                    cu_seqlens_k=None,
                    max_seqlen_q=None,
                    max_seqlen_k=None,
                    dropout_p=dropout_p,
                    custom_mask_type=int(attn_mask.variant),
                    compute_log_sumexp=compute_log_sumexp,
                    scale=scale,
                    seqlen_k=None,
                )[0].transpose(1, 2)
            else:
                _raise_kernel_warnings(sdpa_params)
                # We can't use efficient attention the only support for lower right is via materialization
                return F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask._materialize(query.device),
                    dropout_p=dropout_p,
                    is_causal=False,
                    scale=scale,
                    enable_gqa=enable_gqa,
                )
        else:
            raise ValueError(
                f"CausalBias.variant must be a CausalVariant type, but found: {attn_mask.variant}"
            )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Defines the behavior of torch.nn.functional.scaled_dot_product_attention when the attn_bias is an AttnBias"""
        if kwargs is None:
            kwargs = {}
        if func is torch.nn.functional.scaled_dot_product_attention:
            return cls._dispatch(*args, **kwargs)
        return super().__torch_function__(func, types, args, kwargs)

    def __repr__(self) -> str:  # type:ignore[override]
        return self._materialize().__repr__()


def causal_upper_left(*size) -> CausalBias:
    """
    Creates an upper-left triangular causal bias.

    This function generates an upper-left triangular matrix to represent causal attention bias with a
    diagonal offset set so that the inclusive values are aligned to the upper left corner of the matrix.
    This equivalent to the `is_causal=True` argument in `scaled_dot_product_attention`.

    The equivalent pytorch code for constructing this bias is:

    .. code-block:: python

        torch.tril(torch.ones(size, dtype=torch.bool))

    For instance, with `shape=(3,4)`, the materialized bias tensor will be:

    .. code-block:: text

        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0]]

    Args:
        size: The size of the bias matrix.

    Returns:
        CausalBias: The UPPER_LEFT triangular causal bias variant.
    """
    if len(size) != 2:
        raise AssertionError("causal_upper_left only supports 2D tensors")
    seq_len_q, seq_len_kv = size
    return CausalBias(CausalVariant.UPPER_LEFT, seq_len_q, seq_len_kv)


def causal_lower_right(*size) -> CausalBias:
    """
    Creates a lower-right triangular causal bias.

    This function generates a lower-right triangular matrix to represent causal attention bias with a
    diagonal offset set so that the inclusive values are aligned to the lower right corner of the matrix.

    The equivalent pytorch code for constructing this bias is:

    .. code-block:: python

        diagonal_offset = size[1] - size[0]
        torch.tril(
            torch.ones(size, dtype=torch.bool),
            diagonal=diagonal_offset,
        )

    For instance, with `shape=(3,4)`, the materialized bias tensor will be:

    .. code-block:: text

        [[1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]

    Args:
        size: The size of the bias matrix.

    Returns:
        CausalBias: The LOWER_RIGHT triangular causal bias variant.
    """
    if len(size) != 2:
        raise AssertionError("causal_lower_right only supports 2D tensors")
    seq_len_q, seq_len_kv = size
    return CausalBias(CausalVariant.LOWER_RIGHT, seq_len_q, seq_len_kv)
<|endoftext|># mypy: allow-untyped-defs
"""Registry for flash attention implementations.

This module contains the registration system for flash attention implementations.
It has no torch dependencies to avoid circular imports during initialization.
"""

import logging
from collections.abc import Callable
from typing import Literal, Protocol


logger = logging.getLogger(__name__)


class FlashAttentionHandle(Protocol):
    def remove(self) -> None: ...


_RegisterFn = Callable[..., FlashAttentionHandle | None]
_FlashAttentionImpl = Literal["FA3", "FA4"]

_FLASH_ATTENTION_IMPLS: dict[str, _RegisterFn] = {}

_FLASH_ATTENTION_ACTIVE: tuple[str, FlashAttentionHandle] | None = None


def register_flash_attention_impl(
    impl: str | _FlashAttentionImpl,
    *,
    register_fn: _RegisterFn,
) -> None:
    """
    Register the callable that activates a flash attention impl.

    .. note::
        This function is intended for SDPA backend providers to register their
        implementations. End users should use :func:`activate_flash_attention_impl`
        to activate a registered implementation.

    Args:
        impl: Implementation identifier (e.g., ``"FA4"``).
        register_fn: Callable that performs the actual dispatcher registration.
            This function will be invoked by :func:`activate_flash_attention_impl`
            and should register custom kernels with the PyTorch dispatcher.
            It may optionally return a handle implementing
            :class:`FlashAttentionHandle` to keep any necessary state alive.

    Example:
        >>> def my_impl_register(module_path: str = "my_flash_impl"):
        ...     # Register custom kernels with torch dispatcher
        ...     pass  # doctest: +SKIP
        >>> register_flash_attention_impl(
        ...     "MyImpl", register_fn=my_impl_register
        ... )  # doctest: +SKIP
    """
    global _FLASH_ATTENTION_IMPLS
    _FLASH_ATTENTION_IMPLS[impl] = register_fn


def activate_flash_attention_impl(
    impl: str | _FlashAttentionImpl,
) -> None:
    """
    Activate into the dispatcher a previously registered flash attention impl.

    .. note::
        Backend providers should NOT automatically activate their implementation
        on import. Users should explicitly opt-in by calling this function or via
        environment variables to ensure multiple provider libraries can coexist.

    Args:
        impl: Implementation identifier to activate. See
            :func:`~torch.nn.attention.list_flash_attention_impls` for available
            implementations.
            If the backend's :func:`register_flash_attention_impl` callable
            returns a :class:`FlashAttentionHandle`, the registry keeps that
            handle alive for the lifetime of the process (until explicit
            uninstall support exists).

    Example:
        >>> activate_flash_attention_impl("FA4")  # doctest: +SKIP
    """
    global _FLASH_ATTENTION_ACTIVE, _FLASH_ATTENTION_IMPLS

    restore_flash_attention_impl(
        _raise_warn=False
    )  # first restore any prev overrides (if any) to default

    register_fn = _FLASH_ATTENTION_IMPLS.get(impl)
    if register_fn is None:
        raise ValueError(
            f"Unknown flash attention impl '{impl}'. "
            f"Available implementations: {list_flash_attention_impls()}"
        )

    handle = register_fn()
    if handle is not None:
        _FLASH_ATTENTION_ACTIVE = (impl, handle)


def list_flash_attention_impls() -> list[str]:
    """Return the names of all available flash attention implementations."""
    return sorted(_FLASH_ATTENTION_IMPLS.keys())


def current_flash_attention_impl() -> str | None:
    """
    Return the currently activated flash attention impl name, if any.

    ``None`` indicates that no custom impl has been activated.
    """
    return (
        _FLASH_ATTENTION_ACTIVE[0]
        if _FLASH_ATTENTION_ACTIVE is not None
        else _FLASH_ATTENTION_ACTIVE
    )


def restore_flash_attention_impl(_raise_warn: bool = True) -> None:
    """
    Restore the default FA2 implementation
    """
    global _FLASH_ATTENTION_ACTIVE

    handle = None
    if _FLASH_ATTENTION_ACTIVE is not None:
        handle = _FLASH_ATTENTION_ACTIVE[1]

    if handle is not None:
        handle.remove()
    elif _raise_warn:
        logger.warning(
            "Trying to restore default FA2 impl when no custom impl was activated"
        )

    _FLASH_ATTENTION_ACTIVE = None  # default
<|endoftext|>"""UBER PROTOTYPE!!!"""
# mypy: allow-untyped-defs

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import cache
from typing import Any, TYPE_CHECKING
from typing_extensions import TypeVarTuple, Unpack

from . import _registry


if TYPE_CHECKING:
    from types import ModuleType

import torch
from torch.library import Library


__all__ = [
    "register_flash_attention_fa4",
]


_FA4_MODULE_PATH: str | None = None


@dataclass
class _FA4Handle:
    library: Library | None

    def remove(self) -> None:
        self.library = None


@cache
def _get_device_major(device: torch.device) -> int:
    major, _ = torch.cuda.get_device_capability(device)
    return major


def register_flash_attention_fa4(
    module_path: str = "flash_attn.cute.interface",
) -> _FA4Handle:
    """
    Register FA4 flash attention kernels with the PyTorch dispatcher.

    Args:
        module_path: Python module path to the FA4 implementation.
    """
    global _FA4_MODULE_PATH
    _ = _fa4_import_module(module_path)
    _FA4_MODULE_PATH = module_path
    return _FA4Handle(_fa4_register_kernels())


@cache
def _fa4_import_module(module_path: str) -> ModuleType:
    module = importlib.import_module(module_path)
    if not hasattr(module, "_flash_attn_fwd") or not hasattr(module, "_flash_attn_bwd"):
        raise RuntimeError(f"Module '{module_path}' does not expose FA4 kernels")
    return module


def _fa4_register_kernels() -> Library:
    lib = Library("aten", "IMPL", "CUDA")
    lib.impl("_flash_attention_forward", _fa4_flash_attention_forward_impl, "CUDA")
    lib.impl(
        "_flash_attention_forward_no_dropout_inplace",
        _fa4_flash_attention_forward_no_dropout_inplace_impl,
        "CUDA",
    )
    lib.impl("_flash_attention_backward", _fa4_flash_attention_backward_impl, "CUDA")
    lib.impl(
        "_scaled_dot_product_flash_attention",
        _fa4_scaled_dot_product_flash_attention_forward_impl,
        "CUDA",
    )
    lib.impl(
        "_scaled_dot_product_flash_attention_backward",
        _fa4_scaled_dot_product_flash_attention_backward_impl,
        "CUDA",
    )
    return lib


def _fa4_common_support_error(
    query: torch.Tensor,
    tensors: tuple[torch.Tensor, ...],
    cum_seq_q: torch.Tensor | None,
    require_fp32: tuple[tuple[str, torch.Tensor], ...] = (),
) -> str | None:
    if not all(t.is_cuda for t in tensors):
        return "inputs must be CUDA tensors"
    if len({t.device for t in tensors}) != 1:
        return "inputs must share device"
    if query.dtype not in (torch.float16, torch.bfloat16):
        return "query dtype must be float16 or bfloat16"
    for name, tensor in require_fp32:
        if tensor.dtype != torch.float32:
            return f"{name} dtype must be float32"
    if cum_seq_q is None and query.dim() != 4:
        return "dense query must be 4D"
    if cum_seq_q is not None and query.dim() != 3:
        return "ragged query must be 3D"
    if not torch.cuda.is_available():
        return "CUDA not available"
    if _get_device_major(query.device) not in (9, 10):
        return "FA4 requires compute capability 9.0 or 10.0"
    return None


def _fa4_forward_support_error(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float,
    return_debug_mask: bool,
    alibi_slopes: torch.Tensor | None,
    seqused_k: torch.Tensor | None,
    cum_seq_q: torch.Tensor | None,
    block_table: torch.Tensor | None = None,
    num_splits: int | None = None,
) -> str | None:
    if dropout_p != 0.0:
        return "dropout_p must be 0"
    if return_debug_mask:
        return "return_debug_mask must be False"
    if alibi_slopes is not None:
        return "alibi_slopes not supported"
    if seqused_k is not None:
        if seqused_k.dtype != torch.int32:
            return "seqused_k must be int32"
        if not seqused_k.is_cuda:
            return "seqused_k must be CUDA"
    major = _get_device_major(query.device)
    if block_table is not None and major != 10:
        return f"paged KV (block_table) not supported on SM {major}0"
    if num_splits is not None and num_splits > 1 and major != 10:
        return f"SplitKV (num_splits > 1) not supported on SM {major}0"
    error = _fa4_common_support_error(
        query,
        (query, key, value),
        cum_seq_q,
    )
    if error is not None:
        if error == "inputs must share device":
            return "query, key, value must be on same device"
        return error
    return None


def _fa4_backward_support_error(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    dropout_p: float,
    cum_seq_q: torch.Tensor | None,
) -> str | None:
    if dropout_p != 0.0:
        return "dropout_p must be 0"
    error = _fa4_common_support_error(
        query,
        (grad_out, query, key, value, out, logsumexp),
        cum_seq_q,
        require_fp32=(("logsumexp", logsumexp),),
    )
    if error is not None:
        return error
    return None


def _aten_to_fa4_window_size(val: int | None) -> int | None:
    """need to convert -1 to None for FA4"""
    return None if val == -1 else val


Ts = TypeVarTuple("Ts")


def _transpose_dense(*tensors: Unpack[Ts]) -> tuple[Unpack[Ts]]:
    return tuple(t.transpose(1, 2) for t in tensors)  # type: ignore[attr-defined]


def _fa4_run_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_q: torch.Tensor | None,
    cu_seq_k: torch.Tensor | None,
    max_q: int | None,
    max_k: int | None,
    scale: float | None,
    is_causal: bool,
    window_size_left: int | None,
    window_size_right: int | None,
    seqused_k: torch.Tensor | None,
    out: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    num_splits: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _FA4_MODULE_PATH is None:
        raise RuntimeError("FA4 not registered")
    module = _fa4_import_module(_FA4_MODULE_PATH)

    kwargs: dict[str, Any] = {
        "softmax_scale": scale,
        "causal": is_causal,
        "window_size_left": _aten_to_fa4_window_size(window_size_left),
        "window_size_right": _aten_to_fa4_window_size(window_size_right),
        "return_lse": True,
        "cu_seqlens_q": cu_seq_q,
        "cu_seqlens_k": cu_seq_k,
        "max_seqlen_q": max_q,
        "max_seqlen_k": max_k,
        "seqused_k": seqused_k.contiguous() if seqused_k is not None else None,
        "page_table": block_table,
        "num_splits": num_splits or 1,
        "out": out,
    }
    out, lse, *_ = module._flash_attn_fwd(query, key, value, **kwargs)
    return out, lse.contiguous()


def _fa4_run_backward(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    cu_seq_q: torch.Tensor | None,
    cu_seq_k: torch.Tensor | None,
    scale: float | None,
    is_causal: bool,
    window_size_left: int | None,
    window_size_right: int | None,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if _FA4_MODULE_PATH is None:
        raise RuntimeError("FA4 not registered")
    module = _fa4_import_module(_FA4_MODULE_PATH)
    dq, dk, dv = module._flash_attn_bwd(
        query,
        key,
        value,
        out,
        grad_out,
        logsumexp.contiguous(),
        softmax_scale=scale,
        causal=is_causal,
        window_size_left=_aten_to_fa4_window_size(window_size_left),
        window_size_right=_aten_to_fa4_window_size(window_size_right),
        cu_seqlens_q=cu_seq_q,
        cu_seqlens_k=cu_seq_k,
        deterministic=deterministic,
    )
    return dq, dk, dv


def _fa4_flash_attention_forward_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cum_seq_q: torch.Tensor | None,
    cum_seq_k: torch.Tensor | None,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    return_debug_mask: bool,
    *,
    scale: float | None = None,
    window_size_left: int | None = None,
    window_size_right: int | None = None,
    seqused_k: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    compute_auxiliary: bool = True,
    num_splits: int | None = None,
):
    error = _fa4_forward_support_error(
        query,
        key,
        value,
        dropout_p,
        return_debug_mask,
        alibi_slopes,
        seqused_k,
        cum_seq_q,
        block_table,
        num_splits,
    )
    if error is not None:
        raise RuntimeError(f"FA4 flash_attention forward unsupported: {error}")
    out, lse = _fa4_run_forward(
        query,
        key,
        value,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        scale,
        is_causal,
        window_size_left,
        window_size_right,
        seqused_k,
        out,
        block_table,
        num_splits,
    )
    if compute_auxiliary:
        rng_state = torch.zeros((2,), dtype=torch.uint64, device=query.device)
        philox_offset = torch.zeros((), dtype=torch.uint64, device=query.device)
        debug_mask = torch.empty(0, dtype=query.dtype, device=query.device)
    else:
        rng_state = None
        philox_offset = None
        debug_mask = None
    return out, lse, rng_state, philox_offset, debug_mask


def _fa4_flash_attention_forward_no_dropout_inplace_impl(
    out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cum_seq_q: torch.Tensor | None,
    cum_seq_k: torch.Tensor | None,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    return_debug_mask: bool,
    *,
    scale: float | None = None,
    window_size_left: int | None = None,
    window_size_right: int | None = None,
    seqused_k: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    num_splits: int | None = None,
):
    _, lse, _, _, _ = _fa4_flash_attention_forward_impl(
        query,
        key,
        value,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        return_debug_mask,
        scale=scale,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        seqused_k=seqused_k,
        alibi_slopes=alibi_slopes,
        out=out,
        block_table=block_table,
        compute_auxiliary=False,
        num_splits=num_splits,
    )
    return lse


def _fa4_flash_attention_backward_impl(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    cum_seq_q: torch.Tensor | None,
    cum_seq_k: torch.Tensor | None,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    rng_state: torch.Tensor,
    unused: torch.Tensor,
    *,
    scale: float | None = None,
    window_size_left: int | None = None,
    window_size_right: int | None = None,
):
    error = _fa4_backward_support_error(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        dropout_p,
        cum_seq_q,
    )
    if error is not None:
        raise RuntimeError(f"FA4 flash_attention backward unsupported: {error}")
    deterministic = torch.are_deterministic_algorithms_enabled()
    dq, dk, dv = _fa4_run_backward(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        cum_seq_q,
        cum_seq_k,
        scale,
        is_causal,
        window_size_left,
        window_size_right,
        deterministic,
    )
    return dq, dk, dv


def _fa4_scaled_dot_product_flash_attention_forward_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: float | None = None,
):
    error = _fa4_forward_support_error(
        query,
        key,
        value,
        dropout_p,
        return_debug_mask,
        None,
        None,
        None,
    )
    if error is not None:
        raise RuntimeError(f"FA4 SDPA forward unsupported: {error}")
    q, k, v = _transpose_dense(query, key, value)

    # Pre-allocate output with query's strides (BHSD layout), then create
    # a BSHD view for the kernel. This ensures the returned output has
    # the same memory layout as the input query.
    out_bhsd = torch.empty_like(query)
    out_bshd = out_bhsd.transpose(1, 2)

    max_q_flash = q.size(1)
    max_k_flash = k.size(1)
    _, lse, rng_state, philox_offset, debug_mask = _fa4_flash_attention_forward_impl(
        q,
        k,
        v,
        None,
        None,
        max_q_flash,
        max_k_flash,
        dropout_p,
        is_causal,
        return_debug_mask,
        scale=scale,
        out=out_bshd,
    )
    max_q = query.size(2)
    max_k = key.size(2)
    return (
        out_bhsd,
        lse,
        None,
        None,
        max_q,
        max_k,
        rng_state,
        philox_offset,
        debug_mask,
    )


def _fa4_scaled_dot_product_flash_attention_backward_impl(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    cum_seq_q: torch.Tensor | None,
    cum_seq_k: torch.Tensor | None,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    *,
    scale: float | None = None,
):
    error = _fa4_backward_support_error(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        dropout_p,
        None,
    )
    if error is not None:
        raise RuntimeError(f"FA4 SDPA backward unsupported: {error}")
    q, k, v, o, go = _transpose_dense(query, key, value, out, grad_out)
    max_q = query.size(2)
    max_k = key.size(2)
    dq, dk, dv = _fa4_flash_attention_backward_impl(
        go,
        q,
        k,
        v,
        o,
        logsumexp,
        None,
        None,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        philox_seed,
        philox_offset,
        scale=scale,
    )
    dq, dk, dv = _transpose_dense(dq, dk, dv)
    return dq, dk, dv


_registry.register_flash_attention_impl("FA4", register_fn=register_flash_attention_fa4)
<|endoftext|># mypy: allow-untyped-defs
"""This module contains functions and classes that alter the behavior of torch.nn.functional.scaled_dot_product_attention"""

import contextlib
from collections.abc import Iterable
from typing import Union
from warnings import warn

import torch.backends.cuda
from torch._C import _SDPBackend as SDPBackend
from torch.backends.cuda import (
    can_use_efficient_attention,
    can_use_flash_attention,
    SDPAParams,
)


__all__: list[str] = [
    "SDPBackend",
    "sdpa_kernel",
    "WARN_FOR_UNFUSED_KERNELS",
    "register_flash_attention_impl",
    "activate_flash_attention_impl",
    "list_flash_attention_impls",
    "current_flash_attention_impl",
    "restore_flash_attention_impl",
]


# Note: [SDPA warnings]
# TODO: Consider using this for sdpa regardless of subclasses
# This only affects users of bias subclasses
# If this is set to True, we will warn the user if they are not using the fused kernels
# As well, it will raise warnings for all the reasons why the fused kernels can't be run.
# To set this to True, run
# torch.nn.attention.WARN_FOR_UNFUSED_KERNELS = True
WARN_FOR_UNFUSED_KERNELS = False


r"""An enum-like class that contains the different backends for scaled dot product attention.
    This backend class is designed to be used with the sdpa_kernel context manager.

    The following Enums are available:
        - ERROR: An error occurred when trying to determine the backend.
        - MATH: The math backend for scaled dot product attention.
        - FLASH_ATTENTION: The flash attention backend for scaled dot product attention.
        - EFFICIENT_ATTENTION: The efficient attention backend for scaled dot product attention.
        - CUDNN_ATTENTION: The cuDNN backend for scaled dot product attention.
        - OVERRIDEABLE: The overridable backend for extension.

    See :func:`torch.nn.attention.sdpa_kernel` for more details.

    .. warning:: This class is in beta and subject to change.
"""
SDPBackend.__module__ = __name__
SDPBackend.__name__ = "SDPBackend"


def _raise_kernel_warnings(params: SDPAParams) -> None:
    """
    If WARN_FOR_UNFUSED_KERNELS is set to True, this will raise warnings
    for all the reasons why the fused kernels can't be run. If using subclasses
    """
    if WARN_FOR_UNFUSED_KERNELS:
        if not can_use_efficient_attention(params):
            warn("Efficient attention can't be used because:", stacklevel=2)
            can_use_efficient_attention(params, True)
        if not can_use_flash_attention(params):
            warn("Flash attention can't be used because:", stacklevel=2)
            can_use_flash_attention(params, True)


_backend_names = {
    "cudnn": "CUDNN_ATTENTION",
    "flash": "FLASH_ATTENTION",
    "mem_efficient": "EFFICIENT_ATTENTION",
    "math": "MATH",
    "overrideable": "OVERRIDEABLE",
}


def _backend_from_string(name: str):
    return getattr(SDPBackend, name)


def _cur_sdpa_kernel_backends(with_priority: bool = False):
    backends = []
    for name, val in _backend_names.items():
        if getattr(torch._C, f"_get_{name}_sdp_enabled")():
            backends.append(getattr(SDPBackend, val))
    if with_priority:
        curr_priority = torch._C._get_sdp_priority_order()
        backends = sorted(
            backends, key=lambda backend: curr_priority.index(int(backend))
        )
    return backends


def _sdpa_kernel(backends: Iterable, set_priority: bool = False) -> None:
    for name, val in _backend_names.items():
        enabled = getattr(SDPBackend, val) in backends
        getattr(torch._C, f"_set_sdp_use_{name}")(enabled)
    if set_priority:
        # backends should be a unique list
        user_priority = [int(backend) for backend in backends]
        previous_priority = torch._C._get_sdp_priority_order()
        for backend in previous_priority:
            if backend not in user_priority:
                user_priority.append(int(backend))
        torch._C._set_sdp_priority_order(user_priority)


@contextlib.contextmanager
def sdpa_kernel(backends: list[SDPBackend] | SDPBackend, set_priority: bool = False):
    r"""
    Context manager to select which backend to use for scaled dot product attention.

    .. warning:: This function is beta and subject to change.

    Args:
        backends (Union[List[SDPBackend], SDPBackend]): A backend or list of backends for scaled dot product attention.
        set_priority (bool=False): Whether the ordering of the backends is interpreted as their priority order.

    Example:

    .. code-block:: python

        from torch.nn.functional import scaled_dot_product_attention
        from torch.nn.attention import SDPBackend, sdpa_kernel

        # Only enable flash attention backend
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            scaled_dot_product_attention(...)

        # Enable the Math or Efficient attention backends
        with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
            scaled_dot_product_attention(...)

        # Enable the cuDNN or flash attention backends, and in that order
        with sdpa_kernel(
            [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION], set_priority=True
        ):
            scaled_dot_product_attention(...)

    This context manager can be used to select which backend to use for scaled dot product attention.
    Upon exiting the context manager, the previous state of the flags will be restored, enabling all backends.
    """
    if not isinstance(backends, (list, SDPBackend)):
        raise AssertionError(
            f"Backend must be an instance of SDPBackend or a list of SDPBackend instances, got {type(backends).__name__}"
        )

    if isinstance(backends, SDPBackend):
        backends = [backends]

    backends = list(dict.fromkeys(backends))

    previous_backends = _cur_sdpa_kernel_backends(with_priority=set_priority)
    try:
        _sdpa_kernel(backends, set_priority)
        yield {}
    finally:
        _sdpa_kernel(previous_backends, set_priority)


# variadic version of sdpa_kernel for dynamo to use while reconstructing
@contextlib.contextmanager
def _sdpa_kernel_variadic(*backends: SDPBackend):
    with sdpa_kernel(list(backends)):
        yield


def _get_flash_version() -> str:
    """This returns the closest matching tag for the flash attention backend"""
    return "2.5.7"


from . import _registry


# Re-export registry types and functions for public API
_FlashAttentionImpl = _registry._FlashAttentionImpl
_RegisterFn = _registry._RegisterFn
register_flash_attention_impl = _registry.register_flash_attention_impl
activate_flash_attention_impl = _registry.activate_flash_attention_impl
list_flash_attention_impls = _registry.list_flash_attention_impls
current_flash_attention_impl = _registry.current_flash_attention_impl
restore_flash_attention_impl = _registry.restore_flash_attention_impl

register_flash_attention_impl.__module__ = __name__
activate_flash_attention_impl.__module__ = __name__
list_flash_attention_impls.__module__ = __name__
current_flash_attention_impl.__module__ = __name__
restore_flash_attention_impl.__module__ = __name__

# Import built-in implementations to trigger self-registration
from . import _fa3, _fa4
<|endoftext|># mypy: allow-untyped-defs
"""This module implements the user facing API for flex_attention in PyTorch."""

from __future__ import annotations

import functools
import inspect
import itertools
import math
import operator
import types
import typing
import warnings
from collections.abc import Callable
from enum import Enum
from typing import Any, cast, Literal, NamedTuple, overload, TypeAlias, TypeVar
from typing_extensions import deprecated, Never, NotRequired, Self, TypedDict

import torch
from torch import Tensor
from torch._higher_order_ops.flex_attention import flex_attention as flex_attention_hop
from torch._higher_order_ops.utils import setup_compilation_env
from torch.fx.experimental.symbolic_shapes import has_free_unbacked_symbols
from torch.nn.attention._utils import _validate_sdpa_input
from torch.utils._pytree import (
    GetAttrKey,
    tree_flatten,
    tree_map_only,
    tree_unflatten,
    TreeSpec,
)


if typing.TYPE_CHECKING:
    from torch._prims_common import DeviceLikeType
    from torch.fx.node import BaseArgumentTypes


# Private debug flag to disable internal compilation wrapping for debugging purposes.
# WARNING: This is intended ONLY for debugging score_mod and mask_mod functions.
# When enabled, this bypasses the required internal compilation that ensures correctness
# and performance. Only use this temporarily when you need to set breakpoints
# in your score_mod/mask_mod functions during development.
#
# This flag only affects the internal compilation when flex_attention is called directly.
# If you have already wrapped flex_attention in torch.compile(), this flag has no effect
# and the user's compilation will still occur.
#
# Usage:
#   import torch.nn.attention.flex_attention as fa
#   fa._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True
#   # Now you can set breakpoints in your score_mod/mask_mod
#   output = fa.flex_attention(q, k, v, score_mod=my_score_mod)
#
_FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = False

_BLOCK_MASK_TOO_SMALL_ERROR = (
    "block_mask was created for a smaller length than you're "
    "using it for, you likely need to create a new block mask."
)
_BLOCK_MASK_TOO_LARGE_ERROR = (
    "block_mask was created for a larger length than you're "
    "using it for, you can either 1. create a new block mask with "
    "the correct length, or 2. 'adjust' the existing block mask to "
    "the correct length by calling block_mask._adjust(q_len, kv_len). "
    "This essentially 'crops' the block mask to the upper left corner, "
    "which does not work for all mask_mods!"
)

_WARNINGS_SHOWN: set[str] = set()


def _warn_once(
    warning_id: str, message: str, category: type[Warning] = UserWarning
) -> None:
    """Helper to ensure each warning is shown only once per process."""
    if warning_id not in _WARNINGS_SHOWN:
        if not torch.compiler.is_compiling():
            warnings.warn(message, category, stacklevel=2)
        _WARNINGS_SHOWN.add(warning_id)


def _validate_block_mask_shape(
    query: Tensor,
    key: Tensor,
    q_len: int | torch.SymInt,
    kv_len: int | torch.SymInt,
    block_mask_q_len: int | torch.SymInt,
    block_mask_kv_len: int | torch.SymInt,
) -> None:
    """Preserve unbacked checks without specializing regular dynamic lengths."""
    has_unbacked_input_lengths = has_free_unbacked_symbols(
        query
    ) or has_free_unbacked_symbols(key)
    if not torch.compiler.is_dynamo_compiling():
        lengths = (q_len, kv_len, block_mask_q_len, block_mask_kv_len)
        has_unbacked_input_lengths = (
            has_unbacked_input_lengths or has_free_unbacked_symbols(lengths)
        )

    # Case 1: unbacked lengths need symbolic checks that can become runtime assertions.
    if has_unbacked_input_lengths:
        torch._check(q_len <= block_mask_q_len, _BLOCK_MASK_TOO_SMALL_ERROR)
        torch._check(kv_len <= block_mask_kv_len, _BLOCK_MASK_TOO_SMALL_ERROR)
        torch._check(q_len >= block_mask_q_len, _BLOCK_MASK_TOO_LARGE_ERROR)
        torch._check(kv_len >= block_mask_kv_len, _BLOCK_MASK_TOO_LARGE_ERROR)
        return

    # Case 2: backed lengths use regular validation so Dynamo can generalize equality.
    if q_len > block_mask_q_len or kv_len > block_mask_kv_len:
        raise RuntimeError(_BLOCK_MASK_TOO_SMALL_ERROR)
    if q_len < block_mask_q_len or kv_len < block_mask_kv_len:
        raise RuntimeError(_BLOCK_MASK_TOO_LARGE_ERROR)

    if q_len != block_mask_q_len:
        raise AssertionError(
            f"query.size(-2) ({q_len}) != block_mask_q_len ({block_mask_q_len})"
        )
    if kv_len != block_mask_kv_len:
        raise AssertionError(
            f"key.size(-2) ({kv_len}) != block_mask_kv_len ({block_mask_kv_len})"
        )


__all__ = [
    "BlockMask",
    "flex_attention",
    "AuxOutput",
    "AuxRequest",
    "FlexKernelOptions",
    "create_block_mask",
    "create_mask",
    "or_masks",
    "and_masks",
    "noop_mask",
]

_score_mod_signature = Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]
_mask_mod_signature = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]
_Backend: TypeAlias = Literal["AUTO", "TRITON", "FLASH", "TRITON_DECODE"]
_R = TypeVar("_R")


class FlexKernelOptions(TypedDict, total=False):
    """Options for controlling the behavior of FlexAttention kernels.

    These options are passed to the underlying Triton kernels to control performance
    and numerical behavior. Most users will not need to specify these options as the
    default autotuning provides good performance.

    The options can be prefixed with ``fwd_`` or ``bwd_`` to apply only to forward or
    backward pass respectively. For example: ``fwd_BLOCK_M`` and ``bwd_BLOCK_M1``.

    Note:
      We currently do not provide any backward compatibility guarantees for these options.
      That being said most of these have remained pretty stable since their introduction. But
      We do not consider this part of the public API just yet. We think that some documentation
      Is better than secret hidden flags, but we may change these options in the future.

    Example Usage:
        .. code-block:: python

            # Using dictionary (backward compatible)
            kernel_opts = {"BLOCK_M": 64, "BLOCK_N": 64, "PRESCALE_QK": True}
            output = flex_attention(q, k, v, kernel_options=kernel_opts)

            # Using TypedDict (recommended for type safety)
            from torch.nn.attention.flex_attention import FlexKernelOptions

            kernel_opts: FlexKernelOptions = {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "PRESCALE_QK": True,
            }
            output = flex_attention(q, k, v, kernel_options=kernel_opts)

            # Forward/backward specific options
            kernel_opts: FlexKernelOptions = {
                "fwd_BLOCK_M": 64,
                "bwd_BLOCK_M1": 32,
                "PRESCALE_QK": False,
            }
            output = flex_attention(q, k, v, kernel_options=kernel_opts)
    """

    # Performance tuning options

    num_warps: NotRequired[int]
    """Number of warps to use in the CUDA kernel. Higher values may improve performance
    but increase register pressure. Default is determined by autotuning."""

    num_stages: NotRequired[int]
    """Number of pipeline stages in the CUDA kernel. Higher values may improve performance
    but increase shared memory usage. Default is determined by autotuning."""

    BLOCK_M: NotRequired[int]
    """Thread block size for the sequence length dimension of Q in forward pass.
    Must be a power of 2. Common values: 16, 32, 64, 128. Default is determined by autotuning."""

    BLOCK_N: NotRequired[int]
    """Thread block size for the sequence length dimension of K/V in forward pass.
    Must be a power of 2. Common values: 16, 32, 64, 128. Default is determined by autotuning."""

    # Backward-specific block sizes (when prefixed with 'bwd_')

    BLOCK_M1: NotRequired[int]
    """Thread block size for Q dimension in backward pass. Use as 'bwd_BLOCK_M1'.
    Default is determined by autotuning."""

    BLOCK_N1: NotRequired[int]
    """Thread block size for K/V dimension in backward pass. Use as 'bwd_BLOCK_N1'.
    Default is determined by autotuning."""

    BLOCK_M2: NotRequired[int]
    """Thread block size for second Q dimension in backward pass. Use as 'bwd_BLOCK_M2'.
    Default is determined by autotuning."""

    BLOCK_N2: NotRequired[int]
    """Thread block size for second K/V dimension in backward pass. Use as 'bwd_BLOCK_N2'.
    Default is determined by autotuning."""

    PRESCALE_QK: NotRequired[bool]
    """Whether to pre-scale QK by 1/sqrt(d) and change of base. This is slightly faster but
    may have more numerical error. Default: False."""

    ROWS_GUARANTEED_SAFE: NotRequired[bool]
    """If True, guarantees that at least one value in each row is not masked out.
    Allows skipping safety checks for better performance. Only set this if you are certain
    your mask guarantees this property. For example, causal attention is guaranteed safe
    because each query has at least 1 key-value to attend to. Default: False."""

    BLOCKS_ARE_CONTIGUOUS: NotRequired[bool]
    """If True, guarantees that all blocks in the mask are contiguous.
    Allows optimizing block traversal. For example, causal masks would satisfy this,
    but prefix_lm + sliding window would not. Default: False."""

    WRITE_DQ: NotRequired[bool]
    """Controls whether gradient scatters are done in the DQ iteration loop of the backward pass.
    Setting this to False will force this to happen in the DK loop which depending on your
    specific score_mod and mask_mod might be faster. Default: True."""

    FORCE_USE_FLEX_ATTENTION: NotRequired[bool]
    """If True, forces the use of the flex attention kernel instead of potentially using
    the more optimized flex-decoding kernel for short sequences. This can be a helpful
    option for debugging. Default: False."""

    USE_TMA: NotRequired[bool]
    """Whether to use Tensor Memory Accelerator (TMA) on supported hardware.
    This is experimental and may not work on all hardware, currently specific
    to NVIDIA GPUs Hopper+. Default: False."""

    # ROCm-specific options

    kpack: NotRequired[int]
    """ROCm-specific kernel packing parameter."""

    matrix_instr_nonkdim: NotRequired[int]
    """ROCm-specific matrix instruction non-K dimension."""

    waves_per_eu: NotRequired[int]
    """ROCm-specific waves per execution unit."""

    BACKEND: NotRequired[_Backend]
    """Selects a specific kernel backend.

    Options:
        - "AUTO": Use current heuristics (typically Triton-based kernels with
          automatic selection between flex_attention and flex_decoding)
        - "TRITON": Standard Triton flex_attention kernel
        - "TRITON_DECODE": Triton flex_decoding kernel, only available for short sequence lengths with specific configurations
        - "FLASH": Experimental: Flash Attention kernel (cute-dsl), user needs to have flash installed

    This option cannot be combined with legacy knobs such as ``FORCE_USE_FLEX_ATTENTION``.
    Raises an error if the requested backend cannot be used. Default: "AUTO"
    """


class _KernelOptionsWithInternals(FlexKernelOptions, total=False):
    OUTPUT_LOGSUMEXP: bool
    OUTPUT_MAX: bool


class AuxRequest(NamedTuple):
    """Request which auxiliary outputs to compute from flex_attention.

    Each field is a boolean indicating whether that auxiliary output should be computed.
    """

    lse: bool = False
    max_scores: bool = False


class AuxOutput(NamedTuple):
    """Auxiliary outputs from flex_attention operation.

    Fields will be None if not requested, or contain the tensor if requested.
    """

    lse: Tensor | None = None
    max_scores: Tensor | None = None


class _ModificationType(Enum):
    """Enum for the type of modification function.
    - SCORE_MOD: score_mod function which accepts a score as the first argument
    - mask_mod: mask function which does not accept a score and is only used for generating
    block mask
    """

    SCORE_MOD = 1
    MASK_MOD = 2
    UNKNOWN = 3


def _get_mod_type(
    fn: _score_mod_signature | _mask_mod_signature | Callable[..., Any],
) -> _ModificationType:
    """Get the type of modification function.
    This function inspects the number of positional arguments of the function to determine
    the type of modification function. If the function has 5 positional arguments, it is
    considered as a score_mod function. If the function has 4 positional arguments, it is
    considered as a mask function.
    """
    if hasattr(fn, "__code__"):
        code = fn.__code__
        num_positional_total = code.co_argcount
        defaults = ()
        if hasattr(fn, "__defaults__"):
            defaults = fn.__defaults__ or ()
        num_defaults = len(defaults)
        num_positional_args = num_positional_total - num_defaults
    else:
        num_positional_args = sum(
            1
            for param in inspect.signature(fn).parameters.values()
            if param.default is inspect.Parameter.empty
        )
    if num_positional_args != 5 and num_positional_args != 4:
        raise AssertionError(
            f"Expected 4 or 5 positional args, got {num_positional_args}"
        )
    if num_positional_args == 5:
        return _ModificationType.SCORE_MOD
    elif num_positional_args == 4:
        return _ModificationType.MASK_MOD
    else:
        return _ModificationType.UNKNOWN


# Need to define it here so that Dynamo doesn't skip it
def _vmap_for_bhqkv(
    fn: Callable[..., _R],
    prefix: tuple[int | None, ...],
    suffix: tuple[int | None, ...] = (),
    out_dims: int | list[int | None] = 0,
    group_dim: bool = False,
) -> Callable[..., _R]:
    """Used to vmap both score_mods and mask_mods over 4-dimensional/5-dimension inputs.
    Mapping over the [b, hq, q_idx, kv_idx] or [b, hkv, g, q_idx, kv_idx] dimensions.

    Args:
        fn (callable): The function to vmap.
        prefix (tuple): The prefix of the vmap. For score mod functions,
                        this should be set to (0,). For mask_mods = ()
        suffix (tuple): We need to add (0,) if gradOut is being mapped over,
                        and (None,) * len(other_buffers).
        out_dims (tuple): For forward cases, keep this as the default 0 since
                          we are only returning 1 output. For backwards, the joint
                          graph returns grads for B, H, Q_idx, KV_idx and other_buffers,
                          so we set this to (0, None, None, None, None) + (None,) * len(other_buffers).

    Returns:
        callable: The vmapped function.
    """
    # We vamp a function 4 times, broadcasting the [b, h, q_idx, kv_idx] dimensions
    dimensions: list[tuple[int | None, int | None, int | None, int | None]] = []
    dimensions = [
        (None, None, None, 0),
        (None, None, 0, None),
        (None, 0, None, None),
    ]

    if group_dim:
        dimensions += [
            (None, 0, None, None),
        ]

    dimensions += [
        (0, None, None, None),
    ]

    for dims in dimensions:
        fn = torch.vmap(fn, in_dims=prefix + dims + suffix, out_dims=out_dims)  # type: ignore[arg-type]
    return fn


def _identity(
    score: Tensor,
    batch: Tensor,
    head: Tensor,
    token_q: Tensor,
    token_kv: Tensor,
) -> Tensor:
    return score


def noop_mask(
    batch: Tensor,
    head: Tensor,
    token_q: Tensor,
    token_kv: Tensor,
) -> Tensor:
    """Returns a noop mask_mod"""
    return batch.new_ones(size=(), dtype=torch.bool, device=batch.device)


def _sliced_mask_mod_error(
    batch: Tensor,
    head: Tensor,
    token_q: Tensor,
    token_kv: Tensor,
) -> Never:
    """
    Raises helpful error when using mask_mod from a sliced BlockMask.

    After slicing a BlockMask, the mask_mod is reset and cannot be used directly.
    Users must reassign mask_mod from the original (unsliced) BlockMask.
    """
    raise RuntimeError(
        "Cannot use mask_mod from a sliced BlockMask. "
        "When you slice a BlockMask using [], the mask_mod attribute is reset. "
        "You must set it from the original BlockMask's mask_mod."
        "\n\nIncorrect usage:"
        "\n  base_mask = create_block_mask(my_mask_fn, ...)"
        "\n  sliced_mask = base_mask[:, :, block_idx]"
        "\n  sliced_mask.mask_mod = apply_offset(sliced_mask.mask_mod, offset)  # WRONG!"
        "\n\nCorrect usage:"
        "\n  base_mask = create_block_mask(my_mask_fn, ...)"
        "\n  sliced_mask = base_mask[:, :, block_idx]"
        "\n  sliced_mask.mask_mod = apply_offset(base_mask.mask_mod, offset)  # Use base_mask!"
    )


_DEFAULT_SPARSE_BLOCK_SIZE = 128
_LARGE_SPARSE_BLOCK_SIZE = 1 << 30


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _ordered_to_dense(
    num_blocks_in_row: Tensor, col_indices: Tensor, num_cols: int
) -> Tensor:
    num_rows = col_indices.shape[-2]
    max_entries = col_indices.shape[-1]
    batch_dims = num_blocks_in_row.shape[:-1]
    device = num_blocks_in_row.device

    def create_dense_one(kv_num_blocks, kv_indices):
        dense_mask = kv_indices.new_zeros(num_rows, num_cols + 1, dtype=torch.int32)

        row_indices = torch.arange(num_rows, dtype=torch.int, device=device).unsqueeze(
            -1
        )
        col_range = torch.arange(max_entries, dtype=torch.int, device=device)
        index_mask = col_range < kv_num_blocks.unsqueeze(-1)

        # We write to one spot "out of bounds"
        valid_indices = torch.where(index_mask, kv_indices, num_cols)

        # set the values in 'a' to 1 where the indices are valid
        dense_mask[row_indices, valid_indices] = dense_mask.new_ones(())
        return dense_mask[:, :num_cols].contiguous()

    create_dense_batched = create_dense_one
    for _ in range(len(batch_dims)):
        create_dense_batched = torch.vmap(create_dense_batched, in_dims=(0, 0))

    out = create_dense_batched(num_blocks_in_row, col_indices)
    return out


def _dense_to_ordered(dense_mask: Tensor) -> tuple[Tensor, Tensor]:
    dense_mask = dense_mask.to(dtype=torch.int32)
    num_blocks_in_row = dense_mask.sum(dim=-1)
    with torch.fx.traceback.annotate({"fallback_to_eager": True}):
        col_indices = torch.argsort(dense_mask, dim=-1, descending=True, stable=True)
    return (
        num_blocks_in_row.to(torch.int32, memory_format=torch.contiguous_format),
        col_indices.to(torch.int32, memory_format=torch.contiguous_format),
    )


def _transpose_ordered(
    num_blocks_in_row: Tensor, col_indices: Tensor, num_cols: int | None = None
) -> tuple[Tensor, Tensor]:
    dense = _ordered_to_dense(
        num_blocks_in_row,
        col_indices,
        col_indices.shape[-1] if num_cols is None else num_cols,
    )
    return _dense_to_ordered(dense.transpose(-2, -1))


def _adjust_num_blocks_and_indices(
    num_blocks: Tensor,
    indices: Tensor,
    new_num_rows: int,
    new_num_cols: int,
) -> tuple[Tensor, Tensor]:
    """Crop an ordered block list while ignoring undefined entries past num_blocks."""
    indices = indices[:, :, :new_num_rows, :new_num_cols]
    num_blocks = num_blocks[:, :, :new_num_rows]
    valid_entries = (
        torch.arange(indices.shape[-1], dtype=num_blocks.dtype, device=indices.device)
        < num_blocks[..., None]
    )
    num_blocks = torch.sum(valid_entries & (indices < new_num_cols), dim=-1).to(
        torch.int32
    )
    return num_blocks, indices


# TreeSpec for an empty tuple — used as the sentinel when there are no closure leaves.
_EMPTY_CLOSURE_SPEC = tree_flatten(())[1]


class _ExtractedLeaf:
    """Sentinel in _StrippedClosure.leaf_entries marking a position that is
    filled from the extracted pytree leaves list during reconstruction."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "_EXTRACTED_LEAF"


_EXTRACTED_LEAF = _ExtractedLeaf()


class _CallableLeaf(typing.NamedTuple):
    """Entry in stripped callable metadata for a recursively processed
    nested callable. Stores enough information to reconstruct it from the
    extracted leaves during unflattening."""

    stripped: _CallableMetadata | Callable[..., Any]
    callable_spec: TreeSpec
    n_extracted: int  # number of extracted leaves this function contributes

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _CallableLeaf):
            return False
        return (
            _callable_entry_eq(self.stripped, other.stripped)
            and self.callable_spec == other.callable_spec
            and self.n_extracted == other.n_extracted
        )

    def __hash__(self) -> int:
        return hash(
            (
                _callable_entry_hash(self.stripped),
                self.callable_spec,
                self.n_extracted,
            )
        )


class _StrippedClosure(typing.NamedTuple):
    """Data container holding the parts of a function needed for reconstruction.

    Created by _extract_callable_pytree when closure tensors are lifted into
    pytree leaves.  Unlike a FunctionType with None-filled cells, this is not
    callable — it is pure data stored in the pytree context.
    """

    code: types.CodeType
    globals_dict: dict[str, Any]
    name: str
    qualname: str
    defaults: tuple[Any, ...] | None
    kwdefaults: dict[str, Any] | None
    extra_dict: dict[str, Any]
    # Per-position info for the closure's flattened leaves.
    # _EXTRACTED_LEAF → position filled from the extracted leaves list.
    # _CallableLeaf   → recursively processed nested callable.
    leaf_entries: tuple[_ExtractedLeaf | _CallableLeaf, ...]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _StrippedClosure):
            return False
        return self.code == other.code and self.leaf_entries == other.leaf_entries

    def __hash__(self) -> int:
        return hash((self.code, self.leaf_entries))


class _StrippedPartial(typing.NamedTuple):
    """Data container holding the parts of a functools.partial needed for reconstruction."""

    leaf_entries: tuple[_ExtractedLeaf | _CallableLeaf, ...]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _StrippedPartial):
            return False
        return self.leaf_entries == other.leaf_entries

    def __hash__(self) -> int:
        return hash(self.leaf_entries)


class _PlainFunction(typing.NamedTuple):
    fn: Callable[..., Any]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _PlainFunction):
            return False
        return self.fn.__code__ == other.fn.__code__

    def __hash__(self) -> int:
        return hash(self.fn.__code__)


_CallableMetadata: TypeAlias = _StrippedClosure | _StrippedPartial | _PlainFunction


def _callable_entry_eq(
    lhs: _CallableMetadata | types.FunctionType,
    rhs: _CallableMetadata | types.FunctionType,
) -> bool:
    if inspect.isfunction(lhs) and inspect.isfunction(rhs):
        return lhs.__code__ == rhs.__code__
    return lhs == rhs


def _callable_entry_hash(value: _CallableMetadata | types.FunctionType) -> int:
    if inspect.isfunction(value):
        return hash(value.__code__)
    return hash(value)


def _extract_callable_leaves(
    leaves: list[Any], _seen: set[int]
) -> tuple[tuple[BaseArgumentTypes, ...], tuple[_ExtractedLeaf | _CallableLeaf, ...]]:
    extracted: list[BaseArgumentTypes] = []
    leaf_entries: list[_ExtractedLeaf | _CallableLeaf] = []
    for leaf in leaves:
        if inspect.isfunction(leaf) or isinstance(leaf, functools.partial):
            child_extracted, child_spec, child_stripped = _extract_callable_pytree(
                leaf, _seen
            )
            if not isinstance(
                child_stripped, (_StrippedClosure, _StrippedPartial, _PlainFunction)
            ) and not inspect.isfunction(child_stripped):
                raise AssertionError(
                    "expected nested callable extraction to produce callable metadata"
                )
            extracted.extend(child_extracted)
            leaf_entries.append(
                _CallableLeaf(child_stripped, child_spec, len(child_extracted))
            )
        else:
            extracted.append(leaf)
            leaf_entries.append(_EXTRACTED_LEAF)
    return tuple(extracted), tuple(leaf_entries)


def _extract_callable_pytree(
    fn, _seen: set[int] | None = None
) -> tuple[
    tuple[BaseArgumentTypes, ...],
    TreeSpec,
    _CallableMetadata | Callable[..., Any],
]:
    """Extract closure contents as a flattened sub-pytree.

    Returns (extracted_leaves, callable_spec, fn_or_stripped) where:
    - extracted_leaves: flattened non-function contents from the closure,
      functools.partial payload, plus any tensors/scalars recursively extracted
      from nested function closures
    - callable_spec: TreeSpec describing how to reconstruct the callable state
    - fn_or_stripped: either the original fn (skipped extraction) or a
      stripped callable carrying the parts needed for reconstruction

    Functions found among the closure or partial leaves are recursively
    processed: their own closure tensors are extracted into the leaves list,
    and their skeleton is stored in the stripped metadata. All other values
    (tensors, scalars, None, etc.) remain as extracted leaves.

    If fn is not a plain function or functools.partial, returns the original
    function unchanged with no closure leaves. Plain functions without a
    closure are wrapped as _PlainFunction so structural equality/hash can use
    their __code__.

    Skipped under Dynamo tracing (torch.compiler.is_compiling) because Dynamo
    can't trace through closure cell introspection and handles freevars via its
    own lifting mechanism.
    """
    if torch.compiler.is_compiling() or not (
        inspect.isfunction(fn) or isinstance(fn, functools.partial)
    ):
        return (), _EMPTY_CLOSURE_SPEC, fn

    # Cycle detection for self-referencing closures.
    if _seen is None:
        _seen = set()
    if id(fn) in _seen:
        return (), _EMPTY_CLOSURE_SPEC, fn
    _seen.add(id(fn))

    if isinstance(fn, functools.partial):
        partial_leaves, partial_spec = tree_flatten((fn.func, fn.args, fn.keywords))
        extracted, leaf_entries = _extract_callable_leaves(partial_leaves, _seen)
        return tuple(extracted), partial_spec, _StrippedPartial(tuple(leaf_entries))

    closure = fn.__closure__
    if not closure:
        return (), _EMPTY_CLOSURE_SPEC, _PlainFunction(fn)

    try:
        contents = tuple(cell.cell_contents for cell in closure)
    except ValueError:
        # Empty cell (created but not yet assigned) — can't extract
        return (), _EMPTY_CLOSURE_SPEC, _PlainFunction(fn)

    closure_leaves, callable_spec = tree_flatten(contents)

    extracted, leaf_entries = _extract_callable_leaves(closure_leaves, _seen)

    stripped = _StrippedClosure(
        code=fn.__code__,
        globals_dict=fn.__globals__,
        name=fn.__name__,
        qualname=fn.__qualname__,
        defaults=fn.__defaults__,
        kwdefaults=fn.__kwdefaults__,
        extra_dict=dict(fn.__dict__) if fn.__dict__ else {},
        leaf_entries=tuple(leaf_entries),
    )

    return tuple(extracted), callable_spec, stripped


def _reconstruct_closure_fn(stripped, extracted_leaves, callable_spec):
    """Rebuild a stripped callable from flattened extracted leaves."""
    if isinstance(stripped, _PlainFunction):
        return stripped.fn
    if not isinstance(stripped, (_StrippedClosure, _StrippedPartial)):
        return stripped

    all_leaves: list[BaseArgumentTypes | Callable[..., Any]] = []
    idx = 0
    for entry in stripped.leaf_entries:
        if isinstance(entry, _CallableLeaf):
            child_fn = _reconstruct_closure_fn(
                entry.stripped,
                extracted_leaves[idx : idx + entry.n_extracted],
                entry.callable_spec,
            )
            all_leaves.append(child_fn)
            idx += entry.n_extracted
        else:
            # _EXTRACTED_LEAF — take from extracted leaves
            all_leaves.append(extracted_leaves[idx])
            idx += 1

    if isinstance(stripped, _StrippedPartial):
        fn, args, keywords = tree_unflatten(all_leaves, callable_spec)
        return functools.partial(fn, *args, **keywords)

    contents = tree_unflatten(all_leaves, callable_spec)
    new_cells = tuple(types.CellType(v) for v in contents)

    restored = types.FunctionType(
        stripped.code,
        stripped.globals_dict,
        stripped.name,
        stripped.defaults,
        new_cells,
    )
    restored.__qualname__ = stripped.qualname
    if stripped.kwdefaults:
        restored.__kwdefaults__ = stripped.kwdefaults
    if stripped.extra_dict:
        restored.__dict__.update(stripped.extra_dict)

    return restored


class _MaskModWrapper:
    """Wraps a mask_mod or stripped callable metadata with value-based equality.

    BlockMask stores an arbitrary callable (mask_mod) in its pytree context.
    The default __eq__ for functions uses identity comparison, which is too
    strict when the same closure is recreated (e.g., defined inside forward()).

    When callable state has been extracted (by _extract_callable_pytree), fn is
    stripped metadata (pure data, not callable). Equality compares the stripped
    callable structure + callable_spec without triggering tensor dispatch.

    When extraction is skipped (e.g., under Dynamo), fn is the original
    callable and equality compares code objects for plain functions or
    delegates to __eq__ for callable objects.
    """

    __slots__ = ("fn", "callable_spec")

    def __init__(self, fn, callable_spec=None) -> None:
        self.fn = fn
        self.callable_spec = callable_spec

    def __call__(self, b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        if isinstance(self.fn, (_StrippedClosure, _StrippedPartial, _PlainFunction)):
            raise RuntimeError(
                "_MaskModWrapper with stripped callable is not callable — "
                "use _reconstruct_closure_fn to rebuild the function first"
            )
        return self.fn(b, h, q_idx, kv_idx)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _MaskModWrapper):
            return False
        if self.fn is other.fn and self.callable_spec is other.callable_spec:
            return True
        if isinstance(
            self.fn, (_StrippedClosure, _StrippedPartial, _PlainFunction)
        ) and isinstance(
            other.fn, (_StrippedClosure, _StrippedPartial, _PlainFunction)
        ):
            return self.fn == other.fn and self.callable_spec == other.callable_spec
        # Non-extracted plain functions: compare code objects
        if inspect.isfunction(self.fn) and inspect.isfunction(other.fn):
            return self.fn.__code__ == other.fn.__code__
        # Callable objects: delegate to their __eq__
        if not isinstance(
            self.fn, (_StrippedClosure, _StrippedPartial, _PlainFunction)
        ) and not isinstance(
            other.fn, (_StrippedClosure, _StrippedPartial, _PlainFunction)
        ):
            return self.fn == other.fn
        return False

    def __hash__(self) -> int:
        if isinstance(self.fn, (_StrippedClosure, _StrippedPartial, _PlainFunction)):
            return hash((self.fn, self.callable_spec))
        if inspect.isfunction(self.fn):
            return hash(self.fn.__code__)
        return hash(self.fn)

    def __repr__(self) -> str:
        return f"_MaskModWrapper({self.fn})"


class BlockMask:
    r"""
    BlockMask is our format for representing a block-sparse attention mask.
    It is somewhat of a cross in-between BCSR and a non-sparse format.

    **Basics**

    A block-sparse mask means that instead of representing the sparsity of
    individual elements in the mask, a KV_BLOCK_SIZE x Q_BLOCK_SIZE block is
    considered sparse only if every element within that block is sparse.
    This aligns well with hardware, which generally expects to perform
    contiguous loads and computation.

    This format is primarily optimized for 1. simplicity, and 2. kernel
    efficiency. Notably, it is *not* optimized for size, as this mask is always
    reduced by a factor of KV_BLOCK_SIZE * Q_BLOCK_SIZE. If the size is a
    concern, the tensors can be reduced in size by increasing the block size.

    The essentials of our format are:

    num_blocks_in_row: Tensor[ROWS]:
    Describes the number of blocks present in each row.

    col_indices: Tensor[ROWS, MAX_BLOCKS_IN_COL]:
    `col_indices[i]` is the sequence of block positions for row i. The values of
    this row after `col_indices[i][num_blocks_in_row[i]]` are undefined.

    For example, to reconstruct the original tensor from this format:

    .. code-block:: python

        dense_mask = torch.zeros(ROWS, COLS)
        for row in range(ROWS):
            for block_idx in range(num_blocks_in_row[row]):
                dense_mask[row, col_indices[row, block_idx]] = 1

    Notably, this format makes it easier to implement a reduction along the
    *rows* of the mask.

    **Details**

    The basics of our format require only kv_num_blocks and kv_indices. The
    primary block-sparse layout is represented by up to 4 tensor pairs:

    1. (kv_num_blocks, kv_indices): Used for the forwards pass of attention, as
    we reduce along the KV dimension.

    2. [OPTIONAL] (full_kv_num_blocks, full_kv_indices): This is optional and
    purely an optimization. As it turns out, applying masking to every block
    is quite expensive! If we specifically know which blocks are "full" and
    don't require masking at all, then we can skip applying mask_mod to these
    blocks. This requires the user to split out a separate mask_mod from the
    score_mod. For causal masks, this is about a 15% speedup.

    3. [GENERATED] (q_num_blocks, q_indices): Required for the backwards pass,
    as computing dKV requires iterating along the mask along the Q dimension. These are autogenerated from 1.

    4. [GENERATED] (full_q_num_blocks, full_q_indices): Same as above, but for
    the backwards pass. These are autogenerated from 2.

    Additional optional tensors may carry deterministic dQ metadata for
    block-sparse FLASH backward:

    5. [OPTIONAL] dq_write_order: Write-order metadata for partial blocks. This
    is produced by create_block_mask when compute_dq_write_order=True, or passed
    directly to BlockMask.from_kv_blocks by callers that precompute it.

    6. [OPTIONAL] dq_write_order_full: Write-order metadata for full blocks,
    produced or passed the same way as dq_write_order.

    7. [OPTIONAL] dq_kv_order: Explicit KV scheduler order used to produce the
    write-order metadata. create_block_mask currently accepts a boolean
    dq_kv_order; BlockMask.from_kv_blocks also accepts a tensor for callers that
    provide precomputed write-order metadata directly.
    """

    seq_lengths: tuple[int, int]
    kv_num_blocks: Tensor
    kv_indices: Tensor
    full_kv_num_blocks: Tensor | None
    full_kv_indices: Tensor | None
    q_num_blocks: Tensor | None
    q_indices: Tensor | None
    full_q_num_blocks: Tensor | None
    full_q_indices: Tensor | None
    dq_write_order: Tensor | None
    dq_write_order_full: Tensor | None
    dq_kv_order: Tensor | None
    dq_kv_order_spt: bool | None
    BLOCK_SIZE: tuple[int, int]
    mask_mod: _mask_mod_signature

    # Attribute lists for pytree flatten/unflatten
    _TENSOR_ATTRS = [
        "kv_num_blocks",
        "kv_indices",
        "full_kv_num_blocks",
        "full_kv_indices",
        "q_num_blocks",
        "q_indices",
        "full_q_num_blocks",
        "full_q_indices",
        "dq_write_order",
        "dq_write_order_full",
        "dq_kv_order",
    ]

    _CONTEXT_ATTRS = [
        "seq_lengths",
        "BLOCK_SIZE",
        "mask_mod",
        "dq_kv_order_spt",
    ]

    def __init__(
        self,
        seq_lengths: tuple[int, int],
        kv_num_blocks: Tensor,
        kv_indices: Tensor,
        full_kv_num_blocks: Tensor | None,
        full_kv_indices: Tensor | None,
        q_num_blocks: Tensor | None,
        q_indices: Tensor | None,
        full_q_num_blocks: Tensor | None,
        full_q_indices: Tensor | None,
        BLOCK_SIZE: tuple[int, int] = (
            _DEFAULT_SPARSE_BLOCK_SIZE,
            _DEFAULT_SPARSE_BLOCK_SIZE,
        ),
        mask_mod: _mask_mod_signature = noop_mask,
        *,
        dq_write_order: Tensor | None = None,
        dq_write_order_full: Tensor | None = None,
        dq_kv_order: Tensor | None = None,
        dq_kv_order_spt: bool | None = None,
    ) -> None:
        if kv_indices.dim() < 2:
            raise RuntimeError("BlockMask must have at least 2 dimensions")
        if kv_num_blocks is None:
            raise AssertionError("kv_num_blocks must be provided")
        if kv_indices is None:
            raise AssertionError("kv_indices must be provided")
        if (full_kv_num_blocks is None) != (full_kv_indices is None):
            raise AssertionError(
                "full_kv_num_blocks and full_kv_indices must be both provided or omitted"
            )
        if (full_q_num_blocks is None) != (full_q_indices is None):
            raise AssertionError(
                "full_q_num_blocks and full_q_indices must be both provided or omitted"
            )

        self.seq_lengths = seq_lengths
        self.kv_num_blocks = kv_num_blocks
        self.kv_indices = kv_indices
        self.full_kv_num_blocks = full_kv_num_blocks
        self.full_kv_indices = full_kv_indices
        self.q_num_blocks = q_num_blocks
        self.q_indices = q_indices
        self.full_q_num_blocks = full_q_num_blocks
        self.full_q_indices = full_q_indices
        if dq_write_order_full is not None and dq_write_order is None:
            raise ValueError("dq_write_order_full requires dq_write_order")

        self.dq_write_order = dq_write_order
        self.dq_write_order_full = dq_write_order_full
        self.dq_kv_order = dq_kv_order
        self.dq_kv_order_spt = dq_kv_order_spt
        self.BLOCK_SIZE = BLOCK_SIZE
        self.mask_mod = mask_mod

    def _dq_kv_order(self) -> Tensor | bool | None:
        if self.dq_kv_order is not None:
            return self.dq_kv_order
        return self.dq_kv_order_spt

    @staticmethod
    def _slice_dq_kv_order(
        dq_kv_order: Tensor | None,
        index: tuple[int | slice | Tensor, ...],
    ) -> Tensor | None:
        if dq_kv_order is None:
            return None
        return dq_kv_order[index[:2] + (slice(None),)]

    def _query_length_for_sliced_blocks(
        self, q_index: int | slice | Tensor, selected_q_blocks: int
    ) -> int:
        """Maps a Q-block selection to its packed logical token length."""
        q_block_size = self.BLOCK_SIZE[0]
        if not isinstance(q_index, slice):
            return selected_q_blocks * q_block_size

        selected_blocks = range(self.kv_num_blocks.shape[-1])[q_index]
        if len(selected_blocks) == 0:
            return 0
        if selected_blocks.step == 1:
            return min(selected_blocks.stop * q_block_size, self.seq_lengths[0]) - min(
                selected_blocks.start * q_block_size, self.seq_lengths[0]
            )

        q_length = len(selected_blocks) * q_block_size
        trailing_q_tokens = self.seq_lengths[0] % q_block_size
        if trailing_q_tokens and self.kv_num_blocks.shape[-1] - 1 in selected_blocks:
            q_length -= q_block_size - trailing_q_tokens
        return q_length

    @classmethod
    def from_kv_blocks(
        cls,
        kv_num_blocks: Tensor,
        kv_indices: Tensor,
        full_kv_num_blocks: Tensor | None = None,
        full_kv_indices: Tensor | None = None,
        BLOCK_SIZE: int | tuple[int, int] = _DEFAULT_SPARSE_BLOCK_SIZE,
        mask_mod: _mask_mod_signature | None = None,
        seq_lengths: tuple[int, int] | None = None,
        compute_q_blocks: bool = True,
        *,
        dq_write_order: Tensor | None = None,
        dq_write_order_full: Tensor | None = None,
        dq_kv_order: Tensor | bool | None = None,
    ) -> Self:
        """
        Creates a BlockMask instance from key-value block information.

        Args:
            kv_num_blocks (Tensor): Number of kv_blocks in each Q_BLOCK_SIZE row tile.
            kv_indices (Tensor): Indices of key-value blocks in each Q_BLOCK_SIZE row tile.
            full_kv_num_blocks (Optional[Tensor]): Number of full kv_blocks in each Q_BLOCK_SIZE row tile.
            full_kv_indices (Optional[Tensor]): Indices of full key-value blocks in each Q_BLOCK_SIZE row tile.
            BLOCK_SIZE (Union[int, tuple[int, int]]): Size of KV_BLOCK_SIZE x Q_BLOCK_SIZE tiles.
            mask_mod (Optional[Callable]): Function to modify the mask.
            dq_write_order (Optional[Tensor]): Precomputed deterministic dQ write-order metadata.
            dq_write_order_full (Optional[Tensor]): Precomputed deterministic dQ write-order metadata for full blocks.
            dq_kv_order (Optional[Union[Tensor, bool]]): KV-column scheduler order used to produce dq_write_order. A bool selects a built-in order; a tensor gives an explicit scheduler-rank to n-block permutation.

        Returns:
            BlockMask: Instance with full Q information generated via _transposed_ordered

        Raises:
            RuntimeError: If kv_indices has < 2 dimensions.
            AssertionError: If only one of full_kv_* args is provided.
        """
        if kv_indices.dim() < 2:
            raise RuntimeError("BlockMask must have at least 2 dimensions")

        if (full_kv_num_blocks is None) != (full_kv_indices is None):
            raise AssertionError(
                "full_kv_num_blocks and full_kv_indices must be both provided or omitted"
            )

        if isinstance(BLOCK_SIZE, int):
            BLOCK_SIZE = (BLOCK_SIZE, BLOCK_SIZE)

        mask_mod = mask_mod if mask_mod is not None else noop_mask
        if seq_lengths is None:
            q_length = kv_indices.shape[-2] * BLOCK_SIZE[0]
            kv_length = kv_indices.shape[-1] * BLOCK_SIZE[1]
            seq_lengths = (q_length, kv_length)
        kv_num_cols = (seq_lengths[1] + BLOCK_SIZE[1] - 1) // BLOCK_SIZE[1]

        # Generate q_num_blocks and q_indices
        if compute_q_blocks:
            q_num_blocks, q_indices = _transpose_ordered(
                kv_num_blocks, kv_indices, kv_num_cols
            )
            if full_kv_num_blocks is not None:
                if full_kv_indices is None:
                    raise AssertionError("full_kv_indices must not be None")
                full_q_num_blocks, full_q_indices = _transpose_ordered(
                    full_kv_num_blocks, full_kv_indices, kv_num_cols
                )
            else:
                full_q_num_blocks, full_q_indices = None, None
        else:
            q_num_blocks, q_indices = None, None
            full_q_num_blocks, full_q_indices = None, None

        if dq_kv_order is not None and not isinstance(dq_kv_order, (bool, Tensor)):
            raise ValueError("dq_kv_order must be a bool, Tensor, or None")
        dq_kv_order_tensor = dq_kv_order if isinstance(dq_kv_order, Tensor) else None
        dq_kv_order_spt = dq_kv_order if isinstance(dq_kv_order, bool) else None

        return cls(
            seq_lengths=seq_lengths,
            kv_num_blocks=kv_num_blocks,
            kv_indices=kv_indices,
            full_kv_num_blocks=full_kv_num_blocks,
            full_kv_indices=full_kv_indices,
            q_num_blocks=q_num_blocks,
            q_indices=q_indices,
            full_q_num_blocks=full_q_num_blocks,
            full_q_indices=full_q_indices,
            BLOCK_SIZE=BLOCK_SIZE,
            mask_mod=mask_mod,
            dq_write_order=dq_write_order,
            dq_write_order_full=dq_write_order_full,
            dq_kv_order=dq_kv_order_tensor,
            dq_kv_order_spt=dq_kv_order_spt,
        )

    @overload
    def as_tuple(
        self, flatten: Literal[True] = ...
    ) -> tuple[
        int,
        int,
        Tensor,
        Tensor,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        bool | None,
        int,
        int,
        _mask_mod_signature,
    ]: ...

    @overload
    def as_tuple(
        self, flatten: Literal[False]
    ) -> tuple[
        tuple[int, int],
        Tensor,
        Tensor,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        bool | None,
        tuple[int, int],
        _mask_mod_signature,
    ]: ...

    def as_tuple(self, flatten: bool = True) -> tuple[Any, ...]:
        """
        Returns a tuple of the attributes of the BlockMask.

        Args:
            flatten (bool): If True, it will flatten the tuple of (KV_BLOCK_SIZE, Q_BLOCK_SIZE)
        """
        if flatten:
            block_size = (self.BLOCK_SIZE[0], self.BLOCK_SIZE[1])  # type: ignore[assignment]
            seq_lengths = (self.seq_lengths[0], self.seq_lengths[1])  # type: ignore[assignment]
        else:
            block_size = (self.BLOCK_SIZE,)  # type: ignore[assignment]
            seq_lengths = (self.seq_lengths,)  # type: ignore[assignment]

        # pyrefly: ignore [not-iterable]
        return (
            *seq_lengths,
            self.kv_num_blocks,
            self.kv_indices,
            self.full_kv_num_blocks,
            self.full_kv_indices,
            self.q_num_blocks,
            self.q_indices,
            self.full_q_num_blocks,
            self.full_q_indices,
            self.dq_write_order,
            self.dq_write_order_full,
            self.dq_kv_order,
            self.dq_kv_order_spt,
            *block_size,
            self.mask_mod,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        *batch_dims, _, _ = self.kv_indices.shape
        return tuple(batch_dims) + self.seq_lengths

    def __str__(self) -> str:
        s = f"BlockMask(shape={self.shape}, sparsity={self.sparsity():.2f}%, \n"
        mask_str = self.to_string().strip()
        s += mask_str
        s += "\n)"
        return s

    def __getitem__(
        self, index: int | slice | Tensor | tuple[int | slice | Tensor, ...]
    ) -> Self:
        """
        Returns a new BlockMask by selecting batch, head, and Q-block rows.

        BlockMask indexing accepts up to three indices. The first two select batch
        and head metadata. The third selects rows in the block-sparse Q grid, not
        individual query tokens. For example, ``block_mask[:, :, 1]`` selects one
        Q-block row, covering the second chunk of ``Q_BLOCK_SIZE`` query tokens,
        not query token 1. Integer indices are normalized to length-one slices so
        the corresponding BlockMask dimension is preserved.

        The returned BlockMask is packed: selecting Q block row ``i`` produces
        local Q block row 0 in the result. Its ``shape[-2]`` is the packed logical
        query-token length of the selected Q-block rows, clipped for partial final
        blocks when the selection is a Python integer or slice. Tensor indexing is
        handled without reading index values, so its logical query length is the
        number of selected Q-block rows times ``Q_BLOCK_SIZE``. Use a Python
        integer or slice when selecting a partial final block and an exact logical
        length is required.

        Callers must slice or gather the query tensor separately and install a new
        ``mask_mod`` if the mask depends on absolute query positions. KV block
        columns are not sliced by this API.

        Args:
            index: Batch, head, and Q-block-row index.

        Returns:
            A BlockMask containing metadata for the selected batch, head, and
            packed Q-block rows.

        Raises:
            IndexError: If more than three indices are provided.
            NotImplementedError: If tensor dq_kv_order metadata is present.

        Example:
            .. code-block:: python

                def causal_mask(b, h, q_idx, kv_idx):
                    return q_idx >= kv_idx


                block_mask = create_block_mask(
                    causal_mask, 4, 2, 512, 512, device="cuda"
                )
                assert block_mask.kv_num_blocks.shape == (4, 2, 4)
                assert block_mask.kv_indices.shape == (4, 2, 4, 4)

                batch_slice = block_mask[0]
                assert batch_slice.kv_num_blocks.shape == (1, 2, 4)
                assert batch_slice.kv_indices.shape == (1, 2, 4, 4)

                head_slice = block_mask[0, 1]
                assert head_slice.kv_num_blocks.shape == (1, 1, 4)
                assert head_slice.kv_indices.shape == (1, 1, 4, 4)

                query_block_slice = block_mask[:, :, :1]
                assert query_block_slice.shape == (4, 2, 128, 512)
                assert query_block_slice.kv_num_blocks.shape == (4, 2, 1)
                assert query_block_slice.kv_indices.shape == (4, 2, 1, 4)

                same_block_slice = block_mask[:, :, 0]
                assert same_block_slice.shape == (4, 2, 128, 512)

                q = torch.randn(4, 2, 512, 64, device="cuda")
                q_chunk = q[:, :, :128, :]
                assert q_chunk.shape[-2] == query_block_slice.shape[-2]
        """
        if self.dq_kv_order is not None:
            raise NotImplementedError(
                "Slicing BlockMask with tensor dq_kv_order is not supported yet. "
                "Construct a new BlockMask with matching precomputed dq_write_order metadata."
            )

        index = (index,) if not isinstance(index, tuple) else index
        if len(index) > 3:
            raise IndexError(
                "BlockMask indexing supports batch, head, and Q-block dimensions"
            )
        padded = (*index, slice(None), slice(None), slice(None))[:3]
        sizes = self.kv_num_blocks.shape[:3]
        index = tuple(
            (slice(i + n, i + n + 1) if -n <= i < 0 else slice(i, i + 1))
            if isinstance(i, int)
            else i
            for i, n in zip(padded, sizes, strict=True)
        )
        new_kv_num_blocks = self.kv_num_blocks[index]
        new_kv_indices = self.kv_indices[index]
        if self.full_kv_num_blocks is not None:
            if self.full_kv_indices is None:
                raise AssertionError("full_kv_indices must not be None")
            new_full_kv_num_blocks = self.full_kv_num_blocks[index]
            new_full_kv_indices = self.full_kv_indices[index]
        else:
            new_full_kv_num_blocks = None
            new_full_kv_indices = None
        new_block_mask = BlockMask.from_kv_blocks(
            new_kv_num_blocks,
            new_kv_indices,
            new_full_kv_num_blocks,
            new_full_kv_indices,
            BLOCK_SIZE=self.BLOCK_SIZE,
            mask_mod=_sliced_mask_mod_error,
            seq_lengths=(
                self._query_length_for_sliced_blocks(
                    index[2], new_kv_indices.shape[-2]
                ),
                self.seq_lengths[1],
            ),
            compute_q_blocks=self.q_indices is not None,
        )
        new_block_mask.dq_kv_order = self._slice_dq_kv_order(self.dq_kv_order, index)
        new_block_mask.dq_kv_order_spt = self.dq_kv_order_spt
        if self.dq_write_order is not None and new_block_mask.dq_kv_order is None:
            dq_wo, dq_wo_full = _compute_dq_write_order_from_block_mask(
                new_block_mask, dq_kv_order=new_block_mask.dq_kv_order_spt
            )
            new_block_mask.dq_write_order = dq_wo
            new_block_mask.dq_write_order_full = dq_wo_full
        return new_block_mask

    def __repr__(self) -> str:
        def shape_or_none(x: torch.Tensor | None):
            return x.shape if x is not None else None

        return (
            f"BlockMask(\n"
            f"    kv_num_blocks={self.kv_num_blocks.shape},\n"
            f"    kv_indices={self.kv_indices.shape},\n"
            f"    full_kv_num_blocks={shape_or_none(self.full_kv_num_blocks)},\n"
            f"    full_kv_indices={shape_or_none(self.full_kv_indices)},\n"
            f"    q_num_blocks={shape_or_none(self.q_num_blocks)},\n"
            f"    q_indices={shape_or_none(self.q_indices)},\n"
            f"    full_q_num_blocks={shape_or_none(self.full_q_num_blocks)},\n"
            f"    full_q_indices={shape_or_none(self.full_q_indices)},\n"
            f"    BLOCK_SIZE={self.BLOCK_SIZE},\n"
            f"    shape={self.shape},\n"
            f"    sparsity={self.sparsity():.2f}%,\n"
            f"    mask_mod={self.mask_mod.__name__ if hasattr(self.mask_mod, '__name__') else self.mask_mod}\n"
            f")"
        )

    def _adjust(self, new_q_len: int, new_kv_len: int) -> Self:
        if self.dq_kv_order is not None:
            raise NotImplementedError(
                "Adjusting BlockMask with tensor dq_kv_order is not supported yet. "
                "Construct a new BlockMask with matching precomputed dq_write_order metadata."
            )

        new_num_rows = (new_q_len + self.BLOCK_SIZE[0] - 1) // self.BLOCK_SIZE[0]
        new_num_cols = (new_kv_len + self.BLOCK_SIZE[1] - 1) // self.BLOCK_SIZE[1]
        new_kv_num_blocks, new_kv_indices = _adjust_num_blocks_and_indices(
            self.kv_num_blocks, self.kv_indices, new_num_rows, new_num_cols
        )
        if self.full_kv_num_blocks is not None:
            if self.full_kv_indices is None:
                raise AssertionError("full_kv_indices must not be None")
            (
                new_full_kv_num_blocks,
                new_full_kv_indices,
            ) = _adjust_num_blocks_and_indices(
                self.full_kv_num_blocks,
                self.full_kv_indices,
                new_num_rows,
                new_num_cols,
            )
        else:
            new_full_kv_num_blocks = None
            new_full_kv_indices = None
        new_block_mask = self.from_kv_blocks(
            new_kv_num_blocks,
            new_kv_indices,
            new_full_kv_num_blocks,
            new_full_kv_indices,
            BLOCK_SIZE=self.BLOCK_SIZE,
            mask_mod=self.mask_mod,
            seq_lengths=(new_q_len, new_kv_len),
            compute_q_blocks=self.q_indices is not None,
        )
        new_block_mask.dq_kv_order = self.dq_kv_order
        new_block_mask.dq_kv_order_spt = self.dq_kv_order_spt
        if self.dq_write_order is not None and new_block_mask.dq_kv_order is None:
            dq_wo, dq_wo_full = _compute_dq_write_order_from_block_mask(
                new_block_mask, dq_kv_order=new_block_mask.dq_kv_order_spt
            )
            new_block_mask.dq_write_order = dq_wo
            new_block_mask.dq_write_order_full = dq_wo_full
        return new_block_mask

    def numel(self) -> int:
        """Returns the number of elements (not accounting for sparsity) in the mask."""
        shape = self.shape

        def _prod(xs):
            return functools.reduce(operator.mul, xs, 1)

        return _prod(shape)

    def sparsity(self) -> float:
        """Computes the percentage of blocks that are sparse (i.e. not computed)"""
        total_blocks = math.prod(self.kv_num_blocks.shape[:-1])
        for seq_len, block_size in zip(self.seq_lengths, self.BLOCK_SIZE, strict=True):
            total_blocks *= _cdiv(seq_len, block_size)

        computed_blocks = self.kv_num_blocks.sum()
        if self.full_kv_num_blocks is not None:
            computed_blocks += self.full_kv_num_blocks.sum()

        dense_ratio = computed_blocks.item() / total_blocks
        return 100 * (1 - dense_ratio)

    def to_dense(self) -> Tensor:
        """Returns a dense block that is equivalent to the block mask."""
        partial_dense = _ordered_to_dense(
            self.kv_num_blocks, self.kv_indices, self.kv_indices.shape[-1]
        )
        if self.full_kv_num_blocks is not None:
            if self.full_kv_indices is None:
                raise AssertionError("full_kv_indices must not be None")
            # pyrefly: ignore [bad-return]
            return partial_dense | _ordered_to_dense(
                self.full_kv_num_blocks,
                self.full_kv_indices,
                self.full_kv_indices.shape[-1],
            )
        return partial_dense

    def to_string(
        self, grid_size: int | tuple[int, int] = (20, 20), limit: int = 4
    ) -> str:
        """Returns a string representation of the block mask. Quite nifty.

        If grid_size is -1, prints out an uncompressed version. Warning, it can be quite big!
        """
        dense_mask = self.to_dense()
        *batch_dims, num_rows, num_cols = dense_mask.shape
        if isinstance(grid_size, int):
            max_rows = grid_size
            max_cols = grid_size
        elif grid_size == -1:
            max_rows = num_rows
            max_cols = num_cols
        else:
            max_rows, max_cols = grid_size

        def create_block_vis(*batch_idx):
            descriptors = []

            descriptors.append(f"{batch_idx}")

            vis = ", ".join(reversed(descriptors)) + "\n"

            def summarize_section(section) -> str:
                percentage = section.float().mean().item()
                if percentage == 1:
                    return "█"
                elif percentage == 0:
                    return " "
                else:
                    return "░"

            row_step = max(1, _cdiv(num_rows, max_rows))
            col_step = max(1, _cdiv(num_cols, max_cols))

            for r in range(0, num_rows, row_step):
                for c in range(0, num_cols, col_step):
                    cur_mask = dense_mask
                    for idx in batch_idx:
                        cur_mask = cur_mask[idx]
                    char = summarize_section(
                        cur_mask[r : r + row_step, c : c + col_step]
                    )
                    vis += char * 2
                vis += "\n"
            return vis

        total_vis = []
        for idx, batch_idx in enumerate(
            itertools.product(*[range(i) for i in batch_dims])
        ):
            if idx == limit:
                total_vis.append("...")
                total_vis.append("To print out more, set BlockMask.to_string(limit=N)")
                total_vis.append(
                    "You can also index (BlockMask[batch, head]) to choose a specific batch or head"
                )
                break
            block_vis = create_block_vis(*batch_idx)
            total_vis.append(block_vis)

        return "\n".join(total_vis)

    def to(self, device: torch.device | str) -> BlockMask:
        """Moves the BlockMask to the specified device.

        Args:
            device (torch.device or str): The target device to move the BlockMask to.
                Can be a torch.device object or a string (e.g., 'cpu', 'cuda:0').

        Returns:
            BlockMask: A new BlockMask instance with all tensor components moved
            to the specified device.

        Note:
            This method does not modify the original BlockMask in-place.
            Instead, it returns a new BlockMask instance where individual tensor attributes
            may or may not be moved to the specified device, depending on their
            current device placement.
        """
        mapped_tensors = tree_map_only(
            torch.Tensor,
            lambda x: x.to(device),
            tuple(getattr(self, attr) for attr in self._TENSOR_ATTRS),
        )
        return BlockMask(
            seq_lengths=self.seq_lengths,
            **dict(zip(self._TENSOR_ATTRS, mapped_tensors, strict=True)),
            BLOCK_SIZE=self.BLOCK_SIZE,
            mask_mod=self.mask_mod,
            dq_kv_order_spt=self.dq_kv_order_spt,
        )

    @staticmethod
    def _wrap_context_value(attr: str, value: Any) -> Any:
        if attr == "mask_mod":
            return _MaskModWrapper(value)
        return value

    @staticmethod
    def _unwrap_context_value(attr: str, value: Any) -> Any:
        if attr == "mask_mod":
            if not isinstance(value, _MaskModWrapper):
                raise AssertionError(f"Expected _MaskModWrapper, got {type(value)}")
            return value.fn
        return value

    def _flatten(
        self,
    ) -> tuple[tuple[BaseArgumentTypes | None, ...], tuple[Any, ...]]:
        """Flatten BlockMask into a list of tensors and context.

        Closure tensors from mask_mod are extracted into the leaves via
        _extract_callable_pytree so they are visible to the tracing
        infrastructure (instead of being hidden in the pytree context).
        """
        optional_tensor_attrs = tuple(
            attr for attr in self._TENSOR_ATTRS if getattr(self, attr) is None
        )
        tensors = tuple(
            getattr(self, attr)
            for attr in self._TENSOR_ATTRS
            if attr not in optional_tensor_attrs
        )
        closure_leaves, callable_spec, stripped = _extract_callable_pytree(
            self.mask_mod
        )
        all_leaves = tensors + closure_leaves
        context = (
            *(
                self._wrap_context_value(attr, getattr(self, attr))
                if attr != "mask_mod"
                else _MaskModWrapper(stripped, callable_spec)
                for attr in self._CONTEXT_ATTRS
            ),
            optional_tensor_attrs,
        )
        return all_leaves, context

    @classmethod
    def _unflatten(
        cls,
        leaves: tuple[Any, ...],
        context: tuple[Any, ...],
    ) -> Self:
        """Unflatten leaves and context back into a BlockMask."""
        optional_tensor_attrs = context[-1]
        tensor_attrs = tuple(
            attr for attr in cls._TENSOR_ATTRS if attr not in optional_tensor_attrs
        )
        n_regular = len(tensor_attrs)
        regular_leaves = leaves[:n_regular]
        closure_leaves = leaves[n_regular:]
        tensor_values = dict.fromkeys(optional_tensor_attrs)
        tensor_values.update(zip(tensor_attrs, regular_leaves))
        context_values = {}
        for attr, val in zip(cls._CONTEXT_ATTRS, context[:-1]):
            if attr == "mask_mod" and isinstance(val, _MaskModWrapper):
                context_values[attr] = _reconstruct_closure_fn(
                    val.fn, closure_leaves, val.callable_spec
                )
            else:
                context_values[attr] = cls._unwrap_context_value(attr, val)
        return cls(
            seq_lengths=cast(tuple[int, int], context_values["seq_lengths"]),
            kv_num_blocks=cast(Tensor, tensor_values["kv_num_blocks"]),
            kv_indices=cast(Tensor, tensor_values["kv_indices"]),
            full_kv_num_blocks=cast(Tensor | None, tensor_values["full_kv_num_blocks"]),
            full_kv_indices=cast(Tensor | None, tensor_values["full_kv_indices"]),
            q_num_blocks=cast(Tensor | None, tensor_values["q_num_blocks"]),
            q_indices=cast(Tensor | None, tensor_values["q_indices"]),
            full_q_num_blocks=cast(Tensor | None, tensor_values["full_q_num_blocks"]),
            full_q_indices=cast(Tensor | None, tensor_values["full_q_indices"]),
            BLOCK_SIZE=cast(tuple[int, int], context_values["BLOCK_SIZE"]),
            mask_mod=cast(_mask_mod_signature, context_values["mask_mod"]),
            dq_write_order=cast(Tensor | None, tensor_values["dq_write_order"]),
            dq_write_order_full=cast(
                Tensor | None, tensor_values["dq_write_order_full"]
            ),
            dq_kv_order=cast(Tensor | None, tensor_values["dq_kv_order"]),
            dq_kv_order_spt=cast(bool | None, context_values["dq_kv_order_spt"]),
        )

    def _flatten_with_keys(
        self,
    ) -> tuple[tuple[tuple[GetAttrKey, Any], ...], tuple[Any, ...]]:
        """Flatten BlockMask with keys for better tracing.

        Closure tensors from mask_mod are extracted into the leaves via
        _extract_callable_pytree so they are visible to the tracing
        infrastructure (instead of being hidden in the pytree context).
        """
        optional_tensor_attrs = tuple(
            attr for attr in self._TENSOR_ATTRS if getattr(self, attr) is None
        )
        tensors = tuple(
            (GetAttrKey(attr), getattr(self, attr))
            for attr in self._TENSOR_ATTRS
            if attr not in optional_tensor_attrs
        )
        closure_leaves, callable_spec, stripped = _extract_callable_pytree(
            self.mask_mod
        )
        closure_with_keys = tuple(
            (GetAttrKey(f"_closure_{i}"), leaf) for i, leaf in enumerate(closure_leaves)
        )
        all_leaves = tensors + closure_with_keys
        context = (
            *(
                self._wrap_context_value(attr, getattr(self, attr))
                if attr != "mask_mod"
                else _MaskModWrapper(stripped, callable_spec)
                for attr in self._CONTEXT_ATTRS
            ),
            optional_tensor_attrs,
        )
        return all_leaves, context


def _broadcast_to_dim(x: Tensor, dim: int) -> Tensor:
    while x.dim() < dim:
        x = x.unsqueeze(0)
    return x


def _round_up_to_multiple(x: int, multiple: int) -> int:
    return (x + multiple - 1) // multiple * multiple


def _convert_mask_to_block_mask(
    mask: Tensor,
    Q_BLOCK_SIZE: int = _DEFAULT_SPARSE_BLOCK_SIZE,
    KV_BLOCK_SIZE: int = _DEFAULT_SPARSE_BLOCK_SIZE,
    separate_full_blocks: bool = False,
) -> tuple[Tensor, Tensor | None]:
    if mask.dtype != torch.bool:
        raise AssertionError(f"mask.dtype must be torch.bool, got {mask.dtype}")
    mask = _broadcast_to_dim(mask, 4)

    def padding_needed_for_multiple(x, multiple):
        return _round_up_to_multiple(x, multiple) - x

    mask = torch.nn.functional.pad(
        mask,
        (
            0,
            padding_needed_for_multiple(mask.shape[-1], KV_BLOCK_SIZE),
            0,
            padding_needed_for_multiple(mask.shape[-2], Q_BLOCK_SIZE),
        ),
    )
    B, H, Q, KV = mask.shape
    if Q % Q_BLOCK_SIZE != 0:
        raise AssertionError(
            f"Q ({Q}) must be divisible by Q_BLOCK_SIZE ({Q_BLOCK_SIZE})"
        )
    if KV % KV_BLOCK_SIZE != 0:
        raise AssertionError(
            f"KV ({KV}) must be divisible by KV_BLOCK_SIZE ({KV_BLOCK_SIZE})"
        )
    mask = mask.view(
        B, H, Q // Q_BLOCK_SIZE, Q_BLOCK_SIZE, KV // KV_BLOCK_SIZE, KV_BLOCK_SIZE
    )  # [B, H, Q//Q_BLOCK_SIZE, Q_BLOCK_SIZE, KV//KV_BLOCK_SIZE, KV_BLOCK_SIZE]
    mask = mask.permute(
        0, 1, 2, 4, 3, 5
    )  # [B, H, Q//Q_BLOCK_SIZE, KV//KV_BLOCK_SIZE, Q_BLOCK_SIZE, KV_BLOCK_SIZE]
    mask_block_sum = mask.sum(
        dim=[-2, -1]
    )  # [B, H, Q//Q_BLOCK_SIZE, KV//KV_BLOCK_SIZE]
    if separate_full_blocks:
        full_block_sum = Q_BLOCK_SIZE * KV_BLOCK_SIZE
        full_blocks = mask_block_sum == full_block_sum
        partial_blocks = (mask_block_sum > 0) & (mask_block_sum < full_block_sum)
        partial_blocks = partial_blocks.to(dtype=torch.int8)
        full_blocks = full_blocks.to(dtype=torch.int8)
        return partial_blocks, full_blocks
    else:
        partial_blocks = mask_block_sum > 0
        partial_blocks = partial_blocks.to(dtype=torch.int8)
        return partial_blocks, None


def or_masks(*mask_mods: _mask_mod_signature) -> _mask_mod_signature:
    """Returns a mask_mod that's the union of provided mask_mods"""
    if not all(callable(arg) for arg in mask_mods):
        raise RuntimeError(f"All inputs should be callable mask_mods: {mask_mods}")

    def or_mask(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        result = b.new_zeros((), dtype=torch.bool)
        for mask in mask_mods:
            result = result | mask(b, h, q_idx, kv_idx)
        return result

    return or_mask


def and_masks(*mask_mods: _mask_mod_signature) -> _mask_mod_signature:
    """Returns a mask_mod that's the intersection of provided mask_mods"""
    if not all(callable(arg) for arg in mask_mods):
        raise RuntimeError(f"All inputs should be callable mask_mods: {mask_mods}")

    def and_mask(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        result = b.new_ones((), dtype=torch.bool)
        for mask in mask_mods:
            result = result & mask(b, h, q_idx, kv_idx)
        return result

    return and_mask


def _convert_block_mask_to_mask(
    block_mask: Tensor,
    KV_BLOCK_SIZE: int = _DEFAULT_SPARSE_BLOCK_SIZE,
    Q_BLOCK_SIZE: int = _DEFAULT_SPARSE_BLOCK_SIZE,
) -> Tensor:
    if block_mask.dim() != 4:
        raise AssertionError(f"block_mask.dim() must be 4, got {block_mask.dim()}")
    B, H, Q, KV = block_mask.shape
    block_mask = block_mask.expand(Q_BLOCK_SIZE, KV_BLOCK_SIZE, *block_mask.shape)
    block_mask = block_mask.permute(2, 3, 4, 0, 5, 1).reshape(
        B, H, Q * Q_BLOCK_SIZE, KV * KV_BLOCK_SIZE
    )
    return block_mask


def _create_sparse_block_from_block_mask(
    block_mask: tuple[Tensor, Tensor | None],
    mask_mod: _mask_mod_signature | None,
    seq_lengths: tuple[int, int],
    Q_BLOCK_SIZE: int = _DEFAULT_SPARSE_BLOCK_SIZE,
    KV_BLOCK_SIZE: int = _DEFAULT_SPARSE_BLOCK_SIZE,
) -> BlockMask:
    partial_blocks, full_blocks = block_mask

    partial_bm = _dense_to_ordered(partial_blocks)
    if full_blocks is not None:
        full_bm: tuple[Tensor | None, Tensor | None] = _dense_to_ordered(full_blocks)
    else:
        full_bm = (None, None)

    return BlockMask.from_kv_blocks(
        partial_bm[0],
        partial_bm[1],
        full_bm[0],
        full_bm[1],
        BLOCK_SIZE=(Q_BLOCK_SIZE, KV_BLOCK_SIZE),
        mask_mod=mask_mod,
        seq_lengths=seq_lengths,
    )


def create_mask(
    mod_fn: _score_mod_signature | _mask_mod_signature,
    B: int | None,
    H: int | None,
    Q_LEN: int,
    KV_LEN: int,
    device: DeviceLikeType | None = None,
) -> Tensor:
    r"""This function creates a mask tensor from a mod_fn function.

    Args:
        mod_fn (Union[_score_mod_signature, _mask_mod_signature]): Function to modify attention scores.
        B (int): Batch size.
        H (int): Number of query heads.
        Q_LEN (int): Sequence length of query.
        KV_LEN (int): Sequence length of key/value.
        device (str): Device to run the mask creation on.

    Returns:
        mask (Tensor): A mask tensor with shape (B, H, M, N).
    """
    if device is None:
        device = torch.accelerator.current_accelerator() or "cpu"
    if B is None:
        B = 1
    if H is None:
        H = 1
    b = torch.arange(0, B, device=device)
    h = torch.arange(0, H, device=device)
    m = torch.arange(0, Q_LEN, device=device)
    n = torch.arange(0, KV_LEN, device=device)
    mod_type = _get_mod_type(mod_fn)

    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex

    with TransformGetItemToIndex():
        if mod_type == _ModificationType.SCORE_MOD:
            score_mod = mod_fn
            score_mod = _vmap_for_bhqkv(score_mod, prefix=(0,))  # first input is score
            out = score_mod(torch.zeros(B, H, Q_LEN, KV_LEN, device=device), b, h, m, n)
            mask = torch.where(torch.isneginf(out), False, True)
            return mask
        elif mod_type == _ModificationType.MASK_MOD:
            mask_mod = mod_fn
            mask_mod = _vmap_for_bhqkv(mask_mod, prefix=())
            mask = mask_mod(b, h, m, n)
            return mask
        else:
            raise AssertionError


def _compute_dq_write_order_from_block_mask(
    block_mask: BlockMask,
    dq_kv_order: bool | None = False,
) -> tuple[Tensor, Tensor | None]:
    """Compute dQ write-order metadata for deterministic block-sparse backward.

    Algorithm (per batch/head):
    1. Convert the partial and full `kv` block lists into a dense `m_block -> n_block`
       contributor mask and merge them.
    2. Build a rank table in the requested scheduler order. ``dq_kv_order=False``
       means ascending n-block order and ``True`` means descending/SPT order.
    3. Gather ranks using backward `q_indices` order to produce semaphore values
       used by deterministic dQ accumulation.
    """
    kv_num_blocks = block_mask.kv_num_blocks
    kv_indices = block_mask.kv_indices
    full_kv_num_blocks = block_mask.full_kv_num_blocks
    full_kv_indices = block_mask.full_kv_indices
    q_num_blocks = block_mask.q_num_blocks
    q_indices = block_mask.q_indices
    full_q_num_blocks = block_mask.full_q_num_blocks
    full_q_indices = block_mask.full_q_indices

    if q_num_blocks is None or q_indices is None:
        raise ValueError(
            "BlockMask must have q_num_blocks and q_indices to compute dq_write_order"
        )

    device = kv_indices.device

    def _expand_bh(tensor: Tensor, target_b: int, target_h: int) -> Tensor:
        return tensor.expand(target_b, target_h, *tensor.shape[2:])

    broadcast_shapes = [
        kv_indices.shape[:2],
        q_indices.shape[:2],
    ]
    if full_kv_indices is not None:
        broadcast_shapes.append(full_kv_indices.shape[:2])
    if full_q_indices is not None:
        broadcast_shapes.append(full_q_indices.shape[:2])
    B, H = torch.broadcast_shapes(*broadcast_shapes)
    kv_num_blocks = _expand_bh(kv_num_blocks, B, H)
    kv_indices = _expand_bh(kv_indices, B, H)
    if full_kv_num_blocks is not None:
        full_kv_num_blocks = _expand_bh(full_kv_num_blocks, B, H)
    if full_kv_indices is not None:
        full_kv_indices = _expand_bh(full_kv_indices, B, H)

    _, _, num_m, _ = kv_indices.shape
    _, _, num_n, _ = q_indices.shape

    has_full = full_kv_num_blocks is not None and full_kv_indices is not None

    dense_partial = _ordered_to_dense(kv_num_blocks, kv_indices, num_n)
    if has_full:
        if full_kv_num_blocks is None or full_kv_indices is None:
            raise AssertionError(
                "full_kv_num_blocks and full_kv_indices must not be None"
            )
        dense_full = _ordered_to_dense(full_kv_num_blocks, full_kv_indices, num_n)
        dense = (dense_partial + dense_full).clamp(max=1)
    else:
        dense = dense_partial

    cumsum = dense.cumsum(dim=-1)
    rank_table = (cumsum - dense).to(torch.int32)

    if dq_kv_order:
        total_per_m = cumsum[:, :, :, -1:]
        rank_table = (total_per_m - 1 - rank_table).to(torch.int32)

    def _gather_write_order(bwd_idx: Tensor) -> Tensor:
        b_i = torch.arange(B, device=device)[:, None, None, None]
        h_i = torch.arange(H, device=device)[None, :, None, None]
        n_i = torch.arange(bwd_idx.shape[2], device=device)[None, None, :, None]
        m_vals = bwd_idx.long().clamp(0, num_m - 1)
        return rank_table[b_i, h_i, m_vals, n_i].to(torch.int32)

    dq_write_order = _gather_write_order(q_indices)

    dq_write_order_full = None
    if has_full and full_q_num_blocks is not None and full_q_indices is not None:
        dq_write_order_full = _gather_write_order(full_q_indices)

    return dq_write_order, dq_write_order_full


def create_block_mask(
    mask_mod: _mask_mod_signature,
    B: int | None,
    H: int | None,
    Q_LEN: int,
    KV_LEN: int,
    device: DeviceLikeType | None = None,
    BLOCK_SIZE: int | tuple[int, int] = _DEFAULT_SPARSE_BLOCK_SIZE,
    _compile=False,
    separate_full_blocks: bool = True,
    compute_dq_write_order: bool = False,
    dq_kv_order: bool = True,
) -> BlockMask:
    r"""This function creates a block mask tuple from a mask_mod function.

    Args:
        mask_mod (Callable): mask_mod function. This is a callable that defines the
            masking pattern for the attention mechanism. It takes four arguments:
            b (batch size), h (number of heads), q_idx (query index), and kv_idx (key/value index).
            It should return a boolean tensor indicating which attention connections are allowed (True)
            or masked out (False).
        B (int): Batch size.
        H (int): Number of query heads.
        Q_LEN (int): Sequence length of query.
        KV_LEN (int): Sequence length of key/value.
        device (str): Device to run the mask creation on.
        BLOCK_SIZE (int or tuple[int, int]): Block size for the block mask. If a single int is provided it is used for both query and key/value.
        separate_full_blocks (bool): If True, fully unmasked blocks are stored
            separately so kernels can skip mask_mod on those blocks. If False,
            all non-empty blocks are stored as partial blocks and mask_mod is
            applied to every block.
        compute_dq_write_order (bool): If True, precompute dQ write-order
            metadata needed by deterministic block-sparse FLASH backward.
        dq_kv_order (bool): KV-column scheduler order used for deterministic
            dQ accumulation when compute_dq_write_order is True. False means
            ascending n-block order and True means descending/SPT order.
            Explicit tensor schedules are not supported by create_block_mask
            yet; they are supported by BlockMask.from_kv_blocks for callers
            that provide precomputed write-order metadata directly.

    Returns:
        BlockMask:  A BlockMask object that contains the block mask information.

    Example Usage:
        .. code-block:: python

            def causal_mask(b, h, q_idx, kv_idx):
                return q_idx >= kv_idx


            block_mask = create_block_mask(causal_mask, 1, 1, 8192, 8192, device="cuda")
            query = torch.randn(1, 1, 8192, 64, device="cuda", dtype=torch.float16)
            key = torch.randn(1, 1, 8192, 64, device="cuda", dtype=torch.float16)
            value = torch.randn(1, 1, 8192, 64, device="cuda", dtype=torch.float16)
            output = flex_attention(query, key, value, block_mask=block_mask)
    """
    if device is None:
        device = torch.accelerator.current_accelerator() or "cpu"
    mod_type = _get_mod_type(mask_mod)
    if mod_type != _ModificationType.MASK_MOD:
        raise AssertionError(
            f"create-block_mask requires a mask_mod function! Got {mask_mod}"
        )
    if B is None:
        B = 1
    if H is None:
        H = 1
    if isinstance(BLOCK_SIZE, int):
        Q_BLOCK_SIZE = BLOCK_SIZE
        KV_BLOCK_SIZE = BLOCK_SIZE
    else:
        Q_BLOCK_SIZE, KV_BLOCK_SIZE = BLOCK_SIZE

    if _compile:
        warnings.warn(
            "_compile flag on create_block_mask was originally added to work around a torch.compile limitation. That limitation has since been addressed. So, to compile create_block_mask, we suggest doing torch.compile(create_block_mask). This still works for now, but will be removed in the future.",
            DeprecationWarning,
            stacklevel=2,
        )
        return torch.compile(create_block_mask)(
            mask_mod,
            B,
            H,
            Q_LEN,
            KV_LEN,
            device,
            BLOCK_SIZE,
            False,
            separate_full_blocks,
            compute_dq_write_order,
            dq_kv_order,
        )

    if not isinstance(dq_kv_order, bool):
        raise ValueError("dq_kv_order must be a bool when using create_block_mask")

    mask_tensor = create_mask(mask_mod, B, H, Q_LEN, KV_LEN, device)
    partial_block_mask, full_block_mask = _convert_mask_to_block_mask(
        mask_tensor,
        Q_BLOCK_SIZE=Q_BLOCK_SIZE,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
        separate_full_blocks=separate_full_blocks,
    )
    block_mask = _create_sparse_block_from_block_mask(
        (partial_block_mask, full_block_mask),
        mask_mod,
        (Q_LEN, KV_LEN),
        Q_BLOCK_SIZE,
        KV_BLOCK_SIZE,
    )

    if compute_dq_write_order:
        dq_wo, dq_wo_full = _compute_dq_write_order_from_block_mask(
            block_mask, dq_kv_order=dq_kv_order
        )
        block_mask.dq_write_order = dq_wo
        block_mask.dq_write_order_full = dq_wo_full
        block_mask.dq_kv_order = None
        block_mask.dq_kv_order_spt = dq_kv_order

    return block_mask


def _create_empty_block_mask(query: Tensor, key: Tensor) -> BlockMask:
    r"""Default block mask for flex attention.
    If users don't specify any block sparse mask info, we create this
    empty block sparse mask. Which creates a BlockMask with 1 block that is the full length
    of the query and key tensors.
    """
    device = query.device
    return BlockMask.from_kv_blocks(
        kv_num_blocks=torch.ones([1, 1, 1], dtype=torch.int32, device=device),
        kv_indices=torch.zeros([1, 1, 1, 1], dtype=torch.int32, device=device),
        BLOCK_SIZE=_LARGE_SPARSE_BLOCK_SIZE,
        seq_lengths=(1, 1),
    )


def _apply_kernel_options(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    return_lse: bool,
    kernel_options: FlexKernelOptions | None,
    return_aux: AuxRequest | None = None,
) -> _KernelOptionsWithInternals:
    kernel_options = cast(
        _KernelOptionsWithInternals,
        {} if kernel_options is None else dict(kernel_options),
    )

    if "BACKEND" in kernel_options and kernel_options.get(
        "FORCE_USE_FLEX_ATTENTION", False
    ):
        # TODO: remove FORCE_USE_FLEX_ATTENTION once BACKEND is fully adopted.
        raise RuntimeError(
            "BACKEND cannot be combined with legacy FORCE_USE_FLEX_ATTENTION. "
            "BACKEND supersedes the legacy knob; please drop FORCE_USE_FLEX_ATTENTION "
            "and only specify the desired BACKEND."
        )

    if "BACKEND" in kernel_options:
        valid_backends = typing.get_args(_Backend)
        if kernel_options["BACKEND"] not in valid_backends:
            raise ValueError(
                f"Invalid BACKEND value '{kernel_options['BACKEND']}'. "
                f"Must be one of {valid_backends}"
            )

    kernel_options.setdefault("BACKEND", "AUTO")
    kernel_options.setdefault("PRESCALE_QK", False)
    kernel_options.setdefault("ROWS_GUARANTEED_SAFE", False)
    kernel_options.setdefault("BLOCKS_ARE_CONTIGUOUS", False)
    # This forces all biases grad scatters to be done in the DQ iteration loop of the backwards
    kernel_options.setdefault("WRITE_DQ", True)

    any_inputs_on_cpu_device = (
        query.device.type == "cpu"
        or key.device.type == "cpu"
        or value.device.type == "cpu"
    )
    any_inputs_on_mps_device = (
        query.device.type == "mps"
        or key.device.type == "mps"
        or value.device.type == "mps"
    )

    # Determine what auxiliary outputs are needed
    output_lse = return_lse
    output_max = False

    if return_aux is not None:
        # New API takes precedence over legacy parameters
        output_lse = return_aux.lse
        output_max = return_aux.max_scores

    # If forward kernel needs to return logsumexp is decided by this rule internally.
    if "OUTPUT_LOGSUMEXP" in kernel_options:
        raise AssertionError("OUTPUT_LOGSUMEXP must not be in kernel_options")
    kernel_options["OUTPUT_LOGSUMEXP"] = True
    if not output_lse:
        # We used to check if q,k,v required grads but since captured buffers can require grad
        # we always write unless in no_grad
        kernel_options["OUTPUT_LOGSUMEXP"] = torch.is_grad_enabled()
        if any_inputs_on_cpu_device or any_inputs_on_mps_device:
            # CPU/MPS support inference only, no LSE/backward yet.
            # TODO: support CPU for training and return lse
            kernel_options["OUTPUT_LOGSUMEXP"] = False
        if any_inputs_on_mps_device:
            # MPS supports inference only; backward / LSE not yet implemented
            kernel_options["OUTPUT_LOGSUMEXP"] = False

    # If forward kernel needs to return max is decided by this rule internally.
    if "OUTPUT_MAX" in kernel_options:
        raise AssertionError("OUTPUT_MAX must not be in kernel_options")
    if kernel_options["BACKEND"] == "FLASH" and output_max:
        raise NotImplementedError(
            "Returning max scores is not supported with BACKEND='FLASH'. "
            "Use return_aux=AuxRequest(lse=True) or omit max_scores."
        )
    kernel_options["OUTPUT_MAX"] = output_max
    if any_inputs_on_cpu_device and output_max:
        # CPU doesn't support returning max yet
        # TODO: support CPU for returning max
        raise NotImplementedError("Returning max scores is not supported on CPU.")

    return kernel_options


def _validate_embed_dim(query: Tensor, key: Tensor, value: Tensor) -> None:
    if query.size(-1) != key.size(-1):
        raise ValueError(
            f"Expect query and key/value to have the same embedding dimension "
            f"but got E={query.size(-1)} and E={key.size(-1)}."
        )


def _validate_device(query: Tensor, key: Tensor, value: Tensor) -> None:
    """TODO: Remove once non cuda/cpu devices support is added
    We only need to check query since we have already that q,k,v are on the same device
    """
    if query.device.type == "cpu" and (
        query.requires_grad or key.requires_grad or value.requires_grad
    ):
        raise NotImplementedError(
            "FlexAttention does not support backward on CPU. Please set the input requires_grad to False or use another device."
        )
    if query.device.type == "mps" and (
        query.requires_grad or key.requires_grad or value.requires_grad
    ):
        raise NotImplementedError(
            "FlexAttention does not support backward on MPS. Please set the input requires_grad to False or use another device."
        )
    supported_devices = {"cuda", "cpu", "xpu", "hpu", "mps"}
    if query.device.type not in supported_devices:
        raise ValueError(
            "FlexAttention is only supported on CUDA, CPU, HPU, or MPS devices. "
            f"Found input tensors on {query.device.type} device."
        )


def _validate_no_nested_tensors(query: Tensor, key: Tensor, value: Tensor) -> None:
    if query.is_nested or key.is_nested or value.is_nested:
        raise NotImplementedError(
            "flex_attention does not support NestedTensor inputs, including "
            "torch.compile(flex_attention) with jagged NestedTensor inputs. "
            "Convert inputs to dense tensors before calling flex_attention."
        )


def _enforce_mem_layouts(
    query: Tensor, key: Tensor, value: Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Enforce memory layouts for query, key, and value tensors.

    For non-FP8 dtypes, no action is taken.

    For FP8 dtypes, we enforce the following memory layouts:
    - Query tensor must be in row-major memory layout, as it will be the left-operand in the FP8 GEMM `q @ k.T`.
    - Key tensor must be in row-major memory layout, as it will be transposed when used as the right-operand
      in the FP8 GEMM `q @ k.T`, meaning it will correctly be in column-major memory layout for the GEMM.
    - Value tensor must be in column-major memory layout, as it will be the right-operand in the FP8 GEMM `softmax_scores @ v`.

    Returns the query, key, and value tensors with the enforced memory layouts.
    """

    def is_row_major(tensor: Tensor) -> bool:
        return tensor.stride()[-1] == 1

    def is_col_major(tensor: Tensor) -> bool:
        return tensor.stride()[-2] == 1

    # These memory layout constraints are only for FP8 GEMMs on NVIDIA GPU architectures >= SM89 and < SM100.
    # This is because GPU arch < SM89 does not support FP8 GEMMs, and
    # SM100 has support for TN, NT, TT, NN layouts for FP8 GEMMs
    # (i.e., left and right operands can be in row or column major layouts)
    # so this check is only needed for older architectures.
    # See: https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md
    fp8_dtypes = (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    gemm_precision = query.dtype

    should_enforce_mem_layout = (
        gemm_precision in fp8_dtypes
        and torch.version.cuda is not None
        and torch.cuda.get_device_capability("cuda") >= (8, 9)
        and torch.cuda.get_device_capability("cuda") < (10, 0)
    )
    if not should_enforce_mem_layout:
        return query, key, value

    # Query must be in row-major memory layout as the left-operand in the FP8 GEMM `q @ k.T`
    if not is_row_major(query):
        query = query.contiguous()

    # Key must be in row-major memory layout as it will be transposed when used as the right-operand
    # in the FP8 GEMM `q @ k.T`, meaning it will correctly be in column-major memory layout for the GEMM.
    if not is_row_major(key):
        key = key.contiguous()

    # Value must be in column-major memory layout as the right-operand in the FP8 GEMM `softmax_scores @ v`
    if not is_col_major(value):
        value = value.transpose(-2, -1).contiguous().transpose(-2, -1)
    return query, key, value


@overload
def flex_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    score_mod: _score_mod_signature | None = ...,
    block_mask: BlockMask | None = ...,
    scale: float | None = ...,
    enable_gqa: bool = ...,
    return_lse: Literal[False] = ...,
    kernel_options: FlexKernelOptions | None = ...,
    *,
    return_aux: None = ...,
) -> Tensor: ...


@overload
@deprecated(
    "return_lse is deprecated and will be removed in v2.10. "
    "Use return_aux=AuxRequest(lse=True) instead.",
    category=FutureWarning,
)
def flex_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    score_mod: _score_mod_signature | None = ...,
    block_mask: BlockMask | None = ...,
    scale: float | None = ...,
    enable_gqa: bool = ...,
    return_lse: Literal[True] = ...,
    kernel_options: FlexKernelOptions | None = ...,
    *,
    return_aux: None = ...,
) -> tuple[Tensor, Tensor]: ...


@overload
def flex_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    score_mod: _score_mod_signature | None = ...,
    block_mask: BlockMask | None = ...,
    scale: float | None = ...,
    enable_gqa: bool = ...,
    return_lse: bool = ...,
    kernel_options: FlexKernelOptions | None = ...,
    *,
    return_aux: AuxRequest,
) -> tuple[Tensor, AuxOutput]: ...


@overload
def flex_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    score_mod: _score_mod_signature | None = ...,
    block_mask: BlockMask | None = ...,
    scale: float | None = ...,
    enable_gqa: bool = ...,
    return_lse: Literal[True] = ...,
    kernel_options: FlexKernelOptions | None = ...,
    *,
    return_aux: AuxRequest,
) -> Never: ...


def flex_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    score_mod: _score_mod_signature | None = None,
    block_mask: BlockMask | None = None,
    scale: float | None = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    kernel_options: FlexKernelOptions | None = None,
    *,
    return_aux: AuxRequest | None = None,
) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, AuxOutput]:
    r"""This function implements scaled dot product attention with an arbitrary attention score modification function
    described in the `Flex Attention <https://arxiv.org/abs/2412.05496>`_ paper. See also the
    `blog post <https://pytorch.org/blog/flexattention/>`_.

    This function computes the scaled dot product attention between query, key, and value tensors with a user-defined
    attention score modification function. The attention score modification function will be applied after the attention
    scores have been calculated between the query and key tensors. The attention scores are calculated as follows:

    The ``score_mod`` function should have the following signature:

    .. code-block:: python

        def score_mod(
            score: Tensor,
            batch: Tensor,
            head: Tensor,
            q_idx: Tensor,
            k_idx: Tensor
        ) -> Tensor:

    Where:
        - ``score``: A scalar tensor representing the attention score,
          with the same data type and device as the query, key, and value tensors.
        - ``batch``, ``head``, ``q_idx``, ``k_idx``: Scalar tensors indicating
          the batch index, query head index, query index, and key/value index, respectively.
          These should have the ``torch.int`` data type and be located on the same device as the score tensor.

    Args:
        query (Tensor): Query tensor; shape :math:`(B, Hq, L, E)`. For FP8 dtypes, should be in row-major memory layout for optimal performance.
        key (Tensor): Key tensor; shape :math:`(B, Hkv, S, E)`. For FP8 dtypes, should be in row-major memory layout for optimal performance.
        value (Tensor): Value tensor; shape :math:`(B, Hkv, S, Ev)`. For FP8 dtypes, should be in column-major memory layout for optimal performance.
        score_mod (Optional[Callable]): Function to modify attention scores. By default no score_mod is applied.
        block_mask (Optional[BlockMask]): BlockMask object that controls the blocksparsity pattern of the attention.
        scale (Optional[float]): Scaling factor applied prior to softmax. If none, the default value is set to :math:`\frac{1}{\sqrt{E}}`.
        enable_gqa (bool): If set to True, enables Grouped Query Attention (GQA) and broadcasts key/value heads to query heads.
        return_lse (bool): Whether to return the logsumexp of the attention scores. Default is False. **Deprecated**: Use ``return_aux=AuxRequest(lse=True)`` instead.
        kernel_options (Optional[FlexKernelOptions]):
            Options to control the behavior of the underlying Triton kernels.
            See :class:`FlexKernelOptions` for available options and usage examples.
        return_aux (Optional[AuxRequest]): Specifies which auxiliary outputs to compute and return.
            If None, only the attention output is returned. Use ``AuxRequest(lse=True, max_scores=True)``
            to request both auxiliary outputs.

    Returns:
        output (Tensor): Attention output; shape :math:`(B, Hq, L, Ev)`.

        When ``return_aux`` is not None:
            aux (AuxOutput): Auxiliary outputs with requested fields populated.

        When ``return_aux`` is None (deprecated paths):
            lse (Tensor): Log-sum-exp of attention scores; shape :math:`(B, Hq, L)`. Only returned if ``return_lse=True``.

    Shape legend:
        - :math:`N: \text{Batch size} ... : \text{Any number of other batch dimensions (optional)}`
        - :math:`S: \text{Source sequence length}`
        - :math:`L: \text{Target sequence length}`
        - :math:`E: \text{Embedding dimension of the query and key}`
        - :math:`Ev: \text{Embedding dimension of the value}`

    .. warning::
        `torch.nn.attention.flex_attention` is a prototype feature in PyTorch.
        Please look forward to a more stable implementation in a future version of PyTorch.
        Read more about feature classification at: https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    """
    # Some basic input validation
    _validate_no_nested_tensors(query, key, value)
    _validate_sdpa_input(query, key, value, allow_lowp_kv=True)
    _validate_embed_dim(query, key, value)
    _validate_device(query, key, value)
    query, key, value = _enforce_mem_layouts(query, key, value)
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise NotImplementedError("NYI: query, key, and value must be 4D tensors")
    if (not enable_gqa) and query.size(-3) != key.size(-3):
        raise ValueError(
            f"Expect query and key/value to have the same number of heads "
            f"but got Hq={query.size(-3)} and Hkv={key.size(-3)}. "
            f"Try setting enable_gqa=True for GQA."
        )
    if enable_gqa:
        Hq = query.size(1)
        Hkv = key.size(1)
        if Hq % Hkv != 0:
            raise ValueError(
                f"Expect number of query heads to be a multiple of kv heads for GQA "
                f"but got Hq={Hq} and Hkv={Hkv}."
            )
    if query.size(0) != key.size(0):
        if block_mask is None:
            raise ValueError(
                f"Expect query and key/value to have the same batch size, "
                f"or non-none block_mask, "
                f"but got block_mask=None, Bq={query.size(0)}, and Bkv={key.size(0)}."
            )

        if block_mask.kv_num_blocks.size(0) != query.size(0):
            raise ValueError(
                f"Expect query and key/value to have the same batch size, "
                f"or block_mask and query to have the same batch size, "
                f"but got Bq={query.size(0)}, Bkv={key.size(0)}, B_block_mask={block_mask.kv_num_blocks.size(0)}."
            )

    if score_mod is None:
        score_mod = _identity

    if block_mask is None:
        block_mask = _create_empty_block_mask(query, key)

    # If BlockMask was sliced, its mask_mod is intentionally replaced with an error-raising stub.
    # This guard ensures we surface the intended error message before any shape-based checks.
    if getattr(block_mask, "mask_mod", None) is _sliced_mask_mod_error:
        raise RuntimeError("Cannot use mask_mod from a sliced BlockMask")

    if (
        block_mask.BLOCK_SIZE[0] == _LARGE_SPARSE_BLOCK_SIZE
        and block_mask.BLOCK_SIZE[1] == _LARGE_SPARSE_BLOCK_SIZE
    ):
        # This corresponds to the case where we essentially have a "no-op" block mask.
        pass
    else:
        q_len = query.size(-2)
        kv_len = key.size(-2)
        block_mask_q_len = block_mask.shape[-2]
        block_mask_kv_len = block_mask.shape[-1]

        _validate_block_mask_shape(
            query, key, q_len, kv_len, block_mask_q_len, block_mask_kv_len
        )

    if scale is None:
        scale = 1.0 / math.sqrt(query.size(-1))

    if query.device != block_mask.kv_num_blocks.device:  # type: ignore[union-attr]
        raise RuntimeError(
            f"Expect q/k/v and block_mask to be on the same device "
            f"but got {query.device} and {block_mask.kv_num_blocks.device}."  # type: ignore[union-attr]
        )

    # Handle deprecation warnings for old parameters
    if return_lse and return_aux is not None:
        raise ValueError(
            "Cannot specify both return_lse and return_aux. "
            "return_lse is deprecated, please use return_aux=AuxRequest(lse=True) instead."
        )
    elif return_lse and return_aux is None:
        _warn_once(
            "deprecated_return_lse",
            "return_lse is deprecated and will be removed in v2.10. "
            "Please use return_aux=AuxRequest(lse=True) instead.",
            category=FutureWarning,
        )

    kernel_options = _apply_kernel_options(
        query,
        key,
        value,
        return_lse,
        kernel_options,
        return_aux,
    )

    def _finalize_outputs(
        out,
        lse,
        max_scores,
        *,
        return_aux: AuxRequest | None,
        return_lse: bool,
        stats_are_log2: bool,
    ):
        """Normalize stats and build return value (aux-aware, legacy-compatible)."""
        ln2 = math.log(2.0)
        return_lse = return_lse or return_aux is not None and return_aux.lse
        return_max = return_aux is not None and return_aux.max_scores

        lse_scaled = lse if (return_lse and lse.numel() > 0) else None
        max_scaled = max_scores if (return_max and max_scores.numel() > 0) else None

        if stats_are_log2:
            lse_scaled = lse_scaled * ln2 if lse_scaled is not None else None
            max_scaled = max_scaled * ln2 if max_scaled is not None else None

        if return_aux is not None:
            return out, AuxOutput(
                lse=lse_scaled,
                max_scores=max_scaled,
            )

        if return_lse:
            return out, lse_scaled

        return out

    if torch.compiler.is_dynamo_compiling():
        # mark head_dim and number of heads to be static
        for x in [query, key, value]:
            torch._dynamo.mark_static(x, -3)
            torch._dynamo.mark_static(x, -1)

        out, lse, max_scores = flex_attention_hop(
            query,
            key,
            value,
            score_mod,
            block_mask.as_tuple(),
            scale,
            kernel_options,  # type: ignore[union-attr]
        )
        return _finalize_outputs(
            out,
            lse,
            max_scores,
            return_aux=return_aux,
            return_lse=return_lse,
            stats_are_log2=kernel_options["BACKEND"] != "FLASH",
        )

    if not _FLEX_ATTENTION_DISABLE_COMPILE_DEBUG:
        _warn_once(
            warning_id="flex_attention_performance",
            message=(
                "flex_attention called without torch.compile() - this will use an unfused implementation that materializes the full scores matrix instead of generating a fused kernel.\n\n"
                "SOLUTION: Use torch.compile(flex_attention)(...)\n\n"
                "If you want to debug your score_mod/mask_mod, you can set:\n"
                "torch.nn.attention.flex_attention._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True\n\n"
                "This will allow you to use print statements or breakpoints. Note: This doesn't work with the backwards pass and may produce incorrect results."
            ),
        )

    if not torch._dynamo.is_dynamo_supported():
        raise RuntimeError("flex_attention requires dynamo support")

    # Dynamo is expecting a callable with "__code__" attribute.
    # We cannot directly pass hop to it. So we wrap it in a dummy function.
    def _flex_attention_hop_wrapper(*args, **kwargs):
        return flex_attention_hop(*args, **kwargs)

    with setup_compilation_env() as backend:
        if _FLEX_ATTENTION_DISABLE_COMPILE_DEBUG:
            flex_fn = _flex_attention_hop_wrapper
        else:
            flex_fn = torch.compile(
                _flex_attention_hop_wrapper, backend=backend, fullgraph=True
            )

        out, lse, max_scores = flex_fn(
            query,
            key,
            value,
            score_mod,
            block_mask.as_tuple(),
            scale,
            kernel_options,
        )
    return _finalize_outputs(
        out,
        lse,
        max_scores,
        return_aux=return_aux,
        return_lse=return_lse,
        stats_are_log2=_FLEX_ATTENTION_DISABLE_COMPILE_DEBUG
        or kernel_options["BACKEND"] != "FLASH",
    )
<|endoftext|># mypy: allow-untyped-defs
"""Defines utilities for interacting with scaled_dot_product_attention"""

import math

import torch


__all__: list[str] = []


def _input_requires_grad(*tensors: torch.Tensor) -> bool:
    """Returns True if any of the tensors requires grad"""
    return any(t.requires_grad for t in tensors)


def _postprocess_flash_output(inpt_tensor: torch.Tensor, og_size: int) -> torch.Tensor:
    """Handles the unpad of the last dimension"""
    if inpt_tensor.size(-1) != og_size:
        return inpt_tensor[..., :og_size]
    return inpt_tensor


def _calculate_scale(head_dim_size: int, scale: float | None) -> float:
    """
    For FlashAttention we pad the head dimension to be a multiple of 8 so we need to scale the output
    by the original head size and not the padded.
    """
    if scale is not None:
        return scale
    return 1.0 / math.sqrt(head_dim_size)


def _validate_sdpa_input(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    allow_lowp_kv=False,
) -> None:
    if not allow_lowp_kv:
        if query.dtype != key.dtype or query.dtype != value.dtype:
            raise ValueError(
                f"Expected query, key, and value to have the same dtype, "
                f"but got query.dtype: {query.dtype}, key.dtype: {key.dtype}, "
                f"and value.dtype: {value.dtype} instead."
            )
    if query.device != key.device or query.device != value.device:
        raise ValueError(
            f"Expected query, key, and value to have the same device type, "
            f"but got query.device: {query.device}, key.device: {key.device}, "
            f"and value.device: {value.device} instead."
        )
    if query.dim() < 2 or key.dim() < 2 or value.dim() < 2:
        raise ValueError(
            f"Expected query, key, and value to all be  at least 2 dimensional, but got query.dim: "
            f"{query.dim()}, key.dim: {key.dim()} and value.dim: {value.dim()} instead."
        )
<|endoftext|>"""
Variable-length attention implementation using Flash Attention.

This module provides a high-level Python interface for variable-length attention
that calls into the optimized Flash Attention kernels.
"""

import logging
from functools import lru_cache
from typing import Any, NamedTuple

import torch


log = logging.getLogger(__name__)

__all__ = ["varlen_attn", "varlen_attn_out", "AuxRequest"]


def _normalize_window_size(window_size: list[int] | None) -> list[int]:
    if window_size is None:
        window_size = [-1, -1]

    if len(window_size) != 2:
        raise ValueError(f"window_size must have length 2, got {len(window_size)}")
    return window_size


@lru_cache(maxsize=8)
def _should_use_cudnn(device_index: int) -> bool:
    """Cache device capability check to avoid repeated CUDA calls."""
    if torch.version.hip is not None:
        return False
    cudnn_version = torch.backends.cudnn.version()
    if cudnn_version is None or cudnn_version < 91800:
        return False
    major_cap = torch.cuda.get_device_capability(device_index)[0]
    if major_cap == 9 or major_cap == 10:
        return True
    return False


def _can_use_cudnn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    max_q: int,
    window_size: list[int],
    enable_gqa: bool = False,
    seqused_k: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    num_splits: int | None = None,
) -> bool:
    if not query.is_cuda or not _should_use_cudnn(query.device.index):
        return False
    if max_q <= 128:
        return False
    if query.dtype not in (torch.float16, torch.bfloat16):
        return False
    if query.shape[-1] % 8 != 0 or value.shape[-1] % 8 != 0:
        return False
    if window_size != [-1, -1]:
        return False
    if enable_gqa or query.size(-2) != key.size(-2):
        return False
    if num_splits is not None or (block_table is not None and seqused_k is None):
        return False
    return True


class AuxRequest(NamedTuple):
    """
    Request which auxiliary outputs to compute from varlen_attn.

    Each field is a boolean indicating whether that auxiliary output should be computed.
    """

    lse: bool = False


@torch.library.custom_op("torch_attn::_varlen_attn", mutates_args={})
def _varlen_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor | None,
    max_q: int,
    max_k: int,
    is_causal: bool = False,
    scale: float | None = None,
    window_size: list[int] | None = None,
    enable_gqa: bool = False,
    seqused_k: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    num_splits: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Private custom op for variable-length attention.

    This is the internal implementation. Users should use the public varlen_attn function instead.
    """
    window_size = _normalize_window_size(window_size)
    use_cudnn = _can_use_cudnn(
        query,
        key,
        value,
        max_q,
        window_size,
        enable_gqa,
        seqused_k,
        block_table,
        num_splits,
    )

    if use_cudnn:
        log.info("Using cuDNN backend for varlen_attn")
        result = torch.ops.aten._cudnn_attention_forward(
            query=query,
            key=key,
            value=value,
            attn_bias=None,
            cum_seq_q=cu_seq_q,
            cum_seq_k=cu_seq_k,
            max_q=max_q,
            max_k=max_k,
            compute_log_sumexp=True,
            dropout_p=0.0,  # dropout_p hardcoded to 0.0
            is_causal=is_causal,
            return_debug_mask=False,  # return_debug_mask
            scale=scale,
            seqused_k=seqused_k,
            block_table=block_table,
        )
        # cuDNN returns: (output, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, debug_attn_mask)
        output, softmax_lse, rng_state = result[0], result[1], result[6]
    else:
        log.info("Using Flash Attention backend for varlen_attn")
        output, softmax_lse, rng_state, _, _ = torch.ops.aten._flash_attention_forward(
            query,
            key,
            value,
            cu_seq_q,
            cu_seq_k,
            max_q,
            max_k,
            0.0,  # dropout_p hardcoded to 0.0
            is_causal,
            return_debug_mask=False,
            scale=scale,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            seqused_k=seqused_k,
            block_table=block_table,
            num_splits=num_splits,
        )

    rng_state_ = torch.zeros(
        (2,), dtype=torch.uint64, device=query.device
    )  # hardcoded since dropout is hardcoded to 0
    return output, softmax_lse, rng_state_


@_varlen_attn.register_fake
def _varlen_attn_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor | None,
    max_q: int,
    max_k: int,
    is_causal: bool = False,
    scale: float | None = None,
    window_size: list[int] | None = None,
    enable_gqa: bool = False,
    seqused_k: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    num_splits: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fake implementation for meta tensor computation and tracing.

    Based on the 3D varlen path from meta__flash_attention_forward:
    - query shape: (total, num_heads, head_dim)
    - logsumexp shape: (num_heads, total_q)
    """
    window_size = _normalize_window_size(window_size)

    # Output has same shape as query
    output = torch.empty_like(query)

    # For varlen path: logsumexp shape is (num_heads, total_q)
    total_q = query.size(0)
    num_heads = query.size(1)
    logsumexp = torch.empty(
        (num_heads, total_q), dtype=torch.float, device=query.device
    )

    rng_state = torch.empty((2,), dtype=torch.uint64, device=query.device)

    return output, logsumexp, rng_state


def varlen_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor | None,
    max_q: int,
    max_k: int,
    *,
    return_aux: AuxRequest | None = None,
    scale: float | None = None,
    window_size: tuple[int, int] = (-1, -1),
    enable_gqa: bool = False,
    seqused_k: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    num_splits: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    r"""Compute variable-length attention using Flash Attention.

    This function is similar to scaled_dot_product_attention but optimized for
    variable-length sequences using cumulative sequence position tensors.

    Args:
        query (Tensor): Query tensor; shape :math:`(T_q, H_q, D)`
        key (Tensor): Key tensor; shape :math:`(T_k, H_{kv}, D)`, or
            :math:`(\text{total\_pages}, \text{page\_size}, H_{kv}, D)` when ``block_table`` is provided.
        value (Tensor): Value tensor; shape :math:`(T_k, H_{kv}, D)`, or
            :math:`(\text{total\_pages}, \text{page\_size}, H_{kv}, D)` when ``block_table`` is provided.
        cu_seq_q (Tensor): Cumulative sequence positions for queries; shape :math:`(N+1,)`
        cu_seq_k (Tensor): Cumulative sequence positions for keys/values; shape :math:`(N+1,)`
        max_q (int): Maximum query sequence length in the batch.
        max_k (int): Maximum key/value sequence length in the batch.
        return_aux (Optional[AuxRequest]): If not None and ``return_aux.lse`` is True, also returns the logsumexp tensor.
        scale (float, optional): Scaling factor for attention scores
        window_size (tuple[int, int], optional): Window size for sliding window attention as (left, right).
            Use (-1, -1) for full attention (default), (-1, 0) for causal attention,
            or (W, 0) for causal attention with sliding window of size W.
        enable_gqa (bool): If set to True, enables Grouped Query Attention (GQA)
            and allows key/value to have fewer heads than query.
            Each KV head is shared by a group of :math:`H_q / H_{kv}` query heads,
            so :math:`H_q` must be divisible by :math:`H_{kv}`.
            Default is False.
        seqused_k (Tensor, optional): Number of valid KV tokens per batch element; shape :math:`(N,)`.
            When set, only the first ``seqused_k[i]`` tokens in the key/value sequence for batch
            element *i* participate in attention. Useful for KV-cache decoding where the cache slot
            is larger than the actual sequence. Inference-only (not supported in backward).
        block_table (Tensor, optional): Block table for paged KV cache; shape
            :math:`(N, \text{max\_pages\_per\_seq})`, dtype ``int32``.
            Requires ``seqused_k``. Inference-only (not supported in backward).

            When ``block_table`` is provided, ``key`` and ``value`` are a "pool" of
            pages of tokens of KV data and the pages belong to any sequence/order.
            The ``block_table`` is what maps each sequence's logical chunks
            back to physical pages in this pool.

            ``seqused_k[i]`` tells the kernel how many tokens in sequence *i* are
            actually valid, since the last page is typically only partially filled.
        num_splits (int, optional): Number of splits for split-KV. Set to ``1``
            to disable split-KV which enables batch invariance. Split-KV
            parallelizes the key/value sequence dimension across multiple thread
            blocks and combines partial results. The split decision depends
            on ``max_k`` (the longest sequence in the batch), so different batch
            compositions can change the reduction order and produce different
            floating-point results for the same sequence. When this is disabled,
            bitwise identical outputs are guaranteed for a given sequence
            regardless of what other sequences are in the batch, at the
            cost of lower GPU utilization when there are few queries. When
            ``None`` (default), the kernel chooses automatically.

    Returns:
        output (Tensor): Output tensor from attention computation; shape :math:`(T_q, H_q, D)`.

        If ``return_aux`` is not None and ``return_aux.lse`` is True:
            lse (Tensor): Log-sum-exp of attention scores; shape :math:`(H_q, T_q)`.

    Shape legend:
        - :math:`N`: Batch size
        - :math:`T_q`: Total number of query tokens in the batch (sum of all query sequence lengths)
        - :math:`T_k`: Total number of key/value tokens in the batch (sum of all key/value sequence lengths)
        - :math:`H_q`: Number of query attention heads
        - :math:`H_{kv}`: Number of key/value attention heads (equal to :math:`H_q` unless GQA is enabled)
        - :math:`D`: Head dimension

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> batch_size, max_seq_len, embed_dim, num_heads = 2, 512, 1024, 16
        >>> head_dim = embed_dim // num_heads
        >>> seq_lengths = []
        >>> for _ in range(batch_size):
        ...     length = torch.randint(1, max_seq_len // 64 + 1, (1,)).item() * 64
        ...     seq_lengths.append(min(length, max_seq_len))
        >>> seq_lengths = torch.tensor(seq_lengths, device="cuda")
        >>> total_tokens = seq_lengths.sum().item()
        >>>
        >>> # Create packed query, key, value tensors
        >>> query = torch.randn(
        ...     total_tokens, num_heads, head_dim, dtype=torch.float16, device="cuda"
        ... )
        >>> key = torch.randn(
        ...     total_tokens, num_heads, head_dim, dtype=torch.float16, device="cuda"
        ... )
        >>> value = torch.randn(
        ...     total_tokens, num_heads, head_dim, dtype=torch.float16, device="cuda"
        ... )
        >>>
        >>> # Build cumulative sequence tensor
        >>> cu_seq = torch.zeros(batch_size + 1, device="cuda", dtype=torch.int32)
        >>> cu_seq[1:] = seq_lengths.cumsum(0)
        >>> max_len = seq_lengths.max().item()
        >>>
        >>> # Call varlen_attn
        >>> output = varlen_attn(
        ...     query, key, value, cu_seq, cu_seq, max_len, max_len
        ... )
    """

    num_heads_q = query.size(1)
    num_heads_k = key.size(2) if block_table is not None else key.size(1)
    if not enable_gqa and num_heads_q != num_heads_k:
        raise ValueError(
            f"Expect query and key/value to have the same number of heads "
            f"but got Hq={num_heads_q} and Hkv={num_heads_k}. "
            f"Try setting enable_gqa=True for GQA."
        )
    if enable_gqa and num_heads_q % num_heads_k != 0:
        raise ValueError(
            f"Expect number of query heads to be a multiple of kv heads for GQA "
            f"but got Hq={num_heads_q} and Hkv={num_heads_k}."
        )

    is_causal = window_size == (-1, 0)
    out, lse, _ = torch.ops.torch_attn._varlen_attn(
        query,
        key,
        value,
        cu_seq_q,
        cu_seq_k,
        max_q,
        max_k,
        is_causal,
        scale,
        list(window_size),
        enable_gqa,
        seqused_k,
        block_table,
        num_splits,
    )
    if return_aux is not None and return_aux.lse:
        return out, lse
    return out


@torch.library.custom_op("torch_attn::_varlen_attn_out", mutates_args={"out"})
def _varlen_attn_out(
    out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor | None,
    max_q: int,
    max_k: int,
    is_causal: bool = False,
    scale: float | None = None,
    window_size: list[int] | None = None,
    enable_gqa: bool = False,
    seqused_k: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    num_splits: int | None = None,
) -> torch.Tensor:
    """
    Private custom op for variable-length attention with pre-allocated output.
    Same as _varlen_attn but writes the attention output into the provided out tensor.
    """
    window_size = _normalize_window_size(window_size)

    log.info("Using Flash Attention backend for varlen_attn_out")
    softmax_lse = torch.ops.aten._flash_attention_forward_no_dropout_inplace(
        out,
        query,
        key,
        value,
        cu_seq_q,
        cu_seq_k,
        max_q,
        max_k,
        0.0,  # dropout_p hardcoded to 0.0
        is_causal,
        False,  # return_debug_mask
        scale=scale,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        seqused_k=seqused_k,
        block_table=block_table,
        num_splits=num_splits,
    )

    return softmax_lse


@_varlen_attn_out.register_fake
def _varlen_attn_out_fake(
    out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor | None,
    max_q: int,
    max_k: int,
    is_causal: bool = False,
    scale: float | None = None,
    window_size: list[int] | None = None,
    enable_gqa: bool = False,
    seqused_k: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    num_splits: int | None = None,
) -> torch.Tensor:
    """
    Fake implementation for meta tensor computation and tracing.
    """
    total_q = query.size(0)
    num_heads = query.size(1)
    logsumexp = torch.empty(
        (num_heads, total_q), dtype=torch.float, device=query.device
    )

    return logsumexp


def varlen_attn_out(
    out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor | None,
    max_q: int,
    max_k: int,
    *,
    return_aux: AuxRequest | None = None,
    scale: float | None = None,
    window_size: tuple[int, int] = (-1, -1),
    enable_gqa: bool = False,
    seqused_k: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    num_splits: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    r"""Compute variable-length attention using Flash Attention with a pre-allocated output tensor.

    Same as :func:`varlen_attn` but writes the attention output into the provided ``out`` tensor
    instead of allocating a new one.

    """
    num_heads_q = query.size(1)
    num_heads_k = key.size(2) if block_table is not None else key.size(1)
    if not enable_gqa and num_heads_q != num_heads_k:
        raise ValueError(
            f"Expect query and key/value to have the same number of heads "
            f"but got Hq={num_heads_q} and Hkv={num_heads_k}. "
            f"Try setting enable_gqa=True for GQA."
        )
    if enable_gqa and num_heads_q % num_heads_k != 0:
        raise ValueError(
            f"Expect number of query heads to be a multiple of kv heads for GQA "
            f"but got Hq={num_heads_q} and Hkv={num_heads_k}."
        )

    is_causal = window_size == (-1, 0)
    lse = torch.ops.torch_attn._varlen_attn_out(
        out,
        query,
        key,
        value,
        cu_seq_q,
        cu_seq_k,
        max_q,
        max_k,
        is_causal,
        scale,
        list(window_size),
        enable_gqa,
        seqused_k,
        block_table,
        num_splits,
    )
    if return_aux is not None and return_aux.lse:
        return out, lse
    return out


def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    (
        query,
        key,
        value,
        cu_seq_q,
        cu_seq_k,
        max_q,
        max_k,
        is_causal,
        scale,
        window_size,
        enable_gqa,
        seqused_k,
        block_table,
        num_splits,
    ) = inputs
    out, lse, rng_state = output

    if seqused_k is not None:
        raise RuntimeError("seqused_k is an inference-only parameter.")
    if block_table is not None:
        raise RuntimeError("block_table is an inference-only parameter.")

    ctx.save_for_backward(query, key, value, cu_seq_q, cu_seq_k, out, lse, rng_state)

    ctx.max_q = max_q
    ctx.max_k = max_k
    ctx.is_causal = is_causal
    ctx.scale = scale
    ctx.window_size = window_size


@torch.library.custom_op("torch_attn::_varlen_attn_backward", mutates_args={})
def _varlen_attn_backward(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    is_causal: bool,
    rng_state: torch.Tensor,
    scale: float | None = None,
    window_size: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    window_size = _normalize_window_size(window_size)

    unused = torch.empty(0, device=query.device)

    use_cudnn = _can_use_cudnn(query, key, value, max_q, window_size)

    if use_cudnn:
        log.info("Using cuDNN backend for varlen_attn")
        if window_size[0] != -1 or window_size[1] != -1:
            raise RuntimeError(
                "cuDNN backend does not support window attention. Please use Flash Attention backend."
            )
        dq, dk, dv = torch.ops.aten._cudnn_attention_backward(
            grad_out=grad_out,
            query=query,
            key=key,
            value=value,
            out=out,
            logsumexp=lse,
            cum_seq_q=cu_seq_q,
            cum_seq_k=cu_seq_k,
            max_q=max_q,
            max_k=max_k,
            dropout_p=0.0,
            philox_seed=rng_state,
            philox_offset=rng_state,  # should be unused
            attn_bias=None,
            is_causal=is_causal,
            scale=scale,
        )
    else:
        log.info("Using Flash Attention backend for varlen_attn")
        dq, dk, dv = torch.ops.aten._flash_attention_backward(
            grad_out,
            query,
            key,
            value,
            out,
            lse,
            cu_seq_q,
            cu_seq_k,
            max_q,
            max_k,
            0.0,
            is_causal,
            rng_state,
            unused,
            scale=scale,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
        )
    return dq, dk, dv


@_varlen_attn_backward.register_fake
def _varlen_attn_backward_fake(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    is_causal: bool,
    rng_state: torch.Tensor,
    scale: float | None = None,
    window_size: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fake implementation for meta tensor computation and tracing.
    """
    window_size = _normalize_window_size(window_size)

    grad_query = torch.empty_like(query)
    grad_key = torch.empty_like(key)
    grad_value = torch.empty_like(value)

    return grad_query, grad_key, grad_value


def _backward(
    ctx: Any, grad_out: torch.Tensor, grad_lse: torch.Tensor, grad_rng: torch.Tensor
) -> tuple[torch.Tensor | None, ...]:
    query, key, value, cu_seq_q, cu_seq_k, out, lse, rng_state = ctx.saved_tensors

    max_q = ctx.max_q
    max_k = ctx.max_k
    is_causal = ctx.is_causal
    scale = ctx.scale
    window_size = ctx.window_size

    dq, dk, dv = torch.ops.torch_attn._varlen_attn_backward(
        grad_out,
        query,
        key,
        value,
        out,
        lse,
        cu_seq_q,
        cu_seq_k,
        max_q,
        max_k,
        is_causal,
        rng_state,
        scale,
        window_size,
    )
    # cu_seq_q, cu_seq_k, max_q, max_k, is_causal, scale, window_size, \
    # enable_gqa, seqused_k, block_table, num_splits
    num_params = 11
    return (dq, dk, dv, *((None,) * num_params))


_varlen_attn.register_autograd(_backward, setup_context=_setup_context)

torch._dynamo.disallow_in_graph(
    torch.ops.aten._flash_attention_forward_no_dropout_inplace
)

from torch.utils.flop_counter import (
    _varlen_attn_backward_flop,
    _varlen_attn_forward_flop,
    _varlen_attn_out_flop,
    flop_registry,
)


flop_registry[torch.ops.torch_attn._varlen_attn] = _varlen_attn_forward_flop
flop_registry[torch.ops.torch_attn._varlen_attn_out] = _varlen_attn_out_flop
flop_registry[torch.ops.torch_attn._varlen_attn_backward] = _varlen_attn_backward_flop
<|endoftext|>"""
PROTOTYPE!
Flash Attention 3 implementation.
For fp8: only supports forward pass right now.
For fp16/bf16: supports forward and backward pass.
"""
# mypy: allow-untyped-defs

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

from dataclasses import dataclass
from functools import cache
from typing_extensions import TypeVarTuple, Unpack

import torch
from torch.library import Library

from . import _registry


__all__ = [
    "register_flash_attention_fa3",
]


_FA3_CUDA_FWD: Callable | None = None  # Cache for torch.ops.flash_attn_3.fwd
_FA3_CUDA_BWD: Callable | None = None  # Cache for torch.ops.flash_attn_3.bwd


@dataclass
class _FA3Handle:
    library: Library | None

    def remove(self) -> None:
        self.library = None
        # Clear the C++ flag
        torch._C._set_sdp_use_fa3(False)


@cache
def _get_device_major(device: torch.device) -> int:
    major, _ = torch.cuda.get_device_capability(device)
    return major


def register_flash_attention_fa3(
    module_path: str = "flash_attn_interface",
) -> _FA3Handle:
    """
    Register FA3 flash attention kernels with the PyTorch dispatcher.

    Args:
        module_path: Python module path to the FA3 implementation.
    """
    _fa3_import_module(module_path)

    # Expose FA3 registration status to C++
    torch._C._set_sdp_use_fa3(True)

    return _FA3Handle(_fa3_register_kernels())


def _fa3_import_module(module_path: str) -> None:
    importlib.import_module(module_path)
    if not hasattr(torch.ops, "flash_attn_3"):
        raise RuntimeError(f"Module '{module_path}' does not expose FA3 kernels")
    if not hasattr(torch.ops.flash_attn_3, "fwd"):
        raise RuntimeError(
            f"Module '{module_path}' does not expose FA3 forward kernels"
        )
    if not hasattr(torch.ops.flash_attn_3, "bwd"):
        raise RuntimeError(
            f"Module '{module_path}' does not expose FA3 backward kernels"
        )
    global _FA3_CUDA_FWD, _FA3_CUDA_BWD
    _FA3_CUDA_FWD = torch.ops.flash_attn_3.fwd
    _FA3_CUDA_BWD = torch.ops.flash_attn_3.bwd


def _fa3_register_kernels() -> Library:
    lib = Library("aten", "IMPL", "CUDA")
    lib.impl(
        "_flash_attention_forward.quantized", _fa3_flash_attention_forward_impl, "CUDA"
    )
    lib.impl(
        "_scaled_dot_product_flash_attention.quantized",
        _fa3_scaled_dot_product_flash_attention_forward_impl,
        "CUDA",
    )
    lib.impl(
        "_flash_attention_forward", _fa3_flash_attention_forward_impl_default, "CUDA"
    )
    lib.impl(
        "_flash_attention_forward_no_dropout_inplace",
        _fa3_flash_attention_forward_no_dropout_inplace_impl,
        "CUDA",
    )
    lib.impl(
        "_scaled_dot_product_flash_attention",
        _fa3_scaled_dot_product_flash_attention_forward_impl_default,
        "CUDA",
    )

    lib.impl("_flash_attention_backward", _fa3_flash_attention_backward_impl, "CUDA")
    lib.impl(
        "_scaled_dot_product_flash_attention_backward",
        _fa3_scaled_dot_product_flash_attention_backward_impl,
        "CUDA",
    )
    return lib


def _fa3_common_support_error(
    query: torch.Tensor,
    tensors: tuple[torch.Tensor, ...],
    dropout_p: float,
    cum_seq_q: torch.Tensor | None,
    q_descale: torch.Tensor | None,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
) -> str | None:
    if dropout_p != 0.0:
        return "dropout_p must be 0"

    if not all(t.is_cuda for t in tensors):
        return "inputs must be CUDA tensors"
    if len({t.device for t in tensors}) != 1:
        return "inputs must share device"
    if query.dtype == torch.float8_e4m3fn and (
        q_descale is None or k_descale is None or v_descale is None
    ):
        warnings.warn(
            "When using SDPA with fp8, descale tensor should always be used"
            " for accurate dequantization. Please use "
            "_scaled_dot_product_attention_quantized and "
            "provide the descale tensors.",
            UserWarning,
        )
    if cum_seq_q is None and query.dim() != 4:
        return "dense query must be 4D"
    if cum_seq_q is not None and query.dim() != 3:
        return "ragged query must be 3D"
    if not torch.cuda.is_available():
        return "CUDA not available"
    if _get_device_major(query.device) != 9:
        return "FA3 requires compute capability 9.0"
    return None


def _fa3_forward_support_error(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float,
    return_debug_mask: bool,
    alibi_slopes: torch.Tensor | None,
    seqused_k: torch.Tensor | None,
    cum_seq_q: torch.Tensor | None,
    q_descale: torch.Tensor | None,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
) -> str | None:
    if return_debug_mask:
        return "return_debug_mask must be False"
    if alibi_slopes is not None:
        return "alibi_slopes not supported"
    if seqused_k is not None:
        if seqused_k.dtype != torch.int32:
            return "seqused_k must be int32"
        if not seqused_k.is_cuda:
            return "seqused_k must be CUDA"
    supported_dtypes = (torch.float8_e4m3fn, torch.float16, torch.bfloat16)
    if not all(t.dtype in supported_dtypes for t in {query, key, value}):
        return f"inputs must be one of {supported_dtypes}"
    if len({t.dtype for t in {query, key, value}}) != 1:
        return "all inputs must have the same dtype"
    error = _fa3_common_support_error(
        query,
        (query, key, value),
        dropout_p,
        cum_seq_q,
        q_descale,
        k_descale,
        v_descale,
    )
    if error is not None:
        if error == "inputs must share device":
            return "query, key, value must be on same device"
        return error
    return None


def _fa3_backward_support_error(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    dropout_p: float,
    cum_seq_q: torch.Tensor | None,
    window_size_left: int | None,
    window_size_right: int | None,
) -> str | None:
    # FA3 backward ONLY supports fp16/bf16, NOT fp8
    if query.dtype == torch.float8_e4m3fn:
        return (
            "FA3 backward does not support fp8 - use inference only (torch.no_grad())"
        )
    if logsumexp.dtype != torch.float32:
        return "logsumexp dtype must be float32"
    supported_dtypes = (torch.float16, torch.bfloat16)
    if not all(t.dtype in supported_dtypes for t in {grad_out, query, key, value, out}):
        return f"inputs must be one of {supported_dtypes}"
    if len({t.dtype for t in {grad_out, query, key, value, out}}) != 1:
        return "all inputs must have the same dtype"
    error = _fa3_common_support_error(
        query,
        (grad_out, query, key, value, out, logsumexp),
        dropout_p,
        cum_seq_q,
        None,
        None,
        None,
    )
    if error is not None:
        return error
    return None


Ts = TypeVarTuple("Ts")


def _transpose_dense(*tensors: Unpack[Ts]) -> tuple[Unpack[Ts]]:
    return tuple(t.transpose(1, 2) for t in tensors)  # type: ignore[attr-defined]


def _maybe_contiguous(x: torch.Tensor | None) -> torch.Tensor | None:
    """Ensure tensor is contiguous in the last dimension."""
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _fa3_run_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_q: torch.Tensor | None,
    cu_seq_k: torch.Tensor | None,
    max_q: int,
    max_k: int,
    scale: float | None,
    is_causal: bool,
    window_size_left: int | None,
    window_size_right: int | None,
    seqused_k: torch.Tensor | None,
    out: torch.Tensor | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    num_splits: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the FA3 forward pass by calling the C++ kernel directly.
    """
    if _FA3_CUDA_FWD is None:
        raise RuntimeError("FA3 not registered")
    # Ensure contiguous in the last dimension
    q = _maybe_contiguous(query)
    k = _maybe_contiguous(key)
    v = (
        value.contiguous()
        if value.dtype == torch.float8_e4m3fn
        and value.stride(-1) != 1
        and value.stride(-3) != 1
        else _maybe_contiguous(value)
    )

    cu_seqlens_q = _maybe_contiguous(cu_seq_q)
    cu_seqlens_k = _maybe_contiguous(cu_seq_k)
    seqused_k = _maybe_contiguous(seqused_k)
    block_table = _maybe_contiguous(block_table)

    out, softmax_lse, out_accum, softmax_lse_accum = _FA3_CUDA_FWD(
        q,
        k,
        v,
        None,  # k_new
        None,  # v_new
        None,  # qv
        out,  # out_ (pre-allocated output)
        cu_seqlens_q,  # cu_seqlens_q
        cu_seqlens_k,  # cu_seqlens_k
        None,  # cu_seqlens_k_new
        None,  # seqused_q
        seqused_k,  # seqused_k
        max_q,  # max_seqlen_q
        max_k,  # max_seqlen_k
        block_table,  # block_table,
        None,  # kv_batch_idx,
        None,  # leftpad_k,
        None,  # rotary_cos,
        None,  # rotary_sin,
        None,  # seqlens_rotary,
        q_descale,  # q_descale,
        k_descale,  # k_descale,
        v_descale,  # v_descale,
        scale,  # softmax_scale,
        is_causal,  # causal,
        window_size_left if window_size_left is not None else -1,  # window_size_left
        window_size_right if window_size_right is not None else -1,  # window_size_right
        0,  # attention_chunk,
        0.0,  # softcap,
        True,  # rotary_interleaved,
        None,  # scheduler_metadata,
        num_splits
        or (1 if torch.are_deterministic_algorithms_enabled() else 0),  # num_splits,
        None,  # pack_gqa,
        torch._C._get_sm_carveout_experimental() or 0,  # sm_margin,
    )
    return out, softmax_lse.contiguous()


def _fa3_run_backward(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    cu_seq_q: torch.Tensor | None,
    cu_seq_k: torch.Tensor | None,
    max_seqlen_q: int | None,
    max_seqlen_k: int | None,
    scale: float | None,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if _FA3_CUDA_BWD is None:
        raise RuntimeError("FA3 not registered")

    # Ensure contiguous
    dout = _maybe_contiguous(grad_out)
    q = query.contiguous() if query.stride(-1) != 1 else query
    k = key.contiguous() if key.stride(-1) != 1 else key
    v = value.contiguous() if value.stride(-1) != 1 else value
    o = _maybe_contiguous(out)
    lse = _maybe_contiguous(logsumexp)

    # Pre-allocate gradient tensors
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    _FA3_CUDA_BWD(
        dout,
        q,
        k,
        v,
        o,
        lse,
        dq,
        dk,
        dv,
        cu_seq_q,
        cu_seq_k,
        None,
        None,
        max_seqlen_q,
        max_seqlen_k,
        scale,
        is_causal,
        window_size_left,
        window_size_right,
        0.0,
        deterministic,
        torch._C._get_sm_carveout_experimental() or 0,
    )
    return dq, dk, dv


def _fa3_flash_attention_forward_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cum_seq_q: torch.Tensor | None,
    cum_seq_k: torch.Tensor | None,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    return_debug_mask: bool,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    *,
    scale: float | None = None,
    window_size_left: int = -1,
    window_size_right: int = -1,
    seqused_k: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    compute_auxiliary: bool = True,
    num_splits: int | None = None,
):
    error = _fa3_forward_support_error(
        query,
        key,
        value,
        dropout_p,
        return_debug_mask,
        alibi_slopes,
        seqused_k,
        cum_seq_q,
        q_descale,
        k_descale,
        v_descale,
    )
    if error is not None:
        raise RuntimeError(f"FA3 flash_attention forward unsupported: {error}")
    out, lse = _fa3_run_forward(
        query,
        key,
        value,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        scale,
        is_causal,
        window_size_left,
        window_size_right,
        seqused_k,
        out,
        q_descale,
        k_descale,
        v_descale,
        block_table,
        num_splits,
    )
    if compute_auxiliary:
        rng_state = torch.zeros((2,), dtype=torch.uint64, device=query.device)
        philox_offset = torch.zeros((), dtype=torch.uint64, device=query.device)
        debug_mask = torch.empty(0, dtype=query.dtype, device=query.device)
    else:
        rng_state = None
        philox_offset = None
        debug_mask = None
    return out, lse, rng_state, philox_offset, debug_mask


def _fa3_flash_attention_forward_no_dropout_inplace_impl(
    out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cum_seq_q: torch.Tensor | None,
    cum_seq_k: torch.Tensor | None,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    return_debug_mask: bool,
    *,
    scale: float | None = None,
    window_size_left: int = -1,
    window_size_right: int = -1,
    seqused_k: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    num_splits: int | None = None,
):
    _, lse, _, _, _ = _fa3_flash_attention_forward_impl(
        query,
        key,
        value,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        return_debug_mask,
        None,
        None,
        None,
        scale=scale,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        seqused_k=seqused_k,
        alibi_slopes=alibi_slopes,
        out=out,
        block_table=block_table,
        compute_auxiliary=False,
        num_splits=num_splits,
    )
    return lse


def _fa3_flash_attention_forward_impl_default(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cum_seq_q: torch.Tensor | None,
    cum_seq_k: torch.Tensor | None,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    return_debug_mask: bool,
    *,
    scale: float | None = None,
    window_size_left: int = -1,
    window_size_right: int = -1,
    seqused_k: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    num_splits: int | None = None,
):
    return _fa3_flash_attention_forward_impl(
        query,
        key,
        value,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        return_debug_mask,
        None,
        None,
        None,
        scale=scale,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        seqused_k=seqused_k,
        alibi_slopes=alibi_slopes,
        out=out,
        block_table=block_table,
        num_splits=num_splits,
    )


def _fa3_flash_attention_backward_impl(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    cum_seq_q: torch.Tensor | None,
    cum_seq_k: torch.Tensor | None,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    rng_state: torch.Tensor,
    unused: torch.Tensor,
    *,
    scale: float | None = None,
    window_size_left: int | None = None,
    window_size_right: int | None = None,
):
    """FA3 implementation of _flash_attention_backward."""
    error = _fa3_backward_support_error(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        dropout_p,
        cum_seq_q,
        window_size_left,
        window_size_right,
    )

    if error is not None:
        raise RuntimeError(f"FA3 flash_attention backward unsupported: {error}")

    deterministic = torch.are_deterministic_algorithms_enabled()

    dq, dk, dv = _fa3_run_backward(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        scale,
        is_causal,
        window_size_left if window_size_left is not None else -1,
        window_size_right if window_size_right is not None else -1,
        deterministic,
    )
    return dq, dk, dv


def _fa3_scaled_dot_product_flash_attention_forward_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: float | None = None,
):
    error = _fa3_forward_support_error(
        query,
        key,
        value,
        dropout_p,
        return_debug_mask,
        None,
        None,
        None,
        q_descale,
        k_descale,
        v_descale,
    )
    if error is not None:
        raise RuntimeError(f"FA3 SDPA forward unsupported: {error}")
    q, k, v = _transpose_dense(query, key, value)

    # Pre-allocate output with query's strides (BHSD layout), then create
    # a BSHD view for the kernel. This ensures the returned output has
    # the same memory layout as the input query.
    out_dtype = torch.bfloat16 if query.dtype == torch.float8_e4m3fn else query.dtype
    out_bhsd = torch.empty_like(query, dtype=out_dtype)
    out_bshd = out_bhsd.transpose(1, 2)

    max_q_flash = q.size(1)
    max_k_flash = k.size(1)
    _, lse, rng_state, philox_offset, debug_mask = _fa3_flash_attention_forward_impl(
        q,
        k,
        v,
        None,
        None,
        max_q_flash,
        max_k_flash,
        dropout_p,
        is_causal,
        return_debug_mask,
        scale=scale,
        out=out_bshd,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    max_q = query.size(2)
    max_k = key.size(2)
    return (
        out_bhsd,
        lse,
        None,
        None,
        max_q,
        max_k,
        rng_state,
        philox_offset,
        debug_mask,
    )


def _fa3_scaled_dot_product_flash_attention_forward_impl_default(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: float | None = None,
):
    return _fa3_scaled_dot_product_flash_attention_forward_impl(
        query,
        key,
        value,
        None,
        None,
        None,
        dropout_p,
        is_causal,
        return_debug_mask,
        scale=scale,
    )


def _fa3_scaled_dot_product_flash_attention_backward_impl(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    cum_seq_q: torch.Tensor | None,
    cum_seq_k: torch.Tensor | None,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    *,
    scale: float | None = None,
):
    """FA3 implementation of _scaled_dot_product_flash_attention_backward."""
    error = _fa3_backward_support_error(
        grad_out, query, key, value, out, logsumexp, dropout_p, None, None, None
    )
    if error is not None:
        raise RuntimeError(f"FA3 SDPA backward unsupported: {error}")

    # SDPA uses BHSD layout, FA3 uses BSHD - transpose
    grad_out_t, q_t, k_t, v_t, out_t = _transpose_dense(
        grad_out, query, key, value, out
    )

    dq, dk, dv = _fa3_flash_attention_backward_impl(
        grad_out_t,
        q_t,
        k_t,
        v_t,
        out_t,
        logsumexp,
        None,  # cum_seq_q (dense attention)
        None,  # cum_seq_k
        max_q,  # max_seqlen_q
        max_k,  # max_seqlen_k
        dropout_p,
        is_causal,
        philox_seed,
        philox_offset,
        scale=scale,
    )

    # Transpose gradients back to BHSD layout
    dq_out, dk_out, dv_out = _transpose_dense(dq, dk, dv)
    return dq_out, dk_out, dv_out


_registry.register_flash_attention_impl("FA3", register_fn=register_flash_attention_fa3)
<|endoftext|># mypy: allow-untyped-defs
"""
This module implements Paged Attention on top of flex_attention.
This module is experimental and subject to change.
"""

import torch
from torch.nn.attention.flex_attention import (
    _identity,
    _mask_mod_signature,
