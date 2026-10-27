              W_{out} = \left\lfloor\frac{W_{in} + 2 * \text{padding[1]} - \text{dilation[1]}
                    \times (\text{kernel\_size[1]} - 1) - 1}{\text{stride[1]}} + 1\right\rfloor

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.MaxPool2d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.MaxPool2d((3, 2), stride=(2, 1))
        >>> input = torch.randn(20, 16, 50, 32)
        >>> output = m(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t
    dilation: _size_2_t

    def forward(self, input: Tensor):
        """Runs the forward pass."""
        return F.max_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )


class MaxPool3d(_MaxPoolNd):
    r"""Applies a 3D max pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, D, H, W)`,
    output :math:`(N, C, D_{out}, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kD, kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            \text{out}(N_i, C_j, d, h, w) ={} & \max_{k=0, \ldots, kD-1} \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                              & \text{input}(N_i, C_j, \text{stride[0]} \times d + k,
                                                             \text{stride[1]} \times h + m, \text{stride[2]} \times w + n)
        \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly padded with negative infinity on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.
    It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimension
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: Implicit negative infinity padding to be added on all three sides
        dilation: a parameter that controls the stride of elements in the window
        return_indices: if ``True``, will return the max indices along with the outputs.
                        Useful for :class:`torch.nn.MaxUnpool3d` later
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or :math:`(C, D_{out}, H_{out}, W_{out})`, where

          .. math::
              D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] - \text{dilation}[0] \times
                (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] - \text{dilation}[1] \times
                (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] - \text{dilation}[2] \times
                (\text{kernel\_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.MaxPool3d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.MaxPool3d((3, 2, 2), stride=(2, 1, 2))
        >>> input = torch.randn(20, 16, 50, 44, 31)
        >>> output = m(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    kernel_size: _size_3_t
    stride: _size_3_t
    padding: _size_3_t
    dilation: _size_3_t

    def forward(self, input: Tensor):
        """Runs the forward pass."""
        return F.max_pool3d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )


class _MaxUnpoolNd(Module):
    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class MaxUnpool1d(_MaxUnpoolNd):
    r"""Computes a partial inverse of :class:`MaxPool1d`.

    :class:`MaxPool1d` is not fully invertible, since the non-maximal values are lost.

    :class:`MaxUnpool1d` takes in as input the output of :class:`MaxPool1d`
    including the indices of the maximal values and computes a partial inverse
    in which all non-maximal values are set to zero.

    Note:
        This operation may behave nondeterministically when the input indices has repeat values.
        See https://github.com/pytorch/pytorch/issues/80827 and :doc:`/notes/randomness` for more information.

    .. note:: :class:`MaxPool1d` can map several input sizes to the same output
              sizes. Hence, the inversion process can get ambiguous.
              To accommodate this, you can provide the needed output size
              as an additional argument :attr:`output_size` in the forward call.
              See the Inputs and Example below.

    Args:
        kernel_size (int or tuple): Size of the max pooling window.
        stride (int or tuple): Stride of the max pooling window.
            It is set to :attr:`kernel_size` by default.
        padding (int or tuple): Padding that was added to the input

    Inputs:
        - `input`: the input Tensor to invert
        - `indices`: the indices given out by :class:`~torch.nn.MaxPool1d`
        - `output_size` (optional): the targeted output size

    Shape:
        - Input: :math:`(N, C, H_{in})` or :math:`(C, H_{in})`.
        - Output: :math:`(N, C, H_{out})` or :math:`(C, H_{out})`, where

          .. math::
              H_{out} = (H_{in} - 1) \times \text{stride}[0] - 2 \times \text{padding}[0] + \text{kernel\_size}[0]

          or as given by :attr:`output_size` in the call operator

    Example::

        >>> # xdoctest: +IGNORE_WANT("do other tests modify the global state?")
        >>> pool = nn.MaxPool1d(2, stride=2, return_indices=True)
        >>> unpool = nn.MaxUnpool1d(2, stride=2)
        >>> input = torch.tensor([[[1., 2, 3, 4, 5, 6, 7, 8]]])
        >>> output, indices = pool(input)
        >>> unpool(output, indices)
        tensor([[[ 0.,  2.,  0.,  4.,  0.,  6.,  0., 8.]]])

        >>> # Example showcasing the use of output_size
        >>> input = torch.tensor([[[1., 2, 3, 4, 5, 6, 7, 8, 9]]])
        >>> output, indices = pool(input)
        >>> unpool(output, indices, output_size=input.size())
        tensor([[[ 0.,  2.,  0.,  4.,  0.,  6.,  0., 8.,  0.]]])

        >>> unpool(output, indices)
        tensor([[[ 0.,  2.,  0.,  4.,  0.,  6.,  0., 8.]]])
    """

    kernel_size: _size_1_t
    stride: _size_1_t
    padding: _size_1_t

    def __init__(
        self,
        kernel_size: _size_1_t,
        stride: _size_1_t | None = None,
        padding: _size_1_t = 0,
    ) -> None:
        super().__init__()
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride if (stride is not None) else kernel_size)
        self.padding = _single(padding)

    def forward(
        self, input: Tensor, indices: Tensor, output_size: list[int] | None = None
    ) -> Tensor:
        """Runs the forward pass."""
        return F.max_unpool1d(
            input, indices, self.kernel_size, self.stride, self.padding, output_size
        )


class MaxUnpool2d(_MaxUnpoolNd):
    r"""Computes a partial inverse of :class:`MaxPool2d`.

    :class:`MaxPool2d` is not fully invertible, since the non-maximal values are lost.

    :class:`MaxUnpool2d` takes in as input the output of :class:`MaxPool2d`
    including the indices of the maximal values and computes a partial inverse
    in which all non-maximal values are set to zero.

    Note:
        This operation may behave nondeterministically when the input indices has repeat values.
        See https://github.com/pytorch/pytorch/issues/80827 and :doc:`/notes/randomness` for more information.

    .. note:: :class:`MaxPool2d` can map several input sizes to the same output
              sizes. Hence, the inversion process can get ambiguous.
              To accommodate this, you can provide the needed output size
              as an additional argument :attr:`output_size` in the forward call.
              See the Inputs and Example below.

    Args:
        kernel_size (int or tuple): Size of the max pooling window.
        stride (int or tuple): Stride of the max pooling window.
            It is set to :attr:`kernel_size` by default.
        padding (int or tuple): Padding that was added to the input

    Inputs:
        - `input`: the input Tensor to invert
        - `indices`: the indices given out by :class:`~torch.nn.MaxPool2d`
        - `output_size` (optional): the targeted output size

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where

          .. math::
            H_{out} = (H_{in} - 1) \times \text{stride[0]} - 2 \times \text{padding[0]} + \text{kernel\_size[0]}

          .. math::
            W_{out} = (W_{in} - 1) \times \text{stride[1]} - 2 \times \text{padding[1]} + \text{kernel\_size[1]}

          or as given by :attr:`output_size` in the call operator

    Example::

        >>> pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        >>> unpool = nn.MaxUnpool2d(2, stride=2)
        >>> input = torch.tensor([[[[ 1.,  2.,  3.,  4.],
                                    [ 5.,  6.,  7.,  8.],
                                    [ 9., 10., 11., 12.],
                                    [13., 14., 15., 16.]]]])
        >>> output, indices = pool(input)
        >>> unpool(output, indices)
        tensor([[[[  0.,   0.,   0.,   0.],
                  [  0.,   6.,   0.,   8.],
                  [  0.,   0.,   0.,   0.],
                  [  0.,  14.,   0.,  16.]]]])
        >>> # Now using output_size to resolve an ambiguous size for the inverse
        >>> input = torch.tensor([[[[ 1.,  2.,  3.,  4.,  5.],
                                    [ 6.,  7.,  8.,  9., 10.],
                                    [11., 12., 13., 14., 15.],
                                    [16., 17., 18., 19., 20.]]]])
        >>> output, indices = pool(input)
        >>> # This call will not work without specifying output_size
        >>> unpool(output, indices, output_size=input.size())
        tensor([[[[ 0.,  0.,  0.,  0.,  0.],
                  [ 0.,  7.,  0.,  9.,  0.],
                  [ 0.,  0.,  0.,  0.,  0.],
                  [ 0., 17.,  0., 19.,  0.]]]])


    """

    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t

    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: _size_2_t | None = None,
        padding: _size_2_t = 0,
    ) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride if (stride is not None) else kernel_size)
        self.padding = _pair(padding)

    def forward(
        self, input: Tensor, indices: Tensor, output_size: list[int] | None = None
    ) -> Tensor:
        """Runs the forward pass."""
        return F.max_unpool2d(
            input, indices, self.kernel_size, self.stride, self.padding, output_size
        )


class MaxUnpool3d(_MaxUnpoolNd):
    r"""Computes a partial inverse of :class:`MaxPool3d`.

    :class:`MaxPool3d` is not fully invertible, since the non-maximal values are lost.
    :class:`MaxUnpool3d` takes in as input the output of :class:`MaxPool3d`
    including the indices of the maximal values and computes a partial inverse
    in which all non-maximal values are set to zero.

    Note:
        This operation may behave nondeterministically when the input indices has repeat values.
        See https://github.com/pytorch/pytorch/issues/80827 and :doc:`/notes/randomness` for more information.

    .. note:: :class:`MaxPool3d` can map several input sizes to the same output
              sizes. Hence, the inversion process can get ambiguous.
              To accommodate this, you can provide the needed output size
              as an additional argument :attr:`output_size` in the forward call.
              See the Inputs section below.

    Args:
        kernel_size (int or tuple): Size of the max pooling window.
        stride (int or tuple): Stride of the max pooling window.
            It is set to :attr:`kernel_size` by default.
        padding (int or tuple): Padding that was added to the input

    Inputs:
        - `input`: the input Tensor to invert
        - `indices`: the indices given out by :class:`~torch.nn.MaxPool3d`
        - `output_size` (optional): the targeted output size

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or :math:`(C, D_{out}, H_{out}, W_{out})`, where

          .. math::
              D_{out} = (D_{in} - 1) \times \text{stride[0]} - 2 \times \text{padding[0]} + \text{kernel\_size[0]}

          .. math::
              H_{out} = (H_{in} - 1) \times \text{stride[1]} - 2 \times \text{padding[1]} + \text{kernel\_size[1]}

          .. math::
              W_{out} = (W_{in} - 1) \times \text{stride[2]} - 2 \times \text{padding[2]} + \text{kernel\_size[2]}

          or as given by :attr:`output_size` in the call operator

    Example::

        >>> # pool of square window of size=3, stride=2
        >>> pool = nn.MaxPool3d(3, stride=2, return_indices=True)
        >>> unpool = nn.MaxUnpool3d(3, stride=2)
        >>> output, indices = pool(torch.randn(20, 16, 51, 33, 15))
        >>> unpooled_output = unpool(output, indices)
        >>> unpooled_output.size()
        torch.Size([20, 16, 51, 33, 15])
    """

    kernel_size: _size_3_t
    stride: _size_3_t
    padding: _size_3_t

    def __init__(
        self,
        kernel_size: _size_3_t,
        stride: _size_3_t | None = None,
        padding: _size_3_t = 0,
    ) -> None:
        super().__init__()
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride if (stride is not None) else kernel_size)
        self.padding = _triple(padding)

    def forward(
        self, input: Tensor, indices: Tensor, output_size: list[int] | None = None
    ) -> Tensor:
        """Runs the forward pass."""
        return F.max_unpool3d(
            input, indices, self.kernel_size, self.stride, self.padding, output_size
        )


class _AvgPoolNd(Module):
    __constants__ = [
        "kernel_size",
        "stride",
        "padding",
        "ceil_mode",
        "count_include_pad",
    ]

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class AvgPool1d(_AvgPoolNd):
    r"""Applies a 1D average pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, L)`,
    output :math:`(N, C, L_{out})` and :attr:`kernel_size` :math:`k`
    can be precisely described as:

    .. math::

        \text{out}(N_i, C_j, l) = \frac{1}{k} \sum_{m=0}^{k-1}
                               \text{input}(N_i, C_j, \text{stride} \times l + m)

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    .. note::
        pad should be at most half of effective kernel size.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding` can each be
    an ``int`` or a one-element tuple.

    Args:
        kernel_size: the size of the window
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
        count_include_pad: when True, will include the zero-padding in the averaging calculation

    Shape:
        - Input: :math:`(N, C, L_{in})` or :math:`(C, L_{in})`.
        - Output: :math:`(N, C, L_{out})` or :math:`(C, L_{out})`, where

          .. math::
              L_{out} = \left\lfloor \frac{L_{in} +
              2 \times \text{padding} - \text{kernel\_size}}{\text{stride}} + 1\right\rfloor

          Per the note above, if ``ceil_mode`` is True and :math:`(L_{out} - 1) \times \text{stride} \geq L_{in}
          + \text{padding}`, we skip the last window as it would start in the right padded region, resulting in
          :math:`L_{out}` being reduced by one.

    Examples::

        >>> # pool with window of size=3, stride=2
        >>> m = nn.AvgPool1d(3, stride=2)
        >>> m(torch.tensor([[[1., 2, 3, 4, 5, 6, 7]]]))
        tensor([[[2., 4., 6.]]])
    """

    kernel_size: _size_1_t
    stride: _size_1_t
    padding: _size_1_t
    ceil_mode: bool
    count_include_pad: bool

    def __init__(
        self,
        kernel_size: _size_1_t,
        stride: _size_1_t = None,
        padding: _size_1_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride if stride is not None else kernel_size)
        self.padding = _single(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input: Tensor) -> Tensor:
        """Runs the forward pass."""
        return F.avg_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
        )


class AvgPool2d(_AvgPoolNd):
    r"""Applies a 2D average pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::

        out(N_i, C_j, h, w)  = \frac{1}{kH * kW} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1}
                               input(N_i, C_j, stride[0] \times h + m, stride[1] \times w + n)

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    .. note::
        pad should be at most half of effective kernel size.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding` can either be:

        - a single ``int`` or a single-element tuple -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Args:
        kernel_size: the size of the window
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
        count_include_pad: when True, will include the zero-padding in the averaging calculation
        divisor_override: if specified, it will be used as divisor, otherwise size of the pooling region will be used.


    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] -
                \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] -
                \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor

          Per the note above, if ``ceil_mode`` is True and :math:`(H_{out} - 1)\times \text{stride}[0]\geq H_{in}
          + \text{padding}[0]`, we skip the last window as it would start in the bottom padded region,
          resulting in :math:`H_{out}` being reduced by one.

          The same applies for :math:`W_{out}`.

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.AvgPool2d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.AvgPool2d((3, 2), stride=(2, 1))
        >>> input = torch.randn(20, 16, 50, 32)
        >>> output = m(input)
    """

    __constants__ = [
        "kernel_size",
        "stride",
        "padding",
        "ceil_mode",
        "count_include_pad",
        "divisor_override",
    ]

    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t
    ceil_mode: bool
    count_include_pad: bool

    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: _size_2_t | None = None,
        padding: _size_2_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int | None = None,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, input: Tensor) -> Tensor:
        """Runs the forward pass."""
        return F.avg_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override,
        )


class AvgPool3d(_AvgPoolNd):
    r"""Applies a 3D average pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, D, H, W)`,
    output :math:`(N, C, D_{out}, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kD, kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            \text{out}(N_i, C_j, d, h, w) ={} & \sum_{k=0}^{kD-1} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1} \\
                                              & \frac{\text{input}(N_i, C_j, \text{stride}[0] \times d + k,
                                                      \text{stride}[1] \times h + m, \text{stride}[2] \times w + n)}
                                                     {kD \times kH \times kW}
        \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on all three sides
    for :attr:`padding` number of points.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    .. note::
        pad should be at most half of effective kernel size.

    The parameters :attr:`kernel_size`, :attr:`stride` can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimension
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    Args:
        kernel_size: the size of the window
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on all three sides
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
        count_include_pad: when True, will include the zero-padding in the averaging calculation
        divisor_override: if specified, it will be used as divisor, otherwise :attr:`kernel_size` will be used

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or
          :math:`(C, D_{out}, H_{out}, W_{out})`, where

          .. math::
              D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] -
                    \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] -
                    \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] -
                    \text{kernel\_size}[2]}{\text{stride}[2]} + 1\right\rfloor

          Per the note above, if ``ceil_mode`` is True and :math:`(D_{out} - 1)\times \text{stride}[0]\geq D_{in}
          + \text{padding}[0]`, we skip the last window as it would start in the padded region,
          resulting in :math:`D_{out}` being reduced by one.

          The same applies for :math:`W_{out}` and :math:`H_{out}`.

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.AvgPool3d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.AvgPool3d((3, 2, 2), stride=(2, 1, 2))
        >>> input = torch.randn(20, 16, 50, 44, 31)
        >>> output = m(input)
    """

    __constants__ = [
        "kernel_size",
        "stride",
        "padding",
        "ceil_mode",
        "count_include_pad",
        "divisor_override",
    ]

    kernel_size: _size_3_t
    stride: _size_3_t
    padding: _size_3_t
    ceil_mode: bool
    count_include_pad: bool

    def __init__(
        self,
        kernel_size: _size_3_t,
        stride: _size_3_t | None = None,
        padding: _size_3_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int | None = None,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, input: Tensor) -> Tensor:
        """Runs the forward pass."""
        return F.avg_pool3d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override,
        )

    def __setstate__(self, d):
        super().__setstate__(d)
        self.__dict__.setdefault("padding", 0)
        self.__dict__.setdefault("ceil_mode", False)
        self.__dict__.setdefault("count_include_pad", True)


class FractionalMaxPool2d(Module):
    r"""Applies a 2D fractional max pooling over an input signal composed of several input planes.

    Fractional MaxPooling is described in detail in the paper `Fractional MaxPooling`_ by Ben Graham

    The max-pooling operation is applied in :math:`kH \times kW` regions by a stochastic
    step size determined by the target output size.
    The number of output features is equal to the number of input planes.

    .. note:: Exactly one of ``output_size`` or ``output_ratio`` must be defined.

    Args:
        kernel_size: the size of the window to take a max over.
                     Can be a single number k (for a square kernel of k x k) or a tuple `(kh, kw)`
        output_size: the target output size of the image of the form `oH x oW`.
                     Can be a tuple `(oH, oW)` or a single number oH for a square image `oH x oH`.
                     Note that we must have :math:`kH + oH - 1 <= H_{in}` and :math:`kW + oW - 1 <= W_{in}`
        output_ratio: If one wants to have an output size as a ratio of the input size, this option can be given.
                      This has to be a number or tuple in the range (0, 1).
                      Note that we must have :math:`kH + (output\_ratio\_H * H_{in}) - 1 <= H_{in}`
                      and :math:`kW + (output\_ratio\_W * W_{in}) - 1 <= W_{in}`
        return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to :meth:`nn.MaxUnpool2d`. Default: ``False``

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where
          :math:`(H_{out}, W_{out})=\text{output\_size}` or
          :math:`(H_{out}, W_{out})=\text{output\_ratio} \times (H_{in}, W_{in})`.

    Examples:
        >>> # pool of square window of size=3, and target output size 13x12
        >>> m = nn.FractionalMaxPool2d(3, output_size=(13, 12))
        >>> # pool of square window and target output size being half of input image size
        >>> m = nn.FractionalMaxPool2d(3, output_ratio=(0.5, 0.5))
        >>> input = torch.randn(20, 16, 50, 32)
        >>> output = m(input)

    .. _Fractional MaxPooling:
        https://arxiv.org/abs/1412.6071
    """

    __constants__ = ["kernel_size", "return_indices", "output_size", "output_ratio"]

    kernel_size: _size_2_t
    return_indices: bool
    output_size: _size_2_t
    output_ratio: _ratio_2_t

    def __init__(
        self,
        kernel_size: _size_2_t,
        output_size: _size_2_t | None = None,
        output_ratio: _ratio_2_t | None = None,
        return_indices: bool = False,
        _random_samples=None,
    ) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.return_indices = return_indices
        self.register_buffer("_random_samples", _random_samples)
        self.output_size = _pair(output_size) if output_size is not None else None
        self.output_ratio = _pair(output_ratio) if output_ratio is not None else None
        if output_size is None and output_ratio is None:
            raise ValueError(
                "FractionalMaxPool2d requires specifying either "
                "an output size, or a pooling ratio"
            )
        if output_size is not None and output_ratio is not None:
            raise ValueError(
                "only one of output_size and output_ratio may be specified"
            )
        if self.output_ratio is not None:
            if not (0 < self.output_ratio[0] < 1 and 0 < self.output_ratio[1] < 1):
                raise ValueError(
                    f"output_ratio must be between 0 and 1 (got {output_ratio})"
                )

    def forward(self, input: Tensor):
        return F.fractional_max_pool2d(
            input,
            self.kernel_size,
            self.output_size,
            self.output_ratio,
            self.return_indices,
            _random_samples=self._random_samples,
        )


class FractionalMaxPool3d(Module):
    r"""Applies a 3D fractional max pooling over an input signal composed of several input planes.

    Fractional MaxPooling is described in detail in the paper `Fractional MaxPooling`_ by Ben Graham

    The max-pooling operation is applied in :math:`kT \times kH \times kW` regions by a stochastic
    step size determined by the target output size.
    The number of output features is equal to the number of input planes.

    .. note:: Exactly one of ``output_size`` or ``output_ratio`` must be defined.

    Args:
        kernel_size: the size of the window to take a max over.
                     Can be a single number `k` (for a square kernel of `k x k x k`) or a tuple `(kt x kh x kw)`,
                     `k` must be greater than 0.
        output_size: the target output size of the image of the form `oT x oH x oW`.
                     Can be a tuple `(oT, oH, oW)` or a single number oH for a square image `oH x oH x oH`
        output_ratio: If one wants to have an output size as a ratio of the input size, this option can be given.
                      This has to be a number or tuple in the range (0, 1)
        return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to :meth:`nn.MaxUnpool3d`. Default: ``False``

    Shape:
        - Input: :math:`(N, C, T_{in}, H_{in}, W_{in})` or :math:`(C, T_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, T_{out}, H_{out}, W_{out})` or :math:`(C, T_{out}, H_{out}, W_{out})`, where
          :math:`(T_{out}, H_{out}, W_{out})=\text{output\_size}` or
          :math:`(T_{out}, H_{out}, W_{out})=\text{output\_ratio} \times (T_{in}, H_{in}, W_{in})`

    Examples:
        >>> # pool of cubic window of size=3, and target output size 13x12x11
        >>> m = nn.FractionalMaxPool3d(3, output_size=(13, 12, 11))
        >>> # pool of cubic window and target output size being half of input size
        >>> m = nn.FractionalMaxPool3d(3, output_ratio=(0.5, 0.5, 0.5))
        >>> input = torch.randn(20, 16, 50, 32, 16)
        >>> output = m(input)

    .. _Fractional MaxPooling:
        https://arxiv.org/abs/1412.6071
    """

    __constants__ = ["kernel_size", "return_indices", "output_size", "output_ratio"]
    kernel_size: _size_3_t
    return_indices: bool
    output_size: _size_3_t
    output_ratio: _ratio_3_t

    def __init__(
        self,
        kernel_size: _size_3_t,
        output_size: _size_3_t | None = None,
        output_ratio: _ratio_3_t | None = None,
        return_indices: bool = False,
        _random_samples=None,
    ) -> None:
        super().__init__()
        if (isinstance(kernel_size, int) and kernel_size <= 0) or (
            isinstance(kernel_size, (tuple, list))
            and not all(k > 0 for k in kernel_size)
        ):
            raise ValueError(
                f"kernel_size must be greater than 0, but got {kernel_size}"
            )
        self.kernel_size = _triple(kernel_size)
        self.return_indices = return_indices
        self.register_buffer("_random_samples", _random_samples)
        self.output_size = _triple(output_size) if output_size is not None else None
        self.output_ratio = _triple(output_ratio) if output_ratio is not None else None
        if output_size is None and output_ratio is None:
            raise ValueError(
                "FractionalMaxPool3d requires specifying either "
                "an output size, or a pooling ratio"
            )
        if output_size is not None and output_ratio is not None:
            raise ValueError(
                "only one of output_size and output_ratio may be specified"
            )
        if self.output_ratio is not None:
            if not (
                0 < self.output_ratio[0] < 1
                and 0 < self.output_ratio[1] < 1
                and 0 < self.output_ratio[2] < 1
            ):
                raise ValueError(
                    f"output_ratio must be between 0 and 1 (got {output_ratio})"
                )

    def forward(self, input: Tensor):
        return F.fractional_max_pool3d(
            input,
            self.kernel_size,
            self.output_size,
            self.output_ratio,
            self.return_indices,
            _random_samples=self._random_samples,
        )


class _LPPoolNd(Module):
    __constants__ = ["norm_type", "kernel_size", "stride", "ceil_mode"]

    norm_type: float
    ceil_mode: bool

    def __init__(
        self,
        norm_type: float,
        kernel_size: _size_any_t,
        stride: _size_any_t | None = None,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        if norm_type == 0:
            raise ValueError(f"norm_type must be a non-zero value, but got {norm_type}")
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.ceil_mode = ceil_mode

    def extra_repr(self) -> str:
        return (
            "norm_type={norm_type}, kernel_size={kernel_size}, stride={stride}, "
            "ceil_mode={ceil_mode}".format(**self.__dict__)
        )


class LPPool1d(_LPPoolNd):
    r"""Applies a 1D power-average pooling over an input signal composed of several input planes.

    On each window, the function computed is:

    .. math::
        f(X) = \sqrt[p]{\sum_{x \in X} x^{p}}

    - At p = :math:`\infty`, one gets Max Pooling over absolute values
    - At p = 1, one gets Sum Pooling (which is proportional to Average Pooling)

    .. note:: If the sum to the power of `p` is zero, the gradient of this function is
              not defined. This implementation will set the gradient to zero in this case.

    Args:
        kernel_size: a single int, the size of the window
        stride: a single int, the stride of the window. Default value is :attr:`kernel_size`
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Note:
        When :attr:`ceil_mode` is ``True``, sliding windows may go off-bounds if they start within the
        left padding or the input. Sliding windows that would start in the right padded region are ignored.

    Shape:
        - Input: :math:`(N, C, L_{in})` or :math:`(C, L_{in})`.
        - Output: :math:`(N, C, L_{out})` or :math:`(C, L_{out})`, where

          .. math::
              L_{out} = \left\lfloor\frac{L_{in} - \text{kernel\_size}}{\text{stride}} + 1\right\rfloor

    Examples::
        >>> # power-2 pool of window of length 3, with stride 2.
        >>> m = nn.LPPool1d(2, 3, stride=2)
        >>> input = torch.randn(20, 16, 50)
        >>> output = m(input)
    """

    kernel_size: _size_1_t
    stride: _size_1_t

    def forward(self, input: Tensor) -> Tensor:
        """Runs the forward pass."""
        return F.lp_pool1d(
            input, float(self.norm_type), self.kernel_size, self.stride, self.ceil_mode
        )


class LPPool2d(_LPPoolNd):
    r"""Applies a 2D power-average pooling over an input signal composed of several input planes.

    On each window, the function computed is:

    .. math::
        f(X) = \sqrt[p]{\sum_{x \in X} x^{p}}

    - At p = :math:`\infty`, one gets Max Pooling over absolute values
    - At p = 1, one gets Sum Pooling (which is proportional to average pooling)

    The parameters :attr:`kernel_size`, :attr:`stride` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    .. note:: If the sum to the power of `p` is zero, the gradient of this function is
              not defined. This implementation will set the gradient to zero in this case.

    Args:
        kernel_size: the size of the window
        stride: the stride of the window. Default value is :attr:`kernel_size`
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Note:
        When :attr:`ceil_mode` is ``True``, sliding windows may go off-bounds if they start within the
        left padding or the input. Sliding windows that would start in the right padded region are ignored.

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} - \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} - \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor

    Examples::

        >>> # power-2 pool of square window of size=3, stride=2
        >>> m = nn.LPPool2d(2, 3, stride=2)
        >>> # pool of non-square window of power 1.2
        >>> m = nn.LPPool2d(1.2, (3, 2), stride=(2, 1))
        >>> input = torch.randn(20, 16, 50, 32)
        >>> output = m(input)

    """

    kernel_size: _size_2_t
    stride: _size_2_t

    def forward(self, input: Tensor) -> Tensor:
        """Runs the forward pass."""
        return F.lp_pool2d(
            input, float(self.norm_type), self.kernel_size, self.stride, self.ceil_mode
        )


class LPPool3d(_LPPoolNd):
    r"""Applies a 3D power-average pooling over an input signal composed of several input planes.

    On each window, the function computed is:

    .. math::
        f(X) = \sqrt[p]{\sum_{x \in X} x^{p}}

    - At p = :math:`\infty`, one gets Max Pooling over absolute values
    - At p = 1, one gets Sum Pooling (which is proportional to average pooling)

    The parameters :attr:`kernel_size`, :attr:`stride` can either be:

        - a single ``int`` -- in which case the same value is used for the height, width and depth dimension
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    .. note:: If the sum to the power of `p` is zero, the gradient of this function is
              not defined. This implementation will set the gradient to zero in this case.

    Args:
        kernel_size: the size of the window
        stride: the stride of the window. Default value is :attr:`kernel_size`
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Note:
        When :attr:`ceil_mode` is ``True``, sliding windows may go off-bounds if they start within the
        left padding or the input. Sliding windows that would start in the right padded region are ignored.

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or
          :math:`(C, D_{out}, H_{out}, W_{out})`, where

          .. math::
              D_{out} = \left\lfloor\frac{D_{in} - \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} - \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} - \text{kernel\_size}[2]}{\text{stride}[2]} + 1\right\rfloor

    Examples::

        >>> # power-2 pool of square window of size=3, stride=2
        >>> m = nn.LPPool3d(2, 3, stride=2)
        >>> # pool of non-square window of power 1.2
        >>> m = nn.LPPool3d(1.2, (3, 2, 2), stride=(2, 1, 2))
        >>> input = torch.randn(20, 16, 50, 44, 31)
        >>> output = m(input)

    """

    kernel_size: _size_3_t
    stride: _size_3_t

    def forward(self, input: Tensor) -> Tensor:
        """Runs the forward pass."""
        return F.lp_pool3d(
            input, float(self.norm_type), self.kernel_size, self.stride, self.ceil_mode
        )


class _AdaptiveMaxPoolNd(Module):
    __constants__ = ["output_size", "return_indices"]
    return_indices: bool

    def __init__(
        self, output_size: _size_any_opt_t, return_indices: bool = False
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.return_indices = return_indices

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"


# FIXME (by @ssnl): Improve adaptive pooling docs: specify what the input and
#   output shapes are, and how the operation computes output.


class AdaptiveMaxPool1d(_AdaptiveMaxPoolNd):
    r"""Applies a 1D adaptive max pooling over an input signal composed of several input planes.

    The output size is :math:`L_{out}`, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size :math:`L_{out}`.
        return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to nn.MaxUnpool1d. Default: ``False``

    Shape:
        - Input: :math:`(N, C, L_{in})` or :math:`(C, L_{in})`.
        - Output: :math:`(N, C, L_{out})` or :math:`(C, L_{out})`, where
          :math:`L_{out}=\text{output\_size}`.

    Examples:
        >>> # target output size of 5
        >>> m = nn.AdaptiveMaxPool1d(5)
        >>> input = torch.randn(1, 64, 8)
        >>> output = m(input)

    """

    output_size: _size_1_t

    def forward(self, input: Tensor):
        """Runs the forward pass."""
        return F.adaptive_max_pool1d(input, self.output_size, self.return_indices)


class AdaptiveMaxPool2d(_AdaptiveMaxPoolNd):
    r"""Applies a 2D adaptive max pooling over an input signal composed of several input planes.

    The output is of size :math:`H_{out} \times W_{out}`, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the image of the form :math:`H_{out} \times W_{out}`.
                     Can be a tuple :math:`(H_{out}, W_{out})` or a single :math:`H_{out}` for a
                     square image :math:`H_{out} \times H_{out}`. :math:`H_{out}` and :math:`W_{out}`
                     can be either a ``int``, or ``None`` which means the size will be the same as that
                     of the input.
        return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to nn.MaxUnpool2d. Default: ``False``

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where
          :math:`(H_{out}, W_{out})=\text{output\_size}`.

    Examples:
        >>> # target output size of 5x7
        >>> m = nn.AdaptiveMaxPool2d((5, 7))
        >>> input = torch.randn(1, 64, 8, 9)
        >>> output = m(input)
        >>> # target output size of 7x7 (square)
        >>> m = nn.AdaptiveMaxPool2d(7)
        >>> input = torch.randn(1, 64, 10, 9)
        >>> output = m(input)
        >>> # target output size of 10x7
        >>> m = nn.AdaptiveMaxPool2d((None, 7))
        >>> input = torch.randn(1, 64, 10, 9)
        >>> output = m(input)

    """

    output_size: _size_2_opt_t

    def forward(self, input: Tensor):
        """Runs the forward pass."""
        return F.adaptive_max_pool2d(input, self.output_size, self.return_indices)


class AdaptiveMaxPool3d(_AdaptiveMaxPoolNd):
    r"""Applies a 3D adaptive max pooling over an input signal composed of several input planes.

    The output is of size :math:`D_{out} \times H_{out} \times W_{out}`, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the image of the form :math:`D_{out} \times H_{out} \times W_{out}`.
                     Can be a tuple :math:`(D_{out}, H_{out}, W_{out})` or a single
                     :math:`D_{out}` for a cube :math:`D_{out} \times D_{out} \times D_{out}`.
                     :math:`D_{out}`, :math:`H_{out}` and :math:`W_{out}` can be either a
                     ``int``, or ``None`` which means the size will be the same as that of the input.

        return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to nn.MaxUnpool3d. Default: ``False``

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or :math:`(C, D_{out}, H_{out}, W_{out})`,
          where :math:`(D_{out}, H_{out}, W_{out})=\text{output\_size}`.

    Examples:
        >>> # target output size of 5x7x9
        >>> m = nn.AdaptiveMaxPool3d((5, 7, 9))
        >>> input = torch.randn(1, 64, 8, 9, 10)
        >>> output = m(input)
        >>> # target output size of 7x7x7 (cube)
        >>> m = nn.AdaptiveMaxPool3d(7)
        >>> input = torch.randn(1, 64, 10, 9, 8)
        >>> output = m(input)
        >>> # target output size of 7x9x8
        >>> m = nn.AdaptiveMaxPool3d((7, None, None))
        >>> input = torch.randn(1, 64, 10, 9, 8)
        >>> output = m(input)

    """

    output_size: _size_3_opt_t

    def forward(self, input: Tensor):
        """Runs the forward pass."""
        return F.adaptive_max_pool3d(input, self.output_size, self.return_indices)


class _AdaptiveAvgPoolNd(Module):
    __constants__ = ["output_size"]

    def __init__(self, output_size: _size_any_opt_t) -> None:
        super().__init__()
        self.output_size = output_size

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"


class AdaptiveAvgPool1d(_AdaptiveAvgPoolNd):
    r"""Applies a 1D adaptive average pooling over an input signal composed of several input planes.

    The output size is :math:`L_{out}`, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size :math:`L_{out}`.

    Shape:
        - Input: :math:`(N, C, L_{in})` or :math:`(C, L_{in})`.
        - Output: :math:`(N, C, L_{out})` or :math:`(C, L_{out})`, where
          :math:`L_{out}=\text{output\_size}`.

    Examples:
        >>> # target output size of 5
        >>> m = nn.AdaptiveAvgPool1d(5)
        >>> input = torch.randn(1, 64, 8)
        >>> output = m(input)

    """

    output_size: _size_1_t

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return F.adaptive_avg_pool1d(input, self.output_size)


class AdaptiveAvgPool2d(_AdaptiveAvgPoolNd):
    r"""Applies a 2D adaptive average pooling over an input signal composed of several input planes.

    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H.
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, S_{0}, S_{1})` or :math:`(C, S_{0}, S_{1})`, where
          :math:`S=\text{output\_size}`.

    Examples:
        >>> # target output size of 5x7
        >>> m = nn.AdaptiveAvgPool2d((5, 7))
        >>> input = torch.randn(1, 64, 8, 9)
        >>> output = m(input)
        >>> # target output size of 7x7 (square)
        >>> m = nn.AdaptiveAvgPool2d(7)
        >>> input = torch.randn(1, 64, 10, 9)
        >>> output = m(input)
        >>> # target output size of 10x7
        >>> m = nn.AdaptiveAvgPool2d((None, 7))
        >>> input = torch.randn(1, 64, 10, 9)
        >>> output = m(input)

    """

    output_size: _size_2_opt_t

    def forward(self, input: Tensor) -> Tensor:
        """Runs the forward pass."""
        return F.adaptive_avg_pool2d(input, self.output_size)


class AdaptiveAvgPool3d(_AdaptiveAvgPoolNd):
    r"""Applies a 3D adaptive average pooling over an input signal composed of several input planes.

    The output is of size D x H x W, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the form D x H x W.
                     Can be a tuple (D, H, W) or a single number D for a cube D x D x D.
                     D, H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, S_{0}, S_{1}, S_{2})` or :math:`(C, S_{0}, S_{1}, S_{2})`,
          where :math:`S=\text{output\_size}`.

    Examples:
        >>> # target output size of 5x7x9
        >>> m = nn.AdaptiveAvgPool3d((5, 7, 9))
        >>> input = torch.randn(1, 64, 8, 9, 10)
        >>> output = m(input)
        >>> # target output size of 7x7x7 (cube)
        >>> m = nn.AdaptiveAvgPool3d(7)
        >>> input = torch.randn(1, 64, 10, 9, 8)
        >>> output = m(input)
        >>> # target output size of 7x9x8
        >>> m = nn.AdaptiveAvgPool3d((7, None, None))
        >>> input = torch.randn(1, 64, 10, 9, 8)
        >>> output = m(input)

    """

    output_size: _size_3_opt_t

    def forward(self, input: Tensor) -> Tensor:
        """Runs the forward pass."""
        return F.adaptive_avg_pool3d(input, self.output_size)
<|endoftext|># mypy: allow-untyped-defs
import math
from collections.abc import Callable
from typing_extensions import deprecated

from torch import Tensor
from torch.nn import _reduction as _Reduction, functional as F

from .distance import PairwiseDistance
from .linear import Linear
from .linear_cross_entropy_options import LinearCrossEntropyOptions
from .module import Module


__all__ = [
    "L1Loss",
    "NLLLoss",
    "NLLLoss2d",
    "PoissonNLLLoss",
    "GaussianNLLLoss",
    "KLDivLoss",
    "MSELoss",
    "BCELoss",
    "BCEWithLogitsLoss",
    "HingeEmbeddingLoss",
    "MultiLabelMarginLoss",
    "SmoothL1Loss",
    "HuberLoss",
    "SoftMarginLoss",
    "CrossEntropyLoss",
    "LinearCrossEntropyLoss",
    "MultiLabelSoftMarginLoss",
    "CosineEmbeddingLoss",
    "MarginRankingLoss",
    "MultiMarginLoss",
    "TripletMarginLoss",
    "TripletMarginWithDistanceLoss",
    "CTCLoss",
]


class _Loss(Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class _WeightedLoss(_Loss):
    def __init__(
        self,
        weight: Tensor | None = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.register_buffer("weight", weight)
        self.weight: Tensor | None


class L1Loss(_Loss):
    r"""Creates a criterion that measures the mean absolute error (MAE) between each element in
    the input :math:`x` and target :math:`y`.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left| x_n - y_n \right|,

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`N` elements each.

    The sum operation still operates over all the elements, and divides by :math:`N`.

    The division by :math:`N` can be avoided if one sets ``reduction = 'sum'``.

    Supports real-valued and complex-valued inputs.

    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then
          :math:`(*)`, same shape as the input.

    Examples:

        >>> loss = nn.L1Loss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    __constants__ = ["reduction"]

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return F.l1_loss(input, target, reduction=self.reduction)


class NLLLoss(_WeightedLoss):
    r"""The negative log likelihood loss. It is useful to train a classification
    problem with `C` classes.

    If provided, the optional argument :attr:`weight` should be a 1D Tensor assigning
    weight to each of the classes. This is particularly useful when you have an
    unbalanced training set.

    The `input` given through a forward call is expected to contain
    log-probabilities of each class. `input` has to be a Tensor of size either
    :math:`(minibatch, C)` or :math:`(minibatch, C, d_1, d_2, ..., d_K)`
    with :math:`K \geq 1` for the `K`-dimensional case. The latter is useful for
    higher dimension inputs, such as computing NLL loss per-pixel for 2D images.

    Obtaining log-probabilities in a neural network is easily achieved by
    adding a  `LogSoftmax`  layer in the last layer of your network.
    You may use `CrossEntropyLoss` instead, if you prefer not to add an extra
    layer.

    The `target` that this loss expects should be a class index in the range :math:`[0, C-1]`
    where `C = number of classes`; if `ignore_index` is specified, this loss also accepts
    this class index (this index may not necessarily be in the class range).

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \\
        l_n = - w_{y_n} x_{n,y_n}, \\
        w_{c} = \text{weight}[c] \cdot \mathbb{1}\{c \not= \text{ignore\_index}\},

    where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight, and
    :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then

    .. math::
        \ell(x, y) = \begin{cases}
            \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n}} l_n, &
            \text{if reduction} = \text{`mean';}\\
            \sum_{n=1}^N l_n,  &
            \text{if reduction} = \text{`sum'.}
        \end{cases}

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`. Otherwise, it is
            treated as if having all ones.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``None``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When
            :attr:`size_average` is ``True``, the loss is averaged over
            non-ignored targets.
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``None``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in
            the meantime, specifying either of those two args will override
            :attr:`reduction`. Default: ``'mean'``

    Shape::
        - Input: :math:`(N, C)` or :math:`(C)`, where `C = number of classes`, `N = batch size`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: :math:`(N)` or :math:`()`, where each value is
          :math:`0 \leq \text{targets}[i] \leq C-1`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of
          K-dimensional loss.
        - Output: If :attr:`reduction` is ``'none'``, shape :math:`(N)` or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of K-dimensional loss.
          Otherwise, scalar.

    Examples:

        >>> log_softmax = nn.LogSoftmax(dim=1)
        >>> loss_fn = nn.NLLLoss()
        >>> # input to NLLLoss is of size N x C = 3 x 5
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> # each element in target must have 0 <= value < C
        >>> target = torch.tensor([1, 0, 4])
        >>> loss = loss_fn(log_softmax(input), target)
        >>> loss.backward()
        >>>
        >>>
        >>> # 2D loss example (used, for example, with image inputs)
        >>> N, C = 5, 4
        >>> loss_fn = nn.NLLLoss()
        >>> data = torch.randn(N, 16, 10, 10)
        >>> conv = nn.Conv2d(16, C, (3, 3))
        >>> log_softmax = nn.LogSoftmax(dim=1)
        >>> # output of conv forward is of shape [N, C, 8, 8]
        >>> output = log_softmax(conv(data))
        >>> # each element in target must have 0 <= value < C
        >>> target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
        >>> # input to NLLLoss is of size N x C x height (8) x width (8)
        >>> loss = loss_fn(output, target)
        >>> loss.backward()
    """

    __constants__ = ["ignore_index", "reduction"]
    ignore_index: int

    def __init__(
        self,
        weight: Tensor | None = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return F.nll_loss(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )


@deprecated(
    "`NLLLoss2d` has been deprecated. "
    "Please use `NLLLoss` instead as a drop-in replacement and see "
    "https://pytorch.org/docs/main/nn.html#torch.nn.NLLLoss for more details.",
    category=FutureWarning,
)
class NLLLoss2d(NLLLoss):
    def __init__(
        self,
        weight: Tensor | None = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(weight, size_average, ignore_index, reduce, reduction)


class PoissonNLLLoss(_Loss):
    r"""Negative log likelihood loss with Poisson distribution of target.

    The loss can be described as:

    .. math::
        \text{target} \sim \mathrm{Poisson}(\text{input})

        \text{loss}(\text{input}, \text{target}) = \text{input} - \text{target} * \log(\text{input})
                                    + \log(\text{target!})

    The last term can be omitted or approximated with Stirling formula. The
    approximation is used for target values more than 1. For targets less or
    equal to 1 zeros are added to the loss.

    Args:
        log_input (bool, optional): if ``True`` the loss is computed as
            :math:`\exp(\text{input}) - \text{target}*\text{input}`, if ``False`` the loss is
            :math:`\text{input} - \text{target}*\log(\text{input}+\text{eps})`.
        full (bool, optional): whether to compute full loss, i. e. to add the
            Stirling approximation term

            .. math::
                \text{target}*\log(\text{target}) - \text{target} + 0.5 * \log(2\pi\text{target}).
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        eps (float, optional): Small value to avoid evaluation of :math:`\log(0)` when
            :attr:`log_input = False`. Default: 1e-8
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Examples:

        >>> loss = nn.PoissonNLLLoss()
        >>> log_input = torch.randn(5, 2, requires_grad=True)
        >>> target = torch.randn(5, 2)
        >>> output = loss(log_input, target)
        >>> output.backward()

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.
        - Output: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(*)`,
          the same shape as the input.
    """

    __constants__ = ["log_input", "full", "eps", "reduction"]
    log_input: bool
    full: bool
    eps: float

    def __init__(
        self,
        log_input: bool = True,
        full: bool = False,
        size_average=None,
        eps: float = 1e-8,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.log_input = log_input
        self.full = full
        self.eps = eps

    def forward(self, log_input: Tensor, target: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return F.poisson_nll_loss(
            log_input,
            target,
            log_input=self.log_input,
            full=self.full,
            eps=self.eps,
            reduction=self.reduction,
        )


class GaussianNLLLoss(_Loss):
    r"""Gaussian negative log likelihood loss.

    The targets are treated as samples from Gaussian distributions with
    expectations and variances predicted by the neural network. For a
    ``target`` tensor modelled as having Gaussian distribution with a tensor
    of expectations ``input`` and a tensor of positive variances ``var`` the loss is:

    .. math::
        \text{loss} = \frac{1}{2}\left(\log\left(\text{max}\left(\text{var},
        \ \text{eps}\right)\right) + \frac{\left(\text{input} - \text{target}\right)^2}
        {\text{max}\left(\text{var}, \ \text{eps}\right)}\right) + \text{const.}

    where :attr:`eps` is used for stability. By default, the constant term of
    the loss function is omitted unless :attr:`full` is ``True``. If ``var`` is not the same
    size as ``input`` (due to a homoscedastic assumption), it must either have a final dimension
    of 1 or have one fewer dimension (with all other sizes being the same) for correct broadcasting.

    Args:
        full (bool, optional): include the constant term in the loss
            calculation. Default: ``False``.
        eps (float, optional): value used to clamp ``var`` (see note below), for
            stability. Default: 1e-6.
        reduction (str, optional): specifies the reduction to apply to the
            output:``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
            will be applied, ``'mean'``: the output is the average of all batch
            member losses, ``'sum'``: the output is the sum of all batch member
            losses. Default: ``'mean'``.

    Shape:
        - Input: :math:`(N, *)` or :math:`(*)` where :math:`*` means any number of additional
          dimensions
        - Target: :math:`(N, *)` or :math:`(*)`, same shape as the input, or same shape as the input
          but with one dimension equal to 1 (to allow for broadcasting)
        - Var: :math:`(N, *)` or :math:`(*)`, same shape as the input, or same shape as the input but
          with one dimension equal to 1, or same shape as the input but with one fewer
          dimension (to allow for broadcasting), or a scalar value
        - Output: scalar if :attr:`reduction` is ``'mean'`` (default) or
          ``'sum'``. If :attr:`reduction` is ``'none'``, then :math:`(N, *)`, same
          shape as the input

    Examples:
        >>> loss = nn.GaussianNLLLoss()
        >>> input = torch.randn(5, 2, requires_grad=True)
        >>> target = torch.randn(5, 2)
        >>> var = torch.ones(5, 2, requires_grad=True)  # heteroscedastic
        >>> output = loss(input, target, var)
        >>> output.backward()

        >>> loss = nn.GaussianNLLLoss()
        >>> input = torch.randn(5, 2, requires_grad=True)
        >>> target = torch.randn(5, 2)
        >>> var = torch.ones(5, 1, requires_grad=True)  # homoscedastic
        >>> output = loss(input, target, var)
        >>> output.backward()

    Note:
        The clamping of ``var`` is ignored with respect to autograd, and so the
        gradients are unaffected by it.

    Reference:
        Nix, D. A. and Weigend, A. S., "Estimating the mean and variance of the
        target probability distribution", Proceedings of 1994 IEEE International
        Conference on Neural Networks (ICNN'94), Orlando, FL, USA, 1994, pp. 55-60
        vol.1, doi: 10.1109/ICNN.1994.374138.
    """

    __constants__ = ["full", "eps", "reduction"]
    full: bool
    eps: float

    def __init__(
        self, *, full: bool = False, eps: float = 1e-6, reduction: str = "mean"
    ) -> None:
        super().__init__(None, None, reduction)
        self.full = full
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor, var: Tensor | float) -> Tensor:
        """
        Runs the forward pass.
        """
        return F.gaussian_nll_loss(
            input, target, var, full=self.full, eps=self.eps, reduction=self.reduction
        )


class KLDivLoss(_Loss):
    r"""The Kullback-Leibler divergence loss.

    For tensors of the same shape :math:`y_{\text{pred}},\ y_{\text{true}}`,
    where :math:`y_{\text{pred}}` is the :attr:`input` and :math:`y_{\text{true}}` is the
    :attr:`target`, we define the **pointwise KL-divergence** as

    .. math::

        L(y_{\text{pred}},\ y_{\text{true}})
            = y_{\text{true}} \cdot \log \frac{y_{\text{true}}}{y_{\text{pred}}}
            = y_{\text{true}} \cdot (\log y_{\text{true}} - \log y_{\text{pred}})

    To avoid underflow issues when computing this quantity, this loss expects the argument
    :attr:`input` in the log-space. The argument :attr:`target` may also be provided in the
    log-space if :attr:`log_target`\ `= True`.

    To summarise, this function is roughly equivalent to computing

    .. code-block:: python

        if not log_target:  # default
            loss_pointwise = target * (target.log() - input)
        else:
            loss_pointwise = target.exp() * (target - input)

    and then reducing this result depending on the argument :attr:`reduction` as

    .. code-block:: python

        if reduction == "mean":  # default
            loss = loss_pointwise.mean()
        elif reduction == "batchmean":  # mathematically correct
            loss = loss_pointwise.sum() / input.size(0)
        elif reduction == "sum":
            loss = loss_pointwise.sum()
        else:  # reduction == "none"
            loss = loss_pointwise

    .. note::
        As all the other losses in PyTorch, this function expects the first argument,
        :attr:`input`, to be the output of the model (e.g. the neural network)
        and the second, :attr:`target`, to be the observations in the dataset.
        This differs from the standard mathematical notation :math:`KL(P\ ||\ Q)` where
        :math:`P` denotes the distribution of the observations and :math:`Q` denotes the model.

    .. warning::
        :attr:`reduction`\ `= "mean"` doesn't return the true KL divergence value, please use
        :attr:`reduction`\ `= "batchmean"` which aligns with the mathematical definition.

    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to `False`, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is `False`. Default: `True`
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is `False`, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: `True`
        reduction (str, optional): Specifies the reduction to apply to the output. Default: `"mean"`
        log_target (bool, optional): Specifies whether `target` is the log space. Default: `False`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.
        - Output: scalar by default. If :attr:`reduction` is `'none'`, then :math:`(*)`,
          same shape as the input.

    Examples:
        >>> kl_loss = nn.KLDivLoss(reduction="batchmean")
        >>> # input should be a distribution in the log space
        >>> input = F.log_softmax(torch.randn(3, 5, requires_grad=True), dim=1)
        >>> # Sample a batch of distributions. Usually this would come from the dataset
        >>> target = F.softmax(torch.rand(3, 5), dim=1)
        >>> output = kl_loss(input, target)
        >>>
        >>> kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        >>> log_target = F.log_softmax(torch.rand(3, 5), dim=1)
        >>> output = kl_loss(input, log_target)
    """

    __constants__ = ["reduction"]

    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        log_target: bool = False,
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.log_target = log_target

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return F.kl_div(
            input, target, reduction=self.reduction, log_target=self.log_target
        )


class MSELoss(_Loss):
    r"""Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input :math:`x` and target :math:`y`.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), &  \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  &  \text{if reduction} = \text{`sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`N` elements each.

    The mean operation still operates over all the elements, and divides by :math:`N`.

    The division by :math:`N` can be avoided if one sets ``reduction = 'sum'``.

    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.

    Examples:

        >>> loss = nn.MSELoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    __constants__ = ["reduction"]

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return F.mse_loss(input, target, reduction=self.reduction)


class BCELoss(_WeightedLoss):
    r"""Creates a criterion that measures the Binary Cross Entropy between the target and
    the input probabilities:

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right],

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
        \end{cases}

    This is used for measuring the error of a reconstruction in for example
    an auto-encoder. Note that the targets :math:`y` should be numbers
    between 0 and 1.

    Notice that if :math:`x_n` is either 0 or 1, one of the log terms would be
    mathematically undefined in the above loss equation. PyTorch chooses to set
    :math:`\log (0) = -\infty`, since :math:`\lim_{x\to 0} \log (x) = -\infty`.
    However, an infinite term in the loss equation is not desirable for several reasons.

    For one, if either :math:`y_n = 0` or :math:`(1 - y_n) = 0`, then we would be
    multiplying 0 with infinity. Secondly, if we have an infinite loss value, then
    we would also have an infinite term in our gradient, since
    :math:`\lim_{x\to 0} \frac{d}{dx} \log (x) = \infty`.
    This would make BCELoss's backward method nonlinear with respect to :math:`x_n`,
    and using it for things like linear regression would not be straight-forward.

    Our solution is that BCELoss clamps its log function outputs to be greater than
    or equal to -100. This way, we can always have a finite loss value and a linear
    backward method.


    Args:
        weight (Tensor, optional): a manual rescaling weight given to the loss
            of each batch element. If given, has to be a Tensor of size `nbatch`.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(*)`, same
          shape as input.

    Examples:

        >>> m = nn.Sigmoid()
        >>> loss = nn.BCELoss()
        >>> input = torch.randn(3, 2, requires_grad=True)
        >>> target = torch.rand(3, 2, requires_grad=False)
        >>> output = loss(m(input), target)
        >>> output.backward()
    """

    __constants__ = ["reduction"]

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return F.binary_cross_entropy(
            input, target, weight=self.weight, reduction=self.reduction
        )


class BCEWithLogitsLoss(_Loss):
    r"""This loss combines a `Sigmoid` layer and the `BCELoss` in one single
    class. This version is more numerically stable than using a plain `Sigmoid`
    followed by a `BCELoss` as, by combining the operations into one layer,
    we take advantage of the log-sum-exp trick for numerical stability.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log \sigma(x_n)
        + (1 - y_n) \cdot \log (1 - \sigma(x_n)) \right],

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
        \end{cases}

    This is used for measuring the error of a reconstruction in for example
    an auto-encoder. Note that the targets `t[i]` should be numbers
    between 0 and 1.

    It's possible to trade off recall and precision by adding weights to positive examples.
    In the case of multi-label classification the loss can be described as:

    .. math::
        \ell_c(x, y) = L_c = \{l_{1,c},\dots,l_{N,c}\}^\top, \quad
        l_{n,c} = - w_{n,c} \left[ p_c y_{n,c} \cdot \log \sigma(x_{n,c})
        + (1 - y_{n,c}) \cdot \log (1 - \sigma(x_{n,c})) \right],

    where :math:`c` is the class number (:math:`c > 1` for multi-label binary classification,
    :math:`c = 1` for single-label binary classification),
    :math:`n` is the number of the sample in the batch and
    :math:`p_c` is the weight of the positive answer for the class :math:`c`.

    :math:`p_c > 1` increases the recall, :math:`p_c < 1` increases the precision.

    For example, if a dataset contains 100 positive and 300 negative examples of a single class,
    then ``pos_weight`` for the class should be equal to :math:`\frac{300}{100}=3`.
    The loss would act as if the dataset contains :math:`3\times 100=300` positive examples.

    Examples:

        >>> target = torch.ones([10, 64], dtype=torch.float32)  # 64 classes, batch size = 10
        >>> output = torch.full([10, 64], 1.5)  # A prediction (logit)
        >>> pos_weight = torch.ones([64])  # All weights are equal to 1
        >>> criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        >>> criterion(output, target)  # -log(sigmoid(1.5))
        tensor(0.20...)

    In the above example, the ``pos_weight`` tensor's elements correspond to the 64 distinct classes
    in a multi-label binary classification scenario. Each element in ``pos_weight`` is designed to adjust the
    loss function based on the imbalance between negative and positive samples for the respective class.
    This approach is useful in datasets with varying levels of class imbalance, ensuring that the loss
    calculation accurately accounts for the distribution in each class.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to the loss
            of each batch element. The dimension of weight supports :ref:`broadcasting to a common shape <broadcasting-semantics>`
            with respect to the output (and target) shape.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        pos_weight (Tensor, optional): a weight of positive examples to be broadcasted with target.
            Must be a tensor with equal size along the class dimension to the number of classes.
            Pay close attention to PyTorch's broadcasting semantics in order to achieve the desired
            operations. For a target of size [B, C, H, W] (where B is batch size) pos_weight of
            size [B, C, H, W] will apply different pos_weights to each element of the batch or
            [C, H, W] the same pos_weights across the batch. To apply the same positive weight
            along all spatial dimensions for a 2D multi-class target [C, H, W] use: [C, 1, 1].
            Default: ``None``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(*)`, same
          shape as input.

    Examples:

        >>> loss = nn.BCEWithLogitsLoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.empty(3).random_(2)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(
        self,
        weight: Tensor | None = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        pos_weight: Tensor | None = None,
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)
        self.weight: Tensor | None
        self.pos_weight: Tensor | None

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Runs the forward pass."""
        return F.binary_cross_entropy_with_logits(
            input,
            target,
            self.weight,
            pos_weight=self.pos_weight,
            reduction=self.reduction,
        )


class HingeEmbeddingLoss(_Loss):
    r"""Measures the loss given an input tensor :math:`x` and a labels tensor :math:`y`
    (containing 1 or -1).
    This is usually used for measuring whether two inputs are similar or
    dissimilar, e.g. using the L1 pairwise distance as :math:`x`, and is typically
    used for learning nonlinear embeddings or semi-supervised learning.

    The loss function for :math:`n`-th sample in the mini-batch is

    .. math::
        l_n = \begin{cases}
            x_n, & \text{if}\; y_n = 1,\\
            \max \{0, margin - x_n\}, & \text{if}\; y_n = -1,
        \end{cases}

    and the total loss functions is

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
        \end{cases}

    where :math:`L = \{l_1,\dots,l_N\}^\top`.

    Args:
        margin (float, optional): Has a default value of `1`.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(*)` where :math:`*` means, any number of dimensions. The sum operation
          operates over all the elements.
        - Target: :math:`(*)`, same shape as the input
        - Output: scalar. If :attr:`reduction` is ``'none'``, then same shape as the input

    Examples:

        >>> loss = nn.HingeEmbeddingLoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5).sign()
        >>> output = loss(input, target)
        >>> output.backward()
    """

    __constants__ = ["margin", "reduction"]
    margin: float

    def __init__(
        self,
        margin: float = 1.0,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.margin = margin

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Runs the forward pass."""
        return F.hinge_embedding_loss(
            input, target, margin=self.margin, reduction=self.reduction
        )


class MultiLabelMarginLoss(_Loss):
    r"""Creates a criterion that optimizes a multi-class multi-classification
    hinge loss (margin-based loss) between input :math:`x` (a 2D mini-batch `Tensor`)
    and output :math:`y` (which is a 2D `Tensor` of target class indices).
    For each sample in the mini-batch:

    .. math::
        \text{loss}(x, y) = \sum_{ij}\frac{\max(0, 1 - (x[y[j]] - x[i]))}{\text{x.size}(0)}

    where :math:`x \in \left\{0, \; \cdots , \; \text{x.size}(0) - 1\right\}`, \
    :math:`y \in \left\{0, \; \cdots , \; \text{y.size}(0) - 1\right\}`, \
    :math:`0 \leq y[j] \leq \text{x.size}(0)-1`, \
    and :math:`i \neq y[j]` for all :math:`i` and :math:`j`.

    :math:`y` and :math:`x` must have the same size.

    The criterion only considers a contiguous block of non-negative targets that
    starts at the front.

    This allows for different samples to have variable amounts of target classes.

    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(C)` or :math:`(N, C)` where `N` is the batch size and `C`
          is the number of classes.
        - Target: :math:`(C)` or :math:`(N, C)`, label targets padded by -1 ensuring same shape as the input.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(N)`.

    Examples:

        >>> loss = nn.MultiLabelMarginLoss()
        >>> x = torch.FloatTensor([[0.1, 0.2, 0.4, 0.8]])
        >>> # for target y, only consider labels 3 and 0, not after label -1
        >>> y = torch.LongTensor([[3, 0, -1, 1]])
        >>> # 0.25 * ((1-(0.1-0.2)) + (1-(0.1-0.4)) + (1-(0.8-0.2)) + (1-(0.8-0.4)))
        >>> loss(x, y)
        tensor(0.85...)

    """

    __constants__ = ["reduction"]

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Runs the forward pass."""
        return F.multilabel_margin_loss(input, target, reduction=self.reduction)


class SmoothL1Loss(_Loss):
    r"""Creates a criterion that uses a squared term if the absolute
    element-wise error falls below beta and an L1 term otherwise.
    It is less sensitive to outliers than :class:`torch.nn.MSELoss` and in some cases
    prevents exploding gradients (e.g. see the paper `Fast R-CNN`_ by Ross Girshick).

    For a batch of size :math:`N`, the unreduced loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1, ..., l_N\}^T

    with

    .. math::
        l_n = \begin{cases}
        0.5 (x_n - y_n)^2 / beta, & \text{if } |x_n - y_n| < beta \\
        |x_n - y_n| - 0.5 * beta, & \text{otherwise }
        \end{cases}

    If `reduction` is not `none`, then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), &  \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  &  \text{if reduction} = \text{`sum'.}
        \end{cases}

    .. note::
        Smooth L1 loss can be seen as exactly :class:`L1Loss`, but with the :math:`|x - y| < beta`
        portion replaced with a quadratic function such that its slope is 1 at :math:`|x - y| = beta`.
        The quadratic segment smooths the L1 loss near :math:`|x - y| = 0`.

    .. note::
        Smooth L1 loss is closely related to :class:`HuberLoss`, being
        equivalent to :math:`huber(x, y) / beta` (note that Smooth L1's beta hyper-parameter is
        also known as delta for Huber). This leads to the following differences:

        * As beta -> 0, Smooth L1 loss converges to :class:`L1Loss`, while :class:`HuberLoss`
          converges to a constant 0 loss. When beta is 0, Smooth L1 loss is equivalent to L1 loss.
        * As beta -> :math:`+\infty`, Smooth L1 loss converges to a constant 0 loss, while
          :class:`HuberLoss` converges to :class:`MSELoss`.
        * For Smooth L1 loss, as beta varies, the L1 segment of the loss has a constant slope of 1.
          For :class:`HuberLoss`, the slope of the L1 segment is beta.

    .. _`Fast R-CNN`: https://arxiv.org/abs/1504.08083

    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        beta (float, optional): Specifies the threshold at which to change between L1 and L2 loss.
            The value must be non-negative. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(*)`, same shape as the input.

    Examples:

        >>> loss = nn.SmoothL1Loss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    __constants__ = ["reduction"]

    def __init__(
        self, size_average=None, reduce=None, reduction: str = "mean", beta: float = 1.0
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.beta = beta

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Runs the forward pass."""
        return F.smooth_l1_loss(input, target, reduction=self.reduction, beta=self.beta)


class HuberLoss(_Loss):
    r"""Creates a criterion that uses a squared term if the absolute
    element-wise error falls below delta and a delta-scaled L1 term otherwise.
    This loss combines advantages of both :class:`L1Loss` and :class:`MSELoss`; the
    delta-scaled L1 region makes the loss less sensitive to outliers than :class:`MSELoss`,
    while the L2 region provides smoothness over :class:`L1Loss` near 0. See
    `Huber loss <https://en.wikipedia.org/wiki/Huber_loss>`_ for more information.

    For a batch of size :math:`N`, the unreduced loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1, ..., l_N\}^T

    with

    .. math::
        l_n = \begin{cases}
        0.5 (x_n - y_n)^2, & \text{if } |x_n - y_n| < delta \\
        delta * (|x_n - y_n| - 0.5 * delta), & \text{otherwise }
        \end{cases}

    If `reduction` is not `none`, then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), &  \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  &  \text{if reduction} = \text{`sum'.}
        \end{cases}

    .. note::
        When delta is set to 1, this loss is equivalent to :class:`SmoothL1Loss`.
        In general, this loss differs from :class:`SmoothL1Loss` by a factor of delta (AKA beta
        in Smooth L1).
        See :class:`SmoothL1Loss` for additional discussion on the differences in behavior
        between the two losses.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
        delta (float, optional): Specifies the threshold at which to change between delta-scaled L1 and L2 loss.
            The value must be positive.  Default: 1.0

    Shape:
        - Input: :math:`(*)` where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(*)`, same shape as the input.

    Examples:

        >>> loss = nn.HuberLoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    __constants__ = ["reduction", "delta"]

    def __init__(self, reduction: str = "mean", delta: float = 1.0) -> None:
        if not isinstance(delta, (float, int)):
            raise TypeError(f"delta must be a float or int, got: {type(delta)}")
        super().__init__(reduction=reduction)
        self.delta = delta

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Runs the forward pass."""
        return F.huber_loss(input, target, reduction=self.reduction, delta=self.delta)


class SoftMarginLoss(_Loss):
    r"""Creates a criterion that optimizes a two-class classification
    logistic loss between input tensor :math:`x` and target tensor :math:`y`
    (containing 1 or -1).

    .. math::
        \text{loss}(x, y) = \sum_i \frac{\log(1 + \exp(-y[i]*x[i]))}{\text{x.nelement}()}

    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(*)`, same
          shape as input.

    Examples:

        >>> loss = nn.SoftMarginLoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5).sign()
        >>> output = loss(input, target)
        >>> output.backward()
    """

    __constants__ = ["reduction"]

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Runs the forward pass."""
        return F.soft_margin_loss(input, target, reduction=self.reduction)


class CrossEntropyLoss(_WeightedLoss):
    r"""This criterion computes the cross entropy loss between input logits
    and target.

    It is useful when training a classification problem with `C` classes.
    If provided, the optional argument :attr:`weight` should be a 1D `Tensor`
    assigning weight to each of the classes.
    This is particularly useful when you have an unbalanced training set.

    The `input` is expected to contain the unnormalized logits for each class (which do `not` need
    to be positive or sum to 1, in general).
    `input` has to be a Tensor of size :math:`(C)` for unbatched input,
    :math:`(minibatch, C)` or :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1` for the
    `K`-dimensional case. The last being useful for higher dimension inputs, such
    as computing cross entropy loss per-pixel for 2D images.

    The `target` that this criterion expects should contain either:

    - Class indices in the range :math:`[0, C)` where :math:`C` is the number of classes; if
      `ignore_index` is specified, this loss also accepts this class index (this index
      may not necessarily be in the class range). The unreduced (i.e. with :attr:`reduction`
      set to ``'none'``) loss for this case can be described as:

      .. math::
          \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
          l_n = - w_{y_n} \log \frac{\exp(x_{n,y_n})}{\sum_{c=1}^C \exp(x_{n,c})}
          \cdot \mathbb{1}\{y_n \not= \text{ignore\_index}\}

      where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight,
      :math:`C` is the number of classes, and :math:`N` spans the minibatch dimension as well as
      :math:`d_1, ..., d_k` for the `K`-dimensional case. If
      :attr:`reduction` is not ``'none'`` (default ``'mean'``), then

      .. math::
          \ell(x, y) = \begin{cases}
              \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n} \cdot \mathbb{1}\{y_n \not= \text{ignore\_index}\}} l_n, &
               \text{if reduction} = \text{`mean';}\\
                \sum_{n=1}^N l_n,  &
                \text{if reduction} = \text{`sum'.}
            \end{cases}

      Note that this case is equivalent to applying :class:`~torch.nn.LogSoftmax`
      on an input, followed by :class:`~torch.nn.NLLLoss`.

    - Probabilities for each class; useful when labels beyond a single class per minibatch item
      are required, such as for blended labels, label smoothing, etc. The unreduced (i.e. with
      :attr:`reduction` set to ``'none'``) loss for this case can be described as:

      .. math::
          \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
          l_n = - \sum_{c=1}^C w_c \log \frac{\exp(x_{n,c})}{\sum_{i=1}^C \exp(x_{n,i})} y_{n,c}

      where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight,
      :math:`C` is the number of classes, and :math:`N` spans the minibatch dimension as well as
      :math:`d_1, ..., d_k` for the `K`-dimensional case. If
      :attr:`reduction` is not ``'none'`` (default ``'mean'``), then

      .. math::
          \ell(x, y) = \begin{cases}
              \frac{\sum_{n=1}^N l_n}{N}, &
               \text{if reduction} = \text{`mean';}\\
                \sum_{n=1}^N l_n,  &
                \text{if reduction} = \text{`sum'.}
            \end{cases}

    .. note::
        The performance of this criterion is generally better when `target` contains class
        indices, as this allows for optimized computation. Consider providing `target` as
        class probabilities only when a single class label per minibatch item is too restrictive.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each class.
            If given, has to be a Tensor of size `C`.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Note that
            :attr:`ignore_index` is only applicable when the target contains class indices.
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in
            the meantime, specifying either of those two args will override
            :attr:`reduction`. Default: ``'mean'``
        label_smoothing (float, optional): A float in [0.0, 1.0]. Specifies the amount
            of smoothing when computing the loss, where 0.0 means no smoothing. The targets
            become a mixture of the original ground truth and a uniform distribution as described in
            `Rethinking the Inception Architecture for Computer Vision <https://arxiv.org/abs/1512.00567>`__. Default: :math:`0.0`.

    Shape:
        - Input: Shape :math:`(C)`, :math:`(N, C)` or :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: If containing class indices, shape :math:`()`, :math:`(N)` or :math:`(N, d_1, d_2, ..., d_K)` with
          :math:`K \geq 1` in the case of K-dimensional loss where each value should be between :math:`[0, C)`. The
          target data type is required to be long when using class indices. If containing class probabilities, the
          target must be the same shape input, and each value should be between :math:`[0, 1]`. This means the target
          data type is required to be float when using class probabilities. Note that PyTorch does not strictly enforce
          probability constraints on the class probabilities and that it is the user's responsibility to ensure
          ``target`` contains valid probability distributions (see below examples section for more details).
        - Output: If reduction is 'none', shape :math:`()`, :math:`(N)` or :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of K-dimensional loss, depending on the shape of the input. Otherwise, scalar.


        where:

        .. math::
            \begin{aligned}
                C ={} & \text{number of classes} \\
                N ={} & \text{batch size} \\
            \end{aligned}

    Examples:

        >>> # Example of target with class indices
        >>> loss = nn.CrossEntropyLoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> output = loss(input, target)
        >>> output.backward()
        >>>
        >>> # Example of target with class probabilities
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5).softmax(dim=1)
        >>> output = loss(input, target)
        >>> output.backward()

    .. note::
        When ``target`` contains class probabilities, it should consist of soft labels—that is,
        each ``target`` entry should represent a probability distribution over the possible classes for a given data sample,
        with individual probabilities between ``[0,1]`` and the total distribution summing to 1.
        This is why the :func:`softmax()` function is applied to the ``target`` in the class probabilities example above.

        PyTorch does not validate whether the values provided in ``target`` lie in the range ``[0,1]``
        or whether the distribution of each data sample sums to ``1``.
        No warning will be raised and it is the user's responsibility
        to ensure that ``target`` contains valid probability distributions.
        Providing arbitrary values may yield misleading loss values and unstable gradients during training.

    Examples:
        >>> # xdoctest: +SKIP
        >>> # Example of target with incorrectly specified class probabilities
        >>> loss = nn.CrossEntropyLoss()
        >>> torch.manual_seed(283)
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> # Provided target class probabilities are not in range [0,1]
        >>> target
        tensor([[ 0.7105,  0.4446,  2.0297,  0.2671, -0.6075],
                [-1.0496, -0.2753, -0.3586,  0.9270,  1.0027],
                [ 0.7551,  0.1003,  1.3468, -0.3581, -0.9569]])
        >>> # Provided target class probabilities do not sum to 1
        >>> target.sum(axis=1)
        tensor([2.8444, 0.2462, 0.8873])
        >>> # No error message and possible misleading loss value
        >>> loss(input, target).item()
        4.6379876136779785
        >>>
        >>> # Example of target with correctly specified class probabilities
        >>> # Use .softmax() to ensure true probability distribution
        >>> target_new = target.softmax(dim=1)
        >>> # New target class probabilities all in range [0,1]
        >>> target_new
        tensor([[0.1559, 0.1195, 0.5830, 0.1000, 0.0417],
                [0.0496, 0.1075, 0.0990, 0.3579, 0.3860],
                [0.2607, 0.1355, 0.4711, 0.0856, 0.0471]])
        >>> # New target class probabilities sum to 1
        >>> target_new.sum(axis=1)
        tensor([1.0000, 1.0000, 1.0000])
        >>> loss(input, target_new).item()
        2.55349063873291
    """

    __constants__ = ["ignore_index", "reduction", "label_smoothing"]
    ignore_index: int
    label_smoothing: float

    def __init__(
        self,
        weight: Tensor | None = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Runs the forward pass."""
        return F.cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )


class LinearCrossEntropyLoss(_WeightedLoss):
    r"""This criterion computes the cross entropy loss between input,
    linearly transformed to logits, and target.

    See :class:`~torch.nn.CrossEntropyLoss` for the definition of cross entropy loss.

    Args:
        in_features (int): Size of each input sample.
        num_classes (int): Number of classes, :math:`C`.
        out_features (tuple[int], optional): specifies dimensions
            :math:`(d_1, d_2, ..., d_K)` for K-dimensional loss.
            Default: ``()``.
        bias (bool, optional): If ``True``, the internal :class:`Linear`
            adds a learnable bias to the logits. Logical shape is
            ``(C, *out_features)``; storage is flat
            (``self.linear.bias.shape == (C * prod(out_features),)``)
            for the same reason as ``self.linear.weight`` -- reshaping
            happens in :meth:`forward` before passing through to
            :func:`~torch.nn.functional.linear_cross_entropy` as
            ``linear_bias``. With ``options != None``, K-dimensional
            bias (``out_features != ()``) falls back to the reference
            implementation with a warning; the chunked path supports
            only ``(C,)``-shaped bias. Default: ``False``.
        device (:class:`torch.device`, optional): the desired device
            of linear weight.  Default: ``None``.
        dtype (:class:`torch.dtype`, optional): the desired dtype of
            linear weight. Default: ``None``.
        weight (Tensor, optional): a manual rescaling weight given to
            each class.  If given, has to be a Tensor of size `C`.
        reduction (str, optional): Specifies the reduction to apply to
            the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied,
            ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed.
            Default: ``'mean'``.
        ignore_index (int, optional): Specifies a target value that is
            ignored and does not contribute to the input
            gradient. Note that :attr:`ignore_index` is only
            applicable when the target contains class indices.
            Default: `None`. When target contains class indices, the
            default value is mapped to `-100`. Note: the default
            :attr:`ignore_index` in
            :class:`~torch.nn.CrossEntropyLoss` is `-100` for both
            target types.
        label_smoothing (float, optional): A float in [0.0, 1.0].
            Specifies the amount of smoothing when computing the loss,
            where 0.0 means no smoothing. The targets become a mixture
            of the original ground truth and a uniform distribution as
            described in `Rethinking the Inception Architecture for
            Computer Vision
            <https://arxiv.org/abs/1512.00567>`__.
            Default: :math:`0.0`.
        options (LinearCrossEntropyOptions, optional): Specify
            chunking strategy options, see
            :class:`~torch.nn.LinearCrossEntropyOptions`
            for more details. To enable reference implementation of
            linear_cross_entropy with chunking disabled, use
            `options=None`. Note: passing a non-``None`` ``options``
            makes the module incompatible with
            :func:`torch.jit.script`; see the note below.

    .. warning::
        With non-``None`` ``options``, the chunked path consumes its
        precomputed gradients in-place, so any second :meth:`backward`
        call raises (even with ``retain_graph=True``). Use
        ``LinearCrossEntropyOptions(allow_retain_graph=True)`` to allow
        repeated backward (one extra gradient-sized allocation per call).

    .. note::
        ``options=None`` is scriptable; an ``options`` instance is not.
        The chunked path does not support higher-order AD, forward-mode
        AD, ``torch.func.grad`` / ``vmap(grad(...))``, or Inductor
        lowering; see :func:`torch.nn.functional.linear_cross_entropy`.

    Shape:
        - Input: Shape :math:`(in_features)`, :math:`(N, in_features)`.
        - Target: If containing class indices, shape :math:`()`,
          :math:`(N)` or :math:`(N, *out_features)` where each value
          should be between :math:`[0, C)`. The target data type is
          required to be long when using class indices.
          If containing class probabilities, the target must have
          shape :math:`(C)` or :math:`(N, C, *out_features)`, and
          each value should be between :math:`[0, 1]`. This means the
          target data type is required to be float when using class
          probabilities. Note that PyTorch does not strictly enforce
          probability constraints on the class probabilities and that
          it is the user's responsibility to ensure ``target``
          contains valid probability distributions (see below examples
          section for more details).
        - Output: If reduction is 'none', shape :math:`()`,
          :math:`(N)` or :math:`(N, *out_features)` depending on the
          shape of the input. Otherwise, scalar.

        where :math:`N` is batch size.

    Examples:
        >>> torch.manual_seed(283)
        >>> # Example of target with class indices
        >>> loss = nn.LinearCrossEntropyLoss(5, 10, out_features=(4, 3))
        >>> input = torch.randn(2, 5, requires_grad=True)
        >>> target = torch.randint(0, 10, (2, 4, 3))
        >>> output = loss(input, target)
        >>> output.backward()
        >>>
        >>> # Example of target with class probabilities
        >>> input = torch.randn(2, 5, requires_grad=True)
        >>> target = torch.randn(2, 10, 4, 3).softmax(dim=1)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    # options is intentionally not in __constants__ (not scriptable).
    __constants__ = [
        "num_classes",
        "out_features",
        "reduction",
        "ignore_index",
        "label_smoothing",
    ]
    num_classes: int
    out_features: tuple[int, ...]
    reduction: str
    ignore_index: int | None
    label_smoothing: float
    options: LinearCrossEntropyOptions | None

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        *,
        out_features: tuple[int, ...] = (),
        bias: bool = False,
        device=None,
        dtype=None,
        reduction: str = "mean",
        weight: Tensor | None = None,
        ignore_index: int | None = None,
        label_smoothing: float = 0.0,
        options: LinearCrossEntropyOptions | None = None,
    ) -> None:
        if weight is not None and weight.shape != (num_classes,):
            raise RuntimeError(
                f"expected weight shape to be {(num_classes,)}, got {tuple(weight.shape)}"
            )
        if label_smoothing < 0 or label_smoothing > 1.0:
            raise RuntimeError(
                f"expected label_smoothing to be in range [0, 1], got {label_smoothing}"
            )
        super().__init__(
            weight=weight, size_average=None, reduce=None, reduction=reduction
        )
        self.num_classes = num_classes
        self.out_features = out_features
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        # Linear does not support multi-dimensional weights. To
        # circumvent this limitation, we store the linear weights as
        # two-dimensional weights and reshape it to multi-dimensional
        # weights in forward prior to passing the weights to
        # linear_cross_entropy.
        self.linear = Linear(
            in_features,
            math.prod(out_features, start=num_classes),
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.options = options

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Runs the forward pass."""
        linear_weight = self.linear.weight.reshape(
            (self.num_classes, *self.out_features, self.linear.in_features)
        )
        linear_bias = (
            self.linear.bias.reshape((self.num_classes, *self.out_features))
            if self.linear.bias is not None
            else None
        )
        return F.linear_cross_entropy(
            input,
            linear_weight,
            target,
            linear_bias=linear_bias,
            weight=self.weight,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            options=self.options,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.linear.in_features},"
            f" num_classes={self.num_classes},"
            f" out_features={self.out_features},"
            f" bias={self.linear.bias is not None},"
            f" reduction={self.reduction},"
            f" ignore_index={self.ignore_index},"
            f" label_smoothing={self.label_smoothing},"
            f" options={self.options}"
        )


class MultiLabelSoftMarginLoss(_WeightedLoss):
    r"""Creates a criterion that optimizes a multi-label one-versus-all
    loss based on max-entropy, between input :math:`x` and target :math:`y` of size
    :math:`(N, C)`.
    For each sample in the minibatch:

    .. math::
        loss(x, y) = - \frac{1}{C} * \sum_i y[i] * \log((1 + \exp(-x[i]))^{-1})
                         + (1-y[i]) * \log\left(\frac{\exp(-x[i])}{(1 + \exp(-x[i]))}\right)

    where :math:`i \in \left\{0, \; \cdots , \; \text{x.nElement}() - 1\right\}`,
    :math:`y[i] \in \left\{0, \; 1\right\}`.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`. Otherwise, it is
            treated as if having all ones.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, C)` where `N` is the batch size and `C` is the number of classes.
        - Target: :math:`(N, C)`, label targets must have the same shape as the input.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(N)`.

    Examples:

        >>> loss = nn.MultiLabelSoftMarginLoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, 5).random_(2)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    __constants__ = ["reduction"]

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Runs the forward pass."""
        return F.multilabel_soft_margin_loss(
            input, target, weight=self.weight, reduction=self.reduction
        )


class CosineEmbeddingLoss(_Loss):
    r"""Creates a criterion that measures the loss given input tensors
    :math:`x_1`, :math:`x_2` and a `Tensor` label :math:`y` with values 1 or -1.
    Use (:math:`y=1`) to maximize the cosine similarity of two inputs, and (:math:`y=-1`) otherwise.
    This is typically used for learning nonlinear
    embeddings or semi-supervised learning.

    The loss function for each sample is:

    .. math::
        \text{loss}(x, y) =
        \begin{cases}
        1 - \cos(x_1, x_2), & \text{if } y = 1 \\
        \max(0, \cos(x_1, x_2) - \text{margin}), & \text{if } y = -1
        \end{cases}

    Args:
        margin (float, optional): Should be a number from :math:`-1` to :math:`1`,
            :math:`0` to :math:`0.5` is suggested. If :attr:`margin` is missing, the
            default value is :math:`0`.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input1: :math:`(N, D)` or :math:`(D)`, where `N` is the batch size and `D` is the embedding dimension.
        - Input2: :math:`(N, D)` or :math:`(D)`, same shape as Input1.
        - Target: :math:`(N)` or :math:`()`.
        - Output: If :attr:`reduction` is ``'none'``, then :math:`(N)`, otherwise scalar.

    Examples:

        >>> loss = nn.CosineEmbeddingLoss()
        >>> input1 = torch.randn(3, 5, requires_grad=True)
        >>> input2 = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.ones(3)
        >>> output = loss(input1, input2, target)
        >>> output.backward()
    """

    __constants__ = ["margin", "reduction"]
    margin: float

    def __init__(
        self,
        margin: float = 0.0,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.margin = margin

    def forward(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor:
        """Runs the forward pass."""
        return F.cosine_embedding_loss(
            input1, input2, target, margin=self.margin, reduction=self.reduction
        )


class MarginRankingLoss(_Loss):
    r"""Creates a criterion that measures the loss given
    inputs :math:`x1`, :math:`x2`, two 1D mini-batch or 0D `Tensors`,
    and a label 1D mini-batch or 0D `Tensor` :math:`y` (containing 1 or -1).

    If :math:`y = 1` then it is assumed the first input should be ranked higher
    (have a larger value) than the second input, and vice-versa for :math:`y = -1`.

    The loss function for each pair of samples in the mini-batch is:

    .. math::
        \text{loss}(x1, x2, y) = \max(0, -y * (x1 - x2) + \text{margin})

    Args:
        margin (float, optional): Has a default value of :math:`0`.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input1: :math:`(N)` or :math:`()` where `N` is the batch size.
        - Input2: :math:`(N)` or :math:`()`, same shape as the Input1.
        - Target: :math:`(N)` or :math:`()`, same shape as the inputs.
        - Output: scalar. If :attr:`reduction` is ``'none'`` and Input size is not :math:`()`, then :math:`(N)`.

    Examples:

        >>> loss = nn.MarginRankingLoss()
        >>> input1 = torch.randn(3, requires_grad=True)
        >>> input2 = torch.randn(3, requires_grad=True)
        >>> target = torch.randn(3).sign()
        >>> output = loss(input1, input2, target)
        >>> output.backward()
    """

    __constants__ = ["margin", "reduction"]
    margin: float

    def __init__(
        self,
        margin: float = 0.0,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.margin = margin

    def forward(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor:
        """Runs the forward pass."""
        return F.margin_ranking_loss(
            input1, input2, target, margin=self.margin, reduction=self.reduction
        )


class MultiMarginLoss(_WeightedLoss):
    r"""Creates a criterion that optimizes a multi-class classification hinge
    loss (margin-based loss) between input :math:`x` (a 2D mini-batch `Tensor`) and
    output :math:`y` (which is a 1D tensor of target class indices,
    :math:`0 \leq y \leq \text{x.size}(1)-1`):

    For each mini-batch sample, the loss in terms of the 1D input :math:`x` and scalar
    output :math:`y` is:

    .. math::
        \text{loss}(x, y) = \frac{\sum_i \max(0, \text{margin} - x[y] + x[i])^p}{\text{x.size}(0)}

    where :math:`i \in \left\{0, \; \cdots , \; \text{x.size}(0) - 1\right\}`
    and :math:`i \neq y`.

    Optionally, you can give non-equal weighting on the classes by passing
    a 1D :attr:`weight` tensor into the constructor.

    The loss function then becomes:

    .. math::
        \text{loss}(x, y) = \frac{\sum_i w[y] * \max(0, \text{margin} - x[y] + x[i])^p}{\text{x.size}(0)}

    Args:
        p (int, optional): Has a default value of :math:`1`. :math:`1` and :math:`2`
            are the only supported values.
        margin (float, optional): Has a default value of :math:`1`.
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`. Otherwise, it is
            treated as if having all ones.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, C)` or :math:`(C)`, where :math:`N` is the batch size and :math:`C` is the number of classes.
        - Target: :math:`(N)` or :math:`()`, where each value is :math:`0 \leq \text{targets}[i] \leq C-1`.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then same shape as the target.

    Examples:

        >>> loss = nn.MultiMarginLoss()
        >>> x = torch.tensor([[0.1, 0.2, 0.4, 0.8]])
        >>> y = torch.tensor([3])
        >>> # 0.25 * ((1-(0.8-0.1)) + (1-(0.8-0.2)) + (1-(0.8-0.4)))
        >>> loss(x, y)
        tensor(0.32...)
    """

    __constants__ = ["p", "margin", "reduction"]
    margin: float
    p: int

    def __init__(
        self,
        p: int = 1,
        margin: float = 1.0,
        weight: Tensor | None = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(weight, size_average, reduce, reduction)
        if p != 1 and p != 2:
            raise ValueError("only p == 1 and p == 2 supported")
        if weight is not None and weight.dim() != 1:
            raise ValueError(
                f"MultiMarginLoss: expected weight to be None or 1D tensor, got {weight.dim()}D instead"
            )
        self.p = p
        self.margin = margin

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Runs the forward pass."""
        return F.multi_margin_loss(
            input,
            target,
            p=self.p,
            margin=self.margin,
            weight=self.weight,
            reduction=self.reduction,
        )


class TripletMarginLoss(_Loss):
    r"""Creates a criterion that measures the triplet loss given an input
    tensors :math:`x1`, :math:`x2`, :math:`x3` and a margin with a value greater than :math:`0`.
    This is used for measuring a relative similarity between samples. A triplet
    is composed by `a`, `p` and `n` (i.e., `anchor`, `positive examples` and `negative
    examples` respectively). The shapes of all input tensors should be
    :math:`(N, D)`.

    The distance swap is described in detail in the paper `Learning shallow
    convolutional feature descriptors with triplet losses`_ by
    V. Balntas, E. Riba et al.

    The loss function for each sample in the mini-batch is:

    .. math::
        L(a, p, n) = \max \{d(a_i, p_i) - d(a_i, n_i) + {\rm margin}, 0\}


    where

    .. math::
        d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_p

    The norm is calculated using the specified p value and a small constant :math:`\varepsilon` is
    added for numerical stability.

    See also :class:`~torch.nn.TripletMarginWithDistanceLoss`, which computes the
    triplet margin loss for input tensors using a custom distance function.

    Args:
        margin (float, optional): Default: :math:`1`.
        p (int, optional): The norm degree for pairwise distance. Default: :math:`2`.
        eps (float, optional): Small constant for numerical stability. Default: :math:`1e-6`.
        swap (bool, optional): The distance swap is described in detail in the paper
            `Learning shallow convolutional feature descriptors with triplet losses` by
            V. Balntas, E. Riba et al. Default: ``False``.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, D)` or :math:`(D)` where :math:`D` is the vector dimension.
        - Output: A Tensor of shape :math:`(N)` if :attr:`reduction` is ``'none'`` and
          input shape is :math:`(N, D)`; a scalar otherwise.

    Examples:

    >>> triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
    >>> anchor = torch.randn(100, 128, requires_grad=True)
    >>> positive = torch.randn(100, 128, requires_grad=True)
    >>> negative = torch.randn(100, 128, requires_grad=True)
    >>> output = triplet_loss(anchor, positive, negative)
    >>> output.backward()

    .. _Learning shallow convolutional feature descriptors with triplet losses:
        https://bmva-archive.org.uk/bmvc/2016/papers/paper119/index.html
    """

    __constants__ = ["margin", "p", "eps", "swap", "reduction"]
    margin: float
    p: float
    eps: float
    swap: bool

    def __init__(
        self,
        margin: float = 1.0,
        p: float = 2.0,
        eps: float = 1e-6,
        swap: bool = False,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        if margin <= 0:
            raise ValueError(
                f"TripletMarginLoss: expected margin to be greater than 0, got {margin} instead"
            )
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        """Runs the forward pass."""
        return F.triplet_margin_loss(
            anchor,
            positive,
            negative,
            margin=self.margin,
            p=self.p,
            eps=self.eps,
            swap=self.swap,
            reduction=self.reduction,
        )


class TripletMarginWithDistanceLoss(_Loss):
    r"""Creates a criterion that measures the triplet loss given input
    tensors :math:`a`, :math:`p`, and :math:`n` (representing anchor,
    positive, and negative examples, respectively), and a nonnegative,
    real-valued function ("distance function") used to compute the relationship
    between the anchor and positive example ("positive distance") and the
    anchor and negative example ("negative distance").

    The unreduced loss (i.e., with :attr:`reduction` set to ``'none'``)
    can be described as:

    .. math::
        \ell(a, p, n) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_i = \max \{d(a_i, p_i) - d(a_i, n_i) + {\rm margin}, 0\}

    where :math:`N` is the batch size; :math:`d` is a nonnegative, real-valued function
    quantifying the closeness of two tensors, referred to as the :attr:`distance_function`;
    and :math:`margin` is a nonnegative margin representing the minimum difference
    between the positive and negative distances that is required for the loss to
    be 0.  The input tensors have :math:`N` elements each and can be of any shape
    that the distance function can handle.

    If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), &  \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  &  \text{if reduction} = \text{`sum'.}
        \end{cases}

    See also :class:`~torch.nn.TripletMarginLoss`, which computes the triplet
    loss for input tensors using the :math:`l_p` distance as the distance function.

    Args:
        distance_function (Callable, optional): A nonnegative, real-valued function that
            quantifies the closeness of two tensors. If not specified,
            `nn.PairwiseDistance` will be used.  Default: ``None``
        margin (float, optional): A nonnegative margin representing the minimum difference
            between the positive and negative distances required for the loss to be 0. Larger
            margins penalize cases where the negative examples are not distant enough from the
            anchors, relative to the positives. Default: :math:`1`.
        swap (bool, optional): Whether to use the distance swap described in the paper
            `Learning shallow convolutional feature descriptors with triplet losses` by
            V. Balntas, E. Riba et al. If True, and if the positive example is closer to the
            negative example than the anchor is, swaps the positive example and the anchor in
            the loss computation. Default: ``False``.
        reduction (str, optional): Specifies the (optional) reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``


    Shape:
        - Input: :math:`(N, *)` where :math:`*` represents any number of additional dimensions
          as supported by the distance function.
        - Output: A Tensor of shape :math:`(N)` if :attr:`reduction` is ``'none'``, or a scalar
          otherwise.

    Examples:

    >>> # Initialize embeddings
    >>> embedding = nn.Embedding(1000, 128)
    >>> anchor_ids = torch.randint(0, 1000, (1,))
    >>> positive_ids = torch.randint(0, 1000, (1,))
    >>> negative_ids = torch.randint(0, 1000, (1,))
    >>> anchor = embedding(anchor_ids)
    >>> positive = embedding(positive_ids)
    >>> negative = embedding(negative_ids)
    >>>
    >>> # Built-in Distance Function
    >>> triplet_loss = \
    >>>     nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance())
    >>> output = triplet_loss(anchor, positive, negative)
    >>> output.backward()
    >>>
    >>> # Custom Distance Function
    >>> def l_infinity(x1, x2):
    >>>     return torch.max(torch.abs(x1 - x2), dim=1).values
    >>>
    >>> # xdoctest: +SKIP("FIXME: Would call backwards a second time")
    >>> triplet_loss = (
    >>>     nn.TripletMarginWithDistanceLoss(distance_function=l_infinity, margin=1.5))
    >>> output = triplet_loss(anchor, positive, negative)
    >>> output.backward()
    >>>
    >>> # Custom Distance Function (Lambda)
    >>> triplet_loss = (
    >>>     nn.TripletMarginWithDistanceLoss(
    >>>         distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y)))
    >>> output = triplet_loss(anchor, positive, negative)
    >>> output.backward()

    Reference:
        V. Balntas, et al.: Learning shallow convolutional feature descriptors with triplet losses:
        https://bmva-archive.org.uk/bmvc/2016/papers/paper119/index.html
    """

    __constants__ = ["margin", "swap", "reduction"]
    margin: float
    swap: bool

    def __init__(
        self,
        *,
        distance_function: Callable[[Tensor, Tensor], Tensor] | None = None,
        margin: float = 1.0,
        swap: bool = False,
        reduction: str = "mean",
    ) -> None:
        super().__init__(size_average=None, reduce=None, reduction=reduction)
        if margin <= 0:
            raise ValueError(
                f"TripletMarginWithDistanceLoss: expected margin to be greater than 0, got {margin} instead"
            )
        self.distance_function: Callable[[Tensor, Tensor], Tensor] | None = (
            distance_function if distance_function is not None else PairwiseDistance()
        )
        self.margin = margin
        self.swap = swap

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        """Runs the forward pass."""
        return F.triplet_margin_with_distance_loss(
            anchor,
            positive,
            negative,
            distance_function=self.distance_function,
            margin=self.margin,
            swap=self.swap,
            reduction=self.reduction,
        )


class CTCLoss(_Loss):
    r"""The Connectionist Temporal Classification loss.

    Calculates loss between a continuous (unsegmented) time series and a target sequence. CTCLoss sums over the
    probability of possible alignments of input to target, producing a loss value which is differentiable
    with respect to each input node. The alignment of input to target is assumed to be "many-to-one", which
    limits the length of the target sequence such that it must be :math:`\leq` the input length.

    Args:
        blank (int, optional): blank label. Default :math:`0`.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the output losses will be divided by the target lengths and
            then the mean over the batch is taken, ``'sum'``: the output losses will be summed.
            Default: ``'mean'``
        zero_infinity (bool, optional):
            Whether to zero infinite losses and the associated gradients.
            Default: ``False``
            Infinite losses mainly occur when the inputs are too short
            to be aligned to the targets.

    Shape:
        - Log_probs: Tensor of size :math:`(T, N, C)` or :math:`(T, C)`,
          where :math:`T = \text{input length}`,
          :math:`N = \text{batch size}`, and
          :math:`C = \text{number of classes (including blank)}`.
          The logarithmized probabilities of the outputs (e.g. obtained with
          :func:`torch.nn.functional.log_softmax`).
        - Targets: Tensor of size :math:`(N, S)` or
          :math:`(\operatorname{sum}(\text{target\_lengths}))`,
          where :math:`N = \text{batch size}` and
          :math:`S = \text{max target length, if shape is } (N, S)`.
          It represents the target sequences. Each element in the target
          sequence is a class index. And the target index cannot be blank (default=0).
          In the :math:`(N, S)` form, targets are padded to the
          length of the longest sequence, and stacked.
          In the :math:`(\operatorname{sum}(\text{target\_lengths}))` form,
          the targets are assumed to be un-padded and
          concatenated within 1 dimension.
        - Input_lengths: Tuple or tensor of size :math:`(N)` or :math:`()`,
          where :math:`N = \text{batch size}`. It represents the lengths of the
          inputs (must each be :math:`\leq T`). And the lengths are specified
          for each sequence to achieve masking under the assumption that sequences
          are padded to equal lengths.
        - Target_lengths: Tuple or tensor of size :math:`(N)` or :math:`()`,
          where :math:`N = \text{batch size}`. It represents lengths of the targets.
          Lengths are specified for each sequence to achieve masking under the
          assumption that sequences are padded to equal lengths. If target shape is
          :math:`(N,S)`, target_lengths are effectively the stop index
          :math:`s_n` for each target sequence, such that ``target_n = targets[n,0:s_n]`` for
          each target in a batch. Lengths must each be :math:`\leq S`
          If the targets are given as a 1d tensor that is the concatenation of individual
          targets, the target_lengths must add up to the total length of the tensor.
        - Output: scalar if :attr:`reduction` is ``'mean'`` (default) or
          ``'sum'``. If :attr:`reduction` is ``'none'``, then :math:`(N)` if input is batched or
          :math:`()` if input is unbatched, where :math:`N = \text{batch size}`.

    Examples:

        >>> # Target are to be padded
        >>> T = 50  # Input sequence length
        >>> C = 20  # Number of classes (including blank)
        >>> N = 16  # Batch size
        >>> S = 30  # Target sequence length of longest target in batch (padding length)
        >>> S_min = 10  # Minimum target length, for demonstration purposes
        >>>
        >>> # Initialize random batch of input vectors, for *size = (T,N,C)
        >>> input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
        >>>
        >>> # Initialize random batch of targets (0 = blank, 1:C = classes)
        >>> target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)
        >>>
        >>> input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
        >>> target_lengths = torch.randint(
        ...     low=S_min,
        ...     high=S,
        ...     size=(N,),
        ...     dtype=torch.long,
        ... )
        >>> ctc_loss = nn.CTCLoss()
        >>> loss = ctc_loss(input, target, input_lengths, target_lengths)
        >>> loss.backward()
        >>>
        >>>
        >>> # Target are to be un-padded
        >>> T = 50  # Input sequence length
        >>> C = 20  # Number of classes (including blank)
        >>> N = 16  # Batch size
        >>>
        >>> # Initialize random batch of input vectors, for *size = (T,N,C)
        >>> input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
        >>> input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
        >>>
        >>> # Initialize random batch of targets (0 = blank, 1:C = classes)
        >>> target_lengths = torch.randint(low=1, high=T, size=(N,), dtype=torch.long)
        >>> target = torch.randint(
        ...     low=1,
        ...     high=C,
        ...     size=(sum(target_lengths),),
        ...     dtype=torch.long,
        ... )
        >>> ctc_loss = nn.CTCLoss()
        >>> loss = ctc_loss(input, target, input_lengths, target_lengths)
        >>> loss.backward()
        >>>
        >>>
        >>> # Target are to be un-padded and unbatched (effectively N=1)
        >>> T = 50  # Input sequence length
        >>> C = 20  # Number of classes (including blank)
        >>>
        >>> # Initialize random batch of input vectors, for *size = (T,C)
        >>> # xdoctest: +SKIP("FIXME: error in doctest")
        >>> input = torch.randn(T, C).log_softmax(1).detach().requires_grad_()
        >>> input_lengths = torch.tensor(T, dtype=torch.long)
        >>>
        >>> # Initialize random batch of targets (0 = blank, 1:C = classes)
        >>> target_lengths = torch.randint(low=1, high=T, size=(), dtype=torch.long)
        >>> target = torch.randint(
        ...     low=1,
        ...     high=C,
        ...     size=(target_lengths,),
        ...     dtype=torch.long,
        ... )
        >>> ctc_loss = nn.CTCLoss()
        >>> loss = ctc_loss(input, target, input_lengths, target_lengths)
        >>> loss.backward()

    Reference:
        A. Graves et al.: Connectionist Temporal Classification:
        Labelling Unsegmented Sequence Data with Recurrent Neural Networks:
        https://www.cs.toronto.edu/~graves/icml_2006.pdf

    Note:
        In order to use CuDNN, the following must be satisfied: the :attr:`targets` must be
        in concatenated format, all :attr:`input_lengths` must be `T`.  :math:`blank=0`,
        :attr:`target_lengths` :math:`\leq 256`, the integer arguments must be of
        dtype :attr:`torch.int32`, and the :attr:`log_probs` itself must be of
        dtype :attr:`torch.float32`.

        The regular implementation uses the (more common in PyTorch) `torch.long` dtype.


    Note:
        In some circumstances when using the CUDA backend with CuDNN, this operator
        may select a nondeterministic algorithm to increase performance. If this is
        undesirable, you can try to make the operation deterministic (potentially at
        a performance cost) by setting ``torch.backends.cudnn.deterministic =
        True``.
        Please see the notes on :doc:`/notes/randomness` for background.
    """

    __constants__ = ["blank", "reduction"]
    blank: int
    zero_infinity: bool

    def __init__(
        self, blank: int = 0, reduction: str = "mean", zero_infinity: bool = False
    ) -> None:
        super().__init__(reduction=reduction)
        self.blank = blank
        self.zero_infinity = zero_infinity

    def forward(
        self,
        log_probs: Tensor,
        targets: Tensor,
        input_lengths: Tensor,
        target_lengths: Tensor,
    ) -> Tensor:
        """Runs the forward pass."""
        return F.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            self.blank,
            self.reduction,
            self.zero_infinity,
        )


# TODO: L1HingeEmbeddingCriterion
# TODO: MSECriterion weight
# TODO: ClassSimplexCriterion
<|endoftext|># mypy: allow-untyped-defs

from torch import Tensor
from torch.types import _size

from .module import Module


__all__ = ["Flatten", "Unflatten"]


class Flatten(Module):
    r"""
    Flattens a contiguous range of dims into a tensor.

    For use with :class:`~nn.Sequential`, see :meth:`torch.flatten` for details.

    Shape:
        - Input: :math:`(*, S_{\text{start}},..., S_{i}, ..., S_{\text{end}}, *)`,'
          where :math:`S_{i}` is the size at dimension :math:`i` and :math:`*` means any
          number of dimensions including none.
        - Output: :math:`(*, \prod_{i=\text{start}}^{\text{end}} S_{i}, *)`.

    Args:
        start_dim: first dim to flatten (default = 1).
        end_dim: last dim to flatten (default = -1).

    Examples::
        >>> input = torch.randn(32, 1, 5, 5)
        >>> # With default parameters
        >>> m = nn.Flatten()
        >>> output = m(input)
        >>> output.size()
        torch.Size([32, 25])
        >>> # With non-default parameters
        >>> m = nn.Flatten(0, 2)
        >>> output = m(input)
        >>> output.size()
        torch.Size([160, 5])
    """

    __constants__ = ["start_dim", "end_dim"]
    start_dim: int
    end_dim: int

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return input.flatten(self.start_dim, self.end_dim)

    def extra_repr(self) -> str:
        """
        Returns the extra representation of the module.
        """
        return f"start_dim={self.start_dim}, end_dim={self.end_dim}"


class Unflatten(Module):
    r"""
    Unflattens a tensor dim expanding it to a desired shape. For use with :class:`~nn.Sequential`.

    * :attr:`dim` specifies the dimension of the input tensor to be unflattened.

    * :attr:`unflattened_size` is the new shape of the unflattened dimension of the tensor and it can be
      a `tuple` of ints or a `list` of ints or `torch.Size` for `Tensor` input.

    Shape:
        - Input: :math:`(*, S_{\text{dim}}, *)`, where :math:`S_{\text{dim}}` is the size at
          dimension :attr:`dim` and :math:`*` means any number of dimensions including none.
        - Output: :math:`(*, U_1, ..., U_n, *)`, where :math:`U` = :attr:`unflattened_size` and
          :math:`\prod_{i=1}^n U_i = S_{\text{dim}}`.

    Args:
        dim (int): Dimension to be unflattened
        unflattened_size (Union[torch.Size, Tuple, List]): New shape of the unflattened dimension

    Examples:
        >>> input = torch.randn(2, 50)
        >>> # With tuple of ints
        >>> m = nn.Sequential(
        >>>     nn.Linear(50, 50),
        >>>     nn.Unflatten(1, (2, 5, 5))
        >>> )
        >>> output = m(input)
        >>> output.size()
        torch.Size([2, 2, 5, 5])
        >>> # With torch.Size
        >>> m = nn.Sequential(
        >>>     nn.Linear(50, 50),
        >>>     nn.Unflatten(1, torch.Size([2, 5, 5]))
        >>> )
        >>> output = m(input)
        >>> output.size()
        torch.Size([2, 2, 5, 5])
    """

    __constants__ = ["dim", "unflattened_size"]
    dim: int
    unflattened_size: _size

    def __init__(self, dim: int, unflattened_size: _size) -> None:
        super().__init__()
        self._require_tuple_int(unflattened_size)
        self.dim = dim
        self.unflattened_size = unflattened_size

    def _require_tuple_int(self, input) -> None:
        if isinstance(input, (tuple, list)):
            for idx, elem in enumerate(input):
                if not isinstance(elem, int):
                    raise TypeError(
                        "unflattened_size must be tuple of ints, "
                        + f"but found element of type {type(elem).__name__} at pos {idx}"
                    )
            return
        raise TypeError(
            f"unflattened_size must be a tuple of ints, but found type {type(input).__name__}"
        )

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return input.unflatten(self.dim, self.unflattened_size)

    def extra_repr(self) -> str:
        """
        Returns the extra representation of the module.
        """
        return f"dim={self.dim}, unflattened_size={self.unflattened_size}"
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import operator
from collections import abc as container_abcs, OrderedDict
from itertools import chain, islice
from typing import Any, overload, TYPE_CHECKING, TypeVar
from typing_extensions import deprecated, Self

import torch
from torch._jit_internal import _copy_to_script_wrapper
from torch.nn.parameter import Parameter

from .module import Module


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping


__all__ = [
    "Container",
    "Sequential",
    "ModuleList",
    "ModuleDict",
    "ParameterList",
    "ParameterDict",
]

T = TypeVar("T", bound=Module)
_V = TypeVar("_V")


# Copied from torch.nn.modules.module, required for a custom __repr__ for ModuleList
def _addindent(s_, numSpaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


@deprecated(
    "`nn.Container` is deprecated. "
    "All of its functionality is now implemented in `nn.Module`. Subclass that instead.",
    category=FutureWarning,
)
class Container(Module):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        for key, value in kwargs.items():
            self.add_module(key, value)


class Sequential(Module):
    r"""A sequential container.

    Modules will be added to it in the order they are passed in the
    constructor. Alternatively, an ``OrderedDict`` of modules can be
    passed in. The ``forward()`` method of ``Sequential`` accepts any
    input and forwards it to the first module it contains. It then
    "chains" outputs to inputs sequentially for each subsequent module,
    finally returning the output of the last module.

    The value a ``Sequential`` provides over manually calling a sequence
    of modules is that it allows treating the whole container as a
    single module, such that performing a transformation on the
    ``Sequential`` applies to each of the modules it stores (which are
    each a registered submodule of the ``Sequential``).

    What's the difference between a ``Sequential`` and a
    :class:`torch.nn.ModuleList`? A ``ModuleList`` is exactly what it
    sounds like--a list for storing ``Module`` s! On the other hand,
    the layers in a ``Sequential`` are connected in a cascading way.

    Example::

        # Using Sequential to create a small model. When `model` is run,
        # input will first be passed to `Conv2d(1,20,5)`. The output of
        # `Conv2d(1,20,5)` will be used as the input to the first
        # `ReLU`; the output of the first `ReLU` will become the input
        # for `Conv2d(20,64,5)`. Finally, the output of
        # `Conv2d(20,64,5)` will be used as input to the second `ReLU`
        model = nn.Sequential(
            nn.Conv2d(1, 20, 5), nn.ReLU(), nn.Conv2d(20, 64, 5), nn.ReLU()
        )

        # Using Sequential with OrderedDict. This is functionally the
        # same as the above code
        model = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(1, 20, 5)),
                    ("relu1", nn.ReLU()),
                    ("conv2", nn.Conv2d(20, 64, 5)),
                    ("relu2", nn.ReLU()),
                ]
            )
        )
    """

    _modules: dict[str, Module]  # type: ignore[assignment]

    @overload
    def __init__(self, *args: Module) -> None: ...

    @overload
    # pyrefly: ignore [inconsistent-overload]
    def __init__(self, arg: OrderedDict[str, Module]) -> None: ...

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator: Iterable[_V], idx: int) -> _V:
        """Get the idx-th item of the iterator."""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError(f"index {idx} is out of range")
        idx %= size
        return next(islice(iterator, idx, None))

    @overload
    def __getitem__(self, idx: slice) -> Sequential: ...

    @overload
    def __getitem__(self, idx: int) -> Module: ...

    @_copy_to_script_wrapper
    def __getitem__(self, idx: slice | int) -> Sequential | Module:
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx: int, module: Module) -> None:
        key: str = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx: slice | int) -> None:
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)
        # To preserve numbering
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(
            zip(str_indices, self._modules.values(), strict=True)
        )

    @_copy_to_script_wrapper
    def __len__(self) -> int:
        return len(self._modules)

    def __add__(self, other) -> Sequential:
        if isinstance(other, Sequential):
            ret = Sequential()
            for layer in self:
                ret.append(layer)
            for layer in other:
                ret.append(layer)
            return ret
        else:
            raise ValueError(
                "add operator supports only objects "
                f"of Sequential class, but {str(type(other))} is given."
            )

    def pop(self, key: int | slice) -> Module:
        """
        Pop ``key`` from self.
        """
        v = self[key]
        del self[key]
        return v

    def __iadd__(self, other) -> Self:
        if isinstance(other, Sequential):
            offset = len(self)
            for i, module in enumerate(other):
                self.add_module(str(i + offset), module)
            return self
        else:
            raise ValueError(
                "add operator supports only objects "
                f"of Sequential class, but {str(type(other))} is given."
            )

    def __mul__(self, other: int) -> Sequential:
        if not isinstance(other, int):
            raise TypeError(
                f"unsupported operand type(s) for *: {type(self)} and {type(other)}"
            )
        elif other <= 0:
            raise ValueError(
                f"Non-positive multiplication factor {other} for {type(self)}"
            )
        else:
            combined = Sequential()
            offset = 0
            for _ in range(other):
                for module in self:
                    combined.add_module(str(offset), module)
                    offset += 1
            return combined

    def __rmul__(self, other: int) -> Sequential:
        return self.__mul__(other)

    def __imul__(self, other: int) -> Self:
        if not isinstance(other, int):
            raise TypeError(
                f"unsupported operand type(s) for *: {type(self)} and {type(other)}"
            )
        elif other <= 0:
            raise ValueError(
                f"Non-positive multiplication factor {other} for {type(self)}"
            )
        else:
            len_original = len(self)
            offset = len(self)
            for _ in range(other - 1):
                for i in range(len_original):
                    self.add_module(str(i + offset), self._modules[str(i)])
                offset += len_original
            return self

    @_copy_to_script_wrapper
    def __dir__(self) -> list[str]:
        keys = super().__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    @_copy_to_script_wrapper
    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    # NB: We can't really type check this function as the type of input
    # may change dynamically (as is tested in
    # TestScript.test_sequential_intermediary_types).  Cannot annotate
    # with Any as TorchScript expects a more precise type
    def forward(self, input):
        """
        Runs the forward pass.
        """
        for module in self:
            input = module(input)
        return input

    def append(self, module: Module) -> Self:
        r"""Append a given module to the end.

        Args:
            module (nn.Module): module to append

        Example::

            >>> import torch.nn as nn
            >>> n = nn.Sequential(nn.Linear(1, 2), nn.Linear(2, 3))
            >>> n.append(nn.Linear(3, 4))
            Sequential(
                (0): Linear(in_features=1, out_features=2, bias=True)
                (1): Linear(in_features=2, out_features=3, bias=True)
                (2): Linear(in_features=3, out_features=4, bias=True)
            )

        """
        self.add_module(str(len(self)), module)
        return self

    def insert(self, index: int, module: Module) -> Self:
        """
        Inserts a module into the Sequential container at the specified index.

        Args:
            index (int): The index to insert the module.
            module (Module): The module to be inserted.

        Example::

            >>> import torch.nn as nn
            >>> n = nn.Sequential(nn.Linear(1, 2), nn.Linear(2, 3))
            >>> n.insert(0, nn.Linear(3, 4))
            Sequential(
                (0): Linear(in_features=3, out_features=4, bias=True)
                (1): Linear(in_features=1, out_features=2, bias=True)
                (2): Linear(in_features=2, out_features=3, bias=True)
            )

        """
        if not isinstance(module, Module):
            raise AssertionError(f"module should be of type: {Module}")
        n = len(self._modules)
        if not (-n <= index <= n):
            raise IndexError(f"Index out of range: {index}")
        if index < 0:
            index += n
        for i in range(n, index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module
        return self

    def extend(self, sequential: Iterable[Module]) -> Self:
        """
        Extends the current Sequential container with layers from another Sequential container.

        Args:
            sequential (Sequential): A Sequential container whose layers will be added to the current container.

        Example::

            >>> import torch.nn as nn
            >>> n = nn.Sequential(nn.Linear(1, 2), nn.Linear(2, 3))
            >>> other = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 5))
            >>> n.extend(other) # or `n + other`
            Sequential(
                (0): Linear(in_features=1, out_features=2, bias=True)
                (1): Linear(in_features=2, out_features=3, bias=True)
                (2): Linear(in_features=3, out_features=4, bias=True)
                (3): Linear(in_features=4, out_features=5, bias=True)
            )

        """
        for layer in sequential:
            self.append(layer)
        return self


class ModuleList(Module):
    r"""Holds submodules in a list.

    :class:`~torch.nn.ModuleList` can be indexed like a regular Python list, but
    modules it contains are properly registered, and will be visible by all
    :class:`~torch.nn.Module` methods.

    Args:
        modules (iterable, optional): an iterable of modules to add

    Example::

        class MyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

            def forward(self, x):
                # ModuleList can act as an iterable, or be indexed using ints
                for i, l in enumerate(self.linears):
                    x = self.linears[i // 2](x) + l(x)
                return x
    """

    _modules: dict[str, Module]  # type: ignore[assignment]

    def __init__(self, modules: Iterable[Module] | None = None) -> None:
        super().__init__()
        if modules is not None:
            self += modules

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of modules."""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError(f"index {idx} is out of range")
        if idx < 0:
            idx += len(self)
        return str(idx)

    @overload
    def __getitem__(self, idx: slice) -> ModuleList: ...

    @overload
    def __getitem__(self, idx: int) -> Module: ...

    @_copy_to_script_wrapper
    def __getitem__(self, idx: int | slice) -> Module | ModuleList:
        if isinstance(idx, slice):
            return self.__class__(list(self._modules.values())[idx])
        else:
            return self._modules[self._get_abs_string_index(idx)]

    def __setitem__(self, idx: int, module: Module) -> None:
        idx = self._get_abs_string_index(idx)
        return setattr(self, str(idx), module)

    def __delitem__(self, idx: int | slice) -> None:
        if isinstance(idx, slice):
            for k in range(len(self._modules))[idx]:
                delattr(self, str(k))
        else:
            delattr(self, self._get_abs_string_index(idx))
        # To preserve numbering, self._modules is being reconstructed with modules after deletion
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(
            zip(str_indices, self._modules.values(), strict=True)
        )

    @_copy_to_script_wrapper
    def __len__(self) -> int:
        return len(self._modules)

    @_copy_to_script_wrapper
    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def __iadd__(self, modules: Iterable[Module]) -> Self:
        return self.extend(modules)

    def __add__(self, other: Iterable[Module]) -> ModuleList:
        combined = ModuleList()
        for i, module in enumerate(chain(self, other)):
            combined.add_module(str(i), module)
        return combined

    def __repr__(self) -> str:
        """Return a custom repr for ModuleList that compresses repeated module representations."""
        list_of_reprs = [repr(item) for item in self]
        if len(list_of_reprs) == 0:
            return self._get_name() + "()"

        start_end_indices = [[0, 0]]
        repeated_blocks = [list_of_reprs[0]]
        for i, r in enumerate(list_of_reprs[1:], 1):
            if r == repeated_blocks[-1]:
                start_end_indices[-1][1] += 1
                continue

            start_end_indices.append([i, i])
            repeated_blocks.append(r)

        lines = []
        main_str = self._get_name() + "("
        for (start_id, end_id), b in zip(
            start_end_indices, repeated_blocks, strict=True
        ):
            local_repr = f"({start_id}): {b}"  # default repr

            if start_id != end_id:
                n = end_id - start_id + 1
                local_repr = f"({start_id}-{end_id}): {n} x {b}"

            local_repr = _addindent(local_repr, 2)
            lines.append(local_repr)

        main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str

    @_copy_to_script_wrapper
    def __dir__(self) -> list[str]:
        keys = super().__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def insert(self, index: int, module: Module) -> None:
        r"""Insert a given module before a given index in the list.

        Args:
            index (int): index to insert.
            module (nn.Module): module to insert
        """
        for i in range(len(self._modules), index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module

    def append(self, module: Module) -> Self:
        r"""Append a given module to the end of the list.

        Args:
            module (nn.Module): module to append
        """
        self.add_module(str(len(self)), module)
        return self

    def pop(self, key: int | slice) -> Module:
        v = self[key]
        del self[key]
        return v

    def extend(self, modules: Iterable[Module]) -> Self:
        r"""Append modules from a Python iterable to the end of the list.

        Args:
            modules (iterable): iterable of modules to append
        """
        if not isinstance(modules, container_abcs.Iterable):
            raise TypeError(
                "ModuleList.extend should be called with an "
                "iterable, but got " + type(modules).__name__
            )
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self

    # remove forward altogether to fallback on Module's _forward_unimplemented


class ModuleDict(Module):
    r"""Holds submodules in a dictionary.

    :class:`~torch.nn.ModuleDict` can be indexed like a regular Python dictionary,
    but modules it contains are properly registered, and will be visible by all
    :class:`~torch.nn.Module` methods.

    :class:`~torch.nn.ModuleDict` is an **ordered** dictionary that respects

    * the order of insertion, and

    * in :meth:`~torch.nn.ModuleDict.update`, the order of the merged
      ``OrderedDict``, ``dict`` or another
      :class:`~torch.nn.ModuleDict` (the argument to
      :meth:`~torch.nn.ModuleDict.update`).

    Note that :meth:`~torch.nn.ModuleDict.update` with other unordered mapping
    types does not preserve the order of the merged mapping.

    Args:
        modules (iterable, optional): a mapping (dictionary) of (string: module)
            or an iterable of key-value pairs of type (string, module)

    Example::

        class MyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.choices = nn.ModuleDict(
                    {"conv": nn.Conv2d(10, 10, 3), "pool": nn.MaxPool2d(3)}
                )
                self.activations = nn.ModuleDict(
                    [["lrelu", nn.LeakyReLU()], ["prelu", nn.PReLU()]]
                )

            def forward(self, x, choice, act):
                x = self.choices[choice](x)
                x = self.activations[act](x)
                return x
    """

    _modules: dict[str, Module]  # type: ignore[assignment]

    def __init__(self, modules: Mapping[str, Module] | None = None) -> None:
        super().__init__()
        if modules is not None:
            self.update(modules)

    @_copy_to_script_wrapper
    def __getitem__(self, key: str) -> Module:
        return self._modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        self.add_module(key, module)

    def __delitem__(self, key: str) -> None:
        del self._modules[key]

    @_copy_to_script_wrapper
    def __len__(self) -> int:
        return len(self._modules)

    @_copy_to_script_wrapper
    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    @_copy_to_script_wrapper
    def __contains__(self, key: str) -> bool:
        return key in self._modules

    def clear(self) -> None:
        """Remove all items from the ModuleDict."""
        self._modules.clear()

    def pop(self, key: str) -> Module:
        r"""Remove key from the ModuleDict and return its module.

        Args:
            key (str): key to pop from the ModuleDict
        """
        v = self[key]
        del self[key]
        return v

    @_copy_to_script_wrapper
    def keys(self) -> container_abcs.KeysView[str]:
        r"""Return an iterable of the ModuleDict keys."""
        return self._modules.keys()

    @_copy_to_script_wrapper
    def items(self) -> container_abcs.ItemsView[str, Module]:
        r"""Return an iterable of the ModuleDict key/value pairs."""
        return self._modules.items()

    @_copy_to_script_wrapper
    def values(self) -> container_abcs.ValuesView[Module]:
        r"""Return an iterable of the ModuleDict values."""
        return self._modules.values()

    def update(self, modules: Mapping[str, Module]) -> None:
        r"""Update the :class:`~torch.nn.ModuleDict` with key-value pairs from a mapping, overwriting existing keys.

        .. note::
            If :attr:`modules` is an ``OrderedDict``, a :class:`~torch.nn.ModuleDict`, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Args:
            modules (iterable): a mapping (dictionary) from string to :class:`~torch.nn.Module`,
                or an iterable of key-value pairs of type (string, :class:`~torch.nn.Module`)
        """
        if not isinstance(modules, container_abcs.Iterable):
            raise TypeError(
                "ModuleDict.update should be called with an "
                "iterable of key/value pairs, but got " + type(modules).__name__
            )

        if isinstance(modules, (OrderedDict, ModuleDict, container_abcs.Mapping)):
            for key, module in modules.items():
                self[key] = module
        else:
            # modules here can be a list with two items
            for j, m in enumerate(modules):
                if not isinstance(m, container_abcs.Iterable):
                    raise TypeError(
                        "ModuleDict update sequence element "
                        "#" + str(j) + " should be Iterable; is" + type(m).__name__
                    )
                # pyrefly: ignore [bad-argument-type]
                if not len(m) == 2:
                    raise ValueError(
                        "ModuleDict update sequence element "
                        # pyrefly: ignore [bad-argument-type]
                        "#" + str(j) + " has length " + str(len(m)) + "; 2 is required"
                    )
                # modules can be Mapping (what it's typed at), or a list: [(name1, module1), (name2, module2)]
                # that's too cumbersome to type correctly with overloads, so we add an ignore here
                self[m[0]] = m[1]  # type: ignore[assignment]

    # remove forward altogether to fallback on Module's _forward_unimplemented


class ParameterList(Module):
    r"""Holds parameters in a list.

    :class:`~torch.nn.ParameterList` can be used like a regular Python
    list, but Tensors that are :class:`~torch.nn.Parameter` are properly registered,
    and will be visible by all :class:`~torch.nn.Module` methods.

    Note that the constructor, assigning an element of the list, the
    :meth:`~torch.nn.ParameterList.append` method and the :meth:`~torch.nn.ParameterList.extend`
    method will convert any :class:`~torch.Tensor` into :class:`~torch.nn.Parameter`.

    Args:
        parameters (iterable, optional): an iterable of elements to add to the list.

    Example::

        class MyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.params = nn.ParameterList(
                    [nn.Parameter(torch.randn(10, 10)) for i in range(10)]
                )

            def forward(self, x):
                # ParameterList can act as an iterable, or be indexed using ints
                for i, p in enumerate(self.params):
                    x = self.params[i // 2].mm(x) + p.mm(x)
                return x
    """

    def __init__(self, values: Iterable[Any] | None = None) -> None:
        super().__init__()
        self._size = 0
        if values is not None:
            self += values

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of modules."""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError(f"index {idx} is out of range")
        if idx < 0:
            idx += len(self)
        return str(idx)

    @overload
    def __getitem__(self, idx: int) -> Any: ...

    @overload
    # pyrefly: ignore [inconsistent-overload]
    def __getitem__(self: T, idx: slice) -> T: ...

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            out = self.__class__()
            for i in range(start, stop, step):
                out.append(self[i])
            return out
        else:
            idx = self._get_abs_string_index(idx)
            return getattr(self, str(idx))

    def __setitem__(self, idx: int, param: Any) -> None:
        # Note that all other function that add an entry to the list part of
        # the ParameterList end up here. So this is the only place where we need
        # to wrap things into Parameter if needed.
        # Objects added via setattr() are not in the list part and thus won't
        # call into this function.
        idx = self._get_abs_string_index(idx)
        if isinstance(param, torch.Tensor) and not isinstance(param, Parameter):
            param = Parameter(param)
        return setattr(self, str(idx), param)

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[Any]:
        return iter(self[i] for i in range(len(self)))

    def __iadd__(self, parameters: Iterable[Any]) -> Self:
        return self.extend(parameters)

    def __dir__(self) -> list[str]:
        keys = super().__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def append(self, value: Any) -> Self:
        """Append a given value at the end of the list.

        Args:
            value (Any): value to append
        """
        new_idx = len(self)
        self._size += 1
        self[new_idx] = value
        return self

    def extend(self, values: Iterable[Any]) -> Self:
        """Append values from a Python iterable to the end of the list.

        Args:
            values (iterable): iterable of values to append
        """
        # Tensor is an iterable but we never want to unpack it here
        if not isinstance(values, container_abcs.Iterable) or isinstance(
            values, torch.Tensor
        ):
            raise TypeError(
                "ParameterList.extend should be called with an "
                "iterable, but got " + type(values).__name__
            )
        for value in values:
            self.append(value)
        return self

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        child_lines = []
        for k, p in enumerate(self):
            if isinstance(p, torch.Tensor):
                size_str = "x".join(str(size) for size in p.size())
                if p.device.type in ["cuda", torch._C._get_privateuse1_backend_name()]:
                    device_str = f" ({p.device})"
                else:
                    device_str = ""
                parastr = "{} containing: [{} of size {}{}]".format(
                    "Parameter" if isinstance(p, Parameter) else "Tensor",
                    p.dtype,
                    size_str,
                    device_str,
                )
                child_lines.append("  (" + str(k) + "): " + parastr)
            else:
                child_lines.append(
                    "  (" + str(k) + "): Object of type: " + type(p).__name__
                )

        tmpstr = "\n".join(child_lines)
        return tmpstr

    def __call__(self, *args, **kwargs):
        raise RuntimeError("ParameterList should not be called.")


class ParameterDict(Module):
    r"""Holds parameters in a dictionary.

    ParameterDict can be indexed like a regular Python dictionary, but Parameters it
    contains are properly registered, and will be visible by all Module methods.
    Other objects are treated as would be done by a regular Python dictionary

    :class:`~torch.nn.ParameterDict` is an **ordered** dictionary.
    :meth:`~torch.nn.ParameterDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict``) does not preserve the order of the
    merged mapping. On the other hand, ``OrderedDict`` or another :class:`~torch.nn.ParameterDict`
    will preserve their ordering.

    Note that the constructor, assigning an element of the dictionary and the
    :meth:`~torch.nn.ParameterDict.update` method will convert any :class:`~torch.Tensor` into
    :class:`~torch.nn.Parameter`.

    Args:
        values (iterable, optional): a mapping (dictionary) of
            (string : Any) or an iterable of key-value pairs
            of type (string, Any)

    Example::

        class MyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.params = nn.ParameterDict(
                    {
                        "left": nn.Parameter(torch.randn(5, 10)),
                        "right": nn.Parameter(torch.randn(5, 10)),
                    }
                )

            def forward(self, x, choice):
                x = self.params[choice].mm(x)
                return x
    """

    def __init__(self, parameters: Any = None) -> None:
        super().__init__()
        self._keys: dict[str, None] = {}
        if parameters is not None:
            self.update(parameters)

    def _key_to_attr(self, key: str) -> str:
        if not isinstance(key, str):
            raise TypeError(
                "Index given to ParameterDict cannot be used as a key as it is "
                f"not a string (type is '{type(key).__name__}'). Open an issue on "
                "github if you need non-string keys."
            )
        else:
            # Use the key as-is so that `.named_parameters()` returns the right thing
            return key

    def __getitem__(self, key: str) -> Any:
        attr = self._key_to_attr(key)
        return getattr(self, attr)

    def __setitem__(self, key: str, value: Any) -> None:
        # Note that all other function that add an entry to the dictionary part of
        # the ParameterDict end up here. So this is the only place where we need
        # to wrap things into Parameter if needed.
        # Objects added via setattr() are not in the dictionary part and thus won't
        # call into this function.
        self._keys[key] = None
        attr = self._key_to_attr(key)
        if isinstance(value, torch.Tensor) and not isinstance(value, Parameter):
            value = Parameter(value)
        setattr(self, attr, value)

    def __delitem__(self, key: str) -> None:
        del self._keys[key]
        attr = self._key_to_attr(key)
        delattr(self, attr)

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __reversed__(self) -> Iterator[str]:
        return reversed(self._keys)

    def copy(self) -> ParameterDict:
        """Return a copy of this :class:`~torch.nn.ParameterDict` instance."""
        # We have to use an OrderedDict because the ParameterDict constructor
        # behaves differently on plain dict vs OrderedDict
        return ParameterDict(OrderedDict((k, self[k]) for k in self._keys))

    def __contains__(self, key: str) -> bool:
        return key in self._keys

    def setdefault(self, key: str, default: Any | None = None) -> Any:
        """Set the default for a key in the Parameterdict.

        If key is in the ParameterDict, return its value.
        If not, insert `key` with a parameter `default` and return `default`.
        `default` defaults to `None`.

        Args:
            key (str): key to set default for
            default (Any): the parameter set to the key
        """
        if key not in self:
            self[key] = default
        return self[key]

    def clear(self) -> None:
        """Remove all items from the ParameterDict."""
        for k in self._keys.copy():
            del self[k]

    def pop(self, key: str) -> Any:
        r"""Remove key from the ParameterDict and return its parameter.

        Args:
            key (str): key to pop from the ParameterDict
        """
        v = self[key]
        del self[key]
        return v

    def popitem(self) -> tuple[str, Any]:
        """Remove and return the last inserted `(key, parameter)` pair from the ParameterDict."""
        k, _ = self._keys.popitem()
        # We need the key in the _keys to be able to access/del
        self._keys[k] = None
        val = self[k]
        del self[k]
        return k, val

    def get(self, key: str, default: Any | None = None) -> Any:
        r"""Return the parameter associated with key if present. Otherwise return default if provided, None if not.

        Args:
            key (str): key to get from the ParameterDict
            default (Parameter, optional): value to return if key not present
        """
        return self[key] if key in self else default  # noqa: SIM401

    def fromkeys(
        self, keys: Iterable[str], default: Any | None = None
    ) -> ParameterDict:
        r"""Return a new ParameterDict with the keys provided.

        Args:
            keys (iterable, string): keys to make the new ParameterDict from
            default (Parameter, optional): value to set for all keys
        """
        return ParameterDict((k, default) for k in keys)

    def keys(self) -> container_abcs.KeysView[str]:
        r"""Return an iterable of the ParameterDict keys."""
        return self._keys.keys()

    def items(self) -> Iterable[tuple[str, Any]]:
        r"""Return an iterable of the ParameterDict key/value pairs."""
        return ((k, self[k]) for k in self._keys)

    def values(self) -> Iterable[Any]:
        r"""Return an iterable of the ParameterDict values."""
        return (self[k] for k in self._keys)

    def update(self, parameters: Mapping[str, Any] | ParameterDict) -> None:
        r"""Update the :class:`~torch.nn.ParameterDict` with key-value pairs from ``parameters``, overwriting existing keys.

        .. note::
            If :attr:`parameters` is an ``OrderedDict``, a :class:`~torch.nn.ParameterDict`, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Args:
            parameters (iterable): a mapping (dictionary) from string to
                :class:`~torch.nn.Parameter`, or an iterable of
                key-value pairs of type (string, :class:`~torch.nn.Parameter`)
        """
